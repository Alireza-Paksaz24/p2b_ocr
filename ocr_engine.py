"""
ocr_engine.py — OCR adapters with correct model APIs, GPU support, and content classification.

RTX 3090 (24 GB VRAM) can run all models here in bfloat16.
Every adapter loads with:
  - torch.bfloat16
  - flash_attention_2 where supported (massively reduces VRAM; falls back to eager automatically)
  - device_map="cuda" or .cuda()
"""

from __future__ import annotations

import io
import logging
import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger("ocr_engine")

def _clean_deepseek_output(text: str) -> str:
    """
    Strip DeepSeek OCR grounding tags and bounding boxes from model output.
    Raw output looks like:
      <|ref|>text<|/ref|><|det|>[[60, 65, 959, 228]]<|/det|>
      actual text content here
    We keep only the actual text lines, preserving markdown structure.
    """
    import re
    # Remove <|ref|>...<|/ref|><|det|>...<|/det|> tag pairs (including multiline)
    text = re.sub(r'<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>', '', text, flags=re.DOTALL)
    # Remove any remaining special tokens like <|grounding|>, <|/grounding|>, etc.
    text = re.sub(r'<\|[^|]+\|>', '', text)
    # Collapse more than 2 consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _best_attn() -> str:
    """Return flash_attention_2 if available, else eager. Safe on all platforms."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "eager"




# ── Region types ─────────────────────────────────────────────────────────────

class RegionType(str, Enum):
    TEXT    = "text"
    TABLE   = "table"
    DIAGRAM = "diagram"
    IMAGE   = "image"
    FORMULA = "formula"
    HEADER  = "header"
    FOOTER  = "footer"
    CAPTION = "caption"


@dataclass
class ContentRegion:
    region_type: RegionType
    content: str
    confidence: float = 1.0
    page: int = 1
    bbox: Optional[tuple] = None


@dataclass
class PageResult:
    page_number: int
    regions: list[ContentRegion] = field(default_factory=list)
    raw_text: str = ""


@dataclass
class OCRResult:
    pages: list[PageResult] = field(default_factory=list)
    model_id: str = ""
    content_summary: dict = field(default_factory=dict)

    def full_text(self) -> str:
        return "\n\n".join(r.content for p in self.pages for r in p.regions)

    def to_markdown(self) -> str:
        lines = []
        for p in self.pages:
            if len(self.pages) > 1:
                lines.append(f"\n---\n## Page {p.page_number}\n")
            for r in p.regions:
                if r.region_type == RegionType.HEADER:
                    lines.append(f"# {r.content}\n")
                elif r.region_type == RegionType.TABLE:
                    lines.append(r.content + "\n")
                elif r.region_type == RegionType.FORMULA:
                    lines.append(f"$$\n{r.content}\n$$\n")
                elif r.region_type == RegionType.DIAGRAM:
                    lines.append(f"> 📊 **[Diagram]** {r.content}\n")
                elif r.region_type == RegionType.IMAGE:
                    lines.append(f"> 🖼️ **[Image]** {r.content}\n")
                elif r.region_type == RegionType.CAPTION:
                    lines.append(f"*{r.content}*\n")
                else:
                    lines.append(r.content + "\n")
        return "\n".join(lines)


# ── Content classifier ────────────────────────────────────────────────────────

_TABLE_PATTERNS = [
    re.compile(r"(\|.*\|.*\n){2,}"),
    re.compile(r"(\t[^\t]+){3,}"),
]
_FORMULA_PATTERNS = [
    re.compile(r"[=∑∫∂∇∈∉⊂⊃≤≥≠±×÷√π∞]"),
    re.compile(r"\b(sin|cos|tan|log|ln|lim|sum|int)\b"),
    re.compile(r"[a-zA-Z]\^[{0-9]"),
    re.compile(r"\$\$.+\$\$", re.DOTALL),
]
_DIAGRAM_HINTS = ["figure", "fig.", "diagram", "chart", "graph", "plot", "flowchart", "schematic"]
_HEADER_PATTERN = re.compile(r"^[A-Z][^a-z\n]{0,80}$", re.MULTILINE)


def classify_text_block(text: str) -> RegionType:
    stripped = text.strip()
    if not stripped:
        return RegionType.TEXT
    lower = stripped.lower()
    for pat in _TABLE_PATTERNS:
        if pat.search(stripped):
            return RegionType.TABLE
    for pat in _FORMULA_PATTERNS:
        if pat.search(stripped):
            return RegionType.FORMULA
    for hint in _DIAGRAM_HINTS:
        if hint in lower:
            return RegionType.DIAGRAM
    if _HEADER_PATTERN.match(stripped) and len(stripped) < 100:
        return RegionType.HEADER
    return RegionType.TEXT


def classify_regions(raw_text: str, page: int = 1) -> list[ContentRegion]:
    regions = []
    for block in re.split(r"\n{2,}", raw_text.strip()):
        block = block.strip()
        if block:
            regions.append(ContentRegion(region_type=classify_text_block(block), content=block, page=page))
    return regions


# ── Base adapter ──────────────────────────────────────────────────────────────

class BaseOCRAdapter:
    model_id: str = "base"
    _loaded: bool = False

    def load(self, model_path: Path) -> None:
        raise NotImplementedError

    def process_image(self, image: Image.Image) -> str:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        return self._loaded


# ── Stub / Tesseract fallback ─────────────────────────────────────────────────

class StubOCRAdapter(BaseOCRAdapter):
    model_id = "stub"

    def load(self, model_path: Path) -> None:
        self._loaded = True

    def is_loaded(self) -> bool:
        return True

    def process_image(self, image: Image.Image) -> str:
        try:
            import pytesseract
            return pytesseract.image_to_string(image)
        except Exception:
            pass
        return (
            "[Model not loaded — download the selected model to enable OCR]\n"
            f"Image size: {image.size[0]}x{image.size[1]}"
        )


# ── DeepSeek OCR (v1) ─────────────────────────────────────────────────────────
# Official API: model.infer(tokenizer, prompt, image_file, output_path, ...)
# Supports Tiny/Small/Base/Large configs via base_size/image_size params.
# RTX 3090: use base_size=1024, image_size=640, crop_mode=True

class DeepSeekOCRAdapter(BaseOCRAdapter):
    model_id = "deepseek-ocr"
    _model = None
    _tokenizer = None

    def load(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            logger.info("Loading DeepSeek OCR from %s", model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            # DeepSeek's custom model code bypasses _best_attn() and hard-requires
            # flash_attn when flash_attention_2 is set. Always use eager instead.
            self._model = AutoModel.from_pretrained(
                str(model_path),
                _attn_implementation="eager",
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
            ).eval().cuda()
            self._loaded = True
            logger.info("DeepSeek OCR loaded on CUDA (eager attn)")
        except Exception as e:
            logger.error("DeepSeek OCR load failed: %s", e)

    def process_image(self, image: Image.Image) -> str:
        if not self._loaded:
            return StubOCRAdapter().process_image(image)
        try:
            import tempfile, os, glob
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            out_dir = tempfile.mkdtemp()
            try:
                res = self._model.infer(
                    self._tokenizer,
                    prompt="<image>\nFree OCR. Convert the document to markdown, preserving structure and formatting. ",
                    image_file=tmp_path,
                    output_path=out_dir,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=True,  # write to disk so we can read it back
                )
                # .infer() returns None and writes the result to a .md or .txt file
                text = ""
                if isinstance(res, str) and res.strip():
                    text = res
                else:
                    # Search for any output file written by infer()
                    for ext in ("*.md", "*.txt", "*.json"):
                        files = glob.glob(os.path.join(out_dir, "**", ext), recursive=True)
                        if files:
                            with open(files[0], encoding="utf-8", errors="replace") as f:
                                text = f.read()
                            break
            finally:
                os.unlink(tmp_path)
                import shutil
                shutil.rmtree(out_dir, ignore_errors=True)
            return _clean_deepseek_output(text) if text.strip() else "[DeepSeek OCR: empty result]"
        except Exception as e:
            logger.error("DeepSeek OCR inference error: %s", e)
            return f"[DeepSeek OCR error: {e}]"


# ── DeepSeek OCR 2 ────────────────────────────────────────────────────────────
# Same .infer() API as v1 but image_size=768, crop_mode=True per official docs.

class DeepSeekOCR2Adapter(BaseOCRAdapter):
    model_id = "deepseek-ocr-2"
    _model = None
    _tokenizer = None

    def load(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            logger.info("Loading DeepSeek OCR-2 from %s", model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            # DeepSeek's custom model code bypasses _best_attn() and hard-requires
            # flash_attn when flash_attention_2 is set. Always use eager instead.
            self._model = AutoModel.from_pretrained(
                str(model_path),
                _attn_implementation="eager",
                trust_remote_code=True,
                use_safetensors=True,
                torch_dtype=torch.bfloat16,
            ).eval().cuda()
            self._loaded = True
            logger.info("DeepSeek OCR-2 loaded on CUDA (eager attn)")
        except Exception as e:
            logger.error("DeepSeek OCR-2 load failed: %s", e)

    def process_image(self, image: Image.Image) -> str:
        if not self._loaded:
            return StubOCRAdapter().process_image(image)
        try:
            import tempfile, os, glob, shutil
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            out_dir = tempfile.mkdtemp()
            try:
                res = self._model.infer(
                    self._tokenizer,
                    prompt="<image>\nFree OCR. Convert the document to markdown, preserving structure and formatting. ",
                    image_file=tmp_path,
                    output_path=out_dir,
                    base_size=1024,
                    image_size=768,
                    crop_mode=True,
                    save_results=True,  # write to disk so we can read it back
                )
                # .infer() returns None and writes the result to a .md or .txt file
                text = ""
                if isinstance(res, str) and res.strip():
                    text = res
                else:
                    for ext in ("*.md", "*.txt", "*.json"):
                        files = glob.glob(os.path.join(out_dir, "**", ext), recursive=True)
                        if files:
                            with open(files[0], encoding="utf-8", errors="replace") as f:
                                text = f.read()
                            break
            finally:
                os.unlink(tmp_path)
                shutil.rmtree(out_dir, ignore_errors=True)
            return _clean_deepseek_output(text) if text.strip() else "[DeepSeek OCR-2: empty result]"
        except Exception as e:
            logger.error("DeepSeek OCR-2 inference error: %s", e)
            return f"[DeepSeek OCR-2 error: {e}]"


# ── PaddleOCR-VL 1.5 ─────────────────────────────────────────────────────────
# CRITICAL: must use flash_attention_2, otherwise VRAM explodes to 40+ GB.
# Weights: 1.92 GB; runtime VRAM with flash_attn: ~3.3 GB on RTX 3090.

class PaddleOCRVLAdapter(BaseOCRAdapter):
    model_id = "paddleocr-vl"
    _model = None
    _processor = None

    def load(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            logger.info("Loading PaddleOCR-VL-1.5 from %s", model_path)
            self._processor = AutoProcessor.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                attn_implementation=_best_attn(),  # flash_attention_2 when available, eager fallback
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )
            self._model.eval()
            self._loaded = True
            logger.info("PaddleOCR-VL-1.5 loaded on CUDA (attn=%s)", _best_attn())
        except Exception as e:
            logger.error("PaddleOCR-VL load failed: %s", e)

    def process_image(self, image: Image.Image) -> str:
        if not self._loaded:
            return StubOCRAdapter().process_image(image)
        try:
            import torch
            # PaddleOCR-VL supports task-specific prompts
            prompt = "Convert the document to markdown format, preserving tables, formulas, and structure."
            inputs = self._processor(
                text=prompt, images=image, return_tensors="pt"
            ).to("cuda", torch.bfloat16)
            with torch.inference_mode():
                ids = self._model.generate(**inputs, max_new_tokens=4096, temperature=0.0)
            output = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
            # Strip the prompt echo if present
            if prompt in output:
                output = output.split(prompt)[-1].strip()
            return output
        except Exception as e:
            logger.error("PaddleOCR-VL inference error: %s", e)
            return f"[PaddleOCR-VL error: {e}]"


# ── GOT-OCR 2.0 ───────────────────────────────────────────────────────────────
# 580M params, ~1.2 GB. Uses model.chat() API from transformers.
# Supports: plain OCR, formatted OCR (markdown), math formulas, tables.

class GotOCR2Adapter(BaseOCRAdapter):
    model_id = "got-ocr-2"
    _model = None
    _tokenizer = None

    def load(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            logger.info("Loading GOT-OCR-2 from %s", model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="cuda",
                use_safetensors=True,
                pad_token_id=self._tokenizer.eos_token_id,
            ).eval()
            self._loaded = True
            logger.info("GOT-OCR-2 loaded on CUDA")
        except Exception as e:
            logger.error("GOT-OCR-2 load failed: %s", e)

    def process_image(self, image: Image.Image) -> str:
        if not self._loaded:
            return StubOCRAdapter().process_image(image)
        try:
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            # ocr_type="format" → markdown output with tables/formulas
            result = self._model.chat(
                self._tokenizer,
                tmp_path,
                ocr_type="format",
            )
            os.unlink(tmp_path)
            return result or ""
        except Exception as e:
            logger.error("GOT-OCR-2 inference error: %s", e)
            return f"[GOT-OCR-2 error: {e}]"


# ── Florence-2 Large ──────────────────────────────────────────────────────────

class Florence2Adapter(BaseOCRAdapter):
    model_id = "florence-2-large"
    _model = None
    _processor = None

    def load(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            logger.info("Loading Florence-2-large from %s", model_path)
            self._processor = AutoProcessor.from_pretrained(
                str(model_path), trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            ).eval()
            self._loaded = True
            logger.info("Florence-2-large loaded on CUDA")
        except Exception as e:
            logger.error("Florence-2 load failed: %s", e)

    def process_image(self, image: Image.Image) -> str:
        if not self._loaded:
            return StubOCRAdapter().process_image(image)
        try:
            import torch
            task = "<OCR_WITH_REGION>"
            inputs = self._processor(text=task, images=image, return_tensors="pt").to("cuda", torch.bfloat16)
            with torch.inference_mode():
                ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=2048,
                    num_beams=3,
                )
            text = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = self._processor.post_process_generation(text, task=task, image_size=image.size)
            return parsed.get(task, text)
        except Exception as e:
            logger.error("Florence-2 inference error: %s", e)
            return f"[Florence-2 error: {e}]"


# ── Nougat Large ──────────────────────────────────────────────────────────────
# Academic paper OCR — produces LaTeX/markdown from scientific documents.

class NougatAdapter(BaseOCRAdapter):
    model_id = "nougat-large"
    _model = None
    _processor = None

    def load(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import NougatProcessor, VisionEncoderDecoderModel
            logger.info("Loading Nougat-large from %s", model_path)
            self._processor = NougatProcessor.from_pretrained(str(model_path))
            self._model = VisionEncoderDecoderModel.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            ).eval()
            self._loaded = True
            logger.info("Nougat-large loaded on CUDA")
        except Exception as e:
            logger.error("Nougat load failed: %s", e)

    def process_image(self, image: Image.Image) -> str:
        if not self._loaded:
            return StubOCRAdapter().process_image(image)
        try:
            import torch
            pixel_values = self._processor(image, return_tensors="pt").pixel_values.to("cuda", torch.bfloat16)
            with torch.inference_mode():
                ids = self._model.generate(
                    pixel_values,
                    min_length=1,
                    max_new_tokens=3584,
                    bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
                )
            text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
            return self._processor.post_process_generation(text, fix_markdown=False)
        except Exception as e:
            logger.error("Nougat inference error: %s", e)
            return f"[Nougat error: {e}]"


# ── Surya OCR ─────────────────────────────────────────────────────────────────
# Uses the `surya` pip package (not raw HF transformers).
# pip install surya-ocr

class SuryaOCRAdapter(BaseOCRAdapter):
    model_id = "surya-ocr"
    _rec_model = None
    _rec_processor = None
    _det_model = None
    _det_processor = None

    def load(self, model_path: Path) -> None:
        try:
            from surya.model.recognition.model import load_model as load_rec
            from surya.model.recognition.processor import load_processor as load_rec_proc
            from surya.model.detection.model import load_model as load_det
            from surya.model.detection.processor import load_processor as load_det_proc
            logger.info("Loading Surya OCR models")
            self._rec_model = load_rec()
            self._rec_processor = load_rec_proc()
            self._det_model = load_det()
            self._det_processor = load_det_proc()
            self._loaded = True
            logger.info("Surya OCR loaded")
        except ImportError:
            logger.error("surya not installed. Run: pip install surya-ocr")
        except Exception as e:
            logger.error("Surya load failed: %s", e)

    def process_image(self, image: Image.Image) -> str:
        if not self._loaded:
            return StubOCRAdapter().process_image(image)
        try:
            from surya.ocr import run_ocr as surya_run
            langs = [["en"]]
            results = surya_run(
                [image], langs,
                self._det_model, self._det_processor,
                self._rec_model, self._rec_processor,
            )
            lines = [line.text for line in results[0].text_lines]
            return "\n".join(lines)
        except Exception as e:
            logger.error("Surya inference error: %s", e)
            return f"[Surya error: {e}]"


# ── KOSMOS-2 ──────────────────────────────────────────────────────────────────

class Kosmos2Adapter(BaseOCRAdapter):
    model_id = "kosmos-2"
    _model = None
    _processor = None

    def load(self, model_path: Path) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
            logger.info("Loading KOSMOS-2 from %s", model_path)
            self._processor = AutoProcessor.from_pretrained(str(model_path))
            self._model = Kosmos2ForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            ).eval()
            self._loaded = True
            logger.info("KOSMOS-2 loaded on CUDA")
        except Exception as e:
            logger.error("KOSMOS-2 load failed: %s", e)

    def process_image(self, image: Image.Image) -> str:
        if not self._loaded:
            return StubOCRAdapter().process_image(image)
        try:
            import torch
            prompt = "<grounding>An image of"
            inputs = self._processor(text=prompt, images=image, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                ids = self._model.generate(
                    pixel_values=inputs["pixel_values"],
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs["image_embeds_position_mask"],
                    use_cache=True,
                    max_new_tokens=512,
                )
            text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
            text, _ = self._processor.post_process_generation(text)
            return text
        except Exception as e:
            logger.error("KOSMOS-2 inference error: %s", e)
            return f"[KOSMOS-2 error: {e}]"


# ── Registry ──────────────────────────────────────────────────────────────────

_ADAPTER_MAP: dict[str, type[BaseOCRAdapter]] = {
    "deepseek_ocr":  DeepSeekOCRAdapter,
    "deepseek_ocr2": DeepSeekOCR2Adapter,
    "paddleocr_vl":  PaddleOCRVLAdapter,
    "got_ocr2":      GotOCR2Adapter,
    "florence2":     Florence2Adapter,
    "nougat":        NougatAdapter,
    "surya":         SuryaOCRAdapter,
    "kosmos2":       Kosmos2Adapter,
    "stub":          StubOCRAdapter,
}

# Cache loaded adapters so we don't reload weights on every job
_loaded_adapters: dict[str, BaseOCRAdapter] = {}


def get_adapter(adapter_key: str, model_path: Optional[Path] = None) -> BaseOCRAdapter:
    """Return a loaded adapter, using cache to avoid reloading."""
    if adapter_key in _loaded_adapters and _loaded_adapters[adapter_key].is_loaded():
        return _loaded_adapters[adapter_key]

    cls = _ADAPTER_MAP.get(adapter_key, StubOCRAdapter)
    adapter = cls()

    if model_path and model_path.exists():
        adapter.load(model_path)
    elif not isinstance(adapter, StubOCRAdapter):
        logger.warning("Model path not found for adapter '%s', using stub", adapter_key)
        adapter = StubOCRAdapter()
        adapter.load(Path("."))

    _loaded_adapters[adapter_key] = adapter
    return adapter


# ── Pipeline ──────────────────────────────────────────────────────────────────

def build_content_summary(pages: list[PageResult]) -> dict:
    counts: dict[str, int] = {rt.value: 0 for rt in RegionType}
    for p in pages:
        for r in p.regions:
            counts[r.region_type.value] += 1
    return {k: v for k, v in counts.items() if v > 0}


def run_ocr(
    images: list[Image.Image],
    model_id: str,
    model_path: Optional[Path] = None,
    adapter_key: str = "stub",
    progress_callback=None,
) -> OCRResult:
    adapter = get_adapter(adapter_key, model_path)
    result = OCRResult(model_id=model_id)
    total = len(images)

    for idx, img in enumerate(images, start=1):
        raw = adapter.process_image(img)
        regions = classify_regions(raw, page=idx)
        result.pages.append(PageResult(page_number=idx, regions=regions, raw_text=raw))
        if progress_callback:
            progress_callback(idx / total)

    result.content_summary = build_content_summary(result.pages)
    return result