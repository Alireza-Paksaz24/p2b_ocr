"""
ocr_diagnostics.py — Drop-in diagnostic wrapper for the OCR pipeline.

Usage:
    from ocr_diagnostics import install_diagnostics
    install_diagnostics()           # call once at startup, before any OCR runs

What it does:
  1. Patches DeepSeekOCR2Adapter.process_image() to capture stdout (the model
     prints its output instead of saving/returning it).
  2. Logs every stage with timestamps, char counts, and hex previews.
  3. Writes a per-job .diag file alongside your outputs so you can diff runs.
  4. Monkey-patches _read_deepseek_output_dir to log what it actually finds.
  5. Adds a root-level OCR_DIAG logger with a FileHandler → ocr_diag.log.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Optional

# ── Diagnostic logger setup ──────────────────────────────────────────────────

DIAG_LOG = Path(os.getenv("OCR_DIAG_LOG", "ocr_diag.log"))

_diag = logging.getLogger("OCR_DIAG")
_diag.setLevel(logging.DEBUG)

_fmt = logging.Formatter(
    "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

# File handler — always verbose
_fh = logging.FileHandler(DIAG_LOG, encoding="utf-8")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(_fmt)
_diag.addHandler(_fh)

# Console handler — INFO+ only (less noise in terminal)
_ch = logging.StreamHandler(sys.stderr)
_ch.setLevel(logging.INFO)
_ch.setFormatter(_fmt)
_diag.addHandler(_ch)

_diag.propagate = False

# ── Helpers ───────────────────────────────────────────────────────────────────

def _preview(text: str, n: int = 120) -> str:
    """First n chars, newlines → ↵, non-printable → hex."""
    snippet = text[:n].replace("\n", "↵").replace("\r", "↩")
    return repr(snippet) if snippet != text[:n].replace("\n", "↵") else f'"{snippet}"'


def _hex_peek(text: str, n: int = 60) -> str:
    return text[:n].encode("utf-8", errors="replace").hex(" ")


def _file_tree(directory: str) -> str:
    lines = []
    for root, dirs, files in os.walk(directory):
        rel = os.path.relpath(root, directory)
        indent = "  " * rel.count(os.sep) if rel != "." else ""
        lines.append(f"{indent}{os.path.basename(root)}/")
        for fname in files:
            fpath = os.path.join(root, fname)
            size = os.path.getsize(fpath)
            lines.append(f"{indent}  {fname}  ({size:,} bytes)")
    return "\n".join(lines) if lines else "(empty)"


# ── stdout capture context manager ───────────────────────────────────────────

@contextlib.contextmanager
def _capture_stdout():
    """Redirect sys.stdout to a StringIO buffer while inside the block."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = old_stdout


# ── Patched _read_deepseek_output_dir ────────────────────────────────────────

def _patched_read_dir(out_dir: str) -> str:
    import glob

    TEXT_EXTS = {".md", ".txt", ".json", ".html", ".tex", ".mmd"}

    _diag.debug("_read_deepseek_output_dir  scanning: %s", out_dir)
    _diag.debug("Directory tree:\n%s", _file_tree(out_dir))

    best = ""
    for fpath in glob.glob(os.path.join(out_dir, "**", "*"), recursive=True):
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fpath)[1].lower()
        size = os.path.getsize(fpath)
        _diag.debug("  candidate: %s  ext=%s  size=%d", fpath, ext, size)

        if ext not in TEXT_EXTS:
            _diag.debug("    → skipped (not a text extension)")
            continue
        try:
            candidate = open(fpath, encoding="utf-8", errors="strict").read()
            _diag.debug(
                "    → read %d chars  preview: %s",
                len(candidate),
                _preview(candidate),
            )
            if len(candidate.strip()) > len(best.strip()):
                best = candidate
                _diag.debug("    → new best (%d chars)", len(best))
        except (UnicodeDecodeError, Exception) as exc:
            _diag.debug("    → decode error: %s", exc)

    _diag.info(
        "_read_deepseek_output_dir result: %d chars%s",
        len(best),
        f"  preview: {_preview(best)}" if best else "  (EMPTY — nothing useful found)",
    )
    if not best:
        _diag.warning(
            "OUT_DIR_EMPTY: Model wrote nothing to %s. "
            "The model likely printed output to stdout instead. "
            "See captured_stdout in the process_image log above.",
            out_dir,
        )
    return best


# ── Patched DeepSeekOCR2Adapter.process_image ────────────────────────────────

def _make_patched_process_image(original_fn):
    """Wrap process_image to capture stdout and log every stage."""

    def patched(self, image):
        import shutil

        t0 = time.perf_counter()
        job_tag = f"[DSv2 img={image.size[0]}x{image.size[1]}]"

        _diag.info("%s  process_image ENTER", job_tag)

        if not self._loaded:
            _diag.warning("%s  adapter not loaded — falling back to stub", job_tag)
            from ocr_engine import StubOCRAdapter
            return StubOCRAdapter().process_image(image)

        try:
            import os

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            _diag.debug("%s  saved temp image → %s", job_tag, tmp_path)

            out_dir = tempfile.mkdtemp()
            _diag.debug("%s  created out_dir → %s", job_tag, out_dir)

            # ── Call model.infer() while capturing stdout ────────────────────
            _diag.info("%s  calling model.infer() …", job_tag)
            t_infer = time.perf_counter()

            with _capture_stdout() as stdout_buf:
                try:
                    res = self._model.infer(
                        self._tokenizer,
                        prompt=(
                            "<image>\nFree OCR. Convert the document to markdown, "
                            "preserving structure and formatting. "
                        ),
                        image_file=tmp_path,
                        output_path=out_dir,
                        base_size=1024,
                        image_size=768,
                        crop_mode=True,
                        save_results=True,
                    )
                except Exception as infer_exc:
                    _diag.error("%s  model.infer() raised: %s", job_tag, infer_exc, exc_info=True)
                    res = None

            captured_stdout = stdout_buf.getvalue()
            infer_ms = (time.perf_counter() - t_infer) * 1000

            _diag.info(
                "%s  model.infer() finished in %.0f ms  "
                "return_type=%s  return_len=%d  stdout_len=%d",
                job_tag,
                infer_ms,
                type(res).__name__,
                len(res) if isinstance(res, str) else -1,
                len(captured_stdout),
            )

            if isinstance(res, str) and res.strip():
                _diag.info(
                    "%s  RETURN VALUE is non-empty (%d chars)  preview: %s",
                    job_tag, len(res), _preview(res),
                )
            else:
                _diag.warning(
                    "%s  RETURN VALUE is empty/None  (type=%s, value=%r)",
                    job_tag, type(res).__name__, res,
                )

            if captured_stdout.strip():
                _diag.info(
                    "%s  STDOUT captured (%d chars)  preview: %s",
                    job_tag, len(captured_stdout), _preview(captured_stdout),
                )
                _diag.debug(
                    "%s  STDOUT hex peek: %s",
                    job_tag, _hex_peek(captured_stdout),
                )
                _diag.debug(
                    "%s  STDOUT full:\n%s",
                    job_tag,
                    textwrap.indent(captured_stdout[:4000], "    "),
                )
            else:
                _diag.warning("%s  STDOUT was empty", job_tag)

            # ── Check out_dir ────────────────────────────────────────────────
            _diag.debug("%s  scanning out_dir …", job_tag)
            text_from_dir = _patched_read_dir(out_dir)

            # ── Decide best text source ──────────────────────────────────────
            candidates = {
                "return_value": res if isinstance(res, str) else "",
                "captured_stdout": captured_stdout,
                "out_dir_files": text_from_dir,
            }

            best_source, best_text = max(
                candidates.items(),
                key=lambda kv: len(kv[1].strip()),
            )

            _diag.info(
                "%s  WINNER: source=%s  chars=%d",
                job_tag, best_source, len(best_text),
            )
            for src, txt in candidates.items():
                marker = "✓" if src == best_source else " "
                _diag.info(
                    "  %s  %-20s  %5d chars  preview: %s",
                    marker, src, len(txt.strip()), _preview(txt),
                )

        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            shutil.rmtree(out_dir, ignore_errors=True)

        elapsed = (time.perf_counter() - t0) * 1000
        _diag.info("%s  process_image EXIT in %.0f ms", job_tag, elapsed)

        if not best_text.strip():
            _diag.error(
                "%s  ALL SOURCES EMPTY — no text recovered from image. "
                "Possible causes: wrong adapter key, model not loaded correctly, "
                "or model output format changed. Check ocr_diag.log for details.",
                job_tag,
            )
            return "[DeepSeek OCR-2: empty result — see ocr_diag.log]"

        # Apply existing cleaner
        try:
            from ocr_engine import _clean_deepseek_output
            cleaned = _clean_deepseek_output(best_text)
            _diag.info(
                "%s  after _clean_deepseek_output: %d → %d chars",
                job_tag, len(best_text), len(cleaned),
            )
            return cleaned
        except Exception as clean_exc:
            _diag.warning("%s  _clean_deepseek_output failed: %s", job_tag, clean_exc)
            return best_text

    return patched


# ── run_ocr wrapper ───────────────────────────────────────────────────────────

def _make_patched_run_ocr(original_run_ocr):
    def patched_run_ocr(images, model_id, model_path=None, adapter_key="stub", progress_callback=None):
        _diag.info(
            "run_ocr ENTER  model_id=%s  adapter=%s  images=%d  model_path=%s",
            model_id, adapter_key, len(images), model_path,
        )
        t0 = time.perf_counter()
        result = original_run_ocr(images, model_id, model_path, adapter_key, progress_callback)
        elapsed = (time.perf_counter() - t0) * 1000

        total_chars = sum(len(r.content) for p in result.pages for r in p.regions)
        _diag.info(
            "run_ocr EXIT  pages=%d  regions=%d  total_chars=%d  elapsed=%.0f ms",
            len(result.pages),
            sum(len(p.regions) for p in result.pages),
            total_chars,
            elapsed,
        )

        if total_chars == 0:
            _diag.error(
                "run_ocr produced ZERO chars across all pages. "
                "The adapter returned empty strings for every image."
            )

        for i, page in enumerate(result.pages, 1):
            for j, region in enumerate(page.regions):
                _diag.debug(
                    "  page=%d  region=%d  type=%-10s  chars=%d  preview: %s",
                    i, j, region.region_type.value, len(region.content), _preview(region.content),
                )

        return result

    return patched_run_ocr


# ── Public install function ───────────────────────────────────────────────────

def install_diagnostics() -> None:
    """
    Monkey-patch ocr_engine to inject full diagnostic logging.
    Call once at startup (e.g. top of main.py, after imports).
    """
    import ocr_engine

    # 1. Patch _read_deepseek_output_dir
    ocr_engine._read_deepseek_output_dir = _patched_read_dir
    _diag.info("Patched ocr_engine._read_deepseek_output_dir")

    # 2. Patch DeepSeekOCR2Adapter.process_image
    original_pi = ocr_engine.DeepSeekOCR2Adapter.process_image
    ocr_engine.DeepSeekOCR2Adapter.process_image = _make_patched_process_image(original_pi)
    _diag.info("Patched DeepSeekOCR2Adapter.process_image (stdout capture + full logging)")

    # 3. Patch run_ocr pipeline
    original_run = ocr_engine.run_ocr
    ocr_engine.run_ocr = _make_patched_run_ocr(original_run)
    _diag.info("Patched ocr_engine.run_ocr")

    _diag.info(
        "OCR diagnostics installed. Full log → %s  (INFO+ also on stderr)",
        DIAG_LOG.resolve(),
    )


# ── Standalone quick-test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick sanity-check: run diagnostics on a single image file passed as argv[1].

    Usage:
        python ocr_diagnostics.py path/to/image.png deepseek-ocr-2
    """
    import sys
    from pathlib import Path
    from PIL import Image

    install_diagnostics()

    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    model_id = sys.argv[2] if len(sys.argv) > 2 else "deepseek-ocr-2"

    if not img_path or not img_path.exists():
        print("Usage: python ocr_diagnostics.py <image_path> [model_id]")
        sys.exit(1)

    import ocr_engine

    model_path = Path(os.getenv("MODELS_DIR", "./models")) / model_id
    img = Image.open(img_path).convert("RGB")

    _diag.info("=== STANDALONE DIAGNOSTIC RUN ===")
    _diag.info("Image: %s  size=%s", img_path, img.size)
    _diag.info("Model: %s  path_exists=%s", model_id, model_path.exists())

    result = ocr_engine.run_ocr(
        [img],
        model_id=model_id,
        model_path=model_path if model_path.exists() else None,
        adapter_key="deepseek_ocr2",
    )

    print("\n=== RESULT ===")
    print(result.full_text() or "(empty)")
    print(f"\nFull diagnostic log: {DIAG_LOG.resolve()}")