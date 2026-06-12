"""
ingestor.py — Convert uploaded files into a list of PIL Images for the OCR pipeline.

Supported input types:
  - Single image  (JPEG, PNG, TIFF, BMP, WEBP)
  - Multiple images (uploaded as individual files)
  - ZIP archive containing images
  - PDF (each page → image via pdf2image / fallback PIL)
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path

from PIL import Image

logger = logging.getLogger("ingestor")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".gif"}


def load_images_from_file(path: Path) -> list[Image.Image]:
    """
    Load one or more PIL Images from a single uploaded file.

    Returns an ordered list of PIL Images (one per page / image).
    """
    suffix = path.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return [_open_image(path)]

    if suffix == ".zip":
        return _from_zip(path)

    if suffix == ".pdf":
        return _from_pdf(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_images_from_bytes(data: bytes, filename: str) -> list[Image.Image]:
    """Load from raw bytes (e.g. from an HTTP upload body)."""
    suffix = Path(filename).suffix.lower()
    tmp = io.BytesIO(data)

    if suffix in IMAGE_EXTENSIONS:
        return [Image.open(tmp).convert("RGB")]

    if suffix == ".zip":
        return _from_zip_bytes(data)

    if suffix == ".pdf":
        return _from_pdf_bytes(data)

    raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _open_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _from_zip(path: Path) -> list[Image.Image]:
    images: list[Image.Image] = []
    with zipfile.ZipFile(path, "r") as zf:
        names = sorted(
            [n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_EXTENSIONS]
        )
        for name in names:
            data = zf.read(name)
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning("Could not open image %s from zip: %s", name, e)
    return images


def _from_zip_bytes(data: bytes) -> list[Image.Image]:
    images: list[Image.Image] = []
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        names = sorted(
            [n for n in zf.namelist() if Path(n).suffix.lower() in IMAGE_EXTENSIONS]
        )
        for name in names:
            raw = zf.read(name)
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                images.append(img)
            except Exception as e:
                logger.warning("Could not open image %s from zip: %s", name, e)
    return images


def _from_pdf(path: Path) -> list[Image.Image]:
    """Convert PDF pages to images using pdf2image (requires poppler)."""
    try:
        from pdf2image import convert_from_path
        return convert_from_path(str(path), dpi=200)
    except ImportError:
        logger.warning("pdf2image not available — falling back to PIL PDF reader")
        return _pdf_fallback(path.read_bytes())
    except Exception as e:
        logger.error("pdf2image error: %s", e)
        return _pdf_fallback(path.read_bytes())


def _from_pdf_bytes(data: bytes) -> list[Image.Image]:
    try:
        from pdf2image import convert_from_bytes
        return convert_from_bytes(data, dpi=200)
    except ImportError:
        return _pdf_fallback(data)
    except Exception as e:
        logger.error("pdf2image bytes error: %s", e)
        return _pdf_fallback(data)


def _pdf_fallback(data: bytes) -> list[Image.Image]:
    """Very basic PIL-based PDF page reader (only works for single-page image PDFs)."""
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return [img]
    except Exception:
        raise RuntimeError(
            "Could not convert PDF. Install poppler (apt install poppler-utils) "
            "for full PDF support."
        )
