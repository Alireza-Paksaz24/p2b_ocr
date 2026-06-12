"""
model_registry.py — Load and manage OCR model configs from models.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

MODELS_YAML = Path(__file__).parent / "models.yaml"
MODELS_DIR = Path(os.getenv("MODELS_DIR", "./models"))


class ModelConfig(BaseModel):
    id: str
    name: str
    description: str
    hf_tag: str
    downloaded: bool = False
    size_gb: float = 0.0
    supports_table: bool = False
    supports_diagram: bool = False
    supports_formula: bool = False
    local_path: Optional[str] = None
    adapter: str = "stub"   # matches adapter key in ocr_engine


def load_models() -> list[ModelConfig]:
    with open(MODELS_YAML, "r") as f:
        data = yaml.safe_load(f)
    models = [ModelConfig(**m) for m in data.get("models", [])]

    for m in models:
        candidate = MODELS_DIR / m.id
        if candidate.exists():
            m.downloaded = True
            m.local_path = str(candidate)

    return models


def get_model(model_id: str) -> Optional[ModelConfig]:
    for m in load_models():
        if m.id == model_id:
            return m
    return None


def mark_downloaded(model_id: str, local_path: str) -> None:
    with open(MODELS_YAML, "r") as f:
        data = yaml.safe_load(f)
    for m in data.get("models", []):
        if m["id"] == model_id:
            m["downloaded"] = True
            break
    with open(MODELS_YAML, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)