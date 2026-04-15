"""Bundled fixtures for smoke-tests and `ragcheck bench`."""
from __future__ import annotations

from pathlib import Path

FIXTURES_ROOT = Path(__file__).parent

BEIR_FIQA_DIR = FIXTURES_ROOT / "beir_fiqa"
MS_MARCO_DIR = FIXTURES_ROOT / "ms_marco"
NEEDLE_DIR = FIXTURES_ROOT / "needle_haystack"


def list_fixture_names() -> list:
    return ["beir_fiqa", "ms_marco", "needle_haystack"]


def fixture_dir(name: str) -> Path:
    mapping = {
        "beir_fiqa": BEIR_FIQA_DIR,
        "ms_marco": MS_MARCO_DIR,
        "needle_haystack": NEEDLE_DIR,
    }
    if name not in mapping:
        raise KeyError(f"unknown fixture: {name!r}. available: {sorted(mapping.keys())}")
    return mapping[name]
