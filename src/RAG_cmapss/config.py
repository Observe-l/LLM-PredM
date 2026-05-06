from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KG_DIR = PROJECT_ROOT / "knowledge_graph" / "CMAPSS"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "CMAPSS" / "RAG"
DEFAULT_MODEL_NAME = "qwen3.5:9b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"

KG_TRIPLE_FILES = [
    "static_domain_triples.csv",
    "sensor_mapping.csv",
    "alias_mapping.csv",
    "dataset_rules.csv",
    "hypothesis_action_rules.csv",
]

