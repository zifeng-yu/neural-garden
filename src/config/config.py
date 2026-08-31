import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _load_config():
    config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("config.yaml 必须是 YAML mapping")

    return config


CONFIG = _load_config()


@lru_cache
def get_dashscope_api_key() -> str:
    api_key = os.getenv("API_KEY")

    if not api_key:
        raise RuntimeError("API_KEY is required when using DashScope")

    return api_key


EMBEDDING_MODEL = CONFIG["dashscope"]["embedding_model"]
PERSIST_DIRECTORY = str(PROJECT_ROOT / CONFIG["chroma"]["persist_directory"])
CHROMA_KNOWLEDGE_TABLE_NAME = CONFIG["chroma"]["knowledge_table_name"]
PILOT_DATASET_PATH = str(PROJECT_ROOT / CONFIG["pilot_dataset"]["path"])
LLM_MODEL = CONFIG["dashscope"]["llm_model"]
CHROMA_CONCEPT_TABLE_NAME = CONFIG["chroma"]["concept_table_name"]
SQLITE_TABLE = str(PROJECT_ROOT / CONFIG["sqlite"]["sqlite_directory"])
CHROMA_INSIGHT_TABLE_NAME = CONFIG["chroma"]["insight_table_name"]
