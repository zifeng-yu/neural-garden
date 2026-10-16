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

ES_URL = CONFIG["elasticsearch"]["url"]
ES_INDEX_NAME = CONFIG["elasticsearch"]["index_name"]
ES_REQUEST_TIMEOUT = CONFIG["elasticsearch"]["request_timeout"]

DASHSCOPE_BASE_URL = CONFIG["dashscope"]["base_url"]
LLM_TEMPERATURE = CONFIG["dashscope"]["temperature"]
LLM_ENABLE_THINKING = CONFIG["dashscope"]["enable_thinking"]
LLM_RETRY_ATTEMPTS = CONFIG["dashscope"]["retry_attempts"]
LLM_RETRY_DELAY = CONFIG["dashscope"]["retry_delay"]
LLM_RETRY_BACKOFF = CONFIG["dashscope"]["retry_backoff"]
EXTRACTION_RETRY_ATTEMPTS = CONFIG["dashscope"]["extraction_retry_attempts"]
CHROMA_DISTANCE_METRIC = CONFIG["chroma"]["distance_metric"]
CONCEPT_SIMILARITY_THRESHOLD = float(
    CONFIG["chroma"].get("concept_similarity_threshold", 0.4)
)
ES_NUMBER_OF_SHARDS = CONFIG["elasticsearch"]["number_of_shards"]
ES_NUMBER_OF_REPLICAS = CONFIG["elasticsearch"]["number_of_replicas"]
AGENT_TEMPERATURE = CONFIG["agent"]["temperature"]
AGENT_REQUEST_TIMEOUT = CONFIG["agent"]["request_timeout"]
AGENT_MAX_RETRIES = CONFIG["agent"]["max_retries"]
AGENT_ENABLE_THINKING = CONFIG["agent"]["enable_thinking"]
AGENT_RECURSION_LIMIT = CONFIG["agent"]["recursion_limit"]
LOG_DIRECTORY = str(PROJECT_ROOT / CONFIG["logging"]["directory"])
LOG_LEVEL = CONFIG["logging"]["level"]
LOG_MAX_BYTES = CONFIG["logging"]["max_bytes"]
LOG_BACKUP_COUNT = CONFIG["logging"]["backup_count"]
LOG_ERROR_BACKUP_COUNT = CONFIG["logging"]["error_backup_count"]
