import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

# 加载 .env
load_dotenv()


def _load_config():

    config_path = Path(__file__).resolve().parents[2] / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    def replace_env(match):

        key = match.group(1)

        value = os.getenv(key)

        if value is None:
            raise ValueError(f"环境变量 {key} 未配置")

        return value

    content = re.sub(r"\$\{(\w+)\}", replace_env, content)

    return yaml.safe_load(content)


CONFIG = _load_config()

API_KEY = CONFIG["dashscope"]["api_key"]
EMBEDDING_MODEL = CONFIG["dashscope"]["embedding_model"]
PERSIST_DIRECTORY = CONFIG["chroma"]["persist_directory"]
CHROMA_KNOWLEDGE_TABLE_NAME = CONFIG["chroma"]["knowledge_table_name"]
PILOT_DATASET_PATH = CONFIG["pilot_dataset"]["path"]
LLM_MODEL = CONFIG["dashscope"]["llm_model"]
CHROMA_CONCEPT_TABLE_NAME = CONFIG["chroma"]["concept_table_name"]
