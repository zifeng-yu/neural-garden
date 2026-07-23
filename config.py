import os
import re

import yaml
from dotenv import load_dotenv

# 加载 .env
load_dotenv()


def _load_config():

    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

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
CHROMA_TABLE_NAME = CONFIG["chroma"]["table_name"]
PILOT_DATASET_PATH = CONFIG["pilot_dataset"]["path"]
