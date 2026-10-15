"""知识单元索引的创建、校验与显式重置。"""

import re
from typing import Any

from elasticsearch import Elasticsearch

from src.config.config import ES_INDEX_NAME, ES_NUMBER_OF_REPLICAS, ES_NUMBER_OF_SHARDS

INDEX_MAPPING: dict[str, Any] = {
    "dynamic": "strict",
    "properties": {
        "knowledge_unit_id": {"type": "keyword"},
        "document_id": {"type": "long"},
        "document_chunk_id": {"type": "long"},
        "retrieval_text": {"type": "text", "analyzer": "cjk"},
        "chunk_hash": {"type": "keyword"},
    },
}


def validate_index_name(index_name: str) -> None:
    # 只接受单个具体名称，避免重置参数成为通配符或多个索引。
    if (
        not isinstance(index_name, str)
        or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", index_name)
        or len(index_name.encode()) > 255
    ):
        raise ValueError("索引名须为单个小写名称，仅含字母、数字、下划线和连字符")


def create_index(client: Elasticsearch, *, reset: bool = False) -> bool:
    """缺失时创建；已有时校验mapping；reset=True仅重置指定索引。"""
    validate_index_name(ES_INDEX_NAME)
    exists = bool(client.indices.exists(index=ES_INDEX_NAME))
    if exists and reset:
        response = client.indices.delete(index=ES_INDEX_NAME)
        if not response["acknowledged"]:
            raise RuntimeError("ES索引删除未确认")
        exists = False
    if not exists:
        response = client.indices.create(
            index=ES_INDEX_NAME,
            settings={
                "number_of_shards": ES_NUMBER_OF_SHARDS,
                "number_of_replicas": ES_NUMBER_OF_REPLICAS,
            },
            mappings=INDEX_MAPPING,
        )
        if not response["acknowledged"]:
            raise RuntimeError("ES索引创建未确认")
        return True

    mappings = client.indices.get_mapping(index=ES_INDEX_NAME)[ES_INDEX_NAME][
        "mappings"
    ]
    if mappings.get("dynamic") != "strict":
        raise ValueError("现有ES索引不是strict mapping，请指定新索引或显式重置")
    for field, expected in INDEX_MAPPING["properties"].items():
        actual = mappings.get("properties", {}).get(field, {})
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"现有ES索引字段{field}不兼容，请指定新索引或显式重置")
    return False
