"""核对SQLite/Chroma知识单元，并按稳定ID同步到ES。"""

from collections.abc import Iterable
from typing import Any, Literal, TypedDict, cast

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from src.config.config import CHROMA_KNOWLEDGE_TABLE_NAME, ES_INDEX_NAME
from src.es.client import get_es_client
from src.es.index import create_index, validate_index_name
from src.get_chroma_collection import get_existing_collection
from src.get_sqlite_connection import get_sqlite_read_connection

SyncMode = Literal["skip", "upsert", "reset"]


class KnowledgeRecord(TypedDict):
    knowledge_unit_id: int
    document_id: int
    document_chunk_id: int
    retrieval_text: str
    chunk_hash: str


def export_data() -> list[KnowledgeRecord]:
    """先核对身份、文本与元数据；不一致时禁止同步或重置ES。"""
    with get_sqlite_read_connection() as conn:
        rows = conn.execute("""
            SELECT k.id AS knowledge_unit_id, k.document_id, k.document_chunk_id,
                   k.embedding_text AS retrieval_text, c.content_hash AS chunk_hash
            FROM document_chunk_knowledge_units AS k
            JOIN document_chunks AS c
              ON c.id = k.document_chunk_id AND c.document_id = k.document_id
            ORDER BY k.id
            """).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM document_chunk_knowledge_units"
        ).fetchone()[0]
    records = [dict(row) for row in rows]
    errors = []
    if len(records) != total:
        errors.append(f"JOIN导出{len(records)}条，知识单元总数{total}条")
    for row in records:
        for field in ("knowledge_unit_id", "document_id", "document_chunk_id"):
            if type(row[field]) is not int or row[field] <= 0:
                errors.append(f"知识单元{row['knowledge_unit_id']}的{field}无效")
        for field in ("retrieval_text", "chunk_hash"):
            if not isinstance(row[field], str) or not row[field].strip():
                errors.append(f"知识单元{row['knowledge_unit_id']}的{field}为空")

    collection = get_existing_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
    result = collection.get(include=["documents", "metadatas"])
    if result["documents"] is None or result["metadatas"] is None:
        raise ValueError("Chroma未返回检索文本或元数据")
    sqlite_by_id = {str(row["knowledge_unit_id"]): row for row in records}
    chroma_by_id = {
        str(unit_id): (text, metadata or {})
        for unit_id, text, metadata in zip(
            result["ids"], result["documents"], result["metadatas"], strict=True
        )
    }
    sqlite_ids, chroma_ids = set(sqlite_by_id), set(chroma_by_id)
    if sqlite_ids - chroma_ids:
        errors.append(f"仅SQLite存在的ID：{sorted(sqlite_ids - chroma_ids)}")
    if chroma_ids - sqlite_ids:
        errors.append(f"仅Chroma存在的ID：{sorted(chroma_ids - sqlite_ids)}")
    for unit_id in sorted(sqlite_ids & chroma_ids):
        row = sqlite_by_id[unit_id]
        text, metadata = chroma_by_id[unit_id]
        if row["retrieval_text"] != text:
            errors.append(f"知识单元{unit_id}的检索文本不一致")
        for field in ("document_id", "document_chunk_id"):
            if str(row[field]) != str(metadata.get(field)):
                errors.append(f"知识单元{unit_id}的{field}不一致")
    if errors:
        raise ValueError("导出数据核对失败：\n" + "\n".join(errors))
    return cast(list[KnowledgeRecord], records)


def upsert_documents(
    client: Elasticsearch,
    records: Iterable[KnowledgeRecord],
) -> int:
    """已有ID更新，缺失ID新增；失败抛错，成功数含未变化的noop。"""
    validate_index_name(ES_INDEX_NAME)
    actions = (
        {
            "_op_type": "update",
            "_index": ES_INDEX_NAME,
            "_id": str(row["knowledge_unit_id"]),
            "doc": {**row, "knowledge_unit_id": str(row["knowledge_unit_id"])},
            "doc_as_upsert": True,
        }
        for row in records
    )
    success_count, errors = bulk(client, actions, refresh="wait_for")
    if errors:
        raise RuntimeError(f"ES部分写入失败：{errors}")
    return success_count


def sync_index(mode: SyncMode = "skip") -> dict[str, Any]:
    """skip不访问源库或ES；upsert保留旧记录；reset清空后全量写入。"""
    if mode not in ("skip", "upsert", "reset"):
        raise ValueError("mode必须是skip、upsert或reset")
    validate_index_name(ES_INDEX_NAME)
    if mode == "skip":
        return {"mode": mode, "index_name": ES_INDEX_NAME, "processed": 0}
    # 校验在任何索引创建/删除前完成，避免源数据异常导致先丢失旧索引。
    records = export_data()
    with get_es_client() as client:
        created = create_index(client, reset=mode == "reset")
        processed = upsert_documents(client, records) if records else 0
        count = client.count(index=ES_INDEX_NAME)["count"]
    return {
        "mode": mode,
        "index_name": ES_INDEX_NAME,
        "index_created": created,
        "processed": processed,
        "document_count": count,
    }
