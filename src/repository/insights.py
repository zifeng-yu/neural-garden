import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO


class SourceTypeEnum(str, Enum):
    USER = "user"
    DOCUMENT = "document"


class SourceStatusEnum(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass
class InsightsDO(BaseDO):
    title: str
    content: str
    action_items: list[str]
    relation_concepts: list[str]
    source_type: SourceTypeEnum
    document_id: int | None
    document_chunk_id: int | None
    source_status: SourceStatusEnum = SourceStatusEnum.ONLINE


def save_insights(
    title: str,
    content: str,
    action_items: list[str],
    relation_concepts: list[str],
    source_type: SourceTypeEnum,
    document_id: int | None,
    document_chunk_id: int | None,
    source_status: SourceStatusEnum = SourceStatusEnum.ONLINE,
) -> int:
    with get_sqlite_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO insights
            (
                title,
                content,
                action_items,
                relation_concepts,
                source_type,
                document_id,
                document_chunk_id,
                source_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                content,
                json.dumps(action_items, ensure_ascii=False),
                json.dumps(relation_concepts, ensure_ascii=False),
                source_type.value,
                document_id,
                document_chunk_id,
                source_status.value,
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("insert insight failed")
        return cursor.lastrowid


def mark_source_status_offline_by_document_id(document_id: int) -> list[int]:
    with get_sqlite_connection() as conn:
        rows = conn.execute(
            """
            update insights set source_status = ? where document_id = ? returning id
            """,
            (
                SourceStatusEnum.OFFLINE.value,
                document_id,
            ),
        ).fetchall()
        return [row["id"] for row in rows]


def query_by_id(id: int) -> InsightsDO | None:
    with get_sqlite_connection() as conn:
        row = conn.execute(
            """
                select * from insights where id = ?
            """,
            (id,),
        ).fetchone()
        if row is None:
            return None

        data = dict(row)

        data["action_items"] = json.loads(data["action_items"])
        data["relation_concepts"] = json.loads(data["relation_concepts"])
        data["source_type"] = SourceTypeEnum(data["source_type"])
        data["source_status"] = SourceStatusEnum(data["source_status"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        return InsightsDO(**data)
