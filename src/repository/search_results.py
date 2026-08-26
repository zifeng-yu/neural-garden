import sqlite3
from dataclasses import dataclass
from enum import Enum

from src.get_sqlite_connection import get_sqlite_connection
from src.repository.base_do import BaseDO


class RetrievalTypeEnum(str, Enum):
    """检索类型"""

    VECTOR = "vector"
    KEYWORD = "KEYWORD/BM25"
    HYBRID = "hybrid"
    GRAPH = "graph"


class SourceTypeEnum(str, Enum):
    """来源"""

    CHROMA_KNOWLEDGE = "chroma_knowledge"
    CHROMA_INSIGHT = "chroma_insight"


@dataclass
class SearchResultDO(BaseDO):
    session_id: str
    document_id: str
    document_chunk_id: int
    rank: int
    score: float
    raw_score: float
    retrieval_type: RetrievalTypeEnum
    source_type: SourceTypeEnum


def _save_search_results(
    conn: sqlite3.Connection,
    session_id: str,
    document_id: int | None,
    document_chunk_id: int | None,
    rank: int,
    score: float,
    raw_score: float,
    retrieval_type: RetrievalTypeEnum,
    source_type: SourceTypeEnum,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO search_results
        (
            session_id,
            document_id,
            document_chunk_id,
            rank,
            score,
            raw_score,
            retrieval_type,
            source_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            document_id,
            document_chunk_id,
            rank,
            score,
            raw_score,
            retrieval_type.value,
            source_type.value,
        ),
    )
    if cursor.lastrowid is None:
        raise RuntimeError("insert search_results failed")
    return cursor.lastrowid


def save_search_results(
    session_id: str,
    document_id: int | None,
    document_chunk_id: int | None,
    rank: int,
    score: float,
    raw_score: float,
    retrieval_type: RetrievalTypeEnum,
    source_type: SourceTypeEnum,
) -> int:
    with get_sqlite_connection() as conn:
        return _save_search_results(
            conn,
            session_id,
            document_id,
            document_chunk_id,
            rank,
            score,
            raw_score,
            retrieval_type,
            source_type,
        )
