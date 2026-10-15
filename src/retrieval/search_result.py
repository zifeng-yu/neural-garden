"""各检索方式共用的结果结构。"""

from dataclasses import dataclass

from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum


@dataclass
class SearchResult:
    """检索统一的契约"""

    content: str
    document_id: int | None
    document_chunk_id: int | None
    score: float
    rank: int
    source_type: SourceTypeEnum
    retrieval_type: RetrievalTypeEnum
    raw_score: float | None = None
