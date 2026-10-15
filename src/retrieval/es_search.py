"""BM25检索，返回统一SearchResult，可选保存搜索记录。"""

from src.config.config import ES_INDEX_NAME
from src.es.client import get_es_client
from src.es.index import validate_index_name
from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum
from src.repository.search_sessions import save_search_session_with_results
from src.retrieval.search_result import SearchResult


def search_bm25(
    query: str,
    candidate_k: int = 10,
    save_session: bool = True,
) -> list[SearchResult]:
    """按BM25分数降序返回候选；不创建索引、不导出或同步数据。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query必须是非空字符串")
    if type(candidate_k) is not int or candidate_k <= 0:
        raise ValueError("candidate_k必须是正整数")
    validate_index_name(ES_INDEX_NAME)
    with get_es_client() as client:
        result = client.search(
            index=ES_INDEX_NAME,
            size=candidate_k,
            query={
                "match": {"retrieval_text": {"query": query.strip(), "operator": "or"}}
            },
            sort=[{"_score": "desc"}, {"knowledge_unit_id": "asc"}],
        )
    if result["timed_out"] or result["_shards"]["failed"] > 0:
        raise RuntimeError("ES查询超时或分片失败")
    hits = [
        SearchResult(
            content=hit["_source"]["retrieval_text"],
            document_id=hit["_source"]["document_id"],
            document_chunk_id=hit["_source"]["document_chunk_id"],
            score=float(hit["_score"]),
            rank=rank,
            source_type=SourceTypeEnum.ES_KNOWLEDGE,
            retrieval_type=RetrievalTypeEnum.KEYWORD,
        )
        for rank, hit in enumerate(result["hits"]["hits"], start=1)
    ]
    if save_session:
        save_search_session_with_results(
            query.strip(), hits, RetrievalTypeEnum.KEYWORD, SourceTypeEnum.ES_KNOWLEDGE
        )
    return hits
