from dataclasses import replace

from src.config.config import CHROMA_KNOWLEDGE_TABLE_NAME
from src.get_chroma_collection import get_existing_collection
from src.repository.search_results import RetrievalTypeEnum, SourceTypeEnum
from src.repository.search_sessions import save_search_session_with_results
from src.retrieval.chroma_search import search_vector
from src.retrieval.es_search import search_bm25
from src.retrieval.search_result import SearchResult


def normalize_hits(hits: list[SearchResult]) -> list[SearchResult]:
    """按chunk身份去重，保留最早命中，复制结果并重编号；不修改输入。"""
    result = []
    visit = set()
    sorted_hits = sorted(hits, key=lambda hit: hit.rank)
    new_rank = 1
    for hit in sorted_hits:
        if hit.document_id is None or hit.document_chunk_id is None:
            continue
        key = (hit.document_id, hit.document_chunk_id)
        if key in visit:
            continue
        visit.add(key)
        result.append(
            replace(
                hit,
                rank=new_rank,
            )
        )
        new_rank += 1
    return result


def fuse_rrf(
    first_hits: list[SearchResult],
    second_hits: list[SearchResult],
    top_k: int = 3,
    rrf_k: int = 60,
) -> list[SearchResult]:
    """先各路normalize，再累加名次贡献，稳定排序，返回最终top_k。"""
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k 必须是正整数")
    if type(rrf_k) is not int or rrf_k <= 0:
        raise ValueError("rrf_k 必须是正整数")
    first_hits = normalize_hits(first_hits)
    second_hits = normalize_hits(second_hits)
    scores: dict[tuple[int, int], float] = {}
    representatives: dict[tuple[int, int], SearchResult] = {}

    # 先遍历第一路，因此同一 chunk 优先保留第一路文本
    for hits in (first_hits, second_hits):
        for hit in hits:
            if hit.document_id is None or hit.document_chunk_id is None:
                continue
            key = (hit.document_id, hit.document_chunk_id)
            scores[key] = scores.get(key, 0.0) + 1 / (rrf_k + hit.rank)
            representatives.setdefault(key, hit)

    sorted_items = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )

    results = []
    for rank, (key, score) in enumerate(sorted_items[:top_k], start=1):
        results.append(
            replace(
                representatives[key],
                score=score,
                raw_score=score,
                rank=rank,
                source_type=SourceTypeEnum.HYBRID_KNOWLEDGE,
                retrieval_type=RetrievalTypeEnum.HYBRID,
            )
        )

    return results


def hybrid_search(
    query: str,
    top_k: int = 3,
    candidate_k: int = 10,
    save_session: bool = True,
) -> list[SearchResult]:
    """取得两路候选，成功后融合；可选只保存最终融合结果。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query必须是非空字符串")
    query = query.strip()
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k 必须是 >0 的整数")
    if type(candidate_k) is not int or candidate_k < top_k:
        raise ValueError("candidate_k 必须是整数且不小于 top_k")

    bm25_result = search_bm25(query=query, candidate_k=candidate_k, save_session=False)

    collection = get_existing_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
    count = collection.count()
    vector_result: list[SearchResult]
    if count == 0:
        vector_result = []
    else:
        vector_result = search_vector(
            query=query,
            top_k=min(candidate_k, count),
            show_score=False,
            save_session=False,
        )

    results = fuse_rrf(vector_result, bm25_result, top_k=top_k)
    if save_session:
        save_search_session_with_results(
            query, results, RetrievalTypeEnum.HYBRID, SourceTypeEnum.HYBRID_KNOWLEDGE
        )
    return results
