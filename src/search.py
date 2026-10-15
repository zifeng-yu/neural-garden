"""统一检索入口：选择向量、BM25或混合模式，返回候选，不生成回答。"""

import argparse
import json
import logging
from dataclasses import asdict
from typing import Literal, get_args

from src.config.logging_config import setup_logging
from src.knowledgeGraph.knowledge_graph import build_concept_graph
from src.retrieval.chroma_search import search_vector
from src.retrieval.es_search import search_bm25
from src.retrieval.graph_search import graph_search_by_hits, graph_search_by_query
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.search_result import SearchResult

logger = logging.getLogger(__name__)
RetrievalMode = Literal["vector", "bm25", "hybrid", "graph"]
RETRIEVAL_MODES = get_args(RetrievalMode)


def search(
    query: str,
    *,
    top_k: int = 3,
    show_score: bool = True,
    save_session: bool = True,
    retrieval_mode: RetrievalMode = "vector",
    use_graph: bool = False,
) -> list[SearchResult]:
    """返回最多 top_k 条最终候选；可选图增强，只保存最终结果。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query必须是非空字符串")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k必须是正整数")
    if retrieval_mode not in RETRIEVAL_MODES:
        raise ValueError(f"retrieval_mode 必须是 {', '.join(RETRIEVAL_MODES)}")
    query = query.strip()
    if retrieval_mode == "graph":
        # 目前每次都会生成一次图 数据量大后需要更改
        return graph_search_by_query(
            query,
            build_concept_graph(),
            min_similarity=0.5,
            save_session=save_session,
            top_k=top_k,
        )
    if retrieval_mode == "vector":
        search_result = search_vector(
            query=query,
            top_k=top_k if not use_graph else top_k * 2,
            show_score=show_score,
            save_session=save_session and not use_graph,
        )
    if retrieval_mode == "bm25":
        search_result = search_bm25(
            query=query,
            candidate_k=top_k if not use_graph else top_k * 2,
            save_session=save_session and not use_graph,
        )
    if retrieval_mode == "hybrid":
        search_result = hybrid_search(
            query=query,
            top_k=top_k if not use_graph else top_k * 2,
            candidate_k=top_k * 3,
            save_session=save_session and not use_graph,
        )
    if not use_graph:
        return search_result
    # 目前每次都会生成一次图 数据量大后需要更改
    return graph_search_by_hits(
        build_concept_graph(),
        search_result,
        query=query,
        save_session=save_session,
        top_k=top_k,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="查询内容")
    parser.add_argument("--mode", choices=RETRIEVAL_MODES, default="vector")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-save-session", action="store_true")
    parser.add_argument("--use-graph", action="store_true")

    args = parser.parse_args()
    setup_logging()
    hits = search(
        " ".join(args.query),
        top_k=args.top_k,
        retrieval_mode=args.mode,
        save_session=not args.no_save_session,
        use_graph=args.use_graph,
    )
    logger.info(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False))


if __name__ == "__main__":
    main()
