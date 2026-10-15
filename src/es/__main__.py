"""python -m src.es：选择同步方式，并可选执行BM25查询。"""

import argparse
import json
import logging
from dataclasses import asdict

from src.config.logging_config import setup_logging
from src.es.ingest import sync_index
from src.retrieval.es_search import search_bm25

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["skip", "upsert", "reset"], default="skip")
    parser.add_argument("--query", default=None, help="可选的BM25查询")
    parser.add_argument("--candidate-k", type=int, default=10)
    parser.add_argument(
        "--no-save-session", action="store_true", help="查询时不保存session和结果"
    )
    args = parser.parse_args()
    if args.query is not None and (not args.query.strip() or args.candidate_k <= 0):
        parser.error("查询须非空，candidate-k须为正整数")
    setup_logging()
    result = sync_index(args.mode)
    logger.info(json.dumps(result, ensure_ascii=False))
    if args.query is not None:
        hits = search_bm25(
            args.query, args.candidate_k, save_session=not args.no_save_session
        )
        logger.info(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False))


if __name__ == "__main__":
    main()
