import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
root = Path("/Users/zifeng/neural-garden")
sys.path.insert(0, str(root))
import src.rag_answer as rag
from src.config.config import ES_INDEX_NAME
from src.config.logging_config import setup_logging
from src.retrieval.es_search import search_bm25
from src.retrieval.hybrid_search import fuse_rrf, normalize_hits
from src.search import search

logger = logging.getLogger(__name__)
setup_logging()
p = root / "notes/hybrid-eval"
cases = [
    json.loads(l) for l in (p / "cases.jsonl").read_text().splitlines() if l.strip()
]
run = datetime.now(timezone.utc).isoformat()
with (p / "results.jsonl").open("a") as f:
    for case in cases:
        logger.info("START %s", case["id"])
        try:
            t = time.perf_counter()
            v = search(case["query"], top_k=10, show_score=False, save_session=False)
            vt = time.perf_counter() - t
            t = time.perf_counter()
            b = search_bm25(case["query"], candidate_k=10, save_session=False)
            bt = time.perf_counter() - t
            vn, bn = normalize_hits(v), normalize_hits(b)
            modes = {"vector": vn[:3], "bm25": bn[:3], "hybrid": fuse_rrf(v, b, 3)}
        except Exception as e:
            for mode in ["vector", "bm25", "hybrid"]:
                f.write(
                    json.dumps(
                        {
                            "run_id": run,
                            "case_id": case["id"],
                            "mode": mode,
                            "error": str(e),
                            "stage": "retrieval",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.flush()
            logger.error("FAILED %s %s", case["id"], str(e))
            continue
        trace = []
        for key in sorted({(h.document_id, h.document_chunk_id) for h in vn + bn}):
            vr = next(
                (h for h in vn if (h.document_id, h.document_chunk_id) == key), None
            )
            br = next(
                (h for h in bn if (h.document_id, h.document_chunk_id) == key), None
            )
            trace.append(
                {
                    "document_id": key[0],
                    "chunk_id": key[1],
                    "vector_rank": vr.rank if vr else None,
                    "bm25_rank": br.rank if br else None,
                    "vector_score": vr.score if vr else None,
                    "bm25_score": br.score if br else None,
                    "rrf_score": sum(1 / (60 + h.rank) for h in [vr, br] if h),
                    "final_rank": next(
                        (
                            h.rank
                            for h in modes["hybrid"]
                            if (h.document_id, h.document_chunk_id) == key
                        ),
                        None,
                    ),
                }
            )
        for mode, hits in modes.items():
            gold = {(g["document_id"], g["chunk_id"]) for g in case["gold_chunks"]}
            rank = next(
                (h.rank for h in hits if (h.document_id, h.document_chunk_id) in gold),
                None,
            )
            r = {
                "run_id": run,
                "case_id": case["id"],
                "query": case["query"],
                "split": case["split"],
                "answerable": case["answerable"],
                "mode": mode,
                "candidate_k": 10,
                "top_k": 3,
                "rrf_k": 60,
                "index": ES_INDEX_NAME,
                "max_chars": 3000,
                "raw_vector": [asdict(h) for h in v],
                "raw_bm25": [asdict(h) for h in b],
                "unique_counts": {"vector": len(vn), "bm25": len(bn)},
                "top_hits": [asdict(h) for h in hits],
                "trace": trace,
                "retrieval_seconds": {"vector": vt, "bm25": bt, "hybrid": vt + bt}[
                    mode
                ],
                "hit_at_3": int(rank is not None) if case["answerable"] else None,
                "mrr_at_3": 1 / rank if rank else (0 if case["answerable"] else None),
                "manual_review": "pending",
            }
            t = time.perf_counter()
            try:
                with patch.object(
                    rag,
                    "search",
                    return_value={"vector": v, "bm25": b, "hybrid": modes["hybrid"]}[
                        mode
                    ],
                ):
                    a = rag.answer_question(
                        case["query"], retrieval_mode=mode, top_k=3, candidate_k=10
                    )
                if a is None:
                    raise RuntimeError("answer_question returned None")
                r["answer"] = a
                r["source_hit"] = (
                    any((s["document_id"], s["chunk_id"]) in gold for s in a["sources"])
                    if case["answerable"]
                    else None
                )
            except Exception as e:
                r["error"] = str(e)
                r["stage"] = "answer"
            r["answer_seconds"] = time.perf_counter() - t
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            logger.info(
                "DONE %s %s %s",
                case["id"],
                mode,
                r.get("answer", {}).get("status", r.get("error")),
            )
logger.info("COMPLETE %s", run)
