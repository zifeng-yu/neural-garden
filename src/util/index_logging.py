"""Structured indexing timings; nested stages inherit document/chunk identity."""

import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any

_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "index_log_context", default=None
)


def _emit(logger: logging.Logger, event: dict[str, Any]) -> None:
    # Observability must not fail a model call or replace its original exception.
    try:
        # 同一阶段事件同时服务于运行排障和离线评估，避免重复写两份日志。
        event = {**event, "log_categories": ["runtime", "evaluation"]}
        logger.log(
            logging.ERROR if event.get("outcome") == "failed" else logging.INFO,
            "INDEX %s",
            json.dumps(event, ensure_ascii=False),
        )
    except Exception:
        pass


@contextmanager
def index_stage(
    logger: logging.Logger, stage: str, **fields: Any
) -> Generator[dict[str, Any], None, None]:
    """Log start/end/failure without changing exception or return semantics.

    Callers supply identifiers and counts only, never credentials or raw prompts.
    Durations are inclusive of nested stages and must not be summed across levels.
    """
    details = {**(_context.get() or {}), **fields, "stage": stage}
    token = _context.set(details)
    started = perf_counter()
    _emit(logger, {**details, "event": "start"})
    try:
        yield details
    except Exception as exc:
        details.update(outcome="failed", error_type=type(exc).__name__)
        raise
    finally:
        elapsed = round(perf_counter() - started, 4)
        event = {**details, "event": "end", "seconds": elapsed}
        event.setdefault("outcome", "ok")
        _emit(logger, event)
        _context.reset(token)
