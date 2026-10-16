import json
import logging

import pytest

from src.util.index_logging import index_stage


def events(caplog):
    return [
        json.loads(r.getMessage().removeprefix("INDEX "))
        for r in caplog.records
        if r.getMessage().startswith("INDEX ")
    ]


def test_nested_stage_identity_does_not_leak_to_next_document(caplog):
    logger = logging.getLogger(__name__)
    caplog.set_level(logging.INFO)
    with index_stage(logger, "document", source="first.md"):
        with index_stage(logger, "knowledge", chunk_no=2) as stats:
            stats["outcome"] = "skipped"
    with index_stage(logger, "document", source="second.md"):
        pass
    finished = [e for e in events(caplog) if e["event"] == "end"]
    assert finished[0]["source"] == "first.md"
    assert finished[0]["chunk_no"] == 2
    assert finished[0]["outcome"] == "skipped"
    assert finished[1]["outcome"] == "ok"
    assert "chunk_no" not in finished[2]
    assert finished[2]["source"] == "second.md"


def test_failed_stage_preserves_exception_without_logging_its_sensitive_text(caplog):
    logger = logging.getLogger(__name__)
    caplog.set_level(logging.INFO)
    failure = RuntimeError("private-prompt-and-secret")
    with pytest.raises(RuntimeError) as caught:
        with index_stage(logger, "document", source="failed.md"):
            raise failure
    assert caught.value is failure
    event = events(caplog)[-1]
    assert event["outcome"] == "failed"
    assert event["error_type"] == "RuntimeError"
    assert event["seconds"] >= 0
    assert "private-prompt-and-secret" not in caplog.text
    with index_stage(logger, "unrelated"):
        pass
    assert "source" not in events(caplog)[-1]


@pytest.mark.parametrize("failure_mode", ["serialization", "handler"])
def test_logging_failure_does_not_interrupt_work_or_mask_error(
    monkeypatch, failure_mode
):
    logger = logging.getLogger(__name__)
    fields = {"request_id": object()} if failure_mode == "serialization" else {}
    if failure_mode == "handler":

        def fail(*args, **kwargs):
            raise OSError("log destination unavailable")

        monkeypatch.setattr(logger, "log", fail)
    completed = []
    with index_stage(logger, "success", **fields):
        completed.append(True)
    assert completed == [True]
    original = ValueError("business error")
    with pytest.raises(ValueError) as caught:
        with index_stage(logger, "failure", **fields):
            raise original
    assert caught.value is original
