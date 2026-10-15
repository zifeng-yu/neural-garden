"""API 健康检查、请求校验与响应契约的离线测试。"""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

import src.api as api


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(api, "setup_logging", Mock())
    backend = Mock()
    monkeypatch.setattr(api, "run_answer", backend)
    with TestClient(api.app) as client:
        yield client, backend


def test_health(env):
    client, backend = env
    assert client.get("/health").json() == {"status": "ok"}
    backend.assert_not_called()


def test_normal_contract(env):
    client, backend = env
    payload = {
        "status": "answered",
        "claims": [{"text": "Supported answer", "source_ids": ["S2"]}],
        "sources": [
            {
                "source_id": "S1",
                "document_id": 4,
                "chunk_id": 17,
                "content_hash": "h17",
                "text": "Related passage",
            },
            {
                "source_id": "S2",
                "document_id": 4,
                "chunk_id": 19,
                "content_hash": "h19",
                "text": "Supporting passage",
            },
        ],
    }
    backend.return_value = payload
    response = client.post("/answers", json={"question": "  question  ", "top_k": 2})
    assert response.status_code == 200, response.text
    assert backend.call_count == 1
    args, kwargs = backend.call_args
    # Accept positional or keyword calls without prescribing implementation style.
    question = args[0] if args else kwargs.get("question")
    top_k = args[1] if len(args) > 1 else kwargs.get("top_k")
    assert (question, top_k) == ("question", 2)
    assert response.json() == payload


def test_blank_question(env):
    client, backend = env
    response = client.post("/answers", json={"question": "   "})
    assert response.status_code == 422
    backend.assert_not_called()
