"""API 检索数量校验与引用数据保留的离线测试。"""

from unittest.mock import Mock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import src.api as api


def test_rejects_float_top_k(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(api, "setup_logging", Mock())
    backend = Mock()
    monkeypatch.setattr(api, "run_answer", backend)
    with TestClient(api.app) as client:
        response = client.post("/answers", json={"question": "我的问题", "top_k": 1.0})
        assert response.status_code == 422
        backend.assert_not_called()


def test_preserves_citation_data(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(api, "setup_logging", Mock())
    backend = Mock()
    monkeypatch.setattr(api, "run_answer", backend)
    payload = {
        "status": "answered",
        "claims": [{"text": "访客可以提前三天预约。", "source_ids": ["S2"]}],
        "sources": [
            {
                "source_id": "S1",
                "document_id": 1,
                "chunk_id": 10,
                "content_hash": "hash10",
                "text": "其他相关资料",
            },
            {
                "source_id": "S2",
                "document_id": 1,
                "chunk_id": 20,
                "content_hash": "hash20",
                "text": "访客预约窗口为三天。",
            },
        ],
    }
    backend.return_value = payload
    with TestClient(api.app) as client:
        response = client.post(
            "/answers",
            json={"question": "根据"},
        )
        assert response.status_code == 200
        assert response.json() == payload


# 最后再说明：当前同步调用在哪个线程执行，换成async def直接调用会有什么影响。
# 当前同步调用在 Python 线程中执行，FastAPI 使用线程池运行同步 def 端点。
# 换成async def 会阻塞线程，阻塞其他请求进入。因为调用模型、访问数据库都是同步api
