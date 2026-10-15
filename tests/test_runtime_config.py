"""验证运行配置传入模型SDK；不调用外部服务。"""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "module_name", ["src.knowledge_retrieval_agent_cli", "src.usage_metrics_agent_cli"]
)
def test_agent_uses_configured_model_connection(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    settings = {
        "LLM_MODEL": "configured-model",
        "DASHSCOPE_BASE_URL": "https://example.invalid/v1",
        "AGENT_TEMPERATURE": 0.25,
        "AGENT_REQUEST_TIMEOUT": 41,
        "AGENT_MAX_RETRIES": 2,
        "AGENT_ENABLE_THINKING": True,
    }
    for name, value in settings.items():
        monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(module, "get_dashscope_api_key", lambda: "test-key")
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setattr(module, "ChatOpenAI", factory)
    graph = Mock()
    monkeypatch.setattr(module, "build_graph", graph)
    if module_name == "src.usage_metrics_agent_cli":
        tools = Mock(return_value=[])
        monkeypatch.setattr(module, "create_usage_tools", tools)
    module.create_app()
    kwargs = factory.call_args.kwargs
    assert kwargs["model"] == settings["LLM_MODEL"]
    assert kwargs["base_url"] == settings["DASHSCOPE_BASE_URL"]
    assert kwargs["temperature"] == settings["AGENT_TEMPERATURE"]
    assert kwargs["timeout"] == settings["AGENT_REQUEST_TIMEOUT"]
    assert kwargs["max_retries"] == settings["AGENT_MAX_RETRIES"]
    assert kwargs["extra_body"] == {"enable_thinking": True}
    graph.assert_called_once()
    expected_prompt = (
        module.SYSTEM_PROMPT
        if module_name == "src.knowledge_retrieval_agent_cli"
        else module.SAGE_SYSTEM_PROMPT
    )
    assert graph.call_args.kwargs["system_prompt"] == expected_prompt
    if module_name == "src.usage_metrics_agent_cli":
        tools.assert_called_once_with()


@pytest.mark.parametrize(
    "module_name", ["src.util.callDashscopellm", "src.knowledge.knowledge_extractor"]
)
def test_llm_uses_configured_generation_settings(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "LLM_MODEL", "configured-model")
    monkeypatch.setattr(module, "LLM_TEMPERATURE", 0.25)
    monkeypatch.setattr(module, "LLM_ENABLE_THINKING", True)
    monkeypatch.setattr(module, "get_dashscope_api_key", lambda: "test-key")
    monkeypatch.setattr(module.dashscope, "api_key", "test-key")
    text = json.dumps({"title": "标题", "summary": "摘要", "keywords": []})
    call = Mock(
        return_value=SimpleNamespace(
            status_code=200,
            output=SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=[{"text": text}]))
                ]
            ),
        )
    )
    monkeypatch.setattr(module.MultiModalConversation, "call", call)
    if module_name == "src.util.callDashscopellm":
        assert module.generate("系统提示", "问题") == text
    else:
        assert module.extractor_by_llm("原文").title == "标题"
    kwargs = call.call_args.kwargs
    assert kwargs["model"] == "configured-model"
    assert kwargs["temperature"] == 0.25
    assert kwargs["enable_thinking"] is True
