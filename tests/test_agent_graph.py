from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.agent_graph.agent_graph import build_graph

TEST_SYSTEM_PROMPT = "测试助手：按工具结果回答。"


def test_system_prompt_is_required():
    from unittest.mock import Mock

    import pytest

    with pytest.raises(TypeError, match="system_prompt"):
        build_graph(Mock(), [])


def test_invalid_system_prompt_is_rejected_before_model_call():
    from unittest.mock import Mock

    import pytest

    model = Mock()
    for prompt in ("", " \n\t", None, 123):
        with pytest.raises(ValueError, match="system_prompt必须是非空字符串"):
            build_graph(model, [], system_prompt=prompt)
    model.invoke.assert_not_called()


def test_caller_system_prompt_reaches_model_unchanged():
    from unittest.mock import Mock

    model = Mock()
    model.invoke.return_value = AIMessage(content="完成")
    prompt = "调用方自己的业务规则"
    graph = build_graph(model, [], system_prompt=prompt)
    graph.invoke({"messages": [HumanMessage(content="你好")], "model_calls": 0})
    assert model.invoke.call_args.args[0][0].content == prompt


@tool
def search_fixture(query: str) -> str:
    """Return a fixed search candidate for graph testing."""
    return '{"document_id": 4, "chunk_id": 20}'


@tool
def read_fixture(document_id: int, chunk_id: int) -> str:
    """Return a fixed source for graph testing."""
    return '{"source_id":"D4:C20","text":"狐狸型承认不确定性"}'


class ScriptedModel:
    def __init__(self):
        self.seen = []

    def invoke(self, messages):
        self.seen.append(list(messages))
        last = messages[-1]
        if isinstance(last, HumanMessage):
            name, args = "search_fixture", {"query": "狐狸"}
        elif last.name == "search_fixture":
            name, args = "read_fixture", {"document_id": 4, "chunk_id": 20}
        else:
            return AIMessage(content="狐狸型承认不确定性[D4:C20]")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": name,
                    "args": args,
                    "id": f"call-{len(self.seen)}",
                    "type": "tool_call",
                }
            ],
        )


def test_graph_keeps_history_and_pairs_tools():
    model = ScriptedModel()
    graph = build_graph(
        model, [search_fixture, read_fixture], system_prompt=TEST_SYSTEM_PROMPT
    )
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="狐狸型是什么？")],
            "model_calls": 0,
        }
    )
    messages = result["messages"]
    assert [m.type for m in messages] == ["human", "ai", "tool", "ai", "tool", "ai"]
    requests = [
        call["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for call in m.tool_calls
    ]
    responses = [m.tool_call_id for m in messages if isinstance(m, ToolMessage)]
    assert requests == responses
    assert result["model_calls"] == 3
    assert len(model.seen[-1]) == 6  # system＋已有5条业务消息
    assert "[D4:C20]" in messages[-1].content


def test_direct_text_reply_skips_tools_and_calls_model_once():
    from unittest.mock import Mock

    executed = Mock()

    @tool
    def forbidden_fixture(query: str) -> str:
        """Track unexpected tool execution during a direct reply."""
        executed(query)
        raise AssertionError("直接回答不应执行工具")

    model = Mock()
    model.invoke.return_value = AIMessage(content="你好！")
    graph = build_graph(model, [forbidden_fixture], system_prompt=TEST_SYSTEM_PROMPT)
    updates = list(
        graph.stream(
            {"messages": [HumanMessage(content="你好")], "model_calls": 0},
            stream_mode="updates",
        )
    )

    # 不仅检查没有工具消息，也检查执行路径没有进入 tools 节点。
    assert [name for update in updates for name in update] == ["model"]
    delta = updates[0]["model"]
    assert delta["model_calls"] == 1
    assert len(delta["messages"]) == 1
    reply = delta["messages"][0]
    assert isinstance(reply, AIMessage)
    assert reply.content == "你好！"
    assert reply.tool_calls == []
    model.invoke.assert_called_once()
    executed.assert_not_called()


def test_two_tool_calls_in_one_message_both_receive_results():
    from unittest.mock import Mock

    executed = Mock()

    @tool
    def echo_fixture(value: str) -> str:
        """Echo the input to verify independent tool calls."""
        executed(value)
        return f"result:{value}"

    model = Mock()
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "echo_fixture",
                    "args": {"value": "first"},
                    "id": "call-first",
                    "type": "tool_call",
                },
                {
                    "name": "echo_fixture",
                    "args": {"value": "second"},
                    "id": "call-second",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(content="两个工具都已完成"),
    ]
    graph = build_graph(model, [echo_fixture], system_prompt=TEST_SYSTEM_PROMPT)
    result = graph.invoke(
        {"messages": [HumanMessage(content="执行两个工具")], "model_calls": 0}
    )
    messages = result["messages"]
    replies = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(replies) == 2
    assert {m.tool_call_id: m.content for m in replies} == {
        "call-first": "result:first",
        "call-second": "result:second",
    }
    assert executed.call_count == 2
    assert sorted(c.args[0] for c in executed.call_args_list) == ["first", "second"]
    # 下一次模型调用必须同时看到两份工具结果。
    next_input = model.invoke.call_args_list[1].args[0]
    assert {m.tool_call_id for m in next_input if isinstance(m, ToolMessage)} == {
        "call-first",
        "call-second",
    }
    assert result["model_calls"] == 2
    assert model.invoke.call_count == 2
    assert messages[-1].content == "两个工具都已完成"


def test_same_thread_retains_first_round_and_appends_followup_once():
    from unittest.mock import Mock

    from langgraph.checkpoint.memory import InMemorySaver

    model = Mock()
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_fixture",
                    "args": {"query": "狐狸"},
                    "id": "round1-search",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_fixture",
                    "args": {"document_id": 4, "chunk_id": 20},
                    "id": "round1-read",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="狐狸型承认不确定性[D4:C20]"),
        AIMessage(content="刚才的原文是[D4:C20]"),
    ]
    graph = build_graph(
        model,
        [search_fixture, read_fixture],
        checkpointer=InMemorySaver(),
        system_prompt=TEST_SYSTEM_PROMPT,
    )
    config = {"configurable": {"thread_id": "same-thread-test"}}
    first = graph.invoke(
        {"messages": [HumanMessage(content="狐狸型是什么？")], "model_calls": 0},
        config=config,
    )
    first_messages = list(first["messages"])
    followup = "刚才的说法来自哪段原文？"
    second = graph.invoke(
        {"messages": [HumanMessage(content=followup)], "model_calls": 0},
        config=config,
    )
    messages = second["messages"]
    assert messages[: len(first_messages)] == first_messages
    assert [m.type for m in messages] == [
        "human",
        "ai",
        "tool",
        "ai",
        "tool",
        "ai",
        "human",
        "ai",
    ]
    assert (
        sum(isinstance(m, HumanMessage) and m.content == followup for m in messages)
        == 1
    )
    assert (
        sum(
            isinstance(m, HumanMessage) and m.content == "狐狸型是什么？"
            for m in messages
        )
        == 1
    )
    assert [m.tool_call_id for m in messages if isinstance(m, ToolMessage)] == [
        "round1-search",
        "round1-read",
    ]
    # 追问时模型真正收到旧历史和新问题，不只检查最后保存的状态。
    followup_input = model.invoke.call_args_list[3].args[0]
    assert followup_input[1:-1] == first_messages
    assert isinstance(followup_input[-1], HumanMessage)
    assert followup_input[-1].content == followup
    assert second["model_calls"] == 1  # 本轮计数重置，不累计成4
    assert graph.get_state(config).values["messages"] == messages


def test_different_threads_have_isolated_snapshots_and_model_inputs():
    from unittest.mock import Mock

    from langgraph.checkpoint.memory import InMemorySaver

    model = Mock()
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_fixture",
                    "args": {"query": "狐狸"},
                    "id": "thread-a-search",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="会话A回答"),
        AIMessage(content="会话B回答"),
    ]
    graph = build_graph(
        model,
        [search_fixture],
        checkpointer=InMemorySaver(),
        system_prompt=TEST_SYSTEM_PROMPT,
    )
    config_a = {"configurable": {"thread_id": "isolation-a"}}
    config_b = {"configurable": {"thread_id": "isolation-b"}}
    result_a = graph.invoke(
        {"messages": [HumanMessage(content="A独有问题：狐狸")], "model_calls": 0},
        config=config_a,
    )
    result_b = graph.invoke(
        {"messages": [HumanMessage(content="B独有问题：你好")], "model_calls": 0},
        config=config_b,
    )
    snapshot_a = graph.get_state(config_a).values
    snapshot_b = graph.get_state(config_b).values
    assert snapshot_a["messages"] == result_a["messages"]
    assert snapshot_b["messages"] == result_b["messages"]
    assert [m.type for m in snapshot_a["messages"]] == ["human", "ai", "tool", "ai"]
    assert [m.type for m in snapshot_b["messages"]] == ["human", "ai"]
    assert snapshot_a["model_calls"] == 2
    assert snapshot_b["model_calls"] == 1
    ids_a = {m.id for m in snapshot_a["messages"]}
    ids_b = {m.id for m in snapshot_b["messages"]}
    assert ids_a.isdisjoint(ids_b)
    assert all("B独有问题" not in m.content for m in snapshot_a["messages"])
    assert all("A独有问题" not in m.content for m in snapshot_b["messages"])
    assert not any(isinstance(m, ToolMessage) for m in snapshot_b["messages"])
    # B的模型输入只有系统提示与B的问题，不包含A的问题、回答或工具历史。
    input_b = model.invoke.call_args_list[2].args[0]
    assert len(input_b) == 2
    assert input_b[0].type == "system"
    assert input_b[1].content == "B独有问题：你好"
    assert model.invoke.call_count == 3


def test_new_in_memory_saver_has_no_previous_thread_state():
    from unittest.mock import Mock

    from langgraph.checkpoint.memory import InMemorySaver

    config = {"configurable": {"thread_id": "same-id-new-saver"}}
    old_model = Mock()
    old_model.invoke.return_value = AIMessage(content="旧会话回答")
    old_saver = InMemorySaver()
    old_graph = build_graph(
        old_model,
        [search_fixture],
        checkpointer=old_saver,
        system_prompt=TEST_SYSTEM_PROMPT,
    )
    old_result = old_graph.invoke(
        {"messages": [HumanMessage(content="旧会话独有问题")], "model_calls": 0},
        config=config,
    )
    assert old_graph.get_state(config).values["messages"] == old_result["messages"]

    new_model = Mock()
    new_model.invoke.return_value = AIMessage(content="新会话回答")
    new_saver = InMemorySaver()
    new_graph = build_graph(
        new_model,
        [search_fixture],
        checkpointer=new_saver,
        system_prompt=TEST_SYSTEM_PROMPT,
    )
    # 同一个thread ID，并不能让一个新Saver读到旧Saver的内存。
    assert new_graph.get_state(config).values == {}
    new_result = new_graph.invoke(
        {"messages": [HumanMessage(content="新会话独有问题")], "model_calls": 0},
        config=config,
    )
    assert [m.content for m in new_result["messages"]] == [
        "新会话独有问题",
        "新会话回答",
    ]
    model_input = new_model.invoke.call_args.args[0]
    assert len(model_input) == 2
    assert model_input[-1].content == "新会话独有问题"
    assert new_graph.get_state(config).values["model_calls"] == 1
    # 新Saver的执行也不能改变仍存活的旧Saver。
    assert old_graph.get_state(config).values["messages"] == old_result["messages"]


def test_endless_tool_requests_raise_recursion_limit_error():
    from unittest.mock import Mock

    import pytest
    from langgraph.errors import GraphRecursionError

    model = Mock()
    request_count = 0

    def keep_requesting(messages):
        nonlocal request_count
        request_count += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_fixture",
                    "args": {"query": "狐狸"},
                    "id": f"endless-{request_count}",
                    "type": "tool_call",
                }
            ],
        )

    model.invoke.side_effect = keep_requesting
    graph = build_graph(model, [search_fixture], system_prompt=TEST_SYSTEM_PROMPT)
    # invoke必须报错，不能把最后一次工具请求作为正常最终回答返回。
    with pytest.raises(GraphRecursionError):
        graph.invoke(
            {"messages": [HumanMessage(content="持续查找")], "model_calls": 0},
            config={"recursion_limit": 4},
        )
    assert model.invoke.call_count == 2
    assert request_count == 2
    # 第二次模型调用已经收到工具结果，确认真实走过 model→tools→model。
    second_input = model.invoke.call_args_list[1].args[0]
    assert any(isinstance(m, ToolMessage) for m in second_input)


def test_tool_technical_error_propagates_instead_of_empty_success():
    from unittest.mock import Mock

    import pytest

    failure = RuntimeError("fixture数据库连接失败")
    executed = Mock()

    @tool
    def failing_fixture(query: str) -> str:
        """Raise a technical failure to verify propagation."""
        executed(query)
        raise failure

    model = Mock()
    model.invoke.return_value = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "failing_fixture",
                "args": {"query": "狐狸"},
                "id": "failing-call",
                "type": "tool_call",
            }
        ],
    )
    graph = build_graph(model, [failing_fixture], system_prompt=TEST_SYSTEM_PROMPT)
    with pytest.raises(RuntimeError, match="fixture数据库连接失败") as caught:
        graph.invoke({"messages": [HumanMessage(content="查资料")], "model_calls": 0})
    assert caught.value is failure
    executed.assert_called_once_with("狐狸")
    # 工具失败后不再调用模型生成所谓正常答案。
    model.invoke.assert_called_once()


def test_read_count_counts_successes_across_batches_and_resets_next_turn():
    import json
    from unittest.mock import Mock

    from langgraph.checkpoint.memory import InMemorySaver

    @tool
    def read_chunk(chunk_id: int) -> str:
        """Return a successful read or a normal not_found result."""
        return json.dumps({"status": "not_found" if chunk_id == 99 else "found"})

    def request(chunk_id, call_id):
        return {
            "name": "read_chunk",
            "args": {"chunk_id": chunk_id},
            "id": call_id,
            "type": "tool_call",
        }

    model = Mock()
    model.invoke.side_effect = [
        # 同批两次成功和一次not_found，必须加2而非加1或加3。
        AIMessage(
            content="",
            tool_calls=[
                request(1, "read-one"),
                request(2, "read-two"),
                request(99, "missing-one"),
            ],
        ),
        AIMessage(content="", tool_calls=[request(3, "read-three")]),
        AIMessage(content="第一轮完成"),
        AIMessage(content="", tool_calls=[request(4, "next-read")]),
        AIMessage(content="第二轮完成"),
    ]
    graph = build_graph(
        model,
        [read_chunk],
        checkpointer=InMemorySaver(),
        system_prompt=TEST_SYSTEM_PROMPT,
    )
    config = {"configurable": {"thread_id": "read-count-rounds"}}
    updates = list(
        graph.stream(
            {
                "messages": [HumanMessage(content="第一轮")],
                "model_calls": 0,
                "read_count": 0,
            },
            config=config,
            stream_mode="updates",
        )
    )
    tool_updates = [update["tools"] for update in updates if "tools" in update]
    assert [delta["read_count"] for delta in tool_updates] == [2, 3]
    first = graph.get_state(config).values
    assert first["read_count"] == 3
    replies = [m for m in first["messages"] if isinstance(m, ToolMessage)]
    assert {m.tool_call_id: json.loads(m.content)["status"] for m in replies} == {
        "read-one": "found",
        "read-two": "found",
        "missing-one": "not_found",
        "read-three": "found",
    }
    second = graph.invoke(
        {
            "messages": [HumanMessage(content="第二轮")],
            "model_calls": 0,
            "read_count": 0,
        },
        config=config,
    )
    assert second["read_count"] == 1
    assert second["messages"][: len(first["messages"])] == first["messages"]


def test_read_count_does_not_count_search_or_not_found():
    import json
    from unittest.mock import Mock

    @tool
    def read_chunk(chunk_id: int) -> str:
        """Return a normal missing-source result."""
        return json.dumps({"status": "not_found"})

    @tool
    def search_count_fixture(query: str) -> str:
        """Return found search candidates, not a successful original read."""
        return json.dumps({"status": "found"})

    model = Mock()
    model.invoke.side_effect = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_count_fixture",
                    "args": {"query": "狐狸"},
                    "id": "search-only",
                    "type": "tool_call",
                },
                {
                    "name": "read_chunk",
                    "args": {"chunk_id": 99},
                    "id": "read-missing",
                    "type": "tool_call",
                },
            ],
        ),
        AIMessage(content="没有原文"),
    ]
    graph = build_graph(
        model, [search_count_fixture, read_chunk], system_prompt=TEST_SYSTEM_PROMPT
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="查询")], "model_calls": 0, "read_count": 0}
    )
    assert result["read_count"] == 0
    assert {
        m.tool_call_id for m in result["messages"] if isinstance(m, ToolMessage)
    } == {"search-only", "read-missing"}
