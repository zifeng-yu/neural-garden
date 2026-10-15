import json

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


class AgentState(MessagesState):
    model_calls: int
    read_count: int


def build_graph(model_with_tools, tools, *, system_prompt: str, checkpointer=None):
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt必须是非空字符串")

    def model_node(state: AgentState):
        # 1. 调用model_with_tools.invoke，输入为SystemMessage加state中的全部消息。
        reply = model_with_tools.invoke(
            [
                SystemMessage(content=system_prompt),
                *state["messages"],
            ]
        )
        # 2. 校验返回是AIMessage；畸形tool_calls、截断、空回复不能算正常完成。
        if not isinstance(reply, AIMessage):
            raise ValueError(f"Expected AIMessage, got {type(reply)}")
        if reply.invalid_tool_calls:
            raise ValueError("模型工具调用参数解析失败")
        # 空回复
        content = reply.content
        if isinstance(content, str) and not content.strip() and not reply.tool_calls:
            raise ValueError("Empty model reply with no tool_calls")
        if isinstance(content, list) and len(content) == 0 and not reply.tool_calls:
            raise ValueError("Empty model reply with no tool_calls")
        # 畸形 / 截断的 tool_calls（常见表现：缺少 name 或 id，或 args 不是 dict）
        for tc in reply.tool_calls or []:
            if not isinstance(tc, dict):
                raise ValueError(f"Malformed tool_call (not dict): {tc}")
            if not tc.get("name") or not tc.get("id"):
                raise ValueError(f"Malformed tool_call missing name/id: {tc}")
            if "args" in tc and not isinstance(tc["args"], dict):
                raise ValueError(f"Malformed tool_call args (not dict): {tc}")
        # response_metadata 里常见的截断标志
        meta = getattr(reply, "response_metadata", None) or {}
        finish_reason = meta.get("finish_reason") or meta.get("stop_reason")
        if finish_reason in ("length", "max_tokens", "content_filter"):
            raise ValueError(f"模型生成未正常完成：{finish_reason}")
        # 3. 仅返moch回新增AIMessage，以及加1后的model_calls。
        return {
            "messages": [reply],
            "model_calls": state.get("model_calls", 0) + 1,
        }

    def route(state: AgentState):
        # 最后一条 AIMessage 有 tool_calls 就返回 "tools"，否则返回 END
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    # 接线约定：START → model → (有 tool_calls ? tools : END) → model
    builder = StateGraph(AgentState)
    builder.add_node("model", model_node)

    tool_node = ToolNode(tools, handle_tool_errors=False)

    def tools_node(state: AgentState):
        delta = tool_node.invoke(state)
        success_count = 0
        for message in delta.get("messages", []):
            if message.name == "read_chunk":
                data = json.loads(message.content)
                if data.get("status") == "found":
                    success_count += 1

        delta["read_count"] = state.get("read_count", 0) + success_count
        return delta

    builder.add_node("tools", tools_node)

    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")  # 工具执行完后回到模型

    return builder.compile(checkpointer=checkpointer)
