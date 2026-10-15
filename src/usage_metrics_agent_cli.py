"""使用指标Agent：查询热门搜索、零结果搜索及文档点击统计。"""

import logging

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from src.agent_graph.agent_graph import build_graph
from src.config.config import (
    AGENT_ENABLE_THINKING,
    AGENT_MAX_RETRIES,
    AGENT_RECURSION_LIMIT,
    AGENT_REQUEST_TIMEOUT,
    AGENT_TEMPERATURE,
    DASHSCOPE_BASE_URL,
    LLM_MODEL,
    SQLITE_TABLE,
    get_dashscope_api_key,
)
from src.config.logging_config import setup_logging
from src.tools.feedback_tools import create_usage_tools

logger = logging.getLogger(__name__)

SAGE_SYSTEM_PROMPT = """你是Neural Garden的使用与反馈查询助手。
你支持三个指标：热门搜索词popular_queries、结果数为零的搜索词zero_result_queries、
文档点击次数popular_documents。每次返回最多5个分组，不是所有记录的总数。
热门搜索调用query_popular_queries，零结果搜索调用query_zero_result_queries，
文档点击调用query_popular_documents；依据本次匹配问题的成功结果回答。
指标和完整日期范围须来自用户明确表述，或同会话中明确无歧义的信息。
新会话只说“看看统计”而没有指标或日期时，先问缺少的内容；不要擅自默认指标或日期。
日期按上海时区；start_date包含当天，end_date_exclusive不包含当天。
10月1日至7日对应结束日期10月8日；回答写10月1日至7日（含），或明确左闭右开区间。
同会话切换指标时可沿用已明确日期，但必须查询新指标，不能挪用上一指标的结果。
items为空表示该范围内没有符合口径的记录；工具失败不能称为零记录。
点击统计是事件次数，不是点击率或人数；零结果记录不能证明检索失败或缺少资料。
只有文档ID时直接称文档ID，不编造标题；不能推断热门或零结果的原因。
不支持CTR、准确率、增长原因等问题时说明缺少相应能力或依据。
本入口不生成知识库引用编号[D#:C#]，依据是匹配问题的统计工具结果。
"""


def create_app():
    model = ChatOpenAI(
        model=LLM_MODEL,
        api_key=SecretStr(get_dashscope_api_key()),
        base_url=DASHSCOPE_BASE_URL,
        temperature=AGENT_TEMPERATURE,
        timeout=AGENT_REQUEST_TIMEOUT,
        max_retries=AGENT_MAX_RETRIES,
        extra_body={"enable_thinking": AGENT_ENABLE_THINKING},
    )

    tools = create_usage_tools()

    return build_graph(
        model.bind_tools(tools),
        tools,
        checkpointer=InMemorySaver(),
        system_prompt=SAGE_SYSTEM_PROMPT,
    )


if __name__ == "__main__":
    setup_logging()
    import argparse
    import json

    from langchain_core.messages import HumanMessage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", type=str, help="首问内容")
    parser.add_argument("--followup", default=None, help="同会话追问")
    parser.add_argument("--thread-id", default="ng-usage-1")
    args = parser.parse_args()

    app = create_app()

    from langchain_core.runnables import RunnableConfig

    config = RunnableConfig(
        configurable={"thread_id": args.thread_id},
        recursion_limit=AGENT_RECURSION_LIMIT,
    )

    logger.info(
        json.dumps(
            {
                "model": LLM_MODEL,
                "temperature": AGENT_TEMPERATURE,
                "thread_id": args.thread_id,
                "db_path": SQLITE_TABLE,
            },
            ensure_ascii=False,
        )
    )

    def run_turn(question: str):
        question_message = HumanMessage(content=question)
        logger.info(question_message.model_dump_json())

        for update in app.stream(
            {
                "messages": [question_message],
                "model_calls": 0,
                "read_count": 0,
            },
            config=config,
            stream_mode="updates",
        ):
            for node_name, delta in update.items():
                logger.info(node_name)
                for message in delta.get("messages", []):
                    logger.info(message.model_dump_json())

        snapshot = app.get_state(config)
        logger.info(snapshot.values["messages"][-1].content)

    run_turn(args.question)

    if args.followup is not None:
        run_turn(args.followup)
