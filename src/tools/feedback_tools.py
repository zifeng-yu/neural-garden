"""三个统计工具；负责日期输入校验与响应组装，SQL放在统计模块。"""

from datetime import date
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.feedback_events_analytics.feedback_analytics import (
    get_popular_documents,
    get_popular_queries,
    get_zero_result_queries,
)


class UsageSummaryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_date: str = Field(strict=True, description="起始日期，含当天，YYYY-MM-DD")
    end_date_exclusive: str = Field(
        strict=True, description="结束日期，不含当天，YYYY-MM-DD"
    )

    @model_validator(mode="after")
    def check_dates(self) -> "UsageSummaryInput":
        try:
            start = date.fromisoformat(self.start_date)
            end = date.fromisoformat(self.end_date_exclusive)
        except ValueError as exc:
            raise ValueError("日期必须是有效的 YYYY-MM-DD 日期") from exc
        if (
            start.isoformat() != self.start_date
            or end.isoformat() != self.end_date_exclusive
        ):
            raise ValueError("日期必须使用规范的 YYYY-MM-DD 格式")
        if start >= end:
            raise ValueError("start_date 必须早于 end_date_exclusive")
        return self


@tool(args_schema=UsageSummaryInput)
def query_popular_queries(start_date: str, end_date_exclusive: str) -> dict[str, Any]:
    """查询热门搜索词Top5及搜索记录次数。
    按上海日期，起始含当天、结束不含当天；日期范围须明确。
    返回的是分组次数，不是人数或所有搜索记录的总数。
    """
    rows = get_popular_queries(start_date, end_date_exclusive)
    return {
        "status": "ok",
        "metric": "popular_queries",
        "start_date": start_date,
        "end_date_exclusive": end_date_exclusive,
        "timezone": "Asia/Shanghai",
        "limit": 5,
        "items": [{"key": row["query"], "count": row["search_count"]} for row in rows],
    }


@tool(args_schema=UsageSummaryInput)
def query_zero_result_queries(
    start_date: str, end_date_exclusive: str
) -> dict[str, Any]:
    """查询结果数为零的搜索词Top5及相应搜索记录次数。
    按上海日期，起始含当天、结束不含当天；日期范围须明确。
    零结果记录不能证明检索失败、缺少资料或其原因。
    """
    rows = get_zero_result_queries(start_date, end_date_exclusive)
    return {
        "status": "ok",
        "metric": "zero_result_queries",
        "start_date": start_date,
        "end_date_exclusive": end_date_exclusive,
        "timezone": "Asia/Shanghai",
        "limit": 5,
        "items": [{"key": row["query"], "count": row["search_count"]} for row in rows],
    }


@tool(args_schema=UsageSummaryInput)
def query_popular_documents(start_date: str, end_date_exclusive: str) -> dict[str, Any]:
    """查询文档点击Top5，返回文档ID及点击事件次数。
    按上海日期，起始含当天、结束不含当天；日期范围须明确。
    排除缺失文档ID；不是点击率或人数，不能根据ID编造标题。
    """
    rows = get_popular_documents(start_date, end_date_exclusive)
    return {
        "status": "ok",
        "metric": "popular_documents",
        "start_date": start_date,
        "end_date_exclusive": end_date_exclusive,
        "timezone": "Asia/Shanghai",
        "limit": 5,
        "items": [
            {"key": row["document_id"], "count": row["click_count"]} for row in rows
        ],
    }


def create_usage_tools() -> list[BaseTool]:
    return [query_popular_queries, query_zero_result_queries, query_popular_documents]
