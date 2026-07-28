# 请求大模型，获得知识单元
import json
import logging

import dashscope
from dashscope import MultiModalConversation
from pydantic import BaseModel, Field

import src.config.logging_config as logging_config
from src.config.config import API_KEY, LLM_MODEL
from src.util.retryUtil import retry

logger = logging.getLogger(__name__)


class LLMException(Exception):
    pass


class Response_extractor(BaseModel):
    title: str = Field(min_length=1, max_length=30)
    summary: str = Field(min_length=1, max_length=500)
    keywords: list[str] = Field(min_length=0, max_length=10)


@retry(times=2)
def extractor_by_llm(content: str) -> Response_extractor:
    prompt = []
    system_prompt_content = """
    你是一位专业的知识结构化专家。请从文档内容中提取关键信息，输出 JSON 格式。
【提取要求】
1. title: 文档标题(20 字以内）
2. summary: 核心摘要(100-300 字，概括主要内容）
3. keywords: 3-10 个关键词（最能代表文档主题的词汇）
4. 如果遇到无效文章输入,title的值为"无效文章输入"
5. 提取后输出json的key:title,summary,keywords三者不可缺少
5. 不允许根据常识补充文档没有出现的信息
6. summary必须只基于原文,不得推测

【输出格式】
严格输出 JSON,不要任何其他文字,例子如下：
{
    "title": "标题",
    "summary": "摘要内容...",
    "keywords": ["关键词 1", "关键词 2", "关键词 3"]
}
    """
    system_prompt = {"role": "system", "content": f"{system_prompt_content}"}
    user_prompt = {"role": "user", "content": f"提取下面文章的知识结构:{content}"}
    prompt.append(system_prompt)
    prompt.append(user_prompt)
    dashscope.api_key = API_KEY
    # dashscope.base_http_api_url = (
    #     "https://ws-nbtll7yjauor7eld.cn-beijing.maas.aliyuncs.com/api/v1"
    # )
    # logger.info(prompt)
    response = MultiModalConversation.call(
        model=LLM_MODEL,
        messages=prompt,
        temperature=0.1,
        enable_thinking=False,
        response_format={"type": "json_object"},
    )
    if response.status_code != 200:
        raise LLMException(f"{response.status_code}:{response.message}")
    logger.info(response.output.choices[0].message.content[0]["text"])
    data = json.loads(response.output.choices[0].message.content[0]["text"])
    if data["title"] == "无效文章输入":
        raise LLMException(f"{data}")

    return Response_extractor(**data)
