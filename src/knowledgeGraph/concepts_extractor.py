import logging

import src.config.logging_config as logging_config
from src.util.callDashscopellm import generate as fetchLLM

logger = logging.getLogger(__name__)


def extract_concepts_from_text(text: str) -> list[str]:
    """
    从文本中提取概念
    Args:
        text:输入文本
    Returns:
        概念列表
    """
    sys_prompt = """
你是一个知识图谱构建助手。

任务：
从文本中提取用于构建知识图谱的核心概念。

提取规则：
1. 只提取名词或名词短语，不包含修饰词。
2. 概念必须具有独立语义，可以作为知识图谱节点。
3. 优先提取：
   - 技术术语
   - 理论/方法
   - 政策
   - 机构
   - 产品
   - 事件
   - 专业领域实体
4. 删除：
   - 普通描述词
   - 时间、地点（除非是重要实体）
   - 完整句子
   - 泛化词语
5. 同义概念只保留一个标准名称。
6. 最多输出10个概念，并按重要程度排序。
7. 不输出事件描述

只输出 JSON 数组，不要输出任何解释。

输出格式：
[
  "概念1",
  "概念2",
  "概念3"
]
"""
    return fetchLLM(
        sys_prompt_content=sys_prompt,
        user_prompt_content="文本内容:" + text,
        response_json=True,
    )


def extract_max_similarity_concept(
    concept: tuple[str, list[str]], similarity_concepts: list[tuple[str, list[str]]]
) -> list[str]:
    """提取最相似的概念"""
    sys_prompt = """
你是一个知识图谱概念归一化专家。

你的任务：
判断一个新概念是否与候选概念列表中的某个概念表示同一个实体或同一个知识概念。

判断规则：
1. 如果两个概念只是表达方式不同、简称、全称、同义词，则认为相同。
2. 如果两个概念存在上下位关系、相关关系，但不是同一个概念，则认为不同。
3. 不要因为语义相关就合并。
4. 不允许将父概念和子概念合并。例如：算法 ≠ 排序算法 数据结构 ≠ HashSet 复杂度 ≠ 时间复杂度 复杂度 ≠ 空间复杂度
5. 判断是否同一实体，不是是否属于同一主题。

例如：
"央行" 和 "中央银行" -> 相同
"负利率" 和 "负利率政策" -> 可能相同
"央行" 和 "货币政策" -> 不同
"苹果公司" 和 "苹果" -> 不同

输入：包含了新概念、新概念来源的文本、候选概念、候选概念来源的文本。来源文本可能有多个。
格式：
[<新概念>央行</新概念><来源文本>来源文本内容...</来源文本><来源文本>来源文本内容...</来源文本>]
[<候选概念>中央银行</候选概念><来源文本>来源文本内容...</来源文本><来源文本>来源文本内容...</来源文本>]
[<候选概念>负利率政策</候选概念><来源文本>来源文本内容...</来源文本><来源文本>来源文本内容...</来源文本>]


输出要求：
只输出 JSON 数组。
如果找到相同概念，返回候选列表中的原始概念名称，如果有找到多个相同概念，最相同的排在返回的数组的第一个位置上。
如果没有相同概念，返回空数组。

格式：
["概念"]
或者：
[]
"""
    user_prompt = f"[<新概念>{concept[0]}</新概念>"
    for chunk_content in concept[1]:
        user_prompt += f"<来源文本>{chunk_content}</来源文本>"
    user_prompt += "]"

    for s_concept, s_concept_chunk_list in similarity_concepts:
        user_prompt += f"[<候选概念>{s_concept}</候选概念>"
        for s_chunk_content in s_concept_chunk_list:
            user_prompt += f"<来源文本>{s_chunk_content}</来源文本>"
        user_prompt += "]"
    return fetchLLM(
        sys_prompt_content=sys_prompt,
        user_prompt_content=user_prompt,
        response_json=True,
    )
