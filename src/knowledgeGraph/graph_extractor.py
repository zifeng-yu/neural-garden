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


def extract_relations_from_text(text: str, concepts: list[str]) -> list[list]:
    """
    从文本中提取概念之间的关系
    Args:
        text:输入文本
        concepts:概念列表
    Returns:
        关系列表 [(概念1,关系类型,概念2)...]
    """
    sys_prompt = """
你是一个知识图谱关系抽取助手。

任务：
根据文本和已有概念列表，提取概念之间的有向关系。

输出格式：
只输出 JSON 数组：

[
 ["source概念", "relation关系", "target概念"]
]

规则：

1. 每条关系表示一个有向边：
source概念 -> target概念

2. source 是动作发起者，target 是关系承受者。
保持语义方向，不要反转。

例如：
文本：
"央行通过量化宽松政策刺激经济"

正确：
[
 ["央行","实施","量化宽松"],
 ["量化宽松","影响","经济"]
]

错误：
[
 ["量化宽松","实施","央行"]
]

3. relation 使用简洁明确的关系类型，例如：
- 属于
- 包含
- 实施
- 使用
- 影响
- 导致
- 依赖
- 替代
- 产生

4. 只抽取文本明确表达的关系。
不要基于常识补充。

5. source 和 target 必须来自给定概念列表。
不要创造新的概念。

6. 如果没有发现关系，返回：

[]

只输出 JSON，不要输出解释。
"""
    user_prompt = f"<文本>{text}</文本>\n<概念>{concepts}</概念>"
    return fetchLLM(
        sys_prompt_content=sys_prompt,
        user_prompt_content=user_prompt,
        response_json=True,
    )
