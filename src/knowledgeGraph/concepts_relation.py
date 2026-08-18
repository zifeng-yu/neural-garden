import logging
from dataclasses import dataclass

from pydantic import BaseModel

import src.config.logging_config as logging_config
from src.util.callDashscopellm import generate as fetchLLM

logger = logging.getLogger(__name__)


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


@dataclass
class ConceptInChunkText:
    concept: str
    chunk_texts: list[str]


class ConceptClusters(BaseModel):
    canonical: str
    members: list[str]


class ConceptMergeResult(BaseModel):
    clusters: list[ConceptClusters]
    unmerged: list[str]


def extract_concept_clusters(
    conceptInChunkTextslist: list[ConceptInChunkText],
) -> ConceptMergeResult:
    sys_prompt = """
你是一个知识图谱专家。

任务：
判断下面概念是否表示同一个知识实体。

规则：
1. 如果只是名称不同、缩写不同、语言不同，可以合并。
2. 如果一个概念是另一个概念的上下位关系，不合并。
3. 如果主题相关但含义不同，不合并。
4. 在候选概念中选择一个更准确、更通用的概念名称。
5. 如果候选概念不能合并，原样返回即可。
6. 不得自己创造概念，必须在提供的概念中选择。
7. 输出内容必须包含输入的所有的概念，输入的上下文片段不用返回。
8. 输入的上下文片段若有多个会用标签分开。
9. 输出的canonical必须严格来自候选概念。
10. 如果两个概念存在因果关系、包含关系、领域关联关系，但不是同一个实体，不合并。
11. 不允许将父概念和子概念合并。例如：算法 ≠ 排序算法 数据结构 ≠ HashSet 复杂度 ≠ 时间复杂度 复杂度 ≠ 空间复杂度
12. 判断是否同一实体，不是是否属于同一主题。

提供的数据格式比如这样：

[<候选概念>日本负利率政策</候选概念>
<上下文片段>日本央行2016年推出负利率政策...</上下文片段><上下文片段>...</上下文片段>]
[<候选概念>日本央行负利率</候选概念>
<上下文片段>BOJ为了刺激经济实施负利率...</上下文片段>]
[<候选概念>BOJ负利率</候选概念>
<上下文片段>Bank of Japan negative interest rate...</上下文片段>]
[<候选概念>负利率时代</候选概念>
<上下文片段>全球进入低利率甚至负利率环境...</上下文片段>]


请输出JSON，
结构为{"clusters":{"canonical":"选出的概念","members":["候选概念1"]},"unmerged":["候选概念2"]}
解释为：clusters表示可以合并的概念，canonical表示为选出的更准确的更通用的概念名称，members表示可以合并的其余候选概念集合，unmerged表示为不可以合并的概念集合

具体JSON格式比如这样：
{
    "clusters": [
    {
        "canonical": "日本央行",
        "members": [
        "日本央行",
        "日本中央银行",
        "BOJ"
        ]
    }
    ],
    "unmerged": [
    "日本银行体系"
    ]
}
    """
    user_prompt = ""
    for conceptInChunkText in conceptInChunkTextslist:
        promot_in_data = f"[<候选概念>{conceptInChunkText.concept}</候选概念>"
        for chunk in conceptInChunkText.chunk_texts:
            promot_in_data += f"<来源文档>{chunk}</来源文档>"
        promot_in_data += "]"
        user_prompt += promot_in_data

    return fetchLLM(
        sys_prompt_content=sys_prompt,
        user_prompt_content=user_prompt,
        response_json=True,
        response_class=ConceptMergeResult,
    )
