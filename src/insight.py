"""
Insight 记录模块

支持：
- Insight 结构化存储
- 关联概念提取
- 时间戳记录
- Markdown 导出
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional
import hashlib


@dataclass
class Insight:
    """Insight 数据结构"""
    
    id: str  # 唯一标识（时间戳 + 哈希）
    title: str  # 一句话总结
    trigger_content: str  # 触发内容
    source: str  # 来源
    content: str  # 洞察内容
    related_concepts: List[str]  # 关联概念
    action_items: List[str]  # 行动建议
    created_at: str  # 记录时间
    
    def to_dict(self) -> dict:
        """转换为字典（用于存储）"""
        return asdict(self)
    
    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        md = f"## 💡 Insight: {self.title}\n\n"
        md += f"**触发内容**：{self.trigger_content}\n"
        md += f"**来源**：{self.source}\n\n"
        md += f"**洞察内容**：\n{self.content}\n\n"
        md += "**关联概念**：\n"
        for concept in self.related_concepts:
            md += f"- {concept}\n"
        md += "\n**行动建议**：\n"
        for action in self.action_items:
            md += f"- {action}\n"
        md += f"\n**记录时间**：{self.created_at}\n"
        return md


def generate_insight_id(content: str) -> str:
    """
    生成 Insight 唯一 ID
    
    格式：时间戳_内容哈希前 8 位
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{timestamp}_{content_hash}"


def create_insight(
    title: str,
    trigger_content: str,
    source: str,
    content: str,
    related_concepts: List[str] = None,
    action_items: List[str] = None,
) -> Insight:
    """
    创建 Insight 记录
    
    Args:
        title: 一句话总结
        trigger_content: 触发内容
        source: 来源
        content: 洞察内容
        related_concepts: 关联概念列表
        action_items: 行动建议列表
        
    Returns:
        Insight 对象
    """
    if related_concepts is None:
        related_concepts = []
    if action_items is None:
        action_items = []
    
    return Insight(
        id=generate_insight_id(content),
        title=title,
        trigger_content=trigger_content,
        source=source,
        content=content,
        related_concepts=related_concepts,
        action_items=action_items,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


# 示例用法
if __name__ == "__main__":
    insight = create_insight(
        title="负利率是对储蓄的惩罚",
        trigger_content="日本负利率政策的底层逻辑",
        source="01-金融深潜 - 日本负利率.md",
        content="负利率不是'奖励借钱'，而是'惩罚存钱'。央行通过让存款贬值，逼着钱流动起来，进入投资和消费。这解释了为什么负利率下股市反而上涨——钱没地方去，只能进股市。",
        related_concepts=["负利率", "货币政策", "资产价格"],
        action_items=["重新审视自己的资产配置", "研究负利率环境下的投资策略"],
    )
    
    print(insight.to_markdown())
