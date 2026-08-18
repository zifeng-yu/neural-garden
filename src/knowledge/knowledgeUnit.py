import logging
from dataclasses import asdict, dataclass

import src.config.logging_config as logging_config
from src.knowledge.knowledge_extractor import Response_extractor, extractor_by_llm

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeUnit:
    """知识单元数据结构"""

    title: str
    summary: str
    keywords: list[str]
    content: str
    id: str

    def to_dict(self) -> dict:
        """转换为字典（用于存储）"""
        return asdict(self)

    def to_embedding_text(self) -> str:
        """生成用于向量化的文本（标题 + 摘要 + 关键词）"""
        return f"{self.title}。{self.summary}。关键词：{', '.join(self.keywords)}"


def extract_knowledge_unit(content: str, id: str) -> KnowledgeUnit | None:
    try:
        response_extractor: Response_extractor = extractor_by_llm(content)
        return KnowledgeUnit(
            title=response_extractor.title,
            summary=response_extractor.summary,
            keywords=response_extractor.keywords,
            content=content,
            id=id,
        )
    except Exception:
        logger.exception("获取知识单元失败，id =%s , 失败原因 ", id)
    return None
