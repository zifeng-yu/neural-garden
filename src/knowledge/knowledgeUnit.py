import hashlib
import logging
from dataclasses import dataclass

import src.config.logging_config as logging_config
from src.knowledge.knowledge_extractor import Response_extractor, extractor_by_llm

logger = logging.getLogger(__name__)


@dataclass
class Knowledge_unit:
    """知识单元数据结构"""

    title: str
    summary: str
    keywords: list[str]
    content: str
    source: str
    unit_id: str

    def to_embedding_text(self) -> str:
        """生成用于向量化的文本（标题 + 摘要 + 关键词）"""
        return f"{self.title}。{self.summary}。关键词：{', '.join(self.keywords)}"


def get_knowledge_unit(content: str, source: str) -> Knowledge_unit | None:
    try:
        response_extractor: Response_extractor = extractor_by_llm(content)
        return Knowledge_unit(
            title=response_extractor.title,
            summary=response_extractor.summary,
            keywords=response_extractor.keywords,
            content=content,
            source=source,
            unit_id=hashlib.md5(
                f"{source}:{response_extractor.title}".encode()
            ).hexdigest()[:16],
        )
    except Exception as e:
        logger.exception(f"获取知识单元失败，失败原因 {e}")
        logger.error(f"获取知识单元失败，文章为：{source}")
    return None
