import logging

import dashscope
from dashscope import TextEmbedding

from src.config.config import EMBEDDING_MODEL, get_dashscope_api_key
from src.util.index_logging import index_stage

logger = logging.getLogger(__name__)


def get_embedding(text: str) -> list[float] | None:
    """
    调用 DashScope API 获取文本向量

    Args:
        text: 输入文本
        api_key: DashScope API Key

    Returns:
        向量列表（float）
    """
    with index_stage(
        logger, "embedding", model=EMBEDDING_MODEL, chars=len(text)
    ) as stats:
        try:
            dashscope.api_key = get_dashscope_api_key()

            response = TextEmbedding.call(model=EMBEDDING_MODEL, input=text)

            stats["http_status"] = response.status_code
            if response.status_code == 200:
                vector = response.output["embeddings"][0]["embedding"]
                stats["dimensions"] = len(vector)
                return vector
            else:
                stats["outcome"] = "failed"
                logger.info(
                    f"⚠️  Embedding API 调用失败：{response.code} - {response.message}"
                )
                return None
        except Exception as exc:
            stats.update(outcome="failed", error_type=type(exc).__name__)
            logger.exception("⚠️  Embedding 调用异常：")
            return None
