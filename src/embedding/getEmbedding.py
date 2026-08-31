import logging

import dashscope
from dashscope import TextEmbedding

from src.config.config import EMBEDDING_MODEL, get_dashscope_api_key

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
    try:
        dashscope.api_key = get_dashscope_api_key()

        response = TextEmbedding.call(model=EMBEDDING_MODEL, input=text)

        if response.status_code == 200:
            return response.output["embeddings"][0]["embedding"]
        else:
            logger.info(
                f"⚠️  Embedding API 调用失败：{response.code} - {response.message}"
            )
            return None
    except Exception:
        logger.exception("⚠️  Embedding 调用异常：")
        return None
