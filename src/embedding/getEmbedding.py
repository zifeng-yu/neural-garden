import logging

import src.config.logging_config as logging_config
from src.config.config import API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def get_embedding(text: str) -> list:
    """
    调用 DashScope API 获取文本向量

    Args:
        text: 输入文本
        api_key: DashScope API Key

    Returns:
        向量列表（float）
    """
    try:
        import dashscope
        from dashscope import TextEmbedding

        dashscope.api_key = API_KEY

        response = TextEmbedding.call(model=EMBEDDING_MODEL, input=text)

        if response.status_code == 200:
            return response.output["embeddings"][0]["embedding"]
        else:
            logger.info(
                f"⚠️  Embedding API 调用失败：{response.code} - {response.message}"
            )
            return None
    except Exception as e:
        logger.info(f"⚠️  Embedding 调用异常：{e}")
        return None
