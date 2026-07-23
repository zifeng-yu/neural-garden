from config import API_KEY, EMBEDDING_MODEL


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

        dashscope.api_key = API_KEY

        response = dashscope.TextEmbedding.call(model=EMBEDDING_MODEL, input=text)

        if response.status_code == 200:
            return response.output["embeddings"][0]["embedding"]
        else:
            print(f"⚠️  Embedding API 调用失败：{response.code} - {response.message}")
            return None
    except Exception as e:
        print(f"⚠️  Embedding 调用异常：{e}")
        return None
