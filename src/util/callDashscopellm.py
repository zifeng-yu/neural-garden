"""
DashScope LLM 调用封装（带重试机制）

支持：
- 自动重试（指数退避）
- 超时控制
- 异常分类（可重试 vs 不可重试）
- 详细日志
"""

import json
import logging
from typing import Any, Optional

import dashscope
from dashscope import MultiModalConversation

from src.config.config import API_KEY, LLM_MODEL
from src.util.llmException import LLMException
from src.util.retryUtil import retry

logger = logging.getLogger(__name__)


# 定义可重试的异常类型
# dashscope 不直接导出异常类，我们通过状态码判断
RETRYABLE_EXCEPTIONS = (LLMException,)


def _should_retry_status(status_code: int) -> bool:
    """
    判断 HTTP 状态码是否应该重试

    可重试：
    - 5xx: 服务端错误
    - 429: 速率限制

    不可重试：
    - 4xx (除 429): 客户端错误
    """
    return bool(status_code >= 500 or status_code == 429)


def _on_retry(retry_count: int, exception: Exception, next_delay: float):
    """重试回调函数"""
    logger.debug(
        f"[LLM 重试] 第 {retry_count} 次失败，"
        f"{next_delay:.1f}s 后重试：{type(exception).__name__}"
    )


def _on_success(result: Any, total_time: float):
    """成功回调函数"""
    logger.debug(f"[LLM 调用] 成功，总耗时 {total_time:.2f}s")


# 重试装饰器配置
# 重试 3 次，初始延迟 1 秒，退避倍数 2.0，超时 30 秒
_llm_retry = retry(
    times=3,
    delay=1.0,
    backoff=2.0,
    exceptions=RETRYABLE_EXCEPTIONS,
    timeout=30.0,
    retry_callback=_on_retry,
    success_callback=_on_success,
)


@_llm_retry
def generate(
    sys_prompt_content: str,
    user_prompt_content: str,
    temperature: float = 0.1,
    enable_thinking: bool = False,
    response_json: bool = False,
    response_class: Optional[type] = None,
    timeout: Optional[float] = None,
) -> Any:
    """
    调用 DashScope LLM 生成响应（带重试机制）

    Args:
        sys_prompt_content: 系统提示词
        user_prompt_content: 用户提示词
        temperature: 温度参数（0.0-2.0，越低越确定）
        enable_thinking: 是否启用思考模式
        response_json: 是否要求 JSON 格式响应
        response_class: 如果指定，将 JSON 解析为该类
        timeout: 单次调用超时秒数（由装饰器自动注入，默认 30s）

    Returns:
        LLM 响应：
        - response_json=False: 返回原始文本
        - response_json=True: 返回解析后的 dict/list
        - response_class 指定：返回该类的实例

    Raises:
        LLMException: 重试 3 次后仍失败

    示例:
        # 简单调用
        text = generate("你是一个助手", "你好")

        # JSON 模式
        data = generate(
            "提取实体",
            "北京是中国首都",
            response_json=True
        )

        # Pydantic 模式
        from pydantic import BaseModel
        class Entity(BaseModel):
            location: str
            country: str

        entity = generate(
            "提取实体",
            "北京是中国首都",
            response_json=True,
            response_class=Entity
        )
    """
    # 构建消息
    prompt = [
        {"role": "system", "content": sys_prompt_content},
        {"role": "user", "content": user_prompt_content},
    ]

    # 配置 API Key
    dashscope.api_key = API_KEY

    # 构建请求参数
    params = {
        "model": LLM_MODEL,
        "messages": prompt,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
    }

    # JSON 格式要求
    if response_json:
        params["response_format"] = {"type": "json_object"}

    # 超时参数（由 retry 装饰器注入）
    if timeout is not None:
        params["timeout"] = timeout

    logger.info(
        f"[LLM 调用] model={LLM_MODEL}, "
        f"sys_len={len(sys_prompt_content)}, "
        f"user_len={len(user_prompt_content)}, "
        f"timeout={timeout}s"
    )

    # 调用 API
    try:
        response = MultiModalConversation.call(**params)
    except Exception as e:
        # 网络层异常（连接错误、超时等）
        logger.warning(f"[LLM 调用] 网络层异常，将重试：{type(e).__name__}: {e}")
        raise LLMException(f"Network error: {e}") from e

    # 检查响应状态
    if response.status_code != 200:
        error_msg = f"{response.status_code}: {response.message}"

        if _should_retry_status(response.status_code):
            # 可重试错误（5xx, 429）
            logger.warning(
                f"[LLM 调用] 服务端错误 ({response.status_code})，将重试：{response.message}"
            )
            raise LLMException(error_msg)
        else:
            # 不可重试错误（4xx 客户端错误）- 直接抛出，不进入重试循环
            logger.error(
                f"[LLM 调用] 客户端错误 ({response.status_code})，不重试：{response.message}"
            )
            # 使用 RuntimeError 而不是 LLMException，避免被 retry 装饰器捕获
            raise RuntimeError(
                f"Client error {response.status_code}: {response.message}"
            )

    # 解析响应
    try:
        text = response.output.choices[0].message.content[0]["text"]
    except (IndexError, KeyError, TypeError) as e:
        error_msg = f"响应格式异常：{response.output}"
        logger.error(f"[LLM 调用] {error_msg}")
        raise LLMException(error_msg) from e

    logger.debug(f"[LLM 调用] 响应长度：{len(text)}")

    # 非 JSON 模式，直接返回文本
    if not response_json:
        return text

    # JSON 模式，解析响应
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        error_msg = f"JSON 解析失败：{str(e)[:100]}"
        logger.error(f"[LLM 调用] {error_msg}")
        if response_json:
            # 要求 JSON 但解析失败，抛出异常
            raise LLMException(error_msg) from e
        else:
            return text

    # 如果指定了响应类，进行转换
    if response_class is not None:
        try:
            return response_class(**data)
        except (TypeError, AttributeError) as e:
            error_msg = f"响应类转换失败：{str(e)}"
            logger.error(f"[LLM 调用] {error_msg}")
            raise LLMException(error_msg) from e

    return data


def generate_streaming(
    sys_prompt_content: str,
    user_prompt_content: str,
    temperature: float = 0.1,
    enable_thinking: bool = False,
    timeout: Optional[float] = None,
):
    """
    流式调用 LLM（不支持重试，因为流式无法重放）

    Args:
        sys_prompt_content: 系统提示词
        user_prompt_content: 用户提示词
        temperature: 温度参数
        enable_thinking: 是否启用思考模式
        timeout: 超时秒数

    Yields:
        增量文本片段

    注意:
        流式调用不支持重试，因为无法重放已消耗的流
    """
    prompt = [
        {"role": "system", "content": sys_prompt_content},
        {"role": "user", "content": user_prompt_content},
    ]

    dashscope.api_key = API_KEY

    params = {
        "model": LLM_MODEL,
        "messages": prompt,
        "temperature": temperature,
        "enable_thinking": enable_thinking,
        "stream": True,
    }

    if timeout is not None:
        params["timeout"] = timeout

    response = MultiModalConversation.call(**params)

    for chunk in response:
        if chunk.output.choices:
            content = chunk.output.choices[0].message.content
            if content:
                yield content
