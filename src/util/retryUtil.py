"""
重试工具模块（增强版）

支持功能：
- 可配置重试次数
- 指数退避延迟（Exponential Backoff）
- 超时控制
- 异常类型过滤（只重试特定异常）
- 详细日志记录
"""

import logging
import time
from functools import wraps
from typing import Optional

import src.config.logging_config as logging_config

logger = logging.getLogger(__name__)


def retry(
    times: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    timeout: Optional[float] = None,
    retry_callback=None,
    success_callback=None,
):
    """
    重试装饰器（支持指数退避、超时、异常过滤）

    Args:
        times: 重试次数（默认 3 次，即最多调用 times 次）
        delay: 初始延迟秒数（默认 1 秒）
        backoff: 退避倍数（默认 2，即延迟序列：1s, 2s, 4s, 8s...）
        exceptions: 需要重试的异常类型元组（默认所有异常）
        timeout: 单次调用超时秒数（None 表示不限制）
        retry_callback: 重试时的回调函数 (retry_count, exception, next_delay)
        success_callback: 成功时的回调函数 (result, total_time)

    Returns:
        被装饰函数的返回值

    Raises:
        最后一次重试仍失败时，抛出原始异常

    示例:
        @retry(times=3, delay=1.0, backoff=2.0)
        def call_api():
            pass

        @retry(
            times=5,
            delay=0.5,
            backoff=2.0,
            exceptions=(NetworkError, TimeoutError),
            timeout=30.0
        )
        def fetch_data():
            pass
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            total_time = 0.0

            for attempt in range(1, times + 1):
                start_time = time.time()

                try:
                    logger.debug(
                        f"[重试] {func.__name__} 开始第 {attempt}/{times} 次调用"
                    )

                    # 如果指定了超时，注入到 kwargs
                    if timeout is not None:
                        kwargs["timeout"] = timeout

                    result = func(*args, **kwargs)

                    # 成功回调
                    elapsed = time.time() - start_time
                    total_time += elapsed
                    if success_callback is not None:
                        success_callback(result, total_time)

                    logger.info(
                        f"[重试] {func.__name__} 第 {attempt} 次调用成功，"
                        f"耗时 {elapsed:.2f}s"
                    )
                    return result

                except exceptions as e:
                    elapsed = time.time() - start_time
                    total_time += elapsed

                    if attempt == times:
                        # 最后一次重试仍失败
                        logger.error(
                            f"[重试] {func.__name__} 失败，已重试 {times} 次，"
                            f"总耗时 {total_time:.2f}s，最终错误：{type(e).__name__}: {e}"
                        )
                        raise
                    else:
                        # 等待后重试
                        if retry_callback is not None:
                            retry_callback(attempt, e, current_delay)

                        logger.warning(
                            f"[重试] {func.__name__} 第 {attempt} 次失败 "
                            f"({type(e).__name__}: {str(e)[:100]}), "
                            f"等待 {current_delay:.1f}s 后重试..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff  # 指数退避

                except Exception as e:
                    # 不在重试列表中的异常，直接抛出
                    logger.error(
                        f"[重试] {func.__name__} 遇到不可重试错误 "
                        f"({type(e).__name__}: {e})"
                    )
                    raise

        return wrapper

    return decorator


def retry_with_fixed_delay(
    times: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
):
    """
    固定延迟重试装饰器（简化版）

    使用固定的重试延迟，不指数退避

    Args:
        times: 重试次数
        delay: 固定延迟秒数
        exceptions: 需要重试的异常类型

    示例:
        @retry_with_fixed_delay(times=3, delay=2.0)
        def call_api():
            pass
    """
    return retry(times=times, delay=delay, backoff=1.0, exceptions=exceptions)


def no_retry(func):
    """
    禁用重试的装饰器（用于明确不需要重试的函数）

    这是一个标记装饰器，用于代码审查和静态分析
    """
    func._no_retry = True
    return func
