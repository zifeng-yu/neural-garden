import logging
from functools import wraps

import src.config.logging_config as logging_config

logger = logging.getLogger(__name__)


def retry(times):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(times):
                try:
                    logger.info(f"调用函数 {func.__name__}")
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.info(f"第{i}次调用，失败：{e}")
                    if i == times - 1:
                        raise

        return wrapper

    return decorator
