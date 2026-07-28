import logging
import logging.config
from pathlib import Path


def setup_logging(log_dir: str = "logs", level: str = "INFO"):
    """
    生产级日志配置
    - 控制台输出(INFO 及以上)
    - 滚动文件(DEBUG 及以上,10MB * 10 个备份）
    - 支持按模块单独控制级别
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    config = {
        "version": 1,
        "disable_existing_loggers": False,  # 保留已有 logger（很重要）
        "formatters": {
            "standard": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(asctime)s | %(levelname)-8s | %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": str(log_path / "app.log"),
                "maxBytes": 10 * 1024 * 1024,  # 10 MB
                "backupCount": 10,
                "encoding": "utf-8",
            },
            # 可选：单独的错误日志文件
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "standard",
                "filename": str(log_path / "error.log"),
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        # "loggers": {
        #     # ========== 按模块控制级别 ==========
        #     # 示例：让某个模块只输出 WARNING 及以上
        #     "app.service": {
        #         "level": "WARNING",
        #         "handlers": ["console", "file", "error_file"],
        #         "propagate": False,  # 不向上传递，避免重复打印
        #     },
        #     "app.api": {
        #         "level": "DEBUG",
        #         "handlers": ["console", "file", "error_file"],
        #         "propagate": False,
        #     },
        #     # 第三方库降噪（非常常用）
        #     "urllib3": {"level": "WARNING", "propagate": False},
        #     "requests": {"level": "WARNING", "propagate": False},
        #     "asyncio": {"level": "WARNING", "propagate": False},
        # },
        "root": {
            "level": level.upper(),  # 全局默认级别
            "handlers": ["console", "file", "error_file"],
        },
    }

    logging.config.dictConfig(config)


setup_logging()

# 方便直接测试
# if __name__ == "__main__":
#     setup_logging()
#     logger = logging.getLogger(__name__)
#     logger.debug("这是 DEBUG")
#     logger.info("这是 INFO")
#     logger.warning("这是 WARNING")
#     logger.error("这是 ERROR")
