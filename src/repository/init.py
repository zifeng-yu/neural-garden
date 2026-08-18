import logging

import src.config.logging_config as logging_config
from src.repository.create_table import create_table_init, drop_table

logger = logging.getLogger(__name__)


def sqlite_table_init():
    drop_table()
    create_table_init()
    logger.info("数据库 初始化化结束")


if __name__ == "__main__":
    sqlite_table_init()
