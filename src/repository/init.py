"""SQLite 建表入口：默认保留数据，仅 --reset 清空并重建项目表。"""

import argparse
import logging

from src.config.logging_config import setup_logging
from src.repository.create_table import create_table_init, drop_table

logger = logging.getLogger(__name__)


def sqlite_table_reset() -> None:
    """显式删除并重建项目 SQLite 表；不操作 Chroma 或 ES。"""
    logger.warning("正在清空并重建项目 SQLite 表，现有数据将被删除")
    drop_table()
    create_table_init()
    logger.info("SQLite 项目表重置完成")


def sqlite_table_create_if_not() -> None:
    """创建缺失的表，保留已有表和数据。"""
    create_table_init()
    logger.info("SQLite 项目表检查完成，已创建缺失表，保留已有数据")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清空并重建项目 SQLite 表，不操作 Chroma 或 ES",
    )
    args = parser.parse_args()
    setup_logging()
    if args.reset:
        sqlite_table_reset()
    else:
        sqlite_table_create_if_not()


if __name__ == "__main__":
    main()
