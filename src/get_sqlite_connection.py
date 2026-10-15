import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from src.config.config import SQLITE_TABLE


@contextmanager
def get_sqlite_read_connection() -> Generator[sqlite3.Connection, None, None]:
    """打开已有数据库的只读连接；不创建目录或数据库，不提交事务。"""
    sqlite_path = Path(SQLITE_TABLE).expanduser()
    uri = sqlite_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_sqlite_connection() -> Generator[sqlite3.Connection, None, None]:
    sqlite_path = Path(SQLITE_TABLE)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
