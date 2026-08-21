import os
import sqlite3
from contextlib import contextmanager

from src.config.config import SQLITE_TABLE


@contextmanager
def get_sqlite_connection():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    sqlite_dir = os.path.join(base_dir, SQLITE_TABLE)
    conn = sqlite3.connect(sqlite_dir)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
