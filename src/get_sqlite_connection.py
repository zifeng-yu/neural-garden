import os
import sqlite3

from src.config.config import SQLITE_TABLE


def get_sqlite_connection():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    sqlite_dir = os.path.join(base_dir, SQLITE_TABLE)
    conn = sqlite3.connect(sqlite_dir)
    conn.row_factory = sqlite3.Row
    return conn
