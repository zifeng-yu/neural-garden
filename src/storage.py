# src/storage.py
"""
SQLite 存储模块 — 用于管理概念、Insight、日志等元数据
"""

import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


class Storage:
    """SQLite 数据库操作封装"""

    def __init__(self, db_path: str = "data/neural_garden.db", in_memory: bool = False):
        self.db_path = db_path
        self.in_memory = in_memory or db_path == ":memory:"
        self._conn = None  # 单连接模式（内存数据库必需）
        self._init_tables()

    def _get_connection(self):
        """获取数据库连接（单连接模式支持内存数据库）"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_tables(self):
        """初始化表结构"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # 概念表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                category TEXT,
                embedding_id TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT
            )
        """)

        # Insight 表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                trigger_content TEXT,
                source TEXT,
                content TEXT NOT NULL,
                related_concepts TEXT,  -- JSON 数组
                action_items TEXT,      -- JSON 数组
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 搜索日志表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                results_count INTEGER,
                latency_ms REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

    # ========== Concept CRUD ==========

    def create_concept(
        self,
        id: str,
        name: str,
        description: str,
        category: str,
        embedding_id: Optional[str] = None,
    ) -> bool:
        """创建概念"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                """INSERT INTO concepts (id, name, description, category, embedding_id) 
                   VALUES (?, ?, ?, ?, ?)""",
                (id, name, description, category, embedding_id),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            # UNIQUE 约束冲突（概念已存在）
            return False

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """查询单个概念"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM concepts WHERE id = ?", (concept_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_concept_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """按名称查询概念"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM concepts WHERE name = ?", (name,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def list_concepts(
        self, category: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """列出概念（可按类别筛选）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        if category:
            cursor.execute(
                "SELECT * FROM concepts WHERE category = ? LIMIT ?", (category, limit)
            )
        else:
            cursor.execute("SELECT * FROM concepts LIMIT ?", (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def update_concept(self, concept_id: str, **kwargs) -> bool:
        """更新概念（动态字段）"""
        if not kwargs:
            return False

        fields = ", ".join(f"{k} = ?" for k in kwargs.keys())
        values = list(kwargs.values()) + [concept_id]

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE concepts SET {fields}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            values,
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete_concept(self, concept_id: str) -> bool:
        """删除概念"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM concepts WHERE id = ?", (concept_id,))
        conn.commit()
        return cursor.rowcount > 0

    # ========== Insight CRUD ==========

    def create_insight(
        self,
        id: str,
        title: str,
        content: str,
        trigger_content: Optional[str] = None,
        source: Optional[str] = None,
        related_concepts: Optional[List[str]] = None,
        action_items: Optional[List[str]] = None,
    ) -> bool:
        """创建 Insight"""
        import json

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO insights 
               (id, title, trigger_content, source, content, related_concepts, action_items)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                id,
                title,
                trigger_content,
                source,
                content,
                json.dumps(related_concepts or []),
                json.dumps(action_items or []),
            ),
        )
        conn.commit()
        return True

    def get_insight(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """查询 Insight"""
        import json

        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM insights WHERE id = ?", (insight_id,))
        row = cursor.fetchone()
        if row:
            data = dict(row)
            # 解析 JSON 字段
            data["related_concepts"] = json.loads(data["related_concepts"])
            data["action_items"] = json.loads(data["action_items"])
            return data
        return None

    # ========== Search Logs ==========

    def log_search(self, query: str, results_count: int, latency_ms: float) -> int:
        """记录搜索日志"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO search_logs (query, results_count, latency_ms) VALUES (?, ?, ?)",
            (query, results_count, latency_ms),
        )
        conn.commit()
        return cursor.lastrowid  # 返回自增 ID

    def get_search_stats(self, days: int = 7) -> Dict[str, Any]:
        """获取搜索统计（最近 N 天）"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total_searches,
                AVG(results_count) as avg_results,
                AVG(latency_ms) as avg_latency
            FROM search_logs
            WHERE created_at >= datetime('now', ?)
        """,
            (f"-{days} days",),
        )
        row = cursor.fetchone()
        return dict(row) if row else {}
