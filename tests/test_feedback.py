"""
Neural Garden Feedback 模块单元测试

测试范围：
1. search_sessions 表 CRUD
2. search_results 表 CRUD
3. feedback_events 表 CRUD
4. 外键约束验证
"""

import sqlite3
import unittest
from datetime import datetime

from src.repository.create_table import create_table_init_for_memory
from src.repository.search_sessions import (
    _save_search_sessions,
    _update_result_count_by_session_id,
)
from src.repository.search_results import (
    RetrievalTypeEnum,
    SourceTypeEnum,
    _save_search_results,
)
from src.repository.feedback_events import EventTypeEnum, _save_feedback_events


class TestSearchSessions(unittest.TestCase):
    """测试 search_sessions 表操作"""

    def setUp(self):
        """每个测试前初始化内存数据库"""
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        create_table_init_for_memory(self.conn)

    def tearDown(self):
        """每个测试后关闭连接"""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()

    def test_save_and_query_session(self):
        """测试保存和查询会话"""
        session_id = "test_session_001"
        query = "测试查询"
        result_count = 5

        _save_search_sessions(self.conn, session_id, query, result_count)

        # 查询验证
        cursor = self.conn.execute(
            "SELECT * FROM search_sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["query"], query)
        self.assertEqual(row["result_count"], result_count)

    def test_update_result_count(self):
        """测试更新结果数量"""
        session_id = "test_session_002"
        _save_search_sessions(self.conn, session_id, "初始查询", 0)

        # 更新
        _update_result_count_by_session_id(self.conn, session_id, 10)

        # 验证已更新
        cursor = self.conn.execute(
            "SELECT result_count FROM search_sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        self.assertEqual(row["result_count"], 10)

    def test_unique_session_id(self):
        """测试 session_id 唯一约束"""
        session_id = "test_session_duplicate"
        _save_search_sessions(self.conn, session_id, "查询 1", 5)

        # 重复插入应该失败
        with self.assertRaises(sqlite3.IntegrityError):
            _save_search_sessions(self.conn, session_id, "查询 2", 3)


class TestSearchResults(unittest.TestCase):
    """测试 search_results 表操作"""

    def setUp(self):
        """每个测试前初始化内存数据库"""
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        create_table_init_for_memory(self.conn)

    def tearDown(self):
        """每个测试后关闭连接"""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()

    def test_save_result_with_enum(self):
        """测试保存搜索结果（使用枚举类型）"""
        session_id = "test_session_003"
        _save_search_sessions(self.conn, session_id, "测试查询", 1)

        _save_search_results(
            conn=self.conn,
            session_id=session_id,
            document_id=1,
            document_chunk_id=10,
            rank=1,
            score=0.85,
            raw_score=0.15,
            retrieval_type=RetrievalTypeEnum.VECTOR,
            source_type=SourceTypeEnum.CHROMA_KNOWLEDGE
        )

        # 查询验证
        cursor = self.conn.execute(
            "SELECT * FROM search_results WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["document_id"], 1)
        self.assertEqual(row["rank"], 1)
        self.assertEqual(row["retrieval_type"], "vector")
        self.assertEqual(row["source_type"], "chroma_knowledge")

    def test_save_result_with_nulls(self):
        """测试保存搜索结果（document_id 和 chunk_id 可为空）"""
        session_id = "test_session_004"
        _save_search_sessions(self.conn, session_id, "测试查询", 1)

        _save_search_results(
            conn=self.conn,
            session_id=session_id,
            document_id=None,
            document_chunk_id=None,
            rank=1,
            score=0.9,
            raw_score=0.1,
            retrieval_type=RetrievalTypeEnum.VECTOR,
            source_type=SourceTypeEnum.CHROMA_INSIGHT
        )

        # 查询验证
        cursor = self.conn.execute(
            "SELECT * FROM search_results WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertIsNone(row["document_id"])
        self.assertIsNone(row["document_chunk_id"])


class TestFeedbackEvents(unittest.TestCase):
    """测试 feedback_events 表操作"""

    def setUp(self):
        """每个测试前初始化内存数据库"""
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        create_table_init_for_memory(self.conn)

    def tearDown(self):
        """每个测试后关闭连接"""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()

    def test_save_click_event(self):
        """测试保存点击事件"""
        session_id = "test_session_005"
        _save_search_sessions(self.conn, session_id, "测试查询", 3)

        _save_feedback_events(
            conn=self.conn,
            session_id=session_id,
            event_type=EventTypeEnum.CLICK,
            document_id=1,
            rank=2,
            dwell_time=None
        )

        # 查询验证
        cursor = self.conn.execute(
            "SELECT * FROM feedback_events WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["event_type"], "click")
        self.assertEqual(row["document_id"], 1)
        self.assertEqual(row["rank"], 2)

    def test_save_dwell_event(self):
        """测试保存停留时间事件"""
        session_id = "test_session_006"
        _save_search_sessions(self.conn, session_id, "测试查询", 3)

        _save_feedback_events(
            conn=self.conn,
            session_id=session_id,
            event_type=EventTypeEnum.DWELL,
            document_id=1,
            rank=1,
            dwell_time=30
        )

        # 查询验证
        cursor = self.conn.execute(
            "SELECT * FROM feedback_events WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["event_type"], "dwell")
        self.assertEqual(row["dwell_time"], 30)

    def test_all_event_types(self):
        """测试所有事件类型"""
        session_id = "test_session_007"
        _save_search_sessions(self.conn, session_id, "测试查询", 3)

        for event_type in EventTypeEnum:
            _save_feedback_events(
                conn=self.conn,
                session_id=session_id,
                event_type=event_type,
                document_id=1,
                rank=1,
                dwell_time=10
            )

        # 验证所有事件都已保存
        cursor = self.conn.execute(
            "SELECT DISTINCT event_type FROM feedback_events WHERE session_id = ?",
            (session_id,)
        )
        rows = cursor.fetchall()
        self.assertEqual(len(rows), len(EventTypeEnum))


class TestForeignKeyConstraints(unittest.TestCase):
    """测试外键约束"""

    def setUp(self):
        """每个测试前初始化内存数据库并开启外键约束"""
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")  # 开启外键约束
        create_table_init_for_memory(self.conn)

    def tearDown(self):
        """每个测试后关闭连接"""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()

    def test_feedback_events_requires_session(self):
        """测试 feedback_events 必须有有效的 session_id"""
        # 尝试插入不存在的 session_id
        with self.assertRaises(sqlite3.IntegrityError):
            _save_feedback_events(
                conn=self.conn,
                session_id="nonexistent_session",
                event_type=EventTypeEnum.CLICK,
                document_id=1,
                rank=1,
                dwell_time=None
            )

    def test_search_results_requires_session(self):
        """测试 search_results 必须有有效的 session_id"""
        # 尝试插入不存在的 session_id
        with self.assertRaises(sqlite3.IntegrityError):
            _save_search_results(
                conn=self.conn,
                session_id="nonexistent_session",
                document_id=1,
                document_chunk_id=10,
                rank=1,
                score=0.9,
                raw_score=0.1,
                retrieval_type=RetrievalTypeEnum.VECTOR,
                source_type=SourceTypeEnum.CHROMA_KNOWLEDGE
            )


if __name__ == "__main__":
    unittest.main()
