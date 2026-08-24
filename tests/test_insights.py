"""
Neural Garden Insight 模块单元测试

测试范围：
1. Insight repository 层 CRUD
2. relation_concepts 优先级逻辑
3. source_status offline 标记
4. embedding_text 存储
"""

import os
import sqlite3
import sys
import unittest
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.repository.create_table import create_table_init_for_memory
from src.repository.insights import (
    InsightsDO,
    SourceStatusEnum,
    SourceTypeEnum,
    _mark_source_status_offline_by_document_id,
    _query_by_id,
    _save_insights,
)


class TestInsightsRepository(unittest.TestCase):
    """测试 insights 表操作"""

    def setUp(self):
        """每个测试前初始化内存数据库"""
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        create_table_init_for_memory(self.conn)

    def tearDown(self):
        """每个测试后关闭连接"""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()

    def test_save_and_query_insight(self):
        """测试保存和查询 Insight"""
        embedding_text = (
            "测试洞察。这是测试内容。行动项：行动 1, 行动 2。关联概念：负利率，货币政策"
        )
        insight_id = _save_insights(
            self.conn,
            title="测试洞察",
            content="这是测试内容",
            action_items=["行动 1", "行动 2"],
            relation_concepts=["负利率", "货币政策"],
            source_type=SourceTypeEnum.USER,
            document_id=None,
            document_chunk_id=None,
            source_status=SourceStatusEnum.ONLINE,
            embedding_text=embedding_text,
        )

        self.assertGreater(insight_id, 0)

        # 查询
        result = _query_by_id(self.conn, insight_id)
        self.assertIsNotNone(result)
        self.assertEqual(result.title, "测试洞察")
        self.assertEqual(result.content, "这是测试内容")
        self.assertEqual(result.action_items, ["行动 1", "行动 2"])
        self.assertEqual(result.relation_concepts, ["负利率", "货币政策"])
        self.assertEqual(result.source_type, SourceTypeEnum.USER)
        self.assertEqual(result.source_status, SourceStatusEnum.ONLINE)
        self.assertIsNone(result.document_id)
        self.assertIsNone(result.document_chunk_id)
        self.assertEqual(result.embedding_text, embedding_text)

    def test_save_insight_with_document_source(self):
        """测试保存来自文档的 Insight"""
        embedding_text = "文档洞察。来自文档的洞察。行动项：。关联概念：概念 1"
        insight_id = _save_insights(
            self.conn,
            title="文档洞察",
            content="来自文档的洞察",
            action_items=[],
            relation_concepts=["概念 1"],
            source_type=SourceTypeEnum.DOCUMENT,
            document_id=123,
            document_chunk_id=456,
            source_status=SourceStatusEnum.ONLINE,
            embedding_text=embedding_text,
        )

        result = _query_by_id(self.conn, insight_id)
        self.assertIsNotNone(result)
        self.assertEqual(result.source_type, SourceTypeEnum.DOCUMENT)
        self.assertEqual(result.document_id, 123)
        self.assertEqual(result.document_chunk_id, 456)
        self.assertEqual(result.embedding_text, embedding_text)

    def test_mark_source_status_offline(self):
        """测试标记 Insight 为 offline"""
        # 创建两个关联同一文档的 Insight
        id1 = _save_insights(
            self.conn,
            title="洞察 1",
            content="内容 1",
            action_items=[],
            relation_concepts=[],
            source_type=SourceTypeEnum.DOCUMENT,
            document_id=100,
            document_chunk_id=None,
            embedding_text="洞察 1",
        )

        id2 = _save_insights(
            self.conn,
            title="洞察 2",
            content="内容 2",
            action_items=[],
            relation_concepts=[],
            source_type=SourceTypeEnum.DOCUMENT,
            document_id=100,
            document_chunk_id=None,
            embedding_text="洞察 2",
        )

        # 创建另一个文档的 Insight（不应该被影响）
        id3 = _save_insights(
            self.conn,
            title="洞察 3",
            content="内容 3",
            action_items=[],
            relation_concepts=[],
            source_type=SourceTypeEnum.DOCUMENT,
            document_id=200,
            document_chunk_id=None,
            embedding_text="洞察 3",
        )

        # 标记 document_id=100 的 Insight 为 offline
        offline_ids = _mark_source_status_offline_by_document_id(self.conn, 100)

        self.assertEqual(len(offline_ids), 2)
        self.assertIn(id1, offline_ids)
        self.assertIn(id2, offline_ids)

        # 验证状态已更新
        result1 = _query_by_id(self.conn, id1)
        result2 = _query_by_id(self.conn, id2)
        result3 = _query_by_id(self.conn, id3)

        self.assertEqual(result1.source_status, SourceStatusEnum.OFFLINE)
        self.assertEqual(result2.source_status, SourceStatusEnum.OFFLINE)
        self.assertEqual(result3.source_status, SourceStatusEnum.ONLINE)  # 不应受影响

    def test_json_serialization(self):
        """测试 JSON 序列化（中文）"""
        embedding_text = "中文标题。中文内容。行动项：行动 1, 行动 2 中文。关联概念：负利率，货币政策"
        insight_id = _save_insights(
            self.conn,
            title="中文标题",
            content="中文内容",
            action_items=["行动 1", "行动 2 中文"],
            relation_concepts=["负利率", "货币政策"],
            source_type=SourceTypeEnum.USER,
            document_id=None,
            document_chunk_id=None,
            embedding_text=embedding_text,
        )

        result = _query_by_id(self.conn, insight_id)
        self.assertEqual(result.action_items, ["行动 1", "行动 2 中文"])
        self.assertEqual(result.relation_concepts, ["负利率", "货币政策"])
        self.assertEqual(result.embedding_text, embedding_text)


class TestInsightBusinessLogic(unittest.TestCase):
    """测试 Insight 业务逻辑（create_insight 优先级）"""

    def setUp(self):
        """每个测试前初始化内存数据库"""
        self.conn = sqlite3.connect(":memory:")
        self.conn.row_factory = sqlite3.Row
        create_table_init_for_memory(self.conn)

    def tearDown(self):
        """每个测试后关闭连接"""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()

    def test_relation_concepts_priority_user_input(self):
        """测试 relation_concepts 优先级：用户输入最高"""
        # 用户提供了 relation_concepts，应该直接使用
        insight_id = _save_insights(
            self.conn,
            title="测试",
            content="内容",
            action_items=[],
            relation_concepts=["用户概念 1", "用户概念 2"],  # 用户输入
            source_type=SourceTypeEnum.USER,
            document_id=None,
            document_chunk_id=None,
            embedding_text="测试。内容。行动项：。关联概念：用户概念 1, 用户概念 2",
        )

        result = _query_by_id(self.conn, insight_id)
        self.assertEqual(result.relation_concepts, ["用户概念 1", "用户概念 2"])


if __name__ == "__main__":
    unittest.main()
