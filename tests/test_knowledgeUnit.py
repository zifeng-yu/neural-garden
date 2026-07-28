"""
知识单元模块的单元测试
"""

import pytest

from src.knowledge.knowledgeUnit import KnowledgeUnit


class TestKnowledgeUnit:
    """测试 KnowledgeUnit 数据类"""

    def test_create_knowledge_unit(self):
        """测试创建知识单元"""
        unit = KnowledgeUnit(
            title="测试标题",
            summary="这是一个测试摘要，长度超过50以满足要求！！！",
            keywords=["测试", "单元", "pytest"],
            content="测试内容",
            source="test.md",
            unit_id="abc123",
        )

        assert unit.title == "测试标题"
