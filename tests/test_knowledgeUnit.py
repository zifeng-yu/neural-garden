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
            summary="这是一个测试摘要，长度超过 50 以满足要求！！！",
            keywords=["测试", "单元", "pytest"],
            content="测试内容",
            source="test.md",
            unit_id="abc123",
        )

        assert unit.title == "测试标题"
        assert unit.summary == "这是一个测试摘要，长度超过 50 以满足要求！！！"
        assert len(unit.keywords) == 3
        assert unit.keywords[0] == "测试"
        assert unit.keywords[1] == "单元"
        assert unit.keywords[2] == "pytest"
        assert unit.content == "测试内容"
        assert unit.source == "test.md"
        assert unit.unit_id == "abc123"

    def test_to_dict(self):
        """测试 to_dict() 方法"""
        unit = KnowledgeUnit(
            title="标题",
            summary="摘要",
            keywords=["key1", "key2"],
            content="内容",
            source="test.md",
            unit_id="xyz789",
        )

        result = unit.to_dict()

        assert isinstance(result, dict)
        assert result["title"] == "标题"
        assert result["summary"] == "摘要"
        assert result["keywords"] == ["key1", "key2"]
        assert result["content"] == "内容"
        assert result["source"] == "test.md"
        assert result["unit_id"] == "xyz789"

    def test_to_embedding_text(self):
        """测试 to_embedding_text() 方法"""
        unit = KnowledgeUnit(
            title="标题",
            summary="摘要内容",
            keywords=["关键词 1", "关键词 2"],
            content="内容",
            source="test.md",
            unit_id="id123",
        )

        result = unit.to_embedding_text()

        assert "标题" in result
        assert "摘要内容" in result
        assert "关键词 1" in result
        assert "关键词 2" in result
        assert result.startswith("标题。摘要内容。关键词：")
        assert "关键词 1, 关键词 2" in result

    def test_empty_keywords(self):
        """测试空关键词列表"""
        unit = KnowledgeUnit(
            title="标题",
            summary="摘要",
            keywords=[],
            content="内容",
            source="test.md",
            unit_id="id1",
        )

        result = unit.to_embedding_text()
        assert "关键词：" in result
        assert "关键词：, " not in result

    def test_special_characters(self):
        """测试特殊字符"""
        unit = KnowledgeUnit(
            title="标题\n带换行",
            summary="摘要\r\n带回车",
            keywords=["key\t1"],
            content="内容",
            source="test.md",
            unit_id="id2",
        )

        assert "\n" in unit.title
        assert "\r\n" in unit.summary
        assert "\t" in unit.keywords[0]

    def test_long_title(self):
        """测试长标题（边界值）"""
        long_title = "这是一" * 20  # 60 字符（"这是一"是 3 字符 × 20）

        unit = KnowledgeUnit(
            title=long_title,
            summary="摘要",
            keywords=["test"],
            content="内容",
            source="test.md",
            unit_id="id3",
        )

        assert len(unit.title) == 60

    def test_unit_id_stability(self):
        """测试 unit_id 稳定性"""
        unit1 = KnowledgeUnit(
            title="标题",
            summary="摘要",
            keywords=["test"],
            content="内容",
            source="test.md",
            unit_id="stable_id_123",
        )

        unit2 = KnowledgeUnit(
            title="标题",
            summary="摘要",
            keywords=["test"],
            content="内容",
            source="test.md",
            unit_id="stable_id_123",
        )

        assert unit1.unit_id == unit2.unit_id
        assert unit1.unit_id == "stable_id_123"
