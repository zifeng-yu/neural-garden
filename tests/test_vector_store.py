"""
Neural Garden Vector Store 单元测试

测试范围：
1. ChromaDB save_dao (knowledge, concept, insight)
2. ChromaDB delete_dao
3. ChromaDB query_dao
"""

import os
import sys
import unittest
import tempfile
import shutil

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store.save_dao import (
    KnowledgeUnitDTO,
    KnowledgeUnitMetadata,
    ConceptDTO,
    InsightDTO,
    InsightMetadata,
    save_knowlege,
    save_concept,
    save_insight,
    update_insight_metadata,
)
from src.vector_store.delete_dao import (
    delete_by_id_knowledge,
    delete_by_normalized_concet_hash_concept,
)
from src.config.config import (
    CHROMA_KNOWLEDGE_TABLE_NAME,
    CHROMA_CONCEPT_TABLE_NAME,
    CHROMA_INSIGHT_TABLE_NAME,
    PERSIST_DIRECTORY,
)
from src.get_chroma_collection import get_collection


class TestChromaKnowledge(unittest.TestCase):
    """测试 ChromaDB Knowledge collection"""

    @classmethod
    def setUpClass(cls):
        """所有测试前创建临时目录"""
        cls.test_dir = tempfile.mkdtemp()
        # 临时修改配置
        import src.config.config as config_module
        config_module.PERSIST_DIRECTORY = cls.test_dir

    @classmethod
    def tearDownClass(cls):
        """所有测试后清理临时目录"""
        import src.config.config as config_module
        config_module.PERSIST_DIRECTORY = PERSIST_DIRECTORY
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_save_and_query_knowledge(self):
        """测试保存和查询 knowledge"""
        dto = KnowledgeUnitDTO(
            id="test_know_1",
            embedding=[0.1, 0.2, 0.3],
            text="测试知识单元",
            metadata=KnowledgeUnitMetadata(
                file_name="test.md",
                document_id=1,
                document_chunk_id=10,
                title="测试标题",
                keywords=["关键词 1", "关键词 2"]
            )
        )
        
        save_knowlege(dto)
        
        # 查询验证
        collection = get_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
        result = collection.get(ids=["test_know_1"])
        
        self.assertEqual(len(result['ids']), 1)
        self.assertEqual(result['documents'][0], "测试知识单元")
        self.assertEqual(result['metadatas'][0]['file_name'], "test.md")

    def test_update_knowledge(self):
        """测试更新 knowledge（upsert）"""
        dto1 = KnowledgeUnitDTO(
            id="test_know_2",
            embedding=[0.1, 0.2, 0.3],
            text="原始文本",
            metadata=KnowledgeUnitMetadata(
                file_name="test.md",
                document_id=1,
                document_chunk_id=10,
                title="原始标题",
                keywords=[]
            )
        )
        save_knowlege(dto1)
        
        # 更新
        dto2 = KnowledgeUnitDTO(
            id="test_know_2",
            embedding=[0.4, 0.5, 0.6],
            text="更新后的文本",
            metadata=KnowledgeUnitMetadata(
                file_name="test.md",
                document_id=1,
                document_chunk_id=10,
                title="更新后的标题",
                keywords=[]
            )
        )
        save_knowlege(dto2)
        
        # 验证已更新
        collection = get_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
        result = collection.get(ids=["test_know_2"])
        
        self.assertEqual(result['documents'][0], "更新后的文本")
        self.assertEqual(result['metadatas'][0]['title'], "更新后的标题")

    def test_delete_knowledge(self):
        """测试删除 knowledge"""
        dto = KnowledgeUnitDTO(
            id="test_know_3",
            embedding=[0.1, 0.2, 0.3],
            text="待删除",
            metadata=KnowledgeUnitMetadata(
                file_name="test.md",
                document_id=1,
                document_chunk_id=10,
                title="标题",
                keywords=[]
            )
        )
        save_knowlege(dto)
        
        # 删除
        delete_by_id_knowledge(["test_know_3"])
        
        # 验证已删除
        collection = get_collection(CHROMA_KNOWLEDGE_TABLE_NAME)
        result = collection.get(ids=["test_know_3"])
        self.assertEqual(len(result['ids']), 0)


class TestChromaConcept(unittest.TestCase):
    """测试 ChromaDB Concept collection"""

    @classmethod
    def setUpClass(cls):
        """所有测试前创建临时目录"""
        cls.test_dir = tempfile.mkdtemp()
        import src.config.config as config_module
        config_module.PERSIST_DIRECTORY = cls.test_dir

    @classmethod
    def tearDownClass(cls):
        """所有测试后清理临时目录"""
        import src.config.config as config_module
        config_module.PERSIST_DIRECTORY = PERSIST_DIRECTORY
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_save_concept(self):
        """测试保存 concept"""
        dto = ConceptDTO(
            id="hash_负利率",
            embedding=[0.1, 0.2, 0.3],
            normalized_concept="负利率"
        )
        
        save_concept(dto)
        
        # 查询验证
        collection = get_collection(CHROMA_CONCEPT_TABLE_NAME)
        result = collection.get(ids=["hash_负利率"])
        
        self.assertEqual(len(result['ids']), 1)
        self.assertEqual(result['documents'][0], "负利率")

    def test_delete_concept(self):
        """测试删除 concept"""
        dto = ConceptDTO(
            id="hash_测试概念",
            embedding=[0.1, 0.2, 0.3],
            normalized_concept="测试概念"
        )
        save_concept(dto)
        
        # 删除
        delete_by_normalized_concet_hash_concept(["hash_测试概念"])
        
        # 验证已删除
        collection = get_collection(CHROMA_CONCEPT_TABLE_NAME)
        result = collection.get(ids=["hash_测试概念"])
        self.assertEqual(len(result['ids']), 0)


class TestChromaInsight(unittest.TestCase):
    """测试 ChromaDB Insight collection"""

    @classmethod
    def setUpClass(cls):
        """所有测试前创建临时目录"""
        cls.test_dir = tempfile.mkdtemp()
        import src.config.config as config_module
        config_module.PERSIST_DIRECTORY = cls.test_dir

    @classmethod
    def tearDownClass(cls):
        """所有测试后清理临时目录"""
        import src.config.config as config_module
        config_module.PERSIST_DIRECTORY = PERSIST_DIRECTORY
        shutil.rmtree(cls.test_dir, ignore_errors=True)

    def test_save_insight(self):
        """测试保存 insight"""
        dto = InsightDTO(
            id="1",
            embedding=[0.1, 0.2, 0.3],
            embedding_text="测试洞察文本",
            metadata=InsightMetadata(
                insight_id=1,
                source_type="user",
                source_status="online",
                document_id=None,
                document_chunk_id=None
            )
        )
        
        save_insight(dto)
        
        # 查询验证
        collection = get_collection(CHROMA_INSIGHT_TABLE_NAME)
        result = collection.get(ids=["1"])
        
        self.assertEqual(len(result['ids']), 1)
        self.assertEqual(result['documents'][0], "测试洞察文本")
        self.assertEqual(result['metadatas'][0]['source_type'], "user")

    def test_update_insight_metadata(self):
        """测试更新 insight metadata"""
        dto = InsightDTO(
            id="2",
            embedding=[0.1, 0.2, 0.3],
            embedding_text="测试文本",
            metadata=InsightMetadata(
                insight_id=2,
                source_type="document",
                source_status="online",
                document_id=100,
                document_chunk_id=None
            )
        )
        save_insight(dto)
        
        # 更新 metadata
        new_metadata = InsightMetadata(
            insight_id=2,
            source_type="document",
            source_status="offline",  # 改为 offline
            document_id=100,
            document_chunk_id=None
        )
        update_insight_metadata("2", new_metadata)
        
        # 验证已更新
        collection = get_collection(CHROMA_INSIGHT_TABLE_NAME)
        result = collection.get(ids=["2"])
        
        self.assertEqual(result['metadatas'][0]['source_status'], "offline")
        self.assertEqual(result['metadatas'][0]['source_type'], "document")

    def test_insight_metadata_nullable(self):
        """测试 InsightMetadata 可空字段"""
        metadata = InsightMetadata(
            insight_id=3,
            source_type="user",
            source_status="online",
            document_id=None,
            document_chunk_id=None
        )
        
        result_dict = metadata.to_dict()
        
        # None 值应该被过滤掉
        self.assertNotIn('document_id', result_dict)
        self.assertNotIn('document_chunk_id', result_dict)
        self.assertIn('insight_id', result_dict)
        self.assertIn('source_type', result_dict)


if __name__ == "__main__":
    unittest.main()
