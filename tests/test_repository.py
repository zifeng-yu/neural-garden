"""
Neural Garden 单元测试套件

测试范围：
1. 增量索引逻辑（del/copy/new）
2. repository 层 CRUD 操作
3. Chroma 清理逻辑

注意：使用内存数据库（:memory:），测试后自动清理，不污染实际数据库
"""

import os
import sys
import unittest
import sqlite3
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.repository.documents import (
    insert_document,
    _query_by_file_name_hash,
    _query_by_content_hash,
    delete_by_id,
)
from src.repository.document_chunks import (
    insert_document_chunk,
    _query_by_document_id,
    delete_by_document_id,
)
from src.repository.document_chunk_concepts import (
    insert_concept,
    _query_by_document_id as query_concepts_by_document_id,
    _query_by_not_document_id_and_in_normalized_concepts,
    delete_by_document_id as delete_concepts_by_document_id,
)
from src.util.getHashValue import get_hash_value as hash
from src.repository.create_table import create_table_init_for_memory


def get_memory_sqlite_connection():
    """获取内存 SQLite 连接（用于测试）"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


class TestDocumentsRepository(unittest.TestCase):
    """测试 documents 表操作"""

    def setUp(self):
        """每个测试前初始化内存数据库"""
        self.conn = get_memory_sqlite_connection()
        create_table_init_for_memory(self.conn)

    def tearDown(self):
        """每个测试后关闭连接（自动清理数据）"""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def test_insert_and_query_by_file_name_hash(self):
        """测试插入和按文件名 hash 查询"""
        file_name = "test_doc1.md"
        file_name_hash = hash(file_name)
        content_hash = hash("test content 1")

        doc_id = insert_document(self.conn, file_name, file_name_hash, content_hash)
        self.assertIsNotNone(doc_id)
        self.assertGreater(doc_id, 0)

        # 查询
        result = _query_by_file_name_hash(self.conn, file_name_hash)
        self.assertIsNotNone(result)
        self.assertEqual(result.file_name, file_name)
        self.assertEqual(result.file_name_hash, file_name_hash)
        self.assertEqual(result.content_hash, content_hash)

    def test_query_by_content_hash(self):
        """测试按内容 hash 查询"""
        file_name = "test_doc2.md"
        file_name_hash = hash(file_name)
        content_hash = hash("test content 2")

        insert_document(self.conn, file_name, file_name_hash, content_hash)

        result = _query_by_content_hash(self.conn, content_hash)
        self.assertIsNotNone(result)
        self.assertEqual(result.file_name, file_name)

    def test_delete_by_id(self):
        """测试删除"""
        file_name = "test_doc3.md"
        file_name_hash = hash(file_name)
        content_hash = hash("test content 3")

        doc_id = insert_document(self.conn, file_name, file_name_hash, content_hash)
        delete_by_id(self.conn, doc_id)

        # 验证已删除
        result = _query_by_file_name_hash(self.conn, file_name_hash)
        self.assertIsNone(result)


class TestDocumentChunksRepository(unittest.TestCase):
    """测试 document_chunks 表操作"""

    def setUp(self):
        """每个测试前初始化内存数据库"""
        self.conn = get_memory_sqlite_connection()
        create_table_init_for_memory(self.conn)

    def test_insert_and_query_by_document_id(self):
        """测试插入分块和按文档 ID 查询"""
        file_name = "test_chunk_doc1.md"
        file_name_hash = hash(file_name)
        content_hash = hash("test content")

        doc_id = insert_document(self.conn, file_name, file_name_hash, content_hash)
        chunk_id = insert_document_chunk(self.conn, doc_id, 1, "chunk content 1", hash("chunk 1"))
        self.assertGreater(chunk_id, 0)

        # 查询
        chunks = _query_by_document_id(self.conn, doc_id)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].split_no, 1)
        self.assertEqual(chunks[0].content, "chunk content 1")

    def test_delete_by_document_id(self):
        """测试级联删除分块"""
        file_name = "test_chunk_doc2.md"
        file_name_hash = hash(file_name)
        content_hash = hash("test content")

        doc_id = insert_document(self.conn, file_name, file_name_hash, content_hash)
        insert_document_chunk(self.conn, doc_id, 1, "chunk 1", hash("chunk 1"))
        insert_document_chunk(self.conn, doc_id, 2, "chunk 2", hash("chunk 2"))

        delete_by_document_id(self.conn, doc_id)

        # 验证已删除
        chunks = _query_by_document_id(self.conn, doc_id)
        self.assertEqual(len(chunks), 0)


class TestDocumentChunkConceptsRepository(unittest.TestCase):
    """测试 document_chunk_concepts 表操作"""

    def setUp(self):
        """每个测试前初始化内存数据库"""
        self.conn = get_memory_sqlite_connection()
        create_table_init_for_memory(self.conn)

    def test_insert_and_query_concepts(self):
        """测试插入概念和查询"""
        file_name = "test_concept_doc1.md"
        file_name_hash = hash(file_name)
        content_hash = hash("test content")

        doc_id = insert_document(self.conn, file_name, file_name_hash, content_hash)
        chunk_id = insert_document_chunk(self.conn, doc_id, 1, "chunk content", hash("chunk"))
        concept_id = insert_concept(self.conn, doc_id, chunk_id, "负利率", "负利率", hash("负利率"))
        self.assertGreater(concept_id, 0)

        # 查询
        concepts = query_concepts_by_document_id(self.conn, doc_id)
        self.assertEqual(len(concepts), 1)
        self.assertEqual(concepts[0].concept, "负利率")
        self.assertEqual(concepts[0].normalized_concept, "负利率")

    def test_query_by_not_document_id_and_in_normalized_concepts(self):
        """测试查询其他文档中的相同概念（用于 Chroma 清理逻辑）"""
        # 文档 A 有概念"负利率"
        file_name_a = "test_concept_doc_a.md"
        file_name_hash_a = hash(file_name_a)
        doc_id_a = insert_document(self.conn, file_name_a, file_name_hash_a, hash("content a"))
        chunk_id_a = insert_document_chunk(self.conn, doc_id_a, 1, "chunk a", hash("chunk a"))
        insert_concept(self.conn, doc_id_a, chunk_id_a, "负利率", "负利率", hash("负利率"))

        # 文档 B 也有概念"负利率"
        file_name_b = "test_concept_doc_b.md"
        file_name_hash_b = hash(file_name_b)
        doc_id_b = insert_document(self.conn, file_name_b, file_name_hash_b, hash("content b"))
        chunk_id_b = insert_document_chunk(self.conn, doc_id_b, 1, "chunk b", hash("chunk b"))
        insert_concept(self.conn, doc_id_b, chunk_id_b, "负利率", "负利率", hash("负利率"))

        # 查询"除了文档 A 外，还有哪些文档有'负利率'概念"
        result = _query_by_not_document_id_and_in_normalized_concepts(
            self.conn, doc_id_a, ["负利率"]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].normalized_concept, "负利率")
        # 应该是文档 B 的概念
        self.assertNotEqual(result[0].document_id, doc_id_a)

    def test_query_by_not_document_id_and_in_normalized_concepts_unique_concept(self):
        """测试查询其他文档中的相同概念（概念只在当前文档中）"""
        # 文档 A 有独有概念"测试独有概念"
        file_name_a = "test_concept_doc_unique.md"
        file_name_hash_a = hash(file_name_a)
        doc_id_a = insert_document(self.conn, file_name_a, file_name_hash_a, hash("content a"))
        chunk_id_a = insert_document_chunk(self.conn, doc_id_a, 1, "chunk a", hash("chunk a"))
        insert_concept(self.conn, doc_id_a, chunk_id_a, "测试独有概念", "测试独有概念", hash("测试独有概念"))

        # 查询其他文档是否有"测试独有概念"
        result = _query_by_not_document_id_and_in_normalized_concepts(
            self.conn, doc_id_a, ["测试独有概念"]
        )
        # 应该为空，因为这个概念只在文档 A 中
        self.assertEqual(len(result), 0)


class TestChromaCleanupLogic(unittest.TestCase):
    """测试 Chroma 清理逻辑（核心测试）"""

    def setUp(self):
        """每个测试前初始化内存数据库"""
        self.conn = get_memory_sqlite_connection()
        create_table_init_for_memory(self.conn)

    def test_cleanup_logic_shared_concept(self):
        """测试共享概念的清理逻辑（不应该删除）"""
        # 场景：文档 A 和文档 B 共享概念"负利率"
        # 删除文档 A 时，"负利率"的 Chroma 数据不应该被删除

        file_name_a = "test_shared_a.md"
        file_name_hash_a = hash(file_name_a)
        file_name_b = "test_shared_b.md"
        file_name_hash_b = hash(file_name_b)

        doc_id_a = insert_document(self.conn, file_name_a, file_name_hash_a, hash("content a"))
        chunk_id_a = insert_document_chunk(self.conn, doc_id_a, 1, "chunk a", hash("chunk a"))
        insert_concept(self.conn, doc_id_a, chunk_id_a, "负利率", "负利率", hash("负利率"))

        doc_id_b = insert_document(self.conn, file_name_b, file_name_hash_b, hash("content b"))
        chunk_id_b = insert_document_chunk(self.conn, doc_id_b, 1, "chunk b", hash("chunk b"))
        insert_concept(self.conn, doc_id_b, chunk_id_b, "负利率", "负利率", hash("负利率"))

        # 模拟删除文档 A 时的概念清理逻辑
        concepts_a = query_concepts_by_document_id(self.conn, doc_id_a)
        concepts_a_normalized = [c.normalized_concept for c in concepts_a]

        other_doc_concepts = _query_by_not_document_id_and_in_normalized_concepts(
            self.conn, doc_id_a, concepts_a_normalized
        )
        other_doc_concepts_normalized = [c.normalized_concept for c in other_doc_concepts]

        delete_concepts = set(concepts_a_normalized) - set(other_doc_concepts_normalized)

        # "负利率"不应该被删除，因为文档 B 也在用
        self.assertNotIn("负利率", delete_concepts)
        self.assertEqual(len(delete_concepts), 0)

    def test_cleanup_logic_unique_concept(self):
        """测试独有概念的清理逻辑（应该删除）"""
        # 场景：文档 A 有独有概念"测试独有"
        # 删除文档 A 时，"测试独有"的 Chroma 数据应该被删除

        file_name_a = "test_unique_a.md"
        file_name_hash_a = hash(file_name_a)

        doc_id_a = insert_document(self.conn, file_name_a, file_name_hash_a, hash("content a"))
        chunk_id_a = insert_document_chunk(self.conn, doc_id_a, 1, "chunk a", hash("chunk a"))
        insert_concept(self.conn, doc_id_a, chunk_id_a, "测试独有", "测试独有", hash("测试独有"))

        # 模拟删除文档 A 时的概念清理逻辑
        concepts_a = query_concepts_by_document_id(self.conn, doc_id_a)
        concepts_a_normalized = [c.normalized_concept for c in concepts_a]

        other_doc_concepts = _query_by_not_document_id_and_in_normalized_concepts(
            self.conn, doc_id_a, concepts_a_normalized
        )
        other_doc_concepts_normalized = [c.normalized_concept for c in other_doc_concepts]

        delete_concepts = set(concepts_a_normalized) - set(other_doc_concepts_normalized)

        # "测试独有"应该被删除，因为只有文档 A 在用
        self.assertIn("测试独有", delete_concepts)
        self.assertEqual(len(delete_concepts), 1)


class TestIncrementalIndexLogic(unittest.TestCase):
    """测试增量索引逻辑（del/copy/new 三态）"""

    def setUp(self):
        """每个测试前初始化内存数据库"""
        self.conn = get_memory_sqlite_connection()
        create_table_init_for_memory(self.conn)

    def test_skip_state(self):
        """测试 skip 状态（文件名 + 内容都相同）"""
        file_name = "test_skip.md"
        file_name_hash = hash(file_name)
        content_hash = hash("same content")

        insert_document(self.conn, file_name, file_name_hash, content_hash)

        # 查询
        result = _query_by_file_name_hash(self.conn, file_name_hash)
        self.assertIsNotNone(result)
        self.assertEqual(result.content_hash, content_hash)
        # 应该返回 "skip"
        # （实际逻辑在 indexer.py 的 process_file 函数中）

    def test_copy_state(self):
        """测试 copy 状态（文件名不同 + 内容相同）"""
        file_name_a = "test_copy_a.md"
        file_name_hash_a = hash(file_name_a)
        content_hash = hash("same content")

        insert_document(self.conn, file_name_a, file_name_hash_a, content_hash)

        # 查询内容 hash
        result = _query_by_content_hash(self.conn, content_hash)
        self.assertIsNotNone(result)
        # 应该触发 copy 逻辑
        # （实际逻辑在 indexer.py 的 process_file 函数中）


if __name__ == "__main__":
    unittest.main()
