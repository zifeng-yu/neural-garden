"""
Neural Garden 单元测试套件 - 核心功能模块

测试范围：
1. document/splitter.py - Markdown 分割
2. util/getHashValue.py - Hash 计算
3. similarity.py - 相似度计算
"""

import os
import sys
import unittest
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document.splitter import markdown_spilt
from src.util.getHashValue import get_hash_value as hash
from src.similarity import calculate_similarity


class TestMarkdownSplitter(unittest.TestCase):
    """测试 Markdown 文档分割"""

    def test_split_simple_markdown(self):
        """测试简单 Markdown 分割"""
        markdown = """# 标题 1

这是第一段内容。

## 标题 1.1

这是第二段内容。

## 标题 1.2

这是第三段内容。
"""
        chunks = markdown_spilt(markdown)
        self.assertGreater(len(chunks), 0)
        # 应该至少分割成 3 个 chunk
        self.assertGreaterEqual(len(chunks), 3)

    def test_split_empty_markdown(self):
        """测试空 Markdown"""
        chunks = markdown_spilt("")
        self.assertEqual(len(chunks), 0)

    def test_split_no_headers(self):
        """测试无标题的 Markdown"""
        markdown = """这是第一段内容。

这是第二段内容。

这是第三段内容。
"""
        chunks = markdown_spilt(markdown)
        # 应该会按长度分割
        self.assertGreater(len(chunks), 0)

    def test_split_preserves_content(self):
        """测试分割后内容完整性"""
        markdown = """# 金融深潜

负利率是央行实施的一种货币政策。

## 背景

日本在 1990 年代实施了负利率政策。

## 影响

负利率导致银行利润下降。
"""
        chunks = markdown_spilt(markdown)
        # 合并所有 chunk，应该包含原文关键内容
        all_content = " ".join(chunks)
        self.assertIn("负利率", all_content)
        self.assertIn("日本", all_content)

    def test_split_long_content(self):
        """测试长内容分割"""
        # 生成一个长段落
        long_paragraph = "这是一段很长的内容。" * 100
        markdown = f"# 标题\n\n{long_paragraph}"
        chunks = markdown_spilt(markdown)
        # 长内容应该被分割成多个 chunk
        self.assertGreater(len(chunks), 1)


class TestHashValue(unittest.TestCase):
    """测试 Hash 计算"""

    def test_hash_consistency(self):
        """测试相同内容 hash 一致"""
        content = "测试内容"
        hash1 = hash(content)
        hash2 = hash(content)
        self.assertEqual(hash1, hash2)

    def test_hash_different_content(self):
        """测试不同内容 hash 不同"""
        hash1 = hash("内容 1")
        hash2 = hash("内容 2")
        self.assertNotEqual(hash1, hash2)

    def test_hash_format(self):
        """测试 hash 格式（MD5 应该是 32 位十六进制）"""
        result = hash("测试")
        self.assertEqual(len(result), 32)
        # 应该是十六进制
        int(result, 16)  # 不应该抛出异常

    def test_hash_empty_string(self):
        """测试空字符串 hash"""
        result = hash("")
        self.assertEqual(len(result), 32)

    def test_hash_special_characters(self):
        """测试特殊字符 hash"""
        content = "特殊字符：!@#$%^&*()_+{}|:<>?"
        result = hash(content)
        self.assertEqual(len(result), 32)

    def test_hash_chinese(self):
        """测试中文字符 hash"""
        content = "中文测试内容"
        result = hash(content)
        self.assertEqual(len(result), 32)


class TestSimilarityCalculation(unittest.TestCase):
    """测试相似度计算"""

    def test_similarity_identical_vectors(self):
        """测试相同向量的相似度（应该为 1.0）"""
        embeddings = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
        result = calculate_similarity(embeddings)
        # 对角线应该是 1.0（自己和自己相似度）
        self.assertAlmostEqual(result[0][0], 1.0, places=5)
        self.assertAlmostEqual(result[1][1], 1.0, places=5)
        # 相同向量相似度应该接近 1.0
        self.assertAlmostEqual(result[0][1], 1.0, places=5)
        self.assertAlmostEqual(result[1][0], 1.0, places=5)

    def test_similarity_orthogonal_vectors(self):
        """测试正交向量的相似度（应该为 0.0）"""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        result = calculate_similarity(embeddings)
        # 正交向量相似度应该为 0
        self.assertAlmostEqual(result[0][1], 0.0, places=5)
        self.assertAlmostEqual(result[1][0], 0.0, places=5)

    def test_similarity_opposite_vectors(self):
        """测试相反向量的相似度（应该为 -1.0）"""
        embeddings = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
        result = calculate_similarity(embeddings)
        # 相反向量相似度应该为 -1
        self.assertAlmostEqual(result[0][1], -1.0, places=5)
        self.assertAlmostEqual(result[1][0], -1.0, places=5)

    def test_similarity_matrix_shape(self):
        """测试相似度矩阵形状"""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        result = calculate_similarity(embeddings)
        # 3 个向量应该返回 3x3 矩阵
        self.assertEqual(result.shape, (3, 3))

    def test_similarity_symmetric(self):
        """测试相似度矩阵对称性"""
        embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
        result = calculate_similarity(embeddings)
        # 相似度矩阵应该是对称的
        self.assertAlmostEqual(result[0][1], result[1][0], places=5)
        self.assertAlmostEqual(result[0][2], result[2][0], places=5)
        self.assertAlmostEqual(result[1][2], result[2][1], places=5)


if __name__ == "__main__":
    unittest.main()
