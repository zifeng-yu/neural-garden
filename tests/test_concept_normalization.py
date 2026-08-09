"""
概念归一化单元测试

测试覆盖：
1. 向量相似度搜索（search_by_threshold_concept）
2. LLM 概念判断（extract_max_similarity_concept）
3. 归一化逻辑（resolve_concept）
"""

import pytest
from unittest.mock import patch, MagicMock


class TestSearchByThresholdConcept:
    """向量相似度搜索测试"""
    
    @patch('src.vector_store.query_dao.get_collection')
    @patch('src.vector_store.query_dao.get_embedding')
    def test_search_similar_found(self, mock_get_embedding, mock_get_collection):
        """测试找到相似概念"""
        # Mock 配置
        mock_embedding = [0.1] * 1024
        mock_get_embedding.return_value = mock_embedding
        
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        
        # Mock ChromaDB 返回结果
        mock_collection.query.return_value = {
            "documents": [["中央银行"]],
            "metadatas": [[{"source": ["01-金融深潜 - 日本负利率.md"]}]],
            "distances": [[0.08]],  # 相似度 = 1 - 0.08 = 0.92
        }
        
        from src.vector_store.query_dao import search_by_threshold_concept
        
        results = search_by_threshold_concept("央行", threshold=0.85)
        
        # 验证
        assert results is not None
        assert len(results) == 1
        assert results[0].conceptName == "中央银行"
    
    @patch('src.vector_store.query_dao.get_collection')
    @patch('src.vector_store.query_dao.get_embedding')
    def test_search_no_similar(self, mock_get_embedding, mock_get_collection):
        """测试没有相似概念（低于阈值）"""
        mock_embedding = [0.1] * 1024
        mock_get_embedding.return_value = mock_embedding
        
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        
        # Mock ChromaDB 返回结果（相似度低于阈值）
        mock_collection.query.return_value = {
            "documents": [["货币政策"]],
            "metadatas": [[{"source": ["test.md"]}]],
            "distances": [[0.3]],  # 相似度 = 1 - 0.3 = 0.7 < 0.85
        }
        
        from src.vector_store.query_dao import search_by_threshold_concept
        
        results = search_by_threshold_concept("央行", threshold=0.85)
        
        # 验证：低于阈值，返回空列表
        assert results == []
    
    @patch('src.vector_store.query_dao.get_collection')
    @patch('src.vector_store.query_dao.get_embedding')
    def test_search_empty_result(self, mock_get_embedding, mock_get_collection):
        """测试空结果"""
        mock_embedding = [0.1] * 1024
        mock_get_embedding.return_value = mock_embedding
        
        mock_collection = MagicMock()
        mock_get_collection.return_value = mock_collection
        
        # Mock ChromaDB 返回空结果
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        
        from src.vector_store.query_dao import search_by_threshold_concept
        
        results = search_by_threshold_concept("未知概念", threshold=0.85)
        
        assert results == []
    
    @patch('src.vector_store.query_dao.get_collection')
    @patch('src.vector_store.query_dao.get_embedding')
    def test_search_embedding_none(self, mock_get_embedding, mock_get_collection):
        """测试 Embedding 生成失败"""
        mock_get_embedding.return_value = None
        
        from src.vector_store.query_dao import search_by_threshold_concept
        
        results = search_by_threshold_concept("测试", threshold=0.85)
        
        assert results is None


class TestExtractMaxSimilarityConcept:
    """LLM 概念判断测试"""
    
    @patch('src.knowledgeGraph.graph_extractor.fetchLLM')
    def test_judge_same_concept(self, mock_fetchLLM):
        """测试判断为同一概念"""
        mock_fetchLLM.return_value = ["中央银行"]
        
        from src.knowledgeGraph.graph_extractor import extract_max_similarity_concept
        
        result = extract_max_similarity_concept("央行", ["中央银行", "货币政策"])
        
        assert result == ["中央银行"]
        assert mock_fetchLLM.called
    
    @patch('src.knowledgeGraph.graph_extractor.fetchLLM')
    def test_judge_different_concept(self, mock_fetchLLM):
        """测试判断为不同概念"""
        mock_fetchLLM.return_value = []
        
        from src.knowledgeGraph.graph_extractor import extract_max_similarity_concept
        
        result = extract_max_similarity_concept("央行", ["货币政策", "财政政策"])
        
        assert result == []
    
    @patch('src.knowledgeGraph.graph_extractor.fetchLLM')
    def test_judge_with_abbreviation(self, mock_fetchLLM):
        """测试简称判断"""
        mock_fetchLLM.return_value = ["量化宽松"]
        
        from src.knowledgeGraph.graph_extractor import extract_max_similarity_concept
        
        result = extract_max_similarity_concept("QE", ["量化宽松", "负利率"])
        
        assert result == ["量化宽松"]


class TestResolveConcept:
    """概念归一化集成测试"""
    
    @patch('src.knowledgeGraph.knowledge_graph.search_by_threshold_concept')
    @patch('src.knowledgeGraph.knowledge_graph.extract_max_similarity_concept')
    def test_resolve_success(self, mock_extract, mock_search):
        """测试归一化成功"""
        # Mock 向量搜索结果
        mock_concept_do = MagicMock()
        mock_concept_do.conceptName = "中央银行"
        mock_search.return_value = [mock_concept_do]
        
        # Mock LLM 判断结果
        mock_extract.return_value = ["中央银行"]
        
        from src.knowledgeGraph.knowledge_graph import resolve_concept
        
        result = resolve_concept("央行")
        
        assert result["is_relove"] is True
        assert result["resolve_concept"] == "中央银行"
    
    @patch('src.knowledgeGraph.knowledge_graph.search_by_threshold_concept')
    def test_resolve_no_similar(self, mock_search):
        """测试没有相似概念"""
        mock_search.return_value = []
        
        from src.knowledgeGraph.knowledge_graph import resolve_concept
        
        result = resolve_concept("未知概念")
        
        assert result["is_relove"] is False
        assert "resolve_concept" not in result
    
    @patch('src.knowledgeGraph.knowledge_graph.search_by_threshold_concept')
    @patch('src.knowledgeGraph.knowledge_graph.extract_max_similarity_concept')
    def test_resolve_llm_judge_different(self, mock_extract, mock_search):
        """测试 LLM 判断为不同概念"""
        mock_concept_do = MagicMock()
        mock_concept_do.conceptName = "货币政策"
        mock_search.return_value = [mock_concept_do]
        
        # LLM 判断为不同概念
        mock_extract.return_value = []
        
        from src.knowledgeGraph.knowledge_graph import resolve_concept
        
        result = resolve_concept("央行")
        
        assert result["is_relove"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
