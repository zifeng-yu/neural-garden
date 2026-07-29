"""
知识提取器模块的单元测试
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.knowledge.knowledge_extractor import (
    LLMException,
    Response_extractor,
    extractor_by_llm,
)


class TestResponseExtractor:
    """测试 Response_extractor Pydantic 模型"""

    def test_valid_response(self):
        """测试有效的响应"""
        response = Response_extractor(
            title="测试标题",
            summary="这是一个有效的摘要，长度足够满足 Pydantic 验证要求，能够概括文档的主要内容。",
            keywords=["测试", "提取", "知识单元"],
        )

        assert response.title == "测试标题"
        assert len(response.summary) > 30
        assert len(response.keywords) == 3

    def test_empty_title(self):
        """测试空标题（应该失败）"""
        with pytest.raises(ValidationError):
            Response_extractor(
                title="",
                summary="摘要",
                keywords=["测试"],
            )

    def test_title_too_long(self):
        """测试标题过长（应该失败）"""
        with pytest.raises(ValidationError):
            Response_extractor(
                title="这是一" * 20,  # 60 字符，超过 30 字限制
                summary="摘要",
                keywords=["测试"],
            )

    def test_summary_too_long(self):
        """测试摘要过长（应该失败）"""
        with pytest.raises(ValidationError):
            Response_extractor(
                title="标题",
                summary="这是一个非常长的摘要。" * 100,  # 超过 500 字
                keywords=["测试"],
            )

    def test_keywords_max_count(self):
        """测试关键词最大数量（10 个）"""
        response = Response_extractor(
            title="标题",
            summary="摘要",
            keywords=["key" + str(i) for i in range(10)],
        )

        assert len(response.keywords) == 10

    def test_keywords_too_many(self):
        """测试关键词过多（应该失败）"""
        with pytest.raises(ValidationError):
            Response_extractor(
                title="标题",
                summary="摘要",
                keywords=["key" + str(i) for i in range(11)],  # 11 个，超过限制
            )

    def test_keywords_empty_list(self):
        """测试空关键词列表（允许）"""
        response = Response_extractor(
            title="标题",
            summary="摘要",
            keywords=[],
        )

        assert len(response.keywords) == 0


class TestExtractorByLLM:
    """测试 extractor_by_llm 函数（使用 Mock）"""

    @patch("src.knowledge.knowledge_extractor.MultiModalConversation.call")
    def test_extractor_success(self, mock_call):
        """测试成功提取"""
        # 准备 Mock 响应（正确的嵌套结构）
        mock_message = MagicMock()
        mock_message.content = [
            {
                "text": '{"title": "Mock 标题", "summary": "这是一个 Mock 摘要，长度足够满足要求。", "keywords": ["mock", "test"]}'
            }
        ]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_output = MagicMock()
        mock_output.choices = [mock_choice]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output = mock_output

        mock_call.return_value = mock_response

        # 执行测试
        result = extractor_by_llm(content="测试内容")

        # 验证结果
        assert result.title == "Mock 标题"
        assert "Mock 摘要" in result.summary
        assert len(result.keywords) == 2
        assert "mock" in result.keywords

        # 验证 Mock 被调用
        mock_call.assert_called_once()

    @patch("src.knowledge.knowledge_extractor.MultiModalConversation.call")
    def test_extractor_api_error(self, mock_call):
        """测试 API 调用失败"""
        # 准备 Mock 响应（模拟 API 错误）
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.message = "Bad Request"
        mock_call.return_value = mock_response

        # 执行测试并验证异常
        with pytest.raises(LLMException) as exc_info:
            extractor_by_llm(content="测试内容")

        assert "400" in str(exc_info.value)
        assert "Bad Request" in str(exc_info.value)

    @patch("src.knowledge.knowledge_extractor.MultiModalConversation.call")
    def test_extractor_invalid_article(self, mock_call):
        """测试无效文章输入"""
        # 准备 Mock 响应（返回无效文章标记）
        mock_message = MagicMock()
        mock_message.content = [
            {"text": '{"title": "无效文章输入", "summary": "", "keywords": []}'}
        ]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_output = MagicMock()
        mock_output.choices = [mock_choice]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output = mock_output

        mock_call.return_value = mock_response

        # 执行测试并验证异常
        with pytest.raises(LLMException):
            extractor_by_llm(content="")

    @patch("src.knowledge.knowledge_extractor.MultiModalConversation.call")
    def test_extractor_invalid_json(self, mock_call):
        """测试无效 JSON 响应"""
        # 准备 Mock 响应（返回无效 JSON）
        mock_message = MagicMock()
        mock_message.content = [{"text": "invalid json"}]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_output = MagicMock()
        mock_output.choices = [mock_choice]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output = mock_output

        mock_call.return_value = mock_response

        # 执行测试并验证异常
        with pytest.raises(Exception):  # json.loads 会抛出 JSONDecodeError
            extractor_by_llm(content="测试内容")

    @patch("src.knowledge.knowledge_extractor.MultiModalConversation.call")
    def test_extractor_retry(self, mock_call):
        """测试重试机制"""
        # 准备 Mock 响应（第一次失败，第二次成功）
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.message = "Internal Server Error"

        # 成功的响应
        mock_message = MagicMock()
        mock_message.content = [
            {
                "text": '{"title": "成功标题", "summary": "成功的摘要，长度足够。", "keywords": ["success"]}'
            }
        ]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_output = MagicMock()
        mock_output.choices = [mock_choice]

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.output = mock_output

        mock_call.side_effect = [mock_response_fail, mock_response_success]

        # 执行测试
        result = extractor_by_llm(content="测试内容")

        # 验证结果
        assert result.title == "成功标题"

        # 验证 Mock 被调用了 2 次（重试）
        assert mock_call.call_count == 2

    @patch("src.knowledge.knowledge_extractor.MultiModalConversation.call")
    def test_extractor_prompt_structure(self, mock_call):
        """测试 Prompt 结构正确性"""
        # 准备 Mock 响应
        mock_message = MagicMock()
        mock_message.content = [
            {"text": '{"title": "标题", "summary": "摘要", "keywords": ["key"]}'}
        ]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_output = MagicMock()
        mock_output.choices = [mock_choice]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output = mock_output

        mock_call.return_value = mock_response

        # 执行测试
        extractor_by_llm(content="测试内容")

        # 验证 Prompt 结构
        call_args = mock_call.call_args
        messages = call_args.kwargs["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "提取下面文章的知识结构" in messages[1]["content"]

    @patch("src.knowledge.knowledge_extractor.MultiModalConversation.call")
    def test_extractor_temperature(self, mock_call):
        """测试 temperature 参数设置"""
        # 准备 Mock 响应
        mock_message = MagicMock()
        mock_message.content = [
            {"text": '{"title": "标题", "summary": "摘要", "keywords": ["key"]}'}
        ]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_output = MagicMock()
        mock_output.choices = [mock_choice]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.output = mock_output

        mock_call.return_value = mock_response

        # 执行测试
        extractor_by_llm(content="测试内容")

        # 验证 temperature 参数
        call_args = mock_call.call_args
        assert call_args.kwargs["temperature"] == 0.1
        assert call_args.kwargs["enable_thinking"] is False
        assert call_args.kwargs["response_format"] == {"type": "json_object"}
