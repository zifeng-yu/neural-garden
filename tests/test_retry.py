"""
重试工具单元测试

测试覆盖：
1. retry 装饰器基本功能
2. 指数退避延迟
3. 异常类型过滤
4. 超时控制
5. 回调函数
6. LLM 调用重试集成
"""

import time
from unittest.mock import MagicMock, call, patch

import pytest

from src.util.llmException import LLMException
from src.util.retryUtil import no_retry, retry, retry_with_fixed_delay


class TestRetryBasic:
    """重试基本功能测试"""

    def test_retry_success_first_try(self):
        """第一次调用成功"""
        call_count = 0

        @retry(times=3)
        def func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = func()

        assert result == "success"
        assert call_count == 1

    def test_retry_success_second_try(self):
        """第二次调用成功"""
        call_count = 0

        @retry(times=3)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("error")
            return "success"

        result = func()

        assert result == "success"
        assert call_count == 2

    def test_retry_fail_all_attempts(self):
        """所有重试都失败"""
        call_count = 0

        @retry(times=3)
        def func():
            nonlocal call_count
            call_count += 1
            raise ValueError("always fail")

        with pytest.raises(ValueError):
            func()

        assert call_count == 3


class TestRetryDelay:
    """重试延迟测试"""

    def test_exponential_backoff(self):
        """指数退避延迟"""
        delays = []

        @retry(times=3, delay=0.1, backoff=2.0)
        def func():
            raise ValueError("fail")

        # Mock time.sleep 来捕获延迟
        with patch("src.util.retryUtil.time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                func()

            # 应该有 2 次 sleep（第 1 次后和第 2 次后）
            assert mock_sleep.call_count == 2

            # 延迟应该是 0.1, 0.2（指数退避）
            calls = mock_sleep.call_args_list
            assert calls[0] == call(0.1)
            assert calls[1] == call(0.2)

    def test_fixed_delay(self):
        """固定延迟（无退避）"""

        @retry_with_fixed_delay(times=3, delay=0.5)
        def func():
            raise ValueError("fail")

        with patch("src.util.retryUtil.time.sleep") as mock_sleep:
            with pytest.raises(ValueError):
                func()

            # 应该有 2 次 sleep，每次都是 0.5s
            assert mock_sleep.call_count == 2
            assert all(call_arg == call(0.5) for call_arg in mock_sleep.call_args_list)


class TestRetryExceptions:
    """异常类型过滤测试"""

    def test_retry_only_specified_exceptions(self):
        """只重试指定异常"""
        call_count = 0

        @retry(times=3, exceptions=(ValueError,))
        def func():
            nonlocal call_count
            call_count += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            func()

        # TypeError 不在重试列表，应该只调用 1 次
        assert call_count == 1

    def test_retry_multiple_exception_types(self):
        """重试多种异常类型"""
        call_count = 0
        exception_sequence = [ValueError, TimeoutError, "success"]

        @retry(times=3, exceptions=(ValueError, TimeoutError))
        def func():
            nonlocal call_count
            call_count += 1
            exc = exception_sequence[call_count - 1]
            if isinstance(exc, type):
                raise exc("error")
            return exc

        result = func()

        assert result == "success"
        assert call_count == 3


class TestRetryCallbacks:
    """回调函数测试"""

    def test_retry_callback(self):
        """重试回调函数"""
        retry_events = []

        def on_retry(retry_count, exception, next_delay):
            retry_events.append(
                {
                    "count": retry_count,
                    "exception": type(exception).__name__,
                    "delay": next_delay,
                }
            )

        @retry(times=3, retry_callback=on_retry)
        def func():
            raise ValueError("fail")

        with patch("src.util.retryUtil.time.sleep"):
            with pytest.raises(ValueError):
                func()

        # 应该有 2 次重试回调（第 1 次和第 2 次失败后）
        assert len(retry_events) == 2
        assert retry_events[0]["count"] == 1
        assert retry_events[1]["count"] == 2

    def test_success_callback(self):
        """成功回调函数"""
        success_events = []

        def on_success(result, total_time):
            success_events.append({"result": result, "total_time": total_time})

        @retry(times=3, success_callback=on_success)
        def func():
            return "success"

        result = func()

        assert result == "success"
        assert len(success_events) == 1
        assert success_events[0]["result"] == "success"


class TestNoRetry:
    """禁用重试测试"""

    def test_no_retry_decorator(self):
        """no_retry 装饰器标记"""

        @no_retry
        def func():
            return "no retry"

        assert func._no_retry is True
        assert func() == "no retry"


class TestLLMRetry:
    """LLM 调用重试集成测试"""

    @patch("dashscope.MultiModalConversation.call")
    def test_llm_retry_on_500_error(self, mock_call):
        """LLM 调用遇到 500 错误时重试"""
        # 模拟前 2 次 500 错误，第 3 次成功
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.message = "Internal Server Error"

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.output.choices = [
            MagicMock(message=MagicMock(content=[{"text": "success"}]))
        ]

        mock_call.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success,
        ]

        from src.util.callDashscopellm import generate

        result = generate("system", "user")

        assert result == "success"
        assert mock_call.call_count == 3

    @patch("dashscope.MultiModalConversation.call")
    def test_llm_no_retry_on_400_error(self, mock_call):
        """LLM 调用遇到 400 错误时不重试"""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.message = "Bad Request"

        mock_call.return_value = mock_response

        from src.util.callDashscopellm import generate

        # 400 错误抛出 RuntimeError，不进入重试
        with pytest.raises(RuntimeError) as exc_info:
            generate("system", "user")

        assert "400" in str(exc_info.value)
        assert mock_call.call_count == 1  # 只调用 1 次，不重试

    @patch("dashscope.MultiModalConversation.call")
    def test_llm_retry_on_429_rate_limit(self, mock_call):
        """LLM 调用遇到 429 速率限制时重试"""
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 429
        mock_response_fail.message = "Rate Limit Exceeded"

        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.output.choices = [
            MagicMock(message=MagicMock(content=[{"text": "success"}]))
        ]

        mock_call.side_effect = [
            mock_response_fail,
            mock_response_success,
        ]

        from src.util.callDashscopellm import generate

        result = generate("system", "user")

        assert result == "success"
        assert mock_call.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
