from unittest.mock import AsyncMock, patch

import pytest
from litellm import ModelResponse, TextCompletionResponse

from config_synth_flow.base.pipeline.chat_pipeline import AsyncChatBasePipeline
from config_synth_flow.base.pipeline.config import AsyncConfig, PipelineConfig


class TestAsyncChatBasePipeline:
    """Test cases for AsyncChatBasePipeline class."""

    @pytest.fixture
    def chat_config(self):
        """Create a chat pipeline configuration for testing."""
        return PipelineConfig(
            import_path="AsyncChatBasePipeline",
            init_kwargs={"litellm_kwargs": {"model": "gpt-4"}},
            async_cfg=AsyncConfig(concurrency=5, batch_size=20),
        )

    @pytest.fixture
    def chat_pipeline(self, chat_config):
        """Create an AsyncChatBasePipeline instance for testing."""
        return AsyncChatBasePipeline(chat_config)

    def test_post_init(self, chat_config):
        """Test post_init method correctly sets litellm_kwargs."""
        pipeline = AsyncChatBasePipeline(chat_config)
        assert pipeline.litellm_kwargs == {"model": "gpt-4"}

    @pytest.mark.asyncio
    @patch("config_synth_flow.base.pipeline.chat_pipeline.acompletion")
    async def test_chat(self, mock_acompletion, chat_pipeline):
        """Test chat method calls acompletion with correct parameters."""
        # Setup mock response
        mock_response = AsyncMock(spec=ModelResponse)
        mock_acompletion.return_value = mock_response

        # Test data
        messages = [{"role": "user", "content": "Hello"}]

        # Call method
        result = await chat_pipeline.chat(messages)

        # Verify acompletion was called with correct args
        mock_acompletion.assert_awaited_once_with(messages=messages, model="gpt-4")

        # Verify result
        assert result == mock_response

    @pytest.mark.asyncio
    @patch("config_synth_flow.base.pipeline.chat_pipeline.atext_completion")
    async def test_completion(self, mock_atext_completion, chat_pipeline):
        """Test completion method calls atext_completion with correct parameters."""
        # Setup mock response
        mock_response = AsyncMock(spec=TextCompletionResponse)
        mock_atext_completion.return_value = mock_response

        # Test data
        prompt = "Complete this sentence:"

        # Call method
        result = await chat_pipeline.completion(prompt)

        # Verify atext_completion was called with correct args
        mock_atext_completion.assert_awaited_once_with(prompt=prompt, model="gpt-4")

        # Verify result
        assert result == mock_response
