from unittest.mock import patch

import pytest

from config_synth_flow.base.pipeline.async_pipeline import AsyncBasePipeline
from config_synth_flow.base.pipeline.config import PipelineConfig


class TestAsyncBasePipeline:
    """Test cases for AsyncBasePipeline class."""

    @pytest.fixture
    def async_config(self):
        """Create an async pipeline configuration for testing."""
        from config_synth_flow.base.pipeline.config import AsyncConfig

        return PipelineConfig(
            import_path="config_synth_flow.base.pipeline.async_pipeline.AsyncBasePipeline",
            init_kwargs={},
            async_cfg=AsyncConfig(concurrency=5, batch_size=20),
        )

    @pytest.fixture
    def mock_async_pipeline(self):
        """Create a mock async pipeline class for testing."""

        class MockAsyncPipeline(AsyncBasePipeline):
            def post_init(self, **kwargs):
                pass

            async def run_each(self, dct):
                return {**dct, "processed": True}

        return MockAsyncPipeline

    def test_init(self, async_config):
        """Test initialization of AsyncBasePipeline."""
        pipeline = AsyncBasePipeline(async_config)
        assert pipeline.config == async_config
        assert pipeline.class_name == "AsyncBasePipeline"
        assert pipeline.config.async_cfg.concurrency == 5
        assert pipeline.config.async_cfg.batch_size == 20

    def test_semaphore_property(self, async_config):
        """Test semaphore property."""
        pipeline = AsyncBasePipeline(async_config)
        semaphore = pipeline.semaphore
        assert semaphore._value == 5

    def test_run_each_not_implemented(self, async_config):
        """Test run_each raises NotImplementedError if not overridden."""
        pipeline = AsyncBasePipeline(async_config)
        with pytest.raises(
            NotImplementedError, match="AsyncBasePipeline is missing `run_each` method"
        ):
            import asyncio

            asyncio.run(pipeline.run_each({"test": "data"}))

    @pytest.mark.asyncio
    async def test_async_call(self, async_config, mock_async_pipeline):
        """Test _async_call method."""
        pipeline = mock_async_pipeline(async_config)
        test_data = [{"id": 1}, {"id": 2}, {"id": 3}]

        result = await pipeline._async_call(test_data)

        assert len(result) == 3
        assert all("processed" in item for item in result)

    def test_call_method(self, async_config, mock_async_pipeline):
        """Test __call__ method."""
        pipeline = mock_async_pipeline(async_config)
        test_data = [{"id": i} for i in range(5)]

        result = list(pipeline(test_data))

        assert len(result) == 5
        assert result[4]["id"] == 4
        assert result[4]["processed"]
