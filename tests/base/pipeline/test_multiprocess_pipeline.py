import pytest

from config_synth_flow.base.pipeline import MultiProcessConfig, PipelineConfig
from config_synth_flow.base.pipeline.multiprocess_pipeline import (
    MultiProcessBasePipeline,
)


class MockMPPipeline(MultiProcessBasePipeline):
    def post_init(self, **kwargs):
        pass

    def run_each(self, dct):
        # Add a processed flag to indicate the item was processed
        dct["processed"] = True
        return dct


class TestMultiProcessBasePipeline:
    """Test cases for MultiProcessBasePipeline class."""

    @pytest.fixture
    def mp_config(self):
        """Create a multiprocess pipeline configuration for testing."""
        return PipelineConfig(
            import_path="config_synth_flow.base.pipeline.multiprocess_pipeline.MultiProcessBasePipeline",
            init_kwargs={},
            mp_cfg=MultiProcessConfig(num_proc=2, show_progress=True),
        )

    @pytest.fixture
    def mock_mp_pipeline(self):
        """Create a mock multiprocess pipeline class for testing."""
        return MockMPPipeline

    def test_init(self, mp_config):
        """Test initialization of MultiProcessBasePipeline."""
        pipeline = MultiProcessBasePipeline(mp_config)
        assert pipeline.config == mp_config
        assert pipeline.class_name == "MultiProcessBasePipeline"
        assert pipeline.config.mp_cfg.num_proc == 2

    def test_call_with_single_process(self, mp_config, mock_mp_pipeline):
        """Test __call__ method when num_proc is 1."""
        # Set num_proc to 1 to test single process path
        mp_config.mp_cfg.num_proc = 1

        pipeline = mock_mp_pipeline(mp_config)

        # Test data
        test_data = [{"id": 1}, {"id": 2}, {"id": 3}]

        # Call the pipeline
        result = list(pipeline(test_data))

        # Verify results
        assert len(result) == 3
        assert all(item["processed"] for item in result)

    def test_call_with_multiple_processes(self, mock_mp_pipeline, mp_config):
        """Test __call__ method with multiple processes."""

        # Setup test data
        test_data = [{"id": 1}, {"id": 2}, {"id": 3}]
        mp_config.mp_cfg.num_proc = 2

        pipeline = mock_mp_pipeline(mp_config)

        # Call the pipeline
        result = list(pipeline(test_data))

        # Verify results
        assert len(result) == 3
        assert all(item["processed"] for item in result)
