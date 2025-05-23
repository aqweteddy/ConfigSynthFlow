import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from config_synth_flow.base.pipeline.base_pipeline import BasePipeline
from config_synth_flow.base.pipeline.config import PipelineConfig


class TestBasePipeline:
    """Test cases for BasePipeline class."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic pipeline configuration for testing."""
        return PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={},
        )

    @pytest.fixture
    def mock_pipeline_class(self):
        """Create a mock pipeline class for testing."""

        class MockPipeline(BasePipeline):
            def post_init(self, **kwargs):
                pass

            def __call__(self, dcts):
                return dcts

        return MockPipeline

    def test_init(self, basic_config):
        """Test initialization of BasePipeline."""
        pipeline = BasePipeline(basic_config)
        assert pipeline.config == basic_config
        assert pipeline.class_name == "BasePipeline"
        assert hasattr(pipeline, "required_packages")

    def test_repr(self, basic_config):
        """Test string representation of BasePipeline."""
        pipeline = BasePipeline(basic_config)
        repr_str = repr(pipeline)
        assert "BasePipeline" in repr_str
        assert str(basic_config) in repr_str

    def test_from_config_with_pipeline_config(self, basic_config, mock_pipeline_class):
        """Test from_config with PipelineConfig."""
        with patch(
            "config_synth_flow.base.pipeline.utils._get_class",
            return_value=mock_pipeline_class(basic_config),
        ):
            pipeline = BasePipeline.from_config(basic_config)
            assert isinstance(pipeline, mock_pipeline_class)

    def test_from_config_with_list(self, basic_config, mock_pipeline_class):
        """Test from_config with list of configs."""
        with patch(
            "config_synth_flow.base.pipeline.utils._get_class",
            return_value=mock_pipeline_class(basic_config),
        ):
            result = BasePipeline.from_config([basic_config, "other_item"])
            assert isinstance(result, list)
            assert isinstance(result[0], mock_pipeline_class)
            assert result[1] == "other_item"

    def test_from_config_with_dict(self, basic_config, mock_pipeline_class):
        """Test from_config with dict of configs."""
        with patch(
            "config_synth_flow.base.pipeline.utils._get_class",
            return_value=mock_pipeline_class(basic_config),
        ):
            result = BasePipeline.from_config({"key1": basic_config, "key2": "value2"})
            assert isinstance(result, dict)
            assert isinstance(result["key1"], mock_pipeline_class)
            assert result["key2"] == "value2"

    def test_from_config_invalid_type(self):
        """Test from_config with invalid type."""
        with pytest.raises(
            ValueError, match="config must be either a list, a dict or a PipelineConfig"
        ):
            BasePipeline.from_config(123)

    def test_from_yaml(self, basic_config, mock_pipeline_class):
        """Test loading pipeline from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            yaml.dump(basic_config.model_dump(), temp_file)
            temp_file_path = temp_file.name

        try:
            with patch(
                "config_synth_flow.base.pipeline.utils._get_class",
                return_value=mock_pipeline_class(basic_config),
            ):
                pipeline = BasePipeline.from_yaml(temp_file_path)
                assert isinstance(pipeline, mock_pipeline_class)
        finally:
            os.unlink(temp_file_path)
