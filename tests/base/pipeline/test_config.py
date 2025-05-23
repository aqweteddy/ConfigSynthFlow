import tempfile
from pathlib import Path
from unittest.mock import patch

import os
import pytest

import yaml

from config_synth_flow.base.pipeline.config import (
    AsyncConfig,
    MultiProcessConfig,
    PipelineConfig,
)


class TestPipelineConfig:
    """Test cases for PipelineConfig class."""

    def test_init_with_import_path(self):
        """Test initialization with import_path."""
        config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"param1": "value1"},
        )
        assert config.import_path == "config_synth_flow.base.pipeline.base.BasePipeline"
        assert config.init_kwargs == {"param1": "value1"}
        assert config.lambda_func is None
        assert config.cfg_path is None

    def test_init_with_lambda_func(self):
        """Test initialization with lambda_func as string."""
        config = PipelineConfig(lambda_func="lambda x: x + 1", init_kwargs={})
        assert config.lambda_func == "lambda x: x + 1"
        assert config.import_path is None
        assert config.cfg_path is None

    def test_init_with_lambda_func_callable(self):
        """Test initialization with lambda_func as callable."""
        lambda_func = "lambda x: x + 1"
        config = PipelineConfig(lambda_func=lambda_func, init_kwargs={})
        assert config.lambda_func == lambda_func
        assert config.import_path is None
        assert config.cfg_path is None

    def test_init_with_cfg_path(self):
        """Test initialization with cfg_path."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_path.write_text(
                "import_path: config_synth_flow.base.pipeline.base.BasePipeline\ninit_kwargs: {}\n"
            )
            config = PipelineConfig(cfg_path=temp_path.as_posix())
        assert config.cfg_path == temp_path.as_posix()
        assert config.import_path == "config_synth_flow.base.pipeline.base.BasePipeline"
        assert config.lambda_func is None

    def test_init_with_async_cfg(self):
        """Test initialization with async_cfg."""
        async_cfg = AsyncConfig(concurrency=10, batch_size=100, show_progress=True)
        config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.async_pipeline.AsyncBasePipeline",
            init_kwargs={},
            async_cfg=async_cfg,
        )
        assert config.async_cfg == async_cfg
        assert config.async_cfg.concurrency == 10
        assert config.async_cfg.batch_size == 100
        assert config.async_cfg.show_progress is True

    def test_init_with_mp_cfg(self):
        """Test initialization with mp_cfg."""
        mp_cfg = MultiProcessConfig(num_proc=4, show_progress=True)
        config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.multiprocess_pipeline.MultiProcessBasePipeline",
            init_kwargs={},
            mp_cfg=mp_cfg,
        )
        assert config.mp_cfg == mp_cfg
        assert config.mp_cfg.num_proc == 4
        assert config.mp_cfg.show_progress is True

    def test_init_with_nested_pipeline_config(self):
        """Test initialization with nested PipelineConfig in init_kwargs."""
        nested_config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"nested_param": "nested_value"},
        )

        config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"pipeline": nested_config},
        )

        assert isinstance(config.init_kwargs["pipeline"], PipelineConfig)
        assert (
            config.init_kwargs["pipeline"].import_path
            == "config_synth_flow.base.pipeline.base.BasePipeline"
        )
        assert config.init_kwargs["pipeline"].init_kwargs == {
            "nested_param": "nested_value"
        }

    def test_init_with_list_of_pipeline_configs(self):
        """Test initialization with a list of PipelineConfig in init_kwargs."""
        nested_config1 = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"nested_param": "nested_value1"},
        )

        nested_config2 = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"nested_param": "nested_value2"},
        )

        config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"pipelines": [nested_config1, nested_config2]},
        )

        assert isinstance(config.init_kwargs["pipelines"], list)
        assert len(config.init_kwargs["pipelines"]) == 2
        assert all(
            isinstance(pc, PipelineConfig) for pc in config.init_kwargs["pipelines"]
        )
        assert (
            config.init_kwargs["pipelines"][0].init_kwargs["nested_param"]
            == "nested_value1"
        )
        assert (
            config.init_kwargs["pipelines"][1].init_kwargs["nested_param"]
            == "nested_value2"
        )

    def test_init_with_dict_of_pipeline_configs(self):
        """Test initialization with a dict of PipelineConfig in init_kwargs."""
        nested_config1 = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"nested_param": "nested_value1"},
        )

        nested_config2 = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"nested_param": "nested_value2"},
        )

        config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"pipeline_dict": {"p1": nested_config1, "p2": nested_config2}},
        )

        assert isinstance(config.init_kwargs["pipeline_dict"], dict)
        assert len(config.init_kwargs["pipeline_dict"]) == 2
        assert all(
            isinstance(pc, PipelineConfig)
            for pc in config.init_kwargs["pipeline_dict"].values()
        )
        assert (
            config.init_kwargs["pipeline_dict"]["p1"].init_kwargs["nested_param"]
            == "nested_value1"
        )
        assert (
            config.init_kwargs["pipeline_dict"]["p2"].init_kwargs["nested_param"]
            == "nested_value2"
        )

    def test_init_with_deeply_nested_config(self):
        """Test initialization with deeply nested PipelineConfig structures."""
        # Create a deeply nested configuration with multiple levels
        level3_config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"level": 3, "value": "deepest"},
        )

        level2_config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={
                "level": 2,
                "nested": level3_config,
                "other_param": "level2_value",
            },
        )

        level1_config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={
                "level": 1,
                "nested": level2_config,
                "additional": "level1_value",
            },
        )

        config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"root": level1_config},
        )

        # Verify the structure and values
        assert isinstance(config.init_kwargs["root"], PipelineConfig)
        assert config.init_kwargs["root"].init_kwargs["level"] == 1
        assert config.init_kwargs["root"].init_kwargs["additional"] == "level1_value"
        
        level2 = config.init_kwargs["root"].init_kwargs["nested"]
        assert isinstance(level2, PipelineConfig)
        assert level2.init_kwargs["level"] == 2
        assert level2.init_kwargs["other_param"] == "level2_value"
        
        level3 = level2.init_kwargs["nested"]
        assert isinstance(level3, PipelineConfig)
        assert level3.init_kwargs["level"] == 3
        assert level3.init_kwargs["value"] == "deepest"

    def test_init_with_mixed_nested_structures(self):
        """Test initialization with mixed nested structures including lists, dicts, and PipelineConfigs."""
        # Create base pipeline configs
        base_config1 = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"id": 1, "value": "first"},
        )
        
        base_config2 = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"id": 2, "value": "second"},
        )

        # Create a nested list of configs
        nested_list_config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={
                "pipeline_list": [base_config1, base_config2],
                "list_metadata": "list_config",
            },
        )

        # Create a nested dict of configs
        nested_dict_config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={
                "pipeline_dict": {"config1": base_config1, "config2": base_config2},
                "dict_metadata": "dict_config",
            },
        )

        # Create the root config with mixed structures
        config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={
                "list_based": nested_list_config,
                "dict_based": nested_dict_config,
                "direct_config": base_config1,
                "regular_list": ["item1", "item2"],
                "regular_dict": {"key1": "value1", "key2": "value2"},
            },
        )

        # Verify the complex structure
        assert isinstance(config.init_kwargs["list_based"], PipelineConfig)
        assert isinstance(config.init_kwargs["dict_based"], PipelineConfig)
        assert isinstance(config.init_kwargs["direct_config"], PipelineConfig)
        
        # Verify list-based nested structure
        list_config = config.init_kwargs["list_based"]
        assert list_config.init_kwargs["list_metadata"] == "list_config"
        assert isinstance(list_config.init_kwargs["pipeline_list"], list)
        assert len(list_config.init_kwargs["pipeline_list"]) == 2
        assert list_config.init_kwargs["pipeline_list"][0].init_kwargs["id"] == 1
        assert list_config.init_kwargs["pipeline_list"][1].init_kwargs["id"] == 2
        
        # Verify dict-based nested structure
        dict_config = config.init_kwargs["dict_based"]
        assert dict_config.init_kwargs["dict_metadata"] == "dict_config"
        assert isinstance(dict_config.init_kwargs["pipeline_dict"], dict)
        assert len(dict_config.init_kwargs["pipeline_dict"]) == 2
        assert dict_config.init_kwargs["pipeline_dict"]["config1"].init_kwargs["id"] == 1
        assert dict_config.init_kwargs["pipeline_dict"]["config2"].init_kwargs["id"] == 2
        
        # Verify regular data structures are preserved
        assert isinstance(config.init_kwargs["regular_list"], list)
        assert isinstance(config.init_kwargs["regular_dict"], dict)
        assert config.init_kwargs["regular_list"] == ["item1", "item2"]
        assert config.init_kwargs["regular_dict"] == {"key1": "value1", "key2": "value2"}

    def test_save_method(self):
        """Test save method."""
        config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"param1": "value1"},
        )

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            config.save(temp_path)

            # Verify file was created and contains expected content
            with open(temp_path, "r") as f:
                saved_config = yaml.unsafe_load(f)

            assert (
                saved_config["import_path"]
                == "config_synth_flow.base.pipeline.base.BasePipeline"
            )
            assert saved_config["init_kwargs"]["param1"] == "value1"
        finally:
            os.unlink(temp_path)

    def test_save_method_with_nested_config(self):
        """Test save method with nested PipelineConfig."""
        nested_config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"nested_param": "nested_value"},
        )

        config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline",
            init_kwargs={"pipeline": nested_config},
        )

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            config.save(temp_path)

            # Verify file was created and contains expected content
            with open(temp_path, "r") as f:
                saved_config = yaml.unsafe_load(f)

            assert (
                saved_config["import_path"]
                == "config_synth_flow.base.pipeline.base.BasePipeline"
            )
            assert "pipeline" in saved_config["init_kwargs"]
            assert (
                saved_config["init_kwargs"]["pipeline"]["import_path"]
                == "config_synth_flow.base.pipeline.base.BasePipeline"
            )
            assert (
                saved_config["init_kwargs"]["pipeline"]["init_kwargs"]["nested_param"]
                == "nested_value"
            )
        finally:
            os.unlink(temp_path)

    def test_model_post_init_default_configs(self):
        """Test that model_post_init creates default configs if not provided."""
        config = PipelineConfig(
            import_path="config_synth_flow.base.pipeline.base.BasePipeline"
        )
        assert isinstance(config.async_cfg, AsyncConfig)
        assert isinstance(config.mp_cfg, MultiProcessConfig)

    def test_model_post_init_error(self):
        """Test that model_post_init raises ValueError if no import_path, lambda_func, or cfg_path."""
        with pytest.raises(ValueError):
            PipelineConfig()

    def test_init_with_invalid_cfg_file(self):
        """Test initialization with invalid configuration file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_path.write_text("invalid yaml content that is not a dictionary")

            with pytest.raises(ValueError):
                PipelineConfig(cfg_path=temp_path.as_posix())


class TestAsyncConfig:
    """Test cases for AsyncConfig class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = AsyncConfig()

    def test_init_with_values(self):
        """Test initialization with specific values."""
        config = AsyncConfig(concurrency=5, batch_size=100, show_progress=True)
        assert config.concurrency == 5
        assert config.batch_size == 100
        assert config.show_progress is True


class TestMultiProcessConfig:
    """Test cases for MultiProcessConfig class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = MultiProcessConfig()
        assert config.num_proc is None
        assert config.show_progress is None

    def test_init_with_values(self):
        """Test initialization with specific values."""
        config = MultiProcessConfig(num_proc=4, show_progress=True)
        assert config.num_proc == 4
        assert config.show_progress is True
