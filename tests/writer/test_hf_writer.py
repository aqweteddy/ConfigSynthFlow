import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from config_synth_flow.base import DictsGenerator, PipelineConfig
from config_synth_flow.writer.hf_writer import HfWriter


class TestHfWriter:
    def setup_method(self):
        """Set up test environment before each test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name)

    def teardown_method(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that HfWriter initializes correctly."""
        writer = HfWriter(
            PipelineConfig(
                import_path="config_synth_flow.writer.hf_writer.HfWriter",
                init_kwargs=dict(
                    output_path=self.output_path,
                    output_format="jsonl",
                    chunk_size=1000,
                ),
            )
        )

        assert writer.output_path == self.output_path
        assert writer.output_format == "jsonl"
        assert writer.chunk_size == 1000
        assert writer.chunk_id == 0

    def test_initialization_with_params(self):
        """Test initialization with custom parameters."""
        writer = HfWriter(
            PipelineConfig(
                import_path="config_synth_flow.writer.hf_writer.HfWriter",
                init_kwargs=dict(
                    output_path=self.output_path,
                    output_format="csv",
                    chunk_size=500,
                ),
            )
        )

        assert writer.output_path == self.output_path
        assert writer.output_format == "csv"
        assert writer.chunk_size == 500
        assert writer.chunk_id == 0

    @patch("config_synth_flow.writer.hf_writer.Dataset")
    def test_save_hf_dataset_jsonl(self, mock_dataset):
        """Test saving data as jsonl file."""
        # Setup mock
        mock_ds = Mock()
        mock_dataset.from_list.return_value = mock_ds

        writer = HfWriter(
            PipelineConfig(
                import_path="config_synth_flow.writer.hf_writer.HfWriter",
                init_kwargs=dict(
                    output_path=self.output_path,
                    output_format="json",
                    chunk_size=500,
                ),
            )
        )

        dcts = [{"text": "sample 1"}, {"text": "sample 2"}]
        writer.save_hf_dataset(dcts)

        # Assertions
        mock_dataset.from_list.assert_called_once_with(dcts)
        mock_ds.to_json.assert_called_once()

    @patch("config_synth_flow.writer.hf_writer.Dataset")
    def test_save_hf_dataset_csv(self, mock_dataset):
        """Test saving data as csv file."""
        # Setup mock
        mock_ds = Mock()
        mock_dataset.from_list.return_value = mock_ds

        writer = HfWriter(
            PipelineConfig(
                import_path="config_synth_flow.writer.hf_writer.HfWriter",
                init_kwargs=dict(
                    output_path=self.output_path,
                    output_format="csv",
                    chunk_size=500,
                ),
            )
        )

        dcts = [{"text": "sample 1"}, {"text": "sample 2"}]
        writer.save_hf_dataset(dcts)

        # Assertions
        mock_dataset.from_list.assert_called_once_with(dcts)
        mock_ds.to_csv.assert_called_once()
        mock_ds.cleanup_cache_files.assert_called_once()

    @patch("config_synth_flow.writer.hf_writer.Dataset")
    def test_call_with_chunking(self, mock_dataset):
        """Test calling with dataset requiring chunking."""
        writer = HfWriter(
            PipelineConfig(
                import_path="config_synth_flow.writer.hf_writer.HfWriter",
                init_kwargs=dict(
                    output_path=self.output_path,
                    chunk_size=2,
                    output_format="jsonl",
                ),
            )
        )
        writer.save_hf_dataset = Mock()

        # Generator with 5 items (should create 3 chunks: 2, 2, 1)
        def sample_generator():
            for i in range(5):
                yield {"text": f"sample {i}"}

        dataset = sample_generator()
        writer(dataset)

        # Should call save_hf_dataset 3 times
        assert writer.save_hf_dataset.call_count == 3

        # Check chunks
        calls = writer.save_hf_dataset.call_args_list
        assert len(calls[0][0][0]) == 2  # First chunk has 2 items
        assert len(calls[1][0][0]) == 2  # Second chunk has 2 items
        assert len(calls[2][0][0]) == 1  # Third chunk has 1 item
