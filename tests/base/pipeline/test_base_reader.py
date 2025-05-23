import os
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test with the dependencies mocked
from config_synth_flow.base.io.base_reader import BaseReader
from config_synth_flow.base.pipeline import DictsGenerator, PipelineConfig


# Create a base test reader that extends BaseReader
class TestReader(BaseReader):
    """A concrete implementation of BaseReader for testing."""

    def set_test_data(self, data):
        self.test_data = data

    def read(self) -> DictsGenerator:
        yield from self.test_data


class TestBaseReader:
    @pytest.fixture
    def reader(self):
        reader = TestReader(
            PipelineConfig(
                import_path="config_synth_flow.base.io.base_reader.TestReader",
                init_kwargs={
                    "resume": False,
                },
            )
        )
        reader.test_data = []
        return reader

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_get_unique_id(self, reader):
        # Test that the same input produces the same ID
        obj1 = {"a": 1, "b": 2}
        obj2 = {"a": 1, "b": 2}
        obj3 = {"b": 2, "a": 1}  # Different order, same content

        id1 = reader.get_unique_id(obj1)
        id2 = reader.get_unique_id(obj2)
        id3 = reader.get_unique_id(obj3)

        assert id1 == id2, "Same content should produce same ID"
        assert id1 == id3, "Different order with same content should produce same ID"

        # Test that different input produces different ID
        obj4 = {"a": 1, "b": 3}
        id4 = reader.get_unique_id(obj4)
        assert id1 != id4, "Different content should produce different ID"

    def test_call_basic(self, reader):
        # Test basic functionality of call() method
        reader.set_test_data([{"key1": "value1"}, {"key2": "value2"}])

        results = list(reader())

        # Each item should have a hash_id
        assert len(results) == 2
        assert "hash_id" in results[0]
        assert "hash_id" in results[1]

        # IDs should be added to the tracking sets
        assert len(reader._unique_ids) == 2
        assert len(reader._tmp_save_processed_ids) == 2

    def test_call_with_existing_hash_id(self, reader):
        # Test when items already have hash_id
        reader.set_test_data(
            [
                {"key1": "value1", "hash_id": "preset_id_1"},
                {"key2": "value2", "hash_id": "preset_id_2"},
            ]
        )

        results = list(reader())

        # hash_id should be preserved
        assert results[0]["hash_id"] == "preset_id_1"
        assert results[1]["hash_id"] == "preset_id_2"

        # IDs should be added to the tracking sets
        assert "preset_id_1" in reader._unique_ids
        assert "preset_id_2" in reader._unique_ids

    def test_resume_functionality(self, reader, temp_dir):
        # Setup writer output path and processed IDs
        reader.set_writer_output_path(temp_dir)

        # Add some IDs to simulate previous processing
        reader._unique_ids = {"existing_id_1", "existing_id_2"}

        # Enable resume mode
        reader.resume = True

        # Test data with mix of new and previously processed items
        reader.set_test_data(
            [
                {"key1": "value1", "hash_id": "existing_id_1"},  # Should be skipped
                {"key2": "value2", "hash_id": "new_id_1"},  # Should be processed
                {"key3": "value3", "hash_id": "existing_id_2"},  # Should be skipped
                {"key4": "value4", "hash_id": "new_id_2"},  # Should be processed
            ]
        )

        results = list(reader())

        # Only new items should be in results
        assert len(results) == 2
        assert results[0]["hash_id"] == "new_id_1"
        assert results[1]["hash_id"] == "new_id_2"

    def test_add_ids(self, reader, temp_dir):
        # Setup writer output path
        reader.set_writer_output_path(temp_dir)

        # Test adding IDs
        reader.add_ids("test_id_1")
        reader.add_ids("test_id_2")

        # Check that IDs were added to the tracking sets
        assert "test_id_1" in reader._unique_ids
        assert "test_id_2" in reader._unique_ids
        assert len(reader._tmp_save_processed_ids) == 2

    def test_add_ids_file_saving(self, reader, temp_dir):
        # Setup writer output path
        reader.set_writer_output_path(temp_dir)

        # Test adding more than 1000 IDs to trigger file saving
        for i in range(1001):
            reader.add_ids(f"test_id_{i}")

        # Check that IDs were saved to file
        processed_ids_file = temp_dir / "processed_ids.txt"
        assert processed_ids_file.exists()

        # First 1000 should be saved to file, last one in memory
        with open(processed_ids_file) as f:
            saved_ids = [line.strip() for line in f.readlines() if line.strip()]

        assert len(saved_ids) == 1000
        assert len(reader._tmp_save_processed_ids) == 1
        assert reader._tmp_save_processed_ids[0] == "test_id_1000"

    def test_set_writer_output_path_new(self, reader, temp_dir):
        # Test setting the output path with no resume
        reader.set_writer_output_path(temp_dir)

        # Check that the directory was created
        assert Path(temp_dir).exists()

        # Check that an empty processed_ids.txt file was created
        processed_ids_file = temp_dir / "processed_ids.txt"
        assert processed_ids_file.exists()
        assert os.path.getsize(processed_ids_file) == 0

    def test_set_writer_output_path_resume(self, reader, temp_dir):
        # Create a processed_ids.txt file
        processed_ids_file = temp_dir / "processed_ids.txt"
        with open(processed_ids_file, "w") as f:
            f.write("test_id_1\ntest_id_2\ntest_id_3\n")

        # Set resume mode and the output path
        reader.resume = True
        reader.set_writer_output_path(temp_dir)

        # Check that the IDs were loaded
        assert len(reader._unique_ids) == 3
        assert "test_id_1" in reader._unique_ids
        assert "test_id_2" in reader._unique_ids
        assert "test_id_3" in reader._unique_ids

    def test_del_with_writer_path(self, reader, temp_dir):
        # Setup writer output path
        reader.set_writer_output_path(temp_dir)

        # Add some IDs to the temporary list
        reader._tmp_save_processed_ids = ["test_id_1", "test_id_2", "test_id_3"]

        # Call destructor manually
        reader.__del__()

        # Check that IDs were saved to file
        processed_ids_file = temp_dir / "processed_ids.txt"
        with open(processed_ids_file) as f:
            saved_ids = [line.strip() for line in f.readlines()]

        assert len(saved_ids) == 3
        assert "test_id_1" in saved_ids
        assert "test_id_2" in saved_ids
        assert "test_id_3" in saved_ids
