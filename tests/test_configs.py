import subprocess
import time
from pathlib import Path

import pytest
import yaml


def get_all_config_files():
    """Find all YAML/YML config files in the tests/configs directory."""
    configs_dir = Path("tests/configs")
    config_files = []

    for file in configs_dir.glob("**/*.yml"):
        config_files.append(str(file))

    for file in configs_dir.glob("**/*.yaml"):
        config_files.append(str(file))

    return config_files


@pytest.fixture(scope="function")
def clean_output_directories():
    """Clean up output directories before and after tests."""
    # Setup: clean existing output directories if needed
    output_dirs = []

    # Find all output paths in config files
    for config_file in get_all_config_files():
        try:
            with open(config_file, "r") as f:
                # Parse YAML content
                config_data = yaml.safe_load(f)

                # Look for output_path in writer section
                if (
                    config_data
                    and "init_kwargs" in config_data
                    and "writer" in config_data["init_kwargs"]
                    and "init_kwargs" in config_data["init_kwargs"]["writer"]
                    and "output_path" in config_data["init_kwargs"]["writer"]["init_kwargs"]
                ):
                    output_path = config_data["init_kwargs"]["writer"]["init_kwargs"]["output_path"]
                    output_dirs.append(Path(output_path))
        except Exception as e:
            print(f"Error reading config file {config_file}: {e}")

    yield



@pytest.mark.parametrize("config_file", get_all_config_files())
@pytest.mark.timeout(600)  # 10 minute timeout per test
def test_config_execution(config_file, clean_output_directories):
    """Test that a config file can be executed successfully."""
    # Print which config is being tested
    print(f"\nTesting config: {config_file}")

    # Extract expected output path from config
    output_path = None
    try:
        with open(config_file) as f:
            config_data = yaml.safe_load(f)
            if (
                config_data
                and "init_kwargs" in config_data
                and "writer" in config_data["init_kwargs"]
                and "init_kwargs" in config_data["init_kwargs"]["writer"]
                and "output_path" in config_data["init_kwargs"]["writer"]["init_kwargs"]
            ):
                output_path = Path(
                    config_data["init_kwargs"]["writer"]["init_kwargs"]["output_path"]
                )
    except Exception as e:
        print(f"Error reading config file {config_file}: {e}")

    # Run the config using the poetry run command with timeout
    start_time = time.time()

    try:
        result = subprocess.run(
            ["poetry", "run", "run-seq", config_file],
            capture_output=True,
            text=True,
            check=False,
            timeout=540,  # 9 minute subprocess timeout (less than pytest timeout)
        )
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        pytest.fail(f"Config execution timed out after {elapsed:.1f} seconds: {config_file}")

    # Print the output for debugging
    if result.stdout:
        print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")

    elapsed = time.time() - start_time
    print(f"Config executed in {elapsed:.1f} seconds")

    # Assert that the command was successful
    assert result.returncode == 0, f"Failed to run config {config_file}: {result.stderr}"

    # Check that output directory exists if specified
    if output_path:
        assert output_path.exists(), f"Output directory {output_path} was not created"

        # Check that config.yml was saved in output directory
        config_copy = output_path / "config.yml"
        assert config_copy.exists(), f"Config file was not saved to {config_copy}"

        # Check for other expected output files based on the specific config
        # This section can be expanded based on the expected outputs of each config type

        if config_file.endswith("test_chat_gen.yml"):
            # For chat generation configs, check for expected output files
            # Adjust these checks based on what your specific configs should produce
            # check number of files in the output directory. At least 1 file for config.yml and 1 file for output.
            assert len(list(output_path.glob("*.*"))) > 1, f"No output files found in {output_path}"
