# Config Tests

This directory contains tests for configuration files in the `tests/configs/` directory.

## Setup

This test suite requires the pytest-timeout plugin. Install it with:

```bash
poetry add pytest-timeout --group dev
```

## Testing Configs

The `test_configs.py` script automatically tests all YAML configuration files in the `tests/configs/` directory.

### How to Run Tests

#### Using Makefile (Recommended)

The project includes Makefile shortcuts for easy testing:

```bash
# Run all config tests
make test-configs

# Run a specific config test
make test-config CONFIG=test_chat_gen.yml

# List all available config files
make list-configs
```

#### Using pytest directly

To run all config tests:

```bash
poetry run pytest tests/test_configs.py -v
```

To run a specific config test:

```bash
poetry run pytest tests/test_configs.py::test_config_execution[tests/configs/test_chat_gen.yml] -v
```

### Adding New Test Configs

1. Create a new YAML file in the `tests/configs/` directory
2. Follow the structure of existing config files
3. Ensure the config specifies:
   - A reader
   - A writer with an output path
   - Required pipelines

### Test Output

The test will:
1. Run each config using `poetry run run-seq`
2. Verify the execution succeeds (exit code 0)
3. Check that expected output directories exist
4. Verify that expected output files are generated

### Timeout Configuration

By default, each test has:
- A 10-minute (600 second) pytest timeout
- A 9-minute (540 second) subprocess timeout

You can adjust these timeouts in the `test_configs.py` file.

### Advanced Configuration

You can modify `test_configs.py` to:

- Clean output directories before/after tests by uncommenting the cleanup code in the fixture
- Add additional validation for specific config types
- Skip certain tests by adding `@pytest.mark.skip` decorators

If a test fails, check the pytest output for detailed error messages and logs. 