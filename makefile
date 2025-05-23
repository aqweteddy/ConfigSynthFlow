.PHONY: run-script format test-configs test-config

format:
	poetry run ruff format .

# Run all config tests
test-cfgs:
	rm -rf tests/result
	poetry run pytest tests/test_configs.py -v

# Run a specific config test (usage: make test-config CONFIG=test_chat_gen.yml)
test-cfg:
	@if [ -z "$(CONFIG)" ]; then \
		echo "Error: CONFIG parameter is required. Example: make test-config CONFIG=test_chat_gen.yml"; \
		exit 1; \
	fi
	poetry run pytest tests/test_configs.py::test_config_execution[tests/configs/$(CONFIG)] -v

# List all available config files for testing
list-cfgs:
	@echo "Available config files:"
	@find tests/configs -name "*.yml" -o -name "*.yaml" | sort