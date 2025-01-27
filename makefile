.PHONY: run-script

format:
	poetry run black .
	poetry run ruff check . --fix