.PHONY: run-script

format:
	poetry run black .
	poetry run ruff check . --fix
doc: 
	poetry run pdoc --html -o docs/ -d google config_synth_flow
doc-server:
	poetry run pdoc -d google config_synth_flow