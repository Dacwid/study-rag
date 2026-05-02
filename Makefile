.PHONY: install ingest ask eval compare lint test

install:
	uv sync

ingest:
	uv run module-rag ingest --raw-dir data/raw

ask:
	@echo "Usage: make ask Q=\"your question\" MODE=baseline"
	uv run module-rag ask "$(Q)" --mode $(or $(MODE),baseline)

eval:
	uv run module-rag eval

compare:
	uv run module-rag compare

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:
	uv run pytest
