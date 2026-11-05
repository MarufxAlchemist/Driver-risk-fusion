# Targets work with GNU make (Mac/Linux, or mingw32-make on Windows).
# Windows without make: use .\make.ps1 <target>, which runs the same commands.

.PHONY: setup lint format test demo-stub serve

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

test:
	uv run pytest -m "not slow"

demo-stub:
	uv run python scripts/demo_stub.py

serve:
	uv run python scripts/mock_server.py
