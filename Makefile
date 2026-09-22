UV ?= uv

# Avoid concurrent commands changing the same project environment.
.NOTPARALLEL:
.PHONY: setup verify reproduce test

setup:
	$(UV) sync --locked --no-dev

verify:
	$(UV) run --locked --no-dev python -B scripts/verify.py

reproduce:
	$(UV) run --locked --no-dev env PYTHON_BIN=python bash scripts/reproduce.sh

test:
	$(UV) run --locked --group dev python -m pytest -q
