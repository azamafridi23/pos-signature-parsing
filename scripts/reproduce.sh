#!/usr/bin/env bash
set -euo pipefail
PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONDONTWRITEBYTECODE=1
cd "$PACKAGE_ROOT"
"$PYTHON_BIN" scripts/verify.py --inputs-only
"$PYTHON_BIN" src/reproduce_paper_results.py
"$PYTHON_BIN" src/generate_prompt_provenance.py
"$PYTHON_BIN" scripts/verify.py --results-only
