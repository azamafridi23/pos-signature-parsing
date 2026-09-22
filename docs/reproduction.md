# Reproduction and maintenance

`pyproject.toml` defines dependencies; `uv.lock` records the resolved versions
and package hashes. `.python-version` selects the tested Python 3.12.14.
The base environment contains NumPy 2.3.5 and Pillow 12.3.0. Tests use the `dev`
group; model collection uses the optional `inference` extra.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12.5 or
newer. Run from the repository root. Make targets require Make and Bash:

| Command | Purpose |
|---|---|
| `make setup` | Sync the locked reproduction environment in `.venv/` |
| `make verify` | Check experimental-input checksums, input identities, gold alignment, and results |
| `make reproduce` | Regenerate results, tables, diagnostic figures, and prompt provenance |
| `make test` | Install the locked test group and run deterministic tests |

Every target uses `--locked`: an incompatible dependency-file change fails
instead of silently updating the lock. Dependencies and Python may download on
first setup. After setup, `UV_OFFLINE=1 make reproduce` runs without network
access. To run tests offline, first run `uv sync --locked --group dev`.

## Without Make

```sh
uv sync --locked --no-dev
uv run --locked --no-dev python scripts/verify.py
uv run --locked --no-dev env PYTHON_BIN=python bash scripts/reproduce.sh
uv run --locked --group dev python -m pytest -q
```

The 2,077 test sentences remain the reference for every condition. Three absent
Qwen critique records and unscorable responses receive zero credit. See
[protocol.md](protocol.md) for scoring and statistical details and
[the prompt snapshot](prompt_source_snapshot.md) for actual input templates.

Font availability can change regenerated figure bytes; figures are not checksummed.
`make verify` still checks numerical agreement after reproduction without a
checksum refresh. The fixed pipeline schematic is supplied rather than regenerated.

## Pip compatibility

The requirements files are generated exports, not separate dependency sources.
Use Python 3.12.14 and choose the export for reproduction, development, or
inference. For example:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
python scripts/verify.py
PYTHON_BIN=python bash scripts/reproduce.sh
```

`requirements-dev.txt` and `requirements-inference.txt` each include their own
complete dependency closure; install the appropriate file with the same pip
options when those tools are needed.

## Updating dependencies intentionally

Edit `pyproject.toml`, run `uv lock`, and export the compatibility files:

```sh
uv export --locked --no-dev --no-emit-project -o requirements.txt
uv export --locked --group dev --no-emit-project -o requirements-dev.txt
uv export --locked --no-dev --extra inference --no-emit-project -o requirements-inference.txt
```

Do not edit the generated requirements files or lockfile manually. Run
`make setup`, `make reproduce`, and `make test` after a dependency change.
The lockfile describes this reproduction release; it does not recreate missing
historical inference environments or guarantee identical model API responses.

## Checksum scope and maintenance

`checksums/INPUT_SHA256SUMS.txt` covers the CoNLL-U datasets, archived responses
and evaluations, prompt templates, and the saved archive-identity and numerical
reference JSON files. Source code, documentation, configuration, and figures
are not included. Routine edits to these files require no checksum refresh.
Git records their changes; numerical reference hashes check reproduced results.

Only after reviewing an intentional change to experimental inputs (including
archive recompression), refresh the input manifest:

```sh
uv run --locked --no-dev python scripts/update_checksums.py
make verify
```

This command does not update archive identities or numerical reference hashes.
Do not change `docs/input_identity.json` or `docs/expected_numeric_hashes.json`
merely to silence a mismatch: investigate it against the intended input or
scientific result first. Inspect staged files before committing; exclude local
runs, credentials, environments, and model caches.

The supervised Stanza reference is outside the archived two-model evaluation.
Its original saved run output and exact historical resource package are not
included. Offline reproduction does not rerun that experiment.
