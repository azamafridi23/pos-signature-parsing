# Reproduction and maintenance

Run the quick-start commands in the root README with Python 3.12. The pinned
runtime needs NumPy and Pillow only; inference libraries are optional.

- `python scripts/verify.py` verifies all delivered file checksums, the 24
  uncompressed archive identities, raw/evaluation agreement against CoNLL-U gold,
  and the numerical reference hashes.
- `PYTHON_BIN=python bash scripts/reproduce.sh` checks the inputs, regenerates
  results, tables and diagnostic figures, reconstructs all archived prompts,
  and verifies numerical agreement. It makes no API calls.
- `python scripts/verify.py --results-only` checks evaluation agreement and
  numerical references after regeneration, without requiring identical figure
  bytes. The fixed pipeline schematic is supplied rather than regenerated.
- `python -m pytest -q` runs the deterministic tests after installing
  `requirements-dev.txt`.

The 2,077 test sentences remain the reference for every condition. Three absent
Qwen critique records and unscorable responses receive zero credit. See
[protocol.md](protocol.md) for scoring, bootstrap inference, and schema details.
[Prompt sources and dynamic templates](prompt_source_snapshot.md) and the
[per-call prompt manifest](prompt_provenance.json) document actual model inputs.

## Publishing an intentional update

After reviewing changes and passing the checks above, run:

```sh
python scripts/update_checksums.py
python scripts/verify.py
```

This refreshes the delivery manifests, not the numerical reference hashes.
Do not update `docs/expected_numeric_hashes.json` merely to silence a result
mismatch. Compressed and uncompressed input identities must remain consistent.
Inspect the staged Git files before committing; do not include local runs,
credentials, environments, or model caches.

The supervised Stanza reference is outside the archived two-model evaluation.
Its original saved run output and exact historical resource package are not
included, and offline reproduction does not rerun that experiment.
