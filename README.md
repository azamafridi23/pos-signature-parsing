# POS-Signature Demonstration Retrieval for In-Context Dependency Parsing

Code, prompts, final model outputs, and offline reproduction for a comparison of
six prompting conditions on the **2,077-sentence UD English-EWT test set**.
The study uses gpt-oss-120b and Qwen2.5-72B-Instruct without fine-tuning.

POS-Signature retrieval selects three gold-parsed training examples using POS
statistics and length compatibility. Semantic retrieval selects three examples
using sentence embeddings; Fixed Few-Shot uses the same three examples for every
query. The comparison evaluates selection policies, including their candidate
filters, rather than isolating POS features alone.

## Results

Scores are percentages over 21,998 gold non-punctuation tokens. Missing or
unscorable outputs receive zero credit without reducing the denominator.

| Condition | gpt-oss UAS | gpt-oss LAS | Qwen UAS | Qwen LAS |
|---|---:|---:|---:|---:|
| Zero-Shot | 75.1 | 64.7 | 75.7 | 62.9 |
| Fixed Few-Shot | 76.5 | 67.3 | 76.3 | 65.3 |
| POS-Signature Few-Shot | 80.4 | 72.4 | 82.1 | 73.2 |
| Semantic Few-Shot | 79.6 | 70.5 | 80.8 | 70.5 |
| Chain-of-Thought | 78.6 | 70.0 | 72.4 | 61.7 |
| Critique-Refine | 79.1 | 70.4 | 74.6 | 63.6 |

See [full-precision results](results/paper_results.json),
[readable tables](results/generated_tables.md), and the
[evaluation protocol](docs/protocol.md). Findings are exploratory and scoped to
one English treebank, gold query tokenization, and labeled in-domain examples.

## Reproduce the paper results

Use Python 3.12. All 24 compressed input archives are included; no additional
archive download or manual extraction is needed.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify.py
PYTHON_BIN=python bash scripts/reproduce.sh
```

After dependency installation, these commands run offline without API keys.
They verify archive integrity, regenerate scores and 10,000 paired-bootstrap
resamples with seed 42, export tables and diagnostic figures, and reconstruct
all 24,921 archived prompt pairs. Numerical outputs must match the supplied
reference hashes. Font differences can change figure bytes; after regeneration,
use `python scripts/verify.py --results-only` for the numerical check.

## Repository map

| Directory | Contents |
|---|---|
| `src/` | Retrieval, collection, evaluation, and analysis implementation |
| `scripts/` | Reproduction, verification, checksum maintenance, and optional utilities |
| `tests/` | Existing deterministic evaluator and analysis tests |
| `prompts/` | Six original prompt source files |
| `data/` | English-EWT train/dev/test splits and upstream notices |
| `outputs/` | Final response and evaluation archives (`.json.gz`) |
| `results/` | Scores, statistical comparisons, diagnostics, and table exports |
| `figures/` | Fixed method schematic and generated diagnostic figures |
| `docs/` | Protocol, prompt provenance, inference guide, and validation record |
| `checksums/` | Input and complete-release SHA-256 manifests |

The LaTeX manuscript is maintained separately; this repository contains no
obsolete manuscript copy. Citation metadata is in [CITATION.cff](CITATION.cff).

## Development and optional inference

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

See [inference instructions](docs/inference.md) for the optional dependencies
and commands. New collection contacts a paid external API and writes to ignored
`runs/` paths. It is separate from reproducing the paper's archived results.
Historical model revisions and exact Stanza resources were not preserved, so
new inference need not produce identical responses.

## Licensing

See [licensing status](LICENSE.md). The bundled English-EWT data retains its
[upstream license and notices](data/UD_English-EWT/LICENSE.txt).
