# Optional new inference

The offline reproduction command does not invoke these tools. The inference
extra is optional and includes larger model-library dependencies. Its lockfile
records a current dependency resolution, not the original collection environment.
New collection uses a paid DeepInfra OpenAI-compatible endpoint; set `DEEPINFRA_API_KEY` in the
environment or a local, Git-ignored `.env` file. Run commands from the root.

```sh
uv sync --locked --no-dev --extra inference
. .venv/bin/activate
```

The model aliases used in the archived study are `openai/gpt-oss-120b` and
`Qwen/Qwen2.5-72B-Instruct`. The collectors default to the former. For Qwen,
add `--model Qwen/Qwen2.5-72B-Instruct --reasoning-effort omit` and use a
separate output directory and tracker. Hosted availability can change.

## Historical experiment settings

The following describes the experiments reported in the paper. Commands below
are for optional new runs; the current dependency lock and hosted endpoints do
not guarantee reconstruction of the original inference environment.

All conditions used temperature 0, top-p 1, seed 42, and a 6,000-token completion
limit. First-pass conditions requested JSON-object responses except CoT, which
requested text; Critique-Refine requested JSON-object responses. gpt-oss used
reasoning effort `none`; Qwen omitted the reasoning-effort field. The paper
reports DeepInfra's default serving precision as MXFP4 for gpt-oss and FP8 for
Qwen. These are provider-described serving settings, not locally verified
checkpoint identities. Provider-side checkpoint revisions were not exposed.
The collectors allowed three total API attempts per call, waiting one second
and then two seconds before retries after exceptions.

Both retrievers rank cosine scores with reversed NumPy `argsort` over candidate
arrays in training-data order; no stable tie-breaking rule is specified.
Semantic embeddings are L2-normalized and stored as `float32`. The paper
identifies `all-mpnet-base-v2` revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`; the supplied loader uses the model
name without explicitly pinning this revision. A new download therefore should
not be assumed to reproduce the historical encoder solely from these commands.
The historical Sentence-Transformers version and device, exact downloaded
Stanza resource package, and full retrieval environment were not preserved.
The preserved outputs are the target for exact numerical reproduction; new
model calls may produce different responses.

## First-pass conditions

Initialize a tracker once for each new model run. The setup command refuses to
overwrite an existing tracker:

```sh
python scripts/init_tracker.py --output runs/experiment_tracker_gpt_oss_120b.csv
```

This five-sentence example writes to ignored local-run paths:

```sh
python src/collect_responses.py \
  --input data/UD_English-EWT/en_ewt-ud-test.conllu \
  --prompt prompts/zero_shot.txt --experiment experiment1 \
  --output runs/gpt_oss_120b/zero_shot/model_responses.json \
  --tracker runs/experiment_tracker_gpt_oss_120b.csv \
  --max-sentences 5
```

Use the following condition settings and a corresponding new output path:

| Condition | Prompt file in `prompts/` | Experiment | Additional options |
|---|---|---|---|
| Zero-Shot | `zero_shot.txt` | `experiment1` | none |
| Fixed Few-Shot | `few_shot_fixed.txt` | `experiment2` | none (examples are in the prompt) |
| POS-Signature | `few_shot_retrieval.txt` | `experiment3` | `--few-shot-k 3 --retrieval-method syntactic` |
| Chain-of-Thought | `cot_few_shot.txt` | `experiment4` | `--response-format text` |
| Semantic | `semantic_retrieval.txt` | `experiment6` | `--few-shot-k 3 --retrieval-method semantic` |

Both retrieval methods require the training split. POS retrieval uses Stanza;
semantic retrieval uses `all-mpnet-base-v2`. These libraries can download model
resources and create caches in `data/processed/`. Current model downloads do
not recover missing historical environment metadata.

## Evaluation and Critique-Refine

Evaluate a new run without API calls:

```sh
python src/evaluate_responses.py \
  --input runs/gpt_oss_120b/zero_shot/model_responses.json \
  --output runs/gpt_oss_120b/zero_shot/model_evaluation.json \
  --exclude-punct
```

For Critique-Refine, first collect and evaluate the Chain-of-Thought condition,
then pass its evaluation to the second collector:

```sh
python src/collect_critique_responses.py \
  --source runs/gpt_oss_120b/cot_few_shot/model_evaluation.json \
  --output runs/gpt_oss_120b/critique_refine/model_responses.json \
  --prompt prompts/critique_refine.txt --experiment experiment5 \
  --tracker runs/experiment_tracker_gpt_oss_120b.csv --max-sentences 5
```

Evaluate the resulting raw responses with the same evaluator. Archived prompts
in `outputs/` are the record of the messages actually used in the paper;
[the prompt snapshot](prompt_source_snapshot.md) describes their templates.

## Additional utilities

`src/analyze_results.py` and `src/extended_error_analysis.py` retain the original
per-model diagnostic routines and their tests. Their conditional diagnostics
can use different subsets; `scripts/reproduce.sh` alone defines the paper's
complete fixed-denominator results. Install the plotting/test group with `uv sync --locked --group dev --extra inference`
and see each command's `--help` for available diagnostics.

`scripts/verify_fewshot_selection.py` checks the fixed demonstration choices.
`scripts/run_stanza_baseline.py` is the supplied optional supervised-reference
runner. It downloads models if needed and writes `results/stanza_baseline.json`;
it is not part of offline reproduction. The original saved reference run and
exact historical Stanza resource package are not included in this release.
