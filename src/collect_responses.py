"""
collect_responses.py
────────────────────
Phase 1 of the UD parsing experiment.

Reads sentences from a CoNLL-U file, sends each one to gpt-oss via deepinfra,
and faithfully archives the complete API interaction to a JSON file:
  - Full prompt (system + user)
  - Raw model output string (unparsed)
  - Model configuration
  - Timestamp
  - Token usage
  - Gold annotation

NO Pydantic validation, NO scoring.  The goal is to preserve every API
response exactly as it came, so that evaluation can be run (and re-run with
different logic) without additional API calls.

Usage
─────
  python collect_responses.py [OPTIONS]

  --input          Path to CoNLL-U test file
                   (default: UD_English-EWT/en_ewt-ud-test.conllu)
  --output         Path for raw responses JSON
                   (default: runs/gpt_oss_120b/zero_shot/model_responses.json)
  --max-sentences  Sentences to process (default: 10)
  --api-key        DeepInfra API key (or set DEEPINFRA_API_KEY env var)
  --resume         Skip sentences whose sent_id is already in --output file
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional
import pandas as pd

from openai import OpenAI


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DEFAULT_CONLLU     = "data/UD_English-EWT/en_ewt-ud-test.conllu"
DEFAULT_OUTPUT     = "runs/gpt_oss_120b/zero_shot/model_responses.json"
DEFAULT_PROMPT     = "prompts/zero_shot.txt"
DEFAULT_TRACKER    = "runs/experiment_tracker_gpt_oss_120b.csv"
DEFAULT_EXPERIMENT = "experiment1"   # column name in the CSV
DEFAULT_FEW_SHOT_K = 0

MAX_RETRIES = 3

DEFAULT_MODEL           = "openai/gpt-oss-120b"
TEMPERATURE             = 0
TOP_P                   = 1
SEED                    = 42
# Reasoning effort is a gpt-oss-specific parameter. For models that do not accept
# it (e.g. Llama, Qwen), pass --reasoning-effort omit so it is left out of the call.
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_RESPONSE_FORMAT = "json_object"   # override with --response-format text for CoT
MAX_TOKENS              = 6000

# No hardcoded SYSTEM_PROMPT. Loaded dynamically from file.


def build_user_message(text: str, tokens: list[dict]) -> str:
    """Mirrors the first two CoNLL-U columns: id <TAB> form."""
    token_lines = "\n".join(f"{t['id']}\t{t['form']}" for t in tokens)
    return (
        f"# text = {text}\n"
        f"{token_lines}"
    )


# ─────────────────────────────────────────────
# 2.  CoNLL-U reader
# ─────────────────────────────────────────────

def parse_conllu(path: str) -> list[dict]:
    """
    Returns a list of sentence dicts, each with:
      sent_id, text, tokens (full 8-field dicts from CoNLL-U, no multi-word spans)
    """
    sentences = []
    current: Optional[dict] = None

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")

            if line == "":
                if current and current["tokens"]:
                    sentences.append(current)
                current = None
                continue

            if line.startswith("#"):
                if current is None:
                    current = {"sent_id": "", "text": "", "tokens": []}
                if line.startswith("# sent_id"):
                    current["sent_id"] = line.split("=", 1)[1].strip()
                elif line.startswith("# text"):
                    current["text"] = line.split("=", 1)[1].strip()
                continue

            if current is None:
                current = {"sent_id": "", "text": "", "tokens": []}

            parts = line.split("\t")
            if len(parts) < 10:
                continue

            tok_id = parts[0]
            if "-" in tok_id or "." in tok_id:   # skip multi-word / empty nodes
                continue

            try:
                current["tokens"].append({
                    "id":     int(tok_id),
                    "form":   parts[1],
                    "lemma":  parts[2],
                    "upos":   parts[3],
                    "xpos":   parts[4],
                    "feats":  parts[5],
                    "head":   int(parts[6]),
                    "deprel": parts[7],
                })
            except (ValueError, IndexError):
                continue

    if current and current["tokens"]:
        sentences.append(current)

    return sentences


# ─────────────────────────────────────────────
# 3.  Collection loop
# ─────────────────────────────────────────────

def collect(
    conllu_path: str,
    output_path: str,
    prompt_path: str,
    tracker_path: str,
    experiment_col: str,
    run_id: str,
    max_sentences: int,
    few_shot_k: int,
    api_key: str,
    response_format: str = DEFAULT_RESPONSE_FORMAT,
    retrieval_method: str = "syntactic",
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    concurrency: int = 1,
    flush_every: int = 25,
) -> None:
    # Read the prompt
    print(f"[INFO] Reading system prompt: {prompt_path}")
    with open(prompt_path, encoding="utf-8") as fh:
        system_prompt = fh.read().strip()

    # Load tracker CSV to find PENDING sentences for this experiment
    tracker_file = Path(tracker_path)
    if not tracker_file.exists():
        print(f"[ERROR] Tracker file not found: {tracker_file}")
        return
    df_tracker = pd.read_csv(tracker_file)
    status_col = experiment_col + "_status"
    if status_col not in df_tracker.columns:
        print(f"[ERROR] Column '{status_col}' not found in tracker. "
              f"Available: {list(df_tracker.columns)}")
        return
    # Normalise to uppercase for case-insensitive comparison
    pending_ids: set[str] = set(
        df_tracker.loc[
            df_tracker[status_col].str.upper() == "PENDING", "sent_id"
        ].astype(str)
    )
    print(f"[INFO] Tracker: {len(pending_ids)} PENDING sentence(s) for '{status_col}'.")
    print(f"[INFO] Reading CoNLL-U: {conllu_path}")
    sentences = parse_conllu(conllu_path)
    print(f"[INFO] Total sentences: {len(sentences)}")

    # Filter to only sentences that are PENDING in the tracker
    sentences = [s for s in sentences if s["sent_id"] in pending_ids]
    print(f"[INFO] After tracker filter: {len(sentences)} sentence(s) to process.")

    if max_sentences > 0:
        sentences = sentences[:max_sentences]
        print(f"[INFO] Limited to: {len(sentences)} sentence(s) by --max-sentences.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing output to append to (if any), without re-processing sentences
    results: list[dict] = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            results = json.load(fh)

    # Defensive de-dup guard: never re-collect a sentence that already has a
    # SUCCESSFUL record on disk, even if the tracker still says PENDING (this can
    # happen if a run was killed between writing the JSON and flushing the tracker).
    # Only records with a completed response are skipped.
    already_ok = {r["sent_id"] for r in results if r.get("api_success")}
    if already_ok:
        before = len(sentences)
        sentences = [s for s in sentences if s["sent_id"] not in already_ok]
        skipped = before - len(sentences)
        if skipped:
            print(f"[INFO] Skipping {skipped} sentence(s) that already have a "
                  f"successful record on disk.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    # Resolve the retrieval function once; both modules share the same result
    # format and the same format_fewshot_example renderer, so the prompt block
    # is identical across conditions — only example *selection* differs.
    find_similar = None
    if few_shot_k > 0:
        if retrieval_method == "semantic":
            from semantic_similarity import FindSemanticSimilar as find_similar
        else:
            from syntactic_similarity import FindSyntacticSimilar as find_similar
        from syntactic_similarity import format_fewshot_example

    # reasoning_effort is only sent to the API when not "omit"; record what was used.
    send_reasoning = reasoning_effort != "omit"
    model_config = {
        "model":            model,
        "temperature":      TEMPERATURE,
        "top_p":            TOP_P,
        "seed":             SEED,
        "reasoning_effort": reasoning_effort if send_reasoning else None,
        "response_format":  {"type": response_format},
        "max_tokens":       MAX_TOKENS,
    }

    # Retrieval (Stanza / Sentence-BERT) is NOT thread-safe, so it is serialised
    # behind a lock. It is fast (~0.1s) relative to the API call (~2-3s), so this
    # costs little while the slow part — the HTTP request — still runs in parallel.
    retrieval_lock = threading.Lock()
    state_lock     = threading.Lock()   # guards results/tracker/counters/printing

    def build_prompt(text: str) -> str:
        """Effective system prompt (dynamic example injection for retrieval runs)."""
        if few_shot_k <= 0:
            return system_prompt
        with retrieval_lock:
            examples = find_similar(text, k=few_shot_k)
        blocks = "\n\n".join(format_fewshot_example(e) for e in examples)
        return system_prompt + "\n\n" + blocks

    def process_sentence(sent: dict) -> dict:
        """Query the model for one sentence and return its archive record.

        Runs in a worker thread: performs no shared-state mutation itself.
        """
        sent_id     = sent["sent_id"]
        text        = sent["text"]
        gold_tokens = sent["tokens"]

        user_msg  = build_user_message(text, gold_tokens)
        timestamp = datetime.now(timezone.utc).isoformat()
        effective_system_prompt = build_prompt(text)

        raw_response_str  = None
        reasoning_content = None
        usage             = None
        api_error         = None

        # Build kwargs so reasoning_effort can be omitted for non-gpt-oss models.
        call_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": effective_system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": response_format},
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,
            max_tokens=MAX_TOKENS,
        )
        if send_reasoning:
            call_kwargs["reasoning_effort"] = reasoning_effort

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(**call_kwargs)
                raw_response_str = response.choices[0].message.content

                # Reasoning models (like gpt-oss) expose their chain-of-thought
                # via reasoning_content on the message object (DeepInfra extension).
                rc = getattr(response.choices[0].message, "reasoning_content", None)
                if rc:
                    reasoning_content = rc

                usage = {
                    "prompt_tokens":     response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens":      response.usage.total_tokens,
                }
                break  # success — exit retry loop

            except Exception as exc:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)   # 1s, 2s
                else:
                    api_error = str(exc)

        return {
            # ── Identification ──────────────────────────────
            "sent_id":      sent_id,
            "text":         text,
            # ── Experimental conditions ──────────────────────
            "experiment":      experiment_col,
            "run_id":          run_id,
            "few_shot_k":      few_shot_k,
            "retrieval_method": retrieval_method if few_shot_k > 0 else None,
            "timestamp":       timestamp,
            "model_config":    model_config,
            # ── Full prompt (for audit / reproducibility) ────
            "prompt": {
                "system": effective_system_prompt,
                "user":   user_msg,
            },
            # ── Raw model output (unmodified string) ─────────
            "raw_response":      raw_response_str,
            # ── Internal reasoning chain (None if not exposed) ──
            "reasoning_content": reasoning_content,
            # ── API usage stats ──────────────────────────────
            "usage":        usage,
            # ── API error (None if call succeeded) ───────────
            "api_error":        api_error,
            # ── api_success: did the API call return a response? ─
            # ── parse_success/parse_error_type: filled by evaluate.py ──
            "api_success":      raw_response_str is not None and api_error is None,
            "parse_success":    None,   # filled in by evaluate.py after JSON parsing
            "parse_error_type": None,   # filled in by evaluate.py
            # ── Gold annotation (full CoNLL-U fields) ────────
            "gold":         gold_tokens,
        }

    # ── Shared state, mutated only under state_lock ────────────────────────────
    order        = [s["sent_id"] for s in sentences]   # canonical corpus order
    new_records: dict[str, dict] = {}
    counters     = {"done": 0, "errors": 0}

    def flush() -> None:
        """Checkpoint JSON *then* tracker. Caller must hold state_lock.

        Order matters: if we are killed between the two writes, the records are on
        disk but the tracker still says PENDING — which the `already_ok` guard above
        turns into a safe skip on resume (rather than a duplicate record).
        """
        ordered = results + [new_records[sid] for sid in order if sid in new_records]
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(ordered, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)          # atomic on POSIX
        df_tracker.to_csv(tracker_file, index=False)

    total = len(sentences)

    def record_result(record: dict) -> None:
        """Merge one finished record into shared state. Caller must NOT hold lock."""
        with state_lock:
            sid = record["sent_id"]
            new_records[sid] = record
            counters["done"] += 1
            if record["api_error"]:
                counters["errors"] += 1

            mask = df_tracker["sent_id"].astype(str) == sid
            df_tracker.loc[mask, status_col] = "FAIL" if record["api_error"] else "DONE"

            n = counters["done"]
            if record["api_error"]:
                print(f"[{n}/{total}] FAIL {sid}: {record['api_error']}")
            else:
                u = record["usage"] or {}
                print(f"[{n}/{total}] ok   {sid}  "
                      f"(prompt {u.get('prompt_tokens')}, completion {u.get('completion_tokens')})")

            if n % flush_every == 0:
                flush()

    if concurrency <= 1:
        # ── Sequential path (original behaviour, incl. the inter-call pause) ──
        for sent in sentences:
            record_result(process_sentence(sent))
            time.sleep(0.5)
    else:
        # ── Concurrent path: the API call is I/O-bound, so threads are ideal ──
        print(f"[INFO] Collecting with concurrency={concurrency} "
              f"(checkpointing every {flush_every} sentences).")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(process_sentence, s): s for s in sentences}
            for fut in as_completed(futures):
                sent = futures[fut]
                try:
                    record_result(fut.result())
                except Exception as exc:   # unexpected (non-API) worker failure
                    with state_lock:
                        counters["errors"] += 1
                        print(f"  [ERROR] worker crashed on {sent['sent_id']}: {exc}")

    with state_lock:
        flush()
        final_total = len(results) + len(new_records)

    print(f"\n[INFO] Done. New: {len(new_records)}, Errors: {counters['errors']}, "
          f"Total stored: {final_total}")
    print(f"[INFO] Saved to: {out_path}")


def update_csv_status(
    df: "pd.DataFrame",
    tracker_file: Path,
    sent_id: str,
    status_col: str,
    new_status: str,
) -> "pd.DataFrame":
    """
    Updates a single row in the in-memory DataFrame, then flushes to disk.
    Returns the updated DataFrame so the caller stays in sync without a re-read.
    """
    import pandas as pd
    mask = df["sent_id"].astype(str) == sent_id
    if mask.any():
        df.loc[mask, status_col] = new_status
        df.to_csv(tracker_file, index=False)
    else:
        print(f"  [WARN] sent_id '{sent_id}' not found in tracker — status not updated.")
    return df


# ─────────────────────────────────────────────
# 4.  Entry point
# ─────────────────────────────────────────────

def main() -> None:
    load_dotenv()   # load .env before reading env vars

    parser = argparse.ArgumentParser(
        description="Collect raw LLM responses for UD dependency parsing."
    )
    parser.add_argument("--input",         default=DEFAULT_CONLLU,
                        help="Path to CoNLL-U test file")
    parser.add_argument("--output",        default=DEFAULT_OUTPUT,
                        help="Path for raw responses JSON file")
    parser.add_argument("--prompt",        default=DEFAULT_PROMPT,
                        help="Path to system prompt text file")
    parser.add_argument("--tracker",       default=DEFAULT_TRACKER,
                        help="Path to experiment tracker CSV file")
    parser.add_argument("--experiment",    default=DEFAULT_EXPERIMENT,
                        help="Experiment column prefix in tracker CSV (e.g. experiment1)")
    parser.add_argument("--max-sentences", type=int, default=0,
                        help="Maximum sentences to collect (0 = no limit, default: 0)")
    parser.add_argument("--few-shot-k",      type=int, default=DEFAULT_FEW_SHOT_K,
                        help="Number of retrieved examples per prompt (0 = use prompt file exactly as-is)")
    parser.add_argument("--retrieval-method", default="syntactic",
                        choices=["syntactic", "semantic"],
                        help="How to select the --few-shot-k examples: 'syntactic' "
                             "(POS-signature similarity, experiment3) or 'semantic' "
                             "(Sentence-BERT cosine similarity, experiment6).")
    parser.add_argument("--run-id",          default="run_001",
                        help="Run identifier for self-consistency tracking (default: run_001)")
    parser.add_argument("--response-format", default=DEFAULT_RESPONSE_FORMAT,
                        choices=["json_object", "text"],
                        help="API response format. Use 'json_object' (default) for all standard "
                             "experiments; use 'text' for CoT prompts that emit reasoning before JSON.")
    parser.add_argument("--api-key",         default=None,
                        help="DeepInfra API key (or set DEEPINFRA_API_KEY env var)")
    parser.add_argument("--model",           default=DEFAULT_MODEL,
                        help="Model id served via DeepInfra (default: openai/gpt-oss-120b). "
                             "e.g. meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                        help="Reasoning effort for gpt-oss models (default: none). "
                             "Pass 'omit' to leave the parameter out entirely for models "
                             "that do not accept it (e.g. Llama, Qwen).")
    parser.add_argument("--concurrency",  type=int, default=1,
                        help="Parallel API calls (default: 1 = original sequential "
                             "behaviour). The call is I/O-bound, so 8-16 is a big "
                             "speed-up; retrieval stays serialised (not thread-safe).")
    parser.add_argument("--flush-every",  type=int, default=25,
                        help="Checkpoint the responses JSON + tracker every N completed "
                             "sentences (default: 25).")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPINFRA_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Set DEEPINFRA_API_KEY in your .env file "
            "or pass --api-key."
        )

    collect(
        conllu_path=args.input,
        output_path=args.output,
        prompt_path=args.prompt,
        tracker_path=args.tracker,
        experiment_col=args.experiment,
        run_id=args.run_id,
        max_sentences=args.max_sentences,
        few_shot_k=args.few_shot_k,
        api_key=api_key,
        response_format=args.response_format,
        retrieval_method=args.retrieval_method,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        concurrency=args.concurrency,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
