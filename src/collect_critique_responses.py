"""
collect_critique_responses.py
─────────────────────────────
Phase 1 (critique refinement) of the UD parsing experiment.

Reads the evaluated parses from a previous experiment (default: cot_few_shot),
formats each sentence and its existing parse into a critique-refinement prompt,
sends it to the LLM, and archives the raw API interaction to JSON.

The model is asked to critically review the parse and produce a corrected
version.  Evaluation is handled separately by evaluate_responses.py.

Usage
─────
  python src/collect_critique_responses.py [OPTIONS]

  --source         Path to source evaluation JSON (from evaluate_responses.py)
                   (default: runs/gpt_oss_120b/cot_few_shot/model_evaluation.json)
  --output         Path for raw responses JSON
                   (default: runs/gpt_oss_120b/critique_refine/model_responses.json)
  --prompt         Path to the system prompt text file
                   (default: prompts/critique_refine.txt)
  --tracker        Path to experiment tracker CSV
                   (default: results/experiment_tracker.csv)
  --experiment     Experiment column prefix in tracker CSV
                   (default: experiment5)
  --run-id         Run identifier (default: run_001)
  --max-sentences  Sentences to process (0 = no limit)
  --api-key        DeepInfra API key (or set DEEPINFRA_API_KEY env var)
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path

import pandas as pd
from openai import OpenAI

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

DEFAULT_SOURCE     = "runs/gpt_oss_120b/cot_few_shot/model_evaluation.json"
DEFAULT_OUTPUT     = "runs/gpt_oss_120b/critique_refine/model_responses.json"
DEFAULT_PROMPT     = "prompts/critique_refine.txt"
DEFAULT_TRACKER    = "runs/experiment_tracker_gpt_oss_120b.csv"
DEFAULT_EXPERIMENT = "experiment5"

MAX_RETRIES = 3

DEFAULT_MODEL           = "openai/gpt-oss-120b"
TEMPERATURE             = 0
TOP_P                   = 1
SEED                    = 42
# gpt-oss-specific; pass --reasoning-effort omit for models that reject it (Qwen, Llama).
DEFAULT_REASONING_EFFORT = "none"
RESPONSE_FORMAT         = "json_object"
MAX_TOKENS              = 6000


# ─────────────────────────────────────────────
# User message builder
# ─────────────────────────────────────────────

def build_critique_user_message(
    text: str,
    gold_tokens: list[dict],
    llm_parse: list[dict],
) -> str:
    """
    Build the user message for the critique-refinement prompt.

    Provides:
      1. The sentence text
      2. The token list (id + form) for reference
      3. The existing parse as a JSON object

    The system prompt instructs the model to review and correct the parse.
    """
    token_lines = "\n".join(f"{t['id']}\t{t['form']}" for t in gold_tokens)
    existing_parse_json = json.dumps({"tokens": llm_parse}, indent=2)
    return (
        f"Sentence: {text}\n"
        f"\n"
        f"Tokens:\n"
        f"{token_lines}\n"
        f"\n"
        f"Existing parse:\n"
        f"{existing_parse_json}"
    )


# ─────────────────────────────────────────────
# Tracker helpers
# ─────────────────────────────────────────────

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
    mask = df["sent_id"].astype(str) == sent_id
    if mask.any():
        df.loc[mask, status_col] = new_status
        df.to_csv(tracker_file, index=False)
    else:
        print(f"  [WARN] sent_id '{sent_id}' not found in tracker — status not updated.")
    return df


# ─────────────────────────────────────────────
# Collection loop
# ─────────────────────────────────────────────

def collect(
    source_path: str,
    output_path: str,
    prompt_path: str,
    tracker_path: str,
    experiment_col: str,
    run_id: str,
    max_sentences: int,
    api_key: str,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    concurrency: int = 1,
    flush_every: int = 25,
) -> None:
    # Read the system prompt
    print(f"[INFO] Reading system prompt: {prompt_path}")
    with open(prompt_path, encoding="utf-8") as fh:
        system_prompt = fh.read().strip()

    # Load source evaluation records
    print(f"[INFO] Reading source evaluation: {source_path}")
    with open(source_path, encoding="utf-8") as fh:
        source_records: list[dict] = json.load(fh)
    print(f"[INFO] Total source records: {len(source_records)}")

    # Filter to records with a valid parse (something to critique)
    source_records = [r for r in source_records if r.get("llm_parse") is not None]
    print(f"[INFO] Records with valid parses: {len(source_records)}")

    # Build lookup by sent_id for fast access
    source_by_id = {r["sent_id"]: r for r in source_records}

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

    # Find PENDING sentence IDs
    pending_ids: set[str] = set(
        df_tracker.loc[
            df_tracker[status_col].str.upper() == "PENDING", "sent_id"
        ].astype(str)
    )
    print(f"[INFO] Tracker: {len(pending_ids)} PENDING sentence(s) for '{status_col}'.")

    # Intersect: only process sentences that are PENDING AND have a valid source parse
    to_process = [
        source_by_id[sid] for sid in pending_ids
        if sid in source_by_id
    ]
    # Sort by sent_id for deterministic order
    to_process.sort(key=lambda r: r["sent_id"])
    print(f"[INFO] After intersection (PENDING ∩ valid source): {len(to_process)} sentence(s).")

    if max_sentences > 0:
        to_process = to_process[:max_sentences]
        print(f"[INFO] Limited to: {len(to_process)} sentence(s) by --max-sentences.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing output to append to (if any)
    results: list[dict] = []
    if out_path.exists():
        with open(out_path, encoding="utf-8") as fh:
            results = json.load(fh)

    # Defensive de-dup guard: never re-collect a sentence that already has a
    # SUCCESSFUL record on disk, even if the tracker still says PENDING (possible if
    # a run was killed between writing the JSON and flushing the tracker). Failed
    # records without a completed response remain eligible.
    already_ok = {r["sent_id"] for r in results if r.get("api_success")}
    if already_ok:
        before = len(to_process)
        to_process = [r for r in to_process if r["sent_id"] not in already_ok]
        skipped = before - len(to_process)
        if skipped:
            print(f"[INFO] Skipping {skipped} sentence(s) that already have a "
                  f"successful record on disk.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepinfra.com/v1/openai",
    )

    # reasoning_effort is only sent to the API when not "omit"; record what was used.
    send_reasoning = reasoning_effort != "omit"
    model_config = {
        "model":            model,
        "temperature":      TEMPERATURE,
        "top_p":            TOP_P,
        "seed":             SEED,
        "reasoning_effort": reasoning_effort if send_reasoning else None,
        "response_format":  {"type": RESPONSE_FORMAT},
        "max_tokens":       MAX_TOKENS,
    }

    state_lock = threading.Lock()   # guards results/tracker/counters/printing

    def process_sentence(src_rec: dict) -> dict:
        """Critique one CoT parse and return its archive record.

        Runs in a worker thread: performs no shared-state mutation itself. Unlike
        collect_responses.py there is no retrieval step, so nothing needs serialising.
        """
        sent_id     = src_rec["sent_id"]
        text        = src_rec["text"]
        gold_tokens = src_rec["gold"]
        llm_parse   = src_rec["llm_parse"]
        source_uas  = src_rec.get("uas")
        source_las  = src_rec.get("las")

        user_msg  = build_critique_user_message(text, gold_tokens, llm_parse)
        timestamp = datetime.now(timezone.utc).isoformat()

        raw_response_str  = None
        reasoning_content = None
        usage             = None
        api_error         = None

        call_kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_msg},
            ],
            response_format={"type": RESPONSE_FORMAT},
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
            "timestamp":       timestamp,
            "model_config":    model_config,
            # ── Source experiment provenance ──────────────────
            "source_experiment": "cot_few_shot",
            "source_uas":        source_uas,
            "source_las":        source_las,
            # ── Full prompt (for audit / reproducibility) ────
            "prompt": {
                "system": system_prompt,
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
    order        = [r["sent_id"] for r in to_process]   # deterministic source order
    new_records: dict[str, dict] = {}
    counters     = {"done": 0, "errors": 0}
    total        = len(to_process)

    def flush() -> None:
        """Checkpoint JSON *then* tracker. Caller must hold state_lock.

        Order matters: if killed between the two writes, the records are on disk but
        the tracker still says PENDING — which the `already_ok` guard turns into a
        safe skip on resume (rather than a duplicate record).
        """
        ordered = results + [new_records[sid] for sid in order if sid in new_records]
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(ordered, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, out_path)          # atomic on POSIX
        df_tracker.to_csv(tracker_file, index=False)

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
        for src_rec in to_process:
            record_result(process_sentence(src_rec))
            time.sleep(0.5)
    else:
        # ── Concurrent path: the API call is I/O-bound, so threads are ideal ──
        print(f"[INFO] Critiquing with concurrency={concurrency} "
              f"(checkpointing every {flush_every} sentences).")
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(process_sentence, r): r for r in to_process}
            for fut in as_completed(futures):
                src_rec = futures[fut]
                try:
                    record_result(fut.result())
                except Exception as exc:   # unexpected (non-API) worker failure
                    with state_lock:
                        counters["errors"] += 1
                        print(f"  [ERROR] worker crashed on {src_rec['sent_id']}: {exc}")

    with state_lock:
        flush()
        final_total = len(results) + len(new_records)

    print(f"\n[INFO] Done. New: {len(new_records)}, Errors: {counters['errors']}, "
          f"Total stored: {final_total}")
    print(f"[INFO] Saved to: {out_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    load_dotenv()   # load .env before reading env vars

    parser = argparse.ArgumentParser(
        description="Collect critique-refinement LLM responses for UD dependency parsing."
    )
    parser.add_argument("--source",        default=DEFAULT_SOURCE,
                        help="Path to source evaluation JSON (from evaluate_responses.py)")
    parser.add_argument("--output",        default=DEFAULT_OUTPUT,
                        help="Path for raw responses JSON file")
    parser.add_argument("--prompt",        default=DEFAULT_PROMPT,
                        help="Path to system prompt text file")
    parser.add_argument("--tracker",       default=DEFAULT_TRACKER,
                        help="Path to experiment tracker CSV file")
    parser.add_argument("--experiment",    default=DEFAULT_EXPERIMENT,
                        help="Experiment column prefix in tracker CSV (e.g. experiment5)")
    parser.add_argument("--run-id",        default="run_001",
                        help="Run identifier for tracking (default: run_001)")
    parser.add_argument("--max-sentences", type=int, default=0,
                        help="Maximum sentences to collect (0 = no limit, default: 0)")
    parser.add_argument("--api-key",       default=None,
                        help="DeepInfra API key (or set DEEPINFRA_API_KEY env var)")
    parser.add_argument("--model",         default=DEFAULT_MODEL,
                        help="Model id served via DeepInfra (default: openai/gpt-oss-120b).")
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                        help="Reasoning effort for gpt-oss (default: none). Pass 'omit' "
                             "to leave the parameter out for models that reject it (Qwen, Llama).")
    parser.add_argument("--concurrency",  type=int, default=1,
                        help="Parallel API calls (default: 1 = original sequential "
                             "behaviour). The call is I/O-bound, so 8-16 is a big speed-up.")
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
        source_path=args.source,
        output_path=args.output,
        prompt_path=args.prompt,
        tracker_path=args.tracker,
        experiment_col=args.experiment,
        run_id=args.run_id,
        max_sentences=args.max_sentences,
        api_key=api_key,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        concurrency=args.concurrency,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
