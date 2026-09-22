"""
evaluate_responses.py
─────────────────────
Phase 2 of the UD parsing experiment.

Reads the raw responses archive produced by collect_responses.py, applies
Pydantic validation, computes UAS / LAS, checks for structural issues
(missing/multiple roots), and writes a full evaluation report to JSON.

Because this script reads from a file instead of the API, it can be run
and re-run freely — change scoring logic, add new error analyses, or try
different validation rules without spending API budget.

Usage
─────
  python evaluate_responses.py [OPTIONS]

  --input          Path to raw responses JSON (from collect_responses.py)
                   (default: outputs/gpt_oss_120b/zero_shot/model_responses.json)
  --output         Path for evaluation results JSON
                   (default: outputs/gpt_oss_120b/zero_shot/model_evaluation.json)
  --exclude-punct  Exclude PUNCT tokens from UAS/LAS (standard UD eval)
  --self-test      Run offline unit tests and exit
"""

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

# ─────────────────────────────────────────────
# 1.  Pydantic schema
# ─────────────────────────────────────────────

class UDToken(BaseModel):
    id:     int = Field(..., description="1-indexed token position")
    form:   str = Field(..., description="Surface form of the word")
    upos:   str = Field(..., description="Universal POS tag")
    head:   int = Field(..., description="Head token id; 0 = root")
    deprel: str = Field(..., description="UD dependency relation label")


class UDParse(BaseModel):
    tokens: list[UDToken] = Field(..., description="List of parsed tokens")


# ─────────────────────────────────────────────
# 2.  Structural warnings (soft checks)
# ─────────────────────────────────────────────

def structural_warnings(parsed: UDParse) -> list[str]:
    """
    Returns a list of UD tree-structure issues found in a parse.
    The parse is NOT rejected — issues are recorded as warnings so that
    every LLM response can be scored and studied, including broken ones.

    Checks:
      - no_root         : no token has head=0
      - multiple_roots  : more than one token has head=0
      - token_count     : predicted token count differs from gold count
    """
    warnings: list[str] = []
    roots = [t for t in parsed.tokens if t.head == 0]
    if len(roots) == 0:
        warnings.append("no_root: no token has head=0")
    elif len(roots) > 1:
        root_ids = [t.id for t in roots]
        warnings.append(f"multiple_roots: tokens {root_ids} all have head=0")
    return warnings


# ─────────────────────────────────────────────
# 3.  UAS / LAS scorer
# ─────────────────────────────────────────────

def compute_uas_las(
    gold_tokens: list[dict],
    pred_tokens: list[UDToken],
    exclude_punct: bool = False,
) -> tuple[float, float]:
    """
    Computes Unlabeled (UAS) and Labeled (LAS) Attachment Score.

    Alignment is by token `id`. A prediction missing a gold token id
    counts as incorrect for that token.

    Parameters
    ----------
    gold_tokens   : full gold token dicts from CoNLL-U
    pred_tokens   : validated UDToken objects from the LLM
    exclude_punct : if True, gold UPOS=PUNCT tokens are excluded from scoring

    Returns
    -------
    (uas, las)  floats in [0, 1], rounded to 4 decimal places
    """
    pred_by_id = {t.id: t for t in pred_tokens}

    total = uas_correct = las_correct = 0

    for g in gold_tokens:
        if exclude_punct and g["upos"] == "PUNCT":
            continue
        total += 1
        p = pred_by_id.get(g["id"])
        if p is None:
            continue
        if p.head == g["head"]:
            uas_correct += 1
            if p.deprel.lower() == g["deprel"].lower():
                las_correct += 1

    if total == 0:
        return 0.0, 0.0

    return round(uas_correct / total, 4), round(las_correct / total, 4)


# ─────────────────────────────────────────────
# 4.  Token-level error breakdown
# ─────────────────────────────────────────────

def token_errors(
    gold_tokens: list[dict],
    pred_tokens: list[UDToken],
    exclude_punct: bool = False,
) -> list[dict]:
    """
    Returns one dict per token that has a head or deprel error.
    Useful for fine-grained analysis of where the model makes mistakes.
    """
    pred_by_id = {t.id: t for t in pred_tokens}
    errors = []

    for g in gold_tokens:
        if exclude_punct and g["upos"] == "PUNCT":
            continue

        p = pred_by_id.get(g["id"])
        if p is None:
            errors.append({
                "token_id":   g["id"],
                "form":       g["form"],
                "gold_upos":  g["upos"],
                "gold_head":  g["head"],
                "gold_deprel": g["deprel"],
                "pred_head":  None,
                "pred_deprel": None,
                "error_type": "missing_token",
            })
            continue

        head_wrong   = p.head    != g["head"]
        deprel_wrong = p.deprel.lower() != g["deprel"].lower()

        if head_wrong or deprel_wrong:
            if head_wrong and deprel_wrong:
                error_type = "head_and_deprel"
            elif head_wrong:
                error_type = "head_only"
            else:
                error_type = "deprel_only"

            errors.append({
                "token_id":    g["id"],
                "form":        g["form"],
                "gold_upos":   g["upos"],
                "gold_head":   g["head"],
                "gold_deprel": g["deprel"],
                "pred_head":   p.head,
                "pred_deprel": p.deprel,
                "error_type":  error_type,
            })

    return errors


# ─────────────────────────────────────────────
# 5.  JSON extraction helper (CoT + plain-JSON)
# ─────────────────────────────────────────────

def extract_json_block(raw: str) -> tuple[str, "str | None"]:
    """
    Returns (json_str, cot_reasoning) from a raw model response.

    - Pure-JSON responses (from json_object mode) start with '{' and are
      returned as-is with cot_reasoning=None.
    - CoT responses (from text mode) contain reasoning text before the JSON
      object. The substring from the first '{' onward is returned as json_str,
      and everything before it is returned as cot_reasoning.

    Raises ValueError if no '{' character is found anywhere in raw.
    """
    stripped = raw.strip()
    if stripped.startswith('{'):
        return stripped, None          # fast path — pure JSON, no reasoning prefix
    brace_pos = stripped.find('{')
    if brace_pos == -1:
        raise ValueError("No JSON object found in response (no '{' character)")
    json_str  = stripped[brace_pos:]
    reasoning = stripped[:brace_pos].strip() or None
    return json_str, reasoning


def evaluate(
    input_path: str,
    output_path: str,
    exclude_punct: bool = True,
) -> None:
    print(f"[INFO] Reading raw responses: {input_path}")
    with open(input_path, encoding="utf-8") as fh:
        raw_records: list[dict] = json.load(fh)
    print(f"[INFO] {len(raw_records)} record(s) to evaluate.")

    eval_results = []

    for idx, rec in enumerate(raw_records, 1):
        sent_id     = rec["sent_id"]
        text        = rec["text"]
        gold_tokens = rec["gold"]
        raw_resp    = rec.get("raw_response")
        api_error   = rec.get("api_error")

        print(f"\n[{idx}/{len(raw_records)}] {sent_id}")

        # ── Base result dict ────────────────────────────────────────────────
        result = {
            # Provenance — what was sent and when
            "sent_id":          sent_id,
            "text":             text,
            "timestamp":        rec.get("timestamp"),
            "model_config":     rec.get("model_config"),
            "prompt":           rec.get("prompt"),
            "usage":            rec.get("usage"),
            # Raw model output (always preserved)
            "raw_response":     raw_resp,
            # Gold annotation
            "gold":             gold_tokens,
            # Filled in below
            "llm_parse":        None,
            "uas":              None,
            "las":              None,
            "exclude_punct":    exclude_punct,
            "token_errors":     None,
            "structural_warnings": [],
            "cot_reasoning":    None,   # reasoning text before JSON (CoT runs only)
            "validation_error": None,
            "api_error":        api_error,
        }

        # ── Skip if the API call itself failed ──────────────────────────────
        if api_error:
            print(f"  [SKIP] API error recorded during collection: {api_error}")
            eval_results.append(result)
            continue

        if not raw_resp:
            result["validation_error"] = "empty_response: raw_response is null"
            print(f"  [SKIP] No raw response stored.")
            eval_results.append(result)
            continue

        # ── Attempt JSON parse (handles pure-JSON and CoT reasoning+JSON) ───
        try:
            json_str, cot_reasoning = extract_json_block(raw_resp)
            parsed_json = json.loads(json_str)
        except (ValueError, json.JSONDecodeError) as e:
            result["validation_error"] = f"json_decode_error: {e}"
            print(f"  [WARN] JSON decode / extraction failed: {e}")
            eval_results.append(result)
            continue
        if cot_reasoning:
            result["cot_reasoning"] = cot_reasoning
            print(f"  [INFO] CoT reasoning captured ({len(cot_reasoning)} chars)")

        # ── Pydantic schema validation ──────────────────────────────────────
        try:
            parsed = UDParse.model_validate(parsed_json)
        except ValidationError as ve:
            result["validation_error"] = str(ve)
            print(f"  [WARN] Pydantic validation failed: {ve}")
            eval_results.append(result)
            continue

        # ── Soft structural checks (warns but does NOT discard) ─────────────
        struct_warns = structural_warnings(parsed)
        if struct_warns:
            for w in struct_warns:
                print(f"  [WARN] Structural issue: {w}")

        # ── Scoring ─────────────────────────────────────────────────────────
        uas, las = compute_uas_las(gold_tokens, parsed.tokens, exclude_punct)
        errs     = token_errors(gold_tokens, parsed.tokens, exclude_punct)

        print(f"  UAS: {uas:.4f}  LAS: {las:.4f}  "
              f"Token errors: {len(errs)}/{len(gold_tokens)}")

        result.update({
            "llm_parse":           [t.model_dump() for t in parsed.tokens],
            "uas":                 uas,
            "las":                 las,
            "token_errors":        errs,
            "structural_warnings": struct_warns,
        })

        eval_results.append(result)

    # ── Write output ────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(eval_results, fh, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Evaluation saved to: {out_path}")

    # ── Summary stats ───────────────────────────────────────────────────────
    scored  = [r for r in eval_results if r["uas"] is not None]
    failed  = [r for r in eval_results if r["validation_error"]]
    api_err = [r for r in eval_results if r["api_error"]]
    no_root = [r for r in eval_results
               if any("no_root" in w for w in r["structural_warnings"])]

    print(f"\n{'─'*50}")
    print(f"  Total sentences :  {len(eval_results)}")
    print(f"  Scored (UAS/LAS):  {len(scored)}")
    print(f"  Validation errors: {len(failed)}")
    print(f"  API errors:        {len(api_err)}")
    print(f"  Missing root:      {len(no_root)}")
    if scored:
        avg_uas = sum(r["uas"] for r in scored) / len(scored)
        avg_las = sum(r["las"] for r in scored) / len(scored)
        print(f"  Average UAS:       {avg_uas:.4f}")
        print(f"  Average LAS:       {avg_las:.4f}")
    print(f"{'─'*50}")


# ─────────────────────────────────────────────
# 6.  Offline self-test
# ─────────────────────────────────────────────

def self_test() -> None:
    print("Running offline self-tests …\n")
    errors = []

    # ── Test 1: Pydantic valid parse ──────────────────────────────────────
    valid_payload = {
        "sent_id": "test-001",
        "tokens": [
            {"id": 1, "form": "The", "upos": "DET",   "head": 2, "deprel": "det"},
            {"id": 2, "form": "cat", "upos": "NOUN",  "head": 3, "deprel": "nsubj"},
            {"id": 3, "form": "sat", "upos": "VERB",  "head": 0, "deprel": "root"},
            {"id": 4, "form": ".",   "upos": "PUNCT", "head": 3, "deprel": "punct"},
        ],
    }
    try:
        UDParse.model_validate(valid_payload)
        print("  [PASS] Pydantic valid parse")
    except ValidationError as ve:
        errors.append(f"Pydantic valid parse failed: {ve}")

    # ── Test 2: Pydantic rejects missing 'form' ────────────────────────────
    try:
        UDParse.model_validate({
            "sent_id": "test-bad",
            "tokens": [{"id": 1, "upos": "DET", "head": 0, "deprel": "root"}],
        })
        errors.append("Pydantic should have rejected missing 'form'")
    except ValidationError:
        print("  [PASS] Pydantic correctly rejected invalid schema")

    # ── Test 3: structural_warnings — no root ─────────────────────────────
    no_root_parse = UDParse.model_validate({
        "sent_id": "test-noroot",
        "tokens": [
            {"id": 1, "form": "cat", "upos": "NOUN", "head": 2, "deprel": "nsubj"},
            {"id": 2, "form": "sat", "upos": "VERB", "head": 1, "deprel": "root"},
        ],
    })
    warns = structural_warnings(no_root_parse)
    if any("no_root" in w for w in warns):
        print("  [PASS] Detected no_root warning")
    else:
        errors.append(f"Expected no_root warning, got: {warns}")

    # ── Test 4: structural_warnings — multiple roots ───────────────────────
    multi_root_parse = UDParse.model_validate({
        "sent_id": "test-multiroot",
        "tokens": [
            {"id": 1, "form": "cat", "upos": "NOUN", "head": 0, "deprel": "root"},
            {"id": 2, "form": "sat", "upos": "VERB", "head": 0, "deprel": "root"},
        ],
    })
    warns = structural_warnings(multi_root_parse)
    if any("multiple_roots" in w for w in warns):
        print("  [PASS] Detected multiple_roots warning")
    else:
        errors.append(f"Expected multiple_roots warning, got: {warns}")

    # ── Test 5: UAS / LAS values ──────────────────────────────────────────
    gold = [
        {"id": 1, "form": "The", "upos": "DET",   "head": 2, "deprel": "det"},
        {"id": 2, "form": "cat", "upos": "NOUN",  "head": 3, "deprel": "nsubj"},
        {"id": 3, "form": "sat", "upos": "VERB",  "head": 0, "deprel": "root"},
        {"id": 4, "form": ".",   "upos": "PUNCT", "head": 3, "deprel": "punct"},
    ]
    pred = [
        UDToken(id=1, form="The", upos="DET",   head=2, deprel="det"),
        UDToken(id=2, form="cat", upos="NOUN",  head=3, deprel="obj"),  # wrong deprel
        UDToken(id=3, form="sat", upos="VERB",  head=0, deprel="root"),
        UDToken(id=4, form=".",   upos="PUNCT", head=3, deprel="punct"),
    ]
    uas, las = compute_uas_las(gold, pred, exclude_punct=False)
    if uas == 1.0:
        print(f"  [PASS] UAS = {uas}")
    else:
        errors.append(f"UAS expected 1.0, got {uas}")
    if las == 0.75:
        print(f"  [PASS] LAS = {las}")
    else:
        errors.append(f"LAS expected 0.75, got {las}")

    # ── Test 6: token_errors breakdown ────────────────────────────────────
    errs = token_errors(gold, pred, exclude_punct=False)
    if len(errs) == 1 and errs[0]["error_type"] == "deprel_only":
        print(f"  [PASS] token_errors correctly identified 1 deprel_only error")
    else:
        errors.append(f"Expected 1 deprel_only error, got: {errs}")

    # ── Test 7: UAS / LAS excluding PUNCT ────────────────────────────────
    uas_np, las_np = compute_uas_las(gold, pred, exclude_punct=True)
    expected_las_np = round(2 / 3, 4)
    if uas_np == 1.0:
        print(f"  [PASS] UAS (no punct) = {uas_np}")
    else:
        errors.append(f"UAS (no punct) expected 1.0, got {uas_np}")
    if las_np == expected_las_np:
        print(f"  [PASS] LAS (no punct) = {las_np}")
    else:
        errors.append(f"LAS (no punct) expected {expected_las_np}, got {las_np}")

    print()
    if errors:
        print("FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("All self-tests passed ✓")


# ─────────────────────────────────────────────
# 7.  Entry point
# ─────────────────────────────────────────────

DEFAULT_INPUT  = "outputs/gpt_oss_120b/zero_shot/model_responses.json"
DEFAULT_OUTPUT = "outputs/gpt_oss_120b/zero_shot/model_evaluation.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate collected LLM responses for UD dependency parsing."
    )
    parser.add_argument("--input",         default=DEFAULT_INPUT,
                        help="Path to raw responses JSON (from collect_responses.py)")
    parser.add_argument("--output",        default=DEFAULT_OUTPUT,
                        help="Path for evaluation results JSON")
    parser.add_argument("--include-punct", action="store_false", dest="exclude_punct",
                        help="Include PUNCT tokens in UAS/LAS scoring (default: excluded)")
    parser.set_defaults(exclude_punct=True)
    parser.add_argument("--self-test",     action="store_true",
                        help="Run offline unit tests and exit")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    evaluate(
        input_path=args.input,
        output_path=args.output,
        exclude_punct=args.exclude_punct,
    )


if __name__ == "__main__":
    main()
