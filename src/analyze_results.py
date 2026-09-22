"""
analyze_results.py
──────────────────
Research-grade evaluation pipeline for "Dependency Parsing by LLMs".

Reads the five model_evaluation.json files produced by evaluate_responses.py
and computes:

  A. Corpus-level UAS / LAS   (token-weighted, not per-sentence average)
  B. Bootstrap significance    (paired bootstrap on token-weighted/micro LAS;
                                sentences resampled in pairs, percentile p-value)
  C. Parse reliability         (parse success & structural validity rates)
  D. Per-relation error        (UAS / LAS broken down by deprel)
  E. Sentence-length analysis  (UAS / LAS per length bin)
  F. Reasoning trace sample    (CoT error records for manual annotation)

All results are written to results/ as JSON files and printed as summary tables.

Usage
─────
  python src/analyze_results.py [OPTIONS]

  --modules        Which modules to run (default: all)
                   corpus  reliability  per-relation  length  bootstrap  reasoning
  --n-resamples    Bootstrap iterations (default: 10000)
  --trace-n        CoT error sentences to extract for manual annotation (default: 50)
  --min-rel-count  Minimum token count to include a deprel in the relation table (default: 50)
"""

import argparse
import gzip
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Experiment order is significant: it fixes the order of pairwise significance
# comparisons in significance_tests.json. Keep semantic_retrieval last.
EXPERIMENT_NAMES = [
    "zero_shot", "few_shot_fixed", "few_shot_retrieval",
    "cot_few_shot", "critique_refine", "semantic_retrieval",
]
DEFAULT_MODEL_SLUG = "gpt_oss_120b"


def experiments_for(slug: str) -> dict[str, str]:
    return {e: f"outputs/{slug}/{e}/model_evaluation.json" for e in EXPERIMENT_NAMES}


# Model-scoped by default; overridden in main() from --model.
EXPERIMENTS: dict[str, str] = experiments_for(DEFAULT_MODEL_SLUG)
RESULTS_DIR = Path("results") / DEFAULT_MODEL_SLUG

# Non-PUNCT sentence length bins (lo, hi inclusive)
LENGTH_BINS   = [(1, 10), (11, 20), (21, 30), (31, 50), (51, 9_999)]
BIN_LABELS    = ["1-10", "11-20", "21-30", "31-50", "51+"]

ALL_MODULES   = ["corpus", "reliability", "per-relation", "length", "bootstrap", "reasoning"]

# ─────────────────────────────────────────────
# 1.  Data loading helpers
# ─────────────────────────────────────────────

def load_experiment(path: str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        path = Path(str(path) + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def scored_only(records: list[dict]) -> list[dict]:
    """Records where the LLM returned a valid, Pydantic-validated parse."""
    return [r for r in records if r.get("llm_parse") is not None]


# ─────────────────────────────────────────────
# Module A — Corpus-Level UAS / LAS
# ─────────────────────────────────────────────

def corpus_uas_las(records: list[dict]) -> dict:
    """
    Token-weighted UAS and LAS pooled across all scored sentences.

    PUNCT tokens are excluded (consistent with evaluate_responses.py).
    Sentences where llm_parse is None (parse failures) are excluded;
    the reliability module quantifies those separately.

    Returns
    -------
    {
        "uas":              float,   # corpus-level UAS
        "las":              float,   # corpus-level LAS
        "total_tokens":     int,     # non-PUNCT gold tokens counted
        "uas_correct":      int,
        "las_correct":      int,
        "missed_tokens":    int,     # gold tokens absent from prediction
        "scored_sentences": int,
    }
    """
    total = uas_ok = las_ok = missed = 0

    for rec in scored_only(records):
        gold     = rec["gold"]
        pred_map = {t["id"]: t for t in rec["llm_parse"]}

        for g in gold:
            if g["upos"] == "PUNCT":
                continue
            total += 1
            p = pred_map.get(g["id"])
            if p is None:
                missed += 1
                continue
            if p["head"] == g["head"]:
                uas_ok += 1
                if p["deprel"].lower() == g["deprel"].lower():
                    las_ok += 1

    if total == 0:
        return {
            "uas": 0.0, "las": 0.0, "total_tokens": 0,
            "uas_correct": 0, "las_correct": 0,
            "missed_tokens": 0, "scored_sentences": 0,
        }

    return {
        "uas":              round(uas_ok  / total, 4),
        "las":              round(las_ok  / total, 4),
        "total_tokens":     total,
        "uas_correct":      uas_ok,
        "las_correct":      las_ok,
        "missed_tokens":    missed,
        "scored_sentences": len(scored_only(records)),
    }


# ─────────────────────────────────────────────
# Module B — Bootstrap Significance Testing
# ─────────────────────────────────────────────

def per_sentence_las_counts(records: list[dict]) -> dict[str, tuple[int, int]]:
    """
    Per-sentence (las_correct, n_tokens) keyed by sent_id — the resampling unit
    for the token-weighted (micro) bootstrap.

    Scoring matches the evaluator: align by token id, exclude gold UPOS==PUNCT,
    LAS = correct head AND deprel (lowercased). Parse failures (llm_parse is
    None) contribute (0, n_tokens) so they penalise rather than being excluded.
    Because gold is shared across conditions, n_tokens per sentence is identical
    across conditions, so the micro-LAS difference is well defined on any
    resample.
    """
    counts: dict[str, tuple[int, int]] = {}
    for rec in records:
        gold = rec["gold"]
        n_tokens = sum(1 for g in gold if g["upos"] != "PUNCT")
        pred = rec.get("llm_parse")
        if pred is None:
            counts[rec["sent_id"]] = (0, n_tokens)
            continue
        pred_by_id = {t["id"]: t for t in pred}
        las_correct = 0
        for g in gold:
            if g["upos"] == "PUNCT":
                continue
            p = pred_by_id.get(g["id"])
            if p is not None and p["head"] == g["head"] \
                    and p["deprel"].lower() == g["deprel"].lower():
                las_correct += 1
        counts[rec["sent_id"]] = (las_correct, n_tokens)
    return counts


def _micro_las(counts: list[tuple[int, int]], idx) -> float:
    """Token-weighted LAS over the sentences selected by index list ``idx``."""
    c = sum(counts[i][0] for i in idx)
    n = sum(counts[i][1] for i in idx)
    return c / n if n else 0.0


def bootstrap_test(
    counts_a: list[tuple[int, int]],
    counts_b: list[tuple[int, int]],
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Two-sided paired bootstrap test on token-weighted (micro) LAS.

    Each unit is a per-sentence (las_correct, n_tokens) pair. Because all
    conditions are evaluated on the same sentences, the same resampled indices
    are drawn for both systems (paired). On each resample we recompute the micro
    LAS of each system over the pooled tokens and take the difference, so the
    test statistic matches the corpus headline metric (Koehn 2004, corpus-level).

    Method:
      1. observed_diff = microLAS(A) - microLAS(B) on the full set
      2. For each resample: draw paired indices idx of size n;
           diff* = microLAS(A[idx]) - microLAS(B[idx])
      3. Two-sided percentile p-value:
           p = 2 * min( frac(diff* <= 0), frac(diff* >= 0) ),  capped at 1.0
         i.e. the smaller tail of the resampled difference distribution past 0.

    Returns
    -------
    {
        "observed_diff": float,   # micro(A) - micro(B); negative means B better
        "p_value":       float,   # two-sided
        "n_resamples":   int,
        "significant":   bool,    # p < 0.05
    }
    """
    n = len(counts_a)
    assert n == len(counts_b), (
        f"Count arrays must be the same length; got {n} vs {len(counts_b)}"
    )

    full          = range(n)
    observed_diff = _micro_las(counts_a, full) - _micro_las(counts_b, full)

    rng = random.Random(seed)
    le = ge = 0
    for _ in range(n_resamples):
        idx = [rng.randint(0, n - 1) for _ in range(n)]
        diff_star = _micro_las(counts_a, idx) - _micro_las(counts_b, idx)
        if diff_star <= 0:
            le += 1
        if diff_star >= 0:
            ge += 1

    p_value = min(1.0, 2.0 * min(le, ge) / n_resamples)
    return {
        "observed_diff": round(observed_diff, 4),
        "p_value":       round(p_value,       4),
        "n_resamples":   n_resamples,
        "significant":   p_value < 0.05,
    }


def run_all_significance_tests(
    all_counts: dict[str, dict[str, tuple[int, int]]],
    n_resamples: int,
) -> dict:
    """Run pairwise micro-LAS bootstrap tests for every ordered pair of experiments.

    Aligns each pair on the intersection of their sent_ids so that
    experiments with different sentence counts are compared fairly.
    """
    exp_names = list(all_counts.keys())
    results: dict[str, dict] = {}

    for i, a in enumerate(exp_names):
        for b in exp_names[i + 1:]:
            key = f"{a}_vs_{b}"
            # Align on common sent_ids (sorted for deterministic order)
            common_ids = sorted(set(all_counts[a]) & set(all_counts[b]))
            counts_a = [all_counts[a][sid] for sid in common_ids]
            counts_b = [all_counts[b][sid] for sid in common_ids]
            print(f"  Bootstrap: {a}  vs  {b}  ({len(common_ids)} sents, "
                  f"{n_resamples:,} resamples) ...", end=" ", flush=True)
            results[key] = bootstrap_test(
                counts_a, counts_b, n_resamples=n_resamples,
            )
            results[key]["n_sentences"] = len(common_ids)
            print("done")

    return results


# ─────────────────────────────────────────────
# Module C — Parse Reliability
# ─────────────────────────────────────────────

def parse_reliability(records: list[dict]) -> dict:
    """
    Quantify LLM-specific failure modes.

    parse_success_rate    = sentences with valid JSON and Pydantic schema / total
    structural_validity_rate = sentences with no tree warnings / parse_success

    Returns
    -------
    {
        "total_sentences":         int,
        "api_errors":              int,
        "parse_success":           int,
        "parse_failures":          int,
        "parse_success_rate":      float,
        "no_root":                 int,
        "multiple_roots":          int,
        "any_structural_issue":    int,
        "structural_validity_rate": float,
    }
    """
    total         = len(records)
    api_errors    = sum(1 for r in records if r.get("api_error"))
    parse_success = sum(
        1 for r in records
        if r.get("validation_error") is None and r.get("llm_parse") is not None
    )
    no_root    = sum(1 for r in records if any("no_root"        in w for w in r.get("structural_warnings", [])))
    multi_root = sum(1 for r in records if any("multiple_roots" in w for w in r.get("structural_warnings", [])))
    any_struct = sum(1 for r in records if r.get("structural_warnings"))

    parse_success_rate   = round(parse_success / total, 4) if total          else 0.0
    struct_validity_rate = round((parse_success - any_struct) / parse_success, 4) if parse_success else 0.0

    return {
        "total_sentences":          total,
        "api_errors":               api_errors,
        "parse_success":            parse_success,
        "parse_failures":           total - parse_success,
        "parse_success_rate":       parse_success_rate,
        "no_root":                  no_root,
        "multiple_roots":           multi_root,
        "any_structural_issue":     any_struct,
        "structural_validity_rate": struct_validity_rate,
    }


# ─────────────────────────────────────────────
# Module D — Per-Relation Error Analysis
# ─────────────────────────────────────────────

def per_relation_analysis(
    records:      list[dict],
    exclude_punct: bool = True,
    min_count:    int  = 50,
) -> dict:
    """
    UAS and LAS per gold dependency relation label.

    Derivation uses existing token_errors field — no re-parsing needed:
      • Token NOT in token_errors     → head and deprel both correct (UAS ✓, LAS ✓)
      • error_type == "deprel_only"   → head correct, deprel wrong   (UAS ✓, LAS ✗)
      • error_type == "head_only"     → head wrong, deprel correct   (UAS ✗, LAS ✗)
      • error_type == "head_and_deprel" → both wrong                 (UAS ✗, LAS ✗)

    Returns dict keyed by deprel, sorted by token count descending.
    Only relations with total >= min_count are included.
    """
    total_cnt  = defaultdict(int)
    uas_ok_cnt = defaultdict(int)
    las_ok_cnt = defaultdict(int)

    for rec in scored_only(records):
        gold     = rec["gold"]
        pred_map = {t["id"]: t for t in rec["llm_parse"]}

        # Fast lookup: token_id → error_type for this sentence
        err_map = {e["token_id"]: e["error_type"] for e in rec.get("token_errors", [])}

        for g in gold:
            if exclude_punct and g["upos"] == "PUNCT":
                continue
            rel = g["deprel"]
            total_cnt[rel] += 1

            p = pred_map.get(g["id"])
            if p is None:
                continue  # missing token → both wrong

            err = err_map.get(g["id"])
            if err is None:
                # Not in error list → perfect match
                uas_ok_cnt[rel] += 1
                las_ok_cnt[rel] += 1
            elif err == "deprel_only":
                # Head correct, deprel wrong
                uas_ok_cnt[rel] += 1
            # "head_only" or "head_and_deprel" → leave both counters unchanged

    result = {}
    for rel, cnt in sorted(total_cnt.items(), key=lambda x: -x[1]):
        if cnt < min_count:
            continue
        result[rel] = {
            "total": cnt,
            "uas":   round(uas_ok_cnt[rel] / cnt, 4),
            "las":   round(las_ok_cnt[rel] / cnt, 4),
        }

    return result


# ─────────────────────────────────────────────
# Module E — Sentence-Length Analysis
# ─────────────────────────────────────────────

def length_analysis(records: list[dict]) -> dict:
    """
    Corpus-level UAS / LAS per sentence-length bin.

    Sentence length is the number of non-PUNCT tokens in gold.
    Bins: 1–10, 11–20, 21–30, 31–50, 51+.

    Returns
    -------
    {
        "1-10":  {"sentences": int, "total_tokens": int, "uas": float, "las": float},
        ...
    }
    """
    bins: dict[str, dict] = {
        lbl: {"sentences": 0, "total": 0, "uas_ok": 0, "las_ok": 0}
        for lbl in BIN_LABELS
    }

    for rec in scored_only(records):
        gold     = rec["gold"]
        pred_map = {t["id"]: t for t in rec["llm_parse"]}

        non_punct = [t for t in gold if t["upos"] != "PUNCT"]
        sent_len  = len(non_punct)

        # Assign to bin
        label = BIN_LABELS[-1]
        for (lo, hi), lbl in zip(LENGTH_BINS, BIN_LABELS):
            if lo <= sent_len <= hi:
                label = lbl
                break

        b = bins[label]
        b["sentences"] += 1

        for g in non_punct:
            b["total"] += 1
            p = pred_map.get(g["id"])
            if p is None:
                continue
            if p["head"] == g["head"]:
                b["uas_ok"] += 1
                if p["deprel"].lower() == g["deprel"].lower():
                    b["las_ok"] += 1

    result: dict[str, dict] = {}
    for lbl, b in bins.items():
        t = b["total"]
        result[lbl] = {
            "sentences":    b["sentences"],
            "total_tokens": t,
            "uas":          round(b["uas_ok"] / t, 4) if t else None,
            "las":          round(b["las_ok"] / t, 4) if t else None,
        }
    return result


# ─────────────────────────────────────────────
# Module F — Reasoning Trace Error Extraction
# ─────────────────────────────────────────────

# Manual annotation categories
REASONING_ERROR_CATEGORIES = [
    "root_misidentification",   # Model reasons plausibly but picks wrong head=0
    "label_confusion",          # Correct attachment, wrong deprel (e.g. obl vs nmod)
    "attachment_ambiguity",     # Model explicitly hedges / expresses uncertainty in trace
    "hallucinated_token",       # Predicted token not present in gold input
    "missing_token",            # Gold token absent from prediction
    "other",
]


def extract_reasoning_sample(
    records: list[dict],
    n:       int = 50,
    seed:    int = 42,
) -> list[dict]:
    """
    Extract up to n CoT error records for manual annotation.

    Selection criteria:
      • cot_reasoning is not None (reasoning trace was captured)
      • llm_parse is not None      (JSON was parseable)
      • At least one token error   (sentence is not perfectly scored)

    Biases towards low-LAS sentences (hardest errors) for analytical richness.

    Each output record has a blank "error_category" field for you to fill in
    from the REASONING_ERROR_CATEGORIES taxonomy.
    """
    candidates = [
        r for r in records
        if r.get("cot_reasoning")
        and r.get("llm_parse") is not None
        and r.get("token_errors")
    ]

    # Sort by LAS ascending; sample from the harder half
    candidates.sort(key=lambda r: r.get("las", 1.0))
    harder_half = candidates[: max(n * 2, len(candidates) // 2)]

    rng    = random.Random(seed)
    sample = rng.sample(harder_half, min(n, len(harder_half)))
    sample.sort(key=lambda r: r["sent_id"])

    return [
        {
            "sent_id":        rec["sent_id"],
            "text":           rec["text"],
            "las":            rec.get("las"),
            "cot_reasoning":  rec.get("cot_reasoning"),
            "token_errors":   rec.get("token_errors"),
            # ── Fill these in manually ──────────────────────────────────
            "error_category": "",   # one of REASONING_ERROR_CATEGORIES
            "notes":          "",   # optional free-text
        }
        for rec in sample
    ]


# ─────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────

def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"  [SAVED] {path}")


def print_summary_table(
    corpus_metrics: dict[str, dict],
    reliability:    dict[str, dict],
    test_set_size:  int = 0,
) -> None:
    """Print a combined summary table.

    Parameters
    ----------
    test_set_size : int
        Full test-set size used as the denominator in the Scored column.
        If 0, falls back to each experiment's own total_sentences.
    """
    cols = 76
    print(f"\n{'─' * cols}")
    print(f"  {'Experiment':<22} {'UAS':>6} {'LAS':>6} {'Scored':>10} {'Parse%':>8} {'Struct%':>8}")
    print(f"{'─' * cols}")
    for exp, cm in corpus_metrics.items():
        rel    = reliability.get(exp, {})
        total  = test_set_size or rel.get("total_sentences", "?")
        scored = cm.get("scored_sentences", "?")
        parse_pct  = rel.get("parse_success_rate",       0) * 100
        struct_pct = rel.get("structural_validity_rate", 0) * 100
        print(
            f"  {exp:<22} {cm['uas']:>6.4f} {cm['las']:>6.4f}"
            f" {scored:>5}/{total:<4} {parse_pct:>7.1f}% {struct_pct:>7.1f}%"
        )
    print(f"{'─' * cols}")


def print_significance_table(sig_results: dict) -> None:
    print(f"\n  {'Comparison':<48} {'Δ LAS':>8} {'p-value':>9} {'Sig?':>5}")
    print(f"  {'─' * 76}")
    for pair, res in sig_results.items():
        sig = "✓" if res["significant"] else "✗"
        print(
            f"  {pair:<48} {res['observed_diff']:>+8.4f}"
            f" {res['p_value']:>9.4f} {sig:>5}"
        )


def print_per_relation_table(
    merged:    dict[str, dict],
    exp_names: list[str],
    top_n:     int = 15,
) -> None:
    top = sorted(merged.items(), key=lambda x: -x[1].get("total", 0))[:top_n]
    col_w = 10
    hdr   = f"  {'Relation':<16} {'Count':>6}  " + \
            "  ".join(f"{'LAS_'+e[:8]:>{col_w}}" for e in exp_names)
    print(f"\n{hdr}")
    print(f"  {'─' * len(hdr)}")
    for rel, stats in top:
        vals = "  ".join(
            f"{stats[e]['las']:>{col_w}.4f}" if isinstance(stats.get(e), dict) else f"{'—':>{col_w}}"
            for e in exp_names
        )
        print(f"  {rel:<16} {stats.get('total', 0):>6}  {vals}")


def print_length_table(len_all: dict[str, dict], exp_names: list[str]) -> None:
    col_w = 16
    print(f"\n  {'Bin':<8}", end="")
    for exp in exp_names:
        print(f"  {'LAS_'+exp[:10]:>{col_w}}", end="")
    print()
    print(f"  {'─' * 80}")
    for label in BIN_LABELS:
        print(f"  {label:<8}", end="")
        for exp in exp_names:
            la  = len_all.get(exp, {}).get(label, {})
            las = la.get("las")
            cnt = la.get("sentences", 0)
            cell = f"{las:.4f} (n={cnt})" if las is not None else "—"
            print(f"  {cell:>{col_w}}", end="")
        print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research-grade evaluation pipeline for UD dependency parsing."
    )
    parser.add_argument(
        "--modules", nargs="+", default=ALL_MODULES, choices=ALL_MODULES,
        help="Modules to run (default: all)",
    )
    parser.add_argument(
        "--n-resamples", type=int, default=10_000,
        help="Bootstrap iterations (default: 10000)",
    )
    parser.add_argument(
        "--trace-n", type=int, default=50,
        help="CoT error sentences to extract for manual annotation (default: 50)",
    )
    parser.add_argument(
        "--min-rel-count", type=int, default=50,
        help="Min gold token count to include a deprel in the relation table (default: 50)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_SLUG,
        help="Model slug: reads outputs/<slug>/<exp>/ and writes results/<slug>/ "
             "(default: gpt_oss_120b).",
    )
    args = parser.parse_args()

    global EXPERIMENTS, RESULTS_DIR
    EXPERIMENTS = experiments_for(args.model)
    RESULTS_DIR = Path("results") / args.model

    modules = set(args.modules)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Model: {args.model}  |  inputs: outputs/{args.model}/  |  "
          f"outputs: {RESULTS_DIR}/")

    # ── Load all experiments ────────────────────────────────────────────────
    print("[INFO] Loading evaluation files …")
    data: dict[str, list[dict]] = {}
    for exp, path in EXPERIMENTS.items():
        p = Path(path)
        if not p.exists() and not Path(str(p) + ".gz").exists():
            print(f"  [WARN] {path} not found — skipping {exp}")
            continue
        data[exp] = load_experiment(path)
        print(f"  {exp:<22}  {len(data[exp])} records")

    if not data:
        print("[ERROR] No evaluation files found. Run evaluate_responses.py first.")
        sys.exit(1)

    corpus_metrics_all: dict[str, dict] = {}
    reliability_all:    dict[str, dict] = {}

    # ── Module A: Corpus-Level UAS / LAS ───────────────────────────────────
    if "corpus" in modules:
        print("\n[MODULE A] Corpus-Level UAS / LAS")
        for exp, records in data.items():
            cm = corpus_uas_las(records)
            corpus_metrics_all[exp] = cm
            print(
                f"  {exp:<22}  UAS={cm['uas']:.4f}  LAS={cm['las']:.4f}"
                f"  ({cm['scored_sentences']} sents, {cm['total_tokens']} tokens)"
            )
        save_json(corpus_metrics_all, RESULTS_DIR / "corpus_metrics.json")

    # ── Module C: Parse Reliability ─────────────────────────────────────────
    if "reliability" in modules:
        print("\n[MODULE C] Parse Reliability")
        for exp, records in data.items():
            rel = parse_reliability(records)
            reliability_all[exp] = rel
            print(
                f"  {exp:<22}  parse={rel['parse_success_rate']*100:.1f}%"
                f"  struct={rel['structural_validity_rate']*100:.1f}%"
                f"  (fail={rel['parse_failures']})"
            )
        save_json(reliability_all, RESULTS_DIR / "parse_reliability.json")

    # ── Combined Summary Table ──────────────────────────────────────────────
    if corpus_metrics_all and reliability_all:
        test_set_size = max(len(records) for records in data.values())
        print_summary_table(corpus_metrics_all, reliability_all, test_set_size)

    # ── Module D: Per-Relation Analysis ─────────────────────────────────────
    if "per-relation" in modules:
        print(f"\n[MODULE D] Per-Relation Error Analysis (min_count={args.min_rel_count})")
        per_rel_all: dict[str, dict] = {}
        for exp, records in data.items():
            pr = per_relation_analysis(records, min_count=args.min_rel_count)
            per_rel_all[exp] = pr
            print(f"  {exp:<22}  {len(pr)} relations above threshold")

        # Merge into one unified table keyed by relation
        merged: dict[str, dict] = {}
        for exp, pr in per_rel_all.items():
            for rel, stats in pr.items():
                if rel not in merged:
                    merged[rel] = {"total": stats["total"]}
                merged[rel][exp] = {"uas": stats["uas"], "las": stats["las"]}

        save_json(merged, RESULTS_DIR / "per_relation.json")
        print_per_relation_table(merged, list(data.keys()))

    # ── Module E: Sentence-Length Analysis ──────────────────────────────────
    if "length" in modules:
        print("\n[MODULE E] Sentence-Length Analysis")
        len_all: dict[str, dict] = {}
        for exp, records in data.items():
            len_all[exp] = length_analysis(records)
        save_json(len_all, RESULTS_DIR / "length_analysis.json")
        print_length_table(len_all, list(data.keys()))

    # ── Module B: Bootstrap Significance Tests ───────────────────────────────
    if "bootstrap" in modules:
        print(f"\n[MODULE B] Bootstrap Significance Testing — micro LAS ({args.n_resamples:,} resamples)")
        all_counts = {exp: per_sentence_las_counts(records) for exp, records in data.items()}
        sig        = run_all_significance_tests(all_counts, args.n_resamples)
        save_json(sig, RESULTS_DIR / "significance_tests.json")
        print_significance_table(sig)

    # ── Module F: Reasoning Trace Extraction ────────────────────────────────
    if "reasoning" in modules:
        print(f"\n[MODULE F] Reasoning Trace Error Extraction (n={args.trace_n})")
        cot_key = "cot_few_shot"
        if cot_key not in data:
            print(f"  [WARN] '{cot_key}' not in loaded experiments — skipping")
        else:
            sample   = extract_reasoning_sample(data[cot_key], n=args.trace_n)
            out_path = RESULTS_DIR / "reasoning_trace_errors.json"
            save_json(sample, out_path)
            print(f"  Extracted {len(sample)} error sentences.")
            print(f"  → Open {out_path} and fill in 'error_category' for each record.")
            print(f"    Valid categories: {', '.join(REASONING_ERROR_CATEGORIES)}")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
