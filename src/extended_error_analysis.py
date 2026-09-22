"""
extended_error_analysis.py
──────────────────────────
Extended error analysis for "Dependency Parsing by LLMs".

Reads the model_evaluation.json files produced by evaluate_responses.py
and computes additional analyses beyond the core analyze_results.py:

  G. Per-sentence score distributions  (box plots + histograms)
  H. Per-label deprel P / R / F1       (precision, recall, F1 per relation)
  I. Arc direction & arc length        (UAS by arc distance and direction)
  J. Parse failure characterization    (length/genre/error-type breakdown)
  K. UPOS-stratified accuracy          (UAS/LAS grouped by dependent POS)
  L. Deprel confusion matrix           (gold→pred label confusions)
  M. EWT genre stratification          (UAS/LAS per genre)

All results are written to results/extended/ as JSON + PDF figures.

Usage
─────
  python src/extended_error_analysis.py [OPTIONS]

  --modules        Which modules to run (default: all)
                   distribution  deprel-f1  arc  failures  upos  confusion  genre
  --min-rel-count  Minimum token count to include a deprel (default: 50)
  --top-n          Top N relations for confusion matrix / charts (default: 15)
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

EXPERIMENT_NAMES = [
    "zero_shot", "few_shot_fixed", "few_shot_retrieval",
    "cot_few_shot", "critique_refine", "semantic_retrieval",
]
DEFAULT_MODEL_SLUG = "gpt_oss_120b"


def experiments_for(slug: str) -> dict[str, str]:
    return {e: f"outputs/{slug}/{e}/model_evaluation.json" for e in EXPERIMENT_NAMES}


# Model-scoped by default; overridden in main() from --model.
EXPERIMENTS: dict[str, str] = experiments_for(DEFAULT_MODEL_SLUG)
RESULTS_DIR = Path("results") / DEFAULT_MODEL_SLUG / "extended"

# Short display names for plots
SHORT_NAMES = {
    "zero_shot":          "Zero-Shot",
    "few_shot_fixed":     "Few-Shot Fixed",
    "few_shot_retrieval": "POS-Signature Few-Shot",
    "cot_few_shot":       "CoT Few-Shot",
    "critique_refine":    "Critique-Refine",
    "semantic_retrieval": "Semantic Few-Shot",
}

# EWT genres extracted from sent_id prefix
GENRE_ORDER = ["email", "reviews", "answers", "newsgroup", "weblog"]

# Arc length bins
ARC_LENGTH_BINS   = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 9_999)]
ARC_LENGTH_LABELS = ["1", "2", "3", "4", "5", "6-10", "11+"]

ALL_MODULES = [
    "distribution", "deprel-f1", "arc", "failures", "upos", "confusion", "genre",
]

# Plot style
COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#577590"]


# ─────────────────────────────────────────────
# Helpers
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


def extract_genre(sent_id: str) -> str:
    """Extract EWT genre from sent_id prefix (e.g. 'email-enron...' → 'email')."""
    return sent_id.split("-")[0]


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    print(f"  [SAVED] {path}")


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [SAVED] {path}")


# ─────────────────────────────────────────────
# Module G — Per-Sentence Score Distribution
# ─────────────────────────────────────────────

def per_sentence_scores(records: list[dict]) -> dict[str, list[float]]:
    """Returns per-sentence UAS and LAS lists (parse failures → 0.0)."""
    uas_list, las_list = [], []
    for r in records:
        if r.get("llm_parse") is None:
            uas_list.append(0.0)
            las_list.append(0.0)
        else:
            uas_list.append(r.get("uas") or 0.0)
            las_list.append(r.get("las") or 0.0)
    return {"uas": uas_list, "las": las_list}


def run_distribution(data: dict[str, list[dict]]) -> None:
    print("\n[MODULE G] Per-Sentence Score Distribution")

    all_scores: dict[str, dict] = {}
    for exp, records in data.items():
        all_scores[exp] = per_sentence_scores(records)

    # Save raw scores
    save_json(
        {exp: {"uas": s["uas"], "las": s["las"]} for exp, s in all_scores.items()},
        RESULTS_DIR / "score_distributions.json",
    )

    exp_names = list(data.keys())

    # ── Box plots (LAS) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    box_data = [all_scores[e]["las"] for e in exp_names]
    bp = ax.boxplot(
        box_data,
        labels=[SHORT_NAMES[e] for e in exp_names],
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white", markersize=5),
    )
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("LAS (per sentence)")
    ax.set_title("Per-Sentence LAS Distribution Across Prompting Strategies")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, RESULTS_DIR / "fig_score_distribution_box.pdf")

    # ── Histograms (LAS) ─────────────────────────────────
    fig, axes = plt.subplots(1, len(exp_names), figsize=(4 * len(exp_names), 4), sharey=True)
    for i, exp in enumerate(exp_names):
        ax = axes[i]
        ax.hist(all_scores[exp]["las"], bins=20, range=(0, 1),
                color=COLORS[i], alpha=0.7, edgecolor="white")
        ax.set_title(SHORT_NAMES[exp], fontsize=10)
        ax.set_xlabel("LAS")
        if i == 0:
            ax.set_ylabel("Sentence count")
        ax.axvline(np.mean(all_scores[exp]["las"]), color="red",
                   linestyle="--", linewidth=1, label=f"mean={np.mean(all_scores[exp]['las']):.3f}")
        ax.legend(fontsize=7)
    fig.suptitle("Per-Sentence LAS Histograms", y=1.02)
    fig.tight_layout()
    save_fig(fig, RESULTS_DIR / "fig_score_distribution_hist.pdf")

    # Print summary stats
    print(f"\n  {'Experiment':<22} {'Mean LAS':>9} {'Median':>8} {'Std':>8} {'LAS=0':>7} {'LAS=1':>7}")
    for exp in exp_names:
        las = all_scores[exp]["las"]
        print(
            f"  {exp:<22} {np.mean(las):>9.4f} {np.median(las):>8.4f} "
            f"{np.std(las):>8.4f} {sum(1 for x in las if x == 0.0):>7} "
            f"{sum(1 for x in las if x == 1.0):>7}"
        )


# ─────────────────────────────────────────────
# Module H — Per-Label Deprel P / R / F1
# ─────────────────────────────────────────────

def deprel_prf1(
    records: list[dict],
    exclude_punct: bool = True,
    min_count: int = 50,
) -> dict:
    """
    Compute precision, recall, and F1 for each deprel label (labeled attachment).

    Precision(rel) = TP(rel) / (TP(rel) + FP(rel))
      TP = gold_deprel == pred_deprel AND head correct
      FP = pred_deprel == rel but gold_deprel != rel (or head wrong)

    Recall(rel) = TP(rel) / (TP(rel) + FN(rel))
      FN = gold_deprel == rel but (pred_deprel != rel or head wrong)
    """
    # Count TP, FP, FN per relation
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    gold_total = defaultdict(int)

    for rec in scored_only(records):
        gold = rec["gold"]
        pred_map = {t["id"]: t for t in rec["llm_parse"]}

        for g in gold:
            if exclude_punct and g["upos"] == "PUNCT":
                continue

            g_rel = g["deprel"].lower()
            gold_total[g_rel] += 1
            p = pred_map.get(g["id"])

            if p is None:
                # Missing token → FN for the gold relation
                fn[g_rel] += 1
                continue

            p_rel = p["deprel"].lower()
            head_correct = p["head"] == g["head"]

            if head_correct and p_rel == g_rel:
                # True positive: correct head AND correct label
                tp[g_rel] += 1
            else:
                # FN for gold relation (missed it)
                fn[g_rel] += 1
                # FP for predicted relation (predicted it wrongly)
                # Only count FP if head is correct but label is wrong,
                # OR if this is a label that was predicted but shouldn't have been
                if head_correct and p_rel != g_rel:
                    fp[p_rel] += 1
                elif not head_correct:
                    # Head wrong → the predicted deprel is attached to wrong head
                    # This is an FP for the predicted label
                    fp[p_rel] += 1

    result = {}
    all_rels = set(gold_total.keys())
    for rel in sorted(all_rels, key=lambda r: -gold_total[r]):
        if gold_total[rel] < min_count:
            continue
        t = tp[rel]
        f_p = fp[rel]
        f_n = fn[rel]
        prec = t / (t + f_p) if (t + f_p) > 0 else 0.0
        rec_ = t / (t + f_n) if (t + f_n) > 0 else 0.0
        f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
        result[rel] = {
            "gold_count": gold_total[rel],
            "tp": t, "fp": f_p, "fn": f_n,
            "precision": round(prec, 4),
            "recall":    round(rec_, 4),
            "f1":        round(f1, 4),
        }

    return result


def run_deprel_f1(data: dict[str, list[dict]], min_count: int, top_n: int) -> None:
    print(f"\n[MODULE H] Per-Label Deprel Precision / Recall / F1 (min_count={min_count})")

    all_prf1: dict[str, dict] = {}
    for exp, records in data.items():
        prf = deprel_prf1(records, min_count=min_count)
        all_prf1[exp] = prf
        print(f"  {exp:<22}  {len(prf)} relations above threshold")

    save_json(all_prf1, RESULTS_DIR / "deprel_prf1.json")

    # Merged view for chart: find common relations across all experiments
    exp_names = list(data.keys())
    # Use the first experiment's relation order (sorted by gold count)
    first_exp = exp_names[0]
    all_rels = list(all_prf1[first_exp].keys())[:top_n]

    # ── Grouped bar chart (F1 for top relations) ──────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_rels))
    width = 0.15
    for i, exp in enumerate(exp_names):
        f1_vals = [all_prf1[exp].get(r, {}).get("f1", 0) for r in all_rels]
        ax.bar(x + i * width, f1_vals, width, label=SHORT_NAMES[exp],
               color=COLORS[i], alpha=0.8)
    ax.set_xticks(x + width * (len(exp_names) - 1) / 2)
    ax.set_xticklabels(all_rels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("F1 Score")
    ax.set_title(f"Per-Relation Labeled F1 (Top {top_n} by Frequency)")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, RESULTS_DIR / "fig_deprel_f1.pdf")

    # Print table
    print(f"\n  {'Relation':<16} {'Gold':>6}  ", end="")
    for exp in exp_names:
        print(f"{'F1_'+exp[:8]:>12}", end="")
    print()
    print(f"  {'─' * 80}")
    for rel in all_rels:
        cnt = all_prf1[first_exp].get(rel, {}).get("gold_count", 0)
        print(f"  {rel:<16} {cnt:>6}  ", end="")
        for exp in exp_names:
            f1 = all_prf1[exp].get(rel, {}).get("f1", 0)
            print(f"{f1:>12.4f}", end="")
        print()


# ─────────────────────────────────────────────
# Module I — Arc Direction & Arc Length
# ─────────────────────────────────────────────

def arc_analysis(records: list[dict], exclude_punct: bool = True) -> dict:
    """
    UAS by arc direction (left/right) and arc length bins.
    Arc direction: left if head < dependent, right if head > dependent.
    Root tokens (head=0) are excluded from direction analysis.
    """
    # Direction
    dir_stats = {
        "left":  {"total": 0, "correct": 0},
        "right": {"total": 0, "correct": 0},
    }

    # Arc length bins
    len_stats = {lbl: {"total": 0, "correct": 0} for lbl in ARC_LENGTH_LABELS}

    for rec in scored_only(records):
        gold = rec["gold"]
        pred_map = {t["id"]: t for t in rec["llm_parse"]}

        for g in gold:
            if exclude_punct and g["upos"] == "PUNCT":
                continue
            if g["head"] == 0:
                # Root — skip from direction analysis but include in arc length
                # (arc length for root is undefined, skip)
                continue

            p = pred_map.get(g["id"])
            head_correct = p is not None and p["head"] == g["head"]

            # Direction
            direction = "left" if g["head"] < g["id"] else "right"
            dir_stats[direction]["total"] += 1
            if head_correct:
                dir_stats[direction]["correct"] += 1

            # Arc length
            arc_len = abs(g["head"] - g["id"])
            label = ARC_LENGTH_LABELS[-1]
            for (lo, hi), lbl in zip(ARC_LENGTH_BINS, ARC_LENGTH_LABELS):
                if lo <= arc_len <= hi:
                    label = lbl
                    break
            len_stats[label]["total"] += 1
            if head_correct:
                len_stats[label]["correct"] += 1

    # Compute UAS rates
    dir_result = {}
    for d, s in dir_stats.items():
        dir_result[d] = {
            "total": s["total"],
            "correct": s["correct"],
            "uas": round(s["correct"] / s["total"], 4) if s["total"] > 0 else None,
        }

    len_result = {}
    for lbl in ARC_LENGTH_LABELS:
        s = len_stats[lbl]
        len_result[lbl] = {
            "total": s["total"],
            "correct": s["correct"],
            "uas": round(s["correct"] / s["total"], 4) if s["total"] > 0 else None,
        }

    return {"direction": dir_result, "arc_length": len_result}


def run_arc(data: dict[str, list[dict]]) -> None:
    print("\n[MODULE I] Arc Direction & Arc Length Analysis")

    all_arc: dict[str, dict] = {}
    exp_names = list(data.keys())

    for exp, records in data.items():
        aa = arc_analysis(records)
        all_arc[exp] = aa
        left_uas = aa["direction"]["left"]["uas"]
        right_uas = aa["direction"]["right"]["uas"]
        print(f"  {exp:<22}  left UAS={left_uas:.4f}  right UAS={right_uas:.4f}")

    save_json(all_arc, RESULTS_DIR / "arc_analysis.json")

    # ── Arc length UAS curve ──────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, exp in enumerate(exp_names):
        uas_vals = [all_arc[exp]["arc_length"][lbl]["uas"] or 0 for lbl in ARC_LENGTH_LABELS]
        ax.plot(ARC_LENGTH_LABELS, uas_vals, marker="o", color=COLORS[i],
                label=SHORT_NAMES[exp], linewidth=2, markersize=5)
    ax.set_xlabel("Arc Length (tokens between head and dependent)")
    ax.set_ylabel("UAS")
    ax.set_title("UAS by Arc Length")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_fig(fig, RESULTS_DIR / "fig_arc_length_uas.pdf")

    # ── Direction bar chart ───────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(exp_names))
    width = 0.35
    left_vals = [all_arc[e]["direction"]["left"]["uas"] or 0 for e in exp_names]
    right_vals = [all_arc[e]["direction"]["right"]["uas"] or 0 for e in exp_names]
    ax.bar(x - width / 2, left_vals, width, label="Left-arc", color="#264653", alpha=0.8)
    ax.bar(x + width / 2, right_vals, width, label="Right-arc", color="#e76f51", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_NAMES[e] for e in exp_names], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("UAS")
    ax.set_title("UAS by Arc Direction (Left vs Right)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, RESULTS_DIR / "fig_arc_direction.pdf")

    # Print arc length table
    print(f"\n  {'Arc Len':<10}", end="")
    for exp in exp_names:
        print(f"  {'UAS_'+exp[:8]:>14}", end="")
    print()
    print(f"  {'─' * 85}")
    for lbl in ARC_LENGTH_LABELS:
        print(f"  {lbl:<10}", end="")
        for exp in exp_names:
            uas = all_arc[exp]["arc_length"][lbl]["uas"]
            cnt = all_arc[exp]["arc_length"][lbl]["total"]
            cell = f"{uas:.4f} ({cnt})" if uas is not None else "—"
            print(f"  {cell:>14}", end="")
        print()


# ─────────────────────────────────────────────
# Module J — Parse Failure Characterization
# ─────────────────────────────────────────────

def run_failures(data: dict[str, list[dict]]) -> None:
    print("\n[MODULE J] Parse Failure Characterization")

    all_failures: dict[str, dict] = {}

    for exp, records in data.items():
        failed = [r for r in records if r.get("llm_parse") is None]
        if not failed:
            all_failures[exp] = {"count": 0, "sentences": []}
            continue

        # Characterize each failure
        lengths = []
        genres = []
        error_types = []
        failure_records = []

        for r in failed:
            gold = r.get("gold", [])
            non_punct_len = sum(1 for t in gold if t.get("upos") != "PUNCT")
            genre = extract_genre(r["sent_id"])
            err_type = "api_error" if r.get("api_error") else "validation_error"

            lengths.append(non_punct_len)
            genres.append(genre)
            error_types.append(err_type)
            failure_records.append({
                "sent_id":    r["sent_id"],
                "text":       r["text"],
                "length":     non_punct_len,
                "genre":      genre,
                "error_type": err_type,
                "validation_error": r.get("validation_error", ""),
            })

        # Compute distributions
        success = [r for r in records if r.get("llm_parse") is not None]
        success_lengths = [
            sum(1 for t in r["gold"] if t.get("upos") != "PUNCT") for r in success
        ]

        all_failures[exp] = {
            "count":            len(failed),
            "mean_length":      round(np.mean(lengths), 1) if lengths else 0,
            "median_length":    float(np.median(lengths)) if lengths else 0,
            "success_mean_length": round(np.mean(success_lengths), 1) if success_lengths else 0,
            "genre_dist":       dict(Counter(genres).most_common()),
            "error_type_dist":  dict(Counter(error_types).most_common()),
            "sentences":        failure_records,
        }

        print(f"  {exp:<22}  {len(failed)} failures, "
              f"mean len={all_failures[exp]['mean_length']}, "
              f"genres={dict(Counter(genres).most_common())}")

    save_json(all_failures, RESULTS_DIR / "parse_failures.json")

    # Print summary
    print(f"\n  {'Experiment':<22} {'Fails':>6} {'Mean Len':>9} {'Succ Len':>9} {'Genres'}")
    print(f"  {'─' * 75}")
    for exp, info in all_failures.items():
        if info["count"] == 0:
            print(f"  {exp:<22} {'0':>6}")
        else:
            print(
                f"  {exp:<22} {info['count']:>6} {info['mean_length']:>9.1f} "
                f"{info['success_mean_length']:>9.1f}  {info['genre_dist']}"
            )


# ─────────────────────────────────────────────
# Module K — UPOS-Stratified Accuracy
# ─────────────────────────────────────────────

def upos_accuracy(records: list[dict], exclude_punct: bool = True) -> dict:
    """UAS and LAS grouped by gold UPOS tag of the dependent."""
    stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "uas_ok": 0, "las_ok": 0})

    for rec in scored_only(records):
        gold = rec["gold"]
        pred_map = {t["id"]: t for t in rec["llm_parse"]}

        for g in gold:
            if exclude_punct and g["upos"] == "PUNCT":
                continue
            upos = g["upos"]
            stats[upos]["total"] += 1

            p = pred_map.get(g["id"])
            if p is None:
                continue
            if p["head"] == g["head"]:
                stats[upos]["uas_ok"] += 1
                if p["deprel"].lower() == g["deprel"].lower():
                    stats[upos]["las_ok"] += 1

    result = {}
    for upos in sorted(stats.keys(), key=lambda u: -stats[u]["total"]):
        s = stats[upos]
        result[upos] = {
            "total": s["total"],
            "uas": round(s["uas_ok"] / s["total"], 4) if s["total"] else 0,
            "las": round(s["las_ok"] / s["total"], 4) if s["total"] else 0,
        }
    return result


def run_upos(data: dict[str, list[dict]]) -> None:
    print("\n[MODULE K] UPOS-Stratified Accuracy")

    all_upos: dict[str, dict] = {}
    exp_names = list(data.keys())

    for exp, records in data.items():
        ua = upos_accuracy(records)
        all_upos[exp] = ua

    save_json(all_upos, RESULTS_DIR / "upos_accuracy.json")

    # Get all UPOS tags sorted by frequency (from first experiment)
    first = exp_names[0]
    upos_order = list(all_upos[first].keys())

    # ── Grouped bar chart (LAS by UPOS) ──────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(upos_order))
    width = 0.15
    for i, exp in enumerate(exp_names):
        las_vals = [all_upos[exp].get(u, {}).get("las", 0) for u in upos_order]
        ax.bar(x + i * width, las_vals, width, label=SHORT_NAMES[exp],
               color=COLORS[i], alpha=0.8)
    ax.set_xticks(x + width * (len(exp_names) - 1) / 2)
    ax.set_xticklabels(upos_order, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("LAS")
    ax.set_title("LAS by UPOS Tag of Dependent Token")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, RESULTS_DIR / "fig_upos_las.pdf")

    # Print table
    print(f"\n  {'UPOS':<10} {'Count':>6}", end="")
    for exp in exp_names:
        print(f"  {'LAS_'+exp[:8]:>12}", end="")
    print()
    print(f"  {'─' * 80}")
    for upos in upos_order:
        cnt = all_upos[first].get(upos, {}).get("total", 0)
        print(f"  {upos:<10} {cnt:>6}", end="")
        for exp in exp_names:
            las = all_upos[exp].get(upos, {}).get("las", 0)
            print(f"  {las:>12.4f}", end="")
        print()


# ─────────────────────────────────────────────
# Module L — Deprel Confusion Matrix
# ─────────────────────────────────────────────

def deprel_confusion(
    records: list[dict],
    exclude_punct: bool = True,
    top_n: int = 15,
) -> dict:
    """
    Build a confusion matrix for deprel labels.
    Only considers tokens where the head is CORRECT but the label is WRONG
    (i.e., 'deprel_only' errors), which isolates pure labeling confusion
    from attachment errors.
    """
    confusion: dict[str, Counter] = defaultdict(Counter)
    gold_counts: Counter = Counter()

    for rec in scored_only(records):
        gold = rec["gold"]
        pred_map = {t["id"]: t for t in rec["llm_parse"]}

        for g in gold:
            if exclude_punct and g["upos"] == "PUNCT":
                continue
            g_rel = g["deprel"].lower()
            gold_counts[g_rel] += 1

            p = pred_map.get(g["id"])
            if p is None:
                continue
            p_rel = p["deprel"].lower()
            # Only count where head is correct but label is wrong
            if p["head"] == g["head"] and p_rel != g_rel:
                confusion[g_rel][p_rel] += 1

    # Keep only top-N relations by gold frequency
    top_rels = [rel for rel, _ in gold_counts.most_common(top_n)]

    # Build matrix dict
    matrix = {}
    for g_rel in top_rels:
        top_confusions = confusion[g_rel].most_common(5)
        matrix[g_rel] = {
            "gold_count": gold_counts[g_rel],
            "total_label_errors": sum(confusion[g_rel].values()),
            "top_confusions": [
                {"predicted": p_rel, "count": cnt} for p_rel, cnt in top_confusions
            ],
        }

    return matrix


def run_confusion(data: dict[str, list[dict]], top_n: int) -> None:
    print(f"\n[MODULE L] Deprel Confusion Matrix (top {top_n} relations)")

    all_confusion: dict[str, dict] = {}
    exp_names = list(data.keys())

    for exp, records in data.items():
        cm = deprel_confusion(records, top_n=top_n)
        all_confusion[exp] = cm

    save_json(all_confusion, RESULTS_DIR / "deprel_confusion.json")

    # Generate heatmap for best-performing experiment (few_shot_retrieval)
    best_exp = "few_shot_retrieval"
    if best_exp not in all_confusion:
        best_exp = exp_names[0]

    cm = all_confusion[best_exp]
    rels = list(cm.keys())

    # Build a full confusion count matrix for the top relations
    # Row = gold, Col = predicted (only label errors with correct head)
    matrix_data = np.zeros((len(rels), len(rels)))
    for i, g_rel in enumerate(rels):
        for conf in cm[g_rel]["top_confusions"]:
            p_rel = conf["predicted"]
            if p_rel in rels:
                j = rels.index(p_rel)
                matrix_data[i, j] = conf["count"]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix_data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(rels)))
    ax.set_yticks(range(len(rels)))
    ax.set_xticklabels(rels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(rels, fontsize=8)
    ax.set_xlabel("Predicted Deprel")
    ax.set_ylabel("Gold Deprel")
    ax.set_title(f"Deprel Confusion Matrix — {SHORT_NAMES[best_exp]}\n(correct head, wrong label)")
    plt.colorbar(im, ax=ax, label="Count")
    # Add text annotations for non-zero cells
    for i in range(len(rels)):
        for j in range(len(rels)):
            if matrix_data[i, j] > 0:
                ax.text(j, i, int(matrix_data[i, j]),
                        ha="center", va="center", fontsize=7,
                        color="white" if matrix_data[i, j] > matrix_data.max() * 0.5 else "black")
    fig.tight_layout()
    save_fig(fig, RESULTS_DIR / "fig_deprel_confusion.pdf")

    # Print top confusions for each relation (best experiment)
    print(f"\n  Top label confusions for {SHORT_NAMES[best_exp]}:")
    print(f"  {'Gold Deprel':<16} {'Label Errs':>11}  Top Confusions")
    print(f"  {'─' * 70}")
    for rel in rels:
        info = cm[rel]
        confs = ", ".join(
            f"{c['predicted']}({c['count']})" for c in info["top_confusions"][:3]
        )
        print(f"  {rel:<16} {info['total_label_errors']:>11}  {confs}")


# ─────────────────────────────────────────────
# Module M — EWT Genre Stratification
# ─────────────────────────────────────────────

def genre_analysis(records: list[dict], exclude_punct: bool = True) -> dict:
    """Corpus-level UAS/LAS per EWT genre."""
    stats: dict[str, dict] = {
        g: {"total": 0, "uas_ok": 0, "las_ok": 0, "sentences": 0}
        for g in GENRE_ORDER
    }

    for rec in scored_only(records):
        genre = extract_genre(rec["sent_id"])
        if genre not in stats:
            continue
        stats[genre]["sentences"] += 1

        gold = rec["gold"]
        pred_map = {t["id"]: t for t in rec["llm_parse"]}

        for g in gold:
            if exclude_punct and g["upos"] == "PUNCT":
                continue
            stats[genre]["total"] += 1
            p = pred_map.get(g["id"])
            if p is None:
                continue
            if p["head"] == g["head"]:
                stats[genre]["uas_ok"] += 1
                if p["deprel"].lower() == g["deprel"].lower():
                    stats[genre]["las_ok"] += 1

    result = {}
    for genre in GENRE_ORDER:
        s = stats[genre]
        result[genre] = {
            "sentences": s["sentences"],
            "total_tokens": s["total"],
            "uas": round(s["uas_ok"] / s["total"], 4) if s["total"] else None,
            "las": round(s["las_ok"] / s["total"], 4) if s["total"] else None,
        }
    return result


def run_genre(data: dict[str, list[dict]]) -> None:
    print("\n[MODULE M] EWT Genre Stratification")

    all_genre: dict[str, dict] = {}
    exp_names = list(data.keys())

    for exp, records in data.items():
        ga = genre_analysis(records)
        all_genre[exp] = ga

    save_json(all_genre, RESULTS_DIR / "genre_analysis.json")

    # ── Grouped bar chart (LAS by genre) ──────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(GENRE_ORDER))
    width = 0.15
    for i, exp in enumerate(exp_names):
        las_vals = [all_genre[exp].get(g, {}).get("las", 0) or 0 for g in GENRE_ORDER]
        ax.bar(x + i * width, las_vals, width, label=SHORT_NAMES[exp],
               color=COLORS[i], alpha=0.8)
    ax.set_xticks(x + width * (len(exp_names) - 1) / 2)
    ax.set_xticklabels([g.capitalize() for g in GENRE_ORDER], fontsize=10)
    ax.set_ylabel("LAS")
    ax.set_title("LAS by EWT Genre")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, RESULTS_DIR / "fig_genre_las.pdf")

    # Print table
    print(f"\n  {'Genre':<12} {'Sents':>6}", end="")
    for exp in exp_names:
        print(f"  {'LAS_'+exp[:10]:>16}", end="")
    print()
    print(f"  {'─' * 100}")
    for genre in GENRE_ORDER:
        sents = all_genre[exp_names[0]].get(genre, {}).get("sentences", 0)
        print(f"  {genre:<12} {sents:>6}", end="")
        for exp in exp_names:
            las = all_genre[exp].get(genre, {}).get("las")
            cell = f"{las:.4f}" if las is not None else "—"
            print(f"  {cell:>16}", end="")
        print()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extended error analysis for UD dependency parsing."
    )
    parser.add_argument(
        "--modules", nargs="+", default=ALL_MODULES, choices=ALL_MODULES,
        help="Modules to run (default: all)",
    )
    parser.add_argument(
        "--min-rel-count", type=int, default=50,
        help="Min gold token count to include a deprel (default: 50)",
    )
    parser.add_argument(
        "--top-n", type=int, default=15,
        help="Top N relations for charts / confusion matrix (default: 15)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_SLUG,
        help="Model slug: reads outputs/<slug>/<exp>/ and writes "
             "results/<slug>/extended/ (default: gpt_oss_120b).",
    )
    args = parser.parse_args()

    global EXPERIMENTS, RESULTS_DIR
    EXPERIMENTS = experiments_for(args.model)
    RESULTS_DIR = Path("results") / args.model / "extended"

    modules = set(args.modules)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Model: {args.model}  |  writing to {RESULTS_DIR}/")

    # ── Load all experiments ────────────────────────────────
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

    # ── Run selected modules ────────────────────────────────
    if "distribution" in modules:
        run_distribution(data)

    if "deprel-f1" in modules:
        run_deprel_f1(data, args.min_rel_count, args.top_n)

    if "arc" in modules:
        run_arc(data)

    if "failures" in modules:
        run_failures(data)

    if "upos" in modules:
        run_upos(data)

    if "confusion" in modules:
        run_confusion(data, args.top_n)

    if "genre" in modules:
        run_genre(data)

    print("\n[INFO] Done. All outputs saved to results/extended/")


if __name__ == "__main__":
    main()
