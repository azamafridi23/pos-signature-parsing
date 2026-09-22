#!/usr/bin/env python3
"""Regenerate the paper's quantitative results from archived evaluations.

This is the authoritative offline analysis path for the paper. Every condition
is aligned to the same 2,077-sentence gold reference. Missing records,
unscorable responses, and missing token identifiers receive zero credit while
their gold non-punctuation tokens remain in the denominator.

Outputs are written under ``results``. The script also
updates the full-denominator relation/length JSON files and their paper figures.
No API access is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import gzip
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised by dependency check
    raise SystemExit(
        "numpy is required. Install requirements.txt or run "
        "scripts/reproduce.sh with a compatible PYTHON_BIN."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results"
REFERENCE_PATH = ROOT / "outputs/gpt_oss_120b/zero_shot/model_evaluation.json.gz"
RELATION_PATH = ROOT / "results/per_relation.json"
LENGTH_PATH = ROOT / "results/length_analysis.json"

MODELS = {
    "gpt_oss_120b": "gpt-oss-120b",
    "qwen25_72b": "Qwen2.5-72B",
}
CONDITIONS = [
    ("zero_shot", "Zero-Shot"),
    ("few_shot_fixed", "Fixed Few-Shot"),
    ("few_shot_retrieval", "POS-Signature Few-Shot"),
    ("semantic_retrieval", "Semantic Few-Shot"),
    ("cot_few_shot", "Chain-of-Thought"),
    ("critique_refine", "Critique-Refine"),
]
CONDITION_LABEL = dict(CONDITIONS)
LENGTH_BINS = [
    ("1–10", 1, 10),
    ("11–20", 11, 20),
    ("21–30", 21, 30),
    ("31–50", 31, 50),
    ("51+", 51, 10**9),
]
PRIMARY_PAIRS = {
    frozenset(("few_shot_retrieval", "semantic_retrieval")),
    frozenset(("few_shot_retrieval", "few_shot_fixed")),
}


def load_json(path: Path) -> Any:
    if not path.exists():
        path = Path(str(path) + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def score_tokens(
    gold: list[dict[str, Any]],
    prediction: list[dict[str, Any]] | None,
) -> tuple[int, int, int, dict[int, tuple[bool, bool]]]:
    """Return denominator, UAS-correct, LAS-correct, and token correctness."""
    pred_by_id = {
        token["id"]: token
        for token in prediction or []
        if isinstance(token, dict) and isinstance(token.get("id"), int)
    }
    total = uas_correct = las_correct = 0
    token_status: dict[int, tuple[bool, bool]] = {}
    for gold_token in gold:
        if gold_token["upos"] == "PUNCT":
            continue
        total += 1
        predicted = pred_by_id.get(gold_token["id"])
        uas = predicted is not None and predicted.get("head") == gold_token["head"]
        las = (
            uas
            and str(predicted.get("deprel", "")).lower()
            == gold_token["deprel"].lower()
        )
        uas_correct += int(uas)
        las_correct += int(las)
        token_status[gold_token["id"]] = (uas, las)
    return total, uas_correct, las_correct, token_status


def structural_audit(
    gold: list[dict[str, Any]],
    prediction: list[dict[str, Any]] | None,
) -> dict[str, bool]:
    """Perform non-scoring tree checks over the gold token inventory.

    The original evaluator checked only the number of predicted ``head=0``
    tokens. This post-hoc audit additionally checks exact token inventory, legal
    head ranges, self-loops, and whether every token reaches the single root.
    """
    if prediction is None:
        return {
            "schema_valid": False,
            "exact_token_inventory": False,
            "root_conforming": False,
            "legal_heads": False,
            "self_loop_free": False,
            "acyclic_and_connected": False,
            "fully_tree_conforming": False,
        }

    expected_ids = {token["id"] for token in gold}
    predicted_ids = [
        token.get("id") for token in prediction if isinstance(token, dict)
    ]
    integer_ids = all(isinstance(token_id, int) for token_id in predicted_ids)
    exact_inventory = (
        integer_ids
        and len(predicted_ids) == len(expected_ids)
        and len(set(predicted_ids)) == len(predicted_ids)
        and set(predicted_ids) == expected_ids
    )
    roots = [
        token
        for token in prediction
        if isinstance(token, dict) and token.get("head") == 0
    ]
    root_conforming = len(roots) == 1
    legal_heads = all(
        isinstance(token, dict)
        and isinstance(token.get("head"), int)
        and (token["head"] == 0 or token["head"] in expected_ids)
        for token in prediction
    )
    self_loop_free = all(
        isinstance(token, dict)
        and token.get("head") != token.get("id")
        for token in prediction
    )

    reaches_root = False
    if exact_inventory and root_conforming and legal_heads and self_loop_free:
        pred_by_id = {token["id"]: token for token in prediction}
        reaches_root = True
        for start in expected_ids:
            current = start
            seen: set[int] = set()
            while current != 0:
                if current in seen or current not in pred_by_id:
                    reaches_root = False
                    break
                seen.add(current)
                current = pred_by_id[current]["head"]
            if not reaches_root:
                break

    return {
        "schema_valid": True,
        "exact_token_inventory": exact_inventory,
        "root_conforming": root_conforming,
        "legal_heads": legal_heads,
        "self_loop_free": self_loop_free,
        "acyclic_and_connected": reaches_root,
        "fully_tree_conforming": (
            exact_inventory
            and root_conforming
            and legal_heads
            and self_loop_free
            and reaches_root
        ),
    }


def parse_conllu_stats(path: Path) -> dict[str, int]:
    sentences = words = punct = 0
    saw_word = False
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                if saw_word:
                    sentences += 1
                    saw_word = False
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 4 or not fields[0].isdigit():
                continue
            saw_word = True
            words += 1
            punct += int(fields[3] == "PUNCT")
    if saw_word:
        sentences += 1
    return {
        "sentences": sentences,
        "syntactic_words": words,
        "gold_punct": punct,
        "non_punctuation_words": words - punct,
    }


def build_scored_results() -> tuple[dict[str, Any], dict[str, Any]]:
    reference_records = load_json(REFERENCE_PATH)
    # The bootstrap index order is part of seeded reproducibility. Use the
    # lexicographic order established by the original paired-analysis script.
    sent_ids = sorted(record["sent_id"] for record in reference_records)
    gold_by_id = {record["sent_id"]: record["gold"] for record in reference_records}
    text_by_id = {record["sent_id"]: record["text"] for record in reference_records}
    n_tokens = np.asarray(
        [
            sum(token["upos"] != "PUNCT" for token in gold_by_id[sent_id])
            for sent_id in sent_ids
        ],
        dtype=np.int32,
    )

    relation_support: Counter[str] = Counter()
    for sent_id in sent_ids:
        for token in gold_by_id[sent_id]:
            if token["upos"] != "PUNCT":
                relation_support[token["deprel"]] += 1

    result: dict[str, Any] = {
        "protocol": {
            "reference_path": str(REFERENCE_PATH.relative_to(ROOT)),
            "sentences": len(sent_ids),
            "non_punctuation_tokens": int(n_tokens.sum()),
            "punctuation_policy": "exclude tokens whose gold UPOS is PUNCT",
            "missing_policy": "zero correct; gold sentence and token support retained",
            "alignment": "predictions aligned by token id; last duplicate id wins",
            "bootstrap_sentence_order": "sent_id, lexicographically sorted",
            "relation_subtypes": "retained",
        },
        "corpus": {},
        "reliability": {},
        "relations": {},
        "length": {},
    }
    internal: dict[str, Any] = {
        "sent_ids": sent_ids,
        "gold_by_id": gold_by_id,
        "text_by_id": text_by_id,
        "n_tokens": n_tokens,
        "las_arrays": {},
        "uas_arrays": {},
        "record_maps": {},
        "token_status": {},
    }

    base_bin_counts = {
        label: {"sentences": 0, "tokens": 0}
        for label, _, _ in LENGTH_BINS
    }
    zero_token_sentences = 0
    for sent_id, length in zip(sent_ids, n_tokens):
        if length == 0:
            zero_token_sentences += 1
            continue
        label = next(
            label for label, lower, upper in LENGTH_BINS if lower <= length <= upper
        )
        base_bin_counts[label]["sentences"] += 1
        base_bin_counts[label]["tokens"] += int(length)

    for model_slug in MODELS:
        result["corpus"][model_slug] = {}
        result["reliability"][model_slug] = {}
        result["relations"][model_slug] = {}
        result["length"][model_slug] = {}
        for condition, _ in CONDITIONS:
            path = ROOT / f"outputs/{model_slug}/{condition}/model_evaluation.json"
            records = load_json(path)
            record_map = {record["sent_id"]: record for record in records}
            internal["record_maps"][(model_slug, condition)] = record_map

            las_counts = np.zeros(len(sent_ids), dtype=np.int32)
            uas_counts = np.zeros(len(sent_ids), dtype=np.int32)
            status_by_id: dict[str, dict[int, tuple[bool, bool]]] = {}
            relation_correct: Counter[str] = Counter()
            length_buckets = {
                label: {
                    "sentences": values["sentences"],
                    "total_tokens": values["tokens"],
                    "uas_correct": 0,
                    "las_correct": 0,
                }
                for label, values in base_bin_counts.items()
            }
            reliability_counts = {
                "archive_records": len(records),
                "schema_valid": 0,
                "absent": 0,
                "validation_failures": 0,
                "exact_token_inventory": 0,
                "root_conforming": 0,
                "root_warnings": 0,
                "legal_heads": 0,
                "self_loop_free": 0,
                "acyclic_and_connected": 0,
                "fully_tree_conforming": 0,
            }

            for index, sent_id in enumerate(sent_ids):
                record = record_map.get(sent_id)
                if record is None:
                    reliability_counts["absent"] += 1
                    prediction = None
                else:
                    prediction = record.get("llm_parse")
                    if prediction is None:
                        reliability_counts["validation_failures"] += 1

                total, uas_correct, las_correct, token_status = score_tokens(
                    gold_by_id[sent_id], prediction
                )
                if total != int(n_tokens[index]):
                    raise AssertionError(f"Gold denominator changed for {sent_id}")
                uas_counts[index] = uas_correct
                las_counts[index] = las_correct
                status_by_id[sent_id] = token_status

                if n_tokens[index]:
                    label = next(
                        label
                        for label, lower, upper in LENGTH_BINS
                        if lower <= n_tokens[index] <= upper
                    )
                    length_buckets[label]["uas_correct"] += uas_correct
                    length_buckets[label]["las_correct"] += las_correct

                pred_by_id = {
                    token["id"]: token
                    for token in prediction or []
                    if isinstance(token, dict) and isinstance(token.get("id"), int)
                }
                for gold_token in gold_by_id[sent_id]:
                    if gold_token["upos"] == "PUNCT":
                        continue
                    predicted = pred_by_id.get(gold_token["id"])
                    if (
                        predicted is not None
                        and predicted.get("head") == gold_token["head"]
                        and str(predicted.get("deprel", "")).lower()
                        == gold_token["deprel"].lower()
                    ):
                        relation_correct[gold_token["deprel"]] += 1

                audit = structural_audit(gold_by_id[sent_id], prediction)
                for key, passed in audit.items():
                    reliability_counts[key] += int(passed)
                if audit["schema_valid"] and not audit["root_conforming"]:
                    reliability_counts["root_warnings"] += 1

            internal["las_arrays"][(model_slug, condition)] = las_counts
            internal["uas_arrays"][(model_slug, condition)] = uas_counts
            internal["token_status"][(model_slug, condition)] = status_by_id

            total_tokens = int(n_tokens.sum())
            result["corpus"][model_slug][condition] = {
                "sentences": len(sent_ids),
                "total_tokens": total_tokens,
                "uas_correct": int(uas_counts.sum()),
                "las_correct": int(las_counts.sum()),
                "uas": float(uas_counts.sum() / total_tokens),
                "las": float(las_counts.sum() / total_tokens),
            }
            result["reliability"][model_slug][condition] = reliability_counts
            result["relations"][model_slug][condition] = {
                relation: {
                    "support": support,
                    "las_correct": relation_correct[relation],
                    "las": relation_correct[relation] / support,
                }
                for relation, support in sorted(
                    relation_support.items(), key=lambda item: (-item[1], item[0])
                )
            }
            for bucket in length_buckets.values():
                bucket["uas"] = bucket["uas_correct"] / bucket["total_tokens"]
                bucket["las"] = bucket["las_correct"] / bucket["total_tokens"]
            result["length"][model_slug][condition] = length_buckets

    result["length_protocol"] = {
        "definition": "number of gold non-PUNCT tokens",
        "zero_scored_token_sentences_excluded": zero_token_sentences,
        "status": "post-hoc descriptive analysis; no interaction test",
    }
    result["relation_support"] = dict(
        sorted(relation_support.items(), key=lambda item: (-item[1], item[0]))
    )
    return result, internal


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def bootstrap_pairwise(
    internal: dict[str, Any],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    systems = [
        (model_slug, condition)
        for model_slug in MODELS
        for condition, _ in CONDITIONS
    ]
    correct = np.stack(
        [internal["las_arrays"][system] for system in systems], axis=0
    ).astype(np.int32)
    n_tokens = internal["n_tokens"]
    n_sentences = len(n_tokens)
    bootstrap_scores = np.empty((n_resamples, len(systems)), dtype=np.float64)

    rng = random.Random(seed)
    batch_size = 100
    for start in range(0, n_resamples, batch_size):
        size = min(batch_size, n_resamples - start)
        indices = np.empty((size, n_sentences), dtype=np.int32)
        for row in range(size):
            indices[row] = [
                rng.randint(0, n_sentences - 1) for _ in range(n_sentences)
            ]
        denominators = n_tokens[indices].sum(axis=1, dtype=np.int64)
        selected = np.take(correct, indices, axis=1)
        numerators = selected.sum(axis=2, dtype=np.int64).T
        bootstrap_scores[start : start + size] = numerators / denominators[:, None]

    system_index = {system: index for index, system in enumerate(systems)}
    results: dict[str, Any] = {}
    primary_raw: dict[str, float] = {}

    for model_slug in MODELS:
        comparisons: dict[str, Any] = {}
        condition_keys = [condition for condition, _ in CONDITIONS]
        for first_index, first in enumerate(condition_keys):
            for second in condition_keys[first_index + 1 :]:
                key = f"{first}_vs_{second}"
                first_scores = bootstrap_scores[
                    :, system_index[(model_slug, first)]
                ]
                second_scores = bootstrap_scores[
                    :, system_index[(model_slug, second)]
                ]
                differences = first_scores - second_scores
                observed = (
                    internal["las_arrays"][(model_slug, first)].sum()
                    - internal["las_arrays"][(model_slug, second)].sum()
                ) / internal["n_tokens"].sum()
                lower, upper = np.percentile(differences, [2.5, 97.5])
                lower_tail = (1 + int(np.count_nonzero(differences <= 0))) / (
                    n_resamples + 1
                )
                upper_tail = (1 + int(np.count_nonzero(differences >= 0))) / (
                    n_resamples + 1
                )
                p_value = min(1.0, 2 * min(lower_tail, upper_tail))
                status = (
                    "primary"
                    if frozenset((first, second)) in PRIMARY_PAIRS
                    else "exploratory"
                )
                comparisons[key] = {
                    "first": first,
                    "second": second,
                    "difference": float(observed),
                    "ci_95": [float(lower), float(upper)],
                    "p_raw": float(p_value),
                    "status": status,
                    "n_sentences": n_sentences,
                    "n_resamples": n_resamples,
                }
                if status == "primary":
                    primary_raw[f"{model_slug}:{key}"] = p_value

        within_adjusted = holm_adjust(
            {key: value["p_raw"] for key, value in comparisons.items()}
        )
        for key, adjusted in within_adjusted.items():
            comparisons[key]["p_holm_15"] = adjusted
        results[model_slug] = comparisons

    primary_adjusted = holm_adjust(primary_raw)
    for compound_key, adjusted in primary_adjusted.items():
        model_slug, comparison_key = compound_key.split(":", 1)
        results[model_slug][comparison_key]["p_holm_primary_4"] = adjusted

    return {
        "method": {
            "statistic": "paired sentence bootstrap of token-weighted micro LAS",
            "seed": seed,
            "resamples": n_resamples,
            "sentences_per_resample": n_sentences,
            "ci": "2.5th and 97.5th percentiles",
            "p_value": (
                "2 * min((1 + count(delta<=0))/(B+1), "
                "(1 + count(delta>=0))/(B+1)); capped at 1"
            ),
            "primary_family": (
                "POS-Signature vs Semantic and POS-Signature vs Fixed for both "
                "models; Holm adjustment across four tests"
            ),
            "complete_tables": (
                "Holm adjustment across all 15 pairwise comparisons within "
                "each model; exploratory"
            ),
        },
        "models": results,
    }


def coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError("not integer-coercible")


def parse_raw_response(raw_response: str | None) -> list[dict[str, Any]] | None:
    """Extract the archived JSON token schema using the established scoring rules."""
    if not raw_response:
        return None
    stripped = raw_response.strip()
    brace_position = stripped.find("{")
    if brace_position < 0:
        return None
    try:
        parsed = json.loads(stripped[brace_position:])
        tokens = parsed["tokens"]
        if not isinstance(tokens, list):
            return None
        normalized = []
        for token in tokens:
            if not isinstance(token, dict):
                return None
            normalized.append(
                {
                    "id": coerce_int(token["id"]),
                    "form": token["form"] if isinstance(token["form"], str) else str(token["form"]),
                    "upos": token["upos"] if isinstance(token["upos"], str) else str(token["upos"]),
                    "head": coerce_int(token["head"]),
                    "deprel": (
                        token["deprel"]
                        if isinstance(token["deprel"], str)
                        else str(token["deprel"])
                    ),
                }
            )
        return normalized
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None










def critique_churn(internal: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    total_reference_tokens = int(internal["n_tokens"].sum())
    for model_slug in MODELS:
        cot_records = internal["record_maps"][(model_slug, "cot_few_shot")]
        critique_records = internal["record_maps"][(model_slug, "critique_refine")]
        analyzed_tokens = changed = fixed = broken = wrong_to_wrong = 0
        analyzed_sentences = 0
        for sent_id in internal["sent_ids"]:
            cot = cot_records.get(sent_id)
            critique = critique_records.get(sent_id)
            if (
                cot is None
                or critique is None
                or cot.get("llm_parse") is None
                or critique.get("llm_parse") is None
            ):
                continue
            analyzed_sentences += 1
            gold = internal["gold_by_id"][sent_id]
            cot_pred = {token["id"]: token for token in cot["llm_parse"]}
            critique_pred = {token["id"]: token for token in critique["llm_parse"]}
            for token in gold:
                if token["upos"] == "PUNCT":
                    continue
                analyzed_tokens += 1
                before = cot_pred.get(token["id"])
                after = critique_pred.get(token["id"])
                before_correct = (
                    before is not None
                    and before.get("head") == token["head"]
                    and str(before.get("deprel", "")).lower()
                    == token["deprel"].lower()
                )
                after_correct = (
                    after is not None
                    and after.get("head") == token["head"]
                    and str(after.get("deprel", "")).lower()
                    == token["deprel"].lower()
                )
                prediction_changed = (
                    (before or {}).get("head") != (after or {}).get("head")
                    or str((before or {}).get("deprel", "")).lower()
                    != str((after or {}).get("deprel", "")).lower()
                )
                if not prediction_changed:
                    continue
                changed += 1
                if not before_correct and after_correct:
                    fixed += 1
                elif before_correct and not after_correct:
                    broken += 1
                else:
                    wrong_to_wrong += 1
        net = fixed - broken
        result[model_slug] = {
            "analyzed_sentences": analyzed_sentences,
            "analyzed_tokens": analyzed_tokens,
            "changed": changed,
            "wrong_to_correct": fixed,
            "correct_to_wrong": broken,
            "wrong_to_different_wrong": wrong_to_wrong,
            "net_correct": net,
            "conditional_change": net / analyzed_tokens,
            "full_corpus_change": net / total_reference_tokens,
        }
    return result


def retrieved_texts(record: dict[str, Any] | None) -> list[str]:
    if record is None:
        return []
    system = (record.get("prompt") or {}).get("system", "")
    return re.findall(r"Input:\n# text = ([^\n]+)", system)


def token_difference_rows(
    gold: list[dict[str, Any]],
    first_prediction: list[dict[str, Any]] | None,
    second_prediction: list[dict[str, Any]] | None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    first = {token["id"]: token for token in first_prediction or []}
    second = {token["id"]: token for token in second_prediction or []}
    rows = []
    for token in gold:
        if token["upos"] == "PUNCT":
            continue
        first_token = first.get(token["id"])
        second_token = second.get(token["id"])
        first_pair = (
            (first_token or {}).get("head"),
            (first_token or {}).get("deprel"),
        )
        second_pair = (
            (second_token or {}).get("head"),
            (second_token or {}).get("deprel"),
        )
        gold_pair = (token["head"], token["deprel"])
        if first_pair == second_pair == gold_pair:
            continue
        rows.append(
            {
                "id": token["id"],
                "form": token["form"],
                "gold": {"head": token["head"], "deprel": token["deprel"]},
                "first": {"head": first_pair[0], "deprel": first_pair[1]},
                "second": {"head": second_pair[0], "deprel": second_pair[1]},
            }
        )
    return rows[:limit]


def qualitative_examples(internal: dict[str, Any]) -> dict[str, Any]:
    model_order = {model: index for index, model in enumerate(MODELS)}
    sent_ids = internal["sent_ids"]
    n_by_id = dict(zip(sent_ids, map(int, internal["n_tokens"])))

    semantic_candidates = []
    syntax_failure_candidates = []
    critique_candidates = []
    for model_slug in MODELS:
        syntax_records = internal["record_maps"][(model_slug, "few_shot_retrieval")]
        semantic_records = internal["record_maps"][(model_slug, "semantic_retrieval")]
        cot_records = internal["record_maps"][(model_slug, "cot_few_shot")]
        critique_records = internal["record_maps"][(model_slug, "critique_refine")]
        syntax_counts = internal["las_arrays"][(model_slug, "few_shot_retrieval")]
        semantic_counts = internal["las_arrays"][(model_slug, "semantic_retrieval")]

        for index, sent_id in enumerate(sent_ids):
            syntax_record = syntax_records.get(sent_id)
            semantic_record = semantic_records.get(sent_id)
            if (
                syntax_record
                and semantic_record
                and syntax_record.get("llm_parse") is not None
                and semantic_record.get("llm_parse") is not None
            ):
                margin = int(semantic_counts[index] - syntax_counts[index])
                if margin > 0:
                    semantic_candidates.append(
                        (-margin, -n_by_id[sent_id], model_order[model_slug], sent_id)
                    )
                if syntax_counts[index] == 0:
                    syntax_failure_candidates.append(
                        (-n_by_id[sent_id], model_order[model_slug], sent_id)
                    )

            cot_record = cot_records.get(sent_id)
            critique_record = critique_records.get(sent_id)
            if (
                cot_record
                and critique_record
                and cot_record.get("llm_parse") is not None
                and critique_record.get("llm_parse") is not None
            ):
                gold = internal["gold_by_id"][sent_id]
                _, _, _, before = score_tokens(gold, cot_record["llm_parse"])
                _, _, _, after = score_tokens(gold, critique_record["llm_parse"])
                broken = sum(
                    before[token_id][1] and not after[token_id][1]
                    for token_id in before
                )
                if broken:
                    critique_candidates.append(
                        (-broken, -n_by_id[sent_id], model_order[model_slug], sent_id)
                    )

    if not semantic_candidates or not syntax_failure_candidates or not critique_candidates:
        raise AssertionError("Could not select all predefined qualitative examples")

    semantic_choice = sorted(semantic_candidates)[0]
    semantic_model = list(MODELS)[semantic_choice[2]]
    semantic_sid = semantic_choice[3]
    syntax_choice = sorted(syntax_failure_candidates)[0]
    syntax_model = list(MODELS)[syntax_choice[1]]
    syntax_sid = syntax_choice[2]
    critique_choice = sorted(critique_candidates)[0]
    critique_model = list(MODELS)[critique_choice[2]]
    critique_sid = critique_choice[3]

    def retrieval_example(model_slug: str, sent_id: str) -> dict[str, Any]:
        syntax = internal["record_maps"][(model_slug, "few_shot_retrieval")][sent_id]
        semantic = internal["record_maps"][(model_slug, "semantic_retrieval")][sent_id]
        index = sent_ids.index(sent_id)
        return {
            "model": model_slug,
            "sent_id": sent_id,
            "text": internal["text_by_id"][sent_id],
            "scored_tokens": n_by_id[sent_id],
            "gold": internal["gold_by_id"][sent_id],
            "syntax_las_correct": int(
                internal["las_arrays"][(model_slug, "few_shot_retrieval")][index]
            ),
            "semantic_las_correct": int(
                internal["las_arrays"][(model_slug, "semantic_retrieval")][index]
            ),
            "syntax_retrieved_sentences": retrieved_texts(syntax),
            "semantic_retrieved_sentences": retrieved_texts(semantic),
            "syntax_prompt": syntax.get("prompt"),
            "semantic_prompt": semantic.get("prompt"),
            "syntax_parse": syntax.get("llm_parse"),
            "semantic_parse": semantic.get("llm_parse"),
            "token_differences": token_difference_rows(
                internal["gold_by_id"][sent_id],
                syntax["llm_parse"],
                semantic["llm_parse"],
            ),
        }

    cot = internal["record_maps"][(critique_model, "cot_few_shot")][critique_sid]
    critique = internal["record_maps"][(critique_model, "critique_refine")][critique_sid]
    return {
        "selection_protocol": {
            "semantic_wins": (
                "largest positive difference in LAS-correct token count for "
                "Semantic minus POS-Signature among schema-valid pairs; ties by "
                "more scored tokens, model order, then sentence id"
            ),
            "syntax_failure": (
                "longest schema-valid query with zero LAS-correct tokens under "
                "POS-Signature; ties by model order then sentence id"
            ),
            "critique_regression": (
                "largest number of LAS-correct CoT tokens changed to incorrect "
                "by Critique-Refine; ties by more scored tokens, model order, "
                "then sentence id"
            ),
        },
        "semantic_wins": retrieval_example(semantic_model, semantic_sid),
        "syntax_failure": retrieval_example(syntax_model, syntax_sid),
        "critique_regression": {
            "model": critique_model,
            "sent_id": critique_sid,
            "text": internal["text_by_id"][critique_sid],
            "scored_tokens": n_by_id[critique_sid],
            "gold": internal["gold_by_id"][critique_sid],
            "cot_prompt": cot.get("prompt"),
            "critique_prompt": critique.get("prompt"),
            "cot_parse": cot.get("llm_parse"),
            "critique_parse": critique.get("llm_parse"),
            "token_differences": token_difference_rows(
                internal["gold_by_id"][critique_sid],
                cot["llm_parse"],
                critique["llm_parse"],
            ),
        },
    }


def relation_output(scored: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": {
            "reference_sentences": scored["protocol"]["sentences"],
            "reference_non_punctuation_tokens": scored["protocol"][
                "non_punctuation_tokens"
            ],
            "missing_or_unscorable_policy": "zero correct; gold support retained",
            "relation_subtypes": "retained",
        },
        "support": scored["relation_support"],
        "models": scored["relations"],
    }


def length_output(scored: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": {
            "reference_sentences": scored["protocol"]["sentences"],
            "reference_non_punctuation_tokens": scored["protocol"][
                "non_punctuation_tokens"
            ],
            "length_definition": scored["length_protocol"]["definition"],
            "zero_scored_token_sentences_excluded_from_bins": scored[
                "length_protocol"
            ]["zero_scored_token_sentences_excluded"],
            "missing_or_unscorable_policy": (
                "zero correct; sentence and gold tokens retained"
            ),
            "status": scored["length_protocol"]["status"],
        },
        "models": scored["length"],
    }


def fmt_percent(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}"


def fmt_p(value: float) -> str:
    return f"{value:.4f}"


def render_markdown(
    dataset: dict[str, Any],
    scored: dict[str, Any],
    pairwise: dict[str, Any],
    budget: dict[str, Any],
    churn: dict[str, Any],
    qualitative: dict[str, Any],
) -> str:
    lines = [
        "# Generated Paper Tables",
        "",
        "Generated by `src/reproduce_paper_results.py`. Values use the fixed "
        "2,077-sentence, 21,998-token denominator unless a caption states otherwise.",
        "",
        "## Dataset",
        "",
        "| Split | Sentences | Syntactic words | Gold PUNCT | Non-punctuation words |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "dev", "test"):
        row = dataset[split]
        lines.append(
            f"| {split.title()} | {row['sentences']:,} | "
            f"{row['syntactic_words']:,} | {row['gold_punct']:,} | "
            f"{row['non_punctuation_words']:,} |"
        )

    lines.extend(["", "## Final-record token usage", "", "Provider-reported usage in the included final records; not a lifetime collection-cost estimate.", "", "| Model | Records | Records with usage | Prompt tokens | Completion tokens | Total tokens |", "|---|---:|---:|---:|---:|---:|"])
    for model in (*MODELS, "total"):
        row = budget[model]
        lines.append(f"| {MODELS.get(model, 'Total')} | {row['records']:,} | {row['records_with_usage']:,} | {row['prompt_tokens']:,} | {row['completion_tokens']:,} | {row['total_tokens']:,} |")

    lines.extend(
        [
            "",
            "## Main corpus results",
            "",
            "| Condition | gpt-oss UAS | gpt-oss LAS | Qwen UAS | Qwen LAS |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, label in CONDITIONS:
        gpt = scored["corpus"]["gpt_oss_120b"][condition]
        qwen = scored["corpus"]["qwen25_72b"][condition]
        lines.append(
            f"| {label} | {fmt_percent(gpt['uas'], 1)} | "
            f"{fmt_percent(gpt['las'], 1)} | {fmt_percent(qwen['uas'], 1)} | "
            f"{fmt_percent(qwen['las'], 1)} |"
        )

    for model_slug, model_label in MODELS.items():
        lines.extend(
            [
                "",
                f"## Complete paired comparisons — {model_label}",
                "",
                "| Comparison (first − second) | Status | ΔLAS (pp) | 95% CI | "
                "Raw p | Holm p (15) |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for comparison in pairwise["models"][model_slug].values():
            first = CONDITION_LABEL[comparison["first"]]
            second = CONDITION_LABEL[comparison["second"]]
            lower, upper = comparison["ci_95"]
            lines.append(
                f"| {first} − {second} | {comparison['status'].title()} | "
                f"{100 * comparison['difference']:+.2f} | "
                f"[{100 * lower:+.2f}, {100 * upper:+.2f}] | "
                f"{fmt_p(comparison['p_raw'])} | "
                f"{fmt_p(comparison['p_holm_15'])} |"
            )

    lines.extend(["", "## Post-hoc sentence-length LAS", ""])
    for model_slug, model_label in MODELS.items():
        lines.extend(
            [
                f"### {model_label}",
                "",
                "| Bin | Sentences | Tokens | "
                + " | ".join(label for _, label in CONDITIONS)
                + " |",
                "|---|---:|---:|" + "---:|" * len(CONDITIONS),
            ]
        )
        for label, _, _ in LENGTH_BINS:
            base = scored["length"][model_slug]["zero_shot"][label]
            values = [
                fmt_percent(scored["length"][model_slug][condition][label]["las"], 1)
                for condition, _ in CONDITIONS
            ]
            lines.append(
                f"| {label} | {base['sentences']:,} | "
                f"{base['total_tokens']:,} | " + " | ".join(values) + " |"
            )
        lines.append("")

    lines.extend(["## Post-hoc structural audit", ""])
    for model_slug, model_label in MODELS.items():
        lines.extend(
            [
                f"### {model_label}",
                "",
                "| Condition | Schema-valid | Exact token inventory | "
                "Root-conforming | Legal heads | Acyclic + connected | Full audit |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition, label in CONDITIONS:
            row = scored["reliability"][model_slug][condition]
            lines.append(
                f"| {label} | {row['schema_valid']:,}/2,077 | "
                f"{row['exact_token_inventory']:,}/2,077 | "
                f"{row['root_conforming']:,}/2,077 | "
                f"{row['legal_heads']:,}/2,077 | "
                f"{row['acyclic_and_connected']:,}/2,077 | "
                f"{row['fully_tree_conforming']:,}/2,077 |"
            )
        lines.append("")

    lines.extend(
        [
            "## Critique churn",
            "",
            "| Outcome | gpt-oss | Qwen |",
            "|---|---:|---:|",
        ]
    )
    churn_rows = [
        ("Analyzed sentences", "analyzed_sentences", "{:,}"),
        ("Analyzed tokens", "analyzed_tokens", "{:,}"),
        ("Changed prediction", "changed", "{:,}"),
        ("Wrong → correct", "wrong_to_correct", "{:,}"),
        ("Correct → wrong", "correct_to_wrong", "{:,}"),
        ("Wrong → different wrong", "wrong_to_different_wrong", "{:,}"),
        ("Net LAS-correct tokens", "net_correct", "{:+,}"),
    ]
    for label, key, template in churn_rows:
        lines.append(
            f"| {label} | {template.format(churn['gpt_oss_120b'][key])} | "
            f"{template.format(churn['qwen25_72b'][key])} |"
        )

    lines.extend(
        [
            "",
            "## Deterministically selected qualitative cases",
            "",
            "Full prompts, parses, demonstration texts, and differing tokens are in "
            "`results/qualitative_examples.json`.",
            "",
        ]
    )
    for key in ("semantic_wins", "syntax_failure", "critique_regression"):
        example = qualitative[key]
        lines.append(
            f"- **{key.replace('_', ' ').title()}:** `{example['sent_id']}` "
            f"({MODELS[example['model']]}), “{example['text']}”"
        )
    return "\n".join(lines).rstrip() + "\n"









def final_budget():
    budget = {}
    for model in MODELS:
        rows = [row for condition, _ in CONDITIONS for row in load_json(ROOT / f"outputs/{model}/{condition}/model_responses.json.gz")]
        usage = [row["usage"] for row in rows if row.get("usage")]
        budget[model] = {"records": len(rows), "records_with_usage": len(usage), **{key: sum((u.get(key) or 0) for u in usage) for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}
    budget["total"] = {key: sum(row[key] for row in budget.values()) for key in budget[next(iter(MODELS))]}
    return budget


def main():
    dataset = {split: parse_conllu_stats(ROOT / f"data/UD_English-EWT/en_ewt-ud-{split}.conllu") for split in ("train", "dev", "test")}
    scored, internal = build_scored_results()
    pairwise = bootstrap_pairwise(internal, 10000, 42)
    budget = final_budget()
    churn = critique_churn(internal)
    qualitative = qualitative_examples(internal)
    result = {"schema_version": 2, "dataset": dataset, "scoring": scored, "pairwise": pairwise, "budget": budget, "critique_churn": churn, "qualitative_selection": qualitative["selection_protocol"]}
    write_json(OUT_DIR / "paper_results.json", result)
    write_json(OUT_DIR / "qualitative_examples.json", qualitative)
    relation, length = relation_output(scored), length_output(scored)
    write_json(RELATION_PATH, relation)
    write_json(LENGTH_PATH, length)
    (OUT_DIR / "generated_tables.md").write_text(render_markdown(dataset, scored, pairwise, budget, churn, qualitative), encoding="utf-8")
    from generate_final_relation_diagnostics import render_figure, render_length_figure
    render_figure(relation)
    render_length_figure(length)
    from export_tables import export_all
    export_all(result)
    print("Regenerated final-record results: 2,077 sentences, 21,998 scoring tokens, 10,000 paired resamples.")


if __name__ == "__main__":
    main()
