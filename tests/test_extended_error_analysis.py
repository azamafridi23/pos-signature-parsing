"""
test_extended_error_analysis.py
───────────────────────────────
Unit tests for each analysis function in extended_error_analysis.py.

Uses hand-crafted synthetic evaluation records with known expected outputs
so every assertion is deterministic and verifiable by hand.

Run:
    python tests/test_extended_error_analysis.py
    # or with pytest:
    pytest tests/test_extended_error_analysis.py -v
"""

import sys
import math

# ─────────────────────────────────────────────
# Synthetic test data builders
# ─────────────────────────────────────────────

def _make_gold_token(id, form, upos, head, deprel):
    """Minimal gold token dict matching the schema in model_evaluation.json."""
    return {
        "id": id, "form": form, "lemma": form.lower(),
        "upos": upos, "xpos": "_", "feats": "_",
        "head": head, "deprel": deprel,
    }


def _make_pred_token(id, form, upos, head, deprel):
    """Minimal LLM-predicted token dict."""
    return {"id": id, "form": form, "upos": upos, "head": head, "deprel": deprel}


def _make_record(sent_id, text, gold, llm_parse, uas=None, las=None,
                 token_errors=None, validation_error=None, api_error=None):
    """Build one evaluation record for testing."""
    return {
        "sent_id": sent_id,
        "text": text,
        "timestamp": "2025-01-01T00:00:00Z",
        "model_config": {},
        "prompt": None,
        "usage": None,
        "raw_response": None,
        "gold": gold,
        "llm_parse": llm_parse,
        "uas": uas,
        "las": las,
        "exclude_punct": True,
        "token_errors": token_errors or [],
        "structural_warnings": [],
        "cot_reasoning": None,
        "validation_error": validation_error,
        "api_error": api_error,
    }


# ─────────────────────────────────────────────
# A reusable 4-token sentence for many tests:
#
#   "The cat sat here"
#
#   Token 1: The   DET    head=2  deprel=det
#   Token 2: cat   NOUN   head=3  deprel=nsubj
#   Token 3: sat   VERB   head=0  deprel=root
#   Token 4: here  ADV    head=3  deprel=advmod
#   Token 5: .     PUNCT  head=3  deprel=punct
#
# Arc directions:
#   Token 1 (head=2): left-arc  (head > id) → right direction  — wait,
#     direction = "left" if head < id, "right" if head > id
#     Token 1: head=2, id=1 → head > id → "right"  (head is to the right)
#     Token 2: head=3, id=2 → head > id → "right"
#     Token 4: head=3, id=4 → head < id → "left"  (head is to the left)
#     Token 3: root (head=0), excluded from direction
#
# Arc lengths (non-root, non-punct):
#   Token 1: |2-1| = 1
#   Token 2: |3-2| = 1
#   Token 4: |3-4| = 1
# ─────────────────────────────────────────────

GOLD_SENT_1 = [
    _make_gold_token(1, "The",  "DET",   2, "det"),
    _make_gold_token(2, "cat",  "NOUN",  3, "nsubj"),
    _make_gold_token(3, "sat",  "VERB",  0, "root"),
    _make_gold_token(4, "here", "ADV",   3, "advmod"),
    _make_gold_token(5, ".",    "PUNCT", 3, "punct"),
]

# Perfect prediction (all correct)
PRED_PERFECT = [
    _make_pred_token(1, "The",  "DET",   2, "det"),
    _make_pred_token(2, "cat",  "NOUN",  3, "nsubj"),
    _make_pred_token(3, "sat",  "VERB",  0, "root"),
    _make_pred_token(4, "here", "ADV",   3, "advmod"),
    _make_pred_token(5, ".",    "PUNCT", 3, "punct"),
]

# Prediction with errors:
#   Token 1: correct head (2), wrong deprel → "amod" instead of "det" (deprel_only error)
#   Token 2: wrong head (0 instead of 3), correct deprel "nsubj" (head_only error)
#   Token 3: correct
#   Token 4: wrong head (2 instead of 3), wrong deprel "obl" instead of "advmod"
PRED_ERRORS = [
    _make_pred_token(1, "The",  "DET",   2, "amod"),    # head OK, deprel wrong
    _make_pred_token(2, "cat",  "NOUN",  0, "nsubj"),   # head wrong, deprel OK
    _make_pred_token(3, "sat",  "VERB",  0, "root"),    # perfect
    _make_pred_token(4, "here", "ADV",   2, "obl"),     # head wrong, deprel wrong
    _make_pred_token(5, ".",    "PUNCT", 3, "punct"),    # punct (excluded)
]


# ─────────────────────────────────────────────
# Import the module under test
# ─────────────────────────────────────────────

import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from extended_error_analysis import (
    extract_genre,
    scored_only,
    per_sentence_scores,
    deprel_prf1,
    arc_analysis,
    upos_accuracy,
    deprel_confusion,
    genre_analysis,
)


# ═══════════════════════════════════════════════
# Test: extract_genre
# ═══════════════════════════════════════════════

def test_extract_genre():
    """Genre is the prefix before the first hyphen in sent_id."""
    assert extract_genre("email-enronsent04_02-0013") == "email"
    assert extract_genre("reviews-1234") == "reviews"
    assert extract_genre("weblog-blogspot.com_abc-0001") == "weblog"
    assert extract_genre("answers-20111108104413AAuBMSp_ans-0007") == "answers"
    assert extract_genre("newsgroup-groups.google.com-0001") == "newsgroup"
    print("  ✓ extract_genre")


# ═══════════════════════════════════════════════
# Test: scored_only
# ═══════════════════════════════════════════════

def test_scored_only():
    """scored_only filters out records with llm_parse=None."""
    rec_ok = _make_record("s1", "A", GOLD_SENT_1, PRED_PERFECT, uas=1.0, las=1.0)
    rec_fail = _make_record("s2", "B", GOLD_SENT_1, None, validation_error="json_err")

    result = scored_only([rec_ok, rec_fail])
    assert len(result) == 1
    assert result[0]["sent_id"] == "s1"
    print("  ✓ scored_only")


# ═══════════════════════════════════════════════
# Test Module G: per_sentence_scores
# ═══════════════════════════════════════════════

def test_per_sentence_scores():
    """
    - Successful parse: uses the record's uas/las values
    - Failed parse (llm_parse=None): assigns 0.0
    """
    rec_ok = _make_record("s1", "A", GOLD_SENT_1, PRED_PERFECT, uas=0.85, las=0.75)
    rec_fail = _make_record("s2", "B", GOLD_SENT_1, None, validation_error="err")
    rec_ok2 = _make_record("s3", "C", GOLD_SENT_1, PRED_ERRORS, uas=0.60, las=0.40)

    scores = per_sentence_scores([rec_ok, rec_fail, rec_ok2])

    assert scores["uas"] == [0.85, 0.0, 0.60], f"UAS mismatch: {scores['uas']}"
    assert scores["las"] == [0.75, 0.0, 0.40], f"LAS mismatch: {scores['las']}"
    assert len(scores["uas"]) == 3
    assert len(scores["las"]) == 3
    print("  ✓ per_sentence_scores")


def test_per_sentence_scores_all_failures():
    """All parse failures → all zeros."""
    rec1 = _make_record("s1", "A", GOLD_SENT_1, None, validation_error="err")
    rec2 = _make_record("s2", "B", GOLD_SENT_1, None, api_error="timeout")

    scores = per_sentence_scores([rec1, rec2])

    assert scores["uas"] == [0.0, 0.0]
    assert scores["las"] == [0.0, 0.0]
    print("  ✓ per_sentence_scores (all failures)")


def test_per_sentence_scores_empty():
    """Empty input → empty lists."""
    scores = per_sentence_scores([])
    assert scores == {"uas": [], "las": []}
    print("  ✓ per_sentence_scores (empty)")


# ═══════════════════════════════════════════════
# Test Module H: deprel_prf1
# ═══════════════════════════════════════════════

def test_deprel_prf1_perfect():
    """
    Perfect prediction → all TP, no FP/FN.
    PUNCT excluded, so we have 4 non-punct tokens.
    Deprels: det(1), nsubj(1), root(1), advmod(1)
    All should have P=1.0, R=1.0, F1=1.0.
    """
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_PERFECT, uas=1.0, las=1.0)
    result = deprel_prf1([rec], min_count=1)

    for rel in ["det", "nsubj", "root", "advmod"]:
        assert rel in result, f"Missing relation: {rel}"
        assert result[rel]["tp"] == 1, f"{rel} TP should be 1, got {result[rel]['tp']}"
        assert result[rel]["fp"] == 0, f"{rel} FP should be 0, got {result[rel]['fp']}"
        assert result[rel]["fn"] == 0, f"{rel} FN should be 0, got {result[rel]['fn']}"
        assert result[rel]["precision"] == 1.0
        assert result[rel]["recall"] == 1.0
        assert result[rel]["f1"] == 1.0

    print("  ✓ deprel_prf1 (perfect prediction)")


def test_deprel_prf1_with_errors():
    """
    PRED_ERRORS has these non-punct errors:
      Token 1: gold=det, pred=amod, head correct → FN for 'det', FP for 'amod'
      Token 2: gold=nsubj, pred=nsubj, head WRONG → FN for 'nsubj', FP for 'nsubj'
      Token 3: gold=root, pred=root, head correct → TP for 'root'
      Token 4: gold=advmod, pred=obl, head WRONG → FN for 'advmod', FP for 'obl'

    Expected:
      det:    TP=0, FP=0, FN=1 → P=0,   R=0,   F1=0
      nsubj:  TP=0, FP=1, FN=1 → P=0,   R=0,   F1=0
      root:   TP=1, FP=0, FN=0 → P=1.0, R=1.0, F1=1.0
      advmod: TP=0, FP=0, FN=1 → P=0,   R=0,   F1=0
      amod:   TP=0, FP=1, FN=0 → P=0,   R=N/A  (gold_count=0, below min_count)
      obl:    TP=0, FP=1, FN=0 → P=0,   R=N/A  (gold_count=0, below min_count)
    """
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_ERRORS, uas=0.5, las=0.25)
    result = deprel_prf1([rec], min_count=1)

    # det: gold=1, TP=0, FN=1 (head correct but deprel wrong → FN)
    assert result["det"]["tp"] == 0
    assert result["det"]["fn"] == 1
    assert result["det"]["recall"] == 0.0

    # root: perfect
    assert result["root"]["tp"] == 1
    assert result["root"]["f1"] == 1.0

    # nsubj: head wrong → FN for nsubj, FP for nsubj (pred deprel is "nsubj" but head wrong)
    assert result["nsubj"]["tp"] == 0
    assert result["nsubj"]["fn"] == 1
    assert result["nsubj"]["fp"] == 1  # predicted nsubj but head wrong

    # advmod: head wrong, deprel wrong → FN for advmod
    assert result["advmod"]["tp"] == 0
    assert result["advmod"]["fn"] == 1

    print("  ✓ deprel_prf1 (with errors)")


def test_deprel_prf1_min_count_filter():
    """Relations below min_count should be excluded."""
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_PERFECT, uas=1.0, las=1.0)

    # Each relation appears only once; min_count=2 should return empty
    result = deprel_prf1([rec], min_count=2)
    assert len(result) == 0, f"Expected empty result, got {len(result)} relations"
    print("  ✓ deprel_prf1 (min_count filter)")


def test_deprel_prf1_missing_token():
    """Missing token in prediction → FN for the gold relation."""
    # Prediction missing token 4 entirely
    pred_missing = [
        _make_pred_token(1, "The",  "DET",   2, "det"),
        _make_pred_token(2, "cat",  "NOUN",  3, "nsubj"),
        _make_pred_token(3, "sat",  "VERB",  0, "root"),
        # Token 4 ("here") is missing
        _make_pred_token(5, ".",    "PUNCT", 3, "punct"),
    ]
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, pred_missing)
    result = deprel_prf1([rec], min_count=1)

    # advmod: gold has 1, pred is missing → FN=1, TP=0
    assert result["advmod"]["tp"] == 0
    assert result["advmod"]["fn"] == 1
    assert result["advmod"]["recall"] == 0.0

    # det, nsubj, root should still be perfect
    assert result["det"]["f1"] == 1.0
    assert result["nsubj"]["f1"] == 1.0
    assert result["root"]["f1"] == 1.0
    print("  ✓ deprel_prf1 (missing token)")


# ═══════════════════════════════════════════════
# Test Module I: arc_analysis
# ═══════════════════════════════════════════════

def test_arc_analysis_perfect():
    """
    Perfect prediction, all heads correct.
    Non-punct, non-root tokens: 1 (det), 2 (nsubj), 4 (advmod)

    Directions:
      Token 1: head=2, id=1 → head > id → "right"
      Token 2: head=3, id=2 → head > id → "right"
      Token 4: head=3, id=4 → head < id → "left"

    Arc lengths (all = 1):
      Token 1: |2-1| = 1
      Token 2: |3-2| = 1
      Token 4: |3-4| = 1
    """
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_PERFECT, uas=1.0, las=1.0)
    result = arc_analysis([rec])

    # Direction
    assert result["direction"]["right"]["total"] == 2  # tokens 1, 2
    assert result["direction"]["right"]["correct"] == 2
    assert result["direction"]["right"]["uas"] == 1.0

    assert result["direction"]["left"]["total"] == 1   # token 4
    assert result["direction"]["left"]["correct"] == 1
    assert result["direction"]["left"]["uas"] == 1.0

    # Arc length — all are distance 1
    assert result["arc_length"]["1"]["total"] == 3
    assert result["arc_length"]["1"]["correct"] == 3
    assert result["arc_length"]["1"]["uas"] == 1.0

    # Other bins should be 0
    assert result["arc_length"]["2"]["total"] == 0
    print("  ✓ arc_analysis (perfect)")


def test_arc_analysis_with_errors():
    """
    PRED_ERRORS:
      Token 1: head correct (2) → right, correct
      Token 2: head WRONG (0 vs 3) → right, incorrect
      Token 4: head WRONG (2 vs 3) → left, incorrect

    Direction:
      right: total=2, correct=1 → UAS=0.5
      left:  total=1, correct=0 → UAS=0.0

    Arc length (all gold arc_len=1):
      bin "1": total=3, correct=1 → UAS=0.3333
    """
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_ERRORS)
    result = arc_analysis([rec])

    assert result["direction"]["right"]["total"] == 2
    assert result["direction"]["right"]["correct"] == 1
    assert result["direction"]["right"]["uas"] == 0.5

    assert result["direction"]["left"]["total"] == 1
    assert result["direction"]["left"]["correct"] == 0
    assert result["direction"]["left"]["uas"] == 0.0

    assert result["arc_length"]["1"]["total"] == 3
    assert result["arc_length"]["1"]["correct"] == 1
    assert round(result["arc_length"]["1"]["uas"], 4) == 0.3333
    print("  ✓ arc_analysis (with errors)")


def test_arc_analysis_longer_arcs():
    """Test that arc length bins work for distances > 1."""
    # A sentence where gold arc lengths vary:
    #   Token 1 (id=1, head=5): arc len = 4, direction = right
    #   Token 2 (id=2, head=10): arc len = 8, direction = right
    gold = [
        _make_gold_token(1,  "A", "DET",  5,  "det"),
        _make_gold_token(2,  "B", "NOUN", 10, "nsubj"),
        _make_gold_token(10, "C", "VERB", 0,  "root"),  # root, excluded
    ]
    pred = [
        _make_pred_token(1,  "A", "DET",  5,  "det"),   # correct
        _make_pred_token(2,  "B", "NOUN", 10, "nsubj"), # correct
        _make_pred_token(10, "C", "VERB", 0,  "root"),
    ]
    rec = _make_record("s1", "A B ... C", gold, pred)
    result = arc_analysis([rec])

    # Token 1: arc_len=4 → bin "4"
    assert result["arc_length"]["4"]["total"] == 1
    assert result["arc_length"]["4"]["correct"] == 1

    # Token 2: arc_len=8 → bin "6-10"
    assert result["arc_length"]["6-10"]["total"] == 1
    assert result["arc_length"]["6-10"]["correct"] == 1

    print("  ✓ arc_analysis (longer arcs)")


def test_arc_excludes_punct_and_root():
    """PUNCT tokens (token 5) and root tokens (token 3) are excluded."""
    rec = _make_record("s1", "The cat sat here .", GOLD_SENT_1, PRED_PERFECT)
    result = arc_analysis([rec])

    # Total tokens in direction analysis = 3 (tokens 1,2,4 — no root, no punct)
    total_dir = result["direction"]["left"]["total"] + result["direction"]["right"]["total"]
    assert total_dir == 3, f"Expected 3 tokens in direction analysis, got {total_dir}"

    # Total in arc length analysis = 3 (same tokens)
    total_arc = sum(result["arc_length"][b]["total"] for b in result["arc_length"])
    assert total_arc == 3, f"Expected 3 tokens in arc length analysis, got {total_arc}"
    print("  ✓ arc_analysis (excludes PUNCT and root)")


# ═══════════════════════════════════════════════
# Test Module J: parse failure characterization
# (run_failures is tested via integration — we test the logic inline)
# ═══════════════════════════════════════════════

def test_failure_detection():
    """Records with llm_parse=None are identified as failures."""
    rec_ok = _make_record("email-s1", "Good", GOLD_SENT_1, PRED_PERFECT, uas=1.0, las=1.0)
    rec_fail_val = _make_record("answers-s2", "Bad", GOLD_SENT_1, None,
                                validation_error="json_decode_error")
    rec_fail_api = _make_record("weblog-s3", "Ugly", GOLD_SENT_1, None,
                                api_error="timeout")

    records = [rec_ok, rec_fail_val, rec_fail_api]
    failed = [r for r in records if r.get("llm_parse") is None]

    assert len(failed) == 2
    assert failed[0]["sent_id"] == "answers-s2"
    assert failed[1]["sent_id"] == "weblog-s3"

    # Check error type classification
    for r in failed:
        err_type = "api_error" if r.get("api_error") else "validation_error"
        if r["sent_id"] == "answers-s2":
            assert err_type == "validation_error"
        else:
            assert err_type == "api_error"

    # Check genre extraction for failures
    genres = [extract_genre(r["sent_id"]) for r in failed]
    assert genres == ["answers", "weblog"]
    print("  ✓ failure detection (Module J logic)")


def test_failure_length_computation():
    """Failed sentence length is computed as non-PUNCT gold token count."""
    rec_fail = _make_record("email-s1", "The cat sat here .", GOLD_SENT_1, None,
                            validation_error="err")
    gold = rec_fail["gold"]
    non_punct_len = sum(1 for t in gold if t.get("upos") != "PUNCT")
    assert non_punct_len == 4  # "The cat sat here" — 4 non-punct tokens
    print("  ✓ failure length computation")


# ═══════════════════════════════════════════════
# Test Module K: upos_accuracy
# ═══════════════════════════════════════════════

def test_upos_accuracy_perfect():
    """Perfect prediction → UAS=1.0, LAS=1.0 for all UPOS tags."""
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_PERFECT, uas=1.0, las=1.0)
    result = upos_accuracy([rec])

    # Non-punct UPOS tags: DET, NOUN, VERB, ADV
    assert "DET" in result
    assert result["DET"]["total"] == 1
    assert result["DET"]["uas"] == 1.0
    assert result["DET"]["las"] == 1.0

    assert result["NOUN"]["uas"] == 1.0
    assert result["VERB"]["uas"] == 1.0
    assert result["ADV"]["uas"] == 1.0

    # PUNCT should NOT be in the results (excluded)
    assert "PUNCT" not in result
    print("  ✓ upos_accuracy (perfect)")


def test_upos_accuracy_with_errors():
    """
    PRED_ERRORS:
      Token 1 (DET):  head correct, deprel wrong → UAS=1, LAS=0
      Token 2 (NOUN): head wrong, deprel correct → UAS=0, LAS=0
      Token 3 (VERB): perfect → UAS=1, LAS=1
      Token 4 (ADV):  head wrong, deprel wrong → UAS=0, LAS=0
    """
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_ERRORS)
    result = upos_accuracy([rec])

    assert result["DET"]["uas"] == 1.0    # head correct
    assert result["DET"]["las"] == 0.0    # deprel wrong

    assert result["NOUN"]["uas"] == 0.0   # head wrong
    assert result["NOUN"]["las"] == 0.0

    assert result["VERB"]["uas"] == 1.0
    assert result["VERB"]["las"] == 1.0

    assert result["ADV"]["uas"] == 0.0    # head wrong
    assert result["ADV"]["las"] == 0.0
    print("  ✓ upos_accuracy (with errors)")


def test_upos_accuracy_missing_token():
    """Missing predicted token → UAS=0 and LAS=0 for that UPOS."""
    pred_missing = [
        _make_pred_token(1, "The",  "DET",   2, "det"),
        # Token 2 (NOUN) missing
        _make_pred_token(3, "sat",  "VERB",  0, "root"),
        _make_pred_token(4, "here", "ADV",   3, "advmod"),
        _make_pred_token(5, ".",    "PUNCT", 3, "punct"),
    ]
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, pred_missing)
    result = upos_accuracy([rec])

    assert result["NOUN"]["total"] == 1
    assert result["NOUN"]["uas"] == 0.0
    assert result["NOUN"]["las"] == 0.0
    print("  ✓ upos_accuracy (missing token)")


# ═══════════════════════════════════════════════
# Test Module L: deprel_confusion
# ═══════════════════════════════════════════════

def test_deprel_confusion_perfect():
    """Perfect prediction → no confusions at all."""
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_PERFECT)
    result = deprel_confusion([rec], top_n=10)

    for rel, info in result.items():
        assert info["total_label_errors"] == 0, \
            f"{rel} should have 0 label errors, got {info['total_label_errors']}"
    print("  ✓ deprel_confusion (perfect)")


def test_deprel_confusion_with_errors():
    """
    PRED_ERRORS:
      Token 1: gold=det, pred=amod, head CORRECT → confusion: det→amod (1 error)
      Token 2: gold=nsubj, pred=nsubj, head WRONG → NOT a confusion (head wrong)
      Token 3: gold=root, pred=root, correct → no confusion
      Token 4: gold=advmod, pred=obl, head WRONG → NOT a confusion (head wrong)

    Only token 1 counts as a deprel confusion (correct head, wrong label).
    """
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_ERRORS)
    result = deprel_confusion([rec], top_n=10)

    # det should have 1 label error → confused with "amod"
    assert result["det"]["total_label_errors"] == 1
    assert len(result["det"]["top_confusions"]) == 1
    assert result["det"]["top_confusions"][0]["predicted"] == "amod"
    assert result["det"]["top_confusions"][0]["count"] == 1

    # nsubj: head wrong, so NOT counted as a confusion
    assert result["nsubj"]["total_label_errors"] == 0

    # advmod: head wrong, so NOT counted as a confusion
    assert result["advmod"]["total_label_errors"] == 0

    # root: perfect
    assert result["root"]["total_label_errors"] == 0
    print("  ✓ deprel_confusion (with errors)")


def test_deprel_confusion_top_n_filter():
    """top_n limits how many gold deprels appear in the matrix."""
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_PERFECT)

    # top_n=2 should only include the 2 most frequent gold deprels
    result = deprel_confusion([rec], top_n=2)
    assert len(result) == 2, f"Expected 2 relations, got {len(result)}"
    print("  ✓ deprel_confusion (top_n filter)")


# ═══════════════════════════════════════════════
# Test Module M: genre_analysis
# ═══════════════════════════════════════════════

def test_genre_analysis_single_genre():
    """Records from one genre produce results only for that genre."""
    rec = _make_record("email-test-0001", "The cat sat here",
                       GOLD_SENT_1, PRED_PERFECT, uas=1.0, las=1.0)
    result = genre_analysis([rec])

    assert result["email"]["sentences"] == 1
    assert result["email"]["uas"] == 1.0
    assert result["email"]["las"] == 1.0

    # Other genres should have 0 sentences
    for genre in ["reviews", "answers", "newsgroup", "weblog"]:
        assert result[genre]["sentences"] == 0
        assert result[genre]["uas"] is None
    print("  ✓ genre_analysis (single genre)")


def test_genre_analysis_multiple_genres():
    """Records from different genres are counted separately."""
    rec1 = _make_record("email-s1", "The cat sat here",
                        GOLD_SENT_1, PRED_PERFECT, uas=1.0, las=1.0)
    rec2 = _make_record("reviews-s2", "The cat sat here",
                        GOLD_SENT_1, PRED_ERRORS, uas=0.5, las=0.25)
    result = genre_analysis([rec1, rec2])

    assert result["email"]["sentences"] == 1
    assert result["reviews"]["sentences"] == 1
    assert result["email"]["uas"] == 1.0
    # reviews has errors, UAS should be < 1.0
    assert result["reviews"]["uas"] < 1.0
    print("  ✓ genre_analysis (multiple genres)")


def test_genre_analysis_excludes_punct():
    """PUNCT tokens are excluded from UAS/LAS computation."""
    rec = _make_record("email-s1", "The cat sat here .", GOLD_SENT_1, PRED_PERFECT)
    result = genre_analysis([rec])

    # 4 non-punct tokens, not 5
    assert result["email"]["total_tokens"] == 4
    print("  ✓ genre_analysis (excludes PUNCT)")


def test_genre_analysis_parse_failure_excluded():
    """Records with llm_parse=None are excluded from genre scoring."""
    rec_ok = _make_record("email-s1", "OK", GOLD_SENT_1, PRED_PERFECT)
    rec_fail = _make_record("email-s2", "FAIL", GOLD_SENT_1, None,
                            validation_error="err")
    result = genre_analysis([rec_ok, rec_fail])

    # Only 1 sentence scored (the successful one)
    assert result["email"]["sentences"] == 1
    assert result["email"]["uas"] == 1.0
    print("  ✓ genre_analysis (parse failure excluded)")


# ═══════════════════════════════════════════════
# Test: Cross-module consistency
# ═══════════════════════════════════════════════

def test_consistency_uas_las_ordering():
    """UAS should always be >= LAS (since LAS requires both head AND deprel correct)."""
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_ERRORS)

    upos_res = upos_accuracy([rec])
    for upos, stats in upos_res.items():
        assert stats["uas"] >= stats["las"], \
            f"UAS < LAS for {upos}: UAS={stats['uas']}, LAS={stats['las']}"

    arc_res = arc_analysis([rec])
    # Can't directly compare UAS to LAS in arc_analysis since it only computes UAS,
    # but we verify UAS is between 0 and 1
    for direction in ["left", "right"]:
        uas = arc_res["direction"][direction]["uas"]
        if uas is not None:
            assert 0.0 <= uas <= 1.0
    print("  ✓ cross-module consistency (UAS >= LAS)")


def test_consistency_total_tokens():
    """
    Total non-punct tokens should be the same across upos_accuracy and arc_analysis.
    arc_analysis excludes root tokens from direction/length, so totals differ there,
    but upos should count all non-punct.
    """
    rec = _make_record("s1", "The cat sat here", GOLD_SENT_1, PRED_PERFECT)

    upos_res = upos_accuracy([rec])
    total_upos = sum(s["total"] for s in upos_res.values())

    # 4 non-punct tokens
    assert total_upos == 4, f"UPOS total should be 4, got {total_upos}"

    arc_res = arc_analysis([rec])
    total_arc = sum(arc_res["arc_length"][b]["total"] for b in arc_res["arc_length"])
    # arc_analysis excludes root → 3 tokens
    assert total_arc == 3, f"Arc total should be 3 (no root), got {total_arc}"
    print("  ✓ cross-module consistency (token counts)")


# ═══════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Extended Error Analysis — Unit Tests")
    print("=" * 60)

    tests = [
        # Helpers
        ("extract_genre",                      test_extract_genre),
        ("scored_only",                        test_scored_only),
        # Module G
        ("Module G: per_sentence_scores",      test_per_sentence_scores),
        ("Module G: all failures",             test_per_sentence_scores_all_failures),
        ("Module G: empty input",              test_per_sentence_scores_empty),
        # Module H
        ("Module H: deprel_prf1 perfect",      test_deprel_prf1_perfect),
        ("Module H: deprel_prf1 with errors",  test_deprel_prf1_with_errors),
        ("Module H: deprel_prf1 min_count",    test_deprel_prf1_min_count_filter),
        ("Module H: deprel_prf1 missing token", test_deprel_prf1_missing_token),
        # Module I
        ("Module I: arc_analysis perfect",     test_arc_analysis_perfect),
        ("Module I: arc_analysis with errors", test_arc_analysis_with_errors),
        ("Module I: arc_analysis longer arcs", test_arc_analysis_longer_arcs),
        ("Module I: arc excludes punct/root",  test_arc_excludes_punct_and_root),
        # Module J
        ("Module J: failure detection",        test_failure_detection),
        ("Module J: failure length",           test_failure_length_computation),
        # Module K
        ("Module K: upos perfect",             test_upos_accuracy_perfect),
        ("Module K: upos with errors",         test_upos_accuracy_with_errors),
        ("Module K: upos missing token",       test_upos_accuracy_missing_token),
        # Module L
        ("Module L: confusion perfect",        test_deprel_confusion_perfect),
        ("Module L: confusion with errors",    test_deprel_confusion_with_errors),
        ("Module L: confusion top_n",          test_deprel_confusion_top_n_filter),
        # Module M
        ("Module M: genre single",             test_genre_analysis_single_genre),
        ("Module M: genre multiple",           test_genre_analysis_multiple_genres),
        ("Module M: genre excludes punct",     test_genre_analysis_excludes_punct),
        ("Module M: genre parse failure",      test_genre_analysis_parse_failure_excluded),
        # Cross-module
        ("Cross-module: UAS >= LAS",           test_consistency_uas_las_ordering),
        ("Cross-module: token counts",         test_consistency_total_tokens),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            failed += 1
            errors.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ✗ {name}: {type(e).__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 60}")

    if errors:
        print("\n  Failed tests:")
        for name, msg in errors:
            print(f"    ✗ {name}: {msg}")
        sys.exit(1)
    else:
        print("\n  All tests passed! ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
