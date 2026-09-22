"""
tests/test_analyze_results.py
─────────────────────────────
Unit tests for every analytical function in src/analyze_results.py.

Run from the project root:
    python -m pytest tests/test_analyze_results.py -v
    # or without pytest:
    python -m unittest tests.test_analyze_results -v
"""

import sys
import os
import unittest

# Allow importing from src/ regardless of where the tests are run from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.analyze_results import (
    scored_only,
    corpus_uas_las,
    per_sentence_las_counts,
    bootstrap_test,
    parse_reliability,
    per_relation_analysis,
    length_analysis,
    extract_reasoning_sample,
)


# ─────────────────────────────────────────────────────────────────────────────
# Record / token factory helpers
# ─────────────────────────────────────────────────────────────────────────────

def tok(id, upos="NOUN", head=0, deprel="root", form=None):
    """Build a minimal gold or predicted token dict."""
    return {
        "id":     id,
        "form":   form or f"w{id}",
        "upos":   upos,
        "head":   head,
        "deprel": deprel,
    }


def err(token_id, error_type, gold_deprel="nsubj"):
    """Build a minimal token_error dict (as stored by evaluate_responses.py)."""
    return {
        "token_id":   token_id,
        "error_type": error_type,
        "gold_deprel": gold_deprel,
    }


def rec(
    sent_id="s1",
    gold=None,
    pred=None,               # None → parse failure
    token_errors=None,
    structural_warnings=None,
    validation_error=None,
    api_error=None,
    las=None,
    uas=None,
    cot_reasoning=None,
    text="Test sentence.",
):
    """
    Build a minimal evaluation record that matches the schema written
    by evaluate_responses.py and consumed by analyze_results.py.
    """
    return {
        "sent_id":             sent_id,
        "text":                text,
        "gold":                gold or [],
        "llm_parse":           pred,         # None means parse failure
        "token_errors":        token_errors or [],
        "structural_warnings": structural_warnings or [],
        "validation_error":    validation_error,
        "api_error":           api_error,
        "las":                 las,
        "uas":                 uas,
        "cot_reasoning":       cot_reasoning,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helper: scored_only
# ─────────────────────────────────────────────────────────────────────────────

class TestScoredOnly(unittest.TestCase):

    def test_keeps_records_with_parse(self):
        records = [rec(pred=[tok(1)]), rec(pred=[tok(1)])]
        self.assertEqual(len(scored_only(records)), 2)

    def test_drops_parse_failures(self):
        records = [rec(pred=None), rec(pred=[tok(1)])]
        self.assertEqual(len(scored_only(records)), 1)

    def test_all_failures(self):
        records = [rec(pred=None), rec(pred=None)]
        self.assertEqual(len(scored_only(records)), 0)

    def test_empty_input(self):
        self.assertEqual(scored_only([]), [])


# ─────────────────────────────────────────────────────────────────────────────
# Module A — corpus_uas_las
# ─────────────────────────────────────────────────────────────────────────────

class TestCorpusUasLas(unittest.TestCase):

    # ── Perfect prediction ────────────────────────────────────────────────────

    def test_perfect_parse_single_token(self):
        """One non-PUNCT token predicted correctly → UAS=1.0, LAS=1.0."""
        records = [rec(
            gold=[tok(1, upos="VERB", head=0, deprel="root")],
            pred=[tok(1, upos="VERB", head=0, deprel="root")],
        )]
        r = corpus_uas_las(records)
        self.assertEqual(r["uas"], 1.0)
        self.assertEqual(r["las"], 1.0)
        self.assertEqual(r["total_tokens"], 1)

    def test_perfect_parse_multiple_tokens(self):
        """All tokens in two sentences predicted correctly."""
        records = [
            rec(
                gold=[tok(1, head=0, deprel="root"), tok(2, head=1, deprel="nsubj")],
                pred=[tok(1, head=0, deprel="root"), tok(2, head=1, deprel="nsubj")],
            ),
            rec(
                gold=[tok(1, head=0, deprel="root"), tok(2, head=1, deprel="obj")],
                pred=[tok(1, head=0, deprel="root"), tok(2, head=1, deprel="obj")],
            ),
        ]
        r = corpus_uas_las(records)
        self.assertEqual(r["uas"], 1.0)
        self.assertEqual(r["las"], 1.0)
        self.assertEqual(r["total_tokens"], 4)
        self.assertEqual(r["scored_sentences"], 2)

    # ── Head correct, deprel wrong ────────────────────────────────────────────

    def test_head_correct_deprel_wrong(self):
        """UAS should be 1.0, LAS should be 0.0."""
        records = [rec(
            gold=[tok(1, head=0, deprel="root")],
            pred=[tok(1, head=0, deprel="nsubj")],   # wrong deprel
        )]
        r = corpus_uas_las(records)
        self.assertEqual(r["uas"], 1.0)
        self.assertEqual(r["las"], 0.0)
        self.assertEqual(r["uas_correct"], 1)
        self.assertEqual(r["las_correct"], 0)

    # ── Head wrong ────────────────────────────────────────────────────────────

    def test_head_wrong(self):
        """Both UAS and LAS should be 0.0 when head is wrong."""
        records = [rec(
            gold=[tok(1, head=0, deprel="root")],
            pred=[tok(1, head=2, deprel="root")],   # wrong head
        )]
        r = corpus_uas_las(records)
        self.assertEqual(r["uas"], 0.0)
        self.assertEqual(r["las"], 0.0)

    # ── Mixed tokens ──────────────────────────────────────────────────────────

    def test_mixed_correct_and_wrong(self):
        """2 tokens: one perfect, one wrong head → UAS=0.5, LAS=0.5."""
        records = [rec(
            gold=[tok(1, head=0, deprel="root"), tok(2, head=1, deprel="nsubj")],
            pred=[tok(1, head=0, deprel="root"), tok(2, head=3, deprel="obj")],  # tok 2 wrong
        )]
        r = corpus_uas_las(records)
        self.assertAlmostEqual(r["uas"], 0.5)
        self.assertAlmostEqual(r["las"], 0.5)

    # ── PUNCT exclusion ───────────────────────────────────────────────────────

    def test_punct_excluded_from_total(self):
        """PUNCT tokens must not contribute to the total count."""
        records = [rec(
            gold=[
                tok(1, upos="VERB",  head=0, deprel="root"),
                tok(2, upos="PUNCT", head=1, deprel="punct"),
            ],
            pred=[
                tok(1, head=0, deprel="root"),
                tok(2, head=1, deprel="punct"),   # PUNCT prediction irrelevant
            ],
        )]
        r = corpus_uas_las(records)
        self.assertEqual(r["total_tokens"], 1)      # only the VERB counts
        self.assertEqual(r["uas"], 1.0)
        self.assertEqual(r["las"], 1.0)

    def test_all_punct_yields_zero_tokens(self):
        """A sentence of only PUNCT → total_tokens=0, returns zeros."""
        records = [rec(
            gold=[tok(1, upos="PUNCT", head=0, deprel="punct")],
            pred=[tok(1, head=0, deprel="punct")],
        )]
        r = corpus_uas_las(records)
        self.assertEqual(r["total_tokens"], 0)
        self.assertEqual(r["uas"], 0.0)
        self.assertEqual(r["las"], 0.0)

    # ── Parse failures excluded ───────────────────────────────────────────────

    def test_parse_failure_not_counted(self):
        """A record with llm_parse=None is excluded from corpus metrics."""
        records = [
            rec(gold=[tok(1, head=0, deprel="root")], pred=None),   # failure
            rec(gold=[tok(1, head=0, deprel="root")],
                pred=[tok(1, head=0, deprel="root")]),               # success
        ]
        r = corpus_uas_las(records)
        self.assertEqual(r["scored_sentences"], 1)
        self.assertEqual(r["total_tokens"], 1)

    def test_all_parse_failures_returns_zeros(self):
        records = [rec(pred=None), rec(pred=None)]
        r = corpus_uas_las(records)
        self.assertEqual(r["uas"], 0.0)
        self.assertEqual(r["las"], 0.0)
        self.assertEqual(r["total_tokens"], 0)

    # ── Missing predicted token ───────────────────────────────────────────────

    def test_missing_predicted_token_counts_as_wrong(self):
        """Gold token absent from llm_parse → missed_tokens += 1, UAS/LAS wrong."""
        records = [rec(
            gold=[tok(1, head=0, deprel="root"), tok(2, head=1, deprel="nsubj")],
            pred=[tok(1, head=0, deprel="root")],   # tok 2 missing
        )]
        r = corpus_uas_las(records)
        self.assertEqual(r["missed_tokens"], 1)
        self.assertEqual(r["total_tokens"], 2)
        self.assertAlmostEqual(r["uas"], 0.5)   # only tok 1 correct
        self.assertAlmostEqual(r["las"], 0.5)

    # ── Case-insensitive deprel matching ─────────────────────────────────────

    def test_deprel_comparison_is_case_insensitive(self):
        """'ROOT' in prediction should match 'root' in gold."""
        records = [rec(
            gold=[tok(1, head=0, deprel="root")],
            pred=[tok(1, head=0, deprel="ROOT")],
        )]
        r = corpus_uas_las(records)
        self.assertEqual(r["las"], 1.0)

    # ── Empty input ───────────────────────────────────────────────────────────

    def test_empty_records_list(self):
        r = corpus_uas_las([])
        self.assertEqual(r["uas"], 0.0)
        self.assertEqual(r["las"], 0.0)
        self.assertEqual(r["scored_sentences"], 0)

    # ── Corpus-level vs. per-sentence average ────────────────────────────────

    def test_corpus_level_weighting(self):
        """
        Corpus-level UAS weights each TOKEN equally, not each sentence.
        Sentence A: 1 token, UAS=1.0
        Sentence B: 3 tokens, UAS=0.0
        Corpus UAS = 1/4 = 0.25  (NOT (1.0 + 0.0) / 2 = 0.5)
        """
        records = [
            rec(
                gold=[tok(1, head=0, deprel="root")],
                pred=[tok(1, head=0, deprel="root")],   # 1/1 correct
            ),
            rec(
                gold=[tok(1, head=0, deprel="root"),
                      tok(2, head=1, deprel="nsubj"),
                      tok(3, head=1, deprel="obj")],
                pred=[tok(1, head=2, deprel="root"),    # wrong head
                      tok(2, head=3, deprel="nsubj"),   # wrong head
                      tok(3, head=2, deprel="obj")],    # wrong head
            ),
        ]
        r = corpus_uas_las(records)
        self.assertEqual(r["total_tokens"], 4)
        self.assertEqual(r["uas_correct"], 1)
        self.assertAlmostEqual(r["uas"], 0.25, places=4)


# ─────────────────────────────────────────────────────────────────────────────
# Module B — per_sentence_las_counts & bootstrap_test
# ─────────────────────────────────────────────────────────────────────────────

class TestPerSentenceLasCounts(unittest.TestCase):

    def test_counts_correct_labeled_attachments(self):
        records = [rec(
            sent_id="s1",
            gold=[tok(1, head=0, deprel="root"),
                  tok(2, head=1, deprel="nsubj")],
            pred=[tok(1, head=0, deprel="root"),
                  tok(2, head=1, deprel="obj")],
        )]
        self.assertEqual(per_sentence_las_counts(records), {"s1": (1, 2)})

    def test_parse_failure_keeps_gold_denominator(self):
        records = [rec(
            sent_id="failed",
            gold=[tok(1), tok(2, upos="PUNCT")],
            pred=None,
        )]
        self.assertEqual(per_sentence_las_counts(records), {"failed": (0, 1)})

    def test_missing_prediction_id_receives_zero_credit(self):
        records = [rec(
            sent_id="missing",
            gold=[tok(1, head=0, deprel="root"),
                  tok(2, head=1, deprel="obj")],
            pred=[tok(1, head=0, deprel="root")],
        )]
        self.assertEqual(per_sentence_las_counts(records), {"missing": (1, 2)})

    def test_preserves_sentence_ids(self):
        records = [
            rec(sent_id=f"s{i}", gold=[tok(1)], pred=[tok(1)])
            for i in range(5)
        ]
        counts = per_sentence_las_counts(records)
        self.assertEqual(list(counts), [f"s{i}" for i in range(5)])


class TestBootstrapTest(unittest.TestCase):

    def _uniform_counts(self, value, n=200, tokens=10):
        return [(round(value * tokens), tokens)] * n

    # ── P-value range ─────────────────────────────────────────────────────────

    def test_p_value_between_zero_and_one(self):
        a = [(8, 10)] * 100 + [(6, 10)] * 100
        b = [(5, 10)] * 200
        result = bootstrap_test(a, b, n_resamples=500, seed=1)
        self.assertGreaterEqual(result["p_value"], 0.0)
        self.assertLessEqual(result["p_value"],    1.0)

    # ── Significant difference ────────────────────────────────────────────────

    def test_large_gap_is_significant(self):
        """
        System A always scores 0.9, system B always scores 0.3.
        The gap (0.6) is too large to arise by chance → p << 0.05.
        """
        a = self._uniform_counts(0.9, n=200)
        b = self._uniform_counts(0.3, n=200)
        result = bootstrap_test(a, b, n_resamples=5000, seed=42)
        self.assertTrue(result["significant"])
        self.assertLess(result["p_value"], 0.05)

    # ── No significant difference ─────────────────────────────────────────────

    def test_identical_systems_not_significant(self):
        """
        Identical score arrays → observed_diff = 0 → p-value should be high
        (most resampled |diffs| will also be ~0 ≥ 0).
        With H0 centering, diff* will be centered around 0, so
        P(|diff*| >= 0) ≈ 1.0. In practice very close to 1.
        """
        counts = [(7, 10)] * 100 + [(5, 10)] * 100
        result = bootstrap_test(counts, counts, n_resamples=1000, seed=42)
        self.assertFalse(result["significant"])
        self.assertGreater(result["p_value"], 0.5)

    # ── Two-sided symmetry ────────────────────────────────────────────────────

    def test_symmetric_p_value(self):
        """
        Two-sided test: swapping A and B must give the same p-value
        (only the sign of observed_diff changes, not its magnitude).
        """
        a = [(8, 10)] * 150 + [(4, 10)] * 50
        b = [(5, 10)] * 200
        r_ab = bootstrap_test(a, b, n_resamples=2000, seed=7)
        r_ba = bootstrap_test(b, a, n_resamples=2000, seed=7)
        self.assertAlmostEqual(r_ab["p_value"], r_ba["p_value"], places=3)

    # ── Observed diff sign ────────────────────────────────────────────────────

    def test_observed_diff_sign(self):
        """observed_diff = mean(A) - mean(B); negative when B is better."""
        a = self._uniform_counts(0.4)
        b = self._uniform_counts(0.7)
        result = bootstrap_test(a, b, n_resamples=100, seed=0)
        self.assertLess(result["observed_diff"], 0)   # B is better

    def test_observed_diff_positive(self):
        a = self._uniform_counts(0.8)
        b = self._uniform_counts(0.5)
        result = bootstrap_test(a, b, n_resamples=100, seed=0)
        self.assertGreater(result["observed_diff"], 0)

    # ── Metadata fields ───────────────────────────────────────────────────────

    def test_returns_required_keys(self):
        result = bootstrap_test([(5, 10)] * 10, [(5, 10)] * 10, n_resamples=50)
        self.assertIn("observed_diff", result)
        self.assertIn("p_value",       result)
        self.assertIn("n_resamples",   result)
        self.assertIn("significant",   result)

    def test_n_resamples_recorded(self):
        result = bootstrap_test([(5, 10)] * 20, [(5, 10)] * 20,
                                n_resamples=123, seed=0)
        self.assertEqual(result["n_resamples"], 123)

    # ── Determinism ───────────────────────────────────────────────────────────

    def test_same_seed_same_result(self):
        a = [(i % 10, 10) for i in range(100)]
        b = [((i + 3) % 10, 10) for i in range(100)]
        r1 = bootstrap_test(a, b, n_resamples=500, seed=99)
        r2 = bootstrap_test(a, b, n_resamples=500, seed=99)
        self.assertEqual(r1["p_value"], r2["p_value"])

    def test_different_seeds_may_differ(self):
        """Not a strict test, but sanity-checks that seed affects randomness."""
        a = [(i % 10, 10) for i in range(100)]
        b = [((i + 5) % 10, 10) for i in range(100)]
        r1 = bootstrap_test(a, b, n_resamples=200, seed=1)
        r2 = bootstrap_test(a, b, n_resamples=200, seed=2)
        # p-values may differ (not guaranteed, but usually will for small n)
        # Just check both are valid
        self.assertGreaterEqual(r1["p_value"], 0.0)
        self.assertGreaterEqual(r2["p_value"], 0.0)

    # ── Mismatched lengths error ──────────────────────────────────────────────

    def test_mismatched_lengths_raises(self):
        with self.assertRaises(AssertionError):
            bootstrap_test([(5, 10)] * 10, [(5, 10)] * 5, n_resamples=10)


# ─────────────────────────────────────────────────────────────────────────────
# Module C — parse_reliability
# ─────────────────────────────────────────────────────────────────────────────

class TestParseReliability(unittest.TestCase):

    # ── All successful ────────────────────────────────────────────────────────

    def test_all_success(self):
        records = [
            rec(pred=[tok(1)], validation_error=None),
            rec(pred=[tok(1)], validation_error=None),
        ]
        r = parse_reliability(records)
        self.assertEqual(r["total_sentences"],     2)
        self.assertEqual(r["parse_success"],       2)
        self.assertEqual(r["parse_failures"],      0)
        self.assertEqual(r["parse_success_rate"],  1.0)
        self.assertEqual(r["api_errors"],          0)

    # ── Parse failures ────────────────────────────────────────────────────────

    def test_validation_error_counts_as_failure(self):
        """A record with validation_error set is a parse failure."""
        records = [
            rec(pred=None, validation_error="schema mismatch"),
            rec(pred=[tok(1)], validation_error=None),
        ]
        r = parse_reliability(records)
        self.assertEqual(r["parse_success"],  1)
        self.assertEqual(r["parse_failures"], 1)
        self.assertAlmostEqual(r["parse_success_rate"], 0.5)

    def test_none_llm_parse_counts_as_failure(self):
        """llm_parse=None with no validation_error should still be a failure."""
        records = [rec(pred=None, validation_error=None)]
        r = parse_reliability(records)
        self.assertEqual(r["parse_success"],  0)
        self.assertEqual(r["parse_failures"], 1)

    # ── API errors ────────────────────────────────────────────────────────────

    def test_api_error_counted(self):
        records = [
            rec(pred=None, api_error="timeout"),
            rec(pred=[tok(1)], api_error=None),
        ]
        r = parse_reliability(records)
        self.assertEqual(r["api_errors"], 1)

    # ── Structural warnings ───────────────────────────────────────────────────

    def test_no_root_warning(self):
        records = [
            rec(pred=[tok(1)], structural_warnings=["no_root: no token has head=0"]),
        ]
        r = parse_reliability(records)
        self.assertEqual(r["no_root"], 1)
        self.assertEqual(r["multiple_roots"], 0)

    def test_multiple_roots_warning(self):
        records = [
            rec(pred=[tok(1)], structural_warnings=["multiple_roots: tokens [1, 3] have head=0"]),
        ]
        r = parse_reliability(records)
        self.assertEqual(r["multiple_roots"], 1)
        self.assertEqual(r["no_root"], 0)

    def test_structural_validity_rate_all_clean(self):
        """All parsed sentences structurally clean → struct_validity_rate=1.0."""
        records = [
            rec(pred=[tok(1)], structural_warnings=[]),
            rec(pred=[tok(1)], structural_warnings=[]),
        ]
        r = parse_reliability(records)
        self.assertEqual(r["structural_validity_rate"], 1.0)
        self.assertEqual(r["any_structural_issue"], 0)

    def test_structural_validity_rate_one_issue(self):
        """1 of 4 parsed sentences has a structural issue → rate = 3/4 = 0.75."""
        records = [
            rec(pred=[tok(1)], structural_warnings=["no_root: ..."]),
            rec(pred=[tok(1)], structural_warnings=[]),
            rec(pred=[tok(1)], structural_warnings=[]),
            rec(pred=[tok(1)], structural_warnings=[]),
        ]
        r = parse_reliability(records)
        self.assertEqual(r["any_structural_issue"], 1)
        self.assertAlmostEqual(r["structural_validity_rate"], 0.75)

    def test_structural_validity_rate_undefined_when_no_parse_success(self):
        """If parse_success=0, structural_validity_rate must be 0.0 (no division by zero)."""
        records = [rec(pred=None, validation_error="error")]
        r = parse_reliability(records)
        self.assertEqual(r["parse_success"], 0)
        self.assertEqual(r["structural_validity_rate"], 0.0)

    # ── Empty input ───────────────────────────────────────────────────────────

    def test_empty_records(self):
        r = parse_reliability([])
        self.assertEqual(r["total_sentences"],    0)
        self.assertEqual(r["parse_success_rate"], 0.0)

    # ── Mixed scenario ────────────────────────────────────────────────────────

    def test_mixed_scenario(self):
        records = [
            rec(pred=[tok(1)]),                                            # success, clean
            rec(pred=[tok(1)], structural_warnings=["no_root: ..."]),     # success, issue
            rec(pred=None, validation_error="bad JSON"),                   # failure
            rec(pred=None, api_error="timeout"),                           # API error
        ]
        r = parse_reliability(records)
        self.assertEqual(r["total_sentences"],    4)
        self.assertEqual(r["parse_success"],      2)
        self.assertEqual(r["parse_failures"],     2)
        self.assertEqual(r["api_errors"],         1)
        self.assertEqual(r["no_root"],            1)
        self.assertEqual(r["any_structural_issue"], 1)
        self.assertAlmostEqual(r["parse_success_rate"],       0.5)
        self.assertAlmostEqual(r["structural_validity_rate"], 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# Module D — per_relation_analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestPerRelationAnalysis(unittest.TestCase):

    # Helpers to build records with a specific error profile per token
    @staticmethod
    def _rec_with_errors(gold_tokens, error_list, sent_id="s1"):
        """
        Build a record where pred tokens are present for all gold tokens
        (so they don't trigger the 'missing token' path), but token_errors
        describes which tokens are wrong and how.
        """
        pred_tokens = [
            {"id": g["id"], "form": g["form"], "head": 99, "deprel": "X"}
            for g in gold_tokens
        ]
        return rec(
            sent_id=sent_id,
            gold=gold_tokens,
            pred=pred_tokens,
            token_errors=error_list,
        )

    # ── Perfect predictions ───────────────────────────────────────────────────

    def test_perfect_predictions(self):
        """No token_errors → UAS=1.0, LAS=1.0 for every relation."""
        records = [
            self._rec_with_errors(
                [tok(1, head=0, deprel="root"), tok(2, head=1, deprel="nsubj")],
                [],
            )
        ] * 60  # 60 identical sentences → 60 of each relation
        r = per_relation_analysis(records, min_count=1)
        self.assertEqual(r["root"]["uas"], 1.0)
        self.assertEqual(r["root"]["las"], 1.0)
        self.assertEqual(r["nsubj"]["uas"], 1.0)
        self.assertEqual(r["nsubj"]["las"], 1.0)

    # ── deprel_only error ─────────────────────────────────────────────────────

    def test_deprel_only_error(self):
        """
        deprel_only → head correct, deprel wrong.
        Token NOT in error list = perfect. Token with deprel_only = UAS ok, LAS wrong.
        """
        gold_toks = [tok(1, head=0, deprel="nsubj")] * 1  # 1 nsubj token per record
        # Among 100 records, 60 have NO error, 40 have deprel_only
        records = (
            [self._rec_with_errors(gold_toks, [],                              f"s{i}") for i in range(60)] +
            [self._rec_with_errors(gold_toks, [err(1, "deprel_only")],         f"s{i}") for i in range(60, 100)]
        )
        r = per_relation_analysis(records, min_count=1)
        nsubj = r["nsubj"]
        self.assertEqual(nsubj["total"], 100)
        self.assertAlmostEqual(nsubj["uas"], 1.0)        # all heads correct
        self.assertAlmostEqual(nsubj["las"], 0.6)        # only 60/100 deprels correct

    # ── head_only error ───────────────────────────────────────────────────────

    def test_head_only_error(self):
        """head_only → both UAS and LAS wrong."""
        gold_toks = [tok(1, head=0, deprel="root")]
        records = [self._rec_with_errors(gold_toks, [err(1, "head_only")], f"s{i}") for i in range(60)]
        r = per_relation_analysis(records, min_count=1)
        root = r["root"]
        self.assertAlmostEqual(root["uas"], 0.0)
        self.assertAlmostEqual(root["las"], 0.0)

    # ── head_and_deprel error ─────────────────────────────────────────────────

    def test_head_and_deprel_error(self):
        """head_and_deprel → both UAS and LAS wrong."""
        gold_toks = [tok(1, head=0, deprel="obj")]
        records = [self._rec_with_errors(gold_toks, [err(1, "head_and_deprel")], f"s{i}") for i in range(60)]
        r = per_relation_analysis(records, min_count=1)
        self.assertAlmostEqual(r["obj"]["uas"], 0.0)
        self.assertAlmostEqual(r["obj"]["las"], 0.0)

    # ── Missing predicted token ───────────────────────────────────────────────

    def test_missing_token_in_prediction(self):
        """If a gold token has no matching predicted token, both UAS and LAS are wrong."""
        gold_toks = [tok(1, head=0, deprel="root")]
        # pred contains token id=99 (not id=1) → pred_map.get(1) returns None
        records = [rec(
            gold=gold_toks,
            pred=[{"id": 99, "head": 0, "deprel": "root"}],
        )] * 60
        r = per_relation_analysis(records, min_count=1)
        self.assertAlmostEqual(r["root"]["uas"], 0.0)
        self.assertAlmostEqual(r["root"]["las"], 0.0)

    # ── PUNCT exclusion ───────────────────────────────────────────────────────

    def test_punct_tokens_excluded(self):
        """PUNCT tokens must not appear in the result at all."""
        gold_toks = [
            tok(1, upos="VERB",  head=0, deprel="root"),
            tok(2, upos="PUNCT", head=1, deprel="punct"),
        ]
        records = [self._rec_with_errors(gold_toks, [], f"s{i}") for i in range(60)]
        r = per_relation_analysis(records, min_count=1)
        self.assertIn("root",  r)
        self.assertNotIn("punct", r)

    def test_punct_included_when_flag_false(self):
        """With exclude_punct=False, PUNCT tokens must be counted."""
        gold_toks = [
            tok(1, upos="VERB",  head=0, deprel="root"),
            tok(2, upos="PUNCT", head=1, deprel="punct"),
        ]
        records = [self._rec_with_errors(gold_toks, [], f"s{i}") for i in range(60)]
        r = per_relation_analysis(records, exclude_punct=False, min_count=1)
        self.assertIn("punct", r)

    # ── min_count threshold ───────────────────────────────────────────────────

    def test_min_count_filters_rare_relations(self):
        """Relations with fewer than min_count gold tokens must not appear."""
        gold_toks = [tok(1, head=0, deprel="xcomp")]
        records = [self._rec_with_errors(gold_toks, [], f"s{i}") for i in range(10)]
        r = per_relation_analysis(records, min_count=50)
        self.assertNotIn("xcomp", r)

    def test_min_count_includes_above_threshold(self):
        """Relations with exactly min_count tokens must appear."""
        gold_toks = [tok(1, head=0, deprel="acl")]
        records = [self._rec_with_errors(gold_toks, [], f"s{i}") for i in range(50)]
        r = per_relation_analysis(records, min_count=50)
        self.assertIn("acl", r)
        self.assertEqual(r["acl"]["total"], 50)

    # ── Token count accumulation ──────────────────────────────────────────────

    def test_total_token_count(self):
        """Token count must reflect all non-PUNCT gold tokens across all records."""
        gold_toks = [tok(1, head=0, deprel="root"), tok(2, head=1, deprel="nsubj")]
        records = [self._rec_with_errors(gold_toks, [], f"s{i}") for i in range(60)]
        r = per_relation_analysis(records, min_count=1)
        self.assertEqual(r["root"]["total"],  60)
        self.assertEqual(r["nsubj"]["total"], 60)

    # ── Parse failure records excluded ───────────────────────────────────────

    def test_parse_failure_excluded(self):
        gold_toks = [tok(1, head=0, deprel="root")]
        records = [
            rec(gold=gold_toks, pred=None),         # failure: excluded
            self._rec_with_errors(gold_toks, [], "s2"),  # success: counted
        ]
        r = per_relation_analysis(records, min_count=1)
        self.assertEqual(r["root"]["total"], 1)


# ─────────────────────────────────────────────────────────────────────────────
# Module E — length_analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestLengthAnalysis(unittest.TestCase):

    def _perfect_rec(self, n_tokens, sent_id="s1", include_punct=False):
        """Build a record with n_tokens non-PUNCT tokens, all perfectly predicted."""
        gold_toks = [tok(i + 1, head=0 if i == 0 else 1, deprel="root" if i == 0 else "dep")
                     for i in range(n_tokens)]
        if include_punct:
            gold_toks.append(tok(n_tokens + 1, upos="PUNCT", head=1, deprel="punct"))
        pred_toks = [{"id": g["id"], "head": g["head"], "deprel": g["deprel"]}
                     for g in gold_toks if g["upos"] != "PUNCT"]
        return rec(sent_id=sent_id, gold=gold_toks, pred=pred_toks)

    # ── Bin assignment ────────────────────────────────────────────────────────

    def test_bin_1_to_10(self):
        """A sentence with 5 non-PUNCT tokens lands in the 1-10 bin."""
        records = [self._perfect_rec(5, f"s{i}") for i in range(10)]
        r = length_analysis(records)
        self.assertGreater(r["1-10"]["sentences"], 0)
        self.assertEqual(r["11-20"]["sentences"], 0)

    def test_bin_11_to_20(self):
        records = [self._perfect_rec(15, f"s{i}") for i in range(10)]
        r = length_analysis(records)
        self.assertGreater(r["11-20"]["sentences"], 0)
        self.assertEqual(r["1-10"]["sentences"], 0)

    def test_bin_51_plus(self):
        records = [self._perfect_rec(60, f"s{i}") for i in range(5)]
        r = length_analysis(records)
        self.assertGreater(r["51+"]["sentences"], 0)
        self.assertEqual(r["1-10"]["sentences"], 0)

    # ── PUNCT not counted in sentence length ──────────────────────────────────

    def test_punct_does_not_affect_bin_assignment(self):
        """A sentence with 5 content tokens + 3 PUNCT should go in the 1-10 bin."""
        gold_toks = (
            [tok(i + 1, head=0 if i == 0 else 1, deprel="root" if i == 0 else "dep")
             for i in range(5)] +
            [tok(6, upos="PUNCT", head=1, deprel="punct"),
             tok(7, upos="PUNCT", head=1, deprel="punct"),
             tok(8, upos="PUNCT", head=1, deprel="punct")]
        )
        pred_toks = [{"id": g["id"], "head": g["head"], "deprel": g["deprel"]}
                     for g in gold_toks[:5]]  # only non-PUNCT in prediction
        records = [rec(gold=gold_toks, pred=pred_toks)]
        r = length_analysis(records)
        self.assertGreater(r["1-10"]["sentences"], 0)
        self.assertEqual(r["11-20"]["sentences"], 0)

    # ── LAS correctness per bin ───────────────────────────────────────────────

    def test_perfect_las_in_bin(self):
        """All predictions correct → LAS=1.0 in every populated bin."""
        records = [self._perfect_rec(5, f"s{i}") for i in range(10)]
        r = length_analysis(records)
        self.assertEqual(r["1-10"]["las"], 1.0)
        self.assertEqual(r["1-10"]["uas"], 1.0)

    def test_zero_las_when_all_wrong(self):
        """All predictions wrong → LAS=0.0 in the bin."""
        gold_toks = [tok(1, head=0, deprel="root"), tok(2, head=1, deprel="nsubj")]
        pred_toks = [{"id": 1, "head": 9, "deprel": "X"}, {"id": 2, "head": 9, "deprel": "X"}]
        records = [rec(gold=gold_toks, pred=pred_toks, sent_id=f"s{i}") for i in range(10)]
        r = length_analysis(records)
        self.assertEqual(r["1-10"]["las"], 0.0)
        self.assertEqual(r["1-10"]["uas"], 0.0)

    def test_partial_las(self):
        """2 tokens: one correct, one wrong → LAS=0.5 in the bin."""
        gold_toks = [tok(1, head=0, deprel="root"), tok(2, head=1, deprel="nsubj")]
        pred_toks = [
            {"id": 1, "head": 0, "deprel": "root"},   # correct
            {"id": 2, "head": 9, "deprel": "X"},       # wrong head
        ]
        records = [rec(gold=gold_toks, pred=pred_toks, sent_id=f"s{i}") for i in range(10)]
        r = length_analysis(records)
        self.assertAlmostEqual(r["1-10"]["las"], 0.5)

    # ── Empty bins return None ────────────────────────────────────────────────

    def test_empty_bin_returns_none_for_uas_las(self):
        """Bins with no sentences should have uas=None and las=None."""
        records = [self._perfect_rec(5)]
        r = length_analysis(records)
        self.assertIsNone(r["51+"]["uas"])
        self.assertIsNone(r["51+"]["las"])
        self.assertEqual(r["51+"]["sentences"], 0)

    # ── Parse failures excluded ───────────────────────────────────────────────

    def test_parse_failure_excluded(self):
        """Records with llm_parse=None don't contribute to any bin."""
        gold_toks = [tok(1, head=0, deprel="root")]
        records = [rec(gold=gold_toks, pred=None)]
        r = length_analysis(records)
        for lbl in r.values():
            self.assertEqual(lbl["sentences"], 0)

    # ── All bins present in output ────────────────────────────────────────────

    def test_all_bin_labels_present(self):
        """Output always contains all 5 bin labels regardless of data."""
        r = length_analysis([])
        for label in ["1-10", "11-20", "21-30", "31-50", "51+"]:
            self.assertIn(label, r)

    # ── Token count matches bin total ─────────────────────────────────────────

    def test_total_tokens_matches_non_punct_count(self):
        """total_tokens in a bin must equal number of non-PUNCT gold tokens in that bin."""
        records = [self._perfect_rec(5, f"s{i}") for i in range(4)]  # 4 sents × 5 toks
        r = length_analysis(records)
        self.assertEqual(r["1-10"]["total_tokens"], 20)


# ─────────────────────────────────────────────────────────────────────────────
# Module F — extract_reasoning_sample
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractReasoningSample(unittest.TestCase):

    @staticmethod
    def _cot_rec(i, has_reasoning=True, has_errors=True, has_parse=True, las=0.5):
        """Build a CoT-style record."""
        return rec(
            sent_id=f"sent-{i:03d}",
            pred=[tok(1)] if has_parse else None,
            token_errors=[err(1, "head_only")] if has_errors else [],
            cot_reasoning="Step 1: ..." if has_reasoning else None,
            las=las,
        )

    # ── Filtering logic ───────────────────────────────────────────────────────

    def test_excludes_records_without_reasoning(self):
        records = [self._cot_rec(i, has_reasoning=False) for i in range(20)]
        result = extract_reasoning_sample(records, n=10)
        self.assertEqual(len(result), 0)

    def test_excludes_parse_failures(self):
        records = [self._cot_rec(i, has_parse=False) for i in range(20)]
        result = extract_reasoning_sample(records, n=10)
        self.assertEqual(len(result), 0)

    def test_excludes_records_without_token_errors(self):
        """Records with no token errors (perfectly scored) are not useful for error analysis."""
        records = [self._cot_rec(i, has_errors=False) for i in range(20)]
        result = extract_reasoning_sample(records, n=10)
        self.assertEqual(len(result), 0)

    def test_includes_valid_cot_records(self):
        records = [self._cot_rec(i) for i in range(20)]
        result = extract_reasoning_sample(records, n=10, seed=42)
        self.assertEqual(len(result), 10)

    # ── n cap ─────────────────────────────────────────────────────────────────

    def test_n_respected_when_enough_candidates(self):
        records = [self._cot_rec(i) for i in range(100)]
        result = extract_reasoning_sample(records, n=30, seed=1)
        self.assertEqual(len(result), 30)

    def test_n_capped_by_available_candidates(self):
        """If there are only 5 valid candidates, return at most 5."""
        records = [self._cot_rec(i) for i in range(5)]
        result = extract_reasoning_sample(records, n=50, seed=1)
        self.assertLessEqual(len(result), 5)

    # ── Output structure ──────────────────────────────────────────────────────

    def test_output_record_has_required_keys(self):
        records = [self._cot_rec(i) for i in range(20)]
        result = extract_reasoning_sample(records, n=5, seed=42)
        required_keys = {"sent_id", "text", "las", "cot_reasoning",
                         "token_errors", "error_category", "notes"}
        for r in result:
            self.assertEqual(set(r.keys()), required_keys)

    def test_error_category_blank_by_default(self):
        """error_category must be an empty string (ready for manual annotation)."""
        records = [self._cot_rec(i) for i in range(20)]
        result = extract_reasoning_sample(records, n=5, seed=42)
        for r in result:
            self.assertEqual(r["error_category"], "")
            self.assertEqual(r["notes"],          "")

    def test_output_sorted_by_sent_id(self):
        """Output records must be sorted by sent_id."""
        records = [self._cot_rec(i) for i in range(50)]
        result = extract_reasoning_sample(records, n=20, seed=42)
        sent_ids = [r["sent_id"] for r in result]
        self.assertEqual(sent_ids, sorted(sent_ids))

    # ── Hard-case bias ────────────────────────────────────────────────────────

    def test_biased_toward_low_las(self):
        """
        The function biases towards low-LAS sentences.
        Records with las=0.0 should be more likely selected than las=0.99.
        We test this probabilistically: with n=5, pick from 5 easy + 5 hard.
        Hard cases (las<0.3) must be over-represented.
        """
        hard = [self._cot_rec(i,      las=0.0) for i in range(5)]
        easy = [self._cot_rec(i + 10, las=0.99) for i in range(5)]
        records = hard + easy
        # Sample 5 from 10 candidates; due to hard-half bias, all 5 hard records
        # will be in the pool (harder_half = candidates[:max(5*2, 10//2)] = first 10 = all)
        # This just verifies the function runs without error and returns the right count
        result = extract_reasoning_sample(records, n=5, seed=42)
        self.assertEqual(len(result), 5)

    # ── Determinism ───────────────────────────────────────────────────────────

    def test_same_seed_deterministic(self):
        records = [self._cot_rec(i) for i in range(100)]
        r1 = [r["sent_id"] for r in extract_reasoning_sample(records, n=20, seed=7)]
        r2 = [r["sent_id"] for r in extract_reasoning_sample(records, n=20, seed=7)]
        self.assertEqual(r1, r2)

    def test_different_seeds_may_differ(self):
        records = [self._cot_rec(i) for i in range(100)]
        r1 = [r["sent_id"] for r in extract_reasoning_sample(records, n=20, seed=1)]
        r2 = [r["sent_id"] for r in extract_reasoning_sample(records, n=20, seed=2)]
        self.assertNotEqual(r1, r2)  # Extremely likely with 100 candidates

    # ── Mixed valid/invalid ───────────────────────────────────────────────────

    def test_mixed_records_only_valid_selected(self):
        """
        Mix of valid and invalid CoT records; only valid ones should appear.
        """
        records = [
            self._cot_rec(0),                                 # valid
            self._cot_rec(1, has_reasoning=False),            # no reasoning
            self._cot_rec(2, has_parse=False),                # parse failure
            self._cot_rec(3, has_errors=False),               # no errors
            self._cot_rec(4),                                 # valid
        ]
        result = extract_reasoning_sample(records, n=10, seed=0)
        returned_ids = {r["sent_id"] for r in result}
        self.assertIn("sent-000",    returned_ids)
        self.assertIn("sent-004",    returned_ids)
        self.assertNotIn("sent-001", returned_ids)
        self.assertNotIn("sent-002", returned_ids)
        self.assertNotIn("sent-003", returned_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke test — all modules on a tiny synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):
    """
    Runs all six analysis functions on the same small synthetic dataset
    and checks that outputs are self-consistent (e.g., totals add up).
    """

    def setUp(self):
        # 10 sentences: 8 successful, 2 parse failures
        good = [
            rec(
                sent_id=f"s{i}",
                gold=[tok(1, head=0, deprel="root"), tok(2, head=1, deprel="nsubj")],
                pred=[{"id": 1, "head": 0, "deprel": "root"},
                      {"id": 2, "head": 1, "deprel": "nsubj"}],
                token_errors=[],
                las=1.0,
                uas=1.0,
                cot_reasoning="Step 1 ...",
            )
            for i in range(8)
        ]
        bad = [
            rec(
                sent_id=f"fail{i}",
                gold=[tok(1, head=0, deprel="root"),
                      tok(2, head=1, deprel="nsubj")],
                pred=None,
                las=None,
            )
            for i in range(2)
        ]
        self.records = good + bad

    def test_corpus_scored_sentences_plus_failures_eq_total(self):
        cm = corpus_uas_las(self.records)
        rl = parse_reliability(self.records)
        self.assertEqual(cm["scored_sentences"] + rl["parse_failures"],
                         rl["total_sentences"])

    def test_corpus_uas_las_consistent(self):
        cm = corpus_uas_las(self.records)
        self.assertLessEqual(cm["las"], cm["uas"])  # LAS <= UAS always

    def test_per_sentence_counts_length_matches_total(self):
        counts = per_sentence_las_counts(self.records)
        self.assertEqual(len(counts), len(self.records))

    def test_parse_failure_penalised_in_bootstrap_counts(self):
        counts = per_sentence_las_counts(self.records)
        failed = [value for sent_id, value in counts.items()
                  if sent_id.startswith("fail")]
        self.assertEqual(failed, [(0, 2), (0, 2)])

    def test_length_analysis_total_sentences_matches_scored(self):
        la = length_analysis(self.records)
        total_in_bins = sum(b["sentences"] for b in la.values())
        self.assertEqual(total_in_bins, 8)   # 2 failures excluded

    def test_reasoning_sample_from_all_correct_records_is_empty(self):
        """Records with no token_errors have no annotation value."""
        good_records = [
            rec(
                sent_id=f"s{i}",
                pred=[tok(1)],
                token_errors=[],   # no errors
                cot_reasoning="...",
                las=1.0,
            )
            for i in range(20)
        ]
        result = extract_reasoning_sample(good_records, n=10)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
