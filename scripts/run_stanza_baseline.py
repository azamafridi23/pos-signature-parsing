#!/usr/bin/env python3
"""Supervised reference baseline: run Stanza on the EWT test set under conditions
matched to the LLM evaluation, and score with the same rules.

Matched-conditions protocol:
  - GOLD tokenization (feed the gold syntactic words pretokenized) so tokenization
    is not a confound;
  - PREDICTED POS (Stanza's own tagger), matching the LLM's no-gold-POS setting;
  - score with the SAME rules as src/evaluate_responses.py:
      * align by token position/id,
      * exclude tokens whose GOLD UPOS == PUNCT,
      * UAS = correct head; LAS = correct head AND deprel (full string, lowercased),
      * micro-average (pool correct/total over all tokens).

Run:  python scripts/run_stanza_baseline.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST = os.path.join(ROOT, "data", "UD_English-EWT", "en_ewt-ud-test.conllu")
OUT = os.path.join(ROOT, "results", "stanza_baseline.json")


def read_conllu(path):
    """Yield sentences as lists of gold token dicts (integer-id words only)."""
    sents, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            tid = cols[0]
            if "-" in tid or "." in tid:  # skip MWT ranges and empty nodes
                continue
            cur.append({
                "id": int(tid),
                "form": cols[1],
                "upos": cols[3],
                "head": int(cols[6]),
                "deprel": cols[7],
            })
        if cur:
            sents.append(cur)
    return sents


def main():
    import stanza

    home = os.path.expanduser("~/stanza_resources")
    if not os.path.exists(os.path.join(home, "en")):
        print("Downloading Stanza English models ...")
        stanza.download("en")

    gold_sents = read_conllu(TEST)
    print(f"Loaded {len(gold_sents)} test sentences from {os.path.relpath(TEST, ROOT)}")

    nlp = stanza.Pipeline(
        lang="en",
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,   # feed gold tokens; do not re-tokenize
        use_gpu=False,
        verbose=False,
    )

    # Feed all sentences as pretokenized lists of forms.
    docs = [[t["form"] for t in s] for s in gold_sents]
    print("Parsing with Stanza (predicted POS + heads + deprels) ...")
    parsed = nlp(docs)

    total = uas_correct = las_correct = 0
    mismatch = 0
    for gold, sent in zip(gold_sents, parsed.sentences):
        words = sent.words
        if len(words) != len(gold):
            mismatch += 1
        pred_by_id = {w.id: w for w in words}
        for g in gold:
            if g["upos"] == "PUNCT":      # exclude gold PUNCT (matches evaluator)
                continue
            total += 1
            w = pred_by_id.get(g["id"])
            if w is None:
                continue
            if w.head == g["head"]:
                uas_correct += 1
                if (w.deprel or "").lower() == g["deprel"].lower():
                    las_correct += 1

    uas = round(uas_correct / total, 4)
    las = round(las_correct / total, 4)
    result = {
        "parser": "stanza",
        "stanza_version": stanza.__version__,
        "protocol": "gold tokenization, predicted POS, PUNCT excluded, micro-averaged",
        "total_tokens": total,
        "uas_correct": uas_correct,
        "las_correct": las_correct,
        "uas": uas,
        "las": las,
        "sentence_count_mismatch": mismatch,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Stanza supervised baseline (matched conditions) ===")
    print(f"  tokens scored : {total}")
    print(f"  UAS           : {100 * uas:.1f}")
    print(f"  LAS           : {100 * las:.1f}")
    if mismatch:
        print(f"  [warn] {mismatch} sentences had a token-count mismatch")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
