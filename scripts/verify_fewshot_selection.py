"""
verify_fewshot_selection.py
────────────────────────────
Verifies a set of few-shot sentences against the maximum-coverage criterion:

  1. 8–15 tokens (PUNCT excluded from count)
  2. Fully projective dependency structure (no crossing arcs)
  3. Not present in the test or dev split
  4. Collectively cover all 8 target deprels:
       nsubj, obj, obl, det, advmod, amod, compound, case

Also searches the EWT training split for the best 3-sentence set that
satisfies all criteria (use --search to activate).

Usage:
  # Verify a custom set of sentences by sent_id:
  python verify_fewshot_selection.py --verify sent_id1 sent_id2 sent_id3

  # Search for the best trio from EWT train:
  python verify_fewshot_selection.py --search

  # Both:
  python verify_fewshot_selection.py --search --verify sent_id1 sent_id2 sent_id3
"""

import argparse
import os
import sys
from itertools import combinations

BASE       = "data/UD_English-EWT"
TRAIN_FILE = os.path.join(BASE, "en_ewt-ud-train.conllu")
TEST_FILE  = os.path.join(BASE, "en_ewt-ud-test.conllu")
DEV_FILE   = os.path.join(BASE, "en_ewt-ud-dev.conllu")

TARGET_DEPRELS = {"nsubj", "obj", "obl", "det", "advmod", "amod", "compound", "case"}


# ─── CoNLL-U parser ───────────────────────────────────────────────────────────

def parse_conllu(path):
    sentences = []
    current = None
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
            if "-" in tok_id or "." in tok_id:
                continue
            try:
                current["tokens"].append({
                    "id":     int(tok_id),
                    "form":   parts[1],
                    "upos":   parts[3],
                    "head":   int(parts[6]),
                    "deprel": parts[7],
                })
            except (ValueError, IndexError):
                continue
    if current and current["tokens"]:
        sentences.append(current)
    return sentences


# ─── Checks ───────────────────────────────────────────────────────────────────

def is_projective(tokens):
    """True iff the dependency tree has no crossing arcs."""
    arcs = []
    for t in tokens:
        if t["head"] == 0:
            continue
        arcs.append(tuple(sorted([t["id"], t["head"]])))
    for i in range(len(arcs)):
        i1, j1 = arcs[i]
        for j in range(i + 1, len(arcs)):
            i2, j2 = arcs[j]
            if i1 < i2 < j1 < j2 or i2 < i1 < j2 < j1:
                return False
    return True


def deprels_covered(tokens):
    """Set of base deprel labels (before colon) in the token list."""
    return {t["deprel"].split(":")[0] for t in tokens}


def token_count_no_punct(tokens):
    return sum(1 for t in tokens if t["upos"] != "PUNCT")


def check_sentence(sent, held_out_texts):
    tokens  = sent["tokens"]
    n       = token_count_no_punct(tokens)
    proj    = is_projective(tokens)
    covered = deprels_covered(tokens)
    in_held = sent["text"].strip() in held_out_texts

    length_ok = 8 <= n <= 15
    passes    = length_ok and proj and not in_held

    return passes, {
        "sent_id":            sent["sent_id"],
        "text":               sent["text"],
        "n_tokens_no_punct":  n,
        "length_ok":          length_ok,
        "projective":         proj,
        "in_test_or_dev":     in_held,
        "all_deprels":        sorted(covered),
        "target_hits":        sorted(covered & TARGET_DEPRELS),
        "target_missed":      sorted(TARGET_DEPRELS - covered),
    }


def check_trio_coverage(sents):
    combined = set()
    for s in sents:
        combined |= deprels_covered(s["tokens"])
    covered = combined & TARGET_DEPRELS
    return covered, TARGET_DEPRELS - covered


# ─── Verify ───────────────────────────────────────────────────────────────────

def verify_sentences(sent_ids, id_map, held_out_texts):
    print(f"\n{'═'*70}")
    print("  VERIFICATION")
    print(f"{'═'*70}")

    found = []
    for sid in sent_ids:
        if sid not in id_map:
            print(f"\n  ⚠  sent_id not found in train: {sid}")
            continue
        sent = id_map[sid]
        passes, report = check_sentence(sent, held_out_texts)
        icon = "✅ PASS" if passes else "❌ FAIL"
        print(f'\n  {icon}  "{sent["text"]}"')
        print(f"    sent_id            : {report['sent_id']}")
        print(f"    tokens (no punct)  : {report['n_tokens_no_punct']}  |  length_ok: {report['length_ok']}")
        print(f"    projective         : {report['projective']}")
        print(f"    in test/dev?       : {report['in_test_or_dev']}")
        print(f"    all deprels        : {report['all_deprels']}")
        print(f"    target hits        : {report['target_hits']}")
        print(f"    target missed      : {report['target_missed']}")
        if passes:
            found.append(sent)

    if len(found) == 3:
        cov, missing = check_trio_coverage(found)
        print(f"\n  Collective coverage : {sorted(cov)}")
        if missing:
            print(f"  ⚠ MISSING from 8-deprel target: {sorted(missing)}")
        else:
            print(f"  ✓ All 8 target deprels collectively covered!")
    elif found:
        cov, missing = check_trio_coverage(found)
        print(f"\n  Partial collective ({len(found)} sentences): {sorted(cov)}")


# ─── Search ───────────────────────────────────────────────────────────────────

def search_best_trio(train_sents, held_out_texts):
    print(f"\n{'═'*70}")
    print("  SEARCHING FOR BEST TRIO")
    print(f"{'═'*70}")

    candidates = []
    for sent in train_sents:
        if sent["text"].strip() in held_out_texts:
            continue
        passes, _ = check_sentence(sent, held_out_texts)
        if passes:
            covered = deprels_covered(sent["tokens"]) & TARGET_DEPRELS
            candidates.append((len(covered), sent))

    candidates.sort(key=lambda x: x[0], reverse=True)
    print(f"  Eligible candidates: {len(candidates)}")

    top_pool = [s for _, s in candidates[:80]]

    best_coverage = -1
    best_trio     = None
    best_missing  = TARGET_DEPRELS.copy()

    for combo in combinations(top_pool, 3):
        cov, missing = check_trio_coverage(combo)
        if len(cov) > best_coverage:
            best_coverage = len(cov)
            best_trio     = combo
            best_missing  = missing
            if best_coverage == len(TARGET_DEPRELS):
                break

    print(f"\n  Best trio: {best_coverage}/{len(TARGET_DEPRELS)} deprels covered")
    if best_missing:
        print(f"  ⚠ Still missing: {sorted(best_missing)}")
    else:
        print(f"  ✓ All 8 target deprels covered!")

    if best_trio:
        for i, sent in enumerate(best_trio, 1):
            _, report = check_sentence(sent, held_out_texts)
            print(f"\n  ── Sentence {i} " + "─"*50)
            print(f"    sent_id  : {sent['sent_id']}")
            print(f"    text     : {sent['text']}")
            print(f"    tokens   : {report['n_tokens_no_punct']}  |  projective: {report['projective']}")
            print(f"    target hits: {report['target_hits']}")
            print(f"    all deprels: {report['all_deprels']}")

    # Also print top-10 individual candidates
    print(f"\n{'─'*70}")
    print("  TOP-10 INDIVIDUAL CANDIDATES")
    print(f"{'─'*70}")
    for rank, (cov_count, sent) in enumerate(candidates[:10], 1):
        hit = sorted(deprels_covered(sent["tokens"]) & TARGET_DEPRELS)
        n   = token_count_no_punct(sent["tokens"])
        print(f"  [{rank:2d}] ({cov_count}/8 | {n} tok) {sent['text'][:75]}")
        print(f"        sent_id: {sent['sent_id']}")
        print(f"        hits   : {hit}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify and/or search for few-shot sentences satisfying the max-coverage criterion."
    )
    parser.add_argument("--verify", nargs="+", metavar="SENT_ID",
                        help="sent_id values to verify (from EWT train)")
    parser.add_argument("--search", action="store_true",
                        help="Search for the best 3-sentence set in EWT train")
    args = parser.parse_args()

    if not args.verify and not args.search:
        parser.print_help()
        sys.exit(0)

    print("Loading corpora …")
    train_sents = parse_conllu(TRAIN_FILE)
    test_sents  = parse_conllu(TEST_FILE)
    dev_sents   = parse_conllu(DEV_FILE)
    held_out    = {s["text"].strip() for s in test_sents} | {s["text"].strip() for s in dev_sents}
    id_map      = {s["sent_id"]: s for s in train_sents}
    print(f"  Train: {len(train_sents)} | Test: {len(test_sents)} | Dev: {len(dev_sents)}")

    if args.verify:
        verify_sentences(args.verify, id_map, held_out)

    if args.search:
        search_best_trio(train_sents, held_out)


if __name__ == "__main__":
    main()
