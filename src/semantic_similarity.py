"""
Semantic Similarity Search Module
Retrieves semantically similar sentences from the UD_English-EWT training set
using Sentence-BERT embeddings (KATE-style retrieval; Liu et al., 2022).

This is the *semantic* counterpart of syntactic_similarity.py and exists to
isolate the example-selection criterion: it returns results in the identical
format and is rendered with the identical format_fewshot_example(), so the
only difference between the two retrieval conditions is how the k examples
are chosen.

Deliberate differences from the syntactic pipeline:
  - Similarity is cosine over sentence embeddings of the raw text
    (all-mpnet-base-v2), not POS-signature vectors.
  - No length filter: plain top-k nearest neighbours, matching standard
    semantic-retrieval practice in prior work.

Shared guarantees (same as FindSyntacticSimilar):
  - The test sentence itself is never returned (exact-text match excluded).
  - No duplicate sentence texts in the results.

Each result is a dict with:
  {
    "text":   str,          # sentence text
    "tokens": list[dict]    # CoNLL-U fields: id, form, upos, head, deprel
  }
"""

import os
import json as _json
from typing import Any, Dict, List

import numpy as np

from syntactic_similarity import (
    TRAIN_DATA_PATH,
    load_training_data,
    format_fewshot_example,  # re-exported so callers can import it from here
)

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Lazy-loaded SentenceTransformer model
_MODEL = None

def get_embedding_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL

# Global cache to embed the training set only once per process
_CACHE: Dict[str, Any] = {
    "sentences":   None,   # np.array of sentence strings
    "embeddings":  None,   # np.array (n_train, dim), L2-normalized
    "token_lists": None,   # list of token dicts [{id, form, upos, head, deprel}, ...]
}


def prepare_semantic_cache(filepath: str = TRAIN_DATA_PATH):
    """
    Precomputes and caches sentence embeddings for all training sentences.
    Loads from disk if the cache exists; otherwise embeds once and saves.
    """
    global _CACHE

    if _CACHE["embeddings"] is not None:
        return  # Already initialized in memory

    cache_dir  = os.path.join(os.path.dirname(filepath), "..", "processed")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.normpath(os.path.join(cache_dir, "train_semantic_cache.npz"))

    if os.path.exists(cache_path):
        print(f"Loading cached semantic embeddings from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            if str(data["model_name"]) != EMBEDDING_MODEL:
                raise ValueError(
                    f"cache built with {data['model_name']}, expected {EMBEDDING_MODEL}"
                )
            _CACHE["sentences"]   = data["sentences"]
            _CACHE["embeddings"]  = data["embeddings"]
            _CACHE["token_lists"] = [_json.loads(s) for s in data["token_lists"]]
            return
        except Exception as e:
            print(f"Failed to load cache from disk ({e}). Rebuilding...")

    data = load_training_data(filepath)
    if not data:
        print(f"No training data loaded. Ensure the dataset path is correct: {filepath}")
        return

    sentences   = [text for text, _tags, _tokens in data]
    token_lists = [tokens for _text, _tags, tokens in data]

    print(f"Embedding {len(sentences)} training sentences with {EMBEDDING_MODEL} "
          f"(one-time precomputation)...")
    model = get_embedding_model()
    embeddings = model.encode(
        sentences,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalized → cosine = dot product
    ).astype(np.float32)

    _CACHE["sentences"]   = np.array(sentences)
    _CACHE["embeddings"]  = embeddings
    _CACHE["token_lists"] = token_lists

    try:
        np.savez_compressed(
            cache_path,
            sentences=_CACHE["sentences"],
            embeddings=_CACHE["embeddings"],
            token_lists=np.array([_json.dumps(tl) for tl in token_lists]),
            model_name=EMBEDDING_MODEL,
        )
        print(f"Saved cache to {cache_path}.")
    except Exception as e:
        print(f"Warning: Failed to save cache to disk: {e}")

    print("Caching complete.")


def FindSemanticSimilar(test_sentence: str, k: int = 3) -> List[dict]:
    """
    Returns top-k semantically similar training sentences with full CoNLL-U
    annotations, ordered from most to least similar.

    Plain cosine top-k over Sentence-BERT embeddings — no length filter —
    with the same exclusion rules as FindSyntacticSimilar (the test sentence
    itself and duplicate texts are skipped).
    """
    prepare_semantic_cache()

    if _CACHE["embeddings"] is None or _CACHE["token_lists"] is None:
        return []

    train_sentences = _CACHE["sentences"]
    train_embeddings = _CACHE["embeddings"]
    token_lists      = _CACHE["token_lists"]

    model = get_embedding_model()
    test_vector = model.encode(
        [test_sentence],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(np.float32)

    similarities  = np.dot(train_embeddings, test_vector)
    ranked_indices = np.argsort(similarities)[::-1]

    results: List[dict] = []
    seen_texts: set = set()
    test_sentence_stripped = test_sentence.strip()

    for i in ranked_indices:
        if len(results) >= k:
            break
        text = str(train_sentences[i]).strip()
        if text == test_sentence_stripped:
            continue
        if text in seen_texts:
            continue
        seen_texts.add(text)
        results.append({
            "text":   text,
            "tokens": token_lists[i],
        })

    return results


# --- Example Usage --- #
if __name__ == "__main__":
    test_string = "I love eating pizza with my friends."
    k = 3

    print(f"Finding top {k} semantically similar sentences for:")
    print(f"'{test_string}'\n")

    similar = FindSemanticSimilar(test_string, k=k)

    if similar:
        print(f"Top {k} similar sentences:\n")
        for i, example in enumerate(similar, 1):
            print(f"── Example {i} ──────────────────────────────")
            print(format_fewshot_example(example))
            print()
    else:
        print("No similar sentences found or training data could not be loaded.")
