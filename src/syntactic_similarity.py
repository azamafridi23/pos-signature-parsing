"""
Syntactic Similarity Search Module
Retrieves syntactically similar sentences from the UD_English-EWT dataset
using a syntactic signature similarity method.

Each result is a dict with:
  {
    "text":   str,          # sentence text
    "tokens": list[dict]    # CoNLL-U fields: id, form, upos, head, deprel
  }

Use format_fewshot_example() to render a result as a ready-to-inject
few-shot prompt block (input + gold JSON output).
"""

import os
import stanza
import numpy as np
from collections import Counter
from typing import Any, Dict, List, Tuple
import json as _json

# Assume training file is in the current directory or globally defined
TRAIN_DATA_PATH = "data/UD_English-EWT/en_ewt-ud-train.conllu"

# Lazy-loaded Stanza pipeline
_NLP = None

def get_nlp_pipeline():
    global _NLP
    if _NLP is None:
        # Note: Model should be downloaded beforehand using stanza.download('en')
        _NLP = stanza.Pipeline(lang='en', processors='tokenize,pos', use_gpu=False)
    return _NLP

# Global Cache to process and store training vectors only once
_CACHE: Dict[str, Any] = {
    "sentences":   None,   # np.array of sentence strings
    "lengths":     None,   # np.array of token counts
    "vectors":     None,   # np.array of feature vectors
    "pos_vocab":   None,
    "bigram_vocab": None,
    "token_lists": None,   # list of token dicts [{id, form, upos, head, deprel}, ...]
}

def load_training_data(filepath: str = TRAIN_DATA_PATH) -> List[Tuple[str, List[str], List[dict]]]:
    """
    Loads sentences, their gold UPOS tags, and full token annotations from a CoNLL-U file.

    Args:
        filepath: Path to the CoNLL-U dataset file.

    Returns:
        List of tuples: (sentence_text, list_of_upos_tags, list_of_token_dicts).
        Each token dict contains: id, form, upos, head, deprel.
    """
    data = []
    if not os.path.exists(filepath):
        print(f"Warning: Training file not found at {filepath}")
        return data

    with open(filepath, 'r', encoding='utf-8') as f:
        current_text = ""
        current_tags: List[str] = []
        current_tokens: List[dict] = []
        for line in f:
            line = line.strip()
            if not line:
                if current_text and current_tags:
                    data.append((current_text, current_tags, current_tokens))
                current_text = ""
                current_tags = []
                current_tokens = []
                continue
            if line.startswith("# text = "):
                current_text = line[len("# text = "):]
            elif not line.startswith("#") and "\t" in line:
                parts = line.split('\t')
                # Skip multi-word tokens (e.g. "1-2") or empty nodes (e.g. "1.1")
                if len(parts) >= 10 and '-' not in parts[0] and '.' not in parts[0]:
                    current_tags.append(parts[3])   # UPOS
                    try:
                        current_tokens.append({
                            "id":     int(parts[0]),
                            "form":   parts[1],
                            "upos":   parts[3],
                            "head":   int(parts[6]),
                            "deprel": parts[7],
                        })
                    except (ValueError, IndexError):
                        pass

        # Catch the last sentence if no trailing blank line
        if current_text and current_tags:
            data.append((current_text, current_tags, current_tokens))

    return data

def get_pos_tags(text: str) -> List[str]:
    """
    Predicts Universal POS (UPOS) tags for an input sentence using Stanza.
    Only used for test sentences.

    Args:
        text: Raw sentence text.

    Returns:
        List of POS tags.
    """
    doc = get_nlp_pipeline()(text)
    tags = []
    for sentence in doc.sentences:
        for word in sentence.words:
            tags.append(word.upos)
    return tags

def build_vocabularies(all_tags_list: List[List[str]]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Builds fixed vocabularies for POS unigrams and bigrams based on the training dataset.
    This ensures consistent feature vector dimensions.
    """
    pos_set = set()
    bigram_set = set()

    for tags in all_tags_list:
        for i, tag in enumerate(tags):
            pos_set.add(tag)
            if i < len(tags) - 1:
                bigram_set.add((tag, tags[i+1]))

    return sorted(list(pos_set)), sorted(list(bigram_set))

def extract_syntactic_signature(
    tags: List[str],
    pos_vocab: List[str],
    bigram_vocab: List[Tuple[str, str]]
) -> np.ndarray:
    """
    Computes a syntactic feature vector for a given sequence of POS tags.
    Features:
    - Length features (sentence length, punctuation, verbs, CCONJ, SCONJ)
    - POS unigram/bigram distribution (normalized frequencies)
    - Ratio features (Noun, Verb, ADP, PRON relative to length)
    """
    length = len(tags)
    dim = 5 + len(pos_vocab) + len(bigram_vocab) + 4

    if length == 0:
        return np.zeros(dim)

    # A. Length Features (Normalized to prevent dominating similarity)
    length_ratio = length / (length + 10)
    num_punct = tags.count("PUNCT") / max(1, length)
    num_verbs = (tags.count("VERB") + tags.count("AUX")) / max(1, length)
    num_cconj = tags.count("CCONJ") / max(1, length)
    num_sconj = tags.count("SCONJ") / max(1, length)

    length_features = [length_ratio, num_punct, num_verbs, num_cconj, num_sconj]

    # B. POS Distribution Features
    tag_counts = Counter(tags)
    pos_unigrams = [tag_counts.get(pos, 0) / length for pos in pos_vocab]

    bigrams = [(tags[i], tags[i+1]) for i in range(len(tags)-1)]
    bigram_counts = Counter(bigrams)
    # Safely calculate denominator for bigrams
    bigram_denom = max(1, length - 1)
    pos_bigrams = [bigram_counts.get(bg, 0) / bigram_denom for bg in bigram_vocab]

    # C. Ratio Features
    noun_ratio = tags.count("NOUN") / length
    verb_ratio = (tags.count("VERB") + tags.count("AUX")) / length
    adp_ratio = tags.count("ADP") / length
    pron_ratio = tags.count("PRON") / length

    ratio_features = [noun_ratio, verb_ratio, adp_ratio, pron_ratio]

    # Feature Scaling: Weight features to emphasize local structure (bigrams)
    feature_vector = np.concatenate([
        0.2 * np.array(length_features, dtype=float),
        0.3 * np.array(pos_unigrams, dtype=float),
        0.4 * np.array(pos_bigrams, dtype=float),
        0.1 * np.array(ratio_features, dtype=float)
    ])

    # 3. Vector Normalization (L2 norm)
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm

    return feature_vector

def prepare_training_cache(filepath: str = TRAIN_DATA_PATH):
    """
    Precomputes and caches syntactic signature vectors for all training sentences.
    Ensures dataset is processed only once. Loads from disk if cache exists.
    """
    global _CACHE

    if _CACHE["vectors"] is not None:
        return  # Already initialized in memory

    cache_dir  = os.path.join(os.path.dirname(filepath), "..", "processed")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.normpath(os.path.join(cache_dir, "train_syntactic_cache.npz"))
    if os.path.exists(cache_path):
        print(f"Loading cached syntactic features from {cache_path}...")
        try:
            data = np.load(cache_path, allow_pickle=True)
            _CACHE["sentences"]   = data["sentences"]
            _CACHE["lengths"]     = data["lengths"]
            _CACHE["vectors"]     = data["vectors"]
            _CACHE["pos_vocab"]   = data["pos_vocab"].tolist()
            _CACHE["bigram_vocab"] = [tuple(bg) for bg in data["bigram_vocab"]]
            # token_lists were serialised as JSON strings
            if "token_lists" in data:
                _CACHE["token_lists"] = [_json.loads(s) for s in data["token_lists"]]
            else:
                _CACHE["token_lists"] = None  # old cache — will rebuild below
                raise KeyError("token_lists missing from cache")
            return
        except Exception as e:
            print(f"Failed to load cache from disk ({e}). Rebuilding...")

    data = load_training_data(filepath)
    if not data:
        print(f"No training data loaded. Ensure the dataset path is correct: {filepath}")
        return

    print(f"Precomputing syntactic signatures for {len(data)} training sentences...")

    sentences   = []
    all_tags    = []
    lengths     = []
    token_lists = []

    for text, tags, tokens in data:
        sentences.append(text)
        all_tags.append(tags)
        lengths.append(len(tags))
        token_lists.append(tokens)

    pos_vocab, bigram_vocab = build_vocabularies(all_tags)

    vectors = []
    for tags in all_tags:
        vec = extract_syntactic_signature(tags, pos_vocab, bigram_vocab)
        vectors.append(vec)

    _CACHE["sentences"]   = np.array(sentences)
    _CACHE["lengths"]     = np.array(lengths)
    _CACHE["vectors"]     = np.array(vectors)
    _CACHE["pos_vocab"]   = pos_vocab
    _CACHE["bigram_vocab"] = bigram_vocab
    _CACHE["token_lists"] = token_lists   # kept in Python list (not numpy array)

    # Save cache to disk (token_lists stored as JSON string array)
    try:
        np.savez_compressed(
            cache_path,
            sentences=_CACHE["sentences"],
            lengths=_CACHE["lengths"],
            vectors=_CACHE["vectors"],
            pos_vocab=_CACHE["pos_vocab"],
            bigram_vocab=np.array(_CACHE["bigram_vocab"], dtype=str),
            token_lists=np.array([_json.dumps(tl) for tl in token_lists]),
        )
        print(f"Saved cache to {cache_path}.")
    except Exception as e:
        print(f"Warning: Failed to save cache to disk: {e}")

    print("Caching complete.")

def compute_similarity(test_vector: np.ndarray, train_vectors: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a test vector and an array of training vectors.
    Since vectors are L2-normalized, cosine similarity is equivalent to the dot product.
    """
    return np.dot(train_vectors, test_vector)

def FindSyntacticSimilar(test_sentence: str, k: int = 3) -> List[dict]:
    """
    Returns top-k syntactically similar training sentences with full CoNLL-U annotations.

    Guarantees:
      - The test sentence itself is never returned (exact-text match excluded).
      - No duplicate sentence texts in the results.
      - Length filter uses a minimum window of ±3 tokens to avoid over-restricting short sentences.

    Args:
        test_sentence: The sentence to compare.
        k:             The number of top similar sentences to return.

    Returns:
        List of dicts (length ≤ k), each with:
          {
            "text":   str,         # sentence text
            "tokens": list[dict]   # CoNLL-U token fields: id, form, upos, head, deprel
          }
        Ordered from most to least similar.
    """
    # 1. Ensure training data is processed
    prepare_training_cache()

    if _CACHE["vectors"] is None or _CACHE["token_lists"] is None:
        return []

    pos_vocab       = _CACHE["pos_vocab"]
    bigram_vocab    = _CACHE["bigram_vocab"]
    train_sentences = _CACHE["sentences"]
    train_lengths   = _CACHE["lengths"]
    train_vectors   = _CACHE["vectors"]
    token_lists     = _CACHE["token_lists"]   # plain Python list

    # 2. Extract features from the test sentence
    test_tags   = get_pos_tags(test_sentence)
    test_length = len(test_tags)

    if test_length == 0:
        return []

    test_vector = extract_syntactic_signature(test_tags, pos_vocab, bigram_vocab)

    # 3. Length Filtering
    # Use at least ±3 tokens so short sentences aren't over-restricted and the
    # fallback (which silently changes retrieval behaviour) is avoided.
    tolerance     = max(3, int(0.3 * test_length))
    length_diff   = np.abs(train_lengths - test_length)
    valid_indices = np.where(length_diff <= tolerance)[0]

    if len(valid_indices) == 0:
        # Should be rare with the minimum tolerance; log it for transparency.
        print(f"[WARN] Length filter (±{tolerance}) matched nothing for "
              f"test_length={test_length}. Falling back to full corpus.")
        valid_indices = np.arange(len(train_sentences))

    filtered_sentences   = train_sentences[valid_indices]
    filtered_vectors     = train_vectors[valid_indices]
    filtered_token_lists = [token_lists[i] for i in valid_indices]

    # 4. Similarity Computation
    similarities = compute_similarity(test_vector, filtered_vectors)

    # 5. Rank all candidates descending by similarity
    ranked_indices = np.argsort(similarities)[::-1]

    # 6. Collect top-k, skipping exact-text duplicates and the test sentence itself
    results: List[dict] = []
    seen_texts: set = set()
    test_sentence_stripped = test_sentence.strip()

    for i in ranked_indices:
        if len(results) >= k:
            break
        text = str(filtered_sentences[i]).strip()   # normalise once
        # Fix 1: skip if this is the test sentence itself
        if text == test_sentence_stripped:
            continue
        # Fix 2: skip near-duplicate texts already in results
        if text in seen_texts:
            continue
        seen_texts.add(text)
        results.append({
            "text":   text,
            "tokens": filtered_token_lists[i],
        })

    return results


def format_fewshot_example(example: dict) -> str:
    """
    Formats a single result from FindSyntacticSimilar as a few-shot prompt block.

    The block mirrors the format used by collect_responses.py:
      - Input:  '# text = ...' header + 'id<TAB>form' lines  (same as build_user_message)
      - Output: the gold annotation as a JSON object matching the zero_shot.txt schema

    Args:
        example: A dict with keys 'text' (str) and 'tokens' (list[dict]).

    Returns:
        A formatted string ready to be injected into a few-shot system prompt.
    """
    import json
    text   = example["text"]
    tokens = example["tokens"]

    # --- Input block (mirrors build_user_message in collect_responses.py) ---
    token_lines = "\n".join(f"{t['id']}\t{t['form']}" for t in tokens)
    input_block = f"# text = {text}\n{token_lines}"

    # --- Output block (gold annotation as JSON, matching zero_shot.txt schema) ---
    output_block = json.dumps(
        {
            "tokens": [
                {
                    "id":     t["id"],
                    "form":   t["form"],
                    "upos":   t["upos"],
                    "head":   t["head"],
                    "deprel": t["deprel"],
                }
                for t in tokens
            ]
        },
        ensure_ascii=False,
        indent=2,
    )

    return f"Input:\n{input_block}\n\nOutput:\n{output_block}"

# --- Example Usage --- #
if __name__ == "__main__":
    # Ensure you have the dataset file 'en_ewt-ud-train.conllu' in your directory.
    # Note: If it's your first time, you may need to run stanza.download('en')

    test_string = "I love eating pizza with my friends."
    k = 3

    print(f"Finding top {k} syntactically similar sentences for:")
    print(f"'{test_string}'\n")

    similar = FindSyntacticSimilar(test_string, k=k)

    if similar:
        print(f"Top {k} similar sentences:\n")
        for i, example in enumerate(similar, 1):
            print(f"── Example {i} ──────────────────────────────")
            print(format_fewshot_example(example))
            print()
    else:
        print("No similar sentences found or training data could not be loaded.")
