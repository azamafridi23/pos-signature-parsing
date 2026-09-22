# Evaluation Protocol and File Schema

## Shared scoring

All models and conditions are aligned to the same lexicographically ordered EWT test sentence identifiers. The common denominator is 21,998 tokens whose gold UPOS is not `PUNCT`. UAS compares heads; LAS compares heads and case-insensitive full dependency-relation strings, including subtypes. Predictions are matched by token ID, with the last duplicate ID used. Missing predictions receive no credit. Tree diagnostics are reported separately and do not invalidate otherwise scorable token attachments.

Schema extraction uses the first opening JSON brace and requires the remainder to decode, then normalizes the required token fields. This follows the established archive scorer. The verifier confirms agreement with every provided evaluation record and checks gold token IDs, forms, UPOS, heads, and relations against the bundled CoNLL-U test file.

Paired inference uses 10,000 sentence resamples, `random.Random(42)`, identical resampled indices across systems, and token-weighted micro LAS on each sample. Intervals are pointwise percentile intervals. Two-sided tail probabilities use add-one smoothing. Four POS-versus-Semantic / POS-versus-Fixed contrasts form the primary Holm family. The other comparisons are exploratory; separate 15-comparison Holm values are also supplied. Read `first` and `second` to determine every contrast's direction.

Critique edit rates are conditional on available scorable parses at both stages. Corpus LAS and its contrasts continue to use the full test denominator. Relation and length diagnostics are descriptive.

## Archived records

Each `outputs/<model>/<condition>/model_responses.json.gz` decompresses to the exact final JSON response file. Records include `sent_id`, submitted `prompt.system` and `prompt.user`, `raw_response`, configuration, recorded usage, available execution metadata, and `gold` annotations.

Each paired `model_evaluation.json.gz` contains extracted `llm_parse` values, gold annotations, and the archived scoring/validation fields. A null parse remains a failure; absence of a record is distinct from a null parse. The verifier checks both consistently.

`docs/input_identity.json` records the uncompressed identity of all 24 input files. The input checksums separately cover their compressed bytes. Compression changes storage, not the response content.

## Prompt and historical limits

The fixed critique label inventory omits `det`; the CoT worked example overgeneralizes punctuation attachment. Both are preserved as used. Effective messages can differ from fixed templates, so archived messages are authoritative for each call. Their effects were not independently isolated.

Provider-side checkpoint revisions and the exact historical Stanza resource package were not preserved. The development split was unused, and the record does not establish that design decisions preceded test-output inspection. This is an exploratory final-output evaluation.

## Reproduction outputs

The numerical reference hashes cover the original dataset statistics, scoring objects (excluding the storage-path description), paired tests, critique churn, and qualitative/relation/length outputs. Final-record token usage is recomputed from all supplied records with usage fields.

Figure rendering uses Pillow and locally available fonts. Numeric results must match the reference hashes; figure pixels may vary across platforms. Figures are not checksummed; `make verify` validates numerical agreement after regeneration without refreshing the input manifest.
