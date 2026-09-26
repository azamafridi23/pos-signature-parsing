# Evaluation Protocol and File Schema

## Shared scoring

All models and conditions are aligned to the same lexicographically ordered EWT test sentence identifiers. The common denominator is 21,998 tokens whose gold UPOS is not `PUNCT`. UAS compares heads; LAS compares heads and case-insensitive full dependency-relation strings, including subtypes. Predictions are matched by token ID, with the last duplicate ID used. Missing predictions receive no credit. Tree diagnostics are reported separately and do not invalidate otherwise scorable token attachments.

Schema extraction uses the first opening JSON brace and requires the remainder to decode, then normalizes the required token fields. This follows the established archive scorer. The verifier confirms agreement with every provided evaluation record and checks gold token IDs, forms, UPOS, heads, and relations against the bundled CoNLL-U test file.

Paired inference uses 10,000 bootstrap samples with `random.Random(42)`. Each sample draws 2,077 sentences with replacement and uses identical sentence indices across systems. Token-weighted micro LAS is recomputed using the scored-token count in that sample as its denominator; the original corpus denominator of 21,998 does not remain fixed across resamples. Pointwise 95% confidence intervals use the 2.5th and 97.5th percentiles of the sampled LAS differences.

Approximate two-sided percentile-bootstrap p-values use add-one smoothing. For `R = 10,000` and sampled differences `delta = first - second`, the lower and upper tail estimates are `(1 + count(delta <= 0)) / (R + 1)` and `(1 + count(delta >= 0)) / (R + 1)`. The reported p-value is twice the smaller estimate, capped at one. Zero differences count in both tails; this is an approximate bootstrap estimate, not an exact test.

Holm correction at alpha .05 is applied jointly to four focal contrasts: POS-Signature versus Semantic and versus Fixed for each model. These are labeled `primary` in the result files. Other comparisons are exploratory; separate 15-comparison Holm values are also supplied for each model. Read `first` and `second` to determine every contrast's direction. Confidence intervals are pointwise and are not adjusted for multiple comparisons.

Critique edit rates are conditional on available scorable parses at both stages. Observed corpus LAS and observed corpus contrasts continue to use the full test denominator. Relation and length diagnostics are descriptive.

## Archived records

Each `outputs/<model>/<condition>/model_responses.json.gz` decompresses to the exact final JSON response file. Records include `sent_id`, submitted `prompt.system` and `prompt.user`, `raw_response`, configuration, recorded usage, available execution metadata, and `gold` annotations.

Each paired `model_evaluation.json.gz` contains extracted `llm_parse` values, gold annotations, and the archived scoring/validation fields. A null parse remains a failure; absence of a record is distinct from a null parse. The verifier checks both consistently.

`docs/input_identity.json` records the uncompressed identity of all 24 input files. The input checksums separately cover their compressed bytes. Compression changes storage, not the response content.

## Prompt and historical limits

The fixed critique label inventory omits `det`; the CoT worked example overgeneralizes punctuation attachment. Both are preserved as used. Effective messages can differ from fixed templates, so archived messages are authoritative for each call. Their effects were not independently isolated.

Prompts, retrieval weights, and fixed examples were finalized before evaluation; test outputs were inspected afterward to interpret the results. The development split was unused. Relation- and length-based diagnostics are post-hoc and should be interpreted as exploratory.

Provider-side checkpoint revisions and the exact historical Stanza resource package were not preserved. Additional historical inference and retrieval metadata, including unavailable environment details, are documented in [inference.md](inference.md).

## Reproduction outputs

The numerical reference hashes cover the original dataset statistics, scoring objects (excluding the storage-path description), paired tests, critique churn, and qualitative/relation/length outputs. Final-record token usage is recomputed from all supplied records with usage fields.

Figure rendering uses Pillow and locally available fonts. Numeric results must match the reference hashes; figure pixels may vary across platforms. Figures are not checksummed; `make verify` validates numerical agreement after regeneration without refreshing the input manifest.
