# Stanza reference resources

The saved supervised reference result records **Stanza 1.11.1**. Inspection
of the original experiment machine recovered English resources under
**resources version 1.11.0**. Software and resource versions are separate
identifiers.

## Package identifiers

The baseline requests `tokenize,pos,lemma,depparse`, uses
`tokenize_pretokenized=True` and `use_gpu=False`, and has no package override.
The recovered resource registry identifies these English default packages:

| Component | Package |
|---|---|
| Tokenizer | `combined` |
| POS tagger | `combined_charlm` |
| Lemmatizer | `combined_nocharlm` |
| Dependency parser | `combined_charlm` |
| Pretrained word embeddings | `conll17` |
| Forward character language model | `1billion` |
| Backward character language model | `1billion` |

The supervised baseline uses gold-tokenized input and CPU execution.
The original script, cached POS/parser files, and saved result have timestamps
on 9 June 2026 around 02:13, 02:15, and 02:16, respectively. This supports
linking the recovered default resources to the baseline. The recovered files
do not establish the resource configuration for every earlier POS-retrieval run.

## Saved reference result

The existing result summary records 21,998 scored tokens, 19,962 correct heads,
19,369 correct labeled arcs, zero sentence token-count mismatches, and Stanza
1.11.1. Its stored proportions are `uas=0.9074` and `las=0.8805`.
These are recovered records; no new baseline run was performed.

Model weights are not bundled in this repository. The saved result and resource
information were recovered after artifact v0.1.0 was published; that immutable
Zenodo archive does not contain this later documentation. Environment versions
and retrieval configuration are documented in [inference.md](inference.md).
