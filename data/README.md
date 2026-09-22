# Data

`UD_English-EWT/` contains the Universal Dependencies English Web Treebank files
used by the study:

- `en_ewt-ud-train.conllu`: demonstration-retrieval pool;
- `en_ewt-ud-dev.conllu`: retained for dataset completeness but not used in the
  reported experiments; and
- `en_ewt-ud-test.conllu`: the 2,077-sentence evaluation split.

The original corpus `README.md` and `LICENSE.txt` are retained beside the data.
The annotations and database are distributed under CC BY-SA 4.0; consult those
files for attribution and underlying-text notices.

Retrieval caches are intentionally not versioned. The collection code creates
them under `data/processed/` when needed.
