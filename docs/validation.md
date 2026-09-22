# Validation

Validated on 22 September 2026 in a fresh Python 3.12.14 environment with
NumPy 2.3.5 and Pillow 12.3.0.

- Recomputed corpus scores, 10,000 paired-bootstrap resamples (seed 42),
  relation/length diagnostics, critique churn, qualitative selections, and table
  exports. All 19 delivered result files matched the prior final package byte
  for byte, and all numerical reference hashes passed.
- Verified all 24 compressed archives and their uncompressed identities.
  All 24,921 raw response records agreed with evaluation records and bundled
  CoNLL-U gold annotations. Final-record usage totals 66,003,325 tokens.
- Reconstructed all 24,921 system/user prompt pairs from source templates,
  demonstration serialization, and extracted first-pass parses.
- Passed 119 deterministic tests: 112 retained tests plus seven cases covering
  compressed diagnostic loading and safe tracker initialization.
- Preserved the 24 archives, six prompt sources, and five upstream data/license
  files byte for byte. Regenerated diagnostic figures from unchanged numbers.

No new API calls, model inference, or Stanza baseline run was performed.
Font availability can change regenerated figure bytes; numerical reference
hashes determine result agreement. This validates the supplied LLM artifact,
not the missing historical Stanza run record or future hosted-model behavior.

## Locked environment workflow

The uv/Make workflow was also validated on 22 September 2026 using uv 0.12.5
and a fresh Python 3.12.14 environment. All four targets (`make setup`,
`make verify`, `make reproduce`, and `make test`) passed. After initial package
downloads, validation ran with `UV_OFFLINE=1`. Setup installed only NumPy and
Pillow; the test target installed the separate locked development group.

The lockfile did not change during validation. All 119 tests passed, and all
archive, data, prompt, and numerical result files remained byte-identical.
The requirements files are generated hash-bearing exports of the same lockfile.
Optional inference dependencies were resolved but were not installed or run.

## Experimental-input checksum scope

On 22 September 2026, removed the whole-repository checksum manifest. The input
manifest now covers 35 scientific input/reference files, excluding documentation.
`make verify` and `make reproduce` passed offline with unchanged numerical
references, and all 122 tests passed. Regression tests confirm that routine
source/documentation/configuration/figure edits do not change the manifest,
while altered or missing experimental inputs fail verification. Archives,
datasets, prompts, and delivered numerical results remained byte-identical.
