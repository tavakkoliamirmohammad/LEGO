# Candidate Schema

Every row in `evaluation/survey.md` is a YAML block with the structure
below. Any candidate missing any required field is dropped from the
survey before builders are dispatched.

## Required fields

```yaml
id: <slug>                       # also the candidates/<id>/ directory name;
                                 # convention: NN-<suite>-<kernel>-<layout>
suite: <name and version>        # e.g. "PolyBench/C 4.2.1"
kernel: <kernel name>            # specific kernel within the suite
upstream_url: <full URL>         # tarball, git ref, or release tag — exact
license: <SPDX id>               # accepted: MIT, BSD-2-Clause, BSD-3-Clause,
                                 # Apache-2.0, ISC, public-domain, CC0
                                 # copyleft (GPL/LGPL/AGPL) -> drop
language: <one of: c, cpp, fortran, rust, julia>
baseline:
  source_files: [<path within upstream>]
  build: "<exact build command>"
  threading: "<single-threaded | NN-thread OpenMP | etc>"
layout_trick: <short description>
layout_trick_citation: <bibtex key in references.bib>
why_compiler_cant: |
  <one paragraph naming the specific compiler pass that would be
   required (polyhedral, non-affine vectorization, etc.) and why it
   does NOT fire on naive code at the suite's baseline build flags>
lego_expressibility: |
  <Python sketch of the LEGO expression, using OrderBy + TileBy as
   building blocks; GroupBy permitted with one-line justification>
predicted_win:
  value: <"X.Yx" or "X.Yx – Z.Wx" or "unknown">
  source: <bibtex key OR "unknown">
  type: <"published" | "extrapolated" | "unknown">
power_of_two_restriction:
  baseline_assumes_pow2: <true | false>
  test_at_non_pow2_size: <true | false>
measurement_protocol: |
  Median of 100 runs after 25 warmup; taskset core pinning; numactl,
  governor, turbo as observed.
estimated_builder_effort: <"X-Y days">
risk_flags:
  - <one line per known risk>
```

## Drop rules

A candidate is dropped before the builder is dispatched if any of the
following are true:

1. Any required field is missing or empty.
2. License is not in the accepted list.
3. `language` is not one of the five CPU emission targets.
4. `predicted_win.source` is not "unknown" but the BibTeX key is not
   present in `references.bib`.
5. `lego_expressibility` requires a primitive outside `Row`, `Col`,
   `RegP`, `GenP`, `OrderBy`, `TileBy` (e.g. references the Tensor API
   directly).
6. `why_compiler_cant` does not name a specific compiler pass.

## Granularity rule

One row per **(kernel × layout-trick)** tuple, not per kernel. The same
kernel under different layouts becomes multiple candidates so the
paper's evaluation matrix can compare layouts head-to-head.

## Honesty rules

- No invented numbers. `predicted_win.type: unknown` is the right
  answer when no published number exists.
- Every BibTeX key must resolve to a real paper with DOI or arXiv ID.
- "Why the compiler can't recover this" must be specific enough that a
  reviewer can verify it (name the pass, name the flag, link the GCC
  bug if relevant).
