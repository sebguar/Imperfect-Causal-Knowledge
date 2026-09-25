# Certification log

This log records the certification of the reproducibility package: the
environment it was certified in, the platform constraint that fixes that
environment, the test run, the analysis and figure re-derivation performed over
frozen inputs, the results it reproduced, the verification pass over its
outputs, and the artifact set that ships.

## Environment

Operating system: Windows 11 Home, build 10.0.26200, x86-64.
Python: CPython 3.11.9, MSC v.1938, 64-bit.

Twelve pinned distributions.

Runtime, from `requirements.txt`:

| distribution | version |
|---|---|
| causal-learn | 0.1.4.8 |
| matplotlib | 3.11.0 |
| networkx | 3.6.1 |
| numpy | 2.4.6 |
| omegaconf | 2.3.1 |
| pandas | 3.0.3 |
| pyyaml | 6.0.3 |
| scikit-learn | 1.9.0 |
| scipy | 1.17.1 |

Development and test, from `requirements-dev.txt`:

| distribution | version |
|---|---|
| pillow | 12.3.0 |
| pytest | 9.1.1 |
| ruff | 0.15.20 |

The live environment was verified against these pins. Every installed version,
read through `importlib.metadata`, equals its pin: twelve of twelve, none
differing. A dry-run install of both requirement files resolved every
requirement as already satisfied and proposed no download, no upgrade and no
new installation, so both dry runs are no-ops. The full transitive freeze of
that environment is `env_snapshots/pip_freeze_windows_py311.txt`.

## Why this platform

The summary tables under each results root are written with CRLF line endings,
and eleven digest constants across the seven figure modules hash exactly those
bytes. Each figure script re-reads the summary file it draws from, takes the
SHA-256 of the raw bytes on disk, and aborts on a mismatch before drawing
anything.

A platform whose writer emitted LF would produce a different byte string for
numerically identical tables, so re-emitting the summaries there fails the
pins on the line endings alone, independently of content. Two rules follow, and
both are load-bearing: the summary writer is barred from any line-ending
change, and certification runs on Windows.

Separately, the artifacts tree is marked in `.gitattributes` so that Git stores
and returns its bytes unchanged instead of normalizing them on checkout. That
marking is what lets the digest pins hold in any clone.

## Test run

`pytest` collects 678 tests: 651 passed, 27 skipped, 0 failed, 0 errors. The 27
skips are data-presence guards, which stand down when the per-cell outputs they
read are absent.

## The re-derivation

No grid was run. The per-cell outputs were taken as frozen numeric inputs and
only the analysis and figure layer was re-derived over them. The three summary
rebuilds below read the existing per-cell files and rewrite the summary tables;
none of them regenerates a cell.

Every command line, in executed order, with its exit status:

```
python -m scripts.run_cross_seed_grid --root results/cross_seed_N6 --n-seeds 6 --summary-only
  exit 0
python -m scripts.run_cross_seed_grid --root results/cross_seed_N20 --n-seeds 20 --summary-only
  exit 0
python -m scripts.run_cross_seed_grid --root results/cross_seed_L1d_N20 --n-seeds 20 --summary-only --l1-discovered
  exit 0
python -m scripts.verify_grid_prefix
  exit 0
python -m scripts.verify_grid_prefix --ref results/cross_seed_N20 --new results/cross_seed_L1d_N20 --n-seeds 20 --kinds scoring-only --all-topologies
  exit 0
python -m scripts.chain_detectability_gate --root results/cross_seed_N20 --out results/cross_seed_N20/summary/chain_detectability_gate.md
  exit 0
python -m scripts.run_t4_analysis --root results/cross_seed_N20 --n-seeds 20 --reports t4a t4b t4c
  exit 0
python -m scripts.run_h4_analysis --root results/cross_seed_N20 --n-seeds 20
  exit 0
python -m scripts.run_h3_analysis --root results/cross_seed_L1d_N20 --n-seeds 20
  exit 0
python -m scripts.run_h3_reemission --root results/cross_seed_L1d_N20 --n-seeds 20
  exit 0
python -m scripts.fig1_cost_ladder --root results/cross_seed_N20
  exit 0
python -m scripts.fig2_validity_gap --root results/cross_seed_N20
  exit 0
python -m scripts.fig3_detectability --root results/cross_seed_N20
  exit 0
python -m scripts.fig4_h4_scatter --root results/cross_seed_N20
  exit 0
python -m scripts.fig5_regime_contrast --root results/cross_seed_N20
  exit 0
python -m scripts.fig6_h3_interpolation --root results/cross_seed_L1d_N20
  exit 0
python -m scripts.fig7_h3_interaction --root results/cross_seed_L1d_N20
  exit 0
```

The two prefix verifications passed: the first reported 288 of 288 comparisons
float-exact, the second 720 of 720 float-exact.

All eleven digest constants matched byte-exact against the freshly rebuilt
summary tables.

All 21 figure artifacts came back byte-identical to the baseline: seven raster
images, seven vector files and seven caption files.

## Results identity

The five reported results, as regenerated:

| hypothesis | result |
|---|---|
| H1 | Null |
| H2 | Supported |
| H3a | Mixed (1/2) |
| H3b | Null |
| H4 | Null |

Precision note. Two of these are emitted by their reports in the mechanical
criterion vocabulary rather than in the three-label vocabulary, by design:
`t4a_h1_monotonicity.md` records NOT MET, and `t4b_validity_gap.md` records MET
on 3 of 3 topologies. The Null and Supported labels for those two therefore live
in the reported record rather than in the artifact, and an audit should not
expect to find those labels in those two files. The other three carry their
labels directly: `h4_verdict.md` gives Null, and `h3_verdict_v2.md` gives
Mixed (1/2) and Null.

## Verification pass

A verification pass ran over the freshly emitted artifacts: 26 checks executed,
none failed. Its report is 48,828 bytes, SHA-256
`94399f8479f4228e53c6d41892313f727e7119efe2184f7fbf6256f3db923b08`.

## The shipped artifact set

60 files, 3,647,140 bytes in total, all of them under the summary and figure
directories of the three results roots. The per-cell directories are not in the
repository; they travel separately.

## One diagnosed difference

In the smaller grid root, the structural-descriptor-by-seed table
(`s_of_g_by_seed.csv`) regenerates with 72 data rows where an earlier copy held
48. The columns are identical, the 48 shared rows carry zero numeric
differences, and the 24 additional rows cover the four chain cells across the
six seeds of that root. The descriptor is model-derived, so the summary rebuild
computes it for all twelve cells regardless of which of them have runs in that
root. A strict superset, not a drift.

## Long paths

The per-cell tree that travels separately is deep. Measured from the repository
root, its longest path reaches 121 characters, in a per-seed cell directory. On
Windows with long-path support disabled the whole absolute path must stay under
260 characters, so unpacking that tree beneath a parent directory longer than
roughly 138 characters fails before anything can run. Clone to a short path, or
enable long paths.

## Documented residuals

- `G2_CRITERION`, `G3_CRITERION`: the two criterion constants; their values are
  the criterion texts quoted in the H1 and H2 reports.
- `Gap_cost`: the code identifier for the H2 gap; the thesis symbol is Gap_c.
- the d2/d3 identifier family (modules and constants):
  - `h3_d2`, `run_d2_check`, `d2_fork`, `D2Result`, `D2_CELL`: "Truth-side check
    for triangle/linear/effect-modifying (LD-2)."
    (`icknowledge/analysis/h3_d2.py:1`)
  - `D3_REGION3_CLAUSE`, `D3_ON_FIGURE_TEXT`: "[PS-8] The clause a
    region-3-driven orientation label carries VERBATIM."
    (`icknowledge/analysis/h3_v2.py:113`)
- `V_h`: "the classifier's feature set" (`icknowledge/descriptor/s_of_g.py:274`);
  the remainder of that quoted line names the row index set and is omitted here
  to keep this file ASCII.
- `t4`, `t4a`, `t4b`, `t4c`: the H1, H2 and detectability-gate report family and
  its driver.
- lowercase `v1`/`v2`: the first-stage and re-emitted H3 verdict reports
  (`h3_verdict.md` / `h3_verdict_v2.md`); their step labels (1a-3g) are
  report-internal and not aligned between the two.
- "PS-8 sub-clause (i)" in the figure code: from source, "Which region ONE
  family arm's seeds predominantly occupied"
  (`scripts/fig7_h3_interaction.py:281`).
- author-name references (Ehyaei, Von Kuegelgen, Karimi): literature anchors, by
  design.
- PDF Creator/Producer strings: matplotlib version.
- the supersession SHA `d8bdd23`: names the superseded first-stage emission in
  the development history, which is not present in this repository. It appears
  in `icknowledge/analysis/h3_v2.py:65-66` and
  `tests/test_h3_reemission.py:463,485`, and in two tracked artifacts,
  `artifacts/cross_seed_L1d_N20/summary/h3_verdict_v2.md:6-7` and
  `artifacts/cross_seed_L1d_N20/summary/h3_instance_table_v2.csv:2`.
- manifest residuals (HU Box data, not the repository): run timestamps, Windows
  paths, tool version strings, git SHAs.
- the PS-4 byte-identity assertion is implemented as value-exact identity after
  a round-trip parse (`scripts/verify_grid_prefix.py`); the register wording
  stands.
- the calibration provenance file ships with its register citations and
  generation timestamp removed (values untouched; the local original is
  preserved); its recorded source commit predates the shipped history and is
  documentary.
- post-certification text edit: line 10 of the emitted report
  `t4c_detectability_gate.md` carried a grammatical artifact of the vocabulary
  pass ("mechanics's"); it was corrected in the writer and, in the same commit,
  in the tracked copy under `artifacts/`. No value, verdict or other byte
  changed, and the corrected writer/artifact pair was re-verified by a fresh
  re-run of the README analysis block: all 26 outputs byte-identical.
- post-certification text edit: two sentences in `README.md` were corrected. The
  frozen-inputs sentence named a mechanism diagnostic that ships in neither
  channel, and the figure-pin sentence overstated pin coverage: figures 6 and 7
  additionally read `h3_instance_table_v2.csv`, which carries no pin. No command
  line, no value, no verdict and no tracked artifact changed; the 60-file,
  3,647,140-byte artifact set and all eleven digest constants are unaffected, so
  the analysis block was not re-run.
- a second unresolvable commit reference: the `git_commit` column of
  `h3_covariates.csv` (column 17) and `h3_per_seed.csv` (column 27) records
  `e004b6185e7a245bad2163ca6fe7a4a11da34e0f` in all 240 rows of each file. Like
  the supersession SHA above, it names a commit of the development history,
  which is not present in this repository, and it is documentary: no value in
  either table depends on resolving it. The tables ship unedited, so their
  digests and the artifact set are unchanged.
