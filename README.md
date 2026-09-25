# The Price of Imperfect Causal Knowledge: Group-Asymmetric Recourse Burden in Credit Scoring

**Type:** Master's Thesis

**Author:** Sebastian Guardia Vaca

**1st Examiner:** Prof. Dr. Stefan Lessmann

**2nd Examiner:** Prof. Dr. Jan Mendling

**Affiliation:** Chair of Information Systems, School of Business and Economics, HU Berlin

## Table of Content

- [Summary](#summary)
- [Working with the repo](#working-with-the-repo)
  - [Dependencies](#dependencies)
  - [Setup](#setup)
- [Reproducing results](#reproducing-results)
  - [What ships where](#what-ships-where)
  - [Analysis and figures](#analysis-and-figures)
  - [Rebuilding from the cell-level data](#rebuilding-from-the-cell-level-data)
  - [How the grids were generated](#how-the-grids-were-generated)
- [Results](#results)
- [Project structure](#project-structure)

## Summary

Algorithmic recourse tells a rejected applicant what to change to be accepted. Causal recourse
methods compute that advice under a structural causal model (SCM) of the applicant's features, so
that changing one feature propagates to its descendants. In practice the SCM is never known
exactly. This thesis asks what happens to the fairness of recourse as the causal knowledge
available to the recourse generator is degraded, and whether the resulting burden falls
asymmetrically across demographic groups.

The design is a knowledge ladder with four rungs applied to one fixed recourse procedure: L2 (the
true SCM), L1-oracle (the true graph with equations estimated from data), L1-discovered (a graph
recovered by causal discovery, equations estimated on it) and L0 (no causal model). Every rung's
actions are scored against the true SCM. The protected attribute enters each synthetic SCM either
additively or as an effect modifier, and the contrast between the two regimes is the mechanism
under test. Four hypotheses are evaluated across three graph topologies (triangle, collider,
chain) and two functional families (linear, nonlinear-Gaussian), on 20 seeds per cell:

- **H1** - realized recourse cost degrades monotonically down the ladder;
- **H2** - the gap between the burden the generator believes it imposes and the burden it
  realizes is carried by validity, and separates SCM-informed rungs from the associational
  baseline;
- **H3** - replacing the true graph with a discovered one costs more under nonlinear mechanisms
  than under linear ones (H3a), and a discovered graph can be worse than no graph (H3b);
- **H4** - the group-asymmetric part of the burden increase tracks a pre-specified structural
  descriptor S(g) of how badly each group's sub-SCM is approximated, and does so only under
  effect modification.

All decision rules, thresholds and populations were register-logged prior to outcome exposure;
the thesis appendix ("Pre-specification and Deviation Record", ids **PS-n** / **LD-n**) is where
every rule cited in this repository resolves. Null results are reported as findings.

H1, H3b and H4 return Null, H2 is Supported, and H3a is Mixed: where the family-by-provenance
interaction fires, it runs counter to the predicted direction. Imperfect causal knowledge
degrades recourse validity systematically without producing the predicted group-asymmetric cost
burden.

**Keywords**: algorithmic recourse, counterfactual explanations, structural causal models, causal
discovery, effect modification, group fairness, credit scoring

**Full text and cell-level data**: available during the evaluation period on the HU Box cloud
service, folder `Imperfect-Causal-Knowledge (09.2026)`:
https://box.hu-berlin.de/d/f3df3b5c268545f3bf4d/

## Working with the repo

### Dependencies

Python **>= 3.11**. All results were produced on CPython 3.11.9 on Windows with the package
versions pinned in `requirements.txt`; the complete transitive freeze of that environment is in
`env_snapshots/`. No GPU is used.

### Setup

1. Clone this repository. Its longest repository-relative path is 83 characters. With the HU Box
   grids placed under `results/`, the longest path is 116 characters, so on Windows either enable
   long paths or clone into a parent directory shorter than about 143 characters.

2. Create a virtual environment and activate it:

```
python -m venv thesis-env
source thesis-env/bin/activate        # Windows: thesis-env\Scripts\activate
```

3. Install requirements:

```
pip install --upgrade pip
pip install -r requirements.txt       # runtime
pip install -r requirements-dev.txt   # tests and linting (optional)
```

4. Run the test suite (678 tests):

```
pytest
```

All commands below run from the repository root with `python -m`; the package is used in place.
On a stock Windows console, prefix the analysis and figure commands with
`PYTHONIOENCODING=utf-8` (or `set PYTHONIOENCODING=utf-8`), because some scripts print
mathematical symbols that the default cp1252 encoding cannot render.

## Reproducing results

### What ships where

The reported numbers are read from frozen artifacts, not re-run. The experiment grids were
generated once from a single pinned meta-entropy and are never regenerated; reproduction
re-derives the analysis and figure layer over them. Two channels carry the material:

- **This repository** tracks, under `artifacts/`, the summary tables, the seven figures with
  their captions, and every verdict and criterion report. Every reported result can be read
  from here without running anything. From these files alone, the figures and the H1, H2 and
  detectability-gate reports regenerate, because they read only the summary tables.
- **The HU Box folder** (link above) carries the cell-level data: the per-individual scoring
  tables, per-cell aggregates and run manifests for all three grids, about 660 MB. With it, the
  H3 and H4 verdict reports, a rebuild of the summary tables and the grid-prefix verification
  regenerate as well.

One frozen file has no regeneration path: the classifier-calibration record referenced by
Appendix A.1, which was produced during model development and has no writer. It ships as a
frozen input only, under `artifacts/cross_seed_N20/summary/`. The `mechanism` report that
`scripts.run_t4_analysis` emits when `--reports` is left at its default is a development
diagnostic; it is not part of the reported record and ships in neither channel.

Certification of this package ran on Windows in the pinned environment. On other platforms the
regenerated values and verdict outcomes are expected to match; byte-level identity of the
emitted files is not claimed.

### Analysis and figures

Outputs are written under `results/`, which is gitignored. The analysis drivers read a grid's
`summary/` directory and refuse to overwrite an existing output, so first copy the tracked
summary tables (the `.csv` files, not the reports) into a fresh workspace. One of those
CSVs, `detectability_gate_provenance.csv`, is itself an output of the t4c report, so it is
removed after the copy; the run below regenerates it:

```
mkdir -p results/cross_seed_N20/summary results/cross_seed_L1d_N20/summary
cp artifacts/cross_seed_N20/summary/*.csv results/cross_seed_N20/summary/
cp artifacts/cross_seed_L1d_N20/summary/*.csv results/cross_seed_L1d_N20/summary/
rm results/cross_seed_N20/summary/detectability_gate_provenance.csv
```

Then:

```
# H1 monotonicity, H2 validity gap and the detectability-gate record (summary tables only)
python -m scripts.run_t4_analysis --root results/cross_seed_N20 --n-seeds 20 --reports t4a t4b t4c

# PS-2 detectability gate on the chain cells (writes nothing without --out)
python -m scripts.chain_detectability_gate --root results/cross_seed_N20 --out results/cross_seed_N20/summary/chain_detectability_gate.md

# figures 1-7
python -m scripts.fig1_cost_ladder        # H1
python -m scripts.fig2_validity_gap       # H2
python -m scripts.fig3_detectability      # detectability-gate record
python -m scripts.fig4_h4_scatter         # H4
python -m scripts.fig5_regime_contrast    # H4 contrast
python -m scripts.fig6_h3_interpolation   # H3
python -m scripts.fig7_h3_interaction     # H3a
```

Each figure script pins the SHA-256 of the summary table it draws from and aborts on a mismatch.
Figures 6 and 7 additionally read `h3_instance_table_v2.csv`, which carries no pin. Compare the
regenerated files with the tracked copies under `artifacts/`.

### Rebuilding from the cell-level data

Place the three grid directories from the HU Box folder under `results/` (`cross_seed_N6`,
`cross_seed_N20`, `cross_seed_L1d_N20`). Then, per grid, the summary tables rebuild from the
cells:

```
python -m scripts.run_cross_seed_grid --root results/cross_seed_N20 --n-seeds 20 --summary-only
python -m scripts.run_cross_seed_grid --root results/cross_seed_L1d_N20 --n-seeds 20 --summary-only
```

The H3 and H4 verdicts and the prefix verification follow:

```
# H4  (PS-9)
python -m scripts.run_h4_analysis --root results/cross_seed_N20 --n-seeds 20

# H3  (PS-8), then its LD-2 re-emission
python -m scripts.run_h3_analysis --root results/cross_seed_L1d_N20 --n-seeds 20
python -m scripts.run_h3_reemission --root results/cross_seed_L1d_N20 --n-seeds 20

# prefix property (PS-1): the six-seed early look is a float-exact prefix of the 20-seed grid
python -m scripts.verify_grid_prefix --ref results/cross_seed_N6 --new results/cross_seed_N20

# PS-4 strict-subset check between the three-rung and four-rung trees
python -m scripts.verify_grid_prefix --ref results/cross_seed_N20 --new results/cross_seed_L1d_N20 --n-seeds 20 --all-topologies --kinds scoring-only
```

The H3 verdict is emitted twice by design: `h3_verdict.md` is the first-stage reading, preserved
as halted, and `h3_verdict_v2.md` is the re-emission under LD-2 that supersedes it. Some shipped
code and the gate-provenance CSV name audit records that this package does not carry; the values
they refer to travel in the shipped CSV and in the code constants.

### How the grids were generated

The grids are fixed inputs. The commands below document how they were produced and are not a
reproduction step: a fresh run is a new experiment, and the reported numbers are read from the
frozen grids in the HU Box folder.

```
# three-rung grid (L0 / L1-oracle / L2): 12 cells x 20 seeds
python -m scripts.run_cross_seed_grid --root results/cross_seed_N20 --n-seeds 20

# four-rung grid adding L1-discovered (PS-4): 12 cells x 20 seeds, separate tree
python -m scripts.run_cross_seed_grid --root results/cross_seed_L1d_N20 --n-seeds 20 --l1-discovered
```

Randomness derives from one pinned meta-entropy (20260710) through NumPy's `SeedSequence`, recorded
in every run manifest; the seed count was staged 6 -> 20 under the prefix property (PS-1). The
single-cell drivers `python -m icknowledge.recourse.run_{triangle,collider,chain}` exist for
one-cell inspection and write unconditionally into their configured output directory; do not run
them in a tree that already holds results.

## Results

Verdicts: H1 Null, H2 Supported, H3a Mixed (the firing interaction runs counter to the predicted
direction, and the report says so), H3b Null, H4 Null. Each is read from a frozen report under
`artifacts/`: `cross_seed_N20/summary/t4a_h1_monotonicity.md` (H1), `t4b_validity_gap.md` (H2),
`h4_verdict.md` (H4) and `cross_seed_L1d_N20/summary/h3_verdict_v2.md` (H3a, H3b). The figures
are in the two `figures/` directories beside them.

## Project structure

```
├── README.md
├── requirements.txt                 -- runtime dependencies, pinned to the producing environment
├── requirements-dev.txt             -- pytest, ruff, pillow
├── env_snapshots/                   -- full pip freeze of the producing environment
├── pytest.ini · ruff.toml
├── artifacts/                       -- frozen summary tables, reports, figures and captions (tracked)
├── configs/                         -- OmegaConf YAML per topology x family (+ classifier config)
├── data/                            -- empty by design: all data synthetic, sampled in-pipeline
├── results/                         -- gitignored workspace for regenerated outputs and the cell-level data
├── icknowledge/                     -- the package (used in place via python -m)
│   ├── scm/                         -- ground-truth SCMs: triangle, collider, chain x linear / NLG
│   ├── classifier/                  -- group-blind logistic regression + label calibration
│   ├── estimation/                  -- L1-oracle / L1-discovered equation estimation (supplied form)
│   ├── discovery/                   -- PC-Stable, CPDAG -> DAG completion, provenance record
│   ├── recourse/                    -- the single recourse procedure, the four knowledge conditions,
│   │                                   scoring, single-cell drivers run_{triangle,collider,chain}
│   ├── descriptor/                  -- S(g), the pre-specified structural descriptor (PS-3)
│   ├── aggregation/                 -- per-individual -> per-group / per-cell aggregates, common-found rule
│   ├── analysis/                    -- H1-H4 readers and verdict functions
│   └── utils/                       -- seeding, manifests, config
├── scripts/                         -- grid runner, prefix verifier, gates, analysis drivers, fig1-fig7
└── tests/                           -- 678 tests: construction gates, wiring gates, verdict logic, figures
```
