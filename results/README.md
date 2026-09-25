# results/

Populated by experiments; contents gitignored. This file documents the tree so
the layout survives in version control even though the data does not.

## Single-seed reference runs

    recourse_{triangle,triangle_nlg,collider,collider_nlg}/
      scoring_table_{regime}_{condition}.csv    # per-individual, the source of truth
      aggregates/aggregate_{by_group,by_cell,pairwise}_{regime}.csv
      manifest.json

Written by `python -m icknowledge.recourse.run_{triangle,collider} [config]`.
These are single-seed smoke runs at the pinned master seed 20260710 and are NOT
overwritten by the cross-seed grid — they remain the reference cells the
aggregation backward-compatibility gate
(`tests/test_common_found_filter.py::test_backward_compat_shipped_aggregates`)
checks against.
They are development-only and are not distributed: neither the repository nor
the data folder carries them, so the checks that read them skip on a clone.

## Cross-seed grid (PS-1 first-6 early look)

    cross_seed_N6/
      {topology}_{family}_{regime}_seed_{ii}/   # 8 cells x 6 seeds = 48
        scoring_table_{regime}_{condition}.csv
        aggregate_{by_group,by_cell,pairwise}_{regime}.csv
        manifest.json
      summary/
        cross_seed_by_group.csv                 # mean/sd/min/max across seeds, descriptive
        cross_seed_by_cell.csv
        cross_seed_pairwise.csv
        per_seed_by_{group,cell}.csv            # the raw 6, kept for the per-seed gate
        s_of_g_by_seed.csv                      # PS-3 fit-independence check
        wall_clock_log.csv                      # incl. the suspend detector
        detectability_inputs.md                 # hand-off: grid produces, analysis evaluates

Written by `python -m scripts.run_cross_seed_grid`. Seeds are the pinned
`SeedSequence(20260710).spawn(6)` children, shared across all 8 cells; the full grid
extends to N=20 with `--n-seeds 20`, under which seeds 0-5 are bit-identical.
