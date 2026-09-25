# data/

All experiment data is synthetic. Every run samples its own dataset in memory from
the ground-truth SCM (`icknowledge/scm/`) and never persists it here; nothing in
the pipeline writes to `data/`. The seeds that reproduce each run's sample are
recorded in that run's `manifest.json` (`seeding` block: pinned meta-entropy
20260710 plus the run's master seed; on grid runs, the full spawned seed list).
Contents of this directory are gitignored.
