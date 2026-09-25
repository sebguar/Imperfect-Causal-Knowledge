"""Structural descriptors of the ground-truth SCM (PS-3) — currently S(g) only.

Deliberately depends on `icknowledge.scm` and — since the NLG family specification's branch
integrates by
Monte Carlo — the pinned meta-entropy constant in `icknowledge.utils.seeding`, and
nothing else: everything here must stay data-, classifier-, and fit-independent, which
is what makes S(g) pre-committable (PS-3). Nothing in this package may import from
`aggregation`, `classifier`, `estimation`, or `recourse`.
"""

from icknowledge.descriptor.s_of_g import (
    NLG_NOT_IMPLEMENTED,
    S_OF_G_MC_SAMPLES,
    S_OF_G_MC_SEED,
    SG_MC_STREAM_LABEL,
    s_of_g,
)

__all__ = [
    "s_of_g",
    "NLG_NOT_IMPLEMENTED",
    "S_OF_G_MC_SAMPLES",
    "S_OF_G_MC_SEED",
    "SG_MC_STREAM_LABEL",
]
