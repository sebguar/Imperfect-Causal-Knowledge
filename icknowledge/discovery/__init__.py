"""Causal discovery for the L1-discovered rung (PS-7).

Two halves, both truth-free: `synthesis` translates a DISCOVERED graph into the
supplied-form contract the estimation layer already speaks, and
`pc_stable` runs the PC-Stable search and completes its CPDAG to an
oriented DAG under the tie-break rule, returning a `DiscoveryRecord` for
the manifest. Neither half ever receives a true SCM, graph, family or regime —
the only declared input is the family key, and it reaches the synthesizer only.

Estimation and SCM assembly are elsewhere (`estimation.linear`,
`estimation.discovered`); wiring into the pipeline is the L1-discovered path (PS-4).
"""

from icknowledge.discovery.pc_stable import (
    LIBRARY,
    discover_and_synthesize,
    discover_graph,
)
from icknowledge.discovery.provenance import (
    CHANNELS,
    DiscoveredEdge,
    DiscoveryRecord,
)
from icknowledge.discovery.synthesis import PROTECTED, synthesize_form_spec

__all__ = [
    "CHANNELS",
    "LIBRARY",
    "PROTECTED",
    "DiscoveredEdge",
    "DiscoveryRecord",
    "discover_and_synthesize",
    "discover_graph",
    "synthesize_form_spec",
]
