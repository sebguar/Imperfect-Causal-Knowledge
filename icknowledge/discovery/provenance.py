"""Provenance record for one PC-Stable discovery run.

Everything PS-7 says must be recorded rather than inherited — library,
version, the *resolved* argument dict, the canonical ordering, the A-encoding
transform, the per-edge orientation channel and the bidirected-edge count — is
carried on ONE frozen object with a JSON-safe `as_dict()`. The manifest writer
(PS-4) consumes that dict verbatim; nothing here writes a file.

TRUTH-FREE. No field on this record is derived from a true SCM, a true graph or
a functional family. A record is a description of what the SEARCH did, not of
how well it did — comparing a discovered graph against the truth (skeleton-SHD,
orientation-error count; PS-8's diagnostic covariates) happens at the scoring
seam in the scoring layer, where the truth legitimately lives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["CHANNELS", "DiscoveredEdge", "DiscoveryRecord"]

#: The three orientation channels an edge in the returned DAG can carry.
#:
#: ``"ci"``       — direction supplied by conditional-independence testing:
#:                  an unshielded-collider orientation or a Meek propagation of
#:                  one. [PS-7] Per PS-8's structure-conditioned
#:                  reading, the collider is the only one of the three graphs where
#:                  this channel can fire at all.
#: ``"bk"``       — direction fixed by the PS-7 background knowledge
#:                  ("A is a root"). Every edge OUT of A carries this label, by
#:                  the attribution rule in `pc_stable._orient_cpdag`.
#: ``"tiebreak"`` — direction supplied by the deterministic tie-break
#:                  (lower → higher in the canonical order), i.e. the edge was
#:                  Markov-ambiguous after CI testing and Meek closure.
CHANNELS = ("ci", "bk", "tiebreak")


@dataclass(frozen=True)
class DiscoveredEdge:
    """One oriented edge of the returned DAG, with its orientation provenance.

    ``was_bidirected`` records that PC returned this adjacency as ``i ↔ j``.
    [PS-7 note (d)] Under causal sufficiency by
    construction, a bidirected endpoint pair cannot mean latent confounding, so
    it is read as finite-sample collider-orientation conflict, demoted to
    undirected, and resolved by the tie-break — the flag is what keeps that
    demotion visible in the manifest instead of silent.
    """

    source: str
    target: str
    channel: Literal["ci", "bk", "tiebreak"]
    was_bidirected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "channel": self.channel,
            "was_bidirected": self.was_bidirected,
        }


@dataclass(frozen=True)
class DiscoveryRecord:
    """Everything the manifest needs to reconstruct one discovery run (PS-7).

    ``library`` / ``library_version`` — distribution name and installed version.
    ``resolved_args`` — the explicit keyword dict handed to `causallearn…PC.pc`,
        with `BackgroundKnowledge` replaced by its forbidden-edge list; ``data``
        is not a member, the sample is recorded under ``dataset``.
    ``canonical_order`` — lexical node order (PS-7 note (g)), recorded not
        recomputed: the tie-break direction and PC's column order define against it.
    ``encoding_note`` — the ±1 → 0/1 A-transform applied for the CI test only, and
        its inertness: Fisher-z partial correlation is affine-invariant, so no CI
        decision moves.
    ``edges`` — every directed edge of the returned DAG, with its channel.
    ``n_bidirected`` — adjacencies PC returned as ``i ↔ j`` (note (d)).
    ``adjacency`` — ``adjacency[i][j] == 1`` iff ``order[i] → order[j]``; the shape
        PS-8's skeleton-SHD / orientation-error covariates are computed from.
    ``dataset`` — estimation-sample identity, ``{n_rows, columns, sha256}``, the
        same shape the estimation block records, so the seam asserts hash equality.
    """

    library: str
    library_version: str
    resolved_args: dict[str, Any]
    canonical_order: tuple[str, ...]
    encoding_note: str
    edges: tuple[DiscoveredEdge, ...]
    n_bidirected: int
    adjacency: tuple[tuple[int, ...], ...]
    dataset: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe view — plain dicts, lists, str, int, bool throughout."""
        return {
            "library": self.library,
            "library_version": self.library_version,
            "resolved_args": self.resolved_args,
            "canonical_order": list(self.canonical_order),
            "encoding_note": self.encoding_note,
            "edges": [edge.as_dict() for edge in self.edges],
            "n_bidirected": self.n_bidirected,
            "adjacency": [list(row) for row in self.adjacency],
            "dataset": dict(self.dataset),
        }
