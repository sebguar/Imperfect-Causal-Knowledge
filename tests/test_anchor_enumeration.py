"""Structural tests for anchor.py's subset enumeration.

The G1 pulled-back anchor's argmin ranges over all non-empty subsets of the
actionable set. These assert the *set of subsets considered* — not the argmin — so
they survive coefficient / classifier changes and pin the k=2 backward-compatibility
and k=3 completeness that the enumeration fix guarantees.
"""

from __future__ import annotations

from icknowledge.recourse.anchor import nonempty_subsets


def test_k2_enumeration_is_backward_compatible():
    """k=2 (triangle): exactly the two singletons and the pair — three subsets,
    identical to the previous singleton-plus-full-joint enumeration."""
    subsets = {frozenset(s) for s in nonempty_subsets(("X1", "X2"))}
    assert subsets == {
        frozenset({"X1"}),
        frozenset({"X2"}),
        frozenset({"X1", "X2"}),
    }
    # No duplicates, count is exactly 2^2 − 1 = 3.
    assert len(nonempty_subsets(("X1", "X2"))) == 3


def test_k3_enumeration_is_complete():
    """k=3 (collider): all seven non-empty subsets — three singletons, three pairs,
    the full triple. The three pairs are what the old enumeration missed."""
    subsets = {frozenset(s) for s in nonempty_subsets(("X1", "X2", "X3"))}
    singletons = {frozenset({v}) for v in ("X1", "X2", "X3")}
    pairs = {frozenset(p) for p in (("X1", "X2"), ("X1", "X3"), ("X2", "X3"))}
    triple = {frozenset({"X1", "X2", "X3"})}
    assert subsets == singletons | pairs | triple
    assert len(nonempty_subsets(("X1", "X2", "X3"))) == 7
