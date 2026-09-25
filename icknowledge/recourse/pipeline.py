"""End-to-end recourse pipeline, topology- and family-generic.

Wires SCM -> group-blind classifier h -> equation estimation ->
action space -> the single brute-force procedure under the requested knowledge
conditions (L0, L2, L1-oracle, L1-discovered) -> per-individual scoring against
the true SCM -> the G1 anchor smoke check. Runs the additive and effect-modifying
regimes; the topology comes in as (scm_builder, form_spec_fn) and the functional
family as the ``family`` argument.

L1-discovered (PS-7, PS-4) is wired but LAZY-GATED: its PC-Stable
search, its second OLS fit and its SCM assembly execute only when the caller asks
for the rung by name. The DEFAULT condition set is unchanged at three rungs, so
every three-rung caller — the three topology drivers, the cross-seed grid without
``--l1-discovered``, every existing test — produces byte-identical output.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from omegaconf import DictConfig

from icknowledge.classifier import ClassifierResult, build_classifier
from icknowledge.discovery import DiscoveryRecord, discover_and_synthesize
from icknowledge.estimation import (
    FittedEquations,
    LinearOLSEstimator,
    NLGOLSEstimator,
    build_estimated_scm,
    build_estimated_scm_from_discovery,
    dataset_identity,
)
from icknowledge.recourse.action_space import build_action_space
from icknowledge.recourse.anchor import (
    AnchorReport,
    anchor_smoke_check,
    grid_diagonal_step,
    grid_span_limit,
)
from icknowledge.recourse.model_conditions import (
    L0AssociationalModel,
    L1DiscoveredEstimatedSCMModel,
    L1OracleEstimatedSCMModel,
    L2TrueSCMModel,
)
from icknowledge.recourse.procedure import generate_recourse
from icknowledge.recourse.scoring import score_condition, summarize
from icknowledge.scm import make_linear_triangle, triangle_form_spec

# run_regime is parametrized by (scm_builder, form_spec_fn) so every topology
# reuses the SAME pipeline function rather than duplicating it. The defaults keep
# the triangle callers (tests, run_triangle) unchanged. `run_anchor` gates the
# anchor.py check: its closed form enumerates all 2^k − 1 acted subsets and holds
# on the LINEAR family only, so the drivers pass run_anchor = (family == "linear").

# run_regime conditions default is the FULL wired ladder: "run" means the full
# ladder — the aggregation layer consumes all three conditions. The legacy
# test that asserted a two-condition world (test_recourse_triangle.py) passes
# its old ("L0", "L2") tuple explicitly rather than relying on this default.
_DEFAULT_CONDITIONS = ("L0", "L1-oracle", "L2")

# Estimator selection is by the FAMILY key the caller/config supplies,
# NOT by an attribute of the SCM object. [the L1-oracle form-template-known semantics; the NLG
# family specification] The `SCM` class
# carries no `family` field, and this pipeline deliberately does not add
# one: the estimator must be paired with the FORM TEMPLATE, and the template is a
# supplied input, not something read off the ground truth. Keying both off
# one config field keeps "which template was fitted by which estimator" a single
# auditable choice recorded in the manifest.
_ESTIMATORS = {"linear": LinearOLSEstimator, "nlg": NLGOLSEstimator}


@dataclass
class RegimeRun:
    """Everything one regime's end-to-end run produces (the deliverable + checks)."""

    regime: str
    family: str
    classifier: ClassifierResult
    table: pd.DataFrame  # per-individual x condition scoring table (the deliverable)
    summary: pd.DataFrame  # validity-rate + realized-cost by group x condition
    anchor: AnchorReport | None  # None when run_anchor=False (e.g. collider; see run_regime)
    fitted: FittedEquations  # estimation output (manifest provenance source)
    # -- [PS-4] LAZY: populated only when "L1-discovered" is scored. -----------
    # `fitted` stays SINGULAR and keeps meaning "the L1-oracle estimation record" —
    # its four consumers (the three topology drivers + run_cross_seed_grid) are
    # unchanged by the four-rung extension, so a three-rung run's manifest
    # is byte-identical.
    #: PS-7 provenance for the PC-Stable run; None on a three-rung run.
    discovery: DiscoveryRecord | None = None
    #: The SECOND estimator fit — same the L1-oracle form-template-known semantics estimator,
    #: synthesized template, same
    #: sample. Distinct from `fitted`: that one is fitted against the TRUE graph's
    #: supplied template, this one against the DISCOVERED graph's. None off the rung.
    fitted_discovered: FittedEquations | None = None


def run_regime(
    cfg: DictConfig,
    regime: str,
    conditions: Sequence[str] = _DEFAULT_CONDITIONS,
    *,
    scm_builder=make_linear_triangle,
    form_spec_fn=triangle_form_spec,
    run_anchor: bool = True,
    family: str = "linear",
) -> RegimeRun:
    """Run the full pipeline for one SCM regime under ``conditions``.

    ``conditions`` is a subset of {"L0", "L2", "L1-oracle", "L1-discovered"}; L0
    and L2 are always required, and "L1-discovered" additionally triggers the
    lazy discovery+estimation path (see the PS-4 block below).
    ``scm_builder``/``form_spec_fn`` select the topology × family
    (default: linear triangle); ``family`` selects the L1-oracle equation
    estimator and is recorded in the fit provenance; ``run_anchor`` gates the
    anchor.py L2 check (see the module note — disabled for every nonlinear-Gaussian
    cell, whose closed form does not exist).
    """
    rcfg = cfg.recourse
    if family not in _ESTIMATORS:
        raise ValueError(
            f"unknown functional family {family!r}; wired: {sorted(_ESTIMATORS)}."
        )
    if run_anchor and family != "linear":
        # [G1; the ℓ₂ intervention-cost convention] The Ehyaei closed form the anchor pulls
        # back assumes a
        # LINEAR SCM and a linear classifier, so it is not merely loose on a
        # nonlinear-Gaussian cell — it is inapplicable. Refuse loudly rather than
        # report a meaningless mismatch as a smoke-check failure.
        raise ValueError(
            f"the G1 anchor smoke check is defined only for family='linear'; got "
            f"{family!r}. Pass run_anchor=False (the Ehyaei closed form assumes a "
            "linear SCM and a linear classifier)."
        )
    scm = scm_builder(regime)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # near-unregularized LR convergence noise
        clf = build_classifier(scm, cfg)
    data = clf.dataset

    # -- Estimate structural equations. -----------------------------------------
    # [the L1-oracle form-template-known semantics] The estimation sample IS the
    # classifier's training
    # sample — the exact dataset object used for fitting h, plumbed through as a
    # first-class object with NO redraw and NO split (its identity/hash is
    # recorded in the fit metadata, so the manifest lets an auditor verify that
    # the L1-discovered PC step later runs on THIS SAME sample).
    training_sample = data
    form_spec = form_spec_fn(regime)  # form template SUPPLIED, not inferred
    # For family="nlg" this is `NLGOLSEstimator` — the SAME
    # unregularized-OLS fit, on the template's fixed-tanh design columns. No
    # nonlinear-least-squares path exists here by design (see
    # estimation/nonlinear_gaussian.py).
    fitted = _ESTIMATORS[family]().fit(
        training_sample,
        form_spec,
        extra_metadata={
            "regime": regime,
            "family": family,
            "master_seed": int(cfg.seed),
            "sample": "classifier-training-sample (ClassifierResult.dataset)",
        },
    )
    # [the raw-units convention] Nothing here standardizes: the estimated coefficients live in
    # the same raw feature units as L0's/L2's actions and the classifier weights.
    estimated_scm = build_estimated_scm(scm, fitted)

    action_space = build_action_space(
        data,
        list(rcfg.actionable),
        mode=str(rcfg.grid.mode),
        k=float(rcfg.grid.k),
        resolution=int(rcfg.grid.resolution),
    )

    # -- Instantiate the knowledge conditions. ----------------------------------
    # The true SCM is ALWAYS the realizer/scorer for EVERY condition. The
    # same L2 model object is the L2 causal model AND the scoring realizer.
    true_model = L2TrueSCMModel(scm, clf.feature_names, action_space.acted)
    l0_model = L0AssociationalModel(clf.feature_names, action_space.acted)
    l1_model = L1OracleEstimatedSCMModel(estimated_scm, clf.feature_names, action_space.acted)

    models = {"L0": l0_model, "L2": true_model, "L1-oracle": l1_model}

    # -- L1-discovered, LAZY-GATED. [PS-7, PS-4] --------------------------------
    # Everything below runs ONLY when the rung is actually scored. That is not an
    # optimization: PC-Stable + a second OLS fit are the whole cost of the rung,
    # and a three-rung run must be bit-identical to the sealed three-rung output — no extra
    # RNG draw, no extra fit, no new manifest key. The centerpiece regression gate
    # (tests/test_l1_discovered_wiring.py) runs the SAME config+seed both ways and
    # asserts the three shared scoring tables are float-exact identical.
    discovery_record: DiscoveryRecord | None = None
    fitted_discovered: FittedEquations | None = None
    if "L1-discovered" in conditions:
        # Truth-free by signature: the search sees the estimation sample and
        # nothing else; `family` is a DECLARED cell property and reaches only the
        # form synthesizer (PS-7; discovery/pc_stable.py module note).
        discovered_graph, synthesized_spec, discovery_record = discover_and_synthesize(
            data, family
        )
        # [the L1-oracle form-template-known semantics] THE shared-sample assertion: the PC
        # step must run on the
        # SAME sample that fitted the equations and trained h. Checked, not trusted —
        # it is the one constraint that makes ΔCost_ld attributable to graph
        # provenance rather than to a sample-composition difference.
        #
        # WHY THE HASHES ARE COMPARABLE AT ALL — the note (g) coincidence.
        # `discover_graph` hashes `data.reindex(columns=canonical_order)`, i.e. the
        # frame in CANONICAL LEXICAL order (A < X1 < X2 < X3), while the estimation
        # side hashes `training_sample` in the SCM's topological column order. The
        # two digests agree only because those orders coincide on all three topologies
        # — a pre-stated structural property, not an accident to be relied on
        # silently. This is one of three load-bearing sites of that coincidence; the
        # other two are (i) the CPDAG tie-break, whose lower→higher direction
        # is read off the same lexical order (discovery/pc_stable.py `_orient_cpdag`),
        # and (ii) the canonical-order pin itself in `discovery.pc_stable.discover_graph`.
        # All three are named here so they are mutually discoverable: a topology whose
        # topological order stops being lexical breaks this assertion FIRST, loudly,
        # before it can quietly move a tie-break.
        estimation_dataset = dataset_identity(training_sample)
        if discovery_record.dataset != estimation_dataset:
            raise ValueError(
                "[the L1-oracle form-template-known semantics] the L1-discovered PC step did "
                "not run on the "
                "estimation sample: discovery dataset identity "
                f"{discovery_record.dataset} != estimation dataset identity "
                f"{estimation_dataset}. L1-oracle and L1-discovered MUST share one "
                "sample, or ΔCost_ld confounds graph provenance with sample "
                "composition. (If the column lists differ, see the note (g) "
                "canonical-ordering comment at this call site.)"
            )
        # The SAME the L1-oracle form-template-known semantics estimator, on the SAME sample, under
        # a template SYNTHESIZED
        # from the discovered graph (the form-preserving estimation basis; no
        # A·X column in either regime). Only the GRAPH differs from
        # the L1-oracle fit.
        fitted_discovered = _ESTIMATORS[family]().fit(
            training_sample,
            synthesized_spec,
            extra_metadata={
                "regime": regime,
                "family": family,
                "master_seed": int(cfg.seed),
                "sample": "same-as-estimation (the L1-oracle form-template-known semantics)",
            },
        )
        discovered_scm = build_estimated_scm_from_discovery(
            discovered_graph, fitted_discovered, training_sample
        )
        models["L1-discovered"] = L1DiscoveredEstimatedSCMModel(
            discovered_scm, clf.feature_names, action_space.acted
        )

    unknown = [c for c in conditions if c not in models]
    if unknown:
        raise ValueError(f"unknown condition(s) {unknown}; wired: {sorted(models)}.")
    if not {"L0", "L2"} <= set(conditions):
        raise ValueError("conditions must include L0 and L2 (G1 anchor smoke check).")

    # ONE procedure, N causal models (model-agnostic contract).
    solutions = {c: generate_recourse(clf, models[c], action_space, data) for c in conditions}

    # Scoring: realized ALWAYS via true_model; believed via the condition-held model.
    tables = [
        score_condition(clf, solutions[c], true_model, models[c], data, c)
        for c in conditions
    ]
    table = pd.concat(tables, ignore_index=True)
    summary = summarize(table, protected=clf.protected_attr)

    report = (
        anchor_smoke_check(
            clf.model,
            true_model,
            solutions["L2"],
            solutions["L0"],
            data,
            grid_diagonal_step(action_space.axis_grids),
            grid_span_limit(action_space.axis_grids),
            tolerance=float(rcfg.anchor.tolerance),
            n_sample=int(rcfg.anchor.n_sample),
            seed=int(cfg.seed),
        )
        if run_anchor
        else None
    )

    return RegimeRun(
        regime=regime,
        family=family,
        classifier=clf,
        table=table,
        summary=summary,
        anchor=report,
        fitted=fitted,
        discovery=discovery_record,
        fitted_discovered=fitted_discovered,
    )


def format_regime_summary(run: RegimeRun) -> str:
    """Short human-readable summary for one regime (the printed deliverable digest)."""
    n_individuals = len(run.classifier.negative_pool_indices)
    lines = [
        f"=== {run.family} SCM [{run.regime}] ===",
        f"negatively-classified individuals (recourse pool): {n_individuals}",
        "validity-rate + realized-cost by group x condition (realized cost over VALID actions):",
        run.summary.to_string(index=False),
        run.anchor.format()
        if run.anchor is not None
        else "G1 anchor smoke check: skipped (run_anchor=False)",
    ]
    return "\n".join(lines)
