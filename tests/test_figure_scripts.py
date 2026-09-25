"""Smoke and contract tests for the seven thesis figure scripts.

The scripts are pure renderers over the frozen ``summary/*.csv`` artifacts, so
they are exercised here against hand-built summary trees in ``tmp_path``. No real
results tree is touched: the suite must stay green on a clean checkout, where
results/ is gitignored and absent (same contract as ``test_t4_reports``).

Nothing is asserted pixel-by-pixel — that is brittle and tests the renderer, not
the deliverable. What IS asserted is the house set: each script runs to completion
and emits a structurally valid PDF, PNG and caption; it refuses a short seed axis
and a tampered frozen input; it draws no SE or CI and SAYS so; and every number in
its caption is recomputed from the data rather than typed into the prose.

fig3 additionally carries the two-populations separation: the gate it
annotates comes from the frozen provenance record and must NOT equal what the
20-seed strip it plots would produce.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from icknowledge.analysis.loading import CONDITIONS, N_SEEDS, N_SEEDS_FULL, grid_cells
from scripts import (
    fig1_cost_ladder,
    fig2_validity_gap,
    fig3_detectability,
    fig4_h4_scatter,
    fig5_regime_contrast,
    fig6_h3_interpolation,
    fig7_h3_interaction,
)

GROUPS = (-1.0, 1.0)


def _by_group(n_seeds: int = N_SEEDS) -> pd.DataFrame:
    """12 cells x 3 conditions x 2 groups x n_seeds, with a ladder and a collapse.

    Values are arbitrary but structured: costs descend along the ladder, and the
    collider cells carry a two-band validity pattern so Figure 2's caption logic
    (collapse vs intermediate band) is exercised on both branches.
    """
    rows = []
    for cell in grid_cells():
        collider = cell.topology == "collider"
        for condition_index, condition in enumerate(CONDITIONS):
            for group in GROUPS:
                for seed in range(n_seeds):
                    cost = (
                        (4.0 if group < 0 else 0.8)
                        - 0.1 * condition_index
                        + 0.01 * seed
                        + (1.5 if collider else 0.0)
                    )
                    if not collider:
                        validity = 1.0
                    elif cell.family == "nlg":
                        validity = 1.0 if seed % 2 else 0.02  # collapse band
                    else:
                        validity = 1.0 if seed % 2 else 0.55  # intermediate band
                    rows.append(
                        {
                            "topology": cell.topology,
                            "family": cell.family,
                            "regime": cell.regime,
                            "condition": condition,
                            "group": group,
                            "seed_idx": seed,
                            "realized_mean_cost": cost,
                            "realized_cost_common_found_threeway": cost,
                            "realized_validity_rate": validity,
                        }
                    )
    return pd.DataFrame(rows)


def _by_cell(n_seeds: int = N_SEEDS) -> pd.DataFrame:
    rows = []
    for cell in grid_cells():
        for condition_index, condition in enumerate(CONDITIONS):
            for seed in range(n_seeds):
                delta = 3.2 - 0.1 * condition_index + 0.02 * seed
                rows.append(
                    {
                        "topology": cell.topology,
                        "family": cell.family,
                        "regime": cell.regime,
                        "condition": condition,
                        "seed_idx": seed,
                        "Δ_cost": delta,
                        "Δ_cost_common_found_threeway": delta,
                    }
                )
    return pd.DataFrame(rows)


def _assert_pdf(path: Path) -> None:
    assert path.exists(), f"{path} was not written"
    blob = path.read_bytes()
    assert len(blob) > 1024, f"{path} is implausibly small ({len(blob)} bytes)"
    assert blob.startswith(b"%PDF-"), f"{path} lacks a PDF header"
    assert blob.rstrip().endswith(b"%%EOF"), f"{path} is a truncated PDF"


def _assert_png(path: Path) -> None:
    assert path.exists(), f"{path} was not written"
    assert path.stat().st_size > 1024
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        assert image.format == "PNG"
        # >= 200 dpi is the review-copy requirement in the figure-rendering spec.
        assert round(image.info["dpi"][0]) >= 200
        assert min(image.size) > 200


def _assert_caption(path: Path, must_contain: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"{path} is empty"
    # Captions are reflowed to a fixed width, so a fragment assertion must not
    # depend on where the wrap happened to land.
    flowed = " ".join(text.split())
    for fragment in must_contain:
        assert fragment in flowed, f"{path} does not state {fragment!r}"


# --------------------------------------------------------------------------- #
# Confirmatory figures — fig1 / fig2 / fig3, at N=20 on the full 12-cell grid.
# Render-only, from the sealed cross_seed_N20 artifacts. No gate is re-evaluated.
# --------------------------------------------------------------------------- #

#: The three scripts and the frozen-input constant each one guards.
CONFIRMATORY_FIGURES = (
    (fig1_cost_ladder, "fig1_cost_ladder", ("BY_GROUP_SHA256",)),
    (fig2_validity_gap, "fig2_validity_gap", ("BY_GROUP_SHA256",)),
    (fig3_detectability, "fig3_detectability", ("BY_CELL_SHA256", "GATE_PROVENANCE_SHA256")),
)


def _gate_provenance(n_seeds: int = 6) -> pd.DataFrame:
    """A frozen-gate provenance record whose ratios CANNOT come from the strip.

    The means are deliberately offset from the fixture strip's own mean, so
    `assert_gate_is_not_derived` is exercised on a record that is genuinely a
    different population — which is what the real seed-0..5 record is.
    """
    rows = []
    for cell in grid_cells():
        mean, sd = 4.0, 0.11
        rows.append(
            {
                "topology": cell.topology,
                "family": cell.family,
                "regime": cell.regime,
                "n_seeds_evaluated_for_gate": n_seeds,
                "mean": mean,
                "sd": sd,
                "abs_mean_over_sd": abs(mean) / sd,
                "cond_i_sign_consistent": True,
                "cond_i_magnitude": True,
                "cond_i_pass": True,
                "delta_S": 0.0 if cell.regime == "additive" else -0.4,
                "delta_S_identical_across_seeds": True,
                "cond_ii_pass": True,
                "cond_ii_note": "test fixture",
                "gate_pass": True,
                "source_record": "tests/fixture",
            }
        )
    return pd.DataFrame(rows)


def _pin_confirmatory_digests(monkeypatch, root: Path) -> None:
    """Point each script's frozen-input guard at the fixture's own bytes.

    The guard's whole job is to refuse anything that is not the registered
    artifact, so a synthetic tree can only be rendered by substituting the
    expectation — which is itself the assertion that the guard is load-bearing.
    """
    digests = {
        "BY_GROUP_SHA256": "per_seed_by_group.csv",
        "BY_CELL_SHA256": "per_seed_by_cell.csv",
        "GATE_PROVENANCE_SHA256": "detectability_gate_provenance.csv",
    }
    for module, _, constants in CONFIRMATORY_FIGURES:
        for constant in constants:
            path = root / "summary" / digests[constant]
            monkeypatch.setattr(
                module, constant, hashlib.sha256(path.read_bytes()).hexdigest()
            )


@pytest.fixture()
def n20_root(tmp_path, monkeypatch) -> Path:
    """A 12-cell x 20-seed tree plus the frozen gate record fig3 reads."""
    summary = tmp_path / "summary"
    summary.mkdir(parents=True)
    _by_group(N_SEEDS_FULL).to_csv(
        summary / "per_seed_by_group.csv", index=False, encoding="utf-8"
    )
    _by_cell(N_SEEDS_FULL).to_csv(
        summary / "per_seed_by_cell.csv", index=False, encoding="utf-8"
    )
    _gate_provenance().to_csv(
        summary / "detectability_gate_provenance.csv", index=False, encoding="utf-8"
    )
    _pin_confirmatory_digests(monkeypatch, tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("module", "stem", "caption_fragments"),
    [
        (
            fig1_cost_ladder,
            "fig1_cost_ladder",
            ["PS-8", "pattern-only", "common-found", "threeway", "PS-1",
             "FIRST APPEARANCE"],
        ),
        (
            fig2_validity_gap,
            "fig2_validity_gap",
            ["PS-8", "pattern-only", "PS-1", "LD-3", "FIRST APPEARANCE"],
        ),
        (
            fig3_detectability,
            "fig3_detectability",
            [
                "PS-8", "pattern-only", "PS-2", "PS-1",
                "collider (the SCM specification)",
                "chain (the chain SCM specification)",
                "detectability_gate_provenance.csv", "seeds 0-5",
            ],
        ),
    ],
    ids=["fig1", "fig2", "fig3"],
)
def test_script_emits_pdf_png_and_caption(n20_root, module, stem, caption_fragments):
    out = n20_root / "figures"
    assert module.main(["--root", str(n20_root), "--out", str(out)]) == 0
    _assert_pdf(out / f"{stem}.pdf")
    _assert_png(out / f"{stem}.png")
    _assert_caption(out / f"{stem}.caption.txt", caption_fragments)


@pytest.mark.parametrize(
    ("module", "stem"),
    [(module, stem) for module, stem, _ in CONFIRMATORY_FIGURES],
    ids=["fig1", "fig2", "fig3"],
)
def test_confirmatory_captions_carry_no_stale_early_look_strings(
    n20_root, module, stem
):
    """The stale N=6 caption claims must be absent: the grid is 20 seeds, 12 cells.

    Targets the stale CLAIMS, not every mention of the early look — fig2
    contrasts what the N=6 look showed against what N=20 shows, and that
    contrast is legitimate.
    """
    assert module.main(["--root", str(n20_root)]) == 0
    text = (n20_root / "figures" / f"{stem}.caption.txt").read_text(encoding="utf-8")
    flowed = " ".join(text.split())
    for stale in (
        "N = 6 seeds",
        "eight cells",
        "six per-seed",
        "the six seeds",
    ):
        assert stale not in flowed, f"{stem} still states {stale!r}"
    # No week reference of any form survives in a caption.
    week_reference = re.compile(r"\bW[0-9]{1,2}\b|\bweeks?[ \-]?[0-9]", re.IGNORECASE)
    found = week_reference.search(flowed)
    assert found is None, f"{stem} caption carries a week reference: {found.group(0)!r}"


@pytest.mark.parametrize("module", [m for m, _, _ in CONFIRMATORY_FIGURES],
                         ids=["fig1", "fig2", "fig3"])
def test_confirmatory_figures_compute_no_standard_error(module):
    """PS-9: SD is descriptive; SE and CI are not displayed.

    Asserted on the SOURCE, because the claim is about what the figure is allowed
    to ENCODE, not about pixels: an SE band would need a sqrt(n) divisor or a
    `sem` call, and both are checked for.
    """
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert ".sem(" not in source, f"{module.__name__} computes a standard error"
    assert "np.sqrt(" not in source, f"{module.__name__} has a sqrt(n) divisor"


@pytest.mark.parametrize(
    ("module", "stem"),
    [(module, stem) for module, stem, _ in CONFIRMATORY_FIGURES],
    ids=["fig1", "fig2", "fig3"],
)
def test_confirmatory_captions_state_the_no_confidence_interval_policy(
    n20_root, module, stem
):
    """The captions must SAY that no SE/CI is shown — a reader cannot infer it."""
    assert module.main(["--root", str(n20_root)]) == 0
    text = (n20_root / "figures" / f"{stem}.caption.txt").read_text(encoding="utf-8")
    lowered = " ".join(text.lower().split())
    assert "no standard errors or confidence intervals" in lowered
    assert "PS-9" in text


@pytest.mark.parametrize("module", [m for m, _, _ in CONFIRMATORY_FIGURES],
                         ids=["fig1", "fig2", "fig3"])
def test_confirmatory_figures_reject_a_short_seed_axis(n20_root, monkeypatch, module):
    """A partially-written grid fails loudly rather than being plotted silently."""
    for name in ("per_seed_by_group", "per_seed_by_cell"):
        path = n20_root / "summary" / f"{name}.csv"
        frame = pd.read_csv(path)
        frame[frame["seed_idx"] < N_SEEDS_FULL - 1].to_csv(
            path, index=False, encoding="utf-8"
        )
    _pin_confirmatory_digests(monkeypatch, n20_root)
    with pytest.raises(ValueError, match="seeds"):
        module.main(["--root", str(n20_root), "--out", str(n20_root / "figures")])


@pytest.mark.parametrize("module", [m for m, _, _ in CONFIRMATORY_FIGURES],
                         ids=["fig1", "fig2", "fig3"])
def test_confirmatory_figures_abort_on_a_tampered_frozen_input(n20_root, module):
    """The input-digest guard is the point: a modified sealed artifact never renders."""
    name = (
        "per_seed_by_cell.csv"
        if module is fig3_detectability
        else "per_seed_by_group.csv"
    )
    path = n20_root / "summary" / name
    frame = pd.read_csv(path)
    column = "Δ_cost" if module is fig3_detectability else "realized_validity_rate"
    frame.loc[0, column] += 0.5
    frame.to_csv(path, index=False, encoding="utf-8")
    with pytest.raises(RuntimeError, match="DIGEST MISMATCH"):
        module.main(["--root", str(n20_root), "--out", str(n20_root / "figures")])


def test_fig1_default_out_is_the_root_figures_dir(n20_root):
    assert fig1_cost_ladder.main(["--root", str(n20_root)]) == 0
    _assert_pdf(n20_root / "figures" / "fig1_cost_ladder.pdf")


def test_fig1_common_found_cost_equality_is_recomputed_not_asserted():
    """The caption's equality claim must come from the data it describes."""
    frame = _by_group(N_SEEDS_FULL)
    assert fig1_cost_ladder.common_found_cost_equality(frame) == pytest.approx(0.0)
    frame.loc[0, "realized_mean_cost"] += 0.25
    assert fig1_cost_ladder.common_found_cost_equality(frame) == pytest.approx(0.25)


def test_fig1_ylim_is_free_per_topology():
    """PS-8 encoding: every topology is scaled from its OWN data, all three of them.

    Asserted as INDEPENDENCE rather than as "no two ranges coincide": two
    topologies with the same spread may legitimately share a range, and a test
    that forbade it would be asserting a property PS-8 never claimed. What PS-8
    forbids is one topology's values setting another's axis — so move one
    topology's data and require that only its own limits follow.
    """
    by_group = _by_group(N_SEEDS_FULL)
    panels = {
        cell: fig1_cost_ladder._panel_data(by_group, cell) for cell in grid_cells()
    }
    limits = fig1_cost_ladder._ylim_by_topology(panels)
    assert set(limits) == {"triangle", "collider", "chain"}

    for moved in ("triangle", "collider", "chain"):
        shifted = {
            cell: {
                group: {
                    condition: values + (50.0 if cell.topology == moved else 0.0)
                    for condition, values in per_condition.items()
                }
                for group, per_condition in data.items()
            }
            for cell, data in panels.items()
        }
        after = fig1_cost_ladder._ylim_by_topology(shifted)
        assert after[moved] != limits[moved], f"{moved}'s own range did not follow it"
        for other in set(limits) - {moved}:
            assert after[other] == limits[other], (
                f"{moved}'s values moved {other}'s axis — that is the cross-topology "
                "scaling PS-8 forbids"
            )


def test_fig1_covers_every_cell_of_the_full_grid():
    """A layout pinned to 8 cells would silently drop four."""
    assert len(fig1_cost_ladder.ROWS) * len(fig1_cost_ladder.COLUMNS) == len(grid_cells())


def test_fig2_narrative_numbers_are_recomputed_not_asserted():
    """Every count in fig2's narrative moves when the data moves."""
    data = fig2_validity_gap.validity_by_cell(_by_group(N_SEEDS_FULL))
    before = fig2_validity_gap.caption_facts(data)
    cell = next(iter(data))
    data[cell] = np.zeros_like(data[cell])  # drive one whole cell into collapse
    after = fig2_validity_gap.caption_facts(data)
    assert after["collapse_total"] > before["collapse_total"]
    assert after["below_high_total"] > before["below_high_total"]


def test_fig2_topology_dividers_are_computed_not_hardcoded():
    """Two dividers for three topologies, each at the block boundary."""
    cells = list(fig2_validity_gap.validity_by_cell(_by_group(N_SEEDS_FULL)))
    boundaries = fig2_validity_gap._topology_boundaries(cells)
    assert len(boundaries) == 2
    for boundary in boundaries:
        below = cells[int(boundary - 0.5)]
        above = cells[int(boundary + 0.5)]
        assert below.topology != above.topology


def test_fig2_draws_no_mean_or_sd_glyph():
    """The ValidityDisp read's judgment survives N=20: the moments misdescribe, so
    none is drawn."""
    source = Path(fig2_validity_gap.__file__).read_text(encoding="utf-8")
    assert "errorbar(" not in source
    assert ".std(" not in source
    assert ".mean(" not in source


def test_fig3_rejects_a_broken_no_op(n20_root):
    """If common-found ever stops coinciding with Δ_cost, the figure must stop."""
    by_cell = _by_cell(N_SEEDS_FULL)
    target = by_cell.index[by_cell["condition"] == "L2"][0]
    by_cell.loc[target, "Δ_cost_common_found_threeway"] += 0.5
    with pytest.raises(ValueError, match="no-op no longer holds"):
        fig3_detectability.strip_series(by_cell, grid_cells()[0])


def test_fig3_gate_values_come_from_the_provenance_record(n20_root):
    """Every displayed gate quantity traces to the CSV, not to the plotted strip."""
    gate = fig3_detectability.load_gate(n20_root)
    by_cell = _by_cell(N_SEEDS_FULL)
    for cell in grid_cells():
        row = fig3_detectability.gate_row(gate, cell)
        strip = fig3_detectability.strip_series(by_cell, cell)
        assert int(row["n_seeds_evaluated_for_gate"]) == 6
        # The displayed ratio is the record's, and it is NOT the strip's.
        strip_ratio = abs(float(strip.mean())) / float(strip.std(ddof=1))
        assert float(row["abs_mean_over_sd"]) != pytest.approx(strip_ratio)


def test_fig3_refuses_a_gate_recomputed_from_the_twenty_seed_strip(n20_root):
    """the chain SCM specification's failure mode, asserted rather than trusted.

    A provenance record whose ratios ARE the strip's is exactly what a silently
    re-evaluated gate looks like, and it must abort the render.
    """
    by_cell = _by_cell(N_SEEDS_FULL)
    gate = fig3_detectability.load_gate(n20_root)
    for index, cell in enumerate(grid_cells()):
        values = fig3_detectability.strip_series(by_cell, cell)
        gate.loc[index, "mean"] = float(values.mean())
        gate.loc[index, "sd"] = float(values.std(ddof=1))
        gate.loc[index, "abs_mean_over_sd"] = abs(float(values.mean())) / float(
            values.std(ddof=1)
        )
    with pytest.raises(ValueError, match="equals the ratio the 20-seed strip"):
        fig3_detectability.assert_gate_is_not_derived(gate, by_cell)


def test_fig3_refuses_a_gate_stamped_with_the_wrong_seed_count(n20_root):
    """A record stamped 20 is a re-evaluated gate wearing the frozen record's name."""
    gate = fig3_detectability.load_gate(n20_root)
    gate["n_seeds_evaluated_for_gate"] = N_SEEDS_FULL
    with pytest.raises(ValueError, match="FIRST-6-SEED"):
        fig3_detectability.assert_gate_is_not_derived(gate, _by_cell(N_SEEDS_FULL))


def test_fig3_refuses_to_render_without_the_provenance_record(n20_root):
    """No silent fallback to recomputing the gate from the strip."""
    (n20_root / "summary" / "detectability_gate_provenance.csv").unlink()
    with pytest.raises(FileNotFoundError, match="the chain SCM specification"):
        fig3_detectability.main(["--root", str(n20_root)])


def test_fig3_has_a_panel_per_topology(n20_root):
    """PS-8 encoding: each topology gets its own panel and its own y-range."""
    assert fig3_detectability.main(["--root", str(n20_root)]) == 0
    source = Path(fig3_detectability.__file__).read_text(encoding="utf-8")
    assert "len(TOPOLOGIES), 1" in source
    assert "free y-axis (PS-8)" in source


# --------------------------------------------------------------------------- #
# H4 figures — fig4 / fig5. Render-only, from the h4 artifacts, no refit.
# --------------------------------------------------------------------------- #


#: Which frozen-input constant each H4 script guards, and the file behind it.
H4_DIGEST_CONSTANTS = {
    "PER_SEED_DDB_SHA256": "h4_per_seed_ddb.csv",
    "DDB_BY_CELL_SHA256": "h4_ddb_by_cell.csv",
    "REGIME_CONTRAST_SHA256": "h4_regime_contrast.csv",
}


def _pin_h4_digests(monkeypatch, root: Path) -> None:
    """Point fig4/fig5's frozen-input guards at the fixture's own bytes.

    Same reasoning as the H3 and confirmatory fixtures: a synthetic tree can only
    render by substituting the expectation, which is itself the proof that the
    guard is load-bearing.
    """
    for module in (fig4_h4_scatter, fig5_regime_contrast):
        for constant, name in H4_DIGEST_CONSTANTS.items():
            if not hasattr(module, constant):
                continue
            digest = hashlib.sha256((root / "summary" / name).read_bytes()).hexdigest()
            monkeypatch.setattr(module, constant, digest)


@pytest.fixture()
def h4_root(tmp_path, monkeypatch):
    """A minimal h4 artifact tree: per-seed ΔΔB + by-cell + contrast, 12 cells.

    Built from `h4`'s own aggregation rather than hand-written summary rows, so a
    change to the by-cell schema breaks the figures' test here rather than at
    render time on the real tree.
    """
    from icknowledge.analysis import h4

    rows = []
    for cell in grid_cells():
        # Treatment carries a signed offset, control is centred — enough structure
        # for both captions' branches to have something to describe.
        centre = -0.05 if cell.regime == "effect_modifying" else 0.0
        for condition in h4.RUNGS:
            for seed in range(N_SEEDS_FULL):
                rows.append(
                    {
                        "topology": cell.topology,
                        "family": cell.family,
                        "regime": cell.regime,
                        "seed_idx": seed,
                        "seed": 1000 + seed,
                        "condition": condition,
                        "ddB": centre + 0.001 * (seed - 10),
                    }
                )
    per_seed = pd.DataFrame(rows)
    delta_s = pd.DataFrame(
        [
            {
                "topology": c.topology,
                "family": c.family,
                "regime": c.regime,
                "delta_S": -0.4 if c.regime == "effect_modifying" else 0.0,
            }
            for c in grid_cells()
        ]
    )
    by_cell = h4.by_cell(per_seed, delta_s)
    contrast = h4.regime_contrast(by_cell)

    summary = tmp_path / "summary"
    summary.mkdir(parents=True)
    per_seed.to_csv(summary / "h4_per_seed_ddb.csv", index=False)
    by_cell.to_csv(summary / "h4_ddb_by_cell.csv", index=False)
    contrast.to_csv(summary / "h4_regime_contrast.csv", index=False)
    _pin_h4_digests(monkeypatch, tmp_path)
    return tmp_path


@pytest.mark.parametrize(
    ("module", "stem", "caption_fragments"),
    [
        (
            fig4_h4_scatter,
            "fig4_h4_scatter",
            ["PS-8", "PS-9", "REGISTER-LOGGED PREDICTION", "descriptive SD"],
        ),
        (
            fig5_regime_contrast,
            "fig5_regime_contrast",
            ["PS-8", "PS-9 CONTRAST CRITERION", "SIGNAL-2 DISCRIMINATOR"],
        ),
    ],
    ids=["fig4", "fig5"],
)
def test_h4_figure_emits_pdf_png_and_caption(h4_root, module, stem, caption_fragments):
    out = h4_root / "figures"
    assert module.main(["--root", str(h4_root), "--out", str(out)]) == 0
    _assert_pdf(out / f"{stem}.pdf")
    _assert_png(out / f"{stem}.png")
    _assert_caption(out / f"{stem}.caption.txt", caption_fragments)


def test_h4_figures_compute_no_standard_error(h4_root):
    """PS-9: SD is descriptive; SE and CI are not displayed.

    Asserted on the SOURCE, because the claim is about what the figure is allowed
    to ENCODE, not about pixels. A future edit adding an SE band would have to
    introduce a sqrt(n) divisor or a `sem` call, and both are checked for. The
    captions are separately required to STATE the policy, so the two assertions
    together cover "does not draw it" and "says it does not draw it".
    """
    for module in (fig4_h4_scatter, fig5_regime_contrast):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert ".sem(" not in source, f"{module.__name__} computes a standard error"
        assert "np.sqrt(" not in source, f"{module.__name__} has a sqrt(n) divisor"
        assert "yerr=row.sd_ddB" in source or "yerr=values.std(ddof=1)" in source, (
            f"{module.__name__} no longer draws a descriptive SD band"
        )


def test_h4_captions_state_the_no_confidence_interval_policy(h4_root):
    """The captions must SAY that no SE/CI is shown — a reader cannot infer it."""
    for module, stem in (
        (fig4_h4_scatter, "fig4_h4_scatter"),
        (fig5_regime_contrast, "fig5_regime_contrast"),
    ):
        assert module.main(["--root", str(h4_root)]) == 0
        text = (h4_root / "figures" / f"{stem}.caption.txt").read_text(encoding="utf-8")
        lowered = " ".join(text.lower().split())
        assert "no standard errors or confidence intervals" in lowered
        assert "PS-9" in text


def test_fig4_caption_counts_come_from_the_data(h4_root):
    """The printed agree/20 counts and the pass tallies are read at render time."""
    assert fig4_h4_scatter.main(["--root", str(h4_root)]) == 0
    text = (h4_root / "figures" / "fig4_h4_scatter.caption.txt").read_text(
        encoding="utf-8"
    )
    by_cell = pd.read_csv(
        h4_root / "summary" / "h4_ddb_by_cell.csv", float_precision="round_trip"
    )
    em = by_cell[by_cell["regime"] == "effect_modifying"]
    for condition in ("L1-oracle", "L0"):
        expected = int(em[em["condition"] == condition]["sign_gate_pass"].sum())
        assert f"{expected} of 6" in text


# --------------------------------------------------------------------------- #
# H3 figures — fig6 / fig7. Presentation layer over the FROZEN H3 artifacts.
# --------------------------------------------------------------------------- #

#: The two H3 figure scripts, kept as one tuple so a guard added to one is exercised on
#: both without a second parametrize list drifting out of sync.
H3_FIGURE_MODULES = (fig6_h3_interpolation, fig7_h3_interaction)


def _h3_per_seed() -> pd.DataFrame:
    """12 instances x 20 seeds, with an additive exact-zero arm and a region-3 EM arm.

    Structured, not realistic: the additive arms sit at dcost_ld == 0 exactly (so
    the PS-7 note (c) "form-preserving estimation exact zero" annotation has a branch to take and
    the interaction pairs are exact ties), and the effect-modifying arms sit
    below both baselines in region 3 with the NLG arm the less negative of the
    two, so the strict-inequality interaction event fires on every EM pair.
    """
    rows = []
    for cell in grid_cells():
        modifying = cell.regime == "effect_modifying"
        for seed in range(N_SEEDS_FULL):
            if modifying:
                dcost_ld = (-0.40 if cell.family == "linear" else -0.15) - 0.001 * seed
                interaction_diff = 0.25
                region = 3
            else:
                dcost_ld = 0.0
                interaction_diff = 0.0
                region = 1
            rows.append(
                {
                    "topology": cell.topology,
                    "family": cell.family,
                    "regime": cell.regime,
                    "seed": 1000 + seed,
                    "seed_idx": seed,
                    "dcost_ld": dcost_ld,
                    "dcost_l0": 0.30 + 0.005 * seed,
                    "region": region,
                    "interaction_diff": interaction_diff,
                    "interaction_event": interaction_diff > 0,
                }
            )
    return pd.DataFrame(rows)


def _h3_instance_table_v2() -> pd.DataFrame:
    """The LD-2 classification columns fig6/fig7 read: channels and standing halts."""
    rows = []
    for cell in grid_cells():
        modifying = cell.regime == "effect_modifying"
        linear = cell.family == "linear"
        class_d = 20 if (linear and modifying and cell.topology == "triangle") else 0
        class_c = 20 if (linear and modifying and not class_d) else 0
        rows.append(
            {
                "topology": cell.topology,
                "family": cell.family,
                "regime": cell.regime,
                "denominator": "fourway",
                "n_seeds": N_SEEDS_FULL,
                "region_1_count": 0 if modifying else N_SEEDS_FULL,
                "region_2_count": 0,
                "region_3_count": N_SEEDS_FULL if modifying else 0,
                "halt_class_c": class_c,
                "halt_class_d": class_d,
                "halt": False,
            }
        )
    return pd.DataFrame(rows)


def _pin_digest(monkeypatch, path: Path) -> None:
    """Point both scripts' frozen-input guard at the fixture's own bytes.

    The guard's whole job is to refuse anything that is not the registered
    artifact, so a synthetic tree can only be rendered by substituting the
    expectation — which is itself the assertion that the guard is load-bearing.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for module in H3_FIGURE_MODULES:
        monkeypatch.setattr(module, "PER_SEED_SHA256", digest)


@pytest.fixture()
def h3_root(tmp_path, monkeypatch) -> Path:
    summary = tmp_path / "summary"
    summary.mkdir(parents=True)
    per_seed = summary / "h3_per_seed.csv"
    _h3_per_seed().to_csv(per_seed, index=False, encoding="utf-8")
    # The real v2 table carries a five-line supersession header (LD-2); it is
    # reproduced so the readers are exercised against the commented format.
    header = "\n".join(
        f"# {line}"
        for line in [
            "H3 instance table (v2) — re-emitted under LD-2.",
            "SUPERSEDES: <fixture>",
        ]
    )
    (summary / "h3_instance_table_v2.csv").write_text(
        header + "\n" + _h3_instance_table_v2().to_csv(index=False),
        encoding="utf-8",
        newline="",
    )
    _pin_digest(monkeypatch, per_seed)
    return tmp_path


@pytest.mark.parametrize(
    ("module", "stem", "caption_fragments"),
    [
        (
            fig6_h3_interpolation,
            "fig6_h3_interpolation",
            ["fourway", "PS-8", "region 2",
             "as corrected to the mutually exclusive interval partition",
             "no standard errors or confidence intervals"],
        ),
        (
            fig7_h3_interaction,
            "fig7_h3_interaction",
            ["fourway", "cannot be read as a discovery-penalty differential",
             "non-gating", "PS-8"],
        ),
    ],
    ids=["fig6", "fig7"],
)
def test_h3_figure_emits_pdf_png_and_caption(h3_root, module, stem, caption_fragments):
    out = h3_root / "figures"
    assert module.main(["--root", str(h3_root), "--out", str(out)]) == 0
    _assert_pdf(out / f"{stem}.pdf")
    _assert_png(out / f"{stem}.png")
    _assert_caption(out / f"{stem}.caption.txt", caption_fragments)


def test_h3_figures_compute_no_standard_error(h3_root):
    """PS-9: SD is descriptive; SE and CI are not displayed.

    Same source-text pattern as ``test_h4_figures_compute_no_standard_error``,
    extended to the H3 scripts: an SE band would need a `sem` call or a sqrt(n)
    divisor, and both are checked for. fig6 is the only one of the two that draws
    a dispersion band at all, so it alone must still draw the ddof=1 SD.
    """
    for module in H3_FIGURE_MODULES:
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert ".sem(" not in source, f"{module.__name__} computes a standard error"
        assert "np.sqrt(" not in source, f"{module.__name__} has a sqrt(n) divisor"
    fig6_source = Path(fig6_h3_interpolation.__file__).read_text(encoding="utf-8")
    assert "xerr=float(ld.std(ddof=1))" in fig6_source, (
        "fig6 no longer draws a descriptive SD band"
    )


def test_h3_captions_state_the_no_confidence_interval_policy(h3_root):
    """The captions must SAY that no SE/CI is shown — a reader cannot infer it."""
    for module, stem in (
        (fig6_h3_interpolation, "fig6_h3_interpolation"),
        (fig7_h3_interaction, "fig7_h3_interaction"),
    ):
        assert module.main(["--root", str(h3_root)]) == 0
        text = (h3_root / "figures" / f"{stem}.caption.txt").read_text(encoding="utf-8")
        lowered = " ".join(text.lower().split())
        assert "no standard errors or confidence intervals" in lowered
        assert "PS-9" in text


@pytest.mark.parametrize("module", H3_FIGURE_MODULES, ids=["fig6", "fig7"])
def test_h3_figures_reject_a_short_seed_axis(h3_root, monkeypatch, module):
    """A partially-written grid fails loudly rather than being plotted silently."""
    path = h3_root / "summary" / "h3_per_seed.csv"
    frame = pd.read_csv(path)
    frame[frame["seed_idx"] < N_SEEDS_FULL - 1].to_csv(
        path, index=False, encoding="utf-8"
    )
    _pin_digest(monkeypatch, path)
    with pytest.raises(ValueError, match="seeds"):
        module.main(["--root", str(h3_root), "--out", str(h3_root / "figures")])


@pytest.mark.parametrize("module", H3_FIGURE_MODULES, ids=["fig6", "fig7"])
def test_h3_figures_abort_on_a_tampered_frozen_input(h3_root, module):
    """The input-digest guard is the point: a modified frozen artifact never renders."""
    path = h3_root / "summary" / "h3_per_seed.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "dcost_ld"] += 0.5
    frame.to_csv(path, index=False, encoding="utf-8")
    with pytest.raises(RuntimeError, match="DIGEST MISMATCH"):
        module.main(["--root", str(h3_root), "--out", str(h3_root / "figures")])


def test_fig6_region_counts_are_recomputed_not_asserted(h3_root, monkeypatch):
    """The caption's region tallies must come from the data they describe.

    Same pattern as fig1's common-found-equality test: perturb the fixture and assert the
    printed number MOVES. A region count copied as a literal from the verdict
    report would not.
    """
    path = h3_root / "summary" / "h3_per_seed.csv"
    frame = pd.read_csv(path)
    total = len(frame)
    assert fig6_h3_interpolation.region_counts(frame)["region_2_total"] == 0

    assert fig6_h3_interpolation.main(["--root", str(h3_root)]) == 0
    caption = h3_root / "figures" / "fig6_h3_interpolation.caption.txt"
    assert f"region 2 contains 0 of {total}" in " ".join(
        caption.read_text(encoding="utf-8").split()
    )

    frame.loc[frame.index[0], "region"] = 2
    frame.to_csv(path, index=False, encoding="utf-8")
    _pin_digest(monkeypatch, path)
    assert fig6_h3_interpolation.region_counts(frame)["region_2_total"] == 1
    assert fig6_h3_interpolation.main(["--root", str(h3_root)]) == 0
    assert f"region 2 contains 1 of {total}" in " ".join(
        caption.read_text(encoding="utf-8").split()
    )


def test_fig7_gate_counts_are_recomputed_not_asserted(h3_root, monkeypatch):
    """The interaction counts and the orientation label move with the data."""
    path = h3_root / "summary" / "h3_per_seed.csv"
    frame = pd.read_csv(path)
    instances = pd.read_csv(
        h3_root / "summary" / "h3_instance_table_v2.csv", comment="#"
    )
    assert fig7_h3_interaction.orientation_verdict(frame, instances) == ("Mixed", 1)

    # Fire the additive collider pair as well: both pairs pass -> Supported (2/2).
    mask = (
        (frame["topology"] == "collider")
        & (frame["regime"] == "additive")
        & (frame["family"] == "nlg")
    )
    frame.loc[mask, "interaction_diff"] = 0.1
    frame.loc[mask, "interaction_event"] = True
    frame.to_csv(path, index=False, encoding="utf-8")
    _pin_digest(monkeypatch, path)
    assert fig7_h3_interaction.main(["--root", str(h3_root)]) == 0
    text = (h3_root / "figures" / "fig7_h3_interaction.caption.txt").read_text(
        encoding="utf-8"
    )
    assert "Supported" in text
    assert "(2/2 pairs pass)" in " ".join(text.split())


def test_fig7_carries_the_d3_clause_and_matches_the_analysis_copy(h3_root):
    """PS-8 requires the clause ON THE FIGURE and verbatim in the source."""
    from icknowledge.analysis.h3_v2 import D3_REGION3_CLAUSE

    source = Path(fig7_h3_interaction.__file__).read_text(encoding="utf-8")
    # The literal is written as adjacent string parts to stay inside the line
    # limit; undoing the source-level concatenation is what makes "verbatim in
    # the source" an assertion about the text rather than about the runtime value.
    assert D3_REGION3_CLAUSE in source.replace('"\n    "', "")
    assert fig7_h3_interaction.D3_ON_FIGURE_TEXT == D3_REGION3_CLAUSE

    assert fig7_h3_interaction.main(["--root", str(h3_root)]) == 0
    text = " ".join(
        (h3_root / "figures" / "fig7_h3_interaction.caption.txt")
        .read_text(encoding="utf-8")
        .split()
    )
    assert D3_REGION3_CLAUSE in text


def test_fig7_refuses_to_render_a_drifted_d3_clause(h3_root, monkeypatch):
    """A silent edit of the mandatory clause must stop the figure, not ship it."""
    monkeypatch.setattr(
        fig7_h3_interaction, "D3_ON_FIGURE_TEXT", "interaction reflects something else."
    )
    with pytest.raises(RuntimeError, match="VERBATIM"):
        fig7_h3_interaction.main(["--root", str(h3_root)])


# --------------------------------------------------------------------------- #
# Restyle contracts — fig4/fig5 guards, fig5's PS-8 layout, fig6/fig7 status
# --------------------------------------------------------------------------- #

H4_FIGURES = (
    (fig4_h4_scatter, "fig4_h4_scatter"),
    (fig5_regime_contrast, "fig5_regime_contrast"),
)


@pytest.mark.parametrize("module", [m for m, _ in H4_FIGURES], ids=["fig4", "fig5"])
def test_h4_figures_reject_a_short_seed_axis(h4_root, monkeypatch, module):
    """fig4/fig5 carry the seed guard."""
    path = h4_root / "summary" / "h4_per_seed_ddb.csv"
    frame = pd.read_csv(path)
    frame[frame["seed_idx"] < N_SEEDS_FULL - 1].to_csv(
        path, index=False, encoding="utf-8"
    )
    _pin_h4_digests(monkeypatch, h4_root)
    with pytest.raises(ValueError, match="seeds"):
        module.main(["--root", str(h4_root), "--out", str(h4_root / "figures")])


@pytest.mark.parametrize("module", [m for m, _ in H4_FIGURES], ids=["fig4", "fig5"])
def test_h4_figures_abort_on_a_tampered_frozen_input(h4_root, module):
    """A modified H4 artifact never renders."""
    path = h4_root / "summary" / "h4_per_seed_ddb.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "ddB"] += 0.5
    frame.to_csv(path, index=False, encoding="utf-8")
    with pytest.raises(RuntimeError, match="DIGEST MISMATCH"):
        module.main(["--root", str(h4_root), "--out", str(h4_root / "figures")])


@pytest.mark.parametrize(("module", "stem"), H4_FIGURES, ids=["fig4", "fig5"])
def test_h4_captions_name_the_twoway_population(h4_root, module, stem):
    """The H4 magnitude sits on the pairwise-vs-L2 set, arity TWO."""
    assert module.main(["--root", str(h4_root)]) == 0
    text = (h4_root / "figures" / f"{stem}.caption.txt").read_text(encoding="utf-8")
    flowed = " ".join(text.split())
    assert "twoway" in flowed
    assert "N_common_found_pairwise_vs_L2" in flowed
    assert "PS-4" in flowed


def test_fig5_gives_every_topology_its_own_axis(h4_root):
    """PS-8 compliance is structural — no shared ruler across topologies.

    Asserted as INDEPENDENCE, per the correction made on fig1: the property is not
    "no two ranges coincide" but "one topology's values never set another's axis".
    A layout with one shared y-scale fails this; per-topology sub-axes pass it.
    """
    from icknowledge.analysis import h4
    from icknowledge.analysis.loading import TOPOLOGIES

    per_seed = pd.read_csv(h4_root / "summary" / "h4_per_seed_ddb.csv")
    by_cell = pd.read_csv(h4_root / "summary" / "h4_ddb_by_cell.csv")

    fig5_regime_contrast._style()
    figure = fig5_regime_contrast.build_figure(per_seed, by_cell)
    baseline = [axis.get_ylim() for axis in figure.axes]
    assert len(baseline) == len(fig5_regime_contrast.RUNGS) * len(TOPOLOGIES)
    plt.close(figure)

    moved = TOPOLOGIES[0]
    inflated = per_seed.copy()
    mask = inflated["topology"] == moved
    inflated.loc[mask, "ddB"] = inflated.loc[mask, "ddB"] * 50.0
    delta_s = by_cell[["topology", "family", "regime", "delta_S"]].drop_duplicates()
    figure = fig5_regime_contrast.build_figure(inflated, h4.by_cell(inflated, delta_s))
    after = [axis.get_ylim() for axis in figure.axes]
    plt.close(figure)

    columns = len(TOPOLOGIES)
    for index, (before, now) in enumerate(zip(baseline, after, strict=True)):
        own_panel = (index % columns) == TOPOLOGIES.index(moved)
        if own_panel:
            assert before != now, f"the {moved} panel did not rescale with its own data"
        else:
            assert before == now, (
                f"{moved} values moved another topology's axis — that is the "
                "shared ruler PS-8 forbids"
            )


def test_fig5_encodes_topology_with_the_reserved_markers():
    """Topology is encoded on fig5, not only labelled."""
    assert fig5_regime_contrast.TOPOLOGY_MARKER == {
        "triangle": "o",
        "collider": "D",
        "chain": "s",
    }


def test_every_linear_control_strip_carries_exactly_one_status(h3_root):
    """A blank status reads as an unexamined cell, not a clean one."""
    per_seed = pd.read_csv(h3_root / "summary" / "h3_per_seed.csv")
    instances = pd.read_csv(
        h3_root / "summary" / "h3_instance_table_v2.csv", comment="#"
    )
    seen = []
    for cell in grid_cells():
        if cell.family != "linear":
            continue
        block = per_seed[
            (per_seed["topology"] == cell.topology)
            & (per_seed["family"] == "linear")
            & (per_seed["regime"] == cell.regime)
        ].sort_values("seed_idx")
        status = fig6_h3_interpolation.linear_status(
            instances,
            cell.topology,
            cell.regime,
            block["dcost_ld"].to_numpy(dtype=float),
            block["region"].astype(int).to_numpy(),
        )
        assert status, f"{cell.label} renders an EMPTY status"
        seen.append(status)
    assert len(seen) == 6


def test_fig6_status_prefers_a_halt_over_every_other_label(h3_root):
    """A standing halt outranks a channel label — it is not a footnote."""
    instances = pd.read_csv(
        h3_root / "summary" / "h3_instance_table_v2.csv", comment="#"
    )
    instances["halt"] = True
    status = fig6_h3_interpolation.linear_status(
        instances,
        "collider",
        "effect_modifying",
        np.full(N_SEEDS_FULL, 0.1),
        np.full(N_SEEDS_FULL, 3),
    )
    assert "HARNESS HALT STANDS" in status


def test_fig6_reports_a_clean_channel_that_is_not_an_exact_zero(h3_root):
    """A "clean" channel whose mean is not exactly zero must render a status, not blank."""
    instances = pd.read_csv(
        h3_root / "summary" / "h3_instance_table_v2.csv", comment="#"
    )
    for column in ("halt", "halt_class_c", "halt_class_d"):
        instances[column] = 0
    status = fig6_h3_interpolation.linear_status(
        instances,
        "chain",
        "additive",
        np.full(N_SEEDS_FULL, -0.000152),
        np.ones(N_SEEDS_FULL, dtype=int),
    )
    assert status.startswith("clean")
    assert "form-preserving estimation" not in status


def test_fig6_declares_no_family_fill_mapping():
    """The dict encoded nothing visible on this figure, so it is gone."""
    assert not hasattr(fig6_h3_interpolation, "FAMILY_FILLED")


def test_fig6_xlim_is_free_per_topology(h3_root):
    """PS-8 encoding: one topology's values never set another's x-range."""
    per_seed = pd.read_csv(h3_root / "summary" / "h3_per_seed.csv")
    baseline = fig6_h3_interpolation._xlim_by_topology(per_seed)
    assert set(baseline) == set(fig6_h3_interpolation.ROWS)

    moved = fig6_h3_interpolation.ROWS[0]
    inflated = per_seed.copy()
    mask = inflated["topology"] == moved
    inflated.loc[mask, "dcost_ld"] = inflated.loc[mask, "dcost_ld"] - 25.0
    after = fig6_h3_interpolation._xlim_by_topology(inflated)
    assert after[moved] != baseline[moved]
    for topology in set(baseline) - {moved}:
        assert after[topology] == baseline[topology]


def test_fig7_reports_region_occupancy_per_family_arm(h3_root):
    """PS-8 sub-clause (i): per ARM, not pooled — a pooled figure can be carried by one arm."""
    per_seed = pd.read_csv(h3_root / "summary" / "h3_per_seed.csv")
    for regime in ("additive", "effect_modifying"):
        for family in ("linear", "nlg"):
            region, count, total = fig7_h3_interaction.arm_region_occupancy(
                per_seed, "collider", regime, family
            )
            assert total == N_SEEDS_FULL
            assert 0 < count <= total
            assert region in (1, 2, 3)
    phrase = fig7_h3_interaction.arm_region_phrase(
        per_seed, "collider", "effect_modifying"
    )
    assert phrase.startswith("linear ") and "nonlinear-Gaussian" in phrase


def test_fig7_arm_counts_are_recomputed_not_asserted(h3_root):
    """Move some seeds' regions and the per-arm count must follow them."""
    per_seed = pd.read_csv(h3_root / "summary" / "h3_per_seed.csv")
    before = fig7_h3_interaction.arm_region_occupancy(
        per_seed, "collider", "effect_modifying", "nlg"
    )
    mask = (
        (per_seed["topology"] == "collider")
        & (per_seed["regime"] == "effect_modifying")
        & (per_seed["family"] == "nlg")
    )
    per_seed.loc[per_seed.index[mask][:3], "region"] = 2
    after = fig7_h3_interaction.arm_region_occupancy(
        per_seed, "collider", "effect_modifying", "nlg"
    )
    assert after[1] != before[1]


def test_fig7_renders_a_standing_halt_on_the_page(h3_root):
    """A halted row is drawn, not only suppressed.

    The frozen artifacts carry no halted row, so the machinery is exercised on a
    synthetic one — otherwise the branch would be untested precisely because the
    real data happens not to trigger it.
    """
    instances = pd.read_csv(
        h3_root / "summary" / "h3_instance_table_v2.csv", comment="#"
    )
    assert not instances["halt"].astype(bool).any(), "fixture is already halted"
    instances.loc[
        (instances["topology"] == "collider") & (instances["family"] == "linear"),
        "halt",
    ] = True
    assert fig7_h3_interaction.standing_halt(instances, "collider")

    per_seed = pd.read_csv(h3_root / "summary" / "h3_per_seed.csv")
    fig7_h3_interaction._style()
    figure = fig7_h3_interaction.build_figure(per_seed, instances)
    drawn = [text.get_text() for axis in figure.axes for text in axis.texts]
    plt.close(figure)
    assert any("HARNESS HALT STANDS" in text for text in drawn), (
        "a standing halt is not visible anywhere on the rendered page"
    )


def test_fig7_ylim_is_free_per_topology(h3_root):
    """PS-8: every topology panel is scaled from its own data."""
    per_seed = pd.read_csv(h3_root / "summary" / "h3_per_seed.csv")
    instances = pd.read_csv(
        h3_root / "summary" / "h3_instance_table_v2.csv", comment="#"
    )

    def limits(frame):
        fig7_h3_interaction._style()
        figure = fig7_h3_interaction.build_figure(frame, instances)
        out = {axis.get_title(): axis.get_ylim() for axis in figure.axes}
        plt.close(figure)
        return out

    baseline = limits(per_seed)
    inflated = per_seed.copy()
    mask = inflated["topology"] == "chain"
    inflated.loc[mask, "dcost_ld"] = inflated.loc[mask, "dcost_ld"] - 40.0
    after = limits(inflated)

    for title, before in baseline.items():
        if title.startswith("chain"):
            assert after[title] != before, "the chain panel did not rescale"
        else:
            assert after[title] == before, f"chain values moved {title!r}'s axis"


def test_style_literals_are_identical_across_every_script():
    """The scripts are self-contained BY CHOICE, so drift is the risk.

    Each script declares its own REGIME_COLOUR / TOPOLOGY_MARKER / FAMILY_FILLED
    rather than importing a shared module, which keeps every figure readable on its
    own. The cost is that a one-character edit in one file could silently
    de-synchronize the encoding across the set; this is what makes that impossible
    without a visible failure.
    """
    expected = {
        "REGIME_COLOUR": {"effect_modifying": "#0072B2", "additive": "#D55E00"},
        "TOPOLOGY_MARKER": {"triangle": "o", "collider": "D", "chain": "s"},
        "FAMILY_FILLED": {"linear": True, "nlg": False},
    }
    modules = (
        fig1_cost_ladder,
        fig2_validity_gap,
        fig3_detectability,
        fig4_h4_scatter,
        fig5_regime_contrast,
        fig6_h3_interpolation,
        fig7_h3_interaction,
    )
    seen: dict[str, list[str]] = {name: [] for name in expected}
    for module in modules:
        for name, literal in expected.items():
            if hasattr(module, name):
                assert getattr(module, name) == literal, (
                    f"{module.__name__}.{name} has drifted from the house literal"
                )
                seen[name].append(module.__name__)
    for name, users in seen.items():
        assert users, f"no script declares {name} — this test would pass vacuously"
