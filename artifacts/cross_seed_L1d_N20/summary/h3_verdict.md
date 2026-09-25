# H3 verdict — mechanical PS-8 reading at N=20 (four-rung tree)

Produced by `icknowledge/analysis/h3.py` + `h3_covariates.py`. Every label
below is the OUTPUT of a pure function applying **PS-8** with the
**region-partition correction**, to frozen numbers. Nothing is narrated; no
threshold was chosen after the numbers existed.

- **Population:** PS-4 four-way common-found (`fourway`). Every
  table below carries the `fourway` denominator label.
- **Realized cost source:** `realized_cost_common_found_fourway` in `per_seed_by_group.csv`,
  combined across groups by the stored `N_common_found_fourway` counts. The
  cell-level `Δ_cost_common_found_fourway` column is the between-group GAP
  and is deliberately not used.
- **dcost_ld** := cost(L1-discovered) − cost(L1-oracle);
  **dcost_l0** := cost(L0) − cost(L1-oracle).
- **Regions (PS-8, corrected — mutually exclusive):** region 1
  `dcost_ld ∈ [min(0,dcost_l0), max(0,dcost_l0)]` (CLOSED; ties → region 1);
  region 2 `dcost_ld > max(0,dcost_l0)`; region 3 `dcost_ld < min(0,dcost_l0)`.
- **Gate:** k = **16/20**, numerator = **region 2
  only**; region-3 seeds are non-firing but RETAINED in the denominator.
- **Bands (PS-9, unchanged):** ≥5/6 Supported ·
  4/6 Mixed · ≤3/6 Null.
- **Linear is the CONTROL arm.** It is never banded into a worse-than-no-graph verdict.

---

## HALTS · FLAGS · ESCALATIONS — at a glance

- **N_valid halts (PS-8):** 0
- **HARNESS HALTs (PS-8 halt-scope rule):** 3 — chain/linear/effect_modifying, collider/linear/effect_modifying, triangle/linear/effect_modifying
- **PS-7 note (f)/(g) class-(b) escalations:** 0 — none: no class-(b) call was made on any linear cell.
- **Structural-expectation violations (PS-8 orientation-covariate split):** 0
- **Reversal seeds (dcost_l0 < 0):** 0 of 240
- **t instability flags:** 1

---

## 3a — N_valid gate (PS-8)

| topology | family | regime | denominator | N_valid | N_required | halt_n_valid |
|---|---|---|---|---|---|---|
| chain | linear | additive | fourway | 20 | 20 | False |
| chain | linear | effect_modifying | fourway | 20 | 20 | False |
| chain | nlg | additive | fourway | 20 | 20 | False |
| chain | nlg | effect_modifying | fourway | 20 | 20 | False |
| collider | linear | additive | fourway | 20 | 20 | False |
| collider | linear | effect_modifying | fourway | 20 | 20 | False |
| collider | nlg | additive | fourway | 20 | 20 | False |
| collider | nlg | effect_modifying | fourway | 20 | 20 | False |
| triangle | linear | additive | fourway | 20 | 20 | False |
| triangle | linear | effect_modifying | fourway | 20 | 20 | False |
| triangle | nlg | additive | fourway | 20 | 20 | False |
| triangle | nlg | effect_modifying | fourway | 20 | 20 | False |

**No N_valid halt.** Every instance carries N_valid = 20/20 on the `fourway` population, so the common-found no-op holds and the integer gate is applied against its calibrated denominator without re-basing.

---

## CONTROL ARM (linear) — PS-8

All six linear instances, banded or not. A clean control is a POSITIVE
reportable outcome (PS-8: "the linear control is what separates 'discovery
genuinely hurts NLG' from 'seed noise'"), so it is stated explicitly rather
than left as silence.

| topology | regime | denominator | N_valid | region_1_count | region_2_count | region_3_count | reversal_seed_count |
|---|---|---|---|---|---|---|---|
| chain | additive | fourway | 20 | 19 | 0 | 1 | 0 |
| chain | effect_modifying | fourway | 20 | 8 | 0 | 12 | 0 |
| collider | additive | fourway | 20 | 20 | 0 | 0 | 0 |
| collider | effect_modifying | fourway | 20 | 0 | 0 | 20 | 0 |
| triangle | additive | fourway | 20 | 20 | 0 | 0 | 0 |
| triangle | effect_modifying | fourway | 20 | 0 | 0 | 20 | 0 |

- **chain / linear / additive** — linear control clean — interpolation held, diagnostic not triggered (region-1 count 19/20).
- **chain / linear / effect_modifying** — HARNESS HALT — class-(a) majority among 12+0 violating seeds; no verdict emitted for the affected component(s).
  - class (a) seeds: [0, 1, 6, 8, 9, 10, 11, 12, 13, 16, 17, 19]
  - class (b) seeds: —
- **collider / linear / additive** — linear control clean — interpolation held, diagnostic not triggered (region-1 count 20/20).
- **collider / linear / effect_modifying** — HARNESS HALT — class-(a) majority among 20+0 violating seeds; no verdict emitted for the affected component(s).
  - class (a) seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  - class (b) seeds: —
- **triangle / linear / additive** — linear control clean — interpolation held, diagnostic not triggered (region-1 count 20/20).
- **triangle / linear / effect_modifying** — HARNESS HALT — class-(a) majority among 20+0 violating seeds; no verdict emitted for the affected component(s).
  - class (a) seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
  - class (b) seeds: —

**PS-7 note (f)/(g) escalations: none.** No violating linear seed classified
class (b), so the pre-stated structural guarantee (a correctly-recovered
skeleton's canonical-order tie-break reproduces the true X–X
orientations) was not contradicted on any linear cell.

### HARNESS HALT record (PS-8)

The H3 verdict pipeline **stops for the affected component(s)** rather
than emitting a label. The fix is a register-logged correction in a
follow-up adjudication before any re-run — never a silent retune.

- **HALT** — chain / linear / effect_modifying: region-1 count 8/20 < 16; violating seeds classify 12 class-(a) / 0 class-(b) → class-(a) majority.
- **HALT** — collider / linear / effect_modifying: region-1 count 0/20 < 16; violating seeds classify 20 class-(a) / 0 class-(b) → class-(a) majority.
- **HALT** — triangle / linear / effect_modifying: region-1 count 0/20 < 16; violating seeds classify 20 class-(a) / 0 class-(b) → class-(a) majority.

**Propagation — components invalidated by the halt(s):**

- `orientation_pairs`: collider × additive, collider × effect_modifying
- `q2_linear_controls`: collider linear control (worse-than-no-graph + fallback tier), effect_modifying
- `corroborating_pairs`: chain × effect_modifying, triangle × effect_modifying

---

## 3c — H3a interaction, orientation channel (PS-8, confirmatory-primary)

The two collider pairs (× additive, × effect-modifying). Per-seed event =
`dcost_ld(NLG, s) − dcost_ld(linear, s) > 0` **strictly**, paired by
`seed_idx` (CRN); a zero difference is non-firing (PS-8).

| topology | regime | denominator | n_seeds | interaction_event_count | zero_difference_count | mean_interaction_diff | gate_pass |
|---|---|---|---|---|---|---|---|
| collider | additive | fourway | 20 | 0 | 20 | +0.000000 | False |
| collider | effect_modifying | fourway | 20 | 19 | 0 | +0.284746 | True |

### VERDICT: **NO LABEL EMITTED — HALTED**

The collider-linear HARNESS HALT invalidates both collider interaction
pairs — the confirmatory-primary orientation channel — per PS-8's
halt-scope rule. The seed counts above are recorded for the register;
**they are not read as a verdict** and no band is applied to them.

**Structure-conditioned reading (PS-8, structural, pre-stated).** The
interaction is not a flat six-pair count: the collider is the only one of the
three topologies whose graph admits any CI-based orientation, so it alone carries
the orientation channel; chain and triangle corroborate on the
skeleton/adjacency channel and cannot gate it.

**Pre-stated predicted pattern (PS-8):** *collider firing and chain/triangle
attenuated is the EXPECTED result and is read as Supported on the
orientation channel — it may not be reported as Null.*

---

## worse-than-no-graph co-primary — NLG region-2 banded verdict

**DUAL ROLE — computed once, emitted once.** This single banded result
serves as the **worse-than-no-graph co-primary** (always reported) **and** as the **H3b
NLG-existence fallback tier** (invoked ONLY if 3c is Mixed/Null). The
fallback section cross-references it by the anchor `#worse-than-no-graph-co-primary--nlg-region-2-banded-verdict` and never
reprints these counts, so Results cannot double-count one computation as
two verdicts.

| topology | regime | denominator | N_valid | region_1_count | region_2_count | region_3_count | reversal_seed_count | region_2_gate_pass |
|---|---|---|---|---|---|---|---|---|
| chain | additive | fourway | 20 | 20 | 0 | 0 | 0 | False |
| chain | effect_modifying | fourway | 20 | 9 | 0 | 11 | 0 | False |
| collider | additive | fourway | 20 | 20 | 0 | 0 | 0 | False |
| collider | effect_modifying | fourway | 20 | 0 | 0 | 20 | 0 | False |
| triangle | additive | fourway | 20 | 20 | 0 | 0 | 0 | False |
| triangle | effect_modifying | fourway | 20 | 15 | 0 | 5 | 0 | False |

### VERDICT: **Null** — 0/6 NLG instances reach region-2 count ≥ 16/20

Region-3 seeds are counted in **neither** the numerator **nor** removed from
the denominator; their per-instance tally is the `region_3_count` column
above. `reversal_seed_count` is the per-instance incidence of `dcost_l0 < 0`
— the case where the corrected partition could change a region assignment.

**Linear control arm beside the NLG verdict (no band applied):**

| topology | regime | denominator | region_1_count | region_2_count | region_3_count |
|---|---|---|---|---|---|
| chain | additive | fourway | 19 | 0 | 1 |
| chain | effect_modifying | fourway | 8 | 0 | 12 |
| collider | additive | fourway | 20 | 0 | 0 |
| collider | effect_modifying | fourway | 0 | 0 | 20 |
| triangle | additive | fourway | 20 | 0 | 0 |
| triangle | effect_modifying | fourway | 0 | 0 | 20 |

**Control-arm invalidation (PS-8 halt-scope rule):** collider linear control (worse-than-no-graph + fallback tier), effect_modifying — invalidated by the HARNESS HALT above. The counts are shown for the
register; they do not discharge the control-arm role until a follow-up
adjudication.

---

## 3e — CORROBORATING — NON-GATING (PS-8)

Chain and triangle, purely descriptive. **No threshold is applied.** Per the
registered symmetric no-consequence clause, whichever way these numbers
point they move no label, no gate and no band.

| topology | family | regime | denominator | mean_skeleton_shd | region_1_count | region_2_count | region_3_count |
|---|---|---|---|---|---|---|---|
| chain | linear | additive | fourway | 0.1000 | 19 | 0 | 1 |
| chain | linear | effect_modifying | fourway | 0.1000 | 8 | 0 | 12 |
| chain | nlg | additive | fourway | 0.1000 | 20 | 0 | 0 |
| chain | nlg | effect_modifying | fourway | 0.0500 | 9 | 0 | 11 |
| triangle | linear | additive | fourway | 0.0000 | 20 | 0 | 0 |
| triangle | linear | effect_modifying | fourway | 1.0000 | 0 | 0 | 20 |
| triangle | nlg | additive | fourway | 0.0000 | 20 | 0 | 0 |
| triangle | nlg | effect_modifying | fourway | 0.0000 | 15 | 0 | 5 |

**The four corroborating interaction pairs' seed counts:**

| topology | regime | denominator | n_seeds | interaction_event_count | zero_difference_count |
|---|---|---|---|---|---|
| chain | additive | fourway | 20 | 2 | 18 |
| chain | effect_modifying | fourway | 20 | 14 | 0 |
| triangle | additive | fourway | 20 | 0 | 20 |
| triangle | effect_modifying | fourway | 20 | 20 | 0 |

**Descriptive direction vs. 3c.** Pre-stated expectation: NLG skeleton-SHD >
linear skeleton-SHD on chain and triangle. Observed on 0/4 corroborating cells:

- chain × additive: NLG 0.1000 = linear 0.1000
- chain × effect_modifying: NLG 0.0500 < linear 0.1000
- triangle × additive: NLG 0.0000 = linear 0.0000
- triangle × effect_modifying: NLG 0.0000 < linear 1.0000

Reported as a **tension** with the pre-stated pattern. Either way **no label moves** — the
no-consequence clause is symmetric by registration.

---

## 3f — Descriptive extras

**Normalized interpolation coordinate t = dcost_ld / dcost_l0.** Descriptive
only (PS-8); it enters no gate. `t_instability_flag` fires where
`|mean dcost_l0|` falls within its own cross-seed SD of zero — a near-zero
denominator a reader must not over-read.

| topology | family | regime | denominator | mean_dcost_ld | sd_dcost_ld | mean_dcost_l0 | sd_dcost_l0 | mean_t | t_instability_flag |
|---|---|---|---|---|---|---|---|---|---|
| chain | linear | additive | fourway | -0.000152 | +0.000841 | +0.117054 | +0.051077 | -0.001604 | False |
| chain | linear | effect_modifying | fourway | -0.046546 | +0.052963 | +0.075781 | +0.068633 | -2.408035 | False |
| chain | nlg | additive | fourway | +0.000048 | +0.000163 | +0.066001 | +0.036142 | +0.000963 | False |
| chain | nlg | effect_modifying | fourway | -0.010750 | +0.019520 | +0.044164 | +0.039865 | -0.703816 | False |
| collider | linear | additive | fourway | +0.000000 | +0.000000 | +0.864050 | +0.205468 | +0.000000 | False |
| collider | linear | effect_modifying | fourway | -0.442922 | +0.185257 | +0.400885 | +0.197507 | -2.122693 | False |
| collider | nlg | additive | fourway | +0.000000 | +0.000000 | +0.383547 | +0.094211 | +0.000000 | False |
| collider | nlg | effect_modifying | fourway | -0.158176 | +0.061236 | +0.223191 | +0.115601 | -1.139318 | False |
| triangle | linear | additive | fourway | +0.000000 | +0.000000 | +0.172175 | +0.104886 | +0.000000 | False |
| triangle | linear | effect_modifying | fourway | -0.264896 | +0.050161 | +0.046533 | +0.004288 | -5.751415 | False |
| triangle | nlg | additive | fourway | +0.000000 | +0.000000 | +0.000382 | +0.000644 | +0.000000 | True |
| triangle | nlg | effect_modifying | fourway | -0.000042 | +0.005448 | +0.003981 | +0.001434 | +0.063329 | False |

**Supporting evidence (PS-6 / the common-found population) — reported, enters no verdict rule.**

| topology | family | regime | denominator | validity_A_neg | validity_A_pos | Gap_cost_L1d |
|---|---|---|---|---|---|---|
| chain | linear | additive | all-eligible (PS-6/common-found reporting population) | 0.7690 | 0.9702 | +0.000000 |
| chain | linear | effect_modifying | all-eligible (PS-6/common-found reporting population) | 0.3065 | 0.9834 | +0.000000 |
| chain | nlg | additive | all-eligible (PS-6/common-found reporting population) | 0.7713 | 0.9588 | +0.000000 |
| chain | nlg | effect_modifying | all-eligible (PS-6/common-found reporting population) | 0.3756 | 0.9687 | +0.000000 |
| collider | linear | additive | all-eligible (PS-6/common-found reporting population) | 0.8385 | 0.9845 | +0.000000 |
| collider | linear | effect_modifying | all-eligible (PS-6/common-found reporting population) | 0.0364 | 0.9983 | +0.000000 |
| collider | nlg | additive | all-eligible (PS-6/common-found reporting population) | 0.6826 | 0.9139 | +0.000000 |
| collider | nlg | effect_modifying | all-eligible (PS-6/common-found reporting population) | 0.1324 | 0.9935 | +0.000000 |
| triangle | linear | additive | all-eligible (PS-6/common-found reporting population) | 0.9255 | 0.9889 | +0.000000 |
| triangle | linear | effect_modifying | all-eligible (PS-6/common-found reporting population) | 0.0017 | 1.0000 | +0.000000 |
| triangle | nlg | additive | all-eligible (PS-6/common-found reporting population) | 0.9972 | 0.9889 | +0.000000 |
| triangle | nlg | effect_modifying | all-eligible (PS-6/common-found reporting population) | 0.9177 | 1.0000 | +0.000000 |

---

## 3g — H3b NLG-existence fallback tier (PS-8, compute-once seam)

The orientation channel (3c) emitted **no label** — it is HALTED, not
Supported, Mixed or Null. The fallback's invocation condition (3c Mixed
or Null) is therefore not satisfiable, and the **fallback-tier control**
is itself invalidated by the same collider-linear HARNESS HALT per
PS-8's halt-scope rule. **The fallback tier is NOT invoked.**

---

## Diagnostic covariates per instance (PS-8)

`skeleton_shd` on the FULL node set including A (BK constrains A-edge
ORIENTATION only, so an A–X adjacency error is a real discovery error).
`orientation_wrong_count` on the X–X edges only, marginalized over channel
(the strict orientation-error term). `tiebreak_resolved_count` marginalized over
correctness (provenance, not error). `wrong∧tiebreak` is the class-(b)
diagnostic cell. **nSID** on the ENDOGENOUS sub-graph excluding A, per-topology
normalized (d_eff = 3 collider/chain, 2 triangle); descriptive, enters no rule.

| topology | family | regime | denominator | mean_skeleton_shd | mean_orientation_wrong | mean_tiebreak_resolved | mean_wrong_and_tiebreak | mean_n_sid | structural_expectation_violations |
|---|---|---|---|---|---|---|---|---|---|
| chain | linear | additive | fourway | 0.1000 | 0.0000 | 2.1000 | 0.0000 | 0.0000 | 0 |
| chain | linear | effect_modifying | fourway | 0.1000 | 0.0000 | 2.1000 | 0.0000 | 0.0000 | 0 |
| chain | nlg | additive | fourway | 0.1000 | 0.0000 | 2.1000 | 0.0000 | 0.0000 | 0 |
| chain | nlg | effect_modifying | fourway | 0.0500 | 0.0000 | 2.0500 | 0.0000 | 0.0000 | 0 |
| collider | linear | additive | fourway | 0.0000 | 0.0000 | 1.7000 | 0.0000 | 0.0000 | 0 |
| collider | linear | effect_modifying | fourway | 0.0000 | 0.0000 | 1.9000 | 0.0000 | 0.0000 | 0 |
| collider | nlg | additive | fourway | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| collider | nlg | effect_modifying | fourway | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| triangle | linear | additive | fourway | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0 |
| triangle | linear | effect_modifying | fourway | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| triangle | nlg | additive | fourway | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0 |
| triangle | nlg | effect_modifying | fourway | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0 |

**Structural-expectation violations: 0.** On every chain/triangle cell
with `skeleton_shd = 0`, `tiebreak_resolved_count` equals the number of
true X–X edges and `orientation_wrong_count` is 0, as pre-stated
(PS-8, per PS-7 note (f)).

**Reversal-seed incidence (dcost_l0 < 0): 0 seed(s) across all
12 instances** — the per-instance tally is the `reversal_seed_count` column.
This is the incidence at which the corrected partition could change a
region assignment relative to the superseded original region glosses (PS-8).

