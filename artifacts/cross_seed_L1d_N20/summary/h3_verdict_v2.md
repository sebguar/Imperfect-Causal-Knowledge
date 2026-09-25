# H3 verdict (v2) — re-emitted under LD-2

<!-- SUPERSESSION HEADER — H3 verdict re-emitted under LD-2 -->

> **SUPERSEDES:**
> - `results/cross_seed_L1d_N20/summary/h3_verdict.md` @ commit `d8bdd23`
> - `results/cross_seed_L1d_N20/summary/h3_instance_table.csv` @ commit `d8bdd23`
> 
> Those artifacts are **audit objects** and are NOT overwritten, altered or
> deleted. This file is a **versioned re-emission** of the Step-3
> classification/gate/label layer ONLY. Per LD-2, nothing re-runs:
> discovery, estimation, recourse, scoring, covariates (h3_covariates.csv)
> and per-seed quantities (h3_per_seed.csv) are frozen and correct; only the
> Step-3 classification/gate/label layer re-executes.

Produced by `icknowledge/analysis/h3_v2.py`, consuming the FROZEN
`h3_per_seed.csv` and `h3_covariates.csv`. Every label below is the OUTPUT of a
pure function applying **PS-8**, including its region
partition, as **corrected by LD-2**, to frozen numbers. No
threshold, band, gate numerator, k or region definition moved.

## Epistemic status

This verdict is **OUTCOME-EXPOSED, TEXTUALLY FORCED and LABEL-INVARIANT**.
It is *not* outcome-independent, and it is **NOT outcome-blind and does
not claim to be**: the halted run's numbers were visible when the
correction was settled. Its standing rests on three grounds (PS-8, LD-2):

1. **TEXTUAL FORCING** — the correction direction is uniquely determined by
   appendix text predating every number (PS-7 note (c)'s additive scoping,
   and the interaction-blindness consequence); no alternative correction
   exists that the observed numbers could have selected among.
2. **LABEL-INVARIANCE** — the corrected taxonomy is applied identically
   whatever labels it produces; no threshold, band, gate numerator, k or
   region definition moves.
3. **ADDITIVE-GRID CONFIRMATION** — dcost_ld is exactly 0.000000 on every
   perfect-recovery additive cell, so PS-7 note (c)'s identity holds
   byte-exactly precisely where it is scoped.

**The correction (LD-2).** Limb a3 of the class-(a) rule justified itself by
"graph perfect ⟹ under form-preserving estimation + PS-7 note (c)
ΔCost_ld = 0 by construction". PS-7 note (c) scopes that identity to the
ADDITIVE regime, explicitly excluding EM, where PS-7 records that the two L1
rungs differ by graph AND interaction-blindness. Limb a3's regime-blind
application was a drafting error in the note.

---

- **Population:** PS-4 four-way common-found (`fourway`). Every
  table below carries the `fourway` denominator label.
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

## Step 1 — mechanism spot-checks (read-only, confirmatory, not load-bearing)

**1a — per-group dcost_ld, three EM linear cells** (frozen `per_seed_by_group.csv`, fourway costs). The negative dcost_ld is driven by the **A = −1** group on all three cells: chain −0.065922 (A=−1) vs +0.040125 (A=+1); collider −0.533621 vs +0.056312; triangle −0.377208 vs +0.126783. No flag. This check does not gate Step 3 — PS-7's registration is the authority for class (c).

**1b — collider/linear/EM seed 00, discovered vs L1-oracle X3 equation.** The L1-discovered coefficients are not persisted by any artifact, so the fit was reconstructed from the manifest's own config and the manifest's stored discovered GRAPH (PC-Stable NOT re-run); the reconstruction was verified by (i) the estimation-sample `dataset_identity` sha256 matching the manifest exactly and (ii) the oracle refit reproducing the manifest's stored oracle X3 dict coefficient-for-coefficient. Result: identical structure minus the `A*X1` interaction column, pooled main coefficients (oracle X1 +0.493566 / X2 +0.474082 / A +1.056575 / A*X1 +0.310718 vs discovered X1 +0.483213 / X2 +0.464037 / A +1.120211). No structural surprise beyond the missing A·X column, so PS-7 note (b)'s byte-identity scope is not contradicted.

---

## Step 2 — truth-side check (triangle/linear/EM; LD-2)

Truth-side only: population quantities in closed form from the FROZEN
builder coefficients (`icknowledge/scm/triangles.py`, read programmatically
off the builder signature). No production data was read, nothing under
`results/` was consulted for this section beyond the manifest field naming
`N_disc`, and PC-Stable was not re-run.

**Frozen builder coefficients — triangle / linear / effect_modifying.**

    X1 := a·A + U1                     a = 2.0
    X2 := g_of_A·X1 + U2               g_pos = 0.9, g_neg = 0.3
    U1, U2 ~ N(0, σ²)                  σ = 1.0

Under the centered A ∈ {−1,+1} encoding, `g_of_A = ḡ + γ·A` exactly, so the
X2 equation's coefficients are:

| term | coefficient | value |
|---|---|---|
| A main effect | β_A | **0.0** — the builder carries NO additive A term under EM |
| X1 | ḡ = (g_pos+g_neg)/2 | 0.6000 |
| A·X1 interaction | γ = (g_pos−g_neg)/2 | 0.3000 |
| noise sd | σ | 1.0000 |

**N_disc = 4000**, read from the manifest field `discovery.effect_modifying.dataset.n_rows` and
verified identical across all 20 seeds (never assumed). α = 0.05, CI test = Fisher-z.

**Population correlations over BOTH conditioning sets PC tests at this order.**
PC removes the adjacency if independence is accepted on ANY tested subset, so
the operative subset is the one with the smallest |ρ|.

| conditioning set S | \|S\| | ρ(A, X2 \| S) | Fisher-z rejection prob at N_disc | operative |
|---|---|---|---|---|
| ∅ | 0 | +0.705882353 | 1.000000 |  |
| {X1} | 1 | +0.000000000 | 0.025000 | **← operative** |

**Derivation of the partial (exact, and exact for EVERY parameter value).**

    X2 = (ḡ + γA)(aA + U1) + U2 = ḡa·A + γa + ḡ·U1 + γ·A·U1 + U2   (A² = 1)

    Cov(A, X2)  = ḡa                    Var(X1)  = a² + σ²
    Cov(A, X1)  = a                     Var(A)   = 1
    Cov(X1, X2) = ḡ(a² + σ²) = ḡ·Var(X1)

    Cov(A, X2 | X1) = Cov(A,X2) − Cov(A,X1)·Cov(X1,X2)/Var(X1)
                    = ḡa − a·ḡ = 0     ← identically zero, all parameters

The interaction-carried dependence γ·A·X1 is orthogonal to A at first order,
so a LINEAR lens conditioning on X1 sees nothing left. The pinned Fisher-z
test is not malfunctioning: it correctly reports that no *linear* conditional
dependence remains.

**Corroboration statistic (LD-2 formula).**

    r_op ≈ Φ( √(N_disc − |S| − 3) · |atanh ρ_op| − z_{0.975} )
         = Φ( √(4000 − 1 − 3) · |atanh 0.000000000| − 1.959964 )
         = Φ( −1.959964 ) = **0.025000**

At ρ_op = 0 the registered one-tail form returns α/2 = 0.0250; the
exact two-sided size of the same test is α = 0.05. Both readings sit in the
SAME fork zone by more than an order of magnitude, so the distinction moves
no label.

**Stored separating sets.** The manifests store NO PC separating sets (the `discovery` block carries library, resolved args, canonical order, edges, adjacency and dataset identity only), so both candidate subsets are covered above rather than deferring to a stored one.

### TRUTH-SIDE FORK — ZONE: **GENUINE**

r_op = 0.025000. The zone applied EXACTLY as committed in LD-2:

> **GENUINE (class d)** iff r_op ≤ 0.5: the pinned linear test, correctly applying its linear lens, more-likely-than-not drops an interaction-carried edge; the systematic 20/20 SHD = 1 is regime-induced skeleton misspecification, the halt LIFTS, the edge loss is reported as a finding (H3's misspecification mechanism arriving via the interaction/regime channel rather than the predicted nonlinearity/family channel).

**BRANCH: the triangle/linear/effect_modifying halt **LIFTS**.**

**Sanity note (reported only).** Observed drop rate of the A—X2 adjacency:
20/20 seeds. Per-seed probability the test ACCEPTS independence (i.e. drops
the edge) = 1 − r_op = 0.975000; probability all 20
independent seeds drop it = 0.6027 (= 0.95^20 = 0.3585 under the exact two-sided size).
The observed 20/20 and r_op are **mutually consistent**: a test that accepts
independence ~95–97.5% of the time is expected to drop the edge on all 20
seeds a large fraction of the time.

---

## Step 3a — reclassification under the LD-2 taxonomy

**Class definitions (PS-8, as corrected by LD-2):**

- **class (a)** (LD-2) — "config/estimator/code signature, or systematic unexplained adjacency error — HARNESS alarm." Limb a3 (perfect graph ⟹ ΔCost_ld = 0) is restricted by the diagnostic-taxonomy correction to ADDITIVE cells, "where the note-(c) identity licenses it."
- **class (b)** (PS-8 per-seed class-(a)/(b) rule) — "only for a wrong tie-break-channel orientation." A class-(b) call on a linear cell contradicts the PS-7 note (f)/(g) structural guarantee and is escalated.
- **class (c)** (LD-2) — "A violating seed on an EM cell with skeleton_shd = 0, orientation_wrong_count = 0, and region-3 placement is classified **class (c): registered interaction-blindness channel per PS-7** — the discovered rung's group-agnostic form prescribes cheaper pooled actions; the register-logged consequence, not an alarm; **no halt**."
- **class (d)** (LD-2) — "systematic adjacency error traceable by a truth-side check to DGP-driven CI-misspecification under the pinned linear test — genuine discovery behaviour on the regime channel, NOT an alarm, NOT class (a)."
- **sporadic-finite-sample** (LD-2) — "**sporadic finite-sample adjacency noise** — isolated seeds, non-systematic (e.g. chain's 2-seed spurious X1→X3) — reported, NOT an alarm, NOT class (a)."
- **unclassified** — no defined class covers this seed. Reported, never folded into class (a); an occurrence requires a follow-up adjudication.

**Halt rule (LD-2):** HARNESS HALT iff class (a) is a majority of violating seeds; (c), (d) and sporadic-finite-sample are not (a).

**a1 fracture input.** An instance's adjacency error is read as SYSTEMATIC
when a majority of its seeds carry `skeleton_shd > 0`, and as isolated
otherwise. On this grid the two cases are 20/20 (triangle/linear/EM) and
2/20 (chain/linear), so every cut strictly between them yields identical
labels — the split carries no discretion here.

| topology | family | regime | denominator | region_1_count | violating | seeds_shd_positive | systematic | fork_zone | class_breakdown | halt |
|---|---|---|---|---|---|---|---|---|---|---|
| chain | linear | additive | fourway | 19 | 0 | 2 | False | — | — | — |
| chain | linear | effect_modifying | fourway | 8 | 12 | 2 | False | — | 11 c + 1 sporadic | LIFTS |
| collider | linear | additive | fourway | 20 | 0 | 0 | False | — | — | — |
| collider | linear | effect_modifying | fourway | 0 | 20 | 0 | False | — | 20 c | LIFTS |
| triangle | linear | additive | fourway | 20 | 0 | 0 | False | — | — | — |
| triangle | linear | effect_modifying | fourway | 0 | 20 | 20 | True | GENUINE | 20 d | LIFTS |

- **chain / linear / additive** — linear control clean — interpolation held, diagnostic not triggered (region-1 count 19/20).
- **chain / linear / effect_modifying** — diagnostic TRIGGERED (region-1 count 8/20 < 16); violating seeds classify 11 c + 1 sporadic → HALT LIFTS — class (a) is 0/12 violating seeds, not a majority.
  - class c seeds: [0, 1, 6, 8, 9, 10, 11, 12, 13, 17, 19]
  - class sporadic seeds: [16]
- **collider / linear / additive** — linear control clean — interpolation held, diagnostic not triggered (region-1 count 20/20).
- **collider / linear / effect_modifying** — diagnostic TRIGGERED (region-1 count 0/20 < 16); violating seeds classify 20 c → HALT LIFTS — class (a) is 0/20 violating seeds, not a majority.
  - class c seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
- **triangle / linear / additive** — linear control clean — interpolation held, diagnostic not triggered (region-1 count 20/20).
- **triangle / linear / effect_modifying** — diagnostic TRIGGERED (region-1 count 0/20 < 16); violating seeds classify 20 d → HALT LIFTS — class (a) is 0/20 violating seeds, not a majority.
  - class d seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

**Halts LIFTED: 3** — chain/linear/effect_modifying, collider/linear/effect_modifying, triangle/linear/effect_modifying
**Halts STANDING: 0** — none

### Step 3b — propagation under the standing halt(s) (PS-8)

- `orientation_pairs`: — none —
- `q2_linear_controls`: — none —
- `corroborating_pairs`: — none —

Every halt-scope (PS-8) invalidation recorded in the superseded run is
LIFTED: the collider orientation-channel pairs, the collider
worse-than-no-graph linear control and fallback-tier control, and the
chain/triangle corroborating
pairs are all re-enabled.

---

## Step 3c — H3a interaction, orientation channel (PS-8)

The two collider pairs (× additive, × effect-modifying). Per-seed event =
`dcost_ld(NLG, s) − dcost_ld(linear, s) > 0` **strictly**, paired by
`seed_idx` (CRN); a zero difference is non-firing (PS-8). The seed
counts are FROZEN — banded here, not recomputed.

| topology | regime | denominator | n_seeds | interaction_event_count | zero_difference_count | mean_interaction_diff | gate_pass |
|---|---|---|---|---|---|---|---|
| collider | additive | fourway | 20 | 0 | 20 | +0.000000 | False |
| collider | effect_modifying | fourway | 20 | 19 | 0 | +0.284746 | True |

### VERDICT: **Mixed (1/2)** — 1/2 pairs pass the 16/20 seed gate

#### MANDATORY ANNOTATION (PS-8) — accompanies this label

(i) **Region occupancy per family arm** — the pair that carries the firing:

- **collider / linear / effect_modifying** — seeds predominantly occupied **region 3** (20/20); mean dcost_ld = -0.442922 (**negative** = L1-discovered CHEAPER than L1-oracle).
- **collider / nlg / effect_modifying** — seeds predominantly occupied **region 3** (20/20); mean dcost_ld = -0.158176 (**negative** = L1-discovered CHEAPER than L1-oracle).

(ii) **Sign direction relative to H3's prediction.** H3 predicts a discovery
PENALTY (L1-discovered costlier). Both arms of the firing pair sit in region 3
(cheaper than the oracle), so the interaction difference is a difference of
MAGNITUDES in the cheaper direction — **opposite** to the predicted direction.

This label is produced from region-3 differentials and therefore carries the
PS-8 clause **VERBATIM**:

> *"interaction reflects form-blindness magnitude in the region-3 (cheaper) direction, opposite to H3's predicted discovery-penalty direction; cannot be read as a discovery-penalty differential."*

Per PS-8 this annotation discharges its standing 'reported
explicitly, never folded into holds' instruction and the region partition's region-3
inspect flag; it moves no gate, band, or count.

**Structure-conditioned reading (PS-8, structural, pre-stated).** The
interaction is not a flat six-pair count: the collider is the only one of the
three topologies whose graph admits any CI-based orientation, so it alone carries
the orientation channel; chain and triangle corroborate on the
skeleton/adjacency channel and cannot gate it.

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

Region 2 is EMPTY on every instance of this grid. Region-3 seeds are counted
in **neither** the numerator **nor** removed from the denominator; their
per-instance tally is the `region_3_count` column above.

**Linear control arm beside the NLG verdict (no band applied):**

| topology | regime | denominator | region_1_count | region_2_count | region_3_count |
|---|---|---|---|---|---|
| chain | additive | fourway | 19 | 0 | 1 |
| chain | effect_modifying | fourway | 8 | 0 | 12 |
| collider | additive | fourway | 20 | 0 | 0 |
| collider | effect_modifying | fourway | 0 | 0 | 20 |
| triangle | additive | fourway | 20 | 0 | 0 |
| triangle | effect_modifying | fourway | 0 | 0 | 20 |

**Control-arm invalidation (PS-8 halt-scope rule): none.** No halt stands, so
the collider linear control discharges its worse-than-no-graph and fallback-tier role.

---

## Step 3e — control-arm status, all six linear instances

Additive cells read "clean" exactly as in the superseded run. An EM cell is
NEVER reported as "clean" and NEVER as a "harness bug" for a non-(a)
channel: the registered channel is stated explicitly.

| topology | family | regime | denominator | region_1_count | status | registered_channel |
|---|---|---|---|---|---|---|
| chain | linear | additive | fourway | 19 | clean | clean — interpolation held, diagnostic not triggered |
| chain | linear | effect_modifying | fourway | 8 | registered channel | class (c) registered form-blindness channel (PS-7) on 11 seed(s); sporadic finite-sample adjacency noise on 1 seed(s) — halt LIFTS (no class-(a) majority) |
| collider | linear | additive | fourway | 20 | clean | clean — interpolation held, diagnostic not triggered |
| collider | linear | effect_modifying | fourway | 0 | registered channel | class (c) registered form-blindness channel (PS-7) on 20 seed(s) — halt LIFTS (no class-(a) majority) |
| triangle | linear | additive | fourway | 20 | clean | clean — interpolation held, diagnostic not triggered |
| triangle | linear | effect_modifying | fourway | 0 | registered channel | class (d) regime-induced skeleton misspecification (zone GENUINE) on 20 seed(s) — halt LIFTS (no class-(a) majority) |

---

## Step 3f — H3b NLG-existence fallback tier (PS-8, compute-once seam)

3c is **Mixed**, so the fallback tier's invocation condition **is
satisfied**. It is discharged by CROSS-REFERENCE to the single banded
result at `#worse-than-no-graph-co-primary--nlg-region-2-banded-verdict`; the counts are **not** reprinted here, per
PS-8's compute-once seam.

**Triangle propagation note (verified).** A triangle-linear halt propagates to
the `corroborating_pairs` slot only — it does not touch the collider pairs — so
a standing triangle halt could not have halted 3c. On this re-emission the
triangle halt lifts in any case, so the question is moot and v1's 3g handling
is not invoked.

---

## CORROBORATING — NON-GATING (PS-8)

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

| topology | regime | denominator | n_seeds | interaction_event_count | zero_difference_count |
|---|---|---|---|---|---|
| chain | additive | fourway | 20 | 2 | 18 |
| chain | effect_modifying | fourway | 20 | 14 | 0 |
| triangle | additive | fourway | 20 | 0 | 20 |
| triangle | effect_modifying | fourway | 20 | 20 | 0 |

Per the registered symmetric no-consequence clause, whichever way these
numbers point they move no label, no gate and no band.

---

## Diagnostic covariates per instance (PS-8) — frozen, re-tabled

Covariates below are governed by the orientation-covariate split and the nSID normalization rule (PS-8).

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

**Descriptive extras (frozen).**

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

## Discussion obligation (LD-2)

> the family channel (Fisher-z degrading NLG skeletons) returned empty on
> this grid — skeleton-SHD(NLG) ≤ skeleton-SHD(linear) on all corroborating
> cells, nSID = 0 everywhere — and the regime channel (form-blindness under
> EM, with group-asymmetric validity collapse) is the live finding; the two
> statements are kept separable: what was predicted and nulled vs. what was
> found under a register-logged consequence clause (PS-7).

No hypothesis or assumption changes. No threshold, band, gate numerator, k,
or region definition moved in this re-emission.

