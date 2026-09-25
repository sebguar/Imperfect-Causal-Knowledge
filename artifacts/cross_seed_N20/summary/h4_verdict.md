# H4 verdict — mechanical PS-9 reading at N=20

Produced by `icknowledge/analysis/h4.py`. The label below is the OUTPUT of a
pure function applying **PS-9**, as operationalized in its decisions, to
frozen numbers — it is not narrated, and no threshold in it was chosen after
the numbers existed.

- **ΔΔB(c)** := ΔB_{−1}(c) − ΔB_{+1}(c), on the **common-found**
  population (`ΔB_g_vs_L2_common_found`).
- **Per-seed gate:** an EM instance shows the predicted sign at rung
  c iff ≥ **16/20** seeds satisfy sign(ΔS·ΔΔB) > 0.
- **Instance tally (PS-9):** ≥5/6 Supported · 4/6 Mixed · ≤3/6 Null.
- **SD is descriptive only**; no SE, no CI (the inference stance).
- L2 is the reference rung and carries no ΔB (the NaN convention); it never enters an
  aggregation here.

---

## Rung L1-oracle

### VERDICT: **Null**

- 0/6 instances pass the 16/20 seed gate → Null

#### Two-level tally — the six effect-modifying instances

| topology | family | delta_S | mean_ddB | sd_ddB | n_seeds_neg | n_seeds_pos | n_seeds_sign_agree | sign_gate_pass |
|---|---|---|---|---|---|---|---|---|
| chain | linear | -0.476973 | -0.003377 | +0.027053 | 11 | 9 | 11 | **NO** |
| chain | nlg | -0.342228 | -0.003111 | +0.023282 | 10 | 10 | 10 | **NO** |
| collider | linear | -0.404882 | +0.029080 | +0.078678 | 7 | 13 | 7 | **NO** |
| collider | nlg | -0.299159 | +0.023555 | +0.108186 | 8 | 12 | 8 | **NO** |
| triangle | linear | -0.600000 | -0.003065 | +0.007831 | 13 | 6 | 13 | **NO** |
| triangle | nlg | -0.185335 | +0.000059 | +0.005910 | 9 | 10 | 9 | **NO** |

**0/6 instances pass** the 16/20 seed gate.

Failing instance(s): chain/linear, chain/nlg, collider/linear, collider/nlg, triangle/linear, triangle/nlg.

**Outlier-discrimination clause:** not triggered (6 instance(s) failing; the clause reads when exactly one does).

#### Additive-control count read (absence vs. attenuation)

| topology | family | mean_ddB | sd_ddB | n_seeds_neg | n_seeds_pos | control_read |
|---|---|---|---|---|---|---|
| chain | linear | +0.002841 | +0.029492 | 11 | 9 | ABSENCE (flat within seed noise): 11 negative / 9 positive, 9 crossing(s) |
| chain | nlg | +0.001332 | +0.023933 | 8 | 12 | ABSENCE (flat within seed noise): 8 negative / 12 positive, 8 crossing(s) |
| collider | linear | +0.018732 | +0.076672 | 9 | 11 | ABSENCE (flat within seed noise): 9 negative / 11 positive, 9 crossing(s) |
| collider | nlg | +0.019495 | +0.089597 | 8 | 12 | ABSENCE (flat within seed noise): 8 negative / 12 positive, 8 crossing(s) |
| triangle | linear | +0.007168 | +0.028693 | 9 | 11 | ABSENCE (flat within seed noise): 9 negative / 11 positive, 9 crossing(s) |
| triangle | nlg | +0.000177 | +0.000959 | 10 | 9 | ABSENCE (flat within seed noise): 10 negative / 9 positive, 9 crossing(s) |

#### Regime contrast (per topology — the level PS-9 reads at)

| topology | family | abs_mean_ddB_treatment | abs_mean_ddB_control | ratio_treatment_over_control | contrast_holds | saturated_both_arms |
|---|---|---|---|---|---|---|
| chain | ALL | +0.003244 | +0.002087 | +1.554777 | YES | YES |
| chain | linear | +0.003377 | +0.002841 | +1.188784 | YES | YES |
| chain | nlg | +0.003111 | +0.001332 | +2.335069 | YES | YES |
| collider | ALL | +0.026317 | +0.019114 | +1.376902 | YES | **NO** |
| collider | linear | +0.029080 | +0.018732 | +1.552389 | YES | **NO** |
| collider | nlg | +0.023555 | +0.019495 | +1.208280 | YES | **NO** |
| triangle | ALL | +0.001562 | +0.003672 | +0.425394 | **NO** | YES |
| triangle | linear | +0.003065 | +0.007168 | +0.427592 | **NO** | YES |
| triangle | nlg | +0.000059 | +0.000177 | +0.336214 | **NO** | YES |

No contrast failure counted. triangle: strict inequality does NOT hold, but both arms are saturated-tiny (|mean ΔΔB| < 0.02), so PS-9's contrast-on-saturated-topology note reads it as directionally-correct-but-attenuated rather than as a failure. The remaining topologies satisfy the strict inequality.

#### Supporting correlation (the inference stance / PS-8)

Pearson r over the **six** (ΔS, mean ΔΔB) effect-modifying points: **r = +0.2112**.

Within-topology difference form — six points, one per effect-modifying SCM
instance — never twelve raw (S(g), ΔB_g) points, because both axes are
topology-scaled (PS-8). **Supporting evidence only (the inference stance):** it enters no
threshold and no verdict branch.

---

## Rung L0

### VERDICT: **Null**

- 3/6 instances pass the 16/20 seed gate → Null
- contrast fails on 2 topologies (collider, chain) → forced Null (PS-9 Null clause)

#### Two-level tally — the six effect-modifying instances

| topology | family | delta_S | mean_ddB | sd_ddB | n_seeds_neg | n_seeds_pos | n_seeds_sign_agree | sign_gate_pass |
|---|---|---|---|---|---|---|---|---|
| chain | linear | -0.476973 | -0.002519 | +0.077962 | 11 | 9 | 11 | **NO** |
| chain | nlg | -0.342228 | -0.037047 | +0.029440 | 18 | 2 | 18 | YES |
| collider | linear | -0.404882 | +0.296994 | +0.237684 | 4 | 16 | 4 | **NO** |
| collider | nlg | -0.299159 | +0.147938 | +0.083088 | 1 | 19 | 1 | **NO** |
| triangle | linear | -0.600000 | -0.211893 | +0.010395 | 20 | 0 | 20 | YES |
| triangle | nlg | -0.185335 | -0.018525 | +0.003648 | 20 | 0 | 20 | YES |

**3/6 instances pass** the 16/20 seed gate.

Failing instance(s): chain/linear, collider/linear, collider/nlg.

**Outlier-discrimination clause:** not triggered (3 instance(s) failing; the clause reads when exactly one does).

#### Additive-control count read (absence vs. attenuation)

| topology | family | mean_ddB | sd_ddB | n_seeds_neg | n_seeds_pos | control_read |
|---|---|---|---|---|---|---|
| chain | linear | +0.105990 | +0.054308 | 1 | 19 | ATTENUATION-CANDIDATE (non-flat): 19/20 seeds positive, 1 crossing(s) |
| chain | nlg | +0.048351 | +0.024991 | 1 | 19 | ATTENUATION-CANDIDATE (non-flat): 19/20 seeds positive, 1 crossing(s) |
| collider | linear | +0.894220 | +0.208539 | 0 | 20 | ATTENUATION-CANDIDATE (non-flat): 20/20 seeds positive, 0 crossing(s) |
| collider | nlg | +0.378000 | +0.062074 | 0 | 20 | ATTENUATION-CANDIDATE (non-flat): 20/20 seeds positive, 0 crossing(s) |
| triangle | linear | +0.181463 | +0.102101 | 2 | 18 | ATTENUATION-CANDIDATE (non-flat): 18/20 seeds positive, 2 crossing(s) |
| triangle | nlg | -0.001218 | +0.000526 | 20 | 0 | ATTENUATION-CANDIDATE (non-flat): 20/20 seeds negative, 0 crossing(s) |

#### Regime contrast (per topology — the level PS-9 reads at)

| topology | family | abs_mean_ddB_treatment | abs_mean_ddB_control | ratio_treatment_over_control | contrast_holds | saturated_both_arms |
|---|---|---|---|---|---|---|
| chain | ALL | +0.019783 | +0.077170 | +0.256350 | **NO** | **NO** |
| chain | linear | +0.002519 | +0.105990 | +0.023762 | **NO** | **NO** |
| chain | nlg | +0.037047 | +0.048351 | +0.766212 | **NO** | **NO** |
| collider | ALL | +0.222466 | +0.636110 | +0.349729 | **NO** | **NO** |
| collider | linear | +0.296994 | +0.894220 | +0.332126 | **NO** | **NO** |
| collider | nlg | +0.147938 | +0.378000 | +0.391371 | **NO** | **NO** |
| triangle | ALL | +0.115209 | +0.091340 | +1.261317 | YES | **NO** |
| triangle | linear | +0.211893 | +0.181463 | +1.167694 | YES | **NO** |
| triangle | nlg | +0.018525 | +0.001218 | +15.210026 | YES | YES |

Contrast FAILS on: collider, chain.

#### Supporting correlation (the inference stance / PS-8)

Pearson r over the **six** (ΔS, mean ΔΔB) effect-modifying points: **r = +0.3761**.

Within-topology difference form — six points, one per effect-modifying SCM
instance — never twelve raw (S(g), ΔB_g) points, because both axes are
topology-scaled (PS-8). **Supporting evidence only (the inference stance):** it enters no
threshold and no verdict branch.

---

## Signal-2 discriminator reading (PS-9)

**6 additive-control (instance, rung) reading(s) are
NON-FLAT** (attenuation candidates). Per the discriminator rule this
makes the
PS-9 Signal-2 channel a **LIVE alternative explanation** for ΔΔB —
estimation error on the unmodified, regime-shared downstream edge
`coef_error[X₂→X₃]` would move BOTH arms, which is what a non-flat control
looks like. It **must be named in the H4 write-up**, not left in the
exploratory appendix.

| condition | topology | family | mean_ddB | control_read |
|---|---|---|---|---|
| L0 | chain | linear | +0.105990 | ATTENUATION-CANDIDATE (non-flat): 19/20 seeds positive, 1 crossing(s) |
| L0 | chain | nlg | +0.048351 | ATTENUATION-CANDIDATE (non-flat): 19/20 seeds positive, 1 crossing(s) |
| L0 | collider | linear | +0.894220 | ATTENUATION-CANDIDATE (non-flat): 20/20 seeds positive, 0 crossing(s) |
| L0 | collider | nlg | +0.378000 | ATTENUATION-CANDIDATE (non-flat): 20/20 seeds positive, 0 crossing(s) |
| L0 | triangle | linear | +0.181463 | ATTENUATION-CANDIDATE (non-flat): 18/20 seeds positive, 2 crossing(s) |
| L0 | triangle | nlg | -0.001218 | ATTENUATION-CANDIDATE (non-flat): 20/20 seeds negative, 0 crossing(s) |

---

## Watch-fors

### common-found vs. all-eligible

`N_common_found == N_eligible` in every (cell, seed, rung, group). The
the common-found population N=6 no-op finding **still holds at N=20** — the common-found
population changes no number here, though it remains the primary
population by the common-found population regardless.

### Per-seed ΔΔB bimodality

No (cell, rung) shows a per-seed ΔΔB gap wider than its own SD with ≥3
seeds either side.
