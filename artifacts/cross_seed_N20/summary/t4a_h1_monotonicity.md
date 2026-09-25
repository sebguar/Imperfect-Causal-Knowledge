# H1 monotonicity and the PS-5 criterion read (N=20, 12 cells)

DV: realized cost over the **common-found three-way** population
(the common-found population primary). Mean ± SD over the 20 seeds of this grid.

## SUPERSEDES

- `results/cross_seed_N6/summary/t4a_h1_monotonicity.md`

Those artifacts are **audit objects** and are NOT overwritten. They record the **PS-1 early look** — 8 cells × 6 seeds, before the chain joined the grid at the chain SCM specification. This artifact is the **full registered scope**: 12 cells × 20 seeds. Where the two disagree, the disagreement is reported below rather than resolved in favour of the earlier read.

## Outcome exposure — stated, not claimed away

**PS-5 was register-logged prior to outcome exposure.** It is quoted below as it now stands. No threshold, band, quantifier or population in it was moved for this read: every one applied here is the one on record before these numbers were computed.

> a monotone ordering L2 ≤ L1-oracle ≤ L0 with non-overlapping variability on at least one of the three topologies

**This application of it is OUTCOME-EXPOSED.** The numbers existed before this document was generated, and the choice to compute the read now, on this tree, was made with the N=6 early look already known. That is disclosed rather than described away: this is **not** an outcome-independent read, and nothing here should be cited as one. What is pre-committed is the RULE; what is post-data is the OCCASION of applying it.

## 1. Common-found realized cost, mean ± SD over seeds

| topology | family | regime | group | L0_mean | L1-oracle_mean | L2_mean | L0_sd | L1-oracle_sd | L2_sd |
|---|---|---|---|---|---|---|---|---|---|
| chain | linear | additive | -1.000000 | 2.840650 | 2.705795 | 2.702559 | 0.116620 | 0.138563 | 0.138833 |
| chain | linear | additive | 1.000000 | 0.494046 | 0.462341 | 0.461945 | 0.024619 | 0.023917 | 0.026083 |
| chain | linear | effect_modifying | -1.000000 | 2.811575 | 2.735448 | 2.735986 | 0.109861 | 0.158619 | 0.156261 |
| chain | linear | effect_modifying | 1.000000 | 0.548527 | 0.473258 | 0.470420 | 0.021160 | 0.025990 | 0.029207 |
| chain | nlg | additive | -1.000000 | 1.803254 | 1.728530 | 1.726519 | 0.064618 | 0.082742 | 0.079419 |
| chain | nlg | additive | 1.000000 | 0.361614 | 0.333908 | 0.333229 | 0.015762 | 0.015488 | 0.016481 |
| chain | nlg | effect_modifying | -1.000000 | 1.790390 | 1.752685 | 1.753383 | 0.062018 | 0.083570 | 0.083134 |
| chain | nlg | effect_modifying | 1.000000 | 0.421375 | 0.349735 | 0.347322 | 0.018490 | 0.018562 | 0.020200 |
| collider | linear | additive | -1.000000 | 6.611072 | 5.622292 | 5.603509 | 0.465127 | 0.303240 | 0.299646 |
| collider | linear | additive | 1.000000 | 0.918278 | 0.804986 | 0.804936 | 0.078550 | 0.056280 | 0.057988 |
| collider | linear | effect_modifying | -1.000000 | 6.555235 | 6.113763 | 6.084822 | 0.479900 | 0.507087 | 0.499329 |
| collider | linear | effect_modifying | 1.000000 | 0.967542 | 0.793984 | 0.794123 | 0.090540 | 0.045187 | 0.045070 |
| collider | nlg | additive | -1.000000 | 2.755876 | 2.305526 | 2.284299 | 0.194666 | 0.236165 | 0.213822 |
| collider | nlg | additive | 1.000000 | 0.509956 | 0.418111 | 0.416378 | 0.062918 | 0.056950 | 0.056988 |
| collider | nlg | effect_modifying | -1.000000 | 2.753716 | 2.506597 | 2.481166 | 0.204799 | 0.290100 | 0.255767 |
| collider | nlg | effect_modifying | 1.000000 | 0.526000 | 0.403263 | 0.401388 | 0.064754 | 0.052708 | 0.048937 |
| triangle | linear | additive | -1.000000 | 4.654844 | 4.451871 | 4.443819 | 0.072161 | 0.117960 | 0.116778 |
| triangle | linear | additive | 1.000000 | 0.760203 | 0.731526 | 0.730641 | 0.029618 | 0.027612 | 0.027313 |
| triangle | linear | effect_modifying | -1.000000 | 3.693393 | 3.693388 | 3.693390 | 0.086451 | 0.086451 | 0.086453 |
| triangle | linear | effect_modifying | 1.000000 | 0.975631 | 0.766798 | 0.763735 | 0.042083 | 0.031079 | 0.031846 |
| triangle | nlg | additive | -1.000000 | 3.997509 | 3.997363 | 3.997477 | 0.070132 | 0.069963 | 0.070107 |
| triangle | nlg | additive | 1.000000 | 0.653998 | 0.652457 | 0.652748 | 0.027883 | 0.027245 | 0.027730 |
| triangle | nlg | effect_modifying | -1.000000 | 2.928177 | 2.928126 | 2.928174 | 0.078898 | 0.078824 | 0.078898 |
| triangle | nlg | effect_modifying | 1.000000 | 0.756423 | 0.737787 | 0.737895 | 0.033087 | 0.030962 | 0.032367 |

## 2. The common-found no-op — reported as a FINDING, not a caveat

Rows checked: **1440** (12 cells × 3 conditions × 2 groups × 20 seeds).

- `max |N_common_found_threeway − N_eligible|` = **0**
- `max |N_found − N_eligible|` = **0**
- `max |cost(common-found) − cost(all-eligible)|` = **0.000e+00**
- exact equality across every row: **YES**

**Substantive reading.** Found is complete under exhaustive grid search on
this DGP, so the common-found primary coincides *exactly* with the
all-eligible legacy read. This is a statement about the design, not a
technicality: **recourse failure here is always VALIDITY failure, never
FINDABILITY failure** — the brute-force procedure always returns a feasible
action; that action may fail to flip the true classifier, but it is never
absent. The common-found population's machinery is correct and binding in principle; it
resolves to equality on this grid because the DGP makes findability trivial.

## 3. H1 signed monotonicity — per cell × group (signed primary)

Expected: `cost(L2) ≤ cost(L1-oracle) ≤ cost(L0)`. `diff = lower − upper`;
the rung holds when `diff_mean ≤ 0`. `within_seed_noise` flags a rung whose
`|mean|` does not exceed its seed SD.

| topology | family | regime | group | rung | diff_mean | diff_sd | holds_on_mean | sign_consistent_across_seeds | n_seeds_holding | abs_mean_over_sd | within_seed_noise |
|---|---|---|---|---|---|---|---|---|---|---|---|
| triangle | linear | additive | -1.000000 | L1-oracle<=L0 | -0.202972 | 0.124463 | YES | NO | 20 | 1.630782 | NO |
| triangle | linear | additive | -1.000000 | L2<=L1-oracle | -0.008053 | 0.032160 | YES | NO | 11 | 0.250393 | YES |
| triangle | linear | additive | 1.000000 | L1-oracle<=L0 | -0.028677 | 0.015106 | YES | YES | 20 | 1.898460 | NO |
| triangle | linear | additive | 1.000000 | L2<=L1-oracle | -0.000885 | 0.003573 | YES | NO | 11 | 0.247655 | YES |
| triangle | linear | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.000005 | 0.000008 | YES | NO | 20 | 0.606844 | YES |
| triangle | linear | effect_modifying | -1.000000 | L2<=L1-oracle | 0.000002 | 0.000006 | NO | NO | 17 | 0.289005 | YES |
| triangle | linear | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.208833 | 0.014467 | YES | YES | 20 | 14.434644 | NO |
| triangle | linear | effect_modifying | 1.000000 | L2<=L1-oracle | -0.003063 | 0.007829 | YES | NO | 14 | 0.391274 | YES |
| triangle | nlg | additive | -1.000000 | L1-oracle<=L0 | -0.000147 | 0.000437 | YES | NO | 20 | 0.335632 | YES |
| triangle | nlg | additive | -1.000000 | L2<=L1-oracle | 0.000114 | 0.000388 | NO | NO | 14 | 0.294386 | YES |
| triangle | nlg | additive | 1.000000 | L1-oracle<=L0 | -0.001541 | 0.001654 | YES | YES | 20 | 0.931664 | YES |
| triangle | nlg | additive | 1.000000 | L2<=L1-oracle | 0.000291 | 0.001313 | NO | NO | 11 | 0.221517 | YES |
| triangle | nlg | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.000051 | 0.000202 | YES | NO | 20 | 0.253070 | YES |
| triangle | nlg | effect_modifying | -1.000000 | L2<=L1-oracle | 0.000048 | 0.000203 | NO | NO | 15 | 0.237354 | YES |
| triangle | nlg | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.018636 | 0.006881 | YES | YES | 20 | 2.708337 | NO |
| triangle | nlg | effect_modifying | 1.000000 | L2<=L1-oracle | 0.000108 | 0.005805 | NO | NO | 10 | 0.018525 | YES |
| collider | linear | additive | -1.000000 | L1-oracle<=L0 | -0.988780 | 0.243292 | YES | YES | 20 | 4.064176 | NO |
| collider | linear | additive | -1.000000 | L2<=L1-oracle | -0.018783 | 0.082581 | YES | NO | 11 | 0.227448 | YES |
| collider | linear | additive | 1.000000 | L1-oracle<=L0 | -0.113292 | 0.031951 | YES | YES | 20 | 3.545825 | NO |
| collider | linear | additive | 1.000000 | L2<=L1-oracle | -0.000051 | 0.006902 | YES | NO | 12 | 0.007338 | YES |
| collider | linear | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.441472 | 0.235516 | YES | YES | 20 | 1.874488 | NO |
| collider | linear | effect_modifying | -1.000000 | L2<=L1-oracle | -0.028941 | 0.083273 | YES | NO | 13 | 0.347547 | YES |
| collider | linear | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.173558 | 0.054597 | YES | YES | 20 | 3.178860 | NO |
| collider | linear | effect_modifying | 1.000000 | L2<=L1-oracle | 0.000138 | 0.010669 | NO | NO | 13 | 0.012965 | YES |
| collider | nlg | additive | -1.000000 | L1-oracle<=L0 | -0.450350 | 0.114194 | YES | YES | 20 | 3.943714 | NO |
| collider | nlg | additive | -1.000000 | L2<=L1-oracle | -0.021227 | 0.104065 | YES | NO | 12 | 0.203980 | YES |
| collider | nlg | additive | 1.000000 | L1-oracle<=L0 | -0.091845 | 0.017409 | YES | YES | 20 | 5.275765 | NO |
| collider | nlg | additive | 1.000000 | L2<=L1-oracle | -0.001732 | 0.014694 | YES | NO | 12 | 0.117895 | YES |
| collider | nlg | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.247119 | 0.141465 | YES | YES | 20 | 1.746858 | NO |
| collider | nlg | effect_modifying | -1.000000 | L2<=L1-oracle | -0.025431 | 0.117161 | YES | NO | 12 | 0.217058 | YES |
| collider | nlg | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.122736 | 0.026124 | YES | YES | 20 | 4.698154 | NO |
| collider | nlg | effect_modifying | 1.000000 | L2<=L1-oracle | -0.001875 | 0.019638 | YES | NO | 11 | 0.095504 | YES |
| chain | linear | additive | -1.000000 | L1-oracle<=L0 | -0.134855 | 0.060390 | YES | YES | 20 | 2.233070 | NO |
| chain | linear | additive | -1.000000 | L2<=L1-oracle | -0.003236 | 0.032870 | YES | NO | 9 | 0.098447 | YES |
| chain | linear | additive | 1.000000 | L1-oracle<=L0 | -0.031705 | 0.009641 | YES | YES | 20 | 3.288599 | NO |
| chain | linear | additive | 1.000000 | L2<=L1-oracle | -0.000395 | 0.003848 | YES | NO | 10 | 0.102693 | YES |
| chain | linear | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.076127 | 0.084433 | YES | NO | 20 | 0.901620 | YES |
| chain | linear | effect_modifying | -1.000000 | L2<=L1-oracle | 0.000538 | 0.028080 | NO | NO | 11 | 0.019168 | YES |
| chain | linear | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.075268 | 0.013185 | YES | YES | 20 | 5.708451 | NO |
| chain | linear | effect_modifying | 1.000000 | L2<=L1-oracle | -0.002839 | 0.006945 | YES | NO | 12 | 0.408741 | YES |
| chain | nlg | additive | -1.000000 | L1-oracle<=L0 | -0.074724 | 0.042912 | YES | YES | 20 | 1.741340 | NO |
| chain | nlg | additive | -1.000000 | L2<=L1-oracle | -0.002011 | 0.027653 | YES | NO | 12 | 0.072728 | YES |
| chain | nlg | additive | 1.000000 | L1-oracle<=L0 | -0.027706 | 0.007638 | YES | YES | 20 | 3.627400 | NO |
| chain | nlg | additive | 1.000000 | L2<=L1-oracle | -0.000679 | 0.003936 | YES | NO | 12 | 0.172401 | YES |
| chain | nlg | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.037705 | 0.047302 | YES | YES | 20 | 0.797106 | YES |
| chain | nlg | effect_modifying | -1.000000 | L2<=L1-oracle | 0.000699 | 0.026359 | NO | NO | 10 | 0.026514 | YES |
| chain | nlg | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.071640 | 0.014890 | YES | YES | 20 | 4.811326 | NO |
| chain | nlg | effect_modifying | 1.000000 | L2<=L1-oracle | -0.002413 | 0.007282 | YES | NO | 12 | 0.331308 | YES |

## 4. Same ladder on the cell-level Δ_cost (common-found)

| topology | family | regime | rung | diff_mean | diff_sd | holds_on_mean | sign_consistent_across_seeds | abs_mean_over_sd | within_seed_noise |
|---|---|---|---|---|---|---|---|---|---|
| triangle | linear | additive | L1-oracle<=L0 | -0.174295 | 0.109553 | YES | NO | 1.590965 | NO |
| triangle | linear | additive | L2<=L1-oracle | -0.007168 | 0.028693 | YES | NO | 0.249810 | YES |
| triangle | linear | effect_modifying | L1-oracle<=L0 | 0.208828 | 0.014471 | NO | YES | 14.430936 | NO |
| triangle | linear | effect_modifying | L2<=L1-oracle | 0.003065 | 0.007831 | NO | NO | 0.391397 | YES |
| triangle | nlg | additive | L1-oracle<=L0 | 0.001395 | 0.001249 | NO | YES | 1.116726 | NO |
| triangle | nlg | additive | L2<=L1-oracle | -0.000177 | 0.000959 | YES | NO | 0.184146 | YES |
| triangle | nlg | effect_modifying | L1-oracle<=L0 | 0.018585 | 0.006932 | NO | YES | 2.680976 | NO |
| triangle | nlg | effect_modifying | L2<=L1-oracle | -0.000059 | 0.005910 | YES | NO | 0.010050 | YES |
| collider | linear | additive | L1-oracle<=L0 | -0.875488 | 0.216182 | YES | YES | 4.049782 | NO |
| collider | linear | additive | L2<=L1-oracle | -0.018732 | 0.076672 | YES | NO | 0.244318 | YES |
| collider | linear | effect_modifying | L1-oracle<=L0 | -0.267914 | 0.242296 | YES | NO | 1.105733 | NO |
| collider | linear | effect_modifying | L2<=L1-oracle | -0.029080 | 0.078678 | YES | NO | 0.369603 | YES |
| collider | nlg | additive | L1-oracle<=L0 | -0.358505 | 0.100822 | YES | YES | 3.555834 | NO |
| collider | nlg | additive | L2<=L1-oracle | -0.019495 | 0.089597 | YES | NO | 0.217583 | YES |
| collider | nlg | effect_modifying | L1-oracle<=L0 | -0.124383 | 0.138629 | YES | NO | 0.897235 | YES |
| collider | nlg | effect_modifying | L2<=L1-oracle | -0.023555 | 0.108186 | YES | NO | 0.217729 | YES |
| chain | linear | additive | L1-oracle<=L0 | -0.103150 | 0.054230 | YES | NO | 1.902075 | NO |
| chain | linear | additive | L2<=L1-oracle | -0.002841 | 0.029492 | YES | NO | 0.096326 | YES |
| chain | linear | effect_modifying | L1-oracle<=L0 | -0.000859 | 0.085683 | YES | NO | 0.010021 | YES |
| chain | linear | effect_modifying | L2<=L1-oracle | 0.003377 | 0.027053 | NO | NO | 0.124832 | YES |
| chain | nlg | additive | L1-oracle<=L0 | -0.047018 | 0.037660 | YES | NO | 1.248482 | NO |
| chain | nlg | additive | L2<=L1-oracle | -0.001332 | 0.023933 | YES | NO | 0.055675 | YES |
| chain | nlg | effect_modifying | L1-oracle<=L0 | 0.033935 | 0.040365 | NO | NO | 0.840704 | YES |
| chain | nlg | effect_modifying | L2<=L1-oracle | 0.003111 | 0.023282 | NO | NO | 0.133644 | YES |

## 5. The L2 → L1-oracle reversal, per cell, with its seed SD

`[Δ_cost(L1-oracle) − Δ_cost(L2)]`. A **positive** sign means the ladder
ordering holds at cell level; **negative** is the L2→L1-oracle reversal
(L1-oracle buys cheaper actions that fail the true classifier more often).

| topology | family | regime | per_seed | delta_mean | delta_sd | sign | sign_consistent | abs_mean_over_sd | within_seed_noise |
|---|---|---|---|---|---|---|---|---|---|
| triangle | linear | additive | [0.052103, 0.004901, -0.003505, -0.007816, 0.030315, 0.072453, 0.02194, -0.034891, -0.003566, 0.00292, -0.025528, -0.014793, 0.034217, 0.018162, 0.000508, -0.000651, -0.042492, -0.016145, 0.025866, 0.029355] | 0.007168 | 0.028693 | + | NO | 0.249810 | YES |
| triangle | linear | effect_modifying | [-0.015917, 0.002081, -0.003763, -0.007128, -0.002544, -0.002684, -0.014742, -0.009395, -0.00651, -0.012401, 0.004573, -0.00481, -0.001157, 0.0, 0.009323, -0.013888, 0.009119, 0.010156, -0.001807, 0.000198] | -0.003065 | 0.007831 | - | NO | 0.391397 | YES |
| triangle | nlg | additive | [-0.000547, 0.000302, 0.0, 0.00062, -0.000471, -0.000363, -0.000203, 1.4e-05, -0.000721, 0.000651, 0.001002, -0.000591, -0.000547, 0.000808, 0.000125, -4.4e-05, 0.003679, 0.000234, -0.000186, -0.000229] | 0.000177 | 0.000959 | + | NO | 0.184146 | YES |
| triangle | nlg | effect_modifying | [-0.005729, 0.000225, -0.000684, -0.008632, 0.003492, 0.005929, -0.001771, -0.005747, 0.000216, -0.012713, -9.7e-05, -0.002805, 0.00306, 0.000725, 0.005949, -0.003967, 0.013062, 0.002671, 0.0, 0.008004] | 0.000059 | 0.005910 | + | NO | 0.010050 | YES |
| collider | linear | additive | [0.048926, 0.087904, -0.078157, -0.115116, -0.046426, 0.041906, 0.147678, -1.8e-05, -0.011743, 0.041233, 0.041672, -0.073951, -0.022868, 0.057671, 0.041666, 0.131765, 0.15938, -0.055304, -0.048372, 0.026798] | 0.018732 | 0.076672 | + | NO | 0.244318 | YES |
| collider | linear | effect_modifying | [0.10458, 0.073775, 0.020361, -0.115772, -0.079561, -0.009166, 0.098856, -0.001402, -0.04894, 0.04937, 0.025819, 0.031033, 0.016394, 0.115677, 0.041868, 0.219025, 0.062509, -0.090639, -0.010158, 0.077965] | 0.029080 | 0.078678 | + | NO | 0.369603 | YES |
| collider | nlg | additive | [0.084087, 0.085551, -0.052028, -0.119671, -0.123987, 0.041414, 0.115065, 0.008847, -0.011644, 0.009941, 0.003366, -0.073085, -0.053631, 0.030991, 0.094215, 0.179638, 0.200889, -0.043675, -0.04504, 0.058654] | 0.019495 | 0.089597 | + | NO | 0.217583 | YES |
| collider | nlg | effect_modifying | [0.138546, 0.030559, 0.035789, -0.112505, -0.249774, -0.057945, 0.104414, -0.012508, -0.06432, -0.003611, -0.012709, 0.027202, 0.019355, 0.060214, 0.068887, 0.281378, 0.075598, -0.024843, 0.015516, 0.151861] | 0.023555 | 0.108186 | + | NO | 0.217729 | YES |
| chain | linear | additive | [0.050339, 0.034592, -0.027287, -0.046236, -0.014823, 0.022813, -0.018469, 0.001132, 0.012084, 0.032899, -0.021063, -0.022546, 0.012746, -0.003509, 0.022875, 0.072561, -0.00759, -0.014459, -0.023393, -0.00585] | 0.002841 | 0.029492 | + | NO | 0.096326 | YES |
| chain | linear | effect_modifying | [-0.012958, 0.002428, -0.048103, -0.060344, -0.036088, 0.032277, 0.02701, -0.006352, -0.0055, -0.010247, 0.009654, -0.028326, -2.4e-05, -0.000225, 0.025183, 0.048279, 0.012, 0.005817, -0.026415, 0.004391] | -0.003377 | 0.027053 | - | NO | 0.124832 | YES |
| chain | nlg | additive | [0.016201, 0.036006, -0.024154, -0.048432, -0.040475, 0.01973, -0.004108, 0.005047, 0.007118, 0.010876, 0.00098, -0.01185, 0.005704, -0.003366, 0.026473, 0.05079, 0.005809, -0.009476, -0.02346, 0.007235] | 0.001332 | 0.023933 | + | NO | 0.055675 | YES |
| chain | nlg | effect_modifying | [-0.014597, 0.011072, -0.036776, -0.058024, -0.034438, 0.022978, 0.009591, 0.000419, -0.006065, -0.011051, 0.007621, -0.017674, -0.000492, -1.3e-05, 0.014865, 0.036339, 0.029036, 0.000417, -0.025003, 0.009566] | -0.003111 | 0.023282 | - | NO | 0.133644 | YES |

## 6. H1 — the mechanical PS-5 criterion read, per topology

The criterion (PS-5) asks two things of the ladder: that the **ordering** `L2 ≤ L1-oracle ≤ L0`
hold, and that its **variability be non-overlapping**. Both are reported, and
the variability question is reported under BOTH available readings rather than
one being chosen silently:

- **paired** — `|mean(diff)| / SD(diff)` on the per-seed difference. Seeds are
  shared across conditions, so this is the tighter and more powerful reading,
  and it is the one reported here. Separated iff the ratio > 1.
- **marginal** — `band_gap = (mean_upper − SD_upper) − (mean_lower + SD_lower)`,
  in raw cost units. **Positive** = the two ±1 SD bands are disjoint, and the
  value is the clear space between them. **Negative** = they overlap, and
  `band_overlap_width` is by how much.

A rung counts as separated only when the ordering ALSO holds on the mean: a
disjoint band on the wrong side of the ordering is a violation, not a pass.

### 6.1 Per (cell, group, rung) — the numbers

| topology | family | regime | group | rung | diff_mean | diff_sd | holds_on_mean | n_seeds_holding | abs_mean_over_sd | band_gap | bands_overlap | band_overlap_width | paired_separates | marginal_separates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| triangle | linear | additive | -1.000000 | L1-oracle<=L0 | -0.202972 | 0.124463 | YES | 20 | 1.630782 | 0.012851 | NO | 0.000000 | YES | YES |
| triangle | linear | additive | -1.000000 | L2<=L1-oracle | -0.008053 | 0.032160 | YES | 11 | 0.250393 | -0.226685 | YES | 0.226685 | NO | NO |
| triangle | linear | additive | 1.000000 | L1-oracle<=L0 | -0.028677 | 0.015106 | YES | 20 | 1.898460 | -0.028553 | YES | 0.028553 | YES | NO |
| triangle | linear | additive | 1.000000 | L2<=L1-oracle | -0.000885 | 0.003573 | YES | 11 | 0.247655 | -0.054040 | YES | 0.054040 | NO | NO |
| triangle | linear | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.000005 | 0.000008 | YES | 20 | 0.606844 | -0.172898 | YES | 0.172898 | NO | NO |
| triangle | linear | effect_modifying | -1.000000 | L2<=L1-oracle | 0.000002 | 0.000006 | NO | 17 | 0.289005 | -0.172905 | YES | 0.172905 | NO | NO |
| triangle | linear | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.208833 | 0.014467 | YES | 20 | 14.434644 | 0.135671 | NO | 0.000000 | YES | YES |
| triangle | linear | effect_modifying | 1.000000 | L2<=L1-oracle | -0.003063 | 0.007829 | YES | 14 | 0.391274 | -0.059862 | YES | 0.059862 | NO | NO |
| triangle | nlg | additive | -1.000000 | L1-oracle<=L0 | -0.000147 | 0.000437 | YES | 20 | 0.335632 | -0.139949 | YES | 0.139949 | NO | NO |
| triangle | nlg | additive | -1.000000 | L2<=L1-oracle | 0.000114 | 0.000388 | NO | 14 | 0.294386 | -0.140184 | YES | 0.140184 | NO | NO |
| triangle | nlg | additive | 1.000000 | L1-oracle<=L0 | -0.001541 | 0.001654 | YES | 20 | 0.931664 | -0.053587 | YES | 0.053587 | NO | NO |
| triangle | nlg | additive | 1.000000 | L2<=L1-oracle | 0.000291 | 0.001313 | NO | 11 | 0.221517 | -0.055266 | YES | 0.055266 | NO | NO |
| triangle | nlg | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.000051 | 0.000202 | YES | 20 | 0.253070 | -0.157671 | YES | 0.157671 | NO | NO |
| triangle | nlg | effect_modifying | -1.000000 | L2<=L1-oracle | 0.000048 | 0.000203 | NO | 15 | 0.237354 | -0.157770 | YES | 0.157770 | NO | NO |
| triangle | nlg | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.018636 | 0.006881 | YES | 20 | 2.708337 | -0.045413 | YES | 0.045413 | YES | NO |
| triangle | nlg | effect_modifying | 1.000000 | L2<=L1-oracle | 0.000108 | 0.005805 | NO | 10 | 0.018525 | -0.063436 | YES | 0.063436 | NO | NO |
| collider | linear | additive | -1.000000 | L1-oracle<=L0 | -0.988780 | 0.243292 | YES | 20 | 4.064176 | 0.220413 | NO | 0.000000 | YES | YES |
| collider | linear | additive | -1.000000 | L2<=L1-oracle | -0.018783 | 0.082581 | YES | 11 | 0.227448 | -0.584103 | YES | 0.584103 | NO | NO |
| collider | linear | additive | 1.000000 | L1-oracle<=L0 | -0.113292 | 0.031951 | YES | 20 | 3.545825 | -0.021538 | YES | 0.021538 | YES | NO |
| collider | linear | additive | 1.000000 | L2<=L1-oracle | -0.000051 | 0.006902 | YES | 12 | 0.007338 | -0.114217 | YES | 0.114217 | NO | NO |
| collider | linear | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.441472 | 0.235516 | YES | 20 | 1.874488 | -0.545515 | YES | 0.545515 | YES | NO |
| collider | linear | effect_modifying | -1.000000 | L2<=L1-oracle | -0.028941 | 0.083273 | YES | 13 | 0.347547 | -0.977475 | YES | 0.977475 | NO | NO |
| collider | linear | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.173558 | 0.054597 | YES | 20 | 3.178860 | 0.037830 | NO | 0.000000 | YES | YES |
| collider | linear | effect_modifying | 1.000000 | L2<=L1-oracle | 0.000138 | 0.010669 | NO | 13 | 0.012965 | -0.090395 | YES | 0.090395 | NO | NO |
| collider | nlg | additive | -1.000000 | L1-oracle<=L0 | -0.450350 | 0.114194 | YES | 20 | 3.943714 | 0.019519 | NO | 0.000000 | YES | YES |
| collider | nlg | additive | -1.000000 | L2<=L1-oracle | -0.021227 | 0.104065 | YES | 12 | 0.203980 | -0.428760 | YES | 0.428760 | NO | NO |
| collider | nlg | additive | 1.000000 | L1-oracle<=L0 | -0.091845 | 0.017409 | YES | 20 | 5.275765 | -0.028024 | YES | 0.028024 | YES | NO |
| collider | nlg | additive | 1.000000 | L2<=L1-oracle | -0.001732 | 0.014694 | YES | 12 | 0.117895 | -0.112206 | YES | 0.112206 | NO | NO |
| collider | nlg | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.247119 | 0.141465 | YES | 20 | 1.746858 | -0.247780 | YES | 0.247780 | YES | NO |
| collider | nlg | effect_modifying | -1.000000 | L2<=L1-oracle | -0.025431 | 0.117161 | YES | 12 | 0.217058 | -0.520436 | YES | 0.520436 | NO | NO |
| collider | nlg | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.122736 | 0.026124 | YES | 20 | 4.698154 | 0.005274 | NO | 0.000000 | YES | YES |
| collider | nlg | effect_modifying | 1.000000 | L2<=L1-oracle | -0.001875 | 0.019638 | YES | 11 | 0.095504 | -0.099769 | YES | 0.099769 | NO | NO |
| chain | linear | additive | -1.000000 | L1-oracle<=L0 | -0.134855 | 0.060390 | YES | 20 | 2.233070 | -0.120328 | YES | 0.120328 | YES | NO |
| chain | linear | additive | -1.000000 | L2<=L1-oracle | -0.003236 | 0.032870 | YES | 9 | 0.098447 | -0.274160 | YES | 0.274160 | NO | NO |
| chain | linear | additive | 1.000000 | L1-oracle<=L0 | -0.031705 | 0.009641 | YES | 20 | 3.288599 | -0.016830 | YES | 0.016830 | YES | NO |
| chain | linear | additive | 1.000000 | L2<=L1-oracle | -0.000395 | 0.003848 | YES | 10 | 0.102693 | -0.049605 | YES | 0.049605 | NO | NO |
| chain | linear | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.076127 | 0.084433 | YES | 20 | 0.901620 | -0.192353 | YES | 0.192353 | NO | NO |
| chain | linear | effect_modifying | -1.000000 | L2<=L1-oracle | 0.000538 | 0.028080 | NO | 11 | 0.019168 | -0.315419 | YES | 0.315419 | NO | NO |
| chain | linear | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.075268 | 0.013185 | YES | 20 | 5.708451 | 0.028118 | NO | 0.000000 | YES | YES |
| chain | linear | effect_modifying | 1.000000 | L2<=L1-oracle | -0.002839 | 0.006945 | YES | 12 | 0.408741 | -0.052358 | YES | 0.052358 | NO | NO |
| chain | nlg | additive | -1.000000 | L1-oracle<=L0 | -0.074724 | 0.042912 | YES | 20 | 1.741340 | -0.072636 | YES | 0.072636 | YES | NO |
| chain | nlg | additive | -1.000000 | L2<=L1-oracle | -0.002011 | 0.027653 | YES | 12 | 0.072728 | -0.160150 | YES | 0.160150 | NO | NO |
| chain | nlg | additive | 1.000000 | L1-oracle<=L0 | -0.027706 | 0.007638 | YES | 20 | 3.627400 | -0.003545 | YES | 0.003545 | YES | NO |
| chain | nlg | additive | 1.000000 | L2<=L1-oracle | -0.000679 | 0.003936 | YES | 12 | 0.172401 | -0.031290 | YES | 0.031290 | NO | NO |
| chain | nlg | effect_modifying | -1.000000 | L1-oracle<=L0 | -0.037705 | 0.047302 | YES | 20 | 0.797106 | -0.107884 | YES | 0.107884 | NO | NO |
| chain | nlg | effect_modifying | -1.000000 | L2<=L1-oracle | 0.000699 | 0.026359 | NO | 10 | 0.026514 | -0.167404 | YES | 0.167404 | NO | NO |
| chain | nlg | effect_modifying | 1.000000 | L1-oracle<=L0 | -0.071640 | 0.014890 | YES | 20 | 4.811326 | 0.034588 | NO | 0.000000 | YES | YES |
| chain | nlg | effect_modifying | 1.000000 | L2<=L1-oracle | -0.002413 | 0.007282 | YES | 12 | 0.331308 | -0.036350 | YES | 0.036350 | NO | NO |

### 6.2 Rolled up to the topology — the unit the criterion names

A **series** is one (cell, group). It is *ordered* when both rungs hold on the
mean, and *separated* when both rungs also separate under the stated reading.

| topology | n_series | n_series_ordered | n_series_paired_separated | n_series_marginal_separated | binding_rung | max_abs_mean_over_sd_on_binding_rung | min_band_gap | max_band_overlap_width | any_series_meets_h1_criterion | all_series_meet_h1_criterion |
|---|---|---|---|---|---|---|---|---|---|---|
| triangle | 8 | 3 | 0 | 0 | L2<=L1-oracle | 0.391274 | -0.226685 | 0.226685 | NO | NO |
| collider | 8 | 7 | 0 | 0 | L2<=L1-oracle | 0.347547 | -0.977475 | 0.977475 | NO | NO |
| chain | 8 | 6 | 0 | 0 | L2<=L1-oracle | 0.408741 | -0.315419 | 0.315419 | NO | NO |

### 6.3 VERDICT under the pinned rule

> **PS-5:** a monotone ordering L2 ≤ L1-oracle ≤ L0 with non-overlapping variability on at least one of the three topologies

**NOT MET.**

The criterion makes the topology the unit and quantifies existentially ACROSS topologies
("at least one"). It does not say how the cells and groups WITHIN a topology
combine, so the verdict is computed under both the weak reading (at least one
series in the topology separates) and the strong reading (every series does):

- topologies meeting the criterion, **weak** reading: **none**
- topologies meeting the criterion, **strong** reading: **none**
- the two readings agree: **YES** — so the verdict does not depend on which within-topology aggregation is adopted, and no aggregation choice is doing any work here.

*No prose beyond the rule is written here: the PS-5 text is the whole criterion,*
*and the table above is the whole evidence.*

## Comparison against the N=6 early look

**Not computed.** No comparison root was available on this run (`--compare-root` absent or the tree is not present), so the N=6 → N=20 comparison is UNAVAILABLE rather than assumed to agree.
