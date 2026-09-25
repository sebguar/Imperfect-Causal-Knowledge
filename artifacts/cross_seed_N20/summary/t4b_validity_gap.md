# H2 / ValidityDisp and the Gap_c read (N=20, 12 cells)

## SUPERSEDES

- `results/cross_seed_N6/summary/t4b_validity_gap.md`

Those artifacts are **audit objects** and are NOT overwritten. They record the **PS-1 early look** — 8 cells × 6 seeds, before the chain joined the grid at the chain SCM specification. This artifact is the **full registered scope**: 12 cells × 20 seeds. Where the two disagree, the disagreement is reported below rather than resolved in favour of the earlier read.

## Outcome exposure — stated, not claimed away

**Gap_c (PS-6) was register-logged prior to outcome exposure.** It is quoted below as it now stands. No threshold, band, quantifier or population in it was moved for this read: every one applied here is the one on record before these numbers were computed.

> the gap grows as knowledge decreases, largest at L0, extending [VonKugelgen2022] to a continuum; a flat gap disconfirms H2

**This application of it is OUTCOME-EXPOSED.** The numbers existed before this document was generated, and the choice to compute the read now, on this tree, was made with the N=6 early look already known. That is disclosed rather than described away: this is **not** an outcome-independent read, and nothing here should be cited as one. What is pre-committed is the RULE; what is post-data is the OCCASION of applying it.

## 1. ValidityDisp at L1-oracle = realized validity(A=−1) − realized validity(A=+1)

| topology | family | regime | validity_neg_mean | validity_neg_sd | validity_pos_mean | validity_pos_sd | disp_mean | disp_sd | sd_ge_abs_mean_disp | sd_ge_abs_mean_validity_neg | n_high_validity_seeds | n_collapse_seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| triangle | linear | additive | 0.925503 | 0.130289 | 0.988922 | 0.019589 | -0.063418 | 0.112405 | YES | NO | 20 | 0 |
| triangle | linear | effect_modifying | 0.999874 | 0.000363 | 0.984096 | 0.031674 | 0.015778 | 0.031743 | YES | NO | 20 | 0 |
| triangle | nlg | additive | 0.997210 | 0.008424 | 0.988927 | 0.021879 | 0.008283 | 0.014723 | YES | NO | 20 | 0 |
| triangle | nlg | effect_modifying | 0.998196 | 0.007032 | 0.976214 | 0.038332 | 0.021983 | 0.040108 | YES | NO | 20 | 0 |
| collider | linear | additive | 0.838543 | 0.231089 | 0.984503 | 0.021398 | -0.145960 | 0.210430 | YES | NO | 18 | 2 |
| collider | linear | effect_modifying | 0.851887 | 0.286734 | 0.976519 | 0.039088 | -0.124632 | 0.278987 | YES | NO | 17 | 3 |
| collider | nlg | additive | 0.682600 | 0.415231 | 0.913879 | 0.119636 | -0.231279 | 0.302104 | YES | NO | 13 | 7 |
| collider | nlg | effect_modifying | 0.740052 | 0.372918 | 0.898007 | 0.137560 | -0.157955 | 0.371387 | YES | NO | 15 | 5 |
| chain | linear | additive | 0.771076 | 0.253334 | 0.970519 | 0.039859 | -0.199443 | 0.222003 | YES | NO | 15 | 4 |
| chain | linear | effect_modifying | 0.857336 | 0.261542 | 0.983816 | 0.026956 | -0.126480 | 0.257162 | YES | NO | 16 | 4 |
| chain | nlg | additive | 0.771380 | 0.341230 | 0.959062 | 0.057672 | -0.187682 | 0.286614 | YES | NO | 16 | 4 |
| chain | nlg | effect_modifying | 0.804984 | 0.318996 | 0.973137 | 0.035081 | -0.168153 | 0.303148 | YES | NO | 16 | 4 |

### Per-seed A=−1 realized validity at L1-oracle (seed order)

The distribution shape is the finding: on the collider the series switches
near-binary rather than varying smoothly, so the moments above misdescribe it.

| topology | family | regime | seed 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 |
|---|---|---|---|
| triangle | linear | additive | 1.0000 , 1.0000 , 0.9671 , 0.9285 , 1.0000 , 1.0000 , 1.0000 , 0.6384 , 0.9632 , 1.0000 , 0.7530 , 0.8510 , 1.0000 , 1.0000 , 0.9975 , 1.0000 , 0.5664 , 0.8449 , 1.0000 , 1.0000 |
| triangle | linear | effect_modifying | 1.0000 , 1.0000 , 1.0000 , 0.9995 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 0.9995 , 0.9985 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 |
| triangle | nlg | additive | 1.0000 , 0.9990 , 1.0000 , 0.9990 , 1.0000 , 1.0000 , 1.0000 , 0.9985 , 1.0000 , 0.9848 , 1.0000 , 1.0000 , 1.0000 , 0.9985 , 1.0000 , 1.0000 , 0.9644 , 1.0000 , 1.0000 , 1.0000 |
| triangle | nlg | effect_modifying | 0.9995 , 1.0000 , 1.0000 , 0.9970 , 1.0000 , 1.0000 , 1.0000 , 0.9995 , 1.0000 , 0.9685 , 1.0000 , 1.0000 , 1.0000 , 0.9995 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 |
| collider | linear | additive | 1.0000 , 1.0000 , 0.5505 , 0.5027 , 0.6040 , 1.0000 , 1.0000 , 0.9868 , 0.7813 , 1.0000 , 1.0000 , 0.3223 , 0.9193 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 0.4665 , 0.6374 , 1.0000 |
| collider | linear | effect_modifying | 1.0000 , 1.0000 , 0.9408 , 0.0068 , 0.7114 , 0.8989 , 1.0000 , 0.9776 , 0.4450 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 0.2418 , 0.8154 , 1.0000 |
| collider | nlg | additive | 1.0000 , 0.9986 , 0.2853 , 0.0246 , 0.0040 , 1.0000 , 1.0000 , 0.9855 , 0.7004 , 1.0000 , 0.9425 , 0.1157 , 0.2986 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 0.2457 , 0.0510 , 1.0000 |
| collider | nlg | effect_modifying | 1.0000 , 0.9665 , 0.9731 , 0.0178 , 0.0000 , 0.0697 , 1.0000 , 0.7691 , 0.2478 , 0.9775 , 0.7137 , 0.9394 , 0.9275 , 1.0000 , 1.0000 , 1.0000 , 1.0000 , 0.3443 , 0.8547 , 1.0000 |
| chain | linear | additive | 1.0000 , 0.8522 , 0.4818 , 0.1735 , 0.4671 , 1.0000 , 0.6077 , 1.0000 , 0.9924 , 1.0000 , 0.6192 , 0.5000 , 0.9891 , 0.9229 , 1.0000 , 1.0000 , 0.8182 , 0.6426 , 0.4699 , 0.8850 |
| chain | linear | effect_modifying | 1.0000 , 1.0000 , 0.4966 , 0.2172 , 0.4752 , 1.0000 , 0.9973 , 1.0000 , 0.9861 , 1.0000 , 1.0000 , 0.6446 , 0.9959 , 0.9987 , 1.0000 , 1.0000 , 1.0000 , 0.9974 , 0.3377 , 1.0000 |
| chain | nlg | additive | 1.0000 , 0.9819 , 0.2086 , 0.0410 , 0.0605 , 1.0000 , 0.9096 , 1.0000 , 0.9911 , 1.0000 , 0.9315 , 0.5138 , 1.0000 , 0.8856 , 1.0000 , 1.0000 , 0.9274 , 0.7516 , 0.2971 , 0.9278 |
| chain | nlg | effect_modifying | 1.0000 , 1.0000 , 0.2625 , 0.0273 , 0.3664 , 1.0000 , 1.0000 , 1.0000 , 0.9014 , 1.0000 , 0.9932 , 0.5675 , 0.9904 , 0.9987 , 1.0000 , 1.0000 , 1.0000 , 0.7175 , 0.2749 , 1.0000 |

Collider cells with `SD ≥ |mean|` on the **ValidityDisp**: **4 / 4** — this is
the seed-fragility statement PS-9 was opened on, and it reproduces.

Collider cells with `SD ≥ |mean|` on the raw **A=−1 validity level**: **0 / 4**.
The two flags do not agree, and the distinction matters: the *disparity* is
fragile in every collider cell, while the *level* only reads as fragile
where enough seeds have actually collapsed to ~0.

> **Read the flag with its cause.** `SD ≥ |mean|` fires on the triangle cells
> too, but for the opposite reason: there the disparity is ~0 with ~0
> dispersion, so the ratio is uninformative — the finding is that the
> asymmetry is ABSENT, not that it is fragile. On the collider the disparity
> is large (|mean| 0.20–0.36) AND its SD exceeds it, which is genuine seed
> fragility. The flag alone does not distinguish the two; the `disp_mean`
> column beside it does.

## 2. Gap_c confirmation (PS-6): zero at L1-oracle and L2, nonzero at L0

| topology | family | regime | max_abs_Gap_cost_L0 | min_abs_Gap_cost_L0 | max_abs_Gap_cost_L1-oracle | max_abs_Gap_cost_L2 | zero_at_L1_and_L2 | nonzero_at_L0_all_seeds |
|---|---|---|---|---|---|---|---|---|
| triangle | linear | additive | 0.001700 | 0.000583 | 0.000000 | 0.000000 | YES | YES |
| triangle | linear | effect_modifying | 0.001724 | 0.000000 | 0.000000 | 0.000000 | YES | NO |
| triangle | nlg | additive | 0.000090 | 0.000000 | 0.000000 | 0.000000 | YES | NO |
| triangle | nlg | effect_modifying | 0.000234 | 0.000035 | 0.000000 | 0.000000 | YES | YES |
| collider | linear | additive | 0.252324 | 0.020188 | 0.000000 | 0.000000 | YES | YES |
| collider | linear | effect_modifying | 0.163827 | 0.032399 | 0.000000 | 0.000000 | YES | YES |
| collider | nlg | additive | 0.061327 | 0.000000 | 0.000000 | 0.000000 | YES | NO |
| collider | nlg | effect_modifying | 0.088519 | 0.013063 | 0.000000 | 0.000000 | YES | YES |
| chain | linear | additive | 0.022222 | 0.003871 | 0.000000 | 0.000000 | YES | YES |
| chain | linear | effect_modifying | 0.040693 | 0.013235 | 0.000000 | 0.000000 | YES | YES |
| chain | nlg | additive | 0.011518 | 0.000000 | 0.000000 | 0.000000 | YES | NO |
| chain | nlg | effect_modifying | 0.011771 | 0.000603 | 0.000000 | 0.000000 | YES | YES |

## 3. valid-subset small-N guard

Threshold: `N_valid_subset < 30` at L1-oracle. On a flagged
(cell, group, seed) the valid-subset cost is a near-empty average and its
contribution to the common-found population **secondary** contrast is unreliable. The cross-seed
cost is therefore given twice — over all 20 seeds, and over the seeds
clearing the threshold — so small-N contamination is separable from cost signal.

| topology | family | regime | group | N_valid_subset_per_seed | n_seeds_flagged_small | flagged_seed_idx | collapse_seeds_passing_guard | cost_mean_all_seeds | cost_sd_all_seeds | cost_mean_guarded | cost_sd_guarded | n_seeds_guarded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| triangle | linear | additive | -1.000000 | [2004, 1980, 1909, 1844, 2031, 2031, 2026, 1282, 1937, 1974, 1488, 1731, 1988, 1992, 1994, 2016, 1130, 1710, 2020, 1966] | 0 | [] | [] | 4.419431 | 0.147176 | 4.419431 | 0.147176 | 20 |
| triangle | linear | additive | 1.000000 | [411, 410, 455, 425, 399, 370, 393, 387, 400, 418, 473, 422, 436, 439, 412, 415, 430, 405, 382, 455] | 0 | [] | [] | 0.726628 | 0.029231 | 0.726628 | 0.029231 | 20 |
| triangle | linear | effect_modifying | -1.000000 | [2003, 1979, 1973, 1983, 2030, 2030, 2025, 2006, 2010, 1968, 1976, 2032, 1984, 1991, 1998, 2013, 1995, 2024, 2020, 1966] | 0 | [] | [] | 3.693792 | 0.086767 | 3.693792 | 0.086767 | 20 |
| triangle | linear | effect_modifying | 1.000000 | [589, 530, 623, 580, 576, 526, 543, 579, 546, 576, 595, 559, 584, 573, 548, 560, 535, 526, 556, 567] | 0 | [] | [] | 0.759569 | 0.035524 | 0.759569 | 0.035524 | 20 |
| triangle | nlg | additive | -1.000000 | [2004, 1978, 1974, 1983, 2031, 2031, 2026, 2004, 2011, 1944, 1976, 2034, 1988, 1989, 1999, 2015, 1924, 2024, 2020, 1966] | 0 | [] | [] | 4.002948 | 0.077817 | 4.002948 | 0.077817 | 20 |
| triangle | nlg | additive | 1.000000 | [399, 369, 432, 392, 368, 347, 363, 379, 359, 390, 461, 397, 403, 402, 389, 401, 388, 383, 372, 427] | 0 | [] | [] | 0.650036 | 0.025367 | 0.650036 | 0.025367 | 20 |
| triangle | nlg | effect_modifying | -1.000000 | [1994, 1974, 1969, 1975, 2026, 2024, 2020, 1995, 2001, 1904, 1972, 2029, 1979, 1981, 1993, 2006, 1993, 2016, 2014, 1961] | 0 | [] | [] | 2.931626 | 0.084628 | 2.931626 | 0.084628 | 20 |
| triangle | nlg | effect_modifying | 1.000000 | [529, 499, 576, 541, 489, 475, 538, 556, 515, 515, 609, 545, 513, 530, 512, 535, 480, 503, 531, 481] | 0 | [] | [] | 0.719576 | 0.039450 | 0.719576 | 0.039450 | 20 |
| collider | linear | additive | -1.000000 | [761, 717, 409, 368, 450, 732, 752, 748, 618, 711, 730, 234, 672, 752, 732, 770, 759, 355, 487, 748] | 0 | [] | [11, 17] | 5.564792 | 0.310038 | 5.564792 | 0.310038 | 20 |
| collider | linear | additive | 1.000000 | [104, 132, 132, 126, 110, 161, 100, 115, 90, 150, 133, 165, 140, 126, 143, 80, 84, 102, 100, 102] | 0 | [] | [] | 0.798527 | 0.054417 | 0.798527 | 0.054417 | 20 |
| collider | linear | effect_modifying | -1.000000 | [761, 717, 699, 5, 530, 658, 752, 741, 352, 711, 730, 726, 731, 752, 732, 770, 759, 184, 623, 748] | 1 | [3] | [8, 17] | 5.937986 | 0.749206 | 6.060998 | 0.522537 | 19 |
| collider | linear | effect_modifying | 1.000000 | [124, 141, 136, 137, 117, 164, 111, 114, 98, 156, 152, 149, 156, 134, 142, 101, 99, 123, 105, 110] | 0 | [] | [] | 0.785575 | 0.048206 | 0.785575 | 0.048206 | 20 |
| collider | nlg | additive | -1.000000 | [761, 716, 212, 18, 3, 732, 752, 747, 554, 711, 688, 84, 218, 752, 732, 770, 759, 187, 39, 748] | 2 | [3, 4] | [2, 11, 12, 17, 18] | 2.101121 | 0.578248 | 2.238248 | 0.398081 | 18 |
| collider | nlg | additive | 1.000000 | [173, 191, 157, 134, 103, 181, 148, 149, 132, 210, 178, 131, 148, 177, 207, 131, 142, 130, 132, 148] | 0 | [] | [] | 0.394798 | 0.065998 | 0.394798 | 0.065998 | 20 |
| collider | nlg | effect_modifying | -1.000000 | [761, 693, 723, 13, 0, 51, 752, 583, 196, 695, 521, 682, 678, 752, 732, 770, 759, 262, 653, 748] | 2 | [3, 4] | [5, 8, 17] | 2.472306 | 0.400599 | 2.498123 | 0.395614 | 18 |
| collider | nlg | effect_modifying | 1.000000 | [179, 193, 130, 129, 134, 188, 158, 155, 139, 216, 179, 110, 135, 177, 199, 136, 153, 142, 110, 141] | 0 | [] | [] | 0.376708 | 0.068268 | 0.376708 | 0.068268 | 20 |
| chain | linear | additive | -1.000000 | [761, 611, 358, 127, 348, 732, 457, 758, 785, 711, 452, 363, 723, 694, 732, 770, 621, 489, 359, 662] | 0 | [] | [2, 3, 4, 18] | 2.621116 | 0.187043 | 2.621116 | 0.187043 | 20 |
| chain | linear | additive | 1.000000 | [143, 158, 159, 143, 138, 195, 137, 157, 111, 191, 159, 168, 181, 156, 176, 133, 126, 140, 143, 135] | 0 | [] | [] | 0.452796 | 0.022700 | 0.452796 | 0.022700 | 20 |
| chain | linear | effect_modifying | -1.000000 | [761, 717, 369, 159, 354, 732, 750, 758, 780, 711, 730, 468, 728, 751, 732, 770, 759, 759, 258, 748] | 0 | [] | [2, 3, 4, 18] | 2.683307 | 0.235909 | 2.683307 | 0.235909 | 20 |
| chain | linear | effect_modifying | 1.000000 | [165, 176, 199, 192, 160, 201, 153, 168, 113, 203, 157, 168, 194, 167, 190, 143, 143, 153, 153, 150] | 0 | [] | [] | 0.468834 | 0.022229 | 0.468834 | 0.022229 | 20 |
| chain | nlg | additive | -1.000000 | [761, 704, 155, 30, 45, 731, 684, 758, 784, 711, 680, 373, 731, 666, 732, 769, 703, 572, 227, 694] | 0 | [] | [2, 3, 4, 18] | 1.608764 | 0.288134 | 1.608764 | 0.288134 | 20 |
| chain | nlg | additive | 1.000000 | [167, 170, 167, 156, 131, 202, 145, 170, 125, 208, 172, 166, 191, 167, 177, 135, 147, 151, 145, 152] | 0 | [] | [] | 0.324093 | 0.022120 | 0.324093 | 0.022120 | 20 |
| chain | nlg | effect_modifying | -1.000000 | [761, 717, 195, 20, 273, 732, 752, 758, 713, 711, 725, 412, 724, 751, 732, 770, 759, 546, 210, 748] | 1 | [3] | [2, 4, 18] | 1.706210 | 0.199014 | 1.743109 | 0.114297 | 19 |
| chain | nlg | effect_modifying | 1.000000 | [170, 174, 191, 195, 169, 204, 168, 175, 133, 215, 163, 178, 194, 173, 184, 149, 163, 157, 162, 172] | 0 | [] | [] | 0.344953 | 0.017791 | 0.344953 | 0.017791 | 20 |

Flagged (cell, group) rows: **4 / 24**.

> **Limit of the guard as specified.** `N_valid_subset ≥ 30` is an absolute
> COUNT threshold, not a validity-rate one, so a collapse seed with a large
> eligible pool can clear it — `collapse_seeds_passing_guard` names every seed
> that passes the count guard while its realized validity is below 0.5. Those
> seeds still enter the guarded mean, so the guarded column is *less*
> contaminated than the all-seed column, not contamination-free.

## 4. H2 — the mechanical Gap_c read, per topology

The quantity is **the per-condition gap `|believed − realized|`** — PS-6's
`Gap_cost`. The ladder is read in the criterion's direction, knowledge DECREASING:
`L2 → L1-oracle → L0`.

### 4.1 The gap ladder, per cell × condition

| topology | family | regime | condition | n_seeds | abs_gap_mean | abs_gap_sd | abs_gap_min | abs_gap_max |
|---|---|---|---|---|---|---|---|---|
| triangle | linear | additive | L0 | 20 | 0.001142 | 0.000272 | 0.000583 | 0.001700 |
| triangle | linear | additive | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| triangle | linear | additive | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| triangle | linear | effect_modifying | L0 | 20 | 0.000454 | 0.000446 | 0.000000 | 0.001724 |
| triangle | linear | effect_modifying | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| triangle | linear | effect_modifying | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| triangle | nlg | additive | L0 | 20 | 0.000022 | 0.000026 | 0.000000 | 0.000090 |
| triangle | nlg | additive | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| triangle | nlg | additive | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| triangle | nlg | effect_modifying | L0 | 20 | 0.000153 | 0.000059 | 0.000035 | 0.000234 |
| triangle | nlg | effect_modifying | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| triangle | nlg | effect_modifying | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| collider | linear | additive | L0 | 20 | 0.052600 | 0.050669 | 0.020188 | 0.252324 |
| collider | linear | additive | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| collider | linear | additive | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| collider | linear | effect_modifying | L0 | 20 | 0.072741 | 0.035515 | 0.032399 | 0.163827 |
| collider | linear | effect_modifying | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| collider | linear | effect_modifying | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| collider | nlg | additive | L0 | 20 | 0.023374 | 0.012445 | 0.000000 | 0.061327 |
| collider | nlg | additive | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| collider | nlg | additive | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| collider | nlg | effect_modifying | L0 | 20 | 0.034564 | 0.016413 | 0.013063 | 0.088519 |
| collider | nlg | effect_modifying | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| collider | nlg | effect_modifying | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| chain | linear | additive | L0 | 20 | 0.012752 | 0.003964 | 0.003871 | 0.022222 |
| chain | linear | additive | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| chain | linear | additive | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| chain | linear | effect_modifying | L0 | 20 | 0.023313 | 0.006543 | 0.013235 | 0.040693 |
| chain | linear | effect_modifying | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| chain | linear | effect_modifying | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| chain | nlg | additive | L0 | 20 | 0.005238 | 0.002860 | 0.000000 | 0.011518 |
| chain | nlg | additive | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| chain | nlg | additive | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| chain | nlg | effect_modifying | L0 | 20 | 0.005398 | 0.003406 | 0.000603 | 0.011771 |
| chain | nlg | effect_modifying | L1-oracle | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| chain | nlg | effect_modifying | L2 | 20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

### 4.2 Rolled up to the topology, clause by clause

The criterion's clauses are counted separately because they can disagree.
`non_decreasing` and `largest_at_L0` are what "grows as knowledge decreases,
largest at L0" asserts; `strictly_increasing` is the stronger reading in which
EVERY rung grows; `flat` is the criterion's own disconfirming case.

| topology | n_cells | n_non_decreasing | n_largest_at_L0 | n_strictly_increasing | n_flat | meets_h2_criterion |
|---|---|---|---|---|---|---|
| triangle | 4 | 4 | 4 | 0 | 0 | YES |
| collider | 4 | 4 | 4 | 0 | 0 | YES |
| chain | 4 | 4 | 4 | 0 | 0 | YES |

### 4.3 VERDICT under the pinned rule

> **Gap_c:** the gap grows as knowledge decreases, largest at L0, extending [VonKugelgen2022] to a continuum; a flat gap disconfirms H2

**MET** — satisfied on 3/3 topologies (triangle, collider, chain).

- cells with a **flat** gap (the disconfirming case): **0**
- cells where the gap is **strictly increasing at every rung**: **0**

**Structure of the satisfied ladder, stated rather than smoothed.** PS-6 makes
`Gap_cost` exactly **zero** at L1-oracle and at L2 *by construction* — at those
rungs the acting model is the true one, so believed and realized coincide. The
ladder that satisfies Gap_c here is therefore `0 = 0 < positive`: **non-decreasing
and strictly largest at L0, but not strictly increasing at every rung.**
The criterion's operative clauses ("largest at L0"; "a flat gap disconfirms") are
met; its "continuum" language is met only in the two-point sense the design permits.
The empirical content of this table is the **L0 magnitude** and the fact that it
is nonzero on every cell — not the two zeros above it, which are structural.

## Comparison against the N=6 early look

**Not computed.** No comparison root was available on this run (`--compare-root` absent or the tree is not present), so the N=6 → N=20 comparison is UNAVAILABLE rather than assumed to agree.
