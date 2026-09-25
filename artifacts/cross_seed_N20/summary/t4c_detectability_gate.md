# Detectability-gate formal record (PS-2)

Formal discharge of **PS-2** for the collider and the analogous check for
every cell, evaluated on the **PS-1 first-6-seed early look**.

> **This is a 6-seed gate on an N=20 tree, and that is the
> discipline working, not a stale number.** PS-1 forecloses re-evaluating the
> gate on a growing seed count — doing so would reintroduce exactly the
> stopping-rule freedom the staged 6→20 design exists to prevent. The gate is
> computed over the seed-0..5 prefix (the seed-generation prefix property),
> and every cell is asserted equal to its frozen record before this document is
> written. The `n_seeds_evaluated_for_gate` column below carries that fact into
> the artifact, as the chain SCM specification requires.

- **(i)** `sign(Δ_cost(L2))` consistent across all 6 seeds **and**
  `|mean(Δ_cost(L2))| > SD(Δ_cost(L2))`.
- **(ii)** `sign(ΔS)` unambiguous under the **production** S(g). S(g) is
  data-, classifier- and fit-independent by construction (PS-3), so it
  must also be identical across seeds; that invariance is re-checked here
  rather than assumed. In the **additive control** ΔS is flat *by
 construction* (PS-3 convention 4), so (ii) requires only the invariance.

| topology | family | regime | mean | sd | abs_mean_over_sd | cond_i_sign_consistent | cond_i_magnitude | cond_i_pass | delta_S | delta_S_identical_across_seeds | cond_ii_pass | cond_ii_note | gate_pass | n_seeds_evaluated_for_gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| triangle | linear | additive | 3.727707 | 0.058849 | 63.343897 | YES | YES | YES | 0.000000 | YES | YES | N/A (additive control: Delta S flat by construction) | YES | 6 |
| triangle | linear | effect_modifying | 2.957301 | 0.097124 | 30.448666 | YES | YES | YES | -0.600000 | YES | YES | sign unambiguous | YES | 6 |
| triangle | nlg | additive | 3.340271 | 0.056630 | 58.984023 | YES | YES | YES | 0.001983 | YES | YES | N/A (additive control: Delta S flat by construction) | YES | 6 |
| triangle | nlg | effect_modifying | 2.239280 | 0.039787 | 56.281365 | YES | YES | YES | -0.185335 | YES | YES | sign unambiguous | YES | 6 |
| collider | linear | additive | 4.788193 | 0.191368 | 25.020910 | YES | YES | YES | 0.000000 | YES | YES | N/A (additive control: Delta S flat by construction) | YES | 6 |
| collider | linear | effect_modifying | 5.272316 | 0.308842 | 17.071250 | YES | YES | YES | -0.404882 | YES | YES | sign unambiguous | YES | 6 |
| collider | nlg | additive | 1.811379 | 0.082451 | 21.969051 | YES | YES | YES | 0.000497 | YES | YES | N/A (additive control: Delta S flat by construction) | YES | 6 |
| collider | nlg | effect_modifying | 2.000189 | 0.114999 | 17.393109 | YES | YES | YES | -0.299159 | YES | YES | sign unambiguous | YES | 6 |
| chain | linear | additive | 2.302231 | 0.079133 | 29.093254 | YES | YES | YES | 0.000000 | YES | YES | N/A (additive control: Delta S flat by construction) | YES | 6 |
| chain | linear | effect_modifying | 2.309343 | 0.100232 | 23.040000 | YES | YES | YES | -0.476973 | YES | YES | sign unambiguous | YES | 6 |
| chain | nlg | additive | 1.423975 | 0.039646 | 35.917163 | YES | YES | YES | 0.001771 | YES | YES | N/A (additive control: Delta S flat by construction) | YES | 6 |
| chain | nlg | effect_modifying | 1.449895 | 0.041994 | 34.526543 | YES | YES | YES | -0.342228 | YES | YES | sign unambiguous | YES | 6 |

### Per-seed Δ_cost(L2)

| topology | family | regime | delta_cost_L2_per_seed |
|---|---|---|---|
| triangle | linear | additive | [3.711041, 3.699872, 3.828214, 3.656926, 3.757926, 3.712263] |
| triangle | linear | effect_modifying | [2.917041, 2.806173, 3.063483, 3.06101, 2.963917, 2.932181] |
| triangle | nlg | additive | [3.339752, 3.316595, 3.354105, 3.419952, 3.362755, 3.248467] |
| triangle | nlg | effect_modifying | [2.243603, 2.208267, 2.244398, 2.311398, 2.228672, 2.199343] |
| collider | linear | additive | [4.709108, 4.909964, 4.94506, 4.969678, 4.471408, 4.723939] |
| collider | linear | effect_modifying | [5.534824, 5.328163, 5.376558, 5.429149, 4.665147, 5.300052] |
| collider | nlg | additive | [1.856401, 1.953298, 1.746571, 1.811776, 1.742498, 1.75773] |
| collider | nlg | effect_modifying | [2.074206, 2.184747, 1.900518, 1.975311, 1.992543, 1.873809] |
| chain | linear | additive | [2.222791, 2.366961, 2.323926, 2.397164, 2.195897, 2.30665] |
| chain | linear | effect_modifying | [2.387639, 2.438316, 2.271988, 2.345411, 2.16418, 2.248522] |
| chain | nlg | additive | [1.395959, 1.435569, 1.4542, 1.466019, 1.359316, 1.432786] |
| chain | nlg | effect_modifying | [1.470735, 1.470806, 1.458497, 1.43944, 1.370928, 1.488967] |

**Record: 12 / 12 cells PASS.**

### Provenance — where each cell's gate was originally discharged

The three frozen records cover the grid between them but none covers it alone.
Every row below was re-derived from the seed-0..5 prefix of this tree and
checked, field by field, against the record named in `source_record` at that
record's own rendered precision. A mismatch would have raised before this
document was written.

| topology | family | regime | mean_frozen | mean_first6 | sd_frozen | sd_first6 | abs_mean_over_sd_frozen | abs_mean_over_sd_first6 | delta_S_frozen | delta_S_first6 |
|---|---|---|---|---|---|---|---|---|---|---|
| triangle | linear | additive | 3.727707 | 3.727707 | 0.058849 | 0.058849 | 63.343897 | 63.343897 | 0.000000 | 0.000000 |
| triangle | linear | effect_modifying | 2.957301 | 2.957301 | 0.097124 | 0.097124 | 30.448666 | 30.448666 | -0.600000 | -0.600000 |
| triangle | nlg | additive | 3.340271 | 3.340271 | 0.056630 | 0.056630 | 58.984023 | 58.984023 | 0.001983 | 0.001983 |
| triangle | nlg | effect_modifying | 2.239280 | 2.239280 | 0.039787 | 0.039787 | 56.281365 | 56.281365 | -0.185335 | -0.185335 |
| collider | linear | additive | 4.788193 | 4.788193 | 0.191368 | 0.191368 | 25.020910 | 25.020910 | 0.000000 | 0.000000 |
| collider | linear | effect_modifying | 5.272316 | 5.272316 | 0.308842 | 0.308842 | 17.071250 | 17.071250 | -0.404882 | -0.404882 |
| collider | nlg | additive | 1.811379 | 1.811379 | 0.082451 | 0.082451 | 21.969051 | 21.969051 | 0.000497 | 0.000497 |
| collider | nlg | effect_modifying | 2.000189 | 2.000189 | 0.114999 | 0.114999 | 17.393109 | 17.393109 | -0.299159 | -0.299159 |
| chain | linear | additive | 2.302231 | 2.302231 | 0.079133 | 0.079133 | 29.09 | 29.09 | 0.000000 | 0.000000 |
| chain | linear | effect_modifying | 2.309343 | 2.309343 | 0.100232 | 0.100232 | 23.04 | 23.04 | -0.476973 | -0.476973 |
| chain | nlg | additive | 1.423975 | 1.423975 | 0.039646 | 0.039646 | 35.92 | 35.92 | 0.001771 | 0.001771 |
| chain | nlg | effect_modifying | 1.449895 | 1.449895 | 0.041994 | 0.041994 | 34.53 | 34.53 | -0.342228 | -0.342228 |

The machine-readable consolidation of this table — one row per cell, all three
topologies, carrying `n_seeds_evaluated_for_gate` and `source_record` — is
written alongside as `detectability_gate_provenance.csv`. It carries FROZEN
FACTS ONLY and establishes no display convention for any figure.
