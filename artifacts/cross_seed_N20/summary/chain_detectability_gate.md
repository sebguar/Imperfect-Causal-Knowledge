# PS-2 chain detectability gate — first-6-seed checkpoint

Evaluated on seeds 0-5 of the N=20 grid (`results/cross_seed_N20/`), per
PS-1's staged 6→20 discipline and its prefix property. Same rule as
the SCM specification's collider amendment.

- **(i)** `sign(Δ_cost(L2))` consistent across all 6 seeds AND
  `|mean(Δ_cost(L2))| > SD(Δ_cost(L2))`.
- **(ii)** `sign(ΔS)` unambiguous under the production S(g) — binding on the
  **effect-modifying arm only**. Under the additive control ΔS is flat by
  construction (PS-3 convention 4) — exactly 0 on the linear branch,
  within MC noise on the NLG branch — so (ii) checks the PS-3
  fit-independence invariance there instead of a sign.

| family | regime | mean | sd | \|mean\|/sd | cond_i_sign | cond_i_mag | cond_i | ΔS | ΔS stable | cond_ii | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| linear | additive | 2.302231 | 0.079133 | 29.09 | YES | YES | YES | +0.000000 | YES | YES | YES |
| linear | effect_modifying | 2.309343 | 0.100232 | 23.04 | YES | YES | YES | -0.476973 | YES | YES | YES |
| nlg | additive | 1.423975 | 0.039646 | 35.92 | YES | YES | YES | +0.001771 | YES | YES | YES |
| nlg | effect_modifying | 1.449895 | 0.041994 | 34.53 | YES | YES | YES | -0.342228 | YES | YES | YES |

### Per-seed Δ_cost(L2)

| family | regime | per-seed |
|---|---|---|
| linear | additive | [+2.222791, +2.366961, +2.323926, +2.397164, +2.195897, +2.306650] |
| linear | effect_modifying | [+2.387639, +2.438316, +2.271988, +2.345411, +2.164180, +2.248522] |
| nlg | additive | [+1.395959, +1.435569, +1.454200, +1.466019, +1.359316, +1.432786] |
| nlg | effect_modifying | [+1.470735, +1.470806, +1.458497, +1.439440, +1.370928, +1.488967] |

### Verdict

**GATE PASSES on both chain effect-modifying cells.** PS-2 is discharged
for the chain, joining the triangle and the collider:
all three topologies now have discharged detectability gates.

The gate is **not re-evaluated at N=20** — PS-1 forecloses re-evaluation
on a growing seed count, which would reintroduce stopping-rule freedom.