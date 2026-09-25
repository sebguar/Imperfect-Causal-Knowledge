# Detectability-gate inputs (for the analysis pass)

 PS-2 gate, stated for reference — **not evaluated here**:

1. `sign(Δ_cost(L2))` consistent across all 6 seeds, AND
   `|mean(Δ_cost(L2))|` across seeds `>` per-seed SD of `Δ_cost(L2)`.
2. `sign(ΔS)` unambiguous under the production S(g), effect-modifying regime.

The grid run produces these numbers; **the analysis pass evaluates the gate and writes the H1 verdict**.

Pre-freeze the SCM amendment reference values (NLG collider, n_mc=10,000):
additive `S(−1) ≈ 0.715, S(+1) ≈ 0.718, ΔS ≈ −0.003` at `|ΔS|/MC_SE ≈ 1.4`;
effect-modifying `S(−1) ≈ 0.582, S(+1) ≈ 0.885, ΔS ≈ −0.303` at `|ΔS|/MC_SE ≈ 131`.

---

## Δ_cost(L2) across seeds

| topology | family | regime | per-seed Δ_cost(L2) | mean | SD | all same sign? | \|mean\|/SD |
|---|---|---|---|---|---|---|---|
| triangle | linear | additive | +3.711041, +3.699872, +3.828214, +3.656926, +3.757926, +3.712263 | +3.727707 | 0.058849 | YES | 63.34 |
| triangle | linear | effect_modifying | +2.917041, +2.806173, +3.063483, +3.061010, +2.963917, +2.932181 | +2.957301 | 0.097124 | YES | 30.45 |
| triangle | nlg | additive | +3.339752, +3.316595, +3.354105, +3.419952, +3.362755, +3.248467 | +3.340271 | 0.056630 | YES | 58.98 |
| triangle | nlg | effect_modifying | +2.243603, +2.208267, +2.244398, +2.311398, +2.228672, +2.199343 | +2.239280 | 0.039787 | YES | 56.28 |
| collider | linear | additive | +4.709108, +4.909964, +4.945060, +4.969678, +4.471408, +4.723939 | +4.788193 | 0.191368 | YES | 25.02 |
| collider | linear | effect_modifying | +5.534824, +5.328163, +5.376558, +5.429149, +4.665147, +5.300052 | +5.272316 | 0.308842 | YES | 17.07 |
| collider | nlg | additive | +1.856401, +1.953298, +1.746571, +1.811776, +1.742498, +1.757730 | +1.811379 | 0.082451 | YES | 21.97 |
| collider | nlg | effect_modifying | +2.074206, +2.184747, +1.900518, +1.975311, +1.992543, +1.873809 | +2.000189 | 0.114999 | YES | 17.39 |
| chain | linear | additive | (no runs) | | | | |
| chain | linear | effect_modifying | (no runs) | | | | |
| chain | nlg | additive | (no runs) | | | | |
| chain | nlg | effect_modifying | (no runs) | | | | |

## S(g) under the production descriptor (PS-3 fit-independence check)

Recomputed once per seed. Under PS-3 S(g) depends only on the
ground-truth SCM (the NLG branch's only randomness is the pinned `sg_mc`
construction stream), so every value below must be **identical across all
seeds** — a seed-varying value would mean the experiment seed leaked in.

| topology | family | regime | S(−1) | S(+1) | ΔS | identical across seeds? |
|---|---|---|---|---|---|---|
| triangle | linear | additive | 0.500000 | 0.500000 | +0.000000 | YES |
| triangle | linear | effect_modifying | 0.300000 | 0.900000 | -0.600000 | YES |
| triangle | nlg | additive | 0.157420 | 0.155437 | +0.001983 | YES |
| triangle | nlg | effect_modifying | 0.094452 | 0.279787 | -0.185335 | YES |
| collider | linear | additive | 0.707107 | 0.707107 | +0.000000 | YES |
| collider | linear | effect_modifying | 0.538516 | 0.943398 | -0.404882 | YES |
| collider | nlg | additive | 0.716305 | 0.715807 | +0.000497 | YES |
| collider | nlg | effect_modifying | 0.582934 | 0.882093 | -0.299159 | YES |
| chain | linear | additive | 0.750000 | 0.750000 | +0.000000 | YES |
| chain | linear | effect_modifying | 0.547723 | 1.024695 | -0.476973 | YES |
| chain | nlg | additive | 0.648251 | 0.646480 | +0.001771 | YES |
| chain | nlg | effect_modifying | 0.524108 | 0.866336 | -0.342228 | YES |
