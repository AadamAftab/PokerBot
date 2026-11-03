# Adversarial evaluation report — Pokerbots baseline

Scope: my own simulator (standard heads-up no-limit hold'em, 100 bb stacks reset each hand, duplicate deals) against opponents I wrote.
**The official rules, protocol and duplicate-deal mechanics are still unavailable; nothing below is measured against the real engine.**

## 1. Ground truth and ledger
| item | value |
|---|---|
| Official rules / protocol / lifecycle / submission limits | **unresolved** (see §8 for the consequence of each) |
| Known-good baseline | `baseline_v0/` (files + `MANIFEST.sha256`), policy = `Champ(replay=False, risk=False)` |
| Regression suite before any change | 9/9 passed; 11/11 after adding opponents and tests (golden baseline test still bit-identical) |
| Source hashes of runs | dev `e5101c83eab79439`, held-out `1dad8ecb44c7b9e9` (differ only by league/test additions; policy files unchanged) |
| Equity sims per decision | capped at 300 for all arms (relative comparisons only) |

**Pre-registered protocol** (also in `adv_eval.py`): primary endpoint J = expected match points per 200-deal duplicate match; unit = one match with fresh bots and its own deal seed; secondary = chips/deal with a bootstrap over matches (deals within a match are not independent). Dev seeds = match ids 0–9 (exploration). Held-out seeds = ids 1000–1029 (confirmation only). Diagnostic seeds = 2000+ (separate again).
Raw results: `results/raw_*.json`, `results/diag_*.json`.

## 2. Ranked weaknesses of the baseline (held-out, 30 matches per cell)
| # | opponent | W-D-L | J [95% Wilson] | chips/deal [95% match-bootstrap] | dev J (10 matches) |
|---|---|---|---|---|---|
| 1 | Replayer (replay-aware copy of my own bot) | 0-0-30 | 0.00 [0.00, 0.11] | −4.1 [−4.7, −3.5] | 0.00 |
| 2 | ProberHalfFreq (stabs 50% of checks) | 4-0-26 | 0.13 [0.05, 0.30] | −2.0 [−2.8, −1.3] | 0.40 |
| 3 | Prober (stabs 85%, half pot) | 7-0-23 | 0.23 [0.12, 0.41] | −1.9 [−2.7, −1.1] | 0.20 |
| 4 | Prober33 | 8-0-22 | 0.27 [0.14, 0.44] | −1.4 [−2.4, −0.3] | 0.40 |
| 5 | Prober100 | 11-0-19 | 0.37 [0.22, 0.54] | −0.7 [−1.9, +0.4] | 0.40 |
| 6 | Prober75 | 13-0-17 | 0.43 [0.27, 0.61] | −0.1 [−1.0, +0.9] | 0.60 |
| 7 | Jammer (all-in every hand) | 18-0-12 | 0.60 [0.42, 0.75] | +2.7 [+0.7, +4.7] | 0.70 |
| 8 | Rock | 27-0-3 | 0.90 [0.74, 0.97] | +2.2 [+1.6, +2.8] | 0.90 |
| – | all others (13 opponents) | all wins | 1.00 | positive | – |

Ranks 2–6 are one root cause. Pooled over the five Prober variants the baseline wins 43/150 matches: **J = 0.29 [0.22, 0.36]**, significantly worse than a coin flip (z ≈ −5.2).
Root-cause grouping: (1) replay information, (2) over-folding to stabs, (3) range miscalibration against high-frequency aggressors (Jammer), (4) range miscalibration against value-heavy opponents (Rock).

## 3. Prober vulnerability — reproduced and explained (diagnostic seeds, instrumented)
- Hero folds far more than the break-even defence rate to postflop lead bets, and the excess grows with bet size:

| Prober bet size | Hero folds | break-even (MDF) fold rate |
|---|---|---|
| 0.33 pot | 21.6% | 23.2% |
| 0.5 pot | 46.1% | 33.3% |
| 0.75 pot | 58.3% | 41.8% |
| 1.0 pot | 71.5% | 50.0% |

- 42–45% of those folds had **true** equity (vs Prober's actual cards, oracle) above the pot-odds price.
- Chip attribution against Prober, per hand: "folded postflop to a bet" contributes −2.3 to −2.8; "faced an all-in and continued" (1–2% of hands) contributes −0.25 to −1.17.
- Capped check range: P(strong hand | Hero checks) = 8.2% versus 70.8% when Hero bets; P(check | strong) = 15%. Stabbing Hero's checks is therefore profitable.

## 4. Does bet size reveal hand strength?
- Information test (16,265 bets/raises, hand strength = equity vs a random hand): size alone as a monotone predictor of "strong" has AUC 0.456 [0.449, 0.463] — no usable monotone leak.
- But buckets differ: small bets (<0.45 pot) are strong 19% of the time, mid bets 76%, big bets 39%; raises are almost never weak (2.4%).
- Within the mid bucket, size separates value (≈0.65 pot) from bluff (≈0.60 pot) with **AUC 0.81**, because Hero's sizes are deterministic functions of strength class.
- Exploitation test (paired by match seed against a size-blind control, dev seeds): Hero's chips/deal 4.9 vs SizeBlind control, 4.0 vs SizeReader, 4.3 vs FineSizeReader (split at 0.625 pot). Intervals overlap; Hero still wins 10/10 and 30/30 matches. **Information leak confirmed, exploitable loss small and statistically inconclusive; no effect on J measured.**

## 5. Switching opponents (dev seeds, per-match cluster bootstrap)
- Switcher: late-match +9.7 [+8.9, +10.7] vs steady-state calling station +9.1 [+8.7, +9.5]. No adaptation cost.
- Flipper: right after the switch the gap to steady LAG is −3.9 [−8.4, +0.3] (inconclusive); late phase +9.0 vs +10.3 (overlapping).
- DelayedExploiter: chips/deal +2.7, +2.8, +2.8 before/during/after its observation window. This attack was ineffective in this form.

## 6. Audit of the "Bayesian" range model
It is not a Bayesian model: a hard top-q truncation (equivalent to a likelihood of 1 or 0, ignoring bluffs and slow-plays), with hand-set constants (2.2, 0.33, square root) and a global aggression statistic that mixes different spots; opponent statistics are point estimates with 3–6 observation priors and no uncertainty.
Measured calibration (bias = equity vs the opponent's true cards minus modelled equity, 23,824 decisions facing aggression):
- overall +0.054 [+0.048, +0.062] (model too pessimistic); lowest bin 0.12 modelled vs 0.25 true (+0.131); top bin −0.055 — the model is over-confident.
- sign depends on the opponent: vs Jammer +0.177 (q used 0.10); vs LAG +0.079; vs Prober family +0.02 to +0.09; vs BasicEq −0.054, Rock −0.075, TightPassive −0.108.
- no aggression faced (q = 1 assumption): overall bias +0.006 [−0.001, +0.012] — acceptable in this league, not proven in general.
- `spotq` reduces the overall bias to +0.031 and Jammer to +0.069, Prober33 to −0.006; residual bias remains in the lowest-equity bin (+0.091) and against LAG (+0.081).

## 7. Recommended single minimal patch and its held-out evidence
**Promote `spotq`** (`patches.py`, 35 lines, opponent-observed aggression frequency per spot replaces the fixed range-width rule). It was already implemented; no new code was written for this recommendation.
Held-out, paired by match seed, 30 matches per cell (95% match-bootstrap CIs):

| opponent | base J | spotq J | ΔJ [95% CI] | Δ chips/deal [95% CI] |
|---|---|---|---|---|
| Replayer | 0.00 | 0.17 | +0.17 [+0.03, +0.30] | +2.1 [+1.5, +2.7] |
| Prober | 0.23 | 0.70 | +0.47 [+0.27, +0.67] | +3.4 [+2.3, +4.5] |
| Prober33 | 0.27 | 0.87 | +0.60 [+0.43, +0.77] | +3.4 [+2.3, +4.5] |
| Prober75 | 0.43 | 0.73 | +0.30 [+0.07, +0.53] | +2.0 [+0.9, +3.2] |
| Prober100 | 0.37 | 0.80 | +0.43 [+0.23, +0.63] | +3.5 [+2.3, +4.8] |
| ProberHalfFreq | 0.13 | 0.93 | +0.80 [+0.63, +0.93] | +3.6 [+2.6, +4.5] |
| Jammer | 0.60 | 0.97 | +0.37 [+0.20, +0.53] | +14.8 [+11.7, +18.0] |
| Rock | 0.90 | 1.00 | +0.10 [0.00, +0.23] | +0.2 [−0.3, +0.6] |
| **pooled Prober family (150)** | **0.287** | **0.807** | **+0.520 [+0.427, +0.607]** | |

Eight comparisons were made; after a Bonferroni correction (α = 0.00625) the Prober, Prober33, Prober100, ProberHalfFreq and Jammer effects clearly survive; Replayer, Prober75 and Rock are borderline and should not be claimed individually.
Runtime and failures (under 18-way parallel load, so NOT production latency): ms/hand 113 (patch) vs 120 (base), p99 decision 112 vs 113 ms, max 482 vs 416 ms, peak RSS 73 vs 67 MB, caught exceptions 0 and 0, hero-illegal actions 0/417,455 and 0/732,341. On an idle core earlier: p50 5.6 ms, p99 24 ms, max 33 ms per decision.

## 8. Limits and unresolved questions
- **Opponents are mine.** The patch and the Prober/Jammer attacks were designed by the same author from the same understanding of the bot; gains against independently written bots are unproven.
- **Regression on the remaining league was not re-run on held-out seeds for the patch** (those cells are saturated at J = 1.0). Dev run on 11 opponents earlier showed no significant chip loss (Random −2.5 ± 2.7).
- Unresolved official facts and what each changes: variant rules (everything), protocol and process lifecycle (replay logic, opponent statistics), duplicate ordering (replay detection), artifact and time limits (sims budget), tie-break formula (whether chips matter).
- Not yet tested: bet-size diversification, check-range protection (check-raises/traps), Bayesian mixture models, strategy-switch detection, equilibrium baseline, extreme-state suites, and the real sandbox. Residual weaknesses after the patch: Replayer 0.17, Prober75 0.73, Prober 0.70.

## 9. Reproduce
```
python test_regression.py
python adv_eval.py dev base all base            # then: python adv_report.py results/raw_dev_base.json
python adv_eval.py heldout base Replayer,Prober,Prober33,Prober75,Prober100,ProberHalfFreq,Jammer,Rock,FineSizeReader,SizeBlind,Flipper,DelayedExploiter,Switcher base
python adv_eval.py heldout spotq Replayer,Prober,Prober33,Prober75,Prober100,ProberHalfFreq,Jammer,Rock spotq
python diag_run.py base Prober,Prober33,Prober75,Prober100,ProberHalfFreq,Jammer,Rock,BasicEq,SizeReader,TightPassive,LAG 12 100 diag   # and the same with spotq
python diag_analyze.py prober|size|calib base|spotq [opponents]
```
