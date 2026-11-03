# Improving the Discard Hold'em bot: research and ranked plan (3 Nov 2026)

Evidence levels: **[M]** measured in this repo, **[L]** from literature (search snippets only, primary papers NOT opened),
**[R]** my reasoning, untested.

## 0. A correction from measurement
Earlier I listed "faster evaluator -> more simulations" as an improvement. **[M] Falsified.** Same deals, 12 matches x 400 hands,
bot vs a fixed copy of itself (`results/dh_compute.log`):

| sims per decision | J | chips/hand (95% CI) |
|---|---|---|
| 60  | 0.50 | +0.03 +/- 0.38 |
| 200 (current) | 0.58 | +0.03 +/- 0.46 |
| 600 | 0.25 | -0.35 +/- 0.36 |

No benefit from 10x compute (600 is, if anything, nominally worse, within noise). Simulation noise is not the bottleneck;
**decision logic and opponent modelling are.** Consequence: cut the default sims to ~100 to buy a larger time-bank margin for free.

## 1. Ranked improvements

### Tier 1: highest expected value per effort
1. **Tune the hand-set thresholds by paired self-play + held-out league.** [M/R] Every constant in `_act` (0.62/0.80/0.45 equity cut-offs,
   bet fractions 0.33/0.65/0.8, open/defend percentiles, bluff frequencies) was set by hand and never tuned. Since compute does not
   help, these constants are the likely leverage. Method: random/CMA search on ~10 parameters, paired duplicate deals, evaluate on a
   held-out opponent set that is never used for tuning; accept only gains that survive held-out data (guards against overfitting to my own league).
2. **Randomise and balance (reduce exploitability).** [L/R] My value/bluff/call decisions are near-deterministic functions of equity,
   so an adaptive opponent can read them. Standard remedy: bluff:value ratio matched to bet size (bluff share = bet/(pot+2*bet)) and
   calls at minimum-defence frequency; mix near thresholds instead of hard cut-offs. Principle behind equilibrium play in Libratus/Pluribus-style bots.
3. **Strong/adaptive test opponents.** [R] Everything so far was measured against bots I wrote. Add (a) an opponent that learns my
   thresholds, (b) size-based bettors, (c) a small CFR-solved opponent in a reduced game. Without this, "wins every match" is weak evidence.

### Tier 2: solid, moderate effort
4. **Richer opponent profile with confidence gating.** [L] Hoehn et al. (AAAI 2005) report that learning a maximal exploit within a
   few hundred hands is impractical but good short-term exploitation is feasible, i.e. exploit coarse tendencies only. Extend the
   current gate to VPIP/PFR, 3-bet, c-bet, fold-to-c-bet, aggression factor, showdown frequency, each a Beta posterior; make the
   deviation size proportional to confidence (the restricted-Nash-response idea: exploit up to a limit, stay near equilibrium when unsure).
5. **Bayesian model of the opponent's *discard policy*.** [R] Today: union of "heuristic keep" and "lowest card" with a consistency
   counter. Better: posterior over a small set of policies (lowest rank, heuristic keep, keep-pairs, keep-high, random) updated from
   every showdown triple (their 3 cards become public then). Sharper opponent ranges after the discard = better turn/river decisions.
6. **Preflop from a real solver.** [R] My 3-card table is equity vs a *random* hand. Preflop is small enough to solve offline with
   CFR over ~100 hand buckets and the 4-raise cap; this is the one place a genuine equilibrium blueprint is feasible without a cluster.
7. **Check-range inference and bet-size reading for Discard Hold'em.** [M in standard HE: size leakage was measurable; untested here]
   Condition the opponent range on checks (capped range) and on size, not only on frequency.

### Tier 3: situational
8. **Match-state variance control.** [R] Tie-break clips each match at +/-C (400 / 1000), about 1 chip per hand. My bot wins 5-14 chips/hand
   against weak bots, so most of that is wasted. Late in a match: protect a lead well above C; when far behind, gamble (the loss is
   clipped anyway). Earlier standard-HE tests showed no match-win gain, so re-test here before shipping.
9. **Own-discard signalling.** [R] The PDF argues for hiding high-card blockers on near-ties. Unproven; test against an
   opponent that reacts to discards. My near-tie rule already discards the lower card.
10. **Evaluation variance reduction (AIVAT-style).** [L] AIVAT is reported to cut required hands by >10x in HUNL evaluation; it needs
   known strategies for the evaluated agents (fine for my league bots). Would let each experiment run on far fewer matches.

### Not worth it here
- **Real-time subgame solving (Modicum / nested solving).** [L/R] Needs ~hundreds of ms per decision; the time bank gives ~150 ms/hand
  total in Stage 1. Maybe river-only in Stage 2; not now.
- **Full MCCFR blueprint with EMD buckets.** [L/R] Weeks of cluster time; 5 MB limit; no way to verify the PDF's size claim.
- **Cactus-Kev evaluator / more simulations.** [M] Compute is not the bottleneck (section 0).

## 2. What I did not verify
Search returned snippets, not full papers; several recent arXiv entries (2026) are unreviewed and I did not rely on them. No source
analysed *simultaneous-discard* Hold'em specifically (closest: Crazy Pineapple strategy articles, which are forum-grade).

## Sources
- Johanson, Zinkevich, Bowling, Computing Robust Counter-Strategies (restricted Nash response): https://www.cs.ualberta.ca/~games/poker/publications/NIPS07-rnash.pdf
- Johanson & Bowling, Data Biased Robust Counter Strategies: https://webdocs.cs.ualberta.ca/%7Ebowling/papers/09aistats.pdf
- Hoehn et al., Effective Short-Term Opponent Exploitation in Simplified Poker: https://cdn.aaai.org/AAAI/2005/AAAI05-123.pdf
- Brown & Sandholm, Safe and Nested Subgame Solving: https://arxiv.org/pdf/1705.02955
- Brown, Sandholm, Amos, Depth-Limited Solving (Modicum): https://arxiv.org/pdf/1805.08195
- Burch et al., AIVAT: https://ojs.aaai.org/index.php/AAAI/article/view/11481
- Liu et al., Safe Opponent-Exploitation Subgame Refinement (NeurIPS 2022): https://proceedings.neurips.cc/paper_files/paper/2022/file/b12a1d1014e952e676f5d6931d03241a-Paper-Conference.pdf
- Crazy Pineapple discard guidance (forum/strategy articles): https://www.pokernews.com/strategy/home-game-heroes-strategy-extra-hole-cards-flop-games-22385.htm
