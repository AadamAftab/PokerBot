# Replay (duplicate-deal) inference — status, facts vs assumptions, and tests to run once the spec arrives

**Status: no replay-dependent behaviour is to be added or enabled.** The production candidate is `spotq` with `replay=False`.
`NoReplaySpotQ` (patches.py) is the replay-disabled control: it stores no card history across hands.

## What the audit measured (my simulator only)
Against a replay-aware copy of the same policy, spotq loses −2.71 chips/deal (wins 2/16 matches); against the replay-DISABLED mirror
it is +0.83 chips/deal (11/16). Of the replay-aware opponent's gain: ~72% comes from the ~5% of hands where it also knew the whole
board, ~28% from hands where it knew Hero's exact cards but not the runout. Its detection was ~99% accurate with ~0.1% false beliefs.
No change of play can hide hole cards from a bot that held them earlier, so the exposure is structural, not a strategy leak.

## Facts vs assumptions
| claim the inference relies on | guaranteed by the official engine? |
|---|---|
| each deal is played twice with seats swapped (stated on the event page) | **stated, mechanism not specified** |
| both halves use the same hole cards and the same board | **assumption** (could be re-shuffled board, or only cards swapped) |
| the two halves are consecutive (alternating pattern) | **assumption** (could be batched, interleaved, or randomised) |
| the opponent's showdown cards are revealed to the bot | **assumption** (standard poker, not confirmed here) |
| the bot process persists across hands (memory survives) | **assumption** (could restart every hand/deal) |
| a bot may remember and reuse its own earlier cards | **unknown — rules prohibit collusion and require code review; no statement on this** |
| the opponent's earlier hole cards are exactly my current cards | follows only if the first three rows hold |

## Tests to run when the official spec / engine is available
1. Read the duplicate-deal implementation: what is repeated (cards, board, seed), order of halves, seat/identity swap.
2. Process lifecycle: log PIDs and an in-process counter across hands, duplicate halves, matches, rounds.
3. Showdown semantics: are the opponent's hole cards delivered on showdown, and in what field?
4. Written answer from the organizers on whether remembering one's own earlier cards/boards is permitted. If ambiguous, treat as NOT permitted.
5. If permitted: measure false-positive rate of the detector on official-engine deal logs (the preflop showdown-card match has ~1/1326 chance collisions;
   the lag rule and flop checks exist to suppress them); require 0 false positives in >=10,000 official hands before enabling.
6. Re-run the paired audit (`python replay_audit.py`) against the official engine's replay-aware vs replay-disabled opponents.
7. Regression: `NoReplaySpotQ` must keep passing `test_no_replay_control_stores_no_card_history_and_never_uses_replay`.
