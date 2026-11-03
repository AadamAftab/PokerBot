# Discard Hold'em â€” spec digest (source of truth map)

Sources: **Protocol Specification v1.0** (PS) and **Rulebook** (RB), both supplied by the organisers.
The Problem Statement (full rules + worked examples, released 1 Nov 2026) and the organisers' starter kit (engine.py) have since been read and tested against; the bot and my engine agree with the official engine hand by hand (parity_test.py).
Everything below is "CONFIRMED (PS/RB)" unless it says otherwise.

## Game
- Heads-up no-limit Hold'em, **3 hole cards**, blinds 1/2, stacks reset to 200 every hand, seats alternate every hand.
- After the **flop betting** (only if nobody folded) both players **discard one hole card simultaneously, face up**.
  Showdown = best 5 of the remaining 2 hole cards + 5 board cards. Discard happens even if both are all-in.
- Max 4 bets/raises per street (the big blind counts as the first raise preflop â†’ `raises_left` starts at 3 preflop, 4 postflop).
- `min_raise = opp_bet + max(s, 2)`, `s` = size of last bet/raise on the street; if all-in is below that, only the all-in raise is legal.
- Button acts first preflop; the big blind acts first on every later street.

## Protocol
- JSON lines over stdin/stdout, UTF-8; flush every reply; **nothing but protocol JSON on stdout** (debug â†’ stderr, 64 KB cap).
- Messages: `init`, `new_hand`, `action_request`, `discard_request`, `discard_result`, `hand_result`, `notice`, `match_end`.
  Replies: `{"type":"action","action":"fold|check|call|raise","amount":N}` and `{"type":"discard","card":"Kd"}`.
- Ignore unknown message types/fields. Reply when nothing is outstanding is ignored.
- Amounts are "to" totals on the street. `history` in every request lists all voluntary actions (blinds omitted).
- `hand_result` reveals the opponent's two remaining cards **only at showdown**, plus their discard.
- Illegal/unparsable action â†’ `check` if legal else `fold` (+notice). Bad discard â†’ lowest-ranked card (ties: s,h,d,c order).
- Crash / EOF / over time â†’ forfeit: loss, opponent credited Î”=+C.

## Limits (marked *provisional*)
- Time bank per match: 60 s / 400 hands (Stage 1), 150 s / ~1000 hands (Stage 2). Charged from request to full reply line.
  Per reply â‰¤ 5 s. Wall-clock kill 180 s / 450 s. Start-up â‰¤ 5 s (not charged). 1 vCPU, 1 GB RAM, no network, read-only FS.
- Python 3.11, stdlib + numpy + scipy. Upload: `bot.py` or a zip â‰¤ 5 MB (per the portal text I was shown earlier).
- â†’ ~150 ms/hand on average; the bot's governor uses `time_left_ms` in every request.

## Scoring (RB)
- Match result win/draw/loss on net chips; Swiss pairing on match points then capped chips.
- Tie-break Î£ clip(Î”_m, Â±C) / Î£H with C=400 (Stage 1) / 1000 (Stage 2). Stage 2 alone decides prizes.
- Duplicate deals exist ("the bot is not told which deals are duplicates"). Exact pairing/order/seat swap **UNCONFIRMED**.

## Rule 10.3 (RB) â€” replay / duplicate-memory
Teams must report weaknesses, not use them. Using the fact that duplicate deals expose the opponent's cards from an earlier hand
could be read as exploiting an engine weakness. **Decision: the bot stores no cross-hand card information and has no replay logic.**
Aggregate action-frequency opponent modelling (spot stats) is kept; it uses only public betting actions.

## Unresolved â†’ ask the organisers (draft below)
1. Duplicate deals: are hands paired (adjacent / shuffled)? Do players swap seats with identical cards? Same board? (affects nothing in the bot, only evaluation design)
2. Is remembering the opponent's *shown-down cards* from earlier hands and using them in the same match allowed? (we are NOT doing it; asking for clarity)
3. Provisional time bank/limits: final values? Is `time_left_ms` before or after the current request is charged?
4. Does a short all-in raise (< min raise) count towards the 4-raise cap and re-open the action?
5. Is a starter kit / local match runner being released? (our own engine implements PS exactly; cannot claim parity until we have theirs)
6. Is the opponent's discard shown to the bot before the turn betting? (PS says yes: `discard_result`.)

### Draft message
> Hello â€” two clarification requests about the Pokerbots rules. (1) Rule 10.3: our bot keeps only aggregate betting-frequency
> statistics about the opponent within a match (no memory of earlier hands' cards). Is that permitted, and is any use of cards
> revealed at showdown in earlier hands (e.g. through duplicate deals) intended to be disallowed? (2) Could you confirm how
> duplicate deals are paired (adjacent hands? seats swapped with identical hole cards and board?) and whether the provisional time
> bank values (60 s / 400 hands) are final? Thank you.

## Engine fidelity statement
`dh_engine.py` implements PS 1.0 from the text only. Parity with the organisers' engine is **unverified** until their starter kit /
runner is available. The test-suite (`test_dh.py`) pins the PS example hand, raise cap, min-raise formula, corrections, forfeits.

