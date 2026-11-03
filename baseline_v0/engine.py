"""Heads-up no-limit hold'em engine with duplicate matches.

Seat 0 is the button/small blind (acts first preflop, last postflop). Stacks reset
every hand. Bots implement:
    new_hand(is_button)            -> None
    act(state)                     -> 'f' | 'c' | ('r', raise_to_total)
    observe(actor_is_me, kind, state) -> None   (called after every action)
    end_hand(info)                 -> None   (my_hole, opp_hole if shown down, board seen, chips)
"""
import numpy as np
from pokerlib import eval_best

SB, BB, STACK = 1, 2, 200
STATS = {'actions': 0, 'illegal': 0, 'hero_illegal': 0, 'log': []}   # illegal = would be rejected by a strict engine


def play_hand(bots, holes, board):
    """bots[0] sits on the button. Returns chips won by seat 0."""
    stacks = [STACK - SB, STACK - BB]
    bets = [SB, BB]
    hist = []
    for b, is_btn in zip(bots, (True, False)):
        b.new_hand(is_btn)
    folded = None

    for street, nshown in enumerate((0, 3, 4, 5)):
        if street > 0:
            bets = [0, 0]
        cur = 0 if street == 0 else 1
        acted = [False, False]
        last_inc = BB
        shown = [int(c) for c in board[:nshown]]
        while True:
            p, o = cur, 1 - cur
            if stacks[p] == 0 or (stacks[o] == 0 and bets[p] >= bets[o]):
                break
            to_call = bets[o] - bets[p]
            max_to = min(bets[p] + stacks[p], bets[o] + stacks[o])
            min_to = min(bets[o] + max(last_inc, BB), max_to)
            pot_total = (2 * STACK - stacks[0] - stacks[1])
            state = dict(hole=[int(c) for c in holes[p]], board=shown, street=street,
                         pot=pot_total, to_call=to_call, my_bet=bets[p], opp_bet=bets[o],
                         my_stack=stacks[p], opp_stack=stacks[o], min_raise_to=min_to,
                         max_raise_to=max_to, is_button=(p == 0), bb=BB,
                         hist=[(a == p, k, amt) for a, k, amt in hist])
            a = bots[p].act(state)
            STATS['actions'] += 1
            if not (a == 'c' or (a == 'f' and to_call > 0) or
                    (isinstance(a, tuple) and a[0] == 'r' and (min_to <= a[1] <= max_to))):
                STATS['illegal'] += 1
                STATS['hero_illegal'] += bool(getattr(bots[p], 'is_hero', False))
                if len(STATS['log']) < 30:
                    STATS['log'].append((type(bots[p]).__name__, a, to_call, min_to, max_to, bets[p], bets[o]))
            if a == 'f' and to_call > 0:
                kind, amt = 'f', 0
            elif isinstance(a, tuple) and a[0] == 'r' and max_to > bets[o] and max_to > bets[p]:
                to = int(max(min_to, min(a[1], max_to)))
                inc = to - bets[o]
                stacks[p] -= to - bets[p]
                bets[p] = to
                last_inc = max(last_inc, inc)
                acted[o] = False
                kind, amt = 'r', to
            else:  # call / check (illegal folds of a free check become checks)
                pay = min(to_call, stacks[p])
                stacks[p] -= pay
                bets[p] += pay
                kind, amt = 'c', pay
            acted[p] = True
            hist.append((p, kind, amt))
            for i in (0, 1):
                bots[i].observe(i == p, kind, state)
            if kind == 'f':
                folded = p
                break
            if acted[0] and acted[1] and bets[0] == bets[1]:
                break
            cur = o
        if folded is not None:
            break

    invested = [STACK - stacks[0], STACK - stacks[1]]
    revealed = False
    if folded is not None:
        res = invested[1] if folded == 1 else -invested[0]
    else:
        revealed = True
        full = np.array(board, dtype=np.int64)
        s0 = eval_best(np.concatenate([holes[0], full])[None, :])[0]
        s1 = eval_best(np.concatenate([holes[1], full])[None, :])[0]
        res = invested[1] if s0 > s1 else (-invested[0] if s0 < s1 else 0)
    seen = [int(c) for c in (board if revealed else board[:nshown])]
    for i in (0, 1):
        bots[i].end_hand(dict(my_hole=[int(c) for c in holes[i]],
                              opp_hole=[int(c) for c in holes[1 - i]] if revealed else None,
                              board=seen, chips=res if i == 0 else -res))
    return res


def duplicate_match(make_a, make_b, n_deals, seed=0):
    """Play n_deals duplicate deals; returns per-deal chip result for A (list)."""
    rng = np.random.default_rng(seed)
    a, b = make_a(), make_b()
    results = []
    for _ in range(n_deals):
        cards = rng.permutation(52)
        h0, h1, board = cards[:2], cards[2:4], cards[4:9]
        d1 = play_hand([a, b], [h0, h1], board)        # A on button with h0
        d2 = play_hand([b, a], [h0, h1], board)        # B on button with h0, A holds h1
        results.append(d1 - d2)
    return results
