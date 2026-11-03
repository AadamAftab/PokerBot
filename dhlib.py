"""Discard Hold'em primitives (numpy only): cards, evaluator, discard heuristic, 3-card preflop
table, and Monte-Carlo equity that understands the discard stage.

Cards are ints 0..51: rank = c >> 2 (0='2' ... 12='A'), suit = c & 3 with suit order s,h,d,c
(the order the rulebook uses for discard tie-breaks).
"""
import itertools
import numpy as np

RANKS = "23456789TJQKA"
SUITS = "shdc"
_R = np.arange(13)
_B = 14


def parse_card(s):
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


def card_str(c):
    return RANKS[c >> 2] + SUITS[c & 3]


# ----------------------------------------------------------------------------- evaluator
COMBOS = {k: np.array(list(itertools.combinations(range(k), 5)), dtype=np.int64) for k in (5, 6, 7)}


def eval5(cards):
    """cards (M,5) -> (M,) int64 score, higher is better."""
    r = cards >> 2
    s = cards & 3
    cnt = (r[:, :, None] == _R).sum(axis=1)
    key = np.where(cnt > 0, cnt * 16 + _R, 0)
    sk = -np.sort(-key, axis=1)[:, :5]
    c = sk >> 4
    rk = np.where(sk > 0, (sk & 15) + 1, 0)
    enc = (((rk[:, 0] * _B + rk[:, 1]) * _B + rk[:, 2]) * _B + rk[:, 3]) * _B + rk[:, 4]
    c1, c2 = c[:, 0], c[:, 1]
    flush = (s == s[:, :1]).all(axis=1)
    distinct = c1 == 1
    straight = distinct & (rk[:, 0] - rk[:, 4] == 4)
    wheel = distinct & (rk[:, 0] == 13) & (rk[:, 1] == 4)
    enc = np.where(wheel, 4 * _B ** 4, enc)
    st = straight | wheel
    cat = np.select([st & flush, c1 == 4, (c1 == 3) & (c2 == 2), flush, st, c1 == 3, (c1 == 2) & (c2 == 2), c1 == 2],
                    [8, 7, 6, 5, 4, 3, 2, 1], default=0)
    return cat * _B ** 5 + enc


def eval_best(cards):
    n, k = cards.shape
    if k == 5:
        return eval5(cards)
    idx = COMBOS[k]
    return eval5(cards[:, idx].reshape(n * len(idx), 5)).reshape(n, len(idx)).max(axis=1)


# ----------------------------------------------------------------------------- discard heuristic
_WIN = np.zeros((10, 13), dtype=bool)
for _w in range(9):
    _WIN[_w, _w:_w + 5] = True
_WIN[9, [12, 0, 1, 2, 3]] = True          # A-2-3-4-5


def keep_score(keep, flop):
    """Heuristic value of keeping `keep` (M,2) given `flop` (M,3): made-hand strength plus draw bonus."""
    five = np.concatenate([keep, flop], axis=1)
    base = eval5(five) / float(_B ** 5)
    suits = five & 3
    fd = ((suits[:, :, None] == np.arange(4)).sum(axis=1).max(axis=1) == 4)
    present = (five >> 2)[:, :, None] == _R
    present = present.any(axis=1)                                       # (M,13)
    cnt = present.astype(np.int8) @ _WIN.T.astype(np.int8)              # (M,10)
    miss = _WIN[None, :, :] & ~present[:, None, :]
    outs = ((cnt == 4)[:, :, None] & miss).any(axis=1).sum(axis=1)
    made = (cnt == 5).any(axis=1)
    sd = np.where(made, 0.0, np.where(outs >= 2, 0.6, np.where(outs == 1, 0.25, 0.0)))
    return base + 0.8 * fd + sd


def keep_best(cards3, flop):
    """Heuristic discard. cards3 (M,3), flop (M,3) -> (disc_idx (M,), kept (M,2))."""
    K = np.stack([cards3[:, [1, 2]], cards3[:, [0, 2]], cards3[:, [0, 1]]], axis=1)    # (M,3,2): keep when discarding i
    sc = np.stack([keep_score(K[:, i], flop) for i in range(3)], axis=1)
    disc = sc.argmax(axis=1)
    return disc, K[np.arange(len(cards3)), disc]


def lowest_discard(cards3):
    """Index the rulebook default (and the starter bot) would discard: lowest rank, ties by suit order s,h,d,c."""
    return int(min(range(3), key=lambda i: ((cards3[i] >> 2), cards3[i] & 3)))


# ----------------------------------------------------------------------------- 3-card preflop table
def triple_key(a, b, c):
    cs = sorted([(a >> 2, a & 3), (b >> 2, b & 3), (c >> 2, c & 3)], key=lambda x: (-x[0], x[1]))
    r0, r1, r2 = cs[0][0], cs[1][0], cs[2][0]
    s0, s1, s2 = cs[0][1], cs[1][1], cs[2][1]
    if s0 == s1 == s2:
        pat = 'm'
    elif s0 == s1:
        pat = 'ab'
    elif s0 == s2:
        pat = 'ac'
    elif s1 == s2:
        pat = 'bc'
    else:
        pat = 'r'
    if r0 == r1 == r2:
        pat = 'r'
    elif r0 == r1 and pat == 'bc':
        pat = 'ac'
    elif r1 == r2 and pat == 'ab':
        pat = 'ac'
    return (r0, r1, r2, pat)


_T = None
_T_SORTED = None


def preflop_array():
    """(52,52,52) float32 equity-vs-random lookup, and the sorted distribution for percentiles."""
    global _T, _T_SORTED
    if _T is None:
        from dh_table import TABLE
        _T = np.zeros((52, 52, 52), dtype=np.float32)
        vals = []
        for a, b, c in itertools.combinations(range(52), 3):
            v = TABLE[triple_key(a, b, c)]
            vals.append(v)
            for p in itertools.permutations((a, b, c)):
                _T[p] = v
        _T_SORTED = np.sort(np.array(vals, dtype=np.float32))
    return _T


def preflop_value(hole3):
    T = preflop_array()
    return float(T[hole3[0], hole3[1], hole3[2]])


def preflop_pct(hole3):
    preflop_array()
    return float(np.searchsorted(_T_SORTED, preflop_value(hole3)) / len(_T_SORTED))


# ----------------------------------------------------------------------------- simulation
def _deck(used):
    m = np.ones(52, dtype=bool)
    m[list(used)] = False
    return np.nonzero(m)[0]


def _draw(deck, N, need, rng):
    perm = np.argsort(rng.random((N, len(deck))), axis=1)[:, :need]
    return deck[perm]


def _stack_board(board, N):
    return np.broadcast_to(np.asarray(board, dtype=np.int64), (N, len(board)))


def _narrow(mask, st, q):
    if q >= 0.999 or mask.sum() < 20:
        return mask
    thr = np.quantile(st[mask], 1 - q)
    m2 = mask & (st >= thr)
    return m2 if m2.sum() >= 20 else mask


def sim_flop3(hole3, flop, q=1.0, n=300, rng=None):
    """Flop, before the discard: equity of each of the 3 possible keeps vs a random opponent 3-card hand
    (who also discards by heuristic), optionally narrowed to the top-q by the opponent's best keep."""
    rng = rng or np.random.default_rng()
    hole3 = np.asarray(hole3, dtype=np.int64)
    flop = np.asarray(flop, dtype=np.int64)
    deck = _deck(set(hole3.tolist()) | set(flop.tolist()))
    N = int(min(n / max(q, 0.1), 6 * n)) if q < 0.999 else n
    d = _draw(deck, N, 5, rng)
    opp3, run = d[:, :3], d[:, 3:]
    fl = _stack_board(flop, N)
    _, okeep = keep_best(opp3, fl)
    full = np.concatenate([fl, run], axis=1)
    mask = np.ones(N, dtype=bool)
    if q < 0.999:
        mask = _narrow(mask, keep_score(okeep, fl), q)
    theirs = eval_best(np.concatenate([okeep, full], axis=1))
    eqs = []
    for i in range(3):
        keep = np.delete(hole3, i)
        mine = eval_best(np.concatenate([np.broadcast_to(keep, (N, 2)), full], axis=1))
        o = (mine > theirs) + 0.5 * (mine == theirs)
        eqs.append(float(o[mask].mean()))
    return eqs


def sim_post(keep2, board, opp_disc, q=1.0, n=300, rng=None, eps=0.15):
    """After the discard: my 2 cards vs the opponent's 2 unknown cards, conditioned on their public discard."""
    rng = rng or np.random.default_rng()
    keep2 = np.asarray(keep2, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    nb = len(board)
    deck = _deck(set(keep2.tolist()) | set(board.tolist()) | {int(opp_disc)})
    N = int(min(n * 3 / max(q, 0.15), 8 * n))
    d = _draw(deck, N, 2 + 5 - nb, rng)
    opp, run = d[:, :2], d[:, 2:]
    trip = np.concatenate([opp, np.full((N, 1), opp_disc, dtype=np.int64)], axis=1)
    dd, _ = keep_best(trip, _stack_board(board[:3], N))
    consistent = (dd == 2) | ((opp_disc >> 2) <= (opp >> 2).min(axis=1))      # heuristic keep OR lowest-card discard
    mask = consistent | (rng.random(N) < eps)
    bd = _stack_board(board, N)
    if q < 0.999:
        mask = _narrow(mask, eval_best(np.concatenate([opp, bd], axis=1)), q)
    full = np.concatenate([bd, run], axis=1)
    mine = eval_best(np.concatenate([np.broadcast_to(keep2, (N, 2)), full], axis=1))
    theirs = eval_best(np.concatenate([opp, full], axis=1))
    o = (mine > theirs) + 0.5 * (mine == theirs)
    return float(o[mask].mean())


def sim_pre(hole3, q=1.0, n=300, rng=None):
    """Preflop equity vs the top-q (by preflop table) of 3-card hands, both sides discarding by heuristic."""
    rng = rng or np.random.default_rng()
    hole3 = np.asarray(hole3, dtype=np.int64)
    deck = _deck(set(hole3.tolist()))
    N = int(min(n / max(q, 0.1), 6 * n)) if q < 0.999 else n
    d = _draw(deck, N, 8, rng)
    opp3, board = d[:, :3], d[:, 3:]
    mask = np.ones(N, dtype=bool)
    if q < 0.999:
        T = preflop_array()
        mask = _narrow(mask, T[opp3[:, 0], opp3[:, 1], opp3[:, 2]], q)
    flop = board[:, :3]
    _, okeep = keep_best(opp3, flop)
    _, hkeep = keep_best(np.broadcast_to(hole3, (N, 3)), flop)
    mine = eval_best(np.concatenate([hkeep, board], axis=1))
    theirs = eval_best(np.concatenate([okeep, board], axis=1))
    o = (mine > theirs) + 0.5 * (mine == theirs)
    return float(o[mask].mean())
