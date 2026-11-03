"""Hand evaluation and Monte Carlo equity (standard 52-card Texas Hold'em).

Cards are ints 0..51: rank = c >> 2 (0=2 ... 12=A), suit = c & 3.
Everything is vectorised with NumPy so one equity estimate costs a few ms.
"""
import itertools
import numpy as np

COMBOS = {k: np.array(list(itertools.combinations(range(k), 5)), dtype=np.int64) for k in (5, 6, 7)}
_RANKS = np.arange(13)
_B = 14  # base for encoding ranks (+1 so that 0 means "no card")


def eval5(cards):
    """cards: (M,5) int array -> (M,) int64 score, higher is better."""
    r = cards >> 2
    s = cards & 3
    cnt = (r[:, :, None] == _RANKS).sum(axis=1)                  # (M,13)
    key = np.where(cnt > 0, cnt * 16 + _RANKS, 0)
    sk = -np.sort(-key, axis=1)[:, :5]                            # best (count, rank) first
    c = sk >> 4
    rk = np.where(sk > 0, (sk & 15) + 1, 0)                       # ranks 1..13
    enc = (((rk[:, 0] * _B + rk[:, 1]) * _B + rk[:, 2]) * _B + rk[:, 3]) * _B + rk[:, 4]
    c1, c2 = c[:, 0], c[:, 1]
    flush = (s == s[:, :1]).all(axis=1)
    distinct = c1 == 1
    straight = distinct & (rk[:, 0] - rk[:, 4] == 4)
    wheel = distinct & (rk[:, 0] == 13) & (rk[:, 1] == 4)         # A-5-4-3-2
    enc = np.where(wheel, 4 * _B ** 4, enc)
    st = straight | wheel
    cat = np.select(
        [st & flush, c1 == 4, (c1 == 3) & (c2 == 2), flush, st,
         c1 == 3, (c1 == 2) & (c2 == 2), c1 == 2],
        [8, 7, 6, 5, 4, 3, 2, 1], default=0)
    return cat * _B ** 5 + enc


def eval_best(cards):
    """cards: (N,k) with k in 5..7 -> (N,) best-5-card score."""
    n, k = cards.shape
    if k == 5:
        return eval5(cards)
    idx = COMBOS[k]
    sel = cards[:, idx].reshape(n * len(idx), 5)
    return eval5(sel).reshape(n, len(idx)).max(axis=1)


_STR = None  # preflop strength lookup (52x52), built lazily


def _class_key(a, b):
    ra, rb = a >> 2, b >> 2
    return (max(ra, rb), min(ra, rb), (a & 3) == (b & 3) and ra != rb)


def preflop_strength():
    global _STR
    if _STR is None:
        try:
            from preflop_table import TABLE
        except ImportError:
            TABLE = None
        _STR = np.zeros((52, 52))
        if TABLE is not None:
            for a in range(52):
                for b in range(52):
                    if a != b:
                        _STR[a, b] = TABLE[_class_key(a, b)]
    return _STR


def equity(hole, board, q=1.0, n=600, rng=None):
    """P(win)+0.5*P(tie) for `hole` vs one opponent.

    q < 1 restricts the opponent to the top-q fraction of hands by current
    strength (preflop table or made hand on board) - a crude range model for
    "they bet, so they are probably strong".
    """
    rng = rng or np.random.default_rng()
    hole = np.asarray(hole, dtype=np.int64)
    board = np.asarray(board, dtype=np.int64)
    used = set(hole.tolist()) | set(board.tolist())
    deck = np.array([c for c in range(52) if c not in used], dtype=np.int64)
    nb = len(board)
    need = 2 + 5 - nb
    N = n if q >= 0.999 else int(n / max(q, 0.12))
    perm = np.argsort(rng.random((N, len(deck))), axis=1)[:, :need]
    draw = deck[perm]
    opp, run = draw[:, :2], draw[:, 2:]
    if q < 0.999:
        if nb == 0:
            st = preflop_strength()[opp[:, 0], opp[:, 1]]
        else:
            st = eval_best(np.concatenate([opp, np.broadcast_to(board, (N, nb))], axis=1))
        keep = st >= np.quantile(st, 1 - q)
        opp, run = opp[keep], run[keep]
    m = len(opp)
    full = np.concatenate([np.broadcast_to(board, (m, nb)), run], axis=1)
    mine = eval_best(np.concatenate([np.broadcast_to(hole, (m, 2)), full], axis=1))
    theirs = eval_best(np.concatenate([opp, full], axis=1))
    return float(((mine > theirs).sum() + 0.5 * (mine == theirs).sum()) / m)
