"""CFR feasibility analysis for heads-up no-limit hold'em under this event's constraints.

Part 1: information-set count / storage / compute for abstraction sizes (analytic; ASSUMPTIONS marked).
Part 2: MEASURED throughput of external-sampling MCCFR in pure Python on Leduc hold'em, used to
        convert node-visit requirements into wall-clock.
Run: python cfr_feasibility.py
"""
import random, time, itertools, sys
from functools import lru_cache


# ---------------------------------------------------------------- Part 1: state-space arithmetic
def street_tree(k_sizes, max_raises):
    """Betting tree within one street (heads-up, first actor opens). Actions: fold/check-call, k raises.
    Returns (decision_nodes, continue_leaves) where continue_leaves = lines that end the street with both in."""
    @lru_cache(None)
    def rec(raises, facing_bet, acted):
        # returns (nodes, continues)
        nodes, cont = 1, 0
        # check/call
        if (facing_bet) or acted:           # call, or check-back after a check -> street ends
            cont += 1
        else:                               # first check: opponent acts
            n, c = rec(raises, False, True); nodes += n; cont += c
        # fold only when facing a bet (leaf, no extra nodes)
        # raises
        if raises < max_raises:
            for _ in range(k_sizes):
                n, c = rec(raises + 1, True, True); nodes += n; cont += c
        return nodes, cont
    return rec(0, False, False)


def abstraction_table():
    rows = []
    cfgs = [("tiny",      dict(pre=169, B=(20, 20, 20),    k=2, r=2)),
            ("small",     dict(pre=169, B=(50, 50, 50),    k=3, r=3)),
            ("medium",    dict(pre=169, B=(200, 200, 200), k=5, r=4)),
            ("Pluribus-like", dict(pre=169, B=(1000, 1000, 1000), k=8, r=6))]
    V = 2000          # ASSUMPTION: average regret updates needed per info set for a usable blueprint
    for name, c in cfgs:
        nodes, cont = street_tree(c['k'], c['r'])
        nodes_pre, cont_pre = nodes, cont       # preflop uses same shape for simplicity
        hist = 1
        infosets = 0
        for s, buckets in enumerate((c['pre'],) + tuple(c['B'])):
            infosets += buckets * hist * nodes
            hist *= cont
        avg_actions = c['k'] + 2
        mb = infosets * avg_actions * 2 / 2 ** 20            # float16 average-strategy table
        rows.append((name, infosets, mb, infosets * V))
    return rows


# ---------------------------------------------------------------- Part 2: Leduc external-sampling MCCFR (measured)
CARDS = [0, 0, 1, 1, 2, 2]
nodes_visited = 0


class Node:
    __slots__ = ('regret', 'strat_sum', 'n')
    def __init__(self, n): self.regret = [0.0] * n; self.strat_sum = [0.0] * n; self.n = n
    def strategy(self, acts):
        pos = [max(r, 0.0) if i in acts else 0.0 for i, r in enumerate(self.regret)]; t = sum(pos)
        return [p / t for p in pos] if t > 0 else [1.0 / len(acts) if i in acts else 0.0 for i in range(self.n)]


table = {}


def legal(hist_round, raises):
    # 0 fold, 1 call/check, 2 raise
    a = [1]
    if hist_round and hist_round[-1] == 2: a = [0, 1] + ([2] if raises < 2 else [])
    else: a = [1] + ([2] if raises < 2 else [])
    return a


def traverse(cards, board, rnd, hist, raises, pot_in, player, traverser):
    """returns utility for traverser. pot_in: chips each player has put in [p0, p1]."""
    global nodes_visited
    nodes_visited += 1
    rh = hist[rnd]
    acts = legal(rh, raises)
    key = (player, cards[player], board if rnd else -1, tuple(tuple(h) for h in hist[:rnd + 1]))
    node = table.get(key)
    if node is None: node = table[key] = Node(3)
    strat = node.strategy(acts)
    def apply(a):
        h2 = [list(x) for x in hist]; h2[rnd].append(a)
        pin = list(pot_in)
        if a == 0:
            return ('fold', player), None
        r2 = raises
        if a == 2:
            pin[player] = pin[1 - player] + (2 if rnd == 0 else 4); r2 += 1
        elif a == 1:
            pin[player] = pin[1 - player]
        # round end?
        last = h2[rnd]
        end = a == 1 and len(last) >= 2          # call after a bet, or check-check
        return (h2, pin, r2, end), None

    def value(a):
        res, _ = apply(a)
        if res[0] == 'fold':
            return -pot_in[player] if player == traverser else pot_in[player]
        h2, pin, r2, end = res
        if end:
            if rnd == 1:                                           # showdown
                def rank(p): return (2 if cards[p] == board else 1, cards[p])
                r0, r1 = rank(0), rank(1)
                u0 = pin[1] if r0 > r1 else (-pin[0] if r0 < r1 else 0)
                return u0 if traverser == 0 else -u0
            return traverse(cards, board, 1, h2 + [[]], 0, pin, 0, traverser)
        return traverse(cards, board, rnd, h2, r2, pin, 1 - player, traverser)

    if player == traverser:
        vals = {a: value(a) for a in acts}
        ev = sum(strat[a] * vals[a] for a in acts)
        for a in acts: node.regret[a] += vals[a] - ev
        return ev
    r = random.random(); acc = 0.0
    for a in acts:
        acc += strat[a]
        if r <= acc: break
    for b in acts: node.strat_sum[b] += strat[b]
    return value(a)


def benchmark(seconds=8.0):
    global nodes_visited
    nodes_visited = 0; t = time.perf_counter(); it = 0
    while time.perf_counter() - t < seconds:
        deck = CARDS[:]; random.shuffle(deck)
        for trv in (0, 1):
            traverse(deck[:2], deck[2], 0, [[]], 0, [1, 1], 0, trv)
        it += 1
    dt = time.perf_counter() - t
    return it / dt, nodes_visited / dt, len(table)


if __name__ == '__main__':
    print("PART 1 - abstracted HU NLHE size (analytic; imperfect-recall buckets; V=2000 updates/infoset ASSUMED)")
    print(f"{'abstraction':15}{'info sets':>16}{'strategy MB (fp16)':>20}{'node visits needed':>22}")
    for name, n, mb, visits in abstraction_table():
        print(f"{name:15}{n:16,.0f}{mb:20,.1f}{visits:22,.2e}")
    its, nps, size = benchmark()
    print(f"\nPART 2 - MEASURED pure-Python external-sampling MCCFR on Leduc: {its:,.0f} iterations/s, "
          f"{nps:,.0f} node visits/s, {size} info sets touched")
    print("\nWall-clock to reach the node-visit requirement at the measured pure-Python rate (1 core / 18 cores):")
    for name, n, mb, visits in abstraction_table():
        s = visits / nps
        print(f"{name:15}{s/3600:12,.1f} core-hours   {s/3600/18:12,.1f} h on 18 cores   "
              f"(~{s/3600/18/24:,.1f} days)")
