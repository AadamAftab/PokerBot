"""Hand-by-hand parity between my engine (dh_engine.py) and the organisers' engine (Starter_Kit/engine.py).
Both play the same deals (official deal generation is replicated) with the same deterministic, partly-illegal bots.
usage: python parity_test.py PATH_TO_Starter_Kit [seeds=3] [hands=400]"""
import os, random, re, subprocess, sys, tempfile, shutil
import numpy as np
from dhlib import parse_card
from dh_engine import Match, InProc
import parity_core

kit = os.path.abspath(sys.argv[1])
nseeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
hands = int(sys.argv[3]) if len(sys.argv) > 3 else 400
RANKS, SUITS = "23456789TJQKA", "shdc"


class PBot:
    def __init__(self, salt): self.salt = salt
    def handle(self, m): return parity_core.decide(self.salt, m)


class OfficialDeals(Match):
    """My engine, but dealing exactly what the official engine deals for a given seed."""
    def __init__(self, *a, seed_str='0', **kw):
        super().__init__(*a, **kw)
        self.seed_str, self.k = seed_str, 0

    def _deal(self):
        deck = [r + s for r in RANKS for s in SUITS]
        random.Random(f"{self.seed_str}:{self.k}").shuffle(deck)
        self.k += 1
        c = [parse_card(x) for x in deck]
        return c[0:3], c[3:6], c[6:11]


work = tempfile.mkdtemp(prefix='parity_', dir=os.path.dirname(kit))
for salt in 'AB':
    open(os.path.join(work, f'bot_{salt}.py'), 'w').write(f"import sys; sys.path.insert(0, r'{os.path.dirname(os.path.abspath(__file__))}')\nimport parity_core; parity_core.run('{salt}')\n")
bad_total = 0
for seed in range(nseeds):
    out = subprocess.run([sys.executable, os.path.join(kit, 'engine.py'), os.path.join(work, 'bot_A.py'), os.path.join(work, 'bot_B.py'),
                          '--hands', str(hands), '--seed', str(seed), '--log-dir', os.path.join(work, f'log{seed}')],
                         capture_output=True, text=True, cwd=kit).stdout
    off = {}
    for line in open(os.path.join(work, f'log{seed}', 'match.log'), encoding='utf-8'):
        m = re.match(r'#(\d+)\s+button (\w) \|.* (\w) ([+-]\d+), (\w) ([+-]\d+)$', line.strip())
        if m:
            nets = {m.group(3): int(m.group(4)), m.group(5): int(m.group(6))}
            off[int(m.group(1))] = (nets['A'], nets['B'])
    notes = re.findall(r'notices (\d+)', out)
    mine = OfficialDeals([InProc(PBot('A')), InProc(PBot('B'))], n_hands=hands, seed=0, seed_str=str(seed),
                         time_bank_ms=10**9, per_response_ms=10**9)
    got = {}
    orig = mine.hand
    def wrapped(hid, btn, hole, board, orig=orig, got=got):
        net = orig(hid, btn, hole, board)
        got[hid] = (net[0], net[1])          # player 0 = A, player 1 = B
        return net
    mine.hand = wrapped
    res = mine.run()
    diffs = [h for h in range(1, hands + 1) if off.get(h) != got.get(h)]
    my_notices = [sum(res['stats']['illegal'][p] + res['stats']['malformed'][p] for _ in [0]) for p in (0, 1)]
    print(f"seed {seed}: official hands {len(off)}, mine {len(got)}, mismatching hands {len(diffs)} {diffs[:8]}  "
          f"totals official A={sum(v[0] for v in off.values()):+d} mine A={res['total'][0]:+d}  notices official {notes} mine {my_notices}")
    bad_total += len(diffs) + (len(off) != hands) + (len(got) != hands)
shutil.rmtree(work, ignore_errors=True)
print("PARITY OK" if bad_total == 0 else f"PARITY FAILED ({bad_total} problems)")
