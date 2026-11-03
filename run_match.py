"""Usage: python run_match.py [n_deals]  - Hero vs every zoo bot, duplicate format."""
import sys, time
import numpy as np
from engine import duplicate_match, BB
import bots

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
zoo = [bots.RandomBot, bots.CallStation, bots.Maniac, bots.Rock, bots.BasicEq]
print(f"Hero vs zoo, {n} duplicate deals each (= {2*n} hands)\n")
print(f"{'opponent':12} {'bb/hand':>8} {'+-se':>6} {'deals won':>10}  secs")
for opp in zoo:
    t = time.time()
    r = np.array(duplicate_match(lambda: bots.Hero(seed=1), opp, n, seed=42), dtype=float)
    bbph = r.mean() / (2 * BB)
    se = r.std(ddof=1) / np.sqrt(len(r)) / (2 * BB)
    print(f"{opp.__name__:12} {bbph:8.2f} {se:6.2f} {(r > 0).sum():5d}/{(r != 0).sum():<4d} {time.time()-t:5.1f}")
