"""Match-level evaluation: P(win)/P(draw)/Swiss points per matchup, many full matches in parallel.

usage: python winprob.py [matches_per_pair=8] [deals_per_match=200]
"""
import sys, time
from multiprocessing import Pool
import numpy as np
import bots
from champ import Champ
from engine import duplicate_match


def factory(name, seed):
    return {
        'Hero':         lambda: bots.Hero(seed=seed),
        'Champ-base':   lambda: Champ(seed=seed, replay=False, risk=False),
        'Champ-risk':   lambda: Champ(seed=seed, replay=False, risk=True),
        'Champ-replay': lambda: Champ(seed=seed, replay=True, risk=False),
        'Champ-full':   lambda: Champ(seed=seed, replay=True, risk=True),
        'BasicEq':      bots.BasicEq, 'Rock': bots.Rock, 'CallStation': bots.CallStation,
        'Maniac': bots.Maniac,
    }[name]


def run(task):
    a, b, seed, deals = task
    r = duplicate_match(factory(a, seed), factory(b, seed + 777), deals, seed=seed)
    return a, b, float(sum(r))


if __name__ == '__main__':
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    deals = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    A = ['Hero', 'Champ-base', 'Champ-risk', 'Champ-replay', 'Champ-full']
    B = ['BasicEq', 'Rock', 'CallStation', 'Champ-replay']
    tasks = [(a, b, 1000 + k, deals) for a in A for b in B for k in range(m)]
    t = time.time()
    with Pool() as pool:
        res = pool.map(run, tasks, chunksize=1)
    print(f"{len(tasks)} matches of {deals} deals in {time.time()-t:.0f}s\n")
    print(f"{'bot':13} {'vs':13} {'win':>5} {'draw':>5} {'points':>7} {'avg chips':>10}")
    for a in A:
        for b in B:
            c = np.array([x[2] for x in res if x[0] == a and x[1] == b])
            w, d = (c > 0).mean(), (c == 0).mean()
            print(f"{a:13} {b:13} {w:5.2f} {d:5.2f} {w + 0.5 * d:7.2f} {c.mean():10.0f}")
