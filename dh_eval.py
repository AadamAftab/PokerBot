"""Paired duplicate-deal evaluation in Discard Hold'em.

Match = 400 hands = 200 duplicate deal pairs (each deal played from both seats), as in Stage 1.
Primary metric J = P(win) + 0.5 P(draw) on net chips per match; secondary: chips/hand.
Arms are evaluated on the SAME deal seeds so card luck cancels.

usage: python dh_eval.py ARM[,ARM...] OPP[,OPP...] [matches=10] [seed0=0] [hands=400] [out.json]
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys, json, time
from multiprocessing import Pool
import numpy as np
from dh_engine import Match, InProc
import dh_bot, dh_league

FIXED_N = 200          # deterministic sim count so results do not depend on machine load


def make_arm(name, seed):
    if name == 'v1':      # frozen reference: no exploitation, no discard tracking, no tie-break
        return dh_bot.Bot(seed=seed, fixed_n=FIXED_N, params=dict(ex=0.0, eps=0.0, tie=0.0))
    if name.startswith('n:'):                          # compute-sensitivity arms: v2 with a different sim count
        return dh_bot.Bot(seed=seed, fixed_n=int(name[2:]))
    if name == 'v2':
        return dh_bot.Bot(seed=seed, fixed_n=FIXED_N, params=dict(ex=1.0, eps=1.0, tie=1.0))
    if name.startswith('v1:'):                         # v1:key=val,key=val parameter override
        params = {k: float(v) for k, v in (kv.split('=') for kv in name[3:].split(','))}
        return dh_bot.Bot(seed=seed, fixed_n=FIXED_N, params=params)
    raise KeyError(name)


def run_one(task):
    arm, opp, seed, hands = task
    hero = make_arm(arm, seed)
    villain = dh_league.LEAGUE[opp](seed=seed + 777) if opp != 'Mirror' else dh_bot.Bot(seed=seed + 5, fixed_n=FIXED_N)
    # experiments run with the clock disabled (fixed sim counts); the governor is validated separately by time_check.py
    m = Match([InProc(hero, 'hero'), InProc(villain, opp)], n_hands=hands, seed=seed, time_bank_ms=10**9, per_response_ms=10**9)
    t = time.perf_counter()
    r = m.run()
    r.update(arm=arm, opp=opp, seed=seed, wall=time.perf_counter() - t, errors=getattr(hero, 'errors', 0))
    r['lat_hero'] = [float(np.percentile(r['stats']['lat'][0], q)) for q in (50, 99, 100)]
    del r['stats']['lat']
    return r


def J(net):
    return 1.0 if net > 0 else (0.5 if net == 0 else 0.0)


def J_match(r):
    """Match points for player 0; a forfeit is a loss for the forfeiter regardless of chips."""
    if r['forfeit']:
        return 0.0 if r['forfeit'][0] == 0 else 1.0
    return J(r['total'][0])


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    arms = sys.argv[1].split(',')
    opps = sys.argv[2].split(',')
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    s0 = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    hands = int(sys.argv[5]) if len(sys.argv) > 5 else 400
    out = sys.argv[6] if len(sys.argv) > 6 else None
    tasks = [(a, o, s0 + k, hands) for o in opps for a in arms for k in range(n)]
    t = time.time()
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as p:
        res = p.map(run_one, tasks, chunksize=1)
    print(f"{len(tasks)} matches x {hands} hands in {time.time()-t:.0f}s  (seeds {s0}..{s0+n-1}, FIXED_N={FIXED_N})\n")
    print(f"{'opponent':16}{'arm':14}{'J':>6}{'  J 95% (Wilson)':>18}{'chips/hand':>12}{'  ±':>7}{'forfeit':>8}{'illegal':>8}{'err':>5}")
    for o in opps:
        for a in arms:
            rr = [r for r in res if r['opp'] == o and r['arm'] == a]
            nets = np.array([r['total'][0] for r in rr], float)
            j = np.array([J_match(r) for r in rr])
            lo, hi = wilson(int((j == 1).sum() + 0.5 * (j == 0.5).sum()), len(j))
            cph = nets / hands
            se = cph.std(ddof=1) / np.sqrt(len(cph)) if len(cph) > 1 else 0
            print(f"{o:16}{a:14}{j.mean():6.2f}   [{lo:.2f},{hi:.2f}]{cph.mean():12.2f}{1.96*se:7.2f}"
                  f"{sum(1 for r in rr if r['forfeit']):8d}{sum(r['stats']['illegal'][0] for r in rr):8d}{sum(r['errors'] for r in rr):5d}")
    if out:
        json.dump(res, open(out, 'w'), default=str)


if __name__ == '__main__':
    main()
