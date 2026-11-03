"""Collect instrumented decision logs.  usage: python diag_run.py ARM OPP1,OPP2,.. [matches=12] [deals=100] [tag=diag]"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys, json, time, random
from multiprocessing import Pool
import paired_eval as pe
import engine
from probe import PROBES


def run(task):
    arm, opp, blk, deals = task
    random.seed(blk)
    hero = PROBES[arm](blk, pe.MAXSIMS)
    r = engine.duplicate_match(lambda: hero, pe.OPPS[opp], deals, seed=5000 + blk)
    return dict(arm=arm, opp=opp, m=blk, res=r, dec=hero.dec, hands=hero.hands)


if __name__ == '__main__':
    arm, opps = sys.argv[1], sys.argv[2].split(',')
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    deals = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    tag = sys.argv[5] if len(sys.argv) > 5 else 'diag'
    tasks = [(arm, o, 2000 + k, deals) for o in opps for k in range(n)]      # diagnostic seeds (distinct from dev/heldout)
    t = time.time()
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as p:
        out = p.map(run, tasks, chunksize=1)
    json.dump(out, open(f'results/{tag}_{arm}.json', 'w'))
    print(f'saved results/{tag}_{arm}.json  {len(tasks)} matches, {sum(len(o["dec"]) for o in out)} decisions, {time.time()-t:.0f}s')
