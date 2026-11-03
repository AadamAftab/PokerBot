"""Adversarial evaluation driver (raw results are preserved in results/raw_<phase>_<tag>.json).

PRE-REGISTERED PROTOCOL
  primary endpoint : J = expected match points per 200-deal duplicate match (win 1, draw .5, loss 0)
  unit of analysis : one match (fresh bot instances, its own deal seed); deals inside a match are NOT independent
  secondary        : chips per duplicate deal (cluster bootstrap over matches)
  seed sets        : dev = match ids 0..9 (exploration, may be inspected freely)
                     heldout = match ids 1000..1029 (confirmation only; never used to design opponents/patches)
usage: python adv_eval.py PHASE ARMS[,..] OPPONENTS|all [tag]
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys, json, time, hashlib
from multiprocessing import Pool
import paired_eval as pe

PHASES = {'dev': range(0, 10), 'heldout': range(1000, 1030), 'dev2': range(100, 110), 'heldout2': range(3000, 3030)}
FILES = ['pokerlib.py', 'engine.py', 'bots.py', 'champ.py', 'patches.py', 'patches2.py', 'league.py', 'league2.py', 'paired_eval.py']


def source_hash():
    h = hashlib.sha256()
    for f in FILES:
        h.update(open(f, 'rb').read())
    return h.hexdigest()[:16]


def main():
    phase, arms = sys.argv[1], sys.argv[2].split(',')
    opps = list(pe.OPPS) if sys.argv[3] == 'all' else sys.argv[3].split(',')
    tag = sys.argv[4] if len(sys.argv) > 4 else 'run'
    deals = 200
    tasks = [(a, o, k, deals) for o in opps for a in arms for k in PHASES[phase]]
    t = time.time()
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as p:
        rows = p.map(pe.run_block, tasks, chunksize=1)
    cfg = dict(phase=phase, arms=arms, opps=opps, deals=deals, matches=list(PHASES[phase]),
               maxsims=pe.MAXSIMS, source_hash=source_hash(), secs=round(time.time() - t))
    path = f'results/raw_{phase}_{tag}.json'
    json.dump(dict(config=cfg, rows=rows), open(path, 'w'))
    print(f"saved {path}  ({len(tasks)} matches, {cfg['secs']}s, source {cfg['source_hash']})")


if __name__ == '__main__':
    main()
