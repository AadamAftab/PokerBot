"""Paired duplicate-deal A/B evaluation against an opponent league.

Every arm plays the *same* deal sequence (same seed) against the same opponent, so the
per-deal difference removes most card luck. Blocks of `deals` deals form one "match"
(Swiss-style win/draw/loss). Runs blocks in parallel (one single-threaded BLAS per process).

usage: python paired_eval.py ARM_A ARM_B [blocks=10] [deals=200] [opponents,comma,sep]
"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys, time, json, ctypes, random
from multiprocessing import Pool
import numpy as np
import bots, league, league2
from champ import Champ
import engine

MAXSIMS = 300      # same for all arms; keeps the experiments fast, relative comparison is what matters


def peak_mb():
    try:
        class PMC(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(c)
        k32, ps = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        ps.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
        assert ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2 ** 20
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


ARMS = {
    'base':    lambda sd: Champ(seed=sd, replay=False, risk=False, max_sims=MAXSIMS),
    'risk':    lambda sd: Champ(seed=sd, replay=False, risk=True, max_sims=MAXSIMS),
    'replay':  lambda sd: Champ(seed=sd, replay=True, risk=False, max_sims=MAXSIMS),
}
try:                                    # patched arms register themselves here
    from patches import PATCH_ARMS
    ARMS.update({k: (lambda sd, f=f: f(sd, MAXSIMS)) for k, f in PATCH_ARMS.items()})
except ImportError:
    pass
try:
    from patches2 import ARM_CLASSES
    ARMS.update({k: (lambda sd, f=f: f(sd, MAXSIMS)) for k, f in ARM_CLASSES.items()})
except ImportError:
    pass

OPPS = {
    'Random': bots.RandomBot, 'CallStation': bots.CallStation, 'Maniac': bots.Maniac,
    'Rock': bots.Rock, 'BasicEq': bots.BasicEq,
    'Prober': league.Prober, 'Jammer': league.Jammer, 'Reverser': league.Reverser,
    'Switcher': league.Switcher, 'PreflopFolder': league.PreflopFolder,
    'Replayer': lambda: Champ(seed=99, replay=True, risk=False, max_sims=MAXSIMS),
}


OPPS.update(league2.OPPS2)
OPPS['ReplayerOff'] = lambda: Champ(seed=99, replay=False, risk=False, max_sims=MAXSIMS)   # mirror control: same policy, replay disabled


def run_block(task):
    arm, opp, blk, deals = task
    random.seed(blk)
    engine.STATS.update(actions=0, illegal=0, hero_illegal=0, log=[])
    hero = ARMS[arm](blk)
    r = engine.duplicate_match(lambda: hero, OPPS[opp], deals, seed=5000 + blk)
    lat = np.array(hero.lat) * 1000
    return dict(arm=arm, opp=opp, blk=blk, res=r, p50=float(np.percentile(lat, 50)),
                p99=float(np.percentile(lat, 99)), mx=float(lat.max()), per_hand=float(lat.sum() / (2 * deals)),
                errors=hero.errors, illegal=engine.STATS['hero_illegal'], actions=engine.STATS['actions'],
                mem=peak_mb(), fold_est=getattr(hero, 'fold_rate', None))


def ci(d):
    d = np.asarray(d, float)
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))


def main():
    a, b = sys.argv[1], sys.argv[2]
    blocks = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    deals = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    opps = sys.argv[5].split(',') if len(sys.argv) > 5 else list(OPPS)
    tasks = [(arm, o, k, deals) for o in opps for arm in {a, b} for k in range(blocks)]
    t = time.time()
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as p:
        out = p.map(run_block, tasks, chunksize=1)
    print(f"{len(tasks)} blocks x {deals} deals in {time.time()-t:.0f}s  (arms {a} vs {b}; MAXSIMS={MAXSIMS})\n")
    hdr = f"{'opponent':14}{a+' c/deal':>16}{b+' c/deal':>16}{'diff (paired)':>20}{'pts '+a:>9}{'pts '+b:>9}"
    print(hdr)
    dump = {}
    for o in opps:
        R = {arm: np.array([x['res'] for x in sorted((y for y in out if y['arm'] == arm and y['opp'] == o),
                                                       key=lambda z: z['blk'])], float) for arm in (a, b)}
        ma, ea = ci(R[a].ravel()); mb, eb = ci(R[b].ravel()); md, ed = ci((R[b] - R[a]).ravel())
        pts = {arm: np.mean([(1 if s > 0 else 0.5 if s == 0 else 0) for s in R[arm].sum(1)]) for arm in (a, b)}
        print(f"{o:14}{ma:9.1f}±{ea:<6.1f}{mb:9.1f}±{eb:<6.1f}{md:+10.1f}±{ed:<8.1f}{pts[a]:9.2f}{pts[b]:9.2f}")
        dump[o] = {arm: R[arm].tolist() for arm in R}
    for arm in {a, b}:
        rows = [x for x in out if x['arm'] == arm]
        print(f"\n[{arm}] ms/hand mean {np.mean([x['per_hand'] for x in rows]):.1f}  "
              f"p50 decision {np.mean([x['p50'] for x in rows]):.1f}ms  p99 {np.mean([x['p99'] for x in rows]):.1f}ms  "
              f"max {max(x['mx'] for x in rows):.0f}ms  peak RSS {max(x['mem'] for x in rows):.0f}MB  "
              f"caught errors {sum(x['errors'] for x in rows)}  hero-illegal {sum(x['illegal'] for x in rows)}/"
              f"{sum(x['actions'] for x in rows)} actions")
    json.dump(dump, open(f'results_{a}_vs_{b}.json', 'w'))
    fe = {o: np.mean([x['fold_est'] for x in out if x['arm'] == a and x['opp'] == o and x['fold_est'] is not None]) for o in opps}
    print('\nfinal fold-rate estimate of opponent (arm', a, '):', {k: round(v, 2) for k, v in fe.items()})


if __name__ == '__main__':
    main()
