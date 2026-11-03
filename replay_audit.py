"""Phase 1 audit: why does spotq lose to the replay-aware opponent?  Decomposes Hero's chips by what the opponent knew.
usage: python replay_audit.py [matches=16] [deals=100]"""
import os
for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
import sys, json, random
from multiprocessing import Pool
import numpy as np
import paired_eval as pe
import engine
from probe import PROBES, ReplayerProbe


def run(task):
    opp_kind, blk, deals = task
    random.seed(blk)
    hero = PROBES['spotq'](blk, pe.MAXSIMS)
    holder = []
    def mk():
        o = ReplayerProbe(seed=99, max_sims=pe.MAXSIMS, replay=(opp_kind == 'on')); holder.append(o); return o
    r = engine.duplicate_match(lambda: hero, mk, deals, seed=7000 + blk)
    return dict(kind=opp_kind, m=blk, res=r, hands=hero.hands, olog={str(k): v for k, v in holder[0].hlog.items()})


def boot(vals_by_match, B=3000):
    rng = np.random.default_rng(0); a = np.array(vals_by_match, float)
    est = a[:, 0].sum() / max(a[:, 1].sum(), 1)
    idx = rng.integers(0, len(a), (B, len(a)))
    bs = a[idx, 0].sum(1) / np.maximum(a[idx, 1].sum(1), 1)
    return est, *np.percentile(bs, [2.5, 97.5])


def category(h, o):
    odd = h % 2 == 1
    if o is None: return 'replay half, replay NOT detected' if odd else 'first half, no replay belief'
    if not odd: return 'first half, FALSE replay belief'
    if o['full']: return f"replay half, detected, outcome known ({o['outcome']})"
    return 'replay half, detected, partial knowledge'


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    deals = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    tasks = [(k, 3000 + i, deals) for k in ('on', 'off') for i in range(n)]
    with Pool(max(1, (os.cpu_count() or 4) - 2)) as p:
        out = p.map(run, tasks, chunksize=1)
    json.dump(out, open('results/replay_audit.json', 'w'))
    for kind in ('off', 'on'):
        ms = [o for o in out if o['kind'] == kind]
        tot = np.array([np.mean(m['res']) for m in ms])
        print(f"\n===== spotq vs replay {'ENABLED' if kind=='on' else 'DISABLED (mirror control)'}: {len(ms)} matches x {deals} deals; "
              f"chips/deal {tot.mean():+.2f}; match wins {sum(sum(m['res'])>0 for m in ms)}/{len(ms)}")
        cats = {}
        for m in ms:
            per = {}
            for h in m['hands']:
                c = category(h['h'], m['olog'].get(str(h['h'])))
                a = per.setdefault(c, [0.0, 0]); a[0] += h['chips']; a[1] += 1
            for c, v in per.items(): cats.setdefault(c, []).append((v[0], v[1]))
        nh = sum(v[1] for l in cats.values() for v in l)
        print(f"{'situation (what the opponent believed)':52}{'% hands':>8}{'Hero chips/hand [95% CI]':>30}{'contribution/hand':>19}")
        for c, l in sorted(cats.items(), key=lambda kv: sum(v[0] for v in kv[1])):
            est, lo, hi = boot(l); share = sum(v[1] for v in l) / nh
            print(f"{c:52}{100*share:7.1f}%{est:11.2f} [{lo:+7.2f},{hi:+7.2f}]{share*est:19.2f}")
        paths = {}
        for m in ms:
            for o in m['olog'].values(): paths[(o['path'], o['lag'])] = paths.get((o['path'], o['lag']), 0) + 1
        if paths: print("  detection paths (path, lag in hands): ", dict(sorted(paths.items(), key=lambda kv: -kv[1])[:6]))
