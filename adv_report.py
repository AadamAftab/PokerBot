"""Rank (arm, opponent) cells by expected match points J.  usage: python adv_report.py raw.json [raw2.json ...]"""
import sys, json
import numpy as np


def wilson(p, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def cluster_ci(match_means, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.asarray(match_means, float)
    bs = rng.choice(m, size=(B, len(m))).mean(1)
    return m.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def cells(rows):
    out = {}
    for r in rows:
        out.setdefault((r['arm'], r['opp']), []).append(np.array(r['res'], float))
    res = []
    for (arm, opp), ms in out.items():
        tot = np.array([m.sum() for m in ms])
        w, d = (tot > 0).sum(), (tot == 0).sum()
        n = len(ms); J = (w + 0.5 * d) / n
        lo, hi = wilson(J, n)
        mean, clo, chi = cluster_ci([m.mean() for m in ms])
        res.append(dict(arm=arm, opp=opp, n=n, W=int(w), D=int(d), L=int(n - w - d), J=J, jlo=lo, jhi=hi,
                        chips=mean, clo=clo, chi=chi))
    return res


def main():
    rows = []
    cfgs = []
    for f in sys.argv[1:]:
        d = json.load(open(f)); rows += d['rows']; cfgs.append(d['config'])
    for c in cfgs:
        print(f"[{c['phase']}] source {c['source_hash']} arms {c['arms']} matches/cell {len(c['matches'])} x {c['deals']} deals")
    cs = sorted(cells(rows), key=lambda c: (c['arm'], c['J'], c['chips']))
    print(f"\n{'arm':8}{'opponent':18}{'n':>3}{'W-D-L':>9}{'J':>6}{'J 95% (Wilson)':>17}{'chips/deal [95% cluster-boot]':>34}")
    for c in cs:
        print(f"{c['arm']:8}{c['opp']:18}{c['n']:3d}{c['W']:4d}-{c['D']}-{c['L']:<3d}{c['J']:6.2f}   [{c['jlo']:.2f},{c['jhi']:.2f}]"
              f"{c['chips']:12.1f}  [{c['clo']:7.1f},{c['chi']:7.1f}]")


if __name__ == '__main__':
    main()
