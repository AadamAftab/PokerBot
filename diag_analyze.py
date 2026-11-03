"""Analyses of instrumented logs.   usage: python diag_analyze.py {prober|size|calib} ARM [opp1,opp2,...]"""
import sys, json
import numpy as np


def load(arm, tag='diag'):
    return json.load(open(f'results/{tag}_{arm}.json'))


def boot_mean(per_match, B=3000, seed=0):
    """per_match: list of (sum, count). Cluster bootstrap CI of sum/count."""
    rng = np.random.default_rng(seed)
    a = np.array(per_match, float)
    est = a[:, 0].sum() / max(a[:, 1].sum(), 1)
    idx = rng.integers(0, len(a), size=(B, len(a)))
    bs = a[idx, 0].sum(1) / np.maximum(a[idx, 1].sum(1), 1)
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float('nan')
    gt = (pos[:, None] > neg[None, :]).mean(); eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def classify(hand_rows, decs):
    fold_post = any(d['act'] == 'f' and d['st'] >= 1 for d in decs)
    fold_pre = any(d['act'] == 'f' and d['st'] == 0 for d in decs)
    allin = any(d['tc'] >= d['stk'] > 0 and d['act'] in ('call', 'raise') for d in decs)
    if fold_post: return 'folded postflop to a bet'
    if fold_pre: return 'folded preflop'
    if allin: return 'faced an all-in and continued'
    if hand_rows['sd']: return 'showdown (no all-in)'
    return 'opponent folded / no showdown'


def prober(arm, opps):
    D = load(arm)
    for opp in opps:
        ms = [m for m in D if m['opp'] == opp]
        if not ms: continue
        rows = [r for m in ms for r in m['dec']]
        print(f"\n=== {arm} vs {opp}: {len(ms)} matches, {sum(len(m['hands']) for m in ms)} hands, "
              f"{np.mean([sum(m['res'])/len(m['res']) for m in ms]):+.1f} chips/deal ===")
        lead = [r for r in rows if r['st'] >= 1 and r['tc'] > 0 and r['mb'] == 0]
        print("facing a postflop lead bet:      n    fold%   call%  raise%   MDF-implied fold%   (equity when folding / calling)")
        for lo, hi in [(0, .4), (.4, .65), (.65, .9), (.9, 99)]:
            sub = [r for r in lead if lo <= r['ff'] < hi]
            if len(sub) < 5: continue
            f = np.mean([r['act'] == 'f' for r in sub]); c = np.mean([r['act'] == 'call' for r in sub]); rr = np.mean([r['act'] == 'raise' for r in sub])
            mdf = np.mean([1 - 1 / (1 + r['ff']) for r in sub])
            ef = np.mean([r['et'] for r in sub if r['act'] == 'f'] or [np.nan]); ec = np.mean([r['et'] for r in sub if r['act'] == 'call'] or [np.nan])
            print(f"  bet {lo:.2f}-{min(hi,9):.2f} pot   {len(sub):6d}  {100*f:5.1f}  {100*c:6.1f}  {100*rr:6.1f}   {100*mdf:12.1f}   (true equity folded {ef:.2f} / called {ec:.2f})")
        # equity actually folded away vs opponent's true cards
        fl = [r for r in lead if r['act'] == 'f']
        if fl:
            need = np.mean([r['tc'] / (r['pot'] + r['tc']) for r in fl])
            profitable = np.mean([r['et'] > r['tc'] / (r['pot'] + r['tc']) for r in fl])
            print(f"  of {len(fl)} folds to lead bets, {100*profitable:.0f}% had TRUE equity above the pot-odds price (avg price {need:.2f}, avg true equity {np.mean([r['et'] for r in fl]):.2f})")
        post0 = [r for r in rows if r['st'] >= 1 and r['tc'] == 0]
        print(f"  Hero postflop decisions with no bet to face: {len(post0)}, check {100*np.mean([r['act']=='check' for r in post0]):.0f}% / bet {100*np.mean([r['act']=='bet' for r in post0]):.0f}%")
        # chip attribution by situation
        agg = {}
        for m in ms:
            by_hand = {}
            for r in m['dec']: by_hand.setdefault(r['h'], []).append(r)
            for h in m['hands']:
                k = classify(h, by_hand.get(h['h'], []))
                a = agg.setdefault(k, [0, 0.0]); a[0] += 1; a[1] += h['chips']
        n = sum(v[0] for v in agg.values()); tot = sum(v[1] for v in agg.values())
        print(f"  chip attribution by situation (chips/hand overall {tot/n:+.2f}):")
        for k, (c, s) in sorted(agg.items(), key=lambda kv: kv[1][1]):
            print(f"    {k:34s} {100*c/n:5.1f}% of hands   {s/c:+7.2f} chips/hand   contributes {s/n:+6.2f}")


def size(arm, opps):
    D = load(arm)
    ms = [m for m in D if (not opps or m['opp'] in opps)]
    rows = [(m['m'], r) for m in ms for r in m['dec'] if r['hs'] is not None]
    agg = [(i, r) for i, r in rows if r['act'] in ('bet', 'raise')]
    print(f"{arm}: {len(ms)} matches, {len(agg)} bets/raises with strength proxy HS (equity vs random hand)")
    print("\nsize bucket (pot fraction)      n    mean HS   P(HS>0.62 'strong')   P(HS<0.45 'weak/bluff')")
    for name, lo, hi, kind in [('small bet <0.45', 0, .45, 'bet'), ('mid bet 0.45-0.72', .45, .72, 'bet'), ('big bet >=0.72', .72, 9, 'bet'),
                               ('raise (any)', 0, 9, 'raise')]:
        sub = [r for _, r in agg if r['act'] == kind and lo <= r['sz'] < hi]
        if sub:
            hs = np.array([r['hs'] for r in sub])
            print(f"  {name:26s} {len(sub):6d}   {hs.mean():.2f}      {100*(hs>0.62).mean():5.1f}%               {100*(hs<0.45).mean():5.1f}%")
    sz = np.array([r['sz'] for _, r in agg]); hs = np.array([r['hs'] for _, r in agg]); mid = np.array([i for i, _ in agg])
    a = auc(sz[hs > 0.62], sz[hs <= 0.62])
    rng = np.random.default_rng(0); uniq = np.unique(mid)
    bs = []
    for _ in range(300):
        pick = rng.choice(uniq, len(uniq)); ix = np.concatenate([np.where(mid == u)[0] for u in pick])
        bs.append(auc(sz[ix][hs[ix] > 0.62], sz[ix][hs[ix] <= 0.62]))
    print(f"\nAUC of bet SIZE alone for 'strong vs not strong' (0.5 = no leak, 1 = perfect leak): {a:.3f}  [95% CI {np.nanpercentile(bs,2.5):.3f}, {np.nanpercentile(bs,97.5):.3f}]")
    midb = [(r['sz'], r['hs']) for _, r in agg if r['act'] == 'bet' and .45 <= r['sz'] < .72]
    v = [s for s, h in midb if h > .62]; b = [s for s, h in midb if h < .45]
    print(f"inside the mid-size bucket, can size separate VALUE (HS>.62, n={len(v)}) from BLUFF (HS<.45, n={len(b)})?  AUC {auc(v, b):.3f}")
    post = [r for _, r in rows if r['st'] >= 1 and r['tc'] == 0]
    chk = [r['hs'] for r in post if r['act'] == 'check']; bet = [r['hs'] for r in post if r['act'] == 'bet']
    allh = np.array([r['hs'] for r in post])
    strong_checks = np.mean(np.array(chk) > .62) if chk else float('nan')
    print(f"\ncheck-range cap (postflop, no bet to face): n={len(post)}")
    print(f"  P(strong | Hero checks) = {100*strong_checks:.1f}%   P(strong | Hero bets) = {100*np.mean(np.array(bet)>.62):.1f}%   base rate of strong = {100*(allh>.62).mean():.1f}%")
    print(f"  P(Hero checks | strong hand) = {100*np.mean([r['act']=='check' for r in post if r['hs']>.62]):.1f}%   -> a check is {'near-' if strong_checks<0.05 else ''}'capped' evidence of weakness")


def calib(arm, opps):
    D = load(arm)
    ms = [m for m in D if (not opps or m['opp'] in opps)]
    def cat(r): return 'facing aggression' if (r['tc'] > 0 and (r['st'] >= 1 or r['tc'] > 2)) else 'no aggression faced'
    print(f"{arm}: calibration of modelled equity e against equity vs the opponent's TRUE cards (et). bias = et - e")
    for c in ['facing aggression', 'no aggression faced']:
        allr = [(m['m'], r) for m in ms for r in m['dec'] if cat(r) == c]
        if len(allr) < 30: continue
        e_all = np.array([r['e'] for _, r in allr]); edges = np.quantile(e_all, [0, .2, .4, .6, .8, 1.0])
        print(f"\n[{c}]  n={len(allr)}")
        print("   e-bin           n   mean e   mean et    bias [95% cluster CI]")
        for i in range(5):
            lo, hi = edges[i], edges[i + 1] + (1e-9 if i == 4 else 0)
            sub = [(mm, r) for mm, r in allr if lo <= r['e'] < hi]
            pm = {}
            for mm, r in sub: a = pm.setdefault(mm, [0.0, 0]); a[0] += r['et'] - r['e']; a[1] += 1
            est, l, h = boot_mean(list(pm.values()))
            print(f"   {lo:.2f}-{edges[i+1]:.2f}  {len(sub):6d}   {np.mean([r['e'] for _,r in sub]):.2f}    {np.mean([r['et'] for _,r in sub]):.2f}    {est:+.3f} [{l:+.3f},{h:+.3f}]")
        pm = {}
        for mm, r in allr: a = pm.setdefault(mm, [0.0, 0]); a[0] += r['et'] - r['e']; a[1] += 1
        est, l, h = boot_mean(list(pm.values()))
        print(f"   overall bias {est:+.3f} [{l:+.3f},{h:+.3f}]")
        if c == 'facing aggression':
            for opp in sorted({m['opp'] for m in ms}):
                pm = {}
                for m in ms:
                    if m['opp'] != opp: continue
                    for r in m['dec']:
                        if cat(r) == c: a = pm.setdefault(m['m'], [0.0, 0]); a[0] += r['et'] - r['e']; a[1] += 1
                if pm and sum(v[1] for v in pm.values()) >= 30:
                    est, l, h = boot_mean(list(pm.values()))
                    print(f"     vs {opp:16s} n={sum(v[1] for v in pm.values()):5d}  bias {est:+.3f} [{l:+.3f},{h:+.3f}]  mean q used {np.mean([r['q'] for m in ms if m['opp']==opp for r in m['dec'] if cat(r)==c]):.2f}")


if __name__ == '__main__':
    which, arm = sys.argv[1], sys.argv[2]
    opps = sys.argv[3].split(',') if len(sys.argv) > 3 else []
    {'prober': prober, 'size': size, 'calib': calib}[which](arm, opps)
