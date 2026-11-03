"""Phase 2 experimental patches over the known-good `spotq` (range model and everything else unchanged).

Common mechanism being tested: Hero's checking range is capped (only ~8% of checks are strong hands vs ~71% of bets),
so stabs into it are profitable.  Each patch is ONE change; the combined arm is evaluated last.

TrapSpotQ(p, thr)  Selective strong-hand checks.  Justification: only when out of position and first to act is a check
                   ever followed by an opponent decision, so that is the only place a check can be probed.  Only hands with
                   equity > thr (default 0.75) are trapped, where giving a free card costs least.  p is a pre-registered
                   grid value (0.3, 0.5): p is the share of those hands moved into the checking range.
MixedSpotQ         Mixed betting/checking.  The hard threshold 'bet if e>0.62 (90%)' becomes a linear ramp
                   P(bet)=clip((e-0.50)/0.24, 0, 1) with the same midpoint (0.62), same bet sizes.  Mixing is concentrated
                   where the EV difference between betting and checking is small, so the EV cost is second-order.
CallProbeSpotQ     Call-threshold adjustment against probes.  Only converts a FOLD into a CALL, when the opponent's measured
                   lead frequency f >= 0.5 and equity is within delta = 0.07*clip((f-0.5)/0.4,0,1) of the usual call threshold.
TrapCallSpotQ      TrapSpotQ(p=0.3) + CallProbeSpotQ (the cumulative policy).
"""
from patches import SpotQChamp
from bots import _raise_to


class _Exp(SpotQChamp):
    def __init__(self, seed=None, max_sims=800, **kw):
        super().__init__(seed=seed, replay=False, risk=False, max_sims=max_sims)


def _oop_first(s):
    return s['street'] >= 1 and s['to_call'] == 0 and not s['is_button'] and s['my_bet'] == 0


class TrapSpotQ(_Exp):
    def __init__(self, p=0.3, thr=0.75, **kw):
        super().__init__(**kw); self.p, self.thr = p, thr

    def _decide(self, s, e, facing_raise):
        if _oop_first(s) and e > self.thr and self.rng.random() < self.p:
            return 'c'
        return super()._decide(s, e, facing_raise)


class MixedSpotQ(_Exp):
    def _decide(self, s, e, facing_raise):
        if s['street'] >= 1 and s['to_call'] == 0 and e > 0.45:
            p = min(1.0, max(0.0, (e - 0.50) / 0.24))
            if self.rng.random() < p:
                return _raise_to(s, 0.65 if e > 0.62 else 0.33)
            return 'c'
        return super()._decide(s, e, facing_raise)


class CallProbeSpotQ(_Exp):
    def _decide(self, s, e, facing_raise):
        res = super()._decide(s, e, facing_raise)
        if res == 'f' and s['street'] >= 1 and s['to_call'] > 0 and s['my_bet'] == 0:
            k, n = self.spot['lead']; f = k / n
            if f >= 0.5:
                need = s['to_call'] / (s['pot'] + s['to_call'])
                delta = 0.07 * min(1.0, (f - 0.5) / 0.4)
                if e >= need + 0.04 - delta:
                    return 'c'
        return res


class TrapCallSpotQ(CallProbeSpotQ, TrapSpotQ):
    def __init__(self, p=0.3, thr=0.75, **kw):
        TrapSpotQ.__init__(self, p=p, thr=thr, **kw)


ARM_CLASSES = {
    'trap30': lambda sd, ms: TrapSpotQ(seed=sd, max_sims=ms, p=0.3),
    'trap50': lambda sd, ms: TrapSpotQ(seed=sd, max_sims=ms, p=0.5),
    'mixed': lambda sd, ms: MixedSpotQ(seed=sd, max_sims=ms),
    'callprobe': lambda sd, ms: CallProbeSpotQ(seed=sd, max_sims=ms),
    'trap30_call': lambda sd, ms: TrapCallSpotQ(seed=sd, max_sims=ms, p=0.3),
}
