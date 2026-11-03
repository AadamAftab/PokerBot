"""Instrumented diagnostics. NEVER part of a submission: it reads engine.ORACLE (the true deal).

ProbeMixin logs one row per Hero decision:
  e   modelled equity Hero used          q   range-width fraction used (1.0 = no narrowing)
  et  equity against the opponent's TRUE cards (MC over runouts; oracle)   -> calibration of e
  hs  strength proxy = equity vs a uniformly random hand (only when Hero may bet/check)
  act f|check|call|bet|raise, sz = size as pot fraction (bets: of pot; raises: of pot after calling)
A separate RNG is used so instrumentation never perturbs the bot's own random stream.
"""
import numpy as np
import engine
from champ import Champ
from patches import SpotQChamp
from pokerlib import equity, eval_best


def known_eq(my, opp, bk, rng, n=150):
    used = set(my) | set(opp) | set(bk)
    deck = np.array([c for c in range(52) if c not in used], dtype=np.int64)
    need = 5 - len(bk)
    draw = deck[np.argsort(rng.random((n, len(deck))), axis=1)[:, :need]]
    full = np.concatenate([np.broadcast_to(np.array(bk, dtype=np.int64), (n, len(bk))), draw], axis=1)
    mine = eval_best(np.concatenate([np.broadcast_to(np.array(my, dtype=np.int64), (n, 2)), full], axis=1))
    theirs = eval_best(np.concatenate([np.broadcast_to(np.array(opp, dtype=np.int64), (n, 2)), full], axis=1))
    return float(((mine > theirs).sum() + 0.5 * (mine == theirs).sum()) / n)


class ProbeMixin:
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.dec, self.hands, self.hidx = [], [], -1
        self._q, self._e = 1.0, None
        self._prng = np.random.default_rng(12345)

    def new_hand(self, is_button):
        super().new_hand(is_button); self.hidx += 1

    def _range_q(self, s):
        q = super()._range_q(s); self._q = q; return q

    def _equity(self, s):
        self._q = 1.0
        out = super()._equity(s); self._e = out[0]; return out

    def act(self, s):
        self._e, self._q = None, 1.0
        a = super().act(s)
        if self._e is None:
            return a
        tc, pot = s['to_call'], s['pot']
        opp = [c for c in engine.ORACLE['holes'] if set(c) != set(s['hole'])][0]
        row = dict(h=self.hidx, st=s['street'], tc=tc, pot=pot, mb=s['my_bet'], ob=s['opp_bet'], stk=s['my_stack'],
                   ff=tc / max(pot - tc, 1), e=self._e, q=self._q,
                   et=known_eq(s['hole'], opp, s['board'], self._prng), hs=None, sz=None)
        if a == 'f': row['act'] = 'f'
        elif a == 'c': row['act'] = 'check' if tc == 0 else 'call'
        else:
            row['act'] = 'bet' if tc == 0 else 'raise'
            extra = a[1] - s['my_bet'] - tc
            row['sz'] = extra / max(pot if tc == 0 else pot + tc, 1)
        if tc == 0 or row['act'] in ('bet', 'raise'):
            row['hs'] = equity(s['hole'], s['board'], q=1.0, n=150, rng=self._prng)
        self.dec.append(row)
        return a

    def end_hand(self, info):
        self.hands.append(dict(h=self.hidx, chips=info['chips'], sd=info['opp_hole'] is not None))
        super().end_hand(info)


class ProbeBase(ProbeMixin, Champ):
    pass


class ProbeSpot(ProbeMixin, SpotQChamp):
    pass


class ReplayerProbe(Champ):
    """The replay-aware opponent, instrumented: logs what it believed on every hand (audit only)."""
    def __init__(self, seed=99, max_sims=300, replay=True):
        super().__init__(seed=seed, replay=replay, risk=False, max_sims=max_sims)
        self.hidx, self.hlog = -1, {}

    def new_hand(self, is_button):
        super().new_hand(is_button); self.hidx += 1

    def _replay_act(self, s, rec):
        h = self.hlog.get(self.hidx)
        if h is None:
            my, vis = frozenset(s['hole']), s['board']
            path = 'by_opp' if self.by_opp.get(my) is rec else (
                'by_flop' if len(vis) >= 3 and self.by_flop.get(frozenset(vis[:3])) is rec else 'parity')
            bk = rec['board'] if len(rec['board']) >= len(vis) else vis
            outcome = None
            if len(bk) == 5:
                a = eval_best(np.array([s['hole'] + bk], dtype=np.int64))[0]
                b = eval_best(np.array([rec['my_hole'] + bk], dtype=np.int64))[0]
                outcome = 'opp_wins' if a > b else ('opp_loses' if a < b else 'tie')
            h = self.hlog[self.hidx] = dict(path=path, street=s['street'], full=len(bk) == 5, outcome=outcome,
                                            lag=len(self.rec) - rec['idx'])
        return super()._replay_act(s, rec)


from patches2 import TrapSpotQ, MixedSpotQ


class ProbeTrap(ProbeMixin, TrapSpotQ):
    pass


class ProbeMixed(ProbeMixin, MixedSpotQ):
    pass


PROBES = {
    'trap30': lambda sd, ms: ProbeTrap(seed=sd, max_sims=ms, p=0.3),
    'trap50': lambda sd, ms: ProbeTrap(seed=sd, max_sims=ms, p=0.5),
    'mixed': lambda sd, ms: ProbeMixed(seed=sd, max_sims=ms),
    'base': lambda sd, ms: ProbeBase(seed=sd, replay=False, risk=False, max_sims=ms),
    'spotq': lambda sd, ms: ProbeSpot(seed=sd, replay=False, risk=False, max_sims=ms),
}
