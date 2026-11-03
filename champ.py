"""Champ = Hero + features aimed at the event's structure.

1. Duplicate-replay exploitation: every deal is played twice with swapped seats, so in the
   second half the opponent holds exactly the cards we held in the first half (and the board
   is the same). We detect replays and use that knowledge.
2. Match-aware risk: a match pays 1 / 0.5 / 0, so once ahead we shrink variance and when
   far behind late we take more of it (rho in [-1, 1], see Hero._decide).
3. Time governor: spends the cumulative time bank evenly over the remaining hands.
4. Safety: any exception falls back to a legal move.
"""
import math
import time
import numpy as np
from bots import Hero, _raise_to
from pokerlib import eval_best


def _ncdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


class Champ(Hero):
    is_hero = True
    def __init__(self, seed=None, total_hands=400, replay=True, risk=True,
                 risk_scale=1.0, bank=60.0, max_sims=800, strict_lag=True):
        super().__init__(seed, n_sims=500)
        self.total_hands, self.max_sims = total_hands, max_sims
        self.use_replay, self.use_risk, self.risk_scale = replay, risk, risk_scale
        self.bank, self.used, self.cost = bank, 0.0, 6e-5     # cost: seconds per simulation
        self.decisions = 0
        self.rec = []                  # finished hands
        self.by_opp, self.by_flop = {}, {}
        self.par_hit, self.par_miss = [0, 0], [0, 0]
        self.chips = 0.0
        self.pair_sumsq, self.pairs = 0.0, 0
        self._half = 0.0
        self.replay_acts = 0
        self.strict_lag = strict_lag   # preflop showdown-card match only trusted at a confirmed hand lag
        self.lag_hits = {}             # lag (hands apart) -> flop-confirmed replays seen
        self.errors = 0
        self.lat = []                  # per-decision latency (s)

    @classmethod
    def finals(cls, **kw):
        """Stage 2: ~500 deals per match, chip margin matters for tie-breaks -> milder risk play."""
        return cls(total_hands=1000, risk_scale=0.5, **kw)

    # ------------------------------------------------------------------ bookkeeping
    def end_hand(self, info):
        my, opp, board, chips = info['my_hole'], info['opp_hole'], info['board'], info['chips']
        p, prev = len(self.rec) % 2, (self.rec[-1] if self.rec else None)
        if prev is not None:                     # learn which hands are replays of the previous one
            truth = None
            if opp is not None:
                truth = set(opp) == set(prev['my_hole'])
            elif len(board) >= 3 and len(prev['board']) >= 3:
                truth = set(board[:3]) == set(prev['board'][:3])
            if truth is True: self.par_hit[p] += 1
            elif truth is False: self.par_miss[p] += 1
        idx = len(self.rec)
        if len(board) >= 3:                      # flop equality is collision-proof evidence: learn the lag
            src = self.by_flop.get(frozenset(board[:3]))
            if src is not None and not (set(src['my_hole']) & set(my)):
                lag = idx - src['idx']
                self.lag_hits[lag] = self.lag_hits.get(lag, 0) + 1
        rec = dict(my_hole=my, opp_hole=opp, board=board, idx=idx)
        self.rec.append(rec)
        if opp is not None: self.by_opp[frozenset(opp)] = rec
        if len(board) >= 3: self.by_flop[frozenset(board[:3])] = rec
        self.chips += chips
        if len(self.rec) % 2 == 1:
            self._half = chips
        else:
            self.pair_sumsq += (self._half + chips) ** 2
            self.pairs += 1

    # ------------------------------------------------------------------ replay detection
    @staticmethod
    def _consistent(s, rec):
        my, vis, opp, kb = set(s['hole']), s['board'], set(rec['my_hole']), rec['board']
        if my & opp or opp & set(vis) or set(kb) & my or opp & set(kb): return False
        if rec['opp_hole'] is not None and set(rec['opp_hole']) != my: return False
        if len(vis) >= 3 and len(kb) >= 3 and set(vis[:3]) != set(kb[:3]): return False
        return all(vis[i] == kb[i] for i in range(3, min(len(vis), len(kb))))

    def _source(self, s):
        """Earlier hand this one is the seat-swapped replay of, or None."""
        if not (self.use_replay and self.rec): return None
        my, vis = frozenset(s['hole']), s['board']
        rec = self.by_opp.get(my)                      # we now hold what they showed down before
        if rec is not None and self.strict_lag and len(vis) < 3:
            lag = len(self.rec) - rec['idx']           # only 1/1326 chance-matches are plausible at lag 1;
            if lag != 1 and self.lag_hits.get(lag, 0) < 2:   # elsewhere require a flop-confirmed lag
                rec = None
        if rec is None and len(vis) >= 3:              # same flop as an earlier hand
            rec = self.by_flop.get(frozenset(vis[:3]))
        if rec is None:                                # learned alternating pattern (preflop knowledge)
            p = len(self.rec) % 2
            if self.par_hit[p] >= 2 and self.par_miss[p] == 0:
                rec = self.rec[-1]
        return rec if rec is not None and self._consistent(s, rec) else None

    def _known_equity(self, my, opp, bk, n=300):
        used = set(my) | set(opp) | set(bk)
        deck = np.array([c for c in range(52) if c not in used], dtype=np.int64)
        need = 5 - len(bk)
        draw = deck[np.argsort(self.rng.random((n, len(deck))), axis=1)[:, :need]]
        full = np.concatenate([np.broadcast_to(np.array(bk, dtype=np.int64), (n, len(bk))), draw], axis=1)
        mine = eval_best(np.concatenate([np.broadcast_to(np.array(my, dtype=np.int64), (n, 2)), full], axis=1))
        theirs = eval_best(np.concatenate([np.broadcast_to(np.array(opp, dtype=np.int64), (n, 2)), full], axis=1))
        return float(((mine > theirs).sum() + 0.5 * (mine == theirs).sum()) / n)

    def _replay_act(self, s, rec):
        my, opp, vis, kb = s['hole'], rec['my_hole'], s['board'], rec['board']
        bk = kb if len(kb) >= len(vis) else vis          # board cards we already know about
        tc, r = s['to_call'], self.rng.random()
        if len(bk) == 5:                                 # whole outcome is known
            a = eval_best(np.array([my + bk], dtype=np.int64))[0]
            b = eval_best(np.array([opp + bk], dtype=np.int64))[0]
            if a > b:                                    # we win: extract as much as they will pay
                if tc > 0: return _raise_to(s, 0.9) if self.fold_rate < 0.45 else 'c'
                return _raise_to(s, 0.75 if self.fold_rate < 0.45 else (0.4 if self.fold_rate < 0.7 else 0.25))
            if a < b:                                    # we lose: give up, or bluff folders
                if tc == 0 and self.fold_rate > 0.6 and r < 0.4: return _raise_to(s, 0.5)
                return 'f' if tc > 0 else 'c'
            return 'c'
        e = self._known_equity(my, opp, bk)
        return self._decide(s, e, s['to_call'] > (2 if s['street'] == 0 else 0))

    # ------------------------------------------------------------------ risk + time
    def _update_rho(self):
        self.rho = 0.0
        if not self.use_risk or self.pairs < 25: return
        var = (40.0 ** 2 * 10 + self.pair_sumsq) / (10 + self.pairs)   # per-deal chip variance
        deals_left = max((self.total_hands - len(self.rec)) / 2, 1)
        p_win = _ncdf(self.chips / math.sqrt(var * deals_left))
        rho = max(-1.0, min(1.0, (p_win - 0.5) / 0.4))
        self.rho = (rho * 0.5 if rho < 0 else rho) * self.risk_scale

    def _equity(self, s):
        left = max(self.total_hands - len(self.rec), 1)
        remaining = self.bank * 0.85 - self.used
        per_hand_dec = max(2.0, self.decisions / len(self.rec)) if len(self.rec) > 5 else 3.0
        n = 30 if remaining < 1.0 else int(max(40, min(self.max_sims, remaining / left / per_hand_dec / self.cost)))
        self.n_sims = n
        t = time.perf_counter()
        out = super()._equity(s)
        self.cost = 0.8 * self.cost + 0.2 * (time.perf_counter() - t) / n
        return out

    # ------------------------------------------------------------------ main entry
    def act(self, s):
        t0 = time.perf_counter()
        self.decisions += 1
        try:
            self._update_rho()
            rec = self._source(s)
            if rec is not None:
                self.replay_acts += 1
                a = self._replay_act(s, rec)
            else:
                a = super().act(s)
        except Exception:
            self.errors += 1
            a = 'c' if s['to_call'] == 0 else 'f'
        dt = time.perf_counter() - t0
        self.used += dt
        self.lat.append(dt)
        return a
