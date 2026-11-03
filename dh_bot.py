"""Discard Hold'em bot (protocol 1.0). Equity-driven, range-aware, time-governed.

- No cross-hand card memory (replay inference is deliberately NOT implemented; see DH_SPEC_DIGEST.md).
- Opponent model = aggregate action frequencies by spot (spotq), which is ordinary opponent modelling.
- Any exception falls back to: check if legal else fold (discard: lowest rank), exactly the engine's own default.
"""
import json, sys, time, os
import numpy as np
from dhlib import (parse_card, card_str, sim_flop3, sim_post, sim_pre, preflop_value, preflop_pct,
                   keep_best, lowest_discard)

PRIORS = {'lead': 0.40, 'vs_bet': 0.20, 'pre_open': 0.50, 'pre_3bet': 0.15}
N0 = 8


class Bot:
    def __init__(self, seed=None, fixed_n=None, clock=time.perf_counter, **kw):
        self.rng = np.random.default_rng(seed)
        self.fixed_n = fixed_n                       # None = time governor; int = deterministic sims (experiments)
        self.clock = clock
        self.total_hands = 400
        self.hand_id = 0
        self.hole, self.opp_disc = [], None
        self.flop_q = 1.0
        self.spot = {k: [p * N0, N0] for k, p in PRIORS.items()}     # [aggressive actions, opportunities]
        self.faced, self.folded = 3.0, 1.0                           # opponent facing OUR bets / folding to them
        self.ms_per_sim = 0.25
        self.errors = 0
        self.done = False
        self.pre_faced, self.pre_folded = 4.0, 1.0                    # opponent facing OUR preflop raise / folding
        self.cons = [0.85 * 8.0, 8.0]                                # opp discards consistent with a sensible policy
        self.p = dict(open_btn=0.40, defend_btn=0.10, iso_bb=0.55, ex=1.0, eps=1.0, tie=1.0)
        self.p.update(kw.get('params', {}) or {})

    # ------------------------------------------------------------------ protocol
    def handle(self, m):
        t = m.get('type')
        try:
            if t == 'init':
                self.total_hands = int(m.get('total_hands', 400))
            elif t == 'new_hand':
                self.hand_id = int(m['hand_id'])
                self.hole = [parse_card(c) for c in m['hole_cards']]
                self.opp_disc = None
                self.flop_q = 1.0
            elif t == 'discard_result':
                self.hole = [parse_card(c) for c in m['hole_cards']]
                self.opp_disc = parse_card(m['opp_discard'])
            elif t == 'hand_result':
                self._learn(m.get('history', []))
                self._learn_discard(m)
            elif t == 'match_end':
                self.done = True
            elif t == 'action_request':
                try:
                    return self._act(m)
                except Exception:
                    self.errors += 1
                    return self._fallback_action(m)
            elif t == 'discard_request':
                try:
                    return self._discard(m)
                except Exception:
                    self.errors += 1
                    hc = [parse_card(c) for c in m['hole_cards']]
                    return {'type': 'discard', 'card': card_str(hc[lowest_discard(hc)])}
        except Exception:
            self.errors += 1
        return None

    @staticmethod
    def _fallback_action(m):
        legal = m.get('legal_actions', [])
        return {'type': 'action', 'action': 'check' if 'check' in legal else ('fold' if 'fold' in legal else 'call')}

    # ------------------------------------------------------------------ opponent model
    def _learn(self, history):
        """Rebuild the hand from the complete history and update spot frequencies (no cards are stored)."""
        bets = {'me': 0, 'opp': 0}
        street = None
        for h in history:
            a, who = h['action'], h['actor']
            if h['street'] != street:
                street = h['street']
                if street == 'preflop':      # blinds are not listed; the button (1) acts first preflop
                    bets = {'opp': 1, 'me': 2} if who == 'opp' else {'me': 1, 'opp': 2}
                else:
                    bets = {'me': 0, 'opp': 0}
            tc = bets['me' if who == 'opp' else 'opp'] - bets[who]
            if who == 'opp':
                if street == 'preflop':
                    spot = 'pre_open' if bets['me'] <= 2 else 'pre_3bet'
                else:
                    spot = 'lead' if tc == 0 else 'vs_bet'
                c = self.spot[spot]
                c[1] += 1
                c[0] += a == 'raise'
                if street != 'preflop' and tc > 0:         # postflop response to our bet (preflop folds excluded)
                    self.faced += 1
                    self.folded += a == 'fold'
                elif street == 'preflop' and tc > 0 and bets['me'] > 2:   # response to our preflop raise (not to the blind)
                    self.pre_faced += 1
                    self.pre_folded += a == 'fold'
            if a in ('call', 'raise'):
                bets[who] = h.get('amount', bets[who])

    def _learn_discard(self, m):
        """At showdown the opponent's full 3-card hand is public: was their discard what a sensible policy does?
        Only a counter is kept (no cards)."""
        try:
            if m.get('result') != 'showdown' or not m.get('opp_discard') or not m.get('opp_hole_cards'):
                return
            trip = np.array([[parse_card(c) for c in m['opp_hole_cards']] + [parse_card(m['opp_discard'])]])
            flop = np.array([[parse_card(c) for c in m['board'][:3]]])
            d, _ = keep_best(trip, flop)
            ok = int(d[0]) == 2 or (trip[0, 2] >> 2) <= (trip[0, :2] >> 2).min()
            self.cons[1] += 1
            self.cons[0] += bool(ok)
        except Exception:
            self.errors += 1

    @property
    def disc_eps(self):
        """Weight given to opponent hands whose discard looks inconsistent with a sensible policy."""
        if not self.p['eps']:
            return 0.15
        return float(min(0.9, max(0.05, 1.0 - self.cons[0] / self.cons[1])))

    @property
    def fold_rate(self):
        return self.folded / self.faced

    @property
    def obs_conf(self):
        """Evidence gate: 0 until we have seen a few opponent responses, 1 after ~12 (priors count as 4)."""
        return float(min(1.0, max(0.0, (self.faced - 4.0) / 12.0)))

    def _range_q(self, req):
        if req['street'] == 'preflop':
            spot = 'pre_open' if req['my_bet'] <= 2 else 'pre_3bet'
        else:
            spot = 'lead' if req['my_bet'] == 0 else 'vs_bet'
        k, n = self.spot[spot]
        frac = req['to_call'] / max(req['pot'] - req['to_call'], 1)
        size_adj = min(1.2, max(0.6, (0.6 / max(frac, 0.05)) ** 0.3))
        return min(1.0, max(0.08, (k / n) * size_adj))

    # ------------------------------------------------------------------ time governor
    def _n(self, base, req):
        if self.fixed_n:
            return self.fixed_n
        tl = req.get('time_left_ms', 60000)
        hands_left = max(1, self.total_hands - self.hand_id + 1)
        spendable = max(tl - 4000.0, 0.0) * 0.40                    # keep a reserve and only plan to use ~40% of the rest
        per_req = spendable / hands_left / 4.0                      # up to ~4 sim-requests per hand
        n = int(per_req / self.ms_per_sim)
        return 0 if tl < 2500 else int(min(max(n, 30), 2 * base))

    def _timed(self, fn, n, *a, **k):
        t0 = self.clock()
        out = fn(*a, n=n, rng=self.rng, **k)
        self.ms_per_sim = 0.8 * self.ms_per_sim + 0.2 * ((self.clock() - t0) * 1000 / max(n, 1))
        return out

    # ------------------------------------------------------------------ decisions
    def _raise_to(self, req, frac):
        tc, pot, ob = req['to_call'], req['pot'], req['opp_bet']
        t = int(round(ob + frac * (pot + tc)))
        t = max(req['min_raise'], min(t, req['max_raise']))
        if t >= 0.6 * req['max_raise']:
            t = req['max_raise']
        return {'type': 'action', 'action': 'raise', 'amount': int(t)}

    @staticmethod
    def _act_simple(kind):
        return {'type': 'action', 'action': kind}

    def _equity(self, req):
        board = [parse_card(c) for c in req['board']]
        tc = req['to_call']
        facing = req['opp_bet'] > 2 if req['street'] == 'preflop' else tc > 0
        q = self._range_q(req) if facing else 1.0
        if req['street'] == 'preflop':
            if not facing:
                return preflop_value(self.hole), False
            n = self._n(250, req)
            if n == 0:
                return preflop_value(self.hole), True
            return self._timed(sim_pre, n, self.hole, q=q), True
        n = self._n(300, req)
        if len(self.hole) == 3:                                     # flop betting, before the discard
            self.flop_q = q if facing else self.flop_q
            if n == 0:
                return 0.5, facing
            return max(self._timed(sim_flop3, n, self.hole, board, q=q)), facing
        if n == 0:
            return 0.5, facing
        return self._timed(sim_post, n, self.hole, board, self.opp_disc, q=q, eps=self.disc_eps), facing

    def _act(self, req):
        legal = req['legal_actions']
        can_raise = 'raise' in legal
        tc = req['to_call']
        e, facing = self._equity(req)
        r = self.rng.random()
        if req['street'] == 'preflop' and not facing:
            p = preflop_pct(self.hole)
            if tc > 0:                                                 # button, first to act
                open_thr = self.p['open_btn']
                if self.p['ex']:                                        # steal wider against opponents that fold to raises
                    pf = self.pre_folded / self.pre_faced
                    pc = min(1.0, max(0.0, (self.pre_faced - 4.0) / 10.0))
                    open_thr *= 1.0 - pc * min(1.0, max(0.0, (pf - 0.3) / 0.3))
                if can_raise and p >= open_thr:
                    return self._raise_to(req, 1.0)
                return self._act_simple('call' if p >= self.p['defend_btn'] else 'fold')
            if can_raise and p >= self.p['iso_bb']:                     # big blind facing a limp
                return self._raise_to(req, 0.9)
            return self._act_simple('check')

        pot = req['pot']
        need = tc / (pot + tc) if tc else 0.0
        if tc > 0:
            if can_raise and (e > 0.80 or (e > 0.68 and r < 0.4)):
                return self._raise_to(req, 0.8)
            if e >= need + 0.04:
                return self._act_simple('call')
            if can_raise and e > 0.25 and r < 0.06 + 0.15 * max(0.0, self.fold_rate - 0.4):
                return self._raise_to(req, 0.8)
            return self._act_simple('fold')
        fr, conf = self.fold_rate, self.obs_conf
        station = bool(self.p['ex']) and fr < 0.25 and conf > 0                # calls too much: value-bet thinner, never bluff
        if can_raise and e > (0.55 if station else 0.62):
            return self._raise_to(req, 0.75 if station else 0.65) if r < 0.9 else self._act_simple('check')
        if can_raise and e > 0.45:
            return self._raise_to(req, 0.33) if r < 0.2 else self._act_simple('check')
        bluff = min(0.5, max(0.05, 0.12 + 0.9 * (fr - 0.35)))
        if self.p['ex']:
            bluff = 0.0 if station else min(0.65, max(0.05, 0.12 + 0.9 * (fr - 0.35)) * (0.5 + 0.5 * conf) + (0.1 * conf if fr > 0.6 else 0.0))
        if can_raise and r < bluff:
            return self._raise_to(req, 0.6)
        return self._act_simple('check')

    def _discard(self, req):
        hole = [parse_card(c) for c in req['hole_cards']]
        flop = [parse_card(c) for c in req['board']]
        n = self._n(300, req)
        if n == 0:
            d, _ = keep_best(np.array([hole]), np.array([flop]))
            i = int(d[0])
        else:
            eqs = self._timed(sim_flop3, n, hole, flop, q=self.flop_q)
            i = int(np.argmax(eqs))
            if self.p['tie']:        # near-equal: discard the lowest card (simulation noise should not pick randomly)
                near = [j for j in range(3) if eqs[j] >= max(eqs) - 0.01]
                i = min(near, key=lambda j: (hole[j] >> 2, hole[j] & 3))
        return {'type': 'discard', 'card': card_str(hole[i])}


def main():
    seed = os.environ.get('BOT_SEED')
    bot = Bot(seed=int(seed) if seed else None)
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        reply = bot.handle(msg)
        if reply is not None:
            out.write(json.dumps(reply) + '\n')
            out.flush()
        if bot.done:
            break


if __name__ == '__main__':
    main()
