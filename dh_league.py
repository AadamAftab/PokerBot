"""Discard Hold'em opponents. Every agent implements handle(msg)->reply|None on the real protocol messages.

Starter     - the official starter bot (check/call; discard lowest rank)
CallStation - calls everything, discards lowest
Random      - uniformly random legal action / discard
Maniac      - raises pot-size at every chance
Jammer      - all-in whenever possible
Prober      - half-pot bet whenever checked to, otherwise equity-based
Rock        - equity-based, only strong hands, never bluffs
LAG         - equity-based, wide and aggressive, bluffs often
Solid       - equity-based, balanced thresholds, MC discard
"""
import numpy as np
from dhlib import (parse_card, card_str, sim_flop3, sim_post, preflop_value, preflop_pct, keep_best, lowest_discard)


class Agent:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.hole, self.opp_disc = [], None
        self.done = False

    def handle(self, m):
        t = m.get('type')
        if t == 'new_hand':
            self.hole = [parse_card(c) for c in m['hole_cards']]; self.opp_disc = None
        elif t == 'discard_result':
            self.hole = [parse_card(c) for c in m['hole_cards']]; self.opp_disc = parse_card(m['opp_discard'])
        elif t == 'action_request':
            return self.act(m)
        elif t == 'discard_request':
            return self.discard(m)
        elif t == 'match_end':
            self.done = True
        return None

    @staticmethod
    def check_call(m):
        return {'type': 'action', 'action': 'check' if 'check' in m['legal_actions'] else 'call'}

    @staticmethod
    def raise_to(m, frac):
        if 'raise' not in m['legal_actions']:
            return Agent.check_call(m)
        t = int(round(m['opp_bet'] + frac * (m['pot'] + m['to_call'])))
        return {'type': 'action', 'action': 'raise', 'amount': int(max(m['min_raise'], min(t, m['max_raise'])))}

    def discard(self, m):
        hc = [parse_card(c) for c in m['hole_cards']]
        return {'type': 'discard', 'card': card_str(hc[lowest_discard(hc)])}


class Starter(Agent):
    def act(self, m):
        return self.check_call(m)


CallStation = Starter


class Random(Agent):
    def act(self, m):
        a = self.rng.choice(m['legal_actions'])
        if a == 'raise':
            return {'type': 'action', 'action': 'raise', 'amount': int(self.rng.integers(m['min_raise'], min(m['max_raise'], m['min_raise'] + 40) + 1))}
        return {'type': 'action', 'action': str(a)}

    def discard(self, m):
        return {'type': 'discard', 'card': m['hole_cards'][int(self.rng.integers(3))]}


class Maniac(Agent):
    def act(self, m):
        return self.raise_to(m, 1.0)


class Jammer(Agent):
    def act(self, m):
        if 'raise' in m['legal_actions']:
            return {'type': 'action', 'action': 'raise', 'amount': m['max_raise']}
        return self.check_call(m)


class EqAgent(Agent):
    """Equity-threshold bot. n sims are small so league runs stay fast."""
    open_pct, value, bluff, margin, n, mc_discard, probe = 0.35, 0.65, 0.05, 0.03, 120, True, False

    def __init__(self, seed=None, **kw):
        super().__init__(seed)
        self.flop_q = 1.0
        for k, v in kw.items():
            setattr(self, k, v)

    def act(self, m):
        legal, tc, st = m['legal_actions'], m['to_call'], m['street']
        r = self.rng.random()
        if st == 'preflop':
            p = preflop_pct(self.hole)
            if m['opp_bet'] <= 2:
                if tc > 0:
                    return self.raise_to(m, 1.0) if p >= self.open_pct else (self.check_call(m) if p >= 0.12 else {'type': 'action', 'action': 'fold'})
                return self.raise_to(m, 0.9) if p >= 0.6 else self.check_call(m)
            e = 0.35 + 0.5 * p
        else:
            board = [parse_card(c) for c in m['board']]
            if len(self.hole) == 3:
                e = max(sim_flop3(self.hole, board, n=self.n, rng=self.rng))
            else:
                e = sim_post(self.hole, board, self.opp_disc, n=self.n, rng=self.rng)
        if tc == 0 and st != 'preflop' and self.probe:
            return self.raise_to(m, 0.5) if r < 0.85 else self.check_call(m)
        need = tc / (m['pot'] + tc) if tc else 0
        if tc > 0:
            if e > self.value + 0.12 and 'raise' in legal:
                return self.raise_to(m, 0.8)
            return self.check_call(m) if e >= need + self.margin else {'type': 'action', 'action': 'fold'}
        if e > self.value:
            return self.raise_to(m, 0.65)
        if r < self.bluff:
            return self.raise_to(m, 0.6)
        return self.check_call(m)

    def discard(self, m):
        hc = [parse_card(c) for c in m['hole_cards']]
        fl = [parse_card(c) for c in m['board']]
        if self.mc_discard:
            i = int(np.argmax(sim_flop3(hc, fl, n=max(60, self.n // 2), rng=self.rng)))
        else:
            d, _ = keep_best(np.array([hc]), np.array([fl]))
            i = int(d[0])
        return {'type': 'discard', 'card': card_str(hc[i])}


class Prober(EqAgent):
    probe = True
    value, bluff = 0.7, 0.0


class Rock(EqAgent):
    open_pct, value, bluff, margin = 0.7, 0.72, 0.0, 0.08


class LAG(EqAgent):
    open_pct, value, bluff, margin = 0.15, 0.55, 0.28, -0.03


class Solid(EqAgent):
    pass


class LowDiscardSolid(EqAgent):
    """Solid play but discards the lowest rank - tests robustness of the opponent-discard inference."""
    mc_discard = False

    def discard(self, m):
        return Agent.discard(self, m)


class Folder(EqAgent):
    """Folds to every postflop bet and half its preflop spots: should be farmed by bluffing/stealing."""
    def act(self, m):
        if m['to_call'] > 0 and (m['street'] != 'preflop' or m['opp_bet'] > 2 or self.rng.random() < 0.5):
            return {'type': 'action', 'action': 'fold'}
        return self.check_call(m)


class RandDiscSolid(EqAgent):
    """Solid betting but random discards (tests the discard-consistency tracker)."""
    def discard(self, m):
        return {'type': 'discard', 'card': m['hole_cards'][int(self.rng.integers(3))]}


LEAGUE = {
    'Folder': Folder, 'RandDiscSolid': RandDiscSolid,
    'Starter': Starter, 'Random': Random, 'Maniac': Maniac, 'Jammer': Jammer, 'Prober': Prober,
    'Rock': Rock, 'LAG': LAG, 'Solid': Solid, 'LowDiscardSolid': LowDiscardSolid,
}
