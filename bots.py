"""Opponent zoo + the strategy bot."""
import random
import numpy as np
from pokerlib import equity, preflop_strength


def _raise_to(s, frac):
    """Raise sized as a fraction of the pot (after calling), clamped legal."""
    to = s['my_bet'] + s['to_call'] + int(frac * (s['pot'] + s['to_call']))
    return ('r', max(s['min_raise_to'], min(to, s['max_raise_to'])))


class Base:
    def new_hand(self, is_button): pass
    def observe(self, actor_is_me, kind, state): pass
    def end_hand(self, info): pass


class RandomBot(Base):
    def act(self, s):
        x = random.random()
        if x < 0.2 and s['to_call'] > 0: return 'f'
        if x < 0.7: return 'c'
        return _raise_to(s, random.choice((0.5, 1.0)))


class CallStation(Base):
    def act(self, s): return 'c'


class Maniac(Base):
    def act(self, s): return _raise_to(s, 1.0)


class Rock(Base):
    """Plays only strong preflop hands, then bets/calls only with a real made hand."""
    def act(self, s):
        if s['street'] == 0:
            st = preflop_strength()[s['hole'][0], s['hole'][1]]
            if st > 0.62: return _raise_to(s, 0.75) if s['to_call'] <= 2 else 'c'
            return 'c' if s['to_call'] == 0 else 'f'
        e = equity(s['hole'], s['board'], n=200)
        if e > 0.7: return _raise_to(s, 0.6)
        if e > 0.55 and s['to_call'] <= 0.5 * s['pot']: return 'c'
        return 'c' if s['to_call'] == 0 else 'f'


class BasicEq(Base):
    """Equity vs a random hand with pot odds; no opponent modelling."""
    def act(self, s):
        e = equity(s['hole'], s['board'], n=300)
        need = s['to_call'] / (s['pot'] + s['to_call']) if s['to_call'] else 0
        if e > 0.68: return _raise_to(s, 0.7)
        if e >= need + 0.03 and (e > 0.42 or s['to_call'] == 0): return 'c'
        return 'f' if s['to_call'] > 0 else 'c'


class Hero(Base):
    """Range-aware equity bot with simple opponent modelling.

    - Estimates equity against a *narrowed* opponent range when they bet/raise.
    - Value bets strong hands, bluffs more against opponents who fold a lot.
    - Tracks opponent fold-to-bet and raise rates (with priors).
    """
    def __init__(self, seed=None, n_sims=500):
        self.rng = np.random.default_rng(seed)
        self.n_sims = n_sims
        self.faced, self.folded = 3.0, 1.0      # opp facing our bets / folding (prior 33%)
        self.opp_dec, self.opp_raise = 6.0, 2.0  # opp decisions / raises (prior 33%)
        self.opp_raised_street = False

    # --- opponent model ---
    def new_hand(self, is_button): self.opp_raised_street = False

    def observe(self, actor_is_me, kind, state):
        if actor_is_me:
            return
        self.opp_dec += 1
        if kind == 'r':
            self.opp_raise += 1
        # opponent acting while facing our bet
        if state['to_call'] > 0:
            self.faced += 1
            if kind == 'f': self.folded += 1

    @property
    def fold_rate(self): return self.folded / self.faced

    @property
    def aggr(self): return self.opp_raise / self.opp_dec

    # --- decisions ---
    rho = 0.0   # risk: >0 protect a lead (tighter, smaller pots), <0 chase (looser, bigger)

    def _range_q(self, s):
        """Fraction of opponent hands (strongest first) kept after they bet/raise."""
        frac = s['to_call'] / max(s['pot'] - s['to_call'], 1)
        q = 1 / (1 + 2.2 * frac)
        return min(1.0, max(0.1, q * (self.aggr / 0.33) ** 0.5))

    def _equity(self, s):
        facing_raise = s['to_call'] > (BB_PREFLOP if s['street'] == 0 else 0)
        q = self._range_q(s) if facing_raise else 1.0
        e = equity(s['hole'], s['board'], q=q, n=self.n_sims, rng=self.rng)
        return e, facing_raise

    def act(self, s):
        e, facing_raise = self._equity(s)
        return self._decide(s, e, facing_raise)

    def _decide(self, s, e, facing_raise):
        tc, pot, street, rho = s['to_call'], s['pot'], s['street'], self.rho
        r = self.rng.random()
        need = tc / (pot + tc) if tc else 0.0
        size = lambda f: _raise_to(s, f * (1 - 0.4 * rho))
        bluff_mult = max(0.0, 1 - 0.8 * rho)

        if street == 0 and not facing_raise:                       # open / limp decisions
            if e > 0.55 + 0.04 * rho: return size(0.9 if s['is_button'] else 1.0)
            if tc == 0: return 'c'
            return 'c' if e > 0.40 + 0.03 * rho else 'f'

        if tc > 0:                                                 # facing a bet
            if e > 0.80 + 0.05 * rho or (e > 0.68 + 0.05 * rho and r < 0.4): return size(0.8)
            if e >= need + 0.04 + 0.08 * rho: return 'c'
            if e > 0.25 and r < (0.06 + 0.15 * max(0, self.fold_rate - 0.4)) * bluff_mult:
                return size(0.8)                                   # occasional bluff-raise
            return 'f'

        if e > 0.62 + 0.06 * rho:                                  # value bet
            return size(0.65) if r < 0.9 else 'c'
        if e > 0.45 + 0.05 * rho:
            return size(0.33) if r < 0.2 else 'c'
        bluff = min(0.5, max(0.05, 0.12 + 0.9 * (self.fold_rate - 0.35))) * bluff_mult
        return size(0.6) if r < bluff else 'c'


BB_PREFLOP = 2
