"""Second adversarial league. Each opponent documents the assumption it attacks.
All use a private seeded RNG (derived from the global `random`, which run_block seeds per match)
so a match is reproducible. None may emit an illegal action (see test_regression.py).

TightPassive    - folds/limps nearly everything, raises only premiums.         [baseline archetype]
LAG             - loose-aggressive, equity-driven.                             [baseline archetype]
ProberV         - stabs when checked to, with configurable size and frequency. [check-range attack]
PartialJammer   - shoves a fraction p of hands preflop.                        [range-width attack]
Flipper         - TightPassive for 100 hands, then LAG.                        [strategy switch]
DelayedExploiter- measures how often Hero folds to its bets for 40 observations, then
                  stabs relentlessly if Hero over-folds, else value-bets only.  [overfold attack]
SizeReader      - learns how often it loses when it calls Hero's bets of each size bucket and
                  adjusts its calling threshold accordingly.                   [bet-size leakage attack]
SizeBlind       - identical to SizeReader but ignores sizes (control).         [control for the above]
"""
import random
import numpy as np
from bots import Base, _raise_to
from pokerlib import equity, preflop_strength


class _Rng:
    def __init__(self):
        self.rng = np.random.default_rng(random.randrange(2 ** 32))

    def eq(self, s, n=120):
        return equity(s['hole'], s['board'], n=n, rng=self.rng)

    def pre(self, s):
        return float(preflop_strength()[s['hole'][0], s['hole'][1]])


def _need(s):
    return s['to_call'] / (s['pot'] + s['to_call']) if s['to_call'] else 0.0


class TightPassive(_Rng, Base):
    def act(self, s):
        tc = s['to_call']
        if s['street'] == 0:
            st = self.pre(s)
            if st > 0.70: return _raise_to(s, 0.9) if tc <= 2 else 'c'
            if tc == 0: return 'c'
            return 'c' if (st > 0.58 and tc <= 4) else 'f'
        e = self.eq(s)
        if e > 0.82 and self.rng.random() < 0.5: return _raise_to(s, 0.5)
        if tc == 0: return 'c'
        return 'c' if e > _need(s) + 0.10 else 'f'


class LAG(_Rng, Base):
    def act(self, s):
        tc = s['to_call']
        e = self.pre(s) if s['street'] == 0 else self.eq(s)
        r = self.rng.random()
        if tc == 0:
            return _raise_to(s, 0.6) if (e > 0.45 and r < 0.8) or r < 0.30 else 'c'
        if e > 0.72: return _raise_to(s, 0.8)
        return 'c' if e > _need(s) - 0.02 else 'f'


class ProberV(_Rng, Base):
    def __init__(self, size=0.5, freq=0.85):
        super().__init__(); self.size, self.freq = size, freq

    def act(self, s):
        tc = s['to_call']
        if s['street'] == 0:
            st = self.pre(s)
            if tc <= 2: return _raise_to(s, 0.9) if st > 0.40 else 'c'
            return 'c' if st > 0.58 else 'f'
        if tc == 0:
            return _raise_to(s, self.size) if self.rng.random() < self.freq else 'c'
        e = self.eq(s)
        if e > 0.82: return _raise_to(s, 0.8)
        return 'c' if e > 0.55 else 'f'


class PartialJammer(_Rng, Base):
    def __init__(self, p=0.3):
        super().__init__(); self.p = p; self.jam = False

    def new_hand(self, is_button): self.jam = self.rng.random() < self.p

    def act(self, s):
        if self.jam and s['street'] == 0 and s['max_raise_to'] > s['opp_bet']:
            return ('r', s['max_raise_to'])
        return 'c'


class Flipper(_Rng, Base):
    def __init__(self, switch_at=100):
        super().__init__(); self.n = 0; self.switch_at = switch_at
        self.a, self.b = TightPassive.__new__(TightPassive), LAG.__new__(LAG)
        self.a.rng = self.b.rng = self.rng

    def new_hand(self, is_button): self.n += 1

    def act(self, s):
        return (self.a if self.n <= self.switch_at else self.b).act(s)


class DelayedExploiter(_Rng, Base):
    def __init__(self, observe_n=40):
        super().__init__(); self.faced = 0; self.folded = 0; self.observe_n = observe_n

    def observe(self, actor_is_me, kind, state):
        if not actor_is_me and state['to_call'] > 0:      # Hero acted while facing our bet
            self.faced += 1; self.folded += kind == 'f'

    @property
    def stab(self):
        return self.faced >= self.observe_n and self.folded / self.faced > 0.45

    def act(self, s):
        tc = s['to_call']
        if s['street'] == 0:
            st = self.pre(s)
            if tc <= 2: return _raise_to(s, 0.9) if st > (0.35 if self.stab else 0.55) else 'c'
            return 'c' if st > 0.58 else 'f'
        e = self.eq(s)
        if tc == 0:
            if self.stab or e > 0.65: return _raise_to(s, 0.6)
            return 'c'
        return 'c' if e > _need(s) + 0.05 else 'f'


class SizeReader(_Rng, Base):
    """Adaptive: P(we lose | we called a bet of this size bucket) learned from showdowns."""
    use_sizes = True

    def __init__(self):
        super().__init__()
        self.stat = {b: [2.0, 4.0] for b in range(3)}     # [losses, calls] with a 50% prior of weight 4
        self.pending = None

    @staticmethod
    def bucket(s):
        f = s['to_call'] / max(s['pot'] - s['to_call'], 1)
        return 0 if f < 0.45 else (1 if f < 0.72 else 2)

    def new_hand(self, is_button): self.pending = None

    def act(self, s):
        tc = s['to_call']
        if s['street'] == 0:
            st = self.pre(s)
            if tc <= 2: return _raise_to(s, 0.8) if st > 0.5 else 'c'
            return 'c' if st > 0.55 else 'f'
        e = self.eq(s)
        if tc == 0:
            return _raise_to(s, 0.5) if e > 0.55 else 'c'
        b = self.bucket(s)
        loss = self.stat[b][0] / self.stat[b][1] if self.use_sizes else 0.5
        thr = _need(s) + (0.12 if loss > 0.62 else (-0.08 if loss < 0.40 else 0.0))
        call = e > thr
        if call: self.pending = b
        return 'c' if call else 'f'

    def end_hand(self, info):
        if self.pending is not None and info['opp_hole'] is not None and self.use_sizes:
            st = self.stat[self.pending]; st[1] += 1; st[0] += info['chips'] < 0
        self.pending = None


class FineSizeReader(SizeReader):
    """Like SizeReader but with a bucket boundary at 0.625 pot, i.e. between Hero's 0.60 'bluff'
    and 0.65 'value' bet sizes (found by the leakage audit)."""
    def __init__(self):
        super().__init__(); self.stat = {b: [2.0, 4.0] for b in range(4)}

    @staticmethod
    def bucket(s):
        f = s['to_call'] / max(s['pot'] - s['to_call'], 1)
        return 0 if f < 0.45 else (1 if f < 0.625 else (2 if f < 0.72 else 3))


class SizeBlind(SizeReader):
    use_sizes = False


OPPS2 = {
    'TightPassive': TightPassive, 'LAG': LAG,
    'Prober33': lambda: ProberV(0.33, 0.85), 'Prober75': lambda: ProberV(0.75, 0.85),
    'Prober100': lambda: ProberV(1.0, 0.85), 'ProberHalfFreq': lambda: ProberV(0.5, 0.5),
    'PartialJammer30': lambda: PartialJammer(0.3), 'Flipper': Flipper,
    'DelayedExploiter': DelayedExploiter, 'SizeReader': SizeReader, 'FineSizeReader': FineSizeReader, 'SizeBlind': SizeBlind,
}
