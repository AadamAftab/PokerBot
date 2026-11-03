"""Adversarial opponents, each written to attack one specific assumption of Hero/Champ.

Prober        - bets half pot whenever checked to (attacks "a check means weak" + over-folding)
Jammer        - all-in every hand (attacks the fixed range-narrowing fraction q)
Reverser      - bets junk, slow-plays strong hands (attacks "a bet means strong")
Switcher      - folds to everything for 60 hands, then becomes a calling station
                (attacks cumulative, never-decaying fold-rate estimate -> over-bluffing)
PreflopFolder - folds half its small blinds, otherwise a calling station
                (attacks fold-rate pollution: preflop folds counted as postflop folds)
"""
import random
from bots import Base, _raise_to
from pokerlib import equity


class Prober(Base):
    def act(self, s):
        tc = s['to_call']
        if s['street'] == 0:
            e = equity(s['hole'], [], n=120)
            if tc <= 2: return _raise_to(s, 0.9) if e > 0.40 else 'c'
            return 'c' if e > 0.58 else 'f'
        if tc == 0:
            return _raise_to(s, 0.5) if random.random() < 0.85 else 'c'
        e = equity(s['hole'], s['board'], n=120)
        if e > 0.82: return _raise_to(s, 0.8)
        return 'c' if e > 0.55 else 'f'


class Jammer(Base):
    def act(self, s):
        return ('r', s['max_raise_to']) if s['max_raise_to'] > s['opp_bet'] else 'c'


class Reverser(Base):
    def act(self, s):
        e = equity(s['hole'], s['board'], n=120)
        tc, pot = s['to_call'], s['pot']
        if tc == 0:
            return _raise_to(s, 0.75) if e < 0.42 else 'c'      # bluff junk, trap strength
        need = tc / (pot + tc)
        return 'c' if e > need + 0.02 else 'f'


class Switcher(Base):
    def __init__(self): self.n = 0
    def new_hand(self, is_button): self.n += 1
    def act(self, s):
        if self.n <= 60 and s['to_call'] > 0: return 'f'
        return 'c'


class PreflopFolder(Base):
    def act(self, s):
        if s['street'] == 0 and s['is_button'] and s['to_call'] > 0 and random.random() < 0.5:
            return 'f'
        return 'c'
