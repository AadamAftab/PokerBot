"""Candidate patches, each an opt-in subclass; the baseline files stay untouched.

SpotQ: replace the fixed range-width rule (q = f(bet size) * global-aggression) with the opponent's
OBSERVED aggression frequency in the same kind of spot.  If a player bets 85% of the time they are
checked to, a bet from them tells us almost nothing (their range is ~everything); if they only bet
8% of the time it means a lot.  The old rule assumed "a bet means roughly the top 30-60%" regardless.
"""
from champ import Champ

PRIORS = {'lead': 0.40, 'vs_bet': 0.20, 'pre_open': 0.50, 'pre_3bet': 0.15}   # prior aggression freq
N0 = 8.0                                                                      # prior weight (opportunities)


class SpotQChamp(Champ):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.spot = {k: [p * N0, N0] for k, p in PRIORS.items()}   # [aggressive actions, opportunities]

    @staticmethod
    def _opp_spot(state):
        """Spot of an action taken by the opponent; `state` is from the opponent's own perspective."""
        if state['street'] == 0:
            return 'pre_open' if state['opp_bet'] <= 2 else 'pre_3bet'
        return 'lead' if state['to_call'] == 0 else 'vs_bet'

    def observe(self, actor_is_me, kind, state):
        super().observe(actor_is_me, kind, state)
        if not actor_is_me:
            c = self.spot[self._opp_spot(state)]
            c[1] += 1
            c[0] += kind == 'r'

    def _range_q(self, s):
        # spot of the aggressive action we are now facing, from our own perspective
        if s['street'] == 0:
            spot = 'pre_open' if s['my_bet'] <= 2 else 'pre_3bet'
        else:
            spot = 'lead' if s['my_bet'] == 0 else 'vs_bet'
        k, n = self.spot[spot]
        frac = s['to_call'] / max(s['pot'] - s['to_call'], 1)
        size_adj = min(1.2, max(0.6, (0.6 / max(frac, 0.05)) ** 0.3))    # bigger bet -> somewhat narrower
        return min(1.0, max(0.08, (k / n) * size_adj))


from champ import Champ as _C

PATCH_ARMS = {
    'replay_loose': lambda sd, ms: _C(seed=sd, replay=True, risk=False, max_sims=ms, strict_lag=False),
    'replay_strict': lambda sd, ms: _C(seed=sd, replay=True, risk=False, max_sims=ms, strict_lag=True),
    'spotq': lambda sd, ms: SpotQChamp(seed=sd, replay=False, risk=False, max_sims=ms),
}
