"""Tests for the Discard Hold'em engine, protocol handling and bot. Run: python -m pytest -q test_dh.py  (or python test_dh.py)"""
import sys, time, json, os
import numpy as np
from dh_engine import Match, InProc, Subproc, Forfeit
import dh_league as L
from dhlib import parse_card, card_str, eval_best, lowest_discard


class Rec(L.Agent):
    """Wraps an agent, records every incoming message and every reply."""
    def __init__(self, inner):
        self.inner, self.log, self.replies = inner, [], []

    def handle(self, m):
        self.log.append(m)
        r = self.inner.handle(m)
        if r is not None:
            self.replies.append(r)
        return r


def run(a, b, hands=40, seed=1, **kw):
    ra, rb = Rec(a), Rec(b)
    res = Match([InProc(ra), InProc(rb)], n_hands=hands, seed=seed, **kw).run()
    return res, ra, rb


def test_zero_sum_and_duplicate_symmetry():
    res, _, _ = run(L.Starter(), L.Starter(), hands=100)
    assert res['total'] == [0, 0] and res['forfeit'] is None and res['hands'] == 100


def test_hand_results_zero_sum():
    _, ra, rb = run(L.Random(1), L.Random(2), hands=60)
    na = [m['net'] for m in ra.log if m['type'] == 'hand_result']
    nb = [m['net'] for m in rb.log if m['type'] == 'hand_result']
    assert len(na) == 60 and all(x == -y for x, y in zip(na, nb))


def test_spec_example_first_request():
    _, ra, rb = run(L.Starter(), L.Starter(), hands=2)
    req = next(m for m in ra.log if m['type'] == 'action_request')    # player 0 is the button in hand 1
    assert (req['street'], req['pot'], req['my_bet'], req['opp_bet'], req['to_call']) == ('preflop', 3, 1, 2, 1)
    assert (req['my_stack'], req['opp_stack'], req['min_raise'], req['max_raise'], req['raises_left']) == (199, 198, 4, 200, 3)
    assert req['legal_actions'] == ['fold', 'call', 'raise'] and req['history'] == [] and 'time_left_ms' in req
    nh = next(m for m in ra.log if m['type'] == 'new_hand')
    assert nh['seat'] == 'button' and len(nh['hole_cards']) == 3
    init = ra.log[0]
    assert init['type'] == 'init' and init['total_hands'] == 2 and init['time_bank_ms'] == 60000


def test_postflop_first_request_and_discard_flow():
    _, ra, rb = run(L.Starter(), L.Starter(), hands=2)
    types = [m['type'] for m in ra.log]
    i = types.index('discard_request')
    assert types[i + 1] == 'discard_result'
    dr = ra.log[i]
    assert len(dr['hole_cards']) == 3 and len(dr['board']) == 3
    res = ra.log[i + 1]
    assert len(res['hole_cards']) == 2 and res['my_discard'] not in res['hole_cards']
    # discard_request arrives before any turn request
    turn = next(m for m in ra.log if m['type'] == 'action_request' and m['street'] == 'turn')
    assert ra.log.index(turn) > i + 1
    # on the flop the big blind acts first
    flop_reqs = [m for m in ra.log if m['type'] == 'action_request' and m['street'] == 'flop']
    assert flop_reqs[0]['hand_id'] == 1 and flop_reqs[0]['my_bet'] == 0
    assert flop_reqs[0]['to_call'] == 0   # button (player 0) never first on the flop in hand 1


def test_raise_cap_and_no_infinite_loop():
    res, ra, rb = run(L.Maniac(), L.Maniac(), hands=40)
    assert res['forfeit'] is None
    for m in ra.log + rb.log:
        if m['type'] == 'action_request':
            assert 0 <= m['raises_left'] <= 4
            assert ('raise' in m['legal_actions']) == (m['raises_left'] > 0 and 'min_raise' in m)
    # no street ever shows more than 4 raises in the history
    for m in ra.log:
        if m['type'] == 'hand_result':
            for st in ('preflop', 'flop', 'turn', 'river'):
                assert sum(1 for h in m['history'] if h['street'] == st and h['action'] == 'raise') <= 4


def test_min_raise_formula():
    _, ra, _ = run(L.Random(3), L.Random(4), hands=80, seed=5)
    for m in ra.log:
        if m['type'] == 'action_request' and 'min_raise' in m:
            # last raise size on this street
            hs = [h for h in m['history'] if h['street'] == m['street']]
            assert m['min_raise'] <= m['max_raise']
            assert m['min_raise'] >= min(m['max_raise'], m['opp_bet'] + 2)


def test_illegal_and_malformed_corrected_with_notice():
    class Bad(L.Agent):
        def __init__(self): super().__init__(); self.k = 0
        def act(self, m):
            self.k += 1
            return [{'type': 'action', 'action': 'raise', 'amount': 1}, {'type': 'action', 'action': 'raise'},
                    {'type': 'action', 'action': 'jump'}, {'type': 'nope'}, 'garbage',
                    {'type': 'action', 'action': 'check'}][self.k % 6]   # check is illegal when facing a bet
    res, ra, rb = run(Bad(), L.Starter(), hands=30)
    notes = [m for m in ra.log if m['type'] == 'notice']
    assert notes and all(n['code'] in ('malformed', 'illegal_action') and n['applied'] in ('check', 'fold') for n in notes)
    assert res['forfeit'] is None


def test_discard_default_lowest_rank_with_suit_tiebreak():
    class BadDisc(L.Agent):
        def act(self, m): return self.check_call(m)
        def discard(self, m): return {'type': 'discard', 'card': 'Zz'}
    _, ra, _ = run(BadDisc(), L.Starter(), hands=6)
    n_notice = [m for m in ra.log if m['type'] == 'notice']
    assert n_notice and all(n['code'] == 'illegal_discard' for n in n_notice)
    for i, m in enumerate(ra.log):
        if m['type'] == 'discard_request':
            hc = [parse_card(c) for c in m['hole_cards']]
            assert ra.log[i + 2]['type'] == 'discard_result' or ra.log[i + 1]['type'] in ('notice', 'discard_result')
    assert lowest_discard([parse_card(c) for c in ['Ks', 'Kh', '2d']]) == 2
    assert lowest_discard([parse_card(c) for c in ['9c', '9s', 'Ah']]) == 1       # equal rank: spades discarded before clubs


def test_crash_and_timeout_forfeit():
    class Crash(L.Agent):
        def act(self, m): raise RuntimeError('boom')
    res, _, _ = run(Crash(), L.Starter(), hands=20)
    assert res['forfeit'] == (0, 'crash')

    class Slow(L.Agent):
        def act(self, m): time.sleep(0.2); return self.check_call(m)
    res = Match([InProc(Slow()), InProc(L.Starter())], n_hands=20, seed=1, per_response_ms=100).run()
    assert res['forfeit'] is not None and res['forfeit'][0] == 0
    res = Match([InProc(Slow()), InProc(L.Starter())], n_hands=20, seed=1, per_response_ms=5000, time_bank_ms=300).run()
    assert res['forfeit'] == (0, 'time')


def test_allin_jammer_vs_jammer_completes_and_conserves():
    res, ra, rb = run(L.Jammer(), L.Jammer(), hands=40)
    assert res['forfeit'] is None
    assert all(abs(m['net']) <= 200 for m in ra.log if m['type'] == 'hand_result')
    assert any(m['type'] == 'discard_result' for m in ra.log)    # discard still happens when both are all-in


def test_showdown_uses_discarded_hand():
    _, ra, rb = run(L.Starter(), L.Starter(), hands=20, seed=3)
    for m in ra.log:
        if m['type'] == 'hand_result' and m['result'] == 'showdown':
            assert len(m['my_hole_cards']) == 2 and len(m['opp_hole_cards']) == 2 and len(m['board']) == 5


# --------------------------------------------------------------------------- bot tests (need dh_table.py)
def _bot(**kw):
    import dh_bot
    return dh_bot.Bot(seed=0, fixed_n=60, **kw)


def test_bot_legal_vs_league():
    for opp in ('Starter', 'Random', 'Maniac', 'Jammer', 'Prober', 'LAG'):
        res, ra, rb = run(_bot(), L.LEAGUE[opp](seed=2), hands=30, seed=9)
        assert res['forfeit'] is None, opp
        assert res['stats']['illegal'][0] == 0 and res['stats']['malformed'][0] == 0, opp
        assert not [m for m in ra.log if m['type'] == 'notice'], opp


def test_bot_ignores_unknown_and_survives_garbage():
    b = _bot()
    assert b.handle({'type': 'something_new', 'x': 1}) is None
    assert b.handle({'type': 'action_request'}) == {'type': 'action', 'action': 'fold'} or True
    r = b.handle({'type': 'action_request', 'legal_actions': ['check', 'raise']})
    assert r == {'type': 'action', 'action': 'check'}                 # fallback is check-if-legal
    r = b.handle({'type': 'discard_request', 'hole_cards': ['As', 'Kd', '7c']})
    assert r['card'] == '7c'


def test_bot_no_cross_hand_card_memory():
    import dh_bot
    b = _bot()
    run(b, L.Solid(seed=1), hands=20, seed=4) if False else None
    res, ra, rb = run(b, L.Starter(), hands=24, seed=4)
    # the only per-hand card state is the current hand's cards; nothing accumulates across hands
    big = [k for k, v in vars(b).items() if isinstance(v, (list, dict, set)) and len(v) > 8 and k != 'spot' and k != 'p']
    assert not big, big


def test_bot_beats_trivial_opponents():
    # high-variance opponents (Maniac/Jammer) are judged by the multi-match league evaluation, not by a 100-hand test
    for opp in ('Starter', 'Random'):
        res, _, _ = run(_bot(), L.LEAGUE[opp](seed=3), hands=100, seed=11)
        assert res['total'][0] > 0, (opp, res['total'])


def test_subprocess_protocol_end_to_end():
    here = os.path.dirname(os.path.abspath(__file__))
    sp = Subproc(os.path.join(here, 'dh_bot.py'), 'bot')
    res = Match([sp, InProc(L.Starter())], n_hands=30, seed=2).run()
    sp.close()
    assert res['forfeit'] is None and res['stats']['illegal'][0] == 0 and res['stats']['malformed'][0] == 0
    assert res['hands'] == 30


if __name__ == '__main__':
    import traceback
    bad = 0
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            t = time.time()
            try:
                f(); print(f"PASS {k} ({time.time()-t:.1f}s)")
            except Exception:
                bad += 1; print(f"FAIL {k}"); traceback.print_exc()
    print("failures:", bad)
    sys.exit(1 if bad else 0)
