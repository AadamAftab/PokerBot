"""Regression tests. Run: python test_regression.py   (or pytest)."""
import json, random
import numpy as np
import bots, league, engine
from champ import Champ
from patches import SpotQChamp
from pokerlib import eval_best


def c(s):
    return "23456789TJQKA".index(s[0]) * 4 + "cdhs".index(s[1])


def h(*s):
    return np.array([[c(x) for x in s]])


def test_evaluator_category_order_and_kickers():
    order = [h("2c", "3d", "5h", "7s", "9c", "Jd", "Kh"), h("2c", "2d", "5h", "7s", "9c", "Jd", "Kh"),
             h("2c", "2d", "5h", "5s", "9c", "Jd", "Kh"), h("2c", "2d", "2h", "7s", "9c", "Jd", "Kh"),
             h("Ac", "2d", "3h", "4s", "5c", "Jd", "Kh"), h("9c", "Td", "Jh", "Qs", "Kc", "2d", "3h"),
             h("2c", "4c", "6c", "8c", "Tc", "Jd", "Kh"), h("2c", "2d", "2h", "7s", "7c", "Jd", "Kh"),
             h("2c", "2d", "2h", "2s", "9c", "Jd", "Kh"), h("5c", "6c", "7c", "8c", "9c", "Jd", "Kh")]
    sc = [int(eval_best(x)[0]) for x in order]
    assert sc == sorted(sc) and len(set(sc)) == 10
    assert eval_best(h("Kc", "Kd", "Qh", "Qs", "4c", "4d", "Ah"))[0] > eval_best(h("Kc", "Kd", "Qh", "Qs", "4c", "4d", "Jh"))[0]


def test_duplicate_of_identical_deterministic_bots_is_exactly_zero():
    r = engine.duplicate_match(bots.CallStation, bots.CallStation, 30, seed=3)
    assert all(x == 0 for x in r)


def test_engine_results_bounded_and_zero_sum_per_hand():
    rng = np.random.default_rng(0)
    for _ in range(60):
        cards = rng.permutation(52)
        a, b = bots.Maniac(), bots.RandomBot()
        res = engine.play_hand([a, b], [cards[:2], cards[2:4]], cards[4:9])
        assert abs(res) <= engine.STACK


def test_illegal_actions_are_sanitised_and_counted():
    class Bad(bots.Base):
        def act(self, s): return 'f'          # folding when a free check is available
    engine.STATS.update(actions=0, illegal=0, hero_illegal=0, log=[])
    engine.duplicate_match(Bad, bots.CallStation, 20, seed=1)
    assert engine.STATS['illegal'] > 0        # counted ...
    # ... and never crashes / never produces a bogus result


def test_baseline_behaviour_is_unchanged_by_patch_hooks():
    g = json.load(open('golden_baseline.json'))
    for name, opp in [('CallStation', bots.CallStation), ('Maniac', bots.Maniac)]:
        r = engine.duplicate_match(lambda: bots.Hero(seed=3, n_sims=200), opp, 25, seed=21)
        assert [float(x) for x in r] == g[name], name


def test_replay_logic_has_no_false_positives_on_independent_deals():
    rng = np.random.default_rng(5)
    bot = Champ(seed=1, replay=True, risk=False, max_sims=150)
    for _ in range(80):                         # every hand a fresh independent deal
        cards = rng.permutation(52)
        engine.play_hand([bot, bots.BasicEq()], [cards[:2], cards[2:4]], cards[4:9])
    assert bot.replay_acts == 0 and bot.errors == 0


def test_replay_logic_activates_on_real_duplicates_and_never_errors():
    bot = Champ(seed=1, replay=True, risk=False, max_sims=150)
    engine.duplicate_match(lambda: bot, bots.BasicEq, 40, seed=2)
    assert bot.replay_acts > 0 and bot.errors == 0


def test_spotq_statistics_converge_on_known_opponents():
    bot = SpotQChamp(seed=1, max_sims=150, replay=False, risk=False)
    engine.duplicate_match(lambda: bot, league.Jammer, 40, seed=4)
    k, n = bot.spot['pre_open']
    assert k / n > 0.8                          # Jammer always raises first-in
    bot = SpotQChamp(seed=1, max_sims=150, replay=False, risk=False)
    engine.duplicate_match(lambda: bot, bots.CallStation, 40, seed=4)
    assert bot.spot['lead'][0] / bot.spot['lead'][1] < 0.25


def test_spotq_no_exceptions_no_illegal_actions_across_league():
    for opp in (bots.RandomBot, bots.Maniac, league.Jammer, league.Prober, league.Reverser):
        engine.STATS.update(actions=0, illegal=0, hero_illegal=0, log=[])
        random.seed(0)
        bot = SpotQChamp(seed=2, max_sims=120, replay=False, risk=False)
        engine.duplicate_match(lambda: bot, opp, 25, seed=8)
        assert bot.errors == 0 and engine.STATS['hero_illegal'] == 0, opp.__name__


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for f in fns:
        f()
        print('PASS', f.__name__)
    print(f'{len(fns)} passed')
