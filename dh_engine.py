"""Discard Hold'em match engine following the Protocol Specification v1.0 (see doc notes in DH_SPEC_DIGEST.md).

Agents speak the real JSON message protocol. Two adapters:
  InProc(bot)       - bot object with .handle(msg_dict) -> reply_dict | None   (fast, for experiments)
  Subproc(path)     - runs `python bot.py`, JSON lines over pipes               (protocol-faithful)
The engine enforces legality, corrections, time bank (60 s / 400 hands), 5 s per reply and forfeits.

Deals are duplicated: hand 2k+1 and 2k+2 use the same (button cards, big-blind cards, board) with the two
players swapped, since seats alternate. (Whether the real engine does this, and the order, is UNCONFIRMED.)
"""
import json, queue, subprocess, sys, threading, time
import numpy as np
from dhlib import parse_card, card_str, eval_best

SB, BB, STACK = 1, 2, 200
STREETS = ['preflop', 'flop', 'turn', 'river']
NBOARD = [0, 3, 4, 5]


# ----------------------------------------------------------------------------- adapters
class InProc:
    def __init__(self, bot, name=None):
        self.bot, self.name = bot, name or type(bot).__name__
        self.alive = True

    def send(self, msg):
        try:
            self.bot.handle(msg)
        except Exception:
            self.alive = False

    def ask(self, msg, timeout):
        t0 = time.perf_counter()
        try:
            r = self.bot.handle(msg)
        except Exception:
            self.alive = False
            return None, (time.perf_counter() - t0) * 1000, 'crash'
        return r, (time.perf_counter() - t0) * 1000, 'ok'

    def close(self): pass


class Subproc:
    def __init__(self, path, name=None, python=None):
        self.name = name or path
        self.p = subprocess.Popen([python or sys.executable, path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, bufsize=0)
        self.q = queue.Queue()
        threading.Thread(target=self._rd, daemon=True).start()
        self.alive = True

    def _rd(self):
        for line in iter(self.p.stdout.readline, b''):
            self.q.put(line)
        self.q.put(None)

    def _w(self, msg):
        try:
            self.p.stdin.write((json.dumps(msg) + '\n').encode())
            self.p.stdin.flush()
            return True
        except Exception:
            self.alive = False
            return False

    def send(self, msg): self._w(msg)

    def ask(self, msg, timeout):
        t0 = time.perf_counter()
        if not self._w(msg):
            return None, 0.0, 'crash'
        try:
            line = self.q.get(timeout=timeout / 1000.0)
        except queue.Empty:
            return None, (time.perf_counter() - t0) * 1000, 'timeout'
        el = (time.perf_counter() - t0) * 1000
        if line is None:
            self.alive = False
            return None, el, 'crash'
        try:
            return json.loads(line), el, 'ok'
        except ValueError:
            return {'__malformed__': True}, el, 'ok'

    def close(self):
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


# ----------------------------------------------------------------------------- engine
class Forfeit(Exception):
    def __init__(self, who, why):
        self.who, self.why = who, why


class Match:
    def __init__(self, agents, n_hands=400, seed=0, time_bank_ms=60000, per_response_ms=5000, dup=True):
        self.ag = agents
        self.n_hands = n_hands
        self.rng = np.random.default_rng(seed)
        self.bank = [float(time_bank_ms)] * 2
        self.per = per_response_ms
        self.tb = time_bank_ms
        self.dup = dup
        self.stats = dict(illegal=[0, 0], malformed=[0, 0], timeouts=[0, 0], lat=[[], []], hands=0)

    # --- low level ---
    def _ask(self, p, msg):
        msg['time_left_ms'] = int(self.bank[p])
        reply, el, st = self.ag[p].ask(msg, min(self.per, self.bank[p]) + 50)
        self.stats['lat'][p].append(el)
        if st != 'ok' or reply is None:
            self.stats['timeouts'][p] += 1
            raise Forfeit(p, st if st != 'ok' else 'eof')
        self.bank[p] -= el
        if el > self.per or self.bank[p] < 0:
            raise Forfeit(p, 'time')
        return reply

    def _notice(self, p, code, applied):
        self.ag[p].send({'type': 'notice', 'code': code, 'applied': applied})

    def _deal(self):
        d = self.rng.permutation(52)
        return [int(x) for x in d[:3]], [int(x) for x in d[3:6]], [int(x) for x in d[6:11]]

    # --- one hand ---
    def hand(self, hid, btn, hole_by_player, board5):
        bb = 1 - btn
        stacks = [STACK, STACK]
        bets = [0, 0]
        stacks[btn] -= SB; bets[btn] = SB
        stacks[bb] -= BB; bets[bb] = BB
        hist = []                                  # (street, player, action, amount)
        holes = [list(h) for h in hole_by_player]
        disc = [None, None]
        for p in (0, 1):
            seat = 'button' if p == btn else 'big_blind'
            self.ag[p].send({'type': 'new_hand', 'hand_id': hid, 'seat': seat, 'hole_cards': [card_str(c) for c in holes[p]],
                             'stack': STACK, 'opp_stack': STACK})
        folded = None
        board_vis = []

        def hview(p):
            return [{'street': s, 'actor': 'me' if a == p else 'opp', 'action': k, **({'amount': m} if k in ('call', 'raise') else {})}
                    for s, a, k, m in hist]

        def pot(): return 2 * STACK - stacks[0] - stacks[1]

        for si, street in enumerate(STREETS):
            if si > 0:
                bets = [0, 0]
            board_vis = board5[:NBOARD[si]]
            raises_left = 3 if si == 0 else 4
            last_inc = BB if si == 0 else 0
            acted = [False, False]
            cur = btn if si == 0 else bb
            while True:
                p, o = cur, 1 - cur
                if stacks[p] == 0 or (stacks[o] == 0 and bets[p] >= bets[o]):
                    break
                if acted[0] and acted[1] and bets[0] == bets[1]:
                    break
                to_call = bets[o] - bets[p]
                max_to = min(bets[p] + stacks[p], bets[o] + stacks[o])
                can_raise = raises_left > 0 and max_to > bets[o]
                usual = bets[o] + max(last_inc, 2)
                min_to = min(usual, max_to)
                legal = (['fold', 'call'] if to_call > 0 else ['check'])
                if can_raise:
                    legal.append('raise')
                req = {'type': 'action_request', 'hand_id': hid, 'street': street, 'board': [card_str(c) for c in board_vis],
                       'pot': pot(), 'my_bet': bets[p], 'opp_bet': bets[o], 'to_call': to_call, 'my_stack': stacks[p],
                       'opp_stack': stacks[o], 'legal_actions': legal, 'raises_left': raises_left, 'history': hview(p)}
                if can_raise:
                    req['min_raise'], req['max_raise'] = min_to, max_to
                r = self._ask(p, req)
                act, amt = self._parse(p, r, legal, can_raise, min_to, max_to, to_call)
                if act == 'fold':
                    hist.append((street, p, 'fold', 0)); folded = p; break
                if act == 'raise':
                    stacks[p] -= amt - bets[p]
                    last_inc = amt - bets[o]
                    bets[p] = amt
                    raises_left -= 1
                    acted[o] = False
                    hist.append((street, p, 'raise', amt))
                elif act == 'call':
                    pay = min(to_call, stacks[p])
                    stacks[p] -= pay; bets[p] += pay
                    hist.append((street, p, 'call', bets[p]))
                else:
                    hist.append((street, p, 'check', 0))
                acted[p] = True
                cur = o
            if folded is not None:
                break
            # refund any uncalled excess
            if bets[0] != bets[1]:
                big = 0 if bets[0] > bets[1] else 1
                ex = bets[big] - bets[1 - big]
                stacks[big] += ex; bets[big] -= ex
            if si == 1:                                           # discard stage (even if both all-in)
                self._discard(hid, holes, board_vis, stacks, pot, disc)
        invest = [STACK - stacks[0], STACK - stacks[1]]
        showdown = folded is None
        if showdown:
            sc = [int(eval_best(np.array([holes[p] + board5]))[0]) for p in (0, 1)]
            net0 = invest[1] if sc[0] > sc[1] else (-invest[0] if sc[0] < sc[1] else 0)
        else:
            net0 = invest[1] if folded == 1 else -invest[0]
        net = [net0, -net0]
        for p in (0, 1):
            self.ag[p].send({'type': 'hand_result', 'hand_id': hid, 'result': 'showdown' if showdown else 'fold', 'net': net[p],
                             'board': [card_str(c) for c in (board5 if showdown else board_vis)],
                             'my_hole_cards': [card_str(c) for c in holes[p]],
                             'opp_hole_cards': [card_str(c) for c in holes[1 - p]] if showdown else None,
                             'opp_discard': card_str(disc[1 - p]) if disc[1 - p] is not None else None,
                             'history': hview(p)})
        self.stats['hands'] += 1
        return net

    def _parse(self, p, r, legal, can_raise, min_to, max_to, to_call):
        fb = 'check' if 'check' in legal else 'fold'
        if not isinstance(r, dict) or r.get('type') != 'action' or r.get('__malformed__'):
            self.stats['malformed'][p] += 1
            self._notice(p, 'malformed', fb)
            return fb, 0
        a = r.get('action')
        if a not in legal:
            self.stats['illegal'][p] += 1
            self._notice(p, 'illegal_action', fb)
            return fb, 0
        if a == 'raise':
            amt = r.get('amount')
            if isinstance(amt, bool) or not isinstance(amt, int) or not (min_to <= amt <= max_to):
                self.stats['illegal'][p] += 1
                self._notice(p, 'illegal_action', fb)
                return fb, 0
            return 'raise', amt
        return a, 0

    def _discard(self, hid, holes, flop, stacks, pot, disc):
        replies = []
        for p in (0, 1):
            req = {'type': 'discard_request', 'hand_id': hid, 'hole_cards': [card_str(c) for c in holes[p]],
                   'board': [card_str(c) for c in flop], 'pot': pot(), 'my_stack': stacks[p], 'opp_stack': stacks[1 - p]}
            replies.append(self._ask(p, req))
        for p in (0, 1):
            r = replies[p]
            card = None
            if isinstance(r, dict) and r.get('type') == 'discard' and isinstance(r.get('card'), str):
                try:
                    c = parse_card(r['card'])
                    if c in holes[p]:
                        card = c
                except (ValueError, IndexError):
                    pass
            if card is None:
                self.stats['illegal'][p] += 1
                card = min(holes[p], key=lambda c: (c >> 2, c & 3))
                self._notice(p, 'illegal_discard', card_str(card))
            disc[p] = card
        for p in (0, 1):
            holes[p] = [c for c in holes[p] if c != disc[p]]
        for p in (0, 1):
            self.ag[p].send({'type': 'discard_result', 'hand_id': hid, 'my_discard': card_str(disc[p]),
                             'opp_discard': card_str(disc[1 - p]), 'hole_cards': [card_str(c) for c in holes[p]]})

    # --- match ---
    def run(self):
        for p in (0, 1):
            self.ag[p].send({'type': 'init', 'protocol_version': '1.0', 'total_hands': self.n_hands, 'starting_stack': STACK,
                             'small_blind': SB, 'big_blind': BB, 'time_bank_ms': self.tb, 'per_response_ms': self.per})
        total = [0, 0]
        forfeit = None
        try:
            hid = 0
            while hid < self.n_hands:
                b_cards, bb_cards, board = self._deal()
                for rep in (0, 1) if self.dup else (0,):
                    if hid >= self.n_hands:
                        break
                    hid += 1
                    btn = 0 if (hid % 2 == 1) else 1
                    hole = [None, None]
                    hole[btn], hole[1 - btn] = list(b_cards), list(bb_cards)
                    net = self.hand(hid, btn, hole, board)
                    total[0] += net[0]; total[1] += net[1]
                    if not self.dup:
                        break
        except Forfeit as f:
            forfeit = f
        reason = 'complete'
        for p in (0, 1):
            self.ag[p].send({'type': 'match_end', 'reason': 'complete' if forfeit is None else ('forfeit' if forfeit.who == p else 'opp_forfeit')})
        return dict(total=total, forfeit=None if forfeit is None else (forfeit.who, forfeit.why), stats=self.stats,
                    hands=self.stats['hands'], bank=self.bank)
