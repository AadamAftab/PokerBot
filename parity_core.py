"""Deterministic pseudo-random bot used to check that my engine and the organisers' engine agree hand by hand.
Decisions depend only on the request (and a salt), so the same deals give the same play in either engine.
~6% of replies are deliberately illegal/malformed to check that corrections agree too."""
import sys, json, zlib

RANKS = "23456789TJQKA"


def h(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def decide(salt, m):
    t = m.get('type')
    if t == 'action_request':
        key = h(salt, m['hand_id'], m['street'], len(m['history']), m['pot'])
        r = key % 100
        legal = m['legal_actions']
        if r < 2:
            return {'type': 'action', 'action': 'raise', 'amount': 1}                      # illegal amount
        if r < 4:
            return {'type': 'action', 'action': 'check' if 'check' not in legal else 'call'}  # illegal when it applies
        if r < 6:
            return {'type': 'action'}                                                       # malformed
        if 'raise' in legal and r < 45:
            lo, hi = m['min_raise'], m['max_raise']
            span = hi - lo
            amt = lo + (h(salt, key, 'amt') % (span + 1)) * (1 if r % 3 else 0) if r % 4 else hi
            if r % 5 == 0:
                amt = lo
            return {'type': 'action', 'action': 'raise', 'amount': int(max(lo, min(hi, amt)))}
        if 'fold' in legal and r > 85:
            return {'type': 'action', 'action': 'fold'}
        return {'type': 'action', 'action': 'check' if 'check' in legal else 'call'}
    if t == 'discard_request':
        hc = m['hole_cards']
        r = h(salt, m['hand_id'], 'disc') % 100
        if r < 5:
            return {'type': 'discard', 'card': 'Zz'}
        if r < 8:
            return {'type': 'discard'}
        return {'type': 'discard', 'card': hc[h(salt, m['hand_id'], 'd2') % 3]}
    return None


def run(salt):
    for line in sys.stdin:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        if m.get('type') == 'match_end':
            break
        r = decide(salt, m)
        if r is not None:
            print(json.dumps(r), flush=True)
