"""Real-clock check: the bot as a subprocess (true pipes), governor ON, official provisional limits.
usage: python time_check.py [bot_path] [hands=400] [bank_ms=60000] [opponents,comma,sep]
Reports forfeits, bank used, worst single reply, time-to-ready and illegal/malformed counts."""
import os, sys, time
import numpy as np
from dh_engine import Match, InProc, Subproc
import dh_league as L

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dh_bot.py')
hands = int(sys.argv[2]) if len(sys.argv) > 2 else 400
bank = int(sys.argv[3]) if len(sys.argv) > 3 else 60000
opps = sys.argv[4].split(',') if len(sys.argv) > 4 else ['Starter', 'Maniac', 'Jammer', 'Solid']
print(f"bot={os.path.basename(path)} hands={hands} bank={bank}ms")
for o in opps:
    sp = Subproc(path, 'bot')
    t0 = time.perf_counter()
    m = Match([sp, InProc(L.LEAGUE[o](seed=1))], n_hands=hands, seed=21, time_bank_ms=bank)
    r = m.run()
    wall = time.perf_counter() - t0
    sp.close()
    lat = np.array(r['stats']['lat'][0])
    print(f"{o:10} forfeit={r['forfeit']} hands={r['hands']} bank_left={r['bank'][0]/1000:.1f}s used={(bank-r['bank'][0])/1000:.1f}s "
          f"p50={np.percentile(lat,50):.0f}ms p99={np.percentile(lat,99):.0f}ms max={lat.max():.0f}ms first={lat[0]:.0f}ms "
          f"illegal={r['stats']['illegal'][0]} malformed={r['stats']['malformed'][0]} net={r['total'][0]:+d} wall={wall:.0f}s")
