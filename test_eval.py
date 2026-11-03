import numpy as np
from pokerlib import eval_best, equity
def c(s):
    r="23456789TJQKA".index(s[0]); su="cdhs".index(s[1]); return r*4+su
def h(*s): return np.array([[c(x) for x in s]])
order = [
 h("2c","3d","5h","7s","9c","Jd","Kh"),        # high card
 h("2c","2d","5h","7s","9c","Jd","Kh"),        # pair
 h("2c","2d","5h","5s","9c","Jd","Kh"),        # two pair
 h("2c","2d","2h","7s","9c","Jd","Kh"),        # trips
 h("Ac","2d","3h","4s","5c","Jd","Kh"),        # wheel
 h("9c","Td","Jh","Qs","Kc","2d","3h"),        # straight
 h("2c","4c","6c","8c","Tc","Jd","Kh"),        # flush
 h("2c","2d","2h","7s","7c","Jd","Kh"),        # full house
 h("2c","2d","2h","2s","9c","Jd","Kh"),        # quads
 h("5c","6c","7c","8c","9c","Jd","Kh"),        # straight flush
]
scores=[int(eval_best(x)[0]) for x in order]
print(scores); assert scores==sorted(scores) and len(set(scores))==10
# tie / kicker checks
assert eval_best(h("Ac","Kd","Qh","Js","9c","2d","3h"))[0] > eval_best(h("Ac","Kd","Qh","Js","8c","2d","3h"))[0]
assert eval_best(h("2c","2d","2h","2s","9c","Jd","Kh"))[0] < eval_best(h("3c","3d","3h","3s","9c","2d","4h"))[0]
# two pair w/ three pairs: kicker should be best remaining card (A beats the third pair 4s)
a=eval_best(h("Kc","Kd","Qh","Qs","4c","4d","Ah"))[0]; b=eval_best(h("Kc","Kd","Qh","Qs","4c","4d","Jh"))[0]; assert a>b
import time
t=time.time(); e=equity([c("Ac"),c("Ad")],[],n=2000); print("AA vs random", e, time.time()-t)
t=time.time(); e=equity([c("7c"),c("2d")],[],n=2000); print("72o vs random", e)
t=time.time(); e=equity([c("Ac"),c("Kc")],[c("2c"),c("7c"),c("Jd")],q=0.4,n=600); print("AKs flush draw vs top40%", e, time.time()-t)
