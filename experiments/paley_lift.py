"""paley_lift.py -- is SoS_4 == SoS_2 (= 1/2 + (1+sqrt p)/(2(p-1))) for Max-Cut on the Paley graph P_p?"""
import sys, json, math, numpy as np, torch
from circulant_sos import CirculantSoS
out=[]
for p in [int(x) for x in sys.argv[1:]]:
    H = sorted(set(min(x, p-x) for x in (pow(a,2,p) for a in range(1,p))))
    P = CirculantSoS(p, device="cuda").set_instance(H)
    P.solve(iters=200000, tol=1e-12); lo, hi = P.certified_bounds()
    s2 = 0.5 + (1 + math.sqrt(p)) / (2 * (p - 1))
    print(f"p={p:3d} |H|={len(H):2d}  SoS4 in [{lo:.9f}, {hi:.9f}]  SoS2 = {s2:.9f}  SoS2-SoS4_hi = {s2-hi:.2e}  SoS2-SoS4_lo = {s2-lo:.2e}  its {P.iters_done}", flush=True)
    out.append({"p": p, "sos4_lo": lo, "sos4_hi": hi, "sos2": s2, "iters": P.iters_done})
json.dump(out, open("../results/paley_lift.json", "w"), indent=1)
