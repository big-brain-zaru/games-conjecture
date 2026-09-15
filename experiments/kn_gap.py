"""kn_gap.py -- degree-4 gap shape of Max-Cut on the complete graph K_n (certified)."""
import sys, json, itertools, numpy as np, torch
from sos_gpu import BooleanSoS
from weight_ascent import signed_complete
out = []
for n in range(5, int(sys.argv[1]) + 1):
    E = np.array(list(itertools.combinations(range(n), 2))); m = len(E)
    S = BooleanSoS(n, E, degree=4, device="cuda", dtype=torch.float64).set_instance(-np.ones(m), np.ones(m))
    S.solve(iters=40000, tol=1e-11); lo, hi = S.certified_bounds()
    opt = (n // 2) * (n - n // 2) / m
    C = (1 - opt) / (1 - hi); Clo = (1 - opt) / (1 - lo)
    print(f"K_{n:2d}: opt {opt:.6f}  SoS4 [{lo:.7f},{hi:.7f}]  C4 in [{Clo:.6f},{C:.6f}]  its {S.iters_done}", flush=True)
    out.append({"n": n, "opt": opt, "sos4_lo": lo, "sos4_hi": hi, "C4_lo": Clo, "C4_hi": C})
json.dump(out, open("../results/kn_gap.json", "w"), indent=1)
