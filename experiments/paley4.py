"""paley4.py -- The index-4 generalised Paley circulants Cay(Z_p, H), H the
multiplicative subgroup of index 4 (exists with -1 in H iff p = 1 mod 8).
This family is where the degree-4 gap shape C_4 was observed to GROW
(p=17: 1.081179, p=41: 1.093516), so it gets a dedicated, longer-budget run.

Reported:
  C4_rigorous = (1 - opt_dual_bound) / (1 - sos4_certified_lower)   [always valid]
  C4_if_opt   = (1 - opt_incumbent)  / (1 - sos4_certified_upper)   [valid iff CP-SAT proved]
"""
import warnings; warnings.filterwarnings("ignore")
import argparse, json, time
import numpy as np, torch
from sympy import primitive_root
from circulant_sos import CirculantSoS
from circulant_scan import circ_edges, maxcut_cpsat, sos2_circulant

def index4(p):
    if (p - 1) % 8: return None
    g = primitive_root(p)
    H = sorted({pow(g, 4 * i, p) for i in range((p - 1) // 4)})
    assert (p - 1) in H
    return sorted({min(h, p - h) for h in H})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", default="17,41,73,89,97,113")
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--cp", type=float, default=600)
    ap.add_argument("--out", default="../results/paley4.jsonl")
    a = ap.parse_args()
    for p in [int(x) for x in a.ps.split(",")]:
        gens = index4(p)
        if not gens: print(f"skip p={p} (p != 1 mod 8)", flush=True); continue
        t0 = time.time()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        S = CirculantSoS(p, device=dev)
        S.solve(iters=1) if False else None
        S.set_instance(gens); S.solve(iters=a.iters, tol=1e-12)
        lo, hi = S.certified_bounds(); tsos = time.time() - t0
        e = circ_edges(p, gens)
        t1 = time.time()
        opt, ub, pr = maxcut_cpsat(p, e, time_limit=a.cp, workers=11)
        s2 = sos2_circulant(p, gens)
        rec = {"p": p, "deg": 2 * len(gens), "gens": gens, "m": len(e), "n_var": S.n_var, "nI": S.nI,
               "opt_incumbent": opt, "opt_dual_bound": ub, "proved": pr,
               "sos2": s2, "sos4_lo": lo, "sos4_hi": hi, "sos_iters": S.iters_done,
               "C2": (1 - opt) / max(1 - s2, 1e-12),
               "C4_rigorous": (1 - ub) / max(1 - min(1.0, lo), 1e-12),
               "C4_if_opt": (1 - opt) / max(1 - min(1.0, hi), 1e-12),
               "sos_s": round(tsos, 1), "cp_s": round(time.time() - t1, 1)}
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
