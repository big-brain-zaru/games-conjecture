"""paley_index.py -- generalised Paley circulants Cay(Z_p, H_k) for a chosen index k,
with LONG budgets so both the degree-4 certificate and the max-cut optimality proof close.
Gives a curve of the degree-4 gap shape indexed by the graph degree."""
import warnings; warnings.filterwarnings("ignore")
import argparse, json, time
import numpy as np, torch
from sympy import primitive_root
from circulant_sos import CirculantSoS
from circulant_scan import circ_edges, maxcut_cpsat, sos2_circulant

def subgroup_gens(p, k):
    if (p - 1) % k: return None
    g = primitive_root(p)
    H = sorted({pow(g, k * i, p) for i in range((p - 1) // k)})
    if (p - 1) not in H: return None
    return sorted({min(h, p - h) for h in H})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="37:6,61:6,73:6,89:4,97:8,113:8")
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--cp", type=float, default=2400)
    ap.add_argument("--out", default="../results/paley_index.jsonl")
    a = ap.parse_args()
    for job in a.jobs.split(","):
        p, k = [int(x) for x in job.split(":")]
        gens = subgroup_gens(p, k)
        if not gens:
            print(f"skip p={p} k={k}", flush=True); continue
        t0 = time.time()
        S = CirculantSoS(p, device="cuda" if torch.cuda.is_available() else "cpu").set_instance(gens)
        S.solve(iters=a.iters, tol=1e-12)
        lo, hi = S.certified_bounds(); tsos = time.time() - t0
        e = circ_edges(p, gens); t1 = time.time()
        opt, ub, pr = maxcut_cpsat(p, e, time_limit=a.cp, workers=11)
        rec = {"p": p, "index_k": k, "deg": 2 * len(gens), "gens": gens, "m": len(e),
               "n_var": S.n_var, "nI": S.nI, "opt": opt, "opt_ub": ub, "proved": pr,
               "sos2": sos2_circulant(p, gens), "sos4_lo": lo, "sos4_hi": hi,
               "cert_width": hi - lo, "sos_iters": S.iters_done,
               "C4_rigorous": (1 - ub) / max(1 - min(1.0, lo), 1e-12),
               "C4_if_tight": (1 - opt) / max(1 - min(1.0, hi), 1e-12),
               "sos_s": round(tsos, 1), "cp_s": round(time.time() - t1, 1)}
        print(json.dumps({kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in rec.items()}), flush=True)
        with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
