"""genpaley.py -- Generalised Paley graphs Cay(Z_p, H) with H the multiplicative
subgroup of index k (so H = -H), all-anti Max-Cut.  The L=17 winner
Cay(Z_17,{+-1,+-4}) is the index-4 case, so this is the structured family the
search picked out.  Certified degree-4 SoS (symmetry-reduced) + CP-SAT optimum."""
import warnings; warnings.filterwarnings("ignore")
import argparse, json, time
import numpy as np, torch
from circulant_sos import CirculantSoS
from circulant_scan import circ_edges, maxcut_cpsat, sos2_circulant

def primitive_root(p):
    from sympy import primitive_root as pr
    return pr(p)

def subgroup(p, k):
    """The subgroup of Z_p^* of index k (order (p-1)/k); returns None unless -1 is in it."""
    if (p - 1) % k: return None
    g = primitive_root(p)
    H = sorted({pow(g, k * i, p) for i in range((p - 1) // k)})
    if (p - 1) not in H:                      # -1 = p-1 must lie in H for an undirected graph
        return None
    return H

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", default="13,17,29,37,41,53,61,73,89,97,101,109,113")
    ap.add_argument("--ks", default="2,4,6,8")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--cp", type=float, default=60)
    ap.add_argument("--maxdeg", type=int, default=40)
    ap.add_argument("--out", default="../results/genpaley.jsonl")
    a = ap.parse_args()
    best = 0.0
    for p in [int(x) for x in a.ps.split(",")]:
        for k in [int(x) for x in a.ks.split(",")]:
            H = subgroup(p, k)
            if H is None or len(H) < 2 or len(H) > a.maxdeg: continue
            gens = sorted({min(h, p - h) for h in H})
            t0 = time.time()
            dev = "cuda" if (torch.cuda.is_available() and p >= 29) else "cpu"
            S = CirculantSoS(p, device=dev).set_instance(gens)
            S.solve(iters=a.iters, tol=1e-11)
            lo, hi = S.certified_bounds()
            e = circ_edges(p, gens)
            opt, ub, proved = maxcut_cpsat(p, e, time_limit=a.cp)
            s2 = sos2_circulant(p, gens)
            C4 = (1 - ub) / max(1 - min(1.0, lo), 1e-12)
            rec = {"p": p, "index_k": k, "deg": len(H), "gens": gens, "m": len(e), "opt": opt,
                   "opt_ub": ub, "proved": proved, "sos2": s2, "sos4_lo": lo, "sos4_hi": hi,
                   "C2": (1 - opt) / max(1 - s2, 1e-12), "C4": C4,
                   "tight": abs(hi - lo) < 1e-5, "s": round(time.time() - t0, 1)}
            best = max(best, C4)
            print(json.dumps({kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in rec.items()}), flush=True)
            with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
        print(f"### after p={p}: best C_4 = {best:.6f}", flush=True)
