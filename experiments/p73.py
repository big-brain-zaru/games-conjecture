"""p73.py -- close the two halves of Cay(Z_73, H_4) separately.
(a) degree-4 certificate: very long symmetry-reduced ADMM on the GPU
(b) max-cut: long CP-SAT, plus a strong local-search incumbent (many-restart
    Kernighan-Lin style 1-opt/2-opt) to check the incumbent is not beatable."""
import warnings; warnings.filterwarnings("ignore")
import argparse, json, time
import numpy as np, torch
from sympy import primitive_root
from circulant_sos import CirculantSoS
from circulant_scan import circ_edges, maxcut_cpsat

def index4(p):
    g = primitive_root(p)
    H = sorted({pow(g, 4*i, p) for i in range((p-1)//4)})
    return sorted({min(h, p-h) for h in H})

def local_search(p, gens, restarts=4000, seed=0):
    rng = np.random.default_rng(seed)
    e = circ_edges(p, gens); m = len(e)
    adj = [[] for _ in range(p)]
    for a, b in e.tolist(): adj[a].append(b); adj[b].append(a)
    adj = [np.array(x) for x in adj]
    best = -1
    for r in range(restarts):
        x = rng.integers(0, 2, size=p)
        improved = True
        while improved:
            improved = False
            for v in rng.permutation(p):
                same = int((x[adj[v]] == x[v]).sum()); diff = len(adj[v]) - same
                if same > diff:
                    x[v] ^= 1; improved = True
        cut = int(sum(1 for a, b in e.tolist() if x[a] != x[b]))
        if cut > best: best, bx = cut, x.copy()
    return best, m, best / m

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=73)
    ap.add_argument("--iters", type=int, default=400000)
    ap.add_argument("--cp", type=float, default=10800)
    ap.add_argument("--restarts", type=int, default=4000)
    ap.add_argument("--out", default="../results/p73.json")
    a = ap.parse_args()
    p = a.p; gens = index4(p); e = circ_edges(p, gens)
    t0 = time.time()
    cut, m, frac = local_search(p, gens, restarts=a.restarts)
    print(json.dumps({"stage": "local_search", "p": p, "m": m, "best_cut": cut,
                      "fraction": frac, "s": round(time.time()-t0, 1)}), flush=True)
    t0 = time.time()
    S = CirculantSoS(p, device="cuda" if torch.cuda.is_available() else "cpu").set_instance(gens)
    S.solve(iters=a.iters, tol=1e-13, verbose=True, log_every=50000)
    lo, hi = S.certified_bounds()
    print(json.dumps({"stage": "sos4", "p": p, "sos4_lo": lo, "sos4_hi": hi, "width": hi-lo,
                      "iters": S.iters_done, "s": round(time.time()-t0, 1)}), flush=True)
    t0 = time.time()
    opt, ub, pr = maxcut_cpsat(p, e, time_limit=a.cp, workers=11)
    rec = {"p": p, "gens": gens, "m": len(e), "local_search_fraction": frac,
           "opt_incumbent": opt, "opt_dual_bound": ub, "proved": pr,
           "sos4_lo": lo, "sos4_hi": hi,
           "C4_rigorous": (1-ub)/max(1-min(1.0, lo), 1e-12),
           "C4_if_incumbent_optimal": (1-opt)/max(1-min(1.0, lo), 1e-12),
           "cp_s": round(time.time()-t0, 1)}
    print(json.dumps({k: (round(v,6) if isinstance(v,float) else v) for k,v in rec.items()}), flush=True)
    json.dump(rec, open(a.out, "w"), indent=1)
