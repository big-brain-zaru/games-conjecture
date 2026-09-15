"""
expander_gap.py -- degree-2 vs degree-4 gap shape of Max-Cut on sparse expanders.

Mohanty-Raghavendra-Xu (STOC 2020) lift the degree-2 (spectral) pseudo-expectation
of Max-Cut on random d-regular graphs to degree 4, so that SoS_4 = SoS_2 - o_n(1).
Consequence for the gap shape: C_4 -> C_2 ~ (1 - mc_d)/(1/2 - sqrt(d-1)/d) which
for d = 3 is about 2.6, far above the record 1.09 found on small structured
instances.  Here we measure it exactly at finite n: random d-regular graphs and
cubic Cayley graphs, certified two-sided SoS_4, SoS_2 by the same solver, and a
proved maximum cut (CP-SAT).
"""
import argparse, json, time
import numpy as np, torch, networkx as nx
from sos_gpu import BooleanSoS
from weight_ascent import cpsat_min_unsat

def solve(n, E, degree, iters, tol, dev):
    m = len(E)
    S = BooleanSoS(n, E, degree=degree, device=dev, dtype=torch.float64).set_instance(-np.ones(m), np.ones(m))
    S.solve(iters=iters, tol=tol); lo, hi = S.certified_bounds()
    return lo, hi, S.iters_done

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[16, 20, 24, 30])
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--iters", type=int, default=40000)
    ap.add_argument("--cp_time", type=float, default=300)
    ap.add_argument("--out", default="../results/expander_gap.jsonl")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for n in a.n:
        for t in range(a.trials):
            G = nx.random_regular_graph(a.d, n, seed=100 * n + t)
            E = np.array(sorted(tuple(sorted(e)) for e in G.edges())); m = len(E)
            girth = min((len(c) for c in nx.minimum_cycle_basis(G)), default=0)
            lam = np.linalg.eigvalsh(nx.to_numpy_array(G))
            t0 = time.time()
            lo2, hi2, it2 = solve(n, E, 2, a.iters, 1e-11, dev)
            lo4, hi4, it4 = solve(n, E, 4, a.iters, 1e-10, dev)
            xs, unsat, unsat_ub, pr = cpsat_min_unsat(n, E, -np.ones(m), np.ones(m) / m, time_limit=a.cp_time)
            C2 = unsat / (1 - hi2); C4_lo = unsat_ub / (1 - lo4); C4_hi = unsat / (1 - hi4)
            spectral = 0.5 - lam[0] / (2 * a.d)
            rec = {"n": n, "d": a.d, "trial": t, "girth": girth, "lambda_min": lam[0], "spectral_bound": spectral,
                   "opt": 1 - unsat, "proved": pr, "sos2_lo": lo2, "sos2_hi": hi2, "sos4_lo": lo4, "sos4_hi": hi4,
                   "C2": C2, "C4_lo": C4_lo, "C4_hi": C4_hi, "retention": (C4_lo - 1) / (C2 - 1), "secs": time.time() - t0}
            print(f"n={n:3d} d={a.d} t={t} girth {girth}  opt {1-unsat:.5f}({'proved' if pr else 'bound'})  spectral {spectral:.5f}  "
                  f"SoS2 [{lo2:.5f},{hi2:.5f}]  SoS4 [{lo4:.5f},{hi4:.5f}]  C2 {C2:.4f}  C4 [{C4_lo:.4f},{C4_hi:.4f}]  "
                  f"ret {(C4_lo-1)/(C2-1):.3f}  {time.time()-t0:.0f}s", flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
