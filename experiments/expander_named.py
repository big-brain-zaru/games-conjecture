"""expander_named.py -- degree-4 vs degree-2 gap shape on named high-girth cubic graphs and larger random cubic graphs."""
import sys, json, time, numpy as np, torch, networkx as nx
from expander_gap import solve
from weight_ascent import cpsat_min_unsat
def high_girth_random(d, n, g, seed):
    rng = np.random.default_rng(seed)
    for t in range(20000):
        G = nx.random_regular_graph(d, n, seed=int(rng.integers(1 << 30)))
        girth = min(len(c) for c in nx.minimum_cycle_basis(G))
        if girth >= g:
            return G, girth
    return G, girth
cases = {"mcgee24": (nx.LCF_graph(24, [12, 7, -7], 8), 7),
         "rand32_g5": high_girth_random(3, 32, 5, 1), "rand40_g5": high_girth_random(3, 40, 5, 2)}
dev = "cuda"
for name in sys.argv[1:]:
    G, girth = cases[name]
    n = G.number_of_nodes(); E = np.array(sorted(tuple(sorted(e)) for e in G.edges())); m = len(E)
    lam = np.linalg.eigvalsh(nx.to_numpy_array(G)); t0 = time.time()
    lo2, hi2, _ = solve(n, E, 2, 40000, 1e-11, dev)
    lo4, hi4, it4 = solve(n, E, 4, 40000, 1e-10, dev)
    xs, unsat, unsat_ub, pr = cpsat_min_unsat(n, E, -np.ones(m), np.ones(m) / m, time_limit=600)
    C2 = unsat / (1 - hi2); C4_lo = unsat_ub / (1 - lo4); C4_hi = unsat / (1 - hi4)
    rec = {"name": name, "n": n, "girth": girth, "lambda_min": lam[0], "opt": 1 - unsat, "proved": pr,
           "sos2": hi2, "sos4_lo": lo4, "sos4_hi": hi4, "C2": C2, "C4_lo": C4_lo, "C4_hi": C4_hi, "its4": it4, "secs": time.time() - t0}
    print(f"n={n:3d} {name} girth {girth} opt {1-unsat:.5f}({'proved' if pr else 'bound'}) SoS2 {hi2:.5f} SoS4 [{lo4:.5f},{hi4:.5f}] C2 {C2:.4f} C4 [{C4_lo:.4f},{C4_hi:.4f}] {time.time()-t0:.0f}s", flush=True)
    with open("../results/expander_gap.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
