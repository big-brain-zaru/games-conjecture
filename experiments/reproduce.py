"""
reproduce.py -- Re-derive every headline number in docs/FINDINGS.md from scratch.

Run from experiments/:   python reproduce.py
Each check recomputes the quantity (it does not read it from a result file) and
compares against the published value.  Prints "N checks, M mismatches".
"""
from __future__ import annotations

import math
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

CHECKS = []
FAIL = []


def check(name, got, want, tol=1e-5):
    ok = abs(got - want) <= tol
    CHECKS.append((name, got, want, ok))
    if not ok:
        FAIL.append(name)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}: got {got:.6f}, published {want:.6f}")
    return ok


def main():
    t0 = time.time()
    from gap_search import brute_opt_boolean, complete_graph, cycle, boolean_game, cycle_reference
    from ug_sos import sos4_boolean
    from sos_gpu import BooleanSoS
    import networkx as nx

    print("== 1. Degree-2 calibration: odd cycles reproduce the closed form 4L/pi^2 ==")
    for L, want in [(5, 2.0944272), (7, 2.8850961), (9, 3.6848264), (11, 4.4885587),
                    (13, 5.2944184), (15, 6.1015417)]:
        e = cycle(L); m = len(e); s = -np.ones(m, dtype=int); w = np.ones(m)
        opt, _ = brute_opt_boolean(e, s, w, L)
        g = boolean_game(e, s, w)
        d2, _ = sos4_boolean(g, degree=2, eps=1e-10)
        C2 = (1 - opt) / (1 - min(1.0, d2))
        check(f"C_2(C_{L})", C2, want, tol=2e-4)
        check(f"C_2(C_{L}) closed form", cycle_reference(L), want, tol=2e-4)

    print("== 2. Degree-4 solver validation ==")
    for nm, e, n, wopt, w2, w4 in [
        ("K5", complete_graph(5), 5, 0.6, 0.625, 0.625),
        ("K7", complete_graph(7), 7, 0.5714286, 0.5833333, 0.5833333),
        ("C5", cycle(5), 5, 0.8, 0.9045085, 0.8),
    ]:
        m = len(e); s = -np.ones(m, dtype=int); w = np.ones(m)
        opt, _ = brute_opt_boolean(e, s, w, n)
        g = boolean_game(e, s, w)
        d2, _ = sos4_boolean(g, degree=2, eps=1e-10)
        d4, _ = sos4_boolean(g, degree=4, eps=1e-10)
        check(f"{nm} opt", opt, wopt)
        check(f"{nm} SoS_2", min(1.0, d2), w2, tol=1e-4)
        check(f"{nm} SoS_4", min(1.0, d4), w4, tol=1e-4)
    check("C_4(K5) = 1 + 1/(n(n-2))", 1 + 1 / (5 * 3), 1.0666667)
    P = nx.petersen_graph(); e = np.array(sorted((min(a, b), max(a, b)) for a, b in P.edges()))
    m = len(e); s = -np.ones(m, dtype=int); w = np.ones(m)
    opt, _ = brute_opt_boolean(e, s, w, 10)
    g = boolean_game(e, s, w); d4, _ = sos4_boolean(g, degree=4, eps=1e-10)
    check("Petersen opt", opt, 0.8)
    check("Petersen C_4", (1 - opt) / (1 - min(1.0, d4)), 1.0, tol=1e-3)

    print("== 3. Best degree-4 gap found: Cay(Z_9, {+-1,+-2}) ==")
    L = 9
    e = np.array(sorted(set((min(v, (v + s_) % L), max(v, (v + s_) % L))
                            for v in range(L) for s_ in (1, 2))), dtype=np.int64)
    e = np.array([x for x in e if x[0] != x[1]])
    m = len(e); s = -np.ones(m, dtype=int); w = np.ones(m)
    opt, _ = brute_opt_boolean(e, s, w, L)
    S = BooleanSoS(L, e, degree=4, device="cpu", dtype=torch.float64).set_instance(s.astype(float), w)
    S.solve(iters=8000, tol=1e-12)
    lo, hi = S.certified_bounds()
    check("Cay(Z9,{1,2}) opt", opt, 2 / 3)
    check("Cay(Z9,{1,2}) SoS_4", lo, 0.689674, tol=1e-5)
    check("Cay(Z9,{1,2}) certificate width", hi - lo, 0.0, tol=1e-5)
    check("Cay(Z9,{1,2}) C_4", (1 - opt) / (1 - lo), 1.074139, tol=1e-4)

    print("== 3b. Certified degree-4 champion: Cay(Z_17, H_4), the index-4 Paley circulant ==")
    from circulant_sos import CirculantSoS as _CS
    from circulant_scan import circ_edges, maxcut_cpsat
    gens17 = [1, 4]
    S17 = _CS(17, device="cpu").set_instance(gens17)
    S17.solve(iters=30000, tol=1e-12)
    lo17, hi17 = S17.certified_bounds()
    e17 = circ_edges(17, gens17)
    o17, ub17, pr17 = maxcut_cpsat(17, e17, time_limit=60)
    check("Cay(Z17,H4) opt", o17, 13 / 17)
    check("Cay(Z17,H4) optimality proved", 1.0 if pr17 else 0.0, 1.0, tol=0)
    check("Cay(Z17,H4) SoS_4", lo17, 0.782373, tol=1e-5)
    check("Cay(Z17,H4) certificate width", hi17 - lo17, 0.0, tol=1e-5)
    check("Cay(Z17,H4) C_4", (1 - ub17) / (1 - lo17), 1.081179, tol=1e-4)

    print("== 4. Symmetry-reduced circulant SoS agrees with the dense solver ==")
    from circulant_sos import CirculantSoS
    C = CirculantSoS(9, device="cpu").set_instance([1, 2])
    C.solve(iters=8000, tol=1e-12)
    clo, chi = C.certified_bounds()
    check("symmetric SoS_4 == dense SoS_4 (L=9)", clo, lo, tol=1e-5)
    check("symmetric certificate width", chi - clo, 0.0, tol=1e-5)

    print("== 5. Khot-Vishnoi: construction, transversal identity, exact SDP values ==")
    from kv_instance import khot_vishnoi
    from kv_transversal import kv_transversal_data, subcube_choice, transversal_value
    from kv_invariant_sdp import solve_kv_invariant
    g = khot_vishnoi(3, 0.2)
    check("KV k=3 vertices", g.n, 32, tol=0)
    check("KV k=3 labels", g.k, 8, tol=0)
    check("KV k=3 constraints", g.m, 1472, tol=0)
    gall = khot_vishnoi(3, 0.2, keep_all_distances=True)
    check("KV total weight = 1-(1-eta)^N", gall.total_weight, 1 - 0.8 ** 8, tol=1e-9)
    D = kv_transversal_data(3, 0.2)
    check("KV k=3 subcube value", transversal_value(D, subcube_choice(D)), 0.439189, tol=1e-6)
    D3 = kv_transversal_data(3, 0.3)
    check("KV k=3 eta=0.3 subcube value", transversal_value(D3, subcube_choice(D3)), 0.234424, tol=1e-6)
    for k, eta, want in [(3, 0.2, 0.6157095), (3, 0.3, 0.3948238), (4, 0.1, 0.7950017)]:
        v, *_ = solve_kv_invariant(k, eta, verbose=False)
        check(f"KV k={k} eta={eta} exact basic SDP", v, want, tol=1e-6)

    print("== 6. The group framework reproduces Khot-Vishnoi ==")
    from group_ug import elementary_abelian, group_unique_game, hamming_noise_f2
    from group_census import simplex_code
    G = elementary_abelian(8)
    H = simplex_code(3)
    assert G.is_normal(H)
    mu = hamming_noise_f2(8, 0.2, D["dmin"], D["dmax"])
    gg = group_unique_game(G, H, mu)
    check("group framework -> KV vertices", gg.n, 32, tol=0)
    check("group framework -> KV labels", gg.k, 8, tol=0)
    check("group framework -> KV constraints", gg.m, 1472, tol=0)

    print("== 7. Certified basic-SDP solver: primal = dual ==")
    from ug_core import random_unique_game, maxsat_optimum
    from ug_sdp import sdp_cvxpy, sdp_block_ascent, dual_certificate
    gu = random_unique_game(12, 3, m=30, seed=1)
    exact, _, _ = sdp_cvxpy(gu)
    lo2, V, _ = sdp_block_ascent(gu, r=8, sweeps=400, device="cpu")
    hi2, _ = dual_certificate(gu, V)
    check("block ascent == cvxpy", lo2, exact, tol=2e-4)
    check("dual certificate width", hi2 - lo2, 0.0, tol=1e-3)
    opt, _, _ = maxsat_optimum(gu)
    check("opt <= SDP", 1.0 if opt <= exact + 1e-9 else 0.0, 1.0, tol=0)

    print(f"\n{len(CHECKS)} checks, {len(FAIL)} mismatches   ({time.time()-t0:.0f}s)")
    if FAIL:
        print("mismatched:", ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
