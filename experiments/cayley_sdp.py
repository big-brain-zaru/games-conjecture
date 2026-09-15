"""
cayley_sdp.py -- the exact degree-2 (Goemans-Williamson) value of a signed
Cayley-graph instance, from the irreps, and the irrep that carries it.

For a Q-invariant Boolean 2Lin instance on Cay(Q, S) with class weights omega_t
(fractions of total edge weight) and signs b_t, a Q-invariant SDP solution is a
positive-definite kernel K on Q with K(e) = 1; by Fourier K corresponds to
Khat(rho) >= 0 with sum_rho d_rho tr Khat(rho) = |Q|, the objective is linear in
Khat, and the maximum puts all mass on the top eigenvector of the best block:

    SoS_2 = 1/2 + (1/2) * max_rho lambda_max( A_rho ),
    A_rho = sum_t omega_t b_t * (rho(s_t) + rho(s_t)^*)/2      (a single term if s_t^2 = e).

The argmax rho is the irrep carrying the degree-2 gap; its dimension is the
quantity abelian groups cannot vary.  Validated below against the dense
degree-2 solver on non-abelian groups and against the closed form on cycles.
"""
from __future__ import annotations

import numpy as np


def class_weights(G, gens, weights):
    """omega_t: fraction of edges (times weight) in each generator class."""
    n = G.n
    mult = np.array([n if int(G.mul[g, g]) != G.e else n // 2 for g in gens], float)
    om = np.asarray(weights, float) * mult
    return om / om.sum()


def sos2_cayley(G, irreps, gens, signs=None, weights=None, return_arg=False):
    gens = [int(g) for g in gens]
    b = np.array([-1.0] * len(gens)) if signs is None else np.asarray(signs, float)
    w = np.ones(len(gens)) if weights is None else np.asarray(weights, float)
    om = class_weights(G, gens, w)
    best, arg = -np.inf, None
    for k, (d, R) in enumerate(irreps):
        A = np.zeros((d, d), dtype=complex)
        for t, g in enumerate(gens):
            if int(G.mul[g, g]) == G.e:
                A += om[t] * b[t] * R[g]
            else:
                A += om[t] * b[t] * (R[g] + R[int(G.inv[g])]) / 2
        A = (A + A.conj().T) / 2
        lam = float(np.linalg.eigvalsh(A).max())
        if lam > best:
            best, arg = lam, (k, d)
    val = 0.5 + 0.5 * best
    return (val, arg) if return_arg else val


def _selftest():
    import warnings; warnings.filterwarnings("ignore")
    import torch
    from group_ug import cyclic, extraspecial_2, heisenberg, group_from_permutations
    from group_library import dihedral, dicyclic, metacyclic
    from group_sos import complex_irreps, GroupSoS
    from sos_gpu import BooleanSoS
    from circulant_scan import sos2_circulant
    # cycles: closed form
    for L, gens in [(9, [1, 2]), (13, [1, 3]), (15, [1, 2, 4])]:
        G = cyclic(L); irr = complex_irreps(G)
        v = sos2_cayley(G, irr, gens)
        ref = sos2_circulant(L, gens)
        assert abs(v - ref) < 1e-9, (L, v, ref)
    print("  cycles: irrep formula == closed form")
    # non-abelian: against the dense degree-2 ADMM with certified bounds
    rng = np.random.default_rng(0)
    cases = [("S3", group_from_permutations("S3", [[1, 0, 2], [1, 2, 0]], 3)),
             ("D4", extraspecial_2(1)), ("Q8", dicyclic(2)), ("D5", dihedral(5)),
             ("A4", group_from_permutations("A4", [[1, 2, 0, 3], [0, 2, 3, 1]], 4)),
             ("Heis3", heisenberg(3)), ("Z7x|Z3", metacyclic(7, 3, 2)), ("S4", group_from_permutations("S4", [[1, 0, 2, 3], [1, 2, 3, 0]], 4))]
    for name, G in cases:
        irr = complex_irreps(G)
        P = GroupSoS(G, device="cpu")
        for trial in range(2):
            k = int(rng.integers(1, min(4, len(P.pair_reps)) + 1))
            gens = [int(x) for x in rng.choice(P.pair_reps, size=k, replace=False)]
            signs = [int(x) for x in rng.choice([-1, 1], size=k)]
            if all(s > 0 for s in signs):
                signs[0] = -1
            wts = rng.uniform(0.5, 2.0, size=k)
            P.set_instance(gens, signs=signs, weights=wts)
            E, B, W = P.edges()
            ref = BooleanSoS(G.n, E, degree=2, device="cpu", dtype=torch.float64).set_instance(B.astype(float), W)
            ref.solve(iters=20000, tol=1e-12); lo, hi = ref.certified_bounds()
            v, (kk, d) = sos2_cayley(G, irr, gens, signs=signs, weights=wts, return_arg=True)
            ok = lo - 1e-6 <= v <= hi + 1e-6
            print(f"  {name:7s} gens={gens} signs={signs}: irrep formula {v:.6f}  dense [{lo:.6f},{hi:.6f}]  "
                  f"carried by irrep of dim {d}  {'ok' if ok else 'MISMATCH'}")
            assert ok
    print("cayley_sdp selftest: OK")


if __name__ == "__main__":
    _selftest()
