"""
group_census.py -- Census of unique games built from group quotients
(see group_ug.py), measuring the GAP SHAPE

        C_d(I) = (1 - opt(I)) / (1 - R_d(I))

for d = 2 (basic SDP) and d = 4 (level-2 Lasserre), with opt exact and R_d
certified.  C_d is the constant Khot-Moshkovitz ask about: they note that no
(1-eps, 1-C eps) Lasserre gap with C -> infinity is known for unique games at
any constant degree.

The question this census asks: does replacing the abelian group of the
Khot-Vishnoi construction by a NON-ABELIAN group increase C_4?  All known
(1-eps, delta) instances are abelian and their soundness is degree-4 SoS
certifiable via hypercontractivity (BBHKSZ 2012, Thm 6.11); no such certificate
is known for non-abelian Cayley graphs.

Every number is verified: opt by exhaustive/CP-SAT with a re-evaluated
assignment, R_2 by a certified primal/dual pair, R_4 by cvxpy/SCS (small) with
the sandwich opt <= R_4 <= R_2 checked.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from group_ug import (Group, elementary_abelian, cyclic, direct_product, semidirect,
                      heisenberg, extraspecial_2, group_from_permutations,
                      group_unique_game, word_noise, hamming_noise_f2,
                      psi_from_permutation_rep, psi_valid, sdp_value_of_psi,
                      labelling_to_transversal, transversal_value)
from ug_core import brute_force_optimum, cpsat_optimum, UniqueGame
from ug_sdp import sdp_block_ascent, dual_certificate
from ug_sos import lasserre2


# ----------------------------------------------------------------------------
def simplex_code(k: int) -> np.ndarray:
    """The Khot-Vishnoi subgroup: bitmasks of chi_S on N = 2^k points."""
    N = 1 << k
    out = []
    for S in range(N):
        m = 0
        for x in range(N):
            if bin(S & x).count("1") % 2 == 1:
                m |= (1 << x)
        out.append(m)
    return np.array(sorted(out))


def candidate_groups():
    """(Group, [normal subgroups to try], generating set for the noise walk)."""
    out = []
    # --- abelian baselines ---------------------------------------------------
    G = elementary_abelian(6)
    out.append(("F2^6 / simplex-ish", G, [np.array(sorted(G.subgroup_generated([0b000111, 0b011001])))],
                [1 << i for i in range(6)]))
    G = elementary_abelian(8)                                    # KV k=3 lives here
    out.append(("F2^8 / KV simplex", G, [simplex_code(3)], [1 << i for i in range(8)]))
    Z4 = cyclic(4); Z2 = cyclic(2)
    G = direct_product(Z4, direct_product(Z2, Z2))
    out.append(("Z4xZ2xZ2", G, None, None))
    G = direct_product(cyclic(3), cyclic(9))
    out.append(("Z3xZ9", G, None, None))
    # --- non-abelian ---------------------------------------------------------
    out.append(("Heis(3) [order 27]", heisenberg(3), None, None))
    out.append(("2^(1+2) = D4 [order 8]", extraspecial_2(1), None, None))
    out.append(("2^(1+4) [order 32]", extraspecial_2(2), None, None))
    # S_4, A_4, SL(2,3)-like via permutations
    out.append(("S4", group_from_permutations("S4", [[1, 0, 2, 3], [1, 2, 3, 0]], 4), None, None))
    out.append(("A4", group_from_permutations("A4", [[1, 2, 0, 3], [0, 2, 3, 1]], 4), None, None))
    out.append(("S3xS3", group_from_permutations("S3xS3", [[1, 0, 2, 3, 4, 5], [1, 2, 0, 3, 4, 5],
                                                           [0, 1, 2, 4, 3, 5], [0, 1, 2, 4, 5, 3]], 6), None, None))
    # dihedral of order 16 and the semidihedral/quaternion family via permutations of an 8-cycle
    out.append(("D8 [order 16]", group_from_permutations("D8", [[1, 2, 3, 4, 5, 6, 7, 0],
                                                                [0, 7, 6, 5, 4, 3, 2, 1]], 8), None, None))
    # F_2^4 x| Z_3 (Z3 acting by a fixed-point-free order-3 linear map on F_2^4 = F_4^2)
    F16 = elementary_abelian(4)
    # F_4^2 with omega multiplication: identify F_2^4 = F_4^2, multiply by omega
    def om(x):
        a, b = x & 3, (x >> 2) & 3
        mulw = lambda t: {0: 0, 1: 2, 2: 3, 3: 1}[t]             # multiply by omega in F_4
        return mulw(a) | (mulw(b) << 2)
    act = {0: np.arange(16), 1: np.array([om(x) for x in range(16)])}
    act[2] = act[1][act[1]]
    Z3 = cyclic(3)
    G = semidirect(F16, Z3, act, name="F2^4 x| Z3")
    out.append(("F2^4 x| Z3 [order 48]", G, None, None))
    return out


def analyse(g: UniqueGame, do_sos4=True, cpsat_time=300, tag=""):
    rec = {"tag": tag, "n": g.n, "k": g.k, "m": g.m,
           **{a: b for a, b in g.meta.items() if a in ("group", "H_order", "Q_order", "abelian", "name")}}
    # --- exact optimum -------------------------------------------------------
    t = time.time()
    if g.k ** g.n <= 4_000_000:
        opt, L = brute_force_optimum(g)
        rec["opt_method"] = "brute"; rec["opt_certified"] = True
    else:
        opt, L, info = cpsat_optimum(g, time_limit=cpsat_time, workers=10)
        rec["opt_method"] = "cpsat"; rec["opt_certified"] = bool(info["optimal"])
        rec["opt_bound"] = info["bound_value"]
    rec["opt"] = float(opt); rec["opt_time"] = time.time() - t
    # --- basic SDP (certified) ----------------------------------------------
    t = time.time()
    r2, V, _ = sdp_block_ascent(g, r=min(48, g.n * g.k), sweeps=800, tol=1e-12, restarts=2)
    r2_hi, lam = dual_certificate(g, V)
    rec.update({"sdp_lo": float(r2), "sdp_hi": float(r2_hi), "sdp_time": time.time() - t})
    rec["C2"] = (1 - opt) / max(1 - min(1.0, r2), 1e-12)
    rec["C2_conservative"] = (1 - opt) / max(1 - min(1.0, r2_hi), 1e-12)
    # --- degree-4 (level-2 Lasserre) ----------------------------------------
    D = 1 + g.n * g.k + (g.n * (g.n - 1) // 2) * g.k ** 2
    rec["lasserre2_D"] = int(D)
    if do_sos4 and D <= 1400:
        t = time.time()
        try:
            r4, info4 = lasserre2(g, eps=1e-8, max_iters=40000)
            rec.update({"sos4": float(min(1.0, r4)), "sos4_status": info4["status"], "sos4_time": time.time() - t})
            rec["C4"] = (1 - opt) / max(1 - min(1.0, r4), 1e-12)
            rec["sandwich_ok"] = bool(opt - 1e-4 <= r4 <= r2_hi + 1e-3)
        except Exception as ex:
            rec["sos4_error"] = str(ex)[:200]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etas", default="0.3,0.5")
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--max_Q", type=int, default=14)
    ap.add_argument("--max_H", type=int, default=8)
    ap.add_argument("--cpsat_time", type=float, default=120)
    ap.add_argument("--no_sos4", action="store_true")
    ap.add_argument("--only", default=None, help="substring filter on the group name")
    ap.add_argument("--out", default="../results/group_census.jsonl")
    a = ap.parse_args()
    etas = [float(x) for x in a.etas.split(",")]
    out = open(a.out, "a")
    cands = candidate_groups()
    if a.only:
        cands = [c for c in cands if a.only.lower() in c[0].lower()]
    cands.sort(key=lambda c: c[1].n)
    for name, G, Hlist, gens in cands:
        print(f"[{name}] order {G.n}, abelian={G.is_abelian()}", flush=True)
        try:
            G.check()
        except AssertionError as ex:
            print(f"!! {name}: group check failed: {ex}", flush=True); continue
        ab = G.is_abelian()
        if Hlist is None:
            Hlist = []
            for order in (2, 3, 4, 5, 8):
                if order > a.max_H or G.n % order:
                    continue
                Hlist += [H for H in G.normal_subgroups(order=order, max_count=4)]
        if gens is None:
            gens = list(range(1, min(G.n, 6)))
        for H in Hlist:
            N, nQ = len(H), G.n // len(H)
            if N < 2 or nQ < 4 or nQ > a.max_Q or N > a.max_H:
                continue
            for eta in etas:
                try:
                    mu = word_noise(G, gens, eta, radius=a.radius, H=H)
                    if mu.sum() <= 0 or not np.isfinite(mu).all():
                        continue
                    g = group_unique_game(G, H, mu, name=f"{name} |H|={N} eta={eta}")
                    if g.m == 0:
                        continue
                    # verify the transversal identity on a random labelling
                    rng = np.random.default_rng(0)
                    L = rng.integers(0, g.k, size=g.n)
                    T = labelling_to_transversal(G, H, L)
                    assert abs(transversal_value(G, H, mu, T) - g.value(L)) < 1e-10
                    print(f"  -> {name} |H|={N} eta={eta}: n={g.n} k={g.k} m={g.m} "
                          f"search={g.k}^{g.n} D_lasserre2={1 + g.n*g.k + (g.n*(g.n-1)//2)*g.k**2}", flush=True)
                    rec = analyse(g, tag=f"{name}|H|={N}|eta={eta}", cpsat_time=a.cpsat_time,
                                  do_sos4=not a.no_sos4)
                    rec["abelian"] = bool(ab); rec["group_order"] = int(G.n); rec["eta"] = eta
                    rec["radius"] = a.radius
                    print(json.dumps({kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                      for kk, vv in rec.items()
                                      if kk in ("tag", "abelian", "n", "k", "m", "opt", "opt_certified",
                                                "sdp_lo", "sdp_hi", "C2", "sos4", "C4", "sandwich_ok",
                                                "lasserre2_D")}), flush=True)
                    out.write(json.dumps(rec) + "\n"); out.flush()
                except Exception as ex:
                    print(f"!! {name} |H|={N} eta={eta}: {type(ex).__name__}: {str(ex)[:160]}", flush=True)


if __name__ == "__main__":
    main()
