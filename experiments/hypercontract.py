"""
hypercontract.py -- does the degree-2 gap survive degree 4 exactly when the
carrier eigenspace is NOT hypercontractive?

Barak-Brandao-Harrow-Kelner-Steurer-Zhou (2012) show that degree-4 SoS certifies
small-set expansion / refutes KV-type gaps whenever the top eigenspace of the
(label-extended) graph is 2->4 hypercontractive: every f in the eigenspace has
||f||_4 <= C ||f||_2.  Contrapositive, quantified: for a vertex-transitive signed
instance with top eigenspace V of the signed adjacency (which carries the
degree-2 gap), define

    H(V) = n * max_{f in V, ||f||_2 = 1} sum_v f_v^4      (1 = flat, n = a delta)
    mult = dim V.

Hypothesis H14: retention rho = (C_4 - 1)/(C_2 - 1) increases with H(V) (and with
mult), and instances with H(V) = O(1) have rho ~ 0.  This turns the sweep data
into a test of the mechanism, and H(V) into a cheap proxy objective for
searching large instances.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def signed_adjacency(n, E, B, W):
    A = np.zeros((n, n))
    for (u, v), b, w in zip(E.tolist(), B.tolist(), W.tolist()):
        A[u, v] += b * w; A[v, u] += b * w
    return A


def top_eigenspace(A, tol=1e-7):
    lam, V = np.linalg.eigh(A)
    top = lam[-1]
    idx = np.where(lam > top - tol * max(1.0, abs(top)))[0]
    return top, V[:, idx]


def hypercontractivity(V, starts=30, iters=500, seed=0):
    """n * max over unit f in span(V) of sum f^4, by the (monotone) higher-order power method."""
    n, m = V.shape
    rng = np.random.default_rng(seed)
    best = 0.0
    inits = [rng.standard_normal(m) for _ in range(starts)] + [V[v] for v in range(min(n, 10))]
    for c in inits:
        c = c / np.linalg.norm(c)
        for _ in range(iters):
            f = V @ c
            g = V.T @ (f ** 3)
            nc = g / np.linalg.norm(g)
            if np.linalg.norm(nc - c) < 1e-12:
                c = nc; break
            c = nc
        f = V @ c
        best = max(best, float(n * np.sum(f ** 4)))
    return best


def analyse_instance(n, E, B, W):
    A = signed_adjacency(n, E, B, W)
    deg = A.__abs__().sum(1)
    top, V = top_eigenspace(A)
    sos2_eig = 0.5 + 0.5 * top / deg.mean() if np.allclose(deg, deg.mean()) else None
    return {"mult": V.shape[1], "H": hypercontractivity(V), "sos2_eig": sos2_eig, "lambda_top": top / deg.mean()}


def group_rows(path="../results/group_sweep.jsonl"):
    from group_library import library
    from group_sos import GroupSoS
    rows = [json.loads(l) for l in open(path) if l.strip()]
    lib = {G.name: G for G in library(130)}
    cache = {}
    out = []
    for r in rows:
        if not (r["proved"] and r["cert_width"] < 1e-4 and r["retention"] is not None):
            continue
        G = lib[r["group"]]
        if G.name not in cache:
            cache[G.name] = GroupSoS(G, device="cpu")
        P = cache[G.name]
        P.set_instance(r["gens"], signs=r["signs"])
        E, B, W = P.edges()
        a = analyse_instance(G.n, np.asarray(E), np.asarray(B, float), np.asarray(W, float))
        out.append({**{k: r[k] for k in ("group", "order", "gens", "signs", "carrier_dim", "C2", "C4_lo", "retention", "sos2")}, **a})
    return out


def circulant_rows(Lmax=21, iters=20000, device="cuda"):
    """Recompute certified C_4 for all circulant instances with <= 2 classes, L odd <= Lmax."""
    import itertools, torch
    from circulant_sos import CirculantSoS
    from circulant_scan import sos2_circulant
    from weight_ascent import cpsat_min_unsat
    out = []
    for L in range(7, Lmax + 1, 2):
        h = (L - 1) // 2
        P = CirculantSoS(L, device=device)
        for k in (1, 2):
            for gens in itertools.combinations(range(1, h + 1), k):
                for signs in itertools.product([-1.0, 1.0], repeat=k):
                    if all(s > 0 for s in signs):
                        continue
                    P.set_instance(list(gens), signs=list(signs))
                    P.solve(iters=iters, tol=1e-10); lo, hi = P.certified_bounds()
                    E, B, W = [], [], []
                    for g, s in zip(gens, signs):
                        for v in range(L):
                            E.append((v, (v + g) % L)); B.append(s); W.append(1.0)
                    E = np.array(E); B = np.array(B); W = np.array(W) / len(W)
                    xs, unsat, unsat_ub, pr = cpsat_min_unsat(L, E, B, W, time_limit=60)
                    # exact SoS2 from the eigenvalues of the signed circulant
                    a = analyse_instance(L, E, B, W)
                    s2 = a["sos2_eig"]
                    C2 = unsat / (1 - s2) if s2 < 1 - 1e-12 else None
                    C4 = unsat / (1 - lo)
                    rec = {"L": L, "gens": list(gens), "signs": list(signs), "opt": 1 - unsat, "proved": pr,
                           "sos4_lo": lo, "sos4_hi": hi, "cert_width": hi - lo, "sos2": s2, "C2": C2, "C4_lo": C4,
                           "retention": (C4 - 1) / (C2 - 1) if C2 and C2 > 1 + 1e-9 else None, **a}
                    out.append(rec)
                    print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "group"
    if which == "group":
        rows = group_rows()
        json.dump(rows, open("../results/hypercontract_group.json", "w"), indent=1)
    else:
        rows = circulant_rows(Lmax=int(sys.argv[2]) if len(sys.argv) > 2 else 21)
        json.dump(rows, open("../results/hypercontract_circulant.json", "w"), indent=1)
    # digest
    good = [r for r in rows if r["retention"] is not None and r["C2"] > 1.02]
    print(f"\n{len(rows)} instances, {len(good)} with a real degree-2 gap (C2 > 1.02)")
    if good:
        H = np.array([r["H"] for r in good]); ret = np.array([r["retention"] for r in good])
        mult = np.array([r["mult"] for r in good])
        print(f"corr(retention, H) = {np.corrcoef(ret, H)[0,1]:.3f}   corr(retention, log H) = {np.corrcoef(ret, np.log(H))[0,1]:.3f}"
              f"   corr(retention, mult) = {np.corrcoef(ret, mult)[0,1]:.3f}")
        for lo_, hi_ in ((1, 1.5), (1.5, 2.5), (2.5, 4), (4, 8), (8, 1e9)):
            sel = (H >= lo_) & (H < hi_)
            if sel.any():
                print(f"  H in [{lo_},{hi_}): n={sel.sum():3d}  mean retention {ret[sel].mean():.4f}  max {ret[sel].max():.4f}  "
                      f"frac(ret>0.05) {(ret[sel] > 0.05).mean():.2f}")
