"""Exact, machine-checkable certificates for the Paley degree-4 question.

`paley_exact.py` reduces the question to: does there exist q with

    M(q) = M0 + sum_k q_k A_k  psd ?

where M0 carries the forced entries (1 on the diagonal, X elsewhere) and A_k is
the 0/1 indicator of Aut-orbit k of 4-subsets.  Crucially the A_k have PAIRWISE
DISJOINT supports and ZERO DIAGONAL, which is what makes exact certification easy.

INFEASIBILITY (this file).  By the theorem of alternatives, no q works iff there
is Y with

    (1) Y psd,   (2) <Y, A_k> = 0 for every k,   (3) <Y, M0> < 0.

Such a Y is a finite object that can be checked in exact arithmetic, so it turns
"the solver says infeasible" into a proof.  The construction:

  * solve min <Y,M0> s.t. Y psd, <Y,A_k> = 0, tr Y = n  numerically;
  * add epsilon*I.  This keeps (2) exactly, because every A_k has zero diagonal,
    and moves (3) by only epsilon*tr(M0) = epsilon*n, while making Y strictly
    positive definite with margin epsilon;
  * round to rationals with denominator D;
  * restore (2) EXACTLY: <Y,A_k> is a sum over the disjoint support of A_k, so
    subtracting <Y,A_k>/|supp A_k| from each entry there fixes orbit k without
    touching any other orbit or the diagonal;
  * verify (1) by exact rational LDL^T (all pivots > 0),
    (2) by exact summation,
    (3) in Q(sqrt p): <Y,M0> = A + B*sqrt(p) with A, B rational, and the sign of
    A + B*sqrt(p) is decided exactly by comparing A^2 with B^2*p.

Nothing in the verification uses floating point.

Run: python paley_certify.py 13 17
"""
import argparse
import itertools
import json
import os
from fractions import Fraction

import numpy as np

from paley_exact import aut_orbits4, index_pairs, legendre

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------- exact M0
def exact_M0_parts(p):
    """M0 = R + sqrt(p) * S with R, S exact rational matrices."""
    chi = legendre(p)
    pairs, pidx = index_pairs(p)
    n = 1 + len(pairs)
    R = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    S = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    d = Fraction(1, p - 1)

    def setX(i, j, a, b):
        R[i][j] = R[j][i] = -d
        S[i][j] = S[j][i] = -d * int(chi[(a - b) % p])

    R[0][0] = Fraction(1)
    for e in pairs:
        i = pidx[frozenset(e)]
        R[i][i] = Fraction(1)
        setX(0, i, e[0], e[1])
    for e, f in itertools.combinations(pairs, 2):
        se, sf = set(e), set(f)
        if len(se & sf) == 1:
            (a,), (b,) = tuple(se - sf), tuple(sf - se)
            setX(pidx[frozenset(e)], pidx[frozenset(f)], a, b)
    return R, S, pairs, pidx, n


def orbit_supports(p, pairs, pidx):
    """For each Aut-orbit k, the list of (i,j) off-diagonal index pairs it owns."""
    orbit_of, reps = aut_orbits4(p)
    sup = [[] for _ in reps]
    for e, f in itertools.combinations(pairs, 2):
        se, sf = set(e), set(f)
        if se & sf:
            continue
        k = orbit_of[frozenset(se | sf)]
        sup[k].append((pidx[frozenset(e)], pidx[frozenset(f)]))
    return sup, reps


# ------------------------------------------------------------ exact linalg
def ldl_positive_definite(Yf, n):
    """Exact LDL^T over Q. Returns (True, min_pivot) iff every pivot is > 0."""
    a = [row[:] for row in Yf]
    piv = []
    for k in range(n):
        d = a[k][k]
        if d <= 0:
            return False, d
        piv.append(d)
        inv = Fraction(1) / d
        for i in range(k + 1, n):
            f = a[i][k] * inv
            if f:
                for j in range(k, n):
                    a[i][j] -= f * a[k][j]
        for i in range(k + 1, n):
            a[k][i] = a[i][k]
    return True, min(piv)


def sign_a_plus_b_sqrt(A, B, p):
    """Exact sign of A + B*sqrt(p) for rational A, B."""
    if B == 0:
        return (A > 0) - (A < 0)
    if A == 0:
        return (B > 0) - (B < 0)
    if A > 0 and B > 0:
        return 1
    if A < 0 and B < 0:
        return -1
    # opposite signs: compare A^2 with B^2 p
    lhs, rhs = A * A, B * B * p
    if lhs == rhs:
        return 0
    bigger_is_A = lhs > rhs
    return (1 if A > 0 else -1) if bigger_is_A else (1 if B > 0 else -1)


# --------------------------------------------------------------------------- #
def certify_infeasible(p, D=10 ** 7, verbose=True):
    import cvxpy as cp
    R, S, pairs, pidx, n = exact_M0_parts(p)
    sup, reps = orbit_supports(p, pairs, pidx)
    M0 = np.array([[float(R[i][j]) + np.sqrt(p) * float(S[i][j]) for j in range(n)]
                   for i in range(n)])

    # ---- numerical dual
    Y = cp.Variable((n, n), symmetric=True)
    cons = [Y >> 0, cp.trace(Y) == n]
    for k, s in enumerate(sup):
        cons.append(cp.sum(cp.hstack([Y[i, j] for (i, j) in s])) == 0)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(Y, M0))), cons)
    prob.solve(solver=cp.SCS, eps=1e-10, max_iters=200000)
    Yv = np.array(Y.value)
    Yv = (Yv + Yv.T) / 2
    v = float(np.sum(Yv * M0))
    if verbose:
        print(f"p = {p}: numerical dual  <Y,M0> = {v:+.6e}   (negative means infeasible)")
    if v >= 0:
        return dict(p=p, proved=False, reason="numerical dual not negative", value=v)

    # ---- make strictly definite; A_k have zero diagonal so (2) is untouched
    lam = float(np.linalg.eigvalsh(Yv).min())
    eps = min(-v / (4 * n), 1e-3)
    Yv = Yv + (eps - min(lam, 0)) * np.eye(n)

    # ---- round to rationals
    Yf = [[Fraction(int(round(Yv[i][j] * D)), D) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            Yf[j][i] = Yf[i][j]

    # ---- restore <Y,A_k> = 0 exactly (disjoint, off-diagonal supports)
    for s in sup:
        tot = sum(Yf[i][j] for (i, j) in s)
        if tot:
            adj = tot / len(s)
            for (i, j) in s:
                Yf[i][j] -= adj
                Yf[j][i] = Yf[i][j]

    # ---- exact checks
    resid = max(abs(sum(Yf[i][j] for (i, j) in s)) for s in sup)
    pd, minpiv = ldl_positive_definite(Yf, n)
    A = sum(Yf[i][j] * R[i][j] for i in range(n) for j in range(n))
    B = sum(Yf[i][j] * S[i][j] for i in range(n) for j in range(n))
    sg = sign_a_plus_b_sqrt(A, B, p)

    ok = (resid == 0) and pd and (sg < 0)
    if verbose:
        print(f"  (2) max |<Y,A_k>|      : {resid}  {'EXACT ZERO' if resid == 0 else 'FAIL'}")
        print(f"  (1) Y positive definite: {pd}   min pivot {float(minpiv):.3e}")
        print(f"  (3) <Y,M0> = A + B*sqrt(p), A = {float(A):+.6f}, B = {float(B):+.6f}, "
              f"sign = {sg}")
        print(f"  ==> {'PROVED INFEASIBLE' if ok else 'certificate FAILED'}")
    return dict(p=p, proved=bool(ok), n=n, n_orbits=len(reps), denom=D,
                constraint_residual=str(resid), positive_definite=bool(pd),
                min_pivot=float(minpiv), A=float(A), B=float(B), sign=int(sg),
                numerical_dual_value=v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primes", nargs="*", type=int, default=[13, 17])
    ap.add_argument("--denom", type=int, default=10 ** 7)
    args = ap.parse_args()
    out = []
    for p in args.primes:
        r = certify_infeasible(p, D=args.denom)
        out.append(r)
        print()
    dst = os.path.join(ROOT, "results", "paley_certify.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    for r in out:
        if r["proved"]:
            print(f"p = {r['p']}: SoS_4 < SoS_2 is PROVED (exact dual certificate, "
                  f"{r['n']}x{r['n']}, denominator {r['denom']})")


if __name__ == "__main__":
    main()
