"""Exact, machine-checkable proofs that SoS_4 < SoS_2 on P_13 and P_17.

`paley_exact.py` reduces the question to: does there exist q with

    M(q) = M0 + sum_k q_k A_k   psd ?

where M0 carries the forced entries and A_k is the 0/1 indicator of Aut-orbit k
of 4-subsets.  The A_k have PAIRWISE DISJOINT supports and ZERO DIAGONAL, which
is what makes exact certification easy.

By the theorem of alternatives, no q works iff there is a Y with

    (1) Y psd,   (2) <Y, A_k> = 0 for every k,   (3) <Y, M0> < 0.

Such a Y is a finite object, so it turns "the solver says infeasible" into a
proof.  Everything in the VERIFICATION below is integer arithmetic; no floating
point is involved once Y has been rounded.

  * solve the dual numerically, then add epsilon*I.  That keeps (2) exactly,
    because every A_k has zero diagonal, and moves (3) by only epsilon*tr(M0),
    while making Y strictly definite with margin epsilon.
  * scale by D and round to the integer matrix Z.
  * restore (2) EXACTLY by integer redistribution: <Z,A_k> is a sum over the
    disjoint support of A_k, so subtracting floor(r/L) from every entry there and
    one more from r mod L of them makes the sum exactly zero, touching no other
    orbit and no diagonal entry.
  * verify (1) by the Bareiss fraction-free elimination on Z: all leading
    principal minors positive, computed in exact integer arithmetic.  (An earlier
    version used rational LDL and blew up; the minors here stay integers.)
  * verify (3) exactly: (p-1)*<Z,M0> = A + B*sqrt(p) with A, B INTEGERS, because
    every entry of M0 is 1 or (-1 -+ sqrt p)/(p-1).  The sign of A + B*sqrt(p) is
    then decided by comparing A^2 with B^2*p, in integers.

Run: python paley_certify.py 13 17
"""
import argparse
import itertools
import json
import os

import numpy as np

from paley_exact import aut_orbits4, index_pairs, legendre

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------- exact M0, scaled by p-1
def exact_M0_parts(p):
    """(p-1)*M0 = R + sqrt(p) * S with R, S INTEGER matrices."""
    chi = legendre(p)
    pairs, pidx = index_pairs(p)
    n = 1 + len(pairs)
    R = [[0] * n for _ in range(n)]
    S = [[0] * n for _ in range(n)]

    def setX(i, j, a, b):
        R[i][j] = R[j][i] = -1
        S[i][j] = S[j][i] = -int(chi[(a - b) % p])

    R[0][0] = p - 1
    for e in pairs:
        i = pidx[frozenset(e)]
        R[i][i] = p - 1
        setX(0, i, e[0], e[1])
    for e, f in itertools.combinations(pairs, 2):
        se, sf = set(e), set(f)
        if len(se & sf) == 1:
            (a,), (b,) = tuple(se - sf), tuple(sf - se)
            setX(pidx[frozenset(e)], pidx[frozenset(f)], a, b)
    return R, S, pairs, pidx, n


def orbit_supports(p, pairs, pidx):
    orbit_of, reps = aut_orbits4(p)
    sup = [[] for _ in reps]
    for e, f in itertools.combinations(pairs, 2):
        se, sf = set(e), set(f)
        if se & sf:
            continue
        sup[orbit_of[frozenset(se | sf)]].append(
            (pidx[frozenset(e)], pidx[frozenset(f)]))
    return sup, reps


# ------------------------------------------------------------- exact integer PD
def bareiss_positive_definite(Z, n):
    """Exact integer test: all leading principal minors of Z are > 0."""
    M = [row[:] for row in Z]
    prev = 1
    minors = []
    for k in range(n):
        if M[k][k] <= 0:
            return False, k, minors
        minors.append(M[k][k])
        pk = M[k][k]
        for i in range(k + 1, n):
            Mik = M[i][k]
            if Mik:
                rowi, rowk = M[i], M[k]
                for j in range(k + 1, n):
                    rowi[j] = (rowi[j] * pk - Mik * rowk[j]) // prev
            else:
                rowi = M[i]
                for j in range(k + 1, n):
                    rowi[j] = (rowi[j] * pk) // prev
        prev = pk
    return True, n, minors


def sign_a_plus_b_sqrt(A, B, p):
    if B == 0:
        return (A > 0) - (A < 0)
    if A == 0:
        return (B > 0) - (B < 0)
    if A > 0 and B > 0:
        return 1
    if A < 0 and B < 0:
        return -1
    lhs, rhs = A * A, B * B * p
    if lhs == rhs:
        return 0
    return (1 if A > 0 else -1) if lhs > rhs else (1 if B > 0 else -1)


# --------------------------------------------------------------------------- #
def certify_infeasible(p, D=10 ** 6, verbose=True):
    import cvxpy as cp
    R, S, pairs, pidx, n = exact_M0_parts(p)
    sup, reps = orbit_supports(p, pairs, pidx)
    M0 = np.array([[(R[i][j] + np.sqrt(p) * S[i][j]) / (p - 1) for j in range(n)]
                   for i in range(n)])

    Y = cp.Variable((n, n), symmetric=True)
    cons = [Y >> 0, cp.trace(Y) == n]
    for s in sup:
        cons.append(cp.sum(cp.hstack([Y[i, j] for (i, j) in s])) == 0)
    prob = cp.Problem(cp.Minimize(cp.sum(cp.multiply(Y, M0))), cons)
    prob.solve(solver=cp.SCS, eps=1e-9, max_iters=100000)
    Yv = np.array(Y.value); Yv = (Yv + Yv.T) / 2
    v = float(np.sum(Yv * M0))
    if verbose:
        print(f"p = {p}: n = {n}, {len(reps)} orbits; numerical dual <Y,M0> = {v:+.6e}")
    if v >= 0:
        return dict(p=p, proved=False, reason="dual not negative", value=v)

    lam = float(np.linalg.eigvalsh(Yv).min())
    eps = min(-v / (4 * n), 1e-2)
    Yv = Yv + (eps - min(lam, 0.0)) * np.eye(n)

    Z = [[int(round(Yv[i][j] * D)) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            Z[j][i] = Z[i][j]

    # restore <Z, A_k> = 0 exactly, in integers
    for s in sup:
        L = len(s)
        r = sum(Z[i][j] for (i, j) in s)
        a, rem = r // L, r - (r // L) * L
        for t, (i, j) in enumerate(s):
            Z[i][j] -= a + (1 if t < rem else 0)
            Z[j][i] = Z[i][j]

    resid = max(abs(sum(Z[i][j] for (i, j) in s)) for s in sup)
    pd, upto, minors = bareiss_positive_definite(Z, n)
    A = sum(Z[i][j] * R[i][j] for i in range(n) for j in range(n))
    B = sum(Z[i][j] * S[i][j] for i in range(n) for j in range(n))
    sg = sign_a_plus_b_sqrt(A, B, p)
    ok = (resid == 0) and pd and (sg < 0)

    if verbose:
        print(f"  (2) max |<Z,A_k>|        : {resid}  {'EXACT ZERO' if resid == 0 else 'FAIL'}")
        print(f"  (1) Bareiss minors > 0   : {pd} (checked {upto} of {n}; "
              f"last minor has {len(str(abs(minors[-1]))) if minors else 0} digits)")
        print(f"  (3) (p-1)<Z,M0> = A + B*sqrt(p), A = {A}, B = {B}, sign = {sg}")
        print(f"  ==> {'PROVED: SoS_4 < SoS_2 at p = %d' % p if ok else 'certificate FAILED'}")
    return dict(p=p, proved=bool(ok), n=n, n_orbits=len(reps), denom=D,
                constraint_residual=int(resid), bareiss_pd=bool(pd),
                A=str(A), B=str(B), sign=int(sg), numerical_dual_value=v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primes", nargs="*", type=int, default=[13, 17])
    ap.add_argument("--denom", type=int, default=10 ** 6)
    args = ap.parse_args()
    out = []
    for p in args.primes:
        out.append(certify_infeasible(p, D=args.denom))
        print()
    dst = os.path.join(ROOT, "results", "paley_certify.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
