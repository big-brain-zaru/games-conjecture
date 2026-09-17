"""The Paley degree-4 question, reduced to a small finite feasibility SDP.

REDUCTION (proved in the docstring below, validated numerically by --validate).

For Max-Cut on the Paley graph P_p the degree-2 optimum is attained exactly on

    F = { X psd, diag(X) = 1, range(X) contained in E_min },

E_min the lambda_min eigenspace of the adjacency matrix, of dimension (p-1)/2,
because min <A,X> >= lambda_min * tr X with equality iff range(X) is in E_min.
F is convex and Aut(P_p)-invariant, so if ANY X in F is degree-4 extendable then
so is the Aut-average of that X.  The Aut-invariant elements of F are the
matrices c1*I + r*(residue adjacency) + s*(non-residue adjacency) whose range
lies in E_min, which forces a multiple of the projection.  Hence:

    SoS_4(P_p) = SoS_2(P_p)   <=>   the SINGLE matrix

        X_ij = (-1 - sqrt(p)*chi(i-j)) / (p-1),   X_ii = 1

    is degree-4 extendable, and the extension may be taken Aut-invariant too.

Setting all odd moments to zero (valid: the instance is invariant under global
sign flip, so the odd part of any solution can be averaged away), the level-2
moment matrix splits as M_odd (+) M_even with M_odd = X, already psd.  M_even is
indexed by {empty} union {pairs} with

    M[0,0] = 1,  M[0,{i,j}] = X_ij,
    M[{i,j},{k,l}] = 1                  if {i,j} = {k,l}
                   = X_{the two outer}  if they share exactly one point
                   = q({i,j,k,l})       if disjoint

and q, one free number per 4-subset, is the ONLY unknown.  So

    SoS_4(P_p) = SoS_2(P_p)  <=>  exists q with M_even(q) psd,

a feasibility SDP with (1 + C(p,2)) rows and, after imposing Aut-invariance,
one variable per Aut-orbit of 4-subsets: 65 at p = 29, 109 at 37, 134 at 41.

That is small.  Solving max t s.t. M_even(q) - t*I psd decides it:
  t* > 0  -> feasible with room to spare, so a nearby RATIONAL q also works and
             the positive answer can be certified exactly (see paley_certify.py);
  t* = 0  -> feasible only on the boundary;
  t* < 0  -> infeasible, and the dual gives a certificate of infeasibility.

Run:
  python paley_exact.py --validate 29     # check the reduction against the ADMM solution
  python paley_exact.py 13 17 29          # decide each p
"""
import argparse
import itertools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------- #
def legendre(p):
    chi = np.zeros(p, dtype=int)
    qr = set(pow(a, 2, p) for a in range(1, p))
    for k in range(1, p):
        chi[k] = 1 if k in qr else -1
    return chi


def canonical_X(p):
    chi = legendre(p)
    X = np.eye(p)
    for i in range(p):
        for j in range(p):
            if i != j:
                X[i, j] = (-1 - np.sqrt(p) * chi[(i - j) % p]) / (p - 1)
    return X


def aut_orbits4(p):
    """Aut(P_p)-orbits of 4-subsets; returns (orbit_of: frozenset->k, reps)."""
    QR = sorted(set(pow(a, 2, p) for a in range(1, p)))
    aut = [(a, b) for a in QR for b in range(p)]
    orbit_of, reps = {}, []
    for S in itertools.combinations(range(p), 4):
        fs = frozenset(S)
        if fs in orbit_of:
            continue
        k = len(reps)
        for (a, b) in aut:
            orbit_of[frozenset((a * x + b) % p for x in S)] = k
        reps.append(tuple(sorted(S)))
    return orbit_of, reps


def index_pairs(p):
    pairs = list(itertools.combinations(range(p), 2))
    return pairs, {frozenset(e): i + 1 for i, e in enumerate(pairs)}   # row 0 = empty set


def build_parts(p):
    """M_even(q) = M0 + sum_k q_k A_k, with A_k the indicator of Aut-orbit k."""
    X = canonical_X(p)
    pairs, pidx = index_pairs(p)
    n = 1 + len(pairs)
    M0 = np.zeros((n, n))
    M0[0, 0] = 1.0
    for e in pairs:
        i = pidx[frozenset(e)]
        M0[0, i] = M0[i, 0] = X[e[0], e[1]]
        M0[i, i] = 1.0
    for e, f in itertools.combinations(pairs, 2):
        se, sf = set(e), set(f)
        common = se & sf
        if len(common) == 1:
            (a,), (b,) = tuple(se - sf), tuple(sf - se)
            v = X[a, b]
            ie, if_ = pidx[frozenset(e)], pidx[frozenset(f)]
            M0[ie, if_] = M0[if_, ie] = v
    orbit_of, reps = aut_orbits4(p)
    A = [np.zeros((n, n)) for _ in reps]
    for e, f in itertools.combinations(pairs, 2):
        se, sf = set(e), set(f)
        if se & sf:
            continue
        k = orbit_of[frozenset(se | sf)]
        ie, if_ = pidx[frozenset(e)], pidx[frozenset(f)]
        A[k][ie, if_] = A[k][if_, ie] = 1.0
    return M0, A, reps, X, pairs, pidx


# --------------------------------------------------------------------------- #
def validate(p):
    """Check the reduction against the independently computed ADMM solution."""
    from circulant_sos import CirculantSoS, orbit_key
    print(f"=== validating the reduction at p = {p} ===")
    chi = legendre(p)
    H = sorted(set(min(x, p - x) for x in range(1, p) if chi[x] == 1))
    ypath = os.path.join(ROOT, "results", f"paley_moments_{p}_y.npy")
    if not os.path.exists(ypath):
        raise SystemExit(f"need {ypath}; run paley_moments.py {p}")
    C = CirculantSoS(p, device="cpu").set_instance(H)
    y = np.load(ypath); y = y / y[0]

    M0, A, reps, X, pairs, pidx = build_parts(p)

    # (a) the ADMM degree-2 part must equal the canonical X
    d2 = max(abs(y[C.orbs[orbit_key({0, d}, p)]] -
                 (-1 - np.sqrt(p) * chi[d]) / (p - 1)) for d in range(1, p))
    print(f"  degree-2 part vs canonical X            : max dev {d2:.3e}")

    # (b) plug the ADMM 4-set moments into M_even and test psd
    q = np.array([y[C.orbs[orbit_key(set(S), p)]] for S in reps])
    M = M0 + sum(qk * Ak for qk, Ak in zip(q, A))
    lam = float(np.linalg.eigvalsh(M).min())
    print(f"  lambda_min of M_even at the ADMM point  : {lam:+.3e}")

    # (c) the objective this X gives must be the closed-form SoS_2
    m = p * (p - 1) / 4
    A_adj = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i != j and chi[(i - j) % p] == 1:
                A_adj[i, j] = 1
    val = (A_adj * (1 - X)).sum() / 4 / m
    closed = 0.5 + (1 + np.sqrt(p)) / (2 * (p - 1))
    print(f"  objective at canonical X                : {val:.12f}")
    print(f"  closed-form SoS_2                       : {closed:.12f}   dev {abs(val-closed):.2e}")
    print(f"  M_even size {M.shape[0]} x {M.shape[0]}, {len(reps)} Aut-orbit variables")
    ok = d2 < 1e-8 and lam > -1e-6 and abs(val - closed) < 1e-12
    print(f"  REDUCTION {'VALIDATED' if ok else 'FAILED'}")
    return ok


# --------------------------------------------------------------------------- #
def decide(p, solver="SCS", verbose=False):
    import cvxpy as cp
    M0, A, reps, X, pairs, pidx = build_parts(p)
    n = M0.shape[0]
    q = cp.Variable(len(reps))
    t = cp.Variable()
    M = M0 + sum(q[k] * A[k] for k in range(len(reps)))
    prob = cp.Problem(cp.Maximize(t), [M - t * np.eye(n) >> 0])
    kw = dict(verbose=verbose)
    if solver == "SCS":
        kw.update(eps=1e-10, max_iters=200000)
    prob.solve(solver=getattr(cp, solver), **kw)
    tv = float(t.value)
    qv = np.array(q.value).ravel()
    Mv = M0 + sum(qv[k] * A[k] for k in range(len(reps)))
    lam = float(np.linalg.eigvalsh(Mv).min())
    return dict(p=p, n=n, n_orbits=len(reps), t_star=tv, lam_min_at_q=lam,
                status=prob.status, q=[float(v) for v in qv],
                reps=[list(r) for r in reps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primes", nargs="*", type=int, default=[13, 17, 29])
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--solver", default="SCS")
    args = ap.parse_args()

    if args.validate:
        validate(args.validate)
        return

    out = []
    for p in args.primes:
        r = decide(p, solver=args.solver)
        verdict = ("FEASIBLE with interior" if r["t_star"] > 1e-7 else
                   "FEASIBLE on the boundary" if r["t_star"] > -1e-7 else
                   "INFEASIBLE")
        print(f"p = {p:3d}  rows {r['n']:5d}  vars {r['n_orbits']:4d}  "
              f"t* = {r['t_star']:+.3e}  lam_min {r['lam_min_at_q']:+.3e}  -> {verdict}")
        r["verdict"] = verdict
        out.append(r)
        with open(os.path.join(ROOT, "results", f"paley_exact_{p}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(r, f, indent=1)
    print()
    print("t* > 0 means a whole ball of q works, so a rational q does too and the")
    print("positive answer can be made exact.  t* < 0 means no q works at all.")


if __name__ == "__main__":
    main()
