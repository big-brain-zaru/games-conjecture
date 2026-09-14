"""
kv_transversal.py -- The Khot-Vishnoi game U_{k,eta} as a transversal problem,
and its exact solution.

Reformulation (proved below in _selftest against the generic instance value).
Let C = {bitmask of chi_S : S subset [k]} < F_2^N, N = 2^k.  C is linear of
dimension k (chi_S . chi_T = chi_{S xor T}), so F_2^N / C has 2^N / N cosets --
exactly the vertices (classes) of the KV game, and a labelling is a choice of one
function per coset, i.e. a TRANSVERSAL T of C in F_2^N.

Key identity: for f = rep_i chi_S and g = rep_j chi_{S xor c}, the bitmask
difference is  f xor g = rep_i xor rep_j xor chi_c, which is CONSTANT over the
whole edge bundle (i, j, c).  Hence every bundle has weight N * wt'(f xor g) and

    value(T) = N * sum_{ {f,g} subset T } wt'(f xor g)  /  W ,
    W = sum over all unordered pairs at admissible distance of wt'.

So the KV optimum is a maximum-weight transversal problem: 2^N/N variables
(cosets), N values each (which element of the coset), pairwise costs depending
only on the difference.  For k=3: 32 variables, 8 values, 496 pairwise tables.
Symmetry: T -> T xor z for z in F_2^N (order 2^N, reduced by fixing 0 in T) and
the point permutations GL(k,2) (order 168 for k=3).

Exact solver: CP-SAT with one table constraint per coset pair.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import numpy as np


def kv_transversal_data(k: int, eta: float, d_window=None):
    N = 1 << k
    n_f = 1 << N
    # C: bitmask of chi_S, bit x set iff chi_S(x) = -1 iff |S cap x| odd
    C = []
    for S in range(N):
        m = 0
        for x in range(N):
            if bin(S & x).count("1") % 2 == 1:
                m |= (1 << x)
        C.append(m)
    C = np.array(sorted(C), dtype=np.int64)
    assert len(set(C.tolist())) == N
    # cosets
    coset = -np.ones(n_f, dtype=np.int64)
    reps = []
    for f in range(n_f):
        if coset[f] < 0:
            c = len(reps); reps.append(f)
            coset[C ^ f] = c
    reps = np.array(reps)
    n_cos = len(reps)
    # weights
    d_of = np.array([bin(x).count("1") for x in range(n_f)])
    if d_window is None:
        dmin = max(1, int(math.ceil(eta * N / 2))); dmax = min(N, int(math.floor(2 * eta * N)))
    else:
        dmin, dmax = d_window
    W = sum(math.comb(N, d) * eta ** d * (1 - eta) ** (N - d) for d in range(dmin, dmax + 1))
    wt = np.where((d_of >= dmin) & (d_of <= dmax),
                  2.0 * 2.0 ** -N * eta ** d_of * (1 - eta) ** (N - d_of), 0.0)
    return dict(k=k, N=N, n_f=n_f, C=C, coset=coset, reps=reps, n_cos=n_cos,
                wt=wt, W=W, dmin=dmin, dmax=dmax, d_of=d_of, eta=eta)


def transversal_value(D, choice: np.ndarray) -> float:
    """choice[i] in [0, N): the coset i is represented by reps[i] ^ C[choice[i]].
    Returns the KV game value of the corresponding labelling."""
    N, reps, C, wt, W = D["N"], D["reps"], D["C"], D["wt"], D["W"]
    T = reps ^ C[choice]
    tot = 0.0
    for a in range(len(T)):
        tot += wt[T[a] ^ T[a + 1:]].sum()
    return float(N * tot / W)


def subcube_choice(D) -> np.ndarray:
    """The subcube transversal: functions that are +1 at the k unit points."""
    k, N, reps, C = D["k"], D["N"], D["reps"], D["C"]
    mask = sum(1 << (1 << i) for i in range(k))
    out = np.zeros(len(reps), dtype=np.int64)
    for i, r in enumerate(reps):
        cand = np.where(((r ^ C) & mask) == 0)[0]
        assert len(cand) == 1, (i, len(cand))
        out[i] = cand[0]
    return out


def cpsat_transversal(D, time_limit=3600, workers=12, hint=None, verbose=False, scale=None,
                      break_gl=True):
    """Exact maximum-weight transversal by CP-SAT.  Integer costs are exact:
    wt(z) = 2 * 2^-N * eta^d (1-eta)^(N-d), so wt(z) / (2 * 2^-N * (1-eta)^N)
    * q^N = p^d q^(N-d) with eta = p/q in lowest terms."""
    from ortools.sat.python import cp_model
    from fractions import Fraction
    N, reps, C, wt, W = D["N"], D["reps"], D["C"], D["wt"], D["W"]
    n = len(reps)
    fr = Fraction(D["eta"]).limit_denominator(10 ** 6)
    p, q = fr.numerator, fr.denominator - fr.numerator   # eta = p/(p+q); (1-eta) = q/(p+q)
    den = p + q
    base = 2.0 * 2.0 ** -N / den ** N
    iwt = np.zeros(D["n_f"], dtype=np.int64)
    sel = wt > 0
    iwt[sel] = np.round(wt[sel] / base).astype(np.int64)
    assert np.allclose(iwt * base, wt, rtol=1e-9), "integer weights mismatch"
    tot_int = float(iwt.sum())          # sum over all ordered? -> unordered pairs below
    mdl = cp_model.CpModel()
    x = [mdl.NewIntVar(0, N - 1, f"c{i}") for i in range(n)]
    zero_coset = int(D["coset"][0])
    mdl.Add(x[zero_coset] == int(np.where(C == reps[zero_coset])[0][0]))   # 0 in T
    terms = []
    for a, b in itertools.combinations(range(n), 2):
        tbl, costs = [], set()
        for ca in range(N):
            for cb in range(N):
                z = int((reps[a] ^ C[ca]) ^ (reps[b] ^ C[cb]))
                costs.add(int(iwt[z]))
        if costs == {0}:
            continue
        cv = mdl.NewIntVarFromDomain(cp_model.Domain.FromValues(sorted(costs)), f"w{a}_{b}")
        for ca in range(N):
            for cb in range(N):
                z = int((reps[a] ^ C[ca]) ^ (reps[b] ^ C[cb]))
                tbl.append((ca, cb, int(iwt[z])))
        mdl.AddAllowedAssignments([x[a], x[b], cv], tbl)
        terms.append(cv)
    mdl.Maximize(sum(terms))
    if hint is not None:
        for i in range(n):
            mdl.AddHint(x[i], int(hint[i]))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = verbose
    solver.parameters.symmetry_level = 4
    t0 = time.time()
    st = solver.Solve(mdl)
    ch = np.array([solver.Value(x[i]) for i in range(n)], dtype=np.int64)
    val = transversal_value(D, ch)
    # integer objective -> value:  value = N * (sum iwt * base) / W
    conv = N * base / W
    return val, ch, {"time": time.time() - t0, "status": solver.StatusName(st),
                     "optimal": st == cp_model.OPTIMAL,
                     "objective_value": float(solver.ObjectiveValue()) * conv,
                     "bound_value": float(solver.BestObjectiveBound()) * conv,
                     "n_pair_tables": len(terms)}


def _selftest():
    from kv_instance import khot_vishnoi
    for k, eta in [(3, 0.2), (3, 0.3)]:
        D = kv_transversal_data(k, eta)
        g = khot_vishnoi(k, eta)
        assert D["n_cos"] == g.n and D["N"] == g.k
        # the transversal formulation must agree with the generic instance evaluator
        rng = np.random.default_rng(0)
        for _ in range(5):
            ch = rng.integers(0, D["N"], size=D["n_cos"])
            v1 = transversal_value(D, ch)
            # translate to a labelling of the generic instance
            from kv_instance import class_structure
            cls, S_of, kreps, F = class_structure(k)
            T = D["reps"] ^ D["C"][ch]
            L = np.zeros(g.n, dtype=np.int64)
            for f in T:
                L[cls[f]] = S_of[f]
            v2 = g.value(L)
            assert abs(v1 - v2) < 1e-12, (v1, v2)
        sc = subcube_choice(D)
        print(f"  k={k} eta={eta}: {D['n_cos']} cosets, window {D['dmin']}..{D['dmax']}, "
              f"subcube value {transversal_value(D, sc):.6f}  (5 random labellings matched the generic evaluator)")
    print("kv_transversal selftest: OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--time", type=float, default=3600)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        D = kv_transversal_data(a.k, a.eta)
        sc = subcube_choice(D)
        sv = transversal_value(D, sc)
        print(f"KV k={a.k} eta={a.eta}: {D['n_cos']} cosets x {D['N']} values, subcube value {sv:.6f}", flush=True)
        val, ch, info = cpsat_transversal(D, time_limit=a.time, workers=a.workers, hint=sc)
        rec = {"k": a.k, "eta": a.eta, "n_cosets": int(D["n_cos"]), "N": int(D["N"]),
               "subcube_value": sv, "value": val, "bound_value": info["bound_value"],
               "optimal": info["optimal"], "status": info["status"], "time": info["time"],
               "choice": ch.tolist()}
        print(json.dumps({kk: vv for kk, vv in rec.items() if kk != "choice"}), flush=True)
        out = a.out or f"../results/kv{a.k}_eta{a.eta}_transversal_exact.json"
        json.dump(rec, open(out, "w"), indent=1)
