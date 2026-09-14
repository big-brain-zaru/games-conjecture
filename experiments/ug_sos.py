"""
ug_sos.py -- Level-2 Lasserre / degree-4 sum-of-squares relaxation for Unique Games.

Formulation (0/1 one-hot variables x_{v,a}, sum_a x_{v,a} = 1, x_{v,a} x_{v,b} = 0):
  pseudo-moments y[U, rho] for every set U of <= 4 vertices and assignment rho to U
  (a consistent family of local distributions on <= 4 vertices), and the moment
  matrix M indexed by partial assignments on <= 2 vertices:
      M[(S,s),(T,t)] = 0            if s, t disagree on S cap T
                     = y[S u T, s u t]   otherwise,
  M >= 0 (PSD), y[{}] = 1, and marginalisation  sum_a y[U + (v->a)] = y[U].
Objective: sum_e w_e sum_a y[{u->a, v->pi_e(a)}] / W.

Facts used for testing: with n <= 4 vertices the relaxation is exact (all
moments exist); in general opt <= SoS4 <= basic SDP value.

For the Boolean case (k = 2, Max-2Lin(2)) a cheaper +-1 formulation is provided:
monomials x_T, |T| <= 4, moment matrix on {1, x_i, x_i x_j}.

Solver: cvxpy + SCS (Clarabel's dense PSD Hessian is unusable beyond ~100 rows).
"""
from __future__ import annotations

import itertools
import time
from typing import Optional

import numpy as np
import scipy.sparse as sp
from ug_core import UniqueGame


# ----------------------------------------------------------------------------
# General alphabet: level-2 Lasserre in 0/1 one-hot variables
# ----------------------------------------------------------------------------
def lasserre2(g: UniqueGame, solver: str = "SCS", verbose: bool = False, eps: float = 1e-6,
              max_iters: int = 20000, return_moments: bool = False, **kw):
    """Returns (value, info[, moments dict])."""
    import cvxpy as cp
    n, k = g.n, g.k
    # --- index all partial assignments on <= 4 vertices --------------------
    # key: tuple of (v, a) sorted by v
    yidx = {(): 0}
    keys = [()]
    for r in (1, 2, 3, 4):
        for U in itertools.combinations(range(n), r):
            for rho in itertools.product(range(k), repeat=r):
                key = tuple(zip(U, rho))
                yidx[key] = len(keys)
                keys.append(key)
    ny = len(keys)
    y = cp.Variable(ny)
    cons = [y[0] == 1, y >= 0]
    # marginalisation: for U with |U| <= 3 and v not in U: sum_a y[U + (v,a)] = y[U]
    A_rows, A_cols, A_vals, rhs_rows = [], [], [], []
    row = 0
    for key in keys:
        if len(key) > 3:
            continue
        Uset = {v for v, _ in key}
        for v in range(n):
            if v in Uset:
                continue
            for a in range(k):
                ext = tuple(sorted(key + ((v, a),)))
                A_rows.append(row); A_cols.append(yidx[ext]); A_vals.append(1.0)
            A_rows.append(row); A_cols.append(yidx[key]); A_vals.append(-1.0)
            row += 1
    A = sp.csr_matrix((A_vals, (A_rows, A_cols)), shape=(row, ny))
    cons.append(A @ y == 0)
    # --- moment matrix on partial assignments of <= 2 vertices --------------
    basis = [()]
    for v in range(n):
        for a in range(k):
            basis.append(((v, a),))
    for u, v in itertools.combinations(range(n), 2):
        for a in range(k):
            for b in range(k):
                basis.append(((u, a), (v, b)))
        # (u,a),(v,b) with u<v, sorted already
    D = len(basis)
    # M = sum_i y_i * E_i  -> build a sparse map (row, col) -> y index or None(=0)
    rows, cols, ys = [], [], []
    for i, s in enumerate(basis):
        ds = dict(s)
        for j in range(i, D):
            t = basis[j]
            ok = True
            merged = dict(ds)
            for v, a in t:
                if v in merged and merged[v] != a:
                    ok = False; break
                merged[v] = a
            if not ok:
                continue
            key = tuple(sorted(merged.items()))
            if len(key) > 4:
                continue  # cannot happen: |S|,|T| <= 2
            rows.append(i); cols.append(j); ys.append(yidx[key])
    rows = np.array(rows); cols = np.array(cols); ys = np.array(ys)
    # Build M via a linear map: vec(M) = P y, with P sparse (D*D x ny); symmetric fill
    lin_r = np.concatenate([rows * D + cols, cols * D + rows])
    lin_c = np.concatenate([ys, ys])
    # remove duplicate diagonal entries (i==j counted twice)
    diag = rows == cols
    lin_r = np.concatenate([rows * D + cols, (cols * D + rows)[~diag]])
    lin_c = np.concatenate([ys, ys[~diag]])
    P = sp.csr_matrix((np.ones(len(lin_r)), (lin_r, lin_c)), shape=(D * D, ny))
    M = cp.reshape(P @ y, (D, D), order="C")
    cons.append(M >> 0)
    # --- objective ----------------------------------------------------------
    c = np.zeros(ny)
    W = g.total_weight
    for e, (u, v) in enumerate(g.edges.tolist()):
        for a in range(k):
            b = int(g.perms[e, a])
            key = tuple(sorted(((u, a), (v, b))))
            if u == v:
                # self-loop: satisfied iff a == b -> moment y[(u,a)] if a==b else 0
                if a == b:
                    c[yidx[((u, a),)]] += g.weights[e] / W
                continue
            c[yidx[key]] += g.weights[e] / W
    prob = cp.Problem(cp.Maximize(c @ y), cons)
    t0 = time.time()
    if solver == "SCS":
        prob.solve(solver="SCS", verbose=verbose, eps=eps, max_iters=max_iters, **kw)
    else:
        prob.solve(solver=solver, verbose=verbose, **kw)
    info = {"status": prob.status, "time": time.time() - t0, "D": D, "ny": ny, "solver": solver}
    if return_moments:
        return float(prob.value), info, {keys[i]: float(y.value[i]) for i in range(ny)}
    return float(prob.value), info


# ----------------------------------------------------------------------------
# Boolean alphabet: degree-4 SoS in +-1 variables for Max-2Lin(2)
# ----------------------------------------------------------------------------
def sos4_boolean(g: UniqueGame, solver: str = "SCS", verbose: bool = False, eps: float = 1e-6,
                 max_iters: int = 20000, degree: int = 4, return_moments: bool = False, **kw):
    """Degree-2 or degree-4 SoS for a k=2 unique game (constraints x_u x_v = b_e).
    degree=2 is the Goemans-Williamson SDP.  Returns (value, info)."""
    import cvxpy as cp
    assert g.k == 2
    n = g.n
    # b_e = +1 if identity permutation, -1 if swap
    b = np.where(g.perms[:, 0] == 0, 1.0, -1.0)
    if degree == 2:
        basis = [()] + [(i,) for i in range(n)]
    else:
        basis = [()] + [(i,) for i in range(n)] + list(itertools.combinations(range(n), 2))
    # monomials appearing in products: symmetric difference of index sets, size <= 4
    monos = {}
    def mid(T):
        T = tuple(sorted(T))
        if T not in monos:
            monos[T] = len(monos)
        return monos[T]
    mid(())
    D = len(basis)
    rows, cols, ms = [], [], []
    for i, s in enumerate(basis):
        for j in range(i, D):
            t = basis[j]
            T = tuple(sorted(set(s) ^ set(t)))
            rows.append(i); cols.append(j); ms.append(mid(T))
    nm = len(monos)
    y = cp.Variable(nm)
    rows = np.array(rows); cols = np.array(cols); ms = np.array(ms)
    diag = rows == cols
    lin_r = np.concatenate([rows * D + cols, (cols * D + rows)[~diag]])
    lin_c = np.concatenate([ms, ms[~diag]])
    P = sp.csr_matrix((np.ones(len(lin_r)), (lin_r, lin_c)), shape=(D * D, nm))
    M = cp.reshape(P @ y, (D, D), order="C")
    cons = [y[monos[()]] == 1, M >> 0]
    c = np.zeros(nm)
    W = g.total_weight
    const = 0.0
    for e, (u, v) in enumerate(g.edges.tolist()):
        if u == v:
            const += (g.weights[e] / W) * (1.0 if b[e] > 0 else 0.0)
            continue
        # satisfied iff x_u x_v = b_e  ->  (1 + b_e x_u x_v)/2
        const += g.weights[e] / W / 2
        c[mid((u, v))] += g.weights[e] / W * b[e] / 2
    prob = cp.Problem(cp.Maximize(c @ y + const), cons)
    t0 = time.time()
    if solver == "SCS":
        prob.solve(solver="SCS", verbose=verbose, eps=eps, max_iters=max_iters, **kw)
    else:
        prob.solve(solver=solver, verbose=verbose, **kw)
    info = {"status": prob.status, "time": time.time() - t0, "D": D, "nm": nm, "degree": degree}
    if return_moments:
        # pseudo-expectations of x_u x_v for every edge (u,v), in edge order
        pm = np.array([1.0 if u == v else float(y.value[monos[(min(u, v), max(u, v))]])
                       for u, v in g.edges.tolist()])
        return float(prob.value), info, pm
    return float(prob.value), info


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _selftest():
    from ug_core import random_unique_game, brute_force_optimum, max2lin_from_constraints
    from ug_sdp import sdp_cvxpy
    rng = np.random.default_rng(0)
    # 1. n=4 vertices, k=3: level-2 Lasserre must be exact
    g = random_unique_game(4, 3, m=6, seed=1)
    opt, _ = brute_force_optimum(g)
    v2, info = lasserre2(g, eps=1e-8)
    assert abs(v2 - opt) < 1e-5, (v2, opt)
    print(f"  n=4,k=3: brute {opt:.5f} lasserre2 {v2:.5f}  (D={info['D']}, {info['time']:.1f}s)")
    # 2. n=7, k=3: opt <= lasserre2 <= basic sdp
    g = random_unique_game(7, 3, m=15, seed=2)
    opt, _ = brute_force_optimum(g)
    v2, info = lasserre2(g, eps=1e-7)
    sdp, _, _ = sdp_cvxpy(g)
    assert opt - 1e-5 <= v2 <= sdp + 1e-4, (opt, v2, sdp)
    print(f"  n=7,k=3: brute {opt:.5f} <= lasserre2 {v2:.5f} <= sdp {sdp:.5f}  (D={info['D']}, {info['time']:.1f}s)")
    # 3. Boolean: 5-cycle Max-Cut (all constraints x_u x_v = -1): opt 0.8, GW = (5/2)(1+cos(pi/5))/5...
    cons = [(i, (i + 1) % 5, 1) for i in range(5)]      # x_v - x_u = 1 mod 2  -> x_u != x_v
    g = max2lin_from_constraints(5, 2, cons)
    opt, _ = brute_force_optimum(g)
    gw, _ = sos4_boolean(g, degree=2, eps=1e-8)
    s4, _ = sos4_boolean(g, degree=4, eps=1e-8)
    gw_exact = (1 + np.cos(np.pi / 5)) / 2 * 1.0    # per-edge value of the optimal GW vectors on C5
    print(f"  C5 max-cut: opt {opt:.5f}  GW {gw:.5f} (exact {gw_exact:.5f})  SoS4 {s4:.5f}")
    assert abs(gw - gw_exact) < 1e-4 and abs(s4 - opt) < 1e-4
    # 4. Boolean vs general on same instance
    g = random_unique_game(8, 2, m=16, seed=5)
    opt, _ = brute_force_optimum(g)
    s4, _ = sos4_boolean(g, degree=4, eps=1e-8)
    l2, _ = lasserre2(g, eps=1e-8)
    print(f"  n=8,k=2: brute {opt:.5f}  SoS4(bool) {s4:.5f}  lasserre2 {l2:.5f}")
    assert s4 >= opt - 1e-5 and l2 >= opt - 1e-5 and abs(s4 - l2) < 2e-3
    print("ug_sos selftest: OK")


if __name__ == "__main__":
    _selftest()
