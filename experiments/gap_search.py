"""
gap_search.py -- Track A: adversarial search for relaxation gaps of the
(1-eps, 1-C*eps) shape on small Boolean unique games (Max-2Lin(2)).

Instance space: a fixed simple graph (edges E), free signs b_e in {+1,-1}
(constraint x_u x_v = b_e) and free weights w on the simplex.
Score:  C(I) = (1 - opt(I)) / (1 - R_d(I)),  R_d = degree-d SoS value (d=2: GW SDP).
C is invariant under mixing with satisfiable parts, so it measures the *shape*
of the gap: the constant Khot-Moshkovitz ask for.  Odd cycles give
C(C_L) = (1/L) / ((1 - cos(pi/L))/2) ~ 4L/pi^2 at degree 2 and C = 1 at degree 4.

opt(I): exact brute force over 2^(n-1) assignments.
R_d(I): cvxpy/SCS; every accepted C is recomputed at tight tolerance at the end.

Gradient of C w.r.t. normalised weights (W = 1):
  d opt/dw_e = sat_e(x*) - opt        (x* an optimal assignment)
  d R  /dw_e = psat_e - R             (psat_e = (1 + b_e Etilde[x_u x_v])/2, Danskin)
  d C  /dw_e = [ (1-opt) * dR/dw_e - (1-R) * dopt/dw_e ] / (1-R)^2
Moves: exponentiated-gradient step on w; single/double sign flips; edge
deletion/revival (weight -> 0 / small); annealing acceptance on C.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from typing import Optional

import numpy as np
from ug_core import UniqueGame, max2lin_from_constraints
from ug_sos import sos4_boolean


def boolean_game(edges: np.ndarray, signs: np.ndarray, weights: np.ndarray) -> UniqueGame:
    cons = [(int(u), int(v), 0 if s > 0 else 1) for (u, v), s in zip(edges, signs)]
    n = int(edges.max()) + 1
    return max2lin_from_constraints(n, 2, cons, weights=weights)


def brute_opt_boolean(edges: np.ndarray, signs: np.ndarray, weights: np.ndarray, n: int):
    """Exact optimum of sum_e w_e [x_u x_v = b_e] / W over x in {+-1}^n (x_0 = +1 wlog)."""
    W = weights.sum()
    N = 1 << (n - 1)
    best, bestx = -1.0, None
    chunk = 1 << 16
    u, v = edges[:, 0], edges[:, 1]
    for start in range(0, N, chunk):
        ids = np.arange(start, min(N, start + chunk), dtype=np.int64)
        bits = ((ids[:, None] >> np.arange(n - 1)[None, :]) & 1)
        x = np.concatenate([np.ones((len(ids), 1), dtype=np.int8), (1 - 2 * bits).astype(np.int8)], axis=1)
        sat = (x[:, u] * x[:, v]) == signs[None, :]
        val = sat @ weights / W
        i = int(np.argmax(val))
        if val[i] > best:
            best, bestx = float(val[i]), x[i].copy()
    return best, bestx


def evaluate(edges, signs, w, n, degree, eps=1e-6):
    """Returns C, opt, R, x*, psat (pseudo-satisfaction per edge)."""
    g = boolean_game(edges, signs, w)
    R, info, pm = sos4_boolean(g, degree=degree, eps=eps, return_moments=True)
    R = min(1.0, R)
    psat = (1.0 + signs * pm) / 2.0
    opt, x = brute_opt_boolean(edges, signs, w, n)
    C = (1.0 - opt) / max(1.0 - R, 1e-9)
    return C, opt, R, x, psat


def grad_C(edges, signs, w, opt, R, x, psat):
    u, v = edges[:, 0], edges[:, 1]
    sat = ((x[u] * x[v]) == signs).astype(float)
    dopt = sat - opt
    dR = psat - R
    return ((1 - opt) * dR - (1 - R) * dopt) / max(1 - R, 1e-9) ** 2


def search(edges, n, degree, iters=300, seed=0, init_signs=None, init_w=None, verbose=True,
           eta=0.5, eps=1e-6, temp0=0.05, min_w=1e-3):
    rng = np.random.default_rng(seed)
    m = len(edges)
    signs = init_signs.copy() if init_signs is not None else rng.choice([-1, 1], size=m)
    w = init_w.copy() if init_w is not None else rng.dirichlet(np.ones(m))
    C, opt, R, x, psat = evaluate(edges, signs, w, n, degree, eps)
    best = (C, signs.copy(), w.copy(), opt, R)
    hist = [C]
    for it in range(iters):
        T = temp0 * (1 - it / iters)
        r = rng.random()
        s2, w2 = signs.copy(), w.copy()
        if r < 0.35:                                   # exact gradient step on weights
            gC = grad_C(edges, signs, w, opt, R, x, psat)
            step = eta * (0.5 + rng.random())
            w2 = w * np.exp(step * gC / (np.abs(gC).max() + 1e-12))
        elif r < 0.65:                                 # sign flips
            for e in rng.choice(m, size=rng.integers(1, 3), replace=False):
                s2[e] = -s2[e]
        elif r < 0.85:                                 # delete an edge (sparsify)
            e = rng.integers(m); w2[e] = 0.0
        else:                                          # revive / boost a random edge
            e = rng.integers(m); w2[e] = w2.max() * rng.random()
        if w2.sum() <= 0:
            continue
        w2 = np.where(w2 < min_w * w2.max(), 0.0, w2)  # prune tiny weights
        w2 /= w2.sum()
        if (w2 > 0).sum() < 3:
            continue
        C2, opt2, R2, x2, psat2 = evaluate(edges, s2, w2, n, degree, eps)
        if C2 > C or (T > 0 and rng.random() < math.exp(-(C - C2) / T)):
            signs, w, C, opt, R, x, psat = s2, w2, C2, opt2, R2, x2, psat2
            if C > best[0]:
                best = (C, signs.copy(), w.copy(), opt, R)
        hist.append(C)
        if verbose and it % 25 == 0:
            print(f"  it {it:4d}  C={C:.4f} (best {best[0]:.4f})  opt={opt:.4f}  R{degree}={R:.4f}  active={int((w>0).sum())}", flush=True)
    return best, hist


def optimize_weights(edges, signs, n, degree, w0=None, steps=60, eps=1e-6, eta0=1.0, verbose=False, prune=1e-3):
    """Deterministic exponentiated-gradient ascent on C over the weight simplex for fixed
    signs, with backtracking.  Returns (C, w, opt, R)."""
    m = len(edges)
    w = (np.ones(m) / m) if w0 is None else w0 / w0.sum()
    C, opt, R, x, psat = evaluate(edges, signs, w, n, degree, eps)
    eta = eta0
    for s in range(steps):
        gC = grad_C(edges, signs, w, opt, R, x, psat)
        gC = gC / (np.abs(gC).max() + 1e-12)
        improved = False
        for _ in range(6):
            w2 = w * np.exp(eta * gC)
            w2 = np.where(w2 < prune * w2.max(), 0.0, w2)      # allow edges to die
            w2 /= w2.sum()
            C2, opt2, R2, x2, psat2 = evaluate(edges, signs, w2, n, degree, eps)
            if C2 > C + 1e-9:
                w, C, opt, R, x, psat = w2, C2, opt2, R2, x2, psat2
                improved = True; eta = min(eta * 1.5, 4.0)
                break
            eta *= 0.5
        if verbose:
            print(f"    step {s:3d} C={C:.5f} opt={opt:.5f} R={R:.5f} eta={eta:.3f}", flush=True)
        if not improved and eta < 1e-3:
            break
    return C, w, opt, R


def search_signs(edges, n, degree, n_patterns=20, seed=0, seeds_signs=(), steps=40, eps=1e-6, verbose=True):
    """Outer loop: structured seed sign patterns + random ones; inner: optimize_weights."""
    rng = np.random.default_rng(seed)
    m = len(edges)
    best = (-1.0, None, None, None, None)
    patterns = [np.asarray(s) for s in seeds_signs]
    patterns += [rng.choice([-1, 1], size=m) for _ in range(n_patterns)]
    for i, signs in enumerate(patterns):
        C, w, opt, R = optimize_weights(edges, signs, n, degree, steps=steps, eps=eps)
        if verbose:
            print(f"  pattern {i:3d}: C={C:.4f} opt={opt:.4f} R{degree}={R:.4f} active={int((w>1e-6).sum())}", flush=True)
        if C > best[0]:
            best = (C, signs.copy(), w.copy(), opt, R)
    return best


def complete_graph(n):
    return np.array(list(itertools.combinations(range(n), 2)), dtype=np.int64)


def hypercube(d):
    n = 1 << d
    return np.array([(x, x ^ (1 << i)) for x in range(n) for i in range(d) if x < x ^ (1 << i)], dtype=np.int64)


def cycle(n):
    return np.array([(i, (i + 1) % n) for i in range(n)], dtype=np.int64)


def cycle_reference(L: int):
    """Exact C at degree 2 for the odd cycle C_L (all anti-edges)."""
    return (1.0 / L) / ((1 - math.cos(math.pi / L)) / 2)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="K", help="K (complete), Q (hypercube dim), C (cycle)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--degree", type=int, default=2)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="../results/gap_search.jsonl")
    a = ap.parse_args()
    if a.graph == "K":
        edges, n = complete_graph(a.n), a.n
    elif a.graph == "Q":
        edges, n = hypercube(a.n), 1 << a.n
    else:
        edges, n = cycle(a.n), a.n
    print(f"reference: odd cycles at degree 2: C5={cycle_reference(5):.4f} C7={cycle_reference(7):.4f} C9={cycle_reference(9):.4f}")
    for seed in range(a.seeds):
        t0 = time.time()
        best, hist = search(edges, n, a.degree, iters=a.iters, seed=seed)
        C, signs, w, opt, R = best
        g = boolean_game(edges, signs, w)
        R_tight, info = sos4_boolean(g, degree=a.degree, eps=1e-9)
        opt2, x = brute_opt_boolean(edges, signs, w, n)
        C_tight = (1 - opt2) / max(1 - min(1.0, R_tight), 1e-12)
        rec = {"graph": a.graph, "n": n, "m": len(edges), "degree": a.degree, "seed": seed, "iters": a.iters,
               "C": C_tight, "opt": opt2, "relax": R_tight, "signs": signs.tolist(), "weights": w.tolist(),
               "active_edges": int((w > 0).sum()), "time": time.time() - t0}
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{a.graph}{a.n} d={a.degree} seed={seed}] best C={C_tight:.4f}  opt={opt2:.5f}  R={R_tight:.5f}  active={rec['active_edges']}  ({time.time()-t0:.0f}s)")
