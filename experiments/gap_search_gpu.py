"""
gap_search_gpu.py -- Track A at scale: adversarial search for (1-eps, 1-C eps)
gaps of degree-2 / degree-4 SoS on Boolean unique games, using the GPU ADMM
solver (warm-started between weight steps) and exact brute-force optima.

Reported C uses the CERTIFIED lower bound on the relaxation value, so every
recorded C is a rigorous lower bound on the true gap ratio of that instance.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import numpy as np
import torch
from gap_search import brute_opt_boolean, complete_graph, hypercube, cycle, cycle_reference
from sos_gpu import BooleanSoS


class Evaluator:
    def __init__(self, n, edges, degree, device=None, iters=600, tol=1e-5):
        self.n, self.edges, self.degree = n, edges, degree
        D = 1 + n + (n * (n - 1) // 2 if degree == 4 else 0)
        if device is None:
            device = "cuda" if (D > 250 and torch.cuda.is_available()) else "cpu"
        dt = torch.float64 if device == "cpu" else torch.float32
        self.S = BooleanSoS(n, edges, degree=degree, device=device, dtype=dt)
        self.iters, self.tol = iters, tol
        self.warm = None

    def __call__(self, signs, w, warm=True, iters=None):
        S = self.S.set_instance(signs, w)
        S.solve(iters=iters or self.iters, tol=self.tol, warm=self.warm if warm else None)
        lo, hi = S.certified_bounds()
        if warm:
            self.warm = (S.Z.clone(), S.U.clone())
        psat = S.edge_pseudo_sat(signs)
        opt, x = brute_opt_boolean(self.edges, signs, w, self.n)
        R = min(1.0, lo)
        C = (1 - opt) / max(1 - R, 1e-9)
        return C, opt, R, min(1.0, hi), x, psat


def grad_C(edges, signs, opt, R, x, psat):
    u, v = edges[:, 0], edges[:, 1]
    sat = ((x[u] * x[v]) == signs).astype(float)
    return ((1 - opt) * (psat - R) - (1 - R) * (sat - opt)) / max(1 - R, 1e-9) ** 2


def optimize_weights(ev, signs, w0, steps=40, eta0=1.0, prune=1e-3, verbose=False):
    w = w0 / w0.sum()
    C, opt, R, Rhi, x, psat = ev(signs, w, warm=False)
    eta = eta0
    for s in range(steps):
        g = grad_C(ev.edges, signs, opt, R, x, psat)
        g = g / (np.abs(g).max() + 1e-12)
        improved = False
        for _ in range(5):
            w2 = w * np.exp(eta * g)
            w2 = np.where(w2 < prune * w2.max(), 0.0, w2); w2 /= w2.sum()
            C2, opt2, R2, Rhi2, x2, psat2 = ev(signs, w2)
            if C2 > C + 1e-6:
                w, C, opt, R, Rhi, x, psat = w2, C2, opt2, R2, Rhi2, x2, psat2
                improved = True; eta = min(eta * 1.5, 4.0); break
            eta *= 0.5
        if verbose:
            print(f"    step {s:3d} C={C:.4f} opt={opt:.5f} R=[{R:.5f},{Rhi:.5f}] active={(w>0).sum()} eta={eta:.2f}", flush=True)
        if not improved and eta < 1e-3:
            break
    return C, w, opt, R, Rhi


def anneal(ev, signs0, w0, iters=200, seed=0, temp0=0.1, prune=1e-3, verbose=True, eta=0.7):
    """Combined annealing over signs + gradient weight moves + sparsification."""
    rng = np.random.default_rng(seed)
    m = len(ev.edges)
    signs, w = signs0.copy(), w0 / w0.sum()
    C, opt, R, Rhi, x, psat = ev(signs, w, warm=False)
    best = (C, signs.copy(), w.copy(), opt, R, Rhi)
    for it in range(iters):
        T = temp0 * (1 - it / iters)
        r = rng.random(); s2, w2 = signs.copy(), w.copy()
        if r < 0.4:
            g = grad_C(ev.edges, signs, opt, R, x, psat); g /= (np.abs(g).max() + 1e-12)
            w2 = w * np.exp(eta * (0.5 + rng.random()) * g)
        elif r < 0.7:
            act = np.where(w > 0)[0]
            for e in rng.choice(act, size=min(len(act), rng.integers(1, 3)), replace=False):
                s2[e] = -s2[e]
        elif r < 0.85:
            act = np.where(w > 0)[0]
            if len(act) > 3:
                w2[rng.choice(act)] = 0.0
        else:
            e = rng.integers(m); w2[e] = max(w2.max() * rng.random(), 1e-3); s2[e] = rng.choice([-1, 1])
        w2 = np.where(w2 < prune * w2.max(), 0.0, w2)
        if w2.sum() <= 0 or (w2 > 0).sum() < 3:
            continue
        w2 /= w2.sum()
        C2, opt2, R2, Rhi2, x2, psat2 = ev(s2, w2)
        if C2 > C or (T > 0 and rng.random() < math.exp(-(C - C2) / T)):
            signs, w, C, opt, R, Rhi, x, psat = s2, w2, C2, opt2, R2, Rhi2, x2, psat2
            if C > best[0]:
                best = (C, signs.copy(), w.copy(), opt, R, Rhi)
        if verbose and it % 20 == 0:
            print(f"  it {it:4d} C={C:.4f} (best {best[0]:.4f}) opt={opt:.5f} R=[{R:.5f},{Rhi:.5f}] active={(w>0).sum()}", flush=True)
    return best


def seeds_for(n, edges):
    """Structured seed sign/weight patterns: odd cycles of every length <= n on the vertex set."""
    eidx = {tuple(e): i for i, e in enumerate(edges.tolist())}
    m = len(edges)
    out = []
    for L in range(5, n + 1, 2):
        s = np.ones(m, dtype=int); w = np.zeros(m)
        ok = True
        for i in range(L):
            key = (min(i, (i + 1) % L), max(i, (i + 1) % L))
            if key not in eidx:
                ok = False; break
            s[eidx[key]] = -1; w[eidx[key]] = 1.0
        if ok:
            out.append((f"C{L}", s, w))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="K"); ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--degree", type=int, default=4); ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--random_starts", type=int, default=2)
    ap.add_argument("--out", default="../results/gap_search_gpu.jsonl")
    a = ap.parse_args()
    if a.graph == "K":
        edges, n = complete_graph(a.n), a.n
    elif a.graph == "Q":
        edges, n = hypercube(a.n), 1 << a.n
    else:
        edges, n = cycle(a.n), a.n
    m = len(edges)
    ev = Evaluator(n, edges, a.degree)
    print(f"[{a.graph}{a.n}, n={n}, m={m}, degree {a.degree}, D={ev.S.D}]  odd-cycle refs d=2: " +
          " ".join(f"C{L}={cycle_reference(L):.3f}" for L in range(5, min(n, 15) + 1, 2)), flush=True)
    starts = seeds_for(n, edges)
    rng = np.random.default_rng(0)
    for i in range(a.random_starts):
        starts.append((f"rand{i}", rng.choice([-1, 1], size=m), rng.dirichlet(np.ones(m))))
    for name, s0, w0 in starts:
        for seed in range(a.seeds):
            t0 = time.time()
            best = anneal(ev, s0, w0, iters=a.iters, seed=seed, verbose=True)
            C, signs, w, opt, R, Rhi = best
            # tight re-certification
            ev.tol = 1e-8; C2, opt2, R2, Rhi2, _, _ = ev(signs, w, warm=False, iters=8000); ev.tol = 1e-5
            rec = {"graph": a.graph, "n": n, "m": m, "degree": a.degree, "start": name, "seed": seed,
                   "C_certified": C2, "opt": opt2, "relax_lo": R2, "relax_hi": Rhi2,
                   "active_edges": int((w > 0).sum()), "signs": signs.tolist(), "weights": w.tolist(),
                   "time": time.time() - t0}
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  start {name:6s} seed {seed}: C={C2:.4f} opt={opt2:.5f} R=[{R2:.5f},{Rhi2:.5f}] active={rec['active_edges']} ({time.time()-t0:.0f}s)", flush=True)
