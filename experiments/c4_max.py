"""
c4_max.py -- Exhaustive (up to isomorphism) maximisation of the degree-4 gap shape

        C_4(I) = (1 - opt(I)) / (1 - SoS_4(I))

over Boolean 2Lin instances on n vertices.  By the dilution identity (docs/HYPOTHESES.md H8),
sup_I C_4(I) = infinity is EXACTLY the (1-eps, 1-C eps) Lasserre gap that
Khot-Moshkovitz (ECCC TR14-142) state is unknown for unique games at any constant
degree.  At degree 2 the supremum is infinite (odd cycles, C_2 = 4L/pi^2).

Search space.  An instance is (signs b in {+-1}^E, weights w >= 0) on K_n.
* Switching:  x_i -> -x_i flips the signs of the edges of a cut and changes
  neither opt nor SoS_d nor the weights.  So we may normalise every edge at
  vertex 0 to +1; what remains is an arbitrary sign pattern on K_{n-1}, i.e. a
  GRAPH on n-1 vertices (its edges = the -1 edges).
* Relabelling:  S_{n-1} acts.  So the sign patterns up to switching and
  isomorphism are exactly the isomorphism classes of graphs on n-1 vertices --
  enumerated completely by networkx's graph atlas for n-1 <= 7.
Therefore for n <= 8 this enumerates EVERY instance shape; weights are then
optimised inside each class by exponentiated gradient on the exact Danskin
gradient.  For n >= 9 the shapes are sampled.

opt: exact brute force over 2^(n-1) assignments.
SoS_4: ADMM with a certified primal/dual pair (sos_gpu.BooleanSoS); C_4 is always
reported from the CERTIFIED LOWER bound on SoS_4, so it is a rigorous lower bound
on the instance's true gap ratio.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch

from gap_search import brute_opt_boolean, complete_graph
from sos_gpu import BooleanSoS


class Ev:
    """Evaluator for a fixed n on the complete graph."""

    def __init__(self, n, degree=4, iters=2000, tol=1e-9, device="cpu"):
        self.n = n
        self.edges = complete_graph(n)
        self.m = len(self.edges)
        self.S = BooleanSoS(n, self.edges, degree=degree, device=device, dtype=torch.float64)
        self.iters, self.tol = iters, tol
        self.eidx = {tuple(e): i for i, e in enumerate(self.edges.tolist())}

    def __call__(self, signs, w):
        S = self.S.set_instance(signs.astype(float), w)
        S.solve(iters=self.iters, tol=self.tol)
        lo, hi = S.certified_bounds()
        R = min(1.0, lo)
        opt, x = brute_opt_boolean(self.edges, signs, w, self.n)
        return (1 - opt) / max(1 - R, 1e-12), opt, R, min(1.0, hi), x, S.edge_pseudo_sat(signs)

    def signs_from_minus_graph(self, minus_edges):
        """minus_edges: iterable of pairs on {1..n-1} (vertex 0 is all +1)."""
        s = np.ones(self.m, dtype=np.int64)
        for a, b in minus_edges:
            s[self.eidx[(min(a, b), max(a, b))]] = -1
        return s


def grad_C(edges, signs, opt, R, x, psat):
    u, v = edges[:, 0], edges[:, 1]
    sat = ((x[u] * x[v]) == signs).astype(float)
    return ((1 - opt) * (psat - R) - (1 - R) * (sat - opt)) / max(1 - R, 1e-9) ** 2


def optimize_weights(ev, signs, w0=None, steps=50, eta0=1.0, floor=0.0, verbose=False):
    w = np.ones(ev.m) / ev.m if w0 is None else np.maximum(w0, 0) / max(w0.sum(), 1e-12)
    C, opt, R, Rhi, x, psat = ev(signs, w)
    eta = eta0
    for s in range(steps):
        g = grad_C(ev.edges, signs, opt, R, x, psat)
        g = g / (np.abs(g).max() + 1e-12)
        improved = False
        for _ in range(6):
            w2 = w * np.exp(eta * g)
            w2 = np.maximum(w2, floor * w2.max())
            w2 /= w2.sum()
            C2, opt2, R2, Rhi2, x2, psat2 = ev(signs, w2)
            if C2 > C + 1e-9:
                w, C, opt, R, Rhi, x, psat = w2, C2, opt2, R2, Rhi2, x2, psat2
                improved = True; eta = min(eta * 1.6, 6.0); break
            eta *= 0.5
        if verbose:
            print(f"     step {s:3d} C={C:.6f} opt={opt:.6f} R=[{R:.6f},{Rhi:.6f}]", flush=True)
        if not improved and eta < 1e-4:
            break
    return C, w, opt, R, Rhi


def minus_graph_shapes(n, sample=None, seed=0):
    """All isomorphism classes of graphs on n-1 vertices (exhaustive for n-1 <= 7
    via the networkx atlas), else a random sample."""
    import networkx as nx
    k = n - 1
    if k <= 7 and sample is None:
        out = []
        for G in nx.graph_atlas_g():
            if G.number_of_nodes() == k:
                out.append((f"atlas{len(out)}", [tuple(e) for e in G.edges()]))
        return out, True
    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations(range(k), 2))
    out = []
    for i in range(sample or 3000):
        mask = rng.random(len(pairs)) < rng.uniform(0.1, 0.9)
        out.append((f"rand{i}", [p for p, mm in zip(pairs, mask) if mm]))
    return out, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--top", type=int, default=40, help="how many shapes to weight-optimise")
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--out", default="../results/c4_max.jsonl")
    a = ap.parse_args()
    ev = Ev(a.n, degree=a.degree)
    shapes, exhaustive = minus_graph_shapes(a.n, sample=a.sample)
    print(f"[n={a.n}, degree {a.degree}, D={ev.S.D}] {len(shapes)} sign-shapes "
          f"({'EXHAUSTIVE up to switching+isomorphism' if exhaustive else 'sampled'})", flush=True)
    t0 = time.time()
    # pass 1: uniform weights
    rows = []
    for name, me in shapes:
        signs = ev.signs_from_minus_graph([(u + 1, v + 1) for u, v in me])
        C, opt, R, Rhi, x, psat = ev(signs, np.ones(ev.m))
        rows.append((C, name, me, opt, R, Rhi))
    rows.sort(reverse=True, key=lambda r: r[0])
    print(f"  pass 1 (uniform weights) done in {time.time()-t0:.0f}s; "
          f"best C_{a.degree} = {rows[0][0]:.6f} (shape {rows[0][1]}, {len(rows[0][2])} minus-edges), "
          f"median {np.median([r[0] for r in rows]):.6f}", flush=True)
    # pass 2: weight optimisation on the top shapes, multi-start.
    # NOTE max_I C_4 is non-decreasing in n (embed an (n-1)-vertex instance with
    # zero weights on the new vertex), so we always include sparse starts that can
    # reach a sub-instance, plus any --seed_json best weights from smaller n.
    best = None
    out = open(a.out, "a")
    rng = np.random.default_rng(0)
    seeds = [np.ones(ev.m)]
    for _ in range(a.restarts):
        seeds.append(rng.dirichlet(np.ones(ev.m) * rng.choice([0.15, 0.5, 2.0])))
    # a start supported on a random (n-1)-subset, so the n-1 optimum is reachable
    for drop in range(ev.n):
        w0 = np.ones(ev.m)
        for i, (u, v) in enumerate(ev.edges.tolist()):
            if u == drop or v == drop:
                w0[i] = 0.0
        seeds.append(w0)
    for C0, name, me, opt0, R0, Rhi0 in rows[:a.top]:
        signs = ev.signs_from_minus_graph([(u + 1, v + 1) for u, v in me])
        C, w, opt, R, Rhi = -1, None, None, None, None
        for w0 in seeds:
            if w0.sum() <= 0:
                continue
            c_, w_, o_, r_, rh_ = optimize_weights(ev, signs, w0=w0, steps=a.steps)
            if c_ > C:
                C, w, opt, R, Rhi = c_, w_, o_, r_, rh_
        rec = {"n": a.n, "degree": a.degree, "shape": name, "minus_edges": [list(e) for e in me],
               "C_uniform": C0, "C_optimised": C, "opt": opt, "sos_lo": R, "sos_hi": Rhi,
               "weights": w.tolist(), "exhaustive_shapes": exhaustive, "n_shapes": len(shapes)}
        out.write(json.dumps(rec) + "\n"); out.flush()
        if best is None or C > best[0]:
            best = (C, name, opt, R, Rhi, w)
            print(f"  new best: C_{a.degree} = {C:.6f}  (uniform {C0:.6f})  opt={opt:.6f} "
                  f"SoS in [{R:.6f},{Rhi:.6f}]  shape {name} ({len(me)} minus-edges)", flush=True)
    print(f"[n={a.n}] FINAL best C_{a.degree} = {best[0]:.6f}   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
