"""
fs_track.py -- Feige-Schechtman-type sphere instances for Max-Cut, as seeds for
Track A at n = 30..60 where degree-4 SoS is far from exact.

Instance: n points u_1..u_n on S^{d-1} (random or a structured net); an edge
(i,j) with constraint x_i x_j = -1 (cut) and weight w_ij = f(<u_i,u_j>) where f
is supported on inner products <= -(1-delta) (nearly antipodal pairs).  The
vectors u_i themselves are a GW-SDP solution of value  sum w_ij (1-<u_i,u_j>)/2
>= 1 - delta/2, while the true max cut is 1 - Theta(sqrt(delta)) for large n
(Feige-Schechtman 2002).  We measure GW (certified), degree-4 SoS (certified),
and exact opt (MILP), and record C_2, C_4.  The open question is whether C_4
grows like C_2 or collapses.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from ug_core import max2lin_from_constraints, milp_optimum
from ug_sdp import sdp_block_ascent, dual_certificate
from sos_gpu import BooleanSoS
from gap_search import brute_opt_boolean


def sphere_instance(n: int, dim: int, delta: float, seed: int = 0, antipodal_pairs: bool = True):
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n // 2, dim)) if antipodal_pairs else rng.standard_normal((n, dim))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    if antipodal_pairs:
        U = np.concatenate([U, -U], axis=0)          # symmetric point set (helps the gap)
    G = U @ U.T
    edges, w = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if G[i, j] <= -(1 - delta):
                edges.append((i, j)); w.append(1.0 + G[i, j] + delta)   # in (0, delta]; heavier when more antipodal
    return np.array(edges, dtype=np.int64), np.array(w), U


def run(n, dim, delta, seed, milp_time=900, sos_iters=2500):
    edges, w, U = sphere_instance(n, dim, delta, seed)
    m = len(edges)
    if m < 3:
        return None
    signs = -np.ones(m, dtype=np.int64)
    cons = [(int(u), int(v), 1) for u, v in edges]
    g = max2lin_from_constraints(n, 2, cons, weights=w)
    rec = {"n": n, "dim": dim, "delta": delta, "seed": seed, "m": m}
    # vectors' own SDP value (a valid GW lower bound)
    Gm = U @ U.T
    rec["gw_from_vectors"] = float(sum(wi * (1 - Gm[i, j]) / 2 for (i, j), wi in zip(edges, w)) / w.sum())
    t = time.time()
    if n <= 22:
        opt, x = brute_opt_boolean(edges, signs, w, n); rec["opt_certified"] = True
    else:
        opt, L, info = milp_optimum(g, time_limit=milp_time, fix_vertex0=True)
        rec["opt_certified"] = bool(info["optimal"]); rec["opt_bound"] = info["bound"]
    rec["opt"] = float(opt); rec["opt_time"] = time.time() - t
    t = time.time()
    gw, V, _ = sdp_block_ascent(g, r=min(32, n), sweeps=600, tol=1e-11)
    gw_hi, _ = dual_certificate(g, V)
    rec.update({"gw_lo": float(gw), "gw_hi": float(gw_hi), "gw_time": time.time() - t})
    rec["C2"] = (1 - opt) / max(1 - gw, 1e-12)
    t = time.time()
    S = BooleanSoS(n, edges, degree=4, device="cuda").set_instance(signs.astype(float), w)
    S.solve(iters=sos_iters, tol=1e-7)
    lo, hi = S.certified_bounds()
    rec.update({"sos4_lo": float(lo), "sos4_hi": float(hi), "sos4_D": S.D, "sos4_iters": S.iters_done, "sos4_time": time.time() - t})
    rec["C4"] = (1 - opt) / max(1 - min(1.0, lo), 1e-12)
    rec["C4_upper_if_hi"] = (1 - opt) / max(1 - min(1.0, hi), 1e-12)
    print(json.dumps({a: (round(b, 6) if isinstance(b, float) else b) for a, b in rec.items()}), flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="20,30,40")
    ap.add_argument("--dims", default="3,4")
    ap.add_argument("--deltas", default="0.3,0.5")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="../results/fs_track.jsonl")
    a = ap.parse_args()
    for n in [int(x) for x in a.ns.split(",")]:
        for dim in [int(x) for x in a.dims.split(",")]:
            for delta in [float(x) for x in a.deltas.split(",")]:
                for seed in range(a.seeds):
                    rec = run(n, dim, delta, seed)
                    if rec:
                        with open(a.out, "a") as f:
                            f.write(json.dumps(rec) + "\n")
