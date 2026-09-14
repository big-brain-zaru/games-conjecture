"""
hypercube_track.py -- Track B: Max-2Lin(Z2) on the hypercube Q_d.

Implements the Agarwal-Kindler-Kolla-Trevisan gap instance Delta[k,d] (Section 3.1):
  edge e in direction i(e); H(v) = Hamming weight of v on coordinates > k.
  * i(e) > k                          -> equality edge
  * i(e) <= k and H(v1) > (d-k)/2     -> equality edge
  * otherwise                          -> inequality edge  (x_u x_v = -1)
AKKT: combinatorial violated fraction Omega(k/d) (for k <= O(sqrt d)), GW-SDP
violated fraction O(sqrt(k)/d); triangle inequalities "break" the gap; they
conjecture GW + triangle inequalities solves UG on Q_d.

This script computes, for each (d, k): exact opt (MILP, certified when the
solver proves optimality), GW-SDP value (GPU block ascent + dual certificate),
degree-4 SoS (GPU ADMM, certified bounds; D = 1 + n + C(n,2), so d <= 6..7),
and the gap ratios C_2 = (1-opt)/(1-GW), C_4 = (1-opt)/(1-SoS4).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch
from ug_core import UniqueGame, max2lin_from_constraints, milp_optimum
from ug_sdp import sdp_block_ascent, dual_certificate
from sos_gpu import BooleanSoS
from gap_search import brute_opt_boolean


def akkt_instance(d: int, k: int):
    """Returns (edges (m,2), signs (m,) in {+1,-1}) on Q_d with the Delta[k,d] rule.
    Coordinates are 0-based: 'first k' = bits 0..k-1; H(v) = popcount of bits k..d-1."""
    n = 1 << d
    hi_mask = ((1 << d) - 1) ^ ((1 << k) - 1)
    edges, signs = [], []
    for v in range(n):
        for i in range(d):
            u = v ^ (1 << i)
            if u < v:
                continue
            if i >= k:
                s = +1
            else:
                H = bin(v & hi_mask).count("1")            # same for u (they differ in bit i < k)
                s = +1 if H > (d - k) / 2 else -1
            edges.append((v, u)); signs.append(s)
    return np.array(edges, dtype=np.int64), np.array(signs, dtype=np.int64)


def to_game(edges, signs, n):
    cons = [(int(u), int(v), 0 if s > 0 else 1) for (u, v), s in zip(edges, signs)]
    return max2lin_from_constraints(n, 2, cons)


def run(d, k, do_sos4=True, milp_time=600, verbose=True):
    n = 1 << d
    edges, signs = akkt_instance(d, k)
    m = len(edges)
    g = to_game(edges, signs, n)
    rec = {"d": d, "k": k, "n": n, "m": m, "n_inequality_edges": int((signs < 0).sum())}
    # exact optimum
    t = time.time()
    if n <= 22:
        opt, x = brute_opt_boolean(edges, signs, np.ones(m), n); rec["opt_method"] = "brute"; rec["opt_certified"] = True
    else:
        opt, L, info = milp_optimum(g, time_limit=milp_time, fix_vertex0=True)
        rec["opt_method"] = "milp"; rec["opt_certified"] = bool(info["optimal"]); rec["opt_bound"] = info["bound"]
    rec["opt"] = float(opt); rec["opt_time"] = time.time() - t
    all_ones = 1.0 - (signs < 0).sum() / m
    rec["all_ones_value"] = float(all_ones)
    # GW SDP (basic SDP with k=2) via block ascent + certificate
    t = time.time()
    gw, V, info = sdp_block_ascent(g, r=min(32, n), sweeps=400, tol=1e-10)
    gw_hi, lam = dual_certificate(g, V)
    rec.update({"gw_lo": float(gw), "gw_hi": float(gw_hi), "gw_time": time.time() - t})
    rec["C2_certified"] = (1 - opt) / max(1 - gw, 1e-12)
    # degree-4 SoS
    if do_sos4 and n <= 128:
        t = time.time()
        S = BooleanSoS(n, edges, degree=4, device="cuda").set_instance(signs.astype(float), np.ones(m))
        S.solve(iters=4000 if n <= 64 else 1500, tol=1e-7)
        lo, hi = S.certified_bounds()
        rec.update({"sos4_lo": float(lo), "sos4_hi": float(hi), "sos4_D": S.D, "sos4_iters": S.iters_done,
                    "sos4_time": time.time() - t})
        rec["C4_certified"] = (1 - opt) / max(1 - min(1.0, lo), 1e-12)
    if verbose:
        print(json.dumps({a: (round(b, 6) if isinstance(b, float) else b) for a, b in rec.items()}), flush=True)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="4,5,6,7,8")
    ap.add_argument("--out", default="../results/hypercube_akkt.jsonl")
    a = ap.parse_args()
    for d in [int(x) for x in a.dims.split(",")]:
        ks = sorted(set([1, 2, max(1, int(round(np.sqrt(d)))), max(1, int(round(0.5 * np.sqrt(d)))), d // 2]))
        for k in ks:
            if k > d:
                continue
            rec = run(d, k, do_sos4=(d <= 7))
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
