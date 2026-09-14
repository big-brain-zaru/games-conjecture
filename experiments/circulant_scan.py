"""
circulant_scan.py -- Does the degree-4 gap shape C_4 grow?

For all-anti Max-Cut on Cay(Z_L, +-S) with uniform weights:
  * SoS_4 : symmetry-reduced ADMM with certified two-sided bounds (circulant_sos.py)
  * SoS_2 : closed form for a vertex-transitive weighted circulant,
              SoS_2 = 1/2 - (1/2) * min_alpha sum_d w_d cos(2 pi alpha d / L) / sum_d w_d
            (verified against the dense solver in _selftest)
  * opt   : CP-SAT max-cut; both the incumbent and the dual bound are kept, so
              C_4 >= (1 - opt_upper_bound) / (1 - SoS_4_lower_bound)
            is a RIGOROUS lower bound on the instance's gap shape even when CP-SAT
            does not close the instance.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch

from circulant_sos import CirculantSoS


def circ_edges(L, gens):
    e = sorted(set((min(v, (v + s) % L), max(v, (v + s) % L)) for v in range(L) for s in gens))
    return np.array([x for x in e if x[0] != x[1]], dtype=np.int64)


def sos2_circulant(L, gens, w=None):
    gens = sorted(set(min(g % L, (-g) % L) for g in gens if g % L))
    w = np.ones(len(gens)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    al = np.arange(L)
    val = np.zeros(L)
    for wi, g in zip(w, gens):
        val += wi * np.cos(2 * np.pi * al * g / L)
    return 0.5 - 0.5 * float(val.min())


def maxcut_cpsat(L, edges, time_limit=90, workers=10):
    from ortools.sat.python import cp_model
    m = len(edges)
    mdl = cp_model.CpModel()
    x = [mdl.NewBoolVar(f"x{i}") for i in range(L)]
    mdl.Add(x[0] == 0)
    cut = []
    for a, b in edges.tolist():
        c = mdl.NewBoolVar("")
        mdl.Add(x[a] + x[b] == 1).OnlyEnforceIf(c)
        mdl.Add(x[a] == x[b]).OnlyEnforceIf(c.Not())
        cut.append(c)
    mdl.Maximize(sum(cut))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers
    st = s.Solve(mdl)
    val = s.ObjectiveValue() / m
    bound = s.BestObjectiveBound() / m
    return val, bound, s.StatusName(st) == "OPTIMAL"


def run(L, gens, iters=20000, cp_time=90, device=None):
    device = device or ("cuda" if torch.cuda.is_available() and L >= 25 else "cpu")
    e = circ_edges(L, gens)
    t0 = time.time()
    C = CirculantSoS(L, device=device).set_instance(gens)
    C.solve(iters=iters, tol=1e-11)
    lo, hi = C.certified_bounds()
    tsos = time.time() - t0
    t0 = time.time()
    opt, opt_ub, proved = maxcut_cpsat(L, e, time_limit=cp_time)
    s2 = sos2_circulant(L, gens)
    rec = {"L": L, "gens": list(gens), "m": len(e), "opt": opt, "opt_ub": opt_ub, "opt_proved": proved,
           "sos2": s2, "sos4_lo": lo, "sos4_hi": hi,
           "C2": (1 - opt) / max(1 - s2, 1e-12),
           "C4_rigorous_lb": (1 - opt_ub) / max(1 - min(1.0, lo), 1e-12),
           "C4_at_incumbent": (1 - opt) / max(1 - min(1.0, lo), 1e-12),
           "nI": C.nI, "n_var": C.n_var, "sos_iters": C.iters_done,
           "sos_s": round(tsos, 1), "cp_s": round(time.time() - t0, 1), "device": device}
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", default="9,13,15,17,19,21,23,25,27,29,31,33")
    ap.add_argument("--gensets", default="1-2;1-3;2-3;1-2-3;1-2-4;1-4;3-4")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--cp_time", type=float, default=90)
    ap.add_argument("--out", default="../results/circulant_scan.jsonl")
    a = ap.parse_args()
    gensets = [[int(x) for x in g.split("-")] for g in a.gensets.split(";")]
    best = 0.0
    for L in [int(x) for x in a.Ls.split(",")]:
        if L % 2 == 0:
            continue
        for gens in gensets:
            if max(gens) >= L / 2 + 1:
                continue
            try:
                rec = run(L, gens, iters=a.iters, cp_time=a.cp_time)
            except Exception as ex:
                print(f"!! L={L} gens={gens}: {type(ex).__name__}: {str(ex)[:120]}", flush=True); continue
            best = max(best, rec["C4_rigorous_lb"])
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()
                              if k not in ("nI", "n_var", "device")}), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"### after L={L}: best rigorous C_4 lower bound = {best:.6f}", flush=True)
