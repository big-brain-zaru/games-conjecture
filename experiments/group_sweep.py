"""
group_sweep.py -- the non-abelian sweep: a COMPLETE census of the degree-4 gap
shape over signed Cayley-graph instances on every group in the library.

Per group Q:
  stage 1  every generator-class set S with |S| <= kmax and every sign pattern
           (uniform weights): SoS_2 exactly from the irreps (cayley_sdp), an
           incumbent cut by local search, and the filter
               U = (1 - incumbent) / (1 - SoS_2)  >=  C_4 ;
           instances with U <= record are ruled out rigorously.
  stage 2  every survivor: certified degree-4 value (group_sos, symmetry-reduced,
           tight two-sided bounds) and a proved optimum (CP-SAT), giving C_4 and
           the retention rho = (C_4 - 1)/(C_2 - 1), together with the dimension of
           the irrep that carries the degree-2 gap.
The hypothesis under test: retention depends on the carrier irrep's dimension,
which is the one thing an abelian group cannot vary (every abelian irrep is
one-dimensional).  Everything is recorded, including the negatives.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch

from group_library import library
from group_sos import GroupSoS, complex_irreps
from cayley_sdp import sos2_cayley


def local_search_cut(n, E, B, W, restarts=200, seed=0):
    rng = np.random.default_rng(seed)
    adj = [[] for _ in range(n)]
    for (a, b), s, w in zip(E.tolist(), B.tolist(), W.tolist()):
        adj[a].append((b, s, w)); adj[b].append((a, s, w))
    adj = [(np.array([x[0] for x in L], dtype=np.int64), np.array([x[1] for x in L], float),
            np.array([x[2] for x in L], float)) for L in adj]
    Wt = W.sum()
    best = -1.0
    for _ in range(restarts):
        x = rng.choice([-1, 1], size=n)
        improved = True
        while improved:
            improved = False
            for v in rng.permutation(n):
                nb, s, w = adj[v]
                # satisfied iff x_v x_u = s ; gain from flipping v
                cur = float((w * ((x[v] * x[nb]) == s)).sum())
                alt = float((w * ((-x[v] * x[nb]) == s)).sum())
                if alt > cur + 1e-12:
                    x[v] = -x[v]; improved = True
        val = float((W * ((x[E[:, 0]] * x[E[:, 1]]) == B)).sum() / Wt)
        best = max(best, val)
    return best


def maxcut_cpsat(n, E, B, W, time_limit=60, workers=10):
    from ortools.sat.python import cp_model
    mdl = cp_model.CpModel()
    x = [mdl.NewBoolVar(f"x{i}") for i in range(n)]
    mdl.Add(x[0] == 0)
    scale = 10 ** 6
    terms, tot = [], 0
    for (a, b), s, w in zip(E.tolist(), B.tolist(), W.tolist()):
        c = mdl.NewBoolVar("")
        if s < 0:
            mdl.Add(x[a] + x[b] == 1).OnlyEnforceIf(c); mdl.Add(x[a] == x[b]).OnlyEnforceIf(c.Not())
        else:
            mdl.Add(x[a] == x[b]).OnlyEnforceIf(c); mdl.Add(x[a] + x[b] == 1).OnlyEnforceIf(c.Not())
        iw = int(round(w * scale)); terms.append(iw * c); tot += iw
    mdl.Maximize(sum(terms))
    s_ = cp_model.CpSolver(); s_.parameters.max_time_in_seconds = time_limit; s_.parameters.num_workers = workers
    st = s_.Solve(mdl)
    return s_.ObjectiveValue() / tot, s_.BestObjectiveBound() / tot, s_.StatusName(st) == "OPTIMAL"


def sweep_group(G, kmax, record, iters, cp_time, out, device, verbose=True,
                min_dim=1, min_U=0.0, batch=64, refine_iters=15000):
    t0 = time.time()
    irr = complex_irreps(G)
    P = GroupSoS(G, device=device)
    classes = P.pair_reps
    dims_str = sorted(d for d, _ in irr)
    # stage 1
    rows = []
    for k in range(1, min(kmax, len(classes)) + 1):
        for gens in itertools.combinations(classes, k):
            for signs in itertools.product([-1, 1], repeat=k):
                if all(s > 0 for s in signs):
                    continue
                s2, (ki, dcar) = sos2_cayley(G, irr, gens, signs=signs, return_arg=True)
                if s2 >= 1 - 1e-9:
                    continue                                  # satisfiable, no gap possible
                P.set_instance(list(gens), signs=list(signs))
                E, B, W = P.edges()
                if len(E) < 3:
                    continue
                inc = local_search_cut(G.n, E, B, W, restarts=60)
                U = (1 - inc) / (1 - s2)
                rows.append({"gens": list(map(int, gens)), "signs": list(signs), "m": len(E), "sos2": s2,
                             "incumbent": inc, "U": U, "carrier_dim": int(dcar)})
    short = [r for r in rows if r["U"] > record and r["carrier_dim"] >= min_dim and r["U"] > min_U]
    t1 = time.time()
    # stage 2, batched: all survivors of a group through the symmetry-reduced ADMM at once
    done, best = [], None
    from group_sos import GroupSoSBatch
    from cayley_sdp import class_weights
    for b0 in range(0, len(short), batch):
        chunk = short[b0:b0 + batch]
        Bt = GroupSoSBatch(P).set_instances([r["gens"] for r in chunk], [r["signs"] for r in chunk])
        Bt.solve(iters=iters, tol=1e-9)
        lo_t, hi_t = Bt.certified_bounds()
        for r, lo, hi in zip(chunk, lo_t.tolist(), hi_t.tolist()):
            P.set_instance(r["gens"], signs=r["signs"])
            E, B, W = P.edges()
            opt, ub, pr = maxcut_cpsat(G.n, E, B, W, time_limit=cp_time)
            C4_hi = (1 - opt) / max(1 - min(1.0, hi), 1e-12)
            if C4_hi > record and (hi - lo) > 1e-5:        # could still beat the record: refine
                P.solve(iters=refine_iters, tol=1e-12)
                lo, hi = P.certified_bounds()
                C4_hi = (1 - opt) / max(1 - min(1.0, hi), 1e-12)
            C4_lo = (1 - ub) / max(1 - min(1.0, lo), 1e-12)
            C2 = (1 - ub) / max(1 - r["sos2"], 1e-12)
            rec = {**r, "group": G.name, "order": G.n, "irrep_dims": dims_str,
                   "opt": opt, "opt_ub": ub, "proved": pr, "sos4_lo": lo, "sos4_hi": hi,
                   "cert_width": hi - lo, "C2": C2, "C4_lo": C4_lo, "C4_hi": C4_hi,
                   "retention": (C4_lo - 1) / (C2 - 1) if C2 > 1 + 1e-9 else None}
            done.append(rec)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if best is None or C4_lo > best["C4_lo"]:
                best = rec
    summary = {"group": G.name, "order": G.n, "irrep_dims": dims_str, "n_classes": len(classes),
               "n_instances": len(rows), "n_shortlist": len(short), "n_solved": len(done),
               "max_U": max((r["U"] for r in rows), default=None),
               "best_C4_lo": best["C4_lo"] if best else None,
               "best_instance": {k: best[k] for k in ("gens", "signs", "opt", "sos4_lo", "C2", "C4_lo", "carrier_dim")} if best else None,
               "n_undecided": sum(1 for r in done if r["C4_hi"] > record),
               "stage1_s": round(t1 - t0, 1), "stage2_s": round(time.time() - t1, 1)}
    if verbose:
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in summary.items()}), flush=True)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_order", type=int, default=60)
    ap.add_argument("--min_order", type=int, default=6)
    ap.add_argument("--kmax", type=int, default=2)
    ap.add_argument("--record", type=float, default=1.093586)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--cp_time", type=float, default=30)
    ap.add_argument("--only", default=None)
    ap.add_argument("--min_dim", type=int, default=1, help="only route instances carried by an irrep of dim >= this to stage 2")
    ap.add_argument("--min_U", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default="../results/group_sweep.jsonl")
    ap.add_argument("--summary", default="../results/group_sweep_summary.jsonl")
    a = ap.parse_args()
    groups = [G for G in library(a.max_order) if G.n >= a.min_order]
    if a.only:
        groups = [G for G in groups if a.only.lower() in G.name.lower()]
    done = set()
    try:
        done = {json.loads(l)["group"] for l in open(a.summary) if l.strip()}
    except FileNotFoundError:
        pass
    groups = [G for G in groups if G.name not in done]
    si, sn = [int(x) for x in a.shard.split("/")]
    groups = [G for i, G in enumerate(groups) if i % sn == si]
    print(f"{len(groups)} groups (shard {a.shard}), orders {a.min_order}..{a.max_order}, kmax={a.kmax}, "
          f"min carrier dim {a.min_dim}, min U {a.min_U}", flush=True)
    overall = None
    for G in groups:
        dev = "cpu"                     # small complex blocks: LAPACK beats cuSOLVER latency
        try:
            s = sweep_group(G, a.kmax, a.record, a.iters, a.cp_time, a.out, dev,
                            min_dim=a.min_dim, min_U=a.min_U, batch=a.batch)
        except Exception as ex:
            print(f"!! {G.name}: {type(ex).__name__}: {str(ex)[:160]}", flush=True); continue
        with open(a.summary, "a") as f:
            f.write(json.dumps(s) + "\n")
        if s["best_C4_lo"] is not None and (overall is None or s["best_C4_lo"] > overall[0]):
            overall = (s["best_C4_lo"], G.name, s["best_instance"])
            print(f"### new overall best C_4 = {overall[0]:.6f} on {G.name}: {s['best_instance']}", flush=True)
    print("DONE; overall best:", overall, flush=True)
