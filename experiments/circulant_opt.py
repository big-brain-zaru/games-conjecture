"""
circulant_opt.py -- Maximise the degree-4 gap shape over ALL Z_L-invariant
Boolean 2Lin instances (all-anti signs, arbitrary weights per difference class).

This space contains the odd cycle (weight on class d = 1 only), which is the
degree-2 champion with C_2(C_L) -> 4L/pi^2, and every weighted circulant.  The
instance stays Z_L-invariant, so the degree-4 relaxation is solved with the
symmetry-reduced certified solver (circulant_sos.py) and the optimum with CP-SAT.

Ascent on the weight simplex uses the exact Danskin gradients
    d R_4 / d w_d  =  psat_d - R_4 ,   psat_d = (1 - y_{ {0,d} }) / 2
    d opt / d w_d  =  sat_d  - opt ,   sat_d  = cut fraction of class d at x*
so  d C_4 / d w_d = [ (1-opt) dR_4 - (1-R_4) dopt ] / (1-R_4)^2 .
Every reported C_4 uses the certified lower bound on R_4 and the CP-SAT dual
bound on opt, so it is a rigorous lower bound on that instance's gap shape.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from circulant_sos import CirculantSoS
from circulant_scan import circ_edges, maxcut_cpsat, sos2_circulant


class WEval:
    def __init__(self, L, iters=8000, cp_time=30, device=None):
        self.L = L
        self.h = (L - 1) // 2
        self.device = device or ("cuda" if torch.cuda.is_available() and L >= 25 else "cpu")
        self.S = CirculantSoS(L, device=self.device)
        self.iters, self.cp_time = iters, cp_time
        self.cache = {}

    def __call__(self, w, cp_time=None):
        L, h = self.L, self.h
        w = np.maximum(np.asarray(w, float), 0.0)
        w = w / w.sum()
        act = [d + 1 for d in range(h) if w[d] > 1e-12]
        self.S.set_instance(act, weights=[w[d - 1] for d in act])
        self.S.solve(iters=self.iters, tol=1e-11)
        lo, hi = self.S.certified_bounds()
        R = min(1.0, lo)
        # pseudo-satisfaction per class:  psat_d = (1 - y_{0,d})/2
        y = self.S.y.detach().cpu().numpy()
        psat = np.array([(1 - y[int(self.S.pair_var[d].item())]) / 2 for d in range(h)])
        # exact optimum (weighted CP-SAT) and per-class satisfaction of the optimal cut
        x, opt_w_solver, bound_w, proved = self._best_cut(L, act, w, cp_time or self.cp_time)
        sat = np.zeros(h)
        for d in range(1, h + 1):
            if w[d - 1] <= 1e-12:
                continue
            cut = sum(1 for v in range(L) if x[v] != x[(v + d) % L])
            sat[d - 1] = cut / L
        opt_w = float(sum(w[d - 1] * sat[d - 1] for d in range(1, h + 1)))
        assert opt_w <= bound_w + 1e-6, (opt_w, bound_w)
        C = (1 - bound_w) / max(1 - R, 1e-12)          # rigorous: uses the CP-SAT dual bound
        return dict(C=C, opt=opt_w, opt_ub=bound_w, R=R, Rhi=min(1.0, hi),
                    psat=psat, sat=sat, proved=proved)

    def _best_cut(self, L, act, w, cp_time):
        """Weighted max cut assignment via CP-SAT with integer weights."""
        from ortools.sat.python import cp_model
        mdl = cp_model.CpModel()
        x = [mdl.NewBoolVar(f"x{i}") for i in range(L)]
        mdl.Add(x[0] == 0)
        terms = []
        scale = 10 ** 6
        for d in act:
            iw = int(round(w[d - 1] * scale))
            if iw <= 0:
                continue
            for v in range(L):
                u = (v + d) % L
                if u == v:
                    continue
                a, b = min(u, v), max(u, v)
                c = mdl.NewBoolVar("")
                mdl.Add(x[a] + x[b] == 1).OnlyEnforceIf(c)
                mdl.Add(x[a] == x[b]).OnlyEnforceIf(c.Not())
                terms.append(iw * c)
        mdl.Maximize(sum(terms))
        s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = cp_time; s.parameters.num_workers = 10
        st = s.Solve(mdl)
        tot = float(sum(int(round(w[d - 1] * scale)) * L for d in act if int(round(w[d - 1] * scale)) > 0))
        return ([s.Value(x[i]) for i in range(L)], s.ObjectiveValue() / tot, s.BestObjectiveBound() / tot,
                s.StatusName(st) == "OPTIMAL")


def search(L, steps=40, restarts=3, iters=8000, cp_time=25, seed=0, verbose=True):
    ev = WEval(L, iters=iters, cp_time=cp_time)
    h = ev.h
    rng = np.random.default_rng(seed)
    starts = [np.eye(h)[0]] + [np.ones(h)]                       # odd cycle, and all classes
    starts += [np.array([1.0 if d in (1, 2) else 0.0 for d in range(1, h + 1)])]
    for _ in range(restarts):
        starts.append(rng.dirichlet(np.ones(h) * 0.4))
    best = None
    for si, w0 in enumerate(starts):
        w = np.maximum(w0, 0); w = w / w.sum()
        r = ev(w)
        C = r["C"]
        eta = 1.0
        for t in range(steps):
            dR = r["psat"] - r["R"]
            dopt = r["sat"] - r["opt"]
            g = ((1 - r["opt"]) * dR - (1 - r["R"]) * dopt) / max(1 - r["R"], 1e-9) ** 2
            g = g / (np.abs(g).max() + 1e-12)
            improved = False
            for _ in range(5):
                w2 = w * np.exp(eta * g)
                if w2.sum() <= 0:
                    break
                w2 = w2 / w2.sum()
                r2 = ev(w2)
                if r2["C"] > C + 1e-7:
                    w, r, C = w2, r2, r2["C"]; improved = True; eta = min(eta * 1.6, 5.0); break
                eta *= 0.5
            if not improved and eta < 1e-3:
                break
        if best is None or C > best[0]:
            best = (C, w.copy(), r)
        if verbose:
            print(f"  L={L} start {si}: C_4 = {C:.6f}  (opt {r['opt']:.6f}, SoS4 {r['R']:.6f}, "
                  f"{int((w > 1e-6).sum())} active classes)", flush=True)
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", default="9,11,13,15,17,19,21,23,25")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--cp_time", type=float, default=25)
    ap.add_argument("--out", default="../results/circulant_opt.jsonl")
    a = ap.parse_args()
    overall = 0.0
    for L in [int(x) for x in a.Ls.split(",")]:
        if L % 2 == 0:
            continue
        t0 = time.time()
        C, w, r = search(L, steps=a.steps, restarts=a.restarts, iters=a.iters, cp_time=a.cp_time)
        overall = max(overall, C)
        rec = {"L": L, "C4": C, "opt": r["opt"], "sos4_lo": r["R"], "sos4_hi": r["Rhi"],
               "weights": w.tolist(), "active": int((w > 1e-6).sum()), "time": round(time.time() - t0, 1)}
        print(f"### L={L}: best C_4 = {C:.6f}   (running max {overall:.6f})   {rec['time']}s", flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
