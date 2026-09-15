"""
weight_ascent.py -- learn the extremal degree-4 gap shape on n vertices instead of
guessing a family.

    g_4(n) := sup over signed, weighted instances on n vertices of
              C_4(I) = (1 - opt(I)) / (1 - SoS_4(I)).

The Khot-Moshkovitz question at degree 4 is exactly whether g_4(n) -> infinity.
Instead of enumerating structured (Cayley) instances we maximise C_4 directly
over the weight simplex of the signed complete graph on n vertices (both signs
on every pair), by an alternating ascent that is monotone and rigorous:

  * SoS step: for the current weights w, solve degree-4 SoS; extract a FEASIBLE
    pseudo-expectation (identity mixing) and its per-edge pseudo-unsat u~_e.
    For every w', 1 - SoS_4(w') <= u~ . w'  (the fixed pseudo-expectation is a
    feasible point), hence  C_4(w') >= min_cuts u(x).w' / u~ . w'.
  * LP step: maximise that ratio over w' (Charnes-Cooper: u~ . w' = 1,
    max t s.t. u(x).w' >= t for every cut x).  Cuts enter by enumeration
    (n <= 18, all 2^(n-1) cuts) or by CP-SAT separation (larger n).
The LP value t is a certified lower bound on C_4(w') and never decreases across
rounds (the previous w is LP-feasible with value >= the previous certified
ratio).  At the end, opt is proved by CP-SAT / enumeration and SoS_4 is
re-certified two-sided, so the reported C_4 interval is rigorous.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch
from scipy.optimize import linprog

from sos_gpu import BooleanSoS


def signed_complete(n):
    pairs = list(itertools.combinations(range(n), 2))
    E = np.array(pairs + pairs, dtype=np.int64)
    B = np.array([-1.0] * len(pairs) + [1.0] * len(pairs))
    return E, B


def all_cut_unsat(n, E, B):
    """U[x, e] = 1 if cut x violates edge e (x_0 = +1 fixed), dtype float32."""
    k = n - 1
    idx = np.arange(2 ** k, dtype=np.int64)
    X = np.ones((2 ** k, n), dtype=np.int8)
    for v in range(1, n):
        X[:, v] = np.where((idx >> (v - 1)) & 1, -1, 1)
    prod = X[:, E[:, 0]] * X[:, E[:, 1]]                       # (2^k, m)
    return (prod != B.astype(np.int8)).astype(np.float32)


def feasible_pseudo_unsat(S, B):
    """Per-edge pseudo-unsat of the identity-mixed (hence feasible) pseudo-expectation."""
    y = S.y.clone(); y[0] = 1.0
    M = y[S.cls]
    lam = float(torch.linalg.eigvalsh((M + M.T) / 2)[0])
    a = -lam / (1 - lam) if lam < 0 else 0.0
    yf = (1 - a) * y; yf[0] = 1.0
    yv = yf.detach().cpu().numpy()
    pm = yv[S.pair_id.cpu().numpy()]
    return (1 - B * pm) / 2, a


def lp_step(Ucuts, ut):
    """max t s.t. Ucuts w >= t, ut.w = 1, w >= 0.  Returns (w normalised, t)."""
    m = Ucuts.shape[1]
    c = np.zeros(m + 1); c[-1] = -1.0
    A_ub = np.hstack([-Ucuts.astype(np.float64), np.ones((Ucuts.shape[0], 1))])
    b_ub = np.zeros(Ucuts.shape[0])
    A_eq = np.zeros((1, m + 1)); A_eq[0, :m] = ut
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * m + [(None, None)], method="highs")
    assert res.status == 0, res.message
    w = res.x[:m]; t = res.x[-1]
    return w / w.sum(), t


def cpsat_min_unsat(n, E, B, w, time_limit=30, workers=8):
    from ortools.sat.python import cp_model
    mdl = cp_model.CpModel()
    x = [mdl.NewBoolVar(f"x{i}") for i in range(n)]
    mdl.Add(x[0] == 0)
    scale = 10 ** 7; terms = []; tot = 0
    for (a, b), s, wt in zip(E.tolist(), B.tolist(), w.tolist()):
        iw = int(round(wt * scale))
        if iw == 0:
            continue
        c = mdl.NewBoolVar("")
        if s < 0:
            mdl.Add(x[a] + x[b] == 1).OnlyEnforceIf(c); mdl.Add(x[a] == x[b]).OnlyEnforceIf(c.Not())
        else:
            mdl.Add(x[a] == x[b]).OnlyEnforceIf(c); mdl.Add(x[a] + x[b] == 1).OnlyEnforceIf(c.Not())
        terms.append(iw * c); tot += iw
    mdl.Maximize(sum(terms))
    sv = cp_model.CpSolver(); sv.parameters.max_time_in_seconds = time_limit; sv.parameters.num_workers = workers
    st = sv.Solve(mdl)
    xs = np.array([1 if sv.Value(v) == 0 else -1 for v in x])
    return xs, 1 - sv.ObjectiveValue() / tot, 1 - sv.BestObjectiveBound() / tot, sv.StatusName(st) == "OPTIMAL"


class Ascent:
    def __init__(self, n, device="cuda", enum_limit=18, seed=0):
        self.n = n
        self.E, self.B = signed_complete(n)
        self.m = len(self.E)
        self.device = device
        self.S = BooleanSoS(n, self.E, degree=4, device=device, dtype=torch.float64)
        self.enum = n <= enum_limit
        self.Ucuts = all_cut_unsat(n, self.E, self.B) if self.enum else None
        self.rng = np.random.default_rng(seed)

    def min_unsat(self, w, exact_time=60):
        """(1 - opt, upper bound on 1 - opt, proved, cut)."""
        if self.enum:
            vals = self.Ucuts @ w.astype(np.float32)
            i = int(np.argmin(vals))
            v = float(self.Ucuts[i].astype(np.float64) @ w)
            return v, v, True, i
        xs, v, vb, pr = cpsat_min_unsat(self.n, self.E, self.B, w, time_limit=exact_time)
        return v, vb, pr, xs

    def sos(self, w, iters, tol, warm=None):
        S = self.S
        S.set_instance(self.B, w)
        S.solve(iters=iters, tol=tol, warm=warm)
        lo, hi = S.certified_bounds()
        ut, a = feasible_pseudo_unsat(S, self.B)
        return lo, hi, ut, a, (S.Z, S.U)

    def run(self, w0, rounds=40, iters=20000, tol=1e-9, pool=None, verbose=True, stall=1e-6):
        w = w0 / w0.sum()
        hist = []
        best = (0.0, w.copy())
        if not self.enum:
            pool = [] if pool is None else list(pool)
            for _ in range(64):
                pool.append(self.rng.choice([-1, 1], size=self.n))
        for r in range(rounds):
            t0 = time.time()
            lo, hi, ut, a, _ = self.sos(w, iters, tol)
            unsat, unsat_ub, proved, cut = self.min_unsat(w)
            C_lo_here = unsat / (1 - lo)
            if self.enum:
                w_new, t = lp_step(self.Ucuts, ut)
            else:
                while True:
                    Up = np.array([(x[self.E[:, 0]] * x[self.E[:, 1]] != self.B) for x in pool], dtype=np.float32)
                    w_new, t = lp_step(Up, ut)
                    xs, v, vb, pr = cpsat_min_unsat(self.n, self.E, self.B, w_new, time_limit=20)
                    if v < t - 1e-7:
                        pool.append(xs)
                        continue
                    t = min(t, v)
                    break
            hist.append({"round": r, "sos4_lo": lo, "sos4_hi": hi, "mix": a, "unsat": unsat, "C_lo": C_lo_here,
                         "lp_t": t, "support": int((w_new > 1e-9).sum()), "secs": time.time() - t0})
            if verbose:
                print(f"  n={self.n} r={r:2d}  SoS4 [{lo:.6f},{hi:.6f}] (mix {a:.1e})  1-opt {unsat:.6f}  "
                      f"C4>= {C_lo_here:.6f}  LP-> {t:.6f}  supp {hist[-1]['support']}  {time.time()-t0:.1f}s", flush=True)
            if C_lo_here > best[0]:
                best = (C_lo_here, w.copy())
            if t <= best[0] + stall and r > 2:
                break
            w = w_new
        return best, hist

    def verify(self, w, iters=60000, tol=1e-11):
        lo, hi, ut, a, _ = self.sos(w, iters, tol)
        unsat, unsat_ub, proved, _ = self.min_unsat(w, exact_time=600)
        return {"sos4_lo": lo, "sos4_hi": hi, "unsat": unsat, "unsat_ub": unsat_ub, "proved": proved,
                "C4_lo": unsat_ub / (1 - lo), "C4_hi": unsat / (1 - hi) if hi < 1 else float("inf")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--out", default="../results/weight_ascent.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    for n in a.n:
        A = Ascent(n, device=a.device)
        overall = None
        for s in range(a.starts):
            rng = np.random.default_rng(1000 * n + s)
            if s == 0:
                w0 = np.concatenate([np.ones(A.m // 2), np.zeros(A.m // 2)])          # K_n max-cut
            elif s == 1:
                w0 = rng.dirichlet(np.ones(A.m) * 0.3)
            else:
                w0 = rng.dirichlet(np.ones(A.m))
            print(f"== n={n} start {s}", flush=True)
            (Cb, wb), hist = A.run(w0, rounds=a.rounds, iters=a.iters, tol=a.tol)
            ver = A.verify(wb)
            rec = {"n": n, "start": s, "C4_ascent": Cb, "verify": ver, "rounds": len(hist),
                   "w": [round(float(x), 10) for x in wb], "hist": hist}
            print(f"   start {s}: C4 in [{ver['C4_lo']:.6f}, {ver['C4_hi']:.6f}]  proved={ver['proved']}  "
                  f"support {int((wb > 1e-9).sum())}", flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if overall is None or ver["C4_lo"] > overall[0]:
                overall = (ver["C4_lo"], s)
        print(f"#### n={n}: best certified C4 >= {overall[0]:.6f} (start {overall[1]})", flush=True)
