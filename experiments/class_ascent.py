"""
class_ascent.py -- maximise the degree-4 gap shape C_4 over ALL Z_L-invariant
Boolean 2Lin instances on the cycle group Z_L (every class d = 1..(L-1)/2 with
either sign and any weight), by the same certified alternating ascent as
weight_ascent.py but in the symmetry-reduced solver (circulant_sos):

   * SoS step   : degree-4 SoS at the current class weights -> feasible
                  pseudo-expectation y_d -> pseudo-unsat u~_{d,b} = (1 - b y_d)/2.
   * LP step    : max t s.t. sum_{d,b} w_{d,b} u_{d,b}(x) >= t for every cut x
                  (CP-SAT separation), sum w u~ = 1, w >= 0.
                  t is a certified lower bound on C_4 of the new weights.
   * kicks      : multiplicative noise on the weights, then LP-polish; accept
                  if the certified value improves.
Everything reported is a certified LOWER bound on C_4 (feasible pseudo-expectation,
proved minimum cut); the final instance is re-certified two-sided.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from scipy.optimize import linprog

from circulant_sos import CirculantSoS
from weight_ascent import cpsat_min_unsat


def circ_instance(L, w):
    """w: (h, 2) weights for (class d, sign -1) and (class d, sign +1).
    Returns merged (gens, weights, signs, kappa): contradictory mass kappa is
    violated by every assignment and every pseudo-expectation alike."""
    h = (L - 1) // 2
    gens, wts, sg = [], [], []
    kappa = 0.0
    for d in range(1, h + 1):
        a, b = w[d - 1, 0], w[d - 1, 1]
        kappa += min(a, b)
        net = a - b
        if abs(net) > 1e-12:
            gens.append(d); wts.append(abs(net)); sg.append(-1.0 if net > 0 else 1.0)
    return gens, np.array(wts), np.array(sg), kappa


def cut_unsat_classes(L, x):
    """u_{d,b}(x): fraction of class-d edges violated by cut x, for b = -1 and +1."""
    h = (L - 1) // 2
    out = np.zeros((h, 2))
    for d in range(1, h + 1):
        prod = x * np.roll(x, -d)
        out[d - 1, 0] = np.mean(prod != -1)
        out[d - 1, 1] = np.mean(prod != 1)
    return out


class CircAscent:
    def __init__(self, L, device="cuda"):
        self.L = L; self.h = (L - 1) // 2
        self.P = CirculantSoS(L, device=device)
        self.pool = []
        rng = np.random.default_rng(L)
        for _ in range(32):
            self.pool.append(rng.choice([-1, 1], size=L))

    def sos(self, w, iters, tol):
        gens, wts, sg, kappa = circ_instance(self.L, w)
        P = self.P
        if not gens:                                   # fully contradictory: uniform pseudo-expectation
            return 0.5, 0.5, np.full((self.h, 2), 0.5), 0.0, kappa, (gens, wts, sg)
        P.set_instance(gens, weights=wts, signs=sg)
        P.solve(iters=iters, tol=tol)
        lo, hi = P.certified_bounds()
        y = P.y.clone(); y[0] = 1.0
        B = P.blocks(y)
        lam = float(torch.linalg.eigvalsh((B + B.conj().transpose(1, 2)) / 2).real.min())
        a = -lam / (1 - lam) if lam < 0 else 0.0
        yd = ((1 - a) * y)[P.pair_var].detach().cpu().numpy()      # feasible y_{0,d}
        ut = np.stack([(1 + yd) / 2, (1 - yd) / 2], axis=1)          # unsat for b=-1: (1 - (-1) y)/2
        return lo, hi, ut, a, kappa, (gens, wts, sg)

    def min_cut_unsat(self, w, time_limit=60):
        gens, wts, sg, kappa = circ_instance(self.L, w)
        L = self.L
        E, B, W = [], [], []
        for g, wt, s in zip(gens, wts, sg):
            for v in range(L):
                u = (v + g) % L
                if g * 2 != L or v < u:                 # each undirected edge exactly once
                    E.append((v, u)); B.append(s); W.append(wt / L)
        if not E:
            return np.ones(L, dtype=int), kappa, kappa, True
        E = np.array(E); B = np.array(B); W = np.array(W)
        xs, v, vb, pr = cpsat_min_unsat(L, E, B, W / W.sum(), time_limit=time_limit)
        scale = W.sum()
        return xs, v * scale + kappa, vb * scale + kappa, pr

    def lp(self, ut):
        h = self.h
        cols = 2 * h
        rows = np.array([cut_unsat_classes(self.L, x).reshape(-1) for x in self.pool])
        while True:
            c = np.zeros(cols + 1); c[-1] = -1
            A_ub = np.hstack([-rows, np.ones((len(rows), 1))]); b_ub = np.zeros(len(rows))
            utf = np.maximum(ut.reshape(-1), 1e-6)          # floor keeps the LP bounded; still a valid lower bound
            A_eq = np.zeros((1, cols + 1)); A_eq[0, :cols] = utf
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                          bounds=[(0, 1e4)] * cols + [(None, None)], method="highs")
            assert res.status == 0, res.message
            w = res.x[:cols].reshape(h, 2); t = res.x[-1]
            w = w / w.sum()
            xs, v, vb, pr = self.min_cut_unsat(w, time_limit=30)
            den = float(utf @ w.reshape(-1))                           # pseudo-unsat at normalised w
            if v < t * den - 1e-7:                                      # LP claims min cut unsat >= t*den
                self.pool.append(xs)
                rows = np.vstack([rows, cut_unsat_classes(self.L, xs).reshape(-1)])
                continue
            # working value at w: best-known min cut / pseudo-unsat (certified only if CP-SAT proved the cut;
            # the final instance is re-proved with a long time limit in value())
            return w, v / den, pr

    def value(self, w, iters, tol, cp_time=120):
        """Certified C_4 interval at w."""
        lo, hi, ut, a, kappa, _ = self.sos(w, iters, tol)
        xs, v, vb, pr = self.min_cut_unsat(w, time_limit=cp_time)
        # lo/hi are SoS4 values of the merged instance, whose total weight is 1 - 2 kappa
        s = 1 - 2 * kappa
        den_lo = (1 - hi) * s + kappa; den_hi = (1 - lo) * s + kappa
        return vb / den_hi, v / max(den_lo, 1e-15), pr, (hi - lo), ut

    def polish(self, w, rounds, iters, tol, verbose=True, stall=1e-7):
        best_t, best_w = -1.0, w.copy()
        for r in range(rounds):
            t0 = time.time()
            lo, hi, ut, a, kappa, inst = self.sos(w, iters, tol)
            w_new, t, pr = self.lp(ut)
            if verbose:
                print(f"    r={r:2d} SoS4 [{lo:.6f},{hi:.6f}] mix {a:.1e}  LP-> C4>= {t:.6f} (proved {pr})  "
                      f"supp {(w_new > 1e-9).sum()}  {time.time()-t0:.1f}s", flush=True)
            if t > best_t + stall:
                best_t, best_w = t, w_new.copy()
                w = w_new
            else:
                break
        return best_t, best_w

    def run(self, w0, rounds, iters, tol, kicks, sigma, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        t, w = self.polish(w0, rounds, iters, tol, verbose)
        print(f"  polished start: C4 >= {t:.6f}", flush=True)
        for k in range(kicks):
            wk = w * np.exp(sigma * rng.standard_normal(w.shape))
            wk += sigma * 0.05 * rng.random(w.shape) * w.max()      # let dead classes re-enter
            wk /= wk.sum()
            tk, wk = self.polish(wk, rounds, iters, tol, verbose=False)
            if tk > t + 1e-7:
                t, w = tk, wk
                print(f"  kick {k}: improved to C4 >= {t:.6f}  supp {(w > 1e-9).sum()}", flush=True)
            elif verbose:
                print(f"  kick {k}: {tk:.6f} (no gain)", flush=True)
        return t, w


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, nargs="+", default=[17, 41])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--iters", type=int, default=15000)
    ap.add_argument("--final_iters", type=int, default=100000)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--kicks", type=int, default=6)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--seed_gens", type=int, nargs="*", default=None, help="seed instance: classes (sign -1)")
    ap.add_argument("--out", default="../results/class_ascent.jsonl")
    a = ap.parse_args()
    for L in a.L:
        A = CircAscent(L)
        h = A.h
        seeds = []
        if a.seed_gens:
            w0 = np.zeros((h, 2)); w0[[g - 1 for g in a.seed_gens], 0] = 1.0; seeds.append(("given", w0))
        w0 = np.zeros((h, 2)); w0[:, 0] = 1.0; seeds.append(("all-minus", w0))
        rng = np.random.default_rng(L)
        seeds.append(("dirichlet", rng.dirichlet(np.ones(2 * h)).reshape(h, 2)))
        for name, w0 in seeds:
            print(f"== L={L} seed {name}", flush=True)
            t, w = A.run(w0, a.rounds, a.iters, a.tol, a.kicks, a.sigma)
            C_lo, C_hi, pr, width, _ = A.value(w, a.final_iters, 1e-11)
            gens, wts, sg, kappa = circ_instance(L, w)
            print(f"   FINAL L={L} seed {name}: C4 in [{C_lo:.6f}, {C_hi:.6f}]  proved {pr}  width {width:.1e}  "
                  f"gens {gens} signs {sg.tolist()} kappa {kappa:.4f}", flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps({"L": L, "seed": name, "C4_lo": C_lo, "C4_hi": C_hi, "proved": pr, "cert_width": width,
                                    "gens": gens, "weights": wts.tolist(), "signs": sg.tolist(), "kappa": kappa,
                                    "w": w.tolist()}) + "\n")
