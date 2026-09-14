"""
sos_gpu.py -- GPU ADMM solver for the degree-4 (and degree-2) sum-of-squares
moment relaxation of Max-2Lin(2) / Boolean unique games, with CERTIFIED bounds.

Primal:  max  sum_T c_T y_T   s.t.  M(y) = sum_T y_T E_T  >= 0,  y_{} = 1,
where T ranges over monomials x_T (|T| <= 4), the basis of M is {1, x_i, x_i x_j},
and E_T is the 0/1 pattern of positions (s,t) with s xor t = T.  Since x_i^2 = 1
the whole diagonal of M carries y_{} (E_{} = I).

Dual:    min  c_{} + tr(S)   s.t.  S >= 0,  <S, E_T> = -c_T  for all T != {}.
Given any symmetric S0, projecting each class T != {} to the right sum (adding a
constant on the class pattern) and then adding max(0, -lambda_min) * I keeps
every class constraint (I only touches T = {}) -> a rigorous UPPER bound.
Given any y, M(y) with lambda_min < 0 is mixed with the identity (y_{} = 1,
rest 0) -> a rigorous LOWER bound (feasible primal point).

ADMM on  min -<Ctil, M> + i_L(M) + i_PSD(Z),  M = Z, where L is the affine
subspace {M = P y, y_{} = 1}: projection onto L = average over each class
pattern (scatter/gather), projection onto PSD = eigh clip.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np


class BooleanSoS:
    def __init__(self, n: int, edges: np.ndarray, degree: int = 4, device: str = "cuda", dtype=None):
        import torch
        self.torch = torch
        self.n, self.degree, self.device = n, degree, device
        self.dtype = dtype or torch.float32
        self.edges = np.asarray(edges, dtype=np.int64)
        if degree == 2:
            basis = [()] + [(i,) for i in range(n)]
        else:
            basis = [()] + [(i,) for i in range(n)] + list(itertools.combinations(range(n), 2))
        self.D = D = len(basis)
        monos = {(): 0}
        cls = np.zeros((D, D), dtype=np.int64)
        for i, s in enumerate(basis):
            ss = set(s)
            for j, t in enumerate(basis):
                T = tuple(sorted(ss ^ set(t)))
                if T not in monos:
                    monos[T] = len(monos)
                cls[i, j] = monos[T]
        self.monos = monos
        self.nm = len(monos)
        self.cls = torch.tensor(cls, device=device)                       # (D, D) class id
        counts = torch.zeros(self.nm, device=device, dtype=self.dtype)
        counts.index_add_(0, self.cls.reshape(-1), torch.ones(D * D, device=device, dtype=self.dtype))
        self.counts = counts                                                # positions per class
        self.pair_id = torch.tensor([monos[(min(u, v), max(u, v))] if u != v else 0 for u, v in self.edges.tolist()],
                                    device=device)
        self.edge_is_loop = torch.tensor([u == v for u, v in self.edges.tolist()], device=device)

    # -- objective from (signs, weights) ------------------------------------
    def set_instance(self, signs, weights):
        torch = self.torch
        signs = torch.tensor(np.asarray(signs, dtype=np.float64), device=self.device, dtype=self.dtype)
        w = torch.tensor(np.asarray(weights, dtype=np.float64), device=self.device, dtype=self.dtype)
        w = w / w.sum()
        # satisfied iff x_u x_v = b: (1 + b x_u x_v)/2
        c = torch.zeros(self.nm, device=self.device, dtype=self.dtype)
        const = 0.5 * w[~self.edge_is_loop].sum() + (w[self.edge_is_loop] * (signs[self.edge_is_loop] > 0).to(self.dtype)).sum()
        coef = 0.5 * w * signs
        c.index_add_(0, self.pair_id[~self.edge_is_loop], coef[~self.edge_is_loop])
        self.c = c; self.c[0] = 0.0
        self.const = float(const)
        # Ctil matrix: value = const + sum_T c_T y_T ; y_T = mean of M over class T
        self.Ctil = (self.c / self.counts)[self.cls]                       # (D, D)
        return self

    # -- projections ----------------------------------------------------------
    def class_means(self, M):
        torch = self.torch
        sums = torch.zeros(self.nm, device=self.device, dtype=self.dtype)
        sums.index_add_(0, self.cls.reshape(-1), M.reshape(-1))
        return sums / self.counts

    def proj_L(self, M):
        y = self.class_means(M)
        y[0] = 1.0
        return y[self.cls], y

    def proj_psd(self, M):
        torch = self.torch
        M = (M + M.T) / 2
        w, U = torch.linalg.eigh(M)
        w = torch.clamp(w, min=0)
        return (U * w) @ U.T

    # -- solve ---------------------------------------------------------------
    def solve(self, iters: int = 3000, rho: float = None, tol: float = 1e-6, warm=None, verbose: bool = False,
              log_every: int = 200, alpha: float = 1.6, adapt_every: int = 25):
        """ADMM with over-relaxation (alpha) and residual-balancing rho adaptation."""
        torch = self.torch
        D = self.D
        if rho is None:
            rho = float(self.Ctil.norm()) * 2.0 / D + 1e-6
        if warm is None:
            Z = torch.eye(D, device=self.device, dtype=self.dtype)
            U = torch.zeros(D, D, device=self.device, dtype=self.dtype)
        else:
            Z, U = warm
        y = None
        for it in range(iters):
            M, y = self.proj_L(Z - U + self.Ctil / rho)
            Mh = alpha * M + (1 - alpha) * Z
            Znew = self.proj_psd(Mh + U)
            U = U + Mh - Znew
            r = float((M - Znew).norm()); s = float(rho * (Znew - Z).norm())
            Z = Znew
            if verbose and it % log_every == 0:
                print(f"    admm it {it:5d}  val {self.const + float(self.c @ y):.6f}  r {r:.2e} s {s:.2e} rho {rho:.2e}")
            if it % adapt_every == 0 and it > 0:
                if r > 10 * s:
                    rho *= 2.0; U = U / 2.0
                elif s > 10 * r:
                    rho /= 2.0; U = U * 2.0
            if r < tol and s < tol and it > 20:
                break
        self.Z, self.U, self.y, self.rho = Z, U, y, rho
        self.iters_done = it + 1
        return self.const + float(self.c @ y)

    # -- certificates ---------------------------------------------------------
    def certified_bounds(self):
        """(lower, upper): lower from a feasible primal point (identity mixing),
        upper from a feasible dual point built from the ADMM dual iterate."""
        torch = self.torch
        y = self.y.clone(); y[0] = 1.0
        M = y[self.cls]
        lam = float(torch.linalg.eigvalsh((M + M.T) / 2)[0])
        val = self.const + float(self.c @ y)
        if lam < 0:
            a = -lam / (1 - lam)
            lower = (1 - a) * val + a * self.const          # y -> (1-a) y + a e_{}
        else:
            lower = val
        # dual: S ~ -rho U ; enforce <S, E_T> = -c_T for T != {} by per-class constant shift
        S = -self.U * self.rho
        S = (S + S.T) / 2
        sums = torch.zeros(self.nm, device=self.device, dtype=self.dtype)
        sums.index_add_(0, self.cls.reshape(-1), S.reshape(-1))
        shift = (-self.c - sums) / self.counts
        shift[0] = 0.0
        S = S + shift[self.cls]
        lam_s = float(torch.linalg.eigvalsh((S + S.T) / 2)[0])
        if lam_s < 0:
            S = S + (-lam_s) * torch.eye(self.D, device=self.device, dtype=self.dtype)
        upper = self.const + float(torch.trace(S))
        return lower, upper

    def edge_pseudo_sat(self, signs):
        """psat_e = (1 + b_e y_{uv})/2 from the current y."""
        y = self.y.detach().cpu().numpy()
        pm = np.where(self.edge_is_loop.cpu().numpy(), 1.0, y[self.pair_id.cpu().numpy()])
        return (1 + np.asarray(signs) * pm) / 2


def _selftest():
    import torch
    from gap_search import complete_graph, cycle, boolean_game, brute_opt_boolean
    from ug_sos import sos4_boolean
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # C5 max-cut: degree 2 = 0.90451, degree 4 = 0.8
    edges = cycle(5); signs = -np.ones(5); w = np.ones(5)
    for deg, ref in ((2, (1 + math.cos(math.pi / 5)) / 2), (4, 0.8)):
        S = BooleanSoS(5, edges, degree=deg, device=dev, dtype=torch.float64).set_instance(signs, w)
        v = S.solve(iters=5000, tol=1e-9); lo, hi = S.certified_bounds()
        print(f"  C5 deg {deg}: admm {v:.6f}  certified [{lo:.6f}, {hi:.6f}]  ref {ref:.6f}  ({S.iters_done} its)")
        assert lo - 1e-5 <= ref <= hi + 1e-5 and hi - lo < 1e-3
    # random K9 instance vs cvxpy/SCS at degree 4
    rng = np.random.default_rng(3)
    edges = complete_graph(9); m = len(edges)
    signs = rng.choice([-1.0, 1.0], size=m); w = rng.dirichlet(np.ones(m))
    g = boolean_game(edges, signs.astype(int), w)
    ref, _ = sos4_boolean(g, degree=4, eps=1e-9)
    S = BooleanSoS(9, edges, degree=4, device=dev, dtype=torch.float64).set_instance(signs, w)
    t = time.time(); v = S.solve(iters=6000, tol=1e-8); lo, hi = S.certified_bounds()
    print(f"  K9 deg 4: admm {v:.6f}  certified [{lo:.6f}, {hi:.6f}]  scs {ref:.6f}  ({S.iters_done} its, {time.time()-t:.1f}s)")
    assert lo - 1e-4 <= ref <= hi + 1e-4
    # timing at n=30 (D=466) float32
    edges = complete_graph(30); m = len(edges)
    signs = rng.choice([-1.0, 1.0], size=m); w = rng.dirichlet(np.ones(m))
    S = BooleanSoS(30, edges, degree=4, device=dev).set_instance(signs, w)
    t = time.time(); v = S.solve(iters=1500, tol=1e-5); lo, hi = S.certified_bounds()
    print(f"  K30 deg 4 (D={S.D}): admm {v:.5f} certified [{lo:.5f}, {hi:.5f}] in {time.time()-t:.1f}s ({S.iters_done} its)")
    print("sos_gpu selftest: OK")


if __name__ == "__main__":
    _selftest()
