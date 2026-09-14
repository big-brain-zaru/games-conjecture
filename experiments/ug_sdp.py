"""
ug_sdp.py -- The basic SDP relaxation of Unique Games (Khot 2002 / Feige-Lovasz;
constraints (11)-(14) of Khot-Vishnoi Fig. 4, normalised so each vertex block
has trace 1), solved three ways:

  1. cvxpy (Clarabel / SCS): exact interior-point / ADMM, small instances (nk <= ~600).
  2. Burer-Monteiro on GPU (PyTorch): a *feasible* low-rank solution V (nk x r)
     with exact block structure, so its objective is a certified LOWER bound on
     the SDP value.  Scales to nk ~ 10^5.
  3. The rounding algorithms (basic Gaussian, propagation/CMM-style, plus local
     search polish), each returning an assignment whose value is re-verified.

Variables: X = V V^T, V has one row per (vertex, label).  For every vertex v:
    X[(v,a),(v,b)] = 0  (a != b),   sum_a X[(v,a),(v,a)] = 1.
Optional: X >= 0 entrywise (KV constraint (13)), and the "sum-to-one" constraint
    sum_{a,b} X[(u,a),(v,b)] = 1 on edges (KV constraint (14)).
Objective: sum_e w_e sum_a X[(u,a),(v,pi_e(a))] / W.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
from ug_core import UniqueGame, local_search_polish, greedy_coloring


# ----------------------------------------------------------------------------
# 1. cvxpy
# ----------------------------------------------------------------------------
def sdp_cvxpy(g: UniqueGame, nonneg: bool = False, sum_to_one_on_edges: bool = False,
              triangle_on_edges: bool = False, solver: str = "CLARABEL", verbose: bool = False,
              **solver_kw):
    """Solve the basic UG SDP exactly.  Returns (value, X, info)."""
    import cvxpy as cp
    n, k = g.n, g.k
    D = n * k
    X = cp.Variable((D, D), PSD=True)
    cons = []
    for v in range(n):
        blk = X[v * k:(v + 1) * k, v * k:(v + 1) * k]
        cons.append(cp.trace(blk) == 1)
        # off-diagonal zero
        mask = np.ones((k, k)) - np.eye(k)
        cons.append(cp.multiply(mask, blk) == 0)
    if nonneg:
        cons.append(X >= 0)
    # objective matrix C: C[(u,a),(v,pi(a))] = w_e / W (symmetrised)
    C = np.zeros((D, D))
    W = g.total_weight
    for e, (u, v) in enumerate(g.edges):
        for a in range(k):
            i, j = u * k + a, v * k + int(g.perms[e, a])
            C[i, j] += g.weights[e] / W / 2
            C[j, i] += g.weights[e] / W / 2
    if sum_to_one_on_edges:
        for e, (u, v) in enumerate(g.edges):
            cons.append(cp.sum(X[u * k:(u + 1) * k, v * k:(v + 1) * k]) == 1)
    if triangle_on_edges:
        # 1 + <x,y> >= <x,z> + <y,z> restricted to x,y in adjacent blocks, z anything: too many;
        # we add the l2^2 triangle inequalities among the three points 0 (origin), x, y:
        # ||x-y||^2 <= ||x||^2 + ||y||^2 is implied by nonneg.  Placeholder for future use.
        pass
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(C, X))), cons)
    t0 = time.time()
    prob.solve(solver=solver, verbose=verbose, **solver_kw)
    info = {"status": prob.status, "time": time.time() - t0, "solver": solver}
    return float(prob.value), np.asarray(X.value), info


# ----------------------------------------------------------------------------
# 2. Burer-Monteiro on GPU
# ----------------------------------------------------------------------------
class BMState:
    def __init__(self, g: UniqueGame, r: int, device: str = "cuda", seed: int = 0, dtype=None):
        import torch
        self.torch = torch
        self.g = g
        self.r = r
        self.device = device
        self.dtype = dtype or torch.float32
        torch.manual_seed(seed)
        n, k = g.n, g.k
        assert r >= k, "rank must be >= alphabet size for the exact Stiefel parametrisation"
        self.G = torch.randn(n, k, r, device=device, dtype=self.dtype, requires_grad=True)   # free -> Q via QR
        self.d = torch.randn(n, k, device=device, dtype=self.dtype, requires_grad=True)      # free -> unit vector
        self.eu = torch.tensor(g.edges[:, 0], device=device)
        self.ev = torch.tensor(g.edges[:, 1], device=device)
        self.perm = torch.tensor(g.perms, device=device)
        self.w = torch.tensor(g.weights / g.total_weight, device=device, dtype=self.dtype)

    def vectors(self):
        """Feasible V: block v = diag(d_v) Q_v with Q_v orthonormal rows."""
        torch = self.torch
        Q, _ = torch.linalg.qr(self.G.transpose(1, 2))         # (n, r, k) with orthonormal columns
        Q = Q.transpose(1, 2)                                   # (n, k, r) orthonormal rows
        dn = self.d / self.d.norm(dim=1, keepdim=True)          # unit vector per vertex
        return dn[:, :, None] * Q                                # (n, k, r)

    def objective(self, V=None):
        torch = self.torch
        V = self.vectors() if V is None else V
        Vu = V[self.eu]                                          # (m, k, r)
        Vv = torch.gather(V[self.ev], 1, self.perm[:, :, None].expand(-1, -1, self.r))  # rows permuted
        return (self.w * (Vu * Vv).sum(dim=(1, 2))).sum()

    def optimize(self, iters: int = 2000, lr: float = 0.05, nonneg_penalty: float = 0.0,
                 verbose: bool = False, log_every: int = 200):
        torch = self.torch
        opt = torch.optim.Adam([self.G, self.d], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
        best = -1.0
        for it in range(iters):
            opt.zero_grad()
            V = self.vectors()
            obj = self.objective(V)
            loss = -obj
            if nonneg_penalty > 0:
                # penalise negative inner products on edges only (cheap proxy for X >= 0)
                Vu = V[self.eu]; Vv = V[self.ev]
                ip = torch.einsum("mar,mbr->mab", Vu, Vv)
                loss = loss + nonneg_penalty * (torch.clamp(-ip, min=0) ** 2).sum()
            loss.backward()
            opt.step(); sched.step()
            val = float(obj)
            best = max(best, val)
            if verbose and (it % log_every == 0 or it == iters - 1):
                print(f"  BM it {it:5d}  obj {val:.6f}")
        return best

    def numpy_vectors(self) -> np.ndarray:
        with self.torch.no_grad():
            return self.vectors().detach().cpu().numpy()


def sdp_burer_monteiro(g: UniqueGame, r: Optional[int] = None, iters: int = 2000, lr: float = 0.05,
                       restarts: int = 1, device: str = "cuda", verbose: bool = False, **kw):
    """Feasible low-rank SDP solution; returns (value_lower_bound, V (n,k,r), info).
    The returned value is recomputed in float64 numpy from V for verification."""
    r = r or max(g.k, min(64, int(math.ceil(math.sqrt(2 * g.n))) + g.k))
    best_val, best_V = -1.0, None
    t0 = time.time()
    for s in range(restarts):
        st = BMState(g, r, device=device, seed=s)
        st.optimize(iters=iters, lr=lr, verbose=verbose, **kw)
        V = st.numpy_vectors().astype(np.float64)
        val = sdp_value_of_vectors(g, V)
        if val > best_val:
            best_val, best_V = val, V
    return best_val, best_V, {"r": r, "iters": iters, "restarts": restarts, "time": time.time() - t0}


def sdp_value_of_vectors(g: UniqueGame, V: np.ndarray, check_feasible: bool = True, tol: float = 1e-4) -> float:
    """Objective of a vector solution V (n,k,r); asserts block feasibility."""
    n, k, r = V.shape
    if check_feasible:
        Gm = np.einsum("nar,nbr->nab", V, V)
        off = Gm - np.einsum("nab,ab->nab", Gm, np.eye(k))
        assert np.abs(off).max() < tol, f"orthogonality violated: {np.abs(off).max()}"
        tr = np.einsum("naa->n", Gm)
        assert np.abs(tr - 1).max() < tol, f"trace violated: {np.abs(tr-1).max()}"
    Vu = V[g.edges[:, 0]]
    Vv = np.take_along_axis(V[g.edges[:, 1]], g.perms[:, :, None], axis=1)
    return float((g.weights * np.einsum("mar,mar->m", Vu, Vv)).sum() / g.total_weight)


def gram_to_vectors(X: np.ndarray, n: int, k: int, tol: float = 1e-7) -> np.ndarray:
    """Factor a PSD Gram matrix into vectors (n, k, r)."""
    w, U = np.linalg.eigh((X + X.T) / 2)
    w = np.clip(w, 0, None)
    keep = w > tol * w.max()
    V = U[:, keep] * np.sqrt(w[keep])[None, :]
    return V.reshape(n, k, -1)


# ----------------------------------------------------------------------------
# 2b. Exact block-coordinate ascent (the workhorse) + dual certificate
# ----------------------------------------------------------------------------
def _greedy_coloring(g: UniqueGame) -> np.ndarray:
    """Proper vertex colouring of the base graph (greedy, largest-first)."""
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(g.n))
    G.add_edges_from(map(tuple, g.edges[g.edges[:, 0] != g.edges[:, 1]].tolist()))
    col = nx.coloring.greedy_color(G, strategy="largest_first")
    return np.array([col[v] for v in range(g.n)])


class BlockAscent:
    """Maximise sum_e w_e sum_a <B_u[a], B_v[pi_e(a)]> over blocks B_v (k x r) with
    mutually orthogonal rows and total squared norm 1, by exact block updates:
    for vertex v the objective is linear, sum_a <b_a, g_a>, with g the
    weighted sum of matched neighbour rows.  Writing b_a = d_a q_a (Q Stiefel,
    d unit) we alternate  Q <- polar(diag(d) G)  (Procrustes)  and
    d <- c/|c|, c_a = <q_a, g_a>.  Blocks of one colour class are updated
    simultaneously (no edges inside a class -> exact Gauss-Seidel)."""

    def __init__(self, g: UniqueGame, r: int, device="cuda", seed=0, dtype=None, init=None):
        import torch
        self.torch, self.g, self.r, self.device = torch, g, r, device
        self.dtype = dtype or torch.float32
        gen = torch.Generator(device="cpu").manual_seed(seed)
        n, k = g.n, g.k
        if init is None:
            B = torch.randn(n, k, r, generator=gen).to(device=device, dtype=self.dtype)
            B = self._project_blocks(B)
        else:
            B = torch.tensor(init, device=device, dtype=self.dtype)
        self.B = B
        eu, ev = g.edges[:, 0], g.edges[:, 1]
        inv = g.inverse_perms()
        src = np.concatenate([eu, ev]); dst = np.concatenate([ev, eu])
        prm = np.concatenate([g.perms, inv])              # row a of src matches row prm[a] of dst
        w = np.concatenate([g.weights, g.weights]) / g.total_weight
        loops = src == dst
        src, dst, prm, w = src[~loops], dst[~loops], prm[~loops], w[~loops]
        self.src = torch.tensor(src, device=device); self.dst = torch.tensor(dst, device=device)
        self.prm = torch.tensor(prm, device=device); self.w = torch.tensor(w, device=device, dtype=self.dtype)
        self.colors = greedy_coloring(g)
        ncol = self.colors.max() + 1
        self.color_classes = [torch.tensor(np.where(self.colors == c)[0], device=device) for c in range(ncol)]
        self.class_edges = [torch.tensor(np.where(self.colors[src] == c)[0], device=device) for c in range(ncol)]

    def _project_blocks(self, B):
        torch = self.torch
        Q, _ = torch.linalg.qr(B.transpose(1, 2)); Q = Q.transpose(1, 2)
        d = B.norm(dim=2); d = d / d.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return d[:, :, None] * Q

    def gradient_field(self, sel, chunk_elems: int = 2 ** 27):
        """G_v[a] = sum over directed edges (v->u) in sel of w * B_u[pi(a)] (chunked)."""
        torch = self.torch
        G = torch.zeros_like(self.B)
        step = max(1, chunk_elems // (self.g.k * self.r))
        for i in range(0, len(sel), step):
            ss = sel[i:i + step]
            s, t, p, w = self.src[ss], self.dst[ss], self.prm[ss], self.w[ss]
            rows = torch.gather(self.B[t], 1, p[:, :, None].expand(-1, -1, self.r))
            G.index_add_(0, s, w[:, None, None] * rows)
        return G

    def _block_update(self, idx, G, inner=3):
        torch = self.torch
        Gi = G[idx]
        d = self.B[idx].norm(dim=2)
        for _ in range(inner):
            M = d[:, :, None] * Gi
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            Q = U @ Vh
            c = (Q * Gi).sum(dim=2).clamp_min(0)
            d = c / c.norm(dim=1, keepdim=True).clamp_min(1e-12)
        self.B[idx] = d[:, :, None] * Q

    def objective(self, chunk_elems: int = 2 ** 27) -> float:
        torch = self.torch
        tot = 0.0
        step = max(1, chunk_elems // (self.g.k * self.r))
        for i in range(0, len(self.src), step):
            s, t, p, w = self.src[i:i+step], self.dst[i:i+step], self.prm[i:i+step], self.w[i:i+step]
            rows = torch.gather(self.B[t], 1, p[:, :, None].expand(-1, -1, self.r))
            tot += float((w * (self.B[s] * rows).sum(dim=(1, 2))).sum())
        return tot / 2

    def run(self, sweeps=200, tol=1e-7, verbose=False, log_every=20):
        prev = self.objective()
        hist = [prev]
        for s in range(sweeps):
            for c, idx in enumerate(self.color_classes):
                G = self.gradient_field(self.class_edges[c])
                self._block_update(idx, G)
            cur = self.objective(); hist.append(cur)
            if verbose and (s % log_every == 0):
                print(f"  sweep {s:4d}  obj {cur:.7f}")
            if cur - prev < tol:
                break
            prev = cur
        return hist

    def numpy_vectors(self):
        return self.B.detach().cpu().numpy().astype(np.float64)


def sdp_block_ascent(g: UniqueGame, r: Optional[int] = None, sweeps: int = 300, restarts: int = 1,
                     device: str = "cuda", verbose: bool = False, tol: float = 1e-8, seed0: int = 0):
    """Feasible primal solution by block ascent.  Returns (value, V (n,k,r), info)."""
    r = r or max(g.k + 2, min(96, int(math.ceil(math.sqrt(2 * g.n))) + g.k))
    best, bestV, t0 = -1.0, None, time.time()
    for s in range(restarts):
        ba = BlockAscent(g, r, device=device, seed=seed0 + s)
        ba.run(sweeps=sweeps, tol=tol, verbose=verbose)
        V = ba.numpy_vectors(); val = sdp_value_of_vectors(g, V)
        if val > best:
            best, bestV = val, V
    return best, bestV, {"r": r, "sweeps": sweeps, "restarts": restarts, "time": time.time() - t0}


def dual_certificate(g: UniqueGame, V: np.ndarray, tol_eig: float = 1e-9):
    """Certified UPPER bound on the SDP value from a primal iterate V.
    Dual:  min sum_v y_v  s.t.  S = sum_v (y_v I_v + Z_v) - C >= 0, Z_v symmetric,
    zero-diagonal, supported on block v.  From V: Lambda_v = (C V)_v V_v^T D_v^{-1}
    (stationarity), symmetrised; then shift y by -lambda_min(S) if negative.
    Returns (upper_bound, lambda_min_before_shift).  Valid for ANY V."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    n, k, r = V.shape
    D = n * k
    C = g.label_extended_adjacency(symmetric=True) / (2 * g.total_weight)
    Vf = V.reshape(D, r)
    CV = (C @ Vf).reshape(n, k, r)
    Dg = np.einsum("nar,nar->na", V, V)
    dn = np.sqrt(np.maximum(Dg, 1e-300))
    Vn = V / dn[:, :, None]
    Lam = np.einsum("nar,nbr->nab", CV, Vn) / dn[:, None, :]
    Lam = (Lam + Lam.transpose(0, 2, 1)) / 2
    for v in range(n):
        z = Dg[v] < 1e-10
        if z.any():
            Lam[v][z, :] = 0; Lam[v][:, z] = 0
            Lam[v][np.ix_(z, z)] = np.diag(np.full(int(z.sum()), Lam[v].diagonal().max()))
    # dual variable y_v must be a single scalar per block: take y_v = max_a Lam_v[a,a]
    # (raising the diagonal keeps S PSD-or-better); keep the off-diagonal part as Z_v.
    y = Lam.diagonal(axis1=1, axis2=2).max(axis=1)
    for v in range(n):
        np.fill_diagonal(Lam[v], y[v])
    blocks = sp.block_diag([sp.csr_matrix(Lam[v]) for v in range(n)], format="csr")
    S = (blocks - C).tocsr()
    S = ((S + S.T) / 2).tocsr()
    if D <= 1500:
        lam_min = float(np.linalg.eigvalsh(S.toarray()).min())
    else:
        lam_min = float(spla.eigsh(S, k=1, which="SA", tol=tol_eig, maxiter=50000)[0][0])
    ysum = float(y.sum())
    shift = max(0.0, -lam_min)
    return ysum + n * shift, lam_min


# ----------------------------------------------------------------------------
# 3. Rounding
# ----------------------------------------------------------------------------
def round_gaussian(g: UniqueGame, V: np.ndarray, trials: int = 50, polish: bool = True, seed: int = 0):
    """Basic rounding: random Gaussian direction z; label(v) = argmax_a <z, u_{v,a}>.
    Returns (best_value, best_assignment)."""
    rng = np.random.default_rng(seed)
    n, k, r = V.shape
    best, bestL = -1.0, None
    for _ in range(trials):
        z = rng.standard_normal(r)
        L = np.argmax(V @ z, axis=1)
        if polish:
            L = local_search_polish(g, L, rounds=20, rng=rng)
        val = g.value(L)
        if val > best:
            best, bestL = val, L
    return best, bestL


def round_propagation(g: UniqueGame, V: np.ndarray, trials: int = 50, polish: bool = True, seed: int = 0):
    """Propagation rounding (Khot 2002 / Trevisan): pick a random (v0, a0) weighted
    by |u_{v0,a0}|^2; label(w) = argmax_b <u_{v0,a0}, u_{w,b}>."""
    rng = np.random.default_rng(seed)
    n, k, r = V.shape
    norms = np.einsum("nar,nar->na", V, V).ravel()
    p = norms / norms.sum()
    best, bestL = -1.0, None
    for _ in range(trials):
        idx = rng.choice(n * k, p=p)
        v0, a0 = divmod(idx, k)
        z = V[v0, a0]
        L = np.argmax(V @ z, axis=1)
        if polish:
            L = local_search_polish(g, L, rounds=20, rng=rng)
        val = g.value(L)
        if val > best:
            best, bestL = val, L
    return best, bestL


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _selftest():
    from ug_core import random_unique_game, planted_unique_game, maxsat_optimum
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # A: small random instance; exact SDP >= true optimum; BM feasible <= exact SDP (up to tol)
    g = random_unique_game(12, 3, m=30, seed=1)
    opt, L, _ = maxsat_optimum(g)
    sdp, X, info = sdp_cvxpy(g)
    assert sdp >= opt - 1e-6, (sdp, opt)
    V = gram_to_vectors(X, g.n, g.k)
    val_from_V = sdp_value_of_vectors(g, V, tol=1e-3)
    assert abs(val_from_V - sdp) < 1e-3
    bm, Vb, binfo = sdp_block_ascent(g, r=8, sweeps=300, device=dev)
    ub, lmin = dual_certificate(g, Vb)
    assert bm <= sdp + 2e-4 and ub >= sdp - 2e-4 and ub - bm < 1e-3, (bm, sdp, ub)
    print(f"  block-ascent {bm:.5f} <= exact {sdp:.5f} <= dual {ub:.5f}  (lambda_min {lmin:.2e})")
    rg, Lg = round_gaussian(g, Vb, trials=30)
    rp, Lp = round_propagation(g, Vb, trials=30)
    assert rg <= opt + 1e-9 and rp <= opt + 1e-9
    print(f"  A: opt={opt:.4f}  sdp={sdp:.4f}  bm={bm:.4f}  round_g={rg:.4f}  round_p={rp:.4f}")
    # B: planted instance with noise: rounding recovers near-planted
    g = planted_unique_game(60, 5, degree=6, noise=0.15, seed=2)
    bm, Vb, _ = sdp_block_ascent(g, r=12, sweeps=300, device=dev)
    ub, lmin = dual_certificate(g, Vb)
    rp, Lp = round_propagation(g, Vb, trials=30)
    print(f"  B: planted={g.meta['planted_value']:.4f}  block-ascent={bm:.5f}  dual={ub:.5f} (SCS exact 0.87243)  round_p={rp:.4f}")
    assert bm >= g.meta['planted_value'] - 1e-6
    assert rp >= g.meta["planted_value"] - 0.05
    print("ug_sdp selftest: OK")


if __name__ == "__main__":
    _selftest()
