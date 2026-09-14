"""
circulant_sos.py -- Degree-4 sum-of-squares for Z_L-INVARIANT Boolean 2Lin
instances (circulant Max-Cut), block-diagonalised by the cyclic symmetry.

Why.  The full degree-4 moment matrix has D = 1 + L + C(L,2) rows (L=81 -> 3322),
so a dense eigendecomposition per ADMM step is hopeless.  For a Z_L-invariant
instance the optimal pseudo-moments may be taken Z_L-invariant, and then the
moment matrix commutes with the cyclic shift and splits into L blocks of size
1 + (L-1)/2.  For L = 81 that is 81 blocks of 41 instead of one block of 3322.

Set-up (L odd).  Monomial basis {1} u {x_i} u {x_i x_j}; the Z_L-orbits of the
basis are  r_0 = {} (fixed),  r_1 = {0},  r_{1+d} = {0, d} for d = 1..(L-1)/2,
all free.  Moments y are indexed by Z_L-orbits of subsets of size <= 4
(M[S,T] = y_{S xor T}).  Writing omega = e^{2 pi i / L}, the isotypic blocks are

  Mhat(alpha)_{ij} = sum_{u in Z_L} omega^{alpha u} y[ orb( r_i xor (u + r_j) ) ]     (i, j free)
  Mhat(0)_{0j}     = sqrt(L) * y[ orb(r_j) ],     Mhat(0)_{00} = y_{} = 1
  (the fixed orbit contributes only to alpha = 0)

and  M >= 0  <=>  Mhat(alpha) >= 0 for every alpha.  Verified against the dense
spectrum in _selftest.

Objective (all-anti Max-Cut, Z_L-invariant weights w_d on the difference class d):
  value = 1/2 - (1/2) * sum_d w_d y[orb({0,d})] / sum_d w_d.

Solver: ADMM (as in sos_gpu) in the orbit space; both the primal (a feasible
pseudo-moment after mixing with the trivial one) and the dual (a feasible dual
point after a per-class shift and an identity shift) give CERTIFIED bounds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch


# ----------------------------------------------------------------------------
def orbit_key(S, L):
    """Canonical form of a subset S of Z_L under rotation: the lexicographically
    smallest rotation of the sorted tuple."""
    if not S:
        return ()
    s = sorted(S)
    best = None
    for a in s:
        t = tuple(sorted((x - a) % L for x in s))
        if best is None or t < best:
            best = t
    return best


class CirculantSoS:
    def __init__(self, L: int, device="cuda", dtype=torch.float64):
        assert L % 2 == 1, "L must be odd (every pair orbit is then free)"
        self.L, self.device = L, device
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        h = (L - 1) // 2
        self.h = h
        # orbit representatives of the monomial basis
        reps = [(), (0,)] + [(0, d) for d in range(1, h + 1)]
        self.reps = reps
        self.nI = len(reps)              # 2 + h
        self.nfree = self.nI - 1         # rows present in every block
        t0 = time.time()
        # table: for free i, j and u in Z_L, the orbit of r_i xor (u + r_j)
        orbs = {(): 0}
        T = np.zeros((self.nfree, self.nfree, L), dtype=np.int64)
        for i in range(1, self.nI):
            Si = set(reps[i])
            for j in range(1, self.nI):
                for u in range(L):
                    Sj = set((x + u) % L for x in reps[j])
                    key = orbit_key(Si ^ Sj, L)
                    if key not in orbs:
                        orbs[key] = len(orbs)
                    T[i - 1, j - 1, u] = orbs[key]
        self.orbs = orbs
        self.n_var = len(orbs)
        self.T = torch.tensor(T, device=device)
        # the alpha = 0 extra row: y[orb(r_j)]
        self.row0 = torch.tensor([orbs[orbit_key(set(reps[j]), L)] for j in range(1, self.nI)], device=device)
        counts = torch.zeros(self.n_var, device=device, dtype=dtype)
        counts.index_add_(0, self.T.reshape(-1), torch.ones(self.T.numel(), device=device, dtype=dtype))
        self.counts = counts
        # ||blocks(e_T)||^2 : L*counts (free part, Parseval) + 2L per alpha=0 fixed-row hit + [T = empty]
        norm = L * counts
        norm.index_add_(0, self.row0, 2.0 * L * torch.ones(self.nfree, device=device, dtype=dtype))
        norm[0] += 1.0
        self.norm = torch.where(norm > 0, norm, torch.ones_like(norm))
        # mask: for alpha != 0 the fixed orbit has no row
        self.mask = torch.ones(L, self.nI, self.nI, dtype=dtype, device=device)
        self.mask[1:, 0, :] = 0.0
        self.mask[1:, :, 0] = 0.0
        self.pair_var = torch.tensor([orbs[orbit_key({0, d}, L)] for d in range(1, h + 1)], device=device)
        self.build_time = time.time() - t0

    # -- instance ------------------------------------------------------------
    def set_instance(self, gens, weights=None, signs=None):
        """Z_L-invariant Boolean 2Lin on Cay(Z_L, +-gens): the class-d constraint is
        x_v x_{v+d} = b_d (signs; default -1 = Max-Cut), with Z_L-invariant weights."""
        L, h = self.L, self.h
        w = np.zeros(h)
        gens = [g % L for g in gens]
        gens = sorted(set(min(g, L - g) for g in gens if g % L != 0))
        for i, g in enumerate(gens):
            w[g - 1] = 1.0 if weights is None else float(weights[i])
        assert w.sum() > 0
        self.w = w / w.sum()
        self.gens = gens
        b = np.full(h, -1.0)
        if signs is not None:
            for i, g in enumerate(gens):
                b[g - 1] = float(signs[i])
        self.b = b
        # satisfied iff x_v x_{v+d} = b_d  ->  value = 1/2 + 1/2 sum_d w_d b_d y[{0,d}]
        c = torch.zeros(self.n_var, device=self.device, dtype=self.dtype)
        wt = torch.tensor(self.w * self.b, device=self.device, dtype=self.dtype)
        c.index_add_(0, self.pair_var, 0.5 * wt)
        self.c = c
        self.const = 0.5
        return self

    # -- operators -----------------------------------------------------------
    def blocks(self, y):
        """y (n_var,) -> (L, nI, nI) complex Hermitian blocks (row/col 0 used only for alpha=0)."""
        L, nI = self.L, self.nI
        Y = y[self.T]                                        # (nfree, nfree, L)
        Mh = torch.fft.fft(Y.to(self.cdtype), dim=-1)        # sum_u y * omega^{-alpha u} ... sign fixed below
        Mh = Mh.permute(2, 0, 1).contiguous()                # (L, nfree, nfree)
        out = torch.zeros(L, nI, nI, dtype=self.cdtype, device=self.device)
        out[:, 1:, 1:] = Mh
        out[0, 0, 0] = y[0]
        out[0, 0, 1:] = np.sqrt(L) * y[self.row0].to(self.cdtype)
        out[0, 1:, 0] = np.sqrt(L) * y[self.row0].to(self.cdtype)
        return out

    def blocks_adjoint(self, B):
        """adjoint of `blocks`: (L, nI, nI) -> (n_var,)"""
        L = self.L
        Mh = B[:, 1:, 1:].permute(1, 2, 0).contiguous()      # (nfree, nfree, L)
        Y = torch.fft.ifft(Mh, dim=-1) * L                    # adjoint of fft (up to conj) -> real part
        Y = Y.real.to(self.dtype)
        out = torch.zeros(self.n_var, device=self.device, dtype=self.dtype)
        out.index_add_(0, self.T.reshape(-1), Y.reshape(-1))
        out[0] += B[0, 0, 0].real
        out.index_add_(0, self.row0, 2 * np.sqrt(L) * B[0, 0, 1:].real.to(self.dtype))
        return out

    def proj_L(self, B):
        """Project block-tensor onto the image of `blocks` with y_{} = 1."""
        y = self.blocks_adjoint(B) / self.norm
        y[0] = 1.0
        return self.blocks(y), y

    def proj_psd(self, B):
        B = (B + B.conj().transpose(1, 2)) / 2 * self.mask
        w, U = torch.linalg.eigh(B)
        w = torch.clamp(w.real, min=0).to(B.dtype)
        return ((U * w[:, None, :]) @ U.conj().transpose(1, 2)) * self.mask

    # -- solve ---------------------------------------------------------------
    def solve(self, iters=3000, rho=None, tol=1e-9, verbose=False, log_every=250, alpha=1.7):
        L, nI = self.L, self.nI
        Ctil = self.blocks(self.c / self.norm)     # <Ctil, blocks(y)> = c . y
        rho = rho or max(1e-3, float(Ctil.abs().max()) * 10)
        Z = torch.zeros(L, nI, nI, dtype=self.cdtype, device=self.device)
        Z[:] = torch.eye(nI, dtype=self.cdtype, device=self.device)
        Z = Z * self.mask
        U = torch.zeros_like(Z)
        y = None
        for it in range(iters):
            M, y = self.proj_L(Z - U + Ctil / rho)
            Mh = alpha * M + (1 - alpha) * Z
            Znew = self.proj_psd(Mh + U)
            U = U + Mh - Znew
            r = float((M - Znew).norm()); s = float(rho * (Znew - Z).norm())
            Z = Znew
            if it % 40 == 0 and it > 0:
                if r > 10 * s:
                    rho *= 2; U = U / 2
                elif s > 10 * r:
                    rho /= 2; U = U * 2
            if verbose and it % log_every == 0:
                print(f"    it {it:5d} value {self.const + float(self.c @ y):.8f} r {r:.2e} s {s:.2e}", flush=True)
            if r < tol and s < tol and it > 30:
                break
        self.Z, self.U, self.y, self.rho, self.iters_done = Z, U, y, rho, it + 1
        return self.const + float(self.c @ y)

    def certified_bounds(self):
        y = self.y.clone(); y[0] = 1.0
        B = self.blocks(y)
        lam = float(torch.linalg.eigvalsh((B + B.conj().transpose(1, 2)) / 2).real.min())
        val = self.const + float(self.c @ y)
        if lam < 0:
            a = -lam / (1 - lam)
            lower = (1 - a) * val + a * self.const
        else:
            lower = val
        # dual: S ~ -rho U projected to satisfy <S, E_T> = -c_T, then shifted PSD
        S = -self.U * self.rho
        S = (S + S.conj().transpose(1, 2)) / 2
        g = self.blocks_adjoint(S)
        shift = (-self.c - g) / self.norm
        shift[0] = 0.0
        S = (S + self.blocks(shift)) * self.mask
        lam_s = float(torch.linalg.eigvalsh((S + S.conj().transpose(1, 2)) / 2).real.min())
        if lam_s < 0:
            S = (S + (-lam_s) * torch.eye(self.nI, dtype=self.cdtype, device=self.device)) * self.mask
        upper = self.const + float(torch.einsum("aii->", S).real)
        return lower, upper


# ----------------------------------------------------------------------------
def _selftest():
    from sos_gpu import BooleanSoS
    from gap_search import brute_opt_boolean
    dev = "cpu"
    for L, gens in [(5, [1]), (5, [1, 2]), (7, [1, 2]), (9, [1, 2]), (9, [1]), (11, [1, 2])]:
        e = np.array(sorted(set((min(v, (v + s) % L), max(v, (v + s) % L)) for v in range(L) for s in gens)),
                     dtype=np.int64)
        e = np.array([x for x in e if x[0] != x[1]])
        m = len(e); signs = -np.ones(m, dtype=np.int64); w = np.ones(m)
        ref = BooleanSoS(L, e, degree=4, device=dev, dtype=torch.float64).set_instance(signs.astype(float), w)
        ref.solve(iters=6000, tol=1e-12)
        rlo, rhi = ref.certified_bounds()
        C = CirculantSoS(L, device=dev).set_instance(gens)
        v = C.solve(iters=6000, tol=1e-12)
        clo, chi = C.certified_bounds()
        opt, _ = brute_opt_boolean(e, signs, w, L)
        print(f"  L={L:3d} gens={gens}: opt={opt:.6f}  dense SoS4 in [{rlo:.6f},{rhi:.6f}]  "
              f"symmetric SoS4 in [{clo:.6f},{chi:.6f}]  (nI={C.nI}, vars={C.n_var}, {C.iters_done} its)")
        assert abs(clo - rlo) < 2e-4, (clo, rlo)
    print("circulant_sos selftest: OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--Ls", default="15,21,27,33,39,45")
    ap.add_argument("--gens", default="1,2")
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--out", default="../results/circulant_sym.jsonl")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        gens = [int(x) for x in a.gens.split(",")]
        for L in [int(x) for x in a.Ls.split(",")]:
            t = time.time()
            C = CirculantSoS(L, device="cuda" if torch.cuda.is_available() else "cpu").set_instance(gens)
            v = C.solve(iters=a.iters, tol=1e-11)
            lo, hi = C.certified_bounds()
            rec = {"L": L, "gens": gens, "nI": C.nI, "n_var": C.n_var, "sos4_lo": lo, "sos4_hi": hi,
                   "iters": C.iters_done, "build_s": C.build_time, "solve_s": time.time() - t}
            print(json.dumps({k: (round(vv, 8) if isinstance(vv, float) else vv) for k, vv in rec.items()}), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
