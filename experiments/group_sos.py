"""
group_sos.py -- Degree-4 sum-of-squares for Boolean 2Lin instances invariant under
a REGULAR ACTION OF AN ARBITRARY FINITE GROUP, block-diagonalised by that group's
irreducible representations.

This is the non-abelian generalisation of `circulant_sos.py`, which handled the
cyclic case Q = Z_L by a fast Fourier transform.  The whole point is that every
known (1-eps, delta) unique-games gap instance is abelian and its soundness is
certifiable by low-degree sum-of-squares through hypercontractivity of an abelian
noise operator, while Bafna-Minzer (CCC 2024) identify graphs *without* such a
structure as where hardness must live.  Non-abelian Cayley graphs are that region.

THE INSTANCE.  A group Q, a symmetric generating set S = S^{-1} (given as a list
of inverse-classes), signs b and weights w constant on each class.  Vertices are
the elements of Q, and for each generator s and vertex v there is a constraint
    x_v x_{vs} = b_s .
Left translation by Q is an automorphism, so the instance is Q-invariant and the
optimal degree-4 pseudo-moments may be taken Q-invariant.

THE BLOCK DIAGONALISATION.  Monomial basis {1} u {x_v} u {x_v x_w}; moments are
y_T = Etilde[x_T] with M[S,T] = y_{S xor T}.  Index the basis REDUNDANTLY by pairs
(i, g) with i an orbit representative (r_0 = {}, r_1 = {e}, r_{1+t} = {e, d_t} for
d_t one representative of each class {d, d^{-1}}) and g in Q, the pair standing for
the set g.r_i.  Redundancy is harmless: the redundant matrix is J M J^T with J of
full column rank, so it is positive semidefinite exactly when M is, and it removes
every stabiliser special case.  Then

    M[(i,g),(j,h)] = y[ orbit( r_i xor (g^{-1}h).r_j ) ] = Phi_{ij}(g^{-1}h)

is a block convolution operator, so for every irreducible representation rho,

    Mhat(rho)_{(i,a),(j,b)} = sum_{u in Q} Phi_{ij}(u) rho(u)_{ab}

and  M >= 0  <=>  Mhat(rho) >= 0 for every rho.  Blocks have size nI * dim(rho)
instead of one block of size nI * |Q|.  Verified against the dense spectrum in
_selftest, together with the cyclic solver on an abelian group.

The natural inner product weights each block by dim(rho) (Plancherel); with that
weighting the columns of the map y -> blocks(y) are orthogonal with squared norm
|Q| * (number of table hits), by the column orthogonality relation
sum_rho dim(rho) chi_rho(g) = |Q| [g = e].  That makes the affine projection a
scatter-average, exactly as in the cyclic case.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np
import torch


# ----------------------------------------------------------------------------
# Complex irreducible representations, computed numerically
# ----------------------------------------------------------------------------
def complex_irreps(G, seed: int = 0, tol: float = 1e-6, check: bool = True):
    """Unitary complex irreps of G as arrays (n, d, d).

    Method: the commutant of the left-regular representation is the algebra of
    right translations.  A generic Hermitian element of it has eigenspaces that
    are irreducible G-modules; reading the left action off an orthonormal basis
    of each eigenspace gives a unitary irrep.  Duplicates are removed by
    character, and sum_rho d_rho^2 = |G| is checked.
    """
    n = G.n
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    # left regular: (L_q v)[y] = v[q^{-1} y];  right: (R_h v)[y] = v[y h]
    c = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    A = np.zeros((n, n), dtype=complex)
    for h in range(n):
        # (R_h v)[y] = v[y h]  =>  row y picks column G.mul[y, h]
        A[idx, G.mul[idx, h]] += c[h]
    A = (A + A.conj().T) / 2
    w, V = np.linalg.eigh(A)
    # split into eigenspaces
    bounds, start = [], 0
    for k in range(1, n + 1):
        if k == n or abs(w[k] - w[k - 1]) > tol * max(1.0, abs(w[k - 1])):
            bounds.append((start, k)); start = k
    cand = []
    for s, e in bounds:
        Vb = V[:, s:e]
        d = e - s
        R = np.zeros((n, d, d), dtype=complex)
        for q in range(n):
            LqV = Vb[G.mul[G.inv[q], idx], :]        # row y of L_q V is V[q^{-1} y]
            R[q] = Vb.conj().T @ LqV
        cand.append((d, R))
    # deduplicate by character
    seen, irreps = [], []
    for d, R in cand:
        chi = np.einsum("qaa->q", R)
        if any(np.allclose(chi, c0, atol=1e-6) for c0 in seen):
            continue
        seen.append(chi); irreps.append((d, R))
    if check:
        tot = sum(d * d for d, _ in irreps)
        assert tot == n, f"sum of squared dimensions is {tot}, expected |G| = {n}"
        for d, R in irreps:                      # unitary and a homomorphism
            assert np.allclose(np.einsum("qab,qcb->qac", R, R.conj()), np.eye(d)[None, :, :], atol=1e-7)
            for _ in range(4):
                q, r = int(rng.integers(n)), int(rng.integers(n))
                assert np.allclose(R[q] @ R[r], R[G.mul[q, r]], atol=1e-7)
    return irreps


# ----------------------------------------------------------------------------
def subset_key(S, G):
    """Canonical form of a subset of G under left translation: the smallest
    sorted tuple over the translates s^{-1}.S for s in S."""
    if not S:
        return ()
    best = None
    for s in S:
        si = G.inv[s]
        t = tuple(sorted(int(G.mul[si, x]) for x in S))
        if best is None or t < best:
            best = t
    return best


class GroupSoS:
    def __init__(self, G, device="cuda", dtype=torch.float64, seed: int = 0, verbose: bool = False):
        self.G, self.device = G, device
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        n = G.n
        t0 = time.time()
        self.irreps = complex_irreps(G, seed=seed)
        self.dims = [d for d, _ in self.irreps]
        # orbit representatives of the monomial basis
        e = G.e
        seen = set()
        pair_reps = []
        for d in range(n):
            if d == e or d in seen:
                continue
            seen.add(d); seen.add(int(G.inv[d]))
            pair_reps.append(d)
        self.reps = [tuple()] + [(e,)] + [tuple(sorted((e, d))) for d in pair_reps]
        self.pair_reps = pair_reps
        self.nI = len(self.reps)
        # T[i, j, u] = orbit id of  r_i xor (u . r_j)
        orbs = {(): 0}
        T = np.zeros((self.nI, self.nI, n), dtype=np.int64)
        for i, ri in enumerate(self.reps):
            Si = set(int(x) for x in ri)
            for j, rj in enumerate(self.reps):
                for u in range(n):
                    Sj = set(int(G.mul[u, x]) for x in rj)
                    key = subset_key(sorted(Si ^ Sj), G)
                    if key not in orbs:
                        orbs[key] = len(orbs)
                    T[i, j, u] = orbs[key]
        self.orbs = orbs
        self.n_var = len(orbs)
        self.T = torch.tensor(T, device=device)
        counts = torch.zeros(self.n_var, device=device, dtype=dtype)
        counts.index_add_(0, self.T.reshape(-1), torch.ones(self.T.numel(), device=device, dtype=dtype))
        self.counts = counts
        self.norm = (n * counts).clamp_min(1.0)
        # irrep tensors on the device
        self.R = [torch.tensor(R, device=device, dtype=self.cdtype) for _, R in self.irreps]
        self.dw = [float(d) for d in self.dims]
        # the orbit of a single generator class, for the objective
        self.pair_var = torch.tensor(
            [orbs[subset_key(sorted({e, int(d)}), G)] for d in pair_reps], device=device)
        self.build_time = time.time() - t0
        if verbose:
            print(f"  |G|={n}  irreps {self.dims}  nI={self.nI}  vars={self.n_var}  "
                  f"blocks {[self.nI*d for d in self.dims]}  [{self.build_time:.1f}s]", flush=True)

    # -- instance -----------------------------------------------------------
    def set_instance(self, gens, signs=None, weights=None):
        """gens: group elements, one per generator class (its inverse is implied).
        signs: b_s in {+-1} (default -1, i.e. Max-Cut).  weights: per class."""
        G = self.G
        n = G.n
        gens = [int(g) for g in gens]
        pos = {int(d): t for t, d in enumerate(self.pair_reps)}
        cls, mult = [], []
        for g in gens:
            key = g if g in pos else int(G.inv[g])
            assert key in pos, "generator must not be the identity"
            cls.append(pos[key])
            mult.append(n if int(G.mul[g, g]) != G.e else n // 2)   # edges in this class
        b = np.array([-1.0] * len(gens)) if signs is None else np.asarray(signs, float)
        w = np.ones(len(gens)) if weights is None else np.asarray(weights, float)
        om = w * np.array(mult, float)
        om = om / om.sum()
        c = torch.zeros(self.n_var, device=self.device, dtype=self.dtype)
        c.index_add_(0, self.pair_var[torch.tensor(cls, device=self.device)],
                     torch.tensor(0.5 * om * b, device=self.device, dtype=self.dtype))
        self.c, self.const = c, 0.5
        self.gens, self.b, self.w, self.om = gens, b, w, om
        return self

    def edges(self):
        """Explicit edge list and signs, for the exact solvers."""
        G = self.G
        E, B, W = [], [], []
        seen = set()
        for t, g in enumerate(self.gens):
            for v in range(G.n):
                u = int(G.mul[v, g])
                if u == v:
                    continue
                k = (min(u, v), max(u, v))
                if k in seen:
                    continue
                seen.add(k); E.append(k); B.append(int(self.b[t])); W.append(float(self.w[t]))
        return np.array(E, dtype=np.int64), np.array(B, dtype=np.int64), np.array(W)

    # -- operators ----------------------------------------------------------
    def blocks(self, y):
        Y = y[self.T].to(self.cdtype)                                  # (nI, nI, n)
        out = []
        for d, R in zip(self.dims, self.R):
            B = torch.einsum("iju,uab->iajb", Y, R).reshape(self.nI * d, self.nI * d)
            out.append(B)
        return out

    def blocks_adjoint(self, Bs):
        g = torch.zeros(self.nI, self.nI, self.G.n, device=self.device, dtype=self.dtype)
        for d, R, B, dw in zip(self.dims, self.R, Bs, self.dw):
            X = B.reshape(self.nI, d, self.nI, d)
            g = g + dw * torch.einsum("uab,iajb->iju", R.conj(), X).real.to(self.dtype)
        out = torch.zeros(self.n_var, device=self.device, dtype=self.dtype)
        out.index_add_(0, self.T.reshape(-1), g.reshape(-1))
        return out

    def proj_L(self, Bs):
        y = self.blocks_adjoint(Bs) / self.norm
        y[0] = 1.0
        return self.blocks(y), y

    @staticmethod
    def proj_psd(Bs):
        out = []
        for B in Bs:
            B = (B + B.conj().T) / 2
            w, U = torch.linalg.eigh(B)
            out.append((U * torch.clamp(w.real, min=0).to(B.dtype)[None, :]) @ U.conj().T)
        return out

    # -- ADMM ---------------------------------------------------------------
    def solve(self, iters=4000, rho=None, tol=1e-10, verbose=False, log_every=500, alpha=1.7):
        Ctil = self.blocks(self.c / self.norm)
        rho = rho or max(1e-3, max(float(C.abs().max()) for C in Ctil) * 10)
        Z = [torch.eye(self.nI * d, dtype=self.cdtype, device=self.device) for d in self.dims]
        U = [torch.zeros_like(z) for z in Z]
        y = None
        for it in range(iters):
            M, y = self.proj_L([z - u + C / rho for z, u, C in zip(Z, U, Ctil)])
            Mh = [alpha * m + (1 - alpha) * z for m, z in zip(M, Z)]
            Znew = self.proj_psd([m + u for m, u in zip(Mh, U)])
            U = [u + m - zn for u, m, zn in zip(U, Mh, Znew)]
            r = sum(float((m - zn).norm()) for m, zn in zip(M, Znew))
            s = sum(float(rho * (zn - z).norm()) for zn, z in zip(Znew, Z))
            Z = Znew
            if it % 40 == 0 and it > 0:
                if r > 10 * s:
                    rho *= 2; U = [u / 2 for u in U]
                elif s > 10 * r:
                    rho /= 2; U = [u * 2 for u in U]
            if verbose and it % log_every == 0:
                print(f"    it {it:6d} value {self.const + float(self.c @ y):.9f} r {r:.2e} s {s:.2e}", flush=True)
            if r < tol and s < tol and it > 30:
                break
        self.Z, self.U, self.y, self.rho, self.iters_done = Z, U, y, rho, it + 1
        return self.const + float(self.c @ y)

    def certified_bounds(self):
        """(lower, upper) on the degree-4 value.
        lower: any feasible pseudo-moment (the ADMM primal, mixed with the trivial one if needed).
        upper: a feasible dual point.  Dual: max <blocks(e_0), S> over S >= 0 with
        <blocks(e_T), S> = -c_T for T != 0.  S is taken from the ADMM dual iterate, corrected
        per class, then made PSD by adding a multiple of D = blocks(e_0), which changes only the
        T = 0 pairing.  Because the empty orbit is indexed redundantly, blocks(e_0) is NOT the
        identity: in every nontrivial block its (i=0) rows vanish, and so do those rows of
        blocks(e_T) for every T -- so the dual may be supported off them, and D >= I on the rest."""
        y = self.y.clone(); y[0] = 1.0
        Bs = self.blocks(y)
        lam = min(float(torch.linalg.eigvalsh((B + B.conj().T) / 2).real.min()) for B in Bs)
        val = self.const + float(self.c @ y)
        lower = val if lam >= 0 else (1 - (-lam / (1 - lam))) * val + (-lam / (1 - lam)) * self.const
        e0 = torch.zeros(self.n_var, device=self.device, dtype=self.dtype); e0[0] = 1.0
        D = self.blocks(e0)
        S = [(-u * self.rho) for u in self.U]
        S = [(s + s.conj().T) / 2 for s in S]
        g = self.blocks_adjoint(S)
        shift = (-self.c - g) / self.norm
        shift[0] = 0.0
        S = [s + t for s, t in zip(S, self.blocks(shift))]
        # zero the structurally-zero rows/cols (i = 0 in nontrivial blocks): they pair to 0
        # with every blocks(e_T), so this changes no dual constraint and no dual value.
        Sm = []
        for d, s, Dr in zip(self.dims, S, D):
            keep = (Dr.diagonal().real.abs() > 1e-12)
            m = keep.to(s.dtype)
            Sm.append(s * m[:, None] * m[None, :])
        S = Sm
        lam_s = min(float(torch.linalg.eigvalsh((s + s.conj().T) / 2).real.min()) for s in S)
        if lam_s < 0:
            S = [s + (-lam_s) * Dr for s, Dr in zip(S, D)]
        upper = self.const + float(self.c[0]) + float(self.blocks_adjoint(S)[0])
        return lower, upper


# ----------------------------------------------------------------------------
# Batched solver: many instances of the SAME group at once (identical block structure,
# different objectives).  y has shape (B, n_var); blocks are (B, nI*d, nI*d) per irrep.
# ----------------------------------------------------------------------------
class GroupSoSBatch:
    def __init__(self, base: "GroupSoS"):
        self.P = base
        self.G, self.device, self.dtype, self.cdtype = base.G, base.device, base.dtype, base.cdtype
        self.nI, self.dims, self.R, self.dw, self.T, self.norm, self.n_var = (
            base.nI, base.dims, base.R, base.dw, base.T, base.norm, base.n_var)

    def set_instances(self, gens_list, signs_list, weights_list=None):
        B = len(gens_list)
        c = torch.zeros(B, self.n_var, device=self.device, dtype=self.dtype)
        for b in range(B):
            self.P.set_instance(gens_list[b], signs=signs_list[b],
                                weights=None if weights_list is None else weights_list[b])
            c[b] = self.P.c
        self.c, self.const, self.B = c, 0.5, B
        return self

    def blocks(self, y):                         # y (B, n_var)
        Y = y[:, self.T].to(self.cdtype)         # (B, nI, nI, n)
        return [torch.einsum("ziju,uab->ziajb", Y, R).reshape(self.B, self.nI * d, self.nI * d)
                for d, R in zip(self.dims, self.R)]

    def blocks_adjoint(self, Bs):
        g = torch.zeros(self.B, self.nI, self.nI, self.G.n, device=self.device, dtype=self.dtype)
        for d, R, X, dw in zip(self.dims, self.R, Bs, self.dw):
            X4 = X.reshape(self.B, self.nI, d, self.nI, d)
            g = g + dw * torch.einsum("uab,ziajb->ziju", R.conj(), X4).real.to(self.dtype)
        out = torch.zeros(self.B, self.n_var, device=self.device, dtype=self.dtype)
        out.index_add_(1, self.T.reshape(-1), g.reshape(self.B, -1))
        return out

    def proj_L(self, Bs):
        y = self.blocks_adjoint(Bs) / self.norm[None, :]
        y[:, 0] = 1.0
        return self.blocks(y), y

    @staticmethod
    def proj_psd(Bs):
        out = []
        for X in Bs:
            X = (X + X.conj().transpose(1, 2)) / 2
            w, U = torch.linalg.eigh(X)
            out.append((U * torch.clamp(w.real, min=0).to(X.dtype)[:, None, :]) @ U.conj().transpose(1, 2))
        return out

    def solve(self, iters=3000, tol=1e-9, alpha=1.7, verbose=False, log_every=500):
        Bn = self.B
        Ctil = self.blocks(self.c / self.norm[None, :])
        rho = torch.stack([C.abs().amax(dim=(1, 2)) for C in Ctil]).amax(dim=0) * 10
        rho = rho.clamp_min(1e-3).to(self.dtype)                       # (B,)
        Z = [torch.eye(self.nI * d, dtype=self.cdtype, device=self.device).expand(Bn, -1, -1).clone()
             for d in self.dims]
        U = [torch.zeros_like(z) for z in Z]
        r_ = lambda x: x.view(Bn, 1, 1).to(self.cdtype)
        active = torch.ones(Bn, dtype=torch.bool, device=self.device)
        for it in range(iters):
            M, y = self.proj_L([z - u + C / r_(rho) for z, u, C in zip(Z, U, Ctil)])
            Mh = [alpha * m + (1 - alpha) * z for m, z in zip(M, Z)]
            Znew = self.proj_psd([m + u for m, u in zip(Mh, U)])
            U = [u + m - zn for u, m, zn in zip(U, Mh, Znew)]
            r = sum(((m - zn).abs() ** 2).sum(dim=(1, 2)) for m, zn in zip(M, Znew)).sqrt()
            s = rho * sum(((zn - z).abs() ** 2).sum(dim=(1, 2)) for zn, z in zip(Znew, Z)).sqrt()
            Z = Znew
            if it % 40 == 0 and it > 0:
                up = (r > 10 * s); dn = (s > 10 * r)
                rho = torch.where(up, rho * 2, torch.where(dn, rho / 2, rho))
                fac = torch.where(up, 0.5, torch.where(dn, 2.0, 1.0)).to(self.cdtype).view(Bn, 1, 1)
                U = [u * fac for u in U]
            if verbose and it % log_every == 0:
                print(f"    it {it:5d}  max r {float(r.max()):.2e}  max s {float(s.max()):.2e}", flush=True)
            if it > 30 and float(r.max()) < tol and float(s.max()) < tol:
                break
        self.Z, self.U, self.y, self.rho, self.iters_done = Z, U, y, rho, it + 1
        return self.const + (self.c * y).sum(dim=1)

    def certified_bounds(self):
        Bn = self.B
        y = self.y.clone(); y[:, 0] = 1.0
        Bs = self.blocks(y)
        lam = torch.stack([torch.linalg.eigvalsh((X + X.conj().transpose(1, 2)) / 2).real.min(dim=1).values
                           for X in Bs]).min(dim=0).values                        # (B,)
        val = self.const + (self.c * y).sum(dim=1)
        a = torch.where(lam < 0, -lam / (1 - lam), torch.zeros_like(lam))
        lower = (1 - a) * val + a * self.const
        e0 = torch.zeros(Bn, self.n_var, device=self.device, dtype=self.dtype); e0[:, 0] = 1.0
        D = self.blocks(e0)
        S = [(-u * self.rho.to(self.cdtype).view(Bn, 1, 1)) for u in self.U]
        S = [(x + x.conj().transpose(1, 2)) / 2 for x in S]
        g = self.blocks_adjoint(S)
        shift = (-self.c - g) / self.norm[None, :]
        shift[:, 0] = 0.0
        S = [x + t for x, t in zip(S, self.blocks(shift))]
        Sm = []
        for x, Dr in zip(S, D):
            keep = (Dr[0].diagonal().real.abs() > 1e-12).to(x.dtype)
            Sm.append(x * keep[None, :, None] * keep[None, None, :])
        S = Sm
        lam_s = torch.stack([torch.linalg.eigvalsh((x + x.conj().transpose(1, 2)) / 2).real.min(dim=1).values
                             for x in S]).min(dim=0).values
        mu = torch.clamp(-lam_s, min=0).to(self.cdtype).view(Bn, 1, 1)
        S = [x + mu * Dr for x, Dr in zip(S, D)]
        upper = self.const + self.c[:, 0] + self.blocks_adjoint(S)[:, 0]
        return lower, upper


# ----------------------------------------------------------------------------
def _selftest():
    from group_ug import (cyclic, elementary_abelian, direct_product, semidirect,
                          group_from_permutations, heisenberg, extraspecial_2)
    from sos_gpu import BooleanSoS
    from gap_search import brute_opt_boolean
    dev = "cpu"

    print("  -- irreps --")
    for name, G in [("Z9", cyclic(9)), ("S3", group_from_permutations("S3", [[1, 0, 2], [1, 2, 0]], 3)),
                    ("D4", extraspecial_2(1)), ("A4", group_from_permutations("A4", [[1, 2, 0, 3], [0, 2, 3, 1]], 4)),
                    ("Heis3", heisenberg(3)),
                    ("S4", group_from_permutations("S4", [[1, 0, 2, 3], [1, 2, 3, 0]], 4))]:
        irr = complex_irreps(G)
        print(f"     {name:6s} |G|={G.n:3d}  dims {sorted(d for d, _ in irr)}  "
              f"sum d^2 = {sum(d*d for d, _ in irr)}")

    print("  -- blocks reproduce the dense spectrum --")
    for name, G in [("Z9", cyclic(9)), ("S3", group_from_permutations("S3", [[1, 0, 2], [1, 2, 0]], 3)),
                    ("D4", extraspecial_2(1))]:
        P = GroupSoS(G, device=dev)
        rng = np.random.default_rng(0)
        yv = rng.standard_normal(P.n_var); yv[0] = 1.0
        y = torch.tensor(yv, dtype=torch.float64)
        Bs = P.blocks(y)
        ev_blocks = np.sort(np.concatenate(
            [np.repeat(np.linalg.eigvalsh(B.numpy()), d) for B, d in zip(Bs, P.dims)]))
        # dense redundant matrix
        n = G.n
        D = P.nI * n
        M = np.zeros((D, D))
        Tn = P.T.numpy()
        for i in range(P.nI):
            for g in range(n):
                for j in range(P.nI):
                    for h in range(n):
                        u = int(G.mul[G.inv[g], h])
                        M[i * n + g, j * n + h] = yv[Tn[i, j, u]]
        ev_dense = np.sort(np.linalg.eigvalsh(M))
        ok = np.allclose(ev_dense, ev_blocks, atol=1e-8)
        print(f"     {name:4s} dense {D}x{D} vs blocks {[P.nI*d for d in P.dims]}: spectra match = {ok}")
        assert ok
        # adjointness under the Plancherel-weighted inner product
        Xs = [torch.tensor(rng.standard_normal(B.shape) + 1j * rng.standard_normal(B.shape)) for B in Bs]
        Xs = [(X + X.conj().T) / 2 for X in Xs]
        lhs = sum(dw * float((B.conj() * X).real.sum()) for dw, B, X in zip(P.dw, Bs, Xs))
        rhs = float((y * P.blocks_adjoint(Xs)).sum())
        assert abs(lhs - rhs) < 1e-8 * max(1.0, abs(lhs)), (lhs, rhs)

    print("  -- values agree with the dense degree-4 solver --")
    cases = [("Z9", cyclic(9), [1, 2], None),
             ("S3", group_from_permutations("S3", [[1, 0, 2], [1, 2, 0]], 3), [1, 2], None),
             ("D4", extraspecial_2(1), [1, 2, 3], None),
             ("A4", group_from_permutations("A4", [[1, 2, 0, 3], [0, 2, 3, 1]], 4), [1, 2, 3], None)]
    for name, G, gens, signs in cases:
        P = GroupSoS(G, device=dev).set_instance(gens, signs=signs)
        E, B, W = P.edges()
        if len(E) < 3:
            print(f"     {name}: too few edges, skipped"); continue
        v = P.solve(iters=20000, tol=1e-12)
        lo, hi = P.certified_bounds()
        ref = BooleanSoS(G.n, E, degree=4, device=dev, dtype=torch.float64).set_instance(B.astype(float), W)
        ref.solve(iters=20000, tol=1e-12)
        rlo, rhi = ref.certified_bounds()
        opt, _ = brute_opt_boolean(E, B, W, G.n)
        print(f"     {name:4s} n={G.n:2d} m={len(E):3d}  opt={opt:.6f}  "
              f"dense [{rlo:.6f},{rhi:.6f}]  group [{lo:.6f},{hi:.6f}]  "
              f"blocks {[P.nI*d for d in P.dims]}")
        assert abs(lo - rlo) < 2e-4, (name, lo, rlo)
    print("group_sos selftest: OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
