"""
kv_lasserre2_sym.py -- Level-2 Lasserre (degree-4 SoS) for the Khot-Vishnoi game
U_{3,eta}, block-diagonalised under the translation group F_2^8 of the noisy cube.

Language.  A labelling is a transversal T of the 32 cosets (classes) of the
character subgroup C = {chi_S} < F_2^8.  Pseudo-moments y[A] = Etilde[ prod_{f in A}
1[f in T] ] for sets A of <= 4 functions from distinct classes.
  * y[{}] = 1;  marginalisation  sum_{f in P} y[A + f] = y[A]  (|A| <= 3, class P not hit by A);
  * 0 <= y <= 1;
  * moment matrix M on I = {{}} u {f} u {{f,g} distinct classes}:  M[A,B] = y[A u B] (0 on class clash).
  * value = (1/W) sum_{f<g, distinct classes, d(f,g) in window} wt'(d) y[{f,g}].
Symmetry.  Translations h: A -> A.h preserve everything, so y is constant on
orbits and M commutes with G = F_2^8.  With orbit reps r_i (stabiliser H_i) and
characters alpha,  Mhat(alpha)_{ij} = |H_i H_j|^{-1/2} sum_{h in G} alpha(h) M[r_i, h r_j],
and M >= 0  <=>  Mhat(alpha) >= 0 for all alpha (rows with alpha|H_i != 1 vanish
automatically).  Over all alpha this is the Walsh-Hadamard transform along h of the
gathered tensor Y[i,j,h] = y[orb(r_i u h r_j)].
Solver.  PDHG (Chambolle-Pock) on the GPU:  min -c.y  s.t. 0<=y<=1, A y = b, Phi(y) in PSD.
Certificate.  For any lambda and any S >= 0:  SoS4 <= b.lambda + sum_j max(0, -r_j),
r = -c + A^T lambda - Phi^T S  (uses 0 <= y <= 1).  Rigorous upper bound on the
degree-4 value, hence on opt.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np
import torch
from kv_instance import class_structure, characters


def build(k: int, eta: float, d_window=None):
    N = 1 << k
    n_f = 1 << N
    cls, S_of, reps, F = class_structure(k)          # cls[f] = class id
    n_cls = n_f // N
    d_of = np.array([bin(x).count("1") for x in range(n_f)])
    if d_window is None:
        dmin = max(1, int(math.ceil(eta * N / 2))); dmax = min(N, int(math.floor(2 * eta * N)))
    else:
        dmin, dmax = d_window
    W = sum(math.comb(N, d) * eta ** d * (1 - eta) ** (N - d) for d in range(dmin, dmax + 1))
    wt = np.where((d_of >= dmin) & (d_of <= dmax), 2.0 * 2.0 ** -N * eta ** d_of * (1 - eta) ** (N - d_of), 0.0)
    return dict(N=N, n_f=n_f, cls=cls, n_cls=n_cls, d_of=d_of, W=W, wt=wt, dmin=dmin, dmax=dmax)


def canon_key(sets: np.ndarray) -> np.ndarray:
    """sets: (m, 4) int array of function ids, padded with -1 (sorted, -1 last).
    Canonical form under translation: min over a in A of sorted(A xor a), packed into int64."""
    m = sets.shape[0]
    best = None
    for col in range(4):
        a = sets[:, col]
        valid = a >= 0
        sh = np.where(sets >= 0, sets ^ a[:, None], 999)          # translate by a; pad -> 999 (sorts last)
        sh = np.sort(sh, axis=1)
        sh = np.where(sh == 999, 255 + 1, sh)                       # pad code 256
        key = (sh[:, 0].astype(np.int64) | (sh[:, 1].astype(np.int64) << 9) | (sh[:, 2].astype(np.int64) << 18)
               | (sh[:, 3].astype(np.int64) << 27))
        key = np.where(valid, key, np.iinfo(np.int64).max)
        best = key if best is None else np.minimum(best, key)
    return best


class KVLasserre2:
    def __init__(self, k: int = 3, eta: float = 0.2, device: str = "cuda"):
        assert k == 3, "sizes are hard-wired for k=3 (N=8, 256 functions, 32 classes)"
        self.device = device
        B = build(k, eta); self.B = B
        N, n_f, cls, n_cls = B["N"], B["n_f"], B["cls"], B["n_cls"]
        t0 = time.time()
        # ---- enumerate orbit representatives of sets containing function 0 ----
        others = np.arange(1, n_f)
        sets = [np.array([[0, -1, -1, -1]])]
        # size 2: {0, p}, class(p) != class(0)
        p = others[cls[others] != cls[0]]
        sets.append(np.stack([np.zeros_like(p), p, -np.ones_like(p), -np.ones_like(p)], 1))
        # size 3 and 4: distinct classes, all different from class 0
        import itertools
        by_cls = [np.where(cls == c)[0] for c in range(n_cls)]
        c0 = cls[0]
        oc = [c for c in range(n_cls) if c != c0]
        s3, s4 = [], []
        for c1, c2 in itertools.combinations(oc, 2):
            f1, f2 = np.meshgrid(by_cls[c1], by_cls[c2], indexing="ij")
            f1, f2 = f1.ravel(), f2.ravel()
            s3.append(np.stack([np.zeros_like(f1), f1, f2, -np.ones_like(f1)], 1))
        for c1, c2, c3 in itertools.combinations(oc, 3):
            f1, f2, f3 = np.meshgrid(by_cls[c1], by_cls[c2], by_cls[c3], indexing="ij")
            f1, f2, f3 = f1.ravel(), f2.ravel(), f3.ravel()
            s4.append(np.stack([np.zeros_like(f1), f1, f2, f3], 1))
        sets.append(np.concatenate(s3)); sets.append(np.concatenate(s4))
        allsets = np.concatenate(sets)
        allsets = np.sort(np.where(allsets < 0, 999, allsets), axis=1); allsets = np.where(allsets == 999, -1, allsets)
        keys = canon_key(allsets)
        self.orbit_keys, inv = np.unique(keys, return_inverse=True)      # orbit id = position in sorted keys
        self.n_var = len(self.orbit_keys)
        sizes = (allsets >= 0).sum(1)
        self.var_size = np.zeros(self.n_var, dtype=np.int64); self.var_size[inv] = sizes
        # representative set per orbit (first occurrence)
        self.var_rep = np.zeros((self.n_var, 4), dtype=np.int64); self.var_rep[inv] = allsets
        self.idx_empty = int(np.searchsorted(self.orbit_keys, canon_key(np.array([[0, -1, -1, -1]]))[0]))
        # NOTE: the empty set is not enumerated above; add it explicitly
        self.orbit_keys = np.concatenate([[np.int64(-1)], self.orbit_keys]); self.n_var += 1
        self.var_size = np.concatenate([[0], self.var_size]); self.var_rep = np.concatenate([[[-1, -1, -1, -1]], self.var_rep])
        self.idx_empty = 0
        print(f"  variables (orbits of sets): {self.n_var}  by size: " +
              str({s: int((self.var_size == s).sum()) for s in range(5)}) + f"  [{time.time()-t0:.1f}s]", flush=True)

        def lookup(sets4):
            """orbit ids of sets (m,4) (padded -1, may be unsorted); -1 if a class clash."""
            s = np.where(sets4 < 0, 999, sets4); s = np.sort(s, axis=1)
            # remove duplicates within a row (same function twice)
            for c in range(1, 4):
                dup = (s[:, c] == s[:, c - 1]) & (s[:, c] != 999)
                s[dup, c] = 999
            s = np.sort(s, axis=1)
            # class clash: two distinct functions in the same class
            cl = np.where(s < 999, cls[np.minimum(s, n_f - 1)], -1 - np.arange(4)[None, :] * 0)
            cl = np.where(s < 999, cls[np.minimum(s, n_f - 1)], -1)
            clash = np.zeros(len(s), dtype=bool)
            for a in range(4):
                for b in range(a + 1, 4):
                    clash |= (cl[:, a] >= 0) & (cl[:, a] == cl[:, b])
            s = np.where(s == 999, -1, s)
            empty = (s < 0).all(axis=1)
            key = canon_key(s)
            key[empty] = -1
            pos = np.searchsorted(self.orbit_keys, key)
            pos = np.clip(pos, 0, self.n_var - 1)
            ok = self.orbit_keys[pos] == key
            out = np.where(ok & ~clash, pos, -1)
            return out
        self.lookup = lookup

        # ---- moment index orbits: {} , {0}, {0,p} for p not in class 0, p != 0 ----
        pair_p = others[cls[others] != cls[0]]                      # 248 values
        I_reps = [np.array([-1, -1, -1, -1]), np.array([0, -1, -1, -1])] + [np.array([0, int(q), -1, -1]) for q in pair_p]
        self.nI = len(I_reps)
        H = np.array([n_f, 1] + [2] * len(pair_p), dtype=np.float64)   # stabiliser orders
        self.scale = torch.tensor(1.0 / np.sqrt(np.outer(H, H)), device=device, dtype=torch.float32)
        # table T[i, j, h] = orbit id of r_i u (h . r_j)
        hs = np.arange(n_f)
        T = np.full((self.nI, self.nI, n_f), -1, dtype=np.int64)
        for j, rj in enumerate(I_reps):
            rj_h = np.where(rj[None, :] >= 0, rj[None, :] ^ hs[:, None], -1)      # (n_f, 4)
            for i, ri in enumerate(I_reps):
                un = np.concatenate([np.repeat(ri[None, :], n_f, 0), rj_h], axis=1)   # (n_f, 8)
                # compress to 4 columns: keep distinct non-negative entries (at most 4 distinct classes else clash)
                un_s = np.sort(np.where(un < 0, 999, un), axis=1)
                # dedupe
                for c in range(1, 8):
                    d = (un_s[:, c] == un_s[:, c - 1]) & (un_s[:, c] != 999); un_s[d, c] = 999
                un_s = np.sort(un_s, axis=1)
                too_many = (un_s[:, 4] != 999)
                s4 = np.where(un_s[:, :4] == 999, -1, un_s[:, :4])
                ids = lookup(s4)
                ids[too_many] = -1
                T[i, j] = ids
        self.T = torch.tensor(T, device=device)
        print(f"  moment blocks: 256 x ({self.nI} x {self.nI});  table nnz {(T >= 0).sum()}  [{time.time()-t0:.1f}s]", flush=True)

        # ---- marginalisation constraints ----
        rows, cols, vals = [], [], []
        r = 0
        small = np.where(self.var_size <= 3)[0]
        for v in small:
            A = self.var_rep[v]
            hit = set(int(cls[f]) for f in A if f >= 0)
            for c in range(n_cls):
                if c in hit:
                    continue
                ext = np.stack([np.concatenate([A[A >= 0], [f], -np.ones(3 - (A >= 0).sum(), dtype=np.int64)]) for f in by_cls[c]])
                ext = np.where(ext < 0, -1, ext)
                ids = lookup(ext)
                assert (ids >= 0).all()
                for q in ids:
                    rows.append(r); cols.append(int(q)); vals.append(1.0)
                rows.append(r); cols.append(int(v)); vals.append(-1.0)
                r += 1
        self.nA = r
        idx = torch.tensor(np.array([rows, cols]), device=device)
        self.A = torch.sparse_coo_tensor(idx, torch.tensor(vals, device=device, dtype=torch.float32), (r, self.n_var)).coalesce()
        self.AT = self.A.t().coalesce()
        self.b = torch.zeros(r, device=device)
        print(f"  marginalisation rows: {r}  [{time.time()-t0:.1f}s]", flush=True)
        # ---- objective on pair orbits: 128 pairs per orbit ----
        c = np.zeros(self.n_var)
        for q in pair_p:
            v = lookup(np.array([[0, int(q), -1, -1]]))[0]
            c[v] += N * 128 * B["wt"][q] / B["W"]   # N per bundle (all N pairs of a bundle share the product)
        self.c = torch.tensor(c, device=device, dtype=torch.float32)
        self.obj_scale = float(c.sum())   # value if all pair moments were 1 (= 1 iff no self-loop bundles)

    # ---- operators --------------------------------------------------------
    def wht(self, X):
        # Walsh-Hadamard along last axis (length 256) unnormalised
        n = X.shape[-1]; h = 1
        X = X.clone()
        while h < n:
            X = X.reshape(*X.shape[:-1], n // (2 * h), 2, h)
            a, b = X[..., 0, :], X[..., 1, :]
            X = torch.stack([a + b, a - b], dim=-2).reshape(*X.shape[:-3], n)
            h *= 2
        return X

    def Phi(self, y):
        """y (n_var,) -> blocks (256, nI, nI)"""
        Y = torch.where(self.T >= 0, y[self.T.clamp_min(0)], torch.zeros((), device=self.device))
        Mh = self.wht(Y)                                     # (nI, nI, 256)
        return (Mh * self.scale[:, :, None]).permute(2, 0, 1).contiguous()

    def PhiT(self, Sb):
        """adjoint: blocks (256, nI, nI) -> (n_var,)"""
        G = (Sb.permute(1, 2, 0) * self.scale[:, :, None]).contiguous()
        Yg = self.wht(G)                                     # WHT is self-adjoint
        out = torch.zeros(self.n_var, device=self.device)
        mask = self.T >= 0
        out.index_add_(0, self.T[mask], Yg[mask])
        return out

    @staticmethod
    def proj_psd(Sb):
        Sb = (Sb + Sb.transpose(1, 2)) / 2
        w, U = torch.linalg.eigh(Sb)
        return (U * w.clamp_min(0)[:, None, :]) @ U.transpose(1, 2)

    def op_norm(self, iters=30):
        y = torch.randn(self.n_var, device=self.device)
        for _ in range(iters):
            z = self.PhiT(self.Phi(y)) + torch.sparse.mm(self.AT, torch.sparse.mm(self.A, y[:, None]))[:, 0]
            nrm = z.norm(); y = z / nrm
        return float(nrm.sqrt())

    # ---- warm start: exact moments of the translation-symmetrised subcube labelling ----
    def subcube_moments(self):
        """y[A] = fraction of translates S.h of the subcube transversal S that contain A."""
        N, n_f, cls = self.B["N"], self.B["n_f"], self.B["cls"]
        k = int(math.log2(N))
        pts = [1 << i for i in range(k)]                      # points with a single 1-bit
        # S = functions with value +1 (bit 0) at all points p_i  ->  bits at positions pts are 0
        mask = sum(1 << p for p in pts)
        S = np.array([f for f in range(n_f) if (f & mask) == 0])
        assert len(S) == n_f // N and len(set(cls[S].tolist())) == n_f // N
        inS = np.zeros(n_f, dtype=bool); inS[S] = True
        reps = self.var_rep
        y = np.zeros(self.n_var)
        hs = np.arange(n_f)
        for c in range(0, self.n_var, 20000):
            R = reps[c:c + 20000]                               # (m, 4)
            ok = np.ones((len(R), n_f), dtype=bool)
            for col in range(4):
                f = R[:, col]
                sh = np.where(f[:, None] >= 0, f[:, None] ^ hs[None, :], 0)
                ok &= np.where(f[:, None] >= 0, inS[sh], True)
            y[c:c + 20000] = ok.sum(1) / n_f
        y[self.idx_empty] = 1.0
        return torch.tensor(y, device=self.device, dtype=torch.float32)

    # ---- PDHG with diagonal preconditioning (Pock-Chambolle) ---------------
    def solve(self, iters=4000, verbose=True, log_every=200, warm=True):
        dev = self.device
        ones = torch.ones(self.n_var, device=dev)
        colA = torch.sparse.mm(self.AT, torch.sparse.mm(self.A.abs(), ones[:, None]) * 0 + 1.0)[:, 0] if False else                torch.sparse.mm(self.AT.abs() if hasattr(self.AT, "abs") else self.AT, torch.ones(self.nA, 1, device=dev))[:, 0]
        rowA = torch.sparse.mm(self.A.abs() if hasattr(self.A, "abs") else self.A, ones[:, None])[:, 0]
        # |Phi| column sums: number of (i,j,h) table hits times scale, via PhiT on all-ones blocks (WHT of ones = 256 delta)
        nnz_ij = (self.T >= 0).sum(-1).to(torch.float32)          # (nI, nI)
        rowPhi = (nnz_ij * self.scale)                              # per block entry (same for all alpha)
        colPhi = torch.zeros(self.n_var, device=dev)
        mask = self.T >= 0
        colPhi.index_add_(0, self.T[mask], (self.scale[:, :, None].expand_as(self.T))[mask])
        colPhi = colPhi * 256.0                                     # summed over the 256 characters (|alpha(h)| = 1)
        tau = 1.0 / (colA + colPhi + 1e-6)
        sigA = 1.0 / (rowA + 1e-6)
        sigPhi = (1.0 / (rowPhi + 1e-6))[None, :, :]
        y = self.subcube_moments() if warm else torch.zeros(self.n_var, device=dev)
        y[self.idx_empty] = 1.0
        lam = torch.zeros(self.nA, device=dev)
        S = torch.zeros(256, self.nI, self.nI, device=dev)
        c = self.c
        for it in range(iters):
            g = -c + torch.sparse.mm(self.AT, lam[:, None])[:, 0] - self.PhiT(S)
            y_new = (y - tau * g).clamp(0.0, 1.0); y_new[self.idx_empty] = 1.0
            ybar = 2 * y_new - y
            lam = lam + sigA * (torch.sparse.mm(self.A, ybar[:, None])[:, 0] - self.b)
            S = self.proj_psd(S - sigPhi * self.Phi(ybar))
            y = y_new
            if verbose and it % log_every == 0:
                val = float(c @ y); res = float(torch.sparse.mm(self.A, y[:, None]).norm())
                lam_min = float(torch.linalg.eigvalsh(self.Phi(y)).min())
                self.y, self.lam, self.S = y, lam, S
                print(f"    it {it:5d}  value {val:.6f}  |Ay-b| {res:.2e}  lambda_min(M) {lam_min:.2e}  dual_bound {self.dual_bound():.6f}", flush=True)
        self.y, self.lam, self.S = y, lam, S
        return float(c @ y)

    def dual_bound(self):
        """Rigorous upper bound on the degree-4 value from (lam, S >= 0):
        SoS4 <= b.lam + sum_j max(0, -r_j),  r = -c + A^T lam - Phi^T S."""
        S = self.proj_psd(self.S)
        r = -self.c + torch.sparse.mm(self.AT, self.lam[:, None])[:, 0] - self.PhiT(S)
        r[self.idx_empty] = 0.0      # y_{} is fixed to 1: its multiplier is free (absorbed)
        # the fixed variable y_{}=1 contributes -(r_empty) ... handle via: objective term c_empty = 0 and
        # the constraint y_empty = 1 acts as an equality with free multiplier; bound = b.lam - r_empty_original*1
        r_full = -self.c + torch.sparse.mm(self.AT, self.lam[:, None])[:, 0] - self.PhiT(S)
        bound = float(self.b @ self.lam) - float(r_full[self.idx_empty]) + float(torch.clamp(-r, min=0).sum())
        return bound


if __name__ == "__main__":
    import sys
    eta = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    t0 = time.time()
    P = KVLasserre2(3, eta)
    val = P.solve(iters=iters)
    ub = P.dual_bound()
    rec = {"k": 3, "eta": eta, "iters": iters, "primal_value": val, "dual_upper_bound": ub,
           "n_var": P.n_var, "n_rows": P.nA, "time": time.time() - t0}
    print(json.dumps(rec), flush=True)
    json.dump(rec, open(f"../results/kv3_eta{eta}_lasserre2_sym.json", "w"), indent=1)
