"""
kv_instance.py -- The Khot-Vishnoi (FOCS 2005 / JACM 2015, arXiv:1305.4581)
integrality-gap instance for Unique Games, built exactly as in Section 3.2.

Construction (parameters k, eta; N = 2^k):
  * F = all Boolean functions f : {-1,1}^k -> {-1,1}   (|F| = 2^N)
  * f ~ g  iff  f = g * chi_S for some S subset [k]    (classes of size N)
  * Vertices of the UG = the 2^N / N classes.  Labels = F_2^k (subsets S).
  * Noisy hypercube H on F: wt'({f,g}) = 2 * 2^{-N} * eta^d (1-eta)^{N-d},
    d = Hamming distance(f, g); edges with d outside [eta N/2, 2 eta N] deleted.
  * For f = [P_i] chi_S, g = [P_j] chi_T: the UG edge {P_i, P_j} carries the
    bijection  pi(T xor U) = S xor U, i.e. the linear constraint
        label(P_i) xor label(P_j) = S xor T.
    Its weight is N * wt'({f,g}) (accumulated pair by pair); bundles with
    equal (i, j, S xor T) merge.  Total weight = 1 - (1-eta)^N minus the
    truncated distances.
  * Soundness: opt <= 1/N^eta (Bonami-Beckner small-set expansion of H).
  * Completeness: the vectors u_{f chi_S}^{(x)2} give SDP value >= 1-9eta
    and satisfy orthogonality, nonnegativity, sum-to-N and triangle inequalities.

Feasible sizes on this machine: k=3 (N=8, 32 vertices), k=4 (N=16, 4096
vertices, ~1.4M weighted edges); k=5 (N=32, 2^27 vertices) only via sampling.
"""
from __future__ import annotations

import itertools
import math
import numpy as np
from ug_core import UniqueGame


def boolean_functions(k: int) -> np.ndarray:
    """All 2^N truth tables as +-1 arrays, shape (2^N, N), N = 2^k.
    Truth table index x runs over {0,1}^k in binary order."""
    N = 1 << k
    ids = np.arange(1 << N, dtype=np.int64)
    bits = ((ids[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)
    return 1 - 2 * bits          # bit 0 -> +1, bit 1 -> -1


def characters(k: int) -> np.ndarray:
    """chi_S(x) = prod_{i in S} x_i for all S (rows) and x (cols); shape (N, N)."""
    N = 1 << k
    S = np.arange(N)
    X = np.arange(N)
    par = np.zeros((N, N), dtype=np.int64)
    for i in range(k):
        par ^= ((S[:, None] >> i) & 1) & ((X[None, :] >> i) & 1)
    return 1 - 2 * par           # +1 if |S cap x| even


def class_structure(k: int):
    """Return (cls, S_of, reps): for every function id, its class index and the
    S with f = rep * chi_S; reps[c] = function id of the class representative
    (smallest id in the class)."""
    N = 1 << k
    F = boolean_functions(k)                    # (2^N, N)
    chi = characters(k)                         # (N, N)
    # multiply f by chi_S: table -> table * chi_S ; encode back to id
    pow2 = (1 << np.arange(N)).astype(np.int64)
    def encode(tab):                            # tab in {-1,1}^N -> id
        return (((1 - tab) // 2) * pow2).sum(axis=-1)
    n_f = 1 << N
    cls = -np.ones(n_f, dtype=np.int64)
    S_of = np.zeros(n_f, dtype=np.int64)
    reps = []
    for fid in range(n_f):
        if cls[fid] >= 0:
            continue
        c = len(reps)
        reps.append(fid)
        tabs = F[fid][None, :] * chi             # (N, N): row S = f*chi_S
        ids = encode(tabs)
        assert len(set(ids.tolist())) == N, "class must have N distinct members"
        cls[ids] = c
        S_of[ids] = np.arange(N)
    return cls, S_of, np.array(reps), F


def khot_vishnoi(k: int, eta: float, d_range=None, keep_all_distances: bool = False) -> UniqueGame:
    """Build the KV instance exactly.  d_range overrides the typical-distance
    window [eta N/2, 2 eta N] (inclusive, integers).  With
    keep_all_distances=True every Hamming distance 1..N is kept (the
    un-truncated noisy hypercube)."""
    N = 1 << k
    cls, S_of, reps, F = class_structure(k)
    n_f = 1 << N
    n_v = n_f // N
    if keep_all_distances:
        dmin, dmax = 1, N
    elif d_range is not None:
        dmin, dmax = d_range
    else:
        dmin = max(1, int(math.ceil(eta * N / 2)))
        dmax = min(N, int(math.floor(2 * eta * N)))
    assert dmin <= dmax, f"empty distance window [{dmin},{dmax}] for k={k}, eta={eta}"
    # weight of a single noisy-hypercube edge at distance d (both orientations)
    wd = {d: 2.0 * (2.0 ** -N) * (eta ** d) * ((1 - eta) ** (N - d)) for d in range(dmin, dmax + 1)}
    # enumerate all pairs (f, g) with distance in window: g = f xor mask, popcount(mask) in window
    masks = [m for m in range(1, n_f) if dmin <= bin(m).count("1") <= dmax]
    masks = np.array(masks, dtype=np.int64)
    pc = np.array([bin(m).count("1") for m in masks])
    acc = {}
    fids = np.arange(n_f, dtype=np.int64)
    for m, d in zip(masks, pc):
        g = fids ^ m
        sel = fids < g                                   # each unordered pair once
        f_sel, g_sel = fids[sel], g[sel]
        ci, cj = cls[f_sel], cls[g_sel]
        c = S_of[f_sel] ^ S_of[g_sel]                    # constraint offset
        w = wd[int(d)]        # per (f,g) pair; a bundle of N pairs sums to N*wt' as in the paper
        # canonical orientation: (min class, max class); offset symmetric (xor)
        lo = np.minimum(ci, cj); hi = np.maximum(ci, cj)
        keys = (lo * n_v + hi) * N + c
        uniq, cnt = np.unique(keys, return_counts=True)
        for key, ct in zip(uniq.tolist(), cnt.tolist()):
            acc[key] = acc.get(key, 0.0) + w * ct
    keys = np.array(sorted(acc.keys()), dtype=np.int64)
    ws = np.array([acc[int(kk)] for kk in keys])
    c = keys % N
    ij = keys // N
    lo = ij // n_v
    hi = ij % n_v
    self_loops = lo == hi
    # self-loops (f,g in same class) can occur: label(P) xor label(P) = c is
    # satisfiable iff c == 0; c==0 would need f==g (distance 0), excluded, so
    # these are never satisfiable -> keep as constant unsatisfied weight.
    edges = np.stack([lo, hi], axis=1)
    labels = np.arange(N)
    perms = labels[None, :] ^ c[:, None]
    g = UniqueGame(n_v, N, edges, perms, ws,
                   meta={"family": "khot_vishnoi", "k": k, "N": N, "eta": eta,
                         "d_window": [dmin, dmax], "n_self_loops": int(self_loops.sum()),
                         "self_loop_weight": float(ws[self_loops].sum()),
                         "soundness_bound": N ** (-eta), "sdp_completeness_bound": 1 - 9 * eta,
                         "linear_over": "F2^k"})
    g.meta["reps"] = reps.tolist()
    return g


def kv_sdp_vectors(g: UniqueGame):
    """The KV SDP solution: for vertex P_i with representative f, label S gets the
    vector u_{f chi_S}^{(x)2} in R^{N^2}.  Returns array (n_v, N, N*N) and the
    SDP objective value computed directly from the instance's edges."""
    k, N = g.meta["k"], g.meta["N"]
    reps = np.array(g.meta["reps"])
    _, _, _, F = class_structure(k)
    chi = characters(k)
    vecs = np.zeros((g.n, N, N * N))
    for i, fid in enumerate(reps):
        tabs = F[fid][None, :] * chi / math.sqrt(N)          # u_{f chi_S}
        vecs[i] = np.einsum("sa,sb->sab", tabs, tabs).reshape(N, N * N)
    # objective: sum_e w_e (1/N) sum_a <v_{pi(a)}, w_a>  with pi: label(u)->label(v)
    val = 0.0
    for e, (u, v) in enumerate(g.edges):
        p = g.perms[e]
        val += g.weights[e] * np.einsum("ij,ij->", vecs[u], vecs[v][p]) / N
    return vecs, val / g.total_weight


if __name__ == "__main__":
    import time
    for k, eta in [(3, 0.2), (3, 0.3)]:
        t = time.time()
        g = khot_vishnoi(k, eta)
        print(g.summary(), f"built in {time.time()-t:.2f}s")
        vecs, sdp = kv_sdp_vectors(g)
        print(f"  KV SDP vector objective = {sdp:.4f}  (paper bound >= {1-9*eta:.3f})")
        # exact optimum by MaxSAT (32 vertices, 8 labels)
        from ug_core import maxsat_optimum
        val, L, info = maxsat_optimum(g)
        print(f"  exact opt = {val:.4f} in {info['time']:.1f}s   (paper bound <= {g.meta['soundness_bound']:.4f})")
