"""
ug_core.py -- Core representation of Unique Games instances, generators, exact
solvers, and verification.

Conventions
-----------
A Unique Game G = (V, E, {pi_e}, w) with alphabet [k]:
  * vertices 0..n-1
  * edge e = (u, v) carries a bijection pi_e : [k] -> [k]; the edge is
    SATISFIED by an assignment L iff  L[v] == pi_e[L[u]].
  * weights w_e > 0.  value(L) = sum_{e satisfied} w_e / sum_e w_e.

Every function that claims a value returns it *only* after direct evaluation of
the constraint set (verifier-first, as in the fast-matrix project).  Exact
optima come from MaxSAT (RC2/CaDiCaL) or brute force, never from heuristics.

Label-extended graph: nodes (v, a) for v in V, a in [k]; edge e=(u,v) with
permutation pi contributes k edges (u,a) -- (v, pi[a]) of weight w_e.
An assignment is a choice of one node per vertex-block; its value is the
weight of induced edges over total weight.  This is the spectral object behind
every SDP and every "threshold rank" algorithm.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import scipy.sparse as sp


# ----------------------------------------------------------------------------
# Instance
# ----------------------------------------------------------------------------
@dataclass
class UniqueGame:
    n: int
    k: int
    edges: np.ndarray            # (m, 2) int64
    perms: np.ndarray            # (m, k) int64 ; perms[e][a] = label of v forced by label a of u
    weights: np.ndarray = None   # (m,) float64
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.perms = np.asarray(self.perms, dtype=np.int64).reshape(len(self.edges), self.k)
        if self.weights is None:
            self.weights = np.ones(len(self.edges), dtype=np.float64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        assert self.perms.shape == (len(self.edges), self.k)
        # every row must be a permutation
        srt = np.sort(self.perms, axis=1)
        assert np.all(srt == np.arange(self.k)[None, :]), "perms rows must be bijections"
        assert self.edges.min() >= 0 and self.edges.max() < self.n

    # -- basic quantities --------------------------------------------------
    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def total_weight(self) -> float:
        return float(self.weights.sum())

    def value(self, L: Sequence[int]) -> float:
        """Fraction of weight satisfied by assignment L (length n, entries in [k])."""
        L = np.asarray(L, dtype=np.int64)
        assert L.shape == (self.n,)
        u, v = self.edges[:, 0], self.edges[:, 1]
        sat = self.perms[np.arange(self.m), L[u]] == L[v]
        return float(self.weights[sat].sum() / self.total_weight)

    def satisfied_mask(self, L: Sequence[int]) -> np.ndarray:
        L = np.asarray(L, dtype=np.int64)
        u, v = self.edges[:, 0], self.edges[:, 1]
        return self.perms[np.arange(self.m), L[u]] == L[v]

    def inverse_perms(self) -> np.ndarray:
        inv = np.empty_like(self.perms)
        rows = np.arange(self.m)[:, None]
        inv[rows, self.perms] = np.arange(self.k)[None, :]
        return inv

    # -- label-extended graph ---------------------------------------------
    def label_extended_adjacency(self, symmetric: bool = True) -> sp.csr_matrix:
        """Weighted adjacency of the label-extended graph on n*k nodes.
        node id = v*k + a."""
        e = self.edges
        a = np.arange(self.k)
        rows = (e[:, 0][:, None] * self.k + a[None, :]).ravel()
        cols = (e[:, 1][:, None] * self.k + self.perms).ravel()
        w = np.repeat(self.weights, self.k)
        A = sp.coo_matrix((w, (rows, cols)), shape=(self.n * self.k, self.n * self.k))
        if symmetric:
            A = A + A.T
        return A.tocsr()

    def normalized_label_extended(self) -> sp.csr_matrix:
        """D^{-1/2} A D^{-1/2} of the label-extended graph (symmetric)."""
        A = self.label_extended_adjacency(symmetric=True)
        d = np.asarray(A.sum(axis=1)).ravel()
        d[d == 0] = 1.0
        Dm = sp.diags(1.0 / np.sqrt(d))
        return (Dm @ A @ Dm).tocsr()

    def base_graph_adjacency(self) -> sp.csr_matrix:
        e = self.edges
        A = sp.coo_matrix((self.weights, (e[:, 0], e[:, 1])), shape=(self.n, self.n))
        return (A + A.T).tocsr()

    # -- perfect-completeness check (poly time) ---------------------------
    def satisfiable_perfectly(self) -> Optional[np.ndarray]:
        """If value 1 is achievable, return a satisfying assignment, else None.
        Unique games with completeness 1 are polytime: propagate labels through
        each connected component from one seed label; k tries per component."""
        adj = [[] for _ in range(self.n)]
        inv = self.inverse_perms()
        for e, (u, v) in enumerate(self.edges):
            adj[u].append((v, self.perms[e]))
            adj[v].append((u, inv[e]))
        L = -np.ones(self.n, dtype=np.int64)
        seen = np.zeros(self.n, dtype=bool)
        for s in range(self.n):
            if seen[s]:
                continue
            ok = False
            for start_label in range(self.k):
                L_comp = {}
                stack = [(s, start_label)]
                good = True
                while stack and good:
                    x, lx = stack.pop()
                    if x in L_comp:
                        if L_comp[x] != lx:
                            good = False
                        continue
                    L_comp[x] = lx
                    for y, p in adj[x]:
                        stack.append((y, int(p[lx])))
                if good:
                    for x, lx in L_comp.items():
                        L[x] = lx
                        seen[x] = True
                    ok = True
                    break
            if not ok:
                return None
        assert self.value(L) == 1.0
        return L

    # -- serialization ----------------------------------------------------
    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "n": self.n, "k": self.k,
                "edges": self.edges.tolist(), "perms": self.perms.tolist(),
                "weights": self.weights.tolist(), "meta": self.meta,
            }, f)

    @staticmethod
    def from_json(path: str) -> "UniqueGame":
        with open(path) as f:
            d = json.load(f)
        return UniqueGame(d["n"], d["k"], np.array(d["edges"]), np.array(d["perms"]),
                          np.array(d["weights"]), d.get("meta", {}))

    def summary(self) -> str:
        return (f"UG(n={self.n}, k={self.k}, m={self.m}, total_w={self.total_weight:.4g}, "
                f"meta={ {a: b for a, b in self.meta.items() if not isinstance(b, (list, dict))} })")


# ----------------------------------------------------------------------------
# Generators
# ----------------------------------------------------------------------------
def _random_graph_edges(n: int, m: int, rng: np.random.Generator, simple: bool = True) -> np.ndarray:
    if simple:
        pairs = set()
        max_pairs = n * (n - 1) // 2
        assert m <= max_pairs, "too many edges for a simple graph"
        while len(pairs) < m:
            u = rng.integers(0, n, size=m)
            v = rng.integers(0, n, size=m)
            for a, b in zip(u, v):
                if a != b:
                    pairs.add((min(a, b), max(a, b)))
                if len(pairs) >= m:
                    break
        return np.array(sorted(pairs), dtype=np.int64)
    u = rng.integers(0, n, size=m)
    v = rng.integers(0, n, size=m)
    return np.stack([u, v], axis=1)


def _random_regular_edges(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    import networkx as nx
    G = nx.random_regular_graph(d, n, seed=int(rng.integers(0, 2**31 - 1)))
    return np.array(sorted((min(a, b), max(a, b)) for a, b in G.edges()), dtype=np.int64)


def random_unique_game(n: int, k: int, m: int = None, degree: int = None, seed: int = 0) -> UniqueGame:
    """Uniformly random permutations on a random (simple or d-regular) graph."""
    rng = np.random.default_rng(seed)
    if degree is not None:
        edges = _random_regular_edges(n, degree, rng)
    else:
        edges = _random_graph_edges(n, m, rng)
    perms = np.stack([rng.permutation(k) for _ in range(len(edges))])
    return UniqueGame(n, k, edges, perms, meta={"family": "random", "seed": seed})


def planted_unique_game(n: int, k: int, m: int = None, degree: int = None, noise: float = 0.1,
                        seed: int = 0, linear: bool = False, q: int = None) -> UniqueGame:
    """Planted assignment; each edge's constraint is consistent with the planted
    labeling, then with probability `noise` replaced by a uniformly random
    (generally inconsistent) constraint.  If linear=True the constraints are
    cyclic shifts (Max-2-Lin mod k)."""
    rng = np.random.default_rng(seed)
    if degree is not None:
        edges = _random_regular_edges(n, degree, rng)
    else:
        edges = _random_graph_edges(n, m, rng)
    L = rng.integers(0, k, size=n)
    perms = np.empty((len(edges), k), dtype=np.int64)
    noisy = rng.random(len(edges)) < noise
    for e, (u, v) in enumerate(edges):
        if linear:
            if noisy[e]:
                c = rng.integers(0, k)
            else:
                c = (L[v] - L[u]) % k
            perms[e] = (np.arange(k) + c) % k
        else:
            if noisy[e]:
                perms[e] = rng.permutation(k)
            else:
                p = rng.permutation(k)
                # force p[L[u]] == L[v] by one transposition (stays a bijection)
                j = int(np.where(p == L[v])[0][0])
                p[j], p[L[u]] = p[L[u]], p[j]
                assert p[L[u]] == L[v]
                perms[e] = p
    g = UniqueGame(n, k, edges, perms, meta={"family": "planted", "noise": noise, "seed": seed,
                                             "linear": linear})
    g.meta["planted_value"] = g.value(L)
    g.meta["planted"] = L.tolist()
    return g


def max2lin_from_constraints(n: int, q: int, cons: Iterable[tuple], weights=None) -> UniqueGame:
    """cons: iterable of (u, v, c) meaning  x_v - x_u == c (mod q)."""
    cons = list(cons)
    edges = np.array([(u, v) for u, v, _ in cons], dtype=np.int64)
    perms = np.array([(np.arange(q) + c) % q for _, _, c in cons], dtype=np.int64)
    return UniqueGame(n, q, edges, perms, weights, meta={"family": "max2lin", "q": q})


# ----------------------------------------------------------------------------
# Exact solvers
# ----------------------------------------------------------------------------
def brute_force_optimum(g: UniqueGame, limit: int = 2_000_000):
    """Exhaustive enumeration over k^n assignments (with a safety limit)."""
    total = g.k ** g.n
    assert total <= limit, f"k^n = {total} exceeds limit {limit}"
    best, bestL = -1.0, None
    u, v = g.edges[:, 0], g.edges[:, 1]
    ar = np.arange(g.m)
    W = g.total_weight
    for L in itertools.product(range(g.k), repeat=g.n):
        L = np.array(L)
        val = g.weights[g.perms[ar, L[u]] == L[v]].sum() / W
        if val > best:
            best, bestL = val, L.copy()
    return best, bestL


def maxsat_optimum(g: UniqueGame, weight_scale: int = 10**6, solver: str = "cadical195",
                   timeout: Optional[float] = None, verbose: bool = False):
    """Exact optimum via weighted partial MaxSAT (RC2 core-guided, CaDiCaL).
    Encoding: x[v,a] one-hot per vertex (hard), s[e] soft with weight w_e;
    s[e] -> OR_a y[e,a];  y[e,a] -> x[u,a];  y[e,a] -> x[v, pi_e(a)].
    Returns (value, assignment, info).  The value is re-verified by g.value."""
    from pysat.formula import WCNF
    from pysat.examples.rc2 import RC2
    n, k = g.n, g.k
    x = lambda v, a: v * k + a + 1
    nx = n * k
    s = lambda e: nx + e + 1
    y = lambda e, a: nx + g.m + e * k + a + 1
    wcnf = WCNF()
    for v in range(n):
        wcnf.append([x(v, a) for a in range(k)])
        for a in range(k):
            for b in range(a + 1, k):
                wcnf.append([-x(v, a), -x(v, b)])
    intw = np.maximum(1, np.round(g.weights / g.weights.max() * weight_scale)).astype(np.int64)
    for e, (u, v) in enumerate(g.edges.tolist()):
        wcnf.append([-s(e)] + [y(e, a) for a in range(k)])
        for a in range(k):
            wcnf.append([-y(e, a), x(u, a)])
            wcnf.append([-y(e, a), x(v, int(g.perms[e, a]))])
        wcnf.append([s(e)], weight=int(intw[e]))
    t0 = time.time()
    with RC2(wcnf, solver=solver, adapt=True, exhaust=True, minz=True, verbose=1 if verbose else 0) as rc2:
        model = rc2.compute()
        cost = rc2.cost
    L = np.zeros(n, dtype=np.int64)
    for v in range(n):
        for a in range(k):
            if model[x(v, a) - 1] > 0:
                L[v] = a
    val = g.value(L)
    return val, L, {"time": time.time() - t0, "cost": cost, "solver": solver}


def local_search_polish(g: UniqueGame, L: np.ndarray, rounds: int = 50, rng=None,
                        A_ext: "sp.csr_matrix" = None, colors: np.ndarray = None) -> np.ndarray:
    """Greedy single-vertex relabeling to a local optimum, vectorised:
    gains[(v,a)] = (A_ext @ onehot(L))[(v,a)] is the satisfied weight if v takes
    label a.  Vertices are updated one colour class at a time (Gauss-Seidel on a
    proper colouring of the base graph), so every update is an exact improvement."""
    rng = rng or np.random.default_rng(0)
    L = np.array(L, dtype=np.int64)
    n, k = g.n, g.k
    if A_ext is None:
        A_ext = g.label_extended_adjacency(symmetric=True)
    if colors is None:
        colors = greedy_coloring(g)
    classes = [np.where(colors == c)[0] for c in range(colors.max() + 1)]
    cur = g.value(L)
    for _ in range(rounds):
        improved = False
        for idx in rng.permutation(len(classes)):
            cls = classes[idx]
            x = np.zeros(n * k); x[np.arange(n) * k + L] = 1.0
            gains = (A_ext @ x).reshape(n, k)
            best = gains[cls].argmax(axis=1)
            better = gains[cls, best] > gains[cls, L[cls]] + 1e-12
            if better.any():
                L[cls[better]] = best[better]
                improved = True
        if not improved:
            break
    assert g.value(L) >= cur - 1e-12
    return L


def greedy_coloring(g: UniqueGame) -> np.ndarray:
    """Proper vertex colouring of the base graph (greedy, largest-first)."""
    import networkx as nx
    G = nx.Graph(); G.add_nodes_from(range(g.n))
    e = g.edges[g.edges[:, 0] != g.edges[:, 1]]
    G.add_edges_from(map(tuple, e.tolist()))
    col = nx.coloring.greedy_color(G, strategy="largest_first")
    return np.array([col[v] for v in range(g.n)])


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(1)
    # 1. planted, no noise -> perfectly satisfiable, propagation finds value 1
    g = planted_unique_game(30, 4, m=80, noise=0.0, seed=3)
    L = g.satisfiable_perfectly()
    assert L is not None and g.value(L) == 1.0
    # 2. random small: brute force == maxsat
    for seed in range(3):
        g = random_unique_game(7, 3, m=15, seed=seed)
        bf, _ = brute_force_optimum(g)
        ms, Lm, info = maxsat_optimum(g)
        assert abs(bf - ms) < 1e-9, (bf, ms)
        assert abs(g.value(Lm) - ms) < 1e-9
    # 3. planted noisy: maxsat >= planted value
    g = planted_unique_game(25, 5, m=60, noise=0.2, seed=7)
    ms, Lm, info = maxsat_optimum(g)
    assert ms >= g.meta["planted_value"] - 1e-12
    # 4. label extended graph: value == induced weight
    g = random_unique_game(20, 4, m=50, seed=5)
    A = g.label_extended_adjacency()
    L = rng.integers(0, 4, size=20)
    idx = np.arange(20) * 4 + L
    induced = A[idx][:, idx].sum() / 2
    assert abs(induced / g.total_weight - g.value(L)) < 1e-9
    # 5. max2lin
    g = max2lin_from_constraints(4, 5, [(0, 1, 2), (1, 2, 3), (2, 3, 1), (3, 0, 4)])
    assert g.satisfiable_perfectly() is not None  # 2+3+1+4 = 10 = 0 mod 5
    g2 = max2lin_from_constraints(4, 5, [(0, 1, 2), (1, 2, 3), (2, 3, 1), (3, 0, 3)])
    assert g2.satisfiable_perfectly() is None
    print("ug_core selftest: OK")


if __name__ == "__main__":
    _selftest()


# ----------------------------------------------------------------------------
# Exact solver 2: mixed-integer program (HiGHS via scipy)
# ----------------------------------------------------------------------------
def milp_optimum(g: UniqueGame, time_limit: float = 600.0, mip_gap: float = 0.0, verbose: bool = False,
                 fix_vertex0: bool = False):
    """Exact optimum by MILP: x[v,a] binary one-hot; z[e,a] <= x[u,a], z[e,a] <= x[v,pi(a)],
    maximise sum_e w_e sum_a z[e,a].  Returns (value, assignment, info); info['bound'] is the
    solver's dual bound (value == bound certifies optimality).  Value is re-verified."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix, csr_matrix, vstack
    n, k, m = g.n, g.k, g.m
    nx = n * k
    nz = m * k
    nv = nx + nz
    W = g.total_weight
    c = np.zeros(nv)
    for e in range(m):
        c[nx + e * k: nx + (e + 1) * k] = -g.weights[e] / W          # minimise negative
    # one-hot rows
    A1 = lil_matrix((n, nv))
    for v in range(n):
        A1[v, v * k:(v + 1) * k] = 1.0
    # z <= x rows (two per (e,a))
    rows, cols, vals = [], [], []
    r = 0
    for e, (u, v) in enumerate(g.edges.tolist()):
        for a in range(k):
            zi = nx + e * k + a
            rows += [r, r]; cols += [zi, u * k + a]; vals += [1.0, -1.0]; r += 1
            rows += [r, r]; cols += [zi, v * k + int(g.perms[e, a])]; vals += [1.0, -1.0]; r += 1
    A2 = csr_matrix((vals, (rows, cols)), shape=(r, nv))
    A = vstack([A1.tocsr(), A2]).tocsr()
    lb = np.concatenate([np.ones(n), -np.inf * np.ones(r)])
    ub = np.concatenate([np.ones(n), np.zeros(r)])
    integrality = np.concatenate([np.ones(nx), np.zeros(nz)])
    lo = np.zeros(nv); hi = np.ones(nv)
    if fix_vertex0:
        hi[1:k] = 0.0; lo[0] = 1.0
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality,
               bounds=Bounds(lo, hi), options={"time_limit": time_limit, "mip_rel_gap": mip_gap, "disp": verbose})
    L = np.zeros(n, dtype=np.int64)
    if res.x is not None:
        L = np.argmax(res.x[:nx].reshape(n, k), axis=1)
    val = g.value(L)
    bound = -float(res.mip_dual_bound) if getattr(res, "mip_dual_bound", None) is not None else None
    return val, L, {"time": time.time() - t0, "status": res.status, "message": res.message,
                    "bound": bound, "optimal": bool(res.status == 0)}


# ----------------------------------------------------------------------------
# Exact solver 3: CP-SAT (OR-Tools), integer weights required
# ----------------------------------------------------------------------------
def cpsat_optimum(g: UniqueGame, int_weights: np.ndarray = None, time_limit: float = 600.0, workers: int = 8,
                  fix_vertex0: bool = True, verbose: bool = False):
    """Exact optimum via CP-SAT.  int_weights: integer weights proportional to g.weights
    (if None, weights are rounded after scaling by 1e6).  Returns (value, assignment, info)
    with info['optimal'] True iff CP-SAT proved optimality; value re-verified with g.value."""
    from ortools.sat.python import cp_model
    n, k = g.n, g.k
    if int_weights is None:
        int_weights = np.maximum(1, np.round(g.weights / g.weights.max() * 1e6)).astype(np.int64)
    mdl = cp_model.CpModel()
    x = [[mdl.NewBoolVar(f"x{v}_{a}") for a in range(k)] for v in range(n)]
    for v in range(n):
        mdl.AddExactlyOne(x[v])
    if fix_vertex0:
        mdl.Add(x[0][0] == 1)
    sat = []
    for e, (u, v) in enumerate(g.edges.tolist()):
        s = mdl.NewBoolVar(f"s{e}")
        # s <= sum_a x[u][a] * x[v][pi(a)]  encoded via y_{e,a} <= x[u][a], y <= x[v][pi(a)], s <= sum y
        ys = []
        for a in range(k):
            y = mdl.NewBoolVar(f"y{e}_{a}")
            mdl.AddImplication(y, x[u][a]); mdl.AddImplication(y, x[v][int(g.perms[e, a])])
            ys.append(y)
        mdl.Add(sum(ys) >= s)
        sat.append((s, int(int_weights[e])))
    mdl.Maximize(sum(w * s for s, w in sat))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = verbose
    t0 = time.time()
    st = solver.Solve(mdl)
    L = np.array([next(a for a in range(k) if solver.Value(x[v][a])) for v in range(n)], dtype=np.int64)
    val = g.value(L)
    return val, L, {"time": time.time() - t0, "status": solver.StatusName(st), "optimal": st == cp_model.OPTIMAL,
                    "objective": solver.ObjectiveValue(), "bound": solver.BestObjectiveBound(),
                    "bound_value": solver.BestObjectiveBound() / float(int_weights.sum()) if int_weights.sum() else None}


def cpsat_xor_optimum(g: UniqueGame, int_weights: np.ndarray, kbits: int, time_limit: float = 600.0, workers: int = 8,
                      verbose: bool = False):
    """Exact optimum for a LINEAR unique game over F_2^kbits (perms are XOR by a constant):
    kbits Boolean vars per vertex, one indicator per edge enforced by XOR constraints.
    Vertex 0 fixed to label 0 (valid by label-translation symmetry)."""
    from ortools.sat.python import cp_model
    n, k = g.n, g.k
    assert k == 1 << kbits
    mdl = cp_model.CpModel()
    b = [[mdl.NewBoolVar(f"b{v}_{i}") for i in range(kbits)] for v in range(n)]
    for i in range(kbits):
        mdl.Add(b[0][i] == 0)
    obj = []
    for e, (u, v) in enumerate(g.edges.tolist()):
        c = int(g.perms[e, 0])                      # label(v) = label(u) xor c
        assert np.all(g.perms[e] == (np.arange(k) ^ c))
        s = mdl.NewBoolVar(f"s{e}")
        for i in range(kbits):
            ci = (c >> i) & 1
            # s -> (b_u_i xor b_v_i == ci)
            if ci == 0:
                mdl.Add(b[u][i] == b[v][i]).OnlyEnforceIf(s)
            else:
                mdl.Add(b[u][i] != b[v][i]).OnlyEnforceIf(s)
        obj.append(int(int_weights[e]) * s)
    mdl.Maximize(sum(obj))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = verbose
    t0 = time.time()
    st = solver.Solve(mdl)
    L = np.array([sum(solver.Value(b[v][i]) << i for i in range(kbits)) for v in range(n)], dtype=np.int64)
    val = g.value(L)
    return val, L, {"time": time.time() - t0, "status": solver.StatusName(st), "optimal": st == cp_model.OPTIMAL,
                    "bound_value": solver.BestObjectiveBound() / float(int_weights.sum())}
