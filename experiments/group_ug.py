"""
group_ug.py -- Unique games from group quotients: a generalisation of the
Khot-Vishnoi construction, including the non-abelian case.

THE FRAMEWORK
-------------
Let G be a finite group, H <| G a normal subgroup with |H| = N, Q = G/H, and mu
a symmetric probability measure on G with mu(H) = 0.

Vertices of the game: the cosets Q (|Q| = |G|/N).  Labels: H.
A labelling is a TRANSVERSAL T of H in G (one element per coset), and

    val(T) = sum_{x != y in T} mu(x^{-1} y)      (mu normalised over pairs).

Claim (proved in _selftest by direct comparison with a generic evaluator).
This IS a unique game.  Fix base representatives a_A of each coset; a labelling
is h_A in H with T = {a_A h_A}.  For cosets A, B put c = a_A^{-1} a_B.  As
(h_A, h_B) ranges over H x H, the difference h_A^{-1} c h_B takes each value
z in cH exactly N times, and for fixed z the relation is the BIJECTION
    h_B  =  c^{-1} h_A z        (well defined: H normal => c^{-1} h_A c in H).
So for each ordered coset pair and each z in cH there is one unique constraint of
weight mu(z), and the label-extended graph of the game is exactly Cay(G, mu)
with the cosets of H as the vertex blocks.  Hence
    val(T) = 1 - Phi(T)  in Cay(G, mu),  |T|/|G| = 1/N,
i.e. SOUNDNESS IS SMALL-SET EXPANSION OF Cay(G, mu) AT DENSITY 1/N.

Khot-Vishnoi is the case G = F_2^N (N = 2^k), H = the simplex code
C = {chi_S} (all nonzero words of weight N/2), mu = the truncated noise measure.

COMPLETENESS (the SDP solution).
Let rho be an orthogonal representation of G of dimension d with
    rho|_H  ==  (d/N) . reg_H        <=>   chi_rho(h) = 0 for all h in H \\ {e}.
Put psi(g) = chi_rho(g)/d and give vertex A, label h the vector
    v_{A,h} = u_{a_A h}^{(x)2},   <u_x, u_y> = psi(x^{-1} y),
so <v,w> = psi(.)^2 >= 0.  Orthogonality inside a block is psi(h^{-1}h') = 0 for
h != h' (exactly the condition on rho), the block trace is 1 after scaling, and
the objective is  sum_z mu(z) psi(z)^2.  So a good SDP solution exists whenever
psi ~ 1 on the support of mu.  For KV, rho = (+)_x chi_{e_x} and
psi(f) = 1 - 2 wt(f)/N.

WHY THIS IS THE INTERESTING DIRECTION.
Every known (1-eps, delta) gap instance has an ABELIAN G (noisy cube, short
code), and its soundness is certified by degree-4 sum-of-squares because
hypercontractivity of the abelian noise operator is SoS-provable
(Barak-Brandao-Harrow-Kelner-Steurer-Zhou 2012, Thm 6.11).  For non-abelian G
no such SoS-provable hypercontractivity is available, and Bafna-Minzer (CCC
2024) state that graphs without a certified global-hypercontractivity structure
are exactly where hardness must live.  This module builds those instances.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from ug_core import UniqueGame


# ----------------------------------------------------------------------------
# Groups as multiplication tables
# ----------------------------------------------------------------------------
@dataclass
class Group:
    name: str
    mul: np.ndarray            # (n, n) int: mul[a, b] = a*b
    inv: np.ndarray            # (n,) int
    e: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.mul)

    def check(self):
        n = self.n
        a = np.arange(n)
        assert (self.mul[self.e] == a).all() and (self.mul[:, self.e] == a).all(), "identity"
        assert (self.mul[a, self.inv[a]] == self.e).all(), "inverse"
        # associativity on a random sample (full check for n <= 64)
        rng = np.random.default_rng(0)
        if n <= 64:
            trips = [(x, y, z) for x in range(n) for y in range(n) for z in range(n)]
        else:
            trips = list(zip(rng.integers(0, n, 4000), rng.integers(0, n, 4000), rng.integers(0, n, 4000)))
        for x, y, z in trips:
            assert self.mul[self.mul[x, y], z] == self.mul[x, self.mul[y, z]], "associativity"
        return True

    def is_abelian(self):
        return bool((self.mul == self.mul.T).all())

    def conj(self, g, x):
        """g x g^{-1}"""
        return self.mul[self.mul[g, x], self.inv[g]]

    def subgroup_generated(self, gens):
        S = {self.e}
        frontier = [self.e]
        gens = list(gens)
        while frontier:
            x = frontier.pop()
            for g in gens:
                for y in (self.mul[x, g], self.mul[g, x]):
                    if y not in S:
                        S.add(int(y)); frontier.append(int(y))
        return sorted(S)

    def is_normal(self, H):
        Hs = set(int(h) for h in H)
        for g in range(self.n):
            for h in Hs:
                if int(self.conj(g, h)) not in Hs:
                    return False
        return True

    def normal_subgroups(self, order=None, max_count=200):
        """All normal subgroups (by brute-force closure of small generating sets)."""
        found = {}
        elems = list(range(self.n))
        for r in (1, 2, 3):
            for gens in itertools.combinations(elems[1:], r):
                S = tuple(self.subgroup_generated(gens))
                if S in found:
                    continue
                if order is not None and len(S) != order:
                    continue
                if self.is_normal(S):
                    found[S] = True
                    if len(found) >= max_count:
                        return [np.array(s) for s in found]
        return [np.array(s) for s in found]

    def cosets(self, H):
        """Right cosets Hg?  We use LEFT cosets gH (blocks of the game)."""
        Hs = np.array(sorted(int(h) for h in H))
        cid = -np.ones(self.n, dtype=np.int64)
        reps = []
        for g in range(self.n):
            if cid[g] < 0:
                c = len(reps); reps.append(g)
                cid[self.mul[g, Hs]] = c
        return cid, np.array(reps), Hs


# -- constructions ------------------------------------------------------------
def group_from_permutations(name, perms, degree):
    """Closure of the permutation group generated by `perms` (lists of images)."""
    gens = [tuple(p) for p in perms]
    idp = tuple(range(degree))
    elems = {idp: 0}
    order = [idp]
    frontier = [idp]
    while frontier:
        x = frontier.pop()
        for g in gens:
            y = tuple(x[g[i]] for i in range(degree))     # compose
            if y not in elems:
                elems[y] = len(order); order.append(y); frontier.append(y)
    n = len(order)
    mul = np.zeros((n, n), dtype=np.int64)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            mul[i, j] = elems[tuple(a[b[t]] for t in range(degree))]
    inv = np.zeros(n, dtype=np.int64)
    for i, a in enumerate(order):
        b = [0] * degree
        for t, v in enumerate(a):
            b[v] = t
        inv[i] = elems[tuple(b)]
    return Group(name, mul, inv, e=0, meta={"perms": [list(p) for p in order], "degree": degree})


def elementary_abelian(n_bits):
    n = 1 << n_bits
    a = np.arange(n)
    mul = a[:, None] ^ a[None, :]
    return Group(f"F2^{n_bits}", mul, a.copy(), e=0, meta={"abelian": True})


def cyclic(m):
    a = np.arange(m)
    return Group(f"Z{m}", (a[:, None] + a[None, :]) % m, (-a) % m, e=0)


def direct_product(G1, G2, name=None):
    n1, n2 = G1.n, G2.n
    n = n1 * n2
    idx = lambda a, b: a * n2 + b
    mul = np.zeros((n, n), dtype=np.int64)
    for a1 in range(n1):
        for b1 in range(n2):
            i = idx(a1, b1)
            mul[i] = (G1.mul[a1][:, None] * n2 + G2.mul[b1][None, :]).ravel()
    inv = (G1.inv[:, None] * n2 + G2.inv[None, :]).ravel()
    return Group(name or f"{G1.name}x{G2.name}", mul, inv, e=idx(G1.e, G2.e))


def semidirect(H, K, action, name=None):
    """H x| K with action[k] a permutation array of H (an automorphism), so
    (h1,k1)(h2,k2) = (h1 * action[k1][h2], k1 k2)."""
    nH, nK = H.n, K.n
    n = nH * nK
    idx = lambda h, k: h * nK + k
    mul = np.zeros((n, n), dtype=np.int64)
    for h1 in range(nH):
        for k1 in range(nK):
            i = idx(h1, k1)
            act = action[k1]
            for h2 in range(nH):
                for k2 in range(nK):
                    mul[i, idx(h2, k2)] = idx(H.mul[h1, act[h2]], K.mul[k1, k2])
    inv = np.zeros(n, dtype=np.int64)
    for h in range(nH):
        for k in range(nK):
            ki = K.inv[k]
            inv[idx(h, k)] = idx(action[ki][H.inv[h]], ki)
    return Group(name or f"{H.name}x|{K.name}", mul, inv, e=idx(H.e, K.e))


def heisenberg(p):
    """Upper unitriangular 3x3 matrices over F_p; order p^3."""
    elems = [(a, b, c) for a in range(p) for b in range(p) for c in range(p)]
    idx = {t: i for i, t in enumerate(elems)}
    n = len(elems)
    mul = np.zeros((n, n), dtype=np.int64)
    for i, (a1, b1, c1) in enumerate(elems):
        for j, (a2, b2, c2) in enumerate(elems):
            mul[i, j] = idx[((a1 + a2) % p, (b1 + b2) % p, (c1 + c2 + a1 * b2) % p)]
    inv = np.array([idx[((-a) % p, (-b) % p, (-c + a * b) % p)] for (a, b, c) in elems])
    return Group(f"Heis({p})", mul, inv, e=idx[(0, 0, 0)], meta={"elems": elems, "p": p})


def extraspecial_2(nn):
    """Extraspecial 2-group 2^{1+2n} of plus type (Pauli group mod phases x center):
    elements (x, y, c) in F_2^n x F_2^n x F_2, (x1,y1,c1)(x2,y2,c2) = (x1^x2, y1^y2, c1+c2+ y1.x2)."""
    n2 = 1 << nn
    elems = [(x, y, c) for x in range(n2) for y in range(n2) for c in range(2)]
    idx = {t: i for i, t in enumerate(elems)}
    n = len(elems)
    dot = lambda a, b: bin(a & b).count("1") & 1
    mul = np.zeros((n, n), dtype=np.int64)
    for i, (x1, y1, c1) in enumerate(elems):
        for j, (x2, y2, c2) in enumerate(elems):
            mul[i, j] = idx[(x1 ^ x2, y1 ^ y2, (c1 + c2 + dot(y1, x2)) & 1)]
    inv = np.array([idx[(x, y, (c + dot(y, x)) & 1)] for (x, y, c) in elems])
    return Group(f"2^(1+{2*nn})", mul, inv, e=idx[(0, 0, 0)], meta={"elems": elems, "nn": nn})


# ----------------------------------------------------------------------------
# The unique game
# ----------------------------------------------------------------------------
def group_unique_game(G: Group, H, mu: np.ndarray, name: str = "") -> UniqueGame:
    """mu: (|G|,) nonnegative, symmetric (mu[g] = mu[g^{-1}]), mu[h] = 0 for h in H.
    Returns the unique game on the cosets of H with alphabet H."""
    mu = np.asarray(mu, dtype=np.float64)
    assert np.allclose(mu, mu[G.inv]), "mu must be symmetric"
    cid, reps, Hs = G.cosets(H)
    N = len(Hs)
    nQ = len(reps)
    assert mu[Hs].sum() == 0.0, "mu must vanish on H (no self-loops)"
    hpos = {int(h): i for i, h in enumerate(Hs)}
    edges, perms, weights = [], [], []
    for A in range(nQ):
        for B in range(A + 1, nQ):
            c = G.mul[G.inv[reps[A]], reps[B]]                 # a_A^{-1} a_B
            for z in G.mul[c, Hs]:                              # z in cH
                w = mu[z]
                if w <= 0:
                    continue
                # h_B = c^{-1} h_A z
                ci = G.inv[c]
                p = np.array([hpos[int(G.mul[G.mul[ci, h], z])] for h in Hs], dtype=np.int64)
                edges.append((A, B)); perms.append(p); weights.append(w)
    g = UniqueGame(nQ, N, np.array(edges), np.array(perms), np.array(weights),
                   meta={"family": "group", "group": G.name, "H_order": int(N), "Q_order": int(nQ),
                         "abelian": G.is_abelian(), "name": name})
    g.meta["H"] = Hs.tolist(); g.meta["reps"] = reps.tolist()
    return g


def transversal_value(G: Group, H, mu: np.ndarray, T: np.ndarray) -> float:
    """Direct evaluation: sum over unordered pairs of T of mu(x^{-1}y), normalised
    by the same total the game uses."""
    cid, reps, Hs = G.cosets(H)
    N, nQ = len(Hs), len(reps)
    tot = 0.0
    for i in range(len(T)):
        x = int(T[i])
        for j in range(i + 1, len(T)):
            tot += mu[G.mul[G.inv[x], int(T[j])]]
    # total weight of the game = sum over coset pairs A<B and z in cH of mu(z)
    W = 0.0
    for A in range(nQ):
        for B in range(A + 1, nQ):
            c = G.mul[G.inv[reps[A]], reps[B]]
            W += mu[G.mul[c, Hs]].sum()
    return float(tot / W)


def labelling_to_transversal(G: Group, H, L: np.ndarray) -> np.ndarray:
    cid, reps, Hs = G.cosets(H)
    return np.array([G.mul[reps[A], Hs[L[A]]] for A in range(len(reps))], dtype=np.int64)


# -- noise measures -----------------------------------------------------------
def word_noise(G: Group, gens, eta: float, radius: int = None, H=None):
    """mu(g) proportional to eta^{l(g)} (1-eta)^{...}: a random-walk measure.
    Concretely mu = normalised  sum_{r=1..radius} binom-weighted r-step walks on
    the symmetric generating set, with the H-part removed."""
    n = G.n
    S = sorted(set([int(x) for x in gens] + [int(G.inv[x]) for x in gens]))
    step = np.zeros(n); step[S] = 1.0 / len(S)
    radius = radius or len(S)
    cur = np.zeros(n); cur[G.e] = 1.0
    mu = np.zeros(n)
    for r in range(1, radius + 1):
        nxt = np.zeros(n)
        for s in S:
            nxt[G.mul[:, s]] += cur / len(S)
        cur = nxt
        mu += (eta ** r) * cur
    mu = (mu + mu[G.inv]) / 2
    if H is not None:
        mu[np.asarray(H)] = 0.0
    return mu / mu.sum()


def hamming_noise_f2(n_bits: int, eta: float, dmin: int, dmax: int, H=None):
    """The Khot-Vishnoi measure on F_2^{n_bits}: weight eta^d (1-eta)^{n-d}
    truncated to d in [dmin, dmax]."""
    n = 1 << n_bits
    d = np.array([bin(x).count("1") for x in range(n)])
    mu = np.where((d >= dmin) & (d <= dmax), eta ** d * (1 - eta) ** (n_bits - d), 0.0)
    if H is not None:
        mu[np.asarray(H)] = 0.0
    return mu / mu.sum()


# -- psi / SDP solution from a representation ---------------------------------
def psi_from_permutation_rep(G: Group, K):
    """psi(g) = (#fixed points of g on G/K) / [G:K]."""
    cid, reps, Ks = G.cosets(K)
    d = len(reps)
    fix = np.zeros(G.n)
    for g in range(G.n):
        fix[g] = (cid[G.mul[g, reps]] == np.arange(d)).sum()
    return fix / d, d


def psi_valid(G: Group, H, psi: np.ndarray, tol=1e-9):
    """psi(e)=1, psi=0 on H\\{e}, and the Gram matrix [psi(x^{-1}y)] is PSD."""
    Hs = np.asarray(H)
    ok_e = abs(psi[G.e] - 1) < tol
    ok_H = all(abs(psi[int(h)]) < tol for h in Hs if int(h) != G.e)
    Gram = psi[G.mul[G.inv[np.arange(G.n)][:, None], np.arange(G.n)[None, :]]]
    lam = np.linalg.eigvalsh((Gram + Gram.T) / 2).min()
    return ok_e, ok_H, float(lam)


def sdp_value_of_psi(G: Group, H, mu: np.ndarray, psi: np.ndarray):
    """Objective of the tensor-square vector solution: sum_z mu(z) psi(z)^2,
    normalised the same way as the game."""
    cid, reps, Hs = G.cosets(H)
    nQ = len(reps)
    num = den = 0.0
    for A in range(nQ):
        for B in range(A + 1, nQ):
            c = G.mul[G.inv[reps[A]], reps[B]]
            z = G.mul[c, Hs]
            num += float((mu[z] * psi[z] ** 2).sum())
            den += float(mu[z].sum())
    return num / den


# ----------------------------------------------------------------------------
# Self-test: reproduce Khot-Vishnoi exactly
# ----------------------------------------------------------------------------
def _selftest():
    from kv_transversal import kv_transversal_data, subcube_choice, transversal_value as kv_tv
    # --- 1. G = F_2^8, H = simplex code -> the KV game for k=3 ---------------
    k, eta = 3, 0.2
    D = kv_transversal_data(k, eta)
    G = elementary_abelian(8)
    G.check()
    H = np.array(sorted(int(c) for c in D["C"]))
    assert G.is_normal(H)
    mu = hamming_noise_f2(8, eta, D["dmin"], D["dmax"], H=None)
    g = group_unique_game(G, H, mu, name="KV k=3")
    assert g.n == D["n_cos"] and g.k == D["N"], (g.n, g.k)
    # values must match the verified KV transversal evaluator
    rng = np.random.default_rng(0)
    for _ in range(5):
        ch = rng.integers(0, D["N"], size=D["n_cos"])
        T = D["reps"] ^ D["C"][ch]
        v_kv = kv_tv(D, ch)
        cid, reps, Hs = G.cosets(H)
        L = np.zeros(g.n, dtype=np.int64)
        for x in T:
            A = cid[x]
            L[A] = int(np.where(Hs == G.mul[G.inv[reps[A]], x])[0][0])
        v_grp = g.value(L)
        assert abs(v_kv - v_grp) < 1e-12, (v_kv, v_grp)
    sc = subcube_choice(D)
    print(f"  KV k=3 reproduced from (F_2^8, simplex code, Hamming noise): "
          f"n={g.n} k={g.k} m={g.m}, subcube value {kv_tv(D, sc):.6f}")
    # KV psi: psi(f) = 1 - 2 wt(f)/N
    psi = 1 - 2 * np.array([bin(x).count("1") for x in range(256)]) / 8
    ok_e, ok_H, lam = psi_valid(G, H, psi)
    sdp = sdp_value_of_psi(G, H, mu, psi)
    print(f"    KV psi: psi(e)=1 {ok_e}, psi|H\\e = 0 {ok_H}, Gram lambda_min {lam:.2e}, "
          f"tensor-square SDP objective {sdp:.6f}  (paper bound >= 1-9eta = {1-9*eta:.2f})")
    assert ok_e and ok_H and lam > -1e-9
    # --- 2. a non-abelian example: Heisenberg(3), H = centre-containing normal -
    Hg = heisenberg(3); Hg.check()
    assert not Hg.is_abelian()
    Ns = Hg.normal_subgroups(order=3)
    assert Ns, "no normal subgroup of order 3"
    Hn = Ns[0]
    gens = [i for i, t in enumerate(Hg.meta["elems"]) if t in ((1, 0, 0), (0, 1, 0), (2, 0, 0), (0, 2, 0))]
    mu2 = word_noise(Hg, gens, 0.5, radius=3, H=Hn)
    g2 = group_unique_game(Hg, Hn, mu2, name="Heis3")
    cid, reps, Hs = Hg.cosets(Hn)
    for _ in range(5):
        L = rng.integers(0, len(Hs), size=len(reps))
        T = labelling_to_transversal(Hg, Hn, L)
        assert abs(transversal_value(Hg, Hn, mu2, T) - g2.value(L)) < 1e-12
    print(f"  Heisenberg(3), |H|=3: n={g2.n} k={g2.k} m={g2.m}; transversal evaluator matches the game")
    print("group_ug selftest: OK")


if __name__ == "__main__":
    _selftest()
