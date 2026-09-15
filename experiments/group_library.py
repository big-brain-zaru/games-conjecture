"""
group_library.py -- a library of finite groups as multiplication tables, for the
non-abelian sweep.  Every group is checked for the axioms on construction.

Families: cyclic, elementary abelian, dihedral D_n, dicyclic Dic_n (incl. Q_8),
metacyclic Z_m x|_r Z_k (incl. every Frobenius group Z_p x| Z_q of small order),
Heisenberg mod p, extraspecial 2^{1+2n}, symmetric and alternating groups,
matrix groups over F_p (SL(2,3), GL(2,3), SL(2,5) = binary icosahedral, ...),
PSL(2,7), wreath products Z_p wr Z_q, affine groups F_p^2 x| C for a cyclic C
acting by a matrix, and direct products with small cyclic groups.
"""
from __future__ import annotations

import itertools
import numpy as np

from group_ug import (Group, cyclic, elementary_abelian, direct_product, semidirect,
                      heisenberg, extraspecial_2, group_from_permutations)


# -- small constructors -------------------------------------------------------
def dihedral(n):
    rot = [(i + 1) % n for i in range(n)]
    ref = [(-i) % n for i in range(n)]
    return group_from_permutations(f"D{n}", [rot, ref], n)


def dicyclic(n):
    """Dic_n of order 4n: elements (a, b), a in Z_{2n}, b in {0,1};
    (a1,b1)(a2,b2) = (a1 + (-1)^{b1} a2 + n*b1*b2, b1+b2).  Dic_2 = Q_8."""
    N = 4 * n
    elems = [(a, b) for b in range(2) for a in range(2 * n)]
    idx = {t: i for i, t in enumerate(elems)}
    mul = np.zeros((N, N), dtype=np.int64)
    for i, (a1, b1) in enumerate(elems):
        for j, (a2, b2) in enumerate(elems):
            a = (a1 + (a2 if b1 == 0 else -a2) + n * b1 * b2) % (2 * n)
            mul[i, j] = idx[(a, (b1 + b2) % 2)]
    inv = np.zeros(N, dtype=np.int64)
    e = idx[(0, 0)]
    for i in range(N):
        inv[i] = int(np.where(mul[i] == e)[0][0])
    return Group(f"Dic{n}", mul, inv, e=e)


def metacyclic(m, k, r):
    """Z_m x|_r Z_k with b acting on a by multiplication by r^b; needs r^k = 1 mod m."""
    assert pow(r, k, m) == 1 % m
    Zm, Zk = cyclic(m), cyclic(k)
    act = {b: np.array([(a * pow(r, b, m)) % m for a in range(m)]) for b in range(k)}
    return semidirect(Zm, Zk, act, name=f"Z{m}x|_{r}Z{k}")


def matrix_group_mod_p(p, gens, name, size_limit=2000):
    """Closure of 2x2 matrices over F_p under multiplication."""
    def mm(A, B):
        return ((A[0] * B[0] + A[1] * B[2]) % p, (A[0] * B[1] + A[1] * B[3]) % p,
                (A[2] * B[0] + A[3] * B[2]) % p, (A[2] * B[1] + A[3] * B[3]) % p)
    I = (1, 0, 0, 1)
    elems = {I: 0}; order = [I]; frontier = [I]
    gens = [tuple(g) for g in gens]
    while frontier:
        x = frontier.pop()
        for g in gens:
            for y in (mm(x, g), mm(g, x)):
                if y not in elems:
                    elems[y] = len(order); order.append(y); frontier.append(y)
                    assert len(order) <= size_limit, "matrix group too large"
    n = len(order)
    mul = np.zeros((n, n), dtype=np.int64)
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            mul[i, j] = elems[mm(a, b)]
    inv = np.zeros(n, dtype=np.int64)
    for i in range(n):
        inv[i] = int(np.where(mul[i] == 0)[0][0])
    return Group(name, mul, inv, e=0)


def wreath_cyclic(p, q):
    """Z_p wr Z_q as permutations of p*q points: rotate one block, and cycle the blocks."""
    n = p * q
    blk0 = list(range(n))
    for i in range(p):
        blk0[i] = (i + 1) % p
    cyc = [((i // p + 1) % q) * p + (i % p) for i in range(n)]
    return group_from_permutations(f"Z{p}wrZ{q}", [blk0, cyc], n)


def affine_plane(p, M, name):
    """F_p^2 x| <M> for an invertible 2x2 matrix M over F_p."""
    def mv(A, v):
        return ((A[0] * v[0] + A[1] * v[1]) % p, (A[2] * v[0] + A[3] * v[1]) % p)
    def mm(A, B):
        return ((A[0] * B[0] + A[1] * B[2]) % p, (A[0] * B[1] + A[1] * B[3]) % p,
                (A[2] * B[0] + A[3] * B[2]) % p, (A[2] * B[1] + A[3] * B[3]) % p)
    pw = [(1, 0, 0, 1)]
    while True:
        nxt = mm(pw[-1], M)
        if nxt == pw[0]:
            break
        pw.append(nxt)
        assert len(pw) < 500
    k = len(pw)
    V = elementary_abelian(0) if False else None
    # vector group F_p^2 as a Group
    vecs = [(a, b) for a in range(p) for b in range(p)]
    vidx = {v: i for i, v in enumerate(vecs)}
    n2 = p * p
    vmul = np.zeros((n2, n2), dtype=np.int64)
    for i, u in enumerate(vecs):
        for j, w in enumerate(vecs):
            vmul[i, j] = vidx[((u[0] + w[0]) % p, (u[1] + w[1]) % p)]
    vinv = np.array([vidx[((-u[0]) % p, (-u[1]) % p)] for u in vecs])
    Vg = Group(f"F{p}^2", vmul, vinv, e=vidx[(0, 0)])
    act = {b: np.array([vidx[mv(pw[b], v)] for v in vecs]) for b in range(k)}
    return semidirect(Vg, cyclic(k), act, name=name)


# -- the library -------------------------------------------------------------
def library(max_order=130, include_abelian=False):
    out = []
    def add(G):
        if G.n <= max_order:
            try:
                G.check()
            except AssertionError as ex:
                print(f"  !! {G.name} failed the group check: {ex}"); return
            if include_abelian or not G.is_abelian():
                out.append(G)
    for n in range(3, 40):
        add(dihedral(n))
    for n in range(2, 20):
        add(dicyclic(n))
    # metacyclic / Frobenius
    for m in range(3, 70):
        for k in range(2, 13):
            if m * k > max_order:
                continue
            for r in range(2, m):
                if np.gcd(r, m) != 1 or pow(r, k, m) != 1:
                    continue
                # only faithful, non-dihedral-duplicate actions: r of order exactly k
                if any(pow(r, j, m) == 1 for j in range(1, k)):
                    continue
                add(metacyclic(m, k, r))
                break            # one action per (m, k) is enough for the sweep
    for p in (3, 5):
        add(heisenberg(p))
    add(extraspecial_2(1)); add(extraspecial_2(2)); add(extraspecial_2(3))
    add(group_from_permutations("A4", [[1, 2, 0, 3], [0, 2, 3, 1]], 4))
    add(group_from_permutations("S4", [[1, 0, 2, 3], [1, 2, 3, 0]], 4))
    add(group_from_permutations("A5", [[1, 2, 0, 3, 4], [0, 1, 3, 4, 2]], 5))
    add(group_from_permutations("S5", [[1, 0, 2, 3, 4], [1, 2, 3, 4, 0]], 5))
    add(group_from_permutations("PSL(2,7)", [[1, 2, 3, 4, 5, 6, 0], [0, 2, 4, 1, 6, 3, 5]], 7))
    add(matrix_group_mod_p(3, [(1, 1, 0, 1), (0, 1, 2, 0)], "SL(2,3)"))
    add(matrix_group_mod_p(3, [(1, 1, 0, 1), (0, 1, 2, 0), (2, 0, 0, 1)], "GL(2,3)"))
    add(matrix_group_mod_p(5, [(1, 1, 0, 1), (0, 1, 4, 0)], "SL(2,5)"))
    add(wreath_cyclic(2, 3)); add(wreath_cyclic(3, 3)); add(wreath_cyclic(2, 4)); add(wreath_cyclic(3, 2))
    add(affine_plane(3, (0, 1, 2, 0), "F3^2x|Z4")); add(affine_plane(3, (1, 1, 0, 1), "F3^2x|Z3"))
    add(affine_plane(5, (0, 1, 4, 0), "F5^2x|Z4")); add(affine_plane(5, (2, 0, 0, 3), "F5^2x|Z4b"))
    add(affine_plane(5, (0, 1, 4, 1), "F5^2x|Z6")); add(affine_plane(7, (2, 0, 0, 4), "F7^2x|Z3"))
    add(affine_plane(3, (0, 2, 1, 1), "F3^2x|Z8"))
    add(affine_plane(2, (0, 1, 1, 1), "F2^2x|Z3=A4"))
    # products with small cyclic groups
    base = list(out)
    for G in base:
        for k in (2, 3):
            if G.n * k <= max_order:
                add(direct_product(G, cyclic(k), name=f"{G.name}xZ{k}"))
    # deduplicate by a cheap isomorphism invariant: (order, multiset of element orders,
    # number of conjugacy classes, size of the centre).  Not a full isomorphism test, but it
    # removes the dihedral/metacyclic and extraspecial duplicates the constructors produce.
    def signature(G):
        n = G.n
        orders = []
        for g in range(n):
            x, k = g, 1
            while x != G.e:
                x = int(G.mul[x, g]); k += 1
            orders.append(k)
        classes = set()
        for g in range(n):
            classes.add(min(int(G.conj(h, g)) for h in range(n)))
        centre = sum(1 for g in range(n) if all(G.mul[g, h] == G.mul[h, g] for h in range(n)))
        return (n, tuple(sorted(orders)), len(classes), centre)
    seen, uniq = set(), []
    for G in sorted(out, key=lambda G: (G.n, len(G.name), G.name)):
        sig = signature(G)
        if sig not in seen:
            seen.add(sig); uniq.append(G)
    return sorted(uniq, key=lambda G: (G.n, G.name))


if __name__ == "__main__":
    L = library(130)
    print(f"{len(L)} non-abelian groups of order <= 130")
    for G in L:
        print(f"  {G.name:18s} order {G.n:4d}")
