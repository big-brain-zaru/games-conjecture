"""
wick_lift.py -- an explicit candidate for the degree-4 extension of a degree-2 optimum.

Given the optimal degree-2 Gram matrix Y (PSD, unit diagonal) of a Boolean 2Lin
instance, define the scaled Wick lift

    E[x_a x_b]         = Y_ab
    E[x_a x_b x_c x_d] = c * (Y_ab Y_cd + Y_ac Y_bd + Y_ad Y_bc)   (a,b,c,d distinct)
    odd moments        = 0,  x_i^2 = 1.

Its objective equals SoS_2 whatever c is, so if the moment matrix is PSD for some
c in [0, 1] then SoS_4 = SoS_2 exactly and the whole degree-2 gap survives degree
4 (retention 1) -- at the price of ONE eigendecomposition instead of an SDP.
This is the explicit mechanism behind the Mohanty-Raghavendra-Xu degree-2 ->
degree-4 lifting, tested here at finite n on Paley graphs, generalised Paley
circulants and sparse expanders.  For vertex-transitive instances Y is the
normalised projector onto the top eigenspace of the signed adjacency.
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np


def moment_matrix(Y, c):
    n = Y.shape[0]
    basis = [()] + [(i,) for i in range(n)] + list(itertools.combinations(range(n), 2))
    D = len(basis)
    M = np.zeros((D, D))

    def mom(S):
        S = tuple(sorted(S))
        if len(S) == 0:
            return 1.0
        if len(S) == 2:
            return Y[S[0], S[1]]
        if len(S) == 4:
            a, b, cc, d = S
            return c * (Y[a, b] * Y[cc, d] + Y[a, cc] * Y[b, d] + Y[a, d] * Y[b, cc])
        return 0.0

    for i, s in enumerate(basis):
        for j in range(i, D):
            t = basis[j]
            S = set(s) ^ set(t)
            M[i, j] = M[j, i] = mom(S)
    return M


def best_scale(Y, cs=np.linspace(0, 1, 21)):
    out = []
    for c in cs:
        M = moment_matrix(Y, c)
        lam = np.linalg.eigvalsh(M)[0]
        out.append((c, lam))
    c_best, lam_best = max(out, key=lambda t: t[1])
    return c_best, lam_best, out


def paley_Y(p):
    chi = np.zeros(p, dtype=int)
    for a in range(1, p):
        chi[pow(a, 2, p)] = 1
    chi = np.where(chi == 1, 1, -1); chi[0] = 0
    Y = np.zeros((p, p))
    for u in range(p):
        for v in range(p):
            d = (u - v) % p
            Y[u, v] = 1.0 if d == 0 else (-1 - np.sqrt(p) * chi[d]) / (p - 1)
    return Y


def circulant_Y(L, gens, signs=None):
    """Normalised projector onto the top eigenspace of the signed circulant adjacency."""
    signs = [-1.0] * len(gens) if signs is None else signs
    A = np.zeros((L, L))
    for g, s in zip(gens, signs):
        for v in range(L):
            A[v, (v + g) % L] += s; A[(v + g) % L, v] += s
    lam, V = np.linalg.eigh(A)
    top = lam[-1]
    idx = np.where(lam > top - 1e-8)[0]
    P = V[:, idx] @ V[:, idx].T
    return P / P[0, 0], len(idx)


def report(name, Y, extra=""):
    c, lam, grid = best_scale(Y)
    tag = "PSD -> SoS4 = SoS2" if lam > -1e-9 else "not PSD"
    print(f"{name:28s} n={Y.shape[0]:3d} best c={c:.2f} lambda_min={lam:+.3e}  {tag} {extra}", flush=True)
    return {"name": name, "n": int(Y.shape[0]), "c": float(c), "lambda_min": float(lam), "psd": bool(lam > -1e-9),
            "grid": [(float(a), float(b)) for a, b in grid]}


if __name__ == "__main__":
    rows = []
    for p in (13, 17, 29, 37, 41, 53):
        rows.append(report(f"Paley p={p}", paley_Y(p)))
    for L, gens in ((41, [1, 4, 10, 16, 18]), (17, [6, 7]), (9, [1, 2]), (37, [1, 10, 11])):
        Y, mult = circulant_Y(L, gens)
        rows.append(report(f"Cay(Z{L},{gens})", Y, f"(top mult {mult})"))
    json.dump(rows, open("../results/wick_lift.json", "w"), indent=1)
