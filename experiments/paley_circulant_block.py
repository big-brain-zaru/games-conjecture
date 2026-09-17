"""The f = 0 block of the Paley degree-4 form, as a circulant over C_{(p-1)/4}.

paley_blocks.py showed the f = 0 block V_0 has dimension (p-1)/4 and carries the
permutation module of the square dilations on the pairs {t, -t} of non-residues,
which is the regular representation of C_{(p-1)/4}.  So Q restricted to V_0 is a
CIRCULANT of size (p-1)/4, and its eigenvalues are the discrete Fourier transform
of (p-1)/4 coefficients.  That is why those eigenvalues are cyclotomic of order
(p-1)/4 rather than elements of Q(sqrt p): they are a DFT, and the coefficients,
not the eigenvalues, are the natural objects.

Basis: for each pair {t,-t} with t a non-residue, the real symmetric matrix
S_t = Re(a_t a_t^H), where a_t are the coordinates of the additive character
psi_t inside the lambda_min eigenspace.  Dilation by a square sends S_t to S_{at},
so the (p-1)/4 basis vectors are permuted in a single cycle.

This script extracts the circulant coefficients c_0 .. c_{(p-5)/4} and tests them
for rationality, and checks that their DFT reproduces the measured eigenvalues.

Run: python paley_circulant_block.py [p]
"""
import json
import os
import sys

import numpy as np

from paley_bilinear import emin_basis
from paley_exact import build_parts
from paley_form import sym_basis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    q = np.array(json.load(open(os.path.join(ROOT, "results", f"paley_rigid_{p}.json"),
                                encoding="utf-8"))["q_solved"])
    M0, A, reps, X, pairs, pidx = build_parts(p)
    M = M0 + sum(q[k] * A[k] for k in range(len(reps)))
    B, v, m = emin_basis(p)
    SB = sym_basis(m)

    def phi(a, b):
        E = (np.outer(v[a], v[b]) + np.outer(v[b], v[a])) / 2
        return SB @ E.ravel()

    F = np.stack([phi(*e) for e in pairs])
    Fp = np.linalg.pinv(F)
    Q = Fp @ M[1:, 1:] @ Fp.T
    Q = (Q + Q.T) / 2

    # additive characters inside E_min, as coordinate vectors a_t in C^m
    qr = set(pow(a, 2, p) for a in range(1, p))
    non = [t for t in range(1, p) if t not in qr]
    a_t = {}
    for t in non:
        psi = np.exp(2j * np.pi * t * np.arange(p) / p) / np.sqrt(p)
        a_t[t] = B.T @ psi                      # B real, so coordinates in C^m

    # pairs {t,-t}; a generator of the squares acts on them with one orbit
    seen, prs = set(), []
    for t in non:
        if t in seen:
            continue
        seen.add(t); seen.add((-t) % p)
        prs.append(t)
    k = len(prs)
    print(f"p = {p}: m = {m}, (p-1)/4 = {(p-1)//4}, pairs found {k}")

    g = None
    for cand in range(2, p):
        if cand in qr and all(pow(cand, j, p) != 1 for j in range(1, (p - 1) // 2)):
            g = cand
            break
    print(f"  generator of the squares: {g} (order {(p-1)//2})")

    # order the pairs along the orbit of g
    def rep_of(t):
        return t if t in prs else (-t) % p
    order = [prs[0]]
    while len(order) < k:
        nxt = rep_of((g * order[-1]) % p)
        if nxt == order[0]:
            break
        order.append(nxt)
    print(f"  orbit length under the generator: {len(order)} (need {k})")
    if len(order) != k:
        print("  the squares do not act with a single cycle here; aborting")
        return

    S = []
    for t in order:
        aa = np.outer(a_t[t], np.conj(a_t[t]))
        E = np.real((aa + aa.conj().T) / 2)
        S.append(SB @ E.ravel())
    S = np.stack(S)                              # k x D
    G = S @ S.T
    QS = S @ Q @ S.T
    # Q S_s = sum_t c_{t-s} S_t  ->  in the Gram picture,  QS = G C  with C circulant
    C = np.linalg.solve(G, QS)
    print(f"  Gram circulant check  : {float(np.abs(G - np.array([[G[0,(j-i)%k] for j in range(k)] for i in range(k)])).max()):.2e}")
    print(f"  C circulant check     : {float(np.abs(C - np.array([[C[0,(j-i)%k] for j in range(k)] for i in range(k)])).max()):.2e}")
    c = C[0]
    print(f"\n  circulant coefficients c_j of Q on V_0:")
    for j, cj in enumerate(c):
        t = cj * 3 * p
        tag = f"= {int(round(t))}/(3p)" if abs(t - round(t)) < 1e-4 else ""
        t2 = cj * p
        tag2 = f"= {int(round(t2))}/p" if abs(t2 - round(t2)) < 1e-4 else ""
        print(f"    c_{j} = {cj:+.9f}   {tag} {tag2}")

    ev = np.sort(np.real(np.fft.fft(c)))
    print(f"\n  DFT of c (should be the V_0 spectrum):")
    print("   ", ", ".join(f"{x:+.9f}" for x in ev))
    meas = np.array(json.load(open(os.path.join(ROOT, "results", f"paley_blocks_{p}.json"),
                                   encoding="utf-8"))["spec_V0"])
    print(f"  measured V_0 spectrum (deduplicated):")
    print("   ", ", ".join(f"{x:+.9f}" for x in np.unique(np.round(meas, 7))))

    out = dict(p=p, k=k, generator=g, order=[int(t) for t in order],
               coefficients=[float(x) for x in c],
               dft=[float(x) for x in ev])
    dst = os.path.join(ROOT, "results", f"paley_circulant_block_{p}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
