"""Where does the freedom in the Paley degree-4 extension live?

Measured family dimensions: 0 at p = 29, 2 at p = 37, 3 at p = 41, matching
dim V_0 - 7 = (p-29)/4 at all three (the p = 41 value was predicted before the
run).  The "-7" wants an explanation, and the natural guess is that the freedom
sits entirely in one frequency class.

A null direction dq of the kernel conditions changes the moment matrix by
dM = sum_k dq_k A_k and hence changes the form by dQ = F^+ dM F^+T.  Splitting
dQ across the three classes V_0, V_res, V_non says which block carries the
freedom.  If the whole family lives in V_0, then the residue and non-residue
blocks of Q are rigid and the count reduces to a circulant of size (p-1)/4
subject to a fixed number of conditions.

Run: python paley_family.py [p]
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
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 41
    rec = json.load(open(os.path.join(ROOT, "results", f"paley_exact_{p}.json"),
                         encoding="utf-8"))
    q0 = np.array(rec["q"])
    M0, A, reps, X, pairs, pidx = build_parts(p)
    K = len(reps)
    M = M0 + sum(q0[k] * A[k] for k in range(K))
    w, V = np.linalg.eigh(M)
    Wk = V[:, np.abs(w) < 1e-7]
    B = np.stack([(A[k] @ Wk).ravel() for k in range(K)], axis=1)
    G = B.T @ B                            # K x K, avoids a huge SVD
    ge, gv = np.linalg.eigh(G)
    sv = np.sqrt(np.clip(ge, 0, None))[::-1]
    Vt = gv[:, ::-1].T
    nfam = int((sv < 1e-6 * sv[0]).sum())
    print(f"p = {p}: {K} unknowns, family dimension {nfam}  "
          f"(dim V_0 - 7 = {(p-1)//4 - 7})")
    if nfam == 0:
        print("  unique extension; nothing to decompose")
        return
    null = Vt[-nfam:]                      # (nfam, K)

    B_, v, m = emin_basis(p)
    SB = sym_basis(m)
    D = SB.shape[0]

    def phi(a, b):
        E = (np.outer(v[a], v[b]) + np.outer(v[b], v[a])) / 2
        return SB @ E.ravel()

    F = np.stack([phi(*e) for e in pairs])
    Fp = np.linalg.pinv(F)

    def perm_matrix(a, b):
        Pm = np.zeros((p, p))
        for z in range(p):
            Pm[(a * z + b) % p, z] = 1.0
        return Pm

    Rho = []
    for b in range(p):
        Rb = B_.T @ perm_matrix(1, b) @ B_
        Mb = np.zeros((D, D))
        for l in range(D):
            S = SB[l].reshape(m, m)
            Mb[:, l] = SB @ (Rb @ S @ Rb.T).ravel()
        Rho.append(Mb)

    chi = np.zeros(p, dtype=int)
    qr = set(pow(a, 2, p) for a in range(1, p))
    for k in range(1, p):
        chi[k] = 1 if k in qr else -1

    def projector(f):
        Pi = sum(np.exp(2j * np.pi * f * b / p) * Rho[b] for b in range(p)) / p
        R = np.real(Pi) if f == 0 else 2 * np.real(Pi)
        return (R + R.T) / 2

    P0 = projector(0)
    Pres = sum(projector(f) for f in range(1, p) if chi[f] == 1) / 2
    Pnon = sum(projector(f) for f in range(1, p) if chi[f] == -1) / 2
    print(f"  projector traces: V_0 {np.trace(P0):.1f}, V_res {np.trace(Pres):.1f}, "
          f"V_non {np.trace(Pnon):.1f}  (sum {np.trace(P0)+np.trace(Pres)+np.trace(Pnon):.1f}, D = {D})")

    print("\n  how each null direction moves Q, by frequency class:")
    print(f"    {'dir':>4} {'|dQ|':>10} {'in V_0':>10} {'in V_res':>10} {'in V_non':>10}")
    tot = np.zeros(3)
    for i, dq in enumerate(null):
        dM = sum(dq[k] * A[k] for k in range(K))
        dQ = Fp @ dM[1:, 1:] @ Fp.T
        dQ = (dQ + dQ.T) / 2
        n_all = float(np.linalg.norm(dQ))
        parts = [float(np.linalg.norm(P @ dQ @ P)) for P in (P0, Pres, Pnon)]
        tot += np.array(parts) ** 2
        print(f"    {i:>4} {n_all:10.3e} " + " ".join(f"{x/n_all:10.4f}" for x in parts))
    tot = np.sqrt(tot / tot.sum())
    print(f"\n  overall share of the family: V_0 {tot[0]:.4f}, V_res {tot[1]:.4f}, "
          f"V_non {tot[2]:.4f}")
    print()
    if tot[0] > 0.999:
        print("VERDICT: the family lives ENTIRELY in the zero-frequency block. The residue")
        print("and non-residue blocks of Q are rigid, and the freedom is a circulant of")
        print(f"size (p-1)/4 = {(p-1)//4} constrained down to {nfam} dimensions.")
    else:
        print("VERDICT: the freedom is spread across frequency classes, so the -7 in")
        print("dim V_0 - 7 is not simply 'the V_0 circulant minus fixed conditions'.")

    out = dict(p=p, family_dim=nfam, predicted=(p - 1) // 4 - 7,
               share_V0=float(tot[0]), share_Vres=float(tot[1]), share_Vnon=float(tot[2]))
    dst = os.path.join(ROOT, "results", f"paley_family_{p}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
