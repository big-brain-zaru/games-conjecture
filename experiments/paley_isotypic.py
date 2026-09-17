"""Decompose the Paley degree-4 form Q into Aut-isotypic blocks.

paley_form.py produced the single psd operator Q on Sym(E_min), m = (p-1)/2,
with M_even[{a,b},{c,d}] = <Phi(a,b), Q Phi(c,d)>, Phi(a,b) = (v_a v_b^T + v_b v_a^T)/2,
kernel of dimension exactly 2m, and 9 distinct nonzero eigenvalues at p = 29.

Q must be Aut-equivariant.  E_min is spanned by the additive characters psi_t at
NON-RESIDUE frequencies t, and a translation acts on psi_t by e(-tb/p).  Hence on
Sym^2(E_min) a translation acts on psi_t * psi_s by e(-(t+s)b/p), so the
FREQUENCY f = t + s splits the space:

    Sym^2(E_min) = (+)_f V_f,    V_f = span{ psi_t * psi_s : t,s non-residues, t+s = f }.

Q commutes with translations, so Q preserves every V_f.  Dilation by a square
maps V_f to V_{af}, so the blocks come in three classes: f = 0, f a residue, and
f a non-residue, and Q has the SAME spectrum on every V_f within a class.  That
predicts:

  * dim V_0 + (p-1)/2 * (dim V_res + dim V_non) / ... summing to m(m+1)/2;
  * the distinct eigenvalues of Q are those of Q|V_0, Q|V_res and Q|V_non only;
  * the 2m-dimensional kernel is exactly TWO of the 14-dimensional irreducible
    pieces, i.e. two frequency classes' worth of kernel directions.

This script verifies the equivariance, computes the block dimensions and the
per-class spectra, and checks that they reproduce the 9 distinct eigenvalues.
The result is a complete structural description of the extension.

Run: python paley_isotypic.py [p]
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
    src = os.path.join(ROOT, "results", f"paley_rigid_{p}.json")
    q = np.array(json.load(open(src, encoding="utf-8"))["q_solved"])
    M0, A, reps, X, pairs, pidx = build_parts(p)
    M = M0 + sum(q[k] * A[k] for k in range(len(reps)))
    B, v, m = emin_basis(p)
    SB = sym_basis(m)
    D = SB.shape[0]

    def phi(a, b):
        E = (np.outer(v[a], v[b]) + np.outer(v[b], v[a])) / 2
        return SB @ E.ravel()

    F = np.stack([phi(*e) for e in pairs])
    Fp = np.linalg.pinv(F)
    Q = Fp @ M[1:, 1:] @ Fp.T
    Q = (Q + Q.T) / 2
    print(f"p = {p}: m = {m}, Sym dimension D = {D}")

    # translation action on E_min, then on Sym(E_min)
    def perm_matrix(a, b):
        Pm = np.zeros((p, p))
        for z in range(p):
            Pm[(a * z + b) % p, z] = 1.0
        return Pm

    Bm = B                                   # p x m orthonormal basis of E_min
    Rho = []
    for b in range(p):
        Rb = Bm.T @ perm_matrix(1, b) @ Bm   # m x m
        Mb = np.zeros((D, D))
        for l in range(D):
            S = SB[l].reshape(m, m)
            Mb[:, l] = SB @ (Rb @ S @ Rb.T).ravel()
        Rho.append(Mb)
    comm = max(float(np.abs(Q @ Rb - Rb @ Q).max()) for Rb in Rho)
    print(f"  Q commutes with translations : max |[Q,rho]| = {comm:.3e}")

    # frequency projectors
    chi = np.zeros(p, dtype=int)
    qr = set(pow(a, 2, p) for a in range(1, p))
    for k in range(1, p):
        chi[k] = 1 if k in qr else -1

    dims, specs = {}, {}
    for f in range(p):
        Pi = sum(np.exp(2j * np.pi * f * b / p) * Rho[b] for b in range(p)) / p
        d = float(np.real(np.trace(Pi)))
        dims[f] = d
        if round(d) > 0:
            # eigenvalues of Q restricted to the block
            w, V = np.linalg.eigh((Pi + Pi.conj().T).real / 2)
            k = int(round(d))
            basis = V[:, np.argsort(w)[-k:]] if k > 0 else V[:, :0]
            if basis.shape[1]:
                sub = basis.T @ Q @ basis
                specs[f] = np.linalg.eigvalsh((sub + sub.T) / 2)

    d0 = dims[0]
    dres = [dims[f] for f in range(1, p) if chi[f] == 1]
    dnon = [dims[f] for f in range(1, p) if chi[f] == -1]
    print(f"\n  dim V_0            = {d0:.3f}")
    print(f"  dim V_f, f residue = {np.mean(dres):.3f}  (spread {np.ptp(dres):.1e})")
    print(f"  dim V_f, f non-res = {np.mean(dnon):.3f}  (spread {np.ptp(dnon):.1e})")
    print(f"  check  dim V_0 + {(p-1)//2}*(dres + dnon) = "
          f"{d0 + (p-1)//2*(np.mean(dres)+np.mean(dnon)):.1f}  (D = {D})")

    def show(label, fs):
        if not fs:
            return None
        got = [specs[f] for f in fs if f in specs]
        L = max(len(a) for a in got)
        got = [a for a in got if len(a) == L]
        arr = np.stack(got)
        mean = arr.mean(axis=0)
        print(f"  spectrum of Q on V_f, {label:9s}: " +
              ", ".join(f"{x:+.6f}" for x in mean) +
              f"   (spread across f: {float(np.abs(arr - mean).max()):.1e})")
        return mean

    print()
    s0 = show("f = 0", [0])
    sres = show("f residue", [f for f in range(1, p) if chi[f] == 1])
    snon = show("f non-res", [f for f in range(1, p) if chi[f] == -1])

    allv = np.concatenate([x for x in (s0, sres, snon) if x is not None])
    nz = allv[np.abs(allv) > 1e-8]
    print(f"\n  distinct eigenvalues predicted by the blocks: "
          f"{len(np.unique(np.round(nz,6)))} nonzero, "
          f"{int((np.abs(allv) <= 1e-8).sum())} zero classes")
    evQ = np.linalg.eigvalsh(Q)
    print(f"  distinct nonzero eigenvalues of Q directly  : "
          f"{len(np.unique(np.round(evQ[np.abs(evQ)>1e-8],6)))}")
    zero_classes = [("f=0", s0), ("f residue", sres), ("f non-residue", snon)]
    for lab, s in zero_classes:
        if s is not None:
            k = int((np.abs(s) <= 1e-8).sum())
            if k:
                print(f"  kernel sits in the {lab} class: {k} direction(s) per f, "
                      f"total {k * (1 if lab=='f=0' else (p-1)//2)}")

    out = dict(p=p, m=m, D=D, commutator=comm, dim_V0=d0,
               dim_Vres=float(np.mean(dres)), dim_Vnon=float(np.mean(dnon)),
               spec_V0=[float(x) for x in s0] if s0 is not None else [],
               spec_Vres=[float(x) for x in sres] if sres is not None else [],
               spec_Vnon=[float(x) for x in snon] if snon is not None else [])
    dst = os.path.join(ROOT, "results", f"paley_isotypic_{p}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
