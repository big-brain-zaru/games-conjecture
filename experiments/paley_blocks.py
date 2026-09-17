"""Exact block structure of the Paley degree-4 form Q, and what field it lives in.

Q is the Aut-equivariant psd operator on Sym(E_min) with
M_even[{a,b},{c,d}] = <Phi(a,b), Q Phi(c,d)>.  E_min is spanned by the additive
characters psi_t at NON-RESIDUE frequencies, and a translation multiplies
psi_t * psi_s by e(-(t+s)b/p), so Q preserves each frequency block

    V_f = span{ psi_t * psi_s : t, s non-residues, t + s = f }.

The representation is real and -1 is a residue (p = 1 mod 4), so f and -f lie in
the same class and the REAL invariant blocks are V_f (+) V_{-f} for f nonzero,
of dimension 2 dim V_f.  (The previous script sliced a 2d-dimensional real block
with a d-dimensional cut, which is why the "spread across f" was large; that was
a bug in the extraction, not in the structure.)

Dilation by a square carries V_f to V_{af} and acts transitively on residues and
on non-residues, so there are exactly three classes: f = 0, f a residue, f a
non-residue.  At p = 29 their dimensions are 7, 4, 3 and 7 + 14*(4+3) = 105.

The f = 0 block is special and is the key to the arithmetic.  Its basis is
{ psi_t * psi_{-t} } indexed by the (p-1)/4 pairs {t,-t} of non-residues, and the
square dilations act on those pairs transitively with stabiliser {1,-1}.  So

    V_0  =  the permutation module of C_{(p-1)/2} on (p-1)/4 points
         =  the sum of the (p-1)/4 characters of C_{(p-1)/4}.

Q therefore acts on V_0 with one eigenvalue per character of C_{(p-1)/4}, which
means its eigenvalues there are CYCLOTOMIC of order (p-1)/4, not elements of
Q(sqrt p).  At p = 29 that is order 7, giving a rational eigenvalue on the
trivial character and a Galois orbit of degree 3 on the rest.

That is the explanation of the failed denominator search: the extension does not
live in Q(sqrt p) at all, it lives in Q(sqrt p, zeta_{(p-1)/4}).

Run: python paley_blocks.py [p]
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

    def perm_matrix(a, b):
        Pm = np.zeros((p, p))
        for z in range(p):
            Pm[(a * z + b) % p, z] = 1.0
        return Pm

    Rho = []
    for b in range(p):
        Rb = B.T @ perm_matrix(1, b) @ B
        Mb = np.zeros((D, D))
        for l in range(D):
            S = SB[l].reshape(m, m)
            Mb[:, l] = SB @ (Rb @ S @ Rb.T).ravel()
        Rho.append(Mb)

    chi = np.zeros(p, dtype=int)
    qr = set(pow(a, 2, p) for a in range(1, p))
    for k in range(1, p):
        chi[k] = 1 if k in qr else -1

    print(f"p = {p}: m = {m}, Sym dimension {D}, (p-1)/4 = {(p-1)//4}")

    def block_spectrum(f):
        Pi = sum(np.exp(2j * np.pi * f * b / p) * Rho[b] for b in range(p)) / p
        R = np.real(Pi) if f == 0 else 2 * np.real(Pi)     # real invariant projector
        R = (R + R.T) / 2
        w, V = np.linalg.eigh(R)
        sel = V[:, w > 0.5]
        sub = sel.T @ Q @ sel
        ev = np.linalg.eigvalsh((sub + sub.T) / 2)
        return sel.shape[1], ev

    d0, ev0 = block_spectrum(0)
    print(f"\n  V_0 : real dimension {d0}")
    print("    eigenvalues:", ", ".join(f"{x:+.9f}" for x in ev0))

    res_f = [f for f in range(1, p) if chi[f] == 1]
    non_f = [f for f in range(1, p) if chi[f] == -1]
    dr, evr = block_spectrum(res_f[0])
    dn, evn = block_spectrum(non_f[0])
    # each eigenvalue of Q|V_f appears twice in the real block V_f (+) V_{-f}
    evr_h = evr[::2]
    evn_h = evn[::2]
    print(f"\n  V_f (+) V_-f, f residue : real dimension {dr} (so dim V_f = {dr//2})")
    print("    eigenvalues of Q|V_f:", ", ".join(f"{x:+.9f}" for x in evr_h))
    print(f"  V_f (+) V_-f, f non-res : real dimension {dn} (so dim V_f = {dn//2})")
    print("    eigenvalues of Q|V_f:", ", ".join(f"{x:+.9f}" for x in evn_h))

    # consistency across the class
    spread_r = max(float(np.abs(block_spectrum(f)[1][::2] - evr_h).max()) for f in res_f[:6])
    spread_n = max(float(np.abs(block_spectrum(f)[1][::2] - evn_h).max()) for f in non_f[:6])
    print(f"\n  spectrum constant across the residue class     : spread {spread_r:.2e}")
    print(f"  spectrum constant across the non-residue class : spread {spread_n:.2e}")

    ker = int((np.abs(ev0) < 1e-7).sum()) + \
        (p - 1) // 2 * int((np.abs(evr_h) < 1e-7).sum()) + \
        (p - 1) // 2 * int((np.abs(evn_h) < 1e-7).sum())
    print(f"  kernel accounted for: {ker}   (2m = {2*m})")

    allv = np.concatenate([ev0, evr_h, evn_h])
    nzu = np.unique(np.round(allv[np.abs(allv) > 1e-7], 6))
    evQ = np.linalg.eigvalsh(Q)
    print(f"  distinct nonzero eigenvalues from blocks {len(nzu)}, "
          f"from Q directly {len(np.unique(np.round(evQ[np.abs(evQ)>1e-7],6)))}")

    # rationality: the nonzero f-block eigenvalues look like k/(3p)
    print("\n  rational test on the nonzero f-block eigenvalues (times 3p):")
    for lab, arr in (("residue", evr_h), ("non-residue", evn_h)):
        for x in arr:
            if abs(x) < 1e-7:
                continue
            t = x * 3 * p
            print(f"    {lab:11s} {x:+.9f} * 3p = {t:.6f}   "
                  f"{'INTEGER ' + str(int(round(t))) if abs(t - round(t)) < 1e-6 else 'not an integer'}")
    print("\n  V_0 nonzero eigenvalues: symmetric functions (should be rational if")
    print("  they form one Galois orbit plus the trivial character):")
    nz0 = ev0[np.abs(ev0) > 1e-7]
    uniq0 = np.unique(np.round(nz0, 7))
    print(f"    distinct: {', '.join(f'{x:+.9f}' for x in uniq0)}")
    orbit = np.array([x for x in uniq0 if abs(x - m) > 1e-6])
    if len(orbit):
        e1, e2, e3 = orbit.sum(), float(np.sum([orbit[i]*orbit[j] for i in range(len(orbit))
                                                for j in range(i+1, len(orbit))])), float(np.prod(orbit))
        print(f"    trivial character eigenvalue = {m} (= m)" if abs(uniq0.max() - m) < 1e-6 else "")
        print(f"    orbit of size {len(orbit)}: e1 = {e1:.9f}, e2 = {e2:.9f}, e3 = {e3:.9f}")
        for nm, val in (("e1", e1), ("e2", e2), ("e3", e3)):
            t = val * 3 * p
            print(f"      {nm} * 3p = {t:.6f}  "
                  f"{'INTEGER ' + str(int(round(t))) if abs(t - round(t)) < 1e-5 else ''}")

    out = dict(p=p, m=m, D=D, dim_V0=int(d0), dim_Vres=int(dr // 2), dim_Vnon=int(dn // 2),
               spec_V0=[float(x) for x in ev0],
               spec_Vres=[float(x) for x in evr_h], spec_Vnon=[float(x) for x in evn_h],
               spread_res=spread_r, spread_non=spread_n, kernel_accounted=ker)
    dst = os.path.join(ROOT, "results", f"paley_blocks_{p}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
