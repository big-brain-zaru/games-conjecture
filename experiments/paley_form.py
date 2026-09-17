"""The Paley degree-4 extension as a single quadratic form, and its spectrum.

paley_bilinear.py showed that the unique degree-4 extension at p = 29 satisfies
u_{xy} = Psi(v_x * v_y) for a single linear map Psi on Sym^2(E_min).  Therefore
there is one symmetric psd operator Q = Psi^T Psi on Sym(m), m = (p-1)/2, with

    M_even[{a,b},{c,d}] = < Phi(a,b), Q Phi(c,d) >,     Phi(a,b) := (v_a v_b^T + v_b v_a^T)/2,

and the empty index corresponds to Phi(x,x) = v_x v_x^T, which must give the same
vector for every x.  With the Frobenius inner product,

    < Phi(a,b), Phi(c,d) > = ( X_ac X_bd + X_ad X_bc ) / 2,

which is exactly the WICK (Gaussian) lift.  So Q measures precisely how the true
extension differs from the Wick lift that `wick_lift.py` found to be badly not
psd.  Two things are worth knowing about Q:

  * its SPECTRUM.  If Q has only two distinct eigenvalues, Q = c (I - Pi) for an
    orthogonal projection Pi, and then

        q(S) = c [ Wick(S) - < Phi(a,b), Pi Phi(c,d) > ],

    i.e. the extension is the Wick lift with its component in a single subspace
    deleted, rescaled.  That is a closed form.
  * its KERNEL, of dimension 105 - 77 = 28 = 2m at p = 29.  Since E_min is an
    irreducible Aut-module of dimension m, a 2m-dimensional kernel is exactly two
    irreducible pieces, which is the kind of statement that generalises in p.

This script computes Q, prints its spectrum, and tests both.

Run: python paley_form.py [p]
"""
import itertools
import json
import os
import sys

import numpy as np

from paley_bilinear import emin_basis
from paley_exact import build_parts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sym_basis(m):
    """Orthonormal basis of Sym(m) under the Frobenius inner product."""
    B = []
    for i in range(m):
        E = np.zeros((m, m)); E[i, i] = 1.0
        B.append(E)
    for i in range(m):
        for j in range(i + 1, m):
            E = np.zeros((m, m)); E[i, j] = E[j, i] = 1 / np.sqrt(2)
            B.append(E)
    return np.stack([E.ravel() for E in B])       # (D, m*m)


def main():
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    src = os.path.join(ROOT, "results", f"paley_rigid_{p}.json")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run paley_rigid.py {p} first")
    q = np.array(json.load(open(src, encoding="utf-8"))["q_solved"])

    M0, A, reps, X, pairs, pidx = build_parts(p)
    M = M0 + sum(q[k] * A[k] for k in range(len(reps)))
    B, v, m = emin_basis(p)
    SB = sym_basis(m)
    D = SB.shape[0]

    # Phi(a,b) in the orthonormal Sym basis
    def phi(a, b):
        E = (np.outer(v[a], v[b]) + np.outer(v[b], v[a])) / 2
        return SB @ E.ravel()

    F = np.stack([phi(*e) for e in pairs])              # (#pairs, D)
    N = M[1:, 1:]                                       # pairs block
    Fp = np.linalg.pinv(F)
    Q = Fp @ N @ Fp.T
    Q = (Q + Q.T) / 2
    resid = float(np.abs(F @ Q @ F.T - N).max())
    print(f"p = {p}: Sym(m) dimension D = {D}, m = {m}")
    print(f"  fit of M_even by the form Q : max residual {resid:.3e}")

    # the empty index must be Phi(x,x), same for all x
    phid = np.stack([phi(x, x) for x in range(p)])
    e_from_diag = phid @ Q @ F.T                        # should equal M[0, pairs]
    print(f"  empty-row reproduced        : max dev "
          f"{float(np.abs(e_from_diag - M[0,1:]).max()):.3e}")
    print(f"  <Phi(x,x),Q Phi(y,y)> = 1   : max dev "
          f"{float(np.abs(phid @ Q @ phid.T - 1).max()):.3e}")

    ev = np.linalg.eigvalsh(Q)
    nz = ev[np.abs(ev) > 1e-8]
    print(f"\n  spectrum of Q: {len(ev)} eigenvalues, {D - len(nz)} zero, "
          f"{len(nz)} nonzero")
    uniq = np.unique(np.round(nz, 6))
    print(f"  distinct nonzero eigenvalues: {len(uniq)}")
    print("   ", ", ".join(f"{x:+.6f}" for x in uniq[:12]),
          "..." if len(uniq) > 12 else "")
    print(f"  kernel dimension {D - len(nz)}  (2m = {2*m}, m = {m}, p-1 = {p-1})")

    # Wick comparison: Q = identity would BE the Wick lift
    wick_gap = float(np.abs(Q - np.eye(D)).max())
    print(f"  |Q - I| (0 would mean the extension is exactly Wick): {wick_gap:.3f}")

    verdict = None
    if len(uniq) == 1:
        c = float(uniq[0])
        print(f"\nVERDICT: Q has ONE nonzero eigenvalue {c:.6f}, so Q = c (I - Pi) with Pi")
        print(f"the orthogonal projection onto a {D - len(nz)}-dimensional subspace.")
        print("The extension is therefore CLOSED FORM: the Wick lift, with its component")
        print("in that subspace removed, scaled by c. Identifying the subspace finishes it.")
        verdict = "projection"
    else:
        print(f"\nVERDICT: Q has {len(uniq)} distinct nonzero eigenvalues, so it is not a")
        print("scaled projection. The extension is still one quadratic form on Sym(E_min),")
        print("which is a large reduction, but the form has genuine spectral structure.")
        print("Next: decompose Q into Aut-isotypic blocks; equivariance forces it to be")
        print("constant on each irreducible, so the distinct eigenvalues should be few and")
        print("should match the isotypic pattern.")
        verdict = "general"

    out = dict(p=p, m=m, D=D, fit_residual=resid, kernel_dim=int(D - len(nz)),
               n_distinct_nonzero=int(len(uniq)),
               eigenvalues=[float(x) for x in ev], verdict=verdict,
               wick_distance=wick_gap)
    dst = os.path.join(ROOT, "results", f"paley_form_{p}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
