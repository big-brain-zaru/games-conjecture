"""Does the unique Paley degree-4 extension come from a BILINEAR map?

From paley_exact/paley_rigid at p = 29: M_even is 407 x 407, psd, of rank 77,
and the q realising it is unique.  Being psd of rank 77 means there are vectors

    u_empty, u_e  in R^77   with   M[a,b] = <u_a, u_b>.

The forced entries say more than that.  For pairs sharing exactly one point,
<u_{xy}, u_{xl}> = X_{yl} = <v_y, v_l>, where v_x is the unit vector of the
degree-2 solution inside the lambda_min eigenspace E_min (dimension m = 14).  So
for each fixed x the map v_y -> u_{xy} preserves all inner products: it is an
ISOMETRY T_x of E_min into R^77, with

    u_{xy} = T_x v_y = T_y v_x    and    T_x v_x = u_empty  for every x.

(The last identity follows from <u_empty, u_{xy}> = X_{xy} plus equality in
Cauchy-Schwarz.)  The natural guess is then that u_{xy} depends on x and y only
through the symmetric product v_x * v_y, i.e. that there is a single linear map

    Psi : Sym^2(E_min) -> R^77    with    u_{xy} = Psi(v_x * v_y).

Sym^2(E_min) has dimension m(m+1)/2 = 105 and the rank is 77, so Psi would have
a 28-dimensional kernel, and 28 = p - 1 = 2m exactly.  If this holds, finding the
formula reduces from 65 unknown numbers to identifying ONE map, or equivalently
its 28-dimensional kernel -- a much smaller and much more structured object.

This script tests it by least squares, and reports the residual against the
scale of u.  It also checks the two structural predictions separately:
Psi(v_x * v_x) must be the SAME vector for every x, and the fit must reproduce
the forced entries.

Run: python paley_bilinear.py [p]
"""
import itertools
import json
import os
import sys

import numpy as np

from paley_exact import build_parts, index_pairs, legendre

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def emin_basis(p):
    """Orthonormal basis of the lambda_min eigenspace, and the unit vectors v_x."""
    chi = legendre(p)
    S = np.array([[chi[(i - j) % p] for j in range(p)] for i in range(p)], dtype=float)
    P = 0.5 * (np.eye(p) - np.ones((p, p)) / p - S / np.sqrt(p))
    w, V = np.linalg.eigh(P)
    m = (p - 1) // 2
    B = V[:, -m:]                       # eigenvalue 1 block
    v = B.T / np.linalg.norm(B.T, axis=0)   # columns: v_x in coordinates, unit norm
    return B, v.T, m                    # v[x] is the x-th unit vector (m,)


def sym_coords(a, b, m, idx):
    """Coordinates of the symmetric product a*b in Sym^2(R^m)."""
    out = np.empty(len(idx))
    for k, (i, j) in enumerate(idx):
        out[k] = a[i] * b[j] + a[j] * b[i] if i != j else a[i] * b[i]
    return out


def main():
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    src = os.path.join(ROOT, "results", f"paley_rigid_{p}.json")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run paley_rigid.py {p} first")
    q = np.array(json.load(open(src, encoding="utf-8"))["q_solved"])

    M0, A, reps, X, pairs, pidx = build_parts(p)
    M = M0 + sum(q[k] * A[k] for k in range(len(reps)))
    ev, EV = np.linalg.eigh(M)
    r = int((ev > 1e-7).sum())
    U = (EV[:, -r:] * np.sqrt(ev[-r:])).T            # r x n, columns are u_a
    n = M.shape[0]
    print(f"p = {p}: rank {r}, reconstruction error "
          f"{np.abs(U.T @ U - M).max():.2e}")

    B, v, m = emin_basis(p)
    print(f"  E_min dimension m = {m}, Sym^2 dimension {m*(m+1)//2}, rank {r}, "
          f"predicted kernel {m*(m+1)//2 - r} (2m = {2*m}, p-1 = {p-1})")

    # check the degree-2 solution is reproduced by these v's
    G = v @ v.T
    print(f"  Gram(v) vs X               : max dev {np.abs(G - X).max():.2e}")

    idx = [(i, j) for i in range(m) for j in range(i, m)]
    D = len(idx)

    # least squares for Psi: rows are sym coords, targets are u
    rowsS, rowsU = [], []
    for e in pairs:
        x, y = e
        rowsS.append(sym_coords(v[x], v[y], m, idx))
        rowsU.append(U[:, pidx[frozenset(e)]])
    Smat = np.stack(rowsS)            # (#pairs, D)
    Umat = np.stack(rowsU)            # (#pairs, r)
    Psi, res, rk, sv = np.linalg.lstsq(Smat, Umat, rcond=None)
    pred = Smat @ Psi
    err = float(np.abs(pred - Umat).max())
    scale = float(np.abs(Umat).max())
    print(f"\n  least squares Psi: Sym^2 -> R^{r}")
    print(f"    design rank {rk} of {D}")
    print(f"    max |Psi(v_x*v_y) - u_xy| : {err:.3e}   (u scale {scale:.3f})")

    # structural prediction: Psi(v_x * v_x) must be the same vector for all x
    diag = np.stack([sym_coords(v[x], v[x], m, idx) for x in range(p)]) @ Psi
    spread = float(np.abs(diag - diag.mean(axis=0)).max())
    uempty = U[:, 0]
    print(f"    spread of Psi(v_x*v_x) over x : {spread:.3e}")
    print(f"    |mean Psi(v_x*v_x) - u_empty| : {float(np.abs(diag.mean(axis=0) - uempty).max()):.3e}")

    ok = err < 1e-6 * max(scale, 1.0)
    print()
    if ok:
        print("VERDICT: the extension IS bilinear. Every moment is Psi(v_x * v_y) for one")
        print(f"linear map Psi on Sym^2(E_min), whose kernel has dimension {D - rk}.")
        print("Identifying that kernel identifies the whole degree-4 extension.")
    else:
        print("VERDICT: the extension is NOT of the form Psi(v_x * v_y). The vectors u_xy")
        print("are isometric images T_x v_y, but the T_x do not assemble into a single")
        print("bilinear map, so no formula of that shape can exist. This rules out the")
        print("entire Wick / tensor-square family, including the MRX shape, at finite p.")

    out = dict(p=p, rank=r, m=m, sym_dim=D, design_rank=int(rk), max_error=err,
               u_scale=scale, diag_spread=spread, bilinear=bool(ok))
    dst = os.path.join(ROOT, "results", f"paley_bilinear_{p}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
