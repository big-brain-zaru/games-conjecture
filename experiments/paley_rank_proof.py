"""PROOF of the rank formula, with every step verified numerically.

CLAIM.  Let p = 1 mod 4, m = (p-1)/2, E_min the lambda_min eigenspace of the Paley
graph, spanned by the additive characters psi_t at non-residue frequencies.  If a
degree-4 extension exists and is bilinear -- i.e. u_{ab} = Psi(v_a * v_b) for a
linear Psi on Sym^2(E_min), which is what paley_bilinear.py verifies -- then

    ker Psi  contains  span{ W_f : f in F_p, f nonzero },    dim = p - 1 = 2m,

    hence   rank M_even  <=  dim Sym^2(E_min) - (p-1) = (p^2-1)/8 - (p-1)
                          =  (p-1)(p-7)/8  =  m(m-3)/2.

PROOF.  Two facts, one analytic and one structural.

(1) The unit vectors of the degree-2 optimum are
        v_a = (1/sqrt m) sum_{t in NQR} e(-ta/p) psi_t,
    so their symmetric squares expand over FREQUENCIES f = t + s:
        v_a v_a^T = (1/m) sum_{f in F_p} e(-fa/p) W_f,     W_f := sum_{t+s=f} a_t a_s^T,
    where a_t are the coordinates of psi_t.  The a-dependence enters ONLY through
    the character e(-fa/p).

(2) Psi(v_a v_a^T) = u_empty for EVERY a.  This is forced, not assumed: the empty
    index satisfies <u_empty, u_{ab}> = X_ab and |u_empty| = |v_a| = 1, so equality
    in Cauchy-Schwarz gives u_empty = T_a v_a = Psi(v_a v_a^T) for each a.

Subtracting (2) for two different a and using (1):
        0 = Psi(v_a v_a^T - v_b v_b^T) = (1/m) sum_f (e(-fa/p) - e(-fb/p)) Psi(W_f).
The f = 0 term cancels identically.  Letting a, b range over F_p, the matrix of
coefficients (e(-fa/p) - e(-fb/p)) has rank p - 1 on the nonzero frequencies,
because distinct additive characters are linearly independent.  Hence

        Psi(W_f) = 0   for every f != 0.

Finally W_f != 0 exactly when dim V_f >= 1, and dim V_f is (p-1)/4 at f = 0,
ceil((p-1)/8) at f a residue and floor((p-1)/8) at f a non-residue, all >= 1 once
p >= 13.  So the p - 1 vectors W_f, f != 0, are nonzero, live in distinct
frequency blocks and are therefore linearly independent, giving a kernel of
dimension exactly p - 1 = 2m inside Sym^2(E_min).  Since
dim Sym^2(E_min) = m(m+1)/2 = (p^2-1)/8,

        rank M_even = rank Psi <= (p^2-1)/8 - (p-1) = (p-1)(p-7)/8 = m(m-3)/2.   QED

This also EXPLAINS the measured kernel structure: one kernel direction in each of
the p-1 nonzero frequency blocks, and none at f = 0, which is exactly what
paley_blocks.py found.

WHAT IS NOT PROVED HERE.  The matching lower bound -- that the kernel is no
LARGER than this span, so the inequality is an equality -- is a property of the
particular extension, not a consequence of the constraints.  It is verified at
p = 29, 37, 41, 53, 61.  Bilinearity itself is likewise verified rather than
proved; without it the argument above does not start.

This script checks (1), (2), the independence, the containment in ker Q, and the
exactness, at each p for which a solution is on disk.

Run: python paley_rank_proof.py 29 37 41
"""
import json
import os
import sys

import numpy as np

from paley_bilinear import emin_basis
from paley_exact import build_parts
from paley_form import sym_basis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_q(p):
    for name in (f"paley_rigid_{p}.json", f"paley_exact_{p}.json"):
        f = os.path.join(ROOT, "results", name)
        if os.path.exists(f):
            d = json.load(open(f, encoding="utf-8"))
            return np.array(d.get("q_solved", d.get("q")))
    return None


def verify(p):
    """All the quantities the proof of 13.6 asserts, for one p. Returns a dict."""
    q = load_q(p)
    if q is None:
        return None
    m = (p - 1) // 2
    M0, A, reps, X, pairs, pidx = build_parts(p)
    M = M0 + sum(q[k] * A[k] for k in range(len(reps)))
    B, v, _ = emin_basis(p)
    SB = sym_basis(m)
    D = SB.shape[0]

    def phi(a, b):
        E = (np.outer(v[a], v[b]) + np.outer(v[b], v[a])) / 2
        return SB @ E.ravel()

    F = np.stack([phi(*e) for e in pairs])
    Fp = np.linalg.pinv(F)
    Q = Fp @ M[1:, 1:] @ Fp.T
    Q = (Q + Q.T) / 2

    # coordinates a_t of the additive characters psi_t, t a non-residue
    qr = set(pow(x, 2, p) for x in range(1, p))
    non = [t for t in range(1, p) if t not in qr]
    a_t = {t: B.T @ (np.exp(2j * np.pi * t * np.arange(p) / p) / np.sqrt(p))
           for t in non}

    # W_f = sum_{t+s=f} a_t a_s^T, in the orthonormal Sym basis
    W = {}
    for f in range(p):
        Mx = np.zeros((m, m), dtype=complex)
        for t in non:
            s = (f - t) % p
            if s in a_t:
                Mx += np.outer(a_t[t], a_t[s])
        W[f] = SB @ Mx.ravel()

    # (1) the expansion  v_a v_a^T = (1/m) sum_f e(-fa/p) W_f
    err1 = 0.0
    for a in range(p):
        lhs = SB @ np.outer(v[a], v[a]).ravel()
        rhs = sum(np.exp(-2j * np.pi * f * a / p) * W[f] for f in range(p)) / m
        err1 = max(err1, float(np.abs(lhs - rhs).max()))

    # (2) Psi(v_a v_a^T) is the same for all a: <phi(a,a), Q phi(b,b)> = 1
    dia = np.stack([SB @ np.outer(v[a], v[a]).ravel() for a in range(p)])
    err2 = float(np.abs(dia @ Q @ dia.T - 1.0).max())

    # the claimed kernel: real span of W_f, f != 0
    K = np.stack([x for f in range(1, p) for x in (W[f].real, W[f].imag)])
    sv = np.linalg.svd(K, compute_uv=False)
    dimK = int((sv > 1e-8 * sv[0]).sum())
    qk = float(np.abs(K @ Q).max()) / max(float(np.abs(Q).max()), 1e-30)
    qw0 = float(np.abs(W[0].real @ Q).max())

    ev = np.linalg.eigvalsh(Q)
    rankQ = int((np.abs(ev) > 1e-7).sum())
    pred = (p - 1) * (p - 7) // 8

    ok = (err1 < 1e-9 and err2 < 1e-6 and dimK == p - 1 and qk < 1e-8
          and qw0 > 1e-3 and rankQ == pred == D - (p - 1))

    return dict(p=p, D=D, dim_kernel_claimed=dimK, rank=rankQ, predicted=pred,
                err_expansion=err1, err_diagonal=err2, rel_QW=qk, W0=qw0, ok=bool(ok))


def main():
    primes = [int(x) for x in sys.argv[1:]] or [29, 37, 41]
    out = []
    for p in primes:
        r = verify(p)
        if r is None:
            print(f"p = {p}: no stored solution, skipping")
            continue
        print(f"p = {r['p']}: dim Sym^2 = {r['D']}")
        print(f"  (1) expansion error                {r['err_expansion']:.2e}")
        print(f"  (2) Psi(v_a v_a^T) independent of a {r['err_diagonal']:.2e}")
        print(f"      dim span W_f (f != 0) = {r['dim_kernel_claimed']}  (p-1 = {p-1})")
        print(f"      max |W_f . Q| relative          {r['rel_QW']:.2e}")
        print(f"      |W_0 . Q| (must not vanish)     {r['W0']:.2e}")
        print(f"  rank Q = {r['rank']}, predicted {r['predicted']}")
        print(f"  => {'VERIFIED' if r['ok'] else 'FAILED'}")
        print()
        out.append(r)

    dst = os.path.join(ROOT, "results", "paley_rank_proof.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
