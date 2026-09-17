"""The complete-graph identity is a corollary of Laurent (2003), and a weak one.

Max-Cut on K_n with unit weights is a univariate problem: cutting a set of size w
gives w(n-w) edges, so maximising the cut is minimising (sum_i x_i)^2 over the
hypercube.  For odd n the true minimum is 1, and the relaxation question is
whether degree-d sum-of-squares can certify sum_i x_i != 0.  That is exactly the
parity/knapsack refutation studied by Grigoriev (2001), and settled for the
Lasserre hierarchy by:

  M. Laurent, "Lower bound for the number of iterations in semidefinite
  hierarchies for the cut polytope", Math. Oper. Res. 28(4), 2003, 871-883.

Her Theorem 6 constructs, for odd n, the pseudo-moment sequence

    a_0 = 1,   a_{2r+2} = -a_{2r} * (2r+1)/(n-2r-1),
    equivalently  a_{2r} = (-1/4)^r * C(2r,r) / C((n-1)/2, r),
    y_I = a_{|I|} for even I,  y_I = 0 for odd I,

and proves M_{(n-1)/2}(y) >= 0.  Since a_2 = -1/(n-1), this y has
Ehat[(sum_i x_i)^2] = n + n(n-1)*a_2 = 0, i.e. it sits exactly at the basic SDP
optimum n/(2(n-1)).  Feasibility at order (n-1)/2 implies feasibility at every
lower order, so for odd n >= 5 the DEGREE-4 value is already n/(2(n-1)).

So SoS_4(K_n) = SoS_2(K_n) is not new, and the published statement is much
stronger than ours: it holds for every degree up to n-1, not just 4.  Hence
C_d(K_n) = (n-1)^2/(n(n-2)) for all d <= n-1, which still tends to 1.

This script rebuilds Laurent's y, checks the degree-4 moment matrix is PSD, and
checks the value it certifies equals the closed form our solver measured.

Run: python laurent_kn.py
"""
import itertools
import json
import math
import os

import numpy as np


def laurent_a(n):
    """a_{2r} for r = 0 .. (n-1)/2, by the recurrence (10)."""
    k = (n - 1) // 2
    a = {0: 1.0}
    for r in range(k):
        a[2 * r + 2] = -a[2 * r] * (2 * r + 1) / (n - 2 * r - 1)
    return a


def closed_form_a(n, r):
    """(-1/4)^r * C(2r,r) / C((n-1)/2, r)."""
    return (-0.25) ** r * math.comb(2 * r, r) / math.comb((n - 1) // 2, r)


def moment_matrix(n, a, order):
    """M_order(y) over subsets of [n] of size <= order, y_I = a_{|I|} (even), 0 (odd)."""
    idx = [frozenset(c) for s in range(order + 1) for c in itertools.combinations(range(n), s)]
    N = len(idx)
    M = np.zeros((N, N))
    for i, I in enumerate(idx):
        for j, J in enumerate(idx):
            d = len(I ^ J)
            M[i, j] = a[d] if d % 2 == 0 else 0.0
    return M, idx


def main():
    rows = []
    print(f"{'n':>4} {'lam_min M_2':>14} {'Ehat[(sum x)^2]':>17} {'SoS4 = SoS2':>13} "
          f"{'opt':>10} {'C_4':>10} {'recurrence=closed':>18}")
    for n in range(5, 22, 2):
        a = laurent_a(n)
        # the recurrence and the closed form must agree
        agree = max(abs(a[2 * r] - closed_form_a(n, r)) for r in range((n - 1) // 2 + 1))

        M2, _ = moment_matrix(n, a, 2)
        lam = float(np.linalg.eigvalsh(M2).min())

        # Ehat[(sum_i x_i)^2] = n + n(n-1) a_2
        s2 = n + n * (n - 1) * a[2]

        m = n * (n - 1) / 2
        relax = (m - m * a[2]) / 2 / m          # (1 - a_2)/2, the cut fraction at y
        closed = n / (2 * (n - 1))
        opt = (n + 1) / (2 * n)                 # (n^2-1)/4 edges, odd n
        C4 = (1 - opt) / (1 - closed)
        C4_closed = (n - 1) ** 2 / (n * (n - 2))
        rows.append(dict(n=n, lam_min=lam, sum_sq=s2, relax=relax, closed=closed,
                         opt=opt, C4=C4, C4_closed=C4_closed, recurrence_vs_closed=agree))
        print(f"{n:4d} {lam:14.3e} {s2:17.2e} {relax:13.9f} {opt:10.6f} "
              f"{C4:10.6f} {agree:18.1e}")

    print()
    worst_lam = min(r["lam_min"] for r in rows)
    worst_val = max(abs(r["relax"] - r["closed"]) for r in rows)
    worst_c4 = max(abs(r["C4"] - r["C4_closed"]) for r in rows)
    print(f"min eigenvalue of M_2(y) over all n: {worst_lam:.3e}  (>= 0 means Laurent's y is degree-4 feasible)")
    print(f"value it certifies vs n/(2(n-1)):   {worst_val:.1e}")
    print(f"C_4 vs (n-1)^2/(n(n-2)):            {worst_c4:.1e}")
    print()
    print("Laurent proves M_{(n-1)/2}(y) >= 0, so this holds for every degree up to n-1,")
    print("not only degree 4.  The complete-graph row is a corollary of a 2003 theorem.")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "laurent_kn.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
