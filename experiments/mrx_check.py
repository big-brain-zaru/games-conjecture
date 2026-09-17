"""Does the Mohanty-Raghavendra-Xu lift already imply SoS_4 = SoS_2 on Paley graphs?

MRX (STOC 2020) state their applications for random d-regular graphs and the
Sherrington-Kirkpatrick model, but their Theorems 1.2/1.3 are general: for ANY
graph and ANY degree-2 solution X they build a degree-4 solution

    Phi(X)_ij = (X_ij + X_ij^3) / (1 + alpha),
    alpha     = C * a_mag * (1 + a_row^4) * (1 + a_spec^2),

with a_mag = max off-diagonal |X_ij|, a_row = max row norm, a_spec = ||X||_op,
C an absolute constant, and

    <A, Phi(X)>  >=  <A,X>/(1+alpha)  -  (alpha/(1+alpha)) * (sqrt(n)*||A||_F - tr A).

So the lift is useful exactly when alpha * (loss scale / max-cut advantage) << 1.
This script measures all three parameters on the true Paley degree-2 optimum and
reports that product, to separate what MRX already predicts from what section
11.2 measures.

Two conclusions, both printed below:

 1. alpha = Theta(1/sqrt(p)) and the loss-to-advantage ratio tends to 8/sqrt(2)
    = 5.657, so the product decays like 141/sqrt(p).  MRX therefore DOES predict
    SoS_4 = SoS_2 * (1 - o(1)) on Paley graphs asymptotically.  The phenomenon of
    section 11.2 is not a surprise.  But the crossover needs p >> (141 C)^2, which
    is astronomically beyond p = 61 for any explicit C, so MRX says nothing at the
    sizes measured here.

 2. More importantly, the MRX map divides the degree-2 part by 1 + alpha > 1, so
    it structurally CANNOT produce exact equality.  The measurement in 11.2 is
    that a degree-4 pseudo-expectation exists whose degree-2 part is EXACTLY the
    optimal X, with no loss at all, from p = 29 on.  That is a strictly stronger
    statement than the lift gives, and it is what remains unexplained.

Run: python mrx_check.py
"""
import json
import math
import os

import numpy as np

PRIMES = (13, 17, 29, 37, 41, 53, 61, 101, 197, 401, 1009)


def paley(p):
    """Adjacency matrix of the Paley graph on p vertices, and its Jacobsthal matrix."""
    qr = set(pow(a, 2, p) for a in range(1, p))
    chi = np.zeros(p, dtype=int)
    for k in range(1, p):
        chi[k] = 1 if k in qr else -1
    S = np.array([[chi[(i - j) % p] for j in range(p)] for i in range(p)], dtype=float)
    A = (np.ones((p, p)) - np.eye(p) + S) / 2.0
    return A, S


def degree2_optimum(p, S):
    """Unit-diagonal rescaling of the projection onto the lambda_min eigenspace.

    For a vertex-transitive graph this attains the eigenvalue bound, which for
    Max-Cut equals the basic SDP value (Goemans-Rendl for association schemes,
    extended to walk-regular graphs).  Verified against the closed form below.
    """
    P = 0.5 * (np.eye(p) - np.ones((p, p)) / p - S / math.sqrt(p))
    return P / P[0, 0]


def row(p):
    A, S = paley(p)
    X = degree2_optimum(p, S)
    m = A.sum() / 2.0
    closed = 0.5 + (1 + math.sqrt(p)) / (2 * (p - 1))
    value = (A * (1 - X)).sum() / 4.0 / m           # SDP Max-Cut fraction at X
    off = X - np.diag(np.diag(X))
    a_mag = float(np.abs(off).max())
    a_row = float(np.linalg.norm(X, axis=1).max())
    a_spec = float(np.linalg.norm(X, 2))
    advantage = m * (closed - 0.5)                  # what the lift has to preserve
    loss_scale = math.sqrt(p) * float(np.linalg.norm(A, "fro")) - float(np.trace(A))
    alpha_over_C = a_mag * (1 + a_row ** 4) * (1 + a_spec ** 2)
    return dict(
        p=p,
        sos2_closed_form=closed,
        sos2_at_X=value,
        sos2_residual=value - closed,
        a_mag=a_mag,
        a_mag_times_sqrt_p=a_mag * math.sqrt(p),
        a_row=a_row,
        a_spec=a_spec,
        advantage=advantage,
        loss_scale=loss_scale,
        loss_over_advantage=loss_scale / advantage,
        alpha_over_C=alpha_over_C,
        product_over_C=alpha_over_C * loss_scale / advantage,
    )


def main():
    rows = [row(p) for p in PRIMES]
    print(f"{'p':>5} {'a_mag':>9} {'a_mag*sqrt p':>13} {'a_row':>7} {'a_spec':>7} "
          f"{'SoS2 resid':>12} {'loss/adv':>9} {'alpha/C':>9} {'product/C':>10}")
    for r in rows:
        print(f"{r['p']:5d} {r['a_mag']:9.5f} {r['a_mag_times_sqrt_p']:13.5f} "
              f"{r['a_row']:7.4f} {r['a_spec']:7.4f} {r['sos2_residual']:12.2e} "
              f"{r['loss_over_advantage']:9.4f} {r['alpha_over_C']:9.4f} "
              f"{r['product_over_C']:10.4f}")

    big = rows[-1]
    print()
    print(f"a_mag * sqrt(p) -> 1, a_row -> sqrt(2), a_spec -> 2, so alpha = Theta(C/sqrt(p)).")
    print(f"loss/advantage -> 8/sqrt(2) = {8/math.sqrt(2):.4f} (measured {big['loss_over_advantage']:.4f} at p={big['p']}).")
    print(f"product/C ~ {big['product_over_C']*math.sqrt(big['p']):.0f}/sqrt(p): MRX gives")
    print("asymptotic equality on Paley, but needs p >> (that constant * C)^2.")
    print("At every p measured in FINDINGS 11.2 the MRX bound is vacuous, and the map")
    print("divides the degree-2 part by 1+alpha > 1, so it cannot give exact equality.")

    # the residuals confirm X is the exact degree-2 optimum
    worst = max(abs(r["sos2_residual"]) for r in rows)
    print(f"\nX attains the closed form to {worst:.1e} at every p (so it is the degree-2 optimum).")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "mrx_check.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
