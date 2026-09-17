"""Identify the 65 canonical degree-4 moments on the Paley graph in closed form.

`paley_symmetrize.py` established that the optimal degree-4 pseudo-expectation is
already invariant under the full automorphism group, constant on Aut-orbits to
2e-12, with 65 distinct values at p = 29.  So there is a well-defined function

    m : {Aut-orbits of 4-subsets of F_p} -> R

to identify.  `paley_ansatz.py` already showed m is NOT a function of the
Legendre pattern of the six differences (an 11-parameter SDP was infeasible),
and `paley_symmetrize.py` shows why: 7 Legendre patterns cannot separate 65
orbits, and the spread inside a pattern is 9.8e-2, two orders above the moment
scale's precision.  So m depends on finer arithmetic than the six symbols.

The natural finer invariants are CHARACTER SUMS over the 4-set.  For
S = {a,b,c,d} define

    T4(S) = sum_x chi(x-a) chi(x-b) chi(x-c) chi(x-d),
    T3(S) = sum over the four triples {i,j,k} of sum_x chi(x-i) chi(x-j) chi(x-k).

Under x -> alpha*x + beta with alpha a quadratic residue, chi(alpha*u) = chi(u),
so both are Aut-invariant, integer valued, and O(sqrt p) by Weil.  They are
exactly the kind of quantity that is "finer than the Legendre symbols".

This script regresses the measured moments on those invariants and reports how
much structure they explain.  A near-exact fit would be a closed-form candidate
to verify as a certificate at other p; a poor fit rules the ansatz out, which is
equally useful and much cheaper than another SDP.

Run: python paley_identify.py [p]
"""
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def legendre(p):
    chi = np.zeros(p, dtype=int)
    qr = set(pow(a, 2, p) for a in range(1, p))
    for k in range(1, p):
        chi[k] = 1 if k in qr else -1
    return chi


def invariants(S, chi, p):
    """Character-sum invariants of a 4-subset S of F_p."""
    a, b, c, d = S
    T4 = int(sum(chi[(x - a) % p] * chi[(x - b) % p] * chi[(x - c) % p] * chi[(x - d) % p]
                 for x in range(p)))
    T3 = 0
    for tri in itertools.combinations(S, 3):
        i, j, k = tri
        T3 += int(sum(chi[(x - i) % p] * chi[(x - j) % p] * chi[(x - k) % p] for x in range(p)))
    nres = int(sum(1 for u, v in itertools.combinations(S, 2) if chi[(v - u) % p] == 1))
    return T4, T3, nres


def fit(X, y, names):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    rms = float(np.sqrt((resid ** 2).mean()))
    denom = float(np.sqrt(((y - y.mean()) ** 2).mean()))
    r2 = 1 - (rms / denom) ** 2 if denom > 0 else float("nan")
    return coef, rms, r2, float(np.abs(resid).max()), names


def main():
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    src = os.path.join(ROOT, "results", f"paley_symmetrised_{p}.json")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run paley_symmetrize.py {p} first")
    data = json.load(open(src, encoding="utf-8"))
    chi = legendre(p)

    rows = data["orbits"]
    y = np.array([r["mean"] for r in rows])
    inv = [invariants(tuple(r["rep"]), chi, p) for r in rows]
    T4 = np.array([t[0] for t in inv], dtype=float)
    T3 = np.array([t[1] for t in inv], dtype=float)
    nres = np.array([t[2] for t in inv], dtype=float)

    print(f"p = {p}: {len(rows)} Aut-orbits, moment scale {np.abs(y).max():.6f}")
    print(f"T4 range [{T4.min():.0f}, {T4.max():.0f}]   (Weil bound 3*sqrt p = {3*np.sqrt(p):.1f})")
    print(f"T3 range [{T3.min():.0f}, {T3.max():.0f}]")
    print(f"distinct (T4,T3,nres) triples: {len(set(inv))} of {len(rows)} orbits")

    one = np.ones_like(y)
    # Legendre pattern as one-hot on the number of residue differences (0..6)
    pat = np.stack([(nres == k).astype(float) for k in range(7)], axis=1)

    models = [
        ("pattern only (nres one-hot)", np.concatenate([pat], axis=1),
         [f"nres={k}" for k in range(7)]),
        ("pattern + T4", np.concatenate([pat, T4[:, None]], axis=1),
         [f"nres={k}" for k in range(7)] + ["T4"]),
        ("pattern + T4 + T3", np.concatenate([pat, T4[:, None], T3[:, None]], axis=1),
         [f"nres={k}" for k in range(7)] + ["T4", "T3"]),
        ("const + T4 + T3", np.stack([one, T4, T3], axis=1), ["1", "T4", "T3"]),
        ("pattern + T4 + T3 + T4^2 + T4*T3", np.concatenate(
            [pat, T4[:, None], T3[:, None], (T4 ** 2)[:, None], (T4 * T3)[:, None]], axis=1),
         [f"nres={k}" for k in range(7)] + ["T4", "T3", "T4^2", "T4*T3"]),
    ]

    print()
    best = None
    out = []
    for name, X, names in models:
        coef, rms, r2, mx, _ = fit(X, y, names)
        print(f"{name:34s}  R^2 {r2:10.7f}   rms {rms:.3e}   max|resid| {mx:.3e}")
        out.append(dict(model=name, r2=r2, rms=rms, max_resid=mx,
                        coef={n: float(c) for n, c in zip(names, coef)}))
        if best is None or rms < best[1]:
            best = (name, rms, r2, mx, coef, names)

    name, rms, r2, mx, coef, names = best
    print(f"\nbest: {name}")
    for n, c in zip(names, coef):
        print(f"    {n:10s} {c:+.9f}")

    scale = float(np.abs(y).max())
    print()
    if mx < 1e-9:
        print("VERDICT: EXACT to solver precision. This is a closed-form candidate;")
        print("next step is to verify it as a PSD certificate at another p.")
    elif mx < 0.01 * scale:
        print("VERDICT: explains most of the structure but is NOT exact. The residual")
        print(f"is {mx/scale:.2%} of scale -- a further invariant is missing.")
    else:
        print("VERDICT: character sums T4/T3 do NOT determine the moments.")
        print(f"Residual is {mx/scale:.1%} of scale. This ansatz is refuted; the")
        print("dependence is on something finer than these invariants.")

    dst = os.path.join(ROOT, "results", f"paley_identify_{p}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(dict(p=p, n_orbits=len(rows), scale=scale, models=out), f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
