"""Is the optimal degree-4 pseudo-expectation on the Paley graph already invariant
under the FULL automorphism group, or only under translation?

`circulant_sos.py` imposes the cyclic group Z_p.  But

    Aut(Paley_p) = { x -> a*x + b : a a quadratic residue, b in F_p },  order p(p-1)/2,

which is larger.  The optimal face of the SDP is convex and Aut-invariant, so
averaging any optimal point over Aut gives another optimal point that IS
Aut-invariant, with moments constant on Aut-orbits.  The analysis in
`paley_moments.py` never did that averaging, and bucketed by Legendre pattern
(11 buckets) rather than by Aut-orbit (65 at p = 29).  Those are different
groupings, so the spread it reported inside a Legendre pattern confounds two
things: real arithmetic structure finer than the Legendre symbols, and possible
failure of the solver's point to be Aut-symmetric at all.

This script separates them:

  * degree-2 check: pair moments should be constant on residues and on
    non-residues (2 Aut-orbits).  Deviation measures Aut-asymmetry directly.
  * degree-4 check: spread of the stored moments WITHIN each Aut-orbit.  Near
    zero means the solver already sits at a symmetric point; large means
    symmetrisation is doing real work.  Near zero is NOT evidence of a unique
    optimum -- the ADMM starts at Z = identity with equivariant updates, so it
    cannot leave the symmetric subspace either way.  paley_face.py tests that.
  * then emit the symmetrised value per Aut-orbit -- the canonical object to try
    to identify in closed form.

Run: python paley_symmetrize.py [p]   (default 29, reads results/paley_moments_<p>_y.npy)
"""
import itertools
import json
import os
import sys
from collections import defaultdict

import numpy as np

from circulant_sos import CirculantSoS, orbit_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def legendre(p):
    chi = np.zeros(p, dtype=int)
    qr = set(pow(a, 2, p) for a in range(1, p))
    for k in range(1, p):
        chi[k] = 1 if k in qr else -1
    return chi


def main():
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    ypath = os.path.join(ROOT, "results", f"paley_moments_{p}_y.npy")
    if not os.path.exists(ypath):
        raise SystemExit(f"missing {ypath} -- run paley_moments.py {p} first")

    chi = legendre(p)
    H = sorted(set(min(x, p - x) for x in range(1, p) if chi[x] == 1))
    C = CirculantSoS(p, device="cpu").set_instance(H)
    y = np.load(ypath)
    print(f"p = {p}: {C.n_var} translation-orbit variables, y[empty] = {y[0]:.12f}")
    y = y / y[0]                               # normalise Ehat[1] = 1

    QR = sorted(set(pow(a, 2, p) for a in range(1, p)))
    aut = [(a, b) for a in QR for b in range(p)]
    print(f"|Aut| = {len(aut)} = p(p-1)/2")

    def moment(S):
        return y[C.orbs[orbit_key(set(S), p)]]

    # ---------------------------------------------------------------- degree 2
    res = np.array([moment((0, d)) for d in range(1, p) if chi[d] == 1])
    non = np.array([moment((0, d)) for d in range(1, p) if chi[d] == -1])
    f_res, f_non = (-1 - np.sqrt(p)) / (p - 1), (-1 + np.sqrt(p)) / (p - 1)
    print("\ndegree 2 (2 Aut-orbits: residue / non-residue differences)")
    print(f"  residues    : mean {res.mean():+.10f}  spread {res.std():.2e}  closed form {f_res:+.10f}")
    print(f"  non-residues: mean {non.mean():+.10f}  spread {non.std():.2e}  closed form {f_non:+.10f}")
    d2_asym = max(res.std(), non.std())

    # ---------------------------------------------------------------- degree 4
    orbit_of = {}
    orbits = []
    for S in itertools.combinations(range(p), 4):
        fs = frozenset(S)
        if fs in orbit_of:
            continue
        k = len(orbits)
        members = set()
        for (a, b) in aut:
            members.add(frozenset((a * x + b) % p for x in S))
        for m in members:
            orbit_of[m] = k
        orbits.append(sorted(members))

    print(f"\ndegree 4: {len(orbits)} Aut-orbits over {sum(len(o) for o in orbits)} 4-sets")

    rows = []
    worst = 0.0
    for k, members in enumerate(orbits):
        vals = np.array([moment(tuple(sorted(m))) for m in members])
        spread = float(vals.std())
        worst = max(worst, spread)
        rep = tuple(sorted(min(members, key=lambda s: tuple(sorted(s)))))
        diffs = [(b - a) % p for a, b in itertools.combinations(rep, 2)]
        pat = tuple(sorted(int(chi[d]) for d in diffs))
        rows.append(dict(orbit=k, rep=list(rep), size=len(members),
                         mean=float(vals.mean()), spread=spread, legendre=list(pat)))

    means = np.array([r["mean"] for r in rows])
    scale = float(np.abs(means).max())
    print(f"  moment scale (max |mean|)          : {scale:.6f}")
    print(f"  WORST spread inside an Aut-orbit   : {worst:.3e}")
    print(f"  worst spread / scale               : {worst/scale:.3e}")
    print(f"  degree-2 Aut-asymmetry             : {d2_asym:.3e}")

    # contrast: the grouping paley_moments.py actually used
    by_pat = defaultdict(list)
    for r in rows:
        by_pat[tuple(r["legendre"])].append(r["mean"])
    pat_spread = max(float(np.std(v)) for v in by_pat.values() if len(v) > 1)
    print(f"\n  Legendre patterns                  : {len(by_pat)} (vs {len(orbits)} Aut-orbits)")
    print(f"  worst spread inside a Legendre pat.: {pat_spread:.3e}   <-- what was reported before")

    print()
    if worst < 1e-6:
        print("VERDICT: the stored solution is ALREADY Aut-invariant to solver precision.")
        print("The spread reported inside Legendre patterns is real arithmetic structure,")
        print("not an artefact: a Legendre pattern is a union of several Aut-orbits.")
        print("Symmetrisation is a no-op.")
        print("NOTE: this is NOT evidence that the optimum is unique. The ADMM starts at")
        print("Z = identity and every update is equivariant, because the instance data is")
        print("Aut-invariant, so the trajectory cannot leave the symmetric subspace whatever")
        print("the face looks like. See paley_face.py, which tests uniqueness directly.")
    else:
        print("VERDICT: the stored solution is NOT Aut-invariant; symmetrising is doing")
        print("real work. The per-orbit means below are the canonical optimal moments.")

    # distinct values, which is what the identification step has to explain
    uniq = np.unique(np.round(means, 9))
    print(f"\ndistinct symmetrised moment values: {len(uniq)} over {len(orbits)} orbits")
    print("extremes:", ", ".join(f"{v:+.6f}" for v in list(uniq[:3]) + ["..."] + list(uniq[-3:])
                                 if not isinstance(v, str)))

    out = os.path.join(ROOT, "results", f"paley_symmetrised_{p}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(p=p, n_aut_orbits=len(orbits), worst_orbit_spread=worst,
                       degree2_asymmetry=d2_asym, moment_scale=scale,
                       legendre_pattern_spread=pat_spread, orbits=rows), f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
