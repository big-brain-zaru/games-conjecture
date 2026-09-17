"""Find LOCAL identities forced on the Paley degree-4 extension.

Established so far (paley_exact.py, paley_rigid.py) at p = 29:
  * the question reduces to one feasibility SDP in 65 Aut-orbit unknowns;
  * its value is t* = 0, so every valid extension has a singular moment matrix;
  * M_even is 407 x 407 of rank 77, kernel dimension 330;
  * the kernel conditions determine q UNIQUELY (rank 65 of 65).

A kernel of codimension 77 in dimension 407 is enormous, so the kernel should
contain vectors supported on very few coordinates.  A kernel vector supported on
the pairs inside a small vertex subset T is a LOCAL identity: a linear relation,
with coefficients in Q(sqrt p), among the X's and the q's of the 4-subsets of T
alone.  Enough of those and q is pinned down by hand rather than by a solver,
which is what a closed form actually is.

For a subset T, a vector supported on {empty} union {pairs inside T} lies in
ker M iff it lies in ker of the 407 x (1 + C(|T|,2)) column block M[:, T].  So
the nullity of that column block, for |T| = 4, 5, 6, says exactly how many local
identities live on T.

Run: python paley_local.py [p]
"""
import itertools
import json
import os
import random
import sys

import numpy as np

from paley_exact import build_parts, index_pairs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    src = os.path.join(ROOT, "results", f"paley_rigid_{p}.json")
    if not os.path.exists(src):
        raise SystemExit(f"missing {src} -- run paley_rigid.py {p} first")
    q = np.array(json.load(open(src, encoding="utf-8"))["q_solved"])

    M0, A, reps, X, pairs, pidx = build_parts(p)
    M = M0 + sum(q[k] * A[k] for k in range(len(reps)))
    n = M.shape[0]
    ev = np.linalg.eigvalsh(M)
    rank = int((ev > 1e-7).sum())
    print(f"p = {p}: M_even {n} x {n}, rank {rank}, kernel {n - rank}")
    print(f"  spectral gap: largest 'zero' {ev[n-rank-1]:+.2e} | smallest nonzero {ev[n-rank]:+.2e}")

    rows = []
    print("\n  local identities on a vertex subset T (nullity of the column block):")
    for s in (4, 5, 6, 7):
        cols_needed = 1 + s * (s - 1) // 2
        nulls = []
        subsets = list(itertools.combinations(range(p), s))
        random.seed(0)
        sample = subsets if len(subsets) <= 60 else random.sample(subsets, 60)
        for T in sample:
            idx = [0] + [pidx[frozenset(e)] for e in itertools.combinations(sorted(T), 2)]
            blk = M[:, idx]
            sv = np.linalg.svd(blk, compute_uv=False)
            nulls.append(int((sv < 1e-6).sum()))
        nulls = np.array(nulls)
        print(f"    |T| = {s}: block has {cols_needed:3d} columns, nullity "
              f"min {nulls.min()} max {nulls.max()} mean {nulls.mean():.2f}")
        rows.append(dict(s=s, cols=cols_needed, null_min=int(nulls.min()),
                         null_max=int(nulls.max()), null_mean=float(nulls.mean())))
        if nulls.max() > 0:
            # exhibit one
            T = sample[int(np.argmax(nulls))]
            idx = [0] + [pidx[frozenset(e)] for e in itertools.combinations(sorted(T), 2)]
            U, sv, Vt = np.linalg.svd(M[:, idx], compute_uv=True)
            wv = Vt[-1]
            wv = wv / np.abs(wv).max()
            print(f"      example T = {sorted(T)}")
            print(f"      coefficients (empty, then pairs in lex order):")
            print("       ", " ".join(f"{c:+.6f}" for c in wv))
            rows[-1]["example_T"] = [int(t) for t in sorted(T)]
            rows[-1]["example_vector"] = [float(c) for c in wv]

    print()
    if any(r["null_max"] > 0 for r in rows if r["s"] <= 6):
        print("VERDICT: local identities exist on small vertex sets. Each is a closed-form")
        print("relation among X and the q of the 4-subsets of T, and the Aut-orbit of one")
        print("such identity generates many. This is the route to a formula by hand.")
    else:
        print("VERDICT: no identity is supported on 6 or fewer vertices, so the kernel,")
        print("though huge, is globally spread. A closed form cannot be read off locally;")
        print("the rigidity is a property of the whole instance.")

    dst = os.path.join(ROOT, "results", f"paley_local_{p}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(dict(p=p, n=n, rank=rank, kernel=n - rank, blocks=rows), f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
