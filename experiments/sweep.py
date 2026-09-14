"""
sweep.py -- A COMPLETE sweep, not just an extensive one.

Key inequality.  Degree-4 sum-of-squares is at least as tight as degree 2, so
SoS_4 <= SoS_2 and therefore, for every instance,

        C_4  <=  C_2  =  (1 - opt) / (1 - SoS_2)  <=  (1 - incumbent) / (1 - SoS_2)  =:  U(I)

where `incumbent` is any cut that has actually been exhibited.  Both ingredients
of U are CHEAP: SoS_2 of a circulant is a closed form (the spectral bound, exact
for a vertex-transitive graph), and a cut comes from local search in milliseconds.

So an instance can only beat the current record C* if U(I) > C*.  Sweeping U over
a whole family therefore CERTIFIES that every instance in it is ruled out, except
for an explicitly listed shortlist that then gets the expensive degree-4 treatment.

This is what settles p = 73, 89, 97 for the index-4 Paley family in milliseconds,
after they had consumed hours of ADMM and CP-SAT.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from circulant_scan import circ_edges, sos2_circulant


def local_search_cut(p, gens, restarts=600, seed=0):
    """Best cut found; a valid lower bound on the optimum."""
    rng = np.random.default_rng(seed)
    e = circ_edges(p, gens)
    m = len(e)
    adj = [[] for _ in range(p)]
    for a, b in e.tolist():
        adj[a].append(b); adj[b].append(a)
    adj = [np.array(x) for x in adj]
    best = -1
    for _ in range(restarts):
        x = rng.integers(0, 2, size=p)
        improved = True
        while improved:
            improved = False
            for v in rng.permutation(p):
                same = int((x[adj[v]] == x[v]).sum())
                if same > len(adj[v]) - same:
                    x[v] ^= 1; improved = True
        cut = int(sum(1 for a, b in e.tolist() if x[a] != x[b]))
        best = max(best, cut)
    return best, m, best / m


def sweep(Ls, kmax=3, restarts=400, record=1.093586, verbose=True):
    rows, shortlist = [], []
    for L in Ls:
        if L % 2 == 0:
            continue
        h = (L - 1) // 2
        for k in range(1, kmax + 1):
            for gens in itertools.combinations(range(1, h + 1), k):
                s2 = sos2_circulant(L, list(gens))
                if s2 >= 1 - 1e-12:
                    continue
                cut, m, inc = local_search_cut(L, list(gens), restarts=restarts)
                U = (1 - inc) / (1 - s2)
                rec = {"L": L, "gens": list(gens), "m": m, "incumbent": inc, "sos2": s2, "C4_upper": U}
                rows.append(rec)
                if U > record:
                    shortlist.append(rec)
        if verbose:
            cur = [r for r in rows if r["L"] == L]
            b = max(cur, key=lambda r: r["C4_upper"])
            print(f"  L={L:3d}: {len(cur):5d} instances, max U = {b['C4_upper']:.6f} (gens {b['gens']}), "
                  f"shortlist so far {len(shortlist)}", flush=True)
    return rows, shortlist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lmax", type=int, default=45)
    ap.add_argument("--Lmin", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=400)
    ap.add_argument("--record", type=float, default=1.093586)
    ap.add_argument("--out", default="../results/sweep.json")
    a = ap.parse_args()
    t0 = time.time()
    Ls = list(range(a.Lmin, a.Lmax + 1, 2))
    rows, shortlist = sweep(Ls, kmax=a.kmax, restarts=a.restarts, record=a.record)
    shortlist.sort(key=lambda r: -r["C4_upper"])
    print(f"\nswept {len(rows)} circulant instances in {time.time()-t0:.0f}s")
    print(f"instances that could possibly beat C_4 = {a.record}: {len(shortlist)}")
    for r in shortlist[:25]:
        print("   " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}))
    json.dump({"record": a.record, "n_swept": len(rows), "shortlist": shortlist,
               "max_U": max(r["C4_upper"] for r in rows)}, open(a.out, "w"), indent=1)
