"""
paley_moments.py -- what does the optimal degree-4 pseudo-expectation on the Paley
graph look like?  Solve degree-4 SoS for Max-Cut on P_p (symmetry-reduced), read off
E[x_S] for the 4-sets S, and test whether it is a function of the Legendre-symbol
pattern of the six pairwise differences (the AGL-invariant data of a 4-set).
"""
import sys, json, itertools, collections
import numpy as np, torch
from circulant_sos import CirculantSoS

p = int(sys.argv[1]) if len(sys.argv) > 1 else 29
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
chi = np.zeros(p, dtype=int)
for a in range(1, p):
    chi[pow(a, 2, p)] = 1
chi = np.where(chi == 1, 1, -1); chi[0] = 0
H = sorted(set(min(x, p - x) for x in range(1, p) if chi[x] == 1))
P = CirculantSoS(p, device="cuda").set_instance(H)
val = P.solve(iters=iters, tol=1e-12)
lo, hi = P.certified_bounds()
s2 = 0.5 + (1 + np.sqrt(p)) / (2 * (p - 1))
print(f"p={p}: SoS4 in [{lo:.10f},{hi:.10f}]  SoS2 {s2:.10f}  its {P.iters_done}")
y = P.y.detach().cpu().numpy()
reps = [None] * P.n_var            # orbit representative subsets, indexed like y
for key, i in P.orbs.items():
    reps[i] = tuple(sorted(key))
# pairs
pair = {}
for i, r in enumerate(reps):
    if len(r) == 2:
        pair[r[1]] = y[i]
print("pair moments: residue", np.mean([pair[d] for d in pair if chi[d] == 1]), " nonresidue",
      np.mean([pair[d] for d in pair if chi[d] == -1]), " formula", (-1 - np.sqrt(p)) / (p - 1), (-1 + np.sqrt(p)) / (p - 1))
# 4-sets: Legendre pattern of the 6 differences, as an invariant of the AGL-orbit
by_pattern = collections.defaultdict(list)
for i, r in enumerate(reps):
    if len(r) == 4:
        diffs = [(b - a) % p for a, b in itertools.combinations(r, 2)]
        pat = tuple(sorted(chi[d] for d in diffs))
        # finer invariant: number of residue differences at each vertex (sorted)
        deg = tuple(sorted(sum(chi[(b - a) % p] == 1 for b in r if b != a) for a in r))
        by_pattern[(pat, deg)].append(y[i])
print(f"{sum(len(v) for v in by_pattern.values())} translation-orbits of 4-sets, {len(by_pattern)} Legendre patterns")
out = []
for k in sorted(by_pattern):
    v = np.array(by_pattern[k])
    print(f"  pattern {k}: n={len(v):4d}  mean {v.mean():+.6f}  std {v.std():.2e}  min {v.min():+.6f} max {v.max():+.6f}")
    out.append({"pattern": [[int(x) for x in k[0]], [int(x) for x in k[1]]], "n": int(len(v)), "mean": float(v.mean()), "std": float(v.std())})
json.dump({"p": p, "sos4_lo": lo, "sos4_hi": hi, "sos2": s2, "patterns": out}, open(f"../results/paley_moments_{p}.json", "w"), indent=1)
np.save(f"../results/paley_moments_{p}_y.npy", y)
