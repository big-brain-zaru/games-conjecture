"""
kv_invariant_sdp.py -- The basic Unique-Games SDP of the Khot-Vishnoi instance
U_{k,eta}, solved EXACTLY as a small LP by symmetry reduction.

Symmetry.  The label-extended graph of U_{k,eta} is the noisy N-cube on
{-1,1}^N (N = 2^k), a Cayley graph of F_2^N; the class structure {f chi_S} and
the weights are invariant under (i) translations f -> f.h of F_2^N and
(ii) AGL(k,2) acting on the point set F_2^k.  Averaging an optimal SDP solution
over this group gives an optimal solution of the form  X[f,g] = K(f.g)  with K
AGL(k,2)-invariant.  Then
  * X >= 0            <=>  Khat(alpha) = sum_h K(h) (-1)^{<alpha,h>} >= 0 for all alpha
                            (Fourier transform over F_2^N; alpha also up to AGL-orbits),
  * block orthogonality  K(chi_U) = 0 for U != {}  (hence K = 0 on all nonconstant
                            affine functions, by invariance),
  * block trace 1        N K(1) = 1.
  Objective (K' = N K):  (1/W) sum_{h : d(h) in window} K'(h) eta^{d(h)} (1-eta)^{N-d(h)},
  W = sum_{d in window} C(N,d) eta^d (1-eta)^{N-d};  d(h) = #{x : h(x) = -1}.
So the SDP is an LP in one variable per AGL(k,2)-orbit of Boolean functions on
k variables (k=3: 2^8 functions, k=4: 2^16), with one inequality per orbit.
Cross-checks: the numeric GPU solver on the full instance (k=4, eta=0.1 gave
[0.795002, 0.795010]).
"""
from __future__ import annotations

import itertools
import json
import math
import time

import numpy as np


def gl_generators(k: int):
    """Permutations of the N=2^k points x in F_2^k generating GL(k,2): transvections x_i <- x_i + x_j."""
    N = 1 << k
    pts = np.arange(N)
    gens = []
    # NOTE: translations x -> x ^ e_i are NOT symmetries of the instance: t_a(f chi_S) =
    # chi_S(a) (t_a f) chi_S mixes the classes of t_a f and -t_a f (minus a character is not
    # a character).  Only the linear group GL(k,2) acts; caught by the certified numeric
    # cross-check (AGL version gave 0.7813 < certified primal 0.795002 at k=4, eta=0.1).
    for i in range(k):
        for j in range(k):
            if i != j:
                xj = (pts >> j) & 1
                gens.append(pts ^ (xj << i))
    return gens


def function_orbits(k: int):
    """Orbit id for each of the 2^N functions (N-bit ints, bit x set iff h(x) = -1)
    under AGL(k,2); returns (orbit_id array, list of representatives)."""
    N = 1 << k
    n_f = 1 << N
    ids = np.arange(n_f, dtype=np.int64)
    bits = ((ids[:, None] >> np.arange(N)[None, :]) & 1).astype(np.uint8)     # (n_f, N): bit x
    pow2 = (1 << np.arange(N)).astype(np.int64)
    images = []
    for p in gl_generators(k):
        # (g.h)(x) = h(g^{-1} x); with p = action of g on points, g^{-1}x = pinv[x]
        pinv = np.empty(N, dtype=np.int64); pinv[p] = np.arange(N)
        img = (bits[:, pinv].astype(np.int64) * pow2[None, :]).sum(axis=1)
        images.append(img)
    lab = ids.copy()
    while True:
        new = lab.copy()
        for img in images:
            new = np.minimum(new, lab[img])
            new = np.minimum(new, new[img]) if False else new
        # propagate: label of h := min over generator images (iterate to fixed point)
        if np.array_equal(new, lab):
            break
        lab = new
    reps, orb = np.unique(lab, return_inverse=True)
    return orb, reps, bits


def wht(v: np.ndarray) -> np.ndarray:
    """Walsh-Hadamard transform over F_2^N (unnormalised), length 2^N."""
    v = v.astype(np.float64).copy()
    n = len(v); h = 1
    while h < n:
        v = v.reshape(-1, 2, h)
        a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = a + b; v[:, 1, :] = a - b
        v = v.reshape(n); h *= 2
    return v


def solve_kv_invariant(k: int, eta: float, d_window=None, keep_all=False, verbose=True):
    from scipy.optimize import linprog
    N = 1 << k
    n_f = 1 << N
    t0 = time.time()
    orb, reps, bits = function_orbits(k)
    n_orb = len(reps)
    d_of = bits.sum(axis=1)                                   # d(h)
    if keep_all:
        dmin, dmax = 1, N
    elif d_window is not None:
        dmin, dmax = d_window
    else:
        dmin = max(1, int(math.ceil(eta * N / 2))); dmax = min(N, int(math.floor(2 * eta * N)))
    W = sum(math.comb(N, d) * eta ** d * (1 - eta) ** (N - d) for d in range(dmin, dmax + 1))
    p = np.where((d_of >= dmin) & (d_of <= dmax), eta ** d_of * (1 - eta) ** (N - d_of), 0.0) / W
    c = np.zeros(n_orb); np.add.at(c, orb, p)                  # objective per orbit
    # Fourier matrix on orbits: A[o, o'] = sum_{h in o'} (-1)^{<alpha_o, h>}, alpha_o = rep of orbit o
    A = np.zeros((n_orb, n_orb))
    for o2 in range(n_orb):
        ind = (orb == o2).astype(np.float64)
        A[:, o2] = wht(ind)[reps]
    # constants and affine functions
    const_plus = int(orb[0])                                   # h = +1 everywhere -> id 0
    aff_orbits = set()
    for U in range(1, N):                                       # chi_U for U != 0
        h = 0
        for x in range(N):
            if bin(U & x).count("1") % 2 == 1:
                h |= (1 << x)
        aff_orbits.add(int(orb[h]))                        # only chi_U itself, not -chi_U
    # LP: max c.y  s.t.  -A y <= 0,  y[const_plus] = 1, y[aff] = 0, -1 <= y <= 1
    bounds = [(-1.0, 1.0)] * n_orb
    bounds[const_plus] = (1.0, 1.0)
    for o in aff_orbits:
        bounds[o] = (0.0, 0.0)
    res = linprog(-c, A_ub=-A, b_ub=np.zeros(n_orb), bounds=bounds, method="highs")
    val = -res.fun
    y = res.x
    # certificate check: Khat >= 0 on every alpha (full transform), K'(1)=1, K'(affine)=0
    K = y[orb]
    Khat = wht(K)
    info = {"k": k, "N": N, "eta": eta, "d_window": [dmin, dmax], "n_orbits": n_orb, "W": W,
            "lp_status": res.status, "value": float(val), "min_Khat": float(Khat.min()),
            "time": time.time() - t0}
    if verbose:
        print(json.dumps(info))
    return val, y, orb, reps, info


if __name__ == "__main__":
    import sys
    out = []
    for k, eta in [(3, 0.2), (3, 0.3), (4, 0.1), (4, 0.2)]:
        val, y, orb, reps, info = solve_kv_invariant(k, eta)
        out.append(info)
    json.dump(out, open("../results/kv_invariant_sdp_values.json", "w"), indent=1)
