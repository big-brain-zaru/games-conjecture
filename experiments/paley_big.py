"""Rank and family dimension of the Paley degree-4 extension at large p.

The dense route used up to p = 41 does not scale: at p = 61 the moment matrix has
1831 rows and there are 301 Aut-orbits, so holding one dense indicator matrix per
orbit would need about 8 GB, and the cvxpy SDP would be worse.  Two changes fix
that.

  * A feasible q is obtained from the symmetry-reduced ADMM solver
    (`circulant_sos.py`) rather than from a general SDP.  Its Z_p-equivariant
    dynamics start at a symmetric point, so the iterate stays Aut-invariant, and
    it converges to a point of the optimal face.
  * The orbit indicators are never formed densely.  They are stored as index
    arrays (I, J, orbit) over the disjoint pair positions, and the Gram matrix of
    the kernel conditions is computed as

        G[k,l] = <A_k W, A_l W>_F = tr(A_k A_l P),    P = W W^T,

    which needs only one sparse-times-dense product per orbit.  The family
    dimension is then K - rank(G), and the rank of the moment matrix comes from
    its eigenvalues directly.

Validated against the dense results: p = 29 must give rank 77 and family 0,
p = 37 rank 135 and family 2, p = 41 rank 170 and family 3.

Run: python paley_big.py 53 61        (or 29 41 to re-check against the dense route)
"""
import argparse
import itertools
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def legendre(p):
    chi = np.zeros(p, dtype=int)
    qr = set(pow(a, 2, p) for a in range(1, p))
    for k in range(1, p):
        chi[k] = 1 if k in qr else -1
    return chi


def aut_orbit_map(p):
    """orbit index for every 4-subset, and the representatives."""
    QR = sorted(set(pow(a, 2, p) for a in range(1, p)))
    aut = [(a, b) for a in QR for b in range(p)]
    orbit_of, reps = {}, []
    for S in itertools.combinations(range(p), 4):
        fs = frozenset(S)
        if fs in orbit_of:
            continue
        k = len(reps)
        for (a, b) in aut:
            orbit_of[frozenset((a * x + b) % p) if False else
                      frozenset((a * x + b) % p for x in S)] = k
        reps.append(tuple(sorted(S)))
    return orbit_of, reps


def solve_q(p, iters, device):
    """Feasible Aut-invariant q from the symmetry-reduced ADMM solver."""
    import torch
    from circulant_sos import CirculantSoS, orbit_key
    chi = legendre(p)
    H = sorted(set(min(x, p - x) for x in range(1, p) if chi[x] == 1))
    C = CirculantSoS(p, device=device).set_instance(H)
    val = C.solve(iters=iters, tol=1e-13)
    lo, hi = C.certified_bounds()
    y = C.y.detach().cpu().numpy(); y = y / y[0]
    closed = 0.5 + (1 + np.sqrt(p)) / (2 * (p - 1))
    print(f"  ADMM: value {val:.12f}  bounds [{lo:.10f}, {hi:.10f}]  "
          f"closed form {closed:.12f}  its {C.iters_done}")
    return y, C, orbit_key, lo, hi, closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("primes", nargs="*", type=int, default=[53, 61])
    ap.add_argument("--iters", type=int, default=200000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import scipy.sparse as sp
    import torch
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    out = []
    for p in args.primes:
        m = (p - 1) // 2
        print(f"\n=== p = {p} ===  predicted rank {(p-1)*(p-7)//8}, "
              f"family {(p-1)//4 - 7}")
        y, C, orbit_key, lo, hi, closed = solve_q(p, args.iters, dev)

        pairs = list(itertools.combinations(range(p), 2))
        pidx = {frozenset(e): i + 1 for i, e in enumerate(pairs)}
        n = 1 + len(pairs)
        chi = legendre(p)
        orbit_of, reps = aut_orbit_map(p)
        K = len(reps)
        print(f"  rows {n}, Aut-orbits {K}")

        # forced part
        def Xv(a, b):
            return (-1 - np.sqrt(p) * chi[(a - b) % p]) / (p - 1)

        M = np.zeros((n, n))
        M[0, 0] = 1.0
        for e in pairs:
            i = pidx[frozenset(e)]
            M[0, i] = M[i, 0] = Xv(*e)
            M[i, i] = 1.0
        # share-one and disjoint entries, plus the orbit index arrays
        I, J, KI = [], [], []
        for e, f in itertools.combinations(pairs, 2):
            se, sf = set(e), set(f)
            ie, if_ = pidx[frozenset(e)], pidx[frozenset(f)]
            common = se & sf
            if len(common) == 1:
                (a,), (b,) = tuple(se - sf), tuple(sf - se)
                M[ie, if_] = M[if_, ie] = Xv(a, b)
            elif not common:
                k = orbit_of[frozenset(se | sf)]
                q = y[C.orbs[orbit_key(se | sf, p)]]
                M[ie, if_] = M[if_, ie] = q
                I.append(ie); J.append(if_); KI.append(k)
        I = np.array(I); J = np.array(J); KI = np.array(KI)
        print(f"  disjoint positions {len(I)} = 3*C(p,4) ({3*len(list(itertools.combinations(range(p),4)))//1})"
              if p <= 29 else f"  disjoint positions {len(I)}")

        w, V = np.linalg.eigh(M)
        rank = int((w > 1e-7).sum())
        gap_lo = w[n - rank - 1] if rank < n else float("nan")
        print(f"  lambda_min {w[0]:+.3e}   rank {rank}   predicted {(p-1)*(p-7)//8}")
        print(f"  gap: largest 'zero' {gap_lo:+.2e} | smallest nonzero {w[n-rank]:+.3e}")

        W = V[:, np.abs(w) < 1e-7]
        P = W @ W.T
        G = np.zeros((K, K))
        for l in range(K):
            sel = KI == l
            A_l = sp.csr_matrix((np.ones(2 * sel.sum()),
                                 (np.concatenate([I[sel], J[sel]]),
                                  np.concatenate([J[sel], I[sel]]))), shape=(n, n))
            S = A_l @ P
            S = (S + S.T) / 2
            vals = 2.0 * S[I, J]
            G[:, l] = np.bincount(KI, weights=vals, minlength=K)
        G = (G + G.T) / 2
        ge = np.linalg.eigvalsh(G)
        ge = ge[::-1]
        thr = 1e-9 * max(ge[0], 1e-30)
        rk = int((ge > thr).sum())
        fam = K - rk
        print(f"  Gram of the kernel conditions: rank {rk} of {K} -> family dimension {fam}")
        print(f"    eigenvalue gap: {ge[rk-1]:.3e} (last kept) vs {ge[rk]:.3e} (first dropped)"
              if 0 < rk < K else "")
        pred = (p - 1) // 4 - 7
        print(f"  PREDICTED family {pred}  -> {'MATCH' if fam == pred else 'MISMATCH'}")

        out.append(dict(p=p, n=n, K=K, rank=rank, rank_predicted=(p - 1) * (p - 7) // 8,
                        lambda_min=float(w[0]), family=fam, family_predicted=pred,
                        gram_rank=rk, sos4_lo=float(lo), sos4_hi=float(hi),
                        sos2=float(closed)))
        with open(os.path.join(ROOT, "results", f"paley_big_{p}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(out[-1], f, indent=1)

    print("\n  p    rank  predicted   family  predicted")
    for r in out:
        print(f"{r['p']:4d} {r['rank']:7d} {r['rank_predicted']:10d} "
              f"{r['family']:8d} {r['family_predicted']:10d}")


if __name__ == "__main__":
    main()
