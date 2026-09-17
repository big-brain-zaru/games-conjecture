"""Is the optimal degree-4 pseudo-expectation on the Paley graph unique?

`paley_symmetrize.py` found the stored solution constant on Aut-orbits to 2e-12
and I first read that as evidence of uniqueness.  That inference is wrong.  The
ADMM in `circulant_sos.py` starts at Z = identity, U = 0, which is invariant
under everything, and every update is equivariant because the instance data
(the quadratic residues) is Aut-invariant.  So the whole trajectory is forced to
stay Aut-symmetric regardless of whether the optimum is unique.  The symmetry
was built in, not discovered.

That matters, because it decides whether "identify the moments in closed form"
is even a well-posed target.  If the optimal face is a single point, the 65
values are canonical and worth identifying.  If it is positive-dimensional, ADMM
lands on whichever point its dynamics prefer, those coordinates mean nothing in
particular, and the right target is instead to find ANY nice point in the face.

Test: solve from several random, deliberately NON-symmetric starts and compare
the converged 4-set moments.  Same objective value with different moments means
a positive-dimensional face.

Run: python paley_face.py [p] [iters]
"""
import itertools
import json
import os
import sys

import numpy as np
import torch

from circulant_sos import CirculantSoS, orbit_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def solve_from(C, Z0, iters, tol=1e-12, alpha=1.7):
    """The ADMM of circulant_sos.solve, but started at a caller-supplied Z0."""
    L, nI = C.L, C.nI
    Ctil = C.blocks(C.c / C.norm)
    rho = max(1e-3, float(Ctil.abs().max()) * 10)
    Z = Z0 * C.mask
    U = torch.zeros_like(Z)
    y = None
    for it in range(iters):
        M, y = C.proj_L(Z - U + Ctil / rho)
        Mh = alpha * M + (1 - alpha) * Z
        Znew = C.proj_psd(Mh + U)
        U = U + Mh - Znew
        r = float((M - Znew).norm()); s = float(rho * (Znew - Z).norm())
        Z = Znew
        if it % 40 == 0 and it > 0:
            if r > 10 * s:
                rho *= 2; U = U / 2
            elif s > 10 * r:
                rho /= 2; U = U * 2
        if r < tol and s < tol and it > 30:
            break
    C.Z, C.U, C.y, C.rho, C.iters_done = Z, U, y, rho, it + 1
    return C.const + float(C.c @ y)


def main():
    p = int(sys.argv[1]) if len(sys.argv) > 1 else 29
    iters = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    chi = np.zeros(p, dtype=int)
    qr = set(pow(a, 2, p) for a in range(1, p))
    for k in range(1, p):
        chi[k] = 1 if k in qr else -1
    H = sorted(set(min(x, p - x) for x in range(1, p) if chi[x] == 1))

    sets4 = [S for S in itertools.combinations(range(p), 4)]
    closed = 0.5 + (1 + np.sqrt(p)) / (2 * (p - 1))

    runs = []
    for trial in range(3):
        C = CirculantSoS(p, device=dev).set_instance(H)
        g = torch.Generator(device="cpu").manual_seed(1000 + trial)
        if trial == 0:
            Z0 = torch.zeros(C.L, C.nI, C.nI, dtype=C.cdtype, device=dev)
            Z0[:] = torch.eye(C.nI, dtype=C.cdtype, device=dev)
            label = "default symmetric start (Z = I)"
        else:
            A = torch.randn(C.L, C.nI, C.nI, generator=g, dtype=torch.float64)
            A = (A + A.transpose(1, 2)) / 2
            Z0 = (A @ A.transpose(1, 2)).to(C.cdtype).to(dev)   # random PSD, not Aut-symmetric
            label = f"random non-symmetric start, seed {1000+trial}"
        val = solve_from(C, Z0, iters)
        lo, hi = C.certified_bounds()
        yv = C.y.detach().cpu().numpy(); yv = yv / yv[0]
        m = np.array([yv[C.orbs[orbit_key(set(S), p)]] for S in sets4])
        runs.append((label, val, lo, hi, m, C.iters_done))
        print(f"{label:38s} value {val:.10f}  bounds [{lo:.9f},{hi:.9f}]  its {C.iters_done}")

    print(f"\nclosed-form SoS2 = {closed:.10f}")
    base = runs[0][4]
    print()
    maxdiff = 0.0
    for label, val, lo, hi, m, _ in runs[1:]:
        d = float(np.abs(m - base).max())
        maxdiff = max(maxdiff, d)
        print(f"max |moment difference| vs default start: {d:.3e}   (objective differs by "
              f"{abs(val - runs[0][1]):.2e})")

    scale = float(np.abs(base).max())
    print(f"\nmoment scale {scale:.6f}")
    print()
    if maxdiff < 1e-7:
        print("VERDICT: all starts converge to the SAME moments. The optimal face is a")
        print("single point (or ADMM canonically selects one). The 65 values are")
        print("canonical and identifying them in closed form is well posed.")
    else:
        print("VERDICT: different starts reach the SAME objective with DIFFERENT moments.")
        print(f"The optimal face is positive-dimensional ({maxdiff/scale:.1%} of scale apart).")
        print("So the stored 65 values are an artefact of ADMM's dynamics, not canonical,")
        print("and 'identify the moments' is the wrong target. The right target is to")
        print("exhibit ANY structured point in the face -- a feasibility problem with")
        print("freedom, which is what paley_ansatz.py should be searching.")

    dst = os.path.join(ROOT, "results", f"paley_face_{p}.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(dict(p=p, iters=iters, closed_form_sos2=closed, max_moment_difference=maxdiff,
                       moment_scale=scale,
                       runs=[dict(start=l, value=v, lo=lo, hi=hi, iters=n)
                             for l, v, lo, hi, _, n in runs]), f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
