"""sweep2.py -- stage 2 of the complete sweep: run the certified degree-4 solver on
EVERY instance the cheap filter could not rule out, so the result is a complete
statement over the family rather than a sample."""
import warnings; warnings.filterwarnings("ignore")
import argparse, json, time
import numpy as np, torch
from circulant_sos import CirculantSoS
from circulant_scan import circ_edges, maxcut_cpsat
from sweep import sweep

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lmin", type=int, default=5); ap.add_argument("--Lmax", type=int, default=21)
    ap.add_argument("--kmax", type=int, default=3); ap.add_argument("--restarts", type=int, default=150)
    ap.add_argument("--record", type=float, default=1.093586)
    ap.add_argument("--iters", type=int, default=6000); ap.add_argument("--cp", type=float, default=15)
    ap.add_argument("--out", default="../results/sweep_complete.json")
    a = ap.parse_args()
    t0 = time.time()
    rows, shortlist = sweep(list(range(a.Lmin, a.Lmax + 1, 2)), kmax=a.kmax,
                            restarts=a.restarts, record=a.record, verbose=True)
    print(f"\nstage 1: {len(rows)} instances swept, {len(shortlist)} not ruled out "
          f"by C_4 <= (1-incumbent)/(1-SoS_2)   [{time.time()-t0:.0f}s]", flush=True)
    solvers, best, undecided, done = {}, None, [], []
    t1 = time.time()
    for i, r in enumerate(shortlist):
        L, gens = r["L"], r["gens"]
        if L not in solvers:
            solvers[L] = CirculantSoS(L, device="cpu")
        S = solvers[L].set_instance(gens); S.solve(iters=a.iters, tol=1e-12)
        lo, hi = S.certified_bounds()
        e = circ_edges(L, gens)
        opt, ub, pr = maxcut_cpsat(L, e, time_limit=a.cp)
        lowC = (1 - ub) / max(1 - min(1.0, lo), 1e-12)
        upC = (1 - opt) / max(1 - min(1.0, hi), 1e-12)
        rec = {**r, "opt": opt, "opt_ub": ub, "proved": pr, "sos4_lo": lo, "sos4_hi": hi,
               "C4_low": lowC, "C4_up": upC}
        done.append(rec)
        if upC > a.record + 1e-9:
            undecided.append(rec)
        if best is None or lowC > best["C4_low"]:
            best = rec
            print(f"  best so far: C_4 in [{lowC:.6f}, {upC:.6f}] at L={L} gens={gens}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(shortlist)} degree-4 solves  [{time.time()-t1:.0f}s]", flush=True)
    print(f"\nstage 2: {len(done)} degree-4 solves in {time.time()-t1:.0f}s")
    print(f"instances whose certified interval still allows beating {a.record}: {len(undecided)}")
    for r in sorted(undecided, key=lambda r: -r["C4_up"])[:15]:
        print("   " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                  for k, v in r.items() if k in ("L","gens","opt","proved","sos4_lo","sos4_hi","C4_low","C4_up")}))
    json.dump({"record": a.record, "n_swept": len(rows), "n_shortlist": len(shortlist),
               "n_solved": len(done), "n_undecided": len(undecided),
               "max_C4_low": max(r["C4_low"] for r in done) if done else None,
               "best": best, "undecided": undecided}, open(a.out, "w"), indent=1)
