"""quartic.py -- All 4-regular circulants Cay(Z_L, {+-a,+-b}), all-anti Max-Cut.
Both degree-4 champions found so far (Cay(Z_9,{1,2}) and Cay(Z_17,{1,4})) are in
this family, so it is the sharpest test of whether C_4 grows at fixed degree."""
import warnings; warnings.filterwarnings("ignore")
import argparse, itertools, json, time
import numpy as np, torch
from circulant_sos import CirculantSoS
from circulant_scan import circ_edges, maxcut_cpsat, sos2_circulant

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", default="9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41")
    ap.add_argument("--iters", type=int, default=9000)
    ap.add_argument("--cp", type=float, default=30)
    ap.add_argument("--out", default="../results/quartic.jsonl")
    a = ap.parse_args()
    best, arg = 0.0, None
    for L in [int(x) for x in a.Ls.split(",")]:
        if L % 2 == 0: continue
        h = (L - 1) // 2
        dev = "cuda" if (torch.cuda.is_available() and L >= 27) else "cpu"
        S = CirculantSoS(L, device=dev)
        bl, ba, t0 = 0.0, None, time.time()
        for aa, bb in itertools.combinations(range(1, h + 1), 2):
            S.set_instance([aa, bb]); S.solve(iters=a.iters, tol=1e-11)
            lo, hi = S.certified_bounds()
            e = circ_edges(L, [aa, bb])
            opt, ub, pr = maxcut_cpsat(L, e, time_limit=a.cp)
            C4 = (1 - ub) / max(1 - min(1.0, lo), 1e-12)
            if C4 > bl: bl, ba = C4, (aa, bb, opt, ub, lo, hi, pr, sos2_circulant(L, [aa, bb]))
        if ba:
            aa, bb, opt, ub, lo, hi, pr, s2 = ba
            rec = {"L": L, "gens": [aa, bb], "opt": opt, "opt_ub": ub, "proved": pr,
                   "sos2": s2, "sos4_lo": lo, "sos4_hi": hi, "tight": abs(hi - lo) < 1e-5,
                   "C2": (1 - opt) / max(1 - s2, 1e-12), "C4": bl, "s": round(time.time() - t0, 1)}
            if bl > best: best, arg = bl, rec
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            with open(a.out, "a") as f: f.write(json.dumps(rec) + "\n")
        print(f"### L={L}: best 4-regular C_4 = {bl:.6f}  | overall {best:.6f} at L={arg['L']} gens={arg['gens']}", flush=True)
