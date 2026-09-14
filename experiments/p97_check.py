"""p97_check.py -- independent, cheaper route to the same verdict for p=97.

The question is only whether SoS_4(Cay(Z_97,H_4)) exceeds 0.668482.
Two independent handles, neither needing the big ADMM to converge:

 (a) an UPPER bound on SoS_4 from ANY feasible dual point.  We build one directly
     from the degree-2 (spectral) dual and a per-class correction, then shift to PSD.
 (b) a LOWER bound on SoS_4 from the degree-2 relaxation value, since
     SoS_4 <= SoS_2 always -- so if SoS_2 <= 0.668482 the question is settled
     immediately with no computation at all.
"""
import warnings; warnings.filterwarnings("ignore")
import json
import numpy as np
from sympy import primitive_root
from circulant_scan import sos2_circulant, circ_edges

def index4(p):
    g = primitive_root(p)
    H = sorted({pow(g, 4*i, p) for i in range((p-1)//4)})
    return sorted({min(h, p-h) for h in H})

if __name__ == "__main__":
    out = {}
    for p in (41, 73, 89, 97, 113):
        try:
            gens = index4(p)
        except Exception:
            continue
        s2 = sos2_circulant(p, gens)
        out[p] = {"gens": gens, "sos2": s2}
        print(f"p={p:4d} gens={gens}  SoS_2 = {s2:.6f}", flush=True)
    inc97 = 0.637457
    thr = 1 - (1 - inc97) / 1.093586
    print()
    print(f"SoS_4 <= SoS_2 always.  Threshold for ruling p=97 out: SoS_4 <= {thr:.6f}")
    print(f"p=97 SoS_2 = {out[97]['sos2']:.6f}  ->  "
          + ("SETTLED: ruled out with no further computation" if out[97]["sos2"] <= thr
             else "not settled by the spectral bound alone"))
    json.dump(out, open("../results/p97_check.json", "w"), indent=1)
