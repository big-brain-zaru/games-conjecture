"""circulant_full.py -- scan ALL Z_L-invariant Boolean 2Lin instances with <= kmax
active difference classes, every sign pattern, uniform weights, and report the
degree-4 gap shape C_4 with certified bounds (symmetry-reduced SoS + CP-SAT)."""
import warnings; warnings.filterwarnings("ignore")
import argparse, itertools, json, time
import numpy as np, torch
from circulant_sos import CirculantSoS
from circulant_scan import maxcut_cpsat

def edges_signs(L, gens, sg):
    seen, ed, sn = set(), [], []
    for g, b in zip(gens, sg):
        for v in range(L):
            u = (v + g) % L
            if u == v: continue
            k = (min(u, v), max(u, v))
            if k in seen: continue
            seen.add(k); ed.append(k); sn.append(b)
    return np.array(ed, dtype=np.int64), np.array(sn, dtype=np.int64)

def opt_cpsat(L, ed, sn, tl=8):
    from ortools.sat.python import cp_model
    mdl = cp_model.CpModel(); x = [mdl.NewBoolVar("") for _ in range(L)]
    mdl.Add(x[0] == 0); t = []
    for (a, b), s in zip(ed.tolist(), sn.tolist()):
        c = mdl.NewBoolVar("")
        if s < 0: mdl.Add(x[a] + x[b] == 1).OnlyEnforceIf(c); mdl.Add(x[a] == x[b]).OnlyEnforceIf(c.Not())
        else:     mdl.Add(x[a] == x[b]).OnlyEnforceIf(c); mdl.Add(x[a] + x[b] == 1).OnlyEnforceIf(c.Not())
        t.append(c)
    mdl.Maximize(sum(t))
    s_ = cp_model.CpSolver(); s_.parameters.max_time_in_seconds = tl; s_.parameters.num_workers = 8
    st = s_.Solve(mdl)
    return s_.ObjectiveValue()/len(ed), s_.BestObjectiveBound()/len(ed), s_.StatusName(st)=="OPTIMAL"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", default="9,11,13,15,17,19,21,23,25,27,29,31")
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--cp", type=float, default=8)
    ap.add_argument("--out", default="../results/circulant_full.jsonl")
    a = ap.parse_args()
    overall, argmax = 0.0, None
    for L in [int(x) for x in a.Ls.split(",")]:
        if L % 2 == 0: continue
        h = (L-1)//2
        S = CirculantSoS(L, device="cuda" if (torch.cuda.is_available() and L >= 23) else "cpu")
        bestL, t0, cnt = 0.0, time.time(), 0
        for k in range(1, a.kmax+1):
            for gens in itertools.combinations(range(1, h+1), k):
                for sg in itertools.product([-1, 1], repeat=k):
                    if all(s > 0 for s in sg): continue              # satisfiable, C = 0/0
                    S.set_instance(list(gens), signs=list(sg)); S.solve(iters=a.iters, tol=1e-11)
                    lo, hi = S.certified_bounds(); R = min(1.0, lo)
                    ed, sn = edges_signs(L, gens, sg)
                    o, ub, pr = opt_cpsat(L, ed, sn, tl=a.cp)
                    C = (1-ub)/max(1-R, 1e-12); cnt += 1
                    if C > bestL + 1e-9:
                        bestL = C
                        rec = {"L": L, "gens": list(gens), "signs": list(sg), "m": len(ed), "opt": o,
                               "opt_ub": ub, "proved": pr, "sos4_lo": lo, "sos4_hi": hi, "C4": C}
                        with open(a.out, "a") as f: f.write(json.dumps(rec)+"\n")
                        if C > overall: overall, argmax = C, rec
                        print("  "+json.dumps({kk:(round(vv,6) if isinstance(vv,float) else vv) for kk,vv in rec.items()}), flush=True)
        print(f"### L={L}: {cnt} instances, best C_4 = {bestL:.6f}  | overall {overall:.6f} "
              f"at L={argmax['L']} gens={argmax['gens']} signs={argmax['signs']}  ({time.time()-t0:.0f}s)", flush=True)
