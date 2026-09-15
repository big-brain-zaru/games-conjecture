"""group_analyze.py -- digest results/group_sweep*.jsonl: best gap shapes, retention by
carrier-irrep dimension, and the instances whose degree-2 gap survives degree 4."""
import json, sys, collections
import numpy as np

def load(path):
    rows = []
    try:
        for l in open(path):
            l = l.strip()
            if l:
                rows.append(json.loads(l))
    except FileNotFoundError:
        pass
    return rows

if __name__ == "__main__":
    rows = load("../results/group_sweep.jsonl") + load("../results/group_sweep_A5.jsonl")
    summ = load("../results/group_sweep_summary.jsonl") + load("../results/group_sweep_A5_summary.jsonl")
    # restarts can duplicate records: keep the tightest certificate per instance
    best = {}
    for r in rows:
        k = (r["group"], tuple(r["gens"]), tuple(r["signs"]))
        if k not in best or r["cert_width"] < best[k]["cert_width"]:
            best[k] = r
    rows = list(best.values())
    seen = set(); summ = [s for s in summ if not (s["group"] in seen or seen.add(s["group"]))]
    print(f"{len(summ)} groups summarised, {len(rows)} certified degree-4 solves\n")
    tot_inst = sum(s["n_instances"] for s in summ); tot_short = sum(s["n_shortlist"] for s in summ)
    print(f"stage 1 instances: {tot_inst}   shortlisted: {tot_short}   ruled out by filter: {tot_inst - tot_short}")
    und = sum(s["n_undecided"] for s in summ)
    print(f"undecided after stage 2 (certificate too wide or optimum unproved): {und}\n")
    # best C4 overall (rigorous lower bounds)
    rows_ok = [r for r in rows if r["proved"] and r["cert_width"] < 1e-4]
    rows_ok.sort(key=lambda r: -r["C4_lo"])
    print("top certified degree-4 gap shapes (tight certificate + proved optimum):")
    for r in rows_ok[:12]:
        print(f"  C4={r['C4_lo']:.6f}  C2={r['C2']:.4f}  ret={r['retention'] if r['retention'] is None else round(r['retention'],3)}"
              f"  {r['group']:14s} n={r['order']:3d} gens={r['gens']} signs={r['signs']} carrier dim {r['carrier_dim']}")
    # retention by carrier dimension
    print("\nretention (C4-1)/(C2-1) by dimension of the irrep carrying the degree-2 gap:")
    by = collections.defaultdict(list)
    for r in rows_ok:
        if r["retention"] is not None and r["C2"] > 1.02:
            by[r["carrier_dim"]].append(r["retention"])
    for d in sorted(by):
        v = np.array(by[d])
        print(f"  dim {d}: n={len(v):4d}  mean {v.mean():.4f}  median {np.median(v):.4f}  max {v.max():.4f}  "
              f"frac>0.05: {(v>0.05).mean():.3f}")
    # instances with real retention and real C2
    print("\ninstances with retention > 0.3 and C2 > 1.1:")
    hits = [r for r in rows_ok if r["retention"] and r["retention"] > 0.3 and r["C2"] > 1.1]
    hits.sort(key=lambda r: -(r["C4_lo"]))
    for r in hits[:15]:
        print(f"  {r['group']:14s} n={r['order']:3d} gens={r['gens']} signs={r['signs']}  C2={r['C2']:.4f} C4={r['C4_lo']:.6f} "
              f"ret={r['retention']:.3f} carrier dim {r['carrier_dim']}")
    # per-group table
    print("\nper group: best C4 (rigorous) | max U | carrier dims present")
    for s in sorted(summ, key=lambda s: -(s["best_C4_lo"] or 0))[:20]:
        print(f"  {s['group']:14s} n={s['order']:3d} irreps {s['irrep_dims']}  inst {s['n_instances']:5d} short {s['n_shortlist']:5d}"
              f"  bestC4 {s['best_C4_lo'] if s['best_C4_lo'] is None else round(s['best_C4_lo'],6)}  maxU {round(s['max_U'],3) if s['max_U'] else None}")
