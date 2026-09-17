"""Representation-theoretic block data for the Paley degree-4 form, for many p.

The frequency decomposition of section 13.5 is pure counting and needs no solver:

    Sym^2(E_min) = (+)_f V_f,   V_f = span{ psi_t * psi_s : t,s non-residues, t+s = f },

so dim V_f is just the number of unordered pairs of non-residues summing to f.
Square dilations act transitively on residues and on non-residues, so there are
three classes with dimensions

    d0 = dim V_0,   dres = dim V_f for f a residue,   dnon = dim V_f for f a non-residue,

and d0 + (p-1)/2 * (dres + dnon) = m(m+1)/2 with m = (p-1)/2.

These determine the shape of any Aut-equivariant form Q, hence the number of free
parameters available to the degree-4 extension, hence the dimension of the family
of extensions.  Computing them for many p is instant, which lets the family
dimension measured by the SDP at p = 29, 37, 41 be compared against a structural
prediction rather than guessed from three points.

An equivariant symmetric Q is determined by:
  * its action on V_0, which is a real circulant of size d0 = (p-1)/4  ->  d0 parameters;
  * a Hermitian form on one residue block          ->  dres^2 real parameters;
  * a Hermitian form on one non-residue block      ->  dnon^2 real parameters.

Run: python paley_blockdims.py [pmax]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def block_dims(p):
    qr = set(pow(a, 2, p) for a in range(1, p))
    non = [t for t in range(1, p) if t not in qr]
    S = set(non)
    dim = {f: 0 for f in range(p)}
    for i, t in enumerate(non):
        for s in non[i:]:                      # unordered, with repetition t = s
            dim[(t + s) % p] += 1
    dres = {dim[f] for f in range(1, p) if f in qr}
    dnon = {dim[f] for f in range(1, p) if f not in qr and f != 0}
    return dim[0], dres, dnon


def main():
    pmax = int(sys.argv[1]) if len(sys.argv) > 1 else 110
    rows = []
    print(f"{'p':>4} {'m':>4} {'D':>6} {'d0':>4} {'dres':>5} {'dnon':>5} "
          f"{'check':>6} {'Q params':>9} {'rank m(m-3)/2':>14} {'ker 2m':>7}")
    for p in range(13, pmax + 1, 4):
        if not is_prime(p) or p % 4 != 1:
            continue
        m = (p - 1) // 2
        D = m * (m + 1) // 2
        d0, dres_s, dnon_s = block_dims(p)
        if len(dres_s) != 1 or len(dnon_s) != 1:
            print(f"{p:4d}  non-uniform block dimensions: res {sorted(dres_s)}, "
                  f"non {sorted(dnon_s)}")
            continue
        dres, dnon = dres_s.pop(), dnon_s.pop()
        check = d0 + (p - 1) // 2 * (dres + dnon)
        qpar = d0 + dres ** 2 + dnon ** 2
        rows.append(dict(p=p, m=m, D=D, d0=d0, dres=dres, dnon=dnon,
                         check=check, q_params=qpar,
                         rank=m * (m - 3) // 2, kernel=2 * m))
        print(f"{p:4d} {m:4d} {D:6d} {d0:4d} {dres:5d} {dnon:5d} "
              f"{check:6d} {qpar:9d} {m*(m-3)//2:14d} {2*m:7d}")

    print()
    ok = lambda f: all(f(r) for r in rows)
    print("  d0 = (p-1)/4                       ", ok(lambda r: r["d0"] == (r["p"]-1)//4))
    print("  dres + dnon = d0 = (p-1)/4         ", ok(lambda r: r["dres"]+r["dnon"] == r["d0"]))
    print("  dres - dnon = d0 mod 2             ", ok(lambda r: r["dres"]-r["dnon"] == r["d0"] % 2))
    print("  dres = ceil((p-1)/8)               ", ok(lambda r: r["dres"] == -(-(r["p"]-1)//8)))
    print("  dnon = floor((p-1)/8)              ", ok(lambda r: r["dnon"] == (r["p"]-1)//8))
    print("  D = (p^2-1)/8                      ", ok(lambda r: r["D"] == (r["p"]**2-1)//8))
    print("  d0 + (p-1)/2*(dres+dnon) = D       ", ok(lambda r: r["check"] == r["D"]))
    print("  rank = D - 2m = (p-1)(p-7)/8       ",
          ok(lambda r: r["rank"] == r["D"] - 2*r["m"] == (r["p"]-1)*(r["p"]-7)//8))
    print()
    print("  So the whole block structure is closed form:")
    print("    dim V_0 = (p-1)/4,  dim V_res = ceil((p-1)/8),  dim V_non = floor((p-1)/8),")
    print("    dim Sym^2(E_min) = (p^2-1)/8,  ker Q = 2m = p-1  (one direction per nonzero")
    print("    frequency block),  rank M_even = (p-1)(p-7)/8.")
    print("  Feasibility starts exactly where dim V_non reaches 3, i.e. p >= 29.")

    dst = os.path.join(ROOT, "results", "paley_blockdims.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
