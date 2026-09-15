"""
paley_ansatz.py -- is there an optimal degree-4 pseudo-expectation on the Paley graph
that depends only on the Legendre pattern of a 4-set?

Fix the pair moments at the degree-2 optimum, y_d = (-1 - sqrt(p) chi(d))/(p-1), which forces
the objective to equal SoS_2.  Let the 4-set moments be a function of the invariant
(sorted Legendre symbols of the six differences, sorted residue-degree sequence of the four
points) -- at most a dozen unknowns -- and ask for the moment matrix to be PSD, maximising
its smallest eigenvalue.  A feasible point is an explicit, human-readable certificate that
SoS_4 = SoS_2 on P_p; infeasibility says the extension needs finer arithmetic.
"""
import sys, json, itertools
import numpy as np, cvxpy as cp

def run(p, solver="SCS"):
    chi = np.zeros(p, dtype=int)
    for a in range(1, p):
        chi[pow(a, 2, p)] = 1
    chi = np.where(chi == 1, 1, -1); chi[0] = 0
    y2 = lambda d: 1.0 if d % p == 0 else (-1 - np.sqrt(p) * chi[d % p]) / (p - 1)
    basis = [()] + [(i,) for i in range(p)] + list(itertools.combinations(range(p), 2))
    D = len(basis)
    patt = {}
    def key4(S):
        S = sorted(S)
        diffs = [(b - a) % p for a, b in itertools.combinations(S, 2)]
        pat = tuple(sorted(int(chi[d]) for d in diffs))
        deg = tuple(sorted(sum(chi[(b - a) % p] == 1 for b in S if b != a) for a in S))
        return (pat, deg)
    # collect patterns
    for S in itertools.combinations(range(p), 4):
        k = key4(S)
        if k not in patt:
            patt[k] = len(patt)
    t = cp.Variable(len(patt))
    s = cp.Variable()
    # build M = M0 + sum_k t_k A_k  (sparse structure via index lists)
    M0 = np.zeros((D, D))
    rows = {k: ([], []) for k in range(len(patt))}
    for i, a in enumerate(basis):
        for j in range(i, D):
            b = basis[j]
            S = set(a) ^ set(b)
            if len(S) == 0:
                M0[i, j] = M0[j, i] = 1.0
            elif len(S) == 2:
                u, v = sorted(S); M0[i, j] = M0[j, i] = y2(v - u)
            elif len(S) == 4:
                k = patt[key4(S)]
                rows[k][0].append(i); rows[k][1].append(j)
    A = []
    for k in range(len(patt)):
        Ak = np.zeros((D, D))
        Ak[rows[k][0], rows[k][1]] = 1.0
        Ak = Ak + Ak.T - np.diag(np.diag(Ak))
        A.append(Ak)
    M = M0 + sum(t[k] * A[k] for k in range(len(patt)))
    prob = cp.Problem(cp.Maximize(s), [M - s * np.eye(D) >> 0, t >= -1, t <= 1])
    prob.solve(solver=solver, verbose=False, **({"max_iters": 200000, "eps": 1e-9} if solver == "SCS" else {}))
    lam = float(s.value)
    print(f"p={p}: {len(patt)} pattern unknowns, D={D}: max lambda_min = {lam:+.6e}  -> "
          f"{'FEASIBLE: SoS4 = SoS2 by a Legendre-pattern pseudo-expectation' if lam > -1e-7 else 'infeasible in this ansatz'}", flush=True)
    vals = {str(k): float(t.value[i]) for k, i in patt.items()}
    for k, v in vals.items():
        print(f"    {k}: {v:+.6f}")
    return {"p": p, "lambda_min": lam, "n_patterns": len(patt), "values": vals}

if __name__ == "__main__":
    out = [run(int(x)) for x in sys.argv[1:]]
    json.dump(out, open("../results/paley_ansatz.json", "w"), indent=1)
