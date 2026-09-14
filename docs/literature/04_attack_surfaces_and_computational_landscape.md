# 04 — Attack surfaces distilled from the survey, and the computational landscape

## A. Attack surfaces (each is a precise, unturned stone; ranked by how directly it bears on UGC)

**A1. The SoS-gap vacuum (proof side, computational).** No instance family with completeness 1−ε is
known on which degree-4 (or 6, 8) SoS fails; Khot (2010, 2014) calls this the most pressing question.
Every known candidate is SoS-easy *because* it is a certified small-set expander or globally
hypercontractive. Attack: search for instances where the basic SDP is (1−ε)-good but a degree-4 SoS
certificate does not exist, at sizes where degree-4 SoS is computable (tens of vertices, small
alphabets), using exact solvers for ground truth and symmetry to reach meaningful sizes. Negative
results ("degree-4 SoS is within X of opt on every instance in class C") are also new data.

**A2. Khot's candidate Lasserre gap (TR14-142) and the Khot–Moshkovitz real-code candidate.** An
explicit object whose integral value is unknown. Attack: build finite truncations of the real-code
construction (Gaussian space → discretised), compute the Lasserre value we can certify and the true
optimum by exact search / local search, and probe the Robust Gaussian Isoperimetry question
numerically (which odd, periodic functions pass the noise test with probability within a constant of
a periodised half-space?) — this is a variational problem that GPU optimisation can attack directly.

**A3. Richness ladder (BKM 2021).** Rich 2-to-1 ⇔ UGC. "F-rich" games for families F of pairings,
and a sequence of reductions increasing richness. Attack: quantify, for the Grassmann-based 2-to-1
instances that are provably hard, *how far from rich* their pairing distributions are (a
distributional distance per left vertex), and search for gadgets that increase richness while
preserving the gap — a finite, checkable combinatorial optimisation over small alphabets.

**A4. The hypercube conjecture (AKKT 2014).** "GW-SDP + triangle inequalities solves Max-2Lin(Z2) on
Q_d in polynomial time" — unresolved in print. Attack: compute exact opt (MaxSAT) and the triangle-SDP
value on Q_d instances, d ≤ 12–16 (4096–65536 vertices), including AKKT's own gap family; either find
a counterexample instance (triangle-SDP ≥ 1−ε, opt ≤ 1−Cε) or accumulate evidence. Each outcome is a
publishable computational result.

**A5. Completeness above 1/2.** The 2-to-2 machinery caps at 1/2 by the two-halves split. Attack:
measure on explicit small Grassmann-based 2-to-2 instances how the best *unique* sub-game value
behaves — is 1/2 tight for the specific instances or only for the generic argument? If a structured
choice of halves systematically beats 1/2 on the hard instances, that is a new lever (SAT/MaxSAT search
over choices of halves on small Grassmann graphs Gr_{k,ℓ} with k ≤ 6, ℓ ≤ 3).

**A6. Certified-SSE boundary.** BBKSS/Bafna–Minzer: UG is easy whenever low-degree SoS certifies SSE
via hypercontractivity. Attack: build the certification test (degree-4 SoS for the 2→4-norm /
hypercontractivity of the constraint graph) and run it on structured graph families (abelian and
non-abelian Cayley graphs, lifts, random regular, Grassmann, Johnson, hypercube, "foam"-like cubical
complexes) to *map* which graphs are not certifiable — those are the only place hardness can live.

**A7. Non-abelian and non-linear constraints.** KKMO shows linear over F_2^ℓ suffices for UGC, so
non-abelian structure is not needed for hardness — but it might be where *SoS gaps* are easier to
build (non-abelian Cayley graphs lack the Fourier structure that hypercontractivity proofs use).
Attack: UG instances over small non-abelian groups (S_3, Q_8, A_4, dihedral) on their Cayley graphs;
compare basic SDP, degree-4 SoS, exact opt.

**A8. Dynamical / statistical-physics probes.** Sahai–Gnanasekaran's ergodic-dynamics claims are
numerical and conditional; a controlled replication with exact ground truth would either support a new
hardness indicator or retire it.

## B. Computational landscape (what exists, what was built today, what is missing)

Existing public code for UG specifically: essentially none found (no maintained generator/solver
repo; papers use ad-hoc code). Tooling that exists and is installed here:

| Need | Tool | Status |
|---|---|---|
| Exact optimum, ≤ ~10^3 vertices × ~10 labels | PySAT RC2 + CaDiCaL 1.9.5 (`ug_core.maxsat_optimum`) | built, self-tested |
| Brute force ground truth | `ug_core.brute_force_optimum` | built |
| Basic UG SDP, exact | cvxpy + Clarabel (nk ≤ ~150) / SCS (nk ≤ ~500, slow) | built |
| Basic UG SDP at scale, certified | GPU block-coordinate ascent + eigenvalue dual certificate (`ug_sdp`) | built; 65k-node graph in 6 s, primal=dual to 1e-4 |
| Rounding | Gaussian, propagation, vectorised local search | built |
| Khot–Vishnoi instances | `kv_instance.py` (k=3,4 exact; k≥5 needs sampling/symmetry) | built |
| Degree-4 SoS (Lasserre level 2) for UG | — | **to build** (cvxpy for small; symmetry-reduced for structured) |
| Triangle-inequality SDP for Max-2Lin(2) on Q_d | — | **to build** (ADMM/projection on GPU; O(n³) triangles need separation) |
| Short-code / Reed–Muller instances | galois installed | **to build** |
| Grassmann / Johnson graphs, 2-to-2 instances | — | **to build** (F_2 subspace enumeration, k ≤ 8) |
| Hypercontractivity / 2→4-norm SoS certifier | — | **to build** |
| Real-code (Gaussian) candidate, finite truncation | torch | **to build** |
| Non-abelian group UG generator | sympy.combinatorics | **to build** |
| Symmetry reduction of SDP/SoS (invariant subspace) | numpy/scipy | **to build** (fast-matrix `symgroup.py` is the template) |
| Formal verification | Lean 4 / Mathlib: nothing UG-specific exists (no PCP, no hypercontractivity) | out of scope for now |

Solver notes: Clarabel allocates a dense Hessian for the PSD cone — 300×300 already needs 8 GB; use it
only for nk ≤ 150. SCS handles nk = 300 in ~5 min at 1e-7. The GPU block-ascent + dual certificate is
the workhorse; its dual bound is valid for *any* primal iterate (it is a feasible dual point after the
eigenvalue shift), so every reported SDP value is a certified interval.
