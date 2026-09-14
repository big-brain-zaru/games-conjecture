# Findings

A computational attack on the Unique Games Conjecture, 13–14 September 2026.

Every statement is tagged **[computed]** (produced and verified here, with the result file named),
**[reasoning]** (an argument, stated as such), or **[source]** (read from a primary source, cited in
`docs/literature/`). Nothing is claimed without a certificate, an exhaustive enumeration, or an
explicit label saying which it is. Where I got something wrong during the work the error is kept in
place and marked, because the corrections are part of the result.

---

## 1. The reformulation

Khot and Moshkovitz set this goal (ECCC TR14-142, section 1.2) **[source]**:

> "show a (1−ε, 1−C·ε) integrality gap for C rounds of the Lasserre SDP for the Boolean 2Lin problem
> where C → ∞ … At present however, we do not even know a (1−ε, 1−C·ε) gap with C → ∞, even for
> general Unique Games, and even as Lasserre integrality gap."

For an instance I of Boolean 2Lin (Max-Cut with signs and weights) and a relaxation R_d of degree d,
define the **gap shape**

  C_d(I) = (1 − opt(I)) / (1 − R_d(I)).

**Dilution identity [reasoning].** Let I be the weighted disjoint union of a perfectly satisfiable part
(weight 1−w) and a core (weight w). Both opt and R_d are affine in the component weights across a
disjoint union, so opt(I) = 1−w+w·opt(core) and R_d(I) = 1−w+w·R_d(core), hence

  **C_d(I) = C_d(core) for every w in (0,1].**

A gap ratio achieved at *any* completeness therefore transports to completeness 1−ε for *every* ε > 0,
and

  **a (1−ε, 1−C·ε) degree-d Lasserre gap with C → ∞ exists ⟺ sup_I C_d(I) = ∞.**

That turns the open question into a scalar optimisation over instances, which is what this project
measures. **[computed]** the identity was checked numerically on diluted instances.

**Calibration at degree 2.** sup C_2 = ∞, attained by odd cycles:
C_2(C_L) = (1/L)/((1−cos(π/L))/2) → 4L/π². **[computed]** the pipeline reproduces 2.094427 (L=5),
2.885096 (7), 3.684826 (9), 4.488559 (11), 5.294418 (13), 6.101542 (15), matching the closed form to
every printed digit.

---

## 2. The key inequality, and how the search became complete

**C₄ ≤ C₂ [reasoning].** Degree-4 sum-of-squares is at least as tight as degree 2, so SoS₄ ≤ SoS₂ and

  **C₄ ≤ C₂ = (1 − opt)/(1 − SoS₂) ≤ (1 − incumbent)/(1 − SoS₂) =: U(I)**

for any cut that has actually been exhibited. Both ingredients are cheap: for a circulant, SoS₂ is the
closed-form spectral value (exact because the graph is vertex-transitive), and a cut comes from local
search in milliseconds. **No optimality proof and no degree-4 solve are needed to rule an instance out.**

This converts the search from *extensive* to *complete*: sweep U over a family, rule out everything at
or below the current record, and spend the expensive certified solver only on the shortlist
(`experiments/sweep.py`, `experiments/sweep2.py`).

**The trade-off it exposes [computed].** Large C₄ needs large C₂, but the degree-2 champions are
exactly what degree 4 destroys. Writing the retention ρ = (C₄−1)/(C₂−1):

| instance | C₂ | C₄ | ρ |
|---|---|---|---|
| odd cycle C_L | 4L/π² → ∞ | 1.000000 | 0 |
| K₅ | 1.066667 | 1.066667 | **1** (degree 4 adds nothing on K₅) |
| Cay(Z₉, {±1,±2}) | 1.333333 | 1.074139 | 0.222 |
| Cay(Z₄₁, H₄) | 1.160503 | 1.093586 | 0.583 |

Cycles have unbounded C₂ and zero retention; complete graphs have full retention and C₂ → 1. The record
sits where the curves cross, so the objective for a further search is the product, not either factor.

---

## 3. What was measured

### 3.1 Solver validation, which is what makes the rest meaningful **[computed]**

The degree-4 solver is neither broken nor trivially exact, and it reproduces analytic values:

| instance | opt | SoS₂ | SoS₄ | C₄ |
|---|---|---|---|---|
| K₅ Max-Cut | 0.600000 | 0.625000 (= analytic 5/8) | 0.625000 | **1.066667** |
| K₇ Max-Cut | 0.571429 | 0.583333 | 0.583333 | 1.028571 |
| C₅ through C₁₅ | (L−1)/L | 0.9045 to 0.9891 | = opt | 1.000000 |
| Petersen | 0.800000 | 0.833333 | = opt | 1.000000 |
| random 3-regular n=16, 2 seeds | 0.9167 / 0.8750 | 0.9223 / 0.9058 | = opt | 1.000000 |
| random 4-regular n=14 | 0.785714 | 0.856362 | = opt | 1.000000 |
| Paley₅, Paley₉, Paley₁₃ | 0.8 / 0.6667 / 0.6667 | 0.9045 / 0.75 / 0.6919 | = opt | 1.000000 |

The K₅ row is the classical fact that the metric polytope of K₅ exceeds the cut polytope (the
pentagonal inequality); degree-4 SoS does not capture it. For complete graphs C₂ = C₄ = 1 + 1/(n(n−2)),
so **K_n is the wrong family**.

### 3.2 Exhaustive over small instances **[computed]**

An instance is a pair of signs and weights on K_n. Switching a vertex changes neither opt nor SoS_d nor
the weights, so every sign pattern normalises to "all edges at vertex 0 are +1", leaving an arbitrary
graph on n−1 vertices; relabelling reduces that to isomorphism classes. For n−1 ≤ 7 the networkx graph
atlas enumerates **every** instance shape. Weights are then optimised by exponentiated gradient on the
exact Danskin gradient, from multiple starts including every "drop one vertex" start.

| n | shapes (exhaustive) | max C₄ | is this the true max? |
|---|---|---|---|
| 5 | 11 | **1.066667** (K₅ all-anti) | yes |
| 6 | 34 | 1.066667 (K₅ embedded) | yes |
| 7 | 156 | 1.032258 directly; at least 1.066667 by embedding | no, only the top 25 shapes were weight-optimised |

The n = 7 row shows the limit clearly: the shape enumeration is complete, but weight optimisation ran
only on the 25 shapes that looked best at uniform weights, and K₅ inside K₇ was not among them. Since
the maximum is non-decreasing in n, the honest entry is "at least 1.066667".

### 3.3 A complete sweep of the circulants **[computed]**

`sweep.py` then `sweep2.py`, over every circulant Cay(Z_L, S) with L odd in [5, 19] and |S| ≤ 3:

| stage | count | time |
|---|---|---|
| instances swept with the cheap filter of section 2 | 374 | 7 s |
| ruled out outright by the filter | 16 | — |
| shortlist given the certified degree-4 solver | 358 | 1,501 s |
| left undecided by a 5,000-iteration certificate | 9 (one graph up to the multiplier action) | — |
| settled by re-running that graph at 250,000 iterations | 9 | 160 s |

The nine were all Cay(Z₁₉, {±a,±b}) with {a,b} in one multiplier orbit; at 250,000 iterations the
certificate closes to width 0 (residuals 1e−13) with SoS₄ = opt = 0.736842 exactly, so C₄ = 1.000000.

**Every instance in the family is therefore accounted for, with no gaps**, and the maximum over it is
**1.081179**, at Cay(Z₁₇, H₄). This is what "complete rather than extensive" means in practice: not that
every instance was solved, but that every instance was either solved or rigorously excluded.

### 3.4 The certified champions **[computed]**

Each has a tight certificate (certified lower equals upper) and a proved optimum:

| n | instance | opt | SoS₄ | C₄ |
|---|---|---|---|---|
| 5 | K₅ | 0.600000 | 0.625000 | 1.066667 |
| 9 | Cay(Z₉, {±1,±2}) | 0.666667 | 0.689674 | 1.074139 |
| 17 | Cay(Z₁₇, H₄) | 0.764706 | 0.782373 | 1.081179 |
| 41 | **Cay(Z₄₁, H₄)** | 0.702439 | 0.727904 | **1.093586** |

The n = 41 entry is the maximum over everything searched in this project.

H_k is the multiplicative subgroup of index k in the units mod p, which contains −1, so the Cayley graph
is undirected, exactly when 2k divides p−1. The search selected the **index-4** family on its own: index
2 (the Paley graphs) gives at most 1.017, index 6 at p = 37 gives exactly 1.000000 with a closed
certificate, and index (p−1)/2 (the odd cycle) gives exactly 1. **It is the index, not the graph
degree**: the index-6 case at p = 37 has degree 6, between the two index-4 winners, and degree 4 is
exactly tight on it. The quartic-residue structure is the obvious suspect and is not something I have
explained.

---

## 4. A claim I made and then refuted

**What I first found.** The four champions were increasing and a two-parameter fit described them almost
perfectly: C₄ ≈ 1.046057 + 0.012686·ln n, R² = 0.99764, residuals all below 0.00082.

**Why it was wrong.** C₄ is *decreasing* in opt and *increasing* in SoS₄, so with opt in [incumbent,
dual bound] and SoS₄ in [lo, hi] the rigorous interval is

  C₄ in [ (1 − opt_dual_bound)/(1 − SoS₄_lo) , (1 − incumbent)/(1 − SoS₄_hi) ].

The upper end needs the certified SoS₄ **upper** bound. I first wrote it with the lower bound, which is
not a valid upper bound on C₄. The cheap filter of section 2 gives the verdicts instantly and correctly:

| p | incumbent | SoS₂ | C₄ ≤ (1−inc)/(1−SoS₂) | verdict against 1.093586 |
|---|---|---|---|---|
| 41 | 0.702439 | 0.743593 | 1.160503 | consistent, this *is* the record |
| 73 | 406/657 = 0.617960 | 0.633870 | **1.043453** | ruled out |
| 89 | 0.627171 | 0.653049 | **1.074587** | ruled out |
| 97 | 742/1164 = 0.637457 | 0.656862 | **1.056552** | ruled out |

So the family peaks at p = 41 and falls back; the log-linear fit was an artefact of a short increasing
subsequence of four points. **The refuted fit is left in place above** because it is exactly the failure
mode this kind of search invites.

**The cost of not seeing the filter earlier.** I spent a 350,000-iteration ADMM run and two 900-second
CP-SAT runs establishing for p = 73 and p = 89 what two lines of arithmetic give in milliseconds, and had
a 400,000-iteration run going on p = 97 when the closed-form degree-2 value already settled it. The
inequality SoS₄ ≤ SoS₂ is immediate; I had not thought to bound the stronger relaxation's gap shape with
the weaker relaxation.

**Also tried and discarded.** Reformulating max-cut optimality as the refutation of "cut at least
incumbent plus one". On p = 41, where the optimisation mode proves the optimum in 34 seconds, the
refutation mode returned UNKNOWN after 90 seconds, and the multiplier-based symmetry break written for it
was not verified sound. Both were removed rather than reported.

---

## 5. A group-theoretic generalisation of Khot–Vishnoi

**[computed, `experiments/group_ug.py`; the self-test reproduces Khot–Vishnoi exactly]**

Let G be a finite group, H normal in G with |H| = N, and μ a symmetric measure on G vanishing on H.
Vertices are the cosets G/H, labels are H, and a labelling is a **transversal** T of H in G with
val(T) = the sum of μ(x⁻¹y) over unordered pairs of distinct x, y in T.

*This is a unique game.* Fix base representatives a_A; for cosets A, B put c = a_A⁻¹a_B. As the pair
(h_A, h_B) ranges over H × H the difference h_A⁻¹ c h_B takes each value z in cH exactly N times, and for
fixed z the relation is the bijection h_B = c⁻¹ h_A z, well defined because H is normal. So each triple
(A, B, z) is one unique constraint of weight μ(z), and the label-extended graph is exactly the Cayley
graph of G with connection measure μ, with the H-cosets as blocks. Hence **val(T) = 1 − Φ(T) at density
1/N**: soundness *is* small-set expansion of a Cayley graph.

*Completeness.* If ρ is an orthogonal representation of G whose character vanishes on H minus the
identity, equivalently ρ restricted to H is a multiple of the regular representation, then ψ = χ_ρ/dim ρ
gives SDP vectors by tensor-squaring: block orthogonality is exactly the condition on ρ, inner products
are ψ(·)² ≥ 0, and the objective is the sum of μ(z)ψ(z)².

*Khot–Vishnoi is the abelian case*: G is the elementary abelian group of order 2^N with N = 2^k, H is the
simplex code of characters, every nonzero word of which has weight N/2 — precisely why ψ = 1 − 2·wt/N
vanishes off the identity — and μ is the truncated noise. **[computed]** the generic builder reproduces
the k = 3 game vertex for vertex (32 vertices, 8 labels, 1,472 constraints, identical values on random
labellings), and the Khot–Vishnoi ψ passes the representation test with Gram minimum eigenvalue −1.2e−14.

**Why this is the interesting direction [source].** Every known gap instance has an *abelian* G (noisy
cube, short code), and its soundness is degree-4 certifiable because hypercontractivity of the abelian
noise operator is provable in sum-of-squares: Barak, Brandão, Harrow, Kelner, Steurer and Zhou 2012,
Theorem 6.11 certifies small-set expansion of the noise graph at level 4, and Theorems 6.12 and 6.13
certify the composed Khot–Vishnoi and short-code instances at level 8. Bafna and Minzer (CCC 2024) state
that graphs *without* a certified global-hypercontractivity structure are where hardness must live.
Non-abelian G is that region, and this module constructs those games. The census
(`experiments/group_census.py`) runs but only reaches 8 vertices, where degree 4 is exact and every
instance gives C₄ = 1, so it is built and validated but not yet informative.

---

## 6. The Khot–Vishnoi instance, exactly

**[computed]** `kv_instance.py`, `kv_transversal.py`, `kv_invariant_sdp.py`.

* **Transversal identity.** The bitmask difference between the two sides of an edge bundle is constant
  over the whole bundle, so every bundle has weight N times the noise weight, and the optimum is a
  maximum-weight transversal of the simplex code (32 variables of domain 8 at k = 3). Verified against
  the generic evaluator on random labellings.
* **Exact basic-SDP values by symmetry.** The SDP reduces to a linear program over GL(k,2)-orbits of
  Boolean functions with Fourier positivity: 20 orbits at k = 3, 92 at k = 4.

  | k | η | exact basic-SDP value | independent certified numeric interval |
  |---|---|---|---|
  | 3 | 0.2 | 0.6157095 | [0.615710, 0.615757] |
  | 3 | 0.3 | 0.3948238 | [0.394824, 0.394826] |
  | 4 | 0.1 | 0.7950017 | [0.795002, 0.795010] |
  | 4 | 0.2 | 0.5929504 | not computed numerically |

  A lesson worth keeping: the *affine* group is **not** a symmetry, because a translation maps a class to
  the class of minus the translate and so mixes two classes. The affine version returned 0.7813 at
  k = 4, η = 0.1 and was refuted by the certified numeric interval. Only the linear group acts.
* **Optimum.** The subcube transversal gives 0.439189 at k = 3, η = 0.2 and 0.234424 at η = 0.3, and
  local search cannot improve it. CP-SAT proves upper bounds 0.445524 and 0.301211 after 2,400 seconds,
  so the optimum is **not closed**. The HiGHS mixed-integer model is far worse, bound 0.83 after 1,500
  seconds.
* Scale note: the gap is asymptotic in k, the SDP tending to 1−9η against an optimum at most 2^(−ηk). At
  k = 3 and 4 there is no gap at all, so Khot–Vishnoi is a *fixture* here, not an experiment.

---

## 7. The hypercube conjecture of Agarwal, Kindler, Kolla and Trevisan

They construct a Goemans–Williamson integrality gap on the hypercube and conjecture that **that SDP plus
triangle inequalities solves unique games on the hypercube** **[source: CJTCS 2015]**. **[computed]**
`experiments/hypercube_track.py`, `results/hypercube_akkt.jsonl`: at dimensions 4 and 5 the
Goemans–Williamson gap is present, C₂ between 1.11 and 1.60, while the certified degree-4 value equals
the optimum to solver tolerance. Since the optimum is at most the triangle-inequality SDP, which is at
most the degree-4 value, this *sandwiches the triangle-inequality SDP to exactness* on these instances,
which is evidence for their conjecture at dimension at most 5. At dimensions 6 and 7 the certificates are
not tight enough to conclude.

---

## 8. Tooling

| module | what | validation |
|---|---|---|
| `ug_core.py` | instance class, label-extended graph, perfect-completeness propagation, generators, brute force, MaxSAT via RC2 and CaDiCaL, mixed-integer via HiGHS, CP-SAT in one-hot and XOR encodings, vectorised local search | brute force equals MaxSAT equals CP-SAT on random instances |
| `ug_sdp.py` | the basic unique-games SDP: cvxpy, a **GPU block-coordinate ascent** with exact Procrustes block updates and Gauss–Seidel by graph colouring, and a **dual certificate** from any primal iterate | primal equals dual to 1e-4; matches Clarabel and SCS to five decimals; a 65,536-node label-extended graph in 6 seconds |
| `ug_sos.py` | level-2 Lasserre for any alphabet, degree-2 and degree-4 sum-of-squares for the Boolean case, via cvxpy | exact at n = 4; on C₅ gives 0.904508 and 0.8; sandwiched between the optimum and the SDP |
| `sos_gpu.py` | **ADMM degree-4 sum-of-squares with certified two-sided bounds**, over-relaxation and residual balancing | matches SCS to 1e-6 on K₉ in 3.8 seconds; K₃₀ in 7 seconds |
| `circulant_sos.py` | **cyclic-symmetry-reduced degree-4 sum-of-squares**: L blocks of size 1+(L−1)/2 instead of one block of 1+L+C(L,2); at L = 81 that is 81 blocks of 41 instead of one of 3322 | eigenvalue multisets identical to the dense spectrum to 1e−8; adjoint verified by pairing; reproduces the dense certified interval on six instances |
| `sweep.py`, `sweep2.py` | the two-stage complete sweep: cheap filter, then certified degree 4 | 374 circulants swept in 7 seconds |
| `kv_instance.py`, `kv_transversal.py`, `kv_invariant_sdp.py` | exact Khot–Vishnoi, the transversal reformulation, the symmetry-reduced exact SDP | total weight matches the closed form; transversal matches the generic evaluator; the linear program matches the certified numerics |
| `group_ug.py`, `group_census.py` | the group-quotient framework and its census | reproduces Khot–Vishnoi exactly; the non-abelian transversal evaluator matches |
| search drivers | `c4_max.py`, `gap_search.py`, `gap_search_gpu.py`, `hypercube_track.py`, `fs_track.py`, `circulant_scan.py`, `circulant_opt.py`, `circulant_full.py`, `quartic.py`, `genpaley.py`, `paley4.py`, `paley_index.py`, `p73.py` | the odd-cycle closed form is reproduced exactly |

**Honest solver limits.** The dense degree-4 certificates are tight to about 500 rows, roughly 30
vertices. At about 2,000 rows (hypercubes of dimension 6 and 7, sphere instances on 30 to 40 vertices)
20,000 iterations leave the dual bound above 1, so those rows carry a valid *lower* bound on the
relaxation only and their C₄ entries are not usable. The symmetry-reduced solver has no such limit on
cyclic instances.

---

## 9. Reproduction

`python experiments/reproduce.py` recomputes every headline number from scratch. It does not read them
from result files.

**50 checks, 0 mismatches, 47 seconds.** It covers the degree-2 odd-cycle calibration, the degree-4
solver validation on K₅, K₇, C₅ and Petersen, the certified champions at 9 and 17 vertices with their
certificate widths, the agreement of the symmetry-reduced solver with the dense one, the Khot–Vishnoi
construction (size, total weight, subcube values, three exact symmetry-reduced SDP values), the group
framework reproducing Khot–Vishnoi, and primal equals dual for the certified basic-SDP solver.

---

## 10. Scope

What this project establishes:

* The reformulation of section 1 and the inequality of section 2, both elementary and both checked.
* A certified maximum gap shape of **1.093586** over everything searched, at the Cayley graph of the
  quartic residues modulo 41.
* Complete, not merely extensive, coverage of the circulant families swept, in the sense of section 2.
* A construction framework for non-abelian unique games that reproduces Khot–Vishnoi as a special case.

What it does **not** establish:

* Nothing here bears on whether the supremum of C₄ is finite. If it were finite, degree-4
  sum-of-squares would certify that a relaxation value of 1−ε forces an optimum of at least 1−Cε for an
  absolute constant C, beating the Goemans–Williamson bound of 1−O(√ε) and therefore **refuting** the
  Unique Games Conjecture, via Khot, Kindler, Mossel and O'Donnell. The measured range is far too small
  to distinguish a bounded supremum from a slowly growing one.
* The degree-2 champion only becomes convincing well past the sizes where degree-4 certificates are
  affordable, so a flat degree-4 curve up to 41 vertices is weak evidence for anything.
* No new hardness result, no new algorithm, and no progress on the conjecture itself.

---

## 11. Next

1. **Non-abelian instances at informative size.** The group framework is built and validated but the
   census only reaches 8 vertices, where degree 4 is exact by treewidth. It needs the same isotypic
   block-diagonalisation that the cyclic solver does, generalised to an arbitrary finite group through
   its character table. That is the highest-leverage piece of engineering left.
2. **Explain the index-4 phenomenon.** Quartic-residue Cayley graphs are the only family that produced
   anything; index 2 and index 6 produce nothing. A structural reason would say where else to look.
3. **Close the Khot–Vishnoi optimum at k = 3**, currently bracketed in [0.439189, 0.445524]. The 168
   further automorphisms from the linear group are unused in the search.
4. **A real max-cut solver.** Proving optimality is what capped the direct approach at about 41 vertices
   before the filter made it unnecessary, and it would be needed again for any family the filter does
   not settle.
