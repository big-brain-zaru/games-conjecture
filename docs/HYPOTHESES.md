# Hypotheses (first generation, 13 September 2026)

Each hypothesis carries a decisive test. Outcomes are recorded below the hypothesis as they arrive,
tagged [computed] with the result file. Undecided instances are never counted as negatives.

## H1 — Degree-4 SoS has (1−ε, 1−Cε) gaps with C > C_GW on small unique games
Rationale: Laurent (2003) shows the Lasserre rank of the cut polytope of K_n is ≥ ⌈n/2⌉, so degree-4
inexact Max-2Lin(2) objectives exist for n ≥ 6; what nobody has checked is whether any of them has the
*shape* Khot–Moshkovitz ask for (relaxation 1−ε, integral 1−Cε with C large) rather than a gap deep in
the unsatisfiable regime. C_GW = 2/π·(1/√ε) is the degree-2 constant; a degree-4 C exceeding the
degree-2 C at the same ε would be the first datum of its kind.
Test: adversarial instance search (Track A) on fixed graphs (K_n, Q_d, Paley, Cayley, random d-regular)
with n ≤ 40, k = 2; objective C(I) = (1−opt(I))/(1−SoS_4(I)) subject to SoS_4 ≥ 1−ε; exact opt by
brute force / MILP. Decisive: max C_4 vs max C_2 curves in ε. If C_4 ≈ 1 everywhere → H1 refuted at these
sizes (recorded as a negative map).

## H2 — On the hypercube Q_d, GW-SDP + triangle inequalities is within O(ε) of opt (AKKT conjecture)
Test: AKKT's own gap family and adversarial Q_d instances (d ≤ 10); compute GW, triangle-SDP, SoS_4,
opt. Decisive: a single instance with triangle-SDP ≥ 1−ε and opt ≤ 1−Cε, C ≫ 1, refutes; systematic
C ≈ O(1) supports.

## H3 — The subcube labeling is optimal for the Khot–Vishnoi game U_{k,η} for small η
Rationale: the label-extended graph is the noisy N-cube; the labeling set must be a transversal of
the χ_S-cosets; a subcube of codimension k is such a transversal with stability (1−η)^k (before
distance truncation); the Bonami bound N^{−η−η²} is not attained by any set (it is not tight for
sets of size 2^{N−k}).
Test: exact opt for k=3 (MILP running), k=4 by symmetry-reduced exact search; compare with the
subcube value on the same truncated instance. [computed so far: k=3, η=0.2 subcube value 0.43919
(truncated); local search does not improve it.]

## H4 — The exact basic-SDP value of U_{k,η} is 1 − c·η + o(η) with c ≪ 9
Test: symmetry-reduced basic SDP for k = 3..8 (the invariant SDP under F_2^N ⋊ Aut has dimension
polynomial in N); fit c. [computed so far: k=4, η=0.1: SDP ∈ [0.795002, 0.795010]; the KV vector
solution is far from optimal at this size.]

## H5 — Non-abelian Cayley constraint graphs resist degree-4 SoS certification of hypercontractivity
more often than abelian ones, and host larger SoS_4 gaps for UG
Test: Track D certifier on Cay(G,S) for G ∈ {Z_2^d, Z_n, S_3, S_4, A_4, Q_8, D_n, SL(2,p)} with
matched degree/size; then Track A search restricted to those graphs. Decisive: gap statistics by group
class with certificates.

## H6 — Sahai–Gnanasekaran's ergodic-dynamics hardness indicator does not separate SoS-easy from
SoS-hard instances
Test: replicate their dynamics on instances with exact ground truth and known SoS_4 gap; correlate
their invariant-measure statistic with C(I). (Low priority; retire if uninformative.)

## H7 — Robust Gaussian Isoperimetry (Khot–Moshkovitz) fails numerically in low dimension
Rationale: if some odd, coordinate-periodic f : R^n → {±1} not influenced by O(1) periodised
half-spaces passes the Gaussian noise test within a constant of a half-space, KM's soundness route is
blocked and their candidate may not be a gap. Test: GPU variational search over discretised periodic
functions in n ≤ 6 dimensions; report the best pass probability ratio vs the half-space. Decisive
either way at the level of numerical evidence.

## Results ledger
(appended as results arrive; see docs/LOG.md)

### Ledger, 13 Sep 2026 (day 1) — all [computed], files in results/
- **Pipeline validation.** Track A search seeded with odd cycles C5/C7/C9 on K16 reproduces the analytic
  degree-2 gap ratios 2.0944 / 2.8851 / 3.6848 exactly (`gapgpu_K16_d2.log`, `gap_search_gpu.jsonl`).
- **H1 (degree-4 gaps), sizes n ≤ 16, k = 2:** every instance visited by the search (random starts,
  cycle seeds, annealing over signs/weights on K8, K9, K16) has certified C_4 = 1.000 (degree-4 SoS
  exact). Consistent with Laurent-type gaps having C ≈ 1. **No evidence for H1 at n ≤ 16**; the
  informative regime starts where degree 4 is far from exact (n ≥ 30, Feige–Schechtman seeds running).
- **H2 (AKKT hypercube), d = 4, 5:** Δ[k,d] instances: C_2 = 1.48–1.60 (GW gap present), degree-4 SoS
  value equals opt within solver tolerance (C_4 ∈ [0.95, 1.00], loose only because ADMM at 4000
  iterations is not tight). Since opt ≤ triangle-SDP ≤ SoS_4, this already implies the triangle-SDP is
  exact on these instances (sandwich), supporting H2 at d ≤ 5. d = 6, 7 running (`hypercube_akkt.jsonl`).
- **H3/H4 (KV):** k=3, η=0.2: subcube labeling 0.43919 (truncated instance); exact optimum via MILP
  running. k=4, η=0.1: SDP ∈ [0.795002, 0.795010].
- **H4 (exact KV SDP value) — DONE for k = 3, 4 [computed, `kv_invariant_sdp.py`, `results/kv_invariant_sdp_values.json`].**
  The basic SDP of U_{k,η} reduces exactly to an LP over GL(k,2)-orbits of Boolean functions
  (20 orbits at k=3, 92 at k=4) with Fourier positivity over F_2^N. Values: k=3: η=0.2 → 0.6157095,
  η=0.3 → 0.3948238; k=4: η=0.1 → 0.7950017, η=0.2 → 0.5929504. Independent certified GPU numerics
  on the full instances: [0.615710, 0.615757], [0.394824, 0.394826], [0.795002, 0.795010]. Lesson
  logged: the affine group is NOT a symmetry (translations mix a class with its negative); the AGL
  version gave 0.7813 and was refuted by the certificate. k=5 needs GL(5,2)-orbit enumeration of 2^32 functions.
- **H3 (KV optimum), k=3 [computed, `results/kv3_eta*_exact_cpsat.json`, `kv3_exact_*.log`]:** CP-SAT (12 workers,
  exact integer weights 4^{3-d} resp. 3^d 7^{8-d}, 3000 s) returns the subcube labeling value 0.439189 (η=0.2)
  and 0.234424 (η=0.3) as best found, with proven upper bounds 0.60726 and 0.31618 — optimality NOT proven.
  HiGHS MILP is useless here (incumbent 0.357 / 0.179, bound 0.83 / 0.85 after 1500 s). The basic SDP
  (0.6157 / 0.3948) is also far above; closing k=3 exactly needs symmetry-reduced degree-4 SoS or a
  transversal-structured branch-and-bound (the optimum is a max-noise-stability transversal of the 32
  cosets of the character subgroup in F_2^8, symmetry F_2^8 ⋊ GL(3,2) of order 43008).
- **H1 at n = 16 [computed, `results/gapgpu_K16_d4.log`, `gap_search_gpu.jsonl`]:** all 8 starts (C5..C15 seeds,
  2 random), 200 annealing steps each: certified C_4 = 1.000 (degree-4 SoS exact on every instance visited).
- **Solver note:** at D ≥ 2000 rows (Q_6, Q_7, FS n ≥ 30) the ADMM certificates are loose after 1500–4000
  iterations (e.g. Q_7 upper bound 3.97 — a feasible dual point far from optimal); re-running with 20000 iterations.
- **H1 sharpened [reasoning from BBHKSZ §6]:** degree-4 SoS certifies the raw KV instances (Thm 6.11), so the
  raw noisy-cube family cannot host a degree-4 gap; the composed KKMO instances W_{ε,k}(U) are certified only
  at level 8 (Thm 6.12). The explicit degree-4 question is therefore: level-4 value of W_{ε,k}(U) — accessible
  only through symmetry-reduced SoS. Immediate feasible step: the translation-symmetric level-2 Lasserre for
  the raw KV k=3 game (256 Fourier blocks of ~126 rows instead of one 31,993-row block), which should also
  certify the k=3 optimum (subcube 0.439189) if it is tight.

## H8 — sup_I C_d(I) is the Khot–Moshkovitz question, verbatim  [reasoning + computed]
**Scale invariance.** For a disjoint union I = (satisfiable part, weight 1−w) ⊕ (core, weight w),
opt(I) = 1−w+w·opt(core) and R_d(I) = 1−w+w·R_d(core) (both relaxation and optimum are linear in the
weights across components), so C_d(I) = (1−opt)/(1−R_d) = C_d(core) for every w. **C_d is invariant
under dilution**, so any gap ratio achieved at any completeness can be transported to completeness
1−ε for every ε > 0. Therefore

  "there is a (1−ε, 1−C·ε) degree-d Lasserre gap for Boolean 2Lin with C → ∞"  ⟺  sup_I C_d(I) = ∞.

That is exactly the goal Khot–Moshkovitz state (ECCC TR14-142 §1.2: "we do not even know a
(1−ε, 1−C·ε) gap with C → ∞ … even as Lasserre integrality gap"). At d = 2 the supremum is infinite:
the odd cycle C_L gives C_2 = (1/L)/((1−cos(π/L))/2) → 4L/π² **[computed: C5 2.0944, C7 2.8851,
C9 3.6848 — matches the closed form exactly]**. **At d = 4 the supremum is unknown, and this is the
computational target.**

**Verification of the degree-4 solver on known objects [computed]:**
| instance | opt | SoS_2 | SoS_4 | C_4 |
|---|---|---|---|---|
| K5 max-cut | 0.600000 | 0.625000 (= analytic 5/8) | 0.625000 | **1.0667** |
| K7 max-cut | 0.571429 | 0.583333 | 0.583333 | **1.0286** |
| C5 / C7 / C9 max-cut | (L−1)/L | 0.9045 (C5) | = opt | 1.0000 |
| Petersen max-cut | 0.800000 | 0.833333 | = opt | 1.0000 |
| random 3-regular n=16 | 0.9167 / 0.8750 | 0.9223 / 0.9058 | = opt | 1.0000 |
| random 4-regular n=14 | 0.785714 | 0.856362 | = opt | 1.0000 |

So degree-4 SoS is *not* trivially exact (K5, K7 are genuine gaps) but it IS exact on every
treewidth-2 graph tried (cycles) and on the sparse cubic/quartic instances. For K_n all-anti the
closed form is C_2 = C_4 = 1 + 1/(n(n−2)) → 1, so complete graphs are the wrong family.
**Consequence: my day-1 Track A search was mis-designed** — its sparsification moves drove instances
into the treewidth-2 region where C_4 ≡ 1. Corrected search: dense/expanding seeds, weights-only
moves, no edge deletion.

## H9 — Where the degree-4 gaps actually live (day 2)  [computed]
The search selects **generalised Paley circulants**: Cay(Z_p, H) with H the multiplicative subgroup of
index k in Z_p^* (H = −H). Best found so far:

| instance | n | degree | opt | SoS₄ (certified) | C₂ | C₄ |
|---|---|---|---|---|---|---|
| **Cay(Z₁₇, {±1,±4})**, index 4 | 17 | 4 | 13/17 = 0.764706 | 0.782373 (lo = hi) | 1.720149 | **1.081179** |
| Cay(Z₉, {±1,±2}) | 9 | 4 | 2/3 | 0.689674 (lo = hi) | 1.333333 | 1.074139 |
| Cay(Z₁₇, {±1,±2,±4,±8}), index 2 (Paley) | 17 | 8 | 0.647059 | 0.652957 | 1.038359 | 1.016996 |
| Cay(Z₂₉, …), index 2 (Paley) | 29 | 14 | 0.600985 | 0.614018 | 1.033773 | 1.033765 |
| Cay(Z_p, {±1}) = odd cycle, index (p−1)/2 | any | 2 | (p−1)/p | = opt | 4p/π² | 1.000000 |

Reading: the degree-2 champion is the *smallest* connection set (the cycle) and the degree-4 champion
is an *intermediate* one (degree 4 out of p−1). Both extremes are degree-4-exact: the cycle has
treewidth 2, and the dense Paley/complete cases have C₄ → 1. So the degree-4 gap lives in a middle
band, and so far it does not grow with n.

## H10 — C_4 grows along index-4 generalised Paley circulants  [computed, OPEN]
Data: p=17 (degree 4) C_4 = 1.081179; p=41 (degree 10) C_4 >= 1.093516, both with opt proved optimal
by CP-SAT and SoS_4 certified from below. Fixed-degree families are flat, index-2 (Paley) and
index-(p-1)/2 (cycle) are worse. Test: p = 73, 89, 97, 113 (running). Decisive: if C_4 keeps rising
past ~1.15 the family is the first evidence of the growth Khot-Moshkovitz ask for; if it turns over,
the negative map stands and the next lever is a non-abelian G in the group framework (doc FINDINGS 3).

**H10 RESULT: REFUTED [computed].** p=73 gives C_4 <= 1.028181 and p=89 gives C_4 <= 1.042499, both
rigorous (opt >= incumbent cut, SoS_4 >= a feasible pseudo-moment), both below the p=41 value 1.093586.
The family peaks at p=41 and falls back; the log-linear fit was an artefact of four increasing points.
The negative map stands: max C_4 = 1.093586 over everything searched. Next lever is the non-abelian
group framework.

## H11 — The degree-4 gap is a property of the subgroup INDEX, not the graph degree  [computed]
Evidence: index 4 at p=17 (degree 4) and p=41 (degree 10) both give C_4 > 1.08, while index 6 at p=37
(degree 6, between them) gives C_4 = 1.000000 with a closed certificate, and index 2 (Paley) gives at
most 1.017 at p=17 and exactly 1 at p=13. Test: p = 89, 97, 113 at index 4 (all p = 1 mod 8), and
index 8 at p = 97, 113 (p = 1 mod 16). Decisive: if index 4 keeps producing gaps and other indices keep
producing none, the quartic-residue structure is the mechanism and should be explained, not just
measured.

## H12 — C_4 <= C_2, so the search can be made complete rather than extensive  [computed, USED]
SoS_4 <= SoS_2 gives C_4 <= (1-incumbent)/(1-SoS_2) with both ingredients cheap (closed-form spectral
value + local-search cut). This settles p=73, 89, 97 of the index-4 Paley family in milliseconds
(upper bounds 1.043, 1.075, 1.057, all below the record 1.093586), closing H10 completely, and turns
a family scan into a two-stage complete sweep (experiments/sweep.py, sweep2.py). Corollary framing:
large C_4 needs large C_2, but the degree-2 champions (odd cycles) have retention (C_4-1)/(C_2-1) = 0
while complete graphs have retention 1 and C_2 -> 1; the record sits at the crossover.

## H13 — Complete sweep of the small circulants  [computed, CLOSED]
Over every Cay(Z_L,S) with L odd in [5,19] and |S| <= 3: 374 instances swept in 7 s, 16 ruled out by
the filter, 358 solved with certificates in 1501 s, 9 left undecided at 5,000 iterations and settled at
250,000 (Cay(Z_19,{1,2}) and its multiplier orbit: SoS_4 = opt = 0.736842 exactly, width 0, so C_4 = 1).
Maximum over the family: 1.081179 at Cay(Z_17,H_4). No gaps.

## H14 — Retention is governed by the carrier eigenspace, not by the group  [computed, OPEN]

Generalising H9/H11: for a vertex-transitive signed instance the degree-2 gap is carried by the top
eigenspace V of the signed adjacency (in the group picture, by the irreps achieving the maximum).
Barak–Brandão–Harrow–Kelner–Steurer–Zhou show degree-4 SoS refutes the degree-2 gap when V is
2→4 hypercontractive.  Quantified prediction: retention ρ = (C₄−1)/(C₂−1) increases with
H(V) = n · max_{f∈V, ‖f‖₂=1} Σ f_v⁴ and with dim V.
Test: `hypercontract.py` (group sweep records + recomputed circulants).  First 59 non-abelian records:
mean retention 0.023 for H < 2.5, 0.09 for 2.5 ≤ H < 8; corr(ρ, log H) = 0.26.  Circulants, 321
instances L ≤ 19: corr(ρ, H) = 0.42, corr(ρ, mult) = 0.39, driven by disjoint-K₅ instances (retention 1 by
the K_n identity); mean retention 0.024 at multiplicity 2, 0.054 at multiplicity 4.
**Verdict: WEAKLY SUPPORTED, not decisive.**  The non-abelian sweep past order 21 was stopped (no output in 18 h).

## H15 — On the Paley graphs the degree-2 gap survives degree 4 completely  [computed, being certified]

Stored generalised-Paley data (`genpaley.jsonl`) show SoS₄ = SoS₂ = ½ + (1+√p)/(2(p−1)) to five
decimals for Max-Cut on P_p at p = 29, 37, 41, 53, 61, 73, whereas p = 13 (SoS₄ = opt = 2/3) and
p = 17 (SoS₄ = 0.65296 < 0.66010) lose part or all of the gap.  This is the finite-n, exact form of the
Mohanty–Raghavendra–Xu degree-2 → degree-4 lifting on a deterministic pseudo-random graph.
Tests: `paley_lift.py` (two-sided certificates at 1e-9), `paley_moments.py` (structure of the optimal
degree-4 moments by Legendre pattern), `wick_lift.py` (the naive scaled Wick lift is NOT PSD on any
of them — the extension is subtler than the Gaussian formula).
Caveat: C₂(P_p) → 1, so this family gives retention 1 of a vanishing gap; no growth.

## H16 — The gap shape of Max-Cut on cubic expanders (C₂ ≈ 2.7) survives degree 4  [computed, OPEN]

MRX: on random d-regular graphs degree-4 SoS is no better than the spectral bound up to
(1 − ε − γ(ε)/√d), i.e. asymptotically in d and n.  If this held at d = 3 the degree-4 gap shape
would be ≈ (1 − mc₃)/(½ − √2/3) ≈ 2.7, far above the small-instance record 1.0936.
Tests: `expander_gap.py` (random cubic graphs, certified SoS₄ + proved optimum); the A5 / S5 / PSL(2,7)
cubic Cayley graphs through the symmetry-reduced sweep.  Result so far: n = 12, 16, 20 random cubic
graphs are degree-4 EXACT (C₄ = 1 while C₂ = 1.04–1.37).  Small n is not the MRX regime; the question
is at what n (if any) the lift starts to hold.

## H17 — The best gap shape on n vertices, g₄(n), can be learned instead of guessed  [computed, tool]

Alternating certified ascent (`weight_ascent.py`, `class_ascent.py`): fixed feasible pseudo-expectation
⇒ the ratio is a linear-fractional program in the weights (cuts by enumeration or CP-SAT separation);
fixed weights ⇒ re-solve degree 4.  Every LP value is a certified lower bound and the scheme is
monotone; kicks escape first-order stationary points.  Findings: random dense instances on ≤ 7 vertices
are degree-4 exact (the ascent cannot even start); K₇ is a strict local optimum at 36/35; Z₉ recovers the
known 1.074139 from the all-minus seed by one kick; Cay(Z₄₁,H₄) is a first-order stationary point of C₄
over all 40 (class, sign) weights.  K_n itself: SoS₄(K_n) = SoS₂(K_n) = n/(2(n−1)) for odd n, so
C₄(K_n) = (n−1)²/(n(n−2)) → 1 (`kn_gap.py`).


## H18 — A trade-off: at degree 4 the degree-2 gap survives only where it vanishes  [computed, OPEN]

Synthesis of H15–H17 and section 11 of FINDINGS.  Full retention has been observed only on dense
pseudo-random graphs (Paley p ≥ 29), where C₂ → 1; large C₂ (sparse, high girth: 1.9 at n = 32) is
entirely refuted by degree 4 at every accessible size; the record instances (index-4 generalised Paley,
1.0936) sit in between with partial retention and are locally optimal under weight perturbation.
Prediction: no instance with n ≤ 50 has C₄ > 1.2.  Refutation would be a single certified instance.
Consequence if true: the Khot–Moshkovitz question at degree 4 is decided only in the asymptotic
regime of the lifting theorems, beyond exact computation.

Ledger, day 3: H15 CONFIRMED at p = 29, 37, 41 (certified), H16 REFUTED at n ≤ 32 (degree 4 exact on
all sparse graphs tested), H17 tools validated (Z₉, Z₁₁, Z₄₁ reproduced), H14 weakly supported (partial data).
