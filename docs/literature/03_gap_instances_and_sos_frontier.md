# 03 — Integrality-gap instances, what solves them, and the sum-of-squares frontier

## 1. Khot–Vishnoi (2005) — the canonical basic-SDP gap **[source: KV §3.2, implemented]**

Parameters k, η; N = 2^k. Functions f : {−1,1}^k → {−1,1}; classes f ~ fχ_S; vertices = classes
(2^N/N of them), labels = subsets S ⊆ [k] ≅ F_2^k; noisy hypercube edges weighted
2·2^{−N} η^d (1−η)^{N−d}; edges with Hamming distance outside [ηN/2, 2ηN] dropped; constraint on the
bundle defined by (f, g) = ([P_i]χ_S, [P_j]χ_T): π(T⋆U) = S⋆U, i.e. **label(P_i) ⊕ label(P_j) = S⊕T**
— the game is Max-2Lin over F_2^k.
* Soundness opt ≤ 1/N^η (Bonami–Beckner: any density-1/N set of the noisy N-cube has 1−Φ ≤ N^{−η−η²}).
* Completeness: vectors u_{fχ_S}^{⊗2} ∈ R^{N²} satisfy (11)–(14) and all triangle inequalities
  (coordinates ±1/√N) with objective ≥ 1 − 9η.
* So SDP-with-triangle-inequalities ≥ 1−9η vs opt ≤ N^{−η}: gap (1−ε, k^{−Ω(ε)}) in alphabet size.
* Repo: `experiments/kv_instance.py`. k=3: 32 vertices, 8 labels, 1,472 edges; k=4: 4,096 vertices,
  16 labels, 1,425,408 edges. Basic SDP at k=4, η=0.1: **0.795002 ≤ SDP ≤ 0.795010** (GPU, 6 s), while
  the subcube labeling gives ≈ (1−η)^k = 0.656 and Bonami gives ≤ 0.758 **[computed]**. The
  asymptotic gap needs k ≳ 10 (N ≳ 1000), i.e. 2^{1000} vertices — the phenomenon is *not* observable
  by brute numerics; it must be attacked through symmetry (the automorphism group contains F_2^N ⋊ …).

## 2. Other basic/strong-SDP gaps

* Mohanty–Raghavendra–Xu STOC 2020, *Lifting sum-of-squares lower bounds: degree-2 to degree-4*:
  on random d-regular graphs and the Sherrington–Kirkpatrick model a degree-2 Max-Cut lower bound lifts
  to degree 4, giving degree-4 value ≥ ½ + (√(d−1)/d)(1 − ε − γ(ε)/√d). Asymptotic in n. **[known;
  measured at finite n in FINDINGS 11.2–11.3, where the Paley family attains it exactly from p = 29 and
  the sparse family does not attain it at any computable size]**
* Khot–Saket 2009, Raghavendra–Steurer 2009: gaps that survive super-constant rounds of Sherali–Adams
  on top of the SDP, for every CSP (translating Raghavendra's dictatorship-test machinery). **[known;
  KV §1.3]** Property: sub-metrics on super-constantly many points are ℓ1-embeddable.
* Khot–Popat–Saket 2010: *approximate* Lasserre gap — a UG instance with an approximate vector
  solution to t rounds; shows constant-round Lasserre cannot work if it is insensitive to small
  perturbations. **[source: Springer abstract]**
* BGHMRS 2012 "Making the long code shorter": short-code (Reed–Muller based) gap instances, quasi-
  polynomially smaller than KV; a small-set expander with many large eigenvalues (relevant to the
  ABS running-time question). Kane–Meka used it for 2^{Ω(√loglog n)} Sparsest-Cut gaps. **[source: KV §1.3; known]**
* AKKT 2014 "UG on the Hypercube": GW-SDP integrality gap for Max-2Lin(Z2) whose constraint graph is
  the hypercube Q_d itself; unusual because unsatisfiable hypercube instances have an unsatisfiable
  4-cycle, so no relaxation solution can have every edge contributing > 3/4 — the SDP witness must be
  *non-symmetric* across edges. Conjecture: **GW-SDP + triangle inequalities solves UG on Q_d in
  polynomial time**. I found no resolution in print. **[source: AKKT §1; search]**

## 3. What solves the known gaps

| Instance family | Solved by | Source |
|---|---|---|
| KV noisy cube | constant-degree SoS (BBHKSZ 2012); poly-time certified-SSE rounding (BBKSS 2021) | 1205.4484, 2006.09969 |
| Short code | constant-degree SoS (BBHKSZ); BBKSS (previous best nearly exponential) | same |
| Johnson graph | BBKSS 2021 (first poly-time); Bafna–Minzer 2023 (any constant completeness) | 2304.07284 |
| Grassmann graph, HDX | Bafna–Minzer 2023 (affine UG) | 2304.07284 |
| Hypercube constraint graph | 2^{n^{Ω(1)}} spectral; conjectured poly via triangle SDP; likely covered by BBKSS (certified SSE) | AKKT |
| Expanders / few large eigenvalues | AKKSTV 2008, Kolla 2010 | known |
| Abelian Cayley graphs of degree d (Sparsest Cut) | (1+ε)-approx in n^{O(1)}·exp(d/ε)^{O(d)} (d'Orsi–Jones–Ruotolo–Vadhan–Zhang 2024) | 2412.17115 |
| Semi-random | KMM 2011 | known |
| Random (uniform permutations) | value ≈ 1/k + o(1); SDP certifies; not a gap regime | known |

## 4. The SoS frontier — the precise open question

* **No SoS/Lasserre integrality gap for UG with completeness 1−ε is known at any constant degree ≥ 4.**
  Khot 2010: even 3 rounds might refute UGC. Khot–Moshkovitz 2014: not even a (1−ε, 1−Cε), C→∞
  Lasserre gap is known. Searches for 2020–2026 SoS lower bounds return only random-CSP / planted
  problems (coloring, densest-k-subgraph, independent set, PCA); nothing for near-satisfiable UG.
  **[source: quoted + searches, 13 Sep 2026]**
* Reasons it is hard to construct: (i) the natural symmetric constructions (KV, short code) are
  small-set expanders *certifiable by hypercontractivity*, which is itself SoS-provable (BBHKSZ);
  (ii) random instances have value far from 1; (iii) Khot–Moshkovitz's candidate (real code over
  Gaussian space, reduction from Tulsiani's Lasserre-hard random kCSP(PHLin)) has Lasserre completeness
  for t rounds but unproven soundness — the soundness reduces to Robust Gaussian Isoperimetry.
* **Khot's candidate Lasserre gap (ECCC TR14-142)**: Boolean 2Lin instance with a super-constant-round
  Lasserre solution of value 1−ε; conjectured integral value ≤ 1−Cε with C→∞; "we consider several
  examples and sketch an argument". Sections: real code + Gaussian noise test (§4), augmented with
  constraint test (§5), consistency test (§6), overall construction (§7), soundness against potential
  counterexamples (§8). This is an explicit object whose integral value is *unknown* — a computational
  target if a finite truncation can be built. **[source: TR14-142 §1–3]**
* Bafna–Minzer's framing: UG is easy on every graph with a certified global-hypercontractivity
  structure; PCP reductions therefore must produce constraint graphs *without* that structure. Nobody
  has exhibited an explicit family of near-satisfiable UG instances on which degree-4 SoS provably
  fails. Conversely nobody has proved degree-O(1) SoS succeeds on all instances.

## 5. Random-instance facts relevant to experiments **[known]**

* Uniform random permutations on a sparse random graph: opt = 1/k + O(1/√(kd)); the basic SDP value
  is close to opt (random instances are SDP-easy, cf. refutation thresholds for 2-XOR: spectral
  refutation at m ≫ n, Allen–O'Donnell–Witmer 2015 / Raghavendra–Rao–Schramm 2017).
* Planted instances with noise p: the SDP recovers the planted labeling for d ≫ 1/p²-type thresholds;
  semi-random (adversarial graph, random perms) also easy (KMM). So random ensembles are not where
  hardness lives; a hard distribution would itself be a research result (Khot survey item 1).

## 6. Dynamical-systems and other unconventional angles found

* Sahai–Gnanasekaran 2024 (arXiv:2404.16024): a family of dynamical systems whose equilibria are UG
  solutions; unsatisfiable instances give ergodic dynamics (proved); invariant-measure weight near
  optimal assignments scales polynomially / sub-exponentially / exponentially with the value gap
  (numerical); claims to "reproduce a hypothesized hardness plot". Conditional on their own conjectures.
* Yoshida 2026 (arXiv:2605.17760): tolerant testers for UG in the adjacency-list model with query
  complexity Õ(√m ρ^{−13/2} + n ρ^{−2}/√m) — sublinear certificates of low value.
* Levene–Paulsen 2025: group-based UG generalising XOR games; Z_3 case = 3-labelling of digraphs.

## Bibliography
1. Khot–Vishnoi JACM 2015. 2. Khot–Saket FOCS 2009. 3. Raghavendra–Steurer FOCS 2009 "Integrality gaps
for strong SDP relaxations of unique games". 4. Khot–Popat–Saket APPROX 2010. 5. Barak, Gopalan,
Håstad, Meka, Raghavendra, Steurer FOCS 2012 "Making the long code shorter". 6. BBHKSZ STOC 2012.
7. BBKSS STOC 2021. 8. Bafna–Minzer CCC 2024. 9. AKKT CJTCS 2015. 10. Khot–Moshkovitz ECCC TR14-142 and
2015 manuscript. 11. Heilman RSA 2020. 12. d'Orsi et al. arXiv:2412.17115. 13. Sahai–Gnanasekaran
arXiv:2404.16024. 14. Yoshida arXiv:2605.17760. 15. Tulsiani STOC 2009 (Lasserre gaps for CSPs).

## 7. Addendum (day 1, evening) — the exact BBHKSZ statements **[source: arXiv 1205.4484 §6.7–6.9]**
* Theorem 6.11: *level-4* fictitious random variables (degree-4 SoS) certify small-set expansion of the
  noise graph T_{1−ε} on {±1}^R: Ẽ⟨f, T f⟩ ≤ δ^{1+Ω(ε)} for density-δ sets. Hence degree-4 SoS certifies the
  KV soundness bound R^{−Ω(η)} for the raw instance U_{η,R} (whose label-extended graph is T_{1−η}).
* Theorem 6.12 / 6.13: the **level-8** SoS relaxation of the *gadget-composed* instances W_{ε,k}(U_{η,R})
  (KKMO/Raghavendra alphabet reduction of KV) and W'_{ε,k}(U'_{η,R}) (short code) has value ≤
  1/k^{Ω(ε)} + k^{O(log k)} R^{−Ω(η)}. Obtained from Theorem 6.9 with d = 4 plus Theorem 6.11.
* Consequence for Track A: the natural explicit candidates for a **degree-4** gap with completeness 1−ε are
  the composed instances W_{ε,k}(U) themselves — BBHKSZ's certificate for them is at level 8, and I found no
  later source lowering it to 4 (BBKSS 2021 gives "constant degree" without pinning 4). Their sizes
  (|V(U)|·k^R) put direct computation out of reach except for tiny R; symmetry reduction is the only route.
