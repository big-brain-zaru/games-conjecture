# 01 — Foundations, equivalent forms, and the algorithmic side

Compiled 13 September 2026 from primary sources (PDFs saved in `docs/papers/`). Every statement below
is tagged: **[source: …]** = read directly from the paper; **[known]** = standard fact I am confident of
but did not re-read today; **[computed]** = verified in this repo.

## 1. The object

A Unique Game U = (G(V,E), [N], {π_e}, wt): graph, alphabet [N], bijection π_e on every edge, weights
summing to 1. A labeling λ satisfies e{v,w} iff λ(v) = π_e(λ(w)). opt(U) = max weight satisfied.
**[source: Khot–Vishnoi JACM, Def. 3.1]**

**UGC (Khot 2002).** For every η, ζ > 0 there is N = N(η, ζ) such that it is NP-hard to distinguish
opt(U) ≥ 1−η from opt(U) ≤ ζ. **[source: KV Conj. 3.2; KMS 2018 Conj. 1.2]**

Equivalent forms (all with explicit reductions):

| Form | Statement | Source |
|---|---|---|
| Linear over F_2^ℓ | constraints T_ij x_i ⊕ T'_ij x_j = b_ij with invertible T, or even **T = identity**, i.e. x_i ⊕ x_j = b_ij over F_2^ℓ | KMS 2018 footnote 1, citing KKMO |
| 2Lin(F) over any finite field | x_i − x_j = b, field of constant size | Khot–Moshkovitz 2015, Def. 1–2 |
| Weak UGC (Khot survey Conj. 3.1) | gap (1−ε, 1−Ω(√ε·Γ(1/ε))) for some Γ→∞ suffices; amplified by Rao's parallel repetition (1−Ω(ε²))^k | Khot 2010 survey, Thm 7.3 |
| Rich 2-to-1 | 2-to-1 games where each left vertex's induced partition distribution is uniform over all pairings; **equivalent to UGC** | Braverman–Khot–Minzer ITCS 2021, Thm 8 |
| SSE-type variant | a variant of the Small-Set-Expansion hypothesis is equivalent to UGC; SSEH ⇒ UGC | Raghavendra–Steurer 2010 [known] |
| Boolean 2Lin, gap (1−ε, 1−C·ε) | proving hardness of this with C→∞ for general fields would give UGC via parallel repetition; **not even known as a Lasserre gap** | Khot–Moshkovitz 2015 §1.2 |

What is FALSE or ruled out:
* UGC with perfect completeness is false: propagation solves value-1 instances in polynomial time
  **[computed: `ug_core.satisfiable_perfectly`]**.
* Strong parallel repetition (exponent < 2 in Rao's theorem) is false: odd-cycle game (Raz 2008), so
  UGC is NOT known to be equivalent to Max-Cut (1−ε, 1−(2/π)√ε) hardness. **[source: Khot survey §7]**
* The strengthened CKKRS form (k = log n, ε = δ = 1/(log n)^Ω(1)) is refuted by CMM. **[source: CMM §1]**

## 2. The label-extended graph and small-set expansion

Replace each vertex by N copies and each edge by the matching defined by π_e. A labeling λ ↔ the set
S'_λ = {(v, λ(v))} of density 1/N, and **val(λ) = 1 − Φ(S'_λ)** where Φ is edge expansion.
**[source: KV Def. 3.4, eq. (15)]** So "UG has low optimum" ⇐ "every density-1/N set in the
label-extended graph expands almost perfectly". This is the bridge to SSE and to every spectral
algorithm. Implemented as `UniqueGame.label_extended_adjacency()` **[computed]**.

## 3. The basic SDP (Khot 2002 / Feige–Lovász)

Vectors v_1..v_N per vertex. Maximize Σ_e wt(e) (1/N) Σ_i ⟨v_{π_e(i)}, w_i⟩ subject to
(11) Σ_i ⟨v_i,v_i⟩ = N; (12) ⟨v_i,v_j⟩ = 0 for i≠j; (13) ⟨v_i,w_j⟩ ≥ 0 for all v,w,i,j;
(14) Σ_{i,j} ⟨v_i,w_j⟩ = N for all v,w. **[source: KV Fig. 4]**
Normalised per-block trace 1 in this repo. Constraints (13)–(14) are optional flags in `ug_sdp.sdp_cvxpy`.
The certified GPU solver `ug_sdp.BlockAscent` + `dual_certificate` gives primal = dual to 1e-4 on
every test so far, including the 65,536-node KV label-extended graph in 6 s **[computed]**.

## 4. Polynomial-time algorithms (worst case), value 1−ε given

| Algorithm | Guarantee | Notes | Source |
|---|---|---|---|
| Khot 2002 | 1 − O(k² ε^{1/5} √log(1/ε)) | first | CMM Fig. 1 |
| Trevisan 2005 | 1 − O((ε log n)^{1/3}) (improvable to 1−O(√(ε log n))) | spectral/SDP, n-dependent | CMM Fig. 1 |
| Gupta–Talwar 2006 | 1 − O(ε log n) | LP rounding | CMM Fig. 1 |
| **CMM 2006** | (a) Ω(min(1, 1/√(ε log k)) (1−ε)² (k/√log k)^{−ε/(2−ε)}); (b) 1 − O(√(ε log k)) | SDP + orthogonal separators; near-optimal under UGC | CMM §1 |
| Chlamtac–MM 2006 | 1 − O(ε √(log n log k)) | | [known] |
| KKMO 2004/2007 (matching hardness under UGC) | (1−ε) vs 1/k^{ε/(2−ε)}; and (1−ε) vs 1 − √(2/π)√(ε log k) + o(1) | Thm 1.2–1.3 quoted in CMM | CMM |
| Integrality gap of the basic SDP | SDP 1−ε but opt ≤ O(k^{−ε/9}), analysis yields O(k^{−ε/4+o(ε)}) | KV instance | CMM §1 |

So the basic SDP is **exactly** characterised: CMM's rounding matches the KV gap up to lower-order
terms, and under UGC nothing polynomial beats it (Raghavendra 2008 makes this universal for all CSPs).

## 5. Subexponential and hierarchy algorithms

* **Arora–Barak–Steurer 2010.** exp(k n^ε)-time algorithm: given value 1−ε^c (c an absolute constant,
  statement quoted: value ≥ 1−ε⁶ ⇒ output ≥ 1−O(ε log(1/ε)), time 2^{n^{O(1/ε)}}... the precise
  exponent forms vary by version) **[source: search summary of ABS; to be pinned from the JACM version]**.
  Mechanism: threshold-rank decomposition — a graph with few eigenvalues ≥ 1−η is easy (Kolla-style
  enumeration over the eigenspace), and any graph can be partitioned into such pieces losing few edges.
  Consequence: UGC-type hardness cannot be "exponentially hard"; any reduction proving UGC must have
  polynomial blow-up n^{Ω(1/ε)}-ish, which is why "linear blow-up" PCP techniques are inherently
  inapplicable (Barak blog 2018). **[source: Barak post summary]**
* **Barak–Raghavendra–Steurer 2011.** Degree-d SoS rounds via global correlation; SoS captures ABS.
  **[known; Barak 2018: "SoS does capture the known subexponential algorithms for unique games"]**
* **BBHKSZ 2012.** Constant-degree SoS certifies unsatisfiability of the noisy-cube (KV) and short-code
  instances, separating SoS from the weaker hierarchies (SA+SDP) that need ω(1) rounds. Also: a good
  approximation to the 2→4 norm would refute SSEH. **[source: arXiv 1205.4484 abstract]**
* **Bafna–Barak–Kothari–Schramm–Steurer 2021.** Polynomial-time UG algorithm (1−ε vs δ, ε,δ independent
  of alphabet) whenever a low-degree SoS proof certifies the small-set expansion of the constraint
  graph via a hypercontractive inequality; covers noisy hypercube, short code, Johnson graph; also
  works when non-expanding small sets are "characterised" by a low-degree proof. Rounds low-entropy
  SoS solutions via a global potential. **[source: arXiv 2006.09969]**
* **Bafna–Minzer 2023 (CCC 2024).** Affine UG over globally hypercontractive graphs (Johnson, Grassmann,
  HDX): completeness can be an arbitrarily small constant, soundness independent of the parameters
  that grow with alphabet. Their own remark: graphs lacking such a structural characterisation are
  what PCP reductions must use. **[source: arXiv 2304.07284]**
* **Kolla 2010; AKKSTV 2008; KMM 2011.** Expanders / few large eigenvalues are easy; semi-random
  instances are easy. **[known]**
* **Quantum.** Kempe–Regev–Toner 2008: entangled-prover value of a unique game is approximable in
  polynomial time (via SDP) **[known]**; Levene–Paulsen 2025: quantum-assisted value near 1 forces a
  perfect deterministic strategy **[source: arXiv 2506.18644]**; Mousavi–Spirig 2024 propose a quantum
  UGC; Culf et al. 2025: quantum smooth label cover is undecidable. None bears on the classical UGC.

## 6. Where the algorithmic frontier actually sits (September 2026)

1. Every explicitly known hard-looking family (noisy hypercube/KV, short code, Johnson, Grassmann,
   HDX, hypercube constraint graph) is solved by constant-degree SoS or by certified-SSE rounding.
2. No polynomial-time algorithm is known for general instances beyond CMM; no SoS lower bound at
   *any* constant degree ≥ 4 is known for UG with completeness 1−ε — the two facts are two sides of
   the same ignorance (see doc 03).
3. The 2-to-2 theorem gives NP-hardness for completeness 1/2 − ε, a regime where the subexponential
   algorithm already exists, and KMS state that known algorithmic attacks work "equally well" whether
   completeness ≈ 1 or ≈ 1/2 — i.e. those attacks are provably not on the path to refuting UGC.
   **[source: KMS 2018 App. B]**

## Bibliography (this file)
1. S. Khot, N. Vishnoi. The UGC, integrality gap for cut problems and embeddability of negative type
   metrics into ℓ1. JACM 62(1) 2015; arXiv:1305.4581. `docs/papers/khot_vishnoi_2005_jacm.pdf`
2. S. Khot. On the Unique Games Conjecture (survey), CCC 2010. `docs/papers/khot2010_ugc_survey.pdf`
3. M. Charikar, K. Makarychev, Y. Makarychev. Near-optimal algorithms for unique games. STOC 2006.
   `docs/papers/cmm2006_near_optimal_ug.pdf`
4. S. Khot, D. Minzer, M. Safra. Pseudorandom sets in Grassmann graph have near-perfect expansion.
   ECCC TR18-006. `docs/papers/kms2018_pseudorandom_grassmann.pdf`
5. M. Braverman, S. Khot, D. Minzer. On Rich 2-to-1 Games. ITCS 2021. `docs/papers/bkm2021_rich_2to1.pdf`
6. S. Khot, D. Moshkovitz. Candidate Hard Unique Game. 2015. `docs/papers/khot_moshkovitz2015_candidate_hard_ug.pdf`
7. B. Barak et al. Hypercontractivity, SoS proofs, and their applications. STOC 2012, arXiv:1205.4484.
8. M. Bafna, B. Barak, P. Kothari, T. Schramm, D. Steurer. Playing unique games on certified small-set
   expanders. STOC 2021, arXiv:2006.09969.
9. M. Bafna, D. Minzer. Solving unique games over globally hypercontractive graphs. CCC 2024, arXiv:2304.07284.
10. S. Arora, B. Barak, D. Steurer. Subexponential algorithms for unique games and related problems.
    FOCS 2010 / JACM 2015. https://www.boazbarak.org/Papers/ssesubexp.pdf
11. B. Barak. Unique Games Conjecture – halfway there? Windows on Theory, 10 Jan 2018.
12. N. Agarwal, G. Kindler, A. Kolla, L. Trevisan. Unique Games on the Hypercube. CJTCS 2015,
    arXiv:1405.1374. `docs/papers/akkt2014_ug_hypercube.pdf`
13. R. Levene, V. Paulsen. Unique Games and Games Based on Groups. arXiv:2506.18644.
