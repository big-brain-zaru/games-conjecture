# 02 — The hardness side: what is proven, how, and what blocks 1/2 → 1−ε

Tags as in doc 01. Primary sources: KMS 2018 (ECCC TR18-006), BKM 2021, Khot survey 2010, Khot–Moshkovitz 2015.

## 1. The hardness landscape for unique-type games (September 2026)

| Problem | Completeness | Soundness | Status | Source |
|---|---|---|---|---|
| Unique Games, any alphabet | 1 | — | in P (propagation) | [computed] |
| Unique Games | C(δ)·δ | δ | NP-hard, C(δ)→∞ but C(δ)δ→0 (Feige–Reichman) | Khot survey §9 |
| **Unique Games** | **1/2 − ε** | **ε** | **NP-hard** (corollary of 2-to-2 theorem) | KMS 2018 App. B; BKM §1.1 |
| 2-to-2 Games over F_2^ℓ, linear constraints | 1 − ε | ε | **NP-hard** (theorem) | KMS 2018 Thm 1.4 |
| 2-to-1 Games | 1 − ε | ε | NP-hard ("easily reinterpreted" from 2-to-2) | KMS 2018 footnote 2 |
| 2-to-2 / 2-to-1 with **perfect** completeness | 1 | ε | open (original Khot form); Lasserre gap with perfect completeness holds up to poly rounds (from 3Lin gaps) | KMS App. B |
| d-to-1, d ≥ 3, completeness 1−ε | 1−ε | ε | follows from 2-to-1 | [known] |
| Unique Games with completeness > 1/2 | 1/2+γ | ε | **open**; nothing between 1/2 and UGC | KMS, BKM |
| Max-Cut (1/2+Ω(ε), 1/2+ε/log(1/ε)) | | | NP-hard (from 2-to-2), previously UGC-only | KMS App. B |
| Vertex Cover √2 − ε | | | NP-hard (from 2-to-2) | KMS App. B |
| Independent set (1−1/√2−ε, ε); almost-4-colourable vs no ε-independent set | | | NP-hard | KMS App. B |
| Max-2Lin(2) NP-hardness curve | c ≥ 0.9232 → (1−s)/(1−c) > 1.48969 | | Martinsson 2024 (gadgets from Hadamard predicates) | arXiv:2408.04832 |

Under UGC the 2Lin(2) curve is the GW curve (1−ε vs 1−(2/π)√ε, KKMO); the unconditional curve is a
constant-factor deletion ratio 1.49, so the unconditional/UGC gap for the flagship consequence is
"√ε versus ε" — the same shape as the gap for UG itself. **[source: AKKT §1; Martinsson abstract]**

## 2. Anatomy of the 2-to-2 proof (why 1/2 and not 1−ε)

Chain **[source: KMS 2018 App. C]**:
Grassmann Expansion Hypothesis (KMS 2018, Thm 1.8) ⇒ [Barak–Kothari–Steurer 2018] Linearity-Testing
Hypothesis ⇒ [Khot–Minzer–Safra 2017; Dinur–Khot–Kindler–Minzer–Safra 2018] 2-to-2 Games Conjecture
(imperfect completeness). Reduction starts from Gap-3Lin.

The Grassmann graph Gr_{k,ℓ}: vertices = ℓ-dim subspaces of F_2^k, L ~ L' iff dim(L∩L') = ℓ−1.
Non-expanding sets: for subspaces A ⊆ B (dim A = a, codim B = b), the "zoom-in/zoom-out" subgraph
Gr[A,B] = {L : A ⊆ L ⊆ B} has expansion exactly 1 − 2^{−(a+b)} (+O(2^{−ℓ})).
**Theorem 1.8 (KMS).** ∀α<1 ∃ε>0, r: if Φ(S) ≤ α then S has density ≥ ε inside some Gr[A,B] with
a+b ≤ r; the dependence r = s for α < 1 − 2^{−(s+1)} is tight. Contrapositive: pseudorandom sets
(density o(1) in every constant-co-order copy) have expansion 1−o(1).

**The factor 1/2.** A 2-to-2 constraint T x_i ⊕ T' x_j ∈ {b, b'} is a union of two unique constraints;
splitting each into a random one of its two unique halves keeps every satisfied 2-to-2 constraint
satisfied with probability 1/2 and keeps soundness ≤ ε·(something small) — hence Gap-UG(1/2−ε, ε).
The Grassmann test is inherently a *2-query test of a code with a 2-to-1 (not unique) local test*: the
consistency between two subspaces L, L' of dimension ℓ meeting in ℓ−1 dimensions determines the
label of one from the other only up to a coset of size 2. This is not a loss in the analysis; it is
the structure of the encoding. **[reasoning, consistent with KMS/BKM statements]**

## 3. What the authors say about going further

* KMS 2018 App. B: "As far as the authors know (and we have consulted the algorithmic experts), the
  known algorithmic attacks on the Unique Game problem work equally well whether the completeness is
  ≈ 1 or ≈ 1/2. Thus … compelling evidence that the known algorithmic attacks are (far) short of
  disproving the UGC." Also: the reduction produces graphs that *always have small non-expanding
  sets*, so it gives no SSE hardness — supporting Khot's suspicion that **UGC may be true while SSEH
  is false**. **[source: KMS App. B, quoted]**
* BKM 2021: "One naturally asks whether the proof of the 2-to-1 Games Conjecture extends, without
  substantial effort, to that of the Unique Games Conjecture. We do not believe this to be the case."
  Their proposal: prove hardness of *rich* 2-to-1 games (uniform partition distribution at every left
  vertex); this is *equivalent* to UGC; and a "degree of richness" ladder — a sequence of reductions
  each achieving richer 2-to-1 games — is suggested as a research programme. Richness is what makes the
  *long code* satisfy a relaxed sub-code covering property (Lemma 10), which the Grassmann proof got
  from the Hadamard/Grassmann codes only for linear tests. **[source: BKM §1.2–1.3, quoted]**
* Khot–Moshkovitz 2014/2015: the obstacle to composing Projection Games with the long-code noise
  test is that the long code lacks *sub-code covering*; the Hadamard code has covering but no unique
  local test; the **real code** (periodised half-spaces over Gaussian space) has both. Their reduction
  is from random kCSP(PHLin) (Tulsiani's Lasserre-hard instances), with completeness, Lasserre
  completeness, and *restricted* soundness proven; full soundness hinges on **Robust Gaussian
  Isoperimetry**: "Which functions fail the noise test with probability only a constant times larger
  than a (periodized) half-space? Are functions influenced by a constant number of (periodized)
  half-spaces the only such functions?" Heilman 2017/2020 proved the endpoint case up to a 6·10^{-9}
  error. **[source: KM 2015 §1.3; Heilman arXiv:1708.00917]**
* Barak 2018: previous NP-hardness techniques using linear blow-up reductions are "inherently
  inapplicable"; a UGC proof needs a reduction with large polynomial blow-up (forced by ABS).
* Khot 2010 survey, arguments against UGC (still accurate today except item 2): (1) no natural hard
  distribution known; (2) no other natural problem provably equivalent — now partially addressed by
  Rich 2-to-1 and the SSE variant; (3) **no Lasserre gap instances known — "the current state of
  knowledge does not rule out the possibility that a Lasserre SDP, even with a small (say three)
  number of rounds, may disprove the UGC"**. Khot–Moshkovitz 2014 repeat: "At present we do not
  even know a (1−ε, 1−C·ε) gap with C→∞, even for general Unique Games, and even as Lasserre
  integrality gap." **[source: quoted]** I found no 2015–2026 paper that changes this.

## 4. Evidence ledger

For UGC: 2-to-2 theorem (half-way, NP-hardness in a regime with subexponential algorithms); every
UGC-predicted threshold that has been matched unconditionally so far was matched in the direction
UGC predicts (Max-Cut half-way, VC √2, Grothendieck, Max-3-Cut sharp hardness Heilman 2026 (2608.00333),
Max-k-CSP); SDP gap instances for basic SDP + SA rounds; Raghavendra's universality.
Against: BBHKSZ (all known gap instances are SoS-easy); ABS subexponential algorithm; no hard
distribution; Bafna–Minzer's remark that hard instances must avoid *every* known global-hypercontractivity
structure. Community sentiment shifted from skeptical (pre-2018) to "most likely true" (Barak 2018).

## Bibliography
1. KMS 2018, ECCC TR18-006 (`docs/papers/kms2018_pseudorandom_grassmann.pdf`).
2. Khot, Minzer, Safra. On independent sets, 2-to-2 games and Grassmann graphs. STOC 2017 / ECCC TR16-124.
3. Dinur, Khot, Kindler, Minzer, Safra. Towards a proof of the 2-to-1 games conjecture? STOC 2018; and
   On non-optimally expanding sets in Grassmann graphs, STOC 2018 / Israel J. Math 2021.
4. Barak, Kothari, Steurer. Small-set expansion in shortcode graph and the 2-to-2 conjecture. ITCS 2019, arXiv:1804.08662.
5. Braverman, Khot, Minzer. On Rich 2-to-1 Games. ITCS 2021; full version ECCC TR19-141.
6. Khot, Moshkovitz. Candidate Hard Unique Game (2015) and Candidate Lasserre Integrality Gap for Unique Games (ECCC TR14-142).
7. Heilman. A periodic isoperimetric problem related to the UGC. RSA 2020, arXiv:1708.00917.
8. Martinsson. On the NP-hardness approximation curve for Max-2Lin(2). arXiv:2408.04832.
9. Karthik C.S., Minzer. Improved multilayered PCPs and hypergraph vertex cover. arXiv:2609.06775 (Sept 2026).
10. Moshkovitz. Strong parallel repetition for unique games on small set expanders. arXiv:2103.08743 — **withdrawn 2022** (flaw: SSE conditioning not simulable).
