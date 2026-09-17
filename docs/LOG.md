# Running log

## Day 1 — 13 September 2026

**Brief.** Use p_vs_np and fast-matrix as structural references; deep-dive the UGC literature; build the
tools; aim for breakthroughs with no shortcuts, no assumptions.

**Environment [computed].** Python 3.13 (Store), RTX 5070 8 GB, 24 cores, CUDA 13.1, CuPy 13.6,
PyTorch cu128, PySAT/CaDiCaL 1.9.5, Z3, numba; installed cvxpy 1.9.2 + Clarabel + SCS + galois
(numpy moved to 2.2.6; GPU stack re-verified). No C compiler on PATH (not needed so far).

**Literature.** Eight parallel survey agents were killed by an API session limit; the survey was done
directly from primary sources (PDFs in `docs/papers/`, text extracted with `tools/pdf2txt.py`):
Khot–Vishnoi JACM, Khot 2010 survey, CMM 2006, KMS 2018 (2-to-2 final piece), Braverman–Khot–Minzer
2021 (Rich 2-to-1), Khot–Moshkovitz 2014/2015 (candidate Lasserre gap / candidate hard UG),
AKKT 2014 (UG on the hypercube), plus abstracts of BBHKSZ 2012, BBKSS 2021, Bafna–Minzer 2023, Heilman
2017, d'Orsi et al. 2024, Sahai–Gnanasekaran 2024, Yoshida 2026, Levene–Paulsen 2025, Martinsson 2024,
Karthik–Minzer 2026, and an arXiv sweep of every 2024–2026 paper mentioning unique games. Written up in
`docs/literature/01–04`. Headline: **no degree-4 SoS gap for near-satisfiable UG is known, and no
proof it cannot exist**; that is where the tools point.

**Tools built and self-tested [computed].**
- `experiments/ug_core.py`: instance class, label-extended graph, perfect-completeness propagation,
  generators (random, planted, Max-2Lin), brute force, MaxSAT (RC2/CaDiCaL), MILP (HiGHS), vectorised
  local search. Self-test passes (brute force == MaxSAT on random instances).
- `experiments/kv_instance.py`: exact Khot–Vishnoi construction; total weight verified = 1−(1−η)^N on
  the untruncated instance; KV vector solution reproduced.
- `experiments/ug_sdp.py`: basic SDP via cvxpy (Clarabel ≤ ~150 rows — its dense PSD Hessian needed 8 GB
  at 300 rows; SCS 288 s at 300 rows); **GPU block-coordinate ascent** (exact Procrustes block updates,
  Gauss–Seidel by graph colouring) reaching the SCS/Clarabel value to 5 decimals in seconds; **dual
  certificate** (feasible dual point from the primal iterate + eigenvalue shift) giving primal = dual
  to 1e-4. Adam/QR Burer–Monteiro was tried first and abandoned (0.83 vs exact 0.872 after 5000 its).
- `experiments/ug_sos.py`: level-2 Lasserre for any alphabet and degree-2/4 SoS for k=2; exact on
  n=4, tight on C5 (GW 0.90451 vs opt 0.8, SoS4 = 0.8), sandwiched opt ≤ L2 ≤ SDP on n=7.
- `experiments/gap_search.py`: Track A adversarial gap-ratio search (exact gradient from pseudo-moments).

**Numbers [computed].**
- KV k=4, η=0.1 (4,096 vertices, 16 labels, 1,425,408 weighted constraints): basic SDP ∈
  [0.795002, 0.795010] in 6 s + 6 s; propagation rounding 0.492; subcube labeling ≈ 0.656 untruncated;
  Bonami bound 0.758. `results/kv4_eta0.1_sdp_first_run.json`, vectors in `results/*.npy`.
- KV k=3, η=0.2: subcube labeling value 0.43919 on the truncated instance (local search cannot improve).
  Exact optimum: MaxSAT stalled (>15 min, many distinct weights); MILP (HiGHS) running in background.
- Track A pilot (K8/K9, k=2): annealing-only search reached C=1.8 at degree 2 (odd cycles give 3.66 for
  C9) — optimiser too weak; replaced by exact-gradient weight ascent + seeded sign patterns (running).
  At degree 4 every instance found so far has C = 1.000 (SoS4 exact) — consistent with Laurent-type
  gaps having C ≈ 1 + 1/(n(n−2)); not yet evidence of anything at these sizes.

**Decisions.** Track A needs a GPU degree-4 solver (D ≈ n²/2) with warm starts to reach n ≈ 30–60 where
degree 4 is far from exact; cvxpy/SCS is for calibration only. Track C: exact KV values via MILP +
symmetry. Track B (hypercube) and D (certified-SSE map) queued.

**Evening, day 1.** GPU degree-4 SoS solver (`sos_gpu.py`, ADMM + over-relaxation + residual
balancing) matches SCS to 1e-6 on K9 (3.8 s) and certifies two-sided bounds; K30 (D=466) in 7 s.
Track A search moved to it (`gap_search_gpu.py`; CPU for D ≤ 250). Track B (`hypercube_track.py`,
AKKT Δ[k,d]) and the Feige–Schechtman seed track (`fs_track.py`) launched. First numbers in the
HYPOTHESES ledger. Note for tomorrow: the KV instance is invariant under F_2^N (translations of the
noisy cube) and AGL(k,2); the invariant basic SDP is a small LP over AGL-orbits of Boolean functions
(K̂ ≥ 0 by Fourier over F_2^N; K vanishes on nonconstant affine functions; K(1) = 1/N) — exact
values for k ≤ 4 immediately, k = 5 if the 5-variable affine classification is enumerated. Same idea
block-diagonalises degree-4 SoS on KV/Cayley instances (the fast-matrix symmetry lever).

## Day 2 — 14 September 2026

**The pivot.** A sanity check of the degree-4 solver on K₅ (SoS₄ = 0.625 = the analytic 5/8 while
opt = 0.6) showed it was working and *not* trivially exact — which exposed that day 1's Track A search
had been driving instances into the treewidth-2 region where level-2 Lasserre is exact. That led to
the reformulation in `docs/FINDINGS.md` §1: because the gap shape C_d = (1−opt)/(1−R_d) is invariant
under dilution, Khot–Moshkovitz's open question ("no (1−ε, 1−Cε) Lasserre gap with C → ∞ is known")
is exactly "is sup_I C_d(I) infinite?", which is a scalar optimisation over instances.

**Built.** `kv_transversal.py` (KV as a maximum-weight transversal of the simplex code, verified
against the generic evaluator); `group_ug.py` + `group_census.py` (the group-quotient generalisation,
reproducing KV exactly and extending to non-abelian G); `c4_max.py` (exhaustive search over instance
shapes via switching + the graph atlas); `sos_gpu.py` upgraded with over-relaxation and residual
balancing (K₉ now matches SCS to 1e−6 in 3.8 s instead of 16 s and not converging);
`circulant_sos.py` (symmetry-reduced degree-4 SoS, validated against the dense spectrum and the dense
solver); `circulant_scan.py`, `circulant_opt.py`, `circulant_full.py`; `reproduce.py` (45 checks).

**Measured.** max C₄ = 1.074139 at Cay(Z₉,{±1,±2}); decaying with L; C₄ = 1 on every classical family.
Exact KV basic-SDP values at k = 3, 4 by GL(k,2)-invariant LP, each cross-checked numerically. The
affine group is *not* a symmetry of KV — the AGL version was refuted by the certificate, a mistake
worth keeping in the record.

**Open at end of day 2.** KV k=3 optimum not closed (CP-SAT XOR model: incumbent 0.439189, bound
0.445524 after 2400 s; transversal weighted-CSP model still running). Degree-4 ADMM certificates go
loose past D ≈ 500, so the AKKT hypercube track is conclusive only for d ≤ 5. The group census needs
the same symmetry reduction to reach |Q| ≥ 12, where degree 4 stops being exact.

## Day 2, late — consolidation

The complete sweep closed: 374 circulants, 16 ruled out by the C_4 <= C_2 filter, 358 certified, and
the 9 remaining (one graph up to the multiplier action, Cay(Z_19,{1,2})) settled at 250,000 iterations
with SoS_4 = opt = 0.736842 exactly. Maximum over the family 1.081179; project maximum 1.093586 at
Cay(Z_41,H_4).

Documents consolidated for release: FINDINGS.md rewritten as one coherent record in eleven sections
(it had grown into twelve layered addenda), both retracted claims kept in place and marked, README
rewritten in the style of the sibling repositories with an explicit scope section. Third-party PDFs
excluded from the repository as copyrighted, with a note explaining how to refetch them; the 13 MB
solver artefact excluded as regenerable. reproduce.py: 50 checks, 0 mismatches.

## Day 3 — the algorithmic turn

The user's verdict on day 2 was blunt: too much computation for too little.  Day 3 replaces enumeration
with optimisation and mechanism.

**Built.** `group_sweep.py` stage 2 batched (`GroupSoSBatch`) and routed by carrier dimension;
`weight_ascent.py` and `class_ascent.py` (certified alternating LP/SoS ascent over weights, with kicks);
`kn_gap.py`; `hypercontract.py` (2→4 norm of the carrier eigenspace as a predictor of retention);
`expander_gap.py` (random regular graphs); `paley_lift.py`, `paley_moments.py`, `wick_lift.py`.

**Found.** Paley graphs p ≥ 29: SoS₄ = SoS₂ exactly (stored data; tight certification running).
K_n: degree 4 adds nothing to degree 2 for every n.  Random cubic graphs n ≤ 20 and random dense
instances n ≤ 7: degree 4 is exact.  Z₄₁ record: stationary under all class-weight perturbations.
Non-abelian carriers of dimension 4 (F₂₀) retain 0.2 of the degree-2 gap; dimension 3 mostly 0.

## Day 3, close

Paley p = 53, 61 certified (lower bounds exact to 1e-9). Degree sweep at n = 24: d = 4: C₂ 1.202, C₄ ∈ [1.0000, 1.0000], d = 6: C₂ 1.162, C₄ ∈ [1.0000, 1.0000], d = 8: C₂ 1.101, C₄ ∈ [1.0000, 1.0000], d = 10: C₂ 1.071, C₄ ∈ [1.0000, 1.0001], d = 12: C₂ 1.054, C₄ ∈ [1.0000, 1.0000], d = 14: C₂ 1.055, C₄ ∈ [1.0000, 1.0000], d = 16: C₂ 1.036, C₄ ∈ [0.9933, 1.0093], d = 18: C₂ 1.009, C₄ ∈ [0.9911, 1.0039]; max
certified C₄ 1.0000 — degree 4 stays exact or within 1e-4 of exact for every d. Circulant
hypercontractivity digest on 321 instances (weak support for H14; the outlier is disjoint K₅'s).
The non-abelian sweep past order 21 and the circulant run past L = 19 were stopped as not finishable
in useful time on a shared machine; both are recorded as partial. FINDINGS section 11 is the day-3
record; reproduce.py: 58 checks, 0 mismatches. Pushed to the private repository.

## 17 September 2026 — prior-art pass and framing

Checked the two most interesting day-3 findings against the literature before considering publication.
Both are correct and neither is new.

The complete-graph identity is a corollary of Laurent, Math. Oper. Res. 28(4) 2003, Theorem 6, via
Grigoriev's parity refutation: Max-Cut on K_n with unit weights is minimising (Σᵢxᵢ)², and her explicit
pseudo-moment sequence with a₂ = −1/(n−1) is PSD at Lasserre order (n−1)/2. So the identity holds at
every degree up to n−1, not just 4. `laurent_kn.py` rebuilds her certificate; it reproduces our
numbers to 2e-16, which doubles as an independent check of the ADMM solver.

The Paley result is implied asymptotically by Mohanty–Raghavendra–Xu, whose Theorems 1.2/1.3 are
general rather than specific to random d-regular graphs and SK, as this record had described them.
`mrx_check.py` measures their loss parameters on the true Paley degree-2 optimum: α = Θ(C/√p) against
a loss-to-advantage ratio tending to 8/√2. What survives is the exact finite-p equality, which their
map structurally cannot give, and the threshold at p = 29.

No published Max-Cut-on-Paley statement above degree 2 was found. Nearest neighbours added to the
survey: Kunisky–Yu CCC 2023 (Paley, clique number), de Boor CMU-CS-19-118 (degree-4 MaxCut, random
regular only), Kunisky–Bandeira (the SK analogue), Laurent 2003 and Grigoriev 2001.

Then a framing pass across README, FINDINGS section 10, HYPOTHESES and this log, to state the work at
its actual size: a null result that the plan predicted, a set of reusable certified solvers, and a
record with three corrections in it. The group-theoretic framework remains unchecked against the
literature and is now labelled as such.
