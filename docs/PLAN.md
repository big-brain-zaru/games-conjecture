# Exploratory plan: the Unique Games Conjecture

Principles (inherited from fast-matrix): verifier-first (no value is reported without a direct
evaluation of the constraints; every SDP number is a certified interval, primal ≤ value ≤ dual);
barrier-aware (do not re-run approaches with proven ceilings); evidence-gated milestones; no
overclaiming — a found gap instance is a theorem about that instance, an exhausted search is data.

## Where NOT to spend effort, and why (evidence in docs/literature/)

- **Beating CMM in polynomial time on general instances.** CMM's 1−O(√(ε log k)) matches the basic-SDP
  gap (Khot–Vishnoi) and, under UGC, nothing polynomial does better (KKMO, Raghavendra). Any improvement
  must use a strictly stronger relaxation — that is Track A, not a rounding tweak.
- **Re-deriving that KV / short-code / Johnson / Grassmann / HDX instances are hard.** They are all
  solved by constant-degree SoS or certified-SSE rounding (BBHKSZ 2012, BBKSS 2021, Bafna–Minzer 2024).
  Running SDP/SoS on them at small size measures nothing about UGC. We use KV only as a fixture and as
  the object of Track C (its exact value, which is a separate open problem).
- **Extending the Grassmann proof "a bit".** The authors (KMS, BKM) state the 2-to-2 machinery does
  not extend; the 1/2 is the two-halves split, structural to a 2-to-1 code. Pen-and-paper program.
- **Strong parallel repetition.** False (Raz's odd-cycle counterexample); Moshkovitz's 2021 SSE
  variant was withdrawn for a fatal flaw.
- **Random instances as hard distributions.** Uniform-random UG has value ≈ 1/k; planted/semi-random
  are solved by SDP (KMM 2011). No natural hard distribution is known — finding one is itself Track A's
  goal, not an input.
- **Real quantum hardware / entangled provers.** Entangled UG is in P (Kempe–Regev–Toner); says
  nothing about the classical conjecture.

## Where the box has an opening

The whole field's uncertainty collapses onto one fact: **no integrality gap for degree-4 sum-of-squares
on a (1−ε)-satisfiable unique game is known, and no proof that degree-O(1) SoS succeeds exists either.**
Khot (2010): "a Lasserre SDP, even with a small (say three) number of rounds, may disprove the UGC";
Khot–Moshkovitz (2014): not even a (1−ε, 1−Cε), C→∞ Lasserre gap is known. Every explicit hard-looking
family is SoS-easy for a structural reason (certifiable hypercontractivity). Nobody has run a *computer
search* for instances that defeat degree-4 SoS, because the objects were always designed by hand from
Fourier analysis. That is the stone.

## Track A — Adversarial search for degree-4 SoS gaps (core, weeks 1–4)

Bilevel search: over instances I on a fixed small constraint structure (graph + alphabet, weights and
permutations free), maximise gap(I) = SoS_4(I) − opt(I) subject to SoS_4(I) ≥ 1−ε, with opt(I) exact
(MaxSAT) and SoS_4(I) a certified SDP value. Sizes: n = 12–60 vertices, k = 2–4, where degree-4 SoS
is computable (PSD blocks (nk)²) and exact optima are seconds.

**Novel twists:**
1. **Differentiable instance design.** By Danskin, ∂SoS_4/∂w_e is the pseudo-expectation of the
   edge-e constraint; ∂opt/∂w_e is its indicator under an optimal assignment. Gradient ascent on the
   weight simplex plus discrete moves on permutations. Hypothesis A1: gaps found this way at degree 2
   reproduce the KKMO/GW shape (1−ε vs 1−Θ(√ε)); at degree 4 they either collapse to 1−Θ(ε) (evidence
   that SoS-4 beats the SDP curve on small instances — new algorithmic evidence) or persist (the first
   explicit constant-degree SoS gaps for UG, however small).
2. **Gap-shape metric, not gap size.** Report C(I) = (1−opt)/(1−SoS_4): the (1−ε, 1−Cε) constant
   Khot–Moshkovitz ask for. Track the best C as a function of n; extrapolate; any C growing with n
   at fixed degree is the signal.
3. **Symmetry-stratified search** (fast-matrix method): impose a symmetry group on the instance so the
   SoS SDP shrinks and larger n becomes exact; a gap found under symmetry is a gap.

## Track B — The hypercube conjecture (AKKT 2014), weeks 2–3

Max-2Lin(Z2) on Q_d: exact opt (MaxSAT), GW-SDP, GW-SDP + triangle inequalities (GPU ADMM with lazy
triangle generation), degree-4 SoS where feasible. Hypothesis B1: on AKKT's gap family and on
adversarially designed Q_d instances (Track A machinery restricted to Q_d), the triangle SDP is within
O(ε) of opt — supporting their conjecture; a counterexample refutes it. Either is publishable and new.

## Track C — The value of the Khot–Vishnoi game (weeks 1–3, cheap)

opt(U_{k,η}) is only known within [(1−η)^k-ish, N^{−η}]. Compute exactly for k=3 (MaxSAT running),
k=4 by symmetry + branch-and-bound or MaxSAT, and the exact basic-SDP value for k ≤ 6 via
symmetry-reduced SDP. Hypothesis C1: the subcube labeling is optimal for all η below a threshold
(a noise-stability/transversal isoperimetry statement); C2: the exact basic-SDP value of KV is
1 − Θ(η) with an explicit constant far below 9. Output: the first exact table of KV values.

## Track D — Map the certified-SSE boundary (weeks 2–4)

Implement the degree-4 SoS certificate of hypercontractivity (2→4 norm bound on the top eigenspace)
and run it across graph families: abelian vs non-abelian Cayley graphs (S_3, Q_8, A_4, D_n,
SL(2,p)), random lifts, hypercube, Johnson, Grassmann, Kneser, "foam"/cubical complexes. Hypothesis
D1: non-abelian Cayley graphs with many large eigenvalues fail certification at degree 4 far more
often than abelian ones — and UG instances built on them (Track A restricted) show larger degree-4 gaps.
This tells the proof side where hard instances can live (Bafna–Minzer's own criterion).

## Track E — Khot–Moshkovitz real code, numerically (weeks 3–5)

(i) Robust Gaussian Isoperimetry probe: GPU variational search over odd, coordinate-periodic
functions f : R^n → {±1} (n ≤ 8, discretised) minimising noise-test failure; test whether anything
beats "influenced by O(1) periodised half-spaces" by more than a constant. (ii) A finite truncation
of Khot's TR14-142 candidate Lasserre gap: compute its Lasserre-2 value and its true optimum at small
parameters. Hypothesis E1: at the sizes reachable, the candidate's integral value already shows
C = (1−opt)/(1−Lasserre) > the GW constant, or it does not (data either way for Khot's open question).

## Track F — Richness ladder (background, pen-and-paper + small computations)

Quantify richness (distance of each left vertex's pairing distribution from uniform) for explicit
small 2-to-1 instances; search for gadgets that raise richness while preserving the gap (SAT on small
alphabets). Reporting only if a quantitative handle emerges.

## What would constitute failure, stated upfront

> **Outcome, recorded 17 September 2026.** The first criterion below was met: the full search found no
> degree-4 gap shape above 1.093586, nowhere near C > 2. The pre-committed response was to publish the
> negative map and the tool, and that is what the repository is. The second criterion was also met at
> the sizes reached (dimension 5). Track C reached the k = 3 exact basic-SDP value; the k = 3 *optimum*
> is still only bracketed. This section is left exactly as written before the search, so that the
> criteria can be read against the outcome rather than after it.

- Track A finds no degree-4 gap with C > 2 at n ≤ 60 after the full search budget → publish the
  negative map (best C per (n, k, structure), with certificates) and the tool.
- Track B: triangle SDP within O(ε) on everything tried → evidence for AKKT's conjecture, published as
  such, with the instance archive.
- Track C: k=4 exact value out of reach → report k=3 exact + k=4 bounds.
- Any claim leaves this folder only with a machine-checkable certificate (MaxSAT model / dual SDP
  point / DRAT where applicable) attached.

## Immediate next actions

1. Degree-4 SoS module (`ug_sos.py`) with exact small-instance tests against brute force.
2. Track C: finish KV k=3 exact; k=4 subcube value on the truncated instance; symmetry-reduced SDP.
3. Track A pilot: differentiable gap search at degree 2 on n=12, k=2 (must rediscover odd-cycle /
   GW-type gaps) before degree 4.
