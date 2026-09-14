# Measuring the Gap Shape of Sum-of-Squares Relaxations of Unique Games

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A verifier-first computational study of Khot's Unique Games Conjecture, run on one consumer laptop.
Every value reported here comes from a direct evaluation, a machine-checkable certificate, or an
exhaustive enumeration. Negative results are reported as results, and two claims that turned out to be
wrong during the work are kept in the record, marked, alongside their corrections.

## Scope

**No progress is made on the conjecture itself.** There is no new hardness result and no new algorithm.
What this repository contains is a reformulation of one of the field's stated open questions into a
quantity a computer can search, a certified measurement of that quantity over several families, an
elementary inequality that makes such searches *complete* rather than merely extensive, and a
construction framework for a class of instances that had none. Section 10 of
[docs/FINDINGS.md](docs/FINDINGS.md) states exactly what is and is not established.

## The question, made computational

Khot and Moshkovitz (ECCC TR14-142) write that no (1−ε, 1−C·ε) Lasserre integrality gap for Boolean
2Lin with C → ∞ is known, at any constant degree. Define the **gap shape** of an instance,

  C_d(I) = (1 − opt(I)) / (1 − R_d(I)),

for a degree-d relaxation R_d. Diluting an instance with a perfectly satisfiable part leaves C_d
unchanged, because both the optimum and the relaxation are affine across a disjoint union. So a gap
ratio found at *any* completeness transports to completeness 1−ε for *every* ε, and their question is
exactly

  **a (1−ε, 1−C·ε) degree-d gap with C → ∞ exists  ⟺  sup_I C_d(I) = ∞.**

At degree 2 the supremum is infinite: odd cycles give C_2 = 4L/π². At degree 4 it is open.

## Key results

| Result | Evidence | Status |
|---|---|---|
| **The dilution identity**, turning the Khot–Moshkovitz question into sup_I C_d(I) = ∞ | one line, checked numerically | elementary |
| **C₄ ≤ C₂**, so C₄ ≤ (1 − any exhibited cut)/(1 − SoS₂) | SoS₄ ≤ SoS₂; SoS₂ is closed-form for a circulant | elementary, and the reason searches here are complete |
| **Largest certified degree-4 gap shape: 1.093586**, at the Cayley graph of the quartic residues mod 41 | tight certificate (lower = upper = 0.727904) and a proved optimum (0.702439) | verified |
| The family peaks there: p = 73, 89, 97 are **ruled out** | upper bounds 1.043453, 1.074587, 1.056552 from the filter alone | exact |
| It is the subgroup **index**, not the graph degree | index 6 at p = 37 gives C₄ = 1.000000 with a closed certificate, at a degree between the two index-4 winners | verified |
| Classical families give nothing | cycles, Petersen, Paley, sparse random regular, hypercube all give C₄ = 1; complete graphs give 1 + 1/(n(n−2)) | verified |
| Exhaustive over all instance shapes on ≤ 6 vertices | 11 and 34 switching classes, every one weight-optimised | exact |
| **Complete sweep of the small circulants** | all 374 instances with L odd ≤ 19 and ≤ 3 connection classes: 16 ruled out by the filter, 358 certified, 9 stragglers settled at 250,000 iterations; maximum 1.081179 | complete, no gaps |
| **A group-theoretic generalisation of Khot–Vishnoi** | for any finite group, normal subgroup and symmetric measure, the maximum-weight transversal problem is a unique game whose label-extended graph is a Cayley graph; Khot–Vishnoi is the abelian case, reproduced vertex for vertex | verified |
| **Exact basic-SDP values of the Khot–Vishnoi game** | a GL(k,2)-invariant linear program: 0.6157095, 0.3948238 (k=3), 0.7950017 (k=4), each confirmed by an independent certified numerical solve | exact |
| Evidence for the Agarwal–Kindler–Kolla–Trevisan hypercube conjecture | at dimensions 4 and 5 the degree-4 value equals the optimum, sandwiching the triangle-inequality SDP to exactness | verified at d ≤ 5 |
| A claimed growth rate, **refuted by its own follow-up** | C₄ ≈ 1.046 + 0.0127·ln n at R² 0.998 on four points, killed by p = 73 and p = 89 | retracted, kept in the record |

## Figure

![gap shape](figures/gap_shape_degree2_vs_degree4.png)

Degree 2 grows linearly in the number of vertices. Degree 4, over the same range, moves by 2.5 per cent
and then falls back. Hollow markers are rigorous upper bounds. Whether the red curve is bounded is the
open question: if it were, degree-4 sum-of-squares would beat Goemans–Williamson and the Unique Games
Conjecture would be false.

## Installation

Python 3.9 or later. An NVIDIA GPU is optional and used only by the scalable solvers.

```bash
python -m pip install -r requirements.txt
```

## Usage

All commands run from `experiments/`.

```bash
python reproduce.py                    # recompute every published number: 50 checks, 0 mismatches (~47 s)
```

Self-tests of the machinery:

```bash
python ug_core.py                      # instances, exact solvers, label-extended graph
python ug_sdp.py                       # basic SDP: cvxpy vs GPU block ascent vs dual certificate
python ug_sos.py                       # level-2 Lasserre and degree-2/4 sum-of-squares
python sos_gpu.py                      # ADMM degree-4 with certified bounds
python circulant_sos.py --selftest     # symmetry-reduced degree-4 vs the dense spectrum
python kv_transversal.py --selftest     # Khot–Vishnoi as a maximum-weight transversal
python group_ug.py                     # the group framework, reproducing Khot–Vishnoi
```

The searches:

```bash
python sweep.py --Lmin 5 --Lmax 25             # stage 1: the cheap complete filter
python sweep2.py --Lmin 5 --Lmax 19            # stage 2: certified degree 4 on the shortlist
python c4_max.py --n 6 --degree 4              # exhaustive over instance shapes
python paley_index.py --jobs 17:4,41:4         # the generalised Paley family
python kv_invariant_sdp.py                     # exact Khot–Vishnoi SDP values
python hypercube_track.py --dims 4,5           # the hypercube conjecture
```

## Repository layout

```
.
├── README.md
├── LICENSE                         # MIT
├── requirements.txt
├── docs/
│   ├── FINDINGS.md                 # the scientific record, including the corrections
│   ├── PLAN.md                     # tracks, and the barriers not worth re-running
│   ├── HYPOTHESES.md               # numbered hypotheses, each with a decisive test and its verdict
│   ├── LOG.md                      # chronological log
│   └── literature/                 # the survey, written from primary sources
├── experiments/                    # all code; every module has a self-test
├── results/                        # every instance record, certificate and solver log
├── figures/
└── tools/
```

## Reproducibility notes

- `experiments/reproduce.py` reports **50 checks, 0 mismatches**. It recomputes each quantity rather
  than reading it from a result file.
- Every gap shape reported as a value comes from a *tight* certificate, meaning the certified lower and
  upper bounds on the relaxation agree to the printed precision, together with a proved optimum. Where
  either is missing the number is reported as an interval and labelled as such.
- The source PDFs behind `docs/literature/` are copyrighted and are deliberately not redistributed. Each
  is cited by title, venue and a stable link; `tools/pdf2txt.py` is the extractor used.
- Two solver behaviours worth recording: the dense degree-4 certificates go loose past about 500 rows,
  and Clarabel allocates a dense Hessian for the positive-semidefinite cone that needs 8 GB at 300 rows,
  so it is used only for small calibration instances.

## Citation

```bibtex
@misc{zaru2026uniquegames,
  title={Measuring the Gap Shape of Sum-of-Squares Relaxations of Unique Games:
         A Certified Search, a Completeness Filter, and a Group-Theoretic
         Generalisation of the Khot-Vishnoi Instance},
  author={Zaru, Nadim F.},
  year={2026}
}
```

## Prior work used

- S. Khot and N. Vishnoi, *The Unique Games Conjecture, integrality gap for cut problems and
  embeddability of negative type metrics into ℓ₁*, JACM 62(1), 2015 — the instance, reproduced and
  generalised here.
- S. Khot and D. Moshkovitz, *Candidate Lasserre Integrality Gap for Unique Games*, ECCC TR14-142 — the
  open question this repository measures.
- B. Barak, F. Brandão, A. Harrow, J. Kelner, D. Steurer and Y. Zhou, *Hypercontractivity, sum-of-squares
  proofs, and their applications*, STOC 2012 — the levels at which the known gap instances are certified.
- M. Bafna and D. Minzer, *Solving unique games over globally hypercontractive graphs*, CCC 2024 — the
  statement that graphs without certified hypercontractivity are where hardness must live.
- N. Agarwal, G. Kindler, A. Kolla and L. Trevisan, *Unique Games on the Hypercube*, CJTCS 2015 — the
  conjecture tested in section 7.

Full bibliographies are in `docs/literature/`.

## Author

**Nadim F. Zaru**
Independent Researcher

## License

MIT License, see [LICENSE](LICENSE).
