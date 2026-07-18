# space-robot-dq v0.3.0 — Change Summary

Apply these files on top of the current repository (v0.2.0, commit 6478e63).
Every file in this package either replaces an existing file at the same
path or is new. Nothing needs to be deleted.

## New files

| File | Purpose |
|---|---|
| `src/space_robot_dq/control.py` | **Paper novelty.** (1) Resolved-rate Cartesian tracking on the free-floating plant (fixed-base J_m vs generalized J_g controller) with DLS, null-space posture control, and momentum-conserving base pose propagation. (2) 6-DOF pose tracking of a tumbling target using the dual quaternion logarithm as pose error (capture-approach station-keeping). (3) Open-loop trajectory simulation with base attitude drift metrics; quintic line and tumbling-target generators. |
| `tests/test_control.py` | 23 new unit tests. Includes two independent physics checks on the moving-base simulation: momentum conserved < 1e-12 and system center of mass stationary (not enforced by construction). |
| `examples/generate_paper_figures.py` | Deterministically regenerates paper Figures 3, 4, and the new Figure 5, printing every number cited in the paper. Closes the reproducibility gap. |
| `CITATION.cff` | Citation metadata (GitHub renders a "Cite this repository" button). |

## Modified files

| File | Change |
|---|---|
| `src/space_robot_dq/kinematics.py` | PyTorch import made lazy: the library no longer requires torch (~2 GB) to be installed. Tensor inputs still work if torch is present. |
| `src/space_robot_dq/dynamics.py` | `compute_generalized_jacobian` gains optional `method="analytical"` (~10x faster, agrees with numerical to ~5e-9) and `return_inertia=True` (avoids recomputing H_b, H_bm in simulation loops). Fully backwards compatible. |
| `src/space_robot_dq/__init__.py` | Version 0.3.0; exports the new control functions. |
| `pyproject.toml` | Version 0.3.0; torch moved from required to optional (`pip install space-robot-dq[torch]`); matplotlib added to dev extras. |
| `benchmarks/run_benchmarks.py` | New Benchmark 7 (resolved-rate tracking: J_g controller RMS < 1 mm, >10x better than fixed-base, momentum conserved) bringing the suite to 16 checks; removed the stale "51 unit tests" text. |
| `README.md` | v0.3.0 features, optional-torch install, "Reproducing the Paper Figures" section. |

## Verified results (this environment: Python 3.12, NumPy 2.x, Ubuntu 24)

- Unit tests: **153 passed** (130 existing + 23 new)
- Benchmarks: **19/19 passed** (13 existing + 6 new)
- Library imports and runs without torch installed

Benchmark values to reconcile with the paper's Table 3 (rerun on your
machine for the authoritative numbers; seeded, so they should match):

| Check | Value here |
|---|---|
| FK vs matrix exponential / DH, max position error | 1.22e-12 m |
| FK orientation error | < 2e-6 deg |
| Numerical vs analytical Jacobian | 3.60e-9 |
| Jacobian vs independent reference | 4.58e-9 |
| Momentum over 2 s trajectory, max ‖h‖ | 5.44e-16 |
| IK success (< 50 mm, 100 targets) | 93%, median < 0.01 mm |
| Tracking RMS: fixed-base controller | 26.372 mm |
| Tracking RMS: generalized controller | 0.251 mm (105x better) |
| Base attitude drift over 0.27 m move | 6.88 deg |
| Pose tracking RMS vs 5 deg/s tumble: fixed | 4.267 mm / 0.220 deg |
| Pose tracking RMS vs 5 deg/s tumble: generalized | 0.0084 mm / 0.00058 deg (506x / 379x) |
| Base drift over 10 s tumble station-keeping | 2.44 deg |

Figure-generation output (deterministic; these become the paper's
Section 3 numbers):

- Figure 3 (sinusoidal trajectory, 100 kg base, 20.67 kg arm):
  fixed-base 0.843 m/s vs free-floating 0.643 m/s at t=0
  (24% overestimation), loss range 10–24%, max base reaction 0.030 m/s,
  momentum < 1.1e-15, base attitude drift 6.94 deg over 2 s.
- Figure 4 (mass-ratio sweep at the stated configuration):
  w_free/w_fixed = 0.036 at 0.5:1, 0.351 at 5:1, 0.539 at 10:1,
  0.928 at 100:1; curves cross near 7:1.
- Figure 5 (tracking): table above.
- Figure 6 (tumbling-target pose tracking, 5 deg/s, 10 s): table above.
  Reaction null-space was prototyped and excluded (only ~15% drift
  reduction for this configuration); noted as future work.

## Notes for the manuscript

1. The paper currently states "28 kg total arm mass" — the built-in SRS
   preset's auto-generated links sum to **20.67 kg**. The regenerated
   figures state the correct value.
2. The paper's abstract claim of ~3e-16 m FK accuracy is not what
   v0.2.0/v0.3.0 reproduces (1.22e-12 m here — still sub-picometer).
   Recommend claiming "< 1e-11 m" or the exact rerun value.
3. Table 3's corrupted heavy-base row: at 10,000:1,
   max|J_g − J_m| ≈ 3.1e-4 (from the unit test tolerance path).
