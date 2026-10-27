# PyTorch audit implementation plan

**Goal:** Audit the PyTorch simulation paths and fix reproducible, substantial performance and correctness defects on `feat/pytorch-audit`.

**Approach:** Keep existing APIs and float64/complex128 precision. Trace callers and compare independent mathematical expectations, then add failing regressions before changing shared implementations. Preserve the existing detached-eigenvector convention and explicitly document its gradient limits.

- [x] Map model setup, scattering, matrix utilities, adapters, entry points, and existing tests; run baseline tests.
- [x] Reproduce skew-cell coordinate errors, zero phonon population gradients, repeated eigensolves, rectangular pseudoinverse failures, fabricated solver results, and incorrect complex decompositions in `tests/test_pytorch_audit.py`.
- [x] Fix shared linear algebra and coordinate conversions in `eryx/pdb_torch.py`, `eryx/torch_utils.py`, and `eryx/models_torch.py`.
- [x] Remove duplicated phonon calculations; restore eigenvalue sensitivities with detached-basis Rayleigh quotients and safe handling of masked modes.
- [x] Verify arbitrary-q intensity vectorization against the existing loop for values and gradients; benchmark representative CPU/CUDA workloads.
- [x] Run focused regressions, relevant existing tests, and small model integration checks. Compare failures to baseline; obtain independent code review.
- [ ] Record audit findings, measured results, and verification limits; commit reviewable changes.

Regression command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_pytorch_audit.py -q`.

Relevant existing tests: `tests/test_torch_utils.py`, `tests/test_map_utils_torch.py`, `tests/test_models_torch_pdos.py`, and model/scattering equivalence tests. Baseline output is captured separately from post-change output in `/tmp/eryx-audit-*.log`.

## Findings and fixes

The audit assumed individual CPU/CUDA scientific simulations, including dense reciprocal-space sampling and gradient-based fitting. It traced the PyTorch model setup, Hessian/phonon/covariance/intensity paths, scattering functions, mapping and matrix utilities, and their connected adapters, tests, and entry points. NumPy code supplied coordinate conventions and forward-value comparisons. Visualization and archived experiments were outside this numerical audit.

1. **Phonon populations had zero parameter derivatives.** Eigenvalues from a `no_grad` eigensolve were connected with `0 * D.sum()`. Detached-basis Rayleigh quotients now produce real eigenvalue sensitivities, verified against finite differences for default and custom PDOS populations. Discrete k-vector grouping stays detached to avoid unsupported `torch.unique` backward.
2. **Spring updates reused old values and graphs.** `compute_hessian` now rebuilds the spring tensor from current parameters. A regression changes a spring constant between two forward/backward calls.
3. **Excluded modes poisoned backward.** NaN mode weights were multiplied into structure factors before masking. Finite placeholder weights are now used during arithmetic, then the original output NaN policy is restored. Both sampling modes and selected/all ranks are covered; real-protein gradients are finite with either ADP source on CPU and CUDA.
4. **Arbitrary-q and grid coordinates disagreed.** Two erroneous reciprocal-cell transposes were fixed. Near-half-integer roundoff is canonicalized to the same Brillouin-zone representative as grid sampling. Tests cover skewed cells, repeated reciprocal translations, and full intensity agreement.
5. **The custom pseudoinverse failed rectangular matrices and repeated-value backward.** Native `torch.linalg.pinv` replaces manual diagonal allocation and SVD reconstruction. Identity-matrix derivatives and rectangular complex matrices now pass. Failed solvers propagate their error instead of returning fabricated identity inverses or modes.
6. **Complex decompositions discarded conjugation.** SVD reconstruction and Hermitian symmetrization now use conjugate transposes; reconstruction tests run on CPU and CUDA.
7. **The phonon calculation ran twice.** Removed the discarded duplicate calculation and batched the eigensolve. The arbitrary-q intensity path also replaces one Python iteration per q-vector with tensor contractions. Forward values and gradients are checked against scalar formulas.
8. **PDOS lookup and export used different units.** The established empirical conversion `f_THz = sqrt(lambda) * 100 / (2π)` is retained. Lookup now uses its equivalent SI angular frequency, `sqrt(lambda) * 1e14`. Direct and thermal import tests recover nonconstant weights at exported frequency knots. Two old tests claiming uncalibrated SI eigenvalues were reconciled with the explicitly established conversion in commits `7d78fc5` and `ec7b9b5`.
9. **PDOS frequency diagnostics inverted population weights.** Saved eigenvalues now supply frequencies after custom populations replace `Winv`; histogram generation shares the same getter. Older default-population states retain the `Winv` fallback.
10. **Interpolation extrapolated despite its documented endpoint policy.** Queries outside the PDOS table are now clamped, preventing negative extrapolated weights from otherwise positive endpoint data.

## Validation

Environment: Python 3.13, PyTorch 2.7.1+cu126; CPU and available CUDA hardware.

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_pytorch_audit.py tests/test_equivalence.py \
  tests/test_torch_utils.py tests/test_models_torch_pdos.py -q --disable-warnings
```

Result: **112 passed**, including 70 new regression cases. Before edits, the initial new numerical regressions had 30 failures and 2 passes; later boundary, PDOS, unique-backward, and excluded-mode reproductions also failed before their fixes. Existing mode-equivalence failures were reproduced on the baseline and now pass. Syntax compilation and `git diff --check` pass. An independent read-only numerical review found no blocking issue after corrections.

The original full-suite attempt stopped at 10 failures (25 passed, 1 skipped). Those included obsolete gradcheck arguments, missing state attributes, and tests calling NumPy on tensors requiring gradients. The focused original utilities/PDOS baseline had 5 failures: three missing saved-log fixtures and the two stale frequency-conversion expectations described above. Broad-suite comparison results are recorded below.

Full `tests/` runs, with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`:

| Checkout | Passed | Failed | Skipped | Time |
| --- | ---: | ---: | ---: | ---: |
| Original `e3333dd`, isolated temporary worktree | 156 | 43 | 16 | 148.68 s |
| Updated shared checkout | 238 | 36 | 16 | 159.47 s |

No new failed test identifiers appeared. The updated checkout includes five concurrently added workflow tests, so the passing-count increase is not exclusively attributable to this audit. The remaining failures concern unavailable state logs, outdated gradcheck/state-testing interfaces and initialization assumptions, grid/BZ shape assumptions, and direct eigenvector comparisons across decompositions. Both complete runs exited normally with pytest's test-failure status, not a timeout. The temporary baseline worktree was removed after verification.

## Measured performance

Reproduce from the repository root with:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. python benchmarks/benchmark_pytorch_audit.py
```

The benchmark loads the original model methods from `e3333dd`, checks forward values against the updated methods, warms up twice, and reports the median of five runs. CUDA timings synchronize before and after each run. These are synthetic hot-path timings, not whole-protein or backward benchmarks.

| Case | Device | Original | Updated | Speedup |
| --- | --- | ---: | ---: | ---: |
| 128 k-vectors, 6 modes | CPU | 18.55 ms | 1.74 ms | 10.7× |
| 4,096 q-vectors, 3 modes | CPU | 245.72 ms | 2.32 ms | 105.8× |
| 128 k-vectors, 6 modes | CUDA | 2,952.17 ms | 27.73 ms | 106.5× |
| 4,096 q-vectors, 3 modes | CUDA | 484.33 ms | 19.06 ms | 25.4× |

## Remaining limits

- Eigenvectors and phonon dependence on q through BZ grouping remain detached by design. Tests establish the specified population sensitivities and finite intensity backward, not complete structural or q derivatives. The method docstring states that limit.
- The empirical frequency calibration and existing thermal weighting law were preserved, not physically revalidated. Very low-temperature thermal preprocessing can still overflow; changing that model is outside this audit.
- Historical state-log fixtures and unrelated legacy tests were not regenerated. Gemmi emits nanobind reference-leak diagnostics on interpreter shutdown in the existing environment.
- Concurrent documentation, workflow, and visualization edits in the shared checkout are excluded from the audit commit.
