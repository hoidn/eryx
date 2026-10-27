# Differentiability audit

Baseline: `fbbb21e`; working branch: `feat/differentiability-audit`.

The target is correct first-order derivatives of intensities with respect to the
continuous tensor inputs exposed by the model: spring strengths, arbitrary q,
PDOS samples, and temperature. Standalone scattering also accepts differentiable
coordinates, form factors, ADPs, and projections. PDB parsing, contact topology,
grid sizes, and hard masking/selection boundaries are discrete setup, not smooth
optimization variables.

- [x] Trace parameter-to-intensity paths and reproduce incorrect derivatives,
  stale converted inputs, and failures on repeated backward.
- [x] Add numerical gradient regressions, including noncommuting complex matrix
  perturbations, repeated eigenvalues, independent equal q points, and multiple
  optimizer steps on an actual protein.
- [x] Differentiate the combined spectral matrix instead of discarding mode
  rotation. Preserve diagnostic eigenvectors, population edits, and forward NaN
  masks; report undefined derivatives at degenerate rank selections.
- [x] Preserve q derivatives and original parameter tensors through conversion;
  refresh q-derived state when recomputing phonons.
- [x] Keep thermal PDOS samples as optimizable leaves and rebuild normalization
  per evaluation; remove artificially trainable fixed structural matrices.
- [x] Compare relevant and full suites to baseline; review changes independently;
  document supported optimization and the genuinely nonsmooth boundaries.

Numerical design: for a Hermitian matrix `D = U diag(lambda) U.H`, the derivative
of `U diag(p(lambda)) U.H` uses divided differences
`(p_i - p_j) / (lambda_i - lambda_j)`, with `p'(lambda)` at repeated values.
The native population graph carries diagonal eigenvalue sensitivities; a custom
first-order backward supplies off-diagonal mode-rotation terms. This avoids the
singular eigenvector derivatives described in the
[PyTorch eigh documentation](https://docs.pytorch.org/docs/2.7/generated/torch.linalg.eigh.html).
Selecting a single member of a degenerate eigenspace is not a differentiable
observable, so that backward must raise rather than silently invent a gradient.

Keep explicit `compute_gnm_phonons()` / `compute_covariance_matrix()` refreshes
between optimizer steps: existing pump-probe callers deliberately overwrite
`Winv` before `apply_disorder()` and must keep that behavior.

## Reproduced defects and corrections

1. The old phonon path detached the mode basis. A complex, noncommuting matrix
   perturbation gave an intensity derivative of about -0.297383, versus a central
   finite difference of -0.464440. The combined spectral derivative includes
   that missing mode rotation and handles smooth populations at repeated modes.
2. Arbitrary q lost its phonon derivative through detach, rounding, and unique
   grouping. Trainable q now bypasses discrete deduplication; two identical q
   inputs retain independent derivatives. Fixed q still uses deduplication.
3. Cached dtype/device conversions and artificially trainable structural
   matrices retained consumed graphs across optimizer steps. The model keeps
   original spring/q parameters, converts them afresh, and refreshes q-derived
   state in `compute_gnm_phonons()`. PDB-derived geometry and mass matrices remain
   constants. No input q tensor is implicitly made trainable.
4. Thermal normalization replaced the stored PDOS density with a non-leaf.
   Density is now an optimizable leaf in both modes; normalization is rebuilt
   on every evaluation so temperature updates also take effect.
5. The PDB adapter copied existing tensors through `torch.tensor`, detaching
   their graph. Tensor inputs now use `.to`; NumPy inputs retain copy semantics.
   Standalone scattering no longer fabricates trainable outputs under `no_grad`.

The regressions also cover inference mode, masked inactive degeneracies,
per-q/global population normalization, asymmetric cross-q population edits,
zero-population selected modes, and explicit rejection of higher-order requests.
The 24 initial tests had 21 failures before these fixes. Additional edge-case
tests reproduced their respective failures before the corresponding corrections.

## Validation

- Environment: PyTorch 2.7.1+cu126, CPU and NVIDIA GeForce RTX 3090.
- Focused compatibility suite: **174 passed**, one warning, before the final
  zero-population selected-mode regression. That regression and the related
  mode-rotation/rank cases then passed (**15 passed**).
- Executed the two-step CPU optimizer example in the
  [quickstart](../docs/PYTORCH_QUICKSTART.md#optimization), including finite
  parameter-gradient assertions. Tests additionally cover float32/float64 source
  leaves with both CPU and CUDA models, both ADP sources, and direct/thermal PDOS.
- Full baseline at `fbbb21e`: **238 passed, 36 failed, 16 skipped**. The first
  post-change comparison exposed four obsolete tests requiring fixed tensors
  to be trainable. Their expectations were corrected while retaining all
  numerical comparisons. Final full suite: **278 passed, 34 failed, 16 skipped**,
  517 warnings, 238.25 seconds. All **38** new differentiability cases pass.
  Comparing failing test IDs shows no added failures; basic arbitrary-q
  initialization and the batched phonon gradient test now pass. Remaining
  failures include missing reference logs, stale sampling/shape/eigenvector
  expectations, and pre-existing state-serialization failures.
- Independent review checked the spectral derivative and found the inference,
  masked-degeneracy, cross-q, higher-order, and zero-population edge cases fixed
  above. Local-link checking (34 links), compilation, and `git diff --check` pass.

Full-suite command:
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/ -q --disable-warnings`.
The baseline and final logs are `/tmp/eryx-diff-baseline.log` and
`/tmp/eryx-diff-verified.log`. To rerun the new derivative cases, use
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_differentiability.py -q`.

## Supported scope and limits

This establishes first-order intensity derivatives for the continuous parameters
listed at the top, within a fixed topology and away from hard boundaries. The
custom backward rejects `create_graph=True` to avoid silently incomplete second
derivatives. It does not support `torch.func` transforms. Diagnostic eigenvectors
have native `eigh` restrictions, and arbitrary user population edits must
themselves be smooth and preserve degenerate-mode symmetry.

Consequently the main intensity-fitting workflow does **not** have established
second-order differentiation. Exact Hessians, Hessian-vector products, gradient
penalties, and differentiation through optimizer updates require additional
implementation and numerical second-derivative checks. Native PyTorch operations
elsewhere do not make the complete model twice differentiable.

PDB parsing/contact construction, full structure or cell refinement, interpolation
knots, mode/mask/BZ boundary crossings, and physical thermal-law calibration are
not established by these tests. Optimizing q intentionally costs more than a
fixed-q evaluation because independently trainable points cannot share a rounded
eigensolve. No new large-simulation performance benchmark was run.
