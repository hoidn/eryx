# Wiki log

Append-only knowledge history. Conventions: [AGENTS.md](../AGENTS.md).
Navigation: [index](index.md), [sources](sources.md), [open questions](open-questions.md).

## [2026-10-26] ingest | Repository knowledge baseline

- Reviewed repository sources within the scope listed in the [source catalog](sources.md).
  Added [overview](overview.md), [architecture](architecture.md),
  [sampling](sampling.md), [phonon populations](phonon-populations.md),
  [visualization](visualization.md), and [validation](validation.md).
- Recorded thermal PDOS, gradient-support, and readiness questions in
  [open questions](open-questions.md), with evidence and resolution criteria.
- Established the [maintenance schema](../AGENTS.md), [raw-source boundary](../raw/README.md),
  [wiki index](index.md), and [documentation index](../docs/README.md).
  Replaced duplicated agent guidance with shared-schema routing and repaired
  visualization entry points. Marked old designs/status records as historical.
- Concurrent work updated code, README/PDOS guidance, and added a
  [PyTorch quickstart](../docs/PYTORCH_QUICKSTART.md). Preserved those changes,
  linked the new guide, and rechecked affected sampling/import/gradient claims
  against the working tree. Those software edits are not products of this ingest.
- Coverage: source inspection and documentation reconciliation. No external
  source ingestion, experiment regeneration, full-suite verification, or browser
  validation is implied. Lint/test results are recorded in a separate entry below.

## [2026-10-26] lint | Navigation, evidence, and sampling checks

- A one-off Python check passed for 24 documentation pages and 259 local links,
  including local anchor resolution, all 10 wiki pages in the index, inbound
  links, and topic/source review dates. Scope: wiki, agent instructions, raw README,
  project/docs/visualization entry points, PDOS/quickstart, and historical design
  entry pages. Deep archival documents were not exhaustively linted.
- `git diff --check` passed. Reviewed entry-point routing and searched maintained
  navigation for removed filenames and obsolete sampling claims.
- Ran `python -m pytest tests/test_models_torch_pdos.py::TestPDOSInterpolation tests/test_map_utils_torch.py -q`:
  **4 passed, 3 failed**. All three failures occurred in map-utility setup at the
  missing `logs/eryx.map_utils.generate_grid.log` prerequisite, before numerical
  assertions. Confirmed that the resolution and resolution-mask reference logs
  are also absent; documented prerequisites in `wiki/validation.md` and the gap
  in `wiki/open-questions.md`. The test process also emitted nanobind shutdown
  warnings; this was not a warning-free suite run.
- A direct CPU assertion check of `generate_grid` passed with identity `A_inv`
  and `(-2, 2, 8)` sampling on each axis: shape `(33, 33, 33)`, 35,937 coordinate
  rows, endpoints ±2, and Cartesian q equal to `2π * hkl`.
  This checks the documented example; it does not replace reference-log tests.
- Reconciled concurrent changes to the driver inputs, PDOS frequency handling,
  volume quickstart, and slicing example. These checks apply to the working tree
  as observed during this pass, not an isolated committed revision. No full test
  suite, large simulation, new environment installation, or browser run was performed.

## [2026-10-26] lint | Runnable PyTorch and visualization workflows

- Updated the [README](../README.md), [PyTorch guide](../docs/PYTORCH_QUICKSTART.md),
  [PDOS reference](../docs/PDOS.md), and [visualization routes](../docs/visualization/README.md)
  against the current API. Added the bundled [CPU example](../examples/pytorch_diffuse.py)
  and refreshed the [GIF example](../examples/create_slicing_animation.py).
- Fixed NPZ intensity/coordinate unpacking, shape handling, runner failure
  propagation and CPU fallback; aligned root comparison inputs. Numerical audit
  changes from concurrent work were preserved, not implemented by this update.
- Executed `python -m examples.pytorch_diffuse` (CPU, histidine, 13³ grid),
  `python -m examples.create_slicing_animation` (12 GIF frames), and the Python
  block in the [K3D guide](../docs/visualization/k3d/getting-started/QUICK_START.md)
  (HTML snapshot). Outputs are under `tmp/pytorch-quickstart/`.
- Ran `python -m pytest tests/test_workflow_entrypoints.py tests/test_visualization_core.py -q`:
  29 passed, one third-party deprecation warning. The five new regression checks
  failed before their fixes. Checked 68 local links in the updated entry points
  and visualization topic; none missing. Python compilation and scoped diff
  whitespace checks passed.
- No fresh environment installation, CUDA execution, full protein simulation,
  browser interaction, or physical calibration was verified. Snapshot generation
  is not a browser compatibility test; existing modeling questions remain open.

## [2026-10-26] lint | Executable notebook quickstart

- Reviewed the existing pump-probe and K3D notebook entry cells; none covers the
  small simulation workflow. Added [examples/pytorch_diffuse.ipynb](../examples/pytorch_diffuse.ipynb)
  with editable inputs, repository discovery, exposed model/arrays, inline plots,
  and saved-output assertions. Linked it from the [README](../README.md),
  [guide](../docs/PYTORCH_QUICKSTART.md), and [visualization topic](visualization.md).
- Ran both documented pip install commands in the current environment; all
  requirements were already installed. Ran `python -m examples.pytorch_diffuse`
  using bundled histidine, CPU, 13×13×13 sampling: 1,872 finite points of 2,197.
- Executed all eight notebook code cells with nbclient in separate fresh Python
  kernels, once from the repository root and once from `examples/`. Both passed
  the notebook assertions and produced two inline PNG plots. The notebook's NPZ
  arrays, radial profile, and mode histogram matched the command-line outputs
  at rtol/atol 1e-10, including missing-value masks. An executed copy is saved in
  `tmp/pytorch-notebook/executed-quickstart.ipynb`; the source notebook is clean.
- The documented JupyterLab launch reached server startup with browser opening
  disabled for the check, then the server was terminated. No browser UI behavior,
  fresh dependency installation, CUDA run, full protein run, or physical
  calibration was tested. No numerical implementation changes were made.

## [2026-10-26] lint | Larger scattering and PDOS notebook views

- Replaced the notebook's 13³ histidine smoke-test defaults with a 65³ grid of
  the bundled 5zck peptide crystal, following the structure/population examples
  in [pump_probe_validation.ipynb](../pump_probe_validation.ipynb) and
  [pump_probe_validation.py](../pump_probe_validation.py). The separate
  [command-line smoke test](../examples/pytorch_diffuse.py) remains unchanged.
- The [notebook](../examples/pytorch_diffuse.ipynb) now includes shared log-scale
  slices with Miller-index axes/colorbars, unweighted PDOS and frequency-binned
  scattering weights, a Gaussian population enhancement, matched baseline and
  enhanced maps, a signed difference map, and radial comparison. It saves an
  illustrative two-column direct-population file separately from the DOS histogram.
  PDB displacement damping and mode shapes stay fixed; baseline weights are
  restored with try/finally. No numerical library changes were made.
- Executed all nonempty code cells with nbclient on CPU at default 65³ and finer
  129³ sampling. Both generated four inline figures and passed saved-array,
  finite-mask, nonnegative-response, file-format, and baseline-restoration checks.
  Default 65³ had 273,896 finite points of 274,625; 129³ had 2,145,960 of 2,146,689.
  Inspected the PNGs and regenerated the embedded previews after the last plot edit.
- Loaded the generated `example_population.dat` into a separate model with
  `pdos_mode="direct"` and recomputed the 65³ map. Finite masks matched; the
  relative L2 difference from analytic in-memory weighting was 0.0034693.
  The guide distinguishes sampled-file interpolation from analytic weighting.
- Initial default execution took about 13 seconds; the later repeated run amid
  concurrent verification took about 164 seconds, and the finer run about 47
  seconds. These are observations on this machine, not performance guarantees.
  One concurrent verification process was stopped and rerun sequentially.
- Kept executed previews in the source notebook; backed up the user's prior
  executed copy at `tmp/notebook-visual-check/before-plot-update.ipynb` and retained
  their added empty cell. Updated the [guide](../docs/PYTORCH_QUICKSTART.md),
  [README](../README.md), [visualization topic](visualization.md), and source catalog.
  Notebook schema, local links, and scoped whitespace checks passed. CUDA,
  large-protein runs, and physical frequency calibration were not tested.
