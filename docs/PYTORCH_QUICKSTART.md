# Diffuse scattering with PyTorch

Run a small crystal simulation and inspect diffuse intensity, displacement
correlations, and phonon diagnostics. All inputs for the first run are bundled.
For larger maps and PDOS comparisons, use the [notebook](../examples/pytorch_diffuse.ipynb).

## What the calculation produces

The model reads a crystallographic PDB, constructs neighboring unit cells and an
elastic network, computes displacement modes, and evaluates one-phonon diffuse
scattering. With the default asymmetric-unit grouping, each asymmetric unit has
six rigid-body degrees of freedom. The constructor computes modes and covariance;
the disorder calculation produces a flat intensity tensor.

The example saves an NPZ with `intensity`, `q_vectors`, `hkl_grid`, and `map_shape`.
The first three arrays share point order. Cartesian q includes the 2π factor and
has units Å⁻¹. Reshape intensity using the stored shape, never a guessed cube.

## Run the small command-line example

1. Use the bundled `tests/pdbs/histidine.pdb` initially. Protein runs can use
   `tests/pdbs/6o2h_clean.pdb` or a custom PDB with cell and symmetry information.
2. Activate your Python environment and install the requirements below. Run from
   the repository root; the package is imported directly from the checkout.
3. Run this command sequence:

   ```bash
   python -m pip install -r requirements.txt
   python -m examples.pytorch_diffuse
   ```

   The complete script is [examples/pytorch_diffuse.py](../examples/pytorch_diffuse.py).
   Edit the constants at its top. It runs on CPU without a display, writing to
   `tmp/pytorch-quickstart/` and overwriting previous outputs.

## What happens

- Construct the explicit PyTorch model using the bundled small crystal.
- Calculate intensity using PDB B-factors for displacement damping; preserve NaNs.
- Save `results.npz`, including model-derived `adp` and `covariance` diagnostics.
  Those saved ADPs are not the PDB ADPs used for this example's damping.
- Write `slices.png` for the central h, k, and l planes. The helper hides ticks;
  these are array views, not calibrated Cartesian q plots.
- Write `radial.dat` and `diagnostics.png`: mean finite intensity versus |q|.
  The average covers the sampled box, which may only partially cover outer shells.
- Write `pdos.dat` and a mode-density histogram, using the code's empirical
  frequency scale. This is a diagnostic, not a calibrated experimental spectrum.

## Change the inputs

| Setting | Meaning |
|---|---|
| `PDB` | Start with histidine; change to a protein once the small run works |
| `SAMPLING = (-2, 2, 3)` | Minimum, maximum, oversampling: spacing 1/3 and 13 points per axis |
| `DEVICE` | CPU initially; use a CUDA device with a compatible PyTorch installation |
| `OUT` | Directory for example outputs |
| `expand_p1=True` | Expand the asymmetric unit using crystal symmetry |
| `gnm_cutoff=4.0` | Contact cutoff in Å |
| `gamma_intra`, `gamma_inter` | Spring strengths within/between asymmetric units |
| `res_limit=0.0` | No positive high-resolution cutoff; a positive limit is in Å |
| `use_data_adp=True` | PDB B-factor damping; False selects computed ADPs |

Grid length along h is `int(oversampling * (maximum - minimum) + 1)`, similarly
for k and l. Increasing all three oversampling factors grows cost rapidly.
Computed ADPs/covariance are rescaled against the PDB displacement values.

## Interpret and compare

Preserve missing intensities and use a common finite mask when comparing maps.
Singular modes and resolution filtering can create NaNs; filling them with zero
changes quantitative statistics. Align coordinates before comparing values.

The [statistics module](../eryx/stats.py) provides Pearson correlation overall,
by resolution shell, and by distance from Bragg peaks. The disorder method's
`rank` argument selects one mode; its default −1 sums modes. Saved covariance
describes the selected displacement coordinates: rigid-body coordinates in this
example, not an atom-by-atom covariance matrix.

## Optional: notebook

For interactive exploration, install Jupyter in the same environment and launch:

```bash
python -m pip install jupyterlab ipykernel
python -m jupyterlab examples/pytorch_diffuse.ipynb
```

Select that environment's Python kernel and **Restart Kernel and Run All Cells**.
The notebook uses the bundled **5zck peptide crystal on a 65³ grid**, with saved
preview plots, labeled log-scale slices, a PDOS histogram, enhanced mode weights,
and baseline/enhanced/difference maps. It runs without gradients to limit memory.
Change its sampling to `(-4, 4, 16)` for 129³. Frequency scales remain empirical.

It locates the checkout from the root or `examples/`. Outputs go to
`tmp/pytorch-notebook/`; reruns replace them. `pdos.dat` contains unweighted mode
density; `example_population.dat` is a two-column direct-population input example.
The comparison applies the analytic weights directly; loading the sampled file
instead adds interpolation error. These outputs differ from the small script's.

## Optional: other calculation and visualization routes

- **Selected q points:** supply an N×3 PyTorch tensor through `q_vectors`. GNM
  still requires all three sampling tuples for covariance calculation. Arbitrary
  points cannot be reshaped into a physical volume without known grid ordering.
- **Interactive views:** follow the [K3D example](visualization/k3d/getting-started/QUICK_START.md).
- **Population changes:** see [PDOS](PDOS.md) and the pump-probe demonstration.
- **Legacy comparisons:** `run_torch.py` defaults to grid mode; `--arb-q` selects
  its grid-derived arbitrary-q comparison. These larger runs write fixed filenames.

## Optimization

First-order intensity derivatives support spring constants (`gamma_intra` and
`gamma_inter`), arbitrary `q_vectors`, PDOS density samples, and thermal
`temperature_k`. Pass leaf tensors with `requires_grad=True` for the parameters
you intend to fit. The model retains supplied tensors, including when converting
their dtype or device for a calculation. Loaded `model.pdos_density` is also a
leaf and can be passed to an optimizer.

Rebuild phonons and covariance after parameter updates. `apply_disorder()` uses
the stored populations so deliberate pump-probe edits to `Winv` remain effective.
For example, run from the repository root:

```python
import torch
from eryx.models_torch import OnePhonon

gamma = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
q = torch.tensor([[0.11, 0.07, -0.09]], dtype=torch.float64, requires_grad=True)
model = OnePhonon(
    "tests/pdbs/5zck_p1.pdb", [-1, 1, 2], [-1, 1, 2], [-1, 1, 2],
    gamma_intra=gamma, gamma_inter=0.7, q_vectors=q, device=torch.device("cpu"),
)
optimizer = torch.optim.SGD([gamma, q], lr=1e-8)
for _ in range(2):
    optimizer.zero_grad()
    model.compute_gnm_phonons()
    model.compute_covariance_matrix()
    intensity = model.apply_disorder()
    loss = intensity[torch.isfinite(intensity)].sum()  # Replace with your fit loss.
    loss.backward()
    optimizer.step()
```

[Numerical derivative tests](../tests/test_differentiability.py) compare autograd
with finite differences, including complex mode rotation, repeated eigenvalues,
independent duplicate q points, and successive optimizer steps. Standalone
`scatter_torch.structure_factors` also supports derivatives of coordinates,
form factors, ADPs, and projections.

Limits:

- The combined mode covariance supports first-order backward only. Higher-order
  requests (`create_graph=True`) raise explicitly; `torch.func` transforms are
  not supported by its custom operation.
- PDB parsing, contact topology, grid dimensions, and masks are fixed/discrete.
  BZ boundaries, mode cutoffs, and interpolation knots can be nonsmooth. End-to-end
  structure or unit-cell refinement is outside this tested path.
- Summed intensities handle repeated modes for smooth, symmetric populations.
  Selecting one rank or assigning unequal weights inside a degenerate eigenspace
  has no unique derivative and raises during backward. Manual population edits
  must preserve the symmetry of degenerate modes; uniform scaling and symmetric
  normalization are supported. Diagnostic `model.V` has native eigenvector
  gradient restrictions at degeneracy.
- Recompute any parameter-derived tensors yourself between iterations if you
  supply non-leaf inputs. Histogram/frequency exports are detached diagnostics.
  Mathematical gradients do not validate the physical thermal model or its
  empirical frequency calibration; see [PDOS interpretation](PDOS.md).
