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

PyTorch support does not imply complete derivatives: the current model detaches
phonon eigenvectors. Check gradients for the parameters you intend to fit.
