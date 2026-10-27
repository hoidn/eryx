# eryx

Diffuse scattering simulations for protein crystals, with NumPy and PyTorch
implementations of a one-phonon model. The PyTorch path supports CPU/CUDA,
reciprocal-space grids, and evaluation at supplied q vectors.

## Getting started

From the repository root, create an environment and run the small bundled example:

```bash
conda create --name eryx python=3.10
conda activate eryx
python -m pip install -r requirements.txt
python -m examples.pytorch_diffuse
```

This CPU example uses the bundled histidine crystal on a 13×13×13 grid. It writes
intensity and coordinates, central slice images, a radial intensity profile, and
a mode-density histogram under `tmp/pytorch-quickstart/`. Re-running replaces
those example outputs.

Follow the [PyTorch workflow](docs/PYTORCH_QUICKSTART.md) to use a protein PDB,
adjust sampling, interpret outputs, and compare with experiment.
For larger 65³ peptide-crystal maps, labeled log-scale plots, example PDOS, and
population comparisons, open the [notebook](examples/pytorch_diffuse.ipynb).

**Import selection matters:** `from eryx.models_torch import OnePhonon` selects
PyTorch; `from eryx import OnePhonon` selects NumPy. The package also exports
`OnePhonon_torch` as an explicit PyTorch alias.

## Further reading

- [Project wiki](wiki/index.md) — architecture, concepts, evidence, and open questions
- [Documentation index](docs/README.md) — detailed guides and historical records
- [PDOS inputs and interpretation limits](docs/PDOS.md)
- [Visualization](docs/visualization/README.md), including optional K3D HTML export
- [Sampling and reciprocal-space coordinates](wiki/sampling.md)

The default model uses an elastic network projected onto rigid-body motion of
each asymmetric unit. This is not a molecular-dynamics simulation. Frequency
helpers use an empirical conversion; eigenvectors are detached from autograd.
Validate physical units and parameter gradients before quantitative fitting.
