# Simulation architecture

Last reviewed: 2026-10-26

Choose the implementation explicitly. [Package exports](../eryx/__init__.py)
bind `eryx.OnePhonon` to the NumPy class and `eryx.OnePhonon_torch` to the PyTorch
class when its imports succeed. PDOS examples should import `OnePhonon` directly
from `eryx.models_torch` to avoid ambiguity.

| Responsibility | NumPy | PyTorch |
| --- | --- | --- |
| One-phonon model | [models.py](../eryx/models.py) | [models_torch.py](../eryx/models_torch.py) |
| Structure/crystal handling | [pdb.py](../eryx/pdb.py) | [pdb_torch.py](../eryx/pdb_torch.py) |
| Reciprocal-space utilities | [map_utils.py](../eryx/map_utils.py) | [map_utils_torch.py](../eryx/map_utils_torch.py) |
| Scattering calculations | [scatter.py](../eryx/scatter.py) | [scatter_torch.py](../eryx/scatter_torch.py) |

The PyTorch model also uses NumPy crystallographic objects and
[adapters](../eryx/adapters.py); the columns are not isolated stacks.
Its constructor sets up crystal/sampling data and phonon machinery. Model
methods compute phonons and covariance, and `apply_disorder` computes intensity.
Read the relevant method for actual execution order and cached state before
changing an optimization loop.

The model sets real precision to `torch.float64` and complex precision to
`torch.complex128`. Some eigenvector operations are explicitly detached in
`compute_gnm_phonons`; describe supported derivatives precisely instead of
promising full differentiability of every quantity.

The [older port architecture](../docs/architecture.md) explains design history,
but its phase tables and illustrative code are not a current API specification.
Related: [sampling](sampling.md), [populations](phonon-populations.md),
[validation](validation.md), and [source coverage](sources.md).
