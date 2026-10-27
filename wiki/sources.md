# Source catalog

Last reviewed: 2026-10-26

Initial ingestion date: 2026-10-26. These are repository sources reviewed for
specific claims, not a complete ingestion of the repository. Code references
describe the working tree at review time; use Git history to recover prior
versions. No external articles or papers have been ingested into [raw](../raw/README.md).

| Reviewed source and scope | Extracted knowledge | Synthesis |
| --- | --- | --- |
| [Package exports](../eryx/__init__.py) | Unsuffixed OnePhonon is NumPy; PyTorch has a separate export | [Architecture](architecture.md) |
| [PyTorch model](../eryx/models_torch.py): constructor, grid setup, phonon population path, PDOS loader/interpolator/generator | GNM sampling requirements, explicit temperature, loader checks, thermal transforms, detached eigenvectors | [Sampling](sampling.md), [populations](phonon-populations.md) |
| [NumPy](../eryx/map_utils.py) and [PyTorch grid utilities](../eryx/map_utils_torch.py): generate_grid | Oversampling semantics and coordinate conversion | [Sampling](sampling.md) |
| [NumPy driver](../run_numpy.py), [PyTorch driver](../run_torch.py): inputs and exports | Matching grid inputs after concurrent updates; NPZ coordinate/intensity/shape fields | [Sampling](sampling.md), [validation](validation.md) |
| [Pump-probe driver](../pump_probe_validation.py): argument parser | Pump-profile options exist; no `--pdos-file` argument | [PDOS reference](../docs/PDOS.md) |
| [PDOS tests](../tests/test_models_torch_pdos.py): interpolation tests and fixtures | Intended interpolation, gradient, and input checks | [Validation](validation.md) |
| [Visualization exports](../eryx/visualization/__init__.py), [data handler](../eryx/visualization/core/data_handler.py), [slicing example](../examples/create_slicing_animation.py) | Optional backends, result metadata, and GIF example | [Visualization](visualization.md) |
| [Requirements](../requirements.txt) | Core dependencies listed; pytest and K3D are not listed | [Validation](validation.md), [visualization](visualization.md) |
| [PyTorch quickstart](../docs/PYTORCH_QUICKSTART.md): newly added workflow and interpretation notes | Bundled CPU example and explicit output contract; execution claims were not independently checked in this wiki pass | [Overview](overview.md), [documentation index](../docs/README.md) |
| [Quickstart notebook](../examples/pytorch_diffuse.ipynb) and [script](../examples/pytorch_diffuse.py): complete default CPU workflow | Notebook executes from root/examples with inline figures; saved numerical results match the script, as recorded in the log | [Visualization](visualization.md), [verification log](log.md) |
| [PDOS guide](../docs/PDOS.md), before this correction | Wrong public import/defaults, unsupported CLI flag, formula/code discrepancy | [Populations](phonon-populations.md), [open questions](open-questions.md) |
| [Status snapshot](../PROJECT_STATUS.md), [port architecture](../docs/architecture.md), [q-vector notes](../docs/qvec_conventions.md): status/design sections | Historical plans and examples need a boundary from current behavior | [Overview](overview.md), [architecture](architecture.md) |
| [Visualization hub](../docs/visualization/README.md), [K3D hub](../docs/visualization/k3d/README.md), [state-testing notes](../tests/state_based_testing.md): navigation/status sections | Old readiness labels and broken routes cannot serve as current verification | [Visualization](visualization.md), [validation](validation.md) |

Unreviewed scope includes full scientific derivations, all archived plans and
notebooks, experiment arrays/plots, exhaustive code paths, and browser execution.
Links to deeper references are reading routes, not claims of full ingestion.
Add external-source provenance and focused source summaries as described in
[AGENTS.md](../AGENTS.md). Related: [index](index.md), [log](log.md).
