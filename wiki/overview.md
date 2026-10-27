# Project overview

Last reviewed: 2026-10-26

Eryx simulates diffuse scattering from protein crystals using one-phonon models.
It contains parallel NumPy and PyTorch implementations; the latter includes
custom phonon populations and gradient-oriented calculations. Public exports
are defined in [eryx/__init__.py](../eryx/__init__.py). The availability of an
autograd path does not establish the physical correctness of its derivative.

## Where things belong

| Location | Role |
| --- | --- |
| [eryx](../eryx/) | Simulation, adapters, state capture, and visualization code |
| [tests](../tests/) and [tests/pdbs](../tests/pdbs/) | Tests and crystallographic fixtures |
| [examples](../examples/) and root Python scripts | Executable examples and experiment drivers |
| [docs](../docs/README.md) | Detailed references, guides, and labeled historical designs |
| [wiki](index.md) | Maintained synthesis and unresolved questions |
| [raw](../raw/README.md) | Future immutable external sources |
| [plans](../plans/), [records](../records/), [archive](../archive/) | Planning and historical evidence in their original locations |

## Reading paths

- To run the bundled example: [PyTorch quickstart](../docs/PYTORCH_QUICKSTART.md).
- To understand a calculation: [architecture](architecture.md) →
  [sampling](sampling.md) → [phonon populations](phonon-populations.md).
- To inspect a result: [sampling](sampling.md) → [visualization](visualization.md).
- To change or assess the implementation: [validation](validation.md) →
  [open questions](open-questions.md) → relevant code and tests.

[PROJECT_STATUS.md](../PROJECT_STATUS.md) records a 2025 initiative snapshot.
Its completion labels and `plans/active/` directory names do not establish today's
priorities or verification status. See the [source catalog](sources.md) for the
scope of this wiki's initial review.
