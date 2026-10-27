# Validation and reproducibility

Last reviewed: 2026-10-26

Tests and reports answer different questions. A test file shows intended checks;
a dated run supplies evidence for a particular revision and environment. Record
actual commands, results, scope, and limitations in the [log](log.md).

| Change or question | Relevant evidence |
| --- | --- |
| Grid construction and masks | [test_map_utils_torch.py](../tests/test_map_utils_torch.py), [test_map_utils.py](../tests/test_map_utils.py) |
| Arbitrary-q behavior | [test_arbitrary_q.py](../tests/test_arbitrary_q.py) |
| NumPy/PyTorch intensity agreement | [test_apply_disorder_equivalence.py](../tests/test_apply_disorder_equivalence.py) |
| PDOS loading and interpolation | [test_models_torch_pdos.py](../tests/test_models_torch_pdos.py) |
| Gradient behavior | [test_gradient_flow.py](../tests/test_gradient_flow.py) |
| Visualization data processing | [test_visualization_core.py](../tests/test_visualization_core.py) |
| Intermediate object states | [autotest implementation](../eryx/autotest/), [state-testing notes](../tests/state_based_testing.md) |

Run from the repository root in an environment containing
[requirements.txt](../requirements.txt) and pytest. Select the relevant test file
with `python -m pytest tests/<filename>.py`; the broad suite is
`python -m pytest tests/`. Some tests use fixtures or state logs; inspect the test's
setup before assuming it is self-contained. Historical passing-test statements
in state-testing notes do not establish a current green suite.

In particular, `tests/test_map_utils_torch.py` requires these pre-existing files:
`logs/eryx.map_utils.generate_grid.log`,
`logs/eryx.map_utils.compute_resolution.log`, and
`logs/eryx.map_utils.get_resolution_mask.log`. Its setup fails before numerical
checks when a file is missing. See the [log](log.md) for the observed run; recover
or deliberately regenerate suitable reference evidence before using these tests
to assess equivalence.

For a numerical comparison, hold the PDB, symmetry expansion, h/k/l or q-vector
set, model parameters, masking, ADP choice, dtype, and device fixed. The current
[NumPy](../run_numpy.py) and [PyTorch](../run_torch.py) grid drivers use matching
5zck PDB/grid inputs. Their output files alone cannot establish equivalence.
Drivers also write fixed filenames in the working directory; preserve experiment
outputs before rerunning them.

For gradients, test the intended parameter-to-loss path and compare numerical
derivatives where appropriate. Successful backpropagation alone does not show
that detached eigenvector contributions are included or that the physical model
is valid. See [phonon populations](phonon-populations.md) and
[open questions](open-questions.md).
