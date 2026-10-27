# Open questions and contradictions

Last reviewed: 2026-10-26

## Thermal PDOS interpretation

The former [PDOS guide](../docs/PDOS.md) described
`rho(omega) / (exp(hbar * omega / (kB * T)) - 1)`.
[OnePhonon._load_and_prepare_pdos](../eryx/models_torch.py) instead divides the
input density by `exp(-hbar * omega / (kB * T))`, and `compute_gnm_phonons`
multiplies the interpolated value by that factor at each evaluated frequency.
The guide now describes those operations without asserting their physical validity.

**Needed:** establish whether the input represents a DOS, a population, or a
weighted amplitude; specify the intended thermal law and frequency calibration;
then compare a controlled analytic example with the implementation. This is a
scientific/modeling decision, not something to settle by rewriting a formula in
documentation. See [phonon populations](phonon-populations.md).

## Scope of gradient support

Earlier documentation promised full gradient preservation.
[compute_gnm_phonons](../eryx/models_torch.py) explicitly detaches eigenvectors,
and thermal preprocessing replaces `pdos_density` with a non-leaf tensor.
These observations limit blanket optimization claims.

**Needed:** identify required optimizable parameters and verify each relevant
loss derivative, including graph reuse across iterations. The existing
[gradient tests](../tests/test_gradient_flow.py) are a starting point, not proof
of every optimization use case. See [validation](validation.md).

## Missing grid reference logs

The 2026-10-26 check of [test_map_utils_torch.py](../tests/test_map_utils_torch.py)
failed all three tests in setup at the missing
`logs/eryx.map_utils.generate_grid.log` prerequisite. No numerical comparison
was reached. [Validation](validation.md) lists all required log files.

**Needed:** recover reference logs with provenance, or regenerate them for a
deliberately chosen baseline. This documentation pass does not manufacture new
ground truth to make the tests pass. Results are recorded in the [log](log.md).

## Current project and visualization status

[PROJECT_STATUS.md](../PROJECT_STATUS.md) is a 2025 snapshot, while visualization
documents contain readiness labels and phase reports from earlier work. These
do not establish current priorities or compatibility with installed dependencies.

**Needed:** a new scoped test/browser run for readiness claims, and a user-directed
priority update before treating old plans as the current roadmap. Historical
bodies are retained; [current navigation](index.md) labels their role.

Resolved documentation defects and checks belong in the [log](log.md); append
new evidence when resolving these questions rather than erasing their history.
