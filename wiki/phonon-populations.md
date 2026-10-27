# Phonon populations and PDOS

Last reviewed: 2026-10-26

The PyTorch [OnePhonon implementation](../eryx/models_torch.py) accepts a
two-column frequency/density file through `pdos_path`. The loader converts THz
to angular frequency and prepares tensors; interpolation supplies population
factors during phonon computation. See the [PDOS reference](../docs/PDOS.md) for
input validation, modes, defaults, and extraction.

`direct` mode interpolates supplied population values. `thermal` mode requires
an explicit positive temperature and rebuilds temperature-dependent normalization
and interpolation during each mode evaluation. The old guide's Bose–Einstein
formula did not describe these operations. The [open question](open-questions.md)
records the discrepancy without choosing a new physical model.

`generate_pdos` creates a histogram from computed modes. A histogram density is
not automatically a physically equivalent population input: reusing it in direct
mode changes the meaning of the values. Validate the interpretation for the
experiment instead of assuming a save/reload round trip preserves intensity.

[PDOS tests](../tests/test_models_torch_pdos.py) cover interpolation, integration,
and input errors. [Numerical gradient tests](../tests/test_differentiability.py)
check intensity derivatives and repeated optimization. Density samples remain
leaves in both modes, and the combined mode covariance includes mode-rotation
derivatives. Follow the [optimization recipe](../docs/PYTORCH_QUICKSTART.md#optimization)
for refresh requirements and first-order/nonsmooth limits. These checks do not
establish the physical validity of the population law.

Related: [architecture](architecture.md), [sampling](sampling.md),
[validation](validation.md), and [source catalog](sources.md).
