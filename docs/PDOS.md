# Phonon weights and density-of-states diagnostics

Start with the [PyTorch workflow](PYTORCH_QUICKSTART.md) for an executed simulation
and exported histogram. Always import the model from `eryx.models_torch`.

## Current API

| Parameter/method | Behavior |
|---|---|
| `pdos_path` | Optional whitespace-separated file with exactly two columns and at least two rows |
| First column | Frequency in THz; finite and strictly increasing |
| Second column | Finite density/population values; use nonnegative values for physical weights |
| `pdos_mode="direct"` | Interpolate supplied weights and use them as `Winv` |
| `pdos_mode="thermal"` | Divide input weights by a Boltzmann factor during loading, then apply a Boltzmann factor at interpolation queries |
| `temperature_k` | Defaults to None; a positive value is required with a thermal input file |
| `generate_pdos(bins=100, density=True)` | Histogram mode frequencies from saved eigenvalues; returns two NumPy columns |
| `get_frequencies_thz()` | Flatten inferred frequencies and remove NaN/inf |
| `apply_disorder(rank=-1, use_data_adp=False)` | Sum all modes with computed ADPs; select a rank or PDB ADPs explicitly |

Supply `pdos_path`, `pdos_mode`, and, for thermal mode, `temperature_k` as
constructor arguments in the quickstart script. No PDOS input is needed to
generate its diagnostic histogram. Grid sampling consists of three
(minimum, maximum, oversampling) tuples, including in arbitrary-q GNM mode.

## Interpretation limits

These APIs currently implement experimental mode weighting. They do not establish
a calibrated thermal spectrum or an identity-preserving PDOS export/import cycle:

- Without a PDOS file, `Winv` is inverse eigenvalue. With a file it becomes an
  interpolated population weight. Frequency extraction uses separately saved
  eigenvalues, so changing populations does not change the reported frequencies.
  Older states without saved eigenvalues fall back to inversion of `Winv`.
- Frequency extraction uses `sqrt(eigenvalue) / (2π) * 100`, an empirical scale.
  PDOS queries use the matching rad/s conversion, `sqrt(eigenvalue) * 10^14`.
  This makes the two paths internally consistent; it does not establish a
  physical calibration against measured frequencies.
- Thermal mode uses exponential Boltzmann factors as described above. It does
  not implement the Bose–Einstein occupation formula previously stated here.
- Interpolation clamps out-of-range queries to the nearest endpoint. Check
  frequency coverage; endpoint weights are not evidence for an unmeasured band.
- Eigenvectors are detached in the phonon calculation. Autograd availability
  is not a guarantee of full structural or force-constant derivatives.

## Pump-probe demonstration

[pump_probe_validation.py](../pump_probe_validation.py) modifies mode weights on
one model, computes baseline and pumped maps, and saves their difference. It uses
`tests/pdbs/6o2h_clean.pdb`, CPU, and a fixed 65×65×65 grid. Run its `--help` or
`--list-examples` to inspect options before launching the larger calculation.

Supported options are `--pump-magnitude`, `--pump-energy-percentile`,
`--pump-type` (delta or gaussian), `--pump-width`, and `--slice-idx`.
There is no `--pdos-file` option. The percentile chooses a mode's location in the
frequency distribution; it is not an absolute pump energy. Frequency and width
labels inherit the empirical scale above.

The script writes fixed thermal/pumped/difference filenames in the current
directory and replaces prior outputs. Plot differences with a diverging scale
centered at zero and retain a common valid-point mask for quantitative analysis.
