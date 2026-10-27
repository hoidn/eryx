# Sampling and result coordinates

Last reviewed: 2026-10-26

Both [NumPy](../eryx/map_utils.py) and [PyTorch](../eryx/map_utils_torch.py)
`generate_grid` accept `(minimum, maximum, oversampling)` for each h/k/l axis.
The third value is samples per reciprocal-lattice interval, not total points.
The PyTorch implementation computes
`int(oversampling * (maximum - minimum) + 1)` points per axis. Thus `[-2, 2, 8]`
gives 33 points along an axis. Use the returned `map_shape` when reshaping.

The grid utility converts Miller indices to Cartesian q-vectors using
`2π * (A_inv.T @ hkl_grid.T).T`. Miller coordinates and Cartesian q-vectors are
different representations; use matching conventions when comparing results.

## Arbitrary q-vectors

[OnePhonon.__init__](../eryx/models_torch.py) accepts a PyTorch tensor shaped
`(N, 3)` as `q_vectors`. With the default `model='gnm'`, **all three h/k/l sampling
tuples are still required** for ADP calculation, including in arbitrary-q mode.
Supplying only q-vectors is therefore not a complete default-GNM invocation.

Arbitrary point sets do not necessarily form a regular volume. Preserve their
coordinates and ordering; do not infer a cubic grid from a point count alone.
Resolution masks can produce NaNs, so comparisons and visualization must account
for masked positions. See [validation](validation.md).

## Saved results

[run_numpy.py](../run_numpy.py) and [run_torch.py](../run_torch.py) save NPZ files
containing `q_vectors`, `intensity`, and `map_shape`. Those fields connect the
simulation to the [visualization data handler](../eryx/visualization/core/data_handler.py).
Prefer retaining this metadata over saving only a flattened NPY intensity array.
The grid drivers currently use the same 5zck PDB and h/k/l sampling. Still check
all parameters, coordinate order, and masks before interpreting output agreement
as an equivalence result; see [validation](validation.md).

Related: [architecture](architecture.md), [visualization](visualization.md), and
the [historical q-vector design notes](../docs/qvec_conventions.md).
