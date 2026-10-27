# Diffuse scattering visualization

Start with the [PyTorch workflow](../../docs/PYTORCH_QUICKSTART.md) for a simulation
and static plots, or the [K3D quickstart](../../docs/visualization/k3d/getting-started/QUICK_START.md)
for an interactive volume exported to HTML.

## Data contract

NPZ files contain `intensity`, `q_vectors`, and `map_shape`; the quickstart also
saves `hkl_grid`. Preserve the original shape and point order. Arbitrary q points
need resampling or a scatter plot, not a guessed cubic reshape.

`IntensityDataHandler.load_data()` returns **(q_vectors, intensity, map_shape)**.
Intensity may still be flat. Reshape it with the returned shape before passing
it to a volume renderer. A 3D NPY or NumPy array already supplies its shape; a
flat NPY alone does not.

## Python entry points

| Entry point | Input and result |
|---|---|
| `eryx.visuals.visualize_central_slices` | 3D NumPy array; saves a static plot with `output=...` |
| `eryx.visualization.visualize_diffuse_3d` | Array or filename; K3D returns `(plot, volume)` |
| `eryx.visualization.volume.k3d.create_clipped_visualization` | 3D array only; returns `(plot, volume)` |
| `eryx.visualization.create_slicing_animation` | NPZ filename; returns output GIF/video filename |

K3D is optional: install `k3d` in the active environment. Its helper accepts no
`bounds`, `clipping_planes`, `alpha_coef`, or percentile keyword arguments.
Set properties on the returned plot/volume instead. The helper's default bounds
are fixed display coordinates, not inferred crystallographic coordinates; the
linked example explicitly uses voxel indices and the correct volume axis order.

For GIF animation, install `imageio` and `pillow`, then run
`python -m examples.create_slicing_animation` after the simulation example.
The animation uses index-space planes and is intended for qualitative exploration.
Do not interpret its axes as calibrated Cartesian q coordinates.

Retain NaNs for analysis. Filling or interpolating missing values for rendering
must operate on a copy. Difference maps need a diverging scale centered at zero.

[Documentation index](../../docs/visualization/README.md)
