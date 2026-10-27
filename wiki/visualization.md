# Visualizing diffuse intensity

Last reviewed: 2026-10-26

[IntensityDataHandler](../eryx/visualization/core/data_handler.py) loads intensity
and optional q-vector/map-shape metadata from NPZ/NPY files.
[CoordinateMapper](../eryx/visualization/core/coordinate_mapper.py) and
[NaNProcessor](../eryx/visualization/core/nan_processor.py) support coordinate and
masked-data handling. Preserve [sampling metadata](sampling.md) when exporting
simulations; shape inference alone does not establish the physical coordinates.

The [visualization package](../eryx/visualization/__init__.py) exposes K3D volume
rendering and Matplotlib slicing when their imports succeed. K3D is an optional
dependency absent from the root [requirements](../requirements.txt); availability
and browser compatibility must be checked in the environment being used.

## Existing references

- [Visualization documentation](../docs/visualization/README.md) — backend routes.
- [K3D quick start](../docs/visualization/k3d/getting-started/QUICK_START.md) — existing usage examples.
- [HTML JavaScript API](../docs/visualization/k3d/api/HTML_JAVASCRIPT_API.md) and
  [JavaScript patterns](../docs/visualization/k3d/api/JAVASCRIPT_PATTERNS.md) —
  recorded browser integration findings; verify against the generated export.
- [Performance lessons](../docs/visualization/k3d/api/PERFORMANCE_OPTIMIZATION_LESSONS.md) —
  historical masking and clipping observations, not fresh benchmarks.
- [Slicing example](../examples/create_slicing_animation.py) — consumes
  `tmp/pytorch-quickstart/results.npz` and writes `slicing.gif` beside it;
  requires the bundled simulation output, imageio, and pillow.

Prior “production-ready” labels and phase completion reports are historical
claims, not browser tests performed during this wiki review. Related:
[validation](validation.md), [open questions](open-questions.md).
