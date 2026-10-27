# Interactive diffuse intensity volumes

Export the PyTorch example's intensity to an HTML volume for exploration.
This example uses explicit voxel-index coordinates, not Cartesian q coordinates.

## Prerequisites

1. Run the [PyTorch example](../../../PYTORCH_QUICKSTART.md) from the repository
   root to create `tmp/pytorch-quickstart/results.npz`. It uses bundled data.
2. Install the optional renderer with `python -m pip install k3d` in the same
   environment. A browser with WebGL is needed to view the export.
3. Save the following as `view_volume.py` in the repository root and run
   `python view_volume.py`.

```python
from pathlib import Path
import numpy as np
from eryx.visualization.volume.k3d import create_clipped_visualization

SOURCE = Path("tmp/pytorch-quickstart/results.npz")
OUTPUT = SOURCE.with_name("volume.html")

with np.load(SOURCE) as data:
    shape = tuple(data["map_shape"])
    intensity = data["intensity"].reshape(shape)
finite = intensity[np.isfinite(intensity)]
assert finite.size and finite.max() > 0, "Need finite positive intensity"

# K3D expects z,y,x; the saved simulation volume is h,k,l.
display = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
display = display.transpose(2, 1, 0).astype(np.float32)
plot, volume = create_clipped_visualization(display)
volume.bounds = [-0.5, shape[0] - 0.5,
                 -0.5, shape[1] - 0.5,
                 -0.5, shape[2] - 0.5]
volume.color_range = [0, float(np.percentile(finite, 99))]
volume.alpha_coef = 12.0
plot.axes = ["h index", "k index", "l index"]
OUTPUT.write_text(plot.get_snapshot())
print(OUTPUT.resolve())
```

## What happens

- Load the recorded intensity and shape; never infer shape from point count.
- Fill missing values in a separate display array, preserving the source NPZ.
- Transpose axes and set bounds so x, y, z correspond to h, k, l voxel indices.
- Set contrast from finite intensities and write `volume.html`, replacing a
  previous export at that path. Open it in a browser.

The helper accepts only the volume array and returns **(plot, volume)**. Change
`volume.bounds`, `volume.color_range`, or `volume.alpha_coef` after construction.
For clipping in Python/Jupyter, set `plot.clipping_planes`; do not pass clipping
keywords to the helper. Verify clipping in the exported browser view separately.

The h/k/l grid is not generally an orthogonal Cartesian q grid. Physical q-space
rendering needs the lattice transform or resampling, rather than simply assigning
q minima and maxima to an axis-aligned volume. For quantitative analysis, use the
saved q vectors and preserve the finite-value mask.
