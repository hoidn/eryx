# Eryx 3D Visualization Module

This module provides advanced 3D visualization capabilities for diffuse scattering intensity data, with multiple rendering backends and interactive features.

## Quick Start

### K3D Interactive Clipping (Production Ready) ✅

```python
from eryx.visualization.volume.k3d import create_clipped_visualization
import numpy as np

# Load your diffuse intensity data
data = np.load('torch_diffuse_intensity.npy')
volume_3d = data.reshape(41, 41, 41)  # Perfect cube: 68921 = 41³

# Create interactive visualization with clipping
plot = create_clipped_visualization(
    volume_3d,
    bounds=[-2, 2, -2, 2, -2, 2],      # q-space bounds
    clipping_planes=[[1, 0, 0, 0]],    # qx > 0 half-space
    color_range_percentile=80,         # Auto-adjust for bright spots
    alpha_coef=15.0                    # Transparency control
)

# Export to interactive HTML
with open('diffuse_intensity_clipped.html', 'w') as f:
    f.write(plot.get_snapshot())
```

### Classic Matplotlib Slicing

```python
from eryx.visualization.slicing import SlicingAnimator, PlaneCalculator
from eryx.visualization.core import IntensityDataHandler

# Load and process data
handler = IntensityDataHandler('data.npz')
data, coords, shape = handler.load_data()

# Create slicing animation
animator = SlicingAnimator(data, coords, backend='matplotlib')
animation = animator.generate_animation(
    normal=[1, 1, 1],     # Diagonal cutting plane
    n_frames=50,
    fps=10
)
animation.save('slicing_animation.gif')
```

## Available Visualization Methods

### 1. K3D Volume Rendering (WebGL) 🌟 **Recommended**
- **Status**: Production ready
- **Quality**: Photorealistic volume rendering
- **Features**: Real-time clipping planes, interactive controls, browser animation
- **Export**: Standalone HTML files
- **Location**: `eryx/visualization/volume/k3d/`

#### Key Files:
- `final_k3d_clipping_solution.py` - Main production implementation  
- `k3d_spherical_clipping.py/.ipynb` - Spherical & octant clipping
- `k3d_interactive_clipping.py/.ipynb` - Interactive GUI controls
- `FINAL_WORKING_SOLUTION.md` - Complete documentation and examples

#### Advanced Features:
- **Spherical Clipping**: Show data inside/outside sphere
- **Octant Exclusion**: Remove 1/8 of data to reveal internal structure
- **Combined Masks**: Sphere + octant for cross-section views

### 2. Progressive Slicing Animation (Matplotlib)
- **Status**: Functional, optimizing performance
- **Quality**: Good for publications, scientific analysis
- **Features**: Arbitrary cutting planes, GIF export, frame-by-frame control
- **Export**: Animated GIF, MP4, individual frames
- **Location**: `eryx/visualization/slicing/`

### 3. Volume Rendering Backends
- **Plotly**: Interactive 3D in browsers (planned)
- **PyVista**: Professional 3D rendering (optional)
- **Matplotlib**: Universal fallback for basic 3D

## Architecture Overview

```
eryx/visualization/
├── core/                    # Foundation components
│   ├── data_handler.py     # Data loading, NPZ/NPY support
│   ├── coordinate_mapper.py # Miller index ↔ Cartesian transforms  
│   └── nan_processor.py    # NaN handling strategies
├── slicing/                # Animation and slicing
│   ├── plane_calculator.py # Cutting plane mathematics
│   ├── matplotlib_slicer.py # Matplotlib rendering
│   └── animator.py         # GIF generation
├── volume/                 # 3D volume rendering
│   ├── k3d_backend.py      # Base k3d renderer
│   └── k3d/               # Specialized implementations
│       ├── final_k3d_clipping_solution.py  # Production system
│       ├── k3d_spherical_clipping.py       # Radial clipping
│       ├── k3d_interactive_clipping.py     # GUI controls
│       └── FINAL_WORKING_SOLUTION.md       # Documentation
└── interactive/            # UI controls and widgets
```

## K3D Clipping Planes Reference

### Basic Syntax
```python
# Format: [nx, ny, nz, d] where (nx,ny,nz) is normal vector, d is distance
plot.clipping_planes = [[1, 0, 0, 0]]    # qx > 0 half-space
plot.clipping_planes = [[0, 0, 1, 0.5]]  # qz > 0.5 plane
plot.clipping_planes = [[1, 1, 1, 0]]    # Diagonal cut through origin
```

### Common Analysis Patterns
```python
# Positive half-spaces (common for diffuse scattering)
plot.clipping_planes = [[1, 0, 0, 0]]           # qx > 0
plot.clipping_planes = [[0, 1, 0, 0]]           # qy > 0  
plot.clipping_planes = [[0, 0, 1, 0]]           # qz > 0

# Central slices
plot.clipping_planes = [[1, 0, 0, 0.1]]         # Near qx = -0.1 plane
plot.clipping_planes = [[0, 0, 1, -0.5]]        # qz < 0.5 region

# Multiple constraints
plot.clipping_planes = [[1,0,0,0], [0,1,0,0]]   # First octant (qx>0, qy>0)
```

### Animation Examples
```python
# Linear sweep animation  
for i in range(30):
    pos = -2 + 4 * (i / 29)  # -2 to +2
    plot.clipping_planes = [[1, 0, 0, pos]]
    # Export frame...

# Rotating plane
for i in range(60):
    angle = 2 * np.pi * i / 60
    normal = [np.cos(angle), np.sin(angle), 0]
    plot.clipping_planes = [normal + [0]]
    # Export frame...
```

## Data Requirements

### Supported Formats
- **NPZ files**: `intensity`, `q_vectors`, `map_shape` arrays
- **NPY files**: Direct intensity arrays
- **NumPy arrays**: In-memory data

### Typical Data Shapes
- **Grid data**: Regular sampling, e.g. (25, 103, 175) or (41, 41, 41)
- **Arbitrary q**: Scattered points, any length
- **Large datasets**: >400k voxels supported

### Data Preprocessing
```python
from eryx.visualization.core import NaNProcessor

# Handle NaN values at lattice points
processor = NaNProcessor()
clean_data = processor.apply_strategy(data, strategy='mask')  # or 'zero', 'interpolate'
```

## Performance Guidelines

### K3D Recommendations
- **Optimal size**: 41³ to 100³ voxels for smooth interaction
- **Large datasets**: Use subsampling or level-of-detail
- **Browser compatibility**: Modern browsers with WebGL support
- **Memory**: ~2x data size for processing

### Matplotlib Recommendations  
- **Frame generation**: <2s per frame for 500k voxels
- **Animation length**: 20-60 frames for smooth motion
- **Resolution**: Balance quality vs. file size

## Integration with Eryx Pipeline

### From OnePhonon Results
```python
from eryx import OnePhonon
from eryx.visualization.volume.k3d import create_clipped_visualization

# Run simulation
model = OnePhonon('protein.pdb', hsampling=32, ksampling=32, lsampling=32)
intensity = model.apply_disorder()

# Visualize results
plot = create_clipped_visualization(
    intensity.reshape(model.shape),
    bounds=model.get_q_bounds(),
    clipping_planes=[[1, 0, 0, 0]]  # Show qx > 0 region
)
```

### Custom Analysis Workflows
```python
# Load pre-computed results
data = np.load('torch_diffuse_intensity.npy')
volume = data.reshape(41, 41, 41)

# Planar clipping examples
analyses = {
    'full_volume': [],                           # No clipping
    'positive_qx': [[1, 0, 0, 0]],             # qx > 0
    'central_slice': [[0, 0, 1, -0.1], [0, 0, 1, 0.1]], # |qz| < 0.1
}

for name, planes in analyses.items():
    plot = create_clipped_visualization(volume, clipping_planes=planes)
    with open(f'{name}.html', 'w') as f:
        f.write(plot.get_snapshot())

# Spherical and octant clipping
from eryx.visualization.volume.k3d.k3d_spherical_clipping import SphericalClippingController

controller = SphericalClippingController()
controller.sphere_radius = 15.0
controller.clip_inside = True  # Show inside sphere

# Exclude one octant to reveal structure
controller.octant_cut = True
controller.octant_mode = 'first'  # Excludes x>0, y>0, z>0 octant
controller.method = 'masking'
controller.update_clipping()
```

## Documentation Links

- **Architecture**: [`../../docs/visualization/VISUALIZATION_ARCHITECTURE.md`](../../docs/visualization/VISUALIZATION_ARCHITECTURE.md) - Technical specifications
- **Implementation Plan**: [`../../docs/visualization/IMPLEMENTATION_PLAN.md`](../../docs/visualization/IMPLEMENTATION_PLAN.md) - Development roadmap  
- **Phase 1 Report**: [`../../docs/visualization/PHASE1_COMPLETION_REPORT.md`](../../docs/visualization/PHASE1_COMPLETION_REPORT.md) - Foundation completion
- **Phase 2 Report**: [`../../docs/visualization/PHASE2_COMPLETION_REPORT.md`](../../docs/visualization/PHASE2_COMPLETION_REPORT.md) - Slicing animation completion
- **Format Decision**: [`../../docs/visualization/ANIMATION_FORMAT_DECISION.md`](../../docs/visualization/ANIMATION_FORMAT_DECISION.md) - Design rationale
- **K3D Complete Guide**: [`volume/k3d/FINAL_WORKING_SOLUTION.md`](volume/k3d/FINAL_WORKING_SOLUTION.md) - Comprehensive documentation
- **Main Project Guide**: [`../../CLAUDE.md`](../../CLAUDE.md) - Integration with eryx project

## Quick Troubleshooting

### K3D Issues
- **Import error**: Install with `pip install k3d`
- **Blank visualization**: Check data range, try `np.nan_to_num(data)`
- **Slow performance**: Reduce data size or adjust `alpha_coef`
- **Browser issues**: Try Chrome/Firefox with WebGL enabled

### Data Issues  
- **Wrong shape**: Ensure 3D array, use `.reshape()` if needed
- **NaN values**: Use `NaNProcessor` or `np.nan_to_num()`
- **Poor contrast**: Adjust `color_range_percentile` parameter
- **Memory errors**: Use data subsampling or chunking

## Research Applications

This visualization system enables:

- **Anisotropy Analysis**: Directional clipping reveals preferred orientations
- **Interactive Exploration**: Real-time manipulation of view parameters
- **Publication Figures**: High-quality renderings for papers and presentations  
- **Data Validation**: Visual inspection of simulation results
- **Comparative Analysis**: Multiple datasets with consistent parameters