# Diffuse Intensity Visualization Implementation Plan

## Overview
This document provides a detailed, checkpoint-based implementation plan for advanced 3D visualization features in the eryx project. The plan includes parallel development tracks for redundancy and comparison.

## Implementation Tracks

### Track A: Progressive Slicing Animation
**Primary Goal**: Animated GIF showing progressive reveal of 3D intensity data

### Track B: Interactive 3D Volume Rendering  
**Primary Goal**: WebGL-based interactive heatmap with dynamic controls

## Week 1: Foundation and Infrastructure

### Day 1-2: Project Setup and Data Layer

#### ✅ Checkpoint 1.1: Package Structure [COMPLETED]
**Tasks**:
```bash
mkdir -p eryx/visualization/{core,slicing,volume,interactive}
touch eryx/visualization/__init__.py
```
- ✅ Create module structure as specified in architecture
- ✅ Set up imports and __all__ exports
- ✅ Add to main eryx package imports

**Validation**: 
```python
from eryx.visualization import IntensityDataHandler
assert IntensityDataHandler is not None
```
**Status**: ✅ VERIFIED - All imports working

#### ✅ Checkpoint 1.2: Data Handler Implementation [COMPLETED]
**File**: `eryx/visualization/core/data_handler.py`
```python
class IntensityDataHandler:
    def __init__(self, data_source):
        # Reuse logic from visualize_diffuse.load_intensity_data()
        pass
    
    def load_data(self):
        # Support NPZ and NPY formats
        # Extract intensity, q_vectors, map_shape
        pass
    
    def get_statistics(self):
        # Min, max, mean, std, percentiles
        # Handle NaN values properly
        pass
```

**Tests**:
- ✅ Load all existing test data files
- ✅ Verify shape and data integrity
- ✅ Check NaN locations match expected lattice points
**Status**: ✅ 6 tests passing

### Day 3-4: Coordinate System and NaN Handling

#### ✅ Checkpoint 1.3: Coordinate Mapper [COMPLETED]
**File**: `eryx/visualization/core/coordinate_mapper.py`
```python
class CoordinateMapper:
    def __init__(self, A_inv, shape, sampling):
        # Store orthogonalization matrix
        # Cache grid coordinates
        pass
    
    def hkl_to_cartesian(self, hkl):
        # Use existing transformation: q = 2π * A_inv.T @ hkl
        pass
    
    def get_voxel_coordinates(self):
        # Return (x,y,z) for each voxel
        pass
```

**Validation**:
- ✅ Compare outputs with existing `generate_grid()` function
- ✅ Verify coordinate transformations are reversible
**Status**: ✅ 6 tests passing (including bug fix for matrix multiplication order)

#### ✅ Checkpoint 1.4: NaN Processing Pipeline [COMPLETED]
**File**: `eryx/visualization/core/nan_processor.py`
```python
class NaNProcessor:
    STRATEGIES = {
        'mask': lambda d: np.ma.masked_invalid(d),
        'zero': lambda d: np.nan_to_num(d, nan=0),
        'interpolate': lambda d: self._interpolate_nans(d),
        'remove': lambda d: d[~np.isnan(d)]
    }
    
    @staticmethod
    def apply_strategy(data, strategy='mask'):
        # Apply selected strategy
        # Return processed data and mask
        pass
```

**Tests**:
- ✅ Each strategy on synthetic data with known NaN patterns
- ✅ Performance benchmarks for 500k+ voxel datasets
- ✅ Visual comparison of results
**Status**: ✅ 10 NaN processor tests passing

### Day 5: Integration and Testing

#### ✅ Checkpoint 1.5: Core Integration Tests [COMPLETED]
**File**: `tests/test_visualization_core.py`
```python
def test_full_pipeline():
    # Load data -> Map coordinates -> Process NaNs
    handler = IntensityDataHandler('test_data.npz')
    data, coords, shape = handler.load_data()
    
    mapper = CoordinateMapper(A_inv, shape, sampling)
    xyz = mapper.get_voxel_coordinates()
    
    processor = NaNProcessor()
    clean_data = processor.apply_strategy(data, 'mask')
    
    assert clean_data.shape == data.shape
    assert not np.any(np.isnan(clean_data))
```

**Deliverables**:
- ✅ Working data pipeline
- ✅ Coordinate transformations  
- ✅ NaN handling strategies
- ✅ Test suite with >90% coverage
**Status**: ✅ ALL 24 TESTS PASSING - Phase 1 Complete!

## Week 2: Slicing Animation Development (Parallel Approaches)

### Day 6-7: Plane Mathematics and Slicing Core

#### Checkpoint 2.1: Plane Calculator
**File**: `eryx/visualization/slicing/plane_calculator.py`
```python
class PlaneCalculator:
    def __init__(self, volume_shape, center=None):
        self.shape = volume_shape
        self.center = center or np.array(volume_shape) / 2
    
    def plane_from_normal_distance(self, normal, distance):
        # Return plane equation: ax + by + cz + d = 0
        normal = normal / np.linalg.norm(normal)
        d = -distance
        return np.append(normal, d)
    
    def voxel_to_plane_distances(self, plane):
        # Compute signed distance for each voxel
        # Positive = above plane, negative = below
        pass
    
    def create_cutting_sequence(self, normal, n_frames=50):
        # Generate sequence from far to near
        # Handle bounding box correctly
        pass
```

**Validation**:
- Test with axis-aligned planes (should match array slicing)
- Verify distance calculations with known geometries
- Check plane sequences cover full volume

### Day 8-9: Matplotlib Implementation (Approach A)

#### Checkpoint 2.2: Matplotlib Slicer
**File**: `eryx/visualization/slicing/matplotlib_slicer.py`
```python
class MatplotlibSlicer:
    def __init__(self, volume, coordinates):
        self.volume = volume
        self.coords = coordinates
    
    def create_frame(self, plane):
        # Method 1: Voxel rendering with masking
        distances = self.calculate_distances(plane)
        mask = distances <= 0  # Below or on plane
        
        # Method 2: Extract slice on plane
        slice_data = self.interpolate_on_plane(plane)
        
        # Composite visualization
        return self.render_composite(mask, slice_data)
    
    def render_composite(self, volume_mask, slice_data):
        # 3D scatter/voxel plot for volume
        # 2D heatmap for slice
        # Combine in single figure
        pass
```

**Tests**:
- Single frame generation
- Visual quality assessment
- Performance: <2s per frame for 500k voxels

### Day 10: PyVista Implementation (Approach B - Optional)

#### Checkpoint 2.3: PyVista Slicer
**File**: `eryx/visualization/slicing/pyvista_slicer.py`
```python
class PyVistaSlicer:
    def __init__(self, volume, coordinates):
        import pyvista as pv
        self.mesh = self.create_mesh(volume, coordinates)
    
    def create_mesh(self, volume, coords):
        # Convert to PyVista ImageData or StructuredGrid
        mesh = pv.ImageData(dimensions=volume.shape)
        mesh["intensity"] = volume.flatten(order="F")
        return mesh
    
    def create_frame(self, plane):
        # Use native clipping
        clipped = self.mesh.clip(normal=plane[:3], origin=point)
        
        # Render to image
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(clipped, scalars="intensity")
        return plotter.screenshot()
```

**Validation**:
- Compare output with matplotlib version
- Check performance advantages
- Evaluate visual quality

### Day 11-12: Animation Generation

#### Checkpoint 2.4: Unified Animator
**File**: `eryx/visualization/slicing/animator.py`
```python
class SlicingAnimator:
    BACKENDS = {
        'matplotlib': MatplotlibSlicer,
        'pyvista': PyVistaSlicer
    }
    
    def __init__(self, volume, coordinates, backend='matplotlib'):
        self.slicer = self.BACKENDS[backend](volume, coordinates)
    
    def generate_animation(self, normal, n_frames=50, fps=10):
        calculator = PlaneCalculator(volume.shape)
        planes = calculator.create_cutting_sequence(normal, n_frames)
        
        frames = []
        for i, plane in enumerate(planes):
            print(f"Generating frame {i+1}/{n_frames}")
            frame = self.slicer.create_frame(plane)
            frames.append(frame)
        
        return self.create_gif(frames, fps)
    
    def create_gif(self, frames, fps):
        # Use imageio or PIL for GIF creation
        # Optimize compression
        pass
```

**Deliverables**:
- Working animation for both backends
- Sample GIFs for different normal vectors
- Performance comparison report

## Week 3: Volume Rendering (Multiple Backends)

### Day 13-14: Plotly Volume Rendering

#### Checkpoint 3.1: Plotly Volume Implementation
**File**: `eryx/visualization/volume/plotly_backend.py`
```python
class PlotlyVolumeRenderer:
    def __init__(self, volume, coordinates):
        self.volume = volume
        self.coords = coordinates
    
    def create_volume_plot(self):
        import plotly.graph_objects as go
        
        # Method 1: True volume rendering
        fig = go.Figure(data=go.Volume(
            x=self.coords[:, 0].flatten(),
            y=self.coords[:, 1].flatten(),
            z=self.coords[:, 2].flatten(),
            value=self.volume.flatten(),
            isomin=0.1,
            isomax=0.9,
            opacity=0.1,
            surface_count=15,
            colorscale='Viridis'
        ))
        
        return fig
    
    def create_isosurface_plot(self, levels=[0.2, 0.5, 0.8]):
        # Method 2: Multiple isosurfaces
        fig = go.Figure()
        for level in levels:
            fig.add_trace(go.Isosurface(
                x=self.coords[:, 0].flatten(),
                y=self.coords[:, 1].flatten(),
                z=self.coords[:, 2].flatten(),
                value=self.volume.flatten(),
                isomin=level,
                isomax=level,
                surface=dict(count=1),
                opacity=0.3
            ))
        return fig
```

**Validation**:
- Test with synthetic data (sphere, cube)
- Verify NaN handling
- Check interactive performance

### Day 15: ipyvolume Implementation

#### Checkpoint 3.2: ipyvolume Backend
**File**: `eryx/visualization/volume/ipyvolume_backend.py`
```python
class IpyvolumeRenderer:
    def __init__(self, volume, coordinates):
        import ipyvolume as ipv
        self.volume = volume
        self.coords = coordinates
    
    def create_widget(self):
        ipv.figure()
        ipv.volshow(self.volume, level=[0.1, 0.5, 0.9])
        
        # Add controls
        return ipv.gcc()  # Get current figure
    
    def add_controls(self):
        # Opacity slider
        # Colormap selector
        # Level adjustment
        pass
```

**Tests**:
- Jupyter notebook integration
- Widget responsiveness
- Memory usage monitoring

### Day 16-17: Performance Optimization and Comparison

#### Checkpoint 3.3: Backend Comparison
**File**: `eryx/visualization/benchmarks.py`
```python
def benchmark_backends():
    results = {}
    
    for backend in ['plotly', 'ipyvolume', 'matplotlib']:
        for size in [100_000, 500_000, 1_000_000]:
            data = generate_test_data(size)
            
            start = time.time()
            renderer = get_renderer(backend, data)
            fig = renderer.render()
            
            results[f"{backend}_{size}"] = {
                'time': time.time() - start,
                'memory': get_memory_usage(),
                'fps': measure_interactivity(fig)
            }
    
    return results
```

**Deliverables**:
- Performance comparison table
- Memory usage analysis
- Recommendation matrix

## Week 4: Interactivity and Controls

### Day 18-19: Control Implementation

#### Checkpoint 4.1: Universal Controls
**File**: `eryx/visualization/interactive/controls.py`
```python
class VolumeControls:
    def __init__(self, renderer):
        self.renderer = renderer
        self.controls = {}
    
    def add_rotation_control(self, axis='y', speed=1.0):
        # Programmed rotation
        def rotate():
            angle = 0
            while self.rotating:
                self.renderer.set_camera_angle(axis, angle)
                angle += speed
                time.sleep(1/30)  # 30 FPS
        
        self.controls['rotation'] = rotate
    
    def add_range_control(self, min_val=0, max_val=1):
        # Dynamic range slider
        def update_range(vmin, vmax):
            self.renderer.set_intensity_range(vmin, vmax)
        
        self.controls['range'] = update_range
    
    def add_opacity_control(self):
        # Opacity adjustment
        pass
    
    def add_colormap_selector(self):
        # Colormap choices
        pass
```

### Day 20: Dash Web Application (Optional)

#### Checkpoint 4.2: Dash Integration
**File**: `eryx/visualization/interactive/dash_app.py`
```python
import dash
from dash import dcc, html, Input, Output

def create_dash_app(data):
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        dcc.Graph(id='3d-volume'),
        
        html.Label('Rotation Speed'),
        dcc.Slider(id='rotation-speed', min=0, max=5, value=1),
        
        html.Label('Intensity Range'),
        dcc.RangeSlider(id='intensity-range', min=0, max=1, value=[0.1, 0.9]),
        
        html.Label('Opacity'),
        dcc.Slider(id='opacity', min=0, max=1, value=0.5),
    ])
    
    @app.callback(
        Output('3d-volume', 'figure'),
        [Input('rotation-speed', 'value'),
         Input('intensity-range', 'value'),
         Input('opacity', 'value')]
    )
    def update_figure(speed, range_vals, opacity):
        # Update and return figure
        pass
    
    return app
```

### Day 21: Animation Features

#### Checkpoint 4.3: Advanced Animations
**File**: `eryx/visualization/interactive/animations.py`
```python
class VolumeAnimator:
    def __init__(self, renderer):
        self.renderer = renderer
    
    def create_rotation_sequence(self, duration=10, fps=30):
        # 360-degree rotation
        frames = []
        for angle in np.linspace(0, 360, duration * fps):
            self.renderer.set_camera_angle('y', angle)
            frames.append(self.renderer.capture_frame())
        return frames
    
    def create_range_animation(self, duration=5):
        # Animate dynamic range
        frames = []
        for t in np.linspace(0, 1, duration * 30):
            vmin = 0.5 * (1 + np.sin(2*np.pi*t))
            self.renderer.set_intensity_range(0, vmin)
            frames.append(self.renderer.capture_frame())
        return frames
    
    def export_video(self, frames, output='animation.mp4'):
        # Use ffmpeg or imageio
        pass
```

**Deliverables**:
- Interactive controls for all backends
- Optional Dash application
- Animation export capabilities

## Week 5: Integration, Testing, and Documentation

### Day 22-23: API Unification

#### Checkpoint 5.1: High-Level API
**File**: `eryx/visualization/__init__.py`
```python
def visualize_diffuse_3d(data_path, mode='volume', **kwargs):
    """
    High-level API for 3D visualization.
    
    Parameters
    ----------
    data_path : str
        Path to NPZ file with intensity data
    mode : str
        'volume' for 3D rendering, 'slice' for animation
    **kwargs : dict
        Backend-specific options
    
    Returns
    -------
    figure : object
        Visualization figure (type depends on backend)
    """
    # Load data
    handler = IntensityDataHandler(data_path)
    data, coords, shape = handler.load_data()
    
    # Process NaNs
    processor = NaNProcessor()
    clean_data = processor.apply_strategy(data, kwargs.get('nan_strategy', 'mask'))
    
    # Create visualization
    if mode == 'volume':
        backend = kwargs.get('backend', 'plotly')
        renderer = get_volume_renderer(backend, clean_data, coords)
        return renderer.create_interactive_plot(**kwargs)
    
    elif mode == 'slice':
        normal = kwargs.get('normal', [1, 1, 1])
        animator = SlicingAnimator(clean_data, coords)
        return animator.generate_animation(normal, **kwargs)
```

### Day 24: Comprehensive Testing

#### Checkpoint 5.2: Test Suite
**File**: `tests/test_visualization_integration.py`
```python
class TestVisualizationIntegration:
    def test_volume_all_backends(self):
        # Test each backend with same data
        for backend in ['plotly', 'ipyvolume', 'matplotlib']:
            fig = visualize_diffuse_3d('test.npz', mode='volume', backend=backend)
            assert fig is not None
    
    def test_slicing_animation(self):
        # Test animation generation
        anim = visualize_diffuse_3d('test.npz', mode='slice', normal=[1,0,0])
        assert os.path.exists('animation.gif')
    
    def test_large_dataset(self):
        # Test with 1M+ voxels
        large_data = generate_large_test_data(1_000_000)
        fig = visualize_diffuse_3d(large_data, mode='volume')
        assert measure_fps(fig) > 10
    
    def test_nan_handling(self):
        # Test all NaN strategies
        for strategy in ['mask', 'zero', 'interpolate', 'remove']:
            fig = visualize_diffuse_3d('test.npz', nan_strategy=strategy)
            assert not has_rendering_artifacts(fig)
```

### Day 25: Documentation

#### Checkpoint 5.3: User Documentation
**Files**:
- `docs/visualization_guide.md` - User guide with examples
- `docs/api_reference.md` - Complete API documentation
- `examples/visualization_gallery.ipynb` - Jupyter notebook with examples

**Content Structure**:
```markdown
# Visualization Guide

## Quick Start
```python
from eryx.visualization import visualize_diffuse_3d

# Simple volume rendering
fig = visualize_diffuse_3d('data.npz', mode='volume')
fig.show()

# Slicing animation
anim = visualize_diffuse_3d('data.npz', mode='slice', normal=[1, 1, 0])
anim.save('output.gif')
```

## Advanced Usage
...

## Gallery
[Images and animations showcasing capabilities]
```

### Day 26: Performance Profiling

#### Checkpoint 5.4: Optimization
**Tasks**:
- Profile with cProfile/line_profiler
- Identify bottlenecks
- Implement caching for repeated operations
- Add LOD rendering for large datasets

**Target Metrics**:
- Frame generation: <1s for 500k voxels
- Interactive FPS: >15 for 500k voxels
- Memory usage: <2x data size

### Day 27-28: Final Integration and Release

#### Checkpoint 5.5: Release Preparation
**Tasks**:
- [ ] Code review and cleanup
- [ ] Update main eryx documentation
- [ ] Create example scripts
- [ ] Performance benchmarks documentation
- [ ] Update requirements.txt with optional dependencies
- [ ] Create visualization showcase/gallery

**Final Deliverables**:
1. Complete visualization package
2. Test suite with >90% coverage
3. User and API documentation
4. Example notebooks and scripts
5. Performance benchmark results
6. Backend comparison guide

## Risk Management and Contingencies

### Risk 1: Performance Issues with Large Datasets
**Mitigation**:
- Implement progressive subsampling
- Add data decimation options
- Cache preprocessed data
- **Contingency**: Focus on smaller ROIs initially

### Risk 2: Backend Compatibility Issues
**Mitigation**:
- Make backends optional (extras_require)
- Maintain matplotlib as fallback
- **Contingency**: Prioritize Plotly as primary backend

### Risk 3: Complex Coordinate Transformations
**Mitigation**:
- Extensive testing against existing code
- Visual validation with known patterns
- **Contingency**: Use existing transformation code directly

## Success Metrics

### Week 1 Success Criteria
- [ ] Data pipeline functional
- [ ] Coordinate mapping verified
- [ ] NaN handling tested

### Week 2 Success Criteria
- [ ] Slicing animation working
- [ ] At least one backend complete
- [ ] GIF export functional

### Week 3 Success Criteria
- [ ] Volume rendering operational
- [ ] Multiple backends compared
- [ ] Performance acceptable (>10 FPS)

### Week 4 Success Criteria
- [ ] Interactive controls working
- [ ] Animations smooth
- [ ] Optional Dash app functional

### Week 5 Success Criteria
- [ ] API unified and documented
- [ ] Tests passing (>90% coverage)
- [ ] Documentation complete
- [ ] Performance targets met

## Notes for Implementation

### Priority Order
1. **Essential**: Matplotlib slicing + Plotly volume
2. **Important**: Interactive controls, NaN handling
3. **Nice-to-have**: PyVista, ipyvolume, Dash app

### Development Tips
- Start with 2D slicing, extend to 3D
- Use synthetic data for initial testing
- Profile early and often
- Keep backends modular for easy swapping
- Document assumptions about coordinate systems

### Testing Data
Create synthetic test cases:
- Gaussian blob (simple)
- Lattice with NaNs (realistic)
- Large random data (performance)
- Known patterns (validation)