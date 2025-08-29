# Advanced Diffuse Intensity Visualization Architecture

## Executive Summary

This document outlines the architecture for implementing advanced 3D visualization capabilities for diffuse scattering intensity data in the eryx project. The system will provide two primary visualization modes:

1. **Progressive Slicing Animation**: Animated GIF showing progressive reveal of 3D data along arbitrary cutting planes
2. **Interactive 3D Volume Rendering**: WebGL-based interactive heatmap with dynamic range control and rotation

## System Overview

### Design Principles
- **Modularity**: Separate data handling, rendering, and interaction components
- **Performance**: Handle datasets with >400k voxels efficiently
- **Robustness**: Graceful handling of NaN values at lattice points
- **Flexibility**: Support multiple rendering backends with common interfaces
- **Reusability**: Build on existing eryx visualization infrastructure

### Technical Constraints
- Maintain compatibility with existing NumPy/PyTorch dual implementation
- Preserve Miller index (h,k,l) coordinate system
- Handle typical data shapes (e.g., 25×103×175 voxels)
- Support both grid-based and arbitrary q-vector data

## Component Architecture

### Core Components

```
eryx/
├── visualization/
│   ├── __init__.py
│   ├── core/
│   │   ├── data_handler.py      # Data loading and preprocessing
│   │   ├── coordinate_mapper.py  # Coordinate system transformations
│   │   └── nan_processor.py      # NaN handling strategies
│   ├── slicing/
│   │   ├── plane_calculator.py   # Plane geometry calculations
│   │   ├── slicer.py            # Volume slicing algorithms
│   │   └── animator.py          # Animation generation
│   ├── volume/
│   │   ├── renderer.py          # Volume rendering interface
│   │   ├── plotly_backend.py    # Plotly volume implementation
│   │   ├── pyvista_backend.py   # PyVista implementation (optional)
│   │   └── ipyvolume_backend.py # ipyvolume implementation (optional)
│   └── interactive/
│       ├── controls.py          # UI control generation
│       ├── dash_app.py          # Dash web application (optional)
│       └── jupyter_widgets.py   # Jupyter notebook widgets
```

### Module Specifications

#### 1. Core Data Handler (`data_handler.py`)
```python
class IntensityDataHandler:
    """Unified interface for loading and preprocessing intensity data"""
    
    def __init__(self, data_source: str):
        """Initialize with NPZ file path or existing array"""
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
        """Load intensity, q_vectors, and shape"""
        
    def preprocess(self, nan_strategy: str = 'mask') -> np.ndarray:
        """Apply NaN handling and normalization"""
        
    def subsample(self, factor: int = 1) -> np.ndarray:
        """Reduce data resolution for performance"""
        
    def get_statistics(self) -> Dict[str, float]:
        """Compute min, max, mean, percentiles"""
```

#### 2. Coordinate Mapper (`coordinate_mapper.py`)
```python
class CoordinateMapper:
    """Handle transformations between coordinate systems"""
    
    def __init__(self, A_inv: np.ndarray, shape: Tuple[int, int, int]):
        """Initialize with orthogonalization matrix and grid shape"""
        
    def hkl_to_cartesian(self, hkl: np.ndarray) -> np.ndarray:
        """Convert Miller indices to Cartesian coordinates"""
        
    def get_grid_coordinates(self, sampling: Tuple) -> np.ndarray:
        """Generate full coordinate grid"""
        
    def index_to_hkl(self, i: int, j: int, k: int) -> Tuple[float, float, float]:
        """Convert array indices to Miller indices"""
```

#### 3. NaN Processor (`nan_processor.py`)
```python
class NaNProcessor:
    """Strategies for handling NaN values in intensity data"""
    
    STRATEGIES = ['mask', 'zero', 'interpolate', 'remove']
    
    @staticmethod
    def apply_strategy(data: np.ndarray, strategy: str) -> np.ndarray:
        """Apply selected NaN handling strategy"""
        
    @staticmethod
    def create_mask(data: np.ndarray) -> np.ndarray:
        """Create boolean mask for valid data points"""
        
    @staticmethod
    def interpolate_nans(data: np.ndarray, method: str = 'nearest') -> np.ndarray:
        """Fill NaNs using spatial interpolation"""
```

#### 4. Plane Calculator (`plane_calculator.py`)
```python
class PlaneCalculator:
    """Calculate cutting plane positions and intersections"""
    
    def __init__(self, volume_shape: Tuple[int, int, int]):
        """Initialize with volume dimensions"""
        
    def calculate_plane_position(self, normal: np.ndarray, distance: float) -> np.ndarray:
        """Calculate plane equation coefficients"""
        
    def get_voxel_distances(self, plane: np.ndarray) -> np.ndarray:
        """Compute signed distances from voxels to plane"""
        
    def create_cutting_sequence(self, normal: np.ndarray, n_frames: int) -> List[np.ndarray]:
        """Generate sequence of cutting planes"""
```

#### 5. Volume Slicer (`slicer.py`)
```python
class VolumeSlicer:
    """Extract slices and partial volumes from 3D data"""
    
    def __init__(self, volume: np.ndarray):
        """Initialize with 3D intensity data"""
        
    def slice_at_plane(self, plane: np.ndarray) -> np.ndarray:
        """Extract 2D slice at arbitrary plane"""
        
    def mask_above_plane(self, plane: np.ndarray) -> np.ndarray:
        """Hide voxels above cutting plane"""
        
    def interpolate_on_plane(self, plane: np.ndarray) -> np.ndarray:
        """Interpolate intensity values on plane surface"""
```

#### 6. Animation Generator (`animator.py`)
```python
class SlicingAnimator:
    """Generate animated visualizations of progressive slicing"""
    
    def __init__(self, slicer: VolumeSlicer, renderer: str = 'matplotlib'):
        """Initialize with slicer and rendering backend"""
        
    def create_frame(self, plane: np.ndarray) -> np.ndarray:
        """Render single frame showing volume below plane"""
        
    def generate_animation(self, planes: List[np.ndarray]) -> None:
        """Create full animation sequence"""
        
    def export_gif(self, output_path: str, fps: int = 10) -> None:
        """Export animation as GIF file"""
```

#### 7. Volume Renderer Interface (`renderer.py`)
```python
class VolumeRenderer(ABC):
    """Abstract base class for volume rendering backends"""
    
    @abstractmethod
    def create_volume(self, data: np.ndarray, coordinates: np.ndarray) -> Any:
        """Create volume visualization object"""
        
    @abstractmethod
    def set_colormap(self, colormap: str) -> None:
        """Apply colormap to volume"""
        
    @abstractmethod
    def set_opacity(self, opacity: Union[float, np.ndarray]) -> None:
        """Set volume opacity"""
        
    @abstractmethod
    def add_isosurface(self, level: float) -> None:
        """Add isosurface at intensity level"""
        
    @abstractmethod
    def render(self) -> Any:
        """Generate final visualization"""
```

## Implementation Plan

### Phase 1: Foundation (Week 1)
**Goal**: Establish core data handling and coordinate system infrastructure

#### Checkpoint 1.1: Core Module Setup
- [ ] Create `visualization/` package structure
- [ ] Implement `IntensityDataHandler` with existing NPZ/NPY loading
- [ ] Test with existing data files
- **Validation**: Successfully load and inspect all test datasets

#### Checkpoint 1.2: Coordinate Mapping
- [ ] Implement `CoordinateMapper` using existing grid generation logic
- [ ] Add unit tests for coordinate transformations
- [ ] Validate against existing `map_utils.py` functions
- **Validation**: Coordinate transformations match existing implementation

#### Checkpoint 1.3: NaN Processing
- [ ] Implement all NaN handling strategies
- [ ] Create performance benchmarks for each strategy
- [ ] Test with real data containing lattice point NaNs
- **Validation**: All strategies produce valid, visualizable data

### Phase 2: Slicing Animation - Multiple Approaches (Week 2)

#### Checkpoint 2.1: Plane Mathematics
- [ ] Implement `PlaneCalculator` with arbitrary normal vectors
- [ ] Add plane-voxel intersection algorithms
- [ ] Create unit tests for edge cases
- **Validation**: Correct plane equations for all orientations

#### Checkpoint 2.2: Approach A - Matplotlib Backend
- [ ] Implement matplotlib-based `VolumeSlicer`
- [ ] Create basic frame rendering with voxel masking
- [ ] Add composite visualization (volume + slice plane)
- **Validation**: Generate test frames for simple volumes

#### Checkpoint 2.3: Approach B - PyVista Backend (Optional)
- [ ] Implement PyVista-based slicer (if dependencies acceptable)
- [ ] Use native clipping plane functionality
- [ ] Compare performance with matplotlib approach
- **Validation**: Equivalent output to matplotlib version

#### Checkpoint 2.4: Animation Generation
- [ ] Implement `SlicingAnimator` with both backends
- [ ] Add smooth transitions between frames
- [ ] Optimize GIF compression settings
- **Validation**: Generate sample GIFs for all test cases

### Phase 3: Interactive Volume Rendering - Multiple Backends (Week 3)

#### Checkpoint 3.1: Plotly Volume Implementation
- [ ] Implement `PlotlyVolumeRenderer` with `go.Volume`
- [ ] Add isosurface support for multiple levels
- [ ] Implement dynamic opacity mapping
- **Validation**: Interactive 3D volume in browser

#### Checkpoint 3.2: Plotly Isosurface Alternative
- [ ] Implement multi-level isosurface visualization
- [ ] Compare performance with volume rendering
- [ ] Add smooth transitions between isosurface levels
- **Validation**: Clear visualization of intensity gradients

#### Checkpoint 3.3: ipyvolume Implementation (Jupyter-focused)
- [ ] Implement `IpyvolumeRenderer` for notebook users
- [ ] Add native widget controls
- [ ] Test in Jupyter Lab and classic notebook
- **Validation**: Smooth interaction in notebook environment

#### Checkpoint 3.4: Performance Comparison
- [ ] Benchmark all rendering backends
- [ ] Document memory usage and frame rates
- [ ] Create decision matrix for backend selection
- **Validation**: Performance data for all approaches

### Phase 4: Interactivity and Controls (Week 4)

#### Checkpoint 4.1: Control Interfaces
- [ ] Implement rotation controls (manual and programmed)
- [ ] Add dynamic range adjustment sliders
- [ ] Create opacity and colormap selectors
- **Validation**: All controls affect visualization in real-time

#### Checkpoint 4.2: Animation Features
- [ ] Add programmed rotation sequences
- [ ] Implement dynamic range animation
- [ ] Create smooth transition effects
- **Validation**: Export animated sequences

#### Checkpoint 4.3: Dash Web Application (Optional)
- [ ] Create standalone Dash app for volume exploration
- [ ] Add data upload functionality
- [ ] Implement shareable URL generation
- **Validation**: Deployable web application

#### Checkpoint 4.4: Jupyter Integration
- [ ] Create IPython widgets for all controls
- [ ] Add inline animation playback
- [ ] Implement progressive loading for large datasets
- **Validation**: Seamless notebook experience

### Phase 5: Integration and Polish (Week 5)

#### Checkpoint 5.1: API Unification
- [ ] Create high-level API for both visualization types
- [ ] Add comprehensive docstrings and type hints
- [ ] Write usage examples and tutorials
- **Validation**: Clean, intuitive API

#### Checkpoint 5.2: Performance Optimization
- [ ] Profile and optimize critical paths
- [ ] Implement data caching strategies
- [ ] Add level-of-detail (LOD) rendering
- **Validation**: Handle 1M+ voxel datasets

#### Checkpoint 5.3: Testing Suite
- [ ] Create comprehensive unit tests
- [ ] Add integration tests with real data
- [ ] Implement visual regression tests
- **Validation**: >90% code coverage

#### Checkpoint 5.4: Documentation
- [ ] Write user guide with examples
- [ ] Create API reference documentation
- [ ] Add visualization gallery
- **Validation**: Complete documentation

## Technical Decisions

### Backend Selection Criteria

| Backend | Pros | Cons | Use Case |
|---------|------|------|----------|
| Matplotlib | Universal, no new deps | Limited 3D | Slicing animation |
| Plotly | WebGL, interactive | Limited customization | Primary volume rendering |
| PyVista | Professional 3D | New dependency | Advanced slicing |
| ipyvolume | Jupyter-native | Notebook-only | Jupyter users |

### NaN Handling Strategy Selection

| Strategy | Method | Performance | Quality | Use Case |
|----------|--------|-------------|---------|----------|
| Mask | Set alpha=0 | Fast | Good | Default |
| Zero | Replace with 0 | Fast | Poor | Quick preview |
| Interpolate | Spatial filling | Slow | Best | Publication |
| Remove | Filter out | Fast | Good | Isosurfaces |

## Risk Mitigation

### Performance Risks
- **Risk**: Large datasets (>1M voxels) cause memory/rendering issues
- **Mitigation**: Implement progressive subsampling and LOD rendering

### Compatibility Risks
- **Risk**: New dependencies conflict with existing environment
- **Mitigation**: Make advanced backends optional, maintain matplotlib fallback

### Complexity Risks
- **Risk**: Too many rendering options confuse users
- **Mitigation**: Provide sensible defaults and preset configurations

## Success Criteria

### Functional Requirements
- [ ] Generate slicing animations with arbitrary cutting planes
- [ ] Render interactive 3D volumes with WebGL
- [ ] Handle NaN values without artifacts
- [ ] Support datasets up to 1M voxels

### Performance Requirements
- [ ] Animation frame generation: <1s per frame
- [ ] Interactive rendering: >10 FPS for 500k voxels
- [ ] Memory usage: <2x data size

### Quality Requirements
- [ ] Publication-quality output
- [ ] Smooth animations without flickering
- [ ] Accurate intensity representation
- [ ] Intuitive user controls

## Testing Strategy

### Unit Testing
- Test each component in isolation
- Mock data for predictable results
- Edge cases: empty data, all NaNs, single voxel

### Integration Testing
- Full pipeline tests with real data
- Cross-backend compatibility
- Coordinate system verification

### Visual Testing
- Reference image comparison
- Animation smoothness metrics
- Color accuracy validation

### Performance Testing
- Benchmark suite for all operations
- Memory profiling
- Scaling tests with varying data sizes

## Maintenance and Extension

### Future Extensions
1. VR/AR support for immersive visualization
2. GPU-accelerated slicing algorithms
3. Multi-dataset comparison views
4. Time-series animation support
5. Export to standard 3D formats (OBJ, PLY)

### Maintenance Plan
- Quarterly dependency updates
- Performance regression testing
- User feedback integration
- Documentation updates with each release