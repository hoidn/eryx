# K3D Camera Animation Guide

Complete guide for using camera animations with k3d_spherical_clipping visualization.

## Overview

The camera animation system provides cinematic visualization capabilities for diffuse scattering data, enabling smooth orbital rotations, zoom effects, and synchronized sphere-camera movements for scientific presentation.

## Quick Start

```python
from eryx.visualization.volume.k3d.k3d_spherical_clipping import SphericalClippingController

# Initialize controller with your data
controller = SphericalClippingController()
controller.plot.display()

# Add orbital animation
controller.add_orbital_camera(duration=6.0, elevation=45)

# Or zoom animation
controller.add_zoom_animation(duration=4.0)

# Or synchronized zoom-reveal
controller.create_zoom_reveal_animation()
```

## Available Animations

### 1. Orbital Camera
Smooth circular rotation around the data at a fixed elevation.

```python
controller.add_orbital_camera(
    duration=6.0,        # Animation duration in seconds
    radius_factor=2.5,   # Distance from center (1.0 = optimal, higher = farther)
    elevation=45,        # Vertical angle in degrees (0=horizon, 90=top)
    num_frames=60        # Number of keyframes (higher = smoother)
)
```

**Use cases:**
- General data overview
- Showcasing symmetry
- Creating presentation loops

### 2. Zoom Animation
Smooth zoom in or out with optional rotation.

```python
controller.add_zoom_animation(
    start_distance=None,  # Starting distance (None = 3x optimal)
    end_distance=None,    # Ending distance (None = 1x optimal)
    duration=4.0,         # Animation duration
    elevation=30,         # Camera elevation
    azimuth=45,          # Starting azimuth angle
    num_frames=50        # Number of keyframes
)
```

**Features:**
- Ease-in-out interpolation for smooth motion
- Optional slight rotation during zoom
- Maintains focus on sphere center

### 3. Preset Views
Instant or smooth transition to standard viewpoints.

```python
# Available presets: 'front', 'back', 'left', 'right', 'top', 'bottom', 'isometric', 'optimal'
controller.set_preset_view(
    view_name='isometric',     # Preset name
    transition_duration=1.0     # Smooth transition time (0 = instant)
)
```

**Preset angles:**
- `front`: (0°, 0°) - Looking along +Y axis
- `right`: (90°, 0°) - Looking along -X axis
- `top`: (0°, 90°) - Looking down Z axis
- `isometric`: (45°, 35.264°) - Classic 3D view

### 4. Zoom-Reveal Animation
Synchronized camera zoom with sphere shrinking to reveal internal structure.

```python
controller.create_zoom_reveal_animation(
    duration=5.0,              # Total animation time
    final_radius_factor=0.2    # Final sphere size (0.2 = 20% of original)
)
```

**Features:**
- Camera zooms in as sphere shrinks
- Reveals internal structure progressively
- Uses threading for parallel animation

### 5. Octant Inspection Tour
Systematic camera tour visiting all 8 octant viewpoints.

```python
controller.create_octant_inspection_animation(
    duration=8.0,         # Total tour duration
    pause_duration=0.5    # Pause at each octant
)
```

**Tour path:**
- Visits all 8 octant corners
- Optimal viewing angle for each region
- Pauses for inspection at each position

### 6. Anisotropy Showcase
Multi-elevation orbital animation for highlighting directional features.

```python
controller.add_anisotropy_showcase_animation(
    duration=10.0    # Total animation duration
)
```

**Features:**
- Orbits at multiple elevation angles (30°, 60°, 90°, 60°, 30°)
- Variable radius for visual interest
- 100 total keyframes for comprehensive coverage

## Interactive Controls (Jupyter)

When using in Jupyter notebooks, interactive controls are available:

```python
# The notebook includes widgets for:
# - Animation selection (Orbit, Zoom, Reveal, Tour, etc.)
# - Duration control (2-20 seconds)
# - Elevation angle (0-90 degrees)
# - Distance factor (1-5x optimal)
# - Stop/Reset buttons
# - Preset view dropdown
```

## Camera State Management

### Store and Reset Camera
```python
# Store current camera before animation
controller.store_initial_camera()

# Start animation
controller.add_orbital_camera()

# Later, reset to stored position
controller.reset_camera()
```

### Check Animation State
```python
if controller.is_animating:
    print(f"Currently running: {controller.animation_type}")
    controller.plot.stop_auto_play()
```

## Coordinate System

### Understanding Coordinates
The system uses two coordinate spaces:

1. **Voxel space**: Integer indices (0 to h-1, 0 to k-1, 0 to l-1)
2. **World space**: Physical coordinates matching volume bounds

### Conversion Methods
```python
# Convert voxel to world coordinates
world_pos = controller.voxel_to_world([20, 20, 20])

# Get sphere center in world space
world_center = controller.get_world_sphere_center()

# Get optimal camera distance
distance = controller.get_optimal_camera_distance(
    fov_degrees=30,     # Field of view
    scale_factor=2.5    # Distance multiplier
)
```

## Advanced Usage

### Custom Camera Paths
```python
# Create custom camera animation
import math

frames = []
for i in range(100):
    t = 10.0 * i / 99  # 10 second animation
    
    # Custom path (e.g., spiral)
    theta = 4 * math.pi * i / 99
    radius = 50 + 20 * math.sin(theta / 2)
    z = 20 + 10 * math.cos(theta)
    
    camera = [
        radius * math.cos(theta),  # X position
        radius * math.sin(theta),  # Y position
        z,                         # Z position
        0, 0, 0,                   # Target (origin)
        0, 0, 1                    # Up vector
    ]
    frames.append([t, camera])

controller.plot.camera_animation = frames
controller.plot.start_auto_play()
```

### Synchronized Multi-Property Animation
```python
import threading
import time

def animate_properties():
    """Animate multiple properties simultaneously."""
    steps = 50
    duration = 5.0
    
    for i in range(steps):
        progress = i / (steps - 1)
        
        # Update sphere radius
        controller.sphere_radius = 20 * (1 - 0.5 * progress)
        
        # Update log scale
        controller.log_dynamic_range = 10 + 90 * progress
        
        # Update visualization
        controller.update_clipping()
        time.sleep(duration / steps)

# Start camera animation
controller.add_orbital_camera(duration=5.0)

# Start property animation in parallel
threading.Thread(target=animate_properties).start()
```

## Performance Tips

1. **Frame count**: 60 frames is usually sufficient for smooth animation
2. **Duration**: 4-6 seconds provides good pacing for most animations
3. **Threading**: Use for synchronized animations (camera + sphere)
4. **Memory**: Camera animations have minimal memory overhead

## Troubleshooting

### Animation not starting
```python
# Ensure plot is displayed
controller.plot.display()

# Check if already animating
if controller.is_animating:
    controller.plot.stop_auto_play()

# Start fresh animation
controller.add_orbital_camera()
```

### Camera jumps unexpectedly
```python
# Store initial camera first
controller.store_initial_camera()

# Or set to known preset
controller.set_preset_view('isometric', transition_duration=0)
```

### Sphere not updating during synchronized animation
```python
# Ensure method is 'masking' for real-time updates
controller.method = 'masking'
controller.update_clipping()
```

## Examples

### Example 1: Presentation Sequence
```python
# Overview orbit
controller.add_orbital_camera(duration=4.0, elevation=45)
time.sleep(4.5)

# Zoom to feature
controller.add_zoom_animation(duration=3.0)
time.sleep(3.5)

# Reveal internal structure
controller.create_zoom_reveal_animation(duration=5.0)
```

### Example 2: Comprehensive Analysis
```python
# Start with isometric view
controller.set_preset_view('isometric')

# Anisotropy analysis
controller.add_anisotropy_showcase_animation(duration=10.0)
time.sleep(10.5)

# Octant inspection
controller.create_octant_inspection_animation(duration=8.0)
```

### Example 3: Custom Scientific Animation
```python
# Synchronized sphere pulsing with rotation
import threading

def pulse_sphere():
    for i in range(100):
        phase = 2 * math.pi * i / 20
        controller.sphere_radius = 15 + 5 * math.sin(phase)
        controller.update_clipping()
        time.sleep(0.1)

controller.add_orbital_camera(duration=10.0, elevation=30)
threading.Thread(target=pulse_sphere).start()
```

## Integration with Existing Features

The camera animation system works seamlessly with:
- **Spherical clipping**: Animations respect current sphere settings
- **Octant exclusion**: Can be synchronized with camera position
- **Log scaling**: Intensity scaling independent of camera
- **Data masking**: Real-time updates during animation

## API Reference

### Core Animation Methods
- `add_orbital_camera()` - Circular orbit animation
- `add_zoom_animation()` - Zoom in/out effect
- `set_preset_view()` - Standard viewpoints
- `create_zoom_reveal_animation()` - Synchronized zoom-sphere
- `create_octant_inspection_animation()` - Tour all octants
- `add_anisotropy_showcase_animation()` - Multi-elevation orbit

### Camera State Methods
- `store_initial_camera()` - Save current camera
- `reset_camera()` - Restore saved camera
- `get_default_camera()` - Calculate default position

### Coordinate Methods
- `voxel_to_world()` - Convert voxel to world coordinates
- `get_world_sphere_center()` - Sphere center in world space
- `get_world_bounds()` - Volume bounds in world space
- `get_optimal_camera_distance()` - Calculate ideal camera distance

## Next Steps

1. Experiment with different animation combinations
2. Customize parameters for your data
3. Create presentation sequences
4. Explore synchronized animations
5. Build custom camera paths for specific analysis needs