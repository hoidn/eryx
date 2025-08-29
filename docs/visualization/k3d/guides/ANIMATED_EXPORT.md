# K3D Animated HTML Export

## Overview

The K3D animated export module (`eryx.visualization.volume.k3d.k3d_animated_export`) generates standalone HTML files with embedded animations that work without requiring browser console interaction. This solves the "console hacking" problem by embedding all animation controls directly into the exported HTML.

## Key Features

- **No Console Required**: All animations work out-of-the-box in the browser
- **Interactive Controls**: Built-in UI panel with play/stop buttons and parameter sliders
- **Multiple Animation Types**: Orbit, sweep, zoom, and combined animations
- **Proper K3D API Usage**: Uses `setCamera()`, `setClippingPlanes()`, and `render()` methods
- **Auto-start Option**: Animations can begin automatically on page load
- **Preset Views**: Quick access to front, isometric, and top camera views

## Installation

The module is part of the eryx package. Ensure you have k3d installed:

```bash
pip install k3d
```

## Usage

### Basic Example

```python
from eryx.visualization.volume.k3d.k3d_animated_export import export_animated_visualization
import numpy as np

# Load your diffuse intensity data
data = np.load('diffuse_intensity.npy')
volume = data.reshape(41, 41, 41)  # Reshape if needed

# Export with orbital camera animation
export_animated_visualization(
    volume,
    'animated_diffuse.html',
    animation_type='orbit',
    title='Diffuse Scattering - Orbital View'
)
```

### Animation Types

#### 1. Orbit Animation (`animation_type='orbit'`)
Rotates the camera around the volume at a fixed radius.

**Controls:**
- Radius: Distance from center (2-10)
- Height: Camera elevation (-5 to 5)
- Speed: Animation speed (0.005-0.1)

#### 2. Sweep Animation (`animation_type='sweep'`)
Sweeps a clipping plane through the volume.

**Controls:**
- Axis: X, Y, or Z axis
- Speed: Sweep speed (0.005-0.1)

#### 3. Zoom Animation (`animation_type='zoom'`)
Smoothly zooms in and out.

**Controls:**
- Min Distance: Closest zoom (1-5)
- Max Distance: Farthest zoom (5-20)
- Speed: Zoom speed (0.005-0.1)

#### 4. Combined Animation (`animation_type='combined'`)
Combines orbital camera movement with clipping plane sweep.

### Advanced Options

```python
export_animated_visualization(
    volume_data,
    'output.html',
    animation_type='orbit',
    bounds=[-2, 2, -2, 2, -2, 2],  # Volume bounds
    color_range_percentile=90,      # Color mapping percentile
    alpha_coef=20.0,                # Transparency coefficient
    colormap='Jet',                  # Color map name
    animation_config={
        'auto_start': True           # Start animation on load
    }
)
```

## HTML Structure

The exported HTML files contain:

1. **K3D Visualization**: The main 3D volume rendering
2. **Control Panel**: Interactive controls (top-right corner)
3. **Embedded JavaScript**: Animation logic using K3D API
4. **CSS Styling**: Professional-looking control panel

## Browser Compatibility

- ✅ Chrome/Chromium (recommended)
- ✅ Firefox
- ✅ Edge
- ⚠️ Safari (may have limited WebGL features)

## Control Panel Features

Each exported HTML file includes:

- **▶️ Start**: Begin animation
- **⏹️ Stop**: Stop animation
- **🔄 Reset**: Reset camera and clipping
- **Parameter Sliders**: Real-time adjustment
- **Preset Views**: Quick camera positions
- **Status Display**: Current animation state

## Technical Details

### K3D API Methods Used

The module properly uses K3D's official API:

```javascript
// Camera control
k3d.setCamera([x, y, z, tx, ty, tz, ux, uy, uz]);

// Clipping planes
k3d.setClippingPlanes([[nx, ny, nz, d]]);

// Trigger rendering
k3d.render();

// Reset view
k3d.resetCamera();
```

### Promise-based K3D Instance

The code handles K3D's Promise-based architecture:

```javascript
// Wait for K3D instance
const k3d = await window.K3DInstance;

// Now use K3D methods
k3d.setCamera(...);
```

## Example Files Generated

Running the test script creates:

- `k3d_animated_orbit.html` - Orbital camera movement
- `k3d_animated_sweep.html` - Clipping plane sweep
- `k3d_animated_zoom.html` - Zoom in/out effect
- `k3d_animated_combined.html` - Combined animations
- `k3d_animation_showcase.html` - Master showcase

## Troubleshooting

### Animation Not Starting
- Check browser console for errors
- Ensure WebGL is enabled
- Try a different browser (Chrome recommended)

### Slow Performance
- Reduce `alpha_coef` value (try 10-15)
- Decrease volume resolution if possible
- Close other browser tabs

### Controls Not Responding
- Wait for page to fully load
- Check that K3D instance initialized (see console)
- Refresh the page

## Integration with Eryx Pipeline

```python
from eryx import OnePhonon_torch
from eryx.visualization.volume.k3d.k3d_animated_export import export_animated_visualization

# Generate diffuse scattering
model = OnePhonon_torch(pdb_path, device='cuda')
intensity = model.apply_disorder()

# Reshape to 3D volume
volume = intensity.reshape(41, 41, 41)

# Export with animation
export_animated_visualization(
    volume,
    'diffuse_scattering_animated.html',
    animation_type='combined',
    title='Protein Diffuse Scattering'
)
```

## Benefits Over Console Interaction

1. **User-Friendly**: No technical knowledge required
2. **Reproducible**: Same animation every time
3. **Shareable**: Send HTML file to collaborators
4. **Professional**: Polished UI for presentations
5. **Customizable**: Adjust parameters on-the-fly

## Future Enhancements

- [ ] Add more animation presets
- [ ] Support for multiple volumes
- [ ] Timeline-based animation sequences
- [ ] Export to video (MP4)
- [ ] VR/AR support via WebXR

## Related Documentation

- [K3D Python Documentation](https://k3d-jupyter.org/)
- [Eryx Visualization Guide](./eryx/visualization/README.md)
- [K3D Animation History](./history/2025-08-29_k3d_animation_debugging.md)