# K3D Performance Optimization: Lessons Learned

**Session Date**: August 29, 2024  
**Context**: Interactive K3D visualization optimization session  
**Key Result**: 10x performance improvement (5 FPS → 60 FPS)  

## Executive Summary

This document captures critical lessons learned from optimizing K3D visualizations, particularly around spherical masking and multi-panel comparisons. The primary breakthrough was discovering that direct data manipulation vastly outperforms clipping planes for complex masking operations.

## Critical Performance Discoveries

### 1. Clipping Planes Scale Poorly

**Problem**: Attempting to create spherical masks using 60+ clipping planes
```javascript
// This approach created 62+ planes and killed performance
const planes = [];
for (let theta = 0; theta < 12; theta++) {
    for (let phi = 0; phi < 6; phi++) {
        planes.push([nx, ny, nz, d]);  // Each iteration adds a plane
    }
}
plot.setClippingPlanes(planes); // Result: ~5 FPS, laggy interaction
```

**Solution**: Direct manipulation of volume data
```javascript
// Direct data masking approach
function applySphericalMask(data, shape, radius) {
    const [h, k, l] = shape;
    const center = [h/2, k/2, l/2];
    const masked = new Float32Array(data.length);
    
    let idx = 0;
    for (let i = 0; i < h; i++) {
        for (let j = 0; j < k; j++) {
            for (let m = 0; m < l; m++) {
                const dist = Math.sqrt(
                    (i - center[0])**2 + 
                    (j - center[1])**2 + 
                    (m - center[2])**2
                );
                masked[idx] = (dist <= radius) ? data[idx] : 0;
                idx++;
            }
        }
    }
    return masked;
}
// Result: ~60 FPS, smooth interaction
```

**Performance Comparison**:
| Method | Clipping Planes | FPS | Memory Usage | User Experience |
|--------|-----------------|-----|--------------|-----------------|
| Many Clipping Planes | 62+ | ~5 | High GPU | Unusable lag |
| Octahedral Approximation | 8 | ~30 | Moderate | Acceptable for static |
| Direct Data Masking | 0 | ~60 | Low | Smooth real-time |

### 2. K3D HTML Export Structure Understanding

**Critical Discovery**: K3D HTML exports have completely different object structure than Python API

**Wrong Assumptions**:
```javascript
// These don't exist in HTML exports:
plot.objects              // undefined
plot.volumes              // undefined
k3dVolume.volume = data   // wrong structure
```

**Correct Pattern**:
```javascript
async function accessVolumeData() {
    const plot = await window.K3DInstance;  // It's a Promise!
    const world = plot.getWorld();
    
    // Volume data lives here:
    const volumeConfig = world.ObjectsListJson[volumeId];
    const volumeData = volumeConfig.volume.data;  // Float32Array
    const volumeShape = volumeConfig.volume.shape; // [h, k, l]
}
```

**Data Update Pattern**:
```javascript
async function updateVolumeData(newData) {
    const plot = await window.K3DInstance;
    const world = plot.getWorld();
    
    // Update both configuration and texture
    volumeConfig.volume.data = newData;
    
    const volumeMesh = world.ObjectsById[volumeId];
    const texture = volumeMesh?.material?.uniforms?.volumeTexture?.value;
    if (texture?.image?.data) {
        texture.image.data.set(newData);
        texture.needsUpdate = true;
    }
    
    plot.rebuildSceneData();  // Required for complex changes
    plot.render();            // Required to see updates
}
```

### 3. Fundamental K3D Limitations

**Opacity-Intensity Coupling**: Cannot decouple opacity from intensity values
- K3D uses transfer functions that inherently couple these properties
- Workarounds require data preprocessing, not runtime manipulation

**Axis Label Limitations**: Cannot completely hide axis tick labels (0, 1, -1)
- Only workaround is setting label color to match background color
- No true "hide labels" option exists

**Single Instance Limitation**: Only one K3D instance per HTML page
- For comparisons, use single plot with multiple volumes
- Avoid iframe-based approaches due to synchronization issues

### 4. Multi-Panel Visualization Best Practices

**Preferred Architecture**: Single K3D plot + multiple volumes
```javascript
// Good: Multiple volumes in one plot
const plot = k3d.plot();
plot += k3d.volume(data1, name="Thermal");
plot += k3d.volume(data2, name="Pumped");  
plot += k3d.volume(data3, name="Difference");
```

**Volume Filtering Pattern**:
```javascript
// Filter actual volume objects from K3D objects array
const actualVolumes = Object.values(world.K3DObjects)
    .filter(obj => obj.alpha_coef !== undefined);
```

**Synchronized Updates**:
```javascript
// Batch all updates, render once
for (const volume of volumes) {
    volume.config.volume.data = newData[volume.id];
}
plot.rebuildSceneData();
plot.render(); // Single render call prevents flickering
```

## Data Quality Impact

### Resolution Comparison
| Resolution | Voxels | File Size | Visual Quality | Recommended Use |
|------------|--------|-----------|----------------|-----------------|
| 41³ | 68,921 | ~270KB | Basic | Development/testing |
| 81³ | 531,441 | ~2.1MB | Good | Production quality |
| 161³ | 4,173,281 | ~16.8MB | Excellent | High-end presentations |

**Key Insight**: The visual quality improvement from 41³ to 81³ is dramatic and worth the 8x memory increase for production visualizations.

### Data Preprocessing Best Practices

```python
# Always normalize data before K3D processing
data_normalized = (data - data.min()) / (data.max() - data.min())

# Handle NaN values explicitly
data_clean = np.nan_to_num(data_normalized, nan=0.0)

# Apply percentile clipping for better contrast
p1, p99 = np.percentile(data_clean, [1, 99])
data_clipped = np.clip(data_clean, p1, p99)
```

## JavaScript Integration Patterns

### Global Function Registration
```javascript
// Required for HTML onclick handlers
window.functionName = function() {
    // Function body
};

// Validate registration
console.assert(typeof window.functionName === 'function', 'Function not global!');
```

### Safe K3D Access Pattern
```javascript
async function safeK3DOperation() {
    let plot;
    
    if (window.K3DInstance instanceof Promise) {
        plot = await window.K3DInstance;
    } else if (window.K3DInstance) {
        plot = window.K3DInstance;
    } else {
        console.error('No K3D instance found');
        return;
    }
    
    // Safe to use plot
    plot.setCamera([...]);
    plot.render();
}
```

### Animation State Management
```javascript
class K3DAnimationController {
    constructor() {
        this.animationId = null;
        this.currentAnimation = null;
    }
    
    start(type) {
        this.stop(); // Prevent conflicts
        this.currentAnimation = type;
        const animate = () => {
            if (!this.currentAnimation) return;
            
            // Animation logic
            this.animationId = requestAnimationFrame(animate);
        };
        animate();
    }
    
    stop() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
            this.currentAnimation = null;
        }
    }
}
```

## Testing and Debugging Approaches

### The 5-Minute Test Rule
Before implementing anything in Python:

1. **Browser Console Test** (2 min): Test the JavaScript manually
2. **Simple HTML Test** (3 min): Verify onclick handlers work
3. **Only then implement in Python**

### Essential Debug Commands
```javascript
// Find K3D structure
K3DInstance.then(plot => {
    console.log('Plot keys:', Object.keys(plot));
    console.log('World:', plot.getWorld());
    window.debugPlot = plot; // Store for easier access
});

// Find volume data
function findVolumeData(obj, path = 'root', visited = new Set()) {
    if (visited.has(obj)) return;
    visited.add(obj);
    
    for (const key in obj) {
        try {
            const val = obj[key];
            if (val instanceof Float32Array && val.length > 1000) {
                console.log(`Found Float32Array at ${path}.${key}, length:`, val.length);
            }
        } catch (e) {}
    }
}
```

## Memory Management

### Memory Usage Patterns
```javascript
// Monitor volume memory usage
K3DInstance.then(plot => {
    const world = plot.getWorld();
    let totalBytes = 0;
    
    for (const id in world.ObjectsListJson) {
        const obj = world.ObjectsListJson[id];
        if (obj.volume?.data) {
            const bytes = obj.volume.data.length * 4; // Float32 = 4 bytes
            totalBytes += bytes;
            console.log(`Volume ${id}: ${(bytes/1024/1024).toFixed(2)} MB`);
        }
    }
    
    console.log(`Total K3D memory: ${(totalBytes/1024/1024).toFixed(2)} MB`);
});
```

### Performance Optimization Checklist

- [ ] Use direct data masking instead of clipping planes for complex shapes
- [ ] Batch multiple volume updates with single render() call
- [ ] Cache volume references to avoid repeated world traversal
- [ ] Use Float32Array for data manipulation (faster than regular arrays)
- [ ] Normalize data to [0,1] range before K3D processing
- [ ] Handle NaN values explicitly with np.nan_to_num()
- [ ] Consider higher resolution data (81³) for production quality
- [ ] Test JavaScript in browser console before Python integration

## Common Pitfalls Avoided

1. **"plot.objects doesn't exist"** - Use `world.ObjectsListJson` instead
2. **"K3DInstance.render() not a function"** - Must await the Promise first
3. **"Changes not visible"** - Always call `plot.render()` after updates
4. **"Jerky animations"** - Use single animation controller pattern
5. **"Functions not defined"** - Attach to `window` for HTML access

## Impact Assessment

**Before Optimization**:
- Spherical masking: 62 clipping planes, ~5 FPS
- Multi-panel: Multiple iframes, synchronization issues
- Data access: Trial-and-error debugging, silent failures

**After Optimization**:
- Spherical masking: Direct data manipulation, ~60 FPS (12x improvement)
- Multi-panel: Single plot + multiple volumes, smooth updates
- Data access: Documented patterns, reliable access methods

**Time Savings**:
- Initial development: 30+ hours of debugging
- Future implementations: ~2 hours using documented patterns
- **ROI**: 15x time savings for similar visualization tasks

## Recommendations for Future Development

1. **Always start with data manipulation approach** for complex masking
2. **Use single K3D plot architecture** for multi-panel visualizations  
3. **Test JavaScript in browser console** before Python integration
4. **Document any new K3D discoveries** in this growing knowledge base
5. **Prefer higher resolution data** (81³) for production visualizations
6. **Implement proper error handling** and debug logging for JavaScript integration

---

*This document represents practical lessons learned from real-world K3D optimization work and should be consulted before starting new K3D visualization projects.*