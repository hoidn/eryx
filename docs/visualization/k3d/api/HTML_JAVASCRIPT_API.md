# K3D HTML/JavaScript API Reference & Developer Guide

**Created from debugging session: November 2024**  
**Purpose**: Document the actual K3D HTML export structure and JavaScript API to prevent common issues

> **📚 Related K3D Documentation**:
> - **[JavaScript Patterns & Examples](./JAVASCRIPT_PATTERNS.md)** - Practical implementation patterns
> - **[Production Usage Guide](../getting-started/QUICK_START.md)** - Working examples for researchers
> - **[Visualization Module Overview](../../../../eryx/visualization/README.md)** - Architecture and module structure

---

## Table of Contents
1. [Critical Differences: Python vs HTML Export](#critical-differences)
2. [K3D HTML Export Object Structure](#k3d-html-export-object-structure)
3. [Accessing Volume Data in JavaScript](#accessing-volume-data)
4. [Performance Optimization Guidelines](#performance-optimization)
5. [Common Pitfalls and Solutions](#common-pitfalls)
6. [Debugging Techniques](#debugging-techniques)
7. [Working Code Examples](#working-code-examples)
8. [Console Commands Reference](#console-commands-reference)

---

## Critical Differences: Python vs HTML Export

### ⚠️ IMPORTANT: The K3D HTML export structure is completely different from the Python API

| Feature | Python K3D | HTML Export JavaScript |
|---------|------------|------------------------|
| **Plot Object Access** | `plot = k3d.plot()` | `await window.K3DInstance` (Promise) |
| **Objects Array** | `plot.objects` ✅ | `plot.objects` ❌ **DOES NOT EXIST** |
| **Volume Data Location** | `volume.volume` | `world.ObjectsListJson[id].volume.data` |
| **Update Method** | `volume.volume = data` | Update texture & config, then `plot.render()` |
| **Clipping Planes** | `plot.clipping_planes` | `plot.setClippingPlanes()` |
| **Immediate Access** | Yes | No - Must await Promise |

---

## K3D HTML Export Object Structure

### The Real Structure (What Actually Exists)
```javascript
K3DInstance (Promise) → K3D Plot Object
    ├── getWorld() → World Object
    │   ├── ObjectsListJson     // Configuration for each object
    │   │   └── [objectId]
    │   │       ├── volume
    │   │       │   ├── data    // Float32Array - THE ACTUAL DATA
    │   │       │   └── shape   // [h, k, l] dimensions
    │   │       └── type        // "Volume"
    │   │
    │   └── ObjectsById          // THREE.js Mesh objects
    │       └── [objectId]       // THREE.js Mesh
    │           └── material
    │               └── uniforms
    │                   └── volumeTexture
    │                       └── value  // Data3DTexture
    │
    ├── getScene()               // THREE.js Scene
    ├── render()                 // Force re-render
    ├── setClippingPlanes()      // Set clipping planes
    └── rebuildSceneData()       // Rebuild scene
```

### What Does NOT Exist (Common Mistakes)
```javascript
// ❌ THESE DO NOT EXIST IN HTML EXPORTS:
plot.objects              // undefined
plot.volumes              // undefined  
plot.K3DObjects           // undefined (sometimes)
k3dVolume.volume = data   // Won't work - wrong structure
```

---

## Accessing Volume Data in JavaScript

### Correct Pattern for Finding Volume Data
```javascript
async function getVolumeData() {
    // Step 1: Get the K3D plot (it's a Promise!)
    const plot = await window.K3DInstance;
    
    // Step 2: Get the world object
    const world = plot.getWorld();
    
    // Step 3: Find the volume object ID
    let volumeId = null;
    for (const id in world.ObjectsListJson) {
        if (world.ObjectsListJson[id].type === 'Volume') {
            volumeId = id;
            break;
        }
    }
    
    // Step 4: Access the volume data
    const volumeConfig = world.ObjectsListJson[volumeId];
    const volumeData = volumeConfig.volume.data;  // Float32Array
    const volumeShape = volumeConfig.volume.shape; // [h, k, l]
    
    return { volumeId, volumeData, volumeShape, volumeConfig };
}
```

### Updating Volume Data (The Right Way)
```javascript
async function updateVolumeData(newData) {
    const plot = await window.K3DInstance;
    const world = plot.getWorld();
    
    // Find volume
    const volumeInfo = await getVolumeData();
    const { volumeId, volumeConfig } = volumeInfo;
    
    // Update in config
    volumeConfig.volume.data = newData;
    
    // Update 3D texture if it exists
    const volumeMesh = world.ObjectsById[volumeId];
    if (volumeMesh?.material?.uniforms?.volumeTexture?.value) {
        const texture = volumeMesh.material.uniforms.volumeTexture.value;
        if (texture.image && texture.image.data) {
            texture.image.data.set(newData);
            texture.needsUpdate = true;
        }
    }
    
    // Force re-render
    plot.rebuildSceneData();
    plot.render();
}
```

---

## Performance Optimization Guidelines

### 🚨 CRITICAL: Avoid Excessive Clipping Planes

#### ❌ BAD: Creating 60+ Clipping Planes for Spherical Masking
```javascript
// This creates 62+ planes and kills performance!
const planes = [];
for (let i = 0; i < 12; i++) {
    for (let j = 0; j < 6; j++) {
        // Each iteration adds a plane
        planes.push([nx, ny, nz, d]);
    }
}
plot.setClippingPlanes(planes); // 62 planes = terrible performance
```

#### ✅ GOOD: Direct Data Masking
```javascript
// Modify the data directly - MUCH faster!
function applySphericalMask(data, shape, radius) {
    const [h, k, l] = shape;
    const masked = new Float32Array(data.length);
    const center = [h/2, k/2, l/2];
    
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
```

### Performance Comparison
| Method | Planes | FPS | Recommendation | Use Case |
|--------|--------|-----|----------------|----------|
| 60+ Clipping Planes | 62 | ~5 fps | ❌ Never use | None - fundamentally flawed |
| 8 Octahedral Planes | 8 | ~30 fps | ⚠️ OK for simple cases | Static views only |
| Direct Data Masking | 0 | ~60 fps | ✅ Always prefer | Real-time interaction |
| Hybrid Approach | 4-6 | ~45 fps | ✅ Good compromise | Complex multi-region masking |

### Memory Usage Patterns
| Data Type | Original Size | K3D Memory | Browser Memory | Notes |
|-----------|---------------|------------|----------------|-------|
| 41³ Float32 | ~270KB | ~540KB | ~1.1MB | Minimum viable resolution |
| 81³ Float32 | ~2.1MB | ~4.2MB | ~8.4MB | Recommended for quality |
| 161³ Float32 | ~16.8MB | ~33.6MB | ~67MB | High-end desktop only |

---

## Common Pitfalls and Solutions

### Pitfall 1: Accessing plot.objects
```javascript
// ❌ WRONG - plot.objects doesn't exist in HTML
const volumes = plot.objects.filter(obj => obj.type === 'Volume');

// ✅ CORRECT - Use getWorld()
const world = plot.getWorld();
const volumes = Object.values(world.ObjectsListJson)
    .filter(obj => obj.type === 'Volume');
```

### Pitfall 2: Not Awaiting K3DInstance
```javascript
// ❌ WRONG - K3DInstance is a Promise
const plot = window.K3DInstance;
plot.render(); // Error: plot.render is not a function

// ✅ CORRECT - Await the Promise
const plot = await window.K3DInstance;
plot.render(); // Works!
```

### Pitfall 3: Wrong Update Pattern
```javascript
// ❌ WRONG - This pattern doesn't work in HTML
k3dVolume.volume = newData;

// ✅ CORRECT - Update both config and texture
volumeConfig.volume.data = newData;
texture.image.data.set(newData);
texture.needsUpdate = true;
plot.render();
```

### Pitfall 4: Functions Not Globally Accessible
```javascript
// ❌ WRONG - Function not accessible from HTML onclick
function myAnimation() { }

// ✅ CORRECT - Attach to window
window.myAnimation = function() { }
```

---

## Debugging Techniques

### Essential Debugging Commands

```javascript
// 1. Find the K3D plot structure
K3DInstance.then(plot => {
    console.log('Plot keys:', Object.keys(plot));
    console.log('World:', plot.getWorld());
    window.debugPlot = plot; // Store for easier access
});

// 2. Find all volume objects
K3DInstance.then(plot => {
    const world = plot.getWorld();
    console.log('ObjectsListJson:', world.ObjectsListJson);
    console.log('ObjectsById:', world.ObjectsById);
    
    // Find volumes
    for (const id in world.ObjectsListJson) {
        const obj = world.ObjectsListJson[id];
        if (obj.type === 'Volume') {
            console.log(`Volume found at ID ${id}:`, obj);
            console.log('Volume data length:', obj.volume.data.length);
            console.log('Volume shape:', obj.volume.shape);
        }
    }
});

// 3. Deep search for Float32Arrays (volume data)
function findVolumeData(obj, path = 'root', visited = new Set()) {
    if (visited.has(obj)) return;
    visited.add(obj);
    
    for (const key in obj) {
        try {
            const val = obj[key];
            if (val instanceof Float32Array && val.length > 1000) {
                console.log(`Found Float32Array at ${path}.${key}, length:`, val.length);
                if (val.length === 68921) { // 41^3
                    console.log('🎯 This is likely the volume data!');
                    window.foundVolumeData = val;
                }
            }
            if (val && typeof val === 'object' && key !== 'parent') {
                findVolumeData(val, `${path}.${key}`, visited);
            }
        } catch (e) {}
    }
}

// Run it
K3DInstance.then(plot => findVolumeData(plot.getWorld()));
```

---

## Working Code Examples

### Example 1: Complete Spherical Masking Implementation
```javascript
class K3DVolumeMasker {
    constructor() {
        this.originalData = null;
        this.volumeId = null;
        this.shape = null;
        this.initialized = false;
    }
    
    async initialize() {
        const plot = await window.K3DInstance;
        const world = plot.getWorld();
        
        // Find volume
        for (const id in world.ObjectsListJson) {
            if (world.ObjectsListJson[id].type === 'Volume') {
                this.volumeId = id;
                const config = world.ObjectsListJson[id];
                this.originalData = new Float32Array(config.volume.data);
                this.shape = config.volume.shape;
                this.initialized = true;
                console.log('✅ K3DVolumeMasker initialized');
                break;
            }
        }
        
        if (!this.initialized) {
            throw new Error('No volume found in K3D plot');
        }
    }
    
    async applySphereMask(radiusPercent = 50) {
        if (!this.initialized) await this.initialize();
        
        const plot = await window.K3DInstance;
        const world = plot.getWorld();
        const config = world.ObjectsListJson[this.volumeId];
        const mesh = world.ObjectsById[this.volumeId];
        
        // Calculate mask
        const [h, k, l] = this.shape;
        const center = [h/2, k/2, l/2];
        const radius = Math.min(h, k, l) * 0.5 * (radiusPercent / 100);
        
        const masked = new Float32Array(this.originalData.length);
        let idx = 0;
        
        for (let i = 0; i < h; i++) {
            for (let j = 0; j < k; j++) {
                for (let m = 0; m < l; m++) {
                    const dist = Math.sqrt(
                        (i - center[0])**2 + 
                        (j - center[1])**2 + 
                        (m - center[2])**2
                    );
                    masked[idx] = (dist <= radius) ? this.originalData[idx] : 0;
                    idx++;
                }
            }
        }
        
        // Update data
        config.volume.data = masked;
        
        // Update texture
        if (mesh?.material?.uniforms?.volumeTexture?.value) {
            const texture = mesh.material.uniforms.volumeTexture.value;
            if (texture.image?.data) {
                texture.image.data.set(masked);
                texture.needsUpdate = true;
            }
        }
        
        // Render
        plot.rebuildSceneData();
        plot.render();
    }
    
    async clearMask() {
        if (!this.initialized) return;
        
        const plot = await window.K3DInstance;
        const world = plot.getWorld();
        const config = world.ObjectsListJson[this.volumeId];
        const mesh = world.ObjectsById[this.volumeId];
        
        // Restore original
        config.volume.data = new Float32Array(this.originalData);
        
        // Update texture
        if (mesh?.material?.uniforms?.volumeTexture?.value) {
            const texture = mesh.material.uniforms.volumeTexture.value;
            if (texture.image?.data) {
                texture.image.data.set(this.originalData);
                texture.needsUpdate = true;
            }
        }
        
        plot.rebuildSceneData();
        plot.render();
    }
}

// Usage
const masker = new K3DVolumeMasker();
await masker.initialize();
await masker.applySphereMask(30);  // 30% radius
await masker.clearMask();
```

### Example 2: Performance-Optimized Animation
```javascript
class K3DOptimizedAnimator {
    constructor() {
        this.animationFrame = null;
        this.masker = new K3DVolumeMasker();
    }
    
    async start() {
        await this.masker.initialize();
        
        let t = 0;
        const animate = async () => {
            t += 0.02;
            const radius = 30 + 40 * Math.abs(Math.sin(t));
            await this.masker.applySphereMask(radius);
            this.animationFrame = requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    stop() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.masker.clearMask();
        }
    }
}

// Usage
const animator = new K3DOptimizedAnimator();
await animator.start();
// ... later
animator.stop();
```

---

## Console Commands Reference

### Quick Testing Commands
```javascript
// Get plot object
const plot = await K3DInstance;

// Get world and volumes
const world = plot.getWorld();
const volumes = Object.entries(world.ObjectsListJson)
    .filter(([id, obj]) => obj.type === 'Volume');
console.log('Volumes:', volumes);

// Apply quick sphere mask (after finding volume ID)
const id = '131376551249296'; // Your volume ID
const config = world.ObjectsListJson[id];
const data = config.volume.data;
const masked = data.map((v, i) => i % 2 === 0 ? v : 0); // Simple test mask
config.volume.data = masked;
plot.render();

// Reset
config.volume.data = data;
plot.render();
```

### Diagnostic Commands
```javascript
// Check K3D version and capabilities
K3DInstance.then(plot => {
    console.log('Available methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(plot)));
    console.log('Parameters:', plot.parameters);
    console.log('GUI:', plot.GUI);
});

// Memory check
K3DInstance.then(plot => {
    const world = plot.getWorld();
    let totalBytes = 0;
    for (const id in world.ObjectsListJson) {
        const obj = world.ObjectsListJson[id];
        if (obj.volume?.data) {
            const bytes = obj.volume.data.length * 4; // Float32
            totalBytes += bytes;
            console.log(`Object ${id}: ${(bytes/1024/1024).toFixed(2)} MB`);
        }
    }
    console.log(`Total volume memory: ${(totalBytes/1024/1024).toFixed(2)} MB`);
});
```

---

## Summary of Key Learnings

1. **K3D HTML exports have a completely different API** than Python K3D
2. **Always await K3DInstance** - it's a Promise
3. **Volume data lives in `world.ObjectsListJson[id].volume.data`**, not `plot.objects`
4. **Direct data manipulation is 10x faster** than using clipping planes
5. **Update both config and texture** when modifying volume data
6. **Use `plot.rebuildSceneData()` and `plot.render()`** after updates
7. **Global function scope is required** for HTML onclick handlers

## Additional Key Findings (From Recent Session)

8. **Fundamental K3D Limitations**:
   - Cannot decouple opacity from intensity (coupled through transfer functions)
   - Cannot completely hide axis tick labels (only color matching workaround)
   - Only one K3D instance per HTML page (use single plot with multiple volumes)

9. **Multi-Panel Visualization Patterns**:
   - Single K3D plot + multiple volumes > multiple iframes
   - Filter K3D objects: `Object.values(world.K3DObjects).filter(v => v.alpha_coef !== undefined)`
   - Synchronize all volume updates before calling `plot.render()`

10. **Data Quality Impact**:
    - Higher resolution significantly improves visual quality (81³ vs 41³)
    - Always normalize data to [0,1] before applying transformations
    - Handle NaN values with `np.nan_to_num()` before K3D processing

11. **JavaScript Integration Gotchas**:
    - Functions must be attached to `window` for HTML onclick handlers
    - K3DInstance Promise resolution varies by environment
    - Always use K3D setter methods (not direct property assignment)
    - Call `render()` after ANY property change to see visual updates

---

## When to Use This Guide

- Debugging K3D visualizations that work in Python but not in HTML export
- Optimizing slow K3D animations (likely too many clipping planes)
- Finding where volume data is stored in JavaScript
- Understanding the K3D HTML export object structure
- Implementing data masking or filtering in browser

---

*Last updated: November 2024*  
*Based on K3D version: 2.17.0*  
*Tested with: 41×41×41 volume data (68,921 voxels)*