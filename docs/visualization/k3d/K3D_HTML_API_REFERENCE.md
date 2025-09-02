# K3D HTML Export API Quick Reference

**⚠️ This guide covers ONLY K3D HTML exports, NOT Python K3D**

## Critical Differences from Python K3D

| Operation | ❌ Python K3D (WRONG in HTML) | ✅ HTML Export (CORRECT) |
|-----------|-------------------------------|--------------------------|
| **Plot Access** | `plot = k3d.plot()` | `const plot = await window.K3DInstance` |
| **Objects Access** | `plot.objects[0]` | `world.ObjectsListJson[volumeId]` |
| **Volume Data** | `volume.volume` | `world.ObjectsListJson[id].volume.data` |
| **Property Updates** | `volume.alpha_coef = 0.5` | `json.alpha_coef = 0.5; plot.reload(json, changes)` |
| **Rendering** | Automatic | `plot.render()` or via `plot.reload()` |

## Essential Methods

### Get K3D Plot
```javascript
const plot = await window.K3DInstance;  // Always await!
```

### Find Volume Data
```javascript
const world = plot.getWorld();
let volumeId = null;
for (const id in world.ObjectsListJson) {
    if (world.ObjectsListJson[id].type === 'Volume') {
        volumeId = id;
        break;
    }
}
const volumeConfig = world.ObjectsListJson[volumeId];
```

### Update Volume Properties (THE ONLY WAY)
```javascript
// Step 1: Update JSON configuration
const json = world.ObjectsListJson[volumeId];
json.alpha_coef = newValue;

// Step 2: Apply changes using reload()
const changes = { alpha_coef: newValue };
plot.reload(json, changes);
// Note: reload() handles rendering automatically
```

## Methods That DON'T EXIST

### 🚨 NEVER USE THESE - THEY WILL FAIL:
- `plot.setAttributes()` - **Method does not exist**
- `plot.objects` - **Array does not exist** 
- `plot.volumes` - **Property does not exist**
- `volume.alpha_coef = value` - **Direct assignment doesn't work**

## Common Properties You Can Update

All updates must use the `reload()` pattern:

```javascript
// Opacity/transparency
json.alpha_coef = 15.0;
plot.reload(json, {alpha_coef: 15.0});

// Color range
json.color_range = [0.0, 1.0];
plot.reload(json, {color_range: [0.0, 1.0]});

// Sampling quality
json.samples = 512;
plot.reload(json, {samples: 512});

// Gradient step (affects quality)
json.gradient_step = 0.005;
plot.reload(json, {gradient_step: 0.005});
```

## Volume Data Access & Modification

### Read Volume Data
```javascript
const volumeData = volumeConfig.volume.data;  // Float32Array
const volumeShape = volumeConfig.volume.shape; // [h, k, l]
```

### Update Volume Data
```javascript
// Create new data array
const newData = new Float32Array(originalData.length);
// ... modify newData ...

// Update in configuration
volumeConfig.volume.data = newData;

// Update 3D texture (if exists)
const volumeMesh = world.ObjectsById[volumeId];
if (volumeMesh?.material?.uniforms?.volumeTexture?.value) {
    const texture = volumeMesh.material.uniforms.volumeTexture.value;
    if (texture.image?.data) {
        texture.image.data.set(newData);
        texture.needsUpdate = true;
    }
}

// Force re-render
plot.rebuildSceneData();
plot.render();
```

## Debugging Commands

### Check What Methods Exist
```javascript
K3DInstance.then(plot => {
    console.log('Available methods:', 
        Object.getOwnPropertyNames(Object.getPrototypeOf(plot))
    );
});
```

### Find All Volumes
```javascript
K3DInstance.then(plot => {
    const world = plot.getWorld();
    for (const id in world.ObjectsListJson) {
        const obj = world.ObjectsListJson[id];
        if (obj.type === 'Volume') {
            console.log(`Volume ID: ${id}`, obj);
        }
    }
});
```

### Test Property Update
```javascript
// Quick opacity test
K3DInstance.then(async plot => {
    const world = plot.getWorld();
    const volumeId = Object.keys(world.ObjectsListJson)
        .find(id => world.ObjectsListJson[id].type === 'Volume');
    
    const json = world.ObjectsListJson[volumeId];
    json.alpha_coef = 25.0;  // Make more opaque
    plot.reload(json, {alpha_coef: 25.0});
    
    console.log('Opacity updated successfully!');
});
```

## Performance Guidelines

### ✅ FAST (Recommended)
- Direct data manipulation
- Using reload() for property updates
- 1-6 clipping planes maximum
- Data masking in JavaScript

### ❌ SLOW (Avoid)
- 60+ clipping planes for masking
- Frequent texture updates without batching
- Large data arrays without optimization
- Multiple K3D instances per page

## Common Workflow Pattern

```javascript
// Complete property update workflow
async function updateK3DProperty(property, value) {
    try {
        // 1. Get plot
        const plot = await window.K3DInstance;
        
        // 2. Get world
        const world = plot.getWorld();
        
        // 3. Find volume
        const volumeId = Object.keys(world.ObjectsListJson)
            .find(id => world.ObjectsListJson[id].type === 'Volume');
        
        if (!volumeId) {
            throw new Error('No volume found');
        }
        
        // 4. Update JSON
        const json = world.ObjectsListJson[volumeId];
        json[property] = value;
        
        // 5. Apply with reload
        const changes = {};
        changes[property] = value;
        plot.reload(json, changes);
        
        console.log(`✅ Updated ${property} to ${value}`);
        
    } catch (error) {
        console.error('❌ Update failed:', error);
    }
}

// Usage
await updateK3DProperty('alpha_coef', 20.0);
await updateK3DProperty('color_range', [0.0, 0.8]);
```

## Key Takeaways

1. **Always use `await window.K3DInstance`** to get the plot
2. **Only use `plot.reload(json, changes)`** for property updates
3. **Never use `setAttributes()` or direct property assignment**
4. **Volume data is in `world.ObjectsListJson[id].volume.data`**
5. **Test in browser console first** before integrating into code
6. **Ignore any examples that mention `setAttributes()`** - they're wrong

---

*Last updated: January 2025*  
*For detailed examples and troubleshooting, see [HTML_JAVASCRIPT_API.md](api/HTML_JAVASCRIPT_API.md)*