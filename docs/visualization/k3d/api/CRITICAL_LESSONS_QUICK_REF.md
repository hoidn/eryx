# K3D Critical Lessons - Quick Reference Card

**Print this and keep handy when working with K3D!**

## 🚨 PERFORMANCE RED FLAGS 

### ❌ NEVER DO THIS:
```javascript
// Creating 60+ clipping planes for masking
const planes = [];
for (let i = 0; i < 60; i++) {
    planes.push([nx, ny, nz, d]);
}
plot.setClippingPlanes(planes); // KILLS PERFORMANCE (~5 FPS)
```

### ✅ DO THIS INSTEAD:
```javascript
// Direct data manipulation
const masked = originalData.map((value, i) => {
    return withinMask(i) ? value : 0;
}); // SMOOTH PERFORMANCE (~60 FPS)
```

## 🔧 JAVASCRIPT INTEGRATION RULES

### Rule 1: Global Functions for HTML
```javascript
// ❌ WRONG - Not accessible from HTML onclick
function startAnimation() { }

// ✅ CORRECT - Globally accessible  
window.startAnimation = function() { }
```

### Rule 2: K3D Promise Handling
```javascript
// ❌ WRONG - K3DInstance is a Promise
K3DInstance.setCamera([...]);

// ✅ CORRECT - Always await
const plot = await K3DInstance;
plot.setCamera([...]);
plot.render(); // REQUIRED!
```

### Rule 3: Volume Data Access
```javascript
// ❌ WRONG - plot.objects doesn't exist in HTML
const volumes = plot.objects;

// ✅ CORRECT - Use getWorld()
const world = plot.getWorld();
const volumeData = world.ObjectsListJson[id].volume.data;
```

## ⚡ PERFORMANCE HIERARCHY (FASTEST TO SLOWEST)

| Method | FPS | When to Use |
|--------|-----|-------------|
| **Direct Data Masking** | ~60 | Always prefer for complex shapes |
| **Simple Clipping (≤8 planes)** | ~30 | Basic geometric cuts only |
| **Many Clipping Planes (60+)** | ~5 | Never use |

## 📏 DATA RESOLUTION IMPACT

| Size | Quality | Use Case | Memory |
|------|---------|----------|--------|
| 41³ | Basic | Development | ~270KB |
| 81³ | Good | Production | ~2MB |
| 161³ | Excellent | Presentations | ~16MB |

**Tip**: The jump from 41³ to 81³ is visually dramatic!

## 🐛 TOP 5 DEBUGGING COMMANDS

```javascript
// 1. Check if K3D is available
console.log('K3D?', typeof K3DInstance, K3DInstance instanceof Promise);

// 2. Get plot and inspect
const plot = await K3DInstance;
console.log('Plot keys:', Object.keys(plot));

// 3. Find volume data
const world = plot.getWorld();
console.log('Volumes:', Object.keys(world.ObjectsListJson));

// 4. Check data size
const data = world.ObjectsListJson[id].volume.data;
console.log('Data length:', data.length, 'Expected:', 41*41*41);

// 5. Test render pipeline
plot.rebuildSceneData();
plot.render();
console.log('Render called - did visualization update?');
```

## 💾 DATA PREPARATION CHECKLIST

```python
# ✅ Always do these before K3D:
data = np.nan_to_num(data, nan=0.0)      # Handle NaN
data = (data - data.min()) / (data.max() - data.min())  # Normalize [0,1]
data = data.reshape(h, k, l)             # Ensure 3D shape
```

## 🎯 THE 5-MINUTE TEST RULE

**Before implementing ANYTHING in Python:**

1. **Open browser console** on K3D visualization
2. **Test the operation manually**:
   ```javascript
   const plot = await K3DInstance;
   plot.setCamera([5,5,5,0,0,0,0,1,0]);
   plot.render();
   // Did it work visually? ✅/❌
   ```
3. **Only then code it in Python**

**This rule prevents 90% of JavaScript integration issues!**

## 🚨 K3D HTML EXPORT GOTCHAS

- **Opacity cannot be decoupled from intensity** (transfer function limitation)
- **Axis labels cannot be fully hidden** (color matching only)
- **Only one K3D instance per page** (use multiple volumes instead)
- **Functions need global scope** for HTML onclick handlers
- **Always call render()** after any property changes

## 📞 EMERGENCY DEBUG SEQUENCE

If K3D visualization is broken:

1. **Check browser console** for JavaScript errors
2. **Verify K3D instance**: `await K3DInstance` works?
3. **Check data validity**: `data.length === expected`?
4. **Test manual render**: `plot.render()` fixes it?
5. **Verify global functions**: `typeof window.myFunction === 'function'`?

---

## 💡 GOLDEN RULES

1. **Performance**: Direct data > clipping planes
2. **JavaScript**: Always test in console first  
3. **Functions**: Attach to window for HTML
4. **K3D Access**: Always await the Promise
5. **Updates**: Call render() after changes

**Keep this card visible while coding K3D!** 🎯

---

*Based on 30+ hours of debugging distilled into essentials*  
*Last updated: August 2024*