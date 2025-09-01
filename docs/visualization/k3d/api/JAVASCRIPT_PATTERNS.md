# K3D JavaScript Integration Patterns & Examples

This document contains specific patterns and examples that would have prevented the K3D animation issues. These are battle-tested solutions to common problems.

> **📚 Related Documentation**:
> - **[K3D HTML/JavaScript API Reference](./HTML_JAVASCRIPT_API.md)** - Complete technical API details, HTML export structure, and volume data access
> - **[Production Usage Guide](../getting-started/QUICK_START.md)** - Working examples and quick start
> - **[Project Integration](../../../../CLAUDE.md)** - Eryx-specific patterns and context

## Table of Contents
1. [Global Function Registration Pattern](#global-function-registration-pattern)
2. [K3D Instance Access Pattern](#k3d-instance-access-pattern)
3. [Animation State Management](#animation-state-management)
4. [Debugging Pattern for K3D](#debugging-pattern-for-k3d)
5. [HTML Generation Testing Pattern](#html-generation-testing-pattern)

---

## Global Function Registration Pattern

### ❌ What Breaks (Silent Failure)
```javascript
// In generated HTML - functions aren't accessible
function startAnimation() {
    // Animation code
}
```
**Result**: `Uncaught ReferenceError: startAnimation is not defined`

### ✅ Working Pattern
```javascript
// PATTERN: Global Function Registration for HTML onclick
// WHY: HTML onclick handlers execute in global scope
// WHEN: Always when generating HTML with JavaScript

window.startAnimation = function() {
    console.log('✅ Function called from onclick');
    // Animation code
}

// VALIDATION TEST:
console.assert(typeof window.startAnimation === 'function', 'Function not global!');
```

### Usage in Python
```python
def generate_js_with_global_functions():
    return """
    window.startAnimation = function() { /* ... */ };
    window.stopAnimation = function() { /* ... */ };
    // All functions attached to window for HTML access
    """
```

---

## K3D Instance Access Pattern

### ❌ What Breaks (No Camera Update)
```javascript
// Assumed K3DInstance was the plot object
K3DInstance.camera = [1, 2, 3, 0, 0, 0, 0, 1, 0];
```
**Result**: No error but camera doesn't update

### ✅ Working Pattern
```javascript
// PATTERN: Safe K3D Instance Access
// WHY: K3DInstance is a Promise in Jupyter environments
// WHEN: Always when accessing K3D from injected JavaScript

async function updateK3DCamera(newCamera) {
    // Step 1: Safely get the plot object
    let plot;
    if (window.K3DInstance instanceof Promise) {
        plot = await window.K3DInstance;
        console.log('✅ Resolved K3D Promise');
    } else if (window.K3DInstance) {
        plot = window.K3DInstance;
        console.log('✅ Direct K3D access');
    } else {
        console.error('❌ No K3D instance found');
        return;
    }
    
    // Step 2: Use proper API methods
    plot.setCamera(newCamera);  // NOT plot.camera = 
    plot.render();               // REQUIRED to see changes
    
    console.log('✅ Camera updated');
}

// Make it globally accessible
window.updateK3DCamera = updateK3DCamera;

// VALIDATION TEST:
(async () => {
    await updateK3DCamera([5, 5, 5, 0, 0, 0, 0, 1, 0]);
    // Camera should visibly change
})();
```

---

## Animation State Management

### ❌ What Breaks (Conflicting Animations)
```javascript
// Multiple animations running simultaneously
function startOrbit() {
    setInterval(() => { /* orbit code */ }, 16);
}
function startZoom() {
    setInterval(() => { /* zoom code */ }, 16);
}
```
**Result**: "Spazzy" conflicting animations

### ✅ Working Pattern
```javascript
// PATTERN: Single Animation Controller
// WHY: Multiple requestAnimationFrame loops conflict
// WHEN: Managing any animations in K3D

class K3DAnimationController {
    constructor() {
        this.currentAnimation = null;
        this.animationId = null;
    }
    
    start(animationType) {
        // Stop any existing animation
        this.stop();
        
        console.log(`Starting ${animationType} animation`);
        this.currentAnimation = animationType;
        
        // Single animation loop
        const animate = () => {
            if (!this.currentAnimation) return;
            
            switch(this.currentAnimation) {
                case 'orbit':
                    this.updateOrbit();
                    break;
                case 'zoom':
                    this.updateZoom();
                    break;
            }
            
            this.animationId = requestAnimationFrame(animate);
        };
        
        animate();
    }
    
    stop() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
            this.currentAnimation = null;
            console.log('Animation stopped');
        }
    }
    
    async updateOrbit() {
        // Orbit logic with proper K3D access
        const plot = await window.K3DInstance;
        // ... update camera
        plot.render();
    }
    
    async updateZoom() {
        // Zoom logic
        const plot = await window.K3DInstance;
        // ... update camera
        plot.render();
    }
}

// Single global instance
window.animationController = new K3DAnimationController();

// HTML can now call:
// onclick="animationController.start('orbit')"
// onclick="animationController.stop()"

// VALIDATION TEST:
window.animationController.start('orbit');
setTimeout(() => {
    console.assert(window.animationController.currentAnimation === 'orbit');
    window.animationController.stop();
}, 100);
```

---

## Debugging Pattern for K3D

### ❌ What Breaks (Silent Failures)
```javascript
// Silent failures with no debugging info
k3d.setCamera(camera);
// Is it working? Who knows!
```

### ✅ Working Pattern
```javascript
// PATTERN: K3D Debug Wrapper
// WHY: K3D operations can fail silently
// WHEN: Development and debugging

class K3DDebugger {
    constructor() {
        this.logLevel = 'debug';  // 'debug', 'info', 'error'
        this.history = [];
    }
    
    async executeOperation(operation, description) {
        const timestamp = Date.now();
        const entry = {
            timestamp,
            description,
            status: 'pending'
        };
        
        this.history.push(entry);
        
        try {
            // Get K3D instance
            const plot = await this.getPlot();
            if (!plot) {
                entry.status = 'failed';
                entry.error = 'No K3D instance';
                this.log('error', `❌ ${description}: No K3D instance`);
                return false;
            }
            
            // Execute operation
            this.log('debug', `⏳ ${description}`);
            const result = await operation(plot);
            
            // Verify success (check if render needed)
            if (plot.render) {
                plot.render();
            }
            
            entry.status = 'success';
            entry.result = result;
            this.log('info', `✅ ${description}: Success`);
            
            return true;
            
        } catch (error) {
            entry.status = 'failed';
            entry.error = error.message;
            this.log('error', `❌ ${description}: ${error.message}`);
            console.error(error);
            return false;
        }
    }
    
    async getPlot() {
        if (window.K3DInstance instanceof Promise) {
            return await window.K3DInstance;
        }
        return window.K3DInstance || window.k3d || null;
    }
    
    log(level, message) {
        const levels = ['debug', 'info', 'error'];
        if (levels.indexOf(level) >= levels.indexOf(this.logLevel)) {
            console.log(`[K3D ${level.toUpperCase()}] ${message}`);
        }
    }
    
    getHistory() {
        return this.history;
    }
    
    printSummary() {
        const success = this.history.filter(h => h.status === 'success').length;
        const failed = this.history.filter(h => h.status === 'failed').length;
        console.log(`📊 K3D Operations: ${success} succeeded, ${failed} failed`);
        
        if (failed > 0) {
            console.log('Failed operations:');
            this.history
                .filter(h => h.status === 'failed')
                .forEach(h => console.log(`  - ${h.description}: ${h.error}`));
        }
    }
}

// Usage:
window.k3dDebug = new K3DDebugger();

// Now wrap all K3D operations:
window.updateCamera = async function(camera) {
    return await k3dDebug.executeOperation(
        async (plot) => plot.setCamera(camera),
        'Update camera position'
    );
};

// VALIDATION TEST:
await window.updateCamera([5, 5, 5, 0, 0, 0, 0, 1, 0]);
window.k3dDebug.printSummary();
// Should show: "✅ Update camera position: Success"
```

---

## HTML Generation Testing Pattern

### ❌ What Breaks (Untested JavaScript)
```python
# Generated HTML without testing the JavaScript
html = f"<script>{animation_code}</script>"
# Ship it! (It doesn't work)
```

### ✅ Working Pattern
```python
# PATTERN: Test-First HTML Generation
# WHY: JavaScript must be validated before embedding
# WHEN: Generating any HTML with JavaScript

import tempfile
import subprocess
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def generate_and_test_k3d_html(plot, animation_code):
    """
    Generate HTML with embedded JavaScript and validate it works.
    """
    
    # Step 1: Create test harness HTML
    test_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>K3D Animation Test</title>
    </head>
    <body>
        <div id="test-status">INITIALIZING</div>
        <button id="test-button" onclick="testAnimation()">Test</button>
        
        <!-- K3D Plot -->
        {plot.get_snapshot()}
        
        <!-- Animation Code with Testing -->
        <script>
        // Test framework
        window.testResults = {{
            k3dFound: false,
            functionsRegistered: false,
            animationRuns: false,
            errors: []
        }};
        
        window.addEventListener('error', function(e) {{
            window.testResults.errors.push(e.message);
            document.getElementById('test-status').textContent = 'ERROR: ' + e.message;
        }});
        
        // Injected animation code
        {animation_code}
        
        // Test the animation
        window.testAnimation = async function() {{
            try {{
                // Check K3D exists
                if (window.K3DInstance || window.k3d) {{
                    window.testResults.k3dFound = true;
                }}
                
                // Check functions registered
                if (typeof window.startAnimation === 'function') {{
                    window.testResults.functionsRegistered = true;
                }}
                
                // Try to run animation
                await window.startAnimation();
                window.testResults.animationRuns = true;
                
                // Update status
                const success = Object.values(window.testResults).every(v => 
                    Array.isArray(v) ? v.length === 0 : v === true
                );
                
                document.getElementById('test-status').textContent = 
                    success ? 'SUCCESS' : 'FAILED';
                    
            }} catch (e) {{
                window.testResults.errors.push(e.message);
                document.getElementById('test-status').textContent = 'ERROR: ' + e.message;
            }}
        }};
        
        // Auto-test after load
        setTimeout(window.testAnimation, 2000);
        </script>
    </body>
    </html>
    """
    
    # Step 2: Save and test in real browser
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(test_html)
        temp_path = f.name
    
    try:
        # Step 3: Automated browser test
        driver = webdriver.Chrome()  # or Firefox()
        driver.get(f"file://{temp_path}")
        
        # Wait for test to complete
        wait = WebDriverWait(driver, 10)
        status_element = wait.until(
            EC.text_to_be_present_in_element((By.ID, "test-status"), "SUCCESS")
        )
        
        # Get test results
        results = driver.execute_script("return window.testResults;")
        
        # Validate results
        assert results['k3dFound'], "K3D instance not found"
        assert results['functionsRegistered'], "Functions not registered globally"
        assert results['animationRuns'], "Animation failed to run"
        assert len(results['errors']) == 0, f"JavaScript errors: {results['errors']}"
        
        print("✅ HTML generation test passed!")
        
        # Step 4: Return validated HTML
        return test_html.replace(
            '<div id="test-status">INITIALIZING</div>',
            ''  # Remove test elements for production
        )
        
    finally:
        driver.quit()
        os.unlink(temp_path)

# USAGE:
validated_html = generate_and_test_k3d_html(plot, animation_js)
# Only save HTML that passed tests!
```

---

## The Meta-Pattern: Test in Target Environment

### The Universal Rule That Would Have Prevented Everything:

**🚨 GOLDEN RULE: Test Where It Will Run**

Before implementing ANYTHING:

1. **Open the target environment** (Jupyter, browser, etc.)
2. **Try the operation manually** in console
3. **Verify it works**
4. **THEN implement in Python**

### The 5-Minute Test Workflow:

```javascript
// In browser console, test your idea:
console.log('K3D available?', typeof K3DInstance);
console.log('Is Promise?', K3DInstance instanceof Promise);

// Try your operation:
const plot = await K3DInstance;
plot.setCamera([1,2,3,0,0,0,0,1,0]);
plot.render();

// Did it work visually? 
// YES -> Now you can implement
// NO -> Debug here first!
```

### The Test-First Development Pattern:

1. **Browser Console** (2 min)
   - Test the JavaScript manually
   - Verify visual changes happen
   
2. **Simple HTML** (3 min)
   - Create minimal test.html
   - Verify onclick works
   
3. **Python Integration** (5 min)
   - Generate HTML with tested code
   - Run automated tests
   
4. **Documentation** (2 min)
   - Document what worked
   - Include test results

**Time: 12 minutes to working feature**
**vs: 30 hours debugging broken assumptions**

---

## Common Error Messages and Solutions

### Error: "startAnimation is not defined"
**Cause**: Function not in global scope
**Solution**: Use `window.startAnimation = function() {}`

### Error: "Cannot read property 'camera' of undefined"
**Cause**: K3DInstance is a Promise, not resolved
**Solution**: Use `await K3DInstance` first

### Error: "plot.camera is not a function"
**Cause**: Using wrong API method
**Solution**: Use `plot.setCamera()` not `plot.camera()`

### Visual: "Nothing happens when clicking button"
**Cause**: JavaScript error preventing execution
**Solution**: Check browser console for errors

### Visual: "Animation looks jerky/spazzy"
**Cause**: Multiple animation loops running
**Solution**: Use single animation controller pattern

---

## Recent Patterns from Performance Optimization Session

### Pattern: Direct Data Masking for Performance
```javascript
// PATTERN: High-performance volume masking
// WHY: 60+ clipping planes = 5 FPS, direct data = 60 FPS
// WHEN: Real-time interactive masking needed

class K3DDataMasker {
    constructor() {
        this.originalData = null;
        this.currentMask = null;
        this.volumeInfo = null;
    }
    
    async initialize() {
        const plot = await window.K3DInstance;
        const world = plot.getWorld();
        
        // Find and cache volume info
        for (const id in world.ObjectsListJson) {
            if (world.ObjectsListJson[id].type === 'Volume') {
                this.volumeInfo = {
                    id: id,
                    config: world.ObjectsListJson[id],
                    mesh: world.ObjectsById[id]
                };
                this.originalData = new Float32Array(this.volumeInfo.config.volume.data);
                console.log('✅ DataMasker initialized, data size:', this.originalData.length);
                break;
            }
        }
    }
    
    createSphericalMask(radiusPercent) {
        const shape = this.volumeInfo.config.volume.shape;
        const [h, k, l] = shape;
        const center = [h/2, k/2, l/2];
        const radius = Math.min(h, k, l) * 0.5 * (radiusPercent / 100);
        
        const mask = new Float32Array(this.originalData.length);
        let idx = 0;
        
        for (let i = 0; i < h; i++) {
            for (let j = 0; j < k; j++) {
                for (let m = 0; m < l; m++) {
                    const dist = Math.sqrt(
                        (i - center[0])**2 + 
                        (j - center[1])**2 + 
                        (m - center[2])**2
                    );
                    mask[idx] = (dist <= radius) ? 1.0 : 0.0;
                    idx++;
                }
            }
        }
        
        return mask;
    }
    
    async applyMask(mask) {
        const plot = await window.K3DInstance;
        
        // Apply mask to data
        const maskedData = new Float32Array(this.originalData.length);
        for (let i = 0; i < this.originalData.length; i++) {
            maskedData[i] = this.originalData[i] * mask[i];
        }
        
        // Update K3D visualization
        this.volumeInfo.config.volume.data = maskedData;
        
        // Update THREE.js texture if exists
        const texture = this.volumeInfo.mesh?.material?.uniforms?.volumeTexture?.value;
        if (texture?.image?.data) {
            texture.image.data.set(maskedData);
            texture.needsUpdate = true;
        }
        
        this.currentMask = mask;
        plot.rebuildSceneData();
        plot.render();
    }
    
    async clearMask() {
        const plot = await window.K3DInstance;
        
        // Restore original data
        this.volumeInfo.config.volume.data = new Float32Array(this.originalData);
        
        const texture = this.volumeInfo.mesh?.material?.uniforms?.volumeTexture?.value;
        if (texture?.image?.data) {
            texture.image.data.set(this.originalData);
            texture.needsUpdate = true;
        }
        
        this.currentMask = null;
        plot.rebuildSceneData();
        plot.render();
    }
}

// Global instance for HTML access
window.dataMasker = new K3DDataMasker();
window.dataMasker.initialize();
```

### Pattern: Multi-Volume Synchronized Updates
```javascript
// PATTERN: Synchronized multi-volume updates
// WHY: Multiple render() calls cause flickering
// WHEN: Comparing multiple datasets side-by-side

class K3DMultiVolumeController {
    constructor() {
        this.volumes = [];
        this.updateQueue = [];
    }
    
    async findAllVolumes() {
        const plot = await window.K3DInstance;
        const world = plot.getWorld();
        
        this.volumes = [];
        for (const id in world.ObjectsListJson) {
            if (world.ObjectsListJson[id].type === 'Volume') {
                this.volumes.push({
                    id: id,
                    config: world.ObjectsListJson[id],
                    mesh: world.ObjectsById[id],
                    originalData: new Float32Array(world.ObjectsListJson[id].volume.data)
                });
            }
        }
        
        console.log(`✅ Found ${this.volumes.length} volumes`);
        return this.volumes;
    }
    
    queueUpdate(volumeIndex, newData) {
        this.updateQueue.push({ volumeIndex, newData });
    }
    
    async flushUpdates() {
        const plot = await window.K3DInstance;
        
        // Apply all queued updates
        for (const update of this.updateQueue) {
            const volume = this.volumes[update.volumeIndex];
            volume.config.volume.data = update.newData;
            
            const texture = volume.mesh?.material?.uniforms?.volumeTexture?.value;
            if (texture?.image?.data) {
                texture.image.data.set(update.newData);
                texture.needsUpdate = true;
            }
        }
        
        // Single render call for all updates
        this.updateQueue = [];
        plot.rebuildSceneData();
        plot.render();
    }
    
    async synchronizedMask(mask) {
        // Apply same mask to all volumes
        for (let i = 0; i < this.volumes.length; i++) {
            const volume = this.volumes[i];
            const maskedData = new Float32Array(volume.originalData.length);
            for (let j = 0; j < volume.originalData.length; j++) {
                maskedData[j] = volume.originalData[j] * mask[j];
            }
            this.queueUpdate(i, maskedData);
        }
        
        await this.flushUpdates();
    }
}

// Global instance
window.multiVolumeController = new K3DMultiVolumeController();
window.multiVolumeController.findAllVolumes();
```

## Summary

These patterns represent ~30 hours of debugging distilled into reusable solutions. Always:

1. **Test in browser console first**
2. **Use `window.` for global functions**
3. **Await K3D Promises**
4. **Use setter methods with `.render()`**
5. **Wrap in debug logging during development**
6. **Prefer direct data manipulation over clipping planes** (10x performance gain)
7. **Batch updates and render once** for multi-volume scenarios
8. **Cache volume references** to avoid repeated world traversal

When in doubt, return to the golden rule: **Test where it will run!**

## Performance Lessons

- **60+ clipping planes**: Unusable performance (~5 FPS)
- **Direct data masking**: Excellent performance (~60 FPS) 
- **Single render() call**: Prevents flickering in multi-volume updates
- **Float32Array operations**: Much faster than regular arrays
- **Cached references**: Avoid repeated `getWorld()` calls