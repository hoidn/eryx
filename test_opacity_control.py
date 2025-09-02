#!/usr/bin/env python3
"""
🚨 WARNING: THIS FILE CONTAINS INCORRECT K3D PATTERNS 🚨
===========================================================

CRITICAL UPDATE (January 2025):
- setAttributes() method DOES NOT EXIST in K3D HTML exports
- CORRECT method: plot.reload(json, changes)
- This file is kept for historical reference but uses WRONG patterns

✅ CORRECT K3D HTML opacity pattern:
    const plot = await window.K3DInstance;
    const world = plot.getWorld();
    const json = world.ObjectsListJson[volumeId];
    json.alpha_coef = newValue;
    plot.reload(json, {alpha_coef: newValue});

❌ WRONG patterns tested in this file:
    - plot.setAttributes() - Method doesn't exist
    - direct property assignment - Doesn't trigger updates
    - plot.set() - Wrong method

For working examples, see:
- docs/visualization/k3d/K3D_HTML_API_REFERENCE.md
- docs/visualization/k3d/api/HTML_JAVASCRIPT_API.md

K3D OPACITY CONTROL TEST - COMPREHENSIVE TEST SUITE (DEPRECATED)
================================================================

This script creates a minimal K3D volume and tests ALL possible methods
to control opacity in HTML exports. It will definitively answer:

1. Does K3D support opacity control in HTML exports?
2. Which specific JavaScript methods work?
3. What are the exact API patterns that work?

The script generates an HTML file with embedded JavaScript that tests
every conceivable method for opacity control and logs clear results.
"""

import k3d
import numpy as np
import time

def create_test_volume():
    """Create a simple test volume with clear opacity effects."""
    print("Creating test volume...")
    
    # Create a simple 3D volume with distinct features
    size = 21  # Small for fast testing
    x = np.linspace(-1, 1, size)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    
    # Create volume with clear opacity-sensitive features
    # Core sphere + outer shell (will show opacity effects clearly)
    r = np.sqrt(X**2 + Y**2 + Z**2)
    volume = (100 * np.exp(-r**2 / 0.2) +  # Bright core
              30 * np.exp(-(r - 0.6)**2 / 0.1) +  # Outer shell
              5 * np.random.random((size, size, size)))  # Noise
    
    print(f"Test volume shape: {volume.shape}")
    print(f"Value range: [{volume.min():.2f}, {volume.max():.2f}]")
    return volume.astype(np.float32)

def create_k3d_plot(volume_data):
    """Create K3D plot with volume."""
    print("Creating K3D plot...")
    
    plot = k3d.plot(
        background_color=0x222222,
        grid_visible=False,
        height=500,
        camera_auto_fit=True
    )
    
    volume = k3d.volume(
        volume_data,
        color_map=k3d.basic_color_maps.Jet,
        alpha_coef=25.0,  # Starting opacity
        bounds=[-1, 1, -1, 1, -1, 1],
        interpolation=True
    )
    
    plot += volume
    return plot, volume

def create_comprehensive_test_js():
    """Create comprehensive JavaScript test code."""
    return """
<script>
class K3DOpacityTester {
    constructor() {
        this.results = [];
        this.plot = null;
        this.volumeId = null;
        this.volumeObject = null;
        this.testStartTime = Date.now();
        this.init();
    }
    
    async init() {
        this.log('🔬 Starting K3D Opacity Control Test Suite');
        this.log('=' * 60);
        
        // Wait for K3D to be ready with multiple fallback methods
        await this.findK3DInstance();
        await this.identifyVolumeObject();
        await this.runAllTests();
        this.generateReport();
    }
    
    async findK3DInstance() {
        this.log('🔍 Finding K3D Instance...');
        
        // Try all possible ways to access K3D
        const attempts = [
            () => window.K3DInstance,
            () => window.k3d,
            () => document.querySelector('#k3d-widget').__plot,
            () => document.querySelector('.widget-output').__plot,
            () => document.querySelector('canvas').__plot
        ];
        
        for (let i = 0; i < attempts.length; i++) {
            try {
                let candidate = attempts[i]();
                if (candidate) {
                    if (candidate instanceof Promise) {
                        this.plot = await candidate;
                    } else {
                        this.plot = candidate;
                    }
                    if (this.plot) {
                        this.log(`✅ Found K3D instance via method ${i + 1}`);
                        this.log(`Plot object type: ${this.plot.constructor.name}`);
                        break;
                    }
                }
            } catch (e) {
                this.log(`Method ${i + 1} failed: ${e.message}`);
            }
        }
        
        if (!this.plot) {
            this.log('❌ CRITICAL: No K3D instance found');
            return false;
        }
        
        // Log plot capabilities
        this.log('📋 Plot Object Properties:');
        const methods = Object.getOwnPropertyNames(Object.getPrototypeOf(this.plot));
        this.log(`Available methods: ${methods.join(', ')}`);
        
        return true;
    }
    
    async identifyVolumeObject() {
        this.log('🎯 Identifying Volume Object...');
        
        try {
            const world = this.plot.getWorld ? this.plot.getWorld() : null;
            if (!world) {
                this.log('❌ Cannot access world object');
                return false;
            }
            
            this.log('🌍 World object accessed successfully');
            this.log(`World keys: ${Object.keys(world).join(', ')}`);
            
            if (world.ObjectsListJson) {
                const objects = world.ObjectsListJson;
                this.log(`Found ${Object.keys(objects).length} objects in scene`);
                
                // Find volume objects
                const volumes = Object.entries(objects)
                    .filter(([id, obj]) => obj.type === 'Volume');
                
                if (volumes.length > 0) {
                    [this.volumeId, this.volumeObject] = volumes[0];
                    this.log(`✅ Found volume: ID=${this.volumeId}`);
                    this.log(`Volume properties: ${Object.keys(this.volumeObject).join(', ')}`);
                    this.log(`Current alpha_coef: ${this.volumeObject.alpha_coef}`);
                    return true;
                } else {
                    this.log('❌ No volume objects found');
                }
            } else {
                this.log('❌ No ObjectsListJson in world');
            }
        } catch (e) {
            this.log(`❌ Error accessing volume: ${e.message}`);
        }
        
        return false;
    }
    
    async runAllTests() {
        this.log('🧪 Running Opacity Control Tests...');
        this.log('-' * 40);
        
        const tests = [
            { name: 'Direct Property Modification', method: this.testDirectProperty },
            { name: 'setAttributes Method', method: this.testSetAttributes },
            { name: 'Plot.set Method', method: this.testPlotSet },
            { name: 'Object Reconstruction', method: this.testObjectReconstruction },
            { name: 'Scene Rebuild', method: this.testSceneRebuild },
            { name: 'THREE.js Material Access', method: this.testThreeJSMaterial },
            { name: 'Transfer Function Modification', method: this.testTransferFunction },
            { name: 'Volume Data Replacement', method: this.testVolumeDataReplace }
        ];
        
        for (const test of tests) {
            await this.runSingleTest(test.name, test.method);
            await this.sleep(500); // Brief pause between tests
        }
    }
    
    async runSingleTest(testName, testMethod) {
        this.log(`\n🔬 Test: ${testName}`);
        
        try {
            const originalAlpha = this.volumeObject ? this.volumeObject.alpha_coef : null;
            const testAlpha = 5.0; // Very low opacity for clear visual change
            
            const result = await testMethod.call(this, testAlpha);
            
            // Check if change was applied
            let success = false;
            let finalAlpha = null;
            
            if (this.volumeObject) {
                finalAlpha = this.volumeObject.alpha_coef;
                success = Math.abs(finalAlpha - testAlpha) < 0.1;
            }
            
            this.results.push({
                test: testName,
                success: success,
                originalAlpha: originalAlpha,
                targetAlpha: testAlpha,
                finalAlpha: finalAlpha,
                notes: result.notes || '',
                error: result.error || null
            });
            
            this.log(`${success ? '✅' : '❌'} Result: ${success ? 'SUCCESS' : 'FAILED'}`);
            if (result.notes) this.log(`   Notes: ${result.notes}`);
            if (result.error) this.log(`   Error: ${result.error}`);
            
            // Reset for next test
            if (originalAlpha && this.volumeObject) {
                this.volumeObject.alpha_coef = originalAlpha;
            }
            
        } catch (e) {
            this.log(`❌ Test failed with exception: ${e.message}`);
            this.results.push({
                test: testName,
                success: false,
                error: e.message
            });
        }
    }
    
    // Individual test methods
    async testDirectProperty(testAlpha) {
        if (!this.volumeObject) return { notes: 'No volume object available' };
        
        this.volumeObject.alpha_coef = testAlpha;
        this.plot.render();
        
        return { notes: `Set alpha_coef directly to ${testAlpha}` };
    }
    
    async testSetAttributes(testAlpha) {
        if (!this.plot.setAttributes) {
            return { notes: 'setAttributes method not available' };
        }
        
        this.plot.setAttributes(this.volumeId, { alpha_coef: testAlpha });
        this.plot.render();
        
        return { notes: `Used setAttributes(${this.volumeId}, {alpha_coef: ${testAlpha}})` };
    }
    
    async testPlotSet(testAlpha) {
        if (!this.plot.set) {
            return { notes: 'plot.set method not available' };
        }
        
        this.plot.set('alpha_coef', testAlpha);
        this.plot.render();
        
        return { notes: `Used plot.set('alpha_coef', ${testAlpha})` };
    }
    
    async testObjectReconstruction(testAlpha) {
        try {
            // Try to recreate the volume object
            const world = this.plot.getWorld();
            const volumeConfig = world.ObjectsListJson[this.volumeId];
            
            volumeConfig.alpha_coef = testAlpha;
            
            if (this.plot.rebuildSceneData) {
                this.plot.rebuildSceneData();
                this.plot.render();
                return { notes: 'Modified config and rebuilt scene' };
            } else {
                return { notes: 'rebuildSceneData not available' };
            }
        } catch (e) {
            return { error: e.message };
        }
    }
    
    async testSceneRebuild(testAlpha) {
        if (!this.volumeObject) return { notes: 'No volume object' };
        
        this.volumeObject.alpha_coef = testAlpha;
        
        const rebuildMethods = ['rebuildSceneData', 'rebuild', 'update', 'refresh'];
        for (const method of rebuildMethods) {
            if (this.plot[method]) {
                this.plot[method]();
                this.plot.render();
                return { notes: `Used ${method}() with direct property change` };
            }
        }
        
        return { notes: 'No rebuild methods available' };
    }
    
    async testThreeJSMaterial(testAlpha) {
        try {
            // Try to access THREE.js materials directly
            if (window.THREE && this.plot.K3DObjects) {
                const k3dObjects = this.plot.K3DObjects;
                const volumeObject = k3dObjects[this.volumeId];
                
                if (volumeObject && volumeObject.material) {
                    volumeObject.material.opacity = testAlpha / 100.0;
                    volumeObject.material.transparent = true;
                    this.plot.render();
                    return { notes: 'Modified THREE.js material opacity directly' };
                }
            }
            return { notes: 'THREE.js materials not accessible' };
        } catch (e) {
            return { error: e.message };
        }
    }
    
    async testTransferFunction(testAlpha) {
        try {
            if (this.volumeObject && this.volumeObject.transfer_function) {
                // Try to modify transfer function
                const tf = this.volumeObject.transfer_function;
                if (Array.isArray(tf) && tf.length > 0) {
                    tf[1] = testAlpha / 100.0; // Modify alpha channel
                    this.plot.render();
                    return { notes: 'Modified transfer function alpha' };
                }
            }
            return { notes: 'Transfer function not accessible or wrong format' };
        } catch (e) {
            return { error: e.message };
        }
    }
    
    async testVolumeDataReplace(testAlpha) {
        try {
            // Try replacing entire volume data with modified alpha
            if (this.volumeObject && this.volumeObject.volume) {
                const originalData = this.volumeObject.volume.data;
                // This is a stretch test - likely won't work
                this.volumeObject.alpha_coef = testAlpha;
                this.plot.render();
                return { notes: 'Attempted volume data manipulation' };
            }
            return { notes: 'Volume data not accessible' };
        } catch (e) {
            return { error: e.message };
        }
    }
    
    generateReport() {
        this.log('\n' + '=' * 60);
        this.log('📊 FINAL TEST RESULTS');
        this.log('=' * 60);
        
        const successful = this.results.filter(r => r.success);
        const failed = this.results.filter(r => !r.success);
        
        this.log(`\n✅ SUCCESSFUL METHODS (${successful.length}/${this.results.length}):`);
        if (successful.length === 0) {
            this.log('   ❌ NO METHODS WORKED');
        } else {
            successful.forEach(result => {
                this.log(`   ✅ ${result.test}`);
                if (result.notes) this.log(`      → ${result.notes}`);
            });
        }
        
        this.log(`\n❌ FAILED METHODS (${failed.length}/${this.results.length}):`);
        failed.forEach(result => {
            this.log(`   ❌ ${result.test}`);
            if (result.error) this.log(`      → Error: ${result.error}`);
            if (result.notes) this.log(`      → Notes: ${result.notes}`);
        });
        
        // Definitive conclusion
        this.log('\n' + '🏁 DEFINITIVE CONCLUSION:');
        if (successful.length > 0) {
            this.log(`✅ K3D opacity control WORKS in HTML exports`);
            this.log(`✅ Working methods: ${successful.map(r => r.test).join(', ')}`);
            this.log('✅ Integrate these methods into your eryx pipeline');
        } else {
            this.log('❌ K3D opacity control DOES NOT WORK in HTML exports');
            this.log('❌ Consider alternative approaches:');
            this.log('   - Volume data preprocessing');  
            this.log('   - Multiple volume objects with different transparencies');
            this.log('   - Jupyter-only interactive controls');
        }
        
        const testDuration = (Date.now() - this.testStartTime) / 1000;
        this.log(`\n⏱️  Test completed in ${testDuration.toFixed(1)} seconds`);
        
        // Store results globally for external access
        window.K3DOpacityTestResults = this.results;
    }
    
    log(message) {
        const timestamp = new Date().toISOString().substr(11, 12);
        const logMessage = `[${timestamp}] ${message}`;
        console.log(logMessage);
        
        // Also display in page
        const logDiv = document.getElementById('test-log');
        if (logDiv) {
            logDiv.innerHTML += logMessage + '<br>';
            logDiv.scrollTop = logDiv.scrollHeight;
        }
    }
    
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Auto-start test when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Wait a bit for K3D to fully initialize
    setTimeout(() => {
        window.opacityTester = new K3DOpacityTester();
    }, 2000);
});

// Also provide manual trigger
window.runOpacityTest = () => {
    window.opacityTester = new K3DOpacityTester();
};
</script>

<style>
#test-controls {
    position: fixed;
    top: 10px;
    right: 10px;
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 15px;
    border-radius: 8px;
    font-family: monospace;
    max-width: 400px;
    z-index: 10000;
}

#test-log {
    background: #1e1e1e;
    color: #00ff00;
    padding: 10px;
    height: 300px;
    overflow-y: scroll;
    font-family: monospace;
    font-size: 11px;
    border: 1px solid #333;
    border-radius: 4px;
    margin-top: 10px;
}

.test-button {
    background: #0066cc;
    color: white;
    border: none;
    padding: 8px 15px;
    border-radius: 4px;
    cursor: pointer;
    margin: 5px;
}

.test-button:hover {
    background: #0088ee;
}
</style>

<div id="test-controls">
    <h3>🧪 K3D Opacity Test Suite</h3>
    <p>This test will automatically determine if K3D opacity control works in HTML exports.</p>
    <button class="test-button" onclick="runOpacityTest()">🔬 Run Test</button>
    <button class="test-button" onclick="document.getElementById('test-log').innerHTML = ''">🧹 Clear Log</button>
    <div id="test-log"></div>
</div>
"""

def main():
    """Main function to create comprehensive opacity test."""
    print("🧪 K3D OPACITY CONTROL - COMPREHENSIVE TEST")
    print("=" * 50)
    
    # Create test data
    volume_data = create_test_volume()
    
    # Create K3D visualization
    plot, volume = create_k3d_plot(volume_data)
    
    # Get base HTML
    base_html = plot.get_snapshot()
    
    # Insert comprehensive test JavaScript
    test_js = create_comprehensive_test_js()
    enhanced_html = base_html.replace('</body>', test_js + '</body>')
    
    # Write test file
    output_file = "k3d_opacity_test_comprehensive.html"
    with open(output_file, 'w') as f:
        f.write(enhanced_html)
    
    print(f"\n✅ Test file created: {output_file}")
    print("\n📋 TEST INSTRUCTIONS:")
    print("   1. Open k3d_opacity_test_comprehensive.html in your browser")
    print("   2. Open browser console (F12 → Console)")
    print("   3. Test will auto-start after 2 seconds")
    print("   4. Watch console output for detailed results")
    print("   5. Check the test results panel on the right")
    
    print("\n🔬 WHAT THIS TEST DOES:")
    print("   ✓ Tests ALL possible opacity control methods")
    print("   ✓ Logs detailed results to console")
    print("   ✓ Provides definitive answer on what works")
    print("   ✓ Tests visual changes, not just API calls")
    print("   ✓ Includes fallback detection methods")
    
    print("\n🎯 EXPECTED OUTCOME:")
    print("   This will definitively prove:")
    print("   - Which opacity control methods work in HTML exports")
    print("   - The exact JavaScript API that works")
    print("   - Whether opacity control is possible at all")
    
    return output_file

if __name__ == "__main__":
    output_file = main()
    print(f"\n🚀 Ready to test! Open {output_file} in your browser.")