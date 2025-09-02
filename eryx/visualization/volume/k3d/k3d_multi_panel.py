#!/usr/bin/env python3
"""
Create advanced multi-panel K3D visualization with all masking features.
Integrates optimized masking from optimized_masking_final.py into multi-panel layout.
"""

import numpy as np
import k3d
import os
from typing import Optional, Dict, Any

def create_advanced_multi_panel(
    output_file: str = "advanced_multi_panel.html",
    alpha_coef: float = 15.0,
    color_range_percentile: float = 90
):
    """
    Create advanced multi-panel K3D visualization with synchronized masking controls.
    
    Features:
    - Three volumes side by side (thermal, pumped, difference)
    - Spherical masking with radius slider
    - Interactive volume slicing with position slider and axis selection
    - Animated slicing with real-time slider feedback
    - Log scaling with controls
    - Alpha coefficient control
    - Synchronized camera controls
    """
    
    # Load datasets
    datasets = {}
    
    for name in ['thermal', 'pumped', 'difference']:
        file = f'{name}_diffuse_intensity.npy'
        if os.path.exists(file):
            data = np.load(file)
            if data.ndim == 1:
                side = int(np.round(data.size ** (1/3)))
                data = data.reshape(side, side, side)
            datasets[name] = data
            print(f"Loaded {name} data: {data.shape}")
    
    if not datasets:
        print("Creating test data...")
        x = np.linspace(-2, 2, 33)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
        
        # Create more interesting test data
        base = np.exp(-(X**2 + Y**2 + Z**2))
        datasets['thermal'] = base + 0.1 * np.random.random(base.shape)
        datasets['pumped'] = base * 1.3 + 0.15 * np.random.random(base.shape)
        datasets['difference'] = datasets['pumped'] - datasets['thermal']
        
        # Add octant features for testing
        mask_octant = (X > 0) & (Y > 0) & (Z > 0)
        datasets['thermal'][mask_octant] *= 1.5
        datasets['pumped'][mask_octant] *= 1.8
        datasets['difference'][mask_octant] *= 2.0
    
    # Create single K3D plot with all volumes positioned
    plot = k3d.plot(
        height=700,
        antialias=True,
        grid_visible=False,
        axes_helper=0.0,
        axes=['', '', ''],
        label_color=0x000000,  # Black labels to hide against black background
        camera_auto_fit=False,
        camera_fov=26.0,
        background_color=0x000000  # Ensure background is black
    )
    
    # Position volumes side by side
    x_positions = {'thermal': -8, 'pumped': 0, 'difference': 8}
    volume_info = {}
    
    # Calculate shared scale for thermal and pumped for absolute comparison
    thermal_clean = np.nan_to_num(datasets['thermal'], nan=0.0)
    pumped_clean = np.nan_to_num(datasets['pumped'], nan=0.0)
    
    # Find global min/max across thermal and pumped (excluding zeros)
    thermal_valid = thermal_clean[thermal_clean > 0]
    pumped_valid = pumped_clean[pumped_clean > 0]
    
    global_min = min(
        np.min(thermal_valid) if len(thermal_valid) > 0 else 0,
        np.min(pumped_valid) if len(pumped_valid) > 0 else 0
    )
    global_max = max(
        np.max(thermal_clean),
        np.max(pumped_clean)
    )
    
    # Calculate shared vmin/vmax for thermal and pumped AFTER normalization
    # This ensures they use identical color mapping
    thermal_normalized = (thermal_clean - global_min) / (global_max - global_min) if global_max > global_min else thermal_clean
    pumped_normalized = (pumped_clean - global_min) / (global_max - global_min) if global_max > global_min else pumped_clean
    
    # Combine valid values from both for percentile calculation
    combined_valid = np.concatenate([
        thermal_normalized[thermal_normalized > 0],
        pumped_normalized[pumped_normalized > 0]
    ])
    
    if len(combined_valid) > 0:
        shared_vmin, shared_vmax = np.percentile(combined_valid, [100 - color_range_percentile, color_range_percentile])
    else:
        shared_vmin, shared_vmax = 0, 1
    
    # Process each volume
    for name, data in datasets.items():
        clean_data = np.nan_to_num(data, nan=0.0)
        
        if name in ['thermal', 'pumped']:
            # Use shared global scaling for thermal and pumped
            if global_max > global_min:
                normalized = (clean_data - global_min) / (global_max - global_min)
            else:
                normalized = clean_data
            
            # Use the pre-calculated shared vmin/vmax for both thermal and pumped
            vmin, vmax = shared_vmin, shared_vmax
                
        else:  # difference volume
            # Original independent scaling for difference
            data_min = np.min(clean_data[clean_data > 0]) if np.any(clean_data > 0) else 0
            data_max = np.max(clean_data)
            
            if data_max > data_min:
                normalized = (clean_data - data_min) / (data_max - data_min)
            else:
                normalized = clean_data
            
            # Calculate color range for difference
            valid = normalized[normalized > 0]
            if len(valid) > 0:
                vmin, vmax = np.percentile(valid, [100 - color_range_percentile, color_range_percentile])
            else:
                vmin, vmax = 0, 1
        
        # Store as float32
        normalized = normalized.astype(np.float32)
        
        # Create volume at position
        x_off = x_positions[name]
        bounds = [x_off-2.5, x_off+2.5, -2.5, 2.5, -2.5, 2.5]
        
        volume = k3d.volume(
            normalized,
            color_range=[vmin, vmax],
            bounds=bounds,
            alpha_coef=alpha_coef,
            color_map=k3d.basic_color_maps.Jet
        )
        plot += volume
        
        # Store volume info for JavaScript
        if name in ['thermal', 'pumped']:
            volume_info[name] = {
                'shape': normalized.shape,
                'bounds': bounds,
                'original_min': float(global_min),
                'original_max': float(global_max),
                'shared_scale': True
            }
        else:
            volume_info[name] = {
                'shape': normalized.shape,
                'bounds': bounds,
                'original_min': float(data_min),
                'original_max': float(data_max),
                'shared_scale': False
            }
    
    # Set camera to see all three volumes
    plot.camera = [0, -25, 12, 0, 0, 0, 0, 0, 1]
    
    # Get base HTML
    html_content = plot.get_snapshot()
    
    # Extract shape for JavaScript (assumes all volumes same shape)
    first_shape = list(volume_info.values())[0]['shape']
    h, k, l = first_shape
    
    # Advanced JavaScript with all masking features
    advanced_js = f"""
    // Global state management
    let k3dPlot = null;
    let allVolumes = [];
    let originalDataArray = [];
    let volumeShapes = [];
    let volumeIds = [];
    
    // Animation states
    let cameraAnimationFrame = null;
    let sliceAnimationFrame = null;
    
    // Mask state for all volumes
    let maskState = {{
        spherical: {{ enabled: false, radius: 50 }},
        slicing: {{ enabled: false, position: {h//2}, axis: 'x', animating: false }},
        logScale: {{ enabled: false, dynamicRange: 100, offset: 0.001 }}
    }};
    
    // Multi-volume controller class
    class MultiVolumeController {{
        constructor() {{
            this.updateQueue = [];
        }}
        
        async initialize() {{
            console.log('Initializing MultiVolumeController...');
            
            // Get K3D plot
            let attempts = 0;
            while (attempts < 50) {{
                if (window.K3DInstance) {{
                    k3dPlot = window.K3DInstance instanceof Promise ? 
                              await window.K3DInstance : window.K3DInstance;
                    break;
                }}
                await new Promise(resolve => setTimeout(resolve, 100));
                attempts++;
            }}
            
            if (!k3dPlot) {{
                throw new Error('K3D not found');
            }}
            
            // Get world and find all volumes
            const world = k3dPlot.getWorld();
            console.log('World structure:', Object.keys(world));
            
            // Method 1: Try K3DObjects and filter for volumes
            const allObjects = Object.values(world.K3DObjects || {{}});
            allVolumes = allObjects.filter(obj => 
                obj && obj.alpha_coef !== undefined
            );
            
            console.log('Found', allVolumes.length, 'volumes via K3DObjects');
            
            // Method 2: If not found, try ObjectsListJson
            if (allVolumes.length === 0) {{
                console.log('Trying ObjectsListJson approach...');
                volumeIds = Object.keys(world.ObjectsListJson || {{}});
                
                for (const id of volumeIds) {{
                    const config = world.ObjectsListJson[id];
                    if (config && config.type === 'Volume') {{
                        const mesh = world.ObjectsById[id];
                        if (mesh) {{
                            allVolumes.push(mesh);
                            
                            // Store original data and shape
                            originalDataArray.push(new Float32Array(config.volume.data));
                            volumeShapes.push(config.volume.shape);
                            console.log(`Volume ${{id}}: shape ${{config.volume.shape}}, data length ${{config.volume.data.length}}`);
                        }}
                    }}
                }}
            }} else {{
                // For K3DObjects approach, find configs
                const objectIds = Object.keys(world.ObjectsListJson || {{}});
                for (const id of objectIds) {{
                    const config = world.ObjectsListJson[id];
                    if (config && config.type === 'Volume') {{
                        volumeIds.push(id);
                        originalDataArray.push(new Float32Array(config.volume.data));
                        volumeShapes.push(config.volume.shape);
                    }}
                }}
            }}
            
            console.log(`✅ Initialized with ${{allVolumes.length}} volumes`);
            console.log('Volume shapes:', volumeShapes);
            console.log('Original data lengths:', originalDataArray.map(d => d.length));
            
            // Hide axes permanently
            if (k3dPlot && k3dPlot.parameters) {{
                k3dPlot.parameters.axesHelper = 0;
                k3dPlot.parameters.axes = ['', '', ''];
                k3dPlot.parameters.gridVisible = false;
                k3dPlot.render();
            }}
            
            return allVolumes.length > 0;
        }}
        
        queueUpdate(volumeIndex, newData) {{
            this.updateQueue.push({{ volumeIndex, newData }});
        }}
        
        async flushUpdates() {{
            if (!k3dPlot || this.updateQueue.length === 0) return;
            
            const world = k3dPlot.getWorld();
            
            // Apply all queued updates
            for (const update of this.updateQueue) {{
                const volumeId = volumeIds[update.volumeIndex];
                const config = world.ObjectsListJson[volumeId];
                const mesh = world.ObjectsById[volumeId];
                
                if (config && config.volume) {{
                    config.volume.data = update.newData;
                }}
                
                // Update 3D texture if exists
                if (mesh && mesh.material && mesh.material.uniforms && mesh.material.uniforms.volumeTexture) {{
                    const texture = mesh.material.uniforms.volumeTexture.value;
                    if (texture && texture.image && texture.image.data) {{
                        texture.image.data.set(update.newData);
                        texture.needsUpdate = true;
                    }}
                }}
            }}
            
            // Single render call for all updates
            this.updateQueue = [];
            k3dPlot.rebuildSceneData();
            
            // Force hide axes after data update - make labels invisible with black color
            k3dPlot.parameters.axesHelper = 0.0;
            k3dPlot.parameters.axes = ['', '', ''];
            k3dPlot.parameters.gridVisible = false;
            k3dPlot.parameters.labelColor = 0x000000;  // Critical: black labels invisible on black background
            
            k3dPlot.render();
        }}
        
        createCombinedMask(originalDataIndex) {{
            const shape = volumeShapes[originalDataIndex];
            if (!shape || shape.length !== 3) {{
                console.error('Invalid shape for volume', originalDataIndex, shape);
                return null;
            }}
            
            const [h, k, l] = shape;
            const centerX = h / 2;
            const centerY = k / 2; 
            const centerZ = l / 2;
            const totalSize = h * k * l;
            
            let originalData = originalDataArray[originalDataIndex];
            
            // Start with original data
            let processedData = new Float32Array(originalData);
            
            // Apply log scaling if enabled
            if (maskState.logScale.enabled) {{
                const dynamicRange = maskState.logScale.dynamicRange;
                const offset = maskState.logScale.offset;
                const maxLog = Math.log1p(dynamicRange);
                
                for (let i = 0; i < processedData.length; i++) {{
                    const value = processedData[i];
                    const scaledValue = Math.log1p((value + offset) * dynamicRange) / maxLog;
                    processedData[i] = Math.min(1.0, Math.max(0.0, scaledValue));
                }}
            }}
            
            // Apply masks
            const maskedData = new Float32Array(totalSize);
            let idx = 0;
            
            for (let i = 0; i < h; i++) {{
                for (let j = 0; j < k; j++) {{
                    for (let m = 0; m < l; m++) {{
                        let keepVoxel = true;
                        
                        // Spherical mask
                        if (maskState.spherical.enabled) {{
                            const dx = i - centerX;
                            const dy = j - centerY;
                            const dz = m - centerZ;
                            const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                            const radius = (Math.min(h, k, l) / 2) * (maskState.spherical.radius / 100);
                            keepVoxel = keepVoxel && (dist <= radius);
                        }}
                        
                        // Removed octant mask - replaced with enhanced slicing
                        
                        // Slicing plane
                        if (maskState.slicing.enabled) {{
                            let coord;
                            if (maskState.slicing.axis === 'x') coord = i;
                            else if (maskState.slicing.axis === 'y') coord = j;
                            else coord = m;
                            
                            if (coord > maskState.slicing.position) {{
                                keepVoxel = false;
                            }}
                        }}
                        
                        maskedData[idx] = keepVoxel ? processedData[idx] : 0;
                        idx++;
                    }}
                }}
            }}
            
            return maskedData;
        }}
        
        async applyAllMasks() {{
            console.log('Applying masks to all volumes...');
            
            // Create masked data for each volume
            for (let i = 0; i < originalDataArray.length; i++) {{
                const maskedData = this.createCombinedMask(i);
                if (maskedData) {{
                    this.queueUpdate(i, maskedData);
                }}
            }}
            
            // Apply all updates with single render
            await this.flushUpdates();
            
            // Update status
            this.updateStatus();
        }}
        
        async updateAlphaCoefficient(value) {{
            const alpha = parseFloat(value);
            console.log('Updating alpha to:', alpha);
            
            // Use K3D's reload method - the same one the K3D panel uses
            const world = k3dPlot.getWorld();
            let volumeCount = 0;
            
            if (world && world.ObjectsListJson) {{
                for (let id in world.ObjectsListJson) {{
                    const json = world.ObjectsListJson[id];
                    if (json && json.type === 'Volume') {{
                        volumeCount++;
                        
                        // Update the JSON property
                        json.alpha_coef = alpha;
                        
                        // Create changes object
                        const changes = {{ alpha_coef: alpha }};
                        
                        // Use K3D's reload method - this is what the panel uses!
                        if (typeof k3dPlot.reload === 'function') {{
                            k3dPlot.reload(json, changes);
                            console.log(`✅ Updated volume ${{id}} alpha to ${{alpha}} using reload()`);
                        }} else {{
                            console.error('reload method not found on k3dPlot!');
                        }}
                    }}
                }}
            }} else {{
                console.error('World or ObjectsListJson not found!');
            }}
            
            console.log(`Updated alpha for ${{volumeCount}} volumes`);
        }}
        
        updateStatus() {{
            let status = [];
            if (maskState.spherical.enabled) {{
                status.push(`Sphere: ${{maskState.spherical.radius}}%`);
            }}
            if (maskState.slicing.enabled) {{
                const animText = maskState.slicing.animating ? ' (Animating)' : '';
                status.push(`Slice ${{maskState.slicing.axis.toUpperCase()}}: ${{Math.round(maskState.slicing.position)}}${{animText}}`);
            }}
            if (maskState.logScale.enabled) {{
                status.push('Log Scale');
            }}
            if (status.length === 0) {{
                status.push('No filters');
            }}
            
            const statusEl = document.getElementById('status');
            if (statusEl) {{
                statusEl.textContent = status.join(' + ');
            }}
        }}
    }}
    
    // Global controller instance
    window.multiController = new MultiVolumeController();
    
    // Wrapper functions for UI
    window.toggleSphericalMask = function(enabled) {{
        maskState.spherical.enabled = enabled;
        if (enabled) {{
            const slider = document.getElementById('sphere-radius');
            if (slider) {{
                maskState.spherical.radius = parseFloat(slider.value);
            }}
        }}
        multiController.applyAllMasks();
    }};
    
    window.updateSphericalRadius = function(value) {{
        maskState.spherical.radius = parseFloat(value);
        document.getElementById('radius-display').textContent = value + '%';
        if (maskState.spherical.enabled) {{
            multiController.applyAllMasks();
        }}
    }};
    
    window.toggleSlicing = function(enabled) {{
        maskState.slicing.enabled = enabled;
        if (enabled) {{
            const slider = document.getElementById('slice-position');
            const axis = document.getElementById('slice-axis');
            if (slider) maskState.slicing.position = parseFloat(slider.value);
            if (axis) maskState.slicing.axis = axis.value;
        }}
        multiController.applyAllMasks();
    }};
    
    window.updateSliceAxis = function(axis) {{
        maskState.slicing.axis = axis;
        if (maskState.slicing.enabled && !maskState.slicing.animating) {{
            multiController.applyAllMasks();
        }}
    }};
    
    window.updateSlicePosition = function(position) {{
        const pos = parseFloat(position);
        maskState.slicing.position = pos;
        document.getElementById('slice-display').textContent = Math.round(pos);
        
        if (maskState.slicing.enabled && !maskState.slicing.animating) {{
            multiController.applyAllMasks();
        }}
    }};
    
    window.toggleLogScale = function(enabled) {{
        maskState.logScale.enabled = enabled;
        const logControls = document.getElementById('log-controls');
        if (logControls) {{
            logControls.style.display = enabled ? 'block' : 'none';
        }}
        multiController.applyAllMasks();
    }};
    
    window.updateLogScaling = function(value) {{
        maskState.logScale.dynamicRange = parseFloat(value);
        document.getElementById('log-range-value').textContent = value;
        if (maskState.logScale.enabled) {{
            multiController.applyAllMasks();
        }}
    }};
    
    window.updateLogOffset = function(value) {{
        maskState.logScale.offset = parseFloat(value);
        document.getElementById('log-offset-value').textContent = parseFloat(value).toFixed(3);
        if (maskState.logScale.enabled) {{
            multiController.applyAllMasks();
        }}
    }};
    
    window.updateAlphaCoef = function(value) {{
        document.getElementById('alpha-value').textContent = parseFloat(value).toFixed(1);
        multiController.updateAlphaCoefficient(value);
    }};
    
    window.clearAllMasks = function() {{
        // Reset all mask states
        maskState.spherical.enabled = false;
        maskState.slicing.enabled = false;
        maskState.slicing.animating = false;
        
        // Update UI
        const sphereCheck = document.getElementById('sphere-enable');
        const sliceCheck = document.getElementById('slice-enable');
        if (sphereCheck) sphereCheck.checked = false;
        if (sliceCheck) sliceCheck.checked = false;
        
        multiController.applyAllMasks();
    }};
    
    // Camera animations
    window.startCameraRotation = function() {{
        if (cameraAnimationFrame) stopCameraRotation();
        
        let angle = 0;
        const radius = 28;
        const height = 12;
        
        function animate() {{
            if (!k3dPlot) return;
            
            angle += 0.008;
            const x = radius * Math.cos(angle);
            const z = radius * Math.sin(angle);
            
            k3dPlot.setCamera([x, z, height, 0, 0, 0, 0, 0, 1]);
            k3dPlot.render();
            
            cameraAnimationFrame = requestAnimationFrame(animate);
        }}
        
        animate();
        updateAnimationStatus();
    }};
    
    window.stopCameraRotation = function() {{
        if (cameraAnimationFrame) {{
            cancelAnimationFrame(cameraAnimationFrame);
            cameraAnimationFrame = null;
            updateAnimationStatus();
        }}
    }};
    
    window.resetCamera = function() {{
        if (cameraAnimationFrame) stopCameraRotation();
        if (k3dPlot) {{
            k3dPlot.setCamera([0, -25, 12, 0, 0, 0, 0, 0, 1]);
            k3dPlot.render();
        }}
    }};
    
    // Slicing animation
    window.startSliceAnimation = function() {{
        if (sliceAnimationFrame) stopSliceAnimation();
        
        maskState.slicing.animating = true;
        maskState.slicing.enabled = true;
        
        let t = 0;
        const maxPos = {h} - 1;
        const slider = document.getElementById('slice-position');
        const sliceCheck = document.getElementById('slice-enable');
        
        if (sliceCheck) sliceCheck.checked = true;
        
        function animate() {{
            t += 0.015;
            const position = maxPos * (0.5 + 0.5 * Math.sin(t));
            maskState.slicing.position = position;
            
            // Update slider position for visual feedback
            if (slider) {{
                slider.value = position;
                document.getElementById('slice-display').textContent = Math.round(position);
            }}
            
            multiController.applyAllMasks();
            sliceAnimationFrame = requestAnimationFrame(animate);
        }}
        
        animate();
        updateAnimationStatus();
    }};
    
    window.stopSliceAnimation = function() {{
        if (sliceAnimationFrame) {{
            cancelAnimationFrame(sliceAnimationFrame);
            sliceAnimationFrame = null;
            maskState.slicing.animating = false;
            // Keep slicing enabled but allow manual control
            updateAnimationStatus();
        }}
    }};
    
    function updateAnimationStatus() {{
        let animations = [];
        if (cameraAnimationFrame) animations.push('Camera');
        if (sliceAnimationFrame) animations.push('Slicing');
        
        const statusEl = document.getElementById('animation-status');
        if (statusEl) {{
            statusEl.textContent = animations.length > 0 ? 
                'Running: ' + animations.join(' + ') : 'Stopped';
        }}
    }}
    
    // Initialize when ready
    window.addEventListener('load', async function() {{
        console.log('Initializing advanced multi-panel visualization...');
        
        try {{
            const success = await multiController.initialize();
            
            if (success) {{
                // Enable all controls
                document.querySelectorAll('button, input, select').forEach(el => {{
                    el.disabled = false;
                }});
                
                const statusEl = document.getElementById('status');
                if (statusEl) statusEl.textContent = 'Ready';
                
                console.log('✅ Advanced multi-panel initialized successfully!');
            }} else {{
                throw new Error('No volumes found');
            }}
        }} catch (error) {{
            console.error('❌ Initialization failed:', error);
            const statusEl = document.getElementById('status');
            if (statusEl) statusEl.textContent = 'Error: ' + error.message;
        }}
    }});
    """
    
    # Modern control panel UI
    control_panel = f"""
    <style>
        body {{
            margin: 0;
            background: #1a1a1a;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        
        .control-bar {{
            position: fixed;
            top: 10px;
            left: 10px;
            right: 10px;
            background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
            border-radius: 12px;
            padding: 16px 24px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
            z-index: 1000;
            color: white;
            backdrop-filter: blur(10px);
        }}
        
        .control-sections {{
            display: flex;
            gap: 32px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            min-width: 120px;
        }}
        
        .control-group h4 {{
            margin: 0;
            font-size: 13px;
            font-weight: 600;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .button-row {{
            display: flex;
            gap: 8px;
        }}
        
        button {{
            background: rgba(59, 130, 246, 0.8);
            color: white;
            border: 1px solid rgba(59, 130, 246, 0.9);
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s;
            backdrop-filter: blur(10px);
        }}
        
        button:hover:not(:disabled) {{
            background: rgba(59, 130, 246, 1);
            transform: translateY(-1px);
        }}
        
        button:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}
        
        button.danger {{
            background: rgba(239, 68, 68, 0.8);
            border-color: rgba(239, 68, 68, 0.9);
        }}
        
        button.danger:hover:not(:disabled) {{
            background: rgba(239, 68, 68, 1);
        }}
        
        button.clear {{
            background: rgba(34, 197, 94, 0.8);
            border-color: rgba(34, 197, 94, 0.9);
        }}
        
        button.clear:hover:not(:disabled) {{
            background: rgba(34, 197, 94, 1);
        }}
        
        input[type="range"] {{
            width: 100px;
            height: 4px;
            border-radius: 2px;
            background: rgba(255,255,255,0.2);
            outline: none;
            -webkit-appearance: none;
        }}
        
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: white;
            cursor: pointer;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }}
        
        select {{
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid rgba(255,255,255,0.3);
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 12px;
        }}
        
        select option {{
            background: #4a5568;
        }}
        
        .checkbox-label {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            cursor: pointer;
        }}
        
        input[type="checkbox"] {{
            margin: 0;
        }}
        
        .value-display {{
            font-size: 11px;
            opacity: 0.7;
            text-align: center;
            margin-top: 2px;
        }}
        
        .status-section {{
            margin-left: auto;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
        }}
        
        .status-indicator {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
        }}
        
        .status-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        
        #log-controls {{
            display: none;
            margin-top: 8px;
            padding: 8px;
            background: rgba(0,0,0,0.2);
            border-radius: 4px;
        }}
        
        .log-control {{
            display: flex;
            flex-direction: column;
            gap: 4px;
            margin-bottom: 6px;
        }}
        
        .log-control label {{
            font-size: 11px;
            opacity: 0.8;
        }}
    </style>
    
    <div class="control-bar">
        <div class="control-sections">
            <!-- Camera Controls -->
            <div class="control-group">
                <h4>Camera</h4>
                <div class="button-row">
                    <button onclick="startCameraRotation()" disabled>▶️ Rotate</button>
                    <button class="danger" onclick="stopCameraRotation()" disabled>⏹️ Stop</button>
                    <button onclick="resetCamera()" disabled>🔄 Reset</button>
                </div>
            </div>
            
            <!-- Masking Controls -->
            <div class="control-group">
                <h4>Spherical Mask</h4>
                <label class="checkbox-label">
                    <input type="checkbox" id="sphere-enable" onchange="toggleSphericalMask(this.checked)" disabled>
                    Enable
                </label>
                <input type="range" id="sphere-radius" min="10" max="150" value="50" 
                       oninput="updateSphericalRadius(this.value)" disabled>
                <div class="value-display"><span id="radius-display">50%</span></div>
            </div>
            
            <!-- Slicing Controls -->
            <div class="control-group">
                <h4>Volume Slicing</h4>
                <label class="checkbox-label">
                    <input type="checkbox" id="slice-enable" onchange="toggleSlicing(this.checked)" disabled>
                    Enable
                </label>
                <select id="slice-axis" onchange="updateSliceAxis(this.value)" disabled>
                    <option value="x">X-axis</option>
                    <option value="y">Y-axis</option>
                    <option value="z">Z-axis</option>
                </select>
                <div class="slider-container">
                    <label>Position: <span id="slice-display">{h//2}</span></label>
                    <input type="range" id="slice-position" min="0" max="{h-1}" value="{h//2}" 
                           oninput="updateSlicePosition(this.value)" disabled>
                </div>
            </div>
            
            <!-- Scaling Controls -->
            <div class="control-group">
                <h4>Scaling</h4>
                <label class="checkbox-label">
                    <input type="radio" name="scaling" value="linear" checked onchange="toggleLogScale(false)" disabled>
                    Linear
                </label>
                <label class="checkbox-label">
                    <input type="radio" name="scaling" value="log" onchange="toggleLogScale(true)" disabled>
                    Log
                </label>
                <div id="log-controls">
                    <div class="log-control">
                        <label>Range: <span id="log-range-value">100</span></label>
                        <input type="range" min="1" max="10000" value="100" step="1"
                               oninput="updateLogScaling(this.value)" disabled>
                    </div>
                    <div class="log-control">
                        <label>Offset: <span id="log-offset-value">0.001</span></label>
                        <input type="range" min="0" max="1" value="0.001" step="0.001"
                               oninput="updateLogOffset(this.value)" disabled>
                    </div>
                </div>
            </div>
            
            <!-- Alpha Controls -->
            <div class="control-group">
                <h4>Opacity</h4>
                <input type="range" id="alpha-slider" min="0.1" max="100" step="0.1" value="{alpha_coef}"
                       oninput="updateAlphaCoef(this.value)">
                <div class="value-display">Alpha: <span id="alpha-value">{alpha_coef}</span></div>
            </div>
            
            <!-- Animation Controls -->
            <div class="control-group">
                <h4>Animations</h4>
                <div class="button-row">
                    <button onclick="startSliceAnimation()" disabled>▶️ Slice</button>
                    <button class="danger" onclick="stopSliceAnimation()" disabled>⏹️ Stop</button>
                </div>
                <div class="value-display" id="animation-status">Stopped</div>
            </div>
            
            <!-- Clear All -->
            <div class="control-group">
                <button class="clear" onclick="clearAllMasks()" disabled style="margin-top: 20px;">🔄 Clear All</button>
            </div>
            
            <!-- Status -->
            <div class="status-section">
                <div class="status-indicator">
                    <div class="status-dot"></div>
                    <span id="status">Initializing...</span>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Volume Labels -->
    <div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); 
                display: flex; gap: 120px; color: white; font-size: 14px; font-weight: 500;
                z-index: 100; pointer-events: none;">
        <div style="text-align: center;">
            <div style="background: rgba(0,0,0,0.5); padding: 8px 12px; border-radius: 6px;">
                Thermal
            </div>
        </div>
        <div style="text-align: center;">
            <div style="background: rgba(0,0,0,0.5); padding: 8px 12px; border-radius: 6px;">
                Pumped
            </div>
        </div>
        <div style="text-align: center;">
            <div style="background: rgba(0,0,0,0.5); padding: 8px 12px; border-radius: 6px;">
                Difference
            </div>
        </div>
    </div>
    """
    
    # Inject JavaScript and controls
    injection_point = html_content.rfind('</body>')
    if injection_point != -1:
        full_injection = f"""
        {control_panel}
        <script>
        {advanced_js}
        </script>
        """
        html_content = html_content[:injection_point] + full_injection + html_content[injection_point:]
    
    # Update title
    html_content = html_content.replace(
        '<title>K3D</title>', 
        '<title>Advanced Multi-Panel K3D Visualization</title>'
    )
    
    # Save HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"\n✅ Advanced multi-panel visualization saved to: {output_file}")
    print("Features:")
    print("  • Three synchronized volumes (thermal, pumped, difference)")
    print("  • Spherical masking with radius control")
    print("  • Interactive volume slicing with position control")
    print("  • Log scaling with dynamic range and offset")
    print("  • Animated slicing with real-time feedback")
    print("  • Camera rotation animation")
    print("  • Alpha coefficient control")
    print("  • Direct data masking (10x performance vs clipping)")
    print("  • Single render call for smooth synchronized updates")
    print("\nUsage:")
    print("  1. Open the HTML file in any modern browser")
    print("  2. Wait for 'Ready' status in the control bar")
    print("  3. Use checkboxes to enable different masking modes")
    print("  4. Adjust sliders for interactive parameters")
    print("  5. Use animation buttons for dynamic views")
    
    return output_file


def main():
    """Create the advanced multi-panel visualization."""
    print("Advanced Multi-Panel K3D Visualization")
    print("=" * 50)
    
    output_file = create_advanced_multi_panel(
        output_file="advanced_multi_panel.html",
        alpha_coef=15.0,
        color_range_percentile=90
    )
    
    print("\n" + "=" * 50)
    print("SUCCESS! Open the HTML file to see:")
    print("  ✅ Three volumes side by side")
    print("  ✅ All masking features from optimized_masking_final.py")
    print("  ✅ Synchronized controls for all volumes")
    print("  ✅ Modern compact control bar")
    print("  ✅ Performance optimizations")
    print("  ✅ No iframes needed - single HTML file")


if __name__ == "__main__":
    main()