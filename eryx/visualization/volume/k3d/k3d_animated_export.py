"""
K3D Animated HTML Export Module

This module generates standalone HTML files with embedded K3D animations
that work without requiring console interaction.
"""

import k3d
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import json


def create_animated_html(
    volume_data: np.ndarray,
    bounds: Optional[List[float]] = None,
    animation_type: str = "orbit",
    animation_config: Optional[Dict[str, Any]] = None,
    title: str = "K3D Animated Visualization",
    color_range_percentile: float = 95,
    alpha_coef: float = 20.0,
    colormap: str = "Jet"
) -> str:
    """
    Create a standalone HTML file with embedded K3D animations.
    
    Parameters
    ----------
    volume_data : np.ndarray
        3D volume data to visualize
    bounds : List[float], optional
        Bounds for the volume [xmin, xmax, ymin, ymax, zmin, zmax]
    animation_type : str
        Type of animation: "orbit", "sweep", "zoom", "combined", "custom"
    animation_config : Dict[str, Any], optional
        Configuration for the animation
    title : str
        Title for the HTML page
    color_range_percentile : float
        Percentile for color range adjustment
    alpha_coef : float
        Alpha coefficient for transparency
    colormap : str
        Name of the colormap to use
        
    Returns
    -------
    str
        Complete HTML content with embedded animations
    """
    # Create K3D plot
    plot = k3d.plot(
        height=800,
        antialias=True,
        grid_visible=False,
        menu_visibility=True,
        axes_helper=1.0
    )
    
    # Set bounds
    if bounds is None:
        bounds = [-2, 2, -2, 2, -2, 2]
    plot.grid = bounds
    
    # Handle NaN values and adjust color range
    volume_clean = np.nan_to_num(volume_data, nan=0.0)
    vmin, vmax = np.percentile(
        volume_clean[volume_clean > 0], 
        [100 - color_range_percentile, color_range_percentile]
    )
    
    # Add volume to plot
    volume_obj = k3d.volume(
        volume_clean.astype(np.float32),
        color_range=[vmin, vmax],
        bounds=bounds,
        alpha_coef=alpha_coef,
        color_map=getattr(k3d.basic_color_maps, colormap, k3d.basic_color_maps.Jet)
    )
    plot += volume_obj
    
    # Get base HTML snapshot
    html_content = plot.get_snapshot()
    
    # Generate animation JavaScript based on type
    animation_js = _generate_animation_script(animation_type, animation_config)
    
    # Generate control panel HTML
    control_panel_html = _generate_control_panel(animation_type)
    
    # Inject animation script and controls into HTML
    # Find the closing body tag and inject before it
    injection_point = html_content.rfind('</body>')
    
    if injection_point != -1:
        # CSS for control panel
        css_injection = """
        <style>
            .k3d-animation-controls {
                position: fixed;
                top: 10px;
                right: 10px;
                background: rgba(255, 255, 255, 0.95);
                border: 1px solid #ccc;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                z-index: 1000;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                min-width: 200px;
            }
            .k3d-animation-controls h3 {
                margin: 0 0 10px 0;
                font-size: 14px;
                color: #333;
            }
            .k3d-animation-controls button {
                background: #4CAF50;
                color: white;
                border: none;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                margin: 2px;
                font-size: 12px;
                transition: background 0.3s;
            }
            .k3d-animation-controls button:hover {
                background: #45a049;
            }
            .k3d-animation-controls button.stop {
                background: #f44336;
            }
            .k3d-animation-controls button.stop:hover {
                background: #da190b;
            }
            .k3d-animation-controls button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            .k3d-animation-controls .control-group {
                margin: 10px 0;
            }
            .k3d-animation-controls label {
                display: block;
                font-size: 12px;
                margin: 5px 0;
                color: #666;
            }
            .k3d-animation-controls input[type="range"] {
                width: 100%;
                margin: 5px 0;
            }
            .k3d-animation-controls .value-display {
                font-size: 11px;
                color: #999;
                text-align: right;
            }
        </style>
        """
        
        # Complete injection with CSS, controls, and animation script
        full_injection = f"""
        {css_injection}
        {control_panel_html}
        <script>
        // Wait for K3D to be fully loaded
        window.addEventListener('load', async function() {{
            console.log('Initializing K3D animations...');
            
            // Try to find K3D instance
            let k3d = null;
            let attempts = 0;
            const maxAttempts = 50;
            
            async function findK3D() {{
                // Check for K3DInstance (Promise-based)
                if (typeof window.K3DInstance !== 'undefined') {{
                    try {{
                        k3d = await window.K3DInstance;
                        console.log('✅ K3D instance found and resolved');
                        return true;
                    }} catch (e) {{
                        console.error('Failed to resolve K3DInstance:', e);
                    }}
                }}
                
                // Check for other common K3D references
                const candidates = [window.k3d, window.plot, window.K3D];
                for (const candidate of candidates) {{
                    if (candidate && candidate.setCamera) {{
                        k3d = candidate;
                        console.log('✅ K3D instance found');
                        return true;
                    }}
                }}
                
                return false;
            }}
            
            // Retry finding K3D with delay
            while (attempts < maxAttempts && !await findK3D()) {{
                attempts++;
                await new Promise(resolve => setTimeout(resolve, 100));
            }}
            
            if (!k3d) {{
                console.error('❌ Could not find K3D instance after ' + attempts + ' attempts');
                document.getElementById('animation-status').textContent = 'K3D not found';
                return;
            }}
            
            // Store globally for animation functions
            window.k3d = k3d;
            
            {animation_js}
            
            // Initialize animations
            if (typeof initializeAnimations === 'function') {{
                initializeAnimations();
                console.log('✅ Animations initialized');
            }}
        }});
        </script>
        """
        
        html_content = html_content[:injection_point] + full_injection + html_content[injection_point:]
    
    # Update title
    html_content = html_content.replace('<title>K3D</title>', f'<title>{title}</title>')
    
    return html_content


def _generate_animation_script(animation_type: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Generate JavaScript animation code based on type."""
    
    if config is None:
        config = {}
    
    # Base animation functions that will be available for all types
    base_functions = """
    // Global animation state
    let currentAnimation = null;
    
    // Make all functions global
    window.stopCurrentAnimation = function() {
        if (currentAnimation) {
            currentAnimation.stop();
            currentAnimation = null;
            updateButtonStates();
        }
    }
    
    // Update button states
    window.updateButtonStates = function() {
        const buttons = document.querySelectorAll('.k3d-animation-controls button');
        buttons.forEach(btn => {
            if (btn.classList.contains('stop')) {
                btn.disabled = !currentAnimation;
            } else {
                btn.disabled = !!currentAnimation;
            }
        });
        
        const status = document.getElementById('animation-status');
        if (status) {
            status.textContent = currentAnimation ? 'Running' : 'Stopped';
        }
    }
    """
    
    # Animation implementations based on type
    animations = {
        "masking": """
        window.startMaskingAnimation = function() {
            stopCurrentAnimation();
            
            const enableOctant = document.getElementById('enable-octant')?.checked || false;
            const enableSphere = document.getElementById('enable-sphere')?.checked || false;
            const sphereRadius = parseFloat(document.getElementById('sphere-radius')?.value || 1.5);
            const octantSelection = document.getElementById('octant-selection')?.value || '+++';
            const animateRadius = document.getElementById('animate-radius')?.checked || false;
            const rotateCamera = document.getElementById('rotate-camera')?.checked !== false; // Default true
            
            let t = 0;
            let cameraAngle = 0;
            let animationId;
            let stopped = false;
            
            function update() {
                if (stopped) return;
                
                // Rotate camera if enabled
                if (rotateCamera) {
                    cameraAngle += 0.01;
                    const radius = 6;
                    const x = radius * Math.cos(cameraAngle);
                    const z = radius * Math.sin(cameraAngle);
                    k3d.setCamera([x, 2, z, 0, 0, 0, 0, 1, 0]);
                }
                
                const planes = [];
                
                // Apply octant masking if enabled
                if (enableOctant) {
                    // Octant masking - keep only 1/8 of space
                    // We need to remove 7/8, so we add planes that cut away other octants
                    const signs = octantSelection.split('');
                    
                    // To keep only one octant, we need 6 planes (2 per axis)
                    // to cut away the other 7 octants
                    
                    // For X axis
                    if (signs[0] === '+') {
                        // Keep x > 0, so add plane cutting x < 0
                        planes.push([-1, 0, 0, 0]);  // Remove x < 0
                    } else {
                        // Keep x < 0, so add plane cutting x > 0
                        planes.push([1, 0, 0, 0]);   // Remove x > 0
                    }
                    
                    // For Y axis
                    if (signs[1] === '+') {
                        // Keep y > 0, so add plane cutting y < 0
                        planes.push([0, -1, 0, 0]);  // Remove y < 0
                    } else {
                        // Keep y < 0, so add plane cutting y > 0
                        planes.push([0, 1, 0, 0]);   // Remove y > 0
                    }
                    
                    // For Z axis
                    if (signs[2] === '+') {
                        // Keep z > 0, so add plane cutting z < 0
                        planes.push([0, 0, -1, 0]);  // Remove z < 0
                    } else {
                        // Keep z < 0, so add plane cutting z > 0
                        planes.push([0, 0, 1, 0]);   // Remove z > 0
                    }
                }
                
                // Apply spherical masking if enabled (can be combined with octant)
                if (enableSphere) {
                    // Spherical masking - keep only points within radius
                    // We need planes pointing INWARD to keep the inside
                    const numPlanes = 12;  // More planes = better sphere approximation
                    let currentRadius = sphereRadius;
                    
                    if (animateRadius) {
                        // Animate the radius
                        t += 0.02;
                        currentRadius = sphereRadius * (0.5 + 0.5 * Math.sin(t));
                    }
                    
                    // Create planes tangent to sphere pointing inward
                    for (let i = 0; i < numPlanes; i++) {
                        const theta = (i / numPlanes) * 2 * Math.PI;
                        for (let j = 1; j < numPlanes/2; j++) {
                            const phi = (j / (numPlanes/2)) * Math.PI;
                            
                            // Normal vector pointing INWARD (negative of outward)
                            const nx = -Math.sin(phi) * Math.cos(theta);
                            const ny = -Math.sin(phi) * Math.sin(theta);
                            const nz = -Math.cos(phi);
                            
                            // Plane at distance = radius from origin
                            // The plane equation is nx*x + ny*y + nz*z + d > 0
                            // For inward pointing normal at radius R: d = R
                            planes.push([nx, ny, nz, currentRadius]);
                        }
                    }
                    
                    // Add top and bottom caps
                    planes.push([0, 0, -1, currentRadius]);  // Top cap
                    planes.push([0, 0, 1, currentRadius]);   // Bottom cap
                }
                
                k3d.setClippingPlanes(planes);
                if (k3d.render) k3d.render();
                
                animationId = requestAnimationFrame(update);
            }
            
            update();
            
            currentAnimation = {
                stop: () => {
                    stopped = true;
                    cancelAnimationFrame(animationId);
                    k3d.setClippingPlanes([]);
                    if (k3d.render) k3d.render();
                    updateButtonStates();
                }
            };
            
            updateButtonStates();
        }
        
        // Helper function to update masking based on controls
        window.updateMasking = function() {
            if (currentAnimation) {
                // If animation is running, it will pick up the new values
                return;
            }
            
            // Apply static masking
            const enableOctant = document.getElementById('enable-octant')?.checked || false;
            const enableSphere = document.getElementById('enable-sphere')?.checked || false;
            const sphereRadius = parseFloat(document.getElementById('sphere-radius')?.value || 1.5);
            const octantSelection = document.getElementById('octant-selection')?.value || '+++';
            
            const planes = [];
            
            // Apply octant masking if enabled
            if (enableOctant) {
                const signs = octantSelection.split('');
                // To show only one octant, we cut away the other 7
                if (signs[0] === '+') planes.push([-1, 0, 0, 0]);  // Remove x < 0
                else if (signs[0] === '-') planes.push([1, 0, 0, 0]);   // Remove x > 0
                
                if (signs[1] === '+') planes.push([0, -1, 0, 0]);  // Remove y < 0
                else if (signs[1] === '-') planes.push([0, 1, 0, 0]);   // Remove y > 0
                
                if (signs[2] === '+') planes.push([0, 0, -1, 0]);  // Remove z < 0
                else if (signs[2] === '-') planes.push([0, 0, 1, 0]);   // Remove z > 0
            }
            
            // Apply spherical masking if enabled (can be combined with octant)
            if (enableSphere) {
                const numPlanes = 12;
                for (let i = 0; i < numPlanes; i++) {
                    const theta = (i / numPlanes) * 2 * Math.PI;
                    for (let j = 1; j < numPlanes/2; j++) {
                        const phi = (j / (numPlanes/2)) * Math.PI;
                        const nx = -Math.sin(phi) * Math.cos(theta);
                        const ny = -Math.sin(phi) * Math.sin(theta);
                        const nz = -Math.cos(phi);
                        planes.push([nx, ny, nz, sphereRadius]);
                    }
                }
                planes.push([0, 0, -1, sphereRadius]);  // Top cap
                planes.push([0, 0, 1, sphereRadius]);   // Bottom cap
            }
            
            k3d.setClippingPlanes(planes);
            if (k3d.render) k3d.render();
        }
        """,
        
        "orbit": """
        window.startOrbitAnimation = function() {
            stopCurrentAnimation();
            
            const radius = parseFloat(document.getElementById('orbit-radius')?.value || 6);
            const height = parseFloat(document.getElementById('orbit-height')?.value || 2);
            const speed = parseFloat(document.getElementById('orbit-speed')?.value || 0.02);
            
            let angle = 0;
            let animationId;
            let stopped = false;
            
            function update() {
                if (stopped) return;
                
                angle += speed;
                const x = radius * Math.cos(angle);
                const z = radius * Math.sin(angle);
                
                k3d.setCamera([x, height, z, 0, 0, 0, 0, 1, 0]);
                if (k3d.render) k3d.render();
                
                animationId = requestAnimationFrame(update);
            }
            
            update();
            
            currentAnimation = {
                stop: () => {
                    stopped = true;
                    cancelAnimationFrame(animationId);
                    updateButtonStates();
                }
            };
            
            updateButtonStates();
        }
        """,
        
        "sweep": """
        window.startSweepAnimation = function() {
            stopCurrentAnimation();
            
            const axis = document.getElementById('sweep-axis')?.value || 'x';
            const speed = parseFloat(document.getElementById('sweep-speed')?.value || 0.02);
            
            let t = 0;
            let animationId;
            let stopped = false;
            
            function update() {
                if (stopped) return;
                
                t += speed;
                const position = 2 * Math.sin(t);
                
                let plane;
                switch(axis) {
                    case 'x': plane = [1, 0, 0, position]; break;
                    case 'y': plane = [0, 1, 0, position]; break;
                    case 'z': plane = [0, 0, 1, position]; break;
                    default: plane = [1, 0, 0, position];
                }
                
                k3d.setClippingPlanes([plane]);
                if (k3d.render) k3d.render();
                
                animationId = requestAnimationFrame(update);
            }
            
            update();
            
            currentAnimation = {
                stop: () => {
                    stopped = true;
                    cancelAnimationFrame(animationId);
                    k3d.setClippingPlanes([]);
                    if (k3d.render) k3d.render();
                    updateButtonStates();
                }
            };
            
            updateButtonStates();
        }
        """,
        
        "zoom": """
        window.startZoomAnimation = function() {
            stopCurrentAnimation();
            
            const minDist = parseFloat(document.getElementById('zoom-min')?.value || 2);
            const maxDist = parseFloat(document.getElementById('zoom-max')?.value || 10);
            const speed = parseFloat(document.getElementById('zoom-speed')?.value || 0.02);
            
            let t = 0;
            let animationId;
            let stopped = false;
            
            // Store initial camera position
            let baseCamera = null;
            if (window.K3DInstance && window.K3DInstance.camera) {
                baseCamera = [...window.K3DInstance.camera];
            } else {
                baseCamera = [4, 4, 4, 0, 0, 0, 0, 1, 0];
            }
            
            function update() {
                if (stopped) return;
                
                t += speed;
                const distance = minDist + (maxDist - minDist) * (Math.sin(t) + 1) / 2;
                
                // Calculate direction from target to eye
                const dx = baseCamera[0] - baseCamera[3];
                const dy = baseCamera[1] - baseCamera[4];
                const dz = baseCamera[2] - baseCamera[5];
                const baseDist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                
                if (baseDist > 0) {
                    // Normalize and scale to new distance
                    const scale = distance / baseDist;
                    const newCamera = [
                        baseCamera[3] + dx * scale,  // eye x
                        baseCamera[4] + dy * scale,  // eye y
                        baseCamera[5] + dz * scale,  // eye z
                        baseCamera[3],               // target x
                        baseCamera[4],               // target y
                        baseCamera[5],               // target z
                        baseCamera[6],               // up x
                        baseCamera[7],               // up y
                        baseCamera[8]                // up z
                    ];
                    
                    k3d.setCamera(newCamera);
                    if (k3d.render) k3d.render();
                }
                
                animationId = requestAnimationFrame(update);
            }
            
            update();
            
            currentAnimation = {
                stop: () => {
                    stopped = true;
                    cancelAnimationFrame(animationId);
                    // Restore original camera
                    if (baseCamera) {
                        k3d.setCamera(baseCamera);
                        if (k3d.render) k3d.render();
                    }
                    updateButtonStates();
                }
            };
            
            updateButtonStates();
        }
        """,
        
        "combined": """
        window.startCombinedAnimation = function() {
            stopCurrentAnimation();
            
            let cameraAngle = 0;
            let clippingT = 0;
            let animationId;
            let stopped = false;
            
            function update() {
                if (stopped) return;
                
                // Camera orbit
                cameraAngle += 0.01;
                const x = 6 * Math.cos(cameraAngle);
                const z = 6 * Math.sin(cameraAngle);
                k3d.setCamera([x, 3, z, 0, 0, 0, 0, 1, 0]);
                
                // Clipping sweep
                clippingT += 0.02;
                const clipPos = 2 * Math.sin(clippingT);
                k3d.setClippingPlanes([[1, 0, 0, clipPos]]);
                
                if (k3d.render) k3d.render();
                
                animationId = requestAnimationFrame(update);
            }
            
            update();
            
            currentAnimation = {
                stop: () => {
                    stopped = true;
                    cancelAnimationFrame(animationId);
                    k3d.setClippingPlanes([]);
                    if (k3d.render) k3d.render();
                    updateButtonStates();
                }
            };
            
            updateButtonStates();
        }
        """
    }
    
    # Select the appropriate animation
    animation_code = animations.get(animation_type, animations["orbit"])
    
    # Additional preset views
    preset_views = """
    window.setCameraPreset = function(preset) {
        const views = {
            'front': [0, 0, 6, 0, 0, 0, 0, 1, 0],
            'isometric': [4, 4, 4, 0, 0, 0, 0, 1, 0],
            'top': [0, 6, 0, 0, 0, 0, 0, 0, 1]
        };
        
        if (views[preset]) {
            k3d.setCamera(views[preset]);
            if (k3d.render) k3d.render();
        }
    }
    
    window.resetView = function() {
        stopCurrentAnimation();
        k3d.resetCamera();
        k3d.setClippingPlanes([]);
        if (k3d.render) k3d.render();
    }
    """
    
    # Initialization function
    init_function = """
    function initializeAnimations() {
        // Set up event listeners for controls
        updateButtonStates();
        
        // Auto-start animation if configured
        const autoStart = """ + str(config.get('auto_start', False)).lower() + """;
        if (autoStart) {
            setTimeout(() => {
                if (typeof startOrbitAnimation === 'function') {
                    startOrbitAnimation();
                } else if (typeof startSweepAnimation === 'function') {
                    startSweepAnimation();
                }
            }, 1000);
        }
    }
    """
    
    return base_functions + animation_code + preset_views + init_function


def _generate_control_panel(animation_type: str) -> str:
    """Generate HTML for the control panel based on animation type."""
    
    panels = {
        "masking": """
        <div class="k3d-animation-controls">
            <h3>🎭 Masking Animation</h3>
            <div class="control-group">
                <button onclick="startMaskingAnimation()">▶️ Start</button>
                <button class="stop" onclick="stopCurrentAnimation()">⏹️ Stop</button>
                <button onclick="resetView()">🔄 Reset</button>
            </div>
            
            <div class="control-group">
                <label>Enable Masking:</label>
                <label style="display: block; margin: 5px 0;">
                    <input type="checkbox" id="enable-octant" onchange="updateMasking(); updateControlVisibility()"> 
                    Octant Masking (1/8 space)
                </label>
                <label style="display: block; margin: 5px 0;">
                    <input type="checkbox" id="enable-sphere" onchange="updateMasking(); updateControlVisibility()"> 
                    Spherical Masking
                </label>
            </div>
            
            <div class="control-group" id="octant-controls" style="display: none;">
                <label>Octant Selection:</label>
                <select id="octant-selection" onchange="updateMasking()">
                    <option value="+++">+X +Y +Z (First octant)</option>
                    <option value="++-">+X +Y -Z</option>
                    <option value="+-+">+X -Y +Z</option>
                    <option value="+--">+X -Y -Z</option>
                    <option value="-++">-X +Y +Z</option>
                    <option value="-+-">-X +Y -Z</option>
                    <option value="--+">-X -Y +Z</option>
                    <option value="---">-X -Y -Z (Opposite octant)</option>
                </select>
            </div>
            
            <div class="control-group" id="sphere-controls" style="display: none;">
                <label>Sphere Radius: <span id="radius-value">1.5</span></label>
                <input type="range" id="sphere-radius" min="0.5" max="3.0" value="1.5" step="0.1"
                       oninput="document.getElementById('radius-value').textContent = this.value; updateMasking()">
                
                <label>
                    <input type="checkbox" id="animate-radius"> Animate radius
                </label>
            </div>
            
            <div class="control-group">
                <label>Animation Options:</label>
                <label>
                    <input type="checkbox" id="rotate-camera" checked> Rotate camera
                </label>
            </div>
            
            <div class="control-group">
                <label>Quick Presets:</label>
                <button onclick="document.getElementById('enable-octant').checked=true; document.getElementById('enable-sphere').checked=false; document.getElementById('octant-selection').value='+++'; updateControlVisibility(); updateMasking()">First Octant</button>
                <button onclick="document.getElementById('enable-octant').checked=false; document.getElementById('enable-sphere').checked=true; document.getElementById('sphere-radius').value='1.0'; updateControlVisibility(); updateMasking()">Unit Sphere</button>
                <button onclick="document.getElementById('enable-octant').checked=true; document.getElementById('enable-sphere').checked=true; document.getElementById('octant-selection').value='+++'; document.getElementById('sphere-radius').value='1.5'; updateControlVisibility(); updateMasking()">Octant + Sphere</button>
            </div>
            
            <div class="control-group">
                <label>Camera Views:</label>
                <button onclick="setCameraPreset('front')">Front</button>
                <button onclick="setCameraPreset('isometric')">Isometric</button>
                <button onclick="setCameraPreset('top')">Top</button>
            </div>
            
            <div class="value-display">
                Status: <span id="animation-status">Ready</span>
            </div>
            
            <script>
            // Show/hide controls based on enabled masking types
            function updateControlVisibility() {
                const enableOctant = document.getElementById('enable-octant').checked;
                const enableSphere = document.getElementById('enable-sphere').checked;
                const octantControls = document.getElementById('octant-controls');
                const sphereControls = document.getElementById('sphere-controls');
                
                // Show controls for enabled masking types
                octantControls.style.display = enableOctant ? 'block' : 'none';
                sphereControls.style.display = enableSphere ? 'block' : 'none';
            }
            
            // Initialize visibility on load
            setTimeout(updateControlVisibility, 100);
            </script>
        </div>
        """,
        "orbit": """
        <div class="k3d-animation-controls">
            <h3>🎬 Orbit Animation</h3>
            <div class="control-group">
                <button onclick="startOrbitAnimation()">▶️ Start</button>
                <button class="stop" onclick="stopCurrentAnimation()">⏹️ Stop</button>
                <button onclick="resetView()">🔄 Reset</button>
            </div>
            <div class="control-group">
                <label>Radius: <span id="radius-value">6</span></label>
                <input type="range" id="orbit-radius" min="2" max="10" value="6" step="0.5"
                       oninput="document.getElementById('radius-value').textContent = this.value">
                
                <label>Height: <span id="height-value">2</span></label>
                <input type="range" id="orbit-height" min="-5" max="5" value="2" step="0.5"
                       oninput="document.getElementById('height-value').textContent = this.value">
                
                <label>Speed: <span id="speed-value">0.02</span></label>
                <input type="range" id="orbit-speed" min="0.005" max="0.1" value="0.02" step="0.005"
                       oninput="document.getElementById('speed-value').textContent = this.value">
            </div>
            <div class="control-group">
                <label>Preset Views:</label>
                <button onclick="setCameraPreset('front')">Front</button>
                <button onclick="setCameraPreset('isometric')">Isometric</button>
                <button onclick="setCameraPreset('top')">Top</button>
            </div>
            <div class="value-display">
                Status: <span id="animation-status">Ready</span>
            </div>
        </div>
        """,
        
        "sweep": """
        <div class="k3d-animation-controls">
            <h3>✂️ Clipping Sweep</h3>
            <div class="control-group">
                <button onclick="startSweepAnimation()">▶️ Start</button>
                <button class="stop" onclick="stopCurrentAnimation()">⏹️ Stop</button>
                <button onclick="resetView()">🔄 Reset</button>
            </div>
            <div class="control-group">
                <label>Axis:</label>
                <select id="sweep-axis">
                    <option value="x">X-axis</option>
                    <option value="y">Y-axis</option>
                    <option value="z">Z-axis</option>
                </select>
                
                <label>Speed: <span id="sweep-speed-value">0.02</span></label>
                <input type="range" id="sweep-speed" min="0.005" max="0.1" value="0.02" step="0.005"
                       oninput="document.getElementById('sweep-speed-value').textContent = this.value">
            </div>
            <div class="value-display">
                Status: <span id="animation-status">Ready</span>
            </div>
        </div>
        """,
        
        "zoom": """
        <div class="k3d-animation-controls">
            <h3>🔍 Zoom Animation</h3>
            <div class="control-group">
                <button onclick="startZoomAnimation()">▶️ Start</button>
                <button class="stop" onclick="stopCurrentAnimation()">⏹️ Stop</button>
                <button onclick="resetView()">🔄 Reset</button>
            </div>
            <div class="control-group">
                <label>Min Distance: <span id="min-value">2</span></label>
                <input type="range" id="zoom-min" min="1" max="5" value="2" step="0.5"
                       oninput="document.getElementById('min-value').textContent = this.value">
                
                <label>Max Distance: <span id="max-value">10</span></label>
                <input type="range" id="zoom-max" min="5" max="20" value="10" step="1"
                       oninput="document.getElementById('max-value').textContent = this.value">
                
                <label>Speed: <span id="zoom-speed-value">0.02</span></label>
                <input type="range" id="zoom-speed" min="0.005" max="0.1" value="0.02" step="0.005"
                       oninput="document.getElementById('zoom-speed-value').textContent = this.value">
            </div>
            <div class="value-display">
                Status: <span id="animation-status">Ready</span>
            </div>
        </div>
        """,
        
        "combined": """
        <div class="k3d-animation-controls">
            <h3>🎭 Combined Animation</h3>
            <div class="control-group">
                <button onclick="startCombinedAnimation()">▶️ Start All</button>
                <button class="stop" onclick="stopCurrentAnimation()">⏹️ Stop</button>
                <button onclick="resetView()">🔄 Reset</button>
            </div>
            <div class="control-group">
                <p style="font-size: 12px; color: #666; margin: 10px 0;">
                    Combines orbital camera movement with clipping plane sweep
                </p>
            </div>
            <div class="control-group">
                <label>Quick Views:</label>
                <button onclick="setCameraPreset('front')">Front</button>
                <button onclick="setCameraPreset('isometric')">Isometric</button>
                <button onclick="setCameraPreset('top')">Top</button>
            </div>
            <div class="value-display">
                Status: <span id="animation-status">Ready</span>
            </div>
        </div>
        """
    }
    
    return panels.get(animation_type, panels["orbit"])


def export_animated_visualization(
    volume_data: np.ndarray,
    filename: str,
    animation_type: str = "orbit",
    **kwargs
) -> None:
    """
    Export an animated K3D visualization to HTML file.
    
    Parameters
    ----------
    volume_data : np.ndarray
        3D volume data to visualize
    filename : str
        Output HTML filename
    animation_type : str
        Type of animation: "orbit", "sweep", "zoom", "combined"
    **kwargs
        Additional arguments passed to create_animated_html
    """
    html_content = create_animated_html(
        volume_data,
        animation_type=animation_type,
        **kwargs
    )
    
    with open(filename, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Animated visualization exported to: {filename}")
    print(f"   Animation type: {animation_type}")
    print(f"   Open in browser for interactive controls")


if __name__ == "__main__":
    # Example usage
    print("K3D Animated Export Module")
    print("=" * 40)
    print("Example usage:")
    print()
    print("from eryx.visualization.volume.k3d.k3d_animated_export import export_animated_visualization")
    print("import numpy as np")
    print()
    print("# Load your data")
    print("data = np.load('diffuse_intensity.npy')")
    print("volume = data.reshape(41, 41, 41)")
    print()
    print("# Export with orbit animation")
    print("export_animated_visualization(")
    print("    volume,")
    print("    'animated_diffuse.html',")
    print("    animation_type='orbit',")
    print("    title='Diffuse Scattering with Orbit Animation'")
    print(")")