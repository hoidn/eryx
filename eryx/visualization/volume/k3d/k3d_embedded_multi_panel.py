#!/usr/bin/env python3
"""Create a self-contained multi-panel K3D visualization (no iframes needed)."""

import numpy as np
import k3d
import os

def create_embedded_multi_panel():
    """Create a single HTML with multiple K3D plots side by side."""
    
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
        datasets['thermal'] = np.exp(-(X**2 + Y**2 + Z**2))
        datasets['pumped'] = datasets['thermal'] * 1.2
        datasets['difference'] = datasets['pumped'] - datasets['thermal']
    
    # Create a single K3D plot with all volumes
    plot = k3d.plot(
        height=600,
        antialias=True,
        grid_visible=False,
        axes_helper=0.0,
        axes=['', '', ''],
        label_color=0x1e1e1e,
        camera_auto_fit=False
    )
    
    # Position each volume
    x_positions = {'thermal': -6, 'pumped': 0, 'difference': 6}
    
    for name, data in datasets.items():
        # Normalize
        clean_data = np.nan_to_num(data, nan=0.0)
        if np.any(clean_data > 0):
            data_min = np.min(clean_data[clean_data > 0])
            data_max = np.max(clean_data)
            if data_max > data_min:
                normalized = (clean_data - data_min) / (data_max - data_min)
            else:
                normalized = clean_data
        else:
            normalized = clean_data
            
        # Color range
        valid = normalized[normalized > 0]
        if len(valid) > 0:
            vmin, vmax = np.percentile(valid, [10, 90])
        else:
            vmin, vmax = 0, 1
        
        # Create volume at position
        x_off = x_positions[name]
        bounds = [x_off-2, x_off+2, -2, 2, -2, 2]
        
        volume = k3d.volume(
            normalized.astype(np.float32),
            color_range=[vmin, vmax],
            bounds=bounds,
            alpha_coef=15.0,
            color_map=k3d.basic_color_maps.Jet
        )
        plot += volume
        
        # Don't add text labels - they show as white squares
    
    # Set camera to see all three
    plot.camera = [0, -20, 10, 0, 0, 0, 0, 0, 1]
    
    # Get HTML and add custom controls
    html = plot.get_snapshot()
    
    # Add synchronized control UI
    custom_html = """
    <style>
        body { 
            margin: 0; 
            background: #1e1e1e;
            font-family: Arial, sans-serif;
        }
        .control-bar {
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(42, 42, 42, 0.95);
            padding: 10px 20px;
            border-radius: 8px;
            display: flex;
            gap: 20px;
            align-items: center;
            z-index: 1000;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .control-group {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .control-group label {
            color: #aaa;
            font-size: 12px;
        }
        button {
            background: #4a9eff;
            color: white;
            border: none;
            padding: 5px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        button:hover { background: #3a8eef; }
        button.active { background: #2a7edf; }
        input[type="range"] { width: 80px; }
        #alphaValue { color: #fff; font-size: 12px; }
    </style>
    
    <div class="control-bar">
        <div class="control-group">
            <button id="rotateBtn" onclick="toggleRotation()">Rotate</button>
            <button onclick="resetCamera()">Reset</button>
        </div>
        <div class="control-group">
            <label>Alpha:</label>
            <input type="range" id="alphaSlider" min="0.1" max="50" step="0.1" value="15" 
                   oninput="updateAlpha(this.value)">
            <span id="alphaValue">15.0</span>
        </div>
    </div>
    
    <script>
        let isRotating = false;
        let rotationFrame = null;
        let k3dPlot = null;
        let volumes = [];
        
        // Initialize K3D when ready
        async function initializeK3D() {
            // Try multiple times to get K3D instance
            for (let i = 0; i < 50; i++) {
                if (window.K3DInstance) {
                    try {
                        k3dPlot = window.K3DInstance instanceof Promise ? 
                                  await window.K3DInstance : window.K3DInstance;
                        
                        // Get volume objects - properly filter for actual volumes
                        const world = k3dPlot.getWorld();
                        
                        // Method 1: Get objects from K3DObjects and filter for volumes
                        const allObjects = Object.values(world.K3DObjects);
                        volumes = allObjects.filter(obj => 
                            obj && obj.alpha_coef !== undefined
                        );
                        
                        // If no volumes found, try getting from ObjectsListJson
                        if (volumes.length === 0) {
                            const objectIds = Object.keys(world.ObjectsListJson);
                            objectIds.forEach(id => {
                                const config = world.ObjectsListJson[id];
                                if (config && config.type === 'Volume') {
                                    const obj = world.K3DObjects[id];
                                    if (obj) volumes.push(obj);
                                }
                            });
                        }
                        
                        console.log('K3D initialized with', volumes.length, 'volume objects');
                        return;
                    } catch (e) {
                        console.log('Waiting for K3D...', e);
                    }
                }
                await new Promise(resolve => setTimeout(resolve, 100));
            }
            console.error('Failed to initialize K3D');
        }
        
        // Initialize on load
        window.addEventListener('load', function() {
            setTimeout(initializeK3D, 500);
        });
        
        // Make functions global for onclick handlers
        window.toggleRotation = function() {
            isRotating = !isRotating;
            const btn = document.getElementById('rotateBtn');
            
            if (isRotating) {
                btn.classList.add('active');
                btn.textContent = 'Stop';
                startRotation();
            } else {
                btn.classList.remove('active');
                btn.textContent = 'Rotate';
                stopRotation();
            }
        }
        
        function startRotation() {
            if (!k3dPlot) {
                console.warn('K3D not ready yet');
                return;
            }
            let angle = 0;
            const rotate = () => {
                angle += 0.005;
                const dist = 22;
                const height = 10;
                k3dPlot.setCamera([
                    dist * Math.cos(angle),
                    dist * Math.sin(angle),
                    height,
                    0, 0, 0,
                    0, 0, 1
                ]);
                k3dPlot.render();
                rotationFrame = requestAnimationFrame(rotate);
            };
            rotate();
        }
        
        function stopRotation() {
            if (rotationFrame) {
                cancelAnimationFrame(rotationFrame);
                rotationFrame = null;
            }
        }
        
        window.resetCamera = function() {
            if (isRotating) toggleRotation();
            if (k3dPlot) {
                k3dPlot.setCamera([0, -20, 10, 0, 0, 0, 0, 0, 1]);
                k3dPlot.render();
            } else {
                console.warn('K3D not ready yet');
            }
        }
        
        window.updateAlpha = function(value) {
            document.getElementById('alphaValue').textContent = parseFloat(value).toFixed(1);
            volumes.forEach(vol => {
                if (vol.alpha_coef !== undefined) {
                    vol.alpha_coef = parseFloat(value);
                }
            });
            if (k3dPlot) k3dPlot.render();
        }
    </script>
    """
    
    # Insert custom HTML before </body>
    html = html.replace('</body>', custom_html + '</body>')
    
    # Save
    with open('embedded_multi_panel.html', 'w') as f:
        f.write(html)
    
    print("\n✅ Created embedded_multi_panel.html")
    print("This file can be opened directly - no server needed!")
    print("All three volumes in one K3D instance with synchronized controls.")

if __name__ == "__main__":
    create_embedded_multi_panel()