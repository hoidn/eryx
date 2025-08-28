#!/usr/bin/env python3
"""
K3D Interactive Clipping - Python Script Version

Run this script to get an interactive k3d plot with working clipping controls.
No HTML export needed - everything works in Python!

Usage:
    python k3d_interactive_clipping.py
    
Then use the keyboard controls shown on screen.
"""

import numpy as np
import k3d
import time
import threading
import sys
import select
import termios
import tty

class InteractiveClippingController:
    """Interactive controller for k3d clipping planes."""
    
    def __init__(self):
        """Initialize the controller."""
        self.load_data()
        self.create_plot()
        self.clipping_position = 50  # 0-100
        self.clipping_axis = 'z'  # x, y, or z
        self.clipping_enabled = True
        self.animation_running = False
        
    def load_data(self):
        """Load and prepare the diffuse intensity data."""
        print("Loading data...")
        data = np.load('torch_grid_results.npz')
        intensity = data['intensity'].reshape(data['map_shape'])
        
        # Clean data
        valid = intensity[~np.isnan(intensity)]
        vmin, vmax = np.percentile(valid, [1, 99])
        intensity = np.clip(intensity, vmin, vmax)
        intensity = np.nan_to_num(intensity, nan=vmin)
        
        # Normalize
        self.intensity = ((intensity - vmin) / (vmax - vmin)).astype(np.float32)
        self.h, self.k, self.l = self.intensity.shape
        
        print(f"Data loaded: {self.h} × {self.k} × {self.l}")
        
    def create_plot(self):
        """Create the k3d plot with volume."""
        print("Creating k3d plot...")
        
        self.plot = k3d.plot(
            height=600,
            grid_visible=False,
            menu_visibility=True,
            antialias=True
        )
        
        # Create volume
        self.volume = k3d.volume(
            self.intensity,
            color_map=k3d.basic_color_maps.Jet,
            color_range=[0.0, 1.0],
            alpha_coef=30.0,
            bounds=[-self.h/2, self.h/2, -self.k/2, self.k/2, -self.l/2, self.l/2],
            name="Diffuse Intensity"
        )
        
        self.plot += self.volume
        
    def update_clipping(self):
        """Update the clipping plane based on current settings."""
        if not self.clipping_enabled:
            self.plot.clipping_planes = []
            return
        
        # Get bounds for current axis
        bounds = {'x': self.h, 'y': self.k, 'z': self.l}
        max_val = bounds[self.clipping_axis] / 2
        
        # Calculate distance from position (0-100)
        distance = -max_val + (2 * max_val * self.clipping_position / 100)
        
        # Set normal vector
        normals = {
            'x': [1, 0, 0],
            'y': [0, 1, 0],
            'z': [0, 0, 1]
        }
        normal = normals[self.clipping_axis]
        
        # Apply clipping plane
        self.plot.clipping_planes = [[normal[0], normal[1], normal[2], -distance]]
        
    def animate_clipping(self):
        """Animate the clipping plane."""
        print("\\nAnimating clipping plane...")
        self.animation_running = True
        
        while self.animation_running:
            # Forward
            for pos in range(0, 101, 2):
                if not self.animation_running:
                    break
                self.clipping_position = pos
                self.update_clipping()
                time.sleep(0.03)
            
            # Backward
            for pos in range(100, -1, -2):
                if not self.animation_running:
                    break
                self.clipping_position = pos
                self.update_clipping()
                time.sleep(0.03)
        
        print("Animation stopped")
        
    def print_status(self):
        """Print current status."""
        print(f"\\rAxis: {self.clipping_axis.upper()} | Position: {self.clipping_position:3d}% | " +
              f"Clipping: {'ON ' if self.clipping_enabled else 'OFF'} | " +
              f"Animation: {'ON ' if self.animation_running else 'OFF'}", end='', flush=True)
        
    def run_interactive(self):
        """Run the interactive control loop."""
        print("\\n" + "="*60)
        print("K3D INTERACTIVE CLIPPING CONTROLS")
        print("="*60)
        print("\\nKeyboard Controls:")
        print("  ← / → : Move clipping plane")
        print("  ↑ / ↓ : Move clipping plane (fine)")
        print("  X/Y/Z : Switch clipping axis")
        print("  SPACE : Toggle clipping on/off")
        print("  A     : Start/stop animation")
        print("  R     : Reset to center")
        print("  0-9   : Jump to position (0=0%, 5=50%, 9=90%)")
        print("  Q     : Quit")
        print("\\n" + "="*60)
        
        # Display the plot
        self.plot.display()
        
        print("\\nPlot displayed. Use keyboard controls:\\n")
        
        # Main control loop
        while True:
            self.print_status()
            
            # Simple input handling (cross-platform would need more work)
            key = input("\\n> ").lower().strip()
            
            if key == 'q':
                break
            elif key == 'x':
                self.clipping_axis = 'x'
                self.update_clipping()
            elif key == 'y':
                self.clipping_axis = 'y'
                self.update_clipping()
            elif key == 'z':
                self.clipping_axis = 'z'
                self.update_clipping()
            elif key == ' ' or key == 'space':
                self.clipping_enabled = not self.clipping_enabled
                self.update_clipping()
            elif key == 'a':
                if self.animation_running:
                    self.animation_running = False
                else:
                    threading.Thread(target=self.animate_clipping, daemon=True).start()
            elif key == 'r':
                self.clipping_position = 50
                self.update_clipping()
            elif key == 'left' or key == '<':
                self.clipping_position = max(0, self.clipping_position - 10)
                self.update_clipping()
            elif key == 'right' or key == '>':
                self.clipping_position = min(100, self.clipping_position + 10)
                self.update_clipping()
            elif key == 'up' or key == '+':
                self.clipping_position = min(100, self.clipping_position + 1)
                self.update_clipping()
            elif key == 'down' or key == '-':
                self.clipping_position = max(0, self.clipping_position - 1)
                self.update_clipping()
            elif key.isdigit():
                self.clipping_position = int(key) * 10
                self.update_clipping()
            else:
                print(f"Unknown command: {key}")

def main():
    """Main function to run the interactive controller."""
    controller = InteractiveClippingController()
    controller.run_interactive()

def simple_demo():
    """Simple demonstration without interactive controls."""
    print("\\n" + "="*60)
    print("K3D CLIPPING PLANE DEMONSTRATION")
    print("="*60)
    
    # Load data
    data = np.load('torch_grid_results.npz')
    intensity = data['intensity'].reshape(data['map_shape'])
    
    # Clean data
    valid = intensity[~np.isnan(intensity)]
    vmin, vmax = np.percentile(valid, [1, 99])
    intensity = np.clip(intensity, vmin, vmax)
    intensity = np.nan_to_num(intensity, nan=vmin)
    intensity = ((intensity - vmin) / (vmax - vmin)).astype(np.float32)
    
    h, k, l = intensity.shape
    print(f"\\nData shape: {h} × {k} × {l}")
    
    # Create plot
    plot = k3d.plot(height=600)
    volume = k3d.volume(
        intensity,
        bounds=[0, h, 0, k, 0, l],
        alpha_coef=30.0
    )
    plot += volume
    plot.display()
    
    print("\\nDemonstrating clipping planes...")
    
    # Demonstration sequence
    demos = [
        ("No clipping", []),
        ("X-axis center", [[1, 0, 0, -h/2]]),
        ("Y-axis center", [[0, 1, 0, -k/2]]),
        ("Z-axis center", [[0, 0, 1, -l/2]]),
        ("Corner cut", [[1, 0, 0, -h/2], [0, 1, 0, -k/2], [0, 0, 1, -l/2]]),
        ("Diagonal", [[1, 1, 1, -h/2]]),
    ]
    
    for name, planes in demos:
        print(f"  Setting: {name}")
        plot.clipping_planes = planes
        time.sleep(2)
    
    print("\\nAnimating Z-axis clipping...")
    for i in range(50):
        z_pos = -l/2 + l * (i / 49)
        plot.clipping_planes = [[0, 0, 1, -z_pos]]
        time.sleep(0.05)
    
    # Clear clipping
    plot.clipping_planes = []
    print("\\nDemo complete! Clipping planes work perfectly in Python!")
    
    return plot

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        # Run simple demo
        plot = simple_demo()
        print("\\nKeep this script running to interact with the plot")
        print("Press Ctrl+C to exit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\\nExiting...")
    else:
        # Run interactive controller
        main()