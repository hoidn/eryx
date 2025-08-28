#!/usr/bin/env python3
"""
K3D Spherical (Radius-Based) Clipping Implementation

This provides multiple approaches for spherical clipping:
1. Polyhedron approximation using multiple planes
2. Dynamic data masking
3. Hybrid visualization with sphere guide
"""

import numpy as np
import k3d
from scipy.spatial import SphericalVoronoi
import time


class SphericalClippingController:
    """Controller for spherical clipping of k3d volumes."""
    
    def __init__(self, data_path='torch_grid_results.npz'):
        """Initialize the spherical clipping controller."""
        self.load_data(data_path)
        self.setup_parameters()
        self.create_plot()
        
    def load_data(self, data_path):
        """Load and prepare the data."""
        print("Loading data...")
        data = np.load(data_path)
        intensity = data['intensity'].reshape(data['map_shape'])
        
        # Clean data
        valid = intensity[~np.isnan(intensity)]
        vmin, vmax = np.percentile(valid, [1, 99])
        intensity = np.clip(intensity, vmin, vmax)
        intensity = np.nan_to_num(intensity, nan=vmin)
        
        # Normalize and store
        self.intensity_original = ((intensity - vmin) / (vmax - vmin)).astype(np.float32)
        self.intensity = self.intensity_original.copy()
        self.h, self.k, self.l = self.intensity.shape
        
        print(f"Data loaded: {self.h} × {self.k} × {self.l}")
        
    def setup_parameters(self):
        """Setup clipping parameters."""
        # Sphere parameters
        self.sphere_center = [self.h/2, self.k/2, self.l/2]  # Center of volume
        self.sphere_radius = min(self.h, self.k, self.l) / 4  # Initial radius
        self.clip_inside = True  # True = show inside sphere, False = show outside
        self.method = 'polyhedron'  # 'polyhedron', 'masking', or 'hybrid'
        self.polyhedron_resolution = 2  # Subdivision level for polyhedron
        
        # Generate polyhedron vertices for approximation
        self.generate_polyhedron_planes()
        
    def generate_polyhedron_planes(self):
        """Generate planes for polyhedron approximation of sphere."""
        # Generate vertices of a geodesic polyhedron
        # Start with icosahedron vertices
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Icosahedron vertices (normalized)
        verts = np.array([
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ])
        
        # Normalize vertices
        verts = verts / np.linalg.norm(verts, axis=1)[:, np.newaxis]
        
        # Subdivide for higher resolution if needed
        for _ in range(self.polyhedron_resolution):
            verts = self.subdivide_polyhedron(verts)
        
        self.polyhedron_normals = verts
        print(f"Generated {len(self.polyhedron_normals)} planes for sphere approximation")
        
    def subdivide_polyhedron(self, vertices):
        """Subdivide polyhedron vertices for better sphere approximation."""
        # Simple subdivision - add midpoints of edges
        # This is a simplified approach; proper geodesic subdivision would be better
        new_verts = list(vertices)
        
        # Add edge midpoints
        n = len(vertices)
        for i in range(n):
            for j in range(i+1, n):
                midpoint = (vertices[i] + vertices[j]) / 2
                norm = np.linalg.norm(midpoint)
                if norm > 1e-10:  # Avoid division by zero
                    midpoint = midpoint / norm
                    new_verts.append(midpoint)
        
        # Limit number of vertices
        if len(new_verts) > 20:
            new_verts = new_verts[:20]
            
        return np.array(new_verts)
        
    def create_plot(self):
        """Create the k3d plot."""
        self.plot = k3d.plot(
            height=600,
            grid_visible=False,
            menu_visibility=True
        )
        
        # Create volume
        self.volume = k3d.volume(
            self.intensity,
            bounds=[0, self.h, 0, self.k, 0, self.l],
            alpha_coef=30.0,
            color_map=k3d.basic_color_maps.Jet,
            interpolation=True
        )
        self.plot += self.volume
        
        # Create sphere guide (initially invisible)
        self.create_sphere_guide()
        
    def create_sphere_guide(self):
        """Create a visual sphere guide."""
        # Generate sphere mesh
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        
        x = self.sphere_radius * np.outer(np.cos(u), np.sin(v)) + self.sphere_center[0]
        y = self.sphere_radius * np.outer(np.sin(u), np.sin(v)) + self.sphere_center[1]
        z = self.sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.sphere_center[2]
        
        # Convert to mesh format
        vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
        
        # Create simple triangulation
        indices = []
        nu, nv = len(u), len(v)
        for i in range(nu-1):
            for j in range(nv-1):
                # Two triangles per quad
                p1 = i * nv + j
                p2 = (i+1) * nv + j
                p3 = i * nv + (j+1)
                p4 = (i+1) * nv + (j+1)
                
                indices.extend([p1, p2, p3])
                indices.extend([p2, p4, p3])
        
        indices = np.array(indices, dtype=np.uint32)
        
        # Create wireframe sphere
        self.sphere_mesh = k3d.mesh(
            vertices, indices,
            color=0x00ff00,
            opacity=0.2,
            wireframe=True,
            name="Clipping Sphere"
        )
        
    def apply_polyhedron_clipping(self):
        """Apply spherical clipping using polyhedron approximation."""
        planes = []
        
        for normal in self.polyhedron_normals:
            # Calculate distance for this plane
            # Plane equation: n·(p - c) = r (for points on sphere)
            # Rearranged: n·p = n·c + r
            # In k3d format: [nx, ny, nz, d] where d is the distance
            
            if self.clip_inside:
                # Show inside sphere: normals point outward
                d = -(np.dot(normal, self.sphere_center) + self.sphere_radius)
            else:
                # Show outside sphere: normals point inward
                normal = -normal
                d = -(np.dot(normal, self.sphere_center) - self.sphere_radius)
            
            planes.append([normal[0], normal[1], normal[2], d])
        
        self.plot.clipping_planes = planes
        
    def apply_data_masking(self):
        """Apply spherical clipping by masking the data."""
        # Create coordinate grids
        x = np.arange(self.h)
        y = np.arange(self.k)
        z = np.arange(self.l)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate distance from center
        dist = np.sqrt(
            (X - self.sphere_center[0])**2 + 
            (Y - self.sphere_center[1])**2 + 
            (Z - self.sphere_center[2])**2
        )
        
        # Apply masking
        self.intensity = self.intensity_original.copy()
        
        if self.clip_inside:
            # Show only inside sphere
            self.intensity[dist > self.sphere_radius] = 0
        else:
            # Show only outside sphere
            self.intensity[dist < self.sphere_radius] = 0
        
        # Update volume
        self.volume.volume = self.intensity
        
        # Clear clipping planes when using masking
        self.plot.clipping_planes = []
        
    def apply_hybrid_visualization(self):
        """Apply hybrid visualization with sphere guide."""
        # Apply data masking
        self.apply_data_masking()
        
        # Add sphere guide
        if self.sphere_mesh not in self.plot.objects:
            self.plot += self.sphere_mesh
        
        # Update sphere position and size
        self.update_sphere_guide()
        
    def update_sphere_guide(self):
        """Update the visual sphere guide."""
        # Regenerate sphere vertices
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        
        x = self.sphere_radius * np.outer(np.cos(u), np.sin(v)) + self.sphere_center[0]
        y = self.sphere_radius * np.outer(np.sin(u), np.sin(v)) + self.sphere_center[1]
        z = self.sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + self.sphere_center[2]
        
        vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1).astype(np.float32)
        self.sphere_mesh.vertices = vertices
        
    def update_clipping(self):
        """Update clipping based on current method."""
        if self.method == 'polyhedron':
            self.apply_polyhedron_clipping()
            # Remove sphere guide if present
            if self.sphere_mesh in self.plot.objects:
                self.plot -= self.sphere_mesh
                
        elif self.method == 'masking':
            self.apply_data_masking()
            # Remove sphere guide if present
            if self.sphere_mesh in self.plot.objects:
                self.plot -= self.sphere_mesh
                
        elif self.method == 'hybrid':
            self.apply_hybrid_visualization()
            
    def set_sphere_params(self, center=None, radius=None, inside=None):
        """Set sphere parameters."""
        if center is not None:
            self.sphere_center = center
        if radius is not None:
            self.sphere_radius = radius
        if inside is not None:
            self.clip_inside = inside
        self.update_clipping()
        
    def animate_radius(self, min_radius=5, max_radius=None, steps=50, delay=0.05):
        """Animate sphere radius."""
        if max_radius is None:
            max_radius = min(self.h, self.k, self.l) / 2
            
        print(f"Animating radius from {min_radius} to {max_radius}")
        
        # Expand
        for r in np.linspace(min_radius, max_radius, steps):
            self.sphere_radius = r
            self.update_clipping()
            time.sleep(delay)
            
        # Contract
        for r in np.linspace(max_radius, min_radius, steps):
            self.sphere_radius = r
            self.update_clipping()
            time.sleep(delay)
            
    def demo(self):
        """Run a demonstration of spherical clipping."""
        print("\n" + "="*60)
        print("SPHERICAL CLIPPING DEMONSTRATION")
        print("="*60)
        
        demos = [
            ("Polyhedron approximation - Inside", 'polyhedron', True),
            ("Polyhedron approximation - Outside", 'polyhedron', False),
            ("Data masking - Inside", 'masking', True),
            ("Data masking - Outside", 'masking', False),
            ("Hybrid visualization", 'hybrid', True),
        ]
        
        for name, method, inside in demos:
            print(f"\n{name}")
            self.method = method
            self.clip_inside = inside
            self.sphere_radius = min(self.h, self.k, self.l) / 3
            self.update_clipping()
            time.sleep(2)
            
            # Animate radius
            self.animate_radius(steps=20, delay=0.03)
            
        print("\nDemo complete!")


def create_interactive_notebook():
    """Generate Jupyter notebook for interactive spherical clipping."""
    
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# K3D Spherical (Radius-Based) Clipping\\n\\n",
    "Interactive controls for spherical clipping of 3D volume data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\\n",
    "import k3d\\n",
    "from ipywidgets import interact, widgets\\n",
    "from IPython.display import display\\n",
    "import sys\\n",
    "sys.path.append('.')\\n",
    "from k3d_spherical_clipping import SphericalClippingController"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize controller\\n",
    "controller = SphericalClippingController()\\n",
    "controller.plot.display()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Spherical Clipping Controls"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def update_sphere(method, radius, center_x, center_y, center_z, clip_inside, resolution):\\n",
    "    controller.method = method\\n",
    "    controller.sphere_radius = radius\\n",
    "    controller.sphere_center = [center_x, center_y, center_z]\\n",
    "    controller.clip_inside = clip_inside\\n",
    "    controller.polyhedron_resolution = resolution\\n",
    "    if method == 'polyhedron':\\n",
    "        controller.generate_polyhedron_planes()\\n",
    "    controller.update_clipping()\\n",
    "    \\n",
    "interact(update_sphere,\\n",
    "    method=widgets.RadioButtons(\\n",
    "        options=['polyhedron', 'masking', 'hybrid'],\\n",
    "        value='polyhedron',\\n",
    "        description='Method:'\\n",
    "    ),\\n",
    "    radius=widgets.FloatSlider(\\n",
    "        min=1, max=30, value=15, description='Radius:'\\n",
    "    ),\\n",
    "    center_x=widgets.FloatSlider(\\n",
    "        min=0, max=controller.h, value=controller.h/2, description='Center X:'\\n",
    "    ),\\n",
    "    center_y=widgets.FloatSlider(\\n",
    "        min=0, max=controller.k, value=controller.k/2, description='Center Y:'\\n",
    "    ),\\n",
    "    center_z=widgets.FloatSlider(\\n",
    "        min=0, max=controller.l, value=controller.l/2, description='Center Z:'\\n",
    "    ),\\n",
    "    clip_inside=widgets.Checkbox(\\n",
    "        value=True, description='Show Inside'\\n",
    "    ),\\n",
    "    resolution=widgets.IntSlider(\\n",
    "        min=0, max=3, value=2, description='Resolution:'\\n",
    "    )\\n",
    ");"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Animation controls\\n",
    "def animate():\\n",
    "    controller.animate_radius(min_radius=5, max_radius=25, steps=30, delay=0.03)\\n",
    "    \\n",
    "animate_btn = widgets.Button(description='Animate Radius')\\n",
    "animate_btn.on_click(lambda b: animate())\\n",
    "display(animate_btn)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    with open('k3d_spherical_clipping.ipynb', 'w') as f:
        f.write(notebook_content)
    
    print("Created k3d_spherical_clipping.ipynb")


if __name__ == "__main__":
    import sys
    
    if '--notebook' in sys.argv:
        create_interactive_notebook()
        print("\nJupyter notebook created! Run:")
        print("  jupyter notebook k3d_spherical_clipping.ipynb")
    
    elif '--demo' in sys.argv:
        controller = SphericalClippingController()
        controller.plot.display()
        controller.demo()
        
        print("\nKeep script running to maintain visualization")
        print("Press Ctrl+C to exit")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")
    
    else:
        # Interactive mode
        controller = SphericalClippingController()
        controller.plot.display()
        
        print("\n" + "="*60)
        print("SPHERICAL CLIPPING CONTROLLER")
        print("="*60)
        print("\nCommands:")
        print("  r <radius>  - Set sphere radius")
        print("  c <x> <y> <z> - Set sphere center")
        print("  i/o        - Toggle inside/outside clipping")
        print("  m <method> - Set method (polyhedron/masking/hybrid)")
        print("  a          - Animate radius")
        print("  d          - Run demo")
        print("  q          - Quit")
        print("="*60)
        
        while True:
            try:
                cmd = input("\n> ").strip().lower().split()
                if not cmd:
                    continue
                    
                if cmd[0] == 'q':
                    break
                elif cmd[0] == 'r' and len(cmd) > 1:
                    controller.sphere_radius = float(cmd[1])
                    controller.update_clipping()
                    print(f"Radius set to {controller.sphere_radius}")
                elif cmd[0] == 'c' and len(cmd) > 3:
                    controller.sphere_center = [float(cmd[1]), float(cmd[2]), float(cmd[3])]
                    controller.update_clipping()
                    print(f"Center set to {controller.sphere_center}")
                elif cmd[0] == 'i':
                    controller.clip_inside = True
                    controller.update_clipping()
                    print("Showing inside sphere")
                elif cmd[0] == 'o':
                    controller.clip_inside = False
                    controller.update_clipping()
                    print("Showing outside sphere")
                elif cmd[0] == 'm' and len(cmd) > 1:
                    controller.method = cmd[1]
                    controller.update_clipping()
                    print(f"Method set to {controller.method}")
                elif cmd[0] == 'a':
                    controller.animate_radius()
                elif cmd[0] == 'd':
                    controller.demo()
                else:
                    print("Unknown command")
                    
            except Exception as e:
                print(f"Error: {e}")
