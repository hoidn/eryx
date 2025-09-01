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
import os
import sys

# Add the project root to path if not already there
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from eryx.visualization.core.data_handler import IntensityDataHandler


class SphericalClippingController:
    """Controller for spherical clipping of k3d volumes."""
    
    def __init__(self, data_source='torch'):
        """Initialize the spherical clipping controller.
        
        Args:
            data_source: Data source identifier. Can be:
                        - 'torch': torch_diffuse_intensity.npy or torch_grid_results.npz
                        - 'np': np_diffuse_intensity.npy or np_results.npz  
                        - 'arbq': arb_q_diffuse_intensity.npy or torch_arbq_results.npz
                        - Path to specific file (NPZ or NPY)
                        - numpy array
        """
        self.load_data(data_source)
        self.setup_parameters()
        self.create_plot()
        
    def load_data(self, data_source):
        """Load and prepare the data using DataHandler infrastructure."""
        print("Loading data...")
        
        # Handle numpy arrays directly
        if isinstance(data_source, np.ndarray):
            print("Using provided numpy array directly")
            intensity = data_source
            q_vectors = None
            map_shape = data_source.shape if data_source.ndim == 3 else None
        else:
            # For dataset identifiers like 'torch', 'np', 'arbq', convert to absolute paths
            if isinstance(data_source, str) and data_source in ['torch', 'np', 'arbq']:
                # Get the project root directory (4 levels up from this file)
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
                
                # Try NPZ file first (contains metadata)
                dataset_files = {
                    'torch': {'npz': 'torch_grid_results.npz', 'npy': 'torch_diffuse_intensity.npy'},
                    'np': {'npz': 'np_results.npz', 'npy': 'np_diffuse_intensity.npy'},
                    'arbq': {'npz': 'torch_arbq_results.npz', 'npy': 'arb_q_diffuse_intensity.npy'}
                }
                
                npz_file = os.path.join(project_root, dataset_files[data_source]['npz'])
                npy_file = os.path.join(project_root, dataset_files[data_source]['npy'])
                
                # Use the NPZ file if it exists, otherwise the NPY file
                if os.path.exists(npz_file):
                    data_source = npz_file
                    print(f"Using absolute path: {npz_file}")
                elif os.path.exists(npy_file):
                    data_source = npy_file
                    print(f"Using absolute path: {npy_file}")
                else:
                    print(f"Warning: Neither {npz_file} nor {npy_file} found")
            
            # Use DataHandler to load the data
            handler = IntensityDataHandler(data_source)
            q_vectors, intensity, map_shape = handler.load_data()
        
        if intensity is None:
            # Fallback for backward compatibility
            if isinstance(data_source, str) and os.path.exists(data_source):
                print("DataHandler failed, falling back to direct NPZ loading...")
                data = np.load(data_source)
                if 'intensity' in data and 'map_shape' in data:
                    intensity = data['intensity']
                    map_shape = data['map_shape']
                else:
                    raise ValueError(f"Could not load data from {data_source}")
            else:
                raise ValueError(f"Could not load data from {data_source}")
        
        # Reshape if needed
        if intensity.ndim == 1:
            if map_shape is not None:
                intensity = intensity.reshape(map_shape)
                print(f"Reshaped data using map_shape: {map_shape}")
            else:
                # Try to detect cubic shape
                cube_size = int(round(len(intensity) ** (1/3)))
                if cube_size ** 3 == len(intensity):
                    intensity = intensity.reshape(cube_size, cube_size, cube_size)
                    print(f"Auto-detected cubic shape: {intensity.shape}")
                else:
                    raise ValueError(f"Cannot determine 3D shape for {len(intensity)} elements")
        
        # Clean data
        valid = intensity[~np.isnan(intensity)]
        if len(valid) == 0:
            print("Warning: All values are NaN, using zeros")
            intensity = np.zeros_like(intensity)
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = np.percentile(valid, [1, 99])
            intensity = np.clip(intensity, vmin, vmax)
            intensity = np.nan_to_num(intensity, nan=vmin)
        
        # Normalize and store
        if vmax > vmin:
            self.intensity_original = ((intensity - vmin) / (vmax - vmin)).astype(np.float32)
        else:
            self.intensity_original = intensity.astype(np.float32)
        self.intensity = self.intensity_original.copy()
        self.h, self.k, self.l = self.intensity.shape
        
        print(f"Data loaded: {self.h} × {self.k} × {self.l}")
        
    def setup_parameters(self):
        """Setup clipping parameters."""
        # Sphere parameters
        self.sphere_center = [self.h/2, self.k/2, self.l/2]  # Center of volume
        self.sphere_radius = min(self.h, self.k, self.l) / 4  # Initial radius
        self.clip_inside = True  # True = show inside sphere, False = show outside
        self.method = 'masking'  # 'masking' (default), 'hybrid', or 'polyhedron' (deprecated)
        self.polyhedron_resolution = 2  # Subdivision level for polyhedron (kept for backwards compat)
        
        # Octant cutting parameters
        self.octant_cut = False  # Enable octant cutting
        self.octant_mode = 'first'  # 'first' (x>0, y>0, z>0) or 'custom'
        self.octant_signs = [1, 1, 1]  # Signs for custom octant selection
        
        # Intensity scaling parameters
        self.log_scale = False  # Enable log scaling for better dynamic range
        self.log_dynamic_range = 100  # Dynamic range factor for log scaling
        
        # Camera state management
        self.camera_initial = None  # Will store initial camera position
        self.is_animating = False  # Track animation state
        self.animation_type = None  # Current animation type
        
        # Generate polyhedron vertices for approximation (deprecated but kept for compatibility)
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
        
        # Create octant boundary indicators
        self.octant_planes = []
        
    # ========== COORDINATE SYSTEM UTILITIES ==========
    
    def voxel_to_world(self, voxel_coord):
        """Convert voxel coordinates to world space.
        
        Args:
            voxel_coord: [x, y, z] in voxel indices
            
        Returns:
            [x, y, z] in world space coordinates
        """
        bounds = self.volume.transform.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
        
        x = bounds[0] + (voxel_coord[0] / self.h) * (bounds[1] - bounds[0])
        y = bounds[2] + (voxel_coord[1] / self.k) * (bounds[3] - bounds[2])
        z = bounds[4] + (voxel_coord[2] / self.l) * (bounds[5] - bounds[4])
        
        return [x, y, z]
    
    def get_world_sphere_center(self):
        """Get sphere center in world coordinates.
        
        Returns:
            [x, y, z] world coordinates of sphere center
        """
        return self.voxel_to_world(self.sphere_center)
    
    def get_world_bounds(self):
        """Get the world space bounds of the volume.
        
        Returns:
            [xmin, xmax, ymin, ymax, zmin, zmax] in world space
        """
        return self.volume.transform.bounds
    
    def get_optimal_camera_distance(self, fov_degrees=30, scale_factor=2.5):
        """Calculate optimal camera distance based on data bounds.
        
        Args:
            fov_degrees: Field of view in degrees
            scale_factor: Multiplier for distance (higher = farther)
            
        Returns:
            Optimal camera distance from center
        """
        bounds = self.get_world_bounds()
        
        # Find maximum dimension
        max_dim = max(
            bounds[1] - bounds[0],  # x size
            bounds[3] - bounds[2],  # y size
            bounds[5] - bounds[4]   # z size
        )
        
        # Calculate distance using FOV
        import math
        fov_rad = math.radians(fov_degrees)
        distance = (max_dim / 2) / math.tan(fov_rad / 2)
        
        return distance * scale_factor
    
    # ========== CAMERA STATE MANAGEMENT ==========
    
    def store_initial_camera(self):
        """Store the initial camera position for later reset."""
        if self.plot and hasattr(self.plot, 'camera'):
            self.camera_initial = list(self.plot.camera)
            
    def reset_camera(self):
        """Reset camera to initial position."""
        if self.camera_initial:
            self.plot.camera = self.camera_initial
            self.is_animating = False
            self.animation_type = None
            
    def get_default_camera(self, elevation=30, azimuth=45):
        """Calculate default camera position.
        
        Args:
            elevation: Vertical angle in degrees (0=horizon, 90=top)
            azimuth: Horizontal angle in degrees
            
        Returns:
            Camera array [pos_x, pos_y, pos_z, target_x, target_y, target_z, up_x, up_y, up_z]
        """
        import math
        
        center = self.get_world_sphere_center()
        distance = self.get_optimal_camera_distance()
        
        # Convert angles to radians
        elev_rad = math.radians(elevation)
        azim_rad = math.radians(azimuth)
        
        # Calculate camera position
        x = distance * math.cos(elev_rad) * math.cos(azim_rad)
        y = distance * math.cos(elev_rad) * math.sin(azim_rad)
        z = distance * math.sin(elev_rad)
        
        # Camera array format
        camera = [
            x + center[0], y + center[1], z + center[2],  # Position
            center[0], center[1], center[2],              # Target
            0, 0, 1                                       # Up vector
        ]
        
        return camera
    
    # ========== CAMERA ANIMATIONS ==========
    
    def add_orbital_camera(self, duration=6.0, radius_factor=2.5, elevation=45, num_frames=60):
        """Add smooth orbital camera animation around sphere center.
        
        Args:
            duration: Animation duration in seconds
            radius_factor: Distance multiplier from optimal distance
            elevation: Vertical angle in degrees (0-90)
            num_frames: Number of keyframes to generate
        """
        import math
        
        world_center = self.get_world_sphere_center()
        distance = self.get_optimal_camera_distance(scale_factor=radius_factor)
        
        # Generate orbital path
        frames = []
        for i in range(num_frames + 1):  # +1 to close loop
            t = duration * i / num_frames
            theta = 2 * math.pi * i / num_frames
            phi = math.radians(elevation)
            
            # Spherical to Cartesian
            x = distance * math.sin(phi) * math.cos(theta)
            y = distance * math.sin(phi) * math.sin(theta)
            z = distance * math.cos(phi)
            
            camera = [
                x + world_center[0], y + world_center[1], z + world_center[2],  # Position
                world_center[0], world_center[1], world_center[2],              # Target
                0, 0, 1                                                         # Up vector
            ]
            frames.append([t, camera])
        
        self.plot.camera_animation = frames
        self.is_animating = True
        self.animation_type = 'orbital'
        self.plot.start_auto_play()
    
    def add_zoom_animation(self, start_distance=None, end_distance=None, duration=4.0, 
                          elevation=30, azimuth=45, num_frames=50):
        """Create smooth zoom in/out animation.
        
        Args:
            start_distance: Starting distance (None = current)
            end_distance: Ending distance (None = optimal)
            duration: Animation duration in seconds
            elevation: Camera elevation angle
            azimuth: Camera azimuth angle
            num_frames: Number of keyframes
        """
        import math
        
        world_center = self.get_world_sphere_center()
        
        if start_distance is None:
            start_distance = self.get_optimal_camera_distance(scale_factor=3.0)
        if end_distance is None:
            end_distance = self.get_optimal_camera_distance(scale_factor=1.0)
        
        # Convert angles to radians
        elev_rad = math.radians(elevation)
        azim_rad = math.radians(azimuth)
        
        # Generate zoom path
        frames = []
        for i in range(num_frames):
            t = duration * i / (num_frames - 1)
            progress = i / (num_frames - 1)
            
            # Smooth interpolation (ease-in-out)
            smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))
            distance = start_distance + (end_distance - start_distance) * smooth_progress
            
            # Optional rotation during zoom
            theta = azim_rad + math.pi * progress / 6  # Slight rotation
            
            x = distance * math.cos(elev_rad) * math.cos(theta)
            y = distance * math.cos(elev_rad) * math.sin(theta)
            z = distance * math.sin(elev_rad)
            
            camera = [
                x + world_center[0], y + world_center[1], z + world_center[2],
                world_center[0], world_center[1], world_center[2],
                0, 0, 1
            ]
            frames.append([t, camera])
        
        self.plot.camera_animation = frames
        self.is_animating = True
        self.animation_type = 'zoom'
        self.plot.start_auto_play()
    
    def set_preset_view(self, view_name='isometric', transition_duration=1.0):
        """Set camera to a preset viewpoint with smooth transition.
        
        Args:
            view_name: Preset name ('front', 'back', 'left', 'right', 'top', 'bottom', 'isometric')
            transition_duration: Time to transition to new view
        """
        import math
        
        world_center = self.get_world_sphere_center()
        distance = self.get_optimal_camera_distance()
        
        # Define preset views (azimuth, elevation)
        presets = {
            'front': (0, 0),
            'back': (180, 0),
            'left': (-90, 0),
            'right': (90, 0),
            'top': (0, 90),
            'bottom': (0, -90),
            'isometric': (45, 35.264),  # Classic isometric angle
            'optimal': (45, 30)
        }
        
        if view_name not in presets:
            view_name = 'isometric'
        
        azimuth, elevation = presets[view_name]
        azim_rad = math.radians(azimuth)
        elev_rad = math.radians(elevation)
        
        # Calculate camera position
        x = distance * math.cos(elev_rad) * math.cos(azim_rad)
        y = distance * math.cos(elev_rad) * math.sin(azim_rad)
        z = distance * math.sin(elev_rad)
        
        target_camera = [
            x + world_center[0], y + world_center[1], z + world_center[2],
            world_center[0], world_center[1], world_center[2],
            0, 0, 1
        ]
        
        if transition_duration > 0 and hasattr(self.plot, 'camera') and self.plot.camera is not None:
            # Smooth transition from current to target
            current_camera = list(self.plot.camera)
            
            # Ensure current camera has correct length
            if len(current_camera) != 9:
                # If camera not properly initialized, use default
                current_camera = self.get_default_camera()
                
            frames = []
            
            for i in range(20):
                t = transition_duration * i / 19
                progress = i / 19
                smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))
                
                # Interpolate each camera component
                camera = []
                for j in range(9):
                    val = current_camera[j] + (target_camera[j] - current_camera[j]) * smooth_progress
                    camera.append(val)
                
                frames.append([t, camera])
            
            self.plot.camera_animation = frames
            self.plot.start_auto_play()
        else:
            # Instant transition
            self.plot.camera = target_camera
    
    # ========== SYNCHRONIZED ANIMATIONS ==========
    
    def create_zoom_reveal_animation(self, duration=5.0, final_radius_ratio=0.2, num_frames=50):
        """Zoom in while shrinking sphere to reveal internal structure.
        
        Args:
            duration: Total animation duration in seconds
            final_radius_ratio: Final sphere radius as ratio of initial (0.2 = 20%)
            num_frames: Number of animation frames
        """
        import threading
        import time
        import math
        
        world_center = self.get_world_sphere_center()
        
        # Camera animation (automatic via time series)
        camera_frames = []
        for i in range(num_frames):
            t = duration * i / (num_frames - 1)
            progress = i / (num_frames - 1)
            
            # Smooth progress (ease-in-out)
            smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))
            
            # Zoom from far to close
            distance = self.get_optimal_camera_distance(scale_factor=3.0 - 2.0 * smooth_progress)
            theta = math.pi * progress / 3  # Slight rotation during zoom
            elevation = math.radians(30 + 15 * smooth_progress)  # Rise slightly
            
            x = distance * math.cos(elevation) * math.cos(theta)
            y = distance * math.cos(elevation) * math.sin(theta)
            z = distance * math.sin(elevation)
            
            camera = [
                x + world_center[0], y + world_center[1], z + world_center[2],
                world_center[0], world_center[1], world_center[2],
                0, 0, 1
            ]
            camera_frames.append([t, camera])
        
        self.plot.camera_animation = camera_frames
        
        # Sphere animation (manual in separate thread)
        original_radius = self.sphere_radius
        
        def animate_sphere():
            for i in range(num_frames):
                progress = i / (num_frames - 1)
                smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))
                
                self.sphere_radius = original_radius * (1 - (1 - final_radius_ratio) * smooth_progress)
                self.update_clipping()
                time.sleep(duration / num_frames)
            
            # Reset after animation
            self.sphere_radius = original_radius
            self.update_clipping()
        
        # Start both animations
        self.is_animating = True
        self.animation_type = 'zoom_reveal'
        self.plot.start_auto_play()
        threading.Thread(target=animate_sphere, daemon=True).start()
    
    def create_octant_inspection_animation(self, duration=8.0, pause_per_octant=0.5):
        """Camera tour visiting each octant for comprehensive inspection.
        
        Args:
            duration: Total animation duration
            pause_per_octant: Pause time at each octant view
        """
        import math
        
        world_center = self.get_world_sphere_center()
        radius = self.get_optimal_camera_distance(scale_factor=2.0)
        
        frames = []
        octant_views = []
        
        # Generate octant viewing positions
        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                for z_sign in [1, -1]:
                    # Position camera in each octant
                    theta = math.atan2(y_sign, x_sign)
                    phi = math.acos(z_sign / math.sqrt(3))
                    
                    x = radius * math.sin(phi) * math.cos(theta)
                    y = radius * math.sin(phi) * math.sin(theta)
                    z = radius * math.cos(phi)
                    
                    octant_views.append([x, y, z])
        
        # Create smooth path through octant views
        time_per_octant = duration / len(octant_views)
        
        for i, pos in enumerate(octant_views):
            # Add transition to octant
            for j in range(10):
                t = i * time_per_octant + (time_per_octant - pause_per_octant) * j / 9
                
                if i == 0 and j == 0:
                    # Start from current position
                    camera = [
                        pos[0] + world_center[0],
                        pos[1] + world_center[1],
                        pos[2] + world_center[2],
                        world_center[0], world_center[1], world_center[2],
                        0, 0, 1
                    ]
                else:
                    # Interpolate to next position
                    prev_pos = octant_views[i-1] if j == 0 else pos
                    progress = j / 9
                    
                    x = prev_pos[0] + (pos[0] - prev_pos[0]) * progress
                    y = prev_pos[1] + (pos[1] - prev_pos[1]) * progress
                    z = prev_pos[2] + (pos[2] - prev_pos[2]) * progress
                    
                    camera = [
                        x + world_center[0],
                        y + world_center[1],
                        z + world_center[2],
                        world_center[0], world_center[1], world_center[2],
                        0, 0, 1
                    ]
                
                frames.append([t, camera])
            
            # Optional: Sync octant exclusion with camera position
            if self.octant_cut:
                import threading
                import time
                
                def update_octant():
                    time.sleep(t)
                    self.octant_signs = [
                        1 if pos[0] > 0 else -1,
                        1 if pos[1] > 0 else -1,
                        1 if pos[2] > 0 else -1
                    ]
                    self.update_clipping()
                
                threading.Thread(target=update_octant, daemon=True).start()
        
        self.plot.camera_animation = frames
        self.is_animating = True
        self.animation_type = 'octant_tour'
        self.plot.start_auto_play()
    
    def add_anisotropy_showcase_animation(self, duration=10.0, elevations=None, 
                                         points_per_elevation=20):
        """Multi-elevation orbital animation to highlight anisotropic features.
        
        Args:
            duration: Total animation duration
            elevations: List of elevation angles in degrees (None = default sequence)
            points_per_elevation: Number of orbital points at each elevation
        """
        import math
        
        if elevations is None:
            elevations = [15, 30, 45, 60, 75, 60, 45, 30, 15]  # Up and down sweep
        
        world_center = self.get_world_sphere_center()
        base_radius = self.get_optimal_camera_distance(scale_factor=2.0)
        
        frames = []
        t = 0
        time_increment = duration / (len(elevations) * points_per_elevation)
        
        for elevation in elevations:
            phi = math.radians(elevation)
            
            for i in range(points_per_elevation):
                theta = 2 * math.pi * i / points_per_elevation
                
                # Variable radius for visual interest
                radius = base_radius * (1 + 0.2 * math.sin(theta * 2))
                
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.sin(phi) * math.sin(theta)
                z = radius * math.cos(phi)
                
                camera = [
                    x + world_center[0], y + world_center[1], z + world_center[2],
                    world_center[0], world_center[1], world_center[2],
                    0, 0, 1
                ]
                frames.append([t, camera])
                t += time_increment
        
        self.plot.camera_animation = frames
        self.is_animating = True
        self.animation_type = 'anisotropy_showcase'
        self.plot.start_auto_play()
    
    def stop_animation(self):
        """Stop any running animation and clear animation state."""
        if hasattr(self.plot, 'stop_auto_play'):
            self.plot.stop_auto_play()
        self.is_animating = False
        self.animation_type = None
    
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
        
    def apply_intensity_scaling(self):
        """Apply log or linear scaling to intensity data.
        
        Log scaling helps visualize weak diffuse scattering by compressing
        the dynamic range of the data. Uses log(1 + x*range) transformation
        to avoid log(0) issues.
        """
        if self.log_scale:
            # Apply safe log transform with dynamic range control
            # Using log1p (log(1+x)) to handle zeros gracefully
            scaled = np.log1p(self.intensity_original * self.log_dynamic_range)
            max_log = np.log1p(self.log_dynamic_range)
            self.intensity = (scaled / max_log).astype(np.float32)
            
            print(f"Applied log scaling with dynamic range: {self.log_dynamic_range}")
        else:
            # Use original linear scaling
            self.intensity = self.intensity_original.copy()
        
    def apply_data_masking(self):
        """Apply spherical and/or octant clipping by masking the data."""
        # Apply intensity scaling first (log or linear)
        self.apply_intensity_scaling()
        
        # Create coordinate grids
        x = np.arange(self.h)
        y = np.arange(self.k)
        z = np.arange(self.l)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Apply spherical masking
        dist = np.sqrt(
            (X - self.sphere_center[0])**2 + 
            (Y - self.sphere_center[1])**2 + 
            (Z - self.sphere_center[2])**2
        )
        
        if self.clip_inside:
            # Show only inside sphere
            sphere_mask = dist <= self.sphere_radius
        else:
            # Show only outside sphere
            sphere_mask = dist >= self.sphere_radius
        
        # Apply octant masking if enabled
        if self.octant_cut:
            octant_mask = self.get_octant_mask(X, Y, Z)
            # Invert to EXCLUDE the octant instead of keeping only it
            octant_mask = ~octant_mask  # This excludes the selected octant
            # Combine masks (intersection - both conditions must be true)
            final_mask = sphere_mask & octant_mask
        else:
            final_mask = sphere_mask
        
        # Apply combined mask
        self.intensity[~final_mask] = 0
        
        # Update volume with the masked data
        # Force update by reassigning (k3d sometimes needs this for proper update)
        self.volume.volume = self.intensity.astype(np.float32)
        
        # Also update color range to match the new data range
        valid_data = self.intensity[self.intensity > 0]
        if len(valid_data) > 0:
            self.volume.color_range = [0, np.max(valid_data)]
        
        # Clear clipping planes when using masking
        self.plot.clipping_planes = []
        
        # Debug output
        if self.octant_cut:
            print(f"Octant cutting active: {np.sum(final_mask)} voxels visible out of {final_mask.size}")
    
    def get_octant_mask(self, X, Y, Z):
        """Get mask for octant cutting.
        
        Args:
            X, Y, Z: Coordinate grids
            
        Returns:
            Boolean mask for octant selection
        """
        if self.octant_mode == 'first':
            # First octant: x > center, y > center, z > center
            mask = (X > self.sphere_center[0]) & \
                   (Y > self.sphere_center[1]) & \
                   (Z > self.sphere_center[2])
        elif self.octant_mode == 'custom':
            # Custom octant based on signs
            masks = []
            if self.octant_signs[0] > 0:
                masks.append(X > self.sphere_center[0])
            else:
                masks.append(X <= self.sphere_center[0])
            
            if self.octant_signs[1] > 0:
                masks.append(Y > self.sphere_center[1])
            else:
                masks.append(Y <= self.sphere_center[1])
                
            if self.octant_signs[2] > 0:
                masks.append(Z > self.sphere_center[2])
            else:
                masks.append(Z <= self.sphere_center[2])
            
            mask = masks[0] & masks[1] & masks[2]
        else:
            # No octant cutting
            mask = np.ones_like(X, dtype=bool)
        
        return mask
        
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
    
    def set_octant_params(self, enabled=None, mode=None, signs=None):
        """Set octant cutting parameters.
        
        Args:
            enabled: Boolean to enable/disable octant cutting
            mode: 'first' for x>0,y>0,z>0 or 'custom' for custom signs
            signs: List of [x_sign, y_sign, z_sign] for custom mode (1 or -1)
        """
        if enabled is not None:
            self.octant_cut = enabled
        if mode is not None:
            self.octant_mode = mode
        if signs is not None:
            self.octant_signs = signs
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
        """Run a demonstration of spherical and octant clipping."""
        print("\n" + "="*60)
        print("SPHERICAL & OCTANT CLIPPING DEMONSTRATION")
        print("="*60)
        
        demos = [
            ("Sphere masking - Inside", 'masking', True, False),
            ("Sphere masking - Outside", 'masking', False, False),
            ("Sphere + First octant (x>0, y>0, z>0)", 'masking', True, True),
            ("Hybrid visualization", 'hybrid', True, False),
        ]
        
        for name, method, inside, octant in demos:
            print(f"\n{name}")
            self.method = method
            self.clip_inside = inside
            self.octant_cut = octant
            self.sphere_radius = min(self.h, self.k, self.l) / 3
            self.update_clipping()
            time.sleep(2)
            
            # Animate radius
            self.animate_radius(steps=20, delay=0.03)
            
        # Reset octant cutting
        self.octant_cut = False
        self.update_clipping()
        
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
    "def update_sphere(method, radius, center_x, center_y, center_z, clip_inside, \\n",
    "                   octant_enabled, octant_x_sign, octant_y_sign, octant_z_sign):\\n",
    "    controller.method = method\\n",
    "    controller.sphere_radius = radius\\n",
    "    controller.sphere_center = [center_x, center_y, center_z]\\n",
    "    controller.clip_inside = clip_inside\\n",
    "    controller.octant_cut = octant_enabled\\n",
    "    if octant_enabled:\\n",
    "        controller.octant_mode = 'custom'\\n",
    "        controller.octant_signs = [octant_x_sign, octant_y_sign, octant_z_sign]\\n",
    "    controller.update_clipping()\\n",
    "    \\n",
    "interact(update_sphere,\\n",
    "    method=widgets.RadioButtons(\\n",
    "        options=['masking', 'hybrid'],\\n",
    "        value='masking',\\n",
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
    "    octant_enabled=widgets.Checkbox(\\n",
    "        value=False, description='Enable Octant'\\n",
    "    ),\\n",
    "    octant_x_sign=widgets.RadioButtons(\\n",
    "        options=[1, -1], value=1, description='X sign:'\\n",
    "    ),\\n",
    "    octant_y_sign=widgets.RadioButtons(\\n",
    "        options=[1, -1], value=1, description='Y sign:'\\n",
    "    ),\\n",
    "    octant_z_sign=widgets.RadioButtons(\\n",
    "        options=[1, -1], value=1, description='Z sign:'\\n",
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
        print("SPHERICAL & OCTANT CLIPPING CONTROLLER")
        print("="*60)
        print("\nSphere Commands:")
        print("  r <radius>     - Set sphere radius")
        print("  c <x> <y> <z>  - Set sphere center")
        print("  i/o            - Toggle inside/outside sphere")
        print("\nOctant Commands:")
        print("  oct            - Toggle first octant (x>0,y>0,z>0)")
        print("  oct off        - Disable octant cutting")
        print("  oct <sx sy sz> - Custom octant (signs: 1 or -1)")
        print("\nOther Commands:")
        print("  m <method>     - Set method (masking/hybrid)")
        print("  a              - Animate radius")
        print("  d              - Run demo")
        print("  q              - Quit")
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
                elif cmd[0] == 'oct':
                    if len(cmd) == 1:
                        # Toggle first octant
                        controller.octant_cut = not controller.octant_cut
                        controller.octant_mode = 'first'
                        controller.update_clipping()
                        status = "enabled" if controller.octant_cut else "disabled"
                        print(f"First octant (x>0,y>0,z>0) {status}")
                    elif len(cmd) == 2 and cmd[1] == 'off':
                        controller.octant_cut = False
                        controller.update_clipping()
                        print("Octant cutting disabled")
                    elif len(cmd) == 4:
                        # Custom octant with signs
                        try:
                            signs = [int(cmd[1]), int(cmd[2]), int(cmd[3])]
                            controller.octant_cut = True
                            controller.octant_mode = 'custom'
                            controller.octant_signs = signs
                            controller.update_clipping()
                            print(f"Custom octant set with signs {signs}")
                        except ValueError:
                            print("Invalid signs. Use 1 or -1 for each axis")
                elif cmd[0] == 'm' and len(cmd) > 1:
                    if cmd[1] in ['masking', 'hybrid']:
                        controller.method = cmd[1]
                        controller.update_clipping()
                        print(f"Method set to {controller.method}")
                    else:
                        print("Method must be 'masking' or 'hybrid' (polyhedron is deprecated)")
                elif cmd[0] == 'a':
                    controller.animate_radius()
                elif cmd[0] == 'd':
                    controller.demo()
                else:
                    print("Unknown command")
                    
            except Exception as e:
                print(f"Error: {e}")
