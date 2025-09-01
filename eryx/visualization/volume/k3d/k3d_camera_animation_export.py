#!/usr/bin/env python3
"""
K3D Camera Animation Export - File-based rendering to avoid in-notebook glitches.
This creates smooth camera animations by exporting individual frames as HTML files.
"""

import k3d
import numpy as np
import math
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


class K3DCameraAnimator:
    """Create and export smooth K3D camera animations to files."""
    
    def __init__(self, plot: k3d.Plot):
        """Initialize with an existing k3d plot.
        
        Args:
            plot: K3D plot object with volume data already added
        """
        self.plot = plot
        self.initial_camera = plot.camera.copy() if plot.camera else None
        
    def create_orbital_animation(
        self,
        center: Optional[List[float]] = None,
        radius: float = 100.0,
        elevation: float = 30.0,
        num_frames: int = 60,
        start_angle: float = 0.0,
        end_angle: float = 360.0
    ) -> List[List[float]]:
        """Create orbital camera path around a center point.
        
        Args:
            center: Center point to orbit around [x, y, z]. If None, uses data center.
            radius: Distance from center
            elevation: Camera elevation angle in degrees (0=horizontal, 90=top)
            num_frames: Number of frames to generate
            start_angle: Starting azimuth angle in degrees
            end_angle: Ending azimuth angle in degrees (360 for full orbit)
            
        Returns:
            List of camera arrays [position + target + up]
        """
        if center is None:
            # Calculate data center from bounds
            bounds = self.plot.objects[0].transform.bounds if self.plot.objects else [0,1,0,1,0,1]
            center = [
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            ]
        
        cameras = []
        phi = math.radians(elevation)
        
        for i in range(num_frames):
            progress = i / (num_frames - 1) if num_frames > 1 else 0
            theta = math.radians(start_angle + (end_angle - start_angle) * progress)
            
            # Spherical to Cartesian
            x = radius * math.sin(phi) * math.cos(theta)
            y = radius * math.sin(phi) * math.sin(theta)
            z = radius * math.cos(phi)
            
            camera = [
                x + center[0], y + center[1], z + center[2],  # Position
                center[0], center[1], center[2],              # Target
                0, 0, 1                                       # Up vector
            ]
            cameras.append(camera)
            
        return cameras
    
    def create_zoom_animation(
        self,
        start_distance: float = 150.0,
        end_distance: float = 50.0,
        center: Optional[List[float]] = None,
        azimuth: float = 45.0,
        elevation: float = 30.0,
        num_frames: int = 30
    ) -> List[List[float]]:
        """Create zoom in/out animation.
        
        Args:
            start_distance: Starting distance from center
            end_distance: Ending distance from center
            center: Target point. If None, uses data center
            azimuth: Fixed azimuth angle in degrees
            elevation: Fixed elevation angle in degrees
            num_frames: Number of frames
            
        Returns:
            List of camera arrays
        """
        if center is None:
            bounds = self.plot.objects[0].transform.bounds if self.plot.objects else [0,1,0,1,0,1]
            center = [
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            ]
        
        cameras = []
        theta = math.radians(azimuth)
        phi = math.radians(elevation)
        
        for i in range(num_frames):
            progress = i / (num_frames - 1) if num_frames > 1 else 0
            # Smooth ease-in-out interpolation
            smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))
            distance = start_distance + (end_distance - start_distance) * smooth_progress
            
            x = distance * math.sin(phi) * math.cos(theta)
            y = distance * math.sin(phi) * math.sin(theta)
            z = distance * math.cos(phi)
            
            camera = [
                x + center[0], y + center[1], z + center[2],  # Position
                center[0], center[1], center[2],              # Target
                0, 0, 1                                       # Up vector
            ]
            cameras.append(camera)
            
        return cameras
    
    def create_spiral_animation(
        self,
        center: Optional[List[float]] = None,
        radius: float = 100.0,
        height_range: Tuple[float, float] = (20.0, 80.0),
        num_orbits: float = 2.0,
        num_frames: int = 90
    ) -> List[List[float]]:
        """Create spiral camera path (useful for full coverage).
        
        Args:
            center: Center point
            radius: Spiral radius
            height_range: (min_z, max_z) for vertical movement
            num_orbits: Number of complete orbits
            num_frames: Number of frames
            
        Returns:
            List of camera arrays
        """
        if center is None:
            bounds = self.plot.objects[0].transform.bounds if self.plot.objects else [0,1,0,1,0,1]
            center = [
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            ]
        
        cameras = []
        min_z, max_z = height_range
        
        for i in range(num_frames):
            progress = i / (num_frames - 1) if num_frames > 1 else 0
            theta = 2 * math.pi * num_orbits * progress
            
            # Vary height sinusoidally
            z = min_z + (max_z - min_z) * (0.5 + 0.5 * math.sin(2 * math.pi * progress))
            
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            
            camera = [
                x + center[0], y + center[1], z + center[2],  # Position
                center[0], center[1], center[2],              # Target
                0, 0, 1                                       # Up vector
            ]
            cameras.append(camera)
            
        return cameras
    
    def create_turntable_animation(
        self,
        axis: str = 'z',
        num_frames: int = 60
    ) -> List[List[float]]:
        """Create simple turntable rotation around an axis.
        
        Args:
            axis: Rotation axis ('x', 'y', or 'z')
            num_frames: Number of frames
            
        Returns:
            List of camera arrays
        """
        # Get current camera position
        if self.initial_camera:
            pos = self.initial_camera[:3]
            target = self.initial_camera[3:6]
        else:
            pos = [100, 100, 50]
            target = [0, 0, 0]
        
        cameras = []
        
        for i in range(num_frames):
            progress = i / (num_frames - 1) if num_frames > 1 else 0
            angle = 2 * math.pi * progress
            
            # Calculate rotated position
            if axis == 'z':
                x = pos[0] * math.cos(angle) - pos[1] * math.sin(angle)
                y = pos[0] * math.sin(angle) + pos[1] * math.cos(angle)
                z = pos[2]
            elif axis == 'y':
                x = pos[0] * math.cos(angle) + pos[2] * math.sin(angle)
                y = pos[1]
                z = -pos[0] * math.sin(angle) + pos[2] * math.cos(angle)
            else:  # axis == 'x'
                x = pos[0]
                y = pos[1] * math.cos(angle) - pos[2] * math.sin(angle)
                z = pos[1] * math.sin(angle) + pos[2] * math.cos(angle)
            
            camera = [
                x, y, z,                    # Position
                target[0], target[1], target[2],  # Target
                0, 0, 1                     # Up vector
            ]
            cameras.append(camera)
            
        return cameras
    
    def export_animation(
        self,
        cameras: List[List[float]],
        output_dir: str = "k3d_animation_frames",
        prefix: str = "frame",
        verbose: bool = True
    ) -> List[Path]:
        """Export camera animation as individual HTML frames.
        
        Args:
            cameras: List of camera arrays from animation methods
            output_dir: Directory to save frames
            prefix: Filename prefix
            verbose: Print progress
            
        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Disable auto-fit to maintain consistent framing
        self.plot.camera_auto_fit = False
        
        frame_files = []
        num_frames = len(cameras)
        
        for i, camera in enumerate(cameras):
            # Set camera position
            self.plot.camera = camera
            
            # Export frame
            frame_file = output_path / f"{prefix}_{i:04d}.html"
            with open(frame_file, 'w') as f:
                f.write(self.plot.get_snapshot())
            
            frame_files.append(frame_file)
            
            if verbose and (i % 10 == 0 or i == num_frames - 1):
                print(f"Exported frame {i+1}/{num_frames}: {frame_file.name}")
        
        if verbose:
            print(f"\n✅ Exported {num_frames} frames to {output_path}/")
            print("To create video: ffmpeg -framerate 30 -pattern_type glob -i '*.html' output.mp4")
            print("(Note: Requires converting HTML to images first)")
        
        return frame_files
    
    def create_combined_animation(
        self,
        animations: List[Dict[str, Any]],
        transition_frames: int = 10
    ) -> List[List[float]]:
        """Combine multiple animation sequences with smooth transitions.
        
        Args:
            animations: List of dicts with 'type' and 'params' keys
            transition_frames: Number of frames for transitions between animations
            
        Returns:
            Combined camera array list
        """
        all_cameras = []
        
        for i, anim in enumerate(animations):
            anim_type = anim['type']
            params = anim.get('params', {})
            
            # Generate animation based on type
            if anim_type == 'orbital':
                cameras = self.create_orbital_animation(**params)
            elif anim_type == 'zoom':
                cameras = self.create_zoom_animation(**params)
            elif anim_type == 'spiral':
                cameras = self.create_spiral_animation(**params)
            elif anim_type == 'turntable':
                cameras = self.create_turntable_animation(**params)
            else:
                raise ValueError(f"Unknown animation type: {anim_type}")
            
            # Add transition if not first animation
            if i > 0 and transition_frames > 0:
                prev_camera = all_cameras[-1]
                next_camera = cameras[0]
                
                for t in range(transition_frames):
                    progress = (t + 1) / (transition_frames + 1)
                    smooth_progress = 0.5 * (1 - math.cos(math.pi * progress))
                    
                    # Interpolate between cameras
                    interp_camera = [
                        prev_camera[j] + (next_camera[j] - prev_camera[j]) * smooth_progress
                        for j in range(9)
                    ]
                    all_cameras.append(interp_camera)
            
            all_cameras.extend(cameras)
        
        return all_cameras


def demo_camera_animations():
    """Demonstrate various camera animation exports."""
    from eryx.visualization.volume.k3d.final_k3d_clipping_solution import (
        load_and_clean_data, create_production_visualization
    )
    
    # Load data and create visualization
    print("Loading data...")
    volume_data = load_and_clean_data('torch')
    plot, volume = create_production_visualization(volume_data)
    
    # Add clipping plane after creation
    plot.clipping_planes = [[1, 0, 0, 0]]  # X-plane clipping
    
    # Create animator
    animator = K3DCameraAnimator(plot)
    
    # Example 1: Simple orbital animation
    print("\n📹 Creating orbital animation...")
    orbital_cameras = animator.create_orbital_animation(
        radius=120,
        elevation=35,
        num_frames=60
    )
    animator.export_animation(
        orbital_cameras,
        output_dir="camera_orbital",
        prefix="orbit"
    )
    
    # Example 2: Zoom animation
    print("\n📹 Creating zoom animation...")
    zoom_cameras = animator.create_zoom_animation(
        start_distance=150,
        end_distance=60,
        azimuth=45,
        elevation=30,
        num_frames=30
    )
    animator.export_animation(
        zoom_cameras,
        output_dir="camera_zoom",
        prefix="zoom"
    )
    
    # Example 3: Combined animation sequence
    print("\n📹 Creating combined animation...")
    combined_sequence = [
        {'type': 'zoom', 'params': {'start_distance': 150, 'end_distance': 100, 'num_frames': 20}},
        {'type': 'orbital', 'params': {'radius': 100, 'elevation': 30, 'num_frames': 40}},
        {'type': 'spiral', 'params': {'radius': 90, 'num_orbits': 1.5, 'num_frames': 45}},
        {'type': 'zoom', 'params': {'start_distance': 90, 'end_distance': 130, 'num_frames': 20}}
    ]
    
    combined_cameras = animator.create_combined_animation(
        combined_sequence,
        transition_frames=10
    )
    animator.export_animation(
        combined_cameras,
        output_dir="camera_combined",
        prefix="combined"
    )
    
    print("\n✅ All animations exported successfully!")
    print("\nNext steps:")
    print("1. Open HTML files in browser to view frames")
    print("2. Use browser automation or screen capture to create images")
    print("3. Convert to video with: ffmpeg -framerate 30 -i 'frame_%04d.png' output.mp4")


if __name__ == "__main__":
    demo_camera_animations()