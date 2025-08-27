"""
Animation generator for progressive slicing visualizations.

This module provides the SlicingAnimator class for creating animated GIFs
and videos of 3D volumes being progressively sliced.
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio
from typing import List, Optional, Union, Dict, Any, Tuple
from pathlib import Path
import logging
import tempfile
import shutil
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SlicingAnimator:
    """
    Generate animated visualizations of progressive slicing.
    
    This class coordinates the plane calculation, slicing, and rendering
    to create smooth animations of 3D data being cut by moving planes.
    """
    
    BACKENDS = {
        'matplotlib': 'matplotlib_slicer.MatplotlibSlicer',
    }
    
    def __init__(self, volume: np.ndarray, 
                coordinates: Optional[np.ndarray] = None,
                backend: str = 'matplotlib'):
        """
        Initialize the animator.
        
        Parameters
        ----------
        volume : np.ndarray, shape (nx, ny, nz)
            3D intensity data
        coordinates : np.ndarray, optional
            Real-space coordinates for each voxel
        backend : str
            Which rendering backend to use ('matplotlib', etc.)
        """
        self.volume = np.asarray(volume)
        self.coordinates = coordinates
        self.backend = backend
        
        # Initialize the slicer
        self._init_slicer()
        
        # Cache for rendered frames
        self.frames = []
        
        logger.info(f"SlicingAnimator initialized with {backend} backend, "
                   f"volume shape {self.volume.shape}")
    
    def _init_slicer(self):
        """Initialize the appropriate slicer backend."""
        if self.backend == 'matplotlib':
            from .matplotlib_slicer import MatplotlibSlicer
            self.slicer = MatplotlibSlicer(self.volume, self.coordinates)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def generate_animation(self, normal: Union[np.ndarray, List[float]], 
                         n_frames: int = 50,
                         fps: int = 10,
                         output_path: Optional[str] = None,
                         **kwargs) -> Union[str, List[np.ndarray]]:
        """
        Generate a slicing animation.
        
        Parameters
        ----------
        normal : array-like, shape (3,)
            Normal vector for the cutting planes
        n_frames : int
            Number of frames in the animation
        fps : int
            Frames per second for the output
        output_path : str, optional
            Path to save the animation. If None, returns frames.
        **kwargs : dict
            Additional parameters for rendering:
            - view_angles: (elevation, azimuth) in degrees
            - colormap: matplotlib colormap name
            - show_plane: whether to show the cutting plane
            - show_volume: whether to show the volume
            - subsample: subsampling factor for performance
            - figsize: figure size in inches
            - dpi: figure resolution
            - title_format: format string for frame titles
            
        Returns
        -------
        result : str or list
            Output path if saved, otherwise list of frame arrays
        """
        from .plane_calculator import PlaneCalculator
        
        # Set default parameters
        params = {
            'view_angles': (30, 45),
            'colormap': 'viridis',
            'show_plane': True,
            'show_volume': True,
            'subsample': 4,
            'figsize': (10, 8),
            'dpi': 100,
            'title_format': 'Frame {frame}/{total}',
            'voxel_size': 1,
            'volume_alpha': 0.6,
            'plane_alpha': 0.3,
            'slice_alpha': 0.9,
            'show_slice': True,
            'colorbar': True
        }
        params.update(kwargs)
        
        # Create plane sequence
        logger.info(f"Creating sequence of {n_frames} cutting planes")
        calculator = PlaneCalculator(self.volume.shape)
        planes = calculator.create_cutting_sequence(normal, n_frames)
        
        # Generate frames
        frames = []
        logger.info("Generating animation frames...")
        
        with tqdm(total=n_frames, desc="Rendering frames") as pbar:
            for i, plane in enumerate(planes):
                # Update title
                if params['title_format']:
                    params['title'] = params['title_format'].format(
                        frame=i+1, total=n_frames
                    )
                
                # Create frame
                fig = self.slicer.create_frame(plane, **params)
                
                # Convert to image array
                fig.canvas.draw()
                # Use buffer_rgba for newer matplotlib versions
                try:
                    # Try the new method first (matplotlib 3.8+)
                    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    width, height = fig.canvas.get_width_height()
                    image = image.reshape(height, width, 4)[:, :, :3]  # Drop alpha channel
                except AttributeError:
                    # Fallback to old method for older matplotlib versions
                    try:
                        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        width, height = fig.canvas.get_width_height()
                        image = image.reshape(height, width, 3)
                    except AttributeError:
                        # Last resort fallback
                        image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
                        width, height = fig.canvas.get_width_height()
                        image = image.reshape(height, width, 4)[:, :, 1:4]  # Drop alpha, get RGB
                frames.append(image)
                
                # Clean up
                plt.close(fig)
                pbar.update(1)
        
        self.frames = frames
        logger.info(f"Generated {len(frames)} frames")
        
        # Save or return frames
        if output_path:
            return self.export(output_path, fps=fps)
        else:
            return frames
    
    def export(self, output_path: str, fps: int = 10, 
              optimize: bool = True) -> str:
        """
        Export animation to file.
        
        Parameters
        ----------
        output_path : str
            Output file path (.gif, .mp4, .avi, etc.)
        fps : int
            Frames per second
        optimize : bool
            Whether to optimize the output file
            
        Returns
        -------
        output_path : str
            Path to the saved file
        """
        if not self.frames:
            raise ValueError("No frames to export. Run generate_animation first.")
        
        output_path = Path(output_path)
        suffix = output_path.suffix.lower()
        
        logger.info(f"Exporting {len(self.frames)} frames to {output_path}")
        
        if suffix == '.gif':
            self._export_gif(output_path, fps, optimize)
        elif suffix in ['.mp4', '.avi', '.mov', '.webm']:
            self._export_video(output_path, fps)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
        
        logger.info(f"Animation saved to {output_path}")
        return str(output_path)
    
    def _export_gif(self, output_path: Path, fps: int, optimize: bool):
        """Export as animated GIF."""
        # Calculate duration per frame in milliseconds
        duration = 1000 / fps
        
        # Save using imageio
        with imageio.get_writer(output_path, mode='I', duration=duration) as writer:
            for frame in self.frames:
                writer.append_data(frame)
        
        if optimize and output_path.exists():
            # Optimize file size
            original_size = output_path.stat().st_size
            logger.debug(f"Original GIF size: {original_size / 1024:.1f} KB")
            
            # Could add additional optimization here if needed
            # (e.g., using gifsicle or pillow optimization)
    
    def _export_video(self, output_path: Path, fps: int):
        """Export as video file."""
        try:
            # Try using imageio-ffmpeg if available
            with imageio.get_writer(output_path, fps=fps) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
        except Exception as e:
            logger.warning(f"Video export failed: {e}")
            logger.info("Falling back to GIF format")
            # Fallback to GIF
            gif_path = output_path.with_suffix('.gif')
            self._export_gif(gif_path, fps, optimize=True)
            logger.info(f"Saved as GIF instead: {gif_path}")
    
    def create_comparison(self, normals: List[np.ndarray],
                        n_frames: int = 30,
                        output_path: Optional[str] = None,
                        **kwargs) -> Union[str, plt.Figure]:
        """
        Create a comparison of multiple slicing directions.
        
        Parameters
        ----------
        normals : list of arrays
            List of normal vectors to compare
        n_frames : int
            Number of frames per normal
        output_path : str, optional
            Path to save the comparison
        **kwargs : dict
            Additional rendering parameters
            
        Returns
        -------
        result : str or Figure
            Output path if saved, otherwise matplotlib figure
        """
        from .plane_calculator import PlaneCalculator
        
        n_normals = len(normals)
        fig, axes = plt.subplots(1, n_normals, figsize=(5*n_normals, 5),
                                subplot_kw={'projection': '3d'})
        
        if n_normals == 1:
            axes = [axes]
        
        calculator = PlaneCalculator(self.volume.shape)
        
        # Use middle frame for each normal
        frame_idx = n_frames // 2
        
        for ax, normal in zip(axes, normals):
            # Get plane for middle frame
            planes = calculator.create_cutting_sequence(normal, n_frames)
            plane = planes[frame_idx]
            
            # Simplified rendering for comparison
            distances = calculator.get_voxel_distances(plane)
            below_mask = distances <= 0
            
            # Subsample for visualization
            subsample = kwargs.get('subsample', 8)
            x = self.slicer.x_grid[::subsample, ::subsample, ::subsample]
            y = self.slicer.y_grid[::subsample, ::subsample, ::subsample]
            z = self.slicer.z_grid[::subsample, ::subsample, ::subsample]
            v = self.volume[::subsample, ::subsample, ::subsample]
            m = below_mask[::subsample, ::subsample, ::subsample]
            
            # Plot
            valid = m & ~np.isnan(v)
            if np.any(valid):
                scatter = ax.scatter(x[valid], y[valid], z[valid],
                                   c=v[valid], cmap='viridis',
                                   s=1, alpha=0.6)
            
            # Set title
            ax.set_title(f"Normal: [{normal[0]:.1f}, {normal[1]:.1f}, {normal[2]:.1f}]")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150)
            plt.close()
            return output_path
        else:
            return fig


def create_slicing_animation(data_path: str,
                            normal: Union[np.ndarray, List[float]] = [1, 1, 1],
                            n_frames: int = 50,
                            output_path: str = 'slicing_animation.gif',
                            nan_strategy: str = 'mask',
                            **kwargs) -> str:
    """
    High-level function to create a slicing animation from data file.
    
    Parameters
    ----------
    data_path : str
        Path to NPZ file with intensity data
    normal : array-like
        Normal vector for cutting planes
    n_frames : int
        Number of animation frames
    output_path : str
        Output file path
    nan_strategy : str
        How to handle NaN values
    **kwargs : dict
        Additional parameters for rendering
        
    Returns
    -------
    output_path : str
        Path to saved animation
    """
    from ..core import IntensityDataHandler, NaNProcessor
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    handler = IntensityDataHandler(data_path)
    q_vectors, intensity, map_shape = handler.load_data()
    
    if intensity is None:
        raise ValueError(f"Could not load data from {data_path}")
    
    # Reshape if needed
    if intensity.ndim == 1 and map_shape is not None:
        handler.reshape_to_3d()
        intensity = handler.intensity
    
    # Handle NaN values
    logger.info(f"Processing NaN values with strategy '{nan_strategy}'")
    intensity = NaNProcessor.apply_strategy(intensity, nan_strategy)
    
    # Create animation
    logger.info("Creating animation")
    animator = SlicingAnimator(intensity, coordinates=q_vectors)
    result = animator.generate_animation(normal, n_frames=n_frames,
                                        output_path=output_path,
                                        fps=kwargs.pop('fps', 10),
                                        **kwargs)
    
    logger.info(f"Animation complete: {result}")
    return result