#!/usr/bin/env python3
"""
K3D JavaScript Animator - Python Controller Interface
Clean Python API for controlling JavaScript-based K3D animations
"""

import k3d
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from IPython.display import Javascript, display
import numpy as np


class K3DJSAnimator:
    """Python controller for JavaScript-based K3D animations."""
    
    def __init__(self, plot: k3d.Plot, auto_inject: bool = True):
        """Initialize the animator with a K3D plot.
        
        Args:
            plot: K3D plot instance
            auto_inject: Whether to automatically inject JavaScript engine
        """
        self.plot = plot
        self.animation_queue = []
        self.js_dir = Path(__file__).parent / 'js'
        
        # Load JavaScript code
        self.engine_js = self._load_js_file('k3d_animation_engine.js')
        self.presets_js = self._load_js_file('animation_presets.js')
        
        if auto_inject:
            self.inject_engine()
    
    def _load_js_file(self, filename: str) -> str:
        """Load JavaScript file content.
        
        Args:
            filename: Name of JS file in js/ directory
            
        Returns:
            JavaScript code as string
        """
        filepath = self.js_dir / filename
        if not filepath.exists():
            # Fallback to embedded code if file not found
            return self._get_embedded_js(filename)
        
        with open(filepath, 'r') as f:
            return f.read()
    
    def _get_embedded_js(self, filename: str) -> str:
        """Get embedded JavaScript code as fallback.
        
        Args:
            filename: Name of JS file
            
        Returns:
            Embedded JavaScript code
        """
        if filename == 'k3d_animation_engine.js':
            return self._get_embedded_engine_js()
        elif filename == 'animation_presets.js':
            return self._get_embedded_presets_js()
        return ""
    
    def inject_engine(self, force: bool = False) -> None:
        """Inject animation engine into K3D plot.
        
        Args:
            force: Force re-injection even if already injected
        """
        # Check if already injected
        check_js = """
        if (typeof window.k3dAnimatorInjected === 'undefined' || %s) {
            window.k3dAnimatorInjected = false;
        }
        window.k3dAnimatorInjected;
        """ % str(force).lower()
        
        # Prepare injection code
        injection_js = f"""
        // K3D JS Animator Injection
        (function() {{
            if (window.k3dAnimatorInjected && !{str(force).lower()}) {{
                console.log('K3D Animator already injected');
                return;
            }}
            
            // Wait for K3DInstance to be available
            function waitForK3D() {{
                if (typeof K3DInstance !== 'undefined') {{
                    // Inject engine code
                    {self.engine_js}
                    
                    // Inject presets code
                    {self.presets_js}
                    
                    // Initialize engine
                    if (typeof K3DAnimationEngine !== 'undefined') {{
                        window.animationEngine = new K3DAnimationEngine(K3DInstance);
                        window.k3dAnimatorInjected = true;
                        console.log('K3D Animation Engine injected and initialized');
                    }} else {{
                        console.error('K3DAnimationEngine class not found');
                    }}
                }} else {{
                    console.log('Waiting for K3DInstance...');
                    setTimeout(waitForK3D, 100);
                }}
            }}
            
            waitForK3D();
        }})();
        """
        
        # Inject via plot's additional_js_code
        if hasattr(self.plot, 'additional_js_code'):
            existing_code = self.plot.additional_js_code or ""
            self.plot.additional_js_code = existing_code + "\n" + injection_js
        
        # Also inject directly for immediate effect
        display(Javascript(injection_js))
    
    def add_orbital(self, 
                   duration: float = 5.0,
                   radius: float = 6.0,
                   elevation: float = 30.0,
                   center: List[float] = None,
                   start_angle: float = 0.0,
                   end_angle: float = 360.0,
                   easing: str = 'easeInOut',
                   fps: int = 60) -> 'K3DJSAnimator':
        """Add orbital camera animation to queue.
        
        Args:
            duration: Animation duration in seconds
            radius: Orbital radius
            elevation: Camera elevation in degrees
            center: Center point [x, y, z]
            start_angle: Starting angle in degrees
            end_angle: Ending angle in degrees
            easing: Easing function name
            fps: Frames per second
            
        Returns:
            Self for chaining
        """
        if center is None:
            center = [0, 0, 0]
        
        params = {
            'duration': duration * 1000,  # Convert to milliseconds
            'radius': radius,
            'elevation': elevation,
            'center': center,
            'startAngle': np.radians(start_angle),
            'endAngle': np.radians(end_angle),
            'easing': easing,
            'fps': fps
        }
        
        self.animation_queue.append({
            'type': 'orbital',
            'params': params
        })
        
        return self
    
    def add_zoom(self,
                start_distance: float = 10.0,
                end_distance: float = 3.0,
                duration: float = 3.0,
                azimuth: float = 45.0,
                elevation: float = 30.0,
                center: List[float] = None,
                easing: str = 'easeInOut',
                fps: int = 60) -> 'K3DJSAnimator':
        """Add zoom animation to queue.
        
        Args:
            start_distance: Starting camera distance
            end_distance: Ending camera distance
            duration: Animation duration in seconds
            azimuth: Camera azimuth in degrees
            elevation: Camera elevation in degrees
            center: Target point [x, y, z]
            easing: Easing function name
            fps: Frames per second
            
        Returns:
            Self for chaining
        """
        if center is None:
            center = [0, 0, 0]
        
        params = {
            'duration': duration * 1000,
            'startDistance': start_distance,
            'endDistance': end_distance,
            'azimuth': azimuth,
            'elevation': elevation,
            'center': center,
            'easing': easing,
            'fps': fps
        }
        
        self.animation_queue.append({
            'type': 'zoom',
            'params': params
        })
        
        return self
    
    def add_sweep(self,
                 axis: str = 'x',
                 range: Tuple[float, float] = (-2.0, 2.0),
                 duration: float = 4.0,
                 easing: str = 'linear',
                 fps: int = 60) -> 'K3DJSAnimator':
        """Add clipping plane sweep animation to queue.
        
        Args:
            axis: Axis to sweep ('x', 'y', or 'z')
            range: (start, end) positions for sweep
            duration: Animation duration in seconds
            easing: Easing function name
            fps: Frames per second
            
        Returns:
            Self for chaining
        """
        params = {
            'duration': duration * 1000,
            'axis': axis,
            'startPos': range[0],
            'endPos': range[1],
            'easing': easing,
            'fps': fps
        }
        
        self.animation_queue.append({
            'type': 'sweep',
            'params': params
        })
        
        return self
    
    def add_combined(self,
                    duration: float = 6.0,
                    radius: float = 6.0,
                    elevation: float = 30.0,
                    center: List[float] = None,
                    sweep_axis: str = 'x',
                    sweep_range: Tuple[float, float] = (-2.0, 2.0),
                    easing: str = 'easeInOut',
                    fps: int = 60) -> 'K3DJSAnimator':
        """Add combined camera + clipping animation to queue.
        
        Args:
            duration: Animation duration in seconds
            radius: Orbital radius
            elevation: Camera elevation in degrees
            center: Center point [x, y, z]
            sweep_axis: Clipping plane axis
            sweep_range: Clipping plane range
            easing: Easing function name
            fps: Frames per second
            
        Returns:
            Self for chaining
        """
        if center is None:
            center = [0, 0, 0]
        
        params = {
            'duration': duration * 1000,
            'radius': radius,
            'elevation': elevation,
            'center': center,
            'sweepAxis': sweep_axis,
            'sweepRange': list(sweep_range),
            'easing': easing,
            'fps': fps
        }
        
        self.animation_queue.append({
            'type': 'combined',
            'params': params
        })
        
        return self
    
    def add_preset(self, preset_name: str) -> 'K3DJSAnimator':
        """Add a preset animation to queue.
        
        Args:
            preset_name: Name of preset (e.g., 'orbit360', 'zoomIn', 'cinematicReveal')
            
        Returns:
            Self for chaining
        """
        # This will be applied via JavaScript
        self.animation_queue.append({
            'type': 'preset',
            'name': preset_name
        })
        
        return self
    
    def _generate_js_commands(self) -> str:
        """Generate JavaScript commands from animation queue.
        
        Returns:
            JavaScript code string
        """
        js_commands = []
        
        # Clear existing animations
        js_commands.append("window.animationEngine.clear();")
        
        # Add each animation
        for anim in self.animation_queue:
            if anim['type'] == 'preset':
                js_commands.append(f"applyPreset(window.animationEngine, '{anim['name']}');")
            else:
                params_json = json.dumps(anim['params'])
                js_commands.append(f"window.animationEngine.addAnimation('{anim['type']}', {params_json});")
        
        return "\n".join(js_commands)
    
    def play(self) -> None:
        """Start animation playback."""
        js_code = f"""
        (function() {{
            if (typeof window.animationEngine === 'undefined') {{
                console.error('Animation engine not initialized. Call inject_engine() first.');
                return;
            }}
            
            {self._generate_js_commands()}
            
            window.animationEngine.play();
            console.log('Animation started');
        }})();
        """
        display(Javascript(js_code))
    
    def pause(self) -> None:
        """Pause animation playback."""
        js_code = """
        if (window.animationEngine) {
            window.animationEngine.pause();
            console.log('Animation paused');
        }
        """
        display(Javascript(js_code))
    
    def stop(self) -> None:
        """Stop animation playback."""
        js_code = """
        if (window.animationEngine) {
            window.animationEngine.stop();
            console.log('Animation stopped');
        }
        """
        display(Javascript(js_code))
    
    def reset(self) -> None:
        """Reset to initial state."""
        js_code = """
        if (window.animationEngine) {
            window.animationEngine.reset();
            console.log('Animation reset');
        }
        """
        display(Javascript(js_code))
    
    def clear(self) -> None:
        """Clear animation queue."""
        self.animation_queue = []
        js_code = """
        if (window.animationEngine) {
            window.animationEngine.clear();
            console.log('Animation queue cleared');
        }
        """
        display(Javascript(js_code))
    
    def set_fps(self, fps: int) -> None:
        """Set target frames per second.
        
        Args:
            fps: Target FPS (e.g., 30, 60)
        """
        js_code = f"""
        if (window.animationEngine) {{
            window.animationEngine.fps = {fps};
            window.animationEngine.frameInterval = 1000 / {fps};
            console.log('FPS set to {fps}');
        }}
        """
        display(Javascript(js_code))
    
    def get_state(self) -> None:
        """Get current animation state (logs to console)."""
        js_code = """
        if (window.animationEngine) {
            const state = window.animationEngine.exportState();
            console.log('Animation State:', state);
        }
        """
        display(Javascript(js_code))
    
    def _get_embedded_engine_js(self) -> str:
        """Return embedded animation engine JavaScript."""
        # This would contain the full engine code as a string
        # For brevity, returning a placeholder
        return "// K3DAnimationEngine embedded code"
    
    def _get_embedded_presets_js(self) -> str:
        """Return embedded presets JavaScript."""
        # This would contain the full presets code as a string
        # For brevity, returning a placeholder
        return "// K3DAnimationPresets embedded code"


# Convenience function
def create_animator(plot: k3d.Plot) -> K3DJSAnimator:
    """Create and initialize a K3D JavaScript animator.
    
    Args:
        plot: K3D plot instance
        
    Returns:
        Initialized K3DJSAnimator
    """
    return K3DJSAnimator(plot, auto_inject=True)