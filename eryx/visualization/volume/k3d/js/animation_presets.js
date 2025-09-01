/**
 * K3D Animation Presets
 * Ready-to-use animation configurations
 */

const K3DAnimationPresets = {
    /**
     * Simple orbital rotation
     */
    orbit360: {
        type: 'orbital',
        params: {
            duration: 5000,
            radius: 6,
            elevation: 30,
            startAngle: 0,
            endAngle: 2 * Math.PI,
            easing: 'easeInOut',
            fps: 60
        }
    },
    
    /**
     * Zoom in dramatically
     */
    zoomIn: {
        type: 'zoom',
        params: {
            duration: 3000,
            startDistance: 10,
            endDistance: 3,
            azimuth: 45,
            elevation: 30,
            easing: 'easeInOut',
            fps: 60
        }
    },
    
    /**
     * Zoom out smoothly
     */
    zoomOut: {
        type: 'zoom',
        params: {
            duration: 3000,
            startDistance: 3,
            endDistance: 10,
            azimuth: 45,
            elevation: 30,
            easing: 'easeInOut',
            fps: 60
        }
    },
    
    /**
     * Sweep X-axis clipping plane
     */
    sweepX: {
        type: 'sweep',
        params: {
            duration: 4000,
            axis: 'x',
            startPos: -2,
            endPos: 2,
            easing: 'linear',
            fps: 60
        }
    },
    
    /**
     * Sweep Y-axis clipping plane
     */
    sweepY: {
        type: 'sweep',
        params: {
            duration: 4000,
            axis: 'y',
            startPos: -2,
            endPos: 2,
            easing: 'linear',
            fps: 60
        }
    },
    
    /**
     * Sweep Z-axis clipping plane
     */
    sweepZ: {
        type: 'sweep',
        params: {
            duration: 4000,
            axis: 'z',
            startPos: -2,
            endPos: 2,
            easing: 'linear',
            fps: 60
        }
    },
    
    /**
     * Combined orbit with X-sweep
     */
    orbitWithSweep: {
        type: 'combined',
        params: {
            duration: 6000,
            radius: 6,
            elevation: 30,
            sweepAxis: 'x',
            sweepRange: [-2, 2],
            easing: 'easeInOut',
            fps: 60
        }
    },
    
    /**
     * Cinematic reveal - zoom in with sweep
     */
    cinematicReveal: {
        sequence: [
            {
                type: 'zoom',
                params: {
                    duration: 2000,
                    startDistance: 12,
                    endDistance: 8,
                    azimuth: 45,
                    elevation: 35,
                    easing: 'easeOut'
                }
            },
            {
                type: 'combined',
                params: {
                    duration: 4000,
                    radius: 8,
                    elevation: 35,
                    sweepAxis: 'x',
                    sweepRange: [-2, 0],
                    easing: 'easeInOut'
                }
            },
            {
                type: 'orbital',
                params: {
                    duration: 3000,
                    radius: 6,
                    elevation: 30,
                    startAngle: 0,
                    endAngle: Math.PI,
                    easing: 'easeInOut'
                }
            }
        ]
    },
    
    /**
     * Inspection mode - slow orbit at multiple elevations
     */
    inspect: {
        sequence: [
            {
                type: 'orbital',
                params: {
                    duration: 4000,
                    radius: 7,
                    elevation: 0,
                    startAngle: 0,
                    endAngle: 2 * Math.PI,
                    easing: 'linear'
                }
            },
            {
                type: 'orbital',
                params: {
                    duration: 4000,
                    radius: 7,
                    elevation: 45,
                    startAngle: 0,
                    endAngle: 2 * Math.PI,
                    easing: 'linear'
                }
            },
            {
                type: 'orbital',
                params: {
                    duration: 4000,
                    radius: 7,
                    elevation: 90,
                    startAngle: 0,
                    endAngle: Math.PI,
                    easing: 'linear'
                }
            }
        ]
    },
    
    /**
     * Quick tour - fast overview
     */
    quickTour: {
        sequence: [
            {
                type: 'zoom',
                params: {
                    duration: 1000,
                    startDistance: 10,
                    endDistance: 6,
                    azimuth: 45,
                    elevation: 30,
                    easing: 'easeOut'
                }
            },
            {
                type: 'orbital',
                params: {
                    duration: 2000,
                    radius: 6,
                    elevation: 30,
                    startAngle: 0,
                    endAngle: 2 * Math.PI,
                    easing: 'linear'
                }
            },
            {
                type: 'sweep',
                params: {
                    duration: 1500,
                    axis: 'x',
                    startPos: -2,
                    endPos: 2,
                    easing: 'easeInOut'
                }
            }
        ]
    }
};

/**
 * Helper function to apply preset
 */
function applyPreset(engine, presetName) {
    const preset = K3DAnimationPresets[presetName];
    if (!preset) {
        console.error(`Preset '${presetName}' not found`);
        return;
    }
    
    engine.clear();
    
    if (preset.sequence) {
        // Multi-step sequence
        preset.sequence.forEach(step => {
            engine.addAnimation(step.type, step.params);
        });
    } else {
        // Single animation
        engine.addAnimation(preset.type, preset.params);
    }
    
    return engine;
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { K3DAnimationPresets, applyPreset };
}

// Make available globally
window.K3DAnimationPresets = K3DAnimationPresets;
window.applyPreset = applyPreset;