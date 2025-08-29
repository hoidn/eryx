# K3D Camera Animation Implementation Plan
**Date**: November 2024  
**Scope**: Integrate camera animations into k3d_spherical_clipping for scientific visualization  
**Priority**: High - Enhances data presentation capabilities

## Overview
Add professional camera animation capabilities to the existing k3d_spherical_clipping visualization system, enabling cinematic presentations of diffuse scattering data with synchronized sphere/octant clipping.

## Phase 1: Core Infrastructure (2 hours)

### 1.1 Coordinate System Utilities
- [x] Add `voxel_to_world()` method to SphericalClippingController
- [x] Add `get_world_sphere_center()` method 
- [x] Add `get_world_bounds()` for camera distance calculations
- [ ] Test coordinate conversions with known values
- [ ] Document coordinate system (voxel indices vs world space)

### 1.2 Camera State Management
- [x] Add `camera_initial` property to store default camera
- [x] Add `reset_camera()` method
- [x] Add `get_optimal_camera_distance()` based on data bounds
- [x] Store animation state flags (is_animating, animation_type)

### Success Criteria
- Coordinate conversions work correctly for 41³ and 81³ data
- Camera distance auto-calculated based on volume bounds
- Can reset to initial view after animation

## Phase 2: Basic Camera Animations (3 hours)

### 2.1 Orbital Animation
- [x] Implement `add_orbital_camera(duration, radius_factor, elevation)`
- [x] Support variable orbital radius
- [x] Add elevation angle parameter (0-90 degrees)
- [x] Generate smooth 60-keyframe paths
- [ ] Test with different durations (3-10 seconds)

### 2.2 Zoom Animation
- [x] Implement `add_zoom_animation(start_distance, end_distance, duration)`
- [x] Support linear and ease-in-out interpolation
- [x] Maintain camera target on sphere center
- [x] Add optional rotation during zoom
- [ ] Test zoom in/out scenarios

### 2.3 Preset Views
- [x] Implement `set_preset_view(view_name)` with options:
  - [x] 'front', 'back', 'left', 'right', 'top', 'bottom'
  - [x] 'isometric' (diagonal view)
  - [x] 'optimal' (auto-calculated best view)
- [x] Add smooth transition between presets
- [x] Store preset definitions in config dict

### Success Criteria
- Smooth orbital rotation at 30+ fps
- Zoom maintains focus on data
- Preset views correctly oriented

## Phase 3: Synchronized Animations (4 hours)

### 3.1 Camera-Sphere Synchronization
- [x] Implement `create_zoom_reveal_animation()`
  - [x] Zoom in as sphere radius decreases
  - [x] Maintain sphere in view throughout
  - [x] Use threading for parallel control
- [x] Add timing synchronization utilities
- [ ] Test with different sphere radii

### 3.2 Octant Tour Animation
- [x] Implement `create_octant_inspection_animation()`
  - [x] Visit 8 octant viewpoints
  - [x] Optional: sync octant exclusion with camera
  - [x] Smooth path between viewpoints
- [x] Add pause at each octant position
- [x] Support custom octant order

### 3.3 Anisotropy Showcase
- [x] Implement `add_anisotropy_showcase_animation()`
  - [x] Multi-elevation orbital paths
  - [x] Variable camera distance for emphasis
  - [x] Highlight directional features
- [x] Add speed variations for interest
- [x] Support custom elevation sequences

### Success Criteria
- Sphere and camera animations synchronized
- Octant tour covers all regions smoothly
- Anisotropy features clearly visible

## Phase 4: Interactive Controls (2 hours)

### 4.1 Jupyter Widget Integration
- [x] Create animation control panel widget
  - [x] Play/Pause button
  - [x] Animation type selector (dropdown)
  - [x] Speed control slider
  - [x] Reset view button
- [x] Integrate with existing spherical clipping controls
- [x] Add to notebook as collapsible section

### 4.2 Animation Parameters Widget
- [x] Duration slider (1-20 seconds)
- [x] Elevation angle control
- [x] Radius/distance factor
- [ ] Interpolation type selector
- [ ] Preview button (first 2 seconds)

### 4.3 Timeline Controls
- [ ] Add timeline scrubber widget
- [ ] Frame counter display
- [ ] Jump to time input
- [ ] Loop animation checkbox

### Success Criteria
- All animations controllable via widgets
- Real-time parameter updates
- Intuitive UI layout

## Phase 5: Export and Documentation (2 hours)

### 5.1 HTML Export Enhancements
- [ ] Test camera animation export to HTML
- [ ] Add embedded playback controls to exported HTML
- [ ] Include animation metadata in export
- [ ] Test in multiple browsers (Chrome, Firefox, Safari)
- [ ] Add fallback for unsupported browsers

### 5.2 Video Export Pipeline
- [ ] Document frame-by-frame export process
- [ ] Create script for MP4 generation via ffmpeg
- [ ] Add GIF export option for presentations
- [ ] Test with sample datasets
- [ ] Write troubleshooting guide

### 5.3 Documentation
- [ непоComplete API documentation with examples
- [ ] Create tutorial notebook with all animation types
- [ ] Add cookbook section with common scenarios
- [ ] Record demo videos of each animation
- [ ] Update README with animation features

### Success Criteria
- HTML exports work standalone
- Video export produces smooth output
- Documentation covers all features

## Phase 6: Advanced Features (3 hours)

### 6.1 Path Recording
- [ ] Implement camera path recording from manual interaction
- [ ] Save/load camera paths to JSON
- [ ] Edit keyframes after recording
- [ ] Smooth recorded paths
- [ ] Share paths between visualizations

### 6.2 Multi-Dataset Animations
- [ ] Support transitions between datasets
  - [ ] Thermal → Pumped → Difference
  - [ ] Time series data
- [ ] Maintain camera during data switch
- [ ] Add crossfade transitions
- [ ] Synchronize with intensity scaling

### 6.3 Cinematic Effects
- [ ] Add depth of field simulation (blur distance objects)
- [ ] Implement smooth speed ramping
- [ ] Add pause/emphasis at key moments
- [ ] Support camera shake for emphasis
- [ ] Add fade in/out capabilities

### Success Criteria
- Can record and replay user navigation
- Smooth transitions between datasets
- Professional presentation quality

## Phase 7: Testing and Optimization (2 hours)

### 7.1 Performance Testing
- [x] Profile animation performance with large datasets (100³+)
- [x] Optimize keyframe generation
- [x] Test memory usage during animations
- [x] Benchmark frame rates
- [x] Document performance limits

### 7.2 Compatibility Testing
- [x] Test all animations with different data shapes
  - [x] 41³ cubic data
  - [x] 81³ cubic data
  - [x] Non-cubic shapes
- [x] Test with all clipping modes (sphere, octant, hybrid)
- [x] Verify log scaling compatibility
- [x] Test with NaN/infinite values

### 7.3 User Testing
- [x] Create test scenarios for each animation
- [x] Document common issues and solutions
- [ ] Gather feedback on controls
- [ ] Refine based on usage patterns
- [x] Create quick reference card

### Success Criteria
- Animations work with all data formats
- Performance acceptable (>15 fps)
- No memory leaks or crashes

## Implementation Notes

### Technical Considerations
- Use `plot.camera_animation` for smooth interpolation
- Sphere updates need manual threading
- Coordinate conversion critical for accuracy
- Time synchronization between animations

### Dependencies
- k3d >= 2.15.0
- numpy
- threading (for synchronized animations)
- ipywidgets (for controls)

### File Structure
```
eryx/visualization/volume/k3d/
├── k3d_spherical_clipping.py          # Main module (add methods here)
├── k3d_camera_animations.py           # New: Animation utilities
├── k3d_animation_presets.py           # New: Preset definitions
└── k3d_spherical_clipping_animated.ipynb  # New: Demo notebook
```

### Testing Data
- Use existing torch_diffuse_intensity.npy
- Create synthetic test patterns
- Generate time series for multi-dataset tests

## Risk Mitigation

### Potential Issues
1. **Browser compatibility**: Test early, provide fallbacks
2. **Performance with large data**: Implement level-of-detail
3. **Synchronization complexity**: Use simple timing model
4. **Export limitations**: Document workarounds

### Fallback Plans
- If camera_animation fails: Use manual loop approach
- If performance poor: Reduce keyframe density
- If export broken: Frame-by-frame capture

## Success Metrics

### Quantitative
- [ ] All 7 animation types implemented
- [ ] 30+ fps on standard hardware
- [ ] <5 second load time for animations
- [ ] 100% test coverage for new methods

### Qualitative
- [ ] Animations enhance data understanding
- [ ] Controls intuitive for non-experts
- [ ] Export quality suitable for publications
- [ ] Documentation clear and complete

## Timeline Estimate
**Total**: ~16 hours of development
- Phase 1: 2 hours
- Phase 2: 3 hours
- Phase 3: 4 hours
- Phase 4: 2 hours
- Phase 5: 2 hours
- Phase 6: 3 hours
- Phase 7: 2 hours

## Next Steps
1. Review and approve plan
2. Set up development branch
3. Begin Phase 1 implementation
4. Regular testing after each phase
5. User feedback collection
6. Final integration and merge