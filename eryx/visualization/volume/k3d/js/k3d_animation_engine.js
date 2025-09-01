/**
 * K3D Animation Engine
 * Smooth browser-based animations for K3D plots
 * 
 * @class K3DAnimationEngine
 */
class K3DAnimationEngine {
    constructor(K3DInstance) {
        this.K3D = K3DInstance;
        
        // Animation state
        this.currentFrame = 0;
        this.totalFrames = 60;
        this.isPlaying = false;
        this.isPaused = false;
        this.animationId = null;
        this.startTime = null;
        this.pauseTime = null;
        this.duration = 5000; // milliseconds
        
        // Animation queue
        this.animations = [];
        this.currentAnimationIndex = 0;
        
        // Callbacks
        this.onFrame = null;
        this.onComplete = null;
        
        // Performance
        this.fps = 60;
        this.frameInterval = 1000 / this.fps;
        this.lastFrameTime = 0;
        
        // Initial state
        this.initialCamera = null;
        this.initialClippingPlanes = null;
        
        // Store initial state
        this.saveInitialState();
    }
    
    /**
     * Save initial plot state for reset
     */
    saveInitialState() {
        this.initialCamera = this.K3D.camera ? [...this.K3D.camera] : null;
        // Note: getClippingPlanes doesn't exist, we track it manually
        this.initialClippingPlanes = [[1, 0, 0, 0]]; // Default
    }
    
    /**
     * Interpolation functions
     */
    linear(t) {
        return t;
    }
    
    easeInOut(t) {
        return 0.5 * (1 - Math.cos(Math.PI * t));
    }
    
    easeIn(t) {
        return t * t;
    }
    
    easeOut(t) {
        return t * (2 - t);
    }
    
    smoothstep(t) {
        return t * t * (3 - 2 * t);
    }
    
    /**
     * Calculate camera position for orbital motion
     */
    calculateOrbitalCamera(angle, radius = 6, elevation = 30, center = [0, 0, 0]) {
        const phi = (elevation * Math.PI) / 180;
        const theta = angle;
        
        const x = radius * Math.cos(theta) * Math.cos(phi);
        const y = radius * Math.sin(theta) * Math.cos(phi);
        const z = radius * Math.sin(phi);
        
        return [
            x + center[0], y + center[1], z + center[2],  // Position
            center[0], center[1], center[2],              // Target
            0, 0, 1                                       // Up vector
        ];
    }
    
    /**
     * Calculate zoom camera position
     */
    calculateZoomCamera(distance, azimuth = 45, elevation = 30, center = [0, 0, 0]) {
        const phi = (elevation * Math.PI) / 180;
        const theta = (azimuth * Math.PI) / 180;
        
        const x = distance * Math.cos(theta) * Math.cos(phi);
        const y = distance * Math.sin(theta) * Math.cos(phi);
        const z = distance * Math.sin(phi);
        
        return [
            x + center[0], y + center[1], z + center[2],
            center[0], center[1], center[2],
            0, 0, 1
        ];
    }
    
    /**
     * Calculate clipping plane position
     */
    calculateClippingPlane(axis, position) {
        const planes = {
            'x': [[1, 0, 0, position]],
            'y': [[0, 1, 0, position]],
            'z': [[0, 0, 1, position]]
        };
        return planes[axis] || planes['x'];
    }
    
    /**
     * Add animation to queue
     */
    addAnimation(type, params = {}) {
        const animation = {
            type: type,
            params: Object.assign({
                duration: 3000,
                easing: 'easeInOut',
                fps: 60
            }, params),
            frames: []
        };
        
        // Pre-calculate frames based on type
        switch(type) {
            case 'orbital':
                animation.frames = this.generateOrbitalFrames(animation.params);
                break;
            case 'zoom':
                animation.frames = this.generateZoomFrames(animation.params);
                break;
            case 'sweep':
                animation.frames = this.generateSweepFrames(animation.params);
                break;
            case 'combined':
                animation.frames = this.generateCombinedFrames(animation.params);
                break;
        }
        
        this.animations.push(animation);
        return this;
    }
    
    /**
     * Generate orbital animation frames
     */
    generateOrbitalFrames(params) {
        const frames = [];
        const numFrames = Math.floor(params.duration * params.fps / 1000);
        const radius = params.radius || 6;
        const elevation = params.elevation || 30;
        const center = params.center || [0, 0, 0];
        const startAngle = params.startAngle || 0;
        const endAngle = params.endAngle || 2 * Math.PI;
        
        for (let i = 0; i <= numFrames; i++) {
            const t = i / numFrames;
            const easedT = this[params.easing](t);
            const angle = startAngle + (endAngle - startAngle) * easedT;
            
            frames.push({
                camera: this.calculateOrbitalCamera(angle, radius, elevation, center),
                time: (params.duration * i) / numFrames
            });
        }
        
        return frames;
    }
    
    /**
     * Generate zoom animation frames
     */
    generateZoomFrames(params) {
        const frames = [];
        const numFrames = Math.floor(params.duration * params.fps / 1000);
        const startDistance = params.startDistance || 10;
        const endDistance = params.endDistance || 3;
        const azimuth = params.azimuth || 45;
        const elevation = params.elevation || 30;
        const center = params.center || [0, 0, 0];
        
        for (let i = 0; i <= numFrames; i++) {
            const t = i / numFrames;
            const easedT = this[params.easing](t);
            const distance = startDistance + (endDistance - startDistance) * easedT;
            
            frames.push({
                camera: this.calculateZoomCamera(distance, azimuth, elevation, center),
                time: (params.duration * i) / numFrames
            });
        }
        
        return frames;
    }
    
    /**
     * Generate clipping plane sweep frames
     */
    generateSweepFrames(params) {
        const frames = [];
        const numFrames = Math.floor(params.duration * params.fps / 1000);
        const axis = params.axis || 'x';
        const startPos = params.startPos || -2;
        const endPos = params.endPos || 2;
        
        // Get current camera to maintain it
        const currentCamera = this.K3D.camera ? [...this.K3D.camera] : null;
        
        for (let i = 0; i <= numFrames; i++) {
            const t = i / numFrames;
            const easedT = this[params.easing](t);
            const position = startPos + (endPos - startPos) * easedT;
            
            frames.push({
                camera: currentCamera,
                clippingPlanes: this.calculateClippingPlane(axis, position),
                time: (params.duration * i) / numFrames
            });
        }
        
        return frames;
    }
    
    /**
     * Generate combined camera + clipping animation frames
     */
    generateCombinedFrames(params) {
        const frames = [];
        const numFrames = Math.floor(params.duration * params.fps / 1000);
        const radius = params.radius || 6;
        const elevation = params.elevation || 30;
        const center = params.center || [0, 0, 0];
        const sweepAxis = params.sweepAxis || 'x';
        const sweepRange = params.sweepRange || [-2, 2];
        
        for (let i = 0; i <= numFrames; i++) {
            const t = i / numFrames;
            const easedT = this[params.easing](t);
            
            // Orbital camera
            const angle = 2 * Math.PI * easedT;
            const camera = this.calculateOrbitalCamera(angle, radius, elevation, center);
            
            // Sweeping clipping plane
            const sweepPos = sweepRange[0] + (sweepRange[1] - sweepRange[0]) * easedT;
            const clippingPlanes = this.calculateClippingPlane(sweepAxis, sweepPos);
            
            frames.push({
                camera: camera,
                clippingPlanes: clippingPlanes,
                time: (params.duration * i) / numFrames
            });
        }
        
        return frames;
    }
    
    /**
     * Main animation loop
     */
    animate(timestamp) {
        if (!this.isPlaying) return;
        
        // Initialize start time
        if (!this.startTime) {
            this.startTime = timestamp;
            this.lastFrameTime = timestamp;
        }
        
        // Handle pause/resume
        if (this.pauseTime) {
            this.startTime += timestamp - this.pauseTime;
            this.pauseTime = null;
        }
        
        // Check frame rate limiting
        const deltaTime = timestamp - this.lastFrameTime;
        if (deltaTime < this.frameInterval) {
            this.animationId = requestAnimationFrame(this.animate.bind(this));
            return;
        }
        
        // Get current animation
        const currentAnimation = this.animations[this.currentAnimationIndex];
        if (!currentAnimation) {
            this.stop();
            return;
        }
        
        // Calculate progress
        const elapsed = timestamp - this.startTime;
        const progress = Math.min(elapsed / currentAnimation.params.duration, 1);
        
        // Find appropriate frame
        const frameIndex = Math.floor(progress * (currentAnimation.frames.length - 1));
        const frame = currentAnimation.frames[frameIndex];
        
        // Apply frame
        if (frame) {
            if (frame.camera) {
                this.K3D.camera = frame.camera;
            }
            if (frame.clippingPlanes) {
                this.K3D.clipping_planes = frame.clippingPlanes;
            }
            
            // Trigger render if available
            if (this.K3D.render) {
                this.K3D.render();
            }
            
            // Callback
            if (this.onFrame) {
                this.onFrame(frameIndex, currentAnimation.frames.length, progress);
            }
        }
        
        // Check if animation complete
        if (progress >= 1) {
            this.currentAnimationIndex++;
            this.startTime = timestamp;
            
            if (this.currentAnimationIndex >= this.animations.length) {
                this.stop();
                if (this.onComplete) {
                    this.onComplete();
                }
                return;
            }
        }
        
        // Continue animation
        this.lastFrameTime = timestamp;
        this.animationId = requestAnimationFrame(this.animate.bind(this));
    }
    
    /**
     * Control methods
     */
    play() {
        if (this.isPlaying && !this.isPaused) return;
        
        this.isPlaying = true;
        this.isPaused = false;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        this.animationId = requestAnimationFrame(this.animate.bind(this));
        console.log('Animation started');
    }
    
    pause() {
        if (!this.isPlaying || this.isPaused) return;
        
        this.isPaused = true;
        this.pauseTime = performance.now();
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        console.log('Animation paused');
    }
    
    stop() {
        this.isPlaying = false;
        this.isPaused = false;
        this.startTime = null;
        this.pauseTime = null;
        this.currentAnimationIndex = 0;
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        console.log('Animation stopped');
    }
    
    reset() {
        this.stop();
        
        // Reset to initial state
        if (this.initialCamera) {
            this.K3D.camera = this.initialCamera;
        }
        if (this.initialClippingPlanes) {
            this.K3D.clipping_planes = this.initialClippingPlanes;
        }
        
        if (this.K3D.render) {
            this.K3D.render();
        }
        console.log('Animation reset');
    }
    
    /**
     * Clear animation queue
     */
    clear() {
        this.stop();
        this.animations = [];
        this.currentAnimationIndex = 0;
        console.log('Animation queue cleared');
    }
    
    /**
     * Export current state
     */
    exportState() {
        return {
            camera: this.K3D.camera ? [...this.K3D.camera] : null,
            animations: this.animations,
            currentIndex: this.currentAnimationIndex,
            isPlaying: this.isPlaying,
            isPaused: this.isPaused
        };
    }
}

// Make available globally if K3DInstance exists
if (typeof K3DInstance !== 'undefined') {
    window.K3DAnimationEngine = K3DAnimationEngine;
    window.k3dEngine = new K3DAnimationEngine(K3DInstance);
    console.log('K3D Animation Engine initialized');
}