# K3D Documentation Map & Navigation Guide

**Purpose**: Help users find the right K3D documentation for their needs

---

## 🗺️ Documentation Overview

```
📚 K3D Documentation Structure
│
├── 🚀 Getting Started
│   └── eryx/visualization/README.md
│       Quick overview, architecture, basic examples
│
├── 🔬 For Researchers & End Users  
│   └── eryx/visualization/volume/k3d/FINAL_WORKING_SOLUTION.md
│       Production-ready examples, working code, proven patterns
│
├── 💻 For Developers
│   ├── api/JAVASCRIPT_PATTERNS.md
│   │   Practical JavaScript patterns, common pitfalls, solutions
│   │
│   ├── api/HTML_JAVASCRIPT_API.md
│   │   Deep technical details, HTML export structure, debugging
│   │
│   └── api/PERFORMANCE_OPTIMIZATION_LESSONS.md
│       🆕 Performance breakthroughs, 10x improvement lessons
│
├── 🔧 For Project Integration
│   └── CLAUDE.md (K3D sections)
│       Project-specific patterns, Eryx integration guidelines
│
└── 📓 Interactive Examples
    └── eryx/visualization/volume/k3d/*.ipynb
        Jupyter notebooks with live demonstrations
```

---

## 🎯 Which Document Should I Read?

### "I want to visualize my diffuse scattering data"
**Start here**: [`FINAL_WORKING_SOLUTION.md`](../../eryx/visualization/volume/k3d/FINAL_WORKING_SOLUTION.md)
- Complete working examples
- Copy-paste ready code
- Tested with real data

### "My K3D animation is broken/slow"  
**Start here**: [`PERFORMANCE_OPTIMIZATION_LESSONS.md`](./api/PERFORMANCE_OPTIMIZATION_LESSONS.md)
- **NEW**: 10x performance improvement techniques
- Critical discovery: Direct data masking vs clipping planes
- Memory management and resolution trade-offs

**Then**: [`HTML_JAVASCRIPT_API.md`](./api/HTML_JAVASCRIPT_API.md)
- Debugging techniques and console commands
- HTML export structure understanding

### "I need to add JavaScript animations"
**Start here**: [`JAVASCRIPT_PATTERNS.md`](./api/JAVASCRIPT_PATTERNS.md)
- Global function patterns
- Animation state management
- **NEW**: Direct data masking patterns for performance

### "I'm new to the visualization module"
**Start here**: [`README.md`](../../eryx/visualization/README.md)
- Module architecture
- File organization
- Basic concepts

### "I need interactive controls in Jupyter"
**Start here**: Jupyter notebooks in [`eryx/visualization/volume/k3d/`](../../eryx/visualization/volume/k3d/)
- `k3d_spherical_clipping.ipynb` - Interactive spherical masking
- `k3d_clipping_interactive.ipynb` - GUI controls for clipping

---

## 📖 Document Purposes & Complexity

| Document | Purpose | Audience | Complexity | Read Time |
|----------|---------|----------|------------|-----------|
| **README.md** | Module overview & architecture | All users | Beginner | 5 min |
| **FINAL_WORKING_SOLUTION.md** | Production usage guide | Researchers | Beginner | 10 min |
| **JAVASCRIPT_PATTERNS.md** | JavaScript implementation | Developers | Intermediate | 15 min |
| **HTML_JAVASCRIPT_API.md** | Technical API details | Advanced devs | Advanced | 20 min |
| **PERFORMANCE_OPTIMIZATION_LESSONS.md** | Performance breakthroughs | All developers | Intermediate | 12 min |
| **CLAUDE.md** (K3D sections) | Project integration | All developers | Intermediate | 10 min |

---

## 🔄 Common User Journeys

### Journey 1: Basic Visualization
```
1. README.md → Understand module structure
2. FINAL_WORKING_SOLUTION.md → Copy working example
3. Run code → Success! ✅
```

### Journey 2: Debug Slow Animation
```
1. PERFORMANCE_OPTIMIZATION_LESSONS.md → Learn from real optimization
2. Discover 60+ clipping planes = 5 FPS, data masking = 60 FPS
3. Implement K3DDataMasker class patterns
4. 12x performance improvement! ✅
```

### Journey 3: Add Custom Animation
```
1. JAVASCRIPT_PATTERNS.md → Learn battle-tested patterns
2. HTML_JAVASCRIPT_API.md → Understand K3D structure
3. Use K3DAnimationController class pattern
4. Smooth animation achieved! ✅
```

---

## ⚠️ Common Mistakes to Avoid

### Don't Start With:
- ❌ **HTML_JAVASCRIPT_API.md** if you just want to visualize data
- ❌ **CLAUDE.md** if you're not integrating with Eryx
- ❌ **JavaScript patterns** if you're just using Python
- ❌ **Performance lessons** if you haven't tried basic approach first

### Critical Knowledge:
- 🔴 **HTML exports have different structure than Python K3D**
- 🔴 **`plot.objects` doesn't exist in HTML exports**
- 🔴 **60+ clipping planes will kill performance**
- 🔴 **Functions must be global for HTML onclick**

---

## 📊 Documentation Statistics

- **Total Lines**: ~2,200 lines of K3D documentation (+700 from recent session)
- **Code Examples**: 50+ working examples
- **Debugging Commands**: 20+ console commands
- **Performance Tips**: 10+ optimization patterns
- **Common Pitfalls**: 15+ documented issues with solutions

---

## 🔍 Quick Reference

### Need to find volume data in JavaScript?
```javascript
const world = plot.getWorld();
const volumeConfig = world.ObjectsListJson[volumeId];
const volumeData = volumeConfig.volume.data;  // Float32Array
```
See: [HTML_JAVASCRIPT_API.md](./api/HTML_JAVASCRIPT_API.md#accessing-volume-data-in-javascript)

### Performance issue with clipping?
**NEW BREAKTHROUGH**: Direct data masking = 10x faster than clipping planes!
See: [PERFORMANCE_OPTIMIZATION_LESSONS.md](./api/PERFORMANCE_OPTIMIZATION_LESSONS.md#critical-performance-discoveries)

### Animation not working?
Check global function scope and K3D Promise handling!
See: [JAVASCRIPT_PATTERNS.md](./api/JAVASCRIPT_PATTERNS.md#global-function-registration-pattern)

---

## 📝 Maintenance Notes

### Recently Updated (August 2024)
- **PERFORMANCE_OPTIMIZATION_LESSONS.md** - **NEW**: Captures 12x performance breakthrough
- **HTML_JAVASCRIPT_API.md** - Enhanced with performance patterns
- **JAVASCRIPT_PATTERNS.md** - Added direct data masking patterns
- **Cross-references** - Updated for new performance documentation

### Needs Consolidation
- JavaScript patterns duplicated in CLAUDE.md → Should reference K3D_JAVASCRIPT_PATTERNS.md
- Basic usage examples in 5 files → Keep only in FINAL_WORKING_SOLUTION.md

---

*Last updated: November 2024*  
*Total documentation: 5 main files + 10+ notebooks*  
*Status: Well-documented with minor redundancy to address*