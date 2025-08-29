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
│   ├── docs/visualization/K3D_JAVASCRIPT_PATTERNS.md
│   │   Practical JavaScript patterns, common pitfalls, solutions
│   │
│   └── docs/visualization/K3D_HTML_JAVASCRIPT_API_REFERENCE.md
│       Deep technical details, HTML export structure, debugging
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
**Start here**: [`K3D_HTML_JAVASCRIPT_API_REFERENCE.md`](./K3D_HTML_JAVASCRIPT_API_REFERENCE.md)
- Performance optimization (avoid 60+ clipping planes!)
- Debugging techniques
- Console commands for troubleshooting

### "I need to add JavaScript animations"
**Start here**: [`K3D_JAVASCRIPT_PATTERNS.md`](./K3D_JAVASCRIPT_PATTERNS.md)
- Global function patterns
- Animation state management
- Common JavaScript pitfalls

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
| **K3D_JAVASCRIPT_PATTERNS.md** | JavaScript implementation | Developers | Intermediate | 15 min |
| **K3D_HTML_JAVASCRIPT_API_REFERENCE.md** | Technical API details | Advanced devs | Advanced | 20 min |
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
1. K3D_HTML_JAVASCRIPT_API_REFERENCE.md → Find performance section
2. Discover 60+ clipping planes issue
3. Switch to data masking approach
4. 10x performance improvement! ✅
```

### Journey 3: Add Custom Animation
```
1. K3D_JAVASCRIPT_PATTERNS.md → Learn patterns
2. K3D_HTML_JAVASCRIPT_API_REFERENCE.md → Understand structure
3. Implement with global functions
4. Animation works! ✅
```

---

## ⚠️ Common Mistakes to Avoid

### Don't Start With:
- ❌ **K3D_HTML_JAVASCRIPT_API_REFERENCE.md** if you just want to visualize data
- ❌ **CLAUDE.md** if you're not integrating with Eryx
- ❌ **JavaScript patterns** if you're just using Python

### Critical Knowledge:
- 🔴 **HTML exports have different structure than Python K3D**
- 🔴 **`plot.objects` doesn't exist in HTML exports**
- 🔴 **60+ clipping planes will kill performance**
- 🔴 **Functions must be global for HTML onclick**

---

## 📊 Documentation Statistics

- **Total Lines**: ~1,500 lines of K3D documentation
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
See: [K3D_HTML_JAVASCRIPT_API_REFERENCE.md](./K3D_HTML_JAVASCRIPT_API_REFERENCE.md#accessing-volume-data-in-javascript)

### Performance issue with clipping?
Use data masking instead of clipping planes!
See: [K3D_HTML_JAVASCRIPT_API_REFERENCE.md](./K3D_HTML_JAVASCRIPT_API_REFERENCE.md#performance-optimization-guidelines)

### Animation not working?
Check global function scope!
See: [K3D_JAVASCRIPT_PATTERNS.md](./K3D_JAVASCRIPT_PATTERNS.md#global-function-registration-pattern)

---

## 📝 Maintenance Notes

### Recently Updated
- **K3D_HTML_JAVASCRIPT_API_REFERENCE.md** - Created Nov 2024 from debugging session
- **Cross-references** - Added Nov 2024 to connect all documents

### Needs Consolidation
- JavaScript patterns duplicated in CLAUDE.md → Should reference K3D_JAVASCRIPT_PATTERNS.md
- Basic usage examples in 5 files → Keep only in FINAL_WORKING_SOLUTION.md

---

*Last updated: November 2024*  
*Total documentation: 5 main files + 10+ notebooks*  
*Status: Well-documented with minor redundancy to address*