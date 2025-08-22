# Proposed CLAUDE.md Improvements

## Problem: CLAUDE.md as AI Entry Point

CLAUDE.md is the **primary entry point for AI assistants**, yet it currently lacks critical information that would prevent common misunderstandings. The grid parameter issue could have been avoided with better AI-focused documentation.

## Current Issues in CLAUDE.md

### 1. **Incorrect Basic Usage Example** (Line 76)
```python
# WRONG - This will cause an error!
model = OnePhonon(pdb_path, hsampling=32, ksampling=32, lsampling=32)
```

### 2. **Missing Critical API Semantics**
- No explanation of what sampling parameters mean
- No warning about oversampling vs. grid points
- No performance implications

### 3. **No "Gotchas" Section**
- AI assistants need to know common pitfalls
- Critical for preventing predictable errors

## Proposed CLAUDE.md Structure

```markdown
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL API INFORMATION - READ FIRST

### Grid Sampling Parameters (COMMONLY MISUNDERSTOOD)
The parameters `hsampling`, `ksampling`, `lsampling` are **tuples of (min, max, oversampling_factor)**:
- **NOT** (min, max, num_points)
- The third value is a multiplicative oversampling factor
- Formula: `n_points = (max - min) * oversampling_factor + 1`

#### Examples with Calculations:
```python
# CORRECT: Creates a 5×5×5 grid (125 total points)
hsampling = (-2, 2, 1)  # (2-(-2))*1+1 = 5 points
ksampling = (-2, 2, 1)  # 5 points
lsampling = (-2, 2, 1)  # 5 points
model = OnePhonon(pdb_path, hsampling, ksampling, lsampling)

# WRONG: Don't assume third parameter is number of points!
hsampling = (-2, 2, 5)  # This creates 21 points, NOT 5!
# (2-(-2))*5+1 = 21 points per dimension = 21³ = 9,261 total points
```

#### Performance Impact:
- Oversampling factor of 2 → 8× more points (2³)
- Oversampling factor of 3 → 27× more points (3³)
- Oversampling factor of 5 → 125× more points (5³)
- **Memory usage and computation time scale cubically!**

### Common Pitfalls

1. **Grid Size Explosion**
   - `[-4, 4, 32]` creates 257 points per dimension (16.8 million total!)
   - Always calculate: `(max-min)*oversampling+1` before running

2. **Memory Estimation**
   - Each grid point needs ~1KB for small proteins
   - 100×100×100 grid = 1 million points ≈ 1GB memory

3. **Device Memory Limits**
   - GPU memory is limited (typically 8-24GB)
   - Use CPU for large grids or reduce oversampling

## Project Overview
[existing content...]

## Common Development Commands

### Quick Parameter Check
```python
# Before running, always verify grid size:
h_points = (h_max - h_min) * h_oversampling + 1
k_points = (k_max - k_min) * k_oversampling + 1  
l_points = (l_max - l_min) * l_oversampling + 1
total_points = h_points * k_points * l_points
print(f"Grid will have {total_points:,} points")
print(f"Estimated memory: {total_points/1000:.1f} MB")
```

### Testing with Small Grids
```python
# For development/testing, use oversampling=1:
test_model = OnePhonon(
    pdb_path,
    hsampling=(-2, 2, 1),  # 5 points
    ksampling=(-2, 2, 1),  # 5 points  
    lsampling=(-2, 2, 1),  # 5 points
    # Total: 125 points (fast!)
)
```

### Production Settings
```python
# For production, typical oversampling is 2-4:
prod_model = OnePhonon(
    pdb_path,
    hsampling=(-10, 10, 3),  # 61 points
    ksampling=(-10, 10, 3),  # 61 points
    lsampling=(-10, 10, 3),  # 61 points  
    # Total: 226,981 points
)
```

## Architecture Overview
[existing content...]

## Usage Patterns

### Basic Usage (CORRECTED)
```python
# NumPy version - Grid mode
from eryx import OnePhonon
model = OnePhonon(
    pdb_path,
    hsampling=(-4, 4, 2),  # 17 points: (4-(-4))*2+1
    ksampling=(-4, 4, 2),  # 17 points
    lsampling=(-4, 4, 2),  # 17 points
    # Total: 4,913 points
)
intensity = model.apply_disorder()

# PyTorch version - Arbitrary q-vectors
from eryx import OnePhonon_torch
q_vectors = torch.randn(1000, 3)  # 1000 arbitrary points
model_torch = OnePhonon_torch(
    pdb_path,
    q_vectors=q_vectors,
    hsampling=(-2, 2, 1),  # Still needed for ADP calculation
    ksampling=(-2, 2, 1),
    lsampling=(-2, 2, 1),
    device='cuda'
)
intensity = model_torch.apply_disorder()
```

## Parameter Reference Table

| Parameter | Type | Description | Example | Points Created |
|-----------|------|-------------|---------|----------------|
| hsampling | tuple(float,float,float) | (min, max, oversampling) | (-2, 2, 1) | 5 |
| | | | (-2, 2, 2) | 9 |
| | | | (-2, 2, 3) | 13 |
| ksampling | tuple(float,float,float) | Same as hsampling | (-4, 4, 1) | 9 |
| lsampling | tuple(float,float,float) | Same as hsampling | (-10, 10, 2.5) | 51 |

## Debugging Commands

### Check Actual Grid Size
```python
print(f"Grid shape: {model.q_grid.shape}")
print(f"Map shape: {model.map_shape}")
print(f"Total points: {model.q_grid.shape[0]}")
```

### Memory Usage Check
```python
import torch
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
```

## Important Notes
[existing content, but add:]

### Grid Size Warning
**CRITICAL**: The third parameter in sampling tuples is an **oversampling factor**, not the number of points. This is the #1 source of performance problems. Always calculate the actual grid size before running computations.

### Memory Management
- Start with oversampling=1 for testing
- Increase gradually while monitoring memory
- Production typically uses oversampling=2-4
- Research applications may use oversampling=5-10 (with appropriate hardware)
```

## Additional Recommendations

### 1. Add Validation Helper
Create a helper function that AI assistants can use:
```python
def validate_sampling_params(hsampling, ksampling, lsampling, max_points=1e6):
    """Check if sampling parameters will create reasonable grid."""
    h_pts = (hsampling[1] - hsampling[0]) * hsampling[2] + 1
    k_pts = (ksampling[1] - ksampling[0]) * ksampling[2] + 1
    l_pts = (lsampling[1] - lsampling[0]) * lsampling[2] + 1
    total = h_pts * k_pts * l_pts
    
    if total > max_points:
        raise ValueError(f"Grid too large: {total:,} points > {max_points:,} limit")
    
    return h_pts, k_pts, l_pts, total
```

### 2. Add to .claude/claude_config.yaml
```yaml
project_specifics:
  common_mistakes:
    - description: "Grid sampling parameters"
      wrong: "Assuming third value is number of points"
      correct: "Third value is oversampling factor"
      example: "(-2, 2, 3) creates 13 points, not 3"
```

### 3. Add AI-Specific Warnings
```markdown
## FOR AI ASSISTANTS

When users request grid-based calculations:
1. ALWAYS calculate actual grid size first
2. WARN if total points > 10,000 for testing
3. SUGGEST oversampling=1 for initial development
4. EXPLAIN the memory implications

Example response template:
"Your parameters will create a {h}×{k}×{l} = {total:,} point grid. 
This will use approximately {memory} MB of memory. 
For testing, consider using oversampling=1 instead."
```

## Benefits of These Changes

1. **Prevents Misunderstanding**: Clear explanation at the top of CLAUDE.md
2. **Provides Context**: Shows memory/performance implications
3. **Offers Solutions**: Includes validation code and suggestions
4. **AI-Optimized**: Written specifically for AI assistants to understand and use
5. **Self-Contained**: All critical information in one place

This would have completely prevented the grid parameter misunderstanding.