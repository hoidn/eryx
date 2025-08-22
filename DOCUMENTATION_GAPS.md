# Documentation Gap Analysis: Grid Parameter Discovery Issue

## Executive Summary

The grid parameter misunderstanding was caused by **both documentation and discoverability problems**. While the correct information exists in the codebase, it's buried in low-level implementation functions rather than being documented at the main user-facing entry points.

## Root Cause

### 1. Missing Constructor Documentation
**Most Critical Issue**: The `OnePhonon.__init__()` constructor has **NO docstring** for its parameters.

```python
# Current state in models_torch.py
def __init__(self, pdb_path, hsampling=None, ksampling=None, lsampling=None, ...):
    # No docstring at all!
```

Users naturally look at the constructor first, but find no parameter documentation.

### 2. Terminology Confusion
The documentation uses "oversampling relative to Miller indices" - technical crystallography jargon that can be misinterpreted as "number of sampling points".

### 3. Misleading Examples
All examples use large oversampling factors without explanation:
```python
# From README.md
hsampling=[-4, 4, 32]  # What does 32 mean? Points? Factor?
```

No example shows the calculation: `(4-(-4)) * 32 + 1 = 257 points`

## Where Documentation Exists (But Is Hard to Find)

### ✅ Well-Documented
- `map_utils.py::generate_grid()` - Clear parameter descriptions
- `qvec_conventions.md` - Correct technical explanation

### ❌ Poorly Documented
- `OnePhonon.__init__()` - No docstring
- `OnePhonon_torch.__init__()` - No docstring
- Main README - No parameter calculation explanation

### ⚠️ Inconsistent Documentation
- `PDOS.md` incorrectly refers to the third parameter as "steps"
- Should be "oversampling factor"

## Discovery Path Analysis

A typical user's journey:
1. **README.md** → See example with `[-4, 4, 32]` → Assume 32 is number of points ❌
2. **OnePhonon constructor** → No docstring → Still confused ❌
3. **PDOS.md** → Says "steps" → Reinforces wrong assumption ❌
4. **map_utils.py** (buried in implementation) → Finally find correct info ✅

The correct information requires digging 3-4 levels deep into implementation code.

## Impact Assessment

### Severity: **HIGH**
- Core functionality parameter
- Affects every user
- Causes 10-100x performance/memory issues
- Easy to misunderstand

### Scope
- Both NumPy and PyTorch implementations affected
- All grid-based calculations impacted
- Memory usage scales cubically with misunderstanding

## Recommended Fixes

### Immediate (Before Release)

1. **Add OnePhonon constructor docstring**:
```python
def __init__(self, pdb_path, hsampling=None, ...):
    """
    Initialize OnePhonon diffuse scattering model.
    
    Parameters
    ----------
    hsampling : tuple of (float, float, float)
        (h_min, h_max, oversampling_factor) for h dimension.
        Number of points = (h_max - h_min) * oversampling_factor + 1
        Example: (-2, 2, 3) creates 13 points from h=-2 to h=2
    ksampling : tuple of (float, float, float)
        Same format as hsampling for k dimension
    lsampling : tuple of (float, float, float)
        Same format as hsampling for l dimension
    """
```

2. **Fix PDOS.md parameter table**:
   - Change "steps" → "oversampling_factor"
   - Add formula: `n_points = (max - min) * oversampling + 1`

3. **Update README with clear example**:
```python
# Example: Create a 9×9×9 grid (729 total points)
# Formula: n_points = (max - min) * oversampling + 1
hsampling=[-4, 4, 1]  # (4-(-4))*1+1 = 9 points
ksampling=[-4, 4, 1]  # 9 points
lsampling=[-4, 4, 1]  # 9 points
```

### Future Improvements

4. **Add "Common Pitfalls" section** to documentation
5. **Create interactive parameter calculator** 
6. **Add validation with helpful error messages**:
```python
if oversampling > 10:
    warnings.warn(f"Large oversampling factor {oversampling} will create "
                  f"{n_points} points per dimension. Consider reducing if unintended.")
```

## Lessons Learned

1. **Entry points must be self-documenting** - Users shouldn't need to read implementation code
2. **Examples need explanations** - Show the math, not just magic numbers
3. **Terminology matters** - Avoid jargon or explain it clearly
4. **Consistency is critical** - All docs must use same terminology

## Conclusion

This was a **preventable documentation failure**. The information existed but was not discoverable where users needed it. The fix is straightforward: document parameters at the main entry points with clear explanations and examples.