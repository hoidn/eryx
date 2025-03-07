# State-Based Testing in Eryx

This document describes how to use the state-based testing framework for the PyTorch port of Eryx.

## Overview

State-based testing captures object state before and after method execution, allowing for:
- Testing stateful components like the OnePhonon model
- Verifying PyTorch implementations against NumPy ground truth
- Validating internal state changes during complex operations

## Using State Capture

### Basic Usage

Methods are decorated with `@debug` to capture state:

```python
from eryx.autotest.debug import debug

class MyClass:
    @debug
    def my_method(self, param):
        # Method implementation that changes object state
        self.value = param
```

### Configuration Options

The decorator accepts configuration parameters:

```python
@debug(capture_state=True, max_depth=5, exclude_attrs=["temp_data", "_private_attr"])
def complex_method(self, param):
    # Method implementation
    pass
```

Options:
- `capture_state`: Enable/disable state capture (default: True)
- `max_depth`: Maximum recursion depth for nested objects (default: 10)
- `exclude_attrs`: List of attribute patterns to exclude

## Generated Log Files

Running code with decorated methods produces:

1. Regular function logs: `logs/eryx.module.function.log`
2. Before state logs: `logs/eryx.module.Class._state_before_method.log`
3. After state logs: `logs/eryx.module.Class._state_after_method.log`

## Verifying Logs

Use the `verify_logs.py` script to check log completeness:

```bash
# Basic verification
python scripts/verify_logs.py

# Verify with required attributes
python scripts/verify_logs.py --required-attrs "q_grid,map_shape,hkl_grid"

# Save detailed results to file
python scripts/verify_logs.py --output verification_results.json
```

## State-Based Testing Pattern

In test files, use this pattern:

```python
def test_method_state_based(self):
    # 1. Load before state
    before_state = logger.loadStateLog("logs/eryx.module.Class._state_before_method.log")
    
    # 2. Initialize object with state
    obj = torch_testing.initializeFromState(Class, before_state)
    
    # 3. Call method under test
    obj.method()
    
    # 4. Load expected after state
    expected_after = logger.loadStateLog("logs/eryx.module.Class._state_after_method.log")
    
    # 5. Compare states
    self.assertTrue(torch_testing.compareStates(expected_after, obj.__dict__))
```

## Best Practices

1. **Be Selective**: Only capture necessary attributes with `exclude_attrs`
2. **Test Important State**: Check method-specific attributes in verification
3. **Match Ground Truth**: Ensure PyTorch implementation matches the NumPy version's state changes
4. **Document State Dependencies**: Note which attributes methods read from and modify
