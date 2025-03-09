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
# State-Based Testing with StateBuilder

This document describes the new approach to state-based testing using the StateBuilder pattern.

## Overview

State-based testing involves:
1. Capturing object state before and after method execution
2. Using that state to reconstruct test objects
3. Verifying method behavior by comparing with expected results

The new approach simplifies this process by ensuring test objects have the correct structure, with attributes in their expected locations, while leveraging existing adapter classes for type-specific conversions.

## Key Components

### StateBuilder

`StateBuilder` is the core class that builds properly structured test objects:

```python
from eryx.autotest.state_builder import StateBuilder
from eryx.models_torch import OnePhonon

# Load state data
state_data = logger.loadStateLog("logs/eryx.models.OnePhonon._state_before__build_kvec_Brillouin.log")

# Build test object
builder = StateBuilder(device=torch.device('cpu'))
model = builder.build(OnePhonon, state_data)
```

### Test Helpers

Helper functions simplify common testing operations:

```python
from eryx.autotest.test_helpers import (
    load_test_state,
    build_test_object,
    verify_gradient_flow,
    verify_tensor_matches
)

# Load state and build object in one step
before_state = load_test_state(logger, 'eryx.models', 'OnePhonon', '_build_kvec_Brillouin')
model = build_test_object(OnePhonon, before_state, device=device)

# Call method under test
model._build_kvec_Brillouin()

# Verify gradient flow
loss = torch.sum(model.kvec)
loss.backward()
assert verify_gradient_flow(model.kvec, model.model.A_inv)
```

## Standard Test Pattern

Follow this pattern for state-based tests:

```python
def test_method_name(self):
    # 1. Load state data
    before_state = load_test_state(self.logger, module_name, class_name, method_name)
    
    # 2. Build test object 
    model = build_test_object(TorchClass, before_state, device=self.device)
    
    # 3. Call method under test
    method = getattr(model, method_name)
    method()
    
    # 4. Verify method effects
    # (Check attributes, tensor properties, etc.)
    self.assertTrue(hasattr(model, 'expected_attribute'))
    self.assertEqual(model.tensor.shape, expected_shape)
    
    # 5. Verify gradient flow
    loss = torch.sum(model.output_tensor)
    loss.backward()
    self.assertIsNotNone(model.input_tensor.grad)
    
    # 6. Compare with expected state (if needed)
    after_state = load_test_state(self.logger, module_name, class_name, method_name, before=False)
    # Use verify_tensor_matches or torch_testing.compareStates
```

## Handling Common Patterns

### Testing Matrix Construction

For methods that build matrices like `_build_A`, `_build_M`:

```python
# 1. Load state and build model
before_state = load_test_state(self.logger, 'eryx.models', 'OnePhonon', '_build_A')
model = build_test_object(OnePhonon, before_state)

# 2. Call method
model._build_A()

# 3. Verify Amat tensor properties
self.assertTrue(hasattr(model, 'Amat'))
self.assertEqual(model.Amat.shape, expected_shape)
self.assertTrue(model.Amat.requires_grad)

# 4. Verify gradient flow
loss = torch.sum(model.Amat)
loss.backward()
# Check relevant input tensor gradients
```

### Testing K-vector Methods

For k-vector calculation methods:

```python
# 1. Load state and build model
before_state = load_test_state(self.logger, 'eryx.models', 'OnePhonon', '_build_kvec_Brillouin')
model = build_test_object(OnePhonon, before_state)

# 2. Verify A_inv is in the right place
self.assertTrue(hasattr(model, 'model') and hasattr(model.model, 'A_inv'))

# 3. Call method
model._build_kvec_Brillouin()

# 4. Verify kvec tensors and gradient flow
self.assertTrue(hasattr(model, 'kvec'))
# ... more checks
```

### Testing Covariance Matrix

```python
# Follow standard pattern
# Focus on checking covariance matrix properties
self.assertTrue(hasattr(model, 'covar'))
self.assertEqual(model.covar.shape, expected_shape)
# Verify complex tensor handling works correctly
```

## Debugging Tips

When state-based tests fail:

1. **Check Log Files**: Verify state logs exist and contain expected attributes
   ```
   python scripts/verify_logs.py --required-attrs "model,A_inv"
   ```

2. **Examine State Structure**: Print key attributes to see what's available
   ```python
   print(f"State keys: {state_data.keys()}")
   if 'model' in state_data:
       print(f"Model keys: {state_data['model'].keys()}")
   ```

3. **Verify Attribute Locations**: Check that attributes are in expected places
   ```python
   # Should be true:
   assert hasattr(model, 'model') and hasattr(model.model, 'A_inv') 
   # Should be false:
   assert not hasattr(model, 'A_inv')  # A_inv should not be at top level
   ```

4. **Debug Tensor Creation**: Check tensor properties
   ```python
   # After method call:
   print(f"A_inv shape: {model.model.A_inv.shape}")
   print(f"A_inv requires_grad: {model.model.A_inv.requires_grad}")
   print(f"kvec shape: {model.kvec.shape}")
   print(f"kvec requires_grad: {model.kvec.requires_grad}")
   ```

   Note: Gradient flow is not required for state-restored instances in tests.

## Generating New State Logs

To generate state logs:

```bash
# Generate logs for OnePhonon
python scripts/generate_state_logs.py --component onePhonon

# Generate logs for specific components
python scripts/generate_state_logs.py --component mapUtils
python scripts/generate_state_logs.py --component scatter

# Generate all logs
python scripts/generate_state_logs.py --component all
```

## Verifying State Logs

To verify state logs:

```bash
# Check all logs
python scripts/verify_logs.py

# Check with specific requirements
python scripts/verify_logs.py --required-attrs "model,A_inv,kvec,kvec_norm"

# Save detailed results to file
python scripts/verify_logs.py --output verification_results.json
```
