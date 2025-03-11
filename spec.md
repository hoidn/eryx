# Phonon Test Refactoring - Implementation Specification
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Refactor the `test_models_torch_phonon.py` file to align with the project's state-based testing conventions

## Mid-Level Objective

- Replace direct log loading with standardized state-based testing helpers
- Implement consistent test structure with proper error handling
- Modernize object creation to use StateBuilder pattern
- Ensure gradient flow verification follows project conventions
- Maintain test coverage while improving code maintainability

## Implementation Notes
- Use only the test helpers from `eryx.autotest.test_helpers`
- Follow patterns established in `test_models_torch_kvector.py`
- Preserve test coverage and functionality
- Maintain comprehensive testing for gradient flow
- All refactored tests must pass with the existing codebase

## Context

### Beginning context
- `tests/test_models_torch_phonon.py` - Original phonon test implementation
- `tests/test_models_torch_kvector.py` - Reference implementation following conventions
- `tests/test_base.py` - Base test class

### Ending context  
- `tests/test_models_torch_phonon.py` - Refactored implementation

## Low-Level Tasks
> Ordered from start to finish

1. Refactor the base class structure and setup
```aider
UPDATE tests/test_models_torch_phonon.py:
    REPLACE the TestOnePhononPhonon class with a new class structure that:
    - Inherits from TestBase
    - Initializes module_name and class_name in setUp()
    - Adds a create_models helper method
    
    KEEP all existing test methods but change their structure to match the pattern.
    
    ADD proper imports:
        import unittest
        import os
        import torch
        import numpy as np
        from tests.test_base import TestBase
        from eryx.models_torch import OnePhonon
        from eryx.autotest.test_helpers import load_test_state, build_test_object, ensure_tensor
```

2. Refactor the compute_gnm_hessian test method
```aider
UPDATE tests/test_models_torch_phonon.py:
    REPLACE test_compute_gnm_hessian with a state-based implementation that:
    - Uses load_test_state to load before state
    - Uses build_test_object to create the model
    - Properly handles missing log files with skipTest
    - Follows the pattern from test_models_torch_kvector.py
    - Verifies results against expected after state
    - Validates tensor shapes, dtypes, and gradient requirements
    
    ENSURE proper error handling for log loading
    
    PRESERVE the core test logic that verifies the method matches ground truth
```

3. Refactor compute_gnm_K test method
```aider
UPDATE tests/test_models_torch_phonon.py:
    REPLACE test_compute_gnm_K with a state-based implementation that:
    - Uses load_test_state to load before state
    - Uses build_test_object to create the model
    - Properly handles missing log files with skipTest
    - Verifies results against expected after state
    - Validates tensor shapes, dtypes, and gradient requirements
    
    ENSURE the method tests proper gradient flow
    PRESERVE the core test logic that verifies correct matrix computation
```

4. Refactor compute_gnm_Kinv test method
```aider
UPDATE tests/test_models_torch_phonon.py:
    REPLACE test_compute_gnm_Kinv with a state-based implementation that:
    - Uses load_test_state to load before state
    - Uses build_test_object to create the model
    - Properly handles missing log files with skipTest
    - Verifies results against expected after state
    - Validates tensor shapes, dtypes, and gradient requirements
    
    ENSURE the method tests proper gradient flow
    PRESERVE the core test logic that verifies compute_gnm_Kinv produces correct results
```

5. Refactor compute_hessian test method
```aider
UPDATE tests/test_models_torch_phonon.py:
    REPLACE test_compute_hessian with a state-based implementation that:
    - Uses load_test_state to load before state
    - Uses build_test_object to create the model
    - Properly handles missing log files with skipTest
    - Verifies results against expected after state
    - Validates tensor shapes, dtypes, and gradient requirements
    
    ENSURE the method tests proper gradient flow
    PRESERVE the core test logic that verifies hessian computation
```

6. Refactor compute_gnm_phonons test method
```aider
UPDATE tests/test_models_torch_phonon.py:
    REPLACE test_compute_gnm_phonons with a state-based implementation that:
    - Uses load_test_state to load before state
    - Uses build_test_object to create the model
    - Properly handles missing log files with skipTest
    - Verifies results against expected after state
    - Validates tensor shapes, dtypes, and gradient requirements
    
    ENSURE the method tests eigendecomposition and gradient flow
    PRESERVE the core test logic that verifies V and Winv tensor computation
```

7. Refactor gradient_flow test method
```aider
UPDATE tests/test_models_torch_phonon.py:
    REPLACE test_gradient_flow with an implementation that:
    - Uses the verify_gradient_flow helper from eryx.autotest.test_helpers
    - Tests gradient flow through all key operations
    - Follows project conventions for gradient testing
    
    ADD comprehensive tests for gradient flow through:
    - compute_gnm_K
    - compute_gnm_Kinv
    - compute_hessian operations
```

8. Add log completeness test method
```aider
ADD to tests/test_models_torch_phonon.py:
    CREATE test_log_completeness method that:
    - Verifies phonon-related logs exist
    - Checks for required attributes in logs
    - Uses self.verify_required_logs helper method
    - Skips if verification is disabled
    
    SPECIFY required attributes for each phonon method:
    - compute_gnm_phonons: ["V", "Winv"]
    - compute_hessian: ["hessian"]
    - compute_gnm_K: ["K"]
    - compute_gnm_Kinv: ["Kinv"]
```
