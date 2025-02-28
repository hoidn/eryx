# Specification: Test Framework for PyTorch-NumPy Equivalence
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Establish a comprehensive testing framework to validate numerical equivalence between PyTorch and NumPy implementations of the Eryx diffuse scattering calculations

## Mid-Level Objectives

- Configure the autotest framework to efficiently capture ground truth data from NumPy implementations
- Implement tolerance-based comparison methods for numerically sensitive operations
- Create automated test generation for parallel PyTorch implementations
- Define storage and management strategies for test data

## Implementation Notes

### Autotest Framework Usage
- Leverage existing autotest components (Logger, Serializer, FunctionMapping, Testing)
- Extend rather than replace the existing functionality
- Use torch_testing.py as the foundation for PyTorch-specific testing

### Ground Truth Capture Strategy
- Use the debug decorator to instrument key NumPy functions
- Capture inputs and outputs during normal execution
- Store test data in a structured, retrievable format
- Run the NumPy simulation with varied parameters to generate comprehensive test cases

### Tolerance Settings
- Define appropriate tolerances for different operation types:
  * Basic arithmetic: 1e-7 relative, 1e-8 absolute
  * Eigendecomposition: 1e-5 relative, 1e-5 absolute
  * FFT operations: 1e-5 relative, 1e-6 absolute
  * Complex operations: 1e-6 relative, 1e-7 absolute
- Handle special cases like NaN values and very small numbers

### Test Data Management
- Create a consistent directory structure for test data
- Implement cleaning/pruning utilities to manage data size
- Store only essential inputs/outputs to minimize storage requirements
- Enable selective loading of test data by function or module

## Context

### Beginning Context
- Existing autotest framework (logger.py, serializer.py, functionmapping.py, etc.)
- Preliminary torch_testing.py with comparison methods
- First iterations of PyTorch implementations for core components

### Ending Context
- Extended autotest framework with comprehensive PyTorch support
- Ground truth data capture pipeline
- Automated test generation for PyTorch implementations
- Complete test coverage with appropriate tolerances
- Documentation of testing methodology

## Low-Level Tasks
> Ordered from start to finish

1. Extend torch_testing.py with enhanced comparison methods
```aider
UPDATE eryx/autotest/torch_testing.py:
    EXTEND the TorchTesting class with:
    - Add tensor_equals method with customizable tolerances by operation type
    - Add specialized comparison methods for eigendecomposition results
    - Add handling for NaN/Inf values in tensors
    - Add detailed reporting of numerical differences
    
    IMPLEMENT a TestConfig class to store:
    - Default tolerances by operation type
    - Device placement strategies
    - Test data management parameters
```

2. Create a data capture utility using the debug decorator
```aider
CREATE eryx/autotest/data_capture.py:
    IMPLEMENT a DataCapture class that:
    - Uses the Debug decorator to instrument functions
    - Provides a method to automatically instrument functions from a list
    - Captures diverse test cases by varying parameters
    - Creates a log directory structure matching the module structure
    - Includes a cleanup utility to remove redundant or oversized test data
    
    IMPLEMENT capture_ground_truth function that:
    - Takes a list of target functions/methods to instrument
    - Runs simulation with different parameters
    - Logs inputs/outputs to appropriate directories
```

3. Create test generation utility for PyTorch implementations
```aider
CREATE eryx/autotest/test_generator.py:
    IMPLEMENT a TestGenerator class that:
    - Takes a list of functions/modules as input
    - Generates test files for PyTorch implementations
    - Creates test methods that load ground truth data
    - Adds appropriate assertions with tolerances
    - Generates test fixtures for common setup
    
    IMPLEMENT a command-line interface that:
    - Accepts module names to generate tests for
    - Provides options to customize test generation
    - Allows filtering by function name or pattern
```

4. Create a reference test case for key components
```aider
CREATE tests/reference_test_case.py:
    IMPLEMENT a complete reference test that:
    - Demonstrates capturing ground truth from NumPy implementation
    - Shows testing of PyTorch implementation against ground truth
    - Includes gradient checking examples
    - Shows performance comparison between implementations
    - Illustrates proper tolerance settings
    
    DOCUMENT with clear comments explaining:
    - How to adapt the pattern for other components
    - How to handle special cases
    - Recommended testing strategies
```

5. Create a comprehensive test runner
```aider
CREATE tests/run_all_tests.py:
    IMPLEMENT a test runner that:
    - Discovers and runs all PyTorch implementation tests
    - Reports success/failure with detailed statistics
    - Groups results by module/component
    - Shows performance comparison between NumPy and PyTorch
    - Generates an HTML report summarizing results
    
    INCLUDE command-line options to:
    - Run specific test modules
    - Set custom tolerance levels
    - Generate new ground truth data
    - Clean up old test data
```

6. Create documentation of the testing framework
```aider
CREATE eryx/autotest/README_testing.md:
    DOCUMENT the testing framework:
    - Overview of the autotest extension architecture
    - Step-by-step guide to adding tests for new components
    - Explanation of tolerance settings and when to adjust them
    - Guidelines for generating ground truth data
    - Troubleshooting common test failures
    - Best practices for numerical comparison
    - Examples of the most common testing patterns
```
