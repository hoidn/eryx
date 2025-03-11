

# TODO: Implementation Plan for State-Based Testing

## Introduction

This document outlines the implementation plan for transitioning from function-based to state-based testing for stateful components in the PyTorch port. This transition is necessary because the OnePhonon model and related classes primarily operate by mutating internal state rather than through pure function calls, making traditional input/output testing insufficient.

We are adopting an adapter-based architecture where:
- NumPy components are imported directly
- Adapters convert between NumPy arrays and PyTorch tensors
- State is captured before and after method execution for testing

The key benefits include:
- More accurate testing of stateful components
- Better verification of gradient flow through complex operations
- Clearer architecture with explicit component boundaries
- Reduced implementation effort compared to a complete parallel implementation

This document is organized into phases with concrete tasks and should be read alongside the updated architecture.md, project_rules.md, and phased_plan.md documents.

## Phase 1: Testing Framework Enhancement (1 week) - COMPLETED
> Reference: See "Testing Framework Extensions" section in architecture.md and "State-Based Testing Guidelines" in project_rules.md

### 1.1 Extend Logger for State Capture - COMPLETED
> Location: eryx/autotest/logger.py
> Reference: See "Data Storage and Access" section in phased_plan.md for log naming conventions

- [x] Add `captureState(obj)` method to Logger class in eryx/autotest/logger.py
- [x] Add `loadStateLog(log_file)` method in eryx/autotest/logger.py
- [x] Add `searchStateLogDirectory(log_path_prefix)` method in eryx/autotest/logger.py
- [x] Implement state serialization for complex objects in eryx/autotest/serializer.py
- [x] Add attribute filtering mechanisms for large objects

### 1.2 Add State Comparison Utilities - COMPLETED
> Location: eryx/autotest/torch_testing.py
> Reference: See "State Comparison" section in architecture.md

- [x] Add `compareStates(expected_state, actual_state, tolerances)` method to TorchTesting
- [x] Implement specialized tensor comparison with tolerance support
- [x] Add detailed reporting for state differences
- [x] Add gradient verification for state components

### 1.3 Create State Initialization Utilities - COMPLETED
> Location: eryx/autotest/torch_testing.py
> Reference: See "State Restoration" section in architecture.md

- [x] Add `initializeFromState(torch_class, state_data, device)` method to TorchTesting
- [x] Implement NumPy to PyTorch conversion for state data
- [x] Add support for device placement during state initialization
- [x] Add gradient configuration options for state tensors

### 1.4 Update Testing Documentation - COMPLETED
> Location: eryx/autotest/README.md and docs/testing.md
> Reference: See "Testing Approaches" section in architecture.md

- [x] Document state-based testing approach
- [x] Add examples of state-based test methods
- [x] Update testing guides with state capture/comparison approaches

## Phase 2: Adapter Enhancement (1 week) - COMPLETED
> Reference: See "Adapter Components" section in architecture.md and "Adapter Pattern" in project_rules.md

### 2.1 Enhance PDBToTensor Adapter - COMPLETED
> Location: eryx/adapters.py
> Prerequisites: Tasks 1.1-1.3 complete
> Reference: See "Adapter Usage Pattern" section in architecture.md

- [x] Extend `convert_atomic_model()` to handle complete object state
- [x] Add `convert_state_dict()` method for general state conversion
- [x] Update `convert_crystal()` and `convert_gnm()` to support state-based testing
- [x] Add comprehensive tests for state conversion in tests/test_adapters.py

### 2.2 Enhance TensorToNumpy Adapter - COMPLETED
> Location: eryx/adapters.py
> Prerequisites: Task 2.1 complete

- [x] Add `convert_state_to_numpy()` method for returning state to NumPy
- [x] Implement proper gradient detachment for state components
- [x] Add tests for state conversion back to NumPy in tests/test_adapters.py

### 2.3 Add ModelAdapters State Support - COMPLETED
> Location: eryx/adapters.py
> Prerequisites: Tasks 2.1-2.2 complete
> Reference: See "OnePhonon Model" section in architecture.md

- [x] Add state-based methods to ModelAdapters
- [x] Implement `initialize_from_state()` helper methods
- [x] Add state conversion utilities for OnePhonon

### 2.4 Update Adapter Documentation - COMPLETED
> Location: docs/adapters.md
> Reference: See "Adapter Usage Guidelines" section in project_rules.md

- [x] Document adapter usage patterns
- [x] Add examples of state conversion with adapters
- [x] Update adapter interfaces documentation

## Phase 3: Ground Truth Generation (1 week)
> Reference: See "Ground Truth Testing Strategy" section in phased_plan.md

### 3.1 Create StateCapture Class
> Location: eryx/autotest/state_capture.py

- [x] Implement StateCapture class with configurable state capture
- [x] Add attribute filtering with exclude_attrs option
- [x] Add max_depth parameter to limit recursion
- [x] Handle complex objects like tensors and arrays
- [x] Add proper error handling that logs but doesn't crash

### 3.2 Update Debug Decorator
> Location: eryx/autotest/debug.py

- [x] Update debug decorator to use StateCapture for method state capture
- [x] Add configuration options (max_depth, exclude_attrs)
- [x] Implement consistent log naming convention
- [x] Add error handling that preserves execution flow
- [x] Ensure decorator works with both functions and methods

### 3.3 Create Log Generation Script
> Location: scripts/generate_state_logs.py

- [x] Create a simple script to generate state logs for specific components
- [x] Add command-line options for component selection
- [x] Enable debug mode for test execution
- [x] Integrate with existing run_debug.py functionality
- [x] Add basic output reporting on logs generated

### 3.4 Create Simple Verification Script
> Location: scripts/verify_logs.py

- [x] Implement script to verify log existence and completeness
- [x] Add log pair matching for before/after states
- [x] Check for required attributes in logs
- [x] Provide summary statistics on verification results
- [x] Support saving detailed results to file

### 3.5 Create State-Based Testing Documentation
> Location: docs/state_based_testing.md

- [x] Document the state-based testing approach
- [x] Include examples of decorator usage with configuration
- [x] Describe the state-based testing pattern
- [x] Add best practices for state capture and testing
- [x] Document log naming conventions and verification tools

### 3.6 Improve Serialization for Complex Objects
> Location: eryx/autotest/gemmi_serializer.py

- [x] Fix Gemmi element serialization to extract proper symbols
- [x] Improve element inference to avoid hard-coded values
- [x] Add comprehensive error handling for serialization edge cases
- [x] Document serialization approach for Gemmi objects

## Phase 4: Test Implementation (2 weeks) - COMPLETED
> Reference: See "Component-to-Test Mapping" table in phased_plan.md

### 4.1 Update CP5 Tests (Matrix Construction) - COMPLETED
> Location: tests/test_models_torch.py
> Prerequisites: Phases 1-3 complete

- [x] Update test_build_A to use state-based testing pattern:
  - Use Logger.loadStateLog to load "logs/eryx.models.OnePhonon._state_before__build_A.log"
  - Use TorchTesting.initializeFromState to create model from state
  - Call model._build_A()
  - Compare resulting state with after state
  - Verify gradient flow through Amat tensor
- [x] Update tests for _build_M, _build_M_allatoms, and _project_M similarly
- [x] Verify log completeness using verify_logs.py

### 4.2 Update CP6 Tests (K-vector Methods) - COMPLETED
> Location: tests/test_models_torch_kvector.py
> Prerequisites: Task 4.1 complete

- [x] Update test for _build_kvec_Brillouin to use state-based pattern
- [x] Keep function-based tests for _center_kvec and _at_kvec_from_miller_points
- [x] Verify log completeness for necessary components
- [x] Test gradient flow through k-vector tensors

### 4.3 Update CP7 Tests (Phonon Calculation) - COMPLETED
> Location: tests/test_models_torch_phonon.py
> Prerequisites: Task 4.2 complete

- [x] Update state-based tests for compute_gnm_phonons
- [x] Update state-based tests for compute_hessian
- [x] Update state-based tests for GNM methods
- [x] Verify tensor shapes, dtypes, and gradient flow
- [x] Use verify_logs.py to check log completeness for these components

### 4.4 Update CP8 Tests (Covariance Matrix) - COMPLETED
> Location: tests/test_models_torch_covariance.py
> Prerequisites: Task 4.3 complete

- [x] Update state-based test for compute_covariance_matrix
- [x] Verify gradient flow through covariance calculations
- [x] Test for expected state changes from before to after
- [x] Check for required tensor properties (shape, dtype, etc.)

### 4.5 Update CP9 Tests (Apply Disorder) - COMPLETED
> Location: tests/test_models_torch_disorder.py
> Prerequisites: Task 4.4 complete

- [x] Update state-based test for apply_disorder
- [x] Verify end-to-end gradient flow
- [x] Test with different parameter configurations
- [x] Validate output tensor properties match expectations

## Phase 5: Integration and Documentation (1 week) - IN PROGRESS
> Reference: See "Key Data Flows" section in architecture.md

### 5.1 Update CP10 Tests (End-to-End Integration) - IN PROGRESS
> Location: tests/test_integration.py
> Prerequisites: Phase 4 complete

- [x] Update integration tests to follow state-based testing pattern:
  - Load initial state from logs 
  - Initialize model with that state
  - Run end-to-end pipeline
  - Compare with expected state
- [x] Use verify_logs.py to validate all logs
- [ ] Test gradient flow through the entire model
- [ ] Verify complete end-to-end pipeline works with state-based approach

### 5.2 Update Project Documentation - IN PROGRESS
> Prerequisites: Tasks 5.1 complete

- [x] Complete docs/state_based_testing.md
- [ ] Update architecture.md to reference state-based testing approach
- [ ] Update phased_plan.md to reflect implemented testing approach
- [x] Update progress.md to reflect current implementation status
- [ ] Create guide for adding new state-based tests for future components

### 5.3 Final Verification - NOT STARTED
> Prerequisites: Tasks 5.1-5.2 complete

- [ ] Run verify_logs.py on all generated logs
- [ ] Verify all tests pass with state-based approach
- [ ] Check documentation consistency
- [ ] Validate gradient flow throughout the entire system
- [ ] Create final report on implementation status

## Completion Criteria

This implementation plan is complete when:

1. All state-based testing infrastructure is implemented and tested
2. Ground truth state logs are generated for all stateful methods
3. Test implementations are updated to use state-based approach
4. Documentation is updated to explain the state-based testing approach
5. All tests pass and gradient flow is verified
6. Log verification confirms completeness of state capture

## Timeline

- **Phase 1**: Testing Framework Enhancement - 1 week
- **Phase 2**: Adapter Enhancement - 1 week
- **Phase 3**: Ground Truth Generation - 1 week
- **Phase 4**: Test Implementation - 2 weeks
- **Phase 5**: Integration and Documentation - 1 week

**Total: 6 weeks**
