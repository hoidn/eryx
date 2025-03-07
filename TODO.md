

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

## Phase 1: Testing Framework Enhancement (1 week)
> Reference: See "Testing Framework Extensions" section in architecture.md and "State-Based Testing Guidelines" in project_rules.md

### 1.1 Extend Logger for State Capture
> Location: eryx/autotest/logger.py
> Reference: See "Data Storage and Access" section in phased_plan.md for log naming conventions

- [ ] Add `captureState(obj)` method to Logger class in eryx/autotest/logger.py
  ```python
  def captureState(self, obj):
      """
      Capture complete object state for testing.
      
      Args:
          obj: Object whose state should be captured
          
      Returns:
          Dictionary with serialized state
      """
      state = {}
      for attr_name in dir(obj):
          if attr_name.startswith('_'):
              continue  # Skip private attributes
          attr = getattr(obj, attr_name)
          if callable(attr):
              continue  # Skip methods
          # Convert attribute to serializable form
          state[attr_name] = self.serializer.serialize(attr)
      return state
  ```
- [ ] Add `loadStateLog(log_file)` method in eryx/autotest/logger.py
- [ ] Add `searchStateLogDirectory(log_path_prefix)` method in eryx/autotest/logger.py
- [ ] Implement state serialization for complex objects in eryx/autotest/serializer.py
- [ ] Add attribute filtering mechanisms for large objects

### 1.2 Add State Comparison Utilities
> Location: eryx/autotest/torch_testing.py
> Reference: See "State Comparison" section in architecture.md

- [ ] Add `compareStates(expected_state, actual_state, tolerances)` method to TorchTesting
  ```python
  def compareStates(self, expected_state, actual_state, tolerances=None):
      """
      Compare expected and actual states with appropriate tolerances.
      
      Args:
          expected_state: Expected state dictionary/object
          actual_state: Actual state dictionary/object
          tolerances: Dict with attribute-specific tolerances
          
      Returns:
          Boolean indicating if states match within tolerances
      """
      tolerances = tolerances or {'default': {'rtol': 1e-5, 'atol': 1e-8}}
      
      for key in expected_state:
          if key not in actual_state:
              print(f"Missing attribute in actual state: {key}")
              return False
              
          expected = expected_state[key]
          actual = actual_state[key]
          
          # Get tolerance for this attribute
          tol = tolerances.get(key, tolerances['default'])
          
          if isinstance(expected, np.ndarray) and isinstance(actual, torch.Tensor):
              # Convert tensor to numpy for comparison
              actual = actual.detach().cpu().numpy()
              
          if isinstance(expected, np.ndarray) and isinstance(actual, np.ndarray):
              if not np.allclose(expected, actual, rtol=tol['rtol'], atol=tol['atol']):
                  print(f"Array mismatch for attribute: {key}")
                  return False
          elif expected != actual:
              print(f"Value mismatch for attribute: {key}")
              return False
              
      return True
  ```
- [ ] Implement specialized tensor comparison with tolerance support
- [ ] Add detailed reporting for state differences
- [ ] Add gradient verification for state components

### 1.3 Create State Initialization Utilities
> Location: eryx/autotest/torch_testing.py
> Reference: See "State Restoration" section in architecture.md

- [ ] Add `initializeFromState(torch_class, state_data, device)` method to TorchTesting
  ```python
  def initializeFromState(self, torch_class, state_data, device=None):
      """
      Initialize a PyTorch object from state data.
      
      Args:
          torch_class: PyTorch class to initialize
          state_data: State data dictionary
          device: PyTorch device to place tensors on
          
      Returns:
          Initialized PyTorch object
      """
      if device is None:
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          
      # Create empty instance
      obj = torch_class.__new__(torch_class)
      
      # Initialize each attribute
      for key, value in state_data.items():
          if isinstance(value, np.ndarray):
              # Convert NumPy arrays to PyTorch tensors
              tensor = torch.tensor(value, device=device)
              if tensor.dtype.is_floating_point:
                  tensor.requires_grad_(True)
              setattr(obj, key, tensor)
          else:
              setattr(obj, key, value)
              
      return obj
  ```
- [ ] Implement NumPy to PyTorch conversion for state data
- [ ] Add support for device placement during state initialization
- [ ] Add gradient configuration options for state tensors

### 1.4 Update Testing Documentation
> Location: eryx/autotest/README.md and docs/testing.md
> Reference: See "Testing Approaches" section in architecture.md

- [ ] Document state-based testing approach
- [ ] Add examples of state-based test methods
- [ ] Update testing guides with state capture/comparison approaches

## Phase 2: Adapter Enhancement (1 week)
> Reference: See "Adapter Components" section in architecture.md and "Adapter Pattern" in project_rules.md

### 2.1 Enhance PDBToTensor Adapter
> Location: eryx/adapters.py
> Prerequisites: Tasks 1.1-1.3 complete
> Reference: See "Adapter Usage Pattern" section in architecture.md

- [ ] Extend `convert_atomic_model()` to handle complete object state
  ```python
  def convert_atomic_model(self, model, include_methods=False):
      """
      Convert an AtomicModel to PyTorch tensors.
      
      Args:
          model: AtomicModel instance
          include_methods: Whether to include method results in conversion
          
      Returns:
          Dictionary with PyTorch tensor versions of model attributes
      """
      result = {}
      
      # Handle array attributes
      for attr_name in ['xyz', 'ff_a', 'ff_b', 'ff_c', 'adp', 'cell', 'A_inv']:
          if hasattr(model, attr_name):
              attr_value = getattr(model, attr_name)
              if attr_value is not None:
                  result[attr_name] = self.array_to_tensor(attr_value)
      
      # Handle non-array attributes
      for attr_name in ['space_group', 'n_asu', 'n_conf']:
          if hasattr(model, attr_name):
              result[attr_name] = getattr(model, attr_name)
              
      return result
  ```
- [ ] Add `convert_state_dict()` method for general state conversion
- [ ] Update `convert_crystal()` and `convert_gnm()` to support state-based testing
- [ ] Add comprehensive tests for state conversion in tests/test_adapters.py

### 2.2 Enhance TensorToNumpy Adapter
> Location: eryx/adapters.py
> Prerequisites: Task 2.1 complete

- [ ] Add `convert_state_to_numpy()` method for returning state to NumPy
- [ ] Implement proper gradient detachment for state components
- [ ] Add tests for state conversion back to NumPy in tests/test_adapters.py

### 2.3 Add ModelAdapters State Support
> Location: eryx/adapters.py
> Prerequisites: Tasks 2.1-2.2 complete
> Reference: See "OnePhonon Model" section in architecture.md

- [ ] Add state-based methods to ModelAdapters
- [ ] Implement `initialize_from_state()` helper methods
- [ ] Add state conversion utilities for OnePhonon

### 2.4 Update Adapter Documentation
> Location: docs/adapters.md
> Reference: See "Adapter Usage Guidelines" section in project_rules.md

- [ ] Document adapter usage patterns
- [ ] Add examples of state conversion with adapters
- [ ] Update adapter interfaces documentation

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

## Phase 4: Test Implementation (2 weeks)
> Reference: See "Component-to-Test Mapping" table in phased_plan.md

### 4.1 Update CP5 Tests (Matrix Construction)
> Location: tests/test_models_torch.py
> Prerequisites: Phases 1-3 complete

- [ ] Update test_build_A to use state-based testing pattern:
  - Use Logger.loadStateLog to load "logs/eryx.models.OnePhonon._state_before__build_A.log"
  - Use TorchTesting.initializeFromState to create model from state
  - Call model._build_A()
  - Compare resulting state with after state
  - Verify gradient flow through Amat tensor
- [ ] Update tests for _build_M, _build_M_allatoms, and _project_M similarly
- [ ] Verify log completeness using verify_logs.py

### 4.2 Update CP6 Tests (K-vector Methods)
> Location: tests/test_models_torch_kvector.py
> Prerequisites: Task 4.1 complete

- [ ] Update test for _build_kvec_Brillouin to use state-based pattern
- [ ] Keep function-based tests for _center_kvec and _at_kvec_from_miller_points
- [ ] Verify log completeness for necessary components
- [ ] Test gradient flow through k-vector tensors

### 4.3 Update CP7 Tests (Phonon Calculation)
> Location: tests/test_models_torch_phonon.py
> Prerequisites: Task 4.2 complete

- [ ] Update state-based tests for compute_gnm_phonons
- [ ] Update state-based tests for compute_hessian
- [ ] Update state-based tests for GNM methods
- [ ] Verify tensor shapes, dtypes, and gradient flow
- [ ] Use verify_logs.py to check log completeness for these components

### 4.4 Update CP8 Tests (Covariance Matrix)
> Location: tests/test_models_torch_covariance.py
> Prerequisites: Task 4.3 complete

- [ ] Update state-based test for compute_covariance_matrix
- [ ] Verify gradient flow through covariance calculations
- [ ] Test for expected state changes from before to after
- [ ] Check for required tensor properties (shape, dtype, etc.)

### 4.5 Update CP9 Tests (Apply Disorder)
> Location: tests/test_models_torch_disorder.py
> Prerequisites: Task 4.4 complete

- [ ] Update state-based test for apply_disorder
- [ ] Verify end-to-end gradient flow
- [ ] Test with different parameter configurations
- [ ] Validate output tensor properties match expectations

## Phase 5: Integration and Documentation (1 week)
> Reference: See "Key Data Flows" section in architecture.md

### 5.1 Update CP10 Tests (End-to-End Integration)
> Location: tests/test_integration.py
> Prerequisites: Phase 4 complete

- [ ] Update integration tests to follow state-based testing pattern:
  - Load initial state from logs 
  - Initialize model with that state
  - Run end-to-end pipeline
  - Compare with expected state
- [ ] Use verify_logs.py to validate all logs
- [ ] Test gradient flow through the entire model
- [ ] Verify complete end-to-end pipeline works with state-based approach

### 5.2 Update Project Documentation
> Prerequisites: Tasks 5.1 complete

- [ ] Complete docs/state_based_testing.md if not already finished
- [ ] Update architecture.md to reference state-based testing approach
- [ ] Update phased_plan.md to reflect implemented testing approach
- [ ] Update progress.md to reflect current implementation status
- [ ] Create guide for adding new state-based tests for future components

### 5.3 Final Verification
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
