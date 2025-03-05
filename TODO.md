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

### 3.1 Extend run_debug.py for State Capture
> Location: eryx/autotest/debug.py and eryx/run_debug.py
> Prerequisites: Phase 1 complete

- [ ] Modify @debug decorator to capture before/after state
  ```python
  def debug(func):
      """
      Decorator that logs function inputs/outputs and object state for testing.
      """
      @functools.wraps(func)
      def wrapper(*args, **kwargs):
          # Get logger and serializer
          logger = Logger()
          
          # Capture class instance state for methods
          if args and hasattr(args[0], '__dict__'):
              obj = args[0]
              class_name = obj.__class__.__name__
              method_name = func.__name__
              
              # Capture pre-execution state
              before_state = logger.captureState(obj)
              logger.saveStateLog(f"{class_name}._state_before_{method_name}", before_state)
          
          # Original function call and logging
          result = func(*args, **kwargs)
          
          # Capture post-execution state
          if args and hasattr(args[0], '__dict__'):
              after_state = logger.captureState(obj)
              logger.saveStateLog(f"{class_name}._state_after_{method_name}", after_state)
          
          return result
      return wrapper
  ```
- [ ] Add configuration parameters for state capture depth
- [ ] Update serialization for complex nested structures
- [ ] Add timing measurements for state capture overhead

### 3.2 Generate State-Based Ground Truth Data
> Location: logs/eryx.*.log files
> Prerequisites: Task 3.1 complete
> Reference: See "Component-to-Test Mapping" table in phased_plan.md

- [ ] Generate state logs for OnePhonon methods:
  - [ ] Matrix construction methods (_build_A, _build_M, etc.)
  - [ ] K-vector methods (_build_kvec_Brillouin)
  - [ ] Phonon calculation methods (compute_gnm_phonons, compute_hessian)
  - [ ] Covariance matrix method (compute_covariance_matrix)
  - [ ] Disorder application method (apply_disorder)
- [ ] Generate state logs for GaussianNetworkModel methods
- [ ] Validate state log completeness

### 3.3 Implement State Log Verification
> Location: scripts/verify_state_logs.py
> Prerequisites: Task 3.2 complete

- [ ] Add verification script for state log validity
- [ ] Check state logs for expected attributes
- [ ] Validate state completeness for restoration
- [ ] Measure state log sizes and optimize if needed

## Phase 4: Test Implementation (2 weeks)
> Reference: See "Component-to-Test Mapping" table in phased_plan.md

### 4.1 Update CP5 Tests (Matrix Construction)
> Location: tests/test_models_torch.py
> Prerequisites: Phases 1-3 complete

- [ ] Create state-based tests for _build_A
  ```python
  def test_build_A_state_based(self):
      """Test _build_A with state-based approach."""
      # Load before state
      before_state = self.logger.loadStateLog("logs/eryx.models.OnePhonon._state_before__build_A.log")
      
      # Initialize PyTorch object with state
      model_torch = self.torch_testing.initializeFromState(
          OnePhonon, before_state, device=torch.device('cpu'))
      
      # Call method under test
      model_torch._build_A()
      
      # Load expected after state
      expected_after_state = self.logger.loadStateLog(
          "logs/eryx.models.OnePhonon._state_after__build_A.log")
      
      # Compare states
      self.assertTrue(
          self.torch_testing.compareStates(expected_after_state, model_torch.__dict__),
          "_build_A failed state comparison")
          
      # Verify gradient flow
      self.assertTrue(
          model_torch.Amat.requires_grad,
          "Gradient not enabled for Amat tensor")
  ```
- [ ] Create state-based tests for _build_M
- [ ] Create state-based tests for _build_M_allatoms
- [ ] Create state-based tests for _project_M
- [ ] Verify gradient flow through matrix operations

### 4.2 Update CP6 Tests (K-vector Methods)
> Location: tests/test_models_torch_kvector.py
> Prerequisites: Task 4.1 complete

- [ ] Create state-based test for _build_kvec_Brillouin
- [ ] Keep function-based tests for _center_kvec
- [ ] Keep function-based tests for _at_kvec_from_miller_points
- [ ] Verify combined test approach works correctly

### 4.3 Update CP7 Tests (Phonon Calculation)
> Location: tests/test_models_torch_phonon.py
> Prerequisites: Task 4.2 complete

- [ ] Create state-based test for compute_gnm_phonons
- [ ] Create state-based test for compute_hessian
- [ ] Create state-based tests for GNM methods
- [ ] Verify gradient flow through eigendecomposition

### 4.4 Update CP8 Tests (Covariance Matrix)
> Location: tests/test_models_torch_covariance.py
> Prerequisites: Task 4.3 complete

- [ ] Create state-based test for compute_covariance_matrix
- [ ] Verify gradient flow through covariance calculations
- [ ] Test scaling to match experimental ADPs

### 4.5 Update CP9 Tests (Apply Disorder)
> Location: tests/test_models_torch_disorder.py
> Prerequisites: Task 4.4 complete

- [ ] Create state-based test for apply_disorder
- [ ] Verify end-to-end gradient flow
- [ ] Test with different parameter configurations

## Phase 5: Integration Updates (1 week)
> Reference: See "Key Data Flows" section in architecture.md

### 5.1 Update CP10 Tests (End-to-End Integration)
> Location: tests/test_integration.py
> Prerequisites: Phase 4 complete

- [ ] Update integration tests to use adapter pattern explicitly
  ```python
  def test_end_to_end_integration(self):
      """Test end-to-end integration with adapter pattern."""
      # Initialize adapters
      pdb_adapter = PDBToTensor(device=self.device)
      grid_adapter = GridToTensor(device=self.device)
      results_adapter = TensorToNumpy()
      
      # Load NumPy model
      np_model = NumpyOnePhonon(**self.test_params)
      
      # Apply disorder with NumPy model
      Id_np = np_model.apply_disorder(use_data_adp=True)
      
      # Create PyTorch model with adapters
      # Note: Using adapter to convert model data
      model_data = pdb_adapter.convert_atomic_model(np_model.model)
      grid_data, map_shape = grid_adapter.convert_grid(np_model.q_grid, np_model.map_shape)
      
      # Initialize PyTorch model with converted data
      torch_model = TorchOnePhonon(
          pdb_path=self.test_params['pdb_path'],
          hsampling=self.test_params['hsampling'],
          ksampling=self.test_params['ksampling'],
          lsampling=self.test_params['lsampling'],
          device=self.device
      )
      
      # Apply disorder with PyTorch model
      Id_torch = torch_model.apply_disorder(use_data_adp=True)
      
      # Convert PyTorch output back to NumPy for comparison
      Id_torch_np = results_adapter.tensor_to_array(Id_torch)
      
      # Compare outputs
      self.assertTrue(np.allclose(Id_np, Id_torch_np, rtol=1e-4, atol=1e-7))
  ```
- [ ] Add gradient flow verification through the entire pipeline
- [ ] Add device handling verification
- [ ] Update performance benchmarks

### 5.2 Update Documentation for New Approach
> Location: Multiple documentation files
> Prerequisites: Task 5.1 complete

- [ ] Update architecture.md with adapter patterns
- [ ] Update phased_plan.md with testing approach changes
- [ ] Update progress.md to reflect new testing status
- [ ] Create guide for adding new state-based tests

### 5.3 Final Review and Verification
> Prerequisites: Tasks 5.1-5.2 complete

- [ ] Verify all tests pass with new approach
- [ ] Check documentation consistency
- [ ] Verify gradient flow at all levels
- [ ] Benchmark performance with adapter overhead

## Completion Criteria

This implementation plan is complete when:

1. All state-based testing infrastructure is implemented and tested
2. Ground truth state logs are generated for all stateful methods
3. Test implementations are updated to use state-based approach where appropriate
4. Documentation is updated to clearly explain the adapter-based architecture
5. All tests pass and gradient flow is verified throughout the system
6. Performance is benchmarked with adapter overhead considerations

## Timeline

- **Phase 1**: Testing Framework Enhancement - 1 week
- **Phase 2**: Adapter Enhancement - 1 week
- **Phase 3**: Ground Truth Generation - 1 week
- **Phase 4**: Test Implementation - 2 weeks
- **Phase 5**: Integration Updates - 1 week

**Total: 6 weeks**
