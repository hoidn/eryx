# GaussianNetworkModel Test Specification

## High-Level Objective
- Implement comprehensive test suite for the GaussianNetworkModel, validating both computational correctness and core functionality through logs and reference data

## Mid-Level Objectives
- Create base test class structure
- Implement neighbor list validation tests 
- Implement Hessian and K-matrix computation tests
- Test matrix symmetry properties
- Add edge case tests for numerical stability

## Implementation Notes

### Dependencies and Requirements
- pytest
- numpy
- Generated reference data
- Generated test logs (showing matrix computations)
- Test utilities from infrastructure spec
- Base and edge case configs

### Test Data Structure
```
tests/test_data/
  configs/
    gnm_base_config.yaml           # Base GNM test configuration
    gnm_edge_configs/*.yaml        # Edge case configs
  logs/
    gnm_base/base_run.log         # Standard GNM run logs
    gnm_edge/edge_run_*.log       # Edge case logs
  reference/
    gnm_neighbor_lists.npy        # Reference neighbor lists
    gnm_hessian.npy              # Reference Hessian matrices
    gnm_k_matrices.npy           # Reference K matrices
    gnm_test_params.npz          # Test parameters
```

## Test Classes Structure

1. TestGaussianNetworkModel - Main test class
```python
class TestGaussianNetworkModel:
    @pytest.fixture
    def gnm_base_log_file(self) -> str
    @pytest.fixture
    def gnm_edge_log_file(self) -> str
    @pytest.fixture
    def gnm_model(self) -> GaussianNetworkModel

    def test_neighbor_list_construction(self, gnm_model)
    def test_hessian_symmetry(self, gnm_model)
    def test_k_matrix_computation(self, gnm_model)
    def test_inversion_stability(self, gnm_model)
    def test_spring_constants(self, gnm_model)
    def test_phase_factors(self, gnm_model)
```

## Test Cases

1. Neighbor List Tests
```python
def test_neighbor_list_construction(self, gnm_model):
    """Verify neighbor list computation"""
    # Test against known reference values:
    - Number of neighbors
    - Distance cutoffs
    - Intra vs inter molecular pairs
    - Expected connectivity
    
def test_spring_constants(self, gnm_model):
    """Verify spring constant assignment"""
    # Test against reference values:
    - Intra-molecular gamma values 
    - Inter-molecular gamma values
    - Cutoff behavior
```

2. Matrix Computation Tests
```python
def test_hessian_symmetry(self, gnm_model):
    """Test Hessian matrix properties"""
    # Validate:
    - Matrix symmetry
    - Block structure
    - Eigenvalue properties
    - Reference values from logs
    
def test_k_matrix_computation(self, gnm_model):
    """Test K matrix computation with phases"""
    expected_central_block = np.array([
        # Reference values from test logs
    ])
```

3. Numerical Stability Tests
```python
def test_inversion_stability(self, gnm_model):
    """Test matrix inversion stability"""
    # Test:
    - Small eigenvalue handling
    - Regularization effects
    - Condition number limits
    - Inverse accuracy
    
def test_phase_factors(self, gnm_model):
    """Test phase factor handling in K matrix"""
    # Test:
    - Phase computation accuracy
    - Complex number handling
    - k-vector boundary cases
```

4. Edge Case Tests
```python
def test_edge_case_small_cutoff(self, gnm_edge_log_file):
    """Test behavior with very small cutoff"""
    # Test disconnected components handling
    
def test_edge_case_large_system(self, gnm_edge_log_file):
    """Test behavior with large system size"""
    # Test numerical stability
```

## Reference Data Generation Plan
```python
def generate_gnm_reference_data():
    """Generate reference values for GNM tests"""
    # For small test system (e.g. pentamer):
    1. Generate and save neighbor lists
    2. Compute and save reference Hessian
    3. Compute and save reference K matrices
    4. Save key parameters used
    5. Generate log file with matrix values
```

Implementation Notes:
1. Use same logging infrastructure as OnePhonon tests
2. Follow same numerical comparison standards (rtol=1e-7)
3. Generate reference data using known stable configuration
4. Test both real and complex matrix operations
5. Validate matrix properties before and after operations

Success Criteria:
1. All GNM-specific tests pass
2. Reference values match within tolerances
3. Edge cases properly handled
4. Matrix properties preserved
5. Complete log validation coverage
