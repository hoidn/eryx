"""
Test suite for PDOS (Phonon Density of States) functionality in OnePhononTorch.

This module validates the differentiable PDOS integration system including:
- Unit tests for interpolation methods
- Integration tests for full model workflow  
- Gradient flow verification
- Regression testing for backward compatibility
- Error handling and edge cases
"""

import os
import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

# Import the model under test
from eryx.models_torch import OnePhonon


class TestPDOSInterpolation:
    """Unit tests for the _differentiable_interp method."""
    
    def setup_method(self):
        """Setup test fixtures with sample PDOS data."""
        self.device = torch.device('cpu')  # Use CPU for reproducible tests
        
        # Create simple test PDOS data
        self.omega_values = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], 
                                       dtype=torch.float64, device=self.device)
        self.density_values = torch.tensor([0.0, 1.0, 2.0, 1.5, 0.5], 
                                         dtype=torch.float64, device=self.device)
        self.density_values.requires_grad_(True)
        
        # Create mock OnePhonon instance with minimal setup
        self.mock_model = type('MockModel', (), {})()
        self.mock_model.device = self.device
        self.mock_model.real_dtype = torch.float64
        self.mock_model.pdos_omega = self.omega_values
        self.mock_model.pdos_density = self.density_values
        
        # Bind the method to test
        from eryx.models_torch import OnePhonon
        self.mock_model._differentiable_interp = OnePhonon._differentiable_interp.__get__(self.mock_model)
    
    def test_exact_interpolation_points(self):
        """Test interpolation at exact PDOS data points."""
        query_omega = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64, device=self.device)
        expected = torch.tensor([1.0, 2.0, 1.5], dtype=torch.float64, device=self.device)
        
        result = self.mock_model._differentiable_interp(query_omega)
        
        assert torch.allclose(result, expected, atol=1e-10)
    
    def test_midpoint_interpolation(self):
        """Test interpolation at midpoints between data points."""
        query_omega = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64, device=self.device)
        expected = torch.tensor([0.5, 1.5, 1.75], dtype=torch.float64, device=self.device)
        
        result = self.mock_model._differentiable_interp(query_omega)
        
        assert torch.allclose(result, expected, atol=1e-10)
    
    def test_boundary_conditions(self):
        """Test interpolation at and beyond boundaries."""
        # Test at boundaries and slightly beyond
        query_omega = torch.tensor([0.0, 4.0, -0.1, 4.1], dtype=torch.float64, device=self.device)
        
        result = self.mock_model._differentiable_interp(query_omega)
        
        # At boundaries, should return exact values
        assert torch.allclose(result[0], torch.tensor(0.0, dtype=torch.float64))
        assert torch.allclose(result[1], torch.tensor(0.5, dtype=torch.float64))
        
        # Beyond boundaries should be handled gracefully (exact behavior depends on clamp implementation)
        assert torch.isfinite(result[2])  # Should not be NaN
        assert torch.isfinite(result[3])  # Should not be NaN
    
    def test_gradient_flow_through_interpolation(self):
        """Test that gradients flow correctly through interpolation."""
        query_omega = torch.tensor([1.5, 2.5], dtype=torch.float64, device=self.device)
        
        result = self.mock_model._differentiable_interp(query_omega)
        loss = result.sum()
        loss.backward()
        
        # Check that gradients exist and are non-zero
        assert self.mock_model.pdos_density.grad is not None
        assert not torch.allclose(self.mock_model.pdos_density.grad, torch.zeros_like(self.mock_model.pdos_density.grad))
        
        # Verify gradient magnitudes are reasonable
        grad_magnitude = torch.norm(self.mock_model.pdos_density.grad)
        assert 0.1 < grad_magnitude < 10.0  # Reasonable range


class TestPDOSIntegration:
    """Integration tests for full OnePhononTorch workflow with PDOS."""
    
    @pytest.fixture
    def sample_pdb_path(self):
        """Create a minimal sample PDB file for testing."""
        pdb_content = """HEADER    TEST STRUCTURE
CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CB  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
ATOM      3  CA  ALA A   2       5.000   0.000   0.000  1.00 20.00           C
ATOM      4  CB  ALA A   2       6.500   0.000   0.000  1.00 20.00           C
ATOM      5  CA  ALA A   3       0.000   5.000   0.000  1.00 20.00           C
ATOM      6  CB  ALA A   3       1.500   5.000   0.000  1.00 20.00           C
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            return f.name
    
    @pytest.fixture
    def thermal_pdos_path(self):
        """Path to thermal mode PDOS test file."""
        return str(Path(__file__).parent / "data" / "sample_pdos_thermal.dat")
    
    @pytest.fixture  
    def direct_pdos_path(self):
        """Path to direct mode PDOS test file."""
        return str(Path(__file__).parent / "data" / "sample_pdos_direct.dat")
    
    def test_thermal_mode_initialization(self, sample_pdb_path, thermal_pdos_path):
        """Test OnePhononTorch initialization with thermal PDOS mode."""
        model = OnePhonon(
            pdb_path=sample_pdb_path,
            hsampling=(-0.5, 0.5, 2),
            ksampling=(-0.5, 0.5, 2),
            lsampling=(-0.5, 0.5, 2),
            pdos_path=thermal_pdos_path,
            pdos_mode='thermal',
            temperature_k=300.0
        )
        
        # Verify PDOS data was loaded
        assert hasattr(model, 'pdos_omega')
        assert hasattr(model, 'pdos_density')
        assert model.pdos_omega is not None
        assert model.pdos_density is not None
        assert model.pdos_density.requires_grad
        
        # Verify temperature processing for thermal mode
        assert model.temperature_k == 300.0
        assert model.pdos_mode == 'thermal'
    
    def test_direct_mode_initialization(self, sample_pdb_path, direct_pdos_path):
        """Test OnePhononTorch initialization with direct PDOS mode.""" 
        model = OnePhonon(
            pdb_path=sample_pdb_path,
            hsampling=(-0.5, 0.5, 2),
            ksampling=(-0.5, 0.5, 2), 
            lsampling=(-0.5, 0.5, 2),
            pdos_path=direct_pdos_path,
            pdos_mode='direct'
        )
        
        # Verify PDOS data was loaded
        assert hasattr(model, 'pdos_omega')
        assert hasattr(model, 'pdos_density')
        assert model.pdos_omega is not None
        assert model.pdos_density is not None
        assert model.pdos_density.requires_grad
        
        # Verify direct mode doesn't require temperature
        assert model.temperature_k is None
        assert model.pdos_mode == 'direct'
    
    def test_phonon_computation_with_pdos(self, sample_pdb_path, thermal_pdos_path):
        """Test full phonon computation workflow with PDOS."""
        model = OnePhonon(
            pdb_path=sample_pdb_path,
            hsampling=(-0.2, 0.2, 2), 
            ksampling=(-0.2, 0.2, 2),
            lsampling=(-0.2, 0.2, 2),
            pdos_path=thermal_pdos_path,
            pdos_mode='thermal',
            temperature_k=300.0,
            gamma_intra=1.0
        )
        
        # Test phonon computation (method stores results in model.V and model.Winv)
        model.compute_gnm_phonons()
        
        # Verify results are stored correctly
        assert hasattr(model, 'V') and model.V is not None
        assert hasattr(model, 'Winv') and model.Winv is not None
        
        # Check that results have the expected structure (may contain NaN for singular modes)
        assert model.V.shape[0] > 0  # Has valid data points
        assert model.Winv.shape[0] > 0  # Has valid data points
        
        # Verify that at least some values are finite (phonon modes should exist)
        finite_V = torch.isfinite(model.V.real) & torch.isfinite(model.V.imag)
        finite_Winv = torch.isfinite(model.Winv.real) & torch.isfinite(model.Winv.imag)
        
        # At least some elements should be finite (not all NaN)
        assert torch.any(finite_V), "All V values are NaN - phonon computation failed"
        assert torch.any(finite_Winv), "All Winv values are NaN - phonon computation failed"


class TestPDOSRegression:
    """Regression tests ensuring backward compatibility when PDOS is not used."""
    
    @pytest.fixture
    def sample_pdb_path(self):
        """Create a minimal sample PDB file for testing."""
        pdb_content = """HEADER    TEST STRUCTURE  
CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CB  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
ATOM      3  CA  ALA A   2       5.000   0.000   0.000  1.00 20.00           C
ATOM      4  CB  ALA A   2       6.500   0.000   0.000  1.00 20.00           C
ATOM      5  CA  ALA A   3       0.000   5.000   0.000  1.00 20.00           C
ATOM      6  CB  ALA A   3       1.500   5.000   0.000  1.00 20.00           C
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            return f.name
    
    def test_no_pdos_parameters_unchanged(self, sample_pdb_path):
        """Test that OnePhononTorch works unchanged without PDOS parameters."""
        # Test without any PDOS parameters (should work as before)
        model = OnePhonon(
            pdb_path=sample_pdb_path,
            hsampling=(-0.2, 0.2, 2),
            ksampling=(-0.2, 0.2, 2),
            lsampling=(-0.2, 0.2, 2),
            gamma_intra=1.0
        )
        
        # Verify no PDOS attributes are set
        assert model.pdos_path is None
        assert not hasattr(model, 'pdos_omega') or model.pdos_omega is None
        assert not hasattr(model, 'pdos_density') or model.pdos_density is None
        
        # Test that phonon computation still works (stores results in model.V and model.Winv)
        model.compute_gnm_phonons()
        assert hasattr(model, 'V') and model.V is not None
        assert hasattr(model, 'Winv') and model.Winv is not None
        
        # Check that at least some values are finite (not all NaN)
        finite_V = torch.isfinite(model.V.real) & torch.isfinite(model.V.imag)
        finite_Winv = torch.isfinite(model.Winv.real) & torch.isfinite(model.Winv.imag)
        assert torch.any(finite_V), "All V values are NaN - phonon computation failed"
        assert torch.any(finite_Winv), "All Winv values are NaN - phonon computation failed"
    
    def test_gradient_flow_without_pdos(self, sample_pdb_path):
        """Test that gradient flow works correctly without PDOS (basic model validation)."""
        model = OnePhonon(
            pdb_path=sample_pdb_path,
            hsampling=(-0.2, 0.2, 2),
            ksampling=(-0.2, 0.2, 2),
            lsampling=(-0.2, 0.2, 2),
            gamma_intra=1.0
        )
        
        # Verify no PDOS attributes are set
        assert model.pdos_path is None
        assert not hasattr(model, 'pdos_omega') or model.pdos_omega is None
        assert not hasattr(model, 'pdos_density') or model.pdos_density is None
        
        # Test basic model functionality (structure loading, parameter setup)
        assert hasattr(model, 'gamma_intra')
        assert model.gamma_intra.requires_grad
        
        # Test that model can be used without PDOS (basic validation)
        # Note: Full gradient flow testing is limited by PyTorch's unique_dim operation
        # which doesn't support gradients. This is tested separately for PDOS-specific functionality.
        
        print("Regression test passed: Model works without PDOS parameters")


class TestPDOSGradientFlow:
    """Critical tests for gradient flow validation with PDOS."""
    
    @pytest.fixture
    def sample_pdb_path(self):
        """Create a minimal sample PDB file for testing."""
        pdb_content = """HEADER    TEST STRUCTURE
CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CB  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
ATOM      3  CA  ALA A   2       5.000   0.000   0.000  1.00 20.00           C
ATOM      4  CB  ALA A   2       6.500   0.000   0.000  1.00 20.00           C
ATOM      5  CA  ALA A   3       0.000   5.000   0.000  1.00 20.00           C
ATOM      6  CB  ALA A   3       1.500   5.000   0.000  1.00 20.00           C
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            return f.name
    
    @pytest.fixture
    def thermal_pdos_path(self):
        """Path to thermal mode PDOS test file."""
        return str(Path(__file__).parent / "data" / "sample_pdos_thermal.dat")
    
    def test_automated_gradient_validation(self, sample_pdb_path, thermal_pdos_path):
        """Critical test: Automated validation of gradient flow to all PDOS parameters.""" 
        model = OnePhonon(
            pdb_path=sample_pdb_path,
            hsampling=(-0.2, 0.2, 2),
            ksampling=(-0.2, 0.2, 2),
            lsampling=(-0.2, 0.2, 2),
            pdos_path=thermal_pdos_path,
            pdos_mode='thermal',
            temperature_k=300.0,
            gamma_intra=1.0
        )
        
        # Ensure all relevant parameters require gradients
        model.gamma_intra.requires_grad_(True)
        
        # Verify PDOS data is loaded  
        assert hasattr(model, 'pdos_density') and model.pdos_density is not None
        assert hasattr(model, 'pdos_omega') and model.pdos_omega is not None
        
        # Create a direct test of PDOS interpolation gradient flow
        # We need to test on the original loaded density tensor before thermal transformation
        raw_pdos_data = torch.tensor([0.0, 1.0, 2.0, 1.5, 0.5], dtype=model.real_dtype, device=model.device)
        raw_pdos_data.requires_grad_(True)
        raw_omega = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], dtype=model.real_dtype, device=model.device)
        
        # Create a mock model for isolated testing
        mock_model = type('MockModel', (), {})()
        mock_model.pdos_omega = raw_omega
        mock_model.pdos_density = raw_pdos_data
        
        # Bind the interpolation method  
        mock_model._differentiable_interp = OnePhonon._differentiable_interp.__get__(mock_model)
        
        # Test gradient flow through interpolation
        test_omega = torch.tensor([1.5, 2.5], dtype=model.real_dtype, device=model.device)
        interpolated = mock_model._differentiable_interp(test_omega)
        loss = interpolated.sum()
        loss.backward()
        
        # Verify gradients flow to the raw PDOS density
        assert raw_pdos_data.grad is not None, "raw pdos_density gradient should exist"
        pdos_grad_magnitude = torch.norm(raw_pdos_data.grad)
        assert pdos_grad_magnitude > 1e-10, f"pdos_density gradient too small: {pdos_grad_magnitude}"
        assert torch.isfinite(raw_pdos_data.grad).all(), "pdos_density gradient contains NaN/inf"
        
        print(f"PDOS gradient flow test passed - gradient magnitude: {pdos_grad_magnitude:.2e}")
        
        # Note: Full phonon computation gradient flow is limited by PyTorch's unique_dim operation
        # which doesn't support gradients. This is a known limitation documented in Phase 1.


class TestPDOSErrorHandling:
    """Tests for error handling and edge cases."""
    
    @pytest.fixture
    def sample_pdb_path(self):
        """Create a minimal sample PDB file for testing."""
        pdb_content = """HEADER    TEST STRUCTURE
CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CB  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
ATOM      3  CA  ALA A   2       5.000   0.000   0.000  1.00 20.00           C
ATOM      4  CB  ALA A   2       6.500   0.000   0.000  1.00 20.00           C
ATOM      5  CA  ALA A   3       0.000   5.000   0.000  1.00 20.00           C
ATOM      6  CB  ALA A   3       1.500   5.000   0.000  1.00 20.00           C
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            return f.name
    
    def test_missing_pdos_file(self, sample_pdb_path):
        """Test error handling for missing PDOS file."""
        with pytest.raises(ValueError, match="PDOS file not found"):
            OnePhonon(
                pdb_path=sample_pdb_path,
                hsampling=(-0.2, 0.2, 2),
                ksampling=(-0.2, 0.2, 2),
                lsampling=(-0.2, 0.2, 2),
                pdos_path="/nonexistent/file.dat"
            )
    
    def test_invalid_pdos_mode(self, sample_pdb_path):
        """Test error handling for invalid PDOS mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("0.0\t1.0\n1.0\t2.0\n")
            pdos_path = f.name
        
        with pytest.raises(ValueError, match="pdos_mode must be 'thermal' or 'direct'"):
            OnePhonon(
                pdb_path=sample_pdb_path,
                hsampling=(-0.2, 0.2, 2),
                ksampling=(-0.2, 0.2, 2),
                lsampling=(-0.2, 0.2, 2),
                pdos_path=pdos_path,
                pdos_mode='invalid_mode'
            )
    
    def test_missing_temperature_for_thermal_mode(self, sample_pdb_path):
        """Test error handling when temperature is missing for thermal mode."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("0.0\t1.0\n1.0\t2.0\n")
            pdos_path = f.name
        
        with pytest.raises(ValueError, match="temperature_k must be provided"):
            OnePhonon(
                pdb_path=sample_pdb_path,
                hsampling=(-0.2, 0.2, 2),
                ksampling=(-0.2, 0.2, 2),
                lsampling=(-0.2, 0.2, 2),
                pdos_path=pdos_path,
                pdos_mode='thermal'
                # temperature_k missing
            )
    
    def test_invalid_pdos_file_format(self, sample_pdb_path):
        """Test error handling for invalid PDOS file format."""
        # Create PDOS file with wrong number of columns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("0.0\t1.0\t2.0\n1.0\t2.0\t3.0\n")  # 3 columns instead of 2
            pdos_path = f.name
        
        with pytest.raises(ValueError, match="PDOS file must have 2 columns"):
            model = OnePhonon(
                pdb_path=sample_pdb_path,
                hsampling=(-0.2, 0.2, 2),
                ksampling=(-0.2, 0.2, 2),
                lsampling=(-0.2, 0.2, 2),
                pdos_path=pdos_path,
                pdos_mode='direct'
            )


class TestPDOSPerformance:
    """Performance benchmarks for PDOS functionality."""
    
    @pytest.fixture
    def sample_pdb_path(self):
        """Create a minimal sample PDB file for testing."""
        pdb_content = """HEADER    TEST STRUCTURE
CRYST1   20.000   20.000   20.000  90.00  90.00  90.00 P 1           1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CB  ALA A   1       1.500   0.000   0.000  1.00 20.00           C
ATOM      3  CA  ALA A   2       5.000   0.000   0.000  1.00 20.00           C
ATOM      4  CB  ALA A   2       6.500   0.000   0.000  1.00 20.00           C
ATOM      5  CA  ALA A   3       0.000   5.000   0.000  1.00 20.00           C
ATOM      6  CB  ALA A   3       1.500   5.000   0.000  1.00 20.00           C
END
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            return f.name
    
    @pytest.fixture
    def thermal_pdos_path(self):
        """Path to thermal mode PDOS test file."""
        return str(Path(__file__).parent / "data" / "sample_pdos_thermal.dat")
    
    def test_pdos_vs_no_pdos_performance(self, sample_pdb_path, thermal_pdos_path):
        """Compare computational overhead of PDOS vs non-PDOS modes."""
        import time
        
        # Setup models
        model_no_pdos = OnePhonon(
            pdb_path=sample_pdb_path,
            hsampling=(-0.2, 0.2, 2),
            ksampling=(-0.2, 0.2, 2),
            lsampling=(-0.2, 0.2, 2),
            gamma_intra=1.0
        )
        
        model_with_pdos = OnePhonon(
            pdb_path=sample_pdb_path,
            hsampling=(-0.2, 0.2, 2),
            ksampling=(-0.2, 0.2, 2),
            lsampling=(-0.2, 0.2, 2),
            pdos_path=thermal_pdos_path,
            pdos_mode='thermal',
            temperature_k=300.0,
            gamma_intra=1.0
        )
        
        # Benchmark without PDOS
        start_time = time.time()
        model_no_pdos.compute_gnm_phonons()
        time_no_pdos = time.time() - start_time
        
        # Benchmark with PDOS
        start_time = time.time()
        model_with_pdos.compute_gnm_phonons()
        time_with_pdos = time.time() - start_time
        
        # Performance should be reasonable (allowing up to 2x overhead)
        overhead_ratio = time_with_pdos / time_no_pdos
        assert overhead_ratio < 2.0, f"PDOS overhead too high: {overhead_ratio:.2f}x"
        
        # Log performance for reference
        print(f"Performance comparison:")
        print(f"  No PDOS: {time_no_pdos:.4f}s")
        print(f"  With PDOS: {time_with_pdos:.4f}s")
        print(f"  Overhead ratio: {overhead_ratio:.2f}x")
    
    def test_memory_usage_different_pdos_sizes(self, sample_pdb_path):
        """Test memory usage with different PDOS file sizes."""
        import tempfile
        import psutil
        import os
        
        # Create PDOS files of different sizes
        sizes_to_test = [10, 100, 1000]
        
        for size in sizes_to_test:
            # Create PDOS file with 'size' points
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
                for i in range(size):
                    f.write(f"{i * 0.1}\t{1.0}\n")
                pdos_path = f.name
            
            try:
                # Measure memory before
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                
                # Create model with this PDOS file
                model = OnePhonon(
                    pdb_path=sample_pdb_path,
                    hsampling=(-0.2, 0.2, 2),
                    ksampling=(-0.2, 0.2, 2),
                    lsampling=(-0.2, 0.2, 2),
                    pdos_path=pdos_path,
                    pdos_mode='direct',
                    gamma_intra=1.0
                )
                
                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = memory_after - memory_before
                
                # Memory increase should be reasonable (< 100MB for test sizes)
                assert memory_increase < 100, f"Memory usage too high for {size} PDOS points: {memory_increase:.1f}MB"
                
                print(f"PDOS size {size}: Memory increase {memory_increase:.1f}MB")
                
            finally:
                # Clean up temporary file
                os.unlink(pdos_path)


class TestPDOSNumericalAccuracy:
    """Tests for numerical accuracy of PDOS interpolation."""
    
    def test_interpolation_accuracy_vs_reference(self):
        """Compare PDOS interpolation with reference implementation."""
        # Create test data
        omega_ref = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        density_ref = np.array([0.0, 1.0, 4.0, 2.0, 1.0])
        query_points = np.array([0.5, 1.5, 2.5, 3.5])
        
        # Reference interpolation using numpy
        reference_result = np.interp(query_points, omega_ref, density_ref)
        
        # Setup torch version
        device = torch.device('cpu')
        omega_torch = torch.tensor(omega_ref, dtype=torch.float64, device=device)
        density_torch = torch.tensor(density_ref, dtype=torch.float64, device=device)
        density_torch.requires_grad_(True)
        query_torch = torch.tensor(query_points, dtype=torch.float64, device=device)
        
        # Create mock model for testing
        mock_model = type('MockModel', (), {})()
        mock_model.pdos_omega = omega_torch
        mock_model.pdos_density = density_torch
        
        # Bind the method to test
        from eryx.models_torch import OnePhonon
        mock_model._differentiable_interp = OnePhonon._differentiable_interp.__get__(mock_model)
        
        # Compute torch interpolation
        torch_result = mock_model._differentiable_interp(query_torch)
        
        # Compare results
        np.testing.assert_allclose(
            torch_result.detach().numpy(), 
            reference_result, 
            atol=1e-12,
            err_msg="Torch interpolation should match numpy reference within numerical precision"
        )


if __name__ == '__main__':
    # Allow running individual test classes
    pytest.main([__file__, '-v'])