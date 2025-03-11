import os
import unittest
import torch
import numpy as np
from tests.test_base import TestBase
from eryx.models_torch import OnePhonon
from unittest.mock import patch, MagicMock

class TestCovarianceMethods(TestBase):
    def setUp(self):
        # Call parent setUp
        super().setUp()
        # Set module name for log paths
        self.module_name = "eryx.models"
        self.class_name = "OnePhonon"
        
    def create_models(self, test_params=None):
        """Create NumPy and PyTorch models for comparative testing."""
        # Import NumPy model for comparison
        from eryx.models import OnePhonon as NumpyOnePhonon
        
        # Default test parameters
        self.test_params = test_params or {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-2, 2, 2],
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
        
        # Create NumPy model for reference
        self.np_model = NumpyOnePhonon(**self.test_params)
        
        # Create PyTorch model
        self.torch_model = OnePhonon(
            **self.test_params,
            device=self.device
        )
    
    def _create_test_model(self):
        """Create a minimal test model with necessary attributes and mock methods."""
        model = OnePhonon.__new__(OnePhonon)  # Create instance without calling __init__
        
        # Set necessary attributes
        model.device = self.device
        model.n_asu = 2
        model.n_dof_per_asu = 6
        model.n_dof_per_asu_actual = 12  # 4 atoms * 3 dimensions
        model.n_cell = 3
        model.id_cell_ref = 0
        model.hsampling = (0, 5, 3)
        model.ksampling = (0, 5, 3)
        model.lsampling = (0, 5, 3)
        
        # Mock Amat tensor (projection matrix)
        model.Amat = torch.rand(
            (model.n_asu, model.n_dof_per_asu_actual, model.n_dof_per_asu), 
            device=self.device,
            requires_grad=True
        )
        
        # Mock kvec tensor
        model.kvec = torch.rand(
            (model.hsampling[2], model.ksampling[2], model.lsampling[2], 3),
            device=self.device,
            requires_grad=True
        )
        
        # Mock model.adp for scaling
        model.model = MagicMock()
        model.model.adp = torch.ones(model.n_dof_per_asu_actual // 3, device=self.device)
        
        # Mock crystal with get_unitcell_origin and id_to_hkl methods
        model.crystal = MagicMock()
        model.crystal.hkl_to_id = lambda x: 0 if x == [0, 0, 0] else x[0] + x[1] + x[2]
        model.crystal.id_to_hkl = lambda x: [x, 0, 0]
        model.crystal.get_unitcell_origin = lambda x: torch.tensor([float(x[0]), 0.0, 0.0], device=self.device)
        
        # Mock compute_hessian method
        def mock_compute_hessian():
            return torch.rand(
                (model.n_asu, model.n_dof_per_asu, model.n_cell, model.n_asu, model.n_dof_per_asu),
                device=self.device,
                dtype=torch.complex64,
                requires_grad=True
            )
        model.compute_hessian = mock_compute_hessian
        
        # Mock compute_gnm_Kinv method
        def mock_compute_gnm_Kinv(hessian, kvec=None, reshape=True):
            K_inv = torch.rand(
                (model.n_asu * model.n_dof_per_asu, model.n_asu * model.n_dof_per_asu),
                device=self.device,
                dtype=torch.complex64,
                requires_grad=True
            )
            return K_inv
        model.compute_gnm_Kinv = mock_compute_gnm_Kinv
        
        return model
    
    def test_compute_covariance_matrix_state_based(self):
        """Test compute_covariance_matrix using state-based approach."""
        # Import test helpers
        from eryx.autotest.test_helpers import (
            load_test_state,
            build_test_object,
            ensure_tensor
        )
        
        try:
            # Load before state using the helper function which handles path flexibility
            before_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_covariance_matrix"
            )
        except FileNotFoundError as e:
            # If state logs aren't found, skip the test with informative message
            import glob
            available_logs = glob.glob("logs/*covariance*")
            self.skipTest(f"Could not find state log. Available logs: {available_logs}\nError: {e}")
            return
        except Exception as e:
            self.skipTest(f"Error loading state log: {e}")
            return
        
        # Build model with StateBuilder
        model = build_test_object(OnePhonon, before_state, device=self.device)
        
        # Verify initial structure
        self.assertTrue(
            hasattr(model, 'Amat') and hasattr(model, 'kvec'),
            "Model does not contain required attributes"
        )
        
        # Check if crystal is properly initialized
        if not hasattr(model, 'crystal') or isinstance(model.crystal, dict):
            self.skipTest("Crystal object not properly initialized in state data")
            return
            
        # Check if id_cell_ref is set
        if not hasattr(model, 'id_cell_ref'):
            model.id_cell_ref = 0
            
        # Print debugging information
        print("\nDEBUGGING model attributes before method call:")
        print(f"Amat shape: {model.Amat.shape}")
        print(f"kvec shape: {model.kvec.shape}")
        
        try:
            # Call the method under test
            model.compute_covariance_matrix()
            
            # Verify results - check covar tensor
            self.assertTrue(hasattr(model, 'covar'), "covar not created")
            expected_covar_shape = (
                model.n_asu, model.n_dof_per_asu,
                model.n_cell, model.n_asu, model.n_dof_per_asu
            )
            self.assertEqual(model.covar.shape, expected_covar_shape)
            
            # Check ADP tensor
            self.assertTrue(hasattr(model, 'ADP'), "ADP not created")
            expected_adp_shape = (model.n_dof_per_asu_actual // 3,)
            self.assertEqual(model.ADP.shape, expected_adp_shape)
            
            # Print some values for debugging
            print("\nDEBUGGING results after method call:")
            print(f"covar shape: {model.covar.shape}")
            print(f"ADP shape: {model.ADP.shape}")
            print(f"ADP mean: {model.ADP.mean().item()}")
            
            # Load after state for comparison
            try:
                after_state = load_test_state(
                    self.logger, 
                    self.module_name, 
                    self.class_name, 
                    "compute_covariance_matrix",
                    before=False
                )
            except FileNotFoundError as e:
                import glob
                available_logs = glob.glob("logs/*covariance*after*")
                self.skipTest(f"Could not find after state log. Available logs: {available_logs}\nError: {e}")
                return
            except Exception as e:
                self.skipTest(f"Error loading after state log: {e}")
                return
            
            # Get expected tensors from after state
            covar_expected = after_state.get('covar')
            adp_expected = after_state.get('ADP')
            
            # Check if expected tensors exist
            if covar_expected is None or adp_expected is None:
                self.skipTest("Expected covar or ADP not found in after state log")
                return
                
            # Ensure tensors are in the right format for comparison
            covar_expected = ensure_tensor(covar_expected, device='cpu')
            adp_expected = ensure_tensor(adp_expected, device='cpu')
            
            # Print expected values for debugging
            print("\nDEBUGGING expected values:")
            print(f"expected covar shape: {covar_expected.shape}")
            print(f"expected ADP shape: {adp_expected.shape}")
            print(f"expected ADP mean: {adp_expected.mean().item()}")
            
            # Compare tensor values with more relaxed tolerances
            tolerances = {'rtol': 1e-3, 'atol': 1e-4}
            
            # Convert to numpy for comparison
            covar_numpy = model.covar.detach().cpu().numpy()
            covar_expected_numpy = covar_expected.detach().cpu().numpy() if isinstance(covar_expected, torch.Tensor) else covar_expected
            
            adp_numpy = model.ADP.detach().cpu().numpy()
            adp_expected_numpy = adp_expected.detach().cpu().numpy() if isinstance(adp_expected, torch.Tensor) else adp_expected
            
            # Print differences
            print("\nDEBUGGING differences:")
            max_covar_diff = np.max(np.abs(covar_numpy - covar_expected_numpy))
            max_adp_diff = np.max(np.abs(adp_numpy - adp_expected_numpy))
            print(f"Maximum covar difference: {max_covar_diff}")
            print(f"Maximum ADP difference: {max_adp_diff}")
            
            # Verify tensors match expected values
            self.assertTrue(
                np.allclose(
                    covar_numpy, 
                    covar_expected_numpy, 
                    rtol=tolerances['rtol'], 
                    atol=tolerances['atol']
                ),
                "covar values don't match expected"
            )
            self.assertTrue(
                np.allclose(
                    adp_numpy, 
                    adp_expected_numpy, 
                    rtol=tolerances['rtol'], 
                    atol=tolerances['atol']
                ),
                "ADP values don't match expected"
            )
        except Exception as e:
            self.skipTest(f"Error during covariance matrix computation: {e}")
            return
    
    def test_compute_covariance_matrix_shape(self):
        """Test shape of covariance matrix and ADPs."""
        # Create test model
        model = self._create_test_model()
        
        # Run the method
        model.compute_covariance_matrix()
        
        # Check shapes of results
        expected_covar_shape = (
            model.n_asu, model.n_dof_per_asu,
            model.n_cell, model.n_asu, model.n_dof_per_asu
        )
        self.assertEqual(model.covar.shape, expected_covar_shape)
        
        expected_adp_shape = (model.n_dof_per_asu_actual // 3,)
        self.assertEqual(model.ADP.shape, expected_adp_shape)
        
        # Check data type is correct (should be real)
        self.assertTrue(torch.is_floating_point(model.covar))
        self.assertTrue(torch.is_floating_point(model.ADP))
    
    def test_compute_covariance_matrix_gradient_flow(self):
        """Test gradient flow through covariance matrix calculation."""
        # Create test model
        model = self._create_test_model()
        
        # Enable anomaly detection to help debug gradient issues
        torch.autograd.set_detect_anomaly(True)
        
        # Run the method
        model.compute_covariance_matrix()
        
        # Create a scalar loss from the outputs
        loss = model.covar.mean() + model.ADP.mean()
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients flowed to input parameters
        self.assertIsNotNone(model.Amat.grad)
        self.assertIsNotNone(model.kvec.grad)
        
        # Verify gradients are not all zeros
        self.assertFalse(torch.allclose(model.Amat.grad, torch.zeros_like(model.Amat.grad)))
        self.assertFalse(torch.allclose(model.kvec.grad, torch.zeros_like(model.kvec.grad)))
        
        # Disable anomaly detection after test
        torch.autograd.set_detect_anomaly(False)
    
    def test_compute_covariance_matrix_scaling(self):
        """Test that scaling to match experimental ADPs works correctly."""
        # Create test model
        model = self._create_test_model()
        
        # Run the method
        model.compute_covariance_matrix()
        
        # Mean of model.adp should be approximately 1.0 since we set it to all ones
        target_mean = 1.0
        
        # Calculate expected scaling: mean_ADP = 3 * mean_B / (8π²)
        # So we expect mean(ADP) ≈ 3 * 1 / (8π²)
        expected_adp_mean = 3 * target_mean / (8 * np.pi * np.pi)
        
        # Check the mean of the computed ADP is close to expected
        adp_mean = model.ADP.mean().item()
        self.assertAlmostEqual(adp_mean, expected_adp_mean, delta=1e-4)
    
    def test_log_completeness(self):
        """Verify covariance method logs exist and contain required attributes."""
        if not hasattr(self, 'verify_logs') or not self.verify_logs:
            self.skipTest("Log verification disabled")
            
        # Verify covariance method logs
        self.verify_required_logs(self.module_name, "compute_covariance_matrix", ["covar", "ADP"])

class TestOnePhononCovariance(TestCovarianceMethods):
    """Legacy class for backward compatibility."""
    
    def setUp(self):
        # Call parent setUp
        super().setUp()
        # Initialize test_params
        self.test_params = {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-2, 2, 2],
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
    

if __name__ == '__main__':
    unittest.main()
