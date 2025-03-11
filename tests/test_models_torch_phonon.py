import unittest
import os
import torch
import numpy as np
from tests.test_base import TestBase
from eryx.models_torch import OnePhonon
from eryx.pdb_torch import GaussianNetworkModel as GaussianNetworkModelTorch
from eryx.autotest.test_helpers import load_test_state, build_test_object, ensure_tensor, verify_gradient_flow

class TestOnePhononPhonon(TestBase):
    def setUp(self):
        # Call parent setUp
        super().setUp()
        
        # Set module name for log paths
        self.module_name = "eryx.models"
        self.class_name = "OnePhonon"
        
        # Create minimal test model
        self.model = self._create_test_model()
    
    def _create_test_model(self):
        """Create a minimal test model with necessary attributes for phonon methods."""
        model = OnePhonon.__new__(OnePhonon)  # Create instance without calling __init__
        
        # Set necessary attributes
        model.device = self.device
        model.n_asu = 2
        model.n_cell = 3
        model.n_atoms_per_asu = 4
        model.n_dof_per_asu = 6
        model.n_dof_per_asu_actual = 12  # n_atoms_per_asu * 3
        model.n_dof_per_cell = 12  # n_asu * n_dof_per_asu
        model.id_cell_ref = 0
        
        # Create other necessary attributes
        model.hsampling = (0, 5, 2)
        model.ksampling = (0, 5, 2)
        model.lsampling = (0, 5, 2)
        
        # Create sample matrices
        model.Amat = torch.zeros((model.n_asu, model.n_dof_per_asu_actual, model.n_dof_per_asu), 
                                device=self.device)
        model.Linv = torch.eye(model.n_dof_per_cell, device=self.device)
        
        # Create k-vectors for testing
        model.kvec = torch.zeros((model.hsampling[2], model.ksampling[2], model.lsampling[2], 3), 
                                device=self.device)
        model.kvec_norm = torch.zeros((model.hsampling[2], model.ksampling[2], model.lsampling[2], 1), 
                                     device=self.device)
        
        # Sample methods that would be called
        model.id_to_hkl = lambda cell_id: [0, 0, 0]
        model.get_unitcell_origin = lambda unit_cell: torch.tensor([0.0, 0.0, 0.0], device=self.device)
        
        return model
    
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
    
    def test_compute_hessian(self):
        """Test compute_hessian method using state-based approach."""
        try:
            # Load before state
            before_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_hessian"
            )
            
            # Load after state for expected output
            after_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_hessian",
                before=False
            )
        except FileNotFoundError as e:
            # If state logs aren't found, skip the test with informative message
            import glob
            available_logs = glob.glob("logs/*compute_hessian*")
            self.skipTest(f"Could not find state log. Available logs: {available_logs}\nError: {e}")
            return
        except Exception as e:
            self.skipTest(f"Error loading state log: {e}")
            return
        
        # Build model with StateBuilder
        model = build_test_object(OnePhonon, before_state, device=self.device)
        
        # Call the method under test
        hessian = model.compute_hessian()
        
        # Verify result properties
        self.assertIsInstance(hessian, torch.Tensor, "Result should be a tensor")
        self.assertEqual(hessian.dtype, torch.complex64, "Result should be complex64")
        
        # Expected shape based on model attributes
        expected_shape = (model.n_asu, model.n_dof_per_asu, 
                          model.n_cell, model.n_asu, model.n_dof_per_asu)
        self.assertEqual(hessian.shape, expected_shape, "Hessian has incorrect shape")
        
        # Get expected output from after state
        expected_hessian = after_state.get('hessian')
        if expected_hessian is not None:
            # Convert to tensor for comparison
            expected_hessian = ensure_tensor(expected_hessian, device='cpu')
            
            # Compare with expected output
            hessian_np = hessian.detach().cpu().numpy()
            expected_np = expected_hessian.detach().cpu().numpy() if isinstance(expected_hessian, torch.Tensor) else expected_hessian
            
            self.assertTrue(np.allclose(hessian_np, expected_np, rtol=1e-5, atol=1e-8),
                          "compute_hessian doesn't match ground truth")
    
    def test_compute_gnm_phonons(self):
        """Test compute_gnm_phonons method using state-based approach."""
        try:
            # Load before state
            before_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_gnm_phonons"
            )
            
            # Load after state for expected output
            after_state = load_test_state(
                self.logger, 
                self.module_name, 
                self.class_name, 
                "compute_gnm_phonons",
                before=False
            )
        except FileNotFoundError as e:
            # If state logs aren't found, skip the test with informative message
            import glob
            available_logs = glob.glob("logs/*compute_gnm_phonons*")
            self.skipTest(f"Could not find state log. Available logs: {available_logs}\nError: {e}")
            return
        except Exception as e:
            self.skipTest(f"Error loading state log: {e}")
            return
        
        # Build model with StateBuilder
        model = build_test_object(OnePhonon, before_state, device=self.device)
        
        # Call the method under test
        model.compute_gnm_phonons()
        
        # Verify result properties - V and Winv should be created
        self.assertTrue(hasattr(model, 'V'), "V tensor not created")
        self.assertTrue(hasattr(model, 'Winv'), "Winv tensor not created")
        
        # Get expected tensors from after state
        expected_V = after_state.get('V')
        expected_Winv = after_state.get('Winv')
        
        if expected_V is not None and expected_Winv is not None:
            # Convert to tensors for comparison
            expected_V = ensure_tensor(expected_V, device='cpu')
            expected_Winv = ensure_tensor(expected_Winv, device='cpu')
            
            # Check shapes first
            self.assertEqual(model.V.shape, expected_V.shape,
                           f"V shape mismatch: {model.V.shape} vs {expected_V.shape}")
            self.assertEqual(model.Winv.shape, expected_Winv.shape,
                           f"Winv shape mismatch: {model.Winv.shape} vs {expected_Winv.shape}")
            
            # Convert to numpy for comparison
            V_np = model.V.detach().cpu().numpy()
            Winv_np = model.Winv.detach().cpu().numpy()
            expected_V_np = expected_V.detach().cpu().numpy() if isinstance(expected_V, torch.Tensor) else expected_V
            expected_Winv_np = expected_Winv.detach().cpu().numpy() if isinstance(expected_Winv, torch.Tensor) else expected_Winv
            
            # Use lower tolerance for eigendecomposition
            rtol = 1e-4
            atol = 1e-6
            
            # Check values - eigenvectors may differ by a phase factor, so check their absolute values
            V_match = np.allclose(np.abs(V_np), np.abs(expected_V_np), rtol=rtol, atol=atol)
            Winv_match = np.allclose(Winv_np, expected_Winv_np, rtol=rtol, atol=atol, 
                                   equal_nan=True)  # Handle NaN values
            
            self.assertTrue(V_match, "Eigenvectors V don't match ground truth")
            self.assertTrue(Winv_match, "Eigenvalues Winv don't match ground truth")
    
    def test_gradient_flow(self):
        """Test gradient flow through phonon calculation methods."""
        # Enable anomaly detection to help debug gradient issues
        torch.autograd.set_detect_anomaly(True)
        
        # Create test model
        model = self._create_test_model()
        
        # Test gradient flow through compute_hessian
        # Create a simple hessian matrix with gradient tracking
        hessian = torch.ones((model.n_asu, model.n_atoms_per_asu,
                             model.n_cell, model.n_asu, model.n_atoms_per_asu),
                            dtype=torch.complex64, device=self.device, requires_grad=True)
        
        # Create a GaussianNetworkModelTorch instance for testing
        gnm = GaussianNetworkModelTorch()
        gnm.device = self.device
        gnm.n_asu = model.n_asu
        gnm.n_cell = model.n_cell
        gnm.id_cell_ref = model.id_cell_ref
        
        # Create a simple k-vector with gradient tracking
        kvec = torch.ones(3, device=self.device, requires_grad=True)
        
        # Mock required methods
        gnm.crystal = {
            'id_to_hkl': lambda cell_id: [cell_id, 0, 0],
            'get_unitcell_origin': lambda unit_cell: torch.tensor(
                [float(unit_cell[0]), 0.0, 0.0], device=self.device, requires_grad=True)
        }
        
        # Test compute_K
        Kmat = gnm.compute_K(hessian, kvec)
        
        # Create a loss function
        loss = torch.abs(Kmat).sum()
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients flowed back to inputs
        self.assertIsNotNone(hessian.grad)
        self.assertIsNotNone(kvec.grad)
        self.assertFalse(torch.allclose(hessian.grad, torch.zeros_like(hessian.grad)),
                        "No gradient flow to hessian in compute_K")
        self.assertFalse(torch.allclose(kvec.grad, torch.zeros_like(kvec.grad)),
                        "No gradient flow to kvec in compute_K")
        
        # Reset gradients
        hessian.grad = None
        kvec.grad = None
        
        # Test compute_Kinv
        Kinv = gnm.compute_Kinv(hessian, kvec)
        
        # Create a loss function
        loss = torch.abs(Kinv).sum()
        
        # Compute gradients
        loss.backward()
        
        # Check that gradients flowed back to inputs
        self.assertIsNotNone(hessian.grad)
        self.assertIsNotNone(kvec.grad)
        self.assertFalse(torch.allclose(hessian.grad, torch.zeros_like(hessian.grad)),
                        "No gradient flow to hessian in compute_Kinv")
        self.assertFalse(torch.allclose(kvec.grad, torch.zeros_like(kvec.grad)),
                        "No gradient flow to kvec in compute_Kinv")
        
        # Disable anomaly detection after test
        torch.autograd.set_detect_anomaly(False)
    
    def test_log_completeness(self):
        """Verify phonon-related logs exist and contain required attributes."""
        if not hasattr(self, 'verify_logs') or not self.verify_logs:
            self.skipTest("Log verification disabled")
            
        # Verify phonon method logs
        self.verify_required_logs(self.module_name, "compute_gnm_phonons", ["V", "Winv"])
        self.verify_required_logs(self.module_name, "compute_hessian", ["hessian"])
    
if __name__ == '__main__':
    unittest.main()
