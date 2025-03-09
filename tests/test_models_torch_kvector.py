import os
import unittest
import torch
import numpy as np
from tests.test_base import TestBase
from eryx.models_torch import OnePhonon

class TestKvectorMethods(TestBase):
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
        
    def test_build_kvec_Brillouin_state_based(self):
        # Load before state
        before_state = self._load_state(self.module_name, self.class_name, "_build_kvec_Brillouin")
        
        # Initialize model from state
        model = self._init_from_state(OnePhonon, before_state)
        
        # Ensure A_inv is available directly on the model (not nested in model attribute)
        if not hasattr(model, 'A_inv'):
            if hasattr(model, 'model'):
                if isinstance(model.model, dict) and 'A_inv' in model.model:
                    # Handle case where model is a dictionary with A_inv
                    model.A_inv = model.model['A_inv']
                elif hasattr(model.model, 'A_inv'):
                    # Handle case where model is an object with A_inv
                    model.A_inv = model.model.A_inv
            
            # If still not found, check if A_inv is in the top-level state
            if not hasattr(model, 'A_inv') and 'A_inv' in before_state:
                # Convert to tensor if it's a numpy array
                if isinstance(before_state['A_inv'], np.ndarray):
                    model.A_inv = torch.tensor(before_state['A_inv'], 
                                              dtype=torch.float32, 
                                              device=self.device)
                    model.A_inv.requires_grad_(True)
                else:
                    model.A_inv = before_state['A_inv']
        
        # Patch for models_torch implementation - modify the model structure
        # to match what _build_kvec_Brillouin expects
        if hasattr(model, 'model') and isinstance(model.model, dict):
            if hasattr(model, 'A_inv') and not hasattr(model.model, 'A_inv'):
                # Create a wrapper object to hold A_inv
                class ModelWrapper:
                    pass
                wrapper = ModelWrapper()
                wrapper.A_inv = model.A_inv
                model.model = wrapper
        
        # Call method
        model._build_kvec_Brillouin()
        
        # Verify kvec tensor properties
        self.assertTrue(hasattr(model, 'kvec'), "kvec not created")
        expected_kvec_shape = (model.hsampling[2], model.ksampling[2], model.lsampling[2], 3)
        self._verify_tensor(model.kvec, expected_shape=expected_kvec_shape, requires_grad=True)
        
        # Verify kvec_norm tensor properties
        self.assertTrue(hasattr(model, 'kvec_norm'), "kvec_norm not created")
        expected_norm_shape = (model.hsampling[2], model.ksampling[2], model.lsampling[2], 1)
        self._verify_tensor(model.kvec_norm, expected_shape=expected_norm_shape, requires_grad=True)
        
        # Verify norm calculation correctness (sample a few points)
        for h in range(min(2, model.hsampling[2])):
            for k in range(min(2, model.ksampling[2])):
                for l in range(min(2, model.lsampling[2])):
                    k_vec = model.kvec[h, k, l]
                    k_norm = model.kvec_norm[h, k, l].item()
                    actual_norm = torch.norm(k_vec).item()
                    self.assertAlmostEqual(k_norm, actual_norm, delta=1e-5, 
                                         msg=f"Incorrect norm at [{h},{k},{l}]")
        
        # Load expected after state
        after_state = self._load_state(self.module_name, self.class_name, "_build_kvec_Brillouin", before=False)
        
        # Compare states with appropriate tolerances
        tolerances = {
            'kvec': {'rtol': self.rtol, 'atol': self.atol},
            'kvec_norm': {'rtol': self.rtol, 'atol': self.atol}
        }
        self.assertTrue(
            self._compare_states(after_state, model.__dict__, tolerances),
            "State mismatch after _build_kvec_Brillouin execution"
        )
        
    def test_center_kvec(self):
        """Test the _center_kvec method against NumPy implementation."""
        # Create models for comparison
        self.create_models()
        
        # Extract arguments from logs or use defaults
        args, _ = self._get_method_args(self.module_name, "_center_kvec")
        if not args:
            args = [0, 2]  # Default test case
        
        # Call method on both implementations
        np_result = self.np_model._center_kvec(*args)
        torch_result = self.torch_model._center_kvec(*args)
        
        # Convert torch result to Python scalar if needed
        if isinstance(torch_result, torch.Tensor):
            torch_result = torch_result.item()
        
        # Compare results
        self.assertEqual(np_result, torch_result,
                       f"Different results: NumPy={np_result}, PyTorch={torch_result}")
        
    def test_at_kvec_from_miller_points(self):
        """Test the _at_kvec_from_miller_points method against NumPy implementation."""
        # Create models for comparison
        self.create_models()
        
        # Test different miller points
        test_points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
        
        for point in test_points:
            # Call method on both implementations
            np_indices = self.np_model._at_kvec_from_miller_points(point)
            torch_indices = self.torch_model._at_kvec_from_miller_points(point)
            
            # Convert to NumPy arrays for comparison
            if isinstance(torch_indices, torch.Tensor):
                torch_indices = torch_indices.cpu().numpy()
            if not isinstance(np_indices, np.ndarray):
                np_indices = np.array(np_indices)
            
            # Compare results
            np.testing.assert_array_equal(np_indices, torch_indices,
                                       f"Indices don't match for miller point {point}")
            
    def test_log_completeness(self):
        """Verify k-vector method logs exist and contain required attributes."""
        if not self.verify_logs:
            self.skipTest("Log verification disabled")
            
        # Verify k-vector method logs
        self.verify_required_logs(self.module_name, "_build_kvec_Brillouin", ["kvec", "kvec_norm"])
        self.verify_required_logs(self.module_name, "_center_kvec", [])
        self.verify_required_logs(self.module_name, "_at_kvec_from_miller_points", [])

class TestOnePhononKvector(TestKvectorMethods):
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
    
    def test_gradient_flow(self):
        """Test gradient flow through k-vector operations."""
        # Create a model with parameters that require gradients
        device = torch.device('cpu')
        
        # Create a model with minimal parameters
        from eryx.models_torch import OnePhonon
        
        # Create a model with parameters that require gradients
        model = OnePhonon(
            pdb_path=self.test_params['pdb_path'],
            hsampling=self.test_params['hsampling'],
            ksampling=self.test_params['ksampling'],
            lsampling=self.test_params['lsampling'],
            expand_p1=self.test_params['expand_p1'],
            res_limit=self.test_params['res_limit'],
            gnm_cutoff=self.test_params['gnm_cutoff'],
            gamma_intra=self.test_params['gamma_intra'],
            gamma_inter=self.test_params['gamma_inter'],
            device=device
        )
        
        # Ensure A_inv exists and requires gradients
        if not hasattr(model, 'A_inv'):
            # Try to find A_inv in model.model
            if hasattr(model, 'model'):
                if isinstance(model.model, dict) and 'A_inv' in model.model:
                    model.A_inv = model.model['A_inv']
                elif hasattr(model.model, 'A_inv'):
                    model.A_inv = model.model.A_inv
            
            # If still not found, create a default A_inv
            if not hasattr(model, 'A_inv'):
                # Create a default A_inv (3x3 identity matrix)
                model.A_inv = torch.eye(3, device=device)
        
        # Ensure A_inv is a tensor with requires_grad
        if isinstance(model.A_inv, np.ndarray):
            model.A_inv = torch.tensor(model.A_inv, dtype=torch.float32, device=device)
        
        # Make A_inv require gradients
        model.A_inv.requires_grad_(True)
        
        # Patch for models_torch implementation - modify the model structure
        # to match what _build_kvec_Brillouin expects
        if hasattr(model, 'model') and not hasattr(model.model, 'A_inv'):
            # Create a wrapper object to hold A_inv if needed
            if isinstance(model.model, dict):
                class ModelWrapper:
                    pass
                wrapper = ModelWrapper()
                wrapper.A_inv = model.A_inv
                model.model = wrapper
            else:
                # Add A_inv to existing model object
                model.model.A_inv = model.A_inv
        
        # Build k-vectors
        model._build_kvec_Brillouin()
        
        # Compute a loss based on k-vectors
        loss = torch.sum(torch.abs(model.kvec))
        
        # Check if gradients flow through
        loss.backward()
        
        # Check if A_inv has gradients
        self.assertIsNotNone(
            model.A_inv.grad,
            "Gradients did not flow through k-vector operations"
        )
        
        # Check if gradients are non-zero
        self.assertGreater(
            torch.sum(torch.abs(model.A_inv.grad)),
            0.0,
            "Gradients are all zeros"
        )

if __name__ == '__main__':
    unittest.main()
