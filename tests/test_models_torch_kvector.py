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
        """Test _build_kvec_Brillouin using state-based approach."""
        # Import test helpers
        from eryx.autotest.test_helpers import (
            load_test_state, 
            build_test_object,
            verify_gradient_flow
        )
        
        # 1. Load before state
        before_state = load_test_state(
            self.logger, 
            self.module_name, 
            self.class_name, 
            "_build_kvec_Brillouin"
        )
        
        # 2. Build model with StateBuilder
        model = build_test_object(OnePhonon, before_state, device=self.device)
        
        # 3. Verify initial structure
        self.assertTrue(
            hasattr(model, 'model') and hasattr(model.model, 'A_inv'),
            "Atomic model does not contain A_inv"
        )
        
        # 4. Call the method
        model._build_kvec_Brillouin()
        
        # 5. Verify results
        # Check kvec tensor
        self.assertTrue(hasattr(model, 'kvec'), "kvec not created")
        expected_kvec_shape = (model.hsampling[2], model.ksampling[2], model.lsampling[2], 3)
        self.assertEqual(model.kvec.shape, expected_kvec_shape)
        self.assertTrue(model.kvec.requires_grad, "kvec should require gradients")
        
        # Check kvec_norm tensor
        self.assertTrue(hasattr(model, 'kvec_norm'), "kvec_norm not created")
        expected_norm_shape = (model.hsampling[2], model.ksampling[2], model.lsampling[2], 1)
        self.assertEqual(model.kvec_norm.shape, expected_norm_shape)
        self.assertTrue(model.kvec_norm.requires_grad, "kvec_norm should require gradients")
        
        # 6. Verify gradient flow
        loss = torch.sum(torch.abs(model.kvec))
        loss.backward()
        self.assertIsNotNone(model.model.A_inv.grad, "No gradients on A_inv")
        self.assertGreater(
            torch.sum(torch.abs(model.model.A_inv.grad)).item(), 
            0.0, 
            "Gradients are all zeros"
        )
        
        # 7. Load after state and compare
        after_state = load_test_state(
            self.logger, 
            self.module_name, 
            self.class_name, 
            "_build_kvec_Brillouin", 
            before=False
        )
        
        # 8. Compare only the tensors that should have changed
        kvec_expected = after_state.get('kvec')
        kvec_norm_expected = after_state.get('kvec_norm')
        
        # Handle serialized values
        if isinstance(kvec_expected, bytes):
            kvec_expected = self.serializer.deserialize(kvec_expected)
        if isinstance(kvec_norm_expected, bytes):
            kvec_norm_expected = self.serializer.deserialize(kvec_norm_expected)
        
        # Compare tensor values
        tolerances = {'rtol': 1e-5, 'atol': 1e-8}
        self.assertTrue(
            np.allclose(
                model.kvec.detach().cpu().numpy(), 
                kvec_expected, 
                rtol=tolerances['rtol'], 
                atol=tolerances['atol']
            ),
            "kvec values don't match expected"
        )
        self.assertTrue(
            np.allclose(
                model.kvec_norm.detach().cpu().numpy(), 
                kvec_norm_expected, 
                rtol=tolerances['rtol'], 
                atol=tolerances['atol']
            ),
            "kvec_norm values don't match expected"
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
        """Test gradient flow through k-vector operations using StateBuilder."""
        # Import test helpers
        from eryx.autotest.test_helpers import build_test_object
        
        # 1. Create minimal state data
        minimal_state = {
            'pdb_path': self.test_params['pdb_path'],
            'hsampling': self.test_params['hsampling'],
            'ksampling': self.test_params['ksampling'],
            'lsampling': self.test_params['lsampling'],
            'model': {
                'A_inv': np.eye(3, dtype=np.float32)
            }
        }
        
        # 2. Build model with StateBuilder 
        model = build_test_object(OnePhonon, minimal_state, device=self.device)
        
        # 3. Build k-vectors
        model._build_kvec_Brillouin()
        
        # 4. Compute loss and verify gradient flow
        loss = torch.sum(torch.abs(model.kvec))
        loss.backward()
        
        # 5. Check if A_inv has gradients
        self.assertIsNotNone(
            model.model.A_inv.grad,
            "Gradients did not flow through k-vector operations"
        )
        
        # 6. Check if gradients are non-zero
        self.assertGreater(
            torch.sum(torch.abs(model.model.A_inv.grad)).item(),
            0.0,
            "Gradients are all zeros"
        )

if __name__ == '__main__':
    unittest.main()
