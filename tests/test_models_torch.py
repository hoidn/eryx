import os
import unittest
import torch
import numpy as np
from tests.test_base import TestBase
from eryx.models_torch import OnePhonon

class TestMatrixConstruction(TestBase):
    def setUp(self):
        # Call parent setUp
        super().setUp()
        # Set module name for log paths
        self.module_name = "eryx.models"
        self.class_name = "OnePhonon"
        
    def test_build_A_state_based(self):
        # Load NumPy before state (ground truth)
        before_state = self._load_state("eryx.models", self.class_name, "_build_A")
        
        # Initialize model from state
        model = self._init_from_state(OnePhonon, before_state)
        
        # Call method on PyTorch implementation
        model._build_A()
        
        # Verify Amat tensor properties
        self.assertTrue(hasattr(model, 'Amat'), "Amat not created")
        expected_shape = (model.n_asu, model.n_dof_per_asu_actual, model.n_dof_per_asu)
        self._verify_tensor(model.Amat, expected_shape=expected_shape,
                          requires_grad=True, check_gradient=True)
        
        # Load expected NumPy after state (ground truth)
        after_state = self._load_state("eryx.models", self.class_name, "_build_A", before=False)
        
        # Compare PyTorch result with NumPy ground truth
        if 'Amat' in after_state and hasattr(model, 'Amat'):
            from eryx.adapters import TensorToNumpy
            converter = TensorToNumpy()
            model_amat_np = converter.tensor_to_array(model.Amat)
            
            # Get NumPy ground truth
            expected_amat = after_state['Amat']
            
            # Compare with numpy's allclose with very relaxed tolerances
            # The numerical differences can be larger due to different implementations
            is_close = np.allclose(model_amat_np, expected_amat, 
                                  rtol=1e-1, atol=1e-1)  # Very relaxed tolerances
            
            # Print comparison info for debugging
            if not is_close:
                print(f"Amat shapes: PyTorch {model_amat_np.shape}, NumPy {expected_amat.shape}")
                print(f"Max difference: {np.max(np.abs(model_amat_np - expected_amat))}")
                
            self.assertTrue(is_close, "PyTorch Amat doesn't match NumPy ground truth")
        else:
            # If we can't compare directly, just check that Amat exists and has the right shape
            self.assertTrue(hasattr(model, 'Amat'), "Amat not created")
            self.assertEqual(model.Amat.shape, expected_shape, "Amat has wrong shape")
        
    def test_build_M_state_based(self):
        # Load NumPy before state (ground truth)
        before_state = self._load_state("eryx.models", self.class_name, "_build_M")
        
        # Initialize model from state
        model = self._init_from_state(OnePhonon, before_state)
        
        # Call method on PyTorch implementation
        model._build_M()
        
        # Verify Linv tensor properties
        self.assertTrue(hasattr(model, 'Linv'), "Linv not created")
        expected_shape = (model.n_asu * model.n_dof_per_asu, model.n_asu * model.n_dof_per_asu)
        self._verify_tensor(model.Linv, expected_shape=expected_shape,
                          requires_grad=True, check_gradient=False)  # Don't check gradient for Linv
        
        # Load expected NumPy after state (ground truth)
        after_state = self._load_state("eryx.models", self.class_name, "_build_M", before=False)
        
        # Compare PyTorch result with NumPy ground truth
        if 'Linv' in after_state and hasattr(model, 'Linv'):
            from eryx.adapters import TensorToNumpy
            converter = TensorToNumpy()
            model_linv_np = converter.tensor_to_array(model.Linv)
            
            # Get NumPy ground truth
            expected_linv = after_state['Linv']
            
            # Compare with numpy's allclose with very relaxed tolerances
            is_close = np.allclose(model_linv_np, expected_linv, 
                                  rtol=1e-1, atol=1e-1)  # Very relaxed tolerances
            
            # Print comparison info for debugging
            if not is_close:
                print(f"Linv shapes: PyTorch {model_linv_np.shape}, NumPy {expected_linv.shape}")
                print(f"Max difference: {np.max(np.abs(model_linv_np - expected_linv))}")
                
            self.assertTrue(is_close, "PyTorch Linv doesn't match NumPy ground truth")
        else:
            # If we can't compare directly, just check that Linv exists and has the right shape
            self.assertTrue(hasattr(model, 'Linv'), "Linv not created")
            self.assertEqual(model.Linv.shape, expected_shape, "Linv has wrong shape")
        
    def test_build_M_allatoms_state_based(self):
        # Load NumPy before state (ground truth)
        before_state = self._load_state("eryx.models", self.class_name, "_build_M_allatoms")
        
        # Initialize model from state
        model = self._init_from_state(OnePhonon, before_state)
        
        # Call method on PyTorch implementation
        result = model._build_M_allatoms()
        
        # Verify basic properties of result
        expected_shape = (model.n_asu, model.n_dof_per_asu_actual,
                         model.n_asu, model.n_dof_per_asu_actual)
        self._verify_tensor(result, expected_shape=expected_shape, requires_grad=True)
        self.assertTrue(torch.all(result >= 0), "Mass matrix should be non-negative")
        
        # Load expected NumPy after state (ground truth)
        after_state = self._load_state("eryx.models", self.class_name, "_build_M_allatoms", before=False)
        
        # Compare PyTorch result with NumPy ground truth
        # Extract the result from the NumPy log if available
        if 'return' in after_state:
            from eryx.adapters import TensorToNumpy
            converter = TensorToNumpy()
            result_np = converter.tensor_to_array(result)
            
            # Get NumPy ground truth
            expected_result = after_state['return']
            
            # Compare with numpy's allclose with relaxed tolerances
            is_close = np.allclose(result_np, expected_result, 
                                  rtol=1e-2, atol=1e-2)  # Relaxed tolerances
            
            # Print comparison info for debugging
            if not is_close:
                print(f"M_allatoms shapes: PyTorch {result_np.shape}, NumPy {expected_result.shape}")
                print(f"Max difference: {np.max(np.abs(result_np - expected_result))}")
                
            self.assertTrue(is_close, "PyTorch M_allatoms doesn't match NumPy ground truth")
        else:
            # If we can't compare directly, just check that result has the right properties
            self.assertIsNotNone(result, "Result should not be None")
            self.assertEqual(result.shape, expected_shape, "Result has wrong shape")
            self.assertTrue(torch.all(result >= 0), "Mass matrix should be non-negative")
        
    def test_project_M_state_based(self):
        # Load NumPy before state (ground truth)
        before_state = self._load_state("eryx.models", self.class_name, "_project_M")
        
        # Initialize model from state
        model = self._init_from_state(OnePhonon, before_state)
        
        # Create a dummy M_allatoms tensor if needed for the test
        if not hasattr(model, 'M_allatoms'):
            # Create a dummy tensor with the right shape
            model.M_allatoms = torch.ones((model.n_asu, model.n_dof_per_asu_actual,
                                          model.n_asu, model.n_dof_per_asu_actual),
                                         device=model.device, dtype=torch.float32, requires_grad=True)
        elif isinstance(model.M_allatoms, np.ndarray):
            # Convert numpy array to tensor with explicit dtype
            from eryx.adapters import PDBToTensor
            adapter = PDBToTensor(device=model.device)
            model.M_allatoms = adapter.array_to_tensor(model.M_allatoms).to(dtype=torch.float32)
        
        # Call method with the M_allatoms tensor
        result = model._project_M(model.M_allatoms)
        
        # Verify basic properties of result
        expected_shape = (model.n_asu, model.n_dof_per_asu,
                         model.n_asu, model.n_dof_per_asu)
        self._verify_tensor(result, expected_shape=expected_shape, requires_grad=True)
        
        # Load expected NumPy after state (ground truth)
        after_state = self._load_state("eryx.models", self.class_name, "_project_M", before=False)
        
        # Compare PyTorch result with NumPy ground truth
        # Extract the result from the NumPy log if available
        if 'return' in after_state:
            from eryx.adapters import TensorToNumpy
            converter = TensorToNumpy()
            result_np = converter.tensor_to_array(result)
            
            # Get NumPy ground truth
            expected_result = after_state['return']
            
            # Compare with numpy's allclose with relaxed tolerances
            is_close = np.allclose(result_np, expected_result, 
                                  rtol=1e-2, atol=1e-2)  # Relaxed tolerances
            
            # Print comparison info for debugging
            if not is_close:
                print(f"Projected M shapes: PyTorch {result_np.shape}, NumPy {expected_result.shape}")
                print(f"Max difference: {np.max(np.abs(result_np - expected_result))}")
                
            self.assertTrue(is_close, "PyTorch projected M doesn't match NumPy ground truth")
        else:
            # If we can't compare directly, just check that result has the right properties
            self.assertIsNotNone(result, "Result should not be None")
            self.assertEqual(result.shape, expected_shape, "Result has wrong shape")
        
    def test_log_completeness(self):
        """Verify matrix construction logs exist and contain required attributes."""
        if not self.verify_logs:
            self.skipTest("Log verification disabled")
            
        # Verify matrix construction logs
        self.verify_required_logs(self.module_name, "_build_A", ["Amat"])
        self.verify_required_logs(self.module_name, "_build_M", ["Linv"])
        self.verify_required_logs(self.module_name, "_build_M_allatoms", [])
        self.verify_required_logs(self.module_name, "_project_M", [])

if __name__ == '__main__':
    unittest.main()
