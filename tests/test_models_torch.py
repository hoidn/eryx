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
        # Load before state
        before_state = self._load_state(self.module_name, self.class_name, "_build_A")
        
        # Initialize model from state
        model = self._init_from_state(OnePhonon, before_state)
        
        # Call method
        model._build_A()
        
        # Verify Amat tensor properties
        self.assertTrue(hasattr(model, 'Amat'), "Amat not created")
        expected_shape = (model.n_asu, model.n_dof_per_asu_actual, model.n_dof_per_asu)
        self._verify_tensor(model.Amat, expected_shape=expected_shape,
                          requires_grad=True, check_gradient=True)
        
        # Load expected after state
        after_state = self._load_state(self.module_name, self.class_name, "_build_A", before=False)
        
        # Compare states with appropriate tolerances
        tolerances = {'Amat': {'rtol': self.rtol, 'atol': self.atol}}
        self.assertTrue(
            self._compare_states(after_state, model.__dict__, tolerances),
            "State mismatch after _build_A execution"
        )
        
    def test_build_M_state_based(self):
        # Load before state
        before_state = self._load_state(self.module_name, self.class_name, "_build_M")
        
        # Initialize model from state
        model = self._init_from_state(OnePhonon, before_state)
        
        # Call method
        model._build_M()
        
        # Verify Linv tensor properties
        self.assertTrue(hasattr(model, 'Linv'), "Linv not created")
        expected_shape = (model.n_asu * model.n_dof_per_asu, model.n_asu * model.n_dof_per_asu)
        self._verify_tensor(model.Linv, expected_shape=expected_shape,
                          requires_grad=True, check_gradient=True)
        
        # Load expected after state
        after_state = self._load_state(self.module_name, self.class_name, "_build_M", before=False)
        
        # Compare states with appropriate tolerances
        tolerances = {'Linv': {'rtol': self.rtol, 'atol': self.atol}}
        self.assertTrue(
            self._compare_states(after_state, model.__dict__, tolerances),
            "State mismatch after _build_M execution"
        )
        
    def test_build_M_allatoms_state_based(self):
        # Load before state
        before_state = self._load_state(self.module_name, self.class_name, "_build_M_allatoms")
        
        # Initialize model from state
        model = self._init_from_state(OnePhonon, before_state)
        
        # Call method
        result = model._build_M_allatoms()
        
        # Verify basic properties of result
        expected_shape = (model.n_asu, model.n_dof_per_asu_actual,
                         model.n_asu, model.n_dof_per_asu_actual)
        self._verify_tensor(result, expected_shape=expected_shape, requires_grad=True)
        self.assertTrue(torch.all(result >= 0), "Mass matrix should be non-negative")
        
        # Load expected after state
        after_state = self._load_state(self.module_name, self.class_name, "_build_M_allatoms", before=False)
        
        # Compare model state (not the return value)
        self.assertTrue(
            self._compare_states(after_state, model.__dict__),
            "State mismatch after _build_M_allatoms execution"
        )
        
    def test_project_M_state_based(self):
        # Load before state
        before_state = self._load_state(self.module_name, self.class_name, "_project_M")
        
        # Extract arguments
        args, kwargs = self._get_method_args(self.module_name, "_project_M")
        
        # Initialize model from state
        model = self._init_from_state(OnePhonon, before_state)
        
        # Call method with arguments
        result = model._project_M(*args)
        
        # Verify basic properties of result
        expected_shape = (model.n_asu, model.n_dof_per_asu,
                         model.n_asu, model.n_dof_per_asu)
        self._verify_tensor(result, expected_shape=expected_shape, requires_grad=True)
        
        # Load expected after state
        after_state = self._load_state(self.module_name, self.class_name, "_project_M", before=False)
        
        # Compare model state
        self.assertTrue(
            self._compare_states(after_state, model.__dict__),
            "State mismatch after _project_M execution"
        )
        
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
