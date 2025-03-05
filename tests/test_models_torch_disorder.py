import os
import numpy as np
import torch
import pytest
from eryx.models import OnePhonon as NumpyOnePhonon
from eryx.models_torch import OnePhonon as TorchOnePhonon

class TestOnePhononDisorder:
    """Test class for OnePhonon disorder model implementations."""
    
    @pytest.fixture
    def suppress_warnings(self):
        """Fixture to suppress irrelevant warnings."""
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*")
            yield
    
    @pytest.fixture
    def test_models(self, suppress_warnings):
        """Fixture to create test models with minimal parameters."""
        # Use very small grid for speed
        params = {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-1, 1, 1],
            'ksampling': [-1, 1, 1],
            'lsampling': [-1, 1, 1],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
        
        # Create models
        np_model = NumpyOnePhonon(**params)
        torch_model = TorchOnePhonon(**params, device=torch.device('cpu'))
        
        return np_model, torch_model
    
    def test_apply_disorder_direct_comparison(self, test_models):
        """Test direct comparison of apply_disorder between NumPy and PyTorch implementations."""
        np_model, torch_model = test_models
        
        # Call apply_disorder on both models
        Id_np = np_model.apply_disorder(use_data_adp=True)
        Id_torch = torch_model.apply_disorder(use_data_adp=True)
        
        # Convert PyTorch result to NumPy
        Id_torch_np = Id_torch.detach().cpu().numpy()
        
        # Create mask for non-NaN values
        mask = ~np.isnan(Id_np) & ~np.isnan(Id_torch_np)
        assert np.any(mask), "All values are NaN"
        
        # Calculate comparison metrics
        np_vals = Id_np[mask]
        torch_vals = Id_torch_np[mask]
        
        # Calculate basic statistics
        print(f"NumPy min/max: {np.min(np_vals)}/{np.max(np_vals)}")
        print(f"PyTorch min/max: {np.min(torch_vals)}/{np.max(torch_vals)}")
        
        # Now that we've fixed the scaling issue, we can directly compare the results
        # Calculate correlation
        correlation = np.corrcoef(np_vals, torch_vals)[0, 1]
        print(f"Correlation: {correlation}")
        
        # Check results match well
        assert correlation > 0.99, f"Correlation too low: {correlation}"
        
        # Calculate relative error
        max_diff = np.max(np.abs(np_vals - torch_vals))
        max_magnitude = max(np.max(np.abs(np_vals)), np.max(np.abs(torch_vals)))
        relative_max_diff = max_diff / max_magnitude
        
        # Assert results are close enough
        assert relative_max_diff < 0.01, f"Relative max difference too high: {relative_max_diff}"
        np.testing.assert_allclose(
            np_vals, 
            torch_vals, 
            rtol=1e-4, 
            atol=1e-6, 
            err_msg="NumPy and PyTorch results don't match within tolerances"
        )
    
    def test_apply_disorder_gradient_flow(self, suppress_warnings):
        """Test that gradients flow properly through the apply_disorder function."""
        # Create tensors with requires_grad=True for key parameters
        gamma_intra = torch.tensor(1.0, requires_grad=True)
        gamma_inter = torch.tensor(1.0, requires_grad=True)
        
        # Minimal parameters for speed
        params = {
            'pdb_path': 'tests/pdbs/5zck_p1.pdb',
            'hsampling': [-1, 1, 1],
            'ksampling': [-1, 1, 1],
            'lsampling': [-1, 1, 1],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': gamma_intra,
            'gamma_inter': gamma_inter,
            'device': torch.device('cpu')
        }
        
        # Create PyTorch model
        torch_model = TorchOnePhonon(**params)
        
        # Run apply_disorder
        Id_torch = torch_model.apply_disorder(use_data_adp=True)
        
        # Create a simple loss that will use the output
        Id_no_nan = torch.where(torch.isnan(Id_torch), torch.tensor(0.0), Id_torch)
        loss = torch.sum(Id_no_nan)
        
        # Backpropagate
        loss.backward()
        
        # Verify gradients exist and are non-zero
        assert gamma_intra.grad is not None, "No gradient for gamma_intra"
        assert gamma_inter.grad is not None, "No gradient for gamma_inter"
        
        # Verify gradients have non-zero values
        assert gamma_intra.grad.abs().sum().item() > 0, "Zero gradient for gamma_intra"
        assert gamma_inter.grad.abs().sum().item() > 0, "Zero gradient for gamma_inter"

if __name__ == "__main__":
    # Allow the test to be run directly
    pytest.main(["-xvs", __file__])
