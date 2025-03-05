import unittest
import numpy as np
import torch
import os
import tempfile
import shutil
from pathlib import Path
from eryx.models import OnePhonon as NumpyOnePhonon
from eryx.models_torch import OnePhonon as TorchOnePhonon

class TestOnePhononIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment with temporary directory."""
        # Suppress specific Gemmi warnings that don't affect test functionality
        import warnings
        warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*missing entity_type.*")
        
        # Add test parameter for ignoring known warnings
        self.ignore_warnings = True
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.pdb_path = "tests/pdbs/5zck_p1.pdb"
        self.device = torch.device('cpu')  # Use CPU for consistent testing
        
        # Smaller test parameters for faster testing
        self.test_params = {
            'pdb_path': self.pdb_path,
            'hsampling': [-2, 2, 2],  # Smaller grid
            'ksampling': [-2, 2, 2],
            'lsampling': [-2, 2, 2],
            'expand_p1': True,
            'res_limit': 0.0,
            'gnm_cutoff': 4.0,
            'gamma_intra': 1.0,
            'gamma_inter': 1.0
        }
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_match(self):
        """Test that PyTorch implementation matches NumPy end-to-end."""
        # Define file paths in temp directory
        np_file = Path(self.temp_dir) / "np_result.npy"
        torch_file = Path(self.temp_dir) / "torch_result.npy"
        
        if self.ignore_warnings:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*")
                # Run NumPy implementation
                np_model = NumpyOnePhonon(**self.test_params)
                Id_np = np_model.apply_disorder(use_data_adp=True)
                np.save(np_file, Id_np)
                
                # Run PyTorch implementation
                torch_model = TorchOnePhonon(**self.test_params, device=self.device)
                Id_torch = torch_model.apply_disorder(use_data_adp=True)
                np.save(torch_file, Id_torch.detach().cpu().numpy())
        else:
            # Run NumPy implementation
            np_model = NumpyOnePhonon(**self.test_params)
            Id_np = np_model.apply_disorder(use_data_adp=True)
            np.save(np_file, Id_np)
            
            # Run PyTorch implementation
            torch_model = TorchOnePhonon(**self.test_params, device=self.device)
            Id_torch = torch_model.apply_disorder(use_data_adp=True)
            np.save(torch_file, Id_torch.detach().cpu().numpy())
        
        # Load and compare results
        np_result = np.load(np_file)
        torch_result = np.load(torch_file)
        
        # Get non-NaN mask
        mask = ~np.isnan(np_result) & ~np.isnan(torch_result)
        self.assertTrue(np.any(mask), "All values are NaN")
        
        # Calculate metrics
        mse = np.mean((np_result[mask] - torch_result[mask])**2)
        correlation = np.corrcoef(np_result[mask], torch_result[mask])[0, 1]
        max_diff = np.max(np.abs(np_result[mask] - torch_result[mask]))
        
        # Verify results are close enough
        self.assertLess(mse, 1e-10, f"MSE too high: {mse}")
        self.assertGreater(correlation, 0.99, f"Correlation too low: {correlation}")
        self.assertLess(max_diff, 1e-5, f"Max difference too high: {max_diff}")
    
    def test_parameter_gradients(self):
        """Test gradient flow through model parameters."""
        if self.ignore_warnings:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*")
                # Create model with parameters that require gradients
                gamma_intra = torch.tensor(1.0, requires_grad=True)
                gamma_inter = torch.tensor(1.0, requires_grad=True)
                
                # Create OnePhonon model with these parameters
                torch_model = TorchOnePhonon(
                    pdb_path=self.pdb_path,
                    hsampling=self.test_params['hsampling'],
                    ksampling=self.test_params['ksampling'],
                    lsampling=self.test_params['lsampling'],
                    expand_p1=self.test_params['expand_p1'],
                    res_limit=self.test_params['res_limit'],
                    gnm_cutoff=self.test_params['gnm_cutoff'],
                    gamma_intra=gamma_intra,
                    gamma_inter=gamma_inter,
                    device=self.device
                )
        else:
            # Create model with parameters that require gradients
            gamma_intra = torch.tensor(1.0, requires_grad=True)
            gamma_inter = torch.tensor(1.0, requires_grad=True)
            
            # Create OnePhonon model with these parameters
            torch_model = TorchOnePhonon(
                pdb_path=self.pdb_path,
                hsampling=self.test_params['hsampling'],
                ksampling=self.test_params['ksampling'],
                lsampling=self.test_params['lsampling'],
                expand_p1=self.test_params['expand_p1'],
                res_limit=self.test_params['res_limit'],
                gnm_cutoff=self.test_params['gnm_cutoff'],
                gamma_intra=gamma_intra,
                gamma_inter=gamma_inter,
                device=self.device
            )
        
        # Forward pass
        Id_torch = torch_model.apply_disorder(use_data_adp=True)
        
        # Create a simple loss function
        loss = torch.nansum(Id_torch)
        
        # Backward pass
        loss.backward()
        
        # Verify gradients exist and are non-zero
        self.assertIsNotNone(gamma_intra.grad, "No gradient for gamma_intra")
        self.assertIsNotNone(gamma_inter.grad, "No gradient for gamma_inter")
        
        self.assertFalse(torch.allclose(gamma_intra.grad, torch.zeros_like(gamma_intra.grad)),
                        "Gradient for gamma_intra is zero")
        self.assertFalse(torch.allclose(gamma_inter.grad, torch.zeros_like(gamma_inter.grad)),
                        "Gradient for gamma_inter is zero")
    
    def test_device_compatibility(self):
        """Test model works on different devices if available."""
        if self.ignore_warnings:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="remove_ligands_and_waters.*")
                # Test on CPU
                try:
                    model_cpu = TorchOnePhonon(**self.test_params, device=torch.device('cpu'))
                    result_cpu = model_cpu.apply_disorder(use_data_adp=True)
                    self.assertIsNotNone(result_cpu, "CPU execution failed")
                    
                    # Test on CUDA if available
                    if torch.cuda.is_available():
                        model_cuda = TorchOnePhonon(**self.test_params, device=torch.device('cuda'))
                        result_cuda = model_cuda.apply_disorder(use_data_adp=True)
                        self.assertIsNotNone(result_cuda, "CUDA execution failed")
                        
                        # Verify results match between devices
                        cpu_array = result_cpu.detach().cpu().numpy()
                        cuda_array = result_cuda.detach().cpu().numpy()
                        
                        mask = ~np.isnan(cpu_array) & ~np.isnan(cuda_array)
                        self.assertTrue(np.any(mask), "All values are NaN")
                        
                        correlation = np.corrcoef(cpu_array[mask], cuda_array[mask])[0, 1]
                        self.assertGreater(correlation, 0.99, f"CPU/CUDA correlation too low: {correlation}")
                except RuntimeError as e:
                    if "CUDA" in str(e) and not torch.cuda.is_available():
                        self.skipTest("CUDA test skipped - not available")
                    else:
                        raise
        else:
            # Test on CPU
            try:
                model_cpu = TorchOnePhonon(**self.test_params, device=torch.device('cpu'))
                result_cpu = model_cpu.apply_disorder(use_data_adp=True)
                self.assertIsNotNone(result_cpu, "CPU execution failed")
                
                # Test on CUDA if available
                if torch.cuda.is_available():
                    model_cuda = TorchOnePhonon(**self.test_params, device=torch.device('cuda'))
                    result_cuda = model_cuda.apply_disorder(use_data_adp=True)
                    self.assertIsNotNone(result_cuda, "CUDA execution failed")
                
                # Verify results match between devices
                cpu_array = result_cpu.detach().cpu().numpy()
                cuda_array = result_cuda.detach().cpu().numpy()
                
                mask = ~np.isnan(cpu_array) & ~np.isnan(cuda_array)
                self.assertTrue(np.any(mask), "All values are NaN")
                
                correlation = np.corrcoef(cpu_array[mask], cuda_array[mask])[0, 1]
                self.assertGreater(correlation, 0.99, f"CPU/CUDA correlation too low: {correlation}")
                
        except RuntimeError as e:
            if "CUDA" in str(e) and not torch.cuda.is_available():
                self.skipTest("CUDA test skipped - not available")
            else:
                raise

if __name__ == '__main__':
    unittest.main()
