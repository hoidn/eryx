import unittest
import torch
import numpy as np
from typing import Optional, Tuple

# Import the OnePhonon class
from eryx.models_torch import OnePhonon

class TestBase(unittest.TestCase):
    """Base class for tests with common setup."""
    
    def setUp(self):
        """Set up test environment."""
        # Use CPU for testing
        self.device = torch.device('cpu')
        
        # Create a simple mock PDB path for testing
        # In real tests, this would be a path to a test PDB file
        self.mock_pdb_path = "tests/data/test.pdb"
        
        # Define standard sampling parameters for tests
        self.hsampling = (0, 1, 2)
        self.ksampling = (0, 1, 2)
        self.lsampling = (0, 1, 2)

class TestArbitraryQVectors(TestBase):
    """Test cases for arbitrary q-vectors support in OnePhonon."""
    
    def test_constructor_with_q_vectors(self):
        """Test that the constructor accepts q_vectors parameter correctly."""
        # Create a simple q_vectors tensor
        q_vectors = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], device=self.device)
        
        # Mock the _setup and _setup_phonons methods to avoid actual initialization
        # This allows us to test just the constructor logic
        original_setup = OnePhonon._setup
        original_setup_phonons = OnePhonon._setup_phonons
        
        try:
            # Replace methods with mock versions
            OnePhonon._setup = lambda self, *args, **kwargs: None
            OnePhonon._setup_phonons = lambda self, *args, **kwargs: None
            
            # Create OnePhonon instance with q_vectors
            model = OnePhonon(
                pdb_path=self.mock_pdb_path,
                q_vectors=q_vectors,
                device=self.device
            )
            
            # Verify use_arbitrary_q flag is set properly
            self.assertTrue(model.use_arbitrary_q)
            
            # Check q_vectors tensor is placed on correct device
            self.assertEqual(model.q_vectors.device, self.device)
            
            # Ensure q_vectors tensor has requires_grad=True
            self.assertTrue(model.q_vectors.requires_grad)
            
            # Verify tensor dimensions are preserved
            self.assertEqual(model.q_vectors.shape, (2, 3))
            
            # Check that sampling parameters are set to placeholder values
            self.assertEqual(model.hsampling, (0, 0, 1))
            self.assertEqual(model.ksampling, (0, 0, 1))
            self.assertEqual(model.lsampling, (0, 0, 1))
            
        finally:
            # Restore original methods
            OnePhonon._setup = original_setup
            OnePhonon._setup_phonons = original_setup_phonons
    
    def test_constructor_validation(self):
        """Test input validation in the constructor."""
        # Mock the _setup and _setup_phonons methods
        original_setup = OnePhonon._setup
        original_setup_phonons = OnePhonon._setup_phonons
        
        try:
            # Replace methods with mock versions
            OnePhonon._setup = lambda self, *args, **kwargs: None
            OnePhonon._setup_phonons = lambda self, *args, **kwargs: None
            
            # Test 1: Constructor should raise ValueError when neither q_vectors nor sampling parameters are provided
            with self.assertRaises(ValueError):
                OnePhonon(pdb_path=self.mock_pdb_path)
            
            # Test 2: Constructor should raise ValueError when only some sampling parameters are provided
            with self.assertRaises(ValueError):
                OnePhonon(
                    pdb_path=self.mock_pdb_path,
                    hsampling=self.hsampling,
                    ksampling=self.ksampling
                )
            
            # Test 3: Verify q_vectors shape validation works correctly
            with self.assertRaises(ValueError):
                # Wrong shape - should be [n_points, 3]
                q_vectors = torch.tensor([0.1, 0.2, 0.3])
                OnePhonon(
                    pdb_path=self.mock_pdb_path,
                    q_vectors=q_vectors
                )
            
            # Test 4: Test backward compatibility with original sampling parameter approach
            model = OnePhonon(
                pdb_path=self.mock_pdb_path,
                hsampling=self.hsampling,
                ksampling=self.ksampling,
                lsampling=self.lsampling,
                device=self.device
            )
            
            # Verify use_arbitrary_q flag is False
            self.assertFalse(model.use_arbitrary_q)
            
            # Check that sampling parameters are set correctly
            self.assertEqual(model.hsampling, self.hsampling)
            self.assertEqual(model.ksampling, self.ksampling)
            self.assertEqual(model.lsampling, self.lsampling)
            
            # q_vectors should be None
            self.assertIsNone(model.q_vectors)
            
        finally:
            # Restore original methods
            OnePhonon._setup = original_setup
            OnePhonon._setup_phonons = original_setup_phonons

if __name__ == '__main__':
    unittest.main()
