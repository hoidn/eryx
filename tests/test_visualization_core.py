"""
Integration tests for visualization core components.

Tests the IntensityDataHandler, CoordinateMapper, and NaNProcessor classes.
"""

import unittest
import numpy as np
import tempfile
import os
from pathlib import Path

# Import the visualization components
from eryx.visualization import IntensityDataHandler, CoordinateMapper, NaNProcessor


class TestIntensityDataHandler(unittest.TestCase):
    """Test the IntensityDataHandler class."""
    
    def setUp(self):
        """Create test data files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.intensity_3d = np.random.rand(10, 12, 14)
        self.intensity_1d = np.random.rand(10 * 12 * 14)
        self.q_vectors = np.random.rand(10 * 12 * 14, 3)
        self.map_shape = (10, 12, 14)
        
        # Add some NaN values
        self.intensity_3d[0, 0, 0] = np.nan
        self.intensity_3d[5, 6, 7] = np.nan
        self.intensity_1d[0] = np.nan
        self.intensity_1d[100] = np.nan
        
        # Save test NPZ file
        self.npz_path = os.path.join(self.temp_dir, "test_data.npz")
        np.savez_compressed(
            self.npz_path,
            intensity=self.intensity_3d,
            q_vectors=self.q_vectors,
            map_shape=self.map_shape
        )
        
        # Save test NPY file (1D only)
        self.npy_path = os.path.join(self.temp_dir, "test_data.npy")
        np.save(self.npy_path, self.intensity_1d)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_from_npz(self):
        """Test loading data from NPZ file."""
        handler = IntensityDataHandler(self.npz_path)
        q_vectors, intensity, map_shape = handler.load_data()
        
        self.assertIsNotNone(intensity)
        self.assertIsNotNone(q_vectors)
        self.assertIsNotNone(map_shape)
        
        self.assertEqual(intensity.shape, (10, 12, 14))
        self.assertEqual(q_vectors.shape, (10 * 12 * 14, 3))
        self.assertEqual(map_shape, (10, 12, 14))
        
        # Check NaN preservation
        self.assertTrue(np.isnan(intensity[0, 0, 0]))
        self.assertTrue(np.isnan(intensity[5, 6, 7]))
    
    def test_load_from_npy(self):
        """Test loading data from NPY file."""
        handler = IntensityDataHandler(self.npy_path)
        q_vectors, intensity, map_shape = handler.load_data()
        
        self.assertIsNotNone(intensity)
        self.assertIsNone(q_vectors)  # NPY doesn't have q_vectors
        self.assertIsNone(map_shape)  # NPY doesn't have map_shape
        
        self.assertEqual(intensity.shape, (10 * 12 * 14,))
        self.assertTrue(np.isnan(intensity[0]))
        self.assertTrue(np.isnan(intensity[100]))
    
    def test_load_from_array(self):
        """Test initializing with numpy array directly."""
        handler = IntensityDataHandler(self.intensity_3d)
        q_vectors, intensity, map_shape = handler.load_data()
        
        self.assertIsNotNone(intensity)
        self.assertIsNone(q_vectors)
        self.assertEqual(map_shape, (10, 12, 14))
        np.testing.assert_array_equal(intensity, self.intensity_3d)
    
    def test_reshape_to_3d(self):
        """Test reshaping 1D data to 3D."""
        # Create handler with 1D data
        handler = IntensityDataHandler(self.intensity_1d)
        handler.map_shape = (10, 12, 14)
        
        # Test reshaping
        success = handler.reshape_to_3d()
        self.assertTrue(success)
        self.assertEqual(handler.intensity.shape, (10, 12, 14))
    
    def test_subsample(self):
        """Test data subsampling."""
        handler = IntensityDataHandler(self.intensity_3d)
        handler.load_data()
        
        # Test factor=2 subsampling
        subsampled = handler.subsample(factor=2)
        self.assertEqual(subsampled.shape, (5, 6, 7))
        
        # Test factor=1 (no subsampling)
        subsampled = handler.subsample(factor=1)
        self.assertEqual(subsampled.shape, (10, 12, 14))
    
    def test_get_statistics(self):
        """Test statistics calculation."""
        handler = IntensityDataHandler(self.intensity_3d)
        handler.load_data()
        
        stats = handler.get_statistics()
        
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('mean', stats)
        self.assertIn('std', stats)
        self.assertIn('n_valid', stats)
        self.assertIn('n_nan', stats)
        
        # Check NaN counting
        self.assertEqual(stats['n_nan'], 2)
        self.assertEqual(stats['n_valid'], 10 * 12 * 14 - 2)


class TestCoordinateMapper(unittest.TestCase):
    """Test the CoordinateMapper class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a simple orthogonalization matrix (cubic cell for simplicity)
        self.A_inv = np.array([
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1]
        ])
        
        self.sampling = (
            (-2, 2, 3),  # h: -2 to 2, oversampling 3
            (-3, 3, 2),  # k: -3 to 3, oversampling 2
            (-1, 1, 4),  # l: -1 to 1, oversampling 4
        )
    
    def test_initialization(self):
        """Test CoordinateMapper initialization."""
        mapper = CoordinateMapper(self.A_inv, sampling=self.sampling)
        
        self.assertIsNotNone(mapper.A_inv)
        self.assertIsNotNone(mapper.sampling)
        np.testing.assert_array_equal(mapper.A_inv, self.A_inv)
    
    def test_hkl_to_cartesian(self):
        """Test Miller indices to Cartesian conversion."""
        mapper = CoordinateMapper(self.A_inv)
        
        # Test single vector
        hkl = np.array([1, 0, 0])
        q = mapper.hkl_to_cartesian(hkl)
        expected_q = 2 * np.pi * np.array([0.1, 0, 0])
        np.testing.assert_array_almost_equal(q, expected_q)
        
        # Test array of vectors
        hkl_array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        q_array = mapper.hkl_to_cartesian(hkl_array)
        self.assertEqual(q_array.shape, (3, 3))
    
    def test_cartesian_to_hkl(self):
        """Test Cartesian to Miller indices conversion."""
        mapper = CoordinateMapper(self.A_inv)
        
        # Test roundtrip conversion
        hkl_original = np.array([1.5, -2.3, 0.7])
        q = mapper.hkl_to_cartesian(hkl_original)
        hkl_recovered = mapper.cartesian_to_hkl(q)
        np.testing.assert_array_almost_equal(hkl_original, hkl_recovered)
    
    def test_grid_generation(self):
        """Test grid coordinate generation."""
        mapper = CoordinateMapper(self.A_inv, sampling=self.sampling)
        
        # Test HKL grid generation
        hkl_grid, map_shape = mapper.get_grid_coordinates(return_hkl=True)
        
        # Check expected shape
        expected_h_steps = int(3 * (2 - (-2)) + 1)  # 13
        expected_k_steps = int(2 * (3 - (-3)) + 1)  # 13
        expected_l_steps = int(4 * (1 - (-1)) + 1)  # 9
        
        self.assertEqual(map_shape, (expected_h_steps, expected_k_steps, expected_l_steps))
        self.assertEqual(hkl_grid.shape, (expected_h_steps * expected_k_steps * expected_l_steps, 3))
        
        # Test Q-vector grid generation
        q_grid, map_shape_q = mapper.get_grid_coordinates(return_hkl=False)
        self.assertEqual(map_shape_q, map_shape)
        self.assertEqual(q_grid.shape, hkl_grid.shape)
    
    def test_index_to_hkl(self):
        """Test array index to Miller index conversion."""
        mapper = CoordinateMapper(self.A_inv, sampling=self.sampling)
        
        # Test corner indices
        h, k, l = mapper.index_to_hkl(0, 0, 0)
        self.assertAlmostEqual(h, -2)
        self.assertAlmostEqual(k, -3)
        self.assertAlmostEqual(l, -1)
        
        # Test middle index (approximately)
        h_steps = int(3 * (2 - (-2)) + 1)
        k_steps = int(2 * (3 - (-3)) + 1)
        l_steps = int(4 * (1 - (-1)) + 1)
        
        h, k, l = mapper.index_to_hkl(h_steps // 2, k_steps // 2, l_steps // 2)
        self.assertAlmostEqual(h, 0, places=1)
        self.assertAlmostEqual(k, 0, places=1)
        self.assertAlmostEqual(l, 0, places=1)
    
    def test_hkl_to_index(self):
        """Test Miller index to array index conversion."""
        mapper = CoordinateMapper(self.A_inv, sampling=self.sampling)
        
        # Test origin
        i, j, k = mapper.hkl_to_index(0, 0, 0)
        h_steps = int(3 * (2 - (-2)) + 1)
        k_steps = int(2 * (3 - (-3)) + 1)
        l_steps = int(4 * (1 - (-1)) + 1)
        
        self.assertEqual(i, h_steps // 2)
        self.assertEqual(j, k_steps // 2)
        self.assertEqual(k, l_steps // 2)


class TestNaNProcessor(unittest.TestCase):
    """Test the NaNProcessor class."""
    
    def setUp(self):
        """Create test data with NaN values."""
        # 1D data
        self.data_1d = np.array([1, 2, np.nan, 4, 5, np.nan, 7, 8])
        
        # 2D data
        self.data_2d = np.array([
            [1, 2, np.nan],
            [4, np.nan, 6],
            [7, 8, 9]
        ])
        
        # 3D data
        self.data_3d = np.ones((3, 3, 3))
        self.data_3d[0, 0, 0] = np.nan
        self.data_3d[1, 1, 1] = np.nan
        self.data_3d[2, 2, 2] = np.nan
    
    def test_mask_strategy(self):
        """Test masking NaN values."""
        result = NaNProcessor.apply_strategy(self.data_2d, 'mask')
        
        self.assertIsInstance(result, np.ma.MaskedArray)
        self.assertEqual(result.mask.sum(), 2)  # 2 NaN values
        self.assertFalse(result.mask[0, 0])  # Valid value not masked
        self.assertTrue(result.mask[0, 2])   # NaN value masked
    
    def test_zero_strategy(self):
        """Test replacing NaN with zero."""
        result = NaNProcessor.apply_strategy(self.data_1d, 'zero')
        
        self.assertFalse(np.any(np.isnan(result)))
        self.assertEqual(result[2], 0)
        self.assertEqual(result[5], 0)
        self.assertEqual(result[0], 1)  # Original value preserved
    
    def test_min_strategy(self):
        """Test replacing NaN with minimum value."""
        result = NaNProcessor.apply_strategy(self.data_1d, 'min')
        
        self.assertFalse(np.any(np.isnan(result)))
        min_val = np.min(self.data_1d[~np.isnan(self.data_1d)])
        self.assertEqual(result[2], min_val)
        self.assertEqual(result[5], min_val)
    
    def test_mean_strategy(self):
        """Test replacing NaN with mean value."""
        result = NaNProcessor.apply_strategy(self.data_1d, 'mean')
        
        self.assertFalse(np.any(np.isnan(result)))
        mean_val = np.mean(self.data_1d[~np.isnan(self.data_1d)])
        self.assertAlmostEqual(result[2], mean_val)
        self.assertAlmostEqual(result[5], mean_val)
    
    def test_interpolate_1d(self):
        """Test 1D interpolation of NaN values."""
        result = NaNProcessor.apply_strategy(self.data_1d, 'interpolate', method='linear')
        
        self.assertFalse(np.any(np.isnan(result)))
        # Linear interpolation between 2 and 4 should give 3
        self.assertAlmostEqual(result[2], 3, places=1)
    
    def test_interpolate_2d(self):
        """Test 2D interpolation of NaN values."""
        result = NaNProcessor.apply_strategy(self.data_2d, 'interpolate', method='nearest')
        
        self.assertFalse(np.any(np.isnan(result)))
        self.assertEqual(result.shape, self.data_2d.shape)
    
    def test_interpolate_3d(self):
        """Test 3D interpolation of NaN values."""
        result = NaNProcessor.apply_strategy(self.data_3d, 'interpolate')
        
        self.assertFalse(np.any(np.isnan(result)))
        self.assertEqual(result.shape, self.data_3d.shape)
    
    def test_remove_strategy(self):
        """Test removing NaN values."""
        result = NaNProcessor.apply_strategy(self.data_1d, 'remove')
        
        self.assertFalse(np.any(np.isnan(result)))
        self.assertEqual(len(result), 6)  # 8 original - 2 NaN
        np.testing.assert_array_equal(result, [1, 2, 4, 5, 7, 8])
    
    def test_create_mask(self):
        """Test mask creation."""
        mask = NaNProcessor.create_mask(self.data_2d)
        
        self.assertEqual(mask.shape, self.data_2d.shape)
        self.assertTrue(mask[0, 0])   # Valid value
        self.assertFalse(mask[0, 2])  # NaN value
        self.assertFalse(mask[1, 1])  # NaN value
    
    def test_get_statistics(self):
        """Test NaN statistics calculation."""
        stats = NaNProcessor.get_statistics(self.data_3d)
        
        self.assertEqual(stats['n_nan'], 3)
        self.assertEqual(stats['n_valid'], 27 - 3)
        self.assertEqual(stats['n_total'], 27)
        self.assertAlmostEqual(stats['nan_percentage'], 100 * 3 / 27, places=2)
        
        # Check per-slice statistics
        self.assertIn('nan_per_h_slice', stats)
        self.assertIn('nan_per_k_slice', stats)
        self.assertIn('nan_per_l_slice', stats)


class TestIntegration(unittest.TestCase):
    """Test integration of all components."""
    
    def setUp(self):
        """Create integrated test scenario."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create realistic test data
        self.A_inv = np.array([
            [0.1, 0.01, 0.0],
            [0.0, 0.1, 0.0],
            [0.01, 0.0, 0.1]
        ])
        
        # Generate intensity with some NaN values
        self.intensity = np.random.rand(10, 10, 10)
        # Add NaN at "lattice points"
        for i in range(0, 10, 3):
            for j in range(0, 10, 3):
                for k in range(0, 10, 3):
                    if i % 3 == 0 and j % 3 == 0 and k % 3 == 0:
                        self.intensity[i, j, k] = np.nan
        
        # Save test file
        self.npz_path = os.path.join(self.temp_dir, "integrated_test.npz")
        np.savez_compressed(
            self.npz_path,
            intensity=self.intensity,
            q_vectors=np.random.rand(1000, 3),
            map_shape=(10, 10, 10)
        )
    
    def tearDown(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline(self):
        """Test complete data processing pipeline."""
        # Step 1: Load data
        handler = IntensityDataHandler(self.npz_path)
        q_vectors, intensity, map_shape = handler.load_data()
        
        self.assertIsNotNone(intensity)
        self.assertEqual(map_shape, (10, 10, 10))
        
        # Step 2: Process NaN values
        processor = NaNProcessor()
        processed = processor.apply_strategy(intensity, 'interpolate')
        
        self.assertFalse(np.any(np.isnan(processed)))
        
        # Step 3: Set up coordinate mapping
        sampling = ((-2, 2, 2.5), (-2, 2, 2.5), (-2, 2, 2.5))
        mapper = CoordinateMapper(self.A_inv, shape=map_shape, sampling=sampling)
        
        # Step 4: Get coordinates
        hkl_grid, shape = mapper.get_grid_coordinates(return_hkl=True)
        q_grid, _ = mapper.get_grid_coordinates(return_hkl=False)
        
        self.assertEqual(hkl_grid.shape[0], np.prod(shape))
        self.assertEqual(q_grid.shape[0], np.prod(shape))
        
        # Step 5: Test roundtrip conversion
        q_test = mapper.hkl_to_cartesian(hkl_grid[0])
        hkl_recovered = mapper.cartesian_to_hkl(q_test)
        np.testing.assert_array_almost_equal(hkl_grid[0], hkl_recovered)
    
    def test_different_nan_strategies(self):
        """Test all NaN strategies on same data."""
        handler = IntensityDataHandler(self.intensity)
        handler.load_data()
        
        strategies = ['mask', 'zero', 'min', 'mean', 'interpolate']
        results = {}
        
        for strategy in strategies:
            results[strategy] = NaNProcessor.apply_strategy(self.intensity.copy(), strategy)
        
        # Check all strategies handle NaN
        for strategy in ['zero', 'min', 'mean', 'interpolate']:
            self.assertFalse(np.any(np.isnan(results[strategy])))
        
        # Mask should be a MaskedArray
        self.assertIsInstance(results['mask'], np.ma.MaskedArray)


# Run tests if called directly
if __name__ == '__main__':
    unittest.main()