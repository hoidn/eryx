# Variational Gaussian Approach for Modeling Broadband Diffuse Scattering

## 1. Introduction and Motivation

Diffuse X-ray scattering provides valuable information about correlated atomic motion in crystals, particularly through phonon models like OnePhonon. However, experimental limitations create challenges in modeling and interpreting the data. One significant challenge is broadband illumination, where a range of wavelengths contributes to the measured intensity at each detector pixel.

Traditional approaches treat each detector pixel as sampling a single point in reciprocal space, but in reality, each pixel samples a "slab" or volume. This mismatch can lead to:

- Incorrect interpretation of peak widths and shapes
- Challenges in parameter fitting and optimization
- Difficulty in computing derivatives and gradients

The Variational Gaussian approach addresses these challenges by explicitly modeling uncertainty in k-space, representing each sampling point as a 3D Gaussian distribution rather than a discrete point. This approach is particularly valuable for:

1. Building end-to-end differentiable models
2. Accurate representation of experimental conditions
3. Optimization of model parameters via gradient-based methods
4. Quantifying uncertainty in fitted parameters

## 2. Physical Interpretation

### Broadband Illumination Effects

When a crystal is illuminated with a broadband source (range of wavelengths λ ± Δλ), each detector pixel samples not a point but a volume in reciprocal space. This volume has:

- Width and height determined by the pixel size and detector geometry
- Depth determined by the wavelength distribution

The relationship between wavelength and reciprocal space sampling follows:

$$|\mathbf{k}| = \frac{2\pi}{\lambda}$$

So wavelength uncertainty propagates to uncertainty in the magnitude of k:

$$\sigma_{|k|} = \frac{2\pi}{\lambda^2}\sigma_{\lambda}$$

This uncertainty is predominantly along the beam direction, creating an elongated Gaussian distribution in k-space.

### Scattering Geometry

For a flat detector, the relationship between detector coordinates (x,y) and k-vector components depends on the experimental geometry:

$$\mathbf{k} = \mathbf{k_i} + \mathbf{k_f}$$

where $\mathbf{k_i}$ is the incident wavevector and $\mathbf{k_f}$ is the scattered wavevector. The uncertainty in wavelength affects both $|\mathbf{k_i}|$ and $|\mathbf{k_f}|$.

## 3. Mathematical Framework

### K-Space as a Probability Distribution

In the Variational Gaussian approach, we model each point in k-space not as a discrete point but as a 3D Gaussian probability distribution:

$$p(\mathbf{k}) = \mathcal{N}(\mathbf{k}|\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

Where:
- $\boldsymbol{\mu}$ is the mean k-vector (3D)
- $\boldsymbol{\Sigma}$ is the covariance matrix (3×3)

The covariance matrix encodes both the magnitude and directional uncertainty, with principal elongation along the beam direction.

### Expected Intensity

Instead of calculating the intensity at a single point, we compute the expected intensity under this distribution:

$$E[I(\mathbf{k})] = \int I(\mathbf{k})p(\mathbf{k})d\mathbf{k}$$

This integral can be approximated by:

1. **Monte Carlo sampling**: Draw samples from the Gaussian and average
2. **Laplace approximation**: Use Taylor expansion around the mean

For the Monte Carlo approach, we use the reparameterization trick to maintain differentiability:

$$\mathbf{k} = \boldsymbol{\mu} + \mathbf{L}\boldsymbol{\epsilon}$$

Where $\mathbf{L}$ is the Cholesky decomposition of $\boldsymbol{\Sigma}$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$.

## 4. Implementation in PyTorch

```python
class VariationalGaussianDiffuseScattering(torch.nn.Module):
    def __init__(self, phonon_model, wavelength_distribution, experiment_geometry):
        """
        Parameters:
        -----------
        phonon_model: OnePhonon model implemented as torch.nn.Module
        wavelength_distribution: tuple (mean, std) of wavelength in Angstroms
        experiment_geometry: contains detector distance, pixel size, etc.
        """
        super().__init__()
        self.phonon_model = phonon_model
        self.wavelength_mean = wavelength_distribution[0]
        # Make wavelength std a learnable parameter
        self.wavelength_std = torch.nn.Parameter(torch.tensor(wavelength_distribution[1]))
        self.geometry = experiment_geometry
        
    def detector_to_k_space(self, detector_coords):
        """
        Convert detector coordinates to mean k-vectors
        
        Parameters:
        -----------
        detector_coords: torch.Tensor, shape [n_pixels, 2]
            x,y coordinates on detector
            
        Returns:
        --------
        k_means: torch.Tensor, shape [n_pixels, 3]
            Mean k-vectors in reciprocal space
        """
        # Implementation depends on specific geometry
        # For example, for a simple perpendicular detector:
        detector_distance = self.geometry['detector_distance']
        pixel_size = self.geometry['pixel_size']
        
        # Convert to scattering angles
        x, y = detector_coords[:, 0], detector_coords[:, 1]
        two_theta = torch.atan(torch.sqrt(x**2 + y**2) * pixel_size / detector_distance)
        gamma = torch.atan2(y, x)
        
        # Calculate k-vector components
        k_magnitude = 2 * np.pi / self.wavelength_mean
        kx = k_magnitude * (1 - torch.cos(two_theta))
        ky = k_magnitude * torch.sin(two_theta) * torch.cos(gamma)
        kz = k_magnitude * torch.sin(two_theta) * torch.sin(gamma)
        
        return torch.stack([kx, ky, kz], dim=1)
    
    def construct_k_covariances(self, k_means, detector_coords):
        """
        Construct covariance matrices for each k-point
        
        Parameters:
        -----------
        k_means: torch.Tensor, shape [n_pixels, 3]
            Mean k-vectors
        detector_coords: torch.Tensor, shape [n_pixels, 2]
            Detector coordinates
            
        Returns:
        --------
        k_covs: torch.Tensor, shape [n_pixels, 3, 3]
            Covariance matrices for each k-point
        """
        n_pixels = k_means.shape[0]
        
        # Calculate beam directions for each pixel
        beam_dirs = self.calculate_beam_directions(detector_coords)  # [n_pixels, 3]
        
        # Calculate k-magnitude uncertainty from wavelength uncertainty
        k_magnitudes = torch.norm(k_means, dim=1)
        k_magnitude_variance = (2 * np.pi * self.wavelength_std / self.wavelength_mean**2)**2 * k_magnitudes**2
        
        # Create covariance matrices
        k_covs = torch.zeros(n_pixels, 3, 3, device=k_means.device)
        
        for i in range(n_pixels):
            # Start with small base variance in all directions (due to pixel size)
            base_var = (0.01 * k_magnitudes[i])**2
            k_covs[i] = torch.eye(3, device=k_means.device) * base_var
            
            # Add directional variance along beam
            beam_dir = beam_dirs[i].view(3, 1)
            k_covs[i] += k_magnitude_variance[i] * torch.matmul(beam_dir, beam_dir.t())
        
        return k_covs
        
    def calculate_beam_directions(self, detector_coords):
        """Calculate normalized beam direction vectors for each pixel"""
        # Implementation depends on geometry
        # This is a simplified version
        detector_distance = self.geometry['detector_distance']
        pixel_size = self.geometry['pixel_size']
        
        # Convert to 3D coordinates
        x = detector_coords[:, 0] * pixel_size
        y = detector_coords[:, 1] * pixel_size
        z = torch.ones_like(x) * detector_distance
        
        # Normalize
        norm = torch.sqrt(x**2 + y**2 + z**2)
        return torch.stack([x/norm, y/norm, z/norm], dim=1)
    
    def expected_intensity(self, k_means, k_covs, num_samples=20):
        """
        Compute expected intensity under k-space Gaussian distribution
        
        Parameters:
        -----------
        k_means: torch.Tensor, shape [n_pixels, 3]
            Mean k-vectors
        k_covs: torch.Tensor, shape [n_pixels, 3, 3]
            Covariance matrices
        num_samples: int
            Number of MC samples to use
            
        Returns:
        --------
        expected_I: torch.Tensor, shape [n_pixels]
            Expected intensity values
        """
        batch_size = k_means.shape[0]
        
        # Using reparameterization trick for differentiable sampling
        eps = torch.randn(batch_size, num_samples, 3, device=k_means.device)
        
        # Compute Cholesky decomposition of covariance matrices
        # Add small diagonal term for numerical stability
        jitter = torch.eye(3, device=k_means.device) * 1e-6
        L = torch.linalg.cholesky(k_covs + jitter.unsqueeze(0))
        
        # Generate samples: k = μ + L·ε
        k_samples = k_means.unsqueeze(1) + torch.bmm(
            L, eps.transpose(1, 2)
        ).transpose(1, 2)
        
        # Reshape for batch processing: [batch_size*num_samples, 3]
        k_flat = k_samples.reshape(-1, 3)
        
        # Compute intensity at all sampled points
        I_flat = self.phonon_model(k_flat)
        
        # Reshape back and average over samples
        I_samples = I_flat.reshape(batch_size, num_samples)
        expected_I = I_samples.mean(dim=1)
        
        return expected_I
    
    def forward(self, detector_coords):
        """
        Forward pass: detector coordinates to intensities
        
        Parameters:
        -----------
        detector_coords: torch.Tensor, shape [n_pixels, 2]
            x,y coordinates on detector
            
        Returns:
        --------
        intensities: torch.Tensor, shape [n_pixels]
            Expected intensity at each pixel
        """
        # Convert detector coordinates to mean k-vectors
        k_means = self.detector_to_k_space(detector_coords)
        
        # Construct covariance matrices
        k_covs = self.construct_k_covariances(k_means, detector_coords)
        
        # Compute expected intensity
        intensities = self.expected_intensity(k_means, k_covs)
        
        return intensities
```

## 5. How to Use for Parameter Optimization

To use this framework for parameter optimization, we create an end-to-end differentiable pipeline:

```python
# Initialize models
pdb_path = "protein.pdb"
phonon_params = {"gamma_intra": 1.0, "gamma_inter": 0.8, "gnm_cutoff": 4.0}
wavelength_dist = (1.54, 0.01)  # Angstroms (mean, std)
experiment_geometry = {"detector_distance": 100.0, "pixel_size": 0.1}

# Create models
phonon_model = DifferentiableOnePhonon(pdb_path, **phonon_params)
diffuse_model = VariationalGaussianDiffuseScattering(
    phonon_model, wavelength_dist, experiment_geometry
)

# Prepare optimizer
# Parameters to optimize:
params_to_optimize = [
    {'params': [phonon_model.gamma_intra, phonon_model.gamma_inter]},
    {'params': [diffuse_model.wavelength_std], 'lr': 0.001}
]
optimizer = torch.optim.Adam(params_to_optimize, lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    simulated_intensity = diffuse_model(detector_coords)
    
    # Loss calculation
    loss = torch.nn.functional.mse_loss(simulated_intensity, experimental_intensity)
    
    # Optimization step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## 6. Advantages over Other Approaches

The Variational Gaussian approach offers several advantages:

1. **Physically Accurate Modeling**:
   - Properly accounts for wavelength dispersion effects
   - Models the true 3D sampling volume at each detector pixel

2. **Computational Efficiency**:
   - More efficient than explicit integration or sampling methods
   - Can use modern GPU acceleration techniques

3. **Statistical Rigor**:
   - Provides uncertainty quantification
   - Allows principled incorporation of prior knowledge

4. **Gradient Properties**:
   - Smoother gradients for optimization
   - Differentiable end-to-end pipeline

5. **Extensibility**:
   - Can incorporate other sources of uncertainty:
     - Mosaic spread
     - Detector point spread function
     - Sample positioning errors

## 7. Theoretical Connections

This approach connects to several important theoretical frameworks:

- **Variational Inference**: Using probabilistic methods to approximate complex distributions
- **Monte Carlo Methods**: Sampling-based approximation of expected values
- **Diffraction Theory**: Physically accurate modeling of scattering processes
- **Uncertainty Propagation**: Rigorous handling of experimental uncertainties

## Conclusion

The Variational Gaussian approach provides a mathematically rigorous and computationally efficient framework for modeling diffuse scattering with broadband illumination. By explicitly representing k-space uncertainty as 3D Gaussian distributions, it enables end-to-end differentiable modeling and parameter optimization while maintaining physical accuracy.

This approach is particularly valuable for extracting phonon model parameters from experimental data, where the interplay between model parameters, experimental conditions, and measured intensities must be carefully handled. The ability to compute derivatives of the intensity with respect to k and backpropagate to parameters like gamma makes it a powerful tool for understanding collective atomic motions in crystalline materials.
