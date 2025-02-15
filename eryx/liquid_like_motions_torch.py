import torch
from typing import Tuple, Union, Optional
from .models_torch import ModelBase, DeviceManager
from .base import compute_molecular_transform, compute_crystal_transform
from .pdb import AtomicModel
from .map_utils import generate_grid, get_resolution_mask

class LiquidLikeMotionsTorch(ModelBase):
    """
    PyTorch implementation of Liquid Like Motions model.
    Model in which collective motions decay exponentially with distance
    across the crystal. Mathematically the predicted diffuse scattering 
    is the convolution between the crystal transform and a disorder kernel.
    """
    
    def __init__(
        self,
        pdb_path: str,
        hsampling: Tuple[float, float, float],
        ksampling: Tuple[float, float, float],
        lsampling: Tuple[float, float, float],
        expand_p1: bool = True,
        border: int = 1,
        res_limit: float = 0,
        batch_size: int = 5000,
        n_processes: int = 8,
        asu_confined: bool = False,
        device: str = 'cpu'
    ) -> None:
        super().__init__(device)
        self.device_manager = DeviceManager(device)
        self.hsampling = hsampling
        self.ksampling = ksampling
        self.lsampling = lsampling
        self._setup(pdb_path, expand_p1, border, res_limit, batch_size, n_processes, asu_confined)

    def _setup(
        self,
        pdb_path: str,
        expand_p1: bool,
        border: int,
        res_limit: float,
        batch_size: int,
        n_processes: int,
        asu_confined: bool
    ) -> None:
        # Generate atomic model
        model = AtomicModel(pdb_path, expand_p1=expand_p1)
        model.flatten_model()
        
        # Get grid for padded map
        hsampling_padded = (self.hsampling[0]-border, self.hsampling[1]+border, self.hsampling[2])
        ksampling_padded = (self.ksampling[0]-border, self.ksampling[1]+border, self.ksampling[2])
        lsampling_padded = (self.lsampling[0]-border, self.lsampling[1]+border, self.lsampling[2])
        
        hkl_grid, self.map_shape = generate_grid(
            model.A_inv,
            hsampling_padded,
            ksampling_padded,
            lsampling_padded,
            return_hkl=True
        )
        
        # Convert numpy arrays to torch tensors
        self.res_mask, res_map = get_resolution_mask(model.cell, hkl_grid, res_limit)
        self.res_mask = torch.from_numpy(self.res_mask).to(self.device)
        
        # Compute crystal or molecular transform
        if not asu_confined:
            q_grid_np, transform_np = compute_crystal_transform(
                pdb_path,
                hsampling_padded,
                ksampling_padded,
                lsampling_padded,
                expand_p1=expand_p1,
                res_limit=res_limit,
                batch_size=batch_size,
                n_processes=n_processes
            )
        else:
            q_grid_np, transform_np = compute_molecular_transform(
                pdb_path,
                hsampling_padded,
                ksampling_padded,
                lsampling_padded,
                expand_p1=expand_p1,
                expand_friedel=False,
                res_limit=res_limit,
                batch_size=batch_size,
                n_processes=n_processes
            )
            
        self.q_grid = torch.from_numpy(q_grid_np).to(self.device)
        self.transform = torch.from_numpy(transform_np).to(self.device)
        self.q_mags = torch.linalg.norm(self.q_grid, dim=1)
        
        # Generate mask for padded region
        self.mask = torch.zeros(self.map_shape, device=self.device)
        self.mask[
            border*self.hsampling[2]:-border*self.hsampling[2],
            border*self.ksampling[2]:-border*self.ksampling[2],
            border*self.lsampling[2]:-border*self.lsampling[2]
        ] = 1
        
        self.map_shape_nopad = tuple(
            torch.tensor(self.map_shape) - 
            torch.tensor([
                2*border*self.hsampling[2],
                2*border*self.ksampling[2],
                2*border*self.lsampling[2]
            ])
        )
        
        # Empty lists to populate with all sigmas/gammas that have been scanned
        self.scan_sigmas = []
        self.scan_gammas = []
        self.scan_ccs = []
        self.opt_map = None

    def fft_convolve(
        self,
        transform: torch.Tensor,
        kernel: torch.Tensor
    ) -> torch.Tensor:
        """Convolve transform and kernel using FFT."""
        ft_transform = torch.fft.fftn(transform)
        ft_kernel = torch.fft.fftn(kernel/kernel.sum())
        ft_conv = ft_transform * ft_kernel
        return torch.fft.ifftn(ft_conv).real

    def apply_disorder(
        self,
        sigmas: Union[float, torch.Tensor],
        gammas: Union[float, torch.Tensor] = 1.0
    ) -> torch.Tensor:
        """
        Compute the diffuse map(s) from the crystal transform.
        
        Parameters
        ----------
        sigmas : float or torch.Tensor of shape (n_sigma,) or (n_sigma, 3)
            (an)isotropic displacement parameter for asymmetric unit 
        gammas : float or torch.Tensor of shape (n_gamma,)
            kernel's correlation length
            
        Returns
        -------
        Id : torch.Tensor, (n_sigma*n_gamma, q_grid.shape[0])
            diffuse intensity maps for the corresponding parameters
        """
        # Convert inputs to tensors if needed
        if isinstance(gammas, (float, int)):
            gammas = torch.tensor([gammas], device=self.device, dtype=torch.float64)
        if isinstance(sigmas, (float, int)):
            sigmas = torch.tensor([sigmas], device=self.device, dtype=torch.float64)
            
        # Generate kernel and convolve with transform
        Id = torch.zeros((len(gammas), self.q_grid.shape[0]), device=self.device, dtype=torch.float64)
        kernels = 8.0 * torch.pi * (gammas.unsqueeze(1)**3) / torch.square(
            1 + torch.square(gammas.unsqueeze(1) * self.q_mags)
        )
        
        for num in range(len(gammas)):
            # Use conv3d for small maps (more efficient for small tensors)
            if int(torch.prod(torch.tensor(self.map_shape)).item()) < 1e7:
                kernel_reshaped = kernels[num].reshape(self.map_shape)
                kernel_norm = kernel_reshaped / kernel_reshaped.sum()
                Id[num] = torch.nn.functional.conv3d(
                    self.transform.unsqueeze(0).unsqueeze(0),
                    kernel_norm.unsqueeze(0).unsqueeze(0),
                    padding='same'
                ).squeeze().flatten()
            else:
                Id[num] = self.fft_convolve(
                    self.transform,
                    kernels[num].reshape(self.map_shape)
                ).flatten()
                
        Id = Id.repeat(len(sigmas), 1)

        # Scale with displacement parameters
        if len(sigmas.shape) == 1:
            sigmas = sigmas.repeat_interleave(len(gammas))
            q2s2 = torch.square(self.q_mags) * torch.square(sigmas).unsqueeze(1)
        else:
            sigmas = sigmas.repeat_interleave(len(gammas), dim=0)
            q2s2 = torch.sum(
                self.q_grid.T * torch.matmul(
                    torch.square(sigmas).unsqueeze(1) * torch.eye(3, device=self.device),
                    self.q_grid.T
                ),
                dim=1
            )

        Id *= torch.exp(-1*q2s2) * q2s2
        Id[:, ~self.res_mask] = torch.nan
        return Id
