
 UPDATE modules/gaussian_network_torch.py:                                                                                                                                                             
     IN CLASS GaussianNetworkModelTorch(nn.Module):                                                                                                                                                    
         • IMPLEMENT compute_hessian_torch():                                                                                                                                                          
             - Replace the current stub with a vectorized torch implementation that uses torch operations (e.g. torch.matmul, torch.where) to compute the Hessian from neighbor lists.                 
             - Ensure the computation maintains gradient flow with torch tensors.                                                                                                                      
         • IMPLEMENT compute_K_torch(hessian: torch.Tensor, kvec: torch.Tensor) → torch.Tensor:                                                                                                        
             - Vectorize the summation over cells and use torch.exp for the phase factors.                                                                                                             
             - Confirm that the output tensor is differentiable.                                                                                                                                       
         • IMPLEMENT compute_gnm_phonons_torch():                                                                                                                                                      
             - Use torch.linalg.svd (or torch.svd) for eigendecomposition.                                                                                                                             
             - Store eigenvectors and inverse eigenvalues while ensuring gradients propagate.                                                                                                          
         • VERIFY that build_gamma() and build_neighbor_list() produce tensors on the correct device and the learnable parameters are properly registered.                                             
                                                                                                                                                                                                       
 UPDATE modules/onephonon_torch.py:                                                                                                                                                                    
     IN CLASS OnePhononTorch(nn.Module, ModelRunner):                                                                                                                                                  
         • IMPLEMENT compute_covariance_matrix_torch() fully:                                                                                                                                          
             - Replace the placeholder “return torch.tensor(0.0, device=self.device)” with a vectorized implementation computing the covariance matrix from the phonon mode outputs (using self.V,     
 self.Winv, etc.).                                                                                                                                                                                     
             - Ensure all operations (including sums and element–wise products) are done with torch functions.                                                                                         
         • IMPLEMENT apply_disorder(use_data_adp: bool = False) with full vectorization:                                                                                                               
             - Convert the structure factor computation and grid postprocessing to torch without breaking gradient chains.                                                                             
             - Ensure preprocessing (grid generation, resolution mask computation) either is done fully in torch or is isolated as non–gradient branches that properly feed into the gradient–preservi 
 region.                                                                                                                                                                                               
         • UPDATE forward():                                                                                                                                                                           
             - Chain together the phonon mode extraction, covariance computation, and diffuse intensity computation.                                                                                   
             - Ensure that gradients can backpropagate from the final loss to the learnable parameters (gamma_intra and gamma_inter).                                                                  
                                                                                                                                                                                                       
 UPDATE modules/onephonon_torch.py AND associated utilities:                                                                                                                                           
     • REVIEW all preprocessing steps that currently use NumPy (grid generation, resolution mask, structure factor evaluation):                                                                        
          - Where possible, re–implement these steps in torch or strictly convert outputs to torch tensors in a controlled “gradient entry” point to preserve device consistency.                      
          - REMOVE any inadvertent use of .detach() or np.asarray() that could break the gradient chain.                                                                                               
                                                                                                                                                                                                       
 CREATE tests/test_onephonon_torch.py:                                                                                                                                                                 
     • ADD tests that:                                                                                                                                                                                 
          - Verify that gradients flow to the learnable parameters (e.g. gamma_intra, gamma_inter) via backward() on a suitable loss.                                                                  
          - Compare outputs (e.g. diffuse intensity maps, phonon mode eigenvalues) between the torch and numpy implementations under tight numerical tolerances.                                       
          - Validate that all tensors reside on the expected device (CPU/CUDA) throughout the computational pipeline.                                                                                  
          - Benchmark performance and memory usage to ensure vectorization is effective.                                                                                                               
                                                                                                                                                                                                       
 GENERAL RECOMMENDATIONS:                                                                                                                                                                              
     • Ensure all torch operations maintain appropriate dtype (e.g. torch.float64 for physics computations) and are consistently moved to self.device.                                                 
     • Replace explicit Python loops with batched torch operations wherever feasible.                                                                                                                  
     • Integrate occasional checkpointing if memory rises during large computations.                                                                                                                   
     • Rigorously test each core function using the expanded test suite to verify numerical equivalence with the numpy version.                                                                        
