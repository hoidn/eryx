 # PyTorch Implementation Specification for GNM/OnePhonon Models                                                                                                                  
                                                                                                                                                                                  
 ## High-Level Objective                                                                                                                                                          
 - Create PyTorch implementations of the Gaussian Network Model and One-Phonon model while maintaining numerical accuracy with the original numpy implementation                  
                                                                                                                                                                                  
 ## Mid-Level Objectives                                                                                                                                                          
 1. Create PyTorch GNM implementation with validated numerical accuracy                                                                                                           
 2. Create PyTorch OnePhonon implementation using the torch GNM                                                                                                                   
 3. Maintain test coverage ensuring numerical equivalence                                                                                                                         
 4. Support both CPU and GPU computation                                                                                                                                          
 5. Keep data loading/preprocessing in numpy for now                                                                                                                              
                                                                                                                                                                                  
 ## Implementation Notes                                                                                                                                                          
                                                                                                                                                                                  
 ### Dependencies                                                                                                                                                                 
 - PyTorch >= 2.0                                                                                                                                                                 
 - Existing numpy implementation                                                                                                                                                  
 - pytest for testing                                                                                                                                                             
 - Sample PDB files in tests/pdbs/                                                                                                                                                
                                                                                                                                                                                  
 ### Technical Guidelines                                                                                                                                                         
 - Focus on numerical accuracy first, optimization later                                                                                                                          
 - Use torch.tensor for all core computations                                                                                                                                     
 - Keep numpy for initial data loading/preprocessing                                                                                                                              
 - Add detailed logging for debugging                                                                                                                                             
 - Follow existing code structure where possible                                                                                                                                  
                                                                                                                                                                                  
 ## Test Requirements                                                                                                                                                             
                                                                                                                                                                                  
 ### GaussianNetworkModelTorch Tests                                                                                                                                              
 1. Test neighbor list construction                                                                                                                                               
    - Verify matches numpy implementation                                                                                                                                         
    - Check connectivity patterns                                                                                                                                                 
    - Validate distance cutoffs                                                                                                                                                   
                                                                                                                                                                                  
 2. Test Hessian computation                                                                                                                                                      
    - Verify symmetry properties                                                                                                                                                  
    - Match numpy output exactly                                                                                                                                                  
    - Test with and without k-vectors                                                                                                                                             
    - Validate shape and dtype                                                                                                                                                    
                                                                                                                                                                                  
 3. Test K matrix computation                                                                                                                                                     
    - Verify phase factors                                                                                                                                                        
    - Test eigenvalue properties                                                                                                                                                  
    - Match numpy implementation                                                                                                                                                  
    - Handle edge cases                                                                                                                                                           
                                                                                                                                                                                  
 4. Test numerical stability                                                                                                                                                      
    - Verify matrix inversion                                                                                                                                                     
    - Test gradient flow                                                                                                                                                          
    - Validate device transfers                                                                                                                                                   
                                                                                                                                                                                  
 ### OnePhononTorch Tests                                                                                                                                                         
 1. Test initialization                                                                                                                                                           
    - Verify parameter handling                                                                                                                                                   
    - Check device placement                                                                                                                                                      
    - Validate tensor conversions                                                                                                                                                 
                                                                                                                                                                                  
 2. Test disorder application                                                                                                                                                     
    - Match numpy implementation                                                                                                                                                  
    - Verify shape consistency                                                                                                                                                    
    - Test numerical accuracy                                                                                                                                                     
    - Handle edge cases                                                                                                                                                           
                                                                                                                                                                                  
 3. Test integration                                                                                                                                                              
    - End-to-end workflow                                                                                                                                                         
    - Device compatibility                                                                                                                                                        
    - Memory efficiency                                                                                                                                                           
    - Gradient computation                                                                                                                                                        
                                                                                                                                                                                  
 ## Implementation Order                                                                                                                                                          
 1. GaussianNetworkModelTorch                                                                                                                                                     
    - Basic initialization                                                                                                                                                        
    - Neighbor list computation                                                                                                                                                   
    - Hessian computation                                                                                                                                                         
    - K matrix operations                                                                                                                                                         
    - Device support                                                                                                                                                              
                                                                                                                                                                                  
 2. OnePhononTorch                                                                                                                                                                
    - Model initialization                                                                                                                                                        
    - GNM integration                                                                                                                                                             
    - Disorder computation                                                                                                                                                        
    - Device support                                                                                                                                                              
                                                                                                                                                                                  
 3. Testing Infrastructure                                                                                                                                                        
    - Unit tests                                                                                                                                                                  
    - Integration tests                                                                                                                                                           
    - Performance benchmarks                                                                                                                                                      
    - Validation utilities                                                                                                                                                        
                                                                                                                                                                                  
 ## Validation Requirements                                                                                                                                                       
 - Exact numerical match with numpy (within float precision)                                                                                                                      
 - All tests must pass on both CPU and GPU                                                                                                                                        
 - Memory usage should be reasonable                                                                                                                                              
 - Gradients should flow properly                                                                                                                                                 
 - Log key computations for debugging                                                                                                                                             
                                                                                                                                                                                  
 ## Key Implementation Details                                                                                                                                                    
                                                                                                                                                                                  
 ### GaussianNetworkModelTorch                                                                                                                                                    
 - Keep AtomicModel loading in numpy                                                                                                                                              
 - Convert coordinates to torch.tensor after loading                                                                                                                              
 - Implement compute_hessian() using torch operations                                                                                                                             
 - Implement compute_Kinv() using torch.linalg                                                                                                                                    
 - Add device support via .to(device)                                                                                                                                             
                                                                                                                                                                                  
 ### OnePhononTorch                                                                                                                                                               
 - Use GaussianNetworkModelTorch for phonon computation                                                                                                                           
 - Keep molecular transform computation in numpy initially                                                                                                                        
 - Convert key matrices to torch tensors                                                                                                                                          
 - Implement apply_disorder() using torch operations                                                                                                                              
                                                                                                                                                                                  
 ### Testing Strategy                                                                                                                                                             
 - Create parallel test files for torch implementations                                                                                                                           
 - Ensure exact numerical match with numpy versions                                                                                                                               
 - Test both CPU and GPU paths                                                                                                                                                    
 - Validate edge cases and error conditions                                                                                                                                       
 - Test gradient computation where applicable                                                                                                                                     
                                                                                                                                                                                  
 ### Logging Requirements                                                                                                                                                         
 - Log tensor shapes and devices                                                                                                                                                  
 - Track memory usage                                                                                                                                                             
 - Record computation times                                                                                                                                                       
 - Enable detailed debugging                                                                                                                                                      
 - Compare against numpy results                                                                                                                                                  
