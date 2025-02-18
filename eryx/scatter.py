import numpy as np
import logging
import multiprocess as mp
from functools import partial

def compute_form_factors(q_grid, ff_a, ff_b, ff_c):
    """
    Evaluate atomic form factors at the input q-vectors.
    
    Parameters
    ----------
    q_grid : numpy.ndarray, shape (n_points, 3)
        q-vectors in Angstrom
    ff_a : numpy.ndarray, shape (n_atoms, 4)
        a coefficient of atomic form factors
    ff_b : numpy.ndarray, shape (n_atoms, 4)
        b coefficient of atomic form factors
    ff_c : numpy.ndarray, shape (n_atoms,)
        c coefficient of atomic form factors

    Returns
    -------
    fj : numpy.ndarray, shape (n_points, n_atoms)
        atomic form factors 
    """
    print(f"[DEBUG] compute_form_factors: q_grid.shape = {q_grid.shape}")
    print(f"[DEBUG] compute_form_factors: ff_a.shape = {ff_a.shape}")
    print(f"[DEBUG] compute_form_factors: ff_b.shape = {ff_b.shape}")
    print(f"[DEBUG] compute_form_factors: ff_c.shape = {ff_c.shape}")
    
    Q = np.square(np.linalg.norm(q_grid, axis=1) / (4*np.pi))
    print("[DEBUG] ff_a_original shape:", ff_a.shape)  # e.g., (4, 28, 4)
    print("[DEBUG] ff_b_original shape:", ff_b.shape)  # e.g., (4, 28, 4)
    print("[DEBUG] Q shape:", Q.shape)                 # e.g., (18585,)
    temp_a = ff_a[:, :, np.newaxis]
    temp_b = ff_b[:, :, None]
    print("[DEBUG] temp_a shape (ff_a[:,:,np.newaxis]):", temp_a.shape)  # Expecting (4, 28, 1, 4)
    print("[DEBUG] temp_b shape (ff_b[:,:,None]):", temp_b.shape)      # Expecting (4, 28, 1, 4)
    temp_Q = Q[:, np.newaxis].T
    print("[DEBUG] temp_Q shape (Q[:,np.newaxis].T):", temp_Q.shape)     # Expecting (1, 18585)
    print(f"[DEBUG] compute_form_factors: Q.shape = {Q.shape}")
    exp_term = np.exp(-1 * temp_b * temp_Q)
    print("[DEBUG] exp_term shape:", exp_term.shape)
    fj = temp_a * exp_term
    print(f"[DEBUG] compute_form_factors: intermediate fj.shape = {fj.shape}")
    fj = np.sum(fj, axis=1) + ff_c[:,np.newaxis]
    mean_fj = np.mean(fj)
    logging.debug(f"[TestReference] Mean of form factors: {mean_fj:.8f}")
    return fj.reshape(ff_a.shape[0]*ff_a.shape[1], -1).T

def structure_factors_batch(q_grid, xyz, ff_a, ff_b, ff_c, U=None,
                            compute_qF=False, project_on_components=None,
                            sum_over_atoms=True):
    """
    Compute the structure factors for an atomic model at 
    the given q-vectors. 

    Parameters
    ----------
    q_grid : numpy.ndarray, shape (n_points, 3)
        q-vectors in Angstrom
    xyz : numpy.ndarray, shape (n_atoms, 3)
        atomic xyz positions in Angstroms
    ff_a : numpy.ndarray, shape (n_atoms, 4)
        a coefficient of atomic form factors
    ff_b : numpy.ndarray, shape (n_atoms, 4)
        b coefficient of atomic form factors
    ff_c : numpy.ndarray, shape (n_atoms,)
        c coefficient of atomic form factors
    U : numpy.ndarray, shape (n_atoms,) 
        isotropic displacement parameters
    compute_qF : boolean
        False (default).
        If true, return structure factors at q-vectors times q-vectors
    project_on_components : None (default) or numpy.ndarray, shape (n_atoms, n_components)
        Projection matrix to convert structure factors into component factors
    sum_over_atoms: boolean
        True (default) returns summed structure factor.
        
    Returns
    -------
    A : numpy.ndarray, shape (n_points) or (n_points, n_atoms) or (n_points, n_components)
        structure factors at q-vectors.
        If compute_qF, return [q_x A(q), q_y A(q), q_z A(q)] instead of A(q)
    """
    if U is None:
        U = np.zeros(xyz.shape[0])
    
    fj = compute_form_factors(q_grid, ff_a, ff_b, ff_c)
    qmags = np.linalg.norm(q_grid, axis=1)
    qUq = np.square(qmags[:,np.newaxis]) * U
    
    A = 1j * fj * np.sin(np.dot(q_grid, xyz.T)) * np.exp(-0.5 * qUq)
    A += fj * np.cos(np.dot(q_grid, xyz.T)) * np.exp(-0.5 * qUq)
    if compute_qF:
        A = A[:,:,None] * q_grid[:,None,:]
        A = A.reshape((A.shape[0],A.shape[1]*A.shape[2]))
    if project_on_components is not None:
        A = np.matmul(A, project_on_components)
    if sum_over_atoms:
        A = np.sum(A, axis=1)
    return A 

def structure_factors(q_grid, xyz, ff_a, ff_b, ff_c, U=None,
                      batch_size=100000, n_processes=8,
                      compute_qF=False, project_on_components=None,
                      sum_over_atoms=True):
    """
    Batched version of the structure factor calculation. See 
    docstring for structure_factors_batch for parameters and 
    returns, with the exception of n_processes, which refers 
    to the number of processors available. If greater than 1,
    multiprocessing will be used.
    """
    n_batches = q_grid.shape[0] // batch_size
    if n_batches == 0:
        n_batches = 1
    splits = np.append(np.arange(n_batches) * batch_size, np.array([q_grid.shape[0]]))

    if n_processes == 1:
        dim1_size = q_grid.shape[0]
        if sum_over_atoms:
            A_shape = dim1_size
        else:
            if project_on_components is None:
                dim2_size = xyz.shape[0]
            else:
                dim2_size = project_on_components.shape[1]
            A_shape = (dim1_size, dim2_size)
        A = np.zeros(A_shape, dtype=np.complex128)
        for batch in range(n_batches):
            q_sel = q_grid[splits[batch]: splits[batch+1]]
            A[splits[batch]: splits[batch+1]] = structure_factors_batch(q_sel, xyz, ff_a, ff_b, ff_c, U=U, compute_qF=compute_qF,
                                                                        project_on_components=project_on_components, sum_over_atoms=sum_over_atoms)
    else:
        q_sel = [q_grid[splits[batch]: splits[batch+1]] for batch in range(n_batches)]
        pool = mp.Pool(processes=n_processes)
        sf_partial = partial(structure_factors_batch, xyz=xyz, ff_a=ff_a, ff_b=ff_b, ff_c=ff_c, U=U,
                             compute_qF=compute_qF,
                             project_on_components=project_on_components, sum_over_atoms=sum_over_atoms)
        A = np.concatenate(pool.map(sf_partial, q_sel), axis=0)
       
    first_element = A[0] if A.size > 0 else None
    logging.debug(f"[TestReference] First element of structure factors: {first_element}")
    return A
