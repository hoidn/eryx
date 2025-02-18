import numpy as np
import gemmi

def generate_grid(A_inv, hsampling, ksampling, lsampling, return_hkl=False):
    """
    Generate a grid of q-vectors based on the desired extents 
    and spacing in hkl space.
    
    Parameters
    ----------
    A_inv : numpy.ndarray, shape (3,3)
        fractional cell orthogonalization matrix
    hsampling : tuple, shape (3,)
        (hmin, hmax, oversampling relative to Miller indices)
    ksampling : tuple, shape (3,)
        (kmin, kmax, oversampling relative to Miller indices)
    lsampling : tuple, shape (3,)
        (lmin, lmax, oversampling relative to Miller indices)
    return_hkl : bool
        if True, return hkl indices rather than q-vectors
    
    Returns
    -------
    q_grid or hkl_grid : numpy.ndarray, shape (n_points, 3)
        grid of q-vectors or hkl indices
    map_shape : tuple, shape (3,)
        shape of 3d map
    """
    hsteps = int(hsampling[2]*(hsampling[1]-hsampling[0])+1)
    ksteps = int(ksampling[2]*(ksampling[1]-ksampling[0])+1)
    lsteps = int(lsampling[2]*(lsampling[1]-lsampling[0])+1)
    
    hkl_grid = np.mgrid[lsampling[0]:lsampling[1]:lsteps*1j,
                        ksampling[0]:ksampling[1]:ksteps*1j,
                        hsampling[0]:hsampling[1]:hsteps*1j]
    map_shape = hkl_grid.shape[1:][::-1]
    hkl_grid = hkl_grid.T.reshape(-1,3)
    hkl_grid = hkl_grid[:, [2,1,0]]
    
    if return_hkl:
        return hkl_grid, map_shape
    else:
        q_grid = 2*np.pi*np.inner(A_inv.T, hkl_grid).T
        return q_grid, map_shape

def get_symmetry_equivalents(hkl_grid, sym_ops):
    hkl_list = []
    for key, op in sym_ops.items():
        print(f"DEBUG: Processing symmetry op key {key}, op.shape: {op.shape}")
        # Compute the rotated grid and force 2D shape
        hkl_grid_rot = np.dot(hkl_grid, op.T)
        hkl_grid_rot = np.atleast_2d(hkl_grid_rot)
        print(f"DEBUG: For key {key}, hkl_grid_rot.shape: {hkl_grid_rot.shape} (expected: (n_points, 3))")
        hkl_list.append(hkl_grid_rot)

    stacked = np.vstack(hkl_list)
    print(f"DEBUG: Final stacked symmetry grid shape: {stacked.shape}")
    return stacked
    
def get_ravel_indices(hkl_grid_sym, sampling):
    """
    Map 3d hkl indices to corresponding 1d indices after raveling.
    
    Parameters
    ----------
    hkl_grid_sym : numpy.ndarray, shape (n_asu, n_points, 3)
        stacked hkl indices of symmetry-equivalents
    sampling : tuple, shape (3,)
        sampling rate relative to integral Millers along (h,k,l)
    
    Returns
    -------
    ravel : numpy.ndarray, shape (n_asu, n_points)
        indices in raveled space for hkl_grid_sym
    map_shape_ravel : tuple, shape (3,)  
        shape of expanded / raveled map
    """
    hkl_grid_stacked = hkl_grid_sym.reshape(-1, hkl_grid_sym.shape[-1])
    hkl_grid_int = np.around(hkl_grid_stacked * np.array(sampling)).astype(int)
    lbounds = np.min(hkl_grid_int, axis=0)
    ubounds = np.max(hkl_grid_int, axis=0)
    map_shape_ravel = tuple((ubounds - lbounds + 1)) 
    hkl_grid_int = hkl_grid_int.reshape(hkl_grid_sym.shape)
    
    ravel = np.zeros(hkl_grid_sym.shape[:2]).astype(int)
    for i in range(ravel.shape[0]):
        ravel[i] = np.ravel_multi_index((hkl_grid_int[i] - lbounds).T, map_shape_ravel)

    return ravel, map_shape_ravel

def cos_sq(angles):
    """ Compute cosine squared of input angles in radians. """
    return np.square(np.cos(angles))

def sin_sq(angles):
    """ Compute sine squared of input angles in radianss. """
    return np.square(np.sin(angles))

def compute_resolution(cell, hkl):
    """
    Compute reflections' resolution in 1/Angstrom. To check, see: 
    https://www.ruppweb.org/new_comp/reciprocal_cell.htm.
        
    Parameters
    ----------
    cell : numpy.ndarray, shape (6,)
        unit cell parameters (a,b,c,alpha,beta,gamma) in Ang/deg
    hkl : numpy.ndarray, shape (n_refl, 3)
        Miller indices of reflections
            
    Returns
    -------
    resolution : numpy.ndarray, shape (n_refl)
        resolution associated with each reflection in Angstrom
    """

    a,b,c = [cell[i] for i in range(3)] 
    alpha,beta,gamma = [np.radians(cell[i]) for i in range(3,6)] 
    h,k,l = [hkl[:,i] for i in range(3)]

    pf = 1.0 - cos_sq(alpha) - cos_sq(beta) - cos_sq(gamma) + 2.0*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
    n1 = np.square(h)*sin_sq(alpha)/np.square(a) + np.square(k)*sin_sq(beta)/np.square(b) + np.square(l)*sin_sq(gamma)/np.square(c)
    n2a = 2.0*k*l*(np.cos(beta)*np.cos(gamma) - np.cos(alpha))/(b*c)
    n2b = 2.0*l*h*(np.cos(gamma)*np.cos(alpha) - np.cos(beta))/(c*a)
    n2c = 2.0*h*k*(np.cos(alpha)*np.cos(beta) - np.cos(gamma))/(a*b)

    with np.errstate(divide='ignore'):
        return 1.0 / np.sqrt((n1 + n2a + n2b + n2c) / pf)

def get_hkl_extents(cell, resolution, oversampling=1):
    """
    Determine the min/max hkl for the given cell and resolution.
    
    Parameters
    ----------
    cell : numpy.ndarray, shape (6,)
        unit cell parameters in Angstrom / degrees
    resolution : float
        high-resolution limit
    oversampling : int or tuple of shape (3,)
        oversampling rate relative to integral Miller indices
    
    Returns
    -------
    hsampling : tuple, shape (3,)
        (min, max, interval) along h axis
    ksampling : tuple, shape (3,)
        (min, max, interval) along k axis
    lsampling : tuple, shape (3,)
        (min, max, interval) along l axis        
    """
    if type(oversampling) == int:
        oversampling = 3 * [oversampling]
    g_cell = gemmi.UnitCell(*cell)
    h,k,l = g_cell.get_hkl_limits(resolution)
    return (-h,h,oversampling[0]), (-k,k,oversampling[1]), (-l,l,oversampling[2])

def expand_sym_ops(sym_ops):
    """
    Expand symmetry operations to include Friedel equivalents.

    Parameters
    ----------
    sym_ops : dict or tuple/list of dict
        rotational symmetry operations as 3x3 matrices.
        If a tuple or nested dict is provided, only the raw matrices are used.
    
    Returns
    -------
    sym_ops_exp : dict
        The input symmetry operations, plus their negative (Friedel) counterparts.
    """
    # If sym_ops is a tuple or list, use its first element.
    if isinstance(sym_ops, (tuple, list)):
        sym_ops = sym_ops[0]
    # If any value in sym_ops is a dict, merge them into one flat dict.
    if any(isinstance(val, dict) for val in sym_ops.values()):
        merged = {}
        for sub in sym_ops.values():
            if isinstance(sub, dict):
                merged.update(sub)
            else:
                # If a non-dict value is encountered, add it with a new key.
                merged[len(merged)] = sub
        sym_ops = merged
    sym_ops_exp = dict(sym_ops)
    n = len(sym_ops)
    for key, op in sym_ops.items():
        # For each op, op should be a numpy array; if not, skip expansion for that key.
        if not isinstance(op, np.ndarray):
            continue
        sym_ops_exp[key + n] = -1 * op
    return sym_ops_exp

def compute_multiplicity(model, hsampling, ksampling, lsampling):
    """
    Compute the multiplicity of each voxel in the map.
    
    Parameters
    ----------
    model : AtomicModel 
        instance of AtomicModel class
    hsampling : tuple, shape (3,)
        (min, max, interval) along h axis
    ksampling : tuple, shape (3,)
        (min, max, interval) along k axis
    lsampling : tuple, shape (3,)
        (min, max, interval) along l axis        
    
    Returns
    -------
    hkl_grid : numpy.ndarray, shape (n_points, 3)
        grid of hkl vectors
    multiplicity : numpy.ndarray, 3d
        multiplicity of each grid point in the map
    """
    sym_ops_exp = expand_sym_ops(model.sym_ops)
    hkl_grid, map_shape = generate_grid(model.A_inv, hsampling, ksampling, lsampling, return_hkl=True)
    # Filter to keep only symmetry operators whose dot–product output would have the same number of columns as hkl_grid.
    valid_sym_ops = { key: op for key, op in sym_ops_exp.items() if op.shape[1] == hkl_grid.shape[1] }
    if not valid_sym_ops:
        raise ValueError("No symmetry operators match the grid dimensions.")
    hkl_sym = get_symmetry_equivalents(hkl_grid, valid_sym_ops)
    ravel, map_shape_ravel = get_ravel_indices(hkl_sym, (hsampling[2], ksampling[2], lsampling[2]))
    multiplicity = (np.diff(np.sort(ravel.T,axis=1),axis=1)!=0).sum(axis=1)+1
    return hkl_grid, multiplicity.reshape(map_shape)

def parse_asu_condition(asu_condition):
    """
    Parse Gemmi's string describing which reflections belong to the 
    asymmetric unit into a string compatible with python's eval.
    
    Parameters
    ----------
    asu_condition : str
        Gemmi-style string describing asu
        
    Returns
    -------
    asu_condition : str
        eval-compatible string describing asu
    """
    # convert to numpy boolean operators
    find = ["=", "and", "or", ">==", "<=="] 
    replace = ["==", "&", "|", ">=", "<="]
    for i,j in zip(find, replace):
        asu_condition = asu_condition.replace(i,j)
        
    # add missing parenthesis around individual conditions
    alphas = [idx for idx in range(len(asu_condition)) if (asu_condition[idx].isalpha() and asu_condition[idx+1] not in [" ", ")", "("])]
    counter = 0
    for start in alphas:
        asu_condition = asu_condition[:start+counter] + "(" + asu_condition[start+counter:]
        counter += 1
        end = asu_condition.find(" ", start+counter)
        if end == -1:
            asu_condition = asu_condition + ")"
        else:
            asu_condition = asu_condition[:end] + ")" + asu_condition[end:]
        counter += 1
    
    return asu_condition

def get_asu_mask(space_group, hkl_grid):
    """
    Generate a boolean mask that indicates which hkl indices belong
    to the asymmetric unit.
    
    Parameters
    ----------
    space_group : int or str
        crystal's space group
    hkl_grid : numpy.ndarray, shape (n_points, 3)
        hkl indices 
        
    Returns
    -------
    asu_mask : numpy.ndarray, shape (n_points,)
        True indicates hkls that belong to the asymmetric unit
    """
    sg = gemmi.SpaceGroup(space_group)
    asu_condition = gemmi.ReciprocalAsu(sg).condition_str()
    asu_condition = parse_asu_condition(asu_condition)
    h, k, l = [hkl_grid[:,i] for i in range(3)]
    asu_mask = eval(asu_condition)
    return asu_mask

def get_resolution_mask(cell, hkl_grid, res_limit):
    """
    Generate a boolean mask that indicates which hkl indices belong
    to the asymmetric unit.
    
    Parameters
    ----------
    space_group : int or str
        crystal's space group
    hkl_grid : numpy.ndarray, shape (n_points, 3)
        hkl indices 
    res_limit : float
        high resolution limit in Angstrom
        
    Returns
    -------
    res_mask : numpy.ndarray, shape (n_points,)
        True indicates hkls that are within high resolution limit
    res_map : numpy.ndarray, shape (n_points,)
        resolution in Angstrom for each point in the grid
    """
    res_map = compute_resolution(cell, hkl_grid)
    res_mask = res_map > res_limit
    return res_mask, res_map

def get_dq_map(A_inv, hkl_grid):
    """
    Compute dq, the distance to the nearest Bragg peak, for 
    each reciprocal grid point.
    
    Parameters
    ----------
    A_inv : numpy.ndarray, shape (3,3)
        fractional cell orthogonalization matrix
    hkl_grid : numpy.ndarray, shape (n_points, 3)
        grid of hkl indices
    
    Returns
    -------
    dq : numpy.ndarray, shape (n_points,)
        distance to the nearest Bragg peak
    """
    hkl_closest = np.around(hkl_grid)
    q_closest = 2*np.pi*np.inner(A_inv.T, hkl_closest).T 
    q_grid = 2*np.pi*np.inner(A_inv.T, hkl_grid).T 
    dq = np.linalg.norm(np.abs(q_closest - q_grid), axis=1)
    return np.around(dq, decimals=8)

def get_centered_sampling(map_shape, sampling):
    """
    Get the hsampling, ksampling, and lsampling tuples for the input 
    map shape, assuming that the map is centered about the origin in 
    reciprocal space, i.e. h,k,l=(0,0,0).
    
    Parameters
    ----------
    map_shape : tuple
        map dimensions
    sampling : tuple
        fractional sampling rate along h,k,l axes
    
    Returns
    -------
    list of tuples, [hsampling, lsampling, ksampling]
        in which each tuple corresponds to (min, max, fractional sampling rate)
    """
    extents = [((map_shape[i]-1) / sampling[i] / 2.0) for i in range(3)]
    return [(-extents[i], extents[i], sampling[i]) for i in range(3)]

def resize_map(new_map, old_sampling, new_sampling):
    """
    Resize map if symmetrization has resulted in the inclusion of 
    out-of-bounds regions, but not those valid by Friedel's law.
    
    Parameters
    ----------
    new_map : numpy.ndarray, 3d
        map to potentially crop
    old_sampling : tuple of tuples
        (hsampling, ksampling, lsampling) for original hkl_grid
    new_sampling : tuple of tuples 
        (hsampling, ksampling, lsampling) for new_map
        
    Returns
    -------
    new_map : numpy.ndarray, 3d
        potentially cropped map
    """
    tol = 1e-6
    if np.abs(new_sampling[0][1] - old_sampling[0][1]) > tol:
        excise = int(np.around(2*(new_sampling[0][1] - old_sampling[0][1])))
        new_map = new_map[excise:-excise,:,:]
    if np.abs(new_sampling[1][1] - old_sampling[1][1]) > tol:
        excise = int(np.around(2*(new_sampling[1][1] - old_sampling[1][1])))
        new_map = new_map[:,excise:-excise,:]
    if np.abs(new_sampling[2][1] - old_sampling[2][1]) > tol:
        excise = int(np.around(2*(new_sampling[2][1] - old_sampling[2][1])))
        new_map = new_map[:,:,excise:-excise]
    return new_map
