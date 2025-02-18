do map_utils.py, scatter.py, or pdb.py have to be converted to pytorch or are they either (1) only relevant to loading / preprocessing or (2) not used by the onephonon model?if yes, which components / functions in these modules have to be ported? (list them all)EditLet me analyze the dependencies by tracing the OnePhonon model's usage of these modules:
map_utils.py:

Only used for initial grid setup and preprocessing
All functions can remain in numpy
Not involved in core computations

scatter.py:
NEEDS CONVERSION:

structure_factors() - Called within OnePhonon.apply_disorder() for computing diffuse scattering
compute_form_factors() - Used by structure_factors()

pdb.py:
NEEDS CONVERSION:

GaussianNetworkModel class - This is the core dependency for OnePhonon model, specifically these methods:

compute_hessian()
compute_K()
compute_Kinv()
build_neighbor_list()



Everything else in pdb.py can remain numpy since it's used for:

Loading PDB files
Initial coordinate processing
Symmetry operations
Unit cell setup
These all happen during initialization/preprocessing

So to summarize, we need to convert:

From scatter.py:

pythonCopydef compute_form_factors(q_grid, ff_a, ff_b, ff_c)
def structure_factors(q_grid, xyz, ff_a, ff_b, ff_c, U=None, ...)

From pdb.py:

pythonCopyclass GaussianNetworkModel:
    def compute_hessian(self)
    def compute_K(self, hessian, kvec=None)
    def compute_Kinv(self, hessian, kvec=None)
    def build_neighbor_list(self)
The rest of the functions in these modules can stay as numpy since they're only used for loading, setup, and preprocessing.
Would you like me to create a detailed spec for converting just these specific components?

## Reference
https://claude.ai/chat/ac9f83be-3d46-4be1-bbcc-baf915a4d89a 

