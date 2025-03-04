# Removed Components Documentation

This document lists all components that were removed from the eryx codebase as part of the streamlining process for the PyTorch port.

## Rationale for Removal

The components were removed because they are not part of the critical execution path for the PyTorch port as defined in `to_convert.json` and `call_chains.json`. By removing these components, we can focus on the core functionality needed for the port and reduce maintenance overhead.

## Removed Model Classes

The following model classes were removed from `eryx/models.py`:

1. `RigidBodyTranslations` - Model for rigid body translational disorder
2. `LiquidLikeMotions` - Model for collective motions that decay exponentially with distance
3. `RigidBodyRotations` - Model for rigid body rotational disorder
4. `Ensemble` - Model for ensemble disorder with distinct biological states
5. `NonInteractingDeformableMolecules` - Lattice model with non-interacting deformable molecules
6. `OnePhononBrillouin` - Specialized version of OnePhonon for Brillouin zone analysis

## Removed Base Module Functions

The following functions were removed from `eryx/base.py`:

1. `natural_sort` - Helper function for sorting strings naturally
2. `guinier_reconstruct` - Function to reconstruct a map from an ensemble of complex structure factors
3. `compute_crystal_transform` - Function to compute the crystal transform
4. `compute_molecular_transform` - Function to compute the molecular transform
5. `incoherent_sum_real` - Helper function for computing incoherent sum in real space
6. `incoherent_sum_reciprocal` - Helper function for computing incoherent sum in reciprocal space

## Removed Files

1. `eryx/reference.py` - Contained reference implementations that duplicated functionality
2. `eryx/scripts/parse_yaml.py` - Utility script not used in the core execution path
3. `eryx/scripts/main.py` - Utility script not used in the core execution path
4. `eryx/scripts/tasks.py` - Utility script not used in the core execution path

## Updated Imports

The `eryx/__init__.py` file was updated to remove imports and exports of the removed components.

## Restoring Functionality

If you need to restore any of the removed functionality:

1. Check the backup directory created before the removal: `backup/eryx/`
2. Copy the relevant files or components back into the main codebase
3. Update imports in `eryx/__init__.py` to re-export the restored components

## References

- `to_convert.json` - Definitive scope document for the PyTorch port
- `call_chains.json` - Documentation of execution paths for components
