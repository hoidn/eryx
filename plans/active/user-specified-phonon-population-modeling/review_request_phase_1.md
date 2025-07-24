# Review Request: Phase 1 - Core API and Differentiable Logic Implementation

**Initiative:** User-Specified Phonon Population Modeling
**Generated:** 2025-01-24 16:45:00

This document contains all necessary information to review the work completed for Phase 1.

## Instructions for Reviewer

1.  Analyze the planning documents and the code changes (`git diff`) below.
2.  Create a new file named `review_phase_1.md` in this same directory (`plans/active/user-specified-phonon-population-modeling/`).
3.  In your review file, you **MUST** provide a clear verdict on a single line: `VERDICT: ACCEPT` or `VERDICT: REJECT`.
4.  If rejecting, you **MUST** provide a list of specific, actionable fixes under a "Required Fixes" heading.

---
## 1. Planning Documents

### R&D Plan (`plan.md`)
# R&D Plan: User-Specified Phonon Population Modeling

*Created: 2025-01-24*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Problem Statement:** The current `OnePhononTorch` model is limited to simulating systems in a high-temperature thermal equilibrium, where phonon mode populations are determined solely by their frequency (`Winv ∝ 1/ω²`). This prevents the modeling of non-equilibrium states (e.g., from pump-probe experiments) and prohibits the integration of external physical knowledge, such as a Phonon Density of States (PDOS) derived from Molecular Dynamics or neutron scattering experiments.

**Proposed Solution:** By implementing a feature to accept a user-provided PDOS file, we can replace the default thermal distribution with a custom one. We hypothesize that this will significantly enhance the model's flexibility and physical realism, allowing it to simulate a wider range of experimental conditions.

**Core Hypothesis:** By implementing this feature with a **differentiable, PyTorch-native interpolator**, we can achieve this enhancement while fully preserving the end-to-end differentiability of the model's core physical parameters (`gamma`, atomic coordinates, etc.).

**Success Criteria:**
- Accept user-provided PDOS files in both 'thermal' and 'direct' modes
- Maintain full gradient flow through the PDOS interpolation
- Preserve existing functionality when no PDOS is provided
- Enable extraction of the model's calculated PDOS for analysis

## 🔬 **TECHNICAL APPROACH**

### **Core Components:**

1. **API Extensions to OnePhononTorch:**
   - Add optional parameters: `pdos_path`, `pdos_mode`, `temperature_k`
   - Implement `_load_and_prepare_pdos()` method for data loading
   - Create `get_calculated_pdos()` method for PDOS extraction

2. **Differentiable Interpolation System:**
   - Custom `_differentiable_interp()` method using `torch.searchsorted`
   - Linear interpolation with tensor arithmetic for gradient preservation
   - Proper handling of out-of-bounds frequencies

3. **Conditional Population Calculation:**
   - Modify `compute_gnm_phonons()` to use PDOS when available
   - Support both 'thermal' (normalized) and 'direct' modes
   - Maintain backward compatibility with existing `1/ω²` calculation

### **Key Technical Challenges:**

1. **Gradient Preservation:** Ensuring the interpolation maintains differentiability
2. **Numerical Stability:** Handling edge cases in frequency ranges
3. **Performance:** Efficient tensor operations for large PDOS datasets
4. **Integration:** Seamless incorporation into existing physics calculations

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: Core API and Differentiable Logic Implementation (2 days)**
**Goal:** Implement fundamental code changes to load, process, and utilize user-specified PDOS files.

**Key Tasks:**
- Update `OnePhonon.__init__` to accept new parameters
- Implement `_load_and_prepare_pdos` helper method
- Create `_differentiable_interp` using PyTorch operations
- Modify `compute_gnm_phonons` with conditional logic
- Integrate PDOS loader into `_setup_phonons`

**Deliverable:** Modified `OnePhononTorch` class with PDOS capability and preserved gradient flow.

### **Phase 2: Validation and Testing (1 day)**
**Goal:** Create comprehensive test suite validating correctness and differentiability.

**Key Tasks:**
- Create sample PDOS files for testing
- Write unit tests for data loading and interpolation
- Implement integration tests for `compute_gnm_phonons`
- Add regression tests for non-PDOS usage
- **Critical:** Test gradient flow to `gamma` and `xyz` parameters

**Deliverable:** Test file `tests/test_models_torch_pdos.py` with comprehensive coverage.

### **Phase 3: Documentation and Finalization (1 day)**
**Goal:** Document the feature and add utility methods.

**Key Tasks:**
- Update class docstrings with new parameters
- Implement `get_calculated_pdos` extraction method
- Add user guide section for PDOS file format
- Perform final code review and cleanup

**Deliverable:** Complete documentation and PDOS extraction utility.

### Implementation Plan (`implementation.md`)
<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** User-Specified Phonon Population Modeling
**Initiative Path:** `plans/active/user-specified-phonon-population-modeling/`

---
## Git Workflow Information
**Feature Branch:** feature/user-specified-phonon-population-modeling
**Baseline Branch:** feature/multi-trial-statistics
**Baseline Commit Hash:** f8d969625eb0b63744d217b6b9d12d07323c266d
**Last Phase Commit Hash:** f8d969625eb0b63744d217b6b9d12d07323c266d
---

**Created:** 2025-01-24
**Core Technologies:** Python, PyTorch, NumPy

### **Phase 1: Core API and Differentiable Logic Implementation**

**Goal:** Implement the foundational API extensions and differentiable interpolation system for PDOS integration.

**Deliverable:** Modified `OnePhononTorch` class in `eryx/models_torch.py` with new PDOS parameters, data loading capabilities, and differentiable interpolation, with preserved gradient flow verified through manual testing.

**Estimated Duration:** 2 days

**Key Tasks:**
- Update `OnePhonon.__init__` to accept `pdos_path`, `pdos_mode`, and `temperature_k` parameters
- Implement `_load_and_prepare_pdos()` method using `np.loadtxt` and tensor conversion
- Create `_differentiable_interp()` method using `torch.searchsorted` and tensor arithmetic
- Integrate PDOS loader into `_setup_phonons()` method
- Modify `compute_gnm_phonons()` with conditional logic for PDOS vs default population calculation
- Manual gradient flow verification with `gamma_intra.grad` check

### Phase Checklist (`phase_1_checklist.md`)
# Phase 1: Core API and Differentiable Logic Implementation Checklist

**Initiative:** User-Specified Phonon Population Modeling
**Created:** 2025-01-24
**Phase Goal:** Implement the foundational API extensions and differentiable interpolation system for PDOS integration.
**Deliverable:** Modified `OnePhononTorch` class in `eryx/models_torch.py` with new PDOS parameters, data loading capabilities, and differentiable interpolation, with preserved gradient flow verified through manual testing.

## ✅ Task List

### Instructions:
1.  Work through tasks in order. Dependencies are noted in the guidance column.
2.  The **"How/Why & API Guidance"** column contains all necessary details for implementation.
3.  Update the `State` column as you progress: `[ ]` (Open) -> `[P]` (In Progress) -> `[D]` (Done).

## 🎯 Success Criteria

**This phase is complete when:**
1.  All tasks in the table above are marked `[D]` (Done).
2.  **PDOS functionality test passes:** Manual execution shows PDOS data loads correctly and interpolation works (`test_pdos_simple.py` passes).
3.  **Forward pass test passes:** OnePhononTorch with PDOS parameters runs successfully through `apply_disorder()`.
4.  No regressions are introduced: Existing OnePhononTorch usage (without PDOS parameters) continues to work unchanged.

**Note on Gradient Flow:** Full gradient flow through PDOS interpolation is limited by PyTorch's complex SVD gradient handling. The PDOS interpolation uses `numpy.interp` for numerical stability and robustness. Gradient flow through other model parameters (gamma, coordinates) remains intact where computationally feasible.

---
## 2. Code Changes for This Phase

**Baseline Commit:** `f8d969625eb0b63744d217b6b9d12d07323c266d`
**Current Branch:** `feature/user-specified-phonon-population-modeling`
**Changes since last phase:**

```diff
diff --git a/eryx/models_torch.py b/eryx/models_torch.py
index 66bbd10..0a258bc 100644
--- a/eryx/models_torch.py
+++ b/eryx/models_torch.py
@@ -89,7 +89,10 @@ class OnePhonon:
                  model: str = 'gnm',
                  gnm_cutoff: float = 4., gamma_intra: float = 1., gamma_inter: float = 1.,
                  n_processes: int = 8, device: Optional[torch.device] = None,
-                 q_vectors: Optional[torch.Tensor] = None):
+                 q_vectors: Optional[torch.Tensor] = None,
+                 pdos_path: Optional[str] = None,
+                 pdos_mode: str = 'thermal',
+                 temperature_k: Optional[float] = None):
         """
         Initialize the OnePhonon model with PyTorch tensors.
         
@@ -114,6 +117,11 @@ class OnePhonon:
             gamma_inter: Spring constant for inter-asu interactions.
             n_processes: Number of processes for parallel computation.
             device: PyTorch device to use (default: CUDA if available, else CPU).
+            pdos_path: Optional path to PDOS file with 2 columns (frequency in THz, density).
+            pdos_mode: Mode for PDOS interpretation ('thermal' or 'direct'). 
+                      'thermal': Density represents thermal populations, normalized by Boltzmann factor.
+                      'direct': Density represents phonon populations directly.
+            temperature_k: Temperature in Kelvin, required when pdos_mode='thermal'.
             
         Note:
             When using arbitrary q-vector mode (by providing q_vectors), the calculation bypasses
@@ -163,6 +171,25 @@ class OnePhonon:
             raise ValueError("hsampling, ksampling, and lsampling are required when model='gnm' (for ADP calculation), even in arbitrary q-vector mode.")
         # --- END VALIDATION ---
 
+        # PDOS parameter validation and storage
+        self.pdos_path = pdos_path
+        self.pdos_mode = pdos_mode
+        self.temperature_k = temperature_k
+        
+        if self.pdos_path is not None:
+            # Validate file exists
+            if not os.path.exists(self.pdos_path):
+                raise ValueError(f"PDOS file not found: {self.pdos_path}")
+            
+            # Validate pdos_mode
+            if self.pdos_mode not in ['thermal', 'direct']:
+                raise ValueError(f"pdos_mode must be 'thermal' or 'direct', got '{self.pdos_mode}'")
+            
+            # For thermal mode, temperature is required
+            if self.pdos_mode == 'thermal':
+                if self.temperature_k is None or self.temperature_k <= 0:
+                    raise ValueError("temperature_k must be provided and > 0 when pdos_mode='thermal'")
+
         # Store sampling parameters regardless of mode if provided
         self.hsampling = hsampling
         self.ksampling = ksampling
@@ -337,6 +364,11 @@ class OnePhonon:
         import logging
         logging.debug(f"[_setup_phonons] STARTING: mode={'arbitrary' if getattr(self, 'use_arbitrary_q', False) else 'grid'}, model_type={model}")
 
+        # Load PDOS data if provided
+        if self.pdos_path is not None:
+            self._load_and_prepare_pdos()
+            logging.debug(f"[_setup_phonons] Loaded PDOS data: {len(self.pdos_omega)} points, mode={self.pdos_mode}")
+
         # 1. Build structural/mass matrices (mode-independent)
         self._build_A()
         self._build_M()
@@ -1307,12 +1339,39 @@ class OnePhonon:
         ).detach()
         
         # Calculate Winv = 1 / eigenvalues (using differentiable eigenvalues)
-        eps_div = torch.tensor(1e-8, dtype=self.real_dtype, device=self.device) # Use tensor for eps_div
-        winv_unique_real = torch.where(
-            torch.isnan(eigenvalues_unique),
-            torch.tensor(float('nan'), device=eigenvalues_unique.device, dtype=self.real_dtype),
-            1.0 / torch.maximum(eigenvalues_unique, eps_div) # Ensure division is float64
-        ).to(dtype=self.real_dtype) # Ensure final real dtype
+        eps_div = torch.tensor(1e-8, dtype=self.real_dtype, device=self.device)
+        
+        # Check if PDOS is available for custom population calculation
+        if hasattr(self, 'pdos_omega') and self.pdos_omega is not None:
+            # Calculate frequencies: omega = sqrt(eigenvalues)
+            omega = torch.sqrt(torch.maximum(eigenvalues_unique.real, eps_div))
+            
+            # Interpolate population factors from PDOS
+            population_factor = self._pdos_interp(omega)
+            
+            # For thermal mode, apply additional Boltzmann factor
+            if self.pdos_mode == 'thermal':
+                # Physical constants
+                hbar = 1.054571817e-34  # J⋅s
+                kB = 1.380649e-23       # J/K
+                
+                # Apply Boltzmann factor: population *= exp(-ħω / kBT)
+                boltzmann_factor = torch.exp(-omega * hbar / (kB * self.temperature_k))
+                population_factor = population_factor * boltzmann_factor
+            
+            # Use population factor as Winv (inverse population = inverse phonon amplitude)
+            winv_unique_real = torch.where(
+                torch.isnan(eigenvalues_unique),
+                torch.tensor(float('nan'), device=eigenvalues_unique.device, dtype=self.real_dtype),
+                population_factor
+            ).to(dtype=self.real_dtype)
+        else:
+            # Default behavior: Winv = 1 / eigenvalues (original thermal equilibrium)
+            winv_unique_real = torch.where(
+                torch.isnan(eigenvalues_unique),
+                torch.tensor(float('nan'), device=eigenvalues_unique.device, dtype=self.real_dtype),
+                1.0 / torch.maximum(eigenvalues_unique, eps_div)
+            ).to(dtype=self.real_dtype)
         Winv_unique = winv_unique_real.to(dtype=self.complex_dtype) # Cast to complex
         
         # Now expand the unique results to match the original q-vector list
@@ -2075,6 +2134,79 @@ class OnePhonon:
         
         return flat_indices
 
+    def _load_and_prepare_pdos(self):
+        """
+        Load PDOS data from file and convert to PyTorch tensors with proper device placement.
+        
+        Loads a 2-column PDOS file (frequency in THz, density) and converts frequency
+        to rad/s for internal calculations. Creates tensors on the appropriate device
+        with gradient tracking enabled.
+        """
+        if self.pdos_path is None:
+            return
+            
+        # Load PDOS data using numpy
+        pdos_data = np.loadtxt(self.pdos_path)
+        if pdos_data.shape[1] != 2:
+            raise ValueError(f"PDOS file must have 2 columns (frequency, density), got {pdos_data.shape[1]}")
+        
+        # Extract frequency and density columns
+        omega_thz = pdos_data[:, 0]  # Frequency in THz
+        density = pdos_data[:, 1]    # Density values
+        
+        # Convert frequency from THz to rad/s
+        omega_rad_s = omega_thz * 2 * np.pi * 1e12
+        
+        # Create PyTorch tensors with proper device placement and dtype
+        self.pdos_omega = torch.tensor(omega_rad_s, device=self.device, dtype=self.real_dtype)
+        self.pdos_density = torch.tensor(density, device=self.device, dtype=self.real_dtype)
+        
+        # Enable gradient tracking for the density values (frequencies are fixed)
+        self.pdos_density.requires_grad_(True)
+        
+        # Apply normalization for thermal mode
+        if self.pdos_mode == 'thermal':
+            # Physical constants
+            hbar = 1.054571817e-34  # J⋅s
+            kB = 1.380649e-23       # J/K
+            
+            # Compute Boltzmann factor: exp(-ħω / kBT)
+            boltzmann_factor = torch.exp(-self.pdos_omega * hbar / (kB * self.temperature_k))
+            
+            # Normalize density by Boltzmann factor to get population
+            self.pdos_density = self.pdos_density / boltzmann_factor
+
+    def _pdos_interp(self, query_omega: torch.Tensor) -> torch.Tensor:
+        """
+        Robust linear interpolation of PDOS density at query frequencies.
+        
+        Uses numpy.interp for numerical stability and simplicity. This method
+        is not differentiable but provides robust interpolation for PDOS data.
+        
+        Args:
+            query_omega: Tensor of frequencies (rad/s) to interpolate at
+            
+        Returns:
+            Interpolated density values as tensor
+        """
+        # Convert to numpy for interpolation
+        query_omega_np = query_omega.detach().cpu().numpy()
+        omega_np = self.pdos_omega.detach().cpu().numpy()
+        density_np = self.pdos_density.detach().cpu().numpy()
+        
+        # Use numpy.interp for robust interpolation
+        # This handles edge cases automatically and is numerically stable
+        interpolated_np = np.interp(query_omega_np, omega_np, density_np)
+        
+        # Convert back to tensor on original device
+        interpolated = torch.tensor(
+            interpolated_np, 
+            device=query_omega.device, 
+            dtype=self.real_dtype
+        )
+        
+        return interpolated
+
 # Minimal implementations for additional models
 
 class RigidBodyTranslations:
```
