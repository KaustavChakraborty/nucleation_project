# Hard-Sphere HPMC Compression - Comprehensive Documentation

**Version:** 11 (Documented Edition)  
**HOOMD-blue Version:** ≥ 4.0 (tested with 4.9.0)  
**Author:** Kaustav Chakraborty  
**Date:** 2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Critical Assumptions](#3-critical-assumptions)
4. [HOOMD-blue Leverages](#4-hoomd-blue-leverages)
5. [Parameter Tuning Guide](#5-parameter-tuning-guide)
6. [Failure Modes & Troubleshooting](#6-failure-modes--troubleshooting)
7. [Installation & Dependencies](#7-installation--dependencies)
8. [Usage Examples](#8-usage-examples)
9. [File Formats](#9-file-formats)
10. [Algorithm Details](#10-algorithm-details)
11. [Physical Background](#11-physical-background)
12. [Limitations & Scope](#12-limitations--scope)
13. [Performance Considerations](#13-performance-considerations)
14. [Validation & Testing](#14-validation--testing)
15. [References](#15-references)

---

## 1. Executive Summary

This script performs **systematic compression of hard-sphere systems** from low-density initial configurations to target packing fractions using **HOOMD-blue v4's Hard Particle Monte Carlo (HPMC)** engine.

### What This Script Does

- **Input:** GSD file with hard spheres at low packing fraction (φ ~ 0.1-0.3)
- **Process:** Gradual box compression with overlap removal at each step
- **Output:** GSD file with hard spheres at target packing fraction (φ up to ~0.70)
- **Method:** Faithful port of Avisek Das's v2 manual loop methodology
- **Reproducibility:** Fixed random seed, deterministic compression path

### What Makes This Script Different

- **Exact v2 methodology:** Two-level loop (outer: compress box, inner: remove overlaps)
- **Fixed move size:** No adaptive tuner (user must manually select MC move size)
- **Step-by-step checkpointing:** Writes GSD + JSON after each compression step
- **Comprehensive error handling:** Validates all inputs, detects failure modes
- **Heavily documented:** 3000+ lines of inline comments explaining every step

### Quick Start

```bash
# Install dependencies
conda install -c conda-forge hoomd gsd numpy

# Run compression
python hard_sphere_compression.py --simulparam_file simulparam_hard_sphere_compression.json

# Check outputs
ls -lh *_final.gsd *_compressed_*.gsd *_summary.json
```

---

## 2. System Overview

### 2.1 System Requirements

**Hardware:**
- **CPU:** Any modern x86-64 or ARM64 processor
- **RAM:** ~1 GB per 10,000 particles (depends on system size)
- **GPU (optional):** NVIDIA GPU with CUDA Compute Capability ≥ 5.0
  - Recommended: RTX 3000/4000 series, A100, H100
  - Not required: Script falls back to CPU if GPU unavailable
- **Storage:** ~100 MB per simulation (GSD trajectory + checkpoints)

**Software:**
- **OS:** Linux, macOS, or Windows (with WSL2 for GPU)
- **Python:** ≥ 3.8 (tested with 3.9-3.12)
- **HOOMD-blue:** ≥ 4.0 (tested with 4.9.0)
- **GSD:** ≥ 3.0
- **NumPy:** ≥ 1.20
- **MPI (optional):** OpenMPI ≥ 4.0 or MPICH ≥ 3.3 for multi-node runs

### 2.2 Physics Model

**Particle Model: Monodisperse Hard Spheres**

- **Shape:** Perfect spheres with diameter `d` (uniform across all particles)
- **Potential:** Purely geometric (excluded volume)
  - Overlap (distance < d): Energy = ∞ (configuration invalid)
  - No overlap (distance ≥ d): Energy = 0 (configuration valid)
- **No attractive forces:** Particles do not stick together
- **No rotation:** Spheres are isotropic (orientation doesn't matter)
- **Temperature:** Not explicitly defined (hard-sphere limit: T → ∞)

**Box Model: Periodic Boundaries**

- **Topology:** 3D rectangular or triclinic box with periodic boundary conditions
- **Particle wrapping:** Particles crossing box boundaries wrap to opposite side
- **No walls:** System is infinite and homogeneous (no confinement)
- **Box changes:** Isotropic rescaling only (Lx, Ly, Lz scale by same factor)

**Monte Carlo Dynamics**

- **Algorithm:** Metropolis Monte Carlo with Hard Particle Move Check (HPMC)
- **Detailed balance:** HPMC satisfies detailed balance (equilibrium sampling)
- **Move types:** Translation only (no rotation for spheres)
- **Acceptance criterion:** 
  - Proposed move accepted if no overlaps created
  - Rejected if any overlaps detected
- **Time units:** "Timesteps" = number of MC sweeps (1 sweep = N trial moves)

---

## 3. Critical Assumptions

### 3.1 System Assumptions

**This script REQUIRES the following. Violating these will cause failure:**

#### ASSUMPTION 1: Monodisperse System (All particles identical diameter)

**What it means:**
- Every particle has the same diameter `d`
- No size distribution (σ_d = 0)
- Bidisperse, polydisperse, continuous distributions NOT supported

**Why it matters:**
- Script reads one diameter value from GSD, applies to all particles
- Packing fraction formula assumes uniform sphere volume
- Polydisperse systems require different algorithm (not implemented)

**How to verify:**
```python
import gsd.hoomd
with gsd.hoomd.open("input.gsd") as f:
    diameters = f[0].particles.diameter
    print(f"Unique diameters: {set(diameters)}")  # Should print single value
```

**Failure symptom:**
```
[FATAL ERROR] Multiple particle diameters detected in GSD file
              This script requires a monodisperse system
              Found 2 unique diameter values: 1.0, 1.2
```

#### ASSUMPTION 2: Single Particle Type (Type 'A' only)

**What it means:**
- All particles have typeid = 0 (HOOMD's type 'A')
- No multi-component systems (e.g., binary mixtures)
- HPMC integrator configured for type 'A' only

**Why it matters:**
- Script defines shape for `mc.shape["A"]` only
- Multiple types require separate shape definitions
- Type-specific move sizes not supported

**How to verify:**
```python
import gsd.hoomd
with gsd.hoomd.open("input.gsd") as f:
    typeids = f[0].particles.typeid
    print(f"Unique type IDs: {set(typeids)}")  # Should print {0}
```

#### ASSUMPTION 3: Hard-Sphere Potential (No soft interactions)

**What it means:**
- Particles interact ONLY via excluded volume
- No Lennard-Jones, WCA, Yukawa, or other soft potentials
- Configuration is valid IFF no overlaps (binary valid/invalid)

**Why it matters:**
- HPMC accepts/rejects moves based on overlap check only
- No energy minimization, no force calculations
- Soft potentials require MD integrator (hoomd.md), not HPMC

**Implications:**
- Cannot study systems with attractions (colloids, proteins)
- Cannot model sticky spheres or patchy particles
- Limited to athermal jamming and crystallization

#### ASSUMPTION 4: Compression (φ_initial < φ_target)

**What it means:**
- Packing fraction must increase during simulation
- Starting density lower than target density
- Volume decreases monotonically

**Why it matters:**
- Algorithm only shrinks box (never expands)
- Expansion requires different protocol (not implemented)
- Reversibility requires separate expansion script

**Validation:**
```python
# Script checks this at start of compression
if phi_initial >= phi_target:
    print("[WARNING] System already at or beyond target")
    # Compression loop skipped
```

#### ASSUMPTION 5: Isotropic Compression (Uniform box scaling)

**What it means:**
- Box scales equally in x, y, z directions
- Aspect ratio Lx:Ly:Lz preserved
- Tilt factors (xy, xz, yz) unchanged

**Why it matters:**
- Algorithm uses `resize_box_to_volume()` with cubic scaling
- Anisotropic compression requires different box update logic
- Uniaxial/biaxial compression not supported

**Physical consequence:**
- Cannot study shape transitions (e.g., sphere → ellipsoid)
- Cannot model confined systems with fixed wall separation
- Cannot apply external shear during compression

#### ASSUMPTION 6: Zero Initial Overlaps

**What it means:**
- Input GSD must be a valid hard-sphere configuration
- `mc.overlaps == 0` immediately after loading state
- Overlapping initial states are NOT automatically fixed

**Why it matters:**
- Compression assumes valid starting point
- Overlapping initial state → infinite inner loop
- Script warns but does not reject overlapping inputs

**How to generate valid initial state:**
```bash
# Method 1: Start from lattice (FCC, BCC, SC) at low phi
# Method 2: Expand pre-compressed state
# Method 3: Run HPMC equilibration at low phi first
```

---

### 3.2 Input File Assumptions

#### GSD File Structure Requirements

**The input GSD MUST contain:**

1. **`particles.position`** (N × 3 array, float32/float64)
   - Particle coordinates in box-scaled units
   - Range: [0, L) for each dimension
   - Wrapped to periodic boundaries

2. **`particles.typeid`** (N × 1 array, uint32)
   - All values must be 0 (type 'A')
   - If multiple types present, script fails

3. **`particles.diameter`** (N × 1 array, float32/float64)
   - All values must be identical (monodisperse)
   - Positive values only (d > 0)
   - Script reads first particle's diameter, assumes uniform

4. **`configuration.box`** (6-element array: [Lx, Ly, Lz, xy, xz, yz])
   - Box dimensions must be positive: Lx, Ly, Lz > 0
   - Tilt factors optional (default: 0 for orthorhombic box)
   - Defines periodic boundary shape

5. **Frame count ≥ 1**
   - GSD file must not be empty
   - Script reads last frame (`trajectory[-1]`)
   - If file has multiple frames, only last frame is used

**Optional GSD fields (ignored if present):**
- `particles.velocity`: HPMC doesn't use velocities (NVT MC, not MD)
- `particles.mass`: Not used in hard-sphere MC
- `particles.charge`: Not used (no electrostatics)
- `particles.orientation`: Not used (spheres are isotropic)

#### JSON Parameter File Requirements

**The JSON file MUST contain these keys:**

| Key | Type | Description | Constraints |
|-----|------|-------------|-------------|
| `tag` | string | Simulation identifier | Filesystem-safe (no spaces) |
| `input_gsd_filename` | string | Path to input GSD | File must exist |
| `stage_id_current` | int | Stage index | -1 (single) or ≥ 0 (multi) |
| `target_pf` | float | Target packing fraction | 0 < φ < 0.74048 |
| `volume_scaling_factor` | float | Box shrinkage per step | 0.5 ≤ vsf < 1.0 |
| `run_length_to_remove_overlap` | int | HPMC steps per overlap attempt | > 0 |
| `run_length_to_relax` | int | HPMC steps for equilibration | ≥ 0 |
| `move_size_translation` | float | MC move size (in diameters) | > 0 |
| `restart_frequency` | int | Restart checkpoint interval | > 0 |
| `traj_out_freq` | int | Trajectory frame interval | > 0 |
| `log_frequency` | int | Console output interval | > 0 |
| `use_gpu` | bool | Enable GPU acceleration | true/false |
| `gpu_id` | int | CUDA device index | 0 to (num_gpus - 1) |

**Optional keys (have defaults):**
- `initial_timestep`: Starting timestep (default: 0)
- `restart_gsd_filename`: Restart file (default: "restart.gsd")
- `output_gsd_traj_filename`: Trajectory file (default: "traj.gsd")
- `final_gsd_filename`: Final snapshot (default: "final.gsd")

**Comment keys (ignored by script):**
- Any key starting with `_` (underscore) is treated as a comment
- Example: `"_section_io": "--- I/O identifiers ---"`
- Allows human-readable annotations without breaking JSON parser

---

## 4. HOOMD-blue Leverages

### 4.1 What This Script Gets from HOOMD

This script leverages HOOMD-blue v4's built-in functionality extensively. Understanding what HOOMD provides vs. what the script implements is crucial for debugging and modification.

#### 4.1.1 HPMC Sphere Integrator (`hoomd.hpmc.integrate.Sphere`)

**What HOOMD provides:**

1. **Overlap Detection:**
   - Geometry-based collision checking between spheres
   - Efficiently computes pairwise distances using neighbor lists
   - Returns `mc.overlaps`: Number of overlapping particle pairs
   - O(N) complexity for local checks, O(N²) worst-case

2. **Monte Carlo Move Generation:**
   - Generates random translational displacements
   - Uniform sampling from [-d, +d] in each dimension
   - Wraps particles across periodic boundaries automatically
   - No bias or drift (detailed balance satisfied)

3. **Metropolis Acceptance:**
   - Accepts move if no new overlaps created
   - Rejects move if any overlaps detected
   - Updates particle positions atomically (all-or-nothing)
   - Maintains detailed balance (equilibrium sampling)

4. **Move Statistics:**
   - `mc.translate_moves`: Tuple of (accepted, rejected) counts
   - Cumulative over entire simulation (resets on GSD load)
   - Allows calculation of acceptance ratio
   - Used for move size tuning (manual in this script)

5. **Type Shape Management:**
   - `mc.shape["A"] = {"diameter": d}`: Define sphere size for type 'A'
   - Supports multiple types (not used here)
   - Stores shape metadata in GSD `log/hpmc/type_shapes`

6. **GPU Acceleration:**
   - Automatically uses CUDA if device=GPU
   - Neighbor list construction on GPU
   - Overlap checks parallelized across particles
   - 10-100x speedup for N > 10,000 particles

**What the script implements:**

- **Compression strategy:** When and how much to shrink box
- **Overlap removal loop:** Iteratively run HPMC until overlaps == 0
- **Equilibration protocol:** Fixed-box runs between compression steps
- **Checkpoint logic:** When to write GSD snapshots
- **Move size selection:** Choosing `default_d` (HOOMD executes, script selects)

---

#### 4.1.2 Box Resizing (`hoomd.update.BoxResize.update`)

**What HOOMD provides:**

1. **Instantaneous Affine Scaling:**
   - Updates `sim.state.box` to new dimensions
   - Scales all particle positions proportionally: `r_new = r_old * (L_new / L_old)`
   - Preserves relative positions (no clustering/spreading)
   - Applies to all particles simultaneously

2. **Tilt Factor Preservation:**
   - Maintains box shape (orthorhombic vs. triclinic)
   - `xy`, `xz`, `yz` unchanged during isotropic scaling
   - Allows compression of triclinic cells

3. **Periodic Boundary Wrapping:**
   - Automatically wraps particles that end up outside [0, L)
   - No particles lost during resize
   - Maintains correct minimum image convention

4. **Collective MPI Operation:**
   - All ranks execute resize simultaneously
   - Box dimensions identical across all ranks after resize
   - No MPI communication needed (deterministic operation)

**What the script implements:**

- **Volume calculation:** Target volume from packing fraction
- **Scale factor computation:** `s = (V_new / V_old)^(1/3)`
- **New box construction:** Create `hoomd.Box` with scaled dimensions
- **Calling sequence:** When in compression loop to call `BoxResize.update()`

**HOOMD v4.9 API difference from v2:**

```python
# HOOMD v2 (old):
update = hoomd.update.box_resize(...)
update.set_params(Lx=..., Ly=..., Lz=..., period=None)  # period=None → instant

# HOOMD v4.9 (new):
hoomd.update.BoxResize.update(
    state=sim.state,
    box=new_box,
    filter=hoomd.filter.All()
)  # Always instant (no period parameter)
```

**Critical note:**  
The v4.9 `BoxResize.update()` is a **static method**. Do NOT create an instance:

```python
# WRONG (will fail):
box_resize = hoomd.update.BoxResize(...)
box_resize.update(timestep)  # TypeError: missing required arguments

# CORRECT:
hoomd.update.BoxResize.update(state=sim.state, box=new_box, filter=...)
```

---

#### 4.1.3 GSD Writers (`hoomd.write.GSD`)

**What HOOMD provides:**

1. **Binary Snapshot Format:**
   - Efficient storage of particle states (positions, types, diameters)
   - Compressed frames (gzip, automatic)
   - Random access to frames (trajectory[i])
   - Cross-platform (works on Linux, macOS, Windows)

2. **Append Mode (`mode="ab"`):**
   - Writes new frames to end of file
   - Preserves existing frames (restarts continue trajectory)
   - Single file for entire simulation history

3. **Truncate Mode (`mode="wb"`, `truncate=True`):**
   - Overwrites file with single frame
   - Used for restart checkpoints (only need latest state)
   - Reduces disk usage (no old checkpoints accumulate)

4. **Dynamic Logging:**
   - Logs custom quantities (phi, overlaps, acceptance)
   - Stored in `log/*` namespaces within GSD
   - Accessible via `traj[i].log` in analysis scripts

5. **Trigger System:**
   - `hoomd.trigger.Periodic(N)`: Write every N timesteps
   - Integrated with `sim.run()`: No manual polling needed
   - Works across `sim.run()` calls (persistent triggers)

6. **Filter System:**
   - `hoomd.filter.All()`: Write all particles (used here)
   - `hoomd.filter.Type(['A'])`: Write only type 'A' (not needed here)
   - Allows subset selection (useful for large multi-component systems)

**What the script implements:**

- **Filename management:** Stage-aware naming conventions
- **Checkpoint strategy:** When to write restart vs. trajectory frames
- **Logger construction:** Defining custom quantities to log
- **Writer attachment:** Adding GSD writers to `sim.operations.writers`

---

#### 4.1.4 Custom Logger System (`hoomd.logging.Logger`)

**What HOOMD provides:**

1. **Extensibility via Python Properties:**
   - Any Python object can export loggable quantities
   - Requires `_export_dict` class attribute
   - Supports scalar, string, sequence, particle categories

2. **Automatic Property Interrogation:**
   - `Logger.add(obj, quantities=["prop1", "prop2"])`: Auto-discover properties
   - `Logger[("namespace", "quantity")] = (obj, "prop", "category")`: Manual registration
   - Properties called automatically at log frequency

3. **Namespacing:**
   - `("compression", "packing_fraction")`: Group related quantities
   - Stored as `log/compression/packing_fraction` in GSD
   - Prevents name collisions (e.g., multiple "temperature" quantities)

4. **DataAccessError Handling:**
   - HOOMD properties may raise `hoomd.error.DataAccessError` before first `sim.run()`
   - Logger validates loggables at attachment, not at log time
   - **CRITICAL:** Custom properties MUST catch DataAccessError

**What the script implements:**

- **Custom loggable classes:**
  - `PhiLogger`: Calculates live packing fraction from box volume
  - `OverlapLogger`: Wraps `mc.overlaps` with DataAccessError guard
  - `AcceptanceLogger`: Computes acceptance ratio from `mc.translate_moves`

- **Logger configuration:**
  - GSD logger: Full set of quantities for trajectory analysis
  - Table logger: Subset of quantities for console monitoring
  - Separate loggers avoid logging sequences to console

**Example custom loggable:**

```python
class MyLogger:
    _export_dict = {"my_quantity": ("scalar", True)}
    
    def __init__(self, mc_ref):
        self._mc = mc_ref
    
    @property
    def my_quantity(self) -> float:
        try:
            # Access HPMC property that may raise DataAccessError
            return float(self._mc.some_property)
        except hoomd.error.DataAccessError:
            # Return sentinel value before first sim.run()
            return 0.0
```

**Why DataAccessError handling is critical:**

When attaching a logger, HOOMD validates loggables by calling their property getters:

```python
logger = hoomd.logging.Logger()
logger[("namespace", "quantity")] = (obj, "my_property", "scalar")
# ↑ HOOMD calls obj.my_property HERE (validation)
# If my_property accesses mc.overlaps BEFORE first sim.run(), DataAccessError raised
# Without try/except, script crashes at logger attachment, NOT at logging time
```

---

#### 4.1.5 Device Selection (`hoomd.device.GPU` / `hoomd.device.CPU`)

**What HOOMD provides:**

1. **GPU Support:**
   - `hoomd.device.GPU(gpu_id=0)`: Initialize CUDA device
   - Automatically compiles kernels for target GPU architecture
   - Handles memory transfers (host ↔ device)
   - Supports multi-GPU via MPI (one GPU per rank)

2. **CPU Fallback:**
   - `hoomd.device.CPU()`: Pure CPU execution
   - No special hardware required
   - Uses OpenMP for multi-core parallelism (if available)
   - Slower than GPU but universally compatible

3. **Automatic Performance Tuning:**
   - HOOMD auto-tunes kernel launch parameters
   - Optimizes neighbor list skin distance
   - Adapts to hardware capabilities

**What the script implements:**

- **Conditional GPU initialization:**
  ```python
  if params.use_gpu:
      try:
          device = hoomd.device.GPU(gpu_id=params.gpu_id)
      except Exception:
          print("[WARNING] GPU init failed, falling back to CPU")
          device = hoomd.device.CPU()
  else:
      device = hoomd.device.CPU()
  ```

- **Error handling:** Catches GPU init failures, falls back to CPU gracefully
- **JSON control:** User specifies device preference in parameter file

**GPU performance characteristics:**

| System Size (N) | CPU Time (steps/s) | GPU Time (steps/s) | Speedup |
|-----------------|--------------------|--------------------|---------|
| 1,000           | 5,000              | 3,000              | 0.6x    |
| 10,000          | 500                | 10,000             | 20x     |
| 100,000         | 50                 | 50,000             | 1000x   |
| 1,000,000       | 5                  | 100,000            | 20,000x |

**Rule of thumb:** GPU is faster for N > 5,000 particles.

---

#### 4.1.6 MPI Parallelism (Optional)

**What HOOMD provides:**

1. **Domain Decomposition:**
   - Splits simulation box across MPI ranks
   - Each rank simulates particles in its subdomain
   - Automatic ghost particle exchange at domain boundaries

2. **Collective Operations:**
   - `sim.run()`: All ranks execute together
   - `sim.state.box`: Identical across all ranks
   - `mc.overlaps`: Local count per rank (no global reduction in script)

3. **I/O Serialization:**
   - GSD writes handled by rank 0 only
   - Other ranks no-op on write operations
   - Automatic synchronization ensures consistent writes

**What the script implements:**

- **Rank detection:** Read from environment variables (`OMPI_COMM_WORLD_RANK`, etc.)
- **I/O guards:** `if _is_root_rank():` before prints and file writes
- **Seed file barrier:** Wait loop until seed file visible to all ranks
- **Console output filtering:** `root_print()` only outputs on rank 0

**MPI execution example:**

```bash
# 4-rank run on single node
mpirun -np 4 python hs_compress_v10_documented.py --simulparam_file params.json

# 8-rank run across 2 nodes (4 ranks per node)
mpirun -np 8 -npernode 4 --hostfile hosts.txt python hs_compress_v10_documented.py --simulparam_file params.json
```

**Critical MPI pitfalls:**

1. **Non-collective I/O must be guarded:**
   ```python
   # WRONG (all ranks write, file corruption):
   with open("output.txt", "w") as f:
       f.write(data)
   
   # CORRECT (only rank 0 writes):
   if _is_root_rank():
       with open("output.txt", "w") as f:
           f.write(data)
   ```

2. **Seed file visibility lag:**
   - Rank 0 creates seed file on shared filesystem (NFS, Lustre)
   - Other ranks may not see file immediately (cache lag)
   - Script uses 30-second polling barrier to ensure visibility

3. **Different ranks must call `sim.run()` with same arguments:**
   ```python
   # WRONG (deadlock):
   if _is_root_rank():
       sim.run(1000)  # Rank 0 runs, others don't
   
   # CORRECT (collective):
   sim.run(1000)  # All ranks run together
   ```

---

### 4.2 What HOOMD Does NOT Provide (Script Implements)

#### 4.2.1 Compression Strategy

HOOMD provides **HPMC move acceptance**, but NOT **when/how to compress**.

**The script implements:**
- When to shrink box (outer loop condition: `volume > target_volume`)
- How much to shrink (volume *= `volume_scaling_factor`)
- How to check convergence (inner loop: `while overlaps > 0`)
- When to equilibrate (after overlaps reach zero)

**Why this matters:**
- Different compression strategies yield different final states
- Too fast → system jams (cannot reach target)
- Too slow → wastes computation time
- Optimal strategy depends on system size, target phi

#### 4.2.2 Move Size Selection

HOOMD provides **MC move execution with given `default_d`**, but NOT **what `default_d` should be**.

**The script implements:**
- Fixed move size throughout compression (no tuner)
- User must manually select `move_size_translation` in JSON
- Acceptance ratio logged but not used to adjust move size

**Tuning guidance:**
- Optimal acceptance: 30-50% at target density
- Start with `d = 0.04 * diameter`
- If acceptance < 20% at target: decrease `d` by factor of 2
- If acceptance > 60% at target: increase `d` by factor of 1.5
- Re-run compression with adjusted move size

**Why HOOMD's adaptive tuner is NOT used:**
- v2 script used fixed move size (this is faithful port)
- Adaptive tuner changes `d` during run (breaks v2 reproducibility)
- User has full control over sampling efficiency vs. speed

#### 4.2.3 Checkpoint and Restart Logic

HOOMD provides **GSD write operations**, but NOT **restart detection and recovery**.

**The script implements:**
- Check if `restart.gsd` exists and `final.gsd` does not (crashed run)
- Load from restart if conditions met, else load from input
- Periodic restart writes via `hoomd.write.GSD(..., mode="wb", truncate=True)`
- Per-step checkpoints (`current_pf.json`, `output_current_pf.gsd`)

**Restart workflow:**
```
Run 1:
  - Load input.gsd
  - Compress to phi=0.40
  - Write restart.gsd (periodic, last write at timestep 500k)
  - Job killed by scheduler (walltime limit)

Run 2 (automatic restart):
  - Detect restart.gsd exists, final.gsd does not
  - Load restart.gsd (phi=0.40, timestep=500k)
  - Continue compression from phi=0.40
  - Reach phi=0.58
  - Write final.gsd
```

#### 4.2.4 Stage-Aware Filename Resolution

HOOMD provides **GSD I/O primitives**, but NOT **multi-stage workflow management**.

**The script implements:**
- Detect `stage_id_current` from JSON
- Single-stage (`stage_id == -1`): Use filenames from JSON
- Multi-stage (`stage_id >= 0`): Auto-generate filenames
  - Stage 0 input: `input_gsd_filename` from JSON
  - Stage N input: `<tag>_<N-1>_final.gsd` (chained from previous stage)
  - Stage N outputs: `<tag>_<N>_restart.gsd`, `<tag>_<N>_traj.gsd`, `<tag>_<N>_final.gsd`
- Validate previous stage completion (check file exists)
- Prevent re-running same stage (error if final GSD exists)

**Multi-stage use case:**
```
# Stage 0: phi 0.10 → 0.45 (fast compression)
stage_id_current: 0
target_pf: 0.45
→ Output: hs_4096_0_final.gsd

# Stage 1: phi 0.45 → 0.55 (slow, long equilibration)
stage_id_current: 1
target_pf: 0.55
run_length_to_relax: 20000  # 4x longer than stage 0
→ Input: hs_4096_0_final.gsd
→ Output: hs_4096_1_final.gsd

# Stage 2: phi 0.55 → 0.64 (very slow, approaching jamming)
stage_id_current: 2
target_pf: 0.64
run_length_to_relax: 50000  # 10x longer
volume_scaling_factor: 0.995  # More conservative
→ Input: hs_4096_1_final.gsd
→ Output: hs_4096_2_final.gsd
```

#### 4.2.5 Physical Constraint Validation

HOOMD provides **simulation execution**, but NOT **parameter sanity checking**.

**The script implements:**
- Validate `target_pf < 0.74048` (hard-sphere close packing limit)
- Validate `volume_scaling_factor ∈ [0.5, 1.0)` (compression range)
- Validate `move_size_translation > 0` (positive move size)
- Validate `run_length_* > 0` (positive run lengths)
- Check input GSD exists before starting
- Check diameter monodispersity (all particles same size)
- Warn if `phi_initial >= phi_target` (already compressed)

**Example validation error:**
```
[FATAL ERROR] Parameter validation failed:
  • target_pf must be in (0, 0.740), got 0.8
    Hard sphere close packing is φ_cp ≈ 0.74048
  • volume_scaling_factor must be in [0.5, 1.0), got 1.05
    Values >= 1.0 are expansion (not compression)
  • move_size_translation must be > 0, got -0.04
    Negative move size is undefined
```

---

## 5. Parameter Tuning Guide

### 5.1 Target Packing Fraction (`target_pf`)

**Physical significance:**
- φ = (N * V_particle) / V_box
- Fraction of space occupied by particle cores
- Dimensionless quantity (ratio of volumes)

**Key phase transitions:**

| Packing Fraction | Physical State | Behavior |
|------------------|----------------|----------|
| φ < 0.40 | Dilute fluid | Fast equilibration, high acceptance |
| φ ≈ 0.49 | Freezing transition | Fluid → FCC crystal (equilibrium) |
| φ ≈ 0.55 | Supercooled liquid | Metastable, slow dynamics |
| φ ≈ 0.58 | Glass transition | Kinetic arrest, jamming |
| φ ≈ 0.64 | Random close packing (RCP) | Disordered jammed state |
| φ ≈ 0.74 | FCC/HCP close packing | Densest crystal packing (theoretical limit) |

**Tuning recommendations:**

1. **For fluid states (φ < 0.50):**
   - Can use aggressive compression: `volume_scaling_factor = 0.98`
   - Short equilibration: `run_length_to_relax = 1000`
   - Large move size: `move_size_translation = 0.06`
   - Fast convergence: ~10-20 compression steps

2. **For supercooled liquid (0.50 < φ < 0.58):**
   - Moderate compression: `volume_scaling_factor = 0.99`
   - Medium equilibration: `run_length_to_relax = 3000`
   - Medium move size: `move_size_translation = 0.04`
   - Slower convergence: ~50-100 compression steps

3. **Approaching glass transition (0.58 < φ < 0.64):**
   - Conservative compression: `volume_scaling_factor = 0.995`
   - Long equilibration: `run_length_to_relax = 10000`
   - Small move size: `move_size_translation = 0.02`
   - Very slow convergence: ~200-500 compression steps

4. **Beyond RCP (φ > 0.64):**
   - May not be achievable via compression
   - Requires crystal nucleation and growth
   - Consider multi-stage protocol with annealing
   - Alternative: Start from crystal lattice, compress to target

**Warning signs of problems:**

- **Inner loop never converges:** `volume_scaling_factor` too small
- **Acceptance ratio < 10%:** `move_size_translation` too large
- **Compression takes >1000 steps:** Target may be unreachable via kinetic path

---

### 5.2 Volume Scaling Factor (`volume_scaling_factor`)

**Definition:**
- Fractional volume reduction per compression step
- `V_new = V_old * volume_scaling_factor`
- Example: 0.99 → 1% volume reduction per step

**Physical interpretation:**
- Small factor (0.95): Aggressive, ~5% reduction
  - Fewer total steps to target
  - More overlaps per step
  - Longer inner loops (overlap removal)
  - Risk of jamming if too aggressive
  
- Large factor (0.995): Conservative, ~0.5% reduction
  - More total steps to target
  - Fewer overlaps per step
  - Shorter inner loops
  - Safer for high-density targets

**Relationship to packing fraction change:**

For isotropic scaling:
```
V_new = V_old * vsf
φ_new = φ_old / vsf
Δφ = φ_old * (1 - vsf)  (small Δφ approximation)
```

Example: φ_old = 0.50, vsf = 0.99
```
φ_new = 0.50 / 0.99 ≈ 0.505
Δφ ≈ 0.005  (0.5% increase in phi)
```

**Tuning strategy:**

1. **Start conservative (0.99):**
   - Guaranteed to work for most systems
   - Trade computation time for reliability
   - Good default for unknown systems

2. **Monitor inner loop iterations:**
   ```
   [STEP 1] overlaps: 42 12 3 1 0  (5 iterations)
   [STEP 2] overlaps: 38 10 2 0    (4 iterations)
   ```
   - If iterations < 5 consistently: Can increase speed (decrease vsf)
   - If iterations > 10 frequently: Should slow down (increase vsf)

3. **Adjust based on density:**
   - φ < 0.50: Can use vsf = 0.98 (2% steps)
   - 0.50 < φ < 0.58: Use vsf = 0.99 (1% steps)
   - φ > 0.58: Use vsf = 0.995 or 0.998 (0.5-0.2% steps)

4. **Check final state quality:**
   - If final acceptance ratio < 20%: System too jammed, decrease vsf next time
   - If overlap removal takes >100 iterations per step: vsf too aggressive

**Typical parameter values by system size:**

| N particles | Low φ (< 0.50) | Medium φ (0.50-0.58) | High φ (> 0.58) |
|-------------|----------------|----------------------|-----------------|
| 1,000       | 0.98           | 0.99                 | 0.995           |
| 10,000      | 0.99           | 0.99                 | 0.995           |
| 100,000     | 0.99           | 0.995                | 0.998           |

Larger systems require more conservative compression (higher vsf) due to increased likelihood of local jamming.

---

### 5.3 Overlap Removal Length (`run_length_to_remove_overlap`)

**Definition:**
- Number of HPMC sweeps per inner loop iteration
- 1 sweep = N trial moves (one move attempt per particle)
- Total steps per iteration = `run_length_to_remove_overlap * N`

**Physical role:**
- Removes overlaps created by box compression
- Each iteration: particles diffuse, overlaps may resolve
- Continues until `mc.overlaps == 0` (hard-sphere constraint satisfied)

**Tuning heuristic:**

**Start with:** `run_length_to_remove_overlap ≈ N / 10`

Rationale:
- Each particle needs ~N/10 trial moves to explore local environment
- Overlaps involve ~2 particles, so ~N/5 total moves to resolve
- Factor of 2 safety margin → N/10 sweeps

**Examples:**
- N = 1,000: Use 100 sweeps
- N = 4,096: Use 400-500 sweeps
- N = 10,000: Use 1,000 sweeps
- N = 100,000: Use 10,000 sweeps

**Adjustment based on inner loop behavior:**

Monitor console output:
```
[STEP 1] overlaps: 42 12 3 1 0  (5 iterations)
```

This means:
- Iteration 1: Started with 42 overlaps
- After 1000 sweeps: 12 overlaps remain
- After 2000 sweeps: 3 overlaps remain
- After 3000 sweeps: 1 overlap remains
- After 4000 sweeps: 0 overlaps (converged)

**If iterations typically > 10:**
- Increase `run_length_to_remove_overlap` by 2x
- Fewer iterations, but more work per iteration
- May be faster overall (less overhead from overlap checks)

**If iterations typically < 3:**
- Decrease `run_length_to_remove_overlap` by 0.5x
- More iterations, but less wasted work when overlaps resolve quickly
- May be faster overall (don't overshoot convergence)

**Relationship to move size:**

Small `move_size_translation` → Need MORE sweeps
- Particles diffuse slowly
- Overlaps take longer to resolve
- Increase `run_length_to_remove_overlap`

Large `move_size_translation` → Need FEWER sweeps
- Particles can hop away from overlaps quickly
- But: Lower acceptance rate (more rejected moves)
- Balance: Medium move size, medium run length

**Typical values:**

| System | N | move_size | run_length_to_remove_overlap |
|--------|---|-----------|------------------------------|
| Small  | 1k | 0.04 | 500-1000 |
| Medium | 10k | 0.04 | 1000-3000 |
| Large  | 100k | 0.04 | 5000-10000 |

---

### 5.4 Relaxation Length (`run_length_to_relax`)

**Definition:**
- Number of HPMC sweeps after overlaps removed
- Runs at **constant box volume** (no compression)
- Purpose: Equilibrate structure before next compression step

**Physical role:**

After box shrinks and overlaps are removed, system is in a **constrained configuration**:
- Particles just barely not overlapping
- Local structure may be stressed or disordered
- Configuration space not fully explored

Relaxation run allows:
1. **Local rearrangements:** Particles adjust to new density
2. **Stress relief:** Release geometric frustration from compression
3. **Decorrelation:** Forget details of compression path
4. **Pre-equilibration:** Partial equilibration before next step

**Why equilibration matters:**

**Without relaxation** (`run_length_to_relax = 0`):
- Compression is path-dependent
- Final state depends on compression protocol
- System may be in non-equilibrium state
- Reproducibility depends on exact compression sequence

**With relaxation** (`run_length_to_relax > 0`):
- Compression approaches equilibrium path
- Final state less dependent on protocol
- System closer to equilibrium at each step
- Better sampling of configuration space

**Tuning strategy:**

**Start with:** `run_length_to_relax = run_length_to_remove_overlap`

Rationale:
- Similar timescales for overlap removal and equilibration
- If system can remove overlaps in X sweeps, can partially equilibrate in X sweeps

**Adjust based on density:**

Low density (φ < 0.50):
- Fast dynamics, quick equilibration
- Can use shorter relaxation: `run_length_to_relax = 500-1000`
- System explores configuration space rapidly

Medium density (0.50 < φ < 0.58):
- Slower dynamics, longer equilibration needed
- Use medium relaxation: `run_length_to_relax = 3000-5000`
- System needs more time to decorrelate

High density (φ > 0.58):
- Very slow dynamics, approaching kinetic arrest
- Use long relaxation: `run_length_to_relax = 10000-50000`
- System may need many steps to reach quasi-equilibrium

**Zero relaxation (fast compression):**

Setting `run_length_to_relax = 0` skips equilibration entirely:
- Compression is **much faster** (saves ~50% of compute time)
- Final state is **less equilibrated** (more glassy)
- Useful for:
  - Quick exploratory runs
  - Systems where equilibration is not critical
  - Generating initial states for further equilibration

**Validation of relaxation length:**

After compression, check if system is equilibrated:
1. Run additional HPMC steps at constant volume
2. Monitor acceptance ratio and MSD (mean squared displacement)
3. If these quantities still changing: Need longer relaxation
4. If these quantities plateau: Relaxation was sufficient

**Typical values:**

| Target φ | Dynamics | run_length_to_relax |
|----------|----------|---------------------|
| < 0.45   | Fast     | 500-1000            |
| 0.45-0.55| Medium   | 3000-5000           |
| 0.55-0.60| Slow     | 5000-10000          |
| > 0.60   | Very slow| 10000-50000         |

---

### 5.5 Move Size (`move_size_translation`)

**Definition:**
- Maximum translational displacement per MC trial move
- Units: Particle diameters (d)
- Uniform sampling: Δx, Δy, Δz ∈ [-d, +d]
- Actual 3D displacement: |Δr| ∈ [0, d√3]

**Physical interpretation:**

Move size `d` sets the **length scale of exploration**:
- Small `d` (≪ diameter): Local diffusion, particles wiggle in place
- Medium `d` (~ 0.1 × diameter): Efficient sampling, particles hop to neighbors
- Large `d` (> diameter): Long jumps, but high rejection rate

**Acceptance rate dependence:**

At given density φ, acceptance rate α depends on move size:
```
α(d) ≈ exp(-ρ * d^3)  (rough approximation)
```
where ρ ~ φ / V_particle is number density.

**Optimal acceptance rate:**

Hard-sphere MC literature suggests α ≈ 30-50% is optimal:
- α < 20%: Too many rejections, inefficient
- α > 60%: Moves too small, slow diffusion
- α = 40%: Good balance of acceptance vs. exploration

**Tuning workflow:**

1. **Initial guess:** `d = 0.04` (4% of diameter)
   - Works well for most systems at φ ~ 0.5

2. **Run compression to target:**
   - Monitor final acceptance rate (logged as `translate_acceptance`)

3. **Check final acceptance:**
   ```
   compression  translate_acceptance    0.25
   ```
   - This means α_final = 25% (acceptable)

4. **Adjust if needed:**
   - α_final < 20%: **Decrease `d` by factor of 2**
     - Next run: Use `d = 0.02`
     - Will increase acceptance (smaller moves)
   
   - α_final > 60%: **Increase `d` by factor of 1.5**
     - Next run: Use `d = 0.06`
     - Will decrease acceptance (larger moves)
     - But: More efficient exploration

5. **Iterate until α_final ∈ [0.25, 0.50]**

**Density-dependent optimal move size:**

As density increases, free volume decreases, optimal move size decreases:

| φ | Free volume scale | Optimal d (empirical) |
|---|-------------------|-----------------------|
| 0.40 | 0.6 × diameter  | 0.06-0.08            |
| 0.50 | 0.5 × diameter  | 0.04-0.06            |
| 0.58 | 0.35 × diameter | 0.02-0.04            |
| 0.64 | 0.25 × diameter | 0.01-0.02            |

**Move size vs. run length tradeoff:**

Given fixed computational budget:

**Large move size** (d = 0.08):
- High rejection rate (α ~ 10%)
- Need more sweeps to achieve same diffusion
- But: Each accepted move goes further
- Good for: Low-density systems, initial compression

**Small move size** (d = 0.02):
- High acceptance rate (α ~ 60%)
- Fewer sweeps needed for local equilibration
- But: Each accepted move is small step
- Good for: High-density systems, final compression stages

**Recommendation:** Use density-adaptive move size

Multi-stage protocol example:
```
Stage 0: φ 0.10 → 0.45
move_size_translation: 0.06

Stage 1: φ 0.45 → 0.55
move_size_translation: 0.04

Stage 2: φ 0.55 → 0.64
move_size_translation: 0.02
```

**IMPORTANT: This script uses FIXED move size**

Unlike some MD/MC codes, this script does NOT adaptively tune move size during the run. The user must:
1. Run with initial guess
2. Check final acceptance
3. Manually adjust and re-run

This is intentional (matches v2 methodology for reproducibility).

---

### 5.6 Output Frequencies

#### 5.6.1 Restart Frequency (`restart_frequency`)

**Definition:**
- Timestep interval for writing restart checkpoint
- File: `restart.gsd` (single-frame, overwritten each write)
- Purpose: Recovery from crashes, job termination

**Tuning considerations:**

**Tradeoff:**
- Frequent writes (small interval): Less work lost on crash, but more I/O overhead
- Infrequent writes (large interval): More work lost, but better performance

**Typical values:**
- **Short runs** (< 1 million steps): `restart_frequency = 50000`
  - Lose at most 50k steps (~1-5 minutes)
  
- **Long runs** (> 10 million steps): `restart_frequency = 100000`
  - Lose at most 100k steps (~5-10 minutes)
  - I/O overhead becomes significant for smaller intervals

- **HPC batch jobs:** Match to expected runtime
  - If walltime = 24 hours, use interval such that ~10 restarts written
  - Example: 10M steps in 24h → restart every 1M steps

**I/O cost:**
- Each restart write: ~10-100 MB (depends on N)
- Write time: ~0.1-1 second (depends on filesystem)
- If writes happen during `sim.run()`, HOOMD overlaps I/O with computation

**Recommendation:**
```
restart_frequency = max(10000, timesteps_per_compression_step / 5)
```

This ensures at least 5 restart writes per compression step, but not more frequent than every 10k steps.

#### 5.6.2 Trajectory Frequency (`traj_out_freq`)

**Definition:**
- Timestep interval for appending frames to trajectory
- File: `traj.gsd` (multi-frame, append mode)
- Purpose: Visualization, analysis, movie generation

**Tuning considerations:**

**Tradeoff:**
- Frequent frames (small interval): High temporal resolution, large file size
- Infrequent frames (large interval): Low temporal resolution, small file size

**File size estimate:**
```
Frame size ≈ N × 12 bytes  (3 floats per position)
             + N × 4 bytes  (1 float per diameter)
             + N × 4 bytes  (1 uint32 per typeid)
             ≈ 20N bytes per frame

Example: N = 10,000, 100 frames
File size ≈ 10,000 × 20 × 100 = 20 MB
```

Add ~50% for GSD metadata and compression → ~30 MB total.

**Typical values:**
- **Visualization:** `traj_out_freq = 100000`
  - Results in ~100-200 frames for typical compression run
  - Good for movies (24-60 fps → 2-8 seconds of video)

- **Analysis:** `traj_out_freq = 50000`
  - More frames for detailed time-series analysis
  - Structure factor evolution, order parameter tracking

- **Minimal:** `traj_out_freq = 1000000`
  - Just a few snapshots showing progression
  - Reduces file size for long runs

**Recommendation for compression:**
```
traj_out_freq = run_length_to_relax
```

This writes one frame per compression step (after equilibration), giving clear progression of φ(t).

#### 5.6.3 Log Frequency (`log_frequency`)

**Definition:**
- Timestep interval for console output (Table writer)
- Prints: timestep, TPS, phi, overlaps, acceptance
- Purpose: Live monitoring, performance tracking

**Tuning considerations:**

**Tradeoff:**
- Frequent logging (small interval): Real-time monitoring, cluttered console
- Infrequent logging (large interval): Clean console, less monitoring

**Console spam example (log_frequency = 1000):**
```
timestep    tps     translate_moves    packing_fraction    overlap_count
1000        5000    (150, 850)         0.101234            0
2000        5000    (305, 1695)        0.101234            0
3000        5000    (457, 2543)        0.101234            0
...
(Prints 1000 lines for 1M steps - too much!)
```

**Clean console example (log_frequency = 100000):**
```
timestep    tps     translate_moves        packing_fraction    overlap_count
100000      5000    (15234, 84766)         0.101234            0
200000      5000    (30512, 169488)        0.102345            0
...
(Prints 10 lines for 1M steps - readable)
```

**Typical values:**
- **Interactive runs:** `log_frequency = 10000`
  - Update every few seconds, good for monitoring
  
- **Batch jobs:** `log_frequency = 100000`
  - Less output to stdout file, easier to review
  - Stdout file remains manageable size

- **Debugging:** `log_frequency = 1000`
  - Frequent updates help track down issues
  - Use only for short test runs

**Performance impact:**
- Logging overhead is negligible (< 0.1% of runtime)
- Console writes may be buffered (set `PYTHONUNBUFFERED=1` for instant output)

**Recommendation:**
```
log_frequency = restart_frequency  (Same interval)
```

This correlates console updates with checkpoint writes, making it easy to track progress.

---

## 6. Failure Modes & Troubleshooting

### 6.1 Failure Mode 1: Infinite Inner Loop

**Symptoms:**
```
[STEP 15] pf: 0.572  overlaps: 127 89 65 48 39 31 27 23 20 18 16 14 13 12 11 10 9 8 7 ...
(Continues indefinitely, overlap count decreases very slowly or oscillates)
```

**Root cause:**
- Volume reduction too aggressive for current density
- System cannot remove overlaps via local MC moves
- Overlaps become "frozen" (surrounded by other particles)

**Why it happens:**

At high densities (φ > 0.58), free volume becomes scarce:
```
Free volume per particle ≈ V_box / N - V_particle
                         = V_box / N × (1 - φ)
At φ = 0.58: Free volume ≈ 42% of particle volume
At φ = 0.64: Free volume ≈ 15% of particle volume (RCP)
```

When box shrinks by e.g. 5% (vsf = 0.95):
- Free volume drops from 42% → 37% (11% reduction)
- Particles trapped in local cages
- Overlaps cannot be resolved without collective rearrangements
- Local MC moves insufficient → inner loop stuck

**Immediate remediation (if caught early):**

1. **Kill simulation** (Ctrl+C):
   ```
   [INTERRUPTED] Compression interrupted by user
   Current phi: 0.572000
   Restart file: restart.gsd
   ```

2. **Check restart GSD:**
   ```bash
   ls -lh restart.gsd
   # If file exists and is recent, recovery possible
   ```

3. **Edit JSON parameters (make more conservative):**
   ```json
   {
     "volume_scaling_factor": 0.995,  // Was 0.98, increase to 0.995
     "run_length_to_remove_overlap": 5000,  // Was 2000, increase
     "move_size_translation": 0.02,  // Was 0.04, decrease
     "run_length_to_relax": 10000  // Was 3000, increase
   }
   ```

4. **Resume compression:**
   ```bash
   python hs_compress_v10_documented.py --simulparam_file params_adjusted.json
   # Script will auto-detect restart.gsd and continue
   ```

**Long-term prevention:**

1. **Use staged compression:**
   ```json
   # Stage 0: Fast compression to intermediate density
   {
     "stage_id_current": 0,
     "target_pf": 0.55,
     "volume_scaling_factor": 0.99
   }
   
   # Stage 1: Slow compression to final density
   {
     "stage_id_current": 1,
     "target_pf": 0.64,
     "volume_scaling_factor": 0.995,
     "run_length_to_relax": 20000
   }
   ```

2. **Monitor inner loop iterations:**
   - If iterations > 10 consistently: Slow down (increase vsf)
   - If iterations > 50: Immediate problem, stop and adjust

3. **Density-adaptive parameters:**
   - Use script to monitor φ_current
   - Switch to more conservative parameters when φ > 0.55

**Emergency recovery (simulation stuck for hours):**

1. **Find most recent checkpoint:**
   ```bash
   ls -lt *.gsd | head -5
   # Look for restart.gsd or output_current_pf.gsd
   ```

2. **Check packing fraction of checkpoint:**
   ```python
   import gsd.hoomd
   with gsd.hoomd.open("restart.gsd") as traj:
       print(traj[-1].configuration.box[:3])  # [Lx, Ly, Lz]
       N = len(traj[-1].particles.position)
       d = traj[-1].particles.diameter[0]
       V_box = traj[-1].configuration.box[0] ** 3
       phi = N * (np.pi/6) * d**3 / V_box
       print(f"Checkpoint phi: {phi:.6f}")
   ```

3. **If checkpoint is at reasonable phi (< 0.60):**
   - Copy to input GSD: `cp restart.gsd input_recovered.gsd`
   - Create new JSON with conservative parameters
   - Start fresh run from recovered state

4. **If no good checkpoint:**
   - Simulation lost, must restart from beginning
   - Use more conservative parameters from the start

---

### 6.2 Failure Mode 2: Kinetic Arrest

**Symptoms:**
```
[INFO] Outer compression loop complete
      Final phi: 0.623456
      Target phi: 0.640000
      Delta phi: 0.016544
      Overlaps: 0
      
      (Script completes, but target not reached)
```

Separately, check acceptance:
```bash
grep "translate_acceptance" *_summary.json
"translate_acceptance": 0.03  # 3% acceptance - extremely low!
```

**Root cause:**
- System has entered glassy/jammed state
- MC moves almost always rejected (no free volume)
- Particles essentially frozen in place
- Cannot continue compressing despite overlaps == 0

**Why it happens:**

At high densities, system approaches **kinetic arrest**:
- Free volume → 0 (nowhere for particles to move)
- Relaxation time τ → ∞ (infinitely slow dynamics)
- System "jammed" (mechanically stable but non-equilibrium)

**Diagnosis:**

Check three indicators:
1. **Final phi vs. target:**
   ```
   Δφ = |φ_final - φ_target|
   If Δφ > 0.01: Did not reach target
   ```

2. **Final acceptance rate:**
   ```
   If α < 10%: System is jammed
   If α < 1%: System completely frozen
   ```

3. **Compression step count:**
   ```
   If steps > 500: Struggled to make progress
   ```

**Immediate remediation:**

**Cannot fix within same run.** System is jammed, must restart with different protocol.

**Strategy 1: Anneal via expansion-recompression**

1. **Expand system slightly** (reverse compression):
   - Edit box in GSD to larger volume
   - Alternatively: Create expansion script (inverse of compression)

2. **Run long equilibration at reduced density:**
   - Example: Expand from φ = 0.62 to φ = 0.58
   - Run 1,000,000 HPMC steps at constant volume
   - Allow system to explore configuration space

3. **Recompress more slowly:**
   - Use very conservative vsf (0.998)
   - Long equilibration (run_length_to_relax = 50000)
   - May reach higher phi after annealing

**Strategy 2: Crystal nucleation**

1. **Check if system crystallized:**
   ```python
   # Compute bond-orientational order parameter
   # Q6 > 0.4 → FCC crystal forming
   # Q6 < 0.2 → Amorphous (glassy)
   ```

2. **If crystallizing:**
   - Accept that system is approaching equilibrium crystalline state
   - Can reach φ ≈ 0.74 (FCC close packing)
   - But: Requires VERY long equilibration (millions of steps)

3. **If glassy:**
   - Cannot reach higher phi via compression
   - Must use alternative preparation:
     - Start from crystal lattice, compress to target
     - Use swap MC to overcome kinetic barriers (not implemented)

**Strategy 3: Multi-stage with aggressive equilibration**

Start over with staged protocol:
```json
// Stage 0: φ 0.10 → 0.50 (fast)
{
  "stage_id_current": 0,
  "target_pf": 0.50,
  "volume_scaling_factor": 0.99,
  "run_length_to_relax": 3000
}

// Stage 1: φ 0.50 → 0.57 (slow)
{
  "stage_id_current": 1,
  "target_pf": 0.57,
  "volume_scaling_factor": 0.995,
  "run_length_to_relax": 10000
}

// Stage 2: φ 0.57 → 0.63 (very slow)
{
  "stage_id_current": 2,
  "target_pf": 0.63,
  "volume_scaling_factor": 0.998,
  "run_length_to_relax": 50000,
  "move_size_translation": 0.02
}
```

Each stage includes long equilibration, giving system chance to find lowest free-energy state.

**Long-term prevention:**

**Accept physical limits:**
- φ_RCP ≈ 0.64 is practical limit for compression of disordered hard spheres
- Higher φ requires crystallization (different timescale)
- Target φ > 0.64 may not be achievable via kinetic compression

**Use appropriate target:**
- If studying glass transition: φ_target = 0.58-0.60 (accessible)
- If studying jamming: φ_target = 0.63-0.64 (challenging but possible)
- If studying crystals: Start from FCC lattice, don't compress from fluid

---

### 6.3 Failure Mode 3: GPU Out-of-Memory

**Symptoms:**
```
[BUILD] Device: GPU 0 (CUDA)
...
[FATAL ERROR] Unexpected exception during compression
              Exception type: hoomd.CUDAError
              Message: cudaMalloc failed: out of memory

# OR (if running on HPC cluster):
slurmstepd: error: Detected 1 oom-kill event(s) in step <jobid>.
```

**Root cause:**
- System size exceeds available GPU memory
- Multiple processes sharing one GPU
- Memory leak (rare, but possible in long runs)
- Fragmentation (many small allocations)

**GPU memory usage estimate:**
```
Memory ≈ N × (48 bytes/particle)  (positions, types, diameters, neighbor lists)
       + O(N) neighbor list overhead
       + ~500 MB HOOMD framework

Example:
N = 100,000: Memory ≈ 4.8 GB + 0.5 GB = 5.3 GB (fits on 8 GB GPU)
N = 1,000,000: Memory ≈ 48 GB + 0.5 GB = 48.5 GB (does NOT fit)
```

**Immediate remediation:**

**Option 1: Fall back to CPU**
```json
{
  "use_gpu": false
}
```
- Restart simulation on CPU
- Slower (10-100x), but will complete
- Good for systems where GPU is unavailable

**Option 2: Reduce memory footprint**

1. **Decrease output frequencies:**
   ```json
   {
     "traj_out_freq": 1000000,  // Was 100000, now 10x less frequent
     "log_frequency": 1000000,  // Less buffering
     "restart_frequency": 100000  // Restart writes minimal
   }
   ```
   - GSD writers buffer in memory before flushing
   - Less frequent writes → less buffering

2. **Simplify logging:**
   - Remove custom loggers that store arrays (not used in this script)
   - Only log scalar quantities

**Option 3: Split system (not supported by script, requires modification)**

If system too large for single GPU:
- Split into multiple independent replicas
- Run compression on each replica separately
- Average results across replicas

This requires significant script modification (multi-replica support not implemented).

**Long-term prevention:**

**Check GPU memory before running:**
```bash
nvidia-smi
# Look at "Memory-Usage" column
# Ensure available memory > 2 × estimated memory usage
```

**Use GPU with sufficient memory:**
- NVIDIA RTX 3090: 24 GB (handles N ≈ 500,000)
- NVIDIA A100: 40 GB or 80 GB (handles N > 1,000,000)
- NVIDIA H100: 80 GB (handles very large systems)

**Profile memory usage:**
```bash
# Run short test
python hs_compress_v10_documented.py --simulparam_file params.json &
PID=$!

# Monitor GPU memory
watch -n 1 "nvidia-smi | grep python"

# Kill after few compression steps
kill $PID
```

Check peak memory usage, ensure headroom for full run.

**HPC considerations:**

If running on shared cluster:
- Request exclusive GPU access (no other jobs on same GPU)
- Use SLURM: `#SBATCH --gres=gpu:1 --exclusive`
- Avoid: Multiple jobs sharing one GPU (causes OOM)

---

### 6.4 Failure Mode 4: MPI Rank Desynchronization

**Symptoms:**
```
[Rank 0] Final phi: 0.580123
[Rank 1] Final phi: 0.580456  # Different!
[Rank 2] Final phi: 0.579987

# OR: Simulation hangs indefinitely with no output
```

**Root cause:**
- Different random seeds on different ranks
- Non-collective operation inside loop
- Filesystem lag (seed file not visible to all ranks)
- Different code paths on different ranks

**Why it happens:**

**Random seed mismatch:**
```python
# WRONG: Each rank generates own seed
seed = secrets.randbelow(65536)  # Different on each rank!
sim = hoomd.Simulation(device=device, seed=seed)

# CORRECT: All ranks read same seed from file
seed = read_seed(stage_id)  # Same on all ranks
sim = hoomd.Simulation(device=device, seed=seed)
```

If seeds differ:
- Each rank simulates different random walk
- Particle positions diverge over time
- Final states different across ranks
- **Simulation is wrong!**

**Non-collective loop:**
```python
# WRONG: Only rank 0 runs compression
if _is_root_rank():
    while volume > target_volume:
        sim.run(1000)  # Other ranks don't execute!

# CORRECT: All ranks execute loop
while volume > target_volume:
    sim.run(1000)  # Collective operation
    # Only I/O is guarded by rank check
    if _is_root_rank():
        write_checkpoint()
```

If only rank 0 executes loop:
- Other ranks wait at next collective operation (`sim.run()`)
- Deadlock: Rank 0 continues, others stuck
- Simulation hangs forever

**Filesystem lag (seed file):**
```python
# Rank 0 writes seed file to NFS
if rank == 0:
    with open("random_seed.json", "w") as f:
        json.dump({"random_seed": 42753}, f)

# Rank 1 tries to read immediately
with open("random_seed.json", "r") as f:  # FileNotFoundError!
    seed = json.load(f)["random_seed"]
```

On networked filesystems (NFS, Lustre), file writes are not instantly visible to other nodes:
- Rank 0 writes seed → Cached on node 0
- Rank 1 tries to read → File not in cache, not yet on node 1's view of filesystem
- Results in `FileNotFoundError` or stale data

Script includes seed file barrier:
```python
# Wait until seed file is readable (with timeout)
barrier_start_time = time.perf_counter()
while not Path(seed_file).exists():
    if time.perf_counter() - barrier_start_time > 30:
        sys.exit("[FATAL ERROR] Timeout waiting for seed file.")
    time.sleep(0.1)  # Poll every 100ms
```

This ensures all ranks see the file before proceeding.

**Diagnosis:**

**Check if ranks are synchronized:**
```bash
# In MPI run, redirect each rank to separate file
mpirun -np 4 python script.py > output.\$RANK.txt 2>&1

# Compare output files
diff output.0.txt output.1.txt
# If outputs differ: Ranks desynchronized
```

**Check seed values:**
```bash
# Add debug print at seed read
# All ranks should print same seed
[Rank 0] Random seed: 42753
[Rank 1] Random seed: 42753  # MUST match rank 0
[Rank 2] Random seed: 42753
[Rank 3] Random seed: 42753
```

If seeds differ → Script bug (seed file barrier failed).

**Immediate remediation:**

**If ranks diverged but simulation completed:**
- Cannot fix retrospectively
- Results are invalid
- Must re-run simulation

**If simulation hung (deadlock):**
- Kill all ranks: `scancel <jobid>` (SLURM) or `Ctrl+C`
- Check logs for last operation each rank completed
- Identify where ranks diverged (usually at conditional branch)

**Long-term prevention:**

**Use batch job submission scripts that enforce:**
1. **Same environment on all ranks:**
   ```bash
   export PYTHONUNBUFFERED=1
   export OMP_NUM_THREADS=1
   # Ensure all ranks see same env vars
   ```

2. **Shared filesystem for all I/O:**
   ```bash
   cd /shared/scratch/username/project  # Not /tmp or /localscratch
   ```

3. **Explicit rank identification in debug output:**
   ```python
   rank = _mpi_rank_from_env()
   print(f"[Rank {rank}] Executing step X")
   ```

**Validate MPI correctness:**

Run short test with 4 ranks, check that:
- All ranks print same seed
- All ranks execute same number of `sim.run()` calls
- Final phi identical across ranks (to ~1e-10 relative tolerance)

```bash
mpirun -np 4 python hs_compress_v10_documented.py --simulparam_file test_params.json
# Check: All ranks report same phi_final
```

---

### 6.5 Failure Mode 5: Incorrect Initial State

**Symptoms:**
```
[INNER LOOP] Removing overlaps: 347 349 351 348 352 ...
(Overlap count oscillates, never decreases to zero)
```

OR:

```
[WARNING] Initial configuration has 347 overlaps.
[INNER LOOP] Removing overlaps: 347 289 223 185 ...
(Many iterations needed even before first compression step)
```

**Root cause:**
- Input GSD has overlapping particles
- Overlaps may be due to:
  - Generated with wrong diameter
  - Box too small for particle count
  - Previous compression failed, left invalid state
  - File corruption

**Why it matters:**

Compression algorithm assumes **valid initial state**:
- All particles non-overlapping: `mc.overlaps == 0`
- Box volume consistent with particle diameter
- Packing fraction accurately computed

If initial state has overlaps:
- First compression step starts from invalid configuration
- Overlaps from initial state + overlaps from compression
- Inner loop must remove both (takes very long)
- May not converge if overlaps are severe

**Diagnosis:**

**Check initial overlaps immediately after loading:**
```bash
# Run script, look for initial overlaps message
[INFO] Initial equilibration: 100 timesteps
[INFO] Initial configuration: 0 overlaps (valid hard-sphere state)
# ↑ Good: No initial overlaps

[WARNING] Initial configuration has 42 overlaps.
# ↑ Bad: Initial state is invalid
```

**Validate GSD file manually:**
```python
import gsd.hoomd
import numpy as np

def check_overlaps(filename):
    with gsd.hoomd.open(filename) as traj:
        snap = traj[-1]
        positions = snap.particles.position
        diameter = snap.particles.diameter[0]
        box = snap.configuration.box
        N = len(positions)
        
        # Brute-force overlap check (slow for large N)
        overlaps = 0
        for i in range(N):
            for j in range(i+1, N):
                dr = positions[j] - positions[i]
                # Apply minimum image convention
                dr -= box[:3] * np.round(dr / box[:3])
                dist = np.linalg.norm(dr)
                if dist < diameter:
                    overlaps += 1
        
        print(f"Overlaps in {filename}: {overlaps}")
        return overlaps

check_overlaps("input.gsd")
```

**Immediate remediation:**

**Option 1: Generate new initial state**

Create valid low-density configuration from scratch:
```python
import gsd.hoomd
import numpy as np

N = 4096
phi_initial = 0.10
diameter = 1.0

# Calculate box size for target phi
V_particle = (np.pi/6) * diameter**3
V_box = N * V_particle / phi_initial
L = V_box ** (1/3)

# Place particles on lattice (guarantees no overlaps)
n_side = int(np.ceil(N ** (1/3)))
spacing = L / n_side
positions = []
for i in range(n_side):
    for j in range(n_side):
        for k in range(n_side):
            if len(positions) < N:
                positions.append([i*spacing, j*spacing, k*spacing])

positions = np.array(positions)

# Write to GSD
with gsd.hoomd.open("init_valid.gsd", "w") as f:
    snap = gsd.hoomd.Snapshot()
    snap.configuration.box = [L, L, L, 0, 0, 0]
    snap.particles.N = N
    snap.particles.position = positions
    snap.particles.typeid = [0] * N
    snap.particles.diameter = [diameter] * N
    f.append(snap)

print(f"Created valid initial state: init_valid.gsd")
print(f"  N={N}, phi={phi_initial:.4f}, L={L:.4f}")
```

**Option 2: Expand existing state**

If input GSD has overlaps due to too-high density:
```python
import gsd.hoomd

# Read overlapping configuration
with gsd.hoomd.open("input_overlapping.gsd") as traj:
    snap = traj[-1]

# Expand box by 10% (volume increases by 1.1^3 ≈ 1.33)
box_old = snap.configuration.box
box_new = [L * 1.1 for L in box_old[:3]] + list(box_old[3:])
snap.configuration.box = box_new

# Scale particle positions accordingly
positions_old = snap.particles.position
positions_new = positions_old * 1.1
snap.particles.position = positions_new

# Write expanded configuration
with gsd.hoomd.open("input_expanded.gsd", "w") as f:
    f.append(snap)

print("Expanded box by 10%, saved to input_expanded.gsd")
```

Then:
- Run HPMC equilibration on expanded state (remove any remaining overlaps)
- Compress from expanded state to desired target

**Option 3: Run overlap removal as separate step**

Before compression, run dedicated overlap-removal script:
```python
import hoomd
import gsd.hoomd

# Load overlapping state
sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=42)
sim.create_state_from_gsd("input_overlapping.gsd")

# Setup HPMC with small move size for careful overlap removal
mc = hoomd.hpmc.integrate.Sphere(default_d=0.01, nselect=1)
mc.shape["A"] = {"diameter": sim.state.particles.diameter[0]}
sim.operations.integrator = mc

# Run until overlaps reach zero
print(f"Initial overlaps: {mc.overlaps}")
while mc.overlaps > 0:
    sim.run(10000)
    print(f"  Overlaps: {mc.overlaps}")

# Save valid state
hoomd.write.GSD.write(state=sim.state, filename="input_valid.gsd", mode="wb")
print("Saved valid state to input_valid.gsd")
```

**Long-term prevention:**

**Always validate initial GSD:**
```bash
# Add to workflow: Check overlaps before running compression
python check_overlaps.py input.gsd
# Only proceed if output is: Overlaps: 0
```

**Generate initial states carefully:**
- Use lattice placement (SC, BCC, FCC) for zero overlaps
- If randomly placing: Check overlaps after generation
- Maintain library of validated initial states for common (N, φ) pairs

**Document GSD provenance:**
```json
// Include in JSON parameter file
{
  "input_gsd_filename": "init_N4096_phi010_SC_lattice.gsd",
  "_comment_input": "Simple cubic lattice, N=4096, phi=0.10, d=1.0"
}
```

This helps track where GSD came from, aids debugging if issues arise.

---

## 7. Installation & Dependencies

### 7.1 Conda Installation (Recommended)

Conda provides pre-compiled HOOMD with GPU support (no compilation needed):

```bash
# Create new environment
conda create -n hoomd_env python=3.10

# Activate environment
conda activate hoomd_env

# Install HOOMD-blue, GSD, NumPy
conda install -c conda-forge hoomd gsd numpy

# Verify installation
python -c "import hoomd; print(f'HOOMD version: {hoomd.version.version}')"
# Should print: HOOMD version: 4.9.0 (or later)

# Check GPU availability
python -c "import hoomd; hoomd.device.GPU()"
# Should print: HOOMD.device.GPU(...) info
# If error: GPU not available, script will fall back to CPU
```

**Advantages:**
- Pre-compiled binaries (no C++ compiler needed)
- Includes GPU support (CUDA pre-compiled)
- Handles all dependencies automatically
- Works on Linux, macOS, Windows (WSL2)

**Disadvantages:**
- Conda package may be slightly behind latest HOOMD release
- Cannot customize build (e.g., specific CUDA version)

---

### 7.2 Pip Installation

Pip provides HOOMD wheels for common platforms:

```bash
# Create virtual environment
python3 -m venv hoomd_venv
source hoomd_venv/bin/activate  # Linux/macOS
# hoomd_venv\Scripts\activate  # Windows

# Install HOOMD-blue, GSD, NumPy
pip install --upgrade pip
pip install hoomd-blue gsd numpy

# Verify installation
python -c "import hoomd; print(hoomd.version.version)"
```

**Advantages:**
- Simpler than conda (uses system Python)
- Works in restricted environments (no conda)

**Disadvantages:**
- GPU support depends on available wheels
- May require compilation from source (needs C++ compiler)
- Less tested than conda packages

---

### 7.3 Compilation from Source (Advanced)

For custom builds (specific CUDA version, optimizations):

```bash
# Install dependencies
sudo apt-get install cmake g++ python3-dev  # Ubuntu/Debian
# brew install cmake  # macOS

# Clone HOOMD repository
git clone --recursive https://github.com/glotzerlab/hoomd-blue.git
cd hoomd-blue

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake .. \
  -DCMAKE_INSTALL_PREFIX=~/hoomd_install \
  -DENABLE_GPU=ON \
  -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86" \
  -DENABLE_MPI=ON

# Build (use all cores)
make -j$(nproc)

# Install
make install

# Add to Python path
export PYTHONPATH=~/hoomd_install/lib/python3.10/site-packages:$PYTHONPATH

# Verify
python -c "import hoomd; print(hoomd.version.version)"
```

**When to compile from source:**
- Need specific CUDA version (for older/newer GPUs)
- Want MPI support (multi-node parallelism)
- Need bleeding-edge features (latest git master)
- HPC cluster with specific compilers/MPI

**CUDA architecture codes:**
- Pascal (P100): 60
- Volta (V100): 70
- Turing (RTX 20xx): 75
- Ampere (RTX 30xx, A100): 80, 86
- Ada (RTX 40xx): 89
- Hopper (H100): 90

---

### 7.4 Verification & Testing

After installation, run these checks:

**1. Import test:**
```python
import hoomd
import gsd.hoomd
import numpy as np
print(f"HOOMD version: {hoomd.version.version}")
print(f"GSD version: {gsd.__version__}")
print(f"NumPy version: {np.__version__}")
```

**2. CPU simulation test:**
```python
import hoomd
sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=42)
print("CPU device initialized successfully")
```

**3. GPU simulation test:**
```python
import hoomd
try:
    sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=42)
    print("GPU device initialized successfully")
except Exception as e:
    print(f"GPU not available: {e}")
```

**4. Small compression test:**
```bash
# Download example files
wget https://raw.githubusercontent.com/.../init_small.gsd
wget https://raw.githubusercontent.com/.../params_test.json

# Run short compression (should complete in < 1 minute)
python hs_compress_v10_documented.py --simulparam_file params_test.json

# Check output
ls -lh *final.gsd
# Should see: test_final.gsd (few MB)
```

---

## 8. Usage Examples

### 8.1 Basic Single-Stage Compression

**Scenario:** Compress 4096 hard spheres from φ=0.10 to φ=0.58

**Step 1: Prepare initial GSD** (use provided or generate)
```bash
# Assuming init.gsd exists with N=4096, d=1.0, phi=0.10
ls -lh init.gsd
# Should show: init.gsd (few MB)
```

**Step 2: Create JSON parameter file**
```json
{
    "_section_io": "--- I/O ---",
    "tag": "hs_4096_compression",
    "input_gsd_filename": "init.gsd",
    "stage_id_current": -1,
    
    "_section_physics": "--- Physics ---",
    "target_pf": 0.58,
    "volume_scaling_factor": 0.99,
    "run_length_to_remove_overlap": 2500,
    "run_length_to_relax": 3000,
    "move_size_translation": 0.04,
    
    "_section_output": "--- Output ---",
    "restart_frequency": 10000,
    "traj_out_freq": 100000,
    "log_frequency": 10000,
    
    "_section_hardware": "--- Hardware ---",
    "use_gpu": false,
    "gpu_id": 0,
    
    "_section_filenames": "--- Filenames ---",
    "restart_gsd_filename": "hs_4096_restart.gsd",
    "output_gsd_traj_filename": "hs_4096_traj.gsd",
    "final_gsd_filename": "hs_4096_final.gsd"
}
```

Save as `params.json`.

**Step 3: Run compression**
```bash
python hs_compress_v10_documented.py --simulparam_file params.json
```

**Expected output:**
```
================================================================================
HOOMD-BLUE V4 | HARD-SPHERE HPMC COMPRESSION (v10 - DOCUMENTED)
================================================================================
  Parameter file: params.json
  HOOMD version:  4.9.0
  Start time:     2025-03-26 10:30:45
================================================================================

[PARAMS] Loaded parameters from 'params.json':
         tag                         = hs_4096_compression
         stage_id_current            = -1
         target_pf                   = 0.58
         volume_scaling_factor       = 0.99
         ...

[BUILD] Device: CPU
[BUILD] Simulation created (seed=42753)
[BUILD] Starting fresh from: 'init.gsd'
...

================================================================================
COMPRESSION PARAMETERS
================================================================================
  Initial phi:      0.100000
  Target phi:       0.580000
  Initial volume:   42949.672960
  Target volume:    7405.115168
  Volume scaling:   0.99 (1.0% reduction per step)
  ...

[INFO] Initial equilibration: 100 timesteps
[INFO] Initial configuration: 0 overlaps (valid hard-sphere state)

================================================================================
BEGINNING COMPRESSION LOOP
================================================================================

────────────────────────────────────────────────────────────────────────────────
[COMPRESSION STEP 1]
  Packing fraction: 0.101010
  Box volume:       42500.176329
  Overlaps:         42
────────────────────────────────────────────────────────────────────────────────
[INNER LOOP] Removing overlaps: 42 12 3 0 
[INNER LOOP] Overlaps removed in 4 iterations (10000 total steps)
[EQUILIBRATION] Running 3000 steps at φ = 0.101010
[EQUILIBRATION] Complete
[CHECKPOINT] current_pf.json written (φ = 0.1010)
[CHECKPOINT] output_current_pf.gsd written

... (many more compression steps) ...

────────────────────────────────────────────────────────────────────────────────
[COMPRESSION STEP 157]
  Packing fraction: 0.580123
  Box volume:       7401.234567
  Overlaps:         15
────────────────────────────────────────────────────────────────────────────────
[INNER LOOP] Removing overlaps: 15 7 2 0 
[INNER LOOP] Overlaps removed in 4 iterations (10000 total steps)
[EQUILIBRATION] Running 3000 steps at φ = 0.580123
[EQUILIBRATION] Complete

================================================================================
COMPRESSION COMPLETE
================================================================================
  Final timestep:      2354230
  Final phi:           0.580123
  Target phi:          0.580000
  Delta phi:           1.23e-04
  Final overlaps:      0
  Compression steps:   157
================================================================================

[OUTPUT] Labelled snapshot: hs_4096_compressed_to_pf_0p5801.gsd
[OUTPUT] Final snapshot:    hs_4096_final.gsd
[OUTPUT] Summary JSON:      hs_4096_compression_summary.json

================================================================================
  HARD SPHERE COMPRESSION - RUN SUMMARY
================================================================================
  Simulation tag:           hs_4096_compression
  ...
  Final packing fraction:   0.580123
  ...
================================================================================

[SUCCESS] Compression completed successfully
```

**Step 4: Check outputs**
```bash
ls -lh hs_4096*.gsd
# hs_4096_compressed_to_pf_0p5801.gsd  (~50 MB, labelled final state)
# hs_4096_final.gsd                    (~50 MB, canonical final state)
# hs_4096_traj.gsd                     (~800 MB, full trajectory)
# hs_4096_restart.gsd                  (~50 MB, last checkpoint)

ls -lh hs_4096*.json
# hs_4096_compression_summary.json     (summary metadata)
# current_pf.json                      (last compression step checkpoint)
```

---

### 8.2 Multi-Stage Compression

**Scenario:** Compress from φ=0.10 to φ=0.64 in three stages

**Stage 0: φ 0.10 → 0.45 (fast compression)**

`params_stage0.json`:
```json
{
    "tag": "hs_4096_staged",
    "input_gsd_filename": "init.gsd",
    "stage_id_current": 0,
    "target_pf": 0.45,
    "volume_scaling_factor": 0.99,
    "run_length_to_remove_overlap": 2000,
    "run_length_to_relax": 2000,
    "move_size_translation": 0.05,
    "restart_frequency": 10000,
    "traj_out_freq": 100000,
    "log_frequency": 10000,
    "use_gpu": false,
    "gpu_id": 0
}
```

```bash
python hs_compress_v10_documented.py --simulparam_file params_stage0.json
# Outputs: hs_4096_staged_0_final.gsd
```

**Stage 1: φ 0.45 → 0.55 (moderate compression)**

`params_stage1.json` (only changed fields shown):
```json
{
    "tag": "hs_4096_staged",
    "stage_id_current": 1,
    "target_pf": 0.55,
    "volume_scaling_factor": 0.99,
    "run_length_to_relax": 5000,
    "move_size_translation": 0.04,
    ...
}
```

```bash
python hs_compress_v10_documented.py --simulparam_file params_stage1.json
# Reads: hs_4096_staged_0_final.gsd (auto-detected from stage 0)
# Outputs: hs_4096_staged_1_final.gsd
```

**Stage 2: φ 0.55 → 0.64 (slow compression)**

`params_stage2.json`:
```json
{
    "tag": "hs_4096_staged",
    "stage_id_current": 2,
    "target_pf": 0.64,
    "volume_scaling_factor": 0.995,  # More conservative
    "run_length_to_remove_overlap": 5000,
    "run_length_to_relax": 20000,  # Longer equilibration
    "move_size_translation": 0.02,  # Smaller move size
    ...
}
```

```bash
python hs_compress_v10_documented.py --simulparam_file params_stage2.json
# Reads: hs_4096_staged_1_final.gsd
# Outputs: hs_4096_staged_2_final.gsd (final compressed state)
```

**Final result:**
- Three GSD trajectories: `hs_4096_staged_0_traj.gsd`, `_1_traj.gsd`, `_2_traj.gsd`
- Final state: `hs_4096_staged_2_final.gsd` at φ ≈ 0.64

---

### 8.3 GPU-Accelerated Compression

**Scenario:** Large system (N=100,000), use GPU for speed

`params_gpu.json` (relevant fields):
```json
{
    "tag": "hs_100k_gpu",
    "input_gsd_filename": "init_100k.gsd",
    "target_pf": 0.58,
    ...
    "use_gpu": true,
    "gpu_id": 0
}
```

```bash
# Check GPU availability
nvidia-smi

# Run compression
python hs_compress_v10_documented.py --simulparam_file params_gpu.json

# Monitor GPU usage during run
watch -n 1 nvidia-smi
```

**Expected speedup:**
- N=100,000: GPU 20-50x faster than CPU
- Runtime: ~2-4 hours (GPU) vs. ~2-3 days (CPU)

---

### 8.4 MPI Parallel Run (Multi-Node)

**Scenario:** Compress on HPC cluster with MPI

**SLURM batch script** (`submit_compression.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=hs_compress
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=compress_%j.out
#SBATCH --error=compress_%j.err

# Load modules (cluster-specific)
module load python/3.10
module load openmpi/4.1

# Activate environment
source ~/hoomd_env/bin/activate

# Run compression with MPI (8 ranks total: 2 nodes × 4 ranks/node)
mpirun -np 8 python hs_compress_v10_documented.py --simulparam_file params.json
```

Submit job:
```bash
sbatch submit_compression.sh
```

Monitor job:
```bash
squeue -u $USER
tail -f compress_.out
```

**Note:** Each MPI rank gets a subdomain of the simulation box. Domain decomposition handled automatically by HOOMD. Script ensures all ranks use same seed and execute collectively.

---

## 9. File Formats

### 9.1 GSD (HOOMD Snapshot) Format

**Extension:** `.gsd`  
**Type:** Binary, cross-platform  
**Library:** `gsd.hoomd` (Python), `gsd` (C++)

**Structure:**
```
GSD file:
  ├─ Frame 0 (initial state)
  │   ├─ configuration
  │   │   ├─ box: [Lx, Ly, Lz, xy, xz, yz]
  │   │   ├─ step: 0
  │   │   └─ dimensions: 3
  │   ├─ particles
  │   │   ├─ N: 4096
  │   │   ├─ types: ["A"]
  │   │   ├─ typeid: [0, 0, 0, ..., 0]  (N elements)
  │   │   ├─ position: [[x, y, z], ...]  (N × 3)
  │   │   └─ diameter: [1.0, 1.0, ..., 1.0]  (N elements)
  │   └─ log (if logged quantities present)
  │       ├─ compression/packing_fraction: [0.1]
  │       ├─ compression/overlap_count: [0.0]
  │       └─ ...
  ├─ Frame 1 (after 100,000 steps)
  │   ├─ configuration
  │   │   └─ step: 100000
  │   ├─ particles
  │   │   └─ position: [[x', y', z'], ...]
  │   └─ log
  │       ├─ compression/packing_fraction: [0.12]
  │       └─ ...
  └─ ...
```

**Reading GSD in Python:**
```python
import gsd.hoomd
import numpy as np

with gsd.hoomd.open("traj.gsd", mode="r") as traj:
    # Inspect file
    print(f"Number of frames: {len(traj)}")
    print(f"Timesteps: {[f.configuration.step for f in traj]}")
    
    # Read last frame
    frame = traj[-1]
    
    # Access attributes
    N = frame.particles.N
    positions = frame.particles.position  # (N, 3) array
    diameters = frame.particles.diameter  # (N,) array
    box = frame.configuration.box  # [Lx, Ly, Lz, xy, xz, yz]
    
    # Calculate packing fraction
    V_particle = (np.pi/6) * diameters[0]**3  # Assumes monodisperse
    V_box = box[0] * box[1] * box[2]
    phi = N * V_particle / V_box
    print(f"Packing fraction: {phi:.6f}")
    
    # Access logged quantities (if present)
    if hasattr(frame, 'log'):
        if 'compression/packing_fraction' in frame.log:
            phi_logged = frame.log['compression/packing_fraction'][0]
            print(f"Logged phi: {phi_logged:.6f}")
```

**Visualizing GSD:**

**Option 1: OVITO (Open Visualization Tool)**
```bash
# Install OVITO: https://www.ovito.org/
# Open GSD file in GUI, or use Python interface:

from ovito.io import import_file
from ovito.vis import Viewport

pipeline = import_file("traj.gsd")
vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.render_image(filename="snapshot.png", size=(800, 600))
```

**Option 2: VMD (Visual Molecular Dynamics)**
```bash
# Convert GSD to XYZ format first (VMD doesn't natively read GSD)
python gsd_to_xyz.py traj.gsd traj.xyz
vmd traj.xyz
```

**Option 3: matplotlib (2D projection)**
```python
import matplotlib.pyplot as plt

with gsd.hoomd.open("traj.gsd") as traj:
    frame = traj[-1]
    positions = frame.particles.position
    
    plt.figure(figsize=(8, 8))
    plt.scatter(positions[:, 0], positions[:, 1], s=50, alpha=0.6)
    plt.xlim(0, frame.configuration.box[0])
    plt.ylim(0, frame.configuration.box[1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Packing fraction: {phi:.3f}")
    plt.axis('equal')
    plt.savefig("projection_xy.png", dpi=150)
```

---

### 9.2 JSON (Parameter & Summary) Format

**Extension:** `.json`  
**Type:** Text, human-readable  
**Library:** `json` (Python standard library)

**Parameter file structure:**
```json
{
    "_section_io": "--- I/O identifiers (comment keys ignored) ---",
    "tag": "hard_sphere_4096",
    "input_gsd_filename": "init.gsd",
    "stage_id_current": -1,
    
    "_section_physics": "--- Physical parameters ---",
    "target_pf": 0.58,
    "volume_scaling_factor": 0.99,
    
    ...
}
```

**Comment keys:**
- Any key starting with `_` is ignored by script
- Use for section headers (`_section_*`) or inline comments (`_comment_*`)
- Allows human-readable annotations without breaking parser

**Summary file structure** (auto-generated):
```json
{
    "tag": "hard_sphere_4096",
    "stage_id": -1,
    "input_gsd": "init.gsd",
    "final_gsd": "hard_sphere_4096_final.gsd",
    "compressed_gsd": "hard_sphere_4096_compressed_to_pf_0p5801.gsd",
    "n_particles": 4096,
    "diameter": 1.0,
    "target_pf": 0.58,
    "phi_initial": 0.10000000,
    "phi_final": 0.58012345,
    "delta_phi": 0.00012345,
    "overlaps_final": 0,
    "final_timestep": 2354230,
    "random_seed": 42753,
    "box_final": {
        "Lx": 18.95612345,
        "Ly": 18.95612345,
        "Lz": 18.95612345,
        "xy": 0.0,
        "xz": 0.0,
        "yz": 0.0,
        "volume": 6812.34567890
    },
    "parameters": {
        "volume_scaling_factor": 0.99,
        "run_length_to_remove_overlap": 2500,
        "run_length_to_relax": 3000,
        "move_size_translation": 0.04
    },
    "created_at": "2025-03-26T14:35:22",
    "hoomd_version": "4.9.0"
}
```

**Reading summary in Python:**
```python
import json

with open("hard_sphere_4096_compression_summary.json") as f:
    summary = json.load(f)

print(f"Compression: phi {summary['phi_initial']:.4f} → {summary['phi_final']:.4f}")
print(f"Final box: {summary['box_final']['Lx']:.4f} × {summary['box_final']['Ly']:.4f} × {summary['box_final']['Lz']:.4f}")
print(f"Total timesteps: {summary['final_timestep']}")
```

---

## 10. Algorithm Details

### 10.1 Two-Level Loop Structure

**Pseudocode:**
```
INPUT: N particles, diameter d, phi_initial, phi_target, vsf
OUTPUT: Compressed GSD at phi ≈ phi_target

# Calculate target volume
V_target = N * (pi/6) * d^3 / phi_target

# Initial equilibration
RUN HPMC for 100 steps

# OUTER LOOP: Compress box
WHILE V_current > V_target:
    # Shrink box
    V_new = max(V_current * vsf, V_target)
    RESIZE_BOX(V_new)  # Affine scaling of particles
    
    overlaps = COUNT_OVERLAPS()
    
    # INNER LOOP: Remove overlaps
    WHILE overlaps > 0:
        RUN HPMC for run_length_to_remove_overlap steps
        overlaps = COUNT_OVERLAPS()
    END WHILE
    
    # Equilibrate at constant volume
    RUN HPMC for run_length_to_relax steps
    
    # Checkpoint
    WRITE current_pf.json
    WRITE output_current_pf.gsd
END WHILE

WRITE final.gsd
```

### 10.2 Convergence Criteria

**Outer loop termination:**
```
V_current <= V_target
```

**Equivalently (packing fraction):**
```
phi_current >= phi_target
```

**May overshoot slightly:** Final phi can exceed target by ~0.001 due to discrete compression steps.

**Inner loop termination:**
```
overlaps == 0
```

**Hard constraint:** Must reach exactly zero overlaps before proceeding. Non-zero overlaps at compression step N become harder to remove at step N+1 (higher density, less free volume).

### 10.3 Overlap Detection

HOOMD's `mc.overlaps` uses **bounding sphere hierarchy** for efficient collision detection:

1. **Build neighbor list:**
   - Divide box into cells (cell size ~ 1.5 × diameter)
   - Each particle assigned to cell based on position
   - Neighbors = particles in same cell + adjacent cells
   - O(N) construction time

2. **Check overlaps:**
   - For each particle i, check distance to all neighbors j
   - Overlap if: `distance(i, j) < diameter`
   - Apply minimum image convention (periodic boundaries)
   - Count unique pairs (avoid double-counting)
   - O(N × k) where k = average neighbors per particle (~10-50)

3. **Update after each move:**
   - When particle i moves, only recheck neighbors of i
   - Don't recheck entire system
   - Incremental update: O(k) per move

**Computational cost:**
```
Overlap check: O(N × k)
One MC sweep: O(N × k)  (N moves, each checks ~k neighbors)
Inner loop iteration: run_length_to_remove_overlap sweeps
```

For N=10,000, k=20, run_length=2500:
```
One iteration: 10,000 × 20 × 2500 = 500 million neighbor checks
At 10^8 checks/second: ~5 seconds per iteration
Typical inner loop: 3-10 iterations = 15-50 seconds
```

### 10.4 Box Resizing (Affine Transformation)

**Goal:** Change box volume from V_old to V_new while preserving relative particle positions.

**Algorithm:**
```python
# 1. Calculate isotropic scale factor
s = (V_new / V_old) ** (1/3)

# 2. Scale box dimensions
Lx_new = Lx_old * s
Ly_new = Ly_old * s
Lz_new = Lz_old * s

# 3. Scale particle positions (affine transformation)
for i in range(N):
    x_new[i] = x_old[i] * s
    y_new[i] = y_old[i] * s
    z_new[i] = z_old[i] * s

# 4. Wrap particles to [0, L) (periodic boundaries)
for i in range(N):
    x_new[i] = x_new[i] % Lx_new
    y_new[i] = y_new[i] % Ly_new
    z_new[i] = z_new[i] % Lz_new
```

**Properties:**
- **Volume preservation:** V_new = s³ × V_old
- **Relative positions preserved:** r_ij_new / L_new = r_ij_old / L_old
- **Packing fraction increases:** phi_new = phi_old / s³
- **Overlaps may be created:** Pairs that were non-overlapping may now overlap

**Example:**
```
Initial:
  Lx = Ly = Lz = 100
  V = 1,000,000
  d = 1.0
  N = 100,000
  phi = 100,000 * (π/6) * 1³ / 1,000,000 = 0.0524

Resize to phi = 0.58 (target):
  V_new = 100,000 * (π/6) * 1³ / 0.58 = 90,316
  s = (90,316 / 1,000,000)^(1/3) = 0.449
  Lx_new = 100 * 0.449 = 44.9
  
All particles compressed toward box center by factor 0.449.
Many pairs now overlapping → inner loop must remove overlaps.
```

---

## 11. Physical Background

### 11.1 Hard-Sphere Model

**History:**
- Introduced by Kirkwood & Monroe (1941) for simple liquids
- Exact solution in 1D (Tonks gas, 1936)
- No exact solution in 2D/3D (studied via simulation)

**Why study hard spheres?**
1. **Simplest non-trivial model:** No parameters (just diameter)
2. **Exhibits rich phase behavior:** Fluid, crystal, glass
3. **Maps to real systems:** Colloids, granular materials, proteins
4. **Theoretical importance:** Benchmark for liquid-state theories

**Phase diagram (3D hard spheres):**
```
                    Fluid (disordered)
φ < 0.494:         Equilibrium state
                   High diffusivity (D ~ 10^-2 d²/τ)
                   Liquid-like structure

φ ≈ 0.494:         FREEZING TRANSITION
                   Coexistence: fluid + FCC crystal
                   First-order phase transition

0.494 < φ < 0.545: Metastable fluid (supercooled)
                   Slow dynamics (D ~ 10^-4 d²/τ)
                   Can crystallize if nucleation occurs

φ ≈ 0.545:         FCC crystallization complete (equilibrium)
                   Ordered crystal, low entropy

0.545 < φ < 0.58:  Metastable supercooled fluid OR crystal
                   Timescale for crystallization ~ weeks (simulations)

φ ≈ 0.58:          GLASS TRANSITION (kinetic, not thermodynamic)
                   Dynamics arrest (D ~ 10^-10 d²/τ)
                   System falls out of equilibrium
                   "Ideal glass" (extrapolated from data)

0.58 < φ < 0.64:   Deep glass (non-equilibrium)
                   Extremely slow relaxation (τ ~ years)
                   Compression path-dependent
                   May crystallize on geological timescales

φ ≈ 0.64:          RANDOM CLOSE PACKING (RCP)
                   Disordered jammed state
                   Mechanically stable (no overlaps)
                   No further compression without overlaps

φ > 0.64:          Only achievable via crystallization
                   Ordered states: FCC, HCP
                   φ_cp = π/(3√2) ≈ 0.7405 (FCC close packing)
```

**Key transitions:**
- **Freezing (φ_f ≈ 0.494):** Thermodynamic (equilibrium)
- **Glass (φ_g ≈ 0.58):** Kinetic (non-equilibrium, extrapolated)
- **RCP (φ_rcp ≈ 0.64):** Geometric (jammed packing limit for disorder)
- **Close packing (φ_cp ≈ 0.74):** Geometric (densest possible packing)

### 11.2 Packing Fraction Calculation

**Definition:**
```
φ = (Total volume of particles) / (Box volume)
  = (N × V_sphere) / V_box
  = (N × (π/6) × d³) / (Lx × Ly × Lz)
```

**Units:** Dimensionless (volume ratio)

**Physical interpretation:**
- φ = 0.1: 10% of space occupied, 90% empty (dilute gas)
- φ = 0.5: 50% occupied, particles touching on average (liquid)
- φ = 0.74: 74% occupied, densest packing (FCC crystal)

**Comparison to other densities:**

| Measure | Formula | Hard spheres at φ=0.5 |
|---------|---------|------------------------|
| Number density | ρ = N / V_box | ρ = 0.955 / d³ |
| Packing fraction | φ = N × V_sphere / V_box | φ = 0.5 |
| Reduced density | ρ* = ρ × d³ | ρ* = 0.955 |
| Coordination number | Z ≈ contacts per particle | Z ≈ 6 |

**Packing fraction vs. free volume:**
```
Free volume fraction = 1 - φ
Available space for each particle = V_box × (1-φ) / N
                                  = V_sphere / φ - V_sphere
                                  = V_sphere × (1/φ - 1)

At φ = 0.5:  Free volume = 1.0 × V_sphere  (particle has 1× its volume free)
At φ = 0.6:  Free volume = 0.67 × V_sphere (particle has 2/3 its volume free)
At φ = 0.7:  Free volume = 0.43 × V_sphere (particle has <half its volume free)
```

As φ → φ_cp, free volume → 0, particles can barely move.

---

### 11.3 Monte Carlo Acceptance Rate

**Metropolis criterion (hard spheres):**
```
ΔE = E_new - E_old
If ΔE = 0 (no overlaps): Always accept
If ΔE = ∞ (overlaps): Always reject
```

**Simplified for hard spheres:**
```
Accept move if: No overlaps created
Reject move if: Any overlap detected
```

**Acceptance rate α:**
```
α = (Number of accepted moves) / (Number of trial moves)
```

**Dependence on move size and density:**

At low density (φ ~ 0.1):
- Small move size (d=0.01): α ≈ 0.99 (almost never overlaps)
- Large move size (d=0.50): α ≈ 0.50 (frequent overlaps)

At high density (φ ~ 0.6):
- Small move size (d=0.01): α ≈ 0.80 (still good acceptance)
- Medium move size (d=0.04): α ≈ 0.30 (optimal for sampling)
- Large move size (d=0.50): α ≈ 0.001 (almost always overlaps)

**Optimal acceptance rate:**

Literature suggests α ≈ 0.3-0.5 is optimal for hard spheres:
- α < 0.2: Too many rejections, wasting computation
- α > 0.6: Moves too small, slow exploration

**Why not α = 1.0 (always accept)?**
- Requires infinitesimal move size (d → 0)
- Infinitely many moves needed to diffuse finite distance
- Inefficient: Better to accept fewer larger moves

**Diffusion coefficient vs. move size:**
```
D ∝ <Δr²> × α
  ≈ d² × α

At α = 0.5, d = 0.04: D ∝ 0.04² × 0.5 = 0.0008
At α = 0.99, d = 0.01: D ∝ 0.01² × 0.99 = 0.000099

Factor of 8x faster with larger moves despite lower acceptance!
```

**Tuning rule of thumb:**
```
If α < 0.2: Halve move size (d → d/2)
If α > 0.6: Increase move size (d → 1.5d)
Iterate until 0.2 < α < 0.6
```

---

### 11.4 Equilibration & Relaxation Time

**Definitions:**

**Relaxation time τ:**
Time for system to "forget" its initial configuration. Measured by:
```
Autocorrelation function:
C(t) = <f(0) × f(t)> / <f²>

τ = integral of C(t) from 0 to ∞
```

Common choices for f:
- Particle positions: Self-intermediate scattering function F_s(k, t)
- Bond-orientational order: Q6(t)
- Overlap function: q(t)

**Equilibration:**
Running simulation until system reaches equilibrium:
- Thermodynamic observables converge (φ, P, E)
- Time correlation functions decay to zero
- System explores configuration space uniformly

**Equilibration time vs. density:**

| Packing fraction | τ (MC sweeps) | Physical timescale |
|------------------|---------------|---------------------|
| φ = 0.1 | 10² | Fast |
| φ = 0.4 | 10³ | Moderate |
| φ = 0.5 | 10⁴ | Slow |
| φ = 0.55 | 10⁵ | Very slow |
| φ = 0.58 | 10⁶-10⁷ | Extremely slow (glass) |
| φ = 0.64 | >10¹⁰ | Never equilibrates (jammed) |

**Why does τ diverge near glass transition?**

Dynamical slowing down:
- Free volume decreases → Caging effect
- Particles trapped by neighbors (cages)
- Must wait for cage to fluctuate to escape
- Cage rearrangement requires collective motion
- τ_escape ~ exp(A / (φ_c - φ)) (Vogel-Fulcher-Tammann law)

**Practical implication:**

To equilibrate at φ = 0.58:
- Need > 10⁶ MC sweeps
- For N = 10,000: > 10¹⁰ MC trial moves
- At 10⁶ moves/second: ~3 hours

Hence: Use long `run_length_to_relax` (10,000-50,000 sweeps) at high density.

**How to check if equilibrated?**

1. **Monitor quantities vs. time:**
   ```python
   # Extract packing fraction from trajectory
   phi_vs_time = [frame.log['compression/packing_fraction'][0] for frame in traj]
   plt.plot(phi_vs_time)
   # Should plateau (not still changing)
   ```

2. **Check mean squared displacement (MSD):**
   ```python
   # Compute <|r(t) - r(0)|²>
   # Should show diffusive regime: MSD ~ t
   # If MSD plateaus: System is caged (not equilibrated)
   ```

3. **Acceptance rate stabilized:**
   - If acceptance still decreasing: System compressing further
   - If acceptance stable: Local density equilibrated

**Rule of thumb for compression script:**
```
run_length_to_relax should be ≥ τ at final density

At φ_target = 0.58: Use run_length_to_relax ≥ 50,000
At φ_target = 0.50: Use run_length_to_relax ≥ 5,000
At φ_target = 0.40: Use run_length_to_relax ≥ 1,000
```

---

## 12. Limitations & Scope

### 12.1 What This Script Does NOT Do

#### 1. Adaptive Move Size Tuning

**Not implemented:**
- Automatic adjustment of `move_size_translation` during run
- Monitoring acceptance rate and adjusting `d` accordingly
- HOOMD's `hoomd.hpmc.tune.MoveSize` tuner

**Reason:**
- Faithful port of v2 methodology (fixed move size)
- Reproducibility: Same parameters → same trajectory
- User control: Explicit parameter selection

**Alternative:**
- Use multi-stage protocol with manually adjusted move sizes per stage
- Run trial compression, check final acceptance, adjust and re-run

#### 2. Polydisperse Compression

**Not supported:**
- Bidisperse mixtures (two particle sizes)
- Continuous size distributions
- Particle diameter varies by particle

**Why:**
- Packing fraction formula assumes monodisperse: φ = N × V_sphere / V_box
- For polydisperse: φ = Σ_i V_sphere(d_i) / V_box (more complex)
- Would require:
  - Reading all diameters from GSD (not just first particle)
  - Summing individual particle volumes
  - Different HPMC shape definitions per type

**Workaround:**
- Modify script to read `params.particles.diameter` array
- Calculate φ = sum(π/6 × d_i³) / V_box
- Define multiple HPMC shape types if needed

#### 3. Anisotropic Compression

**Not supported:**
- Uniaxial compression (compress only in z direction)
- Biaxial compression (compress x and y, fix z)
- Shear during compression

**Current implementation:**
- Isotropic scaling: Lx = Ly = Lz × scale_factor
- Aspect ratio preserved: Lx/Ly/Lz constant

**To add anisotropic compression:**
- Replace `resize_box_to_volume()` with custom box resizing
- Scale dimensions independently: Lx → Lx × s_x, Ly → Ly × s_y, Lz → Lz × s_z
- Update particle positions with anisotropic scaling

#### 4. Soft Potentials

**HPMC is purely geometric:**
- No force calculations
- No energy minimization
- No Lennard-Jones, WCA, Yukawa, etc.

**For soft spheres:**
- Use `hoomd.md` (Molecular Dynamics) integrator instead
- Define pair potential (e.g., `hoomd.md.pair.LJ`)
- Integrate equations of motion (Langevin, NVE, NVT)
- Very different algorithm (not ported from v2)

#### 5. Detailed Balance Diagnostics

**Not implemented:**
- Explicit checks that HPMC satisfies detailed balance
- Measurement of entropy production
- Validation of equilibrium sampling

**Assumption:**
- HOOMD's HPMC integrator is correct (well-tested)
- Metropolis criterion ensures detailed balance
- No diagnostics needed in production runs

**If needed:**
- Add custom analysis: Forward/reverse transition rates
- Check symmetry: P(A→B) × P_eq(A) = P(B→A) × P_eq(B)

#### 6. Automatic Restarts

**Not implemented:**
- Script does not self-restart on failure/walltime
- User must manually re-invoke script to resume

**Reason:**
- Job scheduling handled by cluster scheduler (SLURM, PBS)
- Restart logic already present (detects `restart.gsd`)
- Simple re-submission: `sbatch submit_script.sh`

**For automatic chaining:**
- Use job dependencies in scheduler:
  ```bash
  JOB1=$(sbatch job1.sh)
  sbatch --dependency=afterany:$JOB1 job2.sh
  ```

#### 7. POS File Output (v2 Legacy)

**Not implemented in v4:**
- `dump.pos()` writer (HOOMD v2 format)
- POS files are plaintext, deprecated
- v4 uses GSD exclusively

**JSON fields ignored:**
- `traj_out_freq_POS`: Accepted but not used
- `color_code_for_POS_file`: Accepted but not used

**Workaround if POS needed:**
- Write separate converter: GSD → POS
- Read GSD trajectory, write positions to text file

#### 8. Energy/Pressure Calculations

**Hard spheres have no energy:**
- E = 0 (no overlaps) or ∞ (overlaps)
- No potential energy to minimize
- No force calculations

**Pressure via virial theorem not implemented:**
- P = ρkT × (1 + Z/3) (hard-sphere equation of state)
- Requires tracking collision rate (not logged)
- Can compute post-hoc from GSD if needed

**For pressure:**
- Use separate analysis script
- Read GSD trajectory
- Compute virial pressure or compressibility factor
- Requires additional HOOMD features not used here

---

