# HOOMD-blue v4.9 — Hard Convex-Polyhedron NPT Simulation

A production-grade Hard-Particle Monte Carlo (HPMC) simulation framework for
running constant-pressure (NPT) ensembles of **hard convex polyhedra** using
[HOOMD-blue v4](https://hoomd-blue.readthedocs.io/en/v4.9.0/).
The code is designed for both interactive workstations and large-scale HPC
clusters via MPI, and ships with a fully-annotated JSON parameter file and a
ready-to-use unit-cube shape definition.

---

## Table of Contents

1. [Overview](#overview)
2. [Physics Background](#physics-background)
3. [File Inventory](#file-inventory)
4. [Dependencies and Installation](#dependencies-and-installation)
5. [Quick Start](#quick-start)
6. [Input Files in Detail](#input-files-in-detail)
   - [Simulation Parameter JSON](#simulation-parameter-json)
   - [Shape JSON](#shape-json)
   - [Input GSD](#input-gsd)
7. [Parameter Reference](#parameter-reference)
   - [I/O and Run Identity](#io-and-run-identity)
   - [Stage Control](#stage-control)
   - [Shape Definition](#shape-definition)
   - [Run Lengths](#run-lengths)
   - [Output Frequencies](#output-frequencies)
   - [HPMC Particle Moves](#hpmc-particle-moves)
   - [BoxMC NPT Parameters](#boxmc-npt-parameters)
   - [BoxMC Initial Move Deltas](#boxmc-initial-move-deltas)
   - [BoxMCMoveSize Tuner Caps](#boxmcmovesize-tuner-caps)
   - [SDF Pressure Compute](#sdf-pressure-compute)
   - [Hardware Selection](#hardware-selection)
   - [Output Filenames (single-stage mode)](#output-filenames-single-stage-mode)
8. [Output Files](#output-files)
9. [Simulation Algorithm](#simulation-algorithm)
   - [Two-Phase Architecture](#two-phase-architecture)
   - [Equilibration Loop](#equilibration-loop)
   - [Production Run](#production-run)
   - [SDF Pressure Attachment](#sdf-pressure-attachment)
10. [Multi-Stage Pipeline Mode](#multi-stage-pipeline-mode)
11. [MPI Parallel Execution](#mpi-parallel-execution)
12. [Restart and Crash Recovery](#restart-and-crash-recovery)
13. [Code Architecture](#code-architecture)
    - [Section Map](#section-map)
    - [Custom Loggable Classes](#custom-loggable-classes)
    - [MPI-Safe Snapshot Loading](#mpi-safe-snapshot-loading)
    - [Random Seed Management](#random-seed-management)
14. [Analysing the Output](#analysing-the-output)
    - [Reading the Table Log](#reading-the-table-log)
    - [Reading the Scalar GSD Log](#reading-the-scalar-gsd-log)
    - [Reading the Trajectory GSD](#reading-the-trajectory-gsd)
    - [EOS Cross-Check via SDF](#eos-cross-check-via-sdf)
15. [Hard-Cube Example Run](#hard-cube-example-run)
16. [Tuning Guide](#tuning-guide)
    - [Choosing equil_steps](#choosing-equil_steps)
    - [Choosing Initial Move Sizes](#choosing-initial-move-sizes)
    - [Choosing betaP](#choosing-betap)
    - [Choosing sdf_xmax and sdf_dx](#choosing-sdf_xmax-and-sdf_dx)
17. [Adding a New Polyhedron Shape](#adding-a-new-polyhedron-shape)
18. [Error Messages and Troubleshooting](#error-messages-and-troubleshooting)
19. [Known Limitations](#known-limitations)
20. [References](#references)

---

## Overview

This project implements a full NPT HPMC workflow for **any convex polyhedron**
whose vertices can be described in a JSON file. Given an initial particle
configuration (in GSD format) and a target reduced pressure `βP = P/(kBT)`,
the simulation:

- Equilibrates the simulation box at the target pressure using an adaptive
  `BoxMCMoveSize` tuner that automatically determines optimal box-move step sizes.
- Tunes particle translational and rotational acceptance rates via a `MoveSize`
  tuner that remains active throughout production.
- Runs a production phase with fixed box-move sizes to collect statistically
  valid ensemble averages.
- Computes the instantaneous pressure via the Scale Distribution Function (SDF)
  method and logs it for equation-of-state cross-checking.
- Writes trajectory GSD, restart GSD, tabular text logs, and a machine-readable
  summary JSON on completion.

The parameter file (`simulparam_hard_polyhedra_npt.json`) shipped with the code
is configured for **N = 2197 hard cubes** at a target `βP = 50`, starting from
a pre-equilibrated NVT configuration at packing fraction φ ≈ 0.58.

---

## Physics Background

### Hard-Particle Monte Carlo (HPMC)

HPMC simulates systems of hard (athermal) particles where the pair potential is
purely repulsive:

```
U(r) = 0    if particles do not overlap
U(r) = ∞   if particles overlap
```

The Metropolis acceptance criterion therefore simplifies to: accept any trial
move that produces no overlaps; reject all moves that produce overlaps. There is
no energy evaluation — only geometry. This makes HPMC extremely efficient for
hard-core systems.

Each **MC sweep** attempts `N` trial moves (one per particle on average), where
each trial randomly proposes either a translation or a rotation:

- **Translation:** `r_new = r_old + d·ξ`, where `ξ` is a uniform random vector
  on the sphere of radius 1 and `d` (`mc.d["A"]`) is the maximum displacement.
- **Rotation:** A random quaternion rotation by at most `a` (`mc.a["A"]`) radians
  on the unit quaternion sphere is applied to the particle orientation.

### NPT Ensemble via BoxMC

The NPT (constant particle number N, pressure P, temperature T) ensemble is
implemented using `hoomd.hpmc.update.BoxMC`, which proposes trial changes to
the simulation box. Trial box changes are accepted/rejected via the NPT
Metropolis criterion:

```
P_acc = min(1, exp( −βP·ΔV + N·ln(V_new/V_old) ))
```

where `βP = P/(kBT)` is the dimensionless reduced pressure. Four orthogonal
box-move types are enabled:

| Move type | What changes | Key parameter |
|-----------|-------------|---------------|
| **Volume** | Isotropic V → V + ΔV | `boxmc_volume_delta` |
| **Length** | Independent ΔLx, ΔLy, ΔLz | `boxmc_length_delta` |
| **Aspect** | Rescale one axis at constant V | `boxmc_aspect_delta` |
| **Shear** | Change tilt factors xy, xz, yz | `boxmc_shear_delta` |

Volume moves drive the system to the target pressure. Length and aspect moves
allow the box to relax to a non-cubic equilibrium shape (important for crystal
phases). Shear moves allow triclinic box deformations, necessary when simulating
phases such as the simple-cubic or body-centred-cubic crystals of hard cubes.

### Reduced Units

All length quantities are in units of the simulation length unit σ (conventionally
the particle edge length or diameter). The reduced pressure is:

```
βP* = P σ³ / (kBT)
```

For hard cubes with unit volume (`V_particle = 1.0` and `shape_scale = 1.0`),
σ = 1 and the particle volume equals 1.0, so the packing fraction is:

```
φ = N · V_particle / V_box = N / V_box
```

At `βP = 50` and N = 2197, the equilibrium packing fraction for hard cubes is
approximately φ ≈ 0.50–0.55 depending on phase.

---

## File Inventory

```
.
├── HOOMD_hard_polyhedra_NPT.py                  # Main simulation script
├── simulparam_hard_polyhedra_npt.json           # Example parameter file (hard cubes, βP=50)
├── shape_023_Cube_unit_volume_principal_frame.json  # Cube shape definition (unit volume)
└── README.md                                    # This file

# Required before running (not included — must be generated separately):
└── HOOMD_hard_cube_2197_nvt_hpmc_pf0p58_final.gsd  # Input configuration (N=2197 hard cubes)
```

---

## Dependencies and Installation

### Required

| Package | Version | Notes |
|---------|---------|-------|
| Python | ≥ 3.9 | f-strings, `dataclass`, `list[str]` annotations |
| [HOOMD-blue](https://hoomd-blue.readthedocs.io/en/v4.9.0/) | ≥ 4.0 | GPU support requires CUDA build |
| [GSD](https://gsd.readthedocs.io/) | ≥ 3.0 | HOOMD's native file format |
| NumPy | ≥ 1.21 | Array operations for snapshot handling |

### Optional

| Package | Version | Notes |
|---------|---------|-------|
| [mpi4py](https://mpi4py.readthedocs.io/) | ≥ 3.0 | Required for MPI-parallel runs; a serial stub is provided for single-process use |

### Conda Installation (recommended)

```bash
# Create a dedicated environment
conda create -n hoomd4 python=3.11
conda activate hoomd4

# Install HOOMD-blue and dependencies from conda-forge
conda install -c conda-forge hoomd gsd numpy

# For MPI-parallel runs (requires a working MPI installation)
conda install -c conda-forge mpi4py openmpi
```

### Pip Installation

```bash
pip install hoomd gsd numpy
# For MPI:
pip install mpi4py
```

### Verifying the Installation

```bash
python -c "import hoomd; print('HOOMD version:', hoomd.version.version)"
python -c "import gsd; print('GSD version:', gsd.version.version)"
python -c "from mpi4py import MPI; print('MPI available:', MPI.COMM_WORLD.Get_size())"
```

---

## Quick Start

### 1. Prepare an Initial Configuration

The simulation requires an input GSD file containing an overlap-free starting
configuration. Generate one using any HOOMD NVT or compression tool, or use
`freud` / `garnett` to create lattice configurations.

For the supplied example (N = 2197 hard cubes), you need a file named:
```
HOOMD_hard_cube_2197_nvt_hpmc_pf0p58_final.gsd
```

This is typically generated by running a prior NVT HPMC simulation at φ ≈ 0.58
and saving the final frame.

### 2. Run the Simulation (serial)

```bash
python HOOMD_hard_polyhedra_NPT.py \
    --simulparam_file simulparam_hard_polyhedra_npt.json
```

### 3. Run with MPI (parallel, 8 ranks)

```bash
mpirun -n 8 python HOOMD_hard_polyhedra_NPT.py \
    --simulparam_file simulparam_hard_polyhedra_npt.json
```

### 4. Run on a Slurm Cluster

```bash
#!/bin/bash
#SBATCH --job-name=hard_cube_npt
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=24:00:00

conda activate hoomd4
mpirun -n 8 python HOOMD_hard_polyhedra_NPT.py \
    --simulparam_file simulparam_hard_polyhedra_npt.json
```

### Expected Console Output

On a successful run you will see output similar to:

```
*********************************************************
HOOMD-blue version:   4.9.0
*********************************************************
hoomd.version.mpi_enabled:    True
[INFO] Running on CPU (hoomd.device.CPU)
[INFO] Loaded parameters from 'simulparam_hard_polyhedra_npt.json'
[INFO] Fresh run from: 'HOOMD_hard_cube_2197_nvt_hpmc_pf0p58_final.gsd'
[INFO] N=2197 | initial phi=0.580000
[INFO] Target betaP = 50
[INFO] Overlap check PASSED: initial overlap count = 0
[INFO] EQUILIBRATION: 5000000 steps | checking every 2500 steps | max 2000 chunks
  [EQUIL chunk   1/2000] step=      2500 | phi=0.57831 | box_tuner.tuned=False | d=0.04500 | a=0.04500
  [EQUIL chunk   2/2000] step=      5000 | phi=0.57654 | box_tuner.tuned=False | d=0.04612 | a=0.04587
  ...
  [EQUIL chunk 247/2000] step=    617500 | phi=0.53210 | box_tuner.tuned=True  | d=0.04823 | a=0.04911
[INFO] BoxMCMoveSize tuner converged and removed at step 617500.
[INFO] SDF compute attached at step 617500.
[INFO] Production: 45000000 steps starting at step 617500
...
[INFO] Production complete at step 50617500 | overlaps=0
[OUTPUT] Final GSD    => HOOMD_hard_cube_2197_hpmc_npt_P50_final.gsd
[OUTPUT] Summary JSON => HOOMD_hard_cube_2197_hpmc_npt_P50_stage-1_npt_summary.json
```

---

## Input Files in Detail

### Simulation Parameter JSON

`simulparam_hard_polyhedra_npt.json` is a single flat JSON object. Keys
starting with `_` (e.g. `"_section_io"`, `"_comment_stage"`) are documentation
annotations and are silently stripped before parsing — they have no effect on
the simulation. This convention allows you to freely annotate the file without
causing validation errors.

The full schema is described in the [Parameter Reference](#parameter-reference)
section below.

### Shape JSON

`shape_023_Cube_unit_volume_principal_frame.json` defines the convex polyhedron
for HOOMD's HPMC integrator. The script reads two fields (with flexible key-suffix
matching):

| Field | Matched by suffix | Value (cube) | Description |
|-------|------------------|--------------|-------------|
| `8_vertices` | `_vertices` | 8 × [x,y,z] | Vertex coordinates in the body frame, centered at origin |
| `4_volume` | `_volume` | 1.0 | Volume of the reference (unscaled) polyhedron |

The cube is defined with vertices at (±0.5, ±0.5, ±0.5) so its edge length is
1.0 and its volume is exactly 1.0. All vertices are in the **principal frame**
(eigenvectors of the moment-of-inertia tensor), ensuring the particle orientation
quaternions stored in the GSD are physically meaningful.

**Key-suffix matching:** The script searches for any key whose name exactly
equals `vertices` or ends with `_vertices` (e.g. `polyhedron_vertices`,
`8_vertices`). The same rule applies to `volume`. This makes the script
compatible with various shape-library conventions without requiring fixed key names.

### Input GSD

The input GSD file must contain at least one frame with the following particle
data:

| Field | Shape | dtype | Description |
|-------|-------|-------|-------------|
| `particles.position` | (N, 3) | float64 | Particle positions in box-fractional or absolute coordinates |
| `particles.orientation` | (N, 4) | float64 | Unit quaternions (w, x, y, z) |
| `particles.typeid` | (N,) | int32 | Type indices (all 0 for a single-component system) |
| `particles.types` | list | str | Type labels (e.g. `["A"]`) |
| `configuration.box` | [6] | float64 | [Lx, Ly, Lz, xy, xz, yz, ...] |

The **last frame** of the input GSD is always used as the starting configuration,
making it safe to pass in a multi-frame trajectory GSD (e.g. from a prior
run's trajectory file).

The script performs an initial overlap check (`sim.run(0)`) immediately after
loading the configuration. The run will abort with a `[FATAL]` error if any
overlaps are detected, since a non-zero starting overlap count indicates a
physically invalid configuration that cannot be simulated.

---

## Parameter Reference

All parameters are set in the JSON parameter file. Required parameters must be
present; optional parameters fall back to the defaults shown.

### I/O and Run Identity

| Key | Type | Required | Example | Description |
|-----|------|----------|---------|-------------|
| `tag` | str | ✓ | `"HOOMD_hard_cube_2197_hpmc_npt_P50"` | Human-readable label. Used as a prefix for all output files in multi-stage mode and for the summary JSON filename. |
| `input_gsd_filename` | str | ✓ | `"...nvt_final.gsd"` | Path to the starting configuration GSD. In multi-stage mode with `stage_id > 0`, this is automatically overridden by the previous stage's final GSD. |

### Stage Control

| Key | Type | Required | Default | Example | Description |
|-----|------|----------|---------|---------|-------------|
| `stage_id_current` | int | ✓ | — | `-1` | `-1` = single-stage (use explicit filenames from JSON); `0`, `1`, `2`, ... = multi-stage pipeline (all filenames auto-prefixed). See [Multi-Stage Pipeline Mode](#multi-stage-pipeline-mode). |
| `initial_timestep` | int | — | `0` | `0` | Starting timestep for fresh runs. Restart runs continue from the checkpoint's timestep. |

### Shape Definition

| Key | Type | Required | Default | Example | Description |
|-----|------|----------|---------|---------|-------------|
| `shape_json_filename` | str | ✓ | — | `"shape_023_Cube_unit_volume_principal_frame.json"` | Path to the shape JSON containing vertex coordinates and reference volume. |
| `shape_scale` | float | ✓ | — | `1.0` | Linear scale factor applied to all vertices. `particle_volume = reference_volume × shape_scale³`. Use `1.0` to keep the reference shape unchanged. |

### Run Lengths

| Key | Type | Required | Default | Example | Description |
|-----|------|----------|---------|---------|-------------|
| `total_num_timesteps` | int | ✓ | — | `50000000` | Total MC sweeps including both equilibration and production. `production_steps = total_num_timesteps − equil_steps`. |
| `equil_steps` | int | ✓ | — | `5000000` | Sweeps reserved for equilibration. Must be strictly less than `total_num_timesteps`. |
| `equil_steps_check_freq` | int | ✓ | — | `2500` | Chunk size for the equilibration polling loop. After every `equil_steps_check_freq` sweeps, the loop checks whether `box_tuner.tuned` is True and exits early if so. Should divide `equil_steps` evenly for predictable total step counts. |

### Output Frequencies

All values are in MC sweeps (timesteps).

| Key | Type | Required | Default | Example | Description |
|-----|------|----------|---------|---------|-------------|
| `log_frequency` | int | ✓ | — | `5000` | Trigger period for the main Table log (`npt_hpmc_log.log`), box Table log, and scalar GSD log. |
| `traj_gsd_frequency` | int | ✓ | — | `50000` | Trigger period for appending a full particle frame to the trajectory GSD. |
| `restart_gsd_frequency` | int | ✓ | — | `5000` | Trigger period for overwriting the single-frame restart GSD. Smaller values reduce data loss on crash at the cost of I/O overhead. |

### HPMC Particle Moves

| Key | Type | Required | Default | Example | Description |
|-----|------|----------|---------|---------|-------------|
| `move_size_translation` | float | ✓ | — | `0.045` | Initial translational move size `d` (simulation length units). The maximum displacement tried per move is `d`. Tuned adaptively toward `target_particle_trans_move_acc_rate`. |
| `move_size_rotation` | float | ✓ | — | `0.045` | Initial rotational move size `a` (quaternion rotation angle, radians). Tuned adaptively. |
| `trans_move_size_tuner_freq` | int | ✓ | — | `1000` | Translational `MoveSize` tuner trigger period (sweeps). |
| `rot_move_size_tuner_freq` | int | ✓ | — | `1000` | Rotational `MoveSize` tuner trigger period (sweeps). |
| `target_particle_trans_move_acc_rate` | float | ✓ | — | `0.3` | Target translational acceptance fraction in (0, 1). Typical range 0.2–0.4. |
| `target_particle_rot_move_acc_rate` | float | ✓ | — | `0.3` | Target rotational acceptance fraction in (0, 1). |
| `max_translation_move` | float | — | `0.2` | `0.2` | Hard cap on `d` imposed by the `MoveSize` tuner. Prevents excessively large displacements at low density. |
| `max_rotation_move` | float | — | `0.5` | `0.5` | Hard cap on `a` imposed by the `MoveSize` tuner. |

**Note on the MoveSize tuner:** Unlike the BoxMCMoveSize tuner, the particle
MoveSize tuners are **never removed**. They remain active during production so
that `d` and `a` can track the slowly evolving acceptance rate as the box
volume fluctuates in the NPT ensemble. Removing them during production would
cause the acceptance rate to drift as density changes.

### BoxMC NPT Parameters

| Key | Type | Required | Default | Example | Description |
|-----|------|----------|---------|---------|-------------|
| `npt_freq` | int | ✓ | — | `10` | BoxMC trigger period (sweeps). A value of 10 means one box-move attempt for every 10 particle-move sweeps. Smaller values = more frequent box moves. |
| `pressure` | float | ✓ | — | `50` | Target reduced pressure `βP = P/(kBT)`. Must be > 0. In reduced units (σ=1, kBT=1), `βP = 50` for hard cubes corresponds to a dense fluid or crystal phase. |
| `box_tuner_freq` | int | ✓ | — | `2000` | `BoxMCMoveSize` tuner trigger period. Firing less often (larger value) gives more moves between tuning events for more reliable acceptance statistics. |
| `target_box_movement_acc_rate` | float | ✓ | — | `0.3` | Target acceptance fraction for all box move types. In (0, 1). |

### BoxMC Initial Move Deltas

These are the **starting values** for the box-move step sizes. The
`BoxMCMoveSize` tuner adapts them automatically during equilibration toward
`target_box_movement_acc_rate`.

| Key | Type | Default | Example | Description |
|-----|------|---------|---------|-------------|
| `boxmc_volume_delta` | float | `0.1` | `0.1` | Max fractional volume change per volume move. |
| `boxmc_volume_mode` | str | `"standard"` | `"standard"` | `"standard"` = linear ΔV; `"ln"` = logarithmic ΔV (better numerical stability at high density). |
| `boxmc_length_delta` | float | `0.01` | `0.01` | Max length change per independent-length move (σ units). Applied equally to x, y, z. |
| `boxmc_aspect_delta` | float | `0.02` | `0.02` | Max aspect-ratio scaling per aspect move. |
| `boxmc_shear_delta` | float | `0.01` | `0.01` | Max tilt-factor change per shear move. Applied equally to xy, xz, yz. |

### BoxMCMoveSize Tuner Caps

Upper bounds on the delta values that the `BoxMCMoveSize` tuner can set.
These prevent the tuner from proposing unstably large box changes.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_move_volume` | float | `0.1` | Cap on volume delta. |
| `max_move_length` | float | `0.05` | Cap on per-axis length delta. |
| `max_move_aspect` | float | `0.02` | Cap on aspect delta. |
| `max_move_shear` | float | `0.02` | Cap on shear delta (xy, xz, yz). |

### SDF Pressure Compute

The Scale Distribution Function (SDF) compute provides an equation-of-state
cross-check during production. It is attached only after equilibration.

| Key | Type | Default | Example | Description |
|-----|------|---------|---------|-------------|
| `enable_sdf` | bool | `true` | `true` | Attach `hoomd.hpmc.compute.SDF` after equilibration. Logs `betaP` and `Z = βP/ρ`. |
| `sdf_xmax` | float | `0.02` | `0.02` | Upper limit of the SDF histogram. Use `0.02` for φ < 0.58; use `0.005` for very dense systems (φ > 0.58) where particles are near contact. |
| `sdf_dx` | float | `1e-4` | `1e-4` | Histogram bin width. `1e-4` gives ~200 bins for `xmax=0.02`, which is sufficient for accurate extrapolation. Use `1e-5` near close-packing. |

### Hardware Selection

| Key | Type | Required | Default | Description |
|-----|------|----------|---------|-------------|
| `use_gpu` | bool | ✓ | — | `true` attempts GPU initialisation; falls back to CPU if the GPU is unavailable. |
| `gpu_id` | int | ✓ | — | CUDA device index (0-based). Only used when `use_gpu = true`. |

### Output Filenames (single-stage mode)

These keys are **only used when `stage_id_current = -1`**. In multi-stage mode
all filenames are auto-generated as `<tag>_<stage_id>_<suffix>`.

| Key | Default | Example | Description |
|-----|---------|---------|-------------|
| `output_trajectory` | `"npt_hpmc_output_traj.gsd"` | — | Trajectory GSD (append mode). |
| `simulation_log_filename` | `"npt_hpmc_log.log"` | — | Main Table log. |
| `box_log_filename` | `"box_npt_log.log"` | — | Box geometry Table log. |
| `scalar_gsd_log_filename` | `"npt_hpmc_scalar_log.gsd"` | `"HOOMD_hard_cube_2197_hpmc_npt_P50_scalar_log.gsd"` | Scalar-only GSD log (no particle data). |
| `restart_file` | `"npt_hpmc_restart.gsd"` | `"HOOMD_hard_cube_2197_hpmc_npt_P50_restart.gsd"` | Single-frame restart checkpoint. |
| `final_gsd_filename` | `"npt_hpmc_final.gsd"` | `"HOOMD_hard_cube_2197_hpmc_npt_P50_final.gsd"` | Final configuration GSD. |

---

## Output Files

A complete run produces the following files in the working directory:

| File | Frequency | Description |
|------|-----------|-------------|
| `<output_trajectory>` | every `traj_gsd_frequency` sweeps | Full per-particle GSD trajectory (positions, orientations, box). Opened in **append** mode — safe to restart. |
| `<restart_file>` | every `restart_gsd_frequency` sweeps | Single-frame GSD checkpoint. Always overwritten (truncated). Use to resume after a crash or walltime limit. |
| `<final_gsd_filename>` | once, at run end | Final configuration GSD. Suitable as input to the next pipeline stage. Contains `type_shapes` for OVITO visualisation. |
| `<simulation_log_filename>` | every `log_frequency` sweeps | Human-readable Table log. Tab-separated columns: timestep, TPS, walltime, ETR, phi, volume, acceptance rates, move sizes, overlap count, SDF betaP (when enabled). |
| `<box_log_filename>` | every `log_frequency` sweeps | Box geometry Table log. Columns: timestep, Lx, Ly, Lz, xy, xz, yz, volume, phi, cumulative BoxMC move counts. |
| `<scalar_gsd_log_filename>` | every `log_frequency` sweeps | Scalar-only GSD log (no particle data). Very compact (~kilobytes per million steps). Analysable with `gsd.hoomd` without loading trajectory frames. |
| `<tag>_stage<id>_npt_summary.json` | once, at run end | Machine-readable provenance JSON: all key parameters, final box geometry, packing fraction, overlap count, runtime. |
| `random_seed.json` | once, on first run | Persistent RNG seed file. Re-used on restarts and later pipeline stages for reproducibility. |
| `emergency_restart_<tag>.gsd` | on crash only | Emergency snapshot written if the simulation crashes. Preserves the last known configuration. |

### Main Table Log Columns

The `npt_hpmc_log.log` file has the following columns (exact header depends on
HOOMD version):

| Column | Type | Description |
|--------|------|-------------|
| `timestep` | int | Current MC sweep |
| `tps` | float | MC sweeps per wall-clock second |
| `walltime` | float | Elapsed wall-clock seconds |
| `Status/etr` | str | Estimated time remaining (HH:MM:SS) |
| `Status/timestep` | str | `current/final` step fraction |
| `MCStatus/trans_acc_rate` | float | Windowed translational acceptance rate |
| `MCStatus/rot_acc_rate` | float | Windowed rotational acceptance rate |
| `MoveSize/d` | float | Current translational move size |
| `MoveSize/a` | float | Current rotational move size |
| `Box/volume` | float | Current box volume |
| `Box/phi` | float | Current packing fraction φ |
| `HPMC/overlaps` | float | Overlap count (must be 0) |
| `BoxMCStatus/acc_rate` | float | Combined box-move acceptance rate |
| `BoxMCStatus/volume_acc_rate` | float | Volume+length move acceptance rate |
| `BoxMCStatus/aspect_acc_rate` | float | Aspect move acceptance rate |
| `BoxMCStatus/shear_acc_rate` | float | Shear move acceptance rate |
| `SDF/betaP` | float | SDF-measured βP (production only; 0 during equil) |
| `SDF/compressibility_Z` | float | Z = βP/ρ (production only) |

---

## Simulation Algorithm

### Two-Phase Architecture

The simulation executes in two phases that are architecturally distinct:

```
Phase 1: EQUILIBRATION
─────────────────────────────────────────────
 All tuners active:
   • MoveSize tuner (translation)    ──────────────────────────────┐
   • MoveSize tuner (rotation)       ──────────────────────────────┤  Continue into
   • BoxMCMoveSize tuner  ──────────────────────── removed here    │  production
                                                ↓                  │
 Runs in chunks of equil_steps_check_freq                          │
 After each chunk: poll box_tuner.tuned                            │
   • True  → remove BoxMCMoveSize tuner, BREAK                     │
   • False → continue (up to equil_steps total)                    │
                                                                   │
Phase 2: PRODUCTION                                                │
─────────────────────────────────────────────                      │
 Box-move deltas FIXED (BoxMCMoveSize tuner removed)               │
 MoveSize tuners still active ─────────────────────────────────────┘
 SDF pressure compute ATTACHED
 sim.run(total_steps − equil_steps)   (single call)
```

### Equilibration Loop

```python
n_chunks = equil_steps // equil_steps_check_freq

for chunk_idx in range(n_chunks):
    sim.run(equil_steps_check_freq)
    if box_tuner.tuned:
        sim.operations.tuners.remove(box_tuner)
        break   # early exit
```

Each `sim.run(equil_steps_check_freq)` call internally executes all attached
operations (integrator, BoxMC updater, MoveSize tuners, BoxMCMoveSize tuner,
writers) on their respective `Periodic` triggers. No manual tuner calls are needed.

The `BoxMCMoveSize` tuner declares `tuned = True` when **all** of the following
conditions hold simultaneously:
- All 8 box-move types (`volume`, `length_x`, `length_y`, `length_z`,
  `aspect`, `shear_x`, `shear_y`, `shear_z`) have acceptance rates within
  ±`tol` (= 0.03) of `target_box_movement_acc_rate`.

**What happens if the tuner does not converge?**

If the tuner has not converged after all `n_chunks` equilibration chunks, a
`[WARNING]` is printed and production runs with the `BoxMCMoveSize` tuner still
active. The run is still physically valid but box-move sizes may not be optimal.
Increase `equil_steps` or reduce `box_tuner_freq` (fire tuner more often) to
address this.

### Production Run

Production runs as a **single** `sim.run(prod_steps)` call with:
- No `BoxMCMoveSize` tuner (box-move deltas locked).
- MoveSize tuners still active (particle move sizes continue to track acceptance rate).
- SDF pressure compute active.
- All writers (trajectory GSD, restart GSD, Table logs) firing on their triggers.

### SDF Pressure Attachment

The SDF compute is constructed during `build_simulation()` but is **not attached**
at that time. It is attached by the caller (in `main()`) immediately after the
`BoxMCMoveSize` tuner is removed and before `sim.run(prod_steps)`. This ensures:

1. The SDF histogram is accumulated only in the post-equilibration, fixed-box regime.
2. The SDF extrapolation is not contaminated by large volume fluctuations from
   the early equilibration phase.

---

## Multi-Stage Pipeline Mode

For long runs that exceed walltime limits, or for systematic pressure sweeps,
the code supports a chained multi-stage pipeline controlled by `stage_id_current`.

### How It Works

Set `stage_id_current = 0` for the first stage, `1` for the second, and so on.
All output files are automatically prefixed with `<tag>_<stage_id>_`:

```
stage 0: <tag>_0_traj.gsd, <tag>_0_sim.log, <tag>_0_final.gsd, ...
stage 1: <tag>_1_traj.gsd, <tag>_1_sim.log, <tag>_1_final.gsd, ...
```

Stage 0 reads `input_gsd_filename` from the JSON. Stages 1, 2, ... automatically
read `<tag>_<stage_id-1>_final.gsd` as input — no manual filename updates needed.

### Safety Guards

The code prevents accidental overwriting or skipping stages:

- If `<tag>_<sid>_final.gsd` **already exists**: the script exits with an error
  telling you to increment `stage_id_current` to `sid + 1`.
- If the previous stage's final GSD **does not exist**: the script exits telling
  you that stage `sid - 1` did not complete.

### Example: Three-Stage Pressure Ramp

```json
// Stage 0: equilibrate at βP = 30
{"stage_id_current": 0, "pressure": 30, "equil_steps": 5000000, ...}

// Stage 1: compress to βP = 50 starting from stage 0 output
{"stage_id_current": 1, "pressure": 50, "input_gsd_filename": "..." ...}

// Stage 2: long production at βP = 50 starting from stage 1 output
{"stage_id_current": 2, "pressure": 50, "equil_steps": 1000000, "total_num_timesteps": 100000000, ...}
```

The `input_gsd_filename` is only used for stage 0; subsequent stages chain
automatically.

---

## MPI Parallel Execution

The code is designed for transparent scaling from 1 to hundreds of MPI ranks.

### MPI-Safe Snapshot Loading

Raw GSD reads on all N ranks simultaneously would cause N concurrent reads of
the same file — unreliable on parallel filesystems (Lustre, GPFS). Instead the
code uses a **read-on-root, broadcast** pattern:

```
Rank 0: open GSD → read last frame → pack into dict → bcast(dict) → reconstruct snapshot
Rank 1–N: bcast(None) → receive dict → reconstruct snapshot
All ranks: sim.create_state_from_snapshot(snapshot)  [HOOMD decomposes internally]
```

This pattern is also used for restart runs where HOOMD's built-in
`create_state_from_gsd` handles the MPI decomposition internally.

### Console Output

All `root_print()` and `debug_kv()` calls are gated on `_is_root_rank()` so
each INFO/WARNING/DEBUG line appears exactly once regardless of the number of
MPI ranks. The rank is determined from environment variables
(`OMPI_COMM_WORLD_RANK`, `PMI_RANK`, `SLURM_PROCID`) rather than from mpi4py
so it is available even before MPI is fully initialised.

### File Writes

All file I/O (GSD writes, log file writes, seed file creation, summary JSON) is
performed exclusively on rank 0. The `_is_root_rank()` guard prevents N-fold
file conflicts.

### MPI Stub

If `mpi4py` is not installed, a transparent `_MPIStub` class is used. Its
`bcast()` method returns the input object unchanged, making all downstream
broadcast calls work identically in serial mode without any code changes.

---

## Restart and Crash Recovery

### Normal Restart (walltime limit)

The restart GSD (`<restart_file>`) is a single-frame GSD written every
`restart_gsd_frequency` sweeps with `truncate=True`, so it always contains
exactly the most recent configuration.

To resume a run after a walltime interruption, simply resubmit the same job
with the same parameter file. The script will detect the restart GSD and call
`sim.create_state_from_gsd(filename=restart_gsd)` instead of the
read-broadcast-reconstruct path:

```
Restart GSD exists AND Final GSD does NOT exist → RESTART PATH
Restart GSD does NOT exist OR Final GSD exists  → FRESH RUN PATH
```

The restart GSD records the HOOMD timestep, so the continued run picks up
exactly where it left off.

### Crash Recovery

If the simulation crashes with an unexpected exception (Python error, HOOMD
assertion, memory error, etc.), the exception handler in `main()` attempts to:

1. Print a structured diagnostic block showing the last known timestep, box
   geometry, all filenames, the MPI rank, and the seed.
2. Write an **emergency snapshot** named `emergency_restart_<tag>.gsd` using the
   last known simulation state.

The emergency snapshot can be used as an `input_gsd_filename` after fixing the
cause of the crash.

### Seed Persistence

The random seed file (`random_seed.json` or `random_seed_stage_0.json`) is
written once on the first run and **never overwritten**. All restarts and later
pipeline stages read the same seed, ensuring that the combined trajectory is
statistically equivalent to a single uninterrupted run with that seed.

---

## Code Architecture

### Section Map

The script is organised into 12 labelled sections:

| Section | Content |
|---------|---------|
| 1 | MPI-aware console helpers (`root_print`, `fail_with_context`, `debug_kv`) |
| 2 | Custom loggable classes (`Status`, `MCStatus`, `Box_property`, `MoveSizeProp`, `BoxMCStatus`, `BoxSeqProp`, `OverlapCount`, `SDFPressure`) |
| 3 | `SimulationParams` dataclass with `validate()` |
| 4 | JSON loading: `load_simulparams()`, `_REQUIRED_KEYS`, `_OPTIONAL_KEYS` |
| 5 | Seed management: `ensure_seed_file()`, `read_seed()` |
| 6 | Filename resolution: `RunFiles` dataclass, `resolve_filenames()` |
| 7 | Shape loader: `load_convex_polyhedron_shape()`, `_get_json_value_by_suffix()` |
| 8 | MPI snapshot loading: `load_and_broadcast_snapshot()`, `reconstruct_snapshot()` |
| 9 | Simulation builder: `build_simulation()` — steps 9.1–9.17 |
| 10 | Output helpers: `_write_snapshot()`, `write_final_outputs()`, `_print_banner()` |
| 11 | Entry point: `main()` — equilibration loop, production run, exception handler |
| 12 | Script guard: `if __name__ == "__main__"` with traceback handler |

### Custom Loggable Classes

HOOMD's `Logger` can register any Python object whose properties are enumerated
in a class-level `_export_dict`. The dict maps `property_name → (category, True)`.

| Class | Properties logged | Purpose |
|-------|------------------|---------|
| `Status` | `etr`, `timestep_fraction` | ETR string and step progress |
| `MCStatus` | `translate_acceptance_rate`, `rotate_acceptance_rate` | Windowed particle acceptance rates |
| `Box_property` | `L_x/y/z`, `XY/XZ/YZ`, `volume`, `packing_fraction` | Box geometry and φ |
| `MoveSizeProp` | `d`, `a` | Current move sizes |
| `BoxMCStatus` | `acceptance_rate`, `volume/aspect/shear_acc_rate` | Windowed BoxMC acceptance rates |
| `BoxSeqProp` | `volume/aspect/shear_moves_str` | Cumulative BoxMC counts as strings |
| `OverlapCount` | `overlap_count` | Live overlap check |
| `SDFPressure` | `betaP`, `compressibility_Z` | SDF pressure and Z-factor |

**Guard-property pattern:** HOOMD's Logger calls every registered property getter
**once at registration time** to validate the loggable — before any `sim.run()`
call. Therefore every property getter in this codebase guards against
`hoomd.error.DataAccessError` (HPMC counters not yet initialised) and
`ZeroDivisionError` (TPS = 0 before any sweeps), returning a safe fallback value
(typically `0.0` or `"0/?"`) rather than raising.

**Windowed acceptance rates:** HOOMD's counters (`translate_moves`,
`rotate_moves`, `volume_moves`, etc.) are cumulative totals from the start of
the run. Logging their ratio directly would produce a monotonically converging
curve that masks per-window trends. `MCStatus` and `BoxMCStatus` store the
previous counter values and compute the acceptance fraction only over the most
recent logging window (delta_accepted / delta_total).

**Cache design in `BoxMCStatus`:** During a single logging event, HOOMD may
query all four properties (`acceptance_rate`, `volume_acc_rate`,
`aspect_acc_rate`, `shear_acc_rate`) in rapid succession. The
`_refresh_cache()` method computes all rates together at most once per
`sim.timestep` value and caches them, so that all four properties return
internally consistent values from the same delta window regardless of query
order.

### MPI-Safe Snapshot Loading

The three functions involved are:

```python
snap_data = load_and_broadcast_snapshot(input_gsd, comm, rank)
# rank 0: reads GSD, validates arrays, packs into dict, broadcasts
# rank 1+: receives None, which bcast replaces with the broadcast dict

snapshot = reconstruct_snapshot(snap_data, rank)
# rank 0: validates shapes, builds hoomd.Snapshot, populates arrays
# rank 1+: returns empty hoomd.Snapshot() (HOOMD fills from rank 0)

sim.create_state_from_snapshot(snapshot)
# all ranks: HOOMD performs domain decomposition internally
```

### Random Seed Management

```
First run:
  ensure_seed_file() [rank 0] → generate secrets.randbelow(65536) → write JSON
  all ranks wait (poll with 30-second timeout) → read_seed() → pass to Simulation()

Restart or later stage:
  ensure_seed_file() [rank 0] → file exists, no-op
  read_seed() → same seed as before → reproducible trajectory
```

The seed is in [0, 65535] to fit HOOMD's uint16 RNG seed requirement.

---

## Analysing the Output

### Reading the Table Log

```python
import numpy as np

data = np.genfromtxt("npt_hpmc_log.log", names=True, delimiter="\t")
phi      = data["Box_phi"]
timestep = data["timestep"]
betaP    = data["SDF_betaP"]

import matplotlib.pyplot as plt
plt.plot(timestep, phi)
plt.xlabel("MC sweep")
plt.ylabel("Packing fraction φ")
plt.title("Hard cubes NPT φ vs step")
plt.savefig("phi_vs_step.png", dpi=150)
```

### Reading the Scalar GSD Log

```python
import gsd.hoomd
import numpy as np

with gsd.hoomd.open("HOOMD_hard_cube_2197_hpmc_npt_P50_scalar_log.gsd") as f:
    # Each frame corresponds to one log event
    phi_series    = np.array([f[i].log["Box/phi"][0]       for i in range(len(f))])
    betaP_series  = np.array([f[i].log["SDF/betaP"][0]     for i in range(len(f))])
    timesteps     = np.array([f[i].configuration.step      for i in range(len(f))])
```

### Reading the Trajectory GSD

```python
import gsd.hoomd
import numpy as np

with gsd.hoomd.open("npt_hpmc_output_traj.gsd") as traj:
    n_frames = len(traj)
    print(f"Trajectory: {n_frames} frames")
    
    # Read the last frame
    frame = traj[-1]
    positions    = frame.particles.position    # shape (N, 3)
    orientations = frame.particles.orientation # shape (N, 4), quaternion (w,x,y,z)
    box          = frame.configuration.box     # [Lx, Ly, Lz, xy, xz, yz, ...]
```

The trajectory GSD also embeds `type_shapes` in each frame, making it
self-describing for OVITO and other visualisation tools that support the GSD
format.

### EOS Cross-Check via SDF

During production, the time-averaged `<SDF/betaP>` should converge to the target
`pressure` value. Significant deviation indicates inadequate equilibration or
insufficient production length.

```python
# Load scalar log
with gsd.hoomd.open("HOOMD_hard_cube_2197_hpmc_npt_P50_scalar_log.gsd") as f:
    betaP = np.array([f[i].log["SDF/betaP"][0] for i in range(len(f))])
    step  = np.array([f[i].configuration.step  for i in range(len(f))])

# Identify production start (non-zero betaP = SDF attached = post-equil)
prod_mask = betaP > 0

print(f"Mean βP (production) = {betaP[prod_mask].mean():.2f}")
print(f"Target βP            = 50.00")
print(f"Relative error       = {abs(betaP[prod_mask].mean() - 50) / 50 * 100:.2f}%")
```

---

## Hard-Cube Example Run

The supplied parameter file runs the following experiment:

| Parameter | Value | Notes |
|-----------|-------|-------|
| System | N = 2197 hard cubes | 13³ particles |
| Target pressure | βP = 50 | Dense fluid or plastic crystal regime |
| Starting configuration | φ ≈ 0.58 (NVT equilibrated) | Expected to compress slightly |
| Initial move sizes | d = a = 0.045 σ | Conservative start; tuner adapts |
| Equilibration | 5 × 10⁶ sweeps | ~10% of total run |
| Production | 45 × 10⁶ sweeps | |
| Log frequency | 5 000 sweeps | ~9 000 log rows during production |
| Trajectory frequency | 50 000 sweeps | ~900 trajectory frames |
| SDF | xmax = 0.02, dx = 1×10⁻⁴ | ~200 histogram bins |

**Expected equilibrium packing fraction:** For hard cubes at βP ≈ 50, the
equilibrium packing fraction is approximately φ_eq ≈ 0.50–0.55. The box should
expand from the starting φ = 0.58 during equilibration.

**Expected SDF output:** After equilibration, `<SDF/betaP>` should fluctuate
around 50 with standard deviation ∝ 1/√(N × production_steps).

---

## Tuning Guide

### Choosing `equil_steps`

The equilibration should be long enough for the `BoxMCMoveSize` tuner to
converge (declare `tuned=True`). Convergence typically requires several hundred
tuning events. With `box_tuner_freq = 2000` and the tuner needing ~200–500
events to converge:

```
Minimum equil_steps ≈ 500 × box_tuner_freq = 500 × 2000 = 1,000,000
```

Add a safety margin (2–5×) to allow for slow convergence at unusual densities.
The supplied value of 5,000,000 is conservative for this system.

### Choosing Initial Move Sizes

Start with `move_size_translation ≈ 0.03–0.10 × σ` (particle length unit) and
`move_size_rotation ≈ 0.03–0.10` radians. The `MoveSize` tuner will adapt both
within a few hundred sweep windows. Values that are too large (> 0.5) can cause
many simultaneous overlaps and slow convergence.

### Choosing `betaP`

The relationship between `βP` and `φ` depends on the equation of state of your
specific polyhedron. For hard spheres the Carnahan-Starling EOS provides a good
guide. For hard cubes, the EOS is known from simulation data. In the absence of
prior data, run a short test with `total_num_timesteps = 500000` at several
`betaP` values and measure the equilibrium `φ` from the `Box/phi` column.

### Choosing `sdf_xmax` and `sdf_dx`

| φ | Recommended `sdf_xmax` | Recommended `sdf_dx` |
|---|-----------------------|--------------------|
| < 0.50 | 0.02 | 1e-4 |
| 0.50–0.58 | 0.02 | 1e-4 |
| 0.58–0.62 | 0.01 | 1e-4 |
| > 0.62 | 0.005 | 1e-5 |

At very high packing fractions (near close-packing) particles are nearly in
contact and the SDF histogram peak is compressed toward x = 0, requiring a
smaller `xmax` for accurate extrapolation.

---

## Adding a New Polyhedron Shape

1. **Create a shape JSON** with `_vertices` and `_volume` keys (or any key
   ending in those suffixes). Vertex coordinates should be in the **body frame**,
   centred at the origin. The volume should match the vertices exactly.

2. **Choose `shape_scale`**: If your vertices span the range [−L/2, L/2] and
   you want particle volume `V`, set `shape_scale = (V / reference_volume)^(1/3)`.
   For unit-volume shapes (`reference_volume = 1.0`), `shape_scale = V^(1/3)`.

3. **Update the JSON parameter file**:
   ```json
   "shape_json_filename": "shape_my_polyhedron.json",
   "shape_scale": 1.0
   ```

4. **Generate an initial configuration** using any HOOMD NVT or Frenkel-Ladd
   lattice tool, ensuring zero overlaps with the new shape definition.

5. **Verify the shape** by running 0 steps and checking `mc.type_shapes`:
   ```python
   sim.run(0)
   print(mc.type_shapes)
   ```

The script validates that vertices have shape (N_v, 3), that N_v ≥ 4 (minimum
for a tetrahedron), and that `reference_volume > 0`.

---

## Error Messages and Troubleshooting

| Error message | Cause | Fix |
|---------------|-------|-----|
| `[FATAL] Input GSD not found: '<file>'` | Input GSD file missing | Generate the input configuration or fix the `input_gsd_filename` path |
| `[FATAL] Initial convex-polyhedron configuration is NOT overlap-free` | Input GSD has overlapping particles | Re-generate the configuration; verify `shape_scale` matches the scale used to create the GSD |
| `[FATAL] Missing required keys in '...': [...]` | JSON parameter file missing required keys | Add the listed keys to the JSON |
| `[FATAL] Type errors in '...': * 'pressure': expected ..., got str` | JSON key has the wrong type (e.g. string instead of number) | Correct the value type in the JSON |
| `[FATAL] Shape JSON file not found` | Shape JSON missing | Fix `shape_json_filename` path |
| `[FATAL] Polyhedron vertices must have shape (N_vertices, 3)` | Malformed vertex array in shape JSON | Each row must be a 3-element list `[x, y, z]` |
| `[ERROR] Stage N final GSD '...' already exists` | Multi-stage: stage already completed | Increment `stage_id_current` by 1 |
| `[FATAL] Previous stage output not found` | Multi-stage: previous stage not complete | Run stage N-1 before stage N |
| `[WARNING] BoxMCMoveSize tuner did NOT converge` | Equilibration too short | Increase `equil_steps` or decrease `box_tuner_freq` |
| `[FATAL] GPU initialisation failed` (fallback to CPU) | GPU unavailable | Set `use_gpu: false` or fix CUDA installation |
| `[ERROR] Emergency snapshot written → emergency_restart_<tag>.gsd` | Simulation crashed | Inspect the traceback; fix the cause; use the emergency snapshot as `input_gsd_filename` |

---

## Known Limitations

- **Single particle type only.** The integrator is configured for a single
  particle type `"A"`. Multi-component systems require modifications to the
  shape assignment block and move-size tuner registration.

- **Convex polyhedra only.** `hoomd.hpmc.integrate.ConvexPolyhedron` cannot
  simulate non-convex or concave shapes. Use
  `hoomd.hpmc.integrate.ConvexPolyhedronUnion` for complex shapes built from
  convex pieces.

- **BoxMCMoveSize tuner convergence tolerance is hard-coded** to `tol=0.03` and
  `gamma=0.8` in `build_simulation()`. To change these, edit the call to
  `hoomd.hpmc.tune.BoxMCMoveSize.scale_solver` directly in the script.

- **SDF is disabled on non-root MPI ranks.** `sdf.betaP` returns `None` on all
  ranks except rank 0. The `SDFPressure` class guards against this but the SDF
  value in the scalar log will be 0.0 on non-root ranks.

- **No NPT shear flow.** The `reduce=0.0` setting in the shear move
  configuration disables Lees-Edwards boundary conditions. For shear-flow
  simulations, set `reduce` appropriately.

---

## References

1. Anderson, J. A., Irrgang, M. E., & Glotzer, S. C. (2016).
   Scalable Metropolis Monte Carlo for simulation of hard shapes.
   *Computer Physics Communications*, 204, 21–30.
   https://doi.org/10.1016/j.cpc.2016.02.024

2. Anderson, J. A., Jankowski, E., Grubb, T. L., Engel, M., & Glotzer, S. C.
   (2013). Massively parallel Monte Carlo for many-particle simulations on GPUs.
   *Journal of Computational Physics*, 254, 27–38.

3. Eppenga, R., & Frenkel, D. (1984).
   Monte Carlo study of the isotropic and nematic phases of infinitely thin hard platelets.
   *Molecular Physics*, 52(6), 1303–1334.
   https://doi.org/10.1080/00268978400101951
   *(Original SDF pressure estimator)*

4. HOOMD-blue v4.9 documentation.
   https://hoomd-blue.readthedocs.io/en/v4.9.0/

5. GSD file format specification.
   https://gsd.readthedocs.io/

6. Frenkel, D., & Smit, B. (2002).
   *Understanding Molecular Simulation: From Algorithms to Applications* (2nd ed.).
   Academic Press.
   *(NPT Monte Carlo algorithm, Chapter 5)*

7. Dijkstra, M., van Roij, R., & Evans, R. (1999).
   Direct simulation of the phase behavior of binary hard-sphere mixtures.
   *Physical Review Letters*, 82(1), 117.
   *(Hard polyhedra phase behaviour reference)*
