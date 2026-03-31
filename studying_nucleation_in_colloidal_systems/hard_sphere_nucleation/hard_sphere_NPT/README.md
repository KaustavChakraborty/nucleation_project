# `hs_npt_v4.py` — Hard-Sphere HPMC NPT Simulation

**HOOMD-blue v4.9 | Production NPT simulation for monodisperse hard spheres with
two-phase equilibration, adaptive tuning, SDF pressure measurement, full restart
support, and MPI parallelism**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Where This Script Fits in the Pipeline](#2-where-this-script-fits-in-the-pipeline)
3. [Physics: What NPT Means for Hard Spheres](#3-physics-what-npt-means-for-hard-spheres)
4. [The Two-Phase Run Structure](#4-the-two-phase-run-structure)
5. [Algorithm: Step-by-Step](#5-algorithm-step-by-step)
6. [File Inventory](#6-file-inventory)
7. [Dependencies](#7-dependencies)
8. [Installation](#8-installation)
9. [Quick Start](#9-quick-start)
10. [JSON Parameter Reference](#10-json-parameter-reference)
    - [Required keys](#required-keys)
    - [Optional BoxMC delta keys](#optional-boxmc-delta-keys)
    - [Optional BoxMC tuner max-move keys](#optional-boxmc-tuner-max-move-keys)
    - [Optional SDF keys](#optional-sdf-keys)
    - [Optional hardware keys](#optional-hardware-keys)
    - [Optional single-stage filename keys](#optional-single-stage-filename-keys)
    - [Comment keys](#comment-keys)
11. [Multi-Stage Runs](#11-multi-stage-runs)
12. [Output Files Reference](#12-output-files-reference)
    - [Simulation log (`*_sim.log`)](#simulation-log-_simlog)
    - [Box log (`*_box.log`)](#box-log-_boxlog)
    - [GSD trajectory (`*_traj.gsd`)](#gsd-trajectory-_trajgsd)
    - [GSD restart (`*_restart.gsd`)](#gsd-restart-_restartgsd)
    - [GSD scalar log (`*_scalars.gsd`)](#gsd-scalar-log-_scalarsgsd)
    - [Final GSD (`*_final.gsd`)](#final-gsd-_finalgsd)
    - [Summary JSON](#summary-json)
    - [Emergency snapshot](#emergency-snapshot)
13. [Restart and Recovery](#13-restart-and-recovery)
14. [MPI Parallel Execution](#14-mpi-parallel-execution)
15. [GPU Execution](#15-gpu-execution)
16. [Code Architecture: Section-by-Section](#16-code-architecture-section-by-section)
    - [Section 1: MPI Console Helpers](#section-1-mpi-console-helpers)
    - [Section 2: Custom Loggable Classes](#section-2-custom-loggable-classes)
    - [Section 3: SimulationParams Dataclass](#section-3-simulationparams-dataclass)
    - [Section 4: JSON Loading and Validation](#section-4-json-loading-and-validation)
    - [Section 5: Random Seed Management](#section-5-random-seed-management)
    - [Section 6: Filename Resolution](#section-6-filename-resolution)
    - [Section 7: GSD Diameter Reader](#section-7-gsd-diameter-reader)
    - [Section 8: MPI Snapshot Broadcast](#section-8-mpi-snapshot-broadcast)
    - [Section 9: Simulation Builder (17 substeps)](#section-9-simulation-builder-17-substeps)
    - [Section 10: Output Helpers](#section-10-output-helpers)
    - [Section 11: Entry Point](#section-11-entry-point)
17. [The BoxMC Updater: Four Move Types Explained](#17-the-boxmc-updater-four-move-types-explained)
18. [The Equilibration Loop: How the Tuner Works](#18-the-equilibration-loop-how-the-tuner-works)
19. [SDF Pressure Measurement](#19-sdf-pressure-measurement)
20. [Logged Quantities Reference](#20-logged-quantities-reference)
21. [Reading the Scalar GSD Log in Python](#21-reading-the-scalar-gsd-log-in-python)
22. [Diameter Persistence: How and Why](#22-diameter-persistence-how-and-why)
23. [Seed Reproducibility](#23-seed-reproducibility)
24. [Choosing `pressure` (betaP) and Move Deltas](#24-choosing-pressure-betap-and-move-deltas)
25. [Reading Output Files in Python](#25-reading-output-files-in-python)
26. [HPC Cluster / SLURM Job Script](#26-hpc-cluster--slurm-job-script)
27. [Exception Handling Map](#27-exception-handling-map)
28. [Troubleshooting](#28-troubleshooting)
29. [Version History and Feature Changelog](#29-version-history-and-feature-changelog)

---

## 1. Overview

`hs_npt_v4.py` runs a **constant-pressure (NPT) Hard-Particle Monte Carlo (HPMC)** simulation of a monodisperse hard-sphere system using HOOMD-blue v4.9. It is the fourth iteration of this script — adding 474 lines of detailed inline documentation to `hs_npt_v3.py` without changing a single line of algorithm or logic.

The script is designed to follow an NVT equilibration run (e.g. `hs_nvt_v4.py`) and bring the system to thermodynamic equilibrium at a specified target pressure βP = P/(k_BT), allowing the box volume to fluctuate freely. It can then be used directly as the production run from which equation-of-state measurements are extracted.

**Key characteristics at a glance:**

| Property | Value |
|---|---|
| Ensemble | NPT (constant N, P, T) |
| Integrator | `hoomd.hpmc.integrate.Sphere` |
| Box updater | `hoomd.hpmc.update.BoxMC` |
| Move types | Particle translation + volume + length + aspect + shear box moves |
| Run structure | Two phases: equilibration (adaptive tuning) + production (fixed move sizes) |
| Particle move tuner | `hoomd.hpmc.tune.MoveSize` (active throughout) |
| Box move tuner | `hoomd.hpmc.tune.BoxMCMoveSize` (removed after equilibration) |
| Pressure measurement | `hoomd.hpmc.compute.SDF` (attached after equilibration, optional) |
| MPI support | Yes (via mpi4py; serial fallback if absent) |
| GPU support | Yes (selectable via JSON; auto-falls back to CPU) |
| Restart-safe | Yes (automatic checkpoint detection) |
| Multi-stage | Yes (pipeline-compatible stage IDs) |
| Diameter in all GSD files | Yes (`dynamic=` + `write_diameter=True`) |
| Log files | 3 types: simulation Table log, box Table log, scalar GSD log |

---

## 2. Where This Script Fits in the Pipeline

The typical hard-sphere NPT research pipeline:

```
[1] make_lattice_phi.py        ← create low-density initial lattice at φ_low
         │
         ▼
[2] hs_compress_v7.py          ← compress to target φ via manual box-resize loop
         │  outputs: *_final.gsd
         ▼
[3] hs_nvt_v4.py               ← NVT equilibration: randomise structure at constant V
         │  outputs: *_final.gsd
         ▼
[4] hs_npt_v4.py      ◄──── YOU ARE HERE
         │  inputs:  *_final.gsd from NVT
         │  outputs: *_traj.gsd, *_final.gsd, *_sim.log, *_box.log, *_scalars.gsd
         ▼
[5] EOS analysis               ← extract Z(φ) from scalar GSD log via SDF betaP
    or
[5] Phase equilibrium study    ← run at several pressures straddling coexistence
    or
[5] Crystal/fluid structure    ← analyse g(r), S(q) from trajectory GSD
```

The NPT stage is the primary production run for equation-of-state work: by targeting a specific βP you allow the simulation box to find its equilibrium volume, from which the equilibrium packing fraction φ and compressibility factor Z = βPV/N are extracted.

---

## 3. Physics: What NPT Means for Hard Spheres

In HPMC, "temperature" is not a free parameter — the hard-sphere Boltzmann factor is either 0 (overlap) or 1 (no overlap). **"NPT" here means:**

- **N** (number of particles): fixed, read from the input GSD.
- **P** (pressure): fixed at `pressure` = βP = P/(k_BT). This is the reduced pressure in units where σ = 1 and k_BT = 1. Volume fluctuates until the virial pressure equals the target.
- **T** (temperature): implicit; k_BT = 1 in HOOMD hard-sphere units.

The NPT Metropolis criterion for a **box trial move** that changes volume V → V' is:

```
acc. prob. = min(1,  exp(−βP·ΔV + N·ln(V'/V)) )
```

The `−βP·ΔV` term is the mechanical P·ΔV work, and `N·ln(V'/V)` is the Jacobian accounting for the fact that particle positions are scaled with the box. For an isotropic volume move this reduces to the standard NPT criterion; for length, aspect, and shear moves, HOOMD's BoxMC handles the generalised criterion automatically.

Hard-sphere NPT is used to:
- Locate the equation of state Z(φ) by running at many pressures.
- Find coexistence pressures where fluid and crystal have equal chemical potential.
- Characterise the FCC crystal structure at high pressure.
- Generate equilibrium configurations at a specific thermodynamic state point.

---

## 4. The Two-Phase Run Structure

The script divides `total_num_timesteps` into two contiguous phases:

```
total_num_timesteps = equil_steps + prod_steps

Phase 1: EQUILIBRATION  (equil_steps steps)
  ├─ Both MoveSize tuner AND BoxMCMoveSize tuner are active
  ├─ Run in chunks of equil_steps_check_freq
  ├─ After each chunk: check box_tuner.tuned
  ├─ If tuned: remove BoxMCMoveSize tuner, break out of loop early
  └─ If never tuned: warn user, proceed to production with tuner still active

Phase 2: PRODUCTION  (prod_steps = total_num_timesteps - equil_steps steps)
  ├─ BoxMCMoveSize tuner removed (move sizes are now fixed)
  ├─ MoveSize tuner still active (adapts d as φ slowly changes)
  ├─ SDF pressure compute attached (after equilibration, if enable_sdf=True)
  └─ All writers logging normally
```

**Why remove the box tuner for production?** The BoxMCMoveSize tuner changes the box-move delta parameters. If those parameters change during a measurement run, the ensemble is not stationary — the effective sampling distribution shifts. Fixing the deltas after equilibration ensures the production run samples a well-defined NPT ensemble.

**Why keep the particle MoveSize tuner active?** In NPT the box volume fluctuates, which means the packing fraction φ and therefore the optimal translational move size d both change slowly over time. Keeping the MoveSize tuner active allows d to track these slow changes continuously.

---

## 5. Algorithm: Step-by-Step

```
1.  Record start_time = time.time()
2.  Parse CLI: --simulparam_file <path>
3.  Print HOOMD version banner
4.  Load JSON → SimulationParams (type-checked, validated)
5.  Determine MPI rank from environment variables
6.  Resolve filenames (stage-aware)
7.  Print debug_kv "Run summary" with all key parameters
8.  Manage seed file: rank-0 writes random_seed.json once; all ranks read
9.  Poll for seed file (file-based MPI barrier, 30 s timeout)
10. Build simulation:
      a. Select device (CPU / GPU with fallback)
      b. Create hoomd.Simulation(device, seed)
      c. Detect run mode: restart vs fresh
      d. Load GSD state (create_state_from_gsd or MPI broadcast + snapshot)
      e. Read σ from loaded GSD; compute initial φ
      f. Attach HPMC Sphere integrator (nselect=1, initial d)
      g. Construct custom loggable instances (Status, MCStatus, Box_property,
         MoveSizeProp, OverlapCount; BoxMCStatus and BoxSeqProp after BoxMC)
      h. Attach simulation Table log writer → *_sim.log
      i. Attach box Table log writer → *_box.log
      j. Attach GSD trajectory writer (mode="ab") → *_traj.gsd
      k. Attach GSD restart writer (truncate=True) → *_restart.gsd
      l. Attach GSD scalar log writer (filter=Null) → *_scalars.gsd
      m. Configure BoxMC updater (betaP, 4 move types)
      n. Register BoxMCStatus and BoxSeqProp to loggers
      o. Attach MoveSize tuner (particle moves, active throughout)
      p. Attach BoxMCMoveSize tuner (all box moves, removed after equil.)
      q. Construct SDF compute (NOT attached yet; returned to caller)
11. Compute prod_steps = total_num_timesteps - equil_steps
12. EQUILIBRATION LOOP:
      n_chunks = equil_steps // equil_steps_check_freq
      for chunk in range(n_chunks):
          sim.run(equil_steps_check_freq)
          recompute φ from current box volume
          print chunk progress line
          if box_tuner.tuned:
              remove box_tuner from sim.operations.tuners
              set box_tuner_removed = True
              break
      if not box_tuner_removed:
          print [WARNING]
13. Attach SDF compute to sim.operations.computes (if enable_sdf=True)
14. PRODUCTION RUN: sim.run(prod_steps)
15. Write final outputs:
      - _write_snapshot → *_final.gsd
      - summary JSON → *_stage<id>_npt_summary.json
      - console banner
16. finally block:
      - flush + close sim_log_hdl, box_log_hdl
      - print total runtime
```

---

## 6. File Inventory

| File | Role |
|---|---|
| `hs_npt_v4.py` | Main simulation script |
| `simulparam_hs_npt.json` | Parameter file (edit before each run) |
| `random_seed.json` | Auto-generated seed file (single-stage runs) |
| `random_seed_stage_0.json` | Auto-generated seed file (multi-stage runs) |

**Output files** (single-stage defaults, customisable via JSON):

| File | Written every | Contents |
|---|---|---|
| `npt_hpmc_log.log` | `log_frequency` steps | Simulation Table log: tps, walltime, ETR, timestep, particle acceptance rate, d, box volume, φ, overlaps, BoxMC acceptance rates |
| `box_npt_log.log` | `log_frequency` steps | Box Table log: timestep, Lx, Ly, Lz, xy, xz, yz, volume string, φ, BoxMC move counters |
| `npt_hpmc_output_traj.gsd` | `traj_gsd_frequency` steps | Full trajectory: positions, diameters, box, embedded scalars |
| `npt_hpmc_restart.gsd` | `restart_gsd_frequency` steps | Single-frame checkpoint |
| `npt_hpmc_scalar_log.gsd` | `log_frequency` steps | Compact scalar time-series (no particle data) |
| `npt_hpmc_final.gsd` | End of run | Final one-shot snapshot |
| `<tag>_stage<id>_npt_summary.json` | End of run | Machine-readable run summary |
| `emergency_restart_<tag>.gsd` | On unexpected exception | Best-effort recovery snapshot |

---

## 7. Dependencies

### Required

| Package | Minimum version | Purpose |
|---|---|---|
| Python | 3.8+ | Language runtime |
| HOOMD-blue | 4.0 | HPMC simulation engine |
| GSD | 3.0 | Trajectory/snapshot file I/O |
| NumPy | 1.20+ | Array operations for snapshot broadcast |

### Optional

| Package | Purpose | Behaviour if absent |
|---|---|---|
| mpi4py | MPI-parallel execution | Script falls back to serial with no code change needed |

---

## 8. Installation

```bash
# Recommended: conda
conda create -n hoomd4 python=3.10
conda activate hoomd4
conda install -c conda-forge hoomd gsd numpy mpi4py

# Verify
python -c "import hoomd; print(hoomd.version.version)"
python -c "import gsd; print(gsd.version.version)"
```

---

## 9. Quick Start

### 1. Prepare a JSON file

```json
{
    "_note": "NPT run at betaP=11 following NVT equilibration",
    "tag":                                "hs_4096_npt",
    "input_gsd_filename":                 "hs_4096_nvt_final.gsd",
    "stage_id_current":                   -1,
    "total_num_timesteps":                5000000,
    "equil_steps":                        1000000,
    "equil_steps_check_freq":             50000,
    "log_frequency":                      10000,
    "traj_gsd_frequency":                 100000,
    "restart_gsd_frequency":              10000,
    "move_size_translation":              0.08,
    "trans_move_size_tuner_freq":         5000,
    "target_particle_trans_move_acc_rate": 0.25,
    "npt_freq":                           1,
    "pressure":                           11.0,
    "box_tuner_freq":                     5000,
    "target_box_movement_acc_rate":       0.25,
    "use_gpu":                            false,
    "gpu_id":                             0
}
```

### 2. Run (serial)

```bash
python hs_npt_v4.py --simulparam_file simulparam_hs_npt.json
```

### 3. Run (MPI, 8 processes)

```bash
mpirun -n 8 python hs_npt_v4.py --simulparam_file simulparam_hs_npt.json
```

### 4. Expected startup output

```
*********************************************************
HOOMD-blue version:   4.9.x
*********************************************************
hoomd.version.mpi_enabled:   True
[INFO] Loaded parameters from 'simulparam_hs_npt.json'
[INFO] Input GSD    : hs_4096_nvt_final.gsd
[INFO] Diameter (σ) = 1.0
[INFO] N=4096 | σ=1.0 | φ=0.500000
[INFO] Target betaP = 11.0
[INFO] HPMC Sphere: σ=1.0 | d_init=0.08
[INFO] BoxMC: betaP=11.0 | volume_delta=0.1 | ...
[INFO] Tuners: MoveSize (every 5000 steps) | BoxMCMoveSize (every 5000 steps)
[INFO] SDF compute configured: xmax=0.02, dx=0.0001. Will be attached after equilibration.

[INFO] Equilibration: 1000000 steps | checking every 50000 steps | max 20 chunks
[EQUIL chunk 1/20] step=50000 | phi=0.499831 | box_tuner.tuned=False | d=0.08123
...
[INFO] BoxMCMoveSize tuner has converged at step 350000. Tuner removed.
[INFO] SDF compute attached after equilibration (step 350000).

[INFO] Production: 4000000 steps starting at step 350000
...
[INFO] Production complete at step 4350000 | overlaps=0
[OUTPUT] Final GSD       => npt_hpmc_final.gsd
[OUTPUT] Summary JSON    => hs_4096_npt_stage-1_npt_summary.json
```

---

## 10. JSON Parameter Reference

### Required keys

| Key | Type | Description |
|---|---|---|
| `tag` | `str` | Unique run identifier. Used as a prefix for all output filenames in multi-stage mode. |
| `input_gsd_filename` | `str` | Path to the initial configuration GSD. In single-stage mode always used; in multi-stage mode used only for stage 0. |
| `stage_id_current` | `int` | Stage index: `-1` for single-stage, `0, 1, 2, …` for multi-stage pipeline. |
| `total_num_timesteps` | `int` | Total MC sweeps for this run = equilibration + production. Must be > 0. |
| `equil_steps` | `int` | Steps allocated to the equilibration phase. Must be < `total_num_timesteps`. The production phase gets `total - equil` steps. |
| `equil_steps_check_freq` | `int` | Chunk size for the equilibration loop. After every chunk, `box_tuner.tuned` is polled. Must be > 0. Example: if `equil_steps = 1000000` and `equil_steps_check_freq = 50000`, the loop runs up to 20 chunks. |
| `log_frequency` | `int` | Write one row to both Table logs and one frame to the scalar GSD every this many steps. Must be > 0. |
| `traj_gsd_frequency` | `int` | Append one frame to the trajectory GSD every this many steps. Must be > 0. |
| `restart_gsd_frequency` | `int` | Overwrite the single-frame restart checkpoint every this many steps. Must be > 0. Set smaller than `traj_gsd_frequency`. |
| `move_size_translation` | `float` | Initial translational move size d [σ]. The MoveSize tuner will adapt this to hit `target_particle_trans_move_acc_rate`. Must be > 0. |
| `trans_move_size_tuner_freq` | `int` | How often (in steps) the MoveSize tuner fires to update d. Must be > 0. |
| `target_particle_trans_move_acc_rate` | `float` | Target translational acceptance ratio for the MoveSize tuner. Must be in (0, 1). Recommended: 0.20–0.30. |
| `npt_freq` | `int` | How often (in steps) BoxMC fires a box trial move. Setting 1 means one box trial per MC sweep; must be > 0. |
| `pressure` | `float` | Target reduced pressure βP = P/(k_BT) in units where σ = 1. Must be > 0. |
| `box_tuner_freq` | `int` | How often (in steps) the BoxMCMoveSize tuner fires. Must be > 0. |
| `target_box_movement_acc_rate` | `float` | Target acceptance ratio for all box moves. Must be in (0, 1). Recommended: 0.20–0.35. |
| `use_gpu` | `bool` | `true` to request GPU execution. Falls back to CPU if GPU is unavailable. |
| `gpu_id` | `int` | CUDA device index (0-based). Ignored when `use_gpu=false`. |

### Optional BoxMC delta keys

These set the initial step sizes for each box move type. The BoxMCMoveSize tuner will update them during equilibration.

| Key | Type | Default | Description |
|---|---|---|---|
| `boxmc_volume_delta` | `float` | `0.1` | Initial maximum fractional volume change per volume trial. |
| `boxmc_volume_mode` | `str` | `"standard"` | Volume step method: `"standard"` (linear ΔV) or `"ln"` (log-volume step, better at high φ). |
| `boxmc_length_delta` | `float` | `0.01` | Initial maximum change in each box length (Lx, Ly, Lz) per length trial [σ]. |
| `boxmc_aspect_delta` | `float` | `0.02` | Initial maximum scale factor change per aspect trial. |
| `boxmc_shear_delta` | `float` | `0.01` | Initial maximum change per shear trial (applied to xy, xz, yz). |

### Optional BoxMC tuner max-move keys

These cap how large the tuner is allowed to set each delta parameter. Without caps, the tuner could set deltas to extreme values that cause numerical issues.

| Key | Type | Default | Description |
|---|---|---|---|
| `max_move_volume` | `float` | `0.1` | Cap on `boxmc_volume_delta` after tuning. |
| `max_move_length` | `float` | `0.05` | Cap on each length delta after tuning [σ]. |
| `max_move_aspect` | `float` | `0.02` | Cap on aspect delta after tuning. |
| `max_move_shear` | `float` | `0.02` | Cap on each shear delta after tuning. |

### Optional SDF keys

| Key | Type | Default | Description |
|---|---|---|---|
| `enable_sdf` | `bool` | `true` | Whether to attach `hoomd.hpmc.compute.SDF` after equilibration. Set `false` to disable pressure measurement. |
| `sdf_xmax` | `float` | `0.02` | Maximum scale factor for the SDF histogram. A value of 0.02 is appropriate for φ < 0.58. For denser systems use 0.01 or smaller. |
| `sdf_dx` | `float` | `1e-4` | Histogram bin width. Smaller = finer resolution but more memory. `1e-4` gives 200 bins for `xmax=0.02`. |

### Optional hardware keys

These are required in the JSON but have sensible defaults — include them to be explicit.

### Optional single-stage filename keys

Used when `stage_id_current = -1`. All ignored when `stage_id_current >= 0`.

| Key | Type | Default |
|---|---|---|
| `output_trajectory` | `str` | `"npt_hpmc_output_traj.gsd"` |
| `simulation_log_filename` | `str` | `"npt_hpmc_log.log"` |
| `box_log_filename` | `str` | `"box_npt_log.log"` |
| `scalar_gsd_log_filename` | `str` | `"npt_hpmc_scalar_log.gsd"` |
| `restart_file` | `str` | `"npt_hpmc_restart.gsd"` |
| `final_gsd_filename` | `str` | `"npt_hpmc_final.gsd"` |

### Comment keys

Any JSON key whose name starts with `_` is silently stripped before parsing. Use this to annotate your JSON files freely:

```json
{
    "_author": "Alice Smith",
    "_date": "2025-04-01",
    "_note": "fluid branch, approaching coexistence from below",
    "_section_boxmc": "BoxMC parameters",
    "tag": "hs_4096_betaP11",
    ...
}
```

---

## 11. Multi-Stage Runs

Multi-stage mode is for runs spread across multiple HPC job submissions (each with a walltime limit) or for producing clearly labelled output files per simulation phase.

When `stage_id_current >= 0`, all output filenames are derived automatically:

| File | Name |
|---|---|
| Trajectory | `<tag>_<sid>_traj.gsd` |
| Restart | `<tag>_<sid>_restart.gsd` |
| Final | `<tag>_<sid>_final.gsd` |
| Sim log | `<tag>_<sid>_sim.log` |
| Box log | `<tag>_<sid>_box.log` |
| Scalar GSD | `<tag>_<sid>_scalars.gsd` |

**Stage chaining:** Stage 0 reads `input_gsd_filename` from the JSON. Stage N ≥ 1 reads `<tag>_{N-1}_final.gsd` automatically — the user only needs to increment `stage_id_current`.

**Safety guards:**
- If `<tag>_<sid>_final.gsd` already exists: **hard exit** — stage is complete. Increment `stage_id_current`.
- If `<tag>_{N-1}_final.gsd` is absent for N ≥ 1: **hard exit** — previous stage did not complete.

### Example: 3-stage NPT run (3 × 5M steps)

**Stage 0** — set `"stage_id_current": 0, "total_num_timesteps": 5000000`

```bash
python hs_npt_v4.py --simulparam_file simulparam_hs_npt.json
# → produces: hs_4096_npt_0_traj.gsd, hs_4096_npt_0_final.gsd, hs_4096_npt_0_sim.log, ...
```

**Stage 1** — set `"stage_id_current": 1`

```bash
python hs_npt_v4.py --simulparam_file simulparam_hs_npt.json
# reads: hs_4096_npt_0_final.gsd
# → produces: hs_4096_npt_1_traj.gsd, hs_4096_npt_1_final.gsd, ...
```

**Stage 2** — set `"stage_id_current": 2`

```bash
python hs_npt_v4.py --simulparam_file simulparam_hs_npt.json
# reads: hs_4096_npt_1_final.gsd
```

---

## 12. Output Files Reference

### Simulation log (`*_sim.log`)

Plain-text, space-separated Table log written every `log_frequency` steps. First row is a column header. Parse with `pandas.read_csv(sep=r'\s+')` or `numpy.loadtxt`.

**Columns:**

| Column | Source | Description |
|---|---|---|
| `Simulation/tps` | HOOMD built-in | Timesteps per second (performance) |
| `Simulation/walltime` | HOOMD built-in | Elapsed wall time [s] |
| `Simulation/timestep` | HOOMD built-in | Current step (integer scalar) |
| `Status/etr` | `Status` | Estimated time remaining as `H:MM:SS` |
| `Status/timestep` | `Status` | `"current/total"` progress string |
| `MCStatus/acc_rate` | `MCStatus` | Windowed translational acceptance ratio (since last log entry) |
| `MoveSize/d` | `MoveSizeProp` | Current translational move size d [σ] |
| `Box/volume` | `Box_property` | Box volume V [σ³] |
| `Box/phi` | `Box_property` | Packing fraction φ = N·(π/6)·σ³/V |
| `HPMC/overlaps` | `OverlapCount` | Hard-sphere overlap count. **Must always be 0.** |
| `BoxMCStatus/acc_rate` | `BoxMCStatus` | Windowed overall BoxMC acceptance rate |
| `BoxMCStatus/volume_acc_rate` | `BoxMCStatus` | Windowed acceptance rate for volume moves |
| `BoxMCStatus/aspect_acc_rate` | `BoxMCStatus` | Windowed acceptance rate for aspect moves |
| `BoxMCStatus/shear_acc_rate` | `BoxMCStatus` | Windowed acceptance rate for shear moves |
| `SDF/betaP` | `SDFPressure` | Instantaneous βP from SDF (0.0 before production; 0.0 if enable_sdf=False) |
| `SDF/compressibility_Z` | `SDFPressure` | Z = βPV/N (compare with Carnahan-Starling) |

### Box log (`*_box.log`)

A separate Table log recording box geometry at every `log_frequency` steps. Dedicated to tracking the evolution of box shape during NPT equilibration.

**Columns:**

| Column | Description |
|---|---|
| `Simulation/timestep` | Current step |
| `Box_property/l_x` | Box length Lx [σ] |
| `Box_property/l_y` | Box length Ly [σ] |
| `Box_property/l_z` | Box length Lz [σ] |
| `Box_property/XY` | Tilt factor xy |
| `Box_property/XZ` | Tilt factor xz |
| `Box_property/YZ` | Tilt factor yz |
| `Box/volume_str` | Box volume formatted as `"V.VV"` (string) |
| `Box/phi` | Packing fraction φ |
| `BoxMC/vol_moves` | Volume moves as `"accepted,total"` string |
| `BoxMC/aspect_moves` | Aspect moves as `"accepted,total"` string |
| `BoxMC/shear_moves` | Shear moves as `"accepted,total"` string |
| `hpmc/update/BoxMC/volume_moves` | Native HOOMD tuple (accepted, rejected) |
| `hpmc/update/BoxMC/aspect_moves` | Native HOOMD tuple |
| `hpmc/update/BoxMC/shear_moves` | Native HOOMD tuple |

### GSD trajectory (`*_traj.gsd`)

Binary GSD appended every `traj_gsd_frequency` steps. Contains particle positions, diameters, box dimensions, and embedded logged scalars from `logger_sim`. Mode `"ab"` ensures continuity across restarts.

### GSD restart (`*_restart.gsd`)

Single-frame GSD (`truncate=True`) overwritten every `restart_gsd_frequency` steps. Contains the complete particle state and timestep. Used by the restart detection logic on next startup. Always exactly one frame — the most recent checkpoint.

### GSD scalar log (`*_scalars.gsd`)

**The primary file for post-processing and EOS analysis.** A GSD written every `log_frequency` steps using `filter=hoomd.filter.Null()` — zero particle data — storing only scalar time-series. Very compact (kilobytes for millions of frames).

**Contents (datasets in the GSD log namespace):**

| Quantity | Description |
|---|---|
| `timestep` | Current step |
| `tps` | Timesteps per second |
| `walltime` | Elapsed wall time [s] |
| `Box/volume` | Box volume V [σ³] |
| `Box/phi` | Packing fraction φ |
| `MCStatus/acc_rate` | Windowed particle acceptance rate |
| `MoveSize/d` | Current translational move size d |
| `HPMC/overlaps` | Overlap count |
| `BoxMCStatus/acc_rate` | Windowed overall BoxMC acceptance rate |
| `SDF/betaP` | Instantaneous βP (0.0 before production) |
| `SDF/Z` | Instantaneous Z = βPV/N |

### Final GSD (`*_final.gsd`)

Single-frame GSD written via `hoomd.write.GSD.write()` at the end of a successful run. Contains all particle data including diameter. This is the canonical input for the next pipeline stage.

### Summary JSON

Machine-readable file named `<tag>_stage<sid>_npt_summary.json`, written by rank 0 at the end of the run.

```json
{
  "tag":               "hs_4096_npt",
  "stage_id":          -1,
  "simulparam_file":   "simulparam_hs_npt.json",
  "input_gsd":         "hs_4096_nvt_final.gsd",
  "final_gsd":         "npt_hpmc_final.gsd",
  "n_particles":       4096,
  "diameter":          1.0,
  "packing_fraction":  0.50183412,
  "target_betaP":      11.0,
  "overlaps_final":    0,
  "final_timestep":    4350000,
  "random_seed":       31742,
  "box_final": {
    "Lx": 18.49,  "Ly": 18.49,  "Lz": 18.49,
    "xy": 0.0,    "xz": 0.0,    "yz": 0.0
  },
  "runtime_seconds":   7821.44,
  "created":           "2025-04-01T14:22:01"
}
```

### Emergency snapshot

`emergency_restart_<tag>.gsd` — written only if `sim.run()` raises an unexpected exception. Contains the simulation state at the moment of failure. Use this to inspect the configuration or resume from it.

---

## 13. Restart and Recovery

The script uses a two-file detection scheme at startup:

```
restart_GSD_exists  AND  final_GSD_absent  →  RESTART (resume from checkpoint)
any other combination                       →  FRESH RUN (from input_gsd_filename)
```

| State | `restart.gsd` | `final.gsd` | Action |
|---|---|---|---|
| First run | absent | absent | Fresh from input GSD |
| Interrupted | present | absent | Resume from checkpoint |
| Completed | present | present | Fresh run (warning printed) |
| Re-running | absent | present | Fresh run (warning printed) |

**To restart an interrupted run:** resubmit with the identical JSON. The restart GSD is detected and the run resumes automatically from the last checkpoint. The trajectory GSD uses `mode="ab"` so frames from before and after the interruption are in the same file — no concatenation needed.

**Checkpoint frequency recommendation:** Set `restart_gsd_frequency` 5–10× smaller than `traj_gsd_frequency` so you never lose more than `restart_gsd_frequency` steps. Example: if `traj_gsd_frequency=100000`, set `restart_gsd_frequency=10000`.

---

## 14. MPI Parallel Execution

HOOMD-blue uses **spatial domain decomposition**: the box is divided into rectangular subdomains, one per MPI rank. Each rank owns a particle subset. All inter-rank communication for neighbour lists and I/O is handled internally by HOOMD's MPI layer.

The script handles MPI explicitly in two places only:

1. **Rank detection** (`_mpi_rank_from_env()`): reads rank from launcher environment variables before the HOOMD device exists. Priority: `OMPI_COMM_WORLD_RANK` → `PMI_RANK` → `SLURM_PROCID` → default 0.

2. **Snapshot broadcast** (`load_and_broadcast_snapshot()`): rank 0 reads the GSD and broadcasts a data dict to all ranks via `comm.bcast()`. All ranks then reconstruct identical `hoomd.Snapshot` objects. This is the fresh-run path; restart uses `sim.create_state_from_gsd()` which HOOMD handles internally.

All console output goes through `root_print()` so only rank 0 prints, preventing N identical lines per log entry.

**Serial runs (no mpi4py):** The `_MPIStub` class provides a no-op `bcast()` so all code paths work identically on a single CPU without any conditional logic in the simulation code.

---

## 15. GPU Execution

Set `"use_gpu": true` and `"gpu_id": 0` in the JSON. HOOMD will use the specified CUDA device.

If GPU initialization fails for any reason (CUDA not available, wrong device ID, insufficient compute capability), the script **automatically falls back to CPU** with a `[WARNING]` message — the run is not aborted. This makes job scripts portable across heterogeneous nodes.

---

## 16. Code Architecture: Section-by-Section

The script is divided into 11 numbered sections, each with a single responsibility.

### Section 1: MPI Console Helpers

**Functions:** `_mpi_rank_from_env`, `_is_root_rank`, `root_print`, `root_flush_stdout`, `debug_kv` (defined twice — intentional), `fail_with_context`

These functions address a fundamental MPI problem: in an N-rank run, every `print()` call produces N identical lines. `root_print()` routes all output through `_is_root_rank()` so only rank 0 prints.

`debug_kv(title, **kwargs)` prints a titled key=value block — used at state-init decision points and in the exception handler to give maximum diagnostic context. It is defined **twice** in the source (both definitions are identical) — this is preserved as-is from the original.

`fail_with_context(message, **kwargs)` is a structured replacement for bare `sys.exit(message)`. It formats the message plus any keyword context fields into a multi-line `[FATAL]` block before calling `sys.exit()`. Unlike raising an exception, this exits with a clean message the operator can act on without reading a Python traceback.

### Section 2: Custom Loggable Classes

All seven custom loggable classes implement the HOOMD `_export_dict` protocol. Every property getter is guarded against `DataAccessError`, which HOOMD raises when it validates loggables by calling property getters at registration time — before any `sim.run()` call.

| Class | Properties | Purpose |
|---|---|---|
| `Status` | `timestep_fraction`, `etr` | Progress strings for the Table log |
| `MCStatus` | `acceptance_rate` | Windowed particle translational acceptance rate |
| `Box_property` | `L_x`, `L_y`, `L_z`, `XY`, `XZ`, `YZ`, `volume`, `volume_str`, `packing_fraction` | All box geometry quantities plus φ |
| `MoveSizeProp` | `d` | Current translational move size from the integrator |
| `BoxMCStatus` | `acceptance_rate`, `volume_acc_rate`, `aspect_acc_rate`, `shear_acc_rate` | Overall and per-type BoxMC acceptance rates |
| `BoxSeqProp` | `volume_moves_str`, `aspect_moves_str`, `shear_moves_str` | BoxMC counters as `"accepted,total"` strings |
| `OverlapCount` | `overlap_count` | Hard-sphere overlap count |
| `SDFPressure` | `betaP`, `compressibility_Z` | SDF-derived pressure and compressibility factor |

**Why windowed rates?** Both `MCStatus.acceptance_rate` and `BoxMCStatus._windowed_rate()` compute the acceptance ratio in the interval since the previous log call rather than the cumulative rate. This is more informative: the cumulative rate converges to a plateau and loses sensitivity, while the windowed rate responds to changes as the box equilibrates.

**Why per-move-type BoxMC rates?** `[N-18]` adds `volume_acc_rate`, `aspect_acc_rate`, and `shear_acc_rate` individually. If the aggregate rate looks wrong, the per-type rates show exactly which move type is misbehaving, enabling targeted delta adjustments.

**`SDFPressure`:** Wraps `hoomd.hpmc.compute.SDF`. `sdf.betaP` returns `None` on non-root MPI ranks (HOOMD only collects the histogram on rank 0); the guard `if val is not None` handles this cleanly. `compressibility_Z = βPV/N` is the standard EOS quantity compared with Carnahan-Starling or Hall's equation.

### Section 3: SimulationParams Dataclass

A `@dataclass` with type-annotated fields for every JSON key plus runtime-computed fields. The `validate()` method runs physics and logic checks, collecting all errors before raising `ValueError` so the user sees every problem at once.

**Validation checks:**
- `total_num_timesteps > 0`
- `equil_steps < total_num_timesteps` (so production phase ≥ 1 step)
- `equil_steps_check_freq > 0` (prevents ZeroDivisionError in the equil loop)
- `pressure > 0`
- Both acceptance-rate targets in open interval (0, 1)
- `stage_id_current ≥ -1`
- All frequency parameters > 0

`diameter` is `Optional[float] = None` — it is not a JSON key but is populated at runtime by `read_mono_diameter_from_gsd()` after the GSD is loaded.

### Section 4: JSON Loading and Validation

`load_simulparams()` performs validation in six stages:

1. File existence check (clear error before the GSD C extension raises a confusing `FileNotFoundError`)
2. JSON syntax check (`json.JSONDecodeError` → `fail_with_context` with line/column numbers)
3. Top-level type check (must be a dict, not a list or null)
4. Comment-key stripping (keys starting with `_`)
5. Required-key presence check
6. Required-key type checking (all 18 required keys checked against `_REQUIRED_KEYS` dict)

Optional keys are included only when present in the JSON, allowing dataclass defaults to apply. All float-type fields are coerced with `float()` because JSON integer literals arrive as Python `int` objects.

### Section 5: Random Seed Management

HOOMD v4/v5 silently truncates seeds > 65535, breaking reproducibility. `secrets.randbelow(65536)` generates a seed from the OS CSPRNG in the valid [0, 65535] range.

The seed is written once to `random_seed.json` (single-stage) or `random_seed_stage_0.json` (multi-stage) on the first call by rank 0. All subsequent calls — restarts, later stages, all MPI ranks — read the same file. A 30-second file-polling loop acts as a filesystem-safe MPI barrier before `hoomd.Simulation` exists.

### Section 6: Filename Resolution

`resolve_filenames()` implements two modes:

**Single-stage (`stage_id = -1`):** filenames come directly from the JSON. Warns if the final GSD already exists but does not abort.

**Multi-stage (`stage_id ≥ 0`):** all filenames are derived as `<tag>_<sid>_<suffix>`. Hard exits if the current stage's final GSD exists (completed stage guard) or if the previous stage's final GSD is missing (incomplete pipeline guard).

### Section 7: GSD Diameter Reader

`read_mono_diameter_from_gsd()` opens the last frame of a GSD file and returns σ. Validates: file exists, file has frames, diameters array is non-empty, all diameters are positive, all diameters are equal within 1×10⁻¹² (float32→float64 round-trip tolerance).

### Section 8: MPI Snapshot Broadcast

`load_and_broadcast_snapshot()` + `reconstruct_snapshot()` implement the MPI-safe fresh-run state initialization. Rank 0 reads the last GSD frame, validates all array shapes, calls `debug_kv("Loaded input snapshot", ...)` for diagnostics, packs data into a dict, and broadcasts it via `comm.bcast()`. All ranks then reconstruct identical `hoomd.Snapshot` objects. `reconstruct_snapshot()` validates all six required keys and all three array shapes before calling the HOOMD Snapshot API.

### Section 9: Simulation Builder (17 substeps)

`build_simulation()` constructs and returns the fully configured simulation. Substeps 9.1–9.17:

| Step | What happens |
|---|---|
| 9.1 | GPU or CPU device selection, with fallback |
| 9.2 | `hoomd.Simulation(device, seed)` |
| 9.3 | Restart vs fresh detection; `create_state_from_gsd` or broadcast+snapshot |
| 9.4 | `read_mono_diameter_from_gsd`; compute initial φ; print box info |
| 9.5 | HPMC Sphere integrator (`nselect=1`, initial d) |
| 9.6 | Construct 5 custom loggable instances |
| 9.7 | `logger_sim` + simulation Table writer → `*_sim.log` |
| 9.8 | `logger_box` + box Table writer → `*_box.log` |
| 9.9 | Open file handles + attach both Table writers (each wrapped in try/except) |
| 9.10 | GSD trajectory writer (`mode="ab"`, `dynamic=`, `logger=logger_sim`) |
| 9.11 | GSD restart writer (`truncate=True`, `mode="wb"`, `dynamic=`) |
| 9.12 | GSD scalar log writer (`filter=Null`, `mode="ab"`) |
| 9.13 | BoxMC updater (`betaP_variant`, 4 move types configured) |
| 9.14 | Register `BoxMCStatus` and `BoxSeqProp` to both loggers (after BoxMC exists) |
| 9.15 | MoveSize tuner (`moves=["d"]`, `max_translation_move=0.2`) |
| 9.16 | BoxMCMoveSize tuner (8 box-move degrees of freedom, `gamma=0.8`, `tol=0.01`) |
| 9.17 | SDF pressure compute constructed (NOT yet attached; returned to caller) |

Returns: `(sim, mc, boxmc, move_tuner, box_tuner, sdf_compute, sim_log_hdl, box_log_hdl)`

Each writer creation in steps 9.9–9.12 is wrapped in its own `try/except` with detailed error messages including the filename, frequency, and HOOMD error type — making filesystem and API failures immediately actionable.

### Section 10: Output Helpers

`_write_snapshot(sim, filename)` uses `hoomd.write.GSD.write()` — the one-shot static helper that writes all non-default particle fields (including diameter) to a fresh GSD in a single call. Used for the final snapshot and the emergency recovery snapshot.

`write_final_outputs()` calls `_write_snapshot`, writes the summary JSON (rank 0 only, to avoid filesystem race conditions), and prints the console banner with all key run statistics.

### Section 11: Entry Point

`main()` orchestrates the full 16-step workflow. The central try/except/finally structure:

```python
try:
    sim, mc, boxmc, move_tuner, box_tuner, ... = build_simulation(...)
    # equilibration loop
    # production run
    write_final_outputs(...)
except Exception as exc:
    # [N-12] print error twice; print debug_kv context blocks;
    #        write emergency snapshot; re-raise
finally:
    # [N-15] flush + close sim_log_hdl and box_log_hdl
    # print total runtime
```

The `__main__` guard catches `SystemExit` (re-raised silently — message already printed) and `Exception` (prints `traceback.print_exc()` + `sys.exit(1)`).

---

## 17. The BoxMC Updater: Four Move Types Explained

BoxMC fires every `npt_freq` steps. Each fire proposes exactly one trial move, drawn from the active move types with probability proportional to their `weight` values (all set to 1.0 here, so each type is equally likely).

### Volume moves

Propose isotropic scaling: all three box lengths and all particle positions are scaled uniformly. The volume changes as V → V · (1 + δ_V · U) where U ∈ [−1, 1] is a uniform random variable.

`mode="standard"`: linear ΔV. `mode="ln"`: log-volume step. For hard spheres at high density, `mode="ln"` gives better acceptance because the distribution of allowed volume changes is more symmetric in log-space.

### Length moves

Propose changing one box length independently (Lx, Ly, or Lz chosen at random) while keeping the other two and all tilt factors fixed. Enables the box to relax into a non-cubic equilibrium shape, which matters near phase coexistence and in the crystal phase.

### Aspect moves

Propose scaling one axis relative to the others at constant volume. Different from length moves: aspect moves keep V fixed while changing the box shape. This is important for systems where the equilibrium crystal has a non-cubic unit cell.

### Shear moves

Propose changing one tilt factor (xy, xz, or yz chosen at random). Shear moves are essential for any system that may crystallise into a non-orthorhombic structure (FCC in 3D has a rhombohedral primitive cell, and its equilibrium box can have non-zero tilt factors). `reduce=0.0` disables Lees-Edwards lattice-vector reduction, appropriate for equilibrium simulations without shear flow.

---

## 18. The Equilibration Loop: How the Tuner Works

The BoxMCMoveSize tuner (`hoomd.hpmc.tune.BoxMCMoveSize.scale_solver`) adapts eight box-move step sizes simultaneously:

- `volume`: the volume delta δ_V
- `length_x`, `length_y`, `length_z`: the three independent length deltas
- `aspect`: the aspect delta
- `shear_x`, `shear_y`, `shear_z`: the three independent shear deltas

**Tuner logic:** every `box_tuner_freq` steps, the tuner reads the current acceptance ratio for each move type and multiplies its delta by a factor derived from the discrepancy from the target. The factor is controlled by `gamma=0.8` (closer to 1 = more aggressive). The tuner sets `box_tuner.tuned = True` when all eight move-type acceptance ratios are within `tol=0.01` (±1%) of the target.

**The equilibration loop checks `box_tuner.tuned` after every `equil_steps_check_freq` steps.** When it becomes True, the tuner is removed from `sim.operations.tuners` and the loop breaks early. This is the correct moment to start production: all box-move deltas are now optimal and will remain fixed.

**If the loop exhausts all chunks without convergence:** a `[WARNING]` is printed and the production run proceeds with the tuner still active. The user should consider increasing `equil_steps`, decreasing `tol`, or adjusting the initial `boxmc_*_delta` values to be closer to the optimal values.

---

## 19. SDF Pressure Measurement

The **Scale Distribution Function (SDF)** method computes the instantaneous pressure of a hard-sphere system without needing to count explicit collisions. The algorithm:

1. For each particle, find the smallest scale factor x > 0 by which you would need to uniformly expand all particles before the first overlap occurs (in effect: how close is the nearest neighbour in scaled coordinates?).
2. Build a histogram of these x values: s(x).
3. Extrapolate s(x) to x=0. For hard spheres in 3D:

```
βP = ρ · (1 + s(0+) / 6)
```

where ρ = N/V is the number density.

**Why only after equilibration?** SDF assumes a **stationary box** over the measurement interval. During the equilibration phase the box volume fluctuates strongly and the box-move deltas are still being tuned — attaching SDF here would produce noisy, physically meaningless pressure readings. After equilibration the box fluctuates around its equilibrium volume with fixed delta sizes, making the SDF meaningful.

**EOS validation:** during production, `<betaP>_time_average` should converge to the target `pressure`. If it does not, the system has not equilibrated to the target pressure and more equilibration steps are needed.

**Carnahan-Starling comparison (fluid phase):**

```
Z_CS = (1 + φ + φ² − φ³) / (1 − φ)³
```

Compare the SDF-measured Z = βPV/N with Z_CS using the instantaneous φ. Agreement to within a few percent confirms the EOS is correct.

**`sdf_xmax`:** the upper limit of the histogram. For φ < 0.58 (fluid), 0.02 is suitable. For dense solids or near coexistence, use smaller values (0.01 or 0.005) to capture the sharp peak near x=0 with better resolution.

---

## 20. Logged Quantities Reference

### Simulation log — full column map

| Column header | Python type | Source | Notes |
|---|---|---|---|
| `Simulation/tps` | float | HOOMD | Steps/s — performance monitor |
| `Simulation/walltime` | float | HOOMD | Elapsed seconds since sim start |
| `Simulation/timestep` | int | HOOMD | Current step counter |
| `Status/etr` | string | `Status.etr` | `"D days, H:MM:SS"` format |
| `Status/timestep` | string | `Status.timestep_fraction` | `"current/total"` |
| `MCStatus/acc_rate` | float | `MCStatus.acceptance_rate` | Windowed; target ~0.25 |
| `MoveSize/d` | float | `MoveSizeProp.d` | Current d [σ] |
| `Box/volume` | float | `Box_property.volume` | V [σ³]; fluctuates in NPT |
| `Box/phi` | float | `Box_property.packing_fraction` | φ = N(π/6)σ³/V; fluctuates |
| `HPMC/overlaps` | float | `OverlapCount.overlap_count` | Must be 0 |
| `BoxMCStatus/acc_rate` | float | `BoxMCStatus.acceptance_rate` | Aggregate box acceptance |
| `BoxMCStatus/volume_acc_rate` | float | `BoxMCStatus.volume_acc_rate` | Volume-only acceptance |
| `BoxMCStatus/aspect_acc_rate` | float | `BoxMCStatus.aspect_acc_rate` | Aspect-only acceptance |
| `BoxMCStatus/shear_acc_rate` | float | `BoxMCStatus.shear_acc_rate` | Shear-only acceptance |
| `SDF/betaP` | float | `SDFPressure.betaP` | Instantaneous βP; 0 in equil. |
| `SDF/compressibility_Z` | float | `SDFPressure.compressibility_Z` | Z = βPV/N |

---

## 21. Reading the Scalar GSD Log in Python

The compact scalar GSD (`*_scalars.gsd`) is the primary post-processing file. Every frame corresponds to one `log_frequency`-step interval.

```python
import gsd.hoomd
import numpy as np

with gsd.hoomd.open("npt_hpmc_scalar_log.gsd", "r") as f:
    # Number of logged intervals
    n_frames = len(f)
    print(f"Frames in scalar log: {n_frames}")

    # Read all timesteps at once using GSD's log access
    # (HOOMD stores logged quantities under f[i].log)
    timesteps  = np.array([f[i].log["Simulation/timestep"][0] for i in range(n_frames)])
    phi        = np.array([f[i].log["Box/phi"][0]             for i in range(n_frames)])
    volume     = np.array([f[i].log["Box/volume"][0]          for i in range(n_frames)])
    betaP      = np.array([f[i].log["SDF/betaP"][0]           for i in range(n_frames)])
    Z          = np.array([f[i].log["SDF/compressibility_Z"][0] for i in range(n_frames)])
    acc_rate   = np.array([f[i].log["MCStatus/acc_rate"][0]   for i in range(n_frames)])
    box_acc    = np.array([f[i].log["BoxMCStatus/acc_rate"][0] for i in range(n_frames)])
    overlaps   = np.array([f[i].log["HPMC/overlaps"][0]       for i in range(n_frames)])

# EOS analysis: time-average of betaP and phi over production phase
# (skip the equilibration phase, approximately the first equil_steps/log_frequency frames)
equil_frames = 20  # adjust to match equil_steps / log_frequency
prod_betaP = betaP[equil_frames:]
prod_phi   = phi[equil_frames:]
prod_Z     = Z[equil_frames:]

mean_betaP = np.mean(prod_betaP[prod_betaP > 0])  # exclude pre-SDF zeros
mean_phi   = np.mean(prod_phi)
mean_Z     = np.mean(prod_Z[prod_Z > 0])

print(f"<betaP> = {mean_betaP:.4f}  (target: {params_pressure})")
print(f"<phi>   = {mean_phi:.6f}")
print(f"<Z>     = {mean_Z:.4f}")

# Sanity check: no overlaps
assert np.all(overlaps == 0), "Non-zero overlaps detected!"

# Carnahan-Starling comparison (fluid phase only)
phi_cs = mean_phi
Z_CS = (1 + phi_cs + phi_cs**2 - phi_cs**3) / (1 - phi_cs)**3
print(f"Z_CS    = {Z_CS:.4f}  (Carnahan-Starling)")
print(f"Z/Z_CS  = {mean_Z/Z_CS:.4f}  (should be ~1.0 for fluid)")
```

---

## 22. Diameter Persistence: How and Why

Hard-sphere HPMC depends entirely on knowing σ. If σ is absent from a GSD frame, any analysis script that reads the trajectory cannot reconstruct φ, cannot verify particle sizes, and may silently compute incorrect quantities.

This script uses a **dual mechanism**:

**Periodic writers (trajectory, restart):**
```python
hoomd.write.GSD(
    ...,
    dynamic=["property", "attribute"],
)
writer.write_diameter = True
```
`dynamic=["property","attribute"]` is the v4.9 documented API — the `"property"` GSD schema category includes `particles/diameter`. This writes σ to **every** frame. `write_diameter=True` is kept as a belt-and-suspenders fallback for HOOMD v4.0–v4.5.

**One-shot writes (final, emergency):**
`hoomd.write.GSD.write()` always writes all non-default particle fields including diameter to frame 0 of the file, independent of the periodic writer configuration.

---

## 23. Seed Reproducibility

To replay an exact trajectory:

1. Find `random_seed` in `<tag>_stage<id>_npt_summary.json`.
2. Create/edit `random_seed.json`:
   ```json
   { "random_seed": 31742, "created_at": "2025-04-01T14:22:01" }
   ```
3. Run with the identical JSON and identical number of MPI ranks on the same hardware.

**MPI note:** HOOMD does not guarantee bit-for-bit identical trajectories between different rank counts or different hardware, even with the same seed. Same seed + same rank count + same hardware = bit-identical.

---

## 24. Choosing `pressure` (betaP) and Move Deltas

### Pressure βP

The reduced pressure βP = Pσ³/(k_BT) for hard spheres ranges from ~0 (dilute gas) to very large values (dense crystal). Key reference points for hard spheres (σ = 1):

| State point | βP (approx.) | φ (approx.) |
|---|---|---|
| Dilute fluid | 1–5 | 0.10–0.30 |
| Dense fluid | 5–11.5 | 0.30–0.494 |
| Fluid–crystal coexistence | 11.5 | 0.494 (fluid) / 0.545 (crystal) |
| FCC crystal (stable) | > 11.5 | 0.545–0.64 |
| Close packing | → ∞ | 0.7405 |

These are for the infinite-N limit; finite-size effects shift the coexistence boundaries slightly. Use the Carnahan-Starling equation for fluid-phase EOS reference.

### Move delta guidance

The tuner will adapt deltas automatically, but starting close to the optimal values reduces the equilibration time needed. Rough starting points:

| φ | `boxmc_volume_delta` | `boxmc_length_delta` |
|---|---|---|
| 0.35–0.45 | 0.15–0.30 | 0.02–0.05 |
| 0.45–0.52 | 0.05–0.15 | 0.01–0.02 |
| 0.52–0.58 | 0.02–0.05 | 0.005–0.01 |

Target acceptance rates of 20–35% for both particle and box moves. Check the log:
- If `BoxMCStatus/volume_acc_rate > 50%`: `boxmc_volume_delta` is too small.
- If `BoxMCStatus/volume_acc_rate < 5%`: `boxmc_volume_delta` is too large.

---

## 25. Reading Output Files in Python

### Reading the trajectory GSD

```python
import gsd.hoomd
import numpy as np

with gsd.hoomd.open("npt_hpmc_output_traj.gsd", "r") as traj:
    print(f"Frames: {len(traj)}")
    frame = traj[-1]                          # last frame
    pos   = frame.particles.position          # (N, 3)
    sigma = frame.particles.diameter[0]       # σ
    box   = frame.configuration.box          # [Lx, Ly, Lz, xy, xz, yz]
    step  = frame.configuration.step
    print(f"Step {step}: N={frame.particles.N}, σ={sigma:.4f}")
    print(f"Box: Lx={box[0]:.4f}, φ={frame.particles.N * (np.pi/6)*sigma**3 / (box[0]*box[1]*box[2]):.6f}")
```

### Reading the simulation log

```python
import pandas as pd
import numpy as np

df = pd.read_csv("npt_hpmc_log.log", sep=r'\s+')
print(df.columns.tolist())

# Equilibration vs production split (adjust equil_frames for your run)
equil_rows = 100  # equil_steps / log_frequency
prod = df.iloc[equil_rows:]

print(f"Production <phi>   = {prod['Box/phi'].mean():.6f}")
print(f"Production <betaP> = {prod['SDF/betaP'][prod['SDF/betaP'] > 0].mean():.4f}")
print(f"Production <Z>     = {prod['SDF/compressibility_Z'][prod['SDF/compressibility_Z'] > 0].mean():.4f}")
print(f"Mean particle acceptance rate: {prod['MCStatus/acc_rate'].mean():.3f}")
print(f"Mean box acceptance rate:      {prod['BoxMCStatus/acc_rate'].mean():.3f}")
```

### Reading the box log

```python
import pandas as pd

box_df = pd.read_csv("box_npt_log.log", sep=r'\s+')
# Plot Lx, Ly, Lz as a function of time to check box shape convergence
print(box_df[["Simulation/timestep", "Box_property/l_x", "Box_property/l_y",
              "Box_property/l_z", "Box/phi"]].tail(10))
```

### Reading the summary JSON

```python
import json

with open("hs_4096_npt_stage-1_npt_summary.json") as f:
    s = json.load(f)

print(f"N = {s['n_particles']}, σ = {s['diameter']}")
print(f"Final φ = {s['packing_fraction']:.6f}")
print(f"Target βP = {s['target_betaP']}")
print(f"Final overlaps = {s['overlaps_final']}")
print(f"Final box: Lx={s['box_final']['Lx']:.4f}")
print(f"Runtime = {s['runtime_seconds']:.1f} s")
```

---

## 26. HPC Cluster / SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=hs_npt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=regular
#SBATCH --output=slurm_%j.out

module load hoomd/4.9.0
module load mpi4py

# Set HOOMD_WALLTIME_STOP to 10 minutes before the job ends.
# HOOMD will exit sim.run() cleanly when time is up, flushing all writers.
export HOOMD_WALLTIME_STOP=$(( $(date +%s) + 24*3600 - 600 ))

cd $SLURM_SUBMIT_DIR

# For stage 0: "stage_id_current": 0 in JSON
# After walltime hit: resubmit same script unchanged — restart is automatic
# After stage completes: increment stage_id_current to 1 and resubmit

mpirun -n 8 python hs_npt_v4.py \
    --simulparam_file simulparam_hs_npt.json \
    > npt_stdout.log 2>&1
```

**Notes on `HOOMD_WALLTIME_STOP`:** HOOMD monitors this environment variable inside `sim.run()`. When the current wall time exceeds the stop time, HOOMD exits `sim.run()` cleanly (flushing all writers) and returns control to the script. The `finally` block then closes the log file handles and prints the total runtime. The restart checkpoint GSD written just before the stop is the recovery point for the next submission.

For the equilibration loop, `HOOMD_WALLTIME_STOP` interrupts cleanly at any `sim.run(equil_steps_check_freq)` boundary. The tuner state is not persisted across runs (HOOMD does not save tuner state in GSD), so on restart the tuner starts fresh — this is acceptable because the equilibration will continue from the checkpoint with the existing (partially-tuned) box-move deltas.

---

## 27. Exception Handling Map

Every exception pathway in the script:

| Location | Exception | Handler | Output |
|---|---|---|---|
| `import gsd.hoomd` | `ImportError` | `sys.exit` | `[FATAL] gsd package not found` |
| `import hoomd` | `ImportError` | `sys.exit` | `[FATAL] HOOMD-blue not found` |
| `load_simulparams`: file missing | — | `sys.exit` | `[FATAL] Parameter file not found` |
| `load_simulparams`: bad JSON | `json.JSONDecodeError` | `fail_with_context` | Line/column of error |
| `load_simulparams`: not a dict | — | `fail_with_context` | Parsed type shown |
| `load_simulparams`: missing keys | — | `sys.exit` | Lists missing keys |
| `load_simulparams`: wrong types | — | `sys.exit` | Lists all type errors |
| `params.validate()` | `ValueError` | `sys.exit` | All validation errors |
| `resolve_filenames`: stage guard | — | `sys.exit` | Stage N final GSD exists |
| `resolve_filenames`: prev stage missing | — | `sys.exit` | Stage N-1 output missing |
| `resolve_filenames`: input GSD missing | — | `sys.exit` | Input file path |
| `read_seed`: file/key/decode error | multiple | `fail_with_context` | seed_file, stage_id, error |
| Seed file poll timeout | — | `fail_with_context` | seed_file, elapsed seconds, rank |
| `load_and_broadcast_snapshot`: all paths | multiple | `fail_with_context` | input_gsd, shapes, error |
| `reconstruct_snapshot`: all paths | multiple | `fail_with_context` | shapes, N, box, error |
| `build_simulation` (9.3): restart GSD | `Exception` | `fail_with_context` | restart_gsd, error |
| `build_simulation` (9.3): fresh run | `Exception` | `fail_with_context` | input_gsd, rank, error |
| `build_simulation` (9.9): sim_log open | `OSError` | `sys.exit` | sim_log_file, OS error |
| `build_simulation` (9.9): sim Table writer | `Exception` | close hdl + `sys.exit` | file, frequency, error |
| `build_simulation` (9.9): box_log open | `OSError` | close sim_hdl + `sys.exit` | box_log_file, OS error |
| `build_simulation` (9.9): box Table writer | `Exception` | close both + `sys.exit` | file, frequency, error |
| `build_simulation` (9.10): traj GSD | `Exception` | `sys.exit` | file, frequency, mode, error |
| `build_simulation` (9.11): restart GSD | `Exception` | `sys.exit` | file, frequency, truncate, mode, error |
| `build_simulation` (9.12): scalar GSD | `Exception` | `sys.exit` | file, frequency, filter, mode, error |
| `build_simulation` (9.13): BoxMC | `Exception` | `fail_with_context` | pressure, all deltas, error |
| `Status.timestep_fraction` | `Exception` | return `"0/?"` | Pre-run sentinel |
| `Status.seconds_remaining` | `ZeroDivisionError`, `DataAccessError`, `AttributeError` | return `0.0` | Pre-run sentinel |
| `MCStatus.acceptance_rate` | `IndexError`, `ZeroDivisionError`, `DataAccessError` | return `0.0` | Pre-run sentinel |
| `BoxMCStatus._windowed_rate` | `AttributeError`, `ZeroDivisionError`, `DataAccessError` | return `0.0` | Pre-run sentinel |
| `BoxMCStatus.acceptance_rate` | same | return `0.0` | Pre-run sentinel |
| `OverlapCount.overlap_count` | `DataAccessError` | return `0.0` | Pre-run sentinel |
| `MoveSizeProp.d` | `DataAccessError` | return `0.0` | Pre-run sentinel |
| `BoxSeqProp.*_moves_str` | `DataAccessError` | return `"0,0"` | Pre-run sentinel |
| `SDFPressure.betaP` | `DataAccessError`, `TypeError` | return `0.0` | Pre-SDF sentinel |
| `SDFPressure.compressibility_Z` | `DataAccessError`, `TypeError`, `ZeroDivisionError` | return `0.0` | Pre-SDF sentinel |
| GPU init | `Exception` | warn + CPU fallback | `[WARNING] GPU init failed` |
| `sim.run()` (equil or prod) | any `Exception` | emergency snapshot + re-raise | `[ERROR]` + debug_kv blocks |
| Emergency snapshot write | `Exception` | report + continue | Cannot mask original exc. |
| Log handle flush/close | `Exception` | pass | Best-effort, non-masking |
| `__main__` `SystemExit` | `SystemExit` | re-raise | Already printed a message |
| `__main__` `Exception` | `Exception` | `traceback.print_exc()` + `sys.exit(1)` | Full stack trace |

---

## 28. Troubleshooting

### `[FATAL] Parameter file not found`

The path passed to `--simulparam_file` does not exist. Use an absolute path if running from a different directory.

### `[FATAL] JSON parse error`

Syntax error in the JSON. Common causes: trailing comma after the last key, `//` comments (not valid in JSON — use `_`-prefixed keys instead), missing quotes around string values. The error includes the line and column number.

### `[FATAL] Missing required keys`

A required key is absent. The error message lists all missing keys.

### `[FATAL] Type errors`

A key has the wrong Python type (e.g. `"pressure": "11.0"` instead of `"pressure": 11.0`). The error lists all type mismatches.

### `Parameter validation failed`

Multiple physics/logic problems found in the parameters. All are listed at once.

### `[ERROR] Stage N final GSD already exists`

The stage completed successfully. Increment `stage_id_current` in the JSON to proceed to the next stage.

### `[FATAL] Previous stage output not found`

Stage N-1's `_final.gsd` is missing. Check that the previous stage completed and wrote its final GSD.

### `[WARNING] BoxMCMoveSize tuner did NOT converge`

The tuner did not converge within `equil_steps`. The simulation continues with the tuner active during production. Consider increasing `equil_steps` or reducing the BoxMCMoveSize `tol` parameter (requires script modification; it is hardcoded to 0.01).

### Box acceptance rate is 0.0 at start

Normal — `BoxMCStatus` returns 0.0 before `sim.run()` because `DataAccessError` is caught. Real acceptance rates appear after the first chunk of the equilibration loop.

### Very low box acceptance rate (< 5%)

Box-move deltas are too large. Decrease `boxmc_volume_delta`, `boxmc_length_delta`, etc. in the JSON.

### Very high box acceptance rate (> 60%)

Box-move deltas are too small. Increase them. The tuner will adapt them automatically during equilibration, but starting closer to the optimal values saves equilibration time.

### `SDF/betaP` is always 0.0

Either `enable_sdf=false` in the JSON, or the SDF compute has not yet been attached (you are looking at equilibration-phase log entries). The SDF is only attached after the box tuner is removed. Check that equilibration is completing and the SDF attachment message appears in the log.

### `<betaP>` does not converge to `pressure`

The system has not equilibrated. Increase `equil_steps`. Alternatively, the starting configuration (from the input GSD) is too far from the target pressure — run at a closer pressure first and use that final GSD as input.

### Non-zero `HPMC/overlaps`

Should never happen in a valid run. If non-zero:
1. Check the input GSD was produced by a successful run with zero overlaps.
2. Check `mc.shape["A"]["diameter"]` matches the GSD diameter (they should be equal — both come from `read_mono_diameter_from_gsd`).
3. Report the emergency snapshot and full traceback.

---

## 29. Version History and Feature Changelog

| Version | Key changes |
|---|---|
| `hard_sphere_npt_v1p11.py` | Original script: hard-coded parameters, no restart, no stage management |
| `hs_npt_v2.py` | All 20 features [N-01]–[N-20] added vs v1 (see list in file header) |
| `hs_npt_v3.py` | User modifications: `fail_with_context`, `debug_kv`, enhanced snapshot validation with shape checks, per-writer try/except blocks with detailed error messages, BoxMCStatus per-move-type rates [N-18], BoxSeqProp DataAccessError guard [N-19], `sim.create_state_from_gsd()` for restart path |
| `hs_npt_v4.py` | **Current version**: 474 lines of inline documentation added throughout. Zero algorithm changes from v3. All exception handlers, design decisions, and physical choices explained at the point of implementation. |

The `[N-XX]` tags throughout the source code reference the feature list in the file header, which maps each new capability to the corresponding code location.

---

*Documentation for `hs_npt_v4.py` | HOOMD-blue v4.9 Hard-Sphere HPMC NPT Simulation*
