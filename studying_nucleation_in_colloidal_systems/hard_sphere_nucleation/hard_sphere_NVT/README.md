# `hs_nvt_v4.py` — Hard-Sphere HPMC NVT Equilibration

**HOOMD-blue v4 | Production-ready NVT equilibration script for monodisperse hard spheres**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Where This Script Fits in the Pipeline](#2-where-this-script-fits-in-the-pipeline)
3. [Physics: What NVT Means for Hard Spheres](#3-physics-what-nvt-means-for-hard-spheres)
4. [Algorithm](#4-algorithm)
5. [File Inventory](#5-file-inventory)
6. [Dependencies](#6-dependencies)
7. [Installation](#7-installation)
8. [Quick Start](#8-quick-start)
9. [JSON Parameter Reference](#9-json-parameter-reference)
10. [Multi-Stage Runs](#10-multi-stage-runs)
11. [Output Files Reference](#11-output-files-reference)
12. [Restart and Recovery](#12-restart-and-recovery)
13. [MPI Parallel Execution](#13-mpi-parallel-execution)
14. [GPU Execution](#14-gpu-execution)
15. [Code Architecture](#15-code-architecture)
    - [Section 1: MPI Console Helpers](#section-1-mpi-console-helpers)
    - [Section 2: Custom Loggable Classes](#section-2-custom-loggable-classes)
    - [Section 3: SimulationParams Dataclass](#section-3-simulationparams-dataclass)
    - [Section 4: JSON Loading and Validation](#section-4-json-loading-and-validation)
    - [Section 5: Random Seed Management](#section-5-random-seed-management)
    - [Section 6: Filename Resolution](#section-6-filename-resolution)
    - [Section 7: GSD Diameter Reader](#section-7-gsd-diameter-reader)
    - [Section 8: MPI Snapshot Broadcast](#section-8-mpi-snapshot-broadcast)
    - [Section 9: Simulation Builder](#section-9-simulation-builder)
    - [Section 10: Output Writing](#section-10-output-writing)
    - [Section 11: Banner Helper](#section-11-banner-helper)
    - [Section 12: Entry Point and Exception Handling](#section-12-entry-point-and-exception-handling)
16. [Exception Handling Map](#16-exception-handling-map)
17. [Logged Quantities Reference](#17-logged-quantities-reference)
18. [The HDF5 Diagnostics File](#18-the-hdf5-diagnostics-file)
19. [Diameter Persistence: How and Why](#19-diameter-persistence-how-and-why)
20. [Seed Reproducibility](#20-seed-reproducibility)
21. [Choosing move_size_translation (d)](#21-choosing-move_size_translation-d)
22. [Reading Output Files in Python](#22-reading-output-files-in-python)
23. [HPC Cluster / SLURM Job Script](#23-hpc-cluster--slurm-job-script)
24. [Troubleshooting](#24-troubleshooting)
25. [Version History and Changes from v1](#25-version-history-and-changes-from-v1)

---

## 1. Overview

`hs_nvt_v4.py` runs a **fixed-box (NVT) Hard-Particle Monte Carlo (HPMC)** equilibration of a monodisperse hard-sphere fluid using HOOMD-blue v4. It is the fourth iteration of this script, adding full inline documentation and exception handling to `hs_nvt_v3.py` without changing a single line of algorithm or logic.

The script is designed to be used immediately after a compression run (e.g. `hs_compress_v7.py`) has brought the system to the desired packing fraction. Its job is to equilibrate the spatial arrangement of particles at **constant volume and constant particle count** before any measurement run (EOS, nucleation, g(r), S(q)) begins.

**Key characteristics at a glance:**

| Property | Value |
|---|---|
| Ensemble | NVT (constant N, V, T; in HPMC, T is implicit) |
| Integrator | `hoomd.hpmc.integrate.Sphere` |
| Move type | Translation only (`nselect=1`, fixed `d`) |
| Move size adaptation | None (fixed `d` throughout) |
| MPI support | Yes (via mpi4py; serial fallback if absent) |
| GPU support | Yes (selectable via JSON) |
| Restart-safe | Yes (automatic checkpoint detection) |
| Multi-stage | Yes (pipeline-compatible stage IDs) |
| Diameter in all GSD files | Yes (`dynamic=` + `write_diameter=True`) |
| HDF5 diagnostics | Yes (optional; graceful fallback) |

---

## 2. Where This Script Fits in the Pipeline

The typical hard-sphere research pipeline is:

```
[1] make_lattice_phi.py          ← create initial low-density GSD at target φ
         │
         ▼
[2] hs_compress_v7.py            ← compress to target φ using manual box-resize loop
         │  outputs: *_compressed_to_pf_*.gsd  (labelled snapshot)
         │           *_final.gsd               (input to next stage)
         ▼
[3] hs_nvt_v4.py        ◄──── YOU ARE HERE
         │  inputs:  *_final.gsd from compression
         │  outputs: *_traj.gsd, *_final.gsd, *.log, *_diagnostics.h5
         ▼
[4] EOS measurement run          ← measure Z(φ) via hpmc.compute.SDF
    or
[4] Nucleation run               ← seeded / spontaneous nucleation at φ > 0.494
    or
[4] g(r) / S(q) analysis         ← structural characterisation
```

The NVT stage is critical because the compressed configuration has a strong memory of the initial lattice. Equilibration randomises the particle arrangement so that structural measurements reflect the true equilibrium fluid (or crystal) at that density, not an artifact of the compression path.

---

## 3. Physics: What NVT Means for Hard Spheres

In HPMC, there is no temperature in the conventional sense. The Boltzmann factor for hard spheres is either 0 (overlap) or 1 (no overlap). The Metropolis criterion therefore reduces to: **accept any move that does not create an overlap; reject all others.**

"NVT" means:
- **N** (number of particles): fixed, read from the input GSD.
- **V** (volume): fixed, i.e. the simulation box dimensions do not change. BoxMC is not attached; no pressure coupling.
- **T** (temperature): irrelevant to the equilibrium structural properties of hard spheres. kBT = 1 is the implicit unit.

The equilibration is complete when:
- The **translational acceptance ratio** has stabilised (typically 20–30% for a well-chosen `d`).
- The **mean-squared displacement** of particles from their compressed starting positions is on the order of σ² or larger (particles have moved at least one diameter).
- The **radial distribution function g(r)** has converged (checked by comparing g(r) from early and late portions of the trajectory).
- The **overlap count** remains exactly 0 throughout (any non-zero value indicates a bug or corrupt input).

---

## 4. Algorithm

The script executes the following sequence. No step is skipped; no adaptive logic changes the fundamental flow.

```
1.  Parse CLI: --simulparam_file <path>
2.  Print HOOMD version banner
3.  Load JSON → SimulationParams (type-checked, validated)
4.  Determine MPI rank from environment variables
5.  Resolve filenames (stage-aware naming)
6.  Manage seed file: rank-0 writes random_seed.json once; all ranks read it
7.  Poll for seed file availability (file-based MPI barrier)
8.  Build simulation:
      a. Select device (CPU / GPU with fallback)
      b. Create hoomd.Simulation(device, seed)
      c. Detect run mode:
           restart_gsd exists AND final_gsd absent → RESTART from checkpoint
           otherwise                               → FRESH RUN from input GSD
      d. Load GSD state via sim.create_state_from_gsd()
      e. Read σ from GSD; compute initial φ and print box info
      f. Attach HPMC Sphere integrator (nselect=1, fixed d)
      g. Construct custom loggables (Status, MCStatus, Box_property,
         OverlapCount, CurrentTimestep)
      h. Attach Table log writer → .log file (human-readable)
      i. Attach GSD trajectory writer (mode: "ab" restart / "wb" fresh)
      j. Attach GSD restart writer (truncate=True, always "wb")
      k. Attach HDF5 diagnostics writer (optional; skipped if h5py missing)
9.  sim.run(total_num_timesteps)
10. Write final outputs:
      - _write_snapshot → final GSD (with explicit diameter)
      - summary JSON
      - console banner
11. finally block:
      - flush all HOOMD writers
      - close log file handle
      - print total runtime
```

---

## 5. File Inventory

| File | Role |
|---|---|
| `hs_nvt_v4.py` | Main simulation script |
| `simulparam_hs_nvt.json` | Parameter file (one per run; edit to change settings) |
| `random_seed.json` | Auto-generated seed file (single-stage runs) |
| `random_seed_stage_0.json` | Auto-generated seed file (multi-stage runs) |

**Output files** (single-stage defaults, customisable via JSON):

| Output file | When written | Contents |
|---|---|---|
| `nvt_hpmc_log.log` | Every `log_frequency` steps | Human-readable table: tps, walltime, ETR, timestep, acceptance_rate, volume, φ, overlaps |
| `nvt_hpmc_output_traj.gsd` | Every `traj_gsd_frequency` steps | Full trajectory with particle positions + diameters |
| `nvt_hpmc_output_restart.gsd` | Every `restart_gsd_frequency` steps | Single-frame checkpoint (truncated on each write) |
| `nvt_hpmc_diagnostics.h5` | Every `diagnostics_frequency` steps | HDF5 binary: tps, walltime, timestep, acceptance_rate, φ, overlaps, translate_moves, mps |
| `nvt_hpmc_final.gsd` | End of run | Single-frame final state with full diameter data |
| `<tag>_stage<id>_nvt_summary.json` | End of run | Machine-readable run summary |
| `emergency_restart_<tag>.gsd` | On unexpected exception | Best-effort recovery snapshot |

---

## 6. Dependencies

### Required

| Package | Minimum version | Purpose |
|---|---|---|
| Python | 3.8+ | Language runtime |
| HOOMD-blue | 4.0 | HPMC simulation engine |
| GSD | 3.0 | File I/O for simulation trajectories |
| NumPy | 1.20+ | Array operations |

### Optional

| Package | Purpose | Behaviour if absent |
|---|---|---|
| mpi4py | MPI-parallel execution | Script falls back to single-process execution; a stub `MPI.COMM_WORLD.bcast()` is provided |
| h5py | HDF5 diagnostics writer | `[WARNING] HDF5 diagnostics writer disabled` is printed; the run continues normally with only the Table log and GSD trajectory |

---

## 7. Installation

### Conda (recommended)

```bash
conda create -n hoomd4 python=3.10
conda activate hoomd4
conda install -c conda-forge hoomd gsd numpy mpi4py h5py
```

### Pip

```bash
pip install hoomd gsd numpy mpi4py h5py
```

### Verify

```bash
python -c "import hoomd; print(hoomd.version.version)"
python -c "import gsd; print(gsd.version.version)"
```

---

## 8. Quick Start

### 1. Prepare a JSON parameter file

Copy the example below to `simulparam_hs_nvt.json` in your working directory and adjust the values:

```json
{
    "tag":                   "hs_4096_nvt",
    "input_gsd_filename":    "hard_sphere_4096_compression_final.gsd",
    "stage_id_current":      -1,
    "total_num_timesteps":   1000000,
    "move_size_translation": 0.08,
    "log_frequency":         10000,
    "traj_gsd_frequency":    100000,
    "restart_gsd_frequency": 10000,
    "use_gpu":               false,
    "gpu_id":                0,
    "output_trajectory":     "nvt_hpmc_output_traj.gsd",
    "log_filename":          "nvt_hpmc_log.log",
    "restart_file":          "nvt_hpmc_output_restart.gsd",
    "final_gsd_filename":    "nvt_hpmc_final.gsd"
}
```

### 2. Run (serial)

```bash
python hs_nvt_v4.py --simulparam_file simulparam_hs_nvt.json
```

### 3. Run (MPI, 8 processes)

```bash
mpirun -n 8 python3.10 hard_sphere_NVT.py --simulparam_file simulparam_hard_sphere_nvt.json
```

### 4. Check output

```
*********************************************************
HOOMD-blue version:   4.x.x
*********************************************************
[INFO] Loaded parameters from 'simulparam_hs_nvt.json'
[INFO] Diameter read from GSD '...' = 1.0
[INFO] N=4096 | sphere_diameter=1.0 | phi=0.500000 | box=...
[INFO] HPMC Sphere configured | particle_diameter(sigma)=1.0 | translation_move_size(d)=0.08
[INFO] Starting NVT run | 1000000 steps | traj every 100000 | restart every 10000
...
[OUTPUT] Final GSD        => nvt_hpmc_final.gsd
[OUTPUT] Summary JSON     => hs_4096_nvt_stage-1_nvt_summary.json
```

---

## 9. JSON Parameter Reference

All parameters are read from a JSON file. Keys beginning with `_` are treated as comments and stripped before parsing — you can annotate the file freely.

### Required keys

| Key | Type | Description |
|---|---|---|
| `tag` | `str` | Unique identifier for this run. Used as a prefix for all output filenames in multi-stage mode. Choose something descriptive, e.g. `"hs_4096_pf0p50_nvt"`. |
| `input_gsd_filename` | `str` | Path to the GSD file that provides the initial particle configuration. In single-stage mode (`stage_id_current=-1`) this is always used. In multi-stage mode with `stage_id_current >= 1`, this field is only used for stage 0; later stages read their input from the previous stage's `_final.gsd`. |
| `stage_id_current` | `int` | Stage index. Use `-1` for a simple single run. Use `0, 1, 2, ...` for chained multi-stage runs where each stage continues from where the previous left off. See [Multi-Stage Runs](#10-multi-stage-runs) for full details. |
| `total_num_timesteps` | `int` | Total number of HPMC sweeps to run. One sweep = one trial move per particle. Must be > 0. |
| `move_size_translation` | `float` or `int` | Maximum translational displacement magnitude `d` [in σ units]. Every trial move draws `Δr` uniformly from a cube of side `2d` centred on the particle. **This value is fixed throughout the run — there is no adaptive tuner.** See [Choosing d](#21-choosing-move_size_translation-d) for guidance. Must be > 0. |
| `log_frequency` | `int` | Write one row to the `.log` file every this many steps. Also used as the HDF5 diagnostics period when `diagnostics_frequency = 0`. Must be > 0. |
| `traj_gsd_frequency` | `int` | Append one frame to the trajectory GSD every this many steps. Must be > 0. |
| `restart_gsd_frequency` | `int` | Overwrite the single-frame restart checkpoint GSD every this many steps. Must be > 0. Set this smaller than `traj_gsd_frequency` so you never lose more than `restart_gsd_frequency` steps of progress. |
| `use_gpu` | `bool` | `true` to request GPU execution; `false` for CPU. If the GPU is unavailable, the script falls back to CPU with a `[WARNING]` message. |
| `gpu_id` | `int` | CUDA device index (0-based) to use when `use_gpu=true`. Ignored when `use_gpu=false`. |

### Optional keys (with defaults)

| Key | Type | Default | Description |
|---|---|---|---|
| `diagnostics_frequency` | `int` | `0` | Period for the HDF5 diagnostics writer. If `0` (default), the diagnostics writer uses `log_frequency` as its period. Must be >= 0. |
| `initial_timestep` | `int` | `0` | Currently stored in the dataclass but not used to override the timestep at load time (HOOMD restores the timestep from the GSD file on restarts). Reserved for future use. |
| `output_trajectory` | `str` | `"nvt_hpmc_output_traj.gsd"` | Filename for the GSD trajectory. Ignored in multi-stage mode (`stage_id_current >= 0`). |
| `log_filename` | `str` | `"nvt_hpmc_log.log"` | Filename for the human-readable Table log. Ignored in multi-stage mode. |
| `restart_file` | `str` | `"nvt_hpmc_output_restart.gsd"` | Filename for the restart checkpoint GSD. Ignored in multi-stage mode. |
| `final_gsd_filename` | `str` | `"nvt_hpmc_final.gsd"` | Filename for the final one-shot GSD written at the end of the run. Ignored in multi-stage mode. |
| `hdf5_log_filename` | `str` | `"nvt_hpmc_diagnostics.h5"` | Filename for the HDF5 diagnostics file. Ignored in multi-stage mode. |

### Comment keys

Any key whose name starts with `_` is silently stripped before parsing. Use this to annotate your JSON:

```json
{
    "_note": "NVT equilibration at phi=0.50 for N=4096",
    "_date": "2025-03-15",
    "tag": "hs_4096_phi050_nvt",
    ...
}
```

---

## 10. Multi-Stage Runs

Multi-stage mode is designed for two scenarios:

1. **Long runs on HPC clusters** where each job has a walltime limit and the run must be spread over multiple submissions.
2. **Sequential analysis** where you want separate, clearly labelled GSD files for each phase of a simulation (e.g. equilibration stage 0, equilibration stage 1, production stage 2).

### How it works

When `stage_id_current >= 0`, the script ignores the single-stage filename fields in the JSON and instead derives all filenames automatically:

| File | Resolved name |
|---|---|
| Trajectory | `<tag>_<stage_id>_traj.gsd` |
| Restart checkpoint | `<tag>_<stage_id>_restart.gsd` |
| Final snapshot | `<tag>_<stage_id>_final.gsd` |
| Table log | `<tag>_<stage_id>.log` |
| HDF5 diagnostics | `<tag>_<stage_id>_diagnostics.h5` |

The input for each stage comes from:
- Stage 0: `input_gsd_filename` from the JSON (e.g. the compression output).
- Stage 1: `<tag>_0_final.gsd`
- Stage N: `<tag>_{N-1}_final.gsd`

### Safety guards

- If `<tag>_<stage_id>_final.gsd` already exists when the script starts, it **exits immediately** with an `[ERROR]` message. This prevents accidentally overwriting a completed stage. To continue, increment `stage_id_current` in the JSON.
- If the previous stage's `_final.gsd` is missing for stages > 0, the script exits with a `[FATAL]` message telling you which stage failed.

### Example: 3-stage equilibration (3 × 10M steps)

**Stage 0** — edit JSON: `"stage_id_current": 0, "total_num_timesteps": 10000000`
```bash
python hs_nvt_v4.py --simulparam_file simulparam_hs_nvt.json
# produces: hs_4096_nvt_0_traj.gsd, hs_4096_nvt_0_final.gsd, hs_4096_nvt_0.log
```

**Stage 1** — edit JSON: `"stage_id_current": 1`
```bash
python hs_nvt_v4.py --simulparam_file simulparam_hs_nvt.json
# reads: hs_4096_nvt_0_final.gsd
# produces: hs_4096_nvt_1_traj.gsd, hs_4096_nvt_1_final.gsd
```

**Stage 2** — edit JSON: `"stage_id_current": 2`
```bash
python hs_nvt_v4.py --simulparam_file simulparam_hs_nvt.json
# reads: hs_4096_nvt_1_final.gsd
# produces: hs_4096_nvt_2_traj.gsd, hs_4096_nvt_2_final.gsd
```

---

## 11. Output Files Reference

### `*.log` — Human-readable Table log

A plain-text, space-separated file written by `hoomd.write.Table` every `log_frequency` steps. The first row contains column headers; subsequent rows contain values. Parsable directly with `numpy.loadtxt`, `pandas.read_csv(sep=r'\s+')`, or any spreadsheet.

**Columns:**

| Column header | Source | Description |
|---|---|---|
| `Simulation/tps` | HOOMD built-in | Timesteps per second (performance) |
| `Simulation/walltime` | HOOMD built-in | Elapsed wall-clock time in seconds |
| `Simulation/timestep` | `Status.timestep_fraction` | String `"current/total"` showing progress |
| `Status/etr` | `Status.etr` | Estimated time remaining as `HH:MM:SS` |
| `MCStatus/acceptance_rate` | `MCStatus.acceptance_rate` | **Windowed** translational acceptance ratio since the last log entry. Not cumulative. |
| `Box/volume` | `Box_property.volume` | Current simulation box volume V [σ³] |
| `Box/packing_fraction` | `Box_property.packing_fraction` | φ = N·(π/6)·σ³/V. In NVT this is constant. Any drift indicates a bug. |
| `HPMC/overlap_count` | `OverlapCount.overlap_count` | Number of overlapping particle pairs. **Must always be 0 in a valid NVT run.** |

### `*_traj.gsd` — Particle trajectory

A GSD binary file appended every `traj_gsd_frequency` steps. Contains, in every frame: box dimensions, all particle positions, all particle diameters, particle types, type IDs, and (when present in the source GSD) particle image flags.

The file mode is:
- `"ab"` (append-binary) when resuming from a restart checkpoint — new frames continue the existing trajectory seamlessly.
- `"wb"` (write-binary) on fresh runs — starts a clean file.

### `*_restart.gsd` — Checkpoint file

A single-frame GSD file (`truncate=True`) overwritten every `restart_gsd_frequency` steps. Contains the complete simulation state at the checkpoint timestep, including positions, diameters, box, and the current timestep counter. Used exclusively by the restart detection logic at the next run's startup.

**Important:** This file always contains exactly **one frame** — the most recent checkpoint. It is not a history.

### `*_diagnostics.h5` — HDF5 binary diagnostics

An HDF5 file written by `hoomd.write.HDF5Log` every `diagnostics_period` steps (defaults to `log_frequency` if `diagnostics_frequency = 0`). Unlike the `.log` file, HDF5 provides random-access to any timestep, is compact, and is trivially readable from Python or Julia.

See [The HDF5 Diagnostics File](#18-the-hdf5-diagnostics-file) for the full dataset map and Python reading examples.

### `*_final.gsd` — Final snapshot

A single-frame GSD written at the end of a successful run. Particle diameters are explicitly written (independent of HOOMD defaults) via the custom `_write_snapshot()` function which builds the GSD frame manually. This file is the intended input to the next stage or to analysis scripts.

### `<tag>_stage<id>_nvt_summary.json` — Run summary

A machine-readable JSON containing all key run metadata. Fields:

```json
{
  "tag":                "hs_4096_nvt",
  "stage_id":           -1,
  "simulparam_file":    "simulparam_hs_nvt.json",
  "input_gsd":          "hard_sphere_4096_compression_final.gsd",
  "traj_gsd":           "nvt_hpmc_output_traj.gsd",
  "diagnostics_hdf5":   "nvt_hpmc_diagnostics.h5",
  "final_gsd":          "nvt_hpmc_final.gsd",
  "n_particles":        4096,
  "diameter":           1.0,
  "packing_fraction":   0.5,
  "move_size_d":        0.08,
  "diagnostics_frequency": 10000,
  "total_timesteps":    1000000,
  "final_timestep":     1000000,
  "overlaps_final":     0,
  "random_seed":        42731,
  "box_final":          {"Lx": 18.42, "Ly": 18.42, "Lz": 18.42},
  "runtime_seconds":    3612.5,
  "created":            "2025-03-15T14:22:01"
}
```

### `emergency_restart_<tag>.gsd` — Emergency snapshot

Written only if an **unexpected exception** occurs during `sim.run()`. Built the same way as the final snapshot (explicit diameters, manual GSD frame construction). Use this to inspect the state at the time of the crash or to restart from the last safe point.

---

## 12. Restart and Recovery

The script uses a two-file detection scheme at every startup:

```
restart_GSD_exists  AND  final_GSD_absent  →  RESTART from checkpoint
any other combination                       →  FRESH RUN from input GSD
```

This logic handles all practically relevant cases:

| Scenario | `restart.gsd` | `final.gsd` | Script action |
|---|---|---|---|
| Clean first run | absent | absent | Fresh run |
| Run hit walltime / crashed | present | absent | Restart from checkpoint |
| Run completed successfully | present | present | Fresh run (ignores checkpoint) |
| Re-running a completed stage | absent | present | Fresh run with [WARNING] |

### To restart a crashed run

Simply resubmit with the **identical JSON file**. The script detects the restart checkpoint automatically and continues from where it left off.

```bash
# First submission (crashed at step 350,000):
python hs_nvt_v4.py --simulparam_file simulparam_hs_nvt.json

# Resubmission (picks up from last checkpoint automatically):
python hs_nvt_v4.py --simulparam_file simulparam_hs_nvt.json
```

### Trajectory continuity on restart

Because the trajectory writer uses `mode="ab"` on restarts and `mode="wb"` on fresh runs, restarting produces a **continuous trajectory file** — frames from before and after the interruption are in the same GSD file in timestep order. You do not need to concatenate files manually.

### Checkpoint frequency recommendation

Set `restart_gsd_frequency` much smaller than `traj_gsd_frequency` but large enough not to dominate I/O time. For a typical NVT run:

```
restart_gsd_frequency  :  5,000  – 10,000 steps
traj_gsd_frequency     :  50,000 – 100,000 steps
log_frequency          :  5,000  – 10,000 steps
```

---

## 13. MPI Parallel Execution

### Serial runs (no mpi4py)

If mpi4py is not installed, the script substitutes a lightweight stub class that exposes `MPI.COMM_WORLD.bcast(obj, root=0)` returning `obj` unchanged. All code paths execute as if there is a single MPI rank (rank 0). No code changes are needed.

### MPI runs

HOOMD-blue uses **spatial domain decomposition**: the simulation box is divided into rectangular subdomains, one per MPI rank. Each rank owns a subset of particles. HOOMD's internal MPI layer handles all inter-rank communication for neighbour lists, overlap detection, and I/O.

The script handles MPI explicitly in only two places:

1. **Rank detection** (`_mpi_rank_from_env()`): reads the MPI rank from common launcher environment variables **before** the HOOMD device is initialised. This is needed so that `root_print()` can suppress duplicate console output from all N ranks before the `sim.device.communicator.rank` attribute is available.

2. **`root_print()` and `root_flush_stdout()`**: every console print goes through these wrappers so that a run with 8 MPI ranks does not produce 8 copies of every log line.

State loading uses `sim.create_state_from_gsd()` (HOOMD v4 recommended approach): rank 0 reads the GSD, HOOMD decomposes the domain internally, and distributes particles to all ranks. No manual `comm.bcast()` is needed.

### MPI rank detection priority

The `_mpi_rank_from_env()` function checks environment variables in this order:

| Priority | Variable | Set by |
|---|---|---|
| 1 | `OMPI_COMM_WORLD_RANK` | Open MPI (`mpirun`) |
| 2 | `PMI_RANK` | MPICH, Intel MPI, Cray MPICH |
| 3 | `SLURM_PROCID` | SLURM `srun` |
| 4 | Default 0 | Serial run |

---

## 14. GPU Execution

Set `"use_gpu": true` and `"gpu_id": 0` in the JSON. HOOMD will use the CUDA device with the specified index.

```json
{
    "use_gpu": true,
    "gpu_id":  0
}
```

If GPU initialisation fails for any reason (CUDA not available, invalid device ID, insufficient compute capability), the script **automatically falls back to CPU** with a `[WARNING]` message rather than crashing. This makes job scripts more portable across heterogeneous cluster nodes.

**Note**: HOOMD-blue HPMC on GPU uses a different parallelisation strategy than MPI+CPU. For large systems (N > 50,000) on a single node, GPU is typically faster than many CPU cores. For smaller systems, the overhead of GPU data transfer may outweigh the benefit.

---

## 15. Code Architecture

The script is divided into 12 numbered sections, each responsible for one well-defined concern. The section boundaries are clearly marked with `# ===========================================================================` headers.

### Section 1: MPI Console Helpers

**Functions:** `_mpi_rank_from_env()`, `_is_root_rank()`, `root_print()`, `root_flush_stdout()`

These four functions solve a single problem: in an MPI run with N processes, every `print()` call would produce N identical lines on the terminal. By routing all console output through `root_print()`, only rank 0 ever calls `print()`. `root_flush_stdout()` ensures the output buffer is flushed after important progress messages so they appear immediately on screen even if the terminal output is being redirected to a file.

`_mpi_rank_from_env()` is called before the HOOMD `Simulation` object exists (and therefore before `sim.device.communicator.rank` is available). It reads the rank from standard launcher environment variables — a portable approach that works under `mpirun`, `srun`, and plain Python.

### Section 2: Custom Loggable Classes

HOOMD's logging system works by introspecting Python objects for a special `_export_dict` class attribute. When you register an object with a `Logger`, HOOMD calls the object's property getters once at registration time to validate them. This has an important consequence: **every property getter must be safe to call before `sim.run()` has been called** — otherwise `DataAccessError` is raised and the script crashes at startup before any simulation steps run.

All five custom loggable classes in this section implement this guard.

#### `Status`

Provides `timestep_fraction` (string: `"current/total"`) and `etr` (string: estimated time remaining as `HH:MM:SS`). Both properties catch `AttributeError` (raised when `final_timestep` is not yet set by `sim.run()`) and return harmless sentinels `"0/?"` and `"0:00:00"` respectively.

#### `MCStatus`

Provides `acceptance_rate` (scalar): the **windowed** translational acceptance ratio computed as the fraction of accepted moves in the interval since the **last log entry**, not the cumulative fraction since the run started. This is the correct quantity to monitor: the cumulative rate converges to a fixed value over time and loses sensitivity to recent changes in the system, while the windowed rate reflects the current state of the simulation. Catches `IndexError` (empty tuple), `ZeroDivisionError` (no moves yet), and `DataAccessError` (integrator not yet run).

#### `Box_property`

Provides `volume` (scalar, V in σ³) and `packing_fraction` (scalar, φ = N·(π/6)·σ³/V). In an NVT run the box does not change, so both values are constant throughout — any change would indicate a bug. Having φ in the log is the simplest possible sanity check.

#### `OverlapCount`

Provides `overlap_count` (scalar): the number of overlapping particle pairs from the most recent HPMC sweep. In a valid NVT configuration this is always 0. A non-zero value at any point means the input GSD contained overlaps (bad compression output) or a HOOMD bug has allowed an illegal move.

#### `CurrentTimestep`

Provides `timestep` (scalar): the current simulation timestep as a float. Used exclusively by the HDF5 logger, which cannot accept string quantities. The Table log uses `Status.timestep_fraction` instead for the human-readable display.

### Section 3: SimulationParams Dataclass

A Python `@dataclass` with type-annotated fields corresponding exactly to every JSON key. The `validate()` method runs physics and logic checks after construction:
- `total_num_timesteps > 0`
- `move_size_translation > 0`
- All frequency parameters `> 0`
- `stage_id_current >= -1`
- `diagnostics_frequency >= 0`

Errors are collected into a list before raising `ValueError`, so the user sees **all** problems at once rather than one at a time.

The `diameter` field has no JSON equivalent — it is `Optional[float] = None` at construction time and is populated at runtime by `read_mono_diameter_from_gsd()` after the GSD is loaded.

### Section 4: JSON Loading and Validation

`load_simulparams()` performs four validation steps in order:

1. **File existence check**: immediately exits if the JSON file is not found, giving a clean error instead of a confusing Python traceback.
2. **JSON syntax check**: catches `json.JSONDecodeError` (subclass of `ValueError`) for malformed JSON.
3. **Required key presence**: reports all missing required keys at once.
4. **Type checking**: checks every required key against its expected Python type. `move_size_translation` accepts both `int` and `float` (the JSON number type does not distinguish them).

Comment keys (starting with `_`) are stripped before any of these checks, allowing freely annotated JSON files.

### Section 5: Random Seed Management

HOOMD v4 and v5 accept RNG seeds only in the range `[0, 65535]` (unsigned 16-bit integer). Seeds outside this range are silently truncated with a `*Warning*` message, which would change the actual seed and break reproducibility. The script enforces this range by using `secrets.randbelow(65536)`.

The seed is written once to a JSON file (`random_seed.json` for single-stage, `random_seed_stage_0.json` for multi-stage) on the first invocation of the script. All subsequent invocations — restarts, later stages, all MPI ranks — read from this file to ensure the entire pipeline uses the identical seed. The creation timestamp is also stored for audit purposes.

After writing the seed, the script polls for the seed file's existence with a 30-second timeout. In large MPI jobs under some filesystems (Lustre, GPFS), a file written by rank 0 may not be immediately visible to other ranks due to metadata caching. The polling loop is the safest cross-platform barrier before HOOMD's own MPI barrier is available.

### Section 6: Filename Resolution

`resolve_filenames()` implements two modes controlled by `stage_id_current`:

**Single-stage (`stage_id = -1`)**: filenames are taken directly from the JSON. If the final GSD already exists, a `[WARNING]` is printed but the run proceeds (the user may intentionally re-run a single-stage job).

**Multi-stage (`stage_id >= 0`)**: all output filenames are derived as `<tag>_<stage_id>_<suffix>`. If the current stage's final GSD already exists, the script **exits** — this is a hard stop to prevent accidental overwriting of completed data. The input GSD for stage N > 0 is `<tag>_{N-1}_final.gsd`; if this is missing the script exits with a diagnostic message.

### Section 7: GSD Diameter Reader

`read_mono_diameter_from_gsd()` opens the last frame of a GSD file and returns the common particle diameter. It validates:
- The file exists (pre-checks before calling the GSD C extension, giving cleaner errors).
- The file has at least one frame.
- The diameter array is non-empty.
- All diameters are positive.
- All diameters are identical within 1×10⁻¹² (the `1e-12` tolerance accounts for float32→float64 round-trip error in the GSD file format).

### Section 8: MPI Snapshot Broadcast

`load_and_broadcast_snapshot()` and `reconstruct_snapshot()` implement the original v1 script's MPI-safe initialisation pattern. They are preserved for reference and compatibility but are **not called on the main code path** in v4: the fresh-run branch now uses `sim.create_state_from_gsd()` directly (see [N-20]). These functions remain in the code because the restart branch also uses `create_state_from_gsd()` and the broadcast functions are needed for documentation completeness.

### Section 9: Simulation Builder

`build_simulation()` is the core of the script. It builds the complete HOOMD simulation object in 10 substeps:

| Step | What happens |
|---|---|
| 9.1 | Device selection: GPU with fallback to CPU |
| 9.2 | `hoomd.Simulation(device, seed)` created |
| 9.3 | GSD state loaded (restart vs fresh detection) |
| 9.4 | Diameter read; N and φ computed and printed |
| 9.5 | HPMC Sphere integrator configured and attached |
| 9.6 | Custom loggable instances constructed |
| 9.7 | `hoomd.logging.Logger` + `hoomd.write.Table` writer attached |
| 9.8 | GSD trajectory writer attached (`"ab"` or `"wb"` mode) |
| 9.9 | GSD restart writer attached (`truncate=True`) |
| 9.10 | HDF5 diagnostics writer attached (optional, in `try/except`) |

Returns `(sim, mc, log_file_hdl)`. The log file handle is returned to `main()` so it can be safely closed in the `finally` block even if `sim.run()` raises an exception.

### Section 10: Output Writing

`_write_snapshot()` manually constructs a `gsd.hoomd.Frame` from `sim.state.get_snapshot()` and writes it as a single-frame GSD. This path writes `particles.diameter` **explicitly** in every call, independent of HOOMD's periodic writer configuration. It also copies `particles.image` (particle image flags for unwrapped positions) when available, and silently skips them when absent.

`write_final_outputs()` calls `_write_snapshot()` for the final GSD, writes the summary JSON (from rank 0 only), and prints the console summary banner.

### Section 11: Banner Helper

`_print_banner()` prints a 70-character `*` bordered banner to the console, consistent in style with the companion scripts `hs_compress_v7.py` and `hs_npt_v2.py`. Output is routed through `root_print()`.

### Section 12: Entry Point and Exception Handling

`main()` is the top-level function. It orchestrates the 10-step workflow and contains the script's exception handling structure:

```python
try:
    sim, mc, log_handle = build_simulation(...)
    sim.run(...)
    write_final_outputs(...)
except Exception as exc:
    # [N-09] Emergency snapshot
    _write_snapshot(sim, "emergency_restart_<tag>.gsd")
    raise  # full traceback preserved
finally:
    # always executed:
    for writer in sim.operations.writers: writer.flush()
    log_handle.flush(); log_handle.close()
    print("Total runtime: X.XX seconds")
```

The outer `if __name__ == "__main__"` block catches `SystemExit` (expected deliberate exits from validation, missing files, etc.) and re-raises them silently, and catches unexpected `Exception` (programming errors) and prints a full traceback before calling `sys.exit(1)`.

---

## 16. Exception Handling Map

Every exception that can be raised in this script is either caught at the point of origin (with a clear message and `sys.exit`) or propagates to the `except Exception` block in `main()` which writes an emergency snapshot before re-raising.

| Location | Exception type | Handler | Effect |
|---|---|---|---|
| Import of `gsd` | `ImportError` | `sys.exit` | `[FATAL] gsd package not found` |
| Import of `hoomd` | `ImportError` | `sys.exit` | `[FATAL] HOOMD-blue not found` |
| `load_simulparams`: file missing | implicit | `sys.exit` | `[FATAL] Parameter file not found` |
| `load_simulparams`: bad JSON | `json.JSONDecodeError` | `sys.exit` | `[FATAL] JSON parse error` |
| `load_simulparams`: missing keys | — | `sys.exit` | `[FATAL] Missing required keys` |
| `load_simulparams`: wrong types | — | `sys.exit` | `[FATAL] Type errors` |
| `params.validate()` | `ValueError` | `sys.exit` | `[FATAL] Parameter validation failed` |
| `resolve_filenames`: stage guard | — | `sys.exit` | `[ERROR] final GSD already exists` |
| `resolve_filenames`: prev stage missing | — | `sys.exit` | `[FATAL] previous stage output not found` |
| `resolve_filenames`: input missing | — | `sys.exit` | `[FATAL] Input GSD not found` |
| `read_seed`: file missing or corrupt | `FileNotFoundError`, `KeyError` | `sys.exit` | `[FATAL] Cannot read seed` |
| `read_mono_diameter_from_gsd`: file missing | — | `sys.exit` | `[FATAL] Cannot read diameter` |
| `read_mono_diameter_from_gsd`: empty | — | `sys.exit` | `[FATAL] GSD file has no frames` |
| `read_mono_diameter_from_gsd`: bad GSD | `Exception` | `sys.exit` | `[FATAL] Cannot open` |
| `read_mono_diameter_from_gsd`: polydisperse | — | `sys.exit` | `[FATAL] Multiple particle diameters` |
| `load_and_broadcast_snapshot`: file read | `Exception` | `sys.exit` | `[FATAL] Cannot read` |
| `build_simulation`: GPU init | `Exception` | warn + fallback CPU | `[WARNING] GPU initialisation failed` |
| `build_simulation`: N=0 | — | `sys.exit` | `[FATAL] zero particles` |
| `build_simulation`: HDF5 writer | `Exception` | warn + continue | `[WARNING] HDF5 diagnostics writer disabled` |
| `Status.timestep_fraction` | `Exception` | return `"0/?"` | safe pre-run sentinel |
| `Status.seconds_remaining` | `ZeroDivisionError`, `DataAccessError`, `AttributeError` | return `0.0` | safe pre-run sentinel |
| `MCStatus.acceptance_rate` | `IndexError`, `ZeroDivisionError`, `DataAccessError` | return `0.0` | safe pre-run sentinel |
| `OverlapCount.overlap_count` | `DataAccessError` | return `0.0` | safe pre-run sentinel |
| `CurrentTimestep.timestep` | `DataAccessError`, `AttributeError` | return `0.0` | safe pre-run sentinel |
| `_write_snapshot`: image flags | `Exception` | `pass` | silently skip optional field |
| `sim.run()` | any `Exception` | emergency snapshot + `raise` | `[ERROR]` + traceback |
| `finally`: writer flush | `Exception` | `pass` | best-effort, non-masking |
| `finally`: log close | `Exception` | `pass` | best-effort, non-masking |
| `__main__`: `SystemExit` | `SystemExit` | `pass` | silent (message already printed) |
| `__main__`: unexpected | `Exception` | traceback + `sys.exit(1)` | `[FATAL]` + full traceback |

---

## 17. Logged Quantities Reference

### Table log (`.log` file)

Written by `hoomd.write.Table` to a plain-text file every `log_frequency` steps. The first line contains the header; every subsequent line is a timestep record.

| Column | Type | Source class | Meaning |
|---|---|---|---|
| `Simulation/tps` | scalar | HOOMD `Simulation` | Timesteps per second. Performance metric. |
| `Simulation/walltime` | scalar | HOOMD `Simulation` | Seconds elapsed since simulation start. |
| `Simulation/timestep` | string | `Status` | `"current_step/total_steps"`. E.g. `"350000/1000000"`. |
| `Status/etr` | string | `Status` | Estimated time remaining. Format: `H:MM:SS`. |
| `MCStatus/acceptance_rate` | scalar | `MCStatus` | Windowed translational acceptance ratio [0, 1]. Target: 0.2–0.3. |
| `Box/volume` | scalar | `Box_property` | Box volume [σ³]. Constant in NVT. |
| `Box/packing_fraction` | scalar | `Box_property` | φ = N·(π/6)·σ³/V. Constant in NVT. |
| `HPMC/overlap_count` | scalar | `OverlapCount` | Hard-sphere overlap count. Must be 0. |

### HDF5 diagnostics (`.h5` file)

Written by `hoomd.write.HDF5Log` every `diagnostics_period` steps. Datasets are stored under the `hoomd-data` HDF5 group. Read with h5py or any HDF5 library.

| HDF5 path | Type | Source | Meaning |
|---|---|---|---|
| `hoomd-data/Simulation/tps` | float64 array | HOOMD | Timesteps per second |
| `hoomd-data/Simulation/walltime` | float64 array | HOOMD | Wall time [s] |
| `hoomd-data/Simulation/timestep` | float64 array | `CurrentTimestep` | Current timestep as scalar |
| `hoomd-data/MCStatus/acceptance_rate` | float64 array | `MCStatus` | Windowed acceptance rate |
| `hoomd-data/Box/packing_fraction` | float64 array | `Box_property` | φ at each logged step |
| `hoomd-data/HPMC/overlap_count` | float64 array | `OverlapCount` | Overlap count |
| `hoomd-data/hpmc/integrate/Sphere/translate_moves` | int array [N, 2] | HOOMD HPMC | `[accepted, rejected]` totals |
| `hoomd-data/hpmc/integrate/Sphere/mps` | float64 array | HOOMD HPMC | Monte Carlo moves per second |

---

## 18. The HDF5 Diagnostics File

The HDF5 file offers several advantages over the text log:

- **Binary format**: smaller files; exact floating-point values (no round-trip through decimal formatting).
- **Random access**: read any single timestep without scanning the whole file.
- **Direct NumPy integration**: `h5py` returns NumPy arrays directly.
- **Graceful degradation**: if `h5py` is not installed or HOOMD was compiled without HDF5, a `[WARNING]` is printed and the run continues normally.

### Reading the HDF5 file

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("nvt_hpmc_diagnostics.h5", "r") as f:
    # Retrieve datasets (each is a 1-D array over logged timesteps)
    tps           = f["hoomd-data/Simulation/tps"][:]
    walltime      = f["hoomd-data/Simulation/walltime"][:]
    timestep      = f["hoomd-data/Simulation/timestep"][:]
    acceptance    = f["hoomd-data/MCStatus/acceptance_rate"][:]
    phi           = f["hoomd-data/Box/packing_fraction"][:]
    overlaps      = f["hoomd-data/HPMC/overlap_count"][:]
    trans_moves   = f["hoomd-data/hpmc/integrate/Sphere/translate_moves"][:]
    mps           = f["hoomd-data/hpmc/integrate/Sphere/mps"][:]

# Cumulative acceptance rate from translate_moves
accepted = trans_moves[:, 0]
rejected = trans_moves[:, 1]
cumulative_acceptance = accepted / (accepted + rejected)

# Plot acceptance rate over time
plt.figure(figsize=(10, 4))
plt.plot(timestep, acceptance, label="Windowed acceptance rate")
plt.plot(timestep, cumulative_acceptance, label="Cumulative acceptance rate")
plt.axhline(0.25, color='r', linestyle='--', label="Target 25%")
plt.xlabel("Timestep")
plt.ylabel("Acceptance rate")
plt.legend()
plt.tight_layout()
plt.savefig("acceptance_rate.png", dpi=150)
```

### Checking for overlaps

```python
with h5py.File("nvt_hpmc_diagnostics.h5", "r") as f:
    overlaps = f["hoomd-data/HPMC/overlap_count"][:]

if np.any(overlaps > 0):
    bad_steps = timestep[overlaps > 0]
    print(f"WARNING: Overlaps detected at timesteps: {bad_steps}")
else:
    print("OK: No overlaps detected throughout the run.")
```

---

## 19. Diameter Persistence: How and Why

Hard-sphere simulations depend critically on knowing σ. If the diameter is not written to a GSD trajectory, any analysis script that reads the trajectory cannot reconstruct φ, cannot verify particle sizes, and cannot compute contact distances correctly.

This script uses a **dual mechanism** to ensure diameters appear in all output files:

### Periodic writers (trajectory, restart)

```python
hoomd.write.GSD(
    ...,
    dynamic=["property", "attribute"],
)
writer.write_diameter = True
```

`dynamic=["property", "attribute"]` is the v4.9 documented API. The `"property"` category includes `particles/diameter` in the GSD schema. This ensures diameter is written to **every frame**, not just frame 0. `write_diameter=True` is kept as a belt-and-suspenders fallback for HOOMD v4.0–v4.5 builds where `dynamic=` was not yet the canonical mechanism for diameter persistence.

### One-shot writes (final, emergency)

`_write_snapshot()` manually constructs a `gsd.hoomd.Frame` from `sim.state.get_snapshot()` and explicitly sets `frame.particles.diameter` from the snapshot data. This approach is completely independent of HOOMD's GSD writer configuration and guarantees diameter is present regardless of HOOMD version or writer settings.

---

## 20. Seed Reproducibility

To reproduce an exact trajectory from a previous run:

1. Find the `random_seed` field in `<tag>_stage<id>_nvt_summary.json`.
2. Manually create or edit `random_seed.json`:
   ```json
   {
       "random_seed": 42731,
       "created_at": "2025-03-15T14:22:01"
   }
   ```
3. Run with the identical JSON parameters.

The script will read the existing seed file instead of generating a new one, producing bit-for-bit identical results.

**Note on MPI reproducibility**: HOOMD does not guarantee bit-for-bit identical trajectories between different MPI rank counts or different hardware, even with the same seed. The same seed with the same number of ranks on the same hardware does reproduce identically.

---

## 21. Choosing `move_size_translation` (d)

The translational move size `d` determines the maximum displacement in any direction per trial move. Each trial draws `Δr = (Δx, Δy, Δz)` uniformly from the cube `[-d, d]³`.

The acceptance ratio is determined by the probability that a randomly drawn displacement does not create an overlap. The target acceptance ratio for HPMC translation moves is typically **20–30%**.

### Practical guidance

| Packing fraction φ | Typical good d [σ] |
|---|---|
| 0.35–0.45 (dilute fluid) | 0.10–0.20 |
| 0.45–0.52 (dense fluid) | 0.06–0.12 |
| 0.52–0.58 (coexistence) | 0.04–0.08 |
| 0.58–0.64 (crystal) | 0.02–0.05 |

### How to check your choice

Look at the `MCStatus/acceptance_rate` column in the `.log` file. If it is:
- **> 50%**: `d` is too small. Particles barely move; diffusion is slow. Increase `d`.
- **20–30%**: ideal. Good exploration with reasonable acceptance.
- **< 5%**: `d` is too large. Almost every move creates an overlap; the system barely evolves. Decrease `d`.

Note: because `d` is **fixed** (no `MoveSize` adaptive tuner), you should choose it carefully before a long production run. For exploratory runs, start with `d = σ/10` and check the log after a few thousand steps.

---

## 22. Reading Output Files in Python

### Reading the trajectory GSD

```python
import gsd.hoomd
import numpy as np

with gsd.hoomd.open("nvt_hpmc_output_traj.gsd", "r") as traj:
    print(f"Number of frames: {len(traj)}")

    # Read a specific frame (0-indexed; -1 = last)
    frame = traj[-1]
    N         = frame.particles.N
    positions = frame.particles.position   # shape (N, 3)
    diameters = frame.particles.diameter   # shape (N,)
    box       = frame.configuration.box   # [Lx, Ly, Lz, xy, xz, yz]
    step      = frame.configuration.step

    print(f"Frame at step {step}: N={N}, σ={diameters[0]:.4f}")
    print(f"Box: Lx={box[0]:.4f}, Ly={box[1]:.4f}, Lz={box[2]:.4f}")
```

### Computing packing fraction from the trajectory

```python
import gsd.hoomd
import numpy as np
import math

PI_OVER_6 = math.pi / 6.0

with gsd.hoomd.open("nvt_hpmc_output_traj.gsd", "r") as traj:
    phis = []
    steps = []
    for frame in traj:
        N = frame.particles.N
        sigma = frame.particles.diameter[0]
        box = frame.configuration.box
        V = box[0] * box[1] * box[2]  # for orthorhombic box
        phi = N * PI_OVER_6 * sigma**3 / V
        phis.append(phi)
        steps.append(frame.configuration.step)

print(f"phi = {np.mean(phis):.6f} ± {np.std(phis):.2e}")
```

### Reading the summary JSON

```python
import json

with open("hs_4096_nvt_stage-1_nvt_summary.json") as f:
    summary = json.load(f)

print(f"N = {summary['n_particles']}")
print(f"sigma = {summary['diameter']}")
print(f"phi = {summary['packing_fraction']:.6f}")
print(f"overlaps at end = {summary['overlaps_final']}")
print(f"runtime = {summary['runtime_seconds']:.1f} s")
```

### Parsing the text log

```python
import numpy as np
import pandas as pd

# pandas is not a dependency of the simulation script; install separately
df = pd.read_csv("nvt_hpmc_log.log", sep=r'\s+')
print(df.columns.tolist())
print(df[["Simulation/tps", "MCStatus/acceptance_rate",
          "Box/packing_fraction", "HPMC/overlap_count"]].tail())
```

Or with NumPy (no pandas):
```python
import numpy as np

# Read header and data separately
with open("nvt_hpmc_log.log") as f:
    header = f.readline().split()
    data = np.loadtxt(f)

col = {name: i for i, name in enumerate(header)}
acceptance = data[:, col["MCStatus/acceptance_rate"]]
phi        = data[:, col["Box/packing_fraction"]]
```

---

## 23. HPC Cluster / SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=hs_nvt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --partition=regular

# Load your environment
module load hoomd/4.9.0
module load mpi4py

# Set HOOMD_WALLTIME_STOP so HOOMD exits cleanly before the scheduler kills it.
# This allows the restart GSD to be flushed before termination.
# Set to 10 minutes before the scheduled end.
export HOOMD_WALLTIME_STOP=$(( $(date +%s) + 24*3600 - 600 ))

cd $SLURM_SUBMIT_DIR

# The JSON stage_id_current should be set before submission.
# On first submission: "stage_id_current": 0
# On resubmission after walltime: keep "stage_id_current": 0 (restart is automatic)
# After completion: increment to 1 for the next stage.

mpirun -n 8 python hs_nvt_v4.py --simulparam_file simulparam_hs_nvt.json > output.log 2>&1
```

### Notes on walltime management

HOOMD-blue v4 honours the `HOOMD_WALLTIME_STOP` environment variable (a Unix epoch time). When `sim.run()` determines it cannot complete the remaining steps before `HOOMD_WALLTIME_STOP`, it exits `sim.run()` cleanly. The `finally` block then flushes all writers and closes the log file before the process terminates. The restart checkpoint GSD written at the last `restart_gsd_frequency` step is the recovery point.

The recommended margin is 10 minutes: `HOOMD_WALLTIME_STOP = job_start + job_duration - 600`. This gives HOOMD enough time to finish the current batch of steps, flush all buffers, and exit cleanly.

---

## 24. Troubleshooting

### `[FATAL] Parameter file not found`

The JSON file path passed to `--simulparam_file` does not exist. Check the path and filename. If running from a different directory, use an absolute path.

### `[FATAL] JSON parse error`

Your JSON file has a syntax error. Common causes: trailing commas after the last key-value pair, missing quotes around string values, `//`-style comments (JSON does not support them — use `_`-prefixed keys instead). Use an online JSON validator to locate the error.

### `[FATAL] Missing required keys` / `[FATAL] Type errors`

A required key is absent from the JSON, or a key has the wrong value type (e.g. `"total_num_timesteps": "1000000"` instead of `"total_num_timesteps": 1000000`). The error message names the specific keys.

### `[FATAL] Input GSD file not found`

The file named in `input_gsd_filename` does not exist in the working directory. Check the filename and path. For multi-stage runs with `stage_id >= 1`, check that the previous stage completed and produced a `_final.gsd` file.

### `[FATAL] GSD file has no frames`

The input GSD exists but is empty. This happens when a previous run crashed before writing any frames. Re-run from an earlier checkpoint or from the original compression output.

### `[FATAL] Multiple particle diameters`

The input GSD contains particles with different diameters. This script requires a monodisperse system. If you intentionally have a polydisperse system, you need to modify `read_mono_diameter_from_gsd()` and `mc.shape` configuration.

### `[WARNING] HDF5 diagnostics writer disabled`

`h5py` is not installed, or HOOMD was compiled without HDF5 support. The run continues normally. To enable HDF5 output: `pip install h5py` or `conda install h5py`.

### `[WARNING] GPU initialisation failed`

The specified GPU is not available (wrong `gpu_id`, CUDA not present, or compute capability too low). The script falls back to CPU automatically.

### `[ERROR] Stage N final GSD already exists`

You are re-running a stage that already completed. Either increment `stage_id_current` in the JSON to start the next stage, or delete the existing `_final.gsd` if you want to re-run this stage.

### Acceptance rate is 0.0 at the start of the log

This is normal. The `MCStatus.acceptance_rate` returns `0.0` before the first `sim.run()` call (the `DataAccessError` guard). The real acceptance rates appear from the first log entry after `sim.run()` begins.

### Very low acceptance rate (< 5%)

`move_size_translation` is too large. The simulation is still valid but slow. Reduce `d` in the JSON and re-run.

### Very high acceptance rate (> 50%)

`move_size_translation` is too small. The simulation explores configuration space slowly. Increase `d`.

### Non-zero `HPMC/overlap_count`

This should **never happen** in a valid NVT run with a correct input GSD. If you see non-zero overlaps:
1. Check that the input GSD was produced by a successful compression that ended with zero overlaps (look at the compression script's summary JSON).
2. Check that `mc.shape["A"]["diameter"]` matches the particle diameter in the GSD.
3. Report the issue with the emergency snapshot and full traceback.

---

## 25. Version History and Changes from v1

| Version | Key changes |
|---|---|
| `hard_sphere_nvt_v1p19p2.py` | Original script: hard-coded filenames and parameters, no restart detection, no stage management, `mode='wb'` trajectory (overwrites on restart) |
| `hs_nvt_v2.py` | JSON-driven parameters (SimulationParams dataclass), stage-aware filenames, seed file management, restart detection, MPI root_print helpers, DataAccessError guards, packing fraction and overlap count logging, `mode='ab'` trajectory on restart |
| `hs_nvt_v3.py` | HDF5 diagnostics writer (optional), `CurrentTimestep` loggable, `create_state_from_gsd()` for fresh runs instead of MPI broadcast pattern [N-20], `diagnostics_frequency` parameter, explicit `write_diameter=True` on all GSD writers [N-16], manual `_write_snapshot()` with explicit diameters for final/emergency writes [N-17], `traj_mode` logic (`"ab"` restart / `"wb"` fresh) [N-19], top-level `__main__` exception guard with `traceback.print_exc()` |
| `hs_nvt_v4.py` | **Current version**: adds 196 lines of inline documentation and exception handling annotations throughout all sections. **No algorithm changes from v3.** All exception handlers are explained (which exceptions, why they occur, what the sentinel value means). All key design decisions are documented at the point of implementation. |

The `[N-XX]` tags throughout the source code correspond to this table. Tags [N-01] through [N-20] mark features introduced in v2 and v3 that distinguish this script from the original `hard_sphere_nvt_v1p19p2.py`.

---

*Documentation for `hs_nvt_v4.py` | HOOMD-blue v4 Hard-Sphere NVT Equilibration*
