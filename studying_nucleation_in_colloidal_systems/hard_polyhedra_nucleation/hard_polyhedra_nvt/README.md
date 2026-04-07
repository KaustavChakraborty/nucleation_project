# HOOMD-blue v4 Convex-Polyhedron HPMC NVT Equilibration

mpirun -n 8 python3.10 HOOMD_hard_polyhedra_NVT.py  --simulparam_file simulparam_hard_polyhedra_nvt.json

A restart-aware HOOMD-blue v4 workflow for equilibrating one-component systems of hard convex polyhedra in the NVT ensemble using the HPMC `ConvexPolyhedron` integrator.

This project is meant for the stage that comes after you already have a valid, non-overlapping convex-polyhedron configuration, typically the output of a lattice-building, initialization, or compression workflow. The script then performs long NVT Monte Carlo equilibration at fixed box and fixed particle number while writing trajectory, restart, text-log, optional HDF5 diagnostics, and a final summary JSON.

The code is intentionally strict and defensive. It validates:

- simulation-parameter JSON fields and types,
- the convex-polyhedron shape JSON,
- the incoming GSD content,
- single-component particle typing,
- orientation validity,
- shape-metadata consistency,
- zero-overlap starting geometry.

It also writes self-describing GSD files containing `particles/type_shapes`, which is essential for later visualization and downstream analysis.

---

## 1. What this project does

At a high level, this project does the following:

1. Reads a simulation parameter JSON passed through `--simulparam_file`.
2. Resolves all stage-aware input and output filenames.
3. Creates or reuses a persistent random seed file.
4. Loads a convex-polyhedron definition from a separate shape JSON.
5. Loads a fresh state from an input GSD or resumes from a restart GSD.
6. Reconstructs the HOOMD snapshot in an MPI-safe way.
7. Configures `hoomd.hpmc.integrate.ConvexPolyhedron(nselect=1)`.
8. Attaches text logging, trajectory writing, restart writing, and optional HDF5 diagnostics.
9. Runs `sim.run(total_num_timesteps)`.
10. Writes a final self-describing GSD.
11. Verifies the final GSD shape metadata.
12. Writes a machine-readable run-summary JSON.

This is not a compression script, not a box-rescaling script, and not an overlap-removal workflow. The simulation box remains fixed during the NVT run.

---

## 2. Scientific / computational setting

The simulation uses HOOMD-blue’s hard-particle Monte Carlo framework (`hoomd.hpmc`). In this framework:

- there are no continuous interparticle forces,
- there is no potential-energy minimization,
- the geometric constraint of non-overlap is central,
- translational and rotational trial moves are accepted or rejected according to hard-particle Monte Carlo rules.

For convex polyhedra, particle orientation is part of the physical state. That is why this workflow expects a valid quaternion orientation array in the GSD, or inserts identity quaternions on a fresh load when that array is missing or malformed.

The particle geometry itself is not hard-coded in the driver. Instead, it is loaded from a separate shape JSON and supplied to `hoomd.hpmc.integrate.ConvexPolyhedron` as a list of vertices.

The packing fraction is computed as:

`phi = N * V_particle / V_box`

where:

- `N` is the number of particles,
- `V_particle` is the actual scaled polyhedron volume,
- `V_box` is the simulation-box volume.

If the shape JSON stores a reference volume `V0` and the simulation uses `shape_scale = s`, then:

`V_particle = V0 * s^3`

For the supplied cube shape JSON, `4_volume = 1.0`, so at `shape_scale = 1.0` the particle volume is `1.0`.

---

## 3. Design philosophy

This workflow is built around a few strong assumptions.

### 3.1 Start from a valid configuration

The script performs `sim.run(0)` immediately after integrator setup and then queries `mc.overlaps`. If the starting configuration contains any overlaps, the run aborts. The NVT script does not try to repair invalid geometry.

### 3.2 Preserve shape metadata in every important output

Trajectory, restart, and final GSD files should remain self-describing. The script therefore writes `particles/type_shapes` so that visualization tools and later analysis know which convex polyhedron is stored.

### 3.3 Restart cleanly after interruption

If a restart GSD exists but the final GSD does not, the script resumes from the checkpoint. Otherwise it starts fresh from the input GSD.

### 3.4 Fail early when inputs are inconsistent

Rather than proceeding with ambiguous input, the script exits on problems such as:

- missing required JSON keys,
- wrong JSON value types,
- invalid shape JSON,
- invalid particle arrays in the input GSD,
- multiple particle types,
- invalid quaternions,
- mismatched shape metadata,
- nonzero starting overlaps.

---

## 4. Files in this project

### 4.1 Main driver

#### `HOOMD_hard_polyhedra_NVT.py`

This is the simulation engine. It contains:

- MPI-aware rank-0 printing helpers,
- custom loggable classes for ETR, acceptance rates, box properties, overlap count, and timestep,
- a typed `SimulationParams` dataclass,
- JSON loading and validation,
- random-seed file creation and reuse,
- stage-aware filename resolution,
- shape loading and scaling,
- snapshot broadcast and reconstruction,
- simulation building,
- writer attachment,
- final-output generation,
- emergency snapshot logic,
- exception handling and runtime reporting.

### 4.2 Runtime parameter file

#### `simulparam_hard_polyhedra_nvt.json`

This file defines the concrete run. In the supplied example it contains:

- `tag = HOOMD_hard_cube_2197_4096_at_pf0p58_nvt`
- `input_gsd_filename = HOOMD_hard_cube_2197_compression_to_pf0p58_final.gsd`
- `stage_id_current = -1`
- `shape_json_filename = shape_023_Cube_unit_volume_principal_frame.json`
- `shape_scale = 1.0`
- `total_num_timesteps = 50000000`
- `move_size_translation = 0.045`
- `move_size_rotation = 0.045`
- `log_frequency = 10000`
- `traj_gsd_frequency = 100000`
- `restart_gsd_frequency = 10000`
- `use_gpu = false`
- `gpu_id = 0`
- explicit single-stage output filenames.

### 4.3 Shape definition

#### `shape_023_Cube_unit_volume_principal_frame.json`

This file defines the reference convex polyhedron. In the supplied example:

- `1_Name = Cube`
- `2_ShortName = P03`
- `4_volume = 1.0`
- the center of mass is at the origin,
- the inertia tensor components are provided,
- `8_vertices` describe a cube centered at the origin with corners at `±0.5`.

### 4.4 Optional batch launcher

#### `submit_hoomd-v4.sh`

A PBS-style submission script is included. It sets environment variables for a custom GCC / Python / OpenMPI / HOOMD installation and launches the job with `mpirun`.

### 4.5 Runtime-generated files

Depending on mode, the run may produce:

- trajectory GSD,
- restart GSD,
- final GSD,
- human-readable text log,
- optional HDF5 diagnostics,
- random-seed JSON,
- run-summary JSON,
- emergency restart snapshot if an unexpected exception occurs.

---

## 5. Dependencies

The script expects the following software stack.

### 5.1 Required Python packages

- HOOMD-blue >= 4.0
- GSD >= 3.0
- NumPy

### 5.2 Optional Python packages

- `mpi4py` — optional; if unavailable, the script falls back to a serial MPI stub
- `h5py` — effectively optional; needed if the HDF5 diagnostics writer is to work in your environment

### 5.3 Imported HOOMD modules

The script imports:

- `hoomd`
- `hoomd.hpmc`
- `hoomd.logging`
- `hoomd.write`
- `hoomd.trigger`
- `hoomd.filter`
- `hoomd.error.DataAccessError`

If `gsd` or `hoomd` is missing, the script exits immediately with a fatal message.

---

## 6. Supported simulation model and assumptions

The current implementation is intentionally narrow.

### 6.1 Supported

- one-component systems only,
- one particle type only,
- `typeid = 0` for all particles,
- convex polyhedra represented through vertex lists,
- fixed-box NVT Monte Carlo,
- translational and rotational moves,
- fresh runs and restart runs,
- serial or MPI execution,
- CPU execution and best-effort GPU selection.

### 6.2 Not supported by design in the current driver

- multicomponent systems,
- multiple particle types with different shapes,
- adaptive move-size tuning,
- variable-box ensembles (`NPT`, `NPH`, box moves, etc.),
- overlap removal inside this script,
- force-field based dynamics,
- arbitrary type-mixing logic.

---

## 7. High-level algorithm in detail

### Step 1 — Read and validate the simulation parameter JSON

The script parses `--simulparam_file`, loads the JSON, strips keys beginning with `_`, checks that required keys are present, verifies the Python types of those values, constructs a `SimulationParams` dataclass, and runs additional logical validation.

### Step 2 — Resolve filenames

The script determines:

- the input GSD,
- trajectory GSD,
- restart GSD,
- final GSD,
- text log path,
- HDF5 diagnostics path.

Single-stage and multi-stage modes are handled differently. Details are given later.

### Step 3 — Create or reuse the seed file

The random seed is stored in a small JSON file so that restarts and later stages reuse the same seed rather than silently switching random streams.

### Step 4 — Load the shape JSON

The script reads the reference volume and the reference vertex list from the shape JSON, checks that the volume is positive and the vertex array has a valid `(Nv, 3)` form, then scales the vertices by `shape_scale`.

### Step 5 — Build the simulation state

There are two cases.

#### Fresh run

- rank 0 reads the last frame of the input GSD,
- rank 0 broadcasts a minimal data dictionary,
- each rank reconstructs a HOOMD snapshot,
- `sim.create_state_from_snapshot()` is used.

#### Restart run

- `sim.create_state_from_gsd(filename=restart_gsd)` is used directly.

### Step 6 — Validate the loaded state

The script checks:

- particle count > 0,
- position array shape `(N, 3)`,
- typeid array shape `(N,)`,
- unique typeid is exactly `[0]`,
- particle type list exists,
- orientations are finite and nonzero-norm when provided.

If the fresh input GSD contains `particles/type_shapes`, the script also checks that those stored vertices are consistent with the expected convex-polyhedron vertices derived from the shape JSON.

### Step 7 — Configure the HPMC integrator

The driver sets up:

```python
mc = hoomd.hpmc.integrate.ConvexPolyhedron(nselect=1)
mc.shape["A"] = {"vertices": params.shape_vertices}
mc.d["A"] = params.move_size_translation
mc.a["A"] = params.move_size_rotation
sim.operations.integrator = mc
```

Here:

- `nselect = 1` means one trial move per particle per sweep,
- `mc.d["A"]` is the translational move size,
- `mc.a["A"]` is the rotational move size.

### Step 8 — Initialize overlap bookkeeping

The script calls `sim.run(0)` so HOOMD initializes the state and overlap computation. It then reads `mc.overlaps`. If the count is nonzero, the script aborts.

### Step 9 — Attach writers

The workflow attaches:

- a human-readable `Table` writer,
- a trajectory GSD writer,
- a restart GSD writer,
- an optional HDF5 diagnostics writer.

### Step 10 — Run the simulation

The main production run is simply:

```python
sim.run(params.total_num_timesteps)
```

### Step 11 — Write final outputs

At the end, the script writes:

- a final single-frame GSD with explicit shape metadata,
- a summary JSON,
- a final console summary banner.

---

## 8. The simulation parameter JSON in detail

The main runtime JSON contains both required and optional fields.

### 8.1 Required fields

#### `tag`
A string label for the run. Used in stage-aware naming and in the summary JSON.

#### `input_gsd_filename`
Path to the input GSD used for a fresh run or stage-0 run.

#### `stage_id_current`
Controls single-stage vs multi-stage behavior.

- `-1` means single-stage mode.
- `0, 1, 2, ...` means multi-stage mode.

#### `shape_json_filename`
Path to the convex-polyhedron definition JSON.

#### `shape_scale`
Linear scale factor applied to every reference vertex from the shape JSON.

#### `total_num_timesteps`
Total number of Monte Carlo sweeps to run in this stage.

#### `move_size_translation`
Fixed translational HPMC move size for type `"A"`.

#### `move_size_rotation`
Fixed rotational HPMC move size for type `"A"`.

#### `log_frequency`
How often to write one line to the human-readable text log.

#### `traj_gsd_frequency`
How often to append a frame to the trajectory GSD.

#### `restart_gsd_frequency`
How often to overwrite the single-frame restart checkpoint GSD.

#### `use_gpu`
Boolean switch for GPU usage.

#### `gpu_id`
Requested GPU id when `use_gpu = true`.

### 8.2 Optional fields

#### `diagnostics_frequency`
Frequency for HDF5 diagnostics. If `0` or absent, the script falls back to `log_frequency`.

#### `initial_timestep`
Used only for a fresh run after snapshot reconstruction. Lets you start the run at a nonzero timestep.

#### `output_trajectory`
Single-stage trajectory GSD filename.

#### `log_filename`
Single-stage text log filename.

#### `restart_file`
Single-stage restart GSD filename.

#### `final_gsd_filename`
Single-stage final GSD filename.

#### `hdf5_log_filename`
Single-stage HDF5 diagnostics filename.

### 8.3 Validation rules enforced by the script

The script requires:

- `shape_scale > 0`
- `total_num_timesteps > 0`
- `move_size_translation > 0`
- `move_size_rotation >= 0`
- `log_frequency > 0`
- `traj_gsd_frequency > 0`
- `restart_gsd_frequency > 0`
- `stage_id_current >= -1`
- `diagnostics_frequency >= 0`

---

## 9. Shape JSON format and meaning

The separate shape JSON is the authoritative source of geometry.

### 9.1 Required keys used by the script

#### `4_volume`
Reference particle volume.

#### `8_vertices`
Reference vertex array with shape `(Nv, 3)`.

### 9.2 Optional but useful metadata

The supplied cube file also contains:

- `0_Id`
- `1_Name`
- `2_ShortName`
- `3_Comment`
- `5_center_of_mass`
- `6_moment_of_inertia`
- `7_comment_about_moment_of_inertia`

The script uses the name and short name if present, otherwise it falls back to the filename stem.

### 9.3 Meaning of `shape_scale`

If a reference vertex is `r0`, then the actual simulation vertex is `s * r0`, where `s = shape_scale`.

As a consequence:

- lengths scale as `s`,
- areas scale as `s^2`,
- volume scales as `s^3`.

This is why the script computes:

```python
particle_volume = reference_volume * shape_scale**3
```

---

## 10. Input GSD requirements

For a fresh run, the input GSD must satisfy several conditions.

### 10.1 Required content

The last frame of the input GSD must contain:

- at least one particle,
- a valid simulation box,
- a position array of shape `(N, 3)`,
- a typeid array of shape `(N,)`,
- a particle type list,
- effectively one component only,
- a geometrically valid non-overlapping configuration for the requested shape.

### 10.2 Type restriction

The current NVT driver assumes all particles are of a single type and that the unique typeid list is exactly `[0]`.

### 10.3 Orientation handling

If the orientation array is missing or malformed in a fresh input GSD, the script inserts identity quaternions `[1, 0, 0, 0]` for all particles and prints a warning.

If an orientation array is present, the script requires:

- all entries finite,
- all quaternion norms strictly positive.

### 10.4 Shape metadata checking

If `particles/type_shapes` exists in the input GSD, the script checks that the stored convex-polyhedron vertices agree with the expected vertices derived from the shape JSON and `shape_scale`.

This check is there to prevent accidentally using a GSD generated for one shape with a different shape JSON during NVT equilibration.

---

## 11. Single-stage vs multi-stage mode

The code supports two naming modes.

### 11.1 Single-stage mode (`stage_id_current = -1`)

In this mode, filenames are taken directly from the JSON.

For example:

- `output_trajectory`
- `log_filename`
- `restart_file`
- `final_gsd_filename`
- `hdf5_log_filename`

If the final GSD already exists, the script warns that it will be overwritten at the end of the run.

### 11.2 Multi-stage mode (`stage_id_current >= 0`)

In this mode, filenames are generated automatically from `tag` and `stage_id_current`.

For stage `sid`, the script uses:

- trajectory: `<tag>_<sid>_traj.gsd`
- restart: `<tag>_<sid>_restart.gsd`
- final: `<tag>_<sid>_final.gsd`
- log: `<tag>_<sid>.log`
- diagnostics: `<tag>_<sid>_diagnostics.h5`

Input handling in multi-stage mode:

- stage `0` reads `input_gsd_filename`
- stage `N > 0` reads `<tag>_{N-1}_final.gsd`

The script also blocks re-running a completed stage if the current stage’s final GSD already exists.

---

## 12. Random seed management

The script stores the random seed in a JSON file rather than keeping it ephemeral.

### 12.1 Seed-file names

- single-stage mode: `random_seed.json`
- multi-stage mode: `random_seed_stage_0.json`

### 12.2 Why this matters

This lets:

- restarted jobs reuse the original seed,
- repeated stage handling remain consistent,
- post-run provenance be preserved.

### 12.3 Seed range

The seed is drawn from `[0, 65535]` because HOOMD truncates larger seeds.

---

## 13. MPI behavior and parallel execution

The workflow is MPI-aware.

### 13.1 Rank-0 printing

Most status output uses helper functions that print only on rank 0 to avoid duplicated log spam.

### 13.2 Snapshot broadcast pattern

For a fresh run:

- rank 0 reads the last input GSD frame,
- rank 0 broadcasts a minimal data dictionary,
- each rank reconstructs a compatible HOOMD snapshot,
- HOOMD distributes the state after `create_state_from_snapshot()`.

### 13.3 Serial fallback

If `mpi4py` is unavailable, the script uses a simple stub so serial execution still works.

### 13.4 Example MPI launch

```bash
mpirun -n 8 python HOOMD_hard_polyhedra_NVT.py --simulparam_file simulparam_hard_polyhedra_nvt.json
```

---

## 14. Logging and diagnostics

The script creates several forms of output.

### 14.1 Human-readable text log

The `Table` writer records quantities such as:

- TPS,
- walltime,
- timestep fraction,
- estimated time remaining,
- translational acceptance rate,
- rotational acceptance rate,
- box volume,
- packing fraction,
- overlap count.

### 14.2 Trajectory GSD

The trajectory writer appends frames periodically. It writes `dynamic=["property", "attribute"]` and includes shape metadata through a logger attached to `mc.type_shapes`.

### 14.3 Restart GSD

The restart writer uses `truncate=True`, so only a single current checkpoint frame is kept.

### 14.4 Optional HDF5 diagnostics

If available in the environment, the script also writes HDF5 diagnostics containing quantities such as:

- timestep,
- TPS,
- walltime,
- translational acceptance rate,
- rotational acceptance rate,
- packing fraction,
- overlap count,
- `translate_moves`,
- `rotate_moves`,
- HPMC `mps`.

If the HDF5 writer cannot be created, the script warns and continues without HDF5 output.

---

## 15. Restart behavior in detail

The restart logic is simple and deliberate.

### 15.1 Resume case

If:

- `restart_gsd` exists, and
- `final_gsd` does not exist,

then the script resumes from `restart_gsd`.

### 15.2 Fresh-run case

Otherwise the script starts from `input_gsd` using the broadcast/reconstruction path.

### 15.3 Consequence for trajectories and diagnostics

For true restarts:

- trajectory GSD uses append mode,
- HDF5 diagnostics use append mode if enabled.

For fresh runs:

- trajectory GSD uses write mode,
- HDF5 diagnostics use write mode if enabled.

---

## 16. Final outputs and summary JSON

At the end of a successful run, the script writes:

### 16.1 Final GSD

A single-frame final snapshot that explicitly stores:

- positions,
- orientations,
- type ids / types,
- `particles/type_shapes`,
- optional particle fields when present.

### 16.2 Shape verification

After writing the final GSD, the script reopens it and verifies that:

- `particles/type_shapes` exists,
- the shape type is `ConvexPolyhedron`,
- the stored vertex array has the expected shape,
- the stored vertices match the expected vertices.

### 16.3 Summary JSON

The run-summary JSON contains information such as:

- tag,
- stage id,
- simulation parameter file path,
- input / trajectory / diagnostics / final filenames,
- shape name,
- shape short name,
- shape scale,
- particle volume,
- particle count,
- packing fraction,
- move sizes,
- diagnostics frequency,
- total timesteps,
- final timestep,
- final overlap count,
- random seed,
- final box parameters,
- runtime,
- creation time.

---

## 17. Example commands

### 17.1 Serial CPU run

```bash
python HOOMD_hard_polyhedra_NVT.py --simulparam_file simulparam_hard_polyhedra_nvt.json
```

### 17.2 MPI run

```bash
mpirun -n 8 python HOOMD_hard_polyhedra_NVT.py --simulparam_file simulparam_hard_polyhedra_nvt.json
```

### 17.3 Example using the supplied cluster script

```bash
qsub submit_hoomd-v4.sh
```

---

## 18. Meaning of the supplied example configuration

The uploaded example corresponds to a run with:

- a cube polyhedron of unit reference volume,
- shape scale `1.0`,
- a fresh input GSD named `HOOMD_hard_cube_2197_compression_to_pf0p58_final.gsd`,
- `N = 2197` particles in the observed sample output,
- target NVT production length of `50,000,000` timesteps,
- translational and rotational move sizes both set to `0.045`,
- periodic trajectory writing every `100000` steps,
- restart writing every `10000` steps,
- text logging every `10000` steps.

In the sample run output, the loaded box is cubic with approximately:

- `Lx = Ly = Lz = 15.5884`
- `phi = 0.58`
- initial overlap count `0` for the valid cube example.

---

## 19. How to choose move sizes

The driver uses fixed move sizes. There is no move-size tuner in the current script.

### 19.1 Translational move size

`move_size_translation` controls the magnitude of trial translations.

- too small -> poor configurational exploration,
- too large -> acceptance ratio becomes too low.

### 19.2 Rotational move size

`move_size_rotation` controls the magnitude of trial rotations.

- `0.0` freezes orientations,
- a large value may drastically reduce rotational acceptance,
- a moderate value is usually required for anisotropic polyhedra.

Because the script reports windowed acceptance rates, you can inspect the text log or HDF5 diagnostics to see whether your chosen values are reasonable.

---

## 20. Common failure modes and what they mean

### 20.1 Missing package imports

If `gsd` or `hoomd` cannot be imported, the script exits immediately.

### 20.2 Missing input files

If the simulation parameter file, input GSD, or shape JSON does not exist, the script exits with a fatal message.

### 20.3 Bad input GSD content

Examples include:

- zero particles,
- invalid position array shape,
- invalid typeid array,
- multiple particle types,
- non-finite orientations,
- zero-norm quaternions.

### 20.4 Wrong shape JSON for a GSD

If the GSD stores `type_shapes` metadata that does not match the vertices reconstructed from `shape_json_filename` and `shape_scale`, the script aborts rather than running with ambiguous geometry.

### 20.5 Overlapping starting configuration

If `mc.overlaps > 0` after initialization, the script aborts. You must fix the input configuration upstream.

### 20.6 HDF5 diagnostics unavailable

If `h5py` is not available or HOOMD was built without the relevant HDF5 support, the script prints a warning and continues without HDF5 diagnostics.

---

## 21. Troubleshooting checklist

When a run fails, check the following in order.

1. Does the JSON contain all required keys with the correct types?
2. Does `input_gsd_filename` actually exist in the working directory?
3. Does `shape_json_filename` point to the correct shape file?
4. Does the input GSD correspond to the same shape and scale as the JSON?
5. Is the system single-component with `typeid = 0` only?
6. Are the quaternions valid?
7. Is the input configuration truly overlap-free for the requested polyhedron?
8. Are your output directories writable?
9. If using MPI, are all ranks seeing the same files?
10. If using HDF5 diagnostics, is `h5py` installed in the runtime environment?

---

## 22. Known implementation notes / caveats

A few practical points are worth knowing.

### 22.1 The script preserves the original fixed-move-size philosophy

There is no automatic move-size tuning. This is deliberate.

### 22.2 The outermost script handler currently swallows `SystemExit`

At the bottom of the current driver, the `__main__` block catches `SystemExit` and simply passes. This can suppress helpful fatal messages in some cases and make early exits look like silent termination. If you are actively debugging, you may want to temporarily replace that behavior with a direct `main()` call or an `except SystemExit as e:` block that prints the message.

### 22.3 Shape-vertex comparisons are sensitive to implementation details

If you are validating `type_shapes` stored in a GSD against vertices rebuilt from a shape JSON, keep in mind that issues such as vertex ordering and floating-point precision can matter. In practice, order-insensitive comparison and a realistic tolerance are often more robust than a strict raw-array comparison.

### 22.4 The current driver is one-component only

Extending it to multicomponent polyhedron mixtures would require nontrivial changes in state validation, shape assignment, type handling, and output verification.

---

## 23. Extending the project

Typical extension directions include:

- adding adaptive move-size tuning,
- supporting multiple polyhedron types,
- supporting multiple particle species,
- adding richer restart metadata,
- integrating analysis hooks,
- adding post-run acceptance summaries,
- exposing tolerances for shape verification through JSON,
- writing more detailed performance diagnostics.

---

## 24. Minimal example JSON

A compact example mirroring the supplied setup is:

```json
{
  "tag": "HOOMD_hard_cube_2197_4096_at_pf0p58_nvt",
  "input_gsd_filename": "HOOMD_hard_cube_2197_compression_to_pf0p58_final.gsd",
  "stage_id_current": -1,
  "initial_timestep": 0,
  "shape_json_filename": "shape_023_Cube_unit_volume_principal_frame.json",
  "shape_scale": 1.0,
  "total_num_timesteps": 50000000,
  "move_size_translation": 0.045,
  "move_size_rotation": 0.045,
  "log_frequency": 10000,
  "traj_gsd_frequency": 100000,
  "restart_gsd_frequency": 10000,
  "use_gpu": false,
  "gpu_id": 0,
  "output_trajectory": "HOOMD_hard_cube_2197_nvt_hpmc_pf0p58_output_traj.gsd",
  "log_filename": "HOOMD_hard_cube_2197_nvt_hpmc_pf0p58_log.log",
  "restart_file": "HOOMD_hard_cube_2197_nvt_hpmc_pf0p58_output_restart.gsd",
  "final_gsd_filename": "HOOMD_hard_cube_2197_nvt_hpmc_pf0p58_final.gsd"
}
```

---

## 25. Minimal example shape JSON interpretation

For the supplied cube file, the vertex list is:

```json
[
  [-0.5, -0.5, -0.5],
  [-0.5, -0.5,  0.5],
  [-0.5,  0.5, -0.5],
  [-0.5,  0.5,  0.5],
  [ 0.5, -0.5, -0.5],
  [ 0.5, -0.5,  0.5],
  [ 0.5,  0.5, -0.5],
  [ 0.5,  0.5,  0.5]
]
```

That is a cube of side length `1`, centered at the origin, with volume `1`.

---

## 26. Final takeaway

This project is a strict, reproducible, restart-aware NVT equilibration driver for hard convex polyhedra in HOOMD-blue v4. It is best viewed as the production NVT stage that follows successful structure generation or compression. Its strengths are:

- careful validation,
- explicit geometry handling,
- robust restart logic,
- shape-aware GSD writing,
- reproducibility through seed and summary files,
- compatibility with serial and MPI execution.

If your upstream configuration is valid and your shape JSON matches your GSD geometry, this script gives you a clean, well-instrumented production workflow for long hard-polyhedron NVT runs.
