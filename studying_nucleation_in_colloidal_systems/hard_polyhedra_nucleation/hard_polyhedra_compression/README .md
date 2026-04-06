# HOOMD-blue v4 Convex-Polyhedron HPMC Compression

mpirun -n 8 python3.10 hard_polyhedron_compression_v1.py  --simulparam_file simulparam_hard_polyhedron_compression.json

An extensively documented project for compressing a **monodisperse hard convex-polyhedron system** from a low-density starting configuration to a target packing fraction using **HOOMD-blue v4** and the **HPMC (Hard Particle Monte Carlo)** engine.

This project is centered around a single, heavily documented simulation driver:

- `HOOMD_hard_polyhedra_compression.py`

and is configured by:

- `simulparam_hard_polyhedron_compression.json`
- `shape_023_Cube_unit_volume_principal_frame.json`

The current example configuration is for a **unit-volume cube shape** scaled by `shape_scale = 1.0`, compressed to a **target packing fraction** of `0.58` using a **manual box-resize + overlap-removal loop** with fixed translational and rotational move sizes.

---

## 1. What this project does

This project performs **density increase by controlled isotropic compression** for a hard-particle system. It is designed for situations where you already have an initial GSD configuration and want to compress it gradually while maintaining a valid hard-particle state.

The core idea is simple:

1. Start from an initial configuration stored in a GSD file.
2. Shrink the simulation box isotropically by a small amount.
3. Because the shrink can introduce geometric overlaps, run HPMC until overlaps disappear.
4. Relax the system at the new fixed box size.
5. Save checkpoints and continue.
6. Stop when the target packing fraction is reached.

This is not an adaptive NPT scheme and not a soft-potential MD workflow. It is a **manual compression engine** that gives you full control over:

- how aggressively the box shrinks,
- how hard you work to remove overlaps after each shrink,
- how much fixed-box relaxation you apply at each compression stage,
- what shape is simulated,
- whether the run is single-stage or chained across multiple stages.

In short, this project is for **controlled geometric compression of hard anisotropic particles**.

---

## 2. Scientific / computational setting

The simulation uses the **hard-particle Monte Carlo** framework from HOOMD-blue. That means the interaction is purely geometric:

- there are **no continuous forces**,
- there is **no potential energy minimization**,
- a configuration is valid if particles **do not overlap**,
- a configuration is invalid if particles **overlap**.

For convex polyhedra, orientations matter. So this script expects the GSD to contain valid particle orientations and uses a **ConvexPolyhedron** HPMC integrator.

The packing fraction is computed as

\[
\phi = \frac{N V_\mathrm{particle}}{V_\mathrm{box}}
\]

where:

- \(N\) is the number of particles,
- \(V_\mathrm{particle}\) is the actual scaled particle volume,
- \(V_\mathrm{box}\) is the current simulation-box volume.

For this project:

- the reference shape volume comes from the shape JSON,
- the actual particle volume is:

\[
V_\mathrm{particle} = V_\mathrm{reference}\times (\text{shape\_scale})^3
\]

For the supplied cube shape JSON, the reference volume is `1.0`, so with `shape_scale = 1.0`, the particle volume is also `1.0`.

---

## 3. Files in this project

## 3.1 Main driver

### `HOOMD_hard_polyhedra_compression.py`

This is the full simulation engine. It contains:

- parameter loading and validation,
- shape-file loading and validation,
- GSD validation,
- stage-aware filename resolution,
- MPI-aware root-rank printing,
- deterministic seed-file management,
- HOOMD simulation construction,
- HPMC ConvexPolyhedron integrator setup,
- live logging and writer attachment,
- the manual two-level compression loop,
- checkpoint generation,
- final-output generation,
- emergency snapshot logic.

The file is written in a highly explanatory style, with large internal sections describing both the mathematics and the runtime behavior.

---

## 3.2 Simulation parameter file

### `simulparam_hard_polyhedron_compression.json`

This file defines the specific run.

The supplied example contains:

- tag: `HOOMD_hard_cube_2197_compression_to_pf0p58`
- input GSD: `cube_sc_disorder_phi0p10.gsd`
- stage mode: single-stage (`stage_id_current = -1`)
- shape file: `shape_023_Cube_unit_volume_principal_frame.json`
- shape scale: `1.0`
- target packing fraction: `0.58`
- volume scaling factor: `0.99`
- overlap-removal run length: `1000`
- relaxation run length: `2000`
- translational move size: `0.03`
- rotational move size: `0.02`
- CPU execution (`use_gpu = false`)
- explicit trajectory / restart / final output filenames

This JSON is both a **runtime configuration** and a **human-readable experiment descriptor**.

---

## 3.3 Shape definition file

### `shape_023_Cube_unit_volume_principal_frame.json`

This file defines the convex polyhedron being simulated.

In the supplied example:

- shape name: `Cube`
- short name: `P03`
- reference volume: `1.0`
- center of mass: `(0, 0, 0)`
- moment of inertia tensor components are provided
- vertices describe a cube centered at the origin with corners at \(\pm 0.5\)

Because the cube is centered and has unit volume, scaling by `shape_scale = 1.0` leaves the geometry unchanged.

This shape JSON acts as the **authoritative geometric source** for:

- the particle volume used in packing-fraction calculations,
- the vertex list passed to the HPMC ConvexPolyhedron integrator,
- shape-metadata consistency checks against GSD files.

---

## 3.4 Referenced but not included in the upload

### `cube_sc_disorder_phi0p10.gsd`

The JSON references this as the initial state, but it is not part of the uploaded bundle you gave me here. The script expects it to exist at runtime.

That file is expected to contain:

- at least one frame,
- nonzero particle count,
- one particle type only (`typeid = 0`),
- valid particle orientations with shape `(N, 4)`,
- a valid simulation box,
- a configuration suitable for hard-particle simulation.

If it also contains `type_shapes`, the script cross-checks those vertices against the shape JSON to ensure the GSD and the shape definition are consistent.

---

## 4. High-level algorithm

The compression logic is a **two-level nested loop**.

### Outer loop: box compression

As long as the current box volume is larger than the target volume, the script:

1. computes a new box volume
2. rescales the box isotropically
3. updates the packing fraction
4. checks how many overlaps were created by that shrink

The new volume is:

\[
V_\text{new} = \max\left(V_\text{current}\times \text{volume\_scaling\_factor},\;V_\text{target}\right)
\]

This means the script never overshrinks past the target volume.

---

### Inner loop: overlap removal

After each box shrink, the configuration may contain overlaps. The script then repeatedly does:

1. `sim.run(run_length_to_remove_overlap)`
2. read `mc.overlaps`
3. continue until overlaps reach zero

So each compression step is not considered complete until the configuration is again a valid hard-particle state.

---

### Post-overlap relaxation

Once overlaps are eliminated, the script performs a fixed-box equilibration run:

```text
sim.run(run_length_to_relax)
```

This lets the system relax structurally at the new density before the next compression step.

---

### Checkpoint writing after each outer step

After overlap removal and relaxation, the script writes:

- `current_pf.json`
- `output_current_pf.gsd`

so you can monitor progress and inspect the current compressed state.

---

## 5. Runtime workflow in detail

When you launch the script, the flow is:

1. Parse `--simulparam_file`
2. Load and validate the JSON
3. Resolve filenames according to single-stage or multi-stage mode
4. Create or read a random-seed file
5. Build the HOOMD simulation
6. Perform a short initial run (`100` steps) to initialize counters
7. Check the initial overlap count
8. Enter the manual compression loop
9. Write final outputs and summary files
10. Exit successfully, or emit detailed failure messages

This structure makes the simulation highly transparent: every major stage has explicit console output and strong validation.

---

## 6. Core design philosophy

This project is not trying to hide the compression logic behind a black box. Instead it chooses:

- **manual control** over barostat-style automation,
- **explicit overlap handling** after each shrink,
- **deterministic file naming and seed handling**,
- **checkpoint-heavy execution** for restartability,
- **self-describing GSD outputs** for downstream visualization and analysis.

That makes it especially suitable for:

- method development,
- debugging packing and jamming behavior,
- benchmarking compression strategies,
- generating high-density hard-particle configurations,
- staged protocols where you manually adjust parameters between phases.

---

## 7. Installation and dependencies

The script expects:

- Python `>= 3.8`
- HOOMD-blue `>= 4.0` (documented/tested with `4.9.0` in the script comments)
- `gsd >= 3.0`
- `numpy >= 1.20`

A typical environment would be a dedicated HOOMD conda environment or a system Python where HOOMD and GSD are already installed.

Example idea:

```bash
python -c "import hoomd, gsd.hoomd, numpy; print(hoomd.version.version)"
```

If HOOMD cannot be imported, the script exits immediately with a fatal message.

---

## 8. How to run

### 8.1 Single-node CPU

```bash
python3 HOOMD_hard_polyhedra_compression.py --simulparam_file simulparam_hard_polyhedron_compression.json
```

---

### 8.2 Single-node GPU

Set in the JSON:

```json
"use_gpu": true,
"gpu_id": 0
```

Then run the same command:

```bash
python3 HOOMD_hard_polyhedra_compression.py --simulparam_file simulparam_hard_polyhedron_compression.json
```

If GPU initialization fails, the script falls back to CPU and prints a warning.

---

### 8.3 MPI run

```bash
mpirun -np 4 python3 HOOMD_hard_polyhedra_compression.py --simulparam_file simulparam_hard_polyhedron_compression.json
```

Important note: the current JSON contains a fixed `gpu_id`. If you want one GPU per rank, you would typically use a wrapper script or rank-aware environment handling so each rank maps to the correct local GPU.

---

## 9. Single-stage vs multi-stage execution

The script supports two modes.

### 9.1 Single-stage mode

Set:

```json
"stage_id_current": -1
```

In this mode:

- the input GSD comes directly from `input_gsd_filename`
- output filenames come directly from the JSON fields:
  - `restart_gsd_filename`
  - `output_gsd_traj_filename`
  - `final_gsd_filename`

This is the mode used by your supplied JSON.

---

### 9.2 Multi-stage mode

Set:

```json
"stage_id_current": 0
```

or a higher integer.

In multi-stage mode:

- filenames are auto-generated using:

```text
<tag>_<stage_id>_restart.gsd
<tag>_<stage_id>_traj.gsd
<tag>_<stage_id>_final.gsd
```

- stage `0` reads from `input_gsd_filename`
- stage `N > 0` reads from:

```text
<tag>_<N-1>_final.gsd
```

This lets you chain compression campaigns like:

- stage 0: low density to intermediate density
- stage 1: intermediate to higher density
- stage 2: higher to near-jammed density

This is very useful if move sizes or relaxation lengths need to be adjusted progressively.

---

## 10. Restart logic

Restart behavior is intentionally simple.

During the build phase, the script checks:

- whether the restart GSD exists
- whether the final GSD exists

If:

- restart exists, and
- final does **not** exist

then the script resumes from the restart GSD.

Otherwise it starts fresh from the input GSD.

This means:

- periodic restarts support recovery from walltime limits or interrupts,
- completed runs are identified by the existence of the final GSD,
- fresh runs remain easy to reason about.

---

## 11. Random seed handling

The script uses explicit seed-file management for reproducibility.

### Single-stage seed file

```text
random_seed.json
```

### Multi-stage seed file

```text
random_seed_stage_0.json
```

Behavior:

- rank 0 creates the seed file if it does not exist,
- all ranks wait for that file to become visible,
- all ranks read the exact same seed,
- HOOMD is then initialized with that seed.

This is especially important in MPI runs, where inconsistent seeding would produce divergent behavior.

The script also includes a simple file-based barrier with a timeout while waiting for the seed file to appear.

---

## 12. Geometry and shape handling

The shape JSON is not just decorative metadata. It is a central part of the simulation.

The script:

1. loads the reference volume,
2. loads the vertex array,
3. validates that the vertices have shape `(Nv, 3)`,
4. checks that there are at least four vertices,
5. rescales the vertices by `shape_scale`,
6. computes the actual particle volume,
7. passes the scaled vertices to:

```python
hoomd.hpmc.integrate.ConvexPolyhedron
```

For the supplied cube:

- reference volume = `1.0`
- scaled volume = `1.0`
- vertices remain the standard centered cube vertices

This approach makes it easy to swap shapes without changing the compression logic itself.

---

## 13. Input GSD requirements

Before running, the script validates the input GSD carefully.

It expects:

1. the file exists,
2. the file has at least one frame,
3. the particle count is nonzero,
4. all particles belong to a **single type** with `typeid = 0`,
5. particle orientations exist and have shape `(N, 4)`,
6. orientations are finite and nonzero-norm,
7. if `type_shapes` are present, they agree with the shape JSON.

This is a good design choice for convex polyhedra because anisotropic particles require valid orientation data.

If any of these checks fail, the script exits with a detailed fatal error.

---

## 14. Box-resize strategy

The project uses **isotropic affine rescaling**.

If the volume changes from \(V_\text{old}\) to \(V_\text{new}\), the scale factor is:

\[
s=\left(\frac{V_\text{new}}{V_\text{old}}\right)^{1/3}
\]

Then:

- `Lx_new = s * Lx_old`
- `Ly_new = s * Ly_old`
- `Lz_new = s * Lz_old`

Tilt factors `xy`, `xz`, and `yz` are preserved.

This means:

- aspect ratio is preserved,
- the box shape is preserved,
- all particle positions are affinely scaled,
- the shrink itself can introduce overlaps,
- those overlaps are then removed by HPMC.

This is exactly the right logic for a manual geometric compression protocol.

---

## 15. HPMC integrator configuration

The integrator used is:

```python
hoomd.hpmc.integrate.ConvexPolyhedron(nselect=1)
```

Then the script sets:

- `mc.shape["A"] = {"vertices": params.shape_vertices}`
- `mc.d["A"] = params.move_size_translation`
- `mc.a["A"] = params.move_size_rotation`

So the system assumes:

- one particle type only,
- that type is named `"A"`,
- all particles share the same convex-polyhedron geometry,
- translation and rotation move sizes are fixed throughout the run.

This project intentionally does **not** use adaptive move-size tuning.

That is an explicit design decision, not an omission by accident.

---

## 16. Logging strategy

The script constructs custom loggers for:

- live packing fraction
- overlap count
- translational acceptance
- rotational acceptance

These are wrapped carefully because HOOMD can raise `DataAccessError` before the first `sim.run()` when counters are not initialized yet.

The script also attaches:

### GSD logger

For trajectory and restart files, it logs:

- `timestep`
- `tps`
- `walltime`
- `type_shapes`
- `translate_moves`
- `rotate_moves`
- custom compression quantities

### Table writer

For console monitoring, it writes a periodic table including:

- timestep
- tps
- translate moves
- rotate moves
- packing fraction
- overlap count
- translational acceptance
- rotational acceptance

This makes the runtime behavior easy to follow while the simulation is active.

---

## 17. Compression loop in practical terms

Here is the operational meaning of the two-loop structure.

### After each box shrink

The system is denser, but maybe invalid because particles overlap.

### Inner loop

The system is allowed to Monte Carlo evolve until those overlaps disappear.

### Relaxation

Even after overlaps vanish, the new state may still be far from a relaxed configuration at that density. So the script runs a further fixed-box block.

### Checkpoint

Only then does it record the current packing fraction and current compressed snapshot.

This is a robust workflow because it separates:

- **geometric invalidity removal** from
- **structural relaxation**.

That distinction is especially important for dense hard-particle systems.

---

## 18. Checkpoint files during compression

After each outer compression step, the script writes:

### `current_pf.json`

This contains:

- current packing fraction
- outer-step index
- timestep
- overlap count after relaxation
- number of inner-loop iterations used
- current box volume
- timestamp

This is mainly for lightweight monitoring and progress tracking.

---

### `output_current_pf.gsd`

This is a single-frame snapshot overwritten at each compression step.

It gives you the most recent compressed state during the run.

This is useful when:

- monitoring a long simulation,
- debugging jamming behavior,
- inspecting intermediate structures without waiting for final completion.

---

## 19. Final outputs

At successful completion, the script writes several outputs.

### 19.1 Labelled compressed snapshot

A filename of the form:

```text
<tag>_compressed_to_pf_<rounded_phi>.gsd
```

For example, if the final packing fraction were `0.5800`, the label portion becomes `0p5800`.

This is a human-friendly final snapshot.

---

### 19.2 Canonical final GSD

This is the formal run-completion snapshot used for stage chaining and downstream processing.

In your current JSON:

```text
HOOMD_hard_cube_2197_compression_to_pf0p58_final.gsd
```

for single-stage mode.

---

### 19.3 Trajectory GSD

This is appended throughout the run.

It stores the trajectory and logged data for visualization and post-analysis.

---

### 19.4 Restart GSD

This is rewritten periodically as a single-frame checkpoint.

It is the primary restart mechanism.

---

### 19.5 Compression summary JSON

The script writes:

```text
<tag>_compression_summary.json
```

with metadata such as:

- tag
- stage ID
- input GSD
- final GSD
- labelled compressed GSD
- particle count
- shape metadata
- target / initial / final packing fractions
- final overlap count
- final timestep
- seed
- final box lengths
- creation timestamp

This is excellent for provenance and bookkeeping.

---

## 20. Shape metadata in output GSDs

A particularly strong feature of this project is that it explicitly tries to preserve or verify **self-describing shape metadata**.

The final-output logic:

1. flushes all attached writers,
2. writes final snapshots with explicit `type_shapes`,
3. verifies that:
   - trajectory GSD
   - restart GSD
   - final GSD

all contain `particles/type_shapes` and that the stored vertices match the expected shape.

This is very useful for OVITO or any downstream pipeline where shape fidelity matters.

---

## 21. Example meaning of the current JSON configuration

Your supplied JSON corresponds to the following experiment conceptually:

- shape: cube
- reference particle volume: `1.0`
- scale factor: `1.0`
- target packing fraction: `0.58`
- box shrink factor per outer step: `0.99`
- overlap-removal block length: `1000`
- fixed-box relaxation block length: `2000`
- translation move size: `0.03`
- rotation move size: `0.02`
- CPU execution
- explicit restart/trajectory/final filenames
- single-stage mode

Operationally, that means each outer compression step reduces the box volume by about **1%**, unless doing so would overshoot the target volume.

Because a 1% box-volume shrink corresponds to a smaller linear-dimension shrink, the compression is fairly gentle, which is often a sensible starting point for hard-particle systems.

---

## 22. Parameter reference

This is the most important section if you plan to adapt the workflow.

### `tag`
Human-readable identifier for the run.

Used in output filenames and summaries.

Choose something descriptive and filesystem-safe.

---

### `input_gsd_filename`
The initial configuration file.

Must exist at runtime.

Must be compatible with one-component convex-polyhedron HPMC.

---

### `stage_id_current`

Controls single-stage vs multi-stage logic.

- `-1` = single-stage
- `0,1,2,...` = multi-stage

If you are not chaining stages, keep this at `-1`.

---

### `initial_timestep`

Starting timestep for fresh runs.

Useful if you want a run to begin from a nonzero logical timestep.

---

### `shape_json_filename`

Path to the shape-definition JSON.

This file defines the reference polyhedron volume and vertices.

---

### `shape_scale`

Uniform linear scale factor applied to the reference shape.

Effects:

- vertices scale linearly,
- particle volume scales cubically.

If you double the linear scale, volume becomes eight times larger.

---

### `target_pf`

Desired final packing fraction.

The script validates that it lies in `(0, 0.74048)`.

Very high target values may be geometrically or kinetically difficult depending on shape and protocol.

---

### `volume_scaling_factor`

The multiplicative box-volume factor applied at each outer compression step.

Examples:

- `0.99` = gentle 1% volume reduction per step
- `0.995` = even gentler compression
- `0.98` = more aggressive compression

Smaller values compress faster but tend to generate more overlaps and can drive the system into jamming more easily.

---

### `run_length_to_remove_overlap`

How many simulation steps are run per inner-loop overlap-removal attempt.

If too small:

- you may need many inner-loop iterations,
- overlap removal may look painfully slow.

If too large:

- each attempt becomes expensive even when only a few overlaps remain.

---

### `run_length_to_relax`

The fixed-box run length after overlaps are removed.

This is not overlap removal. It is structural relaxation at the new density.

Longer values improve relaxation but increase cost.

---

### `move_size_translation`

Fixed translational move size for type `"A"`.

Too small:
- sampling becomes slow.

Too large:
- rejection rate rises, especially at higher density.

This is one of the most important tuning parameters in the project.

---

### `move_size_rotation`

Fixed rotational move size.

This matters for anisotropic particles. Even for cubes, rotational mobility can affect how efficiently the system reorganizes after compression.

Too small:
- rotational relaxation may be sluggish.

Too large:
- rotations may be rejected too often.

---

### `restart_frequency`

How often the single-frame restart checkpoint is rewritten.

Smaller values give more fault tolerance but produce more I/O.

---

### `traj_out_freq`

How often a trajectory frame is appended.

Choose based on how much temporal detail you want versus how large you want the file to become.

---

### `log_frequency`

How often the console table is written.

This is useful for long runs where live monitoring matters.

---

### `use_gpu`

Whether to request GPU execution.

If `true`, the script attempts to initialize a HOOMD GPU device.

---

### `gpu_id`

Which CUDA device index to use if GPU mode is requested.

---

### `restart_gsd_filename`, `output_gsd_traj_filename`, `final_gsd_filename`

Used directly in single-stage mode.

Ignored in favor of auto-generated names in multi-stage mode.

---

## 23. Tuning advice

The most important parameters for practical success are:

- `volume_scaling_factor`
- `move_size_translation`
- `move_size_rotation`
- `run_length_to_remove_overlap`
- `run_length_to_relax`

### If overlap removal takes forever

Try:

- increasing `volume_scaling_factor` closer to `1.0`
- decreasing `move_size_translation`
- adjusting `move_size_rotation`
- increasing `run_length_to_remove_overlap`

---

### If the system freezes at high density

Try:

- more conservative compression,
- longer relaxation blocks,
- multi-stage compression,
- smaller move sizes at later stages.

---

### If you want faster but riskier compression

Try:

- a smaller `volume_scaling_factor`, such as `0.98`

but be aware that this can dramatically increase overlap-removal difficulty.

---

### If you want safer and more equilibrated compression

Try:

- `0.995` or `0.998`,
- longer relaxation blocks,
- stagewise targets.

---

## 24. Typical failure modes

### 24.1 Infinite or effectively endless inner loop

Symptoms:

- overlap counts decrease very slowly,
- the same outer step takes forever,
- CPU time is consumed but progress is minimal.

Likely causes:

- compression too aggressive,
- move sizes poorly tuned,
- dense jammed state,
- not enough overlap-removal work per iteration.

---

### 24.2 Kinetic arrest

Symptoms:

- overlap count is zero,
- but acceptance becomes very poor,
- structural reorganization essentially stops.

Likely causes:

- system is too dense for the current move sizes,
- insufficient relaxation during the path to high density,
- protocol drives the system into a glassy or jammed regime.

---

### 24.3 Bad input GSD

Symptoms:

- failure before compression starts,
- orientation errors,
- type mismatch,
- missing frames,
- shape inconsistency.

This usually means the initial configuration is incompatible with the current shape or with the script’s one-component assumption.

---

### 24.4 Filesystem / restart issues in MPI

Because seed handling uses a file-based synchronization step, a sluggish or inconsistent shared filesystem can cause delays or a timeout while waiting for the seed file.

---

## 25. Safety and recovery features

The project includes several protection mechanisms.

### Validation before expensive work

It validates:

- JSON types and required keys,
- physical ranges for parameters,
- existence of input files,
- shape-file structure,
- GSD structure,
- stage dependencies.

---

### Emergency handling for runaway inner loop

If the inner loop exceeds a large iteration count, the script attempts to write an emergency snapshot and then exits.

---

### Exception recovery in `main()`

Unexpected exceptions during compression trigger an attempt to write an emergency restart snapshot with shape metadata.

---

### Periodic restart writing

This is the main recovery mechanism for routine interruptions.

---

## 26. Important assumptions

This project assumes:

1. **Monodisperse system**
   - all particles are identical

2. **Single particle type**
   - only type `"A"`

3. **Hard interaction only**
   - no soft forces, no attractions

4. **Isotropic compression**
   - `Lx`, `Ly`, `Lz` scale together

5. **Periodic boundaries**
   - standard periodic HOOMD box

6. **Convex polyhedron shape**
   - the integrator and shape metadata are built around that assumption

If you want mixtures, anisotropic box control, soft interactions, or adaptive move tuning, this script would need extension.

---

## 27. What this script intentionally does not do

This project deliberately does **not** provide:

- adaptive move-size tuning,
- bidisperse or multicomponent support,
- anisotropic compression protocols,
- NPT barostat logic,
- soft-potential MD,
- automatic walltime-aware restart chaining,
- full automated workflow orchestration.

That is part of what makes the script transparent and easier to reason about.

---

## 28. Known implementation notes and quirks

These are not necessarily fatal, but they are worth knowing.

### 28.1 The script text still says “hard sphere” in a few summary banners
Some summary strings refer to “hard sphere” even though the actual simulation is for convex polyhedra. This is just wording and does not change runtime behavior.

---

### 28.2 `output_current_pf.gsd` is written with the built-in single-frame writer
The final labelled and canonical snapshots are written with explicit shape metadata. The trajectory and restart files also log `type_shapes` and are verified at the end. The per-step checkpoint `output_current_pf.gsd` is written through the simpler writer path and is mainly for transient monitoring.

---

### 28.3 One error path in multi-stage filename resolution appears to contain a typo
In the error message for missing previous-stage output, the message text references `stage_id` instead of the local stage variable used elsewhere. That only matters if that particular error branch is hit, but it is worth correcting in a future cleanup.

---

### 28.4 Optional JSON keys are parsed but not currently active in the visible workflow
The script accepts optional keys such as:

- `traj_out_freq_POS`
- `color_code_for_POS_file`

but they are not part of the active compression/output path shown in this script version.

---

## 29. Suggested directory layout

A clean project layout would look like:

```text
project_root/
├── HOOMD_hard_polyhedra_compression.py
├── simulparam_hard_polyhedron_compression.json
├── shape_023_Cube_unit_volume_principal_frame.json
├── cube_sc_disorder_phi0p10.gsd
├── random_seed.json
├── current_pf.json
├── output_current_pf.gsd
├── HOOMD_hard_cube_2197_compression_to_pf0p58_restart.gsd
├── HOOMD_hard_cube_2197_compression_to_pf0p58_traj.gsd
├── HOOMD_hard_cube_2197_compression_to_pf0p58_final.gsd
└── HOOMD_hard_cube_2197_compression_to_pf0p58_compression_summary.json
```

Some of these appear only after the run begins or completes.

---

## 30. Minimal workflow recipe

If you want the shortest operational recipe:

1. Place the script, parameter JSON, shape JSON, and input GSD in one directory.
2. Check that the input GSD matches the shape and contains valid orientations.
3. Run the script with the JSON file.
4. Watch console logs for:
   - overlap counts,
   - packing fraction,
   - acceptance,
   - checkpoint writes.
5. Inspect:
   - `current_pf.json`
   - `output_current_pf.gsd`
   - restart and trajectory GSDs
6. At completion, use:
   - final GSD
   - labelled compressed GSD
   - compression summary JSON

---

## 31. How to adapt this project to other shapes

To simulate another convex polyhedron:

1. prepare a new shape JSON with:
   - name,
   - short name,
   - reference volume,
   - vertex list

2. point `shape_json_filename` to that file

3. set `shape_scale` appropriately

4. ensure the input GSD corresponds to that geometry

5. retune:
   - translation move size,
   - rotation move size,
   - compression aggressiveness,
   - relaxation lengths

Different shapes can have very different packing and relaxation behavior, so parameter retuning is usually necessary.

---

## 32. How to adapt this project to denser targets

If you want to push beyond the current `target_pf = 0.58`, the most practical path is usually:

- use stagewise compression,
- make later stages gentler,
- reduce move sizes at high density,
- increase overlap-removal and relaxation lengths,
- monitor acceptance and overlap-removal difficulty closely.

That is usually much more stable than trying to reach a very dense target in one aggressive stage.

---

## 33. Recommended future improvements

If this project is going to evolve further, the most useful next additions would be:

1. adaptive move-size tuning option,
2. optional max-walltime / auto-resubmit support,
3. optional stage-plan file for multi-stage workflows,
4. explicit final validation that `output_current_pf.gsd` also contains shape metadata,
5. cleanup of legacy wording such as “hard sphere”,
6. removal or implementation of currently inactive optional JSON keys,
7. an analysis notebook or utility script for reading the trajectory and plotting:
   - packing fraction vs timestep,
   - overlap count vs timestep,
   - translational acceptance vs timestep,
   - rotational acceptance vs timestep.

---

## 34. Bottom line

This project is a **manual, transparent, restart-aware HOOMD-blue v4 compression workflow for hard convex polyhedra**.

Its strengths are:

- explicit algorithmic control,
- strong validation,
- good runtime observability,
- useful checkpointing,
- reproducible seed management,
- self-describing output files,
- suitability for staged dense-packing studies.

For anyone developing or debugging hard-particle compression workflows, this is a strong base script because the logic is easy to inspect, easy to modify, and easy to trust.

---
