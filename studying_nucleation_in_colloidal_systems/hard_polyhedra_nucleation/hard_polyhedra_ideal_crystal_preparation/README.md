# Convex-Polyhedron Lattice Generator from Packing Fraction (HOOMD-blue v4)

python3.10 make_lattice_hard_polyhedron_based_on_phi.py input_lattice_configuration.json

## 1. What this project is

This project generates an **initial periodic lattice configuration of identical convex polyhedra** at a user-specified packing fraction and writes that configuration to a **GSD** file that can be used as the starting state for **HOOMD-blue v4 HPMC** simulations.

The code is a polyhedron-based extension of an earlier hard-sphere lattice generator. The central idea is the same as before:

1. choose a lattice type,
2. choose a target packing fraction,
3. compute the cubic unit-cell length needed to realize that packing fraction,
4. replicate the unit cell to obtain a system with at least the requested number of particles,
5. assign particle orientations,
6. write the result to a GSD snapshot,
7. validate the constructed geometry by asking HOOMD itself whether any overlaps are present.

The main difference from the hard-sphere version is that the particle shape is no longer represented by a single diameter. Instead, the particle geometry is defined by a **convex polyhedron JSON file** containing a reference set of vertices and a reference particle volume. The code then linearly scales that reference shape and uses the scaled particle volume in the packing-fraction calculation.

---

## 2. Why this script exists

When preparing HPMC simulations of anisotropic particles, one common need is a **clean, deterministic, overlap-checked starting configuration**.

There are many ways to produce a dense configuration:

- random insertion,
- Monte Carlo compression,
- box rescaling,
- packing solvers,
- custom crystal builders,
- importing externally generated coordinates.

However, each of those approaches has drawbacks when the goal is specifically to produce a simple, reproducible, mathematically controlled initial state:

- random insertion becomes inefficient at moderate density,
- dynamic compression may require tuning and can fail or jam,
- external builders may not match the simulation shape definition exactly,
- random anisotropic placement often leads to overlaps that are not trivial to diagnose.

This script avoids those issues by working from a **known lattice geometry** and by constructing the box length directly from the target packing fraction. It then uses **HOOMD’s own convex-polyhedron overlap engine** to confirm that the generated configuration is geometrically admissible for the chosen shape, size, lattice, and orientation pattern.

So the philosophy of the code is:

- **build analytically**,
- **validate numerically**,
- **fail early if invalid**.

---

## 3. What the script does at a high level

Given an input JSON file, the script does the following in order:

1. reads and validates the user input,
2. reads the reference polyhedron description from a second JSON file,
3. scales the reference vertices by the requested linear factor,
4. computes the actual particle volume after scaling,
5. chooses the conventional cubic lattice basis for SC, BCC, or FCC,
6. computes the cubic unit-cell edge length `a` from the packing fraction,
7. chooses a nearly cubic replication `(nx, ny, nz)` large enough to give at least the requested number of particles,
8. constructs all particle center positions in Cartesian coordinates,
9. assigns all particle orientations either as:
   - one common quaternion (`order` mode), or
   - independent random unit quaternions (`disorder` mode),
10. writes one GSD frame containing the generated configuration,
11. reloads that GSD into HOOMD-blue v4,
12. attaches `hoomd.hpmc.integrate.ConvexPolyhedron()`,
13. defines the polyhedron for particle type `"A"`,
14. runs `simulation.run(0)` to initialize the state and overlap computation,
15. queries `mc.overlaps`,
16. exits with success only if HOOMD reports zero overlaps.

For moderate system sizes, the script also computes a **minimum center-to-center distance diagnostic** as a geometric sanity check.

---

## 4. What is meant by “packing fraction” here

This point is crucial.

The packing fraction in this script is defined as:

\[
\phi = \frac{\text{total particle volume}}{\text{simulation box volume}}.
\]

For a cubic conventional cell with `N_cell` particles per cell and particle volume `V_particle`, the code uses:

\[
\phi = \frac{N_{\text{cell}} V_{\text{particle}}}{a^3},
\]

where:

- `a` is the conventional cubic unit-cell edge length,
- `N_cell = 1` for SC,
- `N_cell = 2` for BCC,
- `N_cell = 4` for FCC.

Solving for `a` gives:

\[
a = \left(\frac{N_{\text{cell}}V_{\text{particle}}}{\phi}\right)^{1/3}.
\]

This formula is the foundation of the script.

The code does **not** obtain the target density by compression, relaxation, Monte Carlo annealing, or dynamic box rescaling. Instead, it determines the unit-cell length analytically so that the **nominal packing fraction is achieved by construction**.

That is why the particle volume must be known. For spheres, the script could infer the volume from a single diameter. For polyhedra, the script needs a reference shape and a linear scale.

---

## 5. Role of the reference shape JSON

The polyhedron geometry is stored separately in a shape JSON file, such as:

- `shape_023_Cube_unit_volume_principal_frame.json`

This shape file is expected to contain at least the following keys:

- `"4_volume"` : the reference particle volume,
- `"8_vertices"` : an `N x 3` list of vertex coordinates in the particle body frame.

Typical metadata also includes a shape name, short name, center of mass, and moment of inertia tensor.

### Why separate the shape into its own JSON file?

Because the same geometric particle can be reused with:

- different packing fractions,
- different lattice types,
- different system sizes,
- different orientation modes,
- different scale factors.

The shape JSON is therefore a reusable **geometric template**, while the input configuration JSON is the **simulation-construction request**.

---

## 6. Meaning of `scale`

The field

```json
"scale": 1.0
```

is the **linear scaling factor** applied to the reference vertices from the shape JSON.

If a reference vertex is

\[
\mathbf{r}_i^{(0)},
\]

the actual vertex used in the simulation becomes

\[
\mathbf{r}_i = \text{scale} \times \mathbf{r}_i^{(0)}.
\]

That means:

- all lengths scale as `scale`,
- all areas scale as `scale^2`,
- the particle volume scales as `scale^3`.

So if the reference shape volume is `V0`, the actual particle volume is:

\[
V_{\text{particle}} = V_0 \times \text{scale}^3.
\]

In other words, `scale` is the polyhedron analogue of “particle size”.

### Why is `scale` necessary?

Because many shape libraries store particles in a normalized or canonical form, such as:

- unit volume,
- unit circumradius,
- unit insphere radius,
- principal-axis frame,
- centered at the origin.

The `scale` field lets you take that reference shape and use it at the actual size you want in the simulation.

---

## 7. Supported lattice types

The script currently supports only three conventional cubic lattices:

### SC

Simple cubic:

- basis positions: `(0,0,0)`
- particles per unit cell: `1`
- nearest-neighbor distance: `a`

### BCC

Body-centered cubic:

- basis positions: `(0,0,0)` and `(1/2, 1/2, 1/2)`
- particles per unit cell: `2`
- nearest-neighbor distance: `sqrt(3) a / 2`

### FCC

Face-centered cubic:

- basis positions:
  - `(0,0,0)`
  - `(1/2,1/2,0)`
  - `(1/2,0,1/2)`
  - `(0,1/2,1/2)`
- particles per unit cell: `4`
- nearest-neighbor distance: `a / sqrt(2)`

### Why only these three?

Because the script is designed around a **cubic conventional cell** and hard-coded basis sets. That keeps the construction deterministic and clear. It also avoids introducing ambiguity in packing-fraction formulas for arbitrary unit cells.

The current version does **not** support:

- HCP,
- diamond,
- user-defined bases,
- triclinic cells,
- multicomponent crystals,
- basis-dependent orientation patterns,
- non-cubic conventional cells.

---

## 8. Input JSON file: fields and meaning

The script reads a main input file from the command line. A typical example is:

```json
{
    "lattice_type": "SC",
    "phi": 0.10,
    "shape_json": "shape_023_Cube_unit_volume_principal_frame.json",
    "scale": 1.0,
    "n_particles_req": 2197,
    "orientation_mode": "order",
    "ordered_orientation": [1.0, 0.0, 0.0, 0.0],
    "output_gsd": "cube_sc_order_phi0p10.gsd"
}
```

Below is the meaning of every field.

### `lattice_type`

Allowed values:

- `"SC"`
- `"BCC"`
- `"FCC"`

This selects which conventional cubic lattice basis is used to place particle centers.

### `phi`

Target packing fraction.

This must be a finite positive number.

It controls how large the conventional unit cell must be for the chosen particle volume and lattice type.

### `shape_json`

Path to the polyhedron definition JSON.

If a relative path is provided, it is resolved relative to the location of the main input JSON file.

### `scale`

Linear scale applied to the reference vertices.

The actual particle volume is computed from the reference volume times `scale^3`.

### `n_particles_req`

Requested particle count.

This is treated as a **lower bound**, not a strict equality target. The script will choose a nearly cubic supercell with at least this many particles.

So the final particle count can be larger than requested.

### `orientation_mode`

Allowed logical choices:

- `"order"`
- `"ordered"`
- `"disorder"`
- `"disordered"`
- `"random"`

The code normalizes these inputs to either:

- `order`
- `disorder`

### `ordered_orientation`

Required when `orientation_mode = "order"`.

This must be a quaternion in HOOMD ordering:

```text
[w, x, y, z]
```

The code normalizes it to a unit quaternion before assigning it to every particle.

### `rng_seed`

Optional integer seed.

Used when `orientation_mode = "disorder"` so that random quaternion assignment is reproducible.

### `output_gsd`

Optional output filename.

If omitted, the script auto-generates a descriptive filename using lattice type, shape name, orientation mode, packing fraction, particle count, and lattice constant.

---

## 9. Shape JSON file: expected structure

The helper `load_polyhedron_shape(...)` expects the shape file to contain at least:

```json
{
    "4_volume": 1.0,
    "8_vertices": [[...], [...], ...]
}
```

The code also reads optional descriptive metadata such as:

- `"1_Name"`
- `"2_ShortName"`

These are used mainly for human-readable reporting and automatic filename generation.

### What the vertices represent

The vertices are assumed to be in the **body frame** of the particle.

That means:

- the shape is defined relative to its own particle-fixed coordinate system,
- the particle orientation quaternion rotates that body-frame shape into the simulation box frame,
- the center of each particle is placed at the lattice point.

The code does **not** reconstruct a convex hull from arbitrary point clouds. It assumes the provided vertices already represent the intended convex polyhedron.

---

## 10. Ordered versus disordered orientation modes

This is one of the most important additions in the polyhedron version.

### Ordered mode

When `orientation_mode = "order"`, every particle receives the **same orientation quaternion**.

This means the entire lattice is orientationally aligned with respect to the global box frame.

Example:

```json
"orientation_mode": "order",
"ordered_orientation": [1.0, 0.0, 0.0, 0.0]
```

This is the identity quaternion, so the body-frame vertices are used without rotation.

### Disordered mode

When `orientation_mode = "disorder"`, the code samples a separate random unit quaternion for each particle.

This creates an orientationally random assembly placed on the chosen crystal lattice of particle centers.

### Why is this distinction useful?

Because for anisotropic particles, center positions alone do not determine overlap freedom. A configuration that is overlap-free in one orientation pattern may overlap badly in another.

So the script allows you to explore two conceptually different starting states:

- **translationally ordered + orientationally ordered**
- **translationally ordered + orientationally disordered**

The overlap check then tells you whether the chosen shape/size/density/orientation combination is geometrically admissible.

---

## 11. How random quaternions are generated

The helper `random_unit_quaternions(...)` produces random unit quaternions using three independent uniform random numbers per particle.

The returned quaternion ordering is:

```text
[qw, qx, qy, qz]
```

which matches the `[w, x, y, z]` convention used in HOOMD snapshot particle orientations.

### Why not just sample four random numbers and normalize?

Because naive normalization of four Gaussian or uniform random numbers does not necessarily give the desired uniform rotational distribution in the intended parameterization unless done carefully. The implemented method is a standard direct construction of random unit quaternions suitable for uniform orientation sampling over 3D rotations.

---

## 12. Function-by-function explanation of the code

This section explains what each major part of the script does, how it does it, and why it exists.

### `a_from_phi(phi, particle_volume, n_per_cell)`

**Purpose:**
Convert the target packing fraction into the conventional cubic cell length `a`.

**How:**
Uses

\[
a = \left(\frac{n_{\text{per cell}} V_{\text{particle}}}{\phi}\right)^{1/3}.
\]

**Why:**
This is the central geometric relation that lets the code produce the requested density directly, without any dynamic equilibration or insertion procedure.

---

### `nearest_replication(n_req, n_per_cell)`

**Purpose:**
Choose a supercell replication `(nx, ny, nz)` that yields at least the requested particle count.

**How:**
It first estimates a cubic replication count and then searches nearby triples, preferring:

1. the smallest total particle count above the requested value,
2. and, among ties, the most cube-like replication.

**Why:**
A nearly cubic box is usually more convenient for analysis, visualization, and later simulation. It also avoids highly elongated boxes when the user merely requested a particle count lower bound.

---

### `build_positions(lattice_type, a, nx, ny, nz)`

**Purpose:**
Construct all particle-center coordinates.

**How:**
It loops over all replicated unit cells and all basis positions, converts fractional basis coordinates into Cartesian coordinates, and shifts the whole structure so the periodic box is centered at the origin.

**Why:**
The origin-centered convention is standard and convenient for simulation snapshots.

---

### `verify_phi(n, particle_volume, Lx, Ly, Lz)`

**Purpose:**
Recompute the realized packing fraction from the final box dimensions and particle volume.

**How:**
Uses:

\[
\phi_{\text{actual}} = \frac{nV_{\text{particle}}}{L_xL_yL_z}.
\]

**Why:**
This verifies that the generated configuration is consistent with the intended density formula.

---

### `load_polyhedron_shape(shape_json_path)`

**Purpose:**
Read the polyhedron geometry from disk.

**How:**
It loads the JSON, checks for required keys, converts the vertex list to a NumPy array, validates the reference volume, and returns the name, short name, reference volume, and vertex array.

**Why:**
This isolates shape parsing and makes failure messages precise if the user provides a malformed shape file.

---

### `scale_vertices(base_vertices, scale)`

**Purpose:**
Apply a uniform linear scaling to the reference polyhedron.

**How:**
Multiplies every vertex coordinate by the scalar `scale`.

**Why:**
The shape file stores a reference geometry, not necessarily the actual particle size desired for the simulation.

---

### `circumsphere_radius(vertices)`

**Purpose:**
Compute the maximum distance of any vertex from the particle origin.

**How:**
Takes the Euclidean norm of each vertex and returns the maximum.

**Why:**
This is a useful geometric diagnostic. It is not itself the overlap criterion.

---

### `validate_unit_quaternion(q, field_name)`

**Purpose:**
Ensure a user-supplied quaternion is valid.

**How:**
Checks:

- it has four entries,
- all entries are finite,
- it is not the zero quaternion,
- and then normalizes it.

**Why:**
HOOMD expects valid unit quaternions for particle orientations.

---

### `normalize_orientation_mode(value)`

**Purpose:**
Accept reasonable aliases for the orientation mode and convert them to one canonical form.

**How:**
Maps inputs like `ordered`, `disordered`, and `random` to either `order` or `disorder`.

**Why:**
This makes the interface friendlier without complicating downstream logic.

---

### `random_unit_quaternions(n, rng)`

**Purpose:**
Generate independent random orientations.

**How:**
Uses a direct quaternion-sampling formula from three uniform random numbers per particle.

**Why:**
This gives orientational disorder in a reproducible way when a fixed seed is supplied.

---

### `build_orientations(...)`

**Purpose:**
Construct the final `(N, 4)` orientation array for the GSD frame.

**How:**
- In `order` mode, it tiles one normalized quaternion across all particles.
- In `disorder` mode, it samples one random quaternion per particle.

**Why:**
This isolates all orientation assignment in one place.

---

### `minimum_center_distance(...)`

**Purpose:**
Compute the smallest pairwise center-to-center distance under periodic boundary conditions.

**How:**
Uses a brute-force double loop and minimum-image convention.

**Why:**
This is only a diagnostic. It is **not** the exact overlap criterion for anisotropic polyhedra, because two particles can overlap or avoid overlap depending on orientation even at the same center distance.

---

### `parse_config(input_json_path)`

**Purpose:**
Read and validate the main user input file.

**How:**
It checks:

- file existence,
- JSON syntax,
- presence of required keys,
- valid lattice type,
- positive and finite `phi` and `scale`,
- positive integer particle request,
- valid shape path,
- valid orientation mode,
- required ordered quaternion when needed,
- integer seed,
- optional output filename.

**Why:**
This prevents the script from progressing with inconsistent or ambiguous input.

---

### `main()`

**Purpose:**
Execute the full workflow.

**How:**
It performs the end-to-end pipeline from input parsing to overlap validation.

**Why:**
Keeping the entire workflow in one top-level function makes the code easier to read and easier to extend.

---

## 13. Detailed execution flow of `main()`

This section explains exactly what happens when you run the program.

### Step 1: Read the command-line argument

The script expects one command-line argument:

```bash
python make_lattice_convex_polyhedron_based_on_phi.py input_lattice_configuration.json
```

That argument is the path to the main input JSON.

---

### Step 2: Parse and validate the main input JSON

The code calls `parse_config(...)`, which returns:

- `lattice_type`
- `phi`
- `scale`
- `n_req`
- `shape_json_path`
- `orientation_mode`
- `ordered_orientation`
- `rng_seed`
- `output_gsd`

At this point the script has not yet built anything. It has only validated that the input is internally sensible.

---

### Step 3: Load the reference polyhedron

The code calls `load_polyhedron_shape(shape_json_path)`.

This returns:

- shape name,
- shape short name,
- reference volume,
- reference vertices.

These values define the reference particle in its body frame.

---

### Step 4: Scale the vertices and compute actual particle volume

The code applies:

```python
vertices = scale_vertices(base_vertices, scale)
particle_volume = base_volume * (scale ** 3)
```

This is one of the most important lines in the program. It turns the reference normalized shape into the actual particle used in the system.

---

### Step 5: Determine lattice-specific constants

The script uses the selected lattice type to retrieve:

- the number of particles per conventional cubic unit cell,
- the nearest-neighbor distance factor.

Then it computes:

```python
a = a_from_phi(phi, particle_volume, n_per_cell)
```

So now the code knows the cubic unit-cell length required by the target packing fraction.

---

### Step 6: Choose the supercell size

The code calls:

```python
(nx, ny, nz), n_actual = nearest_replication(n_req, n_per_cell)
```

This resolves the finite-size system that will actually be written.

The box lengths are then:

```python
Lx = nx * a
Ly = ny * a
Lz = nz * a
```

---

### Step 7: Build all particle positions

The code calls:

```python
positions = build_positions(lattice_type, a, nx, ny, nz)
```

This creates every particle center.

The positions are centered so that the box spans roughly `[-L/2, +L/2)` in each direction.

---

### Step 8: Build all particle orientations

The code calls:

```python
orientations = build_orientations(...)
```

This creates an `(N, 4)` array of quaternions.

- ordered mode gives one common quaternion repeated `N` times,
- disordered mode gives `N` independent random unit quaternions.

---

### Step 9: Recompute the realized packing fraction

The script computes:

```python
phi_actual = verify_phi(n_actual, particle_volume, Lx, Ly, Lz)
```

This is mainly a consistency check and reporting quantity.

---

### Step 10: Print the console summary

Before writing anything, the code prints a detailed report including:

- lattice type,
- shape name,
- shape short name,
- shape JSON path,
- input packing fraction,
- reference shape volume,
- linear scale,
- actual particle volume,
- derived unit-cell length,
- nearest-neighbor distance,
- circumsphere radius,
- particles per unit cell,
- requested particle count,
- actual particle count,
- box dimensions,
- verified packing fraction,
- orientation mode,
- ordered quaternion or random seed.

This printed block is meant to make it easy to verify that the requested system and the generated system are consistent.

---

### Step 11: Write the GSD frame

The code creates a `gsd.hoomd.Frame()` and stores:

- particle count,
- positions,
- orientations,
- type IDs,
- type names,
- box lengths,
- frame step.

The frame is then written as a single snapshot.

### Important note

The current script writes the polyhedron geometry to HOOMD during the overlap-check stage, but unless you add the `particles.type_shapes` metadata block, the GSD file itself will not carry the full visualization shape description for OVITO.

A dedicated section below explains how to add that.

---

### Step 12: Reload the GSD in HOOMD and define the particle shape

The code creates a CPU device and a HOOMD simulation object, then does:

```python
mc = hoomd.hpmc.integrate.ConvexPolyhedron()
mc.shape["A"] = dict(vertices=vertices.tolist())
```

This tells HOOMD that particle type `"A"` is a convex polyhedron with the given scaled vertices.

This is the authoritative overlap definition used by the simulation engine.

---

### Step 13: Ask HOOMD whether overlaps exist

The code loads the just-written GSD file as the simulation state and calls:

```python
hoomd_sim.run(0)
```

This does not advance dynamics in the ordinary sense, but it initializes the state and allows the code to query:

```python
mc.overlaps
```

If that value is greater than zero, the script stops with an error.

If that value is zero, the script reports success.

### Why this check is necessary

For spheres, center distance alone can often be enough to reason about overlap. For anisotropic convex polyhedra, it generally is not.

Two polyhedra can overlap or not overlap depending on:

- shape,
- relative orientation,
- face-to-face geometry,
- edge-to-edge geometry,
- vertex-face contacts,
- periodic images.

So the correct final judge is **HOOMD’s actual convex-polyhedron overlap engine**.

---

### Step 14: Optional center-distance diagnostic

For systems with `N <= 5000`, the code computes the minimum pairwise center distance under periodic boundary conditions.

This is only a geometric diagnostic for the particle centers. It can help you understand how tightly packed the lattice points are, but it should never be mistaken for the exact no-overlap test for anisotropic bodies.

---

## 14. Why the overlap check is done after writing the GSD

A natural question is why the code does not simply test overlaps in memory before writing the file.

The answer is practical and conceptual.

Writing the GSD first has advantages:

1. the written file is exactly what HOOMD later reads,
2. the overlap check is performed on the actual persisted state,
3. there is no ambiguity about conversion between internal arrays and stored data,
4. if the check succeeds, the same file is already available for simulation or inspection,
5. if the check fails, you still know exactly which attempted configuration was rejected.

So the write-then-validate sequence is not accidental; it is a useful design choice.

---

## 15. What the output GSD file contains

In its current form, the generated GSD frame contains:

- particle positions,
- particle orientations,
- type IDs,
- type names,
- orthorhombic simulation box,
- frame step.

This is sufficient for HOOMD to read the configuration and for the script to perform the overlap check once the shape is defined again via the HPMC integrator.

### Important distinction

The **overlap definition** is currently supplied in the script through:

```python
mc.shape["A"] = dict(vertices=vertices.tolist())
```

That tells HOOMD how to interpret particle type `"A"` for overlap purposes.

However, a GSD file meant for **visualization of the actual polyhedron shape in OVITO** should also include particle-type shape metadata.

That is a separate concern from the overlap check.

---

## 16. If you want OVITO to show the actual polyhedra

If you open the generated GSD in OVITO and want to see the true polyhedron shape rather than generic point-like particles or default representations, the GSD file should store the particle-type shape metadata.

That means adding this block in the GSD-writing section, immediately after:

```python
frame.particles.types = ["A"]
```

Add:

```python
frame.particles.type_shapes = [
    {
        "type": "ConvexPolyhedron",
        "rounding_radius": 0.0,
        "vertices": vertices.tolist()
    }
]
```

### Why this is separate from `mc.shape`

Because these two pieces of information serve different purposes:

- `mc.shape["A"]` tells **HOOMD HPMC** how to do overlap checks,
- `frame.particles.type_shapes` tells the **GSD file / visualization layer** how to represent the particle shape.

### Recommended practice

For a production-quality workflow, it is best to keep both:

- define the shape in HOOMD for overlap detection,
- store the shape in the GSD for visualization.

---

## 17. Error handling strategy

The script uses explicit exception classes and structured termination messages.

### `InputValidationError`

Used for bad or incomplete user input, such as:

- missing keys,
- invalid lattice name,
- invalid quaternion,
- non-positive scale,
- invalid seed.

### `FileNotFoundError`

Used when the main input file or the shape JSON cannot be found.

### `RuntimeError`

Used when a supposedly valid construction fails an internal consistency check or when HOOMD reports overlaps.

### Why this matters

A geometry-construction script should fail loudly and precisely. Silent fallback behavior can produce misleading simulation inputs that appear valid but are not what the user intended.

---

## 18. What the script does **not** do

It is important not to attribute capabilities to the code that it does not actually have.

This script does **not**:

- run an equilibrated HPMC simulation,
- perform compression,
- relax orientations,
- search for a stable crystal structure,
- discover the best orientation pattern,
- compute free energies,
- optimize the lattice for a given shape,
- support multiple particle types,
- support polydispersity,
- generate dynamic trajectories,
- guarantee that a chosen `phi` is physically realizable for every shape/orientation combination.

It only generates one candidate initial condition and tests it for overlap freedom.

---

## 19. Important physical and geometric limitations

### 19.1 Translational order does not imply orientational feasibility

For anisotropic particles, putting centers on a perfect crystal lattice does not guarantee that any chosen orientation field is valid.

A lattice that is overlap-free in fully aligned orientation may fail badly in random orientation mode at the same packing fraction.

### 19.2 The packing fraction formula does not guarantee realizability

The formula for `a` ensures that the nominal particle volume fraction is correct, but it does not prove that the polyhedra can be placed without overlap at that density.

That is exactly why the HOOMD overlap check is necessary.

### 19.3 Minimum center distance is only diagnostic

For anisotropic particles, center distance alone is not an exact collision criterion.

### 19.4 The chosen lattice may not be physically meaningful for the shape

The script allows SC/BCC/FCC center lattices, but whether a given polyhedron actually forms or favors such structures is a separate physical question.

### 19.5 The current code assumes a single shape and a single type

All particles are assigned type `"A"` and the same scaled polyhedron geometry.

---

## 20. Typical usage examples

### Example 1: Simple cubic, fully aligned cube

```json
{
    "lattice_type": "SC",
    "phi": 0.10,
    "shape_json": "shape_023_Cube_unit_volume_principal_frame.json",
    "scale": 1.0,
    "n_particles_req": 2197,
    "orientation_mode": "order",
    "ordered_orientation": [1.0, 0.0, 0.0, 0.0],
    "output_gsd": "cube_sc_order_phi0p10.gsd"
}
```

This builds a simple-cubic array of identical cubes with all particles sharing the identity orientation.

### Example 2: Simple cubic, random orientations

```json
{
    "lattice_type": "SC",
    "phi": 0.10,
    "shape_json": "shape_023_Cube_unit_volume_principal_frame.json",
    "scale": 1.0,
    "n_particles_req": 2197,
    "orientation_mode": "disorder",
    "rng_seed": 12345,
    "output_gsd": "cube_sc_disorder_phi0p10.gsd"
}
```

This builds the same center lattice, but every cube receives a different random quaternion.

---

## 21. How to run the script

Typical usage from a shell:

```bash
python make_lattice_convex_polyhedron_based_on_phi.py input_lattice_configuration.json
```

You should run it in an environment where the following imports work:

- `numpy`
- `gsd.hoomd`
- `hoomd`

---

## 22. What to inspect after a successful run

After the script finishes successfully, inspect the following:

1. the console report:
   - does the shape name match what you intended?
   - does the scale match what you intended?
   - does `phi_actual` match the target?
   - is the actual particle count acceptable?
2. the GSD file:
   - can HOOMD read it?
   - if you added `type_shapes`, can OVITO render the actual polyhedron?
3. the HOOMD overlap count:
   - it must be zero.
4. the chosen lattice constant:
   - does it make sense compared with the particle size?

---

## 23. Suggestions for future extensions

A natural next set of extensions would be:

- writing `particles.type_shapes` by default,
- adding support for multiple particle types,
- allowing a user-defined basis file,
- allowing systematic orientation patterns within the unit cell,
- adding exact-supercell targeting when possible,
- adding support for non-cubic boxes,
- exporting a small metadata summary file alongside the GSD,
- adding an option to reject configurations unless `n_actual == n_req`,
- adding a dry-run mode that reports the lattice constant and box without writing a file,
- adding visualization helper scripts for OVITO.

---

## 24. Minimal mental model of the whole project

If you want to remember this project in one sentence, it is this:

> The script takes a reference convex polyhedron, scales it, places its centers on an SC/BCC/FCC lattice at a box size chosen from the target packing fraction, assigns ordered or random orientations, writes the result to GSD, and asks HOOMD-blue v4 whether the resulting configuration is overlap free.

That is exactly what is executed, how it is executed, and why each step is there.

