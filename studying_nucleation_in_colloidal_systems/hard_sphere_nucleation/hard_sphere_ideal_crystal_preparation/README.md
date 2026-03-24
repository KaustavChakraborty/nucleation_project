# HOOMD-blue v4 Hard-Sphere Lattice Generator

python3 make_lattice_hard_sphere_based_on_phi.py input_lattice_configuration.json

A packing-fraction-driven lattice generator for **monodisperse hard spheres** that creates ideal **SC**, **BCC**, or **FCC** crystal configurations and writes them to a **GSD** file for use with **HOOMD-blue v4**. The script derives the cubic lattice constant from the target packing fraction and particle diameter, chooses a near-balanced periodic supercell with at least the requested number of particles, writes the configuration to disk, and then validates the structure using both a manual geometric overlap check and a **HOOMD HPMC Sphere integrator** overlap count.

This project is for building **clean initial crystal configurations**, not for performing equilibration, phase selection, relaxation, or structure prediction. It is best viewed as a **geometry-and-I/O utility** for hard-sphere simulations.

---

## What this script does

Given a JSON input file specifying:

- lattice type (`SC`, `BCC`, or `FCC`)
- packing fraction `phi`
- particle diameter `sigma`
- requested particle count
- optionally an output filename

the script:

1. validates the input,
2. computes the cubic unit-cell length `a` from the chosen packing fraction,
3. chooses a supercell replication `(nx, ny, nz)` that gives at least the requested number of particles,
4. generates ideal lattice coordinates,
5. centers the lattice at the **geometric center of the periodic box**,
6. writes a single-frame `.gsd` file,
7. runs a **HOOMD v4 HPMC hard-sphere overlap check**, and
8. optionally runs a brute-force manual pair-distance overlap check for moderate system sizes.

---

## Supported lattices

This version supports only the following conventional cubic lattices:

- **SC**: simple cubic, 1 particle per unit cell
- **BCC**: body-centered cubic, 2 particles per unit cell
- **FCC**: face-centered cubic, 4 particles per unit cell

The basis positions are hard-coded in the script. There is no support in the current version for HCP, diamond, custom user-defined bases, multicomponent crystals, tilted boxes, or non-cubic conventional cells.

---

## Physical model behind the script

The code assumes **identical hard spheres** of diameter `sigma`. The packing fraction is enforced geometrically through the conventional cubic unit cell:

\[
\phi = \frac{N_{\text{cell}} V_{\text{sphere}}}{a^3}
\]

with

\[
V_{\text{sphere}} = \frac{\pi}{6}\sigma^3
\]

so that

\[
a = \left(\frac{N_{\text{cell}}(\pi/6)\sigma^3}{\phi}\right)^{1/3}.
\]

Here:

- \(N_{\text{cell}} = 1\) for SC
- \(N_{\text{cell}} = 2\) for BCC
- \(N_{\text{cell}} = 4\) for FCC

This means the target density is not obtained by compression, Monte Carlo insertion, or energy minimization. It is obtained by **constructing an ideal lattice with the corresponding lattice constant**.

---

## Coordinate convention and origin

The generated particle positions are written in **Cartesian coordinates** and then shifted so that the full periodic simulation box is centered at the origin. The code constructs each position as

\[
x = (i_x + b_x)a - \frac{L_x}{2},\quad
y = (i_y + b_y)a - \frac{L_y}{2},\quad
z = (i_z + b_z)a - \frac{L_z}{2}
\]

where \((b_x,b_y,b_z)\) is a basis coordinate and \((L_x,L_y,L_z)\) are the final box lengths. Therefore, the coordinates stored in the output GSD file are referenced to a **box-centered origin**, not a corner-origin convention such as `[0, L)`.

So yes: in the current implementation, the origin is the **geometric center of the periodic box**. A particle does not necessarily lie exactly at the origin; that depends on the lattice basis and replication counts.

---

## What is written into the output GSD file

The script writes a single `gsd.hoomd.Frame()` containing:

- `particles.N`
- `particles.position`
- `particles.orientation`
- `particles.typeid`
- `particles.types = ["A"]`
- `particles.diameter`
- `configuration.box = [Lx, Ly, Lz, 0, 0, 0]`
- `configuration.step = 0`

This is important for hard-sphere workflows because the script explicitly writes the diameter array to the GSD frame.

---

## Validation steps performed by the script

The script performs multiple checks before and after writing the file.

### 1. Input validation

The helper `parse_config(...)` checks:

- file existence,
- that the path is a file,
- JSON syntax validity,
- presence of required keys,
- valid lattice name,
- numeric convertibility of `phi`, `diameter`, and `n_particles_req`,
- finiteness of `phi` and `diameter`,
- positivity of `phi`, `diameter`, and requested particle count.

### 2. Packing fraction feasibility

The script compares `phi` against the lattice-specific maximum packing fraction:

- SC: \(\pi/6\)
- BCC: \(\pi\sqrt{3}/8\)
- FCC: \(\pi/(3\sqrt{2})\)

If the requested packing fraction is at or above the close-packing limit for that lattice, the script exits.

### 3. Nearest-neighbor no-overlap check

After computing `a`, the script computes the nearest-neighbor distance using a lattice-specific geometric factor and checks that this distance is not smaller than the particle diameter.

### 4. Internal particle-count consistency check

After position generation, the script checks that the number of generated coordinates matches the expected supercell particle count.

### 5. HOOMD v4 HPMC overlap check

After writing the GSD file, the script creates a HOOMD `Simulation`, attaches an HPMC `Sphere` integrator, sets the particle shape diameter, reads the just-written GSD file, runs `sim.run(0)`, and prints the value of `mc.overlaps`. This is a direct HOOMD-side validation of the hard-sphere geometry.

### 6. Manual overlap sanity check

For systems with `N <= 5000`, the script also performs a brute-force pairwise overlap check with periodic minimum-image convention and reports both:

- number of overlapping pairs,
- smallest pair distance found.

This is useful for debugging because it gives geometric detail that HOOMD’s scalar overlap count alone does not provide.

---

## Example input

An example configuration file included in this project is:

```json
{
    "_comment": "Input for make_lattice_phi.py (Version 2 — packing fraction driven)",
    "lattice_type"   : "FCC",
    "phi"            : 0.10,
    "diameter"       : 1.2407,
    "n_particles_req": 2000,
    "output_gsd"     : "lattice_phi.gsd"
}
```

This requests an FCC lattice at packing fraction `0.10`, with sphere diameter `1.2407`, and at least `2000` particles.

---

## Requirements

Typical Python dependencies are:

- Python 3.x
- NumPy
- GSD
- HOOMD-blue v4

Install HOOMD-blue according to your platform and environment. In practice, this script assumes that `import hoomd`, `import gsd.hoomd`, and `import numpy` all work in the same Python environment.

---

## How to run

Typical usage:

```bash
python make_lattice_phi_v4.py input_lattice_configuration.json
```

The script prints a console summary including lattice type, input packing fraction, maximum packing fraction, unit-cell length, nearest-neighbor distance, requested and actual particle count, box dimensions, and the verified packing fraction. It then reports the GSD filename and both overlap checks.

---

## Expected outputs

After a successful run, you should expect:

1. A `.gsd` file containing one frame with your generated lattice.
2. A console summary of the constructed system.
3. A HOOMD overlap count.
4. For moderate system sizes, a manual pairwise overlap diagnostic.

---

## Assumptions you must not forget

This section is the most important one to read before using the code blindly.

### 1. The particles are monodisperse spheres

The script supports only **one particle type**, `"A"`, with one diameter assigned to all particles. It is not designed for mixtures, alloys, bidisperse spheres, polydispersity, or partial site occupancy.

### 2. The generated structure is an ideal crystal

The script constructs mathematically exact lattice points from a fixed basis and supercell replication. It does not add thermal noise, defects, vacancies, substitutions, strain, relaxation, grain boundaries, or disorder.

### 3. The unit cell is cubic

The entire density formula is built on a **cubic conventional cell** of edge length `a`. This is why the code can use one scalar lattice constant for all three axes at the cell level.

### 4. The box is periodic

Both the GSD box and the manual overlap check assume periodic boundaries. The manual overlap check explicitly uses the minimum-image convention.

### 5. The packing fraction is achieved geometrically, not dynamically

This is not a molecular assembly algorithm. It is not a packing solver. It is not a structure relaxation code. It is a formula-based lattice builder.

### 6. The requested particle count is treated as a lower bound

The supercell finder chooses a replication that gives **at least** `n_particles_req`. Therefore, the final particle count may be larger than requested. The code prints whether the count was adjusted.

### 7. “No overlap” does not mean “thermodynamically stable”

The code checks geometric feasibility, not free-energy stability. A valid SC, BCC, or FCC geometry does not imply that it is the stable phase of your interaction model.

---

## Failure modes and where blind use can mislead you

### 1. Wrong physical interpretation of `phi`

The script’s `phi` is the packing fraction of ideal hard spheres on a perfect lattice. If your later simulation uses soft pair potentials, effective diameters, bonded networks, or anisotropic particles, the geometric meaning of `phi` may no longer match the physics you think you are modeling.

### 2. Asking for an unsupported structure

If you need HCP, diamond, distorted cubic cells, multicomponent crystals, or a user-defined basis, this script is not the right tool without modification.

### 3. Demanding an exact `N`

Because the code picks a supercell with `N >= n_req`, you may not get exactly the particle count you asked for. If exact `N` is a hard requirement, you should check the chosen replication carefully or modify the supercell selection logic.

### 4. Large-`N` manual overlap check cost

The brute-force manual overlap routine scales as \(O(N^2)\), so the script skips it for `N > 5000`. At large system sizes, you therefore rely on the HOOMD overlap count and the geometric construction logic, not on the explicit pairwise minimum-distance report.

### 5. Misreading the coordinate origin

Some workflows expect coordinates in a corner-origin representation. This script writes centered coordinates. If your post-processing or visualization tool assumes positions should lie in `[0, L)`, remember that the GSD file instead contains positions centered around zero.

### 6. Using this as a random initial condition generator

This code does not produce a randomized fluid-like state. It produces a perfect crystal. If you want a disordered initial configuration, a melting path, or a compressed random packing, you need a different workflow.

### 7. Filesystem and environment failures

Even with good input, the script can still fail if:

- HOOMD-blue is not installed in the active environment,
- `gsd.hoomd` is missing,
- the output directory is not writable,
- the input JSON path is wrong,
- the Python environment mixes incompatible package versions.

---

## Known caveats in the current `v4` implementation

These points are specific to the current uploaded version and are worth knowing before treating the script as production-final.

### 1. The JSON file is parsed twice

The script first validates input through `parse_config(...)`, but then inside `main()` it opens the JSON file again and reassigns `lattice_type`, `phi`, `diameter`, and `n_req`. This means some logic is duplicated unnecessarily.

### 2. `output_gsd` from JSON is effectively ignored in the current version

Even though `parse_config(...)` returns `output_gsd_from_json`, the later re-parse in `main()` sets:

```python
output_gsd = None  # will be generated automatically
```

and the final output filename is chosen from that path instead. So the example JSON field

```json
"output_gsd": "lattice_phi.gsd"
```

is not actually honored by the present implementation. The auto-generated filename is used instead.

### 3. There is a redundant overwritten box assignment

The script sets

```python
Lx = Ly = Lz = nx * a
```

and then immediately overwrites all three with the proper separate assignments:

```python
Lx = nx * a
Ly = ny * a
Lz = nz * a
```

So the first line is redundant and has no final effect.

### 4. High-density warning is generic

The current code warns when `phi > 0.74`, which is not a very lattice-aware threshold because SC and BCC close-pack well below that. The actual correctness check comes from comparison with the lattice-specific `phi_max`.

### 5. Error handling is partly modernized and partly legacy-style

The script defines `InputValidationError` and catches several exceptions at the end, but it also still uses `sys.exit(...)` directly in some validation paths inside `main()`. So the code is partway between a cleaner exception-driven design and an older immediate-exit style.

---

## What this project is good for

This script is well-suited for:

- preparing ideal SC/BCC/FCC starting configurations,
- testing HPMC hard-sphere workflows,
- generating reproducible crystal inputs,
- checking packing-fraction-dependent lattice spacing,
- producing GSD files with particle diameter explicitly stored,
- teaching or debugging lattice geometry under periodic boundary conditions.

---

## What this project is not for

This script is not intended for:

- random packings,
- insertion-based dense packing generation,
- fluid initialization,
- energy minimization,
- defect engineering,
- multi-species systems,
- anisotropic particles,
- free surfaces or vacuum slabs,
- thermodynamic phase prediction,
- user-defined arbitrary crystal structures.

---

## Interpreting the HOOMD overlap result

The HOOMD overlap check is performed using an HPMC `Sphere` integrator and the shape is defined with the same input diameter used in the lattice construction. The scalar `mc.overlaps` is then printed. A value of zero indicates that HOOMD sees no overlapping hard-sphere pairs in the written GSD configuration.

This is valuable because it checks the state using the same HOOMD machinery you are likely to use later in simulation, not just the script’s own manual geometry logic.

---

## Practical checklist before using the script blindly

Before running a serious job, confirm all of the following:

- your intended structure is really SC, BCC, or FCC,
- all particles truly should have the same diameter,
- a perfect crystal is the initial state you want,
- periodic boundaries are appropriate,
- an adjusted particle count is acceptable,
- your target packing fraction is meaningful for your intended physical model,
- centered box coordinates will not break downstream analysis,
- you do not rely on the JSON `output_gsd` field in the current `v4` implementation.

---

## Suggested future improvements

Natural next improvements for this project would be:

- remove the second JSON parse inside `main()`,
- honor `output_gsd` from the input file,
- remove the redundant `Lx = Ly = Lz = nx * a` assignment,
- replace remaining `sys.exit(...)` calls with consistent exception raising,
- move helper functions such as `choose_output_filename(...)` outside `main()`,
- add optional exact-`N` mode,
- add support for custom basis files,
- add optional coordinate export in corner-origin convention,
- add optional random perturbation for thermalized starting states,
- add a faster neighbor-list-based manual overlap check for larger systems.

---

## Author and tested stack

The current script header identifies the author as **Kaustav Chakraborty** and notes testing with **HOOMD-blue 4.x**, **GSD 3.x**, and **NumPy 1.x / 2.x**.

---

## Final takeaway

This project is best thought of as a **hard-sphere crystal initializer for HOOMD-blue**, not as a general packing engine and not as a stability oracle. It is reliable for generating ideal SC/BCC/FCC periodic lattices with a target packing fraction and explicit diameter storage in the GSD, but its assumptions are strong and should be respected.
