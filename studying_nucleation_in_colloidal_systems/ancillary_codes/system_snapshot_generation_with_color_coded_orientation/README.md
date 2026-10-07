# Orientational-Cluster Snapshot Renderer for HOOMD GSD Trajectories

## Extremely Detailed User and Developer Guide

This project renders a single frame from a HOOMD-schema GSD trajectory as a collection of convex polyhedral particles, colors the particles according to symmetry-reduced orientational clusters, and optionally creates a periodic-boundary-aware zoomed view of a selected spatial region.

The current version is designed around the following working files:

```text
render_sys_snapshot_from_gsd_orientational_color.py
param_file.json
shape_file.json
trajectory.gsd
```

For the current Elongated Pentagonal Dipyramid (EPD/J16) example, the corresponding files are:

```text
render_sys_snapshot_from_gsd_orientational_color.py

param_file(20261007-121138).json

shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json

hpmc_hard_Elongated_Pentagonal_Dipyramid_vol_1p00_4096_NPT_P150_0_traj.gsd
```

The program is intentionally standalone with respect to the older project modules. It does **not** require project-specific modules such as `header.py`, `color_particles.py`, `ref_frame_calc.py`, `get_invQ.py`, or `write_system_pos.py`.

The renderer itself performs all required stages:

1. reads the parameter file;
2. reads the convex-polyhedron vertex set;
3. validates the shape;
4. automatically determines the proper rotational symmetry group of the particle;
5. reads one requested GSD frame;
6. normalizes particle quaternions;
7. computes symmetry-reduced pairwise orientational separations;
8. discovers orientational clusters;
9. retains the statistically important orientation references;
10. assigns every particle to its nearest retained orientation reference;
11. maps each retained orientational state to a color;
12. renders the complete simulation frame;
13. selects a periodic spatial subsection;
14. renders a zoomed image of that subsection;
15. saves cluster assignments, symmetry information, metadata, and publication-quality PNG images.

---

# 1. What the program is intended to visualize

Suppose a crystal contains anisotropic particles. The particle centers may form a translationally ordered crystal, while the particle orientations can occupy several discrete or approximately discrete orientational states.

The purpose of this program is to produce an image in which:

- particles belonging to the same orientational state have the same color;
- different orientational states have different colors;
- particle symmetry is correctly taken into account;
- the original polyhedral shape is rendered instead of replacing particles by spheres;
- one can inspect either the entire simulation box or a smaller spatial region.

For example, if the system contains five important orientational states and the parameter file contains:

```json
"cmap_original": [
    "#0080ff",
    "#ff8000",
    "#5b0aa2",
    "#db023c",
    "#00ffff"
]
```

then the first retained orientation reference is blue, the second orange, the third purple, the fourth red, and the fifth cyan.

The color is therefore not based on particle position. It represents the particle's **symmetry-reduced orientation**.

---

# 2. Scientific definition of orientation used by the code

A particle orientation stored in the GSD trajectory is represented by a unit quaternion:

```text
[qw, qx, qy, qz]
```

where `qw` is the scalar component.

Two raw quaternions can correspond to physically equivalent particle orientations for two different reasons.

First, unit quaternions double-cover SO(3):

```text
q  and  -q
```

represent the same three-dimensional rotation.

Second, an anisotropic particle may possess nontrivial rotational symmetry. For example, rotating a particle around one of its body symmetry axes may map the entire particle exactly onto itself. Such two orientations should not be classified as different orientational states.

The code therefore:

1. determines all proper rotational symmetries of the supplied vertex set;
2. converts those rotations into quaternions;
3. supplies both `q` and `-q` for each physical symmetry to `freud.environment.AngularSeparationGlobal`;
4. uses the minimum symmetry-equivalent angular difference as the physically relevant orientation separation.

This is a crucial feature. A raw quaternion difference without particle-symmetry reduction would generally overcount physically identical states.

---

# 3. Project directory layout

The simplest recommended directory is:

```text
snapshot_project/
│
├── render_sys_snapshot_from_gsd_orientational_color.py
├── param_file.json
├── shape.json
├── trajectory.gsd
│
└── outputs_orientational_snapshot/
```

The output directory does not need to exist before running the program. The code creates it automatically.

For the current EPD calculation, a typical directory may look like:

```text
snapshot_generation/
│
├── render_sys_snapshot_from_gsd_orientational_color.py
├── param_file(20261007-121138).json
├── shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json
├── hpmc_hard_Elongated_Pentagonal_Dipyramid_vol_1p00_4096_NPT_P150_0_traj.gsd
│
└── outputs_orientational_snapshot/
```

All relative paths in the parameter JSON are interpreted relative to the directory containing the parameter file.

This means that if the parameter file contains:

```json
"gsd_file": "trajectory.gsd"
```

the program searches for `trajectory.gsd` beside the parameter file.

This is more robust than depending on the shell's current working directory.

---

# 4. Required software

The program uses:

- Python
- NumPy
- SciPy
- GSD
- freud
- matplotlib
- Fresnel
- Plato

The environment recommended in the source code is:

```bash
mamba create -n orient_render -c conda-forge \
    python=3.10 numpy scipy matplotlib gsd freud fresnel

mamba activate orient_render

pip install plato-draw
```

If an existing Python environment already contains compatible versions, it may also work. However, the safest reproducible environment is the one above.

## Check the installation

Run:

```bash
python -c "import numpy, scipy, gsd, freud, matplotlib, fresnel; import plato.draw.fresnel; print('All imports OK')"
```

If successful:

```text
All imports OK
```

should appear.

---

# 5. Running the program

The program accepts exactly one command-line argument: the JSON parameter file.

Example:

```bash
python render_sys_snapshot_from_gsd_orientational_color.py param_file.json
```

Using the current filename:

```bash
python3.8 render_sys_snapshot_from_gsd_orientational_color.py param_file\(20261007-121138\).json
```

or, more conveniently, rename the file:

```bash
cp "param_file(20261007-121138).json" param_file.json
```

and run:

```bash
python3.8 render_sys_snapshot_from_gsd_orientational_color.py param_file.json
```

---

# 6. IMPORTANT: use strict JSON, not JSONC

The parameter reader uses Python's standard:

```python
json.load(...)
```

Therefore the runtime parameter file must be valid **strict JSON**.

Do not put comments such as:

```jsonc
// this is a comment
```

inside the actual file used for execution.

Do not use:

```jsonc
"frame_index": -1, // last frame
```

because that is JSONC rather than JSON.

The valid strict-JSON equivalent is:

```json
"frame_index": -1
```

Also avoid trailing commas.

Incorrect:

```json
{
    "frame_index": -1,
}
```

Correct:

```json
{
    "frame_index": -1
}
```

JSON booleans and null values must also be lowercase:

```json
true
false
null
```

not:

```text
True
False
None
```

A commented `.jsonc` file can be kept as human-readable documentation, but it must not be passed to the current program unless the code is modified to parse JSONC.

---

# 7. Current complete parameter file

The current EPD configuration is:

```json
{
    "gsd_file": "hpmc_hard_Elongated_Pentagonal_Dipyramid_vol_1p00_4096_NPT_P150_0_traj.gsd",
    "shape_file": "shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json",
    "frame_index": -1,
    "orientation_angle_tol": 45.0,
    "cluster_size_cutoff": 200,
    "cmap_original": [
        "#0080ff",
        "#ff8000",
        "#5b0aa2",
        "#db023c",
        "#00ffff"
    ],
    "tolerance_for_inv_quat_of_body_calc": 2,
    "pairwise_block_size": 256,
    "pairwise_matrix_ram_limit_mb": 512,
    "output_dir": "outputs_orientational_snapshot",
    "output_prefix": "EPD",
    "render": {
        "camera_euler_zyx_deg": [
            0.0,
            10.0,
            10.0
        ],
        "full_scene_size": null,
        "full_padding_factor": 1.05,
        "full_pixel_scale": 100,
        "zoom_scene_size": null,
        "zoom_padding_factor": 1.32,
        "zoom_pixel_scale": 180,
        "outline": 0.01,
        "roughness": 0.15,
        "specular": 0.8,
        "spec_trans": 0.0,
        "ambient_light": 1.5,
        "directional_light": [
            -1.5,
            0.0,
            0.0
        ],
        "antialiasing": 1.0,
        "pathtrace_samples": 128,
        "show_box": true,
        "show_zoom_box": false,
        "box_width": 0.04,
        "box_color_rgba": [
            0.1,
            0.1,
            0.1,
            1.0
        ]
    },
    "zoom": {
        "enabled": true,
        "center_fractional": [
            0.0,
            0.0,
            0.0
        ],
        "center_particle_index": null,
        "size_fractional": [
            0.3,
            0.3,
            0.3
        ],
        "cluster_ids": null
    }
}
```

The rest of this README explains every parameter.

---

# 8. Top-level parameter reference

| Parameter | Current value | Scientific effect? | Typical reason to change |
|---|---:|---|---|
| `gsd_file` | trajectory filename | Input selection | Use another trajectory |
| `shape_file` | EPD JSON | Input geometry/symmetry | Use another particle |
| `frame_index` | `-1` | Selects data frame | Render another time/frame |
| `orientation_angle_tol` | `45.0` | **Yes** | Change definition of same orientation cluster |
| `cluster_size_cutoff` | `200` | **Yes** | Change which orientation populations are retained |
| `cmap_original` | 5 colors | Final state representation | Change colors / allow more retained states |
| `tolerance_for_inv_quat_of_body_calc` | `2` | **Yes, symmetry detection** | Adjust geometric symmetry matching |
| `pairwise_block_size` | `256` | No intended scientific change | Tune speed / temporary memory |
| `pairwise_matrix_ram_limit_mb` | `512` | No intended scientific change | Choose RAM versus disk-backed matrix |
| `output_dir` | output folder | No | Organize output |
| `output_prefix` | `EPD` | No | Change output filenames |
| `render` | dictionary | Mostly visual only | Camera, quality, lighting, box, padding |
| `zoom` | dictionary | Spatial visualization | Choose region to display |

---

# 9. `gsd_file`

Example:

```json
"gsd_file": "hpmc_hard_Elongated_Pentagonal_Dipyramid_vol_1p00_4096_NPT_P150_0_traj.gsd"
```

This specifies the HOOMD-schema GSD trajectory.

For another system:

```json
"gsd_file": "my_new_system.gsd"
```

The GSD frame must provide:

```python
snap.particles.position
snap.particles.orientation
snap.configuration.box
```

The code expects:

```text
positions    -> shape (N, 3)
orientations -> shape (N, 4)
box          -> [Lx, Ly, Lz, xy, xz, yz]
```

## Important limitation

The current code assumes **one common convex-polyhedron shape for all particles**.

It does not currently support a mixture such as:

```text
particle type A = cube
particle type B = octahedron
particle type C = sphere
```

with different vertex sets in the same frame.

Such a system would require code modification.

---

# 10. `shape_file`

Current example:

```json
"shape_file": "shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json"
```

The supplied EPD file stores the particle vertices as an `N x 3` array under:

```text
8_vertices
```

The reader is intentionally more general than this particular key.

It recursively searches the JSON for arrays that look like:

```text
N rows x 3 coordinates
```

and gives strong preference to fields whose names contain text such as:

```text
vert
vertices
vertex
```

Therefore these are all reasonable layouts:

```json
{
    "vertices": [
        [x1, y1, z1],
        [x2, y2, z2]
    ]
}
```

or:

```json
{
    "shape": {
        "body_vertices": [
            [x1, y1, z1],
            [x2, y2, z2]
        ]
    }
}
```

## Shape requirements

The shape must:

1. contain at least four vertices;
2. contain finite numerical coordinates;
3. span all three dimensions;
4. have positive convex-hull volume;
5. represent a convex polyhedron;
6. correspond to the same body-frame geometry whose orientation is stored in the GSD.

## Very important body-origin consideration

The program computes:

```python
original_center = np.mean(vertices, axis=0)
centered = vertices - original_center
```

and later renders with these centered vertices.

Thus the effective rendering/symmetry origin is the **arithmetic mean of the supplied vertices**.

For the supplied EPD file this is appropriate because the vertices are already stored in a centered principal frame.

For an arbitrary new particle, check that the simulation's body origin is consistent with this centering convention.

If the HOOMD particle position corresponds to a body origin that is not the vertex-average center, automatic recentering can move the rendered geometry relative to the physical reference point used in the original simulation.

For best results, use a body-frame shape file already centered consistently with the simulation particle origin.

---

# 11. `frame_index`

Current:

```json
"frame_index": -1
```

This uses normal Python indexing.

Examples:

```text
 0  = first frame
 1  = second frame
 2  = third frame
-1  = last frame
-2  = second-last frame
```

To render frame 201:

```json
"frame_index": 201
```

The output filename uses the resolved nonnegative index.

For frame 201:

```text
EPD_frame_000201_full.png
EPD_frame_000201_zoom.png
```

## Common mistake

If the trajectory has 202 frames, the valid positive indices are:

```text
0 ... 201
```

not:

```text
1 ... 202
```

---

# 12. `orientation_angle_tol`

Current:

```json
"orientation_angle_tol": 45.0
```

This is one of the most important scientific parameters.

Units:

```text
degrees
```

Allowed range:

```text
0 < orientation_angle_tol <= 180
```

It controls the provisional orientational-cluster discovery.

The algorithm starts from the first currently unassigned particle and uses it as a seed.

A candidate particle must first be within the tolerance of the seed.

Then it is admitted only if it is within the same tolerance of **every particle already accepted into the cluster**.

Schematically:

```text
candidate joins cluster
        |
        v
angle(candidate, member_1) <= tolerance
AND
angle(candidate, member_2) <= tolerance
AND
angle(candidate, member_3) <= tolerance
...
```

This is similar in spirit to a greedy complete-link criterion.

## What happens if you decrease it?

Example:

```json
"orientation_angle_tol": 20.0
```

Result:

- clusters become tighter;
- one broad orientational state may split into several provisional groups;
- more provisional clusters may appear;
- cluster populations generally decrease;
- more reference colors may be needed.

## What happens if you increase it?

Example:

```json
"orientation_angle_tol": 60.0
```

Result:

- more orientation variation is accepted within one provisional cluster;
- nearby states can merge;
- fewer provisional clusters may appear;
- populations of major groups can increase.

## This parameter changes the scientific classification

Do not treat it as a purely graphical setting.

If comparing different systems, use the same value only when that value has the same physical interpretation for the shapes being compared.

---

# 13. The clustering is deterministic but order-dependent

The program scans particles in particle-index order.

The first unassigned particle becomes a seed.

Therefore, for a fixed GSD frame and fixed particle ordering, the result is deterministic.

However, the clustering rule is greedy. If the same orientations were stored in a different particle-index order, provisional group boundaries could potentially differ.

This behavior is intentional because the current implementation preserves the original project's clustering logic.

It is not equivalent to:

- k-means;
- DBSCAN;
- hierarchical complete-link clustering;
- spectral clustering;
- a globally optimized partition.

If a future scientific analysis requires an order-independent clustering definition, the clustering function itself must be replaced rather than merely changing a parameter.

---

# 14. `cluster_size_cutoff`

Current:

```json
"cluster_size_cutoff": 200
```

This parameter decides which provisional groups are important enough to become retained reference orientations.

Suppose provisional group populations are:

```text
1250
1180
960
410
185
72
39
```

with:

```json
"cluster_size_cutoff": 200
```

then the groups:

```text
1250
1180
960
410
```

survive as candidate retained orientation states.

Groups:

```text
185
72
39
```

do not become independent references.

## Important: discarded provisional groups are not removed from the image

After reference selection, the code assigns **every particle** to whichever retained orientation reference has the smallest symmetry-reduced angular separation.

Therefore:

```text
small provisional group
    != particle deletion
```

Instead:

```text
small provisional group
    -> nearest surviving orientation reference
    -> receives that reference color
```

This distinction is essential when interpreting the final image.

---

# 15. Why a final angle can exceed `orientation_angle_tol`

The original tolerance controls **provisional discovery**.

The final coloring is:

```text
assign every particle to nearest retained reference
```

even if a small provisional group was discarded.

Consequently the output CSV can contain a particle with:

```text
angle_to_reference_deg > orientation_angle_tol
```

This does not necessarily indicate a bug.

It means the particle did not remain part of a retained independent state and was ultimately assigned to the closest available reference.

The cluster summary therefore includes:

```text
mean_angle_to_reference_deg
max_angle_to_reference_deg
```

which should be inspected when judging whether retained references adequately represent the entire orientation distribution.

---

# 16. `cmap_original`

Current:

```json
"cmap_original": [
    "#0080ff",
    "#ff8000",
    "#5b0aa2",
    "#db023c",
    "#00ffff"
]
```

The order matters.

Cluster ID 0 receives:

```text
#0080ff
```

Cluster ID 1 receives:

```text
#ff8000
```

and so on.

Clusters are sorted primarily by provisional population before retention.

Thus the largest discovered retained cluster receives the first color.

## Add more colors

If your system has eight meaningful orientations, provide at least eight colors:

```json
"cmap_original": [
    "#0080ff",
    "#ff8000",
    "#5b0aa2",
    "#db023c",
    "#00ffff",
    "#228833",
    "#ccbb44",
    "#aa3377"
]
```

## Critical interaction between number of colors and number of clusters

If ten provisional clusters pass `cluster_size_cutoff` but only five colors are supplied, the program retains only the five largest as reference states.

Every particle is still colored, but only using those five retained references.

Therefore the palette length is not merely cosmetic. It can place an upper bound on the number of retained orientation states.

If you expect up to twelve distinct states, provide at least twelve colors.

---

# 17. Particle-symmetry detection

The program derives the proper rotational symmetry group directly from the shape vertices.

This is done before orientational clustering.

The broad sequence is:

```text
shape vertices
     |
     v
center vertices
     |
     v
choose well-conditioned source triple
     |
     v
generate candidate target triples
     |
     v
compare Gram matrices
     |
     v
build candidate linear transformation
     |
     v
reject improper transformations
     |
     v
SVD projection to nearest proper rotation
     |
     v
Hungarian one-to-one vertex assignment
     |
     v
all-vertex residual test
     |
     v
full-set rotational refinement
     |
     v
deduplicate by vertex permutation
     |
     v
group-closure validation
     |
     v
proper-rotation quaternions
```

Only proper rotations are used.

Mirrors, inversion, and other improper O(3) operations are not treated as quaternion orientation equivalences.

---

# 18. `tolerance_for_inv_quat_of_body_calc`

Current:

```json
"tolerance_for_inv_quat_of_body_calc": 2
```

The historical interpretation preserved by the code is:

```text
p = tolerance_for_inv_quat_of_body_calc
matching tolerance = 10^(-p)
```

Therefore:

```text
p = 1 -> tolerance = 0.1
p = 2 -> tolerance = 0.01
p = 3 -> tolerance = 0.001
p = 4 -> tolerance = 0.0001
```

Current:

```text
p = 2
=> matching tolerance = 0.01
```

in shape-coordinate units.

## Larger p means stricter matching

This is easy to misunderstand.

Changing:

```json
"tolerance_for_inv_quat_of_body_calc": 2
```

to:

```json
"tolerance_for_inv_quat_of_body_calc": 3
```

makes the allowed geometric mismatch **smaller**, not larger.

## When to tighten it

Increase `p` when:

- false symmetry operations seem to be accepted;
- the shape is intentionally slightly asymmetric;
- detected group order is larger than physically expected.

## When to loosen it

Decrease `p` when:

- a mathematically known symmetry is not detected because vertex coordinates contain numerical noise;
- expected equivalent rotations are missing;
- the shape coordinates were generated with modest numerical precision.

## Prefer explicit tolerance for differently scaled shapes

The code also supports:

```json
"symmetry_vertex_tolerance": 0.001
```

If this key is present, it takes precedence over:

```json
"tolerance_for_inv_quat_of_body_calc"
```

This is useful for arbitrary systems because the tolerance is an **absolute coordinate distance**.

A tolerance of `0.01` may be appropriate for a unit-sized particle but inappropriate for a particle whose coordinates are of order 100.

For shapes with different coordinate scales, using:

```json
"symmetry_vertex_tolerance": ...
```

is often clearer.

---

# 19. How to validate symmetry detection for a new shape

Whenever changing the shape file, inspect the terminal output.

The program prints information such as:

```text
Particle rotational symmetry
----------------------------
Matching tolerance: ...
Source triple: ...
Ordered target triples tested: ...
Gram-compatible target triples: ...
Distinct proper rotations: ...
Equivalent quaternions supplied to freud (q and -q): ...
Maximum accepted vertex residual: ...
```

Ask:

1. Is the number of proper rotations physically sensible?
2. Is identity present?
3. Does the group closure test pass?
4. Is the maximum residual substantially below the chosen tolerance?
5. Does changing the tolerance slightly leave the group order stable?

For a new system, do not blindly trust a render before checking these diagnostics.

---

# 20. `pairwise_block_size`

Current:

```json
"pairwise_block_size": 256
```

The pairwise orientational calculation requires an `N x N` matrix.

Rather than asking freud to produce all rows at once, the code computes them in blocks.

For example:

```text
N = 4096
block size = 256
```

means approximately:

```text
rows 0-255
rows 256-511
rows 512-767
...
```

Each block is compared with all `N` reference orientations.

## Does this change the scientific result?

It is intended as a performance/memory control, not a clustering definition.

## Increase block size when

- sufficient RAM is available;
- freud call overhead is significant;
- you want faster throughput.

Examples:

```json
"pairwise_block_size": 512
```

or:

```json
"pairwise_block_size": 1024
```

## Decrease block size when

- the program runs out of memory during the freud calculation;
- the machine has limited RAM;
- a very large particle count is used.

Examples:

```json
"pairwise_block_size": 128
```

or:

```json
"pairwise_block_size": 64
```

A practical starting range is:

```text
128-512
```

for moderately sized systems.

---

# 21. `pairwise_matrix_ram_limit_mb`

Current:

```json
"pairwise_matrix_ram_limit_mb": 512
```

The final pairwise matrix is stored as `float32`.

Its approximate memory requirement is:

```text
memory = N^2 x 4 bytes
```

or:

```text
memory_MiB = N^2 x 4 / 1024^2
```

Examples:

| N particles | Approximate float32 pairwise matrix |
|---:|---:|
| 1,000 | 3.8 MiB |
| 4,096 | 64 MiB |
| 10,000 | 381 MiB |
| 20,000 | 1.49 GiB |
| 40,000 | 5.96 GiB |

For the current 4096-particle system:

```text
~64 MiB
```

so with:

```json
"pairwise_matrix_ram_limit_mb": 512
```

the matrix remains in RAM.

If the estimated size exceeds the configured limit, the program creates a temporary disk-backed file:

```text
.pairwise_orientation_angles_float32.dat
```

using NumPy `memmap`.

After successful or failed processing through the protected cleanup block, the code attempts to delete the temporary file.

## Raise the RAM limit when

- the machine has plenty of unused RAM;
- disk is slow;
- you want to avoid memmap I/O.

Example:

```json
"pairwise_matrix_ram_limit_mb": 4096
```

## Lower the RAM limit when

- several jobs run on the same node;
- memory is constrained;
- you prefer disk usage over RAM pressure.

Example:

```json
"pairwise_matrix_ram_limit_mb": 128
```

## Important

Changing this parameter should not be used to alter the scientific result. It changes storage policy.

---

# 22. Scaling warning for very large systems

The pairwise matrix scales as:

```text
O(N^2)
```

in memory/storage.

The pairwise angular calculation also performs a large amount of work proportional to the number of orientation pairs.

The greedy complete-link-like clustering repeatedly examines entries from this matrix.

Therefore this code is appropriate for systems such as a few thousand particles, but extremely large systems can become expensive.

For very large trajectories, possible future strategies would include:

- particle subsampling;
- clustering based on representative orientation candidates;
- approximate nearest-neighbor orientation search;
- streaming cluster discovery;
- avoiding construction of the complete `N x N` matrix.

Those are algorithmic changes and are not controlled by the existing parameter file.

---

# 23. `output_dir`

Current:

```json
"output_dir": "outputs_orientational_snapshot"
```

This controls where output files are written.

Example:

```json
"output_dir": "images/frame201"
```

Relative output paths are interpreted relative to the parameter file.

The directory is created automatically.

---

# 24. `output_prefix`

Current:

```json
"output_prefix": "EPD"
```

This is used in filenames.

For frame 201:

```text
EPD_frame_000201_full.png
EPD_frame_000201_zoom.png
EPD_frame_000201_particle_clusters.csv
EPD_frame_000201_cluster_summary.csv
EPD_frame_000201_metadata.json
EPD_proper_rotational_symmetries.csv
```

For another particle:

```json
"output_prefix": "TruncatedCube"
```

would produce:

```text
TruncatedCube_frame_000201_full.png
...
```

This setting has no scientific effect.

---

# 25. Rendering parameter overview

The `render` dictionary currently contains:

```json
"render": {
    "camera_euler_zyx_deg": [0.0, 10.0, 10.0],
    "full_scene_size": null,
    "full_padding_factor": 1.05,
    "full_pixel_scale": 100,
    "zoom_scene_size": null,
    "zoom_padding_factor": 1.32,
    "zoom_pixel_scale": 180,
    "outline": 0.01,
    "roughness": 0.15,
    "specular": 0.8,
    "spec_trans": 0.0,
    "ambient_light": 1.5,
    "directional_light": [-1.5, 0.0, 0.0],
    "antialiasing": 1.0,
    "pathtrace_samples": 128,
    "show_box": true,
    "show_zoom_box": false,
    "box_width": 0.04,
    "box_color_rgba": [0.1, 0.1, 0.1, 1.0]
}
```

These mostly change appearance, framing, or image quality.

They do not alter the orientation matrix or cluster discovery.

---

# 26. `camera_euler_zyx_deg`

Current:

```json
"camera_euler_zyx_deg": [0.0, 10.0, 10.0]
```

The code uses:

```python
Rotation.from_euler("zyx", angles, degrees=True)
```

These values determine the viewing rotation.

Try:

```json
"camera_euler_zyx_deg": [0.0, 0.0, 0.0]
```

for an unrotated view.

Try:

```json
"camera_euler_zyx_deg": [30.0, 20.0, 10.0]
```

for a more tilted 3D view.

## If particles overlap too strongly in projection

Change the camera angle.

For example:

```json
"camera_euler_zyx_deg": [20.0, 20.0, 20.0]
```

or:

```json
"camera_euler_zyx_deg": [45.0, 15.0, 25.0]
```

This only changes the visualization.

---

# 27. `full_scene_size`

Current:

```json
"full_scene_size": null
```

`null` means automatic framing.

The program:

1. constructs the eight simulation-box corners;
2. rotates those corners according to the camera;
3. projects them into the viewing plane;
4. measures the projected width and height;
5. multiplies them by `full_padding_factor`.

This is usually the best choice.

## Manual size

A scalar:

```json
"full_scene_size": 40
```

forces a square scene:

```text
40 x 40
```

A pair:

```json
"full_scene_size": [40, 30]
```

sets explicit width and height.

## Important interaction

If `full_scene_size` is not `null`, automatic box-size fitting is bypassed.

Therefore changing:

```json
"full_padding_factor"
```

will no longer affect scene size.

Use either:

```text
automatic scene size + padding
```

or:

```text
explicit scene size
```

with this in mind.

---

# 28. `full_padding_factor`

Current:

```json
"full_padding_factor": 1.05
```

Only relevant when:

```json
"full_scene_size": null
```

This multiplies the automatically calculated projected box extent.

Examples:

```text
1.00 -> almost exact fit
1.05 -> 5% margin
1.10 -> more margin
1.25 -> much more white space
```

If full-system particles near the image edge are cropped:

```json
"full_padding_factor": 1.10
```

or:

```json
"full_padding_factor": 1.15
```

If there is too much empty margin:

```json
"full_padding_factor": 1.02
```

---

# 29. `full_pixel_scale`

Current:

```json
"full_pixel_scale": 100
```

This controls rasterization density for the full-system scene.

Increasing it generally produces a larger/higher-resolution PNG at the same scene dimensions.

Examples:

```json
"full_pixel_scale": 150
```

or publication-quality:

```json
"full_pixel_scale": 200
```

Tradeoff:

```text
higher pixel_scale
    -> more pixels
    -> larger file
    -> more rendering cost
```

It does not change physical particle positions or cluster assignment.

---

# 30. `zoom_scene_size`

Current:

```json
"zoom_scene_size": null
```

`null` means automatic framing based on the notional zoom box.

This is recommended because it works naturally with:

```json
"size_fractional"
```

and:

```json
"zoom_padding_factor"
```

If explicitly set:

```json
"zoom_scene_size": [10, 8]
```

then the automatic padding calculation is bypassed.

---

# 31. `zoom_padding_factor`

Current:

```json
"zoom_padding_factor": 1.32
```

This is the parameter to change when selected particles are visible but parts of the outer particles are being cropped by the image boundary.

The zoom-selection algorithm selects particle **centers**.

A polyhedron has finite size, so a particle center can be inside the selected chunk while its vertices extend beyond the nominal crop extent.

The camera is automatically fitted using the notional zoom box. `zoom_padding_factor` adds extra camera margin.

Examples:

```text
1.10 -> tight
1.20 -> modest margin
1.32 -> current setting
1.40 -> more safety
1.50 -> substantial margin
```

If particles are still cropped:

```json
"zoom_padding_factor": 1.40
```

then:

```json
"zoom_padding_factor": 1.50
```

if necessary.

## This does NOT change which particles are selected

That is very important.

`zoom_padding_factor` changes:

```text
camera field of view
```

not:

```text
spatial particle selection
```

To include more particles, change `zoom.size_fractional`.

---

# 32. `zoom_pixel_scale`

Current:

```json
"zoom_pixel_scale": 180
```

Because fewer particles are displayed in the zoom image, a higher pixel scale is useful for resolving polyhedral facets and edges.

For more publication detail:

```json
"zoom_pixel_scale": 250
```

or:

```json
"zoom_pixel_scale": 300
```

Expect longer rendering and larger image files.

---

# 33. `outline`

Current:

```json
"outline": 0.01
```

This controls the visible outline around the convex polyhedron.

Increase for stronger edges:

```json
"outline": 0.02
```

Decrease for subtler edges:

```json
"outline": 0.005
```

Set very close to zero if facet boundaries should be minimally visible.

This is purely visual.

For differently scaled particles, an outline value that looks good for a unit-volume particle may look too thick or too thin.

---

# 34. `roughness`

Current:

```json
"roughness": 0.15
```

This changes the Fresnel material appearance.

Lower roughness generally gives a smoother/glossier surface.

Example:

```json
"roughness": 0.05
```

Higher roughness gives a more diffuse/matte surface:

```json
"roughness": 0.4
```

For publication figures, moderate roughness often makes facets readable without overly strong reflections.

---

# 35. `specular`

Current:

```json
"specular": 0.8
```

Controls strength of specular reflection.

Higher:

```json
"specular": 1.0
```

gives stronger highlights.

Lower:

```json
"specular": 0.3
```

gives a more matte appearance.

If bright highlights obscure cluster colors, decrease `specular`.

---

# 36. `spec_trans`

Current:

```json
"spec_trans": 0.0
```

Controls specular transmission/transparency-related material behavior.

For the current opaque particle rendering, leave:

```json
"spec_trans": 0.0
```

unless a deliberately transmissive appearance is desired and is supported appropriately by the rendering backend.

---

# 37. `ambient_light`

Current:

```json
"ambient_light": 1.5
```

Ambient light illuminates faces more uniformly.

If shadowed faces are too dark:

```json
"ambient_light": 2.0
```

If the particle looks too flat or washed out:

```json
"ambient_light": 1.0
```

The goal is usually to retain enough directional shading to reveal shape while keeping all orientation colors visible.

---

# 38. `directional_light`

Current:

```json
"directional_light": [-1.5, 0.0, 0.0]
```

This controls the direction/vector supplied to Plato's directional light.

Changing it changes which facets receive directional illumination.

Examples:

```json
"directional_light": [-1.0, -1.0, -1.0]
```

or:

```json
"directional_light": [1.0, -1.0, 0.5]
```

If one side of the image is visually flat, change this together with the camera.

This parameter is visual only.

---

# 39. `antialiasing`

Current:

```json
"antialiasing": 1.0
```

Antialiasing improves edge smoothness.

For most uses the current value should be kept.

If experimenting with rendering quality, change this only after confirming supported behavior in the installed Plato/Fresnel version.

---

# 40. `pathtrace_samples`

Current:

```json
"pathtrace_samples": 128
```

This is a major quality-versus-runtime parameter.

Higher path-tracing samples:

- reduce Monte Carlo rendering noise;
- improve smoothness of lighting;
- increase runtime.

Examples:

Fast preview:

```json
"pathtrace_samples": 32
```

Routine figure:

```json
"pathtrace_samples": 128
```

Higher-quality final figure:

```json
"pathtrace_samples": 256
```

Very high-quality final export:

```json
"pathtrace_samples": 512
```

A productive workflow is:

```text
during camera/zoom tuning -> 32 or 64
final publication render -> 128-512
```

---

# 41. `show_box`

Current:

```json
"show_box": true
```

Controls whether the full-system image includes the simulation-box wireframe.

To show the box:

```json
"show_box": true
```

To render only particles in the full-system image:

```json
"show_box": false
```

This does not change periodic coordinates or particle selection.

It only determines whether a `draw.Box` primitive is added to the render scene.

---

# 42. `show_zoom_box`

Current:

```json
"show_zoom_box": false
```

This version intentionally uses:

```json
false
```

so the zoom image contains only particles.

Recommended:

```json
"show_zoom_box": false
```

for clean publication images.

If debugging the selected chunk and you want to see its notional boundary:

```json
"show_zoom_box": true
```

After confirming the crop, change it back to false.

---

# 43. `box_width`

Current:

```json
"box_width": 0.04
```

Controls the visual thickness of the box wireframe.

It matters only when:

```json
"show_box": true
```

or:

```json
"show_zoom_box": true
```

Thinner:

```json
"box_width": 0.02
```

Thicker:

```json
"box_width": 0.08
```

For a particle-only zoom with `show_zoom_box=false`, this value has no visible effect on the zoom image.

---

# 44. `box_color_rgba`

Current:

```json
"box_color_rgba": [0.1, 0.1, 0.1, 1.0]
```

RGBA means:

```text
red
green
blue
alpha
```

with values typically between 0 and 1.

Current color is a dark gray.

Black:

```json
"box_color_rgba": [0.0, 0.0, 0.0, 1.0]
```

Medium gray:

```json
"box_color_rgba": [0.5, 0.5, 0.5, 1.0]
```

This is visual only.

---

# 45. Zoom subsystem overview

Current:

```json
"zoom": {
    "enabled": true,
    "center_fractional": [0.0, 0.0, 0.0],
    "center_particle_index": null,
    "size_fractional": [0.3, 0.3, 0.3],
    "cluster_ids": null
}
```

The zoom selector works in **fractional periodic-box coordinates**.

This is important because the system can be triclinic.

The code first converts:

```text
Cartesian coordinates
        ->
fractional coordinates
```

using the HOOMD box matrix.

It then uses a minimum-image displacement around the requested center.

As a result, a zoom region near a periodic boundary remains visually contiguous rather than splitting across opposite sides of the image.

---

# 46. HOOMD triclinic box convention

The trajectory box is interpreted as:

```text
[Lx, Ly, Lz, xy, xz, yz]
```

The code constructs lattice vectors:

```text
a = (Lx, 0, 0)

b = (xy*Ly, Ly, 0)

c = (xz*Lz, yz*Lz, Lz)
```

and the box matrix:

```text
B = [a b c]
```

Cartesian and fractional coordinates satisfy:

```text
r = B s
```

This makes the zoom mechanism work for both:

- orthorhombic boxes;
- tilted triclinic boxes.

---

# 47. `zoom.enabled`

Current:

```json
"enabled": true
```

If true:

```text
full image + zoom image
```

are produced.

To disable zoom:

```json
"enabled": false
```

Then the full-system image and scientific CSV outputs are still generated.

---

# 48. `zoom.center_fractional`

Current:

```json
"center_fractional": [0.0, 0.0, 0.0]
```

This is the zoom center in fractional box coordinates when:

```json
"center_particle_index": null
```

The most intuitive values for a centered HOOMD box are approximately in:

```text
[-0.5, 0.5]
```

along each fractional direction.

Examples:

Center:

```json
"center_fractional": [0.0, 0.0, 0.0]
```

Shift toward +x:

```json
"center_fractional": [0.25, 0.0, 0.0]
```

Shift toward +x and -y:

```json
"center_fractional": [0.25, -0.20, 0.0]
```

Near a periodic boundary:

```json
"center_fractional": [0.48, 0.0, 0.0]
```

is allowed. The minimum-image treatment keeps the selected neighborhood continuous.

---

# 49. `zoom.center_particle_index`

Current:

```json
"center_particle_index": null
```

When `null`, the explicit `center_fractional` setting is used.

To center the zoom on particle 1732:

```json
"center_particle_index": 1732
```

Then `center_fractional` is ignored for the actual center calculation.

This is useful when a particular local structure or defect is known by particle index.

Example:

```json
"zoom": {
    "enabled": true,
    "center_fractional": [0.0, 0.0, 0.0],
    "center_particle_index": 1732,
    "size_fractional": [0.25, 0.25, 0.25],
    "cluster_ids": null
}
```

---

# 50. `zoom.size_fractional`

Current:

```json
"size_fractional": [0.3, 0.3, 0.3]
```

This specifies the width of the selected region as a fraction of each box direction.

Current interpretation:

```text
30% of box direction 1
30% of box direction 2
30% of box direction 3
```

The allowed range for each component is:

```text
0 < component <= 1
```

## Smaller zoom chunk

```json
"size_fractional": [0.2, 0.2, 0.2]
```

Result:

- fewer particle centers selected;
- more local detail;
- chunk covers 20% of each box direction.

## Larger chunk

```json
"size_fractional": [0.5, 0.5, 0.5]
```

Result:

- more particles;
- half-box width along each fractional direction.

## Anisotropic chunk

```json
"size_fractional": [0.5, 0.2, 0.3]
```

This is valid.

It selects:

```text
50% along first box direction
20% along second
30% along third
```

This can be useful for viewing a slab, layer, chain, or elongated region.

---

# 51. `zoom.size_fractional` versus `zoom_padding_factor`

These two are easy to confuse.

## `size_fractional`

Changes:

```text
WHICH PARTICLE CENTERS are selected
```

Example:

```json
"size_fractional": [0.3, 0.3, 0.3]
```

to:

```json
"size_fractional": [0.4, 0.4, 0.4]
```

selects a larger physical chunk.

## `zoom_padding_factor`

Changes:

```text
HOW MUCH CAMERA MARGIN surrounds the already selected chunk
```

Example:

```json
"zoom_padding_factor": 1.32
```

to:

```json
"zoom_padding_factor": 1.45
```

does not add particles. It only reduces image-edge cropping.

Use:

```text
missing neighboring particles -> increase size_fractional

same selected particles but outer polyhedra cut by image boundary
-> increase zoom_padding_factor
```

---

# 52. Why particles at the zoom boundary can appear beyond the nominal chunk

The crop selection tests the **particle center**:

```text
abs(center displacement) <= half crop size
```

It does not test every vertex of every oriented polyhedron.

Therefore a selected particle near the boundary can extend outside the notional crop volume.

The renderer still draws the complete polyhedron.

Since `show_zoom_box` is false, the crop boundary itself is not visible.

This is generally desirable for publication images.

If the outer part of that particle reaches the camera edge, increase:

```json
"zoom_padding_factor"
```

---

# 53. `zoom.cluster_ids`

Current:

```json
"cluster_ids": null
```

`null` means:

```text
show all final orientation clusters inside the zoom region
```

To show only cluster 0:

```json
"cluster_ids": [0]
```

To show clusters 0 and 2:

```json
"cluster_ids": [0, 2]
```

To show three selected clusters:

```json
"cluster_ids": [0, 2, 4]
```

This filtering applies only to the zoom-selection mask.

It does not recompute the clusters.

It does not change the full image.

It is useful for highlighting specific orientation states spatially.

---

# 54. Periodic-boundary handling in the zoom

The code computes:

```python
delta_frac = frac - center_frac
delta_frac -= np.round(delta_frac)
```

This applies the minimum-image convention in fractional space.

For example, a particle at fractional x:

```text
-0.49
```

is physically very close to a center at:

```text
+0.49
```

through the periodic boundary.

A non-periodic crop would consider them far apart.

The current implementation instead recognizes them as nearby and renders them together in one recentered local chunk.

This is one of the main advantages of selecting the zoom in fractional periodic coordinates.

---

# 55. Full-system render versus zoom render

The same rendering function is used for both.

The differences are controlled by `is_zoom`.

## Full render uses

```text
full_scene_size
full_padding_factor
full_pixel_scale
show_box
```

## Zoom render uses

```text
zoom_scene_size
zoom_padding_factor
zoom_pixel_scale
show_zoom_box
```

Particle geometry, orientations, and cluster colors are otherwise rendered in the same way.

---

# 56. What files are produced

For `output_prefix = "EPD"` and resolved frame 201:

```text
outputs_orientational_snapshot/
│
├── EPD_frame_000201_full.png
├── EPD_frame_000201_zoom.png
├── EPD_frame_000201_particle_clusters.csv
├── EPD_frame_000201_cluster_summary.csv
├── EPD_frame_000201_metadata.json
└── EPD_proper_rotational_symmetries.csv
```

A temporary file can also appear during a large calculation:

```text
.pairwise_orientation_angles_float32.dat
```

This is normally removed automatically after use.

---

# 57. `*_full.png`

Example:

```text
EPD_frame_000201_full.png
```

Contains:

- all particles from the selected GSD frame;
- color according to final orientational assignment;
- simulation box if `show_box=true`;
- camera and material settings from `render`.

---

# 58. `*_zoom.png`

Example:

```text
EPD_frame_000201_zoom.png
```

Contains:

- only particles whose centers satisfy the zoom spatial mask;
- optionally only requested cluster IDs;
- periodic-boundary-aware recentered coordinates;
- no zoom box when `show_zoom_box=false`.

For the current version this is intentionally a clean particle-only image.

---

# 59. `*_particle_clusters.csv`

This contains one row for every original particle.

Columns:

```text
particle_index
x
y
z
qw
qx
qy
qz
cluster_id
cluster_color
reference_particle_index
angle_to_reference_deg
```

Interpretation:

## `particle_index`

Original index in the GSD frame.

## `x, y, z`

Original Cartesian particle position.

## `qw, qx, qy, qz`

Normalized particle orientation quaternion.

## `cluster_id`

Final retained orientation/color assignment.

## `cluster_color`

Hex color used in the image.

## `reference_particle_index`

Particle whose orientation acts as the retained reference for that final cluster.

## `angle_to_reference_deg`

Symmetry-reduced angular distance from the particle to its selected retained reference.

This CSV is useful for:

- spatial cluster analysis;
- selecting particles by orientation;
- reproducing colors;
- calculating cluster correlations;
- locating particular orientation domains;
- choosing `center_particle_index` for a subsequent zoom.

---

# 60. `*_cluster_summary.csv`

One row per retained final orientation reference.

Columns:

```text
cluster_id
color
reference_particle_index
reference_qw
reference_qx
reference_qy
reference_qz
discovery_cluster_size
final_population
mean_angle_to_reference_deg
max_angle_to_reference_deg
```

The distinction between:

```text
discovery_cluster_size
```

and:

```text
final_population
```

is important.

The discovery size is the number of particles in the original tight provisional group.

The final population is the number of particles assigned to that reference after **all particles** are mapped to the nearest retained orientation.

These values can differ substantially.

---

# 61. `*_proper_rotational_symmetries.csv`

Example:

```text
EPD_proper_rotational_symmetries.csv
```

Columns:

```text
physical_rotation_id
qw
qx
qy
qz
max_vertex_residual
```

Each row is one physical proper rotational symmetry operation.

The file intentionally stores one canonical quaternion per physical operation.

The internal `q` and `-q` duplicates required by quaternion double-cover handling are not written as separate physical symmetries.

This file is useful for validating arbitrary new particle shapes.

---

# 62. `*_metadata.json`

Contains information such as:

```text
gsd_file
shape_file
trajectory_num_frames
requested_frame_index
resolved_frame_index
num_particles
box
orientation_angle_tol_deg
cluster_size_cutoff
num_provisional_clusters
provisional_cluster_sizes_descending
num_retained_clusters
retained_reference_particle_indices
retained_discovery_sizes
final_cluster_populations
cluster_colors
num_physical_proper_rotations
num_equivalent_quaternions_with_signs
symmetry_vertex_tolerance
zoom_selected_particle_indices
```

Keep this file with publication figures.

It records enough settings to determine what frame and classification generated the image.

---

# 63. Recommended workflow for a completely new system

Do not change many parameters simultaneously.

Use the following sequence.

## Step 1: copy the working project

Create:

```text
new_system_snapshot/
```

and copy:

```text
render_sys_snapshot_from_gsd_orientational_color.py
param_file.json
new_trajectory.gsd
new_shape.json
```

## Step 2: change only the input filenames

```json
"gsd_file": "new_trajectory.gsd",
"shape_file": "new_shape.json"
```

## Step 3: change output prefix

```json
"output_prefix": "NewShape"
```

## Step 4: use the last frame initially

```json
"frame_index": -1
```

## Step 5: verify shape and symmetry output

Run once.

Check:

```text
number of vertices
convex-hull volume
distinct proper rotations
maximum vertex residual
```

Do not proceed until symmetry looks physically reasonable.

## Step 6: tune symmetry tolerance if necessary

Prefer explicit:

```json
"symmetry_vertex_tolerance": ...
```

for unusually scaled coordinate systems.

## Step 7: inspect orientational cluster sizes

Start with a reasonable physical estimate for:

```json
"orientation_angle_tol"
```

and a modest:

```json
"cluster_size_cutoff"
```

Check terminal output and `cluster_summary.csv`.

## Step 8: make palette sufficiently long

Provide more colors than the largest plausible number of retained states.

## Step 9: render at low/medium quality while tuning view

For example:

```json
"pathtrace_samples": 32,
"full_pixel_scale": 70,
"zoom_pixel_scale": 100
```

## Step 10: tune camera

Change:

```json
"camera_euler_zyx_deg"
```

until facets and domains are clearly visible.

## Step 11: tune zoom region

Change:

```text
center_fractional
or
center_particle_index
```

and:

```text
size_fractional
```

## Step 12: fix edge cropping

Change:

```json
"zoom_padding_factor"
```

not `size_fractional`, unless you actually want more neighboring particles.

## Step 13: final publication render

Increase:

```json
"pathtrace_samples"
```

and:

```json
pixel_scale
```

only after camera and crop are finalized.

---

# 64. Example parameter file for another arbitrary convex polyhedron

```json
{
    "gsd_file": "my_shape_trajectory.gsd",
    "shape_file": "my_shape_vertices.json",
    "frame_index": -1,

    "orientation_angle_tol": 30.0,
    "cluster_size_cutoff": 100,

    "cmap_original": [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#6A3D9A"
    ],

    "symmetry_vertex_tolerance": 0.001,

    "pairwise_block_size": 256,
    "pairwise_matrix_ram_limit_mb": 1024,

    "output_dir": "snapshot_outputs",
    "output_prefix": "MyPolyhedron",

    "render": {
        "camera_euler_zyx_deg": [20.0, 15.0, 10.0],

        "full_scene_size": null,
        "full_padding_factor": 1.08,
        "full_pixel_scale": 120,

        "zoom_scene_size": null,
        "zoom_padding_factor": 1.35,
        "zoom_pixel_scale": 200,

        "outline": 0.01,
        "roughness": 0.2,
        "specular": 0.6,
        "spec_trans": 0.0,

        "ambient_light": 1.5,
        "directional_light": [-1.0, -0.5, 0.3],

        "antialiasing": 1.0,
        "pathtrace_samples": 128,

        "show_box": true,
        "show_zoom_box": false,

        "box_width": 0.04,
        "box_color_rgba": [0.1, 0.1, 0.1, 1.0]
    },

    "zoom": {
        "enabled": true,
        "center_fractional": [0.0, 0.0, 0.0],
        "center_particle_index": null,
        "size_fractional": [0.25, 0.25, 0.25],
        "cluster_ids": null
    }
}
```

These values are examples, not universal defaults. The scientific parameters must be chosen for the new particle/system.

---

# 65. Quick decision table: what should I change?

| Problem / desired change | Parameter to change |
|---|---|
| Render another trajectory | `gsd_file` |
| Render another particle shape | `shape_file` |
| Render frame 201 | `frame_index: 201` |
| Render last frame | `frame_index: -1` |
| Clusters are splitting too much | increase `orientation_angle_tol` |
| Different orientation states are merging | decrease `orientation_angle_tol` |
| Too many tiny orientation references | increase `cluster_size_cutoff` |
| Important small state is missing | decrease `cluster_size_cutoff` |
| Number of retained states is capped | add more colors to `cmap_original` |
| Wrong particle symmetry group detected | tune symmetry tolerance |
| Shape is at unusual coordinate scale | use `symmetry_vertex_tolerance` |
| Pairwise calculation causes temporary memory problem | decrease `pairwise_block_size` |
| Whole matrix uses too much RAM | decrease `pairwise_matrix_ram_limit_mb` |
| Disk memmap is slow and RAM is available | increase `pairwise_matrix_ram_limit_mb` |
| Need different viewing angle | `camera_euler_zyx_deg` |
| Full image edge is cropped | increase `full_padding_factor` |
| Zoom particle surfaces are cropped at image edge | increase `zoom_padding_factor` |
| Need more particles in zoom | increase `zoom.size_fractional` |
| Need fewer particles in zoom | decrease `zoom.size_fractional` |
| Need different spatial region | change `center_fractional` |
| Need region around a known particle | set `center_particle_index` |
| Show only selected orientation colors in zoom | set `zoom.cluster_ids` |
| Remove box from full image | `show_box: false` |
| Add box to zoom for debugging | `show_zoom_box: true` |
| Improve final image resolution | increase `*_pixel_scale` |
| Reduce path-tracing noise | increase `pathtrace_samples` |
| Fast preview | decrease `pathtrace_samples` |
| Stronger particle edges | increase `outline` |
| Softer particle edges | decrease `outline` |
| Surface too shiny | decrease `specular` / increase `roughness` |
| Surface too dull | increase `specular` / decrease `roughness` |
| Dark faces | increase `ambient_light` or change light direction |

---

# 66. Troubleshooting: `ERROR: Expecting value: line 1 column 1`

Most common cause:

You passed a `.jsonc` file containing comments to a parser that expects strict JSON.

Wrong:

```bash
python script.py documented_parameters.jsonc
```

if that file begins with comments.

Use a strict JSON file:

```bash
python script.py param_file.json
```

Validate manually with:

```bash
python -m json.tool param_file.json
```

If valid, Python prints formatted JSON.

If invalid, it prints the location of the syntax error.

---

# 67. Troubleshooting: GSD file not found

Example error:

```text
GSD trajectory not found: ...
```

Check:

```bash
ls
```

Then verify:

```json
"gsd_file": "exact_filename.gsd"
```

Remember:

relative paths are resolved relative to the parameter file.

If needed use an absolute path:

```json
"gsd_file": "/home/user/project/data/trajectory.gsd"
```

---

# 68. Troubleshooting: shape file not found

Check:

```json
"shape_file": "exact_shape_filename.json"
```

or use an absolute path.

Do not assume the code searches the whole project directory tree.

---

# 69. Troubleshooting: shape does not span 3D

Error may indicate:

```text
The supplied vertices do not span a 3D polyhedron.
```

This means the selected `N x 3` array is planar, linear, degenerate, or the wrong numerical array was selected from the JSON.

Check terminal output:

```text
Selected vertex field: ...
```

Ensure that the selected field is actually the particle vertex list.

---

# 70. Troubleshooting: zero/invalid convex-hull volume

Possible causes:

- all vertices are coplanar;
- duplicate/incorrect coordinates;
- shape JSON does not contain the intended body vertices;
- malformed numerical data.

Check the shape independently before changing clustering parameters.

---

# 71. Troubleshooting: wrong number of rotational symmetries

If too many symmetries are found:

- tolerance may be too loose;
- particle vertices may be accidentally symmetrized;
- shape scale may make absolute tolerance too large.

Use a smaller absolute tolerance.

For historical precision form:

```text
increase p
```

Example:

```json
"tolerance_for_inv_quat_of_body_calc": 3
```

instead of:

```json
2
```

If too few symmetries are found:

- tolerance may be too strict;
- vertex coordinates contain numerical noise.

Use a larger absolute tolerance.

For historical precision form:

```text
decrease p
```

Example:

```json
"tolerance_for_inv_quat_of_body_calc": 2
```

instead of:

```json
3
```

---

# 72. Troubleshooting: group-closure failure

If the code reports that detected proper rotations fail group closure, the accepted operations are internally inconsistent.

Typical action:

1. inspect the shape coordinates;
2. confirm intended symmetry;
3. change the symmetry matching tolerance carefully;
4. check whether the shape is only approximately symmetric rather than exactly symmetric.

Do not bypass group validation without understanding the consequence.

---

# 73. Troubleshooting: too many orientation clusters

Possible reasons:

- `orientation_angle_tol` is too small;
- physical orientation distribution is broad;
- symmetry group is incomplete because symmetry tolerance was too strict;
- particle orientation convention in the GSD does not match assumptions;
- the system genuinely contains many states.

Check symmetry first.

Then consider increasing:

```json
"orientation_angle_tol"
```

gradually.

For example:

```text
20 -> 25 -> 30 -> 35 degrees
```

rather than making a very large jump.

---

# 74. Troubleshooting: too few clusters

Possible reasons:

- `orientation_angle_tol` is too large;
- distinct states are separated by less than the chosen cutoff;
- the particle has a large rotational symmetry group, making some raw orientations physically equivalent;
- `cluster_size_cutoff` removes smaller provisional states;
- palette contains too few colors.

Check both:

```text
number of provisional clusters
number of retained clusters
```

in metadata/terminal output.

These are not the same quantity.

---

# 75. Troubleshooting: expected cluster is missing from final colors

Check three things.

## 1. Cluster size cutoff

Maybe:

```text
provisional population < cluster_size_cutoff
```

Decrease:

```json
"cluster_size_cutoff": ...
```

## 2. Number of colors

Maybe more clusters passed the cutoff than colors were supplied.

Add colors.

## 3. Orientation tolerance

The state may have merged during provisional clustering.

Decrease:

```json
"orientation_angle_tol"
```

if scientifically justified.

---

# 76. Troubleshooting: final cluster maximum angle is large

Inspect:

```text
max_angle_to_reference_deg
```

in the cluster summary.

A large value can occur because particles from discarded small provisional groups are reassigned to the nearest retained reference.

Possible responses:

- decrease `cluster_size_cutoff`;
- add more palette colors;
- inspect the provisional cluster-size distribution;
- decide whether those particles represent genuine additional states.

Do not automatically increase `orientation_angle_tol`; that changes a different stage of the algorithm.

---

# 77. Troubleshooting: zoom image has too few particles

Increase:

```json
"size_fractional": [0.3, 0.3, 0.3]
```

to something like:

```json
"size_fractional": [0.4, 0.4, 0.4]
```

or move the center.

This changes the selection.

---

# 78. Troubleshooting: zoom image has too many particles

Decrease:

```json
"size_fractional"
```

Example:

```json
"size_fractional": [0.2, 0.2, 0.2]
```

---

# 79. Troubleshooting: particles are cut off at image edges

If the desired particle centers are already selected but the finite polyhedra are cut at the image boundary, increase:

```json
"zoom_padding_factor"
```

Current:

```json
1.32
```

Try:

```json
1.40
```

then:

```json
1.50
```

if necessary.

Do not change `size_fractional` unless you also want to change which particle centers are included.

---

# 80. Troubleshooting: changing padding does nothing

Check:

```json
"zoom_scene_size"
```

If it is explicitly set to a number or `[width, height]`, automatic sizing is bypassed.

Set:

```json
"zoom_scene_size": null
```

so:

```json
"zoom_padding_factor"
```

takes effect.

The same logic applies to:

```text
full_scene_size
full_padding_factor
```

---

# 81. Troubleshooting: the zoom region looks split at a periodic boundary

The current code already applies minimum-image wrapping in fractional coordinates.

If it still looks wrong:

1. verify the GSD box tilt factors;
2. verify the particle positions use the same HOOMD box convention;
3. inspect `center_fractional`;
4. ensure the trajectory is not storing an unexpected unwrapped coordinate convention.

The code deliberately recenters the selected periodic displacement around the origin for rendering.

---

# 82. Troubleshooting: zoom image shows a box

Set:

```json
"show_zoom_box": false
```

The current modified version defaults the zoom box to false even if the key is missing, but keeping the key explicitly false makes the intended behavior obvious.

---

# 83. Troubleshooting: full image should also contain only particles

Set:

```json
"show_box": false
```

This removes the simulation box from the full render.

---

# 84. Troubleshooting: image is noisy

Increase:

```json
"pathtrace_samples"
```

Example:

```text
64 -> 128 -> 256
```

For final figures, also increase pixel scale if needed.

---

# 85. Troubleshooting: image rendering is too slow

During tuning:

```json
"pathtrace_samples": 32
```

and lower pixel scales:

```json
"full_pixel_scale": 70,
"zoom_pixel_scale": 100
```

Once camera and crop are finalized, restore high-quality values.

---

# 86. Troubleshooting: colors appear too shiny / washed out

Try:

```json
"roughness": 0.25,
"specular": 0.5
```

If colors look too flat, reverse somewhat:

```json
"roughness": 0.10,
"specular": 0.8
```

Also adjust ambient and directional light.

---

# 87. Troubleshooting: only one cluster is retained

Possible explanations:

- only one provisional cluster exceeds the cutoff;
- tolerance is very large;
- all orientations are symmetry-equivalent;
- the system truly has one dominant orientational state.

Inspect:

```text
provisional_cluster_sizes_descending
```

in metadata before changing parameters.

---

# 88. Troubleshooting: more clusters exist than colors

The terminal prints a warning.

The code keeps only the largest number of retained references equal to the palette length.

Solution:

add more colors.

Example:

```json
"cmap_original": [
    "#0080ff",
    "#ff8000",
    "#5b0aa2",
    "#db023c",
    "#00ffff",
    "#2ca02c",
    "#8c564b",
    "#e377c2"
]
```

---

# 89. Troubleshooting: pairwise matrix uses disk even though RAM is available

Increase:

```json
"pairwise_matrix_ram_limit_mb"
```

Example:

```json
"pairwise_matrix_ram_limit_mb": 2048
```

Only do this if the node actually has enough free memory.

---

# 90. Troubleshooting: temporary `.dat` file remains after a crash

The code normally deletes:

```text
.pairwise_orientation_angles_float32.dat
```

inside `finally`.

If Python is forcibly killed, the machine crashes, or the job is terminated externally, cleanup may not execute.

Check:

```bash
ls -lah outputs_orientational_snapshot
```

If no process is still using the file, it can be removed manually:

```bash
rm outputs_orientational_snapshot/.pairwise_orientation_angles_float32.dat
```

---

# 91. Which parameters are scientific and which are visual?

This distinction is important for reproducibility.

## Scientific / classification parameters

Changing these can alter the orientational result:

```text
shape_file
frame_index
orientation_angle_tol
cluster_size_cutoff
symmetry_vertex_tolerance
tolerance_for_inv_quat_of_body_calc
palette length, if it truncates retained clusters
```

The actual color values are visual, but the **number of available colors** can limit how many retained references survive.

## Performance parameters

These are intended not to change the scientific classification:

```text
pairwise_block_size
pairwise_matrix_ram_limit_mb
```

## Visual parameters

These change the appearance:

```text
camera_euler_zyx_deg
full_scene_size
full_padding_factor
full_pixel_scale
zoom_scene_size
zoom_padding_factor
zoom_pixel_scale
outline
roughness
specular
spec_trans
ambient_light
directional_light
antialiasing
pathtrace_samples
show_box
show_zoom_box
box_width
box_color_rgba
```

## Spatial-display parameters

These decide what appears in the zoom but do not recalculate clusters:

```text
zoom.enabled
zoom.center_fractional
zoom.center_particle_index
zoom.size_fractional
zoom.cluster_ids
```

---

# 92. Recommended publication workflow

A reliable workflow is:

## Analysis pass

Use:

```json
"pathtrace_samples": 32,
"full_pixel_scale": 60,
"zoom_pixel_scale": 100
```

Focus on:

- symmetry detection;
- cluster populations;
- orientation tolerance;
- cluster cutoff;
- zoom location.

## Visual tuning pass

Keep classification parameters fixed.

Tune:

```text
camera angle
zoom center
zoom size
padding
material
lighting
colors
```

## Final render

Use for example:

```json
"pathtrace_samples": 256,
"full_pixel_scale": 150,
"zoom_pixel_scale": 250
```

or higher if necessary.

Save the exact parameter file with the figure.

---

# 93. Reproducibility recommendations

For every figure used in a paper, archive:

```text
trajectory filename/checksum
shape JSON
parameter JSON
renderer version
particle-cluster CSV
cluster-summary CSV
symmetry CSV
metadata JSON
PNG output
```

Do not retain only the PNG.

The CSV/metadata outputs explain how the colors were produced.

---

# 94. Adapting the project to a different particle shape

For a new convex polyhedron:

1. create a JSON containing body-frame vertices;
2. verify the coordinates are centered consistently with the simulation body origin;
3. update `shape_file`;
4. run the program;
5. inspect the selected vertex field;
6. inspect convex-hull volume;
7. inspect detected proper-rotation count;
8. compare with expected point-group rotational symmetry;
9. tune geometric symmetry tolerance only if required;
10. then analyze orientation clusters.

Do not start by tuning `orientation_angle_tol` if the detected particle symmetry is wrong.

The symmetry definition determines the angular distances themselves.

---

# 95. Adapting to a different system size

No `num_particles` parameter is required.

The program reads the number of particles directly from the selected GSD frame.

For larger systems:

- pairwise matrix grows quadratically;
- lower `pairwise_matrix_ram_limit_mb` if RAM is constrained;
- lower `pairwise_block_size` if temporary block memory becomes problematic;
- ensure sufficient disk space if memmap is used.

For smaller systems, the current values are usually conservative.

---

# 96. Adapting to a different box shape

Orthorhombic:

```text
xy = xz = yz = 0
```

works.

Triclinic:

```text
one or more tilt factors nonzero
```

also works.

The zoom is performed in fractional triclinic coordinates rather than assuming Cartesian x/y/z are independent box directions.

---

# 97. Adapting to a different number of orientational states

Suppose a new system is expected to have approximately 12 discrete states.

Do not keep a five-color palette.

Provide at least 12 distinct colors.

Then tune:

```text
orientation_angle_tol
cluster_size_cutoff
```

based on the scientific definition of distinct states.

Remember that `cluster_size_cutoff` and palette length are separate filters.

---

# 98. Understanding the terminal output

A typical run prints sections in approximately this logical order:

```text
Shape
Particle rotational symmetry
Trajectory frame
Pairwise symmetry-reduced orientation matrix
Discovering orientational clusters
Retained orientation references and final populations
Zoom section
Saved ...
Done.
```

Read these messages.

They provide important diagnostics.

Do not treat the script as a black-box image generator.

---

# 99. Interpretation of retained cluster table

The terminal prints something like:

```text
cluster   reference_particle   discovery_size   final_population   max_angle_deg
```

Interpretation:

## `cluster`

Final color/reference ID.

## `reference_particle`

Original GSD particle index used as the orientation representative.

## `discovery_size`

Population of the original greedy provisional cluster.

## `final_population`

Population after every particle is mapped to the nearest retained reference.

## `max_angle_deg`

Largest symmetry-reduced reference distance among finally assigned particles.

A large difference between discovery size and final population means that the reference absorbed particles from discarded or neighboring provisional groups.

---

# 100. Why shape symmetry must be determined before clustering

Suppose a particle has a 72-degree body rotation that maps it onto itself.

Two GSD quaternions differing by that body symmetry should represent the same physical orientation.

If clustering were performed using raw quaternion differences, the program could incorrectly produce multiple colored states that are actually symmetry-equivalent.

The current pipeline avoids this by performing:

```text
shape geometry
    ->
proper body symmetries
    ->
symmetry-reduced angular separation
    ->
clustering
```

The order is therefore scientifically important.

---

# 101. Why only proper rotations are used

Particle orientation quaternions represent elements of SO(3).

Reflections and inversion have determinant `-1` and are not represented by unit rotation quaternions.

Therefore the symmetry detector explicitly rejects improper source-to-target maps.

A particle may have a full point group containing mirrors or inversion, but only the proper rotational subgroup is relevant for quaternion orientation equivalence in this workflow.

---

# 102. Why `q` and `-q` are both supplied

A unit quaternion and its negative represent the same rotation:

```text
q ~ -q
```

The code explicitly constructs an equivalent list:

```text
q1
-q1
q2
-q2
...
```

for freud.

This prevents artificial angular differences caused solely by quaternion sign.

---

# 103. Why the code normalizes GSD quaternions

The GSD orientations should be unit quaternions, but floating-point storage can introduce small norm errors.

The program calculates each quaternion norm, reports the largest deviation from 1, and normalizes all orientations before analysis.

A zero-norm quaternion is rejected as invalid.

---

# 104. Why the pairwise matrix diagonal is set to zero

A particle must have zero orientation difference from itself.

Small numerical noise could produce a tiny residual.

The code explicitly sets:

```python
np.fill_diagonal(matrix, 0.0)
```

so self-separation is exact.

---

# 105. Shape symmetry algorithm in more detail

The symmetry search uses a numerically well-conditioned triple of centered vertices.

It chooses the triple maximizing:

```text
|det([v1 v2 v3])|
```

A large determinant means the three vectors form a stable 3D basis.

Every true symmetry must map this triple onto another ordered triple while preserving pairwise dot products.

Those dot products form the Gram matrix:

```text
G = V^T V
```

Candidate target triples whose Gram matrix does not approximately match are rejected early.

This avoids performing more expensive operations for obviously impossible mappings.

For a surviving triple, the raw map is:

```text
R_raw = target * source^-1
```

Improper maps are rejected.

The candidate is then projected to a nearby proper orthogonal rotation using SVD.

Finally all vertices are matched one-to-one using the Hungarian assignment algorithm.

Only a candidate that maps **all vertices** within tolerance is accepted.

The accepted mapping is refined using all matched vertex pairs.

Distinct symmetries are deduplicated using the induced vertex permutation.

The resulting permutation set is explicitly tested for group closure.

---

# 106. Performance implication of many shape vertices

The symmetry detector considers ordered target triples.

For `n` vertices, the number of ordered triples is:

```text
n * (n-1) * (n-2)
```

For 12 vertices:

```text
12 * 11 * 10 = 1320
```

which is modest.

For a polyhedron with hundreds of supplied surface points rather than true vertices, symmetry detection can become unnecessarily expensive.

Therefore provide the actual convex-polyhedron vertices, not a dense triangulated surface mesh containing thousands of redundant points.

---

# 107. Current shape example: EPD/J16

The current shape file identifies:

```text
Elongated Pentagonal Dipyramid
J16
```

and contains 12 vertices.

It is stored as a unit-volume particle in a principal frame.

This is an ideal input style for the renderer:

- finite vertex list;
- already centered;
- body-frame coordinates;
- physically meaningful scale.

When preparing another shape, use the same general philosophy.

---

# 108. What this code does NOT currently do

The current program does not:

- average clusters over multiple frames;
- track cluster identity through time;
- perform temporal state matching;
- render multiple particle shapes simultaneously;
- infer a polyhedron from a non-convex mesh;
- analyze translational neighbors;
- compute RDF;
- compute contact networks;
- perform cluster connectivity in real space;
- automatically choose `orientation_angle_tol`;
- automatically determine `cluster_size_cutoff`;
- automatically generate arbitrarily many distinct colors;
- use a globally order-independent orientation clustering algorithm;
- fit the zoom camera to every actual rotated particle vertex exactly.

These would require extensions rather than parameter changes.

---

# 109. Parameter-changing philosophy

When adapting the program, separate questions into four categories.

## A. Am I changing the scientific dataset?

Change:

```text
gsd_file
shape_file
frame_index
```

## B. Am I changing what counts as an orientational state?

Change:

```text
orientation_angle_tol
cluster_size_cutoff
symmetry tolerance
palette length when needed
```

## C. Am I only changing performance?

Change:

```text
pairwise_block_size
pairwise_matrix_ram_limit_mb
```

## D. Am I only changing the figure?

Change:

```text
render.*
zoom.*
```

This separation prevents accidental scientific changes while adjusting visual appearance.

---

# 110. Suggested safe workflow when a figure looks wrong

Do not immediately edit the Python code.

Use this sequence:

```text
1. Check input files.
2. Check frame number.
3. Check detected symmetry count.
4. Check provisional cluster sizes.
5. Check retained cluster count.
6. Check palette length.
7. Check final angle statistics.
8. Check zoom selection.
9. Check camera/padding.
10. Check rendering material/lighting.
```

Only edit source code when the required behavior cannot be expressed by the parameter file.

---

# 111. When source-code modification is actually required

Edit the Python code if you need features such as:

- multiple shapes/types in one GSD;
- separate palette rules for particle types;
- order-independent clustering;
- clustering across many frames simultaneously;
- temporal cluster tracking;
- render only particles satisfying arbitrary metadata conditions;
- per-particle sizes;
- a non-convex body;
- transparent background not supported by current scene setup;
- exact camera fitting based on every transformed vertex rather than a padded box;
- interactive GUI selection;
- automatic movie generation.

For normal tuning of the existing workflow, modify the JSON instead.

---

# 112. Minimal checklist before using results scientifically

Before interpreting domain colors as physical orientation states, verify:

- [ ] Correct GSD file
- [ ] Correct frame
- [ ] Correct shape file
- [ ] Shape vertices correspond to simulation body frame
- [ ] Shape is convex and three-dimensional
- [ ] Proper rotation count is physically sensible
- [ ] Symmetry residuals are acceptably small
- [ ] `orientation_angle_tol` is physically justified
- [ ] `cluster_size_cutoff` is documented
- [ ] Palette is long enough
- [ ] Provisional cluster-size distribution inspected
- [ ] Final reference-angle statistics inspected
- [ ] Parameter JSON archived
- [ ] CSV and metadata files archived

---

# 113. Minimal checklist before producing a final figure

- [ ] Camera orientation finalized
- [ ] `full_scene_size` / `zoom_scene_size` choice understood
- [ ] Padding prevents edge clipping
- [ ] Zoom region contains desired particles
- [ ] `show_zoom_box=false` for particle-only crop if desired
- [ ] Colors remain visually distinct
- [ ] Material highlights do not obscure colors
- [ ] Pixel scale adequate
- [ ] Path-tracing samples adequate
- [ ] Final PNG visually inspected
- [ ] Exact parameter file saved with figure

---

# 114. Current recommended EPD settings

The current working EPD values are:

```json
"orientation_angle_tol": 45.0,
"cluster_size_cutoff": 200,
"tolerance_for_inv_quat_of_body_calc": 2,
"pairwise_block_size": 256,
"pairwise_matrix_ram_limit_mb": 512
```

Rendering:

```json
"camera_euler_zyx_deg": [0.0, 10.0, 10.0],
"full_padding_factor": 1.05,
"full_pixel_scale": 100,
"zoom_padding_factor": 1.32,
"zoom_pixel_scale": 180,
"outline": 0.01,
"roughness": 0.15,
"specular": 0.8,
"ambient_light": 1.5,
"pathtrace_samples": 128,
"show_box": true,
"show_zoom_box": false
```

Zoom:

```json
"center_fractional": [0.0, 0.0, 0.0],
"center_particle_index": null,
"size_fractional": [0.3, 0.3, 0.3],
"cluster_ids": null
```

These are working settings for this example. They should not be assumed to be universal scientific defaults for every particle shape or phase.

---

# 115. Final practical summary

For most day-to-day use, remember these rules:

### Change the frame

```json
"frame_index": 201
```

### Change how similar orientations must be to belong to a provisional state

```json
"orientation_angle_tol": ...
```

### Change how large a provisional state must be to remain an independent reference

```json
"cluster_size_cutoff": ...
```

### Allow more independent colored states

Add more entries to:

```json
"cmap_original"
```

### Change the spatial zoom region

```json
"center_fractional"
```

or:

```json
"center_particle_index"
```

and:

```json
"size_fractional"
```

### Fix particles being cropped at the zoom image boundary

Increase:

```json
"zoom_padding_factor"
```

### Show no box in zoom

```json
"show_zoom_box": false
```

### Show no box anywhere

```json
"show_box": false,
"show_zoom_box": false
```

### Improve image quality

Increase:

```json
"pathtrace_samples"
```

and:

```json
"full_pixel_scale"
"zoom_pixel_scale"
```

### Make rendering faster while tuning

Decrease those same quality parameters.

### If symmetry classification looks wrong

Do **not** tune colors or camera.

Inspect and tune:

```text
shape geometry
symmetry_vertex_tolerance
or
tolerance_for_inv_quat_of_body_calc
```

### If clustering looks wrong

After verifying symmetry, tune:

```text
orientation_angle_tol
cluster_size_cutoff
palette length
```

---

# 116. Recommended command for the current project

From the directory containing all four files:

```bash
python3.8 render_sys_snapshot_from_gsd_orientational_color.py "param_file(20261007-121138).json"
```

If using a renamed simpler parameter file:

```bash
python3.8 render_sys_snapshot_from_gsd_orientational_color.py param_file.json
```

The program should finish with:

```text
Done.
```

and the results will be in:

```text
outputs_orientational_snapshot/
```

unless `output_dir` has been changed.

---

# 117. Closing note

The most important conceptual distinction in this project is that **particle symmetry, orientational clustering, final color assignment, spatial zoom selection, and rendering are separate stages**.

A visual problem should be fixed using a visual or zoom parameter.

A clustering problem should be fixed using the orientation-state parameters.

A symmetry problem should be fixed at the particle-geometry/symmetry-detection level.

A memory problem should be fixed using block size or RAM/memmap controls.

Keeping those layers separate is the safest way to adapt this code to arbitrary convex-polyhedral systems without unintentionally changing the underlying scientific interpretation.
