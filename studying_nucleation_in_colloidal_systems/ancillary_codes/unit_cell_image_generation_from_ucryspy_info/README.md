# Publication-Quality PyVista Unit-Cell Renderer

## Detailed README for `render_unit_cell_pyvista_v1p6.py`

This project renders a crystallographic/unit-cell-style arrangement of **oriented convex polyhedral particles** using **PyVista/VTK**. It reads particle positions and quaternions from a text file, reads the particle geometry from a JSON vertex file, reconstructs the particle using `scipy.spatial.ConvexHull`, validates the convex-polyhedron topology, places a copy of the particle at every listed particle center with the listed orientation, draws the unit-cell skeleton, generates multiple reproducible viewing angles, and saves **both opaque and transparent PNGs for every view**.

The code is designed for publication figures, debugging of orientational order, visualization of crystalline/disordered polyhedral packings, and any workflow in which a unit cell contains copies of the same convex particle with different orientations.

---

# 1. What this code does

At a high level, the program performs the following pipeline:

```text
unit-cell text file
    |
    |-- particle IDs
    |-- particle positions
    |-- particle quaternions
    |-- lattice vectors
    |-- optional metadata
    |
    v

shape JSON file
    |
    |-- body/principal-frame vertex coordinates
    |
    v

SciPy ConvexHull
    |
    |-- exact convex-hull triangles from hull.simplices
    |-- outward triangle orientation from hull.equations
    |-- physical polygonal-face identification
    |-- true physical-edge identification
    |-- Euler/topology validation
    |
    v

particle transformation
    |
    |-- independent isotropic particle scaling
    |-- quaternion rotation
    |-- translation to each particle center
    |
    v

unit-cell transformation
    |
    |-- independent isotropic cell scaling
    |-- consistent scaling of particle-center separations
    |
    v

PyVista scene
    |
    |-- polyhedron surfaces
    |-- true physical particle edges
    |-- unit-cell skeleton
    |-- publication lighting
    |-- orthographic or perspective camera
    |
    v

random/reproducible viewing directions
    |
    v

for every view:
    unit_cell_viewXX.png
    unit_cell_viewXX_transparent.png
```

The important design decision is that the rendered particle surface is taken **directly from `scipy.spatial.ConvexHull.simplices`**. The code does not manually reorder polygon vertices and does not manually reconstruct polygon faces for rendering. Coplanar-triangle grouping is used only for topology analysis and for separating true physical edges from triangulation diagonals.

---

# 2. Current source-code assumptions

The current implementation is intended for the following situation:

1. Every particle in the unit-cell file has the **same shape**.
2. The shape is a **convex three-dimensional polyhedron**.
3. The JSON file provides the particle vertices in the particle's body/principal frame.
4. The unit-cell text file provides each particle orientation as a quaternion.
5. Particle positions are already in the global/trajectory coordinate frame.
6. The three lattice vectors are also provided in the same global coordinate frame.
7. The particle shape is preferably centered at the origin before rotation.
8. Every vertex listed in the JSON is expected to belong to the convex hull.
9. The default quaternion convention is scalar-first:
   ```text
   [w, x, y, z]
   ```
10. The code renders all particle records present in the unit-cell text file. It does not automatically remove periodic duplicates or reduce the list to only symmetry-inequivalent/effective particles.

If any of these assumptions differs from your project, see **Section 18: How to adapt the source code**.

---

# 3. Files in a typical project directory

A minimal working directory may look like:

```text
project/
├── render_unit_cell_pyvista_v1p6.py
├── uc_info.txt
├── shape.json
└── README.md
```

After running the program with several views, the directory may contain:

```text
project/
├── render_unit_cell_pyvista_v1p6.py
├── uc_info.txt
├── shape.json
├── README.md
├── unit_cell_view01.png
├── unit_cell_view01_transparent.png
├── unit_cell_view02.png
├── unit_cell_view02_transparent.png
├── unit_cell_view03.png
├── unit_cell_view03_transparent.png
└── ...
```

You can use any file names you want. The input filenames are positional command-line arguments.

---

# 4. Dependencies

The source imports:

```python
numpy
scipy
pyvista
vtk
```

Install them with:

```bash
pip install numpy scipy pyvista vtk
```

or, if you explicitly use Python 3.8:

```bash
python3.8 -m pip install numpy scipy pyvista vtk
```

A Conda environment is also fine:

```bash
conda create -n unitcell_render python=3.8
conda activate unitcell_render
pip install numpy scipy pyvista vtk
```

The script uses off-screen rendering:

```python
pv.Plotter(off_screen=True, ...)
```

so an interactive PyVista window is not required for normal operation.

## Headless/HPC note

On some clusters, the installed VTK build still expects an OpenGL/X11 backend even when `off_screen=True`. If you encounter errors mentioning X11, OpenGL, GLX, EGL, or inability to create a render window, the Python code itself may be fine. The issue is then the VTK rendering backend.

Depending on the cluster configuration, typical solutions include:

```bash
xvfb-run -a python3.8 render_unit_cell_pyvista_v1p6.py ...
```

or using a VTK/PyVista build configured for EGL/OSMesa. The exact solution depends on the cluster installation.

---

# 5. Quick start

For the example project used during development:

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape_20_added_hexagonal_prism_both_side_hparam_0p2_eparam_0p7_unit_volume_principal_frame.json \
    --cell-scale 1.20 \
    --particle-scale 0.75 \
    --cell-line-width 48.0 \
    --nviews 2 \
    --random-seed 123 \
    --expected-num-faces 20 \
    --expected-num-edges 42
```

This produces:

```text
unit_cell_view01.png
unit_cell_view01_transparent.png
unit_cell_view02.png
unit_cell_view02_transparent.png
```

The opaque and transparent files for a given view use the **same camera, same geometry, same lighting, same zoom, and same resolution**. Only the background treatment differs.

To see every available command-line option:

```bash
python3.8 render_unit_cell_pyvista_v1p6.py --help
```

---

# 6. Input file 1: unit-cell information text file

The text parser is label-based rather than line-number-based. Therefore extra text in the file is normally harmless as long as the required recognizable fields are present.

## 6.1 Required particle record format

Each particle must appear in a line that can be matched by the code's regular expression:

```text
ID: 82 (Real) | Position: [ 0.6309, -1.5938, 1.5718 ] | Orientation: [ 0.4399, 0.708, -0.2096, -0.5111 ]
```

The essential pieces are:

```text
ID:
Position: [x, y, z]
Orientation: [q0, q1, q2, q3]
```

The parenthesized tag such as `(Real)` is optional.

The parser stores each record as:

```python
Particle(
    particle_id=...,
    position=np.ndarray(...),
    quaternion=np.ndarray(...),
    tag=...
)
```

### Position convention

The position is assumed to be the particle center in the **global frame**:

```text
Position: [x_global, y_global, z_global]
```

### Orientation convention

By default:

```text
Orientation: [w, x, y, z]
```

where `w` is the scalar quaternion component.

If your source instead stores:

```text
[x, y, z, w]
```

run with:

```bash
--quaternion-order xyzw
```

Do not change the orientation values themselves merely to make the figure look better. First determine the actual convention used by the trajectory/output software.

---

## 6.2 Required lattice-vector format

The file must contain a line recognizable as:

```text
Lattice vectors : [[a_x, a_y, a_z], [b_x, b_y, b_z], [c_x, c_y, c_z]]
```

For example:

```text
Lattice vectors :  [[-0.3815125525, 0.9385144711, -0.5266380310],
                    [-1.0528088808, 0.4007241130,  1.5119285583],
                    [ 1.9555534124, 0.0760942101, -1.2510128021]]
```

Internally the code stores:

```python
lattice_vectors.shape == (3, 3)
```

with the rows interpreted as:

```python
a = lattice_vectors[0]
b = lattice_vectors[1]
c = lattice_vectors[2]
```

This convention is important. If your data file stores lattice vectors as columns rather than rows, either transpose the matrix before writing the text file or modify `read_unit_cell_info()`.

---

## 6.3 Optional metadata recognized by the parser

The source also attempts to parse:

```text
Lattice parameters : [...]
Crystal class : ...
Spacegroup : ...
Number of effective particles : ...
Quaternion required to transform local to global frame : [...]
```

These values are stored in `UnitCellInfo`, but not all are currently used for rendering.

In particular:

```python
local_to_global_quaternion
number_effective_particles
lattice_parameters
spacegroup
crystal_class
```

are metadata fields.

`spacegroup` and `crystal_class` are printed if present.

The `Quaternion required to transform local to global frame` is parsed and stored, but the current rendering pipeline does **not** multiply each particle orientation by this quaternion. The program assumes the particle orientation values already represent the transformation needed to place the body-frame particle in the global frame.

If your future input file stores orientations in a local cell coordinate frame instead of the global frame, this part of the logic must be changed.

---

# 7. Input file 2: particle-shape JSON

The JSON parser searches for the first key whose name ends with:

```text
vertices
```

Therefore all of the following names would be recognized:

```json
"vertices"
"8_vertices"
"particle_vertices"
"my_vertices"
```

The current sample uses:

```json
"8_vertices": [
    [x1, y1, z1],
    [x2, y2, z2],
    ...
]
```

Only the vertex array is needed for the geometry construction.

Other metadata in the JSON may include volume, name, center of mass, moments of inertia, truncation parameters, etc. Those values are not currently used by the renderer.

---

## 7.1 Required vertex conditions

The code requires:

```text
shape = (N, 3)
N >= 4
all coordinates finite
no duplicate/nearly identical vertices
```

It also later requires that **every supplied JSON vertex be a convex-hull vertex**.

If some listed JSON points are interior points, the program will stop with:

```text
Not every JSON vertex belongs to the convex hull.
Interior/non-hull vertex indices: [...]
```

This is intentional. It prevents an input-data problem from being silently hidden.

---

## 7.2 Shape coordinates should be body-frame coordinates

The intended geometry is:

```text
body-frame vertices
    ↓ particle scale
body-frame scaled vertices
    ↓ quaternion rotation
global oriented vertices
    ↓ translation
particle at its unit-cell position
```

Therefore the JSON coordinates should ideally be centered around the particle center:

```text
center of mass ≈ [0, 0, 0]
```

If the JSON shape is not centered, the code will still rotate the coordinates mathematically, but the particle will rotate around the global coordinate origin of the JSON rather than around its desired geometrical center. That usually produces an apparent offset.

If necessary, preprocess the shape by subtracting a desired center:

```python
vertices = vertices - center
```

before using it.

---

# 8. Why SciPy `ConvexHull` is used

A major part of this project is ensuring that the displayed polyhedron is actually the correct convex polyhedron implied by the JSON vertices.

The current implementation uses:

```python
hull = ConvexHull(vertices, qhull_options="Qc")
```

The option:

```text
Qc
```

asks Qhull to retain coplanar information.

The code deliberately does **not** use `QJ`, because `QJ` perturbs input coordinates. For a well-defined nondegenerate convex particle, coordinate perturbation is not desired.

---

# 9. Convex-hull topology pipeline

The geometry construction follows these steps.

## 9.1 Read the vertices exactly

`read_shape_vertices()` returns the vertex array without sorting or replacing the vertices.

The code explicitly avoids changing the point order because connectivity refers to vertex indices.

---

## 9.2 Build the SciPy hull

The function:

```python
build_convex_particle_geometry()
```

calls:

```python
ConvexHull(vertices, qhull_options="Qc")
```

and obtains:

```python
hull.vertices
hull.simplices
hull.equations
hull.volume
hull.area
```

---

## 9.3 Render `hull.simplices` directly

The visible particle surface is the triangular hull returned by SciPy:

```python
hull.simplices
```

This is crucial.

The code does **not** take a polygonal face, sort its vertices by angle, and then rebuild a fan triangulation for the displayed surface. That earlier style of reconstruction can introduce incorrect connectivity if face ordering is wrong.

In this version:

```text
SciPy/Qhull determines the convex hull
→ Qhull triangles become the rendered surface
```

---

## 9.4 Correct triangle winding

Although `hull.simplices` gives the correct triangles, the code explicitly checks the triangle-normal direction against:

```python
hull.equations
```

Each plane equation is of the form:

```text
n · x + d = 0
```

where Qhull supplies an outward normal.

For every triangle:

```python
triangle_normal = cross(p1 - p0, p2 - p0)
```

If:

```python
dot(triangle_normal, outward_normal) < 0
```

the last two triangle indices are swapped.

This gives a consistent outward winding for lighting and surface normals.

---

# 10. Physical faces versus triangulated faces

A convex polyhedron may have polygonal faces such as quadrilaterals or hexagons.

Qhull triangulates them.

For example, a quadrilateral physical face generally becomes two triangular Qhull facets.

Therefore:

```text
number of hull triangles != number of physical polygon faces
```

The function:

```python
group_hull_triangles_into_physical_faces()
```

groups hull triangles that lie on the same plane.

Two normalized planes are treated as the same physical plane when their normals and offsets agree within:

```python
PLANE_NORMAL_TOL = 1.0e-9
PLANE_OFFSET_TOL = 1.0e-9
```

The sign-inverted plane equation is also treated as equivalent.

The resulting number of plane groups is reported as:

```text
Physical polygonal faces
```

---

# 11. True particle edges versus triangulation diagonals

A triangulated polygonal face contains internal diagonal lines that are **not** physical edges of the original polyhedron.

The current code explicitly removes these.

The function:

```python
extract_true_polyhedron_edges()
```

builds triangle-edge adjacency.

For every triangulated edge:

```text
if neighboring triangles belong to the SAME physical plane:
    edge = triangulation diagonal
    do not display it

if neighboring triangles belong to DIFFERENT physical planes:
    edge = true physical polyhedron edge
    display it
```

This is preferable to using a generic feature-angle threshold because it is based directly on verified hull topology.

The resulting true edges are drawn by:

```python
make_pyvista_edge_mesh()
```

and can be hidden with:

```bash
--no-particle-edges
```

---

# 12. Topology validation

For every input shape, the code reports:

```text
JSON vertices
Vertices used by convex hull
Triangular Qhull facets
Physical polygonal faces
True physical edges
Euler V-E+F
Convex-hull volume
Convex-hull surface area
Physical-face size distribution
```

For the example development shape, a successful report is:

```text
JSON vertices                 : 24
Vertices used by convex hull  : 24
Triangular Qhull facets       : 44
Physical polygonal faces      : 20
True physical edges           : 42
Euler V-E+F                   : 2
Convex-hull volume            : 1
...
Physical-face size distribution:
  18 face(s) with 4 vertices
  2 face(s) with 6 vertices
```

Those values are **not hard-coded into the geometry algorithm**.

For a different valid convex polyhedron, the calculated values can be different.

---

## 12.1 Optional strict face-count validation

If you know the correct number of physical faces, pass:

```bash
--expected-num-faces 20
```

or equivalently:

```bash
--num-faces 20
```

If the calculated count is not 20, rendering stops.

If you omit this argument, the program calculates and reports the face count but does not enforce a particular value.

You can also explicitly disable the check with:

```bash
--expected-num-faces 0
```

---

## 12.2 Optional strict edge-count validation

Likewise:

```bash
--expected-num-edges 42
```

or:

```bash
--num-edges 42
```

validates the number of true physical edges.

Again, omitting the option keeps the code general.

---

## 12.3 Euler validation

For a closed convex polyhedron, the code always checks:

```text
V - E + F = 2
```

where:

```text
V = number of hull vertices
E = number of true physical edges
F = number of physical polygonal faces
```

If the result is not 2, rendering is stopped.

This is a topology sanity check.

---

# 13. Important limitation: the particle must be convex

This is one of the most important limitations of the current project.

`scipy.spatial.ConvexHull` calculates the **convex hull** of the supplied points.

Therefore, if the intended shape is concave:

```text
true concave shape
    ≠
ConvexHull(vertex cloud)
```

Any indentation/cavity will be filled by the convex hull.

Do not use the current geometry builder for a genuinely concave particle unless your scientific intention is specifically to render its convex hull.

For a concave particle you should instead supply explicit face connectivity or perform a proper convex decomposition into multiple convex pieces and render all pieces.

See **Section 18.14**.

---

# 14. Quaternion rotation convention

The code implements an active quaternion rotation.

By default:

```text
q = [w, x, y, z]
```

The quaternion is normalized before constructing the rotation matrix.

The transformation is conceptually:

```text
r_global = R(q) r_body
```

Because the vertex array is stored as row vectors, the Python implementation is:

```python
rotated_vertices = scaled_vertices @ rotation.T
```

After rotation:

```python
global_vertices = rotated_vertices + displayed_center
```

---

## 14.1 If the particles look correctly shaped but incorrectly oriented

The most likely causes are:

1. wrong quaternion component order;
2. using a passive instead of active rotation convention;
3. quaternion represents global-to-body rather than body-to-global;
4. additional local-to-global cell rotation is required;
5. trajectory software uses a different quaternion multiplication convention.

First try:

```bash
--quaternion-order xyzw
```

instead of the default:

```bash
--quaternion-order wxyz
```

If the difference is not simply component order, inspect:

```python
quaternion_to_rotation_matrix()
transform_particle_vertices()
```

Do not modify the hull code when the shape itself is correct but only the orientation is wrong.

---

# 15. Unit-cell origin

The code needs an origin \(O\) because cell scaling is performed about that point.

There are two modes.

## 15.1 Explicit origin

If you know which particle center corresponds to the unit-cell origin/corner:

```bash
--origin-id 82
```

The code uses that particle's position directly.

This is the most deterministic mode.

---

## 15.2 Automatic origin inference

If `--origin-id` is omitted, the code tries to infer a unit-cell corner from the listed particle positions.

For every candidate particle center \(p_i\), it constructs the eight ideal corners:

```text
p_i
p_i + a
p_i + b
p_i + c
p_i + a + b
p_i + a + c
p_i + b + c
p_i + a + b + c
```

It then computes the distances between those ideal corners and all listed particle positions.

The Hungarian assignment algorithm:

```python
scipy.optimize.linear_sum_assignment
```

is used to assign the eight ideal corners to eight distinct listed positions with minimum total cost.

The candidate with the lowest RMS residual becomes the inferred origin.

The program prints something like:

```text
Cell origin: [ ... ]
Origin selection: automatically inferred from particle ID 82;
corner RMS residual=...,
max residual=...
```

### Requirement

Automatic inference requires at least eight listed positions.

If your unit-cell text file contains fewer than eight particle records, use:

```bash
--origin-id <ID>
```

---

# 16. Independent cell and particle scaling

The renderer intentionally separates:

```text
unit-cell linear scaling
particle linear scaling
```

This is extremely useful when the true dense packing causes projected particles to visually overlap.

---

## 16.1 Cell scaling

Use:

```bash
--cell-scale S
```

For example:

```bash
--cell-scale 1.20
```

The displayed lattice vectors become:

\[
\mathbf a' = S_\mathrm{cell}\mathbf a,
\]

\[
\mathbf b' = S_\mathrm{cell}\mathbf b,
\]

\[
\mathbf c' = S_\mathrm{cell}\mathbf c.
\]

Particle centers are also moved consistently:

\[
\mathbf r_i'
=
\mathbf O
+
S_\mathrm{cell}
(\mathbf r_i-\mathbf O).
\]

This means the particle fractional position relative to the chosen origin is preserved while the lattice is visually expanded or contracted.

### Examples

```text
--cell-scale 1.00
original center separations

--cell-scale 1.20
20% larger linear cell dimensions

--cell-scale 0.90
10% smaller linear cell dimensions
```

The cell volume multiplier is:

\[
S_\mathrm{cell}^3.
\]

---

## 16.2 Particle scaling

Use:

```bash
--particle-scale S
```

For example:

```bash
--particle-scale 0.75
```

This changes only the body-frame vertex coordinates:

\[
\mathbf v' = S_\mathrm{particle}\mathbf v.
\]

The quaternion and particle center are not changed by this factor.

### Examples

```text
--particle-scale 1.00
original particle size

--particle-scale 0.75
particle linear dimensions are 75% of original

--particle-scale 1.20
particle linear dimensions are 120% of original
```

The particle volume multiplier is:

\[
S_\mathrm{particle}^3.
\]

---

## 16.3 Relative displayed packing fraction

The code prints:

```text
Relative particle/cell volume multiplier
```

which is:

\[
\left(
\frac{S_\mathrm{particle}}
     {S_\mathrm{cell}}
\right)^3.
\]

For:

```text
cell scale     = 1.20
particle scale = 0.75
```

the relative visual packing multiplier is:

\[
(0.75/1.20)^3.
\]

These scaling parameters are visualization controls. They should not automatically be interpreted as physical changes to the simulated system unless that is specifically your intention.

---

# 17. Camera generation

The default views are not arbitrary global-\(x,y,z\) views. The code builds a coordinate basis from the displayed unit-cell geometry.

## 17.1 Cell-derived camera basis

The function:

```python
compute_cell_camera_basis()
```

takes the eight displayed cell corners, centers them, and performs:

```python
np.linalg.svd(...)
```

The right singular vectors provide an orthonormal geometric basis:

```text
e1, e2, e3
```

Signs are chosen consistently with the lattice vectors.

This makes the camera construction less sensitive to how the entire cell is oriented in the simulation's global Cartesian frame.

---

## 17.2 Random azimuth

For every view:

```python
azimuth = rng.uniform(0.0, 2.0 * np.pi)
```

Thus azimuth covers the full \(0^\circ\) to \(360^\circ\) range.

---

## 17.3 Random elevation

Currently:

```python
elevation = np.deg2rad(
    rng.uniform(18.0, 62.0)
)
```

Therefore random elevations are between:

```text
18 degrees and 62 degrees
```

This avoids extremely flat views and extremely top-down views.

To change this range, edit `generate_random_camera_triplets()`.

For example, for a broader elevation range:

```python
elevation = np.deg2rad(
    rng.uniform(5.0, 80.0)
)
```

For mostly side-on views:

```python
elevation = np.deg2rad(
    rng.uniform(5.0, 30.0)
)
```

---

## 17.4 Camera distance

The camera is placed at:

```python
camera_position = center + 4.0 * radius * direction
```

The coefficient:

```text
4.0
```

controls camera distance.

For orthographic projection, apparent object size is mainly determined later by camera fitting and zoom, so changing this usually matters less than in perspective mode.

For perspective mode, increasing this number generally moves the camera farther away.

---

## 17.5 Reproducibility

The random-number generator is:

```python
np.random.default_rng(seed)
```

Therefore:

```bash
--random-seed 123
```

always generates the same camera sequence for the same geometry and code version.

This is useful when a particular view is selected for a publication.

If you run:

```bash
--nviews 10 --random-seed 123
```

today and repeat it later, view 07 will correspond to the same pseudorandom camera sequence.

---

# 18. How to modify the source code for different needs

This section is the main customization reference.

---

## 18.1 Change particle color

At the top of the script:

```python
PARTICLE_COLOR = "#0080ff"
```

Replace it with any PyVista-compatible color.

Examples:

```python
PARTICLE_COLOR = "#ff8000"
PARTICLE_COLOR = "#7b2cbf"
PARTICLE_COLOR = "red"
PARTICLE_COLOR = "steelblue"
```

The current project requirement is:

```text
#0080ff
```

---

## 18.2 Change particle-edge color

Modify:

```python
PARTICLE_EDGE_COLOR = "#17324d"
```

For black edges:

```python
PARTICLE_EDGE_COLOR = "black"
```

For a darker version of the particle color, use another hex value.

---

## 18.3 Change unit-cell skeleton color

Modify:

```python
CELL_EDGE_COLOR = "#202020"
```

Examples:

```python
CELL_EDGE_COLOR = "black"
CELL_EDGE_COLOR = "#555555"
```

---

## 18.4 Change opaque background color

Modify:

```python
BACKGROUND_COLOR = "white"
```

Examples:

```python
BACKGROUND_COLOR = "black"
BACKGROUND_COLOR = "#f5f5f5"
BACKGROUND_COLOR = "lightgray"
```

The transparent partner images are still saved with transparency.

---

## 18.5 Change default image resolution

Current default:

```python
DEFAULT_IMAGE_SIZE = (3000, 3000)
```

You can either modify this constant or use the command line:

```bash
--width 4000 --height 4000
```

For a landscape image:

```bash
--width 4500 --height 3000
```

For a portrait image:

```bash
--width 3000 --height 4500
```

The plotter uses:

```python
window_size=(width, height)
```

so these are actual render dimensions.

---

## 18.6 Change unit-cell skeleton thickness

You normally do not need to edit the source.

Use:

```bash
--cell-line-width 48.0
```

The default is set by:

```python
DEFAULT_CELL_LINE_WIDTH = 8.0
```

and applied in:

```python
plotter.add_mesh(
    cell_edge_mesh,
    line_width=float(cell_line_width),
    ...
)
```

### Note about very thick lines

OpenGL/VTK line-width support can depend on the graphics backend. Extremely large widths may not scale identically on every machine.

If a requested large width appears capped, this may be a renderer/backend limitation rather than a Python logic problem.

A backend-independent alternative would be to render each unit-cell edge as a thin cylinder instead of an OpenGL line. That would require modifying `build_cell_edge_mesh()`/scene construction.

---

## 18.7 Change particle-edge thickness

Use:

```bash
--particle-edge-width 2.5
```

or modify the default:

```python
particle_edge_width: float = 1.5
```

in the main renderer/parser.

---

## 18.8 Remove particle edges

Run:

```bash
--no-particle-edges
```

This leaves only the particle surfaces.

Do not remove the topology calculation itself unless you also want to disable edge-count validation.

---

## 18.9 Change surface material appearance

Inside:

```python
add_all_particles()
```

the surface is added with:

```python
plotter.add_mesh(
    surface_mesh,
    color=PARTICLE_COLOR,
    smooth_shading=False,
    ambient=0.24,
    diffuse=0.76,
    specular=0.22,
    specular_power=30.0,
    show_edges=False,
)
```

### More matte

For a flatter/matte particle:

```python
ambient=0.30
diffuse=0.70
specular=0.05
specular_power=10.0
```

### More glossy

For stronger highlights:

```python
ambient=0.15
diffuse=0.70
specular=0.50
specular_power=50.0
```

### Smooth shading

Current:

```python
smooth_shading=False
```

is intentional for faceted polyhedra.

Setting:

```python
smooth_shading=True
```

will visually smooth normals across neighboring triangles and usually makes a polyhedron look less geometrically sharp. Use it only if that is the desired artistic effect.

---

## 18.10 Change the lighting

The function:

```python
add_publication_lights()
```

defines three lights.

Current positions relative to scene radius:

```python
[+3.5, +2.0, +4.0] * r
[-3.0, +1.5, +2.0] * r
[+0.5, -3.5, +3.0] * r
```

Current intensities:

```text
0.80
0.45
0.35
```

To make the image flatter, reduce contrast between key and fill lights.

To make it more dramatic, increase the primary light and reduce fill intensity.

For example:

```python
specifications = [
    (focal_point + np.array([+3.5, +2.0, +4.0]) * r, 1.00),
    (focal_point + np.array([-3.0, +1.5, +2.0]) * r, 0.25),
    (focal_point + np.array([+0.5, -3.5, +3.0]) * r, 0.20),
]
```

All current lights are:

```python
color="white"
positional=True
```

---

## 18.11 Change the number of views

Normally use:

```bash
--nviews 20
```

Current default:

```python
DEFAULT_N_RANDOM_VIEWS = 10
```

Change the constant if you want a different default without specifying the flag every time.

Remember that every view generates two PNGs.

Therefore:

```text
--nviews 10
```

creates:

```text
20 PNG files
```

---

## 18.12 Change random camera distribution

Edit:

```python
generate_random_camera_triplets()
```

Current azimuth:

```python
rng.uniform(0.0, 2.0 * np.pi)
```

Current elevation:

```python
rng.uniform(18.0, 62.0)
```

Current camera distance:

```python
4.0 * radius
```

These three lines determine most of the random camera distribution.

---

## 18.13 Use one fixed deterministic camera instead of random views

The present code intentionally generates random/reproducible cameras.

For a fixed crystallographic camera, replace or bypass:

```python
generate_random_camera_triplets()
```

For example, after obtaining:

```python
center, e1, e2, e3 = compute_cell_camera_basis(...)
```

you could define:

```python
direction = normalize(e1 + e2 + e3)
camera_position = center + 4.0 * radius * direction
view_up = e3
```

and use one camera tuple.

For a view along a lattice direction, use a normalized lattice vector instead.

Example concept:

```python
direction = normalize(scaled_lattice_vectors[2])
```

Be careful with `view_up`: it cannot be parallel to the viewing direction.

A robust procedure is:

```python
view_up = trial_up - np.dot(trial_up, direction) * direction
view_up = normalize(view_up)
```

---

## 18.14 Support a concave particle

Do **not** try to fix a concave particle by changing Qhull tolerances.

The architecture must change.

For a concave shape, possible approaches are:

### A. Explicit face connectivity

Have the JSON include:

```json
{
  "vertices": [...],
  "faces": [
    [0, 1, 2, 3],
    [4, 5, 6],
    ...
  ]
}
```

Then replace:

```python
build_convex_particle_geometry()
```

with a function that constructs PyVista faces directly from the supplied connectivity.

### B. Convex decomposition

Represent one concave particle as several convex components:

```text
particle
├── convex component 1
├── convex component 2
└── convex component 3
```

Build a hull for each component and apply the same particle quaternion and translation to every component.

### C. Triangular surface mesh input

Read an STL/OBJ/PLY mesh instead of reconstructing from a vertex cloud.

In that case, the topology and face connectivity come from the mesh file.

---

## 18.15 Allow interior JSON points

Currently this check is strict:

```python
if len(hull_vertex_ids) != len(vertices):
    raise RuntimeError(...)
```

If your JSON intentionally contains interior points, you could remove this strict check.

However, then remember:

```text
JSON point count != hull vertex count
```

and any interior points do not contribute to the displayed hull.

A safer design is to keep the check by default and add a command-line flag such as:

```bash
--allow-interior-points
```

if this becomes a real use case.

---

## 18.16 Change coplanar-face tolerances

Current values:

```python
PLANE_NORMAL_TOL = 1.0e-9
PLANE_OFFSET_TOL = 1.0e-9
```

These determine whether neighboring hull triangles are treated as belonging to the same physical polygonal face.

Do not change these merely because the picture looks wrong.

Only consider changing them if topology diagnostics show that one geometrically planar face has been split into several nearly identical planes due to numerical noise.

For noisy coordinates you might try:

```python
PLANE_NORMAL_TOL = 1.0e-7
PLANE_OFFSET_TOL = 1.0e-7
```

but increasing tolerances too far can incorrectly merge distinct faces with small dihedral angles.

Always verify:

```text
face count
edge count
Euler V-E+F
face-size distribution
```

after changing these values.

---

## 18.17 Change quaternion convention

Command-line solution:

```bash
--quaternion-order wxyz
```

or:

```bash
--quaternion-order xyzw
```

If your convention is more fundamentally different, modify:

```python
quaternion_to_rotation_matrix()
```

For example, if your stored quaternion is the inverse rotation, a normalized unit quaternion can be conjugated:

```text
[w, x, y, z]
→
[w, -x, -y, -z]
```

before constructing the matrix.

Do this only if you have verified the convention used by the software that wrote the quaternion.

---

## 18.18 Apply an additional local-to-global rotation

The text parser can read:

```text
Quaternion required to transform local to global frame
```

but the current renderer does not use it in particle placement.

If your particle orientation is defined in a local unit-cell frame, you may need a combined rotation such as:

```text
R_global = R_cell_to_global @ R_particle_local
```

The exact multiplication order depends on your convention.

A clean place to implement this is inside:

```python
transform_particle_vertices()
```

or before it, by combining quaternion-derived rotation matrices.

Do not blindly multiply quaternions without determining the correct active/passive convention and multiplication order.

---

## 18.19 Change output naming

Output naming occurs in the save loop inside:

```python
render_unit_cell_multi_view()
```

For multiple views:

```python
opaque_output = output_png.with_name(
    f"{output_png.stem}_view{view_number:02d}"
    f"{output_png.suffix}"
)
```

Transparent partner:

```python
transparent_output = opaque_output.with_name(
    f"{opaque_output.stem}_transparent"
    f"{opaque_output.suffix}"
)
```

To use directories:

```text
opaque/
transparent/
```

modify this block to construct two output directories.

---

## 18.20 Stop generating transparent copies

In the current version, every view is deliberately saved twice.

The two calls are:

```python
plotter.screenshot(
    str(opaque_output),
    transparent_background=False,
    ...
)
```

and:

```python
plotter.screenshot(
    str(transparent_output),
    transparent_background=True,
    ...
)
```

To save only opaque images, remove or comment the second call and corresponding `saved_paths.append()`/print block.

To save only transparent images, remove the first call and rename the second output if desired.

---

## 18.21 Change transparent-background behavior

PyVista transparency is controlled by:

```python
transparent_background=True
```

The scene geometry is not reconstructed between the opaque and transparent screenshots. Therefore both files use the same camera and object geometry.

If a transparent PNG appears to have a black background in a particular image viewer, first verify that the file actually contains an alpha channel. Some viewers display transparent pixels as black.

---

## 18.22 Show particle IDs

Use:

```bash
--labels
```

The label code is inside:

```python
add_all_particles()
```

Current label appearance:

```python
font_size=14
text_color="black"
shape=None
always_visible=True
```

Change those values there.

---

## 18.23 Change the unit-cell text format

The parsing logic is in:

```python
read_unit_cell_info()
```

Particle records are detected using a regular expression that expects labels equivalent to:

```text
ID:
Position:
Orientation:
```

If your upstream code writes:

```text
ParticleID =
r =
quat =
```

you must modify the regular expression.

Likewise, lattice vectors are found by searching for:

```text
Lattice vectors :
```

If your format differs, change the corresponding `re.search()`.

The parser is otherwise intentionally not dependent on exact line numbers.

---

## 18.24 Change the JSON vertex key

Current logic:

```python
for key in data:
    if str(key).lower().endswith("vertices"):
        vertex_key = key
        break
```

If your JSON has a completely different structure such as:

```json
{
  "geometry": {
    "points": [...]
  }
}
```

modify:

```python
read_shape_vertices()
```

to access:

```python
data["geometry"]["points"]
```

---

## 18.25 Use different shapes for different particles

The current code builds one:

```python
ConvexParticleGeometry
```

and applies it to every particle.

To support particle-specific shapes, change the data model so each particle record can select a geometry.

For example:

```python
geometry_by_type = {
    "A": geometry_A,
    "B": geometry_B,
}
```

Then in `add_all_particles()`:

```python
geometry = geometry_by_type[particle.type]
```

The unit-cell input format would also need a particle-type field.

---

## 18.26 Use different colors for different particles

Currently:

```python
color=PARTICLE_COLOR
```

is common to every particle.

To color by particle ID:

```python
particle_color = color_map[particle.particle_id]
```

then:

```python
plotter.add_mesh(
    surface_mesh,
    color=particle_color,
    ...
)
```

To color by orientation class, first calculate/classify an orientation state and use that state as the color-map key.

---

## 18.27 Draw only selected particle IDs

At the start of the loop in:

```python
add_all_particles()
```

add:

```python
allowed_ids = {82, 2766, 1794}

for particle in info.particles:
    if particle.particle_id not in allowed_ids:
        continue
```

For a general reusable implementation, expose the IDs through a new command-line argument.

---

## 18.28 Draw only effective particles instead of periodic copies

The input text may contain many periodic images even when the number of effective particles in the unit cell is smaller.

The current renderer draws **every particle record**.

To draw only effective particles, you need a reliable rule for deciding which records belong to the chosen half-open cell, for example fractional coordinates:

\[
0 \le f_a < 1,\quad
0 \le f_b < 1,\quad
0 \le f_c < 1.
\]

A robust implementation would:

1. determine the chosen cell origin;
2. form the lattice matrix;
3. transform each global center to fractional coordinates;
4. wrap or test the fractional coordinates;
5. keep one representative per periodic equivalence class.

Do not simply delete particles by ID unless the IDs are guaranteed to be stable across trajectories.

---

## 18.29 Add spheres at lattice points

If you want the unit-cell vertices themselves to be visibly marked, after constructing `cell_corners` you can add spheres:

```python
for corner in cell_corners:
    sphere = pv.Sphere(
        radius=0.03,
        center=corner,
    )
    plotter.add_mesh(
        sphere,
        color="black",
    )
```

The radius should be scaled relative to your cell dimensions.

---

## 18.30 Replace line skeleton with cylinders

For guaranteed thick unit-cell rods, construct a cylinder between every pair of corner points rather than relying on OpenGL line width.

This is useful when:

```bash
--cell-line-width 48
```

is ignored or capped by the graphics driver.

A helper can take two endpoints:

```python
p0
p1
```

and create a cylinder centered at:

```python
0.5 * (p0 + p1)
```

with direction:

```python
normalize(p1 - p0)
```

and height:

```python
norm(p1 - p0)
```

This produces physically thick 3D rods rather than screen-space lines.

---

# 19. Command-line reference

## Positional arguments

### `unit_cell_file`

Example:

```bash
uc_info.txt
```

Contains particle centers/orientations and lattice vectors.

### `shape_json_file`

Example:

```bash
shape.json
```

Contains the particle vertex coordinates.

---

## Optional arguments

### `-o`, `--output`

Default:

```text
unit_cell.png
```

Example:

```bash
-o figure.png
```

For multiple views:

```text
figure_view01.png
figure_view01_transparent.png
...
```

For one view:

```text
figure.png
figure_transparent.png
```

---

### `--width`

Default:

```text
3000
```

Example:

```bash
--width 4000
```

---

### `--height`

Default:

```text
3000
```

Example:

```bash
--height 4000
```

---

### `--cell-scale`

Default:

```text
1.0
```

Example:

```bash
--cell-scale 1.20
```

Scales lattice vectors and center separations.

---

### `--particle-scale`

Default:

```text
1.0
```

Example:

```bash
--particle-scale 0.75
```

Scales particle geometry.

---

### `--cell-line-width`

Default:

```text
8.0
```

Example:

```bash
--cell-line-width 48.0
```

Controls unit-cell skeleton width.

---

### `--particle-edge-width`

Default:

```text
1.5
```

Example:

```bash
--particle-edge-width 3.0
```

Controls true polyhedron-edge width.

---

### `--nviews`

Default:

```text
10
```

Example:

```bash
--nviews 2
```

Number of random camera directions.

Remember: number of PNGs generated is normally:

```text
2 × nviews
```

because both opaque and transparent files are saved.

---

### `--random-seed`

Default:

```text
0
```

Example:

```bash
--random-seed 123
```

Controls reproducible random views.

---

### `--expected-num-faces`, `--num-faces`

Default:

```text
None
```

Example:

```bash
--expected-num-faces 20
```

Strictly validates the physical face count.

---

### `--expected-num-edges`, `--num-edges`

Default:

```text
None
```

Example:

```bash
--expected-num-edges 42
```

Strictly validates the true edge count.

---

### `--origin-id`

Default:

```text
None
```

When omitted, origin inference is attempted.

Example:

```bash
--origin-id 82
```

---

### `--quaternion-order`

Choices:

```text
wxyz
xyzw
```

Default:

```text
wxyz
```

Example:

```bash
--quaternion-order xyzw
```

---

### `--zoom`

Default:

```text
1.12
```

Example:

```bash
--zoom 1.25
```

Values greater than 1 make the rendered structure occupy more of the image.

If objects are being clipped or nearly touch the image boundary, reduce the zoom.

---

### `--perspective`

Default behavior is orthographic projection.

Add:

```bash
--perspective
```

to use perspective projection.

Orthographic projection is usually preferable for crystallographic/structural figures because apparent size does not change with depth.

Perspective may look more natural for presentation graphics.

---

### `--no-particle-edges`

Suppresses physical particle edges:

```bash
--no-particle-edges
```

---

### `--labels`

Shows particle IDs:

```bash
--labels
```

Useful for debugging which record is located where.

---

# 20. Example commands

## 20.1 Minimal run

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape.json
```

Uses default scaling, 10 views, default line widths, orthographic projection, and no strict topology counts.

---

## 20.2 Strict topology validation

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape.json \
    --expected-num-faces 20 \
    --expected-num-edges 42
```

---

## 20.3 Expand cell and shrink particles

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape.json \
    --cell-scale 1.20 \
    --particle-scale 0.75
```

Useful for eliminating visual overlap.

---

## 20.4 Very thick unit-cell skeleton

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape.json \
    --cell-line-width 48.0
```

---

## 20.5 Two reproducible views

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape.json \
    --nviews 2 \
    --random-seed 123
```

---

## 20.6 One view only

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape.json \
    --nviews 1 \
    -o final_unit_cell.png
```

Produces:

```text
final_unit_cell.png
final_unit_cell_transparent.png
```

---

## 20.7 High-resolution landscape figure

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape.json \
    --width 5000 \
    --height 3200 \
    --nviews 4
```

---

## 20.8 Perspective rendering

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape.json \
    --perspective
```

---

## 20.9 Debug with IDs visible

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape.json \
    --labels \
    --nviews 1
```

---

## 20.10 Scalar-last quaternion input

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape.json \
    --quaternion-order xyzw
```

---

# 21. Understanding the console output

A normal run starts by printing the parsed structural information:

```text
Read 10 particle records.
Lattice vectors (rows = a,b,c):
[[...]
 [...]
 [...]]

Space group: ...
Crystal class: ...
```

Then the convex-hull diagnostics:

```text
============================================================
Convex-hull topology validation
============================================================
JSON vertices                 : ...
Vertices used by convex hull  : ...
Triangular Qhull facets       : ...
Physical polygonal faces      : ...
True physical edges           : ...
Euler V-E+F                   : ...
Convex-hull volume            : ...
Convex-hull surface area      : ...
Physical-face size distribution:
...
Face-count validation         : ...
Edge-count validation         : ...
============================================================
```

Then unit-cell origin diagnostics:

```text
Cell origin: [...]
Origin selection: ...
```

Then visualization scaling:

```text
Unit-cell linear scale factor : ...
Particle linear scale factor  : ...
Unit-cell volume multiplier   : ...
Particle volume multiplier    : ...
Relative particle/cell volume multiplier : ...
```

Finally each image pair:

```text
Saved opaque view 01/2: ...
Saved transparent view 01/2: ...
Saved opaque view 02/2: ...
Saved transparent view 02/2: ...
```

---

# 22. Function-by-function source-code map

This section is useful when modifying the project.

## `normalize()`

Purpose:

```text
normalize a 3D vector
```

Used in camera construction and related geometry.

Modify only if you need special handling of nearly-zero vectors.

---

## `parse_numeric_vector()`

Purpose:

```text
parse comma/space separated numerical text
```

Used by the unit-cell text parser.

---

## `read_unit_cell_info()`

Purpose:

```text
read particle IDs
read positions
read quaternions
read lattice vectors
read optional metadata
```

Modify this if the upstream unit-cell text format changes.

---

## `read_shape_vertices()`

Purpose:

```text
load JSON
find vertex key
validate N×3 coordinates
reject duplicate vertices
```

Modify this if the shape-file structure changes.

---

## `planes_are_same()`

Purpose:

```text
decide whether two Qhull facet plane equations represent one physical face
```

Modify tolerances carefully.

---

## `group_hull_triangles_into_physical_faces()`

Purpose:

```text
cluster coplanar Qhull triangles
```

Used for topology diagnostics and true-edge extraction.

It does not reconstruct the rendered surface.

---

## `orient_hull_triangles_outward()`

Purpose:

```text
make triangle winding consistent with Qhull outward normals
```

Usually should not need modification.

---

## `extract_true_polyhedron_edges()`

Purpose:

```text
remove triangulation diagonals
retain physical polyhedron edges
```

Important for clean publication figures.

---

## `physical_face_vertex_counts()`

Purpose:

```text
report how many vertices belong to each physical polygonal face
```

Used for diagnostics such as:

```text
18 quadrilaterals
2 hexagons
```

---

## `build_convex_particle_geometry()`

Purpose:

```text
central convex-hull builder
central topology validator
```

This is the main function to replace if changing from convex-hull geometry to explicit/concave geometry.

---

## `make_pyvista_surface_mesh()`

Purpose:

```text
convert verified triangle connectivity to PyVista PolyData
```

It deliberately does not call `clean()` because vertex indexing should remain unchanged.

---

## `make_pyvista_edge_mesh()`

Purpose:

```text
convert verified true-edge connectivity to PyVista line cells
```

---

## `quaternion_to_rotation_matrix()`

Purpose:

```text
quaternion → 3×3 active rotation matrix
```

Modify this only for orientation-convention changes.

---

## `transform_particle_vertices()`

Purpose:

```text
particle scaling
→ quaternion rotation
→ translation
```

This is the main place to add an additional rotation or local-coordinate transformation.

---

## `generate_cell_corners()`

Purpose:

```text
construct all 8 corners of the parallelepiped
```

---

## `infer_cell_origin()`

Purpose:

```text
automatic origin detection using ideal cell corners + Hungarian assignment
```

Modify if your unit-cell particle list is generated in a fundamentally different way.

---

## `get_cell_origin()`

Purpose:

```text
select explicit origin if --origin-id supplied
otherwise call automatic inference
```

---

## `scale_position_about_origin()`

Purpose:

```text
move particle center consistently with cell-scale
```

Implements:

\[
r' = O + s(r-O).
\]

---

## `build_cell_edge_mesh()`

Purpose:

```text
create 12 unit-cell skeleton line segments
```

Replace with cylinder construction here/near here if very thick physical rods are required.

---

## `compute_cell_camera_basis()`

Purpose:

```text
derive a reproducible orthonormal viewing basis from displayed cell geometry
```

---

## `generate_random_camera_triplets()`

Purpose:

```text
generate random azimuth/elevation cameras from a seed
```

Modify this to change camera distributions or use fixed viewpoints.

---

## `add_publication_lights()`

Purpose:

```text
configure the three-light rendering setup
```

Modify this for lighting style.

---

## `create_plotter()`

Purpose:

```text
create off-screen PyVista renderer
set background
enable SSAA, with MSAA fallback
```

Modify this for global render behavior.

---

## `add_all_particles()`

Purpose:

```text
loop over every particle
scale center
rotate/translate geometry
add surface
add true edges
optionally add ID labels
```

This is the central function for particle-dependent color/shape/style changes.

---

## `render_unit_cell_multi_view()`

Purpose:

```text
orchestrate the entire workflow
parse files
validate geometry
choose origin
scale cell
build scene
generate cameras
save opaque and transparent images
```

Modify this for output-directory handling or alternative rendering workflows.

---

## `build_argument_parser()`

Purpose:

```text
define command-line interface
```

Whenever you add a new user-tunable option, add it here.

---

## `main()`

Purpose:

```text
validate CLI values
forward arguments to renderer
```

Add validation here for any new command-line parameter.

---

# 23. Troubleshooting

## Problem: polyhedron looks grossly incorrect

Check the console topology first.

For a shape whose expected topology is known, pass:

```bash
--expected-num-faces F
--expected-num-edges E
```

If these fail, the problem is in shape geometry/topology, not lighting or camera.

Also check:

```text
Vertices used by convex hull == JSON vertices
Euler V-E+F == 2
```

If the topology is correct but the particle appears wrong, verify that the JSON really represents the intended convex particle and that the shape is not concave.

---

## Problem: shape is correct but orientation is wrong

Try:

```bash
--quaternion-order xyzw
```

If still wrong, investigate active/passive orientation convention or whether the quaternion maps global-to-body instead of body-to-global.

---

## Problem: particles overlap

Increase:

```bash
--cell-scale
```

and/or decrease:

```bash
--particle-scale
```

Example:

```bash
--cell-scale 1.30 --particle-scale 0.70
```

---

## Problem: unit-cell box does not appear to surround expected particles

Possible causes:

1. wrong inferred origin;
2. particle list contains unusual periodic copies;
3. lattice vectors use a different coordinate convention;
4. lattice vectors are columns rather than rows.

Try explicitly setting:

```bash
--origin-id <known corner particle>
```

---

## Problem: automatic origin inference fails

If fewer than eight positions are listed, provide:

```bash
--origin-id ID
```

If more than eight are present but the candidate corner particles are absent, automatic inference may also be inappropriate.

---

## Problem: `Not every JSON vertex belongs to the convex hull`

Your JSON includes one or more interior/non-extreme points.

Either fix the shape file or deliberately modify the strict validation if interior points are scientifically intended.

---

## Problem: duplicate-vertex error

The JSON contains identical/nearly identical coordinates.

Remove duplicate points upstream.

Do not simply weaken the duplicate tolerance unless the points are truly intended to be distinct.

---

## Problem: face count is too high

Coplanar facets may not be merging because the coordinates contain noise.

Inspect:

```python
PLANE_NORMAL_TOL
PLANE_OFFSET_TOL
```

but change them only after confirming the issue.

---

## Problem: face count is too low

Plane tolerances may be too permissive, incorrectly merging distinct faces.

Reduce the tolerances.

---

## Problem: edge count is larger than expected

This often follows from an incorrect physical-face grouping. If a planar polygon is split into multiple face IDs, internal triangulation edges can be misclassified as true edges.

Inspect face count and tolerances first.

---

## Problem: unit-cell lines are not as thick as requested

Some OpenGL implementations cap line width.

If this is important, replace skeleton lines with cylinders.

---

## Problem: transparent PNG looks black

The viewer may render transparency against black.

Open the image in software that displays alpha correctly or place it over a colored background in a graphics editor.

---

## Problem: output is clipped

Reduce:

```bash
--zoom
```

For example:

```bash
--zoom 1.02
```

or increase image dimensions.

---

## Problem: too much empty space

Increase:

```bash
--zoom
```

Example:

```bash
--zoom 1.25
```

---

## Problem: two runs give different views

Use the same:

```bash
--random-seed
```

and the same number/order of calls.

---

## Problem: off-screen rendering fails on cluster

This is commonly a VTK/OpenGL environment issue.

Try the cluster's Xvfb/EGL/OSMesa solution rather than changing hull geometry.

---

# 24. Recommended publication workflow

A practical workflow is:

1. Validate the polyhedron topology using known edge/face counts if available.
2. Start with:
   ```bash
   --cell-scale 1.0 --particle-scale 1.0
   ```
3. If particles visually overlap, tune the two scale factors.
4. Generate 10–30 candidate views:
   ```bash
   --nviews 20 --random-seed 123
   ```
5. Inspect the opaque versions quickly.
6. Choose the best view number.
7. Keep the matching transparent version for figure composition.
8. Increase resolution for the final render:
   ```bash
   --width 5000 --height 5000
   ```
9. Keep the exact command used for reproducibility.
10. Record the random seed and chosen view number in your project notes.

Because the seed is reproducible, you can regenerate the same camera sequence later.

---

# 25. Reproducibility checklist

For a publication-quality reproducible figure, save:

```text
source-code version
unit-cell input file
shape JSON file
command line
random seed
number of views
selected view number
cell scale
particle scale
cell line width
particle edge width
image width/height
projection mode
quaternion ordering
expected face count, if used
expected edge count, if used
Python environment/package versions
```

A convenient command to record package versions is:

```bash
python3.8 -m pip freeze > requirements_frozen.txt
```

---

# 26. Scientific interpretation of scaling

The independent scale factors are intentionally visual.

If the original system has a physical packing fraction \(\phi\), the displayed ratio changes approximately as:

\[
\phi_\mathrm{display}
\propto
\left(
\frac{s_\mathrm{particle}}
     {s_\mathrm{cell}}
\right)^3.
\]

Therefore a figure generated with:

```text
cell-scale != 1
or
particle-scale != 1
```

should not automatically be interpreted as a metrically faithful physical snapshot.

For publication, consider stating in the caption if particles/cell have been visually rescaled for clarity.

Orientations are unchanged by these isotropic scale factors.

---

# 27. Why orthographic projection is the default

The code uses orthographic projection unless:

```bash
--perspective
```

is supplied.

Orthographic projection is useful for structural figures because:

```text
objects do not become smaller merely because they are farther from the camera
parallel directions remain visually parallel
relative sizes are easier to compare
```

Perspective projection may be visually attractive, but introduces depth-dependent apparent scaling.

---

# 28. Anti-aliasing

The renderer tries:

```python
plotter.enable_anti_aliasing("ssaa")
```

If that fails, it tries:

```python
plotter.enable_anti_aliasing("msaa")
```

If both fail, rendering continues without explicitly enabling those modes.

If image quality differs between machines, PyVista/VTK/OpenGL backend differences may be responsible.

---

# 29. Why the scene is constructed only once

The program:

1. builds all particle meshes;
2. builds unit-cell lines;
3. adds lights;
4. calculates scene bounds;
5. then changes only the camera between views.

This is efficient and guarantees that different views contain the same geometry.

For every camera, the opaque screenshot is immediately followed by the transparent screenshot before moving to the next camera.

Therefore each pair is directly corresponding.

---

# 30. Output-pair naming rules

For:

```bash
-o unit_cell.png --nviews 3
```

outputs are:

```text
unit_cell_view01.png
unit_cell_view01_transparent.png
unit_cell_view02.png
unit_cell_view02_transparent.png
unit_cell_view03.png
unit_cell_view03_transparent.png
```

For:

```bash
-o figure.png --nviews 1
```

outputs are:

```text
figure.png
figure_transparent.png
```

For:

```bash
-o results/my_cell.png --nviews 2
```

the program creates the parent directory if needed and writes:

```text
results/my_cell_view01.png
results/my_cell_view01_transparent.png
results/my_cell_view02.png
results/my_cell_view02_transparent.png
```

---

# 31. Adding new command-line options

Suppose you want a command-line option:

```bash
--particle-color "#ff0000"
```

The recommended pattern is:

## Step 1: add parser option

In:

```python
build_argument_parser()
```

add:

```python
parser.add_argument(
    "--particle-color",
    type=str,
    default=PARTICLE_COLOR,
    help="Particle face color.",
)
```

## Step 2: add parameter to renderer

Add to:

```python
render_unit_cell_multi_view(...)
```

a parameter:

```python
particle_color: str = PARTICLE_COLOR
```

## Step 3: pass it through

In `main()`:

```python
particle_color=args.particle_color
```

## Step 4: pass it to particle construction

Add it to:

```python
add_all_particles(...)
```

## Step 5: use it

Replace:

```python
color=PARTICLE_COLOR
```

with:

```python
color=particle_color
```

Use the same pattern for lighting parameters, camera elevation limits, background color, label size, etc.

---

# 32. Suggested future improvements

The current source is already suitable for the present single-shape convex-particle workflow, but useful future additions could include:

```text
deterministic user-specified camera angles
view-index selection
separate opaque/transparent output folders
fractional-coordinate filtering of periodic duplicates
multiple particle shapes in one cell
particle-specific colors
orientation-class color maps
cylindrical unit-cell rods
lattice-point markers
axis triads
scale bars
explicit crystallographic a/b/c labels
STL/OBJ export
vector graphics export where possible
animation/turntable output
automatic contact/overlap diagnostics
explicit support for concave meshes
explicit face-connectivity JSON input
```

When extending the source, preserve the separation between:

```text
geometry construction
particle transformation
unit-cell transformation
camera generation
lighting
rendering
output
```

That separation is what keeps the code relatively easy to debug.

---

# 33. Minimal debugging strategy

If something is wrong, isolate the layer.

### Geometry problem

Symptoms:

```text
wrong number of faces
wrong number of edges
incorrect particle silhouette even before orientation
```

Inspect:

```python
read_shape_vertices()
build_convex_particle_geometry()
```

### Orientation problem

Symptoms:

```text
correct shape but wrong orientation
```

Inspect:

```python
quaternion_to_rotation_matrix()
transform_particle_vertices()
```

### Position/cell problem

Symptoms:

```text
particle centers wrong
cell box displaced
scaling inconsistent
```

Inspect:

```python
read_unit_cell_info()
get_cell_origin()
scale_position_about_origin()
build_cell_edge_mesh()
```

### Camera problem

Symptoms:

```text
bad viewing direction
too much empty space
undesirable angle
```

Inspect:

```python
compute_cell_camera_basis()
generate_random_camera_triplets()
zoom
```

### Appearance problem

Symptoms:

```text
too dark
too glossy
edge lines too weak
```

Inspect:

```python
PARTICLE_COLOR
PARTICLE_EDGE_COLOR
CELL_EDGE_COLOR
add_publication_lights()
plotter.add_mesh(...) material parameters
```

### Output problem

Symptoms:

```text
wrong names
wrong directory
transparent pair missing
```

Inspect the save loop in:

```python
render_unit_cell_multi_view()
```

---

# 34. Validation philosophy

This source intentionally fails early when the geometry is internally inconsistent.

That is preferable to silently generating a visually plausible but topologically incorrect particle.

The following checks are particularly important:

```text
duplicate vertex rejection
all JSON vertices must belong to hull
optional known face-count validation
optional known edge-count validation
Euler characteristic must equal 2
```

If one of these checks fails, first investigate the input geometry before weakening the validation.

---

# 35. One source-code comment that is sample-specific

Inside `add_all_particles()` there is a comment referring to drawing “exactly the 42 physical edges for the supplied shape” rather than triangulation edges.

The **algorithm itself is general** and uses:

```python
geometry.true_edges
```

so it does not assume 42 edges.

That comment describes the development/sample particle only. For a fully generic cleaned-up source, it can be replaced with:

```python
# Draw only verified physical polyhedron edges, excluding
# triangulation diagonals inside coplanar polygonal faces.
```

No functional change is required.

---

# 36. Example development shape

The included example shape JSON represents a centered, principal-frame, unit-volume particle.

For this example the convex-hull diagnostics obtained during development are:

```text
24 hull vertices
44 Qhull triangles
20 physical faces
42 true physical edges
18 quadrilateral faces
2 hexagonal faces
Euler characteristic = 2
volume ≈ 1
```

This is a useful regression test.

If future modifications to the source cause the same JSON to produce a different topology, investigate the change before trusting new images.

---

# 37. Example full production command

```bash
python3.8 render_unit_cell_pyvista_v1p6.py \
    uc_info.txt \
    shape_20_added_hexagonal_prism_both_side_hparam_0p2_eparam_0p7_unit_volume_principal_frame.json \
    -o unit_cell.png \
    --cell-scale 1.20 \
    --particle-scale 0.75 \
    --cell-line-width 48.0 \
    --particle-edge-width 1.5 \
    --width 3000 \
    --height 3000 \
    --nviews 10 \
    --random-seed 123 \
    --expected-num-faces 20 \
    --expected-num-edges 42 \
    --quaternion-order wxyz \
    --zoom 1.12
```

This uses orthographic projection because `--perspective` is not supplied.

For each of the ten views, both opaque and transparent PNGs are generated.

---

# 38. Final notes

The most important principles to preserve when modifying this project are:

1. **Do not manually guess polyhedron connectivity when SciPy/Qhull can determine the convex hull.**
2. **Keep Qhull surface triangles as the actual rendered surface.**
3. **Separate triangulation diagonals from true physical edges.**
4. **Validate known topology whenever face/edge counts are available.**
5. **Treat quaternion convention as a coordinate-convention issue, not a geometry issue.**
6. **Keep cell scaling and particle scaling independent.**
7. **Preserve particle-center fractional relationships when visually scaling the cell.**
8. **Use a fixed random seed when publication reproducibility matters.**
9. **Remember that the current geometry pipeline is for convex particles.**
10. **Keep the opaque and transparent image pair generated from the same unchanged camera.**

With those constraints respected, the source can be adapted to a wide range of convex-particle crystallographic and orientational-order visualization projects.
