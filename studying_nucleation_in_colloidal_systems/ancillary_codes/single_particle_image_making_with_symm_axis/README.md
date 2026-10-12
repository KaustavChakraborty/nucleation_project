# Polyhedron rotational-symmetry axes: automatic detection and publication-quality PyVista rendering

**Project documentation for the supplied `render_polyhedron_rotational_axes(1).py` script (v1p4 logic)**  
**Example input:** `shape_151_sheared_Cube_deg10_unit_volume_principal_frame(1).json`  
**Language:** Python 3.8+  
**Libraries:** NumPy, SciPy, PyVista (including its VTK rendering backend)

> **Important:** This README documents the **exact uploaded implementation**, not a proposed replacement. The downloaded Python file may have a parenthesized upload suffix; for simplicity, the commands below assume you renamed it to `render_polyhedron_rotational_axes.py` and the shape file to `shape.json`. You may instead use the original filenames, remembering to quote paths containing parentheses.

---

## Contents

1. [What this project does](#1-what-this-project-does)
2. [What it does not do](#2-what-it-does-not-do)
3. [Quick start](#3-quick-start)
4. [Installation and graphics environment](#4-installation-and-graphics-environment)
5. [Directory layout and output files](#5-directory-layout-and-output-files)
6. [Input JSON format](#6-input-json-format)
7. [End-to-end program workflow](#7-end-to-end-program-workflow)
8. [Rotational symmetry: mathematical foundations](#8-rotational-symmetry-mathematical-foundations)
9. [How the actual symmetry-finding algorithm works](#9-how-the-actual-symmetry-finding-algorithm-works)
10. [Reconstructing true polyhedron faces and edges](#10-reconstructing-true-polyhedron-faces-and-edges)
11. [How symmetry operations become displayed axes](#11-how-symmetry-operations-become-displayed-axes)
12. [Worked example: the 10° sheared cube](#12-worked-example-the-10-sheared-cube)
13. [How random camera directions are generated](#13-how-random-camera-directions-are-generated)
14. [How the PyVista image is constructed](#14-how-the-pyvista-image-is-constructed)
15. [Complete terminal parameter reference](#15-complete-terminal-parameter-reference)
16. [Ready-to-run customization recipes](#16-ready-to-run-customization-recipes)
17. [Source-code modification guide](#17-source-code-modification-guide)
18. [Camera and viewpoint modifications in the source](#18-camera-and-viewpoint-modifications-in-the-source)
19. [Symmetry and geometric tolerance adjustments](#19-symmetry-and-geometric-tolerance-adjustments)
20. [Interpreting JSON reports](#20-interpreting-json-reports)
21. [How to validate your results](#21-how-to-validate-your-results)
22. [Troubleshooting](#22-troubleshooting)
23. [Performance and publication workflow](#23-performance-and-publication-workflow)
24. [Function-by-function source map](#24-function-by-function-source-map)
25. [Limitations and possible extensions](#25-limitations-and-possible-extensions)
26. [Reproducibility checklist and citation guidance](#26-reproducibility-checklist-and-citation-guidance)

---

## 1. What this project does

The project reads **the 3D vertex coordinates of a convex polyhedron** from a JSON file, finds its *proper rotational symmetries*, and renders the polyhedron together with all the distinct symmetry-axis **lines**.

The key capabilities are:

- Automatically compute the convex hull, real polygonal faces, and physical edges from vertices.
- Detect all numerical **proper rotation matrices** that map the complete vertex set onto itself.
- Reject reflections and other improper transformations.
- Group nontrivial rotation matrices according to their common axes.
- Determine the rotational order of each axis (`C2`, `C3`, `C4`, etc.).
- Draw each distinct undirected axis once as a full two-sided cylindrical rod.
- Give all axes the **same red color** by default, independent of rotational order.
- Render the body as a **blue, flat-shaded solid** with pale visible edges.
- Save PNG images with a **transparent alpha-channel background**; no colored background is baked into the image.
- Generate **N independent random viewing directions** with a terminal argument.
- Save a numerical symmetry report and a reproducible per-view camera manifest.
- Create a fresh PyVista plotter for **each** view and force a render to avoid the repeated/stale screenshot issue encountered with earlier implementations.

A shape with no nontrivial proper rotations will still be rendered; the symmetry report will indicate `C1` behavior and there will be no symmetry-axis rods.

### Visual defaults

| Rendered component | Default behavior |
|---|---|
| Background | Transparent PNG, RGBA |
| Faces | Blue `#70A6D0` |
| Face shading | Flat / faceted; `smooth_shading=False` |
| Physical edges | Pale `#EDEFF5` |
| Symmetry axes | Uniform red `#D02020` |
| Axis geometry | Round cylinders with small spherical ends |
| Camera | Random directions, orthographic projection |
| Labels | Off |
| Legend | Off |
| Image width/height | 2400 × 2400 pixels by default |
| Default face opacity | 1.0, i.e., opaque solid |

**Distinguish two kinds of transparency:** the **background alpha** is always requested by `screenshot(..., transparent_background=True)`, whereas **face transparency** is set separately with `--opacity`. Using `--opacity 1.0` still produces a transparent *background*.

## 2. What it does not do

The code is deliberately scoped to a **single convex rigid particle**:

- It does **not** load a GSD trajectory or HOOMD simulation snapshot.
- It does **not** determine the particle's space group, crystal symmetry, point-group character table, or improper operations.
- It does **not** derive symmetry from masses, chemical species, atomic bonds, surface markings, or vertex labels; vertices are geometrically indistinguishable.
- It does **not** compute reflections, inversion, screw axes, rotoinversions, or translations.
- It does **not** use concave-face or face-connectivity input; it reconstructs the **convex hull**.
- It does **not** produce SVG or a genuine 3D editable model. Its principal image output is PNG.
- It does **not** automatically guarantee that every random viewpoint is aesthetically optimal or evenly separated from every other viewpoint.
- It does **not** infer a named point group (`D2`, `O`, `I`, etc.) as a separate field; it reports the operations and axis inventory from which such a classification can be made.

If the intended shape is **nonconvex**, **decorated**, or contains distinct vertex types, the geometry/symmetry logic needs to be generalized before interpreting its results physically.

---

## 3. Quick start

Place the Python script and the JSON file in one folder. The commands below use normalized filenames; adjust them to match your files.

```bash
python3.8 -m pip install --user numpy scipy pyvista
```

**First, check geometry and symmetry without invoking the graphics renderer:**

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --check-only --output sheared_cube.png
```

**Then generate 10 transparent images at random viewpoints:**

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 10 \
    --output sheared_cube.png \
    --resolution 2400 \
    --axis-length 1.55 \
    --axis-thickness 0.012 \
    --seed 42
```

This creates:

```text
sheared_cube_view_001.png
sheared_cube_view_002.png
...
sheared_cube_view_010.png
sheared_cube.axes.json
sheared_cube.views.json
```

To place everything in its own output folder:

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 10 --output results/sheared_cube.png --seed 42
```

The directory `results/` is created if necessary.

If you omit `--n-views` while running from an interactive terminal, the program asks:

```text
How many random viewing angles (N)?
```

For scripts, batch jobs, and redirected standard input, provide `--n-views N` explicitly. To see all supported options and actual defaults, run:

```bash
python3.8 render_polyhedron_rotational_axes.py --help
```

---

## 4. Installation and graphics environment

### 4.1 Python packages

The source imports these packages:

```python
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
```

It imports `pyvista` only when rendering a picture, inside `create_scene()`:

```python
import pyvista as pv
```

This means `--check-only` can work with **NumPy and SciPy alone**, whereas PNG rendering also requires **PyVista and its VTK dependency**.

Recommended minimal installation for an existing Python 3.8 environment:

```bash
python3.8 -m pip install --user numpy scipy pyvista
```

If you manage a virtual environment:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install numpy scipy pyvista
```

Confirm the library versions actually used:

```bash
python3.8 -c 'import sys, numpy, scipy, pyvista; print(sys.version); print("NumPy:", numpy.__version__); print("SciPy:", scipy.__version__); print("PyVista:", pyvista.__version__)'
```

**Version note:** Python 3.8 is no longer supported by some current scientific-Python releases. If `pip` reports that a current package requires a newer Python version, use compatible older versions from the package's release history or create a newer Python environment. Avoid forcing incompatible versions. You do not need the newest PyVista to understand or run the symmetry calculations.

### 4.2 Linux desktop versus remote compute node

Rendering requires an OpenGL-capable VTK environment. On a Linux desktop with a working X11/Wayland graphics session, off-screen PyVista rendering often works without further configuration.

A remote HPC node may not have an X server, GPU display, EGL, or a compatible software-rendering backend. Depending on its configuration, an X virtual framebuffer can be helpful:

```bash
# Only if Xvfb is installed and appropriate for your VTK build:
xvfb-run -a python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 4 --output test.png --resolution 800 --seed 1
```

`xvfb-run` is **not** a universal fix: it does not add a missing OpenGL/VTK backend. If a machine has headless EGL/OSMesa support, configuration may differ. Test one low-resolution image before submitting a large batch.

### 4.3 Test computation without rendering

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --check-only --output quickcheck.png
```

This tests vertex parsing, convex geometry, proper rotations, axis identification, and JSON report writing, but **does not test PNG generation**.

---

## 5. Directory layout and output files

A straightforward project layout is:

```text
polyhedron_symmetry_axes/
├── README.md
├── render_polyhedron_rotational_axes.py
├── shape.json
└── results/
    ├── sheared_cube_view_001.png
    ├── sheared_cube_view_002.png
    ├── ...
    ├── sheared_cube_view_010.png
    ├── sheared_cube.axes.json
    └── sheared_cube.views.json
```

**Important naming rule:** `--output results/sheared_cube.png` is a **prefix/template**, not the filename of a single generated screenshot. The actual PNGs receive `_view_001`, `_view_002`, and so on before `.png`.

| File | Contents | Why keep it? |
|---|---|---|
| `NAME_view_001.png` etc. | Transparent RGBA rendered image | Figure or visual inspection |
| `NAME.axes.json` | Name, vertex count, rotation count, axis orders/directions/angles | Numerical verification and reuse |
| `NAME.views.json` | Random seed, camera positions, directions, view-up vectors, filenames, image size | Reproducing and analyzing viewpoints |

If `--report another/path/report.json` is supplied, the **symmetry report** moves to that path. The camera manifest still uses the image prefix: `NAME.views.json`.

**Overwrite behavior:** using the same output prefix and N again will overwrite the same numbered PNG names and report/manifest. Choose a different prefix if you want to preserve previous results.

**Partial-run behavior:** the `.axes.json` report is written before rendering. The `.views.json` manifest is written **after** all images finish successfully. If rendering stops partway through, earlier PNGs may exist without a completed camera manifest.

---

## 6. Input JSON format

### 6.1 Minimal supported JSON

The only essential geometric data are at least four finite, non-coplanar 3D points representing a convex polyhedron:

```json
{
  "name": "my_polyhedron",
  "vertices": [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ]
}
```

This describes a tetrahedron, not a cube. The source reads one of the following vertex keys:

```text
vertices
8_vertices
12_vertices
polyhedron_vertices
<any-other-key-ending-in-_vertices>
```

The source tests the explicitly listed keys **first**, then any other `*_vertices` key in the JSON's key order. A numerical prefix such as `8_` is **not parsed as the vertex count**; the code simply checks that the field contains an `(N, 3)` numeric array. If a JSON file has multiple vertex fields, make sure the intended array is selected.

### 6.2 The supplied sheared-cube file

The attached input contains metadata including:

```json
{
  "0_Id": "151",
  "1_Name": "sheared_Cube",
  "2_ShortName": "T06",
  "3_Comment": "Vertices are in principal frame,centered, and correspond to unit volume",
  "4_volume": 0.9999999999999997,
  "8_vertices": [
    [-0.589826272665149, -0.5025579930074546, -0.49492300785199633],
    [-0.4152897133497602, -0.5025579930074546, 0.49492300785199633],
    [-0.589826272665149, 0.5025579930074546, -0.49492300785199633],
    [-0.4152897133497602, 0.5025579930074546, 0.49492300785199633],
    [0.4152897133497602, -0.5025579930074546, -0.49492300785199633],
    [0.589826272665149, -0.5025579930074546, 0.49492300785199633],
    [0.4152897133497602, 0.5025579930074546, -0.49492300785199633],
    [0.589826272665149, 0.5025579930074546, 0.49492300785199633]
  ],
  "9_shear_quantity (deg)": "10"
}
```

This excerpt omits additional input metadata for readability. Its vertex coordinates are the actual values supplied with this project.

The current script uses:

- `8_vertices`: to determine the geometry;
- `1_Name`: as the shape name in the symmetry report (fallback: `name`, then `polyhedron`).

The other supplied fields (`0_Id`, short name, volume, center-of-mass, moment of inertia, shear angle) are **not** directly used to compute the axes or render the shape. In particular, the reported `10` degrees is not used to infer or constrain symmetry: the symmetry is calculated from the vertex coordinates.

### 6.3 Coordinate system

Vertices can be expressed in any consistent Cartesian coordinate frame. The example is already approximately centered and in its **principal-axis frame**; however, the script recalculates its own symmetry center from the selected hull vertices.

- **Translation:** translating every vertex by the same amount does not change the proper rotational group.
- **Rigid rotation:** rotates the numerical axis directions with the body, without changing their number or orders.
- **Uniform scaling:** leaves symmetry unchanged; many render sizes are proportional to the body radius.
- **Anisotropic scaling / shear:** can **change or destroy symmetry**, so recalculate rather than reuse axes from the original shape.
- **Vertex ordering:** does not convey edges or faces. The convex-hull routine constructs those automatically.

**Warning:** vertices alone do not convey whether one site is chemically distinct from another. The code treats every vertex with identical status.

### 6.4 Convexity requirements

The implementation starts with `scipy.spatial.ConvexHull(vertices)`. For a nonconvex particle, the rendered object and symmetry calculations are for its **convex hull**, not its true concave surface.

Any supplied points that do not lie on the convex hull are removed before symmetry analysis. This is sensible for a convex polyhedral vertex catalog with redundant interior points, but be careful if a non-hull point carries physical meaning. Duplicate points should also be cleaned upstream if they cause numerical problems.

If all vertices are coplanar, collinear, or otherwise fail to define a 3D hull, `ConvexHull` may raise a Qhull error; this program expects a full three-dimensional polyhedron.

---

## 7. End-to-end program workflow

The `main()` function executes the following chain:

```text
Command-line arguments
        │
        ▼
read_vertices(json_path)
        │   Read JSON → N×3 coordinates → remove non-hull points
        │   Calculate center = average of retained hull vertices
        ▼
polygon_faces_from_hull(vertices, hull)
        │   Merge coplanar hull triangles into polygon faces
        │   Extract only physical perimeter edges
        ▼
find_proper_rotations(vertices, center, rel_tol)
        │   Construct candidate rigid rotations using matched vertex frames
        │   Keep only determinant +1 rotations mapping full set to itself
        │   Check approximate group closure
        ▼
symmetry_axes(rotations)
        │   Convert each nonidentity rotation to an axis and angle
        │   Merge coincident ±axis directions, assign Cn orders
        ▼
save_report(...)
        │   Write NAME.axes.json
        ▼
[--check-only?] ── yes ──> stop
        │ no
        ▼
random_camera_directions(N, seed)
        │
        ▼
For each viewing direction:
    create_scene(...)
        Draw body + physical edges + red axes + lighting
    Initialize a NEW PyVista render window
    set_camera_for_direction(...)
    Force camera.Modified() and plotter.render()
    plotter.screenshot(... transparent_background=True)
    Check RGBA output and duplicate-frame hash
    Close plotter
        │
        ▼
Write NAME.views.json
```

The separation between the **geometry/symmetry stage** and **graphics stage** is intentional: you can independently verify the axes without launching PyVista.

---

## 8. Rotational symmetry: mathematical foundations

### 8.1 What makes a rotation a symmetry?

Let the shape be represented by a finite set of vertices:

$$
V=\{\mathbf r_1,\ldots,\mathbf r_N\},\qquad \mathbf r_i\in\mathbb R^3.
$$

Let `c` be the reference center and let

$$
\mathbf p_i=\mathbf r_i-\mathbf c.
$$

A **proper rotation** is a real matrix $R\in SO(3)$ satisfying

$$
R^{\mathsf T}R=I,\qquad\det R=+1.
$$

It is a rotational symmetry of the geometric vertex set precisely when it permutes that set:

$$
\boxed{\{R\mathbf p_i:i=1,\ldots,N\}=\{\mathbf p_i:i=1,\ldots,N\}.}
$$

Equivalently, there must exist a bijective permutation $\pi$ for which

$$
R\mathbf p_i=\mathbf p_{\pi(i)}\quad\text{for every }i.
$$

A reflection can also preserve the vertex set, but it has determinant $-1$ and is **not** a proper rotation. The code explicitly constructs determinant-positive candidates, excluding reflected solutions.

### 8.2 Why the vertex-average center works

The implementation uses the arithmetic mean of the retained hull vertices:

$$
\mathbf c=\frac1N\sum_{i=1}^{N}\mathbf r_i.
$$

If a true rotational symmetry maps each vertex to a permutation of all vertices, then

$$
\frac1N\sum_i R(\mathbf r_i-\mathbf c)
=\frac1N\sum_i(\mathbf r_{\pi(i)}-\mathbf c)=\mathbf 0.
$$

Thus the mean is invariant under every such symmetry. It need not coincide with the **volume centroid** for an arbitrary polyhedron, and it is not necessarily the center listed in input metadata.

This vertex-average center is a valid common fixed point for the rotations of a finite geometric vertex set. It is computed **after** redundant/non-hull points are removed.

### 8.3 Cyclic axes and rotational orders

An axis is said to have rotational order $n$, written $C_n$, if rotation by

$$
\theta=\frac{2\pi}{n}
$$

about that axis leaves the shape unchanged, with $n$ the **largest** order of the rotations sharing that axis.

Examples:

| Axis type | Smallest positive symmetry angle | Nonidentity rotations around the same axis |
|---|---:|---|
| `C2` | 180° | 180° |
| `C3` | 120° | 120°, 240° |
| `C4` | 90° | 90°, 180°, 270° |
| `C5` | 72° | 72°, 144°, 216°, 288° |
| `C6` | 60° | 60°, 120°, 180°, 240°, 300° |

The source actually records the *principal* rotation-vector angle in $[0,180^\circ]$, so a 240° rotation may be stored as 120° around the opposite directed unit vector. Its axis **line** remains the same.

**An operation is not an axis.** A `C4` axis contributes three nonidentity group elements but is drawn only **once**. This distinction is central to correctly counting and plotting rotational axes.

### 8.4 Proper rotational group versus complete point group

All detected rotations, including identity, form the polyhedron's **proper rotational subgroup** (numerically, within selected tolerances). An ordinary full point group can additionally contain mirrors and inversion.

For example, the supplied sheared cube has the four proper rotations of `D2`. Its complete point group is not automatically established by this code, because improper symmetries are not tested. Do **not** infer the complete Schoenflies symbol solely from the printed proper-rotation count.

---

## 9. How the actual symmetry-finding algorithm works

The main routine is **`find_proper_rotations(vertices, center, rel_tol=1e-6)`**. It does not try every possible continuous rotation angle. Instead, it constructs **discrete candidates from geometric correspondences** and validates each candidate against the entire vertex set.

### Step 1 — Center the shape and establish numerical scales

```python
p = vertices - center
radius = np.linalg.norm(p, axis=1).max()
absolute_tol = float(rel_tol) * radius
dot_tol = 4.0 * absolute_tol * radius
```

Mathematically:

$$
\rho=\max_i\|\mathbf p_i\|,\quad
\varepsilon_{\rm position}=\varepsilon_{\rm rel}\rho,\quad
\varepsilon_{\rm dot}=4\varepsilon_{\rm rel}\rho^2.
$$

Here `rel_tol` is normally `--symmetry-tol`; by default, $10^{-6}$. A tolerance proportional to $\rho$ makes the vertex-matching test largely insensitive to uniform rescaling.

**Important:** $\varepsilon_{\rm dot}$ tests squared lengths and dot products, while $\varepsilon_{\rm position}$ tests point-to-point distances. They are not interchangeable units.

### Step 2 — Compute rotationally invariant quantities

```python
r2 = np.einsum('ij,ij->i', p, p)
dot = p @ p.T
```

This calculates

$$
r_i^2=\mathbf p_i\cdot\mathbf p_i,
\qquad G_{ij}=\mathbf p_i\cdot\mathbf p_j.
$$

For any orthogonal rotation,

$$
(R\mathbf p_i)\cdot(R\mathbf p_j)=\mathbf p_i\cdot\mathbf p_j.
$$

Consequently, **pairwise lengths and dot products must match** between each chosen collection of source vertices and their images. Testing these invariants discards most invalid candidate correspondences before doing expensive full-shape comparisons.

### Step 3 — Choose three linearly independent source anchors

The code chooses the source anchor indices `a`, `b`, and `c`:

```python
a = int(np.argmax(r2))
b = int(np.argmax(np.linalg.norm(np.cross(p[a], p), axis=1)))
c = int(np.argmax(np.abs(p @ np.cross(p[a], p[b]))))
source = p[[a, b, c]]
```

These choices mean:

1. **Anchor `a`:** choose a farthest vertex from the center to obtain a well-scaled reference.
2. **Anchor `b`:** choose a vertex maximizing $\|\mathbf p_a\times\mathbf p_b\|$, favoring noncollinear directions.
3. **Anchor `c`:** maximize the magnitude of the scalar triple product $|\mathbf p_c\cdot(\mathbf p_a\times\mathbf p_b)|$, favoring a third direction out of the first two's plane.

A proper rotation in 3D is uniquely determined by how it maps a suitable nondegenerate 3D frame. Using three linearly independent radius vectors therefore permits recovery of a full rotation matrix from each plausible image triple.

The implementation checks that the determinant of the anchor matrix is not excessively small:

```python
if abs(np.linalg.det(source)) < 1e-9 * radius**3:
    raise ValueError('Cannot find three independent radius vectors.')
```

This prevents a poorly defined 3D mapping when the shape or anchor set is nearly coplanar.

### Step 4 — Enumerate possible target triples

Every rotation symmetry maps the three anchors to three vertices that preserve the anchor Gram matrix.

The nested loops select candidate indices `j`, `k`, `l` satisfying approximate matching of:

$$
\begin{aligned}
\|\mathbf p_j\|^2&\simeq\|\mathbf p_a\|^2,\\
\|\mathbf p_k\|^2&\simeq\|\mathbf p_b\|^2,\\
\mathbf p_j\cdot\mathbf p_k&\simeq\mathbf p_a\cdot\mathbf p_b,\\
\|\mathbf p_l\|^2&\simeq\|\mathbf p_c\|^2,\\
\mathbf p_j\cdot\mathbf p_l&\simeq\mathbf p_a\cdot\mathbf p_c,\\
\mathbf p_k\cdot\mathbf p_l&\simeq\mathbf p_b\cdot\mathbf p_c.
\end{aligned}
$$

This filtering is cheaper than attempting every arbitrary rotation in $SO(3)$. Only triples consistent with preserved inner products proceed to matrix reconstruction.

### Step 5 — Reconstruct the best-fitting *proper* rotation using SVD (Kabsch alignment)

Let $S$ be the 3×3 matrix whose **rows** are the three source anchor vectors, and $T$ the corresponding candidate target matrix. The code performs:

```python
u, _, vt = np.linalg.svd(source.T @ target)
reflection_fix = np.eye(3)
reflection_fix[2, 2] = np.sign(np.linalg.det(u @ vt))
matrix = vt.T @ reflection_fix @ u.T
```

Writing

$$
S^{\mathsf T}T=U\Sigma V^{\mathsf T},\quad
D=\mathrm{diag}\{1,1,\mathrm{sign}[\det(UV^{\mathsf T})]\},
$$

the recovered column-vector rotation is

$$
\boxed{R=VDU^{\mathsf T}.}
$$

With coordinates stored as **row vectors** in NumPy, it acts as

```python
rotated_points = p @ matrix.T
```

The diagonal correction $D$ excludes an improper reflection that an unconstrained least-squares fit might otherwise produce. The determinant condition is checked again:

```python
if np.linalg.det(matrix) < 1 - 1e-7:
    continue
```

**Why SVD instead of directly solving $R=S^{-1}T$?** The SVD method finds the nearest proper orthogonal map in a least-squares sense, respects the rotational constraint, and is more stable for small coordinate roundoff and imperfect geometry.

### Step 6 — Remove duplicate rotations

Different valid image triples can describe the same rotation. The script skips a candidate if its matrix lies within a **Frobenius norm** difference of `1e-6` from an already accepted rotation:

```python
if any(np.linalg.norm(matrix - old) < 1e-6 for old in matrices):
    continue
```

This is a **matrix-uniqueness tolerance**, distinct from `--symmetry-tol`.

### Step 7 — Verify the entire vertex-set mapping

A plausible mapping of three anchor vertices is **not enough** to establish whole-particle symmetry. The candidate must map **every** vertex onto a unique member of the original set.

```python
lookup = cKDTree(p)
distances, nearest = lookup.query(p @ matrix.T, k=1)
if np.max(distances) <= absolute_tol and len(set(nearest)) == n:
    matrices.append(matrix)
```

The KD-tree efficiently finds the nearest original vertex to each rotated vertex. Acceptance requires both:

$$
\max_i\min_j\|R\mathbf p_i-\mathbf p_j\|
\le\varepsilon_{\rm position},
$$

and

$$
\#\{\text{nearest-vertex indices}\}=N.
$$

The second condition matters: without it, two transformed vertices could both be matched to the same original vertex, resulting in a false positive.

Since there are N transformed vertices and N unique nearest indices, this nearest-neighbor check enforces a bijection for the accepted correspondence.

### Step 8 — Guarantee inclusion of the identity

The identity matrix $I$ is always a valid symmetry. It is appended if the generated candidates did not already include it numerically:

```python
if not any(np.linalg.norm(m - np.eye(3)) < 1e-6 for m in matrices):
    matrices.append(np.eye(3))
```

### Step 9 — Test approximate group closure

A group must be closed under composition. If $R_a$ and $R_b$ are symmetries, then $R_aR_b$ must also be a symmetry.

The script checks every ordered pair:

```python
for ra in matrices:
    for rb in matrices:
        product = ra @ rb
        if not any(np.linalg.norm(product - rc) < 1e-4 for rc in matrices):
            raise RuntimeError(...)
```

Mathematically:

$$
\forall R_a,R_b\in G,\quad
\exists R_c\in G:\ \|R_aR_b-R_c\|_F<10^{-4}.
$$

If closure fails, the script **raises an error** rather than silently returning a possibly incomplete or spurious rotational group. This is an approximate numerical test, not an exact symbolic proof.

### Summary of core numerical thresholds

| Use | Source expression | Default/scaling |
|---|---|---|
| Vertex map acceptance | `absolute_tol = rel_tol * radius` | $10^{-6}\rho$ |
| Gram/dot-product candidate matching | `4 * absolute_tol * radius` | $4\times10^{-6}\rho^2$ |
| Anchor independence | `1e-9 * radius**3` | $10^{-9}\rho^3$ |
| Proper determinant | `det(R) >= 1 - 1e-7` | Absolute check |
| Rotation duplicate detection | `norm(R-Rold) < 1e-6` | Frobenius norm |
| Closure | `norm(Ra@Rb-Rc) < 1e-4` | Frobenius norm |
| Axis angular grouping | `abs(dot(axis1,axis2)) > 1 - 1e-6` | Dimensionless |
| Ignore near-identity angle | `angle < 1e-5` | Radians |

Changing `--symmetry-tol` affects the **first two** rows directly. The other checks are hardcoded and need source edits if a specialized numerical system requires different thresholds.

---

## 10. Reconstructing true polyhedron faces and edges

The function **`polygon_faces_from_hull(vertices, hull, rel_tol=1e-7)`** is as important to figure quality as the renderer.

### 10.1 Why the convex hull alone is not sufficient for clean edges

`scipy.spatial.ConvexHull` normally represents its convex surface as **triangular simplices**. A square face of a cube is represented by two triangles, separated by an artificial diagonal. Showing every triangulation edge would incorrectly display that diagonal.

The goal is to reconstruct the **true planar polygonal faces** and draw only their boundaries.

### 10.2 Coplanar triangle merging

Each hull simplex comes with a plane equation

$$
\mathbf n\cdot\mathbf x+d=0,
$$

with a normal $\mathbf n$ and offset $d$. The source groups triangles when:

```python
np.dot(normal, group['normal']) > 1 - 1e-8
abs(offset - group['offset']) < plane_tol
```

Here

```python
extent = max(np.ptp(vertices, axis=0).max(), 1e-12)
plane_tol = rel_tol * extent
```

The first criterion checks near-parallel, **same-oriented** normals; the second checks whether their planes occupy the same location. Vertices from matching triangles are pooled into one face.

### 10.3 Recover a stable polygon order

For every merged face:

1. Collect all distinct face vertex indices.
2. Find their face-center average.
3. Construct an in-plane vector $\mathbf u$.
4. Compute $\mathbf v=\mathbf n\times\mathbf u$.
5. Compute the polar angle of each vertex around the face center.
6. Sort by angle to produce a cyclic polygon perimeter.

Specifically, the angle calculation is:

```python
relative = points - face_center
angle = np.arctan2(relative @ v, relative @ u)
ordered = ids[np.argsort(angle)].tolist()
```

### 10.4 Draw only genuine physical edges

Each polygon contributes consecutive index pairs, including the closing edge. Edges are stored in a set of sorted pairs:

```python
edges.add(tuple(sorted((ordered[j], ordered[(j + 1) % len(ordered)]))))
```

That removes duplicates where two adjacent faces share the same physical edge.

The implementation then constructs:

- `vtk_faces`: a flat array encoding polygon sizes and vertex-index sequences for `pv.PolyData(vertices, faces)`;
- `edges`: a list of undirected geometric edges for PyVista line rendering;
- `len(groups)`: the reconstructed number of polygon faces.

### 10.5 What to do if you see unwanted diagonals or merged faces

The face-reconstruction tolerance is **not** exposed through a CLI option. Change the default of:

```python
def polygon_faces_from_hull(vertices, hull, rel_tol=1e-7):
```

or modify the call in `main()`:

```python
faces, edges, n_faces = polygon_faces_from_hull(
    vertices, hull, rel_tol=1e-8
)
```

Use a **smaller** tolerance when two distinct nearby planes are erroneously merged. Use a **slightly larger** tolerance when numerically coplanar triangles fail to merge because of finite coordinate precision. Large tolerances can hide real shallow faces, so inspect geometry after changing them.

---

## 11. How symmetry operations become displayed axes

### 11.1 Extract axis-angle form

After computing the valid proper rotations, `symmetry_axes(matrices)` converts each nonidentity matrix into a rotation vector:

```python
rotation_vector = Rotation.from_matrix(matrix).as_rotvec()
angle = np.linalg.norm(rotation_vector)
axis = rotation_vector / angle
```

The rotation vector $\boldsymbol\omega$ is defined so that

$$
\boldsymbol\omega=\theta\hat{\mathbf n},
$$

where $\theta=\|\boldsymbol\omega\|$ is the principal rotation angle in radians and $\hat{\mathbf n}$ is a unit axis direction. The script ignores angles below `1e-5` rad (the identity or a numerically negligible rotation).

An equivalent derivation uses Rodrigues' formula:

$$
R=I\cos\theta+(1-\cos\theta)\hat{\mathbf n}\hat{\mathbf n}^{\mathsf T}
+[\hat{\mathbf n}]_\times\sin\theta.
$$

You do **not** need to implement this formula yourself; SciPy's `Rotation.as_rotvec()` performs the extraction.

### 11.2 Treat axes as undirected lines

An axis with direction $\hat{\mathbf n}$ is the same geometric line as the direction $-\hat{\mathbf n}$. The function `canonical_axis(vec)` chooses a reproducible sign by making the vector's largest-magnitude component positive:

```python
imax = int(np.argmax(np.abs(vec)))
if vec[imax] < 0:
    vec = -vec
```

The script then groups lines according to:

```python
abs(np.dot(group['direction'], axis)) > 1 - 1e-6
```

The `abs()` is essential, because antiparallel vectors specify the same axis line.

### 11.3 Infer the maximal cyclic order

For each axis group the source uses:

```python
group['order'] = len(group['angles_degrees']) + 1
```

For an exact finite rotational group, a cyclic subgroup generated by a `C_n` rotation contains **n operations**, of which **n−1** are nonidentity. Hence the reported order is computed as the count of distinct nontrivial accepted rotations sharing the line, **plus one**.

For example, a `C4` axis normally gathers the rotations by 90°, 180°, and 270°, so it is reported as `C4` and drawn once. In the `angles_degrees` list, the 270° operation will usually appear as a 90° principal angle; the angle list therefore may contain repeated angle magnitudes although the **rotation matrices** are different.

### 11.4 Why every axis is red

In `create_scene()`:

```python
color = args.axis_color
```

The code does not assign different colors to different orders. The default `--axis-color` is `#D02020`, so *every* line is red, irrespective of `C2`, `C3`, etc.

A longer axis-order may be drawn **10% thicker** than a `C2` axis because the current cylinder radius is:

```python
radius=axis_radius * (1.10 if order > 2 else 1.0)
```

If you want absolutely identical thickness for every rotational order, change that line to:

```python
radius=axis_radius
```

---

## 12. Worked example: the 10° sheared cube

The supplied 8-vertex shape produces the following numerical result with the uploaded algorithm:

```text
Shape: sheared_Cube
Geometry: 8 vertices, 6 faces, 12 physical edges
Proper rotation group: 4 operations (including identity)
  C2: 3 distinct undirected axis/axes
  Axis  1: C2, direction [-0.6427876 -0.         0.7660444]
  Axis  2: C2, direction [ 0.7660444 -0.         0.6427876]
  Axis  3: C2, direction [-0.  1.  0.]
```

Thus the axis inventory is:

| Axis | Rotational order | Unit-axis direction (approximately) |
|---|---|---|
| 1 | `C2` | $(-0.6427876,\ 0,\ 0.7660444)$ |
| 2 | `C2` | $(0.7660444,\ 0,\ 0.6427876)$ |
| 3 | `C2` | $(0,\ 1,\ 0)$ |

Each represents a **180° proper rotational symmetry**. There are three such nonidentity transformations, plus the identity:

$$
G_{\rm proper}=\{I,\ C_{2,1},\ C_{2,2},\ C_{2,3}\},\quad |G_{\rm proper}|=4.
$$

These three axis directions are mutually perpendicular, up to numerical precision. Their directions need not coincide with the global `x`, `y`, or `z` axes—even though the vertex data are in a principal frame. The associated **proper rotational subgroup** is `D2` (isomorphic to the Klein four-group).

**Do not confuse the shearing angle and a rotation angle.** The input metadata says the shape was sheared by 10°, but each detected `C2` symmetry corresponds to a **180° symmetry rotation** of the finished geometric shape.

### Expected checks for other shapes

These are useful mathematical benchmark counts for ideal convex polyhedra (and are **not** claims that they were rendered with your current setup):

| Ideal geometry | Proper rotations including identity | Distinct symmetry-axis inventory |
|---|---:|---|
| Generic asymmetric convex polyhedron | 1 | None (`C1`) |
| Rectangular box with three unequal dimensions | 4 | Three `C2` axes |
| Regular tetrahedron | 12 | Four `C3` axes and three `C2` axes |
| Cube | 24 | Three `C4`, four `C3`, six `C2` axes |

The cube's `C2` axes in this table refer to **maximal `C2` lines** distinct from the `C4` lines. A `C4` line also admits a 180° operation, but it is counted and drawn under its **maximal** order, `C4`.

---

## 13. How random camera directions are generated

### 13.1 Why uniformly random angles require care

Selecting both the polar angle $\theta\in[0,\pi]$ and azimuthal angle $\phi\in[0,2\pi)$ uniformly does **not** give a uniform distribution on the sphere. It overrepresents the poles.

Uniform surface area corresponds to the spherical measure

$$
d\Omega=\sin\theta\,d\theta\,d\phi=d\phi\,d(\cos\theta).
$$

The implementation samples

$$
z=\cos\theta\sim\mathcal U(-1,1),\qquad
\phi\sim\mathcal U(0,2\pi),
$$

and constructs

$$
\boxed{
\hat{\mathbf d}=
\left(\sqrt{1-z^2}\cos\phi,\ 
\sqrt{1-z^2}\sin\phi,\ z\right).
}
$$

This yields directions that are uniformly distributed over the viewing sphere **in probability**.

### 13.2 Exact implementation

The function `random_camera_directions(n_views, seed=None)` contains:

```python
rng = np.random.default_rng(seed)
z = rng.uniform(-1.0, 1.0, size=n_views)
azimuth = rng.uniform(0.0, 2.0 * np.pi, size=n_views)
xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
return np.column_stack((xy * np.cos(azimuth),
                        xy * np.sin(azimuth), z))
```

Each returned row is a Cartesian **unit vector** pointing from the particle center toward the camera. The program places the camera at a fixed distance along this vector.

### 13.3 What `--seed` guarantees

Use `--seed 42`, for example, to make the pseudo-random sequence reproducible for a particular NumPy random-number implementation.

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 10 --seed 42 --output trial_A.png

python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 10 --seed 42 --output trial_B.png
```

These commands select the same camera **direction vectors** if run with the same NumPy behavior and unchanged source. Resulting PNG pixels can still differ across graphics hardware, driver, VTK/PyVista version, lighting/backend configuration, and antialiasing implementation.

Different seed values produce different random sequences; an omitted seed uses a non-fixed generator state.

### 13.4 Independent random views are not an evenly spaced survey

The N directions are independently sampled, so:

- Two camera directions may be **close together** by chance.
- There is no enforced minimum angular separation.
- The camera positions are not chosen to maximize visible faces or expose every distinct axis.
- A highly symmetric body may look very similar from physically different directions.

If you need systematically spaced views rather than statistically uniform independent samples, see [Section 18](#18-camera-and-viewpoint-modifications-in-the-source).

---

## 14. How the PyVista image is constructed

### 14.1 One fresh renderer per image

The version documented here uses **`create_scene()`** within the `render_images()` loop. For each output image, it creates a new `pv.Plotter`, builds the same geometry, initializes its render window, applies the new viewpoint, forces `render()`, captures an RGBA screenshot, and closes the plotter.

That strategy is intentionally more conservative than one scene shared across all screenshots. It was introduced because a previous renderer could save multiple pixel-identical frames even after camera coordinates changed.

**Trade-off:** more setup and rendering time per view; better isolation between frames.

### 14.2 Polyhedron surface

The source creates a PyVista polygon mesh:

```python
surface = pv.PolyData(vertices, faces)
```

It adds the mesh with:

```python
plotter.add_mesh(
    surface,
    color=args.face_color,
    opacity=args.opacity,
    smooth_shading=False,
    show_edges=False,
    ambient=0.20,
    diffuse=0.78,
    specular=0.08,
    specular_power=12,
    name='polyhedron',
)
```

**Flat shading** makes different planar faces appear as distinct blue facets. `ambient`, `diffuse`, `specular`, and `specular_power` determine how bright, directional, and glossy those facets appear under the lights.

### 14.3 Physical edges

True polygon perimeter edges are converted into a PyVista line mesh:

```python
edge_cells = np.array([[2, i, j] for i, j in edges], dtype=np.int64).ravel()
edge_mesh = pv.PolyData(vertices)
edge_mesh.lines = edge_cells
plotter.add_mesh(
    edge_mesh,
    color=args.edge_color,
    line_width=args.edge_width,
    render_lines_as_tubes=True,
    lighting=False,
    opacity=0.95,
    name='physical_edges',
)
```

The outline is not generated from hull triangles; it uses the reconstructed **physical** edge list to prevent unwanted diagonals.

`line_width` is a rendering width (practically controlled in screen-space units by VTK), while the symmetry-axis thickness is a **3D world-space cylinder radius**. Their units and scaling are therefore different.

### 14.4 Symmetry rods

For each axis, a full cylinder is created:

```python
tube = pv.Cylinder(
    center=center,
    direction=direction,
    radius=axis_radius * (1.10 if order > 2 else 1.0),
    height=2 * axis_half_length,
    resolution=48,
    capping=True,
)
```

Each rod runs through the computed symmetry center from $-L\hat{\mathbf n}$ to $+L\hat{\mathbf n}$:

$$
\mathbf x(t)=\mathbf c+t\hat{\mathbf n},\qquad -L\le t\le L.
$$

where

$$
L=(\texttt{--axis-length})\rho,
\qquad
r_{\rm rod}=(\texttt{--axis-thickness})\rho.
$$

Two small endpoint spheres are added to give each rod rounded visual ends. These do not represent extra physical axes.

### 14.5 Lights and shaded faces

The three light specifications in `create_scene()` are:

```python
light_specs = [
    ((4, -3, 5), 0.90),
    ((-4, 2, 3), 0.43),
    ((0, 4, -3), 0.25),
]
```

Each light's position is computed as

$$
\mathbf x_{\rm light}=\mathbf c+\rho\,(l_x,l_y,l_z),
$$

with its focal point at the shape center. The tuples give **relative world-space light positions** and the second numbers give light intensities. Because the lights are defined in world coordinates rather than attached to the changing camera, face brightness can vary by viewpoint in physically consistent ways.

### 14.6 Orthographic camera and consistent framing

The camera is located at

$$
\boxed{\mathbf x_{\rm cam}=\mathbf c+5\rho\hat{\mathbf d}.}
$$

It looks toward `center`, normally with `(0,0,1)` as its screen-up reference. If the direction approaches the world `z` axis (absolute dot product greater than 0.96), the source switches to `(0,1,0)` as the up vector to avoid degeneracy.

The code enables **parallel/orthographic projection**:

```python
plotter.enable_parallel_projection()
plotter.camera.parallel_scale = max(axis_half_length * 1.30,
                                    radius * 1.65)
plotter.reset_camera_clipping_range()
```

Parallel projection removes perspective convergence, making images easier to compare across random views. A nearly fixed bounding-sphere scale keeps the particle similarly sized across all images.

### 14.7 Transparent PNG screenshot

The image is saved with:

```python
img = plotter.screenshot(
    filename=str(path),
    transparent_background=True,
    return_img=True,
)
```

**The transparency keyword belongs to `screenshot()`, not `show()`.** Passing it to `Plotter.show()` caused an error in an earlier version of this project on the user's older PyVista installation.

The current source checks that the returned image has four channels:

```python
if img.ndim != 3 or img.shape[2] != 4:
    raise RuntimeError(...)
```

`RGBA` comprises red, green, blue, and alpha. Ideally, empty background pixels have alpha zero and object pixels have nonzero alpha. A black or white preview in an image viewer **does not by itself prove that alpha is missing**; some viewers show transparency over their own default background.

### 14.8 Avoiding duplicate/cached screenshots

The source explicitly executes:

```python
plotter.camera.Modified()
plotter.render()
```

**after** assigning the camera, then calculates a SHA-256 fingerprint of each screenshot's in-memory pixel bytes. If a later image is **pixel-identical** to an earlier image, it raises a runtime error rather than silently treating the outputs as ten unique renders.

A fingerprint check detects exact equality, not perceptual similarity. This should be interpreted as a practical rendering sanity check, not proof that all screenshots have strongly different visual compositions.

---

## 15. Complete terminal parameter reference

All options below correspond to the uploaded source's `parse_arguments()` function. The first argument, `shape_json`, is positional and required.

| Argument | Default | Meaning / user guidance |
|---|---|---|
| `shape_json` | Required | Input JSON containing a 3D vertices array |
| `--output FILE.png` | `polyhedron_symmetry.png` | Image stem and location; individual views get `_view_###.png` |
| `--report FILE.json` | Image stem + `.axes.json` | Override symmetry report path only |
| `--check-only` | Off | Compute/report symmetry without rendering, prompting for N, or needing PyVista |
| `--n-views N` | Terminal prompt if omitted | Positive number of random-view images |
| `--seed INT` | None | Reproducible pseudo-random viewpoint sampling |
| `--resolution INT` | `2400` | Output image width in pixels; minimum 256 |
| `--height INT` | Same as width | Optional output height in pixels; minimum 256 |
| `--face-color COLOR` | `#70A6D0` | Base color for all polyhedron faces |
| `--edge-color COLOR` | `#EDEFF5` | Color of true physical edge outlines |
| `--opacity FLOAT` | `1.0` | Face opacity 0–1; does **not** control background transparency |
| `--edge-width FLOAT` | `1.8` | Line-rendering width of physical edges |
| `--axis-color COLOR` | `#D02020` | Uniform color for **every** symmetry axis |
| `--axis-length FLOAT` | `1.45` | Half-length of each rod, measured relative to max vertex radius |
| `--axis-thickness FLOAT` | `0.014` | 3D cylinder radius relative to max vertex radius |
| `--axis-opacity FLOAT` | `1.0` | Opacity of rods and their endpoint spheres |
| `--symmetry-tol FLOAT` | `1e-6` | Relative vertex-matching tolerance in proper rotation search |
| `--labels` | Off | Add `C2`, `C3`, etc. text at positive canonical axis ends |
| `--show-legend` | Off | Add axis-order count legend |
| `--show-vertices` | Off | Overlay dots on geometric vertices |
| `--show-xyz` | Off | Display a small xyz orientation widget |
| `--title TEXT` | Empty | Add an upper-left text title |
| `--interactive` | Off | Preview first scene interactively, then proceed to saved random views |
| `--transparent` | Redundant | Accepted but hidden legacy option; backgrounds are already transparent |

### Input validation

The source checks:

- Face and axis opacity must lie in `[0,1]`.
- Image width and height must be at least 256 pixels.
- Symmetry tolerance, axis half-length factor, and axis thickness must be **strictly positive**.
- `--output` must have the `.png` extension.
- Unless `--check-only` is active, N must be a positive integer.

**Arguments that do not exist in this version:** `--camera`, `--background`, `--no-legend`, `--axis-only`, `--lighting`, `--zoom`, and `--view-spacing`. Camera/lights/zoom can instead be customized in the source as described below. `--no-legend` from an earlier version is unnecessary because the legend is already off by default.

### CLI arguments versus source-code edits

Prefer **command-line flags** for anything in the table: they are easy to vary without changing the script and are recorded in shell history. Modify the **Python source** only when you need control over a hardcoded internal behavior, such as the light positions, shading coefficients, camera sampling scheme, color per symmetry order, geometric reconstruction tolerance, rod mesh resolution, or layout.

---

## 16. Ready-to-run customization recipes

These examples can be copied directly after replacing the script/input filenames as necessary. They do not require editing source code.

### Recipe A — Ten transparent publication-quality views

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 10 --seed 42 \
    --output figures/shape.png \
    --resolution 2400 \
    --axis-length 1.55 --axis-thickness 0.012
```

### Recipe B — Very large figures for cropping/layout

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 4 --seed 42 \
    --output figures/high_res.png \
    --resolution 4000 --height 4000 \
    --axis-length 1.55 --axis-thickness 0.012
```

Higher pixel count increases image memory, render time, and output size. The source uses super-sampling antialiasing if supported, so very large images can strain GPU/graphics memory.

### Recipe C — Thin, delicate red axes

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 8 --seed 10 --output figures/thin.png \
    --axis-thickness 0.006 --axis-length 1.45
```

### Recipe D — Thicker, more prominent rods

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 8 --seed 10 --output figures/thick.png \
    --axis-thickness 0.025 --axis-length 1.70
```

### Recipe E — Shorter axes, reducing crowding

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 6 --seed 99 --output figures/short_rods.png \
    --axis-length 1.10 --axis-thickness 0.012
```

The rods remain centered and symmetric about the particle center.

### Recipe F — More transparent faces so interior rods become visible

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 6 --seed 99 --output figures/translucent.png \
    --opacity 0.45 --axis-opacity 1.0
```

This is **different** from transparent background: the polyhedron body itself now allows parts of rods behind its faces to show through. The code attempts depth peeling when face opacity is below 1 and axes exist, but exact transparency appearance can depend on VTK/backend capabilities.

### Recipe G — Change the color palette without altering source

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 10 --seed 42 --output figures/new_palette.png \
    --face-color '#86BFD5' \
    --edge-color '#223C50' \
    --axis-color '#E53E3E'
```

Keep each `#RRGGBB` string in quotes in a shell, because unquoted `#` can act as a shell comment in some contexts.

### Recipe H — Add symmetry labels, legend, or vertex markers

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 3 --seed 3 --output figures/annotated.png \
    --labels --show-legend --show-vertices --show-xyz \
    --title 'Rotational axes'
```

By default **none** of those extras is drawn, which is generally preferable for a clean publication figure with external annotations added later in a graphics/layout program.

### Recipe I — Portrait or landscape output

Portrait:

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 5 --seed 8 --output figures/portrait.png \
    --resolution 1800 --height 2600
```

Landscape:

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 5 --seed 8 --output figures/landscape.png \
    --resolution 3000 --height 1800
```

Because the camera scale is based on the bounding sphere and the height of the image, the **visible horizontal extent depends on the aspect ratio**. If cropping occurs in an extreme aspect ratio, change camera framing as in Section 18.

### Recipe J — Investigate an imperfectly symmetric input shape

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --check-only --symmetry-tol 1e-5 --output check_1e-5.png
```

Compare with the default `1e-6` output. A larger threshold tolerates some coordinate noise but can produce **spurious apparent symmetry** if too large. Always verify the actual vertex residuals for physically important results.

### Recipe K — Test render infrastructure quickly

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 2 --seed 42 --output test_render.png \
    --resolution 600
```

Start small before requesting dozens of 4000-pixel images.

---

## 17. Source-code modification guide

This section answers: **“Exactly which lines/functions should I change to make the image look different?”** Find the indicated function by name rather than relying on fixed line numbers, which will change as the file evolves.

### 17.1 Change the default appearance permanently

Find `parse_arguments()` and modify, for example:

```python
parser.add_argument('--face-color', default='#70A6D0')
parser.add_argument('--edge-color', default='#EDEFF5')
parser.add_argument('--axis-color', default='#D02020')
parser.add_argument('--edge-width', type=float, default=1.8)
parser.add_argument('--axis-length', type=float, default=1.45)
parser.add_argument('--axis-thickness', type=float, default=0.014)
parser.add_argument('--resolution', type=int, default=2400)
```

Change each `default=...` value to make it your permanent default. Any explicitly supplied terminal flag **overrides** the corresponding default without editing source again.

Examples of alternate colors:

| Color intention | Suggested hex |
|---|---|
| Pale cyan-blue face | `#86BFD5` |
| Medium slate-blue face | `#668DB8` |
| Deeper blue face | `#4576A8` |
| Very pale edge | `#EFF3F7` |
| Charcoal edge | `#223C50` |
| Bright red axis | `#ED2A2A` |
| Deep red axis | `#B81822` |

These are illustrative visual choices, not additional presets installed in the code.

### 17.2 Adjust how bright or glossy the polyhedron appears

Find `plotter.add_mesh(surface, ...)` inside `create_scene()`:

```python
ambient=0.20,
diffuse=0.78,
specular=0.08,
specular_power=12,
```

| Parameter | Increasing it typically does... | Practical guidance |
|---|---|---|
| `ambient` | Raises baseline brightness, including shadowed faces | Increase for more uniform blue facets |
| `diffuse` | Makes brightness depend more on each face's orientation to lights | Preserve visible faceting; do not eliminate light contrast |
| `specular` | Strengthens glossy highlights | Keep low for a scientific diagram; raise for polished surfaces |
| `specular_power` | Narrows/tightens highlights in common lighting models | Use with `specular` above zero |

**Example: flatter, less shiny presentation**

```python
ambient=0.35,
diffuse=0.65,
specular=0.00,
specular_power=12,
```

**Example: stronger facet contrast**

```python
ambient=0.12,
diffuse=0.85,
specular=0.05,
specular_power=18,
```

The exact brightness depends on PyVista/VTK and the chosen lights. Treat these as design starting points and compare test renders.

### 17.3 Flat shading versus smooth shading

Inside the same surface `add_mesh`, find:

```python
smooth_shading=False,
```

Keep **`False`** for the present publication style: each planar facet should have a constant or nearly constant shading response. Turning it `True` blends surface normals and makes sharp polyhedral edges appear artificially rounded or smoothed. A low-poly particle figure normally benefits from `False`.

### 17.4 Change the number and locations of light sources

Find in `create_scene()`:

```python
light_specs = [
    ((4, -3, 5), 0.90),
    ((-4, 2, 3), 0.43),
    ((0, 4, -3), 0.25),
]
```

Each entry is `(relative_xyz_position, intensity)`.

- The first light is relatively strong and acts as a key light.
- The second provides fill illumination from another direction.
- The third reduces completely dark back-side facets.

For a more dramatic single-light look:

```python
light_specs = [
    ((4, -3, 5), 1.20),
]
```

For softer illumination from multiple directions:

```python
light_specs = [
    ((4, -3, 5), 0.60),
    ((-4, 2, 3), 0.50),
    ((0, 4, -3), 0.35),
    ((1, 0, -4), 0.20),
]
```

The loop beneath the list converts each relative position into `center + radius * np.array(point)`. To change the **light color**, find:

```python
light = pv.Light(..., color='white', intensity=intensity)
```

Changing `color='white'` to another supported color tints both the body and its specular response. For a clean scientific figure, neutral lights are usually easier to reproduce across a multipanel layout.

### 17.5 Thicker or thinner edge outlines

Usually change **`--edge-width`** at the terminal. To change defaults, edit the `parser.add_argument('--edge-width', ...)` declaration.

To remove physical edges entirely, comment out or remove the following block from `create_scene()`:

```python
plotter.add_mesh(edge_mesh, color=args.edge_color,
                 line_width=args.edge_width, render_lines_as_tubes=True,
                 lighting=False, opacity=0.95, name='physical_edges')
```

You may also set `opacity=0.5` in this block to make outlines subtler. Note that the code uses VTK rendered lines rather than world-space edge cylinders, so line appearance can vary with OpenGL implementation and screenshot size.

### 17.6 Hide the symmetry axes but keep the blue polyhedron

This is **not** a currently implemented CLI flag. Find in `create_scene()`:

```python
for i, axis in enumerate(axes):
    ...
```

Temporarily bypass that loop, for example by enclosing the complete existing axis loop under:

```python
if False:  # temporary: render only the polyhedron
    for i, axis in enumerate(axes):
        ...
```

Use the actual indented original body in place of `...`; the ellipsis in this documentation is only a placeholder. A cleaner permanent extension is to add a boolean `--hide-axes` flag and guard the axis-rendering block with it.

This changes **only what is drawn**; the `.axes.json` report will still contain the detected rotational axes.

### 17.7 Change cylinder smoothness

Find the `pv.Cylinder(...)` construction:

```python
resolution=48,
capping=True,
```

- `resolution=12` or `24`: fewer circumferential sides, faster but visibly faceted rods.
- `resolution=48`: current default, a good general-purpose cylinder.
- `resolution=96`: smoother circular outline, slightly more geometry.

To change the endpoint spheres, find:

```python
theta_resolution=16, phi_resolution=16
```

You can raise both to 24 or 32 for smoother endcaps. `capping=True` closes the actual cylinder; the spheres are **additional cosmetic geometry**.

### 17.8 Change rounded endpoint sizes or remove them

Find the code creating `pv.Sphere` inside the axis loop:

```python
cap = pv.Sphere(
    radius=axis_radius * 1.04,
    center=center + sign * direction * axis_half_length,
    theta_resolution=16,
    phi_resolution=16,
)
```

- `1.04`: current cap radius multiplier.
- `1.0`: cap radius equal to the base rod radius.
- `1.3`: slightly larger rounded endpoint.

To create simple flat-ended cylinders, remove or comment out the entire inner:

```python
for sign in (-1, 1):
    ...
```

block while keeping `capping=True` in `pv.Cylinder`.

### 17.9 Give every axis exactly equal rod thickness

The current implementation makes order-greater-than-two axes 10% thicker:

```python
radius=axis_radius * (1.10 if order > 2 else 1.0),
```

Replace with:

```python
radius=axis_radius,
```

This is a useful change when making a figure intended to illustrate **symmetry axes without implying different physical importance** by thickness. `--axis-color` already keeps their colors identical.

### 17.10 Assign different colors to different rotational orders (optional)

The supplied script deliberately makes **all axes red**, matching the requested visual reference. To make `C2` red, `C3` green, `C4` blue, etc., add a dictionary near the imports:

```python
AXIS_COLORS = {
    2: '#D02020',
    3: '#249F51',
    4: '#265FD1',
    5: '#9333A8',
    6: '#B7791F',
}
```

Then inside `create_scene()` replace:

```python
color = args.axis_color
```

with:

```python
color = AXIS_COLORS.get(order, args.axis_color)
```

This changes only image color. It does **not** alter the actual symmetry computation.

If you enable `--show-legend`, also change the legend entry color in its loop if you want legend swatches to match each order; the source currently uses `args.axis_color` there.

### 17.11 Put axes behind/inside a transparent solid

Use `--opacity 0.3` or `0.5` to expose rods passing through the body. With `--opacity 1.0`, the internal portions of the axes are normally occluded by the solid surface. Rod portions projecting outside the particle remain visible in either case.

The code attempts depth peeling for translucent surfaces:

```python
if args.opacity < 1 and axes:
    try:
        plotter.enable_depth_peeling(number_of_peels=12,
                                     occlusion_ratio=0.05)
    except Exception:
        pass
```

To tune peeling performance/quality, adjust `number_of_peels`. This has graphics-backend dependencies; a larger value is not guaranteed to improve every installation.

### 17.12 Change labels and their appearance

Labels are **off by default**. Add `--labels` if needed. To change their appearance, locate:

```python
plotter.add_point_labels(
    positions, labels,
    font_size=max(16, args.resolution // 105),
    text_color='#202C36',
    shape=None,
    show_points=False,
    always_visible=True,
    name='axis_labels',
)
```

Edit `font_size` to control size, `text_color` to improve contrast, `shape` to introduce or remove a label panel, and `always_visible` to control whether labels can remain visible through geometry. On a transparent PNG, dark text can look poor if the final figure has a dark background; consider choosing label colors only after you know the destination layout.

Label positions are currently:

```python
center + a['direction'] * axis_half_length * 1.12
```

The multiplier `1.12` places labels slightly beyond the positive canonical endpoint. Raise it to give more spacing; lower it to move labels closer to the rod ends.

### 17.13 Change the legend or remove it permanently

The legend is absent unless you use `--show-legend`.

If enabled, `create_scene()` adds:

```python
plotter.add_legend(
    legend_rows,
    border=False,
    loc='upper right',
    size=(0.21, min(0.09 * len(legend_rows) + 0.035, 0.4)),
)
```

Change `loc` for placement or `size` for relative layout dimensions. The exact accepted locations depend on your installed PyVista version.

### 17.14 Draw larger or smaller vertex markers

With `--show-vertices`, the code draws vertices as rendered sphere points:

```python
plotter.add_mesh(pv.PolyData(vertices), color=args.edge_color,
                 point_size=9, render_points_as_spheres=True,
                 name='vertices')
```

Change `point_size=9` to something like `5` or `14`. Markers have **screen-space sizing**, unlike physical vertices or red rod radii.

### 17.15 Change screenshot anti-aliasing

Find:

```python
try:
    plotter.enable_anti_aliasing('ssaa')
except (ValueError, RuntimeError, AttributeError):
    try:
        plotter.enable_anti_aliasing('msaa')
    except (ValueError, RuntimeError, AttributeError):
        pass
```

The script first requests **SSAA** (supersampling) and falls back to **MSAA** where possible. Supersampling can improve edge smoothness but costs memory/time.

For very large batch renders on limited hardware, you can test requesting `msaa` directly. On some systems antialiasing may be unavailable; the code already catches several common errors.

---

## 18. Camera and viewpoint modifications in the source

The existing CLI controls **how many** random views to sample and **which seed** to use. More advanced viewpoint behavior is customized in the camera functions.

### 18.1 Change how far the camera is from the body

Find `set_camera_for_direction(...)`:

```python
position = center + 5.0 * radius * direction
```

Change `5.0` to, for example, `7.0` to move the physical camera farther away. Because the script uses **parallel projection**, doing so generally does **not** make the rendered object smaller; it mostly affects clipping and the geometry of where the camera is placed.

**For visible zoom/framing, adjust `parallel_scale`, not just camera distance.**

### 18.2 Zoom in or out consistently across every view

The source sets:

```python
plotter.camera.parallel_scale = max(axis_half_length * 1.30,
                                    radius * 1.65)
```

This is effectively a half-height in world units for orthographic framing.

- A **smaller** `parallel_scale` gives a **larger** shape in the output (zoom in).
- A **larger** `parallel_scale` gives a **smaller** shape with more empty margin (zoom out).

For tighter framing, try:

```python
plotter.camera.parallel_scale = max(axis_half_length * 1.12,
                                    radius * 1.40)
```

For larger margins around the rods, try:

```python
plotter.camera.parallel_scale = max(axis_half_length * 1.55,
                                    radius * 1.85)
```

The bounding-sphere construction makes these sizes relatively consistent across viewpoints. **Important:** in a very wide or tall image, remember that parallel scale relates to the **vertical** half-span, so final horizontal framing also depends on aspect ratio.

### 18.3 Switch from orthographic to perspective projection

The current code calls:

```python
plotter.enable_parallel_projection()
```

For perspective rendering, replace this with:

```python
plotter.disable_parallel_projection()
```

Then tune camera distance and view angle using available PyVista camera controls, for example:

```python
plotter.camera.view_angle = 30
```

Perspective makes nearer parts of a 3D object appear larger. This can give a more photographic feeling but makes direct geometric size comparison across viewpoints less controlled. The source currently records `'projection': 'orthographic'` in the camera manifest; **update that manifest field** if you change the projection model.

### 18.4 Change what direction counts as “up”

The source normally uses:

```python
view_up = np.array([0.0, 0.0, 1.0])
if abs(np.dot(direction, view_up)) > 0.96:
    view_up = np.array([0.0, 1.0, 0.0])
```

To orient views relative to the global `y` axis instead, use:

```python
view_up = np.array([0.0, 1.0, 0.0])
if abs(np.dot(direction, view_up)) > 0.96:
    view_up = np.array([0.0, 0.0, 1.0])
```

This affects the **roll/orientation of the image plane** for a fixed camera direction. It does not change the particle shape or the detected symmetry axes.

### 18.5 Restrict cameras to the upper hemisphere

If the intended figure should show mostly the top half of the particle, modify `random_camera_directions()`:

**Original:**

```python
z = rng.uniform(-1.0, 1.0, size=n_views)
```

**Upper hemisphere only:**

```python
z = rng.uniform(0.0, 1.0, size=n_views)
```

**Lower hemisphere only:**

```python
z = rng.uniform(-1.0, 0.0, size=n_views)
```

Changing `z` in this manner keeps sampling uniform in *solid angle within the selected hemisphere*.

### 18.6 Restrict cameras to an elevation band

Suppose elevation $e$ is measured above the xy-plane, so $z=\sin e$. To keep all views between elevations $e_{\rm min}$ and $e_{\rm max}$, sample `z` uniformly between their sines:

```python
min_elevation_deg = 20.0
max_elevation_deg = 65.0
zmin = np.sin(np.deg2rad(min_elevation_deg))
zmax = np.sin(np.deg2rad(max_elevation_deg))
z = rng.uniform(zmin, zmax, size=n_views)
```

Replace only the line that samples `z`, leaving the subsequent azimuth and Cartesian conversion unchanged. This yields a uniform solid-angle distribution **within that elevation band**.

### 18.7 Use a fixed, reproducible list of hand-selected viewpoints

You may prefer a carefully chosen set of 3–6 views for publication instead of random directions. Replace the return behavior in `random_camera_directions()` with normalized vectors:

```python
def random_camera_directions(n_views, seed=None):
    directions = np.array([
        [2.5, -3.0, 2.0],
        [1.0,  1.0, 1.0],
        [-2.0, 1.0, 1.5],
        [0.0,  0.0, 1.0],
    ], dtype=float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    if n_views > len(directions):
        raise ValueError('Requested more views than the fixed camera list.')
    return directions[:n_views]
```

This is an **alternative source implementation**, not a built-in flag. The above code returns the first N fixed directions; if N exceeds 4, it raises an error. The `--seed` option has no effect in this mode.

For the random camera system, the `--camera` argument from much earlier single-view versions **does not exist** in v1p4. Hardcoded direction arrays are the simple way to recover that functionality.

### 18.8 Obtain approximately even coverage: Fibonacci-sphere viewpoints

Independent random views can cluster. If you want approximately evenly distributed coverage, you can **replace** `random_camera_directions()` with a Fibonacci-sphere variant:

```python
def random_camera_directions(n_views, seed=None):
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    i = np.arange(n_views, dtype=float)
    z = 1.0 - 2.0 * (i + 0.5) / n_views
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    azimuth = phase + golden_angle * i
    xy = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    return np.column_stack((xy*np.cos(azimuth),
                            xy*np.sin(azimuth), z))
```

This changes the algorithm to **deterministic quasi-uniform point placement plus a seeded azimuth offset**. The directions are no longer independent isotropically random draws. If you adopt it, change the manifest description and project documentation accordingly.

### 18.9 Add a random camera roll about the viewing direction

In the provided source, the viewing direction is random but the image is kept approximately upright relative to a chosen global up vector. Thus the camera does **not** additionally randomize roll.

For all three Euler-like degrees of freedom (direction and in-plane roll), an extension could select a perpendicular `up` vector rotated about the camera direction by another seeded uniform angle. Implementing that requires changing the `set_camera_for_direction()` signature or passing per-view rolls from `render_images()` and recording them in the manifest. Avoid simply sampling three Euler angles uniformly: that generally does not give a uniform distribution over 3D rotations.

### 18.10 Change camera positioning without altering the physics

Everything in this section changes the **projection of the same 3D geometry**. It does not change the `find_proper_rotations()` output, the `.axes.json` report, or the actual rod directions in world coordinates.

---

## 19. Symmetry and geometric tolerance adjustments

### 19.1 Difference between a physical distortion and floating-point noise

A real shape can have exact symmetry that is slightly degraded by stored finite-precision coordinates. Conversely, a very large numerical threshold can make a physically asymmetric shape **appear symmetric** when it should not be.

Use `--symmetry-tol` for numerical uncertainty, not to force the algorithm to produce the symmetry group you expect.

### 19.2 Example tolerance scan

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --check-only --symmetry-tol 1e-7 --output reports/tol_1e-7.png

python3.8 render_polyhedron_rotational_axes.py shape.json \
    --check-only --symmetry-tol 1e-6 --output reports/tol_1e-6.png

python3.8 render_polyhedron_rotational_axes.py shape.json \
    --check-only --symmetry-tol 1e-5 --output reports/tol_1e-5.png
```

Despite the `.png` output argument used for prefix validation, `--check-only` produces **JSON symmetry reports**, not PNG images.

Results that remain stable under small, physically justified changes in tolerance are more convincing than results that appear only at very large tolerance.

### 19.3 Tune independent thresholds only when needed

If your data involve unusual precision or highly distorted shapes, the hardcoded matrix duplicate, closure, and axis-grouping thresholds may also matter. Their source expressions are documented in Section 9. Avoid changing them all simultaneously: change one parameter at a time, record the before-and-after counts, and verify closure and vertex matching.

### 19.4 Do not use image appearance to validate numerical symmetry

The image can look symmetric from some camera orientations even if the body is not. A correct research workflow checks the **numerical report** (rotation group closure, axis count, and vertex permutation matching) before deciding which views to use in the final figure.

---

## 20. Interpreting JSON reports

### 20.1 `NAME.axes.json`

The source builds a report with fields of the form:

```json
{
  "name": "sheared_Cube",
  "n_vertices": 8,
  "n_proper_rotations_including_identity": 4,
  "rotation_axis_counts": {
    "C2": 3
  },
  "center": [0.0, 0.0, 0.0],
  "axes": [
    {
      "order": 2,
      "direction_unit_vector": [-0.6427876, 0.0, 0.7660444],
      "angles_degrees": [180.0]
    }
  ]
}
```

**This is a shortened schematic excerpt:** the actual file contains all three axes and full numerical precision. Do not use the rounded vectors above as input to a high-precision symmetry test.

| JSON field | Meaning |
|---|---|
| `name` | Shape name from JSON metadata, or fallback |
| `n_vertices` | Number of hull vertices retained |
| `n_proper_rotations_including_identity` | Total accepted proper-rotation matrices, identity included |
| `rotation_axis_counts` | Number of **undirected** axis lines of each maximal cyclic order |
| `center` | Vertex-average center used as a common fixed point |
| `axes[*].order` | Maximal cyclic order `n` of that axis |
| `axes[*].direction_unit_vector` | Canonically signed unit direction in the input coordinate frame |
| `axes[*].angles_degrees` | Principal angle magnitudes for nonidentity rotations grouped on that line |

The symmetry JSON does **not** currently store every 3×3 rotation matrix explicitly, only the operation count and extracted axes/angles. If you need matrix-level output for later group-theory calculations, extend `save_report()` to serialize `rotations`.

For example, add an optional field:

```python
'rotation_matrices': [r.tolist() for r in rotations],
```

inside the `report = {...}` dictionary. This makes the full numerical proper-rotation representation available to downstream scripts.

### 20.2 `NAME.views.json`

The camera manifest contains:

```json
{
  "n_views": 10,
  "random_seed": 42,
  "transparent_background": true,
  "projection": "orthographic",
  "rendering_mode": "fresh_scene_per_view",
  "image_resolution": [2400, 2400],
  "views": [
    {
      "view_number": 1,
      "image_filename": "sheared_cube_view_001.png",
      "camera_direction_unit_vector": [0.0, 0.0, 1.0],
      "camera_position": [0.0, 0.0, 5.0],
      "focal_point": [0.0, 0.0, 0.0],
      "camera_view_up": [0.0, 1.0, 0.0]
    }
  ]
}
```

Again, this is a **field-layout example**, not an assertion that seed `42` generated the displayed placeholder vectors. Actual vectors are stored at full precision in the output file and depend on the body radius and shape center.

Interpret the fields as follows:

- `camera_direction_unit_vector`: direction **from shape center to camera** (not the camera's line-of-sight toward the center, which has the opposite sign).
- `camera_position`: world-space camera position, normally `center + 5 * radius * direction`.
- `focal_point`: the center toward which the camera looks.
- `camera_view_up`: chosen up-reference used to orient the image plane.
- `projection`: orthographic in the supplied version.
- `rendering_mode`: indicates that independent scenes were constructed for views.

### 20.3 Read reports in Python

```python
import json

with open('sheared_cube.axes.json', 'r') as f:
    axes_report = json.load(f)

with open('sheared_cube.views.json', 'r') as f:
    views_report = json.load(f)

print(axes_report['rotation_axis_counts'])
for axis in axes_report['axes']:
    print(axis['order'], axis['direction_unit_vector'])

for view in views_report['views']:
    print(view['image_filename'], view['camera_direction_unit_vector'])
```

---

## 21. How to validate your results

### 21.1 Validate the supplied sheared-cube input

Run:

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --check-only --output validation.png
```

The expected qualitative/numerical results are **8 vertices**, **6 polygon faces**, **12 physical edges**, **4 proper rotations including identity**, and **three `C2` lines**.

If you receive another axis count, verify that you are reading the intended JSON, that vertex positions were not edited, and that `--symmetry-tol` has not been changed.

### 21.2 Validate axis normalization and orthogonality

The following small standalone check reads the report:

```python
import json
import numpy as np

with open('validation.axes.json', 'r') as f:
    data = json.load(f)

axes = np.array([x['direction_unit_vector'] for x in data['axes']])
print('Axis lengths:', np.linalg.norm(axes, axis=1))
print('Axis dot-product matrix:')
print(axes @ axes.T)
```

For this sheared cube, lengths should be approximately 1 and the dot-product matrix should be close to the 3×3 identity. (Mutual orthogonality is **specific to this shape's D2 inventory**, not a universal requirement for all polyhedral groups.)

### 21.3 Confirm that PNG transparency is real

Using Pillow for **inspection only** (Pillow is not an application dependency):

```bash
python3.8 -m pip install --user pillow
```

```python
from PIL import Image

im = Image.open('sheared_cube_view_001.png')
print('Mode:', im.mode)
print('Image dimensions:', im.size)
if 'A' in im.getbands():
    alpha = im.getchannel('A')
    print('Alpha range:', alpha.getextrema())
else:
    print('No alpha channel found')
```

For a genuinely transparent image containing both background and rendered geometry, the alpha extrema should normally include 0 and a larger value (often 255).

To check the actual image against a contrasting background, open it in an image viewer that displays a checkerboard for transparent pixels, or composite it over a bright solid color in a graphics editor.

### 21.4 Confirm that all N PNGs are distinct

Use `sha256sum` as a quick filesystem check:

```bash
sha256sum sheared_cube_view_*.png
```

The script also hashes the **uncompressed returned RGBA arrays** while rendering. Different hashes prove nonidentical files or pixels, but not visually well-separated camera angles; inspect selected renders and the camera manifest as well.

### 21.5 Verify the random viewing angles

```python
import json
import numpy as np

with open('sheared_cube.views.json', 'r') as f:
    manifest = json.load(f)

D = np.array([x['camera_direction_unit_vector']
              for x in manifest['views']])
print('Unit norms:', np.linalg.norm(D, axis=1))

# Absolute dot accounts for visually related antipodal directions,
# but opposite directions are still distinct camera positions.
A = np.clip(D @ D.T, -1.0, 1.0)
angles = np.degrees(np.arccos(A))
print('Pairwise separation angles in degrees:')
print(np.round(angles, 1))
```

Values off the diagonal should generally be positive for genuinely different sampled directions. Exact or near-duplicate directions are exceptionally unlikely for small N, but visually similar views can arise even for clearly separated directions on a symmetric shape.

### 21.6 Simple physical/symmetry sanity checks

A robust experimental workflow may additionally test:

- Consistency of axis counts under rigid rotations of the input vertex cloud.
- Consistency under uniform scale changes.
- No unexpected extra axes after a mild geometric distortion.
- Expected loss of specific axes after intentional symmetry-breaking perturbations.
- Closure of the returned rotation matrices if you extend the report to include them.
- Stability under a small, justified range of numerical tolerances.

---

## 22. Troubleshooting

### Problem: “All N images are identical”

**Cause in previous implementation:** a reused PyVista plotter could save a stale framebuffer without drawing the updated camera view.

**What the uploaded v1p4 source already does:**

1. Calls `create_scene(...)` **inside** the per-view loop.
2. Initializes each new render window with `plotter.show(..., auto_close=False)`.
3. Assigns the new camera **after** initialization.
4. Calls `plotter.camera.Modified()` and `plotter.render()`.
5. Captures the image using `screenshot()`.
6. Compares SHA-256 hashes of the returned RGBA arrays to catch pixel-identical captures.

If you still encounter identical frames, inspect the `direction` values printed in the terminal. If they differ, the problem may be the graphics backend, the VTK/PyVista driver, or a stale display implementation rather than the random-number generator.

**Important subtlety:** image hashes can be equal if two truly equivalent symmetric views render exactly the same pixels, although this is unlikely for continuously sampled independent random camera positions. The runtime guard is intentionally conservative.

### Problem: `TypeError: "transparent_background" is an invalid keyword argument for show`

Do **not** pass `transparent_background=True` to `plotter.show()` on installations with this API. Keep it on:

```python
plotter.screenshot(..., transparent_background=True)
```

The documented code already follows this pattern.

### Problem: images appear to have a black or white background

An image viewer may composite transparent pixels over its own black or white canvas. Check `PIL.Image.mode` and the alpha range as described in Section 21. If alpha is missing, inspect PyVista/VTK rendering and screenshot support.

### Problem: axes pass behind the opaque blue polyhedron

That is normal depth occlusion with `--opacity 1.0`. Use `--opacity 0.4` if you wish to see rods through the faces. If you want a physically opaque shape but ghosted hidden axes, that is a different rendering technique and requires source customization.

### Problem: very thin red axes look broken or jagged

Try a larger `--axis-thickness`, higher `--resolution`, and/or improved antialiasing. For reference, `0.012` specifies a cylinder **radius as a fraction of the body radius**, not a number of pixels.

### Problem: the polyhedron appears too small

Reduce `plotter.camera.parallel_scale` as documented in Section 18.2. Increasing `--resolution` alone does not increase the fraction of the image occupied by the shape.

### Problem: symmetry rods are cut off by the image boundary

Increase the camera's `parallel_scale` (zoom out) or reduce `--axis-length`. Extremely tall or wide aspect ratios can also crop objects differently.

### Problem: the script reports only `C1` for a nominally symmetric particle

Check the actual stored vertex geometry. Tiny coordinate noise may destroy matching at the default `--symmetry-tol 1e-6` threshold. Perform a controlled tolerance scan, but do not automatically increase tolerance until desired axes appear.

### Problem: the program reports more symmetries than expected

Check whether the shape's convex hull is actually more symmetric than the intended nonconvex or decorated geometry. Check whether the input has chemically/physically distinct vertices that this purely geometric code ignores. A too-large `--symmetry-tol` is another possible cause.

### Problem: unwanted triangular diagonals appear on faces

Inspect the coplanar triangle-merging tolerance in `polygon_faces_from_hull()`. A too-tight plane tolerance can leave coplanar patches separate; a too-loose tolerance can merge shallow but genuinely different planes.

### Problem: `ConvexHull` or Qhull raises an exception

Check that the data contain finite, unique-enough, noncoplanar 3D vertices. A shape with fewer than four valid vertices or essentially zero 3D volume is not a supported polyhedron.

### Problem: `Detected rotations are not closed as a group`

The candidate set passed some tolerance checks but failed the explicit matrix-composition test. Try reducing `--symmetry-tol`; verify geometry and numerical precision. Changing tolerance can reveal that apparent matches were accidental. A full investigation may require exporting the candidate rotation matrices and examining pairwise residuals.

### Problem: `Expected a transparent RGBA screenshot` / only 3 channels returned

The source requests `transparent_background=True` but the backend did not return the expected four-channel image. Inspect PyVista/VTK versions, the available OpenGL renderer, and the display environment. Do not simply remove this check if a genuinely transparent image is a requirement.

### Problem: OpenGL errors, blank screenshot, or a headless-server crash

Try `--check-only` first to separate numerical and graphics issues. Test a 600-pixel single view from an ordinary desktop terminal. If running remotely, use a supported virtual display or headless render backend. Avoid requesting many 4K screenshots before the environment has passed a small test.

### Problem: `ModuleNotFoundError`

Install the missing package into the same interpreter being used to run the script. For example:

```bash
python3.8 -m pip install --user numpy scipy pyvista
python3.8 -c 'import numpy, scipy, pyvista; print("OK")'
```

`pip` and `python3.8 -m pip` do not necessarily refer to the same Python environment.

### Problem: `No (N, 3) vertex array found in the JSON file`

Check the top-level JSON field name and ensure the array has one row per 3D vertex. Supported keys include `vertices`, `8_vertices`, `12_vertices`, `polyhedron_vertices`, and other names ending in `_vertices`. The data must be convertible to floats.

### Problem: too much empty space in the image

This is controlled by orthographic framing, principally `camera.parallel_scale`. Adjust it in the camera function; transparent background still occupies a full rectangular PNG canvas despite having alpha zero outside the object.

### Problem: the legend is visible even though none was requested

The supplied version does not add a legend unless `--show-legend` is specified. Check that you are executing the right version of the script and that a stale image from a previous run is not being viewed.

---

## 23. Performance and publication workflow

### 23.1 Where the computation time goes

The program has two conceptually separate costs:

1. **Geometry and symmetry detection:** includes convex hull computation, anchor-triple candidate searches, SVD calculations, KD-tree validation, and group checks. The broad candidate enumeration may become costly for a large number of vertices; no simple constant-time bound should be assumed for arbitrary shapes.
2. **Rendering:** creates a fresh PyVista plotter for each view. Runtime generally grows with N, pixel dimensions, antialiasing, geometry complexity, and graphics driver performance.

For a small 8-vertex convex polyhedron, rendering at 2400–4000 pixels is typically the more substantial part of the workflow.

### 23.2 Suggested two-pass workflow

**Pass 1: inspect many low-resolution viewpoints**

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 20 --seed 17 \
    --output screening/sheared_cube.png \
    --resolution 700
```

Review the images and the associated `screening/sheared_cube.views.json` file.

**Pass 2: choose a publication presentation**

The current CLI generates new random directions on every run unless the seed is fixed. Re-running with the **same seed and N** preserves the camera directions (for the same underlying random generator behavior), but the script renders all N views again. To render only a specific shortlisted direction, insert it into the fixed-direction variant in Section 18.7, or extend the CLI to load camera vectors from a manifest.

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 20 --seed 17 \
    --output final/sheared_cube.png \
    --resolution 3000
```

### 23.3 Choose the figure size appropriately

At 2400 × 2400 pixels, a decoded RGBA image uses approximately

$$
2400\times2400\times4=23{,}040{,}000\ \text{bytes},
$$

before renderer buffers, antialiasing buffers, mesh memory, or PNG compression. At 4000 × 4000, a single decoded RGBA array is about 64 MB. VTK often allocates additional buffers beyond these figures.

**Do not assume the compressed PNG file size equals GPU-memory usage.** Large images and SSAA can demand considerably more memory than the output file indicates.

### 23.4 Consistency across multi-panel scientific figures

For a paper comparing two or more polyhedra, choose:

- The same resolution and aspect ratio.
- The same face and axis colors.
- The same lighting coefficients and light positions relative to each body's radius.
- The same `--axis-length`, `--axis-thickness`, and `--edge-width` settings, with awareness that physical size scaling uses each body's radius.
- A comparable camera framing policy.
- External panel labels if you need precise typographic consistency.

The script already enforces a transparent background, so the resulting PNGs can be placed on a uniform figure background in LaTeX, Inkscape, Illustrator, PowerPoint, or a graphics editor without a mismatched rectangular backdrop.

### 23.5 Research reproducibility

Archive together:

```text
script .py
input shape .json
command line used
output .axes.json
output .views.json
selected rendered .png files
Python / NumPy / SciPy / PyVista / VTK versions
```

The `.views.json` file is especially useful for showing that visually different illustrations arise solely from camera changes rather than different particle geometry.

---

## 24. Function-by-function source map

| Function | Responsibility | When you would edit it |
|---|---|---|
| `read_vertices` | JSON parsing, hull-vertex removal, center | Support different input JSON layouts or alternate centering |
| `polygon_faces_from_hull` | Coplanar face merge and physical-edge list | Fix edge diagonals or face reconstruction tolerance |
| `find_proper_rotations` | Rotation candidates, proper SVD alignment, full-set validation, closure | Change symmetry search or numeric criteria |
| `canonical_axis` | Reproducible sign for undirected lines | Change how axis vectors are presented in reports |
| `symmetry_axes` | Rotation-vector extraction, grouping, `Cn` assignment | Change axis equivalence tolerance, order reporting |
| `save_report` | Numerical symmetry JSON output | Add full matrices, residuals, or extra metadata |
| `random_camera_directions` | Random camera sampling on a sphere | Restrict hemisphere, add fixed/evenly spaced view list |
| `set_camera_for_direction` | Camera position, up vector, parallel projection, zoom | Camera distance, image framing, roll, perspective |
| `create_scene` | Blue face mesh, edges, rods, lights, typography | Most visual customization beyond existing CLI flags |
| `render_images` | Per-view plotters, screenshots, hashes, file naming, camera manifest | Batch rendering workflow and image validation |
| `parse_arguments` | CLI option definitions/defaults | Change default values or add new flags |
| `main` | Argument checks, pipeline orchestration, console messages | Change workflow, report path logic, or validation |

### Safe-edit principle

If you only want to change **how the figure looks**, prefer editing `parse_arguments()` and `create_scene()`, or use CLI flags. **Do not alter `find_proper_rotations()`, `symmetry_axes()`, or `canonical_axis()` for cosmetic changes.** Those functions control the mathematical meaning of the axes and numerical results.

If you want to change **which viewpoints are produced**, edit `random_camera_directions()` and/or `set_camera_for_direction()`; changing the vertex coordinates to obtain a nicer angle would alter the physical object, not merely its view.

---

## 25. Limitations and possible extensions

### 25.1 Convex objects only

The construction is based on `ConvexHull`. Concave polyhedra require explicit face connectivity, interior/exterior topology, and a different face/edge reconstruction strategy.

### 25.2 Vertex-symmetry rather than labeled-particle symmetry

The algorithm is correct for a **geometric point cloud regarded as a set of equivalent vertices**, but may overestimate physically allowed rotations of a colored, chemically labeled, patchy, or otherwise decorated particle. A future version could associate a type/color to each vertex and require the permutation to preserve those types.

### 25.3 Numeric rather than exact algebraic group computation

The accepted matrices are floating-point approximations. Closure, uniqueness, and axis-grouping tests use thresholds. For exact crystallographic or polyhedral classification, an additional symbolic/group-theory verification stage may be useful.

### 25.4 No full point-group naming

The script identifies proper rotations and maximum cyclic orders, but does not attempt to systematically assign a complete point-group symbol, test mirrors/inversion, or construct character tables. The current output is sufficient for visualizing rotational axes.

### 25.5 Viewing angles are random, not optimized

N independent directions provide broad stochastic coverage but do not guarantee attractive or maximally separated views. Possible additions:

- `--camera x,y,z` for an explicitly chosen single view;
- `--cameras-file camera_list.json` to render preselected directions;
- a Fibonacci-sphere or geodesic sampling option;
- minimum angular separation between sampled directions;
- a contact-sheet preview for choosing the best panels;
- optional random in-plane roll;
- symmetry-inequivalent camera directions, reducing redundant views for high-symmetry shapes.

These are extension ideas, **not existing features** in the uploaded script.

### 25.6 Image and renderer customization

Possible future additions:

- A command-line `--zoom` or `--parallel-scale` parameter.
- A `--light-preset` option.
- Optional line-only or silhouette-only rendering.
- Separate visibility control for rods inside/outside faces.
- Color coding by rotation order or by equivalent axes.
- An export mode for geometric axes and meshes to a 3D format.
- Batch traversal of an entire directory of shape JSON files.
- Automatic rendering of several shapes using matched views and scales.

### 25.7 Potential source-level optimizations

Creating a fresh `pv.Plotter` for every view is deliberately robust against stale frames but can be slower than reusing a correctly refreshed renderer. On a tested modern PyVista/VTK installation, it may be possible to reuse the scene with an explicit redraw, but any such optimization should retain the **distinct-frame screenshot check** so the earlier bug is not reintroduced.

---

## 26. Reproducibility checklist and citation guidance

Before using the generated images in a manuscript, thesis, or technical report, verify the following:

- [ ] The JSON contains the intended vertices, without unexpected duplicates or interior-only points.
- [ ] The intended particle is convex, or you explicitly accept a convex-hull representation.
- [ ] The detected rotational group count is mathematically plausible.
- [ ] The detected axis directions and orders are stable at an appropriate numerical tolerance.
- [ ] All intended axes are visible and have the desired red color.
- [ ] The PNG background has a real alpha channel.
- [ ] Different random view outputs are genuinely different and visually useful.
- [ ] `--seed` and the camera manifest are preserved.
- [ ] Lighting, colors, and geometry scales are consistent across the figures being compared.
- [ ] Final publication figures were visually inspected at the final physical print/display size.
- [ ] The script version and rendering software versions are recorded.

### Suggested methods-style description

> We identified the proper rotational symmetries of each convex polyhedral particle directly from its vertex coordinates. Candidate rotations were reconstructed from Gram-matrix-compatible triples of vertex vectors using determinant-constrained orthogonal Procrustes alignment and retained only when they mapped the full vertex set bijectively onto itself within a numerical tolerance. The resulting operations were grouped by common undirected rotation axes and used to determine their maximal cyclic orders. Convex-hull triangles were merged into planar polygonal faces to reconstruct physical edges. Polyhedra and symmetry-axis rods were visualized with PyVista under orthographic projection, using randomly sampled viewing directions and transparent backgrounds.

Adjust this description to match the exact parameters, shapes, and checks used for a specific published study. This README is software documentation, not a citation to a separately published scientific method or original paper.

---

## One-page parameter cheat sheet

| I want to change... | Easiest way | If I must edit the source, look in... |
|---|---|---|
| Number of images | `--n-views 10` | `parse_arguments`, `render_images` |
| Random viewpoints | `--seed 42` | `random_camera_directions` |
| Restrict camera hemisphere | Source edit | `random_camera_directions` |
| Use hand-selected camera angles | Source edit | `random_camera_directions` |
| Make views more evenly spaced | Source edit | `random_camera_directions` |
| Save higher-resolution PNG | `--resolution 4000` | `parse_arguments` |
| Change aspect ratio | `--resolution 3000 --height 2000` | `parse_arguments` |
| Transparent PNG background | Already enabled | `plotter.screenshot` in `render_images` |
| Blue body color | `--face-color '#86BFD5'` | `create_scene` / defaults in `parse_arguments` |
| Red axis color | `--axis-color '#D02020'` | `create_scene` / defaults in `parse_arguments` |
| Body opacity | `--opacity 0.5` | `create_scene` |
| Axis opacity | `--axis-opacity 0.8` | `create_scene` |
| Physical edge color | `--edge-color '#EDEFF5'` | `create_scene` |
| Edge thickness | `--edge-width 2.5` | `create_scene` |
| Red-axis length | `--axis-length 1.55` | `create_scene` |
| Red-axis thickness | `--axis-thickness 0.012` | `create_scene` |
| Equalize all axis widths | Source edit | `pv.Cylinder(radius=...)` in `create_scene` |
| Endpoint shape | Source edit | `pv.Sphere(...)` in `create_scene` |
| Facet gloss/brightness | Source edit | Surface `add_mesh` in `create_scene` |
| Lighting directions/intensities | Source edit | `light_specs` in `create_scene` |
| Zoom/framing | Source edit | `camera.parallel_scale` in `set_camera_for_direction` |
| Perspective instead of orthographic | Source edit | `set_camera_for_direction` (+ manifest) |
| Camera up/roll | Source edit | `set_camera_for_direction` |
| More visible interior axes | `--opacity 0.4` | `create_scene` / depth peeling |
| Symmetry tolerance | `--symmetry-tol 1e-6` | `find_proper_rotations` |
| Face-merging tolerance | Source edit | `polygon_faces_from_hull` |
| Label axes | `--labels` | `create_scene` |
| Show legend | `--show-legend` | `create_scene` |
| Add title | `--title 'My shape'` | `create_scene` |
| Draw vertex dots | `--show-vertices` | `create_scene` |
| Save all rotation matrices | Source edit | `save_report` |
| Prevent repeated stale images | Already implemented | `render_images` |

**Recommended stable starting command:**

```bash
python3.8 render_polyhedron_rotational_axes.py shape.json \
    --n-views 10 --seed 42 \
    --output figures/polyhedron.png \
    --resolution 2400 \
    --axis-length 1.55 \
    --axis-thickness 0.012 \
    --face-color '#70A6D0' \
    --edge-color '#EDEFF5' \
    --axis-color '#D02020'
```

End of README.
