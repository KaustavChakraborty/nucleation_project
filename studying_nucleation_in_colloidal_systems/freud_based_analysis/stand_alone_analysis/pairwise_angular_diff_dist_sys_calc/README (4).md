# Global Pairwise-Orientation Histogram for HOOMD GSD Trajectories

**Script:** `hist_pairwise_angles_entire_sys.py`
**Version:** 1.0 (standalone)
**Language:** Python 3.9+ (uses `from __future__ import annotations`, so PEP 604/585 annotations are safe on 3.9)
**Lines of code:** 2357
**Entry point:** `main()`, invoked via `raise SystemExit(main())`

---

## Table of Contents

1. [What this program computes](#1-what-this-program-computes)
2. [Scientific background and precise definitions](#2-scientific-background-and-precise-definitions)
3. [Installation and dependencies](#3-installation-and-dependencies)
4. [Inputs](#4-inputs)
5. [Invocation: CLI reference](#5-invocation-cli-reference)
6. [Interactive prompts](#6-interactive-prompts)
7. [End-to-end workflow](#7-end-to-end-workflow)
8. [Part I — Shape processing and symmetry detection](#8-part-i--shape-processing-and-symmetry-detection)
9. [Part II — Trajectory processing and histogramming](#9-part-ii--trajectory-processing-and-histogramming)
10. [Part III — Aggregation across frames](#10-part-iii--aggregation-across-frames)
11. [Outputs: complete file and column reference](#11-outputs-complete-file-and-column-reference)
12. [Console output reference](#12-console-output-reference)
13. [Internal data structures](#13-internal-data-structures)
14. [Complete validation and error catalogue](#14-complete-validation-and-error-catalogue)
15. [Computational complexity and memory model](#15-computational-complexity-and-memory-model)
16. [Reproducibility and determinism](#16-reproducibility-and-determinism)
17. [Differences from the legacy pipeline](#17-differences-from-the-legacy-pipeline)
18. [Known limitations, caveats and gotchas](#18-known-limitations-caveats-and-gotchas)
19. [Worked example](#19-worked-example)
20. [Troubleshooting](#20-troubleshooting)
21. [Function-by-function index](#21-function-by-function-index)

---

## 1. What this program computes

Given

* a **HOOMD-blue GSD trajectory** containing per-particle unit quaternions (`particles.orientation`), and
* a **shape JSON** describing the convex polyhedral particle by its vertices,

the program produces the **global distribution of symmetry-reduced pairwise misorientation angles** between particles, averaged over a chosen set of trajectory frames.

Informally: for every unordered pair of particles $(i,j)$ in a frame, it asks *"by what minimum angle must particle $i$ be rotated to look identical to particle $j$, given that the particle itself has rotational symmetry?"* It then histograms those angles and averages the per-frame normalised histograms.

The result is a curve $P(\theta_{ij})$ — a probability **per bin** (not a probability density) as a function of misorientation angle in degrees. This is the standard diagnostic for orientational order in dense colloidal/hard-particle simulations: a uniform-ish curve indicates orientational disorder, whereas sharp peaks (especially at $\theta = 0$) indicate orientational ordering or a rotator/crystal phase.

The program is deliberately written as an **auditable, self-validating replacement** for an older multi-module pipeline. Every geometric and statistical assumption is checked at runtime, and the run is failed loudly rather than silently producing a plausible-looking but wrong curve.

### The pipeline in one line

```
shape JSON
  → convex hull decomposition (coplanar triangle merging)
  → topology validation against user-supplied edge/face counts
  → candidate rotation axes + candidate rotation angles
  → proper rotational symmetry operations (validated as a group)
  → equivalent quaternion set containing both +q and −q
  → freud.environment.AngularSeparationGlobal
  → unique particle pairs i < j only
  → per-frame conditional histogram within the plotted range
  → equal-weight arithmetic average over the selected frames
  → CSV + CSV + CSV + JSON + PNG
```

---

## 2. Scientific background and precise definitions

### 2.1 Orientations as unit quaternions

HOOMD stores each particle's orientation as a **scalar-first** unit quaternion
$q = (w, x, y, z)$, $\lVert q \rVert = 1$, representing a rotation from the particle's body frame to the lab frame. The set of unit quaternions $S^3$ is a **double cover** of the rotation group $SO(3)$:

$$q \quad\text{and}\quad -q \quad\text{represent the identical physical rotation.}$$

This double cover is the reason the program explicitly emits both signs of every symmetry quaternion (§8.9).

### 2.2 Misorientation between two particles

For two orientations $q_i$ and $q_j$, the relative rotation is

$$\Delta_{ij} = q_i^{-1} \, q_j = \bar{q_i} \, q_j \quad (\text{for unit } q_i),$$

and its rotation angle is

$$\theta(\Delta) = 2\arccos\bigl(\lvert \mathrm{Re}(\Delta) \rvert\bigr) \in [0°, 180°].$$

The absolute value on the scalar part is what makes the angle insensitive to the $q \leftrightarrow -q$ ambiguity.

### 2.3 Symmetry reduction

A physical polyhedron with a nontrivial **proper rotational symmetry group** $G \subset SO(3)$ (order $|G| = n$) is indistinguishable from itself under any $g \in G$. Two particles whose orientations differ only by an element of $G$ are physically in the *same* orientation. The physically meaningful misorientation is therefore the minimum over the symmetry group:

$$\theta_{ij} = \min_{g \in G} \; \theta\bigl(q_i^{-1} \, q_j \, g\bigr).$$

Consequences:

* $\theta_{ij} \in [0°, \theta_{\max}(G)]$ where $\theta_{\max}(G) \le 180°$ and **shrinks as the symmetry group grows**. For a cube ($|G| = 24$, group $O$) the maximum misorientation is $\approx 62.8°$; for a tetrahedron ($|G| = 12$, group $T$) it is $\approx 75.5°$; for a trivial group ($|G| = 1$) it is $180°$.
* This is precisely why the default suggested plotting range of $120°$ often contains *all* pairs for a symmetric particle: the `pairs_outside_range` diagnostic will then be zero.
* $\theta_{ij}$ is symmetric in $i \leftrightarrow j$ **when $G$ is a group** (which the program verifies), which is why counting each pair once is correct and not an approximation.

### 2.4 Why only *proper* rotations

Only proper rotations ($\det R = +1$) are used. Mirror reflections, inversions and rotoreflections are deliberately excluded because particle orientations and unit quaternions parameterise $SO(3)$, not the full point group $O(3)$. Including an improper operation would be a category error — there is no quaternion representing a reflection, and a physical rigid particle cannot be reflected into itself by any motion.

Note this means the program detects the **rotation group** $G$, not the full point group. For a cube the program should find $|G| = 24$, not $|O_h| = 48$.

### 2.5 Pair definition

Only **unique unordered pairs** are retained:

$$\{(i,j) : 0 \le i < j < M\}, \qquad \lvert \text{pairs} \rvert = \frac{M(M-1)}{2}.$$

Self-pairs $i = j$ (which would trivially contribute $\theta = 0$ and produce a spurious delta spike at the origin) are excluded, and $(i,j)$ / $(j,i)$ are never double counted. The program **hard-asserts** the exact count $M(M-1)/2$ at the end of every frame.

### 2.6 Normalisation convention (important)

Each frame $f$ is normalised **independently and conditionally inside the plotted range**:

$$P_f(k) = \frac{h_f(k)}{\sum_{l} h_f(l)} = \frac{h_f(k)}{N_f^{\text{inside}}}$$

where $h_f(k)$ is the raw integer count in bin $k$ and $N_f^{\text{inside}}$ is the number of unique pairs whose angle falls in $[0, \theta_{\max}]$. Pairs above $\theta_{\max}$ are excluded from **both** numerator and denominator.

The final reported curve is the **equal-weight arithmetic mean over frames**:

$$\overline{P}(k) = \frac{1}{F}\sum_{f=1}^{F} P_f(k).$$

Therefore $\sum_k \overline{P}(k) = 1$ exactly (to floating-point roundoff), **even if some pairs lay above the plotted maximum**. This is a deliberate choice and is verified at runtime (a `RuntimeError` is raised if the sum deviates from 1 by more than $10^{-12}$).

> **Read this carefully:** the y-axis is *conditional* probability. If 30% of your pairs lie above `max_angle`, the plotted curve is the distribution *given* that the angle is below `max_angle`, renormalised to 1. The `outside_range_fraction` field in the metadata JSON tells you exactly how much mass was discarded. If that number is not small, either widen `max_angle` or interpret the curve conditionally.

### 2.7 Equal-weight vs. pooled averaging

Two aggregations are computed:

| Quantity | Formula | Role |
|---|---|---|
| **Equal-weight mean** | $\overline{P}(k) = \frac1F\sum_f P_f(k)$ | **Primary**, plotted, first CSV column of interest |
| **Pooled** | $P_{\text{pool}}(k) = \frac{\sum_f h_f(k)}{\sum_l \sum_f h_f(l)}$ | Diagnostic only, written to CSV, not plotted |

They differ only when frames have differing $N_f^{\text{inside}}$ (different particle counts or different fractions outside range). Pooled weights frames in proportion to their in-range pair count; equal-weight treats every frame as one independent sample. Their agreement is a useful sanity check.

### 2.8 The x-coordinate convention

The x value written to file and plotted is the **LEFT BIN EDGE**, $x_k = \text{bin\_edges}[k]$, not the bin centre. This preserves the original plotting convention. If you want centres, add half a bin width: $\Delta = \theta_{\max}/B$.

The y value is **probability mass per bin**. It is **not** divided by the bin width, so it is not a probability density. Changing the number of bins changes the y-scale.

---

## 3. Installation and dependencies

```bash
pip install numpy scipy matplotlib gsd freud-analysis
```

### Import strategy (two-tier, deliberate)

| Tier | Packages | When imported | Why |
|---|---|---|---|
| **Startup** | `numpy`, `scipy` (`optimize.linear_sum_assignment`, `sparse.csgraph.connected_components`, `spatial.ConvexHull`, `spatial.cKDTree`, `spatial.distance_matrix`, `spatial.transform.Rotation`) | Module import time, in a `try/except ImportError` that raises `SystemExit` with an install hint | These drive the geometry/symmetry core |
| **Runtime** | `freud`, `gsd.hoomd`, `matplotlib.pyplot` | Lazily inside `import_runtime_packages()`, called as STEP 3 of `main()` | Lets the symmetry code be imported, inspected and unit-tested **without** freud or GSD installed |

The lazy tier is a genuine design feature: you can `import hist_pairwise_angles_rigorous_standalone_v1p0` and exercise `detect_proper_rotational_symmetries()` on a shape JSON in an environment that has no freud build.

Only the standard library modules `argparse`, `json`, `math`, `sys`, `dataclasses`, `pathlib`, `typing` are otherwise used.

---

## 4. Inputs

### 4.1 GSD trajectory (positional argument 1)

A HOOMD-blue GSD file. The program reads **only** `frame.particles.orientation` — it does not read positions, box, types, diameters, or any other field. Consequences:

* Particle **type is ignored**. In a multi-component system all types are pooled.
* Particle **position is ignored**. This is a *global* (all-pairs) calculation with no neighbour cutoff, no radial resolution and no periodic-image handling. It is not $g(r, \theta)$.
* The file is opened read-only via `gsd.hoomd.open(name=..., mode="r")`, with a positional-argument fallback (`gsd.hoomd.open(path, "r")`) for older GSD releases that reject keyword arguments.
* Frames are accessed by index (`trajectory[frame_index]`), so the file handle must support random access — true for standard GSD.

### 4.2 Shape JSON (positional argument 2)

Any JSON file containing an $N \times 3$ array of vertex coordinates. The reader is **schema-agnostic** and searches recursively (§8.1), so all of the following work:

```json
{"vertices": [[1,1,1], [1,-1,-1], [-1,1,-1], [-1,-1,1]]}
```
```json
{"shape": {"type": "ConvexPolyhedron", "vertices": [[...], ...]}}
```
```json
[{"vertices": [[...], ...]}]
```

Requirements enforced on the selected array:

* shape $(N, 3)$ with $N \ge 4$;
* all entries finite (no NaN/inf);
* **no duplicate or numerically indistinguishable vertices** — checked with a KD-tree at tolerance $\max(10^{-12} R, 10^{-14})$ where $R$ is the maximum centred vertex radius. Duplicates are rejected because they make the convex topology and the one-to-one symmetry permutation ambiguous;
* not all collapsed to a single point.

Units are arbitrary but **the symmetry matching tolerance $10^{-p}$ is absolute in these same units** — see §18.1.

---

## 5. Invocation: CLI reference

```bash
python hist_pairwise_angles_rigorous_standalone_v1p0.py TRAJECTORY.gsd SHAPE.json [options]
```

### Positional arguments

| Argument | Description |
|---|---|
| `trajectory` | Input HOOMD GSD trajectory |
| `shape` | Particle shape JSON containing vertices |

### Optional arguments

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--frames` | int | `None` → prompt (suggests `1`) | Number of **consecutive final** frames to average |
| `--particles` | int | `None` → prompt (suggests min available) | Number of particles taken from the **start** of each frame |
| `--edges` | int | `None` → prompt (suggests `25`) | Known number of polyhedron edges (validation target) |
| `--faces` | int | `None` → prompt (suggests `15`) | Known number of polyhedron faces (validation target) |
| `--precision` | int | `None` → prompt (suggests `2`) | $p$ in the vertex-matching tolerance $\varepsilon = 10^{-p}$ |
| `--bins` | int | `None` → prompt (suggests `50`) | Number of histogram bins |
| `--max-angle` | float | `None` → prompt (suggests `120.0`) | Maximum plotted misorientation angle in degrees |
| `--block-size` | int | `128` | Query orientations passed to freud per call; memory knob only |
| `--output-dir` | str | `None` → directory of the trajectory | Where to write outputs (created with `parents=True, exist_ok=True`) |
| `--no-show` | flag | off | Save the PNG without opening an interactive matplotlib window |

**Semantics of `None` vs. supplied:** `resolve_or_prompt()` implements the rule — if the CLI value is `None`, prompt interactively; if it is supplied, validate it and raise `ValueError` immediately on failure (no silent clamping, no fallback prompt). Supplying **all seven** scientific flags makes the run fully non-interactive and batch/queue safe.

### Constants (edit at the top of the file to change suggestions)

```python
CURRENT_SUGGESTED_PRECISION   = 2       # historical tol_for_inv_quat_calc
CURRENT_SUGGESTED_NUM_EDGES   = 25      # from the supplied EPD parameter file
CURRENT_SUGGESTED_NUM_FACES   = 15      # from the supplied EPD parameter file
CURRENT_SUGGESTED_NUM_BINS    = 50      # configured value in parameter file
CURRENT_SUGGESTED_MAX_ANGLE_DEG = 120.0
CURRENT_SUGGESTED_FRAMES      = 1
DEFAULT_BLOCK_SIZE            = 128
```

> Note that the suggested topology $E=25$, $F=15$ implies, by Euler's formula $V - E + F = 2$, a polyhedron with $V = 12$ vertices. The program does **not** itself check Euler's formula, but this is a useful consistency check on the numbers you type.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Calculation completed and all outputs written |
| `1` | A controlled error was caught and printed to `stderr` as `ERROR: <message>` |

The caught exception classes are exactly `FileNotFoundError`, `ImportError`, `ValueError`, `RuntimeError`, `OSError`, `json.JSONDecodeError`. Anything else (e.g. `KeyboardInterrupt`, `MemoryError`, an unexpected library exception) propagates with a full traceback.

---

## 6. Interactive prompts

`prompt_value()` loops until valid input is received. An **empty line accepts the default** shown in square brackets. A conversion failure (`TypeError`/`ValueError`) or a validator rejection reprints the error message and re-prompts — it never crashes on bad typing.

Prompts appear in this exact order:

| # | Prompt | Validator | Default |
|---|---|---|---|
| 1 | How many consecutive final frames to include in the equal-weight average? | $1 \le v \le$ total frames | `1` |
| 2 | How many particles from the beginning of each selected frame? | $2 \le v \le$ min available across selected frames | min available |
| 3 | Known number of polyhedron **edges** | $v \ge 3$ | `25` |
| 4 | Known number of polyhedron **faces** | $v \ge 4$ | `15` |
| 5 | Invariant-quaternion symmetry precision $p$ (tolerance $10^{-p}$) | $0 \le v \le 15$ | `2` |
| 6 | Number of histogram bins | $v \ge 1$ | `50` |
| 7 | Maximum plotted misorientation angle (degrees) | $0 < v \le 180$ | `120.0` |

After prompt 5, if $p \le 2$ the program prints a **SCIENTIFIC CAUTION** advising a sensitivity comparison at $p = 3$ and $p = 4$ before publishing. It does not block.

---

## 7. End-to-end workflow

`main()` executes the following numbered steps (the in-code comments use the same numbering; note the comments skip a "STEP 21" label — cosmetic only):

| Step | Action |
|---|---|
| 1 | Parse the CLI |
| 2 | Enter the single controlled `try/except` error boundary |
| 3 | Import `freud`, `gsd.hoomd`, `matplotlib.pyplot` |
| 4 | Validate `--block-size >= 1` **before** touching large data |
| 5 | Resolve and open the GSD; resolve the shape path (content checked later) |
| 6 | Read `len(trajectory)`; reject an empty trajectory; print the frame summary |
| 7 | Obtain `frames_to_average`; compute `selected_frame_indices = range(T - F, T)` |
| 8 | For each selected frame, read the orientation array, verify shape $(N,4)$, record $N$; report min/max across frames |
| 9 | Obtain `selected_particle_count` $M$; print that indices $0 \ldots M-1$ are used |
| 10 | Obtain `expected_edges`, `expected_faces` |
| 11 | Obtain `precision_exponent` $p$; print the caution if $p \le 2$ |
| 12 | Obtain `num_bins` $B$ and `maximum_angle_deg`; print the full histogram definition block |
| 13 | `read_shape_vertices()` |
| 14 | `detect_proper_rotational_symmetries()` — the whole of Part I |
| 15 | `compute_frame_averaged_histogram()` — the whole of Parts II and III |
| 16 | Rebuild `bin_edges = linspace(0, max_angle, B+1)` for the output writers |
| 17 | Choose/create the output directory; build the filename prefix |
| 18 | `save_symmetry_outputs()` |
| 19 | `save_histogram_outputs()` (summary CSV, per-frame CSV, PNG) |
| 20 | `save_metadata()` (JSON) |
| — | Print final validation lines and all output paths; return `0` |
| 22 | `except` clause converts expected failures to `ERROR: ...` on stderr and returns `1` |

Note the ordering: **all user input is collected before any expensive work begins**, and the symmetry detection (Step 14) runs before the trajectory loop (Step 15). If your shape or topology is wrong, the program fails within seconds rather than after hours of histogramming.

### Frame selection rule

$$\text{selected} = \{T - F,\ T - F + 1,\ \ldots,\ T - 1\}$$

i.e. the **final $F$ consecutive frames**, zero-based. There is no stride, no thinning, and no equilibration-detection. If you want a different subset you must edit `selected_frame_indices` in `main()`.

### Particle selection rule

The **first $M$ particles by index** in every frame: `array[:requested_particles]`. Not random sampling, not type filtering. This is deterministic and reproducible, but see §18.5 for when it can bias the result.

---

## 8. Part I — Shape processing and symmetry detection

All of this lives in `detect_proper_rotational_symmetries()` and its helpers, organised as 13 internal stages.

### 8.1 Reading the vertices (`read_shape_vertices`)

Three collaborating functions:

**`_as_vertex_array(value)`** — attempts `np.asarray(value, dtype=float)` and returns it only if `ndim == 2`, `shape[1] == 3`, `shape[0] >= 4`, and all entries finite. Otherwise `None`.

**`_collect_vertex_candidates(obj, key_hint)`** — recursive descent over the parsed JSON:

* If the node is itself a valid vertex array, emit one candidate and **stop descending**.
* If it is a `dict`, sort items so that keys containing `"vert"` (case-insensitive) come first, then alphabetically — a deterministic traversal order — and recurse into each value, passing the key name down as `key_hint`.
* If it is a `list`/`tuple`, recurse into each element, keeping the inherited `key_hint`.

Each candidate is scored as

$$\text{score} = \underbrace{10000 \cdot \mathbb{1}[\text{"vert"} \in \text{key}]}_{\text{name priority}} + \underbrace{N_{\text{rows}}}_{\text{size tiebreak}}$$

so a small array under a key named `vertices` beats a large unrelated $N\times3$ array (e.g. a position table) elsewhere in the file, while among equally-named candidates the largest wins.

**`read_shape_vertices(path)`** — expands `~`, resolves to an absolute path, raises `FileNotFoundError` if absent, loads JSON (a malformed file surfaces as `json.JSONDecodeError`, which `main()` catches), picks the maximum-scoring candidate, casts to `float64`, then runs the degeneracy and duplicate checks of §4.2. It prints the winning key hint and the vertex count so you can verify the right array was chosen.

### 8.2 Stage 1 — interpreting the precision

$$\varepsilon_{\text{match}} = 10^{-p}$$

an **absolute Euclidean distance in shape-coordinate units**. A negative $p$ raises `ValueError`.

**Historical note that matters:** in the legacy code the value `tolerance_for_inv_quat_of_body_calc = 2` was used as a *decimal rounding precision* — accepted quaternion components were rounded to 2 decimal places. This program **preserves the numeric value as the suggested default and preserves its interpretation as $10^{-2} = 0.01$ geometric tolerance, but does NOT round the accepted quaternions.** Quaternions are retained at full double precision. $p$ now controls *only* the acceptance test.

### 8.3 Stage 2 — centring the shape

A rigid-body rotation must act about the particle centre. Rotating about the coordinate origin is valid only if the shape file is already centred there. So:

$$\mathbf{c} = \frac{1}{N}\sum_{a=1}^{N} \mathbf{r}_a, \qquad \tilde{\mathbf{r}}_a = \mathbf{r}_a - \mathbf{c}.$$

The program prints $\mathbf{c}$ and $\lVert\mathbf{c}\rVert$, reports **CENTRE CHECK: PASSED** if $\lVert\mathbf{c}\rVert \le \varepsilon_{\text{match}}$ and otherwise prints an informational message saying it will translate. Either way it proceeds with the centred coordinates. It then verifies the recentring worked: if $\lVert \frac1N\sum \tilde{\mathbf{r}}_a \rVert > 100\epsilon_{\text{machine}}$ it raises `RuntimeError("Internal recentering failed numerically.")`.

> The **arithmetic vertex mean** is used, not the volumetric centroid. For a shape whose vertices are not symmetrically distributed these differ. For any shape with a nontrivial rotation group the vertex mean is a fixed point of the group, so it is the correct rotation centre — this is exactly the case of interest.

### 8.4 Stages 3 — convex decomposition and topology validation

**`_merge_coplanar_hull_triangles(centered_vertices, merge_tolerance)`**

1. `ConvexHull(centered_vertices)` (Qhull) returns a **triangulated** surface: `hull.simplices` (triangles) and `hull.equations` (one row $[n_x, n_y, n_z, b]$ per triangle satisfying $\mathbf{n}\cdot\mathbf{x} + b = 0$, with consistently outward unit normals).
2. Triangles belonging to the same planar face have nearly identical equation rows. So the 4-vectors `hull.equations` are inserted into a `cKDTree` and `query_pairs(merge_tolerance)` finds all pairs within Euclidean distance `merge_tolerance` **in 4-dimensional $(n_x,n_y,n_z,b)$ space**.
3. Those pairs populate a dense symmetric `int8` adjacency matrix of size $n_{\text{tri}} \times n_{\text{tri}}$, and `scipy.sparse.csgraph.connected_components(directed=False)` groups the triangles into faces.
4. For each connected component:
   * the merged face's vertex index set is `np.unique(hull.simplices[component].ravel())`;
   * the face plane equation is the **mean** of the member triangle equations, then rescaled by $1/\lVert\mathbf{n}\rVert$ so the normal is exactly unit and $b$ is consistent (a zero normal raises `RuntimeError`);
   * the face's vertices are **ordered cyclically** by building an in-plane orthonormal basis $(\mathbf{a}, \mathbf{b})$ — $\mathbf{a}$ from the first vertex's displacement from the face centroid, $\mathbf{b} = \mathbf{n}\times\mathbf{a}$ — and sorting by $\operatorname{atan2}(\mathbf{d}\cdot\mathbf{b},\ \mathbf{d}\cdot\mathbf{a})$. Degenerate bases raise `RuntimeError`.
5. **Edges** are recovered from the cyclic ordering: for each face, every consecutive pair `(face[k], face[k+1])` (via `np.roll(face, -1)`) is stored as a sorted tuple `(min, max)` in a `set`, which deduplicates the two faces sharing each edge.
6. Returns a frozen `ConvexDecomposition` carrying vertices, sorted edges, faces, unit face equations, the merge tolerance used, and `hull.volume`.

**`find_validated_convex_decomposition(vertices, expected_edges, expected_faces)`**

Scans the historical tolerance ladder $10^{-12}, 10^{-11}, \ldots, 10^{-4}$ (`for exponent in range(12, 3, -1)`), running the merge at each and printing a table:

```
merge tolerance       recovered edges       recovered faces
 1.0e-12                          36                    24
 1.0e-11                          36                    24
 ...
 1.0e-06                          25                    15
```

It **returns on the first exact match to BOTH counts** and prints `TOPOLOGY CHECK: PASSED`, the accepted tolerance and the hull volume. If no tolerance in the ladder reproduces both counts it raises `RuntimeError` reporting the expected and final-trial counts. This is a hard gate: the program will not proceed with a topology it cannot confirm, which prevents the classic silent failure where over- or under-merged faces produce a wrong candidate-axis set and hence a wrong symmetry group.

Because the ladder runs from tight to loose, the **tightest** tolerance that reproduces your topology is the one selected.

### 8.5 Stage 4a — candidate rotation axes (`build_candidate_axis_lines`)

Rotational symmetry axes of a convex polyhedron always pass through the centre and through one of a small number of geometric features. Four classes are generated, matching the original detector:

| Class | Vector | Count |
|---|---|---|
| 1. Centre → vertex | $\tilde{\mathbf{r}}_a$ | $V$ |
| 2. Centre → face centroid | $\frac{1}{\lvert F\rvert}\sum_{a \in F} \tilde{\mathbf{r}}_a$ | $F$ |
| 3. Centre → perpendicular foot on face plane | $-b\,\mathbf{n}$ | $F$ |
| 4. Centre → edge midpoint | $\frac12(\tilde{\mathbf{r}}_a + \tilde{\mathbf{r}}_b)$ | $E$ |

giving $V + 2F + E$ raw candidates before deduplication. (For the suggested $V{=}12, E{=}25, F{=}15$: 67 raw.)

Class 3 is the **corrected** version of the legacy "centre-to-face-normal" construction: in centred coordinates the plane is $\mathbf{n}\cdot\mathbf{x} + b = 0$ with $\lVert\mathbf{n}\rVert = 1$, so the foot of the perpendicular from the origin is exactly $-b\mathbf{n}$. For a face whose centroid is not the foot of the perpendicular (an irregular polygon), classes 2 and 3 differ and both are needed.

**Deduplication.** Each vector is:
* discarded if $\lVert\mathbf{v}\rVert \le 10^{-14}$ (a feature sitting on the centre defines no axis);
* normalised and sign-canonicalised by `_canonical_axis_line()` — walk the components and flip the whole vector if the first component with $\lvert v_c \rvert > 10^{-14}$ is negative, so that $\mathbf{n}$ and $-\mathbf{n}$ map to one representative;
* discarded if $\lvert \mathbf{n}\cdot\mathbf{n}_{\text{prev}} \rvert \ge 1 - 10^{-12}$ for any already-accepted axis (parallel **or** antiparallel ⇒ same geometric line).

The first label that produced each unique line is kept for traceability (`vertex[3]`, `face_normal[7]`, `edge_midpoint[12]`, ...) and is echoed into the symmetry CSV as the discovery axis. The count of unique lines is printed. An empty set raises `RuntimeError`.

Because axis *lines* are canonicalised to a single sign, **negative rotation angles must be tested** to reach the rotations about the flipped direction — which is exactly why the angle list below contains both signs.

### 8.6 Stage 4b — candidate angles (`historical_candidate_angles_rad`)

A hard-coded list of **37 angles**, preserved verbatim from the legacy code, converted to radians:

*Positive (19):* 180, 120, 240, 90, 270, 72, 144, 216, 288, 60, 300, 45, 135, 225, 315, 252, 324, 36, 108
*Negative (18):* −120, −240, −90, −270, −72, −144, −216, −288, −60, −300, −45, −135, −225, −315, −36, −108, −252, −324

These are the non-identity rotations of $C_n$ for $n \in \{2, 3, 4, 5, 8, 10\}$: multiples of $180°$, $120°$, $90°$, $72°$, $60°$, $45°$, $36°$. Note $180°$ appears once only (because $-180° \equiv +180°$).

Total candidates tested $= n_{\text{unique axes}} \times 37$.

> **This is a finite, fixed search space.** Any symmetry whose order is not in $\{2,3,4,5,8,10\}$ — e.g. a $C_7$ or $C_9$ axis — is **not representable** and the group-closure check (§8.8) will fail rather than silently return a subgroup. To support such shapes you must extend this list.

### 8.7 Stages 5–7 — testing, matching and refining each candidate

The container is `operations_by_permutation: dict[tuple[int,...], SymmetryOperation]`, keyed by the induced **integer vertex permutation**. It is seeded explicitly with the identity (`Rotation.identity()`, permutation $(0,1,\ldots,N-1)$, residual $0$), because angle $0$ is not in the candidate list.

For each (axis $\mathbf{n}$, angle $\theta$) pair:

1. **Construct** $R = \exp(\boldsymbol{\omega}^\times)$ via `Rotation.from_rotvec(axis * angle_rad)`, i.e. rotation vector $\boldsymbol{\omega} = \mathbf{n}\theta$.
2. **Apply** to all centred vertices: $\tilde{\mathbf{r}}' = R\tilde{\mathbf{r}}$.
3. **Match one-to-one** via `_one_to_one_vertex_mapping()`:
   * build the full $N\times N$ cost matrix $C_{ab} = \lVert \tilde{\mathbf{r}}'_a - \tilde{\mathbf{r}}_b \rVert$ using `scipy.spatial.distance_matrix`;
   * solve the **linear sum assignment (Hungarian) problem** with `linear_sum_assignment` — this returns the *globally* minimum-cost perfect matching, guaranteeing each rotated vertex is used once and each reference vertex is hit once. This is exactly the permutation condition for a rigid symmetry;
   * an incomplete assignment raises `RuntimeError`;
   * return `mapping[source] = target` and the per-assignment residuals.

   > Using the Hungarian algorithm rather than a greedy nearest-neighbour lookup is a substantive correctness improvement: greedy matching can assign two rotated vertices to the same target and thereby accept a non-symmetry as a symmetry.
4. **Reject** if $\max_a C_{a,\sigma(a)} > \varepsilon_{\text{match}}$. The **maximum**, not the RMS, is used — the symmetry condition is enforced for every vertex individually, not on average.
5. **Refine at full precision** (`_refine_rotation_for_permutation`). The candidate axes were built from finite-precision shape coordinates, so the raw candidate rotation is only approximately the true symmetry. Once the permutation $\sigma$ is known, reorder the targets as $\tilde{\mathbf{r}}_{\sigma(a)}$ and solve the orthogonal Procrustes / Wahba problem with `Rotation.align_vectors(target, source)`, which returns the proper rotation minimising $\sum_a \lVert \mathbf{t}_a - R\mathbf{s}_a\rVert^2$. A guard raises `RuntimeError` if $\det R < 0$ (should be impossible for SciPy's `Rotation`, retained as an explicit invariant).
6. **Re-verify after refinement.** Apply the refined $R$, recompute the assignment **independently**, and:
   * if the refined permutation $\ne$ the discovery permutation, raise `RuntimeError` — the tolerance does not resolve the vertex correspondence uniquely and the symmetry classification would be ambiguous. The user is told to use a tighter $p$;
   * if the refined maximum residual still exceeds $\varepsilon_{\text{match}}$, **skip** the candidate (a candidate can pass the coarse test and fail the strict refit).
7. **Store**, deduplicating by permutation and keeping the discovery with the **smallest refined maximum residual** when several axis/angle combinations find the same permutation.

Two counters are printed: `Raw axis/angle candidates tested` and `Valid discovery hits before permutation deduplication`, followed by `Distinct physical proper rotations detected` and `Largest accepted full-precision vertex residual`.

**Stage 8** sorts the deduplicated operations into a deterministic order by the key $\bigl(\lVert R - I\rVert_F,\ \sigma\bigr)$ — identity first, then increasing rotation magnitude, with the permutation tuple as an exact tiebreak.

### 8.8 Stage 9 — group validation (`validate_permutation_group`)

**Why in permutation space?** Rotation matrices and quaternions carry floating-point noise, so group axioms tested on them require arbitrary tolerances. Their action on a finite vertex set is captured *exactly* by integer permutations, making the checks discrete and unambiguous.

Three axioms are checked exhaustively:

1. **Identity** — $(0,1,\ldots,N-1)$ must be in the detected set.
2. **Inverses** — for every $\sigma$, construct $\sigma^{-1}$ (`inverse[target] = source`) and require membership.
3. **Closure** — for every ordered pair, `_compose_permutations(first, second)[i] = second[first[i]]` must be in the set. This is an $O(n^2 N)$ double loop over all detected operations.

Any failure raises `RuntimeError` stating that the candidate-angle search is incomplete or inconsistent. Success prints `ROTATIONAL-GROUP CHECK: PASSED (identity, inverses, and closure).`

This is the single most valuable check in the program. A missing symmetry operation would silently inflate every computed $\theta_{ij}$; closure failure catches it.

### 8.9 Stages 10–11 — equivalent quaternion set

**`_rotation_to_wxyz(rotation)`** converts SciPy's `[x, y, z, w]` convention to the freud/HOOMD scalar-first `[w, x, y, z]` via index reorder `[3,0,1,2]`, renormalises, and applies the same first-nonzero-component-positive sign convention, yielding one canonical representative per physical rotation.

Then, because unit quaternions double-cover $SO(3)$, both signs are emitted **interleaved**:

```python
equivalent_quaternions[0::2] =  physical_quaternions   # +q
equivalent_quaternions[1::2] = -physical_quaternions   # -q
```

So the array has shape $(2n, 4)$ for $n$ physical rotations, with row $2k$ and row $2k+1$ being $\pm q_k$.

Two assertions follow:

* every row has unit norm to `atol=1e-12`, else `RuntimeError`;
* `equivalent_quaternions[1::2] == -equivalent_quaternions[0::2]` **exactly** (`atol=0, rtol=0`) — appropriate because the rows were produced by unary negation, so bit-exact equality is expected — else `RuntimeError`.

Prints `Q/-Q CHECK: PASSED for every physical rotational symmetry.`

**Why supply both signs explicitly?** freud's `AngularSeparationGlobal` minimises over exactly the equivalent-orientation list it is given. Supplying both signs makes the computed angle invariant to the arbitrary sign convention of the stored HOOMD quaternions, regardless of whether the underlying implementation takes an absolute value of the scalar part internally. It is a belt-and-braces guarantee that costs a factor of 2 in the inner minimisation and nothing in correctness. The legacy pipeline did the same, and the behaviour is preserved.

### 8.10 Stage 12 — printed summary

A table of one canonical $+q$ per physical rotation with its max residual:

```
Detected physical proper rotations (one canonical +q each)
----------------------------------------------------------
 index             w             x             y             z       max residual
     0   1.0000000000  0.0000000000  0.0000000000  0.0000000000   0.000e+00
     1   0.7071067812  0.7071067812  0.0000000000  0.0000000000   3.142e-16
     ...
```

The $-q$ companions are omitted here but **are** written to the symmetry CSV.

### 8.11 Stage 13 — return

A frozen `SymmetryResult` carrying centred vertices, the original centre, the decomposition, the tuple of physical operations, the $(2n,4)$ equivalent-quaternion array, and the matching tolerance.

---

## 9. Part II — Trajectory processing and histogramming

### 9.1 Per-frame orientation validation (`validate_and_normalize_orientations`)

For each selected frame, in order:

1. `np.asarray(snapshot.particles.orientation, dtype=np.float64)`.
2. Reject `ndim != 2` or `shape[1] != 4` → `ValueError` naming the frame and the found shape.
3. Reject `len(array) < requested_particles` → `ValueError` naming the counts.
4. Slice the **first $M$** rows and force C-contiguity (`np.ascontiguousarray`) — freud expects contiguous input.
5. Reject any non-finite entry → `ValueError`.
6. Reject any quaternion with norm $\le 10^{-14}$ → `ValueError` ("contains a zero quaternion").
7. **Record** $\max_i \bigl\lvert \lVert q_i\rVert - 1 \bigr\rvert$ *before* correction. This diagnostic is printed per frame and stored in the metadata JSON; it tells you how much drift/precision loss the GSD writer introduced.
8. **Normalise** `array /= norms[:, None]`. Small storage deviations are silently corrected — the recorded deviation is your audit trail.

### 9.2 Block-wise pair computation (`compute_one_frame_histogram`)

This is the memory-critical routine. It computes the same result as a full $M \times M$ angle matrix but never materialises one.

**freud call convention.** The code calls

```python
angular_separation.compute(orientations, query_orientations, equivalent_quaternions)
#                          ^global       ^query              ^equivalent
```

and expects `angular_separation.angles` to have shape

$$(\;n_{\text{query}},\ n_{\text{global}}\;) = (\text{block\_stop} - \text{block\_start},\ M).$$

A single `freud.environment.AngularSeparationGlobal()` object is constructed once per frame and reused across blocks.

**The shape is checked explicitly** on every block; a mismatch raises `RuntimeError` quoting the expected and received shapes and stating that the code follows the documented `(N_orientations, N_global_orientations)` ordering. This is a deliberate guard against a silent axis transposition across freud versions — an axis swap would still produce a plausible-looking histogram, so it is checked rather than trusted.

**The block loop:**

```
for block_start in range(0, M, block_size):
    block_stop = min(block_start + block_size, M)
    query = orientations[block_start:block_stop]        # small
    compute(orientations, query, equivalent_quaternions) # (block, M) matrix
    angles_deg = rad2deg(angular_separation.angles)
    for local_row, i in enumerate(range(block_start, block_stop)):
        row = angles_deg[local_row, i+1:]               # upper triangle only
        ...
```

**Upper-triangle extraction.** Row `local_row` corresponds to global particle $i$. Columns $0 \ldots i-1$ would duplicate already-counted pairs $(j,i)$; column $i$ is the self-pair. So the slice `[i+1:]` retains exactly $j > i$. This is a **NumPy view**, not a copy — no large boolean mask is built. The last particle's slice is empty and is skipped.

**Per-row reduction, immediately:**

* `processed_unique_pairs += row.size` — counts *all* pairs regardless of range;
* `global_min_angle`/`global_max_angle` updated from `row.min()`/`row.max()` — these are the **true extrema over all pairs**, including those outside the plotted range, seeded from $\pm\infty$;
* `np.histogram(row, bins=bin_edges_deg)` → `raw_counts += counts`. NumPy silently discards values below the first edge or above the last edge, which is exactly the desired conditional behaviour.

The row's temporary view and the block matrix are then released. **No complete all-frame angle list is ever retained.**

> **Binning edge case:** `np.histogram` makes the last bin **closed on both ends**, so an angle exactly equal to `max_angle` is counted in the final bin rather than discarded. All other bins are half-open $[\text{left}, \text{right})$.

**Post-loop accounting:**

* Assert `processed_unique_pairs == M(M-1)/2`, else `RuntimeError` quoting both numbers. This catches any block-index or triangle-slicing error.
* `pairs_inside_range = raw_counts.sum()`; `pairs_outside_range = M(M-1)/2 - pairs_inside_range`.
* If `pairs_inside_range <= 0` → `ValueError` naming the frame and the exact interval (conditional normalisation would be $0/0$).
* $P_f(k) = h_f(k) / N_f^{\text{inside}}$.
* Assert $\sum_k P_f(k) = 1$ to `atol=1e-12`, else `RuntimeError`.

Returns a frozen `FrameHistogram`.

**Choosing `block_size`.** It affects **only peak memory**, never the numerical result. Peak angle-matrix memory is $8 \times \text{block\_size} \times M$ bytes (float64). With $M = 4096$ and `block_size = 128` that is 4 MB per block. Larger blocks mean fewer freud calls (slightly faster) and more memory; `block_size >= M` degenerates to the full matrix.

---

## 10. Part III — Aggregation across frames

`compute_frame_averaged_histogram()`, in seven stages:

1. **Common grid.** `bin_edges = np.linspace(0.0, max_angle_deg, num_bins + 1)` — $B$ equal-width bins of width $\theta_{\max}/B$, identical for every frame. Note the range always starts at exactly $0$.
2. **Per-frame loop.** Load snapshot → validate/normalise → `compute_one_frame_histogram()` → append result → print the per-frame audit line (see §12).
3. **Stack** the per-frame probability vectors into an $(F, B)$ matrix.
4. **Equal-weight mean** `np.mean(matrix, axis=0)` and **population standard deviation** `np.std(matrix, axis=0, ddof=0)`. `ddof=0` is used deliberately: the selected frames are treated as the complete set being summarised, not as a random sample requiring Bessel's correction.
5. **Pooled diagnostic.** Sum the raw integer counts across frames with `dtype=np.int64` (exact integer arithmetic, no float accumulation), then normalise once.
6. **Validate** $\sum_k \overline{P}(k) = 1$ to `atol=1e-12`. Since each row sums to 1, the mean of the rows must too; failure indicates a real bug.
7. **x-coordinate** `x_left_edges = bin_edges[:-1]`.

Returns `(x_left_edges, mean_probability, standard_deviation, pooled_probability, frame_results)`.

> **On the standard deviation:** with $F = 1$ it is identically zero and the plot draws no band. With $F > 1$ it measures **frame-to-frame reproducibility**, not a statistical error on the mean. Consecutive MD/MC frames are strongly correlated, so this band is generally *not* a valid error bar — treat it as a stationarity/convergence diagnostic. For a genuine uncertainty you would need decorrelated frames and would divide by $\sqrt{F_{\text{eff}}}$.

---

## 11. Outputs: complete file and column reference

### 11.1 Filename prefix

```
<output_dir>/<gsd_stem>_global_pairwise_angles_last_<F>_frames_particles_<M>_bins_<B>
```

`<output_dir>` defaults to the trajectory's parent directory, or `--output-dir` if given. The directory is created with `parents=True, exist_ok=True`.

### 11.2 The five output files

| File | Construction | Contents |
|---|---|---|
| `<prefix>.csv` | `prefix.with_suffix(".csv")` | Summary histogram |
| `<prefix>_per_frame.csv` | name + suffix | Per-frame probabilities |
| `<prefix>_symmetry_quaternions.csv` | name + suffix | All $\pm q$ symmetry operations |
| `<prefix>_metadata.json` | name + suffix | Complete run record |
| `<prefix>.png` | `prefix.with_suffix(".png")` | The plot |

### 11.3 Summary CSV — `<prefix>.csv`

Header (`comments=""`, so the header line is bare, not `#`-prefixed):

```
left_bin_edge_deg,right_bin_edge_deg,equal_weight_frame_mean_probability_per_bin,frame_to_frame_standard_deviation,pooled_conditional_probability_per_bin,total_raw_count
```

| Col | Name | Format | Definition |
|---|---|---|---|
| 1 | `left_bin_edge_deg` | `%.12g` | $\text{bin\_edges}[k]$ — **the plotted x value** |
| 2 | `right_bin_edge_deg` | `%.12g` | $\text{bin\_edges}[k+1]$ |
| 3 | `equal_weight_frame_mean_probability_per_bin` | `%.17g` | $\overline{P}(k) = \frac1F\sum_f P_f(k)$ — **the primary result** |
| 4 | `frame_to_frame_standard_deviation` | `%.17g` | $\sigma_k$, population (ddof=0) |
| 5 | `pooled_conditional_probability_per_bin` | `%.17g` | $P_{\text{pool}}(k)$, diagnostic |
| 6 | `total_raw_count` | `%d` | $\sum_f h_f(k)$, exact integer |

`%.17g` guarantees lossless round-trip of IEEE-754 doubles.

### 11.4 Per-frame CSV — `<prefix>_per_frame.csv`

$B$ rows, $1 + F$ columns, all `%.17g`:

```
left_bin_edge_deg,frame_<idx0>_probability,frame_<idx1>_probability,...
```

Column $f+1$ is the full conditional distribution $P_f(\cdot)$ of frame index `<idxf>`. Use this to check for drift, to compute your own weighted averages, or to block-average for a proper error estimate.

### 11.5 Symmetry quaternion CSV — `<prefix>_symmetry_quaternions.csv`

Two rows per physical rotation ($+q$ then $-q$), so $2n$ rows total.

```
physical_operation_index,quaternion_sign,w,x,y,z,max_vertex_residual,discovery_angle_deg,discovery_axis_x,discovery_axis_y,discovery_axis_z
```

| Col | Name | Format | Meaning |
|---|---|---|---|
| 1 | `physical_operation_index` | `%d` | $0 \ldots n-1$, matching the printed table order |
| 2 | `quaternion_sign` | `%d` | `+1` or `-1` |
| 3–6 | `w,x,y,z` | `%.17g` | The signed scalar-first quaternion, full precision, **unrounded** |
| 7 | `max_vertex_residual` | `%.17g` | Worst post-refinement vertex mismatch (shape units) |
| 8 | `discovery_angle_deg` | `%.17g` | The candidate angle that first found this permutation (metadata only) |
| 9–11 | `discovery_axis_{x,y,z}` | `%.17g` | The canonicalised candidate axis that found it (metadata only) |

> Columns 8–11 describe the **discovery**, not the final operation. The authoritative numerical definition is columns 3–6, which come from the full-precision Procrustes refit, not from the discovery axis/angle.

This file is directly reusable: you can feed columns 3–6 into another freud calculation as an equivalent-orientation array without re-running the detection.

### 11.6 Metadata JSON — `<prefix>_metadata.json`

Indented (2 spaces), UTF-8, trailing newline. Top-level keys:

**Inputs and selection**
`gsd_file`, `shape_file`, `total_trajectory_frames`, `selected_frame_indices`, `selected_particle_count_per_frame`, `total_unique_pairs_per_frame`

**Conventions (self-documenting strings)**
`pair_definition` = `"unique unordered pairs i < j; self-pairs excluded"`
`x_coordinate_convention` = `"left bin edges"`
`normalisation` = `"each frame normalised by the number of unique pairs inside the plotted range; equal-weight arithmetic mean over selected frames"`
`quantity` = `"probability per bin (not probability density)"`

**Histogram settings**
`histogram_bins`, `maximum_plotted_angle_deg`, `freud_block_size`, `mean_probability_sum`

**Shape and symmetry**
`symmetry_precision_exponent`, `symmetry_matching_tolerance`, `expected_edges`, `expected_faces`, `recovered_edges`, `recovered_faces`, `convex_hull_merge_tolerance`, `convex_hull_volume`, `original_vertex_center`, `physical_proper_rotation_count`, `equivalent_quaternion_representative_count`

**Per-frame array** `frames[]`, each entry containing
`frame_index`, `total_unique_pairs`, `pairs_inside_range`, `pairs_outside_range`, `outside_range_fraction`, `minimum_angle_deg`, `maximum_angle_deg`, `maximum_input_quaternion_norm_deviation`, `conditional_probability_sum`

This file alone is sufficient to reconstruct the exact command line and to audit every scientific choice after the fact.

### 11.7 The plot — `<prefix>.png`

| Property | Value |
|---|---|
| Figure size | 4.8 × 3.6 inches |
| Figure DPI | 200 |
| Saved DPI | **600** (`savefig(dpi=600, bbox_inches="tight")`) → ≈ 2880 × 2160 px |
| Main curve | `x_left_edges` vs `mean_probability`, `linewidth=1.5` |
| Band | If $F > 1$: `fill_between(mean ± σ)`, `alpha=0.2`, clipped at 0 from below, labelled "Frame-to-frame standard deviation" |
| x-label | $\theta_{ij}(^{\circ})$ |
| y-label | $P(\theta_{ij})$ |
| x-limits | $[0, \text{max\_angle}]$ |
| Ticks | `direction="in"` on both axes |
| Legend | `frameon=False` |
| Layout | `tight_layout()` |
| Display | `plt.show()` unless `--no-show`, in which case `plt.close(figure)` |

The curve is drawn as a **line through the left bin edges**, not as bars — a deliberate stylistic choice matching the legacy plots.

---

## 12. Console output reference

The program prints a structured, auditable log. Sections in order:

```
Trajectory summary
------------------
Trajectory: /abs/path/traj.gsd
Number of available frames: 500
Valid zero-based frame indices: 0 through 499
Selected final frame indices: 490 through 499

Particle-count summary for selected frames
------------------------------------------
Every selected frame contains 4096 particle orientations.
Particles selected in every frame: indices 0 through 4095

Histogram definition
--------------------
Bins: 50
Plotted range: 0 to 120 degrees
Pair set: unique unordered pairs i < j; self-pairs excluded
x coordinate: left bin edge
y coordinate: conditional probability per bin, not density
Frame aggregation: normalise every selected frame inside the plotted range, then take an equal-weight arithmetic mean
freud query block size: 128

Shape vertex array selected from JSON key/path hint: 'vertices'
Number of shape vertices: 12

Shape-centre validation
-----------------------
Original arithmetic vertex centre: [...]
Norm of original centre: ...
User-selected symmetry matching tolerance: 10^(-2) = 0.01 shape-length units
CENTRE CHECK: PASSED within the selected tolerance.
Internal centred-coordinate residual: [...]

Convex-hull topology scan
--------------------------
User-supplied expected topology: edges=25, faces=15, vertices=12
merge tolerance       recovered edges       recovered faces
 1.0e-12  ...
TOPOLOGY CHECK: PASSED.  The recovered edge and face counts match the user inputs.
Accepted coplanar-face merge tolerance: 1.0e-06
Convex-hull volume from the shape file: ...

Unique candidate axis lines generated: 31

Proper rotational-symmetry search
---------------------------------
Candidate angle count: 37
Accepted candidates must map all vertices one-to-one with maximum Euclidean residual <= 0.01
Raw axis/angle candidates tested: 1147
Valid discovery hits before permutation deduplication: 46
Distinct physical proper rotations detected: 12
Largest accepted full-precision vertex residual: ...
ROTATIONAL-GROUP CHECK: PASSED (identity, inverses, and closure).
Equivalent quaternion representatives passed to freud: 24 (= 2 x physical rotations)
Q/-Q CHECK: PASSED for every physical rotational symmetry.

Detected physical proper rotations (one canonical +q each)
----------------------------------------------------------
 index             w             x             y             z       max residual
   ...

Trajectory histogram calculation
--------------------------------
[1/10] frame 490: unique pairs=8,386,560; inside range=8,386,560; outside range=0 (0.000000%); angle min/max=0.0143/61.87 deg; max |norm(q)-1|=1.192e-07
...

Final validation and outputs
----------------------------
Sum of final mean bin probabilities: 1
Summary histogram CSV: ...
Per-frame probability CSV: ...
Symmetry quaternion CSV: ...
Run metadata JSON: ...
Plot PNG: ...
```

**Per-frame audit line fields:** frame index; total unique pairs ($M(M-1)/2$, thousands-separated); pairs inside the plotted range; pairs outside with percentage to 6 decimal places; the true min/max angle over *all* pairs (`%.6g`); and the max input quaternion norm deviation (`%.3e`).

---

## 13. Internal data structures

All four are `@dataclass(frozen=True)` — immutable, so no downstream code can mutate a validated result.

### `ConvexDecomposition`
| Field | Type | Meaning |
|---|---|---|
| `vertices` | `ndarray (N,3)` | **Centred** vertices |
| `edges` | `tuple[tuple[int,int], ...]` | Sorted, deduplicated `(min, max)` index pairs |
| `faces` | `tuple[ndarray, ...]` | Per face, cyclically ordered vertex indices |
| `face_equations` | `ndarray (F,4)` | $[n_x,n_y,n_z,b]$ with $\lVert\mathbf{n}\rVert = 1$, in centred coordinates |
| `merge_tolerance` | `float` | The accepted coplanarity tolerance |
| `volume` | `float` | `ConvexHull.volume` in shape units³ |

### `SymmetryOperation`
| Field | Type | Meaning |
|---|---|---|
| `quaternion_wxyz` | `ndarray (4,)` | Canonical $+q$, scalar-first, full precision |
| `rotation_matrix` | `ndarray (3,3)` | Full double-precision $R$ |
| `permutation` | `tuple[int, ...]` | Exact discrete action on vertex indices — the dedup key |
| `max_vertex_residual` | `float` | Worst post-refit Euclidean mismatch |
| `discovery_axis` | `ndarray (3,)` | Diagnostic metadata |
| `discovery_angle_deg` | `float` | Diagnostic metadata |

### `SymmetryResult`
`centered_vertices`, `original_center`, `decomposition`, `physical_operations` (tuple), `equivalent_quaternions_wxyz` ($(2n,4)$, the array actually passed to freud), `matching_tolerance`.

### `FrameHistogram`
`frame_index`, `raw_counts` (`int64`, length $B$), `conditional_probability` (`float64`, length $B$), `total_unique_pairs`, `pairs_inside_range`, `pairs_outside_range`, `minimum_angle_deg`, `maximum_angle_deg`, `maximum_quaternion_norm_deviation`.

---

## 14. Complete validation and error catalogue

Every runtime check, in the order it can fire:

### Dependencies
| Condition | Exception | Message |
|---|---|---|
| SciPy missing | `SystemExit` (at import) | Install hint |
| freud / gsd / matplotlib missing | `ImportError` | Install hint |

### CLI and trajectory
| Condition | Exception |
|---|---|
| `--block-size < 1` | `ValueError` |
| GSD file absent | `FileNotFoundError` |
| `len(trajectory) < 1` | `ValueError` |
| Any selected frame's orientation array not $(N,4)$ | `ValueError` naming the frame and shape |
| A CLI-supplied value fails its validator | `ValueError` with the same message the prompt would print |

### Shape JSON
| Condition | Exception |
|---|---|
| File absent | `FileNotFoundError` |
| Malformed JSON | `json.JSONDecodeError` |
| No finite $N\times3$, $N\ge4$ array found | `ValueError` |
| All vertices at one point | `ValueError` |
| Duplicate/indistinguishable vertices | `ValueError` naming an example index pair |

### Geometry
| Condition | Exception |
|---|---|
| Merged face has zero normal | `RuntimeError` |
| Cannot build in-plane face basis | `RuntimeError` |
| Degenerate polygon face | `RuntimeError` |
| **No merge tolerance reproduces the expected topology** | `RuntimeError` reporting expected vs. final-trial counts |
| Zero axis passed to canonicalisation | `ValueError` |
| No nonzero candidate axes | `RuntimeError` |

### Symmetry
| Condition | Exception |
|---|---|
| `precision_exponent < 0` | `ValueError` |
| Internal recentring residual $> 100\epsilon$ | `RuntimeError` |
| Incomplete Hungarian assignment | `RuntimeError` |
| Refined rotation has $\det < 0$ | `RuntimeError` |
| **Refinement changed the discovered permutation** | `RuntimeError` advising a tighter $p$ |
| No permutations detected | `RuntimeError` |
| Identity absent | `RuntimeError` |
| **An inverse is missing** | `RuntimeError` — search incomplete |
| **Not closed under composition** | `RuntimeError` — search incomplete |
| Equivalent quaternions not unit norm (`atol=1e-12`) | `RuntimeError` |
| $q/-q$ pairing not exact | `RuntimeError` |

### Trajectory data
| Condition | Exception |
|---|---|
| Orientation array not $(N,4)$ | `ValueError` |
| Fewer orientations than requested particles | `ValueError` |
| NaN/inf in orientations | `ValueError` |
| A zero quaternion | `ValueError` |

### Histogramming
| Condition | Exception |
|---|---|
| **freud output shape $\ne (n_{\text{query}}, M)$** | `RuntimeError` quoting both shapes |
| **Processed pairs $\ne M(M-1)/2$** | `RuntimeError` quoting both counts |
| No pairs inside the plotted range | `ValueError` naming the frame and interval |
| Per-frame $\sum_k P_f(k) \ne 1$ (`atol=1e-12`) | `RuntimeError` |
| Frame-averaged $\sum_k \overline{P}(k) \ne 1$ | `RuntimeError` |

---

## 15. Computational complexity and memory model

Let $V$ = shape vertices, $E$ = edges, $F_{\text{poly}}$ = faces, $n_{\text{tri}}$ = hull triangles, $A$ = unique candidate axes, $n$ = detected rotations, $M$ = particles per frame, $F$ = frames, $B$ = bins.

### Symmetry detection (once)

| Stage | Time | Memory |
|---|---|---|
| Convex hull | $O(V\log V)$ | $O(n_{\text{tri}})$ |
| Coplanar merge, per tolerance | $O(n_{\text{tri}}\log n_{\text{tri}} + n_{\text{tri}}^2)$ | $O(n_{\text{tri}}^2)$ **dense int8 adjacency** |
| Tolerance ladder | $\times 9$ | — |
| Candidate axes | $O((V + 2F_{\text{poly}} + E)^2)$ (pairwise dedup) | $O(A)$ |
| Candidate testing | $A \times 37 \times \bigl[O(V) + O(V^2) + O(V^3)\bigr]$ | $O(V^2)$ per candidate |
| Group validation | $O(n^2 V)$ | $O(nV)$ |

The dominant term is the Hungarian solve, $O(V^3)$, run $A \times 37$ times (plus once more per accepted candidate for the refit re-check). For typical polyhedra ($V \lesssim 100$, $A \lesssim 100$) this is seconds. The $O(n_{\text{tri}}^2)$ dense adjacency matrix is the only structure that would become problematic for a shape with thousands of hull triangles.

### Histogramming (per frame)

| Quantity | Cost |
|---|---|
| freud angular separations | $O(M^2 \cdot 2n)$ — every pair minimised over $2n$ equivalent quaternions |
| Upper-triangle extraction + histogram | $O(M^2)$ |
| Peak angle memory | $8 \cdot \text{block\_size} \cdot M$ bytes |
| Persistent per-frame memory | $O(B)$ — only the count and probability vectors |

**Total time** $\approx F \cdot O(M^2 n)$. This is the bottleneck. Doubling $M$ quadruples the runtime. A highly symmetric particle costs more per pair (larger $n$) but yields a narrower angle range.

**Total persistent memory** $O(FB + V^2 + n_{\text{tri}}^2)$ — independent of $M^2$. This is the central improvement over the legacy code, which built an $M\times M$ matrix and then copied it repeatedly into Python lists.

---

## 16. Reproducibility and determinism

The calculation is **fully deterministic**. There is no random sampling anywhere: frames are the last $F$, particles are the first $M$, the JSON traversal order is sorted, the candidate axis list is built in a fixed order, and the operation list is sorted by an explicit total key.

Re-running the same command on the same files reproduces every output bit-for-bit, modulo:

* BLAS/LAPACK threading nondeterminism inside `linear_sum_assignment` / `align_vectors` at the $10^{-16}$ level (which cannot change a permutation or a bin assignment except in astronomically unlikely tie cases);
* freud version differences in the angular-separation kernel.

For a fully reproducible batch run, supply all seven scientific flags:

```bash
python hist_pairwise_angles_rigorous_standalone_v1p0.py traj.gsd shape.json \
  --frames 100 --particles 4096 --edges 25 --faces 15 \
  --precision 4 --bins 50 --max-angle 120 \
  --block-size 256 --output-dir ./analysis --no-show
```

The metadata JSON records every one of these values, so any output can be traced back to its command line.

---

## 17. Differences from the legacy pipeline

| Aspect | Legacy | This program |
|---|---|---|
| Structure | Multiple project modules | One standalone file |
| `tolerance_for_inv_quat = 2` | Decimal **rounding** of accepted quaternions to 2 dp | Absolute geometric **tolerance** $10^{-2}$; quaternions kept at full precision |
| Vertex matching | Nearest-neighbour style | Globally optimal **Hungarian** one-to-one assignment |
| Rotation accuracy | Raw candidate axis/angle retained | Candidate used only for **discovery**; final rotation from a full-precision Procrustes refit over all vertices, then re-verified |
| Acceptance criterion | — | **Maximum** vertex residual (not RMS), enforced twice |
| Face-normal axis | Centre-to-face-normal construction | Corrected to the perpendicular foot $-b\mathbf{n}$ |
| Topology check | Best-effort | **Hard gate** — no match, no run |
| Group axioms | Not verified | Identity, inverses and closure verified exactly in permutation space |
| Deduplication | By quaternion value | By exact integer **permutation** (immune to float noise) |
| Pair set | $N\times N$ matrix | Explicit $i<j$, count-asserted |
| Memory | Full $N\times N$ matrix copied into Python lists | Block-wise, reduced to counts immediately |
| freud output shape | Assumed | Explicitly checked every block |
| Provenance | — | Symmetry CSV, per-frame CSV and metadata JSON |

Deliberately **preserved** from the legacy code (so results remain comparable): the candidate-angle list, the tolerance ladder $10^{-12} \ldots 10^{-4}$, the four candidate-axis classes, the use of `AngularSeparationGlobal` with explicit $\pm q$, the first-$M$-particles rule, the last-$F$-frames rule, left-bin-edge x values, probability-per-bin y values, and conditional in-range normalisation followed by equal-weight frame averaging.

---

## 18. Known limitations, caveats and gotchas

### 18.1 The tolerance $10^{-p}$ is absolute, not relative
If your shape file uses vertices of magnitude $\sim 0.1$, then $p = 2$ ($\varepsilon = 0.01$) is a **10% tolerance** and will accept near-symmetries that are not symmetries. If magnitudes are $\sim 100$, $p=2$ is a $10^{-4}$ relative tolerance and may reject genuine symmetries. **Always check $\varepsilon$ against your characteristic vertex radius**, which the program effectively reveals via the printed centre norm and hull volume. The built-in caution at $p \le 2$ exists for this reason. Run at $p=3$ and $p=4$ and confirm the detected rotation count $n$ is unchanged.

### 18.2 The candidate-angle list is finite
Only $C_2, C_3, C_4, C_5, C_8, C_{10}$ rotations are reachable. A shape with a $C_6$ axis is partially covered (60°, 120°, 180°, 240°, 300° are all present, so $C_6$ *is* in fact reachable), but $C_7$, $C_9$, $C_{11}$, $C_{12}$ are not ($C_{12}$ needs 30°, absent). If your shape has such an axis, the group-closure check will fail — correctly refusing to proceed rather than silently returning a subgroup. Extend `historical_candidate_angles_rad()` if needed.

### 18.3 Only convex shapes
The whole geometric pipeline is built on `ConvexHull`. A non-convex particle's concave features are invisible; the detected group would be that of its convex hull, which can be strictly larger than the true group. This program is for convex polyhedra only.

### 18.4 No positional information
There is no neighbour cutoff, no box, no periodic-image handling. Every pair contributes regardless of separation. This is a **global** orientational correlation, not a spatially resolved one. If you want $P(\theta \mid r < r_c)$ you need a different tool (`freud.locality` + `AngularSeparationNeighbor`).

### 18.5 Particle selection is by index, not random
Using the first $M$ particles is deterministic and reproducible, but if your GSD file has index-correlated structure — e.g. particles sorted by type, by initial lattice position, or by a spatial sort applied at write time — then $M <$ total silently analyses a **spatially or compositionally biased subset**. If you must subsample, prefer $M = N$ or verify that index order is uncorrelated with structure.

### 18.6 The standard deviation band is not an error bar
See §10. Consecutive frames are correlated.

### 18.7 Conditional normalisation can hide mass
If `outside_range_fraction` is large, the plotted curve is renormalised over a subset. Always read that field in the metadata JSON. Setting `--max-angle 180` guarantees zero exclusion.

### 18.8 Plot legend with $F = 1$
The label on the main curve is commented out in the source, so with a single frame there are no labelled artists and `axis.legend(frameon=False)` emits a matplotlib `UserWarning` ("No artists with labels found...") and draws nothing. Harmless. With $F > 1$ the legend shows only the standard-deviation band.

### 18.9 `with_suffix` and dotted filenames
`output_prefix.with_suffix(".csv")` **replaces** everything after the last dot in the prefix. The prefix normally contains no dot, so `.csv` is appended as intended. But if `gsd_path.stem` itself contains a dot (e.g. `run.v2.gsd` → stem `run.v2`), the summary CSV and PNG names will be mangled (`..._bins_50` built from `run.v2` truncates at `.v2`). The `_per_frame.csv`, `_symmetry_quaternions.csv` and `_metadata.json` writers use `with_name(name + suffix)` and are unaffected. Avoid dots in GSD filenames.

### 18.10 The trajectory handle is never explicitly closed
It is released when the process exits. Not a problem for a script; would be for library reuse.

### 18.11 `plt.show()` blocks by default
On a headless cluster node without `--no-show` and without a suitable `MPLBACKEND`, the run may hang or error at the very end — *after* all files have been written. Always pass `--no-show` in batch jobs.

### 18.12 Figure DPI mismatch
The figure is created at `dpi=200` but saved at `dpi=600`. Combined with `bbox_inches="tight"`, the saved PNG is ~2880×2160 px while font sizes were laid out for the 200-dpi figure. The result is a high-resolution image with proportionally correct but small-looking text. Adjust `figsize`/`rcParams` if you need publication typography.

### 18.13 freud API assumption
The argument order and output shape of `AngularSeparationGlobal.compute` are checked but assumed to follow the documented `(N_orientations, N_global_orientations)` convention. A future freud release that changes this will trigger the explicit `RuntimeError` rather than a wrong answer — which is the intended behaviour, but you will need to update the call.

### 18.14 Very small `num_bins`
`num_bins >= 1` is allowed. `num_bins = 1` gives a single bin containing all in-range pairs with probability 1.0 — technically valid, scientifically useless.

---

## 19. Worked example

Suppose `shape.json` describes a shape with 12 vertices, 25 edges and 15 faces, and `traj.gsd` has 500 frames of 4096 particles.

### Interactive run

```bash
$ python hist_pairwise_angles_rigorous_standalone_v1p0.py traj.gsd shape.json
```

```
Trajectory summary
------------------
Number of available frames: 500
Valid zero-based frame indices: 0 through 499
How many consecutive final frames should be included in the equal-weight average? [1]: 10
Selected final frame indices: 490 through 499
...
How many particles should be used from the beginning of each selected frame? [4096]: <Enter>
Enter the known number of polyhedron edges [25]: <Enter>
Enter the known number of polyhedron faces [15]: <Enter>
Enter invariant-quaternion symmetry precision p; ... [2]: 4
Enter the number of histogram bins ... [50]: <Enter>
Enter the maximum plotted misorientation angle in degrees; ... [120.0]: <Enter>
```

Per frame: $M = 4096$ gives $4096 \times 4095 / 2 = 8{,}386{,}560$ unique pairs. With `block_size = 128`, freud is called $\lceil 4096/128 \rceil = 32$ times per frame, each returning a $128 \times 4096$ matrix (4 MB). The bin width is $120/50 = 2.4°$.

### Equivalent batch run

```bash
python hist_pairwise_angles_rigorous_standalone_v1p0.py traj.gsd shape.json \
  --frames 10 --particles 4096 --edges 25 --faces 15 \
  --precision 4 --bins 50 --max-angle 120 --no-show
```

### Outputs

```
traj_global_pairwise_angles_last_10_frames_particles_4096_bins_50.csv
traj_global_pairwise_angles_last_10_frames_particles_4096_bins_50_per_frame.csv
traj_global_pairwise_angles_last_10_frames_particles_4096_bins_50_symmetry_quaternions.csv
traj_global_pairwise_angles_last_10_frames_particles_4096_bins_50_metadata.json
traj_global_pairwise_angles_last_10_frames_particles_4096_bins_50.png
```

### Reading the summary CSV

```python
import numpy as np
d = np.genfromtxt("..._bins_50.csv", delimiter=",", names=True)
theta = d["left_bin_edge_deg"]                                   # x
P     = d["equal_weight_frame_mean_probability_per_bin"]         # y
sigma = d["frame_to_frame_standard_deviation"]
assert abs(P.sum() - 1.0) < 1e-12
```

---

## 20. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `TOPOLOGY CHECK: FAILED` | Wrong edge/face counts entered, or vertices don't describe the shape you think | Read the printed scan table — it shows what the hull actually recovers at each tolerance. Enter those numbers, or fix the shape JSON. Cross-check with Euler's formula $V - E + F = 2$. |
| `Detected symmetry operations are not closed under composition` | Candidate-angle list too small for this shape, or $\varepsilon$ too tight so some operations were rejected | Try a looser $p$ first; if the count is still short, extend `historical_candidate_angles_rad()` |
| `Detected symmetry set is missing an inverse operation` | Same as above | Same as above |
| `Full-precision rotation refinement changed the discovered vertex permutation` | $\varepsilon$ too **loose** — vertices are closer together than the tolerance | Increase $p$ (tighter tolerance) |
| Only 1 rotation detected (identity) | $\varepsilon$ far too tight, or the shape genuinely has no symmetry, or coordinates are noisy | Loosen $p$ and watch the detected count as a function of $p$; a plateau is the true group |
| `The shape JSON contains duplicate ... vertices` | Redundant vertices in the file | Deduplicate before running |
| `Could not locate a finite N x 3 vertex array` | Unusual JSON layout, or fewer than 4 vertices | Rename the field to `vertices`, or flatten the structure |
| `Unexpected freud AngularSeparationGlobal output shape` | freud version with different argument ordering | Check your freud version against the documented API and adjust the `compute()` call |
| `no unique pair angles fall inside the plotted range` | `max_angle` far too small | Increase `--max-angle`, up to 180 |
| Runs out of memory | `block_size` × $M$ too large | Reduce `--block-size` |
| Very slow | $O(M^2 n)$ | Reduce `--particles`, reduce `--frames`, or accept the cost |
| Hangs at the very end on a cluster | `plt.show()` with no display | Pass `--no-show` |
| `ERROR: <something>` and exit 1 | A controlled validation failure | Read the message; every one names the specific quantity that failed |

---

## 21. Function-by-function index

| Function | Lines | Role |
|---|---|---|
| `prompt_value` | 190 | Loop until valid terminal input; empty line = default |
| `resolve_or_prompt` | 216 | CLI value if supplied (validated, hard-fail), else prompt |
| `_as_vertex_array` | 237 | Test whether an object is a finite $(N\ge4, 3)$ float array |
| `_collect_vertex_candidates` | 255 | Recursive, deterministic JSON search with key-name priority |
| `read_shape_vertices` | 291 | Load, select, validate, deduplicate-check the vertex array |
| `_merge_coplanar_hull_triangles` | 336 | Qhull → coplanar merge → polygonal faces, unit plane equations, cyclic ordering, edge set |
| `find_validated_convex_decomposition` | 426 | Tolerance ladder $10^{-12}\ldots10^{-4}$; hard topology gate |
| `_canonical_axis_line` | 497 | Unit-normalise and fix the $\pm$ sign of an axis line |
| `build_candidate_axis_lines` | 517 | Four axis classes; zero-removal; parallel/antiparallel dedup |
| `historical_candidate_angles_rad` | 607 | The fixed 37-angle list, in radians |
| `_rotation_to_wxyz` | 628 | SciPy `[x,y,z,w]` → freud `[w,x,y,z]`, normalised, sign-canonical |
| `_one_to_one_vertex_mapping` | 645 | Distance matrix + Hungarian assignment + residuals |
| `_refine_rotation_for_permutation` | 690 | Least-squares proper rotation via `align_vectors`, with $\det>0$ guard |
| `_compose_permutations` | 728 | $(\sigma_2 \circ \sigma_1)[i] = \sigma_2[\sigma_1[i]]$ |
| `validate_permutation_group` | 746 | Exact identity / inverse / closure checks |
| `detect_proper_rotational_symmetries` | 810 | The 13-stage symmetry pipeline; returns `SymmetryResult` |
| `import_runtime_packages` | 1134 | Lazy freud / gsd.hoomd / matplotlib import |
| `open_gsd_trajectory` | 1150 | Path resolution + keyword/positional `open` fallback |
| `validate_and_normalize_orientations` | 1166 | Shape/finiteness/norm checks, first-$M$ slice, renormalisation, deviation diagnostic |
| `compute_one_frame_histogram` | 1227 | Block-wise freud calls, $i<j$ extraction, counting, conditional normalisation, pair-count assertion |
| `compute_frame_averaged_histogram` | 1435 | Frame loop, stacking, equal-weight mean, σ, pooled diagnostic, final validation |
| `save_symmetry_outputs` | 1646 | $\pm q$ symmetry CSV |
| `save_histogram_outputs` | 1689 | Summary CSV, per-frame CSV, PNG |
| `save_metadata` | 1790 | Complete run-record JSON |
| `build_argument_parser` | 1878 | The CLI definition |
| `main` | 1924 | The 20-step workflow inside one error boundary |

---

## Citation and provenance note

If you publish results from this program, record in your methods section:

* the **detected proper rotation group order** $n$ (metadata: `physical_proper_rotation_count`);
* the **matching tolerance** $10^{-p}$ and confirmation that $n$ is stable under variation of $p$;
* the **normalisation convention** (conditional in-range, equal-weight frame average, probability per bin) and the **`outside_range_fraction`**;
* the **pair definition** ($i<j$, global, no cutoff) and $M$, $F$, $B$, $\theta_{\max}$.

The metadata JSON contains all of these verbatim.
