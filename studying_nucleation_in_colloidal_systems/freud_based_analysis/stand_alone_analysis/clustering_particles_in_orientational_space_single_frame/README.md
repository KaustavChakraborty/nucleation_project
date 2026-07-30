# Single-Frame Orientational-State Clustering

**Script:** `clustering_particles_in_ori_space_single_frame.py`
**Version:** 1.1 (single-frame)
**Language:** Python 3.9+ (`from __future__ import annotations` makes PEP 604/585 annotations safe on 3.9)
**Size:** 2438 lines, single self-contained file
**Entry point:** `main()` via `raise SystemExit(main())`

---

## Table of Contents

**Orientation**
1. [What this program does, in one page](#1-what-this-program-does-in-one-page)
2. [Output file map at a glance](#2-output-file-map-at-a-glance)

**Understanding the problem**
3. [The physical question](#3-the-physical-question)
4. [Why this is hard](#4-why-this-is-hard)
5. [The solution strategy in five ideas](#5-the-solution-strategy-in-five-ideas)
6. [Mathematical definitions](#6-mathematical-definitions)

**Using the program**
7. [Installation](#7-installation)
8. [Quick start](#8-quick-start)
9. [Inputs](#9-inputs)
10. [Complete CLI reference](#10-complete-cli-reference)
11. [Interactive prompts, in order](#11-interactive-prompts-in-order)
12. [**Tuning guide: applying this to a new system**](#12-tuning-guide-applying-this-to-a-new-system)

**How it executes**
13. [Top-level execution map](#13-top-level-execution-map)
14. [Part A — Shape processing and symmetry group](#14-part-a--shape-processing-and-symmetry-group)
15. [Part B — Distances, clustering and medoids](#15-part-b--distances-clustering-and-medoids)
16. [Part C — State identification](#16-part-c--state-identification)

**Outputs — the detailed reference**
17. [**Output files: what each one saves and why it exists**](#17-output-files-what-each-one-saves-and-why-it-exists)
18. [Cross-file invariants you can check](#18-cross-file-invariants-you-can-check)
19. [Console output reference](#19-console-output-reference)

**Reference material**
20. [Internal data structures](#20-internal-data-structures)
21. [Complete validation and error catalogue](#21-complete-validation-and-error-catalogue)
22. [Complexity, memory and disk model](#22-complexity-memory-and-disk-model)
23. [Determinism and reproducibility](#23-determinism-and-reproducibility)
24. [Relationship to the multi-frame version](#24-relationship-to-the-multi-frame-version)
25. [Limitations, caveats and gotchas](#25-limitations-caveats-and-gotchas)
26. [Troubleshooting](#26-troubleshooting)
27. [Function-by-function index](#27-function-by-function-index)

---

## 1. What this program does, in one page

Given **one frame** of a HOOMD GSD trajectory and a **shape JSON** describing the particle, this program answers:

> *How many distinct orientational states do the particles occupy in this snapshot, what are those orientations, and how many particles are in each?*

The pipeline is:

```
shape JSON
   ↓  convex hull → coplanar face merging → topology validation
   ↓  candidate rotation axes + 37 candidate angles
   ↓  one-to-one vertex matching (Hungarian) → full-precision refit
   ↓  PROVE the result is a group (identity, inverses, closure)
   ↓  emit ±q equivalent-quaternion set
symmetry group G
   ↓
one GSD frame → first M particle quaternions → validate + renormalise
   ↓  freud AngularSeparationGlobal, block-wise
   ↓  condensed i<j distance array on DISK (memmap)
   ↓  complete-linkage clustering, cut at orientation tolerance
   ↓  RE-VERIFY every cluster diameter exactly
   ↓  symmetry-aware medoid per cluster
   ↓  deterministic ordering (largest cluster first)
raw clusters
   ↓  keep only those with size ≥ cluster_size_cutoff
   ↓  name them A, B, C, …
identified orientational states
   ↓
8 output files (4 CSVs + 1 symmetry CSV + 1 plotting CSV + 1 PNG + 1 JSON)
```

**What it deliberately does NOT do** — and this is the whole point of the "single-frame" designation:

| Absent feature | Present in the multi-frame sibling `v1p5` |
|---|---|
| Time averaging over frames | ✔ |
| A fixed reference frame | ✔ |
| Inter-frame state tracking / matching | ✔ |
| Tracking-angle tolerance | ✔ |
| Particle-order permutation stability trials | ✔ |
| Population time series, mean ± σ, presence fraction | ✔ |

So this is the **snapshot tool**: fast, simple, no cross-frame bookkeeping. Use it to explore a trajectory, to tune parameters cheaply, or when a single well-equilibrated configuration is all you care about. Use `v1p5` when you need time-resolved, tracked state populations.

**No positions are used.** There is no neighbour list, no cutoff radius, no simulation box, no periodic images, and no particle-type filtering. Two particles on opposite sides of the box are compared exactly as readily as two neighbours. This is a *global* orientational census, not a spatially resolved correlation function.

---

## 2. Output file map at a glance

Every run writes **eight** files (plus one optional directory). They form a deliberate hierarchy from coarse to fine granularity.

```
                         ┌─────────────────────────────────────┐
  WHAT ARE THE STATES?   │  _identified_states.csv             │  ← START HERE
        state-level      │  one row per identified state A,B,C │     the headline answer
                         └─────────────────────────────────────┘
                                        ▲ subset of (size ≥ cutoff)
                         ┌─────────────────────────────────────┐
  WHAT DID CLUSTERING    │  _all_frame_clusters.csv            │  ← audit / tuning
     ACTUALLY FIND?      │  one row per cluster, incl. tiny    │     did the cutoff discard much?
      cluster-level      └─────────────────────────────────────┘
                                        ▲ expands to members
                         ┌─────────────────────────────────────┐
  WHICH PARTICLE IS      │  _all_analysed_particle_membership   │  ← joins to positions
       WHERE?            │      .csv   (M rows, complete)      │
     particle-level      └─────────────────────────────────────┘
                                        ▲ filtered to state members
                         ┌─────────────────────────────────────┐
                         │  _identified_particle_state_         │  ← clean input for
                         │      membership.csv                 │     downstream analysis
                         └─────────────────────────────────────┘

  SUPPORTING FILES
  ┌──────────────────────────────────────────┬─────────────────────────────────┐
  │ _symmetry_quaternions.csv                │ the group G that defined distance│
  │ _single_frame_state_populations.png      │ the figure                       │
  │ for_plotting_..._state_populations.csv   │ exact plotted coordinates        │
  │ _metadata.json                           │ complete run record / provenance │
  │ _distance_files/  (only with a flag)     │ raw condensed distance memmap    │
  └──────────────────────────────────────────┴─────────────────────────────────┘
```

Full column-by-column specifications, with the purpose of each file spelled out, are in **§17** — that is the section to read if you mainly care about consuming the outputs.

---

# Understanding the problem

## 3. The physical question

In a simulation of anisotropic (non-spherical) particles — hard polyhedra, patchy colloids, liquid crystals of faceted bodies — a dense phase often develops **orientational order**. The particles do not point in arbitrary directions; they settle into a small number of preferred orientations.

* A **fluid** or a **plastic/rotator crystal** shows no sharp states: orientations are broadly distributed.
* An **orientationally ordered crystal** shows a handful of sharp states, typically related by the lattice point group.
* A **partially ordered** or **coexisting** system shows a few dominant states plus a diffuse background.

This program takes one snapshot and reports: the number of states, each state's defining orientation as a quaternion, each state's particle count and population fraction, and how tightly each state is clustered (its angular diameter).

## 4. Why this is hard

### 4.1 Particle symmetry makes "same orientation" ambiguous

A cube rotated 90° about a face axis is **indistinguishable** from the original cube. Two particles whose quaternions differ by exactly that rotation are physically in the *same* orientation. Any comparison must therefore quotient out the particle's own rotational symmetry group $G$. Getting $G$ wrong — missing operations, or wrongly including improper ones — silently corrupts every distance in the analysis and changes the answer.

### 4.2 Unit quaternions double-cover rotations

$q$ and $-q$ describe the identical physical rotation. HOOMD's stored sign is arbitrary. The distance function must be invariant to it.

### 4.3 The pairwise distance matrix is enormous

Clustering $M$ particles requires all $M(M-1)/2$ pairwise distances. For $M = 4096$ that is 8.4 million values (67 MB as float64); for $M = 20000$ it is 200 million (1.6 GB). A square $M \times M$ matrix would be twice as large again, with half the entries redundant.

### 4.4 "Cluster" must mean something precise

Hierarchical clustering with the wrong linkage gives clusters with no bounded internal spread — a chain of particles each 5° from the next can span 180° under single linkage. A meaningful orientational state must have a *bounded diameter*.

---

## 5. The solution strategy in five ideas

### Idea 1 — Reconstruct the symmetry group from the shape, and *prove* it is a group

Rather than trusting a hard-coded symmetry list, the program derives $G$ from the shape JSON: convex hull → candidate axes from vertices/faces/edges → candidate rotations → accept only those mapping the vertex set onto itself one-to-one → refine at full precision → **verify identity, inverses and closure exactly**. If closure fails, the run stops rather than proceeding with an incomplete group.

### Idea 2 — Define distance as the symmetry-reduced misorientation angle

$$d_G(q_i, q_j) = \min_{g \in G} \theta\bigl(q_i^{-1} q_j g\bigr) \in [0°, \theta_{\max}(G)]$$

computed by `freud.environment.AngularSeparationGlobal`, with the equivalent-orientation set containing **both** $+q$ and $-q$ for every $g \in G$.

### Idea 3 — Cluster with complete linkage, then *verify* the diameter

Complete-linkage (farthest-neighbour) clustering cut at height $\varepsilon_{\text{orient}}$ is *intended* to produce clusters whose maximum internal pairwise distance is $\le \varepsilon_{\text{orient}}$. The program does not trust this: it **recomputes the exact diameter of every returned cluster** and raises an error if any exceeds the tolerance. Every cluster in the output therefore carries a hard, verified guarantee:

$$\max_{i,j \in C} d_G(q_i, q_j) \le \varepsilon_{\text{orient}}.$$

This is precisely why complete linkage is used rather than single or average linkage — only complete linkage has a diameter interpretation.

### Idea 4 — Represent each cluster by a symmetry-aware *medoid*

The representative must itself be a valid orientation. Averaging quaternions is ill-defined under symmetry (there is no unique mean of a set of symmetry-equivalent rotations). So the program uses the **medoid**: the actual cluster member minimising the sum of within-cluster distances,

$$m(C) = \arg\min_{i \in C} \sum_{j \in C} d_G(q_i, q_j).$$

A real particle's real orientation, no averaging, robust to outliers.

### Idea 5 — Separate "raw clusters" from "identified states"

Complete linkage will always return *some* partition, typically including many tiny clusters of one to a few particles that are statistical noise rather than physical states. The program keeps both views:

* **raw clusters** — everything, written to `_all_frame_clusters.csv`, for auditing;
* **identified states** — only clusters with size $\ge$ `cluster_size_cutoff`, named `A, B, C, …`, written to `_identified_states.csv` and plotted.

Keeping both is what lets you check whether your cutoff discarded anything important.

---

## 6. Mathematical definitions

### 6.1 Orientation and misorientation

HOOMD stores each particle's orientation as a **scalar-first** unit quaternion $q = (w,x,y,z)$. The relative rotation between particles $i$ and $j$ is $\Delta_{ij} = q_i^{-1} q_j$, with rotation angle

$$\theta(\Delta) = 2\arccos\bigl(\lvert \operatorname{Re}\Delta \rvert\bigr) \in [0°, 180°].$$

The absolute value on the scalar part is what makes this insensitive to the $q \leftrightarrow -q$ ambiguity.

### 6.2 Symmetry-reduced distance

With $G$ the proper rotation group of the body,

$$d_G(q_i, q_j) = \min_{g \in G} \theta\bigl(q_i^{-1} q_j\, g\bigr).$$

Properties that matter here:

* **Symmetric.** $d_G(a,b) = d_G(b,a)$ because $G$ is closed under inverses — which the program verifies. This is what licenses storing only the $i<j$ triangle.
* **Zero iff equivalent.** $d_G(q,q) = 0$ exactly, in theory; freud may return a small nonzero floor in practice.
* **Bounded by $\theta_{\max}(G) \le 180°$**, shrinking as $|G|$ grows: $\approx 62.8°$ for the cube group $O$ ($|G|=24$), $\approx 75.5°$ for the tetrahedral group $T$ ($|G|=12$), $180°$ for the trivial group.
* **Satisfies the triangle inequality**, so it is a genuine metric on the quotient $SO(3)/G$ — which is what licenses hierarchical clustering at all.

### 6.3 Only *proper* rotations

Only proper rotations ($\det R = +1$) are used. Mirrors, inversions and rotoreflections are deliberately excluded: particle orientations and unit quaternions parameterise $SO(3)$, not $O(3)$, and no rigid motion can reflect a physical particle onto itself. So the program detects the **rotation group** $G$, not the full point group — for a cube, $|G| = 24$, not $|O_h| = 48$.

### 6.4 Cluster diameter, medoid, population fraction

$$\operatorname{diam}(C) = \max_{i,j \in C} d_G(q_i, q_j), \qquad m(C) = \arg\min_{i \in C} S_i,\ \ S_i = \sum_{j \in C} d_G(q_i,q_j)$$

Medoid ties (within `MEDOID_TIE_TOL_DEG` $= 10^{-10}$ degrees) are broken by the **smallest particle index**, making the choice fully deterministic. Singleton clusters are their own medoid with $S = 0$, diameter $0$.

$$\text{population fraction of state } s = \frac{\lvert C_s\rvert}{M}$$

> **Note the denominator: $M$, the number of *analysed* particles — not the number of particles in identified states.** So the fractions of the identified states sum to $\le 1$, and the deficit is exactly the below-cutoff population. This is deliberate and is the interpretively correct choice: it tells you what fraction of the system is in each recognised state, with the remainder honestly unaccounted for.

### 6.5 The tolerances, kept rigorously apart

| Constant / parameter | Default | Kind | Role |
|---|---|---|---|
| `orientation_angle_tol` | 41.0° | **Physical** | Maximum permitted cluster diameter. The main scientific knob |
| `cluster_size_cutoff` | 200 | **Physical** | Minimum particle count for a cluster to be an identified state |
| `--angle-validation-tol` | $10^{-5}$° | Numerical | Slack when checking computed angles lie in $[0°,180°]$ |
| `CLUSTER_DIAMETER_EPS_DEG` | $10^{-10}$° | Numerical | Binary-roundoff slack in the diameter check. Deliberately far too small to relax the physical criterion |
| `MEDOID_TIE_TOL_DEG` | $10^{-10}$° | Numerical | Tie window for medoid selection only |
| `10^{-p}` (`--precision`) | $10^{-2}$ | Geometric | Vertex-matching tolerance in shape-coordinate units |

> **Design principle inherited from the multi-frame version:** freud's numerical self-angle floor is **never** allowed to relax the user's physical cluster-diameter criterion. Some freud builds compute internally in single precision, so $d_G(q,q)$ can come back as e.g. $0.02°$ rather than $0$. The diameter gate uses `CLUSTER_DIAMETER_EPS_DEG` $= 10^{-10}$, which cannot absorb such a floor — by design.

> **Vestigial constants.** `NUMERICAL_SELF_ANGLE_HARD_LIMIT_DEG` (0.5) and `NUMERICAL_FLOOR_SAFETY_FACTOR` (4.0) are **defined at the top of the file but never used** in this version. In the multi-frame sibling they guard the reference-medoid distance matrix, which does not exist here. They are harmless; do not be misled into thinking a self-angle check is being performed.

---

# Using the program

## 7. Installation

```bash
pip install numpy scipy matplotlib gsd freud-analysis
```

**Two-tier import strategy:**

| Tier | Packages | When | Why |
|---|---|---|---|
| Startup | `numpy`, `scipy` (`cluster.hierarchy.fcluster/linkage`, `optimize.linear_sum_assignment`, `sparse.csgraph.connected_components`, `spatial.ConvexHull/cKDTree/distance_matrix`, `spatial.transform.Rotation`) | Module import, wrapped in `try/except ImportError` → `SystemExit` with an install hint | Drive the geometry and clustering core |
| Runtime | `freud`, `gsd.hoomd`, `matplotlib.pyplot` | Lazily inside `import_runtime_packages()` | Lets the symmetry code be imported and tested without freud or GSD installed |

Standard library used: `argparse`, `csv`, `json`, `math`, `shutil`, `sys`, `tempfile`, `dataclasses`, `pathlib`, `typing`. (`os` and `typing.Iterable` are imported but unused — harmless.)

---

## 8. Quick start

**Fully interactive:**

```bash
python ref_frame_calc_rigorous_single_frame_v1p1.py trajectory.gsd shape.json
```

**Fully non-interactive** (batch/cluster safe):

```bash
python ref_frame_calc_rigorous_single_frame_v1p1.py trajectory.gsd shape.json \
  --frame 499 --particles 4096 \
  --edges 25 --faces 15 --precision 4 \
  --orientation-angle-tol 41.0 --cluster-size-cutoff 200 \
  --block-size 256 --output-dir ./frame499 --no-show
```

> To run non-interactively you must supply **all seven** scientific flags: `--frame`, `--particles`, `--edges`, `--faces`, `--precision`, `--orientation-angle-tol`, `--cluster-size-cutoff`. Omit any one and the program drops into an interactive prompt, which will hang a batch job. Unlike the multi-frame version, **all prompts occur before any heavy computation**, so a missing flag hangs immediately at startup rather than mid-run — easier to diagnose.

---

## 9. Inputs

### 9.1 GSD trajectory (positional argument 1)

A HOOMD-blue GSD file. The program reads **only** `trajectory[frame_index].particles.orientation`. Positions, box, types, diameters, images, velocities and all other fields are ignored. Consequences:

* Particle **type is ignored** — in a multi-component system all types are pooled into one orientational analysis.
* There is **no spatial information at all**.
* Opened read-only via `gsd.hoomd.open(name=..., mode="r")`, with a positional-argument fallback for older GSD releases that reject keyword arguments.
* The frame is accessed by index, so random access is required (standard for GSD). **Any** frame may be chosen — there is no "last F frames" restriction.

### 9.2 Shape JSON (positional argument 2)

Any JSON containing an $N\times3$ array of vertices. The reader is **schema-agnostic** and searches recursively, so all of these work:

```json
{"vertices": [[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]}
```
```json
{"shape": {"type": "ConvexPolyhedron", "vertices": [[1,1,1], "..."]}}
```
```json
[{"name": "particle", "vertices": [[1,1,1], "..."]}]
```

Requirements enforced on the selected array:

* shape $(N,3)$ with $N \ge 4$, all entries finite;
* **no duplicate or numerically indistinguishable vertices** (KD-tree check at $\max(10^{-12}R,\ 10^{-14})$, where $R$ is the maximum centred vertex radius) — duplicates make the convex topology and the symmetry permutation ambiguous;
* not all collapsed to one point.

**Units are arbitrary, but the symmetry matching tolerance $10^{-p}$ is absolute in those same units.** See §12.2.

---

## 10. Complete CLI reference

```
python ref_frame_calc_rigorous_single_frame_v1p1.py TRAJECTORY.gsd SHAPE.json [options]
```

### Scientific parameters (prompt if omitted)

| Flag | Type | Interactive default | Validator | Meaning |
|---|---|---|---|---|
| `--frame` | int | `T - 1` (last frame) | $0 \le v < T$ | Zero-based trajectory frame index to analyse |
| `--particles` | int | all available in that frame | $2 \le v \le$ available | Particles taken from the **start** of the frame |
| `--edges` | int | `25` | $v \ge 1$ | Expected polyhedron edge count (validation target) |
| `--faces` | int | `15` | $v \ge 1$ | Expected polyhedron face count (validation target) |
| `--precision` | int | `2` | $v \ge 0$ | $p$ in symmetry vertex-matching tolerance $10^{-p}$ |
| `--orientation-angle-tol` | float | `41.0` | $0 < v \le 180$ | **Maximum cluster diameter, degrees** |
| `--cluster-size-cutoff` | int | `min(200, M)` | $1 \le v \le M$ | **Minimum particle count for an identified state** |

### Numerical / infrastructure parameters (never prompted)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--block-size` | int | `128` | Query orientations per freud call. Memory knob only; does not change results |
| `--angle-validation-tol` | float | `1e-5` | Numerical slack for the $[0°,180°]$ angle-range checks (degrees) |
| `--output-dir` | str | trajectory's parent directory | Output location (created with `parents=True, exist_ok=True`) |

### Behaviour flags

| Flag | Effect |
|---|---|
| `--no-show` | Save the PNG without opening an interactive matplotlib window. **Always use in batch jobs** |
| `--keep-distance-files` | Retain the condensed distance memmap in `<prefix>_distance_files/` instead of deleting it |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Completed; all outputs written |
| `1` | Controlled error, printed to stderr as `ERROR: <message>` |

Caught exception classes: `FileNotFoundError`, `ImportError`, `ValueError`, `RuntimeError`, `OSError`, `json.JSONDecodeError`. Anything else (`KeyboardInterrupt`, `MemoryError`, an unexpected library exception) propagates with a full traceback.

### Module-level constants (edit the top of the file to change defaults)

```python
CURRENT_SUGGESTED_PRECISION            = 2
CURRENT_SUGGESTED_NUM_EDGES            = 25
CURRENT_SUGGESTED_NUM_FACES            = 15
CURRENT_SUGGESTED_ORIENTATION_TOL_DEG  = 41.0
CURRENT_SUGGESTED_CLUSTER_SIZE_CUTOFF  = 200
DEFAULT_BLOCK_SIZE                     = 128
DEFAULT_ANGLE_VALIDATION_TOL_DEG       = 1.0e-5
NUMERICAL_SELF_ANGLE_HARD_LIMIT_DEG    = 0.5    # defined, UNUSED in this version
NUMERICAL_FLOOR_SAFETY_FACTOR          = 4.0    # defined, UNUSED in this version
CLUSTER_DIAMETER_EPS_DEG               = 1.0e-10
MEDOID_TIE_TOL_DEG                     = 1.0e-10
```

> The suggested defaults `edges=25, faces=15` imply, via Euler's formula $V - E + F = 2$, a polyhedron with $V = 12$ vertices. The program does not itself check Euler's formula, but it is a useful sanity check on the numbers you type.

---

## 11. Interactive prompts, in order

`prompt_value()` loops until valid input arrives. **An empty line accepts the bracketed default.** A conversion failure or validator rejection reprints the error and re-prompts — it never crashes on bad typing. `resolve_or_prompt()` skips the prompt when the CLI flag was supplied, but then a validator failure raises `ValueError` immediately with no fallback prompt.

All seven prompts occur **before** the shape is read and before any distance computation begins.

| # | Prompt | Default | Validator |
|---|---|---|---|
| 1 | Which trajectory frame should be analysed? | `T - 1` | $0 \le v < T$ |
| 2 | How many particles should be analysed? | all available | $2 \le v \le$ available |
| 3 | Enter the intended number of polyhedron edges | `25` | $v \ge 1$ |
| 4 | Enter the intended number of polyhedron faces | `15` | $v \ge 1$ |
| 5 | Enter invariant-quaternion symmetry precision exponent p | `2` | $v \ge 0$ |
| 6 | Maximum allowed pairwise angle within a cluster (degrees) | `41.0` | $0 < v \le 180$ |
| 7 | Minimum particle count for an identified single-frame state | `min(200, M)` | $1 \le v \le M$ |

Between prompts 1 and 2 the program reads the frame's orientation array, validates its shape, and prints the available particle count, so prompt 2's default and bound are informed by the actual data.

---

## 12. Tuning guide: applying this to a new system

### 12.0 Recipe at a glance

```
Step 1  Determine V, E, F of your particle          → --edges, --faces
Step 2  Find a symmetry precision p that is stable  → --precision
Step 3  Choose the orientation tolerance            → --orientation-angle-tol   ← the key knob
Step 4  Choose the size cutoff                      → --cluster-size-cutoff
Step 5  Choose the frame                            → --frame
Step 6  Size the block for your memory budget       → --block-size
Step 7  Scale up the particle count                 → --particles
```

Because this version is cheap (one frame, no stability trials), it is the **ideal tool for steps 1–4**. Tune here, then carry the parameters over to the multi-frame version if you need time resolution.

### 12.1 Edges and faces (`--edges`, `--faces`)

These are validation targets, not free parameters — they must equal the true topology of your convex polyhedron.

**How to get them.** Either compute directly:

```python
import json, numpy as np
from scipy.spatial import ConvexHull
v = np.array(json.load(open("shape.json"))["vertices"], float)
v -= v.mean(axis=0)
h = ConvexHull(v)
print("vertices:", len(v), "hull triangles:", len(h.simplices))
```

or — easier — just run the program once with any guess. The **topology scan table is printed whether or not it matches**, so you can read the correct merged polygonal counts straight off it and re-run:

```
merge tolerance       recovered edges       recovered faces
 1.0e-12                          36                    24
 ...
 1.0e-06                          25                    15    ← use these
```

Cross-check with Euler's formula: $V - E + F = 2$.

**If no tolerance ever reproduces a sensible topology,** your shape is probably non-convex, has near-coplanar faces that never merge cleanly, or contains duplicate vertices.

### 12.2 Symmetry precision (`--precision`, i.e. $p$)

$\varepsilon_{\text{match}} = 10^{-p}$ is an **absolute Euclidean distance in shape-coordinate units**, not a relative tolerance. This is the single most misunderstood parameter.

* Vertices of magnitude $\sim 1$: $p=2$ means $\varepsilon = 0.01$, a 1% tolerance — reasonable.
* Vertices of magnitude $\sim 0.1$: $p=2$ is a **10% tolerance** — far too loose; it may accept near-symmetries that are not symmetries.
* Vertices of magnitude $\sim 100$: $p=2$ is $10^{-4}$ relative — possibly too tight to survive the JSON file's finite precision.

**Procedure: scan $p$ and look for a plateau.** Run with `--precision 1,2,3,4,5,6` and record `Distinct physical proper rotations detected`:

| p | detected rotations |
|---|---|
| 1 | 48  ← too loose, spurious operations |
| 2 | 24 |
| 3 | 24 |
| 4 | 24  ← plateau: this is the true group |
| 5 | 24 |
| 6 | 12  ← too tight, real operations rejected |

Use a value in the middle of the plateau. Sanity-check against theory: tetrahedron 12, cube/octahedron 24, dodecahedron/icosahedron 60, $n$-gonal prism $2n$, no symmetry 1.

**Errors that tell you $p$ is wrong:**

* *"Full-precision rotation refinement changed the discovered vertex permutation"* → $p$ too **loose**; increase it.
* *"...not closed under composition"* / *"missing an inverse"* → usually $p$ too **tight** (real operations were rejected), occasionally a genuinely unreachable rotation order (§25.2).

### 12.3 Orientation tolerance (`--orientation-angle-tol`) — the key knob

This sets the maximum permitted **cluster diameter** and decides how many states you find. Too small and every particle becomes its own singleton; too large and everything collapses into one state.

The hard ceiling is $\theta_{\max}(G)$ for your particle. The program prints `maximum_pair_angle_deg` — use that as your scale.

**Method A — use the pairwise-angle histogram (recommended).** Compute $P(\theta_{ij})$ first with the companion global pairwise-orientation histogram tool. An orientationally ordered system shows a peak near $\theta = 0$ (within-state pairs) separated by a **minimum** from peaks at larger angles (between-state pairs). Set $\varepsilon_{\text{orient}}$ at that minimum. The default 41° comes from exactly this construction for the reference system.

```
P(θ)
 |  ╱‾╲                    ╱‾╲
 | ╱   ╲                  ╱   ╲
 |╱     ╲________________╱     ╲
 +----------|--------------------→ θ
            41°  ← minimum → tolerance
```

**Method B — scan and look for a plateau in the state count.** This version is cheap enough to scan freely:

```bash
for tol in 10 20 25 30 35 41 50 60 70 90; do
  python ref_frame_calc_rigorous_single_frame_v1p1.py traj.gsd shape.json \
    --frame 499 --particles 2000 --edges 25 --faces 15 --precision 4 \
    --orientation-angle-tol $tol --cluster-size-cutoff 100 \
    --output-dir ./scan_$tol --no-show
done
```

Then read `raw_cluster_count` and `identified_cluster_count` out of each `_metadata.json`:

| tolerance | raw clusters | identified |
|---|---|---|
| 10° | 1592 | 0 |
| 25° | 41 | 2 |
| 35° | 9 | 4 |
| 41° | 6 | **4** ← plateau |
| 50° | 5 | **4** ← plateau |
| 70° | 2 | 2 |
| 90° | 1 | 1 |

The plateau in *identified* cluster count is the physically robust answer. Report the plateau width in your methods section.

**Method C — theory.** If the states are related by the crystal's point group, compute the expected medoid separations analytically and set $\varepsilon_{\text{orient}}$ to roughly half the smallest.

**Sanity check after choosing.** The program prints `maximum validated diameter`. It should sit comfortably *below* your tolerance. If it is pinned exactly at the tolerance, clustering is being cut off artificially and distinct states are probably being merged — reduce the tolerance and see whether the state count increases.

### 12.4 Cluster size cutoff (`--cluster-size-cutoff`)

In this version the cutoff has exactly **one** role (unlike the multi-frame version, where the same number serves three): it is the **minimum instantaneous particle count for a cluster to be promoted to an identified state**. Clusters below it remain in the raw output but get `identified_state_id = -1`, an empty `cluster_name`, and are excluded from the plot.

**Choosing:** think as a fraction of $M$. 200 out of 4096 is $\approx 5\%$.

* **1–2% of $M$** — permissive; catches minority states, admits noise
* **5% of $M$** — balanced; the default's spirit
* **10% of $M$** — conservative; only major states

**How to check your choice.** Open `_all_frame_clusters.csv`, sort by `cluster_size` descending, and look for a gap. A healthy system shows a clear separation:

```
cluster_size:  1204, 1180, 1002, 640,  |  7, 4, 3, 2, 2, 1, 1, ...
                    ← real states →    |   ← noise →
```

Put the cutoff in the gap. If there is no gap, your orientation tolerance is likely wrong (§12.3).

Also read `ignored_below_cutoff_particle_count` from the metadata. If a large fraction of the system is being discarded, either lower the cutoff or revisit the tolerance.

### 12.5 Frame choice (`--frame`)

Any frame $0 \le f < T$. Defaults to the last.

* **Last frame (default)** — the most equilibrated configuration in most runs.
* **An early frame** — to see the initial or transient state structure.
* **Several frames** — the honest use of a single-frame tool is to run it on a handful of frames and confirm the state count and populations are consistent. If they are not, the system is not in a steady state and you should move to the multi-frame version.

Because there is no tracking here, **state names `A, B, C` are NOT comparable across separate runs on different frames.** Names are assigned by descending cluster size within each run, so state `A` in frame 400 and state `A` in frame 499 may be different physical orientations. To compare, match the medoid quaternions yourself, or use the multi-frame version, which does this properly.

### 12.6 Block size (`--block-size`)

Affects **peak memory only**; never changes the numbers. Peak temporary angle-matrix memory is

$$8 \times \text{block\_size} \times M \ \text{bytes}.$$

| $M$ | block 128 | block 512 | block 2048 |
|---|---|---|---|
| 4 096 | 4 MB | 17 MB | 67 MB |
| 20 000 | 20 MB | 82 MB | 328 MB |
| 50 000 | 51 MB | 205 MB | 819 MB |

Larger blocks mean fewer freud calls (modestly faster) and more RAM. Reduce it if you hit `MemoryError` during the distance stage; raise it to 512–1024 on a large-memory node.

### 12.7 Particle count (`--particles`)

Dominates cost: everything scales as $O(M^2)$ or worse. Start with $M \le 2000$ for parameter scans, then scale up for the production number.

The selection is the **first $M$ particles by index**, not random sampling. Deterministic and reproducible, but if your GSD file has index-correlated structure — types sorted, initial lattice ordering preserved, or a spatial sort applied at write time — then $M <$ total silently analyses a biased subset. Prefer $M = $ all, or verify that index order is uncorrelated with structure.

### 12.8 Numerical tolerance (`--angle-validation-tol`)

Default $10^{-5}$°. In this version it is used **only** to permit negligible roundoff outside $[0°, 180°]$ in the raw freud output. Raise it only if you see spurious "negative angular distance" or "exceeds 180 degrees" errors, and keep it far smaller than any physical angle.

---

# How it executes

## 13. Top-level execution map

`main()` runs the following, all inside one `try` block that converts expected failures into `ERROR: …` on stderr and returns `1`:

```
 1  Parse the CLI
 2  Validate --block-size >= 1 and --angle-validation-tol > 0
 3  Lazy-import freud / gsd.hoomd / matplotlib
 4  Open the GSD read-only; resolve the shape path (content checked later)
 5  Read len(trajectory); reject an empty trajectory; print the frame summary
 6  Prompt/resolve --frame
 7  Read that frame's orientation array; verify shape (N,4); print available count
 8  Prompt/resolve: particles M, edges, faces, precision p,
       orientation tolerance, cluster size cutoff          ← ALL user input done here
 9  read_shape_vertices()                                            ─┐
10  detect_proper_rotational_symmetries()                             ├─ PART A
11  Create the output directory and the filename prefix              ─┘
12  Create the distance-file directory (persistent or tempfile.mkdtemp)
13  cluster_one_frame()                                              ─── PART B
       finally: rmtree the temp directory unless --keep-distance-files
14  identified_clusters_for_single_frame()                            ─── PART C
15  Print the one-line frame summary
16  save_symmetry_outputs()          → 1 CSV                          ─┐
17  save_single_frame_outputs()      → 4 CSVs                          ├─ OUTPUTS
18  save_single_frame_population_plot() → 1 PNG + 1 CSV                │
19  save_single_frame_metadata()     → 1 JSON                         ─┘
20  Print the final state summary and all output paths; return 0
```

**Design note on ordering:** every prompt and every cheap validation happens before the shape is even read. The expensive symmetry detection (steps 9–10) precedes the far more expensive clustering (step 13). A wrong shape or topology therefore fails in seconds, not after the $O(M^2)$ work.

---

## 14. Part A — Shape processing and symmetry group

This entire section is **byte-identical** to the multi-frame version `v1p5` (lines 92–1192 here correspond exactly to lines 104–1204 there). It is documented in full for self-containedness.

### 14.1 Reading the vertices

Three collaborating functions:

* **`_as_vertex_array(value)`** — returns `np.asarray(value, float)` if and only if it is 2-D, has 3 columns, has $\ge 4$ rows, and is entirely finite; otherwise `None`.
* **`_collect_vertex_candidates(obj, key_hint)`** — recursive descent. A valid array is emitted and descent stops. A `dict` is traversed with keys containing `"vert"` first, then alphabetically (deterministic order). A `list`/`tuple` is traversed element-wise with the inherited key hint.
* **`read_shape_vertices(path)`** — scores each candidate as
  $$\text{score} = 10000 \cdot \mathbb{1}[\text{"vert"} \in \text{key}] + N_{\text{rows}}$$
  so a small array under a key named `vertices` beats a large unrelated $N\times3$ array elsewhere in the file (e.g. a position table), while among equally named candidates the largest wins. Then runs the degeneracy and duplicate checks and prints the winning key hint and vertex count.

### 14.2 Centring

$$\mathbf{c} = \tfrac1N\textstyle\sum_a \mathbf{r}_a, \qquad \tilde{\mathbf{r}}_a = \mathbf{r}_a - \mathbf{c}$$

A rotation must act about the particle centre; rotating about the coordinate origin is valid only if the shape file is already centred there. The program prints $\mathbf{c}$ and $\lVert\mathbf{c}\rVert$, reports `CENTRE CHECK: PASSED` if $\lVert\mathbf{c}\rVert \le 10^{-p}$ (otherwise an informational note that it will translate), then verifies the recentring residual is below $100\epsilon_{\text{machine}}$, raising `RuntimeError` if not.

> The **arithmetic vertex mean** is used, not the volumetric centroid. For any shape with a nontrivial rotation group the vertex mean is a fixed point of that group, so it is the correct rotation centre — which is exactly the case of interest.

### 14.3 Convex decomposition and the topology gate

**`_merge_coplanar_hull_triangles(vertices, merge_tolerance)`**

1. `ConvexHull` (Qhull) returns a *triangulated* surface: `simplices` plus one plane equation $[n_x,n_y,n_z,b]$ per triangle, with consistently outward unit normals satisfying $\mathbf{n}\cdot\mathbf{x} + b = 0$.
2. Triangles belonging to the same planar face have nearly identical equation rows, so the equations are treated as points in **4-D** and `cKDTree.query_pairs(merge_tolerance)` finds coplanar pairs.
3. Those pairs fill a dense symmetric `int8` adjacency matrix of size $n_{\text{tri}}^2$; `scipy.sparse.csgraph.connected_components(directed=False)` groups triangles into faces.
4. Per merged face: vertex set from `np.unique(simplices.ravel())`; plane equation = the **mean** of member equations, rescaled so $\lVert\mathbf{n}\rVert$ is exactly 1 (a zero normal raises `RuntimeError`); vertices ordered **cyclically** by building an in-plane orthonormal basis ($\mathbf{a}$ from the first vertex's displacement from the face centroid, $\mathbf{b} = \mathbf{n}\times\mathbf{a}$) and sorting by $\operatorname{atan2}(\mathbf{d}\cdot\mathbf{b}, \mathbf{d}\cdot\mathbf{a})$. Degenerate bases raise `RuntimeError`.
5. Edges come from consecutive pairs in the cyclic ordering (`np.roll(face, -1)`), stored as sorted `(min,max)` tuples in a `set`, which automatically deduplicates the two faces sharing each edge.

**`find_validated_convex_decomposition()`** scans the tolerance ladder $10^{-12}, 10^{-11}, \ldots, 10^{-4}$ (`range(12, 3, -1)`), prints every trial, and **returns on the first exact match to BOTH counts**, printing `TOPOLOGY CHECK: PASSED`, the accepted tolerance and the hull volume. If none matches it raises `RuntimeError` reporting expected versus final-trial counts. Because the ladder runs tight→loose, the **tightest** tolerance reproducing your topology is selected.

This is a **hard gate**, and deliberately so: an over- or under-merged face set produces a wrong candidate-axis set and hence a wrong symmetry group — the classic silent failure that would corrupt every subsequent distance.

### 14.4 Candidate axes and angles

**`build_candidate_axis_lines()`** — four classes:

| Class | Vector | Count |
|---|---|---|
| Centre → vertex | $\tilde{\mathbf{r}}_a$ | $V$ |
| Centre → face centroid | mean of the face's vertices | $F$ |
| Centre → perpendicular foot on face plane | $-b\,\mathbf{n}$ | $F$ |
| Centre → edge midpoint | $\tfrac12(\tilde{\mathbf{r}}_a + \tilde{\mathbf{r}}_b)$ | $E$ |

$V + 2F + E$ raw candidates. The perpendicular-foot class is the corrected form of the legacy "centre-to-face-normal" construction: in centred coordinates with $\lVert\mathbf{n}\rVert=1$, the foot of the perpendicular from the origin to the plane is exactly $-b\mathbf{n}$. For an irregular face this differs from the centroid, so both classes are needed.

Deduplication: drop vectors of norm $\le 10^{-14}$ (a feature sitting on the centre defines no axis); normalise and sign-canonicalise via `_canonical_axis_line` (flip so the first component with $\lvert v\rvert > 10^{-14}$ is positive); drop any axis with $\lvert \mathbf{n}\cdot\mathbf{n}_{\text{prev}}\rvert \ge 1 - 10^{-12}$ (parallel **or** antiparallel ⇒ the same geometric line). The first label producing each unique line is retained for traceability. An empty set raises `RuntimeError`.

**`historical_candidate_angles_rad()`** — a fixed list of **37 angles**: 19 positive (180, 120, 240, 90, 270, 72, 144, 216, 288, 60, 300, 45, 135, 225, 315, 252, 324, 36, 108) and 18 negative (the same set excluding $\pm180$, which is self-inverse). These are the non-identity rotations of $C_n$ for $n \in \{2,3,4,5,6,8,10\}$. Negative angles are essential because axis *lines* were canonicalised to a single sign, so rotations about the flipped direction are only reachable via negative angles.

Total candidates tested $= n_{\text{unique axes}} \times 37$.

### 14.5 Candidate testing, matching and refinement

Operations live in `operations_by_permutation: dict[tuple[int,...], SymmetryOperation]`, keyed by the induced **integer vertex permutation**, seeded explicitly with the identity (angle $0$ is not in the candidate list).

For each (axis, angle):

1. **Construct** $R$ from the rotation vector $\boldsymbol{\omega} = \mathbf{n}\theta$ via `Rotation.from_rotvec`.
2. **Apply** to all centred vertices.
3. **Match one-to-one** — `_one_to_one_vertex_mapping()` builds the full $N\times N$ cost matrix $C_{ab} = \lVert\tilde{\mathbf{r}}'_a - \tilde{\mathbf{r}}_b\rVert$ via `scipy.spatial.distance_matrix` and solves the **linear sum assignment (Hungarian) problem**. This returns the globally minimum-cost *perfect* matching, guaranteeing each rotated vertex is used once and each reference vertex hit once — exactly the permutation condition for a rigid symmetry.

   > This is a substantive correctness improvement over greedy nearest-neighbour matching, which can assign two rotated vertices to the same target and thereby accept a non-symmetry as a symmetry. The Hungarian solve cannot.

4. **Reject** if $\max_a C_{a,\sigma(a)} > 10^{-p}$. The **maximum**, not the RMS — the symmetry condition is enforced for every vertex individually, not on average.
5. **Refine at full precision** — the candidate axes derive from finite-precision coordinates, so the raw candidate rotation is only approximately the true symmetry. With $\sigma$ known, `_refine_rotation_for_permutation` solves the orthogonal Procrustes / Wahba problem via `Rotation.align_vectors(target, source)`, minimising $\sum_a\lVert\mathbf{t}_a - R\mathbf{s}_a\rVert^2$, with a `RuntimeError` guard if $\det R < 0$.
6. **Re-verify independently** — apply the refined $R$, recompute the assignment from scratch. If the refined permutation differs from the discovery permutation, raise `RuntimeError` (the tolerance does not resolve the vertex correspondence uniquely; the user is told to use a tighter $p$). If the refined maximum residual still exceeds $10^{-p}$, skip the candidate.
7. **Store**, deduplicating by permutation and keeping the discovery with the **smallest refined maximum residual** when several axis/angle combinations find the same permutation.

Operations are then sorted deterministically by $(\lVert R - I\rVert_F,\ \sigma)$ — identity first, then increasing rotation magnitude, with the permutation tuple as an exact tiebreak.

### 14.6 Group validation — the most valuable check in the program

**Why in permutation space?** Rotation matrices and quaternions carry floating-point noise, so group axioms tested on them would require arbitrary tolerances. Their action on a finite vertex set is captured *exactly* by integer permutations, making the checks discrete and unambiguous.

`validate_permutation_group()` checks exhaustively:

1. **Identity** — $(0,1,\ldots,N-1)$ must be present.
2. **Inverses** — for every $\sigma$, construct $\sigma^{-1}$ (`inverse[target] = source`) and require membership.
3. **Closure** — for every ordered pair, `_compose_permutations(first, second)[i] = second[first[i]]` must be in the set. An $O(n^2 N)$ double loop.

Any failure raises `RuntimeError` stating that the candidate-angle search is incomplete or inconsistent. Success prints `ROTATIONAL-GROUP CHECK: PASSED (identity, inverses, and closure).`

A missing symmetry operation would silently **inflate every computed distance**, splitting one physical state into several. Closure failure catches exactly that.

### 14.7 Equivalent quaternion set

`_rotation_to_wxyz` converts SciPy's `[x,y,z,w]` to freud/HOOMD scalar-first `[w,x,y,z]` (index reorder `[3,0,1,2]`), renormalises, and applies the first-nonzero-component-positive sign convention. Then both signs are emitted, **interleaved**:

```python
equivalent_quaternions[0::2] =  physical_quaternions   # +q
equivalent_quaternions[1::2] = -physical_quaternions   # -q
```

giving a $(2n,4)$ array where row $2k$ and row $2k+1$ are $\pm q_k$. Two assertions follow: every row unit-norm to `atol=1e-12`, and rows $2k+1$ exactly $= -$ rows $2k$ (`atol=0, rtol=0` — appropriate because they were produced by unary negation, so bit-exact equality is expected). Prints `Q/-Q CHECK: PASSED`.

Supplying both signs makes the distance invariant to the arbitrary sign convention of the stored HOOMD quaternions, regardless of freud's internal handling. It costs a factor of two in the inner minimisation and nothing in correctness.

---

## 15. Part B — Distances, clustering and medoids

`cluster_one_frame()` orchestrates four steps.

### 15.1 Load and validate orientations

`validate_and_normalize_orientations()`:

1. `np.asarray(snapshot.particles.orientation, float64)`.
2. Reject `ndim != 2` or `shape[1] != 4` → `ValueError` naming the frame and shape.
3. Reject fewer rows than requested → `ValueError` naming both counts.
4. Slice the **first $M$** rows and force C-contiguity via `np.ascontiguousarray` (freud requires it).
5. Reject any non-finite entry → `ValueError`.
6. Reject any quaternion of norm $\le 10^{-14}$ → `ValueError` ("contains a zero quaternion").
7. **Record** $\max_i \bigl\lvert \lVert q_i\rVert - 1 \bigr\rvert$ *before* correction. This is the audit trail for GSD-writer precision loss; it is printed and stored in the metadata as `maximum_quaternion_norm_deviation`.
8. Normalise `array /= norms[:, None]`. Small storage deviations are silently corrected — the recorded deviation is your evidence of how large they were.

### 15.2 Build the condensed distance array on disk

`compute_condensed_distance_memmap()` is the memory-critical routine.

**Storage layout.** A square $M\times M$ float64 matrix would waste half its entries to the symmetry $d(i,j)=d(j,i)$, plus a useless diagonal. Instead only the strict upper triangle is stored, in **SciPy's condensed order**:

$$(0,1),(0,2),\ldots,(0,M{-}1),\;(1,2),\ldots,(1,M{-}1),\;\ldots,\;(M{-}2,M{-}1)$$

exactly $M(M-1)/2$ values, written to a `np.memmap(dtype=float64, mode="w+", shape=(pair_count,))`. `mode="w+"` creates or overwrites and permits read and write. The raw storage therefore lives on **disk**, not in RAM, and the layout is directly consumable by `scipy.cluster.hierarchy.linkage` with no conversion.

**Index arithmetic.** `condensed_index(M, i, j)` implements the standard formula

$$\text{idx} = M\,i - \frac{i(i+1)}{2} + j - i - 1, \qquad i < j$$

vectorised over NumPy arrays, with `low = min(i,j)`, `high = max(i,j)`, and a `ValueError` for self-pairs (for which condensed indexing is undefined).

**Block loop.** For `block_start in range(0, M, block_size)`:

```python
block_stop = min(block_start + block_size, M)
query = orientations[block_start:block_stop]
calculator.compute(orientations, query, equivalent_quaternions)
block = np.rad2deg(np.asarray(calculator.angles, dtype=np.float64))  # (block, M)
```

One `AngularSeparationGlobal` object is created per frame and reused across all blocks. The output shape is **explicitly checked** against `(block_stop - block_start, M)`, raising `RuntimeError` on mismatch — a guard against a silent axis transposition across freud versions, which would still produce a plausible-looking but wrong result. Non-finite output raises immediately.

**Row-wise upper-triangle extraction.** For each row (global particle $i$), take `block[local_row, i+1:]` — a NumPy **view**, not a copy. This excludes $j<i$ (already stored as $d(j,i)$) and $j=i$ (the self-pair). For $i = M-1$ the slice is empty and is skipped. Per row:

* reject $\min < -\varepsilon_{\text{valid}}$ → `RuntimeError` "negative angular distance detected";
* reject $\max > 180 + \varepsilon_{\text{valid}}$ → `RuntimeError` "angular distance exceeds 180 degrees";
* `np.clip(values, 0.0, 180.0)` — removes only negligible roundoff outside the theoretical interval, since anything substantial has already raised;
* write contiguously at the cursor and advance it. **The row-wise upper-triangle order naturally generates exactly SciPy's condensed ordering**, so no reindexing pass is needed;
* update the true global min/max (seeded from $\pm\infty$).

**Accounting.** After the loop, `cursor != pair_count` raises `RuntimeError` quoting both numbers — this catches any block-index or slicing error. Then `distances.flush()` synchronises the modified pages with the underlying file before linkage reads it.

Returns the memmap plus `DistanceDiagnostics(minimum_angle_deg, maximum_angle_deg)` over all $i<j$ pairs.

### 15.3 Complete-linkage clustering

`complete_linkage_labels()`:

```python
hierarchy = linkage(condensed_distances, method="complete", optimal_ordering=False)
labels = fcluster(hierarchy, t=orientation_angle_tolerance_deg,
                  criterion="distance").astype(np.int64) - 1
```

* `method="complete"` — farthest-pair linkage, $D(A,B) = \max_{i\in A, j\in B} d_G(i,j)$; the only linkage with a diameter interpretation.
* `optimal_ordering=False` — skips leaf reordering, a visualisation nicety not used to define cluster identity.
* `criterion="distance"` — no cluster is formed through a merge above $t$.
* SciPy's 1-based labels are shifted to 0-based for array indexing. The numeric label values are arbitrary; particle membership is the physical information.

> **Memory note:** `linkage` converts its input to a contiguous in-memory float64 array. The memmap therefore reduces persistent *storage*, but peak RAM during linkage still includes one full $8 \times M(M-1)/2$-byte copy. See §22.

### 15.4 Construct, verify and order the clusters

`construct_frame_clusters()`:

1. For each unique label, gather `members = np.where(labels == label)[0]` and call `cluster_distance_statistics()`, which in one pass over the cluster's upper triangle (reading directly from the memmap) computes:
   * $S_i$ for every member — each pair contributes to both endpoints' sums;
   * the exact **diameter**;
   * the **medoid** (min $S$, ties within $10^{-10}$° broken by smallest particle index);
   * $\max_j d(m, j)$, the maximum distance to the medoid.
   Singletons short-circuit to $(i, 0, 0, 0)$.
2. **The diameter gate.** If $\operatorname{diam}(C) > \varepsilon_{\text{orient}} + 10^{-10}$, raise `RuntimeError` quoting both the computed diameter and the requested tolerance to 12 significant figures.
3. Build a `FrameCluster` with the medoid's **sign-canonicalised** quaternion (`canonicalize_quaternion_sign`: normalise, then flip so the first component with $\lvert\cdot\rvert > 10^{-14}$ is positive).
4. **Deterministic ordering** — sort whole records by
   $$(-\text{size},\ \ \text{medoid particle index},\ \ \text{members tuple})$$
   so `local_cluster_id = 0` is always the largest cluster, with exact integer tiebreaks. Sorting whole records (rather than parallel arrays) is what prevents the classic bug where size, membership and representative fall out of sync.
5. Assign `local_cluster_id = 0,1,2,…` and build the `particle_to_local_cluster` lookup array.
6. **Conservation checks** — a particle already assigned raises "particle assigned to multiple clusters"; any unassigned particle raises "clustering did not assign every analysed particle"; and $\sum_C \lvert C\rvert \ne M$ raises "cluster populations do not conserve particles".

### 15.5 Cleanup

A `finally` block flushes and drops the memmap, then deletes the distance file unless `--keep-distance-files`. An outer `finally` in `main()` removes the whole temporary directory (`shutil.rmtree(..., ignore_errors=True)`) unless the flag was given.

---

## 16. Part C — State identification

`identified_clusters_for_single_frame(frame, cluster_size_cutoff)` is a one-line filter:

```python
return tuple(c for c in frame.clusters if c.size >= cluster_size_cutoff)
```

Because `frame.clusters` is already sorted largest-first, the surviving tuple is also largest-first, and `state_id = 0, 1, 2, …` is assigned by enumeration order. `alphabetic_state_name(state_id)` then maps $0 \to A$, $1 \to B$, …, $25 \to Z$, $26 \to AA$, and so on (bijective base-26; negative input raises `ValueError`).

Derived quantities reported by `main()`:

```python
identified_population = sum(c.size for c in identified_clusters)
ignored_population    = selected_particle_count - identified_population
max_diameter          = max(c.diameter_deg for c in frame_result.clusters)
```

> **What "identified" means, precisely.** A cluster is promoted to a named state if and only if its instantaneous particle count in this single frame is $\ge$ `cluster_size_cutoff`. There is no persistence requirement, no cross-frame confirmation, and no statistical test — none of those are available from one snapshot. Interpret an identified state as *"a group of at least this many particles whose orientations all lie within $\varepsilon_{\text{orient}}$ of one another, in this frame"* — nothing more.

---

# Outputs — the detailed reference

## 17. Output files: what each one saves and why it exists

### 17.1 The filename prefix

Every output derives from one prefix:

```
<output_dir>/<gsd_stem>_single_frame_orientation_states_frame_<frame_index>_particles_<M>
```

Example, for `traj.gsd`, frame 499, 4096 particles:

```
traj_single_frame_orientation_states_frame_499_particles_4096
```

`<output_dir>` is the trajectory's parent directory unless `--output-dir` is given; it is created with `parents=True, exist_ok=True`.

**Every** filename is built with `Path.with_name(prefix.name + suffix)` — never `with_suffix()` — so a GSD filename containing dots (`run.v2.gsd`) is handled safely and no name is truncated.

### 17.2 Format conventions

| | Symmetry CSV | The four data CSVs and the plotting CSV |
|---|---|---|
| Writer | `np.savetxt` | `csv.writer` |
| Header | bare (`comments=""`) | bare |
| Float format | `%.17g` — always shows a decimal point | Python `str(float)` — shortest representation that round-trips |
| Integer columns | become floats (`3` → `3`, but all-float array) | stay integers (`3`) |
| Missing values | n/a | empty string `""`, not `nan` |

> The four data CSVs are **more pleasant to parse than the multi-frame version's**, which used `np.savetxt(..., fmt="%.17g")` throughout and therefore rendered every integer column as a float. Here `csv.writer` receives genuine `int` and `float` objects, so `cluster_size` reads back as `1204`, not `1204.0`, while floats still round-trip losslessly (Python's `str(float)` has produced the shortest round-tripping repr since 3.1).

### 17.3 File 1 — `<prefix>_identified_states.csv` ⭐ **the headline answer**

**Purpose.** This is the primary scientific result: one row per **identified orientational state**. If you read only one output file, read this one. It answers "what are the states, and how big is each?"

**Granularity.** One row per identified state. Row count $=$ `identified_cluster_count`.
**Order.** By descending cluster size (so row 0 is state `A`, the most populous).

| # | Column | Type | Meaning and purpose |
|---|---|---|---|
| 1 | `frame_index` | int | The analysed frame. Constant down the file; present so the file is self-describing when concatenated across runs |
| 2 | `state_id` | int | $0,1,2,\ldots$ by descending size. The numeric handle used in the plotting CSV and metadata |
| 3 | `cluster_name` | str | `A`, `B`, `C`, … — the human-readable label used on the plot's x-axis |
| 4 | `local_cluster_id` | int | This cluster's ID in the **raw** clustering. **The join key to `_all_frame_clusters.csv` and the membership files** |
| 5 | `cluster_size` | int | Number of particles in the state |
| 6 | `population_fraction` | float | `cluster_size / M`. **Denominator is all analysed particles**, so these sum to $\le 1$ (§6.4) |
| 7 | `medoid_particle_index` | int | The actual particle whose orientation defines the state. Look it up in the GSD to inspect its neighbourhood |
| 8–11 | `medoid_w`, `medoid_x`, `medoid_y`, `medoid_z` | float | ⭐ **The state's defining orientation** — sign-canonicalised, scalar-first unit quaternion. This is the physically meaningful output; feed it to any downstream orientation calculation |
| 12 | `diameter_deg` | float | Exact $\operatorname{diam}(C)$. How tight the state is. Guaranteed $\le$ your tolerance |
| 13 | `max_angle_to_medoid_deg` | float | $\max_j d_G(m,j)$ — the state's "radius" about its representative. Always $\le$ `diameter_deg` |
| 14 | `medoid_sum_distance_deg` | float | $S_m$, the minimised objective. Diagnostic; scales with cluster size, so compare only within a cluster |
| 15 | `diameter_within_requested_tolerance` | int | `1` if `diameter_deg <= tolerance + 1e-10`. Should be `1` on every row — a redundant belt-and-braces flag, since a violation would already have raised during clustering |

**Typical uses.** Report the state count and populations; extract the medoid quaternions to compare against crystallographic predictions; check `diameter_deg` to confirm the states are tight rather than sprawling up against the tolerance.

### 17.4 File 2 — `<prefix>_all_frame_clusters.csv` ⭐ **the audit and tuning file**

**Purpose.** The complete, unfiltered clustering result — **including every cluster the cutoff rejected**. This is the file that lets you verify your `cluster_size_cutoff` was a sensible choice, and it is the only place where the discarded population is visible cluster-by-cluster.

**Granularity.** One row per raw cluster. Row count $=$ `raw_cluster_count`, which can be large (hundreds or thousands) if the orientation tolerance is small.
**Order.** By descending cluster size — identical ordering to `local_cluster_id`.

| # | Column | Type | Meaning and purpose |
|---|---|---|---|
| 1 | `frame_index` | int | The analysed frame |
| 2 | `local_cluster_id` | int | $0,1,2,\ldots$ by descending size. **The join key** |
| 3 | `cluster_size` | int | Particle count |
| 4 | `meets_cluster_size_cutoff` | int | `1` if `cluster_size >= cutoff`, else `0`. **Sort by this to see exactly what was kept and what was thrown away** |
| 5 | `identified_state_id` | int | The state ID if promoted, else **`-1`** |
| 6 | `cluster_name` | str | `A`/`B`/`C`… if promoted, else **empty string** |
| 7 | `medoid_particle_index` | int | Representative particle |
| 8–11 | `medoid_w`…`medoid_z` | float | Medoid quaternion — present even for rejected clusters, so you can check whether a small cluster happens to sit near a large one |
| 12 | `diameter_deg` | float | Exact diameter |
| 13 | `max_angle_to_medoid_deg` | float | Radius about the medoid |
| 14 | `medoid_sum_distance_deg` | float | $S_m$ |

> Note there is **no** `diameter_within_requested_tolerance` column here — that flag appears only in `_identified_states.csv`.

**Typical uses.**
* **Cutoff tuning (§12.4):** sort by `cluster_size` descending and look for the gap between real states and noise.
* **Checking for near-misses:** a cluster of size 195 with a cutoff of 200 is worth knowing about; only this file shows it.
* **Detecting a bad tolerance:** thousands of rows means the tolerance is far too small; one row means far too large.
* **Spotting split states:** two medium clusters with nearly identical medoid quaternions suggests a single physical state was split by a slightly-too-tight tolerance.

### 17.5 File 3 — `<prefix>_all_analysed_particle_membership.csv` ⭐ **the complete particle ledger**

**Purpose.** The finest-grained, **complete** record: every one of the $M$ analysed particles, with the cluster it landed in and whether that cluster became a state. This is the file to join against particle positions to make spatial maps.

**Granularity.** One row per analysed particle. Row count $= M$ **exactly** — this is a strong invariant worth checking (§18).
**Order.** Grouped by cluster (largest first), and within each cluster by ascending particle index.

| # | Column | Type | Meaning and purpose |
|---|---|---|---|
| 1 | `frame_index` | int | The analysed frame |
| 2 | `particle_index` | int | Zero-based index **into the GSD frame's particle array**, so it joins directly against `particles.position`, `particles.typeid`, etc. |
| 3 | `local_cluster_id` | int | Which raw cluster. Joins to `_all_frame_clusters.csv` |
| 4 | `cluster_size` | int | The size of that cluster, **denormalised onto every particle row** so you can filter by cluster size without a join |
| 5 | `meets_cluster_size_cutoff` | int | `1`/`0` |
| 6 | `identified_state_id` | int | State ID, or `-1` if the particle's cluster was below cutoff |
| 7 | `cluster_name` | str | `A`/`B`/`C`…, or empty string |

**Typical uses.**
* **Spatial mapping** — join on `particle_index` to positions and colour particles by `identified_state_id` to see whether the states are spatially segregated into domains or interleaved. *This is the single most valuable thing you can do with the output, and it is the only route to spatial information, since the analysis itself uses none.*
* **Per-type breakdown** — join to `particles.typeid` to see whether different species prefer different states.
* **Accounting for the discarded particles** — filter `identified_state_id == -1` to isolate exactly which particles the cutoff excluded and where they sit.

### 17.6 File 4 — `<prefix>_identified_particle_state_membership.csv` ⭐ **the clean downstream input**

**Purpose.** The same information as File 3, **filtered to particles that belong to an identified state, with the bookkeeping columns stripped out**. It exists so that downstream tools do not have to filter on `identified_state_id >= 0` and do not have to handle `-1` sentinels or empty strings. Every row is a real particle in a real named state.

**Granularity.** One row per particle in an identified state. Row count $=$ `identified_particle_count` $\le M$.
**Order.** Same as File 3 — grouped by state (largest first), ascending particle index within each.

| # | Column | Type | Meaning |
|---|---|---|---|
| 1 | `frame_index` | int | The analysed frame |
| 2 | `particle_index` | int | Index into the GSD frame |
| 3 | `state_id` | int | Always $\ge 0$ — **no `-1` sentinels appear in this file** |
| 4 | `cluster_name` | str | Always non-empty |

**Why both File 3 and File 4?** They serve different consumers. File 3 is the *complete, auditable* ledger — nothing is hidden, and the row count provably equals $M$. File 4 is the *convenient* form for plotting, histogramming or feeding another program, where the below-cutoff particles are noise you have already decided to exclude. Writing both costs almost nothing (they are produced in a single pass, with both file handles open simultaneously in one `with` statement) and it removes a whole class of downstream filtering mistakes.

### 17.7 File 5 — `<prefix>_symmetry_quaternions.csv` ⭐ **the provenance of the distance metric**

**Purpose.** Records the complete proper rotational symmetry group $G$ that was detected and used to define $d_G$. Every angular distance in every other output depends on this group being correct, so this file is the audit trail for the most consequential upstream decision. It is also **directly reusable**: columns 3–6 are exactly the array that was passed to freud, so another calculation can consume them without re-running detection.

**Granularity.** Two rows per physical rotation — $+q$ then $-q$. Row count $= 2n$ where $n$ = `physical_proper_rotation_count`.
**Order.** By `physical_operation_index`, which follows the deterministic sort by rotation magnitude (identity first).
**Writer.** `np.savetxt` with `fmt="%.17g"` — so all values, including the two integer columns, are written in float format.

| # | Column | Meaning |
|---|---|---|
| 1 | `physical_operation_index` | $0 \ldots n-1$; matches the printed console table |
| 2 | `quaternion_sign` | `+1` or `-1` |
| 3–6 | `w,x,y,z` | ⭐ The signed scalar-first quaternion, full double precision, **unrounded** |
| 7 | `max_vertex_residual` | Worst post-refinement vertex mismatch, in shape units. Should be $\sim 10^{-16}$; a value near $10^{-p}$ means the symmetry is marginal |
| 8 | `discovery_angle_deg` | The candidate angle that first found this permutation |
| 9–11 | `discovery_axis_x/y/z` | The canonicalised candidate axis that found it |

> **Columns 8–11 are metadata about the *discovery*, not the final operation.** The authoritative numerical definition is columns 3–6, which come from the full-precision Procrustes refit — not from the discovery axis/angle. Do not reconstruct rotations from columns 8–11.

**Typical uses.** Verify $n$ matches theory for your shape; check `max_vertex_residual` is tiny; reuse columns 3–6 as an equivalent-orientation set elsewhere.

### 17.8 File 6 — `<prefix>_single_frame_state_populations.png` — the figure

**Purpose.** A publication-ready bar chart of the identified-state populations. It is the visual summary of `_identified_states.csv`.

| Property | Value |
|---|---|
| Content | One bar per **identified** state; bars in `A, B, C, …` order (descending size) |
| Figure size | `(max(5.0, 0.65 × n_states), 3.8)` inches — widens automatically as states accumulate |
| Saved | `dpi=600, bbox_inches="tight"` |
| x | Bar centres $0,1,2,\ldots$; tick labels `A`, `B`, `C`, … (**not** rotated) |
| y | Population fraction (`cluster_size / M`) |
| y-limit | `(0, max(0.05, 1.12 × max fraction))` — the `0.05` floor keeps very small bars visible |
| Ticks | `direction="in"` on both axes |
| Error bars | **None.** A single frame provides no variance, so none is drawn — unlike the multi-frame version's σ bars |
| Legend | None (redundant with the x labels) |
| Empty case | If no cluster meets the cutoff, a centred text panel reads *"No cluster meets the single-frame size cutoff (N)."*, x-ticks are cleared and y-limits are $(0,1)$. The program does not crash |
| Display | `plt.show()` unless `--no-show`, in which case `plt.close(figure)` |

> Because the y-axis denominator is $M$, **the bars will not sum to 1** whenever any particle was below the cutoff. The missing height is the discarded population; read `ignored_below_cutoff_particle_count` from the metadata to quantify it.

### 17.9 File 7 — `for_plotting_<prefix>_single_frame_state_populations.csv` — the exact plotted coordinates

**Purpose.** The precise arrays that were handed to `axis.bar()`, written **before** the figure is drawn. It exists so the figure can be restyled, replotted in another tool, or merged with other datasets **without re-running the analysis and without recomputing anything**. There is no ambiguity about what was plotted.

**Naming.** Note the pattern: `for_plotting_` is prepended to the **PNG's stem**, so the CSV sits next to the PNG with an obviously paired name.

**Granularity.** One row per plotted bar. Row count $=$ `identified_cluster_count`. If no state qualifies, the file is **header-only** — valid CSV, zero data rows.

| # | Column | Meaning |
|---|---|---|
| 1 | `x_bar_center` | The x coordinate passed to `bar()` — $0,1,2,\ldots$ |
| 2 | `state_id` | Numeric state ID |
| 3 | `cluster_name` | The x tick label actually rendered |
| 4 | `cluster_size` | Raw particle count (convenient; not plotted directly) |
| 5 | `y_population_fraction` | The bar height |

### 17.10 File 8 — `<prefix>_metadata.json` ⭐ **the complete run record**

**Purpose.** A single machine-readable record of *everything* — every input path, every parameter, every headline diagnostic, and a compact summary of the result. Its job is provenance: given only this file you can reconstruct the exact command line, audit every scientific choice, and confirm what the run concluded. It is also the natural file to parse when scripting a parameter scan (§12.3).

Written with `json.dump(..., indent=2)`.

**Top-level keys, grouped by purpose:**

*Identification*
| Key | Purpose |
|---|---|
| `analysis_type` | Literal `"single-frame orientational-state clustering"` — distinguishes this from the multi-frame sibling's output at a glance |
| `trajectory_file`, `shape_file` | Absolute resolved input paths |
| `total_trajectory_frames` | $T$, for context on the frame choice |
| `selected_frame_index` | Which frame was analysed |
| `selected_particle_count` | $M$ |

*Self-documenting method strings* — these exist so a reader who has never seen the code still knows what was done:
| Key | Content |
|---|---|
| `frame_clustering_method` | `"complete-linkage hierarchical clustering"` |
| `frame_cluster_constraint` | `"explicitly validated maximum symmetry-reduced pairwise angular diameter <= orientation_angle_tolerance_deg"` |
| `representative_definition` | `"symmetry-aware medoid: actual member minimising sum of within-cluster pairwise angular distances; smallest particle index resolves numerical ties"` |
| `cluster_size_cutoff_definition` | `"instantaneous minimum particle count for an identified single-frame orientational state"` |

*Parameters*
`orientation_angle_tolerance_deg`, `cluster_size_cutoff`, `expected_edges`, `expected_faces`, `symmetry_precision_exponent`, `symmetry_matching_tolerance`, `block_size`, `angle_validation_tolerance_deg`

*Result counts — the numbers to scrape when scanning parameters*
| Key | Meaning |
|---|---|
| `raw_cluster_count` | Total clusters complete linkage produced |
| `identified_cluster_count` | How many passed the cutoff — **the state count** |
| `identified_particle_count` | Particles in identified states |
| `ignored_below_cutoff_particle_count` | $M -$ the above. **Check this is small** |

*Quality diagnostics*
| Key | What to look for |
|---|---|
| `maximum_quaternion_norm_deviation` | GSD-writer precision. Should be $\lesssim 10^{-6}$ |
| `minimum_pair_angle_deg` | Smallest $d_G$ over all pairs. A large value with many clusters hints the tolerance is too small |
| `maximum_pair_angle_deg` | Largest $d_G$ observed — an empirical estimate of $\theta_{\max}(G)$, and the natural ceiling for your tolerance |
| `maximum_cluster_diameter_deg` | Over **all raw** clusters. Compare against your tolerance (§12.3) |
| `physical_proper_rotation_count` | $n = \lvert G\rvert$. **Check against theory** |
| `equivalent_quaternion_count` | $2n$; confirms both signs were supplied |

*Compact result* — `identified_states[]`, one object per state with `state_id`, `cluster_name`, `local_cluster_id`, `cluster_size`, `population_fraction`, `medoid_particle_index`, `diameter_deg`. A summary for quick programmatic access; the full medoid quaternions live in `_identified_states.csv`.

### 17.11 Optional — `<prefix>_distance_files/` (only with `--keep-distance-files`)

**Purpose.** Retains the raw condensed pairwise-distance array so you can re-analyse it without recomputing the $O(M^2 n)$ freud work — for example to try several orientation tolerances, or to compute your own statistics.

**Contents.** One binary file:

```
frame_<frame_index>_condensed_pairwise_angles_float64.dat
```

A headerless `float64` array of exactly $M(M-1)/2$ values in SciPy condensed order (§15.2). Reload it with:

```python
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

M = 4096
d = np.memmap("frame_499_condensed_pairwise_angles_float64.dat",
              dtype=np.float64, mode="r", shape=(M * (M - 1) // 2,))

# Re-cluster at a different tolerance without touching freud again
labels = fcluster(linkage(d, method="complete"), t=35.0, criterion="distance")
```

Without the flag, the file is written into a `tempfile.mkdtemp(prefix="orientation_distance_work_", dir=output_directory)` directory that is removed by `shutil.rmtree` in a `finally` block.

> **Size warning:** $8 \times M(M-1)/2$ bytes — 67 MB at $M = 4096$, 1.6 GB at $M = 20000$. See §22.

---

## 18. Cross-file invariants you can check

These hold by construction and are cheap to verify. If any fails, something is wrong and you should not trust the run.

Let $M$ = `selected_particle_count`, and use the metadata for the scalars.

| Invariant | Why it must hold |
|---|---|
`rows(_all_frame_clusters.csv)` $=$ `raw_cluster_count` | One row per raw cluster |
$\sum$ `cluster_size` over `_all_frame_clusters.csv` $= M$ | Clustering is a partition; asserted in code |
`rows(_all_analysed_particle_membership.csv)` $= M$ | Every analysed particle appears exactly once |
`rows(_identified_states.csv)` $=$ `identified_cluster_count` | One row per identified state |
`rows(_identified_particle_state_membership.csv)` $=$ `identified_particle_count` | Filtered membership |
$\sum$ `cluster_size` over `_identified_states.csv` $=$ `identified_particle_count` | Same particles, two granularities |
`identified_particle_count` $+$ `ignored_below_cutoff_particle_count` $= M$ | Complete accounting |
$\sum$ `population_fraction` over `_identified_states.csv` $=$ `identified_particle_count` $/\,M \le 1$ | Denominator is $M$ (§6.4) |
`rows(for_plotting_*.csv)` $=$ `identified_cluster_count` | One bar per state |
`rows(_symmetry_quaternions.csv)` $= 2 \times$ `physical_proper_rotation_count` | Both signs emitted |
Rows in `_all_frame_clusters.csv` with `meets_cluster_size_cutoff == 1` $=$ `identified_cluster_count` | Definition of identification |
Every `diameter_deg` in every file $\le$ `orientation_angle_tolerance_deg` $+ 10^{-10}$ | The verified diameter gate |
`max_angle_to_medoid_deg` $\le$ `diameter_deg` on every row | The medoid is a member; its worst distance cannot exceed the diameter |

A quick check in pandas:

```python
import json, pandas as pd
meta = json.load(open(f"{p}_metadata.json"))
allc  = pd.read_csv(f"{p}_all_frame_clusters.csv")
allm  = pd.read_csv(f"{p}_all_analysed_particle_membership.csv")
ident = pd.read_csv(f"{p}_identified_states.csv")

M = meta["selected_particle_count"]
assert len(allc)  == meta["raw_cluster_count"]
assert allc.cluster_size.sum() == M
assert len(allm) == M
assert allm.particle_index.nunique() == M
assert len(ident) == meta["identified_cluster_count"]
assert ident.cluster_size.sum() == meta["identified_particle_count"]
assert (ident.diameter_deg <= meta["orientation_angle_tolerance_deg"] + 1e-10).all()
print("all invariants hold")
```

---

## 19. Console output reference

```
Trajectory summary
------------------
Trajectory: /abs/path/traj.gsd
Number of available frames: 500
Valid frame indices: 0 through 499
Selected frame index: 499
Particles available in selected frame: 4096

Shape vertex array selected from JSON key/path hint: 'vertices'
Number of shape vertices: 12

Shape-centre validation
-----------------------
Original arithmetic vertex centre: [...]
Norm of original centre: ...
User-selected symmetry matching tolerance: 10^(-4) = 0.0001 shape-length units
CENTRE CHECK: PASSED within the selected tolerance.
Internal centred-coordinate residual: [...]

Convex-hull topology scan
--------------------------
User-supplied expected topology: edges=25, faces=15, vertices=12
merge tolerance       recovered edges       recovered faces
 1.0e-12                          36                    24
 ...
TOPOLOGY CHECK: PASSED.  The recovered edge and face counts match the user inputs.
Accepted coplanar-face merge tolerance: 1.0e-06
Convex-hull volume from the shape file: ...

Unique candidate axis lines generated: 31

Proper rotational-symmetry search
---------------------------------
Candidate angle count: 37
Raw axis/angle candidates tested: 1147
Valid discovery hits before permutation deduplication: 46
Distinct physical proper rotations detected: 12
Largest accepted full-precision vertex residual: ...
ROTATIONAL-GROUP CHECK: PASSED (identity, inverses, and closure).
Equivalent quaternion representatives passed to freud: 24 (= 2 x physical rotations)
Q/-Q CHECK: PASSED for every physical rotational symmetry.

Detected physical proper rotations (one canonical +q each)
 index             w             x             y             z       max residual
     0   1.0000000000  0.0000000000  0.0000000000  0.0000000000   0.000e+00
   ...

Single-frame complete-linkage clustering
----------------------------------------
Frame 499: raw clusters=9, identified clusters with size >= 200=4, largest cluster=1204, maximum validated diameter=40.8712345678 deg.

Final single-frame state summary
--------------------------------
Analysed frame: 499
Analysed particles: 4096
Raw complete-linkage clusters: 9
Identified clusters with size >= 200: 4
Particles in identified states: 4026
Particles ignored below cutoff: 70
State A: size=1204, fraction=0.293945, medoid particle=317, diameter=40.8712 deg
State B: size=1180, fraction=0.288086, medoid particle=42, diameter=39.4401 deg
State C: size=1002, fraction=0.244629, medoid particle=1188, diameter=38.9017 deg
State D: size=640, fraction=0.15625, medoid particle=905, diameter=37.2213 deg

Saved outputs
-------------
Symmetry quaternions: ..._symmetry_quaternions.csv
identified_states: ..._identified_states.csv
all_clusters: ..._all_frame_clusters.csv
all_membership: ..._all_analysed_particle_membership.csv
identified_particle_states: ..._identified_particle_state_membership.csv
Single-frame population plot: ..._single_frame_state_populations.png
Single-frame plotting CSV: for_plotting_..._single_frame_state_populations.csv
Metadata: ..._metadata.json
```

The `identified_states` / `all_clusters` / `all_membership` / `identified_particle_states` labels in the *Saved outputs* block are the internal dictionary keys returned by `save_single_frame_outputs()`, printed verbatim.

---

# Reference material

## 20. Internal data structures

All six are `@dataclass(frozen=True)` — immutable, so no downstream code can mutate a validated result.

### Geometry (Part A) — identical to the multi-frame version
* **`ConvexDecomposition`** — `vertices` (centred), `edges` (sorted deduplicated index pairs), `faces` (tuple of cyclically ordered index arrays), `face_equations` ($(F,4)$, unit normals, centred coordinates), `merge_tolerance`, `volume`
* **`SymmetryOperation`** — `quaternion_wxyz` (canonical $+q$, full precision), `rotation_matrix`, `permutation` (**the deduplication key**), `max_vertex_residual`, `discovery_axis`, `discovery_angle_deg`
* **`SymmetryResult`** — `centered_vertices`, `original_center`, `decomposition`, `physical_operations`, `equivalent_quaternions_wxyz` ($(2n,4)$ — the array actually passed to freud), `matching_tolerance`

### Clustering (Part B)
* **`DistanceDiagnostics`** — `minimum_angle_deg`, `maximum_angle_deg` over all $i<j$ pairs
* **`FrameCluster`** — `frame_index`, `local_cluster_id`, `members` (int64 array), `size`, `medoid_particle_index`, `medoid_quaternion_wxyz`, `diameter_deg`, `maximum_angle_to_medoid_deg`, `medoid_sum_distance_deg`
* **`FrameClustering`** — `frame_index`, `particle_count`, `clusters` (tuple, sorted largest-first), `particle_to_local_cluster`, `maximum_quaternion_norm_deviation`, `distance_diagnostics`

> Compare with the multi-frame version, whose `FrameClustering` additionally carries `cluster_count_stability_passed`. There is no `FrameTracking` or `StateSummary` here — identified states are just a filtered tuple of `FrameCluster`, with `state_id` implied by position.

---

## 21. Complete validation and error catalogue

### Startup and CLI
| Condition | Exception |
|---|---|
| SciPy missing | `SystemExit` at import, with an install hint |
| freud / gsd / matplotlib missing | `ImportError` with an install hint |
| `--block-size < 1` | `ValueError` |
| `--angle-validation-tol <= 0` | `ValueError` |
| A CLI value fails its validator | `ValueError` carrying the prompt's error message |

### Input files
| Condition | Exception |
|---|---|
| GSD absent | `FileNotFoundError` |
| Trajectory has no frames | `ValueError` |
| Selected frame's orientations not $(N,4)$ | `ValueError` naming the frame and shape |
| Shape JSON absent | `FileNotFoundError` |
| Malformed JSON | `json.JSONDecodeError` |
| No finite $N\times3$, $N\ge4$ array found | `ValueError` |
| All vertices collapse to one point | `ValueError` |
| Duplicate / indistinguishable vertices | `ValueError` naming an example index pair |

### Geometry and symmetry
| Condition | Exception |
|---|---|
| Merged face has a zero normal | `RuntimeError` |
| Cannot build an in-plane face basis / degenerate polygon | `RuntimeError` |
| **No merge tolerance reproduces the expected topology** | `RuntimeError` with expected and final-trial counts |
| Zero axis passed to canonicalisation | `ValueError` |
| No nonzero candidate axes | `RuntimeError` |
| `precision_exponent < 0` | `ValueError` |
| Recentring residual $> 100\epsilon$ | `RuntimeError` |
| Incomplete Hungarian assignment | `RuntimeError` |
| Refined rotation has $\det < 0$ | `RuntimeError` |
| **Refinement changed the discovered permutation** | `RuntimeError` advising a tighter $p$ |
| No permutations detected / identity absent | `RuntimeError` |
| **A required inverse is missing** | `RuntimeError` — search incomplete |
| **Not closed under composition** | `RuntimeError` — search incomplete |
| Equivalent quaternions not unit-norm (`atol=1e-12`) | `RuntimeError` |
| $q/-q$ pairing not bit-exact | `RuntimeError` |

### Frame data and distances
| Condition | Exception |
|---|---|
| Orientations not $(N,4)$ | `ValueError` naming the frame |
| Fewer orientations than requested particles | `ValueError` naming both counts |
| NaN/inf in orientations | `ValueError` |
| A zero quaternion | `ValueError` |
| **freud output shape $\ne$ (block, M)** | `RuntimeError` quoting both shapes |
| freud returned NaN/inf | `RuntimeError` |
| Angle $< -\varepsilon_{\text{valid}}$ | `RuntimeError` "negative angular distance detected" |
| Angle $> 180 + \varepsilon_{\text{valid}}$ | `RuntimeError` "angular distance exceeds 180 degrees" |
| **Condensed-distance accounting mismatch** | `RuntimeError` quoting expected vs. written |
| Condensed indexing on a self-pair | `ValueError` |

### Clustering
| Condition | Exception |
|---|---|
| **Cluster diameter $>$ tolerance $+ 10^{-10}$** | `RuntimeError` quoting both to 12 s.f. |
| Particle assigned to multiple clusters | `RuntimeError` |
| Some analysed particle unassigned | `RuntimeError` |
| Cluster populations do not sum to $M$ | `RuntimeError` |
| `alphabetic_state_name(negative)` | `ValueError` |

### Notable absences relative to the multi-frame version
There is **no** reference-medoid self-angle check, no matrix-symmetry check, no tracking-overlap check, no ambiguity/split detection, and no stability warning — none of those concepts exist in a single-frame analysis. Correspondingly, **this version emits no warnings at all**: it either completes or raises.

---

## 22. Complexity, memory and disk model

Let $V$ = shape vertices, $n_{\text{tri}}$ = hull triangles, $A$ = unique candidate axes, $n$ = detected rotations, $M$ = analysed particles.

### Symmetry detection (once)

| Stage | Time | Memory |
|---|---|---|
| Convex hull | $O(V\log V)$ | $O(n_{\text{tri}})$ |
| Coplanar merge × 9 tolerances | $O(n_{\text{tri}}\log n_{\text{tri}} + n_{\text{tri}}^2)$ each | $O(n_{\text{tri}}^2)$ **dense int8 adjacency** |
| Candidate axes (pairwise dedup) | $O((V+2F+E)^2)$ | $O(A)$ |
| Candidate testing | $A\times37\times O(V^3)$ (Hungarian) | $O(V^2)$ |
| Group validation | $O(n^2V)$ | $O(nV)$ |

Typically seconds for $V \lesssim 100$. The $O(n_{\text{tri}}^2)$ dense adjacency is the only structure that becomes awkward for a shape with thousands of hull triangles.

### The single frame

| Operation | Time | Peak RAM | Disk |
|---|---|---|---|
| freud distances | $O(M^2 n)$ | $8\cdot\text{block}\cdot M$ B | — |
| Condensed memmap write | $O(M^2)$ | negligible | $8\cdot\tfrac{M(M-1)}{2}$ B |
| `linkage` (complete) | $O(M^2)$ | **$8\cdot\tfrac{M(M-1)}{2}$ B** (SciPy copies to RAM) | — |
| Cluster statistics | $\sum_C O(\lvert C\rvert^2)$ | $O(\max\lvert C\rvert)$ | — |
| Output writing | $O(M)$ | negligible | small |

> **The single most important sizing fact:** although the distance array lives on disk, `scipy.cluster.hierarchy.linkage` converts its input to a contiguous in-memory float64 array. Peak RAM therefore still includes one full condensed copy. `--block-size` does not help with this.

| $M$ | condensed pairs | disk | linkage RAM |
|---|---|---|---|
| 2 000 | 2.0 M | 16 MB | 16 MB |
| 4 096 | 8.4 M | 67 MB | 67 MB |
| 10 000 | 50 M | 400 MB | 400 MB |
| 20 000 | 200 M | 1.6 GB | 1.6 GB |
| 50 000 | 1.25 G | 10 GB | 10 GB |

**Total time** $\approx O(M^2 n)$ — a single frame, so roughly $F$ times cheaper than the multi-frame version over $F$ frames, and $(1+K)$ times cheaper again since there are no stability trials.

> **Cluster/HPC note:** the temporary directory is created **inside the output directory** (`tempfile.mkdtemp(dir=output_directory)`), *not* in `/tmp`. Point `--output-dir` at a filesystem with adequate space and I/O bandwidth — a slow network filesystem will bottleneck both the sequential memmap writes and the random-access reads during cluster statistics.

---

## 23. Determinism and reproducibility

The calculation is **fully deterministic**. Unlike the multi-frame version, there is no random number generator anywhere in this script — no permutation trials, so `--random-seed` does not exist.

Every choice is fixed:
* the frame is explicit; particles are the first $M$ by index;
* JSON traversal order is sorted; candidate axes are built in fixed order;
* symmetry operations are sorted by an explicit total key;
* clusters are sorted by $(-\text{size},\ \text{medoid index},\ \text{members})$ — a total order with exact integer tiebreaks;
* medoid ties are broken by smallest particle index;
* state IDs and `A/B/C` names follow directly from that ordering.

Re-running the same command on the same files reproduces every output byte-for-byte, modulo BLAS/LAPACK threading nondeterminism inside `linear_sum_assignment` and `align_vectors` at the $10^{-16}$ level (which cannot change a permutation or a cluster assignment except in astronomically unlikely tie cases) and freud version differences.

For a fully reproducible batch run, supply all seven scientific flags. The metadata JSON records every one, so any output traces back to its command line.

---

## 24. Relationship to the multi-frame version

This script is a deliberate, surgical reduction of `ref_frame_calc_rigorous_fixed_reference_v1p5.py`.

**Identical (byte-for-byte):** everything from `ConvexDecomposition` through `save_symmetry_outputs` — the prompt helpers, shape reading, convex decomposition, topology gate, candidate axes and angles, Hungarian vertex matching, Procrustes refinement, group validation, $\pm q$ emission, GSD opening, orientation validation, `canonicalize_quaternion_sign`, `condensed_index`, `compute_condensed_distance_memmap`, `cluster_distance_statistics`, `complete_linkage_labels`. Lines 92–1192 here map exactly onto lines 104–1204 there.

**Removed:**

| Removed item | Why it cannot exist here |
|---|---|
| `angular_distance_matrix_deg` | Only needed for medoid-vs-medoid comparisons across frames |
| `build_permuted_condensed_distances` | Only needed for stability trials |
| `_cluster_medoid_records_from_labels`, `_append_particle_order_medoid_diagnostics`, `validate_partition_order_stability` | Stability machinery |
| `build_cutoff_qualified_reference`, `reference_medoid_distance_matrix`, `tracking_cutoff_suggestion`, `globally_match_clusters_to_reference_states`, `track_all_frames_to_fixed_reference`, `summarize_fixed_reference_states` | All tracking machinery |
| `FrameTracking`, `StateSummary` dataclasses | No tracking, no multi-frame statistics |
| `DEFAULT_RANDOM_SEED`, `CURRENT_SUGGESTED_FRAMES`, `CURRENT_SUGGESTED_STABILITY_TRIALS` | No randomness, no frame window, no trials |
| `--frames`, `--reference-frame`, `--tracking-angle-tol`, `--stability-trials`, `--random-seed`, `--allow-overlapping-tracking-regions`, `--allow-ambiguous-tracking` | Corresponding features gone |

**Simplified:**
* `construct_frame_clusters()` loses the `angle_validation_tolerance_deg` and `cluster_count_stability_passed` parameters; its diameter error message is shorter (it no longer mentions the self-angle floor).
* `cluster_one_frame()` loses the cutoff, stability, seed and diagnostics-file parameters.
* `--frame` accepts **any** frame index, whereas the multi-frame version could only analyse the last $F$ consecutive frames.

**Added:**
* `identified_clusters_for_single_frame()` — the size filter.
* `alphabetic_state_name()` — a module-level public version of the multi-frame `_alphabetic_group_name`, used here for the primary state naming rather than only for stability diagnostics.
* `save_single_frame_outputs()`, `save_single_frame_population_plot()`, `save_single_frame_metadata()` — the output layer, restructured around the raw/identified split and written with `csv.writer` rather than `np.savetxt`.

**Vestigial:** `NUMERICAL_SELF_ANGLE_HARD_LIMIT_DEG` and `NUMERICAL_FLOOR_SAFETY_FACTOR` remain defined but unused; `os` and `typing.Iterable` are imported but unused.

**Which to use.** Tune parameters and explore with this one; produce time-resolved, tracked populations with `v1p5`. The parameters transfer directly, since `--edges`, `--faces`, `--precision`, `--orientation-angle-tol`, `--cluster-size-cutoff`, `--particles`, `--block-size` and `--angle-validation-tol` all mean exactly the same thing in both.

---

## 25. Limitations, caveats and gotchas

### 25.1 The symmetry tolerance is absolute, not relative
$10^{-p}$ is in shape-coordinate units. Always check it against your characteristic vertex radius. See §12.2.

### 25.2 The candidate-angle list is finite
Only $C_2, C_3, C_4, C_5, C_6, C_8, C_{10}$ rotations are reachable (the list holds multiples of 180°, 120°, 90°, 72°, 60°, 45° and 36°). A $C_7$, $C_9$ or $C_{12}$ axis is **not representable**, and the closure check will fail — correctly refusing to proceed with a subgroup rather than silently returning one. Extend `historical_candidate_angles_rad()` if your particle needs it.

### 25.3 Only convex shapes
The geometric pipeline rests entirely on `ConvexHull`. A non-convex particle's concave features are invisible, so the detected group would be that of its convex hull — potentially strictly larger than the true group, which would wrongly merge distinct orientations.

### 25.4 A single frame is a single sample
This is the defining limitation. One snapshot cannot distinguish a genuine long-lived state from a transient fluctuation, and it provides **no** uncertainty estimate: there are no error bars anywhere in the output because there is no variance to compute. Run on several frames and compare, or use the multi-frame version.

### 25.5 State names are not comparable across runs
`A`, `B`, `C` are assigned by descending size **within one run**. State `A` in frame 400 need not be the same physical orientation as state `A` in frame 499. To compare across frames, match the medoid quaternions from `_identified_states.csv` yourself — or use the multi-frame version, which does exactly that with a Hungarian assignment against a fixed reference.

### 25.6 No positional information
No cutoff, no box, no periodic images, no type filtering. A global census. It cannot distinguish two spatially separated domains that share an orientation. Join `_all_analysed_particle_membership.csv` against GSD positions yourself for anything spatial (§17.5).

### 25.7 Particle selection is by index
The **first $M$** particles are used. Deterministic, but biased if index order correlates with structure. See §12.7.

### 25.8 Complete linkage is greedy
It guarantees the diameter bound but not a globally optimal partition, and different tolerances can produce qualitatively different partitions. Use the plateau scan (§12.3). Note that this version has **no** stability check, so it cannot warn you when the partition is sitting on a knife edge where distance ties matter — that diagnostic exists only in the multi-frame version.

### 25.9 Population fractions do not sum to 1
The denominator is $M$, not the identified population. The deficit is the below-cutoff population. See §6.4 and §17.8.

### 25.10 Peak RAM is set by `linkage`, not by `--block-size`
See §22. At large $M$ the in-memory condensed copy dominates and no flag mitigates it; reduce `--particles`.

### 25.11 `plt.show()` blocks by default
On a headless node without `--no-show` and without a suitable `MPLBACKEND`, the run may hang at the very end — *after* all files have been written. Always pass `--no-show` in batch jobs.

### 25.12 `axis.set_xticks(ticks, labels)` needs matplotlib ≥ 3.5
The two-argument form is used in the plotting routine. Older matplotlib raises a `TypeError`, which is **not** in the caught exception list and will surface as a traceback — after the CSVs have been written but before the PNG.

### 25.13 `_all_analysed_particle_membership.csv` grows with $M$
Exactly $M$ data rows — modest for one frame (4096 rows at $M=4096$), but 50 000 rows at $M = 50\,000$. There is no option to suppress it.

### 25.14 The trajectory handle is never explicitly closed
Released at process exit. Fine for a script, not for library reuse.

### 25.15 The temporary directory lives in the output directory
Not `/tmp`. Plan disk space and I/O bandwidth accordingly (§22).

### 25.16 Vestigial constants may mislead
`NUMERICAL_SELF_ANGLE_HARD_LIMIT_DEG` and `NUMERICAL_FLOOR_SAFETY_FACTOR` are defined but never used. No self-angle validation is performed in this version.

### 25.17 freud API assumption
The argument order and output shape of `AngularSeparationGlobal.compute` are checked but assumed to follow the documented `(N_orientations, N_global_orientations)` convention. A future freud release that changes this triggers the explicit `RuntimeError` rather than a wrong answer — the intended behaviour, but the call would then need updating.

---

## 26. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `TOPOLOGY CHECK: FAILED` | Wrong `--edges`/`--faces`, or a non-convex shape | Read the printed scan table and use the numbers it reports; check Euler's formula |
| `not closed under composition` / `missing an inverse` | $p$ too tight, or an unreachable rotation order | Loosen $p$ first; if the count is still short, extend the angle list (§25.2) |
| `refinement changed the discovered vertex permutation` | $p$ too loose | Increase $p$ |
| Only the identity rotation detected | $p$ far too tight, noisy coordinates, or genuinely no symmetry | Scan $p$ and find the plateau (§12.2) |
| **Thousands of raw clusters, 0 identified** | Orientation tolerance far too small | Increase it; use `maximum_pair_angle_deg` for scale |
| **1 raw cluster** | Orientation tolerance too large | Decrease it toward the histogram minimum (§12.3) |
| **Many raw clusters but 0 identified, and sizes look reasonable** | Cutoff too large | Inspect `_all_frame_clusters.csv` sorted by size; put the cutoff in the gap (§12.4) |
| **Large `ignored_below_cutoff_particle_count`** | Cutoff too high, or tolerance too small so real states fragmented | Check both; the cluster-size distribution in `_all_frame_clusters.csv` tells you which |
| `cluster generated by complete linkage has diameter ... exceeding` | Should be impossible; a SciPy/numerical anomaly | Report it; check the SciPy version and that all distances are finite |
| `negative angular distance detected` / `exceeds 180 degrees` | freud numerical noise beyond the validation tolerance | Raise `--angle-validation-tol` slightly, keeping it far below any physical angle; check the freud build |
| `MemoryError` during clustering | $M$ too large for RAM (§22) | Reduce `--particles`; `--block-size` will **not** help |
| Runs out of disk | The condensed distance file | Reduce `--particles`, or point `--output-dir` at a bigger filesystem |
| `TypeError` in `set_xticks` at the very end | matplotlib < 3.5 | Upgrade matplotlib, or edit the call to the two-step `set_xticks` / `set_xticklabels` form |
| Hangs at the end on a cluster | `plt.show()` with no display | Pass `--no-show` |
| Batch job hangs at the start | A scientific flag was omitted, so it waits at a prompt | Supply all seven scientific flags |
| Bars in the plot do not sum to 1 | Correct behaviour — denominator is $M$ | Read `ignored_below_cutoff_particle_count` (§6.4) |
| State `A` differs between two runs on different frames | Correct behaviour — names are per-run | Match medoid quaternions, or use the multi-frame version (§25.5) |
| `ERROR: <message>`, exit 1 | Controlled validation failure | Read the message; every one names the failing quantity |

---

## 27. Function-by-function index

| Function | Line | Role |
|---|---|---|
| `prompt_value` | 131 | Loop until valid terminal input; empty line = default |
| `resolve_or_prompt` | 157 | CLI value if supplied (validated, hard-fail), else prompt |
| `_as_vertex_array` | 175 | Test for a finite $(N\ge4,3)$ float array |
| `_collect_vertex_candidates` | 193 | Deterministic recursive JSON search with key-name priority |
| `read_shape_vertices` | 229 | Load, select, validate, duplicate-check the vertices |
| `_merge_coplanar_hull_triangles` | 271 | Qhull → coplanar merge → polygonal faces, unit planes, edge set |
| `find_validated_convex_decomposition` | 361 | Tolerance ladder $10^{-12}\ldots10^{-4}$; hard topology gate |
| `_canonical_axis_line` | 429 | Unit-normalise and fix the $\pm$ sign of an axis line |
| `build_candidate_axis_lines` | 449 | Four axis classes; zero-removal; parallel/antiparallel dedup |
| `historical_candidate_angles_rad` | 539 | The fixed 37-angle list, in radians |
| `_rotation_to_wxyz` | 557 | SciPy `[x,y,z,w]` → freud `[w,x,y,z]`, normalised, sign-canonical |
| `_one_to_one_vertex_mapping` | 574 | Distance matrix + Hungarian assignment + residuals |
| `_refine_rotation_for_permutation` | 619 | Least-squares proper rotation via `align_vectors`, $\det>0$ guard |
| `_compose_permutations` | 657 | $(\sigma_2\circ\sigma_1)[i] = \sigma_2[\sigma_1[i]]$ |
| `validate_permutation_group` | 675 | Exact identity / inverse / closure checks |
| `detect_proper_rotational_symmetries` | 739 | The 13-stage symmetry pipeline → `SymmetryResult` |
| `import_runtime_packages` | 1060 | Lazy freud / gsd.hoomd / matplotlib import |
| `open_gsd_trajectory` | 1076 | Path resolution + keyword/positional `open` fallback |
| `validate_and_normalize_orientations` | 1092 | Shape/finiteness/norm checks, first-$M$ slice, renormalisation |
| `save_symmetry_outputs` | 1150 | **File 5** — $\pm q$ symmetry CSV |
| `canonicalize_quaternion_sign` | 1222 | Normalise and fix the sign of a quaternion |
| `condensed_index` | 1236 | Vectorised $(i,j) \to$ SciPy condensed index |
| `compute_condensed_distance_memmap` | 1256 | Block-wise freud → on-disk condensed array, range checks, accounting |
| `cluster_distance_statistics` | 1463 | Medoid, diameter, max-to-medoid, medoid sum, in one pass |
| `complete_linkage_labels` | 1534 | `linkage(method="complete")` + `fcluster(criterion="distance")` |
| `construct_frame_clusters` | 1589 | Diameter gate, deterministic ordering, conservation checks |
| `cluster_one_frame` | 1694 | Orchestrates load → distances → linkage → construct → cleanup |
| `identified_clusters_for_single_frame` | 1754 | The size-cutoff filter that defines "identified state" |
| `save_single_frame_outputs` | 1767 | **Files 1–4** — the four data CSVs |
| `alphabetic_state_name` | 1952 | 0→A, 25→Z, 26→AA |
| `save_single_frame_population_plot` | 1965 | **Files 6–7** — the PNG plus its exact plotting CSV |
| `save_single_frame_metadata` | 2049 | **File 8** — the run-record JSON |
| `build_argument_parser` | 2154 | The CLI definition |
| `main` | 2186 | The 20-step workflow inside one error boundary |

---

## Citation and provenance note

If you publish results from this program, record in your methods section:

* that the analysis is a **single-frame** census, and **which frame** (`selected_frame_index` of `total_trajectory_frames`);
* the **detected proper rotation group order** $n$ and that it is stable under variation of $p$ (`physical_proper_rotation_count`, `symmetry_precision_exponent`);
* the **orientation angle tolerance**, how it was chosen, and the **width of the plateau** over which the identified state count is unchanged;
* the **cluster size cutoff** and the resulting `ignored_below_cutoff_particle_count`;
* $M$ (`selected_particle_count`), and that particles were taken as the first $M$ by index;
* the **verified diameter guarantee** — that every reported state satisfies $\max_{i,j} d_G \le \varepsilon_{\text{orient}}$;
* that populations are fractions of $M$, so identified fractions sum to $\le 1$;
* if you compared several frames, say so explicitly, and state that state labels are not tracked between them.

Every one of these is recorded verbatim in `_metadata.json`.
