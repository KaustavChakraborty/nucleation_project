# Fixed-Reference Orientational-State Clustering and Tracking

**Script:** `ref_frame_calc_rigorous_fixed_reference_v1p5.py`
**Version:** 1.5
**Language:** Python 3.9+ (`from __future__ import annotations` makes PEP 604/585 annotations safe on 3.9)
**Size:** 4112 lines, single self-contained file
**Entry point:** `main()` via `raise SystemExit(main())`

---

## Table of Contents

**Understanding the problem**
1. [The physical question](#1-the-physical-question)
2. [Why this is hard](#2-why-this-is-hard)
3. [The solution strategy in six ideas](#3-the-solution-strategy-in-six-ideas)
4. [Mathematical definitions](#4-mathematical-definitions)

**Using the program**
5. [Installation](#5-installation)
6. [Quick start](#6-quick-start)
7. [Inputs](#7-inputs)
8. [Complete CLI reference](#8-complete-cli-reference)
9. [Interactive prompts, in order](#9-interactive-prompts-in-order)
10. [**Tuning guide: applying this to a new system**](#10-tuning-guide-applying-this-to-a-new-system)

**How it executes**
11. [Top-level execution map](#11-top-level-execution-map)
12. [Part A — Shape processing and symmetry group](#12-part-a--shape-processing-and-symmetry-group)
13. [Part B — Per-frame clustering](#13-part-b--per-frame-clustering)
14. [Part C — Reference selection and tracking](#14-part-c--reference-selection-and-tracking)
15. [Part D — Aggregation into state statistics](#15-part-d--aggregation-into-state-statistics)
16. [Part E — Outputs](#16-part-e--outputs)

**Reference material**
17. [Output file and column reference](#17-output-file-and-column-reference)
18. [Console output reference](#18-console-output-reference)
19. [Internal data structures](#19-internal-data-structures)
20. [Complete validation and error catalogue](#20-complete-validation-and-error-catalogue)
21. [Complexity, memory and disk model](#21-complexity-memory-and-disk-model)
22. [Determinism and reproducibility](#22-determinism-and-reproducibility)
23. [Limitations, caveats and gotchas](#23-limitations-caveats-and-gotchas)
24. [Troubleshooting](#24-troubleshooting)
25. [Function-by-function index](#25-function-by-function-index)

---

# Understanding the problem

## 1. The physical question

In a simulation of anisotropic (non-spherical) particles — hard polyhedra, patchy colloids, liquid crystals of faceted bodies — a dense phase often develops **orientational order**. The particles do not point in arbitrary directions; instead they settle into a small number of preferred orientations. In a plastic/rotator crystal there may be effectively one broad state; in an orientationally ordered crystal there are typically a handful of sharp states related by the lattice symmetry; in a fluid there are none.

The questions this program answers are:

1. **How many distinct orientational states exist** at a given instant?
2. **What are those states**, expressed as concrete orientations (quaternions)?
3. **How is the population distributed** among them — what fraction of particles sits in each state?
4. **Are those states persistent in time?** Do the same states survive frame after frame, or do they appear, vanish, split and merge?
5. **How stable are the answers** with respect to arbitrary numerical choices, such as the order in which particles happen to be stored in the file?

The program answers all five with a **fixed-reference** design: one frame is designated the reference, its states are frozen as the definitive list, and every other frame's states are matched back onto that fixed list. This gives a consistent state labelling across the whole trajectory, so that "state 3" means the same physical orientation in every frame and populations can be averaged meaningfully.

**Crucially, no positions are used.** This is a purely orientational, global analysis: there is no neighbour list, no cutoff radius, no simulation box, no periodic images. Two particles on opposite sides of the box are compared exactly as readily as two neighbours. The output is a global orientational-state census, not a spatially resolved correlation function.

---

## 2. Why this is hard

Four difficulties make a naive implementation wrong or intractable.

### 2.1 Particle symmetry makes "same orientation" ambiguous

A cube rotated 90° about a face axis is **indistinguishable** from the original cube. So two particles whose quaternions differ by 90° about that axis are physically in the *same* orientation. Any comparison of orientations must therefore quotient out the particle's own rotational symmetry group $G$. Getting $G$ wrong — missing operations, or including improper ones — silently corrupts every distance in the analysis.

### 2.2 Unit quaternions double-cover rotations

$q$ and $-q$ describe the identical physical rotation. HOOMD's stored sign is arbitrary. Any distance function must be invariant to it.

### 2.3 The pairwise distance matrix is enormous

Clustering $N$ particles requires all $N(N-1)/2$ pairwise distances. For $N = 4096$ that is 8.4 million values (67 MB in float64); for $N = 20000$ it is 200 million (1.6 GB). Holding a square $N \times N$ matrix would be twice as bad again, and the legacy approach of copying it into Python lists worse still.

### 2.4 Clusters must be *matched* across frames, not merely counted

Hierarchical clustering returns arbitrary integer labels. Cluster "2" in frame 100 has no relation to cluster "2" in frame 101. Establishing correspondence requires (a) a representative orientation per cluster and (b) a matching rule that is one-to-one, so no reference state receives two clusters and no cluster is claimed by two states.

---

## 3. The solution strategy in six ideas

### Idea 1 — Reconstruct the symmetry group from the shape itself, and *prove* it is a group

Rather than trusting a hard-coded symmetry list, the program derives the proper rotation group $G$ directly from the shape JSON: convex hull → candidate axes from vertices/faces/edges → candidate rotations → accept only those that map the vertex set onto itself one-to-one → refine at full precision → **verify identity, inverses and closure exactly**. If closure fails, the program stops rather than proceeding with an incomplete group.

### Idea 2 — Define distance as the symmetry-reduced misorientation angle

$$d_G(q_i, q_j) = \min_{g \in G} \theta\bigl(q_i^{-1} q_j g\bigr) \in [0°, \theta_{\max}(G)]$$

computed by `freud.environment.AngularSeparationGlobal`, with the equivalent-orientation set containing **both** $+q$ and $-q$ for every $g \in G$.

### Idea 3 — Cluster with complete linkage, then *verify* the diameter

Complete-linkage (farthest-neighbour) hierarchical clustering cut at height $\varepsilon_{\text{orient}}$ is *intended* to produce clusters whose maximum internal pairwise distance is $\le \varepsilon_{\text{orient}}$. The program does not trust this: it **recomputes the exact diameter of every returned cluster** and raises an error if any exceeds the requested tolerance. So every cluster in the output carries a hard, verified guarantee:

$$\max_{i,j \in C} d_G(q_i, q_j) \le \varepsilon_{\text{orient}}.$$

This is why complete linkage is used rather than single or average linkage — only complete linkage has a diameter interpretation.

### Idea 4 — Represent each cluster by a symmetry-aware *medoid*

A cluster's representative must itself be a valid orientation. Averaging quaternions is ill-defined under symmetry (there is no unique mean of a set of symmetry-equivalent rotations). Instead the program uses the **medoid**: the actual cluster member minimising the sum of within-cluster distances,

$$m(C) = \arg\min_{i \in C} \sum_{j \in C} d_G(q_i, q_j).$$

The medoid is a real particle's real orientation, requires no averaging, and is robust to outliers.

### Idea 5 — Track by global one-to-one assignment against a frozen reference

One frame is the reference. Its clusters (above a size cutoff) become the numbered fixed states. In every other frame, the eligible clusters' medoids are compared with the reference medoids, and the resulting cost matrix is solved by the **Hungarian algorithm** with explicit "leave unmatched" dummy nodes. This guarantees a global optimum and strict one-to-one correspondence, and it lets clusters go unmatched (state appeared) and states go absent (state vanished) rather than forcing bad matches.

### Idea 6 — Test the answer against arbitrary numerical choices

Hierarchical clustering can depend on the order in which points are supplied when distances tie. The program therefore reshuffles the particle order, re-clusters, and checks whether the **number of clusters above the size cutoff** is unchanged. Failure prints a warning and writes a diagnostic CSV; it does not stop the run, because the check is advisory rather than a correctness condition.

---

## 4. Mathematical definitions

### 4.1 Orientation and misorientation

HOOMD stores each particle's orientation as a **scalar-first** unit quaternion $q = (w,x,y,z)$. The relative rotation between particles $i$ and $j$ is $\Delta_{ij} = q_i^{-1} q_j$ and its rotation angle is

$$\theta(\Delta) = 2\arccos\bigl(\lvert \operatorname{Re}\Delta \rvert\bigr) \in [0°, 180°].$$

### 4.2 Symmetry-reduced distance

With $G$ the proper rotation group of the body,

$$d_G(q_i, q_j) = \min_{g \in G} \theta\bigl(q_i^{-1} q_j\, g\bigr).$$

Properties that matter here:

* **Symmetric.** $d_G(a,b) = d_G(b,a)$ because $G$ is closed under inverses — which the program verifies.
* **Zero iff equivalent.** $d_G(q,q) = 0$ exactly, in theory. In practice freud may return a small nonzero floor (see §4.5).
* **Bounded by $\theta_{\max}(G) \le 180°$**, shrinking as $|G|$ grows: $\approx 62.8°$ for the cube group $O$ ($|G|=24$), $\approx 75.5°$ for the tetrahedral group $T$ ($|G|=12$), $180°$ for the trivial group.
* **Satisfies the triangle inequality**, so it is a genuine metric on the quotient $SO(3)/G$ — which is what licenses hierarchical clustering.

### 4.3 Cluster diameter and the complete-linkage criterion

For a cluster $C$, its **diameter** is

$$\operatorname{diam}(C) = \max_{i,j \in C} d_G(q_i, q_j).$$

Complete linkage merges clusters $A, B$ at height $D(A,B) = \max_{i\in A, j\in B} d_G(i,j)$. Cutting the dendrogram at height $t$ yields clusters with $\operatorname{diam} \le t$. The program cuts at $t = \varepsilon_{\text{orient}}$ and then re-verifies each diameter directly.

### 4.4 Medoid

$$m(C) = \arg\min_{i \in C} \; S_i, \qquad S_i = \sum_{j \in C} d_G(q_i,q_j),$$

with ties (within `MEDOID_TIE_TOL_DEG` = $10^{-10}$ degrees) broken by the **smallest original particle index**, making the choice fully deterministic. Singleton clusters are their own medoid with $S = 0$.

### 4.5 Numerical floors and the several separate tolerances

The program keeps these conceptually distinct tolerances rigorously apart — confusing them is the classic source of subtly wrong results.

| Constant / parameter | Default | Kind | What it does |
|---|---|---|---|
| `orientation_angle_tol` | 41.0° | **Physical** | Maximum permitted cluster diameter. The main scientific knob |
| `tracking_angle_tol` | auto | **Physical** | Maximum medoid–medoid angle for matching a cluster to a reference state |
| `--angle-validation-tol` | $10^{-5}$° | **Numerical** | Slack when checking that angles lie in $[0°,180°]$ and that matrices are symmetric |
| `NUMERICAL_SELF_ANGLE_HARD_LIMIT_DEG` | 0.5° | **Numerical** | Hard ceiling on the observed $d_G(q,q)$ floor. Above this, freud output is deemed untrustworthy and the run aborts |
| `NUMERICAL_FLOOR_SAFETY_FACTOR` | 4.0 | **Numerical** | The measured self-angle floor is multiplied by this when validating matrix symmetry |
| `CLUSTER_DIAMETER_EPS_DEG` | $10^{-10}$° | **Numerical** | Binary-roundoff slack when checking $\operatorname{diam}(C) \le \varepsilon_{\text{orient}}$. Deliberately far too small to relax the physical criterion |
| `MEDOID_TIE_TOL_DEG` | $10^{-10}$° | **Numerical** | Tie window for medoid selection only |

> **Design principle, stated explicitly in the source:** the freud self-angle numerical floor is **never** allowed to relax the user's physical cluster-diameter criterion. Some freud builds compute internally in single precision, so $d_G(q,q)$ can come back as, say, $0.02°$ instead of $0$. The program *measures* that floor on the reference medoid matrix, uses it only to set the tolerance for the symmetry check of that matrix, sets the diagonal to exact zero, and keeps `CLUSTER_DIAMETER_EPS_DEG` at $10^{-10}$ for the physical test.

### 4.6 Population statistics

For state $s$ over $F$ selected frames, with $n_{f,s}$ the particle count assigned to $s$ in frame $f$ (**zero when the state is absent**):

$$\langle n_s \rangle = \frac1F\sum_f n_{f,s}, \qquad \sigma_{n_s} = \sqrt{\tfrac1F\sum_f (n_{f,s} - \langle n_s\rangle)^2}$$

$$\text{fraction: } \phi_{f,s} = n_{f,s}/M, \qquad \text{presence: } p_s = \frac{1}{F}\bigl\lvert\{f : n_{f,s} > 0\}\bigr\rvert$$

A state is **valid** if $\langle n_s \rangle \ge$ `cluster_size_cutoff`. Standard deviations use `ddof=0` (population, not sample) because the selected frames are treated as the complete set being summarised.

---

# Using the program

## 5. Installation

```bash
pip install numpy scipy matplotlib gsd freud-analysis
```

**Two-tier import strategy:**

| Tier | Packages | When | Why |
|---|---|---|---|
| Startup | `numpy`, `scipy` (`cluster.hierarchy.fcluster/linkage`, `optimize.linear_sum_assignment`, `sparse.csgraph.connected_components`, `spatial.ConvexHull/cKDTree/distance_matrix`, `spatial.transform.Rotation`) | Module import, wrapped in `try/except ImportError` → `SystemExit` with an install hint | Drive the geometry, clustering and assignment core |
| Runtime | `freud`, `gsd.hoomd`, `matplotlib.pyplot` | Lazily inside `import_runtime_packages()` at Stage 2 | Lets the symmetry code be imported and tested without freud or GSD present |

Standard library used: `argparse`, `csv`, `json`, `math`, `os`, `shutil`, `sys`, `tempfile`, `dataclasses`, `pathlib`, `typing`.

---

## 6. Quick start

**Fully interactive** (the program prompts for everything):

```bash
python ref_frame_calc_rigorous_fixed_reference_v1p5.py trajectory.gsd shape.json
```

**Fully non-interactive** (batch/cluster safe):

```bash
python ref_frame_calc_rigorous_fixed_reference_v1p5.py trajectory.gsd shape.json \
  --frames 50 --particles 4096 --reference-frame 499 \
  --edges 25 --faces 15 --precision 4 \
  --orientation-angle-tol 41.0 --tracking-angle-tol 15.0 \
  --cluster-size-cutoff 200 --stability-trials 3 \
  --block-size 256 --output-dir ./states --no-show
```

> To run non-interactively you must supply **all ten** scientific flags: `--frames`, `--particles`, `--reference-frame`, `--edges`, `--faces`, `--precision`, `--orientation-angle-tol`, `--cluster-size-cutoff`, `--stability-trials`, `--tracking-angle-tol`. Omitting any one drops the program into an interactive prompt, which will hang a batch job.

---

## 7. Inputs

### 7.1 GSD trajectory (positional argument 1)

A HOOMD-blue GSD file. The program reads **only** `frame.particles.orientation`. Positions, box, types, diameters and all other fields are ignored. Consequences:

* Particle **type is ignored** — in a multi-component system, all types are pooled into one orientational analysis.
* There is **no spatial information at all** — no cutoff, no box, no periodic images.
* Opened read-only via `gsd.hoomd.open(name=..., mode="r")`, with a positional-argument fallback for older GSD releases that reject keyword arguments.
* Frames are accessed by index, so random access is required (standard for GSD).

### 7.2 Shape JSON (positional argument 2)

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

Requirements enforced:

* shape $(N,3)$ with $N \ge 4$, all entries finite;
* **no duplicate or numerically indistinguishable vertices** (KD-tree check at $\max(10^{-12}R,\ 10^{-14})$, where $R$ is the maximum centred vertex radius) — duplicates make the convex topology and symmetry permutation ambiguous;
* not all collapsed to one point.

**Units are arbitrary, but the symmetry matching tolerance $10^{-p}$ is absolute in those same units.** See §10.2.

---

## 8. Complete CLI reference

```
python ref_frame_calc_rigorous_fixed_reference_v1p5.py TRAJECTORY.gsd SHAPE.json [options]
```

### Scientific parameters (prompt if omitted)

| Flag | Type | Interactive default | Validator | Meaning |
|---|---|---|---|---|
| `--frames` | int | `min(1, T)` | $1 \le v \le T$ | Number of **consecutive final** frames to analyse |
| `--particles` | int | min available | $2 \le v \le$ min available | Particles taken from the **start** of each frame |
| `--reference-frame` | int | last selected frame | must be in the selected set | Which frame defines the fixed states |
| `--edges` | int | `25` | $v \ge 1$ | Expected polyhedron edge count |
| `--faces` | int | `15` | $v \ge 1$ | Expected polyhedron face count |
| `--precision` | int | `2` | $v \ge 0$ | $p$ in symmetry vertex-matching tolerance $10^{-p}$ |
| `--orientation-angle-tol` | float | `41.0` | $0 < v \le 180$ | **Maximum cluster diameter, degrees** |
| `--cluster-size-cutoff` | int | `min(200, M)` | $1 \le v \le M$ | Shared minimum-population threshold (three roles — see §10.4) |
| `--stability-trials` | int | `1` | $v \ge 0$ | Random particle-order permutations per frame; `0` disables |
| `--tracking-angle-tol` | float | auto-computed | $0 < v \le 180$ | **Maximum medoid angle for a match.** Prompted *after* clustering |

### Numerical / infrastructure parameters (never prompted)

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--block-size` | int | `128` | Query orientations per freud call. Memory knob; does not change results |
| `--random-seed` | int | `1729` | Seed for the stability permutations |
| `--angle-validation-tol` | float | `1e-5` | Numerical slack for angle-range and matrix-symmetry checks (degrees) |
| `--output-dir` | str | trajectory's parent | Output directory (created with `parents=True, exist_ok=True`) |

### Behaviour flags

| Flag | Effect |
|---|---|
| `--no-show` | Save PNGs without opening interactive matplotlib windows. **Always use in batch jobs** |
| `--keep-distance-files` | Retain the per-frame condensed distance memmaps in `<prefix>_distance_files/` instead of deleting them |
| `--allow-overlapping-tracking-regions` | Permit `tracking_angle_tol` $\ge$ half the minimum reference-medoid separation. Strict mode (default) rejects this |
| `--allow-ambiguous-tracking` | Permit a cluster with multiple candidate states, or a state with multiple candidate clusters. Strict mode (default) raises an error |

### Exit codes

| Code | Meaning |
|---|---|
| `0` | Completed; all outputs written |
| `1` | Controlled error, printed to stderr as `ERROR: <message>` |

Caught exception classes: `FileNotFoundError`, `ImportError`, `ValueError`, `RuntimeError`, `OSError`, `json.JSONDecodeError`. Anything else propagates with a traceback.

### Module-level constants (edit the top of the file to change)

```python
CURRENT_SUGGESTED_PRECISION            = 2
CURRENT_SUGGESTED_NUM_EDGES            = 25
CURRENT_SUGGESTED_NUM_FACES            = 15
CURRENT_SUGGESTED_FRAMES               = 1
CURRENT_SUGGESTED_ORIENTATION_TOL_DEG  = 41.0
CURRENT_SUGGESTED_CLUSTER_SIZE_CUTOFF  = 200
CURRENT_SUGGESTED_STABILITY_TRIALS     = 1
DEFAULT_BLOCK_SIZE                     = 128
DEFAULT_ANGLE_VALIDATION_TOL_DEG       = 1.0e-5
NUMERICAL_SELF_ANGLE_HARD_LIMIT_DEG    = 0.5
NUMERICAL_FLOOR_SAFETY_FACTOR          = 4.0
CLUSTER_DIAMETER_EPS_DEG               = 1.0e-10
MEDOID_TIE_TOL_DEG                     = 1.0e-10
DEFAULT_RANDOM_SEED                    = 1729
```

> The suggested defaults `edges=25, faces=15` imply, via Euler's formula $V - E + F = 2$, a polyhedron with $V = 12$ vertices. The program does not check Euler's formula itself, but it is a good sanity check on the numbers you type.

---

## 9. Interactive prompts, in order

`prompt_value()` loops until valid input arrives. **An empty line accepts the bracketed default.** Bad typing re-prompts rather than crashing. `resolve_or_prompt()` skips the prompt entirely when the corresponding CLI flag was supplied — but then a validator failure raises `ValueError` immediately, with no fallback prompt.

**Note the split:** the first nine prompts appear before any heavy computation. The tenth — the tracking tolerance — appears **after** all frames have been clustered, because its suggested value depends on how far apart the reference medoids turned out to be.

| # | Prompt | Default | Validator |
|---|---|---|---|
| 1 | How many consecutive final frames should be analysed? | `min(1, T)` | $1 \le v \le T$ |
| 2 | How many particles should be analysed in every selected frame? | min available | $2 \le v \le$ min |
| 3 | Which selected frame should define the fixed reference states? | last selected | in selected set |
| 4 | Enter the intended number of polyhedron edges | `25` | $v \ge 1$ |
| 5 | Enter the intended number of polyhedron faces | `15` | $v \ge 1$ |
| 6 | Enter invariant-quaternion symmetry precision exponent p | `2` | $v \ge 0$ |
| 7 | Maximum allowed pairwise angle within a frame-level group (degrees) | `41.0` | $0 < v \le 180$ |
| 8 | Minimum cluster population for stability, tracking, and mean-state validity | `min(200, M)` | $1 \le v \le M$ |
| 9 | How many random particle-order stability trials per frame? | `1` | $v \ge 0$ |
| — | *…symmetry detection runs, then every frame is clustered…* | | |
| 10 | Maximum medoid angle for matching a frame group to a fixed state | auto | $0 < v \le 180$ |

Immediately before prompt 10 the program prints the minimum reference-medoid separation and the strict safe upper bound, so you can make an informed choice:

```
Fixed-reference tracking tolerance
----------------------------------
Minimum reference-medoid separation: 58.31 deg
To guarantee non-overlapping reference acceptance regions, use tracking tolerance strictly below 29.155 deg.
```

---

## 10. Tuning guide: applying this to a new system

This section is the practical heart of the document. Work through it in order.

### 10.0 Recipe at a glance

```
Step 1  Determine V, E, F of your particle          → --edges, --faces
Step 2  Find a symmetry precision p that is stable  → --precision
Step 3  Choose the orientation tolerance            → --orientation-angle-tol   ← the key physical knob
Step 4  Choose the size cutoff                      → --cluster-size-cutoff
Step 5  Choose the reference frame                  → --reference-frame
Step 6  Accept or override the tracking tolerance   → --tracking-angle-tol
Step 7  Set stability trials                        → --stability-trials
Step 8  Size the block for your memory budget       → --block-size
Step 9  Scale up frames and particles               → --frames, --particles
```

### 10.1 Edges and faces (`--edges`, `--faces`)

These are validation targets, not free parameters — they must equal the true topology of your convex polyhedron.

**How to get them:**

```python
import json, numpy as np
from scipy.spatial import ConvexHull
v = np.array(json.load(open("shape.json"))["vertices"], float)
v -= v.mean(axis=0)
h = ConvexHull(v)
print("vertices:", len(v), "hull triangles:", len(h.simplices))
```

That gives $V$ and the *triangulated* face count. To get the merged polygonal $E$ and $F$, simply run the program once with any guess: the **topology scan table is printed whether or not it matches**, so you can read the correct numbers straight off it and re-run.

```
merge tolerance       recovered edges       recovered faces
 1.0e-12                          36                    24
 ...
 1.0e-06                          25                    15    ← use these
```

Cross-check with Euler's formula: $V - E + F = 2$.

**If no tolerance ever reproduces a sensible topology,** your shape is probably non-convex, has near-coplanar faces that never merge cleanly, or has duplicate vertices.

### 10.2 Symmetry precision (`--precision`, i.e. $p$)

$\varepsilon_{\text{match}} = 10^{-p}$ is an **absolute Euclidean distance in shape-coordinate units**, not a relative tolerance. This is the single most misunderstood parameter.

* Vertices of magnitude $\sim 1$: $p=2$ means $\varepsilon = 0.01$, a 1% tolerance — reasonable.
* Vertices of magnitude $\sim 0.1$: $p=2$ is a **10% tolerance** — far too loose; it may accept near-symmetries that are not symmetries.
* Vertices of magnitude $\sim 100$: $p=2$ is $10^{-4}$ relative — possibly too tight to survive the finite precision of the JSON file.

**Procedure: scan $p$ and look for a plateau.** Run with `--precision 1,2,3,4,5,6` and record `Distinct physical proper rotations detected`:

| p | detected rotations |
|---|---|
| 1 | 48  ← too loose, spurious operations |
| 2 | 24 |
| 3 | 24 |
| 4 | 24  ← plateau: this is the true group |
| 5 | 24 |
| 6 | 12  ← too tight, real operations rejected |

Use a value in the middle of the plateau. Sanity-check the count against theory: tetrahedron 12, cube/octahedron 24, dodecahedron/icosahedron 60, $n$-gonal prism $2n$, no symmetry 1.

**Errors that tell you $p$ is wrong:**

* *"Full-precision rotation refinement changed the discovered vertex permutation"* → $p$ too **loose**; increase it.
* *"Detected symmetry operations are not closed under composition"* or *"missing an inverse"* → usually $p$ too **tight** (operations were rejected), occasionally a genuinely unreachable rotation order (see §23.2).

### 10.3 Orientation tolerance (`--orientation-angle-tol`) — the key knob

This sets the maximum permitted **cluster diameter**. It is the parameter that decides how many states you find. Too small and every particle becomes its own singleton; too large and all particles collapse into one state.

The hard upper bound is $\theta_{\max}(G)$, the maximum possible symmetry-reduced angle for your particle — beyond that everything merges trivially. The program prints the observed maximum pair angle per frame; use that as your ceiling.

**Method A — use the pairwise-angle histogram (recommended).** If you have the companion global pairwise-orientation histogram tool, compute $P(\theta_{ij})$ first. An orientationally ordered system shows a peak near $\theta = 0$ (within-state pairs) separated by a **minimum** from peaks at larger angles (between-state pairs). Set $\varepsilon_{\text{orient}}$ at that minimum. The default of 41° comes from exactly this construction for the reference system.

```
P(θ)
 |  ╱‾╲                    ╱‾╲
 | ╱   ╲                  ╱   ╲
 |╱     ╲________________╱     ╲
 +----------|--------------------→ θ
            41°  ← minimum → tolerance
```

**Method B — scan and look for a plateau in the state count.** Run with `--frames 1 --stability-trials 0` (cheap) over a range of tolerances and tabulate:

| tolerance | raw clusters | clusters ≥ cutoff |
|---|---|---|
| 10° | 3184 | 0 |
| 25° | 41 | 2 |
| 35° | 9 | 4 |
| 41° | 6 | **4** ← plateau |
| 50° | 5 | **4** ← plateau |
| 70° | 2 | 2 |
| 90° | 1 | 1 |

The plateau in *cutoff-qualified* cluster count is the physically robust answer. Report the width of the plateau in your methods section.

**Method C — theory.** If the states are related by the crystal's point group, compute the expected medoid separations analytically and set $\varepsilon_{\text{orient}}$ to roughly half the smallest.

**Sanity check after choosing:** the program prints `maximum validated diameter` per frame. It should be comfortably below your tolerance, not pinned at it. If every frame reports a maximum diameter exactly equal to the tolerance, clustering is being cut off artificially and distinct states are probably being merged.

### 10.4 Cluster size cutoff (`--cluster-size-cutoff`)

**One number, three distinct roles.** Understand all three before choosing.

| Role | Where | Effect |
|---|---|---|
| **Stability criterion** | `validate_partition_order_stability` | Only clusters with instantaneous size $\ge$ cutoff count toward the pass/fail comparison. Small clusters are ignored so that harmless reshuffling of a few singletons does not flag a false failure |
| **Tracking eligibility** | `build_cutoff_qualified_reference`, `track_all_frames_to_fixed_reference` | Reference states are reference-frame clusters with size $\ge$ cutoff. In every frame, only clusters with size $\ge$ cutoff are eligible to be matched. Smaller clusters get state `-1` and are counted as unmatched population |
| **Final validity** | `StateSummary.valid_by_mean_size_cutoff` | A state is "valid" only if its *mean* population over all frames (zeros included for absence) is $\ge$ cutoff. Only valid states appear in the plots |

**Choosing:** think in terms of a fraction of $M$. A cutoff of 200 out of 4096 is $\approx 5\%$. Reasonable starting points:

* **1–2% of $M$** — permissive; catches minority states, risks noise
* **5% of $M$** — balanced; the default's spirit
* **10% of $M$** — conservative; only major states

Check the effect by reading `raw clusters` versus `clusters with size >= cutoff` in the per-frame log. If the two differ enormously (e.g. 3000 vs 4), most particles are in tiny clusters and your orientation tolerance is probably too small.

> **Warning:** if the cutoff is so large that the reference frame has no qualifying cluster, the program raises `RuntimeError` — tracking cannot be defined.
>
> **Note on the two size tests:** a state can pass the *instantaneous* test (large enough in the reference frame to become a state) but fail the *mean* test (rarely present, so its average population is small). Such states appear in `_state_summary.csv` with `valid_by_mean_particle_count_cutoff = 0` and are excluded from the plots. That is intentional — it is exactly how transient states are flagged.

### 10.5 Reference frame (`--reference-frame`)

Must be one of the selected frames. Defaults to the **last** one.

* **Last frame (default)** — best when the system is still equilibrating and the final configuration is the most converged.
* **First selected frame** — best when you want to watch states decay away over the window.
* **A visually "clean" frame** — run once with `--frames 1` on several candidates and pick the one with the most well-separated states (largest `Minimum reference-medoid separation`).

The reference frame is special in the tracking: its own clusters map to their states with tracking distance exactly $0$, by construction rather than by computation.

**Diagnostic to run:** repeat the analysis with two or three different reference frames. If the number of valid states and the mean populations agree, your states are robust. If they do not, the system is not in a steady state over your window.

### 10.6 Tracking tolerance (`--tracking-angle-tol`)

The program computes a suggestion for you:

$$\text{suggestion} = \min\!\left(\frac{\varepsilon_{\text{orient}}}{2},\ 0.49 \times d_{\min}\right)$$

where $d_{\min}$ is the minimum pairwise distance between reference medoids. The strict safe bound is $0.5\, d_{\min}$: below it, the acceptance balls around distinct reference states cannot overlap, so no cluster can be a candidate for two states.

**In strict mode (the default), a tolerance $\ge 0.5 d_{\min} - \varepsilon_{\text{valid}}$ is rejected** with a `RuntimeError`. Override with `--allow-overlapping-tracking-regions` only if you understand the consequence: matches become genuinely ambiguous and the Hungarian solution, while still globally optimal, may not correspond to physical continuity.

**Choosing:**
* **Accept the suggestion** in almost all cases.
* **Reduce it** if you see spurious matches — states whose tracking distance is suspiciously large. Check `max_tracking_distance_deg` in `_state_summary.csv`; if it sits close to your tolerance, tighten.
* **Increase it** if states are wrongly reported absent because their medoid drifts between frames. Watch the `unmatched_fraction` column: a large unmatched fraction with a small tolerance means the tolerance is too tight.

### 10.7 Stability trials (`--stability-trials`)

| Value | Use |
|---|---|
| `0` | Exploratory scans. Disables the check entirely; `cluster_count_stability_passed` becomes `None` and the per-frame log prints `not-tested` |
| `1` | Default. Cheap smoke test |
| `3–5` | Recommended for production |
| `10+` | When you suspect tie-driven instability, or for a methods-section claim |

Each trial roughly **doubles** the per-frame clustering cost (one extra linkage plus a permuted memmap). A failing trial prints a warning and continues; it never aborts the run. Inspect `_particle_order_stability_medoid_diagnostics.csv` afterwards.

**Interpretation:** if the qualified cluster count changes under reshuffling, your orientation tolerance is sitting on a knife edge where distance ties matter. Move it to a plateau (§10.3).

### 10.8 Block size (`--block-size`)

Affects **peak memory only**; never changes the numbers. Peak temporary angle-matrix memory is

$$8 \times \text{block\_size} \times M \ \text{bytes}.$$

| $M$ | block 128 | block 512 | block 2048 |
|---|---|---|---|
| 4 096 | 4 MB | 17 MB | 67 MB |
| 20 000 | 20 MB | 82 MB | 328 MB |
| 50 000 | 51 MB | 205 MB | 819 MB |

Larger blocks mean fewer freud calls (modestly faster) and more RAM. Reduce it if you hit `MemoryError` during the distance stage; raise it to 512–1024 on a large-memory node.

### 10.9 Frames and particles (`--frames`, `--particles`)

* **`--particles`** dominates cost: everything scales as $O(M^2)$ or worse. Start with $M \le 2000$ for exploration, then scale up. Note the selection is the **first $M$ particles by index**, so if your GSD file has index-correlated structure (types sorted, spatial sort applied at write time), a subset is biased — see §23.5.
* **`--frames`** scales linearly. Use `--frames 1` while tuning, then extend. More frames give better statistics on $\langle n_s\rangle$ and $\sigma$, and a more meaningful presence fraction.

### 10.10 Numerical tolerances (rarely touched)

* **`--angle-validation-tol`** (default $10^{-5}$°). Raise it only if freud emits an unusually large numerical floor and you see spurious "matrix is not symmetric" errors. Raising it slightly widens the tracking acceptance window (`tolerance + angle_validation_tol`), so keep it far smaller than any physical angle.
* **`NUMERICAL_SELF_ANGLE_HARD_LIMIT_DEG`** (0.5°). If your freud build reports $d_G(q,q) > 0.5°$, something is genuinely wrong — do not simply raise this constant; investigate the installation.
* **`--random-seed`** (1729). Change it to confirm that stability results are not an artefact of one particular permutation sequence.

---

# How it executes

## 11. Top-level execution map

```
STAGE 0   Parse CLI
STAGE 1   Validate --block-size >= 1 and --angle-validation-tol > 0
STAGE 2   Lazy-import freud / gsd.hoomd / matplotlib; open GSD; resolve shape path;
          reject an empty trajectory; print the trajectory summary
STAGE 3   Prompt/resolve number of final frames → selected_frame_indices = [T-F .. T-1]
STAGE 4   Read every selected frame's orientation array; verify shape (N,4); report min/max N
STAGE 5   Prompt/resolve: particles M, reference frame, edges, faces, precision p,
          orientation tolerance, cluster size cutoff, stability trials
STAGE 6   read_shape_vertices() → detect_proper_rotational_symmetries()          ← PART A
STAGE 7   Create the output directory, filename prefix, stability-diagnostics file
          (stale copy deleted), and the temporary/persistent distance-file directory
STAGE 8   For each selected frame: cluster_one_frame()                           ← PART B
          finally: remove the temporary directory unless --keep-distance-files
          -- build_cutoff_qualified_reference()
          -- reference_medoid_distance_matrix() and its numerical validation
          -- tracking_cutoff_suggestion(); print separation info
          -- Prompt/resolve the tracking tolerance; enforce the non-overlap rule
STAGE 9   track_all_frames_to_fixed_reference()                                  ← PART C
          summarize_fixed_reference_states()                                     ← PART D
STAGE 10  save_symmetry_outputs, save_cluster_and_tracking_outputs, save_plots,
          save_analysis_metadata; print the final summary                        ← PART E
```

**Design note on ordering:** all cheap validation and all user input that *can* be collected up front, is. The expensive symmetry detection (Stage 6) precedes the even more expensive clustering (Stage 8). A wrong shape or topology therefore fails within seconds, not hours.

---

## 12. Part A — Shape processing and symmetry group

`read_shape_vertices()` then `detect_proper_rotational_symmetries()`, the latter organised as 13 internal stages.

### 12.1 Reading the vertices

Three collaborating functions:

* **`_as_vertex_array(value)`** — returns `np.asarray(value, float)` if and only if it is 2-D, has 3 columns, has $\ge 4$ rows and is entirely finite; otherwise `None`.
* **`_collect_vertex_candidates(obj, key_hint)`** — recursive descent. A valid array is emitted and descent stops. A `dict` is traversed with keys containing `"vert"` first, then alphabetically (deterministic). A `list`/`tuple` is traversed element-wise with the inherited key hint.
* **`read_shape_vertices(path)`** — scores each candidate as $10000 \cdot \mathbb{1}[\text{"vert"} \in \text{key}] + N_{\text{rows}}$, so a small array under a key named `vertices` beats a large unrelated $N\times3$ array elsewhere in the file, and among equally named candidates the largest wins. Then runs the degeneracy and duplicate checks and prints the winning key hint and vertex count.

### 12.2 Centring

$$\mathbf{c} = \tfrac1N\textstyle\sum_a \mathbf{r}_a, \qquad \tilde{\mathbf{r}}_a = \mathbf{r}_a - \mathbf{c}$$

A rotation must act about the particle centre; rotating about the coordinate origin is only valid for an already-centred shape. The program prints $\mathbf{c}$ and $\lVert\mathbf{c}\rVert$, reports `CENTRE CHECK: PASSED` if $\lVert\mathbf{c}\rVert \le 10^{-p}$ (otherwise an informational note that it will translate), and then verifies the recentring residual is below $100\epsilon_{\text{machine}}$.

> The **arithmetic vertex mean** is used, not the volumetric centroid. For any shape with a nontrivial rotation group the vertex mean is a fixed point of that group, so it is the correct rotation centre.

### 12.3 Convex decomposition and the topology gate

**`_merge_coplanar_hull_triangles(vertices, merge_tolerance)`**

1. `ConvexHull` (Qhull) gives a *triangulated* surface: `simplices` plus one plane equation $[n_x,n_y,n_z,b]$ per triangle with consistently outward unit normals.
2. Triangles of the same planar face have nearly identical equation rows, so the equations are treated as points in 4-D and `cKDTree.query_pairs(merge_tolerance)` finds coplanar pairs.
3. Those pairs fill a dense symmetric `int8` adjacency matrix of size $n_{\text{tri}}^2$; `connected_components` groups triangles into faces.
4. Per face: vertex set from `np.unique(simplices.ravel())`; plane equation = mean of member equations, rescaled so $\lVert\mathbf{n}\rVert = 1$; vertices ordered cyclically by building an in-plane basis ($\mathbf{a}$ from the first vertex's displacement from the centroid, $\mathbf{b} = \mathbf{n}\times\mathbf{a}$) and sorting by $\operatorname{atan2}(\mathbf{d}\cdot\mathbf{b}, \mathbf{d}\cdot\mathbf{a})$.
5. Edges come from consecutive pairs in the cyclic ordering (`np.roll(face, -1)`), stored as sorted `(min,max)` tuples in a `set`, which deduplicates the two faces sharing each edge.

**`find_validated_convex_decomposition()`** scans $10^{-12}, 10^{-11}, \ldots, 10^{-4}$, prints every trial, and **returns on the first exact match to both counts**. If none matches it raises `RuntimeError` reporting expected versus final-trial counts. Because the ladder runs tight→loose, the tightest tolerance reproducing your topology is selected.

This is a **hard gate**, deliberately: an over- or under-merged face set produces a wrong candidate-axis set and hence a wrong symmetry group — a silent failure that would corrupt every subsequent distance.

### 12.4 Candidate axes and angles

**`build_candidate_axis_lines()`** — four classes, matching the original detector:

| Class | Vector | Count |
|---|---|---|
| Centre → vertex | $\tilde{\mathbf{r}}_a$ | $V$ |
| Centre → face centroid | mean of the face's vertices | $F$ |
| Centre → perpendicular foot on face plane | $-b\,\mathbf{n}$ | $F$ |
| Centre → edge midpoint | $\tfrac12(\tilde{\mathbf{r}}_a + \tilde{\mathbf{r}}_b)$ | $E$ |

$V + 2F + E$ raw candidates. The perpendicular-foot class is the corrected form of the legacy "centre-to-face-normal" construction: in centred coordinates with $\lVert\mathbf{n}\rVert=1$, the foot of the perpendicular from the origin to $\mathbf{n}\cdot\mathbf{x}+b=0$ is exactly $-b\mathbf{n}$. For an irregular face this differs from the centroid, so both classes are needed.

Deduplication: drop vectors of norm $\le 10^{-14}$; normalise and sign-canonicalise via `_canonical_axis_line` (flip so the first component with $\lvert v\rvert > 10^{-14}$ is positive); drop any axis with $\lvert \mathbf{n}\cdot\mathbf{n}_{\text{prev}}\rvert \ge 1 - 10^{-12}$ (parallel **or** antiparallel = same line). The first label producing each line is retained for traceability.

**`historical_candidate_angles_rad()`** — a fixed list of **37 angles**: 19 positive (180, 120, 240, 90, 270, 72, 144, 216, 288, 60, 300, 45, 135, 225, 315, 252, 324, 36, 108) and 18 negative (the same set excluding $\pm180$, which is self-inverse). These are the non-identity rotations of $C_n$ for $n \in \{2,3,4,5,6,8,10\}$. Negative angles are required because axis *lines* were canonicalised to one sign.

Total candidates tested $= n_{\text{unique axes}} \times 37$.

### 12.5 Candidate testing, matching and refinement

Operations are stored in `operations_by_permutation: dict[tuple[int,...], SymmetryOperation]`, keyed by the induced integer vertex permutation, and seeded with the identity (angle $0$ is not in the candidate list).

For each (axis, angle):

1. **Construct** $R$ from the rotation vector $\boldsymbol{\omega} = \mathbf{n}\theta$ via `Rotation.from_rotvec`.
2. **Apply** to all centred vertices.
3. **Match one-to-one** — `_one_to_one_vertex_mapping()` builds the full $N\times N$ cost matrix $C_{ab} = \lVert\tilde{\mathbf{r}}'_a - \tilde{\mathbf{r}}_b\rVert$ and solves the **linear sum assignment (Hungarian) problem**. This returns the globally minimum-cost *perfect* matching, guaranteeing each rotated vertex is used once and each reference vertex hit once — exactly the permutation condition for a rigid symmetry. A greedy nearest-neighbour lookup could assign two rotated vertices to one target and so accept a non-symmetry; the Hungarian solve cannot.
4. **Reject** if $\max_a C_{a,\sigma(a)} > 10^{-p}$. The **maximum**, not the RMS — the condition is enforced per vertex, not on average.
5. **Refine at full precision** — the candidate axes come from finite-precision coordinates, so the raw rotation is only approximate. With $\sigma$ known, `_refine_rotation_for_permutation` solves the orthogonal Procrustes / Wahba problem via `Rotation.align_vectors(target, source)`, minimising $\sum_a\lVert\mathbf{t}_a - R\mathbf{s}_a\rVert^2$, with a $\det R > 0$ guard.
6. **Re-verify independently** — apply the refined $R$, recompute the assignment. If the permutation changed, raise `RuntimeError` (the tolerance does not resolve the correspondence uniquely — use a tighter $p$). If the refined maximum residual still exceeds $10^{-p}$, skip.
7. **Store**, keeping the discovery with the smallest refined residual when several axis/angle pairs find the same permutation.

Operations are then sorted by $(\lVert R - I\rVert_F,\ \sigma)$ — identity first, then increasing rotation magnitude.

### 12.6 Group validation

**Why in permutation space?** Rotation matrices and quaternions carry floating-point noise; their action on a finite vertex set is captured *exactly* by integer permutations, making the axioms discrete and unambiguous.

`validate_permutation_group()` checks exhaustively:

1. **Identity** $(0,1,\ldots,N-1)$ is present.
2. **Inverses** — for every $\sigma$, construct $\sigma^{-1}$ (`inverse[target] = source`) and require membership.
3. **Closure** — for every ordered pair, $(\sigma_2\circ\sigma_1)[i] = \sigma_2[\sigma_1[i]]$ must be in the set.

Failure raises `RuntimeError`. This is the most valuable single check in the program: a missing operation would inflate every distance in the analysis, and closure failure catches it.

### 12.7 Equivalent quaternion set

`_rotation_to_wxyz` converts SciPy's `[x,y,z,w]` to freud/HOOMD `[w,x,y,z]` (index reorder `[3,0,1,2]`), renormalises, and applies the first-nonzero-positive sign convention. Then both signs are emitted, interleaved:

```python
equivalent_quaternions[0::2] =  physical_quaternions   # +q
equivalent_quaternions[1::2] = -physical_quaternions   # -q
```

giving a $(2n,4)$ array. Two assertions follow: every row unit-norm to `atol=1e-12`, and rows $2k+1$ exactly $= -$ rows $2k$ (`atol=0, rtol=0`, since they were produced by unary negation, so bit-exact equality is expected).

Supplying both signs makes the distance invariant to the arbitrary sign of the stored HOOMD quaternions, regardless of freud's internal handling. It costs a factor of two in the inner minimisation and nothing in correctness.

---

## 13. Part B — Per-frame clustering

`cluster_one_frame()` orchestrates five steps per frame.

### 13.1 Load and validate orientations

`validate_and_normalize_orientations()`:

1. `np.asarray(snapshot.particles.orientation, float64)`.
2. Reject `ndim != 2` or `shape[1] != 4`; reject fewer rows than requested.
3. Slice the **first $M$** rows; force C-contiguity (freud requires it).
4. Reject non-finite entries; reject any quaternion of norm $\le 10^{-14}$.
5. **Record** $\max_i \lvert \lVert q_i\rVert - 1\rvert$ **before** correction — the audit trail for GSD writer precision, reported per frame and in the metadata.
6. Normalise `array /= norms[:, None]`.

### 13.2 Build the condensed distance array on disk

`compute_condensed_distance_memmap()` is the memory-critical routine.

**Storage.** A square $M\times M$ float64 matrix would waste half its entries to the symmetry $d(i,j)=d(j,i)$ plus a useless diagonal. Instead only the strict upper triangle is stored, in **SciPy's condensed order**:

$$(0,1),(0,2),\ldots,(0,M{-}1),\;(1,2),\ldots,(1,M{-}1),\;\ldots,\;(M{-}2,M{-}1)$$

$M(M-1)/2$ values, written to a `np.memmap(dtype=float64, mode="w+")` so the raw storage lives on **disk**, not RAM. `mode="w+"` creates or overwrites and permits both reading and writing.

**Index arithmetic.** `condensed_index(M, i, j)` implements the standard formula

$$\text{idx} = M\,i - \frac{i(i+1)}{2} + j - i - 1, \qquad i < j$$

vectorised over NumPy arrays, with `low = min(i,j)`, `high = max(i,j)`, and a `ValueError` for self-pairs.

**Block loop.** For `block_start in range(0, M, block_size)`:

```python
query = orientations[block_start:block_stop]
calculator.compute(orientations, query, equivalent_quaternions)
block = np.rad2deg(calculator.angles)     # shape (block, M)
```

One `AngularSeparationGlobal` object is created per frame and reused across blocks. The output shape is **explicitly checked** against `(block_stop - block_start, M)` — a guard against a silent axis transposition across freud versions, which would still produce a plausible-looking result. Non-finite output raises immediately.

**Row-wise extraction.** For each row (global particle $i$), take `block[local_row, i+1:]` — a NumPy *view*, not a copy. This excludes $j<i$ (already stored as $d(j,i)$) and $j=i$ (self-pair). Per row:

* reject $\min < -\varepsilon_{\text{valid}}$ (negative distance) and $\max > 180 + \varepsilon_{\text{valid}}$;
* `np.clip(values, 0, 180)` — removes only negligible roundoff outside the theoretical interval, since anything substantial has already raised;
* write contiguously at the cursor, advance the cursor. **The row-wise upper-triangle order naturally produces exactly SciPy's condensed ordering**, so no reindexing is needed;
* update the true global min/max.

**Accounting.** After the loop, `cursor != pair_count` raises `RuntimeError`. Then `distances.flush()` synchronises the modified pages with the file.

Returns the memmap plus a `DistanceDiagnostics(min, max)`.

### 13.3 Complete-linkage clustering

`complete_linkage_labels()`:

```python
hierarchy = linkage(condensed_distances, method="complete", optimal_ordering=False)
labels = fcluster(hierarchy, t=orientation_angle_tolerance_deg, criterion="distance") - 1
```

`optimal_ordering=False` skips leaf reordering, which is a visualisation nicety and is not used to define cluster identity. `criterion="distance"` means no cluster is formed through a merge above $t$. SciPy's 1-based labels are shifted to 0-based; the numerical label values are arbitrary — particle membership is the physical information.

> **Memory note:** SciPy's `linkage` converts its input to an in-memory contiguous float64 array. The memmap therefore reduces persistent *storage*, but peak RAM during linkage still includes one full $8 \times M(M-1)/2$-byte copy. See §21.

### 13.4 Particle-order stability check

`validate_partition_order_stability()` — advisory, never fatal.

For each of `stability_trials` trials:

1. Draw a permutation from `np.random.default_rng(random_seed + 1_000_003 * frame_index)`. **The seed is offset per frame**, so different frames get different permutations while the whole run stays reproducible.
2. `build_permuted_condensed_distances()` writes a *second* memmap whose condensed entries correspond to the reshuffled particle order — no distances are recomputed, only reindexed via `condensed_index`.
3. Re-run complete linkage on the permuted distances. The trial file is deleted in a `finally` block.
4. `_cluster_medoid_records_from_labels(permuted_labels, permutation, ...)` — note `permutation` is passed as `original_particle_ids`, so cluster membership comes back expressed in **original** particle IDs and all statistics are computed from the **original** condensed distances. This makes the comparison genuinely apples-to-apples.
5. Filter both original and permuted record lists to those with size $\ge$ cutoff, relabel them `A, B, C, …` via `_alphabetic_group_name` (0→A, 25→Z, 26→AA), and compare the **counts**.

**The criterion is the cutoff-qualified cluster count, not exact membership.** Small clusters are deliberately excluded so that harmless reshuffling of a few singletons does not raise a false alarm.

6. Whether or not the counts match, a **medoid correspondence** is computed for diagnostics: an angular distance matrix between original and permuted qualified medoids, solved by `linear_sum_assignment`. Matched pairs are labelled `A ↔ A'`; unmatched groups on either side get their own row. Everything is appended to `_particle_order_stability_medoid_diagnostics.csv`.

Returns `True` (all trials passed), `False` (at least one failed — warning printed, run continues), or `None` (`stability_trials == 0`, not tested).

### 13.5 Construct, verify and order the clusters

`construct_frame_clusters()`:

1. For each label, gather members and call `cluster_distance_statistics()`, which computes — in one pass over the cluster's upper triangle, reading directly from the memmap:
   * $S_i$ for every member (each pair contributes to both endpoints' sums),
   * the exact diameter,
   * the medoid (min $S$, ties by smallest particle index),
   * $\max_j d(m, j)$, the maximum distance to the medoid.
   Singletons short-circuit to $(i, 0, 0, 0)$.
2. **The diameter gate.** If $\operatorname{diam}(C) > \varepsilon_{\text{orient}} + 10^{-10}$, raise `RuntimeError`. The message states explicitly that the numerical self-angle floor is not permitted to relax this criterion.
3. Build a `FrameCluster` with the medoid's **sign-canonicalised** quaternion (`canonicalize_quaternion_sign`: normalise, then flip so the first component with $\lvert\cdot\rvert > 10^{-14}$ is positive).
4. **Deterministic ordering** — sort whole records by $(-\text{size},\ \text{medoid index},\ \text{members})$, so cluster 0 is always the largest. The source comments flag this as a fix for a legacy bug where membership, size and representative lived in separate arrays that could fall out of sync; here they are one record sorted as a unit.
5. Assign `local_cluster_id = 0,1,2,…` and build `particle_to_local_cluster`.
6. **Conservation checks** — no particle in two clusters, every particle assigned, and $\sum_C \lvert C\rvert = M$.

### 13.6 Cleanup

A `finally` block flushes and drops the memmap and deletes the frame's distance file unless `--keep-distance-files`.

---

## 14. Part C — Reference selection and tracking

### 14.1 Filter the reference frame

`build_cutoff_qualified_reference()` retains only reference-frame clusters with size $\ge$ cutoff; these become the numbered fixed states, in the same order (largest first). If none qualifies, `RuntimeError` — tracking cannot be defined.

The console reports raw count, retained count, and the number dropped.

### 14.2 Validate the reference medoid geometry

`reference_medoid_distance_matrix()` computes the full square distance matrix between reference medoids and subjects it to three checks:

1. **Self-angle floor.** $\max_k \lvert d(q_k,q_k)\rvert$ is measured. If it exceeds `NUMERICAL_SELF_ANGLE_HARD_LIMIT_DEG` (0.5°), abort — freud output is not trustworthy. If it merely exceeds `angle_validation_tol`, print a notice.
2. **Diagonal correction.** The diagonal is set to **exact zero**, because a medoid is mathematically identical to itself. Off-diagonal physical distances are untouched.
3. **Symmetry.** $\max\lvert D - D^{\mathsf T}\rvert$ must be within
   $$\varepsilon_{\text{eff}} = \max\bigl(\varepsilon_{\text{valid}},\ 4 \times \text{measured floor}\bigr),$$
   else `RuntimeError`. Using the *measured* floor scaled by a safety factor adapts the check to the actual precision of the freud build rather than assuming one.

Prints `REFERENCE-MEDOID DISTANCE CHECK: PASSED` with both numbers.

### 14.3 Suggest the tracking tolerance

`tracking_cutoff_suggestion()`:

* With one state: suggestion $= \varepsilon_{\text{orient}}/2$, safe bound `None`.
* Otherwise, with $d_{\min}$ the minimum off-diagonal medoid separation:
  $$\text{suggestion} = \min\bigl(\varepsilon_{\text{orient}}/2,\ 0.49\,d_{\min}\bigr), \qquad \text{safe bound} = 0.5\,d_{\min}.$$
* If the suggestion falls below `angle_validation_tol`, raise `RuntimeError` — the medoids are too close to define a meaningful non-overlapping tolerance.

The console prints $d_{\min}$ and the safe bound, then prompt 10 collects the value. In strict mode, a value $\ge$ safe bound $-\ \varepsilon_{\text{valid}}$ raises unless `--allow-overlapping-tracking-regions`.

### 14.4 Match each frame to the fixed states

`track_all_frames_to_fixed_reference()`:

**The reference frame itself** is handled by construction, not computation: each reference cluster maps to its state with distance exactly $0.0$; every other cluster in that frame (i.e. those below the cutoff) is unmatched.

**Every other frame:**

1. Select **eligible** clusters — size $\ge$ cutoff.
2. Compute the eligible-medoid × reference-medoid distance matrix.
3. Solve the assignment via `globally_match_clusters_to_reference_states()`.
4. Scatter the eligible results back into the full per-cluster arrays; ineligible clusters keep state $-1$ and distance `NaN`.
5. **Conservation check** — matched population + unmatched population must equal $M$.
6. If no cluster is eligible, all states are recorded absent.

### 14.5 The assignment itself

`globally_match_clusters_to_reference_states()`, the algorithmic core of the tracking.

**Validity mask.**
$$\text{valid}_{cr} = \bigl[D_{cr} \le \varepsilon_{\text{track}} + \varepsilon_{\text{valid}}\bigr]$$

**Ambiguity detection** (before any assignment):
* `ambiguous_current` = clusters (rows) with more than one valid candidate state;
* `split_candidate_references` = states (columns) with more than one valid candidate cluster — the signature of a state splitting in two.

In strict mode either condition raises `RuntimeError` naming the offending indices. With `--allow-ambiguous-tracking` they are merely recorded and reported.

**The padded cost matrix.** With $C$ clusters and $R$ states, build a square $(C+R)\times(C+R)$ matrix:

```
             ref 0..R-1              dummy 0..C-1 (one per cluster)
cluster 0 [ D if valid else 1e12  |  unmatched_cost on the diagonal only ]
   ...    [                       |                                     ]
dummy 0   [ unmatched_cost on the |                 0.0                 ]
  ...     [ diagonal only         |                                     ]
```

with `unmatched_cost = tracking_tol + max(angle_validation_tol, 1e-9)` and `large_cost = 1e12`.

This padding is what lets clusters go unmatched and states go absent. `linear_sum_assignment` requires a complete matching, so without the dummies it would be forced to pair everything, however badly. With them, "leave this cluster unmatched" is an explicit option costing slightly more than any valid real match — so a valid match is always preferred, but an invalid one (cost $10^{12}$) never is.

**Post-conditions, all checked:**
* every accepted real match satisfies $D \le \varepsilon_{\text{track}} + \varepsilon_{\text{valid}}$, else `RuntimeError`;
* no state is matched twice (`len(matched_states)` equals the count of matched clusters).

Returns the cluster→state map, tracking distances, unmatched cluster IDs, absent state IDs, split candidates and ambiguous clusters.

---

## 15. Part D — Aggregation into state statistics

`summarize_fixed_reference_states()` builds three matrices indexed `[frame_position, state_id]`:

| Matrix | dtype | Fill when absent |
|---|---|---|
| `counts` | int64 | **0** |
| `tracking_distances` | float64 | `NaN` |
| `diameters` | float64 | `NaN` |

plus a `unmatched_counts` vector per frame. A per-frame conservation check requires $\sum_s \text{counts} + \text{unmatched} = M$.

The **absent-state rule is explicit and consequential**: a state that does not appear in a frame contributes a **zero** to its mean, not a skipped entry. So $\langle n_s\rangle$ measures average population over the whole window, and a state present at strength 400 in half the frames and absent in the other half reports $\langle n_s \rangle = 200$ with $p_s = 0.5$. Tracking distances and diameters, by contrast, use `NaN` and are averaged only over frames where the state was present.

Per state, `StateSummary` records mean/σ of count, mean/σ of fraction, presence fraction, max and mean tracking distance (when present), max diameter (when present), and `valid_by_mean_size_cutoff`.

**Sort order:** `(not valid, -mean_particle_count, reference_state_id)` — valid states first, then by decreasing mean population.

> **Important indexing subtlety:** `state_counts` and `state_fractions` are indexed by `reference_state_id` in the *original* (reference-frame, largest-first) order, whereas `state_summaries` is the *sorted* list. Every consumer indexes with `state.reference_state_id`, so the two stay consistent. When reading `_per_frame_populations.csv`, the column `state_k_count` refers to `reference_state_id == k`, **not** to the $k$-th row of `_state_summary.csv`.

---

## 16. Part E — Outputs

Six CSVs, two PNGs, two companion plotting CSVs, one JSON, plus an optional diagnostics CSV and an optional distance-file directory. Full column specifications in §17.

Notable design choices:

* **Every plot ships with a companion CSV containing the exact plotted coordinates**, named `for_plotting_<png_stem>.csv`. Nothing has to be recomputed to reproduce or restyle a figure.
* **Only states passing `valid_by_mean_size_cutoff` are plotted**, and the companion CSVs contain exactly the displayed states — so CSV and figure always agree.
* If no state passes the cutoff, the bar plot renders an explanatory text panel and the CSV is header-only. The program does not crash.
* All output names are built with `with_name(name + suffix)`, so filenames containing dots are handled safely.

---

# Reference material

## 17. Output file and column reference

### 17.1 Filename prefix

```
<output_dir>/<gsd_stem>_fixed_reference_orientation_states_last_<F>_frames_particles_<M>
```

### 17.2 The full file set

| File | Written by | Contents |
|---|---|---|
| `<prefix>_symmetry_quaternions.csv` | `save_symmetry_outputs` | All $\pm q$ symmetry operations |
| `<prefix>_state_summary.csv` | `save_cluster_and_tracking_outputs` | One row per fixed state — **the headline result** |
| `<prefix>_per_frame_populations.csv` | " | Population time series, wide format |
| `<prefix>_frame_clusters.csv` | " | Every cluster in every frame |
| `<prefix>_particle_membership.csv` | " | Every particle's cluster and state, per frame |
| `<prefix>_reference_states.csv` | " | The reference-frame clusters that define the states |
| `<prefix>_mean_state_populations.png` | `save_plots` | Bar chart with σ error bars |
| `for_plotting_<prefix>_mean_state_populations.csv` | " | Exact bar coordinates |
| `<prefix>_state_population_timeseries.png` | " | Line plot per state plus unmatched series |
| `for_plotting_<prefix>_state_population_timeseries.csv` | " | Exact plotted points, long format |
| `<prefix>_metadata.json` | `save_analysis_metadata` | Complete run record |
| `<prefix>_particle_order_stability_medoid_diagnostics.csv` | `_append_particle_order_medoid_diagnostics` | Only if `stability_trials > 0` |
| `<prefix>_distance_files/` | — | Only with `--keep-distance-files` |

All CSVs use `comments=""`, so the header line is bare rather than `#`-prefixed. Numeric CSVs use `%.17g`, which round-trips IEEE-754 doubles losslessly.

### 17.3 `_state_summary.csv` — the headline result

```
reference_state_id,reference_local_cluster_id,reference_size,reference_medoid_particle_index,
w,x,y,z,mean_particle_count,std_particle_count,mean_population_fraction,std_population_fraction,
presence_fraction,max_tracking_distance_deg,mean_tracking_distance_deg_when_present,
max_cluster_diameter_deg,valid_by_mean_particle_count_cutoff
```

| Column | Meaning |
|---|---|
| `reference_state_id` | Stable state ID used everywhere else |
| `reference_local_cluster_id` | Its cluster ID within the reference frame |
| `reference_size` | Its particle count *in the reference frame* |
| `reference_medoid_particle_index` | The actual particle whose orientation defines the state |
| `w,x,y,z` | **The state's defining orientation**, sign-canonicalised, scalar-first |
| `mean_particle_count` | $\langle n_s\rangle$, zeros included for absence |
| `std_particle_count` | σ, `ddof=0` |
| `mean_population_fraction` | $\langle n_s\rangle / M$ |
| `std_population_fraction` | σ of the fraction |
| `presence_fraction` | Fraction of frames where the state was matched at all |
| `max_tracking_distance_deg` | Worst medoid drift observed; compare against your tolerance |
| `mean_tracking_distance_deg_when_present` | Typical drift |
| `max_cluster_diameter_deg` | Worst internal spread; must be $\le$ orientation tolerance |
| `valid_by_mean_particle_count_cutoff` | `1` if $\langle n_s\rangle \ge$ cutoff (plotted), else `0` |

Rows are ordered valid-first, then by decreasing mean population. All values are written with `fmt="%.17g"`, so integer-valued columns read back as e.g. `3.0` and absent statistics as `nan`.

### 17.4 `_per_frame_populations.csv`

```
frame_index,state_0_count,...,state_{S-1}_count,state_0_fraction,...,state_{S-1}_fraction,unmatched_count,unmatched_fraction
```

$F$ rows. `state_k_*` refers to `reference_state_id == k`. `unmatched_count` sums every cluster with state $-1$ — i.e. clusters below the size cutoff plus eligible clusters that found no acceptable state. By construction, $\sum_k \text{state\_}k\text{\_count} + \text{unmatched\_count} = M$ in every row.

### 17.5 `_frame_clusters.csv`

One row per cluster per frame — **including** clusters below the cutoff.

```
frame_index,local_cluster_id,cluster_size,medoid_particle_index,medoid_w,medoid_x,medoid_y,medoid_z,
diameter_deg,max_angle_to_medoid_deg,medoid_sum_distance_deg,reference_state_id,tracking_distance_deg,
diameter_within_requested_tolerance
```

`reference_state_id = -1` marks unmatched; `tracking_distance_deg` is then `nan`. `diameter_within_requested_tolerance` should be `1` for every row — it is a redundant belt-and-braces flag, since a violation would already have raised during clustering.

### 17.6 `_particle_membership.csv`

```
frame_index,particle_index,local_cluster_id,reference_state_id
```

$F \times M$ rows, all integers (`fmt="%d"`). This is the finest-grained output: it tells you exactly which state each individual particle occupied in each frame, so you can compute residence times, transition matrices, or spatial maps by joining against positions from the GSD file. **Note this file can be very large** — 50 frames × 4096 particles = 204 800 rows.

### 17.7 `_reference_states.csv`

```
reference_state_id,reference_size,medoid_particle_index,w,x,y,z,diameter_deg,max_angle_to_medoid_deg,medoid_sum_distance_deg
```

The frozen state definitions. Columns 4–7 are directly reusable as target orientations in another calculation.

### 17.8 `_symmetry_quaternions.csv`

Two rows per physical rotation ($+q$ then $-q$), $2n$ rows total.

```
physical_operation_index,quaternion_sign,w,x,y,z,max_vertex_residual,discovery_angle_deg,discovery_axis_x,discovery_axis_y,discovery_axis_z
```

Columns 8–11 describe the *discovery* and are metadata only. The authoritative definition is columns 3–6, from the full-precision Procrustes refit.

### 17.9 `_particle_order_stability_medoid_diagnostics.csv`

Appended per trial; deleted and recreated at the start of each run. Written with `csv.DictWriter`, so unused cells are genuinely empty strings rather than `nan`.

```
frame_index,trial_index,cluster_size_cutoff,original_total_cluster_count,permuted_total_cluster_count,
original_qualified_cluster_count,permuted_qualified_cluster_count,same_qualified_cluster_count,
original_group_id,original_group_name,permuted_group_id_before_matching,matched_permuted_group_name,
original_cluster_size,permuted_cluster_size,original_medoid_particle_index,permuted_medoid_particle_index,
medoid_angle_difference_deg,match_status
```

`match_status` takes one of four values:

| Value | Meaning |
|---|---|
| `matched_cutoff_qualified_groups_by_minimum_total_medoid_angle` | A matched pair; `medoid_angle_difference_deg` is the key number — it should be $\approx 0$ |
| `unmatched_cutoff_qualified_original_group` | A group in the original run with no counterpart |
| `unmatched_cutoff_qualified_permuted_group` | A group in the permuted run with no counterpart |
| `no_medoid_assignment_because_at_least_one_cutoff_qualified_subset_is_empty` | Trial-level audit row when no comparison was possible |

### 17.10 `for_plotting_*_mean_state_populations.csv`

```
x_bar_center,reference_state_id,x_tick_label,y_mean_population_fraction,y_error_standard_deviation,y_error_lower,y_error_upper,mean_particle_count,standard_deviation_particle_count
```

One row per **displayed** (valid) state. `x_bar_center` is $0,1,2,\ldots$ matching the bar positions.

### 17.11 `for_plotting_*_state_population_timeseries.csv`

Long format — one row per plotted point per series, so it can be replotted with a simple group-by.

```
series_order,series_type,reference_state_id,series_label,x_frame_index,y_population_fraction,marker,linestyle,linewidth
```

`series_type` is `reference_state` or `unmatched_frame_level_groups`. The unmatched series has `series_order = len(valid_states)` and an empty `reference_state_id`. Marker, linestyle and linewidth are recorded so the exact figure style can be reproduced.

### 17.12 `_metadata.json`

Indented JSON containing, among others:

**Inputs and selection** — `trajectory_file`, `shape_file`, `total_trajectory_frames`, `selected_frame_indices`, `selected_particle_count_per_frame`, `reference_frame_index`

**Self-documenting method strings** — `frame_clustering_method`, `frame_cluster_constraint`, `representative_definition`, `tracking_strategy`, `absent_state_population_rule`, `cluster_size_cutoff_definition`, `particle_order_stability_criterion`, `particle_order_stability_failure_action`

**Parameters** — `orientation_angle_tolerance_deg`, `tracking_angle_tolerance_deg`, `nonoverlapping_tracking_safe_upper_bound_deg`, `cluster_size_cutoff`, `expected_edges`, `expected_faces`, `symmetry_precision_exponent`, `symmetry_matching_tolerance`, `physical_proper_rotation_count`, `equivalent_quaternion_count`, `block_size`, `particle_order_stability_trials_per_frame`, `random_seed`, `angle_validation_tolerance_deg`, `particle_order_medoid_diagnostics_file`

**Per-frame array** `frames[]` — `frame_index`, `cluster_count`, `cutoff_qualified_cluster_count`, `max_quaternion_norm_deviation`, `minimum_pair_angle_deg`, `maximum_pair_angle_deg`, `cluster_count_stability_passed` (`true`/`false`/`null`), `maximum_cluster_diameter_deg`

**Per-frame tracking** `tracking[]` — `unmatched_cluster_ids`, `absent_reference_state_ids`, `split_candidate_reference_ids`, `ambiguous_current_cluster_ids`

**Condensed state summaries** `state_summaries[]` — `reference_state_id`, `mean_particle_count`, `mean_population_fraction`, `presence_fraction`, `valid_by_mean_size_cutoff`

### 17.13 The plots

| | Bar plot | Time series |
|---|---|---|
| Figure size | `(max(5.0, 0.65·n_valid), 3.8)` | `(6.0, 4.0)` |
| Saved | `dpi=600, bbox_inches="tight"` | same |
| Content | One bar per valid state, `yerr` = σ of fraction, `capsize=3` | One `o`-marked line per valid state plus a dashed `x` line for the unmatched fraction |
| x | State index, tick labels `State k`, rotated 45° | Trajectory frame index |
| y | Mean population fraction, limit `max(0.05, 1.12·max(mean+σ))` | Population fraction, bottom fixed at 0 |
| Legend | none | `frameon=False, fontsize=8` |
| Empty case | Explanatory text panel | Still drawn with the unmatched series |

---

## 18. Console output reference

```
Trajectory summary
------------------
Trajectory: /abs/path/traj.gsd
Number of available frames: 500
Valid frame indices: 0 through 499
Selected final frame indices: [450, ..., 499]
Particles in selected frames: minimum=4096, maximum=4096

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
 1.0e-12   ...
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
 index    w    x    y    z    max residual
   ...

Frame-level complete-linkage clustering
---------------------------------------
PARTICLE-ORDER CUTOFF-QUALIFIED CLUSTER-COUNT CHECK: frame 450, trial 1/3: PASSED (4 clusters with size >= 200 in both runs; total raw clusters 9 vs 9).
PARTICLE-ORDER CUTOFF-QUALIFIED CLUSTER-COUNT SUMMARY: frame 450: all 3 trial(s) passed for cluster size >= 200.
[1/50] frame 450: raw clusters=9, clusters with size >= 200=4, largest cluster=1204, maximum validated diameter=40.87 deg, cutoff-qualified-cluster-count-order-stability=passed
...

Reference-state size filtering
------------------------------
Reference-frame raw cluster count: 9
Reference states retained with size >= 200: 4
Reference-frame clusters below cutoff and treated as unmatched: 5

REFERENCE-MEDOID DISTANCE CHECK: PASSED. self floor=0.0123 deg; maximum asymmetry=0.0089 deg.

Fixed-reference tracking tolerance
----------------------------------
Minimum reference-medoid separation: 58.31 deg
To guarantee non-overlapping reference acceptance regions, use tracking tolerance strictly below 29.155 deg.

Final fixed-reference state summary
-----------------------------------
Reference frame: 499
Reference-frame raw cluster count: 9
Tracked reference state count after instantaneous size cutoff: 4
Valid tracked states by mean particle-count cutoff (200): 4
State 0: mean count=1198.3, mean fraction=0.2925, presence=1.000, max diameter=40.9 deg, valid=True
...

Saved outputs
-------------
Symmetry quaternions: ...
states: ...
populations: ...
clusters: ...
membership: ...
references: ...
Mean-population plot: ...
Mean-population plotting CSV: ...
Population time-series plot: ...
Population time-series plotting CSV: ...
Metadata: ...
Particle-order medoid diagnostics: ...
```

**Per-frame log fields:** raw cluster count; cutoff-qualified count; largest cluster size (cluster 0, since sorting is largest-first); maximum validated diameter across all clusters; and stability status, one of `passed` / `failed-but-continued` / `not-tested`.

---

## 19. Internal data structures

All eight are `@dataclass(frozen=True)` — immutable, so no downstream code can mutate a validated result.

### Geometry (Part A)
* **`ConvexDecomposition`** — `vertices` (centred), `edges`, `faces` (cyclically ordered index arrays), `face_equations` ($(F,4)$, unit normals, centred coordinates), `merge_tolerance`, `volume`
* **`SymmetryOperation`** — `quaternion_wxyz` (canonical $+q$, full precision), `rotation_matrix`, `permutation` (the deduplication key), `max_vertex_residual`, `discovery_axis`, `discovery_angle_deg`
* **`SymmetryResult`** — `centered_vertices`, `original_center`, `decomposition`, `physical_operations`, `equivalent_quaternions_wxyz` ($(2n,4)$, the array passed to freud), `matching_tolerance`

### Clustering (Part B)
* **`DistanceDiagnostics`** — `minimum_angle_deg`, `maximum_angle_deg` over all $i<j$ pairs
* **`FrameCluster`** — `frame_index`, `local_cluster_id`, `members` (int64 array), `size`, `medoid_particle_index`, `medoid_quaternion_wxyz`, `diameter_deg`, `maximum_angle_to_medoid_deg`, `medoid_sum_distance_deg`
* **`FrameClustering`** — `frame_index`, `particle_count`, `clusters` (tuple, sorted largest-first), `particle_to_local_cluster`, `maximum_quaternion_norm_deviation`, `distance_diagnostics`, `cluster_count_stability_passed` (`bool | None`)

### Tracking (Parts C–D)
* **`FrameTracking`** — `frame_index`, `cluster_to_reference_state` (tuple, $-1$ = unmatched), `cluster_tracking_distance_deg` (`NaN` = unmatched), `unmatched_cluster_ids`, `absent_reference_state_ids`, `split_candidate_reference_ids`, `ambiguous_current_cluster_ids`
* **`StateSummary`** — `reference_state_id`, `reference_local_cluster_id`, `reference_size`, `reference_medoid_particle_index`, `reference_medoid_quaternion_wxyz`, `mean_particle_count`, `standard_deviation_particle_count`, `mean_population_fraction`, `standard_deviation_population_fraction`, `presence_fraction`, `maximum_tracking_distance_deg`, `mean_tracking_distance_deg_when_present`, `maximum_cluster_diameter_deg`, `valid_by_mean_size_cutoff`

---

## 20. Complete validation and error catalogue

Sixty `raise` sites, grouped by phase.

### Startup and CLI
| Condition | Exception |
|---|---|
| SciPy missing | `SystemExit` at import |
| freud / gsd / matplotlib missing | `ImportError` with install hint |
| `--block-size < 1` | `ValueError` |
| `--angle-validation-tol <= 0` | `ValueError` |
| A CLI value fails its validator | `ValueError` with the prompt's error message |

### Input files
| Condition | Exception |
|---|---|
| GSD absent | `FileNotFoundError` |
| Trajectory has no frames | `ValueError` |
| A selected frame's orientations are not $(N,4)$ | `ValueError` naming the frame |
| Shape JSON absent | `FileNotFoundError` |
| Malformed JSON | `json.JSONDecodeError` |
| No finite $N\times3$, $N\ge4$ array | `ValueError` |
| All vertices at one point | `ValueError` |
| Duplicate vertices | `ValueError` naming an example pair |

### Geometry and symmetry
| Condition | Exception |
|---|---|
| Merged face has zero normal / degenerate polygon / no in-plane basis | `RuntimeError` |
| **No merge tolerance reproduces the topology** | `RuntimeError` with expected vs. final counts |
| Zero axis, or no nonzero candidate axes | `ValueError` / `RuntimeError` |
| `precision_exponent < 0` | `ValueError` |
| Recentring residual $> 100\epsilon$ | `RuntimeError` |
| Incomplete Hungarian assignment | `RuntimeError` |
| Refined rotation has $\det < 0$ | `RuntimeError` |
| **Refinement changed the discovered permutation** | `RuntimeError` — use a tighter $p$ |
| No permutations / identity absent / **missing inverse** / **not closed** | `RuntimeError` |
| Equivalent quaternions not unit-norm, or $q/-q$ pairing inexact | `RuntimeError` |

### Per-frame data and distances
| Condition | Exception |
|---|---|
| Orientations not $(N,4)$, too few, non-finite, or a zero quaternion | `ValueError` naming the frame |
| **freud output shape $\ne$ (block, M)** | `RuntimeError` quoting both shapes |
| freud returned NaN/inf | `RuntimeError` |
| Angle $< -\varepsilon_{\text{valid}}$ or $> 180 + \varepsilon_{\text{valid}}$ | `RuntimeError` |
| **Condensed-distance accounting mismatch** | `RuntimeError` quoting expected vs. written |
| Permuted condensed construction incomplete | `RuntimeError` |
| Condensed indexing on a self-pair | `ValueError` |

### Clustering
| Condition | Exception |
|---|---|
| **Cluster diameter $>$ orientation tolerance $+ 10^{-10}$** | `RuntimeError` with both values |
| Particle assigned to multiple clusters | `RuntimeError` |
| Some particle unassigned | `RuntimeError` |
| Cluster populations do not sum to $M$ | `RuntimeError` |

### Reference and tracking
| Condition | Exception |
|---|---|
| No reference-frame cluster meets the cutoff | `RuntimeError` |
| **Reference-medoid self-distance $> 0.5°$** | `RuntimeError` |
| Reference-medoid matrix asymmetric beyond $\varepsilon_{\text{eff}}$ | `RuntimeError` |
| Reference medoids too close for a meaningful tolerance | `RuntimeError` |
| **Ambiguous match or split candidate** (strict mode) | `RuntimeError` naming the indices |
| Tracking tolerance permits overlapping regions (strict mode) | `RuntimeError` |
| Assignment accepted a distance above cutoff | `RuntimeError` |
| A state matched more than once | `RuntimeError` |
| Tracking does not conserve particles | `RuntimeError` |
| State populations do not conserve $M$ | `RuntimeError` |

### Non-fatal warnings
| Condition | Behaviour |
|---|---|
| Particle-order stability trial failed | `WARNING:` printed, diagnostics written, **run continues** |
| Reference-medoid self-angle above `angle_validation_tol` but below 0.5° | `NOTICE:` printed, diagonal zeroed, run continues |

---

## 21. Complexity, memory and disk model

Let $V$ = shape vertices, $A$ = unique candidate axes, $n$ = detected rotations, $M$ = particles/frame, $F$ = frames, $S$ = states, $K$ = stability trials.

### Symmetry detection (once)

| Stage | Time | Memory |
|---|---|---|
| Convex hull | $O(V\log V)$ | $O(n_{\text{tri}})$ |
| Coplanar merge × 9 tolerances | $O(n_{\text{tri}}^2)$ each | $O(n_{\text{tri}}^2)$ **dense int8 adjacency** |
| Candidate testing | $A\times37\times O(V^3)$ (Hungarian) | $O(V^2)$ |
| Group validation | $O(n^2V)$ | $O(nV)$ |

Typically seconds. The $O(n_{\text{tri}}^2)$ adjacency is the only structure that becomes awkward for a shape with thousands of hull triangles.

### Per frame

| Operation | Time | Peak RAM | Disk |
|---|---|---|---|
| freud distances | $O(M^2 n)$ | $8\cdot\text{block}\cdot M$ B | — |
| Condensed memmap | $O(M^2)$ | negligible | $8\cdot\tfrac{M(M-1)}{2}$ B |
| `linkage` (complete) | $O(M^2)$ | **$8\cdot\tfrac{M(M-1)}{2}$ B** (SciPy copies to RAM) | — |
| Cluster statistics | $\sum_C O(\lvert C\rvert^2)$ | $O(\max\lvert C\rvert)$ | — |
| Stability trials | $K\times$(permute + linkage) | as above | $+8\cdot\tfrac{M(M-1)}{2}$ B |

> **The single most important sizing fact:** although the distance array lives on disk, `scipy.cluster.hierarchy.linkage` converts its input to a contiguous in-memory float64 array. Peak RAM therefore still includes one full condensed copy.

| $M$ | condensed size | disk/frame | linkage RAM |
|---|---|---|---|
| 2 000 | 2.0 M pairs | 16 MB | 16 MB |
| 4 096 | 8.4 M pairs | 67 MB | 67 MB |
| 10 000 | 50 M pairs | 400 MB | 400 MB |
| 20 000 | 200 M pairs | 1.6 GB | 1.6 GB |
| 50 000 | 1.25 G pairs | 10 GB | 10 GB |

**Peak disk** during a frame with stability trials $\approx 2\times$ the per-frame figure (original + one permuted copy). With `--keep-distance-files`, all $F$ files persist: $F \times$ per-frame size.

> **Cluster/HPC note:** the temporary directory is created **inside the output directory** (`tempfile.mkdtemp(dir=output_directory)`), *not* in `/tmp`. Point `--output-dir` at a filesystem with adequate space and adequate I/O bandwidth — a slow network filesystem will bottleneck the memmap writes and the random-access reads during cluster statistics.

### Tracking and summary

| Operation | Time |
|---|---|
| Reference medoid matrix | $O(S^2 n)$ |
| Per-frame matching | $O(C_f S\, n) + O((C_f+S)^3)$ Hungarian |
| Summary | $O(FS)$ |

Negligible compared with the clustering, since $S$ and $C_f$ are small.

**Total** $\approx F\cdot(1+K)\cdot O(M^2 n)$. Doubling $M$ quadruples the runtime.

---

## 22. Determinism and reproducibility

The calculation is deterministic apart from one seeded source of randomness.

**Deterministic by construction:**
* frames = last $F$; particles = first $M$;
* JSON traversal is sorted; candidate axes are built in fixed order;
* symmetry operations sorted by an explicit total key;
* clusters sorted by $(-\text{size},\ \text{medoid index},\ \text{members})$;
* medoid ties broken by smallest particle index;
* state summaries sorted by an explicit total key;
* the Hungarian algorithm returns a global optimum.

**Seeded:** the stability permutations, from `np.random.default_rng(random_seed + 1_000_003 * frame_index)`. The same `--random-seed` gives the same permutations and hence the same diagnostics. The stability check never alters the main result, only the reported pass/fail flag.

**Residual nondeterminism:** BLAS/LAPACK threading inside `linear_sum_assignment` and `align_vectors` at the $10^{-16}$ level, and freud version differences. Neither can change a permutation or a cluster assignment except in astronomically unlikely tie cases.

For a fully reproducible batch run, supply all ten scientific flags plus `--random-seed`. The metadata JSON records every one, so any output traces back to its command line.

---

## 23. Limitations, caveats and gotchas

### 23.1 The symmetry tolerance is absolute, not relative
$10^{-p}$ is in shape-coordinate units. Always check it against your characteristic vertex radius. See §10.2.

### 23.2 The candidate-angle list is finite
Only $C_2, C_3, C_4, C_5, C_6, C_8, C_{10}$ rotations are reachable (the list contains multiples of 180°, 120°, 90°, 72°, 60°, 45° and 36°). A $C_7$, $C_9$ or $C_{12}$ axis is **not representable**, and the closure check will fail — correctly refusing to proceed with a subgroup rather than silently returning one. Extend `historical_candidate_angles_rad()` if your particle needs it.

### 23.3 Only convex shapes
The whole geometric pipeline rests on `ConvexHull`. A non-convex particle's concave features are invisible, so the detected group would be that of its convex hull — potentially strictly larger than the true group, which would under-count distinct orientations.

### 23.4 No positional information
No cutoff, no box, no periodic images, no type filtering. This is a global orientational census. It cannot distinguish two spatially separated domains that share an orientation, and it cannot produce a spatially resolved correlation function. (If you need spatial resolution, join `_particle_membership.csv` against the positions from the GSD file yourself.)

### 23.5 Particle selection is by index
The **first $M$** particles are used. Deterministic and reproducible, but if your GSD file has index-correlated structure — types sorted, initial lattice ordering preserved, or a spatial sort applied at write time — then $M < N$ silently analyses a biased subset. Prefer $M = N$, or verify that index order is uncorrelated with structure.

### 23.6 Complete linkage is greedy
It guarantees the diameter bound but not a globally optimal partition. Different tolerances can produce qualitatively different partitions. This is exactly why the plateau scan (§10.3) and the stability check (§10.7) exist — use both.

### 23.7 The stability check is a *count* check
It compares the number of cutoff-qualified clusters, **not** exact membership. Particles can move between clusters while the count is unchanged. The medoid diagnostics CSV (specifically `medoid_angle_difference_deg`) is where you check whether the clusters are actually the same ones.

### 23.8 The reference frame is privileged
States that exist in other frames but not in the reference frame **can never be discovered**. They appear only as unmatched population. If the unmatched fraction is large and roughly constant, run again with a different reference frame, or choose the reference frame where the state count is largest.

### 23.9 Absent states contribute zeros to the mean
A state present at 400 particles in half the frames and absent in the other half reports $\langle n_s\rangle = 200$ and $p_s = 0.5$. **Always read `presence_fraction` alongside `mean_particle_count`** — the mean alone is ambiguous between "consistently moderate" and "intermittently large".

### 23.10 The cutoff is applied twice, with different meanings
Instantaneous size (for eligibility) and mean size (for validity). A state can pass one and fail the other. See §10.4.

### 23.11 σ is not an error bar
Consecutive MD/MC frames are strongly correlated, so `std_particle_count` measures frame-to-frame variability, not the statistical uncertainty on the mean. For a genuine uncertainty you need decorrelated frames and a block-averaging analysis over `_per_frame_populations.csv`.

### 23.12 `plt.show()` blocks by default
On a headless node without `--no-show` and without a suitable `MPLBACKEND`, the run may hang at the end — *after* all files have been written. Always pass `--no-show` in batch jobs.

### 23.13 `_particle_membership.csv` can be very large
$F\times M$ rows. 100 frames × 20 000 particles = 2 million rows. There is no option to suppress it.

### 23.14 The trajectory handle is never explicitly closed
Released at process exit. Fine for a script, not for library reuse.

### 23.15 The temporary directory lives in the output directory
Not `/tmp`. Plan disk space and I/O bandwidth accordingly (§21).

### 23.16 freud API assumption
The argument order and output shape of `AngularSeparationGlobal.compute` are checked but assumed to follow the documented `(N_orientations, N_global_orientations)` convention. A future freud release that changes this triggers the explicit `RuntimeError` rather than a wrong answer — which is the intended behaviour, but the call would then need updating.

### 23.17 Numeric-only CSV columns
`np.savetxt` with `fmt="%.17g"` writes integer-valued columns as floats (`3.0`, not `3`) and missing statistics as `nan`. Parse accordingly. `_particle_membership.csv` is the exception (`fmt="%d"`).

---

## 24. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `TOPOLOGY CHECK: FAILED` | Wrong `--edges`/`--faces`, or a non-convex shape | Read the printed scan table and use the numbers it reports; check Euler's formula |
| `not closed under composition` / `missing an inverse` | $p$ too tight, or an unreachable rotation order | Loosen $p$ first; if the count is still short, extend the angle list (§23.2) |
| `refinement changed the discovered vertex permutation` | $p$ too loose | Increase $p$ |
| Only the identity rotation detected | $p$ far too tight, noisy coordinates, or genuinely no symmetry | Scan $p$ and look for the plateau (§10.2) |
| **Every particle is a singleton** | Orientation tolerance far too small | Increase it; use the printed `maximum_pair_angle_deg` for scale |
| **All particles in one cluster** | Orientation tolerance too large | Decrease it toward the histogram minimum (§10.3) |
| `cluster generated by complete linkage has diameter ... exceeding` | Should be impossible; indicates a SciPy/numerical anomaly | Report it; check the SciPy version and that all distances are finite |
| `Reference frame contains no cluster with size >= cutoff` | Cutoff too large, or tolerance so small everything is tiny | Lower `--cluster-size-cutoff`, or raise the orientation tolerance |
| `Reference-medoid self-distances are too large` | freud numerical precision problem | Investigate the freud build; do not simply raise the constant |
| `Reference-medoid distance matrix is not symmetric` | Numerical floor larger than expected | Raise `--angle-validation-tol` modestly, keeping it far below any physical angle |
| `too close to define a numerically meaningful ... tolerance` | Two reference states are nearly identical | Reduce the orientation tolerance so they do not merge, or accept that they are one state |
| `permits overlapping reference acceptance regions` | Tracking tolerance $\ge$ half the minimum separation | Accept the suggested value, or pass `--allow-overlapping-tracking-regions` deliberately |
| `found an ambiguous match or a split candidate` | Tracking tolerance too large, or a state genuinely split | Reduce the tolerance; inspect `_frame_clusters.csv`; or pass `--allow-ambiguous-tracking` |
| **Large `unmatched_fraction`** | Cutoff too high, tracking tolerance too tight, or states absent from the reference frame | Check `raw clusters` vs `clusters >= cutoff` in the log; try a different reference frame |
| `WARNING: PARTICLE-ORDER ... CHECK FAILED` | Tolerance sitting on a knife edge where ties matter | Move the tolerance onto a plateau; inspect the diagnostics CSV |
| `MemoryError` during linkage | $M$ too large for RAM (§21) | Reduce `--particles`; `--block-size` will **not** help here |
| Runs out of disk | Condensed distance files | Reduce `--particles`, drop `--keep-distance-files`, or point `--output-dir` at a bigger filesystem |
| Hangs at the end on a cluster | `plt.show()` with no display | Pass `--no-show` |
| Batch job hangs at the start | A scientific flag was omitted, so it is waiting at a prompt | Supply all ten scientific flags |
| `ERROR: <message>`, exit 1 | Controlled validation failure | Read the message; every one names the failing quantity |

---

## 25. Function-by-function index

| Function | Line | Role |
|---|---|---|
| `prompt_value` | 143 | Loop until valid terminal input; empty line = default |
| `resolve_or_prompt` | 169 | CLI value if supplied (validated, hard-fail), else prompt |
| `_as_vertex_array` | 187 | Test for a finite $(N\ge4,3)$ float array |
| `_collect_vertex_candidates` | 205 | Deterministic recursive JSON search with key-name priority |
| `read_shape_vertices` | 241 | Load, select, validate, duplicate-check the vertices |
| `_merge_coplanar_hull_triangles` | 283 | Qhull → coplanar merge → polygonal faces, unit planes, edge set |
| `find_validated_convex_decomposition` | 373 | Tolerance ladder $10^{-12}\ldots10^{-4}$; hard topology gate |
| `_canonical_axis_line` | 441 | Unit-normalise and fix the $\pm$ sign of an axis line |
| `build_candidate_axis_lines` | 461 | Four axis classes; zero-removal; parallel/antiparallel dedup |
| `historical_candidate_angles_rad` | 551 | The fixed 37-angle list, in radians |
| `_rotation_to_wxyz` | 569 | SciPy `[x,y,z,w]` → freud `[w,x,y,z]`, normalised, sign-canonical |
| `_one_to_one_vertex_mapping` | 586 | Distance matrix + Hungarian assignment + residuals |
| `_refine_rotation_for_permutation` | 631 | Least-squares proper rotation via `align_vectors`, $\det>0$ guard |
| `_compose_permutations` | 669 | $(\sigma_2\circ\sigma_1)[i] = \sigma_2[\sigma_1[i]]$ |
| `validate_permutation_group` | 687 | Exact identity / inverse / closure checks |
| `detect_proper_rotational_symmetries` | 751 | The 13-stage symmetry pipeline → `SymmetryResult` |
| `import_runtime_packages` | 1072 | Lazy freud / gsd.hoomd / matplotlib import |
| `open_gsd_trajectory` | 1088 | Path resolution + keyword/positional `open` fallback |
| `validate_and_normalize_orientations` | 1104 | Shape/finiteness/norm checks, first-$M$ slice, renormalisation |
| `save_symmetry_outputs` | 1162 | $\pm q$ symmetry CSV |
| `canonicalize_quaternion_sign` | 1265 | Normalise and fix the sign of a quaternion |
| `condensed_index` | 1279 | Vectorised $(i,j) \to$ SciPy condensed index |
| `angular_distance_matrix_deg` | 1299 | One freud call → validated dense degree matrix |
| `compute_condensed_distance_memmap` | 1332 | Block-wise freud → on-disk condensed array, with range checks and accounting |
| `cluster_distance_statistics` | 1539 | Medoid, diameter, max-to-medoid, medoid sum, in one pass |
| `build_permuted_condensed_distances` | 1612 | Reindex a condensed array under a particle permutation |
| `complete_linkage_labels` | 1642 | `linkage(method="complete")` + `fcluster(criterion="distance")` |
| `_alphabetic_group_name` | 1697 | 0→A, 25→Z, 26→AA |
| `_cluster_medoid_records_from_labels` | 1711 | Deterministic cluster/medoid records in original particle IDs |
| `_append_particle_order_medoid_diagnostics` | 1775 | Append one trial's rows to the diagnostics CSV |
| `validate_partition_order_stability` | 1815 | Permutation trials, cutoff-qualified count comparison, medoid matching |
| `construct_frame_clusters` | 2118 | Diameter gate, deterministic ordering, conservation checks |
| `cluster_one_frame` | 2239 | Orchestrates load → distances → linkage → stability → construct → cleanup |
| `build_cutoff_qualified_reference` | 2322 | Filter reference clusters to those meeting the size cutoff |
| `reference_medoid_distance_matrix` | 2355 | Self-angle floor, diagonal zeroing, symmetry validation |
| `tracking_cutoff_suggestion` | 2445 | Suggested and strict-safe tracking tolerances |
| `globally_match_clusters_to_reference_states` | 2472 | Validity mask, ambiguity detection, padded Hungarian assignment |
| `track_all_frames_to_fixed_reference` | 2580 | Per-frame eligibility, matching, scatter-back, conservation |
| `summarize_fixed_reference_states` | 2749 | Count/fraction matrices, per-state statistics, sorting |
| `save_cluster_and_tracking_outputs` | 2880 | The five main CSVs |
| `save_plots` | 3060 | Two PNGs plus their exact companion plotting CSVs |
| `save_analysis_metadata` | 3378 | Complete run-record JSON |
| `build_argument_parser` | 3551 | The CLI definition |
| `main` | 3607 | The 10-stage workflow inside one error boundary |

---

## Citation and provenance note

If you publish results from this program, record in your methods section:

* the **detected proper rotation group order** $n$ and that it is stable under variation of $p$ (metadata: `physical_proper_rotation_count`, `symmetry_precision_exponent`);
* the **orientation angle tolerance**, how it was chosen, and the **width of the plateau** over which the state count is unchanged;
* the **cluster size cutoff** and its three roles;
* the **reference frame** and confirmation that the results are insensitive to that choice;
* the **tracking tolerance** and the minimum reference-medoid separation (`nonoverlapping_tracking_safe_upper_bound_deg`);
* the **particle-order stability outcome** (`cluster_count_stability_passed` per frame) and the number of trials;
* the **absent-state convention** — zero population included in the means — together with `presence_fraction`;
* $M$, $F$, and the **unmatched fraction**.

Every one of these is recorded verbatim in `_metadata.json`.
