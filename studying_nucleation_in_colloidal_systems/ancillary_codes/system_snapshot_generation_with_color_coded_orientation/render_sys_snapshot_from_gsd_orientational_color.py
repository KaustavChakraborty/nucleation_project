#!/usr/bin/env python3
"""
Standalone orientational-cluster snapshot renderer for HOOMD GSD trajectories.

PURPOSE
=======
Read
    1. one HOOMD-schema GSD trajectory,
    2. one convex-polyhedron JSON shape file containing an N x 3 vertex array,
    3. one JSON parameter file,

then:
    * read exactly one requested trajectory frame;
    * infer the proper rotational symmetry group of the particle directly from
      the shape vertices (no project-specific helper modules are required);
    * compute symmetry-reduced orientational separations with
      freud.environment.AngularSeparationGlobal;
    * discover orientational clusters using the supplied
      ``orientation_angle_tol``;
    * retain the statistically relevant clusters, sort them by population,
      and assign every particle to the nearest retained orientation reference;
    * color particles in the same orientational cluster identically;
    * render a publication-quality full-system snapshot with Plato's Fresnel
      backend;
    * render a second, spatially zoomed snapshot of a selected section, with particles only by default (no zoom box);
    * save per-particle and per-cluster CSV files for reproducibility.

The clustering follows the scientific intent of the supplied ref_frame_calc.py:
within a provisional cluster, a new particle is accepted only when its
symmetry-reduced angular separation from EVERY current member is no larger
than ``orientation_angle_tol``.  A pairwise angle matrix is computed once in
blocks, which is substantially more efficient than repeatedly invoking freud
inside Python loops.

The final coloring follows the supplied color_particles.py idea: each particle
is assigned to whichever retained reference orientation has the smallest
symmetry-reduced angular separation.  Therefore every particle receives a
cluster/color even when a small provisional cluster was removed by
``cluster_size_cutoff``.

RUN
===
    python render_gsd_orientational_clusters.py snapshot_cluster_param.json

DEPENDENCIES
============
Recommended conda-forge installation:

    mamba create -n orient_render -c conda-forge \
        python=3.10 numpy scipy matplotlib gsd freud fresnel
    mamba activate orient_render
    pip install plato-draw

Notes
-----
* HOOMD/GSD particle quaternions are expected in scalar-first order
  [w, x, y, z].
* Only proper rotations (SO(3)) are used as particle symmetries. Reflections
  are deliberately excluded because quaternions represent rotations, not
  improper O(3) operations.
* The shape must be a genuinely three-dimensional convex polyhedron.
* ``frame_index`` uses ordinary Python/GSD indexing. Thus -1 is the last frame.
"""

# [DOCUMENTATION] HOW TO READ THIS FILE: this is the original working renderer with explanatory comments added
# [DOCUMENTATION] only. The numerical algorithms, control flow, parameter names, defaults, function calls,
# [DOCUMENTATION] output names, and rendering settings below are unchanged.
# [DOCUMENTATION] The workflow has six scientific stages: (1) read configuration and shape vertices, (2) infer
# [DOCUMENTATION] the particle proper-rotation symmetry group, (3) read one GSD frame, (4) compute
# [DOCUMENTATION] symmetry-reduced pairwise orientation angles, (5) identify/retain orientational clusters and
# [DOCUMENTATION] assign colors, and (6) render/save the full and zoomed snapshots plus diagnostic tables.
# [DOCUMENTATION] Quaternion convention used throughout is scalar-first [w, x, y, z], matching the convention
# [DOCUMENTATION] expected by the trajectory handling in this script. SciPy Rotation internally uses [x, y, z,
# [DOCUMENTATION] w], so explicit component reordering is performed whenever data cross that boundary.
from __future__ import annotations

# [DOCUMENTATION] STANDARD-LIBRARY IMPORTS: argparse provides the command-line interface; csv writes
# [DOCUMENTATION] reproducibility tables; itertools generates combinations/permutations and box corners; json
# [DOCUMENTATION] reads parameters and writes metadata; math/sys provide numerical/system utilities; dataclass
# [DOCUMENTATION] defines compact immutable result containers; pathlib handles robust filesystem paths; typing
# [DOCUMENTATION] documents accepted data types.
import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

# [DOCUMENTATION] NumPy supplies all array storage, vectorized geometry, linear algebra interfaces, masking, and
# [DOCUMENTATION] compact float32 storage for the potentially large N x N orientation-angle matrix.
import numpy as np

# [DOCUMENTATION] SciPy is required by the geometry/symmetry layer. linear_sum_assignment performs optimal
# [DOCUMENTATION] one-to-one vertex matching, ConvexHull validates the 3D particle geometry, and Rotation handles
# [DOCUMENTATION] matrix/quaternion/Euler conversions plus full-set rotation refinement.
try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise SystemExit(
        "SciPy is required. Install numpy/scipy before running this program."
    ) from exc


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
# [DOCUMENTATION] The following dataclasses are immutable containers. Freezing them prevents accidental
# [DOCUMENTATION] reassignment of result fields after a scientific result has been constructed, which makes
# [DOCUMENTATION] later stages easier to reason about.
@dataclass(frozen=True)
# [DOCUMENTATION] SymmetryResult bundles everything produced by the shape-symmetry search. centered_vertices are
# [DOCUMENTATION] the body-frame vertices translated to zero centroid; original_center records the removed
# [DOCUMENTATION] centroid; physical_quaternions_wxyz stores one quaternion per distinct proper rotation;
# [DOCUMENTATION] equivalent_quaternions_wxyz stores both q and -q for each physical rotation for freud;
# [DOCUMENTATION] permutations encode how each rotation permutes vertices; max_residuals quantify the worst
# [DOCUMENTATION] vertex-matching error for each accepted operation; matching_tolerance records the acceptance
# [DOCUMENTATION] threshold actually used.
class SymmetryResult:
    centered_vertices: np.ndarray
    original_center: np.ndarray
    physical_quaternions_wxyz: np.ndarray
    equivalent_quaternions_wxyz: np.ndarray
    permutations: tuple[tuple[int, ...], ...]
    max_residuals: np.ndarray
    matching_tolerance: float


@dataclass(frozen=True)
# [DOCUMENTATION] ClusterResult stores both the discovery-stage grouping and the final coloring assignment.
# [DOCUMENTATION] provisional_clusters are the greedy complete-link-like groups; retained_reference_indices are
# [DOCUMENTATION] representative particle indices that survive the size cutoff; final_cluster_ids gives one
# [DOCUMENTATION] retained color/cluster ID for every particle; nearest_reference_angles_deg stores each
# [DOCUMENTATION] particle's symmetry-reduced angle to its selected reference; retained_seed_sizes records the
# [DOCUMENTATION] original discovery sizes before the final nearest-reference reassignment.
class ClusterResult:
    provisional_clusters: tuple[np.ndarray, ...]
    retained_reference_indices: np.ndarray
    final_cluster_ids: np.ndarray
    nearest_reference_angles_deg: np.ndarray
    retained_seed_sizes: np.ndarray


@dataclass(frozen=True)
# [DOCUMENTATION] RuntimePackages keeps optional/heavy runtime dependencies together after successful import:
# [DOCUMENTATION] GSD for trajectory I/O, freud for symmetry-reduced angular separation, Plato/Fresnel for
# [DOCUMENTATION] rendering, and matplotlib.to_rgba for converting hex colors to numerical RGBA values.
class RuntimePackages:
    gsd_hoomd: Any
    freud: Any
    draw: Any
    to_rgba: Any


# -----------------------------------------------------------------------------
# Generic configuration helpers
# -----------------------------------------------------------------------------
# [DOCUMENTATION] CONFIGURATION HELPER. Opens a UTF-8 JSON file and requires the top-level object to be a
# [DOCUMENTATION] dictionary. Returning a dictionary is important because later code expects named keys such as
# [DOCUMENTATION] gsd_file, shape_file, render, and zoom.
def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


# [DOCUMENTATION] PATH HELPER. Expands a leading ~, interprets relative paths relative to the parameter-file
# [DOCUMENTATION] directory rather than the current shell directory, and finally resolves the result to an
# [DOCUMENTATION] absolute canonical path. This lets the parameter JSON be moved together with its input files
# [DOCUMENTATION] without depending on where Python was launched.
def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


# [DOCUMENTATION] REQUIRED-PARAMETER HELPER. Unlike dict.get, this deliberately raises a clear error when a
# [DOCUMENTATION] scientifically essential parameter is absent. It is used for values for which silently
# [DOCUMENTATION] applying an arbitrary default would be undesirable.
def require_key(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise KeyError(f"Required parameter {key!r} is missing from the parameter file.")
    return data[key]


# [DOCUMENTATION] VALIDATION HELPER. Converts a user-supplied three-component quantity to float64 and rejects
# [DOCUMENTATION] wrong lengths or NaN/Inf values. It is reused for vectors such as the camera Euler angles,
# [DOCUMENTATION] directional light, zoom center, and zoom size.
def as_float_triplet(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain exactly three finite numbers.")
    return arr


# [DOCUMENTATION] COLOR-PALETTE HELPER. Ensures that cmap_original is at least a non-empty list, then normalizes
# [DOCUMENTATION] each entry to a string. Actual color parsing is deferred to matplotlib.to_rgba during
# [DOCUMENTATION] rendering.
def ensure_hex_palette(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError("cmap_original must be a non-empty JSON list of colors.")
    palette = [str(item) for item in value]
    return palette


# -----------------------------------------------------------------------------
# Runtime imports kept separate so the geometry/symmetry code can be inspected
# on systems where rendering packages are not installed.
# -----------------------------------------------------------------------------
# [DOCUMENTATION] RUNTIME-DEPENDENCY LOADER. Heavy/optional packages are imported here instead of at module
# [DOCUMENTATION] import time. This separation allows the geometry and symmetry code to be inspected on systems
# [DOCUMENTATION] where GSD/Freud/Plato are not installed, and it lets the program report all missing packages
# [DOCUMENTATION] together in one readable error.
def import_runtime_packages() -> RuntimePackages:
    # [DOCUMENTATION] Each import failure appends a human-readable package name to this list. The function does
    # [DOCUMENTATION] not immediately abort on the first missing dependency, so the user gets a complete
    # [DOCUMENTATION] installation diagnosis.
    missing: list[str] = []

    try:
        import gsd.hoomd as gsd_hoomd
    except ImportError:
        gsd_hoomd = None
        missing.append("gsd")

    try:
        import freud
    except ImportError:
        freud = None
        missing.append("freud-analysis / freud")

    try:
        import plato.draw.fresnel as draw
    except ImportError:
        draw = None
        missing.append("plato-draw")

    try:
        from matplotlib.colors import to_rgba
    except ImportError:
        to_rgba = None
        missing.append("matplotlib")

    # [DOCUMENTATION] If any runtime dependency was unavailable, execution stops before reading scientific data.
    # [DOCUMENTATION] The installation recipe printed here is informational only; it does not mutate the
    # [DOCUMENTATION] environment.
    if missing:
        raise SystemExit(
            "Missing required runtime package(s): " + ", ".join(missing) + "\n\n"
            "Recommended installation:\n"
            "  mamba create -n orient_render -c conda-forge "
            "python=3.10 numpy scipy matplotlib gsd freud fresnel\n"
            "  mamba activate orient_render\n"
            "  pip install plato-draw\n"
        )

    # [DOCUMENTATION] After all imports succeed, package/module objects are returned explicitly instead of
    # [DOCUMENTATION] relying on hidden globals. The main workflow then passes them into functions that need
    # [DOCUMENTATION] them.
    return RuntimePackages(
        gsd_hoomd=gsd_hoomd,
        freud=freud,
        draw=draw,
        to_rgba=to_rgba,
    )


# -----------------------------------------------------------------------------
# Shape JSON reader
# -----------------------------------------------------------------------------
# [DOCUMENTATION] SHAPE-CANDIDATE TEST. Attempts to interpret an arbitrary JSON value as a finite numeric N x 3
# [DOCUMENTATION] array. A valid candidate must have at least four 3D points; this is only a structural test,
# [DOCUMENTATION] not yet a proof that the points form a nondegenerate 3D convex polyhedron.
def _as_vertices(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if (
        arr.ndim == 2
        and arr.shape[1] == 3
        and arr.shape[0] >= 4
        and np.all(np.isfinite(arr))
    ):
        return arr
    return None


# [DOCUMENTATION] RECURSIVE SHAPE SEARCH. Walks dictionaries/lists in the shape JSON and collects every object
# [DOCUMENTATION] that looks like an N x 3 vertex array. This makes the reader tolerant of different JSON
# [DOCUMENTATION] layouts instead of hard-coding a single key such as "8_vertices".
# [DOCUMENTATION] Each candidate receives a priority. Arrays found under a path containing the word "vert" get a
# [DOCUMENTATION] very large bonus, while the number of rows is a secondary preference. The path string is
# [DOCUMENTATION] retained so the selected field can be reported to the user.
def _collect_vertex_candidates(
    obj: Any,
    key_hint: str = "",
) -> list[tuple[int, str, np.ndarray]]:
    candidates: list[tuple[int, str, np.ndarray]] = []

    # [DOCUMENTATION] First test the current JSON node itself. If it already is an N x 3 numerical array,
    # [DOCUMENTATION] recursion stops at this node and the candidate is recorded.
    direct = _as_vertices(obj)
    if direct is not None:
        priority = (100000 if "vert" in key_hint.lower() else 0) + len(direct)
        candidates.append((priority, key_hint, direct))
        return candidates

    # [DOCUMENTATION] If the current node is a dictionary, recursively inspect every value while extending
    # [DOCUMENTATION] key_hint with the dictionary path. Lists/tuples are traversed similarly with explicit
    # [DOCUMENTATION] [index] notation.
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_hint = f"{key_hint}/{key}" if key_hint else str(key)
            candidates.extend(_collect_vertex_candidates(value, child_hint))
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            child_hint = f"{key_hint}[{idx}]"
            candidates.extend(_collect_vertex_candidates(value, child_hint))

    return candidates


# [DOCUMENTATION] SHAPE READER. Loads the JSON, discovers candidate vertex arrays, chooses the highest-priority
# [DOCUMENTATION] candidate, then performs two geometry sanity checks before the vertices are accepted for
# [DOCUMENTATION] symmetry analysis and rendering.
def read_shape_vertices(shape_file: Path) -> np.ndarray:
    data = load_json(shape_file)
    candidates = _collect_vertex_candidates(data)
    if not candidates:
        raise ValueError(
            f"Could not locate a finite N x 3 vertex array in {shape_file}."
        )

    # [DOCUMENTATION] Select the candidate with the largest priority score. Because likely vertex keys receive a
    # [DOCUMENTATION] 100000-point bonus, a clearly named vertex array wins over unrelated numerical tables in
    # [DOCUMENTATION] normal shape files.
    _, selected_key, vertices = max(candidates, key=lambda item: item[0])
    vertices = np.asarray(vertices, dtype=np.float64)

    # [DOCUMENTATION] Center the vertex cloud conceptually and require rank 3. Rank < 3 would mean all points
    # [DOCUMENTATION] lie in a plane/line/point, which cannot define the intended 3D convex particle.
    if np.linalg.matrix_rank(vertices - vertices.mean(axis=0)) < 3:
        raise ValueError("The supplied vertices do not span a 3D polyhedron.")

    # ConvexHull is also a useful validity check and confirms nonzero volume.
    # [DOCUMENTATION] Construct a SciPy/Qhull convex hull as a second validity check. The renderer later uses
    # [DOCUMENTATION] the raw vertices, but a positive finite hull volume confirms that they describe a genuine
    # [DOCUMENTATION] 3D convex body.
    hull = ConvexHull(vertices)
    if not np.isfinite(hull.volume) or hull.volume <= 0.0:
        raise ValueError("The supplied shape has zero/invalid convex-hull volume.")

    print("\nShape")
    print("-----")
    print(f"File: {shape_file}")
    print(f"Selected vertex field: {selected_key}")
    print(f"Number of vertices: {len(vertices)}")
    print(f"Convex-hull volume: {hull.volume:.12g}")
    return vertices


# -----------------------------------------------------------------------------
# Proper rotational symmetry detection directly from the vertex set
# -----------------------------------------------------------------------------
# [DOCUMENTATION] SYMMETRY HELPER: choose three body-frame vertex vectors that form the most numerically stable
# [DOCUMENTATION] 3D basis. For a 3 x 3 matrix whose columns are those vectors, |det| measures the spanned
# [DOCUMENTATION] parallelepiped volume; maximizing it avoids nearly coplanar triples and reduces numerical
# [DOCUMENTATION] amplification when the matrix is inverted.
def _best_conditioned_source_triple(centered_vertices: np.ndarray) -> tuple[int, int, int]:
    """Choose a linearly independent vertex triple with maximal |det|."""
    best_det = -1.0
    best_triple: tuple[int, int, int] | None = None

    # [DOCUMENTATION] Enumerate every unordered three-vertex combination, form a 3 x 3 basis matrix, and keep
    # [DOCUMENTATION] the triple with the largest absolute determinant.
    for triple in itertools.combinations(range(len(centered_vertices)), 3):
        matrix = centered_vertices[list(triple)].T
        det = abs(float(np.linalg.det(matrix)))
        if det > best_det:
            best_det = det
            best_triple = tuple(int(i) for i in triple)

    if best_triple is None or best_det <= 1.0e-12:
        raise RuntimeError("Could not find a stable linearly independent vertex triple.")
    return best_triple


# [DOCUMENTATION] SYMMETRY HELPER: convert a 3 x 3 proper-rotation matrix into one deterministic quaternion
# [DOCUMENTATION] representative. SciPy returns [x,y,z,w]; the script reorders to [w,x,y,z], normalizes, then
# [DOCUMENTATION] fixes the mathematically irrelevant q <-> -q sign ambiguity for reproducible output
# [DOCUMENTATION] ordering/storage.
def _canonical_wxyz(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to one deterministic scalar-first quaternion."""
    q_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
    q = np.asarray(q_xyzw[[3, 0, 1, 2]], dtype=np.float64)
    q /= np.linalg.norm(q)

    # q and -q represent the same physical rotation. Choose a deterministic sign.
    # [DOCUMENTATION] Find the first quaternion component that is numerically nonzero. If it is negative, flip
    # [DOCUMENTATION] all four signs. q and -q encode the same rotation, so this changes no physics; it simply
    # [DOCUMENTATION] gives each physical rotation a stable canonical representative.
    for component in q:
        if abs(component) > 1.0e-14:
            if component < 0.0:
                q = -q
            break
    return q


# [DOCUMENTATION] SYMMETRY HELPER: after applying a candidate rotation, find the globally optimal one-to-one
# [DOCUMENTATION] correspondence between rotated vertices and original reference vertices. A full pairwise
# [DOCUMENTATION] Euclidean distance matrix is built and the Hungarian algorithm (linear_sum_assignment)
# [DOCUMENTATION] minimizes total assignment cost.
# [DOCUMENTATION] The returned permutation is a discrete fingerprint of the symmetry operation; residuals are
# [DOCUMENTATION] the individual vertex-matching distances used to reject inaccurate candidates.
def _best_vertex_assignment(
    rotated_vertices: np.ndarray,
    reference_vertices: np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray]:
    # [DOCUMENTATION] Broadcasting constructs costs[i,j] = distance from rotated vertex i to reference vertex j.
    # [DOCUMENTATION] The assignment solver then prohibits reusing the same target vertex for multiple source
    # [DOCUMENTATION] vertices.
    costs = np.linalg.norm(
        rotated_vertices[:, None, :] - reference_vertices[None, :, :],
        axis=2,
    )
    row_indices, col_indices = linear_sum_assignment(costs)
    permutation = np.empty(len(reference_vertices), dtype=np.int64)
    permutation[row_indices] = col_indices
    residuals = costs[row_indices, col_indices]
    return tuple(int(x) for x in permutation), residuals


# [DOCUMENTATION] GROUP-CONSISTENCY CHECK. A detected symmetry set must contain identity, every operation must
# [DOCUMENTATION] actually be a permutation of all vertex indices, and the set must be closed under composition.
# [DOCUMENTATION] Failure indicates an inconsistent symmetry tolerance or a numerical/geometry problem.
def _validate_permutation_group(permutations: Iterable[tuple[int, ...]]) -> None:
    permutations = tuple(permutations)
    if not permutations:
        raise RuntimeError("No symmetry permutations were detected.")

    n = len(permutations[0])
    identity = tuple(range(n))
    group = set(permutations)
    if identity not in group:
        raise RuntimeError("Detected symmetry operations do not contain identity.")

    # [DOCUMENTATION] First validate that each tuple contains every vertex index exactly once. Later, nested
    # [DOCUMENTATION] loops test closure: composing any two accepted vertex permutations must produce another
    # [DOCUMENTATION] accepted permutation.
    for p in permutations:
        if tuple(sorted(p)) != identity:
            raise RuntimeError("A detected symmetry operation is not a permutation.")

    # Composition convention: first p, then q -> q[p[i]].
    # [DOCUMENTATION] Now test group closure explicitly. For every ordered pair of accepted symmetry
    # [DOCUMENTATION] permutations p and q, construct their composition and require that the composed permutation
    # [DOCUMENTATION] is itself present in the detected symmetry set.
    for p in permutations:
        for q in permutations:
            composed = tuple(q[p[i]] for i in range(n))
            if composed not in group:
                raise RuntimeError(
                    "Detected proper rotations fail the group-closure test. "
                    "Reduce/increase symmetry_vertex_tolerance carefully."
                )


# [DOCUMENTATION] CORE PARTICLE-SYMMETRY SEARCH. This function determines all proper rotations R in SO(3) that
# [DOCUMENTATION] map the complete centered vertex set onto itself within matching_tolerance. Improper
# [DOCUMENTATION] operations such as mirrors/inversion are intentionally excluded because particle orientations
# [DOCUMENTATION] are represented by unit quaternions.
# [DOCUMENTATION] The key idea is that a noncoplanar source triple fixes a linear map. Any true symmetry must
# [DOCUMENTATION] send that triple to another ordered triple with identical internal dot products (same Gram
# [DOCUMENTATION] matrix). Candidate maps passing this cheap geometric filter are projected to the nearest
# [DOCUMENTATION] orthogonal matrix and then verified against every vertex using one-to-one assignment.
def detect_proper_rotational_symmetry(
    vertices: np.ndarray,
    matching_tolerance: float,
) -> SymmetryResult:
    """Infer every proper rotation that maps the complete vertex set onto itself.

    Strategy
    --------
    A well-conditioned ordered triple of centered vertices fixes a 3D linear
    transformation. Every symmetry must map that source triple to another
    ordered triple with the same Gram matrix. Candidate mappings are projected
    to the nearest proper rotation and accepted only when ALL vertices can be
    matched one-to-one to the original vertex set within matching_tolerance.

    The final operations are deduplicated by the induced vertex permutation,
    which is a discrete and robust identity for each symmetry operation.
    """
    # [DOCUMENTATION] Translate the shape to its centroid before searching for rotational symmetries. Pure
    # [DOCUMENTATION] rotations of a rigid body should be tested about the particle center, not about an
    # [DOCUMENTATION] arbitrary coordinate origin stored in the JSON.
    original_center = np.mean(vertices, axis=0)
    centered = np.asarray(vertices - original_center, dtype=np.float64)
    n = len(centered)
    radius = float(np.max(np.linalg.norm(centered, axis=1)))
    if radius <= 0.0:
        raise ValueError("Degenerate shape: all vertices are at the center.")

    # [DOCUMENTATION] Choose and invert the stable source basis once. source_gram stores all pairwise dot
    # [DOCUMENTATION] products of the three source vectors; these dot products are invariant under an exact
    # [DOCUMENTATION] rotation.
    source_indices = _best_conditioned_source_triple(centered)
    source = centered[list(source_indices)].T  # columns are source vectors
    source_inv = np.linalg.inv(source)
    source_gram = source.T @ source

    # A Gram prefilter greatly reduces candidate SVD/assignment work while being
    # deliberately looser than the final vertex residual criterion.
    # [DOCUMENTATION] Define a deliberately loose Gram-matrix prefilter. It is not the final symmetry criterion;
    # [DOCUMENTATION] it only avoids expensive SVD and Hungarian assignment work for obviously incompatible
    # [DOCUMENTATION] ordered target triples. Final acceptance still uses the stricter all-vertex residual
    # [DOCUMENTATION] threshold.
    gram_tolerance = max(10.0 * matching_tolerance * max(radius, 1.0), 1.0e-10)

    operations: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    candidates_tested = 0
    gram_hits = 0

    # [DOCUMENTATION] Enumerate ORDERED target triples because vertex A->i, B->j, C->k defines a different
    # [DOCUMENTATION] candidate map from a different ordering of the same three targets. Every possible image of
    # [DOCUMENTATION] the source basis is therefore considered.
    for target_indices in itertools.permutations(range(n), 3):
        candidates_tested += 1
        target = centered[list(target_indices)].T

        # [DOCUMENTATION] Reject target triples whose Gram matrix differs from the source. A rotation preserves
        # [DOCUMENTATION] lengths and mutual angles, so matching Gram matrices are a necessary condition for a
        # [DOCUMENTATION] rotational symmetry.
        if not np.allclose(
            target.T @ target,
            source_gram,
            atol=gram_tolerance,
            rtol=0.0,
        ):
            continue
        gram_hits += 1

        # [DOCUMENTATION] Construct the linear transformation that sends the three source basis vectors to the
        # [DOCUMENTATION] chosen target vectors. Numerical noise can make raw_map slightly non-orthogonal even
        # [DOCUMENTATION] for a true symmetry.
        raw_map = target @ source_inv

        # Improper source->target maps are not quaternion rotations.
        # [DOCUMENTATION] Discard maps with nonpositive determinant before quaternion conversion. det < 0
        # [DOCUMENTATION] corresponds to an improper transformation (reflection-containing operation), while the
        # [DOCUMENTATION] orientation space here is SO(3).
        if np.linalg.det(raw_map) <= 0.0:
            continue

        # Project numerical raw_map to the closest orthogonal matrix.
        # [DOCUMENTATION] Use the polar/Kabsch-style SVD projection U V^T to obtain the closest orthogonal
        # [DOCUMENTATION] matrix to raw_map. This removes small numerical distortions introduced by
        # [DOCUMENTATION] finite-precision vertices.
        u, _, vt = np.linalg.svd(raw_map)
        rotation_matrix = u @ vt
        if np.linalg.det(rotation_matrix) <= 0.0:
            continue

        # [DOCUMENTATION] Apply the candidate rotation to every centered vertex. The subsequent Hungarian
        # [DOCUMENTATION] assignment asks whether the entire rotated point set can be matched one-to-one onto
        # [DOCUMENTATION] the original set within tolerance.
        rotated = centered @ rotation_matrix.T
        permutation, residuals = _best_vertex_assignment(rotated, centered)
        max_residual = float(np.max(residuals))
        # [DOCUMENTATION] This is the first full-shape acceptance test. Even if a candidate maps the chosen
        # [DOCUMENTATION] three source vertices correctly, it is rejected unless all vertices simultaneously
        # [DOCUMENTATION] match the original shape within matching_tolerance.
        if max_residual > matching_tolerance:
            continue

        # Full-set Kabsch refinement of the accepted discrete permutation.
        # [DOCUMENTATION] For accepted discrete vertex mapping, perform a full-set Kabsch refinement using all
        # [DOCUMENTATION] matched vertex pairs. This improves the rotation estimate beyond the original
        # [DOCUMENTATION] three-point construction and then rechecks that the discrete permutation remains
        # [DOCUMENTATION] unchanged.
        targets = centered[np.asarray(permutation, dtype=np.int64)]
        refined_rotation, _ = Rotation.align_vectors(targets, centered)
        refined_matrix = refined_rotation.as_matrix()
        refined_rotated = centered @ refined_matrix.T
        refined_perm, refined_residuals = _best_vertex_assignment(
            refined_rotated,
            centered,
        )
        refined_max = float(np.max(refined_residuals))

        if refined_perm != permutation or refined_max > matching_tolerance:
            continue

        # [DOCUMENTATION] Multiple ordered target triples can rediscover the same physical symmetry. The induced
        # [DOCUMENTATION] vertex permutation is used as a robust deduplication key; if rediscovered, only the
        # [DOCUMENTATION] version with the smaller maximum residual is retained.
        previous = operations.get(permutation)
        if previous is None or refined_max < previous[1]:
            operations[permutation] = (refined_matrix, refined_max)

    if not operations:
        raise RuntimeError(
            "No proper rotational symmetry operation was detected. "
            "Check the shape vertices and symmetry_vertex_tolerance."
        )

    # Stable ordering: identity-like rotation first, then permutation tuple.
    # [DOCUMENTATION] Sort accepted operations deterministically, placing the identity-like matrix first and
    # [DOCUMENTATION] using the permutation tuple as a stable secondary key. Deterministic ordering makes
    # [DOCUMENTATION] exported symmetry tables reproducible across runs.
    sorted_items = sorted(
        operations.items(),
        key=lambda item: (
            np.linalg.norm(item[1][0] - np.eye(3)),
            item[0],
        ),
    )

    permutations = tuple(item[0] for item in sorted_items)
    rotation_matrices = [item[1][0] for item in sorted_items]
    residuals = np.asarray([item[1][1] for item in sorted_items], dtype=float)

    # [DOCUMENTATION] Before exposing the symmetry set to downstream orientation calculations, verify that the
    # [DOCUMENTATION] discrete operations truly form a group under composition.
    _validate_permutation_group(permutations)

    # [DOCUMENTATION] Convert each accepted proper-rotation matrix to one canonical scalar-first quaternion.
    # [DOCUMENTATION] This array has one row per distinct PHYSICAL rotational symmetry.
    physical_quaternions = np.asarray(
        [_canonical_wxyz(matrix) for matrix in rotation_matrices],
        dtype=np.float64,
    )

    # freud explicitly requires q and -q for every equivalent physical rotation.
    # [DOCUMENTATION] Build the equivalent-orientation list passed to freud. Every physical rotation appears
    # [DOCUMENTATION] twice, as q and -q, because unit quaternions double-cover SO(3): both signs represent the
    # [DOCUMENTATION] same spatial rotation.
    equivalent = np.empty((2 * len(physical_quaternions), 4), dtype=np.float64)
    equivalent[0::2] = physical_quaternions
    equivalent[1::2] = -physical_quaternions

    print("\nParticle rotational symmetry")
    print("----------------------------")
    print(f"Matching tolerance: {matching_tolerance:.6g}")
    print(f"Source triple: {source_indices}")
    print(f"Ordered target triples tested: {candidates_tested}")
    print(f"Gram-compatible target triples: {gram_hits}")
    print(f"Distinct proper rotations: {len(physical_quaternions)}")
    print(f"Equivalent quaternions supplied to freud (q and -q): {len(equivalent)}")
    print(f"Maximum accepted vertex residual: {np.max(residuals):.6g}")

    return SymmetryResult(
        centered_vertices=centered,
        original_center=original_center,
        physical_quaternions_wxyz=physical_quaternions,
        equivalent_quaternions_wxyz=equivalent,
        permutations=permutations,
        max_residuals=residuals,
        matching_tolerance=matching_tolerance,
    )


# -----------------------------------------------------------------------------
# GSD frame reading and validation
# -----------------------------------------------------------------------------
# [DOCUMENTATION] TRAJECTORY-ORIENTATION VALIDATION. Requires an N x 4 finite quaternion array, rejects
# [DOCUMENTATION] zero-norm rows, reports the worst norm drift, and normalizes every quaternion. Normalization
# [DOCUMENTATION] prevents small trajectory-storage errors from contaminating angular comparisons.
def normalize_quaternions_wxyz(orientations: np.ndarray) -> np.ndarray:
    q = np.asarray(orientations, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected orientations with shape (N,4); got {q.shape}.")
    if not np.all(np.isfinite(q)):
        raise ValueError("Trajectory orientations contain NaN/Inf values.")

    # [DOCUMENTATION] Compute one Euclidean norm per quaternion. Exact rigid-body quaternions should have norm
    # [DOCUMENTATION] 1, but normalizing defensively makes the later angle calculation robust to small
    # [DOCUMENTATION] floating-point deviations.
    norms = np.linalg.norm(q, axis=1)
    if np.any(norms <= np.finfo(float).eps):
        bad = int(np.where(norms <= np.finfo(float).eps)[0][0])
        raise ValueError(f"Particle {bad} has a zero/invalid orientation quaternion.")

    max_deviation = float(np.max(np.abs(norms - 1.0)))
    print(f"Maximum quaternion-norm deviation before normalization: {max_deviation:.3e}")
    return q / norms[:, None]


# [DOCUMENTATION] FRAME-INDEX HELPER. Implements ordinary Python negative indexing explicitly: -1 means the
# [DOCUMENTATION] final frame, -2 the penultimate frame, etc. It also converts the result to a nonnegative index
# [DOCUMENTATION] used in output filenames and metadata.
def resolve_frame_index(requested: int, n_frames: int) -> int:
    resolved = requested if requested >= 0 else n_frames + requested
    if resolved < 0 or resolved >= n_frames:
        raise IndexError(
            f"frame_index={requested} resolves to {resolved}, but trajectory has "
            f"{n_frames} frames (valid indices 0..{n_frames - 1}, or negative Python indices)."
        )
    return resolved


# [DOCUMENTATION] GSD FRAME READER. Opens the HOOMD-schema trajectory, resolves the requested frame, copies
# [DOCUMENTATION] particle positions/orientations and the six HOOMD box parameters, closes the trajectory, then
# [DOCUMENTATION] validates array shapes and box lengths.
# [DOCUMENTATION] The returned HOOMD box ordering is [Lx, Ly, Lz, xy, xz, yz], where xy/xz/yz are dimensionless
# [DOCUMENTATION] tilt factors. The same convention is used later to construct the triclinic box matrix and to
# [DOCUMENTATION] render the box.
def read_gsd_frame(
    gsd_hoomd: Any,
    gsd_file: Path,
    requested_frame: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    # [DOCUMENTATION] Use a context manager so the trajectory file is closed immediately after the selected
    # [DOCUMENTATION] frame data have been copied into NumPy arrays.
    with gsd_hoomd.open(name=str(gsd_file), mode="r") as traj:
        n_frames = len(traj)
        if n_frames == 0:
            raise ValueError(f"Trajectory {gsd_file} contains no frames.")
        frame_index = resolve_frame_index(requested_frame, n_frames)
        snap = traj[frame_index]

        # [DOCUMENTATION] Copy positions as float64 for geometry operations. Orientations are initially copied
        # [DOCUMENTATION] separately because they still need explicit validation/normalization.
        positions = np.asarray(snap.particles.position, dtype=np.float64)
        orientations_raw = np.asarray(snap.particles.orientation, dtype=np.float64)
        box = np.asarray(snap.configuration.box, dtype=np.float64)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"Expected positions with shape (N,3); got {positions.shape}.")
    if len(positions) != len(orientations_raw):
        raise ValueError("Position and orientation arrays have different particle counts.")
    if box.shape != (6,):
        raise ValueError(f"Expected HOOMD box [Lx,Ly,Lz,xy,xz,yz]; got {box}.")
    if np.any(box[:3] <= 0.0):
        raise ValueError(f"Invalid box lengths: {box[:3]}.")

    # [DOCUMENTATION] Normalize quaternions only after all structural checks have passed. The returned
    # [DOCUMENTATION] orientation array is the one used for every subsequent clustering and rendering step.
    orientations = normalize_quaternions_wxyz(orientations_raw)

    print("\nTrajectory frame")
    print("----------------")
    print(f"File: {gsd_file}")
    print(f"Number of frames: {n_frames}")
    print(f"Requested frame_index: {requested_frame}")
    print(f"Resolved frame index: {frame_index}")
    print(f"Particles in frame: {len(positions)}")
    print(f"HOOMD box [Lx,Ly,Lz,xy,xz,yz]: {box.tolist()}")

    return positions, orientations, box, frame_index, n_frames


# -----------------------------------------------------------------------------
# Symmetry-reduced pairwise orientation matrix
# -----------------------------------------------------------------------------
# [DOCUMENTATION] PAIRWISE ORIENTATION ENGINE. Builds an N x N matrix D where D[i,j] is the symmetry-reduced
# [DOCUMENTATION] angular separation, in degrees, between particle i and reference particle j after minimizing
# [DOCUMENTATION] over the particle's equivalent rotational symmetries supplied to freud.
# [DOCUMENTATION] The full matrix is required by the later complete-link-like cluster rule, which repeatedly
# [DOCUMENTATION] asks whether a candidate is within tolerance of every member already in a cluster.
# [DOCUMENTATION] To control memory pressure, freud is called on blocks of query orientations. The final matrix
# [DOCUMENTATION] itself is stored as float32; if its estimated size exceeds ram_limit_mb, storage transparently
# [DOCUMENTATION] switches to a disk-backed NumPy memmap.
def build_pairwise_angle_matrix_deg(
    freud_module: Any,
    orientations: np.ndarray,
    equivalent_quaternions: np.ndarray,
    output_dir: Path,
    block_size: int,
    ram_limit_mb: float,
) -> tuple[np.ndarray, Path | None]:
    """Compute D[i,j] = symmetry-reduced angle of particle i to reference j.

    The matrix is populated in row blocks with freud so the freud calculation
    never needs to allocate the entire N x N result internally. If the final
    float32 matrix exceeds ram_limit_mb it is backed by a temporary memmap.
    """
    n = len(orientations)
    # [DOCUMENTATION] Estimate the final matrix size before allocation. For N particles the storage scales as
    # [DOCUMENTATION] N^2, so this check is important for large trajectories.
    estimated_mb = n * n * np.dtype(np.float32).itemsize / 1024**2

    temp_file: Path | None = None
    # [DOCUMENTATION] Use ordinary RAM when the matrix fits under the configured threshold; otherwise allocate
    # [DOCUMENTATION] the same logical array as a disk-backed memory map. Downstream code can index either
    # [DOCUMENTATION] representation identically.
    if estimated_mb <= ram_limit_mb:
        matrix: np.ndarray = np.empty((n, n), dtype=np.float32)
        storage = "RAM"
    else:
        temp_file = output_dir / ".pairwise_orientation_angles_float32.dat"
        matrix = np.memmap(temp_file, mode="w+", dtype=np.float32, shape=(n, n))
        storage = f"disk memmap: {temp_file}"

    print("\nPairwise symmetry-reduced orientation matrix")
    print("--------------------------------------------")
    print(f"Shape: {n} x {n}")
    print(f"Float32 size: {estimated_mb:.1f} MiB")
    print(f"Storage: {storage}")
    print(f"Freud block size: {block_size}")

    # [DOCUMENTATION] Process rows in contiguous blocks. Each block compares a subset of particle orientations
    # [DOCUMENTATION] against all N reference orientations, preventing freud from needing to materialize the
    # [DOCUMENTATION] entire N x N result internally at once.
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        # [DOCUMENTATION] Create a fresh freud AngularSeparationGlobal calculator for this block.
        # [DOCUMENTATION] equivalent_quaternions tells freud which body-symmetry-related orientations are
        # [DOCUMENTATION] physically identical when computing the minimum angular separation.
        calc = freud_module.environment.AngularSeparationGlobal()
        calc.compute(
            orientations,               # global/reference orientations -> columns
            orientations[start:stop],    # particle orientations -> rows
            equivalent_quaternions,
        )
        # [DOCUMENTATION] freud reports radians. Convert to degrees immediately because orientation_angle_tol
        # [DOCUMENTATION] and all user-facing diagnostics in this workflow are specified in degrees.
        block = np.rad2deg(np.asarray(calc.angles, dtype=np.float64))
        expected_shape = (stop - start, n)
        if block.shape != expected_shape:
            raise RuntimeError(
                "Unexpected freud AngularSeparationGlobal output shape: "
                f"got {block.shape}, expected {expected_shape}."
            )
        # [DOCUMENTATION] Store the completed block in float32 to halve memory relative to float64. The
        # [DOCUMENTATION] clustering tolerance is many orders of magnitude larger than float32 roundoff for
        # [DOCUMENTATION] these degree-valued angles.
        matrix[start:stop] = block.astype(np.float32)

        if start == 0 or stop == n or (start // block_size) % 10 == 0:
            print(f"  computed rows {start:6d} .. {stop - 1:6d} / {n - 1}")

    # [DOCUMENTATION] Enforce exact zero self-separation on the diagonal. This is mathematically required and
    # [DOCUMENTATION] also prevents tiny numerical residuals from affecting seed/candidate logic.
    np.fill_diagonal(matrix, 0.0)
    if isinstance(matrix, np.memmap):
        matrix.flush()

    return matrix, temp_file


# -----------------------------------------------------------------------------
# Orientational clustering and final color assignment
# -----------------------------------------------------------------------------
# [DOCUMENTATION] ORIENTATION-CLUSTER DISCOVERY. This intentionally reproduces the supplied historical greedy
# [DOCUMENTATION] logic rather than replacing it with a different clustering algorithm.
# [DOCUMENTATION] Starting from the first still-unassigned particle, the code first gathers unassigned
# [DOCUMENTATION] candidates within tolerance of the seed. A candidate is accepted only if its symmetry-reduced
# [DOCUMENTATION] angle is <= tolerance_deg to EVERY particle already accepted into that cluster. Once a cluster
# [DOCUMENTATION] is finished, all of its members are marked assigned and cannot seed/join later clusters.
# [DOCUMENTATION] Because seeds and candidates are considered in particle-index order, this is deterministic for
# [DOCUMENTATION] a fixed angle matrix, but it is order-dependent by design; that order dependence is part of
# [DOCUMENTATION] the preserved original workflow.
def greedy_complete_link_clusters(
    angle_matrix_deg: np.ndarray,
    tolerance_deg: float,
) -> tuple[np.ndarray, ...]:
    """Reproduce the supplied complete-link-like greedy clustering logic.

    For each first unassigned particle i:
      1. candidate particles must lie within tolerance of i;
      2. each candidate is admitted only if it lies within tolerance of EVERY
         particle already accepted into the current cluster.

    The procedure is deterministic because particles are considered in index
    order. It intentionally preserves the logic of the original ref-frame code.
    """
    n = angle_matrix_deg.shape[0]
    if angle_matrix_deg.shape != (n, n):
        raise ValueError("angle_matrix_deg must be square.")

    # [DOCUMENTATION] Boolean bookkeeping records whether each particle has already been committed to a
    # [DOCUMENTATION] provisional cluster. A particle belongs to exactly one provisional cluster.
    assigned = np.zeros(n, dtype=bool)
    clusters: list[np.ndarray] = []

    print("\nDiscovering orientational clusters")
    print("---------------------------------")
    print(f"Complete-link acceptance tolerance: {tolerance_deg:.6g} deg")

    # [DOCUMENTATION] Scan particles in index order. Already-assigned indices are skipped; the first unassigned
    # [DOCUMENTATION] particle becomes the next cluster seed.
    for seed in range(n):
        if assigned[seed]:
            continue

        # [DOCUMENTATION] Initial candidates must satisfy two conditions simultaneously: they are not yet
        # [DOCUMENTATION] assigned elsewhere, and their symmetry-reduced angle to the seed is within the
        # [DOCUMENTATION] orientation tolerance.
        candidate_indices = np.where((~assigned) & (angle_matrix_deg[:, seed] <= tolerance_deg))[0]
        cluster: list[int] = [seed]

        for candidate in candidate_indices:
            candidate = int(candidate)
            if candidate == seed:
                continue
            current = np.asarray(cluster, dtype=np.int64)
            # [DOCUMENTATION] Complete-link-like acceptance test: the candidate joins only when it is within
            # [DOCUMENTATION] tolerance of every current cluster member, not merely the seed. This keeps each
            # [DOCUMENTATION] discovered cluster internally tight according to the specified threshold.
            if np.all(angle_matrix_deg[candidate, current] <= tolerance_deg):
                cluster.append(candidate)

        cluster_array = np.asarray(cluster, dtype=np.int64)
        assigned[cluster_array] = True
        clusters.append(cluster_array)

        if len(clusters) <= 10 or assigned.all():
            print(
                f"  provisional cluster {len(clusters)-1:3d}: "
                f"seed={seed:6d}, size={len(cluster_array):6d}, "
                f"assigned={int(np.count_nonzero(assigned)):6d}/{n}"
            )

    # [DOCUMENTATION] After discovery, sort clusters from largest to smallest population. The seed index breaks
    # [DOCUMENTATION] equal-size ties deterministically. This ordering later controls which supplied colors are
    # [DOCUMENTATION] assigned first.
    clusters.sort(key=lambda arr: (-len(arr), int(arr[0])))
    return tuple(clusters)


# [DOCUMENTATION] CLUSTER RETENTION + FINAL COLORING. Discovery clusters smaller than cluster_size_cutoff are
# [DOCUMENTATION] not kept as independent orientation references. The surviving largest clusters define
# [DOCUMENTATION] reference orientations, subject to the number of colors available.
# [DOCUMENTATION] Crucially, every particle is then reassigned to its nearest retained reference orientation.
# [DOCUMENTATION] Therefore particles from discarded small provisional groups still receive one of the retained
# [DOCUMENTATION] cluster colors instead of remaining uncolored.
def finalize_clusters_and_assign_colors(
    angle_matrix_deg: np.ndarray,
    provisional_clusters: tuple[np.ndarray, ...],
    cluster_size_cutoff: int,
    number_of_colors: int,
) -> ClusterResult:
    if number_of_colors <= 0:
        raise ValueError("At least one color is required.")

    # [DOCUMENTATION] Apply the population threshold only to determine which provisional orientations are
    # [DOCUMENTATION] important enough to serve as retained references.
    retained = [cluster for cluster in provisional_clusters if len(cluster) >= cluster_size_cutoff]
    if not retained:
        print(
            "WARNING: no provisional cluster passed cluster_size_cutoff; "
            "the largest provisional cluster will be retained."
        )
        retained = [provisional_clusters[0]]

    # [DOCUMENTATION] The palette places a hard upper bound on independently colored retained references.
    # [DOCUMENTATION] Because provisional clusters are already sorted by size, truncation keeps the largest
    # [DOCUMENTATION] available orientation populations.
    if len(retained) > number_of_colors:
        print(
            f"WARNING: {len(retained)} clusters passed the size cutoff but only "
            f"{number_of_colors} colors were supplied. Only the {number_of_colors} "
            "largest clusters will be used as orientation references; every "
            "particle will still be assigned to its nearest retained reference."
        )
        retained = retained[:number_of_colors]

    # [DOCUMENTATION] Use the first/seed particle of each retained discovery cluster as that cluster's reference
    # [DOCUMENTATION] quaternion, preserving the reference-selection logic of the original workflow.
    reference_indices = np.asarray([int(cluster[0]) for cluster in retained], dtype=np.int64)
    retained_seed_sizes = np.asarray([len(cluster) for cluster in retained], dtype=np.int64)

    # This is the same conceptual final step as the supplied color_particles.py:
    # choose the minimum symmetry-reduced angle to one of the retained refs.
    # [DOCUMENTATION] Extract only the columns of the pairwise matrix corresponding to retained reference
    # [DOCUMENTATION] particles. Row i now contains particle i's angle to every possible retained
    # [DOCUMENTATION] color/reference.
    reference_angles = np.asarray(angle_matrix_deg[:, reference_indices], dtype=np.float64)
    # [DOCUMENTATION] For each particle choose the retained reference with the minimum symmetry-reduced angle.
    # [DOCUMENTATION] The integer result is simultaneously the final cluster ID and an index into palette_used.
    final_ids = np.argmin(reference_angles, axis=1).astype(np.int64)
    nearest_angles = reference_angles[np.arange(len(final_ids)), final_ids]

    print("\nRetained orientation references and final populations")
    print("-----------------------------------------------------")
    print(" cluster   reference_particle   discovery_size   final_population   max_angle_deg")
    for cluster_id, ref_index in enumerate(reference_indices):
        mask = final_ids == cluster_id
        max_angle = float(np.max(nearest_angles[mask])) if np.any(mask) else float("nan")
        print(
            f" {cluster_id:7d} {int(ref_index):20d} {int(retained_seed_sizes[cluster_id]):16d} "
            f"{int(np.count_nonzero(mask)):18d} {max_angle:15.6f}"
        )

    return ClusterResult(
        provisional_clusters=provisional_clusters,
        retained_reference_indices=reference_indices,
        final_cluster_ids=final_ids,
        nearest_reference_angles_deg=nearest_angles,
        retained_seed_sizes=retained_seed_sizes,
    )


# -----------------------------------------------------------------------------
# Box / periodic-coordinate helpers
# -----------------------------------------------------------------------------
# [DOCUMENTATION] PERIODIC-BOX GEOMETRY. Convert HOOMD's six box parameters into a 3 x 3 matrix B whose columns
# [DOCUMENTATION] are the triclinic lattice vectors a, b, c. Cartesian coordinates are related to fractional box
# [DOCUMENTATION] coordinates by r = B s.
# [DOCUMENTATION] For HOOMD [Lx,Ly,Lz,xy,xz,yz], the vectors are a=(Lx,0,0), b=(xy*Ly,Ly,0), c=(xz*Lz,yz*Lz,Lz).
def hoomd_box_matrix(box: np.ndarray) -> np.ndarray:
    """Return matrix B whose columns are HOOMD triclinic box vectors a,b,c."""
    lx, ly, lz, xy, xz, yz = [float(x) for x in box]
    return np.array(
        [
            [lx, xy * ly, xz * lz],
            [0.0, ly, yz * lz],
            [0.0, 0.0, lz],
        ],
        dtype=np.float64,
    )


# [DOCUMENTATION] Convert Cartesian particle positions to fractional triclinic-box coordinates by solving B s =
# [DOCUMENTATION] r. Fractional coordinates make periodic wrapping and axis-aligned crop selection
# [DOCUMENTATION] straightforward even for tilted boxes.
def cartesian_to_fractional(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    b = hoomd_box_matrix(box)
    return np.linalg.solve(b, positions.T).T


# [DOCUMENTATION] Inverse coordinate conversion: multiply fractional coordinates by the triclinic box matrix to
# [DOCUMENTATION] recover Cartesian positions for rendering.
def fractional_to_cartesian(frac: np.ndarray, box: np.ndarray) -> np.ndarray:
    b = hoomd_box_matrix(box)
    return np.asarray(frac, dtype=np.float64) @ b.T


# [DOCUMENTATION] ZOOM/CROP SELECTOR. Select a rectangular chunk in fractional periodic coordinates, optionally
# [DOCUMENTATION] center it on a particular particle, optionally filter by cluster ID, then recenter the
# [DOCUMENTATION] selected periodic neighborhood around the origin for clean rendering.
# [DOCUMENTATION] The function selects PARTICLE CENTERS. The finite geometry of a polyhedron can extend beyond
# [DOCUMENTATION] the nominal crop boundary, which is why the separate zoom_padding_factor controls camera
# [DOCUMENTATION] margin in the renderer.
def choose_zoom_section(
    positions: np.ndarray,
    orientations: np.ndarray,
    cluster_ids: np.ndarray,
    box: np.ndarray,
    zoom_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select and recenter a periodic fractional sub-box for the zoom render.

    Returns
    -------
    zoom_positions_centered, zoom_orientations, zoom_cluster_ids,
    selected_original_indices, zoom_box
    """
    # [DOCUMENTATION] Work in fractional coordinates so a crop size such as [0.3,0.3,0.3] always means 30% of
    # [DOCUMENTATION] each simulation-box direction, including tilted triclinic geometries.
    frac = cartesian_to_fractional(positions, box)

    # [DOCUMENTATION] Two mutually exclusive center modes are supported. If center_particle_index is provided,
    # [DOCUMENTATION] use that particle's fractional position. Otherwise use the explicit center_fractional
    # [DOCUMENTATION] vector from the parameter file.
    if zoom_cfg.get("center_particle_index") is not None:
        center_particle = int(zoom_cfg["center_particle_index"])
        if center_particle < 0 or center_particle >= len(positions):
            raise IndexError(
                f"zoom.center_particle_index={center_particle} is outside 0..{len(positions)-1}."
            )
        center_frac = frac[center_particle].copy()
    else:
        center_frac = as_float_triplet(
            zoom_cfg.get("center_fractional", [0.0, 0.0, 0.0]),
            "zoom.center_fractional",
        )

    # [DOCUMENTATION] size_fractional specifies the crop width along each fractional box direction. Every
    # [DOCUMENTATION] component must be >0 and <=1, where 1 spans the full periodic box length in that
    # [DOCUMENTATION] direction.
    size_frac = as_float_triplet(
        zoom_cfg.get("size_fractional", [0.30, 0.30, 0.30]),
        "zoom.size_fractional",
    )
    if np.any(size_frac <= 0.0) or np.any(size_frac > 1.0):
        raise ValueError("zoom.size_fractional components must lie in (0, 1].")

    # Minimum-image displacement around requested crop center.
    # [DOCUMENTATION] Compute each particle's displacement from the requested crop center in fractional
    # [DOCUMENTATION] coordinates.
    delta_frac = frac - center_frac[None, :]
    # [DOCUMENTATION] Apply the minimum-image convention in fractional space by shifting each displacement into
    # [DOCUMENTATION] approximately [-0.5,0.5). This is what keeps a crop contiguous when its center is near a
    # [DOCUMENTATION] periodic boundary.
    delta_frac -= np.round(delta_frac)
    # [DOCUMENTATION] Keep particles whose minimum-image center displacement lies inside half the requested crop
    # [DOCUMENTATION] size along ALL three fractional directions.
    spatial_mask = np.all(np.abs(delta_frac) <= 0.5 * size_frac[None, :], axis=1)

    # [DOCUMENTATION] An optional second mask can restrict the zoom image to selected final orientational
    # [DOCUMENTATION] cluster IDs. A null value leaves all cluster colors visible inside the spatial crop.
    requested_clusters = zoom_cfg.get("cluster_ids")
    if requested_clusters is not None:
        requested_clusters = np.asarray(requested_clusters, dtype=np.int64)
        spatial_mask &= np.isin(cluster_ids, requested_clusters)

    selected = np.where(spatial_mask)[0]
    if len(selected) == 0:
        raise RuntimeError(
            "The requested zoom section contains no particles. Increase "
            "zoom.size_fractional or change zoom.center_fractional / "
            "zoom.center_particle_index."
        )

    # Use the periodic displacement itself as local fractional coordinates, so
    # the zoomed region is centered at the origin and remains contiguous even
    # when it crosses a periodic boundary.
    # [DOCUMENTATION] Use the already minimum-imaged displacement as the selected particles' local coordinates.
    # [DOCUMENTATION] Thus the chosen crop is translated so its requested center sits at the origin, independent
    # [DOCUMENTATION] of the original box location.
    local_frac = delta_frac[selected]
    local_positions = fractional_to_cartesian(local_frac, box)

    # [DOCUMENTATION] Construct a notional box describing the crop dimensions by scaling only Lx, Ly, and Lz by
    # [DOCUMENTATION] size_fractional. The tilt factors are preserved. Even when the zoom box is not drawn,
    # [DOCUMENTATION] these dimensions are still used to estimate the orthographic camera field of view.
    zoom_box = np.asarray(box, dtype=np.float64).copy()
    zoom_box[0] *= size_frac[0]
    zoom_box[1] *= size_frac[1]
    zoom_box[2] *= size_frac[2]

    print("\nZoom section")
    print("------------")
    print(f"Fractional center: {center_frac.tolist()}")
    print(f"Fractional size:   {size_frac.tolist()}")
    # [DOCUMENTATION] This second check does not alter the crop. It only prints which particle index was used as
    # [DOCUMENTATION] the center when particle-centered zooming was requested, making the terminal log easier to
    # [DOCUMENTATION] reproduce later.
    if zoom_cfg.get("center_particle_index") is not None:
        print(f"Centered on particle: {int(zoom_cfg['center_particle_index'])}")
    print(f"Particles selected: {len(selected)} / {len(positions)}")
    if requested_clusters is not None:
        print(f"Cluster filter: {requested_clusters.tolist()}")

    return (
        local_positions,
        orientations[selected],
        cluster_ids[selected],
        selected,
        zoom_box,
    )


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------
# [DOCUMENTATION] CAMERA-ORIENTATION HELPER. Converts user-facing Z-Y-X Euler angles in degrees to the
# [DOCUMENTATION] scalar-first quaternion expected by the Plato scene. SciPy again returns [x,y,z,w], so the
# [DOCUMENTATION] components are explicitly reordered.
def camera_quaternion_wxyz(euler_zyx_deg: Any) -> np.ndarray:
    angles = as_float_triplet(euler_zyx_deg, "render.camera_euler_zyx_deg")
    q_xyzw = Rotation.from_euler("zyx", angles, degrees=True).as_quat()
    q_wxyz = q_xyzw[[3, 0, 1, 2]]
    q_wxyz /= np.linalg.norm(q_wxyz)
    return q_wxyz


# [DOCUMENTATION] AUTOMATIC ORTHOGRAPHIC FRAMING. Generate the eight triclinic box corners, rotate them by the
# [DOCUMENTATION] same camera quaternion used for rendering, project onto camera x-y, measure the 2D extents,
# [DOCUMENTATION] then multiply by a padding factor.
# [DOCUMENTATION] For the zoom image, this is where zoom_padding_factor creates visual margin so finite-size
# [DOCUMENTATION] particles near the crop boundary are less likely to be clipped.
def projected_scene_size(
    box: np.ndarray,
    camera_q_wxyz: np.ndarray,
    padding: float = 1.08,
) -> tuple[float, float]:
    """Estimate the 2D orthographic scene size from rotated box corners."""
    b = hoomd_box_matrix(box)
    # [DOCUMENTATION] The Cartesian corners are generated from every combination of fractional coordinates
    # [DOCUMENTATION] +/-0.5 along the three box axes: 2^3 = 8 corners.
    frac_corners = np.asarray(
        list(itertools.product([-0.5, 0.5], repeat=3)),
        dtype=np.float64,
    )
    corners = frac_corners @ b.T

    q = np.asarray(camera_q_wxyz, dtype=np.float64)
    q_xyzw = q[[1, 2, 3, 0]]
    # [DOCUMENTATION] Apply the camera rotation to the box corners solely for estimating what width/height they
    # [DOCUMENTATION] occupy in the camera plane. This does not alter particle coordinates stored elsewhere.
    rotated = Rotation.from_quat(q_xyzw).apply(corners)
    # [DOCUMENTATION] np.ptp computes max-min in projected x and y, giving the raw orthographic width and height
    # [DOCUMENTATION] required to contain the rotated box.
    extent = np.ptp(rotated[:, :2], axis=0)

    # Avoid pathological zero dimensions for very thin crops.
    extent = np.maximum(extent, 1.0e-6)
    return float(padding * extent[0]), float(padding * extent[1])


# [DOCUMENTATION] SCENE-SIZE POLICY. If the JSON scene size is null, derive width/height automatically from
# [DOCUMENTATION] projected_scene_size. A single positive number forces a square field of view; a two-element
# [DOCUMENTATION] positive array explicitly sets width and height.
def parse_scene_size(
    configured_size: Any,
    box: np.ndarray,
    rotation: np.ndarray,
    padding: float,
) -> tuple[float, float]:
    if configured_size is None:
        return projected_scene_size(box, rotation, padding)

    arr = np.asarray(configured_size, dtype=np.float64)
    if arr.ndim == 0:
        value = float(arr)
        if value <= 0.0:
            raise ValueError("Configured scene size must be positive.")
        return value, value
    if arr.shape == (2,) and np.all(arr > 0.0):
        return float(arr[0]), float(arr[1])
    raise ValueError("Scene size must be null, one positive number, or [width, height].")


# [DOCUMENTATION] PUBLICATION RENDERER. Build one Plato/Fresnel ConvexPolyhedra primitive containing all
# [DOCUMENTATION] selected particles, color each particle by final cluster ID, optionally add the simulation
# [DOCUMENTATION] box, choose full-vs-zoom camera settings, enable lighting/antialiasing/path tracing, and save
# [DOCUMENTATION] the PNG.
# [DOCUMENTATION] The same function renders both images. The is_zoom flag only selects zoom-specific scene
# [DOCUMENTATION] size/pixel scale/padding and the zoom-box visibility setting; particle geometry and color
# [DOCUMENTATION] assignment are otherwise identical.
def render_snapshot(
    draw: Any,
    to_rgba: Any,
    vertices: np.ndarray,
    positions: np.ndarray,
    orientations: np.ndarray,
    cluster_ids: np.ndarray,
    palette: list[str],
    box: np.ndarray,
    output_png: Path,
    render_cfg: dict[str, Any],
    *,
    is_zoom: bool,
) -> None:
    # [DOCUMENTATION] Translate each integer final cluster ID into the corresponding palette hex color, then
    # [DOCUMENTATION] convert that color to numerical RGBA. The resulting N x 4 array supplies one primitive
    # [DOCUMENTATION] color per particle.
    rgba = np.asarray([to_rgba(palette[int(cid)]) for cid in cluster_ids], dtype=np.float64)

    outline = float(render_cfg.get("outline", 0.01))
    roughness = float(render_cfg.get("roughness", 0.15))
    specular = float(render_cfg.get("specular", 0.8))
    spec_trans = float(render_cfg.get("spec_trans", 0.0))

    # [DOCUMENTATION] Create the Plato convex-polyhedron primitive. A single body-frame vertex set is reused for
    # [DOCUMENTATION] every particle; positions and quaternions place/orient individual copies. Material
    # [DOCUMENTATION] parameters such as roughness/specular control only appearance, not geometry or clustering.
    polyhedra = draw.ConvexPolyhedra(
        colors=rgba,
        positions=np.asarray(positions, dtype=np.float64),
        orientations=np.asarray(orientations, dtype=np.float64),
        outline=outline,
        vertices=np.asarray(vertices, dtype=np.float64),
        primitive_color_mix=0,
        roughness=roughness,
        specular=specular,
        spec_trans=spec_trans,
    )

    # [DOCUMENTATION] Start the scene with particles only. The optional Box primitive is appended separately;
    # [DOCUMENTATION] therefore disabling show_zoom_box truly leaves only particles in the zoom render.
    primitives: list[Any] = [polyhedra]
    # Modified version: the zoom image should contain only the selected particles,
    # without drawing the enclosing zoom-section box.  The full-system snapshot may
    # still show the simulation box unless render.show_box is set to false.
    # [DOCUMENTATION] Choose the box-visibility key based on render type. In this modified working version the
    # [DOCUMENTATION] default is False for zoom renders and True for full-system renders, while the JSON can
    # [DOCUMENTATION] explicitly override either behavior.
    show_box = bool(render_cfg.get("show_zoom_box" if is_zoom else "show_box", False if is_zoom else True))
    if show_box:
        box_color = render_cfg.get("box_color_rgba", [0.10, 0.10, 0.10, 1.0])
        box_width = float(render_cfg.get("box_width", 0.04 if not is_zoom else 0.025))
        # [DOCUMENTATION] When enabled, construct a Plato triclinic box using the same six HOOMD box parameters.
        # [DOCUMENTATION] This object is a visual boundary only; it does not participate in particle selection.
        box_primitive = draw.Box(
            Lx=float(box[0]),
            Ly=float(box[1]),
            Lz=float(box[2]),
            xy=float(box[3]),
            xz=float(box[4]),
            yz=float(box[5]),
            widths=box_width,
            colors=box_color,
            width=box_width,
            color=box_color,
        )
        primitives.append(box_primitive)

    # [DOCUMENTATION] Convert the configured camera Euler angles once and use the identical quaternion both for
    # [DOCUMENTATION] scene rendering and for automatic projected-size estimation.
    rotation = camera_quaternion_wxyz(
        render_cfg.get("camera_euler_zyx_deg", [0.0, 10.0, 10.0])
    )

    # [DOCUMENTATION] Use zoom-specific camera-framing and resolution controls for the crop. In the supplied
    # [DOCUMENTATION] parameter file, zoom_padding_factor is 1.32; increasing it enlarges the camera field of
    # [DOCUMENTATION] view and leaves more margin around the chunk without changing which particles are
    # [DOCUMENTATION] selected.
    if is_zoom:
        configured_size = render_cfg.get("zoom_scene_size")
        pixel_scale = int(render_cfg.get("zoom_pixel_scale", 180))
        padding = float(render_cfg.get("zoom_padding_factor", 1.12))
    else:
        configured_size = render_cfg.get("full_scene_size")
        pixel_scale = int(render_cfg.get("full_pixel_scale", 100))
        padding = float(render_cfg.get("full_padding_factor", 1.05))

    # [DOCUMENTATION] Determine the final orthographic view width and height. If zoom_scene_size/full_scene_size
    # [DOCUMENTATION] is null, this uses the corresponding box dimensions, camera rotation, and padding factor
    # [DOCUMENTATION] automatically.
    scene_size = parse_scene_size(
        configured_size,
        box,
        rotation,
        padding,
    )

    # [DOCUMENTATION] Create the Plato scene from the assembled primitives. pixel_scale controls raster
    # [DOCUMENTATION] resolution for a given physical scene size; it does not change the scientific coordinates.
    scene = draw.Scene(
        tuple(primitives),
        rotation=rotation,
        size=scene_size,
        pixel_scale=pixel_scale,
    )

    # [DOCUMENTATION] Lighting and rendering-quality controls are enabled after scene construction. Ambient
    # [DOCUMENTATION] light reduces overly dark faces; directional light gives shape cues; antialiasing smooths
    # [DOCUMENTATION] edges; pathtracer samples control Monte-Carlo image quality/noise.
    scene.enable("ambient_light", float(render_cfg.get("ambient_light", 1.5)))
    scene.enable(
        "directional_light",
        as_float_triplet(
            render_cfg.get("directional_light", [-1.5, 0.0, 0.0]),
            "render.directional_light",
        ).tolist(),
    )
    scene.enable("antialiasing", float(render_cfg.get("antialiasing", 1.0)))
    scene.enable("pathtracer", samples=int(render_cfg.get("pathtrace_samples", 128)))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    # [DOCUMENTATION] Write the rendered image to disk. The parent directory is created just beforehand, so
    # [DOCUMENTATION] rendering does not require the output folder to exist in advance.
    scene.save(str(output_png))
    print(f"Saved render: {output_png}")


# -----------------------------------------------------------------------------
# Output tables / metadata
# -----------------------------------------------------------------------------
# [DOCUMENTATION] REPRODUCIBILITY OUTPUT: one CSV row per particle. It records original particle index,
# [DOCUMENTATION] position, normalized quaternion, final cluster/color, the retained reference particle used for
# [DOCUMENTATION] that cluster, and the particle's symmetry-reduced angular distance to that reference.
def save_particle_table(
    path: Path,
    positions: np.ndarray,
    orientations: np.ndarray,
    cluster_result: ClusterResult,
    palette: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "particle_index",
                "x", "y", "z",
                "qw", "qx", "qy", "qz",
                "cluster_id",
                "cluster_color",
                "reference_particle_index",
                "angle_to_reference_deg",
            ]
        )
        refs = cluster_result.retained_reference_indices
        # [DOCUMENTATION] Iterate in original particle-index order so CSV rows can be cross-referenced directly
        # [DOCUMENTATION] with the GSD frame.
        for i in range(len(positions)):
            cid = int(cluster_result.final_cluster_ids[i])
            writer.writerow(
                [
                    i,
                    *[f"{x:.12g}" for x in positions[i]],
                    *[f"{x:.12g}" for x in orientations[i]],
                    cid,
                    palette[cid],
                    int(refs[cid]),
                    f"{cluster_result.nearest_reference_angles_deg[i]:.12g}",
                ]
            )


# [DOCUMENTATION] REPRODUCIBILITY OUTPUT: one CSV row per retained orientation cluster. It records the assigned
# [DOCUMENTATION] color, reference particle/quaternion, provisional discovery size, final nearest-reference
# [DOCUMENTATION] population, and mean/max final angular distance to the reference.
def save_cluster_summary(
    path: Path,
    orientations: np.ndarray,
    cluster_result: ClusterResult,
    palette: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cluster_id",
                "color",
                "reference_particle_index",
                "reference_qw",
                "reference_qx",
                "reference_qy",
                "reference_qz",
                "discovery_cluster_size",
                "final_population",
                "mean_angle_to_reference_deg",
                "max_angle_to_reference_deg",
            ]
        )
        # [DOCUMENTATION] Loop through retained clusters in their population-ranked/color order. The boolean
        # [DOCUMENTATION] mask selects all particles whose final assignment equals the current cluster ID.
        for cid, ref_index in enumerate(cluster_result.retained_reference_indices):
            mask = cluster_result.final_cluster_ids == cid
            q = orientations[int(ref_index)]
            angles = cluster_result.nearest_reference_angles_deg[mask]
            writer.writerow(
                [
                    cid,
                    palette[cid],
                    int(ref_index),
                    *[f"{x:.12g}" for x in q],
                    int(cluster_result.retained_seed_sizes[cid]),
                    int(np.count_nonzero(mask)),
                    f"{float(np.mean(angles)):.12g}",
                    f"{float(np.max(angles)):.12g}",
                ]
            )


# [DOCUMENTATION] REPRODUCIBILITY OUTPUT: save one canonical quaternion for each distinct physical proper
# [DOCUMENTATION] rotation of the particle together with its maximum vertex-matching residual. The duplicated
# [DOCUMENTATION] +/- quaternions used internally by freud are intentionally not written as separate physical
# [DOCUMENTATION] symmetries.
def save_symmetry_quaternions(path: Path, symmetry: SymmetryResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["physical_rotation_id", "qw", "qx", "qy", "qz", "max_vertex_residual"])
        for idx, (q, residual) in enumerate(
            zip(symmetry.physical_quaternions_wxyz, symmetry.max_residuals)
        ):
            writer.writerow([idx, *[f"{x:.16g}" for x in q], f"{residual:.16g}"])


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
# [DOCUMENTATION] TOP-LEVEL SCIENTIFIC WORKFLOW. This function wires all independent stages together in a fixed
# [DOCUMENTATION] order: load/validate parameters -> import runtime packages -> read shape -> detect shape
# [DOCUMENTATION] symmetry -> read trajectory frame -> build pairwise angle matrix -> discover/finalize clusters
# [DOCUMENTATION] -> save tables -> render full image -> select/render zoom -> write metadata -> clean temporary
# [DOCUMENTATION] storage.
def run(param_file: Path) -> None:
    # [DOCUMENTATION] Resolve the parameter file first; every relative input/output path below is interpreted
    # [DOCUMENTATION] relative to this file's directory.
    param_file = param_file.expanduser().resolve()
    params = load_json(param_file)
    base_dir = param_file.parent

    # [DOCUMENTATION] Resolve the two required scientific input files and the output directory. output_dir is
    # [DOCUMENTATION] created immediately so later matrix memmaps, CSVs, metadata, and PNG files have a
    # [DOCUMENTATION] guaranteed destination.
    gsd_file = resolve_path(str(require_key(params, "gsd_file")), base_dir)
    shape_file = resolve_path(str(require_key(params, "shape_file")), base_dir)
    output_dir = resolve_path(str(params.get("output_dir", "outputs_orientational_snapshot")), base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gsd_file.is_file():
        raise FileNotFoundError(f"GSD trajectory not found: {gsd_file}")
    if not shape_file.is_file():
        raise FileNotFoundError(f"Shape JSON not found: {shape_file}")

    # [DOCUMENTATION] Read the requested trajectory frame and the central clustering tolerance.
    # [DOCUMENTATION] orientation_angle_tol must be a physically meaningful angle in the interval (0,180]
    # [DOCUMENTATION] degrees.
    frame_index_requested = int(require_key(params, "frame_index"))
    orientation_angle_tol = float(require_key(params, "orientation_angle_tol"))
    if orientation_angle_tol <= 0.0 or orientation_angle_tol > 180.0:
        raise ValueError("orientation_angle_tol must lie in (0, 180] degrees.")

    # [DOCUMENTATION] Load the user-defined orientation-cluster colors. If cmap_original is absent, the
    # [DOCUMENTATION] preserved five-color default is used. Later, only the first n_retained colors are passed
    # [DOCUMENTATION] to the renderer.
    palette = ensure_hex_palette(
        params.get(
            "cmap_original",
            ["#0080ff", "#ff8000", "#5b0aa2", "#db023c", "#00ffff"],
        )
    )

    # [DOCUMENTATION] cluster_size_cutoff determines which provisional orientation groups are important enough
    # [DOCUMENTATION] to become independent retained references. It does NOT cause small-cluster particles to
    # [DOCUMENTATION] disappear; those particles are subsequently reassigned to the nearest retained reference.
    cluster_size_cutoff = int(params.get("cluster_size_cutoff", 1))
    if cluster_size_cutoff < 1:
        raise ValueError("cluster_size_cutoff must be >= 1.")

    # Preserve the historical meaning from the supplied parameter file:
    # tolerance_for_inv_quat_of_body_calc = p  -> coordinate tolerance 10^-p.
    # [DOCUMENTATION] Support two equivalent ways of configuring geometric symmetry matching. An explicit
    # [DOCUMENTATION] symmetry_vertex_tolerance takes precedence. Otherwise preserve the historical project
    # [DOCUMENTATION] convention p = tolerance_for_inv_quat_of_body_calc -> absolute coordinate tolerance
    # [DOCUMENTATION] 10^(-p).
    if "symmetry_vertex_tolerance" in params:
        symmetry_tolerance = float(params["symmetry_vertex_tolerance"])
    else:
        precision = int(params.get("tolerance_for_inv_quat_of_body_calc", 2))
        symmetry_tolerance = 10.0 ** (-precision)
    if symmetry_tolerance <= 0.0:
        raise ValueError("symmetry vertex tolerance must be positive.")

    # [DOCUMENTATION] pairwise_block_size controls how many query-particle rows freud processes at once;
    # [DOCUMENTATION] pairwise_matrix_ram_limit_mb controls only whether the completed N x N float32 matrix
    # [DOCUMENTATION] lives in RAM or in a temporary disk memmap.
    block_size = int(params.get("pairwise_block_size", 256))
    if block_size < 1:
        raise ValueError("pairwise_block_size must be >= 1.")
    ram_limit_mb = float(params.get("pairwise_matrix_ram_limit_mb", 512.0))

    # [DOCUMENTATION] Keep rendering and zoom subsections as dictionaries so their many optional values can be
    # [DOCUMENTATION] passed to specialized helper functions without expanding the run() argument list.
    render_cfg = dict(params.get("render", {}))
    zoom_cfg = dict(params.get("zoom", {}))
    output_prefix = str(params.get("output_prefix", shape_file.stem))

    # [DOCUMENTATION] Initialize dependencies, read/validate the particle geometry, and infer its proper
    # [DOCUMENTATION] rotational symmetry group BEFORE processing trajectory orientations. The resulting
    # [DOCUMENTATION] equivalent quaternions define what counts as the same physical orientation.
    runtime = import_runtime_packages()
    vertices = read_shape_vertices(shape_file)
    symmetry = detect_proper_rotational_symmetry(vertices, symmetry_tolerance)

    # [DOCUMENTATION] Read exactly one requested trajectory frame. frame_index is the resolved nonnegative index
    # [DOCUMENTATION] and is later embedded in filenames such as frame_000201.
    positions, orientations, box, frame_index, n_frames = read_gsd_frame(
        runtime.gsd_hoomd,
        gsd_file,
        frame_index_requested,
    )

    # [DOCUMENTATION] Compute the expensive pairwise symmetry-reduced orientation matrix once. Both provisional
    # [DOCUMENTATION] cluster discovery and final nearest-reference coloring reuse this matrix.
    angle_matrix, temp_angle_file = build_pairwise_angle_matrix_deg(
        runtime.freud,
        orientations,
        symmetry.equivalent_quaternions_wxyz,
        output_dir,
        block_size,
        ram_limit_mb,
    )

    try:
        # [DOCUMENTATION] Stage 1 clustering: discover deterministic provisional groups using the configured
        # [DOCUMENTATION] orientation-angle tolerance.
        provisional_clusters = greedy_complete_link_clusters(
            angle_matrix,
            orientation_angle_tol,
        )
        # [DOCUMENTATION] Stage 2 coloring: keep sufficiently populated reference orientations (limited by
        # [DOCUMENTATION] palette length) and assign every particle to the closest retained reference.
        cluster_result = finalize_clusters_and_assign_colors(
            angle_matrix,
            provisional_clusters,
            cluster_size_cutoff,
            len(palette),
        )

        # [DOCUMENTATION] Trim the palette to exactly the number of retained cluster references so color
        # [DOCUMENTATION] indexing remains one-to-one with final cluster IDs.
        n_retained = len(cluster_result.retained_reference_indices)
        palette_used = palette[:n_retained]

        # [DOCUMENTATION] Construct stable output filenames. The resolved frame index is zero-padded to six
        # [DOCUMENTATION] digits, making files sort naturally when multiple frames are rendered.
        frame_tag = f"frame_{frame_index:06d}"
        full_png = output_dir / f"{output_prefix}_{frame_tag}_full.png"
        zoom_png = output_dir / f"{output_prefix}_{frame_tag}_zoom.png"
        particle_csv = output_dir / f"{output_prefix}_{frame_tag}_particle_clusters.csv"
        cluster_csv = output_dir / f"{output_prefix}_{frame_tag}_cluster_summary.csv"
        symmetry_csv = output_dir / f"{output_prefix}_proper_rotational_symmetries.csv"
        metadata_json = output_dir / f"{output_prefix}_{frame_tag}_metadata.json"

        # [DOCUMENTATION] Write all non-image scientific outputs before rendering. If rendering later fails due
        # [DOCUMENTATION] to a graphics/runtime problem, cluster assignments and symmetry results are still
        # [DOCUMENTATION] preserved on disk.
        save_particle_table(
            particle_csv,
            positions,
            orientations,
            cluster_result,
            palette_used,
        )
        save_cluster_summary(
            cluster_csv,
            orientations,
            cluster_result,
            palette_used,
        )
        save_symmetry_quaternions(symmetry_csv, symmetry)

        print(f"Saved particle assignments: {particle_csv}")
        print(f"Saved cluster summary:      {cluster_csv}")
        print(f"Saved proper symmetries:    {symmetry_csv}")

        # [DOCUMENTATION] First render the complete selected GSD frame using original Cartesian positions, all
        # [DOCUMENTATION] particle orientations, final cluster IDs, and the full simulation box.
        render_snapshot(
            runtime.draw,
            runtime.to_rgba,
            symmetry.centered_vertices,
            positions,
            orientations,
            cluster_result.final_cluster_ids,
            palette_used,
            box,
            full_png,
            render_cfg,
            is_zoom=False,
        )

        # [DOCUMENTATION] Prepare zoom metadata. If zoom.enabled is false, this remains None and no zoom image
        # [DOCUMENTATION] is produced.
        zoom_selected_indices: list[int] | None = None
        # [DOCUMENTATION] When zooming is enabled, select/recenter the periodic chunk, remember the original
        # [DOCUMENTATION] particle indices for metadata, and call the SAME renderer with is_zoom=True. The
        # [DOCUMENTATION] working no-zoom-box behavior is controlled inside render_snapshot by show_zoom_box.
        if bool(zoom_cfg.get("enabled", True)):
            (
                zoom_positions,
                zoom_orientations,
                zoom_cluster_ids,
                zoom_indices,
                zoom_box,
            ) = choose_zoom_section(
                positions,
                orientations,
                cluster_result.final_cluster_ids,
                box,
                zoom_cfg,
            )
            zoom_selected_indices = [int(i) for i in zoom_indices]

            # [DOCUMENTATION] Render the periodic, recentered zoom selection with the same particle geometry and
            # [DOCUMENTATION] cluster palette. Passing is_zoom=True activates zoom-specific scene size, padding,
            # [DOCUMENTATION] pixel scale, and the particles-only no-zoom-box behavior configured for this version.
            render_snapshot(
                runtime.draw,
                runtime.to_rgba,
                symmetry.centered_vertices,
                zoom_positions,
                zoom_orientations,
                zoom_cluster_ids,
                palette_used,
                zoom_box,
                zoom_png,
                render_cfg,
                is_zoom=True,
            )

        # [DOCUMENTATION] Assemble a compact JSON metadata record describing exactly what was analyzed and
        # [DOCUMENTATION] rendered: input paths/frame, box, clustering thresholds and populations, colors,
        # [DOCUMENTATION] symmetry counts/tolerance, and the original indices included in the zoom crop.
        provisional_sizes = [int(len(c)) for c in provisional_clusters]
        final_populations = [
            int(np.count_nonzero(cluster_result.final_cluster_ids == cid))
            for cid in range(n_retained)
        ]
        metadata = {
            "gsd_file": str(gsd_file),
            "shape_file": str(shape_file),
            "trajectory_num_frames": int(n_frames),
            "requested_frame_index": int(frame_index_requested),
            "resolved_frame_index": int(frame_index),
            "num_particles": int(len(positions)),
            "box": [float(x) for x in box],
            "orientation_angle_tol_deg": float(orientation_angle_tol),
            "cluster_size_cutoff": int(cluster_size_cutoff),
            "num_provisional_clusters": int(len(provisional_clusters)),
            "provisional_cluster_sizes_descending": provisional_sizes,
            "num_retained_clusters": int(n_retained),
            "retained_reference_particle_indices": [
                int(x) for x in cluster_result.retained_reference_indices
            ],
            "retained_discovery_sizes": [
                int(x) for x in cluster_result.retained_seed_sizes
            ],
            "final_cluster_populations": final_populations,
            "cluster_colors": palette_used,
            "num_physical_proper_rotations": int(len(symmetry.physical_quaternions_wxyz)),
            "num_equivalent_quaternions_with_signs": int(len(symmetry.equivalent_quaternions_wxyz)),
            "symmetry_vertex_tolerance": float(symmetry.matching_tolerance),
            "zoom_selected_particle_indices": zoom_selected_indices,
        }
        with metadata_json.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        print(f"Saved metadata:             {metadata_json}")

    # [DOCUMENTATION] RESOURCE CLEANUP GUARANTEE. If the pairwise matrix was disk-backed, flush pending writes,
    # [DOCUMENTATION] release the array object, and delete the temporary .dat file. This happens whether the
    # [DOCUMENTATION] main processing succeeded or raised an exception.
    finally:
        # Explicitly release a memmap before deleting its backing file.
        if isinstance(angle_matrix, np.memmap):
            angle_matrix.flush()
        del angle_matrix
        if temp_angle_file is not None and temp_angle_file.exists():
            temp_angle_file.unlink()
            print(f"Removed temporary pairwise-angle memmap: {temp_angle_file}")

    print("\nDone.")


# [DOCUMENTATION] COMMAND-LINE INTERFACE. The program accepts exactly one positional argument: the parameter
# [DOCUMENTATION] JSON path. All scientific and rendering options live in that JSON rather than as many separate
# [DOCUMENTATION] command-line flags.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render one GSD frame colored by symmetry-reduced orientational clusters, "
            "plus a periodic-box-aware zoomed section."
        )
    )
    parser.add_argument(
        "param_file",
        type=Path,
        help="JSON parameter file controlling input paths, frame, clustering, and rendering.",
    )
    return parser


# [DOCUMENTATION] PROGRAM ENTRY WRAPPER. Parse arguments, run the workflow, convert any uncaught exception into
# [DOCUMENTATION] a concise stderr message plus exit code 1, and return 0 on success.
def main() -> int:
    args = build_parser().parse_args()
    try:
        run(args.param_file)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


# [DOCUMENTATION] Standard Python entry-point guard: execute main() only when this file is run as a script, not
# [DOCUMENTATION] when it is imported as a module. SystemExit propagates the integer return code to the shell.
if __name__ == "__main__":
    raise SystemExit(main())
