#!/usr/bin/env python3
"""
Rigorous standalone global pairwise-orientation histogram for HOOMD GSD files.

PURPOSE
-------
This program calculates the GLOBAL distribution of symmetry-reduced angular
separations between particle orientations in a HOOMD-blue GSD trajectory.

It is designed as a carefully corrected standalone replacement for the user's
original pipeline:

    shape JSON
        -> convex decomposition
        -> proper rotational symmetry operations of the body
        -> invariant/equivalent quaternions (+q and -q)
        -> freud.environment.AngularSeparationGlobal
        -> unique particle pairs i < j
        -> per-frame conditional histogram in the plotted angular range
        -> equal-weight average over the final N trajectory frames

RUN
---
    python hist_pairwise_angles_rigorous_standalone.py trajectory.gsd shape.json

The program then asks interactively for:
    * how many final frames to average;
    * how many particles to use from each selected frame;
    * the expected number of polyhedron edges and faces;
    * the symmetry-matching precision p (current historical value: p = 2);
    * the number of histogram bins;
    * the maximum plotted misorientation angle.

The historical parameter ``tolerance_for_inv_quat_of_body_calc = 2`` was used
in the old code as a decimal precision and gave a geometric matching tolerance
of 10**(-2) = 0.01 in shape-coordinate units.  This program preserves that
INTERPRETATION as the suggested value, but it DOES NOT round the accepted
quaternions.  The value p now controls only the absolute Euclidean tolerance
used to decide whether a rotated vertex set matches the original vertex set.

SCIENTIFIC DEFINITIONS USED HERE
--------------------------------
1. Only proper rotations are used.  Mirror reflections and other improper
   operations are deliberately excluded because particle orientations and
   unit quaternions represent SO(3), not the full O(3) point group.

2. ``freud.environment.AngularSeparationGlobal`` is used, as in the original
   calculation.  Its equivalent-orientation input explicitly contains both
   q and -q for every physical rotational symmetry.

3. Only unique unordered particle pairs are retained:

       i < j

   Self-pairs i == j are excluded and (i, j)/(j, i) are not double-counted.

4. Every selected frame is normalised independently INSIDE the plotted range:

       P_f(bin k) = count_f(k) / number_of_pairs_f_inside_plotted_range

   The final requested curve is the equal-weight frame average:

       mean_P(k) = (1/F) * sum_f P_f(k)

   Therefore the final mean probabilities sum to one (up to floating-point
   roundoff), even if some pair angles lie above the plotted maximum angle.

5. The x coordinate is the LEFT BIN EDGE, because this was the user's intended
   plotting convention.  The output is probability per bin, not probability
   density; no division by bin width is performed.

MEMORY STRATEGY
---------------
The old code formed an N x N angle matrix and then copied it repeatedly into
large Python lists.  This program asks freud to process only a block of query
orientations at a time.  Each block is compared with all selected particles,
then immediately reduced to histogram counts.  No complete all-frame angle
list is retained in memory.

DEPENDENCIES
------------
    pip install numpy scipy matplotlib gsd freud-analysis

Tested syntax does not guarantee runtime compatibility with every historical
freud/GSD release.  The code contains explicit shape checks and clear errors
for incompatible APIs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# SciPy is used for the convex hull, one-to-one vertex assignment, and
# full-precision rotation fitting.  These imports are safe at program startup;
# GSD and freud are imported lazily later so that the symmetry code can be
# inspected/tested independently of those packages.
try:
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import ConvexHull, cKDTree, distance_matrix
    from scipy.spatial.transform import Rotation
except ImportError as exc:  # pragma: no cover - depends on user environment
    raise SystemExit(
        "SciPy is required. Install dependencies with:\n"
        "    pip install numpy scipy matplotlib gsd freud-analysis"
    ) from exc


# ---------------------------------------------------------------------------
# Historical suggestions taken from the supplied parameter/code files.
# These are defaults shown in the terminal; the user can replace every value.
# ---------------------------------------------------------------------------
CURRENT_SUGGESTED_PRECISION = 2          # old tol_for_inv_quat_calc
CURRENT_SUGGESTED_NUM_EDGES = 25         # supplied EPD parameter file
CURRENT_SUGGESTED_NUM_FACES = 15         # supplied EPD parameter file
CURRENT_SUGGESTED_NUM_BINS = 50          # configured value in parameter file
CURRENT_SUGGESTED_MAX_ANGLE_DEG = 120.0  # supplied max misorientation angle
CURRENT_SUGGESTED_FRAMES = 1
DEFAULT_BLOCK_SIZE = 128


# ---------------------------------------------------------------------------
# Data containers make the scientific outputs explicit and auditable.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ConvexDecomposition:
    """Merged polygonal representation of a convex vertex set."""

    vertices: np.ndarray
    edges: tuple[tuple[int, int], ...]
    faces: tuple[np.ndarray, ...]
    # One normalised plane equation [nx, ny, nz, b] for each merged face,
    # satisfying n dot x + b = 0 in the CENTERED coordinate system.
    face_equations: np.ndarray
    merge_tolerance: float
    volume: float


@dataclass(frozen=True)
class SymmetryOperation:
    """One distinct physical proper rotation of the particle."""

    quaternion_wxyz: np.ndarray
    rotation_matrix: np.ndarray
    permutation: tuple[int, ...]
    max_vertex_residual: float
    discovery_axis: np.ndarray
    discovery_angle_deg: float


@dataclass(frozen=True)
class SymmetryResult:
    """Complete output of the proper-rotation symmetry search."""

    centered_vertices: np.ndarray
    original_center: np.ndarray
    decomposition: ConvexDecomposition
    physical_operations: tuple[SymmetryOperation, ...]
    # This is the array passed directly to freud.  It contains q and -q.
    equivalent_quaternions_wxyz: np.ndarray
    matching_tolerance: float


@dataclass(frozen=True)
class FrameHistogram:
    """Histogram and diagnostics for one trajectory frame."""

    frame_index: int
    raw_counts: np.ndarray
    conditional_probability: np.ndarray
    total_unique_pairs: int
    pairs_inside_range: int
    pairs_outside_range: int
    minimum_angle_deg: float
    maximum_angle_deg: float
    maximum_quaternion_norm_deviation: float


# ---------------------------------------------------------------------------
# Generic terminal-input helpers.
# ---------------------------------------------------------------------------
def prompt_value(
    prompt: str,
    default: Any,
    converter: Callable[[str], Any],
    validator: Callable[[Any], bool],
    error_message: str,
) -> Any:
    """Prompt until the user enters a valid value; an empty line uses default."""

    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            value = default
        else:
            try:
                value = converter(raw)
            except (TypeError, ValueError):
                print(error_message)
                continue

        if validator(value):
            return value

        print(error_message)


def resolve_or_prompt(
    supplied_value: Any,
    prompt: str,
    default: Any,
    converter: Callable[[str], Any],
    validator: Callable[[Any], bool],
    error_message: str,
) -> Any:
    """Use a CLI value when supplied; otherwise obtain it interactively."""

    if supplied_value is None:
        return prompt_value(prompt, default, converter, validator, error_message)

    if not validator(supplied_value):
        raise ValueError(error_message)
    return supplied_value


# ---------------------------------------------------------------------------
# Shape-JSON reader.
# ---------------------------------------------------------------------------
def _as_vertex_array(value: Any) -> np.ndarray | None:
    """Return value as a finite (N, 3) float array when that is possible."""

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None

    if (
        array.ndim == 2
        and array.shape[1] == 3
        and array.shape[0] >= 4
        and np.all(np.isfinite(array))
    ):
        return array
    return None


def _collect_vertex_candidates(
    obj: Any,
    key_hint: str = "",
) -> list[tuple[int, np.ndarray, str]]:
    """Recursively find possible N x 3 arrays in common shape-JSON layouts."""

    candidates: list[tuple[int, np.ndarray, str]] = []

    direct = _as_vertex_array(obj)
    if direct is not None:
        # Prefer arrays stored under keys such as 'vertices'.  The number of
        # rows is a secondary preference, allowing a large unrelated N x 3
        # array to lose to a clearly named vertex field.
        key_priority = 10_000 if "vert" in key_hint.lower() else 0
        candidates.append((key_priority + direct.shape[0], direct, key_hint))
        return candidates

    if isinstance(obj, dict):
        # Search likely vertex keys first for deterministic behaviour.
        ordered_items = sorted(
            obj.items(),
            key=lambda item: (
                "vert" not in str(item[0]).lower(),
                str(item[0]),
            ),
        )
        for key, value in ordered_items:
            candidates.extend(_collect_vertex_candidates(value, str(key)))

    elif isinstance(obj, (list, tuple)):
        for value in obj:
            candidates.extend(_collect_vertex_candidates(value, key_hint))

    return candidates


def read_shape_vertices(shape_file: str | Path) -> np.ndarray:
    """Read the particle vertices from a JSON file without project modules."""

    path = Path(shape_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Shape JSON not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    candidates = _collect_vertex_candidates(data)
    if not candidates:
        raise ValueError(
            "Could not locate a finite N x 3 vertex array in the shape JSON. "
            "The file should contain a field such as 'vertices'."
        )

    _, vertices, selected_key = max(candidates, key=lambda item: item[0])
    vertices = np.asarray(vertices, dtype=np.float64)

    # Reject exact/near-exact duplicate vertices.  Duplicate vertices make the
    # convex topology and the one-to-one symmetry permutation ambiguous.
    centered = vertices - np.mean(vertices, axis=0)
    characteristic_radius = float(np.max(np.linalg.norm(centered, axis=1)))
    if characteristic_radius <= np.finfo(float).eps:
        raise ValueError("All vertices collapse to one point.")

    duplicate_tolerance = max(characteristic_radius * 1.0e-12, 1.0e-14)
    tree = cKDTree(vertices)
    duplicate_pairs = tree.query_pairs(duplicate_tolerance)
    if duplicate_pairs:
        example = next(iter(duplicate_pairs))
        raise ValueError(
            "The shape JSON contains duplicate or numerically indistinguishable "
            f"vertices; example indices: {example}."
        )

    print(f"Shape vertex array selected from JSON key/path hint: {selected_key!r}")
    print(f"Number of shape vertices: {len(vertices)}")
    return vertices


# ---------------------------------------------------------------------------
# Convex decomposition with explicit topology validation.
# ---------------------------------------------------------------------------
def _merge_coplanar_hull_triangles(
    centered_vertices: np.ndarray,
    merge_tolerance: float,
) -> ConvexDecomposition:
    """Merge coplanar Qhull triangles into polygonal faces.

    This follows the scientific intent of the original ``convexHull`` routine:
    Qhull first triangulates the surface, and triangles with nearly identical
    plane equations are grouped into one polygonal face.
    """

    hull = ConvexHull(centered_vertices)

    # Qhull returns one equation [nx, ny, nz, b] per triangular facet.
    # For a convex hull these normals are consistently outward-facing, so
    # Euclidean proximity of the equation rows is a useful coplanarity test.
    equation_tree = cKDTree(hull.equations)
    close_pairs = equation_tree.query_pairs(merge_tolerance)

    connectivity = np.zeros(
        (len(hull.simplices), len(hull.simplices)),
        dtype=np.int8,
    )
    for first, second in close_pairs:
        connectivity[first, second] = 1
        connectivity[second, first] = 1

    _, component_labels = connected_components(
        connectivity,
        directed=False,
        return_labels=True,
    )

    faces: list[np.ndarray] = []
    face_equations: list[np.ndarray] = []

    for component in sorted(np.unique(component_labels)):
        triangle_indices = np.where(component_labels == component)[0]
        face_vertex_indices = np.unique(hull.simplices[triangle_indices].ravel())

        # Average the nearly identical plane equations, then renormalise the
        # normal and offset together so that ||n|| = 1.
        mean_equation = np.mean(hull.equations[triangle_indices], axis=0)
        normal_norm = float(np.linalg.norm(mean_equation[:3]))
        if normal_norm <= np.finfo(float).eps:
            raise RuntimeError("A merged hull face has an invalid zero normal.")
        mean_equation = mean_equation / normal_norm

        # Sort the polygon vertices around their face centroid.  The ordering is
        # used only to recover edges, not for the later symmetry comparison.
        face_points = centered_vertices[face_vertex_indices]
        face_centroid = np.mean(face_points, axis=0)
        face_normal = mean_equation[:3]

        first_in_plane = face_points[0] - face_centroid
        first_norm = float(np.linalg.norm(first_in_plane))
        if first_norm <= np.finfo(float).eps:
            raise RuntimeError("Cannot construct an in-plane face basis.")
        plane_a = first_in_plane / first_norm
        plane_b = np.cross(face_normal, plane_a)
        plane_b_norm = float(np.linalg.norm(plane_b))
        if plane_b_norm <= np.finfo(float).eps:
            raise RuntimeError("Degenerate polygon face encountered.")
        plane_b /= plane_b_norm

        displacements = face_points - face_centroid
        azimuth = np.arctan2(
            displacements @ plane_b,
            displacements @ plane_a,
        )
        ordered_indices = face_vertex_indices[np.argsort(azimuth)]

        faces.append(np.asarray(ordered_indices, dtype=np.int64))
        face_equations.append(np.asarray(mean_equation, dtype=np.float64))

    edge_set: set[tuple[int, int]] = set()
    for face in faces:
        for first, second in zip(face, np.roll(face, -1)):
            edge_set.add((min(int(first), int(second)), max(int(first), int(second))))

    return ConvexDecomposition(
        vertices=np.asarray(centered_vertices, dtype=np.float64),
        edges=tuple(sorted(edge_set)),
        faces=tuple(faces),
        face_equations=np.asarray(face_equations, dtype=np.float64),
        merge_tolerance=float(merge_tolerance),
        volume=float(hull.volume),
    )


def find_validated_convex_decomposition(
    centered_vertices: np.ndarray,
    expected_edges: int,
    expected_faces: int,
) -> ConvexDecomposition:
    """Scan the original tolerance sequence and REQUIRE a topology match."""

    print("\nConvex-hull topology scan")
    print("--------------------------")
    print(
        f"User-supplied expected topology: edges={expected_edges}, "
        f"faces={expected_faces}, vertices={len(centered_vertices)}"
    )
    print("merge tolerance       recovered edges       recovered faces")

    # Keep the final attempted decomposition only so that a useful diagnostic
    # can be printed if none of the scanned tolerances matches.
    last_decomposition: ConvexDecomposition | None = None

    # This is the same sequence as the old code: 1e-12 through 1e-4.
    for exponent in range(12, 3, -1):
        merge_tolerance = 10.0 ** (-exponent)
        # Reconstruct polygonal faces and physical edges at this trial
        # coplanarity tolerance.
        decomposition = _merge_coplanar_hull_triangles(
            centered_vertices,
            merge_tolerance,
        )
        last_decomposition = decomposition

        # Print every trial so the accepted topology is auditable.
        print(
            f"{merge_tolerance: .1e}"
            f"{len(decomposition.edges):>22d}"
            f"{len(decomposition.faces):>22d}"
        )

        # Accept only an exact match to both user-supplied topological counts.
        if (
            len(decomposition.edges) == expected_edges
            and len(decomposition.faces) == expected_faces
        ):
            print(
                "TOPOLOGY CHECK: PASSED.  The recovered edge and face counts "
                "match the user inputs."
            )
            print(
                f"Accepted coplanar-face merge tolerance: "
                f"{decomposition.merge_tolerance:.1e}"
            )
            print(f"Convex-hull volume from the shape file: {decomposition.volume:.12g}")
            return decomposition

    # The loop always has at least one iteration.  This assertion communicates
    # that invariant to Python and static type checkers.
    assert last_decomposition is not None

    # No tolerance reproduced the known topology.  Stop rather than silently
    # using the final, potentially wrong, trial.
    raise RuntimeError(
        "TOPOLOGY CHECK: FAILED. No tested coplanar-face merge tolerance "
        f"reproduced edges={expected_edges} and faces={expected_faces}. "
        f"The final trial produced edges={len(last_decomposition.edges)} and "
        f"faces={len(last_decomposition.faces)}. Check the shape JSON and the "
        "entered topology before continuing."
    )


# ---------------------------------------------------------------------------
# Candidate axes and historical candidate angles.
# ---------------------------------------------------------------------------
def _canonical_axis_line(axis: np.ndarray) -> np.ndarray:
    """Normalise an axis and choose one deterministic sign for its line."""

    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm <= np.finfo(float).eps:
        raise ValueError("Cannot canonicalise a zero axis.")
    # Convert the direction to a unit vector.
    axis = axis / norm

    # Axis n and -n describe the same geometric line.  Choose the sign by the
    # first component that is numerically nonzero.
    for component in axis:
        if abs(component) > 1.0e-14:
            if component < 0.0:
                axis = -axis
            break
    return axis


def build_candidate_axis_lines(
    centered_vertices: np.ndarray,
    decomposition: ConvexDecomposition,
) -> list[tuple[str, np.ndarray]]:
    """Construct the same classes of candidate axes as the original detector.

    Candidate directions come from:
        * centre -> vertex;
        * centre -> face centroid;
        * centre -> perpendicular foot on each face plane;
        * centre -> edge midpoint.
    """

    # Each raw item is stored as: (human-readable label, vector).
    #
    # The label is used for traceability; the vector is later normalised.
    raw_candidates: list[tuple[str, np.ndarray]] = []

    # -----------------------------------------------------------------------
    # Candidate class 1: centre-to-vertex directions.
    # -----------------------------------------------------------------------
    for index, vertex in enumerate(centered_vertices):
        raw_candidates.append((f"vertex[{index}]", vertex))

    # -----------------------------------------------------------------------
    # Candidate classes 2 and 3: face-centroid and face-normal directions.
    # -----------------------------------------------------------------------
    for face_index, (face, equation) in enumerate(
        zip(decomposition.faces, decomposition.face_equations)
    ):
        # The average position of a face's vertices gives its polygon centroid.
        # In centred coordinates, this is directly the centre-to-face-centroid
        # vector.
        face_centroid = np.mean(centered_vertices[face], axis=0)
        raw_candidates.append((f"face_centroid[{face_index}]", face_centroid))

        # In centered coordinates the plane is n dot x + b = 0.  With unit n,
        # the perpendicular foot from the origin is -b*n.  This is the corrected
        # version of the old centre-to-face-normal construction.
        normal = equation[:3]
        offset = float(equation[3])
        perpendicular_foot = -offset * normal
        raw_candidates.append((f"face_normal[{face_index}]", perpendicular_foot))

    # -----------------------------------------------------------------------
    # Candidate class 4: centre-to-edge-midpoint directions.
    # -----------------------------------------------------------------------
    for edge_index, (first, second) in enumerate(decomposition.edges):
        midpoint = 0.5 * (centered_vertices[first] + centered_vertices[second])
        raw_candidates.append((f"edge_midpoint[{edge_index}]", midpoint))

    # =======================================================================
    # REMOVE ZERO VECTORS AND DUPLICATE AXIS LINES
    # =======================================================================
    unique_lines: list[tuple[str, np.ndarray]] = []

    for label, vector in raw_candidates:
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-14:
            # A vector through the centre with zero length does not define an axis.
            continue

        # Normalise the vector and impose the deterministic n/-n sign convention.
        canonical = _canonical_axis_line(vector)

        # Check whether the same axis line is already present.
        #
        # For unit vectors a and b:
        #
        #       |a \dot b| = 1  means parallel or antiparallel,
        #
        # and therefore the same geometric line.  The absolute value removes
        # the remaining sign distinction.
        if any(
            abs(float(np.dot(canonical, previous_axis))) >= 1.0 - 1.0e-12
            for _, previous_axis in unique_lines
        ):
            continue

        # Keep the first label that generated this unique geometric line.
        unique_lines.append((label, canonical))

    # A shape with no nonzero candidate direction cannot be analysed
    if not unique_lines:
        raise RuntimeError("No nonzero candidate symmetry axes were generated.")

    print(f"Unique candidate axis lines generated: {len(unique_lines)}")
    return unique_lines


def historical_candidate_angles_rad() -> np.ndarray:
    """
        Return the existing hard-coded candidate-angle list
    """

    angles_deg = [
        180, 120, 240, 90, 270,
        72, 144, 216, 288, 60,
        300, 45, 135, 225, 315, 252, 324,
        36, 108,
        -120, -240, -90, -270,
        -72, -144, -216, -288, -60,
        -300, -45, -135, -225, -315,
        -36, -108, -252, -324,
    ]
    return np.deg2rad(np.asarray(angles_deg, dtype=np.float64))


# ---------------------------------------------------------------------------
# Proper-rotation validation and full-precision refinement.
# ---------------------------------------------------------------------------
def _rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    """Convert SciPy's [x,y,z,w] quaternion to freud/HOOMD [w,x,y,z]."""

    xyzw = np.asarray(rotation.as_quat(), dtype=np.float64)
    wxyz = xyzw[[3, 0, 1, 2]]
    wxyz /= np.linalg.norm(wxyz)

    # Choose one deterministic representative for the physical rotation.
    # Both signs are added explicitly only after all physical rotations are found.
    for component in wxyz:
        if abs(component) > 1.0e-14:
            if component < 0.0:
                wxyz = -wxyz
            break
    return wxyz


def _one_to_one_vertex_mapping(
    rotated_vertices: np.ndarray,
    reference_vertices: np.ndarray,
) -> tuple[tuple[int, ...], np.ndarray]:
    """
        Find the globally optimal one-to-one rotated->reference assignment.
    """

    # =======================================================================
    # FORM THE COMPLETE ASSIGNMENT-COST MATRIX
    # =======================================================================
    #
    # costs[i, j] is the Euclidean distance between transformed source vertex i
    # and reference vertex j:
    #
    #                 costs[i,j] = ||r_i(rotated) - r_j(reference)||.
    #
    # For N vertices, this array has shape (N, N).
    costs = distance_matrix(rotated_vertices, reference_vertices)

    # Solve the global minimum-cost bipartite assignment problem using the
    # Hungarian algorithm.
    #
    # The result guarantees:
    #     * each rotated/source vertex is assigned once;
    #     * each reference/target vertex is used once.
    #
    # This is the required permutation condition for a rigid symmetry operation.
    row_indices, column_indices = linear_sum_assignment(costs)

    # For two equally sized vertex sets, a complete assignment must contain one
    # row for every reference vertex.  Reject an incomplete result explicitly.
    if len(row_indices) != len(reference_vertices):
        raise RuntimeError("One-to-one vertex assignment is incomplete.")

    # mapping[source_index] = target_index.
    mapping = np.empty(len(reference_vertices), dtype=np.int64)
    mapping[row_indices] = column_indices

    # Extract the Euclidean residual of every selected source-target assignment.
    residuals = costs[row_indices, column_indices]

    return tuple(int(value) for value in mapping), np.asarray(residuals)


def _refine_rotation_for_permutation(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
) -> Rotation:
    """Return the least-squares proper rotation mapping source onto target.

    Candidate axes are obtained from finite-precision shape coordinates.  Once a
    vertex permutation is known, fitting all vertices simultaneously gives a
    more accurate full-precision rotation than retaining the raw candidate axis.
    """

    # =======================================================================
    # SOLVE THE ORTHOGONAL ALIGNMENT PROBLEM
    # =======================================================================
    #
    # target_vertices has already been reordered according to the discovered
    # permutation, so row i is the intended image of source row i.
    #
    # scipy Rotation.align_vectors(a, b) finds the proper rotation R satisfying
    #
    #                         a ~= R.apply(b)
    #
    # in the least-squares sense.  Therefore use
    #
    #                         a = target_vertices,
    #                         b = source_vertices.
    #
    # The returned second value is an aggregate fitting diagnostic; the program
    # independently recomputes individual vertex residuals afterwards, so it is
    # intentionally assigned to "_".
    refined_rotation, _ = Rotation.align_vectors(target_vertices, source_vertices)

    if np.linalg.det(refined_rotation.as_matrix()) < 0.0:
        # SciPy's Rotation should always be proper, but retain an explicit guard.
        raise RuntimeError("Refined symmetry operation is not a proper rotation.")
    return refined_rotation


def _compose_permutations(
    first: tuple[int, ...],
    second: tuple[int, ...],
) -> tuple[int, ...]:
    """Permutation produced by applying first, then second."""

    # A permutation tuple p is represented by:
    #
    #                         source i -> p[i].
    #
    # Applying "first" sends i to first[i].  Applying "second" afterwards sends
    # that intermediate index to second[first[i]].  Therefore the composed
    # mapping is:
    #
    #                  (second o first)[i] = second[first[i]].
    return tuple(second[first[index]] for index in range(len(first)))


def validate_permutation_group(permutations: Sequence[tuple[int, ...]]) -> None:
    """Check identity, inverses, and closure of the detected operations."""

    # =======================================================================
    # WHY VALIDATE IN PERMUTATION SPACE?
    # =======================================================================
    #
    # Rotation matrices and quaternions contain floating-point noise.  Their
    # action on a finite vertex set is represented exactly by integer
    # permutations, making group checks discrete and unambiguous.
    #
    # A valid finite rotational symmetry group must contain:
    #     1. identity;
    #     2. an inverse for every operation;
    #     3. the composition of every pair of operations.

    if not permutations:
        raise RuntimeError("No symmetry permutations were detected.")

    # A set provides exact and efficient membership tests.
    permutation_set = set(permutations)

    # For N vertices, identity maps every index to itself.
    identity = tuple(range(len(permutations[0])))
    if identity not in permutation_set:
        raise RuntimeError("Detected symmetry set does not contain identity.")

    # -----------------------------------------------------------------------
    # Check inverses.
    # -----------------------------------------------------------------------
    #
    # If permutation[source] = target, then
    #
    #                         inverse[target] = source.
    #
    # Every computed inverse must already be present in the detected set.
    for permutation in permutations:
        inverse = [0] * len(permutation)
        for source, target in enumerate(permutation):
            inverse[target] = source
        if tuple(inverse) not in permutation_set:
            raise RuntimeError(
                "Detected symmetry set is missing an inverse operation; "
                "the candidate-angle search is incomplete or inconsistent."
            )

    # -----------------------------------------------------------------------
    # Check closure under composition.
    # -----------------------------------------------------------------------
    #
    # For every ordered pair (first, second), applying first and then second
    # must produce another detected group element.
    for first in permutations:
        for second in permutations:
            composed = _compose_permutations(first, second)
            if composed not in permutation_set:
                raise RuntimeError(
                    "Detected symmetry operations are not closed under "
                    "composition; the candidate-angle search is incomplete."
                )

    print("ROTATIONAL-GROUP CHECK: PASSED (identity, inverses, and closure).")


def detect_proper_rotational_symmetries(
    vertices: np.ndarray,
    expected_edges: int,
    expected_faces: int,
    precision_exponent: int,
) -> SymmetryResult:
    """Detect proper rotational symmetries with all requested validations."""

    # =======================================================================
    # STAGE 1 — INTERPRET THE USER-SUPPLIED PRECISION
    # =======================================================================
    if precision_exponent < 0:
        raise ValueError("The symmetry precision exponent must be nonnegative.")

    # Convert the terminal input p into an absolute Euclidean distance
    # tolerance in the coordinate units of the shape file.
    matching_tolerance = 10.0 ** (-precision_exponent)

    # =======================================================================
    # STAGE 2 — FIND AND REMOVE ANY TRANSLATION OF THE SHAPE
    # =======================================================================
    #
    # A rigid-body rotation must be applied about the particle centre.
    # Rotating vertices directly about the coordinate origin is valid only
    # when the shape file is already centred at the origin.
    #
    original_center = np.mean(vertices, axis=0)
    # Translate every vertex so that the arithmetic centre becomes the origin:
    centered_vertices = np.asarray(vertices - original_center, dtype=np.float64)

    # The norm quantifies how far the original shape file was displaced from
    # the origin.  A value near zero means that the shape was already centred.
    center_norm = float(np.linalg.norm(original_center))

    print("\nShape-centre validation")
    print("-----------------------")
    print(f"Original arithmetic vertex centre: {original_center}")
    print(f"Norm of original centre: {center_norm:.12g}")
    print(
        f"User-selected symmetry matching tolerance: 10^(-{precision_exponent}) "
        f"= {matching_tolerance:.12g} shape-length units"
    )

    if center_norm <= matching_tolerance:
        print("CENTRE CHECK: PASSED within the selected tolerance.")
    else:
        print(
            "CENTRE CHECK: original vertices are not centred within the selected "
            "tolerance. The program will translate them to their arithmetic "
            "vertex centre before every topology and symmetry operation."
        )

    # Verify that internal recentering has actually removed the translation.
    residual_center = np.mean(centered_vertices, axis=0)
    if np.linalg.norm(residual_center) > 100.0 * np.finfo(float).eps:
        raise RuntimeError("Internal recentering failed numerically.")
    print(f"Internal centred-coordinate residual: {residual_center}")


    # =======================================================================
    # STAGE 3 — VALIDATE THE CONVEX POLYHEDRON TOPOLOGY
    # =======================================================================
    decomposition = find_validated_convex_decomposition(
        centered_vertices,
        expected_edges,
        expected_faces,
    )

    # =======================================================================
    # STAGE 4 — BUILD THE FINITE SEARCH SPACE
    # =======================================================================
    candidate_axes = build_candidate_axis_lines(centered_vertices, decomposition)

    candidate_angles = historical_candidate_angles_rad()

    print("\nProper rotational-symmetry search")
    print("---------------------------------")
    print(f"Candidate angle count: {len(candidate_angles)}")
    print(
        "Accepted candidates must map all vertices one-to-one with maximum "
        f"Euclidean residual <= {matching_tolerance:.12g}."
    )

    # =======================================================================
    # STAGE 5 — PREPARE A PHYSICALLY ROBUST DEDUPLICATION CONTAINER
    # =======================================================================
    # Store operations by their induced vertex permutation.  
    operations_by_permutation: dict[tuple[int, ...], SymmetryOperation] = {}

    # Explicitly seed the group with the identity operation.
    identity_permutation = tuple(range(len(centered_vertices)))
    identity_rotation = Rotation.identity()
    operations_by_permutation[identity_permutation] = SymmetryOperation(
        quaternion_wxyz=_rotation_to_wxyz(identity_rotation),
        rotation_matrix=identity_rotation.as_matrix(),
        permutation=identity_permutation,
        max_vertex_residual=0.0,
        discovery_axis=np.array([1.0, 0.0, 0.0]),
        discovery_angle_deg=0.0,
    )

    tested_candidates = 0
    accepted_discoveries = 0

    # =======================================================================
    # STAGE 6 — TEST EVERY CANDIDATE AXIS/ANGLE COMBINATION
    # =======================================================================

    for _, axis in candidate_axes:
        # Test every retained rotation angle around the current axis.
        for angle_rad in candidate_angles:
            tested_candidates += 1

            # Construct the tentative proper rotation from the rotation vector
            #
            #                         omega = n * theta,
            #
            # where n is the unit axis and theta is the angle in radians.
            candidate_rotation = Rotation.from_rotvec(axis * angle_rad)

            # Apply the tentative rotation to every vertex.
            candidate_rotated = candidate_rotation.apply(centered_vertices)

            # Find the globally optimal one-to-one assignment between rotated
            # vertices and original centred vertices.
            permutation, assignment_residuals = _one_to_one_vertex_mapping(
                candidate_rotated,
                centered_vertices,
            )

            # Reject the candidate when even one assigned vertex lies farther
            # than epsilon_match from its target.
            #
            # Using the maximum residual enforces the symmetry condition for
            # the complete vertex set, not merely in an average-RMS sense.
            if float(np.max(assignment_residuals)) > matching_tolerance:
                continue

            # =================================================================
            # STAGE 7 — REFIT THE DISCOVERED PERMUTATION AT FULL PRECISION
            # =================================================================
            # The raw candidate has discovered a valid permutation.  Refit the
            # proper rotation using all source/target vertex pairs at full
            # floating-point precision, then validate the refined result again.
            target_vertices = centered_vertices[np.asarray(permutation)]
            refined_rotation = _refine_rotation_for_permutation(
                centered_vertices,
                target_vertices,
            )

            # Apply the full-precision fitted rotation to all source vertices.
            refined_rotated = refined_rotation.apply(centered_vertices)

            # Independently determine the one-to-one mapping induced by the
            # refined rotation.  This is not assumed to equal the original
            # discovery mapping; it is explicitly checked below.
            refined_permutation, refined_residuals = _one_to_one_vertex_mapping(
                refined_rotated,
                centered_vertices,
            )

            # The largest fitted source-target distance is the strict residual
            # used to assess and compare accepted operations.
            refined_max_residual = float(np.max(refined_residuals))

            # If refinement changes the permutation, then the selected tolerance
            # does not resolve the vertex correspondence uniquely.  Continuing
            # would produce an ambiguous symmetry classification, so stop and
            # ask the user to use a tighter precision.
            if refined_permutation != permutation:
                raise RuntimeError(
                    "Full-precision rotation refinement changed the discovered "
                    "vertex permutation; symmetry assignment is ambiguous at "
                    "the selected tolerance. Use a tighter precision exponent."
                )

            # A candidate can pass the rough discovery test but fail after
            # full-precision refitting and reassignment.  Retain it only when
            # the strict refined maximum residual also satisfies the user
            # tolerance.
            if refined_max_residual > matching_tolerance:
                continue

            accepted_discoveries += 1

            # Package the accepted full-precision physical operation.
            operation = SymmetryOperation(
                # Canonical scalar-first quaternion for this proper rotation.
                # No decimal rounding is performed.
                quaternion_wxyz=_rotation_to_wxyz(refined_rotation),
                # Full double-precision 3 x 3 matrix.
                rotation_matrix=refined_rotation.as_matrix(),
                # Exact discrete identity of the operation on the vertex set.
                permutation=permutation,
                # Worst source-target Euclidean mismatch after refinement.
                max_vertex_residual=refined_max_residual,
                # Preserve the axis and angle that originally discovered this
                # permutation.  They are diagnostic metadata, not the final
                # numerical definition of the refined operation.
                discovery_axis=np.asarray(axis, dtype=np.float64),
                discovery_angle_deg=float(np.rad2deg(angle_rad)),
            )

            # Several candidate axes or angle representations can discover the
            # same permutation.  Keep only the discovery having the smallest
            # full-precision maximum vertex residual.
            previous = operations_by_permutation.get(permutation)
            if previous is None or operation.max_vertex_residual < previous.max_vertex_residual:
                operations_by_permutation[permutation] = operation

    # =======================================================================
    # STAGE 8 — CONVERT THE DEDUPLICATED DICTIONARY TO A STABLE ORDER
    # =======================================================================
    physical_operations = tuple(
        sorted(
            operations_by_permutation.values(),
            key=lambda operation: (
                np.linalg.norm(operation.rotation_matrix - np.eye(3)),
                operation.permutation,
            ),
        )
    )

    print(f"Raw axis/angle candidates tested: {tested_candidates}")
    print(f"Valid discovery hits before permutation deduplication: {accepted_discoveries}")
    print(f"Distinct physical proper rotations detected: {len(physical_operations)}")

    # Report the worst accepted geometric residual.  Because identity is always
    # present, physical_operations is guaranteed to be nonempty here.
    max_residual = max(op.max_vertex_residual for op in physical_operations)
    print(f"Largest accepted full-precision vertex residual: {max_residual:.12g}")

    # =======================================================================
    # STAGE 9 — VERIFY THAT THE DETECTED OPERATIONS FORM A GROUP
    # =======================================================================
    validate_permutation_group([op.permutation for op in physical_operations])

    # Extract exactly one canonical +q representative for each physical proper
    # rotation.  The order matches physical_operations.
    physical_quaternions = np.asarray(
        [op.quaternion_wxyz for op in physical_operations],
        dtype=np.float64,
    )

    # =======================================================================
    # STAGE 10 — EXPLICITLY GENERATE BOTH QUATERNION SIGNS
    # =======================================================================
    #
    # Unit quaternions double-cover SO(3):
    #
    #                           q and -q
    #
    # represent the same physical three-dimensional rotation.
    equivalent_quaternions = np.empty(
        (2 * len(physical_quaternions), 4),
        dtype=np.float64,
    )
    equivalent_quaternions[0::2] = physical_quaternions
    equivalent_quaternions[1::2] = -physical_quaternions

    # =======================================================================
    # STAGE 11 — VALIDATE THE FINAL QUATERNION ARRAY
    # =======================================================================
    #
    # Every rotation quaternion supplied to freud must be unit length.
    # A tight absolute tolerance is used because these quaternions are generated
    # by scipy Rotation and are expected to be normalised to near machine
    # precision.
    quaternion_norms = np.linalg.norm(equivalent_quaternions, axis=1)
    if not np.allclose(quaternion_norms, 1.0, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("Generated equivalent quaternions are not unit length.")

    # Verify exact array construction of every adjacent q/-q pair.  Since the
    # negative rows were produced by direct unary negation, exact equality is
    # expected and zero numerical tolerance is appropriate.
    if not np.allclose(
        equivalent_quaternions[1::2],
        -equivalent_quaternions[0::2],
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("Explicit q/-q pairing failed.")

    print(
        f"Equivalent quaternion representatives passed to freud: "
        f"{len(equivalent_quaternions)} (= 2 x physical rotations)"
    )
    print("Q/-Q CHECK: PASSED for every physical rotational symmetry.")


    # =======================================================================
    # STAGE 12 — PRINT AN AUDITABLE SUMMARY OF THE PHYSICAL ROTATIONS
    # =======================================================================
    #
    # Only one canonical +q is printed per physical operation here.  The -q
    # companions are still present in equivalent_quaternions and are written to
    # the symmetry output file by the main workflow.
    print("\nDetected physical proper rotations (one canonical +q each)")
    print("----------------------------------------------------------")
    print(" index             w             x             y             z       max residual")
    for operation_index, operation in enumerate(physical_operations):
        w, x, y, z = operation.quaternion_wxyz
        print(
            f"{operation_index:6d} "
            f"{w: .10f} {x: .10f} {y: .10f} {z: .10f} "
            f"{operation.max_vertex_residual: .3e}"
        )

    # =======================================================================
    # STAGE 13 — RETURN ALL SYMMETRY DATA NEEDED BY THE REST OF THE PROGRAM
    # =======================================================================
    return SymmetryResult(
        centered_vertices=centered_vertices,
        original_center=original_center,
        decomposition=decomposition,
        physical_operations=physical_operations,
        equivalent_quaternions_wxyz=equivalent_quaternions,
        matching_tolerance=matching_tolerance,
    )


# ---------------------------------------------------------------------------
# GSD and quaternion validation.
# ---------------------------------------------------------------------------
def import_runtime_packages():
    """Import optional runtime packages with one actionable error message."""

    try:
        import freud  # type: ignore
        import gsd.hoomd  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise ImportError(
            "Missing runtime dependency. Install all required packages with:\n"
            "    pip install numpy scipy matplotlib gsd freud-analysis"
        ) from exc

    return freud, gsd.hoomd, plt


def open_gsd_trajectory(gsd_hoomd_module: Any, path: str | Path):
    """Open a GSD trajectory for reading across common GSD API versions."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"GSD trajectory not found: {resolved}")

    try:
        trajectory = gsd_hoomd_module.open(name=str(resolved), mode="r")
    except TypeError:
        # Compatibility fallback for versions that prefer positional arguments.
        trajectory = gsd_hoomd_module.open(str(resolved), "r")

    return resolved, trajectory


def validate_and_normalize_orientations(
    orientations: np.ndarray,
    requested_particles: int,
    frame_index: int,
) -> tuple[np.ndarray, float]:
    """Validate scalar-first quaternions and normalise tiny storage deviations."""

    # =======================================================================
    # STAGE 1 — CONVERT THE SNAPSHOT DATA TO A NUMERIC ARRAY
    # =======================================================================
    array = np.asarray(orientations, dtype=np.float64)

    # Reject malformed orientation data before any slicing or arithmetic.
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(
            f"Frame {frame_index}: particle orientations must have shape (N, 4); "
            f"found {array.shape}."
        )
    if len(array) < requested_particles:
        raise ValueError(
            f"Frame {frame_index}: contains {len(array)} orientations, fewer than "
            f"the requested {requested_particles}."
        )

    # =======================================================================
    # STAGE 2 — APPLY THE USER'S PARTICLE-SELECTION CONVENTION
    # =======================================================================
    #
    # Preserve the original convention: analyse the first M particles rather
    # than sampling random particles or selecting them by type.
    #
    array = np.ascontiguousarray(array[:requested_particles], dtype=np.float64)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"Frame {frame_index}: orientations contain NaN/inf values.")

    # =======================================================================
    # STAGE 3 — CHECK QUATERNION NORMS
    # =======================================================================
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms <= 1.0e-14):
        raise ValueError(f"Frame {frame_index}: contains a zero quaternion.")

    # Record the largest deviation from unit length BEFORE correction:
    #
    #                  max_i | ||q_i|| - 1 |.
    #
    # This value is returned for terminal and metadata diagnostics.
    max_norm_deviation = float(np.max(np.abs(norms - 1.0)))

    # =======================================================================
    # STAGE 4 — NORMALISE SMALL STORAGE DEVIATIONS
    # =======================================================================
    array /= norms[:, None]

    return array, max_norm_deviation


# ---------------------------------------------------------------------------
# Memory-efficient unique-pair histogram using freud in query blocks.
# ---------------------------------------------------------------------------
def compute_one_frame_histogram(
    freud_module: Any,
    orientations: np.ndarray,
    equivalent_quaternions: np.ndarray,
    frame_index: int,
    bin_edges_deg: np.ndarray,
    block_size: int,
    max_quaternion_norm_deviation: float,
) -> FrameHistogram:
    """Compute one frame using only unique pairs i < j.

    ``AngularSeparationGlobal.compute(global_orientations, orientations, ...)``
    returns an array with shape:

        (number of query orientations, number of global orientations)

    We therefore provide all selected particles as ``global_orientations`` and
    only a small contiguous block as ``orientations``.  For global row i, only
    columns j > i are histogrammed.
    """

    # =======================================================================
    # SCIENTIFIC PAIR DEFINITION
    # =======================================================================
    #
    # If this frame contains M selected particles, the analysis retains only
    # unique unordered pairs:
    #
    #                           i < j.
    particle_count = len(orientations)
    expected_unique_pairs = particle_count * (particle_count - 1) // 2

    # =======================================================================
    # INITIALISE ACCUMULATORS
    # =======================================================================
    #
    # raw_counts[k] will hold the number of in-range pair angles falling in
    # histogram bin k.  It is integer-valued because no weighting occurs.
    raw_counts = np.zeros(len(bin_edges_deg) - 1, dtype=np.int64)

    # This counter includes every retained i<j pair, whether its angle falls
    # inside or outside the plotted histogram range.  It is later checked
    # against M(M-1)/2.
    processed_unique_pairs = 0

    # Track the true minimum and maximum over ALL unique pair angles, not just
    # angles inside the plotted range.  Start with sentinel infinities that are
    # replaced by the first nonempty row.
    global_min_angle = math.inf
    global_max_angle = -math.inf

    # Construct one freud calculator object and reuse it for all query blocks in this frame.
    angular_separation = freud_module.environment.AngularSeparationGlobal()


    # =======================================================================
    # PROCESS QUERY PARTICLES IN MEMORY-LIMITED BLOCKS
    # =======================================================================
    #
    # The second freud orientation argument contains only a small query block.
    # The first contains all M selected orientations.  Thus each call returns a
    # matrix of size approximately:
    #
    #                       block_size x M,
    #
    # instead of materialising the complete M x M matrix at once.
    for block_start in range(0, particle_count, block_size):
        # The final block may contain fewer than block_size particles.
        block_stop = min(block_start + block_size, particle_count)

        # This contiguous slice corresponds to global particle indices:
        #
        #          block_start, ..., block_stop - 1.
        query_orientations = orientations[block_start:block_stop]

        # Ask freud for the symmetry-reduced angular separation between:
        #
        #     each query orientation in query_orientations
        #
        # and
        #
        #     every selected orientation in orientations,
        #
        # minimising over the supplied equivalent body-symmetry quaternions.
        angular_separation.compute(
            orientations,
            query_orientations,
            equivalent_quaternions,
        )
        block_angles_deg = np.rad2deg(
            np.asarray(angular_separation.angles, dtype=np.float64)
        )

        expected_shape = (block_stop - block_start, particle_count)
        if block_angles_deg.shape != expected_shape:
            raise RuntimeError(
                "Unexpected freud AngularSeparationGlobal output shape. "
                f"Expected {expected_shape}, received {block_angles_deg.shape}. "
                "This code follows the documented ordering "
                "(N_orientations, N_global_orientations)."
            )

        # ===================================================================
        # EXTRACT ONLY THE UPPER-TRIANGLE PAIRS i < j
        # ===================================================================
        #
        # Avoid constructing a large block_size x M boolean mask.  Instead,
        # process each query row individually and take a NumPy view beginning at
        # column particle_i + 1.
        for local_row, particle_i in enumerate(range(block_start, block_stop)):
            # Row local_row corresponds to global particle i = particle_i.
            #
            # Columns 0 ... particle_i - 1 would duplicate already counted
            # pairs (j,i).  Column particle_i is the self-pair.  Retain only:
            #
            #                    particle_i + 1 ... M - 1.
            unique_pair_angles = block_angles_deg[local_row, particle_i + 1 :]

            # The final particle has no j>i partner, so its slice is empty.
            if unique_pair_angles.size == 0:
                continue

            # Count all extracted unique pairs, independent of histogram range.
            processed_unique_pairs += int(unique_pair_angles.size)

            # Update true global minimum/maximum diagnostics using this row.
            global_min_angle = min(global_min_angle, float(np.min(unique_pair_angles)))
            global_max_angle = max(global_max_angle, float(np.max(unique_pair_angles)))

            # Histogram only values falling within bin_edges_deg.  NumPy ignores
            # values below the first edge or above the last edge.
            row_histogram, _ = np.histogram(
                unique_pair_angles,
                bins=bin_edges_deg,
            )

            # Add this row's integer counts immediately, then allow the temporary
            # angle view/block to be released.  No all-pair Python list is kept.
            raw_counts += row_histogram.astype(np.int64, copy=False)

    # =======================================================================
    # VERIFY EXACT UNIQUE-PAIR ACCOUNTING
    # =======================================================================
    #
    # The extraction logic must generate exactly:
    #
    #                         M(M-1)/2
    #
    # values.  Any discrepancy indicates a block-index or upper-triangle error.
    if processed_unique_pairs != expected_unique_pairs:
        raise RuntimeError(
            "Unique-pair accounting failed: "
            f"expected {expected_unique_pairs}, processed {processed_unique_pairs}."
        )

    # raw_counts contains only angles inside the requested plotting range.
    pairs_inside_range = int(np.sum(raw_counts))

    # Every processed pair is either inside or outside that range.
    pairs_outside_range = expected_unique_pairs - pairs_inside_range

    # Conditional normalisation is impossible when no pair lies in the selected
    # range, so fail with the exact frame and interval.
    if pairs_inside_range <= 0:
        raise ValueError(
            f"Frame {frame_index}: no unique pair angles fall inside the plotted "
            f"range [{bin_edges_deg[0]}, {bin_edges_deg[-1]}] degrees."
        )

    # =======================================================================
    # CONDITIONAL PROBABILITY PER BIN
    # =======================================================================
    #
    # The requested interpretation is:
    #
    #   among pair angles INSIDE the plotted range, what fraction lies in bin k?
    #
    # Therefore:
    #
    #             P_f(k) = h_f(k) / sum_l h_f(l)
    #                    = h_f(k) / pairs_inside_range.
    #
    # This is probability mass per bin, not probability density; no division by the bin width is performed.
    conditional_probability = raw_counts.astype(np.float64) / pairs_inside_range

    # The conditional probabilities must sum to one up to floating-point roundoff.
    probability_sum = float(np.sum(conditional_probability))
    if not np.isclose(probability_sum, 1.0, atol=1.0e-12, rtol=0.0):
        raise RuntimeError(
            f"Frame {frame_index}: conditional probabilities sum to "
            f"{probability_sum}, not one."
        )

    # Package both the primary per-frame distribution and all diagnostics in an
    # immutable FrameHistogram object.
    return FrameHistogram(
        frame_index=frame_index,
        raw_counts=raw_counts,
        conditional_probability=conditional_probability,
        total_unique_pairs=expected_unique_pairs,
        pairs_inside_range=pairs_inside_range,
        pairs_outside_range=pairs_outside_range,
        minimum_angle_deg=float(global_min_angle),
        maximum_angle_deg=float(global_max_angle),
        maximum_quaternion_norm_deviation=max_quaternion_norm_deviation,
    )


def compute_frame_averaged_histogram(
    freud_module: Any,
    trajectory: Any,
    frame_indices: Sequence[int],
    particle_count: int,
    equivalent_quaternions: np.ndarray,
    num_bins: int,
    maximum_angle_deg: float,
    block_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[FrameHistogram, ...],
]:
    """Compute and equally average conditional per-frame histograms."""

    # =======================================================================
    # FUNCTIONAL ROLE
    # =======================================================================
    #
    # This function coordinates the complete trajectory-level calculation.
    #
    # For each selected frame f, it constructs a conditional histogram:
    #
    #       P_f(k) = count_f(k) / number of unique pairs inside the range.
    #
    # It then performs the requested equal-weight average:
    #
    #       mean_P(k) = (1/F) sum_f P_f(k).

    # =======================================================================
    # STAGE 1 — CONSTRUCT ONE COMMON HISTOGRAM GRID
    # =======================================================================
    #
    # num_bins bins require num_bins + 1 edges.  np.linspace creates equal-width
    # bins spanning exactly:
    #
    #                       0 <= theta <= maximum_angle_deg.

    bin_edges = np.linspace(
        0.0,
        maximum_angle_deg,
        num_bins + 1,
        dtype=np.float64,
    )
    # Store the complete immutable result object for each selected frame.  These
    # objects are later used for averaging, diagnostics and per-frame CSV output.
    frame_results: list[FrameHistogram] = []

    print("\nTrajectory histogram calculation")
    print("--------------------------------")

    # =======================================================================
    # STAGE 2 — ANALYSE EACH REQUESTED FRAME IN ORDER
    # =======================================================================

    for sequence_number, frame_index in enumerate(frame_indices, start=1):
        # Load the selected GSD snapshot.  The trajectory itself is not copied;
        # snapshot references only the requested frame data.
        snapshot = trajectory[frame_index]
        # Validate the orientation array, select the first particle_count
        # particles and normalise small quaternion norm deviations.
        orientations, norm_deviation = validate_and_normalize_orientations(
            snapshot.particles.orientation,
            particle_count,
            frame_index,
        )

        # Compute this frame's unique-pair conditional histogram using block-wise
        # freud calls.  The equivalent_quaternions array contains both q and -q
        # for every physical body rotation.
        frame_result = compute_one_frame_histogram(
            freud_module=freud_module,
            orientations=orientations,
            equivalent_quaternions=equivalent_quaternions,
            frame_index=frame_index,
            bin_edges_deg=bin_edges,
            block_size=block_size,
            max_quaternion_norm_deviation=norm_deviation,
        )

        # Retain the complete result for the later matrix construction and
        # output files.
        frame_results.append(frame_result)

        # Fraction of all unique i<j pairs whose angles were omitted from the
        # conditional histogram because they lie outside [0, maximum_angle_deg].
        outside_fraction = (
            frame_result.pairs_outside_range / frame_result.total_unique_pairs
        )

        # Print a detailed per-frame audit line:
        print(
            f"[{sequence_number}/{len(frame_indices)}] frame {frame_index}: "
            f"unique pairs={frame_result.total_unique_pairs:,}; "
            f"inside range={frame_result.pairs_inside_range:,}; "
            f"outside range={frame_result.pairs_outside_range:,} "
            f"({outside_fraction:.6%}); "
            f"angle min/max={frame_result.minimum_angle_deg:.6g}/"
            f"{frame_result.maximum_angle_deg:.6g} deg; "
            f"max |norm(q)-1|={norm_deviation:.3e}"
        )

    # =======================================================================
    # STAGE 3 — STACK PER-FRAME PROBABILITY VECTORS
    # =======================================================================
    #
    # If F frames and B bins were analysed, probability_matrix has shape:
    #
    #                              (F, B).
    #
    # Row f is the conditional distribution P_f(k) of one frame.
    probability_matrix = np.vstack(
        [result.conditional_probability for result in frame_results]
    )

    # =======================================================================
    # STAGE 4 — EQUAL-WEIGHT FRAME AVERAGE
    # =======================================================================
    #
    # np.mean along axis 0 averages each bin down the frame dimension:
    #
    #               mean_probability[k] = (1/F) sum_f P_f(k).
    #
    # This is the primary requested curve.
    mean_probability = np.mean(probability_matrix, axis=0)

    # Population standard deviation across the selected frames:
    #
    #       sigma[k] = sqrt((1/F) sum_f (P_f(k)-mean_P(k))^2).
    #
    # ddof=0 is used because the selected frames are treated as the complete set
    # being summarised, not as a sample requiring Bessel's correction.
    standard_deviation = np.std(probability_matrix, axis=0, ddof=0)

    # =======================================================================
    # STAGE 5 — CALCULATE A POOLED DIAGNOSTIC DISTRIBUTION
    # =======================================================================
    #
    # First stack and sum raw integer counts over frames:
    #
    #                 H_pooled(k) = sum_f h_f(k).
    #
    # dtype=int64 preserves exact integer counting during the sum.
    pooled_counts = np.sum(
        np.vstack([result.raw_counts for result in frame_results]),
        axis=0,
        dtype=np.int64,
    )

    # Normalise the summed counts once:
    #
    #         P_pooled(k) = H_pooled(k) / sum_l H_pooled(l).
    #
    # This weights frames in proportion to their number of in-range pairs.
    # It is retained only as a diagnostic and is not the primary plotted curve.
    pooled_probability = pooled_counts / float(np.sum(pooled_counts))

    # =======================================================================
    # STAGE 6 — VALIDATE THE FINAL EQUAL-WEIGHT DISTRIBUTION
    # =======================================================================
    #
    # Every row P_f sums to one.  A mean of those rows must therefore also sum
    # to one:
    #
    #          sum_k mean_P(k)
    #        = (1/F) sum_f sum_k P_f(k)
    #        = 1.
    if not np.isclose(np.sum(mean_probability), 1.0, atol=1.0e-12, rtol=0.0):
        raise RuntimeError("Frame-averaged conditional probability does not sum to one.")

    # =======================================================================
    # STAGE 7 — PRESERVE THE USER'S X-COORDINATE CONVENTION
    # =======================================================================
    #
    # For B bins, bin_edges has B+1 values.  The requested x coordinate is the
    # LEFT edge of each interval, not the bin centre:
    #
    #                       x_k = bin_edges[k].
    x_left_edges = bin_edges[:-1]

    # Return five coordinated outputs:
    #
    #     x_left_edges:
    #         plotting/output x values;
    #
    #     mean_probability:
    #         primary equal-weight frame-averaged conditional distribution;
    #
    #     standard_deviation:
    #         frame-to-frame population standard deviation for each bin;
    #
    #     pooled_probability:
    #         count-pooled diagnostic distribution;
    #
    #     tuple(frame_results):
    #         immutable detailed results for every analysed frame.
    return (
        x_left_edges,
        mean_probability,
        standard_deviation,
        pooled_probability,
        tuple(frame_results),
    )


# ---------------------------------------------------------------------------
# Output files.
# ---------------------------------------------------------------------------
def save_symmetry_outputs(
    output_prefix: Path,
    symmetry_result: SymmetryResult,
) -> Path:
    """Write all +q/-q quaternion representatives for audit/reuse."""

    path = output_prefix.with_name(output_prefix.name + "_symmetry_quaternions.csv")
    rows: list[list[float | int]] = []

    for physical_index, operation in enumerate(symmetry_result.physical_operations):
        q = operation.quaternion_wxyz
        for sign, signed_q in ((1, q), (-1, -q)):
            rows.append(
                [
                    physical_index,
                    sign,
                    signed_q[0],
                    signed_q[1],
                    signed_q[2],
                    signed_q[3],
                    operation.max_vertex_residual,
                    operation.discovery_angle_deg,
                    operation.discovery_axis[0],
                    operation.discovery_axis[1],
                    operation.discovery_axis[2],
                ]
            )

    np.savetxt(
        path,
        np.asarray(rows, dtype=np.float64),
        delimiter=",",
        header=(
            "physical_operation_index,quaternion_sign,w,x,y,z,"
            "max_vertex_residual,discovery_angle_deg,"
            "discovery_axis_x,discovery_axis_y,discovery_axis_z"
        ),
        comments="",
        fmt=["%d", "%d", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g", "%.17g"],
    )
    return path


def save_histogram_outputs(
    plt_module: Any,
    output_prefix: Path,
    x_left_edges: np.ndarray,
    mean_probability: np.ndarray,
    standard_deviation: np.ndarray,
    pooled_probability: np.ndarray,
    frame_results: Sequence[FrameHistogram],
    bin_edges: np.ndarray,
    maximum_angle_deg: float,
    show_plot: bool,
) -> tuple[Path, Path, Path]:
    """Save summary CSV, per-frame CSV, and PNG plot."""

    summary_csv = output_prefix.with_suffix(".csv")
    per_frame_csv = output_prefix.with_name(output_prefix.name + "_per_frame.csv")
    png_path = output_prefix.with_suffix(".png")

    total_counts = np.sum(
        np.vstack([result.raw_counts for result in frame_results]),
        axis=0,
        dtype=np.int64,
    )

    summary_table = np.column_stack(
        (
            x_left_edges,
            bin_edges[1:],
            mean_probability,
            standard_deviation,
            pooled_probability,
            total_counts,
        )
    )
    np.savetxt(
        summary_csv,
        summary_table,
        delimiter=",",
        header=(
            "left_bin_edge_deg,right_bin_edge_deg,"
            "equal_weight_frame_mean_probability_per_bin,"
            "frame_to_frame_standard_deviation,"
            "pooled_conditional_probability_per_bin,total_raw_count"
        ),
        comments="",
        fmt=["%.12g", "%.12g", "%.17g", "%.17g", "%.17g", "%d"],
    )

    frame_probability_matrix = np.vstack(
        [result.conditional_probability for result in frame_results]
    ).T
    per_frame_table = np.column_stack((x_left_edges, frame_probability_matrix))
    per_frame_header = "left_bin_edge_deg," + ",".join(
        f"frame_{result.frame_index}_probability"
        for result in frame_results
    )
    np.savetxt(
        per_frame_csv,
        per_frame_table,
        delimiter=",",
        header=per_frame_header,
        comments="",
        fmt="%.17g",
    )

    figure, axis = plt_module.subplots(figsize=(4.8, 3.6), dpi=200)
    axis.plot(
        x_left_edges,
        mean_probability,
        linewidth=1.5,
        # label=f"Equal-weight average of {len(frame_results)} frame(s)",
    )

    if len(frame_results) > 1:
        lower = np.maximum(mean_probability - standard_deviation, 0.0)
        upper = mean_probability + standard_deviation
        axis.fill_between(
            x_left_edges,
            lower,
            upper,
            alpha=0.2,
            linewidth=0.0,
            # label="Frame-to-frame standard deviation",
        )

    axis.set_xlabel(r"$\theta_{ij}(^{\circ})$", fontsize=14)
    axis.set_ylabel(r"$P(\theta_{ij})$", fontsize=14)
    axis.set_xlim(0.0, maximum_angle_deg)
    axis.tick_params(axis="both", direction="in")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(png_path, dpi=600, bbox_inches="tight")

    if show_plot:
        plt_module.show()
    else:
        plt_module.close(figure)

    return summary_csv, per_frame_csv, png_path


def save_metadata(
    output_prefix: Path,
    gsd_path: Path,
    shape_path: Path,
    total_trajectory_frames: int,
    selected_frame_indices: Sequence[int],
    selected_particle_count: int,
    num_bins: int,
    maximum_angle_deg: float,
    precision_exponent: int,
    expected_edges: int,
    expected_faces: int,
    block_size: int,
    symmetry_result: SymmetryResult,
    frame_results: Sequence[FrameHistogram],
    mean_probability: np.ndarray,
) -> Path:
    """Write a JSON record sufficient to audit the calculation settings."""

    path = output_prefix.with_name(output_prefix.name + "_metadata.json")
    metadata = {
        "gsd_file": str(gsd_path),
        "shape_file": str(shape_path),
        "total_trajectory_frames": int(total_trajectory_frames),
        "selected_frame_indices": [int(index) for index in selected_frame_indices],
        "selected_particle_count_per_frame": int(selected_particle_count),
        "pair_definition": "unique unordered pairs i < j; self-pairs excluded",
        "total_unique_pairs_per_frame": int(
            selected_particle_count * (selected_particle_count - 1) // 2
        ),
        "histogram_bins": int(num_bins),
        "maximum_plotted_angle_deg": float(maximum_angle_deg),
        "x_coordinate_convention": "left bin edges",
        "normalisation": (
            "each frame normalised by the number of unique pairs inside the "
            "plotted range; equal-weight arithmetic mean over selected frames"
        ),
        "quantity": "probability per bin (not probability density)",
        "symmetry_precision_exponent": int(precision_exponent),
        "symmetry_matching_tolerance": float(symmetry_result.matching_tolerance),
        "expected_edges": int(expected_edges),
        "expected_faces": int(expected_faces),
        "recovered_edges": int(len(symmetry_result.decomposition.edges)),
        "recovered_faces": int(len(symmetry_result.decomposition.faces)),
        "convex_hull_merge_tolerance": float(
            symmetry_result.decomposition.merge_tolerance
        ),
        "convex_hull_volume": float(symmetry_result.decomposition.volume),
        "original_vertex_center": symmetry_result.original_center.tolist(),
        "physical_proper_rotation_count": int(
            len(symmetry_result.physical_operations)
        ),
        "equivalent_quaternion_representative_count": int(
            len(symmetry_result.equivalent_quaternions_wxyz)
        ),
        "freud_block_size": int(block_size),
        "mean_probability_sum": float(np.sum(mean_probability)),
        "frames": [
            {
                "frame_index": int(result.frame_index),
                "total_unique_pairs": int(result.total_unique_pairs),
                "pairs_inside_range": int(result.pairs_inside_range),
                "pairs_outside_range": int(result.pairs_outside_range),
                "outside_range_fraction": float(
                    result.pairs_outside_range / result.total_unique_pairs
                ),
                "minimum_angle_deg": float(result.minimum_angle_deg),
                "maximum_angle_deg": float(result.maximum_angle_deg),
                "maximum_input_quaternion_norm_deviation": float(
                    result.maximum_quaternion_norm_deviation
                ),
                "conditional_probability_sum": float(
                    np.sum(result.conditional_probability)
                ),
            }
            for result in frame_results
        ],
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    return path


# ---------------------------------------------------------------------------
# Command-line interface.
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate a rigorous frame-averaged global pairwise orientation "
            "histogram from a HOOMD GSD trajectory and particle shape JSON."
        )
    )
    parser.add_argument("trajectory", help="Input HOOMD GSD trajectory")
    parser.add_argument("shape", help="Particle shape JSON containing vertices")

    # All scientific choices remain interactive by default, while optional CLI
    # flags make an already-validated run exactly reproducible in batch jobs.
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--edges", type=int, default=None)
    parser.add_argument("--faces", type=int, default=None)
    parser.add_argument(
        "--precision",
        type=int,
        default=None,
        help=(
            "p in absolute symmetry vertex-matching tolerance 10^(-p); "
            "historical suggested value is 2"
        ),
    )
    parser.add_argument("--bins", type=int, default=None)
    parser.add_argument("--max-angle", type=float, default=None)
    parser.add_argument(
        "--block-size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Number of query orientations passed to freud per memory block",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; default is beside the trajectory",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening an interactive window",
    )
    return parser


def main() -> int:
    """Run the complete global pairwise-orientation analysis workflow.

    The workflow executed here is:

        command-line arguments
            -> dependency loading
            -> GSD trajectory inspection
            -> selection of the final N frames
            -> validation/selection of particle count
            -> collection of polyhedron and histogram settings
            -> shape-vertex reading
            -> proper rotational-symmetry detection
            -> frame-by-frame pairwise-angle histograms
            -> equal-weight averaging over frames
            -> output files and final validation

    Return value
    ------------
    int
        0 means that the calculation completed successfully.
        1 means that a controlled error was detected and reported.

    Scientific conventions fixed by this function
    ----------------------------------------------
    1. The selected frames are the final consecutive N frames.
    2. The first M particles are used in every selected frame.
    3. Only unique unordered pairs i < j are counted.
    4. Self-pairs i == j are excluded.
    5. Each frame is normalised independently inside the displayed angular
       interval, and the final curve is an equal-weight arithmetic mean of
       those per-frame probability vectors.
    6. Histogram x coordinates are the left bin edges
    7. Histogram y values are probability per bin, not probability density.
    """

    # ---------------------------------------------------------------------
    # STEP 1: Parse the command-line interface.
    # ---------------------------------------------------------------------
    # build_argument_parser() defines two mandatory positional arguments:
    #
    #     trajectory : path to the HOOMD GSD trajectory
    #     shape      : path to the shape JSON containing particle vertices
    #
    # It also defines optional flags such as --frames, --particles, --edges,
    # --faces, --precision, --bins, --max-angle, --block-size, --output-dir,
    # and --no-show.

    args = build_argument_parser().parse_args()

    # ---------------------------------------------------------------------
    # STEP 2: Put the full workflow inside one controlled error boundary.
    # ---------------------------------------------------------------------
    try:
        # -----------------------------------------------------------------
        # STEP 3: Import packages that are required only at runtime.
        # -----------------------------------------------------------------
        freud, gsd_hoomd, plt = import_runtime_packages()

        # -----------------------------------------------------------------
        # STEP 4: Validate the memory-block size before reading large data.
        # -----------------------------------------------------------------
        # The block size is the number of query orientations passed to freud
        # at one time.  It controls peak memory use but does not alter which
        # particle pairs are counted or the mathematical histogram result.
        # A zero or negative block size would make the block loop invalid.
        if args.block_size < 1:
            raise ValueError("--block-size must be a positive integer.")

        # -----------------------------------------------------------------
        # STEP 5: Resolve and open the two input files.
        # -----------------------------------------------------------------
        gsd_path, trajectory = open_gsd_trajectory(gsd_hoomd, args.trajectory)
        
        # Resolve the shape path now.  The actual existence and JSON-content
        # checks are performed later by read_shape_vertices().
        shape_path = Path(args.shape).expanduser().resolve()

        # -----------------------------------------------------------------
        # STEP 6: Determine how many frames are present in the trajectory.
        # -----------------------------------------------------------------
        total_frames = len(trajectory)
        
        # A trajectory with no snapshots cannot support any analysis.
        if total_frames < 1:
            raise ValueError("The GSD trajectory contains no frames.")

        print("\nTrajectory summary")
        print("------------------")
        print(f"Trajectory: {gsd_path}")
        print(f"Number of available frames: {total_frames}")
        print(f"Valid zero-based frame indices: 0 through {total_frames - 1}")

        # -----------------------------------------------------------------
        # STEP 7: Ask how many consecutive final frames should be analysed.
        # -----------------------------------------------------------------
        # resolve_or_prompt() follows this rule:
        #   * use args.frames when --frames was supplied;
        #   * otherwise ask interactively;
        #   * pressing Enter accepts CURRENT_SUGGESTED_FRAMES;
        #   * reject values outside [1, total_frames].
        #
        frames_to_average = resolve_or_prompt(
            args.frames,
            (
                "How many consecutive final frames should be included in the "
                "equal-weight average?"
            ),
            CURRENT_SUGGESTED_FRAMES,
            int,
            lambda value: 1 <= value <= total_frames,
            f"Enter an integer from 1 through {total_frames}.",
        )


        first_selected_frame = total_frames - frames_to_average
        selected_frame_indices = list(range(first_selected_frame, total_frames))
        print(
            f"Selected final frame indices: {selected_frame_indices[0]} through "
            f"{selected_frame_indices[-1]}"
        )

        # -----------------------------------------------------------------
        # STEP 8: Inspect particle-orientation counts in every selected frame.
        # -----------------------------------------------------------------
        selected_frame_particle_counts: list[int] = []

        for frame_index in selected_frame_indices:
            # Read only the orientation field needed for this validation.
            orientation_array = np.asarray(
                trajectory[frame_index].particles.orientation
            )
            # A valid orientation array must have one row per particle and four
            # quaternion components per row.
            if orientation_array.ndim != 2 or orientation_array.shape[1] != 4:
                raise ValueError(
                    f"Frame {frame_index} has invalid orientation array shape "
                    f"{orientation_array.shape}; expected (N, 4)."
                )
            # Store N for this selected frame.
            selected_frame_particle_counts.append(len(orientation_array))

        minimum_available_particles = min(selected_frame_particle_counts)
        maximum_available_particles = max(selected_frame_particle_counts)

        print("\nParticle-count summary for selected frames")
        print("------------------------------------------")
        if minimum_available_particles == maximum_available_particles:
            # This is the common fixed-N simulation case.
            print(
                f"Every selected frame contains {minimum_available_particles} "
                "particle orientations."
            )
        else:
            # Variable-N trajectories are supported as long as the user selects
            # no more than the minimum count across the chosen frames.
            print(
                "Selected frames have varying particle counts: "
                f"minimum={minimum_available_particles}, "
                f"maximum={maximum_available_particles}."
            )
            print(
                "The chosen count must not exceed the minimum, because the same "
                "number of particles is used in every frame."
            )

        # -----------------------------------------------------------------
        # STEP 9: Ask how many particles should be included per frame.
        # -----------------------------------------------------------------
        selected_particle_count = resolve_or_prompt(
            args.particles,
            (
                "How many particles should be used from the beginning of each "
                "selected frame?"
            ),
            minimum_available_particles,
            int,
            lambda value: 2 <= value <= minimum_available_particles,
            (
                "Enter an integer from 2 through "
                f"{minimum_available_particles}."
            ),
        )

        # The script intentionally preserves a deterministic selection rule:
        # use particle indices 0, 1, ..., M-1 in every frame.
        print(
            f"Particles selected in every frame: indices 0 through "
            f"{selected_particle_count - 1}"
        )

        # -----------------------------------------------------------------
        # STEP 10: Ask for the known polyhedron topology.
        # -----------------------------------------------------------------
        expected_edges = resolve_or_prompt(
            args.edges,
            "Enter the known number of polyhedron edges",
            CURRENT_SUGGESTED_NUM_EDGES,
            int,
            lambda value: value >= 3,
            "Enter an integer of at least 3.",
        )
        expected_faces = resolve_or_prompt(
            args.faces,
            "Enter the known number of polyhedron faces",
            CURRENT_SUGGESTED_NUM_FACES,
            int,
            lambda value: value >= 4,
            "Enter an integer of at least 4.",
        )

        # -----------------------------------------------------------------
        # STEP 11: Ask for the symmetry vertex-matching tolerance exponent.
        # -----------------------------------------------------------------
        # The user supplies p, and the absolute geometric tolerance is:
        #
        #     epsilon_match = 10**(-p)
        #
        # This tolerance is used to decide whether a candidate rotation maps
        # the centred shape vertices one-to-one onto the original vertex set.
        precision_exponent = resolve_or_prompt(
            args.precision,
            (
                "Enter invariant-quaternion symmetry precision p; the vertex "
                "matching tolerance is 10^(-p). The present historical value "
                "was p=2, i.e. tolerance 0.01"
            ),
            CURRENT_SUGGESTED_PRECISION,
            int,
            lambda value: 0 <= value <= 15,
            "Enter an integer p from 0 through 15.",
        )
        if precision_exponent <= 2:
            print(
                "SCIENTIFIC CAUTION: p <= 2 gives a matching tolerance of at "
                "least 0.01 in shape-length units. This reproduces the present "
                "suggested setting, but a sensitivity comparison with p=3 and "
                "p=4 is advisable for final production results."
            )

        # -----------------------------------------------------------------
        # STEP 12: Ask for histogram resolution and angular interval.
        # -----------------------------------------------------------------
        num_bins = resolve_or_prompt(
            args.bins,
            (
                "Enter the number of histogram bins (suggestion 50 from the "
                "provided parameter file; the old active function hard-coded 30)"
            ),
            CURRENT_SUGGESTED_NUM_BINS,
            int,
            lambda value: value >= 1,
            "Enter a positive integer.",
        )

        # maximum_angle_deg defines both the displayed interval and the set of
        # pairs used in the *conditional* probability normalisation.
        # Only angles in [0, maximum_angle_deg] contribute to the denominator.
        maximum_angle_deg = resolve_or_prompt(
            args.max_angle,
            (
                "Enter the maximum plotted misorientation angle in degrees; "
                "conditional normalisation uses only pairs inside this range"
            ),
            CURRENT_SUGGESTED_MAX_ANGLE_DEG,
            float,
            lambda value: 0.0 < value <= 180.0,
            "Enter a value greater than 0 and no greater than 180 degrees.",
        )

        print("\nHistogram definition")
        print("--------------------")
        print(f"Bins: {num_bins}")
        print(f"Plotted range: 0 to {maximum_angle_deg:g} degrees")
        print("Pair set: unique unordered pairs i < j; self-pairs excluded")
        print("x coordinate: left bin edge")
        print("y coordinate: conditional probability per bin, not density")
        print(
            "Frame aggregation: normalise every selected frame inside the "
            "plotted range, then take an equal-weight arithmetic mean"
        )
        print(f"freud query block size: {args.block_size}")

        # -----------------------------------------------------------------
        # STEP 13: Read the particle shape vertices from the JSON file.
        # -----------------------------------------------------------------
        vertices = read_shape_vertices(shape_path)

        # -----------------------------------------------------------------
        # STEP 14: Detect and validate the proper rotational symmetry group.
        # -----------------------------------------------------------------
        symmetry_result = detect_proper_rotational_symmetries(
            vertices=vertices,
            expected_edges=expected_edges,
            expected_faces=expected_faces,
            precision_exponent=precision_exponent,
        )

        # -----------------------------------------------------------------
        # STEP 15: Calculate and average the selected-frame histograms.
        # -----------------------------------------------------------------
        (
            x_left_edges,
            mean_probability,
            standard_deviation,
            pooled_probability,
            frame_results,
        ) = compute_frame_averaged_histogram(
            freud_module=freud,
            trajectory=trajectory,
            frame_indices=selected_frame_indices,
            particle_count=selected_particle_count,
            equivalent_quaternions=(
                symmetry_result.equivalent_quaternions_wxyz
            ),
            num_bins=num_bins,
            maximum_angle_deg=maximum_angle_deg,
            block_size=args.block_size,
        )

        # -----------------------------------------------------------------
        # STEP 16: Reconstruct the complete bin-edge array for output routines.
        # -----------------------------------------------------------------
        bin_edges = np.linspace(
            0.0,
            maximum_angle_deg,
            num_bins + 1,
            dtype=np.float64,
        )

        # -----------------------------------------------------------------
        # STEP 17: Choose and create the output directory.
        # -----------------------------------------------------------------
        # Default: save beside the input trajectory.
        # Override: --output-dir /some/path
        if args.output_dir is None:
            output_directory = gsd_path.parent
        else:
            output_directory = Path(args.output_dir).expanduser().resolve()

        # parents=True creates missing parent directories; exist_ok=True allows
        # reuse of an existing output directory
        output_directory.mkdir(parents=True, exist_ok=True)

        # Build a descriptive prefix recording major analysis choices.  Each
        # output writer appends its own suffix and extension to this prefix.
        output_prefix = output_directory / (
            f"{gsd_path.stem}_global_pairwise_angles_"
            f"last_{frames_to_average}_frames_"
            f"particles_{selected_particle_count}_bins_{num_bins}"
        )

        # -----------------------------------------------------------------
        # STEP 18: Save the detected symmetry quaternions.
        # -----------------------------------------------------------------
        symmetry_csv = save_symmetry_outputs(output_prefix, symmetry_result)

        # -----------------------------------------------------------------
        # STEP 19: Save histogram tables and the figure.
        # -----------------------------------------------------------------
        # save_histogram_outputs() writes:
        #   * one summary CSV containing mean, standard deviation, and pooled
        #     probabilities;
        #   * one per-frame CSV;
        #   * one PNG plot.
        #
        # show_plot is False only when --no-show was supplied.  The PNG is saved
        # in either case.
        summary_csv, per_frame_csv, png_path = save_histogram_outputs(
            plt_module=plt,
            output_prefix=output_prefix,
            x_left_edges=x_left_edges,
            mean_probability=mean_probability,
            standard_deviation=standard_deviation,
            pooled_probability=pooled_probability,
            frame_results=frame_results,
            bin_edges=bin_edges,
            maximum_angle_deg=maximum_angle_deg,
            show_plot=not args.no_show,
        )

        # -----------------------------------------------------------------
        # STEP 20: Save complete machine-readable run metadata.
        # -----------------------------------------------------------------
        # The JSON records input paths, selected frames, selected particles,
        # topology, tolerance, histogram settings, block size, symmetry-group
        # diagnostics, per-frame pair accounting, and probability validation.
        # This makes the run reproducible and easier to audit later.
        metadata_json = save_metadata(
            output_prefix=output_prefix,
            gsd_path=gsd_path,
            shape_path=shape_path,
            total_trajectory_frames=total_frames,
            selected_frame_indices=selected_frame_indices,
            selected_particle_count=selected_particle_count,
            num_bins=num_bins,
            maximum_angle_deg=maximum_angle_deg,
            precision_exponent=precision_exponent,
            expected_edges=expected_edges,
            expected_faces=expected_faces,
            block_size=args.block_size,
            symmetry_result=symmetry_result,
            frame_results=frame_results,
            mean_probability=mean_probability,
        )

        print("\nFinal validation and outputs")
        print("----------------------------")
        print(f"Sum of final mean bin probabilities: {np.sum(mean_probability):.16g}")
        print(f"Summary histogram CSV: {summary_csv}")
        print(f"Per-frame probability CSV: {per_frame_csv}")
        print(f"Symmetry quaternion CSV: {symmetry_csv}")
        print(f"Run metadata JSON: {metadata_json}")
        print(f"Plot PNG: {png_path}")

    # ---------------------------------------------------------------------
    # STEP 22: Convert expected failures into a clean program exit.
    # ---------------------------------------------------------------------
    except (
        FileNotFoundError,
        ImportError,
        ValueError,
        RuntimeError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
