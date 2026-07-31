#!/usr/bin/env python3
"""
Rigorous standalone orientational-state clustering and fixed-reference tracking.

The program:
  1. reads a HOOMD GSD trajectory and particle-shape JSON;
  2. reconstructs and validates the proper rotational symmetry group of the body;
  3. computes symmetry-reduced pairwise angular distances with freud;
  4. clusters every selected frame with complete-linkage clustering;
  5. explicitly verifies that every returned frame-level cluster has maximum
     pairwise angular diameter <= the user-supplied orientation tolerance;
  6. selects one frame as the fixed reference-state frame;
  7. represents each frame-level cluster by a symmetry-aware medoid, which is
     an actual particle orientation minimising the sum of within-cluster angular
     distances;
  8. matches clusters in every other frame to fixed reference states using a
     separate user-supplied medoid-tracking tolerance and a global one-to-one
     assignment;
  9. detects ambiguous matches and split/merge candidates;
 10. averages each fixed state's particle population over all selected frames,
     assigning zero population when that reference state is absent;
 11. uses cluster_size_cutoff for instantaneous major-cluster stability and
     tracking eligibility, and also for final mean-population validity.

No spatial positions or neighbour definitions are used. This is a global
orientation-only analysis.

Dependencies:
    pip install numpy scipy matplotlib gsd freud-analysis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import ConvexHull, cKDTree, distance_matrix
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise SystemExit(
        "SciPy is required. Install dependencies with:\n"
        "    pip install numpy scipy matplotlib gsd freud-analysis"
    ) from exc


CURRENT_SUGGESTED_PRECISION = 2
CURRENT_SUGGESTED_NUM_EDGES = 25
CURRENT_SUGGESTED_NUM_FACES = 15
CURRENT_SUGGESTED_FRAMES = 1
CURRENT_SUGGESTED_ORIENTATION_TOL_DEG = 41.0
CURRENT_SUGGESTED_CLUSTER_SIZE_CUTOFF = 200
CURRENT_SUGGESTED_STABILITY_TRIALS = 1

'''DEFAULT_BLOCK_SIZE = 128
DEFAULT_ANGLE_VALIDATION_TOL_DEG = 1.0e-5
DEFAULT_RANDOM_SEED = 1729'''


DEFAULT_BLOCK_SIZE = 128

# Baseline tolerance used when validating numerical properties of the
# symmetry-reduced angular-distance calculation. This is not the physical
# within-cluster angular cutoff.
DEFAULT_ANGLE_VALIDATION_TOL_DEG = 1.0e-3

# A self-comparison should theoretically give exactly zero. In practice,
# freud may show a small single-precision numerical floor. Values below this
# hard limit are treated as numerical resolution rather than physical angles.
#
# This limit does NOT modify any calculated pair angle and does NOT loosen the
# user-supplied orientation_angle_tol.
NUMERICAL_SELF_ANGLE_HARD_LIMIT_DEG = 0.5

# The observed self-angle numerical floor is multiplied by this factor when
# validating symmetry and triangle-inequality residuals.
NUMERICAL_FLOOR_SAFETY_FACTOR = 4.0

# Strictly numerical comparison tolerance for validating the user-requested
# cluster diameter. This is intentionally tiny and is independent of freud's
# self-angle numerical floor.
CLUSTER_DIAMETER_EPS_DEG = 1.0e-10

# Used only to identify numerically tied medoid candidates.
MEDOID_TIE_TOL_DEG = 1.0e-10

DEFAULT_RANDOM_SEED = 1729


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


@dataclass(frozen=True)
class DistanceDiagnostics:
    minimum_angle_deg: float
    maximum_angle_deg: float


@dataclass(frozen=True)
class FrameCluster:
    frame_index: int
    local_cluster_id: int
    members: np.ndarray
    size: int
    medoid_particle_index: int
    medoid_quaternion_wxyz: np.ndarray
    diameter_deg: float
    maximum_angle_to_medoid_deg: float
    medoid_sum_distance_deg: float


@dataclass(frozen=True)
class FrameClustering:
    frame_index: int
    particle_count: int
    clusters: tuple[FrameCluster, ...]
    particle_to_local_cluster: np.ndarray
    maximum_quaternion_norm_deviation: float
    distance_diagnostics: DistanceDiagnostics
    cluster_count_stability_passed: bool | None


@dataclass(frozen=True)
class FrameTracking:
    frame_index: int
    cluster_to_reference_state: tuple[int, ...]
    cluster_tracking_distance_deg: tuple[float, ...]
    unmatched_cluster_ids: tuple[int, ...]
    absent_reference_state_ids: tuple[int, ...]
    split_candidate_reference_ids: tuple[int, ...]
    ambiguous_current_cluster_ids: tuple[int, ...]


@dataclass(frozen=True)
class StateSummary:
    reference_state_id: int
    reference_local_cluster_id: int
    reference_size: int
    reference_medoid_particle_index: int
    reference_medoid_quaternion_wxyz: np.ndarray
    mean_particle_count: float
    standard_deviation_particle_count: float
    mean_population_fraction: float
    standard_deviation_population_fraction: float
    presence_fraction: float
    maximum_tracking_distance_deg: float
    mean_tracking_distance_deg_when_present: float
    maximum_cluster_diameter_deg: float
    valid_by_mean_size_cutoff: bool



def canonicalize_quaternion_sign(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64).copy()
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-14:
        raise ValueError("Cannot canonicalise a zero quaternion.")
    q /= norm
    for component in q:
        if abs(component) > 1.0e-14:
            if component < 0.0:
                q = -q
            break
    return q


def condensed_index(
    particle_count: int,
    first: np.ndarray | int,
    second: np.ndarray | int,
) -> np.ndarray:
    i = np.asarray(first, dtype=np.int64)
    j = np.asarray(second, dtype=np.int64)
    low = np.minimum(i, j)
    high = np.maximum(i, j)
    if np.any(low == high):
        raise ValueError("Condensed-distance indexing is undefined for self-pairs.")
    return (
        particle_count * low
        - low * (low + 1) // 2
        + high
        - low
        - 1
    )


def angular_distance_matrix_deg(
    freud_module: Any,
    global_orientations: np.ndarray,
    query_orientations: np.ndarray,
    equivalent_quaternions: np.ndarray,
) -> np.ndarray:

    # Convert both orientation arrays to contiguous float64 storage before
    # passing them to freud.  This makes the expected memory layout explicit.
    global_array = np.ascontiguousarray(global_orientations, dtype=np.float64)
    query_array = np.ascontiguousarray(query_orientations, dtype=np.float64)

    # Create one AngularSeparationGlobal calculator for this matrix operation.
    calculator = freud_module.environment.AngularSeparationGlobal()

    # freud minimises the orientation difference over the supplied equivalent
    # body rotations.  The first argument defines columns/global orientations;
    # the second defines rows/query orientations.
    calculator.compute(global_array, query_array, equivalent_quaternions)
    result = np.rad2deg(np.asarray(calculator.angles, dtype=np.float64))

    expected_shape = (len(query_array), len(global_array))
    if result.shape != expected_shape:
        raise RuntimeError(
            "Unexpected AngularSeparationGlobal output shape: "
            f"expected {expected_shape}, received {result.shape}."
        )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("AngularSeparationGlobal returned NaN/inf values.")
    return result



def compute_condensed_distance_memmap(
    freud_module: Any,
    orientations: np.ndarray,
    equivalent_quaternions: np.ndarray,
    frame_index: int,
    block_size: int,
    distance_file: Path,
    angle_validation_tolerance_deg: float,
) -> tuple[np.memmap, DistanceDiagnostics]:

    # =======================================================================
    # PURPOSE OF THIS FUNCTION
    # =======================================================================
    #
    # Complete-linkage clustering requires all pairwise distances between the
    # selected particle orientations in one frame.
    #
    # A full square N x N float64 matrix would contain N^2 entries and would
    # duplicate every unordered pair:
    #
    #                       d(i,j) = d(j,i).
    #
    # The diagonal d(i,i) is also unnecessary.  Therefore this function stores
    # only the strict upper triangle i<j in SciPy's condensed-distance order:
    #
    #                       N(N-1)/2 values.
    #
    # The result is written to a NumPy memory-mapped file so that the raw
    # distance storage resides on disk and is accessed as an array without
    # requiring one permanent in-memory N x N matrix.
    #
    # freud calculations are additionally performed in query blocks.  At any
    # one time, the largest temporary angle matrix has approximate shape:
    #
    #                       (block_size, particle_count).

    # Number of selected orientation quaternions in this frame.

    particle_count = len(orientations)


    # Exact number of unique unordered non-self pairs:
    #
    #                       C(N,2) = N(N-1)/2.
    #
    # Integer floor division is exact because one of N and N-1 is even.
    pair_count = particle_count * (particle_count - 1) // 2

    # =======================================================================
    # CREATE THE ON-DISK CONDENSED DISTANCE ARRAY
    # =======================================================================
    #
    # mode="w+" means:
    #
    #   * create a new file or overwrite an existing file;
    #   * allow both reading and writing.
    #
    # dtype=float64 keeps the stored values in double precision after the
    # freud output has been converted to a NumPy float64 array.
    #
    # shape=(pair_count,) creates a one-dimensional condensed representation
    # compatible with scipy.cluster.hierarchy.linkage.
    distances = np.memmap(
        distance_file,
        dtype=np.float64,
        mode="w+",
        shape=(pair_count,),
    )

    # Reuse one AngularSeparationGlobal object for all blocks in this frame
    # instead of constructing a new object for every block.
    calculator = freud_module.environment.AngularSeparationGlobal()

    # cursor is the next unwritten position in the condensed memmap.
    #
    # The row-wise upper-triangle extraction below naturally generates SciPy's
    # condensed ordering:
    #
    #   (0,1), (0,2), ..., (0,N-1),
    #   (1,2), (1,3), ..., (1,N-1),
    #   ...
    #   (N-2,N-1).
    cursor = 0

    # Sentinels for true minimum and maximum over all stored i<j distances.
    # They will be replaced during the first nonempty row.
    minimum_angle = math.inf
    maximum_angle = -math.inf


    # =======================================================================
    # LOOP OVER QUERY BLOCKS
    # =======================================================================
    #
    # range(0, particle_count, block_size) generates block starts:
    #
    #                       0, block_size, 2*block_size, ...
    #
    # The last block may contain fewer than block_size particles.
    for block_start in range(0, particle_count, block_size):
        # Do not let the final block extend beyond the orientation array.
        block_stop = min(block_start + block_size, particle_count)
        # Query orientations correspond to global particle indices:
        query = orientations[block_start:block_stop]
        # Compute the symmetry-reduced angular separation between:
        #
        #   global orientations: all selected particles;
        #   query orientations:  the current query block;
        #   equivalent quaternions: the body's q/-q symmetry representatives.
        calculator.compute(orientations, query, equivalent_quaternions)
        # The expected matrix orientation is:
        #
        #      rows    -> query particles in the current block;
        #      columns -> all global selected particles.
        block = np.rad2deg(np.asarray(calculator.angles, dtype=np.float64))
        # Number of query rows in the current block and number of global orientation columns.
        expected_shape = (block_stop - block_start, particle_count)
        if block.shape != expected_shape:
            raise RuntimeError(
                f"Frame {frame_index}: expected freud angle shape "
                f"{expected_shape}, received {block.shape}."
            )
        if not np.all(np.isfinite(block)):
            raise RuntimeError(
                f"Frame {frame_index}: freud returned NaN/inf angular values."
            )

        # ===================================================================
        # EXTRACT THE STRICT UPPER TRIANGLE i<j ROW BY ROW
        # ===================================================================
        #
        # local_row is the row index inside the current block.
        # particle_i is the corresponding global particle index.
        for local_row, particle_i in enumerate(range(block_start, block_stop)):
            # The full row contains distances from global particle particle_i
            # to all global particles:
            #
            #        d(i,0), d(i,1), ..., d(i,i), ..., d(i,N-1).
            #
            # Retain only columns j>i:
            #
            #        d(i,i+1), d(i,i+2), ..., d(i,N-1).
            #
            # This excludes:
            #
            #   * j<i: pairs already stored earlier as d(j,i);
            #   * j=i: the self-pair.
            values = block[local_row, particle_i + 1 :]
            # For the final particle i=N-1, no j>i partner exists, so the slice
            # is empty.  Skip all subsequent min/max and write operations.
            if values.size == 0:
                continue

            # Angular separation should be nonnegative.  Permit only the tiny
            # negative numerical tolerance supplied for validation.
            if float(np.min(values)) < -angle_validation_tolerance_deg:
                raise RuntimeError(
                    f"Frame {frame_index}: negative angular distance detected."
                )
            # A quaternion rotational separation should not exceed 180 degrees.
            # Again, allow only the supplied tiny numerical overshoot.
            if float(np.max(values)) > 180.0 + angle_validation_tolerance_deg:
                raise RuntimeError(
                    f"Frame {frame_index}: angular distance exceeds 180 degrees."
                )

            # Clip only negligible roundoff outside the theoretical interval:
            #
            #                        [0, 180] degrees.
            #
            # Values well outside this interval have already triggered errors.
            values = np.clip(values, 0.0, 180.0)
            # Compute the memmap position immediately after this row's values.
            next_cursor = cursor + values.size
            # Write the complete row segment contiguously into the condensed memmap.
            distances[cursor:next_cursor] = values
            # Advance the write cursor to the first unwritten element
            cursor = next_cursor

            # Update the true global minimum and maximum among all unique
            # off-diagonal pairs processed so far.
            minimum_angle = min(minimum_angle, float(np.min(values)))
            maximum_angle = max(maximum_angle, float(np.max(values)))

    # =======================================================================
    # VERIFY THAT EVERY UNIQUE PAIR WAS WRITTEN EXACTLY ONCE
    # =======================================================================
    if cursor != pair_count:
        raise RuntimeError(
            f"Frame {frame_index}: condensed-distance accounting failed; "
            f"expected {pair_count}, wrote {cursor}."
        )

    # Explicitly synchronise modified memmap pages with the underlying file
    # before returning it to linkage and later readers.
    distances.flush()

    # Return:
    #
    #   1. the readable/writable one-dimensional memmap;
    #   2. a structured diagnostic record for terminal reporting and metadata.
    return distances, DistanceDiagnostics(
        minimum_angle_deg=float(minimum_angle),
        maximum_angle_deg=float(maximum_angle),
    )


def cluster_distance_statistics(
    members: np.ndarray,
    particle_count: int,
    condensed_distances: np.ndarray,
    medoid_tie_tolerance_deg: float,
) -> tuple[int, float, float, float]:
    members = np.asarray(members, dtype=np.int64)
    cluster_size = len(members)

    if cluster_size == 1:
        only = int(members[0])
        return only, 0.0, 0.0, 0.0

    distance_sums = np.zeros(cluster_size, dtype=np.float64)
    diameter = 0.0

    for local_i in range(cluster_size - 1):
        global_i = int(members[local_i])
        global_js = members[local_i + 1 :]
        indices = condensed_index(
            particle_count,
            global_i,
            global_js,
        )
        values = np.asarray(condensed_distances[indices], dtype=np.float64)
        distance_sums[local_i] += float(np.sum(values))
        distance_sums[local_i + 1 :] += values
        diameter = max(diameter, float(np.max(values)))

    minimum_sum = float(np.min(distance_sums))
    tied_local_indices = np.where(
        np.isclose(
            distance_sums,
            minimum_sum,
            atol=medoid_tie_tolerance_deg,
            rtol=0.0,
        )
    )[0]

    # A medoid is an actual cluster member. When several members are tied,
    # choose the smallest original particle index for deterministic output.
    tied_particle_indices = members[tied_local_indices]
    chosen_local = int(
        tied_local_indices[np.argmin(tied_particle_indices)]
    )
    medoid_particle = int(members[chosen_local])

    other_members = members[members != medoid_particle]
    if other_members.size:
        medoid_distances = np.asarray(
            condensed_distances[
                condensed_index(
                    particle_count,
                    medoid_particle,
                    other_members,
                )
            ],
            dtype=np.float64,
        )
        maximum_to_medoid = float(np.max(medoid_distances))
    else:
        maximum_to_medoid = 0.0

    return (
        medoid_particle,
        float(diameter),
        maximum_to_medoid,
        float(distance_sums[chosen_local]),
    )




def build_permuted_condensed_distances(
    original_distances: np.ndarray,
    particle_count: int,
    permutation_new_to_old: np.ndarray,
    output_file: Path,
) -> np.memmap:
    pair_count = particle_count * (particle_count - 1) // 2
    permuted = np.memmap(
        output_file,
        dtype=np.float64,
        mode="w+",
        shape=(pair_count,),
    )

    cursor = 0
    for new_i in range(particle_count - 1):
        old_i = int(permutation_new_to_old[new_i])
        old_js = permutation_new_to_old[new_i + 1 :]
        indices = condensed_index(particle_count, old_i, old_js)
        values = np.asarray(original_distances[indices], dtype=np.float64)
        next_cursor = cursor + len(values)
        permuted[cursor:next_cursor] = values
        cursor = next_cursor

    if cursor != pair_count:
        raise RuntimeError("Permuted condensed-distance construction failed.")
    permuted.flush()
    return permuted


def complete_linkage_labels(
    condensed_distances: np.ndarray,
    orientation_angle_tolerance_deg: float,
) -> np.ndarray:
    # =======================================================================
    # PURPOSE OF COMPLETE-LINKAGE CLUSTERING
    # =======================================================================
    #
    # The input is the condensed vector of all unique pairwise
    # symmetry-reduced angular distances for one frame.
    #
    # Complete linkage defines the distance between two tentative clusters A
    # and B as the largest cross-cluster pair distance:
    #
    #       D_complete(A,B) = max_{i in A, j in B} d_G(i,j).
    #
    # Cutting the resulting hierarchy at orientation_angle_tolerance_deg is
    # intended to produce groups whose pairwise diameters do not exceed that
    # threshold.  The program does not rely solely on this property: the exact
    # diameter of every returned cluster is recalculated and checked later in
    # construct_frame_clusters().

    # scipy.cluster.hierarchy.linkage consumes the condensed distance vector and
    # returns an (N-1) x 4 linkage matrix describing the merge tree.
    #
    # method="complete" selects farthest-pair linkage.
    #
    # optimal_ordering=False avoids the additional leaf-reordering computation.
    # Leaf order is not used to define scientific cluster identity.
    hierarchy = linkage(
        condensed_distances,
        method="complete",
        optimal_ordering=False,
    )
    # Convert the hierarchy into flat clusters by cutting at the user-supplied
    # physical angular threshold.
    #
    # criterion="distance" means no cluster is formed through a linkage merge
    # above t.
    #
    # SciPy labels clusters using positive integers beginning at 1.
    labels = fcluster(
        hierarchy,
        t=orientation_angle_tolerance_deg,
        criterion="distance",
    ).astype(np.int64)
    # Convert the labels to zero-based integers for internal array indexing:
    #
    #                         1,2,... -> 0,1,...
    #
    # The numerical label values are arbitrary; particle membership is the
    # physical information.
    return labels - 1


def _alphabetic_group_name(index: int) -> str:
    """Return 0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, and so on."""

    if index < 0:
        raise ValueError("Group index must be nonnegative.")

    value = index + 1
    name = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def _cluster_medoid_records_from_labels(
    labels: np.ndarray,
    original_particle_ids: np.ndarray,
    particle_count: int,
    condensed_distances: np.ndarray,
    orientations: np.ndarray,
) -> list[dict[str, Any]]:
    """Build deterministic cluster/medoid records in original particle IDs."""

    labels = np.asarray(labels, dtype=np.int64)
    original_particle_ids = np.asarray(original_particle_ids, dtype=np.int64)

    if len(labels) != len(original_particle_ids):
        raise ValueError(
            "Label array and original-particle-ID mapping have different lengths."
        )

    records: list[dict[str, Any]] = []

    for label in np.unique(labels):
        members = np.sort(original_particle_ids[labels == label])
        (
            medoid_particle,
            diameter,
            maximum_to_medoid,
            medoid_sum,
        ) = cluster_distance_statistics(
            members,
            particle_count,
            condensed_distances,
            medoid_tie_tolerance_deg=MEDOID_TIE_TOL_DEG,
        )

        records.append(
            {
                "members": members,
                "size": int(len(members)),
                "medoid_particle": int(medoid_particle),
                "medoid_quaternion": canonicalize_quaternion_sign(
                    orientations[medoid_particle]
                ),
                "diameter_deg": float(diameter),
                "maximum_to_medoid_deg": float(maximum_to_medoid),
                "medoid_sum_distance_deg": float(medoid_sum),
            }
        )

    # Use the same deterministic ordering as construct_frame_clusters(), so
    # original group A/B/C corresponds to final local cluster IDs 0/1/2.
    records.sort(
        key=lambda record: (
            -record["size"],
            record["medoid_particle"],
            tuple(int(value) for value in record["members"]),
        )
    )

    for group_id, record in enumerate(records):
        record["group_id"] = int(group_id)
        record["group_name"] = _alphabetic_group_name(group_id)

    return records


def _append_particle_order_medoid_diagnostics(
    diagnostics_file: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    """Append one trial's cutoff-qualified medoid diagnostics to CSV."""

    fieldnames = [
        "frame_index",
        "trial_index",
        "cluster_size_cutoff",
        "original_total_cluster_count",
        "permuted_total_cluster_count",
        "original_qualified_cluster_count",
        "permuted_qualified_cluster_count",
        "same_qualified_cluster_count",
        "original_group_id",
        "original_group_name",
        "permuted_group_id_before_matching",
        "matched_permuted_group_name",
        "original_cluster_size",
        "permuted_cluster_size",
        "original_medoid_particle_index",
        "permuted_medoid_particle_index",
        "medoid_angle_difference_deg",
        "match_status",
    ]

    diagnostics_file.parent.mkdir(parents=True, exist_ok=True)
    write_header = (
        not diagnostics_file.exists()
        or diagnostics_file.stat().st_size == 0
    )

    with diagnostics_file.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def validate_partition_order_stability(
    freud_module: Any,
    orientations: np.ndarray,
    equivalent_quaternions: np.ndarray,
    condensed_distances: np.ndarray,
    original_labels: np.ndarray,
    particle_count: int,
    orientation_angle_tolerance_deg: float,
    cluster_size_cutoff: int,
    stability_trials: int,
    random_seed: int,
    temporary_directory: Path,
    diagnostics_file: Path,
    frame_index: int,
) -> bool | None:
    """Test cutoff-qualified cluster-count invariance under permutations.

    Only complete-linkage clusters with instantaneous particle count greater
    than or equal to ``cluster_size_cutoff`` participate in the pass/fail count
    and medoid-orientation diagnostics. Smaller clusters remain part of the raw
    clustering but are deliberately ignored by this stability criterion.

    Exact particle-membership equality is not required. A failed trial prints
    a warning, is recorded, and does not stop the main analysis.
    """

    if stability_trials <= 0:
        print(
            "PARTICLE-ORDER CUTOFF-QUALIFIED CLUSTER-COUNT CHECK: "
            f"frame {frame_index}: not tested because stability_trials=0."
        )
        return None

    original_particle_ids = np.arange(particle_count, dtype=np.int64)
    original_all_records = _cluster_medoid_records_from_labels(
        original_labels,
        original_particle_ids,
        particle_count,
        condensed_distances,
        orientations,
    )
    original_total_cluster_count = len(original_all_records)
    original_records = [
        dict(record)
        for record in original_all_records
        if int(record["size"]) >= cluster_size_cutoff
    ]

    # Reassign A/B/C/... only within the cutoff-qualified subset.
    for group_id, record in enumerate(original_records):
        record["group_id"] = int(group_id)
        record["group_name"] = _alphabetic_group_name(group_id)

    original_qualified_cluster_count = len(original_records)
    original_medoid_quaternions = (
        np.asarray(
            [record["medoid_quaternion"] for record in original_records],
            dtype=np.float64,
        )
        if original_records
        else np.empty((0, 4), dtype=np.float64)
    )

    rng = np.random.default_rng(random_seed + 1_000_003 * frame_index)
    all_qualified_cluster_counts_match = True

    for trial in range(stability_trials):
        permutation = rng.permutation(particle_count)
        trial_file = temporary_directory / (
            f"frame_{frame_index}_stability_trial_{trial}.dat"
        )

        permuted_distances = build_permuted_condensed_distances(
            condensed_distances,
            particle_count,
            permutation,
            trial_file,
        )

        try:
            permuted_labels = complete_linkage_labels(
                permuted_distances,
                orientation_angle_tolerance_deg,
            )
        finally:
            del permuted_distances
            try:
                trial_file.unlink()
            except FileNotFoundError:
                pass

        permuted_all_records = _cluster_medoid_records_from_labels(
            permuted_labels,
            permutation,
            particle_count,
            condensed_distances,
            orientations,
        )
        permuted_total_cluster_count = len(permuted_all_records)
        permuted_records = [
            dict(record)
            for record in permuted_all_records
            if int(record["size"]) >= cluster_size_cutoff
        ]

        for group_id, record in enumerate(permuted_records):
            record["group_id"] = int(group_id)
            record["group_name"] = _alphabetic_group_name(group_id)

        permuted_qualified_cluster_count = len(permuted_records)
        same_qualified_cluster_count = (
            permuted_qualified_cluster_count
            == original_qualified_cluster_count
        )

        if same_qualified_cluster_count:
            print(
                "PARTICLE-ORDER CUTOFF-QUALIFIED CLUSTER-COUNT CHECK: "
                f"frame {frame_index}, trial {trial + 1}/{stability_trials}: "
                f"PASSED ({original_qualified_cluster_count} clusters with "
                f"size >= {cluster_size_cutoff} in both runs; total raw "
                f"clusters {original_total_cluster_count} vs "
                f"{permuted_total_cluster_count})."
            )
        else:
            all_qualified_cluster_counts_match = False
            print(
                "WARNING: PARTICLE-ORDER CUTOFF-QUALIFIED CLUSTER-COUNT "
                f"CHECK FAILED: frame {frame_index}, trial "
                f"{trial + 1}/{stability_trials}: original run produced "
                f"{original_qualified_cluster_count} cluster(s) with size >= "
                f"{cluster_size_cutoff}, permuted run produced "
                f"{permuted_qualified_cluster_count}. Total raw cluster counts "
                f"were {original_total_cluster_count} and "
                f"{permuted_total_cluster_count}. The analysis will continue; "
                "inspect the medoid diagnostic CSV."
            )

        common_values = {
            "frame_index": int(frame_index),
            "trial_index": int(trial + 1),
            "cluster_size_cutoff": int(cluster_size_cutoff),
            "original_total_cluster_count": int(original_total_cluster_count),
            "permuted_total_cluster_count": int(permuted_total_cluster_count),
            "original_qualified_cluster_count": int(
                original_qualified_cluster_count
            ),
            "permuted_qualified_cluster_count": int(
                permuted_qualified_cluster_count
            ),
            "same_qualified_cluster_count": bool(
                same_qualified_cluster_count
            ),
        }
        diagnostic_rows: list[dict[str, Any]] = []

        if original_records and permuted_records:
            permuted_medoid_quaternions = np.asarray(
                [record["medoid_quaternion"] for record in permuted_records],
                dtype=np.float64,
            )

            # Rows: cutoff-qualified original groups.
            # Columns: cutoff-qualified permuted groups.
            medoid_angle_matrix = angular_distance_matrix_deg(
                freud_module,
                permuted_medoid_quaternions,
                original_medoid_quaternions,
                equivalent_quaternions,
            )
            original_assignment, permuted_assignment = linear_sum_assignment(
                medoid_angle_matrix
            )
            matched_original = set(int(value) for value in original_assignment)
            matched_permuted = set(int(value) for value in permuted_assignment)

            for original_id, permuted_id in zip(
                original_assignment,
                permuted_assignment,
            ):
                original_id = int(original_id)
                permuted_id = int(permuted_id)
                original_record = original_records[original_id]
                permuted_record = permuted_records[permuted_id]
                original_name = str(original_record["group_name"])

                diagnostic_rows.append(
                    {
                        **common_values,
                        "original_group_id": original_id,
                        "original_group_name": original_name,
                        "permuted_group_id_before_matching": permuted_id,
                        "matched_permuted_group_name": original_name + "'",
                        "original_cluster_size": int(original_record["size"]),
                        "permuted_cluster_size": int(permuted_record["size"]),
                        "original_medoid_particle_index": int(
                            original_record["medoid_particle"]
                        ),
                        "permuted_medoid_particle_index": int(
                            permuted_record["medoid_particle"]
                        ),
                        "medoid_angle_difference_deg": float(
                            medoid_angle_matrix[original_id, permuted_id]
                        ),
                        "match_status": (
                            "matched_cutoff_qualified_groups_by_"
                            "minimum_total_medoid_angle"
                        ),
                    }
                )

            for original_id, original_record in enumerate(original_records):
                if original_id in matched_original:
                    continue
                diagnostic_rows.append(
                    {
                        **common_values,
                        "original_group_id": int(original_id),
                        "original_group_name": str(
                            original_record["group_name"]
                        ),
                        "permuted_group_id_before_matching": "",
                        "matched_permuted_group_name": "",
                        "original_cluster_size": int(original_record["size"]),
                        "permuted_cluster_size": "",
                        "original_medoid_particle_index": int(
                            original_record["medoid_particle"]
                        ),
                        "permuted_medoid_particle_index": "",
                        "medoid_angle_difference_deg": "",
                        "match_status": (
                            "unmatched_cutoff_qualified_original_group"
                        ),
                    }
                )

            for permuted_id, permuted_record in enumerate(permuted_records):
                if permuted_id in matched_permuted:
                    continue
                diagnostic_rows.append(
                    {
                        **common_values,
                        "original_group_id": "",
                        "original_group_name": "",
                        "permuted_group_id_before_matching": int(permuted_id),
                        "matched_permuted_group_name": "",
                        "original_cluster_size": "",
                        "permuted_cluster_size": int(permuted_record["size"]),
                        "original_medoid_particle_index": "",
                        "permuted_medoid_particle_index": int(
                            permuted_record["medoid_particle"]
                        ),
                        "medoid_angle_difference_deg": "",
                        "match_status": (
                            "unmatched_cutoff_qualified_permuted_group"
                        ),
                    }
                )
        else:
            # Keep a trial-level audit row even when one or both qualified
            # subsets are empty and no medoid assignment can be calculated.
            diagnostic_rows.append(
                {
                    **common_values,
                    "original_group_id": "",
                    "original_group_name": "",
                    "permuted_group_id_before_matching": "",
                    "matched_permuted_group_name": "",
                    "original_cluster_size": "",
                    "permuted_cluster_size": "",
                    "original_medoid_particle_index": "",
                    "permuted_medoid_particle_index": "",
                    "medoid_angle_difference_deg": "",
                    "match_status": (
                        "no_medoid_assignment_because_at_least_one_"
                        "cutoff_qualified_subset_is_empty"
                    ),
                }
            )

        _append_particle_order_medoid_diagnostics(
            diagnostics_file,
            diagnostic_rows,
        )

    if all_qualified_cluster_counts_match:
        print(
            "PARTICLE-ORDER CUTOFF-QUALIFIED CLUSTER-COUNT SUMMARY: "
            f"frame {frame_index}: all {stability_trials} trial(s) passed "
            f"for cluster size >= {cluster_size_cutoff}."
        )
    else:
        print(
            "WARNING: PARTICLE-ORDER CUTOFF-QUALIFIED CLUSTER-COUNT "
            f"SUMMARY: frame {frame_index}: at least one of "
            f"{stability_trials} trial(s) changed the number of clusters with "
            f"size >= {cluster_size_cutoff}. The main workflow is continuing."
        )

    return all_qualified_cluster_counts_match



def construct_frame_clusters(
    frame_index: int,
    orientations: np.ndarray,
    labels: np.ndarray,
    condensed_distances: np.ndarray,
    orientation_angle_tolerance_deg: float,
    angle_validation_tolerance_deg: float,
    max_quaternion_norm_deviation: float,
    distance_diagnostics: DistanceDiagnostics,
    cluster_count_stability_passed: bool | None,
) -> FrameClustering:
    particle_count = len(orientations)
    preliminary: list[FrameCluster] = []

    for original_label in np.unique(labels):
        members = np.where(labels == original_label)[0].astype(np.int64)
        (
            medoid_particle,
            diameter,
            maximum_to_medoid,
            medoid_sum,
        ) = cluster_distance_statistics(
            members,
            particle_count,
            condensed_distances,

            # Medoid ties are resolved with a tiny independent numerical tolerance.
            # The freud self-angle resolution must not be reused here.
            medoid_tie_tolerance_deg=MEDOID_TIE_TOL_DEG,
        )

        # Scientifically strict diameter validation:
        #
        #     max_{i,j in cluster} d_G(q_i,q_j)
        #         <= orientation_angle_tolerance_deg
        #
        # CLUSTER_DIAMETER_EPS_DEG only protects against the final few binary
        # floating-point digits. It does not meaningfully increase the physical
        # cutoff.
        if diameter > (
            orientation_angle_tolerance_deg
            + CLUSTER_DIAMETER_EPS_DEG
        ):
            raise RuntimeError(
                f"Frame {frame_index}: cluster generated by complete linkage has "
                f"diameter {diameter:.12g} deg, exceeding the requested "
                f"{orientation_angle_tolerance_deg:.12g} deg. "
                "The numerical self-angle floor is not permitted to relax the "
                "within-cluster diameter criterion."
            )

        preliminary.append(
            FrameCluster(
                frame_index=frame_index,
                local_cluster_id=-1,
                members=members,
                size=len(members),
                medoid_particle_index=medoid_particle,
                medoid_quaternion_wxyz=canonicalize_quaternion_sign(
                    orientations[medoid_particle]
                ),
                diameter_deg=diameter,
                maximum_angle_to_medoid_deg=maximum_to_medoid,
                medoid_sum_distance_deg=medoid_sum,
            )
        )

    # Fixes the old sorting bug: the cluster membership, size and representative
    # medoid remain in one record and are sorted together, never in separate
    # arrays whose indices can become inconsistent.
    preliminary.sort(
        key=lambda cluster: (
            -cluster.size,
            cluster.medoid_particle_index,
            tuple(int(value) for value in cluster.members),
        )
    )

    particle_to_cluster = np.full(particle_count, -1, dtype=np.int64)
    clusters: list[FrameCluster] = []

    for local_id, cluster in enumerate(preliminary):
        if np.any(particle_to_cluster[cluster.members] != -1):
            raise RuntimeError(
                f"Frame {frame_index}: particle assigned to multiple clusters."
            )
        particle_to_cluster[cluster.members] = local_id
        clusters.append(
            FrameCluster(
                frame_index=cluster.frame_index,
                local_cluster_id=local_id,
                members=cluster.members,
                size=cluster.size,
                medoid_particle_index=cluster.medoid_particle_index,
                medoid_quaternion_wxyz=cluster.medoid_quaternion_wxyz,
                diameter_deg=cluster.diameter_deg,
                maximum_angle_to_medoid_deg=cluster.maximum_angle_to_medoid_deg,
                medoid_sum_distance_deg=cluster.medoid_sum_distance_deg,
            )
        )

    if np.any(particle_to_cluster < 0):
        raise RuntimeError(
            f"Frame {frame_index}: clustering did not assign every particle."
        )
    if sum(cluster.size for cluster in clusters) != particle_count:
        raise RuntimeError(
            f"Frame {frame_index}: cluster populations do not conserve particles."
        )

    return FrameClustering(
        frame_index=frame_index,
        particle_count=particle_count,
        clusters=tuple(clusters),
        particle_to_local_cluster=particle_to_cluster,
        maximum_quaternion_norm_deviation=max_quaternion_norm_deviation,
        distance_diagnostics=distance_diagnostics,
        cluster_count_stability_passed=cluster_count_stability_passed,
    )


def cluster_one_frame(
    freud_module: Any,
    trajectory: Any,
    frame_index: int,
    requested_particles: int,
    equivalent_quaternions: np.ndarray,
    orientation_angle_tolerance_deg: float,
    cluster_size_cutoff: int,
    block_size: int,
    angle_validation_tolerance_deg: float,
    stability_trials: int,
    random_seed: int,
    temporary_directory: Path,
    stability_diagnostics_file: Path,
    keep_distance_file: bool,
) -> FrameClustering:

    snapshot = trajectory[frame_index]
    orientations, norm_deviation = validate_and_normalize_orientations(
        snapshot.particles.orientation,
        requested_particles,
        frame_index,
    )

    distance_file = temporary_directory / (
        f"frame_{frame_index}_condensed_pairwise_angles_float64.dat"
    )
    condensed_distances, diagnostics = compute_condensed_distance_memmap(
        freud_module,
        orientations,
        equivalent_quaternions,
        frame_index,
        block_size,
        distance_file,
        angle_validation_tolerance_deg,
    )

    try:
        labels = complete_linkage_labels(
            condensed_distances,
            orientation_angle_tolerance_deg,
        )

        cluster_count_stability_passed = validate_partition_order_stability(
            freud_module,
            orientations,
            equivalent_quaternions,
            condensed_distances,
            labels,
            requested_particles,
            orientation_angle_tolerance_deg,
            cluster_size_cutoff,
            stability_trials,
            random_seed,
            temporary_directory,
            stability_diagnostics_file,
            frame_index,
        )

        result = construct_frame_clusters(
            frame_index,
            orientations,
            labels,
            condensed_distances,
            orientation_angle_tolerance_deg,
            angle_validation_tolerance_deg,
            norm_deviation,
            diagnostics,
            cluster_count_stability_passed,
        )
    finally:
        condensed_distances.flush()
        del condensed_distances
        if not keep_distance_file:
            try:
                distance_file.unlink()
            except FileNotFoundError:
                pass

    return result



def build_cutoff_qualified_reference(
    reference: FrameClustering,
    cluster_size_cutoff: int,
) -> FrameClustering:
    """Retain only reference-frame clusters meeting the instantaneous cutoff."""

    qualified_clusters = tuple(
        cluster
        for cluster in reference.clusters
        if cluster.size >= cluster_size_cutoff
    )
    if not qualified_clusters:
        raise RuntimeError(
            f"Reference frame {reference.frame_index} contains no cluster with "
            f"size >= cluster_size_cutoff ({cluster_size_cutoff}); fixed-state "
            "tracking cannot be defined."
        )

    return FrameClustering(
        frame_index=reference.frame_index,
        particle_count=reference.particle_count,
        clusters=qualified_clusters,
        particle_to_local_cluster=reference.particle_to_local_cluster,
        maximum_quaternion_norm_deviation=(
            reference.maximum_quaternion_norm_deviation
        ),
        distance_diagnostics=reference.distance_diagnostics,
        cluster_count_stability_passed=(
            reference.cluster_count_stability_passed
        ),
    )


def reference_medoid_distance_matrix(
    freud_module: Any,
    reference_clusters: Sequence[FrameCluster],
    equivalent_quaternions: np.ndarray,
    angle_validation_tolerance_deg: float,
) -> np.ndarray:
    """
    Calculate and validate the symmetry-reduced distance matrix between
    reference-state medoids.

    A small nonzero diagonal is treated as the numerical self-angle floor.
    The returned matrix has its diagonal explicitly set to exact zero because
    a reference medoid is mathematically identical to itself.
    """

    quaternions = np.asarray(
        [
            cluster.medoid_quaternion_wxyz
            for cluster in reference_clusters
        ],
        dtype=np.float64,
    )

    matrix = angular_distance_matrix_deg(
        freud_module,
        quaternions,
        quaternions,
        equivalent_quaternions,
    )

    max_self = float(
        np.max(
            np.abs(
                np.diag(matrix)
            )
        )
    )

    if max_self > NUMERICAL_SELF_ANGLE_HARD_LIMIT_DEG:
        raise RuntimeError(
            "Reference-medoid self-distances are too large to be explained "
            "by normal numerical precision: "
            f"maximum={max_self:.12g} deg, hard limit="
            f"{NUMERICAL_SELF_ANGLE_HARD_LIMIT_DEG:.12g} deg."
        )

    effective_metric_tolerance_deg = max(
        angle_validation_tolerance_deg,
        NUMERICAL_FLOOR_SAFETY_FACTOR * max_self,
    )

    if max_self > angle_validation_tolerance_deg:
        print(
            "REFERENCE-MEDOID SELF-ANGLE NOTICE: "
            f"maximum d(q,q)={max_self:.12g} deg. "
            "The diagonal will be set to exact zero before reference-state "
            "separation and tracking-cutoff calculations."
        )

    # Self-distance is mathematically zero. Correct only the diagonal of this
    # medoid-distance matrix. Off-diagonal physical distances are untouched.
    matrix = np.array(matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(matrix, 0.0)

    max_asymmetry = float(
        np.max(
            np.abs(
                matrix - matrix.T
            )
        )
    )

    if max_asymmetry > effective_metric_tolerance_deg:
        raise RuntimeError(
            "Reference-medoid distance matrix is not symmetric within the "
            "measured numerical resolution: "
            f"maximum asymmetry={max_asymmetry:.12g} deg; "
            f"allowed numerical tolerance="
            f"{effective_metric_tolerance_deg:.12g} deg."
        )

    print(
        "REFERENCE-MEDOID DISTANCE CHECK: PASSED. "
        f"self floor={max_self:.12g} deg; "
        f"maximum asymmetry={max_asymmetry:.12g} deg."
    )

    return matrix


def tracking_cutoff_suggestion(
    orientation_angle_tolerance_deg: float,
    reference_distance_matrix: np.ndarray,
    angle_validation_tolerance_deg: float,
) -> tuple[float, float | None]:
    reference_count = len(reference_distance_matrix)
    if reference_count <= 1:
        return orientation_angle_tolerance_deg / 2.0, None

    upper_triangle = reference_distance_matrix[
        np.triu_indices(reference_count, k=1)
    ]
    minimum_separation = float(np.min(upper_triangle))
    safe_exclusive_upper_bound = 0.5 * minimum_separation
    suggestion = min(
        orientation_angle_tolerance_deg / 2.0,
        0.49 * minimum_separation,
    )

    if suggestion <= angle_validation_tolerance_deg:
        raise RuntimeError(
            "Reference medoids are too close to define a numerically meaningful "
            "non-overlapping tracking tolerance."
        )
    return suggestion, safe_exclusive_upper_bound


def globally_match_clusters_to_reference_states(
    current_clusters: Sequence[FrameCluster],
    reference_clusters: Sequence[FrameCluster],
    medoid_distance_matrix_deg: np.ndarray,
    tracking_angle_tolerance_deg: float,
    angle_validation_tolerance_deg: float,
    allow_ambiguous_tracking: bool,
) -> tuple[
    tuple[int, ...],
    tuple[float, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    current_count = len(current_clusters)
    reference_count = len(reference_clusters)

    valid = (
        medoid_distance_matrix_deg
        <= tracking_angle_tolerance_deg + angle_validation_tolerance_deg
    )

    ambiguous_current = tuple(
        int(index)
        for index in np.where(np.sum(valid, axis=1) > 1)[0]
    )
    split_candidate_references = tuple(
        int(index)
        for index in np.where(np.sum(valid, axis=0) > 1)[0]
    )

    if (
        not allow_ambiguous_tracking
        and (ambiguous_current or split_candidate_references)
    ):
        raise RuntimeError(
            "Strict fixed-reference tracking found an ambiguous match or a "
            "split candidate. Reduce the tracking tolerance or inspect the "
            "frame-level cluster output. "
            f"Ambiguous current clusters={ambiguous_current}; "
            f"reference states with multiple candidate clusters="
            f"{split_candidate_references}."
        )

    size = current_count + reference_count
    large_cost = 1.0e12
    unmatched_cost = (
        tracking_angle_tolerance_deg
        + max(angle_validation_tolerance_deg, 1.0e-9)
    )
    cost = np.full((size, size), large_cost, dtype=np.float64)

    actual_cost = np.where(
        valid,
        medoid_distance_matrix_deg,
        large_cost,
    )
    cost[:current_count, :reference_count] = actual_cost

    for current_index in range(current_count):
        cost[current_index, reference_count + current_index] = unmatched_cost

    for reference_index in range(reference_count):
        cost[current_count + reference_index, reference_index] = unmatched_cost

    cost[current_count:, reference_count:] = 0.0

    row_indices, column_indices = linear_sum_assignment(cost)

    cluster_to_state = np.full(current_count, -1, dtype=np.int64)
    tracking_distances = np.full(current_count, np.nan, dtype=np.float64)

    for row, column in zip(row_indices, column_indices):
        if row < current_count and column < reference_count:
            distance = float(medoid_distance_matrix_deg[row, column])
            if distance > (
                tracking_angle_tolerance_deg
                + angle_validation_tolerance_deg
            ):
                raise RuntimeError("Assignment accepted a tracking distance above cutoff.")
            cluster_to_state[row] = column
            tracking_distances[row] = distance

    matched_states = set(int(value) for value in cluster_to_state if value >= 0)
    unmatched_clusters = tuple(
        int(index)
        for index in np.where(cluster_to_state < 0)[0]
    )
    absent_states = tuple(
        state
        for state in range(reference_count)
        if state not in matched_states
    )

    if len(matched_states) != int(np.sum(cluster_to_state >= 0)):
        raise RuntimeError("A reference state was matched more than once.")

    return (
        tuple(int(value) for value in cluster_to_state),
        tuple(float(value) for value in tracking_distances),
        unmatched_clusters,
        absent_states,
        split_candidate_references,
        ambiguous_current,
    )


def track_all_frames_to_fixed_reference(
    freud_module: Any,
    frame_clusterings: Sequence[FrameClustering],
    reference: FrameClustering,
    cluster_size_cutoff: int,
    equivalent_quaternions: np.ndarray,
    tracking_angle_tolerance_deg: float,
    angle_validation_tolerance_deg: float,
    allow_ambiguous_tracking: bool,
) -> tuple[tuple[FrameTracking, ...], FrameClustering]:
    """Track only instantaneous cutoff-qualified clusters.

    Reference states are the reference-frame clusters retained in ``reference``.
    In every frame, only clusters with size >= ``cluster_size_cutoff`` are
    eligible for medoid matching. Smaller clusters are retained in the raw
    frame clustering, assigned state -1, and included in unmatched population.
    """

    reference_clusters = reference.clusters
    reference_frame_index = reference.frame_index
    reference_count = len(reference_clusters)
    tracking_results: list[FrameTracking] = []

    for frame in frame_clusterings:
        full_cluster_count = len(frame.clusters)
        full_cluster_to_state = np.full(
            full_cluster_count,
            -1,
            dtype=np.int64,
        )
        full_tracking_distances = np.full(
            full_cluster_count,
            np.nan,
            dtype=np.float64,
        )

        if frame.frame_index == reference_frame_index:
            for state_id, reference_cluster in enumerate(reference_clusters):
                local_id = int(reference_cluster.local_cluster_id)
                full_cluster_to_state[local_id] = state_id
                full_tracking_distances[local_id] = 0.0

            unmatched_clusters = tuple(
                int(index)
                for index in np.where(full_cluster_to_state < 0)[0]
            )
            tracking_results.append(
                FrameTracking(
                    frame_index=frame.frame_index,
                    cluster_to_reference_state=tuple(
                        int(value) for value in full_cluster_to_state
                    ),
                    cluster_tracking_distance_deg=tuple(
                        float(value) for value in full_tracking_distances
                    ),
                    unmatched_cluster_ids=unmatched_clusters,
                    absent_reference_state_ids=(),
                    split_candidate_reference_ids=(),
                    ambiguous_current_cluster_ids=(),
                )
            )
            continue

        eligible_local_ids = [
            cluster.local_cluster_id
            for cluster in frame.clusters
            if cluster.size >= cluster_size_cutoff
        ]
        eligible_current_clusters = tuple(
            frame.clusters[local_id]
            for local_id in eligible_local_ids
        )

        if eligible_current_clusters:
            current_quaternions = np.asarray(
                [
                    cluster.medoid_quaternion_wxyz
                    for cluster in eligible_current_clusters
                ],
                dtype=np.float64,
            )
            reference_quaternions = np.asarray(
                [
                    cluster.medoid_quaternion_wxyz
                    for cluster in reference_clusters
                ],
                dtype=np.float64,
            )
            distances_matrix = angular_distance_matrix_deg(
                freud_module,
                reference_quaternions,
                current_quaternions,
                equivalent_quaternions,
            )

            (
                eligible_cluster_to_state,
                eligible_tracking_distances,
                _,
                absent_states,
                split_candidates,
                ambiguous_eligible_indices,
            ) = globally_match_clusters_to_reference_states(
                eligible_current_clusters,
                reference_clusters,
                distances_matrix,
                tracking_angle_tolerance_deg,
                angle_validation_tolerance_deg,
                allow_ambiguous_tracking,
            )

            for eligible_index, local_id in enumerate(eligible_local_ids):
                full_cluster_to_state[local_id] = (
                    eligible_cluster_to_state[eligible_index]
                )
                full_tracking_distances[local_id] = (
                    eligible_tracking_distances[eligible_index]
                )

            ambiguous_current = tuple(
                int(eligible_local_ids[index])
                for index in ambiguous_eligible_indices
            )
        else:
            absent_states = tuple(range(reference_count))
            split_candidates = ()
            ambiguous_current = ()

        unmatched_clusters = tuple(
            int(index)
            for index in np.where(full_cluster_to_state < 0)[0]
        )
        matched_population = sum(
            frame.clusters[index].size
            for index, state in enumerate(full_cluster_to_state)
            if state >= 0
        )
        unmatched_population = sum(
            frame.clusters[index].size
            for index in unmatched_clusters
        )
        if matched_population + unmatched_population != frame.particle_count:
            raise RuntimeError(
                f"Frame {frame.frame_index}: tracking does not conserve particles."
            )

        tracking_results.append(
            FrameTracking(
                frame_index=frame.frame_index,
                cluster_to_reference_state=tuple(
                    int(value) for value in full_cluster_to_state
                ),
                cluster_tracking_distance_deg=tuple(
                    float(value) for value in full_tracking_distances
                ),
                unmatched_cluster_ids=unmatched_clusters,
                absent_reference_state_ids=tuple(
                    int(value) for value in absent_states
                ),
                split_candidate_reference_ids=tuple(
                    int(value) for value in split_candidates
                ),
                ambiguous_current_cluster_ids=ambiguous_current,
            )
        )

    return tuple(tracking_results), reference


def summarize_fixed_reference_states(
    frame_clusterings: Sequence[FrameClustering],
    tracking_results: Sequence[FrameTracking],
    reference: FrameClustering,
    cluster_size_cutoff: int,
) -> tuple[
    tuple[StateSummary, ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    frame_count = len(frame_clusterings)
    state_count = len(reference.clusters)
    particle_count = reference.particle_count

    counts = np.zeros((frame_count, state_count), dtype=np.int64)
    tracking_distances = np.full(
        (frame_count, state_count),
        np.nan,
        dtype=np.float64,
    )
    diameters = np.full(
        (frame_count, state_count),
        np.nan,
        dtype=np.float64,
    )
    unmatched_counts = np.zeros(frame_count, dtype=np.int64)

    clustering_by_frame = {
        frame.frame_index: frame
        for frame in frame_clusterings
    }

    for frame_position, tracking in enumerate(tracking_results):
        frame = clustering_by_frame[tracking.frame_index]
        for local_id, state_id in enumerate(
            tracking.cluster_to_reference_state
        ):
            cluster = frame.clusters[local_id]
            if state_id >= 0:
                counts[frame_position, state_id] = cluster.size
                tracking_distances[frame_position, state_id] = (
                    tracking.cluster_tracking_distance_deg[local_id]
                )
                diameters[frame_position, state_id] = cluster.diameter_deg
            else:
                unmatched_counts[frame_position] += cluster.size

        if int(np.sum(counts[frame_position])) + int(
            unmatched_counts[frame_position]
        ) != particle_count:
            raise RuntimeError(
                f"Frame {frame.frame_index}: state populations do not conserve "
                "the selected particle count."
            )

    fractions = counts.astype(np.float64) / float(particle_count)
    summaries: list[StateSummary] = []

    for state_id, reference_cluster in enumerate(reference.clusters):
        state_counts = counts[:, state_id]
        present_mask = state_counts > 0
        finite_tracking = tracking_distances[:, state_id][
            np.isfinite(tracking_distances[:, state_id])
        ]
        finite_diameters = diameters[:, state_id][
            np.isfinite(diameters[:, state_id])
        ]

        summaries.append(
            StateSummary(
                reference_state_id=state_id,
                reference_local_cluster_id=reference_cluster.local_cluster_id,
                reference_size=reference_cluster.size,
                reference_medoid_particle_index=(
                    reference_cluster.medoid_particle_index
                ),
                reference_medoid_quaternion_wxyz=(
                    reference_cluster.medoid_quaternion_wxyz
                ),
                mean_particle_count=float(np.mean(state_counts)),
                standard_deviation_particle_count=float(
                    np.std(state_counts, ddof=0)
                ),
                mean_population_fraction=float(
                    np.mean(fractions[:, state_id])
                ),
                standard_deviation_population_fraction=float(
                    np.std(fractions[:, state_id], ddof=0)
                ),
                presence_fraction=float(np.mean(present_mask)),
                maximum_tracking_distance_deg=(
                    float(np.max(finite_tracking))
                    if finite_tracking.size
                    else math.nan
                ),
                mean_tracking_distance_deg_when_present=(
                    float(np.mean(finite_tracking))
                    if finite_tracking.size
                    else math.nan
                ),
                maximum_cluster_diameter_deg=(
                    float(np.max(finite_diameters))
                    if finite_diameters.size
                    else math.nan
                ),
                valid_by_mean_size_cutoff=(
                    float(np.mean(state_counts)) >= cluster_size_cutoff
                ),
            )
        )

    # Fixes the old representative/size sorting mismatch: every field remains
    # inside one StateSummary record and summaries are sorted as whole records.
    summaries.sort(
        key=lambda state: (
            not state.valid_by_mean_size_cutoff,
            -state.mean_particle_count,
            state.reference_state_id,
        )
    )

    return (
        tuple(summaries),
        counts,
        fractions,
        unmatched_counts,
    )



def save_cluster_and_tracking_outputs(
    output_prefix: Path,
    frame_clusterings: Sequence[FrameClustering],
    tracking_results: Sequence[FrameTracking],
    reference: FrameClustering,
    state_summaries: Sequence[StateSummary],
    state_counts: np.ndarray,
    state_fractions: np.ndarray,
    unmatched_counts: np.ndarray,
    selected_frame_indices: Sequence[int],
    selected_particle_count: int,
    cluster_size_cutoff: int,
    orientation_angle_tolerance_deg: float,
    tracking_angle_tolerance_deg: float,
) -> dict[str, Path]:
    paths = {
        "states": output_prefix.with_name(output_prefix.name + "_state_summary.csv"),
        "populations": output_prefix.with_name(output_prefix.name + "_per_frame_populations.csv"),
        "clusters": output_prefix.with_name(output_prefix.name + "_frame_clusters.csv"),
        "membership": output_prefix.with_name(output_prefix.name + "_particle_membership.csv"),
        "tracked_particle_states": output_prefix.with_name(
            output_prefix.name + "_tracked_particle_state_membership.csv"
        ),
        "references": output_prefix.with_name(output_prefix.name + "_reference_states.csv"),
    }

    state_rows = []
    for state in state_summaries:
        q = state.reference_medoid_quaternion_wxyz
        state_rows.append(
            [
                state.reference_state_id,
                state.reference_local_cluster_id,
                state.reference_size,
                state.reference_medoid_particle_index,
                q[0], q[1], q[2], q[3],
                state.mean_particle_count,
                state.standard_deviation_particle_count,
                state.mean_population_fraction,
                state.standard_deviation_population_fraction,
                state.presence_fraction,
                state.maximum_tracking_distance_deg,
                state.mean_tracking_distance_deg_when_present,
                state.maximum_cluster_diameter_deg,
                int(state.valid_by_mean_size_cutoff),
            ]
        )
    np.savetxt(
        paths["states"],
        np.asarray(state_rows, dtype=np.float64),
        delimiter=",",
        header=(
            "reference_state_id,reference_local_cluster_id,reference_size,"
            "reference_medoid_particle_index,w,x,y,z,mean_particle_count,"
            "std_particle_count,mean_population_fraction,std_population_fraction,"
            "presence_fraction,max_tracking_distance_deg,"
            "mean_tracking_distance_deg_when_present,max_cluster_diameter_deg,"
            "valid_by_mean_particle_count_cutoff"
        ),
        comments="",
        fmt="%.17g",
    )

    population_table = np.column_stack(
        (
            np.asarray(selected_frame_indices, dtype=np.int64),
            state_counts,
            state_fractions,
            unmatched_counts,
            unmatched_counts / float(selected_particle_count),
        )
    )
    state_count = state_counts.shape[1]
    population_header = (
        "frame_index,"
        + ",".join(f"state_{state}_count" for state in range(state_count))
        + ","
        + ",".join(f"state_{state}_fraction" for state in range(state_count))
        + ",unmatched_count,unmatched_fraction"
    )
    np.savetxt(
        paths["populations"],
        population_table,
        delimiter=",",
        header=population_header,
        comments="",
        fmt="%.17g",
    )

    tracking_by_frame = {
        result.frame_index: result
        for result in tracking_results
    }
    cluster_rows = []
    membership_rows = []
    tracked_particle_state_rows: list[tuple[int, int, int, str]] = []

    for frame in frame_clusterings:
        tracking = tracking_by_frame[frame.frame_index]
        for cluster in frame.clusters:
            local_id = cluster.local_cluster_id
            state_id = tracking.cluster_to_reference_state[local_id]
            tracking_distance = tracking.cluster_tracking_distance_deg[local_id]
            q = cluster.medoid_quaternion_wxyz
            cluster_rows.append(
                [
                    frame.frame_index,
                    local_id,
                    cluster.size,
                    cluster.medoid_particle_index,
                    q[0], q[1], q[2], q[3],
                    cluster.diameter_deg,
                    cluster.maximum_angle_to_medoid_deg,
                    cluster.medoid_sum_distance_deg,
                    state_id,
                    tracking_distance,
                    int(cluster.diameter_deg <= orientation_angle_tolerance_deg),
                ]
            )
            for particle in cluster.members:
                membership_rows.append(
                    [
                        frame.frame_index,
                        int(particle),
                        local_id,
                        state_id,
                    ]
                )

                # Write the new concise identity table only when this raw
                # frame-level cluster has been successfully assigned to one of
                # the fixed reference states. Particles in below-cutoff or
                # otherwise unmatched clusters have state_id == -1 and are
                # intentionally omitted. Particles outside the user-selected
                # first ``selected_particle_count`` orientations never appear
                # in frame.clusters and are therefore omitted automatically.
                if state_id >= 0:
                    tracked_particle_state_rows.append(
                        (
                            int(frame.frame_index),
                            int(particle),
                            int(state_id),
                            _alphabetic_group_name(int(state_id)),
                        )
                    )

    np.savetxt(
        paths["clusters"],
        np.asarray(cluster_rows, dtype=np.float64),
        delimiter=",",
        header=(
            "frame_index,local_cluster_id,cluster_size,medoid_particle_index,"
            "medoid_w,medoid_x,medoid_y,medoid_z,diameter_deg,"
            "max_angle_to_medoid_deg,medoid_sum_distance_deg,"
            "reference_state_id,tracking_distance_deg,"
            "diameter_within_requested_tolerance"
        ),
        comments="",
        fmt="%.17g",
    )
    np.savetxt(
        paths["membership"],
        np.asarray(membership_rows, dtype=np.int64),
        delimiter=",",
        header=(
            "frame_index,particle_index,local_cluster_id,reference_state_id"
        ),
        comments="",
        fmt="%d",
    )

    # This additional CSV is intentionally sparse: one row is written only for
    # an analysed particle whose frame-level cluster has a valid fixed-reference
    # identity. The alphabetic name is derived exclusively from the persistent
    # reference_state_id, so A/B/C/... has the same meaning in every frame.
    with paths["tracked_particle_states"].open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "particle_index",
                "reference_state_id",
                "cluster_name",
            ]
        )
        writer.writerows(tracked_particle_state_rows)

    reference_rows = []
    for state_id, cluster in enumerate(reference.clusters):
        q = cluster.medoid_quaternion_wxyz
        reference_rows.append(
            [
                state_id,
                cluster.size,
                cluster.medoid_particle_index,
                q[0], q[1], q[2], q[3],
                cluster.diameter_deg,
                cluster.maximum_angle_to_medoid_deg,
                cluster.medoid_sum_distance_deg,
            ]
        )
    np.savetxt(
        paths["references"],
        np.asarray(reference_rows, dtype=np.float64),
        delimiter=",",
        header=(
            "reference_state_id,reference_size,medoid_particle_index,"
            "w,x,y,z,diameter_deg,max_angle_to_medoid_deg,"
            "medoid_sum_distance_deg"
        ),
        comments="",
        fmt="%.17g",
    )

    return paths


def save_plots(
    plt_module: Any,
    output_prefix: Path,
    state_summaries: Sequence[StateSummary],
    state_fractions: np.ndarray,
    unmatched_counts: np.ndarray,
    selected_frame_indices: Sequence[int],
    selected_particle_count: int,
    show_plot: bool,
) -> tuple[Path, Path, Path, Path]:
    """
    Save both plots and one companion coordinate CSV for each plot.

    The CSV filenames use the PNG stem:

        for_plotting_<exact_png_stem>.csv

    The bar-plot CSV stores bar-centre x coordinates, plotted mean y values,
    and the plotted standard-deviation error bars.

    The time-series CSV uses long format: one row per plotted point per series.
    """

    # Only states passing the mean particle-count cutoff are displayed in the
    # two figures. The CSVs intentionally contain exactly the displayed states,
    # rather than every reference state in the full analysis.
    valid_states = [
        state
        for state in state_summaries
        if state.valid_by_mean_size_cutoff
    ]

    # -----------------------------------------------------------------------
    # Define the two PNG paths.
    # -----------------------------------------------------------------------
    bar_path = output_prefix.with_name(
        output_prefix.name + "_mean_state_populations.png"
    )
    time_path = output_prefix.with_name(
        output_prefix.name + "_state_population_timeseries.png"
    )

    # Companion CSV names use exactly the PNG stem, prefixed by
    # "for_plotting_". For example:
    #
    #   analysis_mean_state_populations.png
    #   for_plotting_analysis_mean_state_populations.csv
    bar_csv_path = bar_path.with_name(
        "for_plotting_" + bar_path.stem + ".csv"
    )
    time_csv_path = time_path.with_name(
        "for_plotting_" + time_path.stem + ".csv"
    )

    # =======================================================================
    # MEAN-STATE-POPULATION BAR PLOT AND ITS EXACT PLOTTING TABLE
    # =======================================================================
    #
    # Define the exact arrays that will be passed to axis.bar().
    if valid_states:
        bar_x = np.arange(len(valid_states), dtype=np.int64)
        bar_means = np.asarray(
            [
                state.mean_population_fraction
                for state in valid_states
            ],
            dtype=np.float64,
        )
        bar_errors = np.asarray(
            [
                state.standard_deviation_population_fraction
                for state in valid_states
            ],
            dtype=np.float64,
        )
        bar_labels = [
            f"State {state.reference_state_id}"
            for state in valid_states
        ]
    else:
        # Header-only CSV when no state passes the cutoff.
        bar_x = np.asarray([], dtype=np.int64)
        bar_means = np.asarray([], dtype=np.float64)
        bar_errors = np.asarray([], dtype=np.float64)
        bar_labels = []

    # Write the exact bar coordinates and uncertainty values before plotting.
    with bar_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_bar_center",
                "reference_state_id",
                "x_tick_label",
                "y_mean_population_fraction",
                "y_error_standard_deviation",
                "y_error_lower",
                "y_error_upper",
                "mean_particle_count",
                "standard_deviation_particle_count",
            ]
        )

        for x_value, state, label, mean, error in zip(
            bar_x,
            valid_states,
            bar_labels,
            bar_means,
            bar_errors,
        ):
            writer.writerow(
                [
                    int(x_value),
                    int(state.reference_state_id),
                    label,
                    float(mean),
                    float(error),
                    float(mean - error),
                    float(mean + error),
                    float(state.mean_particle_count),
                    float(state.standard_deviation_particle_count),
                ]
            )

    figure, axis = plt_module.subplots(
        figsize=(
            max(
                5.0,
                0.65 * max(1, len(valid_states)),
            ),
            3.8,
        )
    )

    if valid_states:
        axis.bar(
            bar_x,
            bar_means,
            yerr=bar_errors,
            capsize=3,
        )
        axis.set_xticks(
            bar_x,
            bar_labels,
            rotation=45,
            ha="right",
        )

        upper = float(np.max(bar_means + bar_errors))
        axis.set_ylim(
            0.0,
            max(0.05, 1.12 * upper),
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No fixed reference state passes the mean particle-count cutoff.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_ylim(0.0, 1.0)

    axis.set_ylabel("Mean population fraction")
    axis.set_xlabel("Fixed reference orientational state")
    axis.tick_params(direction="in")
    figure.tight_layout()
    figure.savefig(
        bar_path,
        dpi=600,
        bbox_inches="tight",
    )

    if show_plot:
        plt_module.show()
    else:
        plt_module.close(figure)

    # =======================================================================
    # STATE-POPULATION TIME-SERIES PLOT AND ITS EXACT PLOTTING TABLE
    # =======================================================================
    #
    # Convert frame indices once to the exact x array used for every series.
    time_x = np.asarray(
        selected_frame_indices,
        dtype=np.int64,
    )

    # The unmatched fraction is the exact y array used for the dashed x-marker
    # series in the plot.
    unmatched_fraction = (
        np.asarray(
            unmatched_counts,
            dtype=np.float64,
        )
        / float(selected_particle_count)
    )

    # Use long format so the CSV can be replotted with a simple group-by over
    # series_order or series_label.
    with time_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "series_order",
                "series_type",
                "reference_state_id",
                "series_label",
                "x_frame_index",
                "y_population_fraction",
                "marker",
                "linestyle",
                "linewidth",
            ]
        )

        # Write every plotted valid-state point.
        for series_order, state in enumerate(valid_states):
            y_values = np.asarray(
                state_fractions[
                    :,
                    state.reference_state_id,
                ],
                dtype=np.float64,
            )

            for x_value, y_value in zip(time_x, y_values):
                writer.writerow(
                    [
                        int(series_order),
                        "reference_state",
                        int(state.reference_state_id),
                        f"State {state.reference_state_id}",
                        int(x_value),
                        float(y_value),
                        "o",
                        "-",
                        1.2,
                    ]
                )

        # Write every plotted unmatched-fraction point.
        unmatched_series_order = len(valid_states)
        for x_value, y_value in zip(
            time_x,
            unmatched_fraction,
        ):
            writer.writerow(
                [
                    int(unmatched_series_order),
                    "unmatched_frame_level_groups",
                    "",
                    "Unmatched frame-level groups",
                    int(x_value),
                    float(y_value),
                    "x",
                    "--",
                    1.0,
                ]
            )

    figure, axis = plt_module.subplots(figsize=(6.0, 4.0))

    for state in valid_states:
        axis.plot(
            time_x,
            state_fractions[:, state.reference_state_id],
            marker="o",
            linewidth=1.2,
            label=f"State {state.reference_state_id}",
        )

    axis.plot(
        time_x,
        unmatched_fraction,
        marker="x",
        linestyle="--",
        linewidth=1.0,
        label="Unmatched frame-level groups",
    )

    axis.set_xlabel("Trajectory frame index")
    axis.set_ylabel("Population fraction")
    axis.set_ylim(bottom=0.0)
    axis.tick_params(direction="in")

    if valid_states or len(selected_frame_indices):
        axis.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    figure.savefig(
        time_path,
        dpi=600,
        bbox_inches="tight",
    )

    if show_plot:
        plt_module.show()
    else:
        plt_module.close(figure)

    return (
        bar_path,
        time_path,
        bar_csv_path,
        time_csv_path,
    )

def save_analysis_metadata(
    output_prefix: Path,
    trajectory_path: Path,
    shape_path: Path,
    total_frames: int,
    selected_frame_indices: Sequence[int],
    selected_particle_count: int,
    reference_frame_index: int,
    expected_edges: int,
    expected_faces: int,
    precision_exponent: int,
    orientation_angle_tolerance_deg: float,
    tracking_angle_tolerance_deg: float,
    safe_tracking_upper_bound_deg: float | None,
    cluster_size_cutoff: int,
    block_size: int,
    stability_trials: int,
    random_seed: int,
    stability_diagnostics_file: Path,
    angle_validation_tolerance_deg: float,
    symmetry_result: SymmetryResult,
    frame_clusterings: Sequence[FrameClustering],
    tracking_results: Sequence[FrameTracking],
    state_summaries: Sequence[StateSummary],
) -> Path:
    path = output_prefix.with_name(output_prefix.name + "_metadata.json")

    metadata = {
        "trajectory_file": str(trajectory_path),
        "shape_file": str(shape_path),
        "total_trajectory_frames": int(total_frames),
        "selected_frame_indices": [int(v) for v in selected_frame_indices],
        "selected_particle_count_per_frame": int(selected_particle_count),
        "reference_frame_index": int(reference_frame_index),
        "frame_clustering_method": "complete-linkage hierarchical clustering",
        "frame_cluster_constraint": (
            "explicitly validated maximum symmetry-reduced pairwise angular "
            "diameter <= orientation_angle_tolerance_deg"
        ),
        "orientation_angle_tolerance_deg": float(
            orientation_angle_tolerance_deg
        ),
        "representative_definition": (
            "symmetry-aware medoid: actual member minimising sum of "
            "within-cluster pairwise angular distances; smallest particle "
            "index resolves numerical ties"
        ),
        "tracking_strategy": (
            "fixed reference states are reference-frame clusters with "
            "instantaneous size >= cluster_size_cutoff; only current-frame "
            "clusters meeting the same cutoff are eligible for global "
            "one-to-one medoid assignment; smaller clusters are unmatched"
        ),
        "tracking_angle_tolerance_deg": float(
            tracking_angle_tolerance_deg
        ),
        "nonoverlapping_tracking_safe_upper_bound_deg": (
            None
            if safe_tracking_upper_bound_deg is None
            else float(safe_tracking_upper_bound_deg)
        ),
        "absent_state_population_rule": (
            "zero population included in averages"
        ),
        "cluster_size_cutoff_definition": (
            "one shared threshold: instantaneous minimum cluster size for "
            "particle-order stability and fixed-reference tracking, and "
            "minimum mean state population for final validity"
        ),
        "cluster_size_cutoff": int(cluster_size_cutoff),
        "expected_edges": int(expected_edges),
        "expected_faces": int(expected_faces),
        "symmetry_precision_exponent": int(precision_exponent),
        "symmetry_matching_tolerance": float(
            symmetry_result.matching_tolerance
        ),
        "physical_proper_rotation_count": int(
            len(symmetry_result.physical_operations)
        ),
        "equivalent_quaternion_count": int(
            len(symmetry_result.equivalent_quaternions_wxyz)
        ),
        "block_size": int(block_size),
        "particle_order_stability_trials_per_frame": int(stability_trials),
        "particle_order_stability_criterion": (
            "same number of complete-linkage clusters with instantaneous size "
            ">= cluster_size_cutoff after particle-order permutation; "
            "exact membership equality and raw small-cluster count are not "
            "required"
        ),
        "particle_order_stability_failure_action": (
            "print warning, write medoid diagnostics, and continue"
        ),
        "particle_order_medoid_diagnostics_file": (
            str(stability_diagnostics_file)
            if stability_trials > 0
            else None
        ),
        "random_seed": int(random_seed),
        "angle_validation_tolerance_deg": float(
            angle_validation_tolerance_deg
        ),
        "frames": [
            {
                "frame_index": int(frame.frame_index),
                "cluster_count": int(len(frame.clusters)),
                "cutoff_qualified_cluster_count": int(
                    sum(
                        cluster.size >= cluster_size_cutoff
                        for cluster in frame.clusters
                    )
                ),
                "max_quaternion_norm_deviation": float(
                    frame.maximum_quaternion_norm_deviation
                ),
                "minimum_pair_angle_deg": float(
                    frame.distance_diagnostics.minimum_angle_deg
                ),
                "maximum_pair_angle_deg": float(
                    frame.distance_diagnostics.maximum_angle_deg
                ),
                "cluster_count_stability_passed": (
                    None
                    if frame.cluster_count_stability_passed is None
                    else bool(frame.cluster_count_stability_passed)
                ),
                "maximum_cluster_diameter_deg": float(
                    max(cluster.diameter_deg for cluster in frame.clusters)
                ),
            }
            for frame in frame_clusterings
        ],
        "tracking": [
            {
                "frame_index": int(result.frame_index),
                "unmatched_cluster_ids": [
                    int(v) for v in result.unmatched_cluster_ids
                ],
                "absent_reference_state_ids": [
                    int(v) for v in result.absent_reference_state_ids
                ],
                "split_candidate_reference_ids": [
                    int(v)
                    for v in result.split_candidate_reference_ids
                ],
                "ambiguous_current_cluster_ids": [
                    int(v)
                    for v in result.ambiguous_current_cluster_ids
                ],
            }
            for result in tracking_results
        ],
        "state_summaries": [
            {
                "reference_state_id": int(state.reference_state_id),
                "mean_particle_count": float(state.mean_particle_count),
                "mean_population_fraction": float(
                    state.mean_population_fraction
                ),
                "presence_fraction": float(state.presence_fraction),
                "valid_by_mean_size_cutoff": bool(
                    state.valid_by_mean_size_cutoff
                ),
            }
            for state in state_summaries
        ],
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rigorous complete-linkage orientational-state clustering with "
            "fixed-reference medoid tracking."
        )
    )
    parser.add_argument("trajectory", help="Input HOOMD GSD trajectory")
    parser.add_argument("shape", help="Particle shape JSON")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--reference-frame", type=int, default=None)
    parser.add_argument("--edges", type=int, default=None)
    parser.add_argument("--faces", type=int, default=None)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--orientation-angle-tol", type=float, default=None)
    parser.add_argument("--tracking-angle-tol", type=float, default=None)
    parser.add_argument("--cluster-size-cutoff", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--stability-trials",
        type=int,
        default=None,
        help=(
            "Random particle-order permutations tested per frame. "
            "Default suggestion: 1."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--angle-validation-tol",
        type=float,
        default=DEFAULT_ANGLE_VALIDATION_TOL_DEG,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--keep-distance-files", action="store_true")
    parser.add_argument(
        "--allow-overlapping-tracking-regions",
        action="store_true",
        help=(
            "Allow tracking cutoff >= half the minimum reference-medoid "
            "separation. Strict mode rejects this."
        ),
    )
    parser.add_argument(
        "--allow-ambiguous-tracking",
        action="store_true",
        help=(
            "Allow current clusters to have multiple candidate reference states "
            "or one reference state to have multiple candidate clusters."
        ),
    )
    return parser


def main() -> int:

    # =======================================================================
    # STAGE 0 — PARSE COMMAND-LINE ARGUMENTS
    # =======================================================================
    args = build_argument_parser().parse_args()

    try:
        # ===================================================================
        # STAGE 1 — VALIDATE NONINTERACTIVE NUMERICAL ARGUMENTS
        # ===================================================================
        if args.block_size < 1:
            raise ValueError("--block-size must be positive.")
        if args.angle_validation_tol <= 0.0:
            raise ValueError("--angle-validation-tol must be positive.")

        # ===================================================================
        # STAGE 2 — IMPORT RUNTIME-ONLY PACKAGES AND OPEN INPUT FILES
        # ===================================================================
        freud, gsd_hoomd, plt = import_runtime_packages()
        # Resolve and open the GSD trajectory in read-only mode.
        trajectory_path, trajectory = open_gsd_trajectory(gsd_hoomd, args.trajectory)
        # Resolve the shape JSON path.  Existence and content are validated later by read_shape_vertices().
        shape_path = Path(args.shape).expanduser().resolve()

        total_frames = len(trajectory)
        # An empty trajectory cannot define selected frames or orientations.
        if total_frames < 1:
            raise ValueError("The trajectory contains no frames.")

        # Print the full trajectory range before asking the user to choose the
        # number of final frames.
        print("\nTrajectory summary")
        print("------------------")
        print(f"Trajectory: {trajectory_path}")
        print(f"Number of available frames: {total_frames}")
        print(f"Valid frame indices: 0 through {total_frames - 1}")

        # ===================================================================
        # STAGE 3 — SELECT THE CONSECUTIVE FINAL FRAME WINDOW
        # ===================================================================
        frames_to_analyse = resolve_or_prompt(
            args.frames,
            "How many consecutive final frames should be analysed?",
            min(CURRENT_SUGGESTED_FRAMES, total_frames),
            int,
            lambda value: 1 <= value <= total_frames,
            f"Enter an integer from 1 through {total_frames}.",
        )
        # If T frames exist and F are requested, the first selected zero-based index is T-F.
        first_frame = total_frames - frames_to_analyse
        # Selected frames are consecutive and include the final trajectory frame: [T-F, T-F+1, ..., T-1].
        selected_frame_indices = list(range(first_frame, total_frames))
        print(f"Selected final frame indices: {selected_frame_indices}")

        # ===================================================================
        # STAGE 4 — INSPECT PARTICLE COUNTS AND ORIENTATION SHAPES
        # ===================================================================
        available_counts = []
        for frame_index in selected_frame_indices:
            orientation_array = np.asarray(
                trajectory[frame_index].particles.orientation
            )
            if orientation_array.ndim != 2 or orientation_array.shape[1] != 4:
                raise ValueError(
                    f"Frame {frame_index}: invalid orientation shape "
                    f"{orientation_array.shape}."
                )
            available_counts.append(len(orientation_array))

        minimum_particles = min(available_counts)
        maximum_particles = max(available_counts)
        print(
            "Particles in selected frames: "
            f"minimum={minimum_particles}, maximum={maximum_particles}"
        )

        # ===================================================================
        # STAGE 5 — RESOLVE ALL USER SCIENTIFIC PARAMETERS
        # ===================================================================
        # Particle count: analyse the first M particles in every selected frame, with:

        # Fixed-reference frame: must be one of the already selected frames.  The final selected frame is the default.
        selected_particle_count = resolve_or_prompt(
            args.particles,
            "How many particles should be analysed in every selected frame?",
            minimum_particles,
            int,
            lambda value: 2 <= value <= minimum_particles,
            f"Enter an integer from 2 through {minimum_particles}.",
        )

        # Fixed-reference frame:
        # must be one of the already selected frames.  The final selected frame is the default.
        reference_frame_index = resolve_or_prompt(
            args.reference_frame,
            "Which selected frame should define the fixed reference states?",
            selected_frame_indices[-1],
            int,
            lambda value: value in selected_frame_indices,
            f"Enter one of these selected frame indices: "
            f"{selected_frame_indices}.",
        )

        # Expected physical edge count used to validate the merged convex-hull topology reconstructed from the shape JSON.
        expected_edges = resolve_or_prompt(
            args.edges,
            "Enter the intended number of polyhedron edges",
            CURRENT_SUGGESTED_NUM_EDGES,
            int,
            lambda value: value >= 1,
            "Enter a positive integer.",
        )
        # Expected physical face count; both edge and face counts must match.
        expected_faces = resolve_or_prompt(
            args.faces,
            "Enter the intended number of polyhedron faces",
            CURRENT_SUGGESTED_NUM_FACES,
            int,
            lambda value: value >= 1,
            "Enter a positive integer.",
        )
        # Symmetry precision exponent p defines the geometric vertex-matching tolerance 10^(-p) in shape-coordinate units.
        precision_exponent = resolve_or_prompt(
            args.precision,
            "Enter invariant-quaternion symmetry precision exponent p",
            CURRENT_SUGGESTED_PRECISION,
            int,
            lambda value: value >= 0,
            "Enter a nonnegative integer.",
        )
        # Physical frame-level cluster diameter cutoff in degrees.
        #
        # Every final cluster is explicitly validated to satisfy:
        #
        #       max_{i,j in C} d_G(q_i,q_j)
        #       <= orientation_tolerance + tiny binary epsilon.
        orientation_tolerance = resolve_or_prompt(
            args.orientation_angle_tol,
            "Maximum allowed pairwise angle within a frame-level group (degrees)",
            CURRENT_SUGGESTED_ORIENTATION_TOL_DEG,
            float,
            lambda value: 0.0 < value <= 180.0,
            "Enter an angle greater than 0 and at most 180 degrees.",
        )

        # A fixed reference state is considered valid only if its particle count,
        # averaged over all selected frames with zeros for absence, is at least
        # this integer cutoff.
        cluster_size_cutoff = resolve_or_prompt(
            args.cluster_size_cutoff,
            "Minimum cluster population for stability, tracking, and mean-state validity",
            min(CURRENT_SUGGESTED_CLUSTER_SIZE_CUTOFF, selected_particle_count),
            int,
            lambda value: 1 <= value <= selected_particle_count,
            f"Enter an integer from 1 through {selected_particle_count}.",
        )
        # Number of additional complete-linkage runs per frame after random
        # particle-order permutations.  These test whether tie handling makes
        # the partition depend on arbitrary particle ordering.
        stability_trials = resolve_or_prompt(
            args.stability_trials,
            "How many random particle-order stability trials per frame?",
            CURRENT_SUGGESTED_STABILITY_TRIALS,
            int,
            lambda value: value >= 0,
            "Enter a nonnegative integer.",
        )

        # ===================================================================
        # STAGE 6 — READ THE SHAPE AND BUILD THE RIGOROUS PROPER-ROTATION GROUP
        # ===================================================================
        vertices = read_shape_vertices(shape_path)

        # This pipeline recentres the shape, validates the requested edge/face topology, discovers proper rotations through one-to-one vertex
        # permutations, refines them at full precision, verifies group closure and returns explicit q/-q quaternion representatives for freud.
        symmetry_result = detect_proper_rotational_symmetries(vertices, expected_edges, expected_faces, precision_exponent)

        # ===================================================================
        # STAGE 7 — CREATE OUTPUT AND TEMPORARY-WORK LOCATIONS
        # ===================================================================
        # Default output location is beside the trajectory.  --output-dir overrides it.
        output_directory = trajectory_path.parent if args.output_dir is None else Path(args.output_dir).expanduser().resolve()
        # Create the directory and any missing parents.  Existing directories are accepted.
        output_directory.mkdir(parents=True, exist_ok=True)

        # Build a descriptive common prefix encoding the trajectory stem, number of final frames, and selected particle count.
        output_prefix = output_directory / f"{trajectory_path.stem}_fixed_reference_orientation_states_last_{frames_to_analyse}_frames_particles_{selected_particle_count}"

        stability_diagnostics_file = output_prefix.with_name(
            output_prefix.name
            + "_particle_order_stability_medoid_diagnostics.csv"
        )
        # Start every run with a fresh diagnostics file rather than appending to
        # a previous run that used the same output prefix.
        try:
            stability_diagnostics_file.unlink()
        except FileNotFoundError:
            pass

        # Condensed pair-distance files can be retained for inspection or automatically deleted after each frame.
        if args.keep_distance_files:
            # Persistent subdirectory when the user explicitly asks to retain condensed distance files.
            temporary_directory = output_directory / (output_prefix.name + "_distance_files")
            temporary_directory.mkdir(parents=True, exist_ok=True)
            # The outer cleanup stage must not delete a user-requested retained directory.
            cleanup_temporary = False
        else:
            # Create a uniquely named working directory under the output directory.  tempfile prevents collisions between runs.
            temporary_directory = Path(tempfile.mkdtemp(prefix="orientation_distance_work_", dir=output_directory))
            # Mark the directory for recursive cleanup after frame clustering.
            cleanup_temporary = True

        # Complete FrameClustering objects will be appended in selected-frame order.
        frame_clusterings: list[FrameClustering] = []

        # ===================================================================
        # STAGE 8 — CLUSTER EACH SELECTED FRAME
        # ===================================================================
        try:
            print("\nFrame-level complete-linkage clustering")
            print("---------------------------------------")
            # sequence is a one-based progress counter; frame_index is the actual zero-based trajectory index.
            for sequence, frame_index in enumerate(selected_frame_indices, start=1):
                # cluster_one_frame performs:
                #
                #   * quaternion validation and normalisation;
                #   * condensed i<j distance calculation;
                #   * complete-linkage clustering;
                #   * cutoff-qualified random particle-order cluster-count trials;
                #   * exact cluster-diameter validation;
                #   * symmetry-aware medoid calculation.
                frame_result = cluster_one_frame(
                    freud,
                    trajectory,
                    frame_index,
                    selected_particle_count,
                    symmetry_result.equivalent_quaternions_wxyz,
                    orientation_tolerance,
                    cluster_size_cutoff,
                    args.block_size,
                    args.angle_validation_tol,
                    stability_trials,
                    args.random_seed,
                    temporary_directory,
                    stability_diagnostics_file,
                    args.keep_distance_files,
                )


                # Retain the complete validated frame result.
                frame_clusterings.append(frame_result)

                # Largest diameter among all clusters in this frame.  Each
                # individual diameter was already checked against the physical
                # orientation cutoff.
                max_diameter = max(
                    cluster.diameter_deg
                    for cluster in frame_result.clusters
                )
                if frame_result.cluster_count_stability_passed is None:
                    stability_text = "not-tested"
                elif frame_result.cluster_count_stability_passed:
                    stability_text = "passed"
                else:
                    stability_text = "failed-but-continued"

                qualified_count = sum(
                    cluster.size >= cluster_size_cutoff
                    for cluster in frame_result.clusters
                )
                print(
                    f"[{sequence}/{len(selected_frame_indices)}] frame "
                    f"{frame_index}: raw clusters={len(frame_result.clusters)}, "
                    f"clusters with size >= {cluster_size_cutoff}="
                    f"{qualified_count}, "
                    f"largest cluster={frame_result.clusters[0].size}, "
                    f"maximum validated diameter={max_diameter:.12g} deg, "
                    "cutoff-qualified-cluster-count-order-stability="
                    f"{stability_text}"
                )
        finally:
            if cleanup_temporary:
                shutil.rmtree(temporary_directory, ignore_errors=True)

        full_reference = {
            frame.frame_index: frame
            for frame in frame_clusterings
        }[reference_frame_index]
        reference = build_cutoff_qualified_reference(
            full_reference,
            cluster_size_cutoff,
        )

        print("\nReference-state size filtering")
        print("------------------------------")
        print(
            f"Reference-frame raw cluster count: "
            f"{len(full_reference.clusters)}"
        )
        print(
            f"Reference states retained with size >= "
            f"{cluster_size_cutoff}: {len(reference.clusters)}"
        )
        print(
            f"Reference-frame clusters below cutoff and treated as unmatched: "
            f"{len(full_reference.clusters) - len(reference.clusters)}"
        )

        reference_distance_matrix = reference_medoid_distance_matrix(
            freud,
            reference.clusters,
            symmetry_result.equivalent_quaternions_wxyz,
            args.angle_validation_tol,
        )
        suggested_tracking, safe_tracking_upper = tracking_cutoff_suggestion(
            orientation_tolerance,
            reference_distance_matrix,
            args.angle_validation_tol,
        )

        print("\nFixed-reference tracking tolerance")
        print("----------------------------------")
        if safe_tracking_upper is None:
            print("Only one reference state was detected.")
        else:
            print(
                "Minimum reference-medoid separation: "
                f"{2.0 * safe_tracking_upper:.12g} deg"
            )
            print(
                "To guarantee non-overlapping reference acceptance regions, "
                "use tracking tolerance strictly below "
                f"{safe_tracking_upper:.12g} deg."
            )

        tracking_tolerance = resolve_or_prompt(
            args.tracking_angle_tol,
            "Maximum medoid angle for matching a frame group to a fixed state",
            suggested_tracking,
            float,
            lambda value: 0.0 < value <= 180.0,
            "Enter an angle greater than 0 and at most 180 degrees.",
        )

        if (
            safe_tracking_upper is not None
            and tracking_tolerance
            >= safe_tracking_upper - args.angle_validation_tol
            and not args.allow_overlapping_tracking_regions
        ):
            raise RuntimeError(
                "The selected tracking tolerance permits overlapping reference "
                "acceptance regions. Choose a value strictly below half the "
                "minimum reference-medoid separation, or explicitly use "
                "--allow-overlapping-tracking-regions."
            )

        tracking_results, reference = track_all_frames_to_fixed_reference(
            freud,
            frame_clusterings,
            reference,
            cluster_size_cutoff,
            symmetry_result.equivalent_quaternions_wxyz,
            tracking_tolerance,
            args.angle_validation_tol,
            args.allow_ambiguous_tracking,
        )

        (
            state_summaries,
            state_counts,
            state_fractions,
            unmatched_counts,
        ) = summarize_fixed_reference_states(
            frame_clusterings,
            tracking_results,
            reference,
            cluster_size_cutoff,
        )

        symmetry_csv = save_symmetry_outputs(
            output_prefix,
            symmetry_result,
        )
        output_paths = save_cluster_and_tracking_outputs(
            output_prefix,
            frame_clusterings,
            tracking_results,
            reference,
            state_summaries,
            state_counts,
            state_fractions,
            unmatched_counts,
            selected_frame_indices,
            selected_particle_count,
            cluster_size_cutoff,
            orientation_tolerance,
            tracking_tolerance,
        )
        (
            bar_plot,
            time_plot,
            bar_plot_csv,
            time_plot_csv,
        ) = save_plots(
            plt,
            output_prefix,
            state_summaries,
            state_fractions,
            unmatched_counts,
            selected_frame_indices,
            selected_particle_count,
            not args.no_show,
        )
        metadata_path = save_analysis_metadata(
            output_prefix,
            trajectory_path,
            shape_path,
            total_frames,
            selected_frame_indices,
            selected_particle_count,
            reference_frame_index,
            expected_edges,
            expected_faces,
            precision_exponent,
            orientation_tolerance,
            tracking_tolerance,
            safe_tracking_upper,
            cluster_size_cutoff,
            args.block_size,
            stability_trials,
            args.random_seed,
            stability_diagnostics_file,
            args.angle_validation_tol,
            symmetry_result,
            frame_clusterings,
            tracking_results,
            state_summaries,
        )

        valid_states = [
            state
            for state in state_summaries
            if state.valid_by_mean_size_cutoff
        ]

        print("\nFinal fixed-reference state summary")
        print("-----------------------------------")
        print(f"Reference frame: {reference_frame_index}")
        print(
            f"Reference-frame raw cluster count: "
            f"{len(full_reference.clusters)}"
        )
        print(
            f"Tracked reference state count after instantaneous size cutoff: "
            f"{len(reference.clusters)}"
        )
        print(
            "Valid tracked states by mean particle-count cutoff "
            f"({cluster_size_cutoff}): {len(valid_states)}"
        )
        for state in state_summaries:
            print(
                f"State {state.reference_state_id}: "
                f"mean count={state.mean_particle_count:.6g}, "
                f"mean fraction={state.mean_population_fraction:.6g}, "
                f"presence={state.presence_fraction:.3f}, "
                f"max diameter={state.maximum_cluster_diameter_deg:.6g} deg, "
                f"valid={state.valid_by_mean_size_cutoff}"
            )

        print("\nSaved outputs")
        print("-------------")
        print(f"Symmetry quaternions: {symmetry_csv}")
        for label, path in output_paths.items():
            print(f"{label}: {path}")
        print(f"Mean-population plot: {bar_plot}")
        print(f"Mean-population plotting CSV: {bar_plot_csv}")
        print(f"Population time-series plot: {time_plot}")
        print(f"Population time-series plotting CSV: {time_plot_csv}")
        print(f"Metadata: {metadata_path}")
        if stability_trials > 0 and stability_diagnostics_file.exists():
            print(
                "Particle-order medoid diagnostics: "
                f"{stability_diagnostics_file}"
            )
        if args.keep_distance_files:
            print(f"Condensed distance files: {temporary_directory}")

        return 0

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


if __name__ == "__main__":
    raise SystemExit(main())
