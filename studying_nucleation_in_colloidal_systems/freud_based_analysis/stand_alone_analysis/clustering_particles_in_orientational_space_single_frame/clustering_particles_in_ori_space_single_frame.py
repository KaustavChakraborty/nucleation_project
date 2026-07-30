#!/usr/bin/env python3
"""
Rigorous standalone single-frame orientational-state clustering.

The program:
  1. reads one user-selected frame from a HOOMD GSD trajectory and a particle-
     shape JSON;
  2. reconstructs and validates the proper rotational symmetry group of the body;
  3. computes symmetry-reduced pairwise angular distances with freud for the
     selected particles in that frame;
  4. clusters the frame with complete-linkage hierarchical clustering;
  5. explicitly verifies that every returned cluster has maximum pairwise
     angular diameter <= the user-supplied orientation tolerance;
  6. represents each cluster by a symmetry-aware medoid, which is an actual
     particle orientation minimising the sum of within-cluster angular distances;
  7. identifies only clusters whose instantaneous particle count is greater than
     or equal to cluster_size_cutoff and assigns deterministic names A, B, C, ...;
  8. writes raw-cluster, identified-state, particle-membership, plotting-data,
     symmetry, and metadata outputs for that one frame.

There is no time averaging, particle-order permutation validation, reference
frame, inter-frame state tracking, or tracking-angle tolerance in this version.
No spatial positions or neighbour definitions are used. This is a global
orientation-only analysis of one frame.

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
CURRENT_SUGGESTED_ORIENTATION_TOL_DEG = 41.0
CURRENT_SUGGESTED_CLUSTER_SIZE_CUTOFF = 200

DEFAULT_BLOCK_SIZE = 128

# Baseline tolerance used when validating numerical properties of the
# symmetry-reduced angular-distance calculation. This is not the physical
# within-cluster angular cutoff.
DEFAULT_ANGLE_VALIDATION_TOL_DEG = 1.0e-5

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


def construct_frame_clusters(
    frame_index: int,
    orientations: np.ndarray,
    labels: np.ndarray,
    condensed_distances: np.ndarray,
    orientation_angle_tolerance_deg: float,
    max_quaternion_norm_deviation: float,
    distance_diagnostics: DistanceDiagnostics,
) -> FrameClustering:
    """Build, validate, and deterministically order all clusters in one frame."""

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
            medoid_tie_tolerance_deg=MEDOID_TIE_TOL_DEG,
        )

        # Retain the same strict scientific validation as the multi-frame code.
        if diameter > (
            orientation_angle_tolerance_deg
            + CLUSTER_DIAMETER_EPS_DEG
        ):
            raise RuntimeError(
                f"Frame {frame_index}: cluster generated by complete linkage has "
                f"diameter {diameter:.12g} deg, exceeding the requested "
                f"{orientation_angle_tolerance_deg:.12g} deg."
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

    # Preserve the existing deterministic cluster ordering.
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
            f"Frame {frame_index}: clustering did not assign every analysed particle."
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
    )

def cluster_one_frame(
    freud_module: Any,
    trajectory: Any,
    frame_index: int,
    requested_particles: int,
    equivalent_quaternions: np.ndarray,
    orientation_angle_tolerance_deg: float,
    block_size: int,
    angle_validation_tolerance_deg: float,
    temporary_directory: Path,
    keep_distance_file: bool,
) -> FrameClustering:
    """Run the unchanged distance, complete-linkage, diameter, and medoid workflow."""

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
        result = construct_frame_clusters(
            frame_index,
            orientations,
            labels,
            condensed_distances,
            orientation_angle_tolerance_deg,
            norm_deviation,
            diagnostics,
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


def identified_clusters_for_single_frame(
    frame: FrameClustering,
    cluster_size_cutoff: int,
) -> tuple[FrameCluster, ...]:
    """Return clusters whose instantaneous size meets the single-frame cutoff."""

    return tuple(
        cluster
        for cluster in frame.clusters
        if cluster.size >= cluster_size_cutoff
    )


def save_single_frame_outputs(
    output_prefix: Path,
    frame: FrameClustering,
    identified_clusters: Sequence[FrameCluster],
    selected_particle_count: int,
    cluster_size_cutoff: int,
    orientation_angle_tolerance_deg: float,
) -> dict[str, Path]:
    """Write raw and cutoff-qualified single-frame tables."""

    paths = {
        "identified_states": output_prefix.with_name(
            output_prefix.name + "_identified_states.csv"
        ),
        "all_clusters": output_prefix.with_name(
            output_prefix.name + "_all_frame_clusters.csv"
        ),
        "all_membership": output_prefix.with_name(
            output_prefix.name + "_all_analysed_particle_membership.csv"
        ),
        "identified_particle_states": output_prefix.with_name(
            output_prefix.name + "_identified_particle_state_membership.csv"
        ),
    }

    identified_local_to_state = {
        int(cluster.local_cluster_id): int(state_id)
        for state_id, cluster in enumerate(identified_clusters)
    }

    with paths["identified_states"].open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "state_id",
                "cluster_name",
                "local_cluster_id",
                "cluster_size",
                "population_fraction",
                "medoid_particle_index",
                "medoid_w",
                "medoid_x",
                "medoid_y",
                "medoid_z",
                "diameter_deg",
                "max_angle_to_medoid_deg",
                "medoid_sum_distance_deg",
                "diameter_within_requested_tolerance",
            ]
        )
        for state_id, cluster in enumerate(identified_clusters):
            q = cluster.medoid_quaternion_wxyz
            writer.writerow(
                [
                    int(frame.frame_index),
                    int(state_id),
                    alphabetic_state_name(state_id),
                    int(cluster.local_cluster_id),
                    int(cluster.size),
                    float(cluster.size / selected_particle_count),
                    int(cluster.medoid_particle_index),
                    float(q[0]),
                    float(q[1]),
                    float(q[2]),
                    float(q[3]),
                    float(cluster.diameter_deg),
                    float(cluster.maximum_angle_to_medoid_deg),
                    float(cluster.medoid_sum_distance_deg),
                    int(
                        cluster.diameter_deg
                        <= orientation_angle_tolerance_deg
                        + CLUSTER_DIAMETER_EPS_DEG
                    ),
                ]
            )

    with paths["all_clusters"].open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_index",
                "local_cluster_id",
                "cluster_size",
                "meets_cluster_size_cutoff",
                "identified_state_id",
                "cluster_name",
                "medoid_particle_index",
                "medoid_w",
                "medoid_x",
                "medoid_y",
                "medoid_z",
                "diameter_deg",
                "max_angle_to_medoid_deg",
                "medoid_sum_distance_deg",
            ]
        )
        for cluster in frame.clusters:
            state_id = identified_local_to_state.get(
                int(cluster.local_cluster_id), -1
            )
            q = cluster.medoid_quaternion_wxyz
            writer.writerow(
                [
                    int(frame.frame_index),
                    int(cluster.local_cluster_id),
                    int(cluster.size),
                    int(cluster.size >= cluster_size_cutoff),
                    int(state_id),
                    "" if state_id < 0 else alphabetic_state_name(state_id),
                    int(cluster.medoid_particle_index),
                    float(q[0]),
                    float(q[1]),
                    float(q[2]),
                    float(q[3]),
                    float(cluster.diameter_deg),
                    float(cluster.maximum_angle_to_medoid_deg),
                    float(cluster.medoid_sum_distance_deg),
                ]
            )

    with paths["all_membership"].open(
        "w", newline="", encoding="utf-8"
    ) as all_handle, paths["identified_particle_states"].open(
        "w", newline="", encoding="utf-8"
    ) as identified_handle:
        all_writer = csv.writer(all_handle)
        identified_writer = csv.writer(identified_handle)

        all_writer.writerow(
            [
                "frame_index",
                "particle_index",
                "local_cluster_id",
                "cluster_size",
                "meets_cluster_size_cutoff",
                "identified_state_id",
                "cluster_name",
            ]
        )
        identified_writer.writerow(
            [
                "frame_index",
                "particle_index",
                "state_id",
                "cluster_name",
            ]
        )

        for cluster in frame.clusters:
            state_id = identified_local_to_state.get(
                int(cluster.local_cluster_id), -1
            )
            state_name = (
                "" if state_id < 0 else alphabetic_state_name(state_id)
            )
            for particle in cluster.members:
                all_writer.writerow(
                    [
                        int(frame.frame_index),
                        int(particle),
                        int(cluster.local_cluster_id),
                        int(cluster.size),
                        int(cluster.size >= cluster_size_cutoff),
                        int(state_id),
                        state_name,
                    ]
                )
                if state_id >= 0:
                    identified_writer.writerow(
                        [
                            int(frame.frame_index),
                            int(particle),
                            int(state_id),
                            state_name,
                        ]
                    )

    return paths


def alphabetic_state_name(index: int) -> str:
    """Return 0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA, and so on."""

    if index < 0:
        raise ValueError("State index must be nonnegative.")
    value = index + 1
    name = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def save_single_frame_population_plot(
    plt_module: Any,
    output_prefix: Path,
    identified_clusters: Sequence[FrameCluster],
    selected_particle_count: int,
    cluster_size_cutoff: int,
    show_plot: bool,
) -> tuple[Path, Path]:
    """Save the identified-state population bar plot and exact plotting CSV."""

    plot_path = output_prefix.with_name(
        output_prefix.name + "_single_frame_state_populations.png"
    )
    csv_path = plot_path.with_name(
        "for_plotting_" + plot_path.stem + ".csv"
    )

    x_values = np.arange(len(identified_clusters), dtype=np.int64)
    counts = np.asarray(
        [cluster.size for cluster in identified_clusters],
        dtype=np.int64,
    )
    fractions = counts.astype(np.float64) / float(selected_particle_count)
    labels = [
        alphabetic_state_name(index)
        for index in range(len(identified_clusters))
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "x_bar_center",
                "state_id",
                "cluster_name",
                "cluster_size",
                "y_population_fraction",
            ]
        )
        for x, state_id, label, count, fraction in zip(
            x_values,
            range(len(identified_clusters)),
            labels,
            counts,
            fractions,
        ):
            writer.writerow(
                [int(x), int(state_id), label, int(count), float(fraction)]
            )

    figure, axis = plt_module.subplots(
        figsize=(max(5.0, 0.65 * max(1, len(identified_clusters))), 3.8)
    )
    if identified_clusters:
        axis.bar(x_values, fractions)
        axis.set_xticks(x_values, labels)
        axis.set_ylim(0.0, max(0.05, 1.12 * float(np.max(fractions))))
    else:
        axis.text(
            0.5,
            0.5,
            "No cluster meets the single-frame size cutoff "
            f"({cluster_size_cutoff}).",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_ylim(0.0, 1.0)

    axis.set_xlabel("Identified orientational state")
    axis.set_ylabel("Population fraction")
    axis.tick_params(direction="in")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=600, bbox_inches="tight")

    if show_plot:
        plt_module.show()
    else:
        plt_module.close(figure)

    return plot_path, csv_path


def save_single_frame_metadata(
    output_prefix: Path,
    trajectory_path: Path,
    shape_path: Path,
    total_frames: int,
    frame: FrameClustering,
    selected_particle_count: int,
    expected_edges: int,
    expected_faces: int,
    precision_exponent: int,
    orientation_angle_tolerance_deg: float,
    cluster_size_cutoff: int,
    block_size: int,
    angle_validation_tolerance_deg: float,
    symmetry_result: SymmetryResult,
    identified_clusters: Sequence[FrameCluster],
) -> Path:
    """Write a single-frame audit record without tracking or averaging fields."""

    path = output_prefix.with_name(output_prefix.name + "_metadata.json")
    identified_population = int(
        sum(cluster.size for cluster in identified_clusters)
    )
    metadata = {
        "analysis_type": "single-frame orientational-state clustering",
        "trajectory_file": str(trajectory_path),
        "shape_file": str(shape_path),
        "total_trajectory_frames": int(total_frames),
        "selected_frame_index": int(frame.frame_index),
        "selected_particle_count": int(selected_particle_count),
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
        "cluster_size_cutoff_definition": (
            "instantaneous minimum particle count for an identified "
            "single-frame orientational state"
        ),
        "cluster_size_cutoff": int(cluster_size_cutoff),
        "raw_cluster_count": int(len(frame.clusters)),
        "identified_cluster_count": int(len(identified_clusters)),
        "identified_particle_count": identified_population,
        "ignored_below_cutoff_particle_count": int(
            selected_particle_count - identified_population
        ),
        "maximum_quaternion_norm_deviation": float(
            frame.maximum_quaternion_norm_deviation
        ),
        "minimum_pair_angle_deg": float(
            frame.distance_diagnostics.minimum_angle_deg
        ),
        "maximum_pair_angle_deg": float(
            frame.distance_diagnostics.maximum_angle_deg
        ),
        "maximum_cluster_diameter_deg": float(
            max(cluster.diameter_deg for cluster in frame.clusters)
        ),
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
        "angle_validation_tolerance_deg": float(
            angle_validation_tolerance_deg
        ),
        "identified_states": [
            {
                "state_id": int(state_id),
                "cluster_name": alphabetic_state_name(state_id),
                "local_cluster_id": int(cluster.local_cluster_id),
                "cluster_size": int(cluster.size),
                "population_fraction": float(
                    cluster.size / selected_particle_count
                ),
                "medoid_particle_index": int(
                    cluster.medoid_particle_index
                ),
                "diameter_deg": float(cluster.diameter_deg),
            }
            for state_id, cluster in enumerate(identified_clusters)
        ],
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rigorous complete-linkage orientational-state clustering for one "
            "user-selected trajectory frame."
        )
    )
    parser.add_argument("trajectory", help="Input HOOMD GSD trajectory")
    parser.add_argument("shape", help="Particle shape JSON")
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Zero-based trajectory frame index to analyse.",
    )
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--edges", type=int, default=None)
    parser.add_argument("--faces", type=int, default=None)
    parser.add_argument("--precision", type=int, default=None)
    parser.add_argument("--orientation-angle-tol", type=float, default=None)
    parser.add_argument("--cluster-size-cutoff", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--angle-validation-tol",
        type=float,
        default=DEFAULT_ANGLE_VALIDATION_TOL_DEG,
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--keep-distance-files", action="store_true")
    return parser

def main() -> int:
    args = build_argument_parser().parse_args()

    try:
        if args.block_size < 1:
            raise ValueError("--block-size must be positive.")
        if args.angle_validation_tol <= 0.0:
            raise ValueError("--angle-validation-tol must be positive.")

        freud, gsd_hoomd, plt = import_runtime_packages()
        trajectory_path, trajectory = open_gsd_trajectory(
            gsd_hoomd,
            args.trajectory,
        )
        shape_path = Path(args.shape).expanduser().resolve()

        total_frames = len(trajectory)
        if total_frames < 1:
            raise ValueError("The trajectory contains no frames.")

        print("\nTrajectory summary")
        print("------------------")
        print(f"Trajectory: {trajectory_path}")
        print(f"Number of available frames: {total_frames}")
        print(f"Valid frame indices: 0 through {total_frames - 1}")

        frame_index = resolve_or_prompt(
            args.frame,
            "Which trajectory frame should be analysed?",
            total_frames - 1,
            int,
            lambda value: 0 <= value < total_frames,
            f"Enter an integer from 0 through {total_frames - 1}.",
        )
        print(f"Selected frame index: {frame_index}")

        orientation_array = np.asarray(
            trajectory[frame_index].particles.orientation
        )
        if orientation_array.ndim != 2 or orientation_array.shape[1] != 4:
            raise ValueError(
                f"Frame {frame_index}: invalid orientation shape "
                f"{orientation_array.shape}."
            )
        available_particles = len(orientation_array)
        print(f"Particles available in selected frame: {available_particles}")

        selected_particle_count = resolve_or_prompt(
            args.particles,
            "How many particles should be analysed?",
            available_particles,
            int,
            lambda value: 2 <= value <= available_particles,
            f"Enter an integer from 2 through {available_particles}.",
        )
        expected_edges = resolve_or_prompt(
            args.edges,
            "Enter the intended number of polyhedron edges",
            CURRENT_SUGGESTED_NUM_EDGES,
            int,
            lambda value: value >= 1,
            "Enter a positive integer.",
        )
        expected_faces = resolve_or_prompt(
            args.faces,
            "Enter the intended number of polyhedron faces",
            CURRENT_SUGGESTED_NUM_FACES,
            int,
            lambda value: value >= 1,
            "Enter a positive integer.",
        )
        precision_exponent = resolve_or_prompt(
            args.precision,
            "Enter invariant-quaternion symmetry precision exponent p",
            CURRENT_SUGGESTED_PRECISION,
            int,
            lambda value: value >= 0,
            "Enter a nonnegative integer.",
        )
        orientation_tolerance = resolve_or_prompt(
            args.orientation_angle_tol,
            "Maximum allowed pairwise angle within a cluster (degrees)",
            CURRENT_SUGGESTED_ORIENTATION_TOL_DEG,
            float,
            lambda value: 0.0 < value <= 180.0,
            "Enter an angle greater than 0 and at most 180 degrees.",
        )
        cluster_size_cutoff = resolve_or_prompt(
            args.cluster_size_cutoff,
            "Minimum particle count for an identified single-frame state",
            min(CURRENT_SUGGESTED_CLUSTER_SIZE_CUTOFF, selected_particle_count),
            int,
            lambda value: 1 <= value <= selected_particle_count,
            f"Enter an integer from 1 through {selected_particle_count}.",
        )

        vertices = read_shape_vertices(shape_path)
        symmetry_result = detect_proper_rotational_symmetries(
            vertices,
            expected_edges,
            expected_faces,
            precision_exponent,
        )

        output_directory = (
            trajectory_path.parent
            if args.output_dir is None
            else Path(args.output_dir).expanduser().resolve()
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        output_prefix = output_directory / (
            f"{trajectory_path.stem}_single_frame_orientation_states_"
            f"frame_{frame_index}_particles_{selected_particle_count}"
        )

        if args.keep_distance_files:
            temporary_directory = output_directory / (
                output_prefix.name + "_distance_files"
            )
            temporary_directory.mkdir(parents=True, exist_ok=True)
            cleanup_temporary = False
        else:
            temporary_directory = Path(
                tempfile.mkdtemp(
                    prefix="orientation_distance_work_",
                    dir=output_directory,
                )
            )
            cleanup_temporary = True

        try:
            print("\nSingle-frame complete-linkage clustering")
            print("----------------------------------------")
            frame_result = cluster_one_frame(
                freud,
                trajectory,
                frame_index,
                selected_particle_count,
                symmetry_result.equivalent_quaternions_wxyz,
                orientation_tolerance,
                args.block_size,
                args.angle_validation_tol,
                temporary_directory,
                args.keep_distance_files,
            )
        finally:
            if cleanup_temporary:
                shutil.rmtree(temporary_directory, ignore_errors=True)

        identified_clusters = identified_clusters_for_single_frame(
            frame_result,
            cluster_size_cutoff,
        )
        identified_population = sum(
            cluster.size for cluster in identified_clusters
        )
        ignored_population = selected_particle_count - identified_population
        max_diameter = max(
            cluster.diameter_deg for cluster in frame_result.clusters
        )

        print(
            f"Frame {frame_index}: raw clusters={len(frame_result.clusters)}, "
            f"identified clusters with size >= {cluster_size_cutoff}="
            f"{len(identified_clusters)}, largest cluster="
            f"{frame_result.clusters[0].size}, maximum validated diameter="
            f"{max_diameter:.12g} deg."
        )

        symmetry_csv = save_symmetry_outputs(output_prefix, symmetry_result)
        output_paths = save_single_frame_outputs(
            output_prefix,
            frame_result,
            identified_clusters,
            selected_particle_count,
            cluster_size_cutoff,
            orientation_tolerance,
        )
        population_plot, population_plot_csv = (
            save_single_frame_population_plot(
                plt,
                output_prefix,
                identified_clusters,
                selected_particle_count,
                cluster_size_cutoff,
                not args.no_show,
            )
        )
        metadata_path = save_single_frame_metadata(
            output_prefix,
            trajectory_path,
            shape_path,
            total_frames,
            frame_result,
            selected_particle_count,
            expected_edges,
            expected_faces,
            precision_exponent,
            orientation_tolerance,
            cluster_size_cutoff,
            args.block_size,
            args.angle_validation_tol,
            symmetry_result,
            identified_clusters,
        )

        print("\nFinal single-frame state summary")
        print("--------------------------------")
        print(f"Analysed frame: {frame_index}")
        print(f"Analysed particles: {selected_particle_count}")
        print(f"Raw complete-linkage clusters: {len(frame_result.clusters)}")
        print(
            f"Identified clusters with size >= {cluster_size_cutoff}: "
            f"{len(identified_clusters)}"
        )
        print(f"Particles in identified states: {identified_population}")
        print(f"Particles ignored below cutoff: {ignored_population}")

        for state_id, cluster in enumerate(identified_clusters):
            print(
                f"State {alphabetic_state_name(state_id)}: "
                f"size={cluster.size}, fraction="
                f"{cluster.size / selected_particle_count:.6g}, "
                f"medoid particle={cluster.medoid_particle_index}, "
                f"diameter={cluster.diameter_deg:.6g} deg"
            )

        print("\nSaved outputs")
        print("-------------")
        print(f"Symmetry quaternions: {symmetry_csv}")
        for label, path in output_paths.items():
            print(f"{label}: {path}")
        print(f"Single-frame population plot: {population_plot}")
        print(f"Single-frame plotting CSV: {population_plot_csv}")
        print(f"Metadata: {metadata_path}")
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
