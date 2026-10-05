#!/usr/bin/env python3
"""
render_unit_cell_pyvista.py
================================
 Run command-
python3.8 render_unit_cell_pyvista_v1p6.py uc_info.txt shape_20_added_hexagonal_prism_both_side_hparam_0p2_eparam_0p7_unit_volume_principal_frame.json --cell-scale 1.20 --particle-scale 0.75 --cell-line-width 48.0 --nviews 2 --random-seed 123 --expected-num-faces 20 --expected-num-edges 42

Publication-quality PyVista renderer for a unit cell populated by oriented
convex polyhedral particles.

THIS VERSION FIXES THE POLYHEDRON CONSTRUCTION
----------------------------------------------
The particle geometry is now built directly and robustly from
``scipy.spatial.ConvexHull``:

    JSON vertices
        -> scipy.spatial.ConvexHull
        -> hull.simplices used DIRECTLY as the surface triangles
        -> triangle winding corrected using hull.equations outward normals
        -> coplanar triangle planes clustered ONLY for topology diagnostics
        -> true physical edges extracted from triangle adjacency

Crucially, the code does NOT reconstruct a polygon by angle-sorting its
vertices and then fan-triangulating it. That earlier strategy can be fragile.
The visible surface here is exactly the convex hull returned by SciPy/Qhull.

The script is now GENERAL with respect to topology validation.

For any supplied shape, the program always computes and prints:

    * number of JSON vertices
    * number of hull vertices
    * number of hull triangles
    * number of physical polygonal faces
    * number of true physical edges
    * Euler characteristic V - E + F
    * convex-hull volume and surface area

If you want strict validation for a particular polyhedron, pass the expected
counts at run time, for example

    --expected-num-faces 20 --expected-num-edges 42

If you do not provide these flags, the code remains fully general and simply
reports the topology it found.

OTHER FEATURES RETAINED
-----------------------
* particle color: #0080ff
* independently tunable unit-cell and particle linear scale factors
* broad unit-cell skeleton lines
* 10 random viewing angles by default
* EVERY viewing angle is saved twice: once on white background and once with a transparent background
* reproducible camera views via --random-seed
* scalar-first [w,x,y,z] quaternions by default
* orthographic projection by default
* high-resolution off-screen rendering

SCALING
-------
The unit-cell scaling is isotropic about the detected unit-cell origin O:

    a' = s_cell a
    b' = s_cell b
    c' = s_cell c

and each particle center is moved consistently as

    r_i' = O + s_cell (r_i - O).

The particle itself is independently scaled in its body/principal frame:

    v_body' = s_particle v_body.

Thus --cell-scale changes particle-center separations while
--particle-scale changes the particle size.

EXAMPLE
-------
python render_unit_cell_pyvista_v1p3.py \
    uc_info.txt \
    shape.json \
    -o unit_cell.png \
    --origin-id 82 \
    --cell-scale 1.20 \
    --particle-scale 0.75 \
    --cell-line-width 8 \
    --nviews 10 \
    --random-seed 123 \
    --expected-num-faces 20 \
    --expected-num-edges 42

OUTPUT PAIRS
------------
For each camera angle, two PNGs are always written.

For example, with ``--nviews 2`` and ``-o unit_cell.png``:

    unit_cell_view01.png
    unit_cell_view01_transparent.png
    unit_cell_view02.png
    unit_cell_view02_transparent.png

The first image in each pair has the normal white background. The second has
the identical camera, geometry, lighting, and resolution, but an alpha-channel
transparent background.

DEPENDENCIES
------------
pip install numpy scipy pyvista vtk
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull


# =============================================================================
# USER-TUNABLE DEFAULTS
# =============================================================================

PARTICLE_COLOR = "#0080ff"
PARTICLE_EDGE_COLOR = "#17324d"
CELL_EDGE_COLOR = "#202020"
BACKGROUND_COLOR = "white"

QUATERNION_ORDER = "wxyz"

DEFAULT_IMAGE_SIZE = (3000, 3000)
DEFAULT_PARTICLE_SCALE = 1.0
DEFAULT_CELL_SCALE = 1.0
DEFAULT_CELL_LINE_WIDTH = 8.0
DEFAULT_N_RANDOM_VIEWS = 10
DEFAULT_RANDOM_SEED = 0
DEFAULT_ZOOM = 1.12

# Expected topology supplied for this convex polyhedron.
DEFAULT_EXPECTED_NUM_FACES = None
DEFAULT_EXPECTED_NUM_EDGES = None

# Numerical tolerances for deciding whether two hull triangles lie on the same
# physical plane. scipy ConvexHull equations are already very accurate; these
# values simply protect against machine-level roundoff.
PLANE_NORMAL_TOL = 1.0e-9
PLANE_OFFSET_TOL = 1.0e-9


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Particle:
    """One particle entry parsed from the unit-cell text file."""

    particle_id: int
    position: np.ndarray
    quaternion: np.ndarray
    tag: str = ""


@dataclass
class UnitCellInfo:
    """Parsed unit-cell data required by the renderer."""

    particles: list[Particle]
    lattice_vectors: np.ndarray
    lattice_parameters: Optional[np.ndarray] = None
    crystal_class: Optional[str] = None
    spacegroup: Optional[str] = None
    number_effective_particles: Optional[float] = None
    local_to_global_quaternion: Optional[np.ndarray] = None


@dataclass
class ConvexParticleGeometry:
    """
    Topological and geometrical representation of the convex particle.

    Attributes
    ----------
    vertices
        Original JSON vertices. No vertex reordering is performed.
    triangles
        Outward-oriented triangular hull facets returned by ConvexHull.
    true_edges
        Physical polyhedron edges. Internal diagonals between coplanar hull
        triangles are deliberately excluded.
    triangle_plane_ids
        For every hull triangle, integer ID of the physical coplanar face.
    num_physical_faces
        Number of unique supporting planes, i.e. polygonal faces.
    face_vertex_counts
        Number of unique vertices belonging to each physical polygonal face.
    hull_volume
        Volume calculated by scipy.spatial.ConvexHull.
    hull_area
        Surface area calculated by scipy.spatial.ConvexHull.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    true_edges: np.ndarray
    triangle_plane_ids: np.ndarray
    num_physical_faces: int
    face_vertex_counts: list[int]
    hull_volume: float
    hull_area: float


# =============================================================================
# BASIC NUMERICAL HELPERS
# =============================================================================


def normalize(vector: np.ndarray, eps: float = 1.0e-15) -> np.ndarray:
    """Return ``vector`` normalized to unit length."""
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("Cannot normalize a nearly zero vector.")
    return vector / norm



def parse_numeric_vector(text: str, expected_size: Optional[int] = None) -> np.ndarray:
    """Parse a comma/space separated numeric vector from text."""
    values = np.fromstring(text.replace(",", " "), sep=" ", dtype=float)
    if expected_size is not None and values.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} numerical values but found "
            f"{values.size}: {text!r}"
        )
    return values


# =============================================================================
# UNIT-CELL TEXT PARSER
# =============================================================================


def read_unit_cell_info(path: str | Path) -> UnitCellInfo:
    """
    Read particle positions/orientations and lattice information from the text.

    The parser is label-based rather than line-number based, so harmless extra
    text does not normally break it.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    particle_pattern = re.compile(
        r"ID:\s*(?P<id>\d+)"
        r"\s*(?:\((?P<tag>[^)]*)\))?"
        r".*?\|\s*Position:\s*\[(?P<pos>[^\]]+)\]"
        r".*?\|\s*Orientation:\s*\[(?P<quat>[^\]]+)\]",
        flags=re.IGNORECASE,
    )

    particles: list[Particle] = []

    for match in particle_pattern.finditer(text):
        particles.append(
            Particle(
                particle_id=int(match.group("id")),
                position=parse_numeric_vector(
                    match.group("pos"), expected_size=3
                ),
                quaternion=parse_numeric_vector(
                    match.group("quat"), expected_size=4
                ),
                tag=(match.group("tag") or "").strip(),
            )
        )

    if not particles:
        raise ValueError(
            f"Could not parse any particle positions/orientations from {path}"
        )

    lattice_match = re.search(
        r"Lattice\s+vectors\s*:\s*(\[\[.*?\]\])",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if lattice_match is None:
        raise ValueError(
            "Could not locate 'Lattice vectors : [[...], [...], [...]]'."
        )

    lattice_vectors = np.asarray(
        ast.literal_eval(lattice_match.group(1)), dtype=float
    )

    if lattice_vectors.shape != (3, 3):
        raise ValueError(
            "Lattice vectors must form a 3 x 3 matrix; "
            f"parsed shape is {lattice_vectors.shape}."
        )

    lattice_parameters = None
    crystal_class = None

    match = re.search(
        r"Lattice\s+parameters\s*:\s*\[([^\]]+)\]"
        r"\s*and\s*Crystal\s+class\s*:\s*([^\n\r]+)",
        text,
        flags=re.IGNORECASE,
    )
    if match is not None:
        lattice_parameters = parse_numeric_vector(match.group(1))
        crystal_class = match.group(2).strip()

    spacegroup = None
    match = re.search(
        r"Spacegroup\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE
    )
    if match is not None:
        spacegroup = match.group(1).strip()

    number_effective_particles = None
    match = re.search(
        r"Number\s+of\s+effective\s+particles\s*:\s*([0-9eE+\-.]+)",
        text,
        flags=re.IGNORECASE,
    )
    if match is not None:
        number_effective_particles = float(match.group(1))

    local_to_global_quaternion = None
    match = re.search(
        r"Quaternion\s+required\s+to\s+transform\s+local\s+to\s+global\s+frame"
        r"\s*:\s*\[([^\]]+)\]",
        text,
        flags=re.IGNORECASE,
    )
    if match is not None:
        local_to_global_quaternion = parse_numeric_vector(
            match.group(1), expected_size=4
        )

    return UnitCellInfo(
        particles=particles,
        lattice_vectors=lattice_vectors,
        lattice_parameters=lattice_parameters,
        crystal_class=crystal_class,
        spacegroup=spacegroup,
        number_effective_particles=number_effective_particles,
        local_to_global_quaternion=local_to_global_quaternion,
    )


# =============================================================================
# SHAPE JSON PARSER
# =============================================================================


def read_shape_vertices(path: str | Path) -> np.ndarray:
    """
    Read the particle vertex list from the supplied JSON file.

    The first key whose name ends with ``vertices`` is used. The function does
    not generate, replace, sort, or otherwise alter the shape vertices.
    """
    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    vertex_key = None
    for key in data:
        if str(key).lower().endswith("vertices"):
            vertex_key = key
            break

    if vertex_key is None:
        raise KeyError(
            f"No JSON key ending in 'vertices' was found in {path}."
        )

    vertices = np.asarray(data[vertex_key], dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            "The JSON vertex array must have shape (N,3); "
            f"found {vertices.shape}."
        )

    if len(vertices) < 4:
        raise ValueError("At least four non-coplanar vertices are required.")

    if not np.all(np.isfinite(vertices)):
        raise ValueError("The JSON vertex list contains NaN/Inf values.")

    # Check exact/near duplicate points because duplicates can obscure topology
    # diagnostics and should be fixed in the shape definition rather than hidden.
    distances = np.linalg.norm(
        vertices[:, None, :] - vertices[None, :, :], axis=2
    )
    np.fill_diagonal(distances, np.inf)
    min_pair_distance = float(np.min(distances))
    if min_pair_distance < 1.0e-12:
        raise ValueError(
            "The shape JSON contains duplicate or numerically identical vertices."
        )

    return vertices


# =============================================================================
# SCIPY CONVEX-HULL TOPOLOGY
# =============================================================================


def planes_are_same(
    n1: np.ndarray,
    d1: float,
    n2: np.ndarray,
    d2: float,
    normal_tol: float = PLANE_NORMAL_TOL,
    offset_tol: float = PLANE_OFFSET_TOL,
) -> bool:
    """
    Return True when two normalized plane equations represent the same plane.

    SciPy/Qhull supplies hull equations in the form

        n.x + d = 0

    with outward normals. The sign-inverted equation describes the same plane,
    so both sign possibilities are accepted for robustness.
    """
    same_sign = (
        np.linalg.norm(n1 - n2) <= normal_tol
        and abs(d1 - d2) <= offset_tol
    )

    opposite_sign = (
        np.linalg.norm(n1 + n2) <= normal_tol
        and abs(d1 + d2) <= offset_tol
    )

    return bool(same_sign or opposite_sign)



def group_hull_triangles_into_physical_faces(
    hull: ConvexHull,
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Cluster coplanar triangular hull facets into physical polygonal faces.

    Important
    ---------
    These groups are used ONLY for topology validation and true-edge detection.
    They are NOT used to reconstruct or triangulate the rendered surface.

    The rendered surface remains ``ConvexHull.simplices`` directly.
    """
    plane_representatives: list[tuple[np.ndarray, float]] = []
    face_triangle_indices: list[list[int]] = []
    triangle_plane_ids = np.empty(len(hull.simplices), dtype=np.int64)

    for triangle_index, equation in enumerate(hull.equations):
        normal = np.asarray(equation[:3], dtype=float)
        offset = float(equation[3])

        norm = np.linalg.norm(normal)
        if norm < 1.0e-15:
            raise RuntimeError("Qhull returned a facet with zero normal.")

        normal = normal / norm
        offset = offset / norm

        assigned_face = None

        for face_id, (rep_normal, rep_offset) in enumerate(
            plane_representatives
        ):
            if planes_are_same(
                normal,
                offset,
                rep_normal,
                rep_offset,
            ):
                assigned_face = face_id
                break

        if assigned_face is None:
            assigned_face = len(plane_representatives)
            plane_representatives.append((normal.copy(), offset))
            face_triangle_indices.append([])

        triangle_plane_ids[triangle_index] = assigned_face
        face_triangle_indices[assigned_face].append(triangle_index)

    return triangle_plane_ids, face_triangle_indices



def orient_hull_triangles_outward(
    vertices: np.ndarray,
    hull: ConvexHull,
) -> np.ndarray:
    """
    Return ConvexHull simplices with a consistent outward winding.

    ``ConvexHull.simplices`` gives the correct triangular facets, but relying on
    their vertex order for rendering normals is unnecessary. Qhull also gives
    an outward plane normal for every simplex in ``hull.equations``. We compare
    each triangle's cross-product normal to that Qhull normal and swap the last
    two indices if needed.
    """
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(hull.simplices, dtype=np.int64).copy()

    for i, (triangle, equation) in enumerate(
        zip(triangles, hull.equations)
    ):
        p0, p1, p2 = vertices[triangle]
        triangle_normal = np.cross(p1 - p0, p2 - p0)
        outward_normal = equation[:3]

        if np.dot(triangle_normal, outward_normal) < 0.0:
            triangles[i, 1], triangles[i, 2] = (
                triangles[i, 2],
                triangles[i, 1],
            )

    return triangles



def extract_true_polyhedron_edges(
    triangles: np.ndarray,
    triangle_plane_ids: np.ndarray,
) -> np.ndarray:
    """
    Extract true physical edges from triangulated convex-hull topology.

    ConvexHull triangulates polygonal faces. Therefore the raw triangle mesh has
    two kinds of edges:

      * true polyhedron edges, shared by triangles on DIFFERENT face planes;
      * triangulation diagonals, shared by triangles on the SAME face plane.

    This function removes the latter exactly from adjacency/topology rather than
    relying on a VTK feature-angle heuristic.
    """
    edge_to_triangles: dict[tuple[int, int], list[int]] = defaultdict(list)

    for triangle_index, triangle in enumerate(triangles):
        i, j, k = map(int, triangle)
        for u, v in ((i, j), (j, k), (k, i)):
            edge = (u, v) if u < v else (v, u)
            edge_to_triangles[edge].append(triangle_index)

    true_edges: list[tuple[int, int]] = []

    for edge, adjacent_triangles in edge_to_triangles.items():
        adjacent_face_ids = {
            int(triangle_plane_ids[t]) for t in adjacent_triangles
        }

        # A proper convex closed hull normally has exactly two adjacent
        # triangles around every triangulated mesh edge. If an edge occurs only
        # once, retain it rather than silently dropping a possible boundary.
        if len(adjacent_triangles) == 1 or len(adjacent_face_ids) >= 2:
            true_edges.append(edge)

    true_edges.sort()
    return np.asarray(true_edges, dtype=np.int64)



def physical_face_vertex_counts(
    hull: ConvexHull,
    face_triangle_indices: list[list[int]],
) -> list[int]:
    """Return the number of unique vertices belonging to each physical face."""
    counts = []

    for triangle_indices in face_triangle_indices:
        vertex_ids: set[int] = set()
        for triangle_index in triangle_indices:
            vertex_ids.update(
                int(v) for v in hull.simplices[triangle_index]
            )
        counts.append(len(vertex_ids))

    return counts



def build_convex_particle_geometry(
    vertices: np.ndarray,
    expected_num_faces: Optional[int] = DEFAULT_EXPECTED_NUM_FACES,
    expected_num_edges: Optional[int] = DEFAULT_EXPECTED_NUM_EDGES,
) -> ConvexParticleGeometry:
    """
    Build and validate the complete convex-particle geometry using SciPy/Qhull.

    No manual polygon triangulation occurs here. The rendered triangles are
    exactly the hull triangles returned by ``scipy.spatial.ConvexHull`` with
    only their winding corrected for consistent outward normals.
    """
    vertices = np.asarray(vertices, dtype=float)

    # "Qc" asks Qhull to retain coplanar information. We deliberately do NOT
    # use QJ, because QJ perturbs the input coordinates and is unnecessary for
    # this non-degenerate three-dimensional particle.
    hull = ConvexHull(vertices, qhull_options="Qc")

    # Every JSON point should be a hull vertex for the intended polyhedron.
    hull_vertex_ids = np.asarray(hull.vertices, dtype=np.int64)

    triangles = orient_hull_triangles_outward(vertices, hull)

    triangle_plane_ids, face_triangle_indices = (
        group_hull_triangles_into_physical_faces(hull)
    )

    true_edges = extract_true_polyhedron_edges(
        triangles=triangles,
        triangle_plane_ids=triangle_plane_ids,
    )

    face_vertex_counts = physical_face_vertex_counts(
        hull,
        face_triangle_indices,
    )

    num_vertices = len(hull_vertex_ids)
    num_edges = len(true_edges)
    num_faces = len(face_triangle_indices)
    euler_characteristic = num_vertices - num_edges + num_faces

    print("\n============================================================")
    print("Convex-hull topology validation")
    print("============================================================")
    print(f"JSON vertices                 : {len(vertices)}")
    print(f"Vertices used by convex hull  : {num_vertices}")
    print(f"Triangular Qhull facets       : {len(triangles)}")
    print(f"Physical polygonal faces      : {num_faces}")
    print(f"True physical edges           : {num_edges}")
    print(f"Euler V-E+F                   : {euler_characteristic}")
    print(f"Convex-hull volume            : {hull.volume:.12g}")
    print(f"Convex-hull surface area      : {hull.area:.12g}")

    distribution = Counter(face_vertex_counts)
    print("Physical-face size distribution:")
    for nvertices_on_face in sorted(distribution):
        print(
            f"  {distribution[nvertices_on_face]} face(s) with "
            f"{nvertices_on_face} vertices"
        )

    if expected_num_faces in (None, 0):
        print("Face-count validation         : disabled (no expected value supplied)")
    else:
        print(f"Face-count validation         : expected {expected_num_faces}")

    if expected_num_edges in (None, 0):
        print("Edge-count validation         : disabled (no expected value supplied)")
    else:
        print(f"Edge-count validation         : expected {expected_num_edges}")

    print("============================================================\n")

    # ---------------------------------------------------------------------
    # Strict validation.
    # ---------------------------------------------------------------------
    if len(hull_vertex_ids) != len(vertices):
        missing = sorted(set(range(len(vertices))) - set(hull_vertex_ids.tolist()))
        raise RuntimeError(
            "Not every JSON vertex belongs to the convex hull. "
            f"Interior/non-hull vertex indices: {missing}"
        )

    if expected_num_faces not in (None, 0) and num_faces != expected_num_faces:
        raise RuntimeError(
            "Convex-hull face-count validation failed: "
            f"expected {expected_num_faces}, obtained {num_faces}. "
            "Rendering has been stopped."
        )

    if expected_num_edges not in (None, 0) and num_edges != expected_num_edges:
        raise RuntimeError(
            "Convex-hull edge-count validation failed: "
            f"expected {expected_num_edges}, obtained {num_edges}. "
            "Rendering has been stopped."
        )

    if euler_characteristic != 2:
        raise RuntimeError(
            "Euler topology validation failed: expected V-E+F=2 for a closed "
            f"convex polyhedron, obtained {euler_characteristic}."
        )

    return ConvexParticleGeometry(
        vertices=vertices.copy(),
        triangles=triangles,
        true_edges=true_edges,
        triangle_plane_ids=triangle_plane_ids,
        num_physical_faces=num_faces,
        face_vertex_counts=face_vertex_counts,
        hull_volume=float(hull.volume),
        hull_area=float(hull.area),
    )


# =============================================================================
# PYVISTA MESH CONSTRUCTION FROM VERIFIED HULL TOPOLOGY
# =============================================================================


def make_pyvista_surface_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
) -> pv.PolyData:
    """
    Convert verified hull triangles to a PyVista surface mesh.

    No ``clean()`` call is used because the input JSON vertex indices are
    already valid and we want to preserve topology exactly.
    """
    points = np.asarray(points, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)

    vtk_faces = np.column_stack(
        [
            np.full(len(triangles), 3, dtype=np.int64),
            triangles,
        ]
    ).ravel()

    mesh = pv.PolyData(points.copy(), vtk_faces)

    # Cell normals respect the deliberately corrected outward triangle winding.
    mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=False,
        inplace=True,
    )

    return mesh



def make_pyvista_edge_mesh(
    points: np.ndarray,
    true_edges: np.ndarray,
) -> pv.PolyData:
    """Construct line cells for exactly the verified true physical edges."""
    points = np.asarray(points, dtype=float)
    true_edges = np.asarray(true_edges, dtype=np.int64)

    vtk_lines = np.column_stack(
        [
            np.full(len(true_edges), 2, dtype=np.int64),
            true_edges,
        ]
    ).ravel()

    mesh = pv.PolyData(points.copy())
    mesh.lines = vtk_lines
    return mesh


# =============================================================================
# QUATERNION ROTATION
# =============================================================================


def quaternion_to_rotation_matrix(
    quaternion: np.ndarray,
    order: str = QUATERNION_ORDER,
) -> np.ndarray:
    """Convert a normalized quaternion into a 3 x 3 active rotation matrix."""
    q = np.asarray(quaternion, dtype=float)

    if q.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {q.shape}")

    norm_q = np.linalg.norm(q)
    if norm_q < 1.0e-14:
        raise ValueError("Encountered a nearly zero quaternion.")

    q = q / norm_q

    if order.lower() == "wxyz":
        w, x, y, z = q
    elif order.lower() == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError("Quaternion order must be 'wxyz' or 'xyzw'.")

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )



def transform_particle_vertices(
    body_vertices: np.ndarray,
    quaternion: np.ndarray,
    displayed_center: np.ndarray,
    particle_scale: float,
    quaternion_order: str,
) -> np.ndarray:
    """Isotropically scale, rotate, then translate particle vertices."""
    rotation = quaternion_to_rotation_matrix(
        quaternion,
        order=quaternion_order,
    )

    scaled_vertices = (
        np.asarray(body_vertices, dtype=float) * float(particle_scale)
    )

    rotated_vertices = scaled_vertices @ rotation.T

    return rotated_vertices + np.asarray(displayed_center, dtype=float)


# =============================================================================
# UNIT-CELL ORIGIN AND SCALING
# =============================================================================


def generate_cell_corners(
    origin: np.ndarray,
    lattice_vectors: np.ndarray,
) -> np.ndarray:
    """Return the eight parallelepiped corners."""
    a, b, c = np.asarray(lattice_vectors, dtype=float)

    return np.asarray(
        [
            origin + ia * a + ib * b + ic * c
            for ia, ib, ic in product((0, 1), repeat=3)
        ],
        dtype=float,
    )



def infer_cell_origin(
    particles: list[Particle],
    lattice_vectors: np.ndarray,
) -> tuple[np.ndarray, int, float, float]:
    """
    Infer which listed particle position is the best unit-cell corner.

    The eight ideal corners generated from each candidate origin are matched to
    distinct listed positions using the Hungarian assignment algorithm.
    """
    positions = np.asarray(
        [particle.position for particle in particles], dtype=float
    )

    if len(positions) < 8:
        raise ValueError(
            "Automatic origin inference needs at least eight listed positions. "
            "Provide --origin-id explicitly otherwise."
        )

    best = None

    for candidate_index, candidate_origin in enumerate(positions):
        ideal_corners = generate_cell_corners(
            candidate_origin,
            lattice_vectors,
        )

        cost = np.linalg.norm(
            ideal_corners[:, None, :] - positions[None, :, :],
            axis=2,
        )

        corner_indices, particle_indices = linear_sum_assignment(cost)
        distances = cost[corner_indices, particle_indices]

        rms = float(np.sqrt(np.mean(distances ** 2)))
        max_residual = float(np.max(distances))

        if best is None or rms < best[0]:
            best = (
                rms,
                max_residual,
                candidate_index,
                candidate_origin.copy(),
            )

    assert best is not None
    rms, max_residual, index, origin = best

    return (
        origin,
        particles[index].particle_id,
        rms,
        max_residual,
    )



def get_cell_origin(
    info: UnitCellInfo,
    origin_id: Optional[int],
) -> tuple[np.ndarray, str]:
    """Return either an explicitly requested or automatically inferred origin."""
    if origin_id is not None:
        for particle in info.particles:
            if particle.particle_id == origin_id:
                return (
                    particle.position.copy(),
                    f"explicit particle ID {origin_id}",
                )

        raise ValueError(
            f"Particle ID {origin_id} requested by --origin-id was not found."
        )

    origin, pid, rms, max_residual = infer_cell_origin(
        info.particles,
        info.lattice_vectors,
    )

    return (
        origin,
        (
            f"automatically inferred from particle ID {pid}; "
            f"corner RMS residual={rms:.6g}, "
            f"max residual={max_residual:.6g}"
        ),
    )



def scale_position_about_origin(
    position: np.ndarray,
    origin: np.ndarray,
    cell_scale: float,
) -> np.ndarray:
    """Scale one particle center isotropically about the unit-cell origin."""
    return (
        np.asarray(origin, dtype=float)
        + float(cell_scale)
        * (
            np.asarray(position, dtype=float)
            - np.asarray(origin, dtype=float)
        )
    )



def build_cell_edge_mesh(
    origin: np.ndarray,
    lattice_vectors: np.ndarray,
) -> pv.PolyData:
    """Build the twelve unit-cell skeleton edges as VTK line cells."""
    labels = list(product((0, 1), repeat=3))
    a, b, c = np.asarray(lattice_vectors, dtype=float)

    points = np.asarray(
        [
            origin + ia * a + ib * b + ic * c
            for ia, ib, ic in labels
        ],
        dtype=float,
    )

    label_to_index = {
        label: index for index, label in enumerate(labels)
    }

    segments: list[tuple[int, int]] = []

    for index, label in enumerate(labels):
        for axis in range(3):
            if label[axis] == 0:
                neighbor = list(label)
                neighbor[axis] = 1
                neighbor_index = label_to_index[tuple(neighbor)]
                segments.append((index, neighbor_index))

    vtk_lines = np.asarray(
        [[2, i, j] for i, j in segments], dtype=np.int64
    ).ravel()

    mesh = pv.PolyData(points)
    mesh.lines = vtk_lines
    return mesh


# =============================================================================
# CAMERA GENERATION
# =============================================================================


def compute_cell_camera_basis(
    origin: np.ndarray,
    lattice_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create an orthonormal basis derived from the displayed cell geometry."""
    corners = generate_cell_corners(origin, lattice_vectors)
    center = np.mean(corners, axis=0)

    centered = corners - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    e1, e2, e3 = vh

    a, b, c = lattice_vectors

    if np.dot(e1, a) < 0:
        e1 = -e1
    if np.dot(e2, b) < 0:
        e2 = -e2

    e3 = normalize(np.cross(e1, e2))

    if np.dot(e3, c) < 0:
        e3 = -e3

    return center, e1, e2, e3



def generate_random_camera_triplets(
    origin: np.ndarray,
    lattice_vectors: np.ndarray,
    scene_points: np.ndarray,
    nviews: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate reproducible random oblique cameras around the unit cell."""
    center, e1, e2, e3 = compute_cell_camera_basis(
        origin,
        lattice_vectors,
    )

    radius = float(
        np.max(np.linalg.norm(scene_points - center, axis=1))
    )
    if radius <= 0.0:
        radius = 1.0

    rng = np.random.default_rng(seed)
    cameras = []

    for _ in range(int(nviews)):
        azimuth = rng.uniform(0.0, 2.0 * np.pi)
        elevation = np.deg2rad(rng.uniform(18.0, 62.0))

        direction = (
            math.cos(elevation) * math.cos(azimuth) * e1
            + math.cos(elevation) * math.sin(azimuth) * e2
            + math.sin(elevation) * e3
        )
        direction = normalize(direction)

        view_up = e3 - np.dot(e3, direction) * direction
        if np.linalg.norm(view_up) < 1.0e-10:
            view_up = e2 - np.dot(e2, direction) * direction
        view_up = normalize(view_up)

        camera_position = center + 4.0 * radius * direction

        cameras.append(
            (camera_position, center.copy(), view_up)
        )

    return cameras


# =============================================================================
# LIGHTING
# =============================================================================


def add_publication_lights(
    plotter: pv.Plotter,
    focal_point: np.ndarray,
    scene_radius: float,
) -> None:
    """Add a moderate three-light setup for faceted-polyhedron rendering."""
    try:
        plotter.remove_all_lights()
    except Exception:
        pass

    r = max(float(scene_radius), 1.0)

    specifications = [
        (
            focal_point + np.array([+3.5, +2.0, +4.0]) * r,
            0.80,
        ),
        (
            focal_point + np.array([-3.0, +1.5, +2.0]) * r,
            0.45,
        ),
        (
            focal_point + np.array([+0.5, -3.5, +3.0]) * r,
            0.35,
        ),
    ]

    for position, intensity in specifications:
        light = pv.Light(
            position=position,
            focal_point=focal_point,
            color="white",
            intensity=float(intensity),
            positional=True,
        )
        plotter.add_light(light)


# =============================================================================
# COMPLETE SCENE CONSTRUCTION
# =============================================================================


def create_plotter(
    width: int,
    height: int,
) -> pv.Plotter:
    """Create the high-resolution off-screen renderer."""
    plotter = pv.Plotter(
        off_screen=True,
        window_size=(int(width), int(height)),
    )

    plotter.set_background(BACKGROUND_COLOR)

    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        try:
            plotter.enable_anti_aliasing("msaa")
        except Exception:
            pass

    return plotter



def add_all_particles(
    plotter: pv.Plotter,
    info: UnitCellInfo,
    geometry: ConvexParticleGeometry,
    cell_origin: np.ndarray,
    cell_scale: float,
    particle_scale: float,
    quaternion_order: str,
    show_particle_edges: bool,
    show_labels: bool,
    particle_edge_width: float,
) -> np.ndarray:
    """
    Add all independently oriented particles using verified hull topology.

    The triangle connectivity and true-edge connectivity are NEVER recalculated
    after rotation. A rotation/translation preserves connectivity exactly.
    """
    all_points = []

    for particle in info.particles:
        displayed_center = scale_position_about_origin(
            particle.position,
            cell_origin,
            cell_scale,
        )

        transformed_vertices = transform_particle_vertices(
            body_vertices=geometry.vertices,
            quaternion=particle.quaternion,
            displayed_center=displayed_center,
            particle_scale=particle_scale,
            quaternion_order=quaternion_order,
        )

        all_points.append(transformed_vertices)

        surface_mesh = make_pyvista_surface_mesh(
            points=transformed_vertices,
            triangles=geometry.triangles,
        )

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

        if show_particle_edges:
            # Draw exactly the 42 physical edges for the supplied shape, not the
            # 66 edges of its triangulated hull mesh.
            edge_mesh = make_pyvista_edge_mesh(
                points=transformed_vertices,
                true_edges=geometry.true_edges,
            )

            plotter.add_mesh(
                edge_mesh,
                color=PARTICLE_EDGE_COLOR,
                line_width=float(particle_edge_width),
                lighting=False,
            )

        if show_labels:
            label_points = pv.PolyData(
                np.asarray([displayed_center], dtype=float)
            )
            label_points["labels"] = np.asarray(
                [str(particle.particle_id)]
            )

            plotter.add_point_labels(
                label_points,
                "labels",
                point_size=0,
                font_size=14,
                text_color="black",
                shape=None,
                always_visible=True,
            )

    return np.vstack(all_points)


# =============================================================================
# MAIN RENDERER
# =============================================================================


def render_unit_cell_multi_view(
    unit_cell_file: str | Path,
    shape_json_file: str | Path,
    output_png: str | Path,
    *,
    width: int = DEFAULT_IMAGE_SIZE[0],
    height: int = DEFAULT_IMAGE_SIZE[1],
    cell_scale: float = DEFAULT_CELL_SCALE,
    particle_scale: float = DEFAULT_PARTICLE_SCALE,
    cell_line_width: float = DEFAULT_CELL_LINE_WIDTH,
    particle_edge_width: float = 1.5,
    perspective: bool = False,
    show_particle_edges: bool = True,
    show_labels: bool = False,
    origin_id: Optional[int] = None,
    quaternion_order: str = QUATERNION_ORDER,
    zoom: float = DEFAULT_ZOOM,
    nviews: int = DEFAULT_N_RANDOM_VIEWS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    expected_num_faces: Optional[int] = DEFAULT_EXPECTED_NUM_FACES,
    expected_num_edges: Optional[int] = DEFAULT_EXPECTED_NUM_EDGES,
) -> list[Path]:
    """Build, validate, render, and save opaque+transparent image pairs for all requested viewing angles."""
    # ---------------------------------------------------------------------
    # Parse files.
    # ---------------------------------------------------------------------
    info = read_unit_cell_info(unit_cell_file)
    body_vertices = read_shape_vertices(shape_json_file)

    print(f"Read {len(info.particles)} particle records.")
    print("Lattice vectors (rows = a,b,c):")
    print(info.lattice_vectors)

    if info.spacegroup is not None:
        print(f"Space group: {info.spacegroup}")
    if info.crystal_class is not None:
        print(f"Crystal class: {info.crystal_class}")

    # ---------------------------------------------------------------------
    # Build + strictly validate particle topology BEFORE rendering anything.
    # ---------------------------------------------------------------------
    geometry = build_convex_particle_geometry(
        vertices=body_vertices,
        expected_num_faces=expected_num_faces,
        expected_num_edges=expected_num_edges,
    )

    # ---------------------------------------------------------------------
    # Determine original cell origin.
    # ---------------------------------------------------------------------
    cell_origin, origin_description = get_cell_origin(
        info,
        origin_id,
    )

    print(f"Cell origin: {cell_origin}")
    print(f"Origin selection: {origin_description}")

    scaled_lattice_vectors = (
        np.asarray(info.lattice_vectors, dtype=float) * float(cell_scale)
    )

    print(f"Unit-cell linear scale factor : {cell_scale}")
    print(f"Particle linear scale factor  : {particle_scale}")
    print(f"Unit-cell volume multiplier   : {cell_scale ** 3:.6f}")
    print(f"Particle volume multiplier    : {particle_scale ** 3:.6f}")
    print(
        "Relative particle/cell volume multiplier : "
        f"{(particle_scale / cell_scale) ** 3:.6f}"
    )

    # ---------------------------------------------------------------------
    # Construct scene once. Only the camera changes between output images.
    # ---------------------------------------------------------------------
    plotter = create_plotter(width=width, height=height)

    particle_points = add_all_particles(
        plotter=plotter,
        info=info,
        geometry=geometry,
        cell_origin=cell_origin,
        cell_scale=cell_scale,
        particle_scale=particle_scale,
        quaternion_order=quaternion_order,
        show_particle_edges=show_particle_edges,
        show_labels=show_labels,
        particle_edge_width=particle_edge_width,
    )

    cell_edge_mesh = build_cell_edge_mesh(
        origin=cell_origin,
        lattice_vectors=scaled_lattice_vectors,
    )

    plotter.add_mesh(
        cell_edge_mesh,
        color=CELL_EDGE_COLOR,
        line_width=float(cell_line_width),
        lighting=False,
    )

    cell_corners = generate_cell_corners(
        cell_origin,
        scaled_lattice_vectors,
    )

    scene_points = np.vstack(
        [particle_points, cell_corners]
    )

    scene_center = np.mean(cell_corners, axis=0)
    scene_radius = float(
        np.max(
            np.linalg.norm(
                scene_points - scene_center,
                axis=1,
            )
        )
    )
    if scene_radius <= 0:
        scene_radius = 1.0

    add_publication_lights(
        plotter,
        focal_point=scene_center,
        scene_radius=scene_radius,
    )

    if not perspective:
        plotter.enable_parallel_projection()

    cameras = generate_random_camera_triplets(
        origin=cell_origin,
        lattice_vectors=scaled_lattice_vectors,
        scene_points=scene_points,
        nviews=nviews,
        seed=random_seed,
    )

    # ---------------------------------------------------------------------
    # Save all views.
    # ---------------------------------------------------------------------
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    for view_number, (
        camera_position,
        focal_point,
        view_up,
    ) in enumerate(cameras, start=1):

        # Set the camera ONCE for this viewing angle. Both output images below
        # are therefore guaranteed to use exactly the same viewing direction,
        # focal point, zoom, geometry, and lighting.
        plotter.camera_position = [
            camera_position.tolist(),
            focal_point.tolist(),
            view_up.tolist(),
        ]

        plotter.reset_camera()
        plotter.camera.zoom(float(zoom))

        # -------------------------------------------------------------
        # Construct the normal/opaque filename for this view.
        # -------------------------------------------------------------
        if nviews == 1:
            opaque_output = output_png
        else:
            opaque_output = output_png.with_name(
                f"{output_png.stem}_view{view_number:02d}"
                f"{output_png.suffix}"
            )

        # The transparent partner uses exactly the same basename, with
        # "_transparent" inserted before the extension.
        transparent_output = opaque_output.with_name(
            f"{opaque_output.stem}_transparent{opaque_output.suffix}"
        )

        # -------------------------------------------------------------
        # 1. Normal white-background publication image.
        # -------------------------------------------------------------
        plotter.screenshot(
            str(opaque_output),
            transparent_background=False,
            return_img=False,
        )

        saved_paths.append(opaque_output)
        print(
            f"Saved opaque view {view_number:02d}/{nviews}: "
            f"{opaque_output.resolve()}"
        )

        # -------------------------------------------------------------
        # 2. Matching transparent-background image.
        #
        # Nothing except the background alpha treatment is changed between
        # these two calls. Thus the two files correspond pixel-for-pixel in
        # camera framing and rendered structure.
        # -------------------------------------------------------------
        plotter.screenshot(
            str(transparent_output),
            transparent_background=True,
            return_img=False,
        )

        saved_paths.append(transparent_output)
        print(
            f"Saved transparent view {view_number:02d}/{nviews}: "
            f"{transparent_output.resolve()}"
        )

    plotter.close()
    return saved_paths


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Render an oriented-particle unit cell using a strictly validated "
            "SciPy ConvexHull particle mesh."
        )
    )

    parser.add_argument(
        "unit_cell_file",
        type=Path,
        help="Text file containing particle positions/orientations and lattice vectors.",
    )

    parser.add_argument(
        "shape_json_file",
        type=Path,
        help="JSON file containing the particle vertices.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("unit_cell.png"),
        help=(
            "Output PNG base name. With multiple views, '_view01', "
            "'_view02', ... are appended."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_IMAGE_SIZE[0],
        help=f"Output image width. Default: {DEFAULT_IMAGE_SIZE[0]}",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_IMAGE_SIZE[1],
        help=f"Output image height. Default: {DEFAULT_IMAGE_SIZE[1]}",
    )

    parser.add_argument(
        "--cell-scale",
        type=float,
        default=DEFAULT_CELL_SCALE,
        help=(
            "Isotropic linear scaling of lattice vectors and relative particle "
            f"centers. Default: {DEFAULT_CELL_SCALE}"
        ),
    )

    parser.add_argument(
        "--particle-scale",
        type=float,
        default=DEFAULT_PARTICLE_SCALE,
        help=(
            "Isotropic linear scaling of each particle itself. "
            f"Default: {DEFAULT_PARTICLE_SCALE}"
        ),
    )

    parser.add_argument(
        "--cell-line-width",
        type=float,
        default=DEFAULT_CELL_LINE_WIDTH,
        help=(
            "Width of the unit-cell skeleton lines. "
            f"Default: {DEFAULT_CELL_LINE_WIDTH}"
        ),
    )

    parser.add_argument(
        "--particle-edge-width",
        type=float,
        default=1.5,
        help="Width of true particle-edge lines. Default: 1.5",
    )

    parser.add_argument(
        "--nviews",
        type=int,
        default=DEFAULT_N_RANDOM_VIEWS,
        help=(
            "Number of random camera views to save. "
            f"Default: {DEFAULT_N_RANDOM_VIEWS}"
        ),
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=(
            "Seed controlling the random viewing directions. "
            f"Default: {DEFAULT_RANDOM_SEED}"
        ),
    )

    parser.add_argument(
        "--expected-num-faces", "--num-faces",
        dest="expected_num_faces",
        type=int,
        default=DEFAULT_EXPECTED_NUM_FACES,
        help=(
            "Expected number of physical polygonal faces for validation. "
            "If omitted, no face-count validation is enforced. Use 0 to disable. "
            "Example: --expected-num-faces 20"
        ),
    )

    parser.add_argument(
        "--expected-num-edges", "--num-edges",
        dest="expected_num_edges",
        type=int,
        default=DEFAULT_EXPECTED_NUM_EDGES,
        help=(
            "Expected number of true physical edges for validation. "
            "If omitted, no edge-count validation is enforced. Use 0 to disable. "
            "Example: --expected-num-edges 42"
        ),
    )

    parser.add_argument(
        "--origin-id",
        type=int,
        default=None,
        help=(
            "Particle ID whose position is the unit-cell origin. If omitted, "
            "the origin is inferred automatically."
        ),
    )

    parser.add_argument(
        "--quaternion-order",
        choices=("wxyz", "xyzw"),
        default=QUATERNION_ORDER,
        help=f"Quaternion component ordering. Default: {QUATERNION_ORDER}",
    )

    parser.add_argument(
        "--zoom",
        type=float,
        default=DEFAULT_ZOOM,
        help=f"Camera zoom factor. Default: {DEFAULT_ZOOM}",
    )

    parser.add_argument(
        "--perspective",
        action="store_true",
        help="Use perspective rather than the default orthographic projection.",
    )

    parser.add_argument(
        "--no-particle-edges",
        action="store_true",
        help="Do not draw the true physical particle edges.",
    )

    parser.add_argument(
        "--labels",
        action="store_true",
        help="Show particle IDs at their displayed centers.",
    )

    return parser



def main() -> None:
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive.")

    if args.cell_scale <= 0:
        parser.error("--cell-scale must be positive.")

    if args.particle_scale <= 0:
        parser.error("--particle-scale must be positive.")

    if args.cell_line_width <= 0:
        parser.error("--cell-line-width must be positive.")

    if args.particle_edge_width <= 0:
        parser.error("--particle-edge-width must be positive.")

    if args.nviews <= 0:
        parser.error("--nviews must be a positive integer.")

    if args.expected_num_faces is not None and args.expected_num_faces < 0:
        parser.error("--expected-num-faces must be >= 0.")

    if args.expected_num_edges is not None and args.expected_num_edges < 0:
        parser.error("--expected-num-edges must be >= 0.")

    if args.zoom <= 0:
        parser.error("--zoom must be positive.")

    render_unit_cell_multi_view(
        unit_cell_file=args.unit_cell_file,
        shape_json_file=args.shape_json_file,
        output_png=args.output,
        width=args.width,
        height=args.height,
        cell_scale=args.cell_scale,
        particle_scale=args.particle_scale,
        cell_line_width=args.cell_line_width,
        particle_edge_width=args.particle_edge_width,
        perspective=args.perspective,
        show_particle_edges=not args.no_particle_edges,
        show_labels=args.labels,
        origin_id=args.origin_id,
        quaternion_order=args.quaternion_order,
        zoom=args.zoom,
        nviews=args.nviews,
        random_seed=args.random_seed,
        expected_num_faces=args.expected_num_faces,
        expected_num_edges=args.expected_num_edges,
    )


if __name__ == "__main__":
    main()
