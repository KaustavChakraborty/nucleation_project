#!/usr/bin/env python3
"""
High-quality PyVista visualizer for a selected polyhedral particle in a HOOMD GSD trajectory.
Includes rigorously detected C5/C2 rotational symmetry axes for the D5h EPD.

What this version does
----------------------
1. Reads the selected particle position + quaternion orientation from each GSD frame.
2. Reads the particle body-frame vertices from the GSD when available, otherwise from JSON.
3. Reconstructs the TRUE CLOSED POLYHEDRON surface from the vertices using scipy.spatial.ConvexHull.
4. Merges coplanar hull triangles back into polygonal faces, so rectangular/polygonal faces
   are displayed as real faces and do not acquire fake diagonal edges.
5. Rotates/translates the complete polyhedron for each selected frame.
6. Renders filled polygon faces + clean edges with high-quality anti-aliasing and lighting.
7. Detects the proper rotational subgroup from the body geometry and overlays the C5/C2 axes.
8. Optionally adds vertex markers, coordinate axes, a title, and an MP4 video.

Example
-------
python epd_pyvista_polyhedron_visualizer_v2p1_symmetry_axes.py \
    trajectory.gsd \
    shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json \
    300 \
    0 \
    --center \
    --resolution 2400 \
    --camera-angle iso \
    --create-video \
    --video-fps 10

Dependencies
------------
pip install numpy scipy matplotlib pyvista gsd imageio imageio-ffmpeg
"""





'''
python3.8 epd_pyvista_polyhedron_visualizer_v2p1_symmetry_axes.py hpmc_hard_Elongated_Pentagonal_Dipyramid_vol_1p00_4096_NPT_P60_0_traj.gsd shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json 300 0 --center --resolution 2400 --camera-angle iso --interactive-selection --create-video --video-fps 10  --transparent-background --color-mode solid --no-title --c5-axis-color "#D62728" --c2-axis-color "#2CA02C" --symmetry-axis-length-factor 1.6 --c5-axis-radius-factor 0.018 --c2-axis-radius-factor 0.012

'''

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import gsd.hoomd

try:
    import pyvista as pv
except ImportError as exc:
    raise SystemExit(
        "PyVista is required.\nInstall with:\n"
        "    pip install pyvista\n"
    ) from exc

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
except ImportError as exc:
    raise SystemExit(
        "SciPy is required.\nInstall with:\n"
        "    pip install scipy\n"
    ) from exc

try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except Exception:
    try:
        import imageio
        IMAGEIO_AVAILABLE = True
    except ImportError:
        IMAGEIO_AVAILABLE = False


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Render a complete polyhedral particle from selected HOOMD GSD frames "
            "using PyVista."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "gsd_file",
        type=str,
        help="HOOMD GSD trajectory file",
    )

    parser.add_argument(
        "shape_json",
        type=str,
        help="JSON file containing body-frame polyhedron vertices",
    )

    parser.add_argument(
        "n_frames",
        type=int,
        help="Take this many consecutive frames from the END of the GSD",
    )

    parser.add_argument(
        "particle",
        type=int,
        help="Particle tag if tags exist; otherwise particle array index",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="polyhedron_frames",
        help="Directory for rendered PNG files",
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=2400,
        help="Square image resolution in pixels",
    )

    parser.add_argument(
        "--camera-angle",
        choices=["iso", "front", "back", "side", "left", "top", "bottom"],
        default="iso",
        help="Viewing direction",
    )

    parser.add_argument(
        "--center",
        action="store_true",
        help=(
            "Render the selected particle centered at the origin. "
            "This removes translational motion but preserves rotation."
        ),
    )

    parser.add_argument(
        "--show-vertices",
        action="store_true",
        help="Draw small spheres at the true polyhedron vertices",
    )

    parser.add_argument(
        "--show-axes",
        action="store_true",
        help="Show a small orientation axes widget",
    )

    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Do not place frame information above the image",
    )

    parser.add_argument(
        "--transparent-background",
        action="store_true",
        help="Save PNG with transparent background",
    )

    parser.add_argument(
        "--color-mode",
        choices=["solid", "frame"],
        default="solid",
        help=(
            "solid = same face color in every frame; "
            "frame = color changes through viridis across selected frames"
        ),
    )

    parser.add_argument(
        "--face-color",
        type=str,
        default="#6FA8DC",
        help="Polyhedron face color when --color-mode solid",
    )

    parser.add_argument(
        "--edge-color",
        type=str,
        default="#202020",
        help="Physical polyhedron edge color",
    )

    parser.add_argument(
        "--background-color",
        type=str,
        default="white",
        help="Rendering background color",
    )

    parser.add_argument(
        "--edge-width",
        type=float,
        default=3.0,
        help="Rendered edge width",
    )

    parser.add_argument(
        "--opacity",
        type=float,
        default=1.0,
        help="Face opacity, between 0 and 1",
    )

    parser.add_argument(
        "--camera-distance-factor",
        type=float,
        default=5.0,
        help="Camera distance as a multiple of particle radius",
    )

    parser.add_argument(
        "--zoom",
        type=float,
        default=0.9,
        help="Final camera zoom factor",
    )

    # ---------------------------------------------------------------------
    # ROTATIONAL-SYMMETRY AXIS DISPLAY
    #
    # The EPD is user-identified as D5h.  Its proper rotational subgroup is
    # D5: one five-fold axis and five two-fold axes.  These axes are detected
    # from the actual body-frame vertex geometry rather than hard-coded to x/y/z.
    # ---------------------------------------------------------------------

    parser.add_argument(
        "--no-symmetry-axes",
        action="store_true",
        help="Disable the C5/C2 rotational-symmetry-axis overlay",
    )

    parser.add_argument(
        "--symmetry-precision-exponent",
        type=int,
        default=2,
        help=(
            "Geometry matching tolerance is 10^(-p) in shape-coordinate units. "
            "The default p=2 follows the supplied pairwise-angle symmetry code."
        ),
    )

    parser.add_argument(
        "--symmetry-axis-length-factor",
        type=float,
        default=1.35,
        help=(
            "Half-length of each displayed rotational axis as a multiple of "
            "the maximum body-frame vertex radius"
        ),
    )

    parser.add_argument(
        "--c5-axis-radius-factor",
        type=float,
        default=0.018,
        help="C5 tube radius as a fraction of the body radius",
    )

    parser.add_argument(
        "--c2-axis-radius-factor",
        type=float,
        default=0.012,
        help="C2 tube radius as a fraction of the body radius",
    )

    parser.add_argument(
        "--c5-axis-color",
        type=str,
        default="#D62728",
        help="Color of the unique C5 rotational axis",
    )

    parser.add_argument(
        "--c2-axis-color",
        type=str,
        default="#2CA02C",
        help="Color of the five C2 rotational axes",
    )

    parser.add_argument(
        "--symmetry-axis-opacity",
        type=float,
        default=1.0,
        help="Opacity of the rotational-symmetry-axis tubes",
    )

    parser.add_argument(
        "--show-symmetry-legend",
        action="store_true",
        help="Show a small C5/C2 color legend",
    )

    parser.add_argument(
        "--create-video",
        action="store_true",
        help="Create an MP4 from the rendered frames",
    )

    parser.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Frames per second for output video",
    )

    parser.add_argument(
        "--interactive-selection",
        action="store_true",
        help=(
            "Ask interactively whether some frames should be excluded. "
            "Without this option all of the chosen last n_frames are rendered."
        ),
    )

    parser.add_argument(
        "--exclude",
        type=str,
        default="",
        help=(
            "Comma-separated absolute GSD frame numbers/ranges to exclude, "
            "for example: 710,715-720,730"
        ),
    )

    return parser.parse_args()


# =============================================================================
# FRAME-SELECTION UTILITIES
# =============================================================================

def parse_frame_specification(text):
    """
    Parse strings such as:
        10
        10,12,15
        10-20
        10,20-25,40
    """
    values = set()

    if not text.strip():
        return values

    for token in text.split(","):
        token = token.strip()
        if not token:
            continue

        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left.strip())
            stop = int(right.strip())

            if stop < start:
                start, stop = stop, start

            values.update(range(start, stop + 1))
        else:
            values.add(int(token))

    return values


def ask_for_frame_exclusion(frame_indices):
    print("\n" + "=" * 78)
    print("FRAME SELECTION")
    print("=" * 78)
    print(f"Frames currently selected: {len(frame_indices)}")
    print(
        f"Absolute GSD frame range: "
        f"{int(frame_indices[0])} ... {int(frame_indices[-1])}"
    )

    response = input(
        "\nRender all of these frames? [Y/n]: "
    ).strip().lower()

    if response in ("", "y", "yes"):
        return np.asarray(frame_indices, dtype=int)

    exclude_text = input(
        "Absolute frames to exclude "
        "(e.g. 710,715-720,730): "
    ).strip()

    excluded = parse_frame_specification(exclude_text)

    filtered = np.asarray(
        [f for f in frame_indices if int(f) not in excluded],
        dtype=int,
    )

    if len(filtered) == 0:
        raise ValueError("All selected frames were excluded.")

    print(f"Remaining frames: {len(filtered)}")
    return filtered


# =============================================================================
# SHAPE READING
# =============================================================================

def _validate_vertices(vertices, source_description="shape"):
    vertices = np.asarray(vertices, dtype=float)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"{source_description}: vertices must have shape (N, 3), "
            f"found {vertices.shape}"
        )

    if len(vertices) < 4:
        raise ValueError(
            f"{source_description}: at least 4 vertices are required "
            "for a 3D polyhedron."
        )

    if not np.all(np.isfinite(vertices)):
        raise ValueError(
            f"{source_description}: vertices contain NaN or infinity."
        )

    return vertices


def read_shape_from_json(json_file):
    json_path = Path(json_file)

    if not json_path.is_file():
        raise FileNotFoundError(
            f"Shape JSON file does not exist:\n    {json_file}"
        )

    with json_path.open("r") as handle:
        data = json.load(handle)

    possible_keys = (
        "vertices",
        "8_vertices",
        "12_vertices",
        "polyhedron_vertices",
    )

    for key in possible_keys:
        if key in data:
            return _validate_vertices(
                data[key],
                source_description=f"JSON key '{key}'",
            )

    raise KeyError(
        "Could not find a vertex array in the JSON.\n"
        f"Available keys: {list(data.keys())}\n"
        f"Tried keys: {possible_keys}"
    )


def parse_shape_object(obj):
    """
    Attempt to recover a vertex array from GSD-stored HPMC shape data.
    """
    if obj is None:
        return None

    if isinstance(obj, bytes):
        try:
            obj = obj.decode("utf-8")
        except Exception:
            return None

    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return None

    if isinstance(obj, dict):
        for key in (
            "vertices",
            "8_vertices",
            "12_vertices",
            "polyhedron_vertices",
        ):
            if key in obj:
                try:
                    return _validate_vertices(
                        obj[key],
                        source_description=f"GSD shape key '{key}'",
                    )
                except Exception:
                    pass

    return None


def get_particle_typeid(frame, particle_index):
    typeid = getattr(frame.particles, "typeid", None)

    if typeid is None or len(typeid) == 0:
        return 0

    return int(typeid[particle_index])


def get_shape_from_gsd(frame, particle_index):
    """
    Try common locations used by different GSD/HOOMD versions.
    """
    particle_typeid = get_particle_typeid(frame, particle_index)

    # 1. frame.particles.type_shapes
    try:
        type_shapes = getattr(frame.particles, "type_shapes", None)

        if type_shapes is not None and len(type_shapes) > particle_typeid:
            vertices = parse_shape_object(type_shapes[particle_typeid])
            if vertices is not None:
                print("Shape found in GSD: frame.particles.type_shapes")
                return vertices
    except Exception:
        pass

    # 2. logged shape information
    try:
        if hasattr(frame, "log") and frame.log is not None:
            for key in frame.log.keys():
                low = str(key).lower()

                if "shape" not in low and "vert" not in low:
                    continue

                obj = frame.log[key]

                vertices = parse_shape_object(obj)
                if vertices is not None:
                    print(f"Shape found in GSD log: {key}")
                    return vertices

                try:
                    if len(obj) > particle_typeid:
                        vertices = parse_shape_object(obj[particle_typeid])
                        if vertices is not None:
                            print(f"Shape found in GSD log: {key}")
                            return vertices
                except Exception:
                    pass
    except Exception:
        pass

    return None


# =============================================================================
# PARTICLE LOOKUP
# =============================================================================

def find_particle_index(frame, requested_particle):
    """
    Use a particle tag when the GSD frame exposes tags.
    Otherwise interpret requested_particle as an array index.
    """
    try:
        tags = getattr(frame.particles, "tag", None)

        if tags is not None and len(tags) > 0:
            tags = np.asarray(tags)
            matches = np.where(tags == requested_particle)[0]

            if len(matches) == 0:
                raise IndexError(
                    f"Particle tag {requested_particle} is not present "
                    "in this frame."
                )

            return int(matches[0]), True
    except AttributeError:
        pass

    n_particles = len(frame.particles.position)

    if requested_particle < 0 or requested_particle >= n_particles:
        raise IndexError(
            f"Particle index {requested_particle} is outside the valid "
            f"range 0 ... {n_particles - 1}."
        )

    return int(requested_particle), False


# =============================================================================
# POLYHEDRON RECONSTRUCTION
# =============================================================================

def _canonical_plane(normal, offset):
    """
    Normalize a plane representation.

    scipy.spatial.ConvexHull already returns normalized outward normals,
    but this function keeps grouping robust.
    """
    normal = np.asarray(normal, dtype=float)
    norm = np.linalg.norm(normal)

    if norm == 0:
        raise ValueError("Encountered a zero hull-plane normal.")

    return normal / norm, float(offset) / norm


def _order_face_vertices(vertices, face_vertex_ids, face_normal):
    """
    Order all vertices belonging to one planar polygon cyclically.

    This is what prevents a square/pentagonal face from being displayed as
    arbitrary independent triangles with fake internal diagonal edges.
    """
    ids = np.asarray(sorted(set(face_vertex_ids)), dtype=int)
    points = vertices[ids]
    center = points.mean(axis=0)

    normal = np.asarray(face_normal, dtype=float)
    normal /= np.linalg.norm(normal)

    # Choose a stable in-plane axis.
    ref = points[0] - center
    ref -= np.dot(ref, normal) * normal

    if np.linalg.norm(ref) < 1.0e-12:
        # Fallback: choose any Cartesian direction not parallel to the normal.
        candidates = np.eye(3)
        dots = np.abs(candidates @ normal)
        ref = candidates[np.argmin(dots)]
        ref -= np.dot(ref, normal) * normal

    u = ref / np.linalg.norm(ref)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    relative = points - center
    x = relative @ u
    y = relative @ v
    angles = np.arctan2(y, x)

    order = np.argsort(angles)
    ordered_ids = ids[order]

    # Ensure winding agrees with the outward ConvexHull plane normal.
    ordered_points = vertices[ordered_ids]

    poly_normal = np.zeros(3)
    for i in range(len(ordered_points)):
        p = ordered_points[i] - center
        q = ordered_points[(i + 1) % len(ordered_points)] - center
        poly_normal += np.cross(p, q)

    if np.dot(poly_normal, normal) < 0:
        ordered_ids = ordered_ids[::-1]

    return ordered_ids.tolist()


def reconstruct_polygon_faces(vertices, plane_tolerance=1.0e-7):
    """
    Reconstruct physical polygon faces from a convex set of body vertices.

    ConvexHull triangulates all surfaces.  We use its facet equations to group
    coplanar triangles, then rebuild each coplanar group as ONE polygon.

    Returns
    -------
    polygon_faces : list[list[int]]
        Ordered vertex indices for each physical face.
    hull : scipy.spatial.ConvexHull
        Hull object, useful for diagnostics.
    """
    vertices = _validate_vertices(vertices, "polyhedron")

    hull = ConvexHull(vertices)

    groups = []

    for simplex, equation in zip(hull.simplices, hull.equations):
        normal, offset = _canonical_plane(equation[:3], equation[3])

        matched_group = None

        for group in groups:
            same_normal = np.allclose(
                normal,
                group["normal"],
                rtol=0.0,
                atol=plane_tolerance,
            )
            same_offset = abs(offset - group["offset"]) <= plane_tolerance

            if same_normal and same_offset:
                matched_group = group
                break

        if matched_group is None:
            matched_group = {
                "normal": normal,
                "offset": offset,
                "vertices": set(),
            }
            groups.append(matched_group)

        matched_group["vertices"].update(int(i) for i in simplex)

    polygon_faces = []

    for group in groups:
        ids = _order_face_vertices(
            vertices,
            group["vertices"],
            group["normal"],
        )
        polygon_faces.append(ids)

    return polygon_faces, hull


def polygon_edge_set(polygon_faces):
    """
    Return the unique physical edges implied by polygonal faces.
    """
    edges = set()

    for face in polygon_faces:
        n = len(face)
        for i in range(n):
            a = int(face[i])
            b = int(face[(i + 1) % n])
            edges.add(tuple(sorted((a, b))))

    return edges


def build_pyvista_polyhedron(vertices, polygon_faces):
    """
    Construct a PyVista PolyData containing polygonal cells.
    """
    faces_flat = []

    for face in polygon_faces:
        faces_flat.append(len(face))
        faces_flat.extend(face)

    faces_flat = np.asarray(faces_flat, dtype=np.int64)

    mesh = pv.PolyData(
        np.asarray(vertices, dtype=float),
        faces=faces_flat,
    )

    try:
        mesh.clean(inplace=True)
    except Exception:
        mesh = mesh.clean()

    # Flat face normals are preferred for a true polyhedral appearance.
    try:
        mesh.compute_normals(
            cell_normals=True,
            point_normals=False,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=True,
            inplace=True,
        )
    except TypeError:
        # Compatibility with older PyVista.
        try:
            mesh.compute_normals(
                cell_normals=True,
                point_normals=False,
                consistent_normals=True,
                auto_orient_normals=True,
                inplace=True,
            )
        except Exception:
            pass

    return mesh



# =============================================================================
# ROTATIONAL-SYMMETRY AXIS DETECTION
# =============================================================================
#
# This section intentionally mirrors the scientific construction used in the
# supplied hist_pairwise_angles_entire_sys.py.  Only PROPER rotations are used.
# For the user-specified D5h EPD, the proper rotational subgroup is D5:
#
#     E
#     2 C5 + 2 C5^2    -> four non-identity rotations about ONE C5 line
#     5 C2             -> five distinct C2 lines
#
# Hence there must be 10 proper rotations in total, one unique C5 axis line,
# and five unique C2 axis lines.
#
# Mirror planes, inversion and S_n operations are NOT inferred here because the
# purpose of this visualizer is specifically to display rotational axes.
# =============================================================================

D5H_EXPECTED_PROPER_ROTATIONS = 10
D5H_EXPECTED_C5_AXIS_LINES = 1
D5H_EXPECTED_C2_AXIS_LINES = 5


def _canonical_symmetry_axis_line(axis):
    """
    Normalize an axis and choose one deterministic sign.

    The geometric axis is an unoriented LINE, so n and -n are equivalent.
    """
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))

    if norm <= 1.0e-14:
        raise ValueError("Cannot canonicalize a zero symmetry-axis vector.")

    axis = axis / norm

    for component in axis:
        if abs(component) > 1.0e-14:
            if component < 0.0:
                axis = -axis
            break

    return axis


def _append_unique_axis_line(unique_axes, candidate_axis, angular_tolerance=1.0e-10):
    """
    Add an axis line only if an equivalent parallel/antiparallel line
    has not already been stored.
    """
    candidate_axis = _canonical_symmetry_axis_line(candidate_axis)

    for existing in unique_axes:
        alignment = abs(float(np.dot(candidate_axis, existing)))

        if alignment >= 1.0 - angular_tolerance:
            return

    unique_axes.append(candidate_axis)


def build_symmetry_candidate_axis_lines(
    centered_vertices,
    polygon_faces,
    physical_edges,
):
    """
    Generate candidate rotational-axis lines from the same geometric classes
    used in the supplied pairwise-angle symmetry detector:

      1. center -> vertex
      2. center -> face centroid
      3. face-normal line through the center
      4. center -> edge midpoint

    No coordinate axis (x/y/z) is assumed to be a symmetry direction.
    """
    centered_vertices = np.asarray(centered_vertices, dtype=np.float64)

    candidates = []

    # 1. Center -> vertex.
    for vertex in centered_vertices:
        if np.linalg.norm(vertex) > 1.0e-14:
            _append_unique_axis_line(candidates, vertex)

    # 2. Center -> face centroid, and
    # 3. face normal.
    for face in polygon_faces:
        ids = np.asarray(face, dtype=np.int64)
        face_points = centered_vertices[ids]

        face_centroid = np.mean(face_points, axis=0)

        if np.linalg.norm(face_centroid) > 1.0e-14:
            _append_unique_axis_line(candidates, face_centroid)

        # Robust polygon normal: accumulate cross products about the centroid.
        face_normal = np.zeros(3, dtype=np.float64)

        for i in range(len(face_points)):
            p = face_points[i] - face_centroid
            q = face_points[(i + 1) % len(face_points)] - face_centroid
            face_normal += np.cross(p, q)

        if np.linalg.norm(face_normal) > 1.0e-14:
            _append_unique_axis_line(candidates, face_normal)

    # 4. Center -> edge midpoint.
    for first, second in sorted(physical_edges):
        midpoint = 0.5 * (
            centered_vertices[int(first)]
            + centered_vertices[int(second)]
        )

        if np.linalg.norm(midpoint) > 1.0e-14:
            _append_unique_axis_line(candidates, midpoint)

    if not candidates:
        raise RuntimeError(
            "No nonzero candidate rotational-symmetry axes were generated."
        )

    return tuple(np.asarray(axis, dtype=np.float64) for axis in candidates)


def historical_symmetry_candidate_angles_rad():
    """
    Candidate-angle set retained from the supplied pairwise-angle symmetry code.

    Using the full historical set avoids hard-coding the search to only 72 deg
    and 180 deg even though the expected particle is D5h.
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


def _symmetry_one_to_one_vertex_mapping(rotated_vertices, reference_vertices):
    """
    Globally optimal one-to-one rotated -> reference vertex assignment.

    This is the same permutation criterion used by the supplied histogram code.
    """
    rotated_vertices = np.asarray(rotated_vertices, dtype=np.float64)
    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)

    displacement = (
        rotated_vertices[:, None, :]
        - reference_vertices[None, :, :]
    )
    costs = np.linalg.norm(displacement, axis=2)

    row_indices, column_indices = linear_sum_assignment(costs)

    if len(row_indices) != len(reference_vertices):
        raise RuntimeError("Symmetry vertex assignment is incomplete.")

    mapping = np.empty(len(reference_vertices), dtype=np.int64)
    mapping[row_indices] = column_indices

    residuals = costs[row_indices, column_indices]

    return (
        tuple(int(value) for value in mapping),
        np.asarray(residuals, dtype=np.float64),
    )


def _refine_symmetry_rotation_for_permutation(
    source_vertices,
    target_vertices,
):
    """
    Full-precision least-squares proper rotation mapping source onto target.
    """
    refined_rotation, _ = Rotation.align_vectors(
        np.asarray(target_vertices, dtype=np.float64),
        np.asarray(source_vertices, dtype=np.float64),
    )

    determinant = float(np.linalg.det(refined_rotation.as_matrix()))

    if determinant < 0.0:
        raise RuntimeError(
            "A refined candidate symmetry operation is not a proper rotation."
        )

    return refined_rotation


def _compose_symmetry_permutations(first, second):
    """Permutation produced by applying first and then second."""
    return tuple(second[first[index]] for index in range(len(first)))


def validate_detected_proper_rotation_group(permutations):
    """
    Validate identity, inverses and closure in exact integer permutation space.
    """
    if not permutations:
        raise RuntimeError("No proper rotational symmetry permutations detected.")

    permutation_set = set(permutations)
    identity = tuple(range(len(permutations[0])))

    if identity not in permutation_set:
        raise RuntimeError(
            "Detected proper rotational symmetry set is missing the identity."
        )

    # Inverses.
    for permutation in permutations:
        inverse = [0] * len(permutation)

        for source, target in enumerate(permutation):
            inverse[target] = source

        if tuple(inverse) not in permutation_set:
            raise RuntimeError(
                "Detected proper rotational symmetry set is missing an inverse."
            )

    # Closure.
    for first in permutations:
        for second in permutations:
            composed = _compose_symmetry_permutations(first, second)

            if composed not in permutation_set:
                raise RuntimeError(
                    "Detected proper rotational symmetry operations are not "
                    "closed under composition."
                )


def _axis_angle_from_refined_rotation(rotation, zero_angle_tolerance=1.0e-10):
    """
    Extract the geometric axis line and principal rotation angle from a
    full-precision scipy Rotation.
    """
    rotvec = np.asarray(rotation.as_rotvec(), dtype=np.float64)
    angle_rad = float(np.linalg.norm(rotvec))

    if angle_rad <= zero_angle_tolerance:
        return None, 0.0

    axis = _canonical_symmetry_axis_line(rotvec / angle_rad)
    angle_deg = float(np.rad2deg(angle_rad))

    return axis, angle_deg


def _group_rotation_operations_by_axis_line(
    operations,
    line_tolerance=1.0e-8,
):
    """
    Group all non-identity proper rotations that act about the same geometric
    axis line.
    """
    axis_groups = []

    for operation in operations:
        axis, angle_deg = _axis_angle_from_refined_rotation(
            operation["rotation"]
        )

        if axis is None:
            continue

        matched_group = None

        for group in axis_groups:
            alignment = abs(float(np.dot(axis, group["axis"])))

            if alignment >= 1.0 - line_tolerance:
                matched_group = group
                break

        if matched_group is None:
            matched_group = {
                "axis": axis,
                "angles_deg": [],
                "operations": [],
            }
            axis_groups.append(matched_group)

        matched_group["angles_deg"].append(angle_deg)
        matched_group["operations"].append(operation)

    return axis_groups


def detect_d5h_rotational_axes(
    body_vertices,
    polygon_faces,
    physical_edges,
    precision_exponent=2,
):
    """
    Detect and VALIDATE the C5/C2 rotational axes of the supplied EPD geometry.

    Important:
    ----------
    This function detects the proper rotational subgroup only.  The user's
    particle is D5h, whose proper rotational subgroup is D5.  Therefore the
    required rotational content is:

        one C5 axis line
        five C2 axis lines
        ten proper rotations including identity

    The function refuses to draw axes if these checks fail, rather than
    silently drawing an incorrect coordinate-based guess.
    """
    if precision_exponent < 0:
        raise ValueError(
            "--symmetry-precision-exponent must be nonnegative."
        )

    matching_tolerance = 10.0 ** (-int(precision_exponent))

    body_vertices = np.asarray(body_vertices, dtype=np.float64)
    body_center = np.mean(body_vertices, axis=0)
    centered_vertices = body_vertices - body_center

    body_radius = float(
        np.max(np.linalg.norm(centered_vertices, axis=1))
    )

    if body_radius <= 1.0e-14:
        raise RuntimeError(
            "Cannot determine symmetry axes because the body radius is zero."
        )

    candidate_axes = build_symmetry_candidate_axis_lines(
        centered_vertices,
        polygon_faces,
        physical_edges,
    )
    candidate_angles = historical_symmetry_candidate_angles_rad()

    # Store each physical operation by its exact vertex permutation.
    operations_by_permutation = {}

    identity_permutation = tuple(range(len(centered_vertices)))
    operations_by_permutation[identity_permutation] = {
        "rotation": Rotation.identity(),
        "permutation": identity_permutation,
        "max_residual": 0.0,
    }

    tested_candidates = 0
    accepted_candidates = 0

    for axis in candidate_axes:
        for angle_rad in candidate_angles:
            tested_candidates += 1

            candidate_rotation = Rotation.from_rotvec(
                axis * float(angle_rad)
            )

            candidate_rotated = candidate_rotation.apply(
                centered_vertices
            )

            permutation, residuals = (
                _symmetry_one_to_one_vertex_mapping(
                    candidate_rotated,
                    centered_vertices,
                )
            )

            if float(np.max(residuals)) > matching_tolerance:
                continue

            target_vertices = centered_vertices[
                np.asarray(permutation, dtype=np.int64)
            ]

            refined_rotation = _refine_symmetry_rotation_for_permutation(
                centered_vertices,
                target_vertices,
            )

            refined_rotated = refined_rotation.apply(centered_vertices)

            refined_permutation, refined_residuals = (
                _symmetry_one_to_one_vertex_mapping(
                    refined_rotated,
                    centered_vertices,
                )
            )

            refined_max_residual = float(
                np.max(refined_residuals)
            )

            if refined_permutation != permutation:
                raise RuntimeError(
                    "Full-precision refinement changed a discovered symmetry "
                    "permutation. Tighten --symmetry-precision-exponent."
                )

            if refined_max_residual > matching_tolerance:
                continue

            accepted_candidates += 1

            operation = {
                "rotation": refined_rotation,
                "permutation": permutation,
                "max_residual": refined_max_residual,
            }

            previous = operations_by_permutation.get(permutation)

            if (
                previous is None
                or refined_max_residual < previous["max_residual"]
            ):
                operations_by_permutation[permutation] = operation

    operations = tuple(
        sorted(
            operations_by_permutation.values(),
            key=lambda operation: (
                np.linalg.norm(
                    operation["rotation"].as_matrix() - np.eye(3)
                ),
                operation["permutation"],
            ),
        )
    )

    validate_detected_proper_rotation_group(
        [operation["permutation"] for operation in operations]
    )

    axis_groups = _group_rotation_operations_by_axis_line(operations)

    classified_groups = []

    for group in axis_groups:
        positive_angles = [
            angle
            for angle in group["angles_deg"]
            if angle > 1.0e-8
        ]

        if not positive_angles:
            continue

        minimum_angle = min(positive_angles)
        estimated_order = int(round(360.0 / minimum_angle))

        if estimated_order < 2:
            continue

        expected_minimum_angle = 360.0 / estimated_order

        # Refinement should make this extremely accurate; 0.1 degree is
        # deliberately much looser than numerical noise but tight enough to
        # reject an accidental classification.
        if abs(minimum_angle - expected_minimum_angle) > 0.1:
            raise RuntimeError(
                "Could not assign an integer rotational order to a detected "
                f"symmetry axis: minimum angle = {minimum_angle:.12g} deg."
            )

        classified_groups.append(
            {
                "axis": np.asarray(group["axis"], dtype=np.float64),
                "order": estimated_order,
                "angles_deg": tuple(sorted(positive_angles)),
            }
        )

    c5_axes = [
        group["axis"]
        for group in classified_groups
        if group["order"] == 5
    ]

    c2_axes = [
        group["axis"]
        for group in classified_groups
        if group["order"] == 2
    ]

    # ---------------------------------------------------------------------
    # STRICT D5 ROTATIONAL-SUBGROUP VALIDATION FOR THE USER'S D5h EPD
    # ---------------------------------------------------------------------
    if len(operations) != D5H_EXPECTED_PROPER_ROTATIONS:
        raise RuntimeError(
            "D5h rotational-axis validation failed: expected "
            f"{D5H_EXPECTED_PROPER_ROTATIONS} proper rotations (D5 subgroup), "
            f"but detected {len(operations)}."
        )

    if len(c5_axes) != D5H_EXPECTED_C5_AXIS_LINES:
        raise RuntimeError(
            "D5h rotational-axis validation failed: expected exactly one C5 "
            f"axis line, but detected {len(c5_axes)}."
        )

    if len(c2_axes) != D5H_EXPECTED_C2_AXIS_LINES:
        raise RuntimeError(
            "D5h rotational-axis validation failed: expected exactly five C2 "
            f"axis lines, but detected {len(c2_axes)}."
        )

    c5_axis = _canonical_symmetry_axis_line(c5_axes[0])

    c2_axes = [
        _canonical_symmetry_axis_line(axis)
        for axis in c2_axes
    ]

    # The five C2 axes of D5 must be perpendicular to the principal C5 axis.
    perpendicularity_residuals = [
        abs(float(np.dot(c5_axis, axis)))
        for axis in c2_axes
    ]

    max_perpendicularity_residual = max(
        perpendicularity_residuals
    )

    if max_perpendicularity_residual > 1.0e-6:
        raise RuntimeError(
            "D5h rotational-axis validation failed: one or more detected C2 "
            "axes are not perpendicular to the C5 axis. "
            f"Maximum |C5 dot C2| = {max_perpendicularity_residual:.3e}."
        )

    # Stable ordering of the five C2 lines around the C5 axis.  This does not
    # change their geometry; it only makes terminal output deterministic.
    reference = c2_axes[0]

    # Construct an in-plane orthonormal basis.
    plane_u = reference / np.linalg.norm(reference)
    plane_v = np.cross(c5_axis, plane_u)
    plane_v /= np.linalg.norm(plane_v)

    def c2_sort_angle(axis):
        x = float(np.dot(axis, plane_u))
        y = float(np.dot(axis, plane_v))

        # An axis line is equivalent under theta -> theta + pi.
        angle = float(np.arctan2(y, x))
        return angle % np.pi

    c2_axes = sorted(c2_axes, key=c2_sort_angle)

    print("\n" + "=" * 78)
    print("ROTATIONAL SYMMETRY AXES")
    print("=" * 78)
    print(
        "Geometry-derived proper rotational subgroup: D5 "
        "(consistent with user-specified D5h)"
    )
    print(f"Matching tolerance            : {matching_tolerance:.12g}")
    print(f"Candidate axis lines          : {len(candidate_axes)}")
    print(f"Candidate axis/angle tests    : {tested_candidates}")
    print(f"Accepted discovery hits       : {accepted_candidates}")
    print(f"Distinct proper rotations     : {len(operations)}")
    print(f"C5 axis lines                 : {len(c5_axes)}")
    print(f"C2 axis lines                 : {len(c2_axes)}")
    print(
        f"Maximum |C5 dot C2|          : "
        f"{max_perpendicularity_residual:.3e}"
    )

    print(
        "C5 body-frame axis            : "
        f"[{c5_axis[0]: .10f}, {c5_axis[1]: .10f}, {c5_axis[2]: .10f}]"
    )

    for index, axis in enumerate(c2_axes, start=1):
        print(
            f"C2[{index}] body-frame axis         : "
            f"[{axis[0]: .10f}, {axis[1]: .10f}, {axis[2]: .10f}]"
        )

    return {
        "body_center": np.asarray(body_center, dtype=np.float64),
        "body_radius": body_radius,
        "c5_axis": np.asarray(c5_axis, dtype=np.float64),
        "c2_axes": tuple(
            np.asarray(axis, dtype=np.float64)
            for axis in c2_axes
        ),
        "proper_rotation_count": len(operations),
        "matching_tolerance": matching_tolerance,
    }


def hoomd_orientation_to_scipy_rotation(orientation):
    """
    Convert one HOOMD scalar-first quaternion [w,x,y,z] to scipy Rotation.
    """
    quat = np.asarray(orientation, dtype=np.float64)

    if quat.shape != (4,):
        raise ValueError(
            f"Expected quaternion with shape (4,), found {quat.shape}"
        )

    qnorm = float(np.linalg.norm(quat))

    if qnorm <= 1.0e-14:
        raise ValueError("Particle quaternion has zero norm.")

    quat = quat / qnorm

    return Rotation.from_quat(
        [quat[1], quat[2], quat[3], quat[0]]
    )


def transform_symmetry_axes_to_frame(
    symmetry_axes_body,
    position,
    orientation,
    center_on_particle=False,
):
    """
    Rotate the BODY-FRAME C5/C2 lines with exactly the same HOOMD orientation
    used for the particle.

    The axes therefore remain rigidly attached to the polyhedron in every PNG
    and consequently in the MP4 assembled from those PNGs.
    """
    if symmetry_axes_body is None:
        return None

    rotation = hoomd_orientation_to_scipy_rotation(orientation)

    body_center = np.asarray(
        symmetry_axes_body["body_center"],
        dtype=np.float64,
    )

    frame_center = rotation.apply(body_center)

    if not center_on_particle:
        frame_center = (
            frame_center
            + np.asarray(position, dtype=np.float64)
        )

    c5_axis = rotation.apply(
        np.asarray(symmetry_axes_body["c5_axis"], dtype=np.float64)
    )
    c5_axis /= np.linalg.norm(c5_axis)

    c2_axes = []

    for body_axis in symmetry_axes_body["c2_axes"]:
        frame_axis = rotation.apply(
            np.asarray(body_axis, dtype=np.float64)
        )
        frame_axis /= np.linalg.norm(frame_axis)
        c2_axes.append(frame_axis)

    return {
        "center": np.asarray(frame_center, dtype=np.float64),
        "body_radius": float(symmetry_axes_body["body_radius"]),
        "c5_axis": np.asarray(c5_axis, dtype=np.float64),
        "c2_axes": tuple(
            np.asarray(axis, dtype=np.float64)
            for axis in c2_axes
        ),
    }


def _make_axis_cylinder(center, direction, radius, half_length):
    """
    Construct one finite cylinder representing an infinite symmetry-axis line.

    The cylinder extends equally in +n and -n, so it represents an axis LINE,
    not a directed vector.
    """
    center = np.asarray(center, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)

    kwargs = dict(
        center=center,
        direction=direction,
        radius=float(radius),
        height=float(2.0 * half_length),
        resolution=64,
    )

    try:
        return pv.Cylinder(capping=True, **kwargs)
    except TypeError:
        # Compatibility with historical PyVista versions.
        return pv.Cylinder(**kwargs)


def add_rotational_symmetry_axes(
    plotter,
    transformed_symmetry_axes,
    args,
):
    """
    Add the C5 and C2 axis tubes to the current 3D scene.

    Returns
    -------
    endpoint_points : ndarray, shape (12, 3)
        Axis endpoints.  They are passed to the existing camera routine so the
        extended symmetry axes cannot be cropped out of the image.
    """
    if transformed_symmetry_axes is None:
        return np.empty((0, 3), dtype=np.float64)

    center = np.asarray(
        transformed_symmetry_axes["center"],
        dtype=np.float64,
    )

    body_radius = float(
        transformed_symmetry_axes["body_radius"]
    )

    half_length = (
        float(args.symmetry_axis_length_factor)
        * body_radius
    )

    c5_radius = max(
        float(args.c5_axis_radius_factor) * body_radius,
        1.0e-5,
    )

    c2_radius = max(
        float(args.c2_axis_radius_factor) * body_radius,
        1.0e-5,
    )

    opacity = float(
        np.clip(args.symmetry_axis_opacity, 0.0, 1.0)
    )

    endpoint_points = []

    # ---------------------------------------------------------------------
    # Unique principal C5 axis: deliberately thicker than the C2 axes.
    # ---------------------------------------------------------------------
    c5_axis = np.asarray(
        transformed_symmetry_axes["c5_axis"],
        dtype=np.float64,
    )

    c5_cylinder = _make_axis_cylinder(
        center,
        c5_axis,
        c5_radius,
        half_length,
    )

    plotter.add_mesh(
        c5_cylinder,
        color=args.c5_axis_color,
        opacity=opacity,
        smooth_shading=True,
        lighting=False,
    )

    endpoint_points.extend(
        [
            center - half_length * c5_axis,
            center + half_length * c5_axis,
        ]
    )

    # ---------------------------------------------------------------------
    # Five symmetry-equivalent C2 axes.
    # ---------------------------------------------------------------------
    for c2_axis in transformed_symmetry_axes["c2_axes"]:
        c2_axis = np.asarray(c2_axis, dtype=np.float64)

        c2_cylinder = _make_axis_cylinder(
            center,
            c2_axis,
            c2_radius,
            half_length,
        )

        plotter.add_mesh(
            c2_cylinder,
            color=args.c2_axis_color,
            opacity=opacity,
            smooth_shading=True,
            lighting=False,
        )

        endpoint_points.extend(
            [
                center - half_length * c2_axis,
                center + half_length * c2_axis,
            ]
        )

    if args.show_symmetry_legend:
        try:
            plotter.add_legend(
                [
                    ["C5 axis", args.c5_axis_color],
                    ["C2 axes", args.c2_axis_color],
                ],
                bcolor=args.background_color,
                border=False,
            )
        except Exception:
            # Legend support varies across older PyVista releases.  The symmetry
            # geometry itself must never fail merely because a legend is absent.
            pass

    return np.asarray(endpoint_points, dtype=np.float64)



# =============================================================================
# RIGID-BODY TRANSFORMATION
# =============================================================================

def transform_vertices(
    body_vertices,
    position,
    orientation,
    center_on_particle=False,
):
    """
    Transform body-frame vertices using the HOOMD quaternion.

    HOOMD stores quaternion orientation as:
        [q_w, q_x, q_y, q_z]

    scipy Rotation.from_quat expects:
        [q_x, q_y, q_z, q_w]
    """
    # Use the SAME scalar-first HOOMD quaternion conversion as the
    # rotational-symmetry axes so body and axes can never drift apart.
    rotation = hoomd_orientation_to_scipy_rotation(orientation)

    rotated = rotation.apply(body_vertices)

    if center_on_particle:
        return rotated

    return rotated + np.asarray(position, dtype=float)


# =============================================================================
# CAMERA AND RENDER QUALITY
# =============================================================================

CAMERA_DIRECTIONS = {
    "iso": np.array([1.0, 1.0, 0.85]),
    "front": np.array([0.0, -1.0, 0.0]),
    "back": np.array([0.0, 1.0, 0.0]),
    "side": np.array([1.0, 0.0, 0.0]),
    "left": np.array([-1.0, 0.0, 0.0]),
    "top": np.array([0.0, 0.0, 1.0]),
    "bottom": np.array([0.0, 0.0, -1.0]),
}


def configure_camera(
    plotter,
    vertices,
    angle_type,
    distance_factor=3.8,
    zoom=1.15,
):
    vertices = np.asarray(vertices, dtype=float)

    center = vertices.mean(axis=0)
    radius = np.max(np.linalg.norm(vertices - center, axis=1))

    if radius <= 1.0e-12:
        radius = 1.0

    direction = CAMERA_DIRECTIONS[angle_type].astype(float)
    direction /= np.linalg.norm(direction)

    distance = max(distance_factor * radius, 1.0)

    camera_position = center + direction * distance

    plotter.camera.position = tuple(camera_position)
    plotter.camera.focal_point = tuple(center)

    # Choose a sensible up vector and avoid degeneracy for top/bottom views.
    if abs(direction[2]) > 0.95:
        plotter.camera.up = (0.0, 1.0, 0.0)
    else:
        plotter.camera.up = (0.0, 0.0, 1.0)

    try:
        plotter.camera.view_angle = 28.0
    except Exception:
        pass

    try:
        plotter.reset_camera_clipping_range()
    except Exception:
        pass

    try:
        plotter.camera.zoom(float(zoom))
    except Exception:
        pass


def configure_render_quality(plotter):
    """
    Enable the best available anti-aliasing without requiring a specific
    PyVista release.
    """
    # Best option on recent PyVista/VTK.
    try:
        plotter.enable_anti_aliasing("ssaa")
        return
    except Exception:
        pass

    # Good fallback on older releases.
    try:
        plotter.enable_anti_aliasing("msaa")
        return
    except Exception:
        pass

    # Oldest compatibility path.
    try:
        plotter.enable_anti_aliasing()
    except Exception:
        pass


def add_scene_lighting(plotter, vertices):
    """
    Add a restrained three-light arrangement when the installed PyVista
    supports explicit Light objects.  Otherwise the default PyVista lighting
    remains active.
    """
    vertices = np.asarray(vertices, dtype=float)
    center = vertices.mean(axis=0)
    radius = np.max(np.linalg.norm(vertices - center, axis=1))

    if radius <= 1.0e-12:
        radius = 1.0

    try:
        plotter.remove_all_lights()

        light_specs = [
            # Key light
            (np.array([4.0, -3.0, 5.0]), 0.85),
            # Fill light
            (np.array([-4.0, -1.0, 2.5]), 0.45),
            # Rim / separation light
            (np.array([1.0, 4.0, 3.0]), 0.35),
        ]

        for direction, intensity in light_specs:
            direction = direction / np.linalg.norm(direction)
            position = center + direction * (6.0 * radius)

            light = pv.Light(
                position=tuple(position),
                focal_point=tuple(center),
                color="white",
                intensity=float(intensity),
            )
            plotter.add_light(light)

    except Exception:
        # Default PyVista lighting is still perfectly usable.
        pass


# =============================================================================
# FRAME COLOR
# =============================================================================

def get_frame_color(frame_local_index, n_selected, color_mode, solid_color):
    if color_mode == "solid":
        return solid_color

    fraction = frame_local_index / max(n_selected - 1, 1)
    rgba = plt.cm.viridis(fraction)

    # PyVista accepts RGB floats in [0, 1].
    return tuple(float(x) for x in rgba[:3])


# =============================================================================
# RENDER ONE FRAME
# =============================================================================

def render_frame(
    output_file,
    transformed_vertices,
    polygon_faces,
    particle_number,
    absolute_frame_number,
    local_frame_index,
    n_selected,
    args,
    transformed_symmetry_axes=None,
):
    mesh = build_pyvista_polyhedron(
        transformed_vertices,
        polygon_faces,
    )

    plotter = pv.Plotter(
        off_screen=True,
        window_size=(args.resolution, args.resolution),
    )

    plotter.set_background(args.background_color)
    configure_render_quality(plotter)
    add_scene_lighting(plotter, transformed_vertices)

    face_color = get_frame_color(
        local_frame_index,
        n_selected,
        args.color_mode,
        args.face_color,
    )

    # IMPORTANT:
    # We render the actual PolyData surface, NOT individual vertex spheres.
    plotter.add_mesh(
        mesh,
        color=face_color,
        opacity=float(np.clip(args.opacity, 0.0, 1.0)),
        show_edges=True,
        edge_color=args.edge_color,
        line_width=float(args.edge_width),
        smooth_shading=False,
        lighting=True,
        ambient=0.16,
        diffuse=0.78,
        specular=0.32,
        specular_power=28.0,
        render_lines_as_tubes=True,
    )

    # ---------------------------------------------------------------------
    # Overlay geometry-derived rotational symmetry axes.
    #
    # These are true 3D objects and therefore obey normal depth occlusion:
    # portions passing through the opaque particle are hidden, while both ends
    # extend beyond the body.  This avoids a misleading "painted on top" axis.
    # ---------------------------------------------------------------------
    symmetry_endpoint_points = np.empty((0, 3), dtype=np.float64)

    if transformed_symmetry_axes is not None:
        symmetry_endpoint_points = add_rotational_symmetry_axes(
            plotter,
            transformed_symmetry_axes,
            args,
        )

    if args.show_vertices:
        radius = np.max(
            np.linalg.norm(
                transformed_vertices
                - np.mean(transformed_vertices, axis=0),
                axis=1,
            )
        )

        sphere_radius = max(radius * 0.022, 1.0e-4)

        for vertex in transformed_vertices:
            sphere = pv.Sphere(
                radius=sphere_radius,
                center=vertex,
                theta_resolution=24,
                phi_resolution=24,
            )

            plotter.add_mesh(
                sphere,
                color=args.edge_color,
                smooth_shading=True,
                lighting=True,
            )

    if args.show_axes:
        try:
            plotter.add_axes(
                line_width=2,
                labels_off=False,
            )
        except TypeError:
            plotter.add_axes()

    if not args.no_title:
        title = (
            f"Particle {particle_number}   |   "
            f"GSD frame {absolute_frame_number}"
        )

        try:
            plotter.add_title(
                title,
                font_size=18,
                color="black"
                if args.background_color.lower() in ("white", "#ffffff")
                else "white",
            )
        except Exception:
            # The visualization should not fail just because an old
            # PyVista version handles titles differently.
            pass

    # Include the extended C5/C2 endpoints when choosing the camera radius.
    # The existing camera algorithm is otherwise unchanged.
    camera_points = transformed_vertices

    if len(symmetry_endpoint_points) > 0:
        camera_points = np.vstack(
            [transformed_vertices, symmetry_endpoint_points]
        )

    configure_camera(
        plotter,
        camera_points,
        args.camera_angle,
        distance_factor=args.camera_distance_factor,
        zoom=args.zoom,
    )

    # Render once before screenshot so lights/camera are fully initialized.
    try:
        plotter.render()
    except Exception:
        pass

    plotter.screenshot(
        str(output_file),
        transparent_background=args.transparent_background,
    )

    plotter.close()


# =============================================================================
# VIDEO
# =============================================================================

def create_video_from_frames(frame_dir, output_video, fps=10):
    if not IMAGEIO_AVAILABLE:
        print(
            "\nWARNING: imageio is unavailable; skipping video.\n"
            "Install with:\n"
            "    pip install imageio imageio-ffmpeg"
        )
        return False

    frame_files = sorted(
        list(Path(frame_dir).glob("frame_*.png"))
    )

    if not frame_files:
        print("No rendered PNG frames were found for video creation.")
        return False

    print("\n" + "=" * 78)
    print("CREATING VIDEO")
    print("=" * 78)
    print(f"Frames : {len(frame_files)}")
    print(f"FPS    : {fps}")
    print(f"Output : {output_video}")

    try:
        writer = imageio.get_writer(
            str(output_video),
            fps=int(fps),
            codec="libx264",
            quality=9,
            pixelformat="yuv420p",
        )

        for i, frame_file in enumerate(frame_files, start=1):
            writer.append_data(imageio.imread(frame_file))

            if i % 25 == 0 or i == len(frame_files):
                print(f"Encoded {i}/{len(frame_files)}")

        writer.close()
        return True

    except Exception as exc:
        print(f"Video creation failed: {exc}")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_arguments()

    if args.resolution < 256:
        raise ValueError("--resolution should be at least 256.")

    if args.n_frames <= 0:
        raise ValueError("n_frames must be positive.")

    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")

    if args.edge_width <= 0:
        raise ValueError("--edge-width must be positive.")

    if args.symmetry_precision_exponent < 0:
        raise ValueError(
            "--symmetry-precision-exponent must be nonnegative."
        )

    if args.symmetry_axis_length_factor <= 0.0:
        raise ValueError(
            "--symmetry-axis-length-factor must be positive."
        )

    if args.c5_axis_radius_factor <= 0.0:
        raise ValueError(
            "--c5-axis-radius-factor must be positive."
        )

    if args.c2_axis_radius_factor <= 0.0:
        raise ValueError(
            "--c2-axis-radius-factor must be positive."
        )

    if not 0.0 <= args.symmetry_axis_opacity <= 1.0:
        raise ValueError(
            "--symmetry-axis-opacity must lie between 0 and 1."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Open trajectory
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("OPENING GSD FILE")
    print("=" * 78)

    try:
        trajectory = gsd.hoomd.open(args.gsd_file, mode="r")
    except Exception as exc:
        raise RuntimeError(
            f"Could not open GSD file:\n    {args.gsd_file}\n{exc}"
        ) from exc

    total_frames = len(trajectory)

    if total_frames == 0:
        raise ValueError("The GSD contains no frames.")

    start_frame = max(0, total_frames - args.n_frames)
    frame_indices = np.arange(start_frame, total_frames, dtype=int)

    if args.exclude:
        excluded = parse_frame_specification(args.exclude)
        frame_indices = np.asarray(
            [f for f in frame_indices if int(f) not in excluded],
            dtype=int,
        )

    if len(frame_indices) == 0:
        raise ValueError("No frames remain after applying --exclude.")

    if args.interactive_selection:
        frame_indices = ask_for_frame_exclusion(frame_indices)

    print(f"GSD file              : {args.gsd_file}")
    print(f"Total trajectory frames: {total_frames}")
    print(f"Frames requested       : {args.n_frames}")
    print(f"Frames to render       : {len(frame_indices)}")
    print(
        f"Selected frame range   : "
        f"{int(frame_indices[0])} ... {int(frame_indices[-1])}"
    )

    # -------------------------------------------------------------------------
    # Particle and shape
    # -------------------------------------------------------------------------
    first_frame = trajectory[int(frame_indices[0])]

    particle_index, has_tags = find_particle_index(
        first_frame,
        args.particle,
    )

    print("\n" + "=" * 78)
    print("PARTICLE INFORMATION")
    print("=" * 78)

    if has_tags:
        print(f"Particle tag          : {args.particle}")
        print(f"Current array index   : {particle_index}")
    else:
        print(f"Particle array index  : {args.particle}")
        print("No usable particle tags were found.")

    body_vertices = get_shape_from_gsd(
        first_frame,
        particle_index,
    )

    if body_vertices is None:
        print("\nNo usable polyhedron vertices were found in the GSD.")
        print(f"Reading body vertices from JSON:\n    {args.shape_json}")
        body_vertices = read_shape_from_json(args.shape_json)

    # -------------------------------------------------------------------------
    # Reconstruct the physical faces once in the particle body frame.
    # -------------------------------------------------------------------------
    polygon_faces, hull = reconstruct_polygon_faces(body_vertices)
    physical_edges = polygon_edge_set(polygon_faces)

    face_sizes = [len(face) for face in polygon_faces]

    print("\n" + "=" * 78)
    print("RECONSTRUCTED POLYHEDRON")
    print("=" * 78)
    print(f"Vertices              : {len(body_vertices)}")
    print(f"Physical faces        : {len(polygon_faces)}")
    print(f"Physical edges        : {len(physical_edges)}")
    print(f"Face vertex counts    : {face_sizes}")
    print(f"Convex-hull volume    : {hull.volume:.10g}")
    print(f"Convex-hull area      : {hull.area:.10g}")

    # Euler check for a closed convex polyhedron: V - E + F = 2.
    euler = (
        len(body_vertices)
        - len(physical_edges)
        + len(polygon_faces)
    )
    print(f"Euler V-E+F           : {euler}")

    if euler != 2:
        print(
            "WARNING: Euler check is not 2. "
            "Inspect the JSON vertices / plane tolerance."
        )

    # -------------------------------------------------------------------------
    # Detect the D5 rotational subgroup expected for this D5h EPD.
    #
    # This is intentionally done ONCE in the body frame.  Per-frame work only
    # rotates the already validated axis lines by the particle quaternion.
    # -------------------------------------------------------------------------
    symmetry_axes_body = None

    if not args.no_symmetry_axes:
        symmetry_axes_body = detect_d5h_rotational_axes(
            body_vertices=body_vertices,
            polygon_faces=polygon_faces,
            physical_edges=physical_edges,
            precision_exponent=args.symmetry_precision_exponent,
        )
    else:
        print("\nRotational-symmetry-axis overlay disabled.")

    # -------------------------------------------------------------------------
    # Render frames
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RENDERING COMPLETE POLYHEDRON")
    print("=" * 78)
    print(
        "Rendering FILLED polygonal faces with physical edges; "
        "vertex-only rendering is disabled by default."
    )

    n_selected = len(frame_indices)

    for local_index, absolute_frame_number in enumerate(frame_indices):
        frame = trajectory[int(absolute_frame_number)]

        particle_index, _ = find_particle_index(
            frame,
            args.particle,
        )

        position = np.asarray(
            frame.particles.position[particle_index],
            dtype=float,
        )

        orientation = np.asarray(
            frame.particles.orientation[particle_index],
            dtype=float,
        )

        transformed_vertices = transform_vertices(
            body_vertices,
            position,
            orientation,
            center_on_particle=args.center,
        )

        transformed_symmetry_axes = transform_symmetry_axes_to_frame(
            symmetry_axes_body,
            position,
            orientation,
            center_on_particle=args.center,
        )

        output_file = output_dir / f"frame_{local_index:06d}.png"

        render_frame(
            output_file=output_file,
            transformed_vertices=transformed_vertices,
            polygon_faces=polygon_faces,
            particle_number=args.particle,
            absolute_frame_number=int(absolute_frame_number),
            local_frame_index=local_index,
            n_selected=n_selected,
            args=args,
            transformed_symmetry_axes=transformed_symmetry_axes,
        )

        count = local_index + 1

        if count % 10 == 0 or count == n_selected:
            print(
                f"Rendered {count:5d}/{n_selected} "
                f"| GSD frame {int(absolute_frame_number)}"
            )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RENDER SUMMARY")
    print("=" * 78)
    print(f"Rendered images        : {n_selected}")
    print(f"Output directory       : {output_dir.resolve()}")
    print(
        f"Image resolution       : "
        f"{args.resolution} x {args.resolution}"
    )
    print(f"Camera                 : {args.camera_angle}")
    print(f"Centered               : {args.center}")
    print(f"Face color mode        : {args.color_mode}")
    print(f"Vertex markers         : {args.show_vertices}")
    print(
        f"Rotational axes        : "
        f"{'OFF' if args.no_symmetry_axes else 'C5 + five C2'}"
    )

    if not args.no_symmetry_axes:
        print(f"C5 axis color          : {args.c5_axis_color}")
        print(f"C2 axes color          : {args.c2_axis_color}")
        print(
            f"Symmetry axis half-size: "
            f"{args.symmetry_axis_length_factor} x body radius"
        )

    if args.create_video:
        video_file = output_dir / f"particle_{args.particle}_polyhedron.mp4"

        success = create_video_from_frames(
            output_dir,
            video_file,
            fps=args.video_fps,
        )

        if success:
            print(f"Video                   : {video_file.resolve()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
