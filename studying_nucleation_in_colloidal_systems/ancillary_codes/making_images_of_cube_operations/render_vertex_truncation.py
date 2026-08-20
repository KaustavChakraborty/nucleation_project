"""
render_one_vertex_truncation_no_arrows.py

Publication-style visualization of the one-vertex cube truncation used in
cube2trncatedCube_v2.py.

The (+x,+y,+z) corner is removed and replaced by three points, each lying
TRUNCATION_PARAM units from that corner along one of its three incident cube
edges. The resulting vertex set is recentered by its arithmetic vertex mean
(the same convention used in the original script) and then uniformly rescaled
so that the convex-hull volume is exactly 1.

Produces:
    output_vertex_truncation/
        1_original_cube.png
        2_operation.png
        3_truncated_cube.png
        combined_panel.png

No symmetry/axis arrows are drawn.

Dependencies:
    numpy scipy pyvista

Run:
    python3 render_one_vertex_truncation_no_arrows.py

For a headless Linux machine, if your PyVista/VTK build needs an X server:
    xvfb-run -a python3 render_one_vertex_truncation_no_arrows.py
"""

import glob
import json
import os

import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull


# -----------------------------------------------------------------------------
# 1. Parameters -- mirrors cube2trncatedCube_v2.py
# -----------------------------------------------------------------------------
CUBE_JSON = "shape_023_Cube_unit_volume_principal_frame.json"
TRUNCATION_PARAM = 0.70
OUTPUT_DIR = "output_vertex_truncation"
WINDOW_SIZE = (1600, 1600)
SAMPLES = 16

# Same visual language as the previous edge-truncation renderer
COL_CUBE = "#8A5FD1"
COL_TRUNC = "#3E9C7E"
COL_HIGHLIGHT_FACE = "#E8552A"
COL_VANISH = "#1A1A1A"
COL_NEW = "#1F6FEB"
COL_EDGE = "black"
BACKGROUND = "white"


# -----------------------------------------------------------------------------
# 2. Geometry helpers
# -----------------------------------------------------------------------------
def load_cube_vertices():
    """Load the original cube JSON if present.

    Besides the canonical filename, this also accepts timestamped filenames
    such as
        shape_023_Cube_unit_volume_principal_frame(20260813-150122).json
    so the renderer works directly with downloaded copies.

    If no JSON is found, fall back to the exact unit cube used by that file.
    """
    candidates = []
    if os.path.exists(CUBE_JSON):
        candidates.append(CUBE_JSON)
    candidates.extend(sorted(glob.glob("shape_023_Cube_unit_volume_principal_frame*.json")))

    if candidates:
        with open(candidates[0]) as f:
            data = json.load(f)
        return np.asarray(data["8_vertices"], dtype=float)

    return np.array(
        [[x, y, z]
         for x in (-0.5, 0.5)
         for y in (-0.5, 0.5)
         for z in (-0.5, 0.5)],
        dtype=float,
    )


def truncate_one_positive_corner(vertices, truncation_param):
    """Faithful clean reimplementation of trunc_cube() from the source script.

    The unique vertex with x>0, y>0, z>0 is removed. It is replaced with
    three points obtained by moving `truncation_param` along the three incident
    edges in the -x, -y and -z directions.

    Returns
    -------
    truncated_vertices : (N+2, 3) ndarray
        Seven untouched cube vertices plus the three new cut vertices.
    removed_vertex : (3,) ndarray
        The disappearing (+,+,+) corner.
    new_points : (3,3) ndarray
        The three new vertices defining the triangular truncation face.
    """
    vertices = np.asarray(vertices, dtype=float)
    mask = np.all(vertices > 0.0, axis=1)
    targets = vertices[mask]

    if len(targets) != 1:
        raise ValueError(
            "Expected exactly one cube vertex with x>0, y>0 and z>0; "
            f"found {len(targets)}."
        )

    removed = targets[0].copy()

    # Exactly matches the source logic for the positive (+,+,+) corner:
    # [x-t, y, z], [x, y-t, z], [x, y, z-t]
    new_points = []
    for axis in range(3):
        p = removed.copy()
        p[axis] -= truncation_param
        new_points.append(p)
    new_points = np.asarray(new_points)

    kept = vertices[~mask]
    truncated = np.vstack([kept, new_points])
    return truncated, removed, new_points


def recenter_by_vertex_mean(vertices):
    """Match translate_to_origin() in the source: subtract arithmetic vertex mean."""
    vertices = np.asarray(vertices, dtype=float)
    return vertices - vertices.mean(axis=0)


def rescale_to_unit_volume(vertices):
    """Uniformly scale a convex vertex set so ConvexHull volume becomes 1."""
    vertices = np.asarray(vertices, dtype=float)
    hull = ConvexHull(vertices)
    scale = (1.0 / hull.volume) ** (1.0 / 3.0)
    return vertices * scale, hull.volume, scale


def convex_polydata(vertices):
    """Create a solid PyVista surface and only the true polyhedron feature edges."""
    vertices = np.asarray(vertices, dtype=float)
    hull = ConvexHull(vertices)

    faces = np.hstack(
        [[3, int(i), int(j), int(k)] for i, j, k in hull.simplices]
    ).astype(np.int64)

    poly = pv.PolyData(vertices, faces)
    poly = poly.compute_normals(
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
    )

    # Suppress the artificial diagonals generated when Qhull triangulates a
    # planar polygonal face. Only actual feature edges remain visible.
    edges = poly.extract_feature_edges(
        feature_angle=5,
        boundary_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
    )
    return poly, edges, hull.volume


def order_polygon(points):
    """Order coplanar points around their centroid and return a plane normal."""
    points = np.asarray(points, dtype=float)
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[2]
    b1, b2 = vt[0], vt[1]
    local = np.stack(
        [(points - centroid) @ b1, (points - centroid) @ b2], axis=1
    )
    order = np.argsort(np.arctan2(local[:, 1], local[:, 0]))
    return points[order], normal


def polygon_patch(points, offset=0.012):
    """Build a slightly outward-offset polygon patch to avoid z-fighting."""
    ordered, normal = order_polygon(points)
    centroid = ordered.mean(axis=0)

    # For this corner truncation, the outward direction points roughly toward
    # (+,+,+), i.e. toward the corner that was removed.
    outward = normal if np.dot(normal, centroid) > 0 else -normal
    patch_points = ordered + outward * offset

    faces = np.array([len(patch_points), *range(len(patch_points))], dtype=np.int64)
    return pv.PolyData(patch_points, faces=faces)


# -----------------------------------------------------------------------------
# 3. Rendering helpers
# -----------------------------------------------------------------------------
def style_mesh(plotter, poly, edges, color, opacity=1.0):
    plotter.add_mesh(
        poly,
        color=color,
        opacity=opacity,
        show_edges=False,
        smooth_shading=False,
        specular=0.25,
        specular_power=15,
        ambient=0.25,
        diffuse=0.8,
    )
    if opacity >= 0.999:
        plotter.add_mesh(edges, color=COL_EDGE, line_width=3)


def enable_antialiasing(plotter):
    """Enable SSAA while remaining compatible with multiple PyVista versions."""
    try:
        plotter.enable_anti_aliasing("ssaa", multi_samples=SAMPLES)
    except TypeError:
        plotter.enable_anti_aliasing("ssaa")


def new_plotter():
    p = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    p.background_color = BACKGROUND
    enable_antialiasing(p)
    return p


def set_common_camera(plotter):
    """Look directly toward the (+,+,+) corner so the truncation is obvious."""
    plotter.camera_position = [
        (3.25, 3.05, 2.75),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    plotter.camera.zoom(1.25)


def add_operation_geometry(plotter, removed_vertex, new_points, cut_patch):
    """Overlay the geometric construction of the one-corner truncation."""
    # The original corner that disappears.
    plotter.add_mesh(
        pv.PolyData(removed_vertex.reshape(1, 3)),
        color=COL_VANISH,
        render_points_as_spheres=True,
        point_size=30,
    )

    # Three cut-back segments, one along each incident cube edge.
    for p in new_points:
        plotter.add_mesh(
            pv.Line(removed_vertex, p),
            color=COL_NEW,
            line_width=7,
        )

    # New vertices created by the truncation.
    plotter.add_mesh(
        pv.PolyData(new_points),
        color=COL_NEW,
        render_points_as_spheres=True,
        point_size=22,
    )

    # The triangular cutting plane itself.
    plotter.add_mesh(
        cut_patch,
        color=COL_HIGHLIGHT_FACE,
        opacity=0.72,
        show_edges=True,
        edge_color=COL_HIGHLIGHT_FACE,
        line_width=4,
    )


# -----------------------------------------------------------------------------
# 4. Build figures
# -----------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----- Geometry exactly corresponding to the uploaded operation -----
    cube_verts = load_cube_vertices()
    trunc_raw, removed_vertex, new_points = truncate_one_positive_corner(
        cube_verts, TRUNCATION_PARAM
    )

    # Same recentering convention as translate_to_origin() in the source.
    trunc_centered = recenter_by_vertex_mean(trunc_raw)

    # Same volume-normalization idea as calculate_the_vertices(..., volume, 1.0).
    trunc_unit, pre_scale_volume, scale_factor = rescale_to_unit_volume(trunc_centered)

    cube_poly, cube_edges, _ = convex_polydata(cube_verts)
    trunc_poly_unit, trunc_edges_unit, final_volume = convex_polydata(trunc_unit)

    # Construction triangle in the ORIGINAL cube coordinates.
    op_patch = polygon_patch(new_points, offset=0.008)

    # Same triangle transformed exactly as the final truncated object.
    center_shift = trunc_raw.mean(axis=0)
    new_points_final = (new_points - center_shift) * scale_factor
    final_patch = polygon_patch(new_points_final, offset=0.012)

    # ------------------------------------------------------------------
    # Panel 1: original cube
    # ------------------------------------------------------------------
    p1 = new_plotter()
    style_mesh(p1, cube_poly, cube_edges, COL_CUBE)
    set_common_camera(p1)
    p1.screenshot(
        os.path.join(OUTPUT_DIR, "1_original_cube.png"),
        transparent_background=False,
    )
    p1.close()

    # ------------------------------------------------------------------
    # Panel 2: operation / construction
    # ------------------------------------------------------------------
    p2 = new_plotter()
    style_mesh(p2, cube_poly, cube_edges, COL_CUBE, opacity=0.28)
    p2.add_mesh(cube_edges, color=COL_EDGE, line_width=2, opacity=0.50)
    add_operation_geometry(p2, removed_vertex, new_points, op_patch)
    set_common_camera(p2)
    p2.screenshot(
        os.path.join(OUTPUT_DIR, "2_operation.png"),
        transparent_background=False,
    )
    p2.close()

    # ------------------------------------------------------------------
    # Panel 3: final unit-volume truncated cube
    # ------------------------------------------------------------------
    p3 = new_plotter()
    style_mesh(p3, trunc_poly_unit, trunc_edges_unit, COL_TRUNC)
    p3.add_mesh(
        final_patch,
        color=COL_HIGHLIGHT_FACE,
        opacity=1.0,
        show_edges=False,
    )
    set_common_camera(p3)
    p3.screenshot(
        os.path.join(OUTPUT_DIR, "3_truncated_cube.png"),
        transparent_background=False,
    )
    p3.close()

    # ------------------------------------------------------------------
    # Combined three-panel figure
    # ------------------------------------------------------------------
    pc = pv.Plotter(
        off_screen=True,
        shape=(1, 3),
        window_size=(WINDOW_SIZE[0] * 3, WINDOW_SIZE[1]),
    )
    pc.background_color = BACKGROUND
    enable_antialiasing(pc)

    pc.subplot(0, 0)
    style_mesh(pc, cube_poly, cube_edges, COL_CUBE)
    set_common_camera(pc)
    pc.add_text("1. Original cube", font_size=18, color="black")

    pc.subplot(0, 1)
    style_mesh(pc, cube_poly, cube_edges, COL_CUBE, opacity=0.28)
    pc.add_mesh(cube_edges, color=COL_EDGE, line_width=2, opacity=0.50)
    add_operation_geometry(pc, removed_vertex, new_points, op_patch)
    set_common_camera(pc)
    pc.add_text("2. Truncate one vertex", font_size=18, color="black")

    pc.subplot(0, 2)
    style_mesh(pc, trunc_poly_unit, trunc_edges_unit, COL_TRUNC)
    pc.add_mesh(
        final_patch,
        color=COL_HIGHLIGHT_FACE,
        opacity=1.0,
        show_edges=False,
    )
    set_common_camera(pc)
    pc.add_text("3. Resulting shape", font_size=18, color="black")

    pc.screenshot(
        os.path.join(OUTPUT_DIR, "combined_panel.png"),
        transparent_background=False,
    )
    pc.close()

    print("One-vertex truncation rendered successfully.")
    print("Removed vertex:", removed_vertex.tolist())
    print("New vertices before recenter/rescale:")
    for p in new_points:
        print("   ", p.tolist())
    print(f"Volume before unit-volume rescaling: {pre_scale_volume:.12f}")
    print(f"Uniform scale factor: {scale_factor:.12f}")
    print(f"Final convex-hull volume: {final_volume:.12f}")
    print("Images written to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
