"""
render_edge_truncation.py

Publication-quality rendering of a single-edge cube truncation using PyVista.

Produces three figures:
  1_original_cube.png       - the starting cube
  2_operation.png           - the cube (translucent) showing the target edge,
                               the two vertices that vanish, and the new
                               vertices the truncation introduces
  3_truncated_cube.png      - the final truncated shape, unit-volume rescaled,
                               with the newly created face tinted for clarity
  combined_panel.png        - all three side by side in one figure

Only needs: numpy, scipy, pyvista  (no coxeter / euclid dependency)

USAGE
-----
    python3 render_edge_truncation.py

On a headless Linux machine with no display, run it under a virtual
framebuffer instead:

    sudo apt-get install -y xvfb          # one-time
    xvfb-run -a python3 render_edge_truncation.py

On a normal desktop (Windows/macOS/Linux with a display or GPU) you can just
run it directly.
"""

import json
import os

import numpy as np
import pyvista as pv
from scipy.spatial import ConvexHull

# ----------------------------------------------------------------------
# 1. Parameters (mirrors the original truncating_one_edge_of_cube.py)
# ----------------------------------------------------------------------
CUBE_JSON = "shape_023_Cube_unit_volume_principal_frame.json"
TRUNCATION_PARAM = 0.50          # distance to cut back along each edge
PRECISION = 1e-4
OUTPUT_DIR = "output_images"
WINDOW_SIZE = (1600, 1600)       # per-panel render resolution
SAMPLES = 16                     # anti-aliasing samples

# Color palette
COL_CUBE = "#8A5FD1"
COL_TRUNC = "#3E9C7E"
COL_HIGHLIGHT_FACE = "#E8552A"
COL_VANISH = "#1A1A1A"
COL_NEW = "#1F6FEB"
COL_EDGE = "black"
BACKGROUND = "white"


# ----------------------------------------------------------------------
# 2. Geometry helpers
# ----------------------------------------------------------------------
def load_cube_vertices():
    """Load the unit-volume cube from JSON, falling back to a hard-coded
    unit cube (side length 1, centered at the origin) if the file isn't
    found, so the script is fully standalone."""
    if os.path.exists(CUBE_JSON):
        with open(CUBE_JSON) as f:
            data = json.load(f)
        return np.array(data["8_vertices"], dtype=float)
    return np.array([[x, y, z] for x in (-0.5, 0.5)
                                for y in (-0.5, 0.5)
                                for z in (-0.5, 0.5)])


def truncate_one_edge(verts, truncation_param, precision=PRECISION):
    """Truncate the single cube edge that sits at x>0, y>0 (i.e. the edge
    joining the two vertices with x>0 and y>0). Faithful, cleaned-up
    reimplementation of the logic in truncating_one_edge_of_cube.py, with
    no dependency on euclid/coxeter.

    Returns
    -------
    new_verts : (N,3) array          vertices of the truncated polyhedron
    v_hi, v_lo : (3,) arrays          the two vertices that vanish
    new_pts : (4,3) array             the 4 new vertices introduced by the cut
    """
    verts = [np.asarray(v, dtype=float) for v in verts]
    vanish = [v for v in verts if v[0] > 0 and v[1] > 0]
    if len(vanish) != 2:
        raise ValueError("Expected exactly 2 vertices with x>0 and y>0.")
    v_hi, v_lo = vanish
    edge_len = np.linalg.norm(v_hi - v_lo)

    new_verts = []
    new_pts = []
    for v in verts:
        if any(np.allclose(v, vv) for vv in vanish):
            continue
        d_hi = np.linalg.norm(v - v_hi)
        d_lo = np.linalg.norm(v - v_lo)
        if abs(d_hi - edge_len) < precision:
            t = truncation_param / d_hi
            p = v_hi + t * (v - v_hi)
            new_verts.append(p)
            new_verts.append(v)
            new_pts.append(p)
        elif abs(d_lo - edge_len) < precision:
            t = truncation_param / d_lo
            p = v_lo + t * (v - v_lo)
            new_verts.append(p)
            new_verts.append(v)
            new_pts.append(p)
        else:
            new_verts.append(v)

    return np.array(new_verts), v_hi, v_lo, np.array(new_pts)


def recenter(verts):
    return verts - verts.mean(axis=0)


def rescale_to_unit_volume(verts, faces_hull_volume):
    return verts * (1.0 / faces_hull_volume) ** (1.0 / 3.0)


def convex_polydata(vertices):
    """Build a PyVista mesh from a convex point cloud, plus a clean set of
    'feature edges' (the true polyhedron edges) so flat faces render
    without visible qhull triangulation seams."""
    hull = ConvexHull(vertices)
    faces = np.hstack([[3, *tri] for tri in hull.simplices]).astype(np.int64)
    poly = pv.PolyData(vertices, faces)
    poly = poly.compute_normals(auto_orient_normals=True,
                                 consistent_normals=True,
                                 split_vertices=False)
    edges = poly.extract_feature_edges(
        feature_angle=5, boundary_edges=False,
        non_manifold_edges=False, manifold_edges=False,
    )
    return poly, edges, hull.volume


def order_polygon(points):
    """Order a set of coplanar points around their centroid so they form a
    proper (non-self-intersecting) polygon, and return the plane normal."""
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid)
    normal = vt[2]
    b1, b2 = vt[0], vt[1]
    local = np.stack([(points - centroid) @ b1, (points - centroid) @ b2], axis=1)
    order = np.argsort(np.arctan2(local[:, 1], local[:, 0]))
    return points[order], normal


# ----------------------------------------------------------------------
# 3. Scene-building helpers
# ----------------------------------------------------------------------
def style_mesh(plotter, poly, edges, color, opacity=1.0):
    plotter.add_mesh(poly, color=color, opacity=opacity, show_edges=False,
                      smooth_shading=False, specular=0.25, specular_power=15,
                      ambient=0.25, diffuse=0.8)
    if opacity >= 0.999:
        plotter.add_mesh(edges, color=COL_EDGE, line_width=3)


def new_plotter():
    p = pv.Plotter(off_screen=True, window_size=WINDOW_SIZE)
    p.background_color = BACKGROUND
    p.enable_anti_aliasing("ssaa", multi_samples=SAMPLES)
    return p


def set_common_camera(plotter):
    # Positive-x, positive-y, elevated view: looks straight at the (x>0, y>0)
    # edge that gets truncated, so the operation and the new face are visible
    # (not hidden on the far side of the shape).
    plotter.camera_position = [(3.4, 3.0, 2.0), (0, 0, 0), (0, 0, 1)]
    plotter.camera.zoom(1.25)


# ----------------------------------------------------------------------
# 4. Build the three panels
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cube_verts = load_cube_vertices()
    trunc_verts_raw, v_hi, v_lo, new_pts = truncate_one_edge(cube_verts, TRUNCATION_PARAM)
    trunc_verts = recenter(trunc_verts_raw)

    cube_poly, cube_edges, cube_vol = convex_polydata(cube_verts)
    trunc_poly, trunc_edges, trunc_vol = convex_polydata(trunc_verts)
    trunc_verts_unit = rescale_to_unit_volume(trunc_verts, trunc_vol)
    trunc_poly_unit, trunc_edges_unit, _ = convex_polydata(trunc_verts_unit)

    # ---------- Panel 1: original cube ----------
    p1 = new_plotter()
    style_mesh(p1, cube_poly, cube_edges, COL_CUBE)
    set_common_camera(p1)
    p1.screenshot(os.path.join(OUTPUT_DIR, "1_original_cube.png"), transparent_background=False)
    p1.close()

    # ---------- Panel 2: the operation ----------
    p2 = new_plotter()
    style_mesh(p2, cube_poly, cube_edges, COL_CUBE, opacity=0.30)
    p2.add_mesh(cube_edges, color=COL_EDGE, line_width=2, opacity=0.5)

    edge_line = pv.Line(v_hi, v_lo)
    p2.add_mesh(edge_line, color=COL_HIGHLIGHT_FACE, line_width=20)
    p2.add_mesh(pv.PolyData(np.vstack([v_hi, v_lo])), color=COL_VANISH,
                render_points_as_spheres=True, point_size=28)

    for v_end in (v_hi, v_lo):
        for p in new_pts:
            if abs(np.linalg.norm(p - v_end) - TRUNCATION_PARAM) < 1e-6:
                p2.add_mesh(pv.Line(v_end, p), color=COL_NEW, line_width=19)
    p2.add_mesh(pv.PolyData(new_pts), color=COL_NEW,
                render_points_as_spheres=True, point_size=20)

    set_common_camera(p2)
    p2.screenshot(os.path.join(OUTPUT_DIR, "2_operation.png"), transparent_background=False)
    p2.close()

    # ---------- Panel 3: final truncated shape ----------
    p3 = new_plotter()
    style_mesh(p3, trunc_poly_unit, trunc_edges_unit, COL_TRUNC)

    new_pts_centered = new_pts - trunc_verts_raw.mean(axis=0)
    new_pts_unit = new_pts_centered * (1.0 / trunc_vol) ** (1.0 / 3.0)
    ordered, normal = order_polygon(new_pts_unit)
    outward = normal if np.dot(normal, ordered.mean(axis=0)) > 0 else -normal
    patch_pts = ordered + outward * 2e-2
    patch = pv.PolyData(patch_pts, faces=np.array([len(patch_pts), *range(len(patch_pts))]))
    p3.add_mesh(patch, color=COL_HIGHLIGHT_FACE, opacity=1.0)

    set_common_camera(p3)
    p3.screenshot(os.path.join(OUTPUT_DIR, "3_truncated_cube.png"), transparent_background=False)
    p3.close()

    # ---------- Combined 3-panel figure ----------
    pc = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(WINDOW_SIZE[0] * 3, WINDOW_SIZE[1]))
    pc.background_color = BACKGROUND
    pc.enable_anti_aliasing("ssaa", multi_samples=SAMPLES)

    pc.subplot(0, 0)
    style_mesh(pc, cube_poly, cube_edges, COL_CUBE)
    set_common_camera(pc)
    pc.add_text("1. Original cube", font_size=18, color="black")

    pc.subplot(0, 1)
    style_mesh(pc, cube_poly, cube_edges, COL_CUBE, opacity=0.30)
    pc.add_mesh(cube_edges, color=COL_EDGE, line_width=2, opacity=0.5)
    pc.add_mesh(edge_line, color=COL_HIGHLIGHT_FACE, line_width=10)
    pc.add_mesh(pv.PolyData(np.vstack([v_hi, v_lo])), color=COL_VANISH,
                render_points_as_spheres=True, point_size=28)
    for v_end in (v_hi, v_lo):
        for p in new_pts:
            if abs(np.linalg.norm(p - v_end) - TRUNCATION_PARAM) < 1e-6:
                pc.add_mesh(pv.Line(v_end, p), color=COL_NEW, line_width=6)
    pc.add_mesh(pv.PolyData(new_pts), color=COL_NEW,
                render_points_as_spheres=True, point_size=20)
    set_common_camera(pc)
    pc.add_text("2. Truncate the edge", font_size=18, color="black")

    pc.subplot(0, 2)
    style_mesh(pc, trunc_poly_unit, trunc_edges_unit, COL_TRUNC)
    pc.add_mesh(patch, color=COL_HIGHLIGHT_FACE, opacity=1.0)
    set_common_camera(pc)
    pc.add_text("3. Resulting shape", font_size=18, color="black")

    pc.screenshot(os.path.join(OUTPUT_DIR, "combined_panel.png"), transparent_background=False)
    pc.close()

    print("Done. Images written to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
