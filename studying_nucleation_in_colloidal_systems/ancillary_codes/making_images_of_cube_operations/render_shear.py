"""
render_cube_shear_operation_no_arrows.py

Publication-quality rendering of the cube-shearing operation in shear_cube_test.py,
using only numpy, scipy, and pyvista (no euclid / coxeter dependency).

Produced figures:
  1_original_cube.png   - starting cube
  2_operation.png       - construction view: fixed face, original moving face,
                           moved face, and the displacement connectors
  3_sheared_cube.png    - final unit-volume sheared cube
  combined_panel.png    - all three panels side by side

Run:
    python3 render_cube_shear_operation_no_arrows.py

Headless Linux:
    xvfb-run -a python3 render_cube_shear_operation_no_arrows.py
"""

import glob
import json
import math
import os

import numpy as np
#   * pyvista: builds the 3-D meshes and renders the publication-style figures.
import pyvista as pv
from scipy.spatial import ConvexHull

# ----------------------------------------------------------------------
# 1. Parameters 
# ----------------------------------------------------------------------
THETA_DEG = 10.0
# All PNG files produced by the script are written into this directory.
# os.makedirs(..., exist_ok=True) in main() creates it automatically if needed.
OUTPUT_DIR = "output_sheared_cube_images"
# Each stand-alone panel is rendered at 1600 x 1600 pixels.  The combined
# three-panel figure later uses three times the horizontal resolution.
WINDOW_SIZE = (1600, 1600)
# Number of supersampling samples requested from PyVista/VTK for anti-aliasing.
# Higher values smooth jagged polygon edges at the cost of rendering work.
SAMPLES = 16

# ------------------------------------------------------------------------------
# Rendering colors
# ------------------------------------------------------------------------------
# Purple used for the original cube.
COL_CUBE = "#8A5FD1"
# Green used for the final, unit-volume sheared cube.
COL_FINAL = "#3E9C7E"
# Orange/red used in the OPERATION panel to identify the original moving face.
COL_HIGHLIGHT_FACE = "#E8552A"
# Blue used for the moved face, moved vertices, and displacement connectors.
COL_MOVED = "#1F6FEB"
# Dark neutral color used to mark the face that remains fixed during shear.
COL_FIXED = "#1A1A1A"
# Black is used for the visible feature edges of opaque polyhedra.
COL_EDGE = "black"
# White background keeps the output suitable for papers and presentations.
BACKGROUND = "white"


# ----------------------------------------------------------------------
# 2. Geometry helpers
# ----------------------------------------------------------------------
# ==============================================================================
# load_cube_vertices()
# ==============================================================================
# Purpose: obtain the eight vertices of the starting cube.
def load_cube_vertices():
    """Load the cube vertices from the standard JSON file if available,
    otherwise fall back to the hard-coded centered unit cube."""
    candidates = [
        "shape_023_Cube_unit_volume_principal_frame.json",
        *sorted(glob.glob("shape_023_Cube_unit_volume_principal_frame*.json")),
        "/mnt/data/shape_023_Cube_unit_volume_principal_frame.json",
        *sorted(glob.glob("/mnt/data/shape_023_Cube_unit_volume_principal_frame*.json")),
    ]
    # Examine candidates in order.  As soon as a real file is found, its
    # vertices are loaded and returned, so later candidates are not consulted.
    for path in candidates:
        if path and os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            return np.array(data["8_vertices"], dtype=float)

    # If no candidate JSON file was found, use an explicit centered unit cube.
    # Every possible sign combination of x,y,z = +/-0.5 appears exactly once.
    return np.array([
        [-0.5, -0.5, -0.5],
        [-0.5, -0.5,  0.5],
        [-0.5,  0.5, -0.5],
        [-0.5,  0.5,  0.5],
        [ 0.5, -0.5, -0.5],
        [ 0.5, -0.5,  0.5],
        [ 0.5,  0.5, -0.5],
        [ 0.5,  0.5,  0.5],
    ], dtype=float)


# ==============================================================================
# recenter()
# ==============================================================================
# A geometric translation that moves the arithmetic mean of the supplied
# vertices to the origin.  This does NOT rotate or distort the polyhedron.
#
# If C = (1/N) sum_i r_i is the current vertex-average center, this returns
#     r_i' = r_i - C
# for every vertex i.
def recenter(vertices):
    return vertices - vertices.mean(axis=0)

# ==============================================================================
# rescale_to_unit_volume()
# ==============================================================================
# Uniformly scales all three Cartesian directions by one scalar factor so that
# the final polyhedron has volume exactly 1 (up to floating-point precision).
#
# In three dimensions, multiplying every length by s multiplies volume by s^3.
# Therefore, for an existing volume V, choosing
#     s = (1/V)^(1/3)
# gives V_new = V*s^3 = 1.
def rescale_to_unit_volume(vertices, volume):
    return vertices * (1.0 / volume) ** (1.0 / 3.0)


# ==============================================================================
# convex_polydata()
# ==============================================================================
# Convert an unordered set of convex-polyhedron vertices into two renderable
# PyVista objects:
#   1. `poly`  : the filled triangulated surface, used to color/shade the solid.
#   2. `edges` : only the true sharp feature edges, used for clean black outlines.
#
# The function also returns the ConvexHull volume.  That volume is later used
# to restore the sheared shape to unit volume.
def convex_polydata(vertices):
    """Build a smooth-seam-free PyVista mesh from a convex point cloud."""
    # SciPy/Qhull determines which points lie on the convex boundary, the
    # triangular facets connecting them, and the enclosed three-dimensional volume.
    hull = ConvexHull(vertices)
    # hull.simplices contains triangle vertex indices with shape (N_triangles, 3).
    # PyVista's face-array convention requires each triangle to be encoded as
    #     [3, i, j, k]
    # where the leading 3 states that three vertices belong to the face.
    # np.hstack concatenates all encoded triangles into one integer array.
    faces = np.hstack([[3, *tri] for tri in hull.simplices]).astype(np.int64)
    # Build a PyVista PolyData surface using the original vertex coordinates and
    # the triangle connectivity generated by ConvexHull.
    poly = pv.PolyData(vertices, faces)
    # Recompute surface normals so lighting is consistent across the hull.
    # This is a rendering operation; it does not modify the intended geometry.
    poly = poly.compute_normals(
        # Orient normals automatically so they point consistently outward/inward.
        auto_orient_normals=True,
        # Force neighboring triangles to use a mutually consistent orientation.
        consistent_normals=True,
        # Do not duplicate vertices merely to split normals at sharp boundaries.
        split_vertices=False,
    )
    # ConvexHull triangulates every polygonal face.  If all triangle edges were
    # displayed, a square cube face would show an unwanted diagonal.  Therefore
    # extract_feature_edges keeps only boundaries where neighboring triangles meet
    # at a sufficiently non-flat angle, i.e. the visually meaningful polyhedron edges.
    edges = poly.extract_feature_edges(
        # A very small 5-degree threshold treats coplanar triangulation seams as flat
        # while retaining the genuine sharp edges of the cube/sheared cube.
        feature_angle=5,
        # Closed convex polyhedra should have no open boundary edges; omit them.
        boundary_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
    )
    return poly, edges, hull.volume


# ==============================================================================
# order_polygon()
# ==============================================================================
# PyVista needs a face's vertices in cyclic perimeter order to draw one polygon.
def order_polygon(points):
    """Order coplanar points around their centroid."""
    # Arithmetic center of the face; used as the origin for angular sorting.
    centroid = points.mean(axis=0)
    # Subtracting the centroid makes the point cloud centered.  SVD of this
    # centered cloud identifies principal in-plane directions and the plane normal.
    # The first two returned quantities are intentionally discarded with `_`.
    _, _, vt = np.linalg.svd(points - centroid)
    # For coplanar points in 3-D, the least-variance singular direction is normal
    # to the plane, hence the third right-singular vector vt[2].
    normal = vt[2]
    # These two orthonormal vectors form local x/y axes inside the face plane.
    b1, b2 = vt[0], vt[1]
    # Project each 3-D centered point into that local 2-D coordinate system.
    local = np.stack([
        (points - centroid) @ b1,
        (points - centroid) @ b2,
    ], axis=1)
    order = np.argsort(np.arctan2(local[:, 1], local[:, 0]))
    return points[order], normal


# ==============================================================================
# polygon_patch()
# ==============================================================================
# Build a single filled PyVista polygon from a set of coplanar face vertices.
# These patches are visualization overlays used in the construction panel.
# `outward_push` can move the patch a tiny distance away from the solid surface
# to prevent z-fighting (two surfaces occupying almost exactly the same pixels).
def polygon_patch(points, outward_push=0.0):
    """Create a polygonal patch from an ordered face point set."""
    # First put the supplied face vertices into cyclic order and obtain its normal.
    ordered, normal = order_polygon(points)
    # SVD normals have an arbitrary sign: n and -n describe the same plane.
    # Choose the sign whose dot product with the face centroid is positive, which
    # points approximately away from the origin for this centered convex geometry.
    outward = normal if np.dot(normal, ordered.mean(axis=0)) > 0 else -normal
    # Shift the overlay slightly along the selected outward normal.
    # A value of zero would leave it exactly coplanar with the underlying face.
    patch_pts = ordered + outward_push * outward
    # Construct a single polygon cell.  PyVista encodes it as
    #     [number_of_vertices, index_0, index_1, ...].
    patch = pv.PolyData(
        patch_pts,
        faces=np.array([len(patch_pts), *range(len(patch_pts))]),
    )
    # Return the renderable patch plus the ordered points and chosen outward normal.
    return patch, ordered, outward


# ==============================================================================
# top_and_bottom_faces()
# ==============================================================================
# Split the principal-frame cube into the two faces normal to the z axis.
# The minimum-z face is treated as the fixed face; the maximum-z face is the one
# displaced by the shear/tilt operation.
#
# `tol` makes the test robust to tiny floating-point deviations from exact z values.
def top_and_bottom_faces(vertices, tol=1e-8):
    """Identify the two opposite z-normal faces of the original cube.

    This matches the effect of shear_cube_test.py for the standard principal-frame cube:
    the z-min face stays fixed and the z-max face is sheared.
    """
    # Extract all z coordinates as a one-dimensional array.
    z = vertices[:, 2]
    # Lowest z value defines the plane of the bottom face.
    zmin = np.min(z)
    # Highest z value defines the plane of the top face.
    zmax = np.max(z)
    # Boolean indexing selects every vertex whose z coordinate is essentially zmin.
    bottom = vertices[np.abs(z - zmin) < tol]
    top = vertices[np.abs(z - zmax) < tol]
    return bottom, top


# ==============================================================================
# shear_cube_operation() -- CORE GEOMETRIC TRANSFORMATION
# ==============================================================================
# This is the key mathematical step.  It constructs the modified cube BEFORE
# anything is rendered.  For the standard cube, the z-min face stays fixed and
# every vertex of the z-max face receives the SAME translation vector.
#
# Given angle theta, the translation is
#
#     Delta r = [ sin(theta), 0, -(1 - cos(theta)) ].
#
# Consequently a top vertex [x, y, z] becomes
#
#     [x + sin(theta),
#      y,
#      z - (1 - cos(theta))].
#
# For an original vertical side-edge vector [0,0,1], the new connector vector is
# [sin(theta), 0, cos(theta)].  Its length is
# sqrt(sin^2(theta)+cos^2(theta)) = 1, so the connector length is preserved while
# it is tilted toward +x.  The y coordinates are untouched.
#
# After constructing this raw shape, the code recenters it and uniformly scales
# it to volume 1.  These final two operations preserve the shape's shear character.
def shear_cube_operation(cube_vertices, theta_deg):
    """Apply the same geometric operation as shear_cube_test.py.

    For the standard centered cube:
    - the z-min face is kept fixed,
    - the opposite z-max face is translated by
          (+sin(theta), 0, -(1-cos(theta)))
      which keeps the side-edge length equal to 1.

    Returns both the pre- and post-normalization geometry for visualization.
    """
    # Python's math.sin/math.cos expect radians, whereas THETA_DEG is stored in
    # degrees.  Convert once and use this radian value consistently below.
    theta = math.radians(theta_deg)
    # Build the one translation vector that will be added to ALL four top vertices.
    shift = np.array([
        # +x displacement: horizontal component of the unit tilted connector.
        math.sin(theta),
        # No y displacement: the shear lies entirely in the x-z plane.
        0.0,
        # z correction: z_new = z - (1-cos(theta)); equivalently the connector
        # between corresponding bottom/top points has z component cos(theta).
        -(1.0 - math.cos(theta)),
    ])

    # Identify the four vertices that remain fixed and the four that will move.
    bottom, top = top_and_bottom_faces(cube_vertices)
    # NumPy broadcasting adds the same three-component shift vector to every
    # row of `top`, producing the displaced opposite face.
    moved_top = top + shift

    # The new convex solid is completely determined by the four untouched bottom
    # vertices plus the four displaced top vertices.
    # Assemble the raw sheared shape.
    # Stack them into one (8,3) vertex array.  No additional vertices are created.
    raw_vertices = np.vstack([bottom, moved_top])

    # The shear changes the arithmetic center of the vertex set.  Translate that
    # center back to [0,0,0] before normalizing the size.
    # Recenter and rescale to unit volume, matching the workflow in the source script.
    # This is a pure translation; pairwise distances and volume are unchanged.
    centered_vertices = recenter(raw_vertices)
    # Build a temporary convex hull solely to obtain the volume of the centered
    # raw sheared shape.  The surface and edges returned here are intentionally ignored.
    _, _, raw_volume = convex_polydata(centered_vertices)
    # Apply the uniform cube-root volume correction so the final shape encloses
    # unit volume.  This is the geometry rendered in the final panel.
    unit_vertices = rescale_to_unit_volume(centered_vertices, raw_volume)

    # The next quantities propagate the specifically important bottom/moved-top
    # face coordinates through EXACTLY the same translation and scale.  This lets
    # those faces be overlaid consistently when needed for visualization.
    # Track the two important faces through recentering and scaling.
    # Center used to translate the complete raw vertex set to the origin.
    raw_center = raw_vertices.mean(axis=0)
    # Apply that same translation to the fixed bottom face.
    bottom_centered = bottom - raw_center
    # Apply that same translation to the displaced top face.
    moved_top_centered = moved_top - raw_center
    # Explicitly store the linear scale s = V^(-1/3) for reuse/diagnostics.
    scale = (1.0 / raw_volume) ** (1.0 / 3.0)
    # Coordinates of the fixed face in the final unit-volume reference frame.
    bottom_unit = bottom_centered * scale
    # Coordinates of the moved face in the final unit-volume reference frame.
    moved_top_unit = moved_top_centered * scale

    # Package every useful stage of the transformation in a dictionary.  Keeping
    # these intermediate arrays makes the later rendering code readable and avoids
    # recomputing the geometry separately for different panels.
    return {
        # Four untouched vertices of the original fixed face.
        "bottom": bottom,
        # Four original vertices before they are moved.
        "top": top,
        # Four vertices after applying the shear displacement.
        "moved_top": moved_top,
        # Eight-vertex sheared solid before recentering/volume normalization.
        "raw_vertices": raw_vertices,
        # Same raw solid translated so its vertex-average center is at the origin.
        "centered_vertices": centered_vertices,
        # Final centered, uniformly rescaled, unit-volume sheared solid.
        "unit_vertices": unit_vertices,
        # Fixed face propagated into the final normalized coordinates.
        "bottom_unit": bottom_unit,
        # Moved face propagated into the final normalized coordinates.
        "moved_top_unit": moved_top_unit,
        # The exact [dx,dy,dz] displacement vector applied to the top face.
        "shift": shift,
        # Convex-hull volume before unit-volume rescaling.
        "raw_volume": raw_volume,
        # Uniform linear factor used to turn raw_volume into volume 1.
        "scale": scale,
    }


# ----------------------------------------------------------------------
# 3. Scene-building helpers
# ----------------------------------------------------------------------
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



def new_plotter(shape=None, window_size=None):
    if shape is None:
        p = pv.Plotter(off_screen=True, window_size=window_size or WINDOW_SIZE)
    else:
        p = pv.Plotter(off_screen=True, shape=shape, window_size=window_size)
    p.background_color = BACKGROUND
    p.enable_anti_aliasing("ssaa", multi_samples=SAMPLES)
    return p



def set_common_camera(plotter):
    # Oblique view that clearly shows both the fixed base and the displaced top face.
    plotter.camera_position = [(3.2, -3.0, 2.2), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    plotter.camera.zoom(1.18)


# ----------------------------------------------------------------------
# 4. Build the three panels
# ----------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cube_vertices = load_cube_vertices()
    cube_poly, cube_edges, _ = convex_polydata(cube_vertices)

    op = shear_cube_operation(cube_vertices, THETA_DEG)
    sheared_poly, sheared_edges, _ = convex_polydata(op["unit_vertices"])

    bottom_patch, _, _ = polygon_patch(op["bottom"], outward_push=2e-2)
    top_patch, _, _ = polygon_patch(op["top"], outward_push=2e-2)
    moved_top_patch, _, _ = polygon_patch(op["moved_top"], outward_push=2e-2)
    moved_top_patch_unit, _, _ = polygon_patch(op["moved_top_unit"], outward_push=2e-2)

    # ---------- Panel 1: original cube ----------
    p1 = new_plotter()
    style_mesh(p1, cube_poly, cube_edges, COL_CUBE)
    set_common_camera(p1)
    p1.screenshot(os.path.join(OUTPUT_DIR, "1_original_cube.png"), transparent_background=False)
    p1.close()

    # ---------- Panel 2: the operation ----------
    p2 = new_plotter()
    style_mesh(p2, cube_poly, cube_edges, COL_CUBE, opacity=0.25)
    p2.add_mesh(cube_edges, color=COL_EDGE, line_width=2, opacity=0.5)

    # Fixed face (bottom) highlighted subtly.
    # p2.add_mesh(bottom_patch, color=COL_FIXED, opacity=0.16)

    # Original top face to be moved.
    # p2.add_mesh(top_patch, color=COL_HIGHLIGHT_FACE, opacity=0.22)
    p2.add_mesh(pv.PolyData(op["top"]), color=COL_HIGHLIGHT_FACE,
                render_points_as_spheres=True, point_size=20)

    # New top face after shearing.
    p2.add_mesh(moved_top_patch, color=COL_MOVED, opacity=0.32)
    p2.add_mesh(pv.PolyData(op["moved_top"]), color=COL_MOVED,
                render_points_as_spheres=True, point_size=20)

    # Connector lines showing the displacement of each moved vertex.
    for p_old, p_new in zip(op["top"], op["moved_top"]):
        p2.add_mesh(pv.Line(p_old, p_new), color=COL_MOVED, line_width=6)

    set_common_camera(p2)
    p2.add_text(f"theta = {THETA_DEG:g} deg", position="upper_right", font_size=15, color="black")
    p2.screenshot(os.path.join(OUTPUT_DIR, "2_operation.png"), transparent_background=False)
    p2.close()

    # ---------- Panel 3: final unit-volume sheared cube ----------
    p3 = new_plotter()
    style_mesh(p3, sheared_poly, sheared_edges, COL_FINAL)
    # Final shape shown without the red highlighted top face.
    set_common_camera(p3)
    p3.screenshot(os.path.join(OUTPUT_DIR, "3_sheared_cube.png"), transparent_background=False)
    p3.close()

    # ---------- Combined 3-panel figure ----------
    pc = new_plotter(shape=(1, 3), window_size=(WINDOW_SIZE[0] * 3, WINDOW_SIZE[1]))

    pc.subplot(0, 0)
    style_mesh(pc, cube_poly, cube_edges, COL_CUBE)
    set_common_camera(pc)
    pc.add_text("1. Original cube", font_size=18, color="black")

    pc.subplot(0, 1)
    style_mesh(pc, cube_poly, cube_edges, COL_CUBE, opacity=0.25)
    pc.add_mesh(cube_edges, color=COL_EDGE, line_width=2, opacity=0.5)
    pc.add_mesh(bottom_patch, color=COL_FIXED, opacity=0.16)
    pc.add_mesh(top_patch, color=COL_HIGHLIGHT_FACE, opacity=0.22)
    pc.add_mesh(pv.PolyData(op["top"]), color=COL_HIGHLIGHT_FACE,
                render_points_as_spheres=True, point_size=20)
    pc.add_mesh(moved_top_patch, color=COL_MOVED, opacity=0.32)
    pc.add_mesh(pv.PolyData(op["moved_top"]), color=COL_MOVED,
                render_points_as_spheres=True, point_size=20)
    for p_old, p_new in zip(op["top"], op["moved_top"]):
        pc.add_mesh(pv.Line(p_old, p_new), color=COL_MOVED, line_width=6)
    set_common_camera(pc)
    pc.add_text(f"2. Shear top face  (theta = {THETA_DEG:g} deg)", font_size=18, color="black")

    pc.subplot(0, 2)
    style_mesh(pc, sheared_poly, sheared_edges, COL_FINAL)
    # Final shape shown without the red highlighted top face.
    set_common_camera(pc)
    pc.add_text("3. Resulting shape", font_size=18, color="black")

    pc.screenshot(os.path.join(OUTPUT_DIR, "combined_panel.png"), transparent_background=False)
    pc.close()

    print("Done. Images written to:", os.path.abspath(OUTPUT_DIR))
    print("Theta (deg):", THETA_DEG)
    print("Shift applied to the moving face:", op["shift"])
    print("Raw volume before unit-volume rescaling:", op["raw_volume"])
    print("Scale factor to unit volume:", op["scale"])


if __name__ == "__main__":
    main()