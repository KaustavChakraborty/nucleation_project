#!/usr/bin/env python3
"""
Render proper rotational axes and sigma_h mirror planes of a convex polyhedron, preserving the sample-style glossy axes/opaque body look.

python3 render_polyhedron_axes_and_sigma_h.py shape_151_sheared_Cube_deg10_unit_volume_principal_frame.json  --n-views 10 --output sheared_cube.png --resolution 2400 --axis-length 1.55 --axis-thickness 0.012 --seed 42

* Each screenshot has a genuinely transparent (alpha-channel) background.
* Axis colors remain sample-style by rotational order (e.g. C2 purple, C3 orange, C4 green, C5 brown); the body remains blue.
* True reflection planes are found using improper orthogonal vertex symmetries.
* Horizontal mirrors (sigma_h) are perpendicular to the highest-order C_n axes.
* Sigma planes are rendered as clean translucent plane patches by default, with optional section mode.
* Use --plane-mode all to show vertical/dihedral mirror planes as well.
* Random viewing directions are uniformly distributed over a sphere.
* N is supplied by --n-views, or requested interactively at the terminal.
* A --seed permits reproducible viewpoints. Camera metadata are saved to JSON.
* A fresh PyVista scene is built for EACH image to prevent stale cached frames.

Dependencies: numpy, scipy, pyvista (Python 3.8+).

Examples:
    python3 render_polyhedron_axes_and_sigma_h.py shape.json --n-views 10
    python3 render_polyhedron_axes_and_sigma_h.py shape.json \
        --n-views 20 --seed 42 --output figures/sheared_cube.png \
        --resolution 2400 --axis-length 1.55 --axis-thickness 0.012
    python3 render_polyhedron_axes_and_sigma_h.py shape.json --check-only

Output for --output figures/sheared_cube.png and --n-views 3:
    figures/sheared_cube_view_001.png
    figures/sheared_cube_view_002.png
    figures/sheared_cube_view_003.png
    figures/sheared_cube.axes.json  (also contains mirror-plane normals)
    figures/sheared_cube.views.json

Mirror planes are verified as *geometric* symmetries, not guessed from visible
faces. The term sigma_h is relative to a chosen principal rotational direction;
by default all axes of maximal order are used, including equivalent axes in
high-symmetry bodies. For a nonstandard convention select --horizontal-reference all.
Each camera view uses an independent renderer to avoid stale screenshots.
"""

# ===============================================================================
# READ THIS FIRST: EXTENSION AND MODIFICATION HANDBOOK
# ===============================================================================
# This additional documentation changes NO executable statement in the original.
# Everything between the original module docstring above and the main program below
# is the original program, with extensive inline explanatory comments inserted.
#
# HIGH-LEVEL PURPOSE
#   Input   : a JSON file containing >=4 points in R^3 forming a 3D convex hull.
#   Analyze : find geometric rotational symmetries, extract C_n axes; find actual
#             reflection planes; classify selected mirrors as sigma_h.
#   Render  : build a lit PyVista scene, show physical faces/edges, rotation-axis
#             cylinders, and optional translucent mirror patches.
#   Output  : transparent PNG views plus .axes.json and .views.json metadata.
#
# START HERE / CALL GRAPH (functions occur below in roughly pipeline order)
#   if __name__ == '__main__'
#     -> main()
#        -> parse_arguments()                          # all CLI controls
#        -> read_vertices()                            # JSON -> N x 3 vertices
#        -> polygon_faces_from_hull()                  # real faces and edges
#        -> find_proper_rotations()                     # 3x3 det+1 matrices
#        -> symmetry_axes()                            # C_n axes and orders
#        -> find_mirror_planes()
#             -> find_one_improper_symmetry()           # one det-1 operation
#             -> canonical_axis()                       # stable normal sign
#        -> classify_mirror_planes()                   # which are sigma_h?
#        -> save_report()                              # .axes.json (always)
#        -> render_images()                            # unless --check-only
#             -> random_camera_directions()
#             -> create_scene()                        # PyVista geometry/materials
#                  -> axis_color_for_order()
#                  -> mirror_rectangle_patch() / etc. # visible finite patches
#             -> set_camera_for_direction()
#             -> screenshot() and .views.json
#
# CORE CONCEPTS AND UNITS
#   vertices : NumPy float ndarray, shape (N, 3), one [x,y,z] per row.
#   center   : arithmetic mean of hull VERTICES (not solid volume centroid).
#   p        : centered vertices = vertices - center, shape (N, 3).
#   radius   : max Euclidean distance of any vertex from center.
#   face     : a physical planar polygon, distinct from triangulated hull mesh.
#   edge     : pair of vertex indices (i,j), one undirected physical boundary.
#   R        : 3x3 proper orthogonal matrix, R.T @ R = I, det(R) = +1.
#   Q        : 3x3 improper orthogonal matrix, det(Q) = -1.
#   axis     : undirected LINE through center; u and -u mean the same line.
#   C_n      : rotations of 360/n degrees (and their powers) around one line.
#   mirror   : plane n dot (x-center) = 0; normal n and -n mean same plane.
#   sigma_h  : a mirror perpendicular to one SELECTED reference C_n axis.
#   rel_tol  : fraction of characteristic body size, not a world-unit distance.
#   *_opacity: in [0,1], where 0 is invisible and 1 is opaque.
#   *_color  : colors passed to VTK/PyVista (e.g., hexadecimal '#RRGGBB').
#
# COMMON MODIFICATIONS: CHANGE ONLY THE INDICATED LOCATION
#   - Change default object color/translucency:
#       parse_arguments(): --face-color, --opacity; surface add_mesh in create_scene.
#   - Change color by C_n order:
#       DEFAULT_AXIS_COLORS; --c2-color ... --c6-color; axis_color_for_order.
#   - Add C7 or another new order's CLI override:
#       add --c7-color to parse_arguments; axis_color_for_order already uses
#       getattr(args, 'c7_color') if that argument exists, else fallback.
#   - Change cylinder length/thickness:
#       --axis-length / --axis-thickness; create_scene cylinder and cap multipliers.
#   - Make mirror planes larger or change their outline:
#       --plane-span, --plane-aspect, --plane-border-radius, --plane-opacity.
#   - Show all mirrors, none, or only sigma_h:
#       --plane-mode all / none / horizontal. Detection is STILL performed.
#   - Alter what 'horizontal' means:
#       --horizontal-reference maximum/all; --principal-axis-index K;
#       classify_mirror_planes contains the exact alignment condition.
#   - Choose plane polygon style:
#       --plane-style rectangle/disc/square/section; edit the appropriate
#       mirror_*_patch function for geometry. Add a matching branch in BOTH
#       create_scene's render loop and its duplicated label-placement loop.
#   - Change camera angle sampling or projection:
#       random_camera_directions and set_camera_for_direction.
#   - Change image dimensions/count/filename:
#       --resolution, --height, --n-views, --output; render_images.
#   - Change lights and highlights:
#       create_scene: light_specs and the surface/cylinder add_mesh properties.
#   - Support different input JSON schemas:
#       read_vertices: the ordered keys list and shape validation.
#   - Change tolerance for recognizing approximate symmetry:
#       --symmetry-tol; find_proper_rotations and find_one_improper_symmetry.
#       WARNING: relaxing tolerances can falsely report symmetries.
#   - Add JSON fields or change how the report is structured:
#       save_report for .axes.json; render_images for .views.json.
#   - Select deterministic, manually chosen viewpoints:
#       replace the directions array produced in render_images, keeping each
#       direction normalized; preserve fresh scene and render/screenshot order.
#
# IMPORTANT LIMITATIONS IN THE *ORIGINAL* EXECUTABLE LOGIC
#   - Only CONVEX polyhedra are modeled; input interior points are discarded.
#   - Near-degenerate/coplanar or noisy vertex inputs can fail ConvexHull or the
#     symmetry tests; changing a rendering option will not fix bad geometry.
#   - --plane-distinct-colors / --uniform-plane-color are PARSED but NOT USED by
#     create_scene. Per-plane color variation needs an implementation in the
#     visible_planes loop, and optionally in the legend and label colors.
#   - --plane-edge-width is PARSED and checked >0 but is NOT used to set a line
#     width. Plane borders are actual tubes governed by --plane-border-radius.
#   - --transparent is a compatibility-only flag: PNGs are always RGBA.
#   - --check-only STILL writes the .axes.json report; it skips PNG/manifest.
#   - The original top docstring's word 'opaque' does not override the actual
#     default --opacity=0.50; body translucency is enabled by default.
#   - The SHA-256 duplicate-image guard catches stale frames but can also reject
#     legitimately pixel-identical projections of a highly symmetric object.
#   - --interactive previews the FIRST scene and then applies the sampled camera
#     for the screenshot; it does not save the freely manipulated preview view.
#   - No global scene label or fixed metadata explains C_n on every output;
#     use --labels/--show-legend or add annotation logic where indicated.
#   - Detection uses numerical thresholds that assume the input shape is large
#     enough for stable floating-point computations and is truly 3-dimensional.
#   - The original exception handler does not intercept every VTK/Scipy error.
#   - C_n values refer to true rotational symmetries, not visually suggestive
#     face diagonals. Never replace the global vertex-set verification with only
#     a local three-point match.
#
# RECOMMENDED WORKFLOW FOR A SAFE CHANGE
#   1. First run your shape with --check-only and inspect <stem>.axes.json.
#   2. Change presentation by CLI options before editing algorithms.
#   3. Make a single edit, keeping the shape JSON and --seed fixed.
#   4. Re-run --check-only; compare rotation counts, axes, mirror normals.
#   5. Render at small resolution, visually inspect PNG alpha and all views.
#   6. Only increase resolution and N after tests pass.
#
# NOTE ON NOTATION
#   '@' is NumPy matrix multiplication; '*' between arrays is elementwise.
#   p @ R.T applies R to every row-point of p; the transpose is deliberate.
#   np.linalg.norm(v) is Euclidean vector length when v is a single vector.
#   A @ B is composition: apply B first, then A (column-vector notation).
#   'det' means determinant and 'trace' means sum of diagonal entries.
# ===============================================================================
import argparse
# IMPORTS: built-ins first, then NumPy and SciPy.
# argparse parses terminal flags; json reads/writes reports; sys handles terminal
# input and fatal error status; defaultdict counts C_n axes; Path builds paths.
# NumPy performs linear algebra; ConvexHull constructs geometric boundaries;
# cKDTree accelerates nearest-vertex queries; Rotation exposes axis-angle form.
# PyVista is deliberately imported INSIDE create_scene, so --check-only can run
# without installing a GPU/visualization stack.
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation


# Default color map for rotational-axis order.
# These defaults match the requested visual convention:
#   C4 -> dark green, C3 -> dark orange, C2 -> dark purple, C5 -> dark brown.
# Additional orders are given sensible darker tones.
# TWEAK HERE (AXIS PALETTE): each dictionary key is a rotational order n.
# C2, C3, C4, etc. share one color per order. A CLI --cN-color wins over a value
# here. The fallback color (parse_arguments) applies to orders not listed or
# overridden. These map entries do NOT affect polyhedron faces or mirror planes.
DEFAULT_AXIS_COLORS = {
    2: '#5B2C83',  # dark purple
    3: '#B85C00',  # dark orange
    4: '#1B5E20',  # dark green
    5: '#6D4C41',  # dark brown
    6: '#1F4E79',  # dark blue
}


# ======================== AXIS COLOR RESOLUTION =============================
# Inputs: order (integer C_n), args (argparse Namespace).
# Output: one PyVista-compatible color. No geometry or symmetry is changed here.
# TWEAK HERE for order-dependent styling; for per-axis coloring, pass an axis ID
# into this function AND change the axis loop in create_scene.
def axis_color_for_order(order, args):
    """Return the display color associated with a rotational order C_n.

    Known orders use dedicated defaults (or CLI overrides). Higher/unlisted
    orders fall back to a user-configurable generic dark color.
    """
    # Build a CLI attribute dynamically, e.g. order=4 -> 'c4_color'. This lets
    # one helper support existing --c2-color through --c6-color without many cases.
    override_attr = 'c{}_color'.format(order)
    # hasattr prevents AttributeError for orders that lack a CLI option.
    # A nonempty CLI value takes precedence over the dictionary below.
    if hasattr(args, override_attr):
        value = getattr(args, override_attr)
        if value:
            return value
    # If no override is present, look in DEFAULT_AXIS_COLORS. If the key is
    # absent (e.g., C7), fall back to --axis-color-fallback.
    return DEFAULT_AXIS_COLORS.get(order, args.axis_color_fallback)


# =========================== JSON INPUT / HULL ===============================
# Input : path to shape JSON; expected one N-by-3 numeric vertex array.
# Output: (original JSON dict, filtered hull vertices, vertex mean, ConvexHull).
# NOT returned: triangle faces; those are constructed downstream.
# TWEAK HERE to accept a new JSON key, conversion, vertex preprocessing, or an
# alternate centering convention. Keep the Nx3 array consistent with all indices.
# The script REQUIRES a 3D convex hull; it does not render concave cavities.
def read_vertices(json_path):
    """Load vertex arrays in common HPMC shape-catalog JSON formats."""
    # JSON keys and values are loaded as ordinary Python objects; no shell
    # commands or other executable content is read from the shape file.
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # The search order matters: 'vertices' wins if it is a valid Nx3 array;
    # then common catalog-specific names, then any key ending in '_vertices'.
    keys = ['vertices', '8_vertices', '12_vertices', 'polyhedron_vertices']
    keys.extend(k for k in data if k.endswith('_vertices') and k not in keys)
    vertices = None
    # Find the FIRST candidate matching shape (N,3). Bad numeric strings can
    # cause NumPy conversion to throw before another key is tried; if input may be
    # messy, add exception handling or explicit schema validation here.
    for key in keys:
        if key in data:
            attempt = np.asarray(data[key], dtype=float)
            if attempt.ndim == 2 and attempt.shape[1] == 3:
                vertices = attempt
                break
    # Raise user-friendly errors before invoking ConvexHull. Finite excludes
    # NaN and infinity, both of which break geometric distance tests.
    if vertices is None:
        raise ValueError('No (N, 3) vertex array found in the JSON file.')
    if len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError('At least four finite 3D vertices are required.')

    # ConvexHull returns a triangulated convex boundary. hull.vertices are
    # indices of EXTREME points. Discarding interior points is necessary because
    # an interior point must not be treated as a physical corner or face vertex.
    # WARNING: a coplanar/near-degenerate shape can raise a SciPy QhullError here.
    # Remove redundant points that do not belong to the convex hull.
    hull = ConvexHull(vertices)
    if len(hull.vertices) < len(vertices):
        print('[info] Ignoring {} redundant/interior vertices.'.format(
            len(vertices) - len(hull.vertices)))
        vertices = vertices[np.sort(hull.vertices)]
        hull = ConvexHull(vertices)

    # The equal-weight mean is fixed by any symmetry that permutes vertices:
    # R(mean(vertices)) = mean(R(vertices)) = mean(vertices).
    # Hence symmetry transformations can be sought around this center for a finite
    # vertex set; the mean is NOT necessarily the integral volume centroid.
    # TWEAKING CENTER: changing it arbitrarily invalidates symmetry detection,
    # plane positions, and cylinder origins. Prefer a mathematically justified one.
    # Symmetry center is guaranteed by the finite set's rotational group;
    # it need not equal the centroid of the solid's volume.
    center = vertices.mean(axis=0)
    return data, vertices, center, hull


# ====================== REAL POLYGON FACES / EDGES ===========================
# Problem: SciPy ConvexHull represents quadrilateral faces as TWO triangles.
# Drawing the triangles' shared diagonal would create a false physical edge.
# Solution: group triangles whose outward planes are coincident, union their
# vertex indices, sort each polygon around its center, and collect boundary
# edges as undirected (min_id,max_id) pairs.
# Inputs: vertices[N,3], hull, rel_tol=1e-7 (face-merge tolerance).
# Returns: VTK faces as flat int array, sorted edge list, integer face count.
# VTK face encoding: [k, v0, ..., v(k-1), next_k, ...].
# TWEAK HERE to change coplanarity tolerance or polygon reconstruction; it
# changes the rendered BODY but not the symmetry solver's original vertices.
def polygon_faces_from_hull(vertices, hull, rel_tol=1e-7):
    """
    Merge ConvexHull coplanar triangles into true polygon faces.

    Crucial for publication images: a quadrilateral face should not show the
    fictitious diagonal inserted by the hull's triangulation.

    Returns (vtk_flat_faces, physical_edges).
    """
    # 'extent' is the largest x/y/z bounding-box span, with a tiny floor
    # to avoid zero. Convert relative tolerance to input-coordinate units.
    extent = max(np.ptp(vertices, axis=0).max(), 1e-12)
    plane_tol = rel_tol * extent
    groups = []

    # hull.simplices: vertex-index triangle triples; hull.equations: [a,b,c,d]
    # with plane a*x+b*y+c*z+d=0. Normalizing converts them to unit normal and
    # signed offset so different triangle representations can be compared.
    for triangle, equation in zip(hull.simplices, hull.equations):
        normal = equation[:3] / np.linalg.norm(equation[:3])
        offset = equation[3] / np.linalg.norm(equation[:3])
        # Look for a previous coplanar group: normals point the SAME way
        # (dot near +1), and plane offsets agree within plane_tol. Different physical
        # faces meeting along an edge must NOT merge, even if they share vertices.
        target = None
        for group in groups:
            if (np.dot(normal, group['normal']) > 1 - 1e-8
                    and abs(offset - group['offset']) < plane_tol):
                target = group
                break
        # Create a new face-group only if no equivalent plane exists.
        # A set() automatically removes shared triangle-vertex duplicates.
        if target is None:
            target = {'normal': normal, 'offset': offset, 'indices': set()}
            groups.append(target)
        target['indices'].update(int(i) for i in triangle)

    # At this point every group is ONE geometric polygon. 'edges' uses a
    # set so the shared edge of adjacent faces appears exactly once.
    vtk_faces = []
    edges = set()
    for group in groups:
        ids = np.array(sorted(group['indices']), dtype=int)
        if len(ids) < 3:
            raise ValueError('Malformed face in convex hull.')
        # Use the average of polygon vertices as a polar sorting origin. All
        # corners are coplanar so project into the face's 2D basis u/v.
        points = vertices[ids]
        face_center = points.mean(axis=0)
        normal = group['normal']
        # Compute an in-plane unit basis u. Subtract its component along the
        # normal first, in case roundoff puts it slightly off the polygon plane.
        u = points[0] - face_center
        u -= np.dot(u, normal) * normal
        u /= np.linalg.norm(u)
        # v = normal cross u gives a second in-plane direction, with
        # u cross v = outward normal. The sign establishes polygon winding.
        v = np.cross(normal, u)  # u cross v points outward
        # Relative coordinates dotted with u/v are 2D coordinates;
        # atan2(y,x) returns angle around face center in [-pi,pi].
        relative = points - face_center
        angle = np.arctan2(relative @ v, relative @ u)
        # Sort vertices by polar angle. VTK now sees a perimeter polygon
        # rather than unordered vertices; store each face in the flat VTK array.
        ordered = ids[np.argsort(angle)].tolist()
        vtk_faces.extend([len(ordered)] + ordered)
        # Collect only adjacent perimeter edges (including wraparound).
        # Sorted vertex-id pairs make (i,j) and (j,i) identical in the set.
        for j in range(len(ordered)):
            edges.add(tuple(sorted((ordered[j], ordered[(j + 1) % len(ordered)]))))

    # The third return value is face COUNT (not a list of faces); it is
    # printed in the summary. np.int64 conforms to PyVista connectivity inputs.
    return np.asarray(vtk_faces, dtype=np.int64), sorted(edges), len(groups)


# ===================== ALL PROPER ROTATIONAL SYMMETRIES ======================
# Goal: find every 3x3 orthogonal R with det(R)=+1 that permutes ALL vertices.
# Do NOT infer symmetry from a few faces; a candidate is accepted only after
# checking the entire transformed vertex set bijectively.
#
# ALGORITHM EXPLAINED:
#   (1) Center the N points about the symmetry center.
#   (2) Choose 3 independent source 'radius vectors' from center to vertices.
#   (3) A distance-preserving rotation must preserve lengths and dot products
#       of these anchor vectors (their Gram matrix).
#   (4) Enumerate all matching target triples. This prunes most impossible
#       mappings without fitting a matrix.
#   (5) Fit the best rotation via SVD / constrained orthogonal Procrustes.
#   (6) For each fitted R, transform every vertex and query a cKDTree.
#   (7) Accept only if every transformed vertex lands near a UNIQUE original.
#   (8) Add identity if absent and check closure under matrix multiplication.
#
# Costs: candidate enumeration may become expensive for very many vertices or
# highly symmetric vertex sets. R entries are approximate floating-point values.
# Inputs : vertices[N,3], center[3], rel_tol (relative to vertex radius).
# Returns: list of 3x3 matrices INCLUDING identity.
# TWEAK HERE only if modifying numerical algorithm or precision guarantees.
# For user-level near-symmetry control, start with --symmetry-tol.
def find_proper_rotations(vertices, center, rel_tol=1e-6):
    """
    Exhaustive vertex-frame construction of proper rotational symmetries.

    Three linearly independent radius vectors determine a unique rotation.
    Their images must preserve all pairwise inner products. Generate ONLY
    candidate image triples with matching Gram matrices, use proper Kabsch
    to recover the rotation, then verify a BIJECTIVE match of the entire
    rotated vertex set against the original set with a KD-tree.

    This does not assume any axes, faces, symmetry point groups, or a fixed
    list of rotation angles. Reflections (det=-1) are never accepted.
    """
    # Subtract the center from each vertex so all sought transformations are
    # linear (3x3 rotations about the origin) rather than affine 4x4 transforms.
    p = vertices - center
    # radius is the largest centered vertex distance, used for tolerances.
    # If zero, geometry cannot define any meaningful rotational axes.
    radius = np.linalg.norm(p, axis=1).max()
    if radius <= 0:
        raise ValueError('Degenerate vertices: zero radius.')
    # Vertex-match absolute tolerance = rel_tol * radius.
    # Dot products have squared-length units; dot_tol is scaled accordingly.
    # The factor 4 is an implementation heuristic, NOT a physical constant.
    absolute_tol = float(rel_tol) * radius
    dot_tol = 4.0 * absolute_tol * radius

    # r2[i] = ||p_i||^2; dot[i,j] = p_i dot p_j.
    # Inner products are preserved by any orthogonal symmetry, so they cheaply
    # filter candidate permutations before expensive SVD and whole-set matching.
    r2 = np.einsum('ij,ij->i', p, p)
    dot = p @ p.T
    n = len(p)

    # Choose a longest vector a, b furthest away from being parallel to
    # a (largest cross-product magnitude), and c maximally outside their plane
    # (largest scalar triple-product magnitude). This improves conditioning.
    # Three well-separated independent anchor vectors are numerically stable.
    a = int(np.argmax(r2))
    b = int(np.argmax(np.linalg.norm(np.cross(p[a], p), axis=1)))
    c = int(np.argmax(np.abs(p @ np.cross(p[a], p[b]))))
    anchors = (a, b, c)
    source = p[list(anchors)]
    # A determinant near zero means source frame vectors are linearly
    # dependent too weakly to define a stable 3D basis. Threshold scales as r^3.
    if abs(np.linalg.det(source)) < 1e-9 * radius**3:
        raise ValueError('Cannot find three independent radius vectors.')

    # KD-tree stores original centered points and supports fast nearest-
    # neighbor queries from transformed points. 'matrices' accumulates accepted R.
    lookup = cKDTree(p)
    matrices = []
    # Candidate anchor-image j must have almost the same squared length
    # as a. np.flatnonzero returns all satisfying candidate vertex indices.
    for j in np.flatnonzero(np.abs(r2 - r2[a]) <= dot_tol):
        # Candidate image k must match length(b) AND dot(a,b) when paired
        # with j. '&' combines NumPy Boolean masks element by element.
        possible_k = np.flatnonzero(
            (np.abs(r2 - r2[b]) <= dot_tol)
            & (np.abs(dot[j] - dot[a, b]) <= dot_tol)
        )
        # Candidate image l must match length(c), dot(a,c), and dot(b,c).
        # All 6 Gram-matrix entries then agree approximately for the source/target
        # 3-vector frames (3 lengths + 3 pairwise dot products).
        for k in possible_k:
            possible_l = np.flatnonzero(
                (np.abs(r2 - r2[c]) <= dot_tol)
                & (np.abs(dot[j] - dot[a, c]) <= dot_tol)
                & (np.abs(dot[k] - dot[b, c]) <= dot_tol)
            )
            for l in possible_l:
                # We have a candidate source -> target mapping. The target triple
                # uses original vertex indices j,k,l in the prescribed anchor order.
                target = p[[j, k, l]]
                # Proper (det=+1) orthogonal Procrustes alignment.
                # KABSCH / PROCRUSTES DETAILS:
                #   Given row vectors in source S and target T, find R minimizing
                #        || S @ R.T - T ||_F^2    subject to R.T R = I, det(R)=+1.
                #   M = S.T @ T; SVD gives U diag(s) V^T.
                #   The determinant correction avoids a reflection when the best unconstrained
                #   orthogonal fit has det=-1. The resulting R is V D U^T.
                # Do not remove correction: otherwise mirrors may masquerade as rotations.
                u, _, vt = np.linalg.svd(source.T @ target)
                reflection_fix = np.eye(3)
                reflection_fix[2, 2] = np.sign(np.linalg.det(u @ vt))
                matrix = vt.T @ reflection_fix @ u.T
                # Numerical determinant safeguard: if det(R) is materially different
                # from +1, reject the candidate as an invalid proper rotation.
                if np.linalg.det(matrix) < 1 - 1e-7:
                    continue
                # Deduplicate nearly identical 3x3 matrices using Frobenius norm.
                # Without this, many anchor triples could discover the SAME physical rotation.
                if any(np.linalg.norm(matrix - old) < 1e-6
                       for old in matrices):
                    continue
                # Row-wise transform uses p @ R.T. KDTree returns a nearest-original
                # index for each transformed point and corresponding Euclidean distance.
                distances, nearest = lookup.query(p @ matrix.T, k=1)
                # Two conditions are BOTH necessary:
                #   max(distance) <= absolute_tol: every transformed vertex is close enough;
                #   len(set(nearest)) == N : no two transformed vertices map to one original.
                # Because source and target each contain N vertices, an injective matching
                # becomes a bijection. This prevents false positives from partial overlap.
                if np.max(distances) <= absolute_tol and len(set(nearest)) == n:
                    matrices.append(matrix)

    # Identity R=I always maps the vertex set to itself and must be included,
    # especially for shapes without nontrivial rotational symmetry.
    # Identity is a symmetry even if numerical candidate reconstruction failed.
    if not any(np.linalg.norm(m - np.eye(3)) < 1e-6 for m in matrices):
        matrices.append(np.eye(3))

    # Closure means if R_a and R_b are recognized symmetries, R_a @ R_b
    # must also be present. A missing product suggests precision/candidate issues.
    # This is a diagnostic, not an automatic solver that repairs a missing group.
    # Check closure: incomplete candidate generation is not silently accepted.
    for ra in matrices:
        for rb in matrices:
            product = ra @ rb
            if not any(np.linalg.norm(product - rc) < 1e-4 for rc in matrices):
                raise RuntimeError('Detected rotations are not closed as a group; '
                                   'try a smaller --symmetry-tol.')
    return matrices


# ======================= FIND ONE IMPROPER SYMMETRY =========================
# An improper orthogonal operation Q has det(Q)=-1: it reverses handedness.
# Examples: a mirror reflection, inversion (Q=-I), or a roto-reflection.
# Fundamental group fact: if one improper Q is a symmetry, every other improper
# symmetry can be expressed as R @ Q for some proper symmetry R.
# Therefore the function can stop at the FIRST verified det=-1 matrix.
#
# It intentionally mirrors the proper-rotation solver's anchor-triple search,
# except the SVD determinant correction FORCES det=-1. An improper symmetry
# is NOT necessarily itself a geometric mirror plane; find_mirror_planes later
# filters the complete improper coset to find pure reflections.
# Returns: a 3x3 improper symmetry matrix or None if no match is found.
def find_one_improper_symmetry(vertices, center, rel_tol=1e-6):
    """Find one det=-1 vertex symmetry, if there is one.

    The improper operations form either an empty set or a coset of the proper
    rotation group. Hence only ONE improper symmetry needs to be found; all
    the others can be generated exactly from the already-detected rotations.
    Works for reflections, inversion, and general roto-reflections alike.
    """
    # Centering, radius scaling, and Gram-matrix compatibility work
    # exactly as in find_proper_rotations. Keep tolerance changes consistent between
    # both solvers to avoid mismatched proper and improper operation sets.
    p = vertices - center
    radius = float(np.linalg.norm(p, axis=1).max())
    absolute_tol = float(rel_tol) * radius
    dot_tol = 4.0 * absolute_tol * radius
    r2 = np.einsum('ij,ij->i', p, p)
    dots = p @ p.T
    n = len(p)

    # Reuse the same stable anchor-frame strategy: one longest radius,
    # one maximally nonparallel radius, and one far from their spanned plane.
    a = int(np.argmax(r2))
    b = int(np.argmax(np.linalg.norm(np.cross(p[a], p), axis=1)))
    c = int(np.argmax(np.abs(p @ np.cross(p[a], p[b]))))
    source = p[[a, b, c]]
    if abs(np.linalg.det(source)) < 1e-9 * radius**3:
        raise ValueError('Cannot find independent anchor vectors for reflections.')

    # Try all approximate Gram-matrix-preserving anchor images. Here
    # we only need ONE whole-shape improper symmetry, so there is no list/closure
    # check and the function returns immediately when one is verified.
    lookup = cKDTree(p)
    for j in np.flatnonzero(np.abs(r2 - r2[a]) <= dot_tol):
        ks = np.flatnonzero(
            (np.abs(r2 - r2[b]) <= dot_tol) &
            (np.abs(dots[j] - dots[a, b]) <= dot_tol))
        for k in ks:
            ls = np.flatnonzero(
                (np.abs(r2 - r2[c]) <= dot_tol) &
                (np.abs(dots[j] - dots[a, c]) <= dot_tol) &
                (np.abs(dots[k] - dots[b, c]) <= dot_tol))
            for l in ls:
                target = p[[j, k, l]]
                # The unconstrained SVD may yield either orientation. Flipping the
                # last diagonal sign in 'correction' enforces determinant -1 even if the raw
                # orthogonal Kabsch solution would have determinant +1.
                u, _, vt = np.linalg.svd(source.T @ target)
                correction = np.eye(3)
                # Force det(Q)=-1 (opposite to the proper Kabsch solution).
                correction[2, 2] = -np.sign(np.linalg.det(u @ vt))
                # q is the det=-1 counterpart to the R computed above. Reject if
                # numerical determinant is too far from the required -1.
                q = vt.T @ correction @ u.T
                if np.linalg.det(q) > -1 + 1e-7:
                    continue
                # Same bijective whole-vertex-set test as the proper solver.
                # A local reflection of only the three anchors would not be sufficient.
                distances, nearest = lookup.query(p @ q.T, k=1)
                if np.max(distances) <= absolute_tol and len(set(nearest)) == n:
                    return q
    # No improper operation discovered => the polyhedron may be chiral
    # (no reflection/inversion/roto-reflection), or numerical tolerances may be
    # too restrictive. The caller treats None as no mirror planes.
    return None


# ======================= PURE MIRROR PLANE DETECTION =========================
# Given the proper group G+ and one improper representative Q0, enumerate
# all improper operations Q = R @ Q0 for R in G+.
# A PURE REFLECTION across a plane has eigenvalues (+1,+1,-1):
#   - points INSIDE the plane stay fixed (two +1 eigenvectors);
#   - the perpendicular NORMAL reverses sign (one -1 eigenvector).
#   - det(Q) = -1, trace(Q) = +1, and Q @ Q = I.
# This excludes inversions and general roto-reflections that are NOT a plane.
# The mirror normal is canonicalized and duplicate opposite normals merged.
# Returns: [{'normal': unit_vector}, ...] for ALL physical mirror planes.
# TWEAK HERE for mirror mathematical detection, not its appearance.
def find_mirror_planes(vertices, center, rotations, rel_tol=1e-6):
    """Find ALL physical reflection planes from the improper symmetry coset.

    Q is a mirror iff it is orthogonal with determinant -1 and has spectrum
    (+1,+1,-1), equivalently trace(Q)=1 and Q^2=I. Normal n is the -1
    eigenvector; n and -n represent the same unoriented plane.
    """
    # If no improper symmetry exists, there cannot be a mirror plane.
    # Return an empty list, without making up planes from visual face geometry.
    representative = find_one_improper_symmetry(vertices, center, rel_tol)
    if representative is None:
        return []

    mirrors = []
    eye = np.eye(3)
    # The coset is generated using all rotations including identity;
    # this is why completeness/closure of find_proper_rotations matters.
    for rotation in rotations:
        q = rotation @ representative
        # Numerical mirror tests: det=-1, trace=1, Q squared = identity.
        # The thresholds here are hardcoded 1e-4: altering --symmetry-tol does not
        # change these checks. Review together if working with noisy coordinates.
        if (abs(np.linalg.det(q) + 1) > 1e-4 or
                abs(np.trace(q) - 1.0) > 1e-4 or
                np.linalg.norm(q @ q - eye) > 1e-4):
            continue
        # For an ideal mirror Q is symmetric. Symmetrize numerically with
        # (Q+Q.T)/2 before eigh, then find eigenvector of the smallest eigenvalue.
        # That eigenvector points NORMAL to the fixed mirror plane.
        eigvals, eigvecs = np.linalg.eigh((q + q.T) / 2.0)
        negative = int(np.argmin(eigvals))
        if abs(eigvals[negative] + 1) > 1e-4:
            continue
        # Canonical axis maps n and -n to the SAME representative. Dot
        # absolute value near 1 detects collinear normals, preventing duplicate planes.
        # Physical reflection planes pass through 'center'; report stores only normal.
        normal = canonical_axis(eigvecs[:, negative])
        if any(abs(np.dot(normal, m['normal'])) > 1 - 1e-6
               for m in mirrors):
            continue
        mirrors.append({'normal': normal})

    # Stable, human-readable ordering by normal components; this impacts
    # the order of planes in JSON and the order drawn, not their correctness.
    mirrors.sort(key=lambda m: (-m['normal'][2], -m['normal'][1],
                                -m['normal'][0]))
    return mirrors


# ========================== CLASSIFY sigma_h ================================
# IMPORTANT: 'horizontal' is NOT an absolute world-Z classification.
# It is relative to a reference ROTATION AXIS: mirror normal must be parallel
# (or antiparallel) to that axis, making the mirror plane perpendicular to it.
# Options:
#   --horizontal-reference maximum = use ALL axes tied for maximum C_n order.
#   --horizontal-reference all     = accept every detected C_n axis.
#   --principal-axis-index K       = override with ONE displayed axis (1-based).
# The function receives principal_index in Python's ZERO-based convention.
# For an ideal cube, there are three equivalent C4 axes: 'maximum' can classify
# multiple mirror planes as sigma_h, unlike selecting one axis by convention.
# The `planes` dictionaries are MUTATED in place and returned for convenience.
def classify_mirror_planes(planes, axes, reference='maximum', principal_index=None):
    """Mark sigma_h planes relative to maximal-C_n or all detected axes.

    A horizontal mirror has a normal collinear with a reference rotation axis.
    In a polyhedron with multiple equal-order principal axes, each is accepted.
    Without a nontrivial rotation axis, no sigma_h classification is assigned.
    """
    # A manually selected axis wins over reference=maximum/all. Validate
    # zero-based range against axes to make bad CLI indices fail immediately.
    if principal_index is not None:
        if not 0 <= principal_index < len(axes):
            raise ValueError('--principal-axis-index must reference a listed C_n axis.')
        reference_axes = [(principal_index, axes[principal_index])]
    # Without an explicit index, choose all axes of largest rotational
    # order. Sorting of axes (symmetry_axes) is what defines CLI index ordering.
    elif axes and reference == 'maximum':
        max_order = max(a['order'] for a in axes)
        reference_axes = [(i, a) for i, a in enumerate(axes)
                          if a['order'] == max_order]
    # For reference='all', or when there are NO axes, use full axes list.
    # For no axes this is empty, so no plane receives sigma_h status.
    else:
        reference_axes = list(enumerate(axes))

    # Compare each mirror normal with each undirected rotational axis.
    # abs(dot(n,u)) close to 1 means normals and axes are parallel, because both
    # are unit vectors. The absolute value allows either sign orientation.
    for plane in planes:
        aligned = [i for i, axis in enumerate(axes)
                   if abs(float(np.dot(plane['normal'], axis['direction'])))
                   > 1 - 1e-6]
        # 'aligned' includes ALL axes perpendicular to the mirror.
        # 'horizontal' includes only those axes currently chosen as references.
        # Store BOTH lists so report consumers can distinguish mathematical alignment
        # from the particular sigma_h convention selected by the user.
        horizontal = [i for i, _ in reference_axes if i in aligned]
        plane['perpendicular_axis_indices'] = aligned
        plane['horizontal_axis_indices'] = horizontal
        # Any selected aligned axis makes is_sigma_h=True; this drives
        # --plane-mode horizontal, plane color, labels, legend, and report counters.
        plane['is_sigma_h'] = bool(horizontal)
    return planes


# ===================== CROSS-SECTION SHAPED MIRROR PATCH ====================
# This routine draws a patch shaped like the TRUE polygonal intersection of the
# mirror plane and the convex body, then expands it by `extension`.
# The intersection of an edge from A to B with signed plane distances dA,dB is
#   fraction = dA/(dA-dB);  intersection = A + fraction*(B-A).
# Edge endpoints sufficiently ON the plane are included directly.
# Inputs: physical edges (not triangulation diagonals), center, unit normal,
#         extension > 0 (1=exact section, >1=visible outside body).
# Output: ordered Nx3 polygon vertex array suitable for VTK PolyData.
# TWEAK HERE if --plane-style section looks too small or too angular.
def mirror_section_polygon(vertices, edges, center, normal,
                           extension=1.10, relative_tol=1e-7):
    """Intersect convex polyhedron with n.(r-center)=0, extend its outline.

    Uses physical polygon edges (not hull triangulation diagonals). The true
    section is a convex polygon; sorting by polar angle gives a planar VTK face.
    """
    # Scale plane membership tolerance by body radius. 'signed[i]' is
    # signed distance from vertex i to n dot (x-center)=0 if n is unit length.
    radius = max(float(np.linalg.norm(vertices - center, axis=1).max()), 1e-12)
    tol = relative_tol * radius
    signed = (vertices - center) @ normal
    candidates = []
    # Scan only physical body edges, because each edge may intersect
    # the slicing plane. Add endpoints close to the plane and crossings where
    # endpoint signs differ by more than the tolerance-squared threshold.
    for i, j in edges:
        di, dj = signed[i], signed[j]
        if abs(di) <= tol:
            candidates.append(vertices[i])
        if abs(dj) <= tol:
            candidates.append(vertices[j])
        # An edge crossing has d_i and d_j opposite in sign. Linear
        # interpolation finds its exact plane crossing for straight edge geometry.
        if di * dj < -(tol * tol):
            fraction = di / (di - dj)
            candidates.append(vertices[i] + fraction * (vertices[j] - vertices[i]))

    # Adjacent edges may report the SAME face corner or crossing.
    # Deduplicate using a small distance tolerance before constructing a polygon.
    points = []
    for candidate in candidates:
        if not any(np.linalg.norm(candidate - old) < 10 * tol for old in points):
            points.append(np.asarray(candidate, dtype=float))
    # At least three distinct points are required for a nondegenerate
    # convex plane-section polygon; otherwise fail instead of rendering junk.
    if len(points) < 3:
        raise RuntimeError('Could not construct mirror/polygon intersection.')

    # Find the 2D in-plane reference frame and sort points by angle
    # around the section centroid. This creates the polygon's perimeter winding.
    points = np.asarray(points)
    face_center = points.mean(axis=0)
    u = points[0] - face_center
    u -= np.dot(u, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    angles = np.arctan2((points - face_center) @ v,
                        (points - face_center) @ u)
    points = points[np.argsort(angles)]
    # Scale all points radially away from the symmetry center, not the
    # section polygon centroid. Because every true mirror goes through the center,
    # this keeps the expanded patch aligned with the same mathematical plane.
    # Modest expansion allows the mirror to remain visible around the body.
    return center + extension * (points - center)


# ============================ PLANE 2D BASIS ================================
# Given unit normal n, construct orthonormal u/v IN the plane:
#   u = normalize(n cross ref)
#   v = normalize(n cross u)
# where ref is a world axis not almost parallel to n.
# Used by square and disc patch helpers. The chosen in-plane orientation has no
# physical significance; only the normal determines a mathematical plane.
def plane_basis_from_normal(normal):
    """Return orthonormal in-plane basis vectors (u, v) for a plane normal."""
    # Use a copy/conversion to floating-point and normalize normal.
    # Do not pass a zero normal; division by zero would be undefined.
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    # Avoid using world X as reference when n nearly points along X,
    # because their cross product would have unstable near-zero length.
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    # Cross products guarantee orthogonality. Normalization makes
    # u,v unit length, allowing radius and span to be expressed in body units.
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


# ==================== DEFAULT RECTANGULAR MIRROR PATCH ======================
# Draw a finite planar quadrilateral centered on the shape's symmetry center.
# This is PRESENTATION GEOMETRY only; the real mirror is infinite and defined
# by its normal. No symmetry detection occurs here.
#   (1) Project hull vertices onto the mirror plane.
#   (2) Pick a long projected direction u to orient the rectangle consistently.
#   (3) Compute v = n cross u.
#   (4) Find projected half-extents hx,hy and multiply by --plane-span.
#   (5) Enforce hx >= --plane-aspect * hy for an obviously rectangular outline.
#   (6) Return four corners in cyclic order.
# Change --plane-span/aspect in the CLI to resize without editing geometry.
def mirror_rectangle_patch(center, normal, vertices, span=1.22, aspect=1.22):
    """A finite, physically oriented rectangular mirror patch.

    Align its first side with the longest projected physical vertex-to-vertex
    direction, avoiding arbitrary camera-dependent rotations.
    """
    # Normalize normal; subtract center so projection is around origin.
    # np.outer(relative @ n, n) is each vertex's component along n.
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    relative = np.asarray(vertices, dtype=float) - center
    # Projected positions are p - (p dot n)n, guaranteed to lie in plane.
    # Choose a nonzero projected vector with maximum length as orientation u.
    projected = relative - np.outer(relative @ n, n)
    lengths = np.linalg.norm(projected, axis=1)
    u = projected[np.argmax(lengths)]
    # Degenerate fallback: if every projected vector is almost zero,
    # choose a stable arbitrary in-plane axis instead of dividing by zero.
    if np.linalg.norm(u) < 1e-10:
        u, _ = plane_basis_from_normal(n)
    else:
        u = u / np.linalg.norm(u)
    # Cross to obtain perpendicular in-plane v, then normalize for safety.
    # The rectangle lies in span(u,v) regardless of camera rotation.
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    # Size based on the actual projected extent of the polyhedron.
    # Projection onto u and v gives the body's half-extent in each
    # direction. span >1 extends patch beyond body; span <1 makes it smaller.
    hx = max(np.max(np.abs(relative @ u)), 1e-10) * span
    hy = max(np.max(np.abs(relative @ v)), 1e-10) * span
    # The aspect logic modifies hx only, ensuring the 'u' direction
    # will be the long side even for a shape with large y extent.
    # Ensure an obvious rectangle rather than a giant square sheet.
    if hx < aspect * hy:
        hx = aspect * hy
    # Four corners, ordered around the rectangle. This also sets the
    # boundary shape drawn as tube segments in create_scene.
    return np.asarray([
        center - hx*u - hy*v,
        center + hx*u - hy*v,
        center + hx*u + hy*v,
        center - hx*u + hy*v
    ])


# =========================== LEGACY SQUARE PATCH ============================
# Square has equal hx=hy=radius*span_factor, not shape-fitted dimensions.
# Uses plane_basis_from_normal, so in-plane orientation may differ from rectangle.
# Swap --plane-style square to use this helper. For non-square ratios, change
# this routine or use the default rectangle helper.
def mirror_square_patch(center, normal, radius, span_factor=1.22):
    """Compatibility with older command-line options."""
    u, v = plane_basis_from_normal(normal)
    half = span_factor * radius
    return np.array([center-half*u-half*v, center+half*u-half*v,
                     center+half*u+half*v, center-half*u+half*v])


# =========================== CIRCULAR DISC PATCH ============================
# Parametric circle: p(theta)=center + rr*(cos(theta)*u + sin(theta)*v).
# --plane-resolution changes n_segments: more segments = smoother boundary,
# more vertices/overhead. --plane-span sets radius multiplier for this style.
# Note: despite being called 'disc', this returns ONLY perimeter vertices;
# create_scene fills the interior by constructing a polygonal PyVista face.
def mirror_disc_patch(center, normal, radius, span_factor=1.32, n_segments=96):
    """Construct a circular plane patch centered on the polyhedron center."""
    u, v = plane_basis_from_normal(normal)
    rr = span_factor * radius
    # endpoint=False excludes duplicated vertex at 2*pi, because border
    # construction connects last vertex back to the first explicitly.
    angles = np.linspace(0.0, 2.0*np.pi, int(n_segments), endpoint=False)
    pts = [center + rr*(np.cos(a)*u + np.sin(a)*v) for a in angles]
    return np.asarray(pts, dtype=float)


# ============================ AXIS SIGN RULE ================================
# An axis is an UNORIENTED line through center; +u and -u represent one axis.
# Normalize, find its largest-absolute component, and force that component >=0.
# This deterministic sign convention is used for both rotation axes and plane
# normals; it prevents double counting and stabilizes sorting/reporting.
def canonical_axis(vec):
    """Canonicalize undirected axis lines: a and -a denote the same axis."""
    # No zero-length check here: callers must supply nonzero vectors.
    # Do not reuse canonical_axis on arbitrary zero vectors.
    vec = np.asarray(vec, float)
    vec = vec / np.linalg.norm(vec)
    # Stabilize signs independent of Rotation.as_rotvec() conventions.
    # Largest-absolute component is more stable under tiny perturbations
    # than simply requiring the first component to be positive when it is near 0.
    imax = int(np.argmax(np.abs(vec)))
    if vec[imax] < 0:
        vec = -vec
    return vec


# ============================ MATRIX -> C_n AXES ============================
# Each proper non-identity rotation has one fixed line (its rotation axis).
# SciPy Rotation.as_rotvec gives a vector pointing along that line, whose
# Euclidean length is the rotation ANGLE in radians.
# Group all matrices whose rotation-vector axes lie on the SAME undirected
# line, and infer C_n order from the number of non-identity powers about it:
#   C2 has 1 nontrivial rotation, C3 has 2, C4 has 3, etc.
# Returns sorted list of dicts: direction[3], angles_degrees[], order.
# Axis indexing in CLI/report follows this sorted order (1-based for users).
def symmetry_axes(matrices):
    """Group all nontrivial proper rotations into shared axis LINES."""
    groups = []
    # Rotation vector = angle * unit_axis. Its magnitude is in radians.
    # Identity has zero angle and no unique axis, so skip it.
    for matrix in matrices:
        rotation_vector = Rotation.from_matrix(matrix).as_rotvec()
        angle = np.linalg.norm(rotation_vector)
        if angle < 1e-5:
            continue
        # Converting rotvec to canonical unit line makes opposite rotation
        # signs share a group; angle magnitudes are kept separately.
        axis = canonical_axis(rotation_vector / angle)
        best_group = None
        # abs(dot) near 1 tests whether two unit directions are parallel OR
        # antiparallel, both of which describe the same undirected axis.
        for group in groups:
            if abs(np.dot(group['direction'], axis)) > 1 - 1e-6:
                best_group = group
                break
        # New axis line => create new bucket; add current nonidentity
        # rotation's angle (convert radians to degrees for reporting).
        if best_group is None:
            best_group = {'direction': axis, 'angles_degrees': []}
            groups.append(best_group)
        best_group['angles_degrees'].append(float(np.degrees(angle)))

    # For a complete cyclic rotational subgroup of order n there are
    # exactly n-1 non-identity matrices. Closure check in the solver helps guard
    # against missing powers, although tolerance issues can still matter.
    # A C_n axis has n-1 distinct nonidentity proper rotations about the line.
    for group in groups:
        group['order'] = len(group['angles_degrees']) + 1
        group['angles_degrees'].sort()

    # Sort highest-order axes FIRST; then deterministic component order.
    # --principal-axis-index points into this list, not the JSON vertex list.
    groups.sort(key=lambda g: (-g['order'], -g['direction'][2],
                               -g['direction'][1], -g['direction'][0]))
    return groups


# =============================== SYMMETRY JSON ===============================
# TWEAK HERE to change the machine-readable .axes.json schema.
# 'rotation_axis_counts' is number of DISTINCT axis LINES per C_n order;
# 'n_proper_rotations_including_identity' is matrix count, not axis count.
# 'mirror_planes' includes ALL pure mirror planes, including those not selected
# for rendering by --plane-mode. Their normals define planes through center.
# Indices are deliberately exported as 1-based for human-readable CLI use.
# JSON serialization uses .tolist() because NumPy arrays are not JSON-native.
# The function writes a report EVEN WITH --check-only.
def save_report(filename, json_data, vertices, center, rotations, axes, planes):
    # defaultdict(int) initializes unseen order counts to 0, avoiding
    # explicit key-presence checks for C2/C3/etc.
    counts = defaultdict(int)
    for a in axes:
        counts[a['order']] += 1
    # The report preserves the input shape name if available, otherwise
    # uses 'polyhedron'. The input's '1_Name' is preferred to its 'name'.
    report = {
        'name': json_data.get('1_Name', json_data.get('name', 'polyhedron')),
        'n_vertices': len(vertices),
        'n_proper_rotations_including_identity': len(rotations),
        'rotation_axis_counts': {'C{}'.format(k): int(v)
                                 for k, v in sorted(counts.items())},
        'center': center.tolist(),
        'n_mirror_planes': len(planes),
        # Count the classified mirrors, NOT merely planes whose normals happen
        # to match a world axis. The result depends on chosen reference C_n axes.
        'n_sigma_h_planes': sum(bool(p['is_sigma_h']) for p in planes),
        # One normal per UNIQUE plane. A plane is center + all vectors x
        # satisfying dot(x,normal)=0; there is no finite patch information in report.
        'mirror_planes': [
            {'normal_unit_vector': p['normal'].tolist(),
             'is_sigma_h': bool(p['is_sigma_h']),
             # Convert internal zero-based array indices to one-based axis indices
             # so a report index K can be passed back to --principal-axis-index K.
             'perpendicular_axis_indices_1_based':
                 [i + 1 for i in p['perpendicular_axis_indices']],
             'horizontal_axis_indices_1_based':
                 [i + 1 for i in p['horizontal_axis_indices']]}
            for p in planes],
        # The full list includes order, unit direction, and angles about
        # that axis. Report readers should not assume exactly one axis per order.
        'axes': [
            {'order': int(a['order']),
             'direction_unit_vector': a['direction'].tolist(),
             'angles_degrees': a['angles_degrees']}
            for a in axes
        ],
    }
    # indent=2 makes human-editable JSON. The folder is prepared in main.
    # Return the same dict for printing without needing to reopen the file.
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    return report


# =========================== RANDOM CAMERA SAMPLING =========================
# Naively sampling polar angle theta uniformly makes too many samples near
# the poles; this routine samples z = cos(theta) uniformly in [-1,1] and
# azimuth phi uniformly in [0,2*pi), giving uniform spherical SURFACE AREA.
# All returned (x,y,z) directions have norm 1 up to roundoff.
# Inputs: number of views and optional deterministic seed.
# Output: NumPy ndarray (n_views,3).
# TWEAK HERE for a designed camera tour, Fibonacci sphere, or fixed viewpoints.
# For exact reproducibility across different NumPy versions, record the sampled
# directions in the saved .views.json, not only the seed.
def random_camera_directions(n_views, seed=None):
    """Sample independent, isotropic camera positions on the unit sphere.

    Uniform z=cos(theta) and uniform azimuth produce uniform solid angle,
    unlike independently sampling polar and azimuthal angles.
    """
    # Use a local modern NumPy random Generator so seeding this function
    # does NOT modify global np.random state elsewhere in the application.
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=n_views)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, size=n_views)
    # sqrt(1-z^2) is the horizontal circle radius at height z.
    # Thus x=xy*cos(phi), y=xy*sin(phi), z=z; points lie on unit sphere.
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((xy * np.cos(azimuth),
                            xy * np.sin(azimuth), z))


# =========================== CAMERA PLACEMENT ===============================
# Camera position = center + 5*radius*direction; camera looks AT center.
# The 'view_up' vector defines image roll, with a fallback near the poles.
# Parallel/orthographic projection avoids perspective size changes across views.
# parallel_scale controls vertical half-frustum size in world units.
# TWEAK HERE to change object framing (1.30/1.65), camera distance (5.0),
# up-axis choice, or parallel vs perspective projection. Increasing scale
# makes the rendered object SMALLER within the PNG.
def set_camera_for_direction(plotter, direction, center, radius, axis_half_length):
    """Keep object scale and framing consistent at every camera direction."""
    # Normalize even if caller gave a vector not exactly unit length;
    # a zero vector is invalid and should be checked by any new caller.
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    # Prefer +Z as up. If viewing almost along Z, choose Y instead:
    # otherwise camera forward and up become nearly parallel (undefined roll).
    view_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, view_up)) > 0.96:
        view_up = np.array([0.0, 1.0, 0.0])

    # 5 radii moves camera safely outside body and axis cylinders.
    # For orthographic projection the camera distance largely affects clipping,
    # not on-image size, which is controlled by parallel_scale.
    position = center + 5.0 * radius * direction
    # PyVista camera_position accepts (position, focal_point, up).
    # convert ndarrays to normal Python lists for the VTK adapter.
    plotter.camera_position = [position.tolist(), center.tolist(), view_up.tolist()]
    # Switch to orthographic projection for stable comparisons.
    # For perspective use plotter.disable_parallel_projection(), then tune
    # camera view angle and distance together in a coordinated change.
    plotter.enable_parallel_projection()
    # The bounding sphere fits at all orientations, giving reproducible sizing.
    # Ensure both the longest symmetry-axis endpoints and body bounding
    # sphere fit with margin; max picks whichever needs more vertical space.
    plotter.camera.parallel_scale = max(axis_half_length * 1.30, radius * 1.65)
    plotter.reset_camera_clipping_range()
    return position, view_up


# ======================= PYVISTA SCENE CONSTRUCTION =========================
# This function builds ALL visible objects for ONE image. It does not save PNGs.
# Every call creates a new Plotter, lights, translucent mirror polygons and
# borders, polyhedron surface and physical edges, symmetry cylinders and sphere
# caps, plus optional labels/legend/title/XYZ widget.
# Outputs: (plotter, body_radius, axis_half_length).
#
# RENDER DRAW ORDER (insertion order, not necessarily transparency compositing)
#   [A] select visible mirrors and optionally enable depth peeling
#   [B] plane faces and tube-shaped borders
#   [C] physical polyhedron polygon faces
#   [D] REAL polyhedron edge wireframe
#   [E] colored C_n axes and their endpoint spheres
#   [F] optional labels, legend, vertices and coordinate axes
#
# WHERE TO TWEAK VISUALS:
#   - Face material: the surface add_mesh(...) block (~original lines 650-661).
#   - Edge width/color: physical_edges add_mesh (~original 664-669).
#   - Axis thickness/color/shine: cylinder loop (~original 671-688).
#   - Mirror fill/outline: visible_planes loop (~original 616-648).
#   - Light locations/intensities: light_specs (~original 588-599).
#   - Antialias quality: enable_anti_aliasing (~original 579-586).
#   - Text, legend, vertex/XYZ decorations: bottom of this function.
#
# Do not attempt to reposition real geometric axes just by changing camera;
# axis and mirror orientation come from verified symmetry computations.
def create_scene(vertices, center, faces, edges, axes, planes, args):
    """Build an independent PyVista scene, with all geometry and lighting.

    A separate plotter is used for every view because some older PyVista/VTK
    installations keep returning the old image if the camera changes between
    screenshot() calls without a fully refreshed render window.
    """
    # Delayed import permits --check-only to work with just NumPy/SciPy.
    # Failure here matters only when rendering is actually requested.
    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit('PyVista is required for rendering. Install with:\n'
                         '    python -m pip install pyvista') from exc

    # All geometry scales to input body radius, so the same settings are
    # reasonable for shapes measured in nm, m, or any other consistent unit.
    # --axis-length is a HALF-length multiplier, so the full cylinder extends
    # from center-axis_half_length*u to center+axis_half_length*u.
    # --axis-thickness is radius multiplier, not diameter.
    radius = float(np.linalg.norm(vertices - center, axis=1).max())
    axis_half_length = args.axis_length * radius
    axis_radius = args.axis_thickness * radius

    # Offscreen rendering avoids interactive display windows by default.
    # --interactive makes an onscreen window available for first-view preview.
    # window_size=(width,height); omitted height falls back to width (square PNG).
    # lighting='none' suppresses PyVista's default lights; we add custom ones.
    plotter = pv.Plotter(
        off_screen=not args.interactive,
        window_size=(args.resolution, args.height or args.resolution),
        lighting='none',
    )
    # Background is intentionally irrelevant: all PNGs are saved with alpha.
    # Try SSAA supersampling (higher quality, may need significant GPU/VRAM).
    # Fallback to MSAA if unsupported; silently proceed with backend defaults if
    # both are unavailable. Add logging here if a renderer seems jagged.
    try:
        plotter.enable_anti_aliasing('ssaa')
    except (ValueError, RuntimeError, AttributeError):
        try:
            plotter.enable_anti_aliasing('msaa')
        except (ValueError, RuntimeError, AttributeError):
            pass

    # Each entry = (position offset in BODY RADII, light intensity).
    # E.g. (4,-3,5) means center + radius*[4,-3,5]. Larger intensity adds more
    # illumination; total effect also depends on material diffuse/specular settings.
    # TWEAK HERE for studio-lighting direction, contrast, or softness.
    light_specs = [
        ((4, -3, 5), 0.95),
        ((-4, 2, 3), 0.48),
        ((0, 4, -3), 0.28),
        ((-2, -4, 1), 0.16),
    ]
    # Instantiate each white positional light and aim its focal point
    # at the polyhedron center; attach it to this scene's Plotter.
    for point, intensity in light_specs:
        light = pv.Light(position=center + radius * np.array(point),
                         focal_point=center, color='white',
                         intensity=intensity)
        plotter.add_light(light)

    # RENDER SELECTION versus DETECTION:
    # 'planes' contains all detected physical mirrors regardless of CLI mode.
    #   horizontal: include only plane['is_sigma_h'] selected by classification;
    #   all       : include every mirror (sigma_h plus other mirror planes);
    #   none      : draw no mirror geometry (but analysis/report still includes them).
    # If you want a new category or only certain plane indices, filter HERE.
    # Draw the translucent mirror sections FIRST. They remain softly visible
    # where they extend outside the body. Depth peeling handles overlaps.
    if args.plane_mode == 'horizontal':
        visible_planes = [p for p in planes if p['is_sigma_h']]
    elif args.plane_mode == 'all':
        visible_planes = planes
    else:
        visible_planes = []

    # Semi-transparent geometry can render incorrectly when overlaps are
    # sorted poorly. Depth peeling improves blending and occlusion; VTK/OpenGL
    # support varies. Failure is swallowed (rendering continues with fallback).
    # Note that if only OTHER mirror planes are translucent, args.opacity and
    # args.plane_opacity still govern this conditional; adjust for new behavior.
    if args.opacity < 1 or (visible_planes and args.plane_opacity < 1):
        try:
            plotter.enable_depth_peeling(number_of_peels=20,
                                         occlusion_ratio=0.0)
        except Exception:
            pass

    # One pass per visible mirror; 'i' is its RENDER index, which can differ
    # from its ordinal number in the complete report due to filtering.
    for i, plane in enumerate(visible_planes):
        # Choose style based on classification: sigma_h gets plane-color etc.,
        # other mirrors get --other-plane-* controls. These are shared colors for a
        # category, NOT unique colors per plane in the current original code.
        # To IMPLEMENT --plane-distinct-colors, select a per-plane palette in this loop.
        # You must decide how stable colors map to plane normals/report indices and
        # whether legend rows and labels should reflect those new colors.
        is_horizontal = plane['is_sigma_h']
        fill_color = args.plane_color if is_horizontal else args.other_plane_color
        outline_color = args.plane_edge_color if is_horizontal else args.other_plane_edge_color
        alpha = args.plane_opacity if is_horizontal else args.other_plane_opacity

        # DISPATCH BY PATCH STYLE. All helpers output vertices in the ACTUAL
        # mirror plane through center; styles differ only in the finite polygon boundary.
        #   rectangle -> oriented shape-fitted rectangle; --plane-span/aspect
        #   square    -> circle-derived square; --plane-span
        #   disc      -> regular N-gon approximation; --plane-span/resolution
        #   section   -> actual intersection cross-section; --plane-extension
        # Adding a FIFTH style also requires modifying the label-placement switch
        # below and argparse choices in parse_arguments().
        if args.plane_style == 'rectangle':
            points = mirror_rectangle_patch(
                center, plane['normal'], vertices,
                span=args.plane_span, aspect=args.plane_aspect)
        elif args.plane_style == 'square':
            points = mirror_square_patch(center, plane['normal'], radius, args.plane_span)
        elif args.plane_style == 'disc':
            points = mirror_disc_patch(center, plane['normal'], radius,
                                       args.plane_span, args.plane_resolution)
        elif args.plane_style == 'section':
            points = mirror_section_polygon(vertices, edges, center,
                                            plane['normal'], args.plane_extension)
        else:
            raise ValueError('Unsupported plane style: {}'.format(args.plane_style))
        # VTK flat face encoding begins with number of vertices nv, followed
        # by the indices 0,...,nv-1. pv.PolyData builds the colored fill face.
        nv = len(points)
        section = pv.PolyData(points, np.r_[nv, np.arange(nv, dtype=np.int64)])
        # No lighting for planes: color appears flat/consistent even under
        # studio lights. Opacity is intentionally low so the solid stays visible.
        plotter.add_mesh(section, color=fill_color, opacity=alpha,
                         lighting=False, show_edges=False,
                         name='mirror_plane_{}'.format(i))
        # Create a PolyData LINE CELL for every polygon side. VTK line-cell
        # format is [2, start_index, end_index], where 2 means two endpoints.
        # Wraparound '(j+1) % nv' closes final side to first.
        lines = pv.PolyData(points)
        lines.lines = np.array([[2, j, (j+1) % nv]
                                for j in range(nv)], dtype=np.int64).ravel()
        # Tube filter gives the outline real 3D thickness. Radius is
        # --plane-border-radius * body_radius (NOT --plane-edge-width).
        # 12 facets around the circumference approximate a cylinder; more facets
        # smooth the outline but increase mesh memory.
        border = lines.tube(radius=args.plane_border_radius * radius,
                            n_sides=12, capping=True)
        # Outline opacity can be higher than plane fill opacity so the plane
        # boundary stays visible even with very transparent fills.
        plotter.add_mesh(border, color=outline_color,
                         lighting=False, opacity=args.plane_edge_opacity,
                         name='mirror_outline_{}'.format(i))

    # Create body surface from REAL polygon faces reconstructed earlier.
    # It is not ConvexHull's triangle mesh, so artificial face diagonals are absent.
    surface = pv.PolyData(vertices, faces)
    # BODY MATERIAL SETTINGS:
    #   color       = --face-color (default muted blue)
    #   opacity     = --opacity (0 clear; 1 fully opaque; default 0.50)
    #   flat shading= smooth_shading=False (facet normal stays visually distinct)
    #   show_edges  = False (physical edges are a separate pass below)
    #   ambient     = 0.24 (base illumination without direct light)
    #   diffuse     = 0.86 (Lambertian lit body)
    #   specular    = 0.12 (strength of glossy highlight)
    #   specular_power=16 (highlight concentration)
    # TWEAK HERE for metallic/glossy/flat body appearance. If increasing opacity,
    # mirror sections behind the body may be hidden; that is expected occlusion.
    plotter.add_mesh(
        surface,
        color=args.face_color,
        opacity=args.opacity,
        smooth_shading=False,
        show_edges=False,
        ambient=0.24,
        diffuse=0.86,
        specular=0.12,
        specular_power=16,
        name='polyhedron',
    )

    # Turn each unique physical edge (i,j) into a VTK 2-point LINE CELL.
    # A single shared edge will not double-draw because 'edges' was a set.
    edge_cells = np.array([[2, i, j] for i, j in edges], dtype=np.int64).ravel()
    # Wireframe uses same vertex coordinates as surface; the lines are
    # drawn as tubes rather than pixel-thin OpenGL lines to improve visibility.
    edge_mesh = pv.PolyData(vertices)
    edge_mesh.lines = edge_cells
    # --edge-width is multiplied by 1.1 before passing to VTK's visual
    # line_width. It is distinct from the mirror border's 3D tube radius.
    # TWEAK HERE if you want a darker outline or thicker real body edges.
    plotter.add_mesh(edge_mesh, color=args.edge_color,
                     line_width=args.edge_width * 1.1, render_lines_as_tubes=True,
                     lighting=False, opacity=1.0, name='physical_edges')

    # Loop each UNIQUE undirected symmetry-axis line, not each rotation.
    # C4 has one cylinder even though 3 nonidentity rotations share that line.
    for i, axis in enumerate(axes):
        # order is a positive integer >=2; direction is a unit 3-vector;
        # color is resolved by --cN-color or DEFAULT_AXIS_COLORS/fallback.
        order = axis['order']
        direction = axis['direction']
        color = axis_color_for_order(order, args)
        # pv.Cylinder(center, direction, radius, height) creates a finite
        # solid 3D cylinder along the mathematical symmetry line.
        # The original code multiplies base axis radius by 1.35 for C3+ and 1.22
        # for C2, making higher-order axes visibly thicker. For a UNIFORM width,
        # replace the order-dependent factor with 1.0 (this would change behavior).
        # resolution=48 specifies number of sides on circular cylinder perimeter;
        # more sides = smoother at higher rendering cost. capping fills ends.
        tube = pv.Cylinder(center=center, direction=direction,
                           radius=axis_radius * (1.35 if order > 2 else 1.22),
                           height=2 * axis_half_length, resolution=48,
                           capping=True)
        # AXIS MATERIAL: axis_opacity controls transparency; ambient/diffuse
        # and specular/specular_power yield glossy colored axes. Unlike the mirror
        # surface, cylinders participate in the custom lighting setup.
        plotter.add_mesh(tube, color=color, opacity=args.axis_opacity,
                         ambient=0.42, diffuse=0.78, specular=0.26,
                         specular_power=26, name='symmetry_axis_{}'.format(i))
        # Add small sphere at EACH cylinder endpoint (+u and -u).
        # This creates a rounded-finish aesthetic. Spheres have radius
        # axis_radius*1.16, measured against the base axis radius (not the C2/C3
        # inflated cylinder radius). Increase if you want bulbous tips.
        for sign in (-1, 1):
            cap = pv.Sphere(radius=axis_radius * 1.16,
                            center=center + sign * direction * axis_half_length,
                            theta_resolution=16, phi_resolution=16)
            # Cap color equals cylinder color, maintaining C_n visual grouping.
            # Endpoint spheres' material is slightly different from cylinder's.
            plotter.add_mesh(cap, color=color, opacity=args.axis_opacity,
                             ambient=0.45, diffuse=0.55,
                             name='cap_{}_{}'.format(i, sign))

    # MIRROR LABELS are optional to avoid clutter on complex shapes.
    # They are NOT included in the geometry report. Place a label at a farthest
    # patch boundary point, choosing 'sigma_h' vs 'mirror' by classification.
    # WARNING: the patch generation is repeated here rather than cached from
    # the render loop. If you change patch types, update BOTH dispatch blocks.
    if args.plane_labels and visible_planes:
        # Place one label near the boundary of each plane; disabled by default.
        label_points = []
        label_text = []
        # The following four branches intentionally mirror the fill-plane
        # geometry switch above. If labels visually drift from plane boundaries,
        # check for inconsistent style settings between both locations.
        for plane in visible_planes:
            if args.plane_style == 'rectangle':
                points = mirror_rectangle_patch(
                    center, plane['normal'], vertices,
                    span=args.plane_span, aspect=args.plane_aspect)
            elif args.plane_style == 'square':
                points = mirror_square_patch(center, plane['normal'], radius, args.plane_span)
            elif args.plane_style == 'disc':
                points = mirror_disc_patch(center, plane['normal'], radius,
                                            args.plane_span, args.plane_resolution)
            elif args.plane_style == 'section':
                points = mirror_section_polygon(vertices, edges, center,
                                                plane['normal'], args.plane_extension)
            else:
                raise ValueError('Unsupported plane style: {}'.format(args.plane_style))
            # Label at perimeter vertex maximally distant from body center;
            # that often moves text outside the body. Multiple labels can still overlap.
            label_points.append(points[np.argmax(np.linalg.norm(points - center, axis=1))])
            label_text.append('sigma_h' if plane['is_sigma_h'] else 'mirror')
        # Font size is tied to width; shape=None removes label background;
        # always_visible means labels may appear over a foreground surface.
        # 'plane-edge-color' is used for ALL plane labels in the original code.
        plotter.add_point_labels(np.asarray(label_points), label_text,
                                 font_size=max(14, args.resolution // 125),
                                 text_color=args.plane_edge_color,
                                 shape=None, show_points=False,
                                 always_visible=True, name='plane_labels')

    # --show-vertices draws small spheres at every physical corner.
    # For large N, choose smaller point_size to avoid hiding narrow faces.
    if args.show_vertices:
        plotter.add_mesh(pv.PolyData(vertices), color=args.edge_color,
                         point_size=9, render_points_as_spheres=True,
                         name='vertices')

    # --labels (plural) controls C_n AXIS labels, not mirror labels.
    # Label position is 1.12 times axis_half_length beyond center along +u.
    # Axis direction sign is chosen canonically, so labels use one selected end.
    if args.labels and axes:
        positions = np.array([center + a['direction'] * axis_half_length * 1.12
                              for a in axes])
        # Each label 'C2','C3',etc. is repeated for every axis of same order.
        # Change format here if labels should include stable axis index, e.g. 'C4 #2'.
        labels = ['C{}'.format(a['order']) for a in axes]
        # Adding many always-visible text labels may clutter transparent PNGs.
        # To improve readability, add placement/overlap handling here.
        plotter.add_point_labels(
            positions, labels, font_size=max(16, args.resolution // 105),
            text_color='#202C36', shape=None, show_points=False,
            always_visible=True, name='axis_labels',
        )

    # --show-legend is an in-image legend summarizing present C_n axis
    # orders and visible mirror categories. Counts are axis COUNTS, not counts of
    # individual rotations. For per-plane unique colors, this logic will need a
    # legend policy (one entry per plane or just one entry for mirror category).
    if args.show_legend and (axes or visible_planes):
        legend_rows = []
        # A sorted SET of orders avoids repeated legend entries. Count all
        # axis objects carrying that C_n order and color by the same resolver.
        for order in sorted(set(a['order'] for a in axes), reverse=True):
            count = sum(a['order'] == order for a in axes)
            legend_rows.append(['C{} axes  ({})'.format(order, count), axis_color_for_order(order, args)])
        # Mirror counts are drawn-only counts: if plane_mode='horizontal',
        # non-horizontal mirrors are omitted from this legend even if report has them.
        if visible_planes:
            n_horizontal = sum(p['is_sigma_h'] for p in visible_planes)
            if n_horizontal:
                legend_rows.append(['sigma_h planes ({})'.format(n_horizontal),
                                    args.plane_color])
            n_other = len(visible_planes) - n_horizontal
            if n_other:
                legend_rows.append(['other mirrors ({})'.format(n_other),
                                    args.other_plane_color])
        # Legend position fixed upper right, width=21% of image,
        # height rises with number of rows capped at 40% of image height.
        plotter.add_legend(legend_rows, border=False, loc='upper right',
                           size=(0.21, min(0.09 * len(legend_rows) + 0.035, 0.4)))
    # --title overlays string at upper left. This text is not included
    # in JSON report. Modify font and position here if making presentation slides.
    if args.title:
        plotter.add_text(args.title, position='upper_left', color=args.edge_color,
                         font_size=max(14, args.resolution // 120), font='arial')
    # --show-xyz adds a PyVista orientation axes widget for reference.
    # This changes visual overlay only; no coordinates are transformed.
    if args.show_xyz:
        plotter.add_axes(line_width=2)

    # Plotter stays open after return. render_images owns its lifetime and
    # must close it in a finally block, even on screenshot errors.
    return plotter, radius, axis_half_length


# =========================== PNG RENDERING LOOP =============================
# Purpose: save N distinct 4-channel (RGBA) PNG files with deterministic filenames.
# The original code constructs a NEW PyVista plotter on every iteration as a
# workaround for cached/stale render windows across different camera poses.
# Each PNG is independently composited on a transparent background.
# Inputs: geometry+symmetry lists, argparse Namespace.
# Outputs: PNG files and a .views.json manifest; no Python value returned.
#
# IMPORTANT OUTPUT PATH EXAMPLE:
#   --output figures/object.png --n-views 2
#     -> figures/object_view_001.png
#     -> figures/object_view_002.png
#     -> figures/object.views.json
#   The single literal 'figures/object.png' is a PREFIX TEMPLATE; it is not
#   additionally saved as a whole-scene PNG by this function.
#
# DO NOT reorder the delicate show() -> set_camera() -> Modified() -> render()
# -> screenshot() sequence without testing: the source explicitly protects
# against some PyVista versions returning previously cached images.
def render_images(vertices, center, faces, edges, axes, planes, args):
    """Save N RGBA PNGs, each from a NEW render window with a NEW camera."""
    # Generate all camera vectors up-front, allowing consistent ordering
    # and preserving a deterministic sequence for a fixed seed.
    directions = random_camera_directions(args.n_views, seed=args.seed)
    # mkdir(parents=True,exist_ok=True) creates nested output directories
    # without failing if they already exist. Be aware existing output PNGs with
    # same names may be overwritten.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Use at least three digits (001..999), but expand if n_views has
    # more digits. This makes filenames naturally lexicographically sortable.
    pad_width = max(3, len(str(args.n_views)))
    # metadata list will feed .views.json; image_fingerprints is a SET
    # used to reject exact repeats in screenshot pixels.
    metadata = []
    image_fingerprints = set()

    # 1-based iteration index is for files/report humans; Python arrays
    # remain zero-based. 'direction' gives camera location relative to object.
    for i, direction in enumerate(directions, start=1):
        # Independent render window prevents cached/stale screenshots.
        # TWEAK HERE only if you intentionally want to reuse a plotter;
        # that optimization risks reintroducing the stale framebuffer bug the original
        # script was designed to prevent.
        plotter, radius, axis_half_length = create_scene(
            vertices, center, faces, edges, axes, planes, args)
        # Build file name from output stem, an incrementing view number,
        # and '.png'. The output suffix was validated in main.
        filename = '{}_view_{:0{width}d}.png'.format(
            args.output.stem, i, width=pad_width)
        path = args.output.parent / filename

        # try/finally below guarantees plotter.close() even when a rendering
        # exception occurs, preventing GPU/VTK resource leaks across failed views.
        try:
            # Initialize the render window BEFORE applying the requested
            # camera, because show() may reset cameras on older PyVista.
            # Interactive preview occurs only on view #1; press q to close the
            # preview, then the program sets a separately sampled random camera before
            # saving. The preview's hand-adjusted viewpoint is not persisted.
            if args.interactive and i == 1:
                print('[info] Preview view 1, then press q to continue.')
                plotter.show(auto_close=False)
            else:
                plotter.show(interactive=False, auto_close=False)

            # Set camera only AFTER show() has created a real render window.
            # Some versions of PyVista reset camera state when the window initializes.
            position, view_up = set_camera_for_direction(
                plotter, direction, center, radius, axis_half_length)

            # CRITICAL FIX: update the camera, then force a FRESH frame. A call
            # to screenshot() by itself may read the previously rendered frame.
            # Explicitly mark VTK camera modified and force a fresh render.
            # A direct screenshot() without these calls may capture the previous image.
            plotter.camera.Modified()
            plotter.render()

            # Screenshot transparency belongs here, not in Plotter.show().
            # transparent_background=True is the decisive RGBA PNG option.
            # return_img=True also yields the image array so we can verify it is RGBA.
            # For an intentional solid background (code change), switch this to False
            # and revisit the 4-channel validation and manifest claims.
            img = plotter.screenshot(
                filename=str(path),
                transparent_background=True,
                return_img=True,
            )
            # Validate 3 dimensions (height,width,channels) and exactly 4
            # channels (R,G,B,A). A 3-channel RGB screenshot is NOT transparent, even if
            # it visually happens to contain black/white pixels in the background.
            if img is not None:
                img = np.asarray(img)
                if img.ndim != 3 or img.shape[2] != 4:
                    raise RuntimeError(
                        'Expected a transparent RGBA screenshot, got image shape {}. '
                        'Check PyVista/VTK transparency support.'.format(img.shape))

                # SHA-256 hashes raw image array bytes, NOT compressed PNG file bytes.
                # Two views producing byte-identical images trigger an error because that may
                # indicate a stale render buffer. Highly symmetric shapes can legitimately
                # produce the same raster image from some camera directions; the check is
                # strict and may need refinement in a reproducible imaging workflow.
                # Image-content check catches stale framebuffers immediately.
                # Tiny 3D differences can be lost by compression; compare bytes.
                import hashlib
                fingerprint = hashlib.sha256(img.tobytes()).hexdigest()
                if fingerprint in image_fingerprints:
                    raise RuntimeError(
                        'Two camera views produced pixel-identical images. '
                        'The render backend may still be caching frames. '
                        'Try updating PyVista/VTK or run with a working OpenGL display.')
                image_fingerprints.add(fingerprint)

            # Manifest records enough information to reconstruct a camera pose:
            # unit direction, actual position, focal center, chosen view-up, and filename.
            # All NumPy vectors are converted to normal JSON lists.
            metadata.append({
                'view_number': i,
                'image_filename': filename,
                'camera_direction_unit_vector': direction.tolist(),
                'camera_position': position.tolist(),
                'focal_point': center.tolist(),
                'camera_view_up': view_up.tolist(),
            })
            # Status print flushes stdout immediately for visibility in shell
            # pipelines or logs; not relevant to the rendered image.
            print('[saved] View {}/{}: {} | direction {}'.format(
                i, args.n_views, path.resolve(), np.round(direction, 4)),
                flush=True)
        # ALWAYS close every PyVista Plotter, even if image is identical,
        # transparency fails, I/O fails, or the user interrupts rendering.
        finally:
            plotter.close()

    # This manifest is saved ONLY after all requested views complete.
    # A mid-loop exception may leave partial PNGs without a fresh .views.json.
    # Consider saving incrementally if supporting long/interruptible render runs.
    manifest_path = args.output.with_suffix('.views.json')
    # 'projection' and 'rendering_mode' values are metadata strings,
    # not instructions read by PyVista. Keep consistent if changing the algorithm.
    # The optional one-based principal-axis index is included for reproducibility.
    manifest = {
        'n_views': args.n_views,
        'random_seed': args.seed,
        'transparent_background': True,
        'projection': 'orthographic',
        'plane_mode': args.plane_mode,
        'horizontal_reference': args.horizontal_reference,
        'principal_axis_index_1_based': args.principal_axis_index,
        'rendering_mode': 'fresh_scene_per_view',
        'image_resolution': [args.resolution, args.height or args.resolution],
        'views': metadata,
    }
    # Pretty-print the manifest as a standalone JSON file alongside PNGs.
    # TWEAK HERE to include image checksums, per-view clipping, render settings,
    # plane styles, color palette, or full reproducibility metadata.
    with open(manifest_path, 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    print('[saved] Camera manifest: {}'.format(manifest_path.resolve()))


# =========================== ARGUMENT PARSER / CLI ===========================
# ALL options users can pass on the command line are declared below.
# A default=... changes what happens when the user OMITS that flag; changing a
# help=... string changes only the printed --help text, not the behavior.
# For a new flag: (1) declare parser.add_argument, (2) validate in main as
# appropriate, (3) USE args.new_flag in the correct functional section.
# The third step is essential: an argument that is merely parsed is a no-op.
#
# EXAMPLE (render one known JSON shape):
#   python render_polyhedron_v5p0_FULLY_DOCUMENTED.py shape.json \
#     --n-views 4 --seed 42 --output results/demo.png --resolution 1200 \
#     --plane-mode all --plane-style disc --plane-span 1.35 \
#     --face-color '#7098B7' --axis-thickness 0.014 --show-legend
#
# EXAMPLE (analyze only / no PyVista needed):
#   python render_polyhedron_v5p0_FULLY_DOCUMENTED.py shape.json \
#     --check-only --output results/demo.png
# Even in check-only mode, the output path MUST have .png suffix; the
# report is created at results/demo.axes.json unless --report overrides it.
def parse_arguments():
    # ArgumentDefaultsHelpFormatter exposes default values in --help.
    # This is helpful when tuning visual settings without reading the code.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Render N transparent views of a convex polyhedron, its C_n axes, and its horizontal mirror planes.',
    )
    # POSITIONAL INPUT: shape_json is mandatory; it is not a --flag.
    # A pathlib.Path is used for filesystem access; file must exist and be JSON.
    parser.add_argument('shape_json', type=Path,
                        help='Shape JSON with a 3D polyhedron vertex array')
    # OUTPUT PREFIX: name.png becomes name_view_001.png etc. and
    # name.axes.json/name.views.json. To organize, pass a parent folder path.
    parser.add_argument('--output', type=Path, default=Path('polyhedron_symmetry.png'),
                        help='Output name prefix: name.png generates name_view_001.png, etc.')
    # REPORT DESTINATION: default None means main derives .axes.json;
    # use --report custom.json to send analysis to a different folder.
    parser.add_argument('--report', type=Path, default=None,
                        help='Symmetry JSON report path (default: output stem + .axes.json)')
    # --check-only is Boolean (present=True, absent=False).
    # It skips both interactive N prompt and PyVista rendering, not analysis/report.
    parser.add_argument('--check-only', action='store_true',
                        help='Only detect symmetry, without rendering or prompting for views')
    # --n-views controls PNG count. If omitted at interactive terminal,
    # main prompts for a positive number; noninteractive shells must supply it.
    parser.add_argument('--n-views', type=int, default=None,
                        help='Number N of random screenshots; prompted if omitted at a terminal')
    # --seed fixes NumPy camera-direction generator. It doesn't control
    # numerical eigensolver/SVD order or graphics backend nondeterminism.
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional seed for reproducible random camera directions')
    # IMAGE RESOLUTION: width is --resolution (default 2400 px);
    # --height overrides height. Without override, output is square.
    parser.add_argument('--resolution', type=int, default=2400,
                        help='Image width in pixels')
    parser.add_argument('--height', type=int, default=None,
                        help='Image height in pixels; default = width')
    # FACE VISUALS: these change only polyhedron BODY material.
    # --face-color accepts color strings; --edge-color controls real edge lines;
    # --opacity controls surface alpha; --edge-width is a VTK line-width parameter.
    parser.add_argument('--face-color', default='#7098B7')
    parser.add_argument('--edge-color', default='#DCE8F0')
    parser.add_argument('--opacity', type=float, default=0.50,
                        help='Polyhedron face opacity from 0 to 1')
    parser.add_argument('--edge-width', type=float, default=1.8)
    # --plane-mode controls WHICH detected mirrors are drawn.
    # Do not confuse 'horizontal' classification with a global XY slice.
    parser.add_argument('--plane-mode', choices=('horizontal', 'all', 'none'),
                        default='horizontal',
                        help='Show sigma_h relative to principal axes, every mirror, or no planes')
    # --horizontal-reference sets mathematical sigma_h reference group.
    # 'maximum' uses every C_n axis tied for highest order; 'all' uses every axis.
    parser.add_argument('--horizontal-reference', choices=('maximum', 'all'),
                        default='maximum',
                        help='Classify sigma_h relative to all maximal-order axes or to every C_n axis')
    # --principal-axis-index is ONE-BASED as printed in console/report.
    # It overrides horizontal-reference; main converts to zero-based index.
    parser.add_argument('--principal-axis-index', type=int, default=None,
                        help='Optional 1-based listed C_n axis defining sigma_h, overriding horizontal-reference')
    # HORIZONTAL MIRROR APPEARANCE:
    # --plane-color is fill, --plane-edge-color is outline; each color is shared
    # by all sigma_h planes by default. --plane-opacity default is only 0.08,
    # so plane fill is extremely subtle while the border remains visible.
    parser.add_argument('--plane-color', default='#66D1C7',
                        help='Color for sigma_h planes')
    parser.add_argument('--plane-edge-color', default='#8FF5EA',
                        help='Muted teal boundary for sigma_h planes')
    parser.add_argument('--plane-opacity', type=float, default=0.08,
                        help='Sigma_h surface opacity (0..1)')
    # --plane-edge-opacity adjusts the tube-shaped OUTLINE alpha.
    # --plane-edge-width is currently unused despite being parsed: outline's
    # real tube thickness is controlled by --plane-border-radius below.
    # To activate this legacy flag, decide whether it means tube radius, screen
    # pixels, or another property, then implement that explicitly in create_scene.
    parser.add_argument('--plane-edge-opacity', type=float, default=0.55)
    parser.add_argument('--plane-edge-width', type=float, default=1.4)
    # --plane-extension applies ONLY to --plane-style section.
    # 1.00 = exact body slice; 1.10 extends 10% radially beyond center.
    parser.add_argument('--plane-extension', type=float, default=1.10,
                        help='Factor to extend each mirror section beyond the polyhedron')
    # --plane-style selects finite visualization geometry, NOT different
    # mathematical planes. rectangle(default), disc, square, and section are the
    # four implemented styles in create_scene.
    parser.add_argument('--plane-style', choices=('rectangle', 'disc', 'square', 'section'), default='rectangle',
                        help='Choose a rectangular, circular, square or polyhedral cross-section mirror patch')
    # --plane-span applies to rectangle/square/disc; section instead uses
    # --plane-extension. In rectangle it scales BODY-PROJECTED half-extents;
    # in square/disc it scales body RADIUS to get patch size.
    parser.add_argument('--plane-span', type=float, default=1.15,
                        help='Scale of rectangle dimensions relative to polyhedron, or square/disc relative to body radius')
    # --plane-aspect applies only to rectangle: minimum hx/hy ratio.
    # Values <=1 are rejected by main, even though they might have meanings
    # for alternative rectangle conventions.
    parser.add_argument('--plane-aspect', type=float, default=1.22,
                        help='Minimum long/short half-extent ratio for rectangular planes')
    # --plane-border-radius controls outline tube radius in units of the
    # polyhedron body radius; larger makes all mirror boundaries bolder.
    parser.add_argument('--plane-border-radius', type=float, default=0.006,
                        help='Radius of colored rectangular plane borders, in body-radius units')
    # ATTENTION --plane-distinct-colors is NOT IMPLEMENTED in original
    # create_scene despite this parsed flag's claim. It defaults to True even if
    # not supplied; --uniform-plane-color sets it False, but neither value is read.
    # To make it functional, add palette selection inside the visible_planes loop
    # and decide whether color must be reproducible by plane index/normal.
    parser.add_argument('--plane-distinct-colors', action='store_true', default=True,
                        help='Use a separate pastel color for each sigma plane (default)')
    parser.add_argument('--uniform-plane-color', action='store_false',
                        dest='plane_distinct_colors',
                        help='Use the same color for all sigma planes')
    # --plane-resolution sets N sides of a disc approximated as a polygon.
    # Low N yields visibly angular circle. Minimum accepted by main is 12.
    parser.add_argument('--plane-resolution', type=int, default=96,
                        help='Number of boundary segments for disc-style mirror planes')
    # OTHER MIRRORS are mirrors not classified sigma_h for the selected
    # reference. Their fill/edge/alpha options matter only with --plane-mode all.
    parser.add_argument('--other-plane-color', default='#C6B8DE',
                        help='Color for non-horizontal mirrors when --plane-mode all')
    parser.add_argument('--other-plane-edge-color', default='#9585B2')
    parser.add_argument('--other-plane-opacity', type=float, default=0.11)
    # --plane-labels adds 'sigma_h'/'mirror' near the finite patch boundary.
    # For more descriptive text use label_text construction in create_scene.
    parser.add_argument('--plane-labels', action='store_true',
                        help='Label mirror planes (off by default to avoid clutter)')
    # --axis-color-fallback applies to orders missing a specific --cN-color
    # (and not present in DEFAULT_AXIS_COLORS). Existing --c6-color IS defined,
    # so its color is not taken from fallback in normal use.
    parser.add_argument('--axis-color-fallback', default='#37474F',
                        help='Fallback color for unlisted axis orders (C6 and above if not overridden)')
    # All --c2-color ... --c6-color flags override their corresponding
    # C_n entries in DEFAULT_AXIS_COLORS. Alter defaults here for regular use;
    # keep the map and argument defaults synchronized for clarity.
    parser.add_argument('--c2-color', default=DEFAULT_AXIS_COLORS[2],
                        help='Color for C2 axes (default: dark purple)')
    parser.add_argument('--c3-color', default=DEFAULT_AXIS_COLORS[3],
                        help='Color for C3 axes (default: dark orange)')
    parser.add_argument('--c4-color', default=DEFAULT_AXIS_COLORS[4],
                        help='Color for C4 axes (default: dark green)')
    parser.add_argument('--c5-color', default=DEFAULT_AXIS_COLORS[5],
                        help='Color for C5 axes (default: dark brown)')
    parser.add_argument('--c6-color', default=DEFAULT_AXIS_COLORS[6],
                        help='Color for C6 axes (default: dark blue)')
    # AXIS GEOMETRY: --axis-length is HALF-length / body_radius.
    # For example 1.45 means full cylinder length is 2*1.45*body_radius.
    # --axis-thickness controls BASE cylinder radius / body_radius; create_scene
    # multiplies it by 1.22 for C2 and by 1.35 for larger order axes.
    parser.add_argument('--axis-length', type=float, default=1.45,
                        help='Axis HALF-length relative to maximum vertex radius')
    parser.add_argument('--axis-thickness', type=float, default=0.018,
                        help='Axis cylinder radius relative to maximum vertex radius')
    # --axis-opacity changes axis cylinder AND endpoint-sphere transparency.
    # For transparent axes, overlapping materials may depend on depth peeling.
    parser.add_argument('--axis-opacity', type=float, default=1.0)
    # --symmetry-tol is a relative numerical recognition threshold,
    # NOT a visual setting. Smaller = stricter, may miss slightly noisy true
    # symmetry; larger = more permissive, may recognize false symmetries.
    # It passes into BOTH proper and improper detection routines.
    parser.add_argument('--symmetry-tol', type=float, default=1e-6,
                        help='Vertex matching tolerance relative to body radius')
    # ANNOTATION SWITCHES:
    # --labels adds C_n labels beside axes;
    # --show-legend adds a categorical legend;
    # --show-vertices adds physical corner markers;
    # --show-xyz adds a coordinate orientation widget;
    # --title adds overlay text in the upper-left;
    # --interactive previews before saving random-view images.
    parser.add_argument('--labels', action='store_true',
                        help='Label symmetry axes (off by default)')
    parser.add_argument('--show-legend', action='store_true',
                        help='Include a legend (off by default)')
    parser.add_argument('--show-vertices', action='store_true')
    parser.add_argument('--show-xyz', action='store_true')
    parser.add_argument('--title', default='')
    parser.add_argument('--interactive', action='store_true',
                        help='Preview scene interactively before N random screenshots')
    # Legacy --transparent is accepted silently but has no conditional
    # code path: screenshots always use transparent_background=True.
    # Compatibility: old --transparent flag is accepted, but is redundant.
    parser.add_argument('--transparent', action='store_true', help=argparse.SUPPRESS)
    # parse_args() validates basic argparse types/choices; semantic
    # range checks and dependent defaults happen in main().
    return parser.parse_args()


# =========================== MAIN PROGRAM DRIVER =============================
# This is the orchestration layer and preferred starting point when stepping
# through execution in a debugger. All analysis happens BEFORE rendering.
#   1. Parse CLI values.
#   2. Validate ranges and required output extension.
#   3. Read/validate N for rendering only.
#   4. Read vertices and reconstruct physical polygon faces.
#   5. Detect proper rotations, extract axes, find all pure reflections.
#   6. Classify sigma_h relative to chosen reference axes.
#   7. Save JSON report and print human-readable symmetry summary.
#   8. Unless --check-only, render N RGBA PNGs and write view manifest.
#
# TWEAK HERE if you want to add an analysis stage, new report type, alternate
# renderer, batch input files, or a different program entry pipeline.
def main():
    # args is an argparse.Namespace. Attributes use underscores, not hyphens:
    # --symmetry-tol -> args.symmetry_tol; --plane-style -> args.plane_style.
    args = parse_arguments()
    # Validate all visible alpha values are in [0,1]. These checks fail
    # fast rather than delegating invalid ranges to PyVista/OpenGL.
    for name, value in [('opacity', args.opacity),
                        ('axis_opacity', args.axis_opacity),
                        ('plane_opacity', args.plane_opacity),
                        ('plane_edge_opacity', args.plane_edge_opacity),
                        ('other_plane_opacity', args.other_plane_opacity)]:
        if not 0 <= value <= 1:
            raise ValueError('--{} must be between 0 and 1'.format(name.replace('_', '-')))
    # Require at least 256 pixels in each dimension to avoid extremely
    # small/unreadable output. For thumbnail workflows, change this explicit rule.
    if args.resolution < 256 or (args.height is not None and args.height < 256):
        raise ValueError('Image dimensions must be at least 256 pixels.')
    # Validate plane geometry controls as used by original code.
    # Notice --plane-edge-width is validated but does not change rendered width.
    # --plane-aspect must be >1; disc polygon resolution must be >=12.
    if (args.plane_extension <= 0 or args.plane_edge_width <= 0 or
            args.plane_span <= 0 or args.plane_aspect <= 1 or
            args.plane_border_radius <= 0 or args.plane_resolution < 12):
        raise ValueError('Plane extension, plane span, plane resolution and edge width must be positive.')
    # Reject physically nonsensical nonpositive tolerance or cylinder
    # length/thickness. This does not validate excessively large positive values.
    if args.symmetry_tol <= 0 or args.axis_length <= 0 or args.axis_thickness <= 0:
        raise ValueError('Symmetry tolerance, axis length and thickness must be positive.')
    # Require output '*.png' suffix because it is used as a template for
    # many .png views and for deriving '.axes.json' / '.views.json'.
    if args.output.suffix.lower() != '.png':
        raise ValueError('--output must be a .png filename, e.g. figures/my_shape.png')

    # When --check-only, N is unnecessary because no views are made.
    # Otherwise, prefer explicit --n-views for reproducible batch scripts.
    if not args.check_only:
        # Only ask interactively if stdin is a real terminal (isatty).
        # CI/notebooks/pipelines must provide --n-views to avoid hanging for input.
        if args.n_views is None:
            if not sys.stdin.isatty():
                raise ValueError('Supply --n-views N when running without a terminal.')
            response = input('How many random viewing angles (N)? ').strip()
            try:
                args.n_views = int(response)
            except ValueError as exc:
                raise ValueError('Number of views N must be a positive integer.') from exc
        # Views count must be a positive integer. Negative or zero would
        # produce no images and misleading manifest metadata.
        if args.n_views <= 0:
            raise ValueError('--n-views must be a positive integer.')

    # DATA FLOW BEGINS. After read_vertices, all downstream arrays index
    # into the filtered hull-vertex array, not original JSON point order if interior
    # points were removed.
    metadata, vertices, center, hull = read_vertices(args.shape_json)
    # Make real faces and edges for the renderer. For physical polygon
    # edge algorithms, use these edges rather than hull.simplices triangle edges.
    faces, edges, n_faces = polygon_faces_from_hull(vertices, hull)
    # Find the complete proper rotational symmetry group. If this fails
    # or its closure check errors, no report is generated.
    operations = find_proper_rotations(vertices, center, args.symmetry_tol)
    # Convert matrices into unique undirected physical axis lines with
    # C_n orders. The sort order determines principal-axis-index semantics.
    axes = symmetry_axes(operations)
    # Find ALL actual mirrors, then tag the ones horizontal relative to
    # chosen reference axes. The nested call makes it explicit that classification
    # does NOT change what is accepted as a mathematical mirror plane.
    planes = classify_mirror_planes(
        find_mirror_planes(vertices, center, operations, args.symmetry_tol),
        axes, reference=args.horizontal_reference,
        principal_index=None if args.principal_axis_index is None
        else args.principal_axis_index - 1)

    # Unless --report is supplied, suffix '.axes.json' is derived from
    # --output. Report directories are created if missing.
    report_path = args.report or args.output.with_suffix('.axes.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # save_report has file I/O side effect (even with --check-only).
    # The returned Python dict is then used to print summary counts.
    report = save_report(report_path, metadata, vertices, center, operations, axes, planes)

    # Summary output names shape, effective hull vertex count, physical
    # face/edge counts, and number of proper group elements including identity.
    print('Shape: {}'.format(report['name']))
    print('Geometry: {} vertices, {} faces, {} physical edges'.format(
        len(vertices), n_faces, len(edges)))
    print('Proper rotation group: {} operations (including identity)'.format(len(operations)))
    # Print each C_n type count then every axis (ONE-BASED) and its unit
    # direction. Copy the displayed axis index into --principal-axis-index if you
    # want to classify horizontal mirrors against one particular line.
    if axes:
        for order, count in sorted(((int(k[1:]), v) for k, v in
                                    report['rotation_axis_counts'].items()), reverse=True):
            print('  C{}: {} distinct undirected axis/axes'.format(order, count))
        for i, item in enumerate(axes, 1):
            print('  Axis {:>2}: C{}, direction {}'.format(i, item['order'],
                                                          np.round(item['direction'], 7)))
    # Without any nonidentity rotation, the object is C1 only; there
    # is no nontrivial C_n axis and classify_mirror_planes assigns no sigma_h.
    else:
        print('No nontrivial proper rotational symmetry axes detected (C1).')
    # Create a human-readable description of sigma_h reference semantics
    # for the console; mathematical classification already happened above.
    reference_label = ('selected C_n axis {}'.format(args.principal_axis_index)
                       if args.principal_axis_index is not None else
                       'highest-order C_n axes' if args.horizontal_reference == 'maximum'
                       else 'all C_n axes')
    print('Mirror planes: {} total, {} sigma_h (relative to {})'.format(
        report['n_mirror_planes'], report['n_sigma_h_planes'], reference_label))
    # For each plane print normal, sigma_h flag and matching reference
    # C_n orders. A '-' indicates no matching selected reference axis.
    for i, plane in enumerate(planes, 1):
        matched = ', '.join('C{}'.format(axes[j]['order'])
                            for j in plane['horizontal_axis_indices']) or '-'
        print('  Plane {:>2}: normal {}, sigma_h={}, perpendicular to {}'.format(
            i, np.round(plane['normal'], 7), plane['is_sigma_h'], matched))
    print('[saved] Symmetry report: {}'.format(report_path.resolve()))

    # If --check-only, STOP after analysis/report. Otherwise call the
    # rendering stage. Note that --plane-mode none does not avoid symmetry work;
    # it only suppresses mirror visualization.
    if not args.check_only:
        render_images(vertices, center, faces, edges, axes, planes, args)


# ========================== PROCESS ENTRY POINT =============================
# Python executes this block only when file is run directly, not imported.
# The catch converts common expected errors to a concise stderr message and
# exit status 1. It does not catch every failure type (e.g. QhullError,
# ImportError outside PyVista, KeyboardInterrupt, or graphics backend crashes).
# TWEAK HERE for traceback logging, custom exception handling, or library use.
if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print('[error] {}'.format(error), file=sys.stderr)
        sys.exit(1)
