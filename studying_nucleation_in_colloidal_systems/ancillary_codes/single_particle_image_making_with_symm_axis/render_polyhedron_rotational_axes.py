#!/usr/bin/env python3
"""
Render proper rotational-symmetry axes of a convex polyhedron from its shape JSON.

python3.8 render_polyhedron_rotational_axes_v1p4.py shape_151_sheared_Cube_deg10_unit_volume_principal_frame.json  --n-views 10 --output sheared_cube.png --resolution 2400 --axis-length 1.55 --axis-thickness 0.012 --seed 42

* Each screenshot has a genuinely transparent (alpha-channel) background.
* All symmetry axes remain uniformly red; flat-shaded body remains blue.
* Random viewing directions are uniformly distributed over a sphere.
* N is supplied by --n-views, or requested interactively at the terminal.
* A --seed permits reproducible viewpoints. Camera metadata are saved to JSON.
* A fresh PyVista scene is built for EACH image to prevent stale cached frames.

Dependencies: numpy, scipy, pyvista (Python 3.8+).

Examples:
    python3.8 render_polyhedron_rotational_axes_v1p4.py shape.json --n-views 10
    python3.8 render_polyhedron_rotational_axes_v1p4.py shape.json \
        --n-views 20 --seed 42 --output figures/sheared_cube.png \
        --resolution 2400 --axis-length 1.55 --axis-thickness 0.012
    python3.8 render_polyhedron_rotational_axes_v1p4.py shape.json --check-only

Output for --output figures/sheared_cube.png and --n-views 3:
    figures/sheared_cube_view_001.png
    figures/sheared_cube_view_002.png
    figures/sheared_cube_view_003.png
    figures/sheared_cube.axes.json
    figures/sheared_cube.views.json

The rotational symmetry search is unchanged from v1p3. Each camera view now
uses a NEW renderer and a forced GPU redraw before screenshot capture. This is
somewhat slower than v1p3 but fixes stale-frame reuse on older PyVista versions.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation


def read_vertices(json_path):
    """Load vertex arrays in common HPMC shape-catalog JSON formats."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    keys = ['vertices', '8_vertices', '12_vertices', 'polyhedron_vertices']
    keys.extend(k for k in data if k.endswith('_vertices') and k not in keys)
    vertices = None
    for key in keys:
        if key in data:
            attempt = np.asarray(data[key], dtype=float)
            if attempt.ndim == 2 and attempt.shape[1] == 3:
                vertices = attempt
                break
    if vertices is None:
        raise ValueError('No (N, 3) vertex array found in the JSON file.')
    if len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError('At least four finite 3D vertices are required.')

    # Remove redundant points that do not belong to the convex hull.
    hull = ConvexHull(vertices)
    if len(hull.vertices) < len(vertices):
        print('[info] Ignoring {} redundant/interior vertices.'.format(
            len(vertices) - len(hull.vertices)))
        vertices = vertices[np.sort(hull.vertices)]
        hull = ConvexHull(vertices)

    # Symmetry center is guaranteed by the finite set's rotational group;
    # it need not equal the centroid of the solid's volume.
    center = vertices.mean(axis=0)
    return data, vertices, center, hull


def polygon_faces_from_hull(vertices, hull, rel_tol=1e-7):
    """
    Merge ConvexHull coplanar triangles into true polygon faces.

    Crucial for publication images: a quadrilateral face should not show the
    fictitious diagonal inserted by the hull's triangulation.

    Returns (vtk_flat_faces, physical_edges).
    """
    extent = max(np.ptp(vertices, axis=0).max(), 1e-12)
    plane_tol = rel_tol * extent
    groups = []

    for triangle, equation in zip(hull.simplices, hull.equations):
        normal = equation[:3] / np.linalg.norm(equation[:3])
        offset = equation[3] / np.linalg.norm(equation[:3])
        target = None
        for group in groups:
            if (np.dot(normal, group['normal']) > 1 - 1e-8
                    and abs(offset - group['offset']) < plane_tol):
                target = group
                break
        if target is None:
            target = {'normal': normal, 'offset': offset, 'indices': set()}
            groups.append(target)
        target['indices'].update(int(i) for i in triangle)

    vtk_faces = []
    edges = set()
    for group in groups:
        ids = np.array(sorted(group['indices']), dtype=int)
        if len(ids) < 3:
            raise ValueError('Malformed face in convex hull.')
        points = vertices[ids]
        face_center = points.mean(axis=0)
        normal = group['normal']
        u = points[0] - face_center
        u -= np.dot(u, normal) * normal
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)  # u cross v points outward
        relative = points - face_center
        angle = np.arctan2(relative @ v, relative @ u)
        ordered = ids[np.argsort(angle)].tolist()
        vtk_faces.extend([len(ordered)] + ordered)
        for j in range(len(ordered)):
            edges.add(tuple(sorted((ordered[j], ordered[(j + 1) % len(ordered)]))))

    return np.asarray(vtk_faces, dtype=np.int64), sorted(edges), len(groups)


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
    p = vertices - center
    radius = np.linalg.norm(p, axis=1).max()
    if radius <= 0:
        raise ValueError('Degenerate vertices: zero radius.')
    absolute_tol = float(rel_tol) * radius
    dot_tol = 4.0 * absolute_tol * radius

    r2 = np.einsum('ij,ij->i', p, p)
    dot = p @ p.T
    n = len(p)

    # Three well-separated independent anchor vectors are numerically stable.
    a = int(np.argmax(r2))
    b = int(np.argmax(np.linalg.norm(np.cross(p[a], p), axis=1)))
    c = int(np.argmax(np.abs(p @ np.cross(p[a], p[b]))))
    anchors = (a, b, c)
    source = p[list(anchors)]
    if abs(np.linalg.det(source)) < 1e-9 * radius**3:
        raise ValueError('Cannot find three independent radius vectors.')

    lookup = cKDTree(p)
    matrices = []
    for j in np.flatnonzero(np.abs(r2 - r2[a]) <= dot_tol):
        possible_k = np.flatnonzero(
            (np.abs(r2 - r2[b]) <= dot_tol)
            & (np.abs(dot[j] - dot[a, b]) <= dot_tol)
        )
        for k in possible_k:
            possible_l = np.flatnonzero(
                (np.abs(r2 - r2[c]) <= dot_tol)
                & (np.abs(dot[j] - dot[a, c]) <= dot_tol)
                & (np.abs(dot[k] - dot[b, c]) <= dot_tol)
            )
            for l in possible_l:
                target = p[[j, k, l]]
                # Proper (det=+1) orthogonal Procrustes alignment.
                u, _, vt = np.linalg.svd(source.T @ target)
                reflection_fix = np.eye(3)
                reflection_fix[2, 2] = np.sign(np.linalg.det(u @ vt))
                matrix = vt.T @ reflection_fix @ u.T
                if np.linalg.det(matrix) < 1 - 1e-7:
                    continue
                if any(np.linalg.norm(matrix - old) < 1e-6
                       for old in matrices):
                    continue
                distances, nearest = lookup.query(p @ matrix.T, k=1)
                if np.max(distances) <= absolute_tol and len(set(nearest)) == n:
                    matrices.append(matrix)

    # Identity is a symmetry even if numerical candidate reconstruction failed.
    if not any(np.linalg.norm(m - np.eye(3)) < 1e-6 for m in matrices):
        matrices.append(np.eye(3))

    # Check closure: incomplete candidate generation is not silently accepted.
    for ra in matrices:
        for rb in matrices:
            product = ra @ rb
            if not any(np.linalg.norm(product - rc) < 1e-4 for rc in matrices):
                raise RuntimeError('Detected rotations are not closed as a group; '
                                   'try a smaller --symmetry-tol.')
    return matrices


def canonical_axis(vec):
    """Canonicalize undirected axis lines: a and -a denote the same axis."""
    vec = np.asarray(vec, float)
    vec = vec / np.linalg.norm(vec)
    # Stabilize signs independent of Rotation.as_rotvec() conventions.
    imax = int(np.argmax(np.abs(vec)))
    if vec[imax] < 0:
        vec = -vec
    return vec


def symmetry_axes(matrices):
    """Group all nontrivial proper rotations into shared axis LINES."""
    groups = []
    for matrix in matrices:
        rotation_vector = Rotation.from_matrix(matrix).as_rotvec()
        angle = np.linalg.norm(rotation_vector)
        if angle < 1e-5:
            continue
        axis = canonical_axis(rotation_vector / angle)
        best_group = None
        for group in groups:
            if abs(np.dot(group['direction'], axis)) > 1 - 1e-6:
                best_group = group
                break
        if best_group is None:
            best_group = {'direction': axis, 'angles_degrees': []}
            groups.append(best_group)
        best_group['angles_degrees'].append(float(np.degrees(angle)))

    # A C_n axis has n-1 distinct nonidentity proper rotations about the line.
    for group in groups:
        group['order'] = len(group['angles_degrees']) + 1
        group['angles_degrees'].sort()

    groups.sort(key=lambda g: (-g['order'], -g['direction'][2],
                               -g['direction'][1], -g['direction'][0]))
    return groups


def save_report(filename, json_data, vertices, center, rotations, axes):
    counts = defaultdict(int)
    for a in axes:
        counts[a['order']] += 1
    report = {
        'name': json_data.get('1_Name', json_data.get('name', 'polyhedron')),
        'n_vertices': len(vertices),
        'n_proper_rotations_including_identity': len(rotations),
        'rotation_axis_counts': {'C{}'.format(k): int(v)
                                 for k, v in sorted(counts.items())},
        'center': center.tolist(),
        'axes': [
            {'order': int(a['order']),
             'direction_unit_vector': a['direction'].tolist(),
             'angles_degrees': a['angles_degrees']}
            for a in axes
        ],
    }
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    return report


def random_camera_directions(n_views, seed=None):
    """Sample independent, isotropic camera positions on the unit sphere.

    Uniform z=cos(theta) and uniform azimuth produce uniform solid angle,
    unlike independently sampling polar and azimuthal angles.
    """
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=n_views)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, size=n_views)
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack((xy * np.cos(azimuth),
                            xy * np.sin(azimuth), z))


def set_camera_for_direction(plotter, direction, center, radius, axis_half_length):
    """Keep object scale and framing consistent at every camera direction."""
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    view_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(direction, view_up)) > 0.96:
        view_up = np.array([0.0, 1.0, 0.0])

    position = center + 5.0 * radius * direction
    plotter.camera_position = [position.tolist(), center.tolist(), view_up.tolist()]
    plotter.enable_parallel_projection()
    # The bounding sphere fits at all orientations, giving reproducible sizing.
    plotter.camera.parallel_scale = max(axis_half_length * 1.30, radius * 1.65)
    plotter.reset_camera_clipping_range()
    return position, view_up


def create_scene(vertices, center, faces, edges, axes, args):
    """Build an independent PyVista scene, with all geometry and lighting.

    A separate plotter is used for every view because some older PyVista/VTK
    installations keep returning the old image if the camera changes between
    screenshot() calls without a fully refreshed render window.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise SystemExit('PyVista is required for rendering. Install with:\n'
                         '    python -m pip install pyvista') from exc

    radius = float(np.linalg.norm(vertices - center, axis=1).max())
    axis_half_length = args.axis_length * radius
    axis_radius = args.axis_thickness * radius

    plotter = pv.Plotter(
        off_screen=not args.interactive,
        window_size=(args.resolution, args.height or args.resolution),
        lighting='none',
    )
    # Background is intentionally irrelevant: all PNGs are saved with alpha.
    try:
        plotter.enable_anti_aliasing('ssaa')
    except (ValueError, RuntimeError, AttributeError):
        try:
            plotter.enable_anti_aliasing('msaa')
        except (ValueError, RuntimeError, AttributeError):
            pass

    light_specs = [
        ((4, -3, 5), 0.90),
        ((-4, 2, 3), 0.43),
        ((0, 4, -3), 0.25),
    ]
    for point, intensity in light_specs:
        light = pv.Light(position=center + radius * np.array(point),
                         focal_point=center, color='white',
                         intensity=intensity)
        plotter.add_light(light)

    surface = pv.PolyData(vertices, faces)
    plotter.add_mesh(
        surface,
        color=args.face_color,
        opacity=args.opacity,
        smooth_shading=False,
        show_edges=False,
        ambient=0.20,
        diffuse=0.78,
        specular=0.08,
        specular_power=12,
        name='polyhedron',
    )

    edge_cells = np.array([[2, i, j] for i, j in edges], dtype=np.int64).ravel()
    edge_mesh = pv.PolyData(vertices)
    edge_mesh.lines = edge_cells
    plotter.add_mesh(edge_mesh, color=args.edge_color,
                     line_width=args.edge_width, render_lines_as_tubes=True,
                     lighting=False, opacity=0.95, name='physical_edges')

    if args.opacity < 1 and axes:
        try:
            plotter.enable_depth_peeling(number_of_peels=12,
                                         occlusion_ratio=0.05)
        except Exception:
            pass

    for i, axis in enumerate(axes):
        order = axis['order']
        direction = axis['direction']
        color = args.axis_color
        tube = pv.Cylinder(center=center, direction=direction,
                           radius=axis_radius * (1.10 if order > 2 else 1.0),
                           height=2 * axis_half_length, resolution=48,
                           capping=True)
        plotter.add_mesh(tube, color=color, opacity=args.axis_opacity,
                         ambient=0.28, diffuse=0.70, specular=0.22,
                         specular_power=22, name='symmetry_axis_{}'.format(i))
        for sign in (-1, 1):
            cap = pv.Sphere(radius=axis_radius * 1.04,
                            center=center + sign * direction * axis_half_length,
                            theta_resolution=16, phi_resolution=16)
            plotter.add_mesh(cap, color=color, opacity=args.axis_opacity,
                             ambient=0.45, diffuse=0.55,
                             name='cap_{}_{}'.format(i, sign))

    if args.show_vertices:
        plotter.add_mesh(pv.PolyData(vertices), color=args.edge_color,
                         point_size=9, render_points_as_spheres=True,
                         name='vertices')

    if args.labels and axes:
        positions = np.array([center + a['direction'] * axis_half_length * 1.12
                              for a in axes])
        labels = ['C{}'.format(a['order']) for a in axes]
        plotter.add_point_labels(
            positions, labels, font_size=max(16, args.resolution // 105),
            text_color='#202C36', shape=None, show_points=False,
            always_visible=True, name='axis_labels',
        )

    if args.show_legend and axes:
        legend_rows = []
        for order in sorted(set(a['order'] for a in axes), reverse=True):
            count = sum(a['order'] == order for a in axes)
            legend_rows.append(['C{} axes  ({})'.format(order, count), args.axis_color])
        plotter.add_legend(legend_rows, border=False, loc='upper right',
                           size=(0.21, min(0.09 * len(legend_rows) + 0.035, 0.4)))
    if args.title:
        plotter.add_text(args.title, position='upper_left', color=args.edge_color,
                         font_size=max(14, args.resolution // 120), font='arial')
    if args.show_xyz:
        plotter.add_axes(line_width=2)

    return plotter, radius, axis_half_length


def render_images(vertices, center, faces, edges, axes, args):
    """Save N RGBA PNGs, each from a NEW render window with a NEW camera."""
    directions = random_camera_directions(args.n_views, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pad_width = max(3, len(str(args.n_views)))
    metadata = []
    image_fingerprints = set()

    for i, direction in enumerate(directions, start=1):
        # Independent render window prevents cached/stale screenshots.
        plotter, radius, axis_half_length = create_scene(
            vertices, center, faces, edges, axes, args)
        filename = '{}_view_{:0{width}d}.png'.format(
            args.output.stem, i, width=pad_width)
        path = args.output.parent / filename

        try:
            # Initialize the render window BEFORE applying the requested
            # camera, because show() may reset cameras on older PyVista.
            if args.interactive and i == 1:
                print('[info] Preview view 1, then press q to continue.')
                plotter.show(auto_close=False)
            else:
                plotter.show(interactive=False, auto_close=False)

            position, view_up = set_camera_for_direction(
                plotter, direction, center, radius, axis_half_length)

            # CRITICAL FIX: update the camera, then force a FRESH frame. A call
            # to screenshot() by itself may read the previously rendered frame.
            plotter.camera.Modified()
            plotter.render()

            # Screenshot transparency belongs here, not in Plotter.show().
            img = plotter.screenshot(
                filename=str(path),
                transparent_background=True,
                return_img=True,
            )
            if img is not None:
                img = np.asarray(img)
                if img.ndim != 3 or img.shape[2] != 4:
                    raise RuntimeError(
                        'Expected a transparent RGBA screenshot, got image shape {}. '
                        'Check PyVista/VTK transparency support.'.format(img.shape))

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

            metadata.append({
                'view_number': i,
                'image_filename': filename,
                'camera_direction_unit_vector': direction.tolist(),
                'camera_position': position.tolist(),
                'focal_point': center.tolist(),
                'camera_view_up': view_up.tolist(),
            })
            print('[saved] View {}/{}: {} | direction {}'.format(
                i, args.n_views, path.resolve(), np.round(direction, 4)),
                flush=True)
        finally:
            plotter.close()

    manifest_path = args.output.with_suffix('.views.json')
    manifest = {
        'n_views': args.n_views,
        'random_seed': args.seed,
        'transparent_background': True,
        'projection': 'orthographic',
        'rendering_mode': 'fresh_scene_per_view',
        'image_resolution': [args.resolution, args.height or args.resolution],
        'views': metadata,
    }
    with open(manifest_path, 'w', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2)
    print('[saved] Camera manifest: {}'.format(manifest_path.resolve()))


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Render N random transparent views of a convex JSON polyhedron and its proper rotational axes.',
    )
    parser.add_argument('shape_json', type=Path,
                        help='Shape JSON with a 3D polyhedron vertex array')
    parser.add_argument('--output', type=Path, default=Path('polyhedron_symmetry.png'),
                        help='Output name prefix: name.png generates name_view_001.png, etc.')
    parser.add_argument('--report', type=Path, default=None,
                        help='Symmetry JSON report path (default: output stem + .axes.json)')
    parser.add_argument('--check-only', action='store_true',
                        help='Only detect symmetry, without rendering or prompting for views')
    parser.add_argument('--n-views', type=int, default=None,
                        help='Number N of random screenshots; prompted if omitted at a terminal')
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional seed for reproducible random camera directions')
    parser.add_argument('--resolution', type=int, default=2400,
                        help='Image width in pixels')
    parser.add_argument('--height', type=int, default=None,
                        help='Image height in pixels; default = width')
    parser.add_argument('--face-color', default='#70A6D0')
    parser.add_argument('--edge-color', default='#EDEFF5')
    parser.add_argument('--opacity', type=float, default=1.0,
                        help='Polyhedron face opacity from 0 to 1')
    parser.add_argument('--edge-width', type=float, default=1.8)
    parser.add_argument('--axis-color', default='#D02020',
                        help='Uniform color for ALL symmetry axes')
    parser.add_argument('--axis-length', type=float, default=1.45,
                        help='Axis HALF-length relative to maximum vertex radius')
    parser.add_argument('--axis-thickness', type=float, default=0.014,
                        help='Axis cylinder radius relative to maximum vertex radius')
    parser.add_argument('--axis-opacity', type=float, default=1.0)
    parser.add_argument('--symmetry-tol', type=float, default=1e-6,
                        help='Vertex matching tolerance relative to body radius')
    parser.add_argument('--labels', action='store_true',
                        help='Label symmetry axes (off by default)')
    parser.add_argument('--show-legend', action='store_true',
                        help='Include a legend (off by default)')
    parser.add_argument('--show-vertices', action='store_true')
    parser.add_argument('--show-xyz', action='store_true')
    parser.add_argument('--title', default='')
    parser.add_argument('--interactive', action='store_true',
                        help='Preview scene interactively before N random screenshots')
    # Compatibility: old --transparent flag is accepted, but is redundant.
    parser.add_argument('--transparent', action='store_true', help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_arguments()
    for name, value in [('opacity', args.opacity),
                        ('axis_opacity', args.axis_opacity)]:
        if not 0 <= value <= 1:
            raise ValueError('--{} must be between 0 and 1'.format(name.replace('_', '-')))
    if args.resolution < 256 or (args.height is not None and args.height < 256):
        raise ValueError('Image dimensions must be at least 256 pixels.')
    if args.symmetry_tol <= 0 or args.axis_length <= 0 or args.axis_thickness <= 0:
        raise ValueError('Symmetry tolerance, axis length and thickness must be positive.')
    if args.output.suffix.lower() != '.png':
        raise ValueError('--output must be a .png filename, e.g. figures/my_shape.png')

    if not args.check_only:
        if args.n_views is None:
            if not sys.stdin.isatty():
                raise ValueError('Supply --n-views N when running without a terminal.')
            response = input('How many random viewing angles (N)? ').strip()
            try:
                args.n_views = int(response)
            except ValueError as exc:
                raise ValueError('Number of views N must be a positive integer.') from exc
        if args.n_views <= 0:
            raise ValueError('--n-views must be a positive integer.')

    metadata, vertices, center, hull = read_vertices(args.shape_json)
    faces, edges, n_faces = polygon_faces_from_hull(vertices, hull)
    operations = find_proper_rotations(vertices, center, args.symmetry_tol)
    axes = symmetry_axes(operations)

    report_path = args.report or args.output.with_suffix('.axes.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = save_report(report_path, metadata, vertices, center, operations, axes)

    print('Shape: {}'.format(report['name']))
    print('Geometry: {} vertices, {} faces, {} physical edges'.format(
        len(vertices), n_faces, len(edges)))
    print('Proper rotation group: {} operations (including identity)'.format(len(operations)))
    if axes:
        for order, count in sorted(((int(k[1:]), v) for k, v in
                                    report['rotation_axis_counts'].items()), reverse=True):
            print('  C{}: {} distinct undirected axis/axes'.format(order, count))
        for i, item in enumerate(axes, 1):
            print('  Axis {:>2}: C{}, direction {}'.format(i, item['order'],
                                                          np.round(item['direction'], 7)))
    else:
        print('No nontrivial proper rotational symmetry axes detected (C1).')
    print('[saved] Symmetry report: {}'.format(report_path.resolve()))

    if not args.check_only:
        render_images(vertices, center, faces, edges, axes, args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as error:
        print('[error] {}'.format(error), file=sys.stderr)
        sys.exit(1)
