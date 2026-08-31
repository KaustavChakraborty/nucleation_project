#!/usr/bin/env python3

import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import gsd.hoomd


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Plot all vertices of a selected HOOMD HPMC polyhedron "
            "over the last N frames of a GSD trajectory. "
            "Colors are based on VERTEX INDEX (not frame)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "gsd_file",
        type=str,
        help="HOOMD GSD trajectory file"
    )

    parser.add_argument(
        "shape_json",
        type=str,
        help=(
            "JSON shape file. Used if polyhedron shape information "
            "cannot be obtained from the GSD file."
        )
    )

    parser.add_argument(
        "n_frames",
        type=int,
        help="Number of consecutive frames to use from the END"
    )

    parser.add_argument(
        "particle",
        type=int,
        help="Particle tag/index to analyze"
    )

    parser.add_argument(
        "--center",
        action="store_true",
        help=(
            "Place the selected particle center at the origin in "
            "every frame. This removes translational motion and "
            "shows only rotational changes."
        )
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=30.0,
        help="Scatter marker size"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the figure"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output figure filename"
    )

    return parser.parse_args()


# ============================================================
# ASK USER FOR FRAME EXCLUSION
# ============================================================

def ask_for_frame_exclusion(frame_indices):
    """
    Ask user if they want to exclude any frames.
    
    Parameters
    ----------
    frame_indices : list or ndarray
        List of frame indices to process
    
    Returns
    -------
    filtered_indices : ndarray
        Frame indices after excluding user-specified frames
    """
    
    print("\n" + "=" * 70)
    print("FRAME SELECTION")
    print("=" * 70)
    
    print(f"\nTotal frames available: {len(frame_indices)}")
    print(f"Frame indices: {list(frame_indices)}")
    
    while True:
        response = input(
            "\nDo you want to process ALL frames? (yes/no): "
        ).strip().lower()
        
        if response in ['yes', 'y']:
            print("\nProcessing all frames.")
            return np.asarray(frame_indices)
        
        elif response in ['no', 'n']:
            
            while True:
                try:
                    exclude_input = input(
                        "\nEnter frames to exclude (comma-separated, e.g., 10,11,12): "
                    ).strip()
                    
                    if not exclude_input:
                        print("No frames excluded.")
                        return np.asarray(frame_indices)
                    
                    # Parse the input
                    excluded_frames = set()
                    for item in exclude_input.split(','):
                        item = item.strip()
                        if '-' in item:
                            # Handle range like "10-15"
                            start, end = item.split('-')
                            excluded_frames.update(
                                range(int(start.strip()), 
                                      int(end.strip()) + 1)
                            )
                        else:
                            excluded_frames.add(int(item))
                    
                    # Filter out excluded frames
                    filtered_indices = np.asarray(
                        [f for f in frame_indices 
                         if f not in excluded_frames]
                    )
                    
                    if len(filtered_indices) == 0:
                        print(
                            "\nError: All frames were excluded! "
                            "Please try again."
                        )
                        continue
                    
                    print(
                        f"\nExcluded frames: {sorted(excluded_frames)}"
                    )
                    print(
                        f"Remaining frames: {len(filtered_indices)} "
                        f"(from {len(frame_indices)} total)"
                    )
                    print(f"Frame indices to process: {list(filtered_indices)}")
                    
                    return filtered_indices
                
                except ValueError:
                    print(
                        "Invalid input. Please enter frame numbers "
                        "separated by commas (e.g., 10,11,12 or 10-12)."
                    )
                    continue
        
        else:
            print("Invalid response. Please enter 'yes' or 'no'.")
            continue


# ============================================================
# READ SHAPE FROM JSON
# ============================================================

def read_shape_from_json(json_file):
    """
    Read polyhedron vertices from JSON.

    Supports:
        "8_vertices"
        "vertices"
    """

    if not os.path.isfile(json_file):

        raise FileNotFoundError(
            "\nShape JSON file not found:\n"
            f"    {json_file}\n"
        )

    with open(json_file, "r") as f:
        data = json.load(f)

    if "8_vertices" in data:

        vertices = np.asarray(
            data["8_vertices"],
            dtype=float
        )

    elif "vertices" in data:

        vertices = np.asarray(
            data["vertices"],
            dtype=float
        )

    else:

        raise KeyError(
            "\nCould not find polyhedron vertices in JSON.\n"
            'Expected key "8_vertices" or "vertices".\n'
            f"Available keys:\n{list(data.keys())}"
        )

    if vertices.ndim != 2 or vertices.shape[1] != 3:

        raise ValueError(
            "\nVertices must have dimensions (N_vertices, 3).\n"
            f"Found shape: {vertices.shape}"
        )

    return vertices


# ============================================================
# PARSE POSSIBLE SHAPE OBJECT FROM GSD
# ============================================================

def parse_shape_object(shape):
    """
    Try to extract a vertex array from different representations
    of an HPMC shape.
    """

    if shape is None:
        return None

    if isinstance(shape, bytes):

        try:
            shape = shape.decode("utf-8")

        except Exception:
            return None

    if isinstance(shape, str):

        try:
            shape = json.loads(shape)

        except Exception:
            return None

    if isinstance(shape, dict):

        if "vertices" in shape:

            try:

                vertices = np.asarray(
                    shape["vertices"],
                    dtype=float
                )

                if (
                    vertices.ndim == 2
                    and vertices.shape[1] == 3
                ):
                    return vertices

            except Exception:
                pass

        if "8_vertices" in shape:

            try:

                vertices = np.asarray(
                    shape["8_vertices"],
                    dtype=float
                )

                if (
                    vertices.ndim == 2
                    and vertices.shape[1] == 3
                ):
                    return vertices

            except Exception:
                pass

    return None


# ============================================================
# GET PARTICLE TYPE ID
# ============================================================

def get_particle_typeid(frame, particle_index):

    if frame.particles.typeid is None:

        return 0

    return int(
        frame.particles.typeid[particle_index]
    )


# ============================================================
# TRY TO READ SHAPE FROM GSD
# ============================================================

def get_shape_from_gsd(frame, particle_index):
    """
    Try several possible locations for HPMC shape information.

    Returns
    -------
    vertices : ndarray or None
    """

    particle_typeid = get_particle_typeid(
        frame,
        particle_index
    )

    try:

        if hasattr(frame.particles, "type_shapes"):

            type_shapes = frame.particles.type_shapes

            if (
                type_shapes is not None
                and len(type_shapes) > particle_typeid
            ):

                vertices = parse_shape_object(
                    type_shapes[particle_typeid]
                )

                if vertices is not None:

                    print(
                        "\nShape information found in GSD:"
                    )

                    print(
                        "    particles.type_shapes"
                    )

                    return vertices

    except Exception:
        pass

    try:

        if hasattr(frame, "log") and frame.log is not None:

            for key in frame.log.keys():

                key_lower = key.lower()

                if (
                    "shape" not in key_lower
                    and "vert" not in key_lower
                ):
                    continue

                obj = frame.log[key]

                vertices = parse_shape_object(obj)

                if vertices is not None:

                    print(
                        "\nShape information found in GSD log:"
                    )

                    print(
                        f"    {key}"
                    )

                    return vertices

                try:

                    if len(obj) > particle_typeid:

                        vertices = parse_shape_object(
                            obj[particle_typeid]
                        )

                        if vertices is not None:

                            print(
                                "\nShape information found "
                                "in GSD log:"
                            )

                            print(
                                f"    {key}"
                            )

                            return vertices

                except Exception:
                    pass

    except Exception:
        pass

    return None


# ============================================================
# FIND PARTICLE INDEX
# ============================================================

def find_particle_index(frame, requested_particle):
    """
    Find the array index of a particle, given either its tag
    (if available) or its array index.

    Returns
    -------
    particle_index : int
    frame_has_tags : bool
    """

    frame_has_tags = False
    
    try:
        if (
            hasattr(frame.particles, 'tag')
            and frame.particles.tag is not None
            and len(frame.particles.tag) > 0
        ):
            frame_has_tags = True
            tags = np.asarray(frame.particles.tag)

            if requested_particle in tags:

                particle_index = int(
                    np.where(tags == requested_particle)[0][0]
                )

                return particle_index, frame_has_tags
    
    except (AttributeError, TypeError):
        frame_has_tags = False

    particle_index = requested_particle

    return particle_index, frame_has_tags


# ============================================================
# TRANSFORM VERTICES
# ============================================================

def transform_vertices(
        body_vertices,
        position,
        orientation,
        center_on_particle=False
):
    """
    Rotate and translate the body vertices to match the
    particle's position and orientation in the frame.
    """

    from scipy.spatial.transform import Rotation

    quat = np.asarray(orientation)

    rotation = Rotation.from_quat([
        quat[1],  # q_x
        quat[2],  # q_y
        quat[3],  # q_z
        quat[0]   # q_w
    ])

    rotated = rotation.apply(body_vertices)

    if center_on_particle:

        transformed = rotated

    else:

        transformed = rotated + np.asarray(position)

    return transformed


# ============================================================
# SET AXES EQUAL
# ============================================================

def set_axes_equal(ax, vertices):
    """
    Make axes have equal aspect ratio by finding the bounding
    box and centering the view.
    """

    max_range = np.array([
        vertices[:, 0].max() - vertices[:, 0].min(),
        vertices[:, 1].max() - vertices[:, 1].min(),
        vertices[:, 2].max() - vertices[:, 2].min()
    ]).max() / 2.0

    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    GSD_FILE = args.gsd_file
    SHAPE_JSON = args.shape_json
    N_FRAMES = args.n_frames
    REQUESTED_PARTICLE = args.particle
    CENTER_ON_PARTICLE = args.center
    POINT_SIZE = args.point_size
    SAVE_FIGURE = not args.no_save

    # ========================================================
    # OPEN GSD FILE
    # ========================================================

    print("\n" + "=" * 70)
    print("OPENING GSD FILE")
    print("=" * 70)

    try:

        traj = gsd.hoomd.open(
            GSD_FILE,
            mode="r"
        )

    except Exception as e:

        print(f"\nError opening GSD file:\n    {e}")
        sys.exit(1)

    # ========================================================
    # DETERMINE FRAME INDICES
    # ========================================================

    total_frames = len(traj)

    print(f"\nGSD file: {GSD_FILE}")
    print(f"Total frames in trajectory: {total_frames}")

    start_frame = max(0, total_frames - N_FRAMES)
    end_frame = total_frames

    frame_indices = np.arange(start_frame, end_frame)

    number_to_use = len(frame_indices)

    print(
        f"Frame range: {start_frame} to {end_frame - 1} "
        f"({number_to_use} frames)"
    )

    # ========================================================
    # ASK USER FOR FRAME EXCLUSION
    # ========================================================

    frame_indices = ask_for_frame_exclusion(frame_indices)
    number_to_use = len(frame_indices)

    # ========================================================
    # GET FIRST FRAME FOR SHAPE DETECTION
    # ========================================================

    first_selected_frame = traj[int(frame_indices[0])]

    # ========================================================
    # GET PARTICLE INDEX
    # ========================================================

    print("\n" + "=" * 70)
    print("PARTICLE INFORMATION")
    print("=" * 70)

    particle_index, frame_has_tags = find_particle_index(
        first_selected_frame,
        REQUESTED_PARTICLE
    )

    if frame_has_tags:

        print(
            f"Particle tag          : "
            f"{REQUESTED_PARTICLE}"
        )

        print(
            f"Current array index  : "
            f"{particle_index}"
        )

    else:

        print(
            f"Particle index       : "
            f"{REQUESTED_PARTICLE}"
        )

        print(
            "Explicit particle tags were not available; "
            "using array index."
        )

    if N_FRAMES > total_frames:

        print(
            "\nWARNING:"
        )

        print(
            f"You requested {N_FRAMES} frames, "
            f"but only {total_frames} exist."
        )

        print(
            f"Using all {total_frames} frames."
        )

    # --------------------------------------------------------
    # Determine shape
    # --------------------------------------------------------

    body_vertices = get_shape_from_gsd(
        first_selected_frame,
        particle_index
    )

    if body_vertices is None:

        print(
            "\nNo usable polyhedron vertex information "
            "was found in the GSD."
        )

        print(
            "Reading shape from JSON:"
        )

        print(
            f"    {SHAPE_JSON}"
        )

        body_vertices = read_shape_from_json(
            SHAPE_JSON
        )

    print(
        f"\nNumber of polyhedron vertices = "
        f"{len(body_vertices)}"
    )

    # --------------------------------------------------------
    # Read requested frames
    # --------------------------------------------------------

    all_vertices = []
    all_positions = []
    actual_frame_numbers = []

    print("\n")
    print("=" * 70)
    print("READING FRAMES")
    print("=" * 70)

    for frame_number in frame_indices:

        frame = traj[int(frame_number)]

        particle_index, frame_has_tags = find_particle_index(
            frame,
            REQUESTED_PARTICLE
        )

        position = np.asarray(
            frame.particles.position[
                particle_index
            ],
            dtype=float
        )

        orientation = np.asarray(
            frame.particles.orientation[
                particle_index
            ],
            dtype=float
        )

        vertices = transform_vertices(
            body_vertices,
            position,
            orientation,
            center_on_particle=CENTER_ON_PARTICLE
        )

        all_vertices.append(
            vertices
        )

        all_positions.append(
            position
        )

        actual_frame_numbers.append(
            frame_number
        )

        print(
            f"Frame {frame_number:7d} | "
            f"index {particle_index:6d} | "
            f"r = "
            f"({position[0]:10.5f}, "
            f"{position[1]:10.5f}, "
            f"{position[2]:10.5f}) | "
            f"q = "
            f"({orientation[0]:9.5f}, "
            f"{orientation[1]:9.5f}, "
            f"{orientation[2]:9.5f}, "
            f"{orientation[3]:9.5f})"
        )

    all_vertices = np.asarray(
        all_vertices
    )

    all_positions = np.asarray(
        all_positions
    )

    actual_frame_numbers = np.asarray(
        actual_frame_numbers
    )

    # ========================================================
    # PLOT - COLOR BY VERTEX INDEX
    # ========================================================

    fig = plt.figure(
        figsize=(9, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # --------------------------------------------------------
    # Create colormap based on number of vertices
    # --------------------------------------------------------

    num_vertices = body_vertices.shape[0]
    cmap = plt.cm.tab20 if num_vertices <= 20 else plt.cm.hsv

    # --------------------------------------------------------
    # Plot each frame's vertices, colored by vertex index
    # --------------------------------------------------------

    for i, frame_number in enumerate(
        actual_frame_numbers
    ):

        vertices = all_vertices[i]

        # Plot each vertex with its own color based on vertex index
        for vertex_idx in range(num_vertices):

            vertex_color = cmap(vertex_idx / num_vertices)

            ax.scatter(
                vertices[vertex_idx, 0],
                vertices[vertex_idx, 1],
                vertices[vertex_idx, 2],
                s=POINT_SIZE,
                color=vertex_color,
                alpha=0.80
            )

    # --------------------------------------------------------
    # Create custom colorbar for vertex indices
    # --------------------------------------------------------

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=0, vmax=num_vertices-1)
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        pad=0.10,
        shrink=0.75
    )

    cbar.set_label(
        "Vertex Index",
        fontsize=12
    )

    # --------------------------------------------------------
    # Particle trajectory
    # --------------------------------------------------------

    if not CENTER_ON_PARTICLE:

        ax.plot(
            all_positions[:, 0],
            all_positions[:, 1],
            all_positions[:, 2],
            linestyle="--",
            linewidth=1.0,
            alpha=0.65,
            label="Particle center"
        )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "x",
        fontsize=12,
        labelpad=8
    )

    ax.set_ylabel(
        "y",
        fontsize=12,
        labelpad=8
    )

    ax.set_zlabel(
        "z",
        fontsize=12,
        labelpad=8
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    if CENTER_ON_PARTICLE:

        ax.set_title(
            f"Particle {REQUESTED_PARTICLE}: "
            f"last {number_to_use} frames\n"
            "Color = Vertex Index | Particle center fixed at origin",
            fontsize=13,
            pad=15
        )

    else:

        ax.set_title(
            f"Particle {REQUESTED_PARTICLE}: "
            f"last {number_to_use} frames\n"
            "Color = Vertex Index",
            fontsize=13,
            pad=15
        )

    # --------------------------------------------------------
    # Equal scaling
    # --------------------------------------------------------

    flattened_vertices = all_vertices.reshape(
        -1,
        3
    )

    set_axes_equal(
        ax,
        flattened_vertices
    )

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    if args.output is None:

        if CENTER_ON_PARTICLE:

            output_file = (
                f"particle_{REQUESTED_PARTICLE}_"
                f"last_{number_to_use}_frames_"
                f"vertex_color_centered.png"
            )

        else:

            output_file = (
                f"particle_{REQUESTED_PARTICLE}_"
                f"last_{number_to_use}_frames_"
                f"vertex_color.png"
            )

    else:

        output_file = args.output

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if SAVE_FIGURE:

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        print("\n")
        print("=" * 70)
        print("OUTPUT")
        print("=" * 70)

        print(
            f"Figure saved as:\n"
            f"    {output_file}"
        )

    plt.show()


# ============================================================
# EXECUTE
# ============================================================

if __name__ == "__main__":
    main()
