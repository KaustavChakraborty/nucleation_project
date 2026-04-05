"""
=============================================================================
HOOMD-blue v4  |  Convex-Polyhedron Lattice Generator
=============================================================================
Input  : JSON file with lattice type (SC / FCC / BCC), packing fraction phi,
         polyhedron shape JSON, linear scale, orientation mode, and requested
         particle count.
Output : A GSD initial-condition file ready for HPMC simulations.

The script derives the correct (cubic) unit-cell length 'a' from phi and the
polyhedron particle volume, resolves the nearest replicable supercell, writes
the GSD file, assigns ordered or random orientations, and verifies the final
configuration using HOOMD-blue v4 HPMC ConvexPolyhedron overlap checking.

Derivation of 'a' from phi:
  phi = N_cell * V_particle / a^3
  a = ( N_cell * V_particle / phi )^(1/3)

where N_cell = 1 (SC), 2 (BCC), 4 (FCC).

JSON schema:
{
    "lattice_type"       : "FCC",       // "SC", "FCC", or "BCC"
    "phi"                : 0.20,        // target packing fraction (phi > 0)
    "shape_json"         : "shape.json",// convex-polyhedron JSON file
    "scale"              : 1.0,         // linear scaling of input vertices
    "n_particles_req"    : 2000,        // *requested* particle count
    "orientation_mode"   : "order",    // "order" or "disorder"
    "ordered_orientation": [1,0,0,0],  // required if orientation_mode=order
    "rng_seed"           : 1,           // optional; used for disorder mode
    "output_gsd"         : "lattice_phi.gsd"
}

Author : Kaustav Chakraborty
Tested : HOOMD-blue 4.x, GSD 3.x, NumPy 1.x / 2.x
=============================================================================
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------

# Used to read the input configuration file in JSON format.
import json
# Used mainly for sys.exit(...), i.e. to stop the program with an error message.
import sys
# Used for mathematical constants/functions like pi, sqrt, ceil.
import math
# Used for Cartesian-product style looping, e.g. iterating over all (ix, iy, iz).
import itertools
# Used to parse command-line arguments, here the input JSON filename.
import argparse
from pathlib import Path
# NumPy is used for array creation, coordinate storage, vectorized math, etc
import numpy as np
# GSD is the file format used by HOOMD-blue for storing simulation snapshots.
# gsd.hoomd.Frame() is the object that represents one simulation frame/state.
try:
    import gsd.hoomd
except ImportError as exc:
    raise SystemExit("[IMPORT ERROR] gsd.hoomd is not installed.") from exc

try:
    import hoomd
except ImportError as exc:
    raise SystemExit("[IMPORT ERROR] HOOMD-blue is not installed.") from exc


# ---------------------------------------------------------------------------
# Lattice definitions
# ---------------------------------------------------------------------------

LATTICE_BASIS = {
    # Simple cubic (SC):
    # One lattice point per unit cell, located at the origin in fractional coordinates.
    "SC":  [(0.0, 0.0, 0.0)],
    # Body-centered cubic (BCC):
    # One atom at the cube corner/origin, one at the body center.
    # Fractional coordinates are with respect to the cubic cell length a.
    "BCC": [(0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5)],
    # Face-centered cubic (FCC):
    # One atom at origin, and one at the center of each of three distinct faces
    # represented in the primitive cubic basis.
    "FCC": [(0.0, 0.0, 0.0),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5)],
}

# Number of particles generated per conventional cubic unit cell.
PARTICLES_PER_UNIT_CELL = {"SC": 1, "FCC": 4, "BCC": 2}

# Nearest-neighbour distance as fraction of 'a' for each lattice
# NN_FRAC: nearest-neighbour distance = NN_FRAC[lattice] * a
NN_FRAC = {
    "SC":  1.0,
    "BCC": math.sqrt(3.0) / 2.0,   # √3/2 ≈ 0.866
    "FCC": 1.0 / math.sqrt(2.0),   # 1/√2 ≈ 0.707
}

class InputValidationError(Exception):
    pass

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def a_from_phi(phi: float, particle_volume: float, n_per_cell: int) -> float:
    """
    Derive the cubic unit-cell length from packing fraction and particle volume.

    For a periodic lattice with n_per_cell particles per conventional cubic unit cell:

        phi = n_per_cell * V_particle / a^3

    so that

        a = [ n_per_cell * V_particle / phi ]^(1/3)
    """
    if particle_volume <= 0.0:
        raise InputValidationError("Particle volume must be > 0.")

    a_cubed = n_per_cell * particle_volume / phi
    return a_cubed ** (1.0 / 3.0)


def nearest_replication(n_req: int, n_per_cell: int):
    """
    Return (nx, ny, nz) and the actual total particle count.
    Chooses the most balanced (cubic-ish) supercell with N ≥ n_req.
    """
    # We first estimate how many unit cells are needed along one axis if the
    # supercell were roughly cubic:
    #
    # total particles = nx * ny * nz * n_per_cell
    #
    # If nx = ny = nz = m, then:
    # total ~ m^3 * n_per_cell
    # m ~ (n_req / n_per_cell)^(1/3)
    #
    # ceil(...) ensures we do not underestimate.
    m = max(1, math.ceil((n_req / n_per_cell) ** (1.0 / 3.0)))
    # best will store the best triplet (nx, ny, nz) found so far.
    best = None
    # best_n will store the corresponding total particle count for that triplet.
    best_n = None

    # We now search in a small window around the guessed size m.
    # dm expands the search range gradually if no acceptable solution is found.
    for dm in range(0, 5):
        # itertools.product(..., repeat=3) generates all triples (nx, ny, nz)
        # in the given range.
        #
        # range(max(1, m - 1), m + dm + 2) means:
        #   start around m-1, but not below 1,
        #   go up to a little beyond m, gradually increasing with dm.
        for nx, ny, nz in itertools.product(range(max(1, m - 1), m + dm + 2),
                                             repeat=3):
            # Total particles for that supercell:
            total = nx * ny * nz * n_per_cell
            # We only consider supercells that give at least the requested number.
            if total >= n_req:
                # Update the best candidate if:
                # 1. we do not yet have one
                # 2. or this one has fewer total particles (closer to request)
                # 3. or same total particles but smaller largest dimension,
                #    which favors a more cube-like cell
                if best_n is None or total < best_n or \
                   (total == best_n and max(nx, ny, nz) < max(*best)):
                    best = (nx, ny, nz)
                    best_n = total
        # If we found at least one valid option at this dm level, stop expanding.
        if best is not None:
            break
    # Returns:
    #   best   = (nx, ny, nz)
    #   best_n = nx * ny * nz * n_per_cell
    return best, best_n


def build_positions(lattice_type: str, a: float,
                    nx: int, ny: int, nz: int):
    """
    Generate all particle positions for a cubic supercell
    (a == b == c enforced by phi constraint for cubic lattice).
    Returns shape (N, 3) array, centred at origin.
    """
    # Get the basis vectors for the chosen lattice.
    # Example:
    # FCC has 4 fractional basis positions per unit cell.
    basis = LATTICE_BASIS[lattice_type]
    # Physical box lengths along x, y, z:
    # each dimension is number of unit cells times cubic cell length a.
    Lx = nx * a
    Ly = ny * a
    Lz = nz * a

    # We will append every particle position here as [x, y, z].
    positions = []
    # Loop over all replicated unit cells.
    # ix, iy, iz are the integer cell indices.
    for ix, iy, iz in itertools.product(range(nx), range(ny), range(nz)):
        # For each unit cell, place all basis atoms.
        for bx, by, bz in basis:
            # Fractional-to-Cartesian conversion:
            # (ix + bx) gives the position in units of the lattice constant a.
            # Multiplying by a converts to real length.
            #
            # Then subtract half the box length so the full structure is centered
            # around the origin instead of starting at (0,0,0).
            x = (ix + bx) * a - Lx / 2.0
            y = (iy + by) * a - Ly / 2.0
            z = (iz + bz) * a - Lz / 2.0

            positions.append([x, y, z])

    # Convert list of coordinate triplets into a NumPy array.
    # dtype=float64 gives high precision for geometry checks and storage.
    return np.array(positions, dtype=np.float64)


def verify_phi(n: int, particle_volume: float, Lx: float, Ly: float, Lz: float):
    # Packing fraction = total particle volume / simulation box volume.
    return n * particle_volume / (Lx * Ly * Lz)


def load_polyhedron_shape(shape_json_path: str):
    """
    Load the convex-polyhedron JSON file.

    Expected keys in the supplied JSON:
      - "4_volume"   : reference particle volume
      - "8_vertices" : list of 3D vertices in the particle/body frame
    """
    path = Path(shape_json_path)

    if not path.exists():
        raise FileNotFoundError(f"Shape JSON file not found: {path}")
    if not path.is_file():
        raise InputValidationError(f"Shape JSON path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            shape_cfg = json.load(fh)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"Invalid JSON syntax in shape file '{path}': "
            f"line {exc.lineno}, column {exc.colno}."
        ) from exc

    if "4_volume" not in shape_cfg:
        raise InputValidationError(
            f"Shape JSON '{path}' does not contain key '4_volume'."
        )
    if "8_vertices" not in shape_cfg:
        raise InputValidationError(
            f"Shape JSON '{path}' does not contain key '8_vertices'."
        )

    try:
        base_volume = float(shape_cfg["4_volume"])
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            f"Shape JSON '{path}' has non-numeric '4_volume'."
        ) from exc

    if not math.isfinite(base_volume) or base_volume <= 0.0:
        raise InputValidationError(
            f"Shape JSON '{path}' has invalid volume {base_volume}."
        )

    vertices = np.asarray(shape_cfg["8_vertices"], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise InputValidationError(
            f"Shape JSON '{path}' vertices must be an Nx3 array."
        )
    if len(vertices) < 4:
        raise InputValidationError(
            f"Shape JSON '{path}' must contain at least 4 vertices."
        )

    shape_name = str(shape_cfg.get("1_Name", path.stem))
    shape_short_name = str(shape_cfg.get("2_ShortName", shape_name))

    return shape_name, shape_short_name, base_volume, vertices


def scale_vertices(base_vertices: np.ndarray, scale: float) -> np.ndarray:
    """
    Uniformly scale the reference vertices by a linear factor 'scale'.
    """
    if not math.isfinite(scale) or scale <= 0.0:
        raise InputValidationError("scale must be a finite positive number.")

    return np.asarray(base_vertices, dtype=np.float64) * float(scale)


def circumsphere_radius(vertices: np.ndarray) -> float:
    """
    Radius of the smallest sphere centered at the particle origin that encloses
    all provided vertices. This is only a diagnostic geometric quantity here.
    """
    return float(np.max(np.linalg.norm(vertices, axis=1)))


def validate_unit_quaternion(q, field_name="ordered_orientation") -> np.ndarray:
    """
    Accept a 4-vector quaternion, ensure it is finite and non-zero,
    and normalize it to unit length.
    HOOMD uses [w, x, y, z] ordering.
    """
    arr = np.asarray(q, dtype=np.float64)

    if arr.shape != (4,):
        raise InputValidationError(
            f"{field_name} must be a list of 4 numbers in [w, x, y, z] order."
        )

    if not np.all(np.isfinite(arr)):
        raise InputValidationError(f"{field_name} must contain only finite numbers.")

    norm = np.linalg.norm(arr)
    if norm <= 0.0:
        raise InputValidationError(f"{field_name} must not be the zero quaternion.")

    return arr / norm


def normalize_orientation_mode(value: str) -> str:
    """
    Accept a few aliases, but normalize to either:
      - 'order'
      - 'disorder'
    """
    mode = str(value).strip().lower()

    if mode in {"order", "ordered"}:
        return "order"
    if mode in {"disorder", "disordered", "random"}:
        return "disorder"

    raise InputValidationError(
        "orientation_mode must be one of: "
        "'order', 'ordered', 'disorder', 'disordered', or 'random'."
    )


def random_unit_quaternions(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n random unit quaternions uniformly from SO(3).

    Returned order is [w, x, y, z], matching HOOMD snapshot orientation storage.
    """
    u1 = rng.random(n)
    u2 = rng.random(n)
    u3 = rng.random(n)

    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * math.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * math.pi * u2)
    qz = np.sqrt(u1)       * np.sin(2.0 * math.pi * u3)
    qw = np.sqrt(u1)       * np.cos(2.0 * math.pi * u3)

    return np.column_stack((qw, qx, qy, qz)).astype(np.float64)


def build_orientations(n_particles: int,
                       orientation_mode: str,
                       ordered_orientation,
                       rng_seed: int) -> np.ndarray:
    """
    Build the particle-orientation array for the GSD frame.

    order:
        every particle gets the same user-provided unit quaternion.
    disorder:
        each particle gets an independently sampled random unit quaternion.
    """
    if orientation_mode == "order":
        q = validate_unit_quaternion(ordered_orientation, "ordered_orientation")
        return np.tile(q, (n_particles, 1)).astype(np.float64)

    if orientation_mode == "disorder":
        rng = np.random.default_rng(rng_seed)
        return random_unit_quaternions(n_particles, rng)

    raise InputValidationError(
        f"Internal error: unsupported orientation_mode '{orientation_mode}'."
    )


def minimum_center_distance(positions, Lx, Ly, Lz):
    """
    Diagnostic only:
    return the minimum center-to-center distance under PBC.
    This is NOT an exact polyhedron-overlap test; HOOMD HPMC is the exact test.
    """
    box = np.array([Lx, Ly, Lz], dtype=np.float64)
    n = len(positions)

    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dr = positions[j] - positions[i]
            dr -= box * np.round(dr / box)
            d = np.linalg.norm(dr)
            if d < min_dist:
                min_dist = d

    return float(min_dist)


def parse_config(input_json_path: str):
    path = Path(input_json_path)

    if not path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {path}")
    if not path.is_file():
        raise InputValidationError(f"Input path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except json.JSONDecodeError as exc:
        raise InputValidationError(
            f"Invalid JSON syntax in '{path}': line {exc.lineno}, column {exc.colno}."
        ) from exc

    required = [
        "lattice_type",
        "phi",
        "scale",
        "n_particles_req",
        "shape_json",
        "orientation_mode",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise InputValidationError(
            f"Missing required JSON keys: {', '.join(missing)}"
        )

    lattice_type = str(cfg["lattice_type"]).upper().strip()
    if lattice_type not in LATTICE_BASIS:
        raise InputValidationError(
            f"lattice_type must be SC, FCC, or BCC. Got '{lattice_type}'."
        )

    try:
        phi = float(cfg["phi"])
        scale = float(cfg["scale"])
        n_req = int(cfg["n_particles_req"])
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "phi, scale, and n_particles_req must be numeric."
        ) from exc

    if not math.isfinite(phi) or phi <= 0.0:
        raise InputValidationError("Packing fraction phi must be finite and > 0.")
    if not math.isfinite(scale) or scale <= 0.0:
        raise InputValidationError("scale must be finite and > 0.")
    if n_req <= 0:
        raise InputValidationError(
            "Requested particle count must be a positive integer."
        )

    raw_shape_json = str(cfg["shape_json"]).strip()
    if not raw_shape_json:
        raise InputValidationError("shape_json must be a non-empty string.")

    shape_json_path = Path(raw_shape_json)
    if not shape_json_path.is_absolute():
        shape_json_path = (path.parent / shape_json_path).resolve()

    orientation_mode = normalize_orientation_mode(cfg["orientation_mode"])

    ordered_orientation = cfg.get("ordered_orientation", None)
    if orientation_mode == "order":
        if ordered_orientation is None:
            raise InputValidationError(
                "For orientation_mode='order', you must provide "
                "'ordered_orientation' as [w, x, y, z]."
            )
        ordered_orientation = validate_unit_quaternion(
            ordered_orientation, "ordered_orientation"
        )

    rng_seed_raw = cfg.get("rng_seed", 1)
    try:
        rng_seed = int(rng_seed_raw)
    except (TypeError, ValueError) as exc:
        raise InputValidationError("rng_seed must be an integer.") from exc

    output_gsd = cfg.get("output_gsd", None)
    if output_gsd is not None and str(output_gsd).strip():
        output_gsd = Path(str(output_gsd).strip())
        if not output_gsd.is_absolute():
            output_gsd = (path.parent / output_gsd).resolve()
    else:
        output_gsd = None

    return (
        lattice_type,
        phi,
        scale,
        n_req,
        str(shape_json_path),
        orientation_mode,
        ordered_orientation,
        rng_seed,
        output_gsd,
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    # -------- Argument parsing --------
    parser = argparse.ArgumentParser(
        description="Generate a HOOMD-blue v4 GSD lattice file for convex polyhedra "
                    "(packing-fraction driven)."
    )
    parser.add_argument("input_json", help="Path to the JSON input file.")
    args = parser.parse_args()

    (
        lattice_type,
        phi,
        scale,
        n_req,
        shape_json_path,
        orientation_mode,
        ordered_orientation,
        rng_seed,
        output_gsd_from_json,
    ) = parse_config(args.input_json)

    # ---- Load and scale the polyhedron shape --------------------------------
    shape_name, shape_short_name, base_volume, base_vertices = load_polyhedron_shape(
        shape_json_path
    )
    vertices = scale_vertices(base_vertices, scale)

    # Reference shape volume scales as scale^3 under uniform scaling.
    particle_volume = base_volume * (scale ** 3)

    # Only a diagnostic geometric size, not the exact overlap rule.
    r_circ = circumsphere_radius(vertices)

    # ---- Lattice / box construction -----------------------------------------
    n_per_cell = PARTICLES_PER_UNIT_CELL[lattice_type]

    # Volume-based packing-fraction relation for arbitrary convex polyhedron.
    a = a_from_phi(phi, particle_volume, n_per_cell)

    # Nearest-neighbour center-to-center spacing on the chosen lattice.
    nn_dist = NN_FRAC[lattice_type] * a

    # ---- Resolve supercell -------------------------------------------------
    # Finds a nearly cubic replication that gives at least n_req particles.
    (nx, ny, nz), n_actual = nearest_replication(n_req, n_per_cell)

    # This line initially assumes cubic replication, but it is immediately
    # corrected below if nx, ny, nz differ.
    Lx = Ly = Lz = nx * a   # cubic supercell (nx == ny == nz enforced by symmetry)

    # For a balanced supercell we want nx == ny == nz. If nearest_replication
    # returns a non-cubic triplet, we just use it as-is (box is still cubic
    # overall if Lx = nx*a, etc.).  Recalculate box lengths properly:
    Lx = nx * a
    Ly = ny * a
    Lz = nz * a
    # These are the actual simulation box lengths.

    # -------- Format helper (replace '.' => 'p') --------
    def fmt(x):
        return f"{x:.2f}".rstrip('0').rstrip('.').replace('.', 'p')

    # Final filename
    # Since cubic lattice  a = b = c
    def choose_output_filename(output_gsd, lattice_type, phi, n_actual, a,
                               shape_short_name, orientation_mode):
        if output_gsd is not None and str(output_gsd).strip():
            return str(output_gsd)

        phi_str = fmt(phi)
        a_str = fmt(a)
        return (
            f"convex_polyhedron_lattice_{shape_short_name}_{lattice_type}"
            f"_{orientation_mode}_pf{phi_str}_{n_actual}"
            f"_lattice_const_{a_str}_{a_str}_{a_str}.gsd"
        )

    # Automatically generated output filename.
    output_gsd = choose_output_filename(
        output_gsd_from_json, lattice_type, phi, n_actual, a,
        shape_short_name, orientation_mode
    )

    # ---- Build positions ---------------------------------------------------
    positions = build_positions(lattice_type, a, nx, ny, nz)
    # Generates all particle positions in the periodic supercell.
    n_actual_check = len(positions)
    # Number of coordinates actually produced.
    if n_actual_check != n_actual:
        raise RuntimeError(
            f"Position count mismatch: generated {n_actual_check}, expected {n_actual}."
        )
    # Internal consistency check:
    # the generated number of positions must match the expected total particle count.

    orientations = build_orientations(
        n_particles=n_actual,
        orientation_mode=orientation_mode,
        ordered_orientation=ordered_orientation,
        rng_seed=rng_seed,
    )

    # ---- Verify packing fraction ----
    # Recomputes the actual packing fraction from the final box and particle count.
    # This helps confirm that the requested phi is indeed achieved.
    phi_actual = verify_phi(n_actual, particle_volume, Lx, Ly, Lz)

    # ---- Console report ----------------------------------------------------
    sep = "=" * 72
    print(sep)
    print("  HOOMD-blue v4  |  Convex-Polyhedron Lattice Generator")
    print(sep)
    print(f"  Lattice type             : {lattice_type}")
    print(f"  Shape name               : {shape_name}")
    print(f"  Shape short name         : {shape_short_name}")
    print(f"  Shape JSON               : {shape_json_path}")
    print(f"  Input packing fraction   : {phi:.6f}")
    print(f"  Reference shape volume   : {base_volume:.6f}")
    print(f"  Linear scale             : {scale:.6f}")
    print(f"  Particle volume          : {particle_volume:.6f}")
    print(f"  Derived unit-cell length : a = {a:.6f}")
    print(f"  Nearest-neighbour dist   : {nn_dist:.6f}")
    print(f"  Circumsphere radius      : {r_circ:.6f}")
    print(f"  Particles / unit cell    : {n_per_cell}")
    print(f"  Requested particles      : {n_req}")
    print(f"  Supercell replication    : {nx} * {ny} * {nz}")
    print(f"  Actual particles (N)     : {n_actual}"
          f"  {'(adjusted)' if n_actual != n_req else '(exact match)'}")
    print(f"  Box dimensions           : Lx={Lx:.4f}, Ly={Ly:.4f}, Lz={Lz:.4f}")
    print(f"  Verified packing frac.   : phi_actual = {phi_actual:.6f}")
    print(f"  Orientation mode         : {orientation_mode}")

    if orientation_mode == "order":
        print(f"  Ordered orientation      : {ordered_orientation.tolist()}")
    else:
        print(f"  Random orientation seed  : {rng_seed}")

    # In principle, phi_actual should match phi exactly if a was derived from phi
    # and box size is built consistently from the same a.
    # This note guards against tiny floating-point effects or design mismatches.
    if abs(phi_actual - phi) > 1e-5:
        print(f"  [NOTE] Small phi discrepancy: "
              f"Delta(phi) = {abs(phi_actual - phi):.2e}")

    # Reports if the requested particle count could not be matched exactly
    # and had to be increased to the nearest constructible supercell.
    if n_req != n_actual:
        print(f"  [NOTE] Requested {n_req} particles; adjusted to {n_actual} "
              f"({nx}*{ny}*{nz} * {n_per_cell}).")
    print(sep)

    # ---- Write GSD ---------------------------------------------------------
    # Create a blank HOOMD/GSD frame object.
    frame = gsd.hoomd.Frame()
    # Total number of particles in the frame.
    frame.particles.N           = n_actual
    # Particle coordinates.
    frame.particles.position    = positions.astype(np.float32)
    # Quaternion orientations for every particle.
    frame.particles.orientation = orientations.astype(np.float32)
    # All particles are assigned type index 0.
    frame.particles.typeid      = np.zeros(n_actual, dtype=np.int32)
    # The actual type name corresponding to typeid 0 is "A".
    frame.particles.types       = ["A"]

    frame.particles.type_shapes = [
        {
            "type": "ConvexPolyhedron",
            "rounding_radius": 0.0,
            "vertices": vertices.tolist()
        }
    ]
    # HOOMD box format:
    # [Lx, Ly, Lz, xy, xz, yz]
    # Since box is orthorhombic/cubic here, all tilt factors are zero.
    frame.configuration.box     = [Lx, Ly, Lz, 0, 0, 0]
    # Simulation step counter for this initial frame.
    frame.configuration.step    = 0
    # Opens the output GSD file for writing and appends the frame as the first snapshot.
    with gsd.hoomd.open(name=output_gsd, mode="w") as traj:
        traj.append(frame)

    print(f"\n  GSD file written => {output_gsd}")
    print(sep)

    # ---- HOOMD v4 HPMC overlap check --------------------------------------
    print("\n  Running HOOMD-v4 HPMC overlap check ...", flush=True)

    hoomd_device = hoomd.device.CPU()
    hoomd_sim = hoomd.Simulation(device=hoomd_device, seed=rng_seed)

    # HPMC convex-polyhedron integrator
    mc = hoomd.hpmc.integrate.ConvexPolyhedron()

    # Define the particle shape using the scaled vertices for particle type "A"
    mc.shape["A"] = dict(vertices=vertices.tolist())

    # Attach integrator and initialize the state from the GSD we just wrote
    hoomd_sim.operations.integrator = mc
    hoomd_sim.create_state_from_gsd(filename=output_gsd)

    # Initialize / attach operations cleanly before querying overlaps
    hoomd_sim.run(0)

    hoomd_overlap_count = mc.overlaps
    print(f"  HOOMD overlap count = {hoomd_overlap_count}")

    if hoomd_overlap_count > 0:
        raise RuntimeError(
            f"HOOMD detected {hoomd_overlap_count} overlapping pair(s). "
            f"The constructed configuration is NOT overlap free. "
            f"Reduce phi and/or choose a different lattice/orientation."
        )
    else:
        print("  [OK] HOOMD detected zero overlapping pairs.")
        print("  [OK] Constructed configuration is overlap free.")

    # ---- Quick geometric sanity check -------------------------------------
    if n_actual <= 5000:
        # For manageable system sizes, do a brute-force minimum-distance diagnostic.
        print("\n  Running center-distance diagnostic ...", end=" ", flush=True)
        min_d = minimum_center_distance(positions, Lx, Ly, Lz)
        print(f"minimum center-to-center distance = {min_d:.6f}")
    else:
        print(f"\n  Skipping center-distance diagnostic (N={n_actual} > 5000).")
    # The pairwise center-distance diagnostic scales as O(N^2), so it is skipped
    # for large systems.


# ============================ ENTRY POINT ==================================
if __name__ == "__main__":
    try:
        main()
    except InputValidationError as exc:
        raise SystemExit(f"[INPUT ERROR] {exc}")
    except FileNotFoundError as exc:
        raise SystemExit(f"[FILE ERROR] {exc}")
    except OSError as exc:
        raise SystemExit(f"[I/O ERROR] {exc}")
    except RuntimeError as exc:
        raise SystemExit(f"[RUNTIME ERROR] {exc}")
    except KeyboardInterrupt:
        raise SystemExit("\n[ABORTED] Interrupted by user.")
    except Exception as exc:
        raise SystemExit(f"[UNEXPECTED ERROR] {type(exc).__name__}: {exc}")
