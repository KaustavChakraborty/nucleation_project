"""
=============================================================================
HOOMD-blue v4  |  Hard-Sphere Lattice Generator  
=============================================================================
Input  : JSON file with lattice type (SC / FCC / BCC), packing fraction phi,
         particle diameter sigma, and requested particle count.
Output : A GSD initial-condition file ready for HPMC or MD simulations.

The script derives the correct (cubic) unit-cell length 'a' from phi and sigma,
resolves the nearest replicable supercell, and writes the GSD file.

Derivation of 'a' from phi:
  phi = N_cell * V_sphere / a^3
  a = ( N_cell * (pi/6) * sigma^3 / phi )^(1/3)

where N_cell = 1 (SC), 2 (BCC), 4 (FCC).

JSON schema (see example_input_phi.json):
{
    "lattice_type"   : "FCC",       // "SC", "FCC", or "BCC"  (case-insensitive)
    "phi"            : 0.50,        // target packing fraction (0 < phi < phi_max)
    "diameter"       : 1.0,         // hard-sphere diameter sigma
    "n_particles_req": 2000,        // *requested* particle count (will be adjusted)
    "output_gsd"     : "lattice_phi.gsd"
}

Maximum close-packing limits for reference:
  SC   phi_max = 0.5236  (pi/6)
  BCC  phi_max = 0.6802  (pi * sqrt(3)/8)
  FCC  phi_max = 0.7405  (pi/(3 * sqrt(2)))

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

# Maximum (random) close-packing fractions — physical upper bounds
# These are the maximum allowed packing fractions for hard spheres arranged
# on these lattices without overlap.
PHI_MAX = {
    "SC":  math.pi / 6.0,            #  0.5236
    "BCC": math.pi * math.sqrt(3) / 8.0,  #  0.6802
    "FCC": math.pi / (3.0 * math.sqrt(2.0)),  #  0.7405
}

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

def a_from_phi(phi: float, diameter: float, n_per_cell: int) -> float:
    """
    Derive the cubic unit-cell length from packing fraction and diameter.
    phi = n_per_cell * (pi/6) * sigma^3 / a^3
    a = [ n_per_cell * (pi/6) * sigma^3 / phi ]^(1/3)
    """
    V_sphere = (math.pi / 6.0) * diameter**3
    a_cubed  = n_per_cell * V_sphere / phi
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
    # total ≈ m^3 * n_per_cell
    # m ≈ (n_req / n_per_cell)^(1/3)
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


def verify_phi(n: int, diameter: float, Lx: float, Ly: float, Lz: float):
    # Compute the radius from the diameter.
    r = diameter / 2.0
    # Volume of one particle.
    V_p = (4.0 / 3.0) * math.pi * r**3
    # Packing fraction = total particle volume / simulation box volume.
    return n * V_p / (Lx * Ly * Lz)


def overlap_check(positions, diameter, Lx, Ly, Lz):
    # Store box lengths in a NumPy array for vectorized minimum-image convention.
    box = np.array([Lx, Ly, Lz])
    # Number of particles.
    n = len(positions)
    # Count how many overlapping pairs are found.
    n_overlaps = 0
    # Track smallest pair distance seen, useful for diagnostics.
    min_dist = np.inf
    # Brute-force O(N^2) pairwise comparison.
    # This is acceptable only for moderate N, hence later code skips it for large N.
    for i in range(n):
        for j in range(i + 1, n):
            # Raw displacement between particle j and i.
            dr = positions[j] - positions[i]
            # Apply minimum-image convention for periodic boundaries:
            # subtract integer multiples of box lengths so dr becomes the shortest
            # periodic displacement.
            dr -= box * np.round(dr / box)
            # Euclidean norm gives scalar pair distance.
            d = np.linalg.norm(dr)
            # Update the minimum observed distance.
            if d < min_dist:
                min_dist = d
            # Count as overlap if distance is less than diameter,
            # allowing a tiny numerical tolerance.
            if d < diameter * (1 - 1e-6):
                n_overlaps += 1

    # Returns:
    #   n_overlaps = number of overlapping pairs
    #   min_dist   = smallest pair distance observed
    return n_overlaps, min_dist


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

    required = ["lattice_type", "phi", "diameter", "n_particles_req"]
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
        diameter = float(cfg["diameter"])
        n_req = int(cfg["n_particles_req"])
    except (TypeError, ValueError) as exc:
        raise InputValidationError(
            "phi, diameter, and n_particles_req must be numeric."
        ) from exc

    output_gsd = cfg.get("output_gsd", None)

    if not math.isfinite(phi):
        raise InputValidationError("Packing fraction phi must be finite.")
    if not math.isfinite(diameter):
        raise InputValidationError("Particle diameter must be finite.")
    if phi <= 0.0:
        raise InputValidationError("Packing fraction must be > 0.")
    if diameter <= 0.0:
        raise InputValidationError("Particle diameter must be > 0.")
    if n_req <= 0:
        raise InputValidationError("Requested particle count must be a positive integer.")

    return lattice_type, phi, diameter, n_req, output_gsd




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
        # -------- Argument parsing --------
    parser = argparse.ArgumentParser(
        description="Generate a HOOMD-blue v4 GSD lattice file "
                    "(specify packing fraction phi).")
    parser.add_argument("input_json",
                        help="Path to the JSON input file.")
    args = parser.parse_args()

    lattice_type, phi, diameter, n_req, output_gsd_from_json = parse_config(args.input_json)

    # ---- Load JSON ---------------------------------------------------------
    with open(args.input_json, "r") as fh:
        cfg = json.load(fh)

    lattice_type = cfg["lattice_type"].upper().strip()
    if lattice_type not in LATTICE_BASIS:
        sys.exit(f"[ERROR] lattice_type must be SC, FCC, or BCC. Got '{lattice_type}'.")

    # Target packing fraction requested by the user.
    phi         = float(cfg["phi"])
    # Hard-sphere diameter sigma.
    diameter    = float(cfg["diameter"])
    # Requested particle count.
    n_req       = int(cfg["n_particles_req"])
    output_gsd = None  # will be generated automatically
    # Number of particles in one conventional unit cell for the chosen lattice.
    n_per_cell  = PARTICLES_PER_UNIT_CELL[lattice_type]
    # Physical maximum packing fraction allowed for that lattice.
    phi_max     = PHI_MAX[lattice_type]

    # ---- Validate input --------------------------------------------------------
    if phi <= 0.0:
        # Packing fraction must be positive.
        sys.exit("[ERROR] Packing fraction must be > 0.")
    if phi >= phi_max:
        # Rejects impossible packing fractions that exceed or reach close-packing.
        sys.exit(f"[ERROR] phi={phi:.4f} ≥ phi_max={phi_max:.4f} for {lattice_type}. "
                 f"Particles cannot fit without overlap.")
    if phi > 0.74:
        # Extra warning for very dense configurations.
        # Note: this threshold is generic; for SC/BCC the actual phi_max is lower.
        print(f"[WARNING] phi={phi:.4f} is very high. Ensure your lattice is stable.")

    # ---- Derive unit-cell length -------------------------------------------
    # Computes the cubic lattice constant a from the requested packing fraction.
    a = a_from_phi(phi, diameter, n_per_cell)

    # Nearest neighbour distance
    # Converts the lattice constant a into nearest-neighbour spacing
    # using the known geometric factor for the lattice.
    nn_dist = NN_FRAC[lattice_type] * a

    # Ensure no overlap
    # This is a second physical consistency check:
    # if nearest-neighbour distance is smaller than sphere diameter,
    # spheres overlap immediately.
    if nn_dist < diameter * (1 - 1e-6):
        sys.exit(
            f"[ERROR] At phi={phi:.4f}, nearest-neighbour distance "
            f"({nn_dist:.4f}) < diameter ({diameter:.4f}). "
            f"This phi exceeds close-packing for {lattice_type}."
        )

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
        
    phi_str = fmt(phi)
    a_str   = fmt(a)

    # Final filename
    # Since cubic lattice  a = b = c
    def choose_output_filename(output_gsd, lattice_type, phi, a):
        if output_gsd is not None and str(output_gsd).strip():
            return str(output_gsd)

        phi_str = fmt(phi)
        a_str = fmt(a)
        return (
            f"hard_sphere_lattice_{lattice_type}"
            f"_pf{phi_str}"
            f"_lattice_const_{a_str}_{a_str}_{a_str}.gsd"
        )

    # Automatically generated output filename.
    # Example:
    # hard_sphere_lattice_FCC_pf0p5_lattice_const_1p61_1p61_1p61.gsd
    output_gsd = choose_output_filename(output_gsd_from_json, lattice_type, phi, a)


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

    # ---- Verify packing fraction ----
    # Recomputes the actual packing fraction from the final box and particle count.
    # This helps confirm that the requested phi is indeed achieved.
    phi_actual = verify_phi(n_actual, diameter, Lx, Ly, Lz)

    # ---- Console report ----------------------------------------------------
    sep = "=" * 65
    print(sep)
    print("  HOOMD-blue v4  |  Lattice Generator  |  Packing fraction based")
    print(sep)
    print(f"  Lattice type            : {lattice_type}")
    print(f"  Input packing fraction : {phi:.6f}")
    print(f"  Maximum packing fraction for {lattice_type}            : {phi_max:.6f}")
    print(f"  Particle diameter      : {diameter}")
    print(f"  Derived unit-cell length: a = {a:.6f}")
    print(f"  Nearest-neighbour dist  : {nn_dist:.6f}  (must >= sigma = {diameter})")
    print(f"  Particles / unit cell   : {n_per_cell}")
    print(f"  Requested particles     : {n_req}")
    print(f"  Supercell replication   : {nx} * {ny} * {nz}")
    print(f"  Actual particles (N)    : {n_actual}"
          f"  {'(adjusted)' if n_actual != n_req else '(exact match)'}")
    print(f"  Box dimensions          : Lx={Lx:.4f}, Ly={Ly:.4f}, Lz={Lz:.4f}")
    print(f"  Verified packing frac.  : phi_actual = {phi_actual:.6f}")
    # This block prints a run summary:
    # chosen lattice, requested phi, lattice constant, nearest-neighbour distance,
    # requested and actual N, and final box size.

    # In principle, phi_actual should match phi exactly if a was derived from phi
    # and box size is built consistently from the same a.
    # This note guards against tiny floating-point effects or design mismatches.
    if abs(phi_actual - phi) > 1e-5:
        print(f"  [NOTE] Small phi discrepancy due to N adjustment: "
              f"Delta(phi) = {abs(phi_actual - phi):.2e}")
    print(sep)

    # Reports if the requested particle count could not be matched exactly
    # and had to be increased to the nearest constructible supercell.
    if n_req != n_actual:
        print(f"\n  [NOTE] Requested {n_req} particles; adjusted to {n_actual} "
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
    frame.particles.orientation = np.tile([1, 0, 0, 0], (n_actual, 1)).astype(np.float32)
    # All particles are assigned type index 0.
    frame.particles.typeid      = np.zeros(n_actual, dtype=np.int32)
    # The actual type name corresponding to typeid 0 is "A".
    frame.particles.types       = ["A"]
    # Every particle is assigned the same diameter sigma.
    frame.particles.diameter    = np.full(n_actual, diameter, dtype=np.float32)
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
    print(f"  Use sim.create_state_from_gsd('{output_gsd}') in HOOMD.")
    print(sep)
    # ---- HOOMD v4 HPMC overlap check --------------------------------------
    print("\n  Running HOOMD-v4 HPMC overlap check …", flush=True)

    hoomd_device = hoomd.device.CPU()
    hoomd_sim = hoomd.Simulation(device=hoomd_device, seed=1)

    # HPMC hard-sphere integrator
    mc = hoomd.hpmc.integrate.Sphere()

    # Define the hard-sphere diameter for particle type "A"
    mc.shape["A"] = dict(diameter=diameter)

    # Attach integrator and initialize the state from the GSD we just wrote
    hoomd_sim.operations.integrator = mc
    hoomd_sim.create_state_from_gsd(filename=output_gsd)

    # Initialize / attach operations cleanly before querying overlaps
    hoomd_sim.run(0)

    hoomd_overlap_count = mc.overlaps
    print(f"  HOOMD overlap count = {hoomd_overlap_count}")

    if hoomd_overlap_count > 0:
        print(f"  [ERROR] HOOMD detected {hoomd_overlap_count} overlapping pair(s).")
    else:
        print("  [OK] HOOMD detected zero overlapping pairs.")

    # ---- Quick overlap sanity check ----------------------------------------
    if n_actual <= 5000:
        # For manageable system sizes, do a brute-force overlap verification.
        print("\n  Running overlap sanity check …", end=" ", flush=True)
        # Count overlaps and find the smallest pair distance.
        n_ov, min_d = overlap_check(positions, diameter, Lx, Ly, Lz)
        print(f"overlaps = {n_ov},  min pair distance = {min_d:.6f}")

        if n_ov > 0:
            print(f"  [ERROR] Overlaps detected!  min_d={min_d:.6f} < sigma={diameter}.")
        else:
            print("  [OK] Zero overlaps in initial configuration.")
    else:
        print(f"\n  Skipping overlap check (N={n_actual} > 5000).")
    # The pairwise overlap check scales as O(N^2), so it is skipped for large systems.


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
