#!/usr/bin/env python3
# =============================================================================
# HOOMD-blue v4  |  Hard-Sphere HPMC Compression (Heavily Documented Version)
# =============================================================================
#
# FILE: hard_sphere_compression_with_fixed_move_size.py
# VERSION: 11
# AUTHOR: Kaustav Chakraborty
# DATE: 2025
#
# =============================================================================
# PURPOSE & METHODOLOGY
# =============================================================================
#
# This script performs systematic compression of a hard-sphere system from an
# initial low-density configuration to a target packing fraction using HOOMD-blue
# v4's Hard Particle Monte Carlo (HPMC) engine.
#
# COMPRESSION ALGORITHM:
# ----------------------------------------------------------
# The algorithm uses a two-level nested loop structure:
#
#   OUTER WHILE LOOP: while current_volume > target_volume
#       1. Calculate new volume: max(current_volume * scaling_factor, target_volume)
#       2. Resize simulation box instantly to new volume
#       3. INNER WHILE LOOP: while overlaps > 0
#           a. Run HPMC for run_length_to_remove_overlap steps
#           b. Check overlap count
#           c. Continue until ZERO overlaps achieved
#       4. Run fixed-box equilibration for run_length_to_relax steps
#       5. Write current_pf.json checkpoint file
#       6. Write output_current_pf.gsd snapshot
#       7. Continue outer loop until target volume reached
#
# This methodology ensures:
#   - Gradual compression via volume_scaling_factor (typically 0.99 = 1% shrinkage)
#   - Complete overlap removal at each step (hard-sphere constraint)
#   - Equilibration at fixed density before next compression step
#   - Reproducibility through fixed random seed
#
# =============================================================================
# KEY HOOMD-BLUE V4 FEATURES LEVERAGED
# =============================================================================
#
# 1. HPMC SPHERE INTEGRATOR (hoomd.hpmc.integrate.Sphere):
#    - Implements hard-sphere Monte Carlo moves
#    - Automatically detects and counts overlaps
#    - Provides acceptance/rejection statistics
#
# 2. INSTANTANEOUS BOX RESIZING (hoomd.update.BoxResize.update):
#    - Static method for immediate affine box rescaling
#    - All particle positions scaled proportionally
#    - Preserves box tilt factors (xy, xz, yz)
#
# 3. GSD TRAJECTORY MANAGEMENT (hoomd.write.GSD):
#    - Mode "ab" (append binary): Restarts add to existing trajectory
#    - Mode "wb" (write binary): Restart file truncated each write
#    - Dynamic attributes: Logs evolve during simulation
#
# 4. CUSTOM LOGGER SYSTEM (hoomd.logging.Logger):
#    - Extensible via Python properties with _export_dict
#    - DataAccessError protection: HPMC counters uninitialized before first run
#    - Categories: scalar (float/int), string, sequence (arrays)
#    - Namespacing: ("category", "quantity_name") tuples
#
# 5. MPI-AWARE EXECUTION:
#    - Rank 0 performs I/O operations
#    - All ranks execute simulation identically
#    - Seed file ensures synchronized random state
#    - Environment variable detection (OMPI_, PMI_, SLURM_)
#
# 6. DEVICE SELECTION (GPU/CPU):
#    - hoomd.device.GPU: CUDA-accelerated HPMC
#    - hoomd.device.CPU: Fallback for systems without GPU
#    - Automatic fallback on GPU initialization failure
#
# =============================================================================
# CRITICAL ASSUMPTIONS & REQUIREMENTS
# =============================================================================
#
# ASSUMPTIONS:
# -----------
# 1. MONODISPERSE SYSTEM: All particles have identical diameter
#    - Script reads diameter from GSD and validates uniformity
#    - Polydisperse systems will be rejected with error
#
# 2. SINGLE PARTICLE TYPE: Only one particle type 'A'
#    - HPMC integrator configured for type 'A' only
#    - Multi-type systems require code modification
#
# 3. HARD-SPHERE POTENTIAL: Purely geometric constraint
#    - No attractive/repulsive forces
#    - Only excluded volume interactions
#    - Overlap = configuration is invalid
#
# 4. COMPRESSION DIRECTION: Packing fraction increases
#    - Initial phi < target phi is enforced
#    - Expansion (phi decreasing) not supported
#
# 5. ISOTROPIC BOX: Box scales uniformly in all directions
#    - Lx, Ly, Lz scaled by same factor
#    - Aspect ratio preserved throughout compression
#
# 6. PERIODIC BOUNDARY CONDITIONS: Implicit in HOOMD
#    - Particles wrap around box edges
#    - No walls or confinement
#
# INPUT FILE REQUIREMENTS:
# -----------------------
# 1. GSD FORMAT: Binary HOOMD snapshot file
#    - Must contain particles.position (Nx3 array)
#    - Must contain particles.typeid (N array, all zeros for type 'A')
#    - Must contain particles.diameter (N array, all equal)
#    - Must have valid box dimensions (Lx, Ly, Lz > 0)
#
# 2. NON-OVERLAPPING INITIAL STATE:
#    - Input configuration must have zero overlaps
#    - Script checks mc.overlaps after loading
#    - Overlapping initial states will fail compression
#
# 3. LOW INITIAL DENSITY:
#    - phi_initial < phi_target is validated
#    - Typical starting density: phi ~ 0.1 - 0.3
#
# =============================================================================
# PARAMETERS REQUIRING CAREFUL TUNING
# =============================================================================
#
# 1. volume_scaling_factor (JSON: "volume_scaling_factor"):
#    - Range: [0.5, 1.0)
#    - Typical: 0.99 (1% volume reduction per step)
#    - Smaller values: Faster compression, higher overlap risk
#    - Larger values: Slower compression, more equilibrated states
#    - Balance: 0.98-0.995 for most systems
#
# 2. run_length_to_remove_overlap (JSON: "run_length_to_remove_overlap"):
#    - Minimum HPMC steps per inner overlap-removal attempt
#    - Range: 500 - 10000 steps
#    - Underprediction: Inner loop may cycle many times
#    - Overprediction: Wastes computation when overlaps resolve quickly
#
# 3. run_length_to_relax (JSON: "run_length_to_relax"):
#    - Fixed-box equilibration after overlaps removed
#    - Range: 500 - 10000 steps
#    - Purpose: Allow local rearrangements, decorrelate from previous state
#    - Too short: Insufficient equilibration
#    - Too long: Diminishing returns, wasted computation
#
# 4. move_size_translation (JSON: "move_size_translation"):
#    - Translational Monte Carlo step size (units: particle diameter)
#    - Range: 0.01 - 0.2 (typical: 0.04)
#    - Too small: Low acceptance, inefficient sampling
#    - Too large: High rejection at high density
#    - FIXED throughout run (no adaptive tuning in this version)
#    - Optimal: ~30-50% acceptance at target density
#
# 5. target_pf (JSON: "target_pf"):
#    - Target packing fraction (dimensionless)
#
# =============================================================================
# FAILURE MODES & TROUBLESHOOTING
# =============================================================================
#
# FAILURE MODE 1: Infinite inner loop (overlaps never reach zero)
# ---------------------------------------------------------------
# SYMPTOMS:
#   - Inner loop runs indefinitely, printing overlap counts
#   - Overlap count oscillates or decreases very slowly
#   - High CPU usage, no progress toward target phi
#
# CAUSES:
#   - volume_scaling_factor too small (too aggressive compression)
#   - move_size_translation too large (poor sampling at high phi)
#   - System entering jammed/glassy state
#   - Initial configuration has defects 
#
# SOLUTIONS:
#   - Increase volume_scaling_factor (e.g., 0.99 => 0.995)
#   - Decrease move_size_translation (e.g., 0.04 => 0.02)
#   - Increase run_length_to_remove_overlap (give more steps)
#   - Anneal system: compress to intermediate phi, expand, recompress
#
# FAILURE MODE 2: Kinetic arrest (no overlaps, but high rejection rate)
# ---------------------------------------------------------------------
# SYMPTOMS:
#   - Compression completes (overlaps = 0) but acceptance => 0%
#   - System "frozen" — particles barely move
#   - Target phi unreachable despite zero overlaps
#
# CAUSES:
#   - System in glassy state (phi > glass transition)
#   - Move size too large for available free volume
#   - Insufficient equilibration at intermediate densities
#
# SOLUTIONS:
#   - Stage compression: Compress to phi_1, equilibrate long, compress to phi_2
#   - Reduce move_size_translation to match local free volume
#   - Increase run_length_to_relax (more equilibration)
#   - Accept that phi_target may be kinetically inaccessible
#
# FAILURE MODE 3: Box volume underflow/overflow
# ----------------------------------------------
# SYMPTOMS:
#   - ValueError: box dimensions negative or NaN
#   - OverflowError in volume calculations
#
# CAUSES:
#   - Extreme target_pf (too high or negative)
#   - volume_scaling_factor < 0.5 (catastrophic compression)
#   - Numerical precision loss (float32 limitations)
#
# SOLUTIONS:
#   - Validate JSON parameters (see validation logic in code)
#   - Use float64 for all volume/dimension calculations
#   - Check input GSD box dimensions are reasonable
#
# =============================================================================
# WHAT THIS SCRIPT DOES NOT DO
# =============================================================================
#
# 1. ADAPTIVE MOVE SIZE TUNING:
#    - move_size_translation is FIXED throughout compression
#    - No hoomd.hpmc.tune.MoveSize tuner (intentional, to match v2)
#    - User must manually select appropriate d value
#
# 2. POLYDISPERSE SYSTEMS:
#    - Only monodisperse hard spheres supported
#    - Bidisperse/continuous size distributions require code rewrite
#
# 3. ANISOTROPIC COMPRESSION:
#    - Box scales isotropically (Lx = Ly = Lz scaling)
#    - Uniaxial/biaxial compression not implemented
#
# 4. SOFT POTENTIALS:
#    - HPMC is purely geometric (hard-core repulsion)
#    - Lennard-Jones, WCA, etc. require MD integrator
#
# 5. DETAILED BALANCE DIAGNOSTICS:
#    - No explicit detailed balance checks
#    - Assumes HPMC integrator maintains detailed balance
#
# 6. AUTOMATIC RESTARTS:
#    - Script does not self-restart on walltime/failure
#    - User must re-invoke script to resume from restart GSD
#
# =============================================================================
# DEPENDENCIES & VERSION REQUIREMENTS
# =============================================================================
#
# REQUIRED PACKAGES:
#   - HOOMD-blue >= 4.0  (tested with 4.9.0)
#   - GSD >= 3.0         (binary file I/O)
#   - NumPy >= 1.20      (array operations, minimal usage)
#   - Python >= 3.8      (dataclasses, type hints)
#
# =============================================================================
# USAGE EXAMPLES
# =============================================================================
#
# SINGLE-NODE CPU RUN:
#   python hs_compress_v10_documented.py --simulparam_file params.json
#
# SINGLE-NODE GPU RUN (specify GPU in JSON: use_gpu=true, gpu_id=0):
#   python hs_compress_v10_documented.py --simulparam_file params.json
#
# MULTI-GPU RUN (MPI, each rank uses different GPU):
#   # Set gpu_id dynamically based on local rank (requires wrapper script)
#   mpirun -np 4 python hs_compress_v10_documented.py --simulparam_file params.json
#
# RESUME FROM RESTART:
#   # If restart.gsd exists and final.gsd does not, script auto-resumes
#   python hs_compress_v10_documented.py --simulparam_file params.json
#
# STAGED COMPRESSION (multi-stage protocol):
#   # Stage 0: phi 0.1 => 0.4
#   # Edit JSON: stage_id_current=0, target_pf=0.4
#   python hs_compress_v10_documented.py --simulparam_file params.json
#   # Stage 1: phi 0.4 => 0.55
#   # Edit JSON: stage_id_current=1, target_pf=0.55
#   python hs_compress_v10_documented.py --simulparam_file params.json
#
# =============================================================================

from __future__ import annotations

# Standard library imports
import argparse                             # Command-line argument parsing
import json                                 # JSON parameter file I/O
import math                                 # Mathematical constants (pi)
import os                                   # Environment variable access (MPI rank detection)
import secrets                              # Cryptographically secure random number generation
import sys                                  # Exit with error codes, stdout flushing
import time                                 # Timestamps, performance timing
from dataclasses import dataclass, field    # Type-safe parameter containers
from pathlib import Path                    # Filesystem path operations
from typing import Optional                 # Type hints for nullable fields

# Third-party imports
import gsd.hoomd        # GSD binary file I/O for HOOMD snapshots
import numpy as np

# HOOMD-blue v4 imports with error handling
try:
    import hoomd
    import hoomd.hpmc
except ImportError as _e:
    sys.exit(
        f"[FATAL] Required package not found: {_e}\n"
        "Install HOOMD-blue v4 and GSD before running this script."
    )

# ---------------------------------------------------------------------------
# Mathematical constants
# ---------------------------------------------------------------------------
# Hard sphere volume: V = (π/6) * d³
# Precompute π/6 to avoid repeated divisions
# ---------------------------------------------------------------------------
_PI_OVER_6 = math.pi / 6.0
# ---------------------------------------------------------------------------


# ===========================================================================
#  SECTION 1: MPI-AWARE HELPER FUNCTIONS
# ===========================================================================
#
# HOOMD-blue supports MPI (Message Passing Interface) for distributed-memory
# parallel execution across multiple nodes/GPUs. In MPI runs, multiple
# identical copies (ranks) of the script execute simultaneously. Only rank 0
# should perform I/O operations (file writes, print statements) to avoid
# race conditions and duplicate outputs.
#
# These helper functions detect the current MPI rank from environment
# variables set by common MPI launchers (OpenMPI, Intel MPI, MPICH, SLURM).
#
# ===========================================================================

def _mpi_rank_from_env() -> int:
    """
    Detect the MPI rank from environment variables.
    
    Returns:
        int: The MPI rank of this process (0-indexed). Returns 0 if no MPI
             environment detected (single-process run).
    
    Notes:
        - Rank 0 is designated the "root" rank for I/O operations.
        - In non-MPI runs (serial execution), this always returns 0.
        - Environment variables are checked in order of specificity.
    
    Examples:
        >>> # In a 4-process MPI run on rank 2:
        >>> _mpi_rank_from_env()
        2
        >>> # In a non-MPI run:
        >>> _mpi_rank_from_env()
        0
    """

    return int(
        os.environ.get(
            "OMPI_COMM_WORLD_RANK",   # OpenMPI standard variable
            os.environ.get(
                "PMI_RANK",            # Intel MPI, MPICH variable
                os.environ.get(
                    "SLURM_PROCID",    # SLURM job scheduler variable
                    0                   # Default: rank 0 (non-MPI run)
                )
            )
        )
    )

def _is_root_rank() -> bool:
    """
    Check if this process is the MPI root rank (rank 0).
    
    Returns:
        bool: True if this is rank 0 (or non-MPI run), False otherwise.
    
    Usage:
        Used to guard I/O operations that should only be performed once
        across all MPI ranks:
        
        >>> if _is_root_rank():
        >>>     print("This message appears once in MPI runs")
        >>>     with open("output.txt", "w") as f:
        >>>         f.write("Data")
    
    Notes:
        - All ranks execute the same code, but only rank 0 performs I/O.
        - Conditional branches based on rank must be carefully designed
          to avoid deadlocks in collective operations.
        - Simulation.run() is a collective operation: all ranks must call it.
    """
    return _mpi_rank_from_env() == 0

def root_print(*args, **kwargs) -> None:
    """
    Print function that only produces output on MPI rank 0.
    
    Args:
        *args: Positional arguments passed to print()
        **kwargs: Keyword arguments passed to print()
    
    Returns:
        None
    
    Behavior:
        - Rank 0: Calls print() with all provided arguments
        - Ranks 1+: No-op (returns immediately without output)
    
    Usage:
        >>> root_print("Starting compression")  # Only rank 0 prints
        >>> root_print(f"Timestep: {sim.timestep}")
    
    Rationale:
        - Prevents duplicate console output in MPI runs (4 ranks = 4x spam)
        - Maintains clean, readable console logs
        - Reduces I/O contention on shared filesystems
    
    Notes:
        - Output is unbuffered by default (Python -u flag or PYTHONUNBUFFERED=1)
        - For logging to files, use explicit rank guards and unique filenames
    """
    if _is_root_rank():
        print(*args, **kwargs)

def root_flush_stdout() -> None:
    """
    Flush stdout buffer on MPI rank 0 only.
    
    Returns:
        None
    
    Purpose:
        Ensures console output is immediately visible during long-running
        simulations. Python buffers stdout by default, which can delay
        output by seconds or minutes in batch jobs.
    
    Usage:
        >>> root_print("Entering inner loop...")
        >>> root_flush_stdout()  # Force output to appear immediately
    
    When to use:
        - After printing status messages inside tight loops
        - Before long-running operations (sim.run(1000000))
        - In HPC environments where console logging is time-critical
    
    Notes:
        - Flushing too frequently can impact performance (syscall overhead)
        - Balance: Flush every N iterations, not every iteration
        - Automatic flush on newline can be enabled with PYTHONUNBUFFERED=1
    """
    if _is_root_rank():
        sys.stdout.flush()


# ===========================================================================
#  SECTION 2: PARAMETER DATACLASS
# ===========================================================================
#
# SimulationParams encapsulates all runtime configuration in a type-safe,
# validated container. This approach provides several advantages over raw
# dictionaries:

@dataclass
class SimulationParams:
    """
    Type-safe, validated container for all simulation runtime parameters.
    
    This dataclass represents the complete configuration of a hard-sphere
    compression simulation, loaded from a JSON parameter file. It enforces
    physical constraints and provides sensible defaults for optional fields.
    
    Attributes:
        # ===== I/O IDENTIFIERS =====
        tag (str):
            Unique identifier for this simulation run. Used as prefix for all
            output files (trajectories, logs, final snapshots). Should be
            descriptive and filesystem-safe (no spaces or special chars).
            Example: "hs_4096_compression_phi058"
        
        input_gsd_filename (str):
            Path to the input GSD file containing the initial hard-sphere
            configuration. 
            Example: "hs_lattice_SC_pf01.gsd"
        
        stage_id_current (int):
            Stage index for multi-stage compression protocols.
            Values:
                -1: Single-stage run (use filenames directly from JSON)
                 0: Multi-stage run, stage 0 (first stage, uses input_gsd_filename)
                 1+: Multi-stage run, stage N (reads <tag>_<N-1>_final.gsd)
        
        # ===== PHYSICAL PARAMETERS =====
        target_pf (float):
            Target packing fraction phi_target (dimensionless).
        
        volume_scaling_factor (float):
            Fractional volume reduction per compression step.
            
            Algorithm:
                new_volume = max(current_volume * volume_scaling_factor, target_volume)
        
        # ===== RUN LENGTH PARAMETERS =====
        run_length_to_remove_overlap (int):

            Number of HPMC sweeps per inner-loop iteration.
            
            Purpose:
                Each inner-loop iteration runs this many MC sweeps, then
                checks if overlaps have been eliminated. If overlaps remain,
                another iteration runs.
            
            Units: HPMC sweeps (1 sweep = N trial moves, where N = particle count)
            
            Example:
                N=4096 particles, run_length_to_remove_overlap=2500
                => Each inner iteration: 2500 sweeps = 10.24 million trial moves
        
        run_length_to_relax (int):
            Number of HPMC sweeps for fixed-box equilibration after overlap removal.
            
            Purpose:
                After overlaps are eliminated, run this many additional sweeps
                at constant volume to allow local structural relaxation and
                decorrelate from the compressed state.
            
            Units: HPMC sweeps
            
            Range: 500 - 10000 (typical), can be 0 (no relaxation)
        
        # ===== MONTE CARLO MOVE PARAMETERS =====
        move_size_translation (float):
            Maximum translational displacement per MC trial move.
            
            Units: Particle diameters (d)
        
        # ===== OUTPUT CONTROL =====
        restart_frequency (int):
            Timestep interval for writing restart checkpoints.
           
            Restart logic:
                If restart.gsd exists AND final.gsd does not exist:
                    => Script auto-resumes from restart.gsd
                Else:
                    => Fresh run from input GSD
        
        traj_out_freq (int):
            Timestep interval for appending frames to trajectory file.
            
            Units: HPMC timesteps
            
            Purpose:
                Build multi-frame trajectory for visualization and analysis.
                Each frame captures particle positions, box dimensions, and
                logged quantities (phi, overlaps, acceptance, etc.)
            
            Typical values: 100000 - 1000000
            
            Storage implications:
                - Each frame: ~N * 12 bytes (3 floats per particle position)
                - Example: 4096 particles, 100 frames → ~5 MB trajectory
            
            Notes:
                - Mode "ab" (append binary): Restarts add to existing trajectory
                - Logged quantities stored in log/* namespace within GSD
        
        log_frequency (int):
            Timestep interval for console (stdout) and log file output.
            
            Units: HPMC timesteps
        
        # ===== OPTIONAL FIELDS (with defaults) =====
        initial_timestep (int):
            Starting timestep for fresh runs. Default: 0
        
        restart_gsd_filename (str):
            Filename for restart checkpoint. Default: "restart.gsd"
            
            Used only if stage_id_current == -1 (single-stage).
            Multi-stage runs use <tag>_<stage_id>_restart.gsd
        
        output_gsd_traj_filename (str):
            Filename for trajectory. Default: "traj.gsd"
            
            Used only if stage_id_current == -1.
            Multi-stage: <tag>_<stage_id>_traj.gsd
        
        final_gsd_filename (str):
            Filename for final snapshot. Default: "final.gsd"
            
            Used only if stage_id_current == -1.
            Multi-stage: <tag>_<stage_id>_final.gsd
        
        # ===== RUNTIME-RESOLVED FIELDS =====
        diameter (Optional[float]):
            Particle diameter, read from input GSD at runtime.
            
            Not specified in JSON (set to None initially).
            Script reads diameter from GSD file and validates monodispersity
            before compression begins.
    """

    # --- I/O identifiers ---
    tag:                  str   # Simulation identifier (output filename prefix)
    input_gsd_filename:   str   # Path to initial GSD configuration
    stage_id_current:     int   # -1 = single-stage; >= 0 = multi-stage

    # --- Physical parameters ---
    target_pf:            float  # target packing fraction
    volume_scaling_factor: float # Box volume multiplier per compression step

    # --- Run length parameters ---
    run_length_to_remove_overlap: int  # steps per inner overlap-removal attempt
    run_length_to_relax:          int  # NVT steps after overlaps reach zero

    # --- HPMC move size (FIXED — no adaptive tuner) ---
    move_size_translation: float  # translational move size d

    # --- Output frequencies ---
    restart_frequency:    int     # Timestep interval for restart checkpoints
    traj_out_freq:        int     # Timestep interval for trajectory frames
    log_frequency:        int     # Timestep interval for console/log output

    # --- Hardware --- 
    use_gpu:              bool   # Enable GPU acceleration (True/False)
    gpu_id:               int    # CUDA device index (0-indexed)

    # --- Optional / single-stage defaults ---
    initial_timestep:             int  = 0                # Starting timestep (fresh runs)
    restart_gsd_filename:         str  = "restart.gsd"    # Restart checkpoint file
    output_gsd_traj_filename:     str  = "traj.gsd"       # Trajectory file
    final_gsd_filename:           str  = "final.gsd"      # Final snapshot file

    # Resolved at runtime from the GSD file
    diameter: Optional[float] = None                      # Particle diameter (read from GSD)

    def validate(self) -> None:
        """
        Validate all parameters against physical and numerical constraints.
        
        This method is called after loading parameters from JSON and before
        starting the simulation. It checks that all values are physically
        reasonable and mathematically valid.
        
        Raises:
            ValueError: If any parameter violates a constraint. The exception
                        message lists all validation errors encountered.
        
        Validation rules:
            1. target_pf: Must be in (0, 0.74048)
               
            2. volume_scaling_factor: Must be in [0.5, 1.0)
               Rationale: Values < 0.5 are catastrophic compression
                          Values >= 1.0 are expansion (not compression)
               
            3. run_length_to_remove_overlap: Must be > 0
               
            4. run_length_to_relax: Must be >= 0
               Rationale: 0 is valid (no relaxation), negative is nonsensical
               
            5. move_size_translation: Must be > 0
               
            6. restart_frequency, traj_out_freq, log_frequency: Must be > 0
               
            7. stage_id_current: Must be >= -1
               Rationale: -1 is single-stage, 0+ is multi-stage, < -1 invalid
        
        Notes:
            - This method does NOT check file existence (handled elsewhere)
            - Warnings for questionable-but-valid values (e.g., target_pf > 0.64)
              are printed separately, not raised as exceptions
        """
        errors: list[str] = []       # Accumulate all errors before raising
        # Validate target packing fraction
        if not (0.0 < self.target_pf < 0.74048):
            errors.append(f"target_pf must be in (0, 0.740), got {self.target_pf}")
        # Validate volume scaling factor
        if not (0.5 <= self.volume_scaling_factor < 1.0):
            errors.append(
                f"volume_scaling_factor must be in [0.5, 1.0), got {self.volume_scaling_factor}"
            )
        # Validate overlap removal run length
        if self.run_length_to_remove_overlap <= 0:
            errors.append(
                f"run_length_to_remove_overlap must be > 0, got {self.run_length_to_remove_overlap}"
            )
        # Validate relaxation run length
        if self.run_length_to_relax < 0:
            errors.append(
                f"run_length_to_relax must be >= 0, got {self.run_length_to_relax}"
            )
        # Validate move size
        if self.move_size_translation <= 0.0:
            errors.append(
                f"move_size_translation must be > 0, got {self.move_size_translation}"
            )
        # Validate output frequencies
        for name, val in [
            ("restart_frequency", self.restart_frequency),
            ("traj_out_freq",     self.traj_out_freq),
            ("log_frequency",     self.log_frequency),
        ]:
            if val <= 0:
                errors.append(f"{name} must be > 0, got {val}")
        # Validate stage ID
        if self.stage_id_current < -1:
            errors.append(
                f"stage_id_current must be >= -1, got {self.stage_id_current}"
            )
        # If any validation errors, raise exception with all error messages
        if errors:
            raise ValueError("Parameter validation failed:\n" +
                             "\n".join(f"  • {e}" for e in errors))


# ===========================================================================
#  SECTION 3: JSON PARAMETER LOADING
# ===========================================================================
#
# This section handles reading and parsing the JSON parameter file,
# performing type checking, and constructing a validated SimulationParams
# instance.
#
# We explicitly check that each required key exists and has the expected type.
# This catches typos, missing fields, and type errors before simulation starts.
#
# ===========================================================================

# Required JSON keys and their expected Python types
# These keys MUST be present in the JSON file for a valid configuration
_REQUIRED_KEYS: dict[str, type] = {
    "tag":                           str,              # Simulation identifier
    "input_gsd_filename":            str,              # Initial GSD path
    "stage_id_current":              int,              # Stage index (-1 or >=0)
    "target_pf":                     (int, float),     # Target packing fraction
    "volume_scaling_factor":         (int, float),     # Volume scaling per step
    "run_length_to_remove_overlap":  int,              # Overlap removal HPMC steps
    "run_length_to_relax":           int,              # Relaxation HPMC steps
    "move_size_translation":         (int, float),     # MC move size
    "restart_frequency":             int,              # Restart checkpoint interval
    "traj_out_freq":                 int,              # Trajectory frame interval
    "log_frequency":                 int,              # Console log interval
    "use_gpu":                       bool,             # GPU enable flag
    "gpu_id":                        int,              # CUDA device index
}

# Optional — have defaults in the dataclass
_OPTIONAL_KEYS: dict[str, type] = {
    "initial_timestep":          int,
    "restart_gsd_filename":      str,
    "output_gsd_traj_filename":  str,
    "final_gsd_filename":        str,
    "traj_out_freq_POS":         int,
    "color_code_for_POS_file":   str,
}

# Keys that are parsed from JSON but are not required (comment-keys in JSON)
# — any key starting with "_" is silently stripped before type-checking.


def load_simulparams(json_path: str) -> SimulationParams:
    """
    Load, validate, and return simulation parameters from a JSON file.
    
    This function performs the complete parameter loading workflow:
        1. Check file existence
        2. Parse JSON (with error handling)
        3. Strip comment keys (keys starting with "_")
        4. Validate presence of required keys
        5. Type-check all keys against expected types
        6. Construct SimulationParams instance
        7. Call params.validate() for physical constraint checks
        8. Return validated params object
    
    Args:
        json_path (str): Path to the JSON parameter file.
    
    Returns:
        SimulationParams: Validated parameter object ready for simulation.
    
    Raises:
        SystemExit: If any validation step fails. Exit codes:
            - File not found
            - JSON parse error (malformed JSON syntax)
            - Missing required keys
            - Type mismatch on required keys
            - Physical constraint violation (from params.validate())
    
    Error handling:
        All errors are fatal (sys.exit). This is by design:
            - Parameter errors should be fixed before running, not ignored
    
    Example usage:
        >>> params = load_simulparams("simulparam.json")
        >>> print(f"Target phi: {params.target_pf}")
    
    Notes:
        - Only rank 0 prints error messages (via sys.exit), but all ranks exit
        - Comment keys (starting with "_") are silently ignored
        - Numeric fields are explicitly coerced to float where needed
    """

    # Convert string path to Path object for robust filesystem operations
    path = Path(json_path)
    
    # Check 1: File existence
    if not path.exists():
        sys.exit(
            f"[FATAL ERROR] Simulation parameter file not found: {json_path}\n"
            f"  Expected location: {path.absolute()}\n"
            f"  Current directory: {Path.cwd()}\n"
            f"  Check file path and spelling."
        )
    
    # Check 2: JSON parsing
    try:
        with path.open(mode='r', encoding='utf-8') as file_handle:
            raw_data: dict = json.load(file_handle)
    except json.JSONDecodeError as json_error:
        # JSONDecodeError includes line/column information
        sys.exit(
            f"[FATAL ERROR] JSON parse error in '{json_path}':\n"
            f"  {json_error}\n"
            f"  => Check JSON syntax (commas, quotes, brackets).\n"
        )
    except Exception as unexpected_error:
        # Catch other I/O errors (permissions, encoding issues)
        sys.exit(
            f"[FATAL ERROR] Failed to read '{json_path}': {unexpected_error}\n"
        )
    
    # Check 3: Strip comment keys
    raw_data = {
        key: value
        for key, value in raw_data.items()
        if not key.startswith("_")
    }
    
    # Check 4: Validate required keys are present
    # Build list of all missing required keys
    missing_keys = [
        key
        for key in _REQUIRED_KEYS
        if key not in raw_data
    ]
    
    if missing_keys:
        sys.exit(
            f"[FATAL ERROR] Missing required keys in '{json_path}':\n" +
            "\n".join(f"  {key}" for key in missing_keys) +
            f"\n\nRequired keys:\n" +
            "\n".join(f"  {key}: {typ.__name__}" for key, typ in _REQUIRED_KEYS.items())
        )
    
    # Check 5: Type checking for required keys
    # Validate that each required key has the expected type
    type_errors: list[str] = []
    
    for key, expected_type in _REQUIRED_KEYS.items():
        if key in raw_data:
            if not isinstance(raw_data[key], expected_type):
                if isinstance(expected_type, tuple):
                    type_str = " or ".join(t.__name__ for t in expected_type)
                else:
                    type_str = expected_type.__name__
                
                actual_type = type(raw_data[key]).__name__
                type_errors.append(
                    f"  '{key}': expected {type_str}, got {actual_type} "
                    f"(value: {raw_data[key]})"
                )
    
    if type_errors:
        sys.exit(
            f"[FATAL ERROR] Type errors in '{json_path}':\n" +
            "\n".join(type_errors)
        )
    
    # Step 6: Build keyword argument dict for SimulationParams constructor
    constructor_kwargs: dict = {
        key: raw_data[key]
        for key in _REQUIRED_KEYS
    }
    
    # Add optional keys if present in JSON (otherwise use dataclass defaults)
    for key in _OPTIONAL_KEYS:
        if key in raw_data:
            constructor_kwargs[key] = raw_data[key]
    
    # Step 7: Numeric type coercion
    # JSON may parse "target_pf": 1 as int, but we need float
    # Explicitly convert numeric parameters that must be float
    for float_field in ["target_pf", "volume_scaling_factor", "move_size_translation"]:
        constructor_kwargs[float_field] = float(constructor_kwargs[float_field])
    
    # Step 8: Construct SimulationParams instance
    # **kwargs unpacks dictionary as keyword arguments
    params = SimulationParams(**constructor_kwargs)
    
    # Step 9: Physical constraint validation
    # This may raise ValueError if constraints violated
    try:
        params.validate()
    except ValueError as validation_error:
        # Re-raise as SystemExit for consistent error handling
        sys.exit(f"[FATAL ERROR] {validation_error}")
    
    # Step 10: Return validated params object
    return params


# ===========================================================================
#  SECTION 4: RANDOM SEED MANAGEMENT
# ===========================================================================
# HOOMD's random number generator is seeded once at Simulation creation.
# All subsequent MC moves use this seed to generate pseudorandom numbers.
# If all ranks use the same seed, the simulation is fully reproducible.

# SEED FILE STRATEGY:
# ------------------
# To ensure reproducibility across runs and stages:
#   1. On first invocation: Rank 0 generates a random seed, writes to JSON file
#   2. On subsequent invocations: All ranks read seed from file
#   3. Multi-stage runs: Stage 0 seed file used for all stages
#
# This ensures:
#   - Restarts: Same seed → continue from exact same random state
#   - Multi-stage: Different stages use same seed sequence
#   - MPI: All ranks synchronized on same seed value
#
# ===========================================================================

# Seed file names for single-stage and multi-stage runs
_SEED_FILE_SINGLE = "random_seed.json"        # stage_id == -1
_SEED_FILE_MULTI  = "random_seed_stage_0.json" # stage_id >= 0


def _os_random_seed() -> int:
    """
    Cryptographically secure seed in [0, 65535].
    """
    return secrets.randbelow(65536)   # randbelow(n) returns [0, n-1], so max = 65535


def ensure_seed_file(stage_id: int, rank: int) -> None:
    """
    Create seed file on rank 0 if it doesn't exist; all other ranks do nothing.
    
    This function implements the seed file initialization logic:
        - First invocation (file doesn't exist): Rank 0 creates it
        - Subsequent invocations: File exists, nothing to do
        - Non-root ranks: Always no-op (avoid race conditions)
    
    Args:
        stage_id (int): Current stage index (-1 for single-stage, >=0 for multi-stage).
        rank (int): MPI rank of the calling process.
    
    Returns:
        None
    
    
    Filename logic:
        - Single-stage (stage_id == -1): "random_seed.json"
        - Multi-stage (stage_id >= 0):   "random_seed_stage_0.json"
    """    

    # Non-root ranks: immediate return (avoid race condition)
    if rank != 0:
        return

    # Determine seed filename based on stage_id
    seed_file = _SEED_FILE_SINGLE if stage_id == -1 else _SEED_FILE_MULTI

    # Check if file already exists
    if not Path(seed_file).exists():
        # File doesn't exist: generate seed and create file
        seed = _os_random_seed()
        
        try:
            with open(seed_file, mode='w', encoding='utf-8') as file_handle:
                json.dump(
                    {
                        "random_seed": seed,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")
                    },
                    file_handle,
                    indent=4
                )
            # Confirmation message (rank 0 only)
            print(f"[INFO] Seed file created: {seed_file}  (seed={seed})")
        
        except Exception as write_error:
            # Propagate exception if file write fails (permissions, disk full, etc.)
            sys.exit(
                f"[FATAL ERROR] Failed to write seed file '{seed_file}': {write_error}\n"
                f"  => Check directory permissions and disk space."
            )
    
    else:
        # File exists: nothing to do, just print confirmation
        print(f"[INFO] Existing seed file found: {seed_file}")


def read_seed(stage_id: int) -> int:
    """
    Read and return the random seed from the appropriate seed file.
    
    Args:
        stage_id (int): Current stage index (-1 or >=0).
    
    Returns:
        int: The random seed value (0-65535).
    
    Raises:
        SystemExit: If seed file doesn't exist or is malformed.
    
    Notes:
        - All MPI ranks call this function (not just rank 0)
        - If different ranks get different seeds, simulation will diverge
        - The seed file barrier in main() prevents reads before file exists
    """
    
    # Determine seed filename    
    seed_file = _SEED_FILE_SINGLE if stage_id == -1 else _SEED_FILE_MULTI
    
    try:
        # Read and parse JSON file
        with open(seed_file, mode='r', encoding='utf-8') as file_handle:
            data = json.load(file_handle)
        
        # Extract and return seed value as integer
        return int(data["random_seed"])
    
    except FileNotFoundError:
        # Seed file doesn't exist (should be caught by barrier)
        sys.exit(
            f"[FATAL ERROR] Seed file not found: '{seed_file}'\n"
        )
    
    except KeyError:
        # "random_seed" key missing (file corrupted or wrong structure)
        sys.exit(
            f"[FATAL ERROR] 'random_seed' key missing in '{seed_file}'\n"
            f"  File may be corrupted or have wrong structure.\n"
            f"  => Delete the file and re-run to regenerate."
        )
    
    except (json.JSONDecodeError, ValueError) as parse_error:
        # Malformed JSON or non-integer seed value
        sys.exit(
            f"[FATAL ERROR] Cannot parse seed from '{seed_file}': {parse_error}\n"
            f"  => Delete the file and re-run to regenerate."
        )
    
    except Exception as unexpected_error:
        # Catch-all for other errors (permissions, I/O errors)
        sys.exit(
            f"[FATAL ERROR] Unexpected error reading '{seed_file}': {unexpected_error}"
        )    


# ===========================================================================
#  SECTION 5: FILENAME RESOLUTION (STAGE-AWARE)
# ===========================================================================
#
# SINGLE-STAGE vs MULTI-STAGE RUNS:
# ---------------------------------
# Single-stage (stage_id == -1):
#     User provides explicit filenames in JSON:
#         restart_gsd_filename: "my_restart.gsd"
#         output_gsd_traj_filename: "my_traj.gsd"
#         final_gsd_filename: "my_final.gsd"
#
# Multi-stage (stage_id >= 0):
#     Filenames follow convention: <tag>_<stage_id>_<suffix>
#         Stage 0: hs_4096_0_restart.gsd, hs_4096_0_traj.gsd, hs_4096_0_final.gsd
#         Stage 1: hs_4096_1_restart.gsd, hs_4096_1_traj.gsd, hs_4096_1_final.gsd
#     
#     Input GSD:
#         Stage 0: Uses input_gsd_filename from JSON
#         Stage N: Uses <tag>_<N-1>_final.gsd (output of previous stage)
# ===========================================================================

@dataclass
class RunFiles:
    """
    Container for all resolved filenames for the current simulation run.
    
    This dataclass holds the actual filenames that will be used for I/O
    operations, after resolving stage-aware naming conventions.
    
    Attributes:
        input_gsd (str):
            GSD file to load initial particle configuration from.
            - Single-stage: From JSON input_gsd_filename
            - Multi-stage stage 0: From JSON input_gsd_filename
            - Multi-stage stage N: <tag>_<N-1>_final.gsd
        
        restart_gsd (str):
            GSD file for restart checkpoints (single-frame, overwritten).
            - Single-stage: From JSON restart_gsd_filename
            - Multi-stage: <tag>_<stage_id>_restart.gsd
        
        traj_gsd (str):
            GSD file for trajectory (multi-frame, appended).
            - Single-stage: From JSON output_gsd_traj_filename
            - Multi-stage: <tag>_<stage_id>_traj.gsd
        
        final_gsd (str):
            GSD file for final snapshot (marks stage completion).
            - Single-stage: From JSON final_gsd_filename
            - Multi-stage: <tag>_<stage_id>_final.gsd
        
        log_txt (str):
            Text log file (not used by current implementation).
    """

    input_gsd:      str     # Input GSD path (initial configuration)
    restart_gsd:    str     # Restart checkpoint path
    traj_gsd:       str     # Trajectory path
    final_gsd:      str     # Final snapshot path
    log_txt:        str     # Log file path (future use)


def resolve_filenames(params: SimulationParams) -> RunFiles:
    """
    Resolve all output filenames with stage-aware naming and validation.
    
    This function implements the complete filename resolution logic:
        1. Determine naming convention (single-stage vs multi-stage)
        2. Construct filenames according to convention
        3. Validate preconditions (final exists? previous stage complete?)
        4. Check input GSD exists
        5. Return RunFiles object with all resolved paths
    
    Args:
        params (SimulationParams): Validated simulation parameters.
    
    Returns:
        RunFiles: Resolved filenames for all I/O operations.
    
    Raises:
        SystemExit: If any validation check fails:
            - Final GSD already exists (multi-stage, prevents re-run)
            - Previous stage's final GSD missing (multi-stage, can't continue)
            - Input GSD doesn't exist (can't start simulation)
            - Invalid stage_id (< -1)
    
    Notes:
        - Only rank 0 prints warnings (via root_print), but all ranks resolve
        - All ranks must agree on filenames (deterministic logic)
        - File existence checks use pathlib.Path for cross-platform compatibility
    """    

    tag = params.tag
    sid = params.stage_id_current


    # -----------------------------------------------------------------------
    # CASE 1: Single-stage run (stage_id == -1)
    # -----------------------------------------------------------------------
    if sid == -1:
        # Use filenames directly from JSON (user-provided)
        input_gsd   = params.input_gsd_filename
        restart_gsd = params.restart_gsd_filename
        traj_gsd    = params.output_gsd_traj_filename
        final_gsd   = params.final_gsd_filename
        log_txt     = f"{tag}.log"

        # Warning if final GSD already exists (not an error in single-stage)
        # User may be resuming from restart, or deliberately overwriting
        if Path(final_gsd).exists():
            root_print(
                f"[WARNING] Final GSD file '{final_gsd}' already exists.\n"
                f"          All output files will be either appended (traj) "
                f"or overwritten (final).\n"
                f"          If this is unintentional, rename or move the existing file."
            )

    # -----------------------------------------------------------------------
    # CASE 2: Multi-stage run (stage_id >= 0)
    # -----------------------------------------------------------------------
    elif sid >= 0:
        # Construct filenames with <tag>_<stage_id>_<suffix> convention
        pfx         = f"{tag}_{sid}"
        restart_gsd = f"{pfx}_restart.gsd"
        traj_gsd    = f"{pfx}_traj.gsd"
        final_gsd   = f"{pfx}_final.gsd"
        log_txt     = f"{pfx}.log"

        # Guard 1: Prevent re-running the same stage
        # If final GSD exists, this stage has completed
        # User must increment stage_id_current to run next stage
        if Path(final_gsd).exists():
            sys.exit(
                f"[FATAL ERROR] Final GSD file '{final_gsd}' already exists.\n"
                f"              This indicates stage {stage_id} has already completed.\n"
                f"              To run the next stage:\n"
                f"                1. Edit JSON: set stage_id_current to {stage_id + 1}\n"
                f"                2. Re-run this script\n"
                f"              To re-run this stage:\n"
                f"                1. Rename or delete '{final_gsd}'\n"
                f"                2. Re-run this script"
            )

        # Determine input GSD based on stage number
        if sid == 0:
            # Stage 0: Use input_gsd_filename from JSON (initial configuration)
            input_gsd = params.input_gsd_filename
        else:
            # Stage N (N >= 1): Use previous stage's final GSD as input
            input_gsd = f"{tag}_{sid - 1}_final.gsd"

            # Guard 2: Check that previous stage completed successfully
            # If previous stage's final GSD is missing, can't continue
            if not Path(input_gsd).exists():
                sys.exit(
                    f"[FATAL ERROR] Input GSD from previous stage not found: '{input_gsd}'\n"
                    f"              Stage {stage_id} requires output from stage {stage_id - 1}.\n"
                    f"              Did stage {stage_id - 1} complete successfully?\n"
                    f"              Check for errors in stage {stage_id - 1} log files.\n"
                    f"              Possible causes:\n"
                    f"                - Stage {stage_id - 1} failed or was terminated\n"
                    f"                - Wrong stage_id_current value in JSON\n"
                    f"                - File deleted or moved\n"
                )

    # -----------------------------------------------------------------------
    # CASE 3: Invalid stage_id (< -1)
    # -----------------------------------------------------------------------
    else:
        # This should be caught by params.validate(), but double-check
        sys.exit(
            f"[FATAL ERROR] Invalid stage_id_current: {stage_id}\n"
            f"              Must be >= -1 (-1 for single-stage, 0+ for multi-stage)."
        )
    
    # -----------------------------------------------------------------------
    # Final validation: Check input GSD exists
    # -----------------------------------------------------------------------
    # This applies to both single-stage and multi-stage
    # If input GSD is missing, simulation cannot start
    if not Path(input_gsd).exists():
        sys.exit(
            f"[FATAL ERROR] Input GSD file not found: '{input_gsd}'\n"
            f"              Cannot start simulation without initial configuration.\n"
            f"              For single-stage runs:\n"
            f"                - Check 'input_gsd_filename' in JSON\n"
            f"                - Verify file path is correct and accessible\n"
            f"              For multi-stage runs:\n"
            f"                - Ensure previous stage completed successfully\n"
            f"                - Check that previous stage wrote '{input_gsd}'\n"
            f"              Current directory: {Path.cwd()}"
        )
    
    # -----------------------------------------------------------------------
    # Return resolved filenames
    # -----------------------------------------------------------------------
    return RunFiles(
        input_gsd=input_gsd,
        restart_gsd=restart_gsd,
        traj_gsd=traj_gsd,
        final_gsd=final_gsd,
        log_txt=log_txt,
    )


# ===========================================================================
#  SECTION 6: PACKING FRACTION UTILITIES
# ===========================================================================
#
# PACKING FRACTION DEFINITION:
# ---------------------------
# For a system of N identical hard spheres with diameter d in a box of
# volume V_box, the packing fraction φ is:
#
#     phi = (N * V_sphere) / V_box
#
# where V_sphere = (pi/6) * d^3 is the volume of a single sphere.

def sphere_volume(diameter: float) -> float:
    """
    Calculate the volume of a sphere given its diameter.
    
    Formula:
        V = (pi/6) * d^3 = (4/3) * pi * (d/2)^3
    
    Args:
        diameter (float): Sphere diameter (any length unit).
    
    Returns:
        float: Sphere volume (same units cubed as diameter).
    """

    return _PI_OVER_6 * diameter**3


def packing_fraction(N: int, diameter: float, box: hoomd.Box) -> float:
    """
    Calculate the packing fraction of a hard-sphere system.
    
    Formula:
        phi = (N * V_sphere) / V_box
    
    Args:
        N (int): Number of particles.
        diameter (float): Particle diameter (same units as box dimensions).
        box (hoomd.Box): HOOMD Box object containing box dimensions.
    
    Returns:
        float: Packing fraction (dimensionless, range [0, ~0.74]).
    """

    return N * sphere_volume(diameter) / box.volume


def read_mono_diameter_from_gsd(gsd_filename: str) -> float:
    """
    Read and validate particle diameter from a monodisperse GSD file.
    
    This function:
        1. Opens the GSD file and reads the last frame
        2. Extracts all particle diameters
        3. Validates that all diameters are identical (monodisperse)
        4. Returns the uniform diameter value
    
    Args:
        gsd_filename (str): Path to GSD file.
    
    Returns:
        float: Uniform particle diameter.
    """

    # Convert to Path object for existence check and absolute path
    path = Path(gsd_filename)


    # Check file exists before attempting to open
    if not path.exists():
        sys.exit(
            f"[FATAL ERROR] Cannot read diameter; GSD file not found: {gsd_filename}\n"
            f"  Expected location: {path.absolute()}\n"
            f"  => Check file path and spelling."
        )
    
    # Attempt to open GSD file and read last frame
    try:
        with gsd.hoomd.open(name=str(path), mode="r") as trajectory:
            # Check GSD has at least one frame
            if len(trajectory) == 0:
                sys.exit(
                    f"[FATAL ERROR] GSD file has no frames: {gsd_filename}\n"
                    f"              Cannot extract particle diameter from empty file.\n"
                    f"              => Regenerate GSD with initial configuration."
                )
            
            # Read last frame (most recent snapshot)
            frame = trajectory[-1]
    
    except Exception as gsd_error:
        # Catch GSD-specific errors (corrupted file, incompatible version, etc.)
        sys.exit(
            f"[FATAL ERROR] Failed to open GSD file '{gsd_filename}': {gsd_error}\n"
        )
    
    # Extract diameter array from frame
    # frame.particles.diameter is a NumPy array of shape (N,)
    diameters = [float(d) for d in frame.particles.diameter]

    # Validation 1: Check diameters array is not empty
    if not diameters:
        sys.exit(
            f"[FATAL ERROR] No particle diameters stored in GSD file: {gsd_filename}\n"
            f"              The file may be missing the 'particles/diameter' dataset.\n"
            f"              => Regenerate GSD with diameter information included."
        )
    
    # Validation 2: Check all diameters are positive
    if any(d <= 0 for d in diameters):
        sys.exit(
            f"[FATAL ERROR] Non-positive particle diameter found in GSD file: {gsd_filename}\n"
            f"              All particle diameters must be > 0.\n"
            f"              Found values: min={min(diameters)}, max={max(diameters)}\n"
            f"              => Check GSD generation script for errors."
        )
    
    # Validation 3: Check all diameters are equal (monodisperse)
    # Use first diameter as reference
    reference_diameter = diameters[0]
    
    # Check if any diameter differs from reference by more than tolerance
    # Tolerance: 1e-12 allows for floating-point roundoff 
    if any(abs(d - reference_diameter) > 1e-12 for d in diameters[1:]):
        # Polydisperse system detected
        unique_diameters = sorted(set(diameters))
        sys.exit(
            f"[FATAL ERROR] Multiple particle diameters detected in GSD file: {gsd_filename}\n"
            f"              This script requires a monodisperse system (all diameters equal).\n"
            f"              Found {len(unique_diameters)} unique diameter values:\n" +
            "\n".join(f"                {d:.15f}" for d in unique_diameters[:10]) +
            (f"\n                ... ({len(unique_diameters) - 10} more)" if len(unique_diameters) > 10 else "") +
            "\n              Solutions:\n"
            f"                1. Regenerate GSD with uniform diameter\n"
            f"                2. Use polydisperse compression script (not implemented)\n"
            f"                3. Modify this script to handle polydispersity"
        )
    
    # All validations passed: return uniform diameter
    return reference_diameter
    


# ===========================================================================
#  SECTION 7: BOX RESIZING (ISOTROPIC COMPRESSION)
# ===========================================================================
#
# BOX RESIZING IN HOOMD V4:
# ------------------------
# HOOMD v4.9 provides hoomd.update.BoxResize.update() as a STATIC method
# for instantaneous affine box rescaling.
#
# AFFINE SCALING:
# --------------
# When the box is resized, all particle positions are scaled proportionally:
#
#     x_new = x_old * (L_new / L_old)
#
# This preserves the relative positions and prevents particles from
# overlapping solely due to rescaling. However, the increased density
# may create new overlaps that must be removed via HPMC.
#
# ISOTROPIC SCALING:
# -----------------
# This function scales Lx, Ly, Lz uniformly (same scale factor for all).
# The box aspect ratio is preserved:
#
#     Lx/Ly/Lz before = Lx/Ly/Lz after
#
# Tilt factors (xy, xz, yz) are unchanged, preserving box shape.
#
# SCALE FACTOR DERIVATION:
# -----------------------
# For isotropic scaling:
#     V_new = Lx_new * Ly_new * Lz_new
#           = (s * Lx) * (s * Ly) * (s * Lz)
#           = s^3 * (Lx * Ly * Lz)
#           = s^3 * V_old
#
# Solving for scale factor s:
#     s = (V_new / V_old)^(1/3)
#
# ===========================================================================

def resize_box_to_volume(sim: hoomd.Simulation, new_volume: float) -> hoomd.Box:
    """
    Resize simulation box isotropically to a target volume.
    
    This function performs an instantaneous affine box rescaling operation:
        1. Calculate scale factor s from volume ratio
        2. Scale Lx, Ly, Lz by factor s (isotropic)
        3. Preserve tilt factors (xy, xz, yz)
        4. Apply rescaling via BoxResize.update() static method
        5. Particle positions automatically scaled (affine transformation)
    
    Args:
        sim (hoomd.Simulation): Active simulation object with loaded state.
        new_volume (float): Target box volume (same units³ as current).
    
    Returns:
        hoomd.Box: The new box object after rescaling.

    Notes:
        - Equivalent to v2's box_resize(..., period=None)
        - Rescaling is collective: all MPI ranks apply simultaneously
        - After rescaling, mc.overlaps may increase (overlaps created)
        - Rescaling does not advance timestep (sim.timestep unchanged)
    
    Error handling:
        - No explicit error handling (HOOMD will raise if new_volume invalid)
        - new_volume <= 0 => hoomd.BoxError
        - new_volume too large => OverflowError (box dimensions exceed float range)
    """
    
    try:
        # Retrieve current box from simulation state
        current_box = sim.state.box
        
        # Calculate isotropic scale factor
        # s = (V_new / V_old)^(1/3)
        scale_factor = (new_volume / current_box.volume) ** (1.0 / 3.0)
        
        # Construct new box with scaled dimensions
        # Tilt factors (xy, xz, yz) preserved unchanged
        new_box = hoomd.Box(
            Lx=current_box.Lx * scale_factor,  # Scale x dimension
            Ly=current_box.Ly * scale_factor,  # Scale y dimension
            Lz=current_box.Lz * scale_factor,  # Scale z dimension
            xy=current_box.xy,                  # Preserve xy tilt
            xz=current_box.xz,                  # Preserve xz tilt
            yz=current_box.yz,                  # Preserve yz tilt
        )
        
        # Apply box resize instantly using HOOMD v4.9 static method
        # This is a collective operation: all MPI ranks execute
        hoomd.update.BoxResize.update(
            state=sim.state,              # Simulation state to update
            box=new_box,                  # New box dimensions
            filter=hoomd.filter.All(),    # Apply to all particles
        )
        
        # Return new box for reference (also available via sim.state.box)
        return new_box
    
    except AttributeError as attr_error:
        # Catch if BoxResize.update() doesn't exist (HOOMD version mismatch)
        sys.exit(
            f"[FATAL ERROR] BoxResize.update() method not found: {attr_error}\n"
        )
    
    except ZeroDivisionError:
        # Catch if current_box.volume == 0 (should never happen)
        sys.exit(
            f"[FATAL ERROR] Current box volume is zero.\n"
            f"              Cannot compute scale factor (division by zero).\n"
            f"              => Check that GSD file has valid box dimensions."
        )
    
    except (OverflowError, ValueError) as numeric_error:
        # Catch if new_volume is extreme (overflow, NaN, negative)
        sys.exit(
            f"[FATAL ERROR] Box resizing failed: {numeric_error}\n"
            f"              Current volume: {current_box.volume}\n"
            f"              Target volume:  {new_volume}\n"
            f"              => Check volume_scaling_factor and target_pf in JSON."
        )
    
    except Exception as unexpected_error:
        # Catch-all for unexpected HOOMD errors
        sys.exit(
            f"[FATAL ERROR] Unexpected error during box resize: {unexpected_error}\n"
            f"              Current volume: {current_box.volume}\n"
            f"              Target volume:  {new_volume}"
        )





# ===========================================================================
#  SECTION 8: HOOMD SIMULATION BUILDER
# ===========================================================================
#
# This section constructs and fully configures the HOOMD Simulation object,
# including:
#   - Device selection (CPU/GPU)
#   - State initialization (fresh vs restart)
#   - HPMC integrator configuration
#   - Particle shape definition
#   - GSD writers (trajectory, restart)
#   - Console logger (Table writer)
#   - Custom loggable quantities (phi, overlaps, acceptance)
#
# CONFIGURATION PHILOSOPHY:
# ------------------------
# All simulation configuration is performed in this function, before any
# sim.run() calls. 
#
# ===========================================================================

def build_simulation(
    params: SimulationParams,
    files: RunFiles,
    seed: int,
) -> tuple[hoomd.Simulation, hoomd.hpmc.integrate.Sphere]:
    """
    Build and fully configure a HOOMD-blue v4 hard-sphere compression simulation.
    
    Args:
        params (SimulationParams): Validated simulation parameters.
        files (RunFiles): Resolved filenames for I/O operations.
        seed (int): Random seed for HOOMD RNG (0-65535).
    
    Returns:
        tuple[hoomd.Simulation, hoomd.hpmc.integrate.Sphere]:
            - sim: Fully configured Simulation object, ready for sim.run()
            - mc: HPMC Sphere integrator, provides overlaps and move stats
    
    Side effects:
        - Prints status messages to console (rank 0 only)
        - Attaches writers to sim.operations (GSD, Table)
        - Sets sim.state to loaded configuration
    
    Notes:
        - All ranks execute this function (collective operations)
        - Rank 0 performs I/O (prints, file checks)
        - Seed must be identical across all ranks (synchronization)
        - Diameter read from GSD, not from JSON (consistency)
    """    

    # -----------------------------------------------------------------------
    # STEP 1: Device selection (GPU or CPU)
    # -----------------------------------------------------------------------
    # User specifies device preference in JSON: use_gpu=true/false
    # If GPU requested but unavailable, fall back to CPU with warning
    
    root_print("\n[BUILD] Step 1/9: Selecting compute device...")
    
    try:
        if params.use_gpu:
            # Attempt GPU initialization
            try:
                device = hoomd.device.GPU(gpu_id=params.gpu_id)
                root_print(f"[BUILD] Device: GPU {params.gpu_id} (CUDA)")
                
            except Exception as gpu_error:
                # GPU init failed (no CUDA, wrong GPU ID, driver issue, etc.)
                # Print warning and fall back to CPU
                root_print(
                    f"[WARNING] GPU initialization failed: {gpu_error}\n"
                    f"          Falling back to CPU device."
                )
                device = hoomd.device.CPU()
                root_print("[BUILD] Device: CPU (fallback)")
        
        else:
            # User requested CPU explicitly
            device = hoomd.device.CPU()
            root_print("[BUILD] Device: CPU")
    
    except Exception as device_error:
        # Unexpected error during device selection
        sys.exit(
            f"[FATAL ERROR] Device selection failed: {device_error}\n"
            f"              => Check HOOMD installation and hardware availability."
        )


    # -----------------------------------------------------------------------
    # STEP 2: Create Simulation object
    # -----------------------------------------------------------------------
    
    root_print("\n[BUILD] Step 2/9: Creating Simulation object...")
    
    try:
        sim = hoomd.Simulation(device=device, seed=seed)
        root_print(f"[BUILD] Simulation created (seed={seed})")
    
    except Exception as sim_error:
        sys.exit(
            f"[FATAL ERROR] Failed to create Simulation: {sim_error}\n"
            f"              Seed: {seed}, Device: {device}"
        )
    
    # -----------------------------------------------------------------------
    # STEP 3: Load state (restart-aware logic)
    # -----------------------------------------------------------------------
    root_print("\n[BUILD] Step 3/9: Loading particle state...")
    
    restart_exists = Path(files.restart_gsd).exists()
    final_exists   = Path(files.final_gsd).exists()

    try:
        if restart_exists and not final_exists:
            # RESTART mode: Resume from checkpoint
            root_print(
                f"[BUILD] Restart GSD found: '{files.restart_gsd}'\n"
                f"[BUILD] Resuming from checkpoint..."
            )
            sim.create_state_from_gsd(filename=files.restart_gsd)
            state_source_gsd = files.restart_gsd
            restart_run_flag = True
        
        else:
            # FRESH RUN mode: Load from input GSD
            if final_exists:
                root_print(
                    f"[WARNING] Final GSD '{files.final_gsd}' already exists.\n"
                    f"          Starting fresh run (will be overwritten at completion)."
                )
            
            root_print(f"[BUILD] Starting fresh from: '{files.input_gsd}'")
            
            # Load from input GSD, frame 0 (first frame)
            # timestep will be set to params.initial_timestep
            sim.create_state_from_gsd(
                filename=files.input_gsd,
                frame=0  # Use first frame (for multi-frame GSDs)
            )
            state_source_gsd = files.input_gsd
            restart_run_flag = False
    
    except Exception as gsd_load_error:
        # Catch GSD loading errors (corrupted file, incompatible version, etc.)
        sys.exit(
            f"[FATAL ERROR] Failed to load GSD file: {gsd_load_error}\n"
            f"              Attempted to load: {files.restart_gsd if restart_exists and not final_exists else files.input_gsd}\n"
        )


    # -----------------------------------------------------------------------
    # STEP 4: Read particle diameter from GSD
    # -----------------------------------------------------------------------
    # Extract diameter from the loaded GSD file (not from JSON)
    # This ensures consistency between initial config and simulation parameters
    
    root_print("\n[BUILD] Step 4/9: Reading particle diameter...")

    try:
        params.diameter = read_mono_diameter_from_gsd(state_source_gsd)
        root_print(f"[BUILD] Particle diameter: {params.diameter}")
    
    except SystemExit:
        raise

    N = sim.state.N_particles

    if N == 0:
        sys.exit(
            f"[FATAL ERROR] Loaded GSD contains zero particles.\n"
            f"              Source: {state_source_gsd}\n"
            f"              => Regenerate GSD with valid initial configuration."
        )
    
    root_print(f"[BUILD] Number of particles: {N}")

    # Calculate and display initial packing fraction
    phi_initial = packing_fraction(N, params.diameter, sim.state.box)
    root_print(
        f"[BUILD] Initial box: "
        f"Lx={sim.state.box.Lx:.4f}, Ly={sim.state.box.Ly:.4f}, Lz={sim.state.box.Lz:.4f}"
    )
    root_print(
        f"[BUILD] Packing fractions: "
        f"phi_initial={phi_initial:.6f}, phi_target={params.target_pf:.6f}"
    )
    root_print(f"[BUILD] Restart run: {restart_run_flag}")

    # ------------------------------------------------------------------
    #  HPMC Sphere integrator  (FIXED move size — no MoveSize tuner)
    # ------------------------------------------------------------------

    root_print("\n[BUILD] Step 6/9: Configuring HPMC Sphere integrator...")
    
    try:
        mc = hoomd.hpmc.integrate.Sphere(
            default_d=params.move_size_translation,  # Translational move size (fixed)
            nselect=1,  # Number of trial moves per particle per sweep (default: 1)
        )
        
        # Define particle shape for type 'A'
        # For spheres, only diameter is needed (no orientation)
        mc.shape["A"] = {"diameter": params.diameter}
        
        # Attach integrator to simulation
        sim.operations.integrator = mc
        
        root_print(
            f"[BUILD] HPMC Sphere configured:\n"
            f"        - Particle diameter: {params.diameter}\n"
            f"        - Move size (d): {params.move_size_translation} (FIXED)\n"
            f"        - Trials per sweep: {N} (nselect=1)"
        )
    
    except Exception as hpmc_error:
        sys.exit(
            f"[FATAL ERROR] HPMC integrator configuration failed: {hpmc_error}\n"
            f"              => Check HOOMD version and parameter values."
        )

    # ------------------------------------------------------------------
    # 7.4  Logger (GSD + Table)
    # ------------------------------------------------------------------
    # IMPORTANT — DataAccessError guard
    # ----------------------------------
    # HOOMD's Logger.__setitem__ validates custom loggables by calling
    # hasattr(obj, method_name), which *executes* the property getter to
    # check it does not raise AttributeError.  Properties that read
    # mc.overlaps or mc.translate_moves raise hoomd.error.DataAccessError
    # before the first sim.run() call because the HPMC counters are not
    # yet initialised.  Every property that reads an HPMC quantity must
    # therefore catch DataAccessError and return a safe sentinel (0.0).

    # Custom loggable: live packing fraction (scalar)
    class _PhiLogger:
        """Logger for live packing fraction calculation."""
        
        # HOOMD logger export specification
        # Dict format: {property_name: (category, loggable_flag)}

        _export_dict = {"packing_fraction": ("scalar", True)}

        def __init__(self, sim_ref, n, d):
            """
            Args:
                sim_ref: Reference to Simulation object
                n: Number of particles (constant)
                d: Particle diameter (constant)
            """
            self._sim = sim_ref; self._N = n; self._d = d

        @property
        def packing_fraction(self) -> float:
            # sim.state.box is always available once state is loaded.
            return packing_fraction(self._N, self._d, self._sim.state.box)

    # Custom loggable: overlap count (scalar)
    class _OverlapLogger:
        """Logger for current overlap count from HPMC integrator."""

        _export_dict = {"overlap_count": ("scalar", True)}

        def __init__(self, mc_ref):
            """
            Args:
                mc_ref: Reference to HPMC integrator
            """
            self._mc = mc_ref

        @property
        def overlap_count(self) -> float:
            # mc.overlaps raises DataAccessError before the first sim.run().
            # Return 0.0 as a pre-run sentinel; real values follow thereafter.
            try:
                return float(self._mc.overlaps)
            except hoomd.error.DataAccessError:
                return 0.0

    # Custom loggable: translational acceptance ratio (scalar)
    class _AcceptanceLogger:
        """Logger for translational move acceptance rate."""

        _export_dict = {"translate_acceptance": ("scalar", True)}

        def __init__(self, mc_ref):
            """
            Args:
                mc_ref: Reference to HPMC integrator
            """
            self._mc = mc_ref

        @property
        def translate_acceptance(self) -> float:
            # mc.translate_moves also raises DataAccessError before sim.run().
            try:
                moves = self._mc.translate_moves  # (accepted, rejected)  tuple
                total = moves[0] + moves[1]
                return float(moves[0]) / float(total) if total > 0 else 0.0
            except hoomd.error.DataAccessError:
                # HPMC counters not initialized yet
                return 0.0

    # Instantiate custom loggers
    try:
        phi_logger        = _PhiLogger(sim, N, params.diameter)
        overlap_logger    = _OverlapLogger(mc)
        acceptance_logger = _AcceptanceLogger(mc)

        root_print(
            f"[BUILD] Custom loggers created:\n"
            f"        - packing_fraction (live phi calculation)\n"
            f"        - overlap_count (from mc.overlaps)\n"
            f"        - translate_acceptance (from mc.translate_moves)"
        )        

    except Exception as logger_error:
        sys.exit(
            f"[FATAL ERROR] Failed to create custom loggers: {logger_error}"
        )

    # -----------------------------------------------------------------------
    # STEP 8: Create GSD logger and attach logged quantities
    # -----------------------------------------------------------------------
    # GSD logger records quantities to GSD trajectory file
    # Categories: scalar (float/int), string, sequence (arrays)
    
    root_print("\n[BUILD] Step 8/9: Configuring GSD and console loggers...")

    try:
        # Create logger for GSD trajectory file
        gsd_logger = hoomd.logging.Logger(categories=["scalar", "string", "sequence"])
        
        # Add built-in HOOMD quantities
        gsd_logger.add(sim, quantities=["timestep", "tps", "walltime"])
        
        # Add HPMC integrator quantities
        gsd_logger.add(mc, quantities=["type_shapes", "translate_moves"])
        
        # Add custom quantities with namespace "compression"
        # Format: logger[(namespace, quantity_name)] = (object, property, category)
        gsd_logger[("compression", "packing_fraction")]     = (phi_logger, "packing_fraction", "scalar")
        gsd_logger[("compression", "overlap_count")]        = (overlap_logger, "overlap_count", "scalar")
        gsd_logger[("compression", "translate_acceptance")] = (acceptance_logger, "translate_acceptance", "scalar")
        
        root_print(
            f"[BUILD] GSD logger configured:\n"
            f"        - Built-in: timestep, tps, walltime\n"
            f"        - HPMC: type_shapes, translate_moves\n"
            f"        - Custom: packing_fraction, overlap_count, translate_acceptance"
        )
    
    except Exception as logger_config_error:
        sys.exit(
            f"[FATAL ERROR] Failed to configure GSD logger: {logger_config_error}\n"
            f"              => Check custom logger definitions and property names."
        ) 

    # -----------------------------------------------------------------------
    # STEP 9a: GSD trajectory writer (append mode)
    # -----------------------------------------------------------------------
    
    try:
        gsd_traj = hoomd.write.GSD(
            trigger=hoomd.trigger.Periodic(params.traj_out_freq),  # Write every N steps
            filename=files.traj_gsd,
            mode="ab",                # Append mode (restarts don't overwrite)
            filter=hoomd.filter.All(),  # Write all particles
            dynamic=["property", "attribute"],  # Log dynamic quantities
            logger=gsd_logger,          # Attach logger
        )
        sim.operations.writers.append(gsd_traj)
        
        root_print(
            f"[BUILD] GSD trajectory writer attached:\n"
            f"        - File: {files.traj_gsd}\n"
            f"        - Frequency: every {params.traj_out_freq} timesteps\n"
            f"        - Mode: append (restarts continue existing file)"
        )
    
    except Exception as traj_writer_error:
        sys.exit(
            f"[FATAL ERROR] Failed to create GSD trajectory writer: {traj_writer_error}\n"
            f"              File: {files.traj_gsd}\n"
        )
    
    # -----------------------------------------------------------------------
    # STEP 9b: GSD restart writer (truncate mode)
    # -----------------------------------------------------------------------
    # Writes single-frame checkpoint for restart capability
    # Mode "wb" (write binary), truncate=True: Overwrites file each write
    
    try:
        gsd_restart = hoomd.write.GSD(
            trigger=hoomd.trigger.Periodic(params.restart_frequency),  # Write every N steps
            filename=files.restart_gsd,
            mode="wb",                  # Write mode (overwrite each time)
            filter=hoomd.filter.All(),  # Write all particles
            truncate=True,              # Overwrite file (single-frame checkpoint)
            dynamic=["property", "attribute"],  # Log dynamic quantities
        )
        sim.operations.writers.append(gsd_restart)
        
        root_print(
            f"[BUILD] GSD restart writer attached:\n"
            f"        - File: {files.restart_gsd}\n"
            f"        - Frequency: every {params.restart_frequency} timesteps\n"
            f"        - Mode: truncate (single-frame checkpoint)"
        )
    
    except Exception as restart_writer_error:
        sys.exit(
            f"[FATAL ERROR] Failed to create GSD restart writer: {restart_writer_error}\n"
            f"              File: {files.restart_gsd}\n"
        )

    # -----------------------------------------------------------------------
    # STEP 9c: Console Table writer (live monitoring)
    # -----------------------------------------------------------------------
    # Prints progress table to stdout for live monitoring during simulation
    
    try:
        # Create separate logger for console output (avoids logging sequences to console)
        table_logger = hoomd.logging.Logger(categories=["scalar", "string"])
        
        # Add quantities to console logger
        table_logger.add(sim, quantities=["timestep", "tps"])
        table_logger.add(mc, quantities=["translate_moves"])
        table_logger[("compression", "packing_fraction")]     = (phi_logger, "packing_fraction", "scalar")
        table_logger[("compression", "overlap_count")]        = (overlap_logger, "overlap_count", "scalar")
        table_logger[("compression", "translate_acceptance")] = (acceptance_logger, "translate_acceptance", "scalar")
        
        # Create Table writer
        table = hoomd.write.Table(
            trigger=hoomd.trigger.Periodic(params.log_frequency),  # Print every N steps
            logger=table_logger,
        )
        sim.operations.writers.append(table)
        
        root_print(
            f"[BUILD] Console Table writer attached:\n"
            f"        - Frequency: every {params.log_frequency} timesteps\n"
            f"        - Columns: timestep, tps, translate_moves, phi, overlaps, acceptance"
        )
    
    except Exception as table_writer_error:
        sys.exit(
            f"[FATAL ERROR] Failed to create Table writer: {table_writer_error}\n"
        )
    
    # -----------------------------------------------------------------------
    # Build complete: Print summary banner and return
    # -----------------------------------------------------------------------
    _print_banner("Simulation build complete — ready for compression", width=80)
    root_flush_stdout()
    
    return sim, mc




# ===========================================================================
#  SECTION 9: COMPRESSION ENGINE (MANUAL TWO-LEVEL LOOP)
# ===========================================================================
#
# ALGORITHM OVERVIEW:
# ------------------
# The compression proceeds via a two-level nested loop:
#
#   OUTER LOOP: while current_volume > target_volume
#       1. Shrink box by volume_scaling_factor (typically 0.99 = 1% reduction)
#       2. INNER LOOP: while overlaps > 0
#           a. Run HPMC for run_length_to_remove_overlap steps
#           b. Check overlap count
#           c. Continue until ZERO overlaps achieved
#       3. Run fixed-box equilibration for run_length_to_relax steps
#       4. Write current_pf.json checkpoint
#       5. Write output_current_pf.gsd snapshot
#
# KEY INVARIANTS:
# --------------
# 1. After each inner loop: mc.overlaps == 0 (hard-sphere constraint)
# 2. After each outer step: phi increases by factor ~ (1 / volume_scaling_factor)^(1/3)
# 3. Box dimensions scale isotropically: Lx/Ly/Lz ratio preserved
#
# CONVERGENCE:
# -----------
# Outer loop terminates when sim.state.box.volume <= target_volume
# At this point, phi_final >= phi_target (may slightly overshoot)
#
# FAILURE MODES:
# -------------
# - Inner loop infinite: Overlaps never reach zero (see troubleshooting section)
# - Kinetic arrest: System jams before reaching target (phi > ~0.58)
# - Numerical overflow: Extreme parameters cause box dimensions to explode
#
# ===========================================================================

def run_compression(
    sim: hoomd.Simulation,
    mc: hoomd.hpmc.integrate.Sphere,
    params: SimulationParams,
    files: RunFiles,
) -> bool:
    """
    Execute the two-level compression loop to reach target packing fraction.
    
    This function implements the exact v2 compression methodology:
        OUTER WHILE: volume_current > volume_target
            new_volume = max(volume_current * volume_scaling_factor, volume_target)
            resize_box(new_volume)
            INNER WHILE: overlaps > 0
                sim.run(run_length_to_remove_overlap)
                overlaps = mc.overlaps
            sim.run(run_length_to_relax)
            write current_pf.json
            write output_current_pf.gsd
    
    Args:
        sim (hoomd.Simulation): Fully configured simulation with loaded state.
        mc (hoomd.hpmc.integrate.Sphere): HPMC integrator attached to sim.
        params (SimulationParams): Simulation parameters (validated).
        files (RunFiles): Resolved filenames for I/O operations.
    
    Returns:
        bool: True if compression completed successfully, False otherwise.
              (In practice, always returns True or raises exception)
    
    Algorithm walkthrough:
        1. Calculate target volume from target_pf
        2. Check if already at target (skip compression if yes)
        3. Run initial 100-step equilibration
        4. OUTER LOOP: compress in volume_scaling_factor increments
            a. Resize box to new volume
            b. INNER LOOP: remove overlaps via HPMC
            c. Fixed-box equilibration
            d. Write checkpoint files
        5. Return True (success)
    
    
    Notes:
        - All ranks execute this function (collective sim.run() calls)
        - Only rank 0 writes files and prints progress
        - Inner loop overlap checks are local (no MPI communication)
        - If interrupted (SIGTERM), restart GSD allows resumption
    """

    # Extract frequently-used parameters
    N          = sim.state.N_particles
    diameter   = params.diameter
    vsf        = params.volume_scaling_factor
    pf_target  = params.target_pf

    # -----------------------------------------------------------------------
    # Calculate initial and target volumes
    # -----------------------------------------------------------------------

    try:
        volume_target = N * sphere_volume(diameter) / pf_target
    except ZeroDivisionError:
        sys.exit(
            f"[FATAL ERROR] target_pf is zero, cannot compute target volume.\n"
            f"              => Check JSON parameter file."
        )
    except (OverflowError, ValueError) as vol_error:
        sys.exit(
            f"[FATAL ERROR] Failed to compute target volume: {vol_error}\n"
            f"              N={N}, diameter={diameter}, target_pf={pf_target}\n"
            f"              => Check that parameters are reasonable."
        )


    box_initial  = sim.state.box
    phi_initial  = packing_fraction(N, diameter, box_initial)
    volume_current = box_initial.volume

    # Print initial state
    root_print(
        f"\n{'='*80}\n"
        f"COMPRESSION PARAMETERS\n"
        f"{'='*80}"
    )
    root_print(
        f"  Initial phi:      {phi_initial:.6f}\n"
        f"  Target phi:       {pf_target:.6f}\n"
        f"  Initial volume:   {volume_current:.4f}\n"
        f"  Target volume:    {volume_target:.4f}\n"
        f"  Volume scaling:   {vsf} ({(1-vsf)*100:.1f}% reduction per step)\n"
        f"  Overlap removal:  {params.run_length_to_remove_overlap} steps per attempt\n"
        f"  Relaxation:       {params.run_length_to_relax} steps per compression\n"
        f"  Move size:        {params.move_size_translation} (fixed)"
    )
    root_print(f"{'='*80}\n")
    root_flush_stdout()


    # -----------------------------------------------------------------------
    # Early termination check
    # -----------------------------------------------------------------------
    # If system is already at or beyond target density, skip compression
    # This can happen if:
    #   - User provides pre-compressed initial state
    #   - Restart from near-target state
    #   - Multi-stage run where stage N-1 reached target

    if volume_current <= volume_target:
        root_print(
            "[INFO] System is already at or beyond target packing fraction.\n"
            "[INFO] Current volume <= target volume. Skipping compression loop.\n"
            "[INFO] No compression steps needed."
        )
        root_flush_stdout()
        return True

    # -----------------------------------------------------------------------
    # Initial equilibration run
    # -----------------------------------------------------------------------
    # Purpose: Initialize HPMC counters, check for overlaps in initial state
    
    root_print("[INFO] Initial equilibration: 100 timesteps")
    root_flush_stdout()

    try:
        sim.run(100)
    except Exception as run_error:
        sys.exit(
            f"[FATAL ERROR] Initial equilibration run failed: {run_error}\n"
            f"              => Check HPMC integrator configuration."
        )
    
    # Check for overlaps in initial state
    try:
        initial_overlaps = mc.overlaps
    except hoomd.error.DataAccessError:
        # Should not happen after sim.run(), but catch anyway
        initial_overlaps = 0
    
    if initial_overlaps > 0:
        root_print(
            f"[WARNING] Initial configuration has {initial_overlaps} overlaps.\n"
            f"          This may indicate a problem with the input GSD.\n"
            f"          Compression will attempt to remove these overlaps."
        )
    else:
        root_print(f"[INFO] Initial configuration: 0 overlaps (valid hard-sphere state)")
    
    root_flush_stdout()
    
    # -----------------------------------------------------------------------
    # OUTER COMPRESSION LOOP
    # -----------------------------------------------------------------------
    # Continue until box volume <= target volume
    # Each iteration: shrink box, remove overlaps, equilibrate, checkpoint
    
    root_print(f"\n{'='*80}")
    root_print("BEGINNING COMPRESSION LOOP")
    root_print(f"{'='*80}\n")
    root_flush_stdout()

    # ---- Outer loop ---------------------------------------------------------
    outer_step = 0     # Counter for compression steps

    try:
        while sim.state.box.volume > volume_target:

            # ---------------------------------------------------------------
            # OUTER STEP: Compute new volume and resize box
            # ---------------------------------------------------------------
            volume_current = sim.state.box.volume

            # New volume: shrink by vsf, but don't go below target
            new_volume = max(volume_current * vsf, volume_target)

            # Resize box instantly (affine scaling of particle positions)
            try:
                new_box = resize_box_to_volume(sim, new_volume)
            except SystemExit:
                # resize_box_to_volume already printed detailed error
                raise
            except Exception as resize_error:
                sys.exit(
                    f"[FATAL ERROR] Box resize failed at outer step {outer_step}: "
                    f"{resize_error}\n"
                    f"              Current volume: {volume_current}\n"
                    f"              Target volume:  {new_volume}"
                )

            phi_current = packing_fraction(N, diameter, sim.state.box)
            outer_step += 1

            # Count overlaps after resize
            overlaps = mc.overlaps

            # Print compression step header
            root_print(
                f"{'─'*80}\n"
                f"[COMPRESSION STEP {outer_step}]\n"
                f"  Packing fraction: {phi_current:.6f}\n"
                f"  Box volume:       {sim.state.box.volume:.4f}\n"
                f"  Overlaps:         {overlaps}\n"
                f"{'─'*80}"
            )
            root_flush_stdout()

            # ---------------------------------------------------------------
            # INNER LOOP: Remove overlaps via HPMC
            # ---------------------------------------------------------------
            # Continue running HPMC until mc.overlaps == 0
            # Each iteration: run run_length_to_remove_overlap steps, check overlaps
            #
            
            inner_count = 0       # Counter for inner loop iterations

            if overlaps > 0:
                root_print(
                    f"[INNER LOOP] Removing overlaps: ",
                    end=""
                )
                root_flush_stdout()
            
            while overlaps > 0:
                # Run HPMC for overlap removal
                try:
                    sim.run(params.run_length_to_remove_overlap)
                except Exception as run_error:
                    sys.exit(
                        f"\n[FATAL ERROR] HPMC run failed during overlap removal:\n"
                        f"              {run_error}\n"
                        f"              Outer step: {outer_step}, Inner iteration: {inner_count}\n"
                        f"              Current overlaps: {overlaps}\n"
                        f"              => Check system state and parameters."
                    )
                            
                overlaps = mc.overlaps
                
                inner_count += 1
                
                # Print overlap count (matches v2: "print(overlaps, end=' ')")
                root_print(f"{overlaps} ", end="")
                root_flush_stdout()
                
                # Safety check: Detect runaway inner loop
                # If inner loop runs for >50000 iterations, something is wrong
                if inner_count > 50000:
                    root_print(
                        f"\n[ERROR] Inner loop exceeded 50000 iterations without "
                        f"removing all overlaps.\n"
                        f"        Current overlaps: {overlaps}\n"
                        f"        This indicates a problem with compression parameters.\n"
                        f"        Possible causes:\n"
                        f"          - volume_scaling_factor too aggressive \n"
                        f"          - move_size_translation too large \n"
                        f"          - System entering jammed state \n"
                        f"          - run_length_to_remove_overlap too small\n"
                        f"        Current state will be saved as emergency snapshot."
                    )
                    
                    # Write emergency snapshot
                    try:
                        emergency_file = f"emergency_step{outer_step}_{params.tag}.gsd"
                        _write_snapshot(sim, emergency_file)
                        root_print(f"[ERROR] Emergency snapshot: {emergency_file}")
                    except Exception:
                        root_print("[ERROR] Failed to write emergency snapshot.")
                    
                    sys.exit("[FATAL ERROR] Compression failed (runaway inner loop).")
                
                # Inner loop complete: overlaps == 0
                if inner_count > 0:
                    root_print(
                        f"\n[INNER LOOP] Overlaps removed in {inner_count} iterations "
                        f"({inner_count * params.run_length_to_remove_overlap} total steps)"
                    )
                else:
                    root_print("[INNER LOOP] No overlaps after resize (0 iterations needed)")
                
                root_flush_stdout()

            # ---------------------------------------------------------------
            # FIXED-BOX EQUILIBRATION
            # ---------------------------------------------------------------
            # Run additional HPMC steps at constant volume to equilibrate

            if params.run_length_to_relax > 0:
                root_print(
                    f"[EQUILIBRATION] Running {params.run_length_to_relax} steps "
                    f"at phi = {phi_current:.6f}"
                )
                root_flush_stdout()
                
                try:
                    sim.run(params.run_length_to_relax)
                except Exception as run_error:
                    sys.exit(
                        f"[FATAL ERROR] Equilibration run failed: {run_error}\n"
                        f"              Outer step: {outer_step}\n"
                        f"              => Check system state."
                    )
                
                root_print("[EQUILIBRATION] Complete")
            else:
                root_print("[EQUILIBRATION] Skipped (run_length_to_relax = 0)")
            
            root_flush_stdout()

            # ---------------------------------------------------------------
            # CHECKPOINT OUTPUTS
            # ---------------------------------------------------------------
            # Write current_pf.json and output_current_pf.gsd
            # These allow monitoring compression progress and resuming mid-compression
            
            # Recalculate phi after equilibration (may have changed slightly)
            phi_after_relax = packing_fraction(N, diameter, sim.state.box)
            phi_rounded = f"{phi_after_relax:.4f}"


            # Write current_pf.json (rank 0 only)
            # We add extra metadata for debugging
            if _is_root_rank():
                try:
                    with open("current_pf.json", mode='w', encoding='utf-8') as json_file:
                        json.dump(
                            {
                                "current_pf":     phi_rounded_str,
                                "outer_step":     outer_step,
                                "timestep":       sim.timestep,
                                "overlaps":       mc.overlaps,
                                "inner_iters":    inner_count,
                                "box_volume":     sim.state.box.volume,
                                "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
                            },
                            json_file,
                            indent=4
                        )
                    root_print(f"[CHECKPOINT] current_pf.json written (phi = {phi_rounded_str})")
                
                except Exception as json_error:
                    root_print(
                        f"[WARNING] Failed to write current_pf.json: {json_error}\n"
                        f"          Compression will continue without checkpoint file."
                    )
            
            # Write output_current_pf.gsd (rank 0 only)
            # v2: dump.gsd(filename='output_current_pf.gsd', ..., overwrite=True)
            # Single-frame GSD, overwritten each compression step
            try:
                _write_snapshot(sim, "output_current_pf.gsd")
                root_print(f"[CHECKPOINT] output_current_pf.gsd written")
            
            except Exception as gsd_error:
                root_print(
                    f"[WARNING] Failed to write output_current_pf.gsd: {gsd_error}\n"
                    f"          Compression will continue without snapshot checkpoint."
                )
            
            root_flush_stdout()
        
        # END OF OUTER LOOP
    
    except KeyboardInterrupt:
        # User interrupted with Ctrl+C
        root_print(
            f"\n[INTERRUPTED] Compression interrupted by user at timestep {sim.timestep}.\n"
            f"              Current phi: {packing_fraction(N, diameter, sim.state.box):.6f}\n"
            f"              Restart GSD: {files.restart_gsd} (if periodic save occurred)"
        )
        raise  # Re-raise to propagate to main()
    
    except Exception as unexpected_error:
        # Unexpected error during compression loop
        root_print(
            f"\n[FATAL ERROR] Unexpected error in compression loop: {unexpected_error}"
        )
        raise  # Re-raise to propagate to main()
    
    # -----------------------------------------------------------------------
    # COMPRESSION COMPLETE
    # -----------------------------------------------------------------------

    phi_final = packing_fraction(N, diameter, sim.state.box)

    final_overlaps = mc.overlaps

    root_print(
        f"\n{'='*80}\n"
        f"COMPRESSION COMPLETE\n"
        f"{'='*80}\n"
        f"  Final timestep:      {sim.timestep}\n"
        f"  Final phi:           {phi_final:.6f}\n"
        f"  Target phi:          {pf_target:.6f}\n"
        f"  Delta phi:           {abs(phi_final - pf_target):.2e}\n"
        f"  Final overlaps:      {final_overlaps}\n"
        f"  Compression steps:   {outer_step}\n"
        f"{'='*80}\n"
    )
    root_flush_stdout()
    
    return True


# ===========================================================================
#  SECTION 10: FINAL OUTPUT WRITING
# ===========================================================================

def write_final_outputs(
    sim: hoomd.Simulation,
    mc:  hoomd.hpmc.integrate.Sphere,
    params: SimulationParams,
    files: RunFiles,
    phi_initial: float,
    phi_final: float,
    seed: int,
) -> None:
    """
    Write all end-of-run GSD snapshots and the console summary.

    Args:
        sim (hoomd.Simulation): Simulation with final state.
        mc (hoomd.hpmc.integrate.Sphere): HPMC integrator with final stats.
        params (SimulationParams): Simulation parameters.
        files (RunFiles): Resolved filenames.
        phi_initial (float): Initial packing fraction.
        phi_final (float): Final packing fraction.
        seed (int): Random seed used for this run.
    
    Returns:
        None

    """

    N = sim.state.N_particles

    # v2: final_compressed_filename = tag + "_compressed_to_pf_" + pf_rounded + ".gsd"
    phi_rounded = f"{phi_final:.4f}".replace(".", "p")
    compressed_gsd = f"{params.tag}_compressed_to_pf_{phi_rounded}.gsd"
    _write_snapshot(sim, compressed_gsd)
    root_print(f"[OUTPUT] Labelled snapshot => {compressed_gsd}")

    # Canonical final state     
    # Standard filename for multi-stage chaining
    _write_snapshot(sim, files.final_gsd)
    root_print(f"[OUTPUT] Final GSD         => {files.final_gsd}")

    # Machine-readable summary
    summary = {
        "tag":              params.tag,
        "stage_id":         params.stage_id_current,
        "input_gsd":        files.input_gsd,
        "final_gsd":        files.final_gsd,
        "compressed_gsd":   compressed_gsd,
        "n_particles":      N,
        "diameter":         params.diameter,
        "target_pf":        params.target_pf,
        "phi_initial":      round(phi_initial, 8),
        "phi_final":        round(phi_final, 8),
        "overlaps_final":   mc.overlaps,
        "final_timestep":   sim.timestep,
        "random_seed":      seed,
        "box_final": {
            "Lx": sim.state.box.Lx,
            "Ly": sim.state.box.Ly,
            "Lz": sim.state.box.Lz,
        },
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if _is_root_rank():
        summary_file = f"{params.tag}_compression_summary.json"
        with open(summary_file, "w") as fh:
            json.dump(summary, fh, indent=2)
        root_print(f"[OUTPUT] Summary JSON      => {summary_file}")

    # -----------------------------------------------------------------------
    # Console summary banner
    # -----------------------------------------------------------------------
    # Human-readable summary of run results

    _print_banner("COMPRESSION OF A HARD SPHERE SYSTEM TO A TARGET PACKING FRACTION")

    lines = [
        f"  Simulparam file           : {_active_json_path}",
        f"  Number of particles       : {N}",
        f"  Particle diameter (sigma) : {params.diameter}",
        f"  Initial packing fraction  : {phi_initial:.6f}",
        f"  Target packing fraction   : {params.target_pf:.6f}",
        f"  Final packing fraction    : {phi_final:.6f}",
        f"  Delta(phi)                : {abs(params.target_pf - phi_final):.2e}",
        f"  Overlaps remaining        : {mc.overlaps}",
        f"  Final box Lx,Ly,Lz       : "
            f"{sim.state.box.Lx:.4f}, {sim.state.box.Ly:.4f}, {sim.state.box.Lz:.4f}",
        f"  Current stage id          : {params.stage_id_current}",
        f"  Final timestep            : {sim.timestep}",
        f"  Random seed               : {seed}",
        f"  Final GSD                 : {files.final_gsd}",
    ]
    root_print("\n".join(lines))
    _print_banner("")
    root_flush_stdout()


def _write_snapshot(sim: hoomd.Simulation, filename: str) -> None:
    """Write current state to a single-frame GSD file."""
    hoomd.write.GSD.write(
        state=sim.state, filename=filename, mode="wb",
        filter=hoomd.filter.All()
    )


# ===========================================================================
#  SECTION 11: UTILITY FUNCTIONS
# ===========================================================================

def _print_banner(title: str, width: int = 80) -> None:
    bar = "*" * width   
    root_print(f"\n{bar}")
    if title:
        root_print(f"  {title}")
        root_print(bar)


# Module-level variable to store the JSON path for the summary banner
_active_json_path: str = ""


# ===========================================================================
#  SECTION 12: MAIN ENTRY POINT
# ===========================================================================

def main() -> None:
    """
    Main entry point for the hard-sphere compression script.
    
    This function orchestrates the complete compression workflow:
        1. Parse command-line arguments
        2. Load and validate simulation parameters
        3. Resolve filenames (stage-aware)
        4. Manage random seed (rank-0 creates, all ranks read)
        5. Build HOOMD simulation (device, state, integrator, writers)
        6. Run compression loop (two-level algorithm)
        7. Write final outputs (snapshots, summary, console banner)
        8. Handle exceptions (emergency snapshots, error reporting)
    
    Command-line usage:
        python hs_compress_v10_documented.py --simulparam_file params.json
    
    Exit codes:
        0: Success (compression completed)
        1: Error (parameter validation, file not found, compression failed)
    
    Exception handling:
        - Expected exceptions (file not found, invalid params): sys.exit with message
        - Unexpected exceptions: Write emergency snapshot, print error, re-raise
        - KeyboardInterrupt (Ctrl+C): Print status, exit gracefully
    
    MPI execution:
        All ranks execute main() simultaneously. Rank 0 performs I/O (file writes,
        console output). All ranks execute collective operations (sim.run()).
    
    Notes:
        - Only rank 0 prints to console (via root_print)
        - All ranks must reach same sim.run() calls (collective operation)
        - Emergency snapshot written on unexpected exception (recovery point)
        - Seed file barrier ensures all ranks see seed before proceeding
    """
    global _active_json_path     # Store JSON path for final summary banner

    # -----------------------------------------------------------------------
    # 1. Parse command-line arguments
    # -----------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "HOOMD-blue v4 hard-sphere HPMC compression.\n"
            "Exact port of the v2 manual volume-scaling loop.\n"
            "Usage: python3 hs_compress.py --simulparam_file simulparam.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--simulparam_file", required=True, metavar="FILE",
        help="Path to the JSON file containing simulation parameters.",
    )
    try:
        args = parser.parse_args()
        _active_json_path = args.simulparam_file
    except SystemExit:
        # argparse prints error message and calls sys.exit()
        # Re-raise to propagate exit
        raise

    # Print startup banner
    _print_banner("HOOMD-blue v4 | Hard-Sphere HPMC Compression", width=80)
    root_print(f"  Parameter file: {_active_json_path}")
    root_print(f"  HOOMD version:  {hoomd.version.version}")
    root_print(f"  Start time:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    _print_banner("", width=80)
    root_flush_stdout()

    # -----------------------------------------------------------------------
    # 2. Load simulation parameters
    # -----------------------------------------------------------------------
    params = load_simulparams(args.simulparam_file)
    root_print(f"[INFO] Loaded parameters from '{args.simulparam_file}'")
    root_print(f"  tag                    = {params.tag}")
    root_print(f"  stage_id_current       = {params.stage_id_current}")
    root_print(f"  target_pf              = {params.target_pf}")
    root_print(f"  volume_scaling_factor  = {params.volume_scaling_factor}")
    root_print(f"  run_length_to_remove_overlap = {params.run_length_to_remove_overlap}")
    root_print(f"  run_length_to_relax    = {params.run_length_to_relax}")
    root_print(f"  move_size_translation  = {params.move_size_translation}")
    root_print(f"  use_gpu  = {params.use_gpu}")
    root_flush_stdout()

    # -----------------------------------------------------------------------
    # 3. Resolve filenames (stage-aware)
    # -----------------------------------------------------------------------
    try:
        files = resolve_filenames(params)
    except SystemExit:
        # resolve_filenames already printed detailed error
        raise

    root_print(f"[INFO] Input GSD  : {files.input_gsd}")
    root_print(f"[INFO] Traj GSD   : {files.traj_gsd}")
    root_print(f"[INFO] Restart GSD: {files.restart_gsd}")
    root_print(f"[INFO] Final GSD  : {files.final_gsd}")
    root_print(f"[INFO] Log file   : {files.log_txt}")
    root_flush_stdout()

    # -----------------------------------------------------------------------
    # 4. Random seed management
    # -----------------------------------------------------------------------
    # Rank 0: Create seed file if it doesn't exist
    # All ranks: Wait for seed file, then read seed value
    _mpi_rank = _mpi_rank_from_env()
    root_print(f"\n[MPI] Rank: {_mpi_rank}")
    
    try:
        # Rank 0 creates seed file (if needed)
        ensure_seed_file(params.stage_id_current, _mpi_rank)
        
        # Simple file-based barrier: Wait until seed file is readable
        # This ensures all ranks see the file before reading
        seed_file = (
            _SEED_FILE_SINGLE if params.stage_id_current == -1 else _SEED_FILE_MULTI
        )
        
        barrier_start_time = time.perf_counter()
        while not Path(seed_file).exists():
            # Timeout after 30 seconds (filesystem lag, network issues)
            if time.perf_counter() - barrier_start_time > 30:
                sys.exit(
                    f"[FATAL ERROR] Timeout waiting for seed file '{seed_file}'.\n"
                    f"              Waited 30 seconds, file not visible.\n"
                    f"              => Check filesystem synchronization (NFS, Lustre)."
                )
            time.sleep(0.1)  # Poll every 100ms
        
        # All ranks read seed
        seed = read_seed(params.stage_id_current)
        root_print(f"[SEED] Random seed: {seed}")
    
    except SystemExit:
        # Seed management functions already printed detailed errors
        raise
    
    root_flush_stdout()

    # -----------------------------------------------------------------------
    # 5. Build simulation
    # -----------------------------------------------------------------------
    try:
        sim, mc = build_simulation(params, files, seed)
    except SystemExit:
        # build_simulation already printed detailed error
        raise

    N = sim.state.N_particles
    phi_initial = packing_fraction(N, params.diameter, sim.state.box)

    root_print(
        f"\n[SIMULATION] Ready for compression:\n"
        f"             N particles:   {N}\n"
        f"             Initial phi:   {phi_initial:.6f}\n"
        f"             Target phi:    {params.target_pf:.6f}\n"
        f"             Initial steps: {sim.timestep}"
    )
    root_flush_stdout()

    # -----------------------------------------------------------------------
    # 6. Run compression with exception handling
    # -----------------------------------------------------------------------
    compression_success = False

    try:
        compression_success = run_compression(sim, mc, params, files)

    except KeyboardInterrupt:
        # User interrupted with Ctrl+C
        root_print(
            f"\n{'='*80}\n"
            f"[INTERRUPTED] Compression interrupted by user (Ctrl+C)\n"
            f"{'='*80}\n"
            f"  Timestep at interrupt: {sim.timestep}\n"
            f"  Current phi:           {packing_fraction(N, params.diameter, sim.state.box):.6f}\n"
            f"  Restart file:          {files.restart_gsd}\n"
            f"\n"
            f"  To resume:\n"
            f"    1. Verify restart file exists: ls -lh {files.restart_gsd}\n"
            f"    2. Re-run: python {sys.argv[0]} --simulparam_file {_active_json_path}\n"
            f"{'='*80}\n"
        )
        sys.exit(0)  # Exit gracefully (not an error)
    
    except Exception as unexpected_exception:
        # Unexpected exception during compression
        # Write emergency snapshot for recovery
        root_print(
            f"\n{'='*80}\n"
            f"[FATAL ERROR] Unexpected exception during compression\n"
            f"{'='*80}\n"
            f"  Exception type: {type(unexpected_exception).__name__}\n"
            f"  Message:        {unexpected_exception}\n"
            f"  Timestep:       {sim.timestep}\n"
            f"  Current phi:    {packing_fraction(N, params.diameter, sim.state.box):.6f}\n"
            f"{'='*80}\n"
        )
        
        # Attempt to write emergency snapshot
        try:
            emergency_filename = f"emergency_restart_{params.tag}_{int(time.time())}.gsd"
            _write_snapshot(sim, emergency_filename)
            root_print(
                f"[RECOVERY] Emergency snapshot written: {emergency_filename}\n"
                f"           You can resume from this snapshot by:\n"
                f"             1. Rename: mv {emergency_filename} {files.restart_gsd}\n"
                f"             2. Re-run: python {sys.argv[0]} --simulparam_file {_active_json_path}"
            )
        except Exception as snapshot_error:
            root_print(
                f"[ERROR] Failed to write emergency snapshot: {snapshot_error}\n"
                f"        Recovery from this state may not be possible."
            )
        
        # Re-raise exception for full traceback
        raise


    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    # Compression completed successfully, write all final files
    phi_final = packing_fraction(N, params.diameter, sim.state.box)

    try:
        write_final_outputs(sim, mc, params, files, phi_initial, phi_final, seed)
    except Exception as output_error:
        root_print(
            f"[WARNING] Error writing final outputs: {output_error}\n"
            f"          Simulation data may be incomplete."
        )

    # -----------------------------------------------------------------------
    # 8. Success exit
    # -----------------------------------------------------------------------
    root_print(
        f"\n{'='*80}\n"
        f"[SUCCESS] Compression completed successfully\n"
        f"{'='*80}\n"
        f"  End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'='*80}\n"
    )
    root_flush_stdout()



# ===========================================================================
#  SCRIPT EXECUTION
# ===========================================================================

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        # Expected exit (from validation errors, user interrupt, etc.)
        # Don't print additional traceback
        pass
    except Exception:
        # Unexpected exception (bug in code)
        # Print full traceback for debugging
        import traceback
        root_print("\n" + "="*80)
        root_print("[FATAL] Unexpected exception (bug in code):")
        root_print("="*80)
        traceback.print_exc()
        root_print("="*80)
        sys.exit(1)
