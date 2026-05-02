#!/usr/bin/env python3
# =============================================================================
# HOOMD-blue v4.9  | Convex-Polyhedron  HPMC NPT Simulation  
# =============================================================================
#
# PURPOSE
# -------
# Run a constant-pressure (NPT) hard Convex-Polyhedron HPMC simulation using
# HOOMD-blue v4.  The code executes two phases in sequence:
#   1. EQUILIBRATION — adaptive tuners are active; box-move sizes are adjusted.
#      If BoxMCMoveSize converges, its final move sizes are used.  If it does
#      not converge, the best observed box move sizes — judged by closeness of
#      the measured box acceptance rate to the requested target — are restored.
#   2. PRODUCTION   — BoxMCMoveSize is always removed before production; box
#      move sizes remain fixed while trajectory/log files are written.
#
# ALGORITHM
# ---------
# 1.  Parse --simulparam_file from the command line.
# 2.  Load and validate all parameters from a JSON file.
# 3.  Resolve stage-aware filenames (single-stage or multi-stage pipeline).
# 4.  Read a cryptographically-secure random seed from disk (create on first run).
# 5.  Build the HOOMD Simulation:
#       a. Choose CPU or GPU device.
#       b. Initialise state from restart GSD (if present) or input GSD.
#       c. Load scaled convex-polyhedron vertices from a shape JSON.
#       d. Configure hoomd.hpmc.integrate.ConvexPolyhedron integrator.
#       e. Perform a zero-step run to verify zero initial overlaps.
#       f. Attach loggers (Table + GSD scalar), GSD trajectory, GSD restart.
#       g. Attach BoxMC updater (volume, length, aspect, shear moves).
#       h. Attach MoveSize tuners for particle translation and rotation.
#       i. Attach BoxMCMoveSize tuner for all box-move types.
#       j. Optionally configure SDF pressure compute (attached post-equil).
# 6.  EQUILIBRATION loop:
#       Run in chunks of `equil_steps_check_freq`; track the box move sizes
#       whose measured box acceptance rate is closest to the target; stop early
#       when box_tuner.tuned is True.
# 7.  Always remove BoxMCMoveSize before production.  If tuning did not
#       converge, restore the best observed box move sizes first.
# 8.  Attach SDF pressure compute (if enabled) after equilibration.
# 9.  PRODUCTION run — sim.run(prod_steps) with fixed box move sizes.
# 9.  Write final GSD, summary JSON, and console banner.
# 10. Always flush and close open log file handles in a `finally` block.
#
# USAGE
# -----
#   python HOOMD_hard_polyhedra_NPT.py --simulparam_file simulparam_hard_polyhedra_npt.json
#   mpirun -n 8 python HOOMD_hard_polyhedra_NPT.py --simulparam_file simulparam_hard_polyhedra_npt.json
#
# DEPENDENCIES
# ------------
#   HOOMD-blue >= 4.0  (https://hoomd-blue.readthedocs.io/en/v4.9.0/)
#   GSD >= 3.0
#   NumPy
#   mpi4py  (optional; serial fallback provided)
# =============================================================================
# OUTPUT FILES (example, single-stage mode)
# ------------------------------------------
#   npt_hpmc_output_traj.gsd    — per-particle trajectory (append mode)
#   npt_hpmc_restart.gsd        — single-frame restart checkpoint (truncated)
#   npt_hpmc_final.gsd          — final configuration after production
#   npt_hpmc_log.log            — human-readable Table log (timestep, phi, …)
#   box_npt_log.log             — box geometry log (Lx, Ly, Lz, xy, xz, yz)
#   npt_hpmc_scalar_log.gsd     — scalar-only GSD log (no particle data)
#   <tag>_stage<id>_npt_summary.json — machine-readable provenance summary
#   random_seed.json            — persistent RNG seed (created once)
# =============================================================================


from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# GSD import
# ---------------------------------------------------------------------------
try:
    import gsd.hoomd
except ImportError as _e:
    sys.exit(f"[FATAL] gsd package not found: {_e}")

# ---------------------------------------------------------------------------
# HOOMD import
# ---------------------------------------------------------------------------
# hoomd          : top-level package (Simulation, Box, Snapshot, version info)
# hoomd.hpmc     : Hard-Particle Monte Carlo integrators, BoxMC updater,
#                  MoveSize / BoxMCMoveSize tuners, SDF pressure compute
# hoomd.logging  : Logger class — collects loggable quantities from HOOMD
#                  objects and custom Python loggables
# hoomd.write    : GSD, Table, and HDF5Log file writers
# hoomd.trigger  : Periodic(N) trigger — fires every N timesteps
# hoomd.filter   : All() selects all particles; Null() selects none
#                  (used to write a scalar-only GSD with zero particle data)
# DataAccessError: raised when an HPMC counter is queried before sim.run()
#   has been called.  Every custom loggable property guards against this
#   because HOOMD's Logger validates loggables by calling property getters
#   at registration time — before any sim.run() call.
try:
    import hoomd
    import hoomd.hpmc
    import hoomd.logging
    import hoomd.write
    import hoomd.trigger
    import hoomd.filter
    from hoomd.error import DataAccessError
except ImportError as _e:
    sys.exit(f"[FATAL] HOOMD-blue not found: {_e}")

# ---------------------------------------------------------------------------
# mpi4py 
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False
    class _MPIStub:
        """
        Minimal MPI stub for serial (non-MPI) runs.

        Provides the subset of mpi4py's MPI.COMM_WORLD interface that this
        script uses, so that the rest of the code is written once and works
        in both serial and MPI-parallel contexts without ``if _MPI_AVAILABLE``
        guards scattered throughout the code.
        """
        class COMM_WORLD:
            @staticmethod
            def bcast(obj, root=0):
                return obj
    MPI = _MPIStub()


# ===========================================================================
#  SECTION 1 — MPI-aware console helpers  
# ===========================================================================

def _mpi_rank_from_env() -> int:
    """
    Determine the current MPI rank by reading launcher environment variables.

    This is used *before* MPI.COMM_WORLD.Get_rank() is safely callable (e.g.
    when deciding whether to print a fatal error message) and also provides
    rank information even when mpi4py is not installed.

    Checks in order of precedence:
      OMPI_COMM_WORLD_RANK — Open MPI launcher
      PMI_RANK             — MPICH / Slurm PMI launcher
      SLURM_PROCID         — Slurm-native launcher without PMI

    Falls back to 0 (rank 0 = "this is the only process") when none of the
    environment variables are set, which is the correct behaviour for serial
    (non-MPI) runs.

    Returns
    -------
    int
        MPI world rank of the calling process (0-based).
    """
    return int(
        os.environ.get(
            "OMPI_COMM_WORLD_RANK",
            os.environ.get("PMI_RANK", os.environ.get("SLURM_PROCID", 0))
        )
    )

def _is_root_rank() -> bool:
    """
    Return True only on MPI rank 0 (or in a serial run where rank is always 0).

    Used as a gate so that console output, file writes, and other rank-0-only
    operations are not duplicated across all N MPI processes.  Without this
    guard, running with ``mpirun -n 8`` would print every INFO line eight times.

    Returns
    -------
    bool
        True iff the calling process is MPI rank 0.
    """
    return _mpi_rank_from_env() == 0

def root_print(*args, **kwargs) -> None:
    """
    Print only from MPI rank 0.

    Prevents N-fold duplicate console output when running under mpirun.
    All INFO, WARNING, and DEBUG messages in this script go through this
    function rather than plain print().

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to the built-in print().
    """
    if _is_root_rank():
        print(*args, **kwargs)

def root_flush_stdout() -> None:
    if _is_root_rank():
        sys.stdout.flush()


def fail_with_context(message: str, **kwargs) -> None:
    """
    Abort the program with a structured, multi-line fatal error message.

    Compared with ``sys.exit(message)``, this function produces a
    much more diagnostic output: every keyword argument is printed on its
    own indented line, making it immediately clear which file, key, or
    parameter value caused the failure — without requiring the operator to
    read a raw Python traceback.

    Always calls ``sys.exit()``, so this function never returns.

    Parameters
    ----------
    message : str
        One-line description of what went wrong.
    **kwargs
        Key-value pairs of diagnostic context.  Common examples:
          json_file=path, error_type=exc_type, error=str(exc),
          input_gsd=path, N=n_particles, expected_shape=(N,3)

    Example
    -------
    >>> fail_with_context(
    ...     "Shape JSON file not found.",
    ...     shape_json_filename="/data/cube.json",
    ...     cwd=os.getcwd(),
    ... )
    [FATAL] Shape JSON file not found.
      shape_json_filename: /data/cube.json
      cwd: /home/user/sim
    """
    lines = [f"[FATAL] {message}"]
    for k, v in kwargs.items():
        lines.append(f"  {k}: {v}")
    sys.exit("\n".join(lines))


def debug_kv(title: str, **kwargs) -> None:
    """
    Pretty-print a compact key=value debug block from rank 0 only.

    Called at key decision points (device selection, state initialisation,
    snapshot loading) and in the emergency exception handler to provide
    maximum diagnostic context without cluttering normal console output.

    The output is always flushed immediately so it appears before any
    subsequent exception traceback even when stdout is buffered.

    Parameters
    ----------
    title : str
        Short description of the block (e.g. "State initialisation decision").
    **kwargs
        Key-value pairs to display.

    Example output
    --------------
    [DEBUG] State initialisation decision
        restart_exists = False
        final_exists   = False
        input_gsd      = /scratch/sim/input.gsd
    """
    if not _is_root_rank():
        return
    print(f"[DEBUG] {title}")
    for k, v in kwargs.items():
        print(f"    {k} = {v}")
    sys.stdout.flush()


# ===========================================================================
#  SECTION 2 — Custom loggable classes 
# ===========================================================================

class Status:
    """
    Tracks estimated time remaining (ETR) for the current simulation phase.

    Registered with the main simulation logger so that every Table log row
    contains a human-readable ETR string and a "current/total" step fraction.

    Parameters
    ----------
    simulation : hoomd.Simulation
        The active HOOMD simulation object whose timestep and TPS are tracked.
    """

    _export_dict = {
        "timestep_fraction": ("string", True),
        "etr":               ("string", True),
    }

    def __init__(self, simulation: hoomd.Simulation) -> None:
        self.simulation = simulation

    @property
    def timestep_fraction(self) -> str:
        """Return 'current_step/final_step' as a string."""
        try:
            return (f"{self.simulation.timestep}/"
                    f"{self.simulation.final_timestep}")
        except Exception:
            return "0/?"   # before sim.run() sets final_timestep

    @property
    def seconds_remaining(self) -> float:
        """
        Estimated seconds until the current sim.run() call completes.

        Computed as:
            ETR = (final_timestep - current_timestep) / current_tps

        Returns 0.0 on any failure so that the ``etr`` property can always
        format a valid timedelta string.
        """
        try:
            return ((self.simulation.final_timestep - self.simulation.timestep)
                    / self.simulation.tps)
        except (ZeroDivisionError, DataAccessError, AttributeError):
            # ZeroDivisionError : tps == 0.0 before any MC sweeps have run
            # DataAccessError   : HPMC counters not yet initialised
            # AttributeError    : final_timestep not yet set by sim.run()
            return 0.0

    @property
    def etr(self) -> str:
        """
        Human-readable estimated time remaining as 'HH:MM:SS.ffffff'.

        Delegates to ``seconds_remaining`` and formats via datetime.timedelta
        for a consistent, easy-to-read representation in the Table log.
        """
        return str(datetime.timedelta(seconds=self.seconds_remaining))


class MCStatus:
    """
    Reports *windowed* translational and rotational acceptance rates.

    HOOMD's HPMC integrator exposes cumulative counters via:
        mc.translate_moves => (accepted, rejected)
        mc.rotate_moves    => (accepted, rejected)

    The class is registered with both the main Table logger and the scalar
    GSD logger so acceptance rates appear in both output files.

    Parameters
    ----------
    integrator : hoomd.hpmc.integrate.ConvexPolyhedron
        The active HPMC integrator whose move counters are monitored.
    """

    _export_dict = {"translate_acceptance_rate": ("scalar", True), "rotate_acceptance_rate":    ("scalar", True),}

    def __init__(self, integrator) -> None:
        self.integrator = integrator

        # Previous cumulative accepted/total counts for translation moves.
        # Initialised to None so the first call knows there is no prior window.
        self.prev_accepted_trans = None
        self.prev_total_trans = None

        # Previous cumulative accepted/total counts for rotation moves.
        self.prev_accepted_rot = None
        self.prev_total_rot = None

    @property
    def translate_acceptance_rate(self) -> float:
        try:
            # translate_moves = (accepted, rejected); index 0 = accepted
            current_accepted = self.integrator.translate_moves[0]
            current_total = sum(self.integrator.translate_moves)

            if self.prev_accepted_trans is None or self.prev_total_trans is None:
                # First call: no previous window available; use cumulative rate.
                self.prev_accepted_trans = current_accepted
                self.prev_total_trans = current_total
                return current_accepted / current_total if current_total else 0.0

            # Windowed delta
            delta_accepted = current_accepted - self.prev_accepted_trans
            delta_total = current_total - self.prev_total_trans

            # Update stored baseline for the next window
            self.prev_accepted_trans = current_accepted
            self.prev_total_trans = current_total

            return delta_accepted / delta_total if delta_total else 0.0
        except (IndexError, ZeroDivisionError, DataAccessError):
            return 0.0

    @property
    def rotate_acceptance_rate(self) -> float:
        """
        Windowed rotational move acceptance fraction.

        Exact same logic as ``translate_acceptance_rate`` but applied to
        ``mc.rotate_moves``.  Separate property so both rates appear as
        independent columns in the Table log.
        """
        try:
            current_accepted = self.integrator.rotate_moves[0]
            current_total = sum(self.integrator.rotate_moves)

            if self.prev_accepted_rot is None or self.prev_total_rot is None:
                self.prev_accepted_rot = current_accepted
                self.prev_total_rot = current_total
                return current_accepted / current_total if current_total else 0.0

            delta_accepted = current_accepted - self.prev_accepted_rot
            delta_total = current_total - self.prev_total_rot

            self.prev_accepted_rot = current_accepted
            self.prev_total_rot = current_total

            return delta_accepted / delta_total if delta_total else 0.0
        except (IndexError, ZeroDivisionError, DataAccessError):
            return 0.0


class Box_property:
    """
    Exposes simulation-box dimensions and derived quantities as loggable scalars.

    Wraps ``sim.state.box`` so that every box parameter (Lx, Ly, Lz, xy, xz,
    yz), the box volume, and the instantaneous packing fraction φ = N·V_p/V
    can be registered individually with HOOMD's Logger.

    Parameters
    ----------
    simulation : hoomd.Simulation
        Active HOOMD simulation whose state.box is read each logging event.
    N : int
        Total number of particles (fixed for the entire run).
    particle_volume : float
        Single-particle volume V_p = reference_volume * shape_scale**3
        (in simulation-length units cubed).
    """

    _export_dict = {
        "L_x":              ("scalar", True),
        "L_y":              ("scalar", True),
        "L_z":              ("scalar", True),
        "XY":               ("scalar", True),
        "XZ":               ("scalar", True),
        "YZ":               ("scalar", True),
        "volume":           ("scalar", True),
        "volume_str":       ("string", True),
        "packing_fraction": ("scalar", True), 
    }

    def __init__(self, simulation: hoomd.Simulation, N: int, particle_volume: float) -> None:
        self.simulation = simulation
        self._N        = N
        self._particle_volume = particle_volume

    # Individual box-vector components ---

    @property
    def L_x(self) -> float:
        """Box length along the x-axis (Lx) in simulation length units."""
        return float(self.simulation.state.box.Lx)

    @property
    def L_y(self) -> float:
        """Box length along the y-axis (Ly) in simulation length units."""
        return float(self.simulation.state.box.Ly)

    @property
    def L_z(self) -> float:
        """Box length along the z-axis (Lz) in simulation length units."""
        return float(self.simulation.state.box.Lz)

    @property
    def XY(self) -> float:
        """Tilt factor xy (shear of y-axis in the x-direction)."""
        return float(self.simulation.state.box.xy)

    @property
    def XZ(self) -> float:
        """Tilt factor xz (shear of z-axis in the x-direction)."""
        return float(self.simulation.state.box.xz)

    @property
    def YZ(self) -> float:
        """Tilt factor yz (shear of z-axis in the y-direction)."""
        return float(self.simulation.state.box.yz)

    @property
    def volume(self) -> float:
        """
        Box volume V = Lx * Ly * Lz * (1 - xy*xz*yz + ...) in length**3.

        HOOMD computes this correctly for triclinic boxes.
        """
        return float(self.simulation.state.box.volume)

    @property
    def volume_str(self) -> str:
        """Box volume formatted as a 2-decimal-place string for Table logs."""
        return f"{self.simulation.state.box.volume:.2f}"

    @property
    def packing_fraction(self) -> float:
        """
        Instantaneous packing fraction phi = N * V_particle / V_box.

        This quantity fluctuates in the NPT ensemble as the box volume changes
        due to accepted BoxMC moves.  Logging phi at every step allows one to
        verify convergence toward the equilibrium density at the target pressure.

        The computation is:
            phi = N * particle_volume / box.volume

        where particle_volume = reference_volume * shape_scale**3.
        """
        V = self.simulation.state.box.volume
        return self._N * self._particle_volume / V


class MoveSizeProp:
    """
    Exposes the current translational (d) and rotational (a) move sizes as
    loggable scalars.

    HOOMD's MoveSize tuner adjusts ``mc.d["A"]`` and ``mc.a["A"]`` during
    the simulation to maintain the target acceptance rate.  Logging these
    values lets the operator see how quickly the tuner is converging and
    whether the final move sizes are physically reasonable.

    Parameters
    ----------
    mc : hoomd.hpmc.integrate.ConvexPolyhedron
        The active HPMC integrator.
    ptype : str
        Particle type label for which to report move sizes (default: "A").
    """

    _export_dict = {"d": ("scalar", True), "a": ("scalar", True)}

    def __init__(self, mc, ptype: str = "A") -> None:
        self.mc = mc
        self.ptype = ptype

    @property
    def d(self) -> float:
        """
        Current translational move size d for particle type ``ptype``.

        ``mc.d`` is a dict-like mapping from type label to the maximum
        translational displacement tried in each MC sweep.  The MoveSize
        tuner adjusts this value to hit the target acceptance rate.

        Returns 0.0 before sim.run() (DataAccessError from HOOMD internals).
        """
        try:
            return float(self.mc.d[self.ptype])
        except DataAccessError:
            return 0.0

    @property
    def a(self) -> float:
        """
        Current rotational move size a for particle type ``ptype``.

        ``mc.a`` is the maximum rotation angle (in radians on the unit
        quaternion sphere) tried in each MC sweep.  Tuned independently
        from the translation move size.

        Returns 0.0 before sim.run() (DataAccessError from HOOMD internals).
        """
        try:
            return float(self.mc.a[self.ptype])
        except DataAccessError:
            return 0.0


class BoxMCStatus:
    # ---------------------------------------------------------------------
    # HOOMD logger export dictionary
    # ---------------------------------------------------------------------
    _export_dict = {
        # Combined windowed acceptance rate for all tracked BoxMC moves:
        "acceptance_rate":      ("scalar", True),  # windowed combined box acceptance
        # Windowed acceptance rate for length/volume-like box moves.
        "length_acc_rate":      ("scalar", True),  # windowed length/volume-move acceptance
        # Windowed acceptance rate for shear box moves.
        "shear_acc_rate":       ("scalar", True),  # windowed shear-move acceptance
        # Windowed raw length/volume-like move counts as a string:
        "length_moves_str":     ("string", True),  # windowed "accepted,rejected,total"
        # Windowed raw shear move counts as:
        "shear_moves_str":      ("string", True),  # windowed "accepted,rejected,total"
        # Windowed raw combined box move counts as:
        "combined_moves_str":   ("string", True),  # windowed "accepted,rejected,total"
    }

    def __init__(self, boxmc, sim):
        # Store the HOOMD BoxMC updater.
        self.boxmc = boxmc
        # Store the HOOMD Simulation object.
        self.sim = sim
        # Previous cumulative shear counter.
        # This will eventually store: (previous_shear_accepted, previous_shear_rejected)
        self._prev_shear = None
        # Previous cumulative length/volume-like counter.
        # This will eventually store: (previous_length_accepted, previous_length_rejected)
        # In this code, length/volume-like counters come from boxmc.volume_moves
        self._prev_length_like = None
        # Timestep at which the cache was last updated.
        self._cached_timestep = None
        # Cached combined acceptance rate for the latest window.
        self._cached_overall = 0.0
        # Cached length/volume-like acceptance rate for the latest window.
        self._cached_length = 0.0
        # Cached shear acceptance rate for the latest window.
        self._cached_shear = 0.0
        # Cached raw length/volume-like counts for the latest window: (accepted, rejected, total)
        self._cached_length_counts = (0, 0, 0)
        # Cached raw shear counts for the latest window: (accepted, rejected, total)
        self._cached_shear_counts = (0, 0, 0)
        # Cached raw combined counts for the latest window: (accepted, rejected, total)
        self._cached_combined_counts = (0, 0, 0)

    @staticmethod
    def _compute_rate(acc, rej):
        total = acc + rej
        return acc / total if total > 0 else 0.0

    @staticmethod
    def _delta_counts(now, prev):
        """
        Return window counts from cumulative HOOMD counters.

        HOOMD v4 BoxMC counters can reset at the beginning of a new sim.run()
        call.  If a reset is detected, use the current counter values directly
        instead of subtracting the previous cached values.
        """
        if prev is None or now[0] < prev[0] or now[1] < prev[1]:
            return now
        return (max(0, now[0] - prev[0]), max(0, now[1] - prev[1]))

    def _refresh_cache(self):
        """
        Refresh the cached BoxMC acceptance diagnostics for the current timestep.

        This method does the main work of the class.

        Steps
        -----
        1. Check whether the cache is already valid for this timestep.
        2. Read raw HOOMD BoxMC counters.
        3. Convert them to integer tuples.
        4. Compute window counts by subtracting previous counters.
        5. Update previous counters.
        6. Compute length, shear, and combined acceptance rates.
        7. Store everything in cached variables.

        Why cache?
        ----------
        HOOMD's logger may request multiple properties from this object at the
        same timestep:

            acceptance_rate
            length_acc_rate
            shear_acc_rate
            length_moves_str
            shear_moves_str
            combined_moves_str

        If every property independently subtracted counters and updated
        previous baselines, the first property call would consume the window,
        and the later properties would incorrectly see zero moves.

        Therefore, all properties call _refresh_cache(), but this function only
        recomputes once per timestep. Subsequent property calls at the same
        timestep simply reuse the cached values.
        """

        # Current HOOMD timestep.
        current_timestep = int(self.sim.timestep)

        # If we already refreshed at this timestep, return immediately.
        #
        # This is essential because the HOOMD logger can query several
        # properties at the same timestep. Without this guard, the first
        # property call would update _prev_length_like/_prev_shear, and the
        # next property call would see zero window counts.
        if self._cached_timestep == current_timestep:
            return

        # -----------------------------------------------------------------
        # Read raw BoxMC counters from HOOMD.
        # -----------------------------------------------------------------
        try:
            length_acc, length_rej = self.boxmc.volume_moves
            shear_acc, shear_rej = self.boxmc.shear_moves
        except DataAccessError:
            length_acc = length_rej = shear_acc = shear_rej = 0

        # In this code, BoxMC.length moves are enabled. In HOOMD v4 these
        # length/volume-like accepted,rejected counts are exposed through
        # boxmc.volume_moves. Shear accepted,rejected counts are exposed
        # separately through boxmc.shear_moves.
        length_now = (int(length_acc), int(length_rej))
        shear_now = (int(shear_acc), int(shear_rej))

        # -----------------------------------------------------------------
        # Convert cumulative counters into window counters.
        # -----------------------------------------------------------------
        dlen_acc, dlen_rej = self._delta_counts(length_now, self._prev_length_like)
        dshr_acc, dshr_rej = self._delta_counts(shear_now, self._prev_shear)

        # Store current cumulative counters as the new baseline for the next
        # logging window.
        self._prev_length_like = length_now
        self._prev_shear = shear_now

        # -----------------------------------------------------------------
        # Compute total attempts for each move family.
        # -----------------------------------------------------------------
        len_total = dlen_acc + dlen_rej
        shr_total = dshr_acc + dshr_rej
        comb_acc = dlen_acc + dshr_acc
        comb_rej = dlen_rej + dshr_rej
        comb_total = comb_acc + comb_rej

        # -----------------------------------------------------------------
        # Compute and cache acceptance rates.
        # -----------------------------------------------------------------
        self._cached_length = self._compute_rate(dlen_acc, dlen_rej)
        self._cached_shear = self._compute_rate(dshr_acc, dshr_rej)
        self._cached_overall = self._compute_rate(comb_acc, comb_rej)

        # -----------------------------------------------------------------
        # Cache raw counts in accepted,rejected,total form.
        # -----------------------------------------------------------------
        self._cached_length_counts = (dlen_acc, dlen_rej, len_total)
        self._cached_shear_counts = (dshr_acc, dshr_rej, shr_total)
        self._cached_combined_counts = (comb_acc, comb_rej, comb_total)
        # Mark cache as valid for this timestep.
        self._cached_timestep = current_timestep

    @staticmethod
    def _counts_to_str(counts):
        return f"{counts[0]},{counts[1]},{counts[2]}"

    @property
    def acceptance_rate(self):
        self._refresh_cache()
        return self._cached_overall

    @property
    def length_acc_rate(self):
        self._refresh_cache()
        return self._cached_length

    @property
    def shear_acc_rate(self):
        self._refresh_cache()
        return self._cached_shear

    @property
    def length_moves_str(self):
        self._refresh_cache()
        return self._counts_to_str(self._cached_length_counts)

    @property
    def shear_moves_str(self):
        self._refresh_cache()
        return self._counts_to_str(self._cached_shear_counts)

    @property
    def combined_moves_str(self):
        self._refresh_cache()
        return self._counts_to_str(self._cached_combined_counts)


class BoxSeqProp:
    """
    Exposes raw BoxMC move counters as formatted strings for the box Table log.

    These are the raw HOOMD counters visible at the logging event.  In the
    equilibration loop, each sim.run(equil_steps_check_freq) chunk resets the
    BoxMC counters in HOOMD v4, so these are effectively per-chunk counters
    when printed immediately after a chunk.
    """

    _export_dict = {
        "length_moves_str":    ("string", True),  # "accepted,rejected,total" for volume/length-like moves
        "shear_moves_str":     ("string", True),  # "accepted,rejected,total" for shear moves
        "combined_moves_str":  ("string", True),  # "accepted,rejected,total" for length+shear
    }

    def __init__(self, boxmc: hoomd.hpmc.update.BoxMC) -> None:
        self.boxmc = boxmc

    @staticmethod
    def _counter_to_tuple(counter):
        try:
            a, r = counter
            a = int(a)
            r = int(r)
            return a, r, a + r
        except DataAccessError:
            return 0, 0, 0

    @staticmethod
    def _tuple_to_str(vals) -> str:
        return f"{vals[0]},{vals[1]},{vals[2]}"

    @property
    def length_moves_str(self) -> str:
        """Raw volume/length-like BoxMC counts as "accepted,rejected,total"."""
        return self._tuple_to_str(self._counter_to_tuple(self.boxmc.volume_moves))

    @property
    def shear_moves_str(self) -> str:
        """Raw shear BoxMC counts as "accepted,rejected,total"."""
        return self._tuple_to_str(self._counter_to_tuple(self.boxmc.shear_moves))

    @property
    def combined_moves_str(self) -> str:
        """Raw combined length+shear BoxMC counts as "accepted,rejected,total"."""
        la, lr, _ = self._counter_to_tuple(self.boxmc.volume_moves)
        sa, sr, _ = self._counter_to_tuple(self.boxmc.shear_moves)
        return self._tuple_to_str((la + sa, lr + sr, la + lr + sa + sr))


class OverlapCount:
    """
    Exposes the current HPMC particle-overlap count as a loggable scalar.

    In a valid hard-particle configuration this must always be 0.  Logging
    it provides a continuous sanity check: any non-zero value indicates a
    coding bug or a corrupt input configuration.

    Parameters
    ----------
    mc : hoomd.hpmc.integrate.ConvexPolyhedron
        The active HPMC integrator.
    """

    _export_dict = {"overlap_count": ("scalar", True)}

    def __init__(self, mc) -> None:
        self._mc = mc

    @property
    def overlap_count(self) -> float:
        """Number of overlapping particle pairs (must be 0 in valid config)."""
        try:
            return float(self._mc.overlaps)
        except DataAccessError:
            return 0.0


class SDFPressure:
    """
    Exposes the SDF-derived pressure and compressibility factor as loggables.

    BACKGROUND — Scale Distribution Function (SDF) pressure estimator
    ------------------------------------------------------------------
    ``hoomd.hpmc.compute.SDF`` measures the pressure in an HPMC simulation
    via the Scale Distribution Function method (Anderson et al. 2016).

    For a system of hard particles the dimensionless pressure \beta P is related
    to the SDF s(x) (the distribution of the smallest scale factor by which
    you would need to grow a particle before it overlaps a neighbour) via:

        \beta P = rho * (1 + s(0+) / (2d))

    where d = 3 (dimensionality) and rho = N/V is the number density.
    The loggable quantity ``sdf.betaP`` gives \beta P directly in units of 1/V.

    WHY SDF IS ATTACHED AFTER EQUILIBRATION
    ----------------------------------------
    SDF assumes a *fixed* box for each individual measurement and extrapolates
    the histogram to x = 0.  During equilibration the box fluctuates
    significantly due to large BoxMC moves, making the extrapolation unreliable.
    The SDF compute is therefore attached only after the box_tuner converges
    and the BoxMCMoveSize tuner is removed, ensuring the box has equilibrated
    before pressure measurements begin.  During production the average <βP>
    should converge to the target ``pressure`` parameter — providing a built-in
    cross-check of the equation of state.

    MPI NOTE
    ---------
    ``sdf.betaP`` returns None on all MPI ranks except rank 0.  Both property
    getters guard against this with an explicit ``if val is None`` check.

    References
    ----------
    Anderson, J. A., et al.  J. Comput. Phys. 2016, 325, 74–97.
    Eppenga, R. & Frenkel, D.  Mol. Phys. 1984, 52, 1303–1334.

    Parameters
    ----------
    sdf_compute : hoomd.hpmc.compute.SDF
        The SDF compute object (must already be attached to the simulation).
    simulation : hoomd.Simulation
        Used to read box volume for the compressibility calculation.
    N : int
        Total number of particles (constant during production).
    """

    _export_dict = {
        "betaP":              ("scalar", True),
        "compressibility_Z":  ("scalar", True),   # Z = betaP/rho = betaPV/N
    }

    def __init__(
        self,
        sdf_compute: hoomd.hpmc.compute.SDF,
        simulation:  hoomd.Simulation,
        N:           int,
    ) -> None:
        self._sdf = sdf_compute
        self._sim = simulation
        self._N   = N

    @property
    def betaP(self) -> float:
        """
        \beta P computed by the SDF method.
        Only valid on rank 0 (None on other ranks; guard returns 0.0).
        """
        try:
            # sdf.betaP returns None on all MPI ranks except rank 0
            val = self._sdf.betaP
            return float(val) if val is not None else 0.0
        except (DataAccessError, TypeError):
            return 0.0

    @property
    def compressibility_Z(self) -> float:
        """
        Z = \beta P/\rho = \beta PV/N.
        """
        try:
            bp = self._sdf.betaP
            if bp is None:
                return 0.0
            V = self._sim.state.box.volume
            rho = self._N / V
            return float(bp) / rho if rho > 0 else 0.0
        except (DataAccessError, TypeError, ZeroDivisionError):
            return 0.0


# ===========================================================================
#  SECTION 3 — Parameter dataclass  
# ===========================================================================

@dataclass
class SimulationParams:
    """
    Typed, validated container for all NPT simulation parameters.

    This dataclass is populated by ``load_simulparams()`` from a JSON file
    and validated by ``validate()``.  Storing parameters in a typed dataclass
    (rather than a plain dict) provides IDE autocompletion, catches typos at
    assignment time, and makes function signatures self-documenting.

    REQUIRED FIELDS (must appear in the JSON)
    ------------------------------------------
    tag : str
        Human-readable label for this run; used to prefix all output file
        names in multi-stage mode (stage_id >= 0).

    input_gsd_filename : str
        Path to the input GSD file containing the initial configuration.
        In multi-stage mode with stage_id > 0 this is overridden by the
        previous stage's final GSD.

    shape_json_filename : str
        Path to a JSON file describing the convex-polyhedron vertex
        coordinates and reference volume.  Expected keys (with optional
        suffixes): "vertices" and "volume".

    shape_scale : float > 0
        Scale factor applied to vertices and volume:
            scaled_vertices = shape_scale * reference_vertices
            particle_volume = shape_scale**3 * reference_volume

    stage_id_current : int >= -1
        -1  => single-stage run (filenames taken directly from JSON).
        0+  => multi-stage pipeline; files are prefixed as <tag>_<sid>_*.

    total_num_timesteps : int > 0
        Total number of MC sweeps (equil + production).

    equil_steps : int, 0 < equil_steps < total_num_timesteps
        Number of MC sweeps dedicated to equilibration.
        Production steps = total_num_timesteps - equil_steps.

    equil_steps_check_freq : int > 0
        Sub-run length for the equilibration loop.  After every
        equil_steps_check_freq steps the loop polls box_tuner.tuned and
        breaks early if True.  Should be a divisor of equil_steps.

    log_frequency : int > 0
        Trigger period (in MC sweeps) for the Table log and scalar GSD log.

    traj_gsd_frequency : int > 0
        Trigger period for writing frames to the trajectory GSD.

    restart_gsd_frequency : int > 0
        Trigger period for overwriting the single-frame restart GSD.

    move_size_translation : float > 0
        Initial translational move size d (in simulation length units).
        The MoveSize tuner will adapt this value during equilibration and
        production to maintain target_particle_trans_move_acc_rate.

    move_size_rotation : float > 0
        Initial rotational move size a (in units of the quaternion rotation
        angle).  Tuned to maintain target_particle_rot_move_acc_rate.

    trans_move_size_tuner_freq : int > 0
        Trigger period for the translational MoveSize tuner.

    rot_move_size_tuner_freq : int > 0
        Trigger period for the rotational MoveSize tuner.

    target_particle_trans_move_acc_rate : float in (0, 1)
        Target translational acceptance fraction (e.g. 0.3 = 30%).

    target_particle_rot_move_acc_rate : float in (0, 1)
        Target rotational acceptance fraction (e.g. 0.3 = 30%).

    npt_freq : int > 0
        Trigger period for the BoxMC updater (how often box moves are
        attempted relative to particle moves).

    pressure : float > 0
        Target reduced pressure βP = P/(kBT).  This is the dimensionless
        parameter passed to BoxMC.

    box_tuner_freq : int > 0
        Trigger period for the BoxMCMoveSize tuner.

    target_box_movement_acc_rate : float in (0, 1)
        Target acceptance fraction for all box-move types.

    use_gpu : bool
        If True, run on GPU.  Falls back to CPU if GPU initialisation fails.

    gpu_id : int
        Index of the GPU to use (only relevant when use_gpu = True).

    OPTIONAL FIELDS (default values shown)
    ----------------------------------------
    boxmc_volume_delta     : float = 0.1    Initial volume-move delta.
    boxmc_volume_mode      : str   = "standard"  "standard" or "ln".
    boxmc_length_delta     : float = 0.01   Initial length-move delta (each axis).
    boxmc_aspect_delta     : float = 0.02   Initial aspect-move delta.
    boxmc_shear_delta      : float = 0.01   Initial shear-move delta (each axis).
    max_move_volume        : float = 0.1    BoxMCMoveSize tuner cap for volume.
    max_move_length        : float = 0.05   BoxMCMoveSize tuner cap for length.
    max_move_aspect        : float = 0.02   BoxMCMoveSize tuner cap for aspect.
    max_move_shear         : float = 0.02   BoxMCMoveSize tuner cap for shear.
    max_translation_move   : float = 0.2    MoveSize tuner cap for translation.
    max_rotation_move      : float = 0.5    MoveSize tuner cap for rotation.
    enable_sdf             : bool  = True   Attach SDF pressure compute.
    sdf_xmax               : float = 0.02   SDF histogram upper limit.
    sdf_dx                 : float = 1e-4   SDF histogram bin width.
    initial_timestep       : int   = 0      Starting timestep for fresh runs.
    output_trajectory      : str   = "npt_hpmc_output_traj.gsd"
    simulation_log_filename: str   = "npt_hpmc_log.log"
    box_log_filename       : str   = "box_npt_log.log"
    scalar_gsd_log_filename: str   = "npt_hpmc_scalar_log.gsd"
    restart_file           : str   = "npt_hpmc_restart.gsd"
    final_gsd_filename     : str   = "npt_hpmc_final.gsd"

    RUNTIME-RESOLVED FIELD
    -----------------------
    particle_volume : float or None
        Computed in build_simulation() after loading the shape JSON:
            particle_volume = reference_volume * shape_scale**3
        Not provided by the user; initially None.
    """

    # --- I/O ---
    tag:                    str
    input_gsd_filename:     str
    shape_json_filename:    str
    shape_scale:            float
    stage_id_current:       int   # -1 = single-stage; >= 0 = multi-stage

    # --- Run lengths ---
    total_num_timesteps:    int   # total steps (equil + production)
    equil_steps:            int   # steps dedicated to equilibration
    equil_steps_check_freq: int   # chunk size for equil loop / early-exit check

    # --- Output frequencies ---
    log_frequency:          int   # Table log trigger
    traj_gsd_frequency:     int   # GSD trajectory trigger
    restart_gsd_frequency:  int   # GSD restart trigger

    # --- HPMC particle move ---
    move_size_translation:          float  # initial d for type A
    move_size_rotation:             float  # initial a for type A
    trans_move_size_tuner_freq:     int    # Translation MoveSize tuner trigger period
    rot_move_size_tuner_freq:       int    # Rotation MoveSize tuner trigger period
    target_particle_trans_move_acc_rate: float  # target acceptance for particle translational moves
    target_particle_rot_move_acc_rate:   float  # target acceptance for particle rotational moves

    # --- BoxMC ---
    npt_freq:                       int    # BoxMC trigger period
    pressure:                       float  # target P/(kBT) = betaP
    box_tuner_freq:                 int    # BoxMCMoveSize tuner trigger period
    target_box_movement_acc_rate:   float  # target acceptance for box moves

    # BoxMC move deltas
    boxmc_volume_delta:     float  = 0.1         # isotropic volume-change step
    boxmc_volume_mode:      str    = "standard"  # "standard" (linear) or "ln" (log)
    boxmc_length_delta:     float  = 0.01        # independent axis-length step
    boxmc_aspect_delta:     float  = 0.02        # aspect-ratio rescaling step
    boxmc_shear_delta:      float  = 0.01        # tilt-factor step

    # BoxMC tuner max_move_size
    max_move_volume:        float  = 0.1         # cap on volume delta
    max_move_length:        float  = 0.05        # cap on per-axis length delta
    max_move_aspect:        float  = 0.02        # cap on aspect delta
    max_move_shear:         float  = 0.02        # cap on shear delta

    # Particle tuner caps
    max_translation_move:   float  = 0.2        # cap on translational move size d
    max_rotation_move:      float  = 0.5        # cap on rotational move size a

    # --- SDF pressure compute  ---
    enable_sdf:             bool   = True  # attach hoomd.hpmc.compute.SDF
    sdf_xmax:               float  = 0.02  # max scale factor for SDF histogram
    sdf_dx:                 float  = 1e-4  # histogram bin width

    # --- Hardware  ---
    use_gpu:                bool   = False
    gpu_id:                 int    = 0

    # --- Single-stage filenames (ignored when stage_id >= 0) ---
    initial_timestep:       int    = 0
    output_trajectory:      str    = "npt_hpmc_output_traj.gsd"
    simulation_log_filename: str   = "npt_hpmc_log.log"
    box_log_filename:       str    = "box_npt_log.log"
    scalar_gsd_log_filename: str   = "npt_hpmc_scalar_log.gsd"
    restart_file:           str    = "npt_hpmc_restart.gsd"
    final_gsd_filename:     str    = "npt_hpmc_final.gsd"

    # runtime-resolved
    particle_volume: Optional[float] = None

    def validate(self) -> None:
        """
        Run physics and logic sanity checks on all parameters.

        All errors are collected into a list before raising so the operator
        sees every problem at once rather than fixing them one at a time and
        re-running to discover the next error.

        Raises
        ------
        ValueError
            If one or more parameters fail validation.  The error message
            lists all failing conditions with their actual values.
        """

        errors: list[str] = []

        if self.total_num_timesteps <= 0:
            errors.append(f"total_num_timesteps must be > 0, got {self.total_num_timesteps}")

        if self.equil_steps >= self.total_num_timesteps:
            errors.append(
                f"equil_steps ({self.equil_steps}) must be < "
                f"total_num_timesteps ({self.total_num_timesteps})"
            )
        if self.equil_steps_check_freq <= 0:
            errors.append(f"equil_steps_check_freq must be > 0")

        if not (0.0 < self.pressure):
            errors.append(f"pressure must be > 0, got {self.pressure}")
        
        if self.shape_scale <= 0.0:
            errors.append(f"shape_scale must be > 0, got {self.shape_scale}")

        if self.move_size_translation <= 0.0:
            errors.append(f"move_size_translation must be > 0, got {self.move_size_translation}")

        if self.move_size_rotation <= 0.0:
            errors.append(f"move_size_rotation must be > 0, got {self.move_size_rotation}")

        if not (0.0 < self.target_particle_trans_move_acc_rate < 1.0):
            errors.append(
                f"target_particle_trans_move_acc_rate must be in (0,1), got {self.target_particle_trans_move_acc_rate}"
            )

        if not (0.0 < self.target_particle_rot_move_acc_rate < 1.0):
            errors.append(
                f"target_particle_rot_move_acc_rate must be in (0,1), got {self.target_particle_rot_move_acc_rate}"
            )

        if self.max_translation_move <= 0.0:
            errors.append(
                f"max_translation_move must be > 0, got {self.max_translation_move}"
            )

        if self.max_rotation_move <= 0.0:
            errors.append(
                f"max_rotation_move must be > 0, got {self.max_rotation_move}"
            )

        if not (0.0 < self.target_box_movement_acc_rate < 1.0):
            errors.append(
                f"target_box_movement_acc_rate must be in (0,1), "
                f"got {self.target_box_movement_acc_rate}"
            )
        if self.stage_id_current < -1:
            errors.append(f"stage_id_current must be >= -1, got {self.stage_id_current}")

        for name, val in [
            ("log_frequency",          self.log_frequency),
            ("traj_gsd_frequency",     self.traj_gsd_frequency),
            ("restart_gsd_frequency",  self.restart_gsd_frequency),
            ("trans_move_size_tuner_freq", self.trans_move_size_tuner_freq),
            ("rot_move_size_tuner_freq",   self.rot_move_size_tuner_freq),
            ("box_tuner_freq",         self.box_tuner_freq),
            ("npt_freq",               self.npt_freq),
        ]:
            if val <= 0:
                errors.append(f"{name} must be > 0, got {val}")
        if errors:
            raise ValueError("Parameter validation failed:\n"
                             + "\n".join(f"  * {e}" for e in errors))


# ===========================================================================
#  SECTION 4 — JSON loading 
# ===========================================================================

_REQUIRED_KEYS: dict[str, type] = {
    "tag":                                  str,
    "input_gsd_filename":                   str,
    "shape_json_filename":                  str,
    "shape_scale":                          (int, float),
    "stage_id_current":                     int,
    "total_num_timesteps":                  int,
    "equil_steps":                          int,
    "equil_steps_check_freq":               int,
    "log_frequency":                        int,
    "traj_gsd_frequency":                   int,
    "restart_gsd_frequency":                int,
    "move_size_translation":                (int, float),
    "move_size_rotation":                   (int, float),
    "trans_move_size_tuner_freq":           int,
    "rot_move_size_tuner_freq":             int,
    "target_particle_trans_move_acc_rate":  (int, float),
    "target_particle_rot_move_acc_rate":    (int, float),
    "npt_freq":                             int,
    "pressure":                             (int, float),
    "box_tuner_freq":                       int,
    "target_box_movement_acc_rate":         (int, float),
    "use_gpu":                              bool,
    "gpu_id":                               int,
}

_OPTIONAL_KEYS: dict[str, type] = {
    "initial_timestep":         int,
    "output_trajectory":        str,
    "simulation_log_filename":  str,
    "box_log_filename":         str,
    "scalar_gsd_log_filename":  str,
    "restart_file":             str,
    "final_gsd_filename":       str,
    "enable_sdf":               bool,
    "sdf_xmax":                 (int, float),
    "sdf_dx":                   (int, float),
    "boxmc_length_delta":       (int, float),
    "boxmc_shear_delta":        (int, float),
    "max_move_length":          (int, float),
    "max_move_shear":           (int, float),
    "max_translation_move":     (int, float),
    "max_rotation_move":        (int, float),
}


def load_simulparams(json_path: str) -> SimulationParams:
    """
    Read, type-check, and return a SimulationParams from a JSON file.
    Keys starting with '_' are stripped (comment-keys).

    Parameters
    ----------
    json_path : str
        Path to the simulation parameter JSON file.

    Returns
    -------
    SimulationParams
        Fully validated parameter object.

    Raises / Exits
    --------------
    sys.exit
        On any file I/O error, JSON parse error, missing key, type error,
        or validation failure.  Error messages include the offending key
        and its actual value for fast diagnosis.
    """

    path = Path(json_path)

    # ---- File existence check ----
    if not path.exists():
        sys.exit(f"[FATAL] Parameter file not found: '{json_path}'")

    # ---- JSON parsing ----
    try:
        with path.open() as fh:
            raw: dict = json.load(fh)
    except json.JSONDecodeError as exc:
        fail_with_context(
            "JSON parse error in simulation parameter file.",
            json_file=str(path.resolve()),
            line=getattr(exc, "lineno", "?"),
            column=getattr(exc, "colno", "?"),
            error=str(exc),
        )
    except OSError as exc:
        fail_with_context(
            "Could not open/read simulation parameter file.",
            json_file=str(path.resolve()),
            error=repr(exc),
        )

    # ---- Top-level type check ----
    if not isinstance(raw, dict):
        fail_with_context(
            "Top-level JSON content must be an object/dictionary.",
            json_file=str(path.resolve()),
            parsed_type=type(raw).__name__,
        )

    # Strip comment-keys
    # Any key whose name starts with "_" is treated as a human annotation
    # (e.g. "_note", "_section_boxmc") and silently removed before
    # validation, allowing freely annotated JSON files without triggering
    # "unexpected key" errors.
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}

    # Check required keys present
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        sys.exit("[FATAL] Missing required keys in '{}': {}".format(
            json_path, missing))

    # Type-check required keys
    type_errors = []
    for key, expected in _REQUIRED_KEYS.items():
        if key in raw and not isinstance(raw[key], expected):
            type_errors.append(
                f"  * '{key}': expected {expected}, "
                f"got {type(raw[key]).__name__} ({raw[key]!r})"
            )
    if type_errors:
        sys.exit("[FATAL] Type errors in '{}':\n{}".format(
            json_path, "\n".join(type_errors)))

    # Build the kwargs dict for SimulationParams(**kw).  Required keys are
    # always present (validated above); optional keys are included only when
    # the JSON provides them, so dataclass defaults apply when absent.
    kw: dict = {k: raw[k] for k in _REQUIRED_KEYS}
    for key in _OPTIONAL_KEYS:
        if key in raw:
            kw[key] = raw[key]

    # Coerce numeric fields
    for fk in [
        "shape_scale",
        "move_size_translation",
        "move_size_rotation",
        "target_particle_trans_move_acc_rate",
        "target_particle_rot_move_acc_rate",
        "pressure",
        "target_box_movement_acc_rate",
        "boxmc_volume_delta",
        "boxmc_length_delta",
        "boxmc_aspect_delta",
        "boxmc_shear_delta",
        "max_move_volume",
        "max_move_length",
        "max_move_aspect",
        "max_move_shear",
        "max_translation_move",
        "max_rotation_move",
        "sdf_xmax",
        "sdf_dx",
    ]:
        if fk in kw:
            try:
                kw[fk] = float(kw[fk])
            except (TypeError, ValueError) as exc:
                fail_with_context(
                    f"Could not convert '{fk}' to float.",
                    key=fk,
                    value=kw[fk],
                    error=str(exc),
                )        

    # ---- Construct dataclass (validate() is called inside) ----
    try:
        params = SimulationParams(**kw)
    except TypeError as exc:
        # Unexpected keyword argument in kw — indicates a key in the JSON that
        # is not a known field of SimulationParams.
        fail_with_context(
            "Unexpected key(s) in parameter file.",
            json_file=str(path.resolve()),
            error=str(exc),
            hint="Check for typos; valid optional keys are: "
                 + ", ".join(_OPTIONAL_KEYS.keys()),
        )

    try:
        params.validate()
    except ValueError as exc:
        fail_with_context(
            "Parameter validation failed.",
            json_file=str(path.resolve()),
            errors=str(exc),
        )

    return params


# ===========================================================================
#  SECTION 5 — Seed management 
# ===========================================================================
#
# A reproducible random seed is essential for debugging and for generating
# independent replicas.  This module:
#   1. Creates a seed file on the very first run (rank 0 only).
#   2. Re-uses the same seed file on subsequent calls (restarts, later stages).
#   3. Broadcast-reads the seed via the MPI communicator so all ranks
#      initialise HOOMD's RNG with identical state.
#

_SEED_FILE_SINGLE = "random_seed.json"
_SEED_FILE_MULTI  = "random_seed_stage_0.json"


def _os_random_seed() -> int:
    """Cryptographically secure seed in [0, 65535] (HOOMD's valid range)."""
    return secrets.randbelow(65536)


def ensure_seed_file(stage_id: int, rank: int) -> None:
    """Rank-0 creates the seed file once; all other calls are no-ops."""

    if rank != 0:
        return

    # Choose the seed filename based on run mode:
    seed_file = _SEED_FILE_SINGLE if stage_id == -1 else _SEED_FILE_MULTI

    if not Path(seed_file).exists():
        # First call: generate a fresh cryptographically secure seed and persist it.
        seed = _os_random_seed()
        try:
            with open(seed_file, "w") as fh:
                json.dump({"random_seed": seed,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                        fh, indent=4)
        except OSError as exc:
            fail_with_context(
                "Could not write seed file.",
                seed_file=seed_file,
                error=repr(exc),
            )
        print(f"[INFO] Seed file created: {seed_file}  (seed={seed})")
    else:
        # Subsequent calls (restarts, later stages): re-use the existing seed
        # so trajectories remain reproducible across job resubmissions.
        print(f"[INFO] Existing seed file found: {seed_file}")


def read_seed(stage_id: int) -> int:
    """
    Read the persistent random seed from disk.

    Parameters
    ----------
    stage_id : int
        Determines which seed file to read.

    Returns
    -------
    int
        The stored seed value in [0, 65535].

    Raises / Exits
    --------------
    sys.exit
        If the file is missing, unreadable, malformed JSON, missing the
        "random_seed" key, or contains a non-integer value.
    """
    seed_file = _SEED_FILE_SINGLE if stage_id == -1 else _SEED_FILE_MULTI
    try:
        with open(seed_file) as fh:
            data = json.load(fh)
        return int(data["random_seed"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError, ValueError) as exc:
        fail_with_context(
            "Cannot read a valid random seed.",
            seed_file=seed_file,
            stage_id=stage_id,
            error=repr(exc),
        )


# ===========================================================================
#  SECTION 6 — Filename resolution 
# ===========================================================================

@dataclass
class RunFiles:
    """All resolved filenames for the current stage."""
    input_gsd:        str
    traj_gsd:         str
    restart_gsd:      str
    final_gsd:        str
    sim_log_txt:      str
    box_log_txt:      str
    scalar_gsd_log:   str   


def resolve_filenames(params: SimulationParams) -> RunFiles:
    """
    Stage-aware filename resolution.  

    stage_id == -1 => single-stage: use filenames from JSON.
    stage_id >= 0  => multi-stage: prefix all files with <tag>_<sid>_
    """
    tag = params.tag
    sid = params.stage_id_current

    if sid == -1:
        # Single-stage mode: filenames come directly from the JSON.
        # No automatic prefixing is applied.
        input_gsd      = params.input_gsd_filename
        traj_gsd       = params.output_trajectory
        restart_gsd    = params.restart_file
        final_gsd      = params.final_gsd_filename
        sim_log_txt    = params.simulation_log_filename
        box_log_txt    = params.box_log_filename
        scalar_gsd_log = params.scalar_gsd_log_filename

        if Path(final_gsd).exists():
            root_print(
                f"[WARNING] Final GSD '{final_gsd}' already exists. "
                "It will be overwritten at the end of this run."
            )

    elif sid >= 0:
        # Multi-stage mode: prefix all output filenames with <tag>_<sid>_
        # so each stage produces a clearly labelled, non-overlapping set
        # of output files.
        pfx            = f"{tag}_{sid}"
        traj_gsd       = f"{pfx}_traj.gsd"
        restart_gsd    = f"{pfx}_restart.gsd"
        final_gsd      = f"{pfx}_final.gsd"
        sim_log_txt    = f"{pfx}_sim.log"
        box_log_txt    = f"{pfx}_box.log"
        scalar_gsd_log = f"{pfx}_scalars.gsd"

        # Hard stop: if the final GSD for this stage already exists the stage
        # completed successfully and must not be overwritten — the next stage
        # reads this file as its input.  The user must increment stage_id.
        if Path(final_gsd).exists():
            sys.exit(
                f"[ERROR] Stage {sid} final GSD '{final_gsd}' already exists.\n"
                f"  => Increment 'stage_id_current' to {sid + 1} to continue."
            )

        # Stage 0 reads the user-provided input GSD; stages 1, 2, … read
        # the final GSD written by the immediately preceding stage, creating
        # an automatically chained pipeline.
        input_gsd = params.input_gsd_filename if sid == 0 \
            else f"{tag}_{sid - 1}_final.gsd"
        if sid > 0 and not Path(input_gsd).exists():
            sys.exit(
                f"[FATAL] Previous stage output not found: '{input_gsd}'.\n"
                f"  => Did stage {sid - 1} complete successfully?"
            )
    else:
        sys.exit(f"[ERROR] stage_id_current must be >= -1, got {sid}")

    if not Path(input_gsd).exists():
        sys.exit(f"[FATAL] Input GSD not found: '{input_gsd}'")

    return RunFiles(
        input_gsd=input_gsd,
        traj_gsd=traj_gsd,
        restart_gsd=restart_gsd,
        final_gsd=final_gsd,
        sim_log_txt=sim_log_txt,
        box_log_txt=box_log_txt,
        scalar_gsd_log=scalar_gsd_log,
    )


# ===========================================================================
#  SECTION 7 — Shape loader
# ===========================================================================

def _get_json_value_by_suffix(data: dict, suffix: str):
    """
    Find a value in a dict by exact key or by key ending with ``_<suffix>``.

    This allows shape JSON files to use either bare keys ("vertices",
    "volume") or annotated keys ("polyhedron_vertices", "convex_volume",
    etc.) without requiring a fixed schema.

    Parameters
    ----------
    data : dict
        The loaded shape JSON data.
    suffix : str
        The key suffix to search for (e.g. "vertices" or "volume").

    Returns
    -------
    The associated value from the dict.

    Raises
    ------
    KeyError
        If no key matches the exact name or the ``_<suffix>`` pattern.
    """
    for key, value in data.items():
        if key == suffix or key.endswith(f"_{suffix}"):
            return value
    raise KeyError(f"Could not find key with suffix '{suffix}' in shape JSON")


def load_convex_polyhedron_shape(shape_json_filename: str, shape_scale: float) -> dict:

    path = Path(shape_json_filename)

    # ---- File existence ----
    if not path.exists():
        fail_with_context("Shape JSON file not found.", shape_json_filename=str(path.resolve()))

    # ---- JSON parsing ----
    try:
        with path.open() as fh:
            raw = json.load(fh)
    except Exception as exc:
        fail_with_context(
            "Could not read shape JSON file.",
            shape_json_filename=str(path.resolve()),
            error_type=type(exc).__name__,
            error=str(exc),
        )

    # ---- Extract vertices and volume ----
    try:
        vertices = np.asarray(_get_json_value_by_suffix(raw, "vertices"), dtype=np.float64)
        reference_volume = float(_get_json_value_by_suffix(raw, "volume"))
    except Exception as exc:
        fail_with_context(
            "Shape JSON does not contain usable vertices/volume information.",
            shape_json_filename=str(path.resolve()),
            error_type=type(exc).__name__,
            error=str(exc),
        )

    # ---- Shape validation: vertices must be (N_v, 3) ----
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        fail_with_context(
            "Polyhedron vertices must have shape (N_vertices, 3).",
            vertices_shape=vertices.shape,
            shape_json_filename=str(path.resolve()),
        )

    if len(vertices) < 4:
        fail_with_context(
            "A convex polyhedron requires at least 4 vertices (tetrahedron).",
            n_vertices=len(vertices),
            shape_json_filename=str(path.resolve()),
        )

    # ---- Volume sanity check ----
    if reference_volume <= 0.0:
        fail_with_context(
            "Reference volume must be > 0.",
            reference_volume=reference_volume,
            shape_json_filename=str(path.resolve()),
        )

    # ---- Apply scale factor ----
    scaled_vertices = (shape_scale * vertices).tolist()
    particle_volume = reference_volume * shape_scale**3

    root_print(
        f"[DEBUG] Shape loaded: {path.resolve()} | "
        f"n_vertices={len(vertices)} | "
        f"ref_volume={reference_volume:.6g} | "
        f"scale={shape_scale} | "
        f"particle_volume={particle_volume:.6g}"
    )


    return {
        "shape_json_path": str(path.resolve()),
        "reference_volume": reference_volume,
        "particle_volume": particle_volume,
        "vertices": scaled_vertices,
    }


# ===========================================================================
#  SECTION 8 — MPI snapshot broadcast 
# ===========================================================================

def load_and_broadcast_snapshot(input_gsd: str, comm, rank: int) -> dict:
    """
    Rank-0 reads the last GSD frame; broadcasts minimal data to all ranks.
    """

    if rank == 0:
        path = Path(input_gsd)
        if not path.exists():
            fail_with_context(
                "Input GSD not found for state initialisation.",
                input_gsd=str(path.resolve()),
            )

        try:
            with gsd.hoomd.open(name=str(path), mode="r") as f:
                if len(f) == 0:
                    fail_with_context(
                        "Input GSD contains no frames.",
                        input_gsd=str(path.resolve()),
                    )
                # f[-1]: read the last (most recent) frame
                frame = f[-1]
                box_data  = frame.configuration.box
                positions = np.asarray(frame.particles.position, dtype=np.float64)
                orientations = np.asarray(frame.particles.orientation, dtype=np.float64)
                typeid    = np.asarray(frame.particles.typeid, dtype=np.int32)
                types     = list(frame.particles.types)
                N         = int(len(positions))

        except Exception as exc:
            fail_with_context(
                "Failed while reading the last frame from input GSD.",
                input_gsd=str(path.resolve()),
                error_type=type(exc).__name__,
                error=str(exc),
            )

        # ---- Array shape validation ----
        if positions.ndim != 2 or positions.shape[1] != 3:
            fail_with_context(
                "Particle position array has invalid shape.",
                input_gsd=str(path.resolve()),
                positions_shape=positions.shape,
            )

        if orientations.ndim != 2 or orientations.shape[1] != 4:
            fail_with_context(
                "Particle orientation array has invalid shape.",
                input_gsd=str(path.resolve()),
                orientations_shape=orientations.shape,
            )

        if len(orientations) != N:
            fail_with_context(
                "Orientation array length does not match number of particles.",
                input_gsd=str(path.resolve()),
                N=N,
                n_orientations=len(orientations),
            )

        if len(typeid) != N:
            fail_with_context(
                "typeid array length does not match number of particles.",
                input_gsd=str(path.resolve()),
                N=N,
                n_typeid=len(typeid),
            )

        if len(types) == 0:
            fail_with_context(
                "No particle types found in input GSD.",
                input_gsd=str(path.resolve()),
            )

        debug_kv(
            "Loaded input snapshot",
            input_gsd=str(path.resolve()),
            N=N,
            positions_shape=positions.shape,
            orientations_shape=orientations.shape,
            typeid_shape=typeid.shape,
            box=list(box_data),
            particle_types=types,
        )

        snap_data = {
            "box": box_data,
            "positions": positions,
            "orientations": orientations,
            "typeid": typeid,
            "types": types,
            "N": N,
        }
    else:
        snap_data = None

    # **** MPI Broadcast ****
    snap_data = comm.bcast(snap_data, root=0)
    return snap_data


def reconstruct_snapshot(snap_data: dict, rank: int) -> hoomd.Snapshot:
    """
    Reconstruct a hoomd.Snapshot from broadcast data with validation.
    """
    # Non-root ranks return an empty Snapshot.
    if rank != 0:
        return hoomd.Snapshot()

    # Validate that all expected keys are present in the broadcast dict before attempting Snapshot construction.
    required = ["box", "positions", "orientations", "typeid", "types", "N"]
    missing = [k for k in required if k not in snap_data]
    if missing:
        fail_with_context(
            "Broadcast snapshot data is incomplete.",
            missing_keys=missing,
            available_keys=sorted(snap_data.keys()),
        )

    # ---- Unpack and re-validate array shapes ----
    positions    = np.asarray(snap_data["positions"], dtype=np.float64)
    orientations = np.asarray(snap_data["orientations"], dtype=np.float64)
    typeid       = np.asarray(snap_data["typeid"], dtype=np.uint32)
    types        = list(snap_data["types"])
    N            = int(snap_data["N"])
    box          = list(snap_data["box"])

    if positions.shape != (N, 3):
        fail_with_context(
            "Snapshot reconstruction failed: positions shape mismatch.",
            expected_shape=(N, 3),
            actual_shape=positions.shape,
        )

    if orientations.shape != (N, 4):
        fail_with_context(
            "Snapshot reconstruction failed: orientations shape mismatch.",
            expected_shape=(N, 4),
            actual_shape=orientations.shape,
        )

    if typeid.shape != (N,):
        fail_with_context(
            "Snapshot reconstruction failed: typeid shape mismatch.",
            expected_shape=(N,),
            actual_shape=typeid.shape,
        )

    try:
        snapshot = hoomd.Snapshot()
        snapshot.configuration.box = box
        snapshot.particles.N = N
        snapshot.particles.types = types
        snapshot.particles.position[:] = positions
        snapshot.particles.orientation[:] = orientations
        snapshot.particles.typeid[:] = typeid
    except Exception as exc:
        fail_with_context(
            "HOOMD Snapshot reconstruction failed.",
            error_type=type(exc).__name__,
            error=str(exc),
            N=N,
            n_types=len(types),
            positions_shape=positions.shape,
            orientations_shape=orientations.shape,
            typeid_shape=typeid.shape,
            box=box,
        )

    return snapshot




# ===========================================================================
#  SECTION 9 — Simulation builder
# ===========================================================================

def build_simulation(
    params: SimulationParams,
    files:  RunFiles,
    seed:   int,
    comm,
    rank:   int,
) -> tuple:
    """
    Build and fully configure the HOOMD v4.9 NPT simulation.

    Returns
    -------
    sim          : hoomd.Simulation
    mc           : HPMC ConvexPolyhedron integrator
    boxmc        : BoxMC updater
    move_tuner   : MoveSize tuner
    box_tuner    : BoxMCMoveSize tuner
    sdf_compute  : hpmc.compute.SDF (or None if enable_sdf=False)
    sim_log_hdl  : open file handle for simulation Table log
    box_log_hdl  : open file handle for box Table log

    The caller (main) is responsible for:
      - Running the equilibration loop and removing box_tuner when converged.
      - Attaching sdf_compute after equilibration.
      - Running the production phase.
      - Closing sim_log_hdl and box_log_hdl in the finally block.

    Parameters
    ----------
    params : SimulationParams
        Validated simulation parameters.
    files : RunFiles
        Resolved filenames for this stage.
    seed : int
        Random seed for HOOMD's RNG (in [0, 65535]).
    comm
        MPI communicator.
    rank : int
        MPI rank of the calling process.

    Returns
    -------
    tuple : (sim, mc, boxmc, trans_move_tuner, rot_move_tuner, box_tuner,
             sdf_compute, sim_log_hdl, box_log_hdl)
    """

    # ------------------------------------------------------------------
    # 9.1  Device 
    # ------------------------------------------------------------------
    if params.use_gpu:
        try:
            device = hoomd.device.GPU(gpu_ids=[params.gpu_id])
            root_print(f"[INFO] Running on GPU {params.gpu_id}")
        except Exception as exc:
            root_print(f"[WARNING] GPU init failed ({exc}). Falling back to CPU.")
            device = hoomd.device.CPU()
    else:
        device = hoomd.device.CPU()
        root_print("[INFO] Running on CPU")

    # ------------------------------------------------------------------
    # 9.2  Simulation object
    # ------------------------------------------------------------------
    sim = hoomd.Simulation(device=device, seed=seed)

    # ------------------------------------------------------------------
    # 9.3  State initialisation  
    # ------------------------------------------------------------------
    # Determine run mode by checking which GSD files exist on disk.
    # Meaningful combinations:
    #   restart=True,  final=False  => crashed/walltime restart — resume from checkpoint
    #   restart=True,  final=True   => completed run — treat as fresh (overwrite)
    #   restart=False, final=False  => clean first run
    #   restart=False, final=True   => re-running completed stage — treat as fresh
    restart_exists = Path(files.restart_gsd).exists()
    final_exists   = Path(files.final_gsd).exists()

    debug_kv(
        "State initialisation decision",
        restart_exists=restart_exists,
        final_exists=final_exists,
        input_gsd=files.input_gsd,
        restart_gsd=files.restart_gsd,
        final_gsd=files.final_gsd,
    )

    if restart_exists and not final_exists:
        root_print(f"[INFO] Restart GSD found: '{files.restart_gsd}'. Resuming.")
        try:
            sim.create_state_from_gsd(filename=files.restart_gsd)
        except Exception as exc:
            fail_with_context(
                "Failed to create HOOMD state from restart GSD.",
                restart_gsd=files.restart_gsd,
                error_type=type(exc).__name__,
                error=str(exc),
                hint="Restart GSD may be corrupted.  Delete it to start fresh.",
            )
        state_source = files.restart_gsd
        root_print("[INFO] Restart run: True")
    else:
        # FRESH RUN PATH: either no restart GSD exists, or both restart and
        # final exist (completed run being re-run intentionally).
        if final_exists:
            root_print(
                f"[WARNING] Final GSD '{files.final_gsd}' already exists; starting fresh."
            )
        root_print(f"[INFO] Fresh run from: '{files.input_gsd}'")
        # Fresh runs use the MPI-safe broadcast pattern:
        #  1. rank 0 reads the last frame from the input GSD
        #  2. data dict is broadcast to all MPI ranks
        #  3. all ranks reconstruct an identical hoomd.Snapshot
        #  4. HOOMD distributes particles into MPI domains internally
        try:
            snap_data = load_and_broadcast_snapshot(files.input_gsd, comm, rank)
            snapshot  = reconstruct_snapshot(snap_data, rank)
            sim.create_state_from_snapshot(snapshot)
        except SystemExit:
            raise
        except Exception as exc:
            fail_with_context(
                "Failed during fresh-run state creation from input GSD.",
                input_gsd=files.input_gsd,
                rank=rank,
                error_type=type(exc).__name__,
                error=str(exc),
            )

        state_source = files.input_gsd
        root_print("[INFO] Restart run: False")

    # ------------------------------------------------------------------
    # 9.4  Read diameter
    # ------------------------------------------------------------------
    shape_info = load_convex_polyhedron_shape(
        params.shape_json_filename,
        params.shape_scale,
    )
    params.particle_volume = shape_info["particle_volume"]

    # Compute initial packing fraction for the startup log message.
    N   = sim.state.N_particles
    phi = N * params.particle_volume / sim.state.box.volume

    root_print(
        f"[INFO] Shape file = {shape_info['shape_json_path']} | "
        f"reference_volume = {shape_info['reference_volume']} | "
        f"shape_scale = {params.shape_scale} | "
        f"particle_volume = {params.particle_volume}"
    )
    root_print(f"[INFO] N={N} | phi={phi:.6f}")

    box = sim.state.box
    root_print(f"[INFO] Box: Lx={box.Lx:.4f} Ly={box.Ly:.4f} Lz={box.Lz:.4f} "
               f"xy={box.xy} xz={box.xz} yz={box.yz}")
    root_print(f"[INFO] Target betaP = {params.pressure}")

    # ------------------------------------------------------------------
    # 9.5  HPMC ConvexPolyhedron integrator 
    # ------------------------------------------------------------------
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(nselect=1)

    mc.shape["A"] = {"vertices": shape_info["vertices"]}
    mc.d["A"] = params.move_size_translation
    mc.a["A"] = params.move_size_rotation

    sim.operations.integrator = mc
    # **** Parallel Section ****
    # MC moves are distributed across MPI domains; each rank handles its own
    # sub-domain and exchanges ghost-particle data with neighbours.

    # Force attachment and perform explicit overlap check
    sim.run(0)

    # ---- Initial overlap check ----
    # A valid hard-particle configuration must have ZERO overlaps.  Any
    # non-zero count here means the input GSD contains overlapping particles,
    # which would make the simulation physically invalid.
    initial_overlaps = int(mc.overlaps)
    if initial_overlaps != 0:
        fail_with_context(
            "Initial convex-polyhedron configuration is NOT overlap-free.",
            input_state_source=files.input_gsd,
            shape_json_filename=params.shape_json_filename,
            shape_scale=params.shape_scale,
            initial_overlaps=initial_overlaps,
        )

    root_print(
        f"[INFO] HPMC ConvexPolyhedron attached | "
        f"d_init={params.move_size_translation} | "
        f"a_init={params.move_size_rotation}"
    )
    root_print("[INFO] HOOMD overlap check passed: initial overlap count = 0")

    # ------------------------------------------------------------------
    # 9.6  Custom loggable instances
    # ------------------------------------------------------------------
    # Each class exposes one or more properties via _export_dict.  HOOMD's
    # Logger calls each property getter once at registration time to
    # validate the loggable — this is why every getter guards against
    status        = Status(sim)
    mc_status     = MCStatus(mc)
    box_property = Box_property(sim, N, params.particle_volume)
    move_prop     = MoveSizeProp(mc, ptype="A")
    overlap_count = OverlapCount(mc)

    # ------------------------------------------------------------------
    # 9.7  Logger: simulation log (Table)
    # ------------------------------------------------------------------
    # This logger collects per-step scalars and strings for the main
    # human-readable Table log and for embedding in the trajectory GSD.
    # categories=["scalar","string"]: only these two categories are needed;
    logger_sim = hoomd.logging.Logger(categories=["scalar", "string"], only_default=False)
    # Built-in HOOMD quantities: tps (steps/s), walltime (elapsed wall s),
    # timestep (current step counter as integer scalar)
    logger_sim.add(sim, quantities=["tps", "walltime", "timestep"])
    logger_sim[("Status",    "etr")]          = (status,        "etr",              "string")
    logger_sim[("Status",    "timestep")]     = (status,        "timestep_fraction","string")
    logger_sim[("MCStatus", "trans_acc_rate")]= (mc_status, "translate_acceptance_rate", "scalar")
    logger_sim[("MCStatus", "rot_acc_rate")]  = (mc_status, "rotate_acceptance_rate", "scalar")
    logger_sim[("MoveSize",  "d")]            = (move_prop,     "d",                "scalar")
    logger_sim[("MoveSize",  "a")]            = (move_prop,     "a",                "scalar")
    logger_sim[("Box",       "volume")]       = (box_property,  "volume",           "scalar")
    logger_sim[("Box",       "phi")]          = (box_property,  "packing_fraction", "scalar") 
    logger_sim[("HPMC",      "overlaps")]     = (overlap_count, "overlap_count",    "scalar")  

    # ------------------------------------------------------------------
    # 9.8  Logger: box log (Table)
    # ------------------------------------------------------------------
    # A separate box logger and Table writer records box geometry at every
    # log_frequency steps.
    logger_box = hoomd.logging.Logger(categories=["scalar", "string"], only_default=False)
    logger_box.add(sim, quantities=["timestep"])
    # Register the six independent box parameters (Lx, Ly, Lz, xy, xz, yz)
    for k, v in {"l_x":"L_x","l_y":"L_y","l_z":"L_z",
                 "XY":"XY","XZ":"XZ","YZ":"YZ"}.items():
        logger_box[("Box_property", k)] = (box_property, v, "scalar")
    logger_box[("Box", "volume_str")] = (box_property, "volume_str", "string")
    logger_box[("Box", "phi")]        = (box_property, "packing_fraction", "scalar") 

    logger_particle = hoomd.logging.Logger()
    logger_particle.add(mc, quantities=["type_shapes"])

    # ------------------------------------------------------------------
    # 9.9  Open Table log file handles  
    #      + explicit writer-specific exception handling for easier debugging
    # ------------------------------------------------------------------
    # ---- Simulation Table log ----
    try:
        sim_log_hdl = open(files.sim_log_txt, "w")
    except OSError as exc:
        sys.exit(
            "[FATAL] Could not open simulation log file for writing.\n"
            f"  sim_log_file: {files.sim_log_txt}\n"
            f"  error: {repr(exc)}"
        )

    try:
        simulation_log_writer = hoomd.write.Table(
            output=sim_log_hdl,
            trigger=hoomd.trigger.Periodic(params.log_frequency),
            logger=logger_sim,
        )
        sim.operations.writers.append(simulation_log_writer)
    except Exception as exc:
        try:
            sim_log_hdl.close()
        except Exception:
            pass
        sys.exit(
            "[FATAL] Failed to create/attach simulation Table writer.\n"
            f"  sim_log_file: {files.sim_log_txt}\n"
            f"  log_frequency: {params.log_frequency}\n"
            f"  error_type: {type(exc).__name__}\n"
            f"  error: {exc}"
        )

    try:
        box_log_hdl = open(files.box_log_txt, "w")
    except OSError as exc:
        try:
            sim_log_hdl.close()
        except Exception:
            pass
        sys.exit(
            "[FATAL] Could not open box log file for writing.\n"
            f"  box_log_file: {files.box_log_txt}\n"
            f"  error: {repr(exc)}"
        )

    try:
        box_log_writer = hoomd.write.Table(
            output=box_log_hdl,
            trigger=hoomd.trigger.Periodic(params.log_frequency),
            logger=logger_box,
        )
        sim.operations.writers.append(box_log_writer)
    except Exception as exc:
        try:
            box_log_hdl.close()
        except Exception:
            pass
        try:
            sim_log_hdl.close()
        except Exception:
            pass
        sys.exit(
            "[FATAL] Failed to create/attach box Table writer.\n"
            f"  box_log_file: {files.box_log_txt}\n"
            f"  log_frequency: {params.log_frequency}\n"
            f"  error_type: {type(exc).__name__}\n"
            f"  error: {exc}"
        )

    # ------------------------------------------------------------------
    # 9.10  GSD trajectory writer  
    # ------------------------------------------------------------------
    try:
        traj_gsd_writer = hoomd.write.GSD(
            filename=files.traj_gsd,
            filter=hoomd.filter.All(),
            trigger=hoomd.trigger.Periodic(params.traj_gsd_frequency),
            mode="ab",                           # append: safe on restart
            dynamic=["property", "attribute"],   # writes diameter each frame
            logger=logger_particle,                   # store scalars in GSD too
        )
        sim.operations.writers.append(traj_gsd_writer)
    except Exception as exc:
        sys.exit(
            "[FATAL] Failed to create/attach trajectory GSD writer.\n"
            f"  traj_gsd: {files.traj_gsd}\n"
            f"  traj_frequency: {params.traj_gsd_frequency}\n"
            f"  mode: ab\n"
            f"  error_type: {type(exc).__name__}\n"
            f"  error: {exc}"
        )

    # ------------------------------------------------------------------
    # 9.11  GSD restart writer 
    # ------------------------------------------------------------------
    try:
        restart_gsd_writer = hoomd.write.GSD(
            filename=files.restart_gsd,
            filter=hoomd.filter.All(),
            trigger=hoomd.trigger.Periodic(params.restart_gsd_frequency),
            truncate=True,
            mode="wb",
            dynamic=["property", "attribute"],
            logger=logger_particle,
        )
        sim.operations.writers.append(restart_gsd_writer)
    except Exception as exc:
        sys.exit(
            "[FATAL] Failed to create/attach restart GSD writer.\n"
            f"  restart_gsd: {files.restart_gsd}\n"
            f"  restart_frequency: {params.restart_gsd_frequency}\n"
            f"  truncate: True\n"
            f"  mode: wb\n"
            f"  error_type: {type(exc).__name__}\n"
            f"  error: {exc}"
        )

    # ------------------------------------------------------------------
    # 9.12  GSD scalar log (separate file, filter=Null)
    # ------------------------------------------------------------------
    # A second GSD writer stores only logged scalars — no particle data.
    try:
        scalar_gsd_logger = hoomd.logging.Logger(categories=["scalar", "string"], only_default=False)
        scalar_gsd_logger.add(sim, quantities=["timestep", "tps", "walltime"])

        scalar_gsd_logger[("Box",      "volume")]        = (box_property,  "volume",                  "scalar")
        scalar_gsd_logger[("Box",      "phi")]           = (box_property,  "packing_fraction",        "scalar")

        scalar_gsd_logger[("MCStatus", "trans_acc_rate")] = (mc_status, "translate_acceptance_rate", "scalar")
        scalar_gsd_logger[("MCStatus", "rot_acc_rate")]   = (mc_status, "rotate_acceptance_rate",    "scalar")

        scalar_gsd_logger[("MoveSize", "d")]             = (move_prop,     "d",                       "scalar")
        scalar_gsd_logger[("MoveSize", "a")]             = (move_prop,     "a",                       "scalar")

        scalar_gsd_logger[("HPMC",     "overlaps")]      = (overlap_count, "overlap_count",           "scalar")

        scalar_gsd_writer = hoomd.write.GSD(
            filename=files.scalar_gsd_log,
            filter=hoomd.filter.Null(),   # no particle data -> tiny file
            trigger=hoomd.trigger.Periodic(params.log_frequency),
            mode="ab",
            logger=scalar_gsd_logger,
        )
        sim.operations.writers.append(scalar_gsd_writer)
    except Exception as exc:
        sys.exit(
            "[FATAL] Failed to create/attach scalar GSD log writer.\n"
            f"  scalar_gsd_log: {files.scalar_gsd_log}\n"
            f"  log_frequency: {params.log_frequency}\n"
            f"  filter: Null\n"
            f"  mode: ab\n"
            f"  error_type: {type(exc).__name__}\n"
            f"  error: {exc}"
        )

    # ------------------------------------------------------------------
    # 9.13  BoxMC updater 
    # ------------------------------------------------------------------
    # BoxMC implements the NPT ensemble by proposing random trial changes
    # to the simulation box and accepting/rejecting them via the NPT
    # Metropolis criterion, which includes the P*delat(V) work term and the
    # N*ln(V_new/V_old) Jacobian.
    # betaP = P/(kBT) is the dimensionless reduced pressure.
    try:
        betaP_variant = hoomd.variant.Constant(params.pressure)
        boxmc = hoomd.hpmc.update.BoxMC(
            trigger=hoomd.trigger.Periodic(params.npt_freq),
            betaP=betaP_variant,
        )

        # Length moves: propose independent changes to Lx, Ly, Lz at
        # constant shape (tilt factors unchanged).
        boxmc.length = dict(
            delta=(params.boxmc_length_delta,  # max |delta Lx| per attempt
                params.boxmc_length_delta,     # max |delta Ly| per attempt
                params.boxmc_length_delta),    # max |delta Lz| per attempt
            weight=1.0,
        )

        # Shear moves: change the tilt factors xy, xz, yz.
        # reduce=0.0 disables Lees-Edwards-style lattice-vector reduction;
        # standard for fluid and solid systems without shear flow.
        boxmc.shear = dict(
            delta=(params.boxmc_shear_delta,   # max |delta xy| per attempt
                params.boxmc_shear_delta,      # max |delta xz| per attempt
                params.boxmc_shear_delta),     # max |delta yz| per attempt
            weight=1.0,
            reduce=0.0,
        )

        sim.operations.updaters.append(boxmc)

    except Exception as exc:
        fail_with_context(
            "Failed while configuring BoxMC updater.",
            pressure=params.pressure,
            npt_freq=params.npt_freq,
            length_delta=params.boxmc_length_delta,
            shear_delta=params.boxmc_shear_delta,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    root_print(
        f"[INFO] BoxMC: betaP={params.pressure} | "
        f"length_delta={params.boxmc_length_delta} | "
        f"shear_delta={params.boxmc_shear_delta}"
    )

    # ------------------------------------------------------------------
    # 9.14  Register BoxMC loggers (after boxmc exists)
    # ------------------------------------------------------------------
    # Register BoxMCStatus and BoxSeqProp loggables — they require a reference
    # to the BoxMC updater so must be created after step 9.13.
    # ------------------------------------------------------------------
    box_mc_status = BoxMCStatus(boxmc, sim)
    logger_sim[("BoxMCStatus", "acc_rate")]          = (box_mc_status, "acceptance_rate",    "scalar")
    logger_sim[("BoxMCStatus", "length_acc_rate")]   = (box_mc_status, "length_acc_rate",    "scalar")
    logger_sim[("BoxMCStatus", "shear_acc_rate")]    = (box_mc_status, "shear_acc_rate",     "scalar")
    logger_sim[("BoxMCStatus", "length_moves")]      = (box_mc_status, "length_moves_str",   "string")
    logger_sim[("BoxMCStatus", "shear_moves")]       = (box_mc_status, "shear_moves_str",    "string")
    logger_sim[("BoxMCStatus", "combined_moves")]    = (box_mc_status, "combined_moves_str", "string")

    # Also log BoxMC rates in the compact scalar GSD.  GSD scalar log keeps
    # numeric rates only; raw string counters are written to the text logs.
    scalar_gsd_logger[("BoxMCStatus", "acc_rate")]        = (box_mc_status, "acceptance_rate",  "scalar")
    scalar_gsd_logger[("BoxMCStatus", "length_acc_rate")] = (box_mc_status, "length_acc_rate",  "scalar")
    scalar_gsd_logger[("BoxMCStatus", "shear_acc_rate")]  = (box_mc_status, "shear_acc_rate",   "scalar")

    seq_prop = BoxSeqProp(boxmc)
    logger_box[("BoxMC", "length_moves")]    = (seq_prop, "length_moves_str",   "string")
    logger_box[("BoxMC", "shear_moves")]     = (seq_prop, "shear_moves_str",    "string")
    logger_box[("BoxMC", "combined_moves")]  = (seq_prop, "combined_moves_str", "string")

    # ------------------------------------------------------------------
    # 9.15  MoveSize tuner for particle moves  
    # ------------------------------------------------------------------
    # MoveSize.scale_solver adjusts mc.d["A"] every trans_move_size_tuner_freq
    # steps by multiplying it by a scale factor so the translational acceptance
    # rate converges toward target_particle_trans_move_acc_rate.
    # ******This tuner is NOT removed after equilibration***** — it remains active during
    # production so d continues to track the slowly changing acceptance rate
    # as box volume fluctuates in the NPT ensemble.
    try:
        trans_move_tuner = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["d"],
            target=params.target_particle_trans_move_acc_rate,
            trigger=hoomd.trigger.Periodic(params.trans_move_size_tuner_freq),
            max_translation_move=params.max_translation_move,
        )
        sim.operations.tuners.append(trans_move_tuner)
    except Exception as exc:
        fail_with_context(
            "Failed to create translational MoveSize tuner.",
            target=params.target_particle_trans_move_acc_rate,
            tuner_freq=params.trans_move_size_tuner_freq,
            max_translation_move=params.max_translation_move,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    try:
        rot_move_tuner = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a"],
            target=params.target_particle_rot_move_acc_rate,
            trigger=hoomd.trigger.Periodic(params.rot_move_size_tuner_freq),
            max_rotation_move=params.max_rotation_move,
        )
        sim.operations.tuners.append(rot_move_tuner)
    except Exception as exc:
        fail_with_context(
            "Failed to create rotational MoveSize tuner.",
            target=params.target_particle_rot_move_acc_rate,
            tuner_freq=params.rot_move_size_tuner_freq,
            max_rotation_move=params.max_rotation_move,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    # ------------------------------------------------------------------
    # 9.16  BoxMCMoveSize tuner  
    # ------------------------------------------------------------------
    # BoxMCMoveSize.scale_solver adjusts the delta parameters for every
    # listed box-move type every box_tuner_freq steps so each move's
    # acceptance rate converges toward target_box_movement_acc_rate.
    # gamma=0.8: how aggressively the tuner scales delta.  Values closer
    #   to 1.0 are more aggressive; HOOMD documentation recommends 0.8.
    # tol=0.01: the tuner declares itself converged (box_tuner.tuned=True)
    #   when the acceptance ratio is within +-0.01 (1%) of the target.
    #   The equilibration loop polls box_tuner.tuned and removes this tuner
    #   once converged, locking in the optimal box-move sizes for production.
    # max_move_size: upper bounds prevent the tuner from setting excessively
    #   large deltas that would cause sudden large box changes and temporarily
    #   break MPI domain decomposition or produce many overlaps.
    box_tuner = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(
        trigger=hoomd.trigger.Periodic(params.box_tuner_freq),
        boxmc=boxmc,
        moves=["length_x", "length_y", "length_z",
               "shear_x",  "shear_y",  "shear_z"],
        target=params.target_box_movement_acc_rate,
        max_move_size={
            "length_x": params.max_move_length,
            "length_y": params.max_move_length,
            "length_z": params.max_move_length,
            "shear_x":  params.max_move_shear,
            "shear_y":  params.max_move_shear,
            "shear_z":  params.max_move_shear,
        },
        gamma=0.8,   # tuner aggressiveness (HOOMD recommendation)
        tol=0.03,    # convergence tolerance
    )
    sim.operations.tuners.append(box_tuner)
    root_print(
        f"[INFO] Tuners: MoveSize (every {params.trans_move_size_tuner_freq} steps) | "
        f"BoxMCMoveSize (every {params.box_tuner_freq} steps)"
    )

    # ------------------------------------------------------------------
    # 9.17  SDF pressure compute 
    #
    # hoomd.hpmc.compute.SDF computes the pressure by measuring the scale
    # distribution function — the distribution of the smallest scale factor
    # x by which you would need to expand particles before a collision
    # occurs.  Extrapolating s(x) to x=0 gives the equation of state:
    #
    #   \beta P = \rho * (1 + s(0+) / 6)     [hard spheres in 3D]
    #
    # The loggable `betaP` property gives the instantaneous pressure in
    # units of 1/(length^3).  In NPT the box fluctuates, so the average
    # <betaP> over the production run should converge to the target value
    # `pressure`. 
    #
    # Parameters (from the JSON):
    #   sdf_xmax : upper limit of the histogram 
    #   sdf_dx   : histogram bin width (1e-4 gives good resolution)
    #
    # SDF is only active after equilibration (when the box has converged)
    # because the SDF calculation assumes a fixed box.  We attach it in
    # the production phase only.  It is returned here so the caller can
    # attach it after removing the box tuner.
    # ------------------------------------------------------------------
    sdf_compute = None
    if params.enable_sdf:
        sdf_compute = hoomd.hpmc.compute.SDF(
            xmax=params.sdf_xmax,
            dx=params.sdf_dx,
        )
        # Do NOT attach yet — will be added after equilibration

        sdf_pressure_logger = SDFPressure(sdf_compute, sim, N)
        logger_sim[("SDF", "betaP")]             = (sdf_pressure_logger, "betaP",             "scalar")
        logger_sim[("SDF", "compressibility_Z")] = (sdf_pressure_logger, "compressibility_Z", "scalar")
        scalar_gsd_logger[("SDF", "betaP")]      = (sdf_pressure_logger, "betaP",             "scalar")
        scalar_gsd_logger[("SDF", "Z")]          = (sdf_pressure_logger, "compressibility_Z", "scalar")

        root_print(
            f"[INFO] SDF compute configured: "
            f"xmax={params.sdf_xmax}, dx={params.sdf_dx}. "
            f"Will be attached after equilibration."
        )

    return (sim, mc, boxmc, trans_move_tuner, rot_move_tuner, box_tuner, sdf_compute, sim_log_hdl, box_log_hdl)


# ===========================================================================
#  SECTION 10 — Output helpers
# ===========================================================================

def _write_snapshot(sim: hoomd.Simulation, mc, filename: str) -> None:
    """
    Write the current simulation state as a single-frame GSD file.
    """
    snapshot = sim.state.get_snapshot()

    # In MPI, particle data in get_snapshot() is present only on rank 0.
    if not _is_root_rank():
        return

    frame = gsd.hoomd.Frame()

    # -----------------------------
    # Box
    # -----------------------------
    box = snapshot.configuration.box
    frame.configuration.box = list(box)

    # -----------------------------
    # Particle data
    # -----------------------------
    N = int(snapshot.particles.N)
    frame.particles.N = N
    frame.particles.types = list(snapshot.particles.types)
    frame.particles.typeid = np.array(snapshot.particles.typeid, dtype=np.uint32)
    frame.particles.position = np.array(snapshot.particles.position, dtype=np.float32)
    frame.particles.orientation = np.array(snapshot.particles.orientation, dtype=np.float32)
    frame.particles.type_shapes = mc.type_shapes

    with gsd.hoomd.open(filename, mode="w") as traj:
        traj.append(frame)



def _boxmc_counter_counts(moves) -> tuple[int, int, int]:
    """Convert a HOOMD (accepted, rejected) counter to (accepted, rejected, total)."""
    try:
        # HOOMD counters are returned as accepted/rejected values.
        accepted, rejected = moves
        accepted = int(accepted)
        rejected = int(rejected)
        total = accepted + rejected
        return accepted, rejected, total
    except (TypeError, ValueError, DataAccessError):
        return 0, 0, 0


def _safe_acceptance_rate(moves) -> float:
    """
    Convert a HOOMD (accepted, rejected) move counter into an acceptance rate.

    Returns 0.0 if no moves were attempted in the current run chunk.  In
    HOOMD-blue v4, BoxMC counters are reset at the start of each sim.run()
    call, so after each equilibration chunk these rates correspond to that
    chunk only.
    """
    accepted, rejected, total = _boxmc_counter_counts(moves)
    return accepted / total if total > 0 else 0.0


def _tuple3_from_delta(delta) -> tuple[float, float, float]:
    """Return a 3-tuple of floats from a scalar or sequence delta."""
    if isinstance(delta, (int, float)):
        val = float(delta)
        return (val, val, val)
    vals = tuple(float(x) for x in delta)
    if len(vals) != 3:
        raise ValueError(f"Expected a 3-component delta, got {delta!r}")
    return vals


def _capture_box_move_state(boxmc, sim, target_acceptance: float) -> dict:

    """
    Capture the current BoxMC move sizes and current acceptance diagnostics.

    This function is called after each equilibration chunk.

    Its purpose is to remember:

        1. current length_delta
        2. current shear_delta
        3. current length/shear weights
        4. current length acceptance rate
        5. current shear acceptance rate
        6. current combined box acceptance rate
        7. raw accepted/rejected/attempted counts
        8. distance from the target box acceptance rate

    Parameters
    ----------
    boxmc
        The active hoomd.hpmc.update.BoxMC object.

    sim
        The active hoomd.Simulation object. Used here only to record the
        current timestep.

    target_acceptance
        Desired box move acceptance rate from the simulation parameter file,
        e.g. params.target_box_movement_acc_rate.

    Returns
    -------
    dict
        A complete snapshot of current BoxMC move sizes and acceptance data.
    """

    # ---------------------------------------------------------------------
    # 1. Read current BoxMC length and shear parameter dictionaries
    # ---------------------------------------------------------------------
    #
    # boxmc.length is dict-like and contains fields such as:
    #
    #     {
    #         "delta":  (dx, dy, dz),
    #         "weight": weight_value,
    #     }
    #
    # boxmc.shear is dict-like and contains fields such as:
    #
    #     {
    #         "delta":  (dxy, dxz, dyz),
    #         "weight": weight_value,
    #         "reduce": reduce_value,
    #     }
    #
    length_params = dict(boxmc.length)
    shear_params = dict(boxmc.shear)

    # ---------------------------------------------------------------------
    # 2. Read raw BoxMC counters
    # ---------------------------------------------------------------------
    # HOOMD v4 exposes accepted/rejected volume and length moves together through boxmc.volume_moves. Since this code enables BoxMC.length moves
    # and does not enable BoxMC.volume moves separately, boxmc.volume_moves is effectively the length-move counter in this run.
    # shear attempts are exposed separately through boxmc.shear_moves.
    
    length_acc, length_rej, length_total = _boxmc_counter_counts(boxmc.volume_moves)
    shear_acc, shear_rej, shear_total = _boxmc_counter_counts(boxmc.shear_moves)

    # ---------------------------------------------------------------------
    # 3. Combine length-like and shear counters
    # ---------------------------------------------------------------------
    #
    # The combined box acceptance rate is computed from all accepted/rejected
    # BoxMC moves included in these two counter families.
    total_acc = length_acc + shear_acc
    total_rej = length_rej + shear_rej
    total = total_acc + total_rej

    # ---------------------------------------------------------------------
    # 4. Compute acceptance rates
    # ---------------------------------------------------------------------
    length_rate = length_acc / length_total if length_total > 0 else 0.0
    shear_rate = shear_acc / shear_total if shear_total > 0 else 0.0
    combined_rate = total_acc / total if total > 0 else 0.0


    # ---------------------------------------------------------------------
    # 5. Return a complete dictionary snapshot
    # ---------------------------------------------------------------------
    #
    # This dictionary is stored as either:
    #
    #     current_box_move_state
    #
    # or, if it is the best so far:
    #
    #     best_box_move_state
    #
    # Later, if BoxMCMoveSize does not converge, best_box_move_state is passed
    # to _restore_box_move_state(...).

    return {
        # Current HOOMD timestep when this state was captured.
        "timestep": int(sim.timestep),
        # Current length move size for length_x, length_y, length_z.
        "length_delta": _tuple3_from_delta(length_params.get("delta", (0.0, 0.0, 0.0))),
        # Current statistical weight for length moves.
        "length_weight": float(length_params.get("weight", 0.0)),
        # Current shear move size for shear_x, shear_y, shear_z.
        "shear_delta": _tuple3_from_delta(shear_params.get("delta", (0.0, 0.0, 0.0))),
        # Current statistical weight for shear moves.
        "shear_weight": float(shear_params.get("weight", 0.0)),
        "shear_reduce": float(shear_params.get("reduce", 0.0)),

        # Acceptance rates from the same raw counters.
        "length_acceptance_rate": float(length_rate),
        "shear_acceptance_rate": float(shear_rate),
        "combined_acceptance_rate": float(combined_rate),

        # Raw accepted/rejected/total counts for debugging.
        "length_accepts": int(length_acc),
        "length_rejects": int(length_rej),
        "length_attempts": int(length_total),
        "shear_accepts": int(shear_acc),
        "shear_rejects": int(shear_rej),
        "shear_attempts": int(shear_total),
        "combined_accepts": int(total_acc),
        "combined_rejects": int(total_rej),
        "combined_attempts": int(total),

        "target_acceptance_rate": float(target_acceptance),
        "score_abs_error": abs(float(combined_rate) - float(target_acceptance)),
    }


def _restore_box_move_state(boxmc, state: dict) -> None:
    """Restore BoxMC length/shear move sizes from a captured state."""
    boxmc.length = dict(
        delta=tuple(state["length_delta"]),
        weight=float(state["length_weight"]),
    )
    boxmc.shear = dict(
        delta=tuple(state["shear_delta"]),
        weight=float(state["shear_weight"]),
        reduce=float(state["shear_reduce"]),
    )


def _tuner_is_attached(sim, tuner) -> bool:
    """Identity-based test for whether a tuner is still attached."""
    return any(obj is tuner for obj in sim.operations.tuners)


def _remove_tuner_if_attached(sim, tuner) -> bool:
    """Remove a tuner if attached; return True if removal occurred."""
    if _tuner_is_attached(sim, tuner):
        sim.operations.tuners.remove(tuner)
        return True
    return False

def write_final_outputs(
    sim:        hoomd.Simulation,
    mc:         hoomd.hpmc.integrate.ConvexPolyhedron,
    boxmc:      hoomd.hpmc.update.BoxMC,
    params:     SimulationParams,
    files:      RunFiles,
    start_time: float,
    seed:       int,
    json_path:  str,
    box_tuning_result: Optional[dict] = None,
) -> None:
    """
    Write all final output files after the production run completes.

    Outputs:
    1. Final GSD snapshot (files.final_gsd) — single-frame GSD with the
       last configuration, suitable as input to the next pipeline stage.
    2. Summary JSON (<tag>_stage<id>_npt_summary.json) — machine-readable
       provenance record containing all key parameters, final box geometry,
       packing fraction, overlap count, and runtime.  Written only on rank 0.
    3. Console banner — human-readable run summary printed to stdout.

    Parameters
    ----------
    sim : hoomd.Simulation
        Completed simulation.
    mc : hoomd.hpmc.integrate.ConvexPolyhedron
        HPMC integrator (for final overlap count and type_shapes).
    boxmc : hoomd.hpmc.update.BoxMC
        BoxMC updater (retained for potential future logging).
    params : SimulationParams
        All simulation parameters (must have particle_volume set).
    files : RunFiles
        Resolved filenames for this stage.
    start_time : float
        Wall-clock start time from time.time() in main().
    seed : int
        Random seed used for this run (for provenance).
    json_path : str
        Path to the simulation parameter JSON file (for provenance).
    box_tuning_result : dict or None
        Provenance for the BoxMCMoveSize decision made before production.
    """

    N   = sim.state.N_particles
    phi = N * params.particle_volume / sim.state.box.volume

    # Final GSD snapshot
    _write_snapshot(sim, mc, files.final_gsd)
    root_print(f"[OUTPUT] Final GSD       => {files.final_gsd}")

    # Machine-readable summary
    # Captures all key provenance information so results can be traced to
    # exact input parameters without opening the original JSON file.
    runtime = time.time() - start_time
    summary = {
        "tag":               params.tag,
        "stage_id":          params.stage_id_current,
        "simulparam_file":   json_path,
        "input_gsd":         files.input_gsd,
        "final_gsd":         files.final_gsd,
        "n_particles":       N,
        "shape_json_filename": params.shape_json_filename,
        "shape_scale":         params.shape_scale,
        "particle_volume":     params.particle_volume,
        "packing_fraction":  round(phi, 8),
        "target_betaP":      params.pressure,
        "overlaps_final":    mc.overlaps,
        "final_timestep":    sim.timestep,
        "random_seed":       seed,
        "box_final": {
            "Lx": sim.state.box.Lx,
            "Ly": sim.state.box.Ly,
            "Lz": sim.state.box.Lz,
            "xy": sim.state.box.xy,
            "xz": sim.state.box.xz,
            "yz": sim.state.box.yz,
        },
        "box_tuning_result": box_tuning_result,
        "runtime_seconds":   round(runtime, 2),
        "created":           time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_file = f"{params.tag}_stage{params.stage_id_current}_npt_summary.json"
    if _is_root_rank():
        try:
            with open(summary_file, "w") as fh:
                json.dump(summary, fh, indent=2)
            root_print(f"[OUTPUT] Summary JSON    => {summary_file}")
        except OSError as exc:
            root_print(
                f"[WARNING] Could not write summary JSON: "
                f"{type(exc).__name__}: {exc}"
            )

    # Console banner
    _print_banner("HOOMD-blue v4.9 | Convex-Polyhedron NPT | Run Complete")
    root_print("\n".join([
        f"  Simulparam file       : {json_path}",
        f"  Tag                   : {params.tag}",
        f"  Stage id              : {params.stage_id_current}",
        f"  Input GSD             : {files.input_gsd}",
        f"  Final GSD             : {files.final_gsd}",
        f"  Particles (N)         : {N}",
        f"  Shape JSON            : {params.shape_json_filename}",
        f"  Shape scale           : {params.shape_scale}",
        f"  Particle volume       : {params.particle_volume}",
        f"  Final packing frac.   : {phi:.6f}",
        f"  Target betaP          : {params.pressure}",
        f"  Box tuning mode       : {(box_tuning_result or {}).get('mode', 'unknown')}",
        f"  Box tuner in production: {(box_tuning_result or {}).get('box_tuner_attached_during_production', 'unknown')}",
        f"  Overlaps at end       : {mc.overlaps}",
        f"  Final timestep        : {sim.timestep}",
        f"  Total timesteps       : {params.total_num_timesteps}",
        f"  Random seed           : {seed}",
        f"  Total runtime         : {runtime:.2f} s",
    ]))
    _print_banner("")


def _print_banner(title: str, width: int = 72) -> None:
    bar = "*" * width
    root_print(f"\n{bar}")
    if title:
        root_print(f"  {title}")
        root_print(bar)


# ===========================================================================
#  SECTION 11 — Entry point
# ===========================================================================

_active_json_path: str = ""


def main() -> None:
    global _active_json_path

    start_time = time.time()

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "HOOMD-blue v4.9 hard-ConvexPolyhedron HPMC NPT simulation.\n"
            "Usage: python hard_polyhedra_NPT.py --simulparam_file simulparam.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--simulparam_file", required=True, metavar="FILE")
    args = parser.parse_args()
    _active_json_path = args.simulparam_file

    # HOOMD version banner (preserved from original)
    print("*********************************************************")
    print("HOOMD-blue version:  ", hoomd.version.version)
    print("*********************************************************")
    print("hoomd.version.mpi_enabled:   ", hoomd.version.mpi_enabled)

    _print_banner("HOOMD-blue v4.9 | Hard-ConvexPolyhedron HPMC NPT")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    params = load_simulparams(args.simulparam_file)
    root_print(f"[INFO] Loaded parameters from '{args.simulparam_file}'")
    root_print(f"  tag                 = {params.tag}")
    root_print(f"  stage_id_current    = {params.stage_id_current}")
    root_print(f"  total_num_timesteps = {params.total_num_timesteps}")
    root_print(f"  equil_steps         = {params.equil_steps}")
    root_print(f"  pressure (betaP)    = {params.pressure}")

    # ------------------------------------------------------------------
    # MPI
    # ------------------------------------------------------------------
    comm      = MPI.COMM_WORLD
    _env_rank = _mpi_rank_from_env()

    # ------------------------------------------------------------------
    # Filenames
    # ------------------------------------------------------------------
    files = resolve_filenames(params)
    root_print(f"[INFO] Input GSD    : {files.input_gsd}")
    root_print(f"[INFO] Traj GSD     : {files.traj_gsd}")
    root_print(f"[INFO] Restart GSD  : {files.restart_gsd}")
    root_print(f"[INFO] Final GSD    : {files.final_gsd}")
    root_print(f"[INFO] Sim log      : {files.sim_log_txt}")
    root_print(f"[INFO] Box log      : {files.box_log_txt}")
    root_print(f"[INFO] Scalar GSD   : {files.scalar_gsd_log}")

    debug_kv(
        "Run summary",
        hoomd_version=hoomd.version.version,
        mpi_enabled=hoomd.version.mpi_enabled,
        stage_id=params.stage_id_current,
        tag=params.tag,
        input_gsd=files.input_gsd,
        traj_gsd=files.traj_gsd,
        restart_gsd=files.restart_gsd,
        final_gsd=files.final_gsd,
        sim_log=files.sim_log_txt,
        box_log=files.box_log_txt,
        scalar_gsd=files.scalar_gsd_log,
        total_steps=params.total_num_timesteps,
        equil_steps=params.equil_steps,
        pressure=params.pressure,
        log_frequency=params.log_frequency,
        traj_gsd_frequency=params.traj_gsd_frequency,
        restart_gsd_frequency=params.restart_gsd_frequency,
        use_gpu=params.use_gpu,
        gpu_id=params.gpu_id,
    )

    # ------------------------------------------------------------------
    # Seed 
    # ------------------------------------------------------------------
    # ensure_seed_file() writes the seed file on the very first run (rank 0
    # only) and is a no-op on subsequent calls, so restarts and later stages
    # always use the same seed.
    ensure_seed_file(params.stage_id_current, _env_rank)
    seed_file = (_SEED_FILE_SINGLE if params.stage_id_current == -1
                 else _SEED_FILE_MULTI)
    _wait_start = time.perf_counter()
    while not Path(seed_file).exists():
        elapsed = time.perf_counter() - _wait_start
        if elapsed > 30:
            fail_with_context(
                "Timeout waiting for seed file.",
                seed_file=seed_file,
                waited_seconds=f"{elapsed:.2f}",
                stage_id=params.stage_id_current,
                mpi_rank=_env_rank,
            )
        time.sleep(0.1)
    seed = read_seed(params.stage_id_current)
    root_print(f"[INFO] Random seed: {seed}")

    # ------------------------------------------------------------------
    # Build simulation
    # ------------------------------------------------------------------
    sim = None
    sim_log_hdl = None
    box_log_hdl = None

    try:
        (sim, mc, boxmc, trans_move_tuner, rot_move_tuner, box_tuner, sdf_compute, sim_log_hdl, box_log_hdl) = build_simulation(params, files, seed, comm, _env_rank)

        N = sim.state.N_particles
        prod_steps = params.total_num_timesteps - params.equil_steps

        # ------------------------------------------------------------------
        # Equilibration loop  
        # Run in chunks of equil_steps_check_freq; exit early when box_tuner
        # declares tuned.  The move_tuner stays active throughout.
        # ------------------------------------------------------------------
        # The equilibration phase runs total equil_steps MC sweeps broken into
        # n_chunks sub-runs of equil_steps_check_freq steps each.  After every
        # sub-run we poll box_tuner.tuned.  Once True (all box-move deltas
        # produce the target acceptance rate within +-tol) the BoxMCMoveSize
        # tuner is removed so move sizes remain fixed during production,
        # giving a stable ensemble.  The early-exit avoids wasting equilibration
        # steps once the tuner has already converged.
        n_chunks = params.equil_steps // params.equil_steps_check_freq

        if n_chunks == 0:
            root_print(
                "[WARNING] equil_steps < equil_steps_check_freq: "
                "equilibration will run as a single chunk without early-exit polling."
            )
            n_chunks = 1  # run at least one equil chunk

        root_print(
            f"\n[INFO] Equilibration: {params.equil_steps} steps | "
            f"checking every {params.equil_steps_check_freq} steps | "
            f"max {n_chunks} chunks"
        )
        root_flush_stdout()

        # Track the best observed BoxMC move-size state during equilibration.
        # "Best" means the combined length+shear BoxMC acceptance rate in the
        # current equilibration chunk is closest to target_box_movement_acc_rate.
        # If BoxMCMoveSize never declares tuned, this state is restored before
        # production and the tuner is removed anyway.
        box_tuner_removed = False
        box_tuner_converged = False
        best_box_move_state = None
        selected_box_move_state = None

        for chunk_idx in range(n_chunks):
            # Run exactly equil_steps_check_freq HPMC sweeps.  Both tuners
            # (MoveSize and BoxMCMoveSize) fire on their own Periodic triggers
            # inside this sim.run() call — no manual tuner calls needed.
            sim.run(params.equil_steps_check_freq)

            # Recompute phi after this chunk; the box has likely changed due
            # to accepted length/shear moves at the target pressure.
            phi_now = N * params.particle_volume / sim.state.box.volume

            # -------------------------------------------------------------------------
            # Capture the current BoxMC move-size / acceptance state after this equilibration chunk.
            # -------------------------------------------------------------------------
            current_box_move_state = _capture_box_move_state(
                boxmc=boxmc,
                sim=sim,
                target_acceptance=params.target_box_movement_acc_rate,
            )

            # The "best" state is defined as the state whose combined box acceptance rate
            # is closest to the target box acceptance rate from the JSON file.
            #
            # Example:
            #
            #     target_box_movement_acc_rate = 0.30
            #
            #     chunk A: combined_acceptance_rate = 0.05
            #              score_abs_error = |0.05 - 0.30| = 0.25
            #
            #     chunk B: combined_acceptance_rate = 0.228
            #              score_abs_error = |0.228 - 0.30| = 0.072
            #
            #     chunk C: combined_acceptance_rate = 0.41
            #              score_abs_error = |0.41 - 0.30| = 0.11
            #
            # Then chunk B is the best state.
            #
            # best_box_move_state is initially None before the first chunk.
            # On the first chunk, we always store current_box_move_state.
            # On later chunks, we replace it only if the new score is smaller.
            if (
                best_box_move_state is None
                or current_box_move_state["score_abs_error"] < best_box_move_state["score_abs_error"]
            ):
                best_box_move_state = current_box_move_state

            root_print(
                f"[EQUIL chunk {chunk_idx + 1}/{n_chunks}] "
                f"step={sim.timestep} | phi={phi_now:.5f} | "
                f"box_tuner.tuned={box_tuner.tuned} | "

                # Combined acceptance from all tracked BoxMC moves.
                # This is the value used to score closeness to the target.
                f"box_acc={current_box_move_state['combined_acceptance_rate']:.5f} | "
                # Length-only acceptance rate.
                # In this code, HOOMD exposes length/volume-like counts through boxmc.volume_moves.
                f"len_acc={current_box_move_state['length_acceptance_rate']:.5f} | "
                # Shear-only acceptance rate.
                f"shr_acc={current_box_move_state['shear_acceptance_rate']:.5f} | "
                # Raw length/volume-like move counts: accepted,rejected,total
                f"len_moves={current_box_move_state['length_accepts']},"
                f"{current_box_move_state['length_rejects']},"
                f"{current_box_move_state['length_attempts']} | "
                # Raw shear move counts: accepted,rejected,total
                f"shr_moves={current_box_move_state['shear_accepts']},"
                f"{current_box_move_state['shear_rejects']},"
                f"{current_box_move_state['shear_attempts']} | "
                # Raw combined box move counts: accepted,rejected,total
                f"box_moves={current_box_move_state['combined_accepts']},"
                f"{current_box_move_state['combined_rejects']},"
                f"{current_box_move_state['combined_attempts']} | "
                # How far the current state is from target acceptance. Smaller is better.
                f"target_gap={current_box_move_state['score_abs_error']:.5f} | "
                # How far the best observed state so far is from target acceptance.
                # This should stay the same or decrease as equilibration progresses.
                f"best_gap={best_box_move_state['score_abs_error']:.5f} | "
                # Current BoxMC length move amplitude tuple: (delta_x, delta_y, delta_z)
                f"length_delta={current_box_move_state['length_delta']} | "
                # Current BoxMC shear move amplitude tuple: (delta_xy, delta_xz, delta_yz)
                f"shear_delta={current_box_move_state['shear_delta']} | "
                # Current particle translational and rotational HPMC move sizes.
                f"d={mc.d['A']:.5f} | a={mc.a['A']:.5f}"
            )
            root_flush_stdout()

            # -------------------------------------------------------------------------
            # Converged-tuner branch.
            # -------------------------------------------------------------------------
            if box_tuner.tuned:
                # Converged: keep the current BoxMC move sizes, but still
                # remove BoxMCMoveSize before production.  HOOMD states that
                # this tuner continues tuning even after tuned=True, so leaving
                # it attached during production is not acceptable.
                selected_box_move_state = current_box_move_state
                # Remove BoxMCMoveSize from sim.operations.tuners.
                #
                # This freezes the current box move sizes:
                #
                #     boxmc.length["delta"]
                #     boxmc.shear["delta"]
                #
                # From this point onward, production will run with fixed BoxMC move sizes.
                _remove_tuner_if_attached(sim, box_tuner)
                # Record bookkeeping flags for later safety checks and summary JSON.
                box_tuner_removed = True
                box_tuner_converged = True
                root_print(
                    f"[INFO] BoxMCMoveSize tuner converged at step "
                    f"{sim.timestep}. Current box move sizes locked and tuner removed."
                )
                # Exit the equilibration loop early because the tuner has converged.
                # No need to spend the remaining equilibration chunks tuning.
                break

        # -------------------------------------------------------------------------
        # Non-converged-tuner branch.
        # -------------------------------------------------------------------------
        #
        # This block runs after the equilibration loop finishes.
        #
        # If the code reaches here with:
        #
        #     box_tuner_converged == False
        if not box_tuner_converged:
            if best_box_move_state is not None:
                # Restore the best observed length_delta and shear_delta.
                #
                # This changes:
                #     boxmc.length = {...}
                #     boxmc.shear  = {...}
                #
                # to the move sizes stored in best_box_move_state.
                #
                # After this call, the BoxMC object is no longer necessarily using the
                # final tuner values from the last equilibration chunk. It is using the
                # best observed values according to score_abs_error.
                _restore_box_move_state(boxmc, best_box_move_state)
                # This is the state production will use.
                selected_box_move_state = best_box_move_state
                # Print a clear warning because this is not ideal convergence.
                # But it is still safe because the tuner will be removed below.
                root_print(
                    f"[WARNING] BoxMCMoveSize tuner did NOT converge during "
                    f"equilibration ({params.equil_steps} steps).\n"
                    f"  => Restored best observed BoxMC move sizes from step "
                    f"{best_box_move_state['timestep']}.\n"
                    f"  => best_box_acc={best_box_move_state['combined_acceptance_rate']:.6f}, "
                    f"target={params.target_box_movement_acc_rate:.6f}, "
                    f"abs_error={best_box_move_state['score_abs_error']:.6f}.\n"
                    f"  => length_delta={best_box_move_state['length_delta']}, "
                    f"shear_delta={best_box_move_state['shear_delta']}."
                )
            else:
                selected_box_move_state = _capture_box_move_state(
                    boxmc=boxmc,
                    sim=sim,
                    target_acceptance=params.target_box_movement_acc_rate,
                )
                root_print(
                    f"[WARNING] BoxMCMoveSize tuner did NOT converge and no "
                    f"best-state record was available. Keeping current BoxMC "
                    f"move sizes."
                )

            # -------------------------------------------------------------
            # Remove the BoxMCMoveSize tuner even though it did not converge.
            # -------------------------------------------------------------
            _remove_tuner_if_attached(sim, box_tuner)
            box_tuner_removed = True
            root_print(
                "[INFO] BoxMCMoveSize tuner removed before production despite non-convergence."
            )

        # Hard safety guard: production is forbidden if the BoxMCMoveSize tuner
        # is still attached.  This encodes the user's priority directly in code.
        if _tuner_is_attached(sim, box_tuner):
            fail_with_context(
                "Refusing to start production while BoxMCMoveSize tuner is still attached.",
                timestep=sim.timestep,
                box_tuner_converged=box_tuner_converged,
                box_tuner_removed=box_tuner_removed,
            )

        # -------------------------------------------------------------------------
        # Store box-tuning provenance for the final summary JSON.
        # -------------------------------------------------------------------------
        box_tuning_result = {
            "mode": "converged_current_move_sizes" if box_tuner_converged else "nonconverged_best_observed_move_sizes",
            "box_tuner_converged": bool(box_tuner_converged),
            "box_tuner_removed_before_production": True,
            "box_tuner_attached_during_production": False,
            "selected_state": selected_box_move_state,
            "best_state": best_box_move_state,
        }

        # Attach SDF after equilibration so that the box has converged 
        # The Scale Distribution Function method computes the instantaneous
        # pressure from the distribution of scale factors that would just
        # bring each particle into contact with its nearest neighbour.
        # It assumes a fixed box for each measurement: attaching it only
        # after the box has equilibrated avoids misleading pressure readings
        # during the large volume fluctuations of the early equilibration
        # phase.  Once attached, sdf_compute.betaP on rank 0 gives the
        # instantaneous NPT pressure, and its time-average should converge
        # to the target betaP as a cross-check of the EOS.
        if sdf_compute is not None:
            sim.operations.computes.append(sdf_compute)
            root_print(
                f"[INFO] SDF compute attached after equilibration "
                f"(step {sim.timestep})."
            )

        # ------------------------------------------------------------------
        # Production run 
        # **** Parallel Section ****
        # ------------------------------------------------------------------
        root_print(
            f"\n[INFO] Production: {prod_steps} steps starting at "
            f"step {sim.timestep}"
        )
        root_flush_stdout()

        sim.run(prod_steps)

        root_print(
            f"[INFO] Production complete at step {sim.timestep} | "
            f"overlaps={mc.overlaps}"
        )

        # ------------------------------------------------------------------
        # Final outputs 
        # ------------------------------------------------------------------
        write_final_outputs(
            sim, mc, boxmc, params, files, start_time, seed,
            args.simulparam_file,
            box_tuning_result=box_tuning_result,
        )

    except Exception as exc:
        # ------------------------------------------------------------------
        # Emergency snapshot on any unexpected exception
        # ------------------------------------------------------------------
        root_print(
            f"\n[ERROR] Unexpected exception at timestep "
            f"{sim.timestep if sim else '?'}:\n"
            f"  {type(exc).__name__}: {exc}"
        )
        root_print(f"[ERROR] {type(exc).__name__}: {exc}")

        try:
            debug_kv(
                "Failure context",
                json_file=args.simulparam_file,
                tag=params.tag if 'params' in locals() else "?",
                input_gsd=files.input_gsd if 'files' in locals() else "?",
                restart_gsd=files.restart_gsd if 'files' in locals() else "?",
                final_gsd=files.final_gsd if 'files' in locals() else "?",
                mpi_rank=_env_rank,
                seed=seed if 'seed' in locals() else "?",
                sim_exists=(sim is not None),
            )
            if sim is not None:
                # Dump the last known box state to correlate the crash with
                # density and box shape at the time of failure.
                box = sim.state.box
                debug_kv(
                    "Last known simulation state",
                    timestep=sim.timestep,
                    N=sim.state.N_particles,
                    Lx=box.Lx,
                    Ly=box.Ly,
                    Lz=box.Lz,
                    xy=box.xy,
                    xz=box.xz,
                    yz=box.yz,
                )
        except Exception:
            root_print("[WARNING] Could not print extended failure context.")

        if sim is not None:
            try:
                ef = f"emergency_restart_{params.tag}.gsd"
                _write_snapshot(sim, mc, ef)
                root_print(f"[ERROR] Emergency snapshot => {ef}")
            except Exception as snap_exc:
                root_print(
                    f"[ERROR] Could not write emergency snapshot: "
                    f"{type(snap_exc).__name__}: {snap_exc}"
                )

        raise

    finally:
        # ------------------------------------------------------------------
        # Always close log file handles
        # ------------------------------------------------------------------
        # The finally block executes whether the try block succeeded or raised
        # an exception.  This guarantees log files are always flushed and their
        # file descriptors released, even if sim.run() crashes mid-step.
        for hdl in [sim_log_hdl, box_log_hdl]:
            if hdl is not None:
                try:
                    hdl.flush()
                    hdl.close()
                except Exception:
                    pass

        # Total runtime
        if _is_root_rank():
            runtime = time.time() - start_time
            print(f"\nTotal runtime: {runtime:.2f} seconds")


# ===========================================================================
#  SECTION 12 — Script entry point
# ===========================================================================

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        root_print("\n" + "=" * 80)
        root_print("[FATAL] Unexpected exception (bug in code):")
        root_print("=" * 80)
        traceback.print_exc()
        root_print("=" * 80)
        sys.exit(1)
