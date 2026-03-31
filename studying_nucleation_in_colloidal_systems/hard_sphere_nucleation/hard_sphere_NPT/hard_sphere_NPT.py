#!/usr/bin/env python3
# =============================================================================
# HOOMD-blue v4.9  |  Hard-Sphere HPMC NPT Simulation  —  hs_npt_v4.py
# =============================================================================
#
# PURPOSE
# -------
# Run a constant-pressure (NPT) hard-sphere HPMC simulation using HOOMD-blue v4.
# Performs an equilibration phase with active tuners, then a production phase
# with tuners removed, matching the structure of the original script.
#
# ALGORITHM  (unchanged from hard_sphere_npt_v1p11.py)
# ---------------------------------------------------
# 1.  Load parameters from a JSON file (--simulparam_file).
# 2.  Read initial configuration from GSD, broadcast via MPI.
# 3.  Configure HPMC Sphere integrator with initial move size d.
# 4.  Configure BoxMC updater at constant pressure betaP = P/(kT) with:
#       - volume moves    (isotropic compression/expansion)
#       - length moves    (independent Lx, Ly, Lz changes)
#       - aspect moves    (anisotropic rescaling)
#       - shear moves     (tilt factor changes)
# 5.  Attach MoveSize tuner for particle translational moves.
# 6.  Attach BoxMCMoveSize tuner for all box move types.
# 7.  EQUILIBRATION loop:
#       Run in chunks of equil_steps_check_freq; stop early if box_tuner
#       declares itself tuned.  Remove box_tuner when done.
# 8.  PRODUCTION run: sim.run(prod_steps) at constant box-move sizes.
# 9.  Write final GSD, machine-readable summary JSON, and console banner.
#
# USAGE
# -----
#   python hs_npt_v2.py --simulparam_file simulparam_hs_npt.json
#   mpirun -n 8 python hs_npt_v2.py --simulparam_file simulparam_hs_npt.json
#
# DEPENDENCIES
# ------------
#   HOOMD-blue >= 4.0  (https://hoomd-blue.readthedocs.io/en/v4.9.0/)
#   GSD >= 3.0
#   NumPy
#   mpi4py  (optional; serial fallback provided)
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
# mpi4py (optional)
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False
    class _MPIStub:
        class COMM_WORLD:
            @staticmethod
            def bcast(obj, root=0):
                return obj
    MPI = _MPIStub()

# ---------------------------------------------------------------------------
_PI_OVER_6 = math.pi / 6.0   # used in packing fraction


# ===========================================================================
#  SECTION 1 — MPI-aware console helpers  
# ===========================================================================

def _mpi_rank_from_env() -> int:
    """Return MPI rank from common launcher env vars; falls back to 0."""
    return int(
        os.environ.get(
            "OMPI_COMM_WORLD_RANK",
            os.environ.get("PMI_RANK", os.environ.get("SLURM_PROCID", 0))
        )
    )

# True only on MPI rank 0 (or in serial runs where rank is always 0).
# Used as a gate so that console output, file writes, and other rank-0-only
# operations are not duplicated across all N MPI processes.
def _is_root_rank() -> bool:
    return _mpi_rank_from_env() == 0

def root_print(*args, **kwargs) -> None:
    """Print only from rank 0 — prevents N-fold duplicate console output."""
    if _is_root_rank():
        print(*args, **kwargs)

# Flush stdout on rank 0 only.  Called after important progress messages
# so they appear immediately when stdout is redirected to a file, which
# is the typical usage on HPC clusters (output buffering can otherwise
# delay messages by minutes).
def root_flush_stdout() -> None:
    if _is_root_rank():
        sys.stdout.flush()

# fail_with_context: structured alternative to sys.exit(message).
# Prepends "[FATAL]" then prints each keyword argument on its own
# indented line so the operator sees exactly which file, key, or value
# caused the problem — without needing to read a raw Python traceback.
def fail_with_context(message: str, **kwargs) -> None:
    """
    Abort with a more informative multi-line error message.
    Useful for parameter / file / shape validation failures.
    """
    lines = [f"[FATAL] {message}"]
    for k, v in kwargs.items():
        lines.append(f"  {k}: {v}")
    sys.exit("\n".join(lines))

# debug_kv: prints a titled block of key=value pairs to stdout from rank 0
# only.  Used at key decision points (state-init, snapshot loading) and in
# the exception handler to give maximum diagnostic context.
def debug_kv(title: str, **kwargs) -> None:
    """
    Pretty-print a compact key=value debug block from rank 0 only.
    """
    if not _is_root_rank():
        return
    print(f"[DEBUG] {title}")
    for k, v in kwargs.items():
        print(f"    {k} = {v}")
    sys.stdout.flush()


# ===========================================================================
#  SECTION 2 — Custom loggable classes  (preserved from original)
# ===========================================================================

class Status:
    """
    Tracks estimated time remaining (ETR) for the simulation.
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
        """Estimated seconds to completion based on current TPS."""
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
        """Human-readable estimated time remaining."""
        return str(datetime.timedelta(seconds=self.seconds_remaining))


class MCStatus:
    """
    Windowed translational acceptance rate between consecutive log events.
    """

    _export_dict = {"acceptance_rate": ("scalar", True)}

    def __init__(self, integrator: hoomd.hpmc.integrate.Sphere) -> None:
        self.integrator     = integrator
        self.prev_accepted  = None
        self.prev_total     = None

    @property
    def acceptance_rate(self) -> float:
        """Windowed translational acceptance ratio (0.0 before first run)."""
        try:
            # integrator.translate_moves is a tuple (accepted, rejected).
            # Index [0] is the cumulative accepted count; sum() gives the
            # cumulative total (accepted + rejected).
            current_accepted = self.integrator.translate_moves[0]
            current_total    = sum(self.integrator.translate_moves)
            if self.prev_accepted is None or self.prev_total is None:
                # First call: no delta available yet.  Seed the previous-state
                # variables and return the cumulative rate for this log entry.
                self.prev_accepted = current_accepted
                self.prev_total    = current_total
                return current_accepted / current_total if current_total else 0.0
            # Windowed rate = accepted-since-last-query / total-since-last-query.
            # This is more responsive than the cumulative rate and directly
            # shows whether acceptance has changed as the box fluctuates.
            delta_accepted = current_accepted - self.prev_accepted
            delta_total    = current_total    - self.prev_total
            self.prev_accepted = current_accepted
            self.prev_total    = current_total
            return delta_accepted / delta_total if delta_total else 0.0
        except (IndexError, ZeroDivisionError, DataAccessError):
            return 0.0


class Box_property:
    """
    Exposes simulation-box dimensions and derived quantities as loggable scalars.
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

    def __init__(
        self,
        simulation: hoomd.Simulation,
        N: int,
        diameter: float,
    ) -> None:
        self.simulation = simulation
        self._N        = N
        self._diameter = diameter

    @property
    def L_x(self) -> float:
        return float(self.simulation.state.box.Lx)

    @property
    def L_y(self) -> float:
        return float(self.simulation.state.box.Ly)

    @property
    def L_z(self) -> float:
        return float(self.simulation.state.box.Lz)

    @property
    def XY(self) -> float:
        return float(self.simulation.state.box.xy)

    @property
    def XZ(self) -> float:
        return float(self.simulation.state.box.xz)

    @property
    def YZ(self) -> float:
        return float(self.simulation.state.box.yz)

    @property
    def volume(self) -> float:
        return float(self.simulation.state.box.volume)

    @property
    def volume_str(self) -> str:
        return f"{self.simulation.state.box.volume:.2f}"

    @property
    def packing_fraction(self) -> float:
        """
        phi = N * (pi/6) * sigma^3 / V

        In NPT the box volume fluctuates; logging phi at every interval is the
        most informative scalar for monitoring convergence toward the target
        pressure.
        """
        V = self.simulation.state.box.volume
        return self._N * _PI_OVER_6 * self._diameter**3 / V


class MoveSizeProp:
    """
    Exposes the current translational move size d as a loggable scalar.
    Preserved exactly from the original.
    """
    _export_dict = {"d": ("scalar", True)}

    def __init__(self, mc: hoomd.hpmc.integrate.Sphere, ptype: str = "A") -> None:
        self.mc    = mc
        self.ptype = ptype

    @property
    def d(self) -> float:
        """Current translational move size for particle type ptype."""
        try:
            return float(self.mc.d[self.ptype])
        except DataAccessError:
            return 0.0


class BoxMCStatus:
    """
    Windowed overall acceptance rate of all BoxMC move types.
    """

    _export_dict = {
        "acceptance_rate": ("scalar", True),
        "volume_acc_rate": ("scalar", True),
        "aspect_acc_rate": ("scalar", True),
        "shear_acc_rate":  ("scalar", True),
    }

    def __init__(self, boxmc: hoomd.hpmc.update.BoxMC) -> None:
        self.boxmc        = boxmc
        self.prev_accepted: dict = {}
        self.prev_total:   dict = {}

    def _windowed_rate(self, move: str) -> float:
        """Return the windowed acceptance rate for a single BoxMC move type."""
        try:
            moves = getattr(self.boxmc, f"{move}_moves")
            current_accepted, current_total = moves
            prev_accepted = self.prev_accepted.get(move, 0)
            prev_total    = self.prev_total.get(move, 0)
            delta_accepted = current_accepted - prev_accepted
            delta_total    = current_total    - prev_total
            self.prev_accepted[move] = current_accepted
            self.prev_total[move]    = current_total
            return delta_accepted / delta_total if delta_total else 0.0
        except (AttributeError, ZeroDivisionError, DataAccessError):
            return 0.0

    @property
    def acceptance_rate(self) -> float:
        """
        Overall windowed acceptance rate aggregated across volume, aspect,
        and shear moves.  Preserved from original.
        """
        try:
            total_accepted = 0
            total_attempts = 0
            for move in ["volume", "aspect", "shear"]:
                moves = getattr(self.boxmc, f"{move}_moves")
                current_accepted, current_total = moves
                delta_accepted = current_accepted - self.prev_accepted.get(move, 0)
                delta_total    = current_total    - self.prev_total.get(move, 0)
                self.prev_accepted[move] = current_accepted
                self.prev_total[move]    = current_total
                total_accepted += delta_accepted
                total_attempts += delta_total
            return total_accepted / total_attempts if total_attempts else 0.0
        except (AttributeError, ZeroDivisionError, DataAccessError):
            return 0.0

    @property
    def volume_acc_rate(self) -> float:
        """Windowed acceptance rate for volume moves only."""
        return self._windowed_rate("volume")

    @property
    def aspect_acc_rate(self) -> float:
        """Windowed acceptance rate for aspect moves only."""
        return self._windowed_rate("aspect")

    @property
    def shear_acc_rate(self) -> float:
        """Windowed acceptance rate for shear moves only."""
        return self._windowed_rate("shear")


class BoxSeqProp:
    """
    Exposes BoxMC move counters as formatted strings for the box log.
    """

    _export_dict = {
        "volume_moves_str": ("string", True),
        "aspect_moves_str": ("string", True),
        "shear_moves_str":  ("string", True),
    }

    def __init__(self, boxmc: hoomd.hpmc.update.BoxMC) -> None:
        self.boxmc = boxmc

    @property
    def volume_moves_str(self) -> str:
        """'accepted,total' string for volume moves."""
        try:
            a, r = self.boxmc.volume_moves
            return f"{a},{r}"
        except DataAccessError:
            return "0,0"

    @property
    def aspect_moves_str(self) -> str:
        """'accepted,total' string for aspect moves."""
        try:
            a, r = self.boxmc.aspect_moves
            return f"{a},{r}"
        except DataAccessError:
            return "0,0"

    @property
    def shear_moves_str(self) -> str:
        """'accepted,total' string for shear moves."""
        try:
            a, r = self.boxmc.shear_moves
            return f"{a},{r}"
        except DataAccessError:
            return "0,0"


class OverlapCount:
    """
    Exposes the current HPMC overlap count as a loggable scalar.

    In a valid hard-sphere simulation this should always be 0.  Logging it
    acts as a continuous sanity check — any non-zero value after the first
    few steps indicates a serious problem.
    """
    _export_dict = {"overlap_count": ("scalar", True)}

    def __init__(self, mc: hoomd.hpmc.integrate.Sphere) -> None:
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
    Exposes the SDF-derived compressibility factor Z = \beta P/\rho as a loggable scalar.
    [N-11]

    hoomd.hpmc.compute.SDF computes the pressure via the scale-distribution-
    function method (Anderson 2016, Eppenga & Frenkel 1984).  For hard spheres
    the pressure is related to the SDF extrapolated to x=0:

       \beta P = \rho * (1 + s(0+) / (2d))     where d = 3 (dimensionality)

    The loggable property `betaP` on SDF is the fully computed βP, and here
    we expose Z = \beta P /\rho  (the compressibility factor) which is what you
    compare against Carnahan-Starling (fluid) or Hall (crystal).

    NOTE: SDF is ONLY meaningful in NVT (fixed box).  In NPT the box
    fluctuates and the SDF pressure is not well-defined.  We therefore
    attach SDF during the equilibration phase only — after box_tuner is
    removed and the box has converged — and only if the JSON parameter
    `enable_sdf` is true.  The SDF result during production is a cross-
    check that the average pressure matches the target betaP.

    References
    ----------
    Anderson et al., J. Comput. Phys. 2016, 325, 74-97.
    Eppenga & Frenkel, Mol. Phys. 1984, 52, 1303-1334.
    """
    _export_dict = {
        "betaP":              ("scalar", True),
        "compressibility_Z":  ("scalar", True),
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
        Compare with Carnahan-Starling (Z_CS) to verify the EOS.
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
    """

    # --- I/O ---
    tag:                    str
    input_gsd_filename:     str
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
    trans_move_size_tuner_freq:     int    # MoveSize tuner trigger period
    target_particle_trans_move_acc_rate: float  # target acceptance for particle moves

    # --- BoxMC ---
    npt_freq:                       int    # BoxMC trigger period
    pressure:                       float  # target P/(kBT) = betaP
    box_tuner_freq:                 int    # BoxMCMoveSize tuner trigger period
    target_box_movement_acc_rate:   float  # target acceptance for box moves

    # BoxMC move deltas
    boxmc_volume_delta:     float  = 0.1
    boxmc_volume_mode:      str    = "standard"
    boxmc_length_delta:     float  = 0.01
    boxmc_aspect_delta:     float  = 0.02
    boxmc_shear_delta:      float  = 0.01

    # BoxMC tuner max_move_size
    max_move_volume:        float  = 0.1
    max_move_length:        float  = 0.05
    max_move_aspect:        float  = 0.02
    max_move_shear:         float  = 0.02

    # --- SDF pressure compute  [N-11] ---
    enable_sdf:             bool   = True  # attach hoomd.hpmc.compute.SDF
    sdf_xmax:               float  = 0.02  # max scale factor for SDF histogram
    sdf_dx:                 float  = 1e-4  # histogram bin width

    # --- Hardware  [N-16] ---
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

    # Resolved at runtime from the GSD file (not a JSON key)
    diameter: Optional[float] = None

    def validate(self) -> None:
        """Physics and logic sanity checks. Raises ValueError on failure."""
        # All errors are collected before raising so the user sees every
        # problem at once rather than fixing them one at a time.
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
        if not (0.0 < self.target_particle_trans_move_acc_rate < 1.0):
            errors.append(
                f"target_particle_trans_move_acc_rate must be in (0,1), "
                f"got {self.target_particle_trans_move_acc_rate}"
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
    "stage_id_current":                     int,
    "total_num_timesteps":                  int,
    "equil_steps":                          int,
    "equil_steps_check_freq":               int,
    "log_frequency":                        int,
    "traj_gsd_frequency":                   int,
    "restart_gsd_frequency":                int,
    "move_size_translation":                (int, float),
    "trans_move_size_tuner_freq":           int,
    "target_particle_trans_move_acc_rate":  (int, float),
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
    "boxmc_volume_delta":       (int, float),
    "boxmc_volume_mode":        str,
    "boxmc_length_delta":       (int, float),
    "boxmc_aspect_delta":       (int, float),
    "boxmc_shear_delta":        (int, float),
    "max_move_volume":          (int, float),
    "max_move_length":          (int, float),
    "max_move_aspect":          (int, float),
    "max_move_shear":           (int, float),
}


def load_simulparams(json_path: str) -> SimulationParams:
    """
    Read, type-check, and return a SimulationParams from a JSON file.
    Keys starting with '_' are stripped (comment-keys).
    """

    path = Path(json_path)
    if not path.exists():
        sys.exit(f"[FATAL] Parameter file not found: '{json_path}'")
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
    for fk in ["move_size_translation", "target_particle_trans_move_acc_rate",
               "pressure", "target_box_movement_acc_rate",
               "boxmc_volume_delta", "boxmc_length_delta",
               "boxmc_aspect_delta", "boxmc_shear_delta",
               "max_move_volume", "max_move_length",
               "max_move_aspect", "max_move_shear",
               "sdf_xmax", "sdf_dx"]:
        if fk in kw:
            kw[fk] = float(kw[fk])

    params = SimulationParams(**kw)
    try:
        params.validate()
    except ValueError as exc:
        sys.exit(f"[FATAL] {exc}")
    return params


# ===========================================================================
#  SECTION 5 — Seed management 
# ===========================================================================

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
        with open(seed_file, "w") as fh:
            json.dump({"random_seed": seed,
                       "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                      fh, indent=4)
        print(f"[INFO] Seed file created: {seed_file}  (seed={seed})")
    else:
        # Subsequent calls (restarts, later stages): re-use the existing seed
        # so trajectories remain reproducible across job resubmissions.
        print(f"[INFO] Existing seed file found: {seed_file}")


def read_seed(stage_id: int) -> int:
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
#  SECTION 7 — GSD diameter reader
# ===========================================================================

def read_mono_diameter_from_gsd(gsd_filename: str) -> float:
    """Return the monodisperse particle diameter from the last GSD frame."""
    path = Path(gsd_filename)
    if not path.exists():
        sys.exit(f"[FATAL] Cannot read diameter; GSD not found: '{gsd_filename}'")
    try:
        with gsd.hoomd.open(name=str(path), mode="r") as traj:
            if len(traj) == 0:
                sys.exit(f"[FATAL] GSD file has no frames: '{gsd_filename}'")
            frame = traj[-1]
    except Exception as exc:
        sys.exit(f"[FATAL] Cannot open '{gsd_filename}': {exc}")

    # GSD stores diameters as float32; converting to a Python list gives
    # native floats for comparison.
    diameters = list(frame.particles.diameter)
    if not diameters:
        sys.exit(f"[FATAL] No particle diameters in '{gsd_filename}'.")
    if any(d <= 0.0 for d in diameters):
        sys.exit(f"[FATAL] Non-positive diameter in '{gsd_filename}'.")
    d0 = diameters[0]
    if any(abs(d - d0) > 1e-12 for d in diameters[1:]):
        sys.exit(f"[FATAL] Polydisperse system in '{gsd_filename}' — "
                 "this script assumes a monodisperse hard-sphere system.")
    return float(d0)


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
                diameters = np.asarray(frame.particles.diameter, dtype=np.float64)
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

        if positions.ndim != 2 or positions.shape[1] != 3:
            fail_with_context(
                "Particle position array has invalid shape.",
                input_gsd=str(path.resolve()),
                positions_shape=positions.shape,
            )

        if len(diameters) != N:
            fail_with_context(
                "Diameter array length does not match number of particles.",
                input_gsd=str(path.resolve()),
                N=N,
                n_diameters=len(diameters),
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
            diameters_shape=diameters.shape,
            typeid_shape=typeid.shape,
            box=list(box_data),
            particle_types=types,
        )

        snap_data = {
            "box": box_data,
            "positions": positions,
            "diameters": diameters,
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
    required = ["box", "positions", "diameters", "typeid", "types", "N"]
    missing = [k for k in required if k not in snap_data]
    if missing:
        fail_with_context(
            "Broadcast snapshot data is incomplete.",
            missing_keys=missing,
            available_keys=sorted(snap_data.keys()),
        )

    positions = np.asarray(snap_data["positions"], dtype=np.float64)
    diameters = np.asarray(snap_data["diameters"], dtype=np.float64)
    typeid    = np.asarray(snap_data["typeid"], dtype=np.uint32)
    types     = list(snap_data["types"])
    N         = int(snap_data["N"])
    box       = list(snap_data["box"])

    if positions.shape != (N, 3):
        fail_with_context(
            "Snapshot reconstruction failed: positions shape mismatch.",
            expected_shape=(N, 3),
            actual_shape=positions.shape,
        )
    if diameters.shape != (N,):
        fail_with_context(
            "Snapshot reconstruction failed: diameters shape mismatch.",
            expected_shape=(N,),
            actual_shape=diameters.shape,
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
        snapshot.particles.diameter[:] = diameters
        snapshot.particles.typeid[:] = typeid
    except Exception as exc:
        fail_with_context(
            "HOOMD Snapshot reconstruction failed.",
            error_type=type(exc).__name__,
            error=str(exc),
            N=N,
            n_types=len(types),
            positions_shape=positions.shape,
            diameters_shape=diameters.shape,
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
    mc           : HPMC Sphere integrator
    boxmc        : BoxMC updater
    move_tuner   : MoveSize tuner
    box_tuner    : BoxMCMoveSize tuner
    sdf_compute  : hpmc.compute.SDF (or None if enable_sdf=False)
    sim_log_hdl  : open file handle for simulation Table log
    box_log_hdl  : open file handle for box Table log
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
    # This guarantees params.diameter is always consistent with the particles HOOMD is simulating
    params.diameter = read_mono_diameter_from_gsd(state_source)
    root_print(f"[INFO] Diameter (σ) = {params.diameter}")

    # Compute initial packing fraction for the startup log message.
    N   = sim.state.N_particles
    phi = N * _PI_OVER_6 * params.diameter**3 / sim.state.box.volume
    root_print(f"[INFO] N={N} | sigma = {params.diameter} | phi ={phi:.6f}")

    box = sim.state.box
    root_print(f"[INFO] Box: Lx={box.Lx:.4f} Ly={box.Ly:.4f} Lz={box.Lz:.4f} "
               f"xy={box.xy} xz={box.xz} yz={box.yz}")
    root_print(f"[INFO] Target betaP = {params.pressure}")

    # ------------------------------------------------------------------
    # 9.5  HPMC Sphere integrator  (preserved)
    # ------------------------------------------------------------------
    mc = hoomd.hpmc.integrate.Sphere(
        default_d=params.move_size_translation,
        nselect=1,    # one trial move per particle per sweep
    )
    mc.shape["A"] = {"diameter": params.diameter}
    sim.operations.integrator = mc
    # **** Parallel Section ****
    # MC moves are distributed across MPI domains; each rank handles its own
    # sub-domain and exchanges ghost-particle data with neighbours.
    root_print(f"[INFO] HPMC Sphere: sigma ={params.diameter} | d_init={params.move_size_translation}")

    # ------------------------------------------------------------------
    # 9.6  Custom loggable instances
    # ------------------------------------------------------------------
    # Each class exposes one or more properties via _export_dict.  HOOMD's
    # Logger calls each property getter once at registration time to
    # validate the loggable — this is why every getter guards against
    status        = Status(sim)
    mc_status     = MCStatus(mc)
    box_property  = Box_property(sim, N, params.diameter)
    move_prop     = MoveSizeProp(mc, ptype="A")
    overlap_count = OverlapCount(mc)

    # ------------------------------------------------------------------
    # 9.7  Logger: simulation log (Table)
    # ------------------------------------------------------------------
    # This logger collects per-step scalars and strings for the main
    # human-readable Table log and for embedding in the trajectory GSD.
    # categories=["scalar","string"]: only these two categories are needed;
    logger_sim = hoomd.logging.Logger(
        categories=["scalar", "string"], only_default=False)
    # Built-in HOOMD quantities: tps (steps/s), walltime (elapsed wall s),
    # timestep (current step counter as integer scalar)
    logger_sim.add(sim, quantities=["tps", "walltime", "timestep"])
    logger_sim[("Status",    "etr")]          = (status,        "etr",              "string")
    logger_sim[("Status",    "timestep")]     = (status,        "timestep_fraction","string")
    logger_sim[("MCStatus",  "acc_rate")]     = (mc_status,     "acceptance_rate",  "scalar")
    logger_sim[("MoveSize",  "d")]            = (move_prop,     "d",                "scalar")
    logger_sim[("Box",       "volume")]       = (box_property,  "volume",           "scalar")
    logger_sim[("Box",       "phi")]          = (box_property,  "packing_fraction", "scalar") 
    logger_sim[("HPMC",      "overlaps")]     = (overlap_count, "overlap_count",    "scalar")  

    # ------------------------------------------------------------------
    # 9.8  Logger: box log (Table)
    # ------------------------------------------------------------------
    # A separate box logger and Table writer records box geometry at every
    # log_frequency steps.
    logger_box = hoomd.logging.Logger(
        categories=["scalar", "string"], only_default=False)
    logger_box.add(sim, quantities=["timestep"])
    # Register the six independent box parameters (Lx, Ly, Lz, xy, xz, yz)
    for k, v in {"l_x":"L_x","l_y":"L_y","l_z":"L_z",
                 "XY":"XY","XZ":"XZ","YZ":"YZ"}.items():
        logger_box[("Box_property", k)] = (box_property, v, "scalar")
    logger_box[("Box", "volume_str")] = (box_property, "volume_str", "string")
    logger_box[("Box", "phi")]        = (box_property, "packing_fraction", "scalar") 


    # ------------------------------------------------------------------
    # 9.9  Open Table log file handles  
    #      + explicit writer-specific exception handling for easier debugging
    # ------------------------------------------------------------------
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
            logger=logger_sim,                   # store scalars in GSD too
        )
        traj_gsd_writer.write_diameter = True
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
        )
        restart_gsd_writer.write_diameter = True
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
        scalar_gsd_logger = hoomd.logging.Logger(
            categories=["scalar", "string"], only_default=False
        )
        scalar_gsd_logger.add(sim, quantities=["timestep", "tps", "walltime"])
        scalar_gsd_logger[("Box",      "volume")]    = (box_property,  "volume",           "scalar")
        scalar_gsd_logger[("Box",      "phi")]       = (box_property,  "packing_fraction", "scalar")
        scalar_gsd_logger[("MCStatus", "acc_rate")]  = (mc_status,     "acceptance_rate",  "scalar")
        scalar_gsd_logger[("MoveSize", "d")]         = (move_prop,     "d",                "scalar")
        scalar_gsd_logger[("HPMC",     "overlaps")]  = (overlap_count, "overlap_count",    "scalar")

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
    # Metropolis criterion, which includes the P·ΔV work term and the
    # N·ln(V_new/V_old) Jacobian.
    # betaP = P/(kBT) is the dimensionless reduced pressure.
    try:
        betaP_variant = hoomd.variant.Constant(params.pressure)
        boxmc = hoomd.hpmc.update.BoxMC(
            trigger=hoomd.trigger.Periodic(params.npt_freq),
            betaP=betaP_variant,
        )

        # Volume moves: propose isotropic scaling V => V*(1 +- delta_V).
        # mode="standard": linear volume change. 
        boxmc.volume = dict(
            weight=1.0,
            mode=params.boxmc_volume_mode,
            delta=params.boxmc_volume_delta,
        )

        # Length moves: propose independent changes to Lx, Ly, Lz at
        # constant shape (tilt factors unchanged).
        boxmc.length = dict(
            delta=(params.boxmc_length_delta,
                params.boxmc_length_delta,
                params.boxmc_length_delta),
            weight=1.0,
        )

        # Aspect moves: rescale one axis relative to the others at constant V.
        # Useful for relaxing aspect ratio in cases where the equilibrium
        # box is not cubic.
        boxmc.aspect = dict(
            delta=params.boxmc_aspect_delta,
            weight=1.0,
        )

        # Shear moves: change the tilt factors xy, xz, yz.
        # reduce=0.0 disables Lees-Edwards-style lattice-vector reduction;
        # standard for fluid and solid systems without shear flow.
        boxmc.shear = dict(
            delta=(params.boxmc_shear_delta,
                params.boxmc_shear_delta,
                params.boxmc_shear_delta),
            weight=1.0,
            reduce=0.0,
        )

        sim.operations.updaters.append(boxmc)

    except Exception as exc:
        fail_with_context(
            "Failed while configuring BoxMC updater.",
            pressure=params.pressure,
            npt_freq=params.npt_freq,
            volume_mode=params.boxmc_volume_mode,
            volume_delta=params.boxmc_volume_delta,
            length_delta=params.boxmc_length_delta,
            aspect_delta=params.boxmc_aspect_delta,
            shear_delta=params.boxmc_shear_delta,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    root_print(
        f"[INFO] BoxMC: betaP={params.pressure} | "
        f"volume_delta={params.boxmc_volume_delta} | "
        f"length_delta={params.boxmc_length_delta} | "
        f"aspect_delta={params.boxmc_aspect_delta} | "
        f"shear_delta={params.boxmc_shear_delta}"
    )

    # ------------------------------------------------------------------
    # 9.14  Register BoxMC loggers (after boxmc exists)
    # ------------------------------------------------------------------
    box_mc_status = BoxMCStatus(boxmc)
    logger_sim[("BoxMCStatus", "acc_rate")]        = (box_mc_status, "acceptance_rate", "scalar")
    logger_sim[("BoxMCStatus", "volume_acc_rate")] = (box_mc_status, "volume_acc_rate", "scalar")  
    logger_sim[("BoxMCStatus", "aspect_acc_rate")] = (box_mc_status, "aspect_acc_rate", "scalar")  
    logger_sim[("BoxMCStatus", "shear_acc_rate")]  = (box_mc_status, "shear_acc_rate",  "scalar")  
    scalar_gsd_logger[("BoxMCStatus", "acc_rate")] = (box_mc_status, "acceptance_rate", "scalar")

    seq_prop = BoxSeqProp(boxmc)
    logger_box[("BoxMC", "vol_moves")]    = (seq_prop, "volume_moves_str", "string")
    logger_box[("BoxMC", "aspect_moves")] = (seq_prop, "aspect_moves_str", "string")
    logger_box[("BoxMC", "shear_moves")]  = (seq_prop, "shear_moves_str",  "string")
    logger_box.add(boxmc, quantities=["volume_moves", "aspect_moves", "shear_moves"])

    # ------------------------------------------------------------------
    # 9.15  MoveSize tuner for particle moves  
    # ------------------------------------------------------------------
    # MoveSize.scale_solver adjusts mc.d["A"] every trans_move_size_tuner_freq
    # steps by multiplying it by a scale factor so the translational acceptance
    # rate converges toward target_particle_trans_move_acc_rate.
    # This tuner is NOT removed after equilibration — it remains active during
    # production so d continues to track the slowly changing acceptance rate
    # as box volume fluctuates in the NPT ensemble.
    move_tuner = hoomd.hpmc.tune.MoveSize.scale_solver(
        moves=["d"],
        target=params.target_particle_trans_move_acc_rate,
        trigger=hoomd.trigger.Periodic(params.trans_move_size_tuner_freq),
        max_translation_move=0.2,
    )
    sim.operations.tuners.append(move_tuner)

    # ------------------------------------------------------------------
    # 9.16  BoxMCMoveSize tuner  
    # ------------------------------------------------------------------
    # BoxMCMoveSize.scale_solver adjusts the delta parameters for every
    # listed box-move type every box_tuner_freq steps so each move's
    # acceptance rate converges toward target_box_movement_acc_rate.
    # gamma=0.8: how aggressively the tuner scales delta.  Values closer
    #   to 1.0 are more aggressive; HOOMD documentation recommends 0.8.
    # tol=0.01: the tuner declares itself converged (box_tuner.tuned=True)
    #   when the acceptance ratio is within ±0.01 (1%) of the target.
    #   The equilibration loop polls box_tuner.tuned and removes this tuner
    #   once converged, locking in the optimal box-move sizes for production.
    # max_move_size: upper bounds prevent the tuner from setting excessively
    #   large deltas that would cause sudden large box changes and temporarily
    #   break MPI domain decomposition or produce many overlaps.
    box_tuner = hoomd.hpmc.tune.BoxMCMoveSize.scale_solver(
        trigger=hoomd.trigger.Periodic(params.box_tuner_freq),
        boxmc=boxmc,
        moves=["volume",
               "length_x", "length_y", "length_z",
               "aspect",
               "shear_x",  "shear_y",  "shear_z"],
        target=params.target_box_movement_acc_rate,
        max_move_size={
            "volume":   params.max_move_volume,
            "length_x": params.max_move_length,
            "length_y": params.max_move_length,
            "length_z": params.max_move_length,
            "aspect":   params.max_move_aspect,
            "shear_x":  params.max_move_shear,
            "shear_y":  params.max_move_shear,
            "shear_z":  params.max_move_shear,
        },
        gamma=0.8,   # tuner aggressiveness (HOOMD recommendation)
        tol=0.01,    # convergence tolerance
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
    #   \beta P = \rho · (1 + s(0+) / 6)     [hard spheres in 3D]
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

    return sim, mc, boxmc, move_tuner, box_tuner, sdf_compute, sim_log_hdl, box_log_hdl


# ===========================================================================
#  SECTION 10 — Output helpers
# ===========================================================================

def _write_snapshot(sim: hoomd.Simulation, filename: str) -> None:
    """Write current state to a single-frame GSD file with diameter"""
    # hoomd.write.GSD.write() is the one-shot static helper that writes the
    # full simulation state to a fresh GSD file in a single call.
    hoomd.write.GSD.write(
        state=sim.state,
        filename=filename,
        mode="wb",
        filter=hoomd.filter.All(),
    )


def write_final_outputs(
    sim:        hoomd.Simulation,
    mc:         hoomd.hpmc.integrate.Sphere,
    boxmc:      hoomd.hpmc.update.BoxMC,
    params:     SimulationParams,
    files:      RunFiles,
    start_time: float,
    seed:       int,
    json_path:  str,
) -> None:
    """
    Write final GSD, summary JSON, and console banner.
    """

    N   = sim.state.N_particles
    phi = N * _PI_OVER_6 * params.diameter**3 / sim.state.box.volume

    # Final GSD snapshot
    _write_snapshot(sim, files.final_gsd)
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
        "diameter":          params.diameter,
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
        "runtime_seconds":   round(runtime, 2),
        "created":           time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_file = f"{params.tag}_stage{params.stage_id_current}_npt_summary.json"
    if _is_root_rank():
        with open(summary_file, "w") as fh:
            json.dump(summary, fh, indent=2)
        root_print(f"[OUTPUT] Summary JSON    => {summary_file}")

    # Console banner
    _print_banner("HOOMD-blue v4.9 | Hard-Sphere NPT | Run Complete")
    root_print("\n".join([
        f"  Simulparam file       : {json_path}",
        f"  Tag                   : {params.tag}",
        f"  Stage id              : {params.stage_id_current}",
        f"  Input GSD             : {files.input_gsd}",
        f"  Final GSD             : {files.final_gsd}",
        f"  Particles (N)         : {N}",
        f"  Diameter (σ)          : {params.diameter}",
        f"  Final packing frac.   : {phi:.6f}",
        f"  Target betaP          : {params.pressure}",
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
            "HOOMD-blue v4.9 hard-sphere HPMC NPT simulation.\n"
            "Usage: python hs_npt_v2.py --simulparam_file simulparam.json"
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

    _print_banner("HOOMD-blue v4.9 | Hard-Sphere HPMC NPT (hs_npt_v2)")

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
        (sim, mc, boxmc, move_tuner, box_tuner,
         sdf_compute, sim_log_hdl, box_log_hdl) = build_simulation(
            params, files, seed, comm, _env_rank)

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
        # produce the target acceptance rate within ±tol) the BoxMCMoveSize
        # tuner is removed so move sizes remain fixed during production,
        # giving a stable ensemble.  The early-exit avoids wasting equilibration
        # steps once the tuner has already converged.
        n_chunks = params.equil_steps // params.equil_steps_check_freq
        root_print(
            f"\n[INFO] Equilibration: {params.equil_steps} steps | "
            f"checking every {params.equil_steps_check_freq} steps | "
            f"max {n_chunks} chunks"
        )
        root_flush_stdout()

        # box_tuner_removed tracks whether the BoxMCMoveSize tuner converged
        # and was removed early; used to issue a warning if it did not.
        box_tuner_removed = False
        for chunk_idx in range(n_chunks):
            # Run exactly equil_steps_check_freq HPMC sweeps.  Both tuners
            # (MoveSize and BoxMCMoveSize) fire on their own Periodic triggers
            # inside this sim.run() call — no manual tuner calls needed.
            sim.run(params.equil_steps_check_freq)

            # Recompute phi after this chunk; the box has likely changed due
            # to accepted volume/length/aspect moves at the target pressure.
            phi_now = (N * _PI_OVER_6 * params.diameter**3
                       / sim.state.box.volume)
            root_print(
                f"[EQUIL chunk {chunk_idx + 1}/{n_chunks}] "
                f"step={sim.timestep} | phi={phi_now:.5f} | "
                f"box_tuner.tuned={box_tuner.tuned} | "
                f"d={mc.d['A']:.5f}"
            )
            root_flush_stdout()

            if box_tuner.tuned:
                # All box-move deltas have converged to the target acceptance
                # rate.  Remove the tuner from the operations list so delta
                # values remain fixed during production — a requirement for a
                # statistically well-defined NPT ensemble measurement.
                sim.operations.tuners.remove(box_tuner)
                box_tuner_removed = True
                root_print(
                    f"[INFO] BoxMCMoveSize tuner has converged at step "
                    f"{sim.timestep}. Tuner removed."
                )
                break

        # If all n_chunks ran without the tuner converging, the box-move
        # deltas may not be fully optimal.  The simulation is still valid
        # but the user should consider increasing equil_steps or loosening
        # the BoxMCMoveSize tol parameter to allow earlier convergence.
        if not box_tuner_removed:
            root_print(
                "[WARNING] BoxMCMoveSize tuner did NOT converge during "
                f"equilibration ({params.equil_steps} steps). "
                "Running production with the tuner still active."
            )

        # Attach SDF after equilibration so that the box has converged  [N-11]
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
            args.simulparam_file
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
                _write_snapshot(sim, ef)
                root_print(f"[ERROR] Emergency snapshot → {ef}")
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
