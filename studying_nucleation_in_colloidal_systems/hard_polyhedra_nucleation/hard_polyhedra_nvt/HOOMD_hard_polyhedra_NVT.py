#!/usr/bin/env python3
# =============================================================================
# HOOMD-blue v4  |  Convex-Polyhedron HPMC NVT Equilibration
# =============================================================================
#
#
# ALGORITHM
# -------------------------------------------------------
# 1. Read parameters from a JSON file passed via --simulparam_file.
# 2. Load the simulation state from a GSD file (restart-aware: if a restart
#    GSD exists and no final GSD exists, resume from the checkpoint).
# 3. Configure the HPMC ConvexPolyhedron integrator with a fixed move size d.
#    (No adaptive tuner — consistent with the original script.)
# 4. Attach writers:
#      - Table writer  => human-readable .log file (tps, walltime, ETR,
#                        timestep fraction, acceptance rate, box volume,
#                        packing fraction, overlap count)
#      - GSD trajectory writer  => appending trajectory file
#      - GSD restart writer     => single-frame truncating checkpoint
# 5. Call sim.run(total_num_timesteps).
# 6. Write a final GSD snapshot and a machine-readable summary JSON.
#
# PRESERVED EXACTLY
# -----------------
# - Status class (timestep_fraction, seconds_remaining, etr)
# - MCStatus class (windowed acceptance_rate with prev tracking)
# - Box_property class (volume property)
# - MPI snapshot broadcast pattern (rank-0 reads, bcast, reconstruct)
# - HPMC ConvexPolyhedron integrator with fixed d (no MoveSize tuner)
# - nselect=1 (one trial move per particle per sweep)
# - All three writers: Table log, GSD trajectory, GSD restart
#
# USAGE
# -----
#   Single CPU / GPU:
#     python HOOMD_hard_polyhedra_NVT.py --simulparam_file simulparam_hs_nvt.json
#
#   MPI (e.g. via mpirun / srun):
#     mpirun -n 8 python HOOMD_hard_polyhedra_NVT.py --simulparam_file simulparam_hs_nvt.json
#
# JSON SCHEMA  (see simulparam_hs_nvt.json for annotated example)
# ---------------------------------------------------------------
# {
#   "tag"                    : "convex_polyhedron_4096_at_pf0p58_nvt",
#   "input_gsd_filename"     : "HOOMD_hard_cube_2197_compression_to_pf0p58_final.gsd",
#   "stage_id_current"       : -1,     // -1 = single-stage; 0,1,... = multi-stage
#   "initial_timestep"       : 0,
#   "total_num_timesteps"    : 1000000,
#   "shape_json_filename"    : "shape_023_Cube_unit_volume_principal_frame.json",
#   "shape_scale"            : 1.0,
#   "total_num_timesteps"    : 50000000,
#   "move_size_translation"  : 0.04,   // fixed d for type A
#   "move_size_rotation"     : 0.02,
#   "log_frequency"          : 10000,
#   "traj_gsd_frequency"     : 100000,
#   "restart_gsd_frequency"  : 10000,
#   "use_gpu"                : false,
#   "gpu_id"                 : 0,
#   // single-stage filenames (ignored when stage_id_current >= 0):
#   "output_trajectory"      : "nvt_hpmc_output_traj.gsd",
#   "log_filename"           : "nvt_hpmc_log.log",
#   "restart_file"           : "nvt_hpmc_output_restart.gsd",
#   "final_gsd_filename"     : "nvt_hpmc_final.gsd"
# }
#
# STAGE NAMING CONVENTION  (stage_id_current >= 0)
# -------------------------------------------------
# All output files are prefixed with  <tag>_<stage_id>_
#   traj      =>  <tag>_<stage_id>_traj.gsd
#   restart   =>  <tag>_<stage_id>_restart.gsd
#   final     =>  <tag>_<stage_id>_final.gsd
#   log       =>  <tag>_<stage_id>.log
# Input for stage N comes from  <tag>_{N-1}_final.gsd
# The run is blocked if  <tag>_<stage_id>_final.gsd  already exists.
#
# DEPENDENCIES
# ------------
#   HOOMD-blue >= 4.0   (https://hoomd-blue.readthedocs.io/en/v4.9.0/)
#   GSD >= 3.0
#   NumPy
#   mpi4py  (optional; falls back gracefully if MPI is not used)
#
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
# GSD import (required)
# ---------------------------------------------------------------------------
try:
    import gsd.hoomd
except ImportError as _e:
    sys.exit(f"[FATAL] gsd package not found: {_e}\n  pip install gsd")

# ---------------------------------------------------------------------------
# HOOMD import (required)
# ---------------------------------------------------------------------------
try:
    import hoomd
    import hoomd.hpmc
    import hoomd.logging
    import hoomd.write
    import hoomd.trigger
    import hoomd.filter
    from hoomd.error import DataAccessError
except ImportError as _e:
    sys.exit(
        f"[FATAL] HOOMD-blue not found: {_e}\n"
        "Install HOOMD-blue v4 before running this script."
    )

# ---------------------------------------------------------------------------
# mpi4py import (optional — falls back to serial if not available)
# ---------------------------------------------------------------------------
try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False
    # Provide a minimal stub so the rest of the code runs unchanged
    class _MPIStub:
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
    Return the MPI rank from common launcher environment variables.
    Works for OpenMPI, PMI-based, and SLURM launchers.
    Falls back to 0 in serial (non-MPI) runs.
    """
    # Priority order for rank detection (first match wins):
    #   1. OMPI_COMM_WORLD_RANK  — set by Open MPI / mpirun
    #   2. PMI_RANK              — set by MPICH / PMI launchers (Cray, Intel MPI)
    #   3. SLURM_PROCID          — set by SLURM srun even without mpirun
    #   4. default 0             — serial run (no MPI launcher present)
    # os.environ.get() returns None when the variable is absent, so the
    # nested calls cascade until one succeeds or the default 0 is used.
    return int(
        os.environ.get(
            "OMPI_COMM_WORLD_RANK",
            os.environ.get("PMI_RANK", os.environ.get("SLURM_PROCID", 0))
        )
    )


def _is_root_rank() -> bool:
    """True only on MPI rank 0 (or in serial execution)."""
    return _mpi_rank_from_env() == 0


def root_print(*args, **kwargs) -> None:
    """Print only from rank 0 to avoid duplicate console output in MPI runs."""
    if _is_root_rank():
        print(*args, **kwargs)


def root_flush_stdout() -> None:
    """Flush stdout only on rank 0."""
    if _is_root_rank():
        sys.stdout.flush()


# ===========================================================================
#  SECTION 2 — Custom logger classes  (preserved from original script)
# ===========================================================================

class Status:
    """
    Tracks and computes the estimated time remaining (ETR) for the simulation.

    Registered as a custom loggable in the Table logger.  Both string
    properties (timestep_fraction, etr) are guarded so they return safe
    sentinel values before the first sim.run() call.
    """

    # HOOMD logger introspection dict — maps property names to (category, flag)
    _export_dict = {
        "timestep_fraction": ("string", True),
        "etr":               ("string", True),
    }

    def __init__(self, simulation: hoomd.Simulation) -> None:
        self.simulation = simulation

    @property
    def timestep_fraction(self) -> str:
        """Return timestep progress as 'current/total' string."""
        try:
            return (
                f"{self.simulation.timestep}/"
                f"{self.simulation.final_timestep}"
            )
        except Exception:
            # final_timestep is only populated once sim.run() has been
            # called for the first time.  Before that, accessing it raises
            # AttributeError.  Return a harmless sentinel so that the logger
            # can register this property without crashing at startup.
            return "0/?"

    @property
    def seconds_remaining(self) -> float:
        """Estimated seconds remaining based on current TPS."""
        try:
            return (
                (self.simulation.final_timestep - self.simulation.timestep)
                / self.simulation.tps
            )
        except (ZeroDivisionError, DataAccessError, AttributeError):
            # ZeroDivisionError  : tps == 0 (simulation hasn't started yet)
            # DataAccessError    : sim counters not yet initialised
            # AttributeError     : final_timestep not yet set by sim.run()
            return 0.0

    @property
    def etr(self) -> str:
        """Estimated time remaining as a formatted timedelta string."""
        return str(datetime.timedelta(seconds=self.seconds_remaining))


class MCStatus:
    """
    Tracks the *windowed* translational acceptance rate between log events.

    On each query, it computes the fraction of accepted moves since the
    previous query (delta_accepted / delta_total), not the cumulative rate.
    This gives the instantaneous acceptance ratio at each log interval, which
    is more informative during a long run.
    """

    _export_dict = {
        "translate_acceptance_rate": ("scalar", True),
        "rotate_acceptance_rate": ("scalar", True),
    }

    def __init__(self, integrator: hoomd.hpmc.integrate.ConvexPolyhedron) -> None:
        self.integrator = integrator
        self.prev_translate_accepted: Optional[int] = None
        self.prev_translate_total: Optional[int] = None
        self.prev_rotate_accepted: Optional[int] = None
        self.prev_rotate_total: Optional[int] = None

    @staticmethod
    def _windowed_rate(current_accepted: int,
                       current_total: int,
                       prev_accepted: Optional[int],
                       prev_total: Optional[int]) -> tuple[float, int, int]:
        if prev_accepted is None or prev_total is None:
            rate = current_accepted / current_total if current_total != 0 else 0.0
            return rate, current_accepted, current_total
        delta_accepted = current_accepted - prev_accepted
        delta_total = current_total - prev_total
        rate = delta_accepted / delta_total if delta_total != 0 else 0.0
        return rate, current_accepted, current_total

    @property
    def translate_acceptance_rate(self) -> float:
        try:
            current_accepted = int(self.integrator.translate_moves[0])
            current_total = int(sum(self.integrator.translate_moves))
            rate, self.prev_translate_accepted, self.prev_translate_total = (
                self._windowed_rate(
                    current_accepted,
                    current_total,
                    self.prev_translate_accepted,
                    self.prev_translate_total,
                )
            )
            return rate
        except (IndexError, ZeroDivisionError, DataAccessError, TypeError):
            return 0.0

    @property
    def rotate_acceptance_rate(self) -> float:
        try:
            current_accepted = int(self.integrator.rotate_moves[0])
            current_total = int(sum(self.integrator.rotate_moves))
            rate, self.prev_rotate_accepted, self.prev_rotate_total = (
                self._windowed_rate(
                    current_accepted,
                    current_total,
                    self.prev_rotate_accepted,
                    self.prev_rotate_total,
                )
            )
            return rate
        except (IndexError, ZeroDivisionError, DataAccessError, TypeError):
            return 0.0


class Box_property:
    """
    Exposes simulation-box derived quantities as loggable scalar properties.

    Properties
    ----------
    volume           : float — box volume V
    packing_fraction : float — phi = N * (pi/6) * sigma^3 / V 
    """

    _export_dict = {
        "volume":           ("scalar", True),
        "packing_fraction": ("scalar", True),
    }

    def __init__(self,
                 simulation: hoomd.Simulation,
                 N: int,
                 particle_volume: float) -> None:
        self.simulation = simulation
        self._N = int(N)
        self._particle_volume = float(particle_volume)

    @property
    def volume(self) -> float:
        return float(self.simulation.state.box.volume)

    @property
    def packing_fraction(self) -> float:
        return packing_fraction(self._N, self._particle_volume, self.simulation.state.box)


class OverlapCount:
    """
    Exposes the current HPMC overlap count as a loggable scalar.
    """

    _export_dict = {"overlap_count": ("scalar", True)}

    def __init__(self, mc: hoomd.hpmc.integrate.ConvexPolyhedron) -> None:
        self._mc = mc

    @property
    def overlap_count(self) -> float:
        """Number of overlapping particle pairs (should be 0 in NVT)."""
        try:
            return float(self._mc.overlaps)
        except DataAccessError:
            # mc.overlaps is computed lazily; it raises DataAccessError
            # before the integrator has been attached and sim.run() called
            # at least once.  Return 0.0 as a pre-run sentinel so that the
            # logger can be constructed without crashing.
            return 0.0


class CurrentTimestep:
    """Expose the current simulation timestep as a scalar loggable quantity."""

    _export_dict = {"timestep": ("scalar", True)}

    def __init__(self, simulation: hoomd.Simulation) -> None:
        self._simulation = simulation

    @property
    def timestep(self) -> float:
        try:
            return float(self._simulation.timestep)
        except (DataAccessError, AttributeError):
            # DataAccessError  : simulation state not yet initialised
            # AttributeError   : timestep attribute absent before first run
            return 0.0


# ===========================================================================
#  SECTION 3 — Parameter dataclass
# ===========================================================================

@dataclass
class SimulationParams:
    """
    Typed, validated container for all runtime parameters.

    Loaded from JSON via load_simulparams().  All fields have Python-type
    annotations so that JSON values are checked at load time, preventing
    silent mis-typed parameter bugs.
    """

    # --- I/O ---
    tag:                  str    # unique run identifier, used in filenames
    input_gsd_filename:   str    # initial configuration GSD file
    stage_id_current:     int    # -1 = single-stage; >= 0 = multi-stage

    # --- shape / physics ---
    shape_json_filename: str
    shape_scale: float

    # --- Run length ---
    total_num_timesteps:  int    # total HPMC sweeps for this stage

    # --- HPMC move ---
    move_size_translation: float # fixed translational move size d for type A
    move_size_rotation: float

    # --- Output frequencies ---
    log_frequency:         int   # Table log trigger period
    traj_gsd_frequency:    int   # GSD trajectory trigger period
    restart_gsd_frequency: int   # GSD restart (checkpoint) trigger period

    # --- Hardware ---
    use_gpu:               bool
    gpu_id:                int

    # --- Optional diagnostics / filenames ---
    diagnostics_frequency: int   = 0
    initial_timestep:      int   = 0
    output_trajectory:     str   = "nvt_hpmc_output_traj.gsd"
    log_filename:          str   = "nvt_hpmc_log.log"
    restart_file:          str   = "nvt_hpmc_output_restart.gsd"
    final_gsd_filename:    str   = "nvt_hpmc_final.gsd"
    hdf5_log_filename:     str   = "nvt_hpmc_diagnostics.h5"

    # --- resolved at runtime ---
    shape_name: Optional[str] = None
    shape_short_name: Optional[str] = None
    particle_volume: Optional[float] = None
    shape_vertices: Optional[list] = None

    def validate(self) -> None:
        """
        Run physics and logic sanity checks.
        Raises ValueError with a clear message on the first failed check.
        All errors are collected into a list so the user sees every problem
        in a single error message rather than having to fix one at a time.
        Called by load_simulparams() immediately after dataclass construction.
        """
        errors: list[str] = []

        if not isinstance(self.shape_json_filename, str) or not self.shape_json_filename.strip():
            errors.append("shape_json_filename must be a non-empty string")
        if self.shape_scale <= 0.0:
            errors.append(f"shape_scale must be > 0, got {self.shape_scale}")
        if self.total_num_timesteps <= 0:
            errors.append(
                f"total_num_timesteps must be > 0, got {self.total_num_timesteps}"
            )
        if self.move_size_translation <= 0.0:
            errors.append(
                f"move_size_translation must be > 0, got {self.move_size_translation}"
            )
        if self.move_size_rotation < 0.0:
            errors.append(
                f"move_size_rotation must be >= 0, got {self.move_size_rotation}"
            )
        for name, val in [
            ("log_frequency",          self.log_frequency),
            ("traj_gsd_frequency",     self.traj_gsd_frequency),
            ("restart_gsd_frequency",  self.restart_gsd_frequency),
        ]:
            if val <= 0:
                errors.append(f"{name} must be > 0, got {val}")
        if self.stage_id_current < -1:
            errors.append(
                f"stage_id_current must be >= -1, got {self.stage_id_current}"
            )
        if self.diagnostics_frequency < 0:
            errors.append(
                f"diagnostics_frequency must be >= 0, got {self.diagnostics_frequency}"
            )

        if errors:
            raise ValueError(
                "Parameter validation failed:\n"
                + "\n".join(f"   {e}" for e in errors)
            )


# ===========================================================================
#  SECTION 4 — JSON loading 
# ===========================================================================

# Required JSON keys with expected Python types
_REQUIRED_KEYS: dict[str, type] = {
    "tag":                    str,
    "input_gsd_filename":     str,
    "stage_id_current":       int,
    "shape_json_filename": str,
    "shape_scale": (int, float),
    "total_num_timesteps":    int,
    "move_size_translation":  (int, float),
    "move_size_rotation": (int, float),
    "log_frequency":          int,
    "traj_gsd_frequency":     int,
    "restart_gsd_frequency":  int,
    "use_gpu":                bool,
    "gpu_id":                 int,
}

_OPTIONAL_KEYS: dict[str, type] = {
    "diagnostics_frequency": int,
    "initial_timestep":      int,
    "output_trajectory":     str,
    "log_filename":          str,
    "restart_file":          str,
    "final_gsd_filename":    str,
    "hdf5_log_filename":     str,
}


def load_simulparams(json_path: str) -> SimulationParams:
    """
    Read, type-check, and return a SimulationParams from a JSON file.
    Keys starting with '_' are stripped.
    """

    path = Path(json_path)
    # Fail immediately with a clear message rather than letting Python
    # raise a confusing FileNotFoundError inside json.load() later.
    if not path.exists():
        sys.exit(f"[FATAL] Parameter file not found: '{json_path}'")

    try:
        with path.open() as fh:
            raw: dict = json.load(fh)
    except json.JSONDecodeError as exc:
        sys.exit(f"[FATAL] JSON parse error in '{json_path}': {exc}")

    # Strip comment-keys (any key starting with '_')
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}

    # Check required keys are present
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        sys.exit(
            f"[FATAL] Missing required keys in '{json_path}':\n"
            + "\n".join(f"   {k}" for k in missing)
        )

    # Type-check required keys
    type_errors: list[str] = []
    for key, expected in _REQUIRED_KEYS.items():
        if key in raw and not isinstance(raw[key], expected):
            type_errors.append(
                f"   '{key}': expected {expected}, "
                f"got {type(raw[key]).__name__} ({raw[key]!r})"
            )
    if type_errors:
        sys.exit(
            f"[FATAL] Type errors in '{json_path}':\n" + "\n".join(type_errors)
        )

    # Build keyword dict for dataclass
    kw: dict = {k: raw[k] for k in _REQUIRED_KEYS}
    for key in _OPTIONAL_KEYS:
        if key in raw:
            kw[key] = raw[key]

    for float_field in ["shape_scale", "move_size_translation", "move_size_rotation"]:
        kw[float_field] = float(kw[float_field])

    params = SimulationParams(**kw)
    try:
        params.validate()
    except ValueError as exc:
        sys.exit(f"[FATAL] {exc}")

    return params


# ===========================================================================
#  SECTION 5 — Random seed management 
# ===========================================================================

_SEED_FILE_SINGLE = "random_seed.json"         # used when stage_id == -1
_SEED_FILE_MULTI  = "random_seed_stage_0.json"  # used when stage_id >= 0


def _os_random_seed() -> int:
    """
    Cryptographically secure seed in [0, 65535].

    HOOMD-blue (v4 and v5) truncates seeds > 65535 with a warning.
    We cap at 65535 to stay within HOOMD's valid seed range.
    """
    return secrets.randbelow(65536)   # returns [0, 65535]


def ensure_seed_file(stage_id: int, rank: int) -> None:
    """
    Rank-0 creates the seed file on the first call.  All other calls are
    no-ops, ensuring the same seed is reused across restarts and stages.
    """

    if rank != 0:
        return
    seed_file = _SEED_FILE_SINGLE if stage_id == -1 else _SEED_FILE_MULTI
    if not Path(seed_file).exists():
        seed = _os_random_seed()
        with open(seed_file, "w") as fh:
            json.dump(
                {
                    "random_seed": seed,
                    "created_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                fh, indent=4,
            )
        print(f"[INFO] Seed file created: {seed_file}  (seed={seed})")
    else:
        # Subsequent invocations (restarts, later stages): re-use the seed
        # written by the first run to keep trajectories reproducible.
        print(f"[INFO] Existing seed file found: {seed_file}")


def read_seed(stage_id: int) -> int:
    """Return the stored random seed for this stage."""
    seed_file = _SEED_FILE_SINGLE if stage_id == -1 else _SEED_FILE_MULTI
    try:
        with open(seed_file) as fh:
            data = json.load(fh)
        return int(data["random_seed"])
    except (FileNotFoundError, KeyError) as exc:
        # FileNotFoundError : ensure_seed_file() should have created the file
        # KeyError : the file exists but the "random_seed" key is absent,
        sys.exit(f"[FATAL] Cannot read seed from '{seed_file}': {exc}")


# ===========================================================================
#  SECTION 6 — Filename resolution 
# ===========================================================================

@dataclass
class RunFiles:
    """All resolved filenames for the current stage."""
    input_gsd:   str
    traj_gsd:    str
    restart_gsd: str
    final_gsd:   str
    log_txt:     str
    diag_hdf5:   str


def resolve_filenames(params: SimulationParams) -> RunFiles:
    """
    Resolve all output filenames based on stage_id.

    Stage-aware naming — mirrors hs_compress_v7.py:
      stage_id == -1  =>  single-stage: use filenames from JSON directly.
      stage_id >= 0   =>  multi-stage: prefix all files with <tag>_<sid>_
                         and read input from the previous stage's final GSD.

    Guards:
      - Exits if the current stage's final GSD already exists (prevents
        accidental re-runs that would overwrite completed data).
      - Exits if the previous stage's final GSD is missing.
    """
    tag = params.tag
    sid = params.stage_id_current

    if sid == -1:
        # ------------------------------------------------------------------
        # Single-stage mode: use filenames supplied in the JSON
        # ------------------------------------------------------------------
        input_gsd   = params.input_gsd_filename
        traj_gsd    = params.output_trajectory
        restart_gsd = params.restart_file
        final_gsd   = params.final_gsd_filename
        log_txt     = params.log_filename
        diag_hdf5   = params.hdf5_log_filename

        # Warn (not abort) if the final GSD already exists in single-stage mode.
        # We do not exit here (unlike multi-stage) because the user may
        # intentionally re-run a single-stage job with different parameters.
        if Path(final_gsd).exists():
            root_print(
                f"[WARNING] Final GSD '{final_gsd}' already exists. "
                "It will be overwritten at the end of this run."
            )

    elif sid >= 0:
        # ------------------------------------------------------------------
        # Multi-stage mode: <tag>_<stage_id>_<suffix>
        # ------------------------------------------------------------------
        pfx         = f"{tag}_{sid}"
        traj_gsd    = f"{pfx}_traj.gsd"
        restart_gsd = f"{pfx}_restart.gsd"
        final_gsd   = f"{pfx}_final.gsd"
        log_txt     = f"{pfx}.log"
        diag_hdf5   = f"{pfx}_diagnostics.h5"

        # Block re-running a completed stage.  In multi-stage mode this is
        # a hard stop (unlike single-stage) because silently overwriting a
        # completed stage would corrupt the pipeline: the next stage reads
        # this stage's final GSD as its own input.
        if Path(final_gsd).exists():
            sys.exit(
                f"[ERROR] Stage {sid} final GSD '{final_gsd}' already exists.\n"
                f"  => Increment 'stage_id_current' to {sid + 1} in the JSON "
                f"to start the next stage."
            )

        # Determine input GSD for this stage
        if sid == 0:
            input_gsd = params.input_gsd_filename
        else:
            input_gsd = f"{tag}_{sid - 1}_final.gsd"
            if not Path(input_gsd).exists():
                sys.exit(
                    f"[FATAL] Expected input from previous stage not found: "
                    f"'{input_gsd}'.\n"
                    f"  => Did stage {sid - 1} complete successfully?"
                )

    else:
        sys.exit(f"[ERROR] stage_id_current must be >= -1, got {sid}")

    # Final check: the input GSD must exist
    if not Path(input_gsd).exists():
        sys.exit(
            f"[FATAL] Input GSD file not found: '{input_gsd}'\n"
            f"  => Check 'input_gsd_filename' in your JSON file."
        )

    return RunFiles(
        input_gsd=input_gsd,
        traj_gsd=traj_gsd,
        restart_gsd=restart_gsd,
        final_gsd=final_gsd,
        log_txt=log_txt,
        diag_hdf5=diag_hdf5,
    )


# ===========================================================================
#  SECTION 7 — Convex-polyhedron helpers
# ===========================================================================

def packing_fraction(N: int, particle_volume: float, box: hoomd.Box) -> float:
    """Return phi = N * V_particle / V_box."""
    return float(N) * float(particle_volume) / float(box.volume)


def load_convex_polyhedron_shape_from_json(shape_json_filename: str,
                                           scale: float) -> tuple[str, str, float, list]:
    """
    Load the shape JSON and return:
        shape_name, shape_short_name, particle_volume, scaled_vertices
    """
    path = Path(shape_json_filename)

    if not path.exists():
        sys.exit(
            f"[FATAL ERROR] Shape JSON file not found: {shape_json_filename}\n"
            f"              Resolved path: {path.absolute()}\n"
            f"              => Check shape_json_filename in the simulation parameter file."
        )

    try:
        with path.open("r", encoding="utf-8") as fh:
            shape_cfg = json.load(fh)
    except json.JSONDecodeError as exc:
        sys.exit(
            f"[FATAL ERROR] Invalid JSON syntax in shape file '{shape_json_filename}': "
            f"line {exc.lineno}, column {exc.colno}\n"
        )
    except Exception as exc:
        sys.exit(
            f"[FATAL ERROR] Failed to open shape JSON '{shape_json_filename}': {exc}\n"
        )

    required_keys = ["4_volume", "8_vertices"]
    missing = [key for key in required_keys if key not in shape_cfg]
    if missing:
        sys.exit(
            f"[FATAL ERROR] Shape JSON '{shape_json_filename}' is missing required key(s): "
            f"{', '.join(missing)}\n"
        )

    try:
        reference_volume = float(shape_cfg["4_volume"])
    except (TypeError, ValueError):
        sys.exit(
            f"[FATAL ERROR] Shape JSON '{shape_json_filename}' contains a non-numeric "
            f"'4_volume' entry.\n"
        )

    if reference_volume <= 0.0:
        sys.exit(
            f"[FATAL ERROR] Shape JSON '{shape_json_filename}' has non-positive "
            f"reference volume: {reference_volume}\n"
        )

    try:
        reference_vertices = np.asarray(shape_cfg["8_vertices"], dtype=np.float64)
    except Exception as exc:
        sys.exit(
            f"[FATAL ERROR] Could not parse vertices from shape JSON "
            f"'{shape_json_filename}': {exc}\n"
        )

    if reference_vertices.ndim != 2 or reference_vertices.shape[1] != 3:
        sys.exit(
            f"[FATAL ERROR] Shape JSON '{shape_json_filename}' does not contain a valid "
            f"vertex array of shape (Nv, 3).\n"
        )

    if len(reference_vertices) < 4:
        sys.exit(
            f"[FATAL ERROR] Shape JSON '{shape_json_filename}' contains too few vertices "
            f"for a 3D convex polyhedron.\n"
        )

    if scale <= 0.0:
        sys.exit(f"[FATAL ERROR] shape_scale must be > 0, got {scale}\n")

    scaled_vertices = (reference_vertices * float(scale)).tolist()
    particle_volume = reference_volume * float(scale) ** 3
    shape_name = str(shape_cfg.get("1_Name", path.stem))
    shape_short_name = str(shape_cfg.get("2_ShortName", shape_name))
    return shape_name, shape_short_name, particle_volume, scaled_vertices


def _safe_get_particle_array(frame, attr_name: str, dtype=None):
    """
    Read a particle array from a GSD frame robustly.

    Returns None when the field is absent or malformed. This is used mainly for
    optional particle information such as image, mass, velocity, and angular
    momentum.
    """
    try:
        value = getattr(frame.particles, attr_name)
    except Exception:
        return None

    if value is None:
        return None

    try:
        arr = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    except Exception:
        return None

    if arr.size == 0:
        return None
    return arr


# ===========================================================================
#  SECTION 8 — Snapshot broadcast 
# ===========================================================================

def load_and_broadcast_snapshot(
    input_gsd: str,
    comm,
    rank: int,
    expected_vertices: Optional[list] = None,
) -> dict:
    """
    Read the last frame of the input GSD on rank 0 and broadcast the minimal
    state needed to reconstruct a fresh HOOMD snapshot on all ranks.

    The function also checks, when present, whether particles/type_shapes in the
    input GSD are consistent with the supplied convex-polyhedron vertices.
    """
    if rank == 0:
        path = Path(input_gsd)
        if not path.exists():
            sys.exit(f"[FATAL ERROR] Input GSD not found: '{input_gsd}'")

        try:
            with gsd.hoomd.open(name=str(path), mode="r") as traj:
                if len(traj) == 0:
                    sys.exit(f"[FATAL ERROR] Input GSD '{input_gsd}' has zero frames.")
                frame = traj[-1]
        except Exception as exc:
            sys.exit(f"[FATAL ERROR] Failed to read input GSD '{input_gsd}': {exc}")

        N = int(frame.particles.N)
        if N <= 0:
            sys.exit(
                f"[FATAL ERROR] Input GSD '{input_gsd}' contains zero particles."
            )

        positions = _safe_get_particle_array(frame, "position", np.float64)
        if positions is None or positions.shape != (N, 3):
            sys.exit(
                f"[FATAL ERROR] Input GSD '{input_gsd}' does not contain a valid "
                f"(N, 3) position array."
            )

        typeid = _safe_get_particle_array(frame, "typeid", np.int32)
        if typeid is None or typeid.shape != (N,):
            sys.exit(
                f"[FATAL ERROR] Input GSD '{input_gsd}' does not contain a valid "
                f"typeid array."
            )

        unique_typeids = np.unique(typeid)
        if unique_typeids.size != 1 or int(unique_typeids[0]) != 0:
            sys.exit(
                f"[FATAL ERROR] This NVT script assumes a one-component system "
                f"with particle typeid = 0 only.\n"
                f"              Found unique typeids: {unique_typeids.tolist()}"
            )

        try:
            types = list(frame.particles.types)
        except Exception as exc:
            sys.exit(
                f"[FATAL ERROR] Failed to read particle types from '{input_gsd}': {exc}"
            )

        if not types:
            sys.exit(f"[FATAL ERROR] Input GSD '{input_gsd}' contains no particle types.")

        orientation = _safe_get_particle_array(frame, "orientation", np.float64)
        inserted_identity_orientation = False
        if orientation is None or orientation.shape != (N, 4):
            orientation = np.zeros((N, 4), dtype=np.float64)
            orientation[:, 0] = 1.0
            inserted_identity_orientation = True
        else:
            if not np.all(np.isfinite(orientation)):
                sys.exit(
                    f"[FATAL ERROR] Input GSD '{input_gsd}' contains non-finite "
                    f"orientation values."
                )
            norms = np.linalg.norm(orientation, axis=1)
            if np.any(norms <= 0.0):
                sys.exit(
                    f"[FATAL ERROR] Input GSD '{input_gsd}' contains zero-norm "
                    f"quaternions."
                )

        try:
            type_shapes = frame.particles.type_shapes
        except Exception:
            type_shapes = None

        if expected_vertices is not None and type_shapes:
            try:
                first_shape = type_shapes[0]
                gsd_vertices = np.asarray(first_shape.get("vertices", []), dtype=np.float64)
                ref_vertices = np.asarray(expected_vertices, dtype=np.float64)

                if gsd_vertices.shape != ref_vertices.shape:
                    print("\n" + "=" * 90, file=sys.stderr, flush=True)
                    print("[DEBUG] Vertex-shape mismatch during GSD validation", file=sys.stderr, flush=True)
                    print("=" * 90, file=sys.stderr, flush=True)
                    print(f"gsd_vertices.shape = {gsd_vertices.shape}", file=sys.stderr, flush=True)
                    print(f"ref_vertices.shape = {ref_vertices.shape}", file=sys.stderr, flush=True)

                    print("\n[DEBUG] gsd_vertices =", file=sys.stderr, flush=True)
                    print(gsd_vertices.tolist(), file=sys.stderr, flush=True)

                    print("\n[DEBUG] ref_vertices =", file=sys.stderr, flush=True)
                    print(ref_vertices.tolist(), file=sys.stderr, flush=True)

                    sys.exit(
                        f"[FATAL ERROR] The convex-polyhedron vertices stored in the input GSD "
                        f"have shape {gsd_vertices.shape}, while the vertices reconstructed "
                        f"from shape_json_filename and shape_scale have shape {ref_vertices.shape}."
                    )

                def _sort_rows(arr):
                    idx = np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))
                    return arr[idx]

                gsd_vertices_sorted = _sort_rows(gsd_vertices)
                ref_vertices_sorted = _sort_rows(ref_vertices)

                vertex_atol = 1.0e-4

                direct_match = np.allclose(gsd_vertices, ref_vertices, rtol=0.0, atol=vertex_atol)
                sorted_match = np.allclose(gsd_vertices_sorted, ref_vertices_sorted,
                                        rtol=0.0, atol=vertex_atol)

                if not sorted_match:
                    print("\n" + "=" * 90, file=sys.stderr, flush=True)
                    print("[DEBUG] Convex-polyhedron vertex mismatch during GSD validation",
                        file=sys.stderr, flush=True)
                    print("=" * 90, file=sys.stderr, flush=True)

                    print(f"Direct order-sensitive match : {direct_match}", file=sys.stderr, flush=True)
                    print(f"Sorted order-insensitive match: {sorted_match}", file=sys.stderr, flush=True)

                    print("\n[DEBUG] Raw gsd_vertices =", file=sys.stderr, flush=True)
                    print(gsd_vertices.tolist(), file=sys.stderr, flush=True)

                    print("\n[DEBUG] Raw ref_vertices =", file=sys.stderr, flush=True)
                    print(ref_vertices.tolist(), file=sys.stderr, flush=True)

                    print("\n[DEBUG] Sorted gsd_vertices =", file=sys.stderr, flush=True)
                    print(gsd_vertices_sorted.tolist(), file=sys.stderr, flush=True)

                    print("\n[DEBUG] Sorted ref_vertices =", file=sys.stderr, flush=True)
                    print(ref_vertices_sorted.tolist(), file=sys.stderr, flush=True)

                    diff = gsd_vertices_sorted - ref_vertices_sorted
                    absdiff = np.abs(diff)
                    max_abs_diff = np.max(absdiff)

                    print(f"\n[DEBUG] max_abs_diff = {max_abs_diff:.16e}",
                        file=sys.stderr, flush=True)

                    mismatch_rows = np.where(np.any(absdiff > vertex_atol, axis=1))[0]
                    print(f"[DEBUG] mismatch row indices = {mismatch_rows.tolist()}",
                        file=sys.stderr, flush=True)

                    if mismatch_rows.size > 0:
                        i = int(mismatch_rows[0])
                        print("\n[DEBUG] First mismatching sorted row:", file=sys.stderr, flush=True)
                        print(f"  row index         = {i}", file=sys.stderr, flush=True)
                        print(f"  gsd_vertices[{i}] = {gsd_vertices_sorted[i].tolist()}",
                            file=sys.stderr, flush=True)
                        print(f"  ref_vertices[{i}] = {ref_vertices_sorted[i].tolist()}",
                            file=sys.stderr, flush=True)
                        print(f"  difference        = {diff[i].tolist()}",
                            file=sys.stderr, flush=True)

                    sys.exit(
                        f"[FATAL ERROR] The convex-polyhedron vertices stored in the input GSD "
                        f"do not match the vertices reconstructed from shape_json_filename "
                        f"and shape_scale.\n"
                        f"              => See the debug dump above for the exact mismatch."
                    )

            except Exception as exc:
                sys.exit(
                    f"[FATAL ERROR] Failed while validating GSD type_shapes: {exc}"
                )


        snap_data = {
            "box": list(frame.configuration.box),
            "N": N,
            "positions": positions,
            "typeid": typeid,
            "types": types,
            "orientation": orientation,
            "inserted_identity_orientation": inserted_identity_orientation,
            "image": _safe_get_particle_array(frame, "image", np.int32),
            "body": _safe_get_particle_array(frame, "body", np.int32),
            "mass": _safe_get_particle_array(frame, "mass", np.float64),
            "charge": _safe_get_particle_array(frame, "charge", np.float64),
            "moment_inertia": _safe_get_particle_array(frame, "moment_inertia", np.float64),
            "velocity": _safe_get_particle_array(frame, "velocity", np.float64),
            "angmom": _safe_get_particle_array(frame, "angmom", np.float64),
        }
    else:
        snap_data = None

    snap_data = comm.bcast(snap_data, root=0)
    return snap_data




def reconstruct_snapshot(snap_data: dict, rank: int) -> hoomd.Snapshot:
    """
    Reconstruct a hoomd.Snapshot from the broadcast data dict.

    On rank 0: populates all fields.
    On non-root ranks: returns an empty Snapshot (HOOMD fills it from
    rank 0 during create_state_from_snapshot).

    Preserved exactly from the original script.
    """
    snapshot = hoomd.Snapshot()

    if rank != 0:
        return snapshot

    snapshot.configuration.box = list(snap_data["box"])
    snapshot.particles.N = int(snap_data["N"])
    snapshot.particles.types = list(snap_data["types"])
    snapshot.particles.position[:] = np.asarray(snap_data["positions"], dtype=np.float64)
    snapshot.particles.typeid[:] = np.asarray(snap_data["typeid"], dtype=np.uint32)
    snapshot.particles.orientation[:] = np.asarray(snap_data["orientation"], dtype=np.float64)

    if snap_data.get("diameter") is not None:
        try:
            snapshot.particles.diameter[:] = np.asarray(snap_data["diameter"], dtype=np.float64)
        except Exception:
            pass
    if snap_data.get("image") is not None:
        try:
            snapshot.particles.image[:] = np.asarray(snap_data["image"], dtype=np.int32)
        except Exception:
            pass
    if snap_data.get("body") is not None:
        try:
            snapshot.particles.body[:] = np.asarray(snap_data["body"], dtype=np.int32)
        except Exception:
            pass
    if snap_data.get("mass") is not None:
        try:
            snapshot.particles.mass[:] = np.asarray(snap_data["mass"], dtype=np.float64)
        except Exception:
            pass
    if snap_data.get("charge") is not None:
        try:
            snapshot.particles.charge[:] = np.asarray(snap_data["charge"], dtype=np.float64)
        except Exception:
            pass
    if snap_data.get("moment_inertia") is not None:
        try:
            snapshot.particles.moment_inertia[:] = np.asarray(
                snap_data["moment_inertia"], dtype=np.float64
            )
        except Exception:
            pass
    if snap_data.get("velocity") is not None:
        try:
            snapshot.particles.velocity[:] = np.asarray(snap_data["velocity"], dtype=np.float64)
        except Exception:
            pass
    if snap_data.get("angmom") is not None:
        try:
            snapshot.particles.angmom[:] = np.asarray(snap_data["angmom"], dtype=np.float64)
        except Exception:
            pass

    return snapshot


def _build_type_shapes_metadata(mc: hoomd.hpmc.integrate.ConvexPolyhedron) -> list:
    """
    Return a serializable copy of the HPMC integrator's type_shapes metadata.

    HOOMD exposes type_shapes as a loggable "object" quantity. This helper is
    used when manually writing single-frame GSD files so that they remain
    self-describing for OVITO and other readers that understand
    particles/type_shapes.
    """
    try:
        return list(mc.type_shapes)
    except Exception as exc:
        sys.exit(
            f"[FATAL ERROR] Could not obtain convex-polyhedron type_shapes from the "
            f"HPMC integrator: {exc}"
        )


# ===========================================================================
#  SECTION 9 — Simulation builder
# ===========================================================================

def build_simulation(
    params: SimulationParams,
    files:  RunFiles,
    seed:   int,
    comm,           # MPI.COMM_WORLD or stub
    rank:   int,
) -> tuple[hoomd.Simulation, hoomd.hpmc.integrate.ConvexPolyhedron, object]:
    """
    Build and fully configure the HOOMD v4 NVT simulation.

    Steps
    -----
    1.  Device selection (CPU or GPU).
    2.  Create Simulation with the resolved seed.
    3.  Load state:
          - If restart GSD exists and final GSD is absent => resume from
            checkpoint (crashed / walltime-interrupted run).
          - Otherwise => fresh run from input GSD via MPI broadcast.
    4.  Configure HPMC ConvexPolyhedron integrator with fixed move sizes.
    5.  Attach Table log writer (human-readable .log file).
    6.  Attach GSD trajectory writer (appending).
    7.  Attach GSD restart writer (truncating checkpoint).

    Returns
    -------
    sim          : hoomd.Simulation
    mc           : HPMC ConvexPolyhedron integrator
    log_file_hdl : open file handle for the Table log (must be closed later)
    """

    # ------------------------------------------------------------------
    # 9.1  Device selection  
    # ------------------------------------------------------------------
    if params.use_gpu:
        try:
            device = hoomd.device.GPU(gpu_ids=[params.gpu_id])
            root_print(f"[INFO] Running on GPU {params.gpu_id}")
        except Exception as exc:
            # hoomd.device.GPU raises RuntimeError if CUDA is not available,
            # the specified gpu_id is out of range, or the GPU lacks the
            # required compute capability.  We fall back to CPU so the job
            # does not crash on a node where the GPU is unavailable.
            root_print(
                f"[WARNING] GPU initialisation failed ({exc}). "
                "Falling back to CPU."
            )
            device = hoomd.device.CPU()
    else:
        device = hoomd.device.CPU()
        root_print("[INFO] Running on CPU")

    # ------------------------------------------------------------------
    # 9.2  Simulation object
    # ------------------------------------------------------------------
    try:
        sim = hoomd.Simulation(device=device, seed=seed)
    except Exception as exc:
        sys.exit(
            f"[FATAL ERROR] Failed to create HOOMD Simulation object: {exc}"
        )

    try:
        (
            params.shape_name,
            params.shape_short_name,
            params.particle_volume,
            params.shape_vertices,
        ) = load_convex_polyhedron_shape_from_json(
            params.shape_json_filename, params.shape_scale
        )
    except SystemExit:
        raise

    # ------------------------------------------------------------------
    # 9.3  State initialisation    (restart-aware)
    # ------------------------------------------------------------------
    # Determine the run mode by inspecting which GSD files exist on disk.
    # Three possible states:
    #   restart_exists=True,  final_exists=False  => crashed/walltime restart
    #   restart_exists=True,  final_exists=True   => completed run; treat as fresh
    #   restart_exists=False, final_exists=False  => clean first run
    # (restart_exists=False, final_exists=True is possible but unusual;
    #    treated as fresh since there is no checkpoint to resume from)
    restart_exists = Path(files.restart_gsd).exists()
    final_exists   = Path(files.final_gsd).exists()

    if restart_exists and not final_exists:
        # -----------------------------------------------------------------
        # RESTART CASE: previous run was interrupted (walltime / crash).
        # Load state directly from the restart checkpoint GSD.
        # All particle data (positions, box, timestep) are restored.
        # -----------------------------------------------------------------
        root_print(
            f"[INFO] Restart GSD found: '{files.restart_gsd}'. "
            "Resuming from checkpoint."
        )
        # create_state_from_gsd works identically on all MPI ranks;
        # HOOMD handles domain-decomposition internally.
        try:
            sim.create_state_from_gsd(filename=files.restart_gsd)
        except Exception as exc:
            sys.exit(
                f"[FATAL ERROR] Failed to load restart GSD '{files.restart_gsd}': {exc}"
            )
        state_source = files.restart_gsd
        inserted_identity_orientation = False
        root_print("[INFO] Restart run: True")

    else:
        # -----------------------------------------------------------------
        # FRESH RUN CASE: no restart, or final already exists (re-run).
        # Use the MPI-safe broadcast pattern from the original script.
        # -----------------------------------------------------------------
        if final_exists:
            root_print(
                f"[WARNING] Final GSD '{files.final_gsd}' already exists. "
                "Starting fresh (it will be overwritten at completion)."
            )

        root_print(f"[INFO] Fresh run from: '{files.input_gsd}'")

        snap_data = load_and_broadcast_snapshot(
            files.input_gsd,
            comm=comm,
            rank=rank,
            expected_vertices=params.shape_vertices,
        )
        inserted_identity_orientation = bool(
            snap_data.get("inserted_identity_orientation", False)
        )
        snapshot = reconstruct_snapshot(snap_data, rank)        

        try:
            sim.create_state_from_snapshot(snapshot)
        except Exception as exc:
            sys.exit(
                f"[FATAL ERROR] Failed to create state from reconstructed snapshot: {exc}"
            )
        if params.initial_timestep > 0:
            try:
                sim.timestep = int(params.initial_timestep)
            except Exception:
                root_print(
                    f"[WARNING] Could not set initial timestep to {params.initial_timestep}. "
                    "Continuing with HOOMD default."
                )
        state_source = files.input_gsd
        root_print("[INFO] Restart run: False")

    # ------------------------------------------------------------------
    # 9.4  Sanity checks and initial reporting
    # ------------------------------------------------------------------
    N = sim.state.N_particles
    if N <= 0:
        sys.exit(
            f"[FATAL ERROR] Loaded state contains zero particles.\n"
            f"              Source: {state_source}"
        )

    phi = packing_fraction(N, params.particle_volume, sim.state.box)
    box = sim.state.box

    root_print(
        f"[INFO] Shape loaded successfully:\n"
        f"       shape_name         = {params.shape_name}\n"
        f"       shape_short_name   = {params.shape_short_name}\n"
        f"       shape_json         = {params.shape_json_filename}\n"
        f"       shape_scale        = {params.shape_scale}\n"
        f"       particle_volume    = {params.particle_volume}\n"
        f"       vertices           = {len(params.shape_vertices)}"
    )
    root_print(
        f"[INFO] N={N} | phi={phi:.6f} | box={sim.state.box}"
    )
    root_print(
        f"[INFO] Box: Lx={box.Lx:.4f}  Ly={box.Ly:.4f}  Lz={box.Lz:.4f}  "
        f"xy={box.xy}  xz={box.xz}  yz={box.yz}"
    )
    if inserted_identity_orientation:
        root_print(
            "[WARNING] Input GSD did not contain a valid (N,4) orientation array.\n"
            "          Identity quaternions [1,0,0,0] were inserted for the fresh run."
        )

    # ------------------------------------------------------------------
    # 9.5  HPMC ConvexPolyhedron integrator
    # ------------------------------------------------------------------
    try:
        mc = hoomd.hpmc.integrate.ConvexPolyhedron(nselect=1)
        mc.shape["A"] = {"vertices": params.shape_vertices}
        mc.d["A"] = params.move_size_translation
        mc.a["A"] = params.move_size_rotation
        sim.operations.integrator = mc
    except Exception as exc:
        sys.exit(
            f"[FATAL ERROR] HPMC ConvexPolyhedron integrator configuration failed: {exc}"
        )

    root_print(
        f"[INFO] HPMC ConvexPolyhedron configured | "
        f"move_size_d = {params.move_size_translation} | "
        f"move_size_a = {params.move_size_rotation}"
    )

    # run(0) initializes overlap checks and makes type_shapes safely available
    try:
        sim.run(0)
    except Exception as exc:
        sys.exit(
            f"[FATAL ERROR] sim.run(0) failed during initialization: {exc}"
        )

    try:
        overlaps_initial = int(mc.overlaps)
    except Exception as exc:
        sys.exit(
            f"[FATAL ERROR] Could not query initial overlaps after initialization: {exc}"
        )

    if overlaps_initial > 0:
        sys.exit(
            f"[FATAL ERROR] The starting convex-polyhedron configuration contains "
            f"{overlaps_initial} overlaps.\n"
            f"              Source GSD: {state_source}\n"
            f"              This NVT script does not remove overlaps.\n"
            f"              => Start from a valid convex-polyhedron GSD or use your "
            f"compression / overlap-removal workflow first."
        )

    root_print(f"[INFO] Initial overlap count = {overlaps_initial}")

    # **** Parallel Section ****
    # MC moves are distributed across MPI processes; each handles a domain.

    # ------------------------------------------------------------------
    # 9.6  Custom loggable instances
    # ------------------------------------------------------------------
    # All properties guarded against DataAccessError 
    status        = Status(sim)
    mc_status     = MCStatus(mc)
    box_prop      = Box_property(sim, N, params.particle_volume)
    overlap_count = OverlapCount(mc)
    step_prop     = CurrentTimestep(sim)

    # ------------------------------------------------------------------
    # 9.7  Logger + Table writer => human-readable .log file  [N-07]
    # ------------------------------------------------------------------
    # hoomd.logging.Logger collects loggable quantities from HOOMD objects
    # and custom loggable classes.  "scalar" captures single float/int values;
    # "string" captures text (used for ETR, timestep_fraction).
    # only_default=False allows registering custom quantities that are not
    # part of HOOMD's built-in default loggable set.
    logger = hoomd.logging.Logger(
        categories=["scalar", "string"],
        only_default=False,
    )

    # Built-in HOOMD quantities
    logger.add(sim, quantities=["tps", "walltime"])

    # Custom quantities (same as original script, plus new ones)
    logger[("Simulation", "timestep")]          = (status,        "timestep_fraction", "string")
    logger[("Status",     "etr")]               = (status,        "etr",               "string")
    logger[("MCStatus", "translate_acceptance_rate")] = (mc_status, "translate_acceptance_rate", "scalar")
    logger[("MCStatus", "rotate_acceptance_rate")] = (mc_status, "rotate_acceptance_rate", "scalar")
    logger[("Box",        "volume")]            = (box_prop,      "volume",            "scalar")
    logger[("Box",        "packing_fraction")]  = (box_prop,      "packing_fraction",  "scalar")   
    logger[("HPMC",       "overlap_count")]     = (overlap_count, "overlap_count",     "scalar")   

    try:
        log_file_hdl = open(files.log_txt, "w")
    except Exception as exc:
        sys.exit(
            f"[FATAL ERROR] Could not open text log file '{files.log_txt}' for writing: {exc}"
        )  

    log_writer = hoomd.write.Table(
        output=log_file_hdl,
        trigger=hoomd.trigger.Periodic(params.log_frequency),
        logger=logger,
    )
    # **** Parallel Section ****
    sim.operations.writers.append(log_writer)

    # ------------------------------------------------------------------
    # 9.8  GSD trajectory writer 
    # ------------------------------------------------------------------
    # Official HOOMD v4.9.0 guidance for storing particle shape uses a Logger
    # with mc.type_shapes attached to the GSD writer. That writes the shape
    # metadata into particles/type_shapes for trajectory analysis and OVITO.
    shape_logger = hoomd.logging.Logger(only_default=False)
    shape_logger.add(mc, quantities=["type_shapes"])

    traj_mode = "ab" if (restart_exists and not final_exists) else "wb"

    # dynamic=["property","attribute"] instructs GSD to write the "property"
    # category (which includes particles/diameter, particles/position,
    # particles/typeid, particles/mass, etc.) and the "attribute" category
    # (particles/N, particles/types, box dimensions) to every frame.
    traj_gsd_writer = hoomd.write.GSD(
        filename=files.traj_gsd,
        filter=hoomd.filter.All(),
        trigger=hoomd.trigger.Periodic(params.traj_gsd_frequency),
        mode=traj_mode,
        dynamic=["property", "attribute"],
        logger=shape_logger,
    )
    sim.operations.writers.append(traj_gsd_writer)

    # ------------------------------------------------------------------
    # 9.9  GSD restart writer  (single-frame truncating checkpoint)
    # ------------------------------------------------------------------
    restart_gsd_writer = hoomd.write.GSD(
        filename=files.restart_gsd,
        filter=hoomd.filter.All(),
        trigger=hoomd.trigger.Periodic(params.restart_gsd_frequency),
        truncate=True,
        mode="wb",
        dynamic=["property", "attribute"],
        logger=shape_logger,
    )
    sim.operations.writers.append(restart_gsd_writer)

    # ------------------------------------------------------------------
    # 9.10  Optional machine-readable diagnostics writer 
    # ------------------------------------------------------------------
    diagnostics_period = (
        params.diagnostics_frequency
        if params.diagnostics_frequency > 0
        else params.log_frequency
    )
    # HDF5 file mode mirrors the trajectory logic:
    #   "a" (append) for true restarts so existing datasets are extended.
    #   "w" (overwrite) for fresh runs so stale data from a previous run
    #       is not mixed with new data.
    hdf5_mode = "a" if (restart_exists and not final_exists) else "w"

    try:
        hdf5_logger = hoomd.logging.Logger(
            hoomd.write.HDF5Log.accepted_categories,
            only_default=False,
        )
        hdf5_logger.add(sim, quantities=["tps", "walltime"])
        hdf5_logger[("Simulation", "timestep")] = (step_prop, "timestep", "scalar")
        hdf5_logger[("MCStatus", "translate_acceptance_rate")] = (mc_status, "translate_acceptance_rate", "scalar")
        hdf5_logger[("MCStatus", "rotate_acceptance_rate")] = (mc_status, "rotate_acceptance_rate", "scalar")
        hdf5_logger[("Box", "packing_fraction")] = (box_prop, "packing_fraction", "scalar")
        hdf5_logger[("HPMC", "overlap_count")] = (overlap_count, "overlap_count", "scalar")
        # translate_moves: tuple (accepted, rejected) — total move statistics.
        # mps: Monte Carlo moves per second — useful for performance monitoring.
        hdf5_logger.add(mc, quantities=["translate_moves", "rotate_moves", "mps"])

        hdf5_writer = hoomd.write.HDF5Log(
            trigger=hoomd.trigger.Periodic(diagnostics_period),
            filename=files.diag_hdf5,
            logger=hdf5_logger,
            mode=hdf5_mode,
        )
        sim.operations.writers.append(hdf5_writer)
        root_print(
            f"[INFO] HDF5 diagnostics  : {files.diag_hdf5} "
            f"(every {diagnostics_period} steps)"
        )
    except Exception as exc:
        # Common reasons for failure:
        #   ImportError   : h5py is not installed in this Python environment.
        #   AttributeError: hoomd.write.HDF5Log does not exist in this build
        #                   (HOOMD compiled without HDF5 support).
        #   OSError       : the output directory is not writable.
        # In all cases we issue a warning and continue without HDF5 output.
        root_print(
            f"[WARNING] HDF5 diagnostics writer disabled: {type(exc).__name__}: {exc}"
        )

    return sim, mc, log_file_hdl


# ===========================================================================
#  SECTION 10 — Final output writing 
# ===========================================================================

def _write_snapshot_with_shape(sim: hoomd.Simulation,
                               filename: str,
                               type_shapes: list) -> None:
    """
    Write a single-frame GSD that explicitly contains:
      - positions
      - orientations
      - type ids / types
      - particles/type_shapes
      - optional particle information when available

    This helper is used for the final snapshot and emergency snapshots.
    """
    snap = sim.state.get_snapshot()

    if not _is_root_rank():
        return

    frame = gsd.hoomd.Frame()
    box = sim.state.box

    frame.configuration.step = int(sim.timestep)
    frame.configuration.box = [box.Lx, box.Ly, box.Lz, box.xy, box.xz, box.yz]

    frame.particles.N = int(snap.particles.N)
    frame.particles.position = np.asarray(snap.particles.position, dtype=np.float32)
    frame.particles.typeid = np.asarray(snap.particles.typeid, dtype=np.uint32)
    frame.particles.types = list(snap.particles.types)

    try:
        frame.particles.orientation = np.asarray(
            snap.particles.orientation, dtype=np.float32
        )
    except Exception:
        sys.exit(
            f"[FATAL ERROR] Simulation snapshot does not contain particle orientations.\n"
            f"              Cannot write convex-polyhedron GSD '{filename}'."
        )

    frame.particles.type_shapes = list(type_shapes)

    # Preserve optional particle information when present.
    for attr_name, dtype in [
        ("diameter", np.float32),
        ("image", np.int32),
        ("body", np.int32),
        ("mass", np.float32),
        ("charge", np.float32),
        ("moment_inertia", np.float32),
        ("velocity", np.float32),
        ("angmom", np.float32),
    ]:
        try:
            value = getattr(snap.particles, attr_name)
            if value is not None:
                setattr(frame.particles, attr_name, np.asarray(value, dtype=dtype))
        except Exception:
            pass

    with gsd.hoomd.open(name=filename, mode="w") as traj:
        traj.append(frame)



def _verify_gsd_type_shapes(filename: str,
                            expected_vertices: list,
                            label: str = "output") -> None:
    """Verify that a written GSD stores the expected convex-polyhedron shape."""
    if not _is_root_rank():
        return

    with gsd.hoomd.open(name=filename, mode="r") as traj:
        if len(traj) == 0:
            sys.exit(f"[FATAL ERROR] {label} GSD '{filename}' has no frames.")
        frame = traj[-1]

    try:
        type_shapes = frame.particles.type_shapes
    except Exception:
        type_shapes = None

    if not type_shapes:
        sys.exit(
            f"[FATAL ERROR] {label} GSD '{filename}' does not contain particles/type_shapes.\n"
            f"              => The file is not self-describing for convex-polyhedron visualization."
        )

    first_shape = type_shapes[0]

    if first_shape.get("type", None) != "ConvexPolyhedron":
        sys.exit(
            f"[FATAL ERROR] {label} GSD '{filename}' has unexpected shape type: "
            f"{first_shape.get('type', None)}"
        )

    stored_vertices = np.asarray(first_shape.get("vertices", []), dtype=np.float64)
    ref_vertices = np.asarray(expected_vertices, dtype=np.float64)

    if stored_vertices.shape != ref_vertices.shape:
        sys.exit(
            f"[FATAL ERROR] {label} GSD '{filename}' stores vertices with shape "
            f"{stored_vertices.shape}, expected {ref_vertices.shape}."
        )

    if not np.allclose(stored_vertices, ref_vertices, rtol=0.0, atol=1e-8):
        sys.exit(
            f"[FATAL ERROR] {label} GSD '{filename}' contains shape vertices that do "
            f"not match the expected convex-polyhedron vertices."
        )

    root_print(
        f"[VERIFY] {label.capitalize()} GSD type_shapes OK => "
        f"file='{filename}' | vertices={len(ref_vertices)}"
    )



def write_final_outputs(
    sim: hoomd.Simulation,
    mc: hoomd.hpmc.integrate.ConvexPolyhedron,
    params: SimulationParams,
    files: RunFiles,
    start_time: float,
    seed: int,
    json_path: str,
) -> None:
    """Write final GSD + summary JSON + console banner."""
    N = sim.state.N_particles
    phi = packing_fraction(N, params.particle_volume, sim.state.box)
    runtime = time.time() - start_time
    type_shapes = _build_type_shapes_metadata(mc)

    _write_snapshot_with_shape(sim, files.final_gsd, type_shapes)
    root_print(f"[OUTPUT] Final GSD        => {files.final_gsd}")
    _verify_gsd_type_shapes(files.final_gsd, params.shape_vertices, label="final")

    summary = {
        "tag": params.tag,
        "stage_id": params.stage_id_current,
        "simulparam_file": json_path,
        "input_gsd": files.input_gsd,
        "traj_gsd": files.traj_gsd,
        "diagnostics_hdf5": files.diag_hdf5,
        "final_gsd": files.final_gsd,
        "shape_json_filename": params.shape_json_filename,
        "shape_name": params.shape_name,
        "shape_short_name": params.shape_short_name,
        "shape_scale": params.shape_scale,
        "particle_volume": params.particle_volume,
        "n_particles": N,
        "packing_fraction": round(phi, 8),
        "move_size_d": params.move_size_translation,
        "move_size_a": params.move_size_rotation,
        "diagnostics_frequency": (
            params.diagnostics_frequency if params.diagnostics_frequency > 0
            else params.log_frequency
        ),
        "total_timesteps": params.total_num_timesteps,
        "final_timestep": sim.timestep,
        "overlaps_final": int(mc.overlaps),
        "random_seed": seed,
        "box_final": {
            "Lx": sim.state.box.Lx,
            "Ly": sim.state.box.Ly,
            "Lz": sim.state.box.Lz,
            "xy": sim.state.box.xy,
            "xz": sim.state.box.xz,
            "yz": sim.state.box.yz,
        },
        "runtime_seconds": round(runtime, 2),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    summary_file = f"{params.tag}_stage{params.stage_id_current}_nvt_summary.json"
    # Only rank 0 writes the summary JSON to avoid N identical files being
    # written simultaneously to the same path, which would corrupt the output.
    if _is_root_rank():
        with open(summary_file, "w") as fh:
            json.dump(summary, fh, indent=2)
        root_print(f"[OUTPUT] Summary JSON     => {summary_file}")

    # --- Console summary banner -------------------------------------------
    _print_banner("HOOMD-blue v4 | Convex-Polyhedron NVT Equilibration | Run Complete")
    lines = [
        f"  Simulparam file       : {json_path}",
        f"  Tag                   : {params.tag}",
        f"  Stage id              : {params.stage_id_current}",
        f"  Input GSD             : {files.input_gsd}",
        f"  Final GSD             : {files.final_gsd}",
        f"  Diagnostics HDF5      : {files.diag_hdf5}",
        f"  Particles (N)         : {N}",
        f"  Shape name              : {params.shape_name}",
        f"  Shape short name        : {params.shape_short_name}",
        f"  Shape JSON              : {params.shape_json_filename}",
        f"  Shape scale             : {params.shape_scale}",
        f"  Particle volume         : {params.particle_volume}",
        f"  Packing fraction (phi)  : {phi:.6f}",
        f"  Move size d             : {params.move_size_translation}",
        f"  Move size a             : {params.move_size_rotation}",
        f"  Total timesteps run   : {params.total_num_timesteps}",
        f"  Final timestep        : {sim.timestep}",
        f"  Overlaps at end       : {mc.overlaps}",
        f"  Random seed           : {seed}",
        f"  Total runtime         : {runtime:.2f} s",
    ]
    root_print("\n".join(lines))
    _print_banner("")


# ===========================================================================
#  SECTION 11 — Pretty-print helpers
# ===========================================================================

def _print_banner(title: str, width: int = 70) -> None:
    """Print a '*'-bordered banner line """
    bar = "*" * width
    root_print(f"\n{bar}")
    if title:
        root_print(f"  {title}")
        root_print(bar)


# ===========================================================================
#  SECTION 12 — Entry point
# ===========================================================================

# Module-level variable to hold the active JSON path (used in summary banner)
_active_json_path: str = ""


def main() -> None:
    """
    Main entry point.

    Workflow
    --------
    1.  Parse CLI arguments.
    2.  Print HOOMD version banner (preserved from original script).
    3.  Load and validate simulation parameters from JSON.
    4.  Initialise MPI communicator.
    5.  Resolve filenames (stage-aware).
    6.  Manage random seed file (rank-0 writes; all read).
    7.  Build simulation (device, state, integrator, writers).
    8.  Run the simulation.
    9.  Write final outputs and summary.
    10. Close log file handle.

    Exception handling
    ------------------
    - JSON / file errors   → sys.exit with clear message.
    - Unexpected exceptions → emergency snapshot written, then re-raised.
    - Log file handle      → closed in a finally block [N-12].
    """
    global _active_json_path

    # Record simulation start time (preserved from original script)
    start_time = time.time()

    # ------------------------------------------------------------------
    # 12.1  CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=(
            "HOOMD-blue v4 convex-polyhedron HPMC NVT equilibration.\n"
            "Usage: python convex_polyhedron_NVT.p --simulparam_file simulparam.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--simulparam_file",
        required=True,
        metavar="FILE",
        help="Path to the JSON file containing simulation parameters.",
    )
    args = parser.parse_args()
    _active_json_path = args.simulparam_file

    # ------------------------------------------------------------------
    # 12.2  HOOMD version banner  (preserved from original script)
    # ------------------------------------------------------------------
    print("*********************************************************")
    print("HOOMD-blue version:  ", hoomd.version.version)
    print("*********************************************************")
    print("hoomd.version.mpi_enabled:   ", hoomd.version.mpi_enabled)

    _print_banner("HOOMD-blue v4 | Convex-Polyhedron HPMC NVT Equilibration")

    # ------------------------------------------------------------------
    # 12.3  Load parameters
    # ------------------------------------------------------------------
    params = load_simulparams(args.simulparam_file)
    root_print(f"[INFO] Loaded parameters from '{args.simulparam_file}'")
    root_print(f"  tag                    = {params.tag}")
    root_print(f"  stage_id_current       = {params.stage_id_current}")
    root_print(f"  shape_json_filename    = {params.shape_json_filename}")
    root_print(f"  shape_scale            = {params.shape_scale}")
    root_print(f"  total_num_timesteps    = {params.total_num_timesteps}")
    root_print(f"  move_size_translation  = {params.move_size_translation}")
    root_print(f"  move_size_rotation     = {params.move_size_rotation}")
    root_print(
        f"  diagnostics_frequency  = "
        f"{params.diagnostics_frequency if params.diagnostics_frequency > 0 else params.log_frequency}"
    )

    # ------------------------------------------------------------------
    # 12.4  MPI communicator 
    # ------------------------------------------------------------------
    # Use mpi4py if available; fall back to the serial stub otherwise.
    comm = MPI.COMM_WORLD

    # Determine the MPI rank via environment variable before device init
    # (device.communicator.rank is only available after the device is created).
    _env_rank = _mpi_rank_from_env()

    # ------------------------------------------------------------------
    # 12.5  Filename resolution 
    # ------------------------------------------------------------------
    files = resolve_filenames(params)
    root_print(f"[INFO] Input GSD    : {files.input_gsd}")
    root_print(f"[INFO] Traj GSD     : {files.traj_gsd}")
    root_print(f"[INFO] Restart GSD  : {files.restart_gsd}")
    root_print(f"[INFO] Final GSD    : {files.final_gsd}")
    root_print(f"[INFO] Log file     : {files.log_txt}")
    root_print(f"[INFO] Diag HDF5    : {files.diag_hdf5}")

    # ------------------------------------------------------------------
    # 12.6  Seed management 
    # ------------------------------------------------------------------
    ensure_seed_file(params.stage_id_current, _env_rank)

    # Simple file-based barrier: poll until the seed file is visible
    seed_file = (
        _SEED_FILE_SINGLE if params.stage_id_current == -1 else _SEED_FILE_MULTI
    )
    _wait_start = time.perf_counter()
    while not Path(seed_file).exists():
        if time.perf_counter() - _wait_start > 30:
            sys.exit("[FATAL] Timeout waiting for seed file.")
        time.sleep(0.1)

    seed = read_seed(params.stage_id_current)
    root_print(f"[INFO] Random seed: {seed}")

    # ------------------------------------------------------------------
    # 12.7  Build simulation
    # ------------------------------------------------------------------
    sim        = None
    mc         = None
    log_handle = None

    try:
        sim, mc, log_handle = build_simulation(params, files, seed, comm, _env_rank)

        N = sim.state.N_particles

        # ------------------------------------------------------------------
        # 12.8  Run the simulation  (preserved from original script)
        # ------------------------------------------------------------------
        # **** Parallel Section ****
        # MC moves and I/O are both distributed across MPI processes.
        root_print(
            f"\n[INFO] Starting NVT run | "
            f"{params.total_num_timesteps} steps | "
            f"traj every {params.traj_gsd_frequency} | "
            f"restart every {params.restart_gsd_frequency}"
        )
        root_flush_stdout()

        sim.run(params.total_num_timesteps)

        root_print(
            f"[INFO] Run complete at timestep {sim.timestep} | "
            f"overlaps = {mc.overlaps}"
        )

        # ------------------------------------------------------------------
        # 12.9  Write final outputs 
        # ------------------------------------------------------------------
        write_final_outputs(
            sim, mc, params, files, start_time, seed, args.simulparam_file
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
        if sim is not None and mc is not None:
            try:
                emergency_file = f"emergency_restart_{params.tag}.gsd"
                type_shapes = _build_type_shapes_metadata(mc)
                _write_snapshot_with_shape(sim, emergency_file, type_shapes)
                root_print(f"[ERROR] Emergency snapshot written => {emergency_file}")
            except Exception:
                root_print("[ERROR] Could not write emergency snapshot.")
        raise   # Re-raise so the full traceback is visible

    finally:
        # Flush any buffered HOOMD writers so recent frames are on disk.
        if sim is not None:
            try:
                for writer in sim.operations.writers:
                    if hasattr(writer, "flush"):
                        writer.flush()
            except Exception:
                pass

        if log_handle is not None:
            try:
                log_handle.flush()
                log_handle.close()
            except Exception:
                pass  

        # ** Serial execution ** Only rank 0 prints total runtime
        if _is_root_rank():
            end_time = time.time()
            runtime  = end_time - start_time
            print(f"\nTotal runtime: {runtime:.2f} seconds")


# ===========================================================================
#  SCRIPT EXECUTION
# ===========================================================================

if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"[FATAL] {e}", file=sys.stderr, flush=True)
        raise
    except Exception:
        # Print the full traceback so the developer can locate and fix the bug.
        import traceback
        root_print("\n" + "="*80)
        root_print("[FATAL] Unexpected exception (bug in code):")
        root_print("="*80)
        traceback.print_exc()
        root_print("="*80)
        sys.exit(1)

