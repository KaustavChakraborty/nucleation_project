#!/usr/bin/env python3
# =============================================================================
# HOOMD-blue v4  |  Hard-Sphere HPMC NVT Equilibration  —  hs_nvt_v4.py
# =============================================================================
#
#
# ALGORITHM
# -------------------------------------------------------
# 1. Read parameters from a JSON file passed via --simulparam_file.
# 2. Load the simulation state from a GSD file (restart-aware: if a restart
#    GSD exists and no final GSD exists, resume from the checkpoint).
# 3. Configure the HPMC Sphere integrator with a fixed move size d.
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
# - HPMC Sphere integrator with fixed d (no MoveSize tuner)
# - nselect=1 (one trial move per particle per sweep)
# - All three writers: Table log, GSD trajectory, GSD restart
#
# USAGE
# -----
#   Single CPU / GPU:
#     python hs_nvt_v3.py --simulparam_file simulparam_hs_nvt.json
#
#   MPI (e.g. via mpirun / srun):
#     mpirun -n 8 python hs_nvt_v3.py --simulparam_file simulparam_hs_nvt.json
#
# JSON SCHEMA  (see simulparam_hs_nvt.json for annotated example)
# ---------------------------------------------------------------
# {
#   "tag"                    : "hs_4096_nvt",
#   "input_gsd_filename"     : "initial.gsd",
#   "stage_id_current"       : -1,     // -1 = single-stage; 0,1,... = multi-stage
#   "initial_timestep"       : 0,
#   "total_num_timesteps"    : 1000000,
#   "move_size_translation"  : 0.08,   // fixed d for type A
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

# ---------------------------------------------------------------------------
# Physical constant
# ---------------------------------------------------------------------------
_PI_OVER_6 = math.pi / 6.0


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

    _export_dict = {"acceptance_rate": ("scalar", True)}

    def __init__(self, integrator: hoomd.hpmc.integrate.Sphere) -> None:
        self.integrator = integrator
        # State from the previous logging call — used to compute the delta
        self.prev_accepted: Optional[int] = None
        self.prev_total:    Optional[int] = None

    @property
    def acceptance_rate(self) -> float:
        """
        Windowed translational acceptance ratio since the last query.

        Returns 0.0 before the first sim.run() call.
        """
        try:
            current_accepted = self.integrator.translate_moves[0]
            current_total    = sum(self.integrator.translate_moves)

            # First call: no previous state; return cumulative rate
            if self.prev_accepted is None or self.prev_total is None:
                self.prev_accepted = current_accepted
                self.prev_total    = current_total
                return (
                    current_accepted / current_total
                    if current_total != 0 else 0.0
                )

            delta_accepted = current_accepted - self.prev_accepted
            delta_total    = current_total    - self.prev_total
            self.prev_accepted = current_accepted
            self.prev_total    = current_total
            return delta_accepted / delta_total if delta_total != 0 else 0.0

        except (IndexError, ZeroDivisionError, DataAccessError):
            # IndexError        : translate_moves tuple is empty (not yet run)
            # ZeroDivisionError : cumulative total is zero on the very first call
            # DataAccessError   : HPMC counters not yet initialised before
            #                     the first sim.run() call.  HOOMD's Logger
            #                     validates custom loggables by calling property
            #                     getters at registration time, so this guard is
            #                     essential — without it the script crashes before
            #                     the simulation even starts.
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

    def __init__(
        self,
        simulation: hoomd.Simulation,
        mc: hoomd.hpmc.integrate.Sphere,
        N: int,
        diameter: float,
    ) -> None:
        self.simulation = simulation
        self.mc         = mc
        self._N         = N
        self._diameter  = diameter

    @property
    def volume(self) -> float:
        """Volume of the simulation box V [length^3]."""
        return float(self.simulation.state.box.volume)

    @property
    def packing_fraction(self) -> float:
        """
        Hard-sphere packing fraction phi = N * (pi/6) * sigma^3 / V.


        """
        V = self.simulation.state.box.volume
        return self._N * _PI_OVER_6 * self._diameter**3 / V


class OverlapCount:
    """
    Exposes the current HPMC overlap count as a loggable scalar.
    """

    _export_dict = {"overlap_count": ("scalar", True)}

    def __init__(self, mc: hoomd.hpmc.integrate.Sphere) -> None:
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

    # --- Run length ---
    total_num_timesteps:  int    # total HPMC sweeps for this stage

    # --- HPMC move ---
    move_size_translation: float # fixed translational move size d for type A

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

    # Resolved at runtime from the loaded GSD (not in JSON)
    diameter: Optional[float] = None

    def validate(self) -> None:
        """
        Run physics and logic sanity checks.
        Raises ValueError with a clear message on the first failed check.
        All errors are collected into a list so the user sees every problem
        in a single error message rather than having to fix one at a time.
        Called by load_simulparams() immediately after dataclass construction.
        """
        errors: list[str] = []

        if self.total_num_timesteps <= 0:
            errors.append(
                f"total_num_timesteps must be > 0, got {self.total_num_timesteps}"
            )
        if self.move_size_translation <= 0.0:
            errors.append(
                f"move_size_translation must be > 0, got {self.move_size_translation}"
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
                + "\n".join(f"  • {e}" for e in errors)
            )


# ===========================================================================
#  SECTION 4 — JSON loading  [N-01, N-14]
# ===========================================================================

# Required JSON keys with expected Python types
_REQUIRED_KEYS: dict[str, type] = {
    "tag":                    str,
    "input_gsd_filename":     str,
    "stage_id_current":       int,
    "total_num_timesteps":    int,
    "move_size_translation":  (int, float),
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
    Keys starting with '_' are stripped (they are comment-keys, as used
    in simulparam_hs_compression_v7.json).
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
            + "\n".join(f"  • {k}" for k in missing)
        )

    # Type-check required keys
    type_errors: list[str] = []
    for key, expected in _REQUIRED_KEYS.items():
        if key in raw and not isinstance(raw[key], expected):
            type_errors.append(
                f"  • '{key}': expected {expected}, "
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

    # Coerce numeric JSON fields that may arrive as int
    kw["move_size_translation"] = float(kw["move_size_translation"])

    params = SimulationParams(**kw)
    try:
        params.validate()
    except ValueError as exc:
        sys.exit(f"[FATAL] {exc}")

    return params


# ===========================================================================
#  SECTION 5 — Random seed management  [N-03]
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
#  SECTION 6 — Filename resolution  [N-02]
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

    [N-02] Stage-aware naming — mirrors hs_compress_v7.py:
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
#  SECTION 7 — GSD diameter reader
# ===========================================================================

def read_mono_diameter_from_gsd(gsd_filename: str) -> float:
    """
    Read the monodisperse particle diameter from the last frame of a GSD file.

    Exits cleanly if:
      - The file does not exist.
      - The file has no frames.
      - The diameters are missing, non-positive, or polydisperse.

    Returns
    -------
    float
        The common particle diameter σ.
    """
    path = Path(gsd_filename)
    # Check existence before calling gsd.hoomd.open() to give a cleaner
    # error message.  gsd.hoomd.open() on a missing file would raise a
    # FileNotFoundError buried inside the GSD C extension.
    if not path.exists():
        sys.exit(f"[FATAL] Cannot read diameter; GSD not found: '{gsd_filename}'")
    try:
        with gsd.hoomd.open(name=str(path), mode="r") as traj:
            # len(traj) == 0 means the file exists but was never written to
            # (e.g. a previous run crashed before the first frame was saved).
            if len(traj) == 0:
                sys.exit(f"[FATAL] GSD file has no frames: '{gsd_filename}'")
            # Read from the last frame (index -1) to get the most recent
            # particle state; this is consistent with how HOOMD restarts work.
            frame = traj[-1]
    except Exception as exc:
        # Catches gsd.hoomd.GSDError (corrupt file), PermissionError,
        # OSError, and any other I/O failure from the GSD C extension.
        sys.exit(f"[FATAL] Cannot open '{gsd_filename}': {exc}")

    diameters = list(frame.particles.diameter)
    # An empty diameter array means the GSD was written without diameter
    # data (e.g. frame.particles.diameter was never set by the writer).
    if not diameters:
        sys.exit(f"[FATAL] No particle diameters in '{gsd_filename}'.")
    # Non-positive diameters would cause HPMC to accept overlapping configs
    # silently — catch this physical impossibility early.
    if any(d <= 0.0 for d in diameters):
        sys.exit(f"[FATAL] Non-positive particle diameter in '{gsd_filename}'.")
    d0 = diameters[0]
    # All diameters must be identical (monodisperse).  The 1e-12 tolerance
    # accounts for float32→float64 round-trip in the GSD file format.
    if any(abs(d - d0) > 1e-12 for d in diameters[1:]):
        sys.exit(
            f"[FATAL] Multiple particle diameters in '{gsd_filename}'. "
            "This script requires a monodisperse system."
        )
    return float(d0)


# ===========================================================================
#  SECTION 8 — Snapshot broadcast 
# ===========================================================================

def load_and_broadcast_snapshot(
    input_gsd: str,
    comm,          # MPI.COMM_WORLD or stub
    rank: int,
) -> dict:
    """
    Rank-0 reads the last frame of *input_gsd* and broadcasts the minimal
    snapshot data to all MPI ranks.

    This preserves the original script's pattern exactly:
      - Only rank 0 performs file I/O (avoids concurrent file-open issues).
      - Data is packed into a plain dict and broadcast via MPI.
      - All ranks reconstruct identical initial conditions.

    Parameters
    ----------
    input_gsd : str
        Path to the GSD file to read.
    comm      : MPI communicator (or stub for serial runs).
    rank      : MPI rank of this process.

    Returns
    -------
    dict with keys: box, positions, diameters, typeid, types, N
    """
    if rank == 0:
        # -----------------------------------------------------------------
        # ** Serial I/O section ** Only rank 0 reads the file
        # -----------------------------------------------------------------
        if not Path(input_gsd).exists():
            sys.exit(f"[FATAL] Input GSD not found: '{input_gsd}'")
        # Only rank 0 performs file I/O.  This prevents all MPI ranks from
        # simultaneously opening the same GSD file, which can cause
        # corruption or I/O contention on parallel filesystems (Lustre, GPFS).
        try:
            with gsd.hoomd.open(input_gsd, mode="r") as f:
                # f[-1] reads the last frame — the most recent configuration.
                # For a fresh-run input this is the only (or final) frame;
                # for a restart GSD it is the single checkpoint frame.
                frame     = f[-1]
                box_data  = frame.configuration.box      # [Lx, Ly, Lz, xy, xz, yz]
                # Cast to float64 / int32 explicitly: GSD stores float32 and
                # int16/int32; HOOMD's Snapshot requires float64 and int32.
                positions = np.array(frame.particles.position,  dtype=np.float64)
                diameters = np.array(frame.particles.diameter,  dtype=np.float64)
                typeid    = np.array(frame.particles.typeid,    dtype=np.int32)
                types     = list(frame.particles.types)
                N         = int(len(positions))
        except Exception as exc:
            # Catches GSDError (corrupt/truncated file), PermissionError,
            # OSError, and any numpy conversion failure.
            sys.exit(f"[FATAL] Cannot read '{input_gsd}': {exc}")

        snap_data = {
            "box":       box_data,
            "positions": positions,
            "diameters": diameters,
            "typeid":    typeid,
            "types":     types,
            "N":         N,
        }
    else:
        snap_data = None

    # -----------------------------------------------------------------
    # ** MPI broadcast ** All ranks receive identical data
    # -----------------------------------------------------------------
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
    if rank == 0:
        snapshot = hoomd.Snapshot()
        snapshot.configuration.box = list(snap_data["box"])
        snapshot.particles.N = snap_data["N"]
        snapshot.particles.types = snap_data["types"]
        snapshot.particles.position[:] = np.asarray(snap_data["positions"], dtype=np.float64)
        snapshot.particles.diameter[:] = np.asarray(snap_data["diameters"], dtype=np.float64)
        # HOOMD v4 Snapshot stores typeid as uint32 internally.  The cast
        # from int32 (read above) to uint32 here is safe because type IDs
        # are always non-negative; all values in [0, N_types-1].
        snapshot.particles.typeid[:] = np.asarray(snap_data["typeid"], dtype=np.uint32)
    else:
        snapshot = hoomd.Snapshot()
    return snapshot


# ===========================================================================
#  SECTION 9 — Simulation builder
# ===========================================================================

def build_simulation(
    params: SimulationParams,
    files:  RunFiles,
    seed:   int,
    comm,           # MPI.COMM_WORLD or stub
    rank:   int,
) -> tuple[hoomd.Simulation, hoomd.hpmc.integrate.Sphere, object]:
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
    4.  Configure HPMC Sphere integrator with fixed move size d.
    5.  Attach Table log writer (human-readable .log file).
    6.  Attach GSD trajectory writer (appending).
    7.  Attach GSD restart writer (truncating checkpoint).

    Returns
    -------
    sim          : hoomd.Simulation
    mc           : HPMC Sphere integrator
    log_file_hdl : open file handle for the Table log (must be closed later)
    """

    # ------------------------------------------------------------------
    # 9.1  Device selection  [N-13]
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
    sim = hoomd.Simulation(device=device, seed=seed)

    # ------------------------------------------------------------------
    # 9.3  State initialisation  [N-04]  (restart-aware)
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
        sim.create_state_from_gsd(filename=files.restart_gsd)
        root_print(f"[INFO] Restart run: True")

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

        # [N-20] Fresh-run state init uses sim.create_state_from_gsd() directly
        sim.create_state_from_gsd(filename=files.input_gsd)
        root_print(f"[INFO] Restart run: False")

    # ------------------------------------------------------------------
    # 9.4  Read diameter from GSD and populate params.diameter
    # ------------------------------------------------------------------
    state_source = files.restart_gsd if (restart_exists and not final_exists) \
                   else files.input_gsd
    params.diameter = float(read_mono_diameter_from_gsd(state_source))
    root_print(
        f"[INFO] Diameter read from GSD '{state_source}' = {params.diameter}"
    )

    N = sim.state.N_particles
    # A zero-particle system would cause division-by-zero in packing fraction
    # calculations and silently produce incorrect HPMC statistics.  Exit now
    # with a clear message rather than letting the run produce bad data.
    if N == 0:
        sys.exit("[FATAL] Loaded GSD contains zero particles.")

    phi = N * _PI_OVER_6 * params.diameter**3 / sim.state.box.volume
    root_print(
        f"[INFO] N={N} | sphere_diameter={params.diameter} | "
        f"phi={phi:.6f} | box={sim.state.box}"
    )

    # Convenience: log box dimensions (mirrors original script)
    box = sim.state.box
    root_print(
        f"[INFO] Box: Lx={box.Lx:.4f}  Ly={box.Ly:.4f}  Lz={box.Lz:.4f}  "
        f"xy={box.xy}  xz={box.xz}  yz={box.yz}"
    )

    # ------------------------------------------------------------------
    # 9.5  HPMC Sphere integrator  (preserved exactly)
    # ------------------------------------------------------------------
    mc = hoomd.hpmc.integrate.Sphere(
        default_d=params.move_size_translation,
        nselect=1,
    )

    # Assign the hard-sphere shape to particle type "A".
    # diameter here is sigma (the GSD particle diameter field)
    mc.shape["A"] = {"diameter": params.diameter}
    # Assign the integrator to the simulation before any writers are added;
    # writers and computes are attached after the integrator.
    sim.operations.integrator = mc

    root_print(
        f"[INFO] HPMC Sphere configured | "
        f"particle_diameter(sigma) = {params.diameter} | "
        f"translation_move_size(d) = {params.move_size_translation}"
    )

    # **** Parallel Section ****
    # MC moves are distributed across MPI processes; each handles a domain.

    # ------------------------------------------------------------------
    # 9.6  Custom loggable instances
    # ------------------------------------------------------------------
    # All properties guarded against DataAccessError [N-06]
    status        = Status(sim)
    mc_status     = MCStatus(mc)
    box_prop      = Box_property(sim, mc, N, params.diameter)
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
    logger[("MCStatus",   "acceptance_rate")]   = (mc_status,     "acceptance_rate",   "scalar")
    logger[("Box",        "volume")]            = (box_prop,      "volume",            "scalar")
    logger[("Box",        "packing_fraction")]  = (box_prop,      "packing_fraction",  "scalar")   # [N-07]
    logger[("HPMC",       "overlap_count")]     = (overlap_count, "overlap_count",     "scalar")   # [N-07]

    # Open the log file; the handle is returned so main() can close it
    # properly in a finally block.  [N-12]
    log_file_hdl = open(files.log_txt, "w")   
    log_writer = hoomd.write.Table(
        output=log_file_hdl,
        trigger=hoomd.trigger.Periodic(params.log_frequency),
        logger=logger,
    )
    # **** Parallel Section ****
    sim.operations.writers.append(log_writer)

    # ------------------------------------------------------------------
    # 9.8  GSD trajectory writer  [N-08, N-16, N-19]
    # ------------------------------------------------------------------
    # [N-19] Trajectory file mode is determined by run type:

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
    )
    traj_gsd_writer.write_diameter = True     # [N-16] belt-and-suspenders
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
    )
    restart_gsd_writer.write_diameter = True  # [N-16] belt-and-suspenders
    sim.operations.writers.append(restart_gsd_writer)

    # ------------------------------------------------------------------
    # 9.10  Optional machine-readable diagnostics writer  [N-18, N-19]
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
        hdf5_logger[("MCStatus", "acceptance_rate")] = (mc_status, "acceptance_rate", "scalar")
        hdf5_logger[("Box", "packing_fraction")] = (box_prop, "packing_fraction", "scalar")
        hdf5_logger[("HPMC", "overlap_count")] = (overlap_count, "overlap_count", "scalar")
        # translate_moves: tuple (accepted, rejected) — total move statistics.
        # mps: Monte Carlo moves per second — useful for performance monitoring.
        hdf5_logger.add(mc, quantities=["translate_moves", "mps"])

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
#  SECTION 10 — Final output writing  [N-10, N-11]
# ===========================================================================

def _write_snapshot(sim: hoomd.Simulation, filename: str) -> None:
    """
    Write the current simulation state as a single-frame GSD file.
    """
    snapshot = sim.state.get_snapshot()
    if not _is_root_rank():
        return

    frame = gsd.hoomd.Frame()
    box = snapshot.configuration.box
    frame.configuration.box = list(box)

    N = int(snapshot.particles.N)
    frame.particles.N = N
    frame.particles.types = list(snapshot.particles.types)
    frame.particles.typeid = np.array(snapshot.particles.typeid, dtype=np.uint32)
    frame.particles.position = np.array(snapshot.particles.position, dtype=np.float32)
    frame.particles.diameter = np.array(snapshot.particles.diameter, dtype=np.float32)

    try:
        # Particle image flags record how many box lengths each particle
        # has crossed (used for computing mean-squared displacement and
        # unwrapped trajectories).
        frame.particles.image = np.array(snapshot.particles.image, dtype=np.int32)
    except Exception:
        # AttributeError : snapshot.particles.image not available
        # ValueError     : image array has the wrong shape
        # TypeError      : cannot convert to int32
        pass

    with gsd.hoomd.open(filename, mode="w") as traj:
        traj.append(frame)


def write_final_outputs(
    sim:         hoomd.Simulation,
    mc:          hoomd.hpmc.integrate.Sphere,
    params:      SimulationParams,
    files:       RunFiles,
    start_time:  float,
    seed:        int,
    json_path:   str,
) -> None:
    """
    Write the final GSD snapshot, a machine-readable summary JSON, and print
    the console summary banner.

    [N-10] Final GSD + summary JSON.
    [N-11] Console banner mirrors hs_compress_v7.py format.
    """
    N   = sim.state.N_particles
    phi = N * _PI_OVER_6 * params.diameter**3 / sim.state.box.volume

    # --- Final GSD snapshot -----------------------------------------------
    _write_snapshot(sim, files.final_gsd)
    root_print(f"[OUTPUT] Final GSD        => {files.final_gsd}")

    # --- Machine-readable summary JSON ------------------------------------
    runtime = time.time() - start_time
    summary = {
        "tag":              params.tag,
        "stage_id":         params.stage_id_current,
        "simulparam_file":  json_path,
        "input_gsd":        files.input_gsd,
        "traj_gsd":         files.traj_gsd,
        "diagnostics_hdf5": files.diag_hdf5,
        "final_gsd":        files.final_gsd,
        "n_particles":      N,
        "diameter":         params.diameter,
        "packing_fraction": round(phi, 8),
        "move_size_d":      params.move_size_translation,
        "diagnostics_frequency": (params.diagnostics_frequency if params.diagnostics_frequency > 0 else params.log_frequency),
        "total_timesteps":  params.total_num_timesteps,
        "final_timestep":   sim.timestep,
        "overlaps_final":   mc.overlaps,
        "random_seed":      seed,
        "box_final": {
            "Lx": sim.state.box.Lx,
            "Ly": sim.state.box.Ly,
            "Lz": sim.state.box.Lz,
        },
        "runtime_seconds":  round(runtime, 2),
        "created":          time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary_file = f"{params.tag}_stage{params.stage_id_current}_nvt_summary.json"
    # Only rank 0 writes the summary JSON to avoid N identical files being
    # written simultaneously to the same path, which would corrupt the output.
    if _is_root_rank():
        with open(summary_file, "w") as fh:
            json.dump(summary, fh, indent=2)
        root_print(f"[OUTPUT] Summary JSON     => {summary_file}")

    # --- Console summary banner -------------------------------------------
    _print_banner("HOOMD-blue v4 | Hard-Sphere NVT Equilibration | Run Complete")
    lines = [
        f"  Simulparam file       : {json_path}",
        f"  Tag                   : {params.tag}",
        f"  Stage id              : {params.stage_id_current}",
        f"  Input GSD             : {files.input_gsd}",
        f"  Final GSD             : {files.final_gsd}",
        f"  Diagnostics HDF5      : {files.diag_hdf5}",
        f"  Particles (N)         : {N}",
        f"  Diameter (sigma)          : {params.diameter}",
        f"  Packing fraction (phi)  : {phi:.6f}",
        f"  Move size d           : {params.move_size_translation}",
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
            "HOOMD-blue v4 hard-sphere HPMC NVT equilibration.\n"
            "Usage: python hs_nvt_v4.py --simulparam_file simulparam.json"
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

    _print_banner("HOOMD-blue v4 | Hard-Sphere HPMC NVT Equilibration (hs_nvt_v3)")

    # ------------------------------------------------------------------
    # 12.3  Load parameters
    # ------------------------------------------------------------------
    params = load_simulparams(args.simulparam_file)
    root_print(f"[INFO] Loaded parameters from '{args.simulparam_file}'")
    root_print(f"  tag                    = {params.tag}")
    root_print(f"  stage_id_current       = {params.stage_id_current}")
    root_print(f"  total_num_timesteps    = {params.total_num_timesteps}")
    root_print(f"  move_size_translation  = {params.move_size_translation}")
    root_print(f"  diagnostics_frequency  = {params.diagnostics_frequency if params.diagnostics_frequency > 0 else params.log_frequency}")

    # ------------------------------------------------------------------
    # 12.4  MPI communicator  [N-05]
    # ------------------------------------------------------------------
    # Use mpi4py if available; fall back to the serial stub otherwise.
    comm = MPI.COMM_WORLD

    # Determine the MPI rank via environment variable before device init
    # (device.communicator.rank is only available after the device is created).
    _env_rank = _mpi_rank_from_env()

    # ------------------------------------------------------------------
    # 12.5  Filename resolution  [N-02]
    # ------------------------------------------------------------------
    files = resolve_filenames(params)
    root_print(f"[INFO] Input GSD    : {files.input_gsd}")
    root_print(f"[INFO] Traj GSD     : {files.traj_gsd}")
    root_print(f"[INFO] Restart GSD  : {files.restart_gsd}")
    root_print(f"[INFO] Final GSD    : {files.final_gsd}")
    root_print(f"[INFO] Log file     : {files.log_txt}")
    root_print(f"[INFO] Diag HDF5    : {files.diag_hdf5}")

    # ------------------------------------------------------------------
    # 12.6  Seed management  [N-03]
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
        # 12.9  Write final outputs  [N-10, N-11]
        # ------------------------------------------------------------------
        write_final_outputs(
            sim, mc, params, files, start_time, seed, args.simulparam_file
        )

    except Exception as exc:
        # ------------------------------------------------------------------
        # [N-09] Emergency snapshot on any unexpected exception
        # ------------------------------------------------------------------
        root_print(
            f"\n[ERROR] Unexpected exception at timestep "
            f"{sim.timestep if sim else '?'}:\n"
            f"  {type(exc).__name__}: {exc}"
        )
        if sim is not None:
            try:
                emergency_file = f"emergency_restart_{params.tag}.gsd"
                _write_snapshot(sim, emergency_file)
                root_print(f"[ERROR] Emergency snapshot written → {emergency_file}")
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
                # Flushing is best-effort; do not mask the original exception.
                pass

        # ------------------------------------------------------------------
        # [N-12] Always close the log file handle, even after an exception
        # ------------------------------------------------------------------
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
    except SystemExit:
        pass
    except Exception:
        # Print the full traceback so the developer can locate and fix the bug.
        import traceback
        root_print("\n" + "="*80)
        root_print("[FATAL] Unexpected exception (bug in code):")
        root_print("="*80)
        traceback.print_exc()
        root_print("="*80)
        sys.exit(1)
