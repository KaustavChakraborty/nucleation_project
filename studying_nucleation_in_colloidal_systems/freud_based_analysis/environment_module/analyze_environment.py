"""
=============================================================================
freud_env_analysis/analyze_environment.py
=============================================================================
Production-grade analysis pipeline for ALL calculations provided by
freud.environment, driven entirely by a JSON parameter file.

Analyses implemented (freud.environment module only):
  1. BondOrder                    – Bond Orientational Order Diagram (BOOD)
  2. AngularSeparationNeighbor    – Per-bond angular separation between neighbours
  3. AngularSeparationGlobal      – Per-particle angular sep vs. global references
  4. LocalDescriptors             – Per-bond spherical harmonic descriptors
  5. LocalBondProjection          – Bond projection onto reference axes
  6. EnvironmentCluster           – Cluster particles by matching local environments
  7. EnvironmentMotifMatch        – Match environments against a reference motif

Frame-averaging rationale
--------------------------
  FRAME-AVERAGED (accumulate with reset=False or pool over chosen frames):
    BondOrder            – histogram smooths with more frames
    AngularSeparationNeighbor – distribution stabilises with statistics
    AngularSeparationGlobal   – distribution stabilises with statistics
    LocalDescriptors     – mean SPH power spectrum stabilises
    LocalBondProjection  – projection distribution stabilises

  LAST-FRAME ONLY (spatial snapshot – averaging is physically meaningless):
    EnvironmentCluster   – cluster map is a single-frame snapshot
    EnvironmentMotifMatch – match map is a single-frame snapshot

  The results summary table printed at the end explicitly states which
  mode was used for each analysis.

Output layout
--------------
  outputs/
    01_bond_order_diagram_2D.png
    02_angular_sep_neighbor.png
    03_angular_sep_global.png
    04_local_descriptors.png
    05_local_bond_projection.png
    06_environment_cluster.png
    07_environment_motif_match.png
    summary.json
  logs/
    run_<timestamp>.log

Usage
------
  python analyze_environment.py params.json

Author: production-grade pipeline – see README.md for full documentation
=============================================================================
"""

from __future__ import annotations

# ── stdlib ──────────────────────────────────────────────────────────────────
import json
import logging
import os
import sys
import time
import traceback
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── third-party ──────────────────────────────────────────────────────────────
import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection

# ── freud / gsd ───────────────────────────────────────────────────────────────
try:
    import freud
    import gsd.hoomd
except ImportError as _exc:
    print(
        f"\n[FATAL] Required package missing: {_exc}\n"
        "  Install via:  conda install -c conda-forge freud gsd\n"
        "            or: pip install freud-analysis gsd\n"
    )
    sys.exit(1)

# ── rowan (optional – needed for angular separation analyses) ─────────────────
try:
    import rowan
    HAS_ROWAN = True
except ImportError:
    HAS_ROWAN = False
    warnings.warn(
        "Package 'rowan' not found.\n"
        "  AngularSeparationNeighbor and AngularSeparationGlobal will be skipped.\n"
        "  Install: pip install rowan",
        stacklevel=2,
    )


# =============================================================================
# ①  LOGGING
# =============================================================================

def _setup_logging(log_dir: Path) -> logging.Logger:
    """
    Configure a logger that writes simultaneously to a timestamped file
    and to stdout with colour-coded severity.

    Parameters
    ----------
    log_dir : Path   Directory where the log file is written.

    Returns
    -------
    logging.Logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{stamp}.log"

    fmt     = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%H:%M:%S"

    # Remove any pre-existing handlers to avoid duplicated output when
    # the module is imported more than once in a session.
    root = logging.getLogger()
    root.handlers.clear()

    logging.basicConfig(
        level   = logging.INFO,
        format  = fmt,
        datefmt = datefmt,
        handlers = [
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("freud_env")
    logger.info("Log file: %s", log_file)
    return logger


# =============================================================================
# ②  CONFIGURATION  (defaults + user merge)
# =============================================================================

# Every key defined here documents the canonical schema.
# The user JSON only needs to supply keys they wish to override.
DEFAULT_CONFIG: Dict[str, Any] = {
    # ── trajectory ──────────────────────────────────────────────────────────
    "trajectory": "trajectory.gsd",
    "output_dir": "outputs",

    # ── frame selection ─────────────────────────────────────────────────────
    # frame_average : true  → accumulate/pool across all selected frames
    #                 false → only the very last selected frame is analysed
    "frame_average": True,
    "frame_start"  : 0,        # first frame index (0-based, inclusive)
    "frame_end"    : -1,       # last frame index (-1 = last in trajectory)
    "frame_step"   : 1,        # stride between consecutive frames

    # ── neighbour query ─────────────────────────────────────────────────────
    # use_num_neighbors : true  → {"num_neighbors": N} query
    #                     false → {"r_max": R} cutoff-based query
    "num_neighbors"    : 12,
    "r_max"            : 2.0,
    "use_num_neighbors": True,

    # ── figure aesthetics ───────────────────────────────────────────────────
    "dpi"        : 300,
    "colormap"   : "viridis",
    "figure_size": [8, 6],   # [width_in, height_in]

    # ════════════════════════════════════════════════════════════════════════
    # ANALYSIS BLOCKS  (set "enabled": false to skip entirely)
    # ════════════════════════════════════════════════════════════════════════

    # ── 1. BondOrder ────────────────────────────────────────────────────────
    "bond_order": {
        "enabled"    : True,
        "n_bins_theta": 100,    # azimuthal bins (0 → 2π)
        "n_bins_phi"  : 100,    # polar bins    (0 → π)
        # mode options:
        #   "bod"  – standard bond orientational order diagram
        #   "lbod" – local-frame BOD (needs orientation data in GSD)
        #   "obcd" – orientation-bond correlation diagram
        #   "oocd" – orientation-orientation correlation diagram
        "mode": "bod",
    },

    # ── 2. AngularSeparationNeighbor ────────────────────────────────────────
    "angular_separation_neighbor": {
        "enabled"          : True,
        # equiv_orientations: list of quaternions [w,x,y,z] encoding the
        # particle's rotational symmetry group.
        # [[1,0,0,0]] = fully asymmetric (no equivalent rotations)
        "equiv_orientations": [[1, 0, 0, 0]],
        "n_histogram_bins"  : 36,
    },

    # ── 3. AngularSeparationGlobal ──────────────────────────────────────────
    "angular_separation_global": {
        "enabled"           : True,
        # Set of reference orientations (quaternions) to compare against.
        "global_orientations": [[1, 0, 0, 0]],
        "equiv_orientations" : [[1, 0, 0, 0]],
        "n_histogram_bins"   : 36,
    },

    # ── 4. LocalDescriptors ─────────────────────────────────────────────────
    "local_descriptors": {
        "enabled"    : True,
        "l_max"      : 6,        # max angular momentum quantum number
        # mode options:
        #   "global"         – orient bonds in the simulation (lab) frame
        #   "local"          – orient in each bond's own frame
        #   "particle_local" – rotate by the particle's quaternion first
        "mode"       : "global",
        "negative_m" : True,     # include negative m coefficients
    },

    # ── 5. LocalBondProjection ──────────────────────────────────────────────
    "local_bond_projection": {
        "enabled"        : True,
        # projection_vecs: reference vectors (in particle local frame)
        # onto which each bond vector is projected. Default = Cartesian axes.
        "projection_vecs": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    },

    # ── 6. EnvironmentCluster   (last-frame only) ───────────────────────────
    "environment_cluster": {
        "enabled"     : True,
        # threshold: max vector difference magnitude for two bond vectors to
        # be considered "matching".  Typically 10–30 % of the first RDF peak.
        "threshold"   : 0.3,
        # registration: if true, perform brute-force RMSD minimisation
        # before comparison (slower but more tolerant to orientation drift)
        "registration": False,
    },

    # ── 7. EnvironmentMotifMatch (last-frame only) ───────────────────────────
    "environment_motif_match": {
        "enabled"     : True,
        "threshold"   : 0.3,
        "registration": False,
        # motif: list of [x,y,z] bond vectors defining the reference
        # environment.  null → auto-extract from particle 0's neighbourhood.
        "motif"       : None,
    },
}


def load_config(path: str | Path) -> Dict[str, Any]:
    """
    Read the user JSON file and deep-merge it with DEFAULT_CONFIG.

    Parameters
    ----------
    path : str or Path   Path to the user's .json parameter file.

    Returns
    -------
    dict   Fully resolved configuration.

    Raises
    ------
    FileNotFoundError   If the file does not exist.
    json.JSONDecodeError  If the file is malformed JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as fh:
        user = json.load(fh)
    return _deep_merge(DEFAULT_CONFIG, user)


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* on top of *base*.
    Sub-dicts are merged key-by-key; scalars and lists are replaced.
    Neither input is mutated.
    """
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# =============================================================================
# ③  GSD TRAJECTORY HELPERS
# =============================================================================

def open_trajectory(gsd_path: str | Path, logger: logging.Logger):
    """
    Open a GSD trajectory for reading.

    Parameters
    ----------
    gsd_path : str or Path
    logger   : logging.Logger

    Returns
    -------
    gsd.hoomd.HOOMDTrajectory
    """
    gsd_path = Path(gsd_path)
    if not gsd_path.exists():
        raise FileNotFoundError(f"GSD file not found: {gsd_path}")
    logger.info("Opening trajectory : %s", gsd_path.resolve())
    traj = gsd.hoomd.open(str(gsd_path), "r")
    logger.info("  Frames : %d   |   Particles (frame 0) : %d",
                len(traj), traj[0].particles.N)
    return traj


def resolve_frame_indices(
    traj,
    start: int,
    end: int,
    step: int,
    logger: logging.Logger,
) -> List[int]:
    """
    Convert user-supplied frame_start / frame_end / frame_step into a
    concrete list of integer frame indices, respecting trajectory length
    and negative indexing.

    Raises
    ------
    ValueError   If the selection produces an empty frame list.
    """
    n = len(traj)
    end_resolved = n + end + 1 if end < 0 else min(end, n)
    indices = list(range(start, end_resolved, step))
    if not indices:
        raise ValueError(
            f"Empty frame selection: start={start}, end={end}, step={step}, "
            f"trajectory length={n}."
        )
    logger.info("  Frames selected : %d … %d  (step=%d, total=%d)",
                indices[0], indices[-1], step, len(indices))
    return indices


def extract_frame_data(frame) -> Dict[str, Any]:
    """
    Extract the data needed for freud analyses from a single GSD frame.

    Returns
    -------
    dict with keys:
        'box'          – freud.box.Box
        'positions'    – (N, 3) float32 ndarray
        'orientations' – (N, 4) float32 ndarray  [w, x, y, z]  or None
        'N'            – int, number of particles
    """
    box       = freud.box.Box.from_box(frame.configuration.box)
    positions = np.asarray(frame.particles.position, dtype=np.float32)

    raw_orient = frame.particles.orientation
    # GSD stores [1,0,0,0] as the default/uninitialised quaternion.
    # Treat orientations as missing only if the array itself is None.
    orientations = (
        np.asarray(raw_orient, dtype=np.float32)
        if raw_orient is not None
        else None
    )

    return {
        "box"         : box,
        "positions"   : positions,
        "orientations": orientations,
        "N"           : len(positions),
    }


def build_nq_args(cfg: Dict[str, Any]) -> Dict:
    """
    Build a neighbour-query argument dict from the top-level config.

    Returns
    -------
    dict   Passed as `neighbors=` to freud compute methods.
    """
    if cfg["use_num_neighbors"]:
        return {"num_neighbors": cfg["num_neighbors"], "exclude_ii": True}
    return {"r_max": cfg["r_max"], "exclude_ii": True}


# =============================================================================
# ④  PLOT UTILITIES
# =============================================================================

RCPARAMS = {
    "font.family"     : "serif",
    "font.size"       : 11,
    "axes.titlesize"  : 13,
    "axes.labelsize"  : 12,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
    "legend.fontsize" : 10,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "figure.dpi"      : 150,   # screen; files saved at cfg["dpi"]
    "savefig.bbox"    : "tight",
    "savefig.pad_inches": 0.05,
}


def apply_style():
    """Apply publication-quality rcParams globally."""
    plt.rcParams.update(RCPARAMS)


def save_fig(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> Path:
    """
    Save *fig* as a PNG and close it.

    Parameters
    ----------
    fig        : matplotlib Figure
    output_dir : output directory (created if necessary)
    stem       : filename stem (no extension)
    dpi        : resolution

    Returns
    -------
    Path   Absolute path of the saved file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fpath = output_dir / f"{stem}.png"
    fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fpath


def add_colorbar(ax, mappable, label: str):
    """Attach a labelled colorbar to *ax*."""
    cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    return cbar


# =============================================================================
# 5  ANALYSIS FUNCTIONS
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 5.1  BondOrder
# ─────────────────────────────────────────────────────────────────────────────

def run_bond_order(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Compute the Bond Orientational Order Diagram (BOOD) using
    ``freud.environment.BondOrder``.

    Physical meaning:-
    A BOOD shows the angular distribution of bonds between a particle and
    its neighbours.  All bonds in the system are projected onto a unit
    sphere and histogrammed by azimuthal (θ) and polar (φ) angle.

    *  Isotropic (liquid) => uniform sphere
    *  FCC crystal        => discrete spots at 90 deg/60 deg patterns
    *  BCC crystal        => four-fold tetrahedral pattern

    Frame averaging
    ~~~~~~~~~~~~~~~
    ``BondOrder.compute(…, reset=False)`` accumulates statistics across
    frames.  This is always meaningful - more frames = smoother histogram.

    Outputs
    ~~~~~~~
    *  3-D interactive sphere (matplotlib window - rotate with mouse)
    *  2-D Mercator-projection heatmap saved to outputs/

    Returns
    -------
    dict with 'bond_order_max' and 'bond_order_mean'
    """

    # ---------------------------------------------------------------------
    # 1. Validate configuration early, with friendly messages
    # ---------------------------------------------------------------------
    try:
        bo_cfg = cfg["bond_order"]
    except KeyError as exc:
        log.error(
            "BondOrder configuration missing: cfg['bond_order'] not found. "
            "Please check your JSON input."
        )
        log.debug("Full traceback:\n%s", traceback.format_exc())
        return {}

    try:
        n_theta = int(bo_cfg["n_bins_theta"])
        n_phi   = int(bo_cfg["n_bins_phi"])
        mode    = bo_cfg.get("mode", "bod")
        nq      = build_nq_args(cfg)
    except KeyError as exc:
        log.error(
            "BondOrder configuration is incomplete. Missing key: %s. "
            "Expected at least: n_bins_theta, n_bins_phi, mode.",
            exc
        )
        log.debug("Full traceback:\n%s", traceback.format_exc())
        return {}
    except Exception as exc:
        log.error(
            "Failed while reading BondOrder configuration: %s: %s",
            type(exc).__name__, exc
        )
        log.debug("Full traceback:\n%s", traceback.format_exc())
        return {}

    if n_theta <= 0 or n_phi <= 0:
        log.error(
            "Invalid BondOrder bin counts: n_bins_theta=%s, n_bins_phi=%s. "
            "Both must be positive integers.",
            n_theta, n_phi
        )
        return {}

    if not frames:
        log.error("BondOrder was requested, but the frame list is empty.")
        return {}


    log.info(
        "── BondOrder  (requested mode=%s, theta-bins=%d, phi-bins=%d) ──────────",
        mode, n_theta, n_phi
    )
    log.info("  freud 3.1.0 compatibility: BondOrder.compute() will be called without 'mode'.")

    # ---------------------------------------------------------------------
    # 2. Initialize freud BondOrder object safely
    # ---------------------------------------------------------------------
    try:
        bod = freud.environment.BondOrder((n_theta, n_phi))
    except Exception as exc:
        log.error(
            "Could not initialize freud.environment.BondOrder with "
            "(n_theta=%d, n_phi=%d). Error: %s: %s",
            n_theta, n_phi, type(exc).__name__, exc
        )
        log.debug("Full traceback:\n%s", traceback.format_exc())
        return {}

    first = True
    n_success = 0
    failed_frames = []

    # ---------------------------------------------------------------------
    # 3. Frame-by-frame computation with richer diagnostics
    # ---------------------------------------------------------------------
    for fi in frames:
        try:
            fd = extract_frame_data(traj[fi])
        except Exception as exc:
            msg = (
                f"Could not extract frame data for frame {fi}. "
                f"Error: {type(exc).__name__}: {exc}"
            )
            log.warning(msg)
            log.debug("Full traceback for frame extraction failure:\n%s",
                      traceback.format_exc())
            failed_frames.append((fi, "extract_frame_data", str(exc)))
            continue

        try:
            system = (fd["box"], fd["positions"])
        except KeyError as exc:
            msg = (
                f"Frame {fi} data is incomplete. Missing key: {exc}. "
                f"Available keys: {list(fd.keys())}"
            )
            log.warning(msg)
            failed_frames.append((fi, "frame_data_incomplete", str(exc)))
            continue

        # Optional diagnostics
        n_particles = None
        box_repr = None
        try:
            n_particles = len(fd["positions"])
        except Exception:
            n_particles = "unknown"

        try:
            box_repr = fd["box"]
        except Exception:
            box_repr = "unavailable"

        try:
            bod.compute(system, neighbors=nq, reset=first)
            first = False
            n_success += 1
            log.info(
                "  BondOrder frame %d computed successfully "
                "(N=%s, reset=%s).",
                fi, n_particles, (n_success == 1)
            )

        except Exception as exc:
            log.warning(
                "  BondOrder failed for frame %d.\n"
                "    Error type : %s\n"
                "    Error      : %s\n"
                "    Particles  : %s\n"
                "    Box        : %s\n"
                "    Neighbors  : %s\n"
                "    Hint       : Check whether positions/box are valid, "
                "neighbor-query settings are appropriate, and the trajectory "
                "frame is not malformed.",
                fi,
                type(exc).__name__,
                exc,
                n_particles,
                box_repr,
                nq,
            )
            log.debug("Full traceback for frame %d:\n%s", fi, traceback.format_exc())
            failed_frames.append((fi, "compute", str(exc)))
            continue

    # ---------------------------------------------------------------------
    # 4. Final status after loop
    # ---------------------------------------------------------------------
    if first:
        log.error(
            "BondOrder could not be computed for any frame. "
            "Total frames attempted: %d. Failed frames: %d.",
            len(frames), len(failed_frames)
        )
        if failed_frames:
            log.error("Failure summary:")
            for fi, stage, err in failed_frames:
                log.error("  frame=%s  stage=%s  error=%s", fi, stage, err)

        log.error(
            "Suggested checks:\n"
            "  1. Verify the trajectory actually contains particle positions.\n"
            "  2. Check that the box information is valid for each frame.\n"
            "  3. Verify neighbor-query settings from build_nq_args(cfg).\n"
            "  4. Confirm the trajectory file is not corrupted.\n"
            "  5. Confirm freud version compatibility."
        )
        return {}

    log.info(
        "BondOrder summary: %d/%d frames succeeded, %d failed.",
        n_success, len(frames), len(failed_frames)
    )

    # ---------------------------------------------------------------------
    # 5. Visualization should not destroy already-computed results
    # ---------------------------------------------------------------------
    try:
        log.info("  Showing 3-D interactive Bond Order Diagram …")
        _show_bod_3d(bod, n_theta, n_phi, cfg["colormap"], log)
    except Exception as exc:
        log.warning(
            "3-D BondOrder visualization failed, but computed data is still valid.\n"
            "  Error type : %s\n"
            "  Error      : %s\n",
            type(exc).__name__,
            exc
        )
        log.debug("Full traceback for 3-D plotting failure:\n%s",
                  traceback.format_exc())

    try:
        fpath = _save_bod_2d(bod, n_theta, n_phi, cfg, out, log)
        log.info("  2-D BOOD saved => %s", fpath)
    except Exception as exc:
        log.error(
            "2-D BondOrder heatmap generation/saving failed.\n"
            "  Error type : %s\n"
            "  Error      : %s\n"
            "  Output dir  : %s\n",
            type(exc).__name__,
            exc,
            out
        )
        log.debug("Full traceback for 2-D save failure:\n%s",
                  traceback.format_exc())

    # ---------------------------------------------------------------------
    # 6. Robust summary-stat extraction
    # ---------------------------------------------------------------------
    try:
        arr = np.asarray(bod.bond_order)
        if arr.size == 0:
            log.warning(
                "BondOrder array is empty after successful computation. "
                "Returning an empty result dictionary."
            )
            return {}

        return {
            "bond_order_max": float(np.max(arr)),
            "bond_order_mean": float(np.mean(arr)),
        }

    except Exception as exc:
        log.error(
            "Failed to compute summary statistics from bod.bond_order.\n"
            "  Error type : %s\n"
            "  Error      : %s",
            type(exc).__name__,
            exc
        )
        log.debug("Full traceback for summary-stat failure:\n%s",
                  traceback.format_exc())
        return {}


def _show_bod_3d(bod, n_theta: int, n_phi: int, cmap_name: str, log: logging.Logger):
    """
    Render a 3-D interactive Bond Order Diagram as a coloured unit sphere.

    The surface colour encodes bond density: bright regions have many bonds
    pointing in that direction; dark regions have few.

    Notes
    -----
    This function is intended to be visualization-only.
    If interactive display fails, it logs a helpful message and returns,
    rather than crashing the whole BondOrder analysis.
    """

    # ------------------------------------------------------------------
    # 1. Basic validation of inputs
    # ------------------------------------------------------------------
    if n_theta <= 0 or n_phi <= 0:
        log.error(
            "Cannot render 3-D BOOD: invalid grid size "
            "(n_theta=%s, n_phi=%s). Both must be positive integers.",
            n_theta, n_phi
        )
        return

    # ------------------------------------------------------------------
    # 2. Build spherical angular grid
    # ------------------------------------------------------------------
    try:
        phi_v   = np.linspace(0, np.pi, n_phi)
        theta_v = np.linspace(0, 2 * np.pi, n_theta)
        phi_g, theta_g = np.meshgrid(phi_v, theta_v)
    except Exception as exc:
        log.error(
            "Failed to construct angular grid for 3-D BOOD.\n"
            "  Error type : %s\n"
            "  Error      : %s",
            type(exc).__name__,
            exc
        )
        log.debug(traceback.format_exc())
        return

    # ------------------------------------------------------------------
    # 3. Convert spherical grid to Cartesian sphere
    # ------------------------------------------------------------------
    try:
        X = np.sin(phi_g) * np.cos(theta_g)
        Y = np.sin(phi_g) * np.sin(theta_g)
        Z = np.cos(phi_g)
    except Exception as exc:
        log.error(
            "Failed to convert angular grid into Cartesian coordinates "
            "for 3-D BOOD.\n"
            "  Error type : %s\n"
            "  Error      : %s",
            type(exc).__name__,
            exc
        )
        log.debug(traceback.format_exc())
        return

    # ------------------------------------------------------------------
    # 4. Read and validate BondOrder array
    # ------------------------------------------------------------------
    try:
        arr = np.asarray(bod.bond_order)
    except Exception as exc:
        log.error(
            "Could not convert bod.bond_order into a NumPy array.\n"
            "  Error type : %s\n"
            "  Error      : %s",
            type(exc).__name__,
            exc
        )
        log.debug(traceback.format_exc())
        return

    if arr.size == 0:
        log.error("Cannot render 3-D BOOD: bod.bond_order is empty.")
        return

    expected_shape = (n_theta, n_phi)
    if arr.shape != expected_shape:
        log.error(
            "Cannot render 3-D BOOD: unexpected bond_order array shape.\n"
            "  Expected : %s\n"
            "  Found    : %s\n"
            "  Hint     : Check whether n_theta/n_phi match the BondOrder histogram size.",
            expected_shape,
            arr.shape
        )
        return

    # ------------------------------------------------------------------
    # 5. Normalization for coloring
    # ------------------------------------------------------------------
    try:
        vmin, vmax = arr.min(), arr.max()
        norm_arr = (arr - vmin) / (vmax - vmin + 1e-30)
    except Exception as exc:
        log.error(
            "Failed to normalize BondOrder array for 3-D coloring.\n"
            "  Error type : %s\n"
            "  Error      : %s",
            type(exc).__name__,
            exc
        )
        log.debug(traceback.format_exc())
        return

    # ------------------------------------------------------------------
    # 6. Colormap lookup
    # ------------------------------------------------------------------
    try:
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        facecolors = cmap(norm_arr)
    except Exception as exc:
        log.error(
            "Failed to load/apply matplotlib colormap '%s'.\n"
            "  Error type : %s\n"
            "  Error      : %s\n"
            "  Hint       : Check whether the colormap name is valid.",
            cmap_name,
            type(exc).__name__,
            exc
        )
        log.debug(traceback.format_exc())
        return

    # ------------------------------------------------------------------
    # 7. Create figure and 3-D axis
    # ------------------------------------------------------------------
    try:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
    except Exception as exc:
        log.error(
            "Failed to create matplotlib 3-D figure/axis for BOOD.\n"
            "  Error type : %s\n"
            "  Error      : %s",
            type(exc).__name__,
            exc
        )
        log.debug(traceback.format_exc())
        return


    # ------------------------------------------------------------------
    # 8. Plot sphere surface
    # ------------------------------------------------------------------
    try:
        ax.plot_surface(
            X, Y, Z,
            facecolors=facecolors,
            rstride=1, cstride=1,
            linewidth=0,
            antialiased=True,
            shade=False,
        )
    except Exception as exc:
        log.error(
            "Failed while plotting the 3-D BOOD surface.\n"
            "  Error type : %s\n"
            "  Error      : %s",
            type(exc).__name__,
            exc
        )
        log.debug(traceback.format_exc())
        plt.close(fig)
        return

    # ------------------------------------------------------------------
    # 9. Colorbar
    # ------------------------------------------------------------------
    try:
        sm = cm.ScalarMappable(
            cmap=cmap,
            norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.08)
        cbar.set_label("Bond density (counts)")
    except Exception as exc:
        log.warning(
            "3-D BOOD surface was created, but colorbar generation failed.\n"
            "  Error type : %s\n"
            "  Error      : %s",
            type(exc).__name__,
            exc
        )
        log.debug(traceback.format_exc())

    # ------------------------------------------------------------------
    # 10. Labels / title / layout
    # ------------------------------------------------------------------
    try:
        ax.set_title("Bond Orientational Order Diagram - 3D", fontsize=17, pad=1)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])
        plt.tight_layout()
    except Exception as exc:
        log.warning(
            "3-D BOOD figure was created, but figure formatting/layout failed.\n"
            "  Error type : %s\n"
            "  Error      : %s",
            type(exc).__name__,
            exc
        )
        log.debug(traceback.format_exc())

    # ------------------------------------------------------------------
    # 11. Backend check + interactive display
    # ------------------------------------------------------------------
    try:
        backend = matplotlib.get_backend().lower()
    except Exception as exc:
        log.warning(
            "Could not determine matplotlib backend before 3-D display.\n"
            "  Error type : %s\n"
            "  Error      : %s\n"
            "  Falling back to skipping interactive display.",
            type(exc).__name__,
            exc
        )
        log.debug(traceback.format_exc())
        plt.close(fig)
        return

    # Truly non-interactive backends
    noninteractive_backends = {"agg", "pdf", "ps", "svg", "template", "cairo"}

    if backend in noninteractive_backends:
        log.info(
            "3-D interactive display skipped because current backend '%s' is non-interactive. "
            "Use QtAgg/TkAgg with a GUI session if you want rotation.",
            backend
        )
        plt.close(fig)
        return

    # ------------------------------------------------------------------
    # 12. Show interactive window
    # ------------------------------------------------------------------
    try:
        log.info(
            "  Matplotlib backend '%s' is interactive. Opening 3-D window...",
            backend
        )
        plt.show()
    except Exception as exc:
        log.warning(
            "Interactive 3-D BOOD display failed even though backend '%s' appears interactive.\n"
            "  Error type : %s\n"
            "  Error      : %s\n",
            backend,
            type(exc).__name__,
            exc
        )
        log.debug(traceback.format_exc())
    finally:
        plt.close(fig)


def _save_bod_2d(bod, n_theta: int, n_phi: int, cfg: Dict, out: Path, log: logging.Logger) -> Path:
    """Save the BOOD as a 2-D theta-phi heatmap (Cartesian projection)."""

    try:
        apply_style()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to apply plotting style: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        arr = np.asarray(bod.bond_order)
    except Exception as exc:
        raise RuntimeError(
            f"Could not convert bod.bond_order to NumPy array: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if arr.size == 0:
        raise ValueError("bod.bond_order is empty; cannot save 2-D BondOrder heatmap.")

    try:
        fig, ax = plt.subplots(figsize=cfg["figure_size"])
    except KeyError as exc:
        raise KeyError(
            "Missing cfg['figure_size'] required for 2-D BondOrder plotting."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Could not create matplotlib figure: {type(exc).__name__}: {exc}"
        ) from exc

    try:
        theta_c = np.linspace(0, 360, n_theta)
        phi_c   = np.linspace(0, 180, n_phi)
        Tg, Pg  = np.meshgrid(theta_c, phi_c, indexing="ij")

        im = ax.pcolormesh(
            Tg, Pg, arr,
            cmap=cfg["colormap"],
            shading="auto"
        )
        add_colorbar(ax, im, "Bond density (counts)")
        ax.set_xlabel(r"Azimuthal $\theta$ ($\circ$)")
        ax.set_ylabel(r"Polar $\phi$ ($\circ$)")
        ax.set_title("Bond Orientational Order Diagram - 2D projection")
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 180)

        # _add_averaging_note(ax, True, len(cfg.get("_frames_used", [])))

    except KeyError as exc:
        raise KeyError(
            f"Missing plotting-related config key: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed while constructing the 2-D BondOrder plot: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        fpath = save_fig(fig, out, "01_bond_order_diagram_2D", cfg["dpi"])
        return fpath
    except KeyError as exc:
        raise KeyError(
            "Missing cfg['dpi'] required for saving the 2-D BondOrder figure."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save BondOrder 2-D figure into '{out}': "
            f"{type(exc).__name__}: {exc}"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# 5.2  AngularSeparationNeighbor
# ─────────────────────────────────────────────────────────────────────────────

def run_angular_separation_neighbor(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Compute the minimum angular separation between the orientations of
    neighbouring particle pairs using
    ``freud.environment.AngularSeparationNeighbor``.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    For each bond (i, j) in the neighbour list, the result is the minimum
    rotation angle that maps particle i's orientation onto particle j's,
    after accounting for any equivalent orientations (symmetry group).

    *  Sharp peak near 0 °  → neighbours are co-orientated (crystalline)
    *  Broad distribution   → orientational disorder (liquid or glass)

    Frame averaging
    ~~~~~~~~~~~~~~~
    Angle arrays from all frames are concatenated.  More frames → more
    statistically robust distribution.

    Requires
    ~~~~~~~~
    *  Orientation quaternions stored in the GSD file
    *  ``rowan`` library

    Returns
    -------
    dict with mean, median, and std of angular separations (degrees)
    """
    if not HAS_ROWAN:
        log.warning("  AngularSeparationNeighbor skipped (rowan not installed).")
        return {}

    as_cfg  = cfg["angular_separation_neighbor"]
    equiv_q = np.array(as_cfg["equiv_orientations"], dtype=np.float32)
    nbins   = as_cfg["n_histogram_bins"]
    nq      = build_nq_args(cfg)

    log.info("── AngularSeparationNeighbor ──────────────────────────────────")

    all_angles: List[np.ndarray] = []

    for fi in frames:
        fd = extract_frame_data(traj[fi])
        if fd["orientations"] is None:
            log.warning("  Frame %d: no orientations stored – skipping.", fi)
            continue
        try:
            asn = freud.environment.AngularSeparationNeighbor()
            asn.compute(
                system            = (fd["box"], fd["positions"]),
                orientations      = fd["orientations"],
                query_points      = fd["positions"],
                query_orientations= fd["orientations"],
                equiv_orientations= equiv_q,
                neighbors         = nq,
            )
            all_angles.append(np.rad2deg(np.asarray(asn.angles)))
        except Exception as exc:
            log.warning("  Frame %d AngSepNeighbor error: %s", fi, exc)
            log.debug(traceback.format_exc())

    if not all_angles:
        log.warning("  AngularSeparationNeighbor: no valid frames.")
        return {}

    angles = np.concatenate(all_angles)

    # ── Figure ─────────────────────────────────────────────────────────────
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram
    ax1.hist(angles, bins=nbins, color="steelblue", edgecolor="white", linewidth=0.4)
    ax1.axvline(np.mean(angles),   color="crimson",    ls="--",
                label=f"Mean   = {np.mean(angles):.1f}°")
    ax1.axvline(np.median(angles), color="darkorange", ls=":",
                label=f"Median = {np.median(angles):.1f}°")
    ax1.set_xlabel("Angular separation (°)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Neighbour Angular Separation Distribution")
    ax1.legend()
    _add_averaging_note(ax1, True, len(frames))

    # Right: CDF
    s = np.sort(angles)
    ax2.plot(s, np.arange(1, len(s) + 1) / len(s), color="steelblue", lw=1.5)
    ax2.set_xlabel("Angular separation (°)")
    ax2.set_ylabel("Cumulative probability")
    ax2.set_title("CDF of Neighbour Angular Separations")
    ax2.set_xlim(0, 180)
    ax2.axhline(0.5, color="grey", ls=":", lw=0.8)

    fig.suptitle("freud.environment.AngularSeparationNeighbor", fontsize=12, y=1.01)
    fig.tight_layout()
    fpath = save_fig(fig, out, "02_angular_sep_neighbor", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    return {
        "ang_sep_neighbor_mean_deg"  : float(np.mean(angles)),
        "ang_sep_neighbor_median_deg": float(np.median(angles)),
        "ang_sep_neighbor_std_deg"   : float(np.std(angles)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.3  AngularSeparationGlobal
# ─────────────────────────────────────────────────────────────────────────────

def run_angular_separation_global(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Compute the minimum angular separation between each particle's
    orientation and a set of global reference orientations using
    ``freud.environment.AngularSeparationGlobal``.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    How close is each particle's orientation to one of the ideal/reference
    orientations?

    *  Low angles → particle is nearly aligned with a reference (ordered)
    *  Large angles → particle deviates strongly (disordered)

    This is used to detect grains that are misoriented relative to a
    target lattice orientation, or to measure rotational drift during a
    simulation.

    Frame averaging
    ~~~~~~~~~~~~~~~
    Per-particle minimum angles are pooled across all frames.

    Requires
    ~~~~~~~~
    Orientation quaternions in the GSD file + ``rowan``.

    Returns
    -------
    dict with mean and median angular separation (degrees)
    """
    if not HAS_ROWAN:
        log.warning("  AngularSeparationGlobal skipped (rowan not installed).")
        return {}

    as_cfg   = cfg["angular_separation_global"]
    global_q = np.array(as_cfg["global_orientations"], dtype=np.float32)
    equiv_q  = np.array(as_cfg["equiv_orientations"],  dtype=np.float32)
    nbins    = as_cfg["n_histogram_bins"]

    log.info("── AngularSeparationGlobal ────────────────────────────────────")

    all_min_angles: List[np.ndarray] = []

    for fi in frames:
        fd = extract_frame_data(traj[fi])
        if fd["orientations"] is None:
            log.warning("  Frame %d: no orientations – skipping.", fi)
            continue
        try:
            asg = freud.environment.AngularSeparationGlobal()
            asg.compute(
                global_orientations= global_q,
                orientations        = fd["orientations"],
                equiv_orientations  = equiv_q,
            )
            # angles shape: (N_particles, N_global_refs)
            # Take per-particle minimum over all global references
            per_particle_min = np.min(np.asarray(asg.angles), axis=1)
            all_min_angles.append(np.rad2deg(per_particle_min))
        except Exception as exc:
            log.warning("  Frame %d AngSepGlobal error: %s", fi, exc)
            log.debug(traceback.format_exc())

    if not all_min_angles:
        log.warning("  AngularSeparationGlobal: no valid frames.")
        return {}

    angles = np.concatenate(all_min_angles)

    # Compute mean per global orientation on last available frame
    last_fi = None
    for fi in reversed(frames):
        fd = extract_frame_data(traj[fi])
        if fd["orientations"] is not None:
            last_fi = fi
            last_fd = fd
            break

    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: histogram of pooled per-particle minimum separation
    ax1.hist(angles, bins=nbins, color="teal", edgecolor="white", linewidth=0.4)
    ax1.axvline(np.mean(angles), color="crimson", ls="--",
                label=f"Mean = {np.mean(angles):.1f}°")
    ax1.set_xlabel("Min. angular sep. to any global ref (°)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Global Angular Separation Distribution")
    ax1.legend()
    _add_averaging_note(ax1, True, len(frames))

    # Right: mean per-reference bar chart (last frame)
    if last_fi is not None:
        asg2 = freud.environment.AngularSeparationGlobal()
        try:
            asg2.compute(global_q, last_fd["orientations"], equiv_q)
            means = np.rad2deg(np.mean(np.asarray(asg2.angles), axis=0))
            x = np.arange(len(means))
            ax2.bar(x, means, color="teal", alpha=0.8)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"Ref {i}" for i in x], rotation=45, ha="right")
            ax2.set_ylabel("Mean separation (°)")
            ax2.set_title(f"Mean Sep. per Global Reference\n(frame {last_fi})")
        except Exception as exc:
            log.warning("  Per-reference bar chart failed: %s", exc)
            ax2.text(0.5, 0.5, "Not available", transform=ax2.transAxes,
                     ha="center", va="center")

    fig.suptitle("freud.environment.AngularSeparationGlobal", fontsize=12, y=1.01)
    fig.tight_layout()
    fpath = save_fig(fig, out, "03_angular_sep_global", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    return {
        "ang_sep_global_mean_deg"  : float(np.mean(angles)),
        "ang_sep_global_median_deg": float(np.median(angles)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.4  LocalDescriptors
# ─────────────────────────────────────────────────────────────────────────────

def run_local_descriptors(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Compute per-bond spherical harmonic descriptors using
    ``freud.environment.LocalDescriptors``.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    Each bond vector (connecting particle i to neighbour j) is projected
    onto the complete set of spherical harmonics Y_l^m up to order l_max.
    The resulting complex coefficients form a rotationally-invariant
    fingerprint of the local environment (in 'global' mode):

    *  |Y_l^m|² averaged over bonds → power spectrum  (real-valued)
    *  Q_l = √(4π/(2l+1) × Σ_m |mean(Y_l^m)|²) → Steinhardt-like order
       parameter at each angular order l.

    Peaks at specific l indicate symmetry:
    *  l=4 or l=6 dominate → FCC / HCP crystal
    *  l=2            → nematic order
    *  l=6            → icosahedral local order

    Frame averaging
    ~~~~~~~~~~~~~~~
    The power spectrum is averaged over all selected frames for stability.

    Returns
    -------
    dict mapping 'local_descriptors_Ql' to {l: Ql_value} for l=0..l_max
    """
    ld_cfg = cfg["local_descriptors"]
    l_max  = ld_cfg["l_max"]
    mode   = ld_cfg["mode"]
    nq     = build_nq_args(cfg)

    log.info("── LocalDescriptors  (l_max=%d, mode=%s) ─────────────────────", l_max, mode)

    # Total number of SPH coefficients for this l_max
    n_coeffs = sum(2 * l + 1 for l in range(l_max + 1))

    accum_power = np.zeros(n_coeffs, dtype=np.float64)
    n_accum     = 0

    for fi in frames:
        fd     = extract_frame_data(traj[fi])
        system = (fd["box"], fd["positions"])

        if mode == "particle_local" and fd["orientations"] is None:
            log.warning(
                "  Frame %d: mode=particle_local needs orientations – skipped.", fi
            )
            continue

        try:
            ld = freud.environment.LocalDescriptors(l_max, mode=mode)
            kwargs = {"system": system, "neighbors": nq}
            if mode == "particle_local":
                kwargs["orientations"] = fd["orientations"]
            ld.compute(**kwargs)
            # sph shape: (N_bonds, n_coeffs) complex
            power = np.mean(np.abs(np.asarray(ld.sph)) ** 2, axis=0).real
            accum_power += power
            n_accum += 1
        except Exception as exc:
            log.warning("  Frame %d LocalDescriptors error: %s", fi, exc)
            log.debug(traceback.format_exc())

    if n_accum == 0:
        log.error("  LocalDescriptors: all frames failed.")
        return {}

    mean_power = accum_power / n_accum

    # ── Reshape into (l_max+1, 2*l_max+1) matrix for visualisation ─────────
    # Indexing: for each l, m runs 0, 1, …, l, then −l, …, −1
    # We centre at column l_max so m=0 is always at the centre column.
    mat   = np.zeros((l_max + 1, 2 * l_max + 1))
    idx   = 0
    Ql    = np.zeros(l_max + 1)

    for l in range(l_max + 1):
        power_l = []
        for m in range(l + 1):          # non-negative m
            mat[l, l_max + m] = mean_power[idx]
            power_l.append(mean_power[idx])
            idx += 1
        for m in range(-l, 0):          # negative m
            mat[l, l_max + m] = mean_power[idx]
            power_l.append(mean_power[idx])
            idx += 1
        # Rotationally invariant combination
        if l > 0:
            Ql[l] = np.sqrt(4 * np.pi / (2 * l + 1) * np.sum(power_l))
        else:
            Ql[0] = np.sqrt(mean_power[0])

    # ── Figure ─────────────────────────────────────────────────────────────
    apply_style()
    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.35)

    # Panel 1: 2-D (l, m) heatmap
    ax1  = fig.add_subplot(gs[0])
    vmax = np.percentile(mat[mat > 0], 99) if np.any(mat > 0) else 1.0
    im   = ax1.imshow(
        mat, aspect="auto", origin="lower", cmap=cfg["colormap"],
        extent=[-l_max - 0.5, l_max + 0.5, -0.5, l_max + 0.5],
        vmin=0, vmax=vmax,
    )
    ax1.set_xlabel("m  (magnetic quantum number)")
    ax1.set_ylabel("l  (angular order)")
    ax1.set_title(rf"Mean |Y$_l^m$|²  (mode: {mode})")
    add_colorbar(ax1, im, "Mean power")
    _add_averaging_note(ax1, True, n_accum)

    # Panel 2: invariant power spectrum Q_l
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(np.arange(l_max + 1), Ql, color="steelblue", alpha=0.85)
    ax2.set_xlabel("Angular order l")
    ax2.set_ylabel(r"$Q_l$  (invariant power)")
    ax2.set_title("Rotationally Invariant Power Spectrum")
    ax2.set_xticks(np.arange(l_max + 1))
    # Annotate dominant l
    dominant_l = int(np.argmax(Ql))
    ax2.axvline(dominant_l, color="crimson", ls="--", lw=1.2,
                label=f"Dominant l={dominant_l}")
    ax2.legend(fontsize=9)

    fig.suptitle(
        f"freud.environment.LocalDescriptors  (l_max={l_max})",
        fontsize=13, y=1.02,
    )
    fpath = save_fig(fig, out, "04_local_descriptors", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    return {"local_descriptors_Ql": {str(l): float(Ql[l]) for l in range(l_max + 1)}}


# ─────────────────────────────────────────────────────────────────────────────
# 5.5  LocalBondProjection
# ─────────────────────────────────────────────────────────────────────────────

def run_local_bond_projection(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Compute the scalar projections of neighbour bond vectors onto
    user-defined reference axes using
    ``freud.environment.LocalBondProjection``.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    For a particle with orientation quaternion q, each bond vector is
    rotated into the particle's local frame and then projected onto a
    set of reference unit vectors.  The result captures how much each
    bond is aligned along each particle axis:

    *  Positive projection → bond points along that axis
    *  Near-zero projection → bond is perpendicular to that axis

    Useful for patchy particles, anisotropic molecules, or any system
    where the bond-axis relationship carries physical meaning.

    Frame averaging
    ~~~~~~~~~~~~~~~
    Projection arrays are concatenated over all selected frames.

    Requires
    ~~~~~~~~
    Orientation quaternions in the GSD file.

    Returns
    -------
    dict with mean and std of projection for each reference axis
    """
    lbp_cfg   = cfg["local_bond_projection"]
    proj_vecs = np.array(lbp_cfg["projection_vecs"], dtype=np.float32)
    n_axes    = len(proj_vecs)
    nq        = build_nq_args(cfg)

    log.info("── LocalBondProjection  (%d axes) ────────────────────────────", n_axes)

    all_proj: List[np.ndarray] = []

    for fi in frames:
        fd = extract_frame_data(traj[fi])
        if fd["orientations"] is None:
            log.warning("  Frame %d: LocalBondProjection needs orientations – skipped.", fi)
            continue
        try:
            lbp = freud.environment.LocalBondProjection()
            lbp.compute(
                system       = (fd["box"], fd["positions"]),
                orientations = fd["orientations"],
                proj_vecs    = proj_vecs,
                neighbors    = nq,
            )
            all_proj.append(np.asarray(lbp.projections))   # (N_bonds, n_axes)
        except Exception as exc:
            log.warning("  Frame %d LocalBondProjection error: %s", fi, exc)
            log.debug(traceback.format_exc())

    if not all_proj:
        log.warning("  LocalBondProjection: no valid frames.")
        return {}

    proj = np.vstack(all_proj)   # (N_total_bonds, n_axes)

    apply_style()
    fig, axes = plt.subplots(2, n_axes, figsize=(4 * n_axes, 8), squeeze=False)

    summary: Dict[str, Any] = {}

    for i in range(n_axes):
        data = proj[:, i]
        # Row 0: histogram
        ax_h = axes[0, i]
        ax_h.hist(data, bins=40, color="mediumpurple",
                  edgecolor="white", linewidth=0.4)
        ax_h.axvline(0, color="black", lw=0.8, ls=":")
        ax_h.set_xlabel(f"Projection value")
        ax_h.set_ylabel("Frequency" if i == 0 else "")
        ax_h.set_title(f"Axis {i}: {proj_vecs[i].tolist()}")
        ax_h.axvline(np.mean(data), color="crimson", ls="--",
                     label=f"μ={np.mean(data):.3f}")
        ax_h.legend(fontsize=8)

        # Row 1: box plot for quick comparison across axes
        ax_b = axes[1, i]
        bp = ax_b.boxplot(data, widths=0.5, patch_artist=True,
                          boxprops=dict(facecolor="mediumpurple", alpha=0.7))
        ax_b.set_xlabel(f"Axis {i}")
        ax_b.set_ylabel("Projection" if i == 0 else "")
        ax_b.set_xticks([])

        summary[f"lbp_mean_axis{i}"] = float(np.mean(data))
        summary[f"lbp_std_axis{i}"]  = float(np.std(data))

    # Add averaging note to first histogram
    _add_averaging_note(axes[0, 0], True, len(frames))

    fig.suptitle("freud.environment.LocalBondProjection", fontsize=13, y=1.01)
    fig.tight_layout()
    fpath = save_fig(fig, out, "05_local_bond_projection", cfg["dpi"])
    log.info("  Saved → %s", fpath)
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 5.6  EnvironmentCluster
# ─────────────────────────────────────────────────────────────────────────────

def run_environment_cluster(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Identify and visualise clusters of particles that share matching local
    environments using ``freud.environment.EnvironmentCluster``.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    Two particles belong to the same environment cluster if the set of
    bond vectors to their respective neighbours can be mapped onto each
    other under rotation (within ``threshold``).

    Clusters typically correspond to:
    *  Crystalline grains of different orientations
    *  Distinct structural polymorphs within the same simulation box
    *  Ordered domains embedded in an amorphous matrix

    Frame averaging
    ~~~~~~~~~~~~~~~
    **NOT applied.**  Cluster assignments are a spatial snapshot.
    Averaging cluster IDs across frames is not physically meaningful.
    Analysis always uses the **last selected frame**.

    Returns
    -------
    dict with num_clusters, largest_cluster_size, and the frame index used
    """
    ec_cfg   = cfg["environment_cluster"]
    thresh   = ec_cfg["threshold"]
    reg      = ec_cfg["registration"]
    nq       = build_nq_args(cfg)
    fi       = frames[-1]    # last frame only

    log.info("── EnvironmentCluster  (frame %d, threshold=%.3f) ────────────", fi, thresh)

    fd     = extract_frame_data(traj[fi])
    system = (fd["box"], fd["positions"])

    try:
        ec = freud.environment.EnvironmentCluster()
        ec.compute(
            system           = system,
            threshold        = thresh,
            cluster_neighbors= nq,
            registration     = reg,
        )
    except Exception as exc:
        log.error("  EnvironmentCluster failed: %s", exc)
        log.debug(traceback.format_exc())
        return {}

    cl_idx    = np.asarray(ec.cluster_idx)
    n_clusters = ec.num_clusters
    log.info("  → %d environment clusters identified", n_clusters)

    unique, counts = np.unique(cl_idx, return_counts=True)
    order          = np.argsort(-counts)        # sort by size descending
    sorted_sizes   = counts[order]
    sorted_ids     = unique[order]

    # ── Figure ─────────────────────────────────────────────────────────────
    apply_style()
    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.38)

    # Panel 1: cluster size distribution (log scale)
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(np.arange(len(sorted_sizes)), sorted_sizes,
            color="slateblue", alpha=0.85)
    ax1.set_xlabel("Cluster rank (largest first)")
    ax1.set_ylabel("Cluster size (particles)")
    ax1.set_title(
        f"Environment Cluster Sizes\n{n_clusters} clusters, frame {fi}"
    )
    if sorted_sizes.max() / max(sorted_sizes.min(), 1) > 20:
        ax1.set_yscale("log")
    _add_averaging_note(ax1, False, 1)

    # Panel 2: spatial 2-D scatter coloured by cluster ID
    ax2   = fig.add_subplot(gs[1])
    pos   = fd["positions"]
    n_show = min(n_clusters, 20)   # only colour the top-20 biggest clusters
    cmap20 = cm.get_cmap("tab20", n_show)
    rank_map = {cid: r for r, cid in enumerate(sorted_ids[:n_show])}

    point_colors = [
        cmap20(rank_map[c]) if c in rank_map else (0.8, 0.8, 0.8, 0.4)
        for c in cl_idx
    ]
    ax2.scatter(pos[:, 0], pos[:, 1], c=point_colors, s=8,
                linewidths=0, alpha=0.85)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(
        f"Cluster Map – frame {fi}\n"
        f"(top {n_show} of {n_clusters} clusters coloured)"
    )
    ax2.set_aspect("equal")

    fig.suptitle("freud.environment.EnvironmentCluster", fontsize=13, y=1.02)
    fpath = save_fig(fig, out, "06_environment_cluster", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    return {
        "env_cluster_num_clusters" : int(n_clusters),
        "env_cluster_largest_size" : int(sorted_sizes[0]) if len(sorted_sizes) else 0,
        "env_cluster_frame"        : int(fi),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.7  EnvironmentMotifMatch
# ─────────────────────────────────────────────────────────────────────────────

def run_environment_motif_match(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Match each particle's local environment against a reference motif using
    ``freud.environment.EnvironmentMotifMatch``.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    A *motif* is a set of bond vectors defining the ideal neighbourhood of
    a target crystal structure (e.g., the 12 FCC nearest-neighbour
    vectors).  The algorithm checks whether each particle's actual
    neighbourhood can be mapped onto this motif (within ``threshold``).

    *  Match fraction ≈ 1 → system is largely in the target structure
    *  Match fraction ≈ 0 → system does not resemble the reference

    Frame averaging
    ~~~~~~~~~~~~~~~
    **NOT applied.**  Match assignments are per-particle snapshots.
    Analysis always uses the **last selected frame**.

    Motif selection
    ~~~~~~~~~~~~~~~
    *  If ``params.environment_motif_match.motif`` is set, it is used directly.
    *  Otherwise the motif is auto-extracted from particle 0's neighbourhood
       in the last frame.

    Returns
    -------
    dict with n_matched, n_total, fraction, and frame index
    """
    emm_cfg = cfg["environment_motif_match"]
    thresh  = emm_cfg["threshold"]
    reg     = emm_cfg["registration"]
    motif_r = emm_cfg["motif"]
    nq      = build_nq_args(cfg)
    fi      = frames[-1]    # last frame only

    log.info("── EnvironmentMotifMatch  (frame %d, threshold=%.3f) ─────────", fi, thresh)

    fd     = extract_frame_data(traj[fi])
    system = (fd["box"], fd["positions"])

    # ── Build motif ─────────────────────────────────────────────────────────
    if motif_r is None:
        log.info("  No motif specified – auto-extracting from particle 0.")
        motif = _extract_motif(fd["box"], fd["positions"], nq, 0)
        log.info("  Auto-motif: %d bond vectors", len(motif))
    else:
        motif = np.array(motif_r, dtype=np.float32)
        log.info("  User motif: %d bond vectors", len(motif))

    if len(motif) == 0:
        log.error("  Motif is empty – skipping EnvironmentMotifMatch.")
        return {}

    try:
        emm = freud.environment.EnvironmentMotifMatch()
        emm.compute(
            system      = system,
            motif       = motif,
            threshold   = thresh,
            neighbors   = nq,
            registration= reg,
        )
        matches = np.asarray(emm.matches, dtype=bool)
    except Exception as exc:
        log.error("  EnvironmentMotifMatch failed: %s", exc)
        log.debug(traceback.format_exc())
        return {}

    n_match = int(np.sum(matches))
    n_total = len(matches)
    frac    = n_match / n_total if n_total > 0 else 0.0
    log.info("  → %d / %d particles match  (%.1f %%)", n_match, n_total, 100 * frac)

    # ── Figure ─────────────────────────────────────────────────────────────
    apply_style()
    fig = plt.figure(figsize=(14, 5))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.40)

    pos    = fd["positions"]
    colors = np.where(matches, "dodgerblue", "lightgrey")

    # Panel 1: spatial map
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(pos[:, 0], pos[:, 1], c=colors, s=10,
                linewidths=0, alpha=0.85)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(
        f"Motif Match Map – frame {fi}\n"
        f"{n_match}/{n_total} matched  ({100*frac:.1f} %)"
    )
    ax1.set_aspect("equal")
    ax1.legend(
        handles=[
            Patch(color="dodgerblue", label=f"Matched ({n_match})"),
            Patch(color="lightgrey",  label=f"Unmatched ({n_total-n_match})"),
        ],
        framealpha=0.85,
    )
    _add_averaging_note(ax1, False, 1)

    # Panel 2: bar chart summary
    ax2 = fig.add_subplot(gs[1])
    bars = ax2.bar(
        ["Matched", "Not matched"],
        [n_match, n_total - n_match],
        color=["dodgerblue", "lightgrey"],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, [n_match, n_total - n_match]):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 str(val), ha="center", va="bottom", fontsize=11)
    ax2.set_ylabel("Number of particles")
    ax2.set_title("Motif Match Summary")

    fig.suptitle("freud.environment.EnvironmentMotifMatch", fontsize=13, y=1.02)
    fpath = save_fig(fig, out, "07_environment_motif_match", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    return {
        "env_motif_match_n_matched"  : n_match,
        "env_motif_match_n_total"    : n_total,
        "env_motif_match_fraction"   : float(frac),
        "env_motif_match_frame"      : int(fi),
    }


def _extract_motif(
    box: freud.box.Box,
    positions: np.ndarray,
    nq_args: dict,
    particle_idx: int,
) -> np.ndarray:
    """
    Extract the bond-vector neighbourhood of a single reference particle.

    Used to auto-construct the motif when none is provided.

    Parameters
    ----------
    box          : freud.box.Box
    positions    : (N,3) ndarray
    nq_args      : neighbour query arguments dict
    particle_idx : int   Index of the reference particle

    Returns
    -------
    (K, 3) ndarray of bond vectors in real-space units
    """
    aq    = freud.locality.AABBQuery(box, positions)
    nlist = aq.query(positions[[particle_idx]], nq_args).toNeighborList()

    bond_vecs = []
    for bond in nlist:
        j  = bond[1]
        dr = box.wrap(positions[j] - positions[particle_idx])
        bond_vecs.append(dr)

    return np.array(bond_vecs, dtype=np.float32) if bond_vecs else np.empty((0, 3), dtype=np.float32)


# =============================================================================
# ⑥  ANNOTATION HELPER
# =============================================================================

def _add_averaging_note(ax, averaged: bool, n_frames: int):
    """
    Add a small annotation in the upper-left corner of *ax* stating
    whether this panel used frame-averaged data and over how many frames.
    """
    if averaged:
        label = f"Frame-averaged  (n={n_frames})"
        color = "#1a7f1a"
    else:
        label = "Last frame only"
        color = "#8b0000"
    ax.text(
        0.02, 0.98, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8,
        color=color,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec=color),
    )


# =============================================================================
# ⑦  RESULTS SUMMARY  +  JSON
# =============================================================================

# Lookup: does each analysis support frame averaging?
FRAME_AVG_SUPPORT = {
    "bond_order"                  : True,
    "angular_separation_neighbor" : True,
    "angular_separation_global"   : True,
    "local_descriptors"           : True,
    "local_bond_projection"       : True,
    "environment_cluster"         : False,
    "environment_motif_match"     : False,
}

ANALYSIS_DISPLAY_NAMES = {
    "bond_order"                  : "BondOrder  (BOOD)",
    "angular_separation_neighbor" : "AngularSeparationNeighbor",
    "angular_separation_global"   : "AngularSeparationGlobal",
    "local_descriptors"           : "LocalDescriptors",
    "local_bond_projection"       : "LocalBondProjection",
    "environment_cluster"         : "EnvironmentCluster",
    "environment_motif_match"     : "EnvironmentMotifMatch",
}


def print_summary(
    cfg: Dict[str, Any],
    results: Dict[str, Any],
    n_frames: int,
    log: logging.Logger,
):
    """
    Print a rich summary table after all analyses complete.
    Columns: analysis name | frame mode | status | representative scalar
    """
    W = 90
    log.info("")
    log.info("═" * W)
    log.info("   FREUD  ENVIRONMENT  ANALYSIS  –  RESULTS  SUMMARY")
    log.info("═" * W)
    log.info(
        f"  {'Analysis':<35} {'Frame mode':<26} {'Status':<10} Key result"
    )
    log.info("─" * W)

    for key, name in ANALYSIS_DISPLAY_NAMES.items():
        sub = cfg.get(key, {})
        enabled = sub.get("enabled", False) if isinstance(sub, dict) else False

        if not enabled:
            log.info(f"  {name:<35} {'—':<26} {'DISABLED':<10}")
            continue

        supports_avg = FRAME_AVG_SUPPORT[key]
        want_avg     = cfg["frame_average"]

        if supports_avg and want_avg:
            mode_str = f"frame-avg ({n_frames} frames)"
        elif supports_avg:
            mode_str = "last frame (avg=OFF)"
        else:
            mode_str = "last frame  ★ always"

        res  = results.get(key, {})
        ok   = bool(res)
        stat = "✓ OK" if ok else "✗ FAIL"

        kresult = ""
        if ok:
            first_k = next(iter(res))
            first_v = res[first_k]
            if isinstance(first_v, float):
                kresult = f"{first_k} = {first_v:.4f}"
            elif isinstance(first_v, int):
                kresult = f"{first_k} = {first_v}"
            elif isinstance(first_v, dict):
                # e.g. Ql dict – show a couple entries
                items = list(first_v.items())[:3]
                kresult = ", ".join(f"l{k}={v:.3f}" for k, v in items) + " …"
            else:
                kresult = str(first_v)[:40]

        log.info(f"  {name:<35} {mode_str:<26} {stat:<10} {kresult}")

    log.info("─" * W)
    log.info("  ★ EnvironmentCluster and EnvironmentMotifMatch are ALWAYS")
    log.info("    analysed on the last selected frame only (snapshot analyses).")
    log.info("  All frame-averaged analyses benefit from longer trajectories.")
    log.info("═" * W)
    log.info("")


def save_summary_json(
    results: Dict[str, Any],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
):
    """Write a human-readable JSON summary of all scalar results."""
    payload = {
        "run_timestamp": datetime.now().isoformat(),
        "trajectory"   : cfg["trajectory"],
        "frame_average": cfg["frame_average"],
        "frame_start"  : cfg["frame_start"],
        "frame_end"    : cfg["frame_end"],
        "frame_step"   : cfg["frame_step"],
        "results"      : results,
    }
    path = out / "summary.json"
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    log.info("Summary JSON => %s", path)


# =============================================================================
# ⑧  MAIN PIPELINE
# =============================================================================

def main(config_path: str):
    """
    Orchestrate the full analysis pipeline.

    1.  Load & merge configuration
    2.  Set up logging
    3.  Open GSD trajectory
    4.  Resolve frame indices
    5.  Run each enabled analysis (in a safe try/except wrapper)
    6.  Print summary table
    7.  Save summary JSON
    """
    cfg = load_config(config_path)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log     = _setup_logging(Path("logs"))

    log.info("freud  %s", freud.__version__)
    log.info("Config : %s", Path(config_path).resolve())
    log.info("Output : %s", out_dir.resolve())

    traj    = open_trajectory(cfg["trajectory"], log)
    all_fi  = resolve_frame_indices(
        traj,
        cfg["frame_start"], cfg["frame_end"], cfg["frame_step"],
        log,
    )

    # For analyses that support averaging: use full list when frame_average=True,
    # else fall back to [last frame].
    avg_frames  = all_fi if cfg["frame_average"] else [all_fi[-1]]
    snap_frames = all_fi        # snapshot analyses always get full list so they
                                # can pick the last element themselves

    results: Dict[str, Any] = {}
    t0 = time.perf_counter()

    def _run(key: str, fn, frames):
        """Wrapper that catches any uncaught exception from an analysis."""
        try:
            results[key] = fn(traj, frames, cfg, out_dir, log)
        except Exception as exc:
            log.error("[%s] Uncaught exception: %s", key, exc)
            log.debug(traceback.format_exc())
            results[key] = {}

    if cfg["bond_order"].get("enabled"):
        _run("bond_order", run_bond_order, avg_frames)

    if cfg["angular_separation_neighbor"].get("enabled"):
        _run("angular_separation_neighbor", run_angular_separation_neighbor, avg_frames)

    if cfg["angular_separation_global"].get("enabled"):
        _run("angular_separation_global", run_angular_separation_global, avg_frames)

    if cfg["local_descriptors"].get("enabled"):
        _run("local_descriptors", run_local_descriptors, avg_frames)

    if cfg["local_bond_projection"].get("enabled"):
        _run("local_bond_projection", run_local_bond_projection, avg_frames)

    if cfg["environment_cluster"].get("enabled"):
        _run("environment_cluster", run_environment_cluster, snap_frames)

    if cfg["environment_motif_match"].get("enabled"):
        _run("environment_motif_match", run_environment_motif_match, snap_frames)

    elapsed = time.perf_counter() - t0
    log.info("All analyses finished in %.2f s.", elapsed)

    print_summary(cfg, results, len(avg_frames), log)
    save_summary_json(results, cfg, out_dir, log)

    traj.close()
    log.info("Done. Outputs in: %s", out_dir.resolve())


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog        = "analyze_environment.py",
        description = "freud.environment analysis pipeline (all seven analyses)",
        epilog      = (
            "Example:\n"
            "  python analyze_environment.py params.json\n\n"
            "Edit params.json to enable/disable individual analyses and\n"
            "adjust all parameters without touching the source code."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs   = "?",
        default = "params.json",
        help    = "Path to the JSON parameter file  (default: params.json)",
    )
    args = parser.parse_args()
    main(args.config)
