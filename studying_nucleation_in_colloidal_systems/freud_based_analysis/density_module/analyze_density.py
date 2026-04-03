"""
=============================================================================
freud_env_analysis/density_addon.py
=============================================================================
ADD-ON MODULE  –  freud.density analyses
=============================================================================
This file contains ONLY the new code needed to extend analyze_environment.py
with every analysis in the freud.density module. It follows the identical
SECTION A–D structure as diffraction_addon.py and order_addon.py.

Analyses implemented (freud.density module):
  16.  RDF                 – Radial Distribution Function g(r)
  17.  CorrelationFunction – Complex pairwise spatial correlation function C(r)
  18.  LocalDensity        – Per-particle local number density
  19.  GaussianDensity     – Gaussian-blurred density field on a grid
  20.  SphereVoxelization  – Binary voxel grid of sphere-occupied space

Frame-averaging rationale
--------------------------
  RDF                 → YES – histogram accumulation with reset=False;
                              more frames → smoother, more accurate g(r)
  CorrelationFunction → YES – same histogram accumulation logic as RDF
  LocalDensity        → YES – per-particle density pooled across frames
  GaussianDensity     → NO  – grid is a spatial snapshot; last frame only
                              (optional frame_average_override)
  SphereVoxelization  → NO  – voxel occupancy is a spatial snapshot;
                              last frame only

=============================================================================
HOW TO INSERT THIS CODE INTO analyze_environment.py
=============================================================================

STEP 1 ── Top-level docstring
  Add to "Analyses implemented":
      16. RDF                  – Radial Distribution Function g(r)
      17. CorrelationFunction  – Complex pairwise spatial correlation C(r)
      18. LocalDensity         – Per-particle local number density
      19. GaussianDensity      – Gaussian-blurred density field on grid
      20. SphereVoxelization   – Binary sphere-voxel occupancy grid

  Add to "Frame-averaging rationale":
      RDF                 → frame-avg (YES)
      CorrelationFunction → frame-avg (YES)
      LocalDensity        → frame-avg (YES)
      GaussianDensity     → last frame (configurable override)
      SphereVoxelization  → last frame only

─────────────────────────────────────────────────────────────────────────────
STEP 2 ── DEFAULT_CONFIG  (Section ②)
  Paste SECTION A after the last existing entry (e.g.
  "rotational_autocorrelation" or "static_sf_direct") inside DEFAULT_CONFIG.

─────────────────────────────────────────────────────────────────────────────
STEP 3 ── Analysis functions  (Section ⑤)
  Paste SECTION B after the last existing analysis function in Section ⑤.

─────────────────────────────────────────────────────────────────────────────
STEP 4 ── Lookup dicts  (Section ⑦)
  Add entries from SECTION C to FRAME_AVG_SUPPORT and
  ANALYSIS_DISPLAY_NAMES.

─────────────────────────────────────────────────────────────────────────────
STEP 5 ── main() dispatch calls  (Section ⑧)
  Paste SECTION D after the last existing _run() call.

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
    logger = logging.getLogger("freud_density")
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


    # ── 16. RDF ──────────────────────────────────────────────────────────────
    "rdf": {
        "enabled"           : True,
        "bins"              : 100,
        "r_max"             : 5.0,
        "r_min"             : 0.0,
        "normalization_mode": "exact"
    },

    # ── 17. CorrelationFunction ───────────────────────────────────────────────
    "correlation_function": {
        "enabled"   : True,
        "bins"      : 50,
        "r_max"     : 5.0,
        "value_mode": "orientation_k",
        "symmetry_k": 4
    },

    # ── 18. LocalDensity ─────────────────────────────────────────────────────
    "local_density": {
        "enabled"  : True,
        "r_max"    : 3.0,
        "diameter" : 1.0
    },

    # ── 19. GaussianDensity ──────────────────────────────────────────────────
    "gaussian_density": {
        "enabled"               : True,
        "width"                 : 128,
        "r_max"                 : 3.0,
        "sigma"                 : 0.5,
        "frame_average_override": False
    },

    # ── 20. SphereVoxelization ───────────────────────────────────────────────
    "sphere_voxelization": {
        "enabled": True,
        "width"  : 64,
        "r_max"  : 1.0
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



def run_rdf(traj, frames: List[int], cfg: Dict[str, Any], out: Path, log: logging.Logger) -> Dict[str, Any]:
    """
    Compute the radial distribution function g(r) using ``freud.density.RDF``.

    What this routine does
    ----------------------
    1. Validates the RDF-related user inputs from ``cfg["rdf"]``.
    2. Builds the freud RDF object.
    3. Iterates over the requested trajectory frames and accumulates RDF data.
    4. Extracts physically useful summary quantities such as:
         - first RDF peak position
         - first RDF peak height
         - first minimum after the first peak
         - coordination number at that first minimum
    5. Writes a two-panel diagnostic figure:
         - g(r)
         - cumulative coordination number n(r)

    Returns
    -------
    Dict[str, Any]
        Summary dictionary. Returns {} only when RDF computation itself
        could not be completed meaningfully.
    """

    # ------------------------------------------------------------------
    # Step 1: Read and validate RDF configuration early.
    # ------------------------------------------------------------------
    '''rdf_cfg   = cfg["rdf"]
    bins      = int(rdf_cfg["bins"])
    r_max     = float(rdf_cfg["r_max"])
    r_min     = float(rdf_cfg["r_min"])
    norm_mode = str(rdf_cfg["normalization_mode"])

    log.info("── RDF  (bins=%d, r_min=%.2f, r_max=%.2f, norm=%s) ──────────",
             bins, r_min, r_max, norm_mode)'''


    try:
        rdf_cfg = cfg["rdf"]
    except KeyError:
        log.error(
            "RDF configuration missing: expected cfg['rdf'] in the JSON config."
        )
        return {}

    try:
        bins = int(rdf_cfg["bins"])
        r_max = float(rdf_cfg["r_max"])
        r_min = float(rdf_cfg["r_min"])
        norm_mode = str(rdf_cfg["normalization_mode"]).strip()
    except KeyError as exc:
        log.error(
            "RDF configuration is incomplete. Missing key: %s. "
            "Required keys are: bins, r_min, r_max, normalization_mode.",
            exc,
        )
        log.debug(traceback.format_exc())
        return {}
    except (TypeError, ValueError) as exc:
        log.error(
            "RDF configuration contains invalid value type(s): %s. "
            "Please check bins/r_min/r_max/normalization_mode in params JSON.",
            exc,
        )
        log.debug(traceback.format_exc())
        return {}

    if bins <= 0:
        log.error("RDF config error: 'bins' must be > 0, but got %d.", bins)
        return {}

    if r_min < 0:
        log.error("RDF config error: 'r_min' must be >= 0, but got %.6f.", r_min)
        return {}

    if r_max <= 0:
        log.error("RDF config error: 'r_max' must be > 0, but got %.6f.", r_max)
        return {}

    if r_max <= r_min:
        log.error(
            "RDF config error: 'r_max' must be greater than 'r_min'. "
            "Received r_min=%.6f, r_max=%.6f.",
            r_min,
            r_max,
        )
        return {}

    allowed_norm_modes = {"exact", "finite_size"}
    if norm_mode not in allowed_norm_modes:
        log.error(
            "RDF config error: unsupported normalization_mode='%s'. "
            "Allowed values are %s.",
            norm_mode,
            sorted(allowed_norm_modes),
        )
        return {}

    if not frames:
        log.error(
            "RDF aborted: no frames were provided to run_rdf(). "
        )
        return {}

    log.info(
        "── RDF  (bins=%d, r_min=%.4f, r_max=%.4f, norm=%s, n_frames=%d) ──────────",
        bins,
        r_min,
        r_max,
        norm_mode,
        len(frames),
    )

    # ------------------------------------------------------------------
    # Step 2: Construct the freud RDF object.
    # ------------------------------------------------------------------
    try:
        rdf = freud.density.RDF(
            bins=bins,
            r_max=r_max,
            r_min=r_min,
            normalization_mode=norm_mode,
        )
    except Exception as exc:
        log.error("  RDF init failed: %s", exc)
        log.debug(traceback.format_exc())
        return {}

    # ------------------------------------------------------------------
    # Step 3: Loop over frames and accumulate RDF statistics.
    # ------------------------------------------------------------------
    first_successful_compute = True
    n_ok = 0
    n_skipped = 0
    skipped_frames: List[int] = []

    for fi in frames:
        try:
            fd = extract_frame_data(traj[fi])
            system = (fd["box"], fd["positions"])
        except IndexError as exc:
            log.warning(
                "RDF frame %d skipped: frame index is out of range for this trajectory. "
                "Error: %s",
                fi,
                exc,
            )
            log.debug(traceback.format_exc())
            n_skipped += 1
            skipped_frames.append(fi)
            continue
        except Exception as exc:
            log.warning(
                "RDF frame %d skipped while extracting trajectory data. "
                "Possible causes: malformed frame, missing particle data, incompatible GSD content. "
                "Error: %s",
                fi,
                exc,
            )
            log.debug(traceback.format_exc())
            n_skipped += 1
            skipped_frames.append(fi)
            continue

        try:
            # reset=True only for the first successful compute call.
            rdf.compute(system, reset=first_successful_compute)
            first_successful_compute = False
            n_ok += 1
        except Exception as exc:
            log.warning(
                "RDF frame %d skipped during freud.density.RDF.compute(...). "
                "This can happen if the box/positions are inconsistent or the frame is invalid. "
                "Error: %s",
                fi,
                exc,
            )
            log.debug(traceback.format_exc())
            n_skipped += 1
            skipped_frames.append(fi)

    # If every frame failed, there is nothing meaningful to return.
    if n_ok == 0:
        log.error(
            "RDF failed: all %d requested frame(s) were skipped. "
            "Skipped frames: %s",
            len(frames),
            skipped_frames,
        )
        return {}

    if n_skipped > 0:
        log.warning(
            "RDF completed with partial success: %d frame(s) processed, %d skipped.",
            n_ok,
            n_skipped,
        )

    # ------------------------------------------------------------------
    # Step 4: Extract RDF arrays safely.
    # ------------------------------------------------------------------
    try:
        r = np.asarray(rdf.bin_centers, dtype=np.float64)
        gr = np.asarray(rdf.rdf, dtype=np.float64)
        nr = np.asarray(rdf.n_r, dtype=np.float64)  # cumulative coordination number
    except Exception as exc:
        log.error(
            "RDF post-processing failed while reading bin_centers/rdf/n_r "
            "from the freud RDF object. Error: %s",
            exc,
        )
        log.debug(traceback.format_exc())
        return {}

    if r.size == 0 or gr.size == 0 or nr.size == 0:
        log.error(
            "RDF produced empty output arrays. "
        )
        return {}


    # ------------------------------------------------------------------
    # Step 5: Derive physical peak/minimum information.
    # ------------------------------------------------------------------
    summary = _extract_rdf_peaks(r, gr, nr, log)

    # Add a few generic diagnostics even if no clear peak/minimum is found.
    summary["rdf_n_frames_used"] = int(n_ok)
    summary["rdf_n_frames_skipped"] = int(n_skipped)
    summary["rdf_r_min"] = float(r[0])
    summary["rdf_r_max"] = float(r[-1])
    summary["rdf_gr_max"] = float(np.nanmax(gr))
    summary["rdf_gr_min"] = float(np.nanmin(gr))

    # Coordination number at first minimum, if the minimum exists.
    if "rdf_r1_min" in summary:
        try:
            r_min_loc = summary["rdf_r1_min"]
            n_at_min = float(np.interp(r_min_loc, r, nr))
            summary["rdf_coordination_number"] = n_at_min
        except Exception as exc:
            log.warning(
                "Could not interpolate coordination number at the first RDF minimum. "
                "Error: %s",
                exc,
            )
            log.debug(traceback.format_exc())

    # ------------------------------------------------------------------
    # Step 6: Plotting.
    # ------------------------------------------------------------------
    try:
        apply_style()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 0: g(r)
        ax0 = axes[0]
        ax0.plot(r, gr, color="steelblue", lw=1.8)
        # ax0.axhline(1.0, color="grey", ls=":", lw=0.8, label="g(r)=1 (ideal gas)")
        ax0.set_xlabel(r"$r$ ")
        ax0.set_ylabel(r"$g(r)$")
        ax0.set_title("Radial Distribution Function")
        ax0.set_xlim(r[0], r[-1])
        ax0.set_ylim(bottom=0)
        ax0.legend(fontsize=9)
        # _add_averaging_note(ax0, True, n_ok)

        # Annotate the first peak if found.
        '''if "rdf_r1" in summary and "rdf_g1" in summary:
            r1 = summary["rdf_r1"]
            g1 = summary["rdf_g1"]
            ax0.axvline(r1, color="crimson", ls="--", lw=1.0, alpha=0.8)
            ax0.text(
                r1 + 0.05,
                g1 * 0.92,
                f"r* = {r1:.3f}\ng(r*) = {g1:.2f}",
                fontsize=8,
                color="crimson",
                va="top",
            )
        else:
            log.info(
                "RDF note: no clear first peak satisfying the current peak rule "
                "(local maximum with g(r) > 1) was found."
            )'''

        # Panel 1: cumulative coordination number n(r)
        ax1 = axes[1]
        ax1.plot(r, nr, color="darkorange", lw=1.8)
        ax1.set_xlabel(r"$r$")
        ax1.set_ylabel(r"$n(r)$  (cumulative coordination number)")
        ax1.set_title("Cumulative Coordination Number")
        ax1.set_xlim(r[0], r[-1])
        ax1.set_ylim(bottom=0)
        # _add_averaging_note(ax1, True, n_ok)

        # Annotate first minimum after first peak, if present.
        if "rdf_r1_min" in summary and "rdf_coordination_number" in summary:
            r_min_loc = summary["rdf_r1_min"]
            n_at_min = summary["rdf_coordination_number"]
            ax1.axvline(r_min_loc, color="crimson", ls="--", lw=1.0, alpha=0.8)
            ax1.text(
                r_min_loc + 0.05,
                max(n_at_min * 0.5, 0.05),
                f"N_c ~ {n_at_min:.1f}\n(r = {r_min_loc:.3f})",
                fontsize=8,
                color="crimson",
            )
        else:
            log.info(
                "RDF note: no clear first minimum after the first peak was found, "
                "so coordination-number annotation was skipped."
            )

        fig.suptitle("freud.density.RDF", fontsize=13, y=1.01)
        fig.tight_layout()

        fpath = save_fig(fig, out, "16_rdf", cfg["dpi"])
        log.info("  Saved RDF figure => %s", fpath)

    except Exception as exc:
        log.error(
            "RDF plotting/saving failed, but numerical RDF results were computed successfully. "
            "Error: %s",
            exc,
        )
        log.debug(traceback.format_exc())

    return summary



def _extract_rdf_peaks(r: np.ndarray, gr: np.ndarray, nr: np.ndarray, log: logging.Logger) -> Dict[str, Any]:

    """
    Identify the first physically relevant RDF peak and the first minimum
    immediately after that peak.

    Peak selection rule
    -------------------
    We look for the *first* local maximum satisfying:
        gr[i] > gr[i-1], gr[i] > gr[i+1], and gr[i] > 1

    Minimum selection rule
    ----------------------
    Once the first peak is found, we search forward for the next local
    minimum:
        gr[j] < gr[j-1] and gr[j] < gr[j+1]

    That minimum usually marks the end of the first coordination shell.

    Returns
    -------
    Dict[str, Any]
        May contain:
          - rdf_r1
          - rdf_g1
          - rdf_r1_index
          - rdf_r1_min
          - rdf_r1_min_index

        If no peak or no minimum is found, the function returns a partial
        dictionary and logs a helpful message.
    """

    summary: Dict[str, Any] = {}

    # -------------------- basic sanity checks ---------------------------
    if r is None or gr is None or nr is None:
        log.warning("RDF peak extraction skipped: one or more input arrays are None.")
        return summary


    # -------------------- find first meaningful peak --------------------
    peak_index: Optional[int] = None

    for i in range(1, len(gr) - 1):
        left = gr[i - 1]
        mid = gr[i]
        right = gr[i + 1]

        # Ignore non-finite data points explicitly.
        if not (np.isfinite(left) and np.isfinite(mid) and np.isfinite(right)):
            continue

        if mid > left and mid > right and mid > 1.0:
            peak_index = i
            summary["rdf_r1"] = float(r[i])
            summary["rdf_g1"] = float(gr[i])
            summary["rdf_r1_index"] = int(i)
            log.info("RDF first peak found at bin %d: r=%.6f, g(r)=%.6f", i, r[i], gr[i])
            break

    if peak_index is None:
        log.warning("No clear first RDF peak was found. ")
        return summary

    # -------------------- find first minimum after peak -----------------
    min_index: Optional[int] = None

    for j in range(peak_index + 1, len(gr) - 1):
        left = gr[j - 1]
        mid = gr[j]
        right = gr[j + 1]

        if not (np.isfinite(left) and np.isfinite(mid) and np.isfinite(right)):
            continue

        if mid < left and mid < right:
            min_index = j
            summary["rdf_r1_min"] = float(r[j])
            summary["rdf_r1_min_index"] = int(j)
            log.info("RDF first minimum after first peak found at bin %d: r=%.6f, g(r)=%.6f", j, r[j], gr[j])
            break

    if min_index is None:
        log.warning(
            "First RDF peak was found, but no subsequent local minimum was detected. "
            "Coordination-shell boundary could not be assigned automatically."
        )

    return summary


# ─────────────────────────────────────────────────────────────────────────────

def run_correlation_function(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Compute the complex pairwise spatial correlation function C(r) using
    ``freud.density.CorrelationFunction``.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    C(r) = ⟨s*(rᵢ) · s(rⱼ)⟩  averaged over all pairs (i,j) at separation r

    where s is a complex-valued per-particle scalar encoding some property.

    When ``value_mode = "orientation_k"`` (default), this analysis uses:

        s_i = exp(i k θᵢ)

    where θᵢ is the in-plane angle of particle i's orientation vector and
    k is the rotational symmetry order (e.g., k=4 for squares, k=6 for
    hexagons).  Then:

        C(r) ≈ 1  → particles separated by r have correlated orientations
        C(r) ≈ 0  → orientations are uncorrelated at distance r
        C(r) < 0  → anti-correlated orientations at distance r

    The length scale ξ at which C(r) decays to 1/e is the orientational
    correlation length, a key metric for ordering transitions:
    *  ξ → ∞ in the long-range-ordered crystalline phase
    *  ξ finite in the hexatic / liquid phase
    *  ξ → 0 in the isotropic liquid

    When ``value_mode = "ones"``, s_i = 1 for all particles, which reduces
    C(r) to the pair density (equivalent to the RDF without normalization).
    This mode works even without orientation data in the GSD file.

    Frame averaging: **YES**
    Accumulate across frames with reset=False for stable C(r) statistics.

    Returns
    -------
    dict with C(r) at r→0, decay length estimate, and value of C at r_max
    """
    cf_cfg   = cfg["correlation_function"]
    bins     = int(cf_cfg["bins"])
    r_max    = float(cf_cfg["r_max"])
    vmode    = str(cf_cfg["value_mode"])
    sym_k    = int(cf_cfg["symmetry_k"])

    log.info("── CorrelationFunction  (bins=%d, r_max=%.2f, mode=%s, k=%d) ─",
             bins, r_max, vmode, sym_k)

    try:
        cf = freud.density.CorrelationFunction(bins=bins, r_max=r_max)
    except Exception as exc:
        log.error("  CorrelationFunction init failed: %s", exc)
        log.debug(traceback.format_exc())
        return {}

    first = True
    for fi in frames:
        fd = extract_frame_data(traj[fi])

        # Build per-particle complex values
        vals = _build_cf_values(fd, vmode, sym_k, fi, log)
        if vals is None:
            log.warning("  CorrelationFunction frame %d: could not build values – skipped.", fi)
            continue

        system = (fd["box"], fd["positions"])
        try:
            cf.compute(
                system=system,
                values=vals,
                query_points=fd["positions"],
                query_values=vals,
                reset=first,
            )
            first = False
        except Exception as exc:
            log.warning("  CorrelationFunction frame %d: %s", fi, exc)
            log.debug(traceback.format_exc())

    if first:
        log.error("  CorrelationFunction: all frames failed.")
        return {}

    r       = np.asarray(cf.bin_centers)
    C_r_raw = np.asarray(cf.correlation)
    # The correlation is complex; take the real part (imaginary ≈ 0 for valid data)
    C_r     = np.real(C_r_raw)

    # ── Estimate orientational correlation length ξ ──────────────────────
    xi, C_xi = _estimate_correlation_length(r, C_r, log)

    # ── Figure ─────────────────────────────────────────────────────────────
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 0: C(r) real part
    ax0 = axes[0]
    ax0.plot(r, C_r, color="mediumorchid", lw=1.8)
    ax0.axhline(0.0, color="grey", ls=":", lw=0.8)
    if xi is not None:
        ax0.axvline(xi, color="crimson", ls="--", lw=1.0, alpha=0.8,
                    label=rf"$\xi$ ≈ {xi:.3f}")
        ax0.axhline(C_xi, color="crimson", ls=":", lw=0.7, alpha=0.6)
    ax0.set_xlabel(r"$r$  (simulation length units)")
    ax0.set_ylabel(r"$\mathrm{Re}[C(r)]$")
    title_sym = f"exp(i{sym_k}θ)" if vmode == "orientation_k" else "s=1"
    ax0.set_title(f"Spatial Correlation Function  [{title_sym}]")
    ax0.set_xlim(r[0], r[-1])
    ax0.legend(fontsize=9)
    _add_averaging_note(ax0, True, len(frames))

    # Panel 1: C(r) on log-linear scale for decay visualisation
    ax1 = axes[1]
    C_pos = np.where(C_r > 0, C_r, np.nan)
    ax1.semilogy(r, C_pos, color="mediumorchid", lw=1.8)
    if xi is not None:
        ax1.axvline(xi, color="crimson", ls="--", lw=1.0, alpha=0.8,
                    label=rf"$\xi$ ≈ {xi:.3f}")
        ax1.legend(fontsize=9)
    ax1.set_xlabel(r"$r$  (simulation length units)")
    ax1.set_ylabel(r"$\mathrm{Re}[C(r)]$  (log scale)")
    ax1.set_title("Correlation Function (log scale)")
    ax1.set_xlim(r[0], r[-1])

    fig.suptitle("freud.density.CorrelationFunction", fontsize=13, y=1.01)
    fig.tight_layout()
    fpath = save_fig(fig, out, "17_correlation_function", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    summary: Dict[str, Any] = {
        "cf_C_r_min"  : float(np.nanmin(C_r)),
        "cf_C_r_max"  : float(np.nanmax(C_r)),
        "cf_value_mode": vmode,
        "cf_symmetry_k": sym_k,
    }
    if xi is not None:
        summary["cf_correlation_length_xi"] = float(xi)
    return summary


def _build_cf_values(
    fd: Dict[str, Any],
    vmode: str,
    sym_k: int,
    frame_idx: int,
    log: logging.Logger,
) -> Optional[np.ndarray]:
    """
    Build the complex per-particle value array for CorrelationFunction.

    Parameters
    ----------
    fd       : frame data dict from extract_frame_data
    vmode    : "orientation_k" or "ones"
    sym_k    : rotational symmetry order for orientation_k mode
    frame_idx: frame number (for warnings)
    log      : logger

    Returns
    -------
    Complex (N,) ndarray, or None on failure.
    """
    N = fd["N"]

    if vmode == "ones":
        # Positional correlation only – no orientation needed
        return np.ones(N, dtype=np.complex128)

    if vmode == "orientation_k":
        if fd["orientations"] is None:
            log.warning(
                "  CorrelationFunction frame %d: orientation_k mode needs "
                "quaternions. Falling back to 'ones'.", frame_idx
            )
            return np.ones(N, dtype=np.complex128)

        # Extract in-plane angle θ from quaternion.
        # For 2-D or the xy-plane projection we extract the rotation about z.
        # q = [w, x, y, z] → θ_z = 2 × atan2(z, w)
        q = fd["orientations"].astype(np.float64)
        theta = 2.0 * np.arctan2(q[:, 3], q[:, 0])  # rotation angle about z
        vals  = np.exp(1j * sym_k * theta)
        return vals.astype(np.complex128)

    log.warning("  Unknown value_mode '%s'; using 'ones'.", vmode)
    return np.ones(N, dtype=np.complex128)


def _estimate_correlation_length(
    r: np.ndarray,
    C_r: np.ndarray,
    log: logging.Logger,
) -> tuple:
    """
    Estimate the orientational correlation length ξ as the distance at
    which |C(r)| drops to C(r_min) × exp(-1) (i.e., 1/e of the initial value).

    Returns (xi, C_at_xi) or (None, None) on failure.
    """
    try:
        # Start from the second bin (avoid self-correlation at r=0)
        C0 = float(np.nanmax(np.abs(C_r[:max(3, len(C_r)//10)])))
        if C0 <= 0:
            return None, None
        threshold = C0 * np.exp(-1.0)
        # Find first crossing from above
        for i in range(1, len(C_r)):
            if np.abs(C_r[i]) <= threshold:
                xi    = float(np.interp(threshold, np.abs(C_r[i:i-2:-1]), r[i:i-2:-1]))
                return xi, threshold
        return None, None
    except Exception as exc:
        log.debug("  Correlation length estimation failed: %s", exc)
        return None, None


# ─────────────────────────────────────────────────────────────────────────────

def run_local_density(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Compute the per-particle local number density using
    ``freud.density.LocalDensity``.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    For each query point (= each particle in the self-query case), count
    all data points within a sphere of radius r_max:

        ρ_local(i) = N_neighbours(i) / V_sphere

    where V_sphere = (4/3)π(r_max + diameter/2)³ and ``diameter`` is the
    circumsphere diameter of the data particles.

    *  Uniform ρ_local ≈ ρ_bulk  → homogeneous phase
    *  Bimodal distribution       → phase coexistence (dense + dilute regions)
    *  Spatial gradient in ρ      → interfaces, density waves, sedimentation
    *  ρ_local >> ρ_bulk           → particle is in a high-density cluster

    This is complementary to the RDF: where g(r) tells you the average pair
    structure, LocalDensity gives you the per-particle environment density,
    enabling you to colour-map density directly onto the particle positions.

    Frame averaging: **YES**
    Density arrays are pooled across frames to produce a stable distribution.
    The spatial map is from the last selected frame.

    Returns
    -------
    dict with mean, std, min, max of ρ_local, and the bulk number density ρ
    """
    ld_cfg   = cfg["local_density"]
    r_max    = float(ld_cfg["r_max"])
    diameter = float(ld_cfg["diameter"])

    log.info("── LocalDensity  (r_max=%.2f, diameter=%.3f) ────────────────", r_max, diameter)

    try:
        ld = freud.density.LocalDensity(r_max=r_max, diameter=diameter)
    except Exception as exc:
        log.error("  LocalDensity init failed: %s", exc)
        log.debug(traceback.format_exc())
        return {}

    all_densities: List[np.ndarray] = []
    all_nneighbors: List[np.ndarray] = []

    for fi in frames:
        fd     = extract_frame_data(traj[fi])
        system = (fd["box"], fd["positions"])
        try:
            ld.compute(system)
            all_densities.append(np.asarray(ld.density, dtype=np.float64))
            all_nneighbors.append(np.asarray(ld.num_neighbors, dtype=np.float64))
        except Exception as exc:
            log.warning("  LocalDensity frame %d: %s", fi, exc)
            log.debug(traceback.format_exc())

    if not all_densities:
        log.error("  LocalDensity: all frames failed.")
        return {}

    # Pool all frames for distribution; keep last-frame spatial map
    pool_dens  = np.concatenate(all_densities)
    pool_nn    = np.concatenate(all_nneighbors)
    last_dens  = all_densities[-1]
    last_pos   = extract_frame_data(traj[frames[-1]])["positions"]

    # Bulk density from last frame
    fd_last    = extract_frame_data(traj[frames[-1]])
    box        = fd_last["box"]
    rho_bulk   = float(fd_last["N"] / box.volume)

    apply_style()
    fig = plt.figure(figsize=(16, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.40)

    # Panel 0: local density distribution (pooled)
    ax0 = fig.add_subplot(gs[0])
    ax0.hist(pool_dens, bins=60, color="teal", edgecolor="white", linewidth=0.3,
             density=True)
    ax0.axvline(rho_bulk, color="crimson", ls="--", lw=1.2,
                label=rf"$\rho_{{bulk}}$ = {rho_bulk:.4f}")
    ax0.axvline(np.mean(pool_dens), color="navy", ls=":", lw=1.0,
                label=f"Mean = {np.mean(pool_dens):.4f}")
    ax0.set_xlabel(r"Local density $\rho_\mathrm{local}$")
    ax0.set_ylabel("Probability density")
    ax0.set_title("Per-Particle Local Density Distribution")
    ax0.legend(fontsize=8)
    _add_averaging_note(ax0, True, len(frames))

    # Panel 1: number of neighbours distribution
    ax1 = fig.add_subplot(gs[1])
    ax1.hist(pool_nn, bins=range(int(pool_nn.max()) + 2),
             color="teal", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Number of neighbours within r_max")
    ax1.set_ylabel("Count")
    ax1.set_title("Neighbour Count Distribution")
    _add_averaging_note(ax1, True, len(frames))

    # Panel 2: 2-D spatial density map (last frame)
    ax2 = fig.add_subplot(gs[2])
    sc = ax2.scatter(
        last_pos[:, 0], last_pos[:, 1],
        c=last_dens, cmap=cfg["colormap"],
        s=10, linewidths=0, alpha=0.9,
    )
    add_colorbar(ax2, sc, r"$\rho_\mathrm{local}$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(f"Local Density Map  (frame {frames[-1]})")
    ax2.set_aspect("equal")
    _add_averaging_note(ax2, False, 1)

    fig.suptitle("freud.density.LocalDensity", fontsize=13, y=1.02)
    fpath = save_fig(fig, out, "18_local_density", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    return {
        "local_density_mean"  : float(np.mean(pool_dens)),
        "local_density_std"   : float(np.std(pool_dens)),
        "local_density_min"   : float(np.min(pool_dens)),
        "local_density_max"   : float(np.max(pool_dens)),
        "local_density_bulk"  : float(rho_bulk),
        "local_density_mean_nn": float(np.mean(pool_nn)),
    }


# ─────────────────────────────────────────────────────────────────────────────

def run_gaussian_density(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Compute a Gaussian-smoothed density field on a voxel grid using
    ``freud.density.GaussianDensity``.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    Each particle is replaced by a Gaussian of width σ centred at its
    position.  The contributions are summed on a regular grid, giving a
    continuous density field ρ(r):

        ρ(r) = Σᵢ G_σ(r - rᵢ)  where  G_σ(x) = exp(−|x|²/2σ²)

    This is equivalent to the "kernel density estimate" or "electron
    density" representation of the particle system.

    Uses:
    *  Visualise density heterogeneity (high-density clusters vs. voids)
    *  Detect phase separation or microphase ordering
    *  Compute density profiles along any axis
    *  Input to further image-processing (peak finding, watershed)

    σ is the Gaussian broadening width.  Choose σ ≈ σ_particle/4 to
    preserve individual particle structure, or σ ≈ L/20 to show
    large-scale density fluctuations.

    Frame averaging: **NO (last frame by default)**
    The density field is a spatial snapshot.  Set
    ``"frame_average_override": true`` to accumulate over all selected
    frames, which averages out thermal fluctuations but blurs fast dynamics.

    Returns
    -------
    dict with max, mean, and std of the density field
    """
    gd_cfg   = cfg["gaussian_density"]
    width    = gd_cfg["width"]       # int or list[int]
    r_max    = float(gd_cfg["r_max"])
    sigma    = float(gd_cfg["sigma"])
    force_avg = bool(gd_cfg.get("frame_average_override", False))

    active_frames = frames if force_avg else [frames[-1]]
    log.info("── GaussianDensity  (width=%s, r_max=%.2f, σ=%.3f, frames=%d) ─",
             width, r_max, sigma, len(active_frames))

    try:
        gd = freud.density.GaussianDensity(width=width, r_max=r_max, sigma=sigma)
    except Exception as exc:
        log.error("  GaussianDensity init failed: %s", exc)
        log.debug(traceback.format_exc())
        return {}

    first = True
    density_accum: Optional[np.ndarray] = None
    n_accum = 0

    for fi in active_frames:
        fd     = extract_frame_data(traj[fi])
        system = freud.locality.AABBQuery(fd["box"], fd["positions"])
        try:
            gd.compute(system)
            field = np.asarray(gd.density, dtype=np.float64)
            if density_accum is None:
                density_accum = field.copy()
            else:
                density_accum += field
            n_accum += 1
            first = False
        except Exception as exc:
            log.warning("  GaussianDensity frame %d: %s", fi, exc)
            log.debug(traceback.format_exc())

    if first or density_accum is None:
        log.error("  GaussianDensity: all frames failed.")
        return {}

    field = density_accum / n_accum  # mean field if frame_average_override

    # ── Visualise: slice or project depending on dimensionality ──────────
    apply_style()
    ndim = field.ndim   # 2 for 2-D box, 3 for 3-D box

    if ndim == 2:
        # 2-D: plot directly as imshow
        fig, ax = plt.subplots(figsize=cfg["figure_size"])
        im = ax.imshow(
            field.T,
            origin="lower",
            aspect="equal",
            cmap=cfg["colormap"],
        )
        add_colorbar(ax, im, r"$\rho(\mathbf{r})$")
        ax.set_xlabel("x bin")
        ax.set_ylabel("y bin")
        ax.set_title(
            f"Gaussian Density Field (2-D)  –  "
            + ("avg over frames" if force_avg else f"frame {active_frames[-1]}")
        )
        _add_averaging_note(ax, force_avg, n_accum)
    else:
        # 3-D: show three orthogonal 2-D slices through the centre
        nx, ny, nz = field.shape
        fig, axes  = plt.subplots(1, 3, figsize=(15, 5))

        slice_data = [
            (field[nx // 2, :, :].T, "yz-plane  (x = mid)", "y bin", "z bin"),
            (field[:, ny // 2, :].T, "xz-plane  (y = mid)", "x bin", "z bin"),
            (field[:, :, nz // 2].T, "xy-plane  (z = mid)", "x bin", "y bin"),
        ]
        vmin = field.min(); vmax = field.max()
        for ax_s, (sl, title, xlabel, ylabel) in zip(axes, slice_data):
            im = ax_s.imshow(sl, origin="lower", aspect="equal",
                             cmap=cfg["colormap"], vmin=vmin, vmax=vmax)
            ax_s.set_title(title)
            ax_s.set_xlabel(xlabel)
            ax_s.set_ylabel(ylabel)
        fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04,
                     label=r"$\rho(\mathbf{r})$")
        frame_info = "avg over frames" if force_avg else f"frame {active_frames[-1]}"
        fig.suptitle(
            f"freud.density.GaussianDensity  –  3-D orthogonal slices  ({frame_info})",
            fontsize=12, y=1.01,
        )

    fig.tight_layout()
    fpath = save_fig(fig, out, "19_gaussian_density", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    return {
        "gaussian_density_max"  : float(np.max(field)),
        "gaussian_density_mean" : float(np.mean(field)),
        "gaussian_density_std"  : float(np.std(field)),
        "gaussian_density_frame": int(active_frames[-1]),
    }


# ─────────────────────────────────────────────────────────────────────────────

def run_sphere_voxelization(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Compute a binary voxel occupancy grid using
    ``freud.density.SphereVoxelization``.

    Physical meaning
    ~~~~~~~~~~~~~~~~
    A sphere of radius r_max is placed around each particle.  A voxel is
    set to 1 if its centre lies inside any sphere, and 0 otherwise.

    The result is a binary grid encoding "is space occupied by a particle?"

    Practical uses:
    *  Packing fraction φ = N_occupied / N_total  (actual occupied volume)
    *  Void detection – voxels with value 0 indicate empty space
    *  Percolation analysis – does the occupied region form a connected path?
    *  Visualise the 3-D arrangement of particles and their excluded volumes
    *  Compare with theoretical packing limits (φ_FCC = π/(3√2) ≈ 0.7405)

    Frame averaging: **NO (last frame only)**
    Voxel occupancy is a binary spatial snapshot.

    Returns
    -------
    dict with packing fraction, voxel grid shape, occupied/total voxel counts
    """
    sv_cfg = cfg["sphere_voxelization"]
    width  = sv_cfg["width"]
    r_max  = float(sv_cfg["r_max"])
    fi     = frames[-1]

    log.info("── SphereVoxelization  (width=%s, r_max=%.3f, frame %d) ──────",
             width, r_max, fi)

    fd     = extract_frame_data(traj[fi])
    system = (fd["box"], fd["positions"])

    try:
        sv = freud.density.SphereVoxelization(width=width, r_max=r_max)
        sv.compute(system)
    except Exception as exc:
        log.error("  SphereVoxelization failed: %s", exc)
        log.debug(traceback.format_exc())
        return {}

    voxels       = np.asarray(sv.voxels)
    n_total      = voxels.size
    n_occupied   = int(np.sum(voxels))
    packing_frac = float(n_occupied / n_total) if n_total > 0 else 0.0
    ndim         = voxels.ndim

    log.info("  → Packing fraction φ = %.4f  (%d / %d voxels occupied)",
             packing_frac, n_occupied, n_total)

    # ── Figure ─────────────────────────────────────────────────────────────
    apply_style()

    if ndim == 2:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
        ax0.imshow(voxels.T, origin="lower", aspect="equal",
                   cmap="binary", interpolation="nearest")
        ax0.set_title(f"Sphere Voxelization (2-D)  –  frame {fi}")
        ax0.set_xlabel("x bin"); ax0.set_ylabel("y bin")
        _add_averaging_note(ax0, False, 1)

        # Row-sum density profile (x-projection)
        profile = np.sum(voxels, axis=1) / voxels.shape[1]
        ax1.plot(np.arange(len(profile)), profile, color="steelblue", lw=1.5)
        ax1.set_xlabel("x bin")
        ax1.set_ylabel("Occupied fraction")
        ax1.set_title("x-Direction Density Profile")

    else:
        # 3-D: show three orthogonal slices
        nx, ny, nz = voxels.shape
        fig, axes  = plt.subplots(1, 3, figsize=(15, 5))
        slice_configs = [
            (voxels[nx // 2, :, :].T, "yz-plane (x=mid)", "y", "z"),
            (voxels[:, ny // 2, :].T, "xz-plane (y=mid)", "x", "z"),
            (voxels[:, :, nz // 2].T, "xy-plane (z=mid)", "x", "y"),
        ]
        for axi, (sl, title, xl, yl) in zip(axes, slice_configs):
            axi.imshow(sl, origin="lower", aspect="equal",
                       cmap="binary", interpolation="nearest")
            axi.set_title(title); axi.set_xlabel(xl + " bin"); axi.set_ylabel(yl + " bin")
        _add_averaging_note(axes[0], False, 1)
        fig.suptitle(
            f"freud.density.SphereVoxelization  –  frame {fi}"
            f"\nφ = {packing_frac:.4f}  ({n_occupied}/{n_total} voxels)",
            fontsize=12, y=1.01,
        )

    fig.tight_layout()
    fpath = save_fig(fig, out, "20_sphere_voxelization", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    return {
        "sphere_vox_packing_fraction" : packing_frac,
        "sphere_vox_n_occupied"       : n_occupied,
        "sphere_vox_n_total"          : n_total,
        "sphere_vox_shape"            : list(voxels.shape),
        "sphere_vox_frame"            : int(fi),
    }


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
# SECTION C  ──  FRAME_AVG_SUPPORT and ANALYSIS_DISPLAY_NAMES additions
# =============================================================================
#
# Add these entries to BOTH dicts in analyze_environment.py:
#
#   FRAME_AVG_SUPPORT additions:
#     "rdf"                  : True,
#     "correlation_function" : True,
#     "local_density"        : True,
#     "gaussian_density"     : False,   # default last frame; override available
#     "sphere_voxelization"  : False,   # last frame only
#
#   ANALYSIS_DISPLAY_NAMES additions:
#     "rdf"                  : "RDF  (g(r))",
#     "correlation_function" : "CorrelationFunction  (C(r))",
#     "local_density"        : "LocalDensity",
#     "gaussian_density"     : "GaussianDensity",
#     "sphere_voxelization"  : "SphereVoxelization",

FRAME_AVG_SUPPORT = {
    "rdf"                  : True,
    "correlation_function" : True,
    "local_density"        : True,
    "gaussian_density"     : False,
    "sphere_voxelization"  : False,
}

ANALYSIS_DISPLAY_NAMES = {
    "rdf"                  : "RDF  (g(r))",
    "correlation_function" : "CorrelationFunction  (C(r))",
    "local_density"        : "LocalDensity",
    "gaussian_density"     : "GaussianDensity",
    "sphere_voxelization"  : "SphereVoxelization",
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
    log.info("   FREUD  DENSITY  ANALYSIS  –  RESULTS  SUMMARY")
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



    # ── 16. RDF ────────────────────────────────────────────────────────────
    if cfg.get("rdf", {}).get("enabled"):
        _run("rdf", run_rdf, avg_frames)

    # ── 17. CorrelationFunction ────────────────────────────────────────────
    if cfg.get("correlation_function", {}).get("enabled"):
        _run("correlation_function", run_correlation_function, avg_frames)

    # ── 18. LocalDensity ───────────────────────────────────────────────────
    if cfg.get("local_density", {}).get("enabled"):
        _run("local_density", run_local_density, avg_frames)

    # ── 19. GaussianDensity  (last frame or override) ──────────────────────
    if cfg.get("gaussian_density", {}).get("enabled"):
        gd_frames = (
            avg_frames
            if cfg["gaussian_density"].get("frame_average_override", False)
            else snap_frames
        )
        _run("gaussian_density", run_gaussian_density, gd_frames)

    # ── 20. SphereVoxelization  (last frame only) ──────────────────────────
    if cfg.get("sphere_voxelization", {}).get("enabled"):
        _run("sphere_voxelization", run_sphere_voxelization, snap_frames)



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
        prog        = "analyze_density.py",
        description = "freud.density analysis pipeline (all five analyses)",
        epilog      = (
            "Example:\n"
            "  python analyze_density.py params_density.json\n\n"
            "Edit params_density.json to enable/disable individual analyses and\n"
            "adjust all parameters without touching the source code."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs   = "?",
        default = "params_density.json",
        help    = "Path to the JSON parameter file  (default: params_density.json)",
    )
    args = parser.parse_args()
    main(args.config)