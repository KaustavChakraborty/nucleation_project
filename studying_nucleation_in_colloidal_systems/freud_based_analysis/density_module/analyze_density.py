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

    # ── frame selection from the END of the trajectory ──────────────────────
    # frame_average : false → analyse only the final trajectory frame
    #                 true  → analyse the last num_frames frames, separated by
    #                         frame_step, always including the final frame
    "frame_average": False,
    "num_frames"  : 30,
    "frame_step"  : 1,

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
        "r_cut"             : None,  # must be supplied explicitly in user JSON
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
    frame_average: bool,
    num_frames: int,
    step: int,
    logger: logging.Logger,
) -> List[int]:
    """
    Select frames relative to the END of the trajectory.

    Rules
    -----
    * frame_average=False:
        Return only the final frame, [N_total - 1].
    * frame_average=True:
        Return up to ``num_frames`` frame indices, separated by ``step``,
        counting backward from the final frame and then reordered into
        chronological order.

    Example
    -------
    For a trajectory containing N_total frames, ``num_frames=6`` and
    ``step=1`` select the final six zero-based indices:

        [N_total-6, N_total-5, ..., N_total-1]

    The returned order is chronological so that ``indices[-1]`` is always
    the actual final trajectory frame.
    """
    n_total = len(traj)

    if n_total <= 0:
        raise ValueError("Cannot select frames from an empty trajectory.")

    if step <= 0:
        raise ValueError(f"frame_step must be a positive integer; received {step}.")

    last_index = n_total - 1

    if not frame_average:
        indices = [last_index]
        logger.info(
            "Frame averaging disabled: using only the final frame index %d.",
            last_index,
        )
        return indices

    if num_frames <= 0:
        raise ValueError(
            f"num_frames must be a positive integer when frame_average=True; "
            f"received {num_frames}."
        )

    # Maximum number of frames reachable while stepping backward from the
    # final frame without producing a negative index.
    max_available = last_index // step + 1
    n_use = min(num_frames, max_available)

    # Construct from newest to oldest, then reverse so downstream code sees
    # chronological order and frames[-1] remains the newest/final frame.
    newest_to_oldest = [last_index - k * step for k in range(n_use)]
    indices = list(reversed(newest_to_oldest))

    if n_use < num_frames:
        logger.warning(
            "Requested %d frame(s) from the trajectory end with step=%d, "
            "but only %d frame(s) are available. Using all available frames.",
            num_frames,
            step,
            n_use,
        )

    logger.info(
        "Frame averaging enabled: selected %d frame(s) from the trajectory end "
        "with step=%d.",
        len(indices),
        step,
    )
    if len(indices) <= 20:
        logger.info("  Selected zero-based frame indices: %s", indices)
    else:
        logger.info(
            "  Selected zero-based frame indices: %d ... %d  (total=%d)",
            indices[0],
            indices[-1],
            len(indices),
        )

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
    plt.show()
    plt.close(fig)
    return fpath


def add_colorbar(ax, mappable, label: str):
    """Attach a labelled colorbar to *ax*."""
    cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label)
    return cbar



def _integrate_rdf_coordination_number(
    bin_edges: np.ndarray,
    gr: np.ndarray,
    r_cut: float,
    number_density: float,
    is_2d: bool,
) -> float:
    """
    Integrate the RDF histogram from its configured r_min up to r_cut.

    The RDF is treated as piecewise constant inside each histogram bin. The
    shell measure is integrated exactly, including a partial final bin when
    r_cut lies inside a bin.

    3-D:
        N_c = rho * integral[4*pi*r^2*g(r) dr]

    2-D:
        N_c = rho * integral[2*pi*r*g(r) dr]
    """
    edges = np.asarray(bin_edges, dtype=np.float64)
    values = np.asarray(gr, dtype=np.float64)

    if edges.ndim != 1 or values.ndim != 1:
        raise ValueError("RDF bin_edges and rdf arrays must both be one-dimensional.")
    if edges.size != values.size + 1:
        raise ValueError(
            "RDF bin_edges must contain exactly one more element than rdf values."
        )
    if not np.all(np.diff(edges) > 0):
        raise ValueError("RDF bin edges must be strictly increasing.")
    if number_density <= 0 or not np.isfinite(number_density):
        raise ValueError(
            f"Number density must be finite and positive; received {number_density}."
        )

    lower = edges[:-1]
    upper = np.minimum(edges[1:], r_cut)
    mask = (lower < r_cut) & (upper > lower)

    if not np.any(mask):
        return 0.0

    if not np.all(np.isfinite(values[mask])):
        raise ValueError(
            "RDF contains non-finite g(r) values inside the requested integration range."
        )

    if is_2d:
        shell_measure = np.pi * (upper[mask] ** 2 - lower[mask] ** 2)
    else:
        shell_measure = (4.0 * np.pi / 3.0) * (
            upper[mask] ** 3 - lower[mask] ** 3
        )

    return float(number_density * np.sum(values[mask] * shell_measure))


def run_rdf(
    traj,
    frames: List[int],
    cfg: Dict[str, Any],
    out: Path,
    log: logging.Logger,
) -> Dict[str, Any]:
    """
    Compute g(r), integrate it up to the user-supplied r_cut, display the RDF
    interactively, and save the figure after the interactive window closes.

    No RDF peak or minimum is selected automatically. The coordination number
    is calculated only from the explicit r_cut supplied in cfg["rdf"].
    """
    # ------------------------------------------------------------------
    # Step 1: Read and validate configuration.
    # ------------------------------------------------------------------
    try:
        rdf_cfg = cfg["rdf"]
        bins = int(rdf_cfg["bins"])
        r_max = float(rdf_cfg["r_max"])
        r_min = float(rdf_cfg["r_min"])
        r_cut_raw = rdf_cfg["r_cut"]
        norm_mode = str(rdf_cfg["normalization_mode"]).strip()
    except KeyError as exc:
        log.error(
            "RDF configuration is incomplete. Missing key: %s. Required keys "
            "are bins, r_min, r_max, r_cut, and normalization_mode.",
            exc,
        )
        log.debug(traceback.format_exc())
        return {}
    except (TypeError, ValueError) as exc:
        log.error("RDF configuration contains invalid value type(s): %s", exc)
        log.debug(traceback.format_exc())
        return {}

    if r_cut_raw is None:
        log.error(
            "RDF config error: 'r_cut' must be supplied explicitly as a number."
        )
        return {}

    try:
        r_cut = float(r_cut_raw)
    except (TypeError, ValueError) as exc:
        log.error("RDF config error: 'r_cut' must be numeric. Error: %s", exc)
        return {}

    if bins <= 0:
        log.error("RDF config error: 'bins' must be > 0, but got %d.", bins)
        return {}
    if r_min < 0:
        log.error("RDF config error: 'r_min' must be >= 0, but got %.6f.", r_min)
        return {}
    if r_max <= r_min:
        log.error(
            "RDF config error: require r_max > r_min, but received "
            "r_min=%.6f and r_max=%.6f.",
            r_min,
            r_max,
        )
        return {}
    if not (r_min < r_cut <= r_max):
        log.error(
            "RDF config error: r_cut must satisfy r_min < r_cut <= r_max. "
            "Received r_min=%.6f, r_cut=%.6f, r_max=%.6f.",
            r_min,
            r_cut,
            r_max,
        )
        return {}

    if norm_mode != "exact":
        log.error(
            "RDF coordination integration requires normalization_mode='exact'. "
            "The 'finite_size' mode rescales g(r) by N/(N-1), which would "
            "rescale the integrated coordination number as well. Received '%s'.",
            norm_mode,
        )
        return {}

    if not frames:
        log.error("RDF aborted: no frames were supplied.")
        return {}

    if r_min > 0:
        log.warning(
            "RDF coordination integration will begin at r_min=%.6f, not at zero. "
            "Any contribution from 0 <= r < r_min is intentionally omitted.",
            r_min,
        )

    log.info(
        "── RDF (bins=%d, r_min=%.4f, r_cut=%.4f, r_max=%.4f, "
        "norm=%s, requested_frames=%d) ──",
        bins,
        r_min,
        r_cut,
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
        log.error("RDF initialization failed: %s", exc)
        log.debug(traceback.format_exc())
        return {}

    # ------------------------------------------------------------------
    # Step 3: Accumulate RDF statistics and record frame densities.
    # ------------------------------------------------------------------
    first_successful_compute = True
    n_ok = 0
    n_skipped = 0
    skipped_frames: List[int] = []
    number_densities: List[float] = []
    dimensionality_flags: List[bool] = []

    for fi in frames:
        try:
            fd = extract_frame_data(traj[fi])
            system = (fd["box"], fd["positions"])
            rdf.compute(system, reset=first_successful_compute)

            first_successful_compute = False
            n_ok += 1
            number_densities.append(float(fd["N"] / fd["box"].volume))
            dimensionality_flags.append(bool(fd["box"].is2D))
        except Exception as exc:
            log.warning("RDF frame %d skipped: %s", fi, exc)
            log.debug(traceback.format_exc())
            n_skipped += 1
            skipped_frames.append(fi)

    if n_ok == 0:
        log.error(
            "RDF failed: all %d requested frame(s) were skipped. Skipped=%s",
            len(frames),
            skipped_frames,
        )
        return {}

    if len(set(dimensionality_flags)) != 1:
        log.error(
            "RDF coordination integration aborted because successful frames mix "
            "2-D and 3-D simulation boxes."
        )
        return {}

    is_2d = dimensionality_flags[0]
    rho_mean = float(np.mean(number_densities))
    rho_min = float(np.min(number_densities))
    rho_max = float(np.max(number_densities))

    if n_skipped:
        log.warning(
            "RDF completed with partial success: %d frame(s) used, %d skipped.",
            n_ok,
            n_skipped,
        )

    if n_ok > 1 and not np.allclose(
        number_densities,
        rho_mean,
        rtol=1.0e-6,
        atol=1.0e-12,
    ):
        log.warning(
            "Number density varies across the selected frames: min=%.8g, "
            "mean=%.8g, max=%.8g. Coordination is integrated using the mean "
            "number density.",
            rho_min,
            rho_mean,
            rho_max,
        )

    # ------------------------------------------------------------------
    # Step 4: Extract g(r) and integrate to the explicit r_cut.
    # ------------------------------------------------------------------
    try:
        r = np.asarray(rdf.bin_centers, dtype=np.float64)
        bin_edges = np.asarray(rdf.bin_edges, dtype=np.float64)
        gr = np.asarray(rdf.rdf, dtype=np.float64)
    except Exception as exc:
        log.error("Could not extract RDF arrays from freud: %s", exc)
        log.debug(traceback.format_exc())
        return {}

    if r.size == 0 or gr.size == 0 or bin_edges.size == 0:
        log.error("RDF produced one or more empty output arrays.")
        return {}

    try:
        coordination_number = _integrate_rdf_coordination_number(
            bin_edges=bin_edges,
            gr=gr,
            r_cut=r_cut,
            number_density=rho_mean,
            is_2d=is_2d,
        )
    except Exception as exc:
        log.error("RDF coordination-number integration failed: %s", exc)
        log.debug(traceback.format_exc())
        return {}

    summary: Dict[str, Any] = {
        "rdf_coordination_number": float(coordination_number),
        "rdf_r_cut": float(r_cut),
        "rdf_number_density_mean": float(rho_mean),
        "rdf_dimension": 2 if is_2d else 3,
        "rdf_n_frames_used": int(n_ok),
        "rdf_n_frames_skipped": int(n_skipped),
        "rdf_r_min": float(r_min),
        "rdf_r_max": float(r_max),
        "rdf_gr_max": float(np.nanmax(gr)),
        "rdf_gr_min": float(np.nanmin(gr)),
    }

    # Log a plain-text copy and print a bold terminal result.
    log.info(
        "Coordination number integrated from r=%.6f to r_cut=%.6f: %.8f",
        r_min,
        r_cut,
        coordination_number,
    )
    separator = "=" * 84
    bold_message = (
        f"COORDINATION NUMBER  [r_min={r_min:.6f}, r_cut={r_cut:.6f}]  "
        f"= {coordination_number:.8f}"
    )
    print("\n" + separator)
    print(f"\033[1m{bold_message}\033[0m")
    print(separator + "\n")

    # ------------------------------------------------------------------
    # Step 5: Display the RDF interactively first, then save it.
    # ------------------------------------------------------------------
    try:
        apply_style()
        fig, ax = plt.subplots(figsize=cfg["figure_size"])
        ax.plot(r, gr, lw=1.8, label=r"$g(r)$")
        ax.axhline(1.0, ls=":", lw=0.9, label=r"$g(r)=1$")
        ax.axvline(
            r_cut,
            ls="--",
            lw=1.2,
            label=rf"$r_{{cut}}={r_cut:.4f}$",
        )
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$g(r)$")
        ax.set_title("Radial Distribution Function")
        ax.set_xlim(r_min, r_max)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9)
        _add_averaging_note(ax, n_ok)
        fig.tight_layout()

        out.mkdir(parents=True, exist_ok=True)
        fpath = out / "16_rdf.png"

        backend = str(matplotlib.get_backend()).lower()
        noninteractive_backends = {"agg", "pdf", "ps", "svg", "cairo", "template", "pgf"}
        if backend in noninteractive_backends:
            log.warning(
                "Matplotlib backend '%s' is non-interactive; the RDF window "
                "cannot be displayed. Saving the figure directly.",
                matplotlib.get_backend(),
            )
        else:
            log.info(
                "Displaying the interactive RDF plot. Close the plot window to "
                "continue; the PNG will then be saved."
            )
            plt.show(block=True)

        # Figure.savefig remains valid after blocking show because we retain the
        # explicit Figure object in 'fig'.
        fig.savefig(fpath, dpi=cfg["dpi"], bbox_inches="tight")
        plt.close(fig)
        log.info("Saved RDF figure => %s", fpath)

    except Exception as exc:
        log.error(
            "RDF plotting/saving failed, but numerical RDF and coordination "
            "results were computed successfully. Error: %s",
            exc,
        )
        log.debug(traceback.format_exc())

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
    n_ok = 0
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
            n_ok += 1
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
    _add_averaging_note(ax0, n_ok)

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
    _add_averaging_note(ax1, n_ok)

    fig.suptitle("freud.density.CorrelationFunction", fontsize=13, y=1.01)
    fig.tight_layout()
    fpath = save_fig(fig, out, "17_correlation_function", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    summary: Dict[str, Any] = {
        "cf_C_r_min"  : float(np.nanmin(C_r)),
        "cf_C_r_max"  : float(np.nanmax(C_r)),
        "cf_value_mode"   : vmode,
        "cf_symmetry_k"   : sym_k,
        "cf_n_frames_used": int(n_ok),
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
    successful_frames: List[int] = []

    for fi in frames:
        fd     = extract_frame_data(traj[fi])
        system = (fd["box"], fd["positions"])
        try:
            ld.compute(system)
            all_densities.append(np.asarray(ld.density, dtype=np.float64))
            all_nneighbors.append(np.asarray(ld.num_neighbors, dtype=np.float64))
            successful_frames.append(fi)
        except Exception as exc:
            log.warning("  LocalDensity frame %d: %s", fi, exc)
            log.debug(traceback.format_exc())

    if not all_densities:
        log.error("  LocalDensity: all frames failed.")
        return {}

    # Pool all successfully processed frames for the distributions; keep the
    # spatial map from the newest successfully processed frame.
    n_density_frames = len(all_densities)
    pool_dens  = np.concatenate(all_densities)
    pool_nn    = np.concatenate(all_nneighbors)
    last_dens  = all_densities[-1]

    # Use the newest frame whose LocalDensity computation actually succeeded,
    # ensuring that the plotted positions and density values correspond.
    last_successful_frame = successful_frames[-1]
    fd_last    = extract_frame_data(traj[last_successful_frame])
    last_pos   = fd_last["positions"]

    # Bulk density from the same newest successful frame.
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
    _add_averaging_note(ax0, n_density_frames)

    # Panel 1: number of neighbours distribution
    ax1 = fig.add_subplot(gs[1])
    ax1.hist(pool_nn, bins=range(int(pool_nn.max()) + 2),
             color="teal", alpha=0.8, edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Number of neighbours within r_max")
    ax1.set_ylabel("Count")
    ax1.set_title("Neighbour Count Distribution")
    _add_averaging_note(ax1, n_density_frames)

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
    ax2.set_title(f"Local Density Map  (frame {last_successful_frame})")
    ax2.set_aspect("equal")
    _add_averaging_note(ax2, 1)

    fig.suptitle("freud.density.LocalDensity", fontsize=13, y=1.02)
    fpath = save_fig(fig, out, "18_local_density", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    return {
        "local_density_mean"  : float(np.mean(pool_dens)),
        "local_density_std"   : float(np.std(pool_dens)),
        "local_density_min"   : float(np.min(pool_dens)),
        "local_density_max"   : float(np.max(pool_dens)),
        "local_density_bulk"  : float(rho_bulk),
        "local_density_mean_nn"     : float(np.mean(pool_nn)),
        "local_density_n_frames_used"   : int(n_density_frames),
        "local_density_map_frame"       : int(last_successful_frame),
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
        frame_info = (
            f"average over {n_accum} frames"
            if n_accum > 1
            else f"frame {active_frames[-1]}"
        )
        ax.set_title(f"Gaussian Density Field (2-D)  –  {frame_info}")
        _add_averaging_note(ax, n_accum)
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
        frame_info = (
            f"average over {n_accum} frames"
            if n_accum > 1
            else f"frame {active_frames[-1]}"
        )
        fig.suptitle(
            f"freud.density.GaussianDensity  –  3-D orthogonal slices  ({frame_info})",
            fontsize=12, y=1.01,
        )
        _add_averaging_note(axes[0], n_accum)

    fig.tight_layout()
    fpath = save_fig(fig, out, "19_gaussian_density", cfg["dpi"])
    log.info("  Saved → %s", fpath)

    return {
        "gaussian_density_max"  : float(np.max(field)),
        "gaussian_density_mean" : float(np.mean(field)),
        "gaussian_density_std"  : float(np.std(field)),
        "gaussian_density_frame"        : int(active_frames[-1]),
        "gaussian_density_n_frames_used": int(n_accum),
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
        _add_averaging_note(ax0, 1)

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
        _add_averaging_note(axes[0], 1)
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

def _add_averaging_note(ax, n_frames: int):
    """
    Annotate a plot with the frame mode actually used to create that panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes receiving the annotation.
    n_frames : int
        Number of successfully processed frames represented by the panel.
        A value of 1 is labelled ``Last frame only``; values greater than 1
        are labelled ``Frame-averaged (n=...)``.

    Notes
    -----
    The label is derived from the actual processed-frame count rather than
    directly from ``cfg["frame_average"]``. This remains correct when one or
    more requested frames fail during an analysis.
    """
    try:
        n_frames = int(n_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"n_frames must be an integer-like value; received {n_frames!r}."
        ) from exc

    if n_frames <= 0:
        raise ValueError(
            f"n_frames must be positive for a completed plot; received {n_frames}."
        )

    if n_frames > 1:
        label = f"Frame-averaged  (n={n_frames})"
        color = "#1a7f1a"
    else:
        label = "Last frame only"
        color = "#8b0000"

    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color=color,
        style="italic",
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",
            alpha=0.6,
            ec=color,
        ),
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
#     "gaussian_density"     : True,    # averaging available via override
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
    "gaussian_density"     : True,
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
        want_avg = bool(cfg["frame_average"])
        res = results.get(key, {})

        # Prefer the number of frames actually processed by the analysis.
        frame_count_keys = {
            "rdf"                 : "rdf_n_frames_used",
            "correlation_function": "cf_n_frames_used",
            "local_density"       : "local_density_n_frames_used",
            "gaussian_density"    : "gaussian_density_n_frames_used",
        }
        count_key = frame_count_keys.get(key)
        actual_n_frames = (
            int(res[count_key])
            if count_key is not None and count_key in res
            else None
        )

        # When an analysis failed before reporting its processed-frame count,
        # show the frame mode that was requested by the configuration.
        if actual_n_frames is None:
            if key == "sphere_voxelization":
                actual_n_frames = 1
            elif key == "gaussian_density":
                override = bool(
                    cfg.get("gaussian_density", {}).get(
                        "frame_average_override",
                        False,
                    )
                )
                actual_n_frames = n_frames if (want_avg and override) else 1
            elif supports_avg and want_avg:
                actual_n_frames = n_frames
            else:
                actual_n_frames = 1

        mode_str = (
            f"frame-avg ({actual_n_frames} frames)"
            if actual_n_frames > 1
            else "last frame only"
        )

        ok = bool(res)
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
    selected_frames: List[int],
    out: Path,
    log: logging.Logger,
):
    """Write a human-readable JSON summary of all scalar results."""
    payload = {
        "run_timestamp": datetime.now().isoformat(),
        "trajectory"   : cfg["trajectory"],
        "frame_average"        : cfg["frame_average"],
        "num_frames_requested"  : cfg["num_frames"],
        "frame_step"            : cfg["frame_step"],
        "selected_frame_indices": selected_frames,
        "results"               : results,
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

    traj = open_trajectory(cfg["trajectory"], log)

    # Select frames relative to the trajectory end. When frame_average=False,
    # resolve_frame_indices returns only the final frame. When True, it returns
    # the requested number of end-relative frames in chronological order.
    avg_frames = resolve_frame_indices(
        traj=traj,
        frame_average=bool(cfg["frame_average"]),
        num_frames=int(cfg["num_frames"]),
        step=int(cfg["frame_step"]),
        logger=log,
    )

    # Snapshot analyses must always receive the actual final selected frame.
    snap_frames = [avg_frames[-1]]

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
    save_summary_json(results, cfg, avg_frames, out_dir, log)

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