# EPD PyVista Polyhedron Visualizer (v2p1, Symmetry Axes)

**Script file:** `epd_pyvista_polyhedron_visualizer_v2p1_symmetry_axes.py`

A publication-quality 3D renderer for a single HPMC particle from a HOOMD
GSD trajectory. It reconstructs the particle as a **true closed polyhedron**
(real filled faces, not a scatter of vertex points), renders each selected
frame as a high-resolution PNG, optionally overlays the particle's
**geometrically-detected rotational symmetry axes** (one C5 axis + five C2
axes), and can stitch the PNGs into an MP4 video.

This README explains, in detail:
- exactly what the script does and how,
- every command-line parameter and what it controls,
- what to change to point it at **your own** trajectory / shape / frame
  selection,
- what you can only change by editing the source code (not exposed as a
  flag), and
- every error the script can raise, what causes it, and how to fix it.

---

## Table of Contents

1. [What This Script Actually Does](#1-what-this-script-actually-does)
2. [Installation / Requirements](#2-installation--requirements)
3. [Input Files You Need](#3-input-files-you-need)
4. [Quick Start](#4-quick-start)
5. [Full Command-Line Reference](#5-full-command-line-reference)
6. [Frame Selection In Detail](#6-frame-selection-in-detail)
7. [The Rotational Symmetry Axis Feature — Read This Before Using Another Shape](#7-the-rotational-symmetry-axis-feature--read-this-before-using-another-shape)
8. [Output Files Produced](#8-output-files-produced)
9. [Adapting This Script To Your Own Polyhedron and Trajectory (Step-by-Step)](#9-adapting-this-script-to-your-own-polyhedron-and-trajectory-step-by-step)
10. [Things You Can Only Change By Editing the Code](#10-things-you-can-only-change-by-editing-the-code)
11. [Complete Error Reference / Troubleshooting](#11-complete-error-reference--troubleshooting)
12. [Fully-Worked Example Commands](#12-fully-worked-example-commands)
13. [Performance Notes](#13-performance-notes)
14. [Quick-Reference Cheat Sheet](#14-quick-reference-cheat-sheet)

---

## 1. What This Script Actually Does

For the particle number you give it, the script performs this pipeline:

1. **Reads** the particle's position and quaternion orientation from every
   frame you select in the GSD trajectory.
2. **Obtains the body-frame vertices** of the polyhedron — first it tries to
   read them directly out of the GSD file (HPMC shape metadata); if that
   isn't available it falls back to reading them from your `shape_json`
   file.
3. **Reconstructs the true closed polyhedron surface** from those vertices
   using `scipy.spatial.ConvexHull`.
4. **Merges coplanar triangles back into real polygonal faces.** `ConvexHull`
   always returns triangles; this script groups triangles that share the
   same plane so a pentagonal or rectangular face is drawn as one clean
   polygon instead of a triangulated mess with fake diagonal edges.
5. **Rotates and translates** the reconstructed polyhedron into each
   selected frame using the particle's quaternion (and position, unless
   `--center` is used).
6. **Renders** filled polygon faces plus clean edges with anti-aliasing and
   three-point lighting, and saves one high-resolution PNG per frame.
7. **Detects the proper rotational symmetry axes** of the particle directly
   from its geometry (not hard-coded to x/y/z) and — if validation passes —
   draws the one **C5** axis and five **C2** axes as 3D tubes that rotate
   rigidly with the particle. **This step is strictly validated for D5h
   symmetry (see [Section 7](#7-the-rotational-symmetry-axis-feature--read-this-before-using-another-shape))
   and will raise an error for particles that are not D5h**, unless you
   disable it with `--no-symmetry-axes`.
8. **Optionally** adds vertex markers, an orientation-axes widget, a
   title, and assembles all the PNGs into an MP4 video.

The example shape supplied with this script,
`shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json`,
is the Elongated Pentagonal Dipyramid (Johnson solid J16), which has D5h
point-group symmetry — this is exactly what the symmetry-axis detector is
built and validated for.

---

## 2. Installation / Requirements

```bash
pip install numpy scipy matplotlib pyvista gsd imageio imageio-ffmpeg
```

| Package | Why it's needed |
|---|---|
| `numpy` | All numerical/array work |
| `scipy` | `ConvexHull` (face reconstruction), `Rotation` (quaternions), `linear_sum_assignment` (symmetry detection) |
| `matplotlib` | Only used for the `viridis` colormap when `--color-mode frame` is selected |
| `pyvista` | The 3D rendering engine — **required**, script exits immediately if missing |
| `gsd` | Reading the HOOMD `.gsd` trajectory file |
| `imageio` + `imageio-ffmpeg` | Only needed for `--create-video`. If missing, image rendering still works; only the video step is skipped (with a warning) |

**PyVista and SciPy are hard requirements** — the script calls
`raise SystemExit(...)` immediately at import time if either is missing, with
the exact `pip install` command printed for you.

**imageio is a soft requirement** — if you never use `--create-video`, you
technically don't need it installed.

You will also generally want a working `ffmpeg` binary on your system (or at
minimum `imageio-ffmpeg`, which ships its own) for the `libx264` video codec
used by the video-writing step.

---

## 3. Input Files You Need

### 3.1 The GSD trajectory file (positional argument `gsd_file`)

Any standard HOOMD-blue GSD trajectory containing HPMC rigid polyhedral
particles, with `particles.position` and `particles.orientation` present in
every frame you plan to render.

### 3.2 The shape JSON file (positional argument `shape_json`)

**This argument is always required on the command line**, even if the GSD
file already contains embedded shape information — the JSON is used purely
as a *fallback* (step 2 above). If the GSD has usable shape data, your JSON
file's contents are never actually read, but the file must still exist and
be a syntactically valid path or the argument parsing/fallback logic can
still fail later if the GSD lookup comes up empty.

The JSON must contain a 2D array of shape `(N_vertices, 3)` under **one** of
these keys (checked in this exact order):

```
"vertices"
"8_vertices"
"12_vertices"
"polyhedron_vertices"
```

The bundled example file uses the key `"8_vertices"` (a naming holdover from
another cataloguing convention) but actually contains **12** vertices — that
is fine, because the script only cares about the *key name*, not the number
of vertices inside it. Excerpt of the supplied EPD shape file:

```json
{
    "0_Id": "37",
    "1_Name": "Elongated Pentagonal Dipyramid",
    "2_ShortName": "J16",
    "4_volume": 1.0000000000000002,
    "8_vertices": [
        [0.7744378962411786, -9.972731702719086e-08, -1.2951643733570227e-17],
        [-0.7744378962411786, -9.972731702719086e-08, -1.2951643733570227e-17],
        ...
        12 vertices total ...
    ]
}
```

**Requirements on the vertex array** (validated by the script, see
[Section 11](#11-complete-error-reference--troubleshooting)):
- Shape must be `(N, 3)`.
- At least 4 vertices (minimum for a 3D polyhedron).
- No `NaN` or infinite values.
- The vertices must form a valid **convex** polyhedron (they are passed
  straight into `ConvexHull`).

---

## 4. Quick Start

Minimal command that will run with the bundled example EPD shape:

```bash
python3.8 epd_pyvista_polyhedron_visualizer_v2p1_symmetry_axes.py \
    hpmc_hard_Elongated_Pentagonal_Dipyramid_vol_1p00_4096_NPT_P60_0_traj.gsd \
    shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json \
    300 \
    0 \
    --center \
    --resolution 2400 \
    --camera-angle iso \
    --create-video \
    --video-fps 10
```

What this does:
- Opens the trajectory `hpmc_hard_..._traj.gsd`.
- Takes the **last 300 frames** of the trajectory (or fewer, if the
  trajectory has under 300 frames total — see [Section 6](#6-frame-selection-in-detail)).
- Renders **particle 0**.
- `--center` removes translational motion, so you see rotation only.
- Renders at **2400×2400** pixels, isometric camera.
- Builds an MP4 from all rendered frames at **10 frames per second**.

Everything is written into the default output directory, `polyhedron_frames/`
(unless you set `--output-dir`).

---

## 5. Full Command-Line Reference

### 5.1 Required positional arguments (must appear in this exact order)

| # | Name | Type | Meaning | Change this to... |
|---|---|---|---|---|
| 1 | `gsd_file` | path | HOOMD GSD trajectory file | **your own** `.gsd` trajectory |
| 2 | `shape_json` | path | JSON with body-frame polyhedron vertices (fallback, see §3.2) | **your own** shape JSON, using one of the four accepted keys |
| 3 | `n_frames` | int | Take this many consecutive frames from the **end** of the trajectory | how many recent frames you want rendered |
| 4 | `particle` | int | Particle **tag** if the GSD has tags, otherwise raw **array index** | the particle number you want to visualize |

> ⚠️ These four are positional — order matters and none can be skipped.
> Example: `... my_traj.gsd my_shape.json 150 42 [flags...]` means "150
> frames, particle 42."

### 5.2 Frame selection

| Flag | Default | Meaning |
|---|---|---|
| `--exclude "TEXT"` | `""` (none excluded) | Comma-separated **absolute GSD frame numbers** and/or ranges to drop from the selected window, e.g. `"710,715-720,730"` |
| `--interactive-selection` | off | After computing the frame window (and applying `--exclude`, if given), **interactively ask** in the terminal whether to drop more frames |

Full details and worked examples in [Section 6](#6-frame-selection-in-detail).

### 5.3 Output location, resolution, camera

| Flag | Default | Meaning |
|---|---|---|
| `--output-dir DIR` | `polyhedron_frames` | Folder where PNGs (and the MP4) are written; created automatically if missing |
| `--resolution N` | `2400` | Square image size in pixels (`N × N`). Must be ≥ 256 |
| `--camera-angle CHOICE` | `iso` | One of `iso`, `front`, `back`, `side`, `left`, `top`, `bottom` |
| `--camera-distance-factor F` | `5.0` | Camera distance as a multiple of the particle's (and, if drawn, the symmetry axes') bounding radius. Larger = camera further away = particle looks smaller |
| `--zoom F` | `0.9` | Final zoom applied after positioning the camera. `> 1` zooms in, `< 1` zooms out |

### 5.4 Appearance

| Flag | Default | Meaning |
|---|---|---|
| `--center` | off | Place the particle's center at the origin every frame (removes translation, keeps rotation) |
| `--show-vertices` | off | Draw small spheres at each true polyhedron vertex, on top of the solid faces |
| `--show-axes` | off | Draw a small RGB coordinate-axes orientation widget |
| `--no-title` | off | Suppress the "Particle N \| GSD frame M" text at the top of each image |
| `--transparent-background` | off | Save PNGs with a transparent (alpha) background instead of the solid `--background-color` |
| `--color-mode {solid,frame}` | `solid` | `solid` = every frame uses the same `--face-color`; `frame` = face color sweeps through the `viridis` colormap across the selected sequence (first frame = dark purple, last frame = yellow) |
| `--face-color HEX` | `#6FA8DC` (light blue) | Polyhedron face color, used when `--color-mode solid` |
| `--edge-color HEX` | `#202020` (near-black) | Color of the polyhedron's physical edges, and of `--show-vertices` spheres |
| `--background-color NAME/HEX` | `white` | Rendering background |
| `--edge-width F` | `3.0` | Rendered edge line width. Must be > 0 |
| `--opacity F` | `1.0` | Face opacity, `0`–`1` |

### 5.5 Rotational symmetry axis overlay

> ⚠️ **Read [Section 7](#7-the-rotational-symmetry-axis-feature--read-this-before-using-another-shape)
> before using this feature on any polyhedron other than a D5h shape like the
> supplied EPD.**

| Flag | Default | Meaning |
|---|---|---|
| `--no-symmetry-axes` | off (i.e. axes ARE drawn by default) | Disable the symmetry-axis detection and overlay entirely |
| `--symmetry-precision-exponent P` | `2` | Vertex-matching tolerance for symmetry detection is `10^(-P)` in the same length units as your shape JSON. Smaller `P` = looser tolerance; larger `P` = stricter |
| `--symmetry-axis-length-factor F` | `1.35` | Half-length of each drawn axis tube, as a multiple of the particle's max body-frame vertex radius |
| `--c5-axis-radius-factor F` | `0.018` | C5 tube thickness, as a fraction of the body radius |
| `--c2-axis-radius-factor F` | `0.012` | C2 tube thickness, as a fraction of the body radius |
| `--c5-axis-color HEX` | `#D62728` (red) | Color of the single C5 axis |
| `--c2-axis-color HEX` | `#2CA02C` (green) | Color of the five C2 axes |
| `--symmetry-axis-opacity F` | `1.0` | Opacity of the axis tubes, `0`–`1` |
| `--show-symmetry-legend` | off | Draw a small "C5 axis / C2 axes" color legend on each frame |

### 5.6 Video

| Flag | Default | Meaning |
|---|---|---|
| `--create-video` | off | After rendering all PNGs, assemble them into an MP4 |
| `--video-fps N` | `10` | **Playback speed control.** Frames per second of the output video. `video length (seconds) ≈ number_of_rendered_frames / video_fps` |

There is **no `--video-quality` flag** in this script version — encoding
quality is fixed in the source. See
[Section 10](#10-things-you-can-only-change-by-editing-the-code) if you want
to change it.

---

## 6. Frame Selection In Detail

The frame window is built in two stages, both of which operate on
**absolute GSD frame numbers** (i.e. the real index into the trajectory, not
a position within your selection):

**Stage A — the base window.**
```
start_frame = max(0, total_frames_in_gsd - n_frames)
frame_indices = start_frame, start_frame+1, ..., total_frames_in_gsd - 1
```
If you ask for more frames than the trajectory has, you simply get the whole
trajectory — this is not an error.

**Stage B — exclusions**, applied in this order:
1. If `--exclude` was given, those absolute frame numbers/ranges are removed
   from the Stage-A window first.
2. If `--interactive-selection` was also given, you are then prompted
   **again**, in the terminal, on the *already-reduced* list.

### 6.1 `--exclude` syntax

A comma-separated list of integers and/or `start-end` ranges:

```bash
--exclude "710,715-720,730"
```
This removes frame 710, frames 715 through 720 inclusive, and frame 730 —
6 frames total — from whatever window Stage A produced.

### 6.2 `--interactive-selection` walkthrough

If passed, after Stage A/B you will see:

```
==============================================================================
FRAME SELECTION
==============================================================================
Frames currently selected: 292
Absolute GSD frame range: 0 ... 299

Render all of these frames? [Y/n]:
```

- Press **Enter** or type `y`/`yes` → keep everything as-is.
- Type `n`/`no` → you are prompted again:
  ```
  Absolute frames to exclude (e.g. 710,715-720,730):
  ```
  Enter frame numbers/ranges exactly as in `--exclude` syntax (see §6.1).
  If you leave this empty, no additional frames are excluded.

### 6.3 Important: filenames use a *local*, not absolute, frame index

The rendered PNGs are named `frame_000000.png`, `frame_000001.png`, ... in
the **order the selected frames were rendered**, starting at 0 — **not**
the absolute GSD frame number. So if your selection is
`[100, 101, 103, 104]` (frame 102 excluded), you get:

```
frame_000000.png   ← absolute GSD frame 100
frame_000001.png   ← absolute GSD frame 101
frame_000002.png   ← absolute GSD frame 103
frame_000003.png   ← absolute GSD frame 104
```

The actual absolute frame number for each image is still shown in the
**on-image title** (e.g. `"Particle 0 | GSD frame 103"`), unless you passed
`--no-title`, and is printed to the terminal as frames render. If you need
the absolute number, don't suppress the title, or keep the terminal log.

---

## 7. The Rotational Symmetry Axis Feature — Read This Before Using Another Shape

This is the single most important thing to understand if you plan to reuse
this script on a **different polyhedron**.

By default (unless you pass `--no-symmetry-axes`), the script:

1. Generates a large set of candidate rotation-axis *lines* purely from your
   polyhedron's own geometry — vertex directions, face-centroid directions,
   face-normal directions, and edge-midpoint directions. **No axis is ever
   assumed to be x, y, or z.**
2. Tests each candidate axis against a large historical list of candidate
   rotation angles (72°, 90°, 120°, 180°, etc.).
3. For every candidate that maps the vertex set exactly back onto itself
   (checked with an optimal one-to-one vertex assignment, then refined to
   full numerical precision), it keeps that operation as a genuine proper
   rotational symmetry of the particle.
4. Groups all discovered non-identity rotations by shared axis line and
   classifies each group's rotational order (2-fold, 5-fold, etc.).
5. **Then it strictly validates** that what it found matches the D5h EPD's
   proper rotational subgroup, D5, exactly:
   - exactly **10** proper rotations total (including the identity),
   - exactly **1** distinct C5 axis line,
   - exactly **5** distinct C2 axis lines,
   - every C2 axis perpendicular to the C5 axis (within `1e-6`).

**If any one of those checks fails, the function raises `RuntimeError` and
the script stops — no PNGs are rendered at all.** This is intentional: the
script refuses to silently draw a wrong or partial symmetry-axis picture.

### What this means practically

| Your particle is... | What happens |
|---|---|
| The bundled Elongated Pentagonal Dipyramid (D5h) | Works out of the box, axes are drawn |
| A near-perfect D5h shape numerically | Works, possibly needs a looser `--symmetry-precision-exponent` (see below) |
| **Any other point group** (cube, tetrahedron, generic Johnson solid, an irregularly-shaped particle, etc.) | **`RuntimeError` — the script will not render anything unless you pass `--no-symmetry-axes`** |

### If you are visualizing a different polyhedron

**Always add `--no-symmetry-axes` to your command line** unless you have
specifically verified your shape is D5h and has exactly one C5 + five C2
proper rotation axes. Everything else in the script (face reconstruction,
rendering, video) is completely shape-agnostic and works for any convex
polyhedron.

```bash
python3.8 epd_pyvista_polyhedron_visualizer_v2p1_symmetry_axes.py \
    my_other_shape_traj.gsd \
    my_other_shape.json \
    200 \
    0 \
    --center \
    --no-symmetry-axes \
    --create-video --video-fps 12
```

### If you want symmetry axes for a *different* point group

This requires **editing the source code** — it is not exposed as a
command-line option. Look at:
- The constants near the top of the symmetry section:
  `D5H_EXPECTED_PROPER_ROTATIONS = 10`, `D5H_EXPECTED_C5_AXIS_LINES = 1`,
  `D5H_EXPECTED_C2_AXIS_LINES = 5`.
- The function `detect_d5h_rotational_axes(...)`, which contains the
  validation logic built around those constants.

You would need to change these expected counts (and generalize the
red/green C5/C2-only coloring and drawing logic in
`add_rotational_symmetry_axes(...)`) to match your target point group. This
is a non-trivial code change, not a flag.

### Tolerance tuning (`--symmetry-precision-exponent`)

The matching tolerance used during symmetry detection is an **absolute**
distance, `10^(-P)`, in the *same coordinate units as your shape JSON*. The
default `P=2` (tolerance `0.01`) is tuned for the supplied EPD JSON, whose
vertex coordinates are normalized to unit volume (roughly in the range
`-0.8` to `0.8`).

- If your shape JSON uses very different absolute units (e.g. vertex
  coordinates on the order of `100` because they're in nanometers, or on the
  order of `1e-6`), this default tolerance will be far too loose or far too
  tight relative to the shape's actual size. Increase `P` (stricter,
  smaller absolute tolerance) for large-coordinate shapes, or decrease `P`
  (looser) for very small-coordinate shapes, or better, renormalize your
  vertex coordinates to a sensible O(1) scale before use.
- If detection fails only barely (e.g. due to floating-point noise from how
  the vertices were generated), try lowering `--symmetry-precision-exponent`
  by 1 (looser tolerance) before assuming the shape is genuinely not D5h.

---

## 8. Output Files Produced

Given `--output-dir polyhedron_frames` (the default) and `--create-video`:

```
polyhedron_frames/
├── frame_000000.png
├── frame_000001.png
├── frame_000002.png
├── ...
├── frame_0000NN.png          (NN = number of selected frames − 1)
└── particle_<PARTICLE>_polyhedron.mp4      (only if --create-video)
```

- Every PNG is square, `--resolution × --resolution` pixels.
- Filenames use the **local** sequential index, zero-padded to 6 digits —
  see [§6.3](#63-important-filenames-use-a-local-not-absolute-frame-index)
  for how this maps back to absolute GSD frame numbers.
- The video filename embeds the particle number you requested, e.g.
  `particle_0_polyhedron.mp4`.
- If `--create-video` is used but `imageio`/`imageio-ffmpeg` isn't
  installed, or video encoding otherwise fails, the PNGs are still written
  successfully — only the video step is skipped, with a printed warning /
  error message (non-fatal).

---

## 9. Adapting This Script To Your Own Polyhedron and Trajectory (Step-by-Step)

1. **Get your shape's body-frame vertices** into a JSON file with one of the
   four accepted keys (`vertices`, `8_vertices`, `12_vertices`,
   `polyhedron_vertices`) mapping to an `(N, 3)` array — see
   [§3.2](#32-the-shape-json-file-positional-argument-shape_json). The
   vertices must describe a valid convex polyhedron.
2. **Point `gsd_file` at your own trajectory** — any HOOMD GSD with
   `particles.position` / `particles.orientation` for HPMC rigid bodies.
3. **Set `particle`** to the tag (if your GSD frames carry particle tags) or
   raw array index (0-based) of the specific particle you want to render.
4. **Choose `n_frames`** — how many frames, counted back from the end of the
   trajectory, form your base selection window.
5. **Decide on frame exclusion**, if any — use `--exclude "..."` for a
   scripted/reproducible exclusion list, and/or `--interactive-selection`
   if you want to eyeball and prune interactively each run.
6. **Add `--no-symmetry-axes`** unless you have confirmed your shape is D5h
   with exactly one C5 axis and five C2 axes (see
   [Section 7](#7-the-rotational-symmetry-axis-feature--read-this-before-using-another-shape)).
   This is the step people most often forget when reusing this script for a
   new shape, and it is the difference between "it renders fine" and "it
   crashes immediately with a RuntimeError".
7. **Pick your camera and appearance flags** — `--camera-angle`,
   `--resolution`, `--face-color`, `--background-color`, `--center`, etc.
   (see [§5.3](#53-output-location-resolution-camera) and
   [§5.4](#54-appearance)).
8. **Add `--create-video --video-fps N`** if you want an MP4 in addition to
   the PNG sequence. Larger `N` = faster-playing / shorter video for the
   same number of rendered frames; smaller `N` = slower-playing / longer
   video.
9. **Run it**, watch the terminal output (it prints file paths, particle
   lookup diagnostics, reconstructed-polyhedron statistics, and — if
   enabled — symmetry-detection diagnostics), and check
   `--output-dir` for the results.

### Minimal "new shape" template

```bash
python3.8 epd_pyvista_polyhedron_visualizer_v2p1_symmetry_axes.py \
    YOUR_TRAJECTORY.gsd \
    YOUR_SHAPE.json \
    NUMBER_OF_FRAMES \
    PARTICLE_INDEX \
    --center \
    --no-symmetry-axes \
    --output-dir YOUR_OUTPUT_DIR \
    --resolution 2400 \
    --camera-angle iso \
    --create-video \
    --video-fps 10
```

Replace the five UPPER-CASE tokens with your own values.

---

## 10. Things You Can Only Change By Editing the Code

Not everything is exposed as a command-line flag. If you need to change any
of the following, you must edit the `.py` file directly.

| What | Where in the code | Current fixed value |
|---|---|---|
| Video encoding quality / codec | `create_video_from_frames()`, the `imageio.get_writer(...)` call | `codec="libx264"`, `quality=9`, `pixelformat="yuv420p"` |
| Lighting rig (3-point light directions/intensities) | `add_scene_lighting()` | key/fill/rim lights at fixed relative directions and intensities `0.85 / 0.45 / 0.35` |
| Camera field-of-view angle | `configure_camera()` | `plotter.camera.view_angle = 28.0` |
| Coplanar-face merge tolerance (how "flat" triangles must be to merge into one polygon) | `reconstruct_polygon_faces()`, `plane_tolerance` parameter | `1.0e-7` |
| Vertex-marker sphere size (`--show-vertices`) | `render_frame()` | `radius * 0.022` of the particle's bounding radius |
| Vertex-marker sphere smoothness | `render_frame()`, `pv.Sphere(...)` | `theta_resolution=24, phi_resolution=24` |
| Title font size | `render_frame()`, `plotter.add_title(...)` | `font_size=18` |
| Accepted JSON vertex key names | `read_shape_from_json()`, `possible_keys` tuple | `"vertices"`, `"8_vertices"`, `"12_vertices"`, `"polyhedron_vertices"` |
| Symmetry point-group target (D5h/D5) | Constants `D5H_EXPECTED_PROPER_ROTATIONS`, `D5H_EXPECTED_C5_AXIS_LINES`, `D5H_EXPECTED_C2_AXIS_LINES`, and the body of `detect_d5h_rotational_axes()` | 10 / 1 / 5 |
| Candidate symmetry rotation angles tested | `historical_symmetry_candidate_angles_rad()` | a fixed list of common angles (multiples of 36°, 45°, 60°, 72°, 90°, 120°, etc.) |
| PNG output filename pattern | `main()`, `output_file = output_dir / f"frame_{local_index:06d}.png"` | 6-digit zero-padded local index |
| Video output filename pattern | `main()`, `video_file = output_dir / f"particle_{args.particle}_polyhedron.mp4"` | as shown |

---

## 11. Complete Error Reference / Troubleshooting

All errors below are real checks present in the script. They are listed in
roughly the order you're likely to hit them.

### Startup / dependency errors

| Error | Cause | Fix |
|---|---|---|
| `SystemExit: PyVista is required. ...` | `pyvista` not installed | `pip install pyvista` |
| `SystemExit: SciPy is required. ...` | `scipy` not installed | `pip install scipy` |
| *(silent, non-fatal)* `--create-video` produces no video, prints `imageio is unavailable; skipping video` | `imageio` not installed | `pip install imageio imageio-ffmpeg` |

### Argument-validation errors (raised in `main()` before anything is opened)

| Error message | Cause | Fix |
|---|---|---|
| `--resolution should be at least 256.` | `--resolution` set too low | Use `--resolution 256` or higher |
| `n_frames must be positive.` | `n_frames` was 0 or negative | Use a positive integer |
| `--video-fps must be positive.` | `--video-fps` was 0 or negative | Use a positive integer |
| `--edge-width must be positive.` | `--edge-width` was 0 or negative | Use a positive number |
| `--symmetry-precision-exponent must be nonnegative.` | Negative value passed | Use `0` or a positive integer |
| `--symmetry-axis-length-factor must be positive.` | Value ≤ 0 | Use a positive number |
| `--c5-axis-radius-factor must be positive.` | Value ≤ 0 | Use a positive number |
| `--c2-axis-radius-factor must be positive.` | Value ≤ 0 | Use a positive number |
| `--symmetry-axis-opacity must lie between 0 and 1.` | Out of range | Use a value in `[0, 1]` |

### File / trajectory errors

| Error message | Cause | Fix |
|---|---|---|
| `Could not open GSD file: ...` | Wrong path, corrupted file, or unreadable GSD | Double-check the path and that the file opens fine in a plain `gsd.hoomd.open()` |
| `The GSD contains no frames.` | Trajectory file is empty | Use a trajectory that actually has frames |
| `No frames remain after applying --exclude.` | Your `--exclude` list removed every frame in the selected window | Reduce the exclusion list, or increase `n_frames` |
| *(interactive)* `ValueError: All selected frames were excluded.` | You excluded everything when prompted under `--interactive-selection` | Answer with fewer exclusions next time |

### Shape / JSON errors

| Error message | Cause | Fix |
|---|---|---|
| `Shape JSON file does not exist: ...` | Wrong path to `shape_json` | Check the path/typo |
| `Could not find a vertex array in the JSON. Available keys: [...] Tried keys: (...)` | Your JSON doesn't use one of the four recognized keys | Rename your vertex key to `vertices` (simplest), or one of `8_vertices` / `12_vertices` / `polyhedron_vertices` |
| `... vertices must have shape (N, 3), found (...)` | Vertex array isn't a plain `Nx3` list of `[x, y, z]` | Fix the JSON structure |
| `... at least 4 vertices are required for a 3D polyhedron.` | Fewer than 4 vertices supplied | Provide a real 3D polyhedron |
| `... vertices contain NaN or infinity.` | Malformed numeric data in the JSON | Clean the vertex data |

### Particle-lookup errors

| Error message | Cause | Fix |
|---|---|---|
| `Particle tag <N> is not present in this frame.` | Your GSD frames have tags, but tag `<N>` doesn't exist in the frame currently being read | Use a valid tag, or check that the particle isn't removed/renumbered partway through the trajectory |
| `Particle index <N> is outside the valid range 0 ... <max>.` | Your GSD has no tags (so `particle` is treated as a raw array index) and the index is out of bounds | Use an index between `0` and `n_particles - 1` |

### Polyhedron reconstruction

| Symptom | Cause | Fix |
|---|---|---|
| Console warning: `Euler check is not 2. Inspect the JSON vertices / plane tolerance.` (non-fatal — rendering continues) | The reconstructed faces/edges/vertices don't satisfy `V - E + F = 2`, which every closed convex polyhedron must. Usually caused by duplicate/near-duplicate vertices, or nearly-but-not-quite coplanar points that fail the `1e-7` plane tolerance | Clean up duplicate vertices in your JSON; if your vertices are numerically noisy, consider relaxing `plane_tolerance` in `reconstruct_polygon_faces()` (see [Section 10](#10-things-you-can-only-change-by-editing-the-code)) |
| `scipy.spatial.qhull.QhullError` (from `ConvexHull`) | Vertices are degenerate — e.g. all coplanar, all collinear, or duplicated to the point that no 3D hull can be formed | Check your vertex data; a valid 3D polyhedron needs non-degenerate, non-coplanar points |

### Symmetry-axis detection (only relevant unless `--no-symmetry-axes`)

| Error message | Cause | Fix |
|---|---|---|
| `D5h rotational-axis validation failed: expected 10 proper rotations (D5 subgroup), but detected <N>.` | Your particle's true rotational symmetry group is not D5 (10 proper rotations) | Pass `--no-symmetry-axes` — see [Section 7](#7-the-rotational-symmetry-axis-feature--read-this-before-using-another-shape) |
| `D5h rotational-axis validation failed: expected exactly one C5 axis line, but detected <N>.` | Same as above | Same fix |
| `D5h rotational-axis validation failed: expected exactly five C2 axis lines, but detected <N>.` | Same as above | Same fix |
| `D5h rotational-axis validation failed: one or more detected C2 axes are not perpendicular to the C5 axis. ...` | Detected axes don't satisfy the D5 geometric constraint — could be a genuinely different/distorted shape, or numerical noise | Pass `--no-symmetry-axes`, or investigate whether your vertex data is slightly off from ideal D5h geometry |
| `Cannot determine symmetry axes because the body radius is zero.` | All vertices sit at the same point (degenerate shape) | Fix your vertex data |
| `No nonzero candidate rotational-symmetry axes were generated.` | Degenerate geometry (e.g. everything centered exactly at the origin with no offset directions) | Fix your vertex data |
| `Full-precision refinement changed a discovered symmetry permutation. Tighten --symmetry-precision-exponent.` | The coarse match and the refined, full-precision match disagreed — the tolerance is too loose for your shape's scale | Increase `--symmetry-precision-exponent` (stricter), or renormalize your shape's coordinate scale |
| `Could not assign an integer rotational order to a detected symmetry axis: minimum angle = ...` | An axis was found with a non-integer-order rotation, which shouldn't happen for a genuine point group | Almost always indicates the particle is not the expected symmetry type — pass `--no-symmetry-axes` |
| `A refined candidate symmetry operation is not a proper rotation.` | A candidate operation turned out to be improper (a reflection), which the detector explicitly excludes | Internal safety check — should not occur for genuine D5h geometry; if it does, treat as evidence the shape isn't D5h |

### Video-creation errors (non-fatal — PNGs are unaffected)

| Symptom | Cause | Fix |
|---|---|---|
| `No rendered PNG frames were found for video creation.` | `--output-dir` had no `frame_*.png` files when video creation ran (e.g. rendering step itself failed earlier, or you pointed `--output-dir` somewhere else than a previous run) | Confirm PNGs actually exist in `--output-dir` before troubleshooting the video step |
| `Video creation failed: <exception text>` | Usually a missing/broken `ffmpeg`/`libx264` backend for `imageio` | `pip install imageio-ffmpeg`, or install system `ffmpeg` with libx264 support |

---

## 12. Fully-Worked Example Commands

### 12.1 Minimal, isometric view, video, no symmetry overlay tuning (defaults)

```bash
python3.8 epd_pyvista_polyhedron_visualizer_v2p1_symmetry_axes.py \
    hpmc_hard_Elongated_Pentagonal_Dipyramid_vol_1p00_4096_NPT_P60_0_traj.gsd \
    shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json \
    300 \
    0 \
    --center \
    --resolution 2400 \
    --camera-angle iso \
    --create-video \
    --video-fps 10
```
- Renders the last 300 frames of the trajectory for particle 0.
- Symmetry axes (C5 red, C2 green) drawn by default since this is the EPD.
- Video at 10 fps ≈ 30 seconds long (300 frames ÷ 10 fps).

### 12.2 Fully-loaded example (interactive frame pruning, transparent PNGs, custom symmetry styling)

```bash
python3.8 epd_pyvista_polyhedron_visualizer_v2p1_symmetry_axes.py \
    hpmc_hard_Elongated_Pentagonal_Dipyramid_vol_1p00_4096_NPT_P60_0_traj.gsd \
    shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json \
    300 \
    0 \
    --center \
    --resolution 2400 \
    --camera-angle iso \
    --interactive-selection \
    --create-video \
    --video-fps 10 \
    --transparent-background \
    --color-mode solid \
    --no-title \
    --c5-axis-color "#D62728" \
    --c2-axis-color "#2CA02C" \
    --symmetry-axis-length-factor 1.6 \
    --c5-axis-radius-factor 0.018 \
    --c2-axis-radius-factor 0.012
```

What each extra flag does here, beyond §12.1:
- `--interactive-selection` — you'll be prompted in the terminal to
  optionally exclude specific frames from the 300-frame window.
- `--transparent-background` — PNGs get an alpha channel instead of solid
  white (useful for compositing into slides/figures).
- `--color-mode solid` — explicit (this is also the default) — every frame
  uses the same `--face-color`.
- `--no-title` — clean images with no on-image text overlay.
- `--c5-axis-color`, `--c2-axis-color` — explicit color choices (these
  happen to match the defaults, shown here for clarity on how to override
  them).
- `--symmetry-axis-length-factor 1.6` — makes the axis tubes extend further
  beyond the particle body than the default `1.35`.
- `--c5-axis-radius-factor`, `--c2-axis-radius-factor` — tube thickness.

### 12.3 A *different*, non-D5h polyhedron (required extra flag highlighted)

```bash
python3.8 epd_pyvista_polyhedron_visualizer_v2p1_symmetry_axes.py \
    my_cube_trajectory.gsd \
    my_cube_shape.json \
    150 \
    3 \
    --center \
    --no-symmetry-axes \
    --resolution 2000 \
    --camera-angle iso \
    --create-video \
    --video-fps 8
```
Without `--no-symmetry-axes` here, this command would raise
`RuntimeError: D5h rotational-axis validation failed: ...` and produce no
output at all, because a cube's proper rotational subgroup (24 operations)
does not match the hard-coded D5 expectation (10 operations). See
[Section 7](#7-the-rotational-symmetry-axis-feature--read-this-before-using-another-shape).

---

## 13. Performance Notes

- **Symmetry-axis detection runs once**, in the body frame, before any
  per-frame rendering begins — it does not repeat per frame, so it does not
  scale with `n_frames`. However, it does scale with the number of
  vertices/faces/edges of your polyhedron (more candidate axes × the fixed
  candidate-angle list × an `O(N^3)`-ish optimal assignment per candidate),
  so very high-vertex-count shapes may take noticeably longer at start-up.
- **Per-frame rendering cost** is dominated by resolution and
  anti-aliasing: `2400×2400` with `ssaa`/`msaa` anti-aliasing is
  deliberately high quality and therefore slower than a quick preview.
  Expect render time per frame to grow roughly with resolution².
- For quick previews before committing to a long run, temporarily lower
  `--resolution` (e.g. `800`) and use a small `n_frames`, then scale back up
  once you're happy with camera angle, colors, and symmetry-axis styling.
- Video encoding time scales with the number of PNGs, not with
  `--video-fps` (fps only changes playback speed of the final file, not how
  long encoding takes).

---

## 14. Quick-Reference Cheat Sheet

```bash
python3.8 epd_pyvista_polyhedron_visualizer_v2p1_symmetry_axes.py \
    <GSD_FILE> \
    <SHAPE_JSON> \
    <N_FRAMES> \
    <PARTICLE_INDEX> \
    [--center] \
    [--no-symmetry-axes]                      # REQUIRED unless your shape is D5h!
    [--output-dir DIR] \
    [--resolution PIXELS] \
    [--camera-angle {iso,front,back,side,left,top,bottom}] \
    [--exclude "10,15-20,30"] \
    [--interactive-selection] \
    [--color-mode {solid,frame}] \
    [--face-color "#RRGGBB"] \
    [--edge-color "#RRGGBB"] \
    [--background-color NAME_OR_HEX] \
    [--edge-width WIDTH] \
    [--opacity 0-1] \
    [--show-vertices] \
    [--show-axes] \
    [--no-title] \
    [--transparent-background] \
    [--create-video] \
    [--video-fps FPS]                         # controls PLAYBACK SPEED
```

| I want to... | Change this |
|---|---|
| Use my own trajectory | positional `gsd_file` |
| Use my own polyhedron shape | positional `shape_json`, using an accepted key (§3.2) |
| Render a different particle | positional `particle` |
| Render more/fewer recent frames | positional `n_frames` |
| Skip specific frames, reproducibly | `--exclude "..."` |
| Skip frames by eyeballing at run-time | `--interactive-selection` |
| Avoid a crash on a non-EPD shape | add `--no-symmetry-axes` |
| See only rotation, not drift | add `--center` |
| Make a faster/slower video (same frames) | raise/lower `--video-fps` |
| Bigger/smaller images | `--resolution` |
| Different viewpoint | `--camera-angle` |
| Change face/edge/background colors | `--face-color` / `--edge-color` / `--background-color` |
| Change video compression quality | edit the code (§10) — not a flag |
| Change lighting setup | edit the code (§10) — not a flag |
| Support a non-D5h symmetry overlay | edit the code (§10) — not a flag |
