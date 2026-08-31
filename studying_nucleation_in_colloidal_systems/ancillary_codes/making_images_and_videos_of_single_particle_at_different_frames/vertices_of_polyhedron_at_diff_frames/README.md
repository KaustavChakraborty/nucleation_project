# EPD Single-Particle Vertex-Color Rotation Checker (v4p0)

**Script file:** `epd_rotation_checking_for_single_particle_v4p0.py`

A diagnostic plotting tool for a single HPMC particle from a HOOMD GSD
trajectory. It overlays the particle's polyhedron vertices from **every
selected frame into one static 3D matplotlib figure**, coloring each point
by **which vertex it is** (vertex index) rather than which frame it came
from. Every occurrence of "vertex 0," across every selected frame, is drawn
in the same color; every occurrence of "vertex 1" is drawn in a different
color; and so on.

This README explains, in full detail:
- exactly what the script does, step by step,
- how to read/interpret the resulting figure scientifically,
- every command-line parameter,
- the **mandatory interactive prompt** this script always asks (there is no
  flag to skip it),
- several **non-obvious/silent behaviors** you should know about before
  trusting the output,
- what to change to point it at your own trajectory/shape, and
- every error the script can raise, what triggers it, and how to fix it.

> **Not the same tool as the PyVista polyhedron/video renderer.** This
> script produces **one static PNG** (a scatter overlay), not a
> per-frame image sequence or an MP4. If you want a rendered rotating solid
> polyhedron and a video, use
> `epd_pyvista_polyhedron_visualizer_v2p1_symmetry_axes.py` instead (see its
> own README). This document is only about the vertex-color scatter script.

---

## Table of Contents

1. [What This Script Actually Does](#1-what-this-script-actually-does)
2. [How To Interpret the Output Plot](#2-how-to-interpret-the-output-plot)
3. [Installation / Requirements](#3-installation--requirements)
4. [Input Files You Need](#4-input-files-you-need)
5. [Quick Start](#5-quick-start)
6. [Full Command-Line Reference](#6-full-command-line-reference)
7. [The Mandatory Interactive Frame-Exclusion Prompt, In Detail](#7-the-mandatory-interactive-frame-exclusion-prompt-in-detail)
8. [Output File Produced](#8-output-file-produced)
9. [The `plt.show()` / Headless-Server Gotcha](#9-the-pltshow--headless-server-gotcha)
10. [Silent Behaviors You Should Know About](#10-silent-behaviors-you-should-know-about)
11. [Adapting This Script To Your Own Polyhedron and Trajectory](#11-adapting-this-script-to-your-own-polyhedron-and-trajectory)
12. [Things You Can Only Change By Editing the Code](#12-things-you-can-only-change-by-editing-the-code)
13. [Complete Error Reference / Troubleshooting](#13-complete-error-reference--troubleshooting)
14. [Fully-Worked Example](#14-fully-worked-example)
15. [Performance Notes](#15-performance-notes)
16. [Quick-Reference Cheat Sheet](#16-quick-reference-cheat-sheet)

---

## 1. What This Script Actually Does

For the one particle number you give it, the script does the following,
in order:

1. **Opens** the GSD trajectory.
2. **Computes a base frame window**: the last `n_frames` frames of the
   trajectory (i.e. `frame = total_frames - n_frames` up through the final
   frame). If `n_frames` is larger than the trajectory, it silently uses
   every available frame instead (with a printed warning) rather than
   erroring.
3. **Always** interactively asks you, in the terminal, whether to use every
   frame in that window or to exclude specific ones — see
   [Section 7](#7-the-mandatory-interactive-frame-exclusion-prompt-in-detail).
   **This prompt cannot be skipped or disabled with a flag** in this script
   version.
4. **Locates the particle** in the first selected frame — by tag if the GSD
   frame carries particle tags, otherwise by treating the number you gave
   as a raw array index.
5. **Obtains the body-frame polyhedron vertices**: first it tries to read
   them straight out of the GSD file's HPMC shape metadata; if that isn't
   available, it falls back to your `shape_json` file.
6. **Loops over every selected frame** and, for each one:
   - re-locates the particle (safer than assuming a fixed array index,
     in case particle ordering changes between frames),
   - reads that particle's position and quaternion orientation,
   - rotates (and, unless `--center` is given, translates) the body-frame
     vertices into that frame's lab-frame position/orientation,
   - prints a diagnostic line with the frame number, array index, position,
     and quaternion.
7. **Builds one 3D matplotlib scatter plot** containing every vertex from
   every selected frame, where the **color of each point is determined
   solely by its vertex index**, not by which frame it came from (see
   [Section 2](#2-how-to-interpret-the-output-plot)).
8. Optionally draws the particle's center-of-mass trajectory as a dashed
   line (only when **not** using `--center`), adds a "Vertex Index"
   colorbar, equal-aspect 3D axes, a title, and **saves one PNG** (unless
   `--no-save`).
9. Calls `plt.show()` — **unconditionally, every run**, regardless of
   `--no-save` (see [Section 9](#9-the-pltshow--headless-server-gotcha)).

### Why this is useful (the scientific point of the plot)

Because every occurrence of a given vertex — across all the frames you
selected — is drawn in one consistent color, the resulting figure turns
into `N_vertices` distinct point clusters. **The size/tightness of each
colored cluster is a direct visual measure of how much that specific
vertex moved (rotationally, and translationally if not centered) across
your selected frame window.** A tight, small cluster of one color means
that vertex barely changed position across those frames; a diffuse, spread
out cluster of one color means that vertex swept through a wide range of
positions — i.e., that corner of the polyhedron experienced more
rotational libration/disorder over that time window. This is exactly the
kind of per-vertex orientational-disorder diagnostic that's useful when
screening a trajectory before doing more rigorous quantitative rotational
analysis.

---

## 2. How To Interpret the Output Plot

- **Color = vertex identity, not time.** Look for a particular color, and
  you are looking at *one specific corner of the polyhedron* across every
  frame you selected simultaneously.
- **A tight cluster of a given color** → that vertex is rotationally/
  translationally stable across the selected window.
- **A spread-out cloud of a given color** → that vertex swept through a
  wide range of positions — evidence of larger rotational motion for that
  part of the particle over that time window.
- **The dashed line** (only drawn when `--center` is *not* used) is the
  particle's **center-of-mass trajectory** — this tells you how much the
  whole particle drifted/translated, independent of the vertex clusters
  around it.
- **With `--center`**, all vertex clusters are drawn around the origin
  (translation removed), so what you're looking at is **rotation only** —
  the clusters directly show orientational spread, uncontaminated by the
  particle's actual path through the box.
- **Colorbar** ("Vertex Index") tells you which color corresponds to which
  vertex index (0-based) in your shape's vertex array — cross-reference
  this against your `shape_json` file if you need to know which physical
  corner of the polyhedron a given vertex index corresponds to.
- **Colormap choice is automatic and vertex-count dependent**: if your
  polyhedron has **20 or fewer** vertices, the script uses matplotlib's
  `tab20` colormap (20 well-separated, easily distinguishable discrete
  colors — this is what you'll get for the 12-vertex EPD example shape). If
  your polyhedron has **more than 20** vertices, the script switches to the
  continuous `hsv` colormap instead, which will make individual vertices
  progressively harder to visually distinguish as vertex count grows. See
  [Section 12](#12-things-you-can-only-change-by-editing-the-code) if you
  need to change this behavior for a very high-vertex-count shape.

---

## 3. Installation / Requirements

```bash
pip install numpy matplotlib gsd scipy
```

| Package | Why it's needed |
|---|---|
| `numpy` | All array/numerical work |
| `matplotlib` | The 3D scatter plot, colormaps, colorbar, and figure saving/showing |
| `gsd` | Reading the HOOMD `.gsd` trajectory file |
| `scipy` | Quaternion-to-rotation conversion (`scipy.spatial.transform.Rotation`) |

### ⚠️ A non-obvious dependency-loading detail

`scipy` is **not imported at the top of the file** — it's imported *locally*,
inside `transform_vertices()`, the first time that function runs (which
happens partway through the frame-reading loop, well after argument
parsing, GSD opening, the interactive frame-exclusion prompt, and shape
lookup have all already completed). This means:

- If `scipy` is missing, the script will **not** fail immediately at
  startup. It will get all the way through opening the file, the
  interactive prompt, and particle/shape lookup, and only then crash with
  `ModuleNotFoundError: No module named 'scipy'` on the **first frame** of
  the reading loop.
- Practically: if you're missing `scipy`, you'll still have to sit through
  the interactive frame-exclusion prompt before finding out. Install
  `scipy` up front to avoid wasting a terminal session on this.

---

## 4. Input Files You Need

### 4.1 The GSD trajectory file (positional argument `gsd_file`)

Any standard HOOMD-blue GSD trajectory with HPMC rigid polyhedral
particles, containing `particles.position` and `particles.orientation` in
every frame you plan to select.

### 4.2 The shape JSON file (positional argument `shape_json`)

**Always required on the command line**, even if the GSD already has
embedded shape data — it's used purely as a fallback (pipeline step 5
above). If the GSD lookup succeeds, the JSON file's *contents* are never
actually read, but the path must still be a valid string.

Unlike some other tools in this project, **this script recognizes only two
possible JSON keys**, checked in this exact order:

```
1. "8_vertices"
2. "vertices"
```

> ⚠️ If your JSON instead uses a key like `"12_vertices"` or
> `"polyhedron_vertices"` (both of which *are* accepted by the companion
> PyVista script in this project), **this script will not find them** and
> will raise a `KeyError`. Rename the key to `"vertices"` or `"8_vertices"`,
> or edit `read_shape_from_json()` — see
> [Section 12](#12-things-you-can-only-change-by-editing-the-code).

The bundled example, `shape_037_Elongated_Pentagonal_Dipyramid_unit_
volume_principal_frame.json`, uses the key `"8_vertices"` (a naming holdover
from another cataloguing convention) but actually contains **12** vertices
— that's fine, since the script only looks at the key *name*, not how many
vertices are inside it:

```json
{
    "1_Name": "Elongated Pentagonal Dipyramid",
    "2_ShortName": "J16",
    "8_vertices": [
        [0.7744378962411786, -9.972731702719086e-08, -1.2951643733570227e-17],
        [-0.7744378962411786, -9.972731702719086e-08, -1.2951643733570227e-17],
        ... 12 vertices total ...
    ]
}
```

**Requirements on the vertex array**, checked explicitly by
`read_shape_from_json()`:
- Must be 2-dimensional with shape `(N_vertices, 3)`.
- If it is not, you get a `ValueError` stating the shape it actually found.
- (Unlike the companion PyVista script, this script does **not** separately
  check for a minimum vertex count or for `NaN`/infinite values — a
  malformed vertex array that happens to still be shape `(N, 3)` will pass
  this check and only fail later, if at all, inside matplotlib's plotting
  calls.)

---

## 5. Quick Start

```bash
python3.8 epd_rotation_checking_for_single_particle_v4p0.py \
    hpmc_hard_Elongated_Pentagonal_Dipyramid_vol_1p00_4096_NPT_P60_0_traj.gsd \
    shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json \
    300 \
    0 \
    --center
```

What happens:
1. Opens the trajectory.
2. Computes a base window of the **last 300 frames**.
3. **Immediately prompts you** (see [Section 7](#7-the-mandatory-interactive-frame-exclusion-prompt-in-detail))
   to confirm whether to use all 300 or exclude some.
4. Renders **particle 0**, with translation removed (`--center`) so the
   plot shows pure rotational spread.
5. Saves a PNG named automatically (see [Section 8](#8-output-file-produced)).
6. Opens a matplotlib window showing the figure (see
   [Section 9](#9-the-pltshow--headless-server-gotcha) if you're on a
   remote/headless machine).

---

## 6. Full Command-Line Reference

### 6.1 Required positional arguments (order matters)

| # | Name | Type | Meaning | Change this to... |
|---|---|---|---|---|
| 1 | `gsd_file` | path | HOOMD GSD trajectory file | your own `.gsd` trajectory |
| 2 | `shape_json` | path | JSON with body-frame polyhedron vertices (fallback; only `"8_vertices"`/`"vertices"` keys recognized, §4.2) | your own shape JSON |
| 3 | `n_frames` | int | Take this many consecutive frames from the **end** of the trajectory, as the *base* window before the interactive prompt | how many recent frames to start from |
| 4 | `particle` | int | Particle **tag** if the GSD has tags, otherwise raw **array index** | the particle number you want to visualize |

### 6.2 Optional flags

| Flag | Type / Default | Meaning |
|---|---|---|
| `--center` | flag, off by default | Place the particle's center at the origin in every frame, removing translation and showing rotation only. Also suppresses the center-of-mass trajectory dashed line (§2) |
| `--point-size FLOAT` | float, default `30.0` | Matplotlib scatter marker size (`s` parameter — marker **area** in points², not radius/diameter). Increase for bigger, more visible dots; decrease if many overlapping points make the plot too dense/cluttered |
| `--no-save` | flag, off by default | Skip the `plt.savefig(...)` call — **does not** skip the final `plt.show()` call (see [Section 9](#9-the-pltshow--headless-server-gotcha)) |
| `--output FILENAME` | string, default `None` (auto-named) | Custom output filename. **The file format is inferred by matplotlib from the extension** you give (`.png`, `.pdf`, `.svg`, `.jpg`, etc. all work) — if omitted, an auto-generated `.png` filename is used (§8) |

There is **no** `--exclude` flag and **no** way to non-interactively specify
frame exclusions from the command line in this script version — the only
mechanism is the mandatory interactive prompt described next.

---

## 7. The Mandatory Interactive Frame-Exclusion Prompt, In Detail

**This prompt runs every single time you run the script, unconditionally.**
There is no flag to answer it automatically or skip it — if you run this
script from a non-interactive context (a cron job, a batch queue script
with no attached terminal, etc.), it will hang waiting for input on
`input(...)` (or fail outright, depending on how stdin is handled in that
context).

### 7.1 What you see

```
======================================================================
FRAME SELECTION
======================================================================

Total frames available: 300
Frame indices: [0, 1, 2, 3, ..., 299]

Do you want to process ALL frames? (yes/no): 
```

- Type `yes` or `y` → every frame in the base window is used, unchanged.
- Type `no` or `n` → you're asked a second question:
  ```
  Enter frames to exclude (comma-separated, e.g., 10,11,12): 
  ```
- Anything else at the first prompt (typos, empty input, etc.) → reprints
  `Invalid response. Please enter 'yes' or 'no'.` and asks again,
  indefinitely, until you answer one of the recognized forms.

### 7.2 Exclusion syntax (second question)

- **Individual frames, comma-separated:** `10,11,12`
- **Inclusive ranges with a hyphen:** `10-15` → excludes 10, 11, 12, 13,
  14, 15
- **Mixed:** `5,10-15,20`
- **Leaving it empty and pressing Enter** → prints `No frames excluded.`
  and behaves exactly as if you'd answered `yes` at the first prompt (all
  frames in the base window are used).

After a valid, non-empty exclusion is parsed, you'll see a confirmation
like:

```
Excluded frames: [10, 11, 12]
Remaining frames: 297 (from 300 total)
Frame indices to process: [0, 1, 2, ..., 9, 13, 14, ..., 299]
```

### 7.3 Retry/loop behaviors you should know about

- **Malformed input** (non-numeric text, a broken range like `10-15-20`,
  etc.) → catches the parsing error, prints
  `Invalid input. Please enter frame numbers separated by commas
  (e.g., 10,11,12 or 10-12).`, and re-asks the *same* exclusion question —
  it does **not** send you back to the yes/no question.
- **Excluding every frame in the current window** → prints
  `Error: All frames were excluded! Please try again.` and re-asks the same
  exclusion question (also does not return to yes/no).
- **Entering a frame number that isn't actually in the current window**
  (e.g. typing `500` when your window only spans frames 0–299) is **not**
  an error and is **not caught** — it is silently a no-op. The confirmation
  message will still echo back everything you *typed* under "Excluded
  frames," even though a number outside the window had no actual effect on
  what gets removed. Double-check the printed "Frame indices to process"
  line (not just "Excluded frames") if you want to confirm the real result.

### 7.4 A genuine edge case: `n_frames` must be a positive integer

If you pass `n_frames` as `0` (or a value that otherwise makes the base
window empty, e.g. a large negative number), the base `frame_indices` array
is empty *before* the prompt even runs. In that situation:
- Answering `yes` leads to an immediate, uncaught `IndexError` a few lines
  later (`frame_indices[0]` on an empty array) when the script tries to
  read the first selected frame.
- Answering `no` and trying to "exclude" frames from an already-empty list
  can never produce a non-empty result, so the script will loop forever on
  `Error: All frames were excluded! Please try again.` with no way to
  escape except `Ctrl+C`.

**Always use a positive `n_frames`.**

---

## 8. Output File Produced

Unless `--no-save` is given, exactly **one** image file is written, to the
**current working directory** (there is no `--output-dir` option in this
script — contrast with the PyVista script, which writes many files into a
configurable folder).

### 8.1 Automatic filename (when `--output` is not given)

The final frame count used (`number_to_use`, i.e. **after** the interactive
exclusion step — this may be smaller than the `n_frames` you passed on the
command line) is embedded directly in the filename:

| Condition | Filename pattern |
|---|---|
| `--center` used | `particle_<PARTICLE>_last_<number_to_use>_frames_vertex_color_centered.png` |
| `--center` not used | `particle_<PARTICLE>_last_<number_to_use>_frames_vertex_color.png` |

Example: particle `0`, requested 300 frames, excluded 8 in the prompt,
used `--center` → `particle_0_last_292_frames_vertex_color_centered.png`.

### 8.2 Custom filename (`--output`)

If you pass `--output myplot.pdf`, that exact string is used as-is —
**matplotlib infers the save format from the extension you give**, so
`.png`, `.pdf`, `.svg`, `.jpg`, etc. are all valid; just make sure the
extension matches the format you actually want.

### 8.3 Fixed save settings (not exposed as flags)

- Resolution: `dpi=300`
- Cropping: `bbox_inches="tight"`

See [Section 12](#12-things-you-can-only-change-by-editing-the-code) if you
need different values.

---

## 9. The `plt.show()` / Headless-Server Gotcha

**The script calls `plt.show()` unconditionally at the very end, every
single run — regardless of whether you passed `--no-save`.** `--no-save`
only skips the `plt.savefig(...)` call; it does **not** skip `plt.show()`.

If you're running this over SSH on a remote machine/cluster without a
usable display (no X11 forwarding, no `$DISPLAY` set — a common setup for
HPC/workstation sessions), `plt.show()` can either:
- raise something like
  `_tkinter.TclError: no display name and no $DISPLAY environment
  variable`, or
- hang the terminal waiting on a GUI event loop that will never receive
  input, depending on your matplotlib backend.

**Your PNG will already have been saved successfully before this happens**
(the `savefig` call runs before `plt.show()`), so this is usually harmless
to your actual output — just an annoying way for the script to end. Options:

- **SSH with X11 forwarding**: connect with `ssh -X` or `ssh -Y` so a
  window can actually be displayed.
- **Force a non-interactive backend** so `plt.show()` becomes a no-op,
  e.g. run with:
  ```bash
  MPLBACKEND=Agg python3.8 epd_rotation_checking_for_single_particle_v4p0.py ...
  ```
- **Edit the script** and remove (or comment out) the final `plt.show()`
  call if you never want an interactive window, only the saved file — see
  [Section 12](#12-things-you-can-only-change-by-editing-the-code).

---

## 10. Silent Behaviors You Should Know About

These are real behaviors in the current code that do **not** raise an
error or print any warning, but can silently affect correctness. Worth
reading once carefully.

### 10.1 An unmatched particle tag is silently reinterpreted as a raw index

`find_particle_index()` checks whether your GSD frame carries particle
tags. If it does, and your requested particle number **is** among those
tags, it correctly resolves the array index for that tag. **But if the GSD
has tags and your requested number is *not* found among them, the function
does not raise an error.** It silently falls through and treats your
number as if it were a raw array index instead — potentially pointing at a
**completely different, wrong particle**, with no warning printed. Always
sanity-check the printed `Particle tag : ...` / `Current array index :
...` lines under "PARTICLE INFORMATION" at the start of a run to make sure
the number you asked for is actually being resolved the way you expect.

### 10.2 Negative particle numbers are accepted as Python/NumPy "from the end" indices

If your requested particle ends up being used as a raw array index (either
because the frame has no tags, or due to §10.1 above), and you passed a
**negative** number, NumPy will interpret it as counting from the end of
the particle array (e.g. `particle -1` → the *last* particle in the
system) instead of raising an out-of-range error. This is standard NumPy
indexing behavior, but it's easy to forget and can silently select an
unintended particle if you meant to type a positive tag/index and made a
sign error.

### 10.3 Excluding a frame number outside the current window is a silent no-op

Covered in [§7.3](#73-retryloop-behaviors-you-should-know-about) — repeated
here because it's easy to miss: the confirmation text echoes back whatever
you typed as "Excluded frames" even if none of those numbers were actually
present in the window, so don't rely on that line alone — check "Frame
indices to process."

---

## 11. Adapting This Script To Your Own Polyhedron and Trajectory

1. **Get your shape's body-frame vertices** into a JSON file using the key
   `"vertices"` (simplest/safest choice — always works) or `"8_vertices"`
   — see [§4.2](#42-the-shape-json-file-positional-argument-shape_json)
   for the exact accepted-key restriction of this script.
2. **Point `gsd_file` at your own trajectory.**
3. **Set `particle`** to the tag or array index of the specific particle
   you want — then verify the "PARTICLE INFORMATION" console output on
   your first run to make sure it resolved the way you intended (§10.1).
4. **Choose `n_frames`** — the size of the base window, counted back from
   the end of the trajectory. Use a **positive integer** (§7.4).
5. **Run it, and answer the interactive prompt** (§7) — decide at that
   point whether you want the full window or want to exclude specific
   frames (e.g. equilibration frames, or frames you know are corrupted/
   outliers).
6. **Add `--center`** if you want a pure-rotation view (translation
   removed) — recommended for orientational-disorder-style analysis, since
   it isolates vertex spread from center-of-mass drift.
7. **Adjust `--point-size`** if your polyhedron's vertex clusters look too
   sparse (increase it) or too cluttered/overlapping (decrease it).
8. **Optionally set `--output`** to control the filename/format, or
   `--no-save` if you only want to look at it interactively and not keep a
   file (remembering §9 — you'll still get a `plt.show()` window/attempt
   either way).

### Minimal "new shape" template

```bash
python3.8 epd_rotation_checking_for_single_particle_v4p0.py \
    YOUR_TRAJECTORY.gsd \
    YOUR_SHAPE.json \
    NUMBER_OF_FRAMES \
    PARTICLE_INDEX \
    --center \
    --point-size 40 \
    --output your_custom_name.png
```

Replace the four UPPER-CASE tokens with your own values, then answer the
interactive prompt when it appears.

---

## 12. Things You Can Only Change By Editing the Code

Not everything is exposed as a command-line flag. To change any of these,
edit the `.py` file directly.

| What | Where in the code | Current fixed value |
|---|---|---|
| Whether `plt.show()` runs at all | end of `main()`, the final `plt.show()` call | always runs, unconditionally (§9) |
| Save resolution | `main()`, the `plt.savefig(...)` call | `dpi=300`, `bbox_inches="tight"` |
| Figure size | `main()`, `plt.figure(figsize=(9, 8))` | 9×8 inches |
| Scatter point transparency | `main()`, `ax.scatter(..., alpha=0.80)` | fixed at `0.80` |
| Colormap choice / switch threshold | `main()`, `cmap = plt.cm.tab20 if num_vertices <= 20 else plt.cm.hsv` | `tab20` for ≤ 20 vertices, `hsv` above that |
| Accepted JSON vertex key names | `read_shape_from_json()` | only `"8_vertices"` (checked first) or `"vertices"` |
| 3D camera/view angle | not set anywhere | matplotlib's default `Axes3D` view (no fixed `ax.view_init(...)` call); you rotate manually in the interactive window, or add your own `ax.view_init(elev=..., azim=...)` call for a reproducible angle |
| Colorbar sizing | `main()`, `fig.colorbar(..., pad=0.10, shrink=0.75)` | as shown |
| Title wording | `main()`, the `ax.set_title(...)` calls | fixed text template, e.g. `"Particle {N}: last {M} frames\nColor = Vertex Index"` |
| Quaternion component order assumed | `transform_vertices()` | HOOMD scalar-first `[q_w, q_x, q_y, q_z]`, converted to SciPy's `[x, y, z, w]` order internally — only change this if your orientation data uses a different convention |
| Output directory | not configurable | always the current working directory; there is no `--output-dir` |

---

## 13. Complete Error Reference / Troubleshooting

All errors below are real, verified checks/behaviors in the script. The
table separates errors the script **catches and exits cleanly from**, from
ones that are **uncaught** and will show you a full Python traceback.

### 13.1 GSD opening — caught, clean exit

| Message | Cause | Fix |
|---|---|---|
| `Error opening GSD file:\n    <exception text>` (script then calls `sys.exit(1)`, no traceback) | Wrong path, corrupted file, unreadable GSD | Check the path and that the file is a valid GSD trajectory |

### 13.2 Shape JSON — uncaught, will show a traceback

| Error type / message | Cause | Fix |
|---|---|---|
| `FileNotFoundError: Shape JSON file not found: <path>` | Wrong path to `shape_json` (only actually raised if the GSD didn't already supply shape data) | Check the path/typo |
| `KeyError: Could not find polyhedron vertices in JSON. Expected key "8_vertices" or "vertices". Available keys: [...]` | Your JSON uses a different key (e.g. `"12_vertices"`, `"polyhedron_vertices"` — **not** recognized by this script, see §4.2) | Rename the key to `"vertices"` or `"8_vertices"`, or edit `read_shape_from_json()` |
| `ValueError: Vertices must have dimensions (N_vertices, 3). Found shape: (...)` | Your JSON's vertex array isn't a plain `Nx3` list of `[x, y, z]` | Fix the JSON structure |

### 13.3 Particle lookup — mostly uncaught / silent, see Section 10

| Symptom | Cause | Fix |
|---|---|---|
| `IndexError: index <N> is out of bounds for axis 0 with size <M>` (raised from inside `get_particle_typeid`, or later from `frame.particles.position[particle_index]` / `orientation[particle_index]`) | The `particle` number you gave, once resolved as a raw array index, is outside the valid `0 ... n_particles-1` range for this system | Check your system's actual particle count; use a valid tag/index |
| No error at all, but the wrong particle appears to be plotted | Requested a tag that doesn't exist in a tagged GSD — silently falls back to raw-index interpretation, see [§10.1](#101-an-unmatched-particle-tag-is-silently-reinterpreted-as-a-raw-index) | Verify the "PARTICLE INFORMATION" printout at the start of every run |
| Unexpected particle selected when using a negative number | NumPy negative-index wraparound, see [§10.2](#102-negative-particle-numbers-are-accepted-as-numpyfrom-the-end-indices) | Use a non-negative particle number unless you specifically intend "from the end" indexing |

### 13.4 Dependency errors — uncaught, will show a traceback

| Message | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'scipy'` — appears **partway through the run**, on the first frame processed (see §3) | `scipy` not installed | `pip install scipy` |
| `ModuleNotFoundError: No module named 'gsd'` — appears immediately at startup | `gsd` not installed | `pip install gsd` |
| `ModuleNotFoundError: No module named 'matplotlib'` — appears immediately at startup | `matplotlib` not installed | `pip install matplotlib` |

### 13.5 Interactive-prompt issues

| Symptom | Cause | Fix |
|---|---|---|
| Script hangs indefinitely with no visible prompt | Running in a non-interactive context (batch job, redirected stdin, some IDE consoles) where `input()` cannot receive input | Run interactively from a real terminal; this script has no non-interactive frame-selection mode |
| Repeats `Error: All frames were excluded! Please try again.` forever, unresponsive to further input other than more exclusions | `n_frames` produced an empty base window (e.g. `0` or a value ≥ total frames in the wrong direction) — see [§7.4](#74-a-genuine-edge-case-n_frames-must-be-a-positive-integer) | `Ctrl+C` and rerun with a positive `n_frames` that actually yields a non-empty window |
| Repeats `Invalid input. Please enter frame numbers...` | Typed non-numeric text, or a malformed range (e.g. `10-15-20`, or a range with the parts in the wrong order in a way that still fails parsing) | Re-enter using the documented syntax (§7.2) |

### 13.6 Display errors at the very end of a successful run

| Message | Cause | Fix |
|---|---|---|
| `_tkinter.TclError: no display name and no $DISPLAY environment variable` (or similar backend errors), appearing **after** "Figure saved as: ..." has already printed | `plt.show()` running on a machine/session with no usable display — see [Section 9](#9-the-pltshow--headless-server-gotcha) | Your file is already saved; use `MPLBACKEND=Agg`, SSH with `-X`/`-Y`, or edit out the `plt.show()` call |

---

## 14. Fully-Worked Example

```bash
python3.8 epd_rotation_checking_for_single_particle_v4p0.py \
    hpmc_hard_Elongated_Pentagonal_Dipyramid_vol_1p00_4096_NPT_P60_0_traj.gsd \
    shape_037_Elongated_Pentagonal_Dipyramid_unit_volume_principal_frame.json \
    300 \
    0 \
    --center
```

Representative terminal transcript (abbreviated):

```
======================================================================
OPENING GSD FILE
======================================================================

GSD file: hpmc_hard_Elongated_Pentagonal_Dipyramid_vol_1p00_4096_NPT_P60_0_traj.gsd
Total frames in trajectory: 300
Frame range: 0 to 299 (300 frames)

======================================================================
FRAME SELECTION
======================================================================

Total frames available: 300
Frame indices: [0, 1, 2, ..., 299]

Do you want to process ALL frames? (yes/no): no

Enter frames to exclude (comma-separated, e.g., 10,11,12): 42,71,177,212,234,240,292,296

Excluded frames: [42, 71, 177, 212, 234, 240, 292, 296]
Remaining frames: 292 (from 300 total)
Frame indices to process: [0, 1, ..., 41, 43, ..., 299]

======================================================================
PARTICLE INFORMATION
======================================================================
Particle index       : 0
Explicit particle tags were not available; using array index.

Number of polyhedron vertices = 12

======================================================================
READING FRAMES
======================================================================
Frame       0 | index      0 | r = (   5.12345,    3.45678,   -2.34567) | q = (  0.99999,   0.00123,   0.00234,   0.00345)
...
Frame     299 | index      0 | r = (   9.87654,    7.65432,   -0.12345) | q = (  0.99900,   0.03456,   0.04567,   0.05678)


======================================================================
OUTPUT
======================================================================
Figure saved as:
    particle_0_last_292_frames_vertex_color_centered.png
```

At this point a matplotlib window attempts to open (see
[Section 9](#9-the-pltshow--headless-server-gotcha)), and the PNG above is
already safely on disk regardless of whether that window succeeds.

---

## 15. Performance Notes

- This script is **much lighter** than the PyVista renderer: it produces
  exactly one static image with no per-frame image rendering and no video
  encoding, so it's fast even for hundreds of frames.
- Total plotted points = `n_selected_frames × n_vertices` individual
  `scatter()` calls (one call per vertex per frame, in a nested loop) — for
  very large frame counts *and* high-vertex-count shapes together, this can
  visibly slow down figure rendering/saving since matplotlib issues one
  draw call per point rather than a single batched call. If you routinely
  work with very large `n_frames × n_vertices` products and find plotting
  slow, batching all points for a given vertex index into a single
  `scatter()` call (passing arrays instead of scalars) would be a
  straightforward code-level performance improvement.
- The dominant cost for large `n_frames` is typically reading that many
  frames out of the GSD file itself, not the plotting step.

---

## 16. Quick-Reference Cheat Sheet

```bash
python3.8 epd_rotation_checking_for_single_particle_v4p0.py \
    <GSD_FILE> \
    <SHAPE_JSON> \
    <N_FRAMES> \
    <PARTICLE_INDEX> \
    [--center] \
    [--point-size SIZE] \
    [--no-save] \
    [--output FILENAME.ext]
# Script will THEN interactively ask you about frame exclusion — always.
```

| I want to... | Change this |
|---|---|
| Use my own trajectory | positional `gsd_file` |
| Use my own polyhedron shape | positional `shape_json`, using key `"vertices"` or `"8_vertices"` only |
| Render a different particle | positional `particle` — then check the printed "PARTICLE INFORMATION" to confirm it resolved correctly |
| Change the base frame count before the prompt | positional `n_frames` (must be positive) |
| Skip specific frames | answer `no` at the interactive prompt, then list them (no CLI flag exists for this) |
| See rotation only, not drift | add `--center` |
| Bigger/smaller plotted points | `--point-size` |
| Only view, don't save | `--no-save` (note: a display window will still be attempted, §9) |
| Custom filename / format | `--output name.png` / `.pdf` / `.svg` / etc. |
| Avoid a crash/hang on a headless server | run with `MPLBACKEND=Agg` prefixed, or SSH with `-X`/`-Y` |
| Change DPI, figure size, colormap threshold, title text | edit the code (§12) — not a flag |
