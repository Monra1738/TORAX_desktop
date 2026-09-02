# TORAX Theme 2 — Native Desktop X-ray Event Explorer

TORAX Theme 2 is a **desktop-only** JAXA/DARTS X-ray event exploration tool built with
**PySide6 + PyVista/VTK + PyQtGraph**. The existing scientific backend remains responsible
for DARTS catalog access, Sesame/WCS coordinates, instrument PI→keV calibration, caching,
and exact science products.

## Quick Start: Setup and Run

### Prerequisites

- Python **>= 3.11** (Python 3.11 or 3.12 recommended)
- An OpenGL-capable environment / graphics drivers for 3D PyVista/VTK rendering

---

### 1. Installation & Environment Setup

#### Windows (PowerShell / Command Prompt)

```powershell
# Create and activate virtual environment (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Or in Windows Command Prompt (cmd.exe):
# .venv\Scripts\activate.bat

# Upgrade packaging tools and install in editable mode
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

> [!TIP]
> If script execution is disabled in PowerShell, run:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`

#### Linux (Ubuntu / Debian / Fedora)

Ensure required OpenGL and Qt/xcb libraries are present:

```bash
# Ubuntu / Debian system dependencies
sudo apt update && sudo apt install -y python3-venv python3-pip libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0

# Fedora system dependencies:
# sudo dnf install python3-devel mesa-libGL mesa-libEGL libxkbcommon libxcb

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade packaging tools and install in editable mode
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
```

#### macOS

```bash
# Install Python 3.11 via Homebrew if needed
brew install python@3.11

# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Upgrade packaging tools and install in editable mode
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[dev]'
```

---

### 2. Launch the Application

Once your virtual environment is active:

```bash
# Standard desktop launch (entry point)
torax-desktop

# Or directly via the Python launcher (cross-platform)
python desktop_launcher.py
```

- **macOS shell shortcut:**
  ```bash
  ./scripts/run_desktop_macos.sh
  ```

- **Headless / No-3D fallback:**
  If testing on a remote server, CI, or machine without OpenGL display:
  ```bash
  torax-desktop --no-3d --screenshot var/exports/desktop_startup.png
  ```

---

### 3. Run Tests

```bash
# Run tests with pytest
pytest

# Or with unittest
python -m unittest discover -v tests
```

---

## Performance-first revision

This build is focused on making the UI feel immediate while keeping the scientific workflow
from Ken's Theme 2 requirements:

- loaded observations stay cached until explicitly removed;
- mission, instrument, and individual observation visibility can be toggled hierarchically;
- one cached PyVista actor is kept per observation;
- toggling a loaded observation only changes actor visibility — no parquet reload, PI conversion,
  WCS conversion, scene clear, or camera reset;
- a whole mission/instrument group is toggled with **one VTK render**, not one render per child;
- a fixed global 3D point budget is shared fairly across every visible observation;
- expensive spectrum/2D/3D changes are debounced while the user drags controls;
- repeated pandas visible/energy/ROI selections are cached;
- energy and RGB preview products use a bounded in-memory product cache;
- hidden 3D or 2D views are not recomputed;
- event actors are updated in place where PyVista permits instead of recreated;
- invisible observations are not rebuilt when the energy band changes;
- per-observation 3D point budgets remain stable when visibility changes;
- voxel grids are smaller by default, capped, thresholded, and cached per observation/settings;
- observation previews publish progressively through a failure-isolated three-worker session;
- transient DuckDB attachment/contention, network timeout, connection-reset, HTTP 429, and
  HTTP 5xx failures retry automatically up to four attempts with bounded backoff;
- shared on-disk DuckDB metadata transactions are serialized while parquet/network/WCS work
  remains concurrent, preventing the large-queue failures that previously disappeared on Retry;
- final observation-load exceptions and retry recoveries are recorded in
  `var/logs/observation_loading.log` with rotation;
- interactive previews are capped at 600,000 rows (15,000 per observation) and allocated fairly
  by mission, instrument, then observation;
- image smoothing is **Auto** by default and adapts to event density;
- spectrum binning is **Auto** by default; raw counts remain visible while an optional
  smoothed overlay makes weak structure easier to follow;
- Previous/Next and Auto Scan move the selected energy band with a saved speed and
  ping-pong at the available energy limits;
- the selected band is linked to the 3D points and 2D energy image by default, so dragging
  the spectrum immediately updates both; disable **Link spectrum band to 3D + 2D image**
  to use the independent **Filter 3D** and **Filter 2D** switches and show all events;
- slice images are displayed **next to one another**, not hidden behind separate tabs;
- live slice dragging updates only the changed slice row rather than rebuilding the slice list.
- target workspaces autosave to local DuckDB and restore after restarting the application.

A synthetic 500k-row state benchmark in the packaging environment reduced a repeated visible-frame
selection from ~8 ms on first evaluation to ~0.01 ms when cached. This is only a state-layer
microbenchmark; real end-to-end performance depends on the Mac, OpenGL, observations, and network/cache.

## Main scientific workspace

The central workspace supports:

- **3D + 2D** (default)
- **3D only**
- **2D only**

The 2D side can display:

- an all-energy image by default, or the selected energy-band image when enabled;
- selected energy slice;
- RGB composite;
- sky-event view.

The 3D camera state is preserved when the 3D panel is hidden and shown again.
Astronomical 2D images always display **RA increasing toward the left**.
The 3D RA and DEC axes use absolute degree coordinates rather than centre-relative offsets.

## Observation workflow

The search region (central sky coordinate + radius) stays fixed until a new search is made. Every
2D product uses that same fixed RA/DEC grid; zoom-out and pan stop at its boundary, including across
RA 0°/360°.
Loaded observations are separate from visible observations.

The loaded-data tree is:

```text
XRISM
  RESOLVE
    000129000
    000130000
  XTEND
    ...
SUZAKU
  XIS
    ...
ASCA
  GIS
    ...
  SIS
    ...
```

You can toggle a whole mission, one instrument, or one observation. Data stay in memory.
Changing visibility never reloads an observation. Adding search results to the current region
publishes each success immediately; individual failures can be inspected and retried without losing
successful observations. Selecting a genuinely different region creates a separate saved workspace.

## Saved workspaces

TORAX automatically saves the active target and viewport, selected/visible observations, pending and
failed references, spectrum scale, 2D zoom, energy and RGB controls, slices, ROI, layout, rendering
mode, and 3D camera in the local `darts_events.duckdb` cache database.
On restart it restores the most recently used workspace and reuses local preview products whenever
available. Choose another target workspace from the **Workspaces** menu; regions remain separate so
their coordinate projections are never mixed.

## Energy-band selection and scientific image display

The blue region on the spectrum selects an energy band. Use **Previous**, **Next**, or
**Auto Scan** to move it; Auto Scan preserves its width and reverses direction at the data
limits. In **Inspector → Global Energy Filter → Band Application**, the link is enabled by
default, so moving the band updates the 3D points/voxels and 2D energy image together.
Disable the link to use the 3D and 2D filter checkboxes independently; with both filters off,
all loaded preview events are shown within the safe point budget and all energies are
compressed into the 2D image. The Sky-event view is always an all-energy point map.

The Energy image and all slice-comparison images preserve their raw event-count
histograms. In **Inspector → Global Energy Filter → Image Display**, the
display-only palette, Linear/Square root/Log stretch, brightness, and contrast can be
changed immediately. Choose **No colormap (grayscale)** for a neutral intensity view:
positive counts run from black to white and zero-count sky remains black. Zero-count sky
is always black, including with Rainbow. These
controls do not reload observations or change WCS, spectra, RGB products, 3D events,
or the camera. RGB composite remains an independent three-energy-band product.

## Cas A workflow

The Search dialog has a **Quick target → Cas A** option using the coordinate in Ken's task:

- RA = **350.8584°**
- DEC = **+58.8113°**

Search with the ASCA, Suzaku, and XRISM instruments enabled, then add the observations you want.
XRISM observations **000129000** and **000130000** are specifically referenced in the Theme 2 task,
but the purpose of TORAX is to combine relevant ASCA + Suzaku + XRISM data rather than restrict
Cas A to XRISM alone.

Under **ENERGY SLICES**, choose:

`Cas A — ASCA literature bands`

This adds the four energy bands quoted in Ken's document:

- 1.55–1.75 keV — Low continuum
- 1.75–1.95 keV — Si He α
- 2.35–2.52 keV — S He α
- 3.93–6.23 keV — High continuum

Open **Analysis → Slice images** to see all active slice images side-by-side while the same slices
remain as independently movable planes in the 3D RA × DEC × Energy volume.

For a reproducible real-data check (and to save the rendered comparison artifacts), run from the
project root:

```bash
python scripts/check_cas_a_real_data.py
```

This downloads the two XRISM Resolve observations, a focused ASCA GIS/SIS pointing, and a Suzaku
XIS pointing into `var/data_cache`, then writes four PNG bands, an ASCA-only comparison sheet,
the extracted PDF reference figures, and `cas_a_manifest.json` under `var/exports/cas_a`.
`--no-download` verifies the existing cache without deliberately downloading missing raw files;
`--include-xtend` opts into the two much larger XRISM Xtend files.

For a headless startup/screenshot check on machines without a working VTK display, use
`torax-desktop --no-3d --screenshot var/exports/desktop_startup.png`.

For a three-channel Cas A view, select **RGB composite** in the 2D product picker, then choose
**Inspector → RGB COMPOSITE → Use Cas A ASCA RGB**.  This maps low continuum
(1.55–1.75 keV) to red, Si He α (1.75–1.95 keV) to green, and S He α (2.35–2.52 keV) to blue.
Changing any RGB centre, width, gain, brightness, or gamma updates the RGB composite and—when
the Event Display is set to **RGB energy bands**—the 3D and sky-event views as well.

## SN1006 workflow

The Search dialog has **Quick target → SN1006**, resolved through Sesame, and starts with a 60 arcmin
radius so the first search is appropriate for looking for the full remnant rather than only the small
Resolve field of view. Select Xtend/Suzaku for wide spatial coverage and Resolve when you need its
high spectral resolution.

Use the RA/DEC ROI tool on the 2D view to compare a central region with an outer-shell region. The
spectrum selector supports **Linear**, **Log Y**, and **Log–Log** display for checking the line-rich
centre versus the hard outer shell. Zero-count bins are safely omitted on logarithmic Y axes.

## Multiple energy slices

Slices are independent from the global display energy filter.

Each slice has:

- lower and upper energy;
- visibility;
- its own color;
- opacity;
- movable 3D plane;
- draggable region on the spectrum;
- corresponding RA/DEC image.

Dragging one slice does not rebuild the slice list or recompute unchanged slice images.

## Data size model

TORAX deliberately separates full/loaded science information from the interactive display:

- observation previews are loaded once and cached;
- the interactive 3D view has a bounded point budget;
- normal interactive images and spectra use all loaded preview rows;
- exact cache-backed products still use the backend exact-data paths;
- repeated labels use pandas categorical storage;
- PyVista receives compact float32 point coordinates for rendering.

### Load every matching observation and use every event

After a target/radius search, click **Load all N matching** in the observation browser.
This queues every parquet observation returned by that fixed sky-region search; no row selection is
required. After loading finishes, open **Inspector → Energy** and click
**All events → exact 2D image**. The exact calculation streams every spatially matching event with
finite, non-negative PI from every visible parquet file into the RA/DEC histogram and reports the
exact contributing event count. Negative PI sentinel rows are excluded because they have no physical
energy.

The rotatable 3D point cloud remains a deterministic, bounded preview because sending tens or
hundreds of millions of individual glyphs to VTK would exhaust desktop memory. This is not hidden:
the status bar separately reports full region-event counts and displayed preview rows. Use the exact
all-event image for full-data spatial verification and the 3D cloud for interactive exploration.

### Cas A literature comparison

The reproducible ASCA comparison uses observations **50018000** and **50018010** from
1993-08-01, the observations associated with Holt et al. (PASJ 46, L151–L155, 1994). The four
energy bands match the literature. TORAX's output is an exact raw-event histogram; the published
panels used PSF restoration to approximately 30 arcsec FWHM. Consequently, a raw map should not be
expected to be pixel-identical to the restored paper figure.

The status bar distinguishes loaded region events, visible region events, all preview rows,
and preview rows in the selected band so downsampling is not mistaken for data loss.

## Scientific functionality

- decimal RA/DEC, sexagesimal RA/DEC, or target name through Sesame;
- DARTS/TORAX catalogs for ASCA, Suzaku, Hitomi and XRISM;
- ASCA GIS/SIS, Suzaku XIS, Hitomi SXS/SXI/HXI, XRISM Resolve/Xtend;
- instrument-aware PI→keV conversion;
- rotatable/zoomable RA × DEC × Energy PyVista scene;
- labeled RA, DEC and Energy axes;
- linked RA/DEC image and energy spectrum;
- raw spectrum plus optional smooth overlay, Reset zoom, and Linear/Log Y/Log–Log scales;
- draggable global energy band with Previous/Next and automatic ping-pong scanning;
- independent opt-in selected-band filtering for 3D and the 2D energy image;
- RA/DEC rectangle driving the spectrum;
- red(low energy) → blue(high energy) scientific color ramp;
- configurable RGB energy channels;
- Events, Density, and Voxels 3D representations with independently smoothed,
  thresholded energy cubes;
- independent event, voxel, image and spectrum controls;
- multiple simultaneous energy slices;
- exact energy/RGB products;
- CSV/Parquet preview export and screenshots.

## Environment and Graphics Notes

Qt/PyVista window rendering requires an OpenGL-capable desktop environment with active display output.
Headless or virtual environments without GPU/OpenGL acceleration can validate science logic, tests, and data caches with `pytest`, or generate offscreen screenshots using:

```bash
torax-desktop --no-3d --screenshot var/exports/desktop_startup.png
```

