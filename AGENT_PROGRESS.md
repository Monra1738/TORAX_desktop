# UDON3 Theme 2 progress

## Large observation loading reliability — completed

- Diagnosed saved failures as DuckDB mixed-configuration and simultaneous file-attachment
  conflicts, not corrupt observation data.
- Standardized shared database connection configuration and serialized short metadata
  transactions while keeping network, parquet, WCS, and calibration work concurrent.
- Added four-attempt bounded automatic retry for transient DuckDB/network/HTTP failures.
- Added rotating full-trace diagnostics at `var/logs/observation_loading.log`.
- Reduced large-batch cache maintenance from every product to every 25 products.

Validation: a 500-record session queue dropped no items; 500 concurrent DuckDB metadata
cycles had no conflicts; 500 real cached parquet previews (3,606,394 rows) all loaded;
the real workspace recovered 128 failed observations and now restores 143 loaded / 0 failed.
Full suite: 156 tests and 20 subtests.

## Ken review follow-up (2026-08-28) — completed

- 3D event, voxel, slice and top-image geometry now use absolute RA/DEC degree coordinates;
  RA seam handling remains contiguous around 0°/360°.
- The lower spectrum's draggable/scan band is linked to the upper 3D and 2D products by
  default, with an explicit switch for the all-event independent-filter workflow.
- Energy voxel width now accepts 0.005 keV (5 eV) for XRISM/Resolve thin slices, while the
  existing cell-budget guard prevents oversized grids from freezing the UI.

## Smooth spectrum, automatic scan, and linked viewer interaction — completed

- Preserved the exact raw histogram and added an independently visible Gaussian smooth curve.
- Added Reset zoom, Previous/Next, width lock, automatic ping-pong scan, Pause, and 1/2/4/8× speed.
- Restored linked spectrum interaction: the selected band drives both 3D and the 2D Energy
  image by default. Disable the link to retain independent filters and all-event display.
- Kept the Sky-event map all-energy and retained the safe global VTK point budget.
- Saved smooth visibility, scan speed, and both filter choices in target workspaces.
- Updated the opening instructions and requirement-by-requirement verification guide.

Validation: 158 tests and 20 subtests passed; Ruff and compilation passed; calibration-aware
Cas A generation passed for the 1993 ASCA GIS/SIS pointings, Suzaku XIS, XRISM Resolve and
XRISM Xtend. The exact all-event product streamed 104,382,998 valid events with matching
histogram sum; headless application screenshots were visually inspected.

## Progressive loading, spectrum scales, and fixed sky viewport — completed

- Added failure-isolated progressive observation sessions with three concurrent workers,
  immediate tree/3D publication, coalesced aggregate refreshes, cancellation, failure details,
  retry, deduplication, and sorting-safe catalog identity.
- Bounded combined previews to 600,000 rows, per observation to 15,000 rows, and retained the
  existing 160,000-point global 3D display budget; exact products still use source records.
- Replaced the Log–Log boolean with Linear, Log Y, and Log–Log spectrum modes, including safe
  zero-bin masking and physical-keV adapters for global and slice regions.
- Added one canonical unwrapped RA/DEC viewport shared by all 2D products, bounded pan/zoom,
  normalized RA labels, full-grid empty products, and contained/restored ROIs.
- Persisted spectrum scale, viewport, bounded 2D zoom, and failed/pending references with legacy
  `spectrum_log_log` migration.

Validation: full suite passed (124 tests, 19 subtests), plus a headless mixed-success main-window
integration smoke test.

## Scalar image display revision — completed

- Kept the native PySide6/PyQtGraph/PyVistaQt/DuckDB application architecture.
- Added display-only scalar palettes, stretch, brightness, and contrast for energy and slice images.
- Guaranteed black zero-count backgrounds for all palettes.
- Kept RGB, 3D event colours, spectra, WCS, and cached count histograms independent.
- Persisted scalar display settings in saved workspaces with safe defaults for older state.

Changed: `science_views.py`, `viewers.py`, `inspector.py`, `main_refresh.py`,
`main_window.py`, `state.py`, `workspace_persistence.py`, tests, and README.

Validation: full suite passed (108 tests), including legacy-workspace defaults and four-slice display-cache isolation.

## Native startup and energy interaction fixes — completed

- Added explicit Lower/Upper energy sliders, synchronized with precise keV fields and the draggable spectrum region.
- Fixed the slice-region Qt callback so it retains a string slice ID instead of forwarding a `LinearRegionItem`.
- Attached mission/instrument provenance to each loaded preview and added a 3D fallback for restored legacy previews.
- Removed the unsupported `-apple-system` stylesheet family and set the native macOS application font explicitly.

Validation: full suite passed (111 tests).

## Native-only code hygiene revision — completed

- Removed obsolete PostgreSQL worker/admin/import services, migrations, and dependencies.
- Retained local DuckDB catalog/cache/workspace functionality and native desktop entry points.
- Added a lazy explicit science facade, modernized imports/types, and made Ruff checks deterministic.
- Documented `another_project_js` as reference-only for a future UI/visualization pass.

Validation: 135 tests and 20 subtests passed; Ruff and compile checks passed; headless `--no-3d`
startup screenshot passed.
