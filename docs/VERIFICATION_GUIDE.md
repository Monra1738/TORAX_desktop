# UDON3 Theme 2 Verification Guide

This guide is the acceptance checklist for Ken's Theme 2 requirements. Complete the
offline checks first, then the real-data checks, then the interactive checklist. Do not
claim a scientific result from a screenshot unless its observation manifest is saved.

## 1. Install and launch

From the project root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

Normal application:

```bash
udon3-desktop
```

Headless smoke screenshot:

```bash
QT_QPA_PLATFORM=offscreen udon3-desktop --no-3d --screenshot var/exports/startup.png
```

The screenshot must be non-empty and show the UDON3 title, Data & Layers, Inspector,
Analysis, and the 3D/2D workspace areas.

## 2. Data validation

Validate all cached observations and write a machine-readable report:

```bash
python scripts/validate_real_data.py \
  --data-root var \
  --json var/exports/data_validation.json
```

Require at least one record for every supported instrument pair:

```bash
python scripts/validate_real_data.py --data-root var --require-all
```

The report must contain finite TIME/PI/X/Y/RA/DEC values, valid WCS projection,
RA in `[0, 360)`, DEC in `[-90, 90]`, non-negative finite calibrated energies,
matching total/preview counts, and all eight instrument pairs when `--require-all`
is used.

### Large observation queues

Load a large result selection and let the progress bar finish. Transient cache/network
problems are retried automatically up to four attempts; the status bar shows the current
retry. Only exhausted or permanent errors appear under **View failures**.

For a failure investigation, inspect:

```bash
tail -n 200 var/logs/observation_loading.log
```

The log includes observation key, attempt number, transient classification, final traceback,
and successful recovery attempt. Pressing **Retry failed** remains available for genuinely
external failures, but it must not be needed for normal DuckDB concurrency.

## 3. Requirement-by-requirement checklist

### Position and catalog

1. Click **Search / Add**. Choose **RA / DEC degrees**, enter RA, DEC, and radius.
   Search and confirm the fixed-region badge and Dataset inspector values.
2. Repeat with **RA / DEC sexagesimal** using `23:23:24` and `+58:48:54`.
3. Repeat with **Target name (Sesame)** using `Cas A`. Confirm the resolved coordinates.
4. Select each of ASCA GIS/SIS, Suzaku XIS, Hitomi SXS/SXI/HXI, and XRISM Resolve/Xtend.
   Confirm the result table includes mission, instrument, and observation ID.

### Events, axes, and visibility

5. Add one observation. In **Event Display**, choose **Events**. Rotate, zoom, and
   reset the 3D camera. Confirm the axes are labeled **Right Ascension (deg)**,
   **Declination (deg)**, and **Energy (keV)**, with absolute RA/DEC values rather than
   centre-relative offsets.
6. In **Data & Layers**, toggle one observation, one instrument, and one mission.
   Confirm visibility changes without a reload or camera reset.
7. Add observations from multiple missions and confirm the status bar distinguishes
   loaded, visible, and displayed preview rows.
8. Select at least ten observations in one operation. Confirm transient messages retry
   automatically and the final loaded/failed totals remain stable after restarting UDON3.
   For a release stress check, run the automated 500-record queue and DuckDB-cycle tests:

   ```bash
   pytest -q \
     tests/test_progressive_scale_viewport.py::ProgressiveLoadingTests::test_five_hundred_record_queue_completes_without_dropping_items \
     tests/test_data_contracts.py::DataContractTests::test_five_hundred_concurrent_metadata_cycles_share_one_database
   ```

### Images and spectra

9. Set the 2D product to **Energy**. Confirm the RA axis increases toward the left.
10. Confirm **Link spectrum band to 3D + 2D image** is checked. Drag the blue spectrum
   band and confirm both the 3D points and 2D Energy image update with the selected band.
   Disable the link, leave both filter checkboxes off, and confirm all preview points and
   all energies are shown. Select **Sky events** and confirm it also shows all events.
11. With the link disabled, enable **Filter 3D points / voxels by selected band** only.
    Move the band and confirm only 3D follows it. Then reverse the checkboxes and confirm
    only the 2D Energy image follows the selected band.
12. Click **Previous** and **Next**. Click **Auto Scan**, test 1/2/4/8× per second,
    and confirm the fixed-width band reverses at both spectrum limits. **Pause** must stop it.
13. Confirm the raw step histogram remains visible. Toggle **Smooth curve**, change
    Smooth from 0 to several bins, and use **Reset zoom**. Smoothing must not alter raw counts.
14. Enable **Use RA/DEC rectangle for spectrum**, enter independent RA/DEC bounds,
    and confirm only the spectrum changes.
15. In **Voxel display**, set **Energy voxel** to `0.005 keV` (5 eV). Confirm the value is
    accepted and switch to Voxel mode; the safety cell budget may coarsen a very large grid.
16. Switch spectrum scale among **Linear**, **Log Y**, and **Log–Log**. Confirm zero
    bins do not create invalid log values.

### RGB and display controls

17. Choose **RGB composite**. Set red, green, and blue centers/widths and confirm the
    shared image grid remains aligned.
18. Change RGB gain, brightness, gamma, and event color mode **RGB energy bands**.
19. In **Image Display**, test grayscale/no-colormap, Linear, Square root, and Log
    stretch, plus brightness, contrast, and smoothing. Confirm zero-count sky is black.

### Voxels and spatial spectral variation

20. Choose **Voxels**. Change spatial voxel, energy voxel, spatial smoothing, energy
    smoothing, and density threshold. Confirm the scene remains bounded and responsive.
21. In **Voxels**, confirm the cube faces, energy colormap, low-density cutoff, and
    independent spatial/energy smoothing make the volume readable.
22. Add multiple energy slices. Verify each slice has independent lower/upper bounds,
    color, opacity, visibility, plane, points, and side-by-side image.

## 4. Cas A acceptance run

1. Choose **Quick target → Cas A**.
2. In the result browser, click **Load all N matching** to test every returned observation, or
   select ASCA observations **50018000** and **50018010** for the 1994-paper comparison.
3. Open **Inspector → Energy** (already open by default), click **All events → exact 2D image**, and confirm the image title
   says **EXACT** and its event count equals the full-data result rather than the preview-row count.
4. Choose **Cas A — ASCA literature bands** under **ENERGY SLICES**.
5. Open **Analysis → Slice images** and save a screenshot of all four bands.
6. Choose **RGB composite → Use Cas A ASCA RGB** and save an RGB screenshot.
7. Run:

```bash
python scripts/check_cas_a_real_data.py --no-download
```

Record the generated manifest. Confirm `same_observation_ids` and `same_energy_bands` are true,
and `same_restoration_pipeline` is false: the paper figure is a restored product, while this check
is an exact raw-event map.

## 5. SN1006 acceptance run

1. Choose **Quick target → SN1006** and retain its wide search radius.
2. Add XRISM Xtend and Suzaku XIS for the full remnant; add Resolve for detailed spectra.
3. Capture one image with Xtend/Suzaku and one Resolve image, recording observation IDs.
4. Enable ROI A for the central region and ROI B for the outer shell.
5. Compare spectra in **Log–Log** mode and save screenshots.
6. Confirm the central region shows stronger line structure and the outer shell is
   brighter above approximately 4 keV. Treat this as unverified until the real-data
   manifest and figures are reviewed.

## 6. Evidence and release checklist

For every completed requirement, retain:

- screenshot filename;
- observation manifest and source URLs/checksums;
- target coordinates/radius;
- energy/RGB/voxel/smoothing settings;
- ROI and slice definitions;
- test command and pass result.

Before sharing with Ken/Nakahira-san, run `pytest -q`, Ruff, compilation, the data
validation script, Cas A validation, and the headless startup screenshot from a clean
environment. `another_project_js` is reference-only and must not be packaged.
