# Desktop interaction model

## Default workspace

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ TORAX   target search   fixed RA/DEC/radius                 Search/Add  │
├───────────────┬──────────────────────────────────┬───────────────────────┤
│ Loaded data   │              3D                  │ Inspector             │
│ visibility    │                                  │                       │
│               ├──────────────────────────────────┤                       │
│ Events/Density/Voxels │         2D               │                       │
│               │ energy / selected slice / RGB   │                       │
│ Energy slices │ RA increases toward the left    │                       │
├───────────────┴──────────────────────────────────┴───────────────────────┤
│ Spectrum | Slice images | Selected slice profile                        │
├──────────────────────────────────────────────────────────────────────────┤
│ loaded / visible / all-preview / preview-in-band / selected energy       │
└──────────────────────────────────────────────────────────────────────────┘
```

The 3D and 2D panes share a draggable Qt splitter. The layout selector supports
`3D + 2D`, `3D only` and `2D only`; hiding a pane does not destroy its state.

## Observation lifecycle

`Loaded` and `Visible` are separate concepts.

- Adding an observation reads/caches its preview once.
- Hiding an observation keeps its cached arrays and 3D actor.
- Showing it again changes actor visibility immediately.
- New searches in the same sky region can append observations.
- A genuinely new target/search region is saved separately to avoid mixing
  incompatible sky selections; use **Workspaces** to return to it later.

## Saved target workspaces

After a 500 ms idle debounce, TORAX saves the region, observations and visibility,
energy and RGB controls, slices, ROI A/B, rendering/layout state, and camera to the
local DuckDB database. The latest workspace restores on launch. The **Workspaces**
menu lists prior target regions; opening one restores it without combining its data
with another target's projection.

## Energy controls

The translucent primary spectrum region is the **selected energy band**. It can be
dragged, stepped with Previous/Next, or moved automatically with a ping-pong scan.
The scan speed is selectable and saved. The raw histogram stays visible; **Smooth
curve** independently shows or hides the Gaussian display overlay.

The **Link spectrum band to 3D + 2D image** control is enabled by default. Dragging or
scanning the spectrum then updates the 3D points/voxels and 2D energy image together.
Disable the link to use **Filter 3D points / voxels** and **Filter 2D energy image**
independently; with both filters off, all loaded preview events (within the global VTK
point budget) and the full preview energy coverage are shown. The Sky-event point map
always shows all energies.

The 3D axes report absolute RA and DEC in degrees. Energy voxel size accepts values down
to **0.005 keV (5 eV)**, useful for thin XRISM/Resolve slices; the voxel grid still applies
its safety cell budget to keep interaction responsive.

Energy slices are separate scientific objects. Multiple slices can exist at the same
time. Each slice has lower/upper keV bounds, visibility, opacity and a 3D plane. Each
colored slice region can also be dragged directly on the spectrum.

The 3D view uses centered sky coordinates in arcminutes and a reversible energy-depth
transform. In **Inspector → Event display → 3D energy geometry**, set the reference energy
and increase **Depth scale** to reveal small Doppler shifts; the axis title records the
display multiplier while picked events continue to report physical keV. The **W49B Fe XXV
quick-look** preset prepares 6.700 keV reference, blue/red ±2.5 eV comparison bands, and
the published Resolve East/West observation IDs (300055010/300056010) for loading.

The Inspector can enable **ROI B** for a second manual RA/DEC rectangle. The spectrum
overlays ROI A in blue and ROI B in orange, supporting comparisons such as SN1006's
centre versus outer shell. Spectrum scale, smoothing visibility, scan speed, and band
application choices are saved with the workspace.

## Performance

The 3D viewer keeps one persistent actor per loaded observation. It does not clear the
whole PyVista scene when visibility changes. Event actors are sampled only for display;
loaded preview arrays remain available for spectrum/image work. Voxel grids use safe
cell caps and are cached per observation/settings. The interactive event budget is a
hard global cap shared across visible observations, including a 143-observation workspace.
