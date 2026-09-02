# Architecture

UDON3 Theme 2 is distributed as a native desktop application.

```text
PySide6 desktop UI
      │
      ├── PyVistaQt / VTK 3D actors and slice planes
      ├── PyQtGraph spectrum, ROI and 2D images
      │
      ▼
Desktop controllers/state
      │
      ▼
Existing scientific infrastructure
      ├── DARTS catalog / parquet sources
      ├── Sesame + WCS
      ├── instrument PI → keV calibration
      ├── deterministic preview/cache products
      └── exact energy/RGB products
```

The desktop frontend owns only interaction/render state. Scientific transformations stay
in the infrastructure modules.

Loaded observations are keyed by `mission/instrument/observation_id` and cached in
memory. The PyVista scene mirrors this with persistent per-observation actors so a
visibility toggle does not imply data I/O or a scene rebuild.
