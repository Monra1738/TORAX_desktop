# Project layout

```text
src/jaxa_udon3/
├── desktop/                 native PySide6 frontend
│   ├── app.py               entry point
│   ├── main_window.py       orchestration and interactions
│   ├── main_refresh.py      selective refresh paths
│   ├── panels.py            search, observation browser, Data & Layers
│   ├── inspector.py         contextual controls
│   ├── viewers.py           2D/Sky/workspace splitter
│   ├── viewer_3d.py         persistent PyVista actors + slice planes
│   ├── analysis.py          spectrum/slice analysis
│   ├── state.py             loaded/visible/slice state
│   ├── science_views.py     fast preview products
│   └── data_controller.py   bridge to scientific infrastructure
├── infrastructure/          DARTS, WCS, calibration, cache, exact products
├── domain/                  scientific/domain models
└── cli/                     optional local cache/bootstrap utilities
```

The old Trame/web frontend, `another_project_js` prototype, PostgreSQL services, and
container deployment are outside the maintained product. `another_project_js` is kept
as a reference for a later UI/visualization pass and is not packaged or tested.
