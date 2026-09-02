#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --windowed --name TORAX --paths src \
  --collect-all pyvista --collect-all pyvistaqt --collect-all pyqtgraph \
  --collect-all PySide6 --collect-all astropy --collect-all pyarrow \
  desktop_launcher.py
printf '\nBuilt: %s/dist/TORAX.app\n' "$PWD"
