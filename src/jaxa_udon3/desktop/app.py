from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UDON3 native DARTS X-ray explorer")
    parser.add_argument("--screenshot", help="save a desktop screenshot after startup and exit")
    parser.add_argument(
        "--no-3d",
        action="store_true",
        help="disable VTK for headless startup/screenshot checks",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.no_3d:
        os.environ["UDON3_DISABLE_3D"] = "1"
    if "UDON3_DATA_ROOT" not in os.environ:
        project_root = Path(__file__).resolve().parents[3]
        project_var = project_root / "var"
        if not getattr(sys, "frozen", False) and project_var.exists():
            os.environ["UDON3_DATA_ROOT"] = str(project_var)
        else:
            os.environ["UDON3_DATA_ROOT"] = str(
                Path.home() / "Library" / "Application Support" / "UDON3"
            )
    os.environ.setdefault("PYVISTA_QT_INTERACTOR", "1")
    try:
        import pyqtgraph as pg
        from PySide6.QtCore import QThreadPool, QTimer
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
    except ImportError as error:
        print(
            "UDON3 desktop dependencies are missing. Install with:\n"
            "  python -m pip install -e '.[dev]'\n\n"
            f"Import error: {error}",
            file=sys.stderr,
        )
        return 2

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    from jaxa_udon3.desktop.main_window import MainWindow
    from jaxa_udon3.desktop.screenshot import save_widget_screenshot

    app = QApplication.instance() or QApplication(sys.argv)
    # Avoid Qt's slow fallback through the non-existent CSS generic “Sans
    # Serif” family on macOS. Helvetica Neue is available on supported macOS.
    app.setFont(QFont("Helvetica Neue", 12))
    app.setApplicationName("UDON3")
    app.setOrganizationName("JAXA / ISAS")
    window = MainWindow()
    window.show()

    if args.screenshot:
        destination = Path(args.screenshot).expanduser().resolve()

        def capture():
            saved = save_widget_screenshot(window, destination)
            print(f"Screenshot saved to {saved} ({saved.stat().st_size:,} bytes)")
            app.quit()

        QTimer.singleShot(1200, capture)
    exit_code = 0
    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        window.close()
        pool = QThreadPool.globalInstance()
        pool.clear()
        pool.waitForDone(10_000)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
