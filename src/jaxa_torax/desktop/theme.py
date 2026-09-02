"""Visual system for the native TORAX scientific desktop application."""

from __future__ import annotations

JAXA_BLUE = "#0284c7"
JAXA_BLUE_BRIGHT = "#38bdf8"
NAVY_980 = "#05070b"
NAVY_950 = "#080d14"
NAVY_925 = "#0a1019"
NAVY_900 = "#0d121d"
NAVY_875 = "#101725"
NAVY_850 = "#121a2a"
NAVY_825 = "#172033"
NAVY_800 = "#1c273e"
NAVY_750 = "#22324a"
BORDER = "#2a394f"
BORDER_SOFT = "#1e293b"
TEXT = "#f1f5f9"
MUTED = "#94a3b8"
MUTED_2 = "#64748b"
SUCCESS = "#10b981"
WARNING = "#f59e0b"
DANGER = "#f43f5e"
VIEWPORT = "#030712"
PLOT_BACKGROUND = "#070b12"
PLOT_TEXT = "#cbd5e1"
PLOT_GRID = "#334155"
AXIS_RA = "#fb7185"
AXIS_DEC = "#34d399"
AXIS_ENERGY = "#60a5fa"
PANEL_RADIUS = 8

MISSION_COLORS = {
    "asca": "#4f83ff",
    "suzaku": "#39d49a",
    "hitomi": "#ff7080",
    "xrism": "#2aa8ff",
}

# Scientific energy color ramp: lowest energy red, highest energy blue.
ENERGY_COLORS = ["#ff2415", "#ff8600", "#ffe21a", "#37d65b", "#12c7d6", "#1f6dff", "#3f32ff"]

APP_QSS = f"""
* {{
    outline: none;
}}
QMainWindow, QWidget {{
    background: {NAVY_980};
    color: {TEXT};
    /* Qt does not resolve the CSS-only -apple-system family on macOS. */
    font-family: "Helvetica Neue";
    font-size: 12px;
}}
QToolTip {{
    background: {NAVY_800}; color: {TEXT}; border: 1px solid {JAXA_BLUE_BRIGHT};
    padding: 6px 8px; border-radius: 6px;
}}
QToolBar {{
    background: {NAVY_900};
    border: 0;
    border-bottom: 1px solid {BORDER};
    spacing: 0;
    padding: 0;
}}
QToolBar::separator {{ width: 1px; background: {BORDER_SOFT}; margin: 8px 5px; }}
QFrame#topBar, QFrame#statusStrip {{
    background: {NAVY_900};
    border: 0;
}}
QFrame#panel, QFrame#card, QGroupBox, QTabWidget::pane {{
    background: {NAVY_900};
    border: 1px solid {BORDER};
    border-radius: {PANEL_RADIUS}px;
}}
QFrame#dockTitle {{
    background: {NAVY_950};
    border-bottom: 1px solid {BORDER};
}}
QLabel#appTitle {{ font-size: 20px; font-weight: 750; letter-spacing: 1px; }}
QLabel#subtitle {{ color: {MUTED}; font-size: 12px; }}
QLabel#sectionTitle {{ color: {MUTED}; font-size: 10px; font-weight: 750; letter-spacing: 0.8px; }}
QLabel#panelTitle {{ color: {TEXT}; font-size: 13px; font-weight: 700; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#tiny {{ color: {MUTED}; font-size: 10px; }}
QLabel#tinyValue {{ color: {TEXT}; font-size: 10px; font-weight: 650; }}
QLabel#value {{ color: {TEXT}; font-weight: 650; }}
QPushButton, QToolButton {{
    background: {NAVY_825}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 7px; padding: 7px 11px; font-weight: 550;
}}
QPushButton:hover, QToolButton:hover {{ background: {NAVY_800}; border-color: {JAXA_BLUE_BRIGHT}; }}
QPushButton:pressed, QToolButton:pressed {{ background: #0c4a6e; }}
QPushButton:checked {{ background: #075985; border-color: {JAXA_BLUE_BRIGHT}; color: #ffffff; }}
QPushButton#primary {{
    background: {JAXA_BLUE}; border-color: {JAXA_BLUE_BRIGHT}; color: white; font-weight: 700;
}}
QPushButton#primary:hover {{ background: #0369a1; }}
QPushButton#danger {{ background:#3b1520; border-color:#7f1d32; color:#fecdd3; }}
QPushButton#modeButton {{
    background:{NAVY_950}; border-radius:6px; padding:7px 8px; font-weight:650;
}}
QPushButton#modeButton:checked {{
    background:#075985; border-color:{JAXA_BLUE_BRIGHT}; color:#ffffff;
}}
QPushButton:disabled, QToolButton:disabled {{ color: {MUTED_2}; background: #0b1724; border-color:#1b3146; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {NAVY_825}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 7px; padding: 6px 8px; min-height: 25px;
    selection-background-color: {JAXA_BLUE};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ background:{NAVY_800}; border-color: {JAXA_BLUE_BRIGHT}; }}
QComboBox QAbstractItemView {{
    background: {NAVY_825}; color: {TEXT}; border:1px solid {BORDER}; selection-background-color: #164b77;
}}
QCheckBox {{ spacing: 6px; color:{TEXT}; }}
QCheckBox::indicator {{ width: 15px; height: 15px; }}
QRadioButton {{ spacing: 6px; }}
QSlider::groove:horizontal {{ background: #263449; height: 4px; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {JAXA_BLUE_BRIGHT}; border:2px solid #dff6ff; width: 12px; margin: -6px 0; border-radius: 7px; }}
QSlider::sub-page:horizontal {{ background: {JAXA_BLUE}; border-radius:2px; }}
QTreeWidget, QTableWidget, QTableView, QListWidget {{
    background: {NAVY_900}; alternate-background-color: {NAVY_875}; color: {TEXT};
    border: 1px solid {BORDER}; gridline-color: {BORDER_SOFT}; selection-background-color: #0c4a6e;
    selection-color: {TEXT}; border-radius:7px;
}}
QTreeWidget::item, QListWidget::item {{ padding:4px 4px; min-height:22px; border-radius:4px; }}
QTreeWidget::item:hover, QListWidget::item:hover {{ background:#172a40; }}
QHeaderView::section {{
    background: {NAVY_875}; color: {MUTED}; border: 0; border-bottom: 1px solid {BORDER};
    padding: 6px; font-weight: 650;
}}
QTabWidget::pane {{ top:-1px; }}
QTabBar::tab {{
    background: {NAVY_950}; color: {MUTED}; border: 0;
    border-bottom:2px solid transparent; padding: 8px 16px; min-width:75px;
}}
QTabBar::tab:selected {{ background: {NAVY_825}; color: {JAXA_BLUE_BRIGHT}; border-bottom-color: {JAXA_BLUE_BRIGHT}; }}
QTabBar::tab:hover {{ color: {TEXT}; background:{NAVY_875}; }}
QScrollArea {{ border: 0; background:transparent; }}
QScrollBar:vertical {{ background:{NAVY_950}; width:9px; margin:0; }}
QScrollBar::handle:vertical {{ background:#334155; min-height:28px; border-radius:4px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QProgressBar {{
    border: 1px solid {BORDER}; border-radius: 4px; background: {NAVY_825}; text-align: center;
    color:{TEXT}; min-height:12px;
}}
QProgressBar::chunk {{ background: {JAXA_BLUE}; border-radius: 3px; }}
QMenu {{ background: {NAVY_825}; color: {TEXT}; border: 1px solid {BORDER}; padding:4px; }}
QMenu::item {{ padding:6px 24px 6px 8px; border-radius:3px; }}
QMenu::item:selected {{ background: #164b77; }}
QStatusBar {{ background: {NAVY_900}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QDockWidget {{ color:{TEXT}; titlebar-close-icon:none; titlebar-normal-icon:none; }}
QDockWidget::title {{ background:{NAVY_900}; border-bottom:1px solid {BORDER}; padding:5px; }}
QDialog {{ background:{NAVY_950}; }}
QMessageBox {{ background:{NAVY_950}; }}
"""

# Desktop-specific refinements kept separate from the shared palette above.
APP_QSS += f"""
QFrame#workspaceBar {{
    background: {NAVY_950};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QLabel#qualityBadge {{
    color: #cfe8ff;
    background: #10385a;
    border: 1px solid #285f89;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 10px;
    font-weight: 750;
}}
QDockWidget > QWidget {{ background: {NAVY_950}; }}
"""
