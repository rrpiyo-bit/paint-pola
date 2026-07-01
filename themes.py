"""テーマ（デザイン）定義。QApplication.setStyleSheet() に渡す QSS を返す。"""

THEMES: dict[str, dict] = {}


def _register(key: str, label: str, qss: str):
    THEMES[key] = {"label": label, "qss": qss}


# ── デフォルト ────────────────────────────────────────────────────────────────
_register("default", "デフォルト", "")


# ── ダークモード ──────────────────────────────────────────────────────────────
_register("dark", "ダークモード", """
QMainWindow, QDialog {
    background-color: #1e1e1e;
    color: #d4d4d4;
}
QMenuBar {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border-bottom: 1px solid #444;
}
QMenuBar::item:selected {
    background-color: #3d3d3d;
}
QMenu {
    background-color: #2d2d2d;
    color: #d4d4d4;
    border: 1px solid #555;
}
QMenu::item:selected {
    background-color: #094771;
}
QMenu::separator {
    height: 1px;
    background: #444;
    margin: 4px 8px;
}
QWidget {
    background-color: #252526;
    color: #d4d4d4;
}
QPushButton {
    background-color: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #555;
    padding: 4px 10px;
    border-radius: 3px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #4a4a4a;
    border-color: #666;
}
QPushButton:pressed {
    background-color: #333;
}
QPushButton:checked {
    background-color: #094771;
    border-color: #1177bb;
}
QLabel {
    color: #d4d4d4;
    background: transparent;
}
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: #3c3c3c;
    color: #d4d4d4;
    border: 1px solid #555;
    padding: 2px 4px;
    border-radius: 2px;
}
QComboBox::drop-down {
    border: none;
    background: #4a4a4a;
}
QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    color: #d4d4d4;
    selection-background-color: #094771;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #555;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -5px 0;
    background: #0e7ad4;
    border-radius: 7px;
}
QScrollArea {
    background: #1e1e1e;
    border: none;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #2d2d2d;
    border: none;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #555;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0; width: 0;
}
QTabWidget::pane {
    border: 1px solid #444;
    background: #252526;
}
QTabBar::tab {
    background: #2d2d2d;
    color: #999;
    padding: 6px 14px;
    border: 1px solid #444;
    border-bottom: none;
}
QTabBar::tab:selected {
    background: #252526;
    color: #d4d4d4;
}
QCheckBox {
    color: #d4d4d4;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #555;
    background: #3c3c3c;
    border-radius: 2px;
}
QCheckBox::indicator:checked {
    background: #0e7ad4;
    border-color: #1177bb;
}
QGroupBox {
    color: #d4d4d4;
    border: 1px solid #444;
    margin-top: 8px;
    padding-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    color: #aaa;
}
QStatusBar {
    background: #2d2d2d;
    color: #999;
    border-top: 1px solid #444;
}
QFrame[frameShape="4"], QFrame[frameShape="5"] {
    color: #444;
}
""")


# ── Windows 95 風 ─────────────────────────────────────────────────────────────
_register("win95", "Windows 95 風", """
QMainWindow, QDialog {
    background-color: #c0c0c0;
    color: #000;
    font-family: "MS Gothic", "MS PGothic", monospace;
    font-size: 12px;
}
QMenuBar {
    background-color: #c0c0c0;
    color: #000;
    border-bottom: 1px solid #808080;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
QMenuBar::item:selected {
    background-color: #000080;
    color: #fff;
}
QMenu {
    background-color: #c0c0c0;
    color: #000;
    border: 2px outset #dfdfdf;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
QMenu::item:selected {
    background-color: #000080;
    color: #fff;
}
QMenu::separator {
    height: 2px;
    background: #808080;
    margin: 2px 4px;
    border-bottom: 1px solid #fff;
}
QWidget {
    background-color: #c0c0c0;
    color: #000;
    font-family: "MS Gothic", "MS PGothic", monospace;
    font-size: 12px;
}
QPushButton {
    background-color: #c0c0c0;
    color: #000;
    border: 2px outset #dfdfdf;
    padding: 3px 8px;
    min-height: 18px;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
QPushButton:hover {
    background-color: #d0d0d0;
}
QPushButton:pressed {
    border-style: inset;
    background-color: #b0b0b0;
}
QPushButton:checked {
    border-style: inset;
    background-color: #a0a0a0;
}
QLabel {
    color: #000;
    background: transparent;
}
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: #fff;
    color: #000;
    border: 2px inset #808080;
    padding: 1px 3px;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
QComboBox::drop-down {
    border: 2px outset #dfdfdf;
    background: #c0c0c0;
    width: 16px;
}
QComboBox QAbstractItemView {
    background-color: #fff;
    color: #000;
    selection-background-color: #000080;
    selection-color: #fff;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #808080;
    border: 1px inset #808080;
}
QSlider::handle:horizontal {
    width: 11px;
    height: 20px;
    margin: -8px 0;
    background: #c0c0c0;
    border: 2px outset #dfdfdf;
}
QScrollArea {
    background: #808080;
    border: 2px inset #808080;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #c0c0c0;
    border: none;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #c0c0c0;
    border: 2px outset #dfdfdf;
    min-height: 16px;
}
QScrollBar::add-line, QScrollBar::sub-line {
    background: #c0c0c0;
    border: 2px outset #dfdfdf;
    height: 16px; width: 16px;
}
QTabWidget::pane {
    border: 2px inset #808080;
    background: #c0c0c0;
}
QTabBar::tab {
    background: #c0c0c0;
    color: #000;
    padding: 4px 10px;
    border: 2px outset #dfdfdf;
    border-bottom: none;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
QTabBar::tab:selected {
    background: #d4d0c8;
    border-bottom: 1px solid #d4d0c8;
}
QCheckBox {
    color: #000;
    spacing: 4px;
}
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 2px inset #808080;
    background: #fff;
}
QCheckBox::indicator:checked {
    image: none;
    background: #fff;
    border: 2px inset #808080;
}
QGroupBox {
    color: #000;
    border: 2px groove #808080;
    margin-top: 8px;
    padding-top: 12px;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
}
QStatusBar {
    background: #c0c0c0;
    color: #000;
    border-top: 2px groove #808080;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
""")


# ── 16bit レトロ風 ────────────────────────────────────────────────────────────
_register("retro16", "16bit レトロ風", """
QMainWindow, QDialog {
    background-color: #2c2137;
    color: #f0d8a8;
    font-family: "MS Gothic", "MS PGothic", "Courier New", monospace;
    font-size: 12px;
}
QMenuBar {
    background-color: #442255;
    color: #f0d8a8;
    border-bottom: 2px solid #66aacc;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
QMenuBar::item:selected {
    background-color: #66aacc;
    color: #1a0a2e;
}
QMenu {
    background-color: #3a1a4a;
    color: #f0d8a8;
    border: 2px solid #66aacc;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
QMenu::item:selected {
    background-color: #66aacc;
    color: #1a0a2e;
}
QMenu::separator {
    height: 2px;
    background: #66aacc;
    margin: 2px 6px;
}
QWidget {
    background-color: #2c2137;
    color: #f0d8a8;
    font-family: "MS Gothic", "MS PGothic", "Courier New", monospace;
    font-size: 12px;
}
QPushButton {
    background-color: #442255;
    color: #f0d8a8;
    border: 2px solid #66aacc;
    padding: 3px 8px;
    min-height: 18px;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
QPushButton:hover {
    background-color: #553366;
    border-color: #88ccee;
    color: #fff;
}
QPushButton:pressed {
    background-color: #331144;
    border-style: inset;
}
QPushButton:checked {
    background-color: #66aacc;
    color: #1a0a2e;
}
QLabel {
    color: #f0d8a8;
    background: transparent;
}
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: #1a0a2e;
    color: #88ee88;
    border: 2px solid #66aacc;
    padding: 1px 3px;
    font-family: "MS Gothic", "MS PGothic", monospace;
    selection-background-color: #66aacc;
    selection-color: #1a0a2e;
}
QComboBox::drop-down {
    border: 2px solid #66aacc;
    background: #442255;
    width: 16px;
}
QComboBox QAbstractItemView {
    background-color: #1a0a2e;
    color: #88ee88;
    selection-background-color: #66aacc;
    selection-color: #1a0a2e;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #1a0a2e;
    border: 1px solid #66aacc;
}
QSlider::handle:horizontal {
    width: 12px;
    height: 18px;
    margin: -6px 0;
    background: #ee8844;
    border: 2px solid #ffcc66;
}
QScrollArea {
    background: #1a0a2e;
    border: 2px solid #442255;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #2c2137;
    border: none;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #66aacc;
    min-height: 16px;
}
QScrollBar::add-line, QScrollBar::sub-line {
    background: #442255;
    border: 1px solid #66aacc;
    height: 14px; width: 14px;
}
QTabWidget::pane {
    border: 2px solid #66aacc;
    background: #2c2137;
}
QTabBar::tab {
    background: #442255;
    color: #aaa;
    padding: 4px 10px;
    border: 2px solid #66aacc;
    border-bottom: none;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
QTabBar::tab:selected {
    background: #2c2137;
    color: #f0d8a8;
    border-bottom: 2px solid #2c2137;
}
QCheckBox {
    color: #f0d8a8;
    spacing: 4px;
}
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 2px solid #66aacc;
    background: #1a0a2e;
}
QCheckBox::indicator:checked {
    background: #ee8844;
    border-color: #ffcc66;
}
QGroupBox {
    color: #f0d8a8;
    border: 2px solid #66aacc;
    margin-top: 8px;
    padding-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    color: #66aacc;
}
QStatusBar {
    background: #442255;
    color: #88ee88;
    border-top: 2px solid #66aacc;
    font-family: "MS Gothic", "MS PGothic", monospace;
}
""")


# ── パステル ──────────────────────────────────────────────────────────────────
_register("pastel", "パステル", """
QMainWindow, QDialog {
    background-color: #fef6f0;
    color: #5a4a42;
}
QMenuBar {
    background-color: #fce4ec;
    color: #5a4a42;
    border-bottom: 1px solid #f8bbd0;
}
QMenuBar::item:selected {
    background-color: #f8bbd0;
    color: #5a4a42;
}
QMenu {
    background-color: #fff3e0;
    color: #5a4a42;
    border: 1px solid #f8bbd0;
}
QMenu::item:selected {
    background-color: #f8bbd0;
    color: #5a4a42;
}
QMenu::separator {
    height: 1px;
    background: #f8bbd0;
    margin: 3px 8px;
}
QWidget {
    background-color: #fef6f0;
    color: #5a4a42;
}
QPushButton {
    background-color: #fce4ec;
    color: #5a4a42;
    border: 1px solid #f8bbd0;
    padding: 4px 10px;
    border-radius: 10px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #f8bbd0;
    color: #5a4a42;
}
QPushButton:pressed {
    background-color: #f48fb1;
}
QPushButton:checked {
    background-color: #ce93d8;
    color: #fff;
    border-color: #ba68c8;
}
QLabel {
    color: #5a4a42;
    background: transparent;
}
QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit {
    background-color: #fff;
    color: #5a4a42;
    border: 1px solid #e0bfc7;
    padding: 2px 4px;
    border-radius: 6px;
}
QComboBox::drop-down {
    border: none;
    background: #fce4ec;
    border-radius: 0 6px 6px 0;
}
QComboBox QAbstractItemView {
    background-color: #fff;
    color: #5a4a42;
    selection-background-color: #f8bbd0;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #f8bbd0;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -5px 0;
    background: #ce93d8;
    border: 2px solid #fff;
    border-radius: 8px;
}
QScrollArea {
    background: #e8d5c4;
    border: none;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: #fef6f0;
    border: none;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #f8bbd0;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0; width: 0;
}
QTabWidget::pane {
    border: 1px solid #f8bbd0;
    background: #fef6f0;
    border-radius: 4px;
}
QTabBar::tab {
    background: #fce4ec;
    color: #8a6a62;
    padding: 6px 14px;
    border: 1px solid #f8bbd0;
    border-bottom: none;
    border-radius: 6px 6px 0 0;
}
QTabBar::tab:selected {
    background: #fef6f0;
    color: #5a4a42;
}
QCheckBox {
    color: #5a4a42;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #e0bfc7;
    background: #fff;
    border-radius: 4px;
}
QCheckBox::indicator:checked {
    background: #ce93d8;
    border-color: #ba68c8;
}
QGroupBox {
    color: #5a4a42;
    border: 1px solid #f8bbd0;
    margin-top: 8px;
    padding-top: 12px;
    border-radius: 6px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    color: #ba68c8;
}
QStatusBar {
    background: #fce4ec;
    color: #8a6a62;
    border-top: 1px solid #f8bbd0;
}
""")


# ── Google Material 風 ────────────────────────────────────────────────────────
_register("google", "Google Material 風", """
QMainWindow, QDialog {
    background-color: #ffffff;
    color: #202124;
}
QMenuBar {
    background-color: #ffffff;
    color: #202124;
    border-bottom: 1px solid #dadce0;
    font-size: 13px;
}
QMenuBar::item:selected {
    background-color: #e8f0fe;
    border-radius: 4px;
    color: #1a73e8;
}
QMenu {
    background-color: #ffffff;
    color: #202124;
    border: 1px solid #dadce0;
    border-radius: 8px;
    padding: 4px 0;
}
QMenu::item {
    padding: 6px 20px;
}
QMenu::item:selected {
    background-color: #e8f0fe;
    color: #1a73e8;
}
QMenu::separator {
    height: 1px;
    background: #e8eaed;
    margin: 4px 12px;
}
QWidget {
    background-color: #f8f9fa;
    color: #202124;
    font-size: 13px;
}
QPushButton {
    background-color: #ffffff;
    color: #1a73e8;
    border: 1px solid #dadce0;
    padding: 6px 16px;
    border-radius: 18px;
    min-height: 20px;
    font-weight: bold;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #e8f0fe;
    border-color: #1a73e8;
}
QPushButton:pressed {
    background-color: #d2e3fc;
}
QPushButton:checked {
    background-color: #1a73e8;
    color: #ffffff;
    border-color: #1a73e8;
}
QLabel {
    color: #202124;
    background: transparent;
}
QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #ffffff;
    color: #202124;
    border: 1px solid #dadce0;
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: #d2e3fc;
}
QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {
    border: 2px solid #1a73e8;
}
QComboBox {
    background-color: #ffffff;
    color: #202124;
    border: 1px solid #dadce0;
    border-radius: 8px;
    padding: 6px 8px;
}
QComboBox::drop-down {
    border: none;
    background: transparent;
    width: 20px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #202124;
    border: 1px solid #dadce0;
    border-radius: 8px;
    selection-background-color: #e8f0fe;
    selection-color: #1a73e8;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #dadce0;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -6px 0;
    background: #1a73e8;
    border: 2px solid #ffffff;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #1a73e8;
    border-radius: 2px;
}
QScrollArea {
    background: #e8eaed;
    border: none;
}
QScrollBar:vertical, QScrollBar:horizontal {
    background: transparent;
    border: none;
    width: 8px;
    height: 8px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
    background: #bdc1c6;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: #80868b;
}
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0; width: 0;
}
QTabWidget::pane {
    border: 1px solid #dadce0;
    background: #ffffff;
    border-radius: 8px;
}
QTabBar::tab {
    background: transparent;
    color: #5f6368;
    padding: 8px 16px;
    border: none;
    border-bottom: 2px solid transparent;
    font-size: 13px;
}
QTabBar::tab:selected {
    color: #1a73e8;
    border-bottom: 2px solid #1a73e8;
}
QTabBar::tab:hover {
    background: #f1f3f4;
}
QCheckBox {
    color: #202124;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px; height: 18px;
    border: 2px solid #5f6368;
    background: #ffffff;
    border-radius: 3px;
}
QCheckBox::indicator:checked {
    background: #1a73e8;
    border-color: #1a73e8;
}
QGroupBox {
    color: #202124;
    border: 1px solid #dadce0;
    border-radius: 8px;
    margin-top: 10px;
    padding-top: 14px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    color: #5f6368;
    font-size: 12px;
}
QStatusBar {
    background: #ffffff;
    color: #5f6368;
    border-top: 1px solid #dadce0;
    font-size: 12px;
}
""")


def get_theme_keys() -> list[str]:
    return list(THEMES.keys())


def get_theme_label(key: str) -> str:
    return THEMES[key]["label"]


def get_theme_qss(key: str) -> str:
    return THEMES[key]["qss"]
