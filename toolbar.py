from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QFrame, QColorDialog, QPushButton, QLabel)
from PyQt6.QtGui import QColor, QPixmap, QPainter, QCursor, QPen, QBrush, QPolygon
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect

from tools import Tool

# ツールのキーボードショートカット（表示・入力両用）
TOOL_SHORTCUTS: dict[Tool, str] = {
    Tool.PEN:        "P",
    Tool.ERASER:     "E",
    Tool.FILL:       "G",
    Tool.EYEDROPPER: "I",
    Tool.LINE:       "L",
    Tool.RECT:       "R",
    Tool.ELLIPSE:    "O",
    Tool.TEXT:       "T",
    Tool.BLUR:       "B",
    Tool.SELECT_RECT:"S",
    Tool.LASSO:      "Q",
    Tool.LASSO_FILL: "W",
    Tool.MOVE:       "V",
    Tool.TRANSFORM:  "F",
}

_HISTORY_MAX = 10  # ツールバーに表示するカラーヒストリーの最大数


_TOOL_ICONS: dict[Tool, str] = {
    Tool.PEN: "✏️",
    Tool.ERASER: "◻",
    Tool.FILL: "\U0001f4a7",
    Tool.EYEDROPPER: "\U0001f4a7",
    Tool.LINE: "╱",
    Tool.RECT: "□",
    Tool.ELLIPSE: "○",
    Tool.TEXT: "T",
    Tool.BLUR: "💧",
    Tool.SELECT_RECT: "⬚",
    Tool.LASSO: "〇",
    Tool.LASSO_FILL: "\U0001f7e2",
    Tool.MOVE: "✥",
    Tool.TRANSFORM: "⤢",
}

_TOOL_LABELS: dict[Tool, str] = {
    Tool.PEN: "ペン",
    Tool.ERASER: "消しゴム",
    Tool.FILL: "塗りつぶし",
    Tool.EYEDROPPER: "スポイト",
    Tool.LINE: "直線",
    Tool.RECT: "四角形",
    Tool.ELLIPSE: "楕円",
    Tool.TEXT: "テキスト",
    Tool.BLUR: "ぼかし",
    Tool.SELECT_RECT: "矩形選択",
    Tool.LASSO: "投げなわ",
    Tool.LASSO_FILL: "囲み内塗りつぶし",
    Tool.MOVE: "移動",
    Tool.TRANSFORM: "自由変形",
}


def _make_cursor_pixmap(draw_fn, size=24, hot=(12, 12)):
    """カスタムカーソル用の小さなQPixmapを生成する。"""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    draw_fn(p, size)
    p.end()
    return QCursor(pm, hot[0], hot[1])


def make_tool_cursors() -> dict[Tool, QCursor]:
    """各ツール用のカスタムカーソルを生成して返す。"""
    cursors: dict[Tool, QCursor] = {}

    def _pen(p: QPainter, s: int):
        p.setPen(QPen(QColor(0, 0, 0), 2))
        p.drawLine(s // 2, 2, s // 2, s - 2)
        p.drawLine(2, s // 2, s - 2, s // 2)

    def _eraser(p: QPainter, s: int):
        p.setPen(QPen(QColor(80, 80, 80), 1))
        p.setBrush(QBrush(QColor(255, 255, 255, 200)))
        p.drawRect(4, 4, s - 8, s - 8)

    def _fill(p: QPainter, s: int):
        # バケツ型
        p.setPen(QPen(QColor(0, 0, 0), 2))
        p.setBrush(QBrush(QColor(100, 160, 255, 180)))
        pts = QPolygon([QPoint(4, 8), QPoint(12, 4), QPoint(20, 8),
                        QPoint(20, 18), QPoint(4, 18)])
        p.drawPolygon(pts)
        # 水滴
        p.setBrush(QBrush(QColor(80, 140, 255)))
        p.drawEllipse(10, 19, 5, 5)

    def _eyedropper(p: QPainter, s: int):
        p.setPen(QPen(QColor(0, 0, 0), 2))
        p.drawLine(4, s - 4, s // 2, s // 2)
        p.drawEllipse(s // 2 - 3, s // 2 - 7, 10, 10)

    def _line(p: QPainter, s: int):
        p.setPen(QPen(QColor(0, 0, 0), 2))
        p.drawLine(4, s - 4, s - 4, 4)
        # 十字
        p.setPen(QPen(QColor(120, 120, 120), 1))
        p.drawLine(s // 2, 2, s // 2, s - 2)
        p.drawLine(2, s // 2, s - 2, s // 2)

    def _rect(p: QPainter, s: int):
        p.setPen(QPen(QColor(0, 0, 0), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(4, 4, s - 8, s - 8)
        p.setPen(QPen(QColor(120, 120, 120), 1))
        p.drawLine(s // 2, 2, s // 2, s - 2)
        p.drawLine(2, s // 2, s - 2, s // 2)

    def _ellipse(p: QPainter, s: int):
        p.setPen(QPen(QColor(0, 0, 0), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(3, 3, s - 6, s - 6)
        p.setPen(QPen(QColor(120, 120, 120), 1))
        p.drawLine(s // 2, 2, s // 2, s - 2)
        p.drawLine(2, s // 2, s - 2, s // 2)

    def _text(p: QPainter, s: int):
        from PyQt6.QtGui import QFont
        p.setPen(QPen(QColor(0, 0, 0), 1))
        f = QFont("Arial", 14, QFont.Weight.Bold)
        p.setFont(f)
        p.drawText(QRect(0, 0, s, s), Qt.AlignmentFlag.AlignCenter, "T")

    def _select(p: QPainter, s: int):
        p.setPen(QPen(QColor(0, 0, 0), 1, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(4, 4, s - 8, s - 8)
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.drawLine(s // 2, 2, s // 2, s - 2)
        p.drawLine(2, s // 2, s - 2, s // 2)

    def _lasso(p: QPainter, s: int):
        from PyQt6.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(12, 4)
        path.cubicTo(20, 4, 22, 12, 18, 16)
        path.cubicTo(14, 20, 6, 20, 4, 14)
        path.cubicTo(2, 8, 6, 4, 12, 4)
        p.setPen(QPen(QColor(0, 0, 0), 1.5, Qt.PenStyle.DashLine))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    def _lasso_fill(p: QPainter, s: int):
        from PyQt6.QtGui import QPainterPath
        path = QPainterPath()
        path.moveTo(12, 4)
        path.cubicTo(20, 4, 22, 12, 18, 16)
        path.cubicTo(14, 20, 6, 20, 4, 14)
        path.cubicTo(2, 8, 6, 4, 12, 4)
        p.setPen(QPen(QColor(0, 0, 0), 1.5, Qt.PenStyle.DashLine))
        p.setBrush(QBrush(QColor(100, 160, 255, 120)))
        p.drawPath(path)

    def _blur(p: QPainter, s: int):
        p.setPen(QPen(QColor(100, 140, 220), 2))
        p.setBrush(QBrush(QColor(100, 140, 220, 60)))
        p.drawEllipse(3, 3, s - 6, s - 6)
        p.setPen(QPen(QColor(60, 100, 180), 1))
        p.drawLine(s // 2, 6, s // 2, s - 6)
        p.drawLine(6, s // 2, s - 6, s // 2)

    def _transform(p: QPainter, s: int):
        p.setPen(QPen(QColor(0, 0, 0), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(6, 6, s - 12, s - 12)
        for x, y in [(4, 4), (s - 8, 4), (4, s - 8), (s - 8, s - 8)]:
            p.fillRect(x, y, 4, 4, QColor(0, 0, 0))

    cursors[Tool.PEN] = _make_cursor_pixmap(_pen, hot=(12, 12))
    cursors[Tool.ERASER] = _make_cursor_pixmap(_eraser, hot=(12, 12))
    cursors[Tool.FILL] = _make_cursor_pixmap(_fill, hot=(4, 20))
    cursors[Tool.EYEDROPPER] = _make_cursor_pixmap(_eyedropper, hot=(4, 20))
    cursors[Tool.LINE] = _make_cursor_pixmap(_line, hot=(12, 12))
    cursors[Tool.RECT] = _make_cursor_pixmap(_rect, hot=(12, 12))
    cursors[Tool.ELLIPSE] = _make_cursor_pixmap(_ellipse, hot=(12, 12))
    cursors[Tool.TEXT] = _make_cursor_pixmap(_text, hot=(12, 12))
    cursors[Tool.SELECT_RECT] = _make_cursor_pixmap(_select, hot=(12, 12))
    cursors[Tool.LASSO] = _make_cursor_pixmap(_lasso, hot=(12, 12))
    cursors[Tool.LASSO_FILL] = _make_cursor_pixmap(_lasso_fill, hot=(12, 12))
    cursors[Tool.BLUR] = _make_cursor_pixmap(_blur, hot=(12, 12))
    cursors[Tool.MOVE] = QCursor(Qt.CursorShape.SizeAllCursor)
    cursors[Tool.TRANSFORM] = _make_cursor_pixmap(_transform, hot=(12, 12))
    return cursors


class ColorSwatch(QFrame):
    clicked = pyqtSignal()

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(32, 32)
        self.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh()

    def set_color(self, color: QColor):
        self._color = color
        self._refresh()

    def color(self) -> QColor:
        return self._color

    def _refresh(self):
        self.setStyleSheet(f"background-color: {self._color.name()}; border: 1px solid #888;")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class Toolbar(QWidget):
    tool_changed = pyqtSignal(Tool)
    color_changed = pyqtSignal(QColor)
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    options_toggled = pyqtSignal(bool)    # ▼ ボタン

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(90)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._buttons: dict[Tool, QPushButton] = {}
        for tool in Tool:
            icon = _TOOL_ICONS.get(tool, "")
            key = TOOL_SHORTCUTS.get(tool, "")
            label = _TOOL_LABELS[tool]
            btn = QPushButton(f"{icon} {label}")
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.setStyleSheet("QPushButton { text-align: left; padding-left: 6px; }")
            btn.setToolTip(f"{label}  [{key}]" if key else label)
            btn.clicked.connect(lambda _, t=tool: self._select(t))
            self._buttons[tool] = btn
            layout.addWidget(btn)

        layout.addWidget(self._separator())

        undo_row = QHBoxLayout()
        undo_btn = QPushButton("↩")
        undo_btn.setFixedHeight(28)
        undo_btn.setToolTip("元に戻す (Ctrl+Z)")
        undo_btn.clicked.connect(self.undo_requested)
        redo_btn = QPushButton("↪")
        redo_btn.setFixedHeight(28)
        redo_btn.setToolTip("やり直し (Ctrl+Y)")
        redo_btn.clicked.connect(self.redo_requested)
        undo_row.addWidget(undo_btn)
        undo_row.addWidget(redo_btn)
        layout.addLayout(undo_row)

        layout.addWidget(self._separator())

        layout.addWidget(QLabel("色"))
        self._swatch = ColorSwatch(QColor(0, 0, 0))
        self._swatch.clicked.connect(self._pick_color)
        layout.addWidget(self._swatch)

        # カラーヒストリー（2行×5列）
        self._history_colors: list[QColor] = []
        self._history_btns: list[QPushButton] = []
        hist_label = QLabel("履歴")
        hist_label.setStyleSheet("font-size:9px; color:#666;")
        layout.addWidget(hist_label)
        _hist_css = ("QPushButton { border: 1px solid #aaa; padding: 0; margin: 0; }"
                     "QPushButton:hover { border: 1px solid #4a90d9; }")
        for row_idx in range(2):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(1)
            row_layout.setContentsMargins(0, 0, 0, 0)
            for col_idx in range(5):
                btn = QPushButton()
                btn.setFixedSize(15, 15)
                btn.setStyleSheet(_hist_css + " QPushButton { background: #fff; }")
                btn.setEnabled(False)
                btn.clicked.connect(lambda _, b=btn: self._on_history_click(b))
                self._history_btns.append(btn)
                row_layout.addWidget(btn)
            layout.addLayout(row_layout)

        # ▼ ツールオプション展開ボタン（色のすぐ下）
        self._opt_btn = QPushButton("▼ オプション")
        self._opt_btn.setCheckable(True)
        self._opt_btn.setFixedHeight(28)
        self._opt_btn.setToolTip("ツールオプションパネルを開閉")
        self._opt_btn.setStyleSheet(
            "QPushButton { font-size:10px; }"
            "QPushButton:checked { background: #4a90d9; color: white; }"
        )
        self._opt_btn.toggled.connect(self.options_toggled)
        layout.addWidget(self._opt_btn)

        layout.addStretch()

        self._select(Tool.PEN)

    def _separator(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        return f

    def _select(self, tool: Tool):
        for t, btn in self._buttons.items():
            btn.setChecked(t == tool)
        self.tool_changed.emit(tool)

    def _pick_color(self):
        c = QColorDialog.getColor(self._swatch.color(), self, "色を選択",
                                   QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self._swatch.set_color(c)
            self.color_changed.emit(c)

    def set_color(self, color: QColor):
        self._swatch.set_color(color)
        self.color_changed.emit(color)

    def set_color_preview(self, color: QColor):
        """スウォッチの見た目だけ更新し、履歴には追加しない
        （HSVスライダーのドラッグ中など、確定前の中間値向け）。"""
        self._swatch.set_color(color)

    def push_color(self, color: QColor):
        """色を使用履歴に追加する（同色は先頭へ移動）。"""
        self._history_colors = [c for c in self._history_colors if c.rgba() != color.rgba()]
        self._history_colors.insert(0, color)
        if len(self._history_colors) > _HISTORY_MAX:
            self._history_colors = self._history_colors[:_HISTORY_MAX]
        for i, btn in enumerate(self._history_btns):
            if i < len(self._history_colors):
                c = self._history_colors[i]
                btn.setStyleSheet(
                    f"QPushButton {{ background-color: {c.name()}; border: 1px solid #aaa; padding: 0; margin: 0; }}"
                    "QPushButton:hover { border: 1px solid #4a90d9; }"
                )
                btn.setToolTip(c.name())
                btn.setEnabled(True)
            else:
                btn.setStyleSheet("QPushButton { background: #fff; border: 1px solid #aaa; padding: 0; margin: 0; }")
                btn.setToolTip("")
                btn.setEnabled(False)

    def _on_history_click(self, btn: QPushButton):
        idx = self._history_btns.index(btn)
        if idx < len(self._history_colors):
            c = self._history_colors[idx]
            self._swatch.set_color(c)
            self.color_changed.emit(c)

    def select_tool(self, tool: Tool):
        """外部（キーボードショートカット等）からツールを選択する。"""
        self._select(tool)
