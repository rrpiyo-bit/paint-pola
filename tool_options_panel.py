from __future__ import annotations

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QSpinBox, QSlider, QComboBox, QFrame,
                              QSizePolicy, QScrollArea)
from PyQt6.QtCore import Qt, pyqtSignal

from tools import Tool
from brush import BRUSH_LABELS


class ToolOptionsPanel(QWidget):
    """ツールごとの詳細設定パネル。ツール切替で内容が変わる。"""

    # 各設定の変更シグナル
    pen_size_changed      = pyqtSignal(int)
    eraser_size_changed   = pyqtSignal(int)
    brush_changed         = pyqtSignal(str)
    symmetry_toggled      = pyqtSignal(bool)
    shape_fill_changed    = pyqtSignal(str)
    fill_expand_changed   = pyqtSignal(int)   # バケツ塗り拡張px（負=縮小）
    select_mode_changed   = pyqtSignal(str)   # "select" | "transform"
    pivot_changed         = pyqtSignal(int, int)  # (ax, ay) 変形基準点
    pivot_mode_changed    = pyqtSignal(str)        # "preset" | "custom"
    transform_mode_changed = pyqtSignal(str)       # "standard" | "perspective" | "mesh"
    mesh_div_changed = pyqtSignal(int)              # メッシュ分割数
    blur_size_changed = pyqtSignal(int)
    blur_strength_changed = pyqtSignal(int)         # 0〜100 (%)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # タイトルバー
        title_bar = QFrame()
        title_bar.setStyleSheet("background:#2a2a2a; color:white;")
        title_bar.setFixedHeight(28)
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(8, 2, 8, 2)
        self._title = QLabel("ツールオプション")
        self._title.setStyleSheet("color:white; font-weight:bold; font-size:11px;")
        tb_layout.addWidget(self._title)
        outer.addWidget(title_bar)

        # スクロールエリア内にコンテンツ
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: #f5f5f5; }")
        self._content = QWidget()
        self._content.setStyleSheet("background: #f5f5f5;")
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._content_layout.setSpacing(6)
        self._content_layout.addStretch()
        scroll.setWidget(self._content)
        outer.addWidget(scroll, 1)

        self._current_tool: Tool | None = None
        self._widgets: list[QWidget] = []

    # ── 外部から呼ぶ ──────────────────────────────────────────────────────────

    def set_tool(self, tool: Tool,
                 pen_size: int = 5, eraser_size: int = 20,
                 brush_key: str = "round", symmetry: bool = False,
                 shape_fill: str = "none", fill_expand: int = 0,
                 select_mode: str = "select"):
        self._current_tool = tool
        self._clear()

        label_map = {
            Tool.PEN: "ペン",
            Tool.ERASER: "消しゴム",
            Tool.FILL: "バケツ塗り",
            Tool.LINE: "直線",
            Tool.RECT: "四角形",
            Tool.ELLIPSE: "楕円",
            Tool.SELECT_RECT: "矩形選択",
            Tool.LASSO: "投げなわ",
            Tool.LASSO_FILL: "囲み内塗りつぶし",
            Tool.BLUR: "ぼかし",
            Tool.MOVE: "移動",
            Tool.TRANSFORM: "自由変形",
            Tool.EYEDROPPER: "スポイト",
            Tool.TEXT: "テキスト",
        }
        self._title.setText(label_map.get(tool, "ツールオプション"))

        if tool == Tool.PEN:
            self._add_spinbox("ブラシサイズ", pen_size, 1, 200,
                              lambda v: self.pen_size_changed.emit(v))
            self._add_brush_combo(brush_key)
            self._add_toggle("対称定規", symmetry,
                             lambda v: self.symmetry_toggled.emit(v))

        elif tool == Tool.ERASER:
            self._add_spinbox("消しゴムサイズ", eraser_size, 1, 300,
                              lambda v: self.eraser_size_changed.emit(v))

        elif tool == Tool.FILL:
            self._add_spinbox("拡張/縮小 (px)", fill_expand, -30, 30,
                              lambda v: self.fill_expand_changed.emit(v),
                              tooltip="正: 塗り範囲を広げる  負: 塗り範囲を縮める")

        elif tool == Tool.BLUR:
            self._add_spinbox("ブラシサイズ", 30, 1, 200,
                              lambda v: self.blur_size_changed.emit(v))
            self._add_blur_strength_slider()

        elif tool in (Tool.RECT, Tool.ELLIPSE, Tool.LINE):
            self._add_spinbox("線の太さ", pen_size, 1, 200,
                              lambda v: self.pen_size_changed.emit(v))
            if tool in (Tool.RECT, Tool.ELLIPSE):
                self._add_fill_combo(shape_fill)

        elif tool in (Tool.SELECT_RECT, Tool.LASSO):
            self._add_select_mode_combo(select_mode)
            self._add_transform_mode_combo()
            self._add_pivot_selector()

        elif tool == Tool.TRANSFORM:
            self._add_transform_mode_combo()
            self._add_pivot_selector()

        elif tool == Tool.MOVE:
            self._add_label("矢印キーで 1px 移動\nShift+矢印で 10px 移動")

        elif tool == Tool.EYEDROPPER:
            self._add_label("Alt キーでも\n一時スポイトになります")

        elif tool == Tool.TEXT:
            self._add_label("キャンバスをクリックして\nテキストを入力")

    def sync_pen_size(self, v: int):
        for w in self._widgets:
            if getattr(w, '_opt_key', None) == 'pen_size':
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)

    def sync_fill_expand(self, v: int):
        for w in self._widgets:
            if getattr(w, '_opt_key', None) == 'fill_expand':
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)

    # ── ビルダー ──────────────────────────────────────────────────────────────

    def _clear(self):
        layout = self._content_layout
        while layout.count() > 1:
            item = layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._widgets.clear()

    def _add_row(self, label: str, widget: QWidget, tooltip: str = ""):
        row = QWidget()
        rl = QVBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        lbl = QLabel(label)
        lbl.setStyleSheet("font-size:11px; color:#333;")
        if tooltip:
            lbl.setToolTip(tooltip)
        rl.addWidget(lbl)
        rl.addWidget(widget)
        self._content_layout.insertWidget(self._content_layout.count() - 1, row)
        self._widgets.append(widget)

    def _add_spinbox(self, label: str, value: int, lo: int, hi: int,
                     callback, tooltip: str = "", key: str = ""):
        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(value)
        if tooltip:
            sb.setToolTip(tooltip)
        if not key:
            # label から自動判定
            if "ブラシ" in label or "線の太さ" in label:
                key = "pen_size"
            elif "消しゴム" in label:
                key = "eraser_size"
            elif "拡張" in label:
                key = "fill_expand"
        sb._opt_key = key  # type: ignore
        sb.valueChanged.connect(callback)
        self._add_row(label, sb, tooltip)

    def _add_brush_combo(self, current_key: str):
        cb = QComboBox()
        keys = list(BRUSH_LABELS.keys())
        for k in keys:
            cb.addItem(BRUSH_LABELS[k], k)
        idx = keys.index(current_key) if current_key in keys else 0
        cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(
            lambda i: self.brush_changed.emit(keys[i]))
        self._add_row("ブラシ種類", cb)

    def _add_fill_combo(self, current: str):
        cb = QComboBox()
        cb.addItem("枠線のみ", "none")
        cb.addItem("塗りのみ", "fill")
        cb.addItem("枠線＋塗り", "both")
        idx = {"none": 0, "fill": 1, "both": 2}.get(current, 0)
        cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(
            lambda i: self.shape_fill_changed.emit(cb.itemData(i)))
        self._add_row("図形塗り", cb)

    def _add_toggle(self, label: str, value: bool, callback):
        from PyQt6.QtWidgets import QCheckBox
        cb = QCheckBox(label)
        cb.setChecked(value)
        cb.toggled.connect(callback)
        self._content_layout.insertWidget(self._content_layout.count() - 1, cb)
        self._widgets.append(cb)

    def _add_select_mode_combo(self, current: str):
        cb = QComboBox()
        cb.addItem("選択のみ", "select")
        cb.addItem("変形（移動・拡縮・回転）", "transform")
        idx = 0 if current == "select" else 1
        cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(
            lambda i: self.select_mode_changed.emit(cb.itemData(i)))
        self._add_row("選択モード", cb)

    def _add_transform_mode_combo(self):
        cb = QComboBox()
        cb.addItem("拡縮・回転", "standard")
        cb.addItem("自由変形（パース）", "perspective")
        cb.addItem("メッシュ変形", "mesh")
        cb.setCurrentIndex(0)

        mesh_div = QSpinBox()
        mesh_div.setRange(2, 8)
        mesh_div.setValue(3)
        mesh_div.setPrefix("分割: ")
        mesh_div.setSuffix(" ×")
        mesh_div.setVisible(False)
        mesh_div.valueChanged.connect(lambda v: self.mesh_div_changed.emit(v))

        def on_mode(i):
            mode = cb.itemData(i)
            mesh_div.setVisible(mode == "mesh")
            self.transform_mode_changed.emit(mode)

        cb.currentIndexChanged.connect(on_mode)
        self._add_row("変形モード", cb)
        self._content_layout.insertWidget(self._content_layout.count() - 1, mesh_div)
        self._widgets.append(mesh_div)

    def _add_pivot_selector(self):
        from main import AnchorWidget
        lbl = QLabel("変形基準点")
        lbl.setStyleSheet("color:#555; font-size:11px;")
        self._content_layout.insertWidget(self._content_layout.count() - 1, lbl)
        self._widgets.append(lbl)

        mode_cb = QComboBox()
        mode_cb.addItem("プリセット（9点）", "preset")
        mode_cb.addItem("任意（ドラッグ）", "custom")
        self._content_layout.insertWidget(self._content_layout.count() - 1, mode_cb)
        self._widgets.append(mode_cb)

        aw = AnchorWidget()
        aw.anchor_changed.connect(lambda ax, ay: self.pivot_changed.emit(ax, ay))
        self._content_layout.insertWidget(self._content_layout.count() - 1, aw)
        self._widgets.append(aw)

        hint = QLabel("キャンバス上の青い十字を\nドラッグで移動できます")
        hint.setStyleSheet("color:#888; font-size:10px;")
        hint.setWordWrap(True)
        hint.setVisible(False)
        self._content_layout.insertWidget(self._content_layout.count() - 1, hint)
        self._widgets.append(hint)

        def on_mode_change(i):
            mode = mode_cb.itemData(i)
            aw.setVisible(mode == "preset")
            hint.setVisible(mode == "custom")
            self.pivot_mode_changed.emit(mode)

        mode_cb.currentIndexChanged.connect(on_mode_change)

    def _add_blur_strength_slider(self):
        row = QWidget()
        rl = QVBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        lbl_title = QLabel("ぼかし強度")
        lbl_title.setStyleSheet("font-size:11px; color:#333;")
        rl.addWidget(lbl_title)
        slider_row = QHBoxLayout()
        slider_row.setSpacing(4)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(1, 100)
        slider.setValue(50)
        val_lbl = QLabel("50%")
        val_lbl.setFixedWidth(36)
        val_lbl.setStyleSheet("font-size:11px; color:#333;")
        def on_change(v):
            val_lbl.setText(f"{v}%")
            self.blur_strength_changed.emit(v)
        slider.valueChanged.connect(on_change)
        slider_row.addWidget(slider)
        slider_row.addWidget(val_lbl)
        rl.addLayout(slider_row)
        self._content_layout.insertWidget(self._content_layout.count() - 1, row)
        self._widgets.append(slider)

    def _add_label(self, text: str):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#555; font-size:11px;")
        lbl.setWordWrap(True)
        self._content_layout.insertWidget(self._content_layout.count() - 1, lbl)
