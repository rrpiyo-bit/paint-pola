from __future__ import annotations

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QScrollArea, QCheckBox, QLabel, QSlider,
                              QInputDialog, QSizePolicy, QFrame, QMenu,
                              QTabWidget, QSpinBox, QColorDialog)
from PyQt6.QtGui import QColor, QPainter, QPen, QImage, QPixmap, QDrag
from PyQt6.QtCore import Qt, QRect, QSize, pyqtSignal, QMimeData, QPoint

from layer import LayerStack, Layer, GroupLayer, BLEND_MODES, BLEND_KEYS, BLEND_LABELS
from actions import ActionPanel

# ── 定数 ──────────────────────────────────────────────────────────────────────
_THUMB_SIZE    = 40          # サムネイル px
_ROW_HEIGHT    = 44          # 1行の高さ
_INDENT        = 16          # グループ子レイヤーのインデント
_CLIP_INDENT   = 14          # クリッピング追加インデント
_CLIP_COLOR    = QColor(80, 140, 220)
_GROUP_COLOR   = QColor(60, 60, 60, 20)   # グループ行の背景
_CHILD_COLOR   = QColor(40, 100, 200, 12) # 子行の背景
_ACTIVE_COLOR  = QColor(0, 120, 215, 50)  # 選択行ハイライト
_PANEL_WIDTH   = 260


def _make_thumb(image: QImage, size: int = _THUMB_SIZE) -> QPixmap:
    """レイヤー画像をチェッカー背景付きサムネイルに変換する。"""
    thumb = QImage(size, size, QImage.Format.Format_ARGB32)
    thumb.fill(Qt.GlobalColor.transparent)
    p = QPainter(thumb)
    # チェッカー（透明表示用）
    checker_size = 5
    for row in range(0, size, checker_size):
        for col in range(0, size, checker_size):
            if (row // checker_size + col // checker_size) % 2 == 0:
                p.fillRect(col, row, checker_size, checker_size, QColor(200, 200, 200))
            else:
                p.fillRect(col, row, checker_size, checker_size, QColor(255, 255, 255))
    scaled = image.scaled(QSize(size, size),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
    ox = (size - scaled.width()) // 2
    oy = (size - scaled.height()) // 2
    p.drawImage(ox, oy, scaled)
    p.end()
    return QPixmap.fromImage(thumb)


class LayerRow(QWidget):
    """レイヤー1行分のウィジェット（サムネイル＋チェックボックス群）。"""

    visibility_changed = pyqtSignal(object, bool)        # (layer, visible)
    clipping_changed   = pyqtSignal(object, bool)
    reference_changed  = pyqtSignal(object, bool)
    rename_requested   = pyqtSignal(object)              # (layer,)
    select_requested   = pyqtSignal(object)              # (layer,)
    toggle_collapse    = pyqtSignal(object)              # GroupLayer のみ

    def __init__(self, layer: Layer | GroupLayer,
                 indent: int = 0,
                 is_active: bool = False,
                 clipped: bool = False,
                 parent=None):
        super().__init__(parent)
        self._layer = layer
        self._indent = indent
        self._clipped = clipped
        self._is_active = is_active
        self.setFixedHeight(_ROW_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        left = self._indent + (_CLIP_INDENT if self._clipped else 0) + 4
        layout.setContentsMargins(left, 3, 4, 3)
        layout.setSpacing(4)

        # 折りたたみボタン（グループのみ）
        if self._layer.is_group:
            self._collapse_btn = QPushButton(
                "▶" if self._layer.collapsed else "▼")  # type: ignore
            self._collapse_btn.setFixedSize(16, 16)
            self._collapse_btn.setFlat(True)
            self._collapse_btn.setStyleSheet("QPushButton { font-size: 9px; padding: 0; }")
            self._collapse_btn.clicked.connect(
                lambda: self.toggle_collapse.emit(self._layer))
            layout.addWidget(self._collapse_btn)

        # 表示チェック
        self._vis = QCheckBox()
        self._vis.setChecked(self._layer.visible)
        self._vis.setToolTip("表示/非表示")
        self._vis.setFixedSize(18, 18)
        self._vis.stateChanged.connect(
            lambda s: self.visibility_changed.emit(self._layer, bool(s)))
        layout.addWidget(self._vis)

        # サムネイル
        self._thumb_label = QLabel()
        self._thumb_label.setFixedSize(_THUMB_SIZE, _THUMB_SIZE)
        self._thumb_label.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Plain)  # type: ignore
        self._thumb_label.setLineWidth(1)
        self._refresh_thumb()
        layout.addWidget(self._thumb_label)

        # 右側: 名前＋チェックボックス群
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(2)

        name_label = QLabel(self._layer.name)
        name_label.setToolTip(self._layer.name)
        if self._layer.is_group:
            f = name_label.font()
            f.setBold(True)
            name_label.setFont(f)
        right.addWidget(name_label)
        self._name_label = name_label

        # アイコンボタン行（clip / ref）
        icon_row = QHBoxLayout()
        icon_row.setContentsMargins(0, 0, 0, 0)
        icon_row.setSpacing(2)

        _ICON_BTN_CSS = (
            "QPushButton { font-size:11px; padding:0 2px; border:1px solid #aaa;"
            " border-radius:2px; background:#e8e8e8; min-width:20px; max-height:18px; }"
            "QPushButton:checked { background:#4a90d9; color:white; border-color:#3570b0; }"
        )

        self._clip = QPushButton("📎")
        self._clip.setCheckable(True)
        self._clip.setChecked(self._layer.clipping)
        self._clip.setToolTip("クリッピング")
        self._clip.setStyleSheet(_ICON_BTN_CSS)
        self._clip.toggled.connect(
            lambda s: self.clipping_changed.emit(self._layer, s))
        icon_row.addWidget(self._clip)

        self._ref = QPushButton("🔍")
        self._ref.setCheckable(True)
        self._ref.setChecked(self._layer.reference)
        self._ref.setToolTip("参照レイヤー")
        self._ref.setStyleSheet(_ICON_BTN_CSS)
        self._ref.toggled.connect(
            lambda s: self.reference_changed.emit(self._layer, s))
        icon_row.addWidget(self._ref)
        icon_row.addStretch()
        right.addLayout(icon_row)

        layout.addLayout(right, 1)

    def _refresh_thumb(self):
        if self._layer.is_group:
            img = self._layer.composite()  # type: ignore
        else:
            img = self._layer.image  # type: ignore
        self._thumb_label.setPixmap(_make_thumb(img))

    def refresh_thumb(self):
        self._refresh_thumb()

    def set_active(self, active: bool):
        self._is_active = active
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)

        # 背景色
        if self._is_active:
            p.fillRect(self.rect(), _ACTIVE_COLOR)
        elif self._layer.is_group:
            p.fillRect(self.rect(), _GROUP_COLOR)
        elif self._indent > 0:
            p.fillRect(self.rect(), _CHILD_COLOR)

        # クリッピング縦線
        if self._clipped:
            x = self._indent + 2
            p.setPen(QPen(_CLIP_COLOR, 2))
            p.drawLine(x, 0, x, self.height())
            p.drawLine(x, self.height() // 2, x + _CLIP_INDENT - 2, self.height() // 2)

        p.end()

    def mouseDoubleClickEvent(self, event):
        self.rename_requested.emit(self._layer)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
            self.select_requested.emit(self._layer)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if not hasattr(self, '_drag_start_pos') or self._drag_start_pos is None:
            return
        if (event.pos() - self._drag_start_pos).manhattanLength() < 10:
            return
        drag = QDrag(self)
        mime = QMimeData()
        mime.setText("layer_drag")
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.MoveAction)
        self._drag_start_pos = None

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None


class LayerPanel(QWidget):
    layers_changed = pyqtSignal()
    layer_structure_changed = pyqtSignal()  # 削除・統合など構造変化時のみ
    structure_will_change = pyqtSignal()    # 構造変更直前（undo用スナップショット）
    merge_down_requested = pyqtSignal()
    merge_all_requested = pyqtSignal()
    merge_folder_requested = pyqtSignal()

    def __init__(self, layer_stack: LayerStack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        self.setFixedWidth(_PANEL_WIDTH)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        root.addWidget(self._tabs)

        # ── タブ1: レイヤー ──────────────────────────────────────────────────
        layer_tab = QWidget()
        lt = QVBoxLayout(layer_tab)
        lt.setContentsMargins(2, 2, 2, 2)
        lt.setSpacing(2)

        # アイコンツールバー（上部に固定表示）
        _TB_CSS = ("QPushButton { font-size:11px; padding:0; border:1px solid #bbb;"
                   " border-radius:1px; background:#eee; }"
                   "QPushButton:hover { background:#d0d8e8; }"
                   "QPushButton:pressed { background:#b0c0d8; }")
        tb1 = QHBoxLayout()
        tb1.setSpacing(0)
        tb1.setContentsMargins(0, 0, 0, 0)
        for icon, slot, tip in [
            ("➕", self._add, "新規レイヤー"),
            ("📁", self._add_group, "新規グループ"),
            ("📋", self._duplicate, "複製"),
            ("🗑", self._remove, "削除"),
            ("⬆", self._move_up, "上へ"),
            ("⬇", self._move_down, "下へ"),
        ]:
            b = QPushButton(icon)
            b.setFixedSize(22, 18)
            b.setToolTip(tip)
            b.setStyleSheet(_TB_CSS)
            b.clicked.connect(slot)
            tb1.addWidget(b)
        lt.addLayout(tb1)

        tb2 = QHBoxLayout()
        tb2.setSpacing(0)
        tb2.setContentsMargins(0, 0, 0, 0)
        for icon, slot, tip in [
            ("📥", self._add_to_group, "グループに追加"),
            ("📤", self._remove_from_group, "グループから出す"),
            ("⏬", self._merge_down, "下に統合"),
            ("👁", self._merge_all_visible, "表示統合"),
            ("🗂", self._merge_folder, "フォルダ統合"),
        ]:
            b = QPushButton(icon)
            b.setFixedSize(22, 18)
            b.setToolTip(tip)
            b.setStyleSheet(_TB_CSS)
            b.clicked.connect(slot)
            tb2.addWidget(b)
        tb2.addStretch()
        lt.addLayout(tb2)

        # 不透明度
        op_row = QHBoxLayout()
        op_row.setContentsMargins(0, 0, 0, 0)
        op_lbl = QLabel("透明度")
        op_lbl.setFixedWidth(30)
        op_lbl.setStyleSheet("font-size:10px;")
        op_row.addWidget(op_lbl)
        self._opacity = QSlider(Qt.Orientation.Horizontal)
        self._opacity.setRange(0, 255)
        self._opacity.setValue(255)
        self._opacity.setFixedHeight(14)
        self._opacity.valueChanged.connect(self._on_opacity)
        op_row.addWidget(self._opacity)
        lt.addLayout(op_row)

        # スクロールエリア（残りスペースに伸縮）
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._inner = QWidget()
        self._inner_layout = QVBoxLayout(self._inner)
        self._inner_layout.setContentsMargins(0, 0, 0, 0)
        self._inner_layout.setSpacing(1)
        self._inner_layout.addStretch()
        self._scroll.setWidget(self._inner)
        lt.addWidget(self._scroll, 1)

        self._tabs.addTab(layer_tab, "レイヤー")

        # ── タブ2: レイヤー設定 ──────────────────────────────────────────────
        settings_tab = QWidget()
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_inner = QWidget()
        st = QVBoxLayout(settings_inner)
        st.setContentsMargins(8, 8, 8, 8)
        st.setSpacing(6)

        # ── ブレンドモード ──
        st.addWidget(QLabel("ブレンドモード"))
        from PyQt6.QtWidgets import QComboBox
        self._blend_combo = QComboBox()
        for key in BLEND_KEYS:
            self._blend_combo.addItem(BLEND_LABELS[key], key)
        self._blend_combo.currentIndexChanged.connect(self._on_blend_mode)
        st.addWidget(self._blend_combo)

        st.addWidget(self._sep())

        # ── 縁取り ──
        self._border_check = QCheckBox("縁取りを有効にする")
        self._border_check.toggled.connect(self._on_border_enabled)
        st.addWidget(self._border_check)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("サイズ"))
        self._border_size = QSpinBox()
        self._border_size.setRange(1, 50)
        self._border_size.setValue(3)
        self._border_size.valueChanged.connect(self._on_border_size)
        size_row.addWidget(self._border_size)
        st.addLayout(size_row)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("色"))
        self._border_color_btn = QPushButton()
        self._border_color_btn.setFixedSize(40, 24)
        self._border_color_btn.clicked.connect(self._on_border_color_pick)
        self._border_color = QColor(0, 0, 0, 255)
        self._refresh_border_color_btn()
        color_row.addWidget(self._border_color_btn)
        color_row.addStretch()
        st.addLayout(color_row)

        st.addWidget(self._sep())

        # ── ドロップシャドウ ──
        self._shadow_check = QCheckBox("ドロップシャドウ")
        self._shadow_check.toggled.connect(self._on_shadow_enabled)
        st.addWidget(self._shadow_check)

        self._shadow_strength = QSlider(Qt.Orientation.Horizontal)
        self._shadow_strength.setRange(0, 100)
        self._shadow_strength.setValue(100)
        self._shadow_strength.valueChanged.connect(self._on_shadow_param)
        st.addWidget(self._labeled_slider("強度 %", self._shadow_strength))

        sh_offset = QHBoxLayout()
        sh_offset.addWidget(QLabel("X"))
        self._shadow_ox = QSpinBox()
        self._shadow_ox.setRange(-100, 100)
        self._shadow_ox.setValue(4)
        self._shadow_ox.valueChanged.connect(self._on_shadow_param)
        sh_offset.addWidget(self._shadow_ox)
        sh_offset.addWidget(QLabel("Y"))
        self._shadow_oy = QSpinBox()
        self._shadow_oy.setRange(-100, 100)
        self._shadow_oy.setValue(4)
        self._shadow_oy.valueChanged.connect(self._on_shadow_param)
        sh_offset.addWidget(self._shadow_oy)
        st.addLayout(sh_offset)

        sh_blur_row = QHBoxLayout()
        sh_blur_row.addWidget(QLabel("ぼかし"))
        self._shadow_blur = QSpinBox()
        self._shadow_blur.setRange(0, 50)
        self._shadow_blur.setValue(5)
        self._shadow_blur.valueChanged.connect(self._on_shadow_param)
        sh_blur_row.addWidget(self._shadow_blur)
        st.addLayout(sh_blur_row)

        sh_color_row = QHBoxLayout()
        sh_color_row.addWidget(QLabel("影色"))
        self._shadow_color_btn = QPushButton()
        self._shadow_color_btn.setFixedSize(40, 24)
        self._shadow_color_btn.clicked.connect(self._on_shadow_color_pick)
        self._shadow_color = QColor(0, 0, 0, 180)
        self._refresh_shadow_color_btn()
        sh_color_row.addWidget(self._shadow_color_btn)
        sh_color_row.addStretch()
        st.addLayout(sh_color_row)

        st.addWidget(self._sep())

        # ── 光彩（外側グロー）──
        self._glow_check = QCheckBox("光彩（外側グロー）")
        self._glow_check.toggled.connect(self._on_glow_enabled)
        st.addWidget(self._glow_check)

        self._glow_strength = QSlider(Qt.Orientation.Horizontal)
        self._glow_strength.setRange(0, 100)
        self._glow_strength.setValue(80)
        self._glow_strength.valueChanged.connect(self._on_glow_param)
        st.addWidget(self._labeled_slider("強度 %", self._glow_strength))

        glow_size_row = QHBoxLayout()
        glow_size_row.addWidget(QLabel("サイズ"))
        self._glow_size = QSpinBox()
        self._glow_size.setRange(1, 50)
        self._glow_size.setValue(8)
        self._glow_size.valueChanged.connect(self._on_glow_param)
        glow_size_row.addWidget(self._glow_size)
        st.addLayout(glow_size_row)

        glow_color_row = QHBoxLayout()
        glow_color_row.addWidget(QLabel("色"))
        self._glow_color_btn = QPushButton()
        self._glow_color_btn.setFixedSize(40, 24)
        self._glow_color_btn.clicked.connect(self._on_glow_color_pick)
        self._glow_color = QColor(255, 255, 200, 255)
        self._refresh_glow_color_btn()
        glow_color_row.addWidget(self._glow_color_btn)
        glow_color_row.addStretch()
        st.addLayout(glow_color_row)

        st.addWidget(self._sep())

        # ── ガウシアンぼかし ──
        self._blur_check = QCheckBox("ガウシアンぼかし")
        self._blur_check.toggled.connect(self._on_blur_enabled)
        st.addWidget(self._blur_check)

        self._blur_strength = QSlider(Qt.Orientation.Horizontal)
        self._blur_strength.setRange(0, 100)
        self._blur_strength.setValue(100)
        self._blur_strength.valueChanged.connect(self._on_blur_param)
        st.addWidget(self._labeled_slider("強度 %", self._blur_strength))

        blur_r_row = QHBoxLayout()
        blur_r_row.addWidget(QLabel("半径"))
        self._blur_radius = QSpinBox()
        self._blur_radius.setRange(1, 50)
        self._blur_radius.setValue(3)
        self._blur_radius.valueChanged.connect(self._on_blur_param)
        blur_r_row.addWidget(self._blur_radius)
        st.addLayout(blur_r_row)

        st.addWidget(self._sep())

        # ── 色調補正 ──
        self._hsl_check = QCheckBox("色調補正")
        self._hsl_check.toggled.connect(self._on_hsl_enabled)
        st.addWidget(self._hsl_check)

        self._hsl_hue = QSlider(Qt.Orientation.Horizontal)
        self._hsl_hue.setRange(-180, 180)
        self._hsl_hue.setValue(0)
        self._hsl_hue.valueChanged.connect(self._on_hsl_param)
        st.addWidget(self._labeled_slider("色相", self._hsl_hue))

        self._hsl_sat = QSlider(Qt.Orientation.Horizontal)
        self._hsl_sat.setRange(-100, 100)
        self._hsl_sat.setValue(0)
        self._hsl_sat.valueChanged.connect(self._on_hsl_param)
        st.addWidget(self._labeled_slider("彩度", self._hsl_sat))

        self._hsl_light = QSlider(Qt.Orientation.Horizontal)
        self._hsl_light.setRange(-100, 100)
        self._hsl_light.setValue(0)
        self._hsl_light.valueChanged.connect(self._on_hsl_param)
        st.addWidget(self._labeled_slider("明度", self._hsl_light))

        st.addWidget(self._sep())
        st.addStretch()

        settings_scroll.setWidget(settings_inner)
        self._tabs.addTab(settings_scroll, "レイヤー設定")

        # ── タブ3: アクション ────────────────────────────────────────────────
        self._action_panel = ActionPanel(layer_stack)
        self._action_panel.action_executed.connect(self._on_action_executed)
        self._action_panel.structure_will_change.connect(
            lambda: self.structure_will_change.emit())
        self._tabs.addTab(self._action_panel, "アクション")

        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._rows: list[tuple[LayerRow, Layer | GroupLayer, list[int]]] = []
        # (widget, layer, path)
        self._drop_target_idx: int = -1  # ドロップ先の行インデックス (-1=無効)
        self._drop_into_group: bool = False  # グループ内にドロップ
        self.setAcceptDrops(True)
        self._inner.setAcceptDrops(True)
        self._scroll.setAcceptDrops(True)
        self.refresh()

    # ── フラットな行リストを再構築 ────────────────────────────────────────────

    def refresh(self):
        # 既存ウィジェットを削除
        while self._inner_layout.count() > 1:  # stretch を残す
            item = self._inner_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._rows.clear()

        active = self.layer_stack.active
        self._insert_rows_recursive(self.layer_stack.layers, [], 0, active, 0)

        # 不透明度スライダー更新
        cur = self.layer_stack.active
        if cur:
            self._opacity.blockSignals(True)
            self._opacity.setValue(cur.opacity)
            self._opacity.blockSignals(False)

    def _connect_row(self, row: LayerRow):
        row.visibility_changed.connect(self._on_visibility)
        row.clipping_changed.connect(self._on_clipping)
        row.reference_changed.connect(self._on_reference)
        row.rename_requested.connect(self._on_rename)
        row.select_requested.connect(self._on_select)
        row.toggle_collapse.connect(self._on_toggle_collapse)

    def _insert_rows_recursive(self, items: list, path_prefix: list[int],
                                depth: int, active, insert_pos: int) -> int:
        """再帰的にレイヤー行を追加する。insert_pos を返す。"""
        for i, layer in enumerate(items):
            cur_path = path_prefix + [i]
            clipped = (not layer.is_group
                       and getattr(layer, 'clipping', False)
                       and i < len(items) - 1
                       and not items[i + 1].is_group)
            row = LayerRow(layer,
                           indent=_INDENT * depth,
                           is_active=(layer is active),
                           clipped=clipped)
            row._path = cur_path  # type: ignore
            self._connect_row(row)
            self._inner_layout.insertWidget(insert_pos, row)
            self._rows.append((row, layer, cur_path))
            insert_pos += 1

            if layer.is_group and not layer.collapsed:
                insert_pos = self._insert_rows_recursive(
                    layer.children, cur_path, depth + 1, active, insert_pos)
        return insert_pos

    def _sep(self) -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        return f

    @staticmethod
    def _labeled_slider(label: str, slider: QSlider) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setFixedWidth(50)
        h.addWidget(lbl)
        h.addWidget(slider)
        val_lbl = QLabel(str(slider.value()))
        val_lbl.setFixedWidth(35)
        slider.valueChanged.connect(lambda v: val_lbl.setText(str(v)))
        h.addWidget(val_lbl)
        return w

    def _refresh_shadow_color_btn(self):
        c = self._shadow_color
        self._shadow_color_btn.setStyleSheet(
            f"background-color: {c.name()}; border: 1px solid #888;")

    def _refresh_glow_color_btn(self):
        c = self._glow_color
        self._glow_color_btn.setStyleSheet(
            f"background-color: {c.name()}; border: 1px solid #888;")

    def _refresh_border_color_btn(self):
        c = self._border_color
        self._border_color_btn.setStyleSheet(
            f"background-color: {c.name()}; border: 1px solid #888;")

    def _on_action_executed(self):
        self.refresh()
        self.layers_changed.emit()
        self.layer_structure_changed.emit()

    def _on_tab_changed(self, index: int):
        if index == 1:
            self._sync_settings_tab()

    def _sync_settings_tab(self):
        active = self.layer_stack.active
        is_layer = active is not None and not active.is_group
        # 全コントロールの有効/無効
        for w in (self._blend_combo, self._border_check, self._border_size,
                  self._border_color_btn, self._shadow_check, self._shadow_strength,
                  self._shadow_ox, self._shadow_oy, self._shadow_blur,
                  self._shadow_color_btn, self._glow_check, self._glow_strength,
                  self._glow_size, self._glow_color_btn, self._blur_check,
                  self._blur_strength, self._blur_radius, self._hsl_check,
                  self._hsl_hue, self._hsl_sat, self._hsl_light):
            w.setEnabled(is_layer)
        if not is_layer:
            return
        lyr: Layer = active  # type: ignore
        def _block_set(widget, value):
            widget.blockSignals(True)
            if hasattr(widget, 'setChecked'):
                widget.setChecked(value)
            elif hasattr(widget, 'setCurrentIndex'):
                widget.setCurrentIndex(value)
            else:
                widget.setValue(value)
            widget.blockSignals(False)
        _block_set(self._blend_combo, BLEND_KEYS.index(lyr.blend_mode) if lyr.blend_mode in BLEND_KEYS else 0)
        _block_set(self._border_check, lyr.border_enabled)
        _block_set(self._border_size, lyr.border_size)
        self._border_color = lyr.border_color
        self._refresh_border_color_btn()
        _block_set(self._shadow_check, lyr.shadow_enabled)
        _block_set(self._shadow_strength, lyr.shadow_strength)
        _block_set(self._shadow_ox, lyr.shadow_offset_x)
        _block_set(self._shadow_oy, lyr.shadow_offset_y)
        _block_set(self._shadow_blur, lyr.shadow_blur)
        self._shadow_color = lyr.shadow_color
        self._refresh_shadow_color_btn()
        _block_set(self._glow_check, lyr.glow_enabled)
        _block_set(self._glow_strength, lyr.glow_strength)
        _block_set(self._glow_size, lyr.glow_size)
        self._glow_color = lyr.glow_color
        self._refresh_glow_color_btn()
        _block_set(self._blur_check, lyr.blur_enabled)
        _block_set(self._blur_strength, lyr.blur_strength)
        _block_set(self._blur_radius, lyr.blur_radius)
        _block_set(self._hsl_check, lyr.hsl_enabled)
        _block_set(self._hsl_hue, lyr.hsl_hue)
        _block_set(self._hsl_sat, lyr.hsl_saturation)
        _block_set(self._hsl_light, lyr.hsl_lightness)

    def _on_border_enabled(self, value: bool):
        active = self.layer_stack.active
        if active and not active.is_group:
            active.border_enabled = value  # type: ignore
            self.layers_changed.emit()

    def _on_border_size(self, value: int):
        active = self.layer_stack.active
        if active and not active.is_group:
            active.border_size = value  # type: ignore
            self.layers_changed.emit()

    def _on_border_color_pick(self):
        c = QColorDialog.getColor(self._border_color, self, "縁取り色を選択",
                                   QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self._border_color = c
            self._refresh_border_color_btn()
            active = self.layer_stack.active
            if active and not active.is_group:
                active.border_color = c  # type: ignore
                self.layers_changed.emit()

    def _on_blend_mode(self, index: int):
        active = self.layer_stack.active
        if active and not active.is_group:
            active.blend_mode = BLEND_KEYS[index]  # type: ignore
            self.layers_changed.emit()

    def _on_shadow_enabled(self, value: bool):
        active = self.layer_stack.active
        if active and not active.is_group:
            active.shadow_enabled = value  # type: ignore
            self.layers_changed.emit()

    def _on_shadow_param(self):
        active = self.layer_stack.active
        if active and not active.is_group:
            active.shadow_strength = self._shadow_strength.value()  # type: ignore
            active.shadow_offset_x = self._shadow_ox.value()  # type: ignore
            active.shadow_offset_y = self._shadow_oy.value()  # type: ignore
            active.shadow_blur = self._shadow_blur.value()  # type: ignore
            self.layers_changed.emit()

    def _on_shadow_color_pick(self):
        c = QColorDialog.getColor(self._shadow_color, self, "影色を選択",
                                   QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self._shadow_color = c
            self._refresh_shadow_color_btn()
            active = self.layer_stack.active
            if active and not active.is_group:
                active.shadow_color = c  # type: ignore
                self.layers_changed.emit()

    def _on_glow_enabled(self, value: bool):
        active = self.layer_stack.active
        if active and not active.is_group:
            active.glow_enabled = value  # type: ignore
            self.layers_changed.emit()

    def _on_glow_param(self):
        active = self.layer_stack.active
        if active and not active.is_group:
            active.glow_strength = self._glow_strength.value()  # type: ignore
            active.glow_size = self._glow_size.value()  # type: ignore
            self.layers_changed.emit()

    def _on_glow_color_pick(self):
        c = QColorDialog.getColor(self._glow_color, self, "光彩色を選択",
                                   QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self._glow_color = c
            self._refresh_glow_color_btn()
            active = self.layer_stack.active
            if active and not active.is_group:
                active.glow_color = c  # type: ignore
                self.layers_changed.emit()

    def _on_blur_enabled(self, value: bool):
        active = self.layer_stack.active
        if active and not active.is_group:
            active.blur_enabled = value  # type: ignore
            self.layers_changed.emit()

    def _on_blur_param(self):
        active = self.layer_stack.active
        if active and not active.is_group:
            active.blur_strength = self._blur_strength.value()  # type: ignore
            active.blur_radius = self._blur_radius.value()  # type: ignore
            self.layers_changed.emit()

    def _on_hsl_enabled(self, value: bool):
        active = self.layer_stack.active
        if active and not active.is_group:
            active.hsl_enabled = value  # type: ignore
            self.layers_changed.emit()

    def _on_hsl_param(self):
        active = self.layer_stack.active
        if active and not active.is_group:
            active.hsl_hue = self._hsl_hue.value()  # type: ignore
            active.hsl_saturation = self._hsl_sat.value()  # type: ignore
            active.hsl_lightness = self._hsl_light.value()  # type: ignore
            self.layers_changed.emit()

    def refresh_thumbs(self):
        """描画後にサムネイルだけ更新する（重い composite を最小化）。"""
        for row, layer, _ in self._rows:
            row.refresh_thumb()

    def set_opacity(self, value: int):
        """数字キー等の外部操作でスライダーを同期する。"""
        self._opacity.blockSignals(True)
        self._opacity.setValue(value)
        self._opacity.blockSignals(False)

    # ── イベントハンドラ ──────────────────────────────────────────────────────

    def _on_select(self, layer: Layer | GroupLayer):
        ls = self.layer_stack
        path = ls.find_path(layer)
        if path is not None:
            ls.set_active_path(path)
        # アクティブ行のハイライト更新
        active = ls.active
        for row, lyr, _ in self._rows:
            row.set_active(lyr is active)
            row.update()
        # 不透明度スライダー
        if active:
            self._opacity.blockSignals(True)
            self._opacity.setValue(active.opacity)
            self._opacity.blockSignals(False)
        # レイヤー設定タブが開いていれば同期
        if self._tabs.currentIndex() == 1:
            self._sync_settings_tab()
        self.layers_changed.emit()

    def _on_toggle_collapse(self, group: GroupLayer):
        group.collapsed = not group.collapsed
        if group.collapsed:
            ls = self.layer_stack
            group_path = ls.find_path(group)
            if group_path is not None:
                ap = ls.active_path
                if len(ap) > len(group_path) and ap[:len(group_path)] == group_path:
                    ls.set_active_path(group_path)
        self.refresh()
        self.layers_changed.emit()

    def _on_visibility(self, layer: Layer | GroupLayer, visible: bool):
        layer.visible = visible
        self.layers_changed.emit()

    def _on_clipping(self, layer: Layer | GroupLayer, clipping: bool):
        layer.clipping = clipping
        self.refresh()
        self.layers_changed.emit()

    def _on_reference(self, layer: Layer | GroupLayer, reference: bool):
        layer.reference = reference
        self.layers_changed.emit()

    def _on_opacity(self, value: int):
        active = self.layer_stack.active
        if active:
            active.opacity = value
            self.layers_changed.emit()

    def _on_rename(self, layer: Layer | GroupLayer):
        name, ok = QInputDialog.getText(
            self, "レイヤー名変更", "新しい名前:", text=layer.name)
        if ok and name.strip():
            layer.name = name.strip()
            self.refresh()

    # ── ボタン操作 ────────────────────────────────────────────────────────────

    def _current_path(self) -> list[int]:
        return list(self.layer_stack.active_path)

    def _get_container_and_index(self, path: list[int]) -> tuple[list, int]:
        """パスから親コンテナとインデックスを返す。"""
        container = self.layer_stack.layers
        for idx in path[:-1]:
            if 0 <= idx < len(container) and container[idx].is_group:
                container = container[idx].children
            else:
                break
        return container, path[-1] if path else 0

    def _add(self):
        self.structure_will_change.emit()
        ls = self.layer_stack
        path = self._current_path()
        container, idx = self._get_container_and_index(path)
        new_layer = Layer(f"レイヤー {len(container) + 1}", ls.width, ls.height)
        insert_at = min(idx, len(container))
        container.insert(insert_at, new_layer)
        new_path = path[:-1] + [insert_at]
        ls.set_active_path(new_path)
        self.refresh()
        self.layers_changed.emit()

    def _add_group(self):
        self.structure_will_change.emit()
        ls = self.layer_stack
        path = self._current_path()
        container, idx = self._get_container_and_index(path)
        group = GroupLayer(f"グループ {len(container) + 1}", ls.width, ls.height)
        insert_at = min(idx, len(container))
        container.insert(insert_at, group)
        new_path = path[:-1] + [insert_at]
        ls.set_active_path(new_path)
        self.refresh()
        self.layers_changed.emit()

    @staticmethod
    def _copy_layer_props(src: Layer, dst: Layer):
        dst.visible = src.visible
        dst.opacity = src.opacity
        dst.clipping = src.clipping
        dst.reference = src.reference
        dst.blend_mode = src.blend_mode
        dst.offset_x = src.offset_x
        dst.offset_y = src.offset_y
        dst.image = src.image.copy()
        dst.border_enabled = src.border_enabled
        dst.border_size = src.border_size
        dst.border_color = QColor(src.border_color)
        dst.shadow_enabled = src.shadow_enabled
        dst.shadow_color = QColor(src.shadow_color)
        dst.shadow_offset_x = src.shadow_offset_x
        dst.shadow_offset_y = src.shadow_offset_y
        dst.shadow_blur = src.shadow_blur
        dst.shadow_strength = src.shadow_strength
        dst.glow_enabled = src.glow_enabled
        dst.glow_color = QColor(src.glow_color)
        dst.glow_size = src.glow_size
        dst.glow_strength = src.glow_strength
        dst.blur_enabled = src.blur_enabled
        dst.blur_radius = src.blur_radius
        dst.blur_strength = src.blur_strength
        dst.hsl_enabled = src.hsl_enabled
        dst.hsl_hue = src.hsl_hue
        dst.hsl_saturation = src.hsl_saturation
        dst.hsl_lightness = src.hsl_lightness

    def _deep_copy_layer(self, src, ls, is_root: bool = True) -> Layer | GroupLayer:
        suffix = " コピー" if is_root else ""
        if src.is_group:
            new_g = GroupLayer(src.name + suffix, ls.width, ls.height)
            new_g.visible = src.visible
            new_g.opacity = src.opacity
            new_g.clipping = src.clipping
            new_g.reference = src.reference
            new_g.collapsed = src.collapsed
            for child in src.children:
                new_g.children.append(self._deep_copy_layer(child, ls, False))
            return new_g
        else:
            c = Layer(src.name + suffix, ls.width, ls.height)
            self._copy_layer_props(src, c)
            return c

    def _duplicate(self):
        self.structure_will_change.emit()
        ls = self.layer_stack
        path = self._current_path()
        container, idx = self._get_container_and_index(path)
        if idx >= len(container):
            return
        src = container[idx]
        copy = self._deep_copy_layer(src, ls)
        container.insert(idx, copy)
        ls.set_active_path(path[:-1] + [idx])
        self.refresh()
        self.layers_changed.emit()

    def _remove(self):
        self.structure_will_change.emit()
        ls = self.layer_stack
        path = self._current_path()
        container, idx = self._get_container_and_index(path)
        if len(path) == 1 and len(container) <= 1:
            return
        if idx >= len(container):
            return
        container.pop(idx)
        if not container:
            ls.set_active_path(path[:-1])
        else:
            new_idx = min(idx, len(container) - 1)
            ls.set_active_path(path[:-1] + [new_idx])
        self.refresh()
        self.layer_structure_changed.emit()
        self.layers_changed.emit()

    def _move_up(self):
        ls = self.layer_stack
        path = self._current_path()
        container, idx = self._get_container_and_index(path)
        if 0 < idx < len(container):
            container[idx], container[idx - 1] = container[idx - 1], container[idx]
            ls.set_active_path(path[:-1] + [idx - 1])
        self.refresh()
        self.layers_changed.emit()

    def _move_down(self):
        ls = self.layer_stack
        path = self._current_path()
        container, idx = self._get_container_and_index(path)
        if 0 <= idx < len(container) - 1:
            container[idx], container[idx + 1] = container[idx + 1], container[idx]
            ls.set_active_path(path[:-1] + [idx + 1])
        self.refresh()
        self.layers_changed.emit()

    def _add_to_group(self):
        """アクティブレイヤーを、同じコンテナ内の直上のグループに移動する。"""
        ls = self.layer_stack
        path = self._current_path()
        container, idx = self._get_container_and_index(path)
        if idx <= 0 or idx >= len(container):
            return
        above = container[idx - 1]
        if not above.is_group:
            return
        layer = container.pop(idx)
        above.children.append(layer)
        new_child_idx = len(above.children) - 1
        ls.set_active_path(path[:-1] + [idx - 1, new_child_idx])
        self.refresh()
        self.layers_changed.emit()

    def _remove_from_group(self):
        """グループ内のアクティブレイヤーを一つ上の階層に取り出す。"""
        ls = self.layer_stack
        path = self._current_path()
        if len(path) < 2:
            return
        container, idx = self._get_container_and_index(path)
        if idx >= len(container):
            return
        layer = container.pop(idx)
        parent_path = path[:-1]
        parent_container, parent_idx = self._get_container_and_index(parent_path)
        insert_pos = parent_idx + 1
        parent_container.insert(insert_pos, layer)
        ls.set_active_path(parent_path[:-1] + [insert_pos])
        self.refresh()
        self.layers_changed.emit()

    # ── ドラッグ＆ドロップ ──────────────────────────────────────────────────────

    def _row_at_pos(self, pos) -> int:
        """スクロールエリア内の座標からドロップ先の行インデックスを返す。"""
        scroll_pos = self._scroll.mapFrom(self, pos)
        inner_pos = self._inner.mapFrom(self._scroll.viewport(), scroll_pos)
        for i, (row, _, _) in enumerate(self._rows):
            ry = row.y()
            rh = row.height()
            if inner_pos.y() < ry + rh // 2:
                return i
        return len(self._rows)

    def _row_hit(self, pos) -> tuple[int, bool]:
        """行インデックスとグループ内ドロップかを返す。"""
        scroll_pos = self._scroll.mapFrom(self, pos)
        inner_pos = self._inner.mapFrom(self._scroll.viewport(), scroll_pos)
        for i, (row, layer, _) in enumerate(self._rows):
            ry = row.y()
            rh = row.height()
            if inner_pos.y() < ry + rh:
                mid = ry + rh // 2
                if layer.is_group and abs(inner_pos.y() - mid) < rh // 4:
                    return i, True
                if inner_pos.y() < mid:
                    return i, False
                return i + 1, False
        return len(self._rows), False

    def dragEnterEvent(self, event):
        if event.mimeData().text() == "layer_drag":
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().text() != "layer_drag":
            return
        event.acceptProposedAction()
        idx, into_group = self._row_hit(event.position().toPoint())
        if idx != self._drop_target_idx or into_group != self._drop_into_group:
            self._drop_target_idx = idx
            self._drop_into_group = into_group
            self._update_drop_indicator()

    def dragLeaveEvent(self, event):
        self._drop_target_idx = -1
        self._drop_into_group = False
        self._update_drop_indicator()

    def dropEvent(self, event):
        if event.mimeData().text() != "layer_drag":
            return
        event.acceptProposedAction()
        drop_idx = self._drop_target_idx
        into_group = self._drop_into_group
        self._drop_target_idx = -1
        self._drop_into_group = False
        self._update_drop_indicator()

        if drop_idx < 0:
            return

        ls = self.layer_stack
        src_path = self._current_path()
        src_container, src_idx = self._get_container_and_index(src_path)
        if src_idx >= len(src_container):
            return

        if into_group and 0 <= drop_idx < len(self._rows):
            _, target_layer, target_path = self._rows[drop_idx]
            if target_layer.is_group and target_layer is not src_container[src_idx]:
                self.structure_will_change.emit()
                layer = src_container.pop(src_idx)
                target_layer.children.insert(0, layer)
                new_path = target_path + [0]
                ls.set_active_path(new_path)
                self.refresh()
                self.layers_changed.emit()
            return

        if drop_idx <= 0:
            dst_path = [0]
        elif drop_idx >= len(self._rows):
            last_path = self._rows[-1][2]
            dst_container, _ = self._get_container_and_index(last_path)
            dst_path = last_path[:-1] + [len(dst_container)]
        else:
            dst_path = list(self._rows[drop_idx][2])

        dst_container, dst_idx = self._get_container_and_index(dst_path)

        if dst_container is src_container:
            if dst_idx == src_idx or dst_idx == src_idx + 1:
                return
            self.structure_will_change.emit()
            layer = src_container.pop(src_idx)
            insert_at = dst_idx if dst_idx < src_idx else dst_idx - 1
            dst_container.insert(insert_at, layer)
            ls.set_active_path(dst_path[:-1] + [insert_at])
        else:
            self.structure_will_change.emit()
            layer = src_container.pop(src_idx)
            dst_container.insert(dst_idx, layer)
            ls.set_active_path(dst_path[:-1] + [dst_idx])

        self.refresh()
        self.layers_changed.emit()

    def _update_drop_indicator(self):
        for row, layer, _ in self._rows:
            row.setStyleSheet("")
        if self._drop_target_idx < 0:
            return
        if self._drop_into_group and 0 <= self._drop_target_idx < len(self._rows):
            row = self._rows[self._drop_target_idx][0]
            row.setStyleSheet("LayerRow { border: 2px solid #4a90d9; }")
        elif 0 <= self._drop_target_idx < len(self._rows):
            row = self._rows[self._drop_target_idx][0]
            row.setStyleSheet("LayerRow { border-top: 3px solid #4a90d9; }")
        elif self._drop_target_idx >= len(self._rows) and self._rows:
            row = self._rows[-1][0]
            row.setStyleSheet("LayerRow { border-bottom: 3px solid #4a90d9; }")

    def _merge_down(self):
        self.merge_down_requested.emit()

    def _merge_all_visible(self):
        self.merge_all_requested.emit()

    def _merge_folder(self):
        self.merge_folder_requested.emit()
