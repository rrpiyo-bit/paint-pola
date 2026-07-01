"""アクション機能 — ワンクリックで複雑なレイヤー操作を実行する。"""
from __future__ import annotations

import random

import numpy as np
import cv2

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QDialog,
                              QLabel, QSpinBox, QCheckBox, QHBoxLayout,
                              QDialogButtonBox, QGroupBox, QFormLayout,
                              QComboBox, QColorDialog, QFrame, QMessageBox)
from PyQt6.QtGui import (QImage, QPainter, QColor, QTransform, QLinearGradient,
                          QRadialGradient, QBrush, QPen)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF

from layer import Layer, GroupLayer, LayerStack


def _copy_offset(src: Layer, dst: Layer):
    """ソースレイヤーのオフセットを新しいレイヤーにコピーする。"""
    dst.offset_x = getattr(src, 'offset_x', 0)
    dst.offset_y = getattr(src, 'offset_y', 0)


def _find_top_index(layer_stack: LayerStack, layer: Layer) -> int:
    """トップレベルリストから layer の位置を探す。子レイヤーの場合は親グループの位置を返す。"""
    def _contains(group, target) -> bool:
        for child in group.children:
            if child is target:
                return True
            if child.is_group and _contains(child, target):
                return True
        return False

    try:
        return layer_stack.layers.index(layer)
    except ValueError:
        for i, top in enumerate(layer_stack.layers):
            if top.is_group and _contains(top, layer):
                return i
        return len(layer_stack.layers) - 1


# ═══════════════════════════════════════════════════════════════════════════════
# ユーティリティ
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_color_overlay(image: QImage, color: QColor) -> QImage:
    """レイヤー画像に色をクリッピング的に乗せる（元画像のアルファを維持）。"""
    result = image.copy()
    p = QPainter(result)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(result.rect(), color)
    p.end()
    return result


def _shift_image(image: QImage, dx: int, dy: int,
                 angle: float, scale: float) -> QImage:
    """画像を移動・回転・拡縮して返す。"""
    w, h = image.width(), image.height()
    cx, cy = w / 2.0, h / 2.0
    t = QTransform()
    t.translate(cx + dx, cy + dy)
    t.rotate(angle)
    t.scale(scale, scale)
    t.translate(-cx, -cy)
    result = QImage(w, h, QImage.Format.Format_ARGB32)
    result.fill(Qt.GlobalColor.transparent)
    p = QPainter(result)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.setTransform(t)
    p.drawImage(0, 0, image)
    p.end()
    return result


def _qimage_to_array(img: QImage) -> np.ndarray:
    img32 = img.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = img32.width(), img32.height()
    if w == 0 or h == 0:
        return np.zeros((max(h, 1), max(w, 1), 4), dtype=np.uint8)
    ptr = img32.bits()
    ptr.setsize(h * w * 4)
    return np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()


def _array_to_qimage(arr: np.ndarray) -> QImage:
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, w * 4,
                  QImage.Format.Format_ARGB32).copy()


def _dilate_alpha(image: QImage, radius: int) -> QImage:
    """画像のアルファチャンネルを膨張させた画像を返す。"""
    if radius <= 0:
        return image.copy()
    arr = _qimage_to_array(image)
    alpha = arr[:, :, 3]
    ksize = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dilated = cv2.dilate(alpha, kernel)
    out = np.zeros_like(arr)
    out[:, :, 3] = dilated
    out[:, :, :3] = arr[:, :, :3]
    return _array_to_qimage(out)


def _blur_image(image: QImage, radius: int) -> QImage:
    """ガウシアンブラーを適用した画像を返す。"""
    if radius <= 0:
        return image.copy()
    arr = _qimage_to_array(image)
    ksize = radius * 2 + 1
    blurred = cv2.GaussianBlur(arr, (ksize, ksize), 0)
    return _array_to_qimage(blurred)


def _std_buttons():
    return QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
    )


def _color_button(color: QColor, parent: QWidget) -> QPushButton:
    btn = QPushButton()
    btn.setFixedSize(60, 24)
    btn._color = color

    def _update():
        btn.setStyleSheet(
            f"background-color: {btn._color.name()}; border: 1px solid #888;")

    def _pick():
        c = QColorDialog.getColor(btn._color, parent, "色を選択",
                                  QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            btn._color = c
            _update()

    btn.clicked.connect(_pick)
    _update()
    return btn


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 線画ずらし（色収差）
# ═══════════════════════════════════════════════════════════════════════════════

class _ChromaLayerRow(QWidget):
    """線画ずらし1本分の色・太さ設定。"""
    removed = pyqtSignal(object)

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self._color_btn = _color_button(color, self)
        row.addWidget(QLabel("色"))
        row.addWidget(self._color_btn)
        self._thickness = QSpinBox()
        self._thickness.setRange(-10, 10)
        self._thickness.setValue(0)
        self._thickness.setSuffix(" px")
        self._thickness.setToolTip("正=太く 負=細く 0=そのまま")
        row.addWidget(QLabel("太さ"))
        row.addWidget(self._thickness)
        rm = QPushButton("×")
        rm.setFixedWidth(24)
        rm.clicked.connect(lambda: self.removed.emit(self))
        row.addWidget(rm)

    def color(self) -> QColor:
        return self._color_btn._color

    def thickness(self) -> int:
        return self._thickness.value()


class ChromaShiftDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("線画ずらし")
        self.setMinimumWidth(360)
        layout = QVBoxLayout(self)

        # ずらし量
        shift_group = QGroupBox("移動")
        sf = QFormLayout(shift_group)
        self._shift_px = QSpinBox()
        self._shift_px.setRange(1, 50)
        self._shift_px.setValue(5)
        self._shift_px.setSuffix(" px")
        sf.addRow("ずらし量（最大）", self._shift_px)
        layout.addWidget(shift_group)

        # 線画本数・色・太さ
        layer_group = QGroupBox("ずらす線画")
        ll = QVBoxLayout(layer_group)
        self._layer_rows: list[_ChromaLayerRow] = []
        self._rows_layout = QVBoxLayout()
        ll.addLayout(self._rows_layout)
        defaults = [
            QColor(255, 60, 60, 200),
            QColor(60, 60, 255, 200),
            QColor(255, 230, 60, 200),
        ]
        for c in defaults:
            self._add_layer_row(c)
        add_btn = QPushButton("＋ 線を追加")
        add_btn.setFixedHeight(24)
        add_btn.clicked.connect(lambda: self._add_layer_row(QColor(180, 180, 180, 200)))
        ll.addWidget(add_btn)
        layout.addWidget(layer_group)

        # 回転
        rot_group = QGroupBox("回転")
        rl = QVBoxLayout(rot_group)
        self._rot_enabled = QCheckBox("回転を有効にする")
        rl.addWidget(self._rot_enabled)
        rf = QFormLayout()
        self._rot_max = QSpinBox()
        self._rot_max.setRange(1, 30)
        self._rot_max.setValue(3)
        self._rot_max.setSuffix(" °")
        self._rot_max.setEnabled(False)
        rf.addRow("最大角度", self._rot_max)
        rl.addLayout(rf)
        self._rot_enabled.toggled.connect(self._rot_max.setEnabled)
        layout.addWidget(rot_group)

        # 拡縮
        scale_group = QGroupBox("拡縮")
        sl = QVBoxLayout(scale_group)
        self._scale_enabled = QCheckBox("拡縮を有効にする")
        sl.addWidget(self._scale_enabled)
        scf = QFormLayout()
        self._scale_max = QSpinBox()
        self._scale_max.setRange(1, 20)
        self._scale_max.setValue(3)
        self._scale_max.setSuffix(" %")
        self._scale_max.setEnabled(False)
        scf.addRow("最大変化率", self._scale_max)
        sl.addLayout(scf)
        self._scale_enabled.toggled.connect(self._scale_max.setEnabled)
        layout.addWidget(scale_group)

        desc = QLabel("選択中のレイヤーをコピーしてフォルダにまとめ、\n"
                      "色ずれを作ります（色収差エフェクト）。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_layer_row(self, color: QColor):
        row = _ChromaLayerRow(color, self)
        row.removed.connect(self._remove_layer_row)
        self._rows_layout.addWidget(row)
        self._layer_rows.append(row)

    def _remove_layer_row(self, row: _ChromaLayerRow):
        if len(self._layer_rows) <= 1:
            return
        self._rows_layout.removeWidget(row)
        self._layer_rows.remove(row)
        row.deleteLater()

    def params(self) -> dict:
        layers = [{"color": r.color(), "thickness": r.thickness()}
                  for r in self._layer_rows]
        return {
            "shift_px": self._shift_px.value(),
            "layers": layers,
            "rotate": self._rot_enabled.isChecked(),
            "rotate_max": self._rot_max.value(),
            "scale": self._scale_enabled.isChecked(),
            "scale_max": self._scale_max.value(),
        }


def _adjust_thickness(img: QImage, thickness: int) -> QImage:
    """線画の太さを調整する。正=膨張（太く）、負=収縮（細く）。"""
    if thickness == 0:
        return img
    arr = _qimage_to_array(img)
    if arr.size == 0:
        return img
    alpha = arr[:, :, 3]
    abs_t = abs(thickness)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs_t * 2 + 1, abs_t * 2 + 1))
    if thickness > 0:
        alpha = cv2.dilate(alpha, kernel, iterations=1)
    else:
        alpha = cv2.erode(alpha, kernel, iterations=1)
    arr[:, :, 3] = alpha
    return _array_to_qimage(arr)


def execute_chroma_shift(layer_stack: LayerStack, source_layer: Layer,
                         params: dict) -> GroupLayer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    shift_px = params["shift_px"]
    do_rotate = params.get("rotate", False)
    rot_max = params.get("rotate_max", 0) if do_rotate else 0.0
    do_scale = params.get("scale", False)
    scale_max = params.get("scale_max", 0) if do_scale else 0.0

    layer_defs = params.get("layers")
    if not layer_defs:
        layer_defs = [
            {"color": QColor(255, 60, 60, 200), "thickness": 0},
            {"color": QColor(60, 60, 255, 200), "thickness": 0},
            {"color": QColor(255, 230, 60, 200), "thickness": 0},
        ]

    src_idx = _find_top_index(layer_stack, source_layer)
    group = GroupLayer(f"{source_layer.name} - 線画ずらし", w, h)

    for i, ld in enumerate(layer_defs):
        color: QColor = ld["color"]
        thickness: int = ld.get("thickness", 0)
        base = _adjust_thickness(src_img.copy(), thickness)
        colored_img = _apply_color_overlay(base, color)
        dx = random.randint(-shift_px, shift_px)
        dy = random.randint(-shift_px, shift_px)
        angle = random.uniform(-rot_max, rot_max) if do_rotate else 0.0
        scale = 1.0 + random.uniform(-scale_max, scale_max) / 100.0 if do_scale else 1.0
        shifted = _shift_image(colored_img, dx, dy, angle, scale)
        layer = Layer(f"ずらし{i+1}", w, h)
        layer.image = shifted
        layer.blend_mode = "screen"
        _copy_offset(source_layer, layer)
        group.children.append(layer)

    top = Layer(f"{source_layer.name} (元)", w, h)
    top.image = src_img.copy()
    _copy_offset(source_layer, top)
    group.children.insert(0, top)

    layer_stack.layers.insert(src_idx, group)
    layer_stack.active_path = [src_idx]
    source_layer.visible = False
    return group


# ═══════════════════════════════════════════════════════════════════════════════
# 2. グロー/発光
# ═══════════════════════════════════════════════════════════════════════════════

class GlowDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("グロー / 発光")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._glow_color_btn = _color_button(QColor(255, 255, 200), self)
        form.addRow("グローの色", self._glow_color_btn)

        self._glow_size = QSpinBox()
        self._glow_size.setRange(2, 50)
        self._glow_size.setValue(12)
        self._glow_size.setSuffix(" px")
        form.addRow("グローサイズ", self._glow_size)

        self._glow_strength = QSpinBox()
        self._glow_strength.setRange(10, 100)
        self._glow_strength.setValue(70)
        self._glow_strength.setSuffix(" %")
        form.addRow("グロー強度", self._glow_strength)

        self._bg_color_btn = _color_button(QColor(20, 20, 30), self)
        form.addRow("背景色", self._bg_color_btn)

        self._bg_opacity = QSpinBox()
        self._bg_opacity.setRange(0, 100)
        self._bg_opacity.setValue(90)
        self._bg_opacity.setSuffix(" %")
        form.addRow("背景不透明度", self._bg_opacity)

        layout.addLayout(form)

        desc = QLabel("線画の周りに発光エフェクト＋暗い背景を\n自動生成します。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "glow_color": self._glow_color_btn._color,
            "glow_size": self._glow_size.value(),
            "glow_strength": self._glow_strength.value(),
            "bg_color": self._bg_color_btn._color,
            "bg_opacity": self._bg_opacity.value(),
        }


def execute_glow(layer_stack: LayerStack, source_layer: Layer,
                 params: dict) -> GroupLayer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    src_idx = _find_top_index(layer_stack, source_layer)

    group = GroupLayer(f"{source_layer.name} - グロー", w, h)

    # 背景レイヤー
    bg = Layer("背景", w, h)
    bg_color = params["bg_color"]
    bg_alpha = int(params["bg_opacity"] * 255 / 100)
    bg_fill = QColor(bg_color.red(), bg_color.green(), bg_color.blue(), bg_alpha)
    p = QPainter(bg.image)
    p.fillRect(0, 0, w, h, bg_fill)
    p.end()
    group.children.append(bg)

    # グローレイヤー
    glow_color = params["glow_color"]
    glow_size = params["glow_size"]
    glow_strength = params["glow_strength"]

    colored = _apply_color_overlay(src_img.copy(), glow_color)
    dilated = _dilate_alpha(colored, glow_size)
    blurred = _blur_image(dilated, glow_size)

    glow_layer = Layer("グロー", w, h)
    glow_layer.image = blurred
    glow_layer.opacity = int(glow_strength * 255 / 100)
    glow_layer.blend_mode = "screen"
    _copy_offset(source_layer, glow_layer)
    group.children.insert(0, glow_layer)

    top = Layer(f"{source_layer.name} (元)", w, h)
    top.image = src_img.copy()
    _copy_offset(source_layer, top)
    group.children.insert(0, top)

    _copy_offset(source_layer, bg)

    layer_stack.layers.insert(src_idx, group)
    layer_stack.active_path = [src_idx]
    source_layer.visible = False
    return group


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 影付け（ドロップシャドウ一括）
# ═══════════════════════════════════════════════════════════════════════════════

class DropShadowDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("影付け")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._shadow_color_btn = _color_button(QColor(0, 0, 0, 160), self)
        form.addRow("影の色", self._shadow_color_btn)

        self._offset_x = QSpinBox()
        self._offset_x.setRange(-50, 50)
        self._offset_x.setValue(4)
        self._offset_x.setSuffix(" px")
        form.addRow("X オフセット", self._offset_x)

        self._offset_y = QSpinBox()
        self._offset_y.setRange(-50, 50)
        self._offset_y.setValue(4)
        self._offset_y.setSuffix(" px")
        form.addRow("Y オフセット", self._offset_y)

        self._blur_radius = QSpinBox()
        self._blur_radius.setRange(0, 30)
        self._blur_radius.setValue(5)
        self._blur_radius.setSuffix(" px")
        form.addRow("ぼかし", self._blur_radius)

        self._strength = QSpinBox()
        self._strength.setRange(10, 100)
        self._strength.setValue(80)
        self._strength.setSuffix(" %")
        form.addRow("強度", self._strength)

        layout.addLayout(form)

        desc = QLabel("線画の下に影レイヤーを自動生成します。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "color": self._shadow_color_btn._color,
            "offset_x": self._offset_x.value(),
            "offset_y": self._offset_y.value(),
            "blur": self._blur_radius.value(),
            "strength": self._strength.value(),
        }


def execute_drop_shadow(layer_stack: LayerStack, source_layer: Layer,
                        params: dict) -> GroupLayer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    src_idx = _find_top_index(layer_stack, source_layer)

    group = GroupLayer(f"{source_layer.name} - 影付き", w, h)

    # 影レイヤー
    shadow_color = params["color"]
    colored = _apply_color_overlay(src_img.copy(), shadow_color)

    ox, oy = params["offset_x"], params["offset_y"]
    shifted = QImage(w, h, QImage.Format.Format_ARGB32)
    shifted.fill(Qt.GlobalColor.transparent)
    p = QPainter(shifted)
    p.drawImage(ox, oy, colored)
    p.end()

    blur_r = params["blur"]
    if blur_r > 0:
        shifted = _blur_image(shifted, blur_r)

    shadow_layer = Layer("影", w, h)
    shadow_layer.image = shifted
    shadow_layer.opacity = int(params["strength"] * 255 / 100)
    _copy_offset(source_layer, shadow_layer)
    group.children.append(shadow_layer)

    top = Layer(f"{source_layer.name} (元)", w, h)
    top.image = src_img.copy()
    _copy_offset(source_layer, top)
    group.children.insert(0, top)

    layer_stack.layers.insert(src_idx, group)
    layer_stack.active_path = [src_idx]
    source_layer.visible = False
    return group


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 背景パターン生成
# ═══════════════════════════════════════════════════════════════════════════════

class BgPatternDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("背景パターン生成")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._pattern = QComboBox()
        self._pattern.addItem("ドット", "dots")
        self._pattern.addItem("ストライプ（縦）", "stripes_v")
        self._pattern.addItem("ストライプ（横）", "stripes_h")
        self._pattern.addItem("ストライプ（斜め）", "stripes_d")
        self._pattern.addItem("チェック", "checker")
        self._pattern.addItem("グラデーション（上→下）", "grad_v")
        self._pattern.addItem("グラデーション（左→右）", "grad_h")
        self._pattern.addItem("グラデーション（円形）", "grad_radial")
        form.addRow("パターン", self._pattern)

        self._color1_btn = _color_button(QColor(255, 200, 200), self)
        form.addRow("色1", self._color1_btn)

        self._color2_btn = _color_button(QColor(200, 200, 255), self)
        form.addRow("色2", self._color2_btn)

        self._spacing = QSpinBox()
        self._spacing.setRange(5, 100)
        self._spacing.setValue(20)
        self._spacing.setSuffix(" px")
        form.addRow("間隔 / サイズ", self._spacing)

        layout.addLayout(form)

        desc = QLabel("背景レイヤーを自動生成して\n現在のレイヤーの下に挿入します。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "pattern": self._pattern.currentData(),
            "color1": self._color1_btn._color,
            "color2": self._color2_btn._color,
            "spacing": self._spacing.value(),
        }


def execute_bg_pattern(layer_stack: LayerStack, source_layer,
                       params: dict) -> Layer | None:
    w, h = layer_stack.width, layer_stack.height
    pat = params["pattern"]
    c1 = params["color1"]
    c2 = params["color2"]
    spacing = params["spacing"]

    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(c1)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)

    if pat == "dots":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c2))
        r = max(2, spacing // 4)
        for y in range(0, h + spacing, spacing):
            for x in range(0, w + spacing, spacing):
                p.drawEllipse(x - r, y - r, r * 2, r * 2)

    elif pat == "stripes_v":
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(c2))
        stripe_w = max(2, spacing // 2)
        for x in range(0, w + spacing, spacing):
            p.fillRect(x, 0, stripe_w, h, c2)

    elif pat == "stripes_h":
        p.setPen(Qt.PenStyle.NoPen)
        stripe_h = max(2, spacing // 2)
        for y in range(0, h + spacing, spacing):
            p.fillRect(0, y, w, stripe_h, c2)

    elif pat == "stripes_d":
        pen = QPen(c2, max(2, spacing // 2))
        p.setPen(pen)
        for offset in range(-max(w, h), max(w, h) + spacing, spacing):
            p.drawLine(offset, 0, offset + h, h)

    elif pat == "checker":
        for y in range(0, h + spacing, spacing):
            for x in range(0, w + spacing, spacing):
                if ((x // spacing) + (y // spacing)) % 2 == 0:
                    p.fillRect(x, y, spacing, spacing, c2)

    elif pat == "grad_v":
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, c1)
        grad.setColorAt(1, c2)
        p.fillRect(0, 0, w, h, QBrush(grad))

    elif pat == "grad_h":
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0, c1)
        grad.setColorAt(1, c2)
        p.fillRect(0, 0, w, h, QBrush(grad))

    elif pat == "grad_radial":
        grad = QRadialGradient(QPointF(w / 2, h / 2), max(w, h) / 2)
        grad.setColorAt(0, c1)
        grad.setColorAt(1, c2)
        p.fillRect(0, 0, w, h, QBrush(grad))

    p.end()

    # ソースレイヤーの下に挿入
    bg_layer = Layer("背景パターン", w, h)
    bg_layer.image = img

    if source_layer and not source_layer.is_group:
        try:
            idx = layer_stack.layers.index(source_layer)
            layer_stack.layers.insert(idx + 1, bg_layer)
            return bg_layer
        except ValueError:
            pass

    layer_stack.layers.append(bg_layer)
    return bg_layer


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 線画色変え
# ═══════════════════════════════════════════════════════════════════════════════

class LineColorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("線画色変え")
        self.setMinimumWidth(280)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._color_btn = _color_button(QColor(60, 40, 30), self)
        form.addRow("変換先の色", self._color_btn)

        self._presets = QComboBox()
        self._presets.addItem("カスタム", None)
        self._presets.addItem("茶色（やわらかい）", QColor(80, 50, 30))
        self._presets.addItem("ネイビー（おしゃれ）", QColor(30, 30, 80))
        self._presets.addItem("ワインレッド", QColor(100, 20, 30))
        self._presets.addItem("ダークグリーン", QColor(20, 60, 30))
        self._presets.addItem("グレー", QColor(80, 80, 80))
        self._presets.currentIndexChanged.connect(self._on_preset)
        form.addRow("プリセット", self._presets)

        layout.addLayout(form)

        desc = QLabel("線画（不透明ピクセル）の色を一括変換します。\n"
                      "元のレイヤーはそのまま残り、\n"
                      "色変え済みのコピーが上に作られます。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_preset(self, idx):
        color = self._presets.itemData(idx)
        if color is not None:
            self._color_btn._color = color
            self._color_btn.setStyleSheet(
                f"background-color: {color.name()}; border: 1px solid #888;")

    def params(self) -> dict:
        return {"color": self._color_btn._color}


def execute_line_color(layer_stack: LayerStack, source_layer: Layer,
                       params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    color = params["color"]

    result = _apply_color_overlay(src_img.copy(), color)

    new_layer = Layer(f"{source_layer.name} ({color.name()})", w, h)
    new_layer.image = result
    _copy_offset(source_layer, new_layer)

    src_idx = _find_top_index(layer_stack, source_layer)
    layer_stack.layers.insert(src_idx, new_layer)
    layer_stack.active_path = [src_idx]
    source_layer.visible = False
    return new_layer


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ポップアウト（ステッカー風）
# ═══════════════════════════════════════════════════════════════════════════════

class PopoutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ポップアウト（ステッカー風）")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._outline_size = QSpinBox()
        self._outline_size.setRange(1, 30)
        self._outline_size.setValue(5)
        self._outline_size.setSuffix(" px")
        form.addRow("縁の太さ", self._outline_size)

        self._outline_color_btn = _color_button(QColor(255, 255, 255), self)
        form.addRow("縁の色", self._outline_color_btn)

        self._shadow_enabled = QCheckBox("影をつける")
        self._shadow_enabled.setChecked(True)
        form.addRow("", self._shadow_enabled)

        self._shadow_offset = QSpinBox()
        self._shadow_offset.setRange(1, 20)
        self._shadow_offset.setValue(3)
        self._shadow_offset.setSuffix(" px")
        form.addRow("影オフセット", self._shadow_offset)

        layout.addLayout(form)

        desc = QLabel("線画を太らせた白縁＋影で\nステッカーのように浮き出させます。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "outline_size": self._outline_size.value(),
            "outline_color": self._outline_color_btn._color,
            "shadow": self._shadow_enabled.isChecked(),
            "shadow_offset": self._shadow_offset.value(),
        }


def execute_popout(layer_stack: LayerStack, source_layer: Layer,
                   params: dict) -> GroupLayer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    src_idx = _find_top_index(layer_stack, source_layer)

    group = GroupLayer(f"{source_layer.name} - ポップアウト", w, h)

    outline_size = params["outline_size"]
    outline_color = params["outline_color"]

    # 影レイヤー（最背面）
    if params["shadow"]:
        so = params["shadow_offset"]
        dilated = _dilate_alpha(src_img, outline_size + 2)
        shadow_img = _apply_color_overlay(dilated, QColor(0, 0, 0, 140))
        shifted = QImage(w, h, QImage.Format.Format_ARGB32)
        shifted.fill(Qt.GlobalColor.transparent)
        p = QPainter(shifted)
        p.drawImage(so, so, shadow_img)
        p.end()
        blurred = _blur_image(shifted, 3)
        shadow_layer = Layer("影", w, h)
        shadow_layer.image = blurred
        _copy_offset(source_layer, shadow_layer)
        group.children.append(shadow_layer)

    # 白縁レイヤー
    dilated = _dilate_alpha(src_img, outline_size)
    outline_img = _apply_color_overlay(dilated, outline_color)
    outline_layer = Layer("縁", w, h)
    outline_layer.image = outline_img
    _copy_offset(source_layer, outline_layer)
    group.children.insert(0, outline_layer)

    top = Layer(f"{source_layer.name} (元)", w, h)
    top.image = src_img.copy()
    _copy_offset(source_layer, top)
    group.children.insert(0, top)

    layer_stack.layers.insert(src_idx, group)
    layer_stack.active_path = [src_idx]
    source_layer.visible = False
    return group


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ランダムタイリング配置
# ═══════════════════════════════════════════════════════════════════════════════

class RandomTileDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ランダムタイリング配置")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._count = QSpinBox()
        self._count.setRange(2, 200)
        self._count.setValue(20)
        form.addRow("配置数", self._count)

        self._scale_min = QSpinBox()
        self._scale_min.setRange(10, 200)
        self._scale_min.setValue(80)
        self._scale_min.setSuffix(" %")
        form.addRow("スケール 最小", self._scale_min)

        self._scale_max = QSpinBox()
        self._scale_max.setRange(10, 300)
        self._scale_max.setValue(120)
        self._scale_max.setSuffix(" %")
        form.addRow("スケール 最大", self._scale_max)

        self._rot_max = QSpinBox()
        self._rot_max.setRange(0, 180)
        self._rot_max.setValue(15)
        self._rot_max.setSuffix(" °")
        form.addRow("回転 最大（±）", self._rot_max)

        self._overlap = QSpinBox()
        self._overlap.setRange(-100, 100)
        self._overlap.setValue(0)
        self._overlap.setSuffix(" %")
        self._overlap.setToolTip("正=配置間隔を広げる 負=重なりを増やす")
        form.addRow("重なり調整", self._overlap)

        layout.addLayout(form)

        self._merge = QCheckBox("1枚のレイヤーに統合する")
        self._merge.setChecked(True)
        layout.addWidget(self._merge)

        desc = QLabel("選択中のレイヤーをキャンバス全体にランダムな位置・\n"
                      "回転・スケールで複製配置します（壁紙パターン等）。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "count": self._count.value(),
            "scale_min": self._scale_min.value() / 100.0,
            "scale_max": self._scale_max.value() / 100.0,
            "rotate_max": self._rot_max.value(),
            "overlap": self._overlap.value() / 100.0,
            "merge": self._merge.isChecked(),
        }


def execute_random_tile(layer_stack: LayerStack, source_layer: Layer,
                        params: dict) -> Layer | GroupLayer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    sw, sh = src_img.width(), src_img.height()
    if sw == 0 or sh == 0:
        return None
    cw, ch = layer_stack.width, layer_stack.height
    src_idx = _find_top_index(layer_stack, source_layer)

    count = params["count"]
    scale_min = params["scale_min"]
    scale_max = params["scale_max"]
    rotate_max = params["rotate_max"]
    overlap = params["overlap"]
    do_merge = params["merge"]

    spacing_factor = max(0.2, 1.0 - overlap)
    cell = max(int(max(sw, sh) * spacing_factor), 1)
    cols = max(1, cw // cell + 2)
    rows = max(1, ch // cell + 2)

    placements = []
    for r in range(rows):
        for c in range(cols):
            base_x = c * cell - cell // 2
            base_y = r * cell - cell // 2
            jitter_x = random.randint(-cell // 3, cell // 3)
            jitter_y = random.randint(-cell // 3, cell // 3)
            placements.append((base_x + jitter_x, base_y + jitter_y))
    random.shuffle(placements)
    placements = placements[:count] if len(placements) > count else placements

    children = []
    for i, (px, py) in enumerate(placements):
        scale = random.uniform(scale_min, scale_max)
        angle = random.uniform(-rotate_max, rotate_max)
        tile = Layer(f"{source_layer.name} {i+1}", sw, sh)
        tile.image = _shift_image(src_img, 0, 0, angle, scale)
        tile.offset_x = px
        tile.offset_y = py
        children.append(tile)

    if do_merge:
        min_x = min(t.offset_x for t in children)
        min_y = min(t.offset_y for t in children)
        max_x = max(t.offset_x + t.image.width() for t in children)
        max_y = max(t.offset_y + t.image.height() for t in children)
        mw = max(max_x - min_x, 1)
        mh = max(max_y - min_y, 1)
        merged = QImage(mw, mh, QImage.Format.Format_ARGB32_Premultiplied)
        merged.fill(Qt.GlobalColor.transparent)
        p = QPainter(merged)
        for t in children:
            p.drawImage(t.offset_x - min_x, t.offset_y - min_y, t.image)
        p.end()
        result = Layer(f"{source_layer.name} - タイリング", mw, mh)
        result.image = merged.convertToFormat(QImage.Format.Format_ARGB32)
        result.offset_x = min_x
        result.offset_y = min_y
        layer_stack.layers.insert(src_idx, result)
        layer_stack.active_path = [src_idx]
        source_layer.visible = False
        return result

    group = GroupLayer(f"{source_layer.name} - タイリング", cw, ch)
    group.children = children
    layer_stack.layers.insert(src_idx, group)
    layer_stack.active_path = [src_idx]
    source_layer.visible = False
    return group


# ═══════════════════════════════════════════════════════════════════════════════
# 8. パスに沿った連続複製
# ═══════════════════════════════════════════════════════════════════════════════

class PathRepeatDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("パスに沿った連続複製")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._spacing = QSpinBox()
        self._spacing.setRange(5, 1000)
        self._spacing.setValue(60)
        self._spacing.setSuffix(" px")
        form.addRow("間隔", self._spacing)

        self._scale_min = QSpinBox()
        self._scale_min.setRange(10, 200)
        self._scale_min.setValue(100)
        self._scale_min.setSuffix(" %")
        form.addRow("スケール 最小", self._scale_min)

        self._scale_max = QSpinBox()
        self._scale_max.setRange(10, 300)
        self._scale_max.setValue(100)
        self._scale_max.setSuffix(" %")
        form.addRow("スケール 最大", self._scale_max)

        self._rot_max = QSpinBox()
        self._rot_max.setRange(0, 180)
        self._rot_max.setValue(0)
        self._rot_max.setSuffix(" °")
        form.addRow("回転 最大（±）", self._rot_max)

        self._follow_path = QCheckBox("進行方向に合わせて回転する")
        form.addRow(self._follow_path)

        layout.addLayout(form)

        self._merge = QCheckBox("1枚のレイヤーに統合する")
        self._merge.setChecked(True)
        layout.addWidget(self._merge)

        desc = QLabel("キャンバス上でパスを描くと、その軌跡に沿って\n"
                      "選択中のレイヤーを等間隔で複製配置します。\n"
                      "ダイアログでOK後、キャンバス上をドラッグして\n"
                      "パスを描いてください。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "spacing": self._spacing.value(),
            "scale_min": self._scale_min.value() / 100.0,
            "scale_max": self._scale_max.value() / 100.0,
            "rotate_max": self._rot_max.value(),
            "follow_path": self._follow_path.isChecked(),
            "merge": self._merge.isChecked(),
        }


def _resample_path(points: list[tuple[float, float]], spacing: float) -> list[tuple[float, float, float]]:
    """パス上の点列を一定間隔でリサンプリングする。(x, y, angle_deg) のリストを返す。"""
    if len(points) < 2:
        return [(points[0][0], points[0][1], 0.0)] if points else []

    seg_lengths = []
    total = 0.0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        seg_lengths.append(d)
        total += d

    if total <= 0:
        return [(points[0][0], points[0][1], 0.0)]

    result = []
    dist_walked = 0.0
    target = 0.0
    seg_idx = 0
    seg_pos = 0.0
    while target <= total:
        while seg_idx < len(seg_lengths) and seg_pos + seg_lengths[seg_idx] < target:
            seg_pos += seg_lengths[seg_idx]
            seg_idx += 1
        if seg_idx >= len(seg_lengths):
            break
        x0, y0 = points[seg_idx]
        x1, y1 = points[seg_idx + 1]
        seg_len = seg_lengths[seg_idx]
        t = (target - seg_pos) / seg_len if seg_len > 0 else 0.0
        x = x0 + (x1 - x0) * t
        y = y0 + (y1 - y0) * t
        import math
        angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
        result.append((x, y, angle))
        target += spacing
    return result


def execute_path_repeat(layer_stack: LayerStack, source_layer: Layer,
                        path_points: list[tuple[float, float]],
                        params: dict) -> Layer | GroupLayer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    sw, sh = src_img.width(), src_img.height()
    if sw == 0 or sh == 0 or len(path_points) < 2:
        return None
    src_idx = _find_top_index(layer_stack, source_layer)

    spacing = max(1, params["spacing"])  # 0以下は無限ループ防止
    scale_min = params["scale_min"]
    scale_max = params["scale_max"]
    rotate_max = params["rotate_max"]
    follow_path = params["follow_path"]
    do_merge = params["merge"]

    samples = _resample_path(path_points, spacing)
    if not samples:
        return None

    children = []
    for i, (x, y, path_angle) in enumerate(samples):
        scale = random.uniform(scale_min, scale_max)
        angle = (path_angle if follow_path else 0.0) + random.uniform(-rotate_max, rotate_max)
        tile = Layer(f"{source_layer.name} {i+1}", sw, sh)
        tile.image = _shift_image(src_img, 0, 0, angle, scale)
        tile.offset_x = int(x - sw / 2)
        tile.offset_y = int(y - sh / 2)
        children.append(tile)

    if do_merge:
        min_x = min(t.offset_x for t in children)
        min_y = min(t.offset_y for t in children)
        max_x = max(t.offset_x + t.image.width() for t in children)
        max_y = max(t.offset_y + t.image.height() for t in children)
        mw = max(max_x - min_x, 1)
        mh = max(max_y - min_y, 1)
        merged = QImage(mw, mh, QImage.Format.Format_ARGB32_Premultiplied)
        merged.fill(Qt.GlobalColor.transparent)
        p = QPainter(merged)
        for t in children:
            p.drawImage(t.offset_x - min_x, t.offset_y - min_y, t.image)
        p.end()
        result = Layer(f"{source_layer.name} - パス複製", mw, mh)
        result.image = merged.convertToFormat(QImage.Format.Format_ARGB32)
        result.offset_x = min_x
        result.offset_y = min_y
        layer_stack.layers.insert(src_idx, result)
        layer_stack.active_path = [src_idx]
        source_layer.visible = False
        return result

    group = GroupLayer(f"{source_layer.name} - パス複製", layer_stack.width, layer_stack.height)
    group.children = children
    layer_stack.layers.insert(src_idx, group)
    layer_stack.active_path = [src_idx]
    source_layer.visible = False
    return group


# ═══════════════════════════════════════════════════════════════════════════════
# 9. 紙質感グレインフィルター
# ═══════════════════════════════════════════════════════════════════════════════

class PaperGrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("紙質感グレイン")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._strength = QSpinBox()
        self._strength.setRange(1, 100)
        self._strength.setValue(25)
        self._strength.setSuffix(" %")
        form.addRow("強度", self._strength)

        self._scale = QSpinBox()
        self._scale.setRange(1, 10)
        self._scale.setValue(2)
        self._scale.setSuffix(" x")
        self._scale.setToolTip("粒子の粗さ（大きいほど粒が大きい）")
        form.addRow("粒の粗さ", self._scale)

        self._mode = QComboBox()
        self._mode.addItem("オーバーレイ（自然な紙質感）", "overlay")
        self._mode.addItem("乗算（陰影を強める）", "multiply")
        form.addRow("合成モード", self._mode)

        layout.addLayout(form)

        desc = QLabel("選択中のレイヤーにランダムなノイズ粒子を重ねて\n"
                      "紙のようなザラついた質感を加えます。\n"
                      "元のレイヤーはそのまま残り、コピーに適用されます。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "strength": self._strength.value() / 100.0,
            "scale": self._scale.value(),
            "mode": self._mode.currentData(),
        }


def execute_paper_grain(layer_stack: LayerStack, source_layer: Layer,
                        params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    strength = params["strength"]
    scale = params["scale"]
    mode = params["mode"]
    src_idx = _find_top_index(layer_stack, source_layer)

    small_w = max(1, w // scale)
    small_h = max(1, h // scale)
    noise = np.random.randint(0, 256, (small_h, small_w), dtype=np.uint8)
    noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    noise_rgb = np.stack([noise, noise, noise], axis=-1).astype(np.float32)

    arr = _qimage_to_array(src_img).astype(np.float32)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]

    if mode == "multiply":
        grain = noise_rgb / 255.0
        blended = rgb * (1.0 - strength + strength * grain)
    else:
        offset = (noise_rgb - 128.0) * strength
        blended = rgb + offset

    blended = np.clip(blended, 0, 255)
    out = arr.copy()
    out[:, :, :3] = blended
    out[:, :, 3] = alpha

    result_layer = Layer(f"{source_layer.name} - 紙質感", w, h)
    result_layer.image = _array_to_qimage(out.astype(np.uint8))
    _copy_offset(source_layer, result_layer)

    layer_stack.layers.insert(src_idx, result_layer)
    layer_stack.active_path = [src_idx]
    source_layer.visible = False
    return result_layer


# ═══════════════════════════════════════════════════════════════════════════════
# アクションパネル
# ═══════════════════════════════════════════════════════════════════════════════

class ActionPanel(QWidget):
    action_executed = pyqtSignal()
    structure_will_change = pyqtSignal()

    def __init__(self, layer_stack: LayerStack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        self.canvas = None  # main.py から注入される（パスピックモード用）

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("アクション")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title)

        desc = QLabel("選択中のレイヤーに対してワンクリックで適用。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        actions = [
            ("🎨 線画ずらし（色収差）", "色収差エフェクト", self._on_chroma_shift),
            ("✨ グロー / 発光", "暗背景＋発光エフェクト", self._on_glow),
            ("🔲 影付け", "ドロップシャドウ自動生成", self._on_drop_shadow),
            ("🎨 背景パターン生成", "ドット・ストライプ・グラデ等", self._on_bg_pattern),
            ("🖌️ 線画色変え", "線画の色を一括変換", self._on_line_color),
            ("⭐ ポップアウト", "ステッカー風に浮き出し", self._on_popout),
            ("🔁 ランダムタイリング配置", "壁紙パターン風に複製配置", self._on_random_tile),
            ("〰️ パスに沿った連続複製", "クリックしたパスに沿って複製配置", self._on_path_repeat),
            ("📜 紙質感グレイン", "ザラついた紙の質感を加える", self._on_paper_grain),
        ]
        for text, tip, slot in actions:
            btn = QPushButton(text)
            btn.setFixedHeight(34)
            btn.setToolTip(tip)
            btn.setStyleSheet("QPushButton { text-align: left; padding-left: 8px; }")
            btn.clicked.connect(slot)
            layout.addWidget(btn)

        layout.addStretch()

    def _require_layer(self, title: str) -> Layer | None:
        active = self.layer_stack.active
        if not active or active.is_group:
            QMessageBox.warning(self, title, "通常レイヤーを選択してください。")
            return None
        return active

    def _run(self, title: str, dialog_cls, execute_fn):
        active = self._require_layer(title)
        if not active:
            return
        dlg = dialog_cls(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.structure_will_change.emit()
        result = execute_fn(self.layer_stack, active, dlg.params())
        if result:
            self.action_executed.emit()

    def _on_chroma_shift(self):
        self._run("線画ずらし", ChromaShiftDialog, execute_chroma_shift)

    def _on_glow(self):
        self._run("グロー", GlowDialog, execute_glow)

    def _on_drop_shadow(self):
        self._run("影付け", DropShadowDialog, execute_drop_shadow)

    def _on_bg_pattern(self):
        active = self.layer_stack.active
        dlg = BgPatternDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.structure_will_change.emit()
        result = execute_bg_pattern(self.layer_stack, active, dlg.params())
        if result:
            self.action_executed.emit()

    def _on_line_color(self):
        self._run("線画色変え", LineColorDialog, execute_line_color)

    def _on_popout(self):
        self._run("ポップアウト", PopoutDialog, execute_popout)

    def _on_random_tile(self):
        self._run("ランダムタイリング配置", RandomTileDialog, execute_random_tile)

    def _on_paper_grain(self):
        self._run("紙質感グレイン", PaperGrainDialog, execute_paper_grain)

    def _on_path_repeat(self):
        active = self._require_layer("パスに沿った連続複製")
        if not active:
            return
        if self.canvas is None:
            QMessageBox.warning(self, "パスに沿った連続複製", "キャンバスに接続されていません。")
            return
        dlg = PathRepeatDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        params = dlg.params()

        def on_path_confirmed(points):
            canvas_points = [(p.x(), p.y()) for p in points]
            self.structure_will_change.emit()
            result = execute_path_repeat(self.layer_stack, active, canvas_points, params)
            if result:
                self.action_executed.emit()

        self.canvas.start_path_pick(on_path_confirmed)
