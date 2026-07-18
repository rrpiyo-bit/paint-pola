"""アクション機能 — ワンクリックで複雑なレイヤー操作を実行する。"""
from __future__ import annotations

import math
import random

import numpy as np
import cv2

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QDialog,
                              QLabel, QSpinBox, QCheckBox, QHBoxLayout,
                              QDialogButtonBox, QGroupBox, QFormLayout,
                              QComboBox, QColorDialog, QFrame, QMessageBox,
                              QScrollArea)
from PyQt6.QtGui import (QImage, QPainter, QColor, QTransform, QLinearGradient,
                          QRadialGradient, QBrush, QPen, QPainterPath)
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


def _render_tile(src: QImage, angle: float, scale: float) -> QImage:
    """回転・拡縮したタイル画像を、はみ出さない十分なサイズで描画して返す。
    _shift_image は元画像と同サイズで返すため、回転や拡大で四隅が
    クリップされてしまう。タイルは対角線長を基準にした正方形に描く。"""
    sw, sh = src.width(), src.height()
    side = max(1, int(math.hypot(sw, sh) * scale) + 2)
    out = QImage(side, side, QImage.Format.Format_ARGB32)
    out.fill(Qt.GlobalColor.transparent)
    t = QTransform()
    t.translate(side / 2, side / 2)
    t.rotate(angle)
    t.scale(scale, scale)
    t.translate(-sw / 2, -sh / 2)
    p = QPainter(out)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.setTransform(t)
    p.drawImage(0, 0, src)
    p.end()
    return out


def execute_random_tile(layer_stack: LayerStack, source_layer: Layer,
                        params: dict) -> Layer | GroupLayer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    if src_img.width() == 0 or src_img.height() == 0:
        return None

    # イラストはキャンバスサイズのレイヤーの一部に描かれていることが多い。
    # レイヤー画像全体をタイルとして扱うと、タイル1個＝ほぼキャンバスサイズと
    # 誤認して格子がほとんど作れず（指定個数に届かない）、配置位置も大半が
    # キャンバス外になる。まず不透明部分の外接矩形だけを切り出す。
    arr = _qimage_to_array(src_img)
    ys, xs = np.nonzero(arr[:, :, 3] > 10)
    if len(xs) == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    src_img = src_img.copy(x0, y0, int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1)
    sw, sh = src_img.width(), src_img.height()

    cw, ch = layer_stack.width, layer_stack.height
    src_idx = _find_top_index(layer_stack, source_layer)

    count = params["count"]
    scale_min = params["scale_min"]
    scale_max = params["scale_max"]
    rotate_max = params["rotate_max"]
    overlap = params["overlap"]
    do_merge = params["merge"]

    avg_scale = (scale_min + scale_max) / 2.0
    spacing_factor = max(0.2, 1.0 - overlap)
    cell = max(int(max(sw, sh) * avg_scale * spacing_factor), 1)
    cols = max(1, (cw + cell - 1) // cell)
    rows = max(1, (ch + cell - 1) // cell)

    # 配置位置は「タイルの中心」のキャンバス座標。格子＋ジッターで散らし、
    # 指定個数に足りない分はキャンバス内のランダム位置で補う（以前は格子が
    # 個数より少ないと黙って減っていた）。
    centers = []
    for r in range(rows):
        for c in range(cols):
            base_x = c * cell + cell // 2
            base_y = r * cell + cell // 2
            jitter = cell // 3
            centers.append((base_x + random.randint(-jitter, jitter),
                            base_y + random.randint(-jitter, jitter)))
    random.shuffle(centers)
    centers = centers[:count]
    while len(centers) < count:
        centers.append((random.randint(0, cw - 1), random.randint(0, ch - 1)))
    # 中心は必ずキャンバス内に収める（端で見切れるのはパターンとして自然だが、
    # 完全にキャンバス外へ出てしまう配置は「消えた」ように見えるため）
    centers = [(min(max(cx, 0), cw - 1), min(max(cy, 0), ch - 1))
               for cx, cy in centers]

    children = []
    for i, (cx, cy) in enumerate(centers):
        scale = random.uniform(scale_min, scale_max)
        angle = random.uniform(-rotate_max, rotate_max)
        img = _render_tile(src_img, angle, scale)
        tile = Layer(f"{source_layer.name} {i+1}", img.width(), img.height())
        tile.image = img
        tile.offset_x = cx - img.width() // 2
        tile.offset_y = cy - img.height() // 2
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
# 新効果 共通ユーティリティ
# ═══════════════════════════════════════════════════════════════════════════════

def _filled_silhouette(alpha: np.ndarray) -> np.ndarray:
    """不透明部分の穴埋め済みシルエット（0/255）を返す。"""
    opaque = (alpha > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(opaque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sil = np.zeros_like(opaque)
    if contours:
        cv2.drawContours(sil, contours, -1, 255, thickness=cv2.FILLED)
    return sil


def _coarse_noise(w: int, h: int, cell: int) -> np.ndarray:
    """0〜1 の滑らかなランダムノイズ（約 cell px のうねり）を返す。"""
    cell = max(1, cell)
    gw = max(2, w // cell)
    gh = max(2, h // cell)
    g = np.random.rand(gh, gw).astype(np.float32)
    return np.clip(cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def _shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """uint8 マスクを (dx, dy) 平行移動する（はみ出しは 0 埋め）。"""
    h, w = mask.shape
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, m, (w, h))


def _split_mask(mask: np.ndarray, k: int) -> list[np.ndarray]:
    """マスクをランダムな種点からのボロノイ分割で k 個の紙片に分ける。"""
    ys, xs = np.nonzero(mask)
    k = min(k, len(xs))
    if k <= 1:
        return [mask]
    idx = np.random.choice(len(xs), size=k, replace=False)
    sy, sx = ys[idx].astype(np.int64), xs[idx].astype(np.int64)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    gy, gx = np.mgrid[y0:y1, x0:x1]
    best_d = None
    best_i = np.zeros((y1 - y0, x1 - x0), dtype=np.int32)
    for i in range(k):
        d = (gy - sy[i]) ** 2 + (gx - sx[i]) ** 2
        if best_d is None:
            best_d = d
        else:
            closer = d < best_d
            best_i[closer] = i
            best_d = np.where(closer, d, best_d)
    sub = mask[y0:y1, x0:x1] > 0
    pieces = []
    for i in range(k):
        pm = np.zeros_like(mask)
        pm[y0:y1, x0:x1][(best_i == i) & sub] = 255
        if pm.any():
            pieces.append(pm)
    return pieces


def _insert_result_layer(layer_stack: LayerStack, source_layer: Layer,
                         result) -> None:
    """アクション結果をソースの位置に挿入し、ソースを隠して選択する。"""
    src_idx = _find_top_index(layer_stack, source_layer)
    layer_stack.layers.insert(src_idx, result)
    layer_stack.active_path = [src_idx]
    source_layer.visible = False


def _group_with_original(source_layer: Layer, suffix: str) -> tuple[GroupLayer, Layer]:
    """元レイヤーのコピーを最上段に持つグループを作って返す。"""
    w, h = source_layer.image.width(), source_layer.image.height()
    group = GroupLayer(f"{source_layer.name} - {suffix}", w, h)
    top = Layer(f"{source_layer.name} (元)", w, h)
    top.image = source_layer.image.copy()
    _copy_offset(source_layer, top)
    group.children.append(top)
    return group, top


# ═══════════════════════════════════════════════════════════════════════════════
# 10. ずれ縁取り
# ═══════════════════════════════════════════════════════════════════════════════

class OffsetBorderDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ずれ縁取り")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._color_btn = _color_button(QColor(255, 255, 255), self)
        form.addRow("縁の色", self._color_btn)

        self._size = QSpinBox()
        self._size.setRange(2, 40)
        self._size.setValue(8)
        self._size.setSuffix(" px")
        form.addRow("縁の太さ", self._size)

        self._shift = QSpinBox()
        self._shift.setRange(0, 60)
        self._shift.setValue(12)
        self._shift.setSuffix(" px")
        self._shift.setToolTip("縁マスクをランダムにずらす最大量")
        form.addRow("ずらし量（最大）", self._shift)

        self._gap = QSpinBox()
        self._gap.setRange(0, 90)
        self._gap.setValue(30)
        self._gap.setSuffix(" %")
        self._gap.setToolTip("縁をランダムに欠けさせる割合")
        form.addRow("欠け", self._gap)

        layout.addLayout(form)

        desc = QLabel("縁取りをわざとずらして「変なところに縁が付く」\n"
                      "偶然の面白さを再現します。欠けで途切れ感も出せます。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "color": self._color_btn._color,
            "size": self._size.value(),
            "shift": self._shift.value(),
            "gap": self._gap.value(),
        }


def execute_offset_border(layer_stack: LayerStack, source_layer: Layer,
                          params: dict) -> GroupLayer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None
    arr = _qimage_to_array(src_img)
    alpha = arr[:, :, 3]
    if not alpha.any():
        return None

    sil = _filled_silhouette(alpha)
    size = params["size"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size * 2 + 1, size * 2 + 1))
    dilated = cv2.dilate(sil, kernel)

    shift = params.get("shift", 0)
    dx = random.randint(-shift, shift) if shift else 0
    dy = random.randint(-shift, shift) if shift else 0
    shifted = _shift_mask(dilated, dx, dy)

    border_area = (shifted > 0) & (sil == 0)
    gap = params.get("gap", 0)
    if gap > 0:
        noise = _coarse_noise(w, h, max(8, size * 3))
        border_area &= noise > (gap / 100.0)

    bc: QColor = params["color"]
    border = np.zeros((h, w, 4), dtype=np.uint8)
    border[border_area] = [bc.blue(), bc.green(), bc.red(), bc.alpha()]

    group, _top = _group_with_original(source_layer, "ずれ縁取り")
    border_layer = Layer("ずれ縁", w, h)
    border_layer.image = _array_to_qimage(border)
    _copy_offset(source_layer, border_layer)
    group.children.append(border_layer)

    _insert_result_layer(layer_stack, source_layer, group)
    return group


# ═══════════════════════════════════════════════════════════════════════════════
# 11. リソ風版ずれ
# ═══════════════════════════════════════════════════════════════════════════════

class SilkscreenDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("リソ風版ずれ")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._color_btns = []
        for i, c in enumerate([QColor(242, 160, 177), QColor(245, 224, 75),
                               QColor(87, 201, 177)]):
            btn = _color_button(c, self)
            self._color_btns.append(btn)
            form.addRow(f"色版 {i + 1}", btn)

        self._shift = QSpinBox()
        self._shift.setRange(0, 100)
        self._shift.setValue(25)
        self._shift.setSuffix(" px")
        form.addRow("版ずれ量（最大）", self._shift)

        self._opacity = QSpinBox()
        self._opacity.setRange(10, 100)
        self._opacity.setValue(90)
        self._opacity.setSuffix(" %")
        form.addRow("色版の不透明度", self._opacity)

        layout.addLayout(form)

        desc = QLabel("線画のシルエットを色版にして、それぞれランダムに\n"
                      "ずらして重ねます。リソグラフ印刷の版ずれ風。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "colors": [b._color for b in self._color_btns],
            "shift": self._shift.value(),
            "opacity": self._opacity.value(),
        }


def execute_silkscreen(layer_stack: LayerStack, source_layer: Layer,
                       params: dict) -> GroupLayer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None
    arr = _qimage_to_array(src_img)
    alpha = arr[:, :, 3]
    if not alpha.any():
        return None

    sil = _filled_silhouette(alpha)
    shift = params.get("shift", 0)
    plate_alpha = int(params.get("opacity", 100) * 255 / 100)

    group, _top = _group_with_original(source_layer, "リソ風版ずれ")
    for i, color in enumerate(params["colors"]):
        dx = random.randint(-shift, shift) if shift else 0
        dy = random.randint(-shift, shift) if shift else 0
        mask = _shift_mask(sil, dx, dy)
        plate = np.zeros((h, w, 4), dtype=np.uint8)
        plate[mask > 0] = [color.blue(), color.green(), color.red(), plate_alpha]
        layer = Layer(f"色版{i + 1}", w, h)
        layer.image = _array_to_qimage(plate)
        _copy_offset(source_layer, layer)
        group.children.append(layer)

    _insert_result_layer(layer_stack, source_layer, group)
    return group


# ═══════════════════════════════════════════════════════════════════════════════
# 12. 切り絵コラージュ
# ═══════════════════════════════════════════════════════════════════════════════

class CollageDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("切り絵コラージュ")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._color_btns = []
        for i, c in enumerate([QColor(242, 160, 177), QColor(245, 224, 75),
                               QColor(87, 201, 177), QColor(150, 180, 255)]):
            btn = _color_button(c, self)
            self._color_btns.append(btn)
            form.addRow(f"色紙 {i + 1}", btn)

        self._coverage = QSpinBox()
        self._coverage.setRange(10, 100)
        self._coverage.setValue(70)
        self._coverage.setSuffix(" %")
        self._coverage.setToolTip("閉じた領域のうち色を塗る割合")
        form.addRow("塗る割合", self._coverage)

        self._expand = QSpinBox()
        self._expand.setRange(0, 30)
        self._expand.setValue(6)
        self._expand.setSuffix(" px")
        self._expand.setToolTip("色紙を線からはみ出させる量")
        form.addRow("はみ出し", self._expand)

        self._shift = QSpinBox()
        self._shift.setRange(0, 30)
        self._shift.setValue(6)
        self._shift.setSuffix(" px")
        form.addRow("ずらし量（最大）", self._shift)

        layout.addLayout(form)

        desc = QLabel("線画の閉じた領域をランダムに拾って色紙で塗り、\n"
                      "少しはみ出し・ずらして貼った切り絵風にします。\n"
                      "広い領域は自動で複数の紙片に分けて塗り分けます。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "colors": [b._color for b in self._color_btns],
            "coverage": self._coverage.value(),
            "expand": self._expand.value(),
            "shift": self._shift.value(),
        }


def execute_collage(layer_stack: LayerStack, source_layer: Layer,
                    params: dict) -> GroupLayer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None
    arr = _qimage_to_array(src_img)
    alpha = arr[:, :, 3]
    if not alpha.any():
        return None

    # 線（不透明部）で区切られた透明領域のうち、画像の外周に接していない
    # 「閉じた領域」だけを塗り対象にする
    free = (alpha <= 10).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(free, connectivity=4)
    edge_labels = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :])) \
        | set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))

    coverage = params.get("coverage", 70) / 100.0
    expand = params.get("expand", 0)
    shift = params.get("shift", 0)
    colors = params["colors"]
    expand_kernel = None
    if expand > 0:
        expand_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))

    candidates = []
    for lab in range(1, n_labels):
        if lab in edge_labels:
            continue
        mask = (labels == lab).astype(np.uint8) * 255
        if int(np.count_nonzero(mask)) < 30:  # ノイズ領域は無視
            continue
        candidates.append(mask)
    if not candidates:
        return None

    chosen_masks = [m for m in candidates if random.random() <= coverage]
    if not chosen_masks:  # 最低1領域は必ず塗る
        chosen_masks = [random.choice(candidates)]

    # 大きい領域は複数の紙片に分割する。実際の線画は内部がひと続きの
    # 領域になりがちで、そのままだと1色しか使われないため。
    piece_size = max(60, min(w, h) // 6)
    pieces: list[np.ndarray] = []
    for mask in chosen_masks:
        area = int(np.count_nonzero(mask))
        k = min(8, max(1, area // (piece_size * piece_size)))
        if k <= 1:
            pieces.append(mask)
        else:
            pieces.extend(_split_mask(mask, k))

    # 全色をまんべんなく使うため、シャッフルした色を順番に割り当てる
    palette = list(colors)
    random.shuffle(palette)
    fills = np.zeros((h, w, 4), dtype=np.uint8)
    for i, mask in enumerate(pieces):
        if expand_kernel is not None:
            mask = cv2.dilate(mask, expand_kernel)
        if shift:
            mask = _shift_mask(mask, random.randint(-shift, shift),
                               random.randint(-shift, shift))
        color = palette[i % len(palette)]
        fills[mask > 0] = [color.blue(), color.green(), color.red(), color.alpha()]

    group, _top = _group_with_original(source_layer, "切り絵コラージュ")
    fill_layer = Layer("色紙", w, h)
    fill_layer.image = _array_to_qimage(fills)
    _copy_offset(source_layer, fill_layer)
    group.children.append(fill_layer)

    _insert_result_layer(layer_stack, source_layer, group)
    return group


# ═══════════════════════════════════════════════════════════════════════════════
# 13. 線の揺らぎ
# ═══════════════════════════════════════════════════════════════════════════════

class WobbleDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("線の揺らぎ")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._strength = QSpinBox()
        self._strength.setRange(1, 40)
        self._strength.setValue(8)
        self._strength.setSuffix(" px")
        form.addRow("揺らぎの強さ", self._strength)

        self._wavelength = QSpinBox()
        self._wavelength.setRange(10, 300)
        self._wavelength.setValue(60)
        self._wavelength.setSuffix(" px")
        self._wavelength.setToolTip("小さいほど細かく波打つ")
        form.addRow("波の大きさ", self._wavelength)

        self._gap = QSpinBox()
        self._gap.setRange(0, 80)
        self._gap.setValue(0)
        self._gap.setSuffix(" %")
        self._gap.setToolTip("線をランダムに途切れさせる割合")
        form.addRow("破線化", self._gap)

        layout.addLayout(form)

        desc = QLabel("線をランダムに波打たせて「描き直したような」\n"
                      "別テイクを作ります。破線化で途切れも加えられます。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "strength": self._strength.value(),
            "wavelength": self._wavelength.value(),
            "gap": self._gap.value(),
        }


def execute_wobble(layer_stack: LayerStack, source_layer: Layer,
                   params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None
    arr = _qimage_to_array(src_img)
    if not arr[:, :, 3].any():
        return None

    strength = params["strength"]
    wavelength = max(10, params["wavelength"])
    nx = (_coarse_noise(w, h, wavelength) - 0.5) * 2.0 * strength
    ny = (_coarse_noise(w, h, wavelength) - 0.5) * 2.0 * strength
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    warped = cv2.remap(arr, xx + nx, yy + ny,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    gap = params.get("gap", 0)
    if gap > 0:
        keep = _coarse_noise(w, h, max(6, wavelength // 4)) > (gap / 100.0)
        warped[:, :, 3] = warped[:, :, 3] * keep

    result = Layer(f"{source_layer.name} - 揺らぎ", w, h)
    result.image = _array_to_qimage(warped)
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 14. スタンプ劣化
# ═══════════════════════════════════════════════════════════════════════════════

class StampDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("スタンプ劣化")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._strength = QSpinBox()
        self._strength.setRange(5, 95)
        self._strength.setValue(40)
        self._strength.setSuffix(" %")
        form.addRow("かすれ強度", self._strength)

        self._grain = QSpinBox()
        self._grain.setRange(1, 8)
        self._grain.setValue(3)
        self._grain.setSuffix(" x")
        self._grain.setToolTip("かすれの粒の粗さ")
        form.addRow("粒の粗さ", self._grain)

        self._blots = QCheckBox("インク溜まりを足す")
        self._blots.setChecked(True)
        form.addRow("", self._blots)

        layout.addLayout(form)

        desc = QLabel("線をランダムにかすれさせて、ゴム版画・はんこの\n"
                      "ような質感にします。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "strength": self._strength.value(),
            "grain": self._grain.value(),
            "blots": self._blots.isChecked(),
        }


def execute_stamp(layer_stack: LayerStack, source_layer: Layer,
                  params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None
    arr = _qimage_to_array(src_img)
    alpha = arr[:, :, 3]
    if not alpha.any():
        return None

    strength = params["strength"] / 100.0
    grain = params["grain"]
    # 細かい粒＋大きなムラの2段ノイズでかすれさせる
    fine = _coarse_noise(w, h, grain * 3)
    coarse = _coarse_noise(w, h, grain * 24)
    keep = (fine > strength * 0.9) & (coarse > strength * 0.5)
    out = arr.copy()
    out[:, :, 3] = alpha * keep

    if params.get("blots", False):
        line_ys, line_xs = np.nonzero(alpha > 127)
        if len(line_xs) > 0:
            opaque = alpha > 127
            rgb_mean = arr[opaque][:, :3].mean(axis=0).astype(np.uint8)
            blot_mask = np.zeros((h, w), dtype=np.uint8)
            n_blots = max(3, len(line_xs) // 4000)
            for _ in range(n_blots):
                i = random.randrange(len(line_xs))
                cv2.circle(blot_mask, (int(line_xs[i]), int(line_ys[i])),
                           random.randint(2, 6), 255, -1)
            blot_area = blot_mask > 0
            out[blot_area, 0] = rgb_mean[0]
            out[blot_area, 1] = rgb_mean[1]
            out[blot_area, 2] = rgb_mean[2]
            out[blot_area, 3] = 255

    result = Layer(f"{source_layer.name} - スタンプ", w, h)
    result.image = _array_to_qimage(out)
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 15. 万華鏡
# ═══════════════════════════════════════════════════════════════════════════════

class KaleidoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("万華鏡")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._segments = QSpinBox()
        self._segments.setRange(2, 12)
        self._segments.setValue(6)
        form.addRow("分割数", self._segments)

        self._mirror = QCheckBox("交互に鏡映する")
        self._mirror.setChecked(True)
        form.addRow("", self._mirror)

        layout.addLayout(form)

        desc = QLabel("絵柄を扇形に切り取り、中心の周りに回転コピーして\n"
                      "万華鏡のような模様を作ります。コピー同士は重なりません。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "segments": self._segments.value(),
            "mirror": self._mirror.isChecked(),
        }


def execute_kaleidoscope(layer_stack: LayerStack, source_layer: Layer,
                         params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    if src_img.width() == 0 or src_img.height() == 0:
        return None
    cw, ch = layer_stack.width, layer_stack.height
    segments = max(2, params["segments"])
    mirror = params.get("mirror", False)
    ox = getattr(source_layer, 'offset_x', 0)
    oy = getattr(source_layer, 'offset_y', 0)
    cx, cy = cw / 2.0, ch / 2.0
    seg_angle = 360.0 / segments

    # 各扇形には「扇形0にある絵柄」だけが複製される仕組みなので、
    # 絵柄の重心が扇形0の中央に来るようソースを前回転させて中身を確保する
    alpha = _qimage_to_array(src_img)[:, :, 3]
    ys, xs = np.nonzero(alpha > 0)
    base = 0.0
    if len(xs) > 0:
        mx, my = float(xs.mean()) + ox, float(ys.mean()) + oy
        theta_c = math.degrees(math.atan2(my - cy, mx - cx))
        base = seg_angle / 2.0 - theta_c

    buf = QImage(cw, ch, QImage.Format.Format_ARGB32_Premultiplied)
    buf.fill(Qt.GlobalColor.transparent)
    p = QPainter(buf)
    p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    radius = math.hypot(cw, ch)  # キャンバス全域を覆う十分な半径
    for i in range(segments):
        p.save()
        # 扇形 i にクリップして描くので、コピー同士は重ならない
        wedge = QPainterPath()
        wedge.moveTo(cx, cy)
        steps = max(2, int(seg_angle // 10) + 1)
        for s in range(steps + 1):
            ang = math.radians(seg_angle * (i + s / steps))
            wedge.lineTo(cx + radius * math.cos(ang),
                         cy + radius * math.sin(ang))
        wedge.closeSubpath()
        p.setClipPath(wedge)
        t = QTransform()
        t.translate(cx, cy)
        if mirror and i % 2 == 1:
            # 隣り合う扇形が鏡映で続くように、反転してから隣の境界へ回転
            t.rotate(seg_angle * (i + 1))
            t.scale(1, -1)
        else:
            t.rotate(seg_angle * i)
        t.rotate(base)
        t.translate(-cx, -cy)
        p.setTransform(t, combine=True)
        p.drawImage(ox, oy, src_img)
        p.restore()
    p.end()

    result = Layer(f"{source_layer.name} - 万華鏡", cw, ch)
    result.image = buf.convertToFormat(QImage.Format.Format_ARGB32)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 16. 等高線
# ═══════════════════════════════════════════════════════════════════════════════

class ContourDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("等高線")
        self.setMinimumWidth(300)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._count = QSpinBox()
        self._count.setRange(1, 10)
        self._count.setValue(4)
        form.addRow("本数", self._count)

        self._spacing = QSpinBox()
        self._spacing.setRange(4, 60)
        self._spacing.setValue(12)
        self._spacing.setSuffix(" px")
        form.addRow("間隔", self._spacing)

        self._color_btn = _color_button(QColor(255, 255, 255), self)
        form.addRow("線の色", self._color_btn)

        self._thickness = QSpinBox()
        self._thickness.setRange(1, 8)
        self._thickness.setValue(2)
        self._thickness.setSuffix(" px")
        form.addRow("線の太さ", self._thickness)

        self._fade = QCheckBox("外側ほど薄くする")
        self._fade.setChecked(True)
        form.addRow("", self._fade)

        layout.addLayout(form)

        desc = QLabel("シルエットの外側に輪郭線を何重にも生成します。\n"
                      "地形図の等高線のような模様になります。")
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
            "spacing": self._spacing.value(),
            "color": self._color_btn._color,
            "thickness": self._thickness.value(),
            "fade": self._fade.isChecked(),
        }


def execute_contour(layer_stack: LayerStack, source_layer: Layer,
                    params: dict) -> GroupLayer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None
    arr = _qimage_to_array(src_img)
    alpha = arr[:, :, 3]
    if not alpha.any():
        return None

    sil = _filled_silhouette(alpha)
    count = params["count"]
    spacing = params["spacing"]
    thickness = params["thickness"]
    fade = params.get("fade", True)
    color: QColor = params["color"]

    k_spacing = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (spacing * 2 + 1, spacing * 2 + 1))
    k_thick = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (thickness * 2 + 1, thickness * 2 + 1))

    out = np.zeros((h, w, 4), dtype=np.uint8)
    cur = sil.copy()
    for i in range(count):
        cur = cv2.dilate(cur, k_spacing)
        ring = (cur > 0) & (cv2.erode(cur, k_thick) == 0)
        a = int(255 * (count - i) / (count + 1)) if fade else color.alpha()
        out[ring] = [color.blue(), color.green(), color.red(), a]

    group, _top = _group_with_original(source_layer, "等高線")
    contour_layer = Layer("等高線", w, h)
    contour_layer.image = _array_to_qimage(out)
    _copy_offset(source_layer, contour_layer)
    group.children.append(contour_layer)

    _insert_result_layer(layer_stack, source_layer, group)
    return group


# ═══════════════════════════════════════════════════════════════════════════════
# 17. アクションガチャ
# ═══════════════════════════════════════════════════════════════════════════════

GACHA_PALETTES: list[tuple[str, list[str]]] = [
    ("パステルポップ", ["#f2a0b1", "#f5e04b", "#57c9b1", "#ffffff"]),
    ("レトロ印刷", ["#e63946", "#457b9d", "#f4a261", "#1d3557"]),
    ("モノ＋差し色", ["#222222", "#ffffff", "#ff3366", "#cccccc"]),
    ("ビタミン", ["#ff6b35", "#ffd23f", "#0ead69", "#3bceac"]),
    ("ゆめかわ", ["#ffb3d9", "#b3d9ff", "#ffffb3", "#e6ccff"]),
]

# ガチャで使う効果（背景パターン生成は要素が違うため除外）
_GACHA_POOL: list[tuple[str, str]] = [
    ("chroma", "線画ずらし"),
    ("glow", "グロー"),
    ("shadow", "影付け"),
    ("line_color", "線画色変え"),
    ("popout", "ポップアウト"),
    ("tile", "タイリング"),
    ("path", "パス複製"),
    ("grain", "紙質感"),
    ("offset_border", "ずれ縁取り"),
    ("silkscreen", "リソ風版ずれ"),
    ("collage", "切り絵"),
    ("wobble", "線の揺らぎ"),
    ("stamp", "スタンプ劣化"),
    ("kaleido", "万華鏡"),
    ("contour", "等高線"),
]


def _gacha_random_path(w: int, h: int) -> list[tuple[float, float]]:
    """ガチャ用: キャンバスを横切るゆるやかな波パスを作る。"""
    n = random.randint(8, 14)
    y0 = random.uniform(h * 0.2, h * 0.8)
    y1 = random.uniform(h * 0.2, h * 0.8)
    amp = random.uniform(h * 0.05, h * 0.25)
    freq = random.uniform(1.0, 3.0)
    pts = []
    for i in range(n):
        t = i / (n - 1)
        x = t * (w - 1)
        y = y0 + (y1 - y0) * t + math.sin(t * math.pi * freq) * amp
        pts.append((x, min(max(y, 0.0), h - 1.0)))
    if random.random() < 0.5:
        pts = [(y * w / h, x * h / w) for x, y in pts]  # 縦方向バリエーション
    return pts


def _gacha_random_params(key: str, colors: list[QColor]) -> dict:
    """効果ごとのランダムパラメータを共有パレットから生成する。"""
    ri = random.randint
    ru = random.uniform
    rb = lambda p=0.5: random.random() < p
    pick = lambda: random.choice(colors)
    light = max(colors, key=lambda c: c.lightness())
    dark = min(colors, key=lambda c: c.lightness())

    if key == "chroma":
        n = min(len(colors), ri(2, 3))
        plates = [{"color": QColor(c.red(), c.green(), c.blue(), 200),
                   "thickness": ri(-1, 2)} for c in random.sample(colors, n)]
        return {"shift_px": ri(8, 45), "layers": plates,
                "rotate": rb(0.4), "rotate_max": ri(1, 6),
                "scale": rb(0.4), "scale_max": ri(2, 8)}
    if key == "glow":
        bg = QColor(dark)
        return {"glow_color": light, "glow_size": ri(6, 24),
                "glow_strength": ri(50, 90),
                "bg_color": bg.darker(ri(150, 300)),
                "bg_opacity": ri(0, 60)}
    if key == "shadow":
        c = QColor(dark) if rb(0.5) else QColor(0, 0, 0)
        c.setAlpha(160)
        return {"color": c, "offset_x": ri(-25, 25), "offset_y": ri(-25, 25),
                "blur": ri(0, 10), "strength": ri(50, 90)}
    if key == "line_color":
        return {"color": pick()}
    if key == "popout":
        return {"outline_size": ri(3, 15),
                "outline_color": light if rb(0.7) else pick(),
                "shadow": rb(0.7), "shadow_offset": ri(2, 8)}
    if key == "tile":
        return {"count": ri(8, 40), "scale_min": ru(0.4, 0.8),
                "scale_max": ru(0.9, 1.6), "rotate_max": ri(0, 60),
                "overlap": ru(-0.4, 0.3), "merge": True}
    if key == "path":
        return {"spacing": ri(150, 450), "scale_min": ru(0.4, 0.7),
                "scale_max": ru(0.7, 1.0), "rotate_max": ri(0, 40),
                "follow_path": rb(0.5), "merge": True}
    if key == "grain":
        return {"strength": ru(0.15, 0.5), "scale": ri(1, 4),
                "mode": random.choice(["overlay", "multiply"])}
    if key == "offset_border":
        return {"color": light if rb(0.7) else pick(),
                "size": ri(4, 20), "shift": ri(5, 40), "gap": ri(0, 60)}
    if key == "silkscreen":
        n = min(len(colors), ri(2, 3))
        return {"colors": random.sample(colors, n),
                "shift": ri(10, 50), "opacity": ri(60, 100)}
    if key == "collage":
        return {"colors": colors, "coverage": ri(40, 90),
                "expand": ri(0, 10), "shift": ri(0, 12)}
    if key == "wobble":
        return {"strength": ri(3, 18), "wavelength": ri(30, 180),
                "gap": ri(0, 40)}
    if key == "stamp":
        return {"strength": ri(20, 60), "grain": ri(1, 5), "blots": rb(0.6)}
    if key == "kaleido":
        return {"segments": random.choice([3, 4, 5, 6, 8]), "mirror": rb(0.6)}
    if key == "contour":
        return {"count": ri(2, 6), "spacing": ri(8, 30),
                "color": pick(), "thickness": ri(1, 4), "fade": rb(0.7)}
    return {}


_GACHA_EXEC = {
    "chroma": execute_chroma_shift,
    "glow": execute_glow,
    "shadow": execute_drop_shadow,
    "line_color": execute_line_color,
    "popout": execute_popout,
    "tile": execute_random_tile,
    "grain": execute_paper_grain,
    "offset_border": execute_offset_border,
    "silkscreen": execute_silkscreen,
    "collage": execute_collage,
    "wobble": execute_wobble,
    "stamp": execute_stamp,
    "kaleido": execute_kaleidoscope,
    "contour": execute_contour,
}


def _flatten_gacha_result(result, w: int, h: int) -> QImage:
    """効果の結果（レイヤー or グループ）をキャンバスサイズ1枚に焼く。"""
    buf = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    buf.fill(Qt.GlobalColor.transparent)
    p = QPainter(buf)
    if result.is_group:
        p.drawImage(0, 0, result.composite())
    else:
        p.drawImage(getattr(result, 'offset_x', 0),
                    getattr(result, 'offset_y', 0), result.image)
    p.end()
    return buf.convertToFormat(QImage.Format.Format_ARGB32)


class GachaDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("アクションガチャ")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._count = QComboBox()
        self._count.addItem("おまかせ（2〜4個）", 0)
        for n in (2, 3, 4):
            self._count.addItem(f"{n}個", n)
        form.addRow("効果の数", self._count)

        self._palette = QComboBox()
        self._palette.addItem("おまかせ", "auto")
        for name, _cols in GACHA_PALETTES:
            self._palette.addItem(name, name)
        self._palette.addItem("完全ランダム色", "random")
        form.addRow("カラーパレット", self._palette)

        layout.addLayout(form)

        desc = QLabel("効果をランダムに選んでランダムな数値で連続適用します。\n"
                      "どんな結果になるかはお楽しみ。気に入らなければ\n"
                      "元に戻す（Ctrl+Z）してもう一回引けます。\n"
                      "使ったレシピは新レイヤーの名前に残ります。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = QDialogButtonBox()
        roll = buttons.addButton("🎲 引く！", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        roll.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "count": self._count.currentData(),
            "palette": self._palette.currentData(),
        }


def execute_gacha(layer_stack: LayerStack, source_layer: Layer,
                  params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    if src_img.width() == 0 or src_img.height() == 0:
        return None
    if not _qimage_to_array(src_img)[:, :, 3].any():
        return None  # 空レイヤーにはかけない
    w, h = layer_stack.width, layer_stack.height

    count = params.get("count", 0) or random.randint(2, 4)
    palette_key = params.get("palette", "auto")
    if palette_key == "random":
        palette_name = "ランダム色"
        colors = [QColor(random.randint(0, 255), random.randint(0, 255),
                         random.randint(0, 255)) for _ in range(4)]
    else:
        if palette_key == "auto":
            palette_name, hex_colors = random.choice(GACHA_PALETTES)
        else:
            palette_name, hex_colors = next(
                (n, c) for n, c in GACHA_PALETTES if n == palette_key)
        colors = [QColor(c) for c in hex_colors]

    chosen = random.sample(_GACHA_POOL, min(count, len(_GACHA_POOL)))

    # ソースをキャンバスサイズの作業レイヤーに正規化してから順に適用する
    work_img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    work_img.fill(Qt.GlobalColor.transparent)
    p = QPainter(work_img)
    p.drawImage(getattr(source_layer, 'offset_x', 0),
                getattr(source_layer, 'offset_y', 0), src_img)
    p.end()
    work = Layer("work", w, h)
    work.image = work_img.convertToFormat(QImage.Format.Format_ARGB32)

    temp_stack = LayerStack(w, h)
    applied: list[str] = []
    for key, label in chosen:
        temp_stack.layers = [work]
        temp_stack.active_path = [0]
        eff_params = _gacha_random_params(key, colors)
        try:
            if key == "path":
                result = execute_path_repeat(
                    temp_stack, work, _gacha_random_path(w, h), eff_params)
            else:
                result = _GACHA_EXEC[key](temp_stack, work, eff_params)
        except Exception:
            result = None  # 1効果の失敗でガチャ全体を止めない
        if result is None:
            continue
        next_work = Layer("work", w, h)
        next_work.image = _flatten_gacha_result(result, w, h)
        work = next_work
        applied.append(label)

    if not applied:
        return None

    final = Layer(f"{source_layer.name} - ガチャ({palette_name}: "
                  f"{'→'.join(applied)})", w, h)
    final.image = work.image
    _insert_result_layer(layer_stack, source_layer, final)
    return final


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

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        inner = QWidget()
        scroll.setWidget(inner)

        layout = QVBoxLayout(inner)
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
            ("🎰 アクションガチャ", "ランダムな効果をランダムな数値で一発適用", self._on_gacha),
            ("🎨 線画ずらし（色収差）", "色収差エフェクト", self._on_chroma_shift),
            ("✨ グロー / 発光", "暗背景＋発光エフェクト", self._on_glow),
            ("🔲 影付け", "ドロップシャドウ自動生成", self._on_drop_shadow),
            ("🎨 背景パターン生成", "ドット・ストライプ・グラデ等", self._on_bg_pattern),
            ("🖌️ 線画色変え", "線画の色を一括変換", self._on_line_color),
            ("⭐ ポップアウト", "ステッカー風に浮き出し", self._on_popout),
            ("🔁 ランダムタイリング配置", "壁紙パターン風に複製配置", self._on_random_tile),
            ("〰️ パスに沿った連続複製", "クリックしたパスに沿って複製配置", self._on_path_repeat),
            ("📜 紙質感グレイン", "ザラついた紙の質感を加える", self._on_paper_grain),
            ("🧩 ずれ縁取り", "縁取りをわざとずらす偶然アート", self._on_offset_border),
            ("🖨️ リソ風版ずれ", "色版をずらして重ねる印刷風", self._on_silkscreen),
            ("✂️ 切り絵コラージュ", "閉じた領域を色紙でランダムに塗る", self._on_collage),
            ("〽️ 線の揺らぎ", "線を波打たせて別テイクを作る", self._on_wobble),
            ("🪧 スタンプ劣化", "はんこ風にかすれさせる", self._on_stamp),
            ("❄️ 万華鏡", "回転コピーで万華鏡模様", self._on_kaleidoscope),
            ("🗺️ 等高線", "外側に輪郭線を何重にも生成", self._on_contour),
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

    def _on_offset_border(self):
        self._run("ずれ縁取り", OffsetBorderDialog, execute_offset_border)

    def _on_silkscreen(self):
        self._run("リソ風版ずれ", SilkscreenDialog, execute_silkscreen)

    def _on_collage(self):
        self._run("切り絵コラージュ", CollageDialog, execute_collage)

    def _on_wobble(self):
        self._run("線の揺らぎ", WobbleDialog, execute_wobble)

    def _on_stamp(self):
        self._run("スタンプ劣化", StampDialog, execute_stamp)

    def _on_kaleidoscope(self):
        self._run("万華鏡", KaleidoDialog, execute_kaleidoscope)

    def _on_contour(self):
        self._run("等高線", ContourDialog, execute_contour)

    def _on_gacha(self):
        self._run("アクションガチャ", GachaDialog, execute_gacha)

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
