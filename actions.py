"""アクション機能 — ワンクリックで複雑なレイヤー操作を実行する。"""
from __future__ import annotations

import math
import random
import traceback

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


def _bgr(color: QColor) -> tuple[int, int, int]:
    """QColor を _qimage_to_array のチャンネル順(B,G,R)に並べ替える。

    _qimage_to_array が返す配列は ARGB32 のバイト列そのままなので、
    index 0 が Blue・2 が Red。ここを (R,G,B) の順で書くと赤と青が
    入れ替わった色になるため、配列へ色を書き込むときは必ずこれを通す。
    """
    return (color.blue(), color.green(), color.red())


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


def _pad_image(image: QImage, margin: int) -> QImage:
    """四方に margin ぶんの透明な余白を足した画像を返す。
    膨張・ぼかし・ずらしは外側に広がるため、元画像と同じサイズのまま
    処理すると画像の端で切り落とされる。先に余白を確保してから処理し、
    レイヤーの offset を margin ぶん戻すことで見た目の位置を保つ。"""
    if margin <= 0:
        return image.copy()
    w, h = image.width(), image.height()
    out = QImage(w + margin * 2, h + margin * 2, QImage.Format.Format_ARGB32)
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.drawImage(margin, margin, image)
    p.end()
    return out


def _offset_layer(layer: Layer, margin: int) -> None:
    """_pad_image で広げた分だけレイヤー位置を戻す。"""
    layer.offset_x -= margin
    layer.offset_y -= margin


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
    # グループはキャンバスサイズで作る（元画像サイズだと子がバッファ外に落ちて消える）
    group = GroupLayer(f"{source_layer.name} - 線画ずらし",
                       layer_stack.width, layer_stack.height)

    # ずらし・太らせ・回転拡縮で外側に広がるぶんの余白を確保する
    max_thick = max((ld.get("thickness", 0) for ld in layer_defs), default=0)
    diag = int((w * w + h * h) ** 0.5)
    grow = int(diag * (abs(scale_max) / 100.0) / 2) if do_scale else 0
    if do_rotate:
        grow += (diag - min(w, h)) // 2
    cmargin = int(shift_px) + int(max_thick) + grow + 2

    for i, ld in enumerate(layer_defs):
        color: QColor = ld["color"]
        thickness: int = ld.get("thickness", 0)
        base = _adjust_thickness(_pad_image(src_img, cmargin), thickness)
        colored_img = _apply_color_overlay(base, color)
        dx = random.randint(-shift_px, shift_px)
        dy = random.randint(-shift_px, shift_px)
        angle = random.uniform(-rot_max, rot_max) if do_rotate else 0.0
        scale = 1.0 + random.uniform(-scale_max, scale_max) / 100.0 if do_scale else 1.0
        shifted = _shift_image(colored_img, dx, dy, angle, scale)
        layer = Layer(f"ずらし{i+1}", shifted.width(), shifted.height())
        layer.image = shifted
        layer.blend_mode = "screen"
        _copy_offset(source_layer, layer)
        _offset_layer(layer, cmargin)
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

    # グループはキャンバスサイズで作る（元画像サイズだと子がバッファ外に落ちて消える）
    group = GroupLayer(f"{source_layer.name} - グロー",
                       layer_stack.width, layer_stack.height)

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

    # 膨張＋ぼかしで外側に広がるぶんの余白を先に確保する
    gmargin = glow_size * 2 + 2
    colored = _apply_color_overlay(_pad_image(src_img, gmargin), glow_color)
    dilated = _dilate_alpha(colored, glow_size)
    blurred = _blur_image(dilated, glow_size)

    glow_layer = Layer("グロー", blurred.width(), blurred.height())
    glow_layer.image = blurred
    glow_layer.opacity = int(glow_strength * 255 / 100)
    glow_layer.blend_mode = "screen"
    _copy_offset(source_layer, glow_layer)
    _offset_layer(glow_layer, gmargin)
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

    # グループはキャンバスサイズで作る（元画像サイズだと子がバッファ外に落ちて消える）
    group = GroupLayer(f"{source_layer.name} - 影付き",
                       layer_stack.width, layer_stack.height)

    # 影レイヤー
    shadow_color = params["color"]
    ox, oy = params["offset_x"], params["offset_y"]
    blur_r = params["blur"]
    # ずらし量とぼかし半径のぶん、先に余白を確保する
    dmargin = max(abs(int(ox)), abs(int(oy))) + int(blur_r) * 2 + 2
    colored = _apply_color_overlay(_pad_image(src_img, dmargin), shadow_color)

    shifted = QImage(colored.width(), colored.height(), QImage.Format.Format_ARGB32)
    shifted.fill(Qt.GlobalColor.transparent)
    p = QPainter(shifted)
    p.drawImage(ox, oy, colored)
    p.end()

    if blur_r > 0:
        shifted = _blur_image(shifted, blur_r)

    shadow_layer = Layer("影", shifted.width(), shifted.height())
    shadow_layer.image = shifted
    shadow_layer.opacity = int(params["strength"] * 255 / 100)
    _copy_offset(source_layer, shadow_layer)
    _offset_layer(shadow_layer, dmargin)
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

    # グループはキャンバスサイズで作る（元画像サイズだと子がバッファ外に落ちて消える）
    group = GroupLayer(f"{source_layer.name} - ポップアウト",
                       layer_stack.width, layer_stack.height)

    outline_size = params["outline_size"]
    outline_color = params["outline_color"]

    # 影レイヤー（最背面）
    if params["shadow"]:
        so = params["shadow_offset"]
        # 膨張＋ずらし＋ぼかしのぶんの余白を先に確保する
        smargin = outline_size + 2 + abs(int(so)) + 6
        padded = _pad_image(src_img, smargin)
        dilated = _dilate_alpha(padded, outline_size + 2)
        shadow_img = _apply_color_overlay(dilated, QColor(0, 0, 0, 140))
        shifted = QImage(padded.width(), padded.height(), QImage.Format.Format_ARGB32)
        shifted.fill(Qt.GlobalColor.transparent)
        p = QPainter(shifted)
        p.drawImage(so, so, shadow_img)
        p.end()
        blurred = _blur_image(shifted, 3)
        shadow_layer = Layer("影", blurred.width(), blurred.height())
        shadow_layer.image = blurred
        _copy_offset(source_layer, shadow_layer)
        _offset_layer(shadow_layer, smargin)
        group.children.append(shadow_layer)

    # 白縁レイヤー
    omargin = outline_size + 2
    dilated = _dilate_alpha(_pad_image(src_img, omargin), outline_size)
    outline_img = _apply_color_overlay(dilated, outline_color)
    outline_layer = Layer("縁", outline_img.width(), outline_img.height())
    outline_layer.image = outline_img
    _copy_offset(source_layer, outline_layer)
    _offset_layer(outline_layer, omargin)
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


def _group_with_original(source_layer: Layer, suffix: str,
                         canvas_size: tuple[int, int]) -> tuple[GroupLayer, Layer]:
    """元レイヤーのコピーを最上段に持つグループを作って返す。
    グループは必ずキャンバスサイズで作る。GroupLayer.composite() は自身の
    サイズのバッファに子をオフセット付きで描くため、元レイヤーの画像サイズ
    （絵に密着した小さいサイズのことがある）で作ると子がバッファ外に落ちて
    絵が丸ごと消えてしまう。"""
    w, h = source_layer.image.width(), source_layer.image.height()
    group = GroupLayer(f"{source_layer.name} - {suffix}", canvas_size[0], canvas_size[1])
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

    size = params["size"]
    shift = params.get("shift", 0)
    # 膨張とずらしで外側に広がるぶんの余白を確保してから処理する
    margin = size + abs(int(shift)) + 2
    alpha = cv2.copyMakeBorder(alpha, margin, margin, margin, margin,
                               cv2.BORDER_CONSTANT, value=0)
    ph, pw = alpha.shape[:2]

    sil = _filled_silhouette(alpha)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size * 2 + 1, size * 2 + 1))
    dilated = cv2.dilate(sil, kernel)

    dx = random.randint(-shift, shift) if shift else 0
    dy = random.randint(-shift, shift) if shift else 0
    shifted = _shift_mask(dilated, dx, dy)

    border_area = (shifted > 0) & (sil == 0)
    gap = params.get("gap", 0)
    if gap > 0:
        noise = _coarse_noise(pw, ph, max(8, size * 3))
        border_area &= noise > (gap / 100.0)

    bc: QColor = params["color"]
    border = np.zeros((ph, pw, 4), dtype=np.uint8)
    border[border_area] = [bc.blue(), bc.green(), bc.red(), bc.alpha()]

    group, _top = _group_with_original(source_layer, "ずれ縁取り",
                                     (layer_stack.width, layer_stack.height))
    border_layer = Layer("ずれ縁", pw, ph)
    border_layer.image = _array_to_qimage(border)
    _copy_offset(source_layer, border_layer)
    _offset_layer(border_layer, margin)
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

    shift = params.get("shift", 0)
    plate_alpha = int(params.get("opacity", 100) * 255 / 100)
    # 版のずらし量ぶんの余白を確保してから処理する
    margin = abs(int(shift)) + 2
    alpha = cv2.copyMakeBorder(alpha, margin, margin, margin, margin,
                               cv2.BORDER_CONSTANT, value=0)
    ph, pw = alpha.shape[:2]
    sil = _filled_silhouette(alpha)

    group, _top = _group_with_original(source_layer, "リソ風版ずれ",
                                     (layer_stack.width, layer_stack.height))
    for i, color in enumerate(params["colors"]):
        dx = random.randint(-shift, shift) if shift else 0
        dy = random.randint(-shift, shift) if shift else 0
        mask = _shift_mask(sil, dx, dy)
        plate = np.zeros((ph, pw, 4), dtype=np.uint8)
        plate[mask > 0] = [color.blue(), color.green(), color.red(), plate_alpha]
        layer = Layer(f"色版{i + 1}", pw, ph)
        layer.image = _array_to_qimage(plate)
        _copy_offset(source_layer, layer)
        _offset_layer(layer, margin)
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

        self._close_gap = QSpinBox()
        self._close_gap.setRange(0, 20)
        self._close_gap.setValue(0)
        self._close_gap.setSuffix(" px")
        self._close_gap.setToolTip("線画が途切れていても、この px までなら閉じた領域とみなす")
        form.addRow("隙間を閉じる", self._close_gap)

        self._line_sensitivity = QSpinBox()
        self._line_sensitivity.setRange(0, 100)
        self._line_sensitivity.setValue(0)
        self._line_sensitivity.setSuffix(" %")
        self._line_sensitivity.setToolTip("上げるほど薄い線も境界として拾う")
        form.addRow("薄い線を拾う", self._line_sensitivity)

        layout.addLayout(form)

        desc = QLabel("線画の閉じた領域をランダムに拾って色紙で塗り、\n"
                      "少しはみ出し・ずらして貼った切り絵風にします。\n"
                      "広い領域は自動で複数の紙片に分けて塗り分けます。\n"
                      "線に隙間がある場合は「隙間を閉じる」を上げてください。")
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
            "close_gap": self._close_gap.value(),
            "line_sensitivity": self._line_sensitivity.value(),
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

    coverage = params.get("coverage", 70) / 100.0
    expand = params.get("expand", 0)
    shift = params.get("shift", 0)
    colors = params["colors"]
    close_gap = max(0, int(params.get("close_gap", 0)))
    # バケツツールと同じ換算。感度を上げるほどしきい値が下がり、
    # アンチエイリアスで薄くなった線も「線」として拾う。
    line_threshold = round(10 * (100 - max(0, min(100, int(params.get("line_sensitivity", 0))))) / 100)

    # 線（不透明部）で区切られた透明領域のうち、画像の外周に接していない
    # 「閉じた領域」だけを塗り対象にする
    free = (alpha <= line_threshold).astype(np.uint8)
    if close_gap > 0:
        # 線を太らせる＝空き領域を削る。数 px の途切れが塞がり、
        # 開いた領域も「閉じた領域」として検出できるようになる。
        ksize = close_gap * 2 + 1
        gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        search = cv2.erode(free, gap_kernel)
    else:
        gap_kernel = None
        search = free
    n_labels, labels = cv2.connectedComponents(search, connectivity=4)
    edge_labels = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :])) \
        | set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    expand_kernel = None
    if expand > 0:
        expand_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (expand * 2 + 1, expand * 2 + 1))

    candidates = []
    for lab in range(1, n_labels):
        if lab in edge_labels:
            continue
        mask = (labels == lab).astype(np.uint8) * 255
        if gap_kernel is not None:
            # 隙間閉じで削った分を戻し、本来の線の手前まで塗る。
            # 元の空き領域でクリップするので線を越えることはない。
            mask = cv2.dilate(mask, gap_kernel) * free
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
    # 紙片の膨張とずらしで外側に広がるぶんの余白を確保する。領域検出（labels）は
    # 元サイズで行う必要があるため、塗り込み段階でだけ広げる。
    margin = int(expand) + abs(int(shift)) + 2
    ph, pw = h + margin * 2, w + margin * 2
    fills = np.zeros((ph, pw, 4), dtype=np.uint8)
    for i, mask in enumerate(pieces):
        mask = cv2.copyMakeBorder(mask, margin, margin, margin, margin,
                                  cv2.BORDER_CONSTANT, value=0)
        if expand_kernel is not None:
            mask = cv2.dilate(mask, expand_kernel)
        if shift:
            mask = _shift_mask(mask, random.randint(-shift, shift),
                               random.randint(-shift, shift))
        color = palette[i % len(palette)]
        fills[mask > 0] = [color.blue(), color.green(), color.red(), color.alpha()]

    group, _top = _group_with_original(source_layer, "切り絵コラージュ",
                                     (layer_stack.width, layer_stack.height))
    fill_layer = Layer("色紙", pw, ph)
    fill_layer.image = _array_to_qimage(fills)
    _copy_offset(source_layer, fill_layer)
    _offset_layer(fill_layer, margin)
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
    # 揺らぎの最大変位ぶんの余白を確保してから歪ませる
    margin = int(strength) + 2
    arr = cv2.copyMakeBorder(arr, margin, margin, margin, margin,
                             cv2.BORDER_CONSTANT, value=0)
    ph, pw = arr.shape[:2]
    nx = (_coarse_noise(pw, ph, wavelength) - 0.5) * 2.0 * strength
    ny = (_coarse_noise(pw, ph, wavelength) - 0.5) * 2.0 * strength
    xx, yy = np.meshgrid(np.arange(pw, dtype=np.float32),
                         np.arange(ph, dtype=np.float32))
    warped = cv2.remap(arr, xx + nx, yy + ny,
                       interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    gap = params.get("gap", 0)
    if gap > 0:
        keep = _coarse_noise(pw, ph, max(6, wavelength // 4)) > (gap / 100.0)
        warped[:, :, 3] = warped[:, :, 3] * keep

    result = Layer(f"{source_layer.name} - 揺らぎ", pw, ph)
    result.image = _array_to_qimage(warped)
    _copy_offset(source_layer, result)
    _offset_layer(result, margin)
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

        self._blot_min = QSpinBox()
        self._blot_min.setRange(1, 40)
        self._blot_min.setValue(2)
        self._blot_min.setSuffix(" px")
        self._blot_min.setToolTip("インク溜まりの最小半径")
        form.addRow("インク溜まり 最小", self._blot_min)

        self._blot_max = QSpinBox()
        self._blot_max.setRange(1, 40)
        self._blot_max.setValue(6)
        self._blot_max.setSuffix(" px")
        self._blot_max.setToolTip("インク溜まりの最大半径")
        form.addRow("インク溜まり 最大", self._blot_max)

        # 最小 > 最大 の状態を作れないように連動させる
        self._blot_min.valueChanged.connect(
            lambda v: self._blot_max.setValue(v) if v > self._blot_max.value() else None)
        self._blot_max.valueChanged.connect(
            lambda v: self._blot_min.setValue(v) if v < self._blot_min.value() else None)

        self._auto_color = QCheckBox("線と同じ色を使う")
        self._auto_color.setChecked(True)
        self._auto_color.setToolTip("オフにすると下の色でインク溜まりを描きます")
        form.addRow("", self._auto_color)

        self._blot_color = _color_button(QColor(20, 20, 20), self)
        form.addRow("インク溜まりの色", self._blot_color)

        # 「線と同じ色」のときは色ボタンを触れないようにする
        self._auto_color.toggled.connect(
            lambda on: self._blot_color.setEnabled(not on))
        self._blot_color.setEnabled(False)

        # インク溜まりOFFなら関連項目をまとめて無効化する
        def _sync_blot_enabled(on: bool):
            self._blot_min.setEnabled(on)
            self._blot_max.setEnabled(on)
            self._auto_color.setEnabled(on)
            self._blot_color.setEnabled(on and not self._auto_color.isChecked())

        self._blots.toggled.connect(_sync_blot_enabled)

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
            "blot_min": self._blot_min.value(),
            "blot_max": self._blot_max.value(),
            # 「線と同じ色」なら色は None にして自動判定に任せる
            "blot_color": None if self._auto_color.isChecked()
                          else self._blot_color._color,
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
            # 半径の指定。最小 > 最大 で渡ってきても落ちないよう入れ替える
            r_min = max(1, int(params.get("blot_min", 2)))
            r_max = max(1, int(params.get("blot_max", 6)))
            if r_min > r_max:
                r_min, r_max = r_max, r_min
            blot_color = params.get("blot_color")
            if blot_color is None:
                # 色指定なし = 線の平均色になじませる（従来の挙動）
                rgb_blot = arr[opaque][:, :3].mean(axis=0).astype(np.uint8)
            else:
                rgb_blot = np.array(_bgr(blot_color), dtype=np.uint8)
            blot_mask = np.zeros((h, w), dtype=np.uint8)
            n_blots = max(3, len(line_xs) // 4000)
            for _ in range(n_blots):
                i = random.randrange(len(line_xs))
                cv2.circle(blot_mask, (int(line_xs[i]), int(line_ys[i])),
                           random.randint(r_min, r_max), 255, -1)
            blot_area = blot_mask > 0
            out[blot_area, 0] = rgb_blot[0]
            out[blot_area, 1] = rgb_blot[1]
            out[blot_area, 2] = rgb_blot[2]
            out[blot_area, 3] = 255

    result = Layer(f"{source_layer.name} - スタンプ", w, h)
    result.image = _array_to_qimage(out)
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 15. 等高線
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

    # 輪郭は外側へ count*spacing だけ広がるので、その分の余白を確保してから
    # 膨張させる。余白がないとレイヤー画像の端でリングが切り落とされる。
    margin = count * spacing + thickness + 1
    sil = cv2.copyMakeBorder(sil, margin, margin, margin, margin,
                             cv2.BORDER_CONSTANT, value=0)
    ow, oh = w + margin * 2, h + margin * 2

    out = np.zeros((oh, ow, 4), dtype=np.uint8)
    cur = sil.copy()
    for i in range(count):
        cur = cv2.dilate(cur, k_spacing)
        ring = (cur > 0) & (cv2.erode(cur, k_thick) == 0)
        a = int(255 * (count - i) / (count + 1)) if fade else color.alpha()
        out[ring] = [color.blue(), color.green(), color.red(), a]

    group, _top = _group_with_original(source_layer, "等高線",
                                     (layer_stack.width, layer_stack.height))
    contour_layer = Layer("等高線", ow, oh)
    contour_layer.image = _array_to_qimage(out)
    _copy_offset(source_layer, contour_layer)
    contour_layer.offset_x -= margin
    contour_layer.offset_y -= margin
    group.children.append(contour_layer)

    _insert_result_layer(layer_stack, source_layer, group)
    return group


# ═══════════════════════════════════════════════════════════════════════════════
# 新効果 共通ユーティリティ（網点・ディザ・ハッチング）
# ═══════════════════════════════════════════════════════════════════════════════

def _luma(bgr: np.ndarray) -> np.ndarray:
    """知覚的な明度(0-255)を求める。

    渡ってくるのは _qimage_to_array の配列（ARGB32 のバイト列そのまま）なので
    チャンネル順は B,G,R。係数を RGB の順で掛けると赤と青の重みが逆になり、
    赤い面と青い面の明暗が入れ替わってしまうので注意。
    """
    return (0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2])


def _halftone_plane(value: np.ndarray, pitch: int, angle_deg: float,
                    supersample: int = 2) -> np.ndarray:
    """1版ぶんの網点を描く。value は 0-255 の濃度（大きいほど点が大きい）。

    セル中心を angle_deg だけ回転した格子に置き、セル内の平均濃度に比例した
    半径の円を描く。版ごとに角度を変えるとモアレが出にくくなり、実際の
    カラー印刷らしい点々になる。戻り値は 0-255 のカバレッジ。
    """
    h, w = value.shape
    ss = max(1, supersample)          # アンチエイリアス用の拡大率
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # セル中心の1ピクセルだけを読むと、細い線画では線がセル中心を外れた
    # 瞬間に濃度0とみなされ、網点が1つも打たれず真っ白になる。
    # セル幅で平均化した濃度マップから読むことで、セル内のどこかに線が
    # あれば必ず点が立つようにする。
    box = max(1, pitch)
    sampled = cv2.blur(value, (box, box))

    # 回転しても画像全体を覆えるよう、格子の走査範囲を対角長ぶん広げる
    diag = int(math.hypot(w, h)) + pitch * 2
    canvas = np.zeros((h * ss, w * ss), dtype=np.uint8)
    # 濃度 100% で点の面積がセルの面積とちょうど等しくなる半径。
    # πr² = pitch² より r = pitch/√π。セルの対角(pitch/2·√2)まで伸ばすと
    # 中間調で既に隣の点とつながってしまい、全部真っ黒に潰れる。
    max_r = pitch * ss / math.sqrt(math.pi)

    for gy in range(-diag // pitch, diag // pitch + 1):
        for gx in range(-diag // pitch, diag // pitch + 1):
            # 格子座標 → 画像座標（回転を適用）
            ux, uy = gx * pitch, gy * pitch
            cx = ux * cos_t - uy * sin_t + w * 0.5
            cy = ux * sin_t + uy * cos_t + h * 0.5
            if not (-pitch <= cx < w + pitch and -pitch <= cy < h + pitch):
                continue
            # セル中心付近の平均濃度を読む
            sx = min(w - 1, max(0, int(cx)))
            sy = min(h - 1, max(0, int(cy)))
            v = float(sampled[sy, sx]) / 255.0
            if v <= 0.001:
                continue
            # 面積が濃度に比例するよう半径は sqrt を取る
            r = max_r * math.sqrt(min(1.0, v))
            if r < 0.3:
                continue
            cv2.circle(canvas, (int(round(cx * ss)), int(round(cy * ss))),
                       int(round(r)), 255, -1, lineType=cv2.LINE_AA)

    if ss > 1:
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    return canvas


_BAYER4 = np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5],
], dtype=np.float32) / 16.0


def _quantize_to_palette(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """各ピクセルをパレット中の最近傍色に置き換える。"""
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 1, 3)
    dist = np.sum((flat - palette.reshape(1, -1, 3)) ** 2, axis=2)
    idx = np.argmin(dist, axis=1)
    return palette[idx].reshape(h, w, 3)


def _hatch_lines(shape: tuple[int, int], angle_deg: float,
                 spacing: int, thickness: int) -> np.ndarray:
    """指定角度の等間隔な平行線マスク(0/255)を作る。"""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    length = int(math.hypot(w, h)) + spacing * 2
    cx, cy = w * 0.5, h * 0.5
    # 線に垂直な方向へ spacing ずつずらしながら引く
    nx, ny = -dy, dx
    for i in range(-length // spacing, length // spacing + 1):
        ox, oy = nx * i * spacing, ny * i * spacing
        p1 = (int(cx + ox - dx * length), int(cy + oy - dy * length))
        p2 = (int(cx + ox + dx * length), int(cy + oy + dy * length))
        cv2.line(mask, p1, p2, 255, max(1, thickness), lineType=cv2.LINE_AA)
    return mask


# ═══════════════════════════════════════════════════════════════════════════════
# 16. カラーハーフトーン（網点）
# ═══════════════════════════════════════════════════════════════════════════════

class HalftoneDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("カラーハーフトーン")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._pitch = QSpinBox()
        self._pitch.setRange(3, 40)
        self._pitch.setValue(10)
        self._pitch.setSuffix(" px")
        self._pitch.setToolTip("網点の間隔。小さいほど細かい網点になる")
        form.addRow("網点の間隔", self._pitch)

        self._mode = QComboBox()
        self._mode.addItem("カラー（RGB3版・版ごとに角度を変える）", "rgb")
        self._mode.addItem("モノクロ（1版）", "mono")
        form.addRow("種類", self._mode)

        self._bg = QComboBox()
        # 既定は透過。イラストに掛けたときに背景が白く埋まると、
        # 下のレイヤーが隠れて使いにくいため。
        self._bg.addItem("背景は透明のまま（イラストだけに効果）", "transparent")
        self._bg.addItem("白地に印刷", "white")
        form.addRow("背景", self._bg)

        self._smooth = QCheckBox("アンチエイリアスをかける")
        self._smooth.setChecked(True)
        form.addRow("", self._smooth)

        layout.addLayout(form)

        desc = QLabel("絵を印刷物のような網点（点々）に変換します。\n"
                      "カラーは元画像の RGB 成分を3版に分解し、版ごとに\n"
                      "角度を変えて重ねます（CMYK 分版ではありません）。\n"
                      "元のレイヤーはそのまま残ります。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "pitch": self._pitch.value(),
            "mode": self._mode.currentData(),
            "background": self._bg.currentData(),
            "smooth": self._smooth.isChecked(),
        }


def execute_halftone(layer_stack: LayerStack, source_layer: Layer,
                     params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    pitch = max(3, int(params.get("pitch", 10)))
    mode = params.get("mode", "rgb")
    background = params.get("background", "white")
    ss = 2 if params.get("smooth", True) else 1

    arr = _qimage_to_array(src_img).astype(np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    # 透明部分は「紙の白」として扱う。そうしないと背景まで真っ黒な網点になる。
    cover = (alpha / 255.0)[:, :, None]
    on_white = rgb * cover + 255.0 * (1.0 - cover)

    out = np.zeros((h, w, 4), dtype=np.float32)
    if mode == "mono":
        # 明度が低いほど点を大きくする（インクが乗る量）
        density = 255.0 - _luma(on_white)
        dots = _halftone_plane(density, pitch, 45.0, ss).astype(np.float32)
        out[:, :, :3] = 0.0
        out[:, :, 3] = dots
    else:
        # 元画像の R/G/B 成分をそのまま3版に分解する（CMYK 分版ではない）。
        # 各版は「その色の光がどれだけ強いか」を点の大きさで表し、
        # 3版を加法混色で重ねると元の色に戻る。赤い面は赤い点が大きく、
        # 緑青の点が小さくなる、という見え方になる。
        # 版の角度は実際の印刷の慣用値に近い 15/75/0 度を使い、モアレを避ける。
        angles = (15.0, 75.0, 0.0)
        planes = []
        for ch, ang in zip(range(3), angles):
            density = on_white[:, :, ch]
            planes.append(_halftone_plane(density, pitch, ang, ss).astype(np.float32) / 255.0)
        if background == "white":
            # 紙に刷る場合は減法混色。各版の点は「その色の光を吸うインク」なので、
            # 濃度は 255-チャンネル値（暗いほど大きな点）で描き直す。
            inks = []
            for ch, ang in zip(range(3), angles):
                density = 255.0 - on_white[:, :, ch]
                inks.append(_halftone_plane(density, pitch, ang, ss).astype(np.float32) / 255.0)
            # R版のインクは R 以外を吸う…ではなく、R成分そのものを落とす
            out[:, :, 0] = 255.0 * (1.0 - inks[0])
            out[:, :, 1] = 255.0 * (1.0 - inks[1])
            out[:, :, 2] = 255.0 * (1.0 - inks[2])
            out[:, :, 3] = 255.0
        else:
            out[:, :, 0] = planes[0] * 255.0
            out[:, :, 1] = planes[1] * 255.0
            out[:, :, 2] = planes[2] * 255.0
            # どれか1版でも点が乗っていれば不透明にする
            ink = np.maximum.reduce(planes)
            # 各版の濃度は「その色の光の強さ」なので、絵の無い所（＝白紙）は
            # 全版が最大になり、背景が真っ白なベタで埋まってしまう。
            # 元が透明だった場所には点を打たず、透過を保つ。
            ink = ink * np.clip(alpha / 255.0, 0.0, 1.0)
            out[:, :, 3] = ink * 255.0

    if background == "white" and mode == "mono":
        # 紙を敷く。網点が乗っていないところは白のまま残る。
        # （カラーは上の分岐で紙ごと描き切っている）
        a = (out[:, :, 3:4] / 255.0)
        out[:, :, :3] = out[:, :, :3] * a + 255.0 * (1.0 - a)
        out[:, :, 3] = 255.0

    result = Layer(f"{source_layer.name} - 網点", w, h)
    result.image = _array_to_qimage(np.clip(out, 0, 255).astype(np.uint8))
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 17. ディザ / レトロ減色
# ═══════════════════════════════════════════════════════════════════════════════

class DitherDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ディザ / レトロ減色")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._method = QComboBox()
        self._method.addItem("Bayer 4x4（規則的な網目・ゲーム機風）", "bayer")
        self._method.addItem("誤差拡散（きめ細かい・粒状）", "diffusion")
        form.addRow("ディザ方式", self._method)

        self._levels = QSpinBox()
        self._levels.setRange(2, 16)
        self._levels.setValue(4)
        self._levels.setToolTip("各色チャンネルの階調数。少ないほどレトロになる")
        form.addRow("階調数", self._levels)

        self._pixel = QSpinBox()
        self._pixel.setRange(1, 16)
        self._pixel.setValue(4)
        self._pixel.setSuffix(" x")
        self._pixel.setToolTip("ドットの粗さ。1 で等倍、大きいほど粗いドット絵になる")
        form.addRow("ドットの粗さ", self._pixel)

        layout.addLayout(form)

        desc = QLabel("色数を落としてディザをかけ、レトロゲームの画面や\n"
                      "初期パソコンの絵のような見た目にします。\n"
                      "元のレイヤーはそのまま残ります。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "method": self._method.currentData(),
            "levels": self._levels.value(),
            "pixel": self._pixel.value(),
        }


def execute_dither(layer_stack: LayerStack, source_layer: Layer,
                   params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    method = params.get("method", "bayer")
    levels = max(2, int(params.get("levels", 4)))
    pixel = max(1, int(params.get("pixel", 4)))
    palette = params.get("palette")     # ガチャから色を渡された場合に使う

    arr = _qimage_to_array(src_img).astype(np.float32)
    rgb, alpha = arr[:, :, :3].copy(), arr[:, :, 3].copy()

    # 先に縮小してから処理し、最後に最近傍で戻すと本物のドット絵になる
    sw, sh = max(1, w // pixel), max(1, h // pixel)
    if pixel > 1:
        rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (sw, sh), interpolation=cv2.INTER_AREA)

    pal_arr = None
    if palette:
        pal_arr = np.array([[c.red(), c.green(), c.blue()] for c in palette],
                           dtype=np.float32)

    if method == "diffusion":
        # Floyd-Steinberg。行ごとに誤差を配るので Python ループが要る。
        # ドット粗さで縮小済みなので、実用的なサイズに収まる。
        buf = rgb.copy()
        step = 255.0 / (levels - 1)
        for y in range(sh):
            for x in range(sw):
                old = buf[y, x].copy()
                if pal_arr is not None:
                    d = np.sum((pal_arr - old) ** 2, axis=1)
                    new = pal_arr[int(np.argmin(d))]
                else:
                    new = np.round(old / step) * step
                buf[y, x] = new
                err = old - new
                if x + 1 < sw:
                    buf[y, x + 1] += err * (7 / 16)
                if y + 1 < sh:
                    if x > 0:
                        buf[y + 1, x - 1] += err * (3 / 16)
                    buf[y + 1, x] += err * (5 / 16)
                    if x + 1 < sw:
                        buf[y + 1, x + 1] += err * (1 / 16)
        quant = np.clip(buf, 0, 255)
    else:
        # Bayer: しきい値マップを足してから量子化する
        tile = np.tile(_BAYER4, (sh // 4 + 1, sw // 4 + 1))[:sh, :sw]
        step = 255.0 / (levels - 1)
        biased = rgb + (tile[:, :, None] - 0.5) * step
        if pal_arr is not None:
            quant = _quantize_to_palette(np.clip(biased, 0, 255), pal_arr)
        else:
            quant = np.clip(np.round(biased / step) * step, 0, 255)

    # アルファもディザして、半透明の縁がドット絵らしく硬いエッジになるようにする
    tile_a = np.tile(_BAYER4, (sh // 4 + 1, sw // 4 + 1))[:sh, :sw]
    alpha_q = np.where(alpha > tile_a * 255.0, 255.0, 0.0)

    out = np.dstack([quant, alpha_q]).astype(np.uint8)
    if pixel > 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)

    result = Layer(f"{source_layer.name} - ディザ", w, h)
    result.image = _array_to_qimage(out)
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 18. クロスハッチング（ペン画風陰影）
# ═══════════════════════════════════════════════════════════════════════════════

class CrosshatchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("クロスハッチング")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._spacing = QSpinBox()
        self._spacing.setRange(3, 40)
        self._spacing.setValue(8)
        self._spacing.setSuffix(" px")
        self._spacing.setToolTip("斜線の間隔。小さいほど密なペン画になる")
        form.addRow("線の間隔", self._spacing)

        self._thickness = QSpinBox()
        self._thickness.setRange(1, 6)
        self._thickness.setValue(1)
        self._thickness.setSuffix(" px")
        form.addRow("線の太さ", self._thickness)

        self._layers_n = QSpinBox()
        self._layers_n.setRange(1, 4)
        self._layers_n.setValue(3)
        self._layers_n.setToolTip("暗い部分に何段まで線を重ねるか")
        form.addRow("重ねる段数", self._layers_n)

        self._color_btn = _color_button(QColor(20, 20, 30), self)
        form.addRow("線の色", self._color_btn)

        layout.addLayout(form)

        desc = QLabel("明るさに応じて斜線の密度を変え、ペン画や銅版画の\n"
                      "ような手描きの陰影に置き換えます。\n"
                      "元のレイヤーはそのまま残ります。")
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
            "thickness": self._thickness.value(),
            "layers": self._layers_n.value(),
            "color": self._color_btn._color,
        }


def execute_crosshatch(layer_stack: LayerStack, source_layer: Layer,
                       params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    spacing = max(3, int(params.get("spacing", 8)))
    thickness = max(1, int(params.get("thickness", 1)))
    n_layers = max(1, min(4, int(params.get("layers", 3))))
    color: QColor = params.get("color") or QColor(20, 20, 30)

    arr = _qimage_to_array(src_img).astype(np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    cover = (alpha / 255.0)[:, :, None]
    on_white = rgb * cover + 255.0 * (1.0 - cover)
    darkness = (255.0 - _luma(on_white)) / 255.0     # 0(明) 〜 1(暗)

    # 段ごとに角度を変え、暗いところほど多くの段が乗るようにする。
    # これで「暗い部分ほど線が密」というハッチング本来の効果になる。
    angles = (45.0, -45.0, 0.0, 90.0)
    ink = np.zeros((h, w), dtype=np.float32)
    for i in range(n_layers):
        # i 段目は darkness がこのしきい値を超えた領域にだけ乗る。
        # しきい値を 0〜1 いっぱいに散らさないと、中間調で全段が一斉に乗って
        # 濃淡が潰れ、どこも同じ密度の網に見えてしまう。
        lo = i / n_layers
        lines = _hatch_lines((h, w), angles[i], spacing, thickness).astype(np.float32) / 255.0
        # しきい値付近でいきなり現れないよう、その段の幅ぶんで滑らかに立ち上げる
        weight = np.clip((darkness - lo) * n_layers, 0.0, 1.0)
        ink = np.maximum(ink, lines * weight)

    out = np.zeros((h, w, 4), dtype=np.float32)
    out[:, :, 0], out[:, :, 1], out[:, :, 2] = _bgr(color)
    # 元が透明だったところには描かない
    out[:, :, 3] = ink * 255.0 * (alpha / 255.0)

    result = Layer(f"{source_layer.name} - ハッチング", w, h)
    result.image = _array_to_qimage(np.clip(out, 0, 255).astype(np.uint8))
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 19. VHS / 走査線ノイズ
# ═══════════════════════════════════════════════════════════════════════════════

class VhsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VHS / 走査線ノイズ")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._jitter = QSpinBox()
        self._jitter.setRange(0, 60)
        self._jitter.setValue(12)
        self._jitter.setSuffix(" px")
        self._jitter.setToolTip("行ごとの横ずれの最大量")
        form.addRow("行ずれ", self._jitter)

        self._bands = QSpinBox()
        self._bands.setRange(0, 12)
        self._bands.setValue(4)
        self._bands.setToolTip("大きく乱れる帯の本数（テープの傷）")
        form.addRow("ノイズ帯", self._bands)

        self._scanline = QSpinBox()
        self._scanline.setRange(0, 100)
        self._scanline.setValue(35)
        self._scanline.setSuffix(" %")
        form.addRow("走査線の濃さ", self._scanline)

        self._scan_pitch = QSpinBox()
        self._scan_pitch.setRange(2, 16)
        self._scan_pitch.setValue(3)
        self._scan_pitch.setSuffix(" px")
        self._scan_pitch.setToolTip("走査線の間隔。大きいほど粗い画面になる")
        form.addRow("走査線の間隔", self._scan_pitch)

        self._noise = QSpinBox()
        self._noise.setRange(0, 100)
        self._noise.setValue(20)
        self._noise.setSuffix(" %")
        form.addRow("ざらつき", self._noise)

        layout.addLayout(form)

        desc = QLabel("行ずれ・ノイズ帯・走査線を重ねて、古いビデオテープを\n"
                      "再生したような映像の乱れを作ります。\n"
                      "元のレイヤーはそのまま残ります。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "jitter": self._jitter.value(),
            "bands": self._bands.value(),
            "scanline": self._scanline.value() / 100.0,
            "scan_pitch": self._scan_pitch.value(),
            "noise": self._noise.value() / 100.0,
        }


def execute_vhs(layer_stack: LayerStack, source_layer: Layer,
                params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    jitter = max(0, int(params.get("jitter", 12)))
    bands = max(0, int(params.get("bands", 4)))
    scan_amt = float(params.get("scanline", 0.35))
    scan_pitch = max(2, int(params.get("scan_pitch", 3)))
    noise_amt = float(params.get("noise", 0.2))

    arr = _qimage_to_array(src_img).astype(np.float32)

    # 行ごとの横ずれ量。ゆるやかな波＋乱数で、テープの揺れらしくする。
    rows = np.arange(h, dtype=np.float32)
    wave = np.sin(rows / max(3.0, h / 40.0)) * (jitter * 0.4)
    shift = wave + np.random.uniform(-jitter * 0.6, jitter * 0.6, size=h)

    # ノイズ帯: 数行まとめて大きくずらす（テープの傷）
    for _ in range(bands):
        by = random.randint(0, max(0, h - 1))
        bh = random.randint(2, max(3, h // 60))
        amp = random.uniform(jitter * 1.5, jitter * 3.0 + 4.0)
        shift[by:by + bh] += amp * random.choice([-1.0, 1.0])

    shifted = arr.copy()
    for y in range(h):
        s = int(round(shift[y]))
        if s:
            shifted[y] = np.roll(arr[y], s, axis=0)

    # 走査線: 一定間隔の行を暗くする
    if scan_amt > 0:
        line = np.ones(h, dtype=np.float32)
        line[::scan_pitch] = 1.0 - scan_amt
        shifted[:, :, :3] *= line[:, None, None]

    # ざらつき
    if noise_amt > 0:
        n = np.random.normal(0.0, 32.0 * noise_amt, (h, w, 1)).astype(np.float32)
        shifted[:, :, :3] += n

    out = np.clip(shifted, 0, 255).astype(np.uint8)
    result = Layer(f"{source_layer.name} - VHS", w, h)
    result.image = _array_to_qimage(out)
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 20. CRT / LED 画面
# ═══════════════════════════════════════════════════════════════════════════════

class CrtDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CRT / LED 画面")
        self.setMinimumWidth(320)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._kind = QComboBox()
        self._kind.addItem("CRT（縦ストライプ＋走査線）", "crt")
        self._kind.addItem("LED（格子状のドット）", "led")
        form.addRow("種類", self._kind)

        self._cell = QSpinBox()
        self._cell.setRange(2, 16)
        self._cell.setValue(6)
        self._cell.setSuffix(" px")
        self._cell.setToolTip("1画素のセルサイズ。小さすぎると縮小表示で\n"
                              "ただ暗くなっただけに見えるので 4px 以上を推奨")
        form.addRow("セルサイズ", self._cell)

        self._bloom = QSpinBox()
        self._bloom.setRange(0, 100)
        self._bloom.setValue(45)
        self._bloom.setSuffix(" %")
        self._bloom.setToolTip("明るい部分のにじみ光。画面が光っている感じになる")
        form.addRow("ブルーム", self._bloom)

        self._boost = QSpinBox()
        self._boost.setRange(100, 300)
        self._boost.setValue(160)
        self._boost.setSuffix(" %")
        self._boost.setToolTip("マスクを掛けると必ず暗くなるので、その補正")
        form.addRow("明るさ補正", self._boost)

        self._sat = QSpinBox()
        self._sat.setRange(50, 250)
        self._sat.setValue(120)
        self._sat.setSuffix(" %")
        form.addRow("彩度補正", self._sat)

        layout.addLayout(form)

        desc = QLabel("RGB のサブピクセルと走査線を重ねて、ブラウン管や\n"
                      "LED パネルに映した画面のように見せます。\n"
                      "セルサイズが小さいと効果が見えにくいので注意。\n"
                      "元のレイヤーはそのまま残ります。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "kind": self._kind.currentData(),
            "cell": self._cell.value(),
            "bloom": self._bloom.value() / 100.0,
            "boost": self._boost.value() / 100.0,
            "saturation": self._sat.value() / 100.0,
        }


def execute_crt(layer_stack: LayerStack, source_layer: Layer,
                params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    kind = params.get("kind", "crt")
    cell = max(2, int(params.get("cell", 6)))
    bloom_amt = float(params.get("bloom", 0.45))
    boost = float(params.get("boost", 1.6))
    sat = float(params.get("saturation", 1.2))

    arr = _qimage_to_array(src_img).astype(np.float32)
    rgb, alpha = arr[:, :, :3].copy(), arr[:, :, 3].copy()

    # セル単位に量子化して「画素」を作る。
    # w//cell で割り切れないと縮小→拡大でセル格子がずれるので、
    # 一度セルの整数倍サイズに合わせてから戻す。
    sw, sh = max(1, w // cell), max(1, h // cell)
    pw, ph = sw * cell, sh * cell
    orig_alpha = alpha.copy()

    def _cellify(a: np.ndarray) -> np.ndarray:
        cropped = a[:ph, :pw]
        small = cv2.resize(cropped, (sw, sh), interpolation=cv2.INTER_AREA)
        big = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_NEAREST)
        if (pw, ph) != (w, h):
            # 端数はセル化せず元のまま残す（絵が欠けるより自然）
            out = a.copy()
            out[:ph, :pw] = big
            return out
        return big

    rgb = _cellify(rgb)
    alpha = _cellify(alpha)
    # セル化で alpha が元の絵の外へにじむと、何もなかった場所に
    # 四角い画素が浮いてしまう。元が完全に透明だった所は透明に戻す。
    alpha = np.where(orig_alpha > 0, alpha, 0.0)

    # 彩度・明るさの補正（マスクで暗くなるぶんを先に持ち上げる）
    if abs(sat - 1.0) > 0.01:
        gray = _luma(rgb)[:, :, None]
        rgb = gray + (rgb - gray) * sat
    rgb = rgb * boost

    # ブルーム: 明るい部分だけを取り出してぼかし、加算で戻す
    if bloom_amt > 0:
        bright = np.clip(rgb - 140.0, 0, None)
        k = max(3, (cell * 2) | 1)
        glow = cv2.GaussianBlur(bright, (k, k), 0)
        rgb = rgb + glow * bloom_amt

    # サブピクセルマスク
    xs = np.arange(w)
    ys = np.arange(h)
    mask = np.ones((h, w, 3), dtype=np.float32)
    third = max(1, cell // 3)
    # 横方向を3等分し、R/G/B のストライプにする（アパーチャグリル）
    band = (xs % cell) // third
    band = np.clip(band, 0, 2)
    for ch in range(3):
        mask[:, :, ch] = np.where(band == ch, 1.0, 0.25)[None, :]

    if kind == "led":
        # LED はセルの外周を暗くして、粒が独立して見えるようにする
        gap_x = ((xs % cell) >= cell - max(1, cell // 4))[None, :, None]
        gap_y = ((ys % cell) >= cell - max(1, cell // 4))[:, None, None]
        mask = mask * np.where(gap_x | gap_y, 0.15, 1.0)
    else:
        # CRT は走査線（横方向の暗い線）
        scan = np.where((ys % cell) >= cell - max(1, cell // 3), 0.45, 1.0)
        mask = mask * scan[:, None, None]

    rgb = rgb * mask

    out = np.dstack([np.clip(rgb, 0, 255), np.clip(alpha, 0, 255)]).astype(np.uint8)
    result = Layer(f"{source_layer.name} - {'LED' if kind == 'led' else 'CRT'}", w, h)
    result.image = _array_to_qimage(out)
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 21. 「○○風」加工 共通ユーティリティ
# ═══════════════════════════════════════════════════════════════════════════════

def _posterize_regions(luma: np.ndarray, levels: int) -> np.ndarray:
    """明度を levels 段に量子化し、各画素の段番号(0..levels-1)を返す。

    ウォーホル風・浮世絵風のように「明るさで版を分ける」効果の土台。
    段番号のまま返すのは、呼び出し側で段ごとに好きな色を割り当てるため。
    """
    lv = max(2, int(levels))
    idx = np.floor(luma / 256.0 * lv).astype(np.int32)
    return np.clip(idx, 0, lv - 1)


def _edge_mask(rgb: np.ndarray, alpha: np.ndarray, thickness: int) -> np.ndarray:
    """絵の輪郭(0/255)を取り出す。

    アメコミ風・ステンドグラス風・設計図風で「黒い線」を引くのに使う。
    暗い画素そのものではなく色の変わり目を拾うので、ベタ塗りの絵でも
    領域の境目に線が入る。
    """
    gray = np.clip(_luma(rgb), 0, 255).astype(np.uint8)
    # 透明部は白(=何もない)扱いにしないと、絵の外周が全部エッジになる
    gray = np.where(alpha > 10, gray, 255).astype(np.uint8)
    edge = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_16S, ksize=3)
    edge = np.absolute(edge).astype(np.uint8)
    edge = np.where(edge > 12, 255, 0).astype(np.uint8)
    # 元から暗い線（線画）もエッジに含める
    edge = np.maximum(edge, np.where((gray < 90) & (alpha > 10), 255, 0)
                      .astype(np.uint8))
    t = max(1, int(thickness))
    if t > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (t, t))
        edge = cv2.dilate(edge, k)
    return edge


def _paper_texture(shape: tuple[int, int], strength: float,
                   scale: int = 3) -> np.ndarray:
    """紙・和紙の風合い用のざらつき(-1..1)を作る。"""
    h, w = shape
    if strength <= 0:
        return np.zeros((h, w), dtype=np.float32)
    small = np.random.rand(max(1, h // max(1, scale)),
                           max(1, w // max(1, scale))).astype(np.float32)
    tex = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    tex = cv2.GaussianBlur(tex, (0, 0), 0.8)
    return (tex - 0.5) * 2.0 * strength


def _to_qcolors(colors) -> list[QColor]:
    """パレット指定を QColor のリストに揃える。hex 文字列も受け付ける。"""
    out = []
    for c in colors or []:
        out.append(c if isinstance(c, QColor) else QColor(c))
    return [c for c in out if c.isValid()]


# ═══════════════════════════════════════════════════════════════════════════════
# 22. アンディ・ウォーホル風（ポップアート4分割）
# ═══════════════════════════════════════════════════════════════════════════════

# 各コマの配色は「背景・肌・髪・線」の4役に割り当てる。ウォーホルの
# マリリンのように、コマごとに全く違う配色にするのが狙い。
_WARHOL_SETS: list[list[str]] = [
    ["#f03c78", "#f7d94c", "#3ec1c9", "#1a1a1a"],
    ["#2fbfa0", "#f2f2f2", "#f0d43a", "#1a1a1a"],
    ["#f5851f", "#f2b8c6", "#f7e04b", "#1a1a1a"],
    ["#2f6fd0", "#f7e04b", "#e8462f", "#1a1a1a"],
    ["#8e44ad", "#ffe9d6", "#f03c78", "#1a1a1a"],
    ["#111111", "#f7e04b", "#e8462f", "#f2f2f2"],
]


class WarholDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("アンディ・ウォーホル風")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._grid = QComboBox()
        self._grid.addItem("2 × 2（4枚）", (2, 2))
        self._grid.addItem("3 × 3（9枚）", (3, 3))
        self._grid.addItem("1 × 3（横3枚）", (3, 1))
        self._grid.addItem("3 × 1（縦3枚）", (1, 3))
        form.addRow("分割", self._grid)

        self._levels = QSpinBox()
        self._levels.setRange(2, 6)
        self._levels.setValue(4)
        self._levels.setToolTip("明るさを何段に分けるか。少ないほどベタ塗りに近づく")
        form.addRow("階調数", self._levels)

        self._gap = QSpinBox()
        self._gap.setRange(0, 40)
        self._gap.setValue(0)
        self._gap.setSuffix(" px")
        self._gap.setToolTip("コマとコマの間の余白")
        form.addRow("コマの間隔", self._gap)

        self._outline = QSpinBox()
        self._outline.setRange(0, 8)
        self._outline.setValue(2)
        self._outline.setSuffix(" px")
        self._outline.setToolTip("0 にすると線を描かず、色面だけになる")
        form.addRow("輪郭線の太さ", self._outline)

        self._random = QCheckBox("配色をランダムに選ぶ")
        self._random.setChecked(True)
        self._random.setToolTip("オフにすると毎回同じ順番の配色になる")
        form.addRow("", self._random)

        layout.addLayout(form)

        desc = QLabel("元の絵を縮小してタイル状に並べ、コマごとに違う配色で\n"
                      "ベタ塗りします。キャンバスの大きさは変わりません。\n"
                      "元のレイヤーはそのまま残ります。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        cols, rows = self._grid.currentData()
        return {
            "cols": cols,
            "rows": rows,
            "levels": self._levels.value(),
            "gap": self._gap.value(),
            "outline": self._outline.value(),
            "random_sets": self._random.isChecked(),
        }


def _warhol_cell(rgb: np.ndarray, alpha: np.ndarray, levels: int,
                 palette: list[QColor], outline: int) -> np.ndarray:
    """1コマぶんをポスタライズして配色する。戻り値は BGRA。"""
    h, w = alpha.shape
    lum = np.clip(_luma(rgb), 0, 255)
    idx = _posterize_regions(lum, levels)

    out = np.zeros((h, w, 4), dtype=np.float32)
    # 明るさの段だけで色を決めると、髪と肌のように「明るさは近いが色が違う」
    # ものが同じ色に潰れてしまう。色相でも区別してから割り当てる。
    hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    # 色相を機械的に等分すると、肌と金髪のように近い色が同じ帯に入って
    # 分離できない。絵の中に実際に出てくる色相を k-means で拾って分ける。
    valid = (alpha > 10) & (sat > 50)
    hue_band = np.full(alpha.shape, -1, dtype=np.int32)
    if int(np.count_nonzero(valid)) >= 8:
        hv = hsv[:, :, 0][valid].astype(np.float32)
        # 色相は環状なので、単位円に写してから分類する
        ang = hv / 180.0 * 2.0 * np.pi
        feat = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
        k = min(3, len(np.unique(hv)))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _compact, lab, _centers = cv2.kmeans(
            feat, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        hue_band[valid] = lab.reshape(-1)

    # 最後の色は輪郭線用に取っておくので、面の塗りには使わない
    fill_n = max(1, len(palette) - 1)
    for lv in range(levels):
        for hb in (-1, 0, 1, 2):
            m = (idx == lv) & (hue_band == hb)
            if not m.any():
                continue
            # 明るさで大まかな位置を決め、色相の違いでその隣にずらす。
            # 塗り用の色数の中で必ず循環させ、輪郭線の色には食い込ませない。
            base = int(round(lv / max(1, levels - 1) * (fill_n - 1)))
            pi = (base + (0 if hb < 0 else hb + 1)) % fill_n
            b, g, r = _bgr(palette[pi])
            out[m, 0], out[m, 1], out[m, 2] = b, g, r

    # 背景（元が透明な所）はパレットの1色目で塗りつぶす。ウォーホルの
    # シルクスクリーンは必ず地の色があるので、透明のままだと締まらない。
    bg_b, bg_g, bg_r = _bgr(palette[0])
    empty = alpha <= 10
    out[empty, 0], out[empty, 1], out[empty, 2] = bg_b, bg_g, bg_r
    out[:, :, 3] = 255.0

    if outline > 0:
        edge = _edge_mask(rgb, alpha, outline)
        ob, og, orr = _bgr(palette[-1])
        m = edge > 0
        out[m, 0], out[m, 1], out[m, 2] = ob, og, orr

    return out


def execute_warhol(layer_stack: LayerStack, source_layer: Layer,
                   params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    cols = max(1, int(params.get("cols", 2)))
    rows = max(1, int(params.get("rows", 2)))
    levels = max(2, int(params.get("levels", 4)))
    gap = max(0, int(params.get("gap", 0)))
    outline = max(0, int(params.get("outline", 2)))
    sets = params.get("palettes")
    if sets:
        # ガチャなどから明示的に配色を渡された場合
        cell_palettes = [_to_qcolors(s) for s in sets]
        cell_palettes = [p for p in cell_palettes if p]
    else:
        chosen = list(_WARHOL_SETS)
        if params.get("random_sets", True):
            random.shuffle(chosen)
        cell_palettes = [_to_qcolors(s) for s in chosen]
    if not cell_palettes:
        return None

    # 1コマの大きさ。間隔ぶんを差し引いてから割る
    cw = max(1, (w - gap * (cols + 1)) // cols)
    ch = max(1, (h - gap * (rows + 1)) // rows)

    arr = _qimage_to_array(src_img).astype(np.float32)
    # 縮小してからポスタライズすると、細部が潰れてベタ塗りらしくなる
    small = cv2.resize(arr, (cw, ch), interpolation=cv2.INTER_AREA)
    s_rgb, s_alpha = small[:, :, :3], small[:, :, 3]

    out = np.zeros((h, w, 4), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            pal = cell_palettes[i % len(cell_palettes)]
            cell = _warhol_cell(s_rgb, s_alpha, levels, pal, outline)
            x0 = gap + c * (cw + gap)
            y0 = gap + r * (ch + gap)
            # 端数でキャンバスからはみ出す場合は切り詰める
            x1, y1 = min(w, x0 + cw), min(h, y0 + ch)
            if x1 <= x0 or y1 <= y0:
                continue
            out[y0:y1, x0:x1] = cell[:y1 - y0, :x1 - x0].astype(np.uint8)

    result = Layer(f"{source_layer.name} - ウォーホル風", w, h)
    result.image = _array_to_qimage(out)
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 23. リキテンスタイン風（アメコミ）
# ═══════════════════════════════════════════════════════════════════════════════

class LichtensteinDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("リキテンスタイン風（アメコミ）")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._pitch = QSpinBox()
        self._pitch.setRange(3, 24)
        self._pitch.setValue(7)
        self._pitch.setSuffix(" px")
        self._pitch.setToolTip("網点の間隔。細かいほど写真寄り、粗いほど印刷物らしい")
        form.addRow("網点の間隔", self._pitch)

        self._outline = QSpinBox()
        self._outline.setRange(1, 10)
        self._outline.setValue(3)
        self._outline.setSuffix(" px")
        form.addRow("輪郭線の太さ", self._outline)

        self._levels = QSpinBox()
        self._levels.setRange(2, 5)
        self._levels.setValue(3)
        self._levels.setToolTip("ベタ塗りの色数。アメコミは少ない色数で刷られる")
        form.addRow("ベタの色数", self._levels)

        self._dot_color = _color_button(QColor(220, 50, 60), self)
        self._dot_color.setToolTip("網点そのものの色。赤系にすると原作らしい")
        form.addRow("網点の色", self._dot_color)

        self._line_color = _color_button(QColor(20, 20, 20), self)
        form.addRow("輪郭線の色", self._line_color)

        self._bg = QCheckBox("背景を白で埋める")
        self._bg.setChecked(True)
        form.addRow("", self._bg)

        layout.addLayout(form)

        desc = QLabel("太い黒の輪郭線・少ない色数のベタ塗り・網点の3つを重ねて、\n"
                      "アメコミの1コマのように仕上げます。\n"
                      "元のレイヤーはそのまま残ります。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "pitch": self._pitch.value(),
            "outline": self._outline.value(),
            "levels": self._levels.value(),
            "dot_color": self._dot_color._color,
            "line_color": self._line_color._color,
            "white_bg": self._bg.isChecked(),
        }


def execute_lichtenstein(layer_stack: LayerStack, source_layer: Layer,
                         params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    pitch = max(3, int(params.get("pitch", 7)))
    outline = max(1, int(params.get("outline", 3)))
    levels = max(2, int(params.get("levels", 3)))
    dot_color = params.get("dot_color") or QColor(220, 50, 60)
    line_color = params.get("line_color") or QColor(20, 20, 20)
    white_bg = params.get("white_bg", True)

    arr = _qimage_to_array(src_img).astype(np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    if not (alpha > 10).any():
        return None

    # 平滑化してからポスタライズすると、階調がベタ面としてまとまる
    smooth = cv2.bilateralFilter(rgb.astype(np.uint8), 9, 75, 75).astype(np.float32)
    lum = np.clip(_luma(smooth), 0, 255)
    idx = _posterize_regions(lum, levels)
    step = 255.0 / max(1, levels - 1)
    flat = (idx.astype(np.float32) * step)

    out = np.zeros((h, w, 4), dtype=np.float32)
    if white_bg:
        out[:, :, :3] = 255.0
        out[:, :, 3] = 255.0
    else:
        out[:, :, 3] = np.where(alpha > 10, 255.0, 0.0)

    # ベタ塗り: 段ごとの明るさをそのままグレーとして敷く
    inside = alpha > 10
    for ch in range(3):
        out[:, :, ch] = np.where(inside, flat, out[:, :, ch])

    # ベタ塗りは「白・網点・黒」の3役に割り切る。アメコミの製版では
    # 中間調そのものを刷らず、網点の粗密で表現する。
    top = levels - 1
    is_dark = inside & (idx == 0)              # 最暗部＝ベタ
    is_light = inside & (idx == top)           # 最明部＝紙の白
    is_mid = inside & ~is_dark & ~is_light     # 中間＝網点で表現

    for ch in range(3):
        out[:, :, ch] = np.where(is_light, 255.0, out[:, :, ch])
        out[:, :, ch] = np.where(is_dark, 30.0, out[:, :, ch])
    if not white_bg:
        out[:, :, 3] = np.where(inside, 255.0, out[:, :, 3])

    # 網点。中間調の中での相対的な暗さを濃度にすると、面の中で粗密が出る。
    # 濃度を上限 70% に抑えないと点が繋がってベタに潰れる。
    if is_mid.any():
        mid_lum = lum[is_mid]
        lo, hi = float(mid_lum.min()), float(mid_lum.max())
        rng = max(1.0, hi - lo)
        density = np.zeros((h, w), dtype=np.float32)
        density[is_mid] = (hi - mid_lum) / rng * 0.7 * 255.0
        dots = _halftone_plane(density, pitch, 45.0, 2).astype(np.float32) / 255.0
        db, dg, dr = _bgr(dot_color)
        # 中間調の地は白のままにして、点だけを色で打つ
        for ch in range(3):
            out[:, :, ch] = np.where(is_mid, 255.0, out[:, :, ch])
        m = is_mid & (dots > 0.4)
        out[m, 0], out[m, 1], out[m, 2] = db, dg, dr
        out[m, 3] = 255.0

    # 輪郭線は最後に乗せて、網点の上からでも必ず見えるようにする
    edge = _edge_mask(rgb, alpha, outline)
    lb, lg, lr = _bgr(line_color)
    m = edge > 0
    out[m, 0], out[m, 1], out[m, 2] = lb, lg, lr
    out[m, 3] = 255.0

    result = Layer(f"{source_layer.name} - アメコミ風", w, h)
    result.image = _array_to_qimage(np.clip(out, 0, 255).astype(np.uint8))
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 24. 浮世絵 / 木版画風
# ═══════════════════════════════════════════════════════════════════════════════

class UkiyoeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("浮世絵 / 木版画風")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._levels = QSpinBox()
        self._levels.setRange(2, 8)
        self._levels.setValue(4)
        self._levels.setToolTip("摺る版の数。少ないほど版画らしい平坦な絵になる")
        form.addRow("版の数（色数）", self._levels)

        self._misalign = QSpinBox()
        self._misalign.setRange(0, 20)
        self._misalign.setValue(4)
        self._misalign.setSuffix(" px")
        self._misalign.setToolTip("版ごとの摺りズレ。手摺りらしい味が出る")
        form.addRow("版ズレ", self._misalign)

        self._sumi = QSpinBox()
        self._sumi.setRange(0, 8)
        self._sumi.setValue(2)
        self._sumi.setSuffix(" px")
        self._sumi.setToolTip("輪郭の墨線（主版）の太さ")
        form.addRow("墨線の太さ", self._sumi)

        self._paper = QSpinBox()
        self._paper.setRange(0, 100)
        self._paper.setValue(35)
        self._paper.setSuffix(" %")
        self._paper.setToolTip("和紙の繊維のざらつき")
        form.addRow("和紙の質感", self._paper)

        self._paper_color = _color_button(QColor(240, 230, 205), self)
        self._paper_color.setToolTip("地の紙の色。生成りにすると古い摺物らしい")
        form.addRow("紙の色", self._paper_color)

        layout.addLayout(form)

        desc = QLabel("色数を絞って版画のように平坦化し、版ズレ・墨線・和紙の\n"
                      "質感を重ねます。元のレイヤーはそのまま残ります。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "levels": self._levels.value(),
            "misalign": self._misalign.value(),
            "sumi": self._sumi.value(),
            "paper": self._paper.value() / 100.0,
            "paper_color": self._paper_color._color,
        }


def execute_ukiyoe(layer_stack: LayerStack, source_layer: Layer,
                   params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    levels = max(2, int(params.get("levels", 4)))
    misalign = max(0, int(params.get("misalign", 4)))
    sumi = max(0, int(params.get("sumi", 2)))
    paper = float(params.get("paper", 0.35))
    paper_color = params.get("paper_color") or QColor(240, 230, 205)

    arr = _qimage_to_array(src_img).astype(np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    if not (alpha > 10).any():
        return None

    # 紙を敷く
    pb, pg, pr = _bgr(paper_color)
    out = np.zeros((h, w, 4), dtype=np.float32)
    out[:, :, 0], out[:, :, 1], out[:, :, 2] = pb, pg, pr
    out[:, :, 3] = 255.0

    # 平滑化して面をまとめる。ここでチャンネルごとに量子化してしまうと、
    # 肌色のように3チャンネルが同じ段に丸まる色が無彩色の灰色になって
    # しまうので、色は元のまま残し「どの版に属するか」だけを段で決める。
    smooth = cv2.bilateralFilter(rgb.astype(np.uint8), 9, 60, 60).astype(np.float32)

    # 段ごとに1版とみなし、版ごとに違う方向へずらして摺る。
    # 全部まとめてずらすと単に絵が動くだけで、版ズレには見えない。
    lum = np.clip(_luma(smooth), 0, 255)
    idx = _posterize_regions(lum, levels)
    inside = alpha > 10

    # 明るさだけで版を分けると、髪と肌のように明るさの近いものが1版に
    # まとまって、彫り分けたように見えない。色相でも版を分ける。
    # 色相を等分すると、肌と金髪のように近い色が同じ帯に入ってしまう。
    # 絵に実際に出てくる色相を k-means で拾って分ける。
    hsv = cv2.cvtColor(np.clip(smooth, 0, 255).astype(np.uint8),
                       cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    valid = inside & (sat > 50)
    hue_band = np.full(alpha.shape, -1, dtype=np.int32)
    n_hue = 0
    if int(np.count_nonzero(valid)) >= 8:
        hv = hsv[:, :, 0][valid].astype(np.float32)
        ang = hv / 180.0 * 2.0 * np.pi
        feat = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype(np.float32)
        n_hue = min(4, len(np.unique(hv)))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _c, lab, _ctr = cv2.kmeans(feat, n_hue, None, crit, 3,
                                   cv2.KMEANS_PP_CENTERS)
        hue_band[valid] = lab.reshape(-1)

    plates = []
    for lv in range(levels):
        for hb in range(-1, max(1, n_hue)):
            plates.append(inside & (idx == lv) & (hue_band == hb))

    for region in plates:
        if int(np.count_nonzero(region)) < 20:
            continue
        # この版の代表色。平均を取ると、明暗差のある面では左右の色が
        # 混ざって濁った灰色になってしまうので、中央値を使う。
        # 墨線（暗い縁）を含めると色が沈むため、彩度のある画素を優先する。
        sat_here = sat[region]
        cand = smooth[region]
        vivid = cand[sat_here > 50]
        plate_color = np.median(vivid if len(vivid) >= 10 else cand, axis=0)
        plate = region.astype(np.uint8) * 255
        if misalign > 0:
            dx = random.randint(-misalign, misalign)
            dy = random.randint(-misalign, misalign)
            plate = _shift_mask(plate, dx, dy)
        m = plate > 0
        for ch in range(3):
            out[m, ch] = plate_color[ch]

    # 墨線（主版）は版ズレさせず最後に乗せる。輪郭がぶれると絵が読めなくなる
    if sumi > 0:
        edge = _edge_mask(rgb, alpha, sumi)
        m = edge > 0
        out[m, 0], out[m, 1], out[m, 2] = 30.0, 28.0, 25.0

    # 和紙の繊維
    if paper > 0:
        tex = _paper_texture((h, w), paper * 40.0)
        out[:, :, :3] = np.clip(out[:, :, :3] + tex[:, :, None], 0, 255)

    result = Layer(f"{source_layer.name} - 浮世絵風", w, h)
    result.image = _array_to_qimage(np.clip(out, 0, 255).astype(np.uint8))
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 25. 印象派 / 油彩タッチ風
# ═══════════════════════════════════════════════════════════════════════════════

class ImpressionistDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("印象派 / 油彩タッチ風")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._length = QSpinBox()
        self._length.setRange(3, 40)
        self._length.setValue(12)
        self._length.setSuffix(" px")
        self._length.setToolTip("1本の筆跡の長さ")
        form.addRow("筆跡の長さ", self._length)

        self._width = QSpinBox()
        self._width.setRange(1, 12)
        self._width.setValue(3)
        self._width.setSuffix(" px")
        form.addRow("筆跡の太さ", self._width)

        self._density = QSpinBox()
        self._density.setRange(50, 600)
        self._density.setValue(220)
        self._density.setSuffix(" %")
        self._density.setToolTip("画面を何回ぶん塗り重ねるか。多いほど密になるが遅くなる")
        form.addRow("筆の密度", self._density)

        self._jitter = QSpinBox()
        self._jitter.setRange(0, 60)
        self._jitter.setValue(20)
        self._jitter.setToolTip("1本ごとの色のばらつき。大きいと絵の具らしくなる")
        form.addRow("色のゆらぎ", self._jitter)

        self._follow = QCheckBox("明暗の流れに沿って筆を向ける")
        self._follow.setChecked(True)
        self._follow.setToolTip("オフにするとすべて同じ角度の筆跡になる")
        form.addRow("", self._follow)

        layout.addLayout(form)

        desc = QLabel("短い筆跡を無数に置いて、油絵のタッチに置き換えます。\n"
                      "密度を上げると時間がかかります。\n"
                      "元のレイヤーはそのまま残ります。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "length": self._length.value(),
            "width": self._width.value(),
            "density": self._density.value() / 100.0,
            "jitter": self._jitter.value(),
            "follow": self._follow.isChecked(),
        }


def execute_impressionist(layer_stack: LayerStack, source_layer: Layer,
                          params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    length = max(3, int(params.get("length", 12)))
    width = max(1, int(params.get("width", 3)))
    density = max(0.1, float(params.get("density", 2.2)))
    jitter = max(0, int(params.get("jitter", 20)))
    follow = params.get("follow", True)

    arr = _qimage_to_array(src_img).astype(np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    if not (alpha > 10).any():
        return None

    # 筆を置ける場所（絵のある所）だけを候補にする
    ys, xs = np.nonzero(alpha > 10)
    if len(xs) == 0:
        return None

    # 筆の向き: 明度の勾配に直交する方向へ引くと、輪郭をなぞる流れになる
    gray = np.clip(_luma(rgb), 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5)

    # 先に下塗りを敷く。筆跡だけだと隙間が白く抜けて点々が残る。
    out = np.zeros((h, w, 4), dtype=np.uint8)
    base = cv2.GaussianBlur(rgb, (0, 0), max(1.0, length / 3.0))
    out[:, :, :3] = np.clip(base, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.where(alpha > 10, 255, 0).astype(np.uint8)

    # 1本の筆が覆う面積からストローク数を決める
    per_stroke = max(1.0, float(length * width))
    n_strokes = int(len(xs) / per_stroke * density)
    n_strokes = max(20, min(n_strokes, 120000))

    order = np.random.randint(0, len(xs), size=n_strokes)
    base_angle = random.uniform(0, math.pi)
    for i in order:
        cx, cy = int(xs[i]), int(ys[i])
        if follow:
            # 勾配ベクトルに直交＝等明度線に沿う向き
            ang = math.atan2(-gx[cy, cx], gy[cy, cx] + 1e-6)
        else:
            ang = base_angle
        ang += random.uniform(-0.25, 0.25)
        half = length * random.uniform(0.6, 1.2) * 0.5
        dx, dy = math.cos(ang) * half, math.sin(ang) * half
        b, g, r = rgb[cy, cx]
        if jitter:
            j = random.randint(-jitter, jitter)
            b, g, r = b + j, g + j, r + j
        color = (int(np.clip(b, 0, 255)), int(np.clip(g, 0, 255)),
                 int(np.clip(r, 0, 255)), 255)
        cv2.line(out, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)),
                 color, max(1, width), lineType=cv2.LINE_AA)

    # 筆がはみ出した先で下地が透けないよう、元が透明だった所は消す。
    # ただし筆1本ぶんは意図的にはみ出させたいので、alpha を少し広げてから切る。
    keep = cv2.dilate((alpha > 10).astype(np.uint8) * 255,
                      cv2.getStructuringElement(
                          cv2.MORPH_ELLIPSE, (length | 1, length | 1)))
    out[:, :, 3] = np.where(keep > 0, out[:, :, 3], 0)

    result = Layer(f"{source_layer.name} - 印象派風", w, h)
    result.image = _array_to_qimage(out)
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 26. ステンドグラス風
# ═══════════════════════════════════════════════════════════════════════════════

class StainedGlassDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ステンドグラス風")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._cell = QSpinBox()
        self._cell.setRange(10, 120)
        self._cell.setValue(34)
        self._cell.setSuffix(" px")
        self._cell.setToolTip("ガラス片のおおよその大きさ")
        form.addRow("ガラス片の大きさ", self._cell)

        self._lead = QSpinBox()
        self._lead.setRange(1, 12)
        self._lead.setValue(3)
        self._lead.setSuffix(" px")
        self._lead.setToolTip("ガラスを繋ぐ鉛線の太さ")
        form.addRow("鉛線の太さ", self._lead)

        self._lead_color = _color_button(QColor(25, 22, 20), self)
        form.addRow("鉛線の色", self._lead_color)

        self._vivid = QSpinBox()
        self._vivid.setRange(100, 300)
        self._vivid.setValue(170)
        self._vivid.setSuffix(" %")
        self._vivid.setToolTip("光が透ける感じを出すための彩度・明度の強調")
        form.addRow("透過光の強さ", self._vivid)

        self._bg = QCheckBox("背景もガラスにする")
        self._bg.setChecked(True)
        self._bg.setToolTip("オフにすると絵のある部分だけがガラスになる")
        form.addRow("", self._bg)

        layout.addLayout(form)

        desc = QLabel("画面をガラス片に分割し、各片を1色で塗って鉛線で囲みます。\n"
                      "元のレイヤーはそのまま残ります。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "cell": self._cell.value(),
            "lead": self._lead.value(),
            "lead_color": self._lead_color._color,
            "vivid": self._vivid.value() / 100.0,
            "glass_bg": self._bg.isChecked(),
        }


def execute_stained_glass(layer_stack: LayerStack, source_layer: Layer,
                          params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    cell = max(10, int(params.get("cell", 34)))
    lead = max(1, int(params.get("lead", 3)))
    lead_color = params.get("lead_color") or QColor(25, 22, 20)
    vivid = float(params.get("vivid", 1.7))
    glass_bg = params.get("glass_bg", True)

    arr = _qimage_to_array(src_img).astype(np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    if not (alpha > 10).any():
        return None

    # ガラス片の種: 格子を基準にランダムに散らす（ボロノイ風の不定形にする）
    step = cell
    seeds = []
    for gy in range(-1, h // step + 2):
        for gx in range(-1, w // step + 2):
            sx = gx * step + random.uniform(-step * 0.35, step * 0.35)
            sy = gy * step + random.uniform(-step * 0.35, step * 0.35)
            seeds.append((sx, sy))
    if not seeds:
        return None
    seed_arr = np.array(seeds, dtype=np.float32)

    # 各画素を最も近い種に割り当てる。全画素×全種の距離は重いので、
    # 種を格子状に置いてある性質を使い、近傍9マスだけを調べる。
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    n_gx = w // step + 3
    labels = np.zeros((h, w), dtype=np.int32)
    best = np.full((h, w), np.inf, dtype=np.float32)
    for k, (sx, sy) in enumerate(seeds):
        # この種が影響しうる範囲だけを更新する
        x0, x1 = max(0, int(sx - step * 2)), min(w, int(sx + step * 2))
        y0, y1 = max(0, int(sy - step * 2)), min(h, int(sy + step * 2))
        if x1 <= x0 or y1 <= y0:
            continue
        d = (xx[y0:y1, x0:x1] - sx) ** 2 + (yy[y0:y1, x0:x1] - sy) ** 2
        sub_best = best[y0:y1, x0:x1]
        m = d < sub_best
        sub_best[m] = d[m]
        labels[y0:y1, x0:x1][m] = k

    # 片ごとに平均色で塗る
    n_seeds = len(seeds)
    flat_lab = labels.reshape(-1)
    inside = (alpha > 10).reshape(-1)
    out = np.zeros((h, w, 4), dtype=np.float32)
    sums = np.zeros((n_seeds, 3), dtype=np.float64)
    counts = np.zeros(n_seeds, dtype=np.int64)
    flat_rgb = rgb.reshape(-1, 3).astype(np.float64)
    np.add.at(counts, flat_lab[inside], 1)
    for ch in range(3):
        np.add.at(sums[:, ch], flat_lab[inside], flat_rgb[inside, ch])
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / np.maximum(counts, 1)[:, None]

    # 透過光らしく彩度と明度を持ち上げる
    mean = np.clip((mean - 128.0) * vivid + 128.0, 0, 255)
    # 片の面積のうちどれだけが絵に覆われていたか。1画素でもかすっただけの
    # 片まで塗ると、絵の外へガラスが大きくはみ出してしまうので、
    # 半分以上が絵だった片だけを「絵のある片」とみなす。
    total = np.zeros(n_seeds, dtype=np.int64)
    np.add.at(total, flat_lab, 1)
    filled = counts > np.maximum(1, total // 2)
    colored = mean[flat_lab].reshape(h, w, 3)
    has_color = filled[flat_lab].reshape(h, w)

    out[:, :, :3] = colored
    if glass_bg:
        # 絵の無い片は暗いガラスにして、画面全体を埋める
        out[~has_color, 0] = 40.0
        out[~has_color, 1] = 35.0
        out[~has_color, 2] = 30.0
        out[:, :, 3] = 255.0
    else:
        out[:, :, 3] = np.where(has_color, 255.0, 0.0)

    # 鉛線: 隣り合う片のラベルが違う所が境界
    dx = np.zeros((h, w), dtype=np.uint8)
    dx[:, 1:] = (labels[:, 1:] != labels[:, :-1]).astype(np.uint8) * 255
    dy = np.zeros((h, w), dtype=np.uint8)
    dy[1:, :] = (labels[1:, :] != labels[:-1, :]).astype(np.uint8) * 255
    border = np.maximum(dx, dy)
    if lead > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lead, lead))
        border = cv2.dilate(border, k)
    lb, lg, lr = _bgr(lead_color)
    m = border > 0
    if not glass_bg:
        # 背景をガラスにしない場合、絵の外にまで鉛線を引かない
        m = m & has_color
    out[m, 0], out[m, 1], out[m, 2] = lb, lg, lr
    out[m, 3] = 255.0

    result = Layer(f"{source_layer.name} - ステンドグラス風", w, h)
    result.image = _array_to_qimage(np.clip(out, 0, 255).astype(np.uint8))
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 27. ブループリント / 設計図風
# ═══════════════════════════════════════════════════════════════════════════════

class BlueprintDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ブループリント / 設計図風")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._paper_color = _color_button(QColor(20, 60, 130), self)
        self._paper_color.setToolTip("図面の地の色")
        form.addRow("地の色", self._paper_color)

        self._ink_color = _color_button(QColor(235, 243, 255), self)
        form.addRow("線の色", self._ink_color)

        self._thickness = QSpinBox()
        self._thickness.setRange(1, 8)
        self._thickness.setValue(2)
        self._thickness.setSuffix(" px")
        form.addRow("線の太さ", self._thickness)

        self._grid = QSpinBox()
        self._grid.setRange(0, 200)
        self._grid.setValue(32)
        self._grid.setSuffix(" px")
        self._grid.setToolTip("0 にすると方眼を描きません")
        form.addRow("方眼の間隔", self._grid)

        self._major = QSpinBox()
        self._major.setRange(2, 20)
        self._major.setValue(5)
        self._major.setToolTip("何マスごとに濃い線を引くか")
        form.addRow("太線の間隔", self._major)

        self._fade = QSpinBox()
        self._fade.setRange(0, 100)
        self._fade.setValue(25)
        self._fade.setSuffix(" %")
        self._fade.setToolTip("青焼きらしいムラ・かすれ")
        form.addRow("かすれ", self._fade)

        layout.addLayout(form)

        desc = QLabel("輪郭だけを抜き出して青地に白線で描き、方眼を敷いて\n"
                      "設計図のように見せます。元のレイヤーはそのまま残ります。")
        desc.setStyleSheet("color: #666; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        buttons = _std_buttons()
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def params(self) -> dict:
        return {
            "paper_color": self._paper_color._color,
            "ink_color": self._ink_color._color,
            "thickness": self._thickness.value(),
            "grid": self._grid.value(),
            "major": self._major.value(),
            "fade": self._fade.value() / 100.0,
        }


def execute_blueprint(layer_stack: LayerStack, source_layer: Layer,
                      params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    paper_color = params.get("paper_color") or QColor(20, 60, 130)
    ink_color = params.get("ink_color") or QColor(235, 243, 255)
    thickness = max(1, int(params.get("thickness", 2)))
    grid = max(0, int(params.get("grid", 32)))
    major = max(2, int(params.get("major", 5)))
    fade = float(params.get("fade", 0.25))

    arr = _qimage_to_array(src_img).astype(np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    if not (alpha > 10).any():
        return None

    pb, pg, pr = _bgr(paper_color)
    ib, ig, ir = _bgr(ink_color)
    out = np.zeros((h, w, 4), dtype=np.float32)
    out[:, :, 0], out[:, :, 1], out[:, :, 2] = pb, pg, pr
    out[:, :, 3] = 255.0

    ink = np.zeros((h, w), dtype=np.float32)

    # 方眼。太線を先に敷き、その上から細線を引く
    if grid > 0:
        thin = np.zeros((h, w), dtype=np.float32)
        thick = np.zeros((h, w), dtype=np.float32)
        for x in range(0, w, grid):
            col = thick if (x // grid) % major == 0 else thin
            col[:, x:x + 1] = 1.0
        for y in range(0, h, grid):
            row = thick if (y // grid) % major == 0 else thin
            row[y:y + 1, :] = 1.0
        ink = np.maximum(ink, thin * 0.18)
        ink = np.maximum(ink, thick * 0.35)

    # 図の輪郭
    edge = _edge_mask(rgb, alpha, thickness).astype(np.float32) / 255.0
    ink = np.maximum(ink, edge)

    # 青焼きのムラ
    if fade > 0:
        tex = _paper_texture((h, w), fade, scale=6)
        ink = np.clip(ink * (1.0 + tex), 0.0, 1.0)
        out[:, :, :3] = np.clip(out[:, :, :3] + tex[:, :, None] * 18.0, 0, 255)

    for ch, v in enumerate((ib, ig, ir)):
        out[:, :, ch] = out[:, :, ch] * (1.0 - ink) + v * ink

    result = Layer(f"{source_layer.name} - 設計図風", w, h)
    result.image = _array_to_qimage(np.clip(out, 0, 255).astype(np.uint8))
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 28. 銅版画 / エッチング風
# ═══════════════════════════════════════════════════════════════════════════════

class EtchingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("銅版画 / エッチング風")
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self._spacing = QSpinBox()
        self._spacing.setRange(2, 20)
        self._spacing.setValue(4)
        self._spacing.setSuffix(" px")
        self._spacing.setToolTip("彫り線の間隔。細かいほど緻密な版画になる")
        form.addRow("線の間隔", self._spacing)

        self._layers = QSpinBox()
        self._layers.setRange(2, 6)
        self._layers.setValue(4)
        self._layers.setToolTip("重ねる彫りの方向数")
        form.addRow("彫りの段数", self._layers)

        self._wobble = QSpinBox()
        self._wobble.setRange(0, 100)
        self._wobble.setValue(45)
        self._wobble.setSuffix(" %")
        self._wobble.setToolTip("線の不規則さ。手彫りらしい揺らぎが出る")
        form.addRow("線の揺らぎ", self._wobble)

        self._ink_color = _color_button(QColor(35, 28, 22), self)
        form.addRow("インクの色", self._ink_color)

        self._paper_color = _color_button(QColor(238, 230, 214), self)
        form.addRow("紙の色", self._paper_color)

        self._grain = QSpinBox()
        self._grain.setRange(0, 100)
        self._grain.setValue(30)
        self._grain.setSuffix(" %")
        form.addRow("紙の質感", self._grain)

        layout.addLayout(form)

        desc = QLabel("明暗を細かい彫り線の密度に置き換えて、銅版画のように\n"
                      "仕上げます。元のレイヤーはそのまま残ります。")
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
            "layers": self._layers.value(),
            "wobble": self._wobble.value() / 100.0,
            "ink_color": self._ink_color._color,
            "paper_color": self._paper_color._color,
            "grain": self._grain.value() / 100.0,
        }


def execute_etching(layer_stack: LayerStack, source_layer: Layer,
                    params: dict) -> Layer | None:
    if source_layer.is_group:
        return None
    src_img: QImage = source_layer.image
    w, h = src_img.width(), src_img.height()
    if w == 0 or h == 0:
        return None

    spacing = max(2, int(params.get("spacing", 4)))
    n_layers = max(2, int(params.get("layers", 4)))
    wobble = float(params.get("wobble", 0.45))
    ink_color = params.get("ink_color") or QColor(35, 28, 22)
    paper_color = params.get("paper_color") or QColor(238, 230, 214)
    grain = float(params.get("grain", 0.3))

    arr = _qimage_to_array(src_img).astype(np.float32)
    rgb, alpha = arr[:, :, :3], arr[:, :, 3]
    if not (alpha > 10).any():
        return None

    lum = np.clip(_luma(rgb), 0, 255)
    # 透明部は紙のまま残したいので、明るさ最大（＝彫らない）とみなす
    lum = np.where(alpha > 10, lum, 255.0)
    darkness = 1.0 - lum / 255.0
    # 絵の明暗の幅が狭いと全面が同じ密度の網になって濃淡が読めなくなる。
    # 絵のある範囲の実際の明暗を 0〜1 いっぱいに引き伸ばす。
    if (alpha > 10).any():
        inside_d = darkness[alpha > 10]
        lo, hi = float(inside_d.min()), float(inside_d.max())
        if hi - lo > 0.01:
            darkness = np.clip((darkness - lo) / (hi - lo), 0.0, 1.0)
        # そのままだと中間調に固まりがちなので、ガンマで明部側を開いて
        # 「白く残る所」と「彫り込む所」の差をはっきりさせる
        darkness = np.power(darkness, 1.6)
        darkness = np.where(alpha > 10, darkness, 0.0)

    angles = [15.0, 105.0, 60.0, 150.0, 35.0, 125.0][:n_layers]
    ink = np.zeros((h, w), dtype=np.float32)
    for i, ang in enumerate(angles):
        lo = i / n_layers
        lines = _hatch_lines((h, w), ang, spacing, 1).astype(np.float32) / 255.0
        if wobble > 0:
            # 線を波打たせて手彫りらしくする。まっすぐだと機械的に見える
            amp = spacing * wobble
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            phase = random.uniform(0, math.pi * 2)
            off = np.sin(yy / max(2.0, spacing * 3.0) + phase) * amp
            map_x = np.clip(xx + off, 0, w - 1).astype(np.float32)
            map_y = yy.astype(np.float32)
            lines = cv2.remap(lines, map_x, map_y, cv2.INTER_LINEAR)
        weight = np.clip((darkness - lo) * n_layers, 0.0, 1.0)
        ink = np.maximum(ink, lines * weight)

    # 真っ黒に近い所はベタで潰す。版画でも最暗部は面で潰れる
    ink = np.maximum(ink, np.clip((darkness - 0.88) * 8.0, 0.0, 1.0))

    pb, pg, pr = _bgr(paper_color)
    ib, ig, ir = _bgr(ink_color)
    out = np.zeros((h, w, 4), dtype=np.float32)
    out[:, :, 0], out[:, :, 1], out[:, :, 2] = pb, pg, pr
    out[:, :, 3] = 255.0
    for ch, v in enumerate((ib, ig, ir)):
        out[:, :, ch] = out[:, :, ch] * (1.0 - ink) + v * ink

    if grain > 0:
        tex = _paper_texture((h, w), grain * 35.0)
        out[:, :, :3] = np.clip(out[:, :, :3] + tex[:, :, None], 0, 255)

    result = Layer(f"{source_layer.name} - 銅版画風", w, h)
    result.image = _array_to_qimage(np.clip(out, 0, 255).astype(np.uint8))
    _copy_offset(source_layer, result)
    _insert_result_layer(layer_stack, source_layer, result)
    return result

# ═══════════════════════════════════════════════════════════════════════════════
# 29. アクションガチャ
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
    ("contour", "等高線"),
    ("halftone", "網点"),
    ("dither", "ディザ"),
    ("crosshatch", "ハッチング"),
    ("vhs", "VHS"),
    ("crt", "CRT/LED"),
    ("warhol", "ウォーホル風"),
    ("lichtenstein", "アメコミ風"),
    ("ukiyoe", "浮世絵風"),
    ("impressionist", "印象派風"),
    ("stained_glass", "ステンドグラス風"),
    ("blueprint", "設計図風"),
    ("etching", "銅版画風"),
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
                "expand": ri(0, 10), "shift": ri(0, 12),
                "close_gap": ri(0, 4), "line_sensitivity": ri(0, 100)}
    if key == "wobble":
        return {"strength": ri(3, 18), "wavelength": ri(30, 180),
                "gap": ri(0, 40)}
    if key == "stamp":
        lo = ri(1, 5)
        return {"strength": ri(20, 60), "grain": ri(1, 5), "blots": rb(0.6),
                "blot_min": lo, "blot_max": lo + ri(1, 8),
                # 7割は線の色になじませ、たまにパレットの色を差す
                "blot_color": None if rb(0.7) else pick()}
    if key == "contour":
        return {"count": ri(2, 6), "spacing": ri(8, 30),
                "color": pick(), "thickness": ri(1, 4), "fade": rb(0.7)}
    if key == "halftone":
        return {"pitch": ri(6, 20), "mode": "rgb" if rb(0.75) else "mono",
                "background": "white" if rb(0.6) else "transparent",
                "smooth": True}
    if key == "dither":
        # パレットを渡すとガチャの配色でそのまま減色される
        return {"method": random.choice(["bayer", "diffusion"]),
                "levels": ri(2, 6), "pixel": ri(2, 8),
                "palette": colors if rb(0.6) else None}
    if key == "crosshatch":
        return {"spacing": ri(5, 16), "thickness": ri(1, 3),
                "layers": ri(2, 4), "color": dark if rb(0.7) else pick()}
    if key == "vhs":
        return {"jitter": ri(4, 30), "bands": ri(1, 8),
                "scanline": ru(0.15, 0.55), "scan_pitch": ri(2, 6),
                "noise": ru(0.05, 0.4)}
    if key == "crt":
        return {"kind": "crt" if rb(0.6) else "led", "cell": ri(4, 12),
                "bloom": ru(0.2, 0.8), "boost": ru(1.3, 2.2),
                "saturation": ru(1.0, 1.8)}
    if key == "warhol":
        cols, rows = random.choice([(2, 2), (2, 2), (3, 3), (3, 1), (1, 3)])
        # ガチャの配色をコマ数ぶん回して使う。1コマごとに並びを変えないと
        # 全コマ同じ色になってしまうので、シャッフルした版を作る
        n_cells = cols * rows
        sets = []
        for _ in range(n_cells):
            shuffled = list(colors)
            random.shuffle(shuffled)
            sets.append(shuffled)
        return {"cols": cols, "rows": rows, "levels": ri(3, 5),
                "gap": ri(0, 12), "outline": ri(0, 4), "palettes": sets}
    if key == "lichtenstein":
        return {"pitch": ri(4, 12), "outline": ri(2, 5), "levels": ri(2, 4),
                "dot_color": pick(), "line_color": dark,
                "white_bg": rb(0.7)}
    if key == "ukiyoe":
        return {"levels": ri(3, 6), "misalign": ri(0, 10), "sumi": ri(1, 4),
                "paper": ru(0.15, 0.5),
                "paper_color": light if rb(0.5) else QColor(240, 230, 205)}
    if key == "impressionist":
        return {"length": ri(6, 20), "width": ri(2, 5),
                "density": ru(1.5, 3.0), "jitter": ri(5, 35),
                "follow": rb(0.75)}
    if key == "stained_glass":
        return {"cell": ri(18, 60), "lead": ri(2, 6), "lead_color": dark,
                "vivid": ru(1.2, 2.2), "glass_bg": rb(0.6)}
    if key == "blueprint":
        return {"paper_color": dark, "ink_color": light,
                "thickness": ri(1, 4), "grid": random.choice([0, 16, 24, 32, 48]),
                "major": ri(3, 8), "fade": ru(0.1, 0.4)}
    if key == "etching":
        return {"spacing": ri(3, 8), "layers": ri(3, 5), "wobble": ru(0.2, 0.7),
                "ink_color": dark, "paper_color": light,
                "grain": ru(0.15, 0.45)}
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
    "contour": execute_contour,
    "halftone": execute_halftone,
    "dither": execute_dither,
    "crosshatch": execute_crosshatch,
    "vhs": execute_vhs,
    "crt": execute_crt,
    "warhol": execute_warhol,
    "lichtenstein": execute_lichtenstein,
    "ukiyoe": execute_ukiyoe,
    "impressionist": execute_impressionist,
    "stained_glass": execute_stained_glass,
    "blueprint": execute_blueprint,
    "etching": execute_etching,
}


def _flatten_gacha_result(result, w: int, h: int) -> QImage:
    """効果の結果（レイヤー or グループ）をキャンバスサイズ1枚に焼く。"""
    buf = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    buf.fill(Qt.GlobalColor.transparent)
    p = QPainter(buf)
    if result.is_group:
        # グループも offset を持ちうるので明示的に反映する。現状の効果は
        # キャンバスサイズのグループしか作らないので 0 のままだが、
        # 0 決め打ちだとサイズ違いのグループが来た瞬間に位置がずれる。
        p.drawImage(getattr(result, 'offset_x', 0),
                    getattr(result, 'offset_y', 0), result.composite())
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
            # 1効果の失敗でガチャ全体を止めない。ただし黙って握り潰すと
            # 本当の不具合に気づけないので、内容だけは残す。
            traceback.print_exc()
            result = None
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
            ("🗺️ 等高線", "外側に輪郭線を何重にも生成", self._on_contour),
            ("🔴 カラーハーフトーン", "印刷物のような網点ドットに変換", self._on_halftone),
            ("👾 ディザ / レトロ減色", "色数を落としてレトロゲーム画面風に", self._on_dither),
            ("✒️ クロスハッチング", "明暗を斜線の密度に置き換えてペン画風に", self._on_crosshatch),
            ("📼 VHS / 走査線ノイズ", "行ずれ・ノイズ帯・走査線で古い映像風に", self._on_vhs),
            ("🖥️ CRT / LED 画面", "RGBサブピクセルと走査線で画面越しの絵に", self._on_crt),
            ("💋 アンディ・ウォーホル風", "縮小してタイル状に並べ、コマごとに違う配色で", self._on_warhol),
            ("💥 リキテンスタイン風", "太い輪郭＋ベタ塗り＋網点でアメコミの1コマに", self._on_lichtenstein),
            ("🌊 浮世絵 / 木版画風", "色数を絞って版ズレ・墨線・和紙の質感を重ねる", self._on_ukiyoe),
            ("🎨 印象派 / 油彩タッチ風", "短い筆跡を無数に置いて油絵のタッチに", self._on_impressionist),
            ("🪟 ステンドグラス風", "ガラス片に分割してベタ塗り＋鉛線で囲む", self._on_stained_glass),
            ("📐 ブループリント / 設計図風", "青地に白線と方眼で図面のように", self._on_blueprint),
            ("🖨️ 銅版画 / エッチング風", "明暗を細かい彫り線の密度に置き換える", self._on_etching),
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

    def _on_contour(self):
        self._run("等高線", ContourDialog, execute_contour)

    def _on_halftone(self):
        self._run("カラーハーフトーン", HalftoneDialog, execute_halftone)

    def _on_dither(self):
        self._run("ディザ / レトロ減色", DitherDialog, execute_dither)

    def _on_crosshatch(self):
        self._run("クロスハッチング", CrosshatchDialog, execute_crosshatch)

    def _on_vhs(self):
        self._run("VHS / 走査線ノイズ", VhsDialog, execute_vhs)

    def _on_crt(self):
        self._run("CRT / LED 画面", CrtDialog, execute_crt)

    def _on_warhol(self):
        self._run("アンディ・ウォーホル風", WarholDialog, execute_warhol)

    def _on_lichtenstein(self):
        self._run("リキテンスタイン風", LichtensteinDialog, execute_lichtenstein)

    def _on_ukiyoe(self):
        self._run("浮世絵 / 木版画風", UkiyoeDialog, execute_ukiyoe)

    def _on_impressionist(self):
        self._run("印象派 / 油彩タッチ風", ImpressionistDialog, execute_impressionist)

    def _on_stained_glass(self):
        self._run("ステンドグラス風", StainedGlassDialog, execute_stained_glass)

    def _on_blueprint(self):
        self._run("ブループリント / 設計図風", BlueprintDialog, execute_blueprint)

    def _on_etching(self):
        self._run("銅版画 / エッチング風", EtchingDialog, execute_etching)

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
