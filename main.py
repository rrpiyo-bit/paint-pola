import sys
import os
import io
import json
import zipfile
import cv2
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                              QScrollArea, QWidget, QHBoxLayout,
                              QMenu, QInputDialog, QFontDialog,
                              QColorDialog, QLabel, QStatusBar, QMessageBox,
                              QDialog, QDialogButtonBox, QSlider, QFormLayout,
                              QSpinBox, QRadioButton, QButtonGroup, QGroupBox,
                              QVBoxLayout, QSplitter)
from PyQt6.QtCore import Qt, QSize, QByteArray, QBuffer, QIODevice, QPropertyAnimation, QEasingCurve, QRectF, pyqtSignal
from PyQt6.QtGui import (QAction, QImage, QPixmap, QKeySequence, QColor,
                          QFont, QPainter, QIcon)

from layer import LayerStack, Layer, GroupLayer, CANVAS_W, CANVAS_H
from animation_panel import AnimationPanel
from canvas import Canvas
from toolbar import Toolbar
from layer_panel import LayerPanel
from navigator import NavigatorPanel
from color_panel import ColorPanel
from tools import Tool
from tool_options_panel import ToolOptionsPanel
from toolbar import make_tool_cursors
from themes import get_theme_keys, get_theme_label, get_theme_qss


class LineExtractionDialog(QDialog):
    """線画抽出ダイアログ。
    閾値処理（暗いピクセルを線として抽出）＋リアルタイムプレビュー付き。"""

    PREVIEW_MAX = 400  # プレビュー画像の最大辺 px

    def __init__(self, source: QImage, parent=None):
        super().__init__(parent)
        self.setWindowTitle("線画抽出")
        self.setMinimumWidth(480)

        # --- numpy グレースケールをあらかじめ作成 ---
        self._source = source
        sw, sh = source.width(), source.height()
        ptr = source.bits()
        ptr.setsize(sh * sw * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(sh, sw, 4).copy()
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        self._gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # プレビュー用縮小スケール
        scale = min(1.0, self.PREVIEW_MAX / max(sw, sh))
        pw = max(1, int(sw * scale))
        ph = max(1, int(sh * scale))
        self._gray_small = cv2.resize(self._gray, (pw, ph), interpolation=cv2.INTER_AREA)
        self._preview_size = (pw, ph)

        # --- レイアウト ---
        root = QVBoxLayout(self)

        # プレビュー
        self._preview_label = QLabel()
        self._preview_label.setFixedSize(pw, ph)
        self._preview_label.setStyleSheet("background:#888; border:1px solid #555;")
        self._preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._preview_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # しきい値スライダー
        form = QFormLayout()
        root.addLayout(form)

        thresh_row = QHBoxLayout()
        self._thresh = QSlider(Qt.Orientation.Horizontal)
        self._thresh.setRange(0, 255)
        self._thresh.setValue(180)
        self._thresh_label = QLabel("180")
        self._thresh_label.setFixedWidth(30)
        thresh_row.addWidget(self._thresh)
        thresh_row.addWidget(self._thresh_label)
        form.addRow("線のしきい値（明るさ）", thresh_row)

        # 説明
        hint = QLabel("スライダーを下げると薄い線も抽出されます。上げると濃い線だけになります。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px;")
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._thresh.valueChanged.connect(self._update_preview)
        self._update_preview(self._thresh.value())

    def _update_preview(self, value: int):
        self._thresh_label.setText(str(value))
        pw, ph = self._preview_size
        # 閾値より暗いピクセルを線（黒）として抽出
        _, mask = cv2.threshold(self._gray_small, value, 255, cv2.THRESH_BINARY_INV)
        # プレビュー: 白背景 + 黒線
        preview = np.full((ph, pw, 3), 255, dtype=np.uint8)
        preview[mask > 0] = [0, 0, 0]
        img = QImage(preview.tobytes(), pw, ph, pw * 3, QImage.Format.Format_RGB888).copy()
        self._preview_label.setPixmap(QPixmap.fromImage(img))

    def threshold(self) -> int:
        return self._thresh.value()

    def extract(self) -> QImage:
        """フルサイズで線画を生成して返す。"""
        h, w = self._gray.shape
        _, mask = cv2.threshold(self._gray, self.threshold(), 255, cv2.THRESH_BINARY_INV)
        bgra = np.zeros((h, w, 4), dtype=np.uint8)
        bgra[mask > 0] = [0, 0, 0, 255]
        return QImage(bgra.tobytes(), w, h, w * 4, QImage.Format.Format_ARGB32).copy()


class DespeckleDialog(QDialog):
    """フィルター → ゴミ取り: 面積が指定px²以下の孤立した不透明の塊を透明化する。
    キャンバス上でリアルタイムにプレビューし、OKで確定・キャンセルで元に戻す。"""

    def __init__(self, canvas, layer, area_mask: np.ndarray | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ゴミ取り")
        self.setMinimumWidth(360)
        self._canvas = canvas
        self._layer = layer
        self._orig_image = layer.image.copy()
        self._area_mask = area_mask  # 選択範囲マスク（None ならレイヤー全体が対象）

        img = self._orig_image
        w, h = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(img.sizeInBytes())
        self._orig_arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()

        root = QVBoxLayout(self)

        row = QHBoxLayout()
        self._size = QSlider(Qt.Orientation.Horizontal)
        self._size.setRange(1, 500)
        self._size.setValue(9)
        self._size_label = QLabel("9 px²")
        self._size_label.setFixedWidth(60)
        row.addWidget(self._size)
        row.addWidget(self._size_label)
        form = QFormLayout()
        form.addRow("除去する面積のしきい値", row)
        root.addLayout(form)

        hint = QLabel("指定した面積（px²）以下の、孤立した塗り残し・ゴミ点を透明化して消します。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666; font-size:11px;")
        root.addWidget(hint)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        self._size.valueChanged.connect(self._update_preview)
        self._update_preview(self._size.value())

    def _update_preview(self, value: int):
        self._size_label.setText(f"{value} px²")
        arr = self._orig_arr.copy()
        alpha = arr[:, :, 3]
        opaque = (alpha > 0).astype(np.uint8)
        if self._area_mask is not None:
            opaque = opaque & self._area_mask
        num, labels, stats, _ = cv2.connectedComponentsWithStats(opaque, connectivity=8)
        # ラベルごとに arr[labels==i]=0 をループすると、ラベル数×画像サイズの計算量になり
        # スライダードラッグ中（valueChanged が高頻度発火）に固まる原因になっていたため、
        # しきい値以下のラベルIDだけを一括で調べる numpy ベクトル化に置き換える。
        areas = stats[:, cv2.CC_STAT_AREA]
        small_label_mask = areas <= value
        small_label_mask[0] = False  # 背景(ラベル0)は対象外
        remove_mask = small_label_mask[labels]
        arr[remove_mask] = 0
        h, w = arr.shape[:2]
        result = QImage(arr.tobytes(), w, h, w * 4, QImage.Format.Format_ARGB32).copy()
        self._layer.image = result
        self._canvas.update()

    def reject(self):
        self._layer.image = self._orig_image
        self._canvas.update()
        super().reject()


class NewCanvasDialog(QDialog):
    PRESETS = [
        ("正方形 2500px",  2500, 2500),
        ("A4 縦 300dpi", 2480, 3508),
        ("A4 横 300dpi", 3508, 2480),
        ("A5 縦 300dpi", 1748, 2480),
        ("正方形 2000px",  2000, 2000),
        ("カスタム", 0, 0),
    ]

    def __init__(self, current_w: int = CANVAS_W, current_h: int = CANVAS_H, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新規キャンバス")
        layout = QFormLayout(self)

        self._preset_group = QButtonGroup(self)
        preset_box = QGroupBox("プリセット")
        preset_layout = QVBoxLayout(preset_box)
        for i, (label, pw, ph) in enumerate(self.PRESETS):
            rb = QRadioButton(label)
            self._preset_group.addButton(rb, i)
            preset_layout.addWidget(rb)
        self._preset_group.button(0).setChecked(True)
        self._preset_group.idClicked.connect(self._on_preset)
        layout.addRow(preset_box)

        self._w = QSpinBox()
        self._w.setRange(1, 10000)
        self._w.setValue(CANVAS_W)
        self._w.setSuffix(" px")
        layout.addRow("幅", self._w)

        self._h = QSpinBox()
        self._h.setRange(1, 10000)
        self._h.setValue(CANVAS_H)
        self._h.setSuffix(" px")
        layout.addRow("高さ", self._h)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

        self._on_preset(0)

    def _on_preset(self, idx: int):
        _, pw, ph = self.PRESETS[idx]
        if pw > 0:
            self._w.setValue(pw)
            self._h.setValue(ph)
            self._w.setEnabled(False)
            self._h.setEnabled(False)
        else:
            self._w.setEnabled(True)
            self._h.setEnabled(True)

    def values(self) -> tuple[int, int]:
        return self._w.value(), self._h.value()


class AnchorWidget(QWidget):
    """9点アンカーポイント選択ウィジェット（CSP風）。"""
    anchor_changed = pyqtSignal(int, int)  # (ax, ay) 0=左/上, 1=中央, 2=右/下

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(72, 72)
        self._ax = 1
        self._ay = 1

    def anchor(self) -> tuple[int, int]:
        return self._ax, self._ay

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(240, 240, 240))
        p.setPen(QColor(180, 180, 180))
        p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        for ay in range(3):
            for ax in range(3):
                cx = 12 + ax * 24
                cy = 12 + ay * 24
                if ax == self._ax and ay == self._ay:
                    p.setBrush(QColor(80, 140, 255))
                    p.setPen(QColor(40, 80, 200))
                    p.drawEllipse(cx - 8, cy - 8, 16, 16)
                else:
                    p.setBrush(QColor(200, 200, 200))
                    p.setPen(QColor(150, 150, 150))
                    p.drawEllipse(cx - 5, cy - 5, 10, 10)
        # ガイド線
        p.setPen(QColor(200, 200, 200))
        for i in range(3):
            y = 12 + i * 24
            p.drawLine(12, y, 60, y)
            x = 12 + i * 24
            p.drawLine(x, 12, x, 60)
        p.end()

    def mousePressEvent(self, event):
        x = int(event.position().x())
        y = int(event.position().y())
        ax = max(0, min(2, (x - 0) * 3 // self.width()))
        ay = max(0, min(2, (y - 0) * 3 // self.height()))
        if ax != self._ax or ay != self._ay:
            self._ax = ax
            self._ay = ay
            self.update()
            self.anchor_changed.emit(ax, ay)


class CanvasPreviewWidget(QWidget):
    """キャンバスサイズ変更のリアルタイムプレビュー。"""
    def __init__(self, composite: QImage, parent=None):
        super().__init__(parent)
        self._composite = composite
        self._current_w = composite.width()
        self._current_h = composite.height()
        self._new_w = composite.width()
        self._new_h = composite.height()
        self._anchor = (1, 1)
        self._mode = "crop"
        self.setFixedSize(220, 220)

    def set_params(self, new_w: int, new_h: int, anchor: tuple[int, int], mode: str):
        self._new_w = new_w
        self._new_h = new_h
        self._anchor = anchor
        self._mode = mode
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor(80, 80, 80))

        pw, ph = self.width() - 20, self.height() - 20
        # 表示領域に合わせてスケーリング
        max_dim = max(self._current_w, self._current_h, self._new_w, self._new_h)
        if max_dim == 0:
            p.end()
            return
        scale = min(pw / max_dim, ph / max_dim)

        ox = (self.width() - max_dim * scale) / 2
        oy = (self.height() - max_dim * scale) / 2

        # 新しいキャンバス範囲（白枠）
        nw_s = self._new_w * scale
        nh_s = self._new_h * scale

        ax, ay = self._anchor
        # アンカーに基づくオフセット
        if ax == 0:
            nx = ox
        elif ax == 1:
            nx = ox + (max_dim * scale - nw_s) / 2
        else:
            nx = ox + max_dim * scale - nw_s

        if ay == 0:
            ny = oy
        elif ay == 1:
            ny = oy + (max_dim * scale - nh_s) / 2
        else:
            ny = oy + max_dim * scale - nh_s

        # 新キャンバス背景（白）
        p.fillRect(QRectF(nx, ny, nw_s, nh_s), QColor(255, 255, 255))

        # 現在の画像を配置
        cw_s = self._current_w * scale
        ch_s = self._current_h * scale

        if self._mode == "scale":
            # リサイズモード: 新サイズに拡縮
            scaled = self._composite.scaled(
                int(nw_s), int(nh_s),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            p.drawImage(int(nx), int(ny), scaled)
        else:
            # 切り抜きモード: アンカー基準で配置
            if ax == 0:
                cx = nx
            elif ax == 1:
                cx = nx + (nw_s - cw_s) / 2
            else:
                cx = nx + nw_s - cw_s

            if ay == 0:
                cy = ny
            elif ay == 1:
                cy = ny + (nh_s - ch_s) / 2
            else:
                cy = ny + nh_s - ch_s

            scaled_img = self._composite.scaled(
                int(cw_s), int(ch_s),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            p.drawImage(int(cx), int(cy), scaled_img)

        # 新キャンバス境界線
        p.setPen(QColor(80, 140, 255))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(QRectF(nx, ny, nw_s, nh_s))

        # サイズ表示
        p.setPen(QColor(200, 200, 200))
        p.drawText(5, self.height() - 5, f"{self._new_w} x {self._new_h}")

        p.end()


class ResizeCanvasDialog(QDialog):
    def __init__(self, current_w: int, current_h: int, parent=None, composite: QImage | None = None):
        super().__init__(parent)
        self.setWindowTitle("キャンバスサイズ変更")
        self._current_w = current_w
        self._current_h = current_h

        root = QHBoxLayout(self)

        # 左: 設定
        left = QVBoxLayout()
        form = QFormLayout()

        self._w = QSpinBox()
        self._w.setRange(1, 10000)
        self._w.setValue(current_w)
        self._w.setSuffix(" px")
        self._w.valueChanged.connect(self._on_param_changed)
        form.addRow("新しい幅", self._w)

        self._h = QSpinBox()
        self._h.setRange(1, 10000)
        self._h.setValue(current_h)
        self._h.setSuffix(" px")
        self._h.valueChanged.connect(self._on_param_changed)
        form.addRow("新しい高さ", self._h)

        left.addLayout(form)

        # アンカーポイント
        anchor_box = QGroupBox("基準点")
        anchor_layout = QHBoxLayout(anchor_box)
        self._anchor = AnchorWidget()
        self._anchor.anchor_changed.connect(self._on_param_changed)
        anchor_layout.addWidget(self._anchor)
        anchor_layout.addWidget(QLabel("キャンバスの\nどこを基準に\n拡大/縮小"))
        anchor_layout.addStretch()
        left.addWidget(anchor_box)

        # モード
        mode_box = QGroupBox("既存レイヤーの処理")
        mode_layout = QVBoxLayout(mode_box)
        self._mode_group = QButtonGroup(self)
        self._rb_crop = QRadioButton("切り抜き")
        self._rb_scale = QRadioButton("リサイズ（拡縮）")
        self._rb_crop.setChecked(True)
        self._mode_group.addButton(self._rb_crop, 0)
        self._mode_group.addButton(self._rb_scale, 1)
        mode_layout.addWidget(self._rb_crop)
        mode_layout.addWidget(self._rb_scale)
        self._rb_crop.toggled.connect(self._on_param_changed)
        left.addWidget(mode_box)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        left.addWidget(btns)

        root.addLayout(left)

        # 右: プレビュー
        if composite is None:
            composite = QImage(current_w, current_h, QImage.Format.Format_ARGB32)
            composite.fill(Qt.GlobalColor.white)
        self._preview = CanvasPreviewWidget(composite)
        root.addWidget(self._preview)

    def _on_param_changed(self, *_args):
        ax, ay = self._anchor.anchor()
        mode = "scale" if self._rb_scale.isChecked() else "crop"
        self._preview.set_params(self._w.value(), self._h.value(), (ax, ay), mode)

    def values(self) -> tuple[int, int, str, tuple[int, int]]:
        mode = "scale" if self._rb_scale.isChecked() else "crop"
        return self._w.value(), self._h.value(), mode, self._anchor.anchor()


class TransformPercentDialog(QDialog):
    """拡大縮小・回転をリアルタイムプレビューしながら%・°で指定するダイアログ。
    canvas.apply_transform_percentage() を値が変わるたびに呼ぶ。"""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self._canvas = canvas
        self.setWindowTitle("変形（拡大縮小・回転）")
        self.setMinimumWidth(340)
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )
        layout = QVBoxLayout(self)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # 横スケール
        sx_row = QHBoxLayout()
        self._sx_spin = QSpinBox()
        self._sx_spin.setRange(1, 1000)
        self._sx_spin.setValue(100)
        self._sx_spin.setSuffix(" %")
        self._sx_spin.setFixedWidth(80)
        self._sx_slider = QSlider(Qt.Orientation.Horizontal)
        self._sx_slider.setRange(1, 1000)
        self._sx_slider.setValue(100)
        sx_row.addWidget(self._sx_spin)
        sx_row.addWidget(self._sx_slider)
        form.addRow("幅", sx_row)

        # 縦スケール
        sy_row = QHBoxLayout()
        self._sy_spin = QSpinBox()
        self._sy_spin.setRange(1, 1000)
        self._sy_spin.setValue(100)
        self._sy_spin.setSuffix(" %")
        self._sy_spin.setFixedWidth(80)
        self._sy_slider = QSlider(Qt.Orientation.Horizontal)
        self._sy_slider.setRange(1, 1000)
        self._sy_slider.setValue(100)
        sy_row.addWidget(self._sy_spin)
        sy_row.addWidget(self._sy_slider)
        form.addRow("高さ", sy_row)

        # 縦横比を固定チェック
        self._lock = QRadioButton("縦横比を固定")
        self._lock.setChecked(True)
        form.addRow(self._lock)

        form.addRow(_HSeparator())

        # 回転
        rot_row = QHBoxLayout()
        self._rot_spin = QSpinBox()
        self._rot_spin.setRange(-180, 180)
        self._rot_spin.setValue(0)
        self._rot_spin.setSuffix(" °")
        self._rot_spin.setFixedWidth(80)
        self._rot_slider = QSlider(Qt.Orientation.Horizontal)
        self._rot_slider.setRange(-180, 180)
        self._rot_slider.setValue(0)
        rot_row.addWidget(self._rot_spin)
        rot_row.addWidget(self._rot_slider)
        form.addRow("回転", rot_row)

        layout.addLayout(form)

        # ボタン
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self._on_cancel)
        layout.addWidget(btn_box)

        # ── シグナル接続 ──
        self._sx_spin.valueChanged.connect(self._on_sx_spin)
        self._sx_slider.valueChanged.connect(self._on_sx_slider)
        self._sy_spin.valueChanged.connect(self._on_sy_spin)
        self._sy_slider.valueChanged.connect(self._on_sy_slider)
        self._rot_spin.valueChanged.connect(self._on_rot_spin)
        self._rot_slider.valueChanged.connect(self._on_rot_slider)

        self._updating = False  # スピン↔スライダー相互更新ループ防止

    # ── スピン↔スライダー同期 ────────────────────────────────────────────────

    def _on_sx_spin(self, v):
        if self._updating:
            return
        self._updating = True
        self._sx_slider.setValue(v)
        if self._lock.isChecked():
            self._sy_spin.setValue(v)
            self._sy_slider.setValue(v)
        self._updating = False
        self._apply()

    def _on_sx_slider(self, v):
        if self._updating:
            return
        self._updating = True
        self._sx_spin.setValue(v)
        if self._lock.isChecked():
            self._sy_spin.setValue(v)
            self._sy_slider.setValue(v)
        self._updating = False
        self._apply()

    def _on_sy_spin(self, v):
        if self._updating:
            return
        self._updating = True
        self._sy_slider.setValue(v)
        self._updating = False
        self._apply()

    def _on_sy_slider(self, v):
        if self._updating:
            return
        self._updating = True
        self._sy_spin.setValue(v)
        self._updating = False
        self._apply()

    def _on_rot_spin(self, v):
        if self._updating:
            return
        self._updating = True
        self._rot_slider.setValue(v)
        self._updating = False
        self._apply()

    def _on_rot_slider(self, v):
        if self._updating:
            return
        self._updating = True
        self._rot_spin.setValue(v)
        self._updating = False
        self._apply()

    def _apply(self):
        self._canvas.apply_transform_percentage(
            self._sx_spin.value(),
            self._sy_spin.value(),
            self._rot_spin.value(),
        )

    def _on_ok(self):
        self._canvas._commit_transform()
        self.accept()

    def _on_cancel(self):
        self._canvas.cancel_transform()
        self.reject()

    def closeEvent(self, event):
        # ×ボタン / Esc でダイアログが閉じられたとき、変形を必ずキャンセルする
        if self._canvas._transform_image:
            self._canvas.cancel_transform()
        super().closeEvent(event)


def _HSeparator() -> QWidget:
    f = QWidget()
    f.setFixedHeight(1)
    f.setStyleSheet("background: #ccc;")
    return f


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._current_path: str | None = None  # 現在開いている .pola ファイルのパス
        self.setWindowTitle("PaintPola")
        icon_path = os.path.join(os.path.dirname(__file__), "images", "pola_block.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.layer_stack = LayerStack()
        self._init_canvas(CANVAS_W, CANVAS_H)

        self.canvas = Canvas(self.layer_stack)
        self.toolbar = Toolbar()
        self.layer_panel = LayerPanel(self.layer_stack)
        self.color_panel = ColorPanel()
        self.tool_options = ToolOptionsPanel()

        self._tool_cursors = make_tool_cursors()

        self.anim_panel = AnimationPanel()
        self.anim_panel.setVisible(False)
        self.anim_panel.set_composite_fn(lambda: self.layer_stack.composite())
        self.anim_panel.onion_skin_changed.connect(self.canvas.update)
        self.canvas._get_onion_images = self.anim_panel.get_onion_images
        self._anim_mode = False

        self._connect_signals()
        self._build_layout()
        # canvas にスクロールエリアを注入（パンニング用）
        self.canvas._scroll_area = self._scroll
        self.canvas._on_structure_restored = self._on_structure_restored
        self.layer_panel._action_panel.canvas = self.canvas
        self._build_menus()
        self._connect_navigator()

        status = QStatusBar()
        self.setStatusBar(status)
        self._status_coord = QLabel("x:0  y:0")
        self._status_zoom = QLabel("100%")
        self._status_canvas_size = QLabel(f"{self.layer_stack.width} x {self.layer_stack.height} px")
        self._status_coord.setFixedWidth(140)
        self._status_zoom.setFixedWidth(80)
        self._status_canvas_size.setFixedWidth(180)
        status.addWidget(self._status_coord)
        status.addWidget(self._status_zoom)
        status.addPermanentWidget(self._status_canvas_size)
        self.canvas.status_message.connect(self._on_status_message)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)

        self.resize(1200, 800)

    def _init_canvas(self, w: int, h: int):
        self.layer_stack.layers.clear()
        self.layer_stack.width = w
        self.layer_stack.height = h
        self.layer_stack.active_path = [0]
        bg = Layer("背景", w, h)
        p = QPainter(bg.image)
        p.fillRect(0, 0, w, h, Qt.GlobalColor.white)
        p.end()
        self.layer_stack.layers.append(bg)

    def _connect_signals(self):
        self.toolbar.tool_changed.connect(self._on_tool_change)
        self.toolbar.color_changed.connect(self._on_color_change)
        self.toolbar.color_changed.connect(self.toolbar.push_color)
        # スポイトで拾った色 → ツールバー＋カラーパネル両方に反映
        # set_color → color_changed → push_color のルートがあるため push_color の直接接続は不要
        self.canvas.color_picked.connect(self.toolbar.set_color)
        self.canvas.color_picked.connect(self._on_color_change)
        self.canvas.color_picked.connect(self.color_panel.set_color)
        # カラーパネルからの色変更
        # HSV スライダーはドラッグ中に大量の中間値を emit するため、
        # ライブプレビュー（ペン色・スウォッチ反映）は color_changed で、
        # 履歴への登録はドラッグ確定時の color_committed でのみ行う
        # （そうしないと履歴が中間値で埋め尽くされてしまう）
        self.color_panel.color_changed.connect(self._on_color_change)
        self.color_panel.color_changed.connect(self.toolbar.set_color_preview)
        self.color_panel.color_committed.connect(self.toolbar.set_color)
        # ツールキーボードショートカット
        self.canvas.tool_shortcut_pressed.connect(self.toolbar.select_tool)
        self.toolbar.undo_requested.connect(self.canvas.undo)
        self.toolbar.redo_requested.connect(self.canvas.redo)
        self.layer_panel.layers_changed.connect(self.canvas.update)
        self.layer_panel.layer_structure_changed.connect(self.canvas.purge_orphan_history)
        self.layer_panel.structure_will_change.connect(self.canvas.save_structure_history)
        self.layer_panel.merge_down_requested.connect(self._merge_down)
        self.layer_panel.merge_all_requested.connect(self._merge_all_visible)
        self.layer_panel.merge_folder_requested.connect(self._merge_selected)
        # 数字キーで不透明度変更 → スライダー同期＋再描画
        self.canvas.layer_opacity_changed.connect(self.layer_panel.set_opacity)
        self.canvas.layer_opacity_changed.connect(lambda _: self.canvas.update())
        # ツールオプションパネル
        self.toolbar.options_toggled.connect(self._toggle_tool_options)
        self.tool_options.pen_size_changed.connect(self._on_pen_size_change)
        self.tool_options.eraser_size_changed.connect(self._on_eraser_size_change)
        self.tool_options.brush_changed.connect(self.canvas.set_brush)
        self.tool_options.symmetry_toggled.connect(
            lambda v: setattr(self.canvas, 'symmetry_enabled', v))
        self.tool_options.shape_fill_changed.connect(
            lambda v: setattr(self.canvas, 'shape_fill', v))
        self.tool_options.fill_expand_changed.connect(
            lambda v: setattr(self.canvas, 'fill_expand', v))
        self.tool_options.select_mode_changed.connect(
            lambda v: setattr(self.canvas, 'select_mode', v))
        self.tool_options.pivot_changed.connect(
            lambda ax, ay: setattr(self.canvas, '_transform_pivot', (ax, ay)))
        self.tool_options.pivot_mode_changed.connect(
            lambda m: setattr(self.canvas, '_pivot_mode', m))
        self.tool_options.transform_mode_changed.connect(
            lambda m: self.canvas.set_transform_mode(m))
        self.tool_options.mesh_div_changed.connect(
            lambda n: self.canvas.set_mesh_div(n))
        self.tool_options.blur_size_changed.connect(
            lambda v: setattr(self.canvas, 'blur_size', v))
        self.tool_options.blur_strength_changed.connect(
            lambda v: setattr(self.canvas, 'blur_strength', v / 100.0))
        # [ ] でペンサイズ変更 → オプションパネルも同期
        self.canvas.brush_size_changed.connect(self.tool_options.sync_pen_size)

    def _connect_navigator(self):
        self.canvas.repainted.connect(self.navigator.refresh)
        self.layer_panel.layers_changed.connect(self.navigator.refresh)
        self.navigator.refresh()

    def _on_tool_change(self, tool: Tool):
        prev = self.canvas.tool
        # Alt 一時スポイト中にツール切替されたら Alt モードを強制終了する
        if self.canvas._alt_eyedropper:
            self.canvas._alt_eyedropper = False
        # テキストツールから別ツールに切り替えたとき未使用の _text_pos を破棄する
        if prev == Tool.TEXT and tool != Tool.TEXT:
            self.canvas._text_pos = None
        # ペン/消しゴム等に切り替えたとき変形中なら確定する
        if (self.canvas._transform_image
                and tool not in (Tool.SELECT_RECT, Tool.LASSO, Tool.TRANSFORM)):
            self.canvas._commit_transform()
        # 自由変形ツールに切り替えたとき、選択範囲があれば自動的に持ち上げる
        if tool == Tool.TRANSFORM and self.canvas._selection_rect:
            layer = self.canvas.layer_stack.active
            if layer and not layer.is_group:
                self.canvas._lift_selection(layer)  # type: ignore
        self.canvas.tool = tool
        self.canvas._stabilizer.reset()
        self.canvas._cursor_widget_pos = None
        # ツールオプションパネルを更新
        self.tool_options.set_tool(
            tool,
            pen_size=self.canvas.pen_size,
            eraser_size=self.canvas.eraser_size,
            brush_key=self.canvas.brush_type,
            symmetry=self.canvas.symmetry_enabled,
            shape_fill=self.canvas.shape_fill,
            fill_expand=self.canvas.fill_expand,
            select_mode=self.canvas.select_mode,
        )
        # ツールに応じたカーソル
        self.canvas._tool_cursor = self._tool_cursors.get(tool)
        self.canvas._restore_tool_cursor()

    def _on_pen_size_change(self, v: int):
        self.canvas.pen_size = v
        self.canvas.update()

    def _on_eraser_size_change(self, v: int):
        self.canvas.eraser_size = v
        self.canvas.update()

    def _on_color_change(self, color: QColor):
        self.canvas.pen_color = color
        # カラーパネルのHSVスライダーを同期（色_panelからの変更は二重反映しない）
        if self.color_panel.current_color().rgba() != color.rgba():
            self.color_panel.set_color(color)

    def _build_layout(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 左: ツールバー ＋ スライドインオプションパネル
        left_area = QWidget()
        left_area.setFixedWidth(90)   # 初期はツールバー幅のみ
        left_layout = QHBoxLayout(left_area)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_layout.addWidget(self.toolbar)
        self.tool_options.setVisible(False)
        left_layout.addWidget(self.tool_options)
        self._left_area = left_area
        root.addWidget(left_area)

        # 中央: キャンバス + アニメーションパネル（下部）
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidget(self.canvas)
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setStyleSheet(
            "QScrollArea { background: #808080; border: none; }"
            "QScrollArea > QWidget > QWidget { background: #808080; }"
        )
        center_layout.addWidget(self._scroll, 1)
        center_layout.addWidget(self.anim_panel)

        root.addWidget(center, 1)

        # 右: ナビゲーター（上）＋ カラーパネル（中）＋ レイヤーパネル（下）
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setFixedWidth(260)
        right_splitter.setChildrenCollapsible(False)

        self.navigator = NavigatorPanel(self.layer_stack, self.canvas, self._scroll)
        right_splitter.addWidget(self.navigator)

        right_splitter.addWidget(self.color_panel)

        right_splitter.addWidget(self.layer_panel)

        right_splitter.setStretchFactor(0, 0)
        right_splitter.setStretchFactor(1, 0)
        right_splitter.setStretchFactor(2, 1)

        # ナビゲーター・カラーパネルを小さめに固定し、残りをレイヤーパネルに割り当てる
        right_splitter.setSizes([160, 260, 9999])

        root.addWidget(right_splitter)

    def _build_menus(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("ファイル")
        self._add_action(file_menu, "新規", self._new, "Ctrl+N")
        self._add_action(file_menu, "開く...", self._open, "Ctrl+O")
        file_menu.addSeparator()
        self._add_action(file_menu, "保存", self._save, "Ctrl+S")
        self._add_action(file_menu, "名前を付けて保存...", self._save_as_pola, "Ctrl+Shift+S")
        file_menu.addSeparator()
        self._add_action(file_menu, "PNG として書き出し...", self._export_png, "Ctrl+E")
        self._add_action(file_menu, "画像をレイヤーとして追加...", self._import_as_layer)
        file_menu.addSeparator()
        self._add_action(file_menu, "終了", self.close, "Ctrl+Q")

        edit_menu = mb.addMenu("編集")
        self._add_action(edit_menu, "元に戻す", self.canvas.undo, "Ctrl+Z")
        self._add_action(edit_menu, "やり直し", self.canvas.redo, "Ctrl+Y")
        edit_menu.addSeparator()
        self._add_action(edit_menu, "コピー", self.canvas.copy_selection, "Ctrl+C")
        self._add_action(edit_menu, "貼り付け", self.canvas.paste_selection, "Ctrl+V")
        self._add_action(edit_menu, "削除", self.canvas.delete_selection, "Delete")
        edit_menu.addSeparator()
        self._add_action(edit_menu, "すべて選択", self.canvas.select_all, "Ctrl+A")
        self._add_action(edit_menu, "選択解除", self.canvas.deselect, "Escape")

        image_menu = mb.addMenu("画像")
        self._add_action(image_menu, "キャンバスサイズ変更...", self._resize_canvas)
        image_menu.addSeparator()
        self._add_action(image_menu, "下に統合", self._merge_down, "Ctrl+M")
        self._add_action(image_menu, "表示レイヤーを統合", self._merge_all_visible, "Ctrl+Shift+M")
        self._add_action(image_menu, "フォルダを結合", self._merge_folder)
        self._add_action(image_menu, "レイヤーをラスタライズ", self._rasterize_layer)
        image_menu.addSeparator()
        self._add_action(image_menu, "線画抽出...", self._extract_line)
        self._add_action(image_menu, "レイヤーを変形（拡大縮小・回転）...", self._transform_layer_dialog)

        filter_menu = mb.addMenu("フィルター")
        self._add_action(filter_menu, "ぼかし (ガウス)...", self._filter_blur)
        self._add_action(filter_menu, "ゴミ取り...", self._filter_despeckle)

        view_menu = mb.addMenu("表示")
        zoom_menu = view_menu.addMenu("ズーム")
        for label, zoom in [("25%", 0.25), ("50%", 0.5), ("100%", 1.0), ("150%", 1.5), ("200%", 2.0)]:
            self._add_action(zoom_menu, label, lambda _, z=zoom: self.canvas.set_zoom(z))

        view_menu.addSeparator()
        self._add_action(view_menu, "左に回転", self.canvas.rotate_ccw, "Ctrl+[")
        self._add_action(view_menu, "右に回転", self.canvas.rotate_cw, "Ctrl+]")
        self._add_action(view_menu, "回転をリセット", self.canvas.reset_rotation, "Ctrl+0")
        view_menu.addSeparator()
        self._add_action(view_menu, "左右反転表示", self.canvas.toggle_flip_h, "Ctrl+Shift+H")
        view_menu.addSeparator()
        self._add_action(view_menu, "グリッド表示切替", self.canvas.toggle_grid, "Ctrl+G")

        mode_menu = mb.addMenu("モード")
        self._anim_mode_action = QAction("アニメーションモード", self)
        self._anim_mode_action.setCheckable(True)
        self._anim_mode_action.setChecked(False)
        self._anim_mode_action.toggled.connect(self._toggle_anim_mode)
        mode_menu.addAction(self._anim_mode_action)

        transform_menu = mb.addMenu("変形")
        self._add_action(transform_menu, "変形を確定 (Enter)", self.canvas._commit_transform)
        self._add_action(transform_menu, "変形をキャンセル", self.canvas.cancel_transform)
        transform_menu.addSeparator()
        self._add_action(transform_menu, "拡大縮小・回転（%指定）...", self._open_transform_percent_dialog)

        design_menu = mb.addMenu("デザイン")
        self._theme_actions: dict[str, QAction] = {}
        for key in get_theme_keys():
            action = QAction(get_theme_label(key), self)
            action.setCheckable(True)
            action.setChecked(key == "default")
            action.triggered.connect(lambda checked, k=key: self._apply_theme(k))
            design_menu.addAction(action)
            self._theme_actions[key] = action

    def _add_action(self, menu: QMenu, label: str, slot, shortcut: str | None = None):
        action = QAction(label, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(slot)
        menu.addAction(action)

    def _new(self):
        # 保存済みパスがあるか、未保存でも履歴がある（編集済み）場合のみ確認
        has_changes = self._current_path is not None or bool(self.canvas._history)
        if has_changes:
            reply = QMessageBox.question(self, "新規", "現在の内容を破棄して新規作成しますか？")
            if reply != QMessageBox.StandardButton.Yes:
                return
        dialog = NewCanvasDialog(self.layer_stack.width, self.layer_stack.height, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        w, h = dialog.values()
        self.canvas.reset_state()
        self._init_canvas(w, h)
        self.canvas._history.clear()
        self.canvas._redo_stack.clear()
        self._current_path = None
        self._update_title()
        self.layer_panel.refresh()
        self.canvas._update_size()

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "開く", "",
            "すべて対応ファイル (*.pola *.png *.jpg *.jpeg *.bmp *.webp);;"
            "PaintPola プロジェクト (*.pola);;"
            "画像ファイル (*.png *.jpg *.jpeg *.bmp *.webp)")
        if not path:
            return
        if path.lower().endswith(".pola"):
            self._load_pola(path)
        else:
            self._load_image_as_canvas(path)

    def _load_image_as_canvas(self, path: str):
        img = QImage(path)
        if img.isNull():
            QMessageBox.warning(self, "エラー", "画像を読み込めませんでした。")
            return
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
        iw, ih = img.width(), img.height()
        self.canvas.reset_state()
        self.layer_stack.layers.clear()
        self.layer_stack.width = iw
        self.layer_stack.height = ih
        self.layer_stack.active_path = [0]
        bg = Layer("背景", iw, ih)
        p = QPainter(bg.image)
        p.drawImage(0, 0, img)
        p.end()
        self.layer_stack.layers.append(bg)
        self.canvas._history.clear()
        self.canvas._redo_stack.clear()
        self._current_path = None
        self._update_title()
        self.layer_panel.refresh()
        self.canvas._update_size()

    # ── .pola 保存 / 読み込み ────────────────────────────────────────────────

    def _save(self):
        if self._current_path:
            self._write_pola(self._current_path)
        else:
            self._save_as_pola()

    def _save_as_pola(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "名前を付けて保存", "untitled.pola",
            "PaintPola プロジェクト (*.pola)")
        if not path:
            return
        if not path.lower().endswith(".pola"):
            path += ".pola"
        self._write_pola(path)

    def _write_pola(self, path: str):
        """レイヤー構造を .pola（ZIP）形式で保存する。"""
        ls = self.layer_stack
        meta: dict = {
            "version": 1,
            "canvas_w": ls.width,
            "canvas_h": ls.height,
            "active_path": list(ls.active_path),
            "view_rotation": self.canvas._rotation,
            "view_flip_h": self.canvas._flip_h,
            "layers": [],
        }

        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                img_index = 0

                def _write_layer(lyr) -> dict:
                    nonlocal img_index
                    info: dict = {
                        "name": lyr.name,
                        "visible": lyr.visible,
                        "opacity": lyr.opacity,
                        "is_group": lyr.is_group,
                    }
                    if lyr.is_group:
                        info["collapsed"] = lyr.collapsed
                        info["clipping"] = lyr.clipping
                        info["reference"] = lyr.reference
                        info["children"] = [_write_layer(c) for c in lyr.children]
                    else:
                        info["clipping"] = lyr.clipping
                        info["reference"] = lyr.reference
                        info["blend_mode"] = lyr.blend_mode
                        info["offset_x"] = lyr.offset_x
                        info["offset_y"] = lyr.offset_y
                        info["border_enabled"] = lyr.border_enabled
                        info["border_size"] = lyr.border_size
                        info["border_color"] = [lyr.border_color.red(), lyr.border_color.green(),
                                                lyr.border_color.blue(), lyr.border_color.alpha()]
                        info["shadow_enabled"] = lyr.shadow_enabled
                        info["shadow_color"] = [lyr.shadow_color.red(), lyr.shadow_color.green(),
                                                lyr.shadow_color.blue(), lyr.shadow_color.alpha()]
                        info["shadow_offset_x"] = lyr.shadow_offset_x
                        info["shadow_offset_y"] = lyr.shadow_offset_y
                        info["shadow_blur"] = lyr.shadow_blur
                        info["shadow_strength"] = lyr.shadow_strength
                        info["glow_enabled"] = lyr.glow_enabled
                        info["glow_color"] = [lyr.glow_color.red(), lyr.glow_color.green(),
                                              lyr.glow_color.blue(), lyr.glow_color.alpha()]
                        info["glow_size"] = lyr.glow_size
                        info["glow_strength"] = lyr.glow_strength
                        info["blur_enabled"] = lyr.blur_enabled
                        info["blur_radius"] = lyr.blur_radius
                        info["blur_strength"] = lyr.blur_strength
                        info["hsl_enabled"] = lyr.hsl_enabled
                        info["hsl_hue"] = lyr.hsl_hue
                        info["hsl_saturation"] = lyr.hsl_saturation
                        info["hsl_lightness"] = lyr.hsl_lightness
                        buf = QByteArray()
                        buf_io = QBuffer(buf)
                        buf_io.open(QIODevice.OpenModeFlag.WriteOnly)
                        lyr.image.save(buf_io, "PNG")
                        buf_io.close()
                        fname = f"layer_{img_index}.png"
                        zf.writestr(fname, bytes(buf))
                        info["image"] = fname
                        img_index += 1
                    return info

                meta["layers"] = [_write_layer(lyr) for lyr in ls.layers]
                zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
        except Exception as e:
            QMessageBox.warning(self, "エラー", f"保存に失敗しました:\n{e}")
            return

        self._current_path = path
        self._update_title()

    @staticmethod
    def _clamp(val, lo, hi, default):
        """数値をクランプする。型が不正ならデフォルトを返す。"""
        if not isinstance(val, (int, float)):
            return default
        return max(lo, min(int(val), hi))

    @staticmethod
    def _safe_color(rgba_list, default: list[int]) -> QColor:
        """RGBA配列を安全にQColorに変換する。"""
        if not isinstance(rgba_list, list) or len(rgba_list) < 3:
            rgba_list = default
        r = max(0, min(255, int(rgba_list[0])))
        g = max(0, min(255, int(rgba_list[1])))
        b = max(0, min(255, int(rgba_list[2])))
        a = max(0, min(255, int(rgba_list[3]))) if len(rgba_list) >= 4 else 255
        return QColor(r, g, b, a)

    @staticmethod
    def _safe_name(name, max_len: int = 200) -> str:
        """レイヤー名をサニタイズする。"""
        if not isinstance(name, str):
            name = "レイヤー"
        return name[:max_len]

    def _load_pola(self, path: str):
        """`.pola` ファイルを読み込んでレイヤースタックを再構築する。"""
        MAX_CANVAS = 10000
        MAX_LAYERS = 500
        MAX_ZIP_SIZE = 500 * 1024 * 1024  # 500MB

        try:
            file_size = os.path.getsize(path)
            if file_size > MAX_ZIP_SIZE:
                QMessageBox.warning(self, "エラー",
                                    f"ファイルサイズが大きすぎます ({file_size // 1024 // 1024}MB)")
                return

            with zipfile.ZipFile(path, "r") as zf:
                total_uncompressed = sum(zi.file_size for zi in zf.infolist())
                if total_uncompressed > MAX_ZIP_SIZE * 2:
                    QMessageBox.warning(self, "エラー", "展開後のサイズが大きすぎます")
                    return

                meta_raw = zf.read("meta.json")
                if len(meta_raw) > 10 * 1024 * 1024:
                    QMessageBox.warning(self, "エラー", "meta.json が大きすぎます")
                    return
                meta = json.loads(meta_raw.decode("utf-8"))

                w = self._clamp(meta.get("canvas_w", CANVAS_W), 1, MAX_CANVAS, CANVAS_W)
                h = self._clamp(meta.get("canvas_h", CANVAS_H), 1, MAX_CANVAS, CANVAS_H)

                layer_count = [0]

                def _read_layer(info: dict) -> Layer | GroupLayer:
                    layer_count[0] += 1
                    if layer_count[0] > MAX_LAYERS:
                        raise ValueError(f"レイヤー数が上限 ({MAX_LAYERS}) を超えています")

                    if not isinstance(info, dict):
                        raise ValueError("レイヤー情報が不正です")

                    name = self._safe_name(info.get("name", "レイヤー"))

                    if info.get("is_group", False):
                        grp = GroupLayer(name, w, h)
                        grp.visible = bool(info.get("visible", True))
                        grp.opacity = self._clamp(info.get("opacity", 255), 0, 255, 255)
                        grp.collapsed = bool(info.get("collapsed", False))
                        grp.clipping = bool(info.get("clipping", False))
                        grp.reference = bool(info.get("reference", False))
                        children = info.get("children", [])
                        if isinstance(children, list):
                            grp.children = [_read_layer(c) for c in children]
                        return grp
                    else:
                        lyr = Layer(name, w, h)
                        lyr.visible = bool(info.get("visible", True))
                        lyr.opacity = self._clamp(info.get("opacity", 255), 0, 255, 255)
                        lyr.clipping = bool(info.get("clipping", False))
                        lyr.reference = bool(info.get("reference", False))
                        bm = info.get("blend_mode", "normal")
                        from layer import BLEND_KEYS
                        lyr.blend_mode = bm if bm in BLEND_KEYS else "normal"

                        lyr.offset_x = self._clamp(info.get("offset_x", 0), -20000, 20000, 0)
                        lyr.offset_y = self._clamp(info.get("offset_y", 0), -20000, 20000, 0)
                        lyr.border_enabled = bool(info.get("border_enabled", False))
                        lyr.border_size = self._clamp(info.get("border_size", 3), 0, 50, 3)
                        lyr.border_color = self._safe_color(
                            info.get("border_color", [0, 0, 0, 255]), [0, 0, 0, 255])

                        lyr.shadow_enabled = bool(info.get("shadow_enabled", False))
                        lyr.shadow_color = self._safe_color(
                            info.get("shadow_color", [0, 0, 0, 180]), [0, 0, 0, 180])
                        lyr.shadow_offset_x = self._clamp(info.get("shadow_offset_x", 4), -100, 100, 4)
                        lyr.shadow_offset_y = self._clamp(info.get("shadow_offset_y", 4), -100, 100, 4)
                        lyr.shadow_blur = self._clamp(info.get("shadow_blur", 5), 0, 100, 5)
                        lyr.shadow_strength = self._clamp(info.get("shadow_strength", 100), 0, 100, 100)

                        lyr.glow_enabled = bool(info.get("glow_enabled", False))
                        lyr.glow_color = self._safe_color(
                            info.get("glow_color", [255, 255, 200, 255]), [255, 255, 200, 255])
                        lyr.glow_size = self._clamp(info.get("glow_size", 8), 0, 100, 8)
                        lyr.glow_strength = self._clamp(info.get("glow_strength", 80), 0, 100, 80)

                        lyr.blur_enabled = bool(info.get("blur_enabled", False))
                        lyr.blur_radius = self._clamp(info.get("blur_radius", 3), 0, 100, 3)
                        lyr.blur_strength = self._clamp(info.get("blur_strength", 100), 0, 100, 100)

                        lyr.hsl_enabled = bool(info.get("hsl_enabled", False))
                        lyr.hsl_hue = self._clamp(info.get("hsl_hue", 0), -180, 180, 0)
                        lyr.hsl_saturation = self._clamp(info.get("hsl_saturation", 0), -100, 100, 0)
                        lyr.hsl_lightness = self._clamp(info.get("hsl_lightness", 0), -100, 100, 0)

                        # パストラバーサル防止: ファイル名のみ使用
                        img_name = info.get("image", "")
                        if not isinstance(img_name, str):
                            raise ValueError("画像ファイル名が不正です")
                        import pathlib
                        safe_name = pathlib.PurePosixPath(img_name).name
                        if not safe_name:
                            raise ValueError(f"画像ファイル名が空です: {img_name}")
                        img_data = zf.read(safe_name)
                        img = QImage.fromData(img_data, "PNG")
                        if img.isNull():
                            raise ValueError(f"画像の読み込みに失敗: {safe_name}")
                        lyr.image = img.convertToFormat(QImage.Format.Format_ARGB32)
                        return lyr

                raw_layers = meta.get("layers", [])
                if not isinstance(raw_layers, list):
                    raise ValueError("layers が配列ではありません")
                layers = [_read_layer(info) for info in raw_layers]
        except Exception as e:
            QMessageBox.warning(self, "エラー", f"ファイルを開けませんでした:\n{e}")
            return

        self.canvas.reset_state()
        ls = self.layer_stack
        ls.layers.clear()
        ls.width = w
        ls.height = h
        ls.layers.extend(layers)
        if "active_path" in meta:
            ap = meta["active_path"]
            if isinstance(ap, list) and ap and all(isinstance(x, int) for x in ap):
                ap = [max(0, x) for x in ap]
                ap[0] = min(ap[0], max(0, len(layers) - 1))
                ls.active_path = ap
            else:
                ls.active_path = [0]
        else:
            ai = min(meta.get("active_index", 0), max(0, len(layers) - 1))
            aci = meta.get("active_child_index", -1)
            ls.active_path = [ai, aci] if aci >= 0 else [ai]
        self.canvas._history.clear()
        self.canvas._redo_stack.clear()
        self.canvas._rotation = meta.get("view_rotation", 0)
        self.canvas._flip_h = meta.get("view_flip_h", False)
        self._current_path = path
        self._update_title()
        self.layer_panel.refresh()
        self.canvas._update_size()
        self._update_canvas_size_label()

    # ── PNG 書き出し ─────────────────────────────────────────────────────────

    def _export_png(self):
        composite = self.layer_stack.composite()
        path, _ = QFileDialog.getSaveFileName(
            self, "PNG として書き出し", "untitled.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)")
        if not path:
            return
        fmt = "PNG"
        if path.lower().endswith((".jpg", ".jpeg")):
            fmt = "JPEG"
        elif path.lower().endswith(".bmp"):
            fmt = "BMP"
        if fmt != "PNG":
            opaque = QImage(composite.size(), QImage.Format.Format_ARGB32)
            opaque.fill(Qt.GlobalColor.white)
            op = QPainter(opaque)
            op.drawImage(0, 0, composite)
            op.end()
            composite = opaque
        if not composite.save(path, fmt):
            QMessageBox.warning(self, "エラー", "書き出しに失敗しました。")

    def _on_structure_restored(self):
        self.layer_panel.refresh()
        self.navigator.refresh()

    def _apply_theme(self, key: str):
        qss = get_theme_qss(key)
        QApplication.instance().setStyleSheet(qss)
        for k, action in self._theme_actions.items():
            action.setChecked(k == key)
        self._current_theme = key

    def _toggle_anim_mode(self, enabled: bool):
        self._anim_mode = enabled
        self.anim_panel.setVisible(enabled)
        if not enabled:
            self.anim_panel._stop_play()
        self.canvas.update()

    def _on_status_message(self, msg: str):
        self._status_coord.setText(msg)

    def _on_zoom_changed(self, zoom: float):
        self._status_zoom.setText(f"{zoom * 100:.0f}%")

    def _update_canvas_size_label(self):
        self._status_canvas_size.setText(
            f"{self.layer_stack.width} x {self.layer_stack.height} px")

    def _update_title(self):
        if self._current_path:
            name = os.path.basename(self._current_path)
            self.setWindowTitle(f"PaintPola — {name}")
        else:
            self.setWindowTitle("PaintPola")

    def _resize_canvas(self):
        composite = self.layer_stack.composite()
        dialog = ResizeCanvasDialog(self.layer_stack.width, self.layer_stack.height,
                                     self, composite)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_w, new_h, mode, anchor = dialog.values()
        if new_w == self.layer_stack.width and new_h == self.layer_stack.height:
            return

        old_w, old_h = self.layer_stack.width, self.layer_stack.height
        ax, ay = anchor

        def _calc_offset(old_dim: int, new_dim: int, a: int) -> int:
            if a == 0:
                return 0
            elif a == 1:
                return (new_dim - old_dim) // 2
            else:
                return new_dim - old_dim

        offset_x = _calc_offset(old_w, new_w, ax)
        offset_y = _calc_offset(old_h, new_h, ay)

        def _resize_layer_image(lyr, nw: int, nh: int, scale_mode: bool):
            new_img = QImage(nw, nh, QImage.Format.Format_ARGB32_Premultiplied)
            new_img.fill(Qt.GlobalColor.transparent)
            p = QPainter(new_img)
            if scale_mode:
                scaled = lyr.image.scaled(
                    nw, nh,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                p.drawImage(0, 0, scaled)
            else:
                p.drawImage(offset_x + getattr(lyr, 'offset_x', 0),
                            offset_y + getattr(lyr, 'offset_y', 0), lyr.image)
            p.end()
            lyr.image = new_img.convertToFormat(QImage.Format.Format_ARGB32)
            if hasattr(lyr, 'offset_x'):
                lyr.offset_x = 0
                lyr.offset_y = 0

        for layer in self.layer_stack.layers:
            if layer.is_group:
                layer._w = new_w  # type: ignore
                layer._h = new_h  # type: ignore
                for child in layer.children:  # type: ignore
                    if not child.is_group:
                        _resize_layer_image(child, new_w, new_h, mode == "scale")
            else:
                _resize_layer_image(layer, new_w, new_h, mode == "scale")

        self.layer_stack.width = new_w
        self.layer_stack.height = new_h
        self.canvas._update_size()
        self.layer_panel.refresh()
        self._update_canvas_size_label()

    def _import_as_layer(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "画像をレイヤーとして追加", "",
            "画像ファイル (*.png *.jpg *.jpeg *.bmp *.webp *.gif *.tiff *.tif)")
        if not paths:
            return
        cw, ch = self.layer_stack.width, self.layer_stack.height
        for path in paths:
            img = QImage(path)
            if img.isNull():
                QMessageBox.warning(self, "エラー", f"読み込めませんでした:\n{path}")
                continue
            img = img.convertToFormat(QImage.Format.Format_ARGB32)
            # キャンバスより大きい場合は縮小、小さい場合はそのまま左上に配置
            if img.width() > cw or img.height() > ch:
                img = img.scaled(cw, ch,
                                 Qt.AspectRatioMode.KeepAspectRatio,
                                 Qt.TransformationMode.SmoothTransformation)
            canvas_img = QImage(cw, ch, QImage.Format.Format_ARGB32_Premultiplied)
            canvas_img.fill(Qt.GlobalColor.transparent)
            p = QPainter(canvas_img)
            p.drawImage(0, 0, img)
            p.end()
            canvas_img = canvas_img.convertToFormat(QImage.Format.Format_ARGB32)

            name = os.path.splitext(os.path.basename(path))[0]
            layer = self.layer_stack.add(name)
            layer.image = canvas_img

        self.layer_panel.refresh()
        self.canvas.update()
        self.navigator.refresh()

    def _extract_line(self):
        composite = self.layer_stack.composite()
        dialog = LineExtractionDialog(composite, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        result = dialog.extract()
        line_layer = self.layer_stack.add("線画")
        line_layer.image = result
        self.layer_panel.refresh()
        self.canvas.update()
        self.navigator.refresh()

    def _transform_layer_dialog(self):
        """画像メニュー: レイヤー全体をliftしてからTransformPercentDialogを開く。"""
        layer = self.layer_stack.active
        if not layer or layer.is_group:
            QMessageBox.warning(self, "変形", "通常レイヤーを選択してください。")
            return
        if not self.canvas.lift_whole_layer():
            return
        dlg = TransformPercentDialog(self.canvas, self)
        dlg.exec()
        self.layer_panel.refresh()
        self.canvas.update()
        self.navigator.refresh()

    def _open_transform_percent_dialog(self):
        """変形メニュー: 既にlift済みの変形状態に%ゲージを開く。lift済みでなければ先にliftする。"""
        if not self.canvas._transform_image:
            layer = self.layer_stack.active
            if not layer or layer.is_group:
                QMessageBox.warning(self, "変形", "通常レイヤーを選択してください。")
                return
            if not self.canvas.lift_whole_layer():
                return
        dlg = TransformPercentDialog(self.canvas, self)
        dlg.exec()
        self.layer_panel.refresh()
        self.canvas.update()
        self.navigator.refresh()

    def _toggle_tool_options(self, visible: bool):
        self.tool_options.setVisible(visible)
        w = 90 + (220 if visible else 0)
        self._left_area.setFixedWidth(w)

    def _merge_down(self):
        self.canvas.reset_state()
        self.canvas.save_structure_history()
        if self.layer_stack.merge_down():
            self.layer_panel.refresh()
            self.canvas.update()
            self.navigator.refresh()
        else:
            self._history_pop_last_structure()
            self.statusBar().showMessage("下に統合できません（グループと通常レイヤーは統合不可）", 3000)

    def _merge_all_visible(self):
        self.canvas.reset_state()
        self.canvas.save_structure_history()
        if self.layer_stack.merge_all_visible():
            self.layer_panel.refresh()
            self.canvas.update()
            self.navigator.refresh()

    def _merge_selected(self):
        """選択されたレイヤーのみを結合（レイヤーパネルから呼ばれる）。"""
        # 現在選択中のレイヤー（ネストしたフォルダの場合も、実際に選択されている
        # そのフォルダ自身を対象にする。active_top は常にトップレベルの祖先を返す
        # ため、ネストしたフォルダを選択した場合に外側ごと統合されてしまうバグがあった）
        active = self.layer_stack.active
        path = self.layer_stack.active_path
        if active and active.is_group and active.children and path:  # type: ignore
            self.canvas.reset_state()
            self.canvas.save_structure_history()
            grp = active
            container, parent_path = self.layer_stack.parent_of(path)
            idx = path[-1]
            # グループ内のレイヤーの全範囲を計算して統合（このフォルダの子のみを対象とする）
            visible_children = [c for c in grp.children if c.visible]
            if visible_children:
                min_x, min_y, mw, mh = self.layer_stack._folder_bounds(visible_children)
            else:
                min_x, min_y = 0, 0
                mw, mh = self.layer_stack.width, self.layer_stack.height
            merged = QImage(mw, mh, QImage.Format.Format_ARGB32_Premultiplied)
            merged.fill(Qt.GlobalColor.transparent)
            mp = QPainter(merged)
            for child in reversed(grp.children):
                if child.visible:
                    self.layer_stack._draw_layer_to(mp, child, min_x, min_y)
            mp.end()
            new_layer = Layer(grp.name, mw, mh)
            new_layer.image = merged.convertToFormat(QImage.Format.Format_ARGB32)
            new_layer.offset_x = min_x
            new_layer.offset_y = min_y
            new_layer.opacity = grp.opacity
            new_layer.visible = grp.visible
            container[idx] = new_layer
            self.layer_stack.active_path = parent_path + [idx]
            self.layer_panel.refresh()
            self.canvas.update()
            self.navigator.refresh()
        else:
            self.statusBar().showMessage("グループを選択してください", 3000)

    def _merge_folder(self):
        self._merge_selected()

    def _rasterize_layer(self):
        """レイヤーをラスタライズ: 縁取り・ドロップシャドウ等の効果を画像に焼き込み、
        効果設定を無効化する。以後の統合で効果が消えたり二重適用されたりしなくなる。"""
        layer = self.canvas.layer_stack.active
        if not layer or layer.is_group:
            self.statusBar().showMessage("通常レイヤーを選択してください", 3000)
            return
        self.canvas._save_history()
        layer.rasterize()
        self.layer_panel.refresh()
        self.canvas.update()
        self.navigator.refresh()

    def _filter_blur(self):
        """フィルター → ぼかし (ガウス): レイヤー全体または選択範囲にガウスぼかしをかける。"""
        import cv2
        import numpy as np

        layer = self.canvas.layer_stack.active
        if not layer or layer.is_group:
            return
        img: QImage = layer.image

        radius, ok = QInputDialog.getInt(
            self, "ぼかし (ガウス)", "ぼかし半径 (px):", 5, 1, 200)
        if not ok:
            return

        self.canvas._save_history()

        k = radius * 2 + 1
        bits = img.bits()
        bits.setsize(img.sizeInBytes())
        arr = np.frombuffer(bits, dtype=np.uint8).reshape(img.height(), img.width(), 4).copy()

        sel = self.canvas._selection_rect
        mask = self.canvas._lasso_mask

        if mask:
            # 投げなわ選択: マスク領域のみぼかす
            m_bits = mask.bits()
            m_bits.setsize(mask.sizeInBytes())
            m_arr = np.frombuffer(m_bits, dtype=np.uint8).reshape(mask.height(), mask.width(), 4)
            alpha_mask = m_arr[:, :, 3] > 0
            blurred = cv2.GaussianBlur(arr, (k, k), 0)
            arr[alpha_mask] = blurred[alpha_mask]
        elif sel:
            # 矩形選択: 選択範囲のみぼかす
            x0 = max(0, sel.left())
            y0 = max(0, sel.top())
            x1 = min(img.width(), sel.right() + 1)
            y1 = min(img.height(), sel.bottom() + 1)
            region = arr[y0:y1, x0:x1].copy()
            blurred = cv2.GaussianBlur(region, (k, k), 0)
            arr[y0:y1, x0:x1] = blurred
        else:
            # 選択なし: レイヤー全体
            arr[:] = cv2.GaussianBlur(arr, (k, k), 0)

        result = QImage(arr.data, img.width(), img.height(), arr.strides[0],
                        QImage.Format.Format_ARGB32).copy()
        p = QPainter(img)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.drawImage(0, 0, result)
        p.end()
        self.canvas.update()

    def _filter_despeckle(self):
        """フィルター → ゴミ取り: 選択レイヤー上の孤立した小さな塊（面積 px² 以下）を透明化する。"""
        layer = self.canvas.layer_stack.active
        if not layer or layer.is_group:
            QMessageBox.warning(self, "ゴミ取り", "通常レイヤーを選択してください。")
            return

        area_mask = None
        sel = self.canvas._selection_rect
        mask = self.canvas._lasso_mask
        w, h = layer.image.width(), layer.image.height()
        if mask is not None:
            m_bits = mask.bits()
            m_bits.setsize(mask.sizeInBytes())
            m_arr = np.frombuffer(m_bits, dtype=np.uint8).reshape(mask.height(), mask.width(), 4)
            area_mask = (m_arr[:, :, 3] > 0).astype(np.uint8)
        elif sel:
            area_mask = np.zeros((h, w), dtype=np.uint8)
            x0 = max(0, sel.left())
            y0 = max(0, sel.top())
            x1 = min(w, sel.right() + 1)
            y1 = min(h, sel.bottom() + 1)
            area_mask[y0:y1, x0:x1] = 1

        self.canvas._save_history()
        dlg = DespeckleDialog(self.canvas, layer, area_mask, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            # キャンセル時は _save_history() で積んだ undo エントリを巻き戻す
            if self.canvas._history and self.canvas._history[-1][0] == "pixel":
                self.canvas._history.pop()
            return
        self.layer_panel.refresh()
        self.canvas.update()
        self.navigator.refresh()

    def _history_pop_last_structure(self):
        if self.canvas._history and self.canvas._history[-1][0] == "structure":
            self.canvas._history.pop()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("PaintPola")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
