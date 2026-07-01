"""アニメーションタイムラインパネル — パラパラ漫画風フレームアニメーション"""
from __future__ import annotations

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                              QLabel, QSpinBox, QScrollArea, QFileDialog,
                              QFrame, QSizePolicy, QMessageBox)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt6.QtCore import Qt, QTimer, QSize, pyqtSignal

_PREVIEW_SIZE = 320
_THUMB_H = 64
_THUMB_W = 80


class FrameThumb(QWidget):
    """フレームのサムネイル1枚分。"""
    clicked = pyqtSignal(int)

    def __init__(self, index: int, image: QImage, parent=None):
        super().__init__(parent)
        self.index = index
        self._image = image
        self._selected = False
        self.setFixedSize(_THUMB_W + 8, _THUMB_H + 24)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_selected(self, val: bool):
        self._selected = val
        self.update()

    def set_image(self, img: QImage):
        self._image = img
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self._selected:
            p.fillRect(self.rect(), QColor(80, 140, 255, 60))
            p.setPen(QColor(80, 140, 255))
            p.drawRect(0, 0, self.width() - 1, self.height() - 1)
        x = (self.width() - _THUMB_W) // 2
        scaled = self._image.scaled(_THUMB_W, _THUMB_H,
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        y = 2 + (_THUMB_H - scaled.height()) // 2
        # チェッカーボード背景
        checker = QImage(scaled.width(), scaled.height(), QImage.Format.Format_ARGB32)
        cp = QPainter(checker)
        for cy in range(0, scaled.height(), 8):
            for cx in range(0, scaled.width(), 8):
                color = QColor(200, 200, 200) if (cx // 8 + cy // 8) % 2 == 0 else QColor(255, 255, 255)
                cp.fillRect(cx, cy, 8, 8, color)
        cp.drawImage(0, 0, scaled)
        cp.end()
        p.drawImage(x, y, checker)
        p.setPen(QColor(200, 200, 200))
        label = str(self.index + 1)
        p.drawText(self.rect().adjusted(0, _THUMB_H + 4, 0, 0),
                   Qt.AlignmentFlag.AlignHCenter, label)
        p.end()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)


class PlaybackPreview(QWidget):
    """再生プレビューウィンドウ（フローティング）。"""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("再生プレビュー")
        self.setFixedSize(_PREVIEW_SIZE + 20, _PREVIEW_SIZE + 40)
        self._image: QImage | None = None
        self._frame_label = QLabel("", self)
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._canvas = QWidget()
        self._canvas.setFixedSize(_PREVIEW_SIZE, _PREVIEW_SIZE)
        layout.addWidget(self._canvas)
        layout.addWidget(self._frame_label)

    def set_frame(self, img: QImage, index: int, total: int):
        self._image = img
        self._frame_label.setText(f"{index + 1} / {total}")
        self._canvas.update()
        self.update()

    def paintEvent(self, event):
        if self._image is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # チェッカーボード背景
        ox, oy = 8, 8
        for cy in range(0, _PREVIEW_SIZE, 8):
            for cx in range(0, _PREVIEW_SIZE, 8):
                color = QColor(200, 200, 200) if (cx // 8 + cy // 8) % 2 == 0 else QColor(255, 255, 255)
                p.fillRect(ox + cx, oy + cy, 8, 8, color)
        scaled = self._image.scaled(_PREVIEW_SIZE, _PREVIEW_SIZE,
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        dx = ox + (_PREVIEW_SIZE - scaled.width()) // 2
        dy = oy + (_PREVIEW_SIZE - scaled.height()) // 2
        p.drawImage(dx, dy, scaled)
        p.end()


class AnimationPanel(QWidget):
    """アニメーションタイムラインパネル。"""
    onion_skin_changed = pyqtSignal()  # オニオンスキン状態が変わった

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames: list[QImage] = []
        self.current_frame: int = -1
        self._playing = False
        self._onion_enabled = True
        self._onion_count = 2
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_next)

        self._preview_win: PlaybackPreview | None = None

        self.setFixedHeight(130)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
        root.setSpacing(2)

        # 上段: ボタン群
        top = QHBoxLayout()
        top.setSpacing(4)

        self._add_btn = QPushButton("＋ フレーム追加")
        self._add_btn.setToolTip("現在のキャンバスを統合してフレームに追加")
        self._add_btn.clicked.connect(self._on_add_frame)
        top.addWidget(self._add_btn)

        self._replace_btn = QPushButton("差し替え")
        self._replace_btn.setToolTip("選択中のフレームを現在のキャンバスで上書き")
        self._replace_btn.clicked.connect(self._on_replace_frame)
        top.addWidget(self._replace_btn)

        self._del_btn = QPushButton("削除")
        self._del_btn.setToolTip("選択中のフレームを削除")
        self._del_btn.clicked.connect(self._on_delete_frame)
        top.addWidget(self._del_btn)

        self._move_left_btn = QPushButton("◀")
        self._move_left_btn.setFixedWidth(28)
        self._move_left_btn.setToolTip("フレームを左へ移動")
        self._move_left_btn.clicked.connect(self._on_move_left)
        top.addWidget(self._move_left_btn)

        self._move_right_btn = QPushButton("▶")
        self._move_right_btn.setFixedWidth(28)
        self._move_right_btn.setToolTip("フレームを右へ移動")
        self._move_right_btn.clicked.connect(self._on_move_right)
        top.addWidget(self._move_right_btn)

        top.addWidget(self._sep_v())

        self._play_btn = QPushButton("▶ 再生")
        self._play_btn.clicked.connect(self._on_play_toggle)
        top.addWidget(self._play_btn)

        top.addWidget(QLabel("fps:"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(8)
        self._fps_spin.setFixedWidth(50)
        self._fps_spin.valueChanged.connect(self._on_fps_changed)
        top.addWidget(self._fps_spin)

        top.addWidget(self._sep_v())

        self._onion_check = QPushButton("🧅 オニオン")
        self._onion_check.setCheckable(True)
        self._onion_check.setChecked(True)
        self._onion_check.setToolTip("オニオンスキン表示切替")
        self._onion_check.toggled.connect(self._on_onion_toggle)
        top.addWidget(self._onion_check)

        self._onion_spin = QSpinBox()
        self._onion_spin.setRange(1, 5)
        self._onion_spin.setValue(2)
        self._onion_spin.setFixedWidth(40)
        self._onion_spin.setToolTip("前後何枚表示するか")
        self._onion_spin.valueChanged.connect(self._on_onion_count)
        top.addWidget(self._onion_spin)

        top.addWidget(self._sep_v())

        self._preview_btn = QPushButton("プレビュー窓")
        self._preview_btn.setCheckable(True)
        self._preview_btn.setToolTip("再生プレビューウィンドウを表示")
        self._preview_btn.toggled.connect(self._on_preview_toggle)
        top.addWidget(self._preview_btn)

        top.addWidget(self._sep_v())

        self._export_btn = QPushButton("GIF書き出し")
        self._export_btn.clicked.connect(self._on_export_gif)
        top.addWidget(self._export_btn)

        top.addStretch()

        frame_label = QLabel()
        self._frame_label = frame_label
        top.addWidget(frame_label)

        root.addLayout(top)

        # 下段: フレームサムネイル一覧（横スクロール）
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setFixedHeight(90)

        self._thumb_container = QWidget()
        self._thumb_layout = QHBoxLayout(self._thumb_container)
        self._thumb_layout.setContentsMargins(4, 2, 4, 2)
        self._thumb_layout.setSpacing(4)
        self._thumb_layout.addStretch()
        self._scroll.setWidget(self._thumb_container)

        root.addWidget(self._scroll)
        self._thumbs: list[FrameThumb] = []
        self._update_label()

    @staticmethod
    def _sep_v() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.VLine)
        f.setFixedWidth(2)
        return f

    # ── 外部から呼ぶAPI ──

    _get_composite = None  # MainWindow が設定する: () -> QImage

    def set_composite_fn(self, fn):
        self._get_composite = fn

    def get_onion_images(self) -> list[tuple[QImage, float]]:
        """オニオンスキンで表示すべき画像と不透明度のリスト。"""
        if not self._onion_enabled or self.current_frame < 0:
            return []
        result = []
        for offset in range(1, self._onion_count + 1):
            idx = self.current_frame - offset
            if 0 <= idx < len(self.frames):
                opacity = 0.3 / offset
                result.append((self.frames[idx], opacity))
        return result

    def clear(self):
        self.frames.clear()
        self.current_frame = -1
        self._stop_play()
        self._rebuild_thumbs()

    # ── 内部 ──

    def _on_add_frame(self):
        if self._get_composite is None:
            return
        img = self._get_composite()
        if self.current_frame >= 0 and self.current_frame < len(self.frames) - 1:
            self.frames.insert(self.current_frame + 1, img.copy())
            self.current_frame += 1
        else:
            self.frames.append(img.copy())
            self.current_frame = len(self.frames) - 1
        self._rebuild_thumbs()
        self.onion_skin_changed.emit()

    def _on_replace_frame(self):
        if self._get_composite is None or self.current_frame < 0:
            return
        self.frames[self.current_frame] = self._get_composite().copy()
        self._thumbs[self.current_frame].set_image(self.frames[self.current_frame])
        self.onion_skin_changed.emit()

    def _on_delete_frame(self):
        if self.current_frame < 0 or not self.frames:
            return
        self.frames.pop(self.current_frame)
        if not self.frames:
            self.current_frame = -1
        elif self.current_frame >= len(self.frames):
            self.current_frame = len(self.frames) - 1
        self._rebuild_thumbs()
        self.onion_skin_changed.emit()

    def _on_move_left(self):
        i = self.current_frame
        if i <= 0:
            return
        self.frames[i], self.frames[i - 1] = self.frames[i - 1], self.frames[i]
        self.current_frame = i - 1
        self._rebuild_thumbs()

    def _on_move_right(self):
        i = self.current_frame
        if i < 0 or i >= len(self.frames) - 1:
            return
        self.frames[i], self.frames[i + 1] = self.frames[i + 1], self.frames[i]
        self.current_frame = i + 1
        self._rebuild_thumbs()

    def _on_frame_clicked(self, index: int):
        self.current_frame = index
        for t in self._thumbs:
            t.set_selected(t.index == index)
        self._update_label()
        self._update_preview()
        self.onion_skin_changed.emit()

    def _on_play_toggle(self):
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _start_play(self):
        if len(self.frames) < 2:
            return
        self._playing = True
        self._play_btn.setText("⏹ 停止")
        fps = self._fps_spin.value()
        self._play_timer.start(1000 // fps)

    def _stop_play(self):
        self._playing = False
        self._play_btn.setText("▶ 再生")
        self._play_timer.stop()

    def _play_next(self):
        n = len(self.frames)
        if n == 0:
            self._stop_play()
            return
        self.current_frame = (self.current_frame + 1) % n
        for t in self._thumbs:
            t.set_selected(t.index == self.current_frame)
        self._update_label()
        self._update_preview()
        self.onion_skin_changed.emit()

    def _on_fps_changed(self, val: int):
        if self._playing and val > 0:
            self._play_timer.start(1000 // val)

    def _on_preview_toggle(self, val: bool):
        if val:
            if self._preview_win is None:
                self._preview_win = PlaybackPreview()
            self._preview_win.show()
            self._update_preview()
        else:
            if self._preview_win:
                self._preview_win.hide()

    def _update_preview(self):
        if self._preview_win and self._preview_win.isVisible() and self.frames and self.current_frame >= 0:
            self._preview_win.set_frame(
                self.frames[self.current_frame],
                self.current_frame,
                len(self.frames))

    def _on_onion_toggle(self, val: bool):
        self._onion_enabled = val
        self.onion_skin_changed.emit()

    def _on_onion_count(self, val: int):
        self._onion_count = val
        self.onion_skin_changed.emit()

    def _on_export_gif(self):
        if not self.frames:
            QMessageBox.information(self, "GIF書き出し", "フレームがありません。")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "GIF書き出し", "", "GIF (*.gif)")
        if not path:
            return
        if not path.lower().endswith(".gif"):
            path += ".gif"
        self._export_gif_to(path)

    def _export_gif_to(self, path: str):
        try:
            from PIL import Image
        except ImportError:
            QMessageBox.warning(self, "エラー",
                                "GIF書き出しには Pillow が必要です。\npip install Pillow")
            return
        pil_frames: list[Image.Image] = []
        for qimg in self.frames:
            img = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
            w, h = img.width(), img.height()
            ptr = img.bits()
            ptr.setsize(h * w * 4)
            pil_img = Image.frombytes("RGBA", (w, h), bytes(ptr))
            pil_frames.append(pil_img)

        fps = self._fps_spin.value()
        duration = 1000 // fps

        pil_frames[0].save(
            path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            disposal=2,
        )
        QMessageBox.information(self, "GIF書き出し",
                                f"保存しました: {path}\n"
                                f"{len(self.frames)} フレーム / {fps} fps")

    def _rebuild_thumbs(self):
        for t in self._thumbs:
            t.deleteLater()
        self._thumbs.clear()
        # stretch を除去
        while self._thumb_layout.count():
            item = self._thumb_layout.takeAt(0)
            if item and item.widget():
                pass  # already deleteLater'd above
        for i, img in enumerate(self.frames):
            t = FrameThumb(i, img, self._thumb_container)
            t.clicked.connect(self._on_frame_clicked)
            t.set_selected(i == self.current_frame)
            self._thumb_layout.addWidget(t)
            self._thumbs.append(t)
        self._thumb_layout.addStretch()
        self._update_label()

    def _update_label(self):
        total = len(self.frames)
        cur = self.current_frame + 1 if self.current_frame >= 0 else 0
        self._frame_label.setText(f"フレーム: {cur} / {total}")
