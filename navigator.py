from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea
from PyQt6.QtGui import QPainter, QColor, QPen, QImage, QTransform
from PyQt6.QtCore import Qt, QRect, QRectF, QPointF, QSize, pyqtSignal

from layer import LayerStack

_NAV_SIZE = 150
_FRAME_COLOR = QColor(220, 60, 60, 200)
_FRAME_FILL = QColor(220, 60, 60, 30)
_BG_COLOR = QColor(50, 50, 50)


class NavigatorView(QWidget):
    scroll_requested = pyqtSignal(float, float)  # 正規化座標 (0-1) の左上

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(_NAV_SIZE, _NAV_SIZE)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._preview: QImage | None = None
        # _view_rect: プレビュー画像座標系での 0-1 正規化矩形
        self._view_rect = QRectF(0, 0, 1, 1)
        self._dragging = False
        self._drag_start: QPointF | None = None
        self._drag_rect_start: QRectF | None = None

    # ── public ───────────────────────────────────────────────────────────────

    def update_preview(self, preview: QImage):
        self._preview = preview
        self.update()

    def set_view_rect(self, rect: QRectF):
        # 幅・高さが 0 以下にならないよう防御
        w = max(0.0, min(1.0, rect.width()))
        h = max(0.0, min(1.0, rect.height()))
        x = max(0.0, min(1.0 - w, rect.x()))
        y = max(0.0, min(1.0 - h, rect.y()))
        self._view_rect = QRectF(x, y, w, h)
        self.update()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _preview_rect(self) -> QRect:
        """プレビュー画像がウィジェット内に配置される領域（中央寄せ）。"""
        if not self._preview or self._preview.isNull():
            return QRect(0, 0, _NAV_SIZE, _NAV_SIZE)
        pw, ph = self._preview.width(), self._preview.height()
        return QRect((_NAV_SIZE - pw) // 2, (_NAV_SIZE - ph) // 2, pw, ph)

    def _to_widget(self, nx: float, ny: float) -> QPointF:
        r = self._preview_rect()
        return QPointF(r.x() + nx * r.width(), r.y() + ny * r.height())

    def _to_norm(self, p: QPointF) -> tuple[float, float]:
        r = self._preview_rect()
        if r.width() == 0 or r.height() == 0:
            return 0.0, 0.0
        nx = (p.x() - r.x()) / r.width()
        ny = (p.y() - r.y()) / r.height()
        return max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))

    def _frame_rect_widget(self) -> QRectF:
        r = self._preview_rect()
        vr = self._view_rect
        return QRectF(
            r.x() + vr.x() * r.width(),
            r.y() + vr.y() * r.height(),
            vr.width() * r.width(),
            vr.height() * r.height(),
        )

    def _clamp_and_emit(self, nx: float, ny: float):
        vw = self._view_rect.width()
        vh = self._view_rect.height()
        nx = max(0.0, min(max(0.0, 1.0 - vw), nx))
        ny = max(0.0, min(max(0.0, 1.0 - vh), ny))
        self.scroll_requested.emit(nx, ny)

    # ── paint ────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), _BG_COLOR)

        if self._preview and not self._preview.isNull():
            p.drawImage(self._preview_rect(), self._preview)

        fr = self._frame_rect_widget()
        p.setPen(QPen(_FRAME_COLOR, 1.5))
        p.setBrush(_FRAME_FILL)
        p.drawRect(fr)
        p.end()

    # ── mouse ────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        fr = self._frame_rect_widget()
        if fr.contains(pos):
            self._dragging = True
            self._drag_start = pos
            self._drag_rect_start = QRectF(self._view_rect)
        else:
            nx, ny = self._to_norm(pos)
            self._clamp_and_emit(nx - self._view_rect.width() / 2,
                                  ny - self._view_rect.height() / 2)

    def mouseMoveEvent(self, event):
        if not self._dragging or self._drag_start is None or self._drag_rect_start is None:
            return
        r = self._preview_rect()
        if r.width() == 0 or r.height() == 0:
            return
        delta = event.position() - self._drag_start
        dnx = delta.x() / r.width()
        dny = delta.y() / r.height()
        self._clamp_and_emit(
            self._drag_rect_start.x() + dnx,
            self._drag_rect_start.y() + dny)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            self._drag_start = None
            self._drag_rect_start = None


class NavigatorPanel(QWidget):
    def __init__(self, layer_stack: LayerStack, canvas, scroll_area: QScrollArea,
                 parent=None, content_widget=None):
        super().__init__(parent)
        self._canvas = canvas
        # スクロール量の正規化に使うウィジェット。canvas が余白付きラッパーに
        # 収められている場合、実際にスクロールされるのはラッパーの方なので、
        # そのサイズを基準にしないと枠の表示位置・クリックスクロールがずれる。
        self._content = content_widget if content_widget is not None else canvas
        self._scroll = scroll_area
        self.setFixedWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QLabel("ナビゲーター"))

        self._view = NavigatorView()
        self._view.scroll_requested.connect(self._on_scroll_requested)
        layout.addWidget(self._view, 0, Qt.AlignmentFlag.AlignHCenter)

        zoom_row = QHBoxLayout()
        for label, factor in [("−", 1 / 1.25), ("＋", 1.25)]:
            b = QPushButton(label)
            b.setFixedHeight(24)
            b.clicked.connect(lambda _, f=factor: self._zoom(f))
            zoom_row.addWidget(b)
        reset_btn = QPushButton("リセット")
        reset_btn.setFixedHeight(24)
        reset_btn.setToolTip("100% + 中央表示")
        reset_btn.clicked.connect(self._reset_view)
        zoom_row.addWidget(reset_btn)
        fit_btn = QPushButton("全体")
        fit_btn.setFixedHeight(24)
        fit_btn.setToolTip("キャンバス全体が画面に収まる倍率にして中央表示")
        fit_btn.clicked.connect(self._fit_view)
        zoom_row.addWidget(fit_btn)
        layout.addLayout(zoom_row)

        rot_row = QHBoxLayout()
        for label, slot in [("↺", canvas.rotate_ccw), ("↻", canvas.rotate_cw),
                             ("回転リセット", canvas.reset_rotation)]:
            b = QPushButton(label)
            b.setFixedHeight(24)
            b.clicked.connect(slot)
            b.clicked.connect(self.refresh)
            rot_row.addWidget(b)
        layout.addLayout(rot_row)

        layout.addStretch()

        self._scroll.horizontalScrollBar().valueChanged.connect(self._sync_frame)
        self._scroll.verticalScrollBar().valueChanged.connect(self._sync_frame)

    # ── public ───────────────────────────────────────────────────────────────

    def refresh(self):
        """描画・レイヤー変更後に呼ぶ。composite は canvas の paintEvent で既に生成済みなので
        ここでは軽量な scaled のみ行う。"""
        composite = self._canvas.layer_stack.composite()

        rot = self._canvas._rotation
        flip = self._canvas._flip_h
        if rot != 0 or flip:
            # Canvas._c2w() と同じ変換順: flip → rotate
            t = QTransform()
            if flip:
                t.scale(-1, 1)
            t.rotate(rot)
            composite = composite.transformed(t, Qt.TransformationMode.SmoothTransformation)

        scaled = composite.scaled(
            QSize(_NAV_SIZE, _NAV_SIZE),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        self._view.update_preview(scaled)
        self._sync_frame()

    # ── internal ─────────────────────────────────────────────────────────────

    def _canvas_origin_in_content(self) -> tuple[int, int]:
        """余白ラッパー内でのキャンバスの左上位置（余白ラッパー座標系）。
        余白のない構成（content_widget 未指定）では常に (0, 0)。"""
        if self._content is self._canvas:
            return 0, 0
        pos = self._canvas.mapTo(self._content, self._canvas.rect().topLeft())
        return pos.x(), pos.y()

    def _sync_frame(self):
        """スクロールバーの現在位置からビュー枠の正規化座標を計算する。
        枠はサムネイル（キャンバス本体のみ）基準の正規化座標なので、
        余白ラッパー座標系のスクロール位置からキャンバス原点オフセット分を
        差し引いてからキャンバスサイズで正規化する。"""
        sv = self._scroll.viewport()
        if sv is None:
            return
        cw, ch = self._canvas.width(), self._canvas.height()
        if cw == 0 or ch == 0:
            return

        ox, oy = self._canvas_origin_in_content()
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        vp_w = sv.width()
        vp_h = sv.height()

        x_ratio = (hbar.value() - ox) / cw
        y_ratio = (vbar.value() - oy) / ch
        w_ratio = vp_w / cw
        h_ratio = vp_h / ch

        self._view.set_view_rect(QRectF(x_ratio, y_ratio, w_ratio, h_ratio))

    def _on_scroll_requested(self, nx: float, ny: float):
        """正規化座標（キャンバス基準）をスクロールバー値に変換してスクロールする。"""
        cw, ch = self._canvas.width(), self._canvas.height()
        ox, oy = self._canvas_origin_in_content()
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        hbar.setValue(int(nx * cw) + ox)
        vbar.setValue(int(ny * ch) + oy)

    def _zoom(self, factor: float):
        new_zoom = self._canvas.zoom * factor
        self._canvas.set_zoom(new_zoom)
        self.refresh()

    def _reset_view(self):
        self._canvas.set_zoom(1.0)
        self._canvas.set_rotation(0)
        self._canvas._flip_h = False
        self._canvas.update()
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        hbar.setValue((hbar.maximum()) // 2)
        vbar.setValue((vbar.maximum()) // 2)
        self.refresh()

    def _fit_view(self):
        """キャンバス全体がビューポートに収まるズーム倍率にして中央に表示する。"""
        sv = self._scroll.viewport()
        if sv is None:
            return
        ls = self._canvas.layer_stack
        cw, ch = ls.width, ls.height
        if self._canvas._rotation % 180 != 0:
            cw, ch = ch, cw
        if cw == 0 or ch == 0:
            return
        # 少しだけ余裕を持たせて全体が確実に見えるようにする
        zoom = min(sv.width() / cw, sv.height() / ch) * 0.95
        self._canvas.set_zoom(zoom)
        hbar = self._scroll.horizontalScrollBar()
        vbar = self._scroll.verticalScrollBar()
        hbar.setValue(hbar.maximum() // 2)
        vbar.setValue(vbar.maximum() // 2)
        self.refresh()
