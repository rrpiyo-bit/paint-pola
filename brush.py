"""brush.py — ブラシエンジン

各ブラシは BrushBase を継承し、stroke_to(img, a, b, color, size) を実装する。
Canvas は BrushBase.stroke_to() だけを呼べばよく、ブラシ固有の処理を知らなくていい。
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QRadialGradient, QBrush
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF


# ── 共通ユーティリティ ─────────────────────────────────────────────────────────

def _lerp_points(a: QPoint, b: QPoint, step: float = 1.0) -> list[QPointF]:
    """a→b 間を step px 間隔で補間した点リストを返す。"""
    dx = b.x() - a.x()
    dy = b.y() - a.y()
    dist = math.hypot(dx, dy)
    if dist < 0.001:
        return [QPointF(a)]
    n = max(1, int(dist / step))
    return [QPointF(a.x() + dx * i / n, a.y() + dy * i / n) for i in range(n + 1)]


# ── 基底クラス ────────────────────────────────────────────────────────────────

class BrushBase(ABC):
    @abstractmethod
    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int) -> None:
        """a → b へのストロークを img に描画する。"""

    def stamp(self, img: QImage, pt: QPoint, color: QColor, size: int) -> None:
        """1点だけ描画。デフォルトは stroke_to(pt, pt)。"""
        self.stroke_to(img, pt, pt, color, size)


# ── 丸ペン（デフォルト） ──────────────────────────────────────────────────────

class RoundBrush(BrushBase):
    """アンチエイリアス付きの基本丸ペン。"""

    def stamp(self, img: QImage, pt: QPoint, color: QColor, size: int) -> None:
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, size, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPoint(pt)
        p.end()

    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int) -> None:
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(color, size, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawLine(a, b)
        p.end()


# ── ソフトブラシ（ガウシアン風エアブラシ） ───────────────────────────────────

class SoftBrush(BrushBase):
    """中心が濃く周辺が薄いグラデーション円。重ね塗りで自然に濃くなる。"""

    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int) -> None:
        step = max(1.0, size * 0.15)
        pts = _lerp_points(a, b, step)
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = size / 2.0
        for pt in pts:
            grad = QRadialGradient(pt, r)
            c_center = QColor(color)
            c_center.setAlpha(int(color.alpha() * 0.35))
            c_edge = QColor(color)
            c_edge.setAlpha(0)
            grad.setColorAt(0.0, c_center)
            grad.setColorAt(1.0, c_edge)
            p.setBrush(QBrush(grad))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(pt, r, r)
        p.end()


# ── スプレー ──────────────────────────────────────────────────────────────────

class SprayBrush(BrushBase):
    """ランダム散布。サイズが大きいほど広がる。"""

    def __init__(self, density: int = 40):
        self._density = density  # 1ストロークあたりの粒数

    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int) -> None:
        pts = _lerp_points(a, b, max(1.0, size * 0.3))
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        dot_color = QColor(color)
        dot_color.setAlpha(max(30, color.alpha() // 4))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(dot_color))
        radius = size / 2.0
        for pt in pts:
            for _ in range(self._density):
                angle = random.uniform(0, 2 * math.pi)
                # 円内一様分布
                dist = radius * math.sqrt(random.random())
                dx = dist * math.cos(angle)
                dy = dist * math.sin(angle)
                dot_r = random.uniform(0.5, max(1.5, size * 0.04))
                p.drawEllipse(QPointF(pt.x() + dx, pt.y() + dy), dot_r, dot_r)
        p.end()


# ── マーカー ──────────────────────────────────────────────────────────────────

class MarkerBrush(BrushBase):
    """半透明の幅広フラットペン。重ね塗りでエッジが際立つ。"""

    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int) -> None:
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        c = QColor(color)
        c.setAlpha(min(255, int(color.alpha() * 0.55)))
        pen = QPen(c, size, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.SquareCap, Qt.PenJoinStyle.MiterJoin)
        p.setPen(pen)
        p.drawLine(a, b)
        p.end()


# ── ぼかしブラシ ──────────────────────────────────────────────────────────────

class BlurBrush(BrushBase):
    """ブラシ範囲内のピクセルを周辺と平均化してぼかす。
    strength (0.0〜1.0) でぼかしの強度を制御。"""

    def __init__(self, strength: float = 1.0):
        self.strength = max(0.0, min(1.0, strength))

    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int) -> None:
        import numpy as np
        import cv2

        w, h = img.width(), img.height()
        r = max(1, size // 2)
        x0 = max(0, min(a.x(), b.x()) - r)
        y0 = max(0, min(a.y(), b.y()) - r)
        x1 = min(w, max(a.x(), b.x()) + r + 1)
        y1 = min(h, max(a.y(), b.y()) + r + 1)
        if x1 <= x0 or y1 <= y0:
            return

        ptr = img.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()
        region = arr[y0:y1, x0:x1].copy()

        # カーネルサイズを強度に連動（1〜size相当）
        k_radius = max(1, int(3 + (size // 2) * self.strength))
        k = k_radius * 2 + 1
        blurred = cv2.GaussianBlur(region, (k, k), 0)

        rh, rw = region.shape[:2]
        ys, xs = np.mgrid[0:rh, 0:rw]
        ax, ay = a.x() - x0, a.y() - y0
        bx, by = b.x() - x0, b.y() - y0
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 == 0:
            dist2 = (xs - ax) ** 2 + (ys - ay) ** 2
        else:
            t = np.clip(((xs - ax) * dx + (ys - ay) * dy) / seg2, 0.0, 1.0)
            px = ax + t * dx
            py = ay + t * dy
            dist2 = (xs - px) ** 2 + (ys - py) ** 2
        mask = dist2 <= (r * r)

        # 強度によるブレンド
        orig = region.copy()
        region[mask] = blurred[mask]
        if self.strength < 1.0:
            alpha = self.strength
            region[mask] = (orig[mask].astype(np.float32) * (1 - alpha)
                            + region[mask].astype(np.float32) * alpha).astype(np.uint8)
        arr[y0:y1, x0:x1] = region

        result = QImage(arr.tobytes(), w, h, w * 4, QImage.Format.Format_ARGB32).copy()
        p = QPainter(img)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.drawImage(0, 0, result)
        p.end()


# ── 手ぶれ補正ラッパー ────────────────────────────────────────────────────────

class StabilizedBrush(BrushBase):
    """内包するブラシに移動平均による手ぶれ補正を適用する。
    Canvas 側で _stabilize_buffer にポイントを蓄積し、
    平均点を使って stroke_to を呼ぶことで実現する。
    このクラス自体は通常の stroke_to も持つ（バッファなし版）。"""

    def __init__(self, inner: BrushBase, smooth: int = 6):
        self.inner = inner
        self.smooth = max(1, smooth)   # 平均をとる直近点数
        self._buf: list[QPointF] = []

    def reset(self):
        self._buf.clear()

    def push(self, pt: QPoint) -> QPointF:
        """バッファに追加し、スムーズ済みの座標を返す。"""
        self._buf.append(QPointF(pt))
        if len(self._buf) > self.smooth:
            self._buf.pop(0)
        sx = sum(p.x() for p in self._buf) / len(self._buf)
        sy = sum(p.y() for p in self._buf) / len(self._buf)
        return QPointF(sx, sy)

    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int) -> None:
        self.inner.stroke_to(img, a, b, color, size)


# ── ブラシレジストリ ──────────────────────────────────────────────────────────

class BrushType:
    ROUND  = "round"
    SOFT   = "soft"
    SPRAY  = "spray"
    MARKER = "marker"
    BLUR   = "blur"


BRUSH_LABELS: dict[str, str] = {
    BrushType.ROUND:  "丸ペン",
    BrushType.SOFT:   "ソフト",
    BrushType.SPRAY:  "スプレー",
    BrushType.MARKER: "マーカー",
    BrushType.BLUR:   "ぼかし",
}

_INSTANCES: dict[str, BrushBase] = {
    BrushType.ROUND:  RoundBrush(),
    BrushType.SOFT:   SoftBrush(),
    BrushType.SPRAY:  SprayBrush(),
    BrushType.MARKER: MarkerBrush(),
    BrushType.BLUR:   BlurBrush(),
}


def get_brush(brush_type: str) -> BrushBase:
    return _INSTANCES.get(brush_type, _INSTANCES[BrushType.ROUND])
