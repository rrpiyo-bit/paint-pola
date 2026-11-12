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
                  color: QColor, size: int, antialias: bool = True) -> None:
        """a → b へのストロークを img に描画する。

        antialias=False なら線のふちをぼかさない（画素が「塗る／塗らない」の
        どちらかになる）。ドット絵や、塗りつぶしで隙間を残したくない線画用。
        ふちそのものがぼかしのブラシ（ソフト・ぼかし）は無視する。
        """

    def stamp(self, img: QImage, pt: QPoint, color: QColor, size: int, antialias: bool = True) -> None:
        """1点だけ描画。デフォルトは stroke_to(pt, pt)。"""
        self.stroke_to(img, pt, pt, color, size, antialias)


# ── 丸ペン（デフォルト） ──────────────────────────────────────────────────────

class RoundBrush(BrushBase):
    """アンチエイリアス付きの基本丸ペン。"""

    def stamp(self, img: QImage, pt: QPoint, color: QColor, size: int, antialias: bool = True) -> None:
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, antialias)
        pen = QPen(color, size, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawPoint(pt)
        p.end()

    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int, antialias: bool = True) -> None:
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, antialias)
        pen = QPen(color, size, Qt.PenStyle.SolidLine,
                   Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.drawLine(a, b)
        p.end()


# ── ソフトブラシ（ガウシアン風エアブラシ） ───────────────────────────────────

class SoftBrush(BrushBase):
    """中心が濃く周辺が薄いグラデーション円。重ね塗りで自然に濃くなる。"""

    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int, antialias: bool = True) -> None:
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

# スプレーの設定名 → (既定値, 最小, 最大)。形は SPRAY_SHAPES、吹き続けは bool。
SPRAY_RANGES: dict[str, tuple[int, int, int]] = {
    "density":       (40, 1, 300),  # 1回の吹き付けで出す粒の数
    "particle_size": (3, 1, 60),    # 粒の大きさ（直径 px）
    "size_jitter":   (60, 0, 100),  # 粒の大きさのばらつき (%)
    "bias":          (0, 0, 100),   # 中心への寄せ具合 (%)。0 で円の中に均一
    "opacity":       (25, 1, 100),  # 粒1つの濃さ（ペン色に対する %）
}
SPRAY_SHAPES = ("circle", "square", "star", "streak")
SPRAY_SHAPE_LABELS = {"circle": "丸", "square": "四角", "star": "星", "streak": "細長い線"}
SPRAY_DEFAULTS: dict = {k: v[0] for k, v in SPRAY_RANGES.items()}
SPRAY_DEFAULTS.update({"shape": "circle", "continuous": True})


def sanitize_spray(field: str, value):
    """保存値や UI から来た値を、使える型と範囲に直す。使えなければ None。"""
    if field in SPRAY_RANGES:
        _, lo, hi = SPRAY_RANGES[field]
        try:
            return max(lo, min(hi, int(value)))
        except (TypeError, ValueError):
            return None
    if field == "shape":
        return value if value in SPRAY_SHAPES else None
    if field == "continuous":
        return value if isinstance(value, bool) else str(value).lower() == "true"
    return None


def _star_path(r: float):
    from PyQt6.QtGui import QPainterPath
    path = QPainterPath()
    for i in range(10):
        rr = r if i % 2 == 0 else r * 0.45
        ang = -math.pi / 2 + i * math.pi / 5
        pt = QPointF(rr * math.cos(ang), rr * math.sin(ang))
        if i == 0:
            path.moveTo(pt)
        else:
            path.lineTo(pt)
    path.closeSubpath()
    return path


class SprayBrush(BrushBase):
    """ランダム散布（クリスタのエアブラシ「スプレー」「飛沫」に相当）。

    ブラシサイズの円の中に粒を散らす。粒の数・大きさ・大きさのばらつき・
    中心への寄せ具合・粒の形・濃さは settings で変えられる。
    """

    def __init__(self, density: int = 40):
        self.settings: dict = dict(SPRAY_DEFAULTS)
        self.settings["density"] = density

    def _particle(self, p: QPainter, x: float, y: float, r: float, shape: str):
        if shape == "circle":
            p.drawEllipse(QPointF(x, y), r, r)
            return
        p.save()
        p.translate(x, y)
        p.rotate(random.uniform(0, 360))
        if shape == "square":
            p.drawRect(QRectF(-r, -r, 2 * r, 2 * r))
        elif shape == "star":
            p.drawPath(_star_path(r * 1.4))
        else:  # streak: 細長い線（飛沫の筋）
            p.drawRect(QRectF(-r * 2.5, -r * 0.35, r * 5, r * 0.7))
        p.restore()

    def _offset(self, radius: float) -> tuple[float, float]:
        """円の中の1点（中心からのずれ）。寄せるほど中心寄りの正規分布から選ぶ。"""
        if random.random() < self.settings["bias"] / 100.0:
            sigma = radius * 0.4
            for _ in range(4):
                dx, dy = random.gauss(0, sigma), random.gauss(0, sigma)
                if dx * dx + dy * dy <= radius * radius:
                    return dx, dy
        # 円の中に均一（面積あたり同じ数）
        angle = random.uniform(0, 2 * math.pi)
        dist = radius * math.sqrt(random.random())
        return dist * math.cos(angle), dist * math.sin(angle)

    def spray_at(self, p: QPainter, pt: QPointF, size: float,
                 to: QPointF | None = None, count: int | None = None) -> None:
        """pt を中心に粒を吹き付ける。to を渡すと pt→to の線上に散らす。

        線のときは粒ごとに中心を線上のどこかから選ぶ。決まった間隔の点を
        中心にすると、中心に寄せたときに点ごとの濃い塊が数珠のように並ぶ。
        """
        st = self.settings
        radius = size / 2.0
        base_r = st["particle_size"] / 2.0
        jit = st["size_jitter"] / 100.0
        shape = st["shape"]
        n = st["density"] if count is None else count
        for _ in range(n):
            if to is None:
                cx, cy = pt.x(), pt.y()
            else:
                t = random.random()
                cx = pt.x() + (to.x() - pt.x()) * t
                cy = pt.y() + (to.y() - pt.y()) * t
            dx, dy = self._offset(radius)
            r = max(0.3, base_r * (1.0 - jit * random.random()))
            self._particle(p, cx + dx, cy + dy, r, shape)

    def _begin(self, img: QImage, color: QColor, antialias: bool) -> QPainter:
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, antialias)
        dot_color = QColor(color)
        dot_color.setAlpha(max(1, int(color.alpha() * self.settings["opacity"] / 100.0)))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(dot_color))
        return p

    def stamp(self, img: QImage, pt: QPoint, color: QColor, size: int,
              antialias: bool = True) -> None:
        p = self._begin(img, color, antialias)
        self.spray_at(p, QPointF(pt), size)
        p.end()

    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int, antialias: bool = True) -> None:
        # 粒の総数は以前と同じく「ブラシ幅の 3 割ごとに density 個」
        n_steps = len(_lerp_points(a, b, max(1.0, size * 0.3)))
        p = self._begin(img, color, antialias)
        self.spray_at(p, QPointF(a), size, QPointF(b),
                      self.settings["density"] * n_steps)
        p.end()


# ── マーカー ──────────────────────────────────────────────────────────────────

class MarkerBrush(BrushBase):
    """半透明の幅広フラットペン。重ね塗りでエッジが際立つ。"""

    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int, antialias: bool = True) -> None:
        p = QPainter(img)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, antialias)
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
                  color: QColor, size: int, antialias: bool = True) -> None:
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
    Canvas 側は self._stabilizer として本クラスを持ち、
    push() に生の座標を渡して平均済みの座標を受け取り、
    それを使って stroke_to を呼ぶ。点の蓄積は本クラスの _buf。
    このクラス自体は通常の stroke_to も持つ（バッファなし版）。"""

    def __init__(self, inner: BrushBase, smooth: int = 6):
        self.inner = inner
        self.smooth = max(1, smooth)   # 平均をとる直近点数
        self._buf: list[QPointF] = []

    def reset(self):
        self._buf.clear()

    def translate(self, delta) -> None:
        """バッファはレイヤーローカル座標なので、レイヤー画像が拡張されて
        原点がずれたときに蓄積済みの点をまとめて補正する。"""
        if not self._buf:
            return
        dx, dy = float(delta.x()), float(delta.y())
        if dx == 0.0 and dy == 0.0:
            return
        self._buf = [QPointF(p.x() + dx, p.y() + dy) for p in self._buf]

    def push(self, pt: QPoint) -> QPointF:
        """バッファに追加し、スムーズ済みの座標を返す。"""
        self._buf.append(QPointF(pt))
        if len(self._buf) > self.smooth:
            self._buf.pop(0)
        sx = sum(p.x() for p in self._buf) / len(self._buf)
        sy = sum(p.y() for p in self._buf) / len(self._buf)
        return QPointF(sx, sy)

    def drain(self) -> list[QPointF]:
        """離したときに呼ぶ。平均の遅れで線が手前で止まらないよう、
        最後の生の座標まで追いつく途中の点を返す。"""
        if not self._buf:
            return []
        last = self._buf[-1]
        out = []
        while any(p != last for p in self._buf):
            out.append(self.push(last))
        return out

    def stroke_to(self, img: QImage, a: QPoint, b: QPoint,
                  color: QColor, size: int, antialias: bool = True) -> None:
        self.inner.stroke_to(img, a, b, color, size, antialias)


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
