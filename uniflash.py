"""ウニフラッシュ（放射状のトゲ）の形を作る。

楕円ツールと同じく、ドラッグした矩形（左上 → 右下）がウニフラ全体の
外側の大きさになる。トゲは外側の楕円から内側へ向かって伸び、内側の
端（入り）と外側の端（抜き）は最初から細くなっている。

形はキャンバスに依存しない純粋な計算なので、プレビューと確定で
同じ seed を使えば、離したときに見ていた形がそのまま描かれる。
"""
from __future__ import annotations

import math
import random

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPolygonF

# 設定名 → (既定値, 最小, 最大)。bool / 色は範囲を持たない。
UNI_RANGES: dict[str, tuple] = {
    "thickness":    (6, 1, 100),    # トゲの根元の太さ (px)
    "spacing":      (4, 1, 60),     # 隣のトゲ（束）との間隔 (度)
    "bundle_count": (4, 2, 12),     # 束にするときの1束の本数
    "length":       (45, 5, 100),   # トゲの長さ（外側の半径に対する %）
    "jitter":       (40, 0, 100),   # 長さのばらつき (%)
    "taper_in":     (35, 0, 100),   # 入り：内側の端を細くする長さ（トゲの長さに対する %）
    "taper_out":    (50, 0, 100),   # 抜き：外側の端を細くする長さ（同上）
}
UNI_MODES = ("single", "bundle")

UNI_DEFAULTS: dict = {k: v[0] for k, v in UNI_RANGES.items()}
UNI_DEFAULTS.update({
    "mode": "single",          # "single"（1本ずつ）| "bundle"（束）
    "fill": False,             # 中（トゲの内側）を塗るか
    "fill_color": "#ffffff",   # 中を塗る色
})

_SAMPLES = 10  # トゲ1本の片側の点の数（多いほど入り抜きがなめらか）


def sanitize_uni(field: str, value):
    """保存値や UI から来た値を、使える型と範囲に直す。使えなければ None。"""
    if field in UNI_RANGES:
        _, lo, hi = UNI_RANGES[field]
        try:
            return max(lo, min(hi, int(value)))
        except (TypeError, ValueError):
            return None
    if field == "mode":
        return value if value in UNI_MODES else None
    if field == "fill":
        if isinstance(value, bool):
            return value
        return str(value).lower() == "true"
    if field == "fill_color":
        c = QColor(str(value))
        return c.name() if c.isValid() else None
    return None


def default_uni() -> dict:
    return dict(UNI_DEFAULTS)


def _taper_profile(t: float, tin: float, tout: float) -> float:
    """トゲの根元側(t=0)から先(t=1)までの太さの割合 0〜1。"""
    total = tin + tout
    if total > 1.0:  # 入り＋抜きが長さを超えるときは同じ比率で縮める
        tin, tout = tin / total, tout / total
    f = 1.0
    if tin > 0 and t < tin:
        f = min(f, math.sin(0.5 * math.pi * t / tin))
    if tout > 0 and t > 1.0 - tout:
        f = min(f, math.sin(0.5 * math.pi * (1.0 - t) / tout))
    return max(0.0, f)


def _spike_polygon(a: QPointF, b: QPointF, width: float,
                   tin: float, tout: float) -> QPolygonF | None:
    """a（内側）から b（外側）へ伸びる、両端が尖ったトゲの多角形。"""
    dx, dy = b.x() - a.x(), b.y() - a.y()
    length = math.hypot(dx, dy)
    if length < 0.5:
        return None
    nx, ny = -dy / length, dx / length  # トゲに直交する向き
    left, right = [], []
    for i in range(_SAMPLES + 1):
        t = i / _SAMPLES
        hw = width / 2.0 * _taper_profile(t, tin, tout)
        px, py = a.x() + dx * t, a.y() + dy * t
        left.append(QPointF(px + nx * hw, py + ny * hw))
        right.append(QPointF(px - nx * hw, py - ny * hw))
    return QPolygonF(left + right[::-1])


def _spikes(rect: QRectF, opts: dict, seed: int):
    """(角度, 内側の比率, 外側の比率, 太さの倍率) を並べて返す。

    比率は外側の楕円の半径に対する値（1.0 が外側の楕円上）。
    """
    rng = random.Random(seed)
    length = opts["length"] / 100.0
    jitter = opts["jitter"] / 100.0
    step = math.radians(opts["spacing"])
    count = max(1, int(round(2 * math.pi / step)))
    step = 2 * math.pi / count  # 一周でぴったり閉じるよう間隔をならす
    start = rng.uniform(0, step)
    bundle = opts["mode"] == "bundle"
    n_in_bundle = opts["bundle_count"] if bundle else 1
    out = []
    for i in range(count):
        center = start + i * step + rng.uniform(-0.15, 0.15) * step
        # 束全体の長さを揃えておき、束の端ほど短くする
        group_short = rng.random() * jitter
        for j in range(n_in_bundle):
            if bundle:
                # 束の幅は間隔の 6 割。残りが束どうしのすき間になる
                k = (j - (n_in_bundle - 1) / 2.0) / max(1, n_in_bundle - 1)
                ang = center + k * step * 0.6
                edge = abs(k) * 2.0  # 束の中心で 0、端で 1
                short = min(1.0, group_short + edge * 0.35 + rng.random() * jitter * 0.3)
            else:
                ang = center
                short = rng.random() * jitter
            inner = (1.0 - length) + length * short * 0.85
            outer = 1.0 - length * rng.random() * jitter * 0.15
            if outer - inner < 0.02:
                inner = outer - 0.02
            width_k = 1.0 - 0.4 * rng.random() * jitter
            out.append((ang, inner, outer, width_k))
    return out


def uni_flash_paths(rect: QRectF, opts: dict, seed: int):
    """(トゲの path, 中を塗る path または None) を返す。"""
    rect = rect.normalized()
    cx, cy = rect.center().x(), rect.center().y()
    rx, ry = rect.width() / 2.0, rect.height() / 2.0
    spikes = QPainterPath()
    spikes.setFillRule(Qt.FillRule.WindingFill)
    if rx < 1 or ry < 1:
        return spikes, None
    tin = opts["taper_in"] / 100.0
    tout = opts["taper_out"] / 100.0
    tips = []  # (角度, 内側の端)。中を塗る形をトゲの根元に沿わせるのに使う
    for ang, inner, outer, width_k in _spikes(rect, opts, seed):
        c, s = math.cos(ang), math.sin(ang)
        a = QPointF(cx + rx * inner * c, cy + ry * inner * s)
        b = QPointF(cx + rx * outer * c, cy + ry * outer * s)
        tips.append((ang, a))
        poly = _spike_polygon(a, b, opts["thickness"] * width_k, tin, tout)
        if poly is not None:
            spikes.addPolygon(poly)
            spikes.closeSubpath()
    fill = None
    if opts.get("fill") and len(tips) >= 3:
        # トゲの内側の端を順に結んだ形を塗る。楕円で塗ると、短いトゲとの間に
        # すき間ができて、トゲが塗りから生えているように見えない。
        tips.sort(key=lambda t: t[0] % (2 * math.pi))
        fill = QPainterPath()
        fill.addPolygon(QPolygonF([pt for _, pt in tips]))
        fill.closeSubpath()
    return spikes, fill


def draw_uni_flash(p: QPainter, rect: QRectF, opts: dict, color: QColor,
                   seed: int, antialias: bool = True) -> None:
    """painter の今の座標系に、ウニフラッシュを描く。"""
    spikes, fill = uni_flash_paths(rect, opts, seed)
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, antialias)
    p.setPen(Qt.PenStyle.NoPen)
    if fill is not None:
        p.setBrush(QBrush(QColor(opts.get("fill_color", "#ffffff"))))
        p.drawPath(fill)
    p.setBrush(QBrush(color))
    p.drawPath(spikes)
    p.restore()
