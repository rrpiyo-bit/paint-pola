"""ベクターレイヤー。

ラスターレイヤーが「描いた瞬間にピクセルへ焼き込む」のに対して、
こちらは線（VectorStroke）を持ち続け、表示のたびに描き直す。
そのため描いたあとで太さ・色・形を何度でも変えられる。

要となる仕掛けは VectorLayer.image をプロパティにしていること。
コードベースには layer.image を直接読む箇所が多数あり（合成・統合・
保存・リサイズ・アニメ）、その全部を書き換えるのは現実的でない。
getter で必要なときだけ描き直せば、既存コードは何も知らないまま
常に最新の絵を受け取れる。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from PyQt6.QtCore import Qt, QPointF, QRectF
from PyQt6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen

from layer import CANVAS_W, CANVAS_H, Layer


@dataclass
class VectorStroke:
    """線1本。太さは線ごとに1つ（筆圧による強弱は持たない）。

    points はキャンバス座標で持つ。レイヤーローカルにすると、
    移動のたびに全点を書き換える必要が出てしまう。

    handles は points と同じ長さの並びで、その点のベジェハンドルを持つ。
    既定はすべて None で、そのときは前後の点から自動で曲がり具合が決まる
    （Catmull-Rom）。手描きの線は点が数十個できるので、全部にハンドルを
    生やすと画面が埋まって触れなくなる。必要な点だけ出す。

    ハンドルは「その点からの相対位置」で持つ。点を動かしたときに
    ハンドルが付いてくるようにするため（絶対座標だと毎回引き算が要る）。
    """
    points: list[tuple[float, float]] = field(default_factory=list)
    # 各点のハンドル。None=自動、(in_dx,in_dy,out_dx,out_dy)=手動
    handles: list[tuple[float, float, float, float] | None] = field(default_factory=list)
    width: float = 5.0
    color: tuple[int, int, int, int] = (0, 0, 0, 255)
    smooth: bool = True

    def __post_init__(self):
        # handles は points と必ず同じ長さにしておく。ここを揃えておかないと
        # 添字でずれて、別の点のハンドルを掴むことになる。
        self._sync_handles()

    def _sync_handles(self) -> None:
        n = len(self.points)
        if len(self.handles) < n:
            self.handles.extend([None] * (n - len(self.handles)))
        elif len(self.handles) > n:
            del self.handles[n:]

    def copy(self) -> "VectorStroke":
        """履歴用の複製。points も handles もタプルなので浅い複製で足りる。"""
        return replace(self, points=list(self.points), handles=list(self.handles))

    def has_handles(self, i: int) -> bool:
        return 0 <= i < len(self.handles) and self.handles[i] is not None

    def auto_handles(self, i: int) -> tuple[float, float, float, float]:
        """その点の「自動で決まっている」ハンドルを相対座標で返す。

        ハンドルを出すとき、いきなり変な形にならないよう、まずは
        今の曲線と同じ形になる値を入れるために使う。
        """
        pts = self.points
        n = len(pts)
        if n < 2 or not (0 <= i < n):
            return (0.0, 0.0, 0.0, 0.0)
        prev = pts[i - 1] if i > 0 else pts[i]
        nxt = pts[i + 1] if i < n - 1 else pts[i]
        # Catmull-Rom と同じ式（前後の点の差の 1/6）
        dx = (nxt[0] - prev[0]) / 6.0
        dy = (nxt[1] - prev[1]) / 6.0
        return (-dx, -dy, dx, dy)

    def set_handles(self, i: int, value) -> None:
        self._sync_handles()
        if 0 <= i < len(self.handles):
            self.handles[i] = value

    def handle_points(self, i: int) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """i 番の点のハンドル位置をキャンバス座標で返す（入る側, 出る側）。"""
        if not self.has_handles(i):
            return None
        px, py = self.points[i]
        ix, iy, ox, oy = self.handles[i]  # type: ignore
        return ((px + ix, py + iy), (px + ox, py + oy))

    def bounds(self) -> QRectF:
        """線幅を含めた外接矩形。当たり判定の足切りと再描画範囲に使う。"""
        if not self.points:
            return QRectF()
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        # ハンドルを引っぱると曲線が点の外側までふくらむので、
        # ハンドルの先も含めて測る。含めないと端が欠けて描かれる。
        for i, h in enumerate(self.handles):
            if h is None or i >= len(self.points):
                continue
            px, py = self.points[i]
            xs.extend((px + h[0], px + h[2]))
            ys.extend((py + h[1], py + h[3]))
        pad = self.width / 2.0 + 1.0
        return QRectF(min(xs) - pad, min(ys) - pad,
                      max(xs) - min(xs) + pad * 2,
                      max(ys) - min(ys) + pad * 2)


def catmull_rom_to_path(points: list[tuple[float, float]],
                        smooth: bool = True,
                        handles: list | None = None) -> QPainterPath:
    """点列を通るなめらかな曲線を QPainterPath にする。

    既定は Catmull-Rom スプラインを三次ベジェに変換したもの。この曲線は
    制御点を必ず通るので、「点を動かした場所を線が通る」という
    直感どおりの編集ができる。

    handles に (in_dx,in_dy,out_dx,out_dy) が入っている点では、
    自動で決まるハンドルの代わりにその値を使う。こうすると、普段は
    点だけを触り、こだわりたい点だけハンドルで曲がり具合を作り込める。
    """
    path = QPainterPath()
    if not points:
        return path
    if len(points) == 1:
        # 点を打っただけ。丸いキャップで小さな点として出したいので、
        # 長さゼロではなくごく短い線にする。
        x, y = points[0]
        path.moveTo(x, y)
        path.lineTo(x + 0.01, y)
        return path

    path.moveTo(*points[0])
    if not smooth:
        for x, y in points[1:]:
            path.lineTo(x, y)
        return path

    n = len(points)

    def _h(i: int):
        if handles is None or not (0 <= i < len(handles)):
            return None
        return handles[i]

    for i in range(n - 1):
        p1 = points[i]
        p2 = points[i + 1]
        h1 = _h(i)
        h2 = _h(i + 1)

        if h1 is not None:
            # 出る側のハンドル
            b1 = (p1[0] + h1[2], p1[1] + h1[3])
        else:
            # 自動：前の点との関係から決める（端は自分自身を前の点とみなす）
            p0 = points[i - 1] if i > 0 else p1
            b1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)

        if h2 is not None:
            # 入る側のハンドル
            b2 = (p2[0] + h2[0], p2[1] + h2[1])
        else:
            p3 = points[i + 2] if i + 2 < n else p2
            b2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)

        path.cubicTo(b1[0], b1[1], b2[0], b2[1], p2[0], p2[1])
    return path


def stroke_to_path(stroke: VectorStroke) -> QPainterPath:
    return catmull_rom_to_path(stroke.points, stroke.smooth, stroke.handles)


def _perp_distance(pt: tuple[float, float],
                   a: tuple[float, float],
                   b: tuple[float, float]) -> float:
    """点 pt から線分 ab までの距離。"""
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return ((pt[0] - ax) ** 2 + (pt[1] - ay) ** 2) ** 0.5
    t = ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    px, py = ax + t * dx, ay + t * dy
    return ((pt[0] - px) ** 2 + (pt[1] - py) ** 2) ** 0.5


def rdp_simplify(points: list[tuple[float, float]],
                 epsilon: float) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker で点を間引く。

    フリーハンドはマウスを動かすたびに点が溜まり、そのままだと
    制御点が多すぎて編集できない。形を保ったまま数を減らす。
    """
    if len(points) < 3 or epsilon <= 0:
        return list(points)

    # 再帰だと長いストロークでスタックを使い切るので、明示的なスタックで回す。
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue
        max_d = -1.0
        idx = start
        for i in range(start + 1, end):
            d = _perp_distance(points[i], points[start], points[end])
            if d > max_d:
                max_d, idx = d, i
        if max_d > epsilon:
            keep[idx] = True
            stack.append((start, idx))
            stack.append((idx, end))
    return [p for p, k in zip(points, keep) if k]


def stroke_hit(stroke: VectorStroke, x: float, y: float,
               tolerance: float = 4.0) -> bool:
    """(x,y) が線の上かどうか。線を選ぶときの当たり判定。

    線幅の半分に許容量を足した太さに膨らませて、その内側に入るかで見る。
    細い線でも掴めるよう、最低限の太さを確保する。
    """
    if not stroke.points:
        return False
    if not stroke.bounds().adjusted(-tolerance, -tolerance,
                                    tolerance, tolerance).contains(QPointF(x, y)):
        return False
    reach = max(stroke.width / 2.0, 1.0) + tolerance
    # 曲線を折れ線に落として、各線分との距離で見る。
    # QPainterPathStroker + contains でも判定できるが、細い線や
    # 開いたパスで取りこぼすことがあるので距離で見るほうが確実。
    for poly in stroke_to_path(stroke).toSubpathPolygons():
        pts = [(p.x(), p.y()) for p in poly]
        if len(pts) == 1:
            px, py = pts[0]
            if (px - x) ** 2 + (py - y) ** 2 <= reach ** 2:
                return True
            continue
        for a, b in zip(pts, pts[1:]):
            if _point_segment_distance((x, y), a, b) <= reach:
                return True
    return False


def _point_segment_distance(pt: tuple[float, float],
                            a: tuple[float, float],
                            b: tuple[float, float]) -> float:
    """点から線分までの距離。線分の外側なら端点までの距離になる。"""
    px, py = pt
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    seg2 = dx * dx + dy * dy
    if seg2 <= 1e-12:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = ((px - ax) * dx + (py - ay) * dy) / seg2
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def nearest_point_index(stroke: VectorStroke, x: float, y: float,
                        tolerance: float) -> int | None:
    """(x,y) に最も近い制御点の番号。許容量より遠ければ None。"""
    best, best_d = None, tolerance
    for i, (px, py) in enumerate(stroke.points):
        d = ((px - x) ** 2 + (py - y) ** 2) ** 0.5
        if d <= best_d:
            best, best_d = i, d
    return best


def nearest_handle(stroke: VectorStroke, x: float, y: float,
                   tolerance: float):
    """(x,y) に最も近いハンドル。(点の番号, "in"|"out") か None。"""
    best, best_d = None, tolerance
    for i in range(len(stroke.points)):
        hp = stroke.handle_points(i)
        if hp is None:
            continue
        for side, (hx, hy) in (("in", hp[0]), ("out", hp[1])):
            d = ((hx - x) ** 2 + (hy - y) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = (i, side), d
    return best


def insert_index_for(stroke: VectorStroke, x: float, y: float) -> int:
    """(x,y) に点を足すとしたら何番目かを返す。

    一番近い線分を探し、その後ろに挟む。
    """
    pts = stroke.points
    if len(pts) < 2:
        return len(pts)
    best_i, best_d = len(pts) - 1, float("inf")
    for i in range(len(pts) - 1):
        d = _point_segment_distance((x, y), pts[i], pts[i + 1])
        if d < best_d:
            best_i, best_d = i, d
    return best_i + 1


def drop_near_duplicates(points: list[tuple[float, float]],
                         min_dist: float = 2.0) -> list[tuple[float, float]]:
    """近すぎる連続点を落とす。

    ほぼ同じ場所に点が重なっていると Catmull-Rom が大きく振れて
    線がループしてしまうので、間引きの前に均しておく。
    """
    if not points:
        return []
    out = [points[0]]
    for p in points[1:]:
        lx, ly = out[-1]
        if ((p[0] - lx) ** 2 + (p[1] - ly) ** 2) ** 0.5 >= min_dist:
            out.append(p)
    if len(out) == 1 and len(points) > 1:
        # 全部が近接していた場合でも、始点と終点は残す。
        out.append(points[-1])
    return out


def simplify_input(points: list[tuple[float, float]],
                   width: float) -> list[tuple[float, float]]:
    """フリーハンド入力を制御点として扱える数まで減らす。

    太い線ほど細かい凹凸は見えないので、許容誤差を太さに比例させる。
    """
    pts = drop_near_duplicates(points, 2.0)
    return rdp_simplify(pts, max(1.0, width * 0.15))


def _pen_for(stroke: VectorStroke) -> QPen:
    pen = QPen(QColor(*stroke.color), max(0.1, stroke.width))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    return pen


def draw_strokes(painter: QPainter, strokes: list[VectorStroke],
                 offset_x: int = 0, offset_y: int = 0) -> None:
    """線を painter に描く。points はキャンバス座標なので offset を引く。"""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for stroke in strokes:
        if not stroke.points:
            continue
        path = stroke_to_path(stroke)
        if offset_x or offset_y:
            path.translate(-offset_x, -offset_y)
        painter.strokePath(path, _pen_for(stroke))


class VectorLayer(Layer):
    """線を保持し続けるレイヤー。

    image は strokes から作られるキャッシュで、strokes を変えたら
    _strokes_dirty を立てる。次に image が読まれたときに描き直される。
    """

    def __init__(self, name: str, w: int = CANVAS_W, h: int = CANVAS_H):
        # image の setter が参照するので、親の __init__ より先に用意する。
        self.strokes: list[VectorStroke] = []
        self._strokes_dirty = False
        super().__init__(name, w, h)
        # キャンバスの大きさを覚えておく。線がはみ出していないうちは
        # この大きさで描き、移動や統合が今までどおり動くようにする。
        self._canvas_w = w
        self._canvas_h = h

    @property
    def is_vector(self) -> bool:
        return True

    @property
    def image(self) -> QImage:
        if self._strokes_dirty:
            self._render_strokes()
        return self._image

    @image.setter
    def image(self, value: QImage) -> None:
        # 履歴からの復元など、外から直接絵を入れられた場合。
        # strokes より新しい絵なので、描き直しは止めておく。
        self._image = value
        self._strokes_dirty = False

    def mark_dirty(self) -> None:
        """strokes を変えたときに呼ぶ。次の表示で描き直される。"""
        self._strokes_dirty = True
        self._effect_cache = None

    def _render_strokes(self) -> None:
        """strokes から image を作り直す。

        キャンバスに収まっているうちは offset を動かさない。
        offset が描くたびに変わると、移動ツールや統合が戸惑うため。
        はみ出したぶんだけ image を広げる。
        """
        self._strokes_dirty = False
        cw = getattr(self, "_canvas_w", CANVAS_W)
        ch = getattr(self, "_canvas_h", CANVAS_H)

        area = QRectF(0, 0, cw, ch)
        for stroke in self.strokes:
            if stroke.points:
                area = area.united(stroke.bounds())

        # キャンバスに収まっているうちは、ぴったりキャンバスの大きさにする。
        # はみ出したぶんだけ外側へ広げる。
        ox = min(0, int(area.left()) - 1) if area.left() < 0 else 0
        oy = min(0, int(area.top()) - 1) if area.top() < 0 else 0
        w = max(cw, int(area.right()) + 2 - ox) if area.right() > cw else cw - ox
        h = max(ch, int(area.bottom()) + 2 - oy) if area.bottom() > ch else ch - oy

        img = QImage(w, h, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        painter = QPainter(img)
        draw_strokes(painter, self.strokes, ox, oy)
        painter.end()

        self._image = img
        self.offset_x = ox
        self.offset_y = oy

    def set_canvas_size(self, w: int, h: int) -> None:
        self._canvas_w = int(w)
        self._canvas_h = int(h)
        self.mark_dirty()

    def add_stroke(self, stroke: VectorStroke) -> None:
        self.strokes.append(stroke)
        self.mark_dirty()

    def copy_strokes(self) -> list[VectorStroke]:
        """履歴用に線をまとめて複製する。"""
        return [s.copy() for s in self.strokes]

    def clear(self):
        self.strokes = []
        self.mark_dirty()

    def translate_strokes(self, dx: float, dy: float) -> None:
        """線をまとめて動かす。

        移動ツール用。points がキャンバス座標なので offset を動かすだけでは
        描き直しで元の位置に戻ってしまう。点そのものを動かす必要がある。
        """
        if dx == 0 and dy == 0:
            return
        for stroke in self.strokes:
            stroke.points = [(x + dx, y + dy) for x, y in stroke.points]
        self.mark_dirty()

    def scale_strokes(self, sx: float, sy: float) -> None:
        """線をまとめて拡大縮小する。キャンバスのサイズ変更用。"""
        for stroke in self.strokes:
            stroke.points = [(x * sx, y * sy) for x, y in stroke.points]
            # ハンドルは点からの相対位置なので、移動では触らなくてよいが
            # 拡大縮小では一緒に伸ばさないと曲がり具合が変わってしまう。
            stroke.handles = [None if h is None
                              else (h[0] * sx, h[1] * sy, h[2] * sx, h[3] * sy)
                              for h in stroke.handles]
            stroke.width = max(0.1, stroke.width * (abs(sx) + abs(sy)) / 2.0)
        self.mark_dirty()

    def rasterize(self) -> None:
        """レイヤー効果を絵に焼き込む。

        効果はベクターのままでは持てないので、焼いた時点で線は捨てる。
        以後はふつうのラスターと同じ扱いになる。
        """
        baked = self.image_with_effects().convertToFormat(
            QImage.Format.Format_ARGB32)
        self.strokes = []
        self.image = baked          # setter が _strokes_dirty を下ろす
        self.border_enabled = False
        self.shadow_enabled = False
        self.glow_enabled = False
        self.blur_enabled = False
        self.hsl_enabled = False

    def to_raster(self) -> Layer:
        """同じ見た目のふつうのラスターレイヤーを作って返す。"""
        lyr = Layer(self.name, self._canvas_w, self._canvas_h)
        lyr.image = self.image.copy()
        lyr.offset_x = self.offset_x
        lyr.offset_y = self.offset_y
        for attr in ("visible", "opacity", "clipping", "reference", "locked",
                     "blend_mode",
                     "border_enabled", "border_size", "border_color",
                     "shadow_enabled", "shadow_color", "shadow_offset_x",
                     "shadow_offset_y", "shadow_blur", "shadow_strength",
                     "glow_enabled", "glow_color", "glow_size", "glow_strength",
                     "blur_enabled", "blur_radius", "blur_strength",
                     "hsl_enabled", "hsl_hue", "hsl_saturation", "hsl_lightness"):
            setattr(lyr, attr, getattr(self, attr))
        return lyr
