from __future__ import annotations
import math
import time
import numpy as np
import cv2

from PyQt6.QtWidgets import QWidget, QApplication, QInputDialog, QFontDialog, QColorDialog
from PyQt6.QtGui import (QPainter, QColor, QPen, QImage, QFont, QPixmap,
                          QTransform, QBrush, QPainterPath, QFontMetrics,
                          QRadialGradient)
from PyQt6.QtCore import (Qt, QPoint, QPointF, QRect, QRectF, QSize, pyqtSignal, QTimer,
                          QEvent)

from layer import LayerStack, Layer, BLEND_KEY_TO_MODE
from tools import Tool
from brush import BrushType, get_brush, StabilizedBrush, BlurBrush, BRUSH_LABELS

# ツールキーボードショートカット（toolbar.py の TOOL_SHORTCUTS と一致させること）
_TOOL_KEY_MAP: dict[Qt.Key, Tool] = {
    Qt.Key.Key_P: Tool.PEN,
    Qt.Key.Key_E: Tool.ERASER,
    Qt.Key.Key_G: Tool.FILL,
    Qt.Key.Key_I: Tool.EYEDROPPER,
    Qt.Key.Key_L: Tool.LINE,
    Qt.Key.Key_R: Tool.RECT,
    Qt.Key.Key_O: Tool.ELLIPSE,
    Qt.Key.Key_T: Tool.TEXT,
    Qt.Key.Key_B: Tool.BLUR,
    Qt.Key.Key_S: Tool.SELECT_RECT,
    Qt.Key.Key_Q: Tool.LASSO,
    Qt.Key.Key_W: Tool.LASSO_FILL,
    Qt.Key.Key_V: Tool.MOVE,
    Qt.Key.Key_F: Tool.TRANSFORM,
}

# ── 定数 ──────────────────────────────────────────────────────────────────────
HISTORY_LIMIT = 100
# 履歴1件はレイヤー画像の丸ごとコピー（2500x2500 なら 25MB）なので、
# 件数だけで制限すると件数上限に達する前にメモリ不足で落ちる。
# 総バイト数でも打ち切り、大きなキャンバスでは件数が減るようにする。
# 実際に戻せる回数はこちらで決まることが多い（2500x2500 で約42件、
# 4000x4000 で約16件）。「戻れる数が少ない」の原因はほぼこの制限側。
HISTORY_MEMORY_LIMIT = 1024 * 1024 * 1024  # 1GB
# ただし直近の操作を取り消せないと困るので、この件数までは必ず残す。
HISTORY_MIN_ENTRIES = 5
MIN_ZOOM = 0.05
MIN_TRANSFORM_SIZE = 1
HANDLE_HIT_RADIUS = 12
GRID_COLOR = QColor(180, 180, 180, 160)
SELECTION_COLOR = QColor(0, 120, 215)


# ── 方眼 ─────────────────────────────────────────────────────────────────────

def _grid_lines(length: int, spacing: int, origin: str) -> list[int]:
    """0〜length の範囲に引く方眼の線の座標を返す。

    origin="corner" は隅（0）を起点にマス目を並べる。
    origin="center" は中央に線の交点が来るようにし、そこから両方向へ
    等間隔に伸ばす（中心を基準に構図を取りたいとき用）。
    """
    g = max(1, int(spacing))
    if origin == "center":
        center = length / 2.0
        # 中心から g 刻みで両側へ。始点は範囲内に入る最小の線。
        first = center - math.floor(center / g) * g
        start = int(round(first))
    else:
        start = 0
    return list(range(start, length + 1, g))


# ── 変形ユーティリティ ────────────────────────────────────────────────────────────

def _constrain_corner_shift(pt: QPointF, fixed_x: float, fixed_y: float,
                             ratio: float) -> QPointF:
    """Shift 拘束: コーナー pt を ratio (w/h) に従い固定辺から調整する。
    fixed_x/fixed_y は動かさない反対側の辺座標。
    ratio 正規化した移動量で「どちらの軸のドラッグが大きいか」を判定する。
    """
    moved_w = abs(pt.x() - fixed_x)
    moved_h = abs(pt.y() - fixed_y)
    sign_x = 1.0 if pt.x() >= fixed_x else -1.0
    sign_y = 1.0 if pt.y() >= fixed_y else -1.0

    if ratio <= 0:
        return pt
    # 正規化移動量が大きい軸を優先し、もう一軸を比率拘束する
    if moved_w / ratio >= moved_h:
        # 幅優先 → 高さを幅から計算
        return QPointF(pt.x(), fixed_y + sign_y * moved_w / ratio)
    else:
        # 高さ優先 → 幅を高さから計算
        return QPointF(fixed_x + sign_x * moved_h * ratio, pt.y())


def _constrain_shape_shift(tool, start: QPoint, end: QPoint) -> QPoint:
    """図形ツール（直線・四角形・楕円）用の Shift 拘束。
    直線: 45度刻みにスナップする。四角形/楕円: 正方形/正円にする（長い方の辺に合わせる）。
    """
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    if tool == Tool.LINE:
        dist = math.hypot(dx, dy)
        if dist == 0:
            return end
        angle = math.atan2(dy, dx)
        step = math.pi / 4  # 45度刻み
        snapped = round(angle / step) * step
        return QPoint(round(start.x() + dist * math.cos(snapped)),
                      round(start.y() + dist * math.sin(snapped)))
    else:
        side = max(abs(dx), abs(dy))
        sx = 1 if dx >= 0 else -1
        sy = 1 if dy >= 0 else -1
        return QPoint(start.x() + sx * side, start.y() + sy * side)


# ── 塗りつぶし ──────────────────────────────────────────────────────────────────

def _alpha(pixel: int) -> int:
    return (pixel >> 24) & 0xFF


LINE_ALPHA_THRESHOLD = 10

# 手ブレ補正の上限。大きいほど線がなめらかになるが、カーソルから遅れて付いてくる。
STABILIZATION_MAX = 30

# 入り抜きの長さの上限（px）
TAPER_MAX = 500


def _taper_width(d: float, total: float | None, size: float,
                 t_in: float, t_out: float, tip_pct: float) -> float:
    """入り抜きを付けたときの、線の始点から d の位置での太さ。

    total が None のときは描いている途中で終わりが分からないので、入りだけ効かせる。
    線が入り＋抜きより短いときは、両方を同じ比率で縮めて線の中に収める。
    """
    if total is not None and t_in + t_out > total:
        k = total / (t_in + t_out) if t_in + t_out > 0 else 0.0
        t_in, t_out = t_in * k, t_out * k
    f = 1.0
    if t_in > 0:
        f = min(f, d / t_in)
    if total is not None and t_out > 0:
        f = min(f, (total - d) / t_out)
    f = max(0.0, min(1.0, f))
    tip = size * tip_pct / 100.0
    return max(1.0, tip + (size - tip) * f)


def _taper_pieces(pts: list[QPointF], size: float, t_in: float, t_out: float,
                  tip_pct: float, total: float | None = None, start_dist: float = 0.0):
    """折れ線 pts を、太さの変わる所だけ 1px 刻みに分けて (a, b, 太さ) で返す。

    太さが一定の区間はまとめて1回で描く（長い線で描画回数が増えすぎないように）。
    start_dist は pts[0] が線の始点からどれだけ進んだ位置か（描いている途中用）。
    """
    def w(d):
        return _taper_width(d, total, size, t_in, t_out, tip_pct)

    d = start_dist
    for a, b in zip(pts, pts[1:]):
        seg = math.hypot(b.x() - a.x(), b.y() - a.y())
        if seg < 1e-6:
            continue
        d0, d1 = d, d + seg
        d = d1
        if w(d0) >= size and w(d1) >= size and w((d0 + d1) / 2) >= size:
            yield a, b, float(size)
            continue
        n = max(1, int(math.ceil(seg)))
        for i in range(n):
            pa = QPointF(a.x() + (b.x() - a.x()) * i / n, a.y() + (b.y() - a.y()) * i / n)
            pb = QPointF(a.x() + (b.x() - a.x()) * (i + 1) / n,
                         a.y() + (b.y() - a.y()) * (i + 1) / n)
            yield pa, pb, w(d0 + seg * (i + 0.5) / n)
# 「線」と判定する alpha の下限。既定の 10 だと薄いアンチエイリアス部分まで
# 線扱いになり、逆に上げすぎると薄い線が無視されて塗りが漏れる。


def _sensitivity_to_threshold(sensitivity: int) -> int:
    """「薄い線を拾う感度」(0-100%) を alpha しきい値(0-10) に換算する。

    感度を上げるほどしきい値が下がり、薄いピクセルまで線とみなすので、
    アンチエイリアスで色が薄くなった部分が「途切れ」と判定されにくくなる。
    0% は従来どおり alpha>10、100% は alpha が 1 でもあれば線。
    しきい値を上げる方向（薄い線を無視する）は塗り漏れを増やすだけなので用意しない。
    """
    s = max(0, min(100, sensitivity))
    return round(LINE_ALPHA_THRESHOLD * (100 - s) / 100)


def _line_free_mask(judge_arr: np.ndarray, threshold: int = LINE_ALPHA_THRESHOLD) -> np.ndarray:
    """参照配列(BGRA)のうち「線ではない＝塗ってよい」ピクセルの真偽マスクを返す。

    基本は「透明＝塗れる／不透明＝線」。ただし線画を白背景で描いた（あるいは
    背景を白で塗り潰した）レイヤーを参照にすると、全面が不透明になって
    どこも塗れなくなってしまう。そこで、不透明部分の大半が単一の明るい色
    （＝紙の地）で占められている場合に限り、その色を背景とみなして
    塗れる側に含める。線そのものは地色と違う色なので境界として残る。
    """
    alpha = judge_arr[:, :, 3]
    free = alpha <= threshold
    opaque = ~free
    n_opaque = int(np.count_nonzero(opaque))
    if n_opaque == 0 or n_opaque < alpha.size * 0.5:
        # 透明背景の普通の線画。従来どおり alpha だけで判定する。
        return free

    # 不透明部分の最頻色を求める（各チャンネル32段に丸めて集計）
    bgr = judge_arr[:, :, :3][opaque]
    keys = ((bgr[:, 0] >> 3).astype(np.int32) << 10) \
        | ((bgr[:, 1] >> 3).astype(np.int32) << 5) \
        | (bgr[:, 2] >> 3).astype(np.int32)
    vals, counts = np.unique(keys, return_counts=True)
    top = int(vals[int(np.argmax(counts))])
    share = int(counts.max()) / n_opaque
    b = ((top >> 10) & 0x1F) << 3
    g = ((top >> 5) & 0x1F) << 3
    r = (top & 0x1F) << 3
    # 地色とみなす条件: 不透明部分の過半を占めていて、かつ明るい色であること。
    if share < 0.5 or (0.114 * b + 0.587 * g + 0.299 * r) < 160:
        return free

    diff = np.abs(judge_arr[:, :, :3].astype(np.int16)
                  - np.array([b, g, r], dtype=np.int16)).max(axis=2)
    return free | (opaque & (diff <= 24))


def _holds(strokes, stroke) -> bool:
    """strokes に stroke そのもの（同じオブジェクト）が入っているか。
    VectorStroke は dataclass なので `in` だと中身が同じ別の線にも一致し、
    undo で差し替わった古い線を掴んだまま編集が空振りしていた。"""
    return any(s is stroke for s in strokes)


def _alpha_array(img: QImage) -> np.ndarray:
    """画像の alpha チャンネルを (h, w) の配列で返す（コピー）。"""
    src = img.convertToFormat(QImage.Format.Format_ARGB32)
    w, h = src.width(), src.height()
    ptr = src.constBits()
    ptr.setsize(h * w * 4)
    return np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)[:, :, 3].copy()


def _shifted_image(img: QImage, size, shift) -> QImage:
    """img を shift だけずらして、size の透明な画像に置いたものを返す。"""
    out = QImage(size, QImage.Format.Format_ARGB32)
    out.fill(Qt.GlobalColor.transparent)
    p = QPainter(out)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
    p.drawImage(shift.x(), shift.y(), img)
    p.end()
    return out


def _flood_fill(image: QImage, x: int, y: int, fill_color: QColor,
                ref_image: QImage | None = None,
                close_gap: int = 0, line_threshold: int = LINE_ALPHA_THRESHOLD,
                include_self: bool = False, tolerance: int = 0,
                color_judge: QImage | None = None):
    """連結領域を numpy/cv2 のラベリングで検出し、一括書き込みする塗りつぶし。
    QImage.pixel()/setPixel() を1ピクセルずつ呼ぶ旧scanline実装は、大キャンバスで
    UIスレッドが長時間ブロックされフリーズ/クラッシュする原因になっていたため廃止。

    close_gap: 線画が途切れていても塗りが漏れないよう、判定用の線マスクだけを
        この px だけ太らせる。実際に塗る範囲は元の線位置まで戻すので、
        塗りが痩せることはない。
    line_threshold: この alpha 以下のピクセルは「線ではない」とみなす。
        値を上げると薄いアンチエイリアス部分を線扱いしなくなり、
        「色が薄いせいで途切れ扱いされる」のを防げる（参照モードのみ有効）。
    include_self: 参照レイヤーだけでなく、塗る対象レイヤー自身に既に描かれている
        ピクセルも境界として扱う（クリスタの「複数参照: 参照レイヤー＋編集レイヤー」
        に相当）。off だと、塗る側に描いた囲み線を素通りして外まで漏れる。
        アニメ塗りのように「線画だけを境界にして下のレイヤーで色を塗り分ける」
        使い方では off が正しいので、既定は off のまま切り替え式にしてある。
    tolerance: 色で判定するとき（参照なし・すべてのレイヤー）、クリックした所と
        各チャンネルの差がこの % 以内なら同じ色とみなす。0 で完全一致。
    color_judge: 色の判定に使う画像（「すべてのレイヤー」の合成）。None なら
        塗るレイヤー自身で判定する。image と同じ大きさ・座標であること。"""
    w, h = image.width(), image.height()
    if not (0 <= x < w and 0 <= y < h):
        return

    if ref_image is not None:
        judge = ref_image
    elif color_judge is not None:
        judge = color_judge
    else:
        judge = image
    fill = fill_color.rgba()

    nbytes = h * w * 4
    judge_ptr = judge.bits(); judge_ptr.setsize(nbytes)
    judge_arr = np.frombuffer(judge_ptr, dtype=np.uint8).reshape(h, w, 4)

    # 参照モード: judge の不透明ピクセルが境界、image の未塗りピクセルが対象
    # 通常モード: image の同色ピクセルが対象
    if ref_image is not None:
        candidate = _line_free_mask(judge_arr, line_threshold)
        if include_self:
            # 自分のレイヤーに既に描かれている部分も境界にする。
            # 参照が線画だけのとき、塗る側に描いた囲み線は判定に入らないため
            # そこを素通りして外まで漏れる。ここで AND を取って止める。
            self_ptr = image.bits(); self_ptr.setsize(nbytes)
            self_arr = np.frombuffer(self_ptr, dtype=np.uint8).reshape(h, w, 4)
            candidate = candidate & (self_arr[:, :, 3] <= line_threshold)
    else:
        if color_judge is None and tolerance <= 0 and judge.pixel(x, y) == fill:
            return
        seed = judge_arr[y, x].copy()
        tol = int(round(max(0, min(100, tolerance)) * 255 / 100))
        if tol <= 0:
            candidate = np.all(judge_arr == seed, axis=2)
        else:
            diff = np.abs(judge_arr.astype(np.int16) - seed.astype(np.int16))
            candidate = diff.max(axis=2) <= tol
    if not candidate[y, x]:
        return
    original_candidate = candidate
    candidate = candidate.astype(np.uint8)
    if close_gap > 0:
        # 線を太らせる = 候補領域を削る。これで数px の途切れが塞がり、
        # 隣の領域へ塗りが漏れ出さなくなる。
        ksize = close_gap * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        closed = cv2.erode(candidate, kernel)
        # 削った結果シード自体が候補から外れると何も塗れなくなるので、
        # その場合は隙間閉じを諦めて元の候補で処理する。
        if closed[y, x]:
            candidate = closed
        else:
            close_gap = 0

    num, labels = cv2.connectedComponents(candidate, connectivity=4)
    seed_label = labels[y, x]
    if seed_label == 0:
        return
    fill_mask = labels == seed_label

    if close_gap > 0:
        # 隙間閉じで削った分だけ塗りを太らせ直し、元の線の手前まで塗る。
        # 太らせ過ぎて線を越えないよう、本来の候補領域でクリップする
        # （自分のレイヤーの線も境界にしているときは、それも含めた候補）。
        ksize = close_gap * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        grown = cv2.dilate(fill_mask.astype(np.uint8), kernel)
        fill_mask = (grown > 0) & original_candidate

    img_ptr = image.bits(); img_ptr.setsize(nbytes)
    img_arr = np.frombuffer(img_ptr, dtype=np.uint8).reshape(h, w, 4)
    if ref_image is not None:
        # 参照モードでは「未塗り(候補)」かつ「既に fill 色ではない」ピクセルのみ書き換える
        fill_color_bgra = np.array([fill_color.blue(), fill_color.green(),
                                     fill_color.red(), fill_color.alpha()], dtype=np.uint8)
        already_filled = np.all(img_arr == fill_color_bgra, axis=2)
        fill_mask = fill_mask & ~already_filled

    img_arr[fill_mask] = (fill_color.blue(), fill_color.green(),
                           fill_color.red(), fill_color.alpha())


def _flood_fill_expanded(image: QImage, x: int, y: int,
                          fill_color: QColor, ref_image: QImage | None,
                          expand: int, close_gap: int = 0,
                          line_threshold: int = LINE_ALPHA_THRESHOLD,
                          include_self: bool = False, tolerance: int = 0,
                          color_judge: QImage | None = None):
    """flood fill 後に expand px だけ塗り範囲を膨張(正)/収縮(負)させる。"""
    if expand == 0:
        _flood_fill(image, x, y, fill_color, ref_image, close_gap, line_threshold,
                    include_self, tolerance, color_judge)
        return

    # fill 前のスナップショット
    before = image.copy()
    _flood_fill(image, x, y, fill_color, ref_image, close_gap, line_threshold,
                include_self, tolerance, color_judge)

    # 「新たに塗られたピクセル」のマスクを numpy で取り出す
    w, h = image.width(), image.height()
    nbytes = h * w * 4
    ptr_after  = image.bits();  ptr_after.setsize(nbytes);  arr_after  = np.frombuffer(ptr_after,  dtype=np.uint8).reshape(h, w, 4).copy()
    ptr_before = before.bits(); ptr_before.setsize(nbytes); arr_before = np.frombuffer(ptr_before, dtype=np.uint8).reshape(h, w, 4).copy()

    fill_rgba = np.array([fill_color.blue(), fill_color.green(),
                          fill_color.red(),  fill_color.alpha()], dtype=np.uint8)

    # 塗ったピクセル = after で fill_color に一致 かつ before と異なる
    painted_mask = (
        np.all(arr_after == fill_rgba, axis=2) &
        ~np.all(arr_before == fill_rgba, axis=2)
    ).astype(np.uint8) * 255

    ksize = abs(expand) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if expand > 0:
        expanded_mask = cv2.dilate(painted_mask, kernel)
    else:
        expanded_mask = cv2.erode(painted_mask, kernel)

    # before を復元してから expanded_mask の範囲に fill_color を適用
    result_arr = arr_before.copy()
    result_arr[expanded_mask > 0] = fill_rgba
    result_img = QImage(result_arr.tobytes(), w, h, w * 4, QImage.Format.Format_ARGB32).copy()

    # CompositionMode_Source で全ピクセルを上書きする（SourceOver だと透明部分が下に透過する）
    p = QPainter(image)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
    p.drawImage(0, 0, result_img)
    p.end()


def _fill_closed_regions_in_area(image: QImage, area_mask: np.ndarray,
                                  fill_color: QColor, ref_image: QImage | None,
                                  expand: int = 0, close_gap: int = 0,
                                  line_threshold: int = LINE_ALPHA_THRESHOLD,
                                  include_self: bool = False) -> int:
    """area_mask（投げなわ選択範囲）内にある、線で閉じた領域だけを自動検出して塗りつぶす。
    area_mask の外周に接している領域（＝閉じていない/範囲外に開いている）は対象外にする。
    expand / close_gap / line_threshold / include_self はバケツ塗り
    （_flood_fill_expanded）と同じ意味。
    戻り値: 実際に塗りつぶした領域の数。"""
    w, h = image.width(), image.height()
    judge = ref_image if ref_image is not None else image

    nbytes = h * w * 4
    ptr = judge.bits(); ptr.setsize(nbytes)
    judge_arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
    # 不透明=線(境界)。ただし白背景の線画は地色を塗れる側に含める。
    free = _line_free_mask(judge_arr, line_threshold)
    if include_self and ref_image is not None:
        # 塗る側のレイヤーに描いてある線も境界にする（バケツと同じ）
        self_ptr = image.bits(); self_ptr.setsize(nbytes)
        self_arr = np.frombuffer(self_ptr, dtype=np.uint8).reshape(h, w, 4)
        free = free & (self_arr[:, :, 3] <= line_threshold)
    true_line = (~free).astype(np.uint8)
    line_mask = true_line
    gap_kernel = None
    if close_gap > 0:
        # 判定用の線だけを太らせて途切れを塞ぐ。塗る範囲は後で元の線まで戻す。
        ksize = close_gap * 2 + 1
        gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        line_mask = cv2.dilate(true_line, gap_kernel)

    # 選択範囲内かつ線でない領域を対象候補としてラベリング
    candidate = ((area_mask > 0) & (line_mask == 0)).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=4)

    # 投げなわ選択範囲の外周ピクセルのうち、その「すぐ外側」が線でない（＝塗りが
    # 投げなわの外へ漏れ出せる）場所だけを「閉じていない」境界とみなす。
    # 外側が線（またはキャンバス外）なら、そこで塞がれているので閉じているとみなしてよい。
    area_bool = area_mask > 0
    padded_area = np.pad(area_bool, 1, mode='constant', constant_values=False)
    padded_line = np.pad(line_mask > 0, 1, mode='constant', constant_values=True)

    def _outside_open(shift_area, shift_line):
        # shift_area: 隣接方向にずらした area_mask（True=そちら側も選択範囲内）
        # shift_line: 同じ方向にずらした line_mask（True=そちら側は線）
        return (~shift_area) & (~shift_line)

    up_open    = _outside_open(padded_area[0:h,   1:w+1], padded_line[0:h,   1:w+1])
    down_open  = _outside_open(padded_area[2:h+2, 1:w+1], padded_line[2:h+2, 1:w+1])
    left_open  = _outside_open(padded_area[1:h+1, 0:w],   padded_line[1:h+1, 0:w])
    right_open = _outside_open(padded_area[1:h+1, 2:w+2], padded_line[1:h+1, 2:w+2])

    border = area_bool & (up_open | down_open | left_open | right_open)

    open_labels = set(np.unique(labels[border]))
    open_labels.discard(0)

    closed_labels = [i for i in range(1, num) if i not in open_labels]
    if not closed_labels:
        return 0

    # 閉じた領域はラベリングの時点で既にピクセル集合が確定しているため、
    # 各領域ごとに scanline flood fill (QImage.pixel/setPixel の逐次呼び出し) を
    # やり直す必要はない。numpy で一括書き込みすることで大キャンバス・多領域でも
    # 高速に処理する（従来の実装は領域数×面積に比例して QImage の低速なピクセル
    # アクセスを繰り返しており、投げなわ内に閉領域が多いと処理落ち・クラッシュしていた）。
    fill_mask = np.isin(labels, closed_labels)
    if gap_kernel is not None:
        # 太らせた線の分だけ痩せた塗りを、元の線の手前まで戻す
        grown = cv2.dilate(fill_mask.astype(np.uint8), gap_kernel) > 0
        fill_mask = grown & (true_line == 0) & area_bool
    if expand:
        ksize = abs(expand) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        op = cv2.dilate if expand > 0 else cv2.erode
        fill_mask = op(fill_mask.astype(np.uint8), kernel) > 0

    ptr = image.bits(); ptr.setsize(nbytes)
    img_arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
    img_arr[fill_mask] = (fill_color.blue(), fill_color.green(),
                           fill_color.red(), fill_color.alpha())

    return len(closed_labels)


# ── ポリゴンマスク ─────────────────────────────────────────────────────────────

def _outline_path_from_mask(mask: np.ndarray) -> QPainterPath:
    """選択マスク（0/1 の 2値）の輪郭を QPainterPath にする。

    穴あきの選択（選択を反転したときなど）は矩形ひとつでは表せないので、
    輪郭を全部拾って複数サブパスとして持たせる。これがないと点線が
    キャンバス外周にしか出ず、どこが選ばれているのか見えない。
    """
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8) * 255,
                                   cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    outline = QPainterPath()
    for cnt in contours:
        if len(cnt) < 2:
            continue
        outline.moveTo(QPointF(float(cnt[0][0][0]), float(cnt[0][0][1])))
        for pt in cnt[1:]:
            outline.lineTo(QPointF(float(pt[0][0]), float(pt[0][1])))
        outline.closeSubpath()
    return outline


def _mask_from_polygon(points: list[QPoint], w: int, h: int) -> QImage:
    mask = QImage(w, h, QImage.Format.Format_ARGB32)
    mask.fill(Qt.GlobalColor.transparent)
    p = QPainter(mask)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QBrush(Qt.GlobalColor.white))
    p.setPen(Qt.PenStyle.NoPen)
    path = QPainterPath()
    if points:
        path.moveTo(QPointF(points[0]))
        for pt in points[1:]:
            path.lineTo(QPointF(pt))
        path.closeSubpath()
    p.drawPath(path)
    p.end()
    return mask


# ── 座標変換 ───────────────────────────────────────────────────────────────────

def _canvas_to_widget_transform(canvas_w: int, canvas_h: int,
                                 zoom: float, rotation: int,
                                 flip_h: bool,
                                 widget_w: int, widget_h: int) -> QTransform:
    """キャンバス座標 → ウィジェット座標。"""
    t = QTransform()
    t.translate(widget_w / 2, widget_h / 2)
    t.rotate(rotation)
    if flip_h:
        t.scale(-1, 1)
    t.scale(zoom, zoom)
    t.translate(-canvas_w / 2, -canvas_h / 2)
    return t


# ── Canvas ─────────────────────────────────────────────────────────────────────

class Canvas(QWidget):
    color_picked = pyqtSignal(QColor)
    status_message = pyqtSignal(str)
    repainted = pyqtSignal()
    brush_size_changed = pyqtSignal(int)   # [ / ] キーでブラシサイズが変わったとき
    eraser_size_changed = pyqtSignal(int)  # Ctrl+ドラッグで消しゴムサイズが変わったとき
    zoom_changed = pyqtSignal(float)
    layer_opacity_changed = pyqtSignal(int)  # 数字キーで不透明度が変わったとき
    tool_shortcut_pressed = pyqtSignal(object)  # Tool — キーボードショートカットでツール切替
    edited = pyqtSignal()  # 絵・レイヤー構造が実際に変更されたとき（未保存マーク用）
    grid_visibility_changed = pyqtSignal(bool)  # 方眼の表示・非表示が変わったとき（メニューのチェックを合わせるため）
    vector_selection_changed = pyqtSignal()  # 選んでいるベクター線が変わったとき（ツールオプションを出し直す）

    # テキスト入力ダイアログをキャンバス内で閉じるため、main から注入する
    ask_text_fn: object = None  # type: ignore  # Callable[[Canvas], None] | None
    _get_onion_images: object = None  # type: ignore  # Callable[[], list[tuple[QImage, float]]] | None

    def __init__(self, layer_stack: LayerStack, parent=None):
        super().__init__(parent)
        self.layer_stack = layer_stack
        self.tool = Tool.PEN
        self._tool_cursor: QCursor | None = None  # ツール固有カーソル（main.pyから設定）
        self.pen_color = QColor(0, 0, 0, 255)
        self.pen_size = 5
        self.eraser_size = 20
        self.zoom = 0.3

        # ブラシ
        self.brush_type: str = BrushType.ROUND
        # 手ブレ補正の強さ（0=なし）。平均をとる点数は「強さ+1」。
        self.stabilization: int = 5
        self._stabilizer = StabilizedBrush(get_brush(BrushType.ROUND),
                                           smooth=self.stabilization + 1)
        # 入り抜き（px で長さ、tip は先端の太さ % ）。ペンと消しゴムで別々に持つ。
        self.taper: dict[str, dict[str, int]] = {
            "pen": {"in": 0, "out": 0, "tip": 0},
            "eraser": {"in": 0, "out": 0, "tip": 0},
        }
        self._taper_stroke: dict | None = None
        # ペンの不透明度(%)。1本の線の中で重なっても濃くならないよう、
        # 線はいったん別の画像に描き、この濃さでレイヤーに重ねる。
        self.pen_opacity: int = 100
        # 透明色で描く（ペンで描いた所が消える。ブラシの形のまま消せる）
        self.pen_transparent: bool = False
        self._pen_buf: dict | None = None
        # Shift+クリックで直線を引くための、前の線の終わり (レイヤーuid, キャンバス座標)
        self._last_stroke_end: tuple[int, QPoint] | None = None

        # 対称定規
        self.symmetry_enabled: bool = False

        # 図形塗りモード: "none"=枠線のみ / "fill"=塗りのみ / "both"=枠線＋塗り
        self.shape_fill: str = "none"
        self.fill_expand: int = 0   # バケツ塗り拡張(正)/縮小(負) px
        self.fill_close_gap: int = 0  # 線画の途切れを塞ぐ px（0=無効）
        # 薄い線をどれだけ拾うか(%)。値が大きいほど薄いピクセルまで「線」とみなし、
        # 「色が薄いせいで途切れ扱いされる」のを防ぐ。
        # 0% = 従来どおり alpha>10 のみ線、100% = alpha が少しでもあれば線。
        self.fill_line_sensitivity: int = 0
        # バケツ塗りの複数参照（クリスタ相当）。"ref"=参照レイヤーのみ、
        # "ref_self"=参照レイヤー＋編集中レイヤー。編集中レイヤーに描いた
        # 囲み線で止めたいときは後者。
        self.fill_reference_mode: str = "ref_self"
        # バケツ塗りの色の誤差(%)。色で判定するとき、これだけ違う色まで同じとみなす。
        self.fill_tolerance: int = 0
        # テキストツールのフォントと文字サイズ(px)。色はペンの色を使う。
        self.text_font_family: str = "Arial"
        self.text_size: int = 40
        # 消しゴムのふちをぼかす
        self.eraser_soft: bool = False
        # Ctrl+ドラッグでブラシサイズを変えている途中の状態
        self._size_drag: dict | None = None
        self.select_mode: str = "select"  # "select" | "transform"

        # ぼかしツール
        self.blur_size: int = 30
        self.blur_strength: float = 0.5  # 0.0〜1.0
        self._blur_brush: BlurBrush = BlurBrush(0.5)

        # view state
        self._rotation = 0
        self._flip_h = False
        self._show_grid = False
        self._grid_size = 100  # canvas px (must stay > 0)
        # 方眼の基準位置。"center" はキャンバス中心に線の交点が来るように、
        # "corner" は左上の隅を起点にマス目を並べる。
        self._grid_origin = "corner"

        # パンニング（Space+ドラッグ）
        self._panning = False
        self._pan_start_widget: QPoint | None = None
        self._scroll_area = None  # main から注入

        # drawing state
        self._last_pos: QPoint | None = None
        self._drawing = False
        self._preview_start: QPoint | None = None
        self._preview_end: QPoint | None = None

        # ストローク中の背景合成キャッシュ（ペン/消しゴム/ぼかし用）
        # レイヤー数が多いと毎フレーム全レイヤー再合成が重くなるため、
        # ストローク開始時に「描画中レイヤー以外」の合成結果を1回だけ作り、
        # ドラッグ中はそれに描画中レイヤーだけを重ねて使い回す。
        self._stroke_bg_cache: QImage | None = None
        self._stroke_layer: object = None

        # selection
        self._selection_rect: QRect | None = None
        # 選択範囲による描画クリップ用（描画前の画像を控えておく）
        self._clip_base_image: QImage | None = None
        self._clip_layer = None
        self._clip_mask: QImage | None = None
        self._lasso_points: list[QPoint] = []
        self._lasso_mask: QImage | None = None
        self._lasso_path_points: list[QPoint] = []  # 確定後の投げ縄パス（表示用）
        self._selection_outline_path: QPainterPath | None = None  # レイヤー形状選択の表示用輪郭（複数パス対応）

        # パスピックモード（アクション用: クリックでパスの点を打つ）
        self._path_pick_active: bool = False
        self._path_pick_points: list[QPoint] = []
        self._path_pick_callback = None  # confirmed_points を受け取るコールバック

        # ベクター描画中の点（キャンバス座標）。離すまでは image に描かず、
        # ここに溜めてプレビュー表示する。None のときは描いていない。
        self._vector_points: list[tuple[float, float]] | None = None
        # 選んでいる線（オブジェクト参照で持つ。分割や削除で番号がずれるため）
        self._vector_selected = None
        # ベクターレイヤーでのペンの役割: "draw"=描く / "select"=線を選んで直す
        self.vector_pen_mode = "draw"
        # 掴んでいるもの: None / ("point", i) / ("handle", i, "in"|"out")
        self._vector_drag = None
        # ハンドルを片側だけ動かすか（Alt 中）。角を作りたいときに使う。
        self._vector_corner_drag = False
        # ドラッグで実際に動かしたか（動かしていなければ履歴を戻す）
        self._vector_drag_moved = False
        # ベクターの消しゴムの効き方: "cut"=交点まで消す / "whole"=線ごと消す
        self.vector_erase_mode = "cut"
        # 1回のドラッグで履歴を積むのは最初の1回だけにするための印
        self._vector_erasing = False

        # transform (floating image)
        self._transform_image: QImage | None = None
        self._transform_rect: QRectF | None = None   # キャンバス座標系での AABB（回転前基準）
        self._transform_orig_rect: QRectF | None = None  # %ゲージ計算の元サイズ
        self._transform_angle: float = 0.0            # 度数、時計回り正
        self._transform_handle: str | None = None
        self._transform_drag_start: QPointF | None = None
        self._transform_rect_start: QRectF | None = None
        self._transform_angle_start: float = 0.0
        # 変形を確定する先のレイヤーを固定することで、変形中にレイヤー切替しても
        # 持ち上げ元レイヤーに正しく書き戻せる
        self._transform_layer: Layer | None = None
        self._transform_erase_rect: QRect | None = None
        self._transform_erase_mask: QImage | None = None
        # ベクターレイヤーを変形しているとき、持ち上げた線の番号。
        # 絵を変形しても線から描き直されて元に戻るので、確定時は点を動かす。
        self._transform_vector_idx: list[int] | None = None
        # 持ち上げた線を除いた絵（プレビュー用）と、そのときの offset
        self._transform_vector_rest: tuple[QImage, int, int] | None = None
        # 変形中の反転（ベクターの点に反映するために覚えておく）
        self._transform_flip: tuple[bool, bool] = (False, False)
        self._transform_pivot: tuple[int, int] = (1, 1)  # (ax, ay) 0=左/上 1=中央 2=右/下
        self._pivot_mode: str = "preset"  # "preset" | "custom"
        self._custom_pivot: QPointF | None = None  # キャンバス座標系の任意ピボット
        self._perspective_mode: bool = False
        self._perspective_corners: list[QPointF] | None = None  # 自由変形時の4隅（キャンバス座標）
        self._perspective_corners_start: list[QPointF] | None = None
        self._perspective_drag_idx: int = -1  # ドラッグ中の隅インデックス
        # メッシュ変形
        self._mesh_mode: bool = False
        self._mesh_div: int = 3  # N×N 分割
        self._mesh_grid: list[list[QPointF]] | None = None  # (N+1)×(N+1) 制御点
        self._mesh_grid_start: list[list[QPointF]] | None = None
        self._mesh_drag_idx: tuple[int, int] = (-1, -1)

        # clipboard
        self._clipboard_image: QImage | None = None
        self._clipboard_offset: QPoint = QPoint(0, 0)

        # 移動ツール用：ドラッグ開始時の元画像とキャンバス座標での開始位置
        self._move_base_image: QImage | None = None
        # 移動を始めた時点のベクター線（ベクターレイヤーのときだけ）
        self._move_base_strokes = None
        self._move_base_pos: QPoint | None = None
        # グループ移動用：子レイヤー全員の元画像リスト
        # (子レイヤー, ベクターなら掴んだ時点の線・ラスターなら None, offset_x, offset_y)
        self._move_group_bases: list[tuple[Layer, list | None, int, int]] | None = None

        # text — クリック後にダイアログを出すので、クリック位置を一時保持する
        self._text_pos: QPoint | None = None

        # 選択範囲内クリック後、ドラッグが始まるまで lift を保留するためのフラグ
        self._lift_pending: bool = False
        self._lift_pending_wp: QPointF | None = None

        # 統合履歴: 各エントリは ("pixel", layer_id, image) または
        #   ("structure", snapshot_dict) の tagged tuple
        self._history: list[tuple] = []
        self._redo_stack: list[tuple] = []

        # カーソル円（ペン・消しゴム用）
        self._cursor_widget_pos: QPointF | None = None

        # 直前の色（Xキーで swap）
        self._prev_color: QColor = QColor(255, 255, 255, 255)

        # Alt 一時スポイト
        self._alt_eyedropper: bool = False
        self._pre_alt_tool: Tool = Tool.PEN

        # マーチングアンツ（選択範囲アニメ）
        self._ant_offset: int = 0
        self._ant_timer = QTimer(self)
        self._ant_timer.setInterval(80)
        self._ant_timer.timeout.connect(self._tick_ants)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._update_size()

    # ── size / view ──────────────────────────────────────────────────────────

    def _update_size(self):
        w = int(self.layer_stack.width * self.zoom)
        h = int(self.layer_stack.height * self.zoom)
        if self._rotation % 180 != 0:
            w, h = h, w
        self.resize(w, h)
        self.update()

    def set_brush(self, brush_type: str):
        self.brush_type = brush_type
        self._stabilizer = StabilizedBrush(get_brush(brush_type),
                                           smooth=self.stabilization + 1)

    def _finish_stabilized_stroke(self):
        """手ブレ補正で遅れている線の終わりを、最後に指があった所まで伸ばす。
        補正を強くするほど遅れが大きく、これが無いと線が短く途切れる。"""
        layer = self.layer_stack.active
        if layer is None or layer.is_group:
            return
        tail = self._stabilizer.drain()
        if not tail:
            return
        if self.tool == Tool.PEN and layer.is_vector:
            if self._vector_points is not None and not self._vector_editing():
                self._vector_points.extend((p.x(), p.y()) for p in tail)
            return
        if self.tool not in (Tool.PEN, Tool.ERASER) or self._last_pos is None:
            return
        for p in tail:
            pt = p.toPoint()
            if pt == self._last_pos:
                continue
            self._stroke_segment(layer, self._last_pos, pt)
            self._last_pos = pt
        self._after_stroke_draw(layer, None)
        self.update()

    def set_stabilization(self, value: int):
        """手ブレ補正の強さを変える（0〜STABILIZATION_MAX）。"""
        self.stabilization = max(0, min(STABILIZATION_MAX, int(value)))
        self._stabilizer.smooth = self.stabilization + 1
        self._stabilizer.reset()

    def _viewport_anchor(self, widget_pos: QPointF | None = None):
        """拡大縮小の基準点を「ビューポート上の位置」と「その下にある
        キャンバス座標」の組で返す。スクロールエリアが無ければ None。

        widget_pos を渡すとその点（Ctrl+ホイールのカーソル位置）、
        省略すると今見えている範囲の中心が基準になる。
        """
        if self._scroll_area is None:
            return None
        vp = self._scroll_area.viewport()
        if widget_pos is None:
            # ビューポート中心をこのウィジェットの座標系へ変換する
            center_vp = QPointF(vp.width() / 2.0, vp.height() / 2.0)
            local = self.mapFrom(vp, center_vp.toPoint())
            widget_pos = QPointF(local)
        else:
            center_vp = QPointF(self.mapTo(vp, widget_pos.toPoint()))
        canvas_pt = self._w2c().map(widget_pos)
        return center_vp, canvas_pt

    def _restore_anchor(self, anchor):
        """_viewport_anchor で覚えた点が、ズーム後も同じ位置に来るように
        スクロールバーを合わせる。これをしないと拡大縮小のたびに
        表示が左上へ寄って、描いていた場所を見失う。"""
        if anchor is None or self._scroll_area is None:
            return
        center_vp, canvas_pt = anchor
        pad = self._scroll_area.widget()
        # resize は非同期にイベントとして届くため、この時点ではまだ
        # _CanvasPad が新しいサイズになっておらず、スクロールバーの
        # maximum も古い値のままになる。そのまま setValue すると
        # 旧 maximum で頭打ちされてしまうので、先に確定させる。
        if pad is not None and hasattr(pad, "_relayout"):
            pad._relayout()
        vp = self._scroll_area.viewport()
        # ズーム後、その canvas 座標がウィジェットのどこに来たか
        new_widget = self._c2w().map(canvas_pt)
        # スクロール内容（_CanvasPad）上の座標へ
        pad_pt = self.mapTo(self._scroll_area.widget(), new_widget.toPoint())
        hbar = self._scroll_area.horizontalScrollBar()
        vbar = self._scroll_area.verticalScrollBar()
        hbar.setValue(int(round(pad_pt.x() - center_vp.x())))
        vbar.setValue(int(round(pad_pt.y() - center_vp.y())))

    def set_zoom(self, zoom: float, anchor_pos: QPointF | None = None):
        """ズームを変更する。今見えている範囲の中心（anchor_pos を渡した
        場合はその点）が動かないようにスクロール位置も合わせる。"""
        anchor = self._viewport_anchor(anchor_pos)
        self.zoom = max(MIN_ZOOM, zoom)
        self._update_size()
        # _update_size → resize → _CanvasPad._relayout が走ってから
        # スクロール位置を決める必要があるため、ここで確定させる
        self._restore_anchor(anchor)
        self.zoom_changed.emit(self.zoom)

    def set_rotation(self, degrees: int):
        self._rotation = degrees % 360
        self._update_size()

    def rotate_cw(self):
        self.set_rotation(self._rotation + 90)

    def rotate_ccw(self):
        self.set_rotation(self._rotation - 90)

    def reset_rotation(self):
        self.set_rotation(0)

    def toggle_flip_h(self):
        self._flip_h = not self._flip_h
        self.update()

    def is_locked(self, layer) -> bool:
        """ロックしたレイヤーかどうか。親グループがロックされていれば中身もロック扱い。

        描画・消しゴム・塗りつぶし・移動・変形はすべてこの先を通るので、
        ここで止めればすべて守れる。
        """
        if layer is None:
            return False
        if getattr(layer, "locked", False):
            return True
        path = self.layer_stack.path_of(layer)
        if not path:
            return False
        items = self.layer_stack.layers
        for idx in path[:-1]:
            if idx >= len(items):
                return False
            parent = items[idx]
            if getattr(parent, "locked", False):
                return True
            items = parent.children
        return False

    def toggle_grid(self):
        self.set_grid_visible(not self._show_grid)

    def set_grid_size(self, size: int):
        self._grid_size = max(1, size)  # 0除算・無限ループ防止
        self.update()

    def set_grid_origin(self, origin: str):
        """方眼の基準を "center"（キャンバス中心）か "corner"（左上の隅）に。"""
        self._grid_origin = "center" if origin == "center" else "corner"
        self.update()

    def set_grid_visible(self, visible: bool):
        visible = bool(visible)
        if visible == self._show_grid:
            return
        self._show_grid = visible
        self.grid_visibility_changed.emit(visible)
        self.update()

    # ── coordinate conversion ────────────────────────────────────────────────

    def _c2w(self) -> QTransform:
        """キャンバス座標 → ウィジェット座標。"""
        return _canvas_to_widget_transform(
            self.layer_stack.width, self.layer_stack.height,
            self.zoom, self._rotation, self._flip_h,
            self.width(), self.height())

    def _w2c(self) -> QTransform:
        """ウィジェット座標 → キャンバス座標。"""
        t, ok = self._c2w().inverted()
        return t if ok else QTransform()

    def _widget_to_canvas(self, p: QPoint) -> QPoint:
        mapped = self._w2c().map(QPointF(p))
        # int() は 0 に向かって切り捨てるので、キャンバスの左・上の外
        # （-0.5 など）が 0 列目扱いになる。floor で正しいピクセルにする。
        return QPoint(math.floor(mapped.x()), math.floor(mapped.y()))

    def _painter_transform(self, p: QPainter):
        p.setTransform(self._c2w())

    # ── history (per layer) ──────────────────────────────────────────────────

    @staticmethod
    def _collect_leaf_layers(group) -> list:
        """グループ内の通常レイヤーを再帰的に収集する。"""
        result = []
        for c in group.children:
            if c.is_group:
                result.extend(Canvas._collect_leaf_layers(c))
            else:
                result.append(c)
        return result

    def _layer_id(self) -> int | None:
        layer = self.layer_stack.active
        return layer.uid if layer and not layer.is_group else None

    def _begin_stroke_cache(self, layer) -> None:
        """ストローク開始時に「描画中レイヤー以外」の合成結果をキャッシュする。
        クリッピング等が絡み安全に省略できない場合はキャッシュしない
        （その場合 paintEvent は毎回フル合成にフォールバックする）。"""
        if self.layer_stack.can_fast_preview(layer):
            self._stroke_layer = layer
            self._stroke_bg_cache = self.layer_stack.composite(skip=layer)
        else:
            self._stroke_layer = None
            self._stroke_bg_cache = None

    def _end_stroke_cache(self) -> None:
        self._stroke_layer = None
        self._stroke_bg_cache = None

    # ── 選択範囲による描画のマスク ────────────────────────────────────────────

    def _begin_clip_to_selection(self, layer) -> None:
        """選択範囲があるとき、描画前のレイヤー画像を控えておく。

        ペン・消しゴム・バケツ塗りは選択範囲を見ずにレイヤー全体へ描いてしまう。
        ツールごとに描画処理へマスクを渡すのは経路が多く漏れやすいので、
        「描いた後で選択範囲の外を元に戻す」方式で一律にクリップする。
        """
        self._clip_base_image = None
        self._clip_layer = None
        if layer is None or layer.is_group:
            return
        # 透明ピクセルのロックも「描く前の透明度に戻す」ので同じ控えを使う。
        if not self._selection_rect and not self._is_alpha_locked(layer):
            return
        self._clip_layer = layer
        self._clip_base_image = layer.image.copy()

    @staticmethod
    def _is_alpha_locked(layer) -> bool:
        return (layer is not None and not layer.is_group
                and not getattr(layer, "is_vector", False)
                and bool(getattr(layer, "alpha_locked", False)))

    def _apply_clip_to_selection(self, dirty: QRect | None = None) -> None:
        """選択範囲の外を、描画前の状態に戻す。

        透明ピクセルがロックされたレイヤーでは、続けて透明度も描く前に戻す。
        dirty は今回描き変えた範囲（レイヤーのローカル座標）。分かっていれば
        透明度の戻しをその範囲だけで行い、大きなレイヤーでも軽くする。
        """
        layer = self._clip_layer
        base = self._clip_base_image
        if layer is None or base is None:
            return
        self._clip_to_selection_area(layer, base)
        if self._is_alpha_locked(layer):
            self._restore_alpha(layer, base, dirty)

    @staticmethod
    def _restore_alpha(layer, base: QImage, dirty: QRect | None = None) -> None:
        """描いた後の色を残しつつ、透明度だけを描く前のものに戻す。

        絵がなかった所（透明）は透明のまま、絵があった所は新しい色で塗られる。
        消しゴムで薄くなった所は元に戻す（ロック中は透明にできない）。
        """
        img = layer.image
        if base.size() != img.size():
            return
        if img.format() != QImage.Format.Format_ARGB32:
            img = img.convertToFormat(QImage.Format.Format_ARGB32)
            layer.image = img
        if base.format() != QImage.Format.Format_ARGB32:
            base = base.convertToFormat(QImage.Format.Format_ARGB32)
        w, h = img.width(), img.height()
        r = QRect(0, 0, w, h)
        if dirty is not None:
            r = dirty.intersected(r)
            if r.isEmpty():
                return
        ptr = img.bits(); ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
        bptr = base.constBits(); bptr.setsize(h * w * 4)
        barr = np.frombuffer(bptr, dtype=np.uint8).reshape(h, w, 4)
        ys = slice(r.top(), r.bottom() + 1)
        xs = slice(r.left(), r.right() + 1)
        a = arr[ys, xs]
        b = barr[ys, xs]
        # 新しい絵が描かれなかった（消された）所は元の色を使う
        lost = (a[:, :, 3] < b[:, :, 3])
        a[lost] = b[lost]
        a[:, :, 3] = b[:, :, 3]
        a[b[:, :, 3] == 0] = 0

    def _segment_dirty(self, a: QPoint, b: QPoint) -> QRect | None:
        """a→b を描いたときに変わりうる範囲。左右対称のときは全体（None）。"""
        if self.symmetry_enabled:
            return None
        m = int(self._tool_size()) + 4
        return QRect(a, b).normalized().adjusted(-m, -m, m, m)

    def _clip_to_selection_area(self, layer, base: QImage) -> None:
        sel = self._clip_sel_rect_local(layer)
        if sel is None:
            return
        # 「元画像を全面に描き戻し、そこへ選択範囲の中だけ新しい絵を戻す」。
        # 選択範囲が矩形でない（投げなわ）場合もマスクで同じように扱える。
        restored = base.copy()
        p = QPainter(restored)
        if self._clip_mask is not None:
            region = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
            region.fill(Qt.GlobalColor.transparent)
            rp = QPainter(region)
            rp.drawImage(0, 0, layer.image)
            rp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            rp.drawImage(-getattr(layer, 'offset_x', 0),
                         -getattr(layer, 'offset_y', 0), self._clip_mask)
            rp.end()
            # 投げなわの中は「消しゴムで透明になった」場合も反映する必要が
            # あるので、いったんマスク内を空にしてから描き戻す。
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            p.drawImage(-getattr(layer, 'offset_x', 0),
                        -getattr(layer, 'offset_y', 0), self._clip_mask)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            p.drawImage(0, 0, region)
        else:
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            p.drawImage(sel, layer.image, sel)
        p.end()
        layer.image = restored

    def _clip_sel_rect_local(self, layer) -> QRect | None:
        """選択範囲をレイヤーのローカル座標へ直し、画像内に収めて返す。"""
        if not self._selection_rect:
            return None
        r = self._selection_rect.translated(-getattr(layer, 'offset_x', 0),
                                            -getattr(layer, 'offset_y', 0))
        r = r.intersected(QRect(0, 0, layer.image.width(), layer.image.height()))
        return r if not r.isEmpty() else None

    def _shift_clip_base(self, layer, shift) -> None:
        """ストローク中にレイヤーが広がったとき、控えた画像も同じだけずらす。

        これをしないと元画像とレイヤー画像の座標がずれ、範囲外を戻すときに
        絵が飛んだ位置に貼り付いてしまう。
        """
        if self._clip_base_image is None:
            return
        self._clip_base_image = _shifted_image(self._clip_base_image,
                                               layer.image.size(), shift)

    @staticmethod
    def _entry_bytes(entry) -> int:
        """履歴エントリ1件が保持している画像バイト数のおおよその合計。"""
        if entry[0] == "pixel":
            img = entry[2]
            return img.sizeInBytes() if hasattr(img, "sizeInBytes") else 0

        if entry[0] == "vector":
            # 線は点の並びなので画像よりずっと軽い。1点あたり 24 バイト見当。
            return sum(len(s.points) * 24 for s in entry[2])

        def _snap_bytes(snap) -> int:
            if snap.get("type") == "group":
                return sum(_snap_bytes(c) for c in snap.get("children", []))
            img = snap.get("image")
            return img.sizeInBytes() if hasattr(img, "sizeInBytes") else 0

        return sum(_snap_bytes(s) for s in entry[1].get("layers", []))

    def _trim_history(self):
        """履歴を件数と総メモリ量の両方で打ち切る。
        件数だけで見ていると大きなキャンバスでは 50 件で 1GB を超え、
        メモリ不足でアプリごと落ちるため、バイト数でも制限する。"""
        while len(self._history) > HISTORY_LIMIT:
            self._history.pop(0)
        total = sum(self._entry_bytes(e) for e in self._history)
        while len(self._history) > HISTORY_MIN_ENTRIES and total > HISTORY_MEMORY_LIMIT:
            total -= self._entry_bytes(self._history.pop(0))

    def _save_history(self):
        lid = self._layer_id()
        layer = self.layer_stack.active
        if lid is None or layer is None:
            return
        ox = getattr(layer, 'offset_x', 0)
        oy = getattr(layer, 'offset_y', 0)
        if layer.is_vector:
            # ベクターは絵ではなく線を控える。絵は線から作り直せるので、
            # こちらのほうが軽く、太さや形の変更も正しく戻せる。
            self._history.append(("vector", lid, layer.copy_strokes(), ox, oy))  # type: ignore
        else:
            self._history.append(("pixel", lid, layer.image.copy(), ox, oy))  # type: ignore
        self._redo_stack.clear()
        self._trim_history()
        self.edited.emit()

    def _snapshot_layer(self, lyr) -> dict:
        if lyr.is_group:
            return {
                "type": "group", "uid": lyr.uid, "name": lyr.name, "visible": lyr.visible,
                "pass_through": getattr(lyr, "pass_through", False),
                "opacity": lyr.opacity, "clipping": lyr.clipping,
                "reference": lyr.reference, "locked": lyr.locked,
                "collapsed": lyr.collapsed,
                "children": [self._snapshot_layer(c) for c in lyr.children],
                "_w": lyr._w, "_h": lyr._h,
            }
        return {
            "type": "vector" if lyr.is_vector else "layer",
            # ベクターは線そのものを控える。絵だけ戻しても、次の描き直しで
            # 元の線から作り直されてしまい、元に戻したことにならない。
            "strokes": lyr.copy_strokes() if lyr.is_vector else None,
            "canvas_size": ((lyr._canvas_w, lyr._canvas_h)
                            if lyr.is_vector else None),
            "uid": lyr.uid, "name": lyr.name, "visible": lyr.visible,
            "opacity": lyr.opacity, "clipping": lyr.clipping,
            "reference": lyr.reference, "locked": lyr.locked,
            "alpha_locked": getattr(lyr, "alpha_locked", False),
            "image": lyr.image.copy(),
            "blend_mode": lyr.blend_mode,
            "offset_x": lyr.offset_x, "offset_y": lyr.offset_y,
            "border_enabled": lyr.border_enabled, "border_size": lyr.border_size,
            "border_color": QColor(lyr.border_color),
            "shadow_enabled": lyr.shadow_enabled, "shadow_color": QColor(lyr.shadow_color),
            "shadow_offset_x": lyr.shadow_offset_x, "shadow_offset_y": lyr.shadow_offset_y,
            "shadow_blur": lyr.shadow_blur, "shadow_strength": lyr.shadow_strength,
            "glow_enabled": lyr.glow_enabled, "glow_color": QColor(lyr.glow_color),
            "glow_size": lyr.glow_size, "glow_strength": lyr.glow_strength,
            "blur_enabled": lyr.blur_enabled, "blur_radius": lyr.blur_radius,
            "blur_strength": lyr.blur_strength,
            "hsl_enabled": lyr.hsl_enabled, "hsl_hue": lyr.hsl_hue,
            "hsl_saturation": lyr.hsl_saturation, "hsl_lightness": lyr.hsl_lightness,
        }

    def _restore_layer(self, snap: dict):
        from layer import GroupLayer
        if snap["type"] == "group":
            g = GroupLayer(snap["name"], snap["_w"], snap["_h"])
            g.visible = snap["visible"]; g.opacity = snap["opacity"]
            g.clipping = snap["clipping"]; g.reference = snap["reference"]
            g.locked = snap.get("locked", False)
            g.collapsed = snap["collapsed"]
            g.pass_through = snap.get("pass_through", False)
            g.children = [self._restore_layer(c) for c in snap["children"]]
            if "uid" in snap:
                g.uid = snap["uid"]
            return g
        if snap.get("type") == "vector":
            from vector import VectorLayer
            cw, ch = snap.get("canvas_size") or (snap["image"].width(),
                                                 snap["image"].height())
            lyr = VectorLayer(snap["name"], cw, ch)
            lyr.strokes = [s.copy() for s in (snap.get("strokes") or [])]
            lyr.mark_dirty()
        else:
            lyr = Layer(snap["name"], snap["image"].width(), snap["image"].height())
            lyr.image = snap["image"].copy()
        for k in ("visible", "opacity", "clipping", "reference", "locked", "blend_mode",
                  "offset_x", "offset_y",
                  "border_enabled", "border_size", "border_color",
                  "shadow_enabled", "shadow_color", "shadow_offset_x", "shadow_offset_y",
                  "shadow_blur", "shadow_strength", "glow_enabled", "glow_color",
                  "glow_size", "glow_strength", "blur_enabled", "blur_radius",
                  "blur_strength", "hsl_enabled", "hsl_hue", "hsl_saturation", "hsl_lightness"):
            setattr(lyr, k, snap[k])
        lyr.alpha_locked = snap.get("alpha_locked", False)
        # 作り直しても同じ番号にしておけば、構造の undo/redo をまたいでも
        # 描画の履歴がそのレイヤーを指し続ける。
        if "uid" in snap:
            lyr.uid = snap["uid"]
        return lyr

    def save_structure_history(self):
        ls = self.layer_stack
        snap = {
            "layers": [self._snapshot_layer(l) for l in ls.layers],
            "active_path": list(ls.active_path),
            # キャンバスサイズも一緒に控える。サイズ変更を元に戻したとき、
            # 画像だけ旧サイズに戻ってキャンバスと食い違うのを防ぐ。
            "canvas_size": (ls.width, ls.height),
        }
        self._history.append(("structure", snap))
        self._redo_stack.clear()
        self._trim_history()
        self.edited.emit()

    _PROP_COALESCE_SEC = 1.0

    def save_property_history(self, key) -> None:
        """レイヤー属性（不透明度・表示・名前など）を変える直前の状態を控える。

        スライダーを動かすと値が何十回も変わるので、同じ属性の変更が
        続いている間（間に他の操作が入らず、間隔が短い）は1つにまとめる。
        """
        now = time.monotonic()
        last = getattr(self, "_last_prop", None)
        if (last is not None and last[0] == key and self._history
                and self._history[-1] is last[1]
                and now - last[2] < self._PROP_COALESCE_SEC):
            self._last_prop = (key, last[1], now)
            return
        self.save_structure_history()
        self._last_prop = (key, self._history[-1], now)

    def _apply_structure_snapshot(self, snap: dict):
        ls = self.layer_stack
        size = snap.get("canvas_size")
        if size and (ls.width, ls.height) != tuple(size):
            ls.width, ls.height = int(size[0]), int(size[1])
            self._update_size()
        ls.layers = [self._restore_layer(s) for s in snap["layers"]]
        path = list(snap.get("active_path") or [snap.get("active_index", 0)])
        if path:
            path[0] = min(path[0], max(0, len(ls.layers) - 1))
        ls.active_path = path

    def _all_layer_ids(self) -> set[int]:
        ids: set[int] = set()
        def _collect(items):
            for item in items:
                ids.add(item.uid)
                if item.is_group:
                    _collect(item.children)
        _collect(self.layer_stack.layers)
        return ids

    def purge_orphan_history(self):
        live = self._all_layer_ids()

        def _alive(e) -> bool:
            if e[0] == "structure":
                return True
            # pixel と vector はどちらもレイヤーに結びついているので、
            # そのレイヤーが残っているものだけ残す。
            return e[1] in live

        self._history = [e for e in self._history if _alive(e)]
        self._redo_stack = [e for e in self._redo_stack if _alive(e)]

    def _find_layer_by_id(self, layer_id: int) -> Layer | None:
        def _search(items):
            for item in items:
                if item.uid == layer_id:
                    return item
                if item.is_group:
                    found = _search(item.children)
                    if found is not None:
                        return found
            return None
        return _search(self.layer_stack.layers)

    def _swap_vector_entry(self, entry, opposite: list) -> None:
        """ベクター履歴を1件適用し、今の線を反対側のスタックへ積む。

        undo と redo で向きが違うだけなので、ここにまとめておく。
        """
        _, lid, strokes, old_ox, old_oy = entry
        layer = self._find_layer_by_id(lid)
        if layer is None or not layer.is_vector:
            return
        sel_idx = None
        if self._vector_selected is not None:
            sel_idx = next((i for i, s in enumerate(layer.strokes)  # type: ignore
                            if s is self._vector_selected), None)
        opposite.append(("vector", lid, layer.copy_strokes(),
                         getattr(layer, 'offset_x', 0),
                         getattr(layer, 'offset_y', 0)))
        layer.strokes = strokes  # type: ignore
        layer.mark_dirty()  # type: ignore
        # 線は複製に置き換わるので、選択は古い線を指したままになる。本数が
        # 変わらない（太さ・色・形の変更）ときは同じ位置の線を選び直し、
        # 線が増減したときは選択を外す。
        if self._vector_selected is not None:
            if sel_idx is not None and len(strokes) == len(opposite[-1][2]):
                self._vector_selected = strokes[sel_idx]
            else:
                self._vector_selected = None
                self._vector_drag = None
            self.vector_selection_changed.emit()

    def undo(self):
        if self._transform_image:
            self.cancel_transform()
            return
        self._step_history(self._history, self._redo_stack)

    def redo(self):
        self._step_history(self._redo_stack, self._history)

    def _step_history(self, src: list, dst: list) -> None:
        """src の末尾を1件適用し、今の状態を dst へ積む。undo と redo の共通部。

        指すレイヤーがもう無い履歴（削除したレイヤーへの描画など）は
        黙って捨てて次へ進む。そこで止まると、何も起きない undo が
        挟まって「効かない」ように見えるため。
        """
        while src:
            entry = src.pop()
            if entry[0] == "vector":
                layer = self._find_layer_by_id(entry[1])
                if layer is None or not layer.is_vector:
                    continue
                self._swap_vector_entry(entry, dst)
            elif entry[0] == "pixel":
                lid, img = entry[1], entry[2]
                old_ox = entry[3] if len(entry) > 3 else 0
                old_oy = entry[4] if len(entry) > 4 else 0
                layer = self._find_layer_by_id(lid)
                if layer is None or layer.is_group or layer.is_vector:
                    continue
                dst.append(("pixel", lid, layer.image.copy(),
                            getattr(layer, 'offset_x', 0), getattr(layer, 'offset_y', 0)))
                layer.image = img
                layer.offset_x = old_ox
                layer.offset_y = old_oy
            elif entry[0] == "structure":
                _, snap = entry
                dst.append(("structure", {
                    "layers": [self._snapshot_layer(l) for l in self.layer_stack.layers],
                    "active_path": list(self.layer_stack.active_path),
                    "canvas_size": (self.layer_stack.width, self.layer_stack.height),
                }))
                self._apply_structure_snapshot(snap)
                # 作り直したレイヤーには選択中のベクター線は無い
                if self._vector_selected is not None:
                    self._vector_selected = None
                    self._vector_drag = None
                    self.vector_selection_changed.emit()
                if self._on_structure_restored:
                    self._on_structure_restored()
            break
        self.update()
        self.edited.emit()

    # Callback for main.py to refresh UI after structure undo/redo
    _on_structure_restored: object = None  # type: ignore

    # ── paint ───────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.save()
        self._painter_transform(p)

        # 透明チェッカーボード
        self._draw_checkerboard(p)

        # オニオンスキン（前のフレームを半透明表示）
        if self._get_onion_images:
            for onion_img, onion_opacity in self._get_onion_images():
                p.setOpacity(onion_opacity)
                p.drawImage(0, 0, onion_img)
            p.setOpacity(1.0)

        p.drawImage(0, 0, self._composite_with_floating())

        if self._show_grid:
            self._draw_grid(p)

        if self.symmetry_enabled:
            cx = self.layer_stack.width // 2
            p.setPen(QPen(QColor(100, 160, 255, 180), 1, Qt.PenStyle.DashLine))
            p.drawLine(cx, 0, cx, self.layer_stack.height)

        if self._preview_start is not None and self._preview_end is not None:
            self._draw_shape_preview(p)

        if self._lasso_points:
            lasso_path = QPainterPath()
            lasso_path.moveTo(QPointF(self._lasso_points[0]))
            for pt in self._lasso_points[1:]:
                lasso_path.lineTo(QPointF(pt))
            self._draw_marching_ants(p, path=lasso_path)
        elif self._selection_outline_path is not None and self._selection_rect:
            self._draw_marching_ants(p, path=self._selection_outline_path)
        elif self._lasso_path_points and self._selection_rect:
            lasso_path = QPainterPath()
            lasso_path.moveTo(QPointF(self._lasso_path_points[0]))
            for pt in self._lasso_path_points[1:]:
                lasso_path.lineTo(QPointF(pt))
            lasso_path.closeSubpath()
            self._draw_marching_ants(p, path=lasso_path)
        elif self._selection_rect:
            self._draw_marching_ants(p, rect=self._selection_rect)

        if self._path_pick_active and self._path_pick_points:
            pen = QPen(QColor(255, 120, 0, 220), 2)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            path = QPainterPath()
            path.moveTo(QPointF(self._path_pick_points[0]))
            for pt in self._path_pick_points[1:]:
                path.lineTo(QPointF(pt))
            p.drawPath(path)
            p.setBrush(QColor(255, 120, 0, 220))
            for pt in self._path_pick_points:
                p.drawEllipse(QPointF(pt), 4, 4)

        # 確定前のベクター線
        self._draw_vector_preview(p)

        p.restore()

        # キャンバス境界の枠線（ウィジェット座標系で描く）
        cw, ch = self.layer_stack.width, self.layer_stack.height
        c2w = self._c2w()
        corners = [
            c2w.map(QPointF(0, 0)), c2w.map(QPointF(cw, 0)),
            c2w.map(QPointF(cw, ch)), c2w.map(QPointF(0, ch)),
        ]
        border_path = QPainterPath()
        border_path.moveTo(corners[0])
        for pt in corners[1:]:
            border_path.lineTo(pt)
        border_path.closeSubpath()
        p.setPen(QPen(QColor(0, 0, 0, 80), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(border_path)

        # ハンドルはウィジェット座標で描く
        if self._transform_image and self._transform_rect:
            self._draw_transform_handles(p)

        # ベクターの制御点も同じくウィジェット座標で（拡大しても同じ大きさ）
        self._draw_vector_handles(p)

        # ブラシカーソル円
        if self._cursor_widget_pos is not None and self.tool in (Tool.PEN, Tool.ERASER, Tool.BLUR):
            self._draw_cursor_circle(p)

        p.end()
        self.repainted.emit()

    _checker_tile: QPixmap | None = None

    def _draw_checkerboard(self, p: QPainter):
        """キャンバス領域に透明を示すチェッカーボードを描く。"""
        w, h = self.layer_stack.width, self.layer_stack.height
        if Canvas._checker_tile is None:
            sz = 16
            tile = QPixmap(sz * 2, sz * 2)
            tp = QPainter(tile)
            tp.fillRect(0, 0, sz * 2, sz * 2, QColor(255, 255, 255))
            tp.fillRect(0, 0, sz, sz, QColor(204, 204, 204))
            tp.fillRect(sz, sz, sz, sz, QColor(204, 204, 204))
            tp.end()
            Canvas._checker_tile = tile
        p.save()
        p.setBrush(QBrush(Canvas._checker_tile))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRect(0, 0, w, h)
        p.restore()

    def _sync_ant_timer(self):
        """選択範囲があればタイマーを動かし、なければ止める。"""
        has_sel = self._selection_rect is not None or bool(self._lasso_points) or bool(self._lasso_path_points)
        if has_sel and not self._ant_timer.isActive():
            self._ant_timer.start()
        elif not has_sel and self._ant_timer.isActive():
            self._ant_timer.stop()

    def _tick_ants(self):
        self._ant_offset = (self._ant_offset + 1) % 16
        self.update()

    def _draw_marching_ants(self, p: QPainter, rect: QRect | None = None,
                             path: QPainterPath | None = None):
        """マーチングアンツ（アニメする点線）で選択範囲を描く。"""
        for color, width, offset in (
            (QColor(255, 255, 255, 220), 2, 0),
            (QColor(0, 0, 0, 220), 1, 0),
        ):
            pen = QPen(color, width, Qt.PenStyle.DashLine)
            pen.setDashOffset(self._ant_offset + offset)
            pen.setDashPattern([4, 4])
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            if rect is not None:
                p.drawRect(rect)
            if path is not None:
                p.drawPath(path)

    def _draw_grid(self, p: QPainter):
        # grid_size は常に >=1 が保証されている
        p.setPen(QPen(GRID_COLOR, 1))
        cw, ch = self.layer_stack.width, self.layer_stack.height
        g = self._grid_size
        for x in _grid_lines(cw, g, self._grid_origin):
            p.drawLine(x, 0, x, ch)
        for y in _grid_lines(ch, g, self._grid_origin):
            p.drawLine(0, y, cw, y)

    def _draw_shape_preview(self, p: QPainter):
        is_selection = self.tool == Tool.SELECT_RECT
        r = QRect(self._preview_start, self._preview_end).normalized()
        if is_selection:
            p.setPen(QPen(SELECTION_COLOR, 1, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
            return
        pen = QPen(self.pen_color, self.pen_size)
        if self.shape_fill == "fill":
            p.setPen(Qt.PenStyle.NoPen)
        else:
            p.setPen(pen)
        if self.shape_fill in ("fill", "both"):
            p.setBrush(QBrush(self.pen_color))
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
        if self.tool == Tool.RECT:
            p.drawRect(r)
        elif self.tool == Tool.ELLIPSE:
            p.drawEllipse(r)
        elif self.tool == Tool.LINE:
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawLine(self._preview_start, self._preview_end)

    def _restore_tool_cursor(self):
        """ツール固有のカーソルに復帰する。"""
        if self.tool in (Tool.PEN, Tool.ERASER):
            self.setCursor(Qt.CursorShape.BlankCursor)
        elif self._tool_cursor:
            self.setCursor(self._tool_cursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    # ── transform helpers ────────────────────────────────────────────────────

    def _pivot_point(self) -> QPointF:
        """現在のピボット設定に基づく基準点（キャンバス座標系）。"""
        r = self._transform_rect
        if not r:
            return QPointF(0, 0)
        if self._pivot_mode == "custom" and self._custom_pivot is not None:
            return self._custom_pivot
        ax, ay = self._transform_pivot
        px = r.left() + r.width() * ax / 2.0
        py = r.top() + r.height() * ay / 2.0
        return QPointF(px, py)

    def _transform_matrix(self) -> QTransform:
        """回転込みの変形行列（キャンバス座標系）。"""
        if not self._transform_rect:
            return QTransform()
        pv = self._pivot_point()
        t = QTransform()
        t.translate(pv.x(), pv.y())
        t.rotate(self._transform_angle)
        t.translate(-pv.x(), -pv.y())
        return t

    def _transform_corners_canvas(self) -> list[QPointF]:
        """変形後の四隅座標（キャンバス座標系）。"""
        if self._perspective_corners:
            return list(self._perspective_corners)
        if not self._transform_rect:
            return []
        r = self._transform_rect
        corners = [
            QPointF(r.left(), r.top()), QPointF(r.right(), r.top()),
            QPointF(r.right(), r.bottom()), QPointF(r.left(), r.bottom()),
        ]
        tm = self._transform_matrix()
        return [tm.map(c) for c in corners]

    def _transform_corners_widget(self) -> list[QPointF]:
        c2w = self._c2w()
        return [c2w.map(c) for c in self._transform_corners_canvas()]

    def _warp_perspective_image(self) -> tuple[QImage, int, int] | None:
        """自由変形モードでcv2.warpPerspectiveを使って変形画像を生成。(image, offset_x, offset_y)を返す。"""
        if not self._perspective_corners or not self._transform_image or not self._transform_rect:
            return None
        img = self._transform_image
        w, h = img.width(), img.height()
        if w < 1 or h < 1:
            return None
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([[c.x(), c.y()] for c in self._perspective_corners])

        bb_min_x = math.floor(min(c.x() for c in self._perspective_corners))
        bb_min_y = math.floor(min(c.y() for c in self._perspective_corners))
        bb_max_x = math.ceil(max(c.x() for c in self._perspective_corners))
        bb_max_y = math.ceil(max(c.y() for c in self._perspective_corners))
        out_w = max(bb_max_x - bb_min_x, 1)
        out_h = max(bb_max_y - bb_min_y, 1)

        offset_dst = dst_pts - np.float32([bb_min_x, bb_min_y])
        M = cv2.getPerspectiveTransform(src_pts, offset_dst)

        bits = img.bits()
        bits.setsize(img.sizeInBytes())
        arr = np.frombuffer(bits, dtype=np.uint8).reshape(h, w, 4).copy()
        bgra = arr  # QImage ARGB32 on little-endian = BGRA in numpy

        warped = cv2.warpPerspective(bgra, M, (out_w, out_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0, 0))
        result = QImage(warped.data, out_w, out_h, warped.strides[0],
                        QImage.Format.Format_ARGB32).copy()
        return result, bb_min_x, bb_min_y

    def _init_mesh_grid(self):
        """変形矩形から均等分割のメッシュ格子を初期化する。"""
        if not self._transform_rect:
            return
        r = self._transform_rect
        n = self._mesh_div
        rows = n + 1
        cols = n + 1
        self._mesh_grid = []
        for row in range(rows):
            line = []
            for col in range(cols):
                x = r.left() + r.width() * col / n
                y = r.top() + r.height() * row / n
                line.append(QPointF(x, y))
            self._mesh_grid.append(line)

    def _warp_mesh_image(self) -> tuple[QImage, int, int] | None:
        """メッシュ変形: 各セルを個別にwarpPerspectiveして合成する。"""
        if not self._mesh_grid or not self._transform_image or not self._transform_rect:
            return None
        img = self._transform_image
        sw, sh = img.width(), img.height()
        if sw < 1 or sh < 1:
            return None
        r = self._transform_rect
        n = self._mesh_div
        grid = self._mesh_grid

        all_pts = [p for row in grid for p in row]
        bb_min_x = math.floor(min(p.x() for p in all_pts))
        bb_min_y = math.floor(min(p.y() for p in all_pts))
        bb_max_x = math.ceil(max(p.x() for p in all_pts))
        bb_max_y = math.ceil(max(p.y() for p in all_pts))
        out_w = max(bb_max_x - bb_min_x, 1)
        out_h = max(bb_max_y - bb_min_y, 1)

        bits = img.bits()
        bits.setsize(img.sizeInBytes())
        arr = np.frombuffer(bits, dtype=np.uint8).reshape(sh, sw, 4).copy()

        result_arr = np.zeros((out_h, out_w, 4), dtype=np.uint8)

        for row in range(n):
            for col in range(n):
                # ソース矩形（元画像内のセル）
                sx0 = sw * col / n
                sy0 = sh * row / n
                sx1 = sw * (col + 1) / n
                sy1 = sh * (row + 1) / n
                src_pts = np.float32([[sx0, sy0], [sx1, sy0], [sx1, sy1], [sx0, sy1]])

                # デスト四角形（メッシュ格子点）
                dst_pts = np.float32([
                    [grid[row][col].x() - bb_min_x, grid[row][col].y() - bb_min_y],
                    [grid[row][col+1].x() - bb_min_x, grid[row][col+1].y() - bb_min_y],
                    [grid[row+1][col+1].x() - bb_min_x, grid[row+1][col+1].y() - bb_min_y],
                    [grid[row+1][col].x() - bb_min_x, grid[row+1][col].y() - bb_min_y],
                ])

                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                cell = cv2.warpPerspective(arr, M, (out_w, out_h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_CONSTANT,
                                           borderValue=(0, 0, 0, 0))
                # アルファ合成（後のセルが上書き）
                alpha = cell[:, :, 3:4].astype(np.float32) / 255.0
                result_arr = (result_arr.astype(np.float32) * (1 - alpha) + cell.astype(np.float32) * alpha).astype(np.uint8)

        result_img = QImage(result_arr.data, out_w, out_h, result_arr.strides[0],
                            QImage.Format.Format_ARGB32).copy()
        return result_img, bb_min_x, bb_min_y

    def _composite_with_floating(self) -> QImage:
        """レイヤー合成結果にフローティング画像を重ねた最終画像を返す。
        変形中は元の切り取り領域を消した状態でフローティング画像を合成して
        リアルタイムプレビューを正しく表示する。"""
        if not self._transform_image or not self._transform_rect:
            if self._stroke_bg_cache is not None and self._stroke_layer is not None:
                result = QImage(self._stroke_bg_cache)
                p = QPainter(result)
                layer = self._stroke_layer
                blend = BLEND_KEY_TO_MODE.get(getattr(layer, 'blend_mode', 'normal'))
                if blend:
                    p.setCompositionMode(blend)
                p.setOpacity(layer.opacity / 255)  # type: ignore
                ox = getattr(layer, 'offset_x', 0)
                oy = getattr(layer, 'offset_y', 0)
                p.drawImage(ox, oy, layer.image_with_effects())  # type: ignore
                p.end()
                return result
            return self.layer_stack.composite()

        # 変形元レイヤーから切り取り領域を消去したプレビュー用合成を作る
        # 元レイヤーを一時的に切り取り済み状態にしてから composite() を呼ぶ
        layer = self._transform_layer
        if layer is not None and self._transform_vector_rest is not None:
            # ベクターは持ち上げた線を除いて描いた絵を、合成のあいだだけ差し込む。
            orig_img = layer.image  # type: ignore
            orig_off = (layer.offset_x, layer.offset_y)  # type: ignore
            rest_img, rx, ry = self._transform_vector_rest
            layer.image = rest_img  # type: ignore
            layer.offset_x, layer.offset_y = rx, ry  # type: ignore
            base = self.layer_stack.composite()
            layer.image = orig_img  # type: ignore
            layer.offset_x, layer.offset_y = orig_off  # type: ignore
        elif layer is not None and not layer.is_group:
            # 元画像をバックアップ
            orig_img = layer.image  # type: ignore
            ox = getattr(layer, 'offset_x', 0)
            oy = getattr(layer, 'offset_y', 0)
            # 消去済みコピーを作成
            erased = QImage(orig_img.width(), orig_img.height(),
                            QImage.Format.Format_ARGB32_Premultiplied)
            erased.fill(Qt.GlobalColor.transparent)
            ep = QPainter(erased)
            ep.drawImage(0, 0, orig_img)
            if self._transform_erase_mask:
                ep.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
                ep.drawImage(-ox, -oy, self._transform_erase_mask)
            elif self._transform_erase_rect:
                ep.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
                ep.fillRect(self._transform_erase_rect.translated(-ox, -oy), Qt.GlobalColor.transparent)
            ep.end()
            layer.image = erased.convertToFormat(QImage.Format.Format_ARGB32)  # type: ignore
            base = self.layer_stack.composite()
            layer.image = orig_img  # type: ignore
        else:
            base = self.layer_stack.composite()

        w, h = self.layer_stack.width, self.layer_stack.height

        if self._mesh_grid:
            result = self._warp_mesh_image()
            if result:
                warped_img, ox, oy = result
                p = QPainter(base)
                p.drawImage(ox, oy, warped_img)
                p.end()
            return base

        if self._perspective_corners:
            result = self._warp_perspective_image()
            if result:
                warped_img, ox, oy = result
                p = QPainter(base)
                p.drawImage(ox, oy, warped_img)
                p.end()
            return base

        r = self._transform_rect
        img = self._transform_image
        rx, ry = int(r.x()), int(r.y())
        rw, rh = int(r.width()), int(r.height())
        if rw != img.width() or rh != img.height():
            scaled = img.scaled(rw, rh,
                                Qt.AspectRatioMode.IgnoreAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
        else:
            scaled = img

        overlay = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        overlay.fill(Qt.GlobalColor.transparent)
        op = QPainter(overlay)
        op.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self._transform_angle != 0.0:
            pv = self._pivot_point()
            op.translate(pv.x(), pv.y())
            op.rotate(self._transform_angle)
            op.translate(-pv.x(), -pv.y())
        op.drawImage(rx, ry, scaled)
        op.end()
        p = QPainter(base)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        p.drawImage(0, 0, overlay)
        p.end()
        return base

    def _draw_transform_handles(self, p: QPainter):
        """ウィジェット座標系でハンドル枠・□・○を描く。
        paintEvent 内の p.restore() より後に呼ぶこと。"""
        if self._mesh_grid:
            self._draw_mesh_handles(p)
            return

        wpts = self._transform_corners_widget()
        if not wpts:
            return

        p.setPen(QPen(SELECTION_COLOR, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.moveTo(wpts[0])
        for pt in wpts[1:]:
            path.lineTo(pt)
        path.closeSubpath()
        p.drawPath(path)

        for pt in wpts:
            p.drawRect(QRectF(pt.x() - 4, pt.y() - 4, 8, 8))

        if self._perspective_corners:
            rot_wp = self._rotation_handle_widget()
            if rot_wp:
                p.setBrush(QBrush(QColor(255, 180, 0)))
                p.setPen(QPen(SELECTION_COLOR, 1))
                p.drawEllipse(QRectF(rot_wp.x() - 6, rot_wp.y() - 6, 12, 12))
            return

        # ピボットポイントを表示
        pv_c = self._pivot_point()
        pv_w = self._c2w().map(self._transform_matrix().map(pv_c))
        is_custom = self._pivot_mode == "custom"
        radius = 7 if is_custom else 6
        color = QColor(80, 200, 255, 200) if is_custom else QColor(255, 80, 80, 180)
        p.setBrush(color)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawEllipse(QRectF(pv_w.x() - radius, pv_w.y() - radius, radius * 2, radius * 2))
        cross = radius + 2
        p.setPen(QPen(QColor(255, 255, 255), 1))
        p.drawLine(QPointF(pv_w.x() - cross, pv_w.y()), QPointF(pv_w.x() + cross, pv_w.y()))
        p.drawLine(QPointF(pv_w.x(), pv_w.y() - cross), QPointF(pv_w.x(), pv_w.y() + cross))

        rot_wp = self._rotation_handle_widget()
        if rot_wp:
            p.setBrush(QBrush(QColor(255, 180, 0)))
            p.setPen(QPen(SELECTION_COLOR, 1))
            p.drawEllipse(QRectF(rot_wp.x() - 6, rot_wp.y() - 6, 12, 12))

    def _draw_mesh_handles(self, p: QPainter):
        """メッシュ格子とハンドルをウィジェット座標系で描く。"""
        if not self._mesh_grid:
            return
        c2w = self._c2w()
        grid = self._mesh_grid
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        p.setPen(QPen(SELECTION_COLOR, 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        # 横線
        for r in range(rows):
            for c in range(cols - 1):
                p.drawLine(c2w.map(grid[r][c]), c2w.map(grid[r][c + 1]))
        # 縦線
        for c in range(cols):
            for r in range(rows - 1):
                p.drawLine(c2w.map(grid[r][c]), c2w.map(grid[r + 1][c]))

        # ハンドル（角=大きめ四角、辺上=小さめ四角、内部=丸）
        for r in range(rows):
            for c in range(cols):
                wpt = c2w.map(grid[r][c])
                is_corner = (r in (0, rows - 1)) and (c in (0, cols - 1))
                is_edge = r in (0, rows - 1) or c in (0, cols - 1)
                if is_corner:
                    p.setBrush(QColor(255, 255, 255))
                    p.drawRect(QRectF(wpt.x() - 4, wpt.y() - 4, 8, 8))
                elif is_edge:
                    p.setBrush(QColor(200, 220, 255))
                    p.drawRect(QRectF(wpt.x() - 3, wpt.y() - 3, 6, 6))
                else:
                    p.setBrush(QColor(255, 200, 100))
                    p.drawEllipse(QRectF(wpt.x() - 3, wpt.y() - 3, 6, 6))

        rot_wp = self._rotation_handle_widget()
        if rot_wp:
            p.setBrush(QBrush(QColor(255, 180, 0)))
            p.setPen(QPen(SELECTION_COLOR, 1))
            p.drawEllipse(QRectF(rot_wp.x() - 6, rot_wp.y() - 6, 12, 12))

    def _draw_cursor_circle(self, p: QPainter):
        """ウィジェット座標系でブラシサイズを示す円を描く。"""
        pos = self._cursor_widget_pos
        if pos is None:
            return
        if self.tool == Tool.BLUR:
            size = self.blur_size
        elif self.tool == Tool.PEN:
            size = self.pen_size
        else:
            size = self.eraser_size
        # キャンバス座標系での直径をウィジェット座標系に変換（ズーム倍率を掛ける）
        radius_w = size * self.zoom / 2
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        # 視認性のため白縁+黒線の2重描き
        p.setPen(QPen(QColor(255, 255, 255, 180), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(pos, radius_w + 1, radius_w + 1)
        p.setPen(QPen(QColor(0, 0, 0, 200), 1))
        p.drawEllipse(pos, radius_w, radius_w)
        p.restore()

    # ── ベクター描画 ──────────────────────────────────────────────────────────

    def _draw_vector_preview(self, p: QPainter):
        """確定前の線を描く。キャンバス座標系の painter に描くこと。

        溜めている点をそのまま結ぶ。確定時は間引いてなめらかにするので
        見た目は少し変わるが、描いている最中の手応えはこちらが素直。
        """
        if not self._vector_points:
            return
        from vector import VectorStroke, stroke_to_path, _pen_for
        stroke = VectorStroke(points=list(self._vector_points),
                              width=float(self.pen_size),
                              color=(self.pen_color.red(), self.pen_color.green(),
                                     self.pen_color.blue(), self.pen_color.alpha()))
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.strokePath(stroke_to_path(stroke), _pen_for(stroke))
        p.restore()

    def _draw_vector_handles(self, p: QPainter):
        """選んでいる線の制御点とハンドルを描く。

        ウィジェット座標で描くので、拡大しても点の大きさは変わらない。
        点は四角、ハンドルは丸にして、掴む前に見分けられるようにする。
        """
        sel = self._vector_selected
        if sel is None or not sel.points:
            return
        layer = self.layer_stack.active
        if layer is None or not layer.is_vector or not _holds(layer.strokes, sel):
            return

        c2w = self._c2w()
        p.save()
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 線そのものを青でなぞって、選択中だと分かるようにする。
        from vector import stroke_to_path
        outline = c2w.map(stroke_to_path(sel))
        p.setPen(QPen(QColor(74, 144, 217, 200), 1.0))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(outline)

        for i, (cx, cy) in enumerate(sel.points):
            wp = c2w.map(QPointF(cx, cy))

            hp = sel.handle_points(i)
            if hp is not None:
                # ハンドルは点から伸びる棒と丸で描く
                p.setPen(QPen(QColor(120, 120, 120, 200), 1.0))
                for hx, hy in hp:
                    whp = c2w.map(QPointF(hx, hy))
                    p.drawLine(wp, whp)
                p.setPen(QPen(QColor(60, 60, 60), 1.0))
                p.setBrush(QColor(255, 255, 255))
                for hx, hy in hp:
                    whp = c2w.map(QPointF(hx, hy))
                    p.drawEllipse(whp, 4.0, 4.0)

            # 制御点は四角。ハンドルを出している点は色を変える。
            p.setPen(QPen(QColor(30, 30, 30), 1.0))
            p.setBrush(QColor(74, 144, 217) if sel.has_handles(i)
                       else QColor(255, 255, 255))
            p.drawRect(QRectF(wp.x() - 3.5, wp.y() - 3.5, 7.0, 7.0))

        p.restore()

    # ── ベクター編集（選択モード） ───────────────────────────────────────────

    def _vector_editing(self) -> bool:
        """ペンが「描く」ではなく線をいじる役割になっているか。"""
        return self.vector_pen_mode in ("select", "delpoint")

    def _vector_tol(self) -> float:
        """掴む判定の許容量（キャンバス座標）。

        拡大率で割るので、画面上では拡大しても同じ大きさに感じられる。
        """
        return 8.0 / max(0.05, self.zoom)

    def _vector_erase_press(self, layer, cp) -> bool:
        """ベクターレイヤーで消しゴムを当てる。

        ピクセルを削るのではなく線そのものを削る。消し方は2つあり、
        「線ごと消す」は当たった線を丸ごと、「交点まで消す」は
        クリックした位置から前後の交点までを消す（CLIP STUDIO と同じ考え方）。

        戻り値は実際に消せたかどうか。ドラッグ中の連続呼び出しで使う。
        """
        from vector import stroke_hit, erase_between_cuts
        x, y = float(cp.x()), float(cp.y())
        # 消しゴムの太さを当たり判定の広さに使う。細くしすぎると狙えないので
        # 拡大率から決まる最低限の許容量と大きいほうを取る。
        reach = max(self._vector_tol(), self.eraser_size / 2.0)

        # 手前の線から順に見る。重なっていたら上にあるものを消す。
        target = None
        index = -1
        for i in range(len(layer.strokes) - 1, -1, -1):
            if stroke_hit(layer.strokes[i], x, y, reach):
                target, index = layer.strokes[i], i
                break
        if target is None:
            return False

        if self.vector_erase_mode == "whole":
            pieces = []
        else:
            pieces = erase_between_cuts(target, x, y, layer.strokes, reach)
            if pieces is None:
                return False

        # 1回のドラッグで履歴は1つ。押した最初だけ積む。
        if not self._vector_erasing:
            self._save_history()
            self._vector_erasing = True

        layer.strokes[index:index + 1] = pieces
        if self._vector_selected is target:
            self._vector_selected = None
            self.vector_selection_changed.emit()
        layer.mark_dirty()
        self.edited.emit()
        self.update()
        return True

    def _vector_select_press(self, layer, cp, event):
        """選択モードでの押下。ハンドル → 制御点 → 線 の順に掴む。

        奥のものから先に見るのは、ハンドルや点が線の上に重なっていても
        手前にあるものを優先して掴めるようにするため。
        """
        from vector import (nearest_point_index, nearest_handle, stroke_hit,
                            insert_index_for)
        x, y = float(cp.x()), float(cp.y())
        tol = self._vector_tol()
        alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        # 「制御点を削除」モードは、ずっと Alt を押しているのと同じ扱いにする。
        # キーを押さえたまま何度もクリックするのは疲れるので、役割で選べるようにした。
        delpoint = self.vector_pen_mode == "delpoint"
        if delpoint:
            alt, shift = True, False
        sel = self._vector_selected

        if sel is not None and _holds(layer.strokes, sel):
            # 1. ハンドルを掴む。ただし Alt+Shift は出し入れの合図なので、
            #    ハンドルが点の近くにあっても掴まずに 2. へ流す。
            #    点を消す役割のときは、ハンドルも掴まない（消すのが目的なので）。
            hit = None if (alt and shift) or delpoint else nearest_handle(sel, x, y, tol)
            if hit is not None:
                self._save_history()
                self._vector_drag = ("handle", hit[0], hit[1])
                self._vector_drag_moved = False
                return

            # 2. 制御点を掴む。Shift 単独は「点を足す」の合図なので、
            #    近くに点があっても掴まずに 3. へ流す。
            idx = None if (shift and not alt) else nearest_point_index(sel, x, y, tol)
            if idx is not None:
                if alt and shift:
                    # ハンドルの出し入れを切り替える
                    self._save_history()
                    if sel.has_handles(idx):
                        sel.set_handles(idx, None)
                        self.status_message.emit("ハンドルをしまいました。")
                    else:
                        sel.set_handles(idx, sel.auto_handles(idx))
                        self.status_message.emit(
                            "ハンドルを出しました。ドラッグで曲がり具合を変えられます。")
                    layer.mark_dirty()
                    self.edited.emit()
                    self.update()
                    return
                if alt:
                    # 点を消す。2点は線の最低限なので、それ以下にはしない。
                    if len(sel.points) <= 2:
                        self.status_message.emit("これ以上は点を減らせません。")
                        return
                    self._save_history()
                    del sel.points[idx]
                    del sel.handles[idx]
                    layer.mark_dirty()
                    self.edited.emit()
                    self.update()
                    return
                self._save_history()
                self._vector_drag = ("point", idx)
                self._vector_drag_moved = False
                return

            # 3. 選択中の線の上なら点を足す
            if shift and stroke_hit(sel, x, y, tol):
                self._save_history()
                at = insert_index_for(sel, x, y)
                sel.points.insert(at, (x, y))
                sel.handles.insert(at, None)
                layer.mark_dirty()
                self._vector_drag = ("point", at)
                # 点を足した時点で変わっているので、履歴は残す
                self._vector_drag_moved = True
                self.update()
                return

        # 4. 線を選び直す（手前＝後に描いたものから探す）
        for stroke in reversed(layer.strokes):
            if stroke_hit(stroke, x, y, tol):
                self._vector_selected = stroke
                self.vector_selection_changed.emit()
                self.update()
                return

        # 何も無いところ＝選択解除
        if self._vector_selected is not None:
            self._vector_selected = None
            self.vector_selection_changed.emit()
        self.update()

    def _vector_drag_move(self, layer, cp) -> bool:
        """選択モードでのドラッグ。掴んでいれば True。"""
        if self._vector_drag is None:
            return False
        sel = self._vector_selected
        if sel is None or not _holds(layer.strokes, sel):
            self._vector_drag = None
            return False
        x, y = float(cp.x()), float(cp.y())

        kind = self._vector_drag[0]
        if kind == "point":
            i = self._vector_drag[1]
            if 0 <= i < len(sel.points):
                sel.points[i] = (x, y)
        else:
            i, side = self._vector_drag[1], self._vector_drag[2]
            if 0 <= i < len(sel.points) and sel.has_handles(i):
                px, py = sel.points[i]
                dx, dy = x - px, y - py
                h = sel.handles[i]
                if self._vector_corner_drag:
                    # Alt 中は片側だけ動かす。角（とがった曲がり）を作れる。
                    sel.handles[i] = ((dx, dy, h[2], h[3]) if side == "in"
                                      else (h[0], h[1], dx, dy))
                else:
                    # 既定は反対側も対称に動かして、点のところを滑らかに保つ。
                    sel.handles[i] = ((dx, dy, -dx, -dy) if side == "in"
                                      else (-dx, -dy, dx, dy))
        self._vector_drag_moved = True
        layer.mark_dirty()
        self.update()
        return True

    def _end_vector_drag(self):
        """選択モードのドラッグ終わり。動かしていなければ履歴を戻す。

        掴んだ時点で履歴を積んでいるので、ただクリックしただけのときに
        「元に戻す」が空振りするのを防ぐ。
        """
        was = self._vector_drag
        self._vector_drag = None
        self._vector_corner_drag = False
        if was is None:
            return
        if not self._vector_drag_moved and self._history:
            self._history.pop()
        else:
            self.edited.emit()
        self._vector_drag_moved = False

    def _commit_vector_stroke(self, layer):
        """溜めた点を1本の線として確定する。"""
        from vector import VectorStroke, simplify_input
        pts = self._vector_points or []
        self._vector_points = None
        if not pts:
            self.update()
            return

        width = float(self.pen_size)
        points = simplify_input(pts, width)
        if len(points) < 1:
            self.update()
            return

        # 変更前の線を控えてから足す（履歴は変更前に積む決まり）。
        self._save_history()
        layer.add_stroke(VectorStroke(
            points=points,
            width=width,
            color=(self.pen_color.red(), self.pen_color.green(),
                   self.pen_color.blue(), self.pen_color.alpha())))
        self.update()

    def _vector_active_layer(self):
        """選択中のベクター線を持っているレイヤーを返す（無ければ None）。"""
        layer = self.layer_stack.active
        if layer is None or not getattr(layer, "is_vector", False):
            return None
        if self._vector_selected is None:
            return None
        if not _holds(layer.strokes, self._vector_selected):
            # レイヤーを切り替えた等で取り残された参照を掃除する
            self._vector_selected = None
            return None
        return layer

    def set_vector_pen_mode(self, mode: str):
        """ベクターレイヤーでのペンの役割を切り替える。

        描く / 線を選んで直す / 制御点を削除 の3つ。
        """
        if mode not in ("draw", "select", "delpoint"):
            mode = "draw"
        if mode == self.vector_pen_mode:
            return
        self.vector_pen_mode = mode
        if mode == "draw":
            # 描くモードに戻ったら選択と制御点を消す（画面がうるさいので）
            self._vector_selected = None
            self._vector_drag = None
        self.vector_selection_changed.emit()
        self.update()

    def set_vector_erase_mode(self, mode: str):
        """ベクターレイヤーでの消しゴムの効き方を切り替える。

        交点まで消す / 線ごと消す の2つ。
        """
        self.vector_erase_mode = mode if mode in ("cut", "whole") else "cut"

    def set_selected_stroke_width(self, width: float):
        layer = self._vector_active_layer()
        if layer is None:
            return
        width = max(1.0, float(width))
        if abs(self._vector_selected.width - width) < 1e-9:
            return
        self._save_history()
        self._vector_selected.width = width
        layer.mark_dirty()
        self.edited.emit()
        self.update()

    def apply_color_to_selected_stroke(self):
        layer = self._vector_active_layer()
        if layer is None:
            return
        col = (self.pen_color.red(), self.pen_color.green(),
               self.pen_color.blue(), self.pen_color.alpha())
        if self._vector_selected.color == col:
            return
        self._save_history()
        self._vector_selected.color = col
        layer.mark_dirty()
        self.edited.emit()
        self.update()

    def set_selected_stroke_smooth(self, smooth: bool):
        layer = self._vector_active_layer()
        if layer is None:
            return
        smooth = bool(smooth)
        if self._vector_selected.smooth == smooth:
            return
        self._save_history()
        self._vector_selected.smooth = smooth
        layer.mark_dirty()
        self.edited.emit()
        self.update()

    def delete_selected_stroke(self):
        layer = self._vector_active_layer()
        if layer is None:
            return
        self._save_history()
        layer.strokes = [s for s in layer.strokes if s is not self._vector_selected]
        self._vector_selected = None
        self._vector_drag = None
        layer.mark_dirty()
        self.edited.emit()
        self.vector_selection_changed.emit()
        self.update()

    def _rotation_handle_canvas(self) -> QPointF | None:
        """上辺中点から回転方向に 30px 離れた回転ハンドル位置（キャンバス座標系）。"""
        if not self._transform_rect:
            return None
        r = self._transform_rect
        # ローカル座標（回転前）で上辺中点を 30px 上にオフセットしてから回転を適用する
        local_top_mid = QPointF((r.left() + r.right()) / 2, r.top() - 30)
        return self._transform_matrix().map(local_top_mid)

    def _rotation_handle_widget(self) -> QPointF | None:
        rh = self._rotation_handle_canvas()
        if rh is None:
            return None
        return self._c2w().map(rh)

    # ── mouse events ─────────────────────────────────────────────────────────

    def start_path_pick(self, callback):
        """パスピックモードを開始する。クリックで点を追加、Enterで確定（callback(points)を呼ぶ）、Escでキャンセル。"""
        self._path_pick_active = True
        self._path_pick_points = []
        self._path_pick_callback = callback
        self.setFocus()
        self.status_message.emit(
            "クリックでパスの点を追加  |  Enter: 確定  |  Backspace: 1点削除  |  Esc: キャンセル"
        )
        self.update()

    def cancel_path_pick(self):
        self._path_pick_active = False
        self._path_pick_points = []
        self._path_pick_callback = None
        self.status_message.emit("キャンセルしました")
        self.update()

    def _confirm_path_pick(self):
        points = list(self._path_pick_points)
        callback = self._path_pick_callback
        self._path_pick_active = False
        self._path_pick_points = []
        self._path_pick_callback = None
        self.status_message.emit(f"パスを確定しました（{len(points)} 点）")
        self.update()
        if callback and len(points) >= 1:
            callback(points)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return

        wp = event.position()
        cp = self._widget_to_canvas(wp.toPoint())

        if self._path_pick_active:
            self._path_pick_points.append(cp)
            self.update()
            return

        layer = self.layer_stack.active

        # Ctrl+左右ドラッグ → ブラシ/消しゴムの太さを変える（描かない）
        if (self.tool in (Tool.PEN, Tool.ERASER) and not self._panning
                and event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            attr = "pen_size" if self.tool == Tool.PEN else "eraser_size"
            self._size_drag = {"attr": attr, "x": wp.x(),
                               "size": getattr(self, attr)}
            event.accept()
            return

        # Space+ドラッグ パンニング
        if self._panning:
            self._pan_start_widget = wp.toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return

        if self.tool == Tool.TRANSFORM:
            self._handle_transform_press(wp, cp, layer)
            return

        if self.tool == Tool.MOVE:
            # ロック中は移動ツールでも動かせない。
            if self.is_locked(layer):
                self.status_message.emit("このレイヤーはロックされています。レイヤーパネルの錠マークを解除してください。")
                return
            if layer and layer.is_group:
                # 中に個別にロックしたレイヤーがあれば、それだけは動かさない。
                children = [c for c in self._collect_leaf_layers(layer)
                            if not self.is_locked(c)]
                if children:
                    def _push_group_history(children=children):
                        for child in children:
                            ox = getattr(child, 'offset_x', 0)
                            oy = getattr(child, 'offset_y', 0)
                            if child.is_vector:
                                self._history.append(
                                    ("vector", child.uid, child.copy_strokes(), ox, oy))  # type: ignore
                            else:
                                self._history.append(("pixel", child.uid, child.image.copy(), ox, oy))  # type: ignore
                        self._redo_stack.clear()
                        self._trim_history()
                        self.edited.emit()
                    # クリックしただけで動かさなかったときに、中身のない取り消しが
                    # 積まれないよう、実際に動き始めた時点で控える。
                    self._move_pending_history = _push_group_history
                    # ベクターは offset ではなく点を動かすので、掴んだ時点の線を控える。
                    self._move_group_bases = [
                        (c, c.copy_strokes() if c.is_vector else None,
                         getattr(c, 'offset_x', 0), getattr(c, 'offset_y', 0))
                        for c in children]  # type: ignore
                    self._move_base_pos = cp
                    self._drawing = True
            elif layer and not layer.is_group:
                if self._selection_rect and self._selection_rect.contains(cp):
                    # 選択範囲の中を掴んだら、レイヤー全体ではなく選択部分だけを
                    # 動かす。ここを見ていないと「選択したのに選択外も一緒に
                    # 動く」ことになる（選択ツール側と同じ挙動に合わせる）。
                    if self._lift_selection(layer):  # type: ignore
                        self._begin_transform_drag('move', wp)
                        self._drawing = True
                    return
                self._move_pending_history = self._save_history
                self._move_base_image = layer.image  # type: ignore
                # ベクターは点を動かすので、掴んだ時点の線を控えておく。
                self._move_base_strokes = (
                    layer.copy_strokes() if layer.is_vector else None)  # type: ignore
                self._move_base_offset = (layer.offset_x, layer.offset_y)  # type: ignore
                self._move_base_pos = cp
                self._last_pos = cp
                self._drawing = True
            return

        if not layer or layer.is_group:
            if layer and layer.is_group and self.tool in (Tool.PEN, Tool.ERASER, Tool.FILL, Tool.BLUR, Tool.LASSO_FILL):
                self.status_message.emit("グループレイヤーには描画できません。子レイヤーを選択してください。")
            return

        # ロックしたレイヤーは描画も移動もさせない。選択中なのに黙って
        # 何も起きないと戸惑うので、理由を必ず出す。
        if self.is_locked(layer):
            self.status_message.emit("このレイヤーはロックされています。レイヤーパネルの錠マークを解除してください。")
            return

        # 見えていないレイヤーに描けてしまうと、画面に何も出ないまま筆跡だけが
        # 残り、後で表示に戻したときに覚えのない線が現れる。他のソフトと同じく止める。
        if not layer.visible:
            self.status_message.emit("非表示のレイヤーには描画できません。目マークを押して表示してください。")
            return

        # ベクターレイヤーは線から絵を描き直すので、画像に直接描く道具は
        # 次の描き直しで消えてしまう。黙って消えると原因が分からないため
        # ここで止めて理由を伝える。
        if layer.is_vector and self.tool in (
                Tool.FILL, Tool.BLUR, Tool.LINE, Tool.RECT, Tool.ELLIPSE,
                Tool.TEXT, Tool.LASSO_FILL):
            self.status_message.emit(
                "ベクターレイヤーにはこの道具はまだ使えません。ペンを使うか、"
                "レイヤーをラスタライズしてください。")
            return

        # 消しゴムはピクセルではなく線そのものを消す。
        # ドラッグでなぞれるよう _drawing は立てるが、画像には触らない。
        if layer.is_vector and self.tool == Tool.ERASER:
            self._vector_erasing = False
            self._drawing = True
            self._vector_erase_press(layer, cp)
            return

        self._drawing = True

        # 描く前にレイヤー画像を必要なだけ広げる（移動後などで筆跡が
        # バッファ外になり消えるのを防ぐ）。塗りつぶしは既存ピクセルの
        # 連結領域しか塗らないので拡張しない。
        line_from = self._shift_line_start(event, layer)
        if self.tool in (Tool.PEN, Tool.ERASER, Tool.BLUR):
            self._grow_for_draw(layer, cp, self._draw_margin())
            if line_from is not None:
                # 直線の始点もはみ出さないよう広げておく
                self._grow_for_draw(layer, line_from, self._draw_margin())

        # 選択範囲があるときは、描いた後で範囲外を元に戻せるよう控えておく。
        # ここは描画系ツールすべてが通るので、1か所で漏れなくクリップできる。
        self._clip_mask = self._lasso_mask
        self._begin_clip_to_selection(layer)

        lox = getattr(layer, 'offset_x', 0)
        loy = getattr(layer, 'offset_y', 0)
        lp = QPoint(cp.x() - lox, cp.y() - loy)

        if self.tool == Tool.PEN and layer.is_vector and self._vector_editing():
            self._vector_select_press(layer, cp, event)

        elif self.tool == Tool.PEN and layer.is_vector:
            # ベクターは確定するまで image に描かない。点を溜めておき、
            # 表示はプレビューで見せて、離したときに1本の線にする。
            self._stabilizer.reset()
            self._stabilizer.push(cp)
            self._vector_points = [(float(cp.x()), float(cp.y()))]
            if line_from is not None:
                self._vector_points.insert(0, (float(line_from.x()), float(line_from.y())))
            self.update()

        elif self.tool == Tool.PEN:
            self._save_history()
            self._begin_stroke_cache(layer)
            self._begin_pen_buffer(layer)
            self._stabilizer.reset()
            if line_from is not None:
                dirty = self._press_line(layer, line_from, lp)
            else:
                smooth_pt = self._stabilizer.push(lp).toPoint()
                self._last_pos = smooth_pt
                self._brush_stamp(self._stroke_target(layer), smooth_pt,
                                  self._taper_start_size(layer, smooth_pt))
                dirty = self._segment_dirty(smooth_pt, smooth_pt)
            self._after_stroke_draw(layer, dirty)
            self.update()

        elif self.tool == Tool.ERASER and self._is_alpha_locked(layer):
            # 透明ピクセルのロック中は透明にできないので、消しゴムは効かない。
            self._drawing = False
            self.status_message.emit(
                "透明ピクセルがロックされているため消せません。レイヤーパネルの ▦ を解除してください。")

        elif self.tool == Tool.ERASER:
            self._save_history()
            self._begin_stroke_cache(layer)
            self._stabilizer.reset()
            if line_from is not None:
                self._press_line(layer, line_from, lp)
            else:
                self._stabilizer.push(lp)
                self._last_pos = lp
                self._erase_point(layer.image, lp, self._taper_start_size(layer, lp))  # type: ignore
            self._apply_clip_to_selection()
            self.update()

        elif self.tool == Tool.FILL:
            self._save_history()
            # 移動したレイヤーや、開き直したファイルのレイヤーは画像が
            # キャンバスより小さいことがある。そのままだとバッファ外の
            # クリックが何も塗らないので、先にキャンバス全体を覆わせる。
            self._ensure_layer_bounds(
                layer, QRect(0, 0, self.layer_stack.width, self.layer_stack.height))
            self._begin_clip_to_selection(layer)
            lp = QPoint(cp.x() - layer.offset_x, cp.y() - layer.offset_y)  # type: ignore
            # 参照画像は _build_fill_reference が対象レイヤーの座標系に
            # 合わせて返す（offset のずれもここで吸収される）。
            thr = _sensitivity_to_threshold(self.fill_line_sensitivity)
            if self.fill_reference_mode == "all":
                # 見えている絵全体の色で境界を決める
                ref_img, color_judge = None, self._build_all_layers_image(layer)
            else:
                ref_img, color_judge = self._build_fill_reference(layer), None
            _flood_fill_expanded(layer.image, lp.x(), lp.y(), self.pen_color, ref_img,  # type: ignore
                                 self.fill_expand, self.fill_close_gap, thr,
                                 self.fill_reference_mode == "ref_self",
                                 self.fill_tolerance, color_judge)
            self._apply_clip_to_selection()
            self.update()

        elif self.tool == Tool.BLUR:
            self._save_history()
            self._begin_stroke_cache(layer)
            self._last_pos = lp
            self._blur_brush.strength = self.blur_strength
            self._blur_brush.stamp(layer.image, lp, self.pen_color, self.blur_size)
            self._apply_clip_to_selection()
            self.update()

        elif self.tool == Tool.EYEDROPPER:
            composite = self.layer_stack.composite()
            if 0 <= cp.x() < composite.width() and 0 <= cp.y() < composite.height():
                picked = QColor.fromRgba(composite.pixel(cp.x(), cp.y()))
                # 透明な所を拾うとペンが「透明な黒」になって描けなくなる。
                # 何もない所は無視し、半透明は色だけを拾う。
                if picked.alpha() > 0:
                    picked.setAlpha(255)
                    self._prev_color = QColor(self.pen_color)
                    self.color_picked.emit(picked)
            if not self._alt_eyedropper:
                self._drawing = False

        elif self.tool in (Tool.LINE, Tool.RECT, Tool.ELLIPSE):
            self._save_history()
            self._preview_start = cp
            self._preview_end = cp

        elif self.tool == Tool.SELECT_RECT:
            if self._transform_image:
                # 変形中 → ハンドルヒットテストして操作継続 or 確定して新規選択
                handle = self._hit_transform_handle(wp)
                if handle:
                    self._begin_transform_drag(handle, wp)
                else:
                    self._commit_transform()
                    self._preview_start = cp
                    self._preview_end = cp
            elif self._selection_rect and layer and not layer.is_group:
                if self.select_mode == "transform":
                    # 変形モード: 選択範囲内クリックで即 lift して変形ハンドルを出す
                    if self._selection_rect.contains(cp):
                        self._lift_selection(layer)  # type: ignore
                        # lift 直後はハンドルヒットテストして move/scale を開始
                        handle = self._hit_transform_handle(wp)
                        self._begin_transform_drag(handle or 'move', wp)
                    else:
                        # 選択外クリック → 変形中なら確定してから新規選択
                        if self._transform_image:
                            self._commit_transform()
                        self._selection_rect = None
                        self._lasso_mask = None
                        self._selection_outline_path = None
                        self._preview_start = cp
                        self._preview_end = cp
                else:
                    # 選択モード（デフォルト）: ドラッグで移動、クリックのみなら維持
                    if self._selection_rect.contains(cp):
                        self._lift_pending = True
                        self._lift_pending_wp = wp
                        self._drawing = True
                    else:
                        self._selection_rect = None
                        self._lasso_mask = None
                        self._selection_outline_path = None
                        self._preview_start = cp
                        self._preview_end = cp
            else:
                # 新規選択開始
                self._selection_rect = None
                self._lasso_mask = None
                self._selection_outline_path = None
                self._preview_start = cp
                self._preview_end = cp

        elif self.tool == Tool.LASSO_FILL:
            # 常に新規に投げなわを描く専用ツール（選択の持ち上げ・変形は行わない）
            self._selection_rect = None
            self._lasso_mask = None
            self._selection_outline_path = None
            self._lasso_path_points = []
            self._lasso_points = [cp]

        elif self.tool == Tool.LASSO:
            if self._transform_image:
                handle = self._hit_transform_handle(wp)
                if handle:
                    self._begin_transform_drag(handle, wp)
                else:
                    self._commit_transform()
                    self._lasso_points = [cp]
                    self._lasso_path_points = []
            elif self._selection_rect and layer and not layer.is_group:
                if self.select_mode == "transform":
                    if self._selection_rect.contains(cp):
                        self._lift_selection(layer)  # type: ignore
                        self._begin_transform_drag("move", wp)
                    else:
                        self._selection_rect = None
                        self._lasso_mask = None
                        self._selection_outline_path = None
                        self._lasso_path_points = []
                        self._lasso_points = [cp]
                else:
                    if self._selection_rect.contains(cp):
                        self._lift_pending = True
                        self._lift_pending_wp = wp
                        self._drawing = True
                    else:
                        self._selection_rect = None
                        self._lasso_mask = None
                        self._selection_outline_path = None
                        self._lasso_path_points = []
                        self._lasso_points = [cp]
            else:
                self._selection_rect = None
                self._lasso_mask = None
                self._selection_outline_path = None
                self._lasso_path_points = []
                self._lasso_points = [cp]

        elif self.tool == Tool.TEXT:
            # クリック位置を確定してからダイアログを表示する
            self._text_pos = cp
            self._drawing = False
            self._ask_text()

    def _begin_transform_drag(self, handle: str, wp: QPointF):
        """変形ハンドルのドラッグ開始状態を記録する。TRANSFORM ツールに限らず
        SELECT_RECT/LASSO の変形モードからも呼ばれるため、ここに一本化する
        （個別に書くと mesh_grid_start / perspective_corners_start の初期化が
        漏れやすく、実際に漏れて自由変形の頂点ドラッグが効かないバグになっていた）。"""
        self._transform_handle = handle
        self._transform_drag_start = wp
        self._transform_rect_start = QRectF(self._transform_rect)
        self._transform_angle_start = self._transform_angle
        if self._mesh_grid:
            self._mesh_grid_start = [[QPointF(p) for p in row] for row in self._mesh_grid]
        elif self._perspective_corners:
            self._perspective_corners_start = [QPointF(pt) for pt in self._perspective_corners]
        self._drawing = True

    def _handle_transform_press(self, wp: QPointF, cp: QPoint,
                                 layer):
        if self._transform_image and self._transform_rect:
            handle = self._hit_transform_handle(wp)
            if handle:
                self._begin_transform_drag(handle, wp)
            else:
                self._commit_transform()
        else:
            if layer and not layer.is_group:
                self._lift_selection(layer)  # type: ignore

    def mouseMoveEvent(self, event):
        wp = event.position()
        cp = self._widget_to_canvas(wp.toPoint())
        self.status_message.emit(f"x:{cp.x()}  y:{cp.y()}")
        # パンニング
        if self._panning and self._pan_start_widget is not None:
            if self._scroll_area is not None:
                delta = wp.toPoint() - self._pan_start_widget
                hbar = self._scroll_area.horizontalScrollBar()
                vbar = self._scroll_area.verticalScrollBar()
                hbar.setValue(hbar.value() - delta.x())
                vbar.setValue(vbar.value() - delta.y())
            self._pan_start_widget = wp.toPoint()
            event.accept()
            return

        if self._size_drag is not None:
            self._update_size_drag(wp.x())
            event.accept()
            return

        if self.tool in (Tool.PEN, Tool.ERASER):
            self._cursor_widget_pos = wp
            self.update()

        # 選択範囲内クリック後、実際にドラッグが始まったら lift を実行
        if self._lift_pending and self._drawing:
            layer = self.layer_stack.active
            self._lift_pending = False
            self._lift_pending_wp = None
            if layer and not layer.is_group and self._lift_selection(layer):  # type: ignore
                self._begin_transform_drag('move', wp)

        if self._drawing and self._transform_handle and self._transform_image:
            # TRANSFORM ツール以外でも変形ドラッグ（SELECT_RECT/LASSO での持ち上げ後）
            self._drag_transform(wp)
            return

        if not self._drawing:
            return
        layer = self.layer_stack.active
        if not layer:
            return

        if self.tool == Tool.MOVE and self._move_base_pos:
            dx = cp.x() - self._move_base_pos.x()
            dy = cp.y() - self._move_base_pos.y()
            pending = getattr(self, "_move_pending_history", None)
            if pending is not None and (dx or dy):
                self._move_pending_history = None
                pending()
            if self._move_group_bases is not None:
                for child, base_strokes, base_ox, base_oy in self._move_group_bases:
                    if base_strokes is not None:
                        child.strokes = [s.copy() for s in base_strokes]
                        child.translate_strokes(dx, dy)
                    else:
                        child.offset_x = base_ox + dx
                        child.offset_y = base_oy + dy
            elif self._move_base_image is not None and not layer.is_group:
                self._move_layer(layer, dx, dy)  # type: ignore
            self.update()
            return

        if layer.is_group:
            return

        if self.tool == Tool.ERASER and layer.is_vector:
            # なぞったところの線を続けて消す。履歴は押下時の1つだけ。
            self._vector_erase_press(layer, cp)
            return

        if self.tool == Tool.PEN and layer.is_vector:
            if self._vector_editing():
                self._vector_corner_drag = bool(
                    event.modifiers() & Qt.KeyboardModifier.AltModifier)
                self._vector_drag_move(layer, cp)
                return
            if self._vector_points is not None:
                sp = self._stabilizer.push(cp)
                self._vector_points.append((sp.x(), sp.y()))
                self.update()
            return

        if self.tool in (Tool.PEN, Tool.ERASER, Tool.BLUR):
            shift = self._grow_for_draw(layer, cp, self._draw_margin())
            if not shift.isNull():
                # レイヤーが広がった分だけローカル座標系の原点がずれるので、
                # 前回位置を新しい原点に合わせ直す（ストロークの断裂防止）。
                if self._last_pos is not None:
                    self._last_pos = self._last_pos + shift
                self._stabilizer.translate(shift)
                self._shift_clip_base(layer, shift)
                self._shift_pen_buffer(layer, shift)

        lox = getattr(layer, 'offset_x', 0)
        loy = getattr(layer, 'offset_y', 0)
        lp = QPoint(cp.x() - lox, cp.y() - loy)

        if self.tool in (Tool.PEN, Tool.ERASER) and self._last_pos is not None:
            smooth_pt = self._stabilizer.push(lp).toPoint()
            dirty = self._segment_dirty(self._last_pos, smooth_pt)
            self._stroke_segment(layer, self._last_pos, smooth_pt)
            self._last_pos = smooth_pt
            self._after_stroke_draw(layer, dirty)
            self.update()

        elif self.tool == Tool.BLUR and self._last_pos is not None:
            self._blur_brush.stroke_to(layer.image, self._last_pos, lp, self.pen_color, self.blur_size)
            self._last_pos = lp
            self._apply_clip_to_selection()
            self.update()

        elif self.tool in (Tool.LINE, Tool.RECT, Tool.ELLIPSE, Tool.SELECT_RECT):
            if (self.tool in (Tool.LINE, Tool.RECT, Tool.ELLIPSE) and self._preview_start is not None
                    and QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier):
                cp = _constrain_shape_shift(self.tool, self._preview_start, cp)
            self._preview_end = cp
            self.update()

        elif self.tool in (Tool.LASSO, Tool.LASSO_FILL):
            self._lasso_points.append(cp)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._size_drag is not None:
            self._size_drag = None
            event.accept()
            return
        if self._drawing and not self._panning:
            self._finish_stabilized_stroke()
            self._taper_finish()
            self._remember_stroke_end()
        self._taper_stroke = None
        self._pen_buf = None
        self._end_stroke_cache()
        if self._panning:
            self._pan_start_widget = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        cp = self._widget_to_canvas(event.position().toPoint())
        layer = self.layer_stack.active
        self._drawing = False
        self._move_base_image = None
        self._move_base_strokes = None
        self._move_group_bases = None
        self._move_base_pos = None
        self._move_pending_history = None

        # lift 保留のままリリース → クリックのみ（ドラッグなし）なので選択範囲を維持
        if self._lift_pending:
            self._lift_pending = False
            self._lift_pending_wp = None
            return

        # 変形ドラッグ終了（TRANSFORM / SELECT_RECT / LASSO 共通）
        if self._transform_handle is not None:
            self._transform_handle = None
            return

        if not layer or layer.is_group:
            return

        if self.tool == Tool.ERASER and layer.is_vector:
            self._vector_erasing = False
            return

        if self.tool == Tool.PEN and layer.is_vector and self._vector_editing():
            self._end_vector_drag()
            return

        if self.tool == Tool.PEN and layer.is_vector and self._vector_points is not None:
            self._commit_vector_stroke(layer)
            return

        if self.tool in (Tool.LINE, Tool.RECT, Tool.ELLIPSE) and self._preview_start is not None:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                cp = _constrain_shape_shift(self.tool, self._preview_start, cp)
            # ドラッグせずクリックしただけ（始点=終点）の場合は退化図形なので何も描かない。
            # QRect(a, a) は幅・高さ 1 になるため、点の一致で判定する。
            is_zero = (self._preview_start == cp)
            if is_zero:
                # 退化図形は描かない。押下時に積んだ履歴を巻き戻す
                lid = self._layer_id()
                if lid is not None and self._history and self._history[-1][0] == "pixel" and self._history[-1][1] == lid:
                    layer.image = self._history.pop()[2]  # type: ignore
            else:
                # 図形の外接矩形（線幅ぶんの余白込み）がレイヤー画像に収まるよう
                # 広げてから描く。はみ出す図形が切れるのを防ぐ。
                pad = max(1, int(self.pen_size))
                shape_rect = QRect(self._preview_start, cp).normalized()
                shape_rect = shape_rect.adjusted(-pad, -pad, pad, pad)
                self._ensure_layer_bounds(layer, shape_rect)
                # 広げた後の画像で控え直す（大きさが変わると範囲外・透明度を戻せない）
                self._begin_clip_to_selection(layer)
                lox = getattr(layer, 'offset_x', 0)
                loy = getattr(layer, 'offset_y', 0)
                sa = QPoint(self._preview_start.x() - lox, self._preview_start.y() - loy)
                sb = QPoint(cp.x() - lox, cp.y() - loy)
                self._commit_shape(layer.image, sa, sb)  # type: ignore
                # 図形は離した時点で確定するので、ここでクリップする
                self._apply_clip_to_selection()
            self._preview_start = None
            self._preview_end = None
            self.update()

        elif self.tool == Tool.SELECT_RECT and self._preview_start is not None:
            sel = QRect(self._preview_start, cp).normalized()
            self._preview_start = None
            self._preview_end = None
            if sel.width() > 1 and sel.height() > 1:
                self._selection_rect = sel
                self._lasso_mask = None
                self._selection_outline_path = None
            else:
                self._selection_rect = None
            self._sync_ant_timer()
            self.update()

        elif self.tool == Tool.LASSO_FILL and self._lasso_points and len(self._lasso_points) <= 2:
            # 2点以下でリリース → 範囲を作れないので何もしない
            self._lasso_points = []
            self.update()

        elif self.tool == Tool.LASSO_FILL and len(self._lasso_points) > 2:
            self._apply_lasso_fill(layer, self._lasso_points)
            self._lasso_points = []
            self.update()

        elif self.tool == Tool.LASSO and self._lasso_points and len(self._lasso_points) <= 2:
            # 2点以下でリリース → 選択できないのでクリア
            self._lasso_points = []
            self._selection_rect = None
            self._lasso_mask = None
            self._selection_outline_path = None
            self._sync_ant_timer()
            self.update()

        elif self.tool == Tool.LASSO and len(self._lasso_points) > 2:
            w, h = self.layer_stack.width, self.layer_stack.height
            self._lasso_mask = _mask_from_polygon(self._lasso_points, w, h)
            xs = [p.x() for p in self._lasso_points]
            ys = [p.y() for p in self._lasso_points]
            sel = QRect(
                max(0, min(xs)), max(0, min(ys)),
                min(w, max(xs)) - max(0, min(xs)),
                min(h, max(ys)) - max(0, min(ys))
            )
            if sel.width() > 0 and sel.height() > 0:
                self._selection_rect = sel
                self._lasso_path_points = list(self._lasso_points)
                self._selection_outline_path = None
            else:
                self._selection_rect = None
                self._lasso_mask = None
                self._selection_outline_path = None
                self._lasso_path_points = []
            self._lasso_points = []
            self._sync_ant_timer()
            self.update()

        self._last_pos = None

    def _build_fill_reference(self, layer) -> QImage | None:
        """塗りつぶしの境界判定に使う参照画像（自分以外の参照レイヤーの合成、
        キャンバス座標系・offset 0,0 に正規化したもの）を作る。
        通常レイヤーはローカル画像に offset_x/offset_y が付いているため、
        グループの composite() 結果（既にキャンバスサイズ・offset 0,0）と
        混在しても正しく重なるよう、必ず offset 付きで描画してから返す。"""
        refs = [r for r in self.layer_stack.references_excluding(layer)
                if r is not layer]
        if len(refs) == 0:
            return None
        w, h = self.layer_stack.width, self.layer_stack.height
        ref_img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        ref_img.fill(Qt.GlobalColor.transparent)
        rp = QPainter(ref_img)
        for r in reversed(refs):
            rp.setOpacity(r.opacity / 255)
            rp.drawImage(getattr(r, 'offset_x', 0), getattr(r, 'offset_y', 0), r.image)
        rp.end()
        return self._to_layer_local(ref_img.convertToFormat(QImage.Format.Format_ARGB32),
                                    layer)

    def _build_all_layers_image(self, layer) -> QImage:
        """「すべてのレイヤー」で塗るときの判定画像（見えている絵の合成）。"""
        return self._to_layer_local(self.layer_stack.composite(), layer)

    def _to_layer_local(self, ref_img: QImage, layer) -> QImage:
        """キャンバス座標の画像を、対象レイヤーのローカル座標・大きさに置き直す。"""
        w, h = self.layer_stack.width, self.layer_stack.height
        # 対象レイヤーのローカル座標系に合わせる。
        # QImage.copy() は範囲外を透明で埋めるので、レイヤーが移動や貼り付けで
        # ずれている（offset がマイナス、またはキャンバス幅を超える）場合、
        # 参照の線が消えた透明画像になり「参照レイヤーが効かない」状態になる。
        # 切り出しではなく、正しい位置に描き直すことで範囲外でも欠けない。
        lw, lh = layer.image.width(), layer.image.height()
        lox = getattr(layer, 'offset_x', 0)
        loy = getattr(layer, 'offset_y', 0)
        if lw != w or lh != h or lox or loy:
            local = QImage(lw, lh, QImage.Format.Format_ARGB32)
            local.fill(Qt.GlobalColor.transparent)
            lp = QPainter(local)
            lp.drawImage(-lox, -loy, ref_img)
            lp.end()
            ref_img = local
        return ref_img

    def _apply_lasso_fill(self, layer, lasso_points: list[QPoint]) -> None:
        """投げなわで囲んだ範囲内にある、線で閉じた領域だけを自動で塗りつぶす。"""
        if not layer or layer.is_group:
            return
        # layer.image はキャンバスと同じ大きさとは限らない（貼り付け直後のレイヤー等は
        # offset_x/offset_y 付きでキャンバスより大きいことがある）。投げなわ範囲は
        # キャンバス座標系の点で来るため、layer.image のローカル座標系に変換してから
        # そのサイズでマスクを作る（Tool.FILL が lp = cp - offset で変換しているのと同じ考え方）。
        lox = getattr(layer, 'offset_x', 0)
        loy = getattr(layer, 'offset_y', 0)
        local_points = [QPoint(p.x() - lox, p.y() - loy) for p in lasso_points]
        lw, lh = layer.image.width(), layer.image.height()
        mask_img = _mask_from_polygon(local_points, lw, lh)
        nbytes = lh * lw * 4
        ptr = mask_img.bits(); ptr.setsize(nbytes)
        mask_arr = np.frombuffer(ptr, dtype=np.uint8).reshape(lh, lw, 4)
        area_mask = (mask_arr[:, :, 3] > 0).astype(np.uint8)
        if not area_mask.any():
            return

        self._save_history()
        # 透明ピクセルのロックを効かせるため、塗る前の画像を控える
        # （投げなわ自体が範囲なので、選択範囲のクリップはここでは関係しない）。
        self._clip_mask = None
        self._begin_clip_to_selection(layer)
        # 参照画像は _build_fill_reference が対象レイヤーの座標系・サイズに
        # 合わせて返すので、ここでの切り出しは不要。
        if self.fill_reference_mode == "all":
            # 見えている絵全体を線画とみなす（自分のレイヤーも合成に入っている）
            ref_img = self._build_all_layers_image(layer)
        else:
            ref_img = self._build_fill_reference(layer)
        filled = _fill_closed_regions_in_area(
            layer.image, area_mask, self.pen_color, ref_img,  # type: ignore
            self.fill_expand, self.fill_close_gap,
            _sensitivity_to_threshold(self.fill_line_sensitivity),
            self.fill_reference_mode == "ref_self")
        self._apply_clip_to_selection()
        if filled == 0 and self._history and self._history[-1][0] == "pixel":
            # 何も塗られなかった場合は空の undo エントリを積まない
            self._history.pop()
        self.update()

    # ── drawing helpers ──────────────────────────────────────────────────────

    def _make_pen(self, color: QColor, size: int) -> QPen:
        return QPen(color, size, Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)

    def _mirror_x(self, pt: QPoint) -> QPoint:
        """対称定規用: キャンバス中心でX軸ミラーした点を返す。

        pt は描いているレイヤーのローカル座標。移動したレイヤーでも
        キャンバスの中心線で折り返すよう、いったんキャンバス座標に直す。
        """
        cx = self.layer_stack.width // 2
        layer = self.layer_stack.active
        ox = getattr(layer, 'offset_x', 0) if layer is not None else 0
        if isinstance(pt, QPointF):
            return QPointF(2 * cx - pt.x() - 2 * ox, pt.y())
        return QPoint(2 * cx - pt.x() - 2 * ox, pt.y())

    def _brush_stamp(self, img: QImage, pt: QPoint, size: float | None = None):
        """ブラシで1点描画（対称定規対応）。size 省略時はペンの太さ。"""
        size = self.pen_size if size is None else size
        brush = get_brush(self.brush_type)
        brush.stamp(img, pt, self.pen_color, size)
        if self.symmetry_enabled:
            brush.stamp(img, self._mirror_x(pt), self.pen_color, size)

    def _brush_stroke(self, img: QImage, a: QPoint, b: QPoint, size: float | None = None):
        """ブラシでストローク描画（対称定規対応）。size 省略時はペンの太さ。"""
        size = self.pen_size if size is None else size
        brush = get_brush(self.brush_type)
        brush.stroke_to(img, a, b, self.pen_color, size)
        if self.symmetry_enabled:
            brush.stroke_to(img, self._mirror_x(a), self._mirror_x(b),
                            self.pen_color, size)

    def _erase_point(self, img: QImage, p: QPoint, size: float | None = None):
        if self.eraser_soft:
            self._soft_erase(img, p, p, size)
            return
        painter = QPainter(img)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.setPen(QPen(Qt.GlobalColor.transparent,
                            self.eraser_size if size is None else size,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPoint(p)
        painter.end()

    def _erase_line(self, img: QImage, a: QPoint, b: QPoint, size: float | None = None):
        if self.eraser_soft:
            self._soft_erase(img, a, b, size)
            return
        painter = QPainter(img)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.setPen(QPen(Qt.GlobalColor.transparent,
                            self.eraser_size if size is None else size,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                            Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(a, b)
        painter.end()

    def _update_size_drag(self, widget_x: float) -> None:
        """Ctrl+ドラッグ中: 右へ動かすほど太く、左へ動かすほど細くする。"""
        d = self._size_drag
        hi = 200 if d["attr"] == "pen_size" else 300
        # 画面上の移動量そのままにすると、拡大中は変化が鈍くなりすぎないよう
        # ズームで割ってキャンバスの px 単位にそろえる（円カーソルの縁が指に付いてくる）
        delta = (widget_x - d["x"]) / max(self.zoom, 0.01)
        size = max(1, min(hi, int(round(d["size"] + delta))))
        if size == getattr(self, d["attr"]):
            return
        setattr(self, d["attr"], size)
        if d["attr"] == "pen_size":
            self.brush_size_changed.emit(size)
        else:
            self.eraser_size_changed.emit(size)
        self.status_message.emit(f"サイズ: {size}")
        self.update()

    def _soft_erase(self, img: QImage, a: QPoint, b: QPoint, size: float | None = None):
        """中心ほど強く、ふちほど弱く消す（ソフトブラシの逆）。"""
        size = self.eraser_size if size is None else size
        r = max(0.5, size / 2.0)
        dx, dy = b.x() - a.x(), b.y() - a.y()
        dist = (dx * dx + dy * dy) ** 0.5
        step = max(1.0, size * 0.15)
        n = max(1, int(dist / step) + 1)
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            pt = QPointF(a.x() + dx * t, a.y() + dy * t)
            grad = QRadialGradient(pt, r)
            grad.setColorAt(0.0, QColor(0, 0, 0, 90))
            grad.setColorAt(1.0, QColor(0, 0, 0, 0))
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(pt, r, r)
        painter.end()

    # ── 入り抜き ──────────────────────────────────────────────────────────────
    # 描いている間は入りだけを効かせて普通に描き、離したときに「描く前の絵」から
    # 線全体を入り抜き付きで描き直す（抜きは終わりが分かるまで決められないため）。

    def _taper_settings(self) -> dict | None:
        """今のツールの入り抜き設定。どちらも 0 なら None（＝使わない）。"""
        key = {Tool.PEN: "pen", Tool.ERASER: "eraser"}.get(self.tool)
        t = self.taper.get(key) if key else None
        if not t or (t["in"] <= 0 and t["out"] <= 0):
            return None
        return t

    def set_taper(self, tool_key: str, field: str, value: int):
        """入り抜きの設定を変える。tool_key は "pen"/"eraser"、field は in/out/tip。"""
        if tool_key not in self.taper or field not in ("in", "out", "tip"):
            return
        hi = 100 if field == "tip" else TAPER_MAX
        self.taper[tool_key][field] = max(0, min(hi, int(value)))

    def _taper_begin(self, layer, lp: QPoint) -> dict | None:
        """押したときに呼ぶ。描き直し用に描く前の絵と、始点を控える。"""
        t = self._taper_settings()
        self._taper_stroke = None
        if t is None or layer is None or layer.is_vector:
            return None
        ox = getattr(layer, 'offset_x', 0)
        oy = getattr(layer, 'offset_y', 0)
        self._taper_stroke = {
            "t": dict(t), "tool": self.tool, "layer": layer,
            "base": layer.image.copy(), "base_off": (ox, oy),
            # 点はキャンバス座標で持つ（途中でレイヤーが広がっても崩れない）
            "pts": [QPointF(lp.x() + ox, lp.y() + oy)], "dist": 0.0,
        }
        return self._taper_stroke

    def _taper_start_size(self, layer, lp: QPoint) -> float:
        """押した瞬間の点の太さ。入りがあれば先端の細さから始める。"""
        ts = self._taper_begin(layer, lp)
        size = self._tool_size()
        if ts is None:
            return size
        t = ts["t"]
        return _taper_width(0.0, None, size, t["in"], 0, t["tip"])

    def _tool_size(self) -> float:
        return float(self.eraser_size if self.tool == Tool.ERASER else self.pen_size)

    def _taper_draw_piece(self, img: QImage, a, b, size: float):
        if self.tool == Tool.ERASER:
            self._erase_line(img, a, b, size)
        else:
            self._brush_stroke(img, a, b, size)

    def _stroke_segment(self, layer, a: QPoint, b: QPoint):
        """ペン・消しゴムの1区間を描く。入りがあれば始点からの距離で細くする。"""
        ts = self._taper_stroke
        target = self._stroke_target(layer)
        if ts is None or ts["layer"] is not layer:
            self._taper_draw_piece(target, a, b, self._tool_size())
            return
        ox = getattr(layer, 'offset_x', 0)
        oy = getattr(layer, 'offset_y', 0)
        ts["pts"].append(QPointF(b.x() + ox, b.y() + oy))
        t = ts["t"]
        start = ts["dist"]
        ts["dist"] += math.hypot(b.x() - a.x(), b.y() - a.y())
        for pa, pb, w in _taper_pieces([QPointF(a), QPointF(b)], self._tool_size(),
                                       t["in"], 0, t["tip"], None, start):
            self._taper_draw_piece(target, pa, pb, w)

    def _taper_finish(self):
        """離したときに呼ぶ。描く前の絵に、入り抜き付きで線を描き直す。"""
        ts = self._taper_stroke
        self._taper_stroke = None
        layer = self.layer_stack.active
        if ts is None or ts["layer"] is not layer or ts["tool"] != self.tool:
            return
        ox = getattr(layer, 'offset_x', 0)
        oy = getattr(layer, 'offset_y', 0)
        pen_buf = self._active_pen_buf(layer)
        img = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        if pen_buf is None:
            # 不透明度つきのペンは別画像に線だけ描き直すので、元の絵は要らない
            p = QPainter(img)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
            bx, by = ts["base_off"]
            p.drawImage(bx - ox, by - oy, ts["base"])
            p.end()

        pts = [QPointF(q.x() - ox, q.y() - oy) for q in ts["pts"]]
        total = sum(math.hypot(b.x() - a.x(), b.y() - a.y()) for a, b in zip(pts, pts[1:]))
        t, size = ts["t"], self._tool_size()
        if total < 1e-6:
            if self.tool == Tool.ERASER:
                self._erase_point(img, pts[0], size)
            else:
                self._brush_stamp(img, pts[0], size)
        for a, b, w in _taper_pieces(pts, size, t["in"], t["out"], t["tip"], total):
            self._taper_draw_piece(img, a, b, w)
        if pen_buf is None:
            layer.image = img
        else:
            pen_buf["buf"] = img
            self._compose_pen_buffer(layer, None)
        self._apply_clip_to_selection()
        self.update()

    # ── 不透明度つきペン・透明色 ─────────────────────────────────────────────

    def _begin_pen_buffer(self, layer) -> None:
        """不透明度 100% 未満や透明色のときは、線を別の画像に描く準備をする。

        直接描くと、1本の線の中で重なった所（折り返しや点の重なり）が
        濃くなってしまう。線だけを別画像に描き、毎回「描く前の絵＋線」を
        作り直すことで、線全体が同じ濃さになる。
        """
        self._pen_buf = None
        if layer is None or layer.is_vector:
            return
        if self.pen_opacity >= 100 and not self.pen_transparent:
            return
        buf = QImage(layer.image.size(), QImage.Format.Format_ARGB32)
        buf.fill(Qt.GlobalColor.transparent)
        self._pen_buf = {
            "layer": layer, "base": layer.image.copy(), "buf": buf,
            "opacity": max(1, min(100, int(self.pen_opacity))) / 100.0,
            "erase": bool(self.pen_transparent),
        }

    def _active_pen_buf(self, layer) -> dict | None:
        pb = self._pen_buf
        if pb is None or pb["layer"] is not layer or self.tool != Tool.PEN:
            return None
        return pb

    def _stroke_target(self, layer) -> QImage:
        """今の線を描き込む画像。不透明度つきペンなら線だけの別画像。"""
        pb = self._active_pen_buf(layer)
        return pb["buf"] if pb is not None else layer.image

    def _compose_pen_buffer(self, layer, dirty: QRect | None) -> None:
        """描く前の絵に、線だけの画像を指定の濃さで重ねてレイヤーに戻す。"""
        pb = self._active_pen_buf(layer)
        if pb is None or pb["base"].size() != layer.image.size():
            return
        r = QRect(0, 0, layer.image.width(), layer.image.height())
        if dirty is not None:
            r = dirty.intersected(r)
            if r.isEmpty():
                return
        p = QPainter(layer.image)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        p.drawImage(r.topLeft(), pb["base"], r)
        p.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_DestinationOut if pb["erase"]
            else QPainter.CompositionMode.CompositionMode_SourceOver)
        p.setOpacity(pb["opacity"])
        p.drawImage(r.topLeft(), pb["buf"], r)
        p.end()

    def _shift_pen_buffer(self, layer, shift) -> None:
        pb = self._active_pen_buf(layer)
        if pb is None:
            return
        for k in ("base", "buf"):
            pb[k] = _shifted_image(pb[k], layer.image.size(), shift)

    def _after_stroke_draw(self, layer, dirty: QRect | None) -> None:
        """ペン・消しゴムで描いた後の仕上げ（線を重ねる → 選択範囲・ロックで戻す）。"""
        self._compose_pen_buffer(layer, dirty)
        self._apply_clip_to_selection(dirty)

    # ── Shift+クリックで直線 ─────────────────────────────────────────────────

    def _shift_line_start(self, event, layer) -> QPoint | None:
        """Shift を押しながらのペン・消しゴムなら、前の線の終わり（キャンバス座標）。"""
        if self.tool not in (Tool.PEN, Tool.ERASER) or self._last_stroke_end is None:
            return None
        if not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            return None
        if layer.is_vector and (self.tool != Tool.PEN or self._vector_editing()):
            return None
        uid, end = self._last_stroke_end
        return QPoint(end) if uid == layer.uid else None

    def _press_line(self, layer, start_canvas: QPoint, lp: QPoint) -> QRect | None:
        """前の線の終わりから押した所まで直線を引き、そのまま続けて描けるようにする。"""
        start = QPoint(start_canvas.x() - getattr(layer, 'offset_x', 0),
                       start_canvas.y() - getattr(layer, 'offset_y', 0))
        self._taper_begin(layer, start)
        self._stroke_segment(layer, start, lp)
        self._stabilizer.push(lp)
        self._last_pos = lp
        return self._segment_dirty(start, lp)

    def _remember_stroke_end(self) -> None:
        """線を描き終えたとき、次の Shift+クリック用に終わりの位置を控える。"""
        layer = self.layer_stack.active
        if layer is None or layer.is_group or self.tool not in (Tool.PEN, Tool.ERASER):
            return
        if layer.is_vector:
            if (self.tool == Tool.PEN and self._vector_points
                    and not self._vector_editing()):
                x, y = self._vector_points[-1]
                self._last_stroke_end = (layer.uid, QPoint(round(x), round(y)))
            return
        if self._last_pos is not None:
            self._last_stroke_end = (layer.uid, QPoint(
                self._last_pos.x() + getattr(layer, 'offset_x', 0),
                self._last_pos.y() + getattr(layer, 'offset_y', 0)))

    def _commit_shape(self, img: QImage, a: QPoint, b: QPoint):
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = self._make_pen(self.pen_color, self.pen_size)
        if self.shape_fill == "fill":
            painter.setPen(Qt.PenStyle.NoPen)
        else:
            painter.setPen(pen)
        if self.shape_fill in ("fill", "both"):
            painter.setBrush(QBrush(self.pen_color))
        else:
            painter.setBrush(Qt.BrushStyle.NoBrush)
        r = QRect(a, b).normalized()
        if self.tool == Tool.RECT:
            painter.drawRect(r)
        elif self.tool == Tool.ELLIPSE:
            painter.drawEllipse(r)
        elif self.tool == Tool.LINE:
            painter.setPen(pen)
            painter.drawLine(a, b)
        painter.end()

    def _move_layer(self, layer: Layer, dx: int, dy: int):
        """レイヤーをオフセットで移動する。画像データはそのまま保持されるため、
        キャンバス外にはみ出た部分も失われない。"""
        if self._move_base_image is None:
            return
        if layer.is_vector:
            # ベクターは点がキャンバス座標なので、オフセットを動かしても
            # 描き直しで元の位置に戻ってしまう。点そのものを動かす。
            # 毎回「掴んだ時点の線」から作り直すことで誤差が積もらない。
            layer.strokes = [s.copy() for s in (self._move_base_strokes or [])]  # type: ignore
            layer.translate_strokes(dx, dy)  # type: ignore
            return
        layer.offset_x = self._move_base_offset[0] + dx
        layer.offset_y = self._move_base_offset[1] + dy

    # ── text ─────────────────────────────────────────────────────────────────

    def _ask_text(self):
        """クリック後に呼ばれる。_text_pos が確定している前提。"""
        # フォント・サイズはツールオプションで、色はペンの色で決まるので
        # ここでは文字だけを聞く（毎回ダイアログを3つ通らなくて済む）。
        text, ok = QInputDialog.getText(self, "テキスト入力", "テキスト:")
        if not (ok and text):
            self._text_pos = None
            return
        font = QFont(self.text_font_family)
        font.setPixelSize(max(1, int(self.text_size)))
        self.draw_text(text, font, QColor(self.pen_color))

    def draw_text(self, text: str, font: QFont, color: QColor):
        layer = self.layer_stack.active
        if not layer or layer.is_group or not self._text_pos:
            return
        self._save_history()
        # 文字がレイヤー画像からはみ出して切れないよう、描画範囲を先に確保する。
        # drawText の基準点はベースラインなので、上方向にも余白が要る。
        fm = QFontMetrics(font)
        br = fm.boundingRect(text)
        text_rect = QRect(self._text_pos.x() + br.x(), self._text_pos.y() + br.y(),
                          max(1, br.width()), max(1, br.height()))
        self._ensure_layer_bounds(layer, text_rect.adjusted(-2, -2, 2, 2))
        # 領域を広げた後で控えないと、元画像とサイズが合わなくなる
        self._clip_mask = self._lasso_mask
        self._begin_clip_to_selection(layer)
        painter = QPainter(layer.image)  # type: ignore
        painter.setFont(font)
        painter.setPen(QPen(color))
        # _text_pos はキャンバス座標。レイヤーローカル座標に変換して描く
        painter.drawText(self._text_pos - QPoint(getattr(layer, 'offset_x', 0),
                                                 getattr(layer, 'offset_y', 0)), text)
        painter.end()
        self._apply_clip_to_selection()
        self._text_pos = None
        self.update()

    # ── selection ────────────────────────────────────────────────────────────

    def copy_selection(self):
        layer = self.layer_stack.active
        if not layer or layer.is_group or not self._selection_rect:
            return
        src: QImage = layer.image  # type: ignore
        ox = getattr(layer, 'offset_x', 0)
        oy = getattr(layer, 'offset_y', 0)
        shifted = QRect(self._selection_rect.translated(-ox, -oy))
        region = src.copy(shifted)
        if self._lasso_mask:
            mask_crop = self._lasso_mask.copy(self._selection_rect)
            p = QPainter(region)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            p.drawImage(0, 0, mask_crop)
            p.end()
        self._clipboard_image = region
        self._clipboard_offset = self._selection_rect.topLeft()
        # ほかのアプリにも貼れるよう、OS のクリップボードにも載せる。
        cb = QApplication.clipboard()
        if cb is not None:
            cb.setImage(region)

    def cut_selection(self):
        """選択範囲をクリップボードにコピーしてから消去する。"""
        if not self._selection_rect:
            return
        if self._warn_locked(self.layer_stack.active):
            return
        self.copy_selection()
        self.delete_selection()

    def clipboard_payload(self) -> tuple[QImage, QPoint] | None:
        """貼り付ける絵と、置く位置（キャンバス座標）を返す。

        このアプリでコピーした絵ならコピー元と同じ位置に、ほかのアプリで
        コピーした絵ならキャンバスの中央に置く。
        """
        cb = QApplication.clipboard()
        if cb is not None and not cb.ownsClipboard():
            img = cb.image()
            if not img.isNull():
                img = img.convertToFormat(QImage.Format.Format_ARGB32)
                pos = QPoint((self.layer_stack.width - img.width()) // 2,
                             (self.layer_stack.height - img.height()) // 2)
                return img, pos
        if self._clipboard_image is None:
            return None
        return self._clipboard_image.copy(), QPoint(self._clipboard_offset)

    def delete_selection(self):
        layer = self.layer_stack.active
        if not layer or layer.is_group or not self._selection_rect:
            return
        if self._warn_locked(layer):
            return
        if layer.is_vector:
            # 絵を消しても線から描き直されて戻るので、掛かった線を消す。
            hit = set(self._strokes_in_selection(layer.strokes))
            keep = [s for i, s in enumerate(layer.strokes) if i not in hit]
            if len(keep) != len(layer.strokes):
                self._save_history()
                layer.strokes = keep
                layer.mark_dirty()
            self.deselect()
            return
        self._save_history()
        ox = getattr(layer, 'offset_x', 0)
        oy = getattr(layer, 'offset_y', 0)
        painter = QPainter(layer.image)  # type: ignore
        if self._lasso_mask:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            painter.drawImage(-ox, -oy, self._lasso_mask)
        else:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(self._selection_rect.translated(-ox, -oy), Qt.GlobalColor.transparent)
        painter.end()
        self._selection_rect = None
        self._lasso_mask = None
        self._selection_outline_path = None
        self.update()

    def select_all(self):
        self._selection_rect = QRect(0, 0, self.layer_stack.width, self.layer_stack.height)
        self._lasso_mask = None
        self._selection_outline_path = None
        self._sync_ant_timer()
        self.update()

    def has_selection(self) -> bool:
        """確定した選択範囲があるか。ドラッグ途中の投げなわは含めない。"""
        return self._selection_rect is not None

    def invert_selection(self):
        """選択範囲を反転する。矩形選択・投げなわ選択どちらにも対応。"""
        if not self._selection_rect:
            return
        w, h = self.layer_stack.width, self.layer_stack.height

        if self._lasso_mask is None:
            # 矩形選択 → 選択範囲マスクを作ってから反転する
            base_mask = QImage(w, h, QImage.Format.Format_ARGB32)
            base_mask.fill(Qt.GlobalColor.transparent)
            p = QPainter(base_mask)
            p.fillRect(self._selection_rect, Qt.GlobalColor.white)
            p.end()
        else:
            base_mask = self._lasso_mask

        inverted = QImage(w, h, QImage.Format.Format_ARGB32)
        inverted.fill(Qt.GlobalColor.white)
        ip = QPainter(inverted)
        ip.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
        ip.drawImage(0, 0, base_mask)
        ip.end()

        inv_ptr = inverted.bits(); inv_ptr.setsize(h * w * 4)
        inv_arr = np.frombuffer(inv_ptr, dtype=np.uint8).reshape(h, w, 4)
        inv_mask = inv_arr[:, :, 3] > 0
        if not inv_mask.any():
            # 全面が選ばれていた場合、反転すると何も残らない。
            # 空の選択を持ち回ると以降の処理が対象なしで詰まるので解除する。
            self.deselect()
            return

        ys, xs = np.nonzero(inv_mask)
        self._lasso_mask = inverted
        self._selection_rect = QRect(int(xs.min()), int(ys.min()),
                                     int(xs.max() - xs.min() + 1),
                                     int(ys.max() - ys.min() + 1))
        self._selection_outline_path = _outline_path_from_mask(inv_mask)
        self._lasso_path_points = []
        self._lasso_points = []
        self._sync_ant_timer()
        self.update()

    def _strokes_in_selection(self, strokes) -> list[int]:
        """選択範囲に実際に描かれた部分が掛かっている線の番号。

        外接矩形だけで判定すると、斜めの長い線は選択の近くを通るだけで
        丸ごと消えてしまう。線を選択範囲の大きさに描いて重なりを見る。
        """
        from vector import draw_strokes
        sel = self._selection_rect
        if sel is None or sel.isEmpty():
            return []
        self_mask = None
        if self._lasso_mask is not None:
            self_mask = _alpha_array(self._lasso_mask.copy(sel)) > 0
        out = []
        for i, st in enumerate(strokes):
            if not st.points or not st.bounds().intersects(QRectF(sel)):
                continue
            img = QImage(sel.width(), sel.height(), QImage.Format.Format_ARGB32)
            img.fill(Qt.GlobalColor.transparent)
            p = QPainter(img)
            draw_strokes(p, [st], sel.x(), sel.y())
            p.end()
            drawn = _alpha_array(img) > 0
            if self_mask is not None:
                drawn &= self_mask
            if drawn.any():
                out.append(i)
        return out

    def deselect(self):
        if self._transform_image:
            # CLIP STUDIO と同じく、変形中の解除は確定してから解除する。
            # 捨てると動かした絵が黙って元に戻ってしまう。
            self._commit_transform()
        self._selection_rect = None
        self._lasso_mask = None
        self._lasso_path_points = []
        self._lasso_points = []
        self._selection_outline_path = None
        self._sync_ant_timer()
        self.update()

    def select_layer_alpha(self, layer=None, threshold: int = 10) -> bool:
        """レイヤーの不透明部分（alpha > threshold）の形で選択範囲を作る。
        （レイヤーパネルでサムネイルをCtrlクリックした時などに呼ばれる想定）
        layer省略時はアクティブレイヤーを使う。グループレイヤーは対象外。"""
        if layer is None:
            layer = self.layer_stack.active
        if not layer or layer.is_group:
            return False

        src: QImage = layer.image  # type: ignore
        lw, lh = src.width(), src.height()
        if lw == 0 or lh == 0:
            return False
        img = src.convertToFormat(QImage.Format.Format_ARGB32)
        ptr = img.bits(); ptr.setsize(lh * lw * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(lh, lw, 4)
        alpha = arr[:, :, 3]
        opaque = (alpha > threshold).astype(np.uint8)
        if not opaque.any():
            return False

        # レイヤーのローカル座標系からキャンバス座標系へ offset_x/offset_y 分ずらして配置する
        # （_apply_lasso_fill と同じ考え方）。
        lox = getattr(layer, 'offset_x', 0)
        loy = getattr(layer, 'offset_y', 0)
        cw, ch = self.layer_stack.width, self.layer_stack.height

        canvas_mask = np.zeros((ch, cw), dtype=np.uint8)
        sx0, sy0 = max(0, -lox), max(0, -loy)
        dx0, dy0 = max(0, lox), max(0, loy)
        copy_w = min(lw - sx0, cw - dx0)
        copy_h = min(lh - sy0, ch - dy0)
        if copy_w <= 0 or copy_h <= 0:
            return False
        canvas_mask[dy0:dy0 + copy_h, dx0:dx0 + copy_w] = \
            opaque[sy0:sy0 + copy_h, sx0:sx0 + copy_w]
        if not canvas_mask.any():
            return False

        ys, xs = np.nonzero(canvas_mask)
        sel = QRect(int(xs.min()), int(ys.min()),
                    int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))

        mask_img = QImage(cw, ch, QImage.Format.Format_ARGB32)
        mask_img.fill(Qt.GlobalColor.transparent)
        mptr = mask_img.bits(); mptr.setsize(ch * cw * 4)
        mask_arr = np.frombuffer(mptr, dtype=np.uint8).reshape(ch, cw, 4)
        mask_arr[canvas_mask > 0] = [255, 255, 255, 255]

        outline = _outline_path_from_mask(canvas_mask)

        self._selection_rect = sel
        self._lasso_mask = mask_img
        self._lasso_path_points = []
        self._lasso_points = []
        self._selection_outline_path = outline
        self._sync_ant_timer()
        self.update()
        return True

    # ── transform ────────────────────────────────────────────────────────────

    def _warn_locked(self, layer) -> bool:
        """ロック中なら理由を出して True を返す。"""
        if self.is_locked(layer):
            self.status_message.emit("このレイヤーはロックされています。レイヤーパネルの錠マークを解除してください。")
            return True
        return False

    def _lift_selection(self, layer: Layer) -> bool:
        """選択範囲をフローティング化する。ピクセル消去は確定時（_commit_transform）に行う。

        変形の入口はすべてここを通るので、ロックの確認もここで行う。
        持ち上げなかったときは False を返す。
        """
        if self._warn_locked(layer):
            return False
        self._transform_flip = (False, False)
        if layer.is_vector:
            return self._lift_vector(layer)
        if not self._selection_rect:
            self._selection_rect = QRect(0, 0, self.layer_stack.width, self.layer_stack.height)

        src: QImage = layer.image
        ox = getattr(layer, 'offset_x', 0)
        oy = getattr(layer, 'offset_y', 0)
        shifted = QRect(self._selection_rect.translated(-ox, -oy))
        region = src.copy(shifted).convertToFormat(
            QImage.Format.Format_ARGB32)
        if self._lasso_mask:
            mask_crop = self._lasso_mask.copy(self._selection_rect)
            p = QPainter(region)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            p.drawImage(0, 0, mask_crop)
            p.end()

        self._transform_image = region
        self._transform_rect = QRectF(self._selection_rect)
        self._transform_orig_rect = QRectF(self._selection_rect)  # %計算の基準
        self._transform_angle = 0.0
        self._custom_pivot = QPointF(self._selection_rect.center())
        if self._mesh_mode:
            self._init_mesh_grid()
            self._perspective_corners = None
        elif self._perspective_mode:
            r = self._transform_rect
            self._perspective_corners = [
                QPointF(r.left(), r.top()), QPointF(r.right(), r.top()),
                QPointF(r.right(), r.bottom()), QPointF(r.left(), r.bottom()),
            ]
        else:
            self._perspective_corners = None
        # 変形確定先レイヤーと消去範囲をここで固定する
        self._transform_layer = layer
        self._transform_erase_rect = QRect(self._selection_rect)
        self._transform_erase_mask = self._lasso_mask
        self._selection_rect = None
        self._lasso_mask = None
        self._selection_outline_path = None
        self.update()
        return True

    def _lift_vector(self, layer) -> bool:
        """ベクターレイヤーの線を持ち上げる。

        選択範囲に掛かった線を丸ごと持ち上げ（選択が無ければ全部）、確定時に
        点を動かす。線の途中で切ると別の線になってしまうので、線単位で扱う。
        メッシュ・自由変形は点の移動で表せないので標準の変形だけにする。
        """
        from vector import draw_strokes
        sel = QRectF(self._selection_rect) if self._selection_rect else None
        chosen = [i for i, s in enumerate(layer.strokes)
                  if s.points and (sel is None or s.bounds().intersects(sel))]
        if not chosen:
            self.status_message.emit("変形する線がありません。")
            return False
        if self._mesh_mode or self._perspective_mode:
            self.status_message.emit(
                "ベクターレイヤーは拡大縮小・回転だけ使えます（メッシュ・自由変形は"
                "ラスタライズしてから）。")
            self._mesh_mode = False
            self._perspective_mode = False

        area = QRectF()
        for i in chosen:
            area = area.united(layer.strokes[i].bounds())
        rect = area.toAlignedRect()
        img = QImage(max(1, rect.width()), max(1, rect.height()),
                     QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        draw_strokes(p, [layer.strokes[i] for i in chosen], rect.x(), rect.y())
        p.end()

        # 持ち上げた線を除いた絵を先に作っておく（プレビューのたびに描き直さない）
        all_strokes = layer.strokes
        layer.strokes = [s for i, s in enumerate(all_strokes) if i not in chosen]
        layer.mark_dirty()
        self._transform_vector_rest = (layer.image.copy(), layer.offset_x, layer.offset_y)
        layer.strokes = all_strokes
        layer.mark_dirty()

        self._transform_vector_idx = chosen
        self._transform_image = img
        self._transform_rect = QRectF(rect)
        self._transform_orig_rect = QRectF(rect)
        self._transform_angle = 0.0
        self._custom_pivot = QPointF(rect.center())
        self._perspective_corners = None
        self._mesh_grid = None
        self._transform_layer = layer
        self._transform_erase_rect = None
        self._transform_erase_mask = None
        self._selection_rect = None
        self._lasso_mask = None
        self._selection_outline_path = None
        self.update()
        return True

    def _commit_vector_transform(self, layer) -> None:
        """持ち上げた線の点に、プレビューと同じ拡大縮小・反転・回転を掛ける。"""
        o = self._transform_orig_rect
        r = self._transform_rect
        sx = r.width() / o.width() if o.width() else 1.0
        sy = r.height() / o.height() if o.height() else 1.0
        fx, fy = self._transform_flip
        pv = self._pivot_point()
        rad = math.radians(self._transform_angle)
        c, s = math.cos(rad), math.sin(rad)

        def rot(dx: float, dy: float) -> tuple[float, float]:
            return dx * c - dy * s, dx * s + dy * c

        def map_pt(x: float, y: float) -> tuple[float, float]:
            u = (x - o.left()) / o.width() if o.width() else 0.0
            v = (y - o.top()) / o.height() if o.height() else 0.0
            if fx:
                u = 1.0 - u
            if fy:
                v = 1.0 - v
            dx, dy = rot(r.left() + u * r.width() - pv.x(),
                         r.top() + v * r.height() - pv.y())
            return pv.x() + dx, pv.y() + dy

        # ハンドルは点からの相対ベクトルなので、平行移動を除いた部分だけ掛ける。
        lx = -sx if fx else sx
        ly = -sy if fy else sy

        def map_vec(dx: float, dy: float) -> tuple[float, float]:
            return rot(dx * lx, dy * ly)

        for i in self._transform_vector_idx or []:
            st = layer.strokes[i].copy()
            st.points = [map_pt(x, y) for x, y in st.points]
            st.handles = [None if h is None else (*map_vec(h[0], h[1]), *map_vec(h[2], h[3]))
                          for h in st.handles]
            st.width = max(0.1, st.width * (abs(sx) + abs(sy)) / 2.0)
            layer.strokes[i] = st
        layer.mark_dirty()

    def _hit_transform_handle(self, wp: QPointF) -> str | None:
        if not self._transform_rect:
            return None
        c2w = self._c2w()

        if self._mesh_grid:
            grid = self._mesh_grid
            for r_idx in range(len(grid)):
                for c_idx in range(len(grid[0])):
                    wpt = c2w.map(grid[r_idx][c_idx])
                    if (wpt - wp).manhattanLength() < HANDLE_HIT_RADIUS:
                        self._mesh_drag_idx = (r_idx, c_idx)
                        return 'mesh_point'
            rot_w = self._rotation_handle_widget()
            if rot_w and (rot_w - wp).manhattanLength() < HANDLE_HIT_RADIUS:
                return 'rotate'
            # 格子内クリック → move
            w2c = self._w2c()
            cp = w2c.map(wp)
            corners = [grid[0][0], grid[0][-1], grid[-1][-1], grid[-1][0]]
            if self._point_in_quad(cp, corners):
                return 'move'
            return None

        if self._perspective_corners:
            corner_names = ['tl', 'tr', 'br', 'bl']
            for i, name in enumerate(corner_names):
                wpt = c2w.map(self._perspective_corners[i])
                if (wpt - wp).manhattanLength() < HANDLE_HIT_RADIUS:
                    self._perspective_drag_idx = i
                    return name
            rot_w = self._rotation_handle_widget()
            if rot_w and (rot_w - wp).manhattanLength() < HANDLE_HIT_RADIUS:
                return 'rotate'
            w2c = self._w2c()
            cp = w2c.map(wp)
            if self._point_in_quad(cp, self._perspective_corners):
                return 'move'
            return None

        tm = self._transform_matrix()
        r = self._transform_rect

        corner_names = ['tl', 'tr', 'br', 'bl']
        corners_c = [
            QPointF(r.left(), r.top()), QPointF(r.right(), r.top()),
            QPointF(r.right(), r.bottom()), QPointF(r.left(), r.bottom()),
        ]
        for name, cc in zip(corner_names, corners_c):
            wpt = c2w.map(tm.map(cc))
            if (wpt - wp).manhattanLength() < HANDLE_HIT_RADIUS:
                return name

        rot_w = self._rotation_handle_widget()
        if rot_w and (rot_w - wp).manhattanLength() < HANDLE_HIT_RADIUS:
            return 'rotate'

        if self._pivot_mode == "custom":
            pv_c = self._pivot_point()
            pv_w = c2w.map(tm.map(pv_c))
            if (pv_w - wp).manhattanLength() < HANDLE_HIT_RADIUS + 4:
                return 'pivot'

        inv_tm, ok = tm.inverted()
        if ok:
            w2c = self._w2c()
            cp = w2c.map(wp)
            local = inv_tm.map(cp)
            if r.contains(local):
                return 'move'
        return None

    @staticmethod
    def _point_in_quad(pt: QPointF, quad: list[QPointF]) -> bool:
        """点が凸四角形内にあるかクロス積で判定。"""
        n = len(quad)
        sign = None
        for i in range(n):
            x1, y1 = quad[i].x(), quad[i].y()
            x2, y2 = quad[(i + 1) % n].x(), quad[(i + 1) % n].y()
            cross = (x2 - x1) * (pt.y() - y1) - (y2 - y1) * (pt.x() - x1)
            if cross != 0:
                s = cross > 0
                if sign is None:
                    sign = s
                elif s != sign:
                    return False
        return True

    def _drag_transform(self, wp: QPointF):
        if not self._transform_rect_start or not self._transform_drag_start:
            return

        h = self._transform_handle
        w2c = self._w2c()

        if self._mesh_grid and self._mesh_grid_start and h not in ('rotate', 'pivot'):
            start_c = w2c.map(self._transform_drag_start)
            cur_c = w2c.map(wp)
            dx = cur_c.x() - start_c.x()
            dy = cur_c.y() - start_c.y()
            if h == 'mesh_point':
                ri, ci = self._mesh_drag_idx
                self._mesh_grid[ri][ci] = QPointF(
                    self._mesh_grid_start[ri][ci].x() + dx,
                    self._mesh_grid_start[ri][ci].y() + dy)
            elif h == 'move':
                for ri in range(len(self._mesh_grid)):
                    for ci in range(len(self._mesh_grid[0])):
                        self._mesh_grid[ri][ci] = QPointF(
                            self._mesh_grid_start[ri][ci].x() + dx,
                            self._mesh_grid_start[ri][ci].y() + dy)
            self.update()
            return

        if self._perspective_corners and self._perspective_corners_start and h not in ('rotate', 'pivot'):
            start_c = w2c.map(self._transform_drag_start)
            cur_c = w2c.map(wp)
            dx = cur_c.x() - start_c.x()
            dy = cur_c.y() - start_c.y()
            if h in ('tl', 'tr', 'br', 'bl'):
                idx = self._perspective_drag_idx
                self._perspective_corners[idx] = QPointF(
                    self._perspective_corners_start[idx].x() + dx,
                    self._perspective_corners_start[idx].y() + dy)
            elif h == 'move':
                for i in range(4):
                    self._perspective_corners[i] = QPointF(
                        self._perspective_corners_start[i].x() + dx,
                        self._perspective_corners_start[i].y() + dy)
            self.update()
            return

        if h == 'pivot':
            cp = w2c.map(wp)
            self._custom_pivot = cp
            self.update()
            return

        if h == 'rotate':
            # 回転: ドラッグ開始点・現在点とピボットの角度差
            if self._pivot_mode == "custom" and self._custom_pivot is not None:
                center_c = self._custom_pivot
            else:
                ax, ay = self._transform_pivot
                rs = self._transform_rect_start
                center_c = QPointF(rs.left() + rs.width() * ax / 2.0,
                                   rs.top() + rs.height() * ay / 2.0)
            center_w = self._c2w().map(center_c)
            start_ang = math.degrees(math.atan2(
                self._transform_drag_start.y() - center_w.y(),
                self._transform_drag_start.x() - center_w.x()))
            cur_ang = math.degrees(math.atan2(
                wp.y() - center_w.y(),
                wp.x() - center_w.x()))
            self._transform_angle = self._transform_angle_start + (cur_ang - start_ang)
            delta_ang = self._transform_angle - self._transform_angle_start
            # メッシュ/自由変形（4隅）は矩形+角度ではなく絶対座標点で状態を持つため、
            # 回転ハンドルの操作はドラッグ開始時点のスナップショットをピボット中心に
            # 回転させて反映する（標準モードの _transform_matrix() 相当をここで手動適用）。
            if self._mesh_grid and self._mesh_grid_start:
                rot = QTransform()
                rot.translate(center_c.x(), center_c.y())
                rot.rotate(delta_ang)
                rot.translate(-center_c.x(), -center_c.y())
                self._mesh_grid = [[rot.map(p) for p in row] for row in self._mesh_grid_start]
            elif self._perspective_corners and self._perspective_corners_start:
                rot = QTransform()
                rot.translate(center_c.x(), center_c.y())
                rot.rotate(delta_ang)
                rot.translate(-center_c.x(), -center_c.y())
                self._perspective_corners = [rot.map(p) for p in self._perspective_corners_start]
            self.update()
            return

        start_c = w2c.map(self._transform_drag_start)
        cur_c = w2c.map(wp)
        dx = cur_c.x() - start_c.x()
        dy = cur_c.y() - start_c.y()
        r = QRectF(self._transform_rect_start)
        shift = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)

        if h == 'move':
            r.translate(dx, dy)
        elif h in ('tl', 'tr', 'bl', 'br'):
            # _transform_rect は回転前のローカル矩形。角ハンドルは _transform_matrix()
            # で回転させた見た目の位置に表示されるため、ドラッグ量（画面＝キャンバス
            # 座標系のベクトル）もその逆回転をかけてローカル座標系に戻してから
            # 辺に適用する必要がある。これをしないと、回転がかかった状態で角を
            # ドラッグしたときにマウスの動きと伸縮方向が一致しない
            # （例: 90度回転時に上下左右が入れ替わる）。
            if self._transform_angle != 0.0:
                rot_only = QTransform()
                rot_only.rotate(-self._transform_angle)
                delta = rot_only.map(QPointF(dx, dy)) - rot_only.map(QPointF(0, 0))
                dx, dy = delta.x(), delta.y()
            orig_w = self._transform_rect_start.width()
            orig_h = self._transform_rect_start.height()
            # ratio は変形開始時のアスペクト比。orig_w/orig_h のどちらかが 0 の場合は
            # Shift 拘束を適用しない（ゼロ除算回避）
            use_shift = shift and orig_w > 0 and orig_h > 0
            ratio = orig_w / orig_h if use_shift else 1.0

            if h == 'tl':
                new_pt = r.topLeft() + QPointF(dx, dy)
                if use_shift:
                    new_pt = _constrain_corner_shift(new_pt, r.right(), r.bottom(), ratio)
                r.setTopLeft(new_pt)
            elif h == 'tr':
                new_pt = r.topRight() + QPointF(dx, dy)
                if use_shift:
                    new_pt = _constrain_corner_shift(new_pt, r.left(), r.bottom(), ratio)
                r.setTopRight(new_pt)
            elif h == 'bl':
                new_pt = r.bottomLeft() + QPointF(dx, dy)
                if use_shift:
                    new_pt = _constrain_corner_shift(new_pt, r.right(), r.top(), ratio)
                r.setBottomLeft(new_pt)
            elif h == 'br':
                new_pt = r.bottomRight() + QPointF(dx, dy)
                if use_shift:
                    new_pt = _constrain_corner_shift(new_pt, r.left(), r.top(), ratio)
                r.setBottomRight(new_pt)

        if r.width() > MIN_TRANSFORM_SIZE and r.height() > MIN_TRANSFORM_SIZE:
            self._transform_rect = r
        self.update()

    def _ensure_layer_bounds(self, layer, canvas_rect: QRect):
        """canvas_rect（キャンバス座標系）が layer.image に収まるよう、必要なら
        image を拡張し offset_x/offset_y を調整する。拡大縮小でキャンバス外に
        絵がはみ出すと、layer.image のサイズで描画がクリップされてしまうため。
        編集中はここでは上限を設けない（はみ出した絵をキャンバスを広げて後で
        復活させられるようにするため）。肥大化したファイルが保存後に開けなく
        なる問題は、保存時の _trim_layer_overflow_for_save 側でキャンバス外の
        不透明部分を切り詰めることで対処する。"""
        ox = getattr(layer, 'offset_x', 0)
        oy = getattr(layer, 'offset_y', 0)
        img: QImage = layer.image  # type: ignore
        cur = QRect(ox, oy, img.width(), img.height())
        needed = cur.united(canvas_rect)
        if needed == cur:
            return
        new_img = QImage(needed.width(), needed.height(), QImage.Format.Format_ARGB32)
        new_img.fill(Qt.GlobalColor.transparent)
        p = QPainter(new_img)
        p.drawImage(ox - needed.x(), oy - needed.y(), img)
        p.end()
        layer.image = new_img  # type: ignore
        layer.offset_x = needed.x()  # type: ignore
        layer.offset_y = needed.y()  # type: ignore

    def _grow_for_draw(self, layer, cp: QPoint, margin: int) -> QPoint:
        """cp（キャンバス座標）を中心に margin ぶんの余白が layer.image に
        収まるよう拡張する。移動ツールでレイヤーをずらすと offset だけが
        変わり image は広がらないため、そのまま描くとバッファ外の筆跡が
        黙って捨てられてしまう。戻り値は offset の変化量で、レイヤー
        ローカル座標でキャッシュしている座標（_last_pos など）の補正に使う。"""
        if layer is None or getattr(layer, 'is_group', False):
            return QPoint(0, 0)
        before_x = getattr(layer, 'offset_x', 0)
        before_y = getattr(layer, 'offset_y', 0)
        m = max(1, int(margin))
        rect = QRect(cp.x() - m, cp.y() - m, m * 2 + 1, m * 2 + 1)
        if self.symmetry_enabled and self.tool == Tool.PEN:
            # 対称側の筆跡もバッファに収める
            mx = 2 * (self.layer_stack.width // 2) - cp.x()
            rect = rect.united(QRect(mx - m, cp.y() - m, m * 2 + 1, m * 2 + 1))
        self._ensure_layer_bounds(layer, rect)
        return QPoint(before_x - getattr(layer, 'offset_x', 0),
                      before_y - getattr(layer, 'offset_y', 0))

    def _draw_margin(self) -> int:
        """描画ツールが1回のスタンプで広がりうる半径。"""
        if self.tool == Tool.ERASER:
            return int(self.eraser_size)
        if self.tool == Tool.BLUR:
            return int(self.blur_size)
        return int(self.pen_size)

    def _commit_transform(self):
        layer = self._transform_layer or self.layer_stack.active
        if not layer or layer.is_group or not self._transform_image or not self._transform_rect:
            return

        self._save_history()

        if self._transform_vector_idx is not None and layer.is_vector:
            self._commit_vector_transform(layer)
            self._clear_transform_state()
            return

        # 変形結果を先に確定させてから、キャンバス座標での実際の描画範囲に
        # layer.image が収まるようにする（拡大縮小・回転でキャンバス外にはみ出す
        # と layer.image のサイズでクリップされてしまうため）。
        mesh_result = self._warp_mesh_image() if self._mesh_grid else None
        perspective_result = self._warp_perspective_image() if (
            not self._mesh_grid and self._perspective_corners) else None

        if mesh_result:
            warped_img, wx, wy = mesh_result
            target_rect = QRect(wx, wy, warped_img.width(), warped_img.height())
        elif perspective_result:
            warped_img, wx, wy = perspective_result
            target_rect = QRect(wx, wy, warped_img.width(), warped_img.height())
        else:
            pv = self._pivot_point()
            m = QTransform()
            m.translate(pv.x(), pv.y())
            m.rotate(self._transform_angle)
            m.translate(-pv.x(), -pv.y())
            target_rect = m.mapRect(self._transform_rect).toAlignedRect()

        self._ensure_layer_bounds(layer, target_rect)

        img: QImage = layer.image  # type: ignore
        ox = getattr(layer, 'offset_x', 0)
        oy = getattr(layer, 'offset_y', 0)
        ep = QPainter(img)
        if self._transform_erase_mask:
            ep.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            ep.drawImage(-ox, -oy, self._transform_erase_mask)
        elif self._transform_erase_rect:
            ep.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            ep.fillRect(self._transform_erase_rect.translated(-ox, -oy), Qt.GlobalColor.transparent)
        ep.end()

        if mesh_result:
            warped_img, wx, wy = mesh_result
            painter = QPainter(img)
            painter.drawImage(wx - ox, wy - oy, warped_img)
            painter.end()
        elif perspective_result:
            warped_img, wx, wy = perspective_result
            painter = QPainter(img)
            painter.drawImage(wx - ox, wy - oy, warped_img)
            painter.end()
        else:
            r = self._transform_rect
            pv = self._pivot_point()
            painter = QPainter(img)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.translate(-ox, -oy)
            painter.translate(pv.x(), pv.y())
            painter.rotate(self._transform_angle)
            painter.translate(-pv.x(), -pv.y())
            painter.drawImage(r, self._transform_image)
            painter.end()

        self._clear_transform_state()

    def _clear_transform_state(self):
        """変形中の状態をすべて捨てる（確定・キャンセル共通）。"""
        self._transform_image = None
        self._transform_rect = None
        self._transform_orig_rect = None
        self._transform_angle = 0.0
        self._transform_layer = None
        self._transform_erase_rect = None
        self._transform_erase_mask = None
        self._transform_vector_idx = None
        self._transform_vector_rest = None
        self._transform_flip = (False, False)
        self._perspective_corners = None
        self._perspective_corners_start = None
        self._perspective_drag_idx = -1
        self._mesh_grid = None
        self._mesh_grid_start = None
        self._mesh_drag_idx = (-1, -1)
        # 変形モードは確定のたびに標準へ戻す。ツールオプションパネルの
        # 「変形モード」コンボは毎回 index 0（標準）で再生成されるため、
        # ここでフラグを残すと表示（標準）と実際の内部状態（前回選んだ
        # パース/メッシュ）がずれ、2回目以降の変形でハンドルが出ない・
        # 掴めないバグになる。
        self._perspective_mode = False
        self._mesh_mode = False
        self.update()

    @property
    def transform_mode(self) -> str:
        """現在の変形モードを "standard" / "perspective" / "mesh" で返す。
        ツールオプションパネルの表示をキャンバスの実状態と一致させるために使う。"""
        if self._mesh_mode:
            return "mesh"
        if self._perspective_mode:
            return "perspective"
        return "standard"

    def set_transform_mode(self, mode: str):
        """"standard" / "perspective" / "mesh" を切り替える。"""
        if mode != "standard" and self._transform_vector_idx is not None:
            # ベクターの線はメッシュや自由変形の形を点で表せない。
            self.status_message.emit(
                "ベクターレイヤーは拡大縮小・回転だけ使えます（メッシュ・自由変形は"
                "ラスタライズしてから）。")
            return
        self._perspective_mode = (mode == "perspective")
        self._mesh_mode = (mode == "mesh")
        if self._transform_image and self._transform_rect:
            self._perspective_corners = None
            self._mesh_grid = None
            self._transform_angle = 0.0
            if mode == "perspective":
                self._perspective_corners = self._transform_corners_canvas()
            elif mode == "mesh":
                self._init_mesh_grid()
            self.update()

    def set_mesh_div(self, n: int):
        self._mesh_div = n
        if self._mesh_mode and self._transform_image and self._transform_rect:
            self._init_mesh_grid()
            self.update()

    def lift_whole_layer(self) -> bool:
        """アクティブレイヤー全体をフローティング化して変形モードに入る。選択範囲は使わない。
        拡大縮小・回転の基準点がキャンバス中心ではなくイラスト（不透明部分）の中心になるよう、
        select_layer_alpha で不透明部分の外接矩形を求めてからliftする。
        レイヤーが全透明などで不透明部分が無い場合はキャンバス全体にフォールバックする。"""
        layer = self.layer_stack.active
        if not layer or layer.is_group or self._warn_locked(layer):
            return False
        # 履歴は確定時に積む。ここで積むとキャンセルしたときに
        # 何もしていない取り消し手順が1つ残ってしまう。
        self._selection_rect = None
        self._lasso_mask = None
        self._selection_outline_path = None
        if not layer.is_vector:
            # 薄いぼかしやキャンバス外にはみ出した部分も含めて丸ごと持ち上げる。
            # 「選択範囲を作ってから」だと alpha の閾値とキャンバス枠で
            # 切れて、元の位置に薄い残像やはみ出し部分が取り残された。
            bbox = self._layer_content_rect(layer)
            if bbox is not None:
                self._selection_rect = bbox
        return self._lift_selection(layer)  # type: ignore

    @staticmethod
    def _layer_content_rect(layer) -> QRect | None:
        """レイヤー画像で alpha>0 の範囲（キャンバス座標）。空なら None。"""
        img = layer.image.convertToFormat(QImage.Format.Format_ARGB32)
        w, h = img.width(), img.height()
        ptr = img.constBits()
        ptr.setsize(h * w * 4)
        alpha = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)[:, :, 3]
        rows = np.flatnonzero(alpha.any(axis=1))
        cols = np.flatnonzero(alpha.any(axis=0))
        if rows.size == 0:
            return None
        return QRect(int(cols[0]) + layer.offset_x, int(rows[0]) + layer.offset_y,
                     int(cols[-1] - cols[0]) + 1, int(rows[-1] - rows[0]) + 1)

    def apply_transform_percentage(self, scale_x_pct: float, scale_y_pct: float, angle_deg: float,
                                    offset_x: float = 0.0, offset_y: float = 0.0):
        """フローティング変形中に拡縮率(%)と回転角を適用してリアルタイムプレビューを更新する。
        scale_x_pct / scale_y_pct は元サイズを100%として指定する。
        offset_x / offset_y は元の中心からのキャンバス座標系での移動量（px）。
        _transform_orig_rect の中心 + offset を新しい中心としてサイズを変える。"""
        if not self._transform_image or not self._transform_orig_rect:
            return
        orig = self._transform_orig_rect
        new_w = orig.width()  * scale_x_pct / 100.0
        new_h = orig.height() * scale_y_pct / 100.0
        cx = orig.center().x() + offset_x
        cy = orig.center().y() + offset_y
        self._transform_rect = QRectF(cx - new_w / 2, cy - new_h / 2, new_w, new_h)
        self._transform_angle = angle_deg
        self.update()

    def flip_transform_horizontal(self):
        """フローティング変形中の画像を左右反転する。"""
        if not self._transform_image:
            return
        self._transform_image = self._transform_image.mirrored(True, False)
        fx, fy = self._transform_flip
        self._transform_flip = (not fx, fy)
        self.update()

    def flip_transform_vertical(self):
        """フローティング変形中の画像を上下反転する。"""
        if not self._transform_image:
            return
        self._transform_image = self._transform_image.mirrored(False, True)
        fx, fy = self._transform_flip
        self._transform_flip = (fx, not fy)
        self.update()

    def cancel_transform(self):
        """変形をキャンセル。lift時にはピクセル消去しないので単純破棄でOK。"""
        if not self._transform_image:
            return
        self._clear_transform_state()

    def reset_state(self):
        """新規/開くなどでキャンバスを差し替える前に一切の作業状態を破棄する。"""
        self._end_stroke_cache()
        self._taper_stroke = None
        self._pen_buf = None
        self._last_stroke_end = None
        self._move_base_image = None
        self._move_base_strokes = None
        self._vector_points = None
        self._vector_selected = None
        self._move_base_pos = None
        self._move_pending_history = None
        self._move_group_bases = None
        self._transform_image = None
        self._transform_rect = None
        self._transform_angle = 0.0
        self._transform_handle = None
        self._transform_drag_start = None
        self._transform_rect_start = None
        self._transform_angle_start = 0.0
        self._transform_layer = None
        self._transform_erase_rect = None
        self._transform_erase_mask = None
        self._transform_vector_idx = None
        self._transform_vector_rest = None
        self._transform_flip = (False, False)
        self._perspective_corners = None
        self._perspective_corners_start = None
        self._perspective_drag_idx = -1
        self._mesh_grid = None
        self._mesh_grid_start = None
        self._mesh_drag_idx = (-1, -1)
        self._perspective_mode = False
        self._mesh_mode = False
        self._selection_rect = None
        self._lasso_mask = None
        self._selection_outline_path = None
        self._lasso_points = []
        self._preview_start = None
        self._preview_end = None
        self._drawing = False
        self._last_pos = None
        self._text_pos = None
        self._lift_pending = False
        self._lift_pending_wp = None

    def wheelEvent(self, event):
        mods = event.modifiers()
        delta = event.angleDelta().y()
        if mods & Qt.KeyboardModifier.ControlModifier:
            # Ctrl+スクロール → ズーム（カーソル下の点を固定する）
            # タッチパッドは小刻みな delta を大量に送るので、量に比例させる
            # （1回ごとに 15% 動かすと速すぎて操作できない）。
            factor = 1.15 ** (delta / 120)
            self.set_zoom(self.zoom * factor, event.position())
            event.accept()
        else:
            super().wheelEvent(event)

    def _owns_key(self, key) -> bool:
        """メニューのショートカットより先に、キャンバス自身が受けるべきキーか。

        Delete はメニューの「消去」にも割り当てているが、線を選んでいる
        ときやパスを打っているときはキャンバスの操作のほうが意図に近い。
        """
        if key == Qt.Key.Key_Escape:
            return bool(self._path_pick_active or self._vector_selected is not None
                        or self._transform_image or self._selection_rect)
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            return bool((self._path_pick_active and self._path_pick_points)
                        or self._vector_selected is not None)
        return False

    def event(self, ev):
        # メニューのショートカットは、受け取ったウィジェットが ShortcutOverride を
        # accept しない限り keyPressEvent より先に発火してキーを奪う。
        if ev.type() == QEvent.Type.ShortcutOverride and self._owns_key(ev.key()):
            ev.accept()
            return True
        # タッチパッドの2本指ピンチ → ズーム（2本指ドラッグはスクロールで動く）
        if (ev.type() == QEvent.Type.NativeGesture
                and ev.gestureType() == Qt.NativeGestureType.ZoomNativeGesture):
            self.set_zoom(self.zoom * (1.0 + ev.value()), ev.position())
            ev.accept()
            return True
        return super().event(ev)

    def keyPressEvent(self, event):
        # パスピックモード中のキー操作
        if self._path_pick_active:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._confirm_path_pick()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Escape:
                self.cancel_path_pick()
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete) and self._path_pick_points:
                self._path_pick_points.pop()
                self.update()
                event.accept()
                return

        # ベクター線を選んでいるとき。変形より先に見る（変形中は線を選べない）
        if self._vector_selected is not None:
            if event.key() == Qt.Key.Key_Escape:
                self._vector_selected = None
                self._vector_drag = None
                self.vector_selection_changed.emit()
                self.update()
                event.accept()
                return
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self.delete_selected_stroke()
                event.accept()
                return

        if self._transform_image:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._commit_transform()
                event.accept()
                return
            if event.key() == Qt.Key.Key_Escape:
                self.cancel_transform()
                event.accept()
                return

        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._selection_rect or self._lasso_path_points:
                self.deselect()
                event.accept()
                return

        if event.key() == Qt.Key.Key_Escape:
            if self._selection_rect:
                self.deselect()
                event.accept()
                return

        # Space → パンニングモード開始（パンニング中は他のキー操作を無視）
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._panning = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if self._panning:
            event.accept()
            return

        # 移動ツール: 矢印キーで1px（Shift+矢印で10px）移動
        if self.tool == Tool.MOVE:
            arrow_map = {
                Qt.Key.Key_Left:  (-1, 0),
                Qt.Key.Key_Right: ( 1, 0),
                Qt.Key.Key_Up:    ( 0,-1),
                Qt.Key.Key_Down:  ( 0, 1),
            }
            if event.key() in arrow_map:
                layer = self.layer_stack.active
                if self._warn_locked(layer):
                    event.accept()
                    return
                step =10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
                ddx, ddy = arrow_map[event.key()]
                if layer and layer.is_group:
                    # 中に個別にロックしたレイヤーがあれば、それだけは動かさない。
                    children = [c for c in self._collect_leaf_layers(layer)
                                if not self.is_locked(c)]
                    if children:
                        for child in children:
                            ox = getattr(child, 'offset_x', 0)
                            oy = getattr(child, 'offset_y', 0)
                            if child.is_vector:
                                self._history.append(
                                    ("vector", child.uid, child.copy_strokes(), ox, oy))  # type: ignore
                            else:
                                self._history.append(("pixel", child.uid, child.image.copy(), ox, oy))  # type: ignore
                        self._redo_stack.clear()
                        self._trim_history()
                        self.edited.emit()
                        for child in children:
                            if child.is_vector:
                                child.translate_strokes(ddx * step, ddy * step)  # type: ignore
                            else:
                                child.offset_x += ddx * step  # type: ignore
                                child.offset_y += ddy * step  # type: ignore
                elif layer and not layer.is_group:
                    self._save_history()
                    if layer.is_vector:
                        # 点がキャンバス座標なので、offset を動かしても
                        # 再レンダーで元の位置に戻ってしまう。点そのものを動かす。
                        layer.translate_strokes(ddx * step, ddy * step)  # type: ignore
                    else:
                        layer.offset_x += ddx * step  # type: ignore
                        layer.offset_y += ddy * step  # type: ignore
                    self.update()
                event.accept()
                return

        # Alt 一時スポイト（押しっぱなし）
        if event.key() == Qt.Key.Key_Alt and not event.isAutoRepeat() and not self._alt_eyedropper:
            self._alt_eyedropper = True
            self._pre_alt_tool = self.tool
            self.tool = Tool.EYEDROPPER
            self.setCursor(Qt.CursorShape.CrossCursor)
            event.accept()
            return

        # ツールショートカットキー（修飾キーなし・オートリピートなし）
        if (event.key() in _TOOL_KEY_MAP
                and not event.isAutoRepeat()
                and not event.modifiers()):
            self.tool_shortcut_pressed.emit(_TOOL_KEY_MAP[event.key()])
            event.accept()
            return

        # X キー: 描画色と直前の色をスワップ
        if event.key() == Qt.Key.Key_X and not event.isAutoRepeat():
            self.pen_color, self._prev_color = self._prev_color, self.pen_color
            self.color_picked.emit(self.pen_color)
            event.accept()
            return

        # 数字キー 1〜9, 0 でアクティブレイヤーの不透明度変更
        # （移動ツールの矢印キーと競合しないよう Tool.MOVE 以外で有効）
        num_keys = {
            Qt.Key.Key_1: 10, Qt.Key.Key_2: 28, Qt.Key.Key_3: 51,
            Qt.Key.Key_4: 76, Qt.Key.Key_5: 128,
            Qt.Key.Key_6: 153, Qt.Key.Key_7: 178, Qt.Key.Key_8: 204,
            Qt.Key.Key_9: 230, Qt.Key.Key_0: 255,
        }
        if event.key() in num_keys and not event.modifiers():
            layer = self.layer_stack.active
            if layer:
                layer.opacity = num_keys[event.key()]
                self.layer_opacity_changed.emit(layer.opacity)
                self.update()
            event.accept()
            return

        # ブラシサイズ [ / ] キー
        if event.key() == Qt.Key.Key_BracketLeft:
            self.pen_size = max(1, self.pen_size - 1)
            self.brush_size_changed.emit(self.pen_size)
            self.update()
            event.accept()
            return
        if event.key() == Qt.Key.Key_BracketRight:
            self.pen_size = min(200, self.pen_size + 1)
            self.brush_size_changed.emit(self.pen_size)
            self.update()
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Alt and self._alt_eyedropper and not event.isAutoRepeat():
            self._alt_eyedropper = False
            self.tool = self._pre_alt_tool
            self._drawing = False
            self._last_pos = None
            self._restore_tool_cursor()
            event.accept()
            return

        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._panning = False
            self._pan_start_widget = None
            self._restore_tool_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def leaveEvent(self, event):
        self._cursor_widget_pos = None
        self.update()
        super().leaveEvent(event)
