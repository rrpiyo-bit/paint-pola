from __future__ import annotations
import math
import numpy as np
import cv2

from PyQt6.QtWidgets import QWidget, QApplication, QInputDialog, QFontDialog, QColorDialog
from PyQt6.QtGui import (QPainter, QColor, QPen, QImage, QFont, QPixmap,
                          QTransform, QBrush, QPainterPath)
from PyQt6.QtCore import Qt, QPoint, QPointF, QRect, QRectF, QSize, pyqtSignal, QTimer

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
HISTORY_LIMIT = 50
MIN_ZOOM = 0.05
MIN_TRANSFORM_SIZE = 1
HANDLE_HIT_RADIUS = 12
GRID_COLOR = QColor(180, 180, 180, 160)
SELECTION_COLOR = QColor(0, 120, 215)


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


# ── 塗りつぶし ──────────────────────────────────────────────────────────────────

def _alpha(pixel: int) -> int:
    return (pixel >> 24) & 0xFF


def _is_line_pixel(pixel: int, threshold: int = 10) -> bool:
    """参照レイヤーのピクセルが「線」（塗りつぶしを堰き止める境界）かどうか判定する。
    不透明なピクセルはすべて境界（白い線も含む）。"""
    return _alpha(pixel) > threshold


def _flood_fill(image: QImage, x: int, y: int, fill_color: QColor,
                ref_image: QImage | None = None):
    """連結領域を numpy/cv2 のラベリングで検出し、一括書き込みする塗りつぶし。
    QImage.pixel()/setPixel() を1ピクセルずつ呼ぶ旧scanline実装は、大キャンバスで
    UIスレッドが長時間ブロックされフリーズ/クラッシュする原因になっていたため廃止。"""
    w, h = image.width(), image.height()
    if not (0 <= x < w and 0 <= y < h):
        return

    judge = ref_image if ref_image is not None else image
    fill = fill_color.rgba()

    nbytes = h * w * 4
    judge_ptr = judge.bits(); judge_ptr.setsize(nbytes)
    judge_arr = np.frombuffer(judge_ptr, dtype=np.uint8).reshape(h, w, 4)

    # 参照モード: judge の不透明ピクセルが境界、image の未塗りピクセルが対象
    # 通常モード: image の同色ピクセルが対象
    if ref_image is not None:
        if _is_line_pixel(judge.pixel(x, y)):
            return
        candidate = (judge_arr[:, :, 3] <= 10).astype(np.uint8)
    else:
        target = judge.pixel(x, y)
        if target == fill:
            return
        img_ptr = image.bits(); img_ptr.setsize(nbytes)
        img_arr_ro = np.frombuffer(img_ptr, dtype=np.uint8).reshape(h, w, 4)
        target_color = QColor.fromRgba(target)
        target_bgra = np.array([target_color.blue(), target_color.green(),
                                 target_color.red(), target_color.alpha()], dtype=np.uint8)
        candidate = np.all(img_arr_ro == target_bgra, axis=2).astype(np.uint8)

    num, labels = cv2.connectedComponents(candidate, connectivity=4)
    seed_label = labels[y, x]
    if seed_label == 0:
        return
    fill_mask = labels == seed_label

    if ref_image is not None:
        img_ptr = image.bits(); img_ptr.setsize(nbytes)
        img_arr = np.frombuffer(img_ptr, dtype=np.uint8).reshape(h, w, 4)
        # 参照モードでは「未塗り(候補)」かつ「既に fill 色ではない」ピクセルのみ書き換える
        fill_color_bgra = np.array([fill_color.blue(), fill_color.green(),
                                     fill_color.red(), fill_color.alpha()], dtype=np.uint8)
        already_filled = np.all(img_arr == fill_color_bgra, axis=2)
        fill_mask = fill_mask & ~already_filled
    else:
        img_arr = img_arr_ro

    img_arr[fill_mask] = (fill_color.blue(), fill_color.green(),
                           fill_color.red(), fill_color.alpha())


def _flood_fill_expanded(image: QImage, x: int, y: int,
                          fill_color: QColor, ref_image: QImage | None,
                          expand: int):
    """flood fill 後に expand px だけ塗り範囲を膨張(正)/収縮(負)させる。"""
    if expand == 0:
        _flood_fill(image, x, y, fill_color, ref_image)
        return

    # fill 前のスナップショット
    before = image.copy()
    _flood_fill(image, x, y, fill_color, ref_image)

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
                                  fill_color: QColor, ref_image: QImage | None) -> int:
    """area_mask（投げなわ選択範囲）内にある、線で閉じた領域だけを自動検出して塗りつぶす。
    area_mask の外周に接している領域（＝閉じていない/範囲外に開いている）は対象外にする。
    戻り値: 実際に塗りつぶした領域の数。"""
    w, h = image.width(), image.height()
    judge = ref_image if ref_image is not None else image

    nbytes = h * w * 4
    ptr = judge.bits(); ptr.setsize(nbytes)
    judge_arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
    line_mask = (judge_arr[:, :, 3] > 10).astype(np.uint8)  # 不透明=線(境界)

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

    nbytes = h * w * 4
    ptr = image.bits(); ptr.setsize(nbytes)
    img_arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
    img_arr[fill_mask] = (fill_color.blue(), fill_color.green(),
                           fill_color.red(), fill_color.alpha())

    return len(closed_labels)


# ── ポリゴンマスク ─────────────────────────────────────────────────────────────

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
    zoom_changed = pyqtSignal(float)
    layer_opacity_changed = pyqtSignal(int)  # 数字キーで不透明度が変わったとき
    tool_shortcut_pressed = pyqtSignal(object)  # Tool — キーボードショートカットでツール切替

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
        self._stabilizer = StabilizedBrush(get_brush(BrushType.ROUND), smooth=6)

        # 対称定規
        self.symmetry_enabled: bool = False

        # 図形塗りモード: "none"=枠線のみ / "fill"=塗りのみ / "both"=枠線＋塗り
        self.shape_fill: str = "none"
        self.fill_expand: int = 0   # バケツ塗り拡張(正)/縮小(負) px
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
        self._lasso_points: list[QPoint] = []
        self._lasso_mask: QImage | None = None
        self._lasso_path_points: list[QPoint] = []  # 確定後の投げ縄パス（表示用）

        # パスピックモード（アクション用: クリックでパスの点を打つ）
        self._path_pick_active: bool = False
        self._path_pick_points: list[QPoint] = []
        self._path_pick_callback = None  # confirmed_points を受け取るコールバック

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
        self._move_base_pos: QPoint | None = None
        # グループ移動用：子レイヤー全員の元画像リスト
        self._move_group_bases: list[tuple[Layer, QImage, int, int]] | None = None

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
        self.setMinimumSize(QSize(w, h))
        self.resize(w, h)
        self.update()

    def set_brush(self, brush_type: str):
        self.brush_type = brush_type
        self._stabilizer = StabilizedBrush(get_brush(brush_type), smooth=6)

    def set_zoom(self, zoom: float):
        self.zoom = max(MIN_ZOOM, zoom)
        self._update_size()
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

    def toggle_grid(self):
        self._show_grid = not self._show_grid
        self.update()

    def set_grid_size(self, size: int):
        self._grid_size = max(1, size)  # 0除算・無限ループ防止
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
        return QPoint(int(mapped.x()), int(mapped.y()))

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
        return id(layer) if layer and not layer.is_group else None

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

    def _save_history(self):
        lid = self._layer_id()
        layer = self.layer_stack.active
        if lid is None or layer is None:
            return
        ox = getattr(layer, 'offset_x', 0)
        oy = getattr(layer, 'offset_y', 0)
        self._history.append(("pixel", lid, layer.image.copy(), ox, oy))  # type: ignore
        self._redo_stack.clear()
        if len(self._history) > HISTORY_LIMIT:
            self._history.pop(0)

    def _snapshot_layer(self, lyr) -> dict:
        if lyr.is_group:
            return {
                "type": "group", "name": lyr.name, "visible": lyr.visible,
                "opacity": lyr.opacity, "clipping": lyr.clipping,
                "reference": lyr.reference, "collapsed": lyr.collapsed,
                "children": [self._snapshot_layer(c) for c in lyr.children],
                "_w": lyr._w, "_h": lyr._h,
            }
        return {
            "type": "layer", "name": lyr.name, "visible": lyr.visible,
            "opacity": lyr.opacity, "clipping": lyr.clipping,
            "reference": lyr.reference, "image": lyr.image.copy(),
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
            g.collapsed = snap["collapsed"]
            g.children = [self._restore_layer(c) for c in snap["children"]]
            return g
        lyr = Layer(snap["name"], snap["image"].width(), snap["image"].height())
        lyr.image = snap["image"].copy()
        for k in ("visible", "opacity", "clipping", "reference", "blend_mode",
                  "offset_x", "offset_y",
                  "border_enabled", "border_size", "border_color",
                  "shadow_enabled", "shadow_color", "shadow_offset_x", "shadow_offset_y",
                  "shadow_blur", "shadow_strength", "glow_enabled", "glow_color",
                  "glow_size", "glow_strength", "blur_enabled", "blur_radius",
                  "blur_strength", "hsl_enabled", "hsl_hue", "hsl_saturation", "hsl_lightness"):
            setattr(lyr, k, snap[k])
        return lyr

    def save_structure_history(self):
        ls = self.layer_stack
        snap = {
            "layers": [self._snapshot_layer(l) for l in ls.layers],
            "active_path": list(ls.active_path),
        }
        self._history.append(("structure", snap))
        self._redo_stack.clear()
        if len(self._history) > HISTORY_LIMIT:
            self._history.pop(0)

    def _apply_structure_snapshot(self, snap: dict):
        ls = self.layer_stack
        ls.layers = [self._restore_layer(s) for s in snap["layers"]]
        path = list(snap.get("active_path") or [snap.get("active_index", 0)])
        if path:
            path[0] = min(path[0], max(0, len(ls.layers) - 1))
        ls.active_path = path

    def purge_layer_history(self, layer_id: int):
        self._history = [e for e in self._history if not (e[0] == "pixel" and e[1] == layer_id)]
        self._redo_stack = [e for e in self._redo_stack if not (e[0] == "pixel" and e[1] == layer_id)]

    def _all_layer_ids(self) -> set[int]:
        ids: set[int] = set()
        def _collect(items):
            for item in items:
                ids.add(id(item))
                if item.is_group:
                    _collect(item.children)
        _collect(self.layer_stack.layers)
        return ids

    def purge_orphan_history(self):
        live = self._all_layer_ids()
        self._history = [e for e in self._history if e[0] == "structure" or (e[0] == "pixel" and e[1] in live)]
        self._redo_stack = [e for e in self._redo_stack if e[0] == "structure" or (e[0] == "pixel" and e[1] in live)]

    def _find_layer_by_id(self, layer_id: int) -> Layer | None:
        def _search(items):
            for item in items:
                if id(item) == layer_id:
                    return item
                if item.is_group:
                    found = _search(item.children)
                    if found is not None:
                        return found
            return None
        return _search(self.layer_stack.layers)

    def undo(self):
        if self._transform_image:
            self.cancel_transform()
            return
        if not self._history:
            return
        entry = self._history.pop()
        if entry[0] == "pixel":
            lid = entry[1]
            img = entry[2]
            old_ox = entry[3] if len(entry) > 3 else 0
            old_oy = entry[4] if len(entry) > 4 else 0
            layer = self._find_layer_by_id(lid)
            if layer is None:
                return
            cur_ox = getattr(layer, 'offset_x', 0)
            cur_oy = getattr(layer, 'offset_y', 0)
            self._redo_stack.append(("pixel", lid, layer.image.copy(), cur_ox, cur_oy))
            layer.image = img
            layer.offset_x = old_ox
            layer.offset_y = old_oy
        elif entry[0] == "structure":
            _, snap = entry
            current_snap = {
                "layers": [self._snapshot_layer(l) for l in self.layer_stack.layers],
                "active_path": list(self.layer_stack.active_path),
            }
            self._redo_stack.append(("structure", current_snap))
            self._apply_structure_snapshot(snap)
            if self._on_structure_restored:
                self._on_structure_restored()
        self.update()

    def redo(self):
        if not self._redo_stack:
            return
        entry = self._redo_stack.pop()
        if entry[0] == "pixel":
            lid = entry[1]
            img = entry[2]
            old_ox = entry[3] if len(entry) > 3 else 0
            old_oy = entry[4] if len(entry) > 4 else 0
            layer = self._find_layer_by_id(lid)
            if layer is None:
                return
            cur_ox = getattr(layer, 'offset_x', 0)
            cur_oy = getattr(layer, 'offset_y', 0)
            self._history.append(("pixel", lid, layer.image.copy(), cur_ox, cur_oy))
            layer.image = img
            layer.offset_x = old_ox
            layer.offset_y = old_oy
        elif entry[0] == "structure":
            _, snap = entry
            current_snap = {
                "layers": [self._snapshot_layer(l) for l in self.layer_stack.layers],
                "active_path": list(self.layer_stack.active_path),
            }
            self._history.append(("structure", current_snap))
            self._apply_structure_snapshot(snap)
            if self._on_structure_restored:
                self._on_structure_restored()
        self.update()

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

        if self._preview_start and self._preview_end:
            self._draw_shape_preview(p)

        if self._lasso_points:
            lasso_path = QPainterPath()
            lasso_path.moveTo(QPointF(self._lasso_points[0]))
            for pt in self._lasso_points[1:]:
                lasso_path.lineTo(QPointF(pt))
            self._draw_marching_ants(p, path=lasso_path)
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

        # ブラシカーソル円
        if self._cursor_widget_pos and self.tool in (Tool.PEN, Tool.ERASER, Tool.BLUR):
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
        for x in range(0, cw + 1, g):
            p.drawLine(x, 0, x, ch)
        for y in range(0, ch + 1, g):
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
        if layer is not None and not layer.is_group:
            # 元画像をバックアップ
            orig_img = layer.image  # type: ignore
            # 消去済みコピーを作成
            erased = QImage(orig_img.width(), orig_img.height(),
                            QImage.Format.Format_ARGB32_Premultiplied)
            erased.fill(Qt.GlobalColor.transparent)
            ep = QPainter(erased)
            ep.drawImage(0, 0, orig_img)
            if self._transform_erase_mask:
                ep.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
                ep.drawImage(0, 0, self._transform_erase_mask)
            elif self._transform_erase_rect:
                ep.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
                ep.fillRect(self._transform_erase_rect, Qt.GlobalColor.transparent)
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

    def _draw_floating_image(self, p: QPainter):
        """キャンバス座標系のPainter（setTransform済み）にフローティング画像を描く。"""
        if not self._transform_rect or not self._transform_image:
            return
        if self._mesh_grid:
            result = self._warp_mesh_image()
            if result:
                warped_img, ox, oy = result
                p.save()
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                p.drawImage(ox, oy, warped_img)
                p.restore()
            return
        if self._perspective_corners:
            result = self._warp_perspective_image()
            if result:
                warped_img, ox, oy = result
                p.save()
                p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                p.drawImage(ox, oy, warped_img)
                p.restore()
            return
        r = self._transform_rect
        img = self._transform_image
        src_rect = QRectF(0, 0, img.width(), img.height())
        p.save()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if self._transform_angle != 0.0:
            pv = self._pivot_point()
            p.translate(pv.x(), pv.y())
            p.rotate(self._transform_angle)
            p.translate(-pv.x(), -pv.y())
        p.drawImage(r, img, src_rect)
        p.restore()

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
            if layer and layer.is_group:
                children = self._collect_leaf_layers(layer)
                if children:
                    for child in children:
                        ox = getattr(child, 'offset_x', 0)
                        oy = getattr(child, 'offset_y', 0)
                        self._history.append(("pixel", id(child), child.image.copy(), ox, oy))  # type: ignore
                    self._redo_stack.clear()
                    if len(self._history) > HISTORY_LIMIT:
                        self._history = self._history[-HISTORY_LIMIT:]
                    self._move_group_bases = [
                        (c, c.image.copy(), getattr(c, 'offset_x', 0), getattr(c, 'offset_y', 0))
                        for c in children]  # type: ignore
                    self._move_base_pos = cp
                    self._drawing = True
            elif layer and not layer.is_group:
                self._save_history()
                self._move_base_image = layer.image  # type: ignore
                self._move_base_offset = (layer.offset_x, layer.offset_y)  # type: ignore
                self._move_base_pos = cp
                self._last_pos = cp
                self._drawing = True
            return

        if not layer or layer.is_group:
            if layer and layer.is_group and self.tool in (Tool.PEN, Tool.ERASER, Tool.FILL, Tool.BLUR, Tool.LASSO_FILL):
                self.status_message.emit("グループレイヤーには描画できません。子レイヤーを選択してください。")
            return

        self._drawing = True

        lox = getattr(layer, 'offset_x', 0)
        loy = getattr(layer, 'offset_y', 0)
        lp = QPoint(cp.x() - lox, cp.y() - loy)

        if self.tool == Tool.PEN:
            self._save_history()
            self._begin_stroke_cache(layer)
            self._stabilizer.reset()
            smooth_pt = self._stabilizer.push(lp).toPoint()
            self._last_pos = smooth_pt
            self._brush_stamp(layer.image, smooth_pt)  # type: ignore
            self.update()

        elif self.tool == Tool.ERASER:
            self._save_history()
            self._begin_stroke_cache(layer)
            self._stabilizer.reset()
            self._last_pos = lp
            self._erase_point(layer.image, lp)  # type: ignore
            self.update()

        elif self.tool == Tool.FILL:
            self._save_history()
            ref_img = self._build_fill_reference(layer)
            _flood_fill_expanded(layer.image, lp.x(), lp.y(), self.pen_color, ref_img, self.fill_expand)  # type: ignore
            self.update()

        elif self.tool == Tool.BLUR:
            self._save_history()
            self._begin_stroke_cache(layer)
            self._last_pos = lp
            self._blur_brush.strength = self.blur_strength
            self._blur_brush.stamp(layer.image, lp, self.pen_color, self.blur_size)
            self.update()

        elif self.tool == Tool.EYEDROPPER:
            composite = self.layer_stack.composite()
            if 0 <= cp.x() < composite.width() and 0 <= cp.y() < composite.height():
                self._prev_color = QColor(self.pen_color)
                self.color_picked.emit(QColor(composite.pixel(cp.x(), cp.y())))
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
                    self._transform_handle = handle
                    self._transform_drag_start = wp
                    self._transform_rect_start = QRectF(self._transform_rect)
                    self._transform_angle_start = self._transform_angle
                    self._drawing = True
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
                        if handle:
                            self._transform_handle = handle
                            self._transform_drag_start = wp
                            self._transform_rect_start = QRectF(self._transform_rect)
                            self._transform_angle_start = self._transform_angle
                            self._drawing = True
                        else:
                            self._transform_handle = 'move'
                            self._transform_drag_start = wp
                            self._transform_rect_start = QRectF(self._transform_rect)
                            self._transform_angle_start = self._transform_angle
                            self._drawing = True
                    else:
                        # 選択外クリック → 変形中なら確定してから新規選択
                        if self._transform_image:
                            self._commit_transform()
                        self._selection_rect = None
                        self._lasso_mask = None
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
                        self._preview_start = cp
                        self._preview_end = cp
            else:
                # 新規選択開始
                self._selection_rect = None
                self._lasso_mask = None
                self._preview_start = cp
                self._preview_end = cp

        elif self.tool == Tool.LASSO_FILL:
            # 常に新規に投げなわを描く専用ツール（選択の持ち上げ・変形は行わない）
            self._selection_rect = None
            self._lasso_mask = None
            self._lasso_path_points = []
            self._lasso_points = [cp]

        elif self.tool == Tool.LASSO:
            if self._transform_image:
                handle = self._hit_transform_handle(wp)
                if handle:
                    self._transform_handle = handle
                    self._transform_drag_start = wp
                    self._transform_rect_start = QRectF(self._transform_rect)
                    self._transform_angle_start = self._transform_angle
                    self._drawing = True
                else:
                    self._commit_transform()
                    self._lasso_points = [cp]
                    self._lasso_path_points = []
            elif self._selection_rect and layer and not layer.is_group:
                if self.select_mode == "transform":
                    if self._selection_rect.contains(cp):
                        self._lift_selection(layer)  # type: ignore
                        self._transform_handle = "move"
                        self._transform_drag_start = wp
                        self._transform_rect_start = QRectF(self._transform_rect)
                        self._transform_angle_start = self._transform_angle
                        self._drawing = True
                    else:
                        self._selection_rect = None
                        self._lasso_mask = None
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
                        self._lasso_path_points = []
                        self._lasso_points = [cp]
            else:
                self._selection_rect = None
                self._lasso_mask = None
                self._lasso_path_points = []
                self._lasso_points = [cp]

        elif self.tool == Tool.TEXT:
            # クリック位置を確定してからダイアログを表示する
            self._text_pos = cp
            self._drawing = False
            self._ask_text()

    def _handle_transform_press(self, wp: QPointF, cp: QPoint,
                                 layer):
        if self._transform_image and self._transform_rect:
            handle = self._hit_transform_handle(wp)
            if handle:
                self._transform_handle = handle
                self._transform_drag_start = wp
                self._transform_rect_start = QRectF(self._transform_rect)
                self._transform_angle_start = self._transform_angle
                if self._mesh_grid:
                    self._mesh_grid_start = [[QPointF(p) for p in row] for row in self._mesh_grid]
                elif self._perspective_corners:
                    self._perspective_corners_start = [QPointF(c) for c in self._perspective_corners]
                self._drawing = True
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

        if self.tool in (Tool.PEN, Tool.ERASER):
            self._cursor_widget_pos = wp
            self.update()

        # 選択範囲内クリック後、実際にドラッグが始まったら lift を実行
        if self._lift_pending and self._drawing:
            layer = self.layer_stack.active
            if layer and not layer.is_group:
                self._lift_selection(layer)  # type: ignore
                self._lift_pending = False
                self._lift_pending_wp = None
                self._transform_handle = 'move'
                self._transform_drag_start = wp
                self._transform_rect_start = QRectF(self._transform_rect)
                self._transform_angle_start = self._transform_angle

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
            if self._move_group_bases is not None:
                for child, base_img, base_ox, base_oy in self._move_group_bases:
                    child.offset_x = base_ox + dx
                    child.offset_y = base_oy + dy
            elif self._move_base_image is not None and not layer.is_group:
                self._move_layer(layer, dx, dy)  # type: ignore
            self.update()
            return

        if layer.is_group:
            return

        lox = getattr(layer, 'offset_x', 0)
        loy = getattr(layer, 'offset_y', 0)
        lp = QPoint(cp.x() - lox, cp.y() - loy)

        if self.tool == Tool.PEN and self._last_pos:
            smooth_pt = self._stabilizer.push(lp).toPoint()
            self._brush_stroke(layer.image, self._last_pos, smooth_pt)  # type: ignore
            self._last_pos = smooth_pt
            self.update()

        elif self.tool == Tool.ERASER and self._last_pos:
            self._erase_line(layer.image, self._last_pos, lp)  # type: ignore
            self._last_pos = lp
            self.update()

        elif self.tool == Tool.BLUR and self._last_pos:
            self._blur_brush.stroke_to(layer.image, self._last_pos, lp, self.pen_color, self.blur_size)
            self._last_pos = lp
            self.update()

        elif self.tool in (Tool.LINE, Tool.RECT, Tool.ELLIPSE, Tool.SELECT_RECT):
            self._preview_end = cp
            self.update()

        elif self.tool in (Tool.LASSO, Tool.LASSO_FILL):
            self._lasso_points.append(cp)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
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
        self._move_group_bases = None
        self._move_base_pos = None

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

        if self.tool in (Tool.LINE, Tool.RECT, Tool.ELLIPSE) and self._preview_start:
            # ドラッグせずクリックしただけ（始点=終点）の場合は退化図形なので何も描かない。
            # QRect(a, a) は幅・高さ 1 になるため、点の一致で判定する。
            is_zero = (self._preview_start == cp)
            if is_zero:
                # 退化図形は描かない。押下時に積んだ履歴を巻き戻す
                lid = self._layer_id()
                if lid is not None and self._history and self._history[-1][0] == "pixel" and self._history[-1][1] == lid:
                    layer.image = self._history.pop()[2]  # type: ignore
            else:
                lox = getattr(layer, 'offset_x', 0)
                loy = getattr(layer, 'offset_y', 0)
                sa = QPoint(self._preview_start.x() - lox, self._preview_start.y() - loy)
                sb = QPoint(cp.x() - lox, cp.y() - loy)
                self._commit_shape(layer.image, sa, sb)  # type: ignore
            self._preview_start = None
            self._preview_end = None
            self.update()

        elif self.tool == Tool.SELECT_RECT and self._preview_start:
            sel = QRect(self._preview_start, cp).normalized()
            self._preview_start = None
            self._preview_end = None
            if sel.width() > 1 and sel.height() > 1:
                self._selection_rect = sel
                self._lasso_mask = None
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
            else:
                self._selection_rect = None
                self._lasso_mask = None
                self._lasso_path_points = []
            self._lasso_points = []
            self._sync_ant_timer()
            self.update()

        self._last_pos = None

    def _build_fill_reference(self, layer) -> QImage | None:
        """塗りつぶしの境界判定に使う参照画像（自分以外の参照レイヤーの合成）を作る。"""
        refs = [r for r in self.layer_stack.references if r is not layer]
        if len(refs) == 0:
            return None
        if len(refs) == 1:
            return refs[0].image
        w, h = self.layer_stack.width, self.layer_stack.height
        ref_img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        ref_img.fill(Qt.GlobalColor.transparent)
        rp = QPainter(ref_img)
        for r in reversed(refs):
            rp.setOpacity(r.opacity / 255)
            rp.drawImage(0, 0, r.image)
        rp.end()
        return ref_img.convertToFormat(QImage.Format.Format_ARGB32)

    def _apply_lasso_fill(self, layer, lasso_points: list[QPoint]) -> None:
        """投げなわで囲んだ範囲内にある、線で閉じた領域だけを自動で塗りつぶす。"""
        if not layer or layer.is_group:
            return
        w, h = self.layer_stack.width, self.layer_stack.height
        mask_img = _mask_from_polygon(lasso_points, w, h)
        nbytes = h * w * 4
        ptr = mask_img.bits(); ptr.setsize(nbytes)
        mask_arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
        area_mask = (mask_arr[:, :, 3] > 0).astype(np.uint8)
        if not area_mask.any():
            return

        self._save_history()
        ref_img = self._build_fill_reference(layer)
        # 通常の塗りつぶしツール(Tool.FILL)と同じく、参照画像・レイヤー画像とも
        # キャンバス座標系のまま扱う（レイヤーオフセットによる座標変換は行わない）
        filled = _fill_closed_regions_in_area(layer.image, area_mask, self.pen_color, ref_img)  # type: ignore
        if filled == 0 and self._history and self._history[-1][0] == "pixel":
            # 何も塗られなかった場合は空の undo エントリを積まない
            self._history.pop()
        self.update()

    # ── drawing helpers ──────────────────────────────────────────────────────

    def _make_pen(self, color: QColor, size: int) -> QPen:
        return QPen(color, size, Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)

    def _mirror_x(self, pt: QPoint) -> QPoint:
        """対称定規用: キャンバス中心でX軸ミラーした点を返す。"""
        cx = self.layer_stack.width // 2
        return QPoint(2 * cx - pt.x(), pt.y())

    def _brush_stamp(self, img: QImage, pt: QPoint):
        """ブラシで1点描画（対称定規対応）。"""
        brush = get_brush(self.brush_type)
        brush.stamp(img, pt, self.pen_color, self.pen_size)
        if self.symmetry_enabled:
            brush.stamp(img, self._mirror_x(pt), self.pen_color, self.pen_size)

    def _brush_stroke(self, img: QImage, a: QPoint, b: QPoint):
        """ブラシでストローク描画（対称定規対応）。"""
        brush = get_brush(self.brush_type)
        brush.stroke_to(img, a, b, self.pen_color, self.pen_size)
        if self.symmetry_enabled:
            brush.stroke_to(img, self._mirror_x(a), self._mirror_x(b),
                            self.pen_color, self.pen_size)

    def _draw_point(self, img: QImage, p: QPoint):
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(self._make_pen(self.pen_color, self.pen_size))
        painter.drawPoint(p)
        painter.end()

    def _draw_line(self, img: QImage, a: QPoint, b: QPoint):
        painter = QPainter(img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(self._make_pen(self.pen_color, self.pen_size))
        painter.drawLine(a, b)
        painter.end()

    def _erase_point(self, img: QImage, p: QPoint):
        painter = QPainter(img)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.setPen(QPen(Qt.GlobalColor.transparent, self.eraser_size,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPoint(p)
        painter.end()

    def _erase_line(self, img: QImage, a: QPoint, b: QPoint):
        painter = QPainter(img)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.setPen(QPen(Qt.GlobalColor.transparent, self.eraser_size,
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap,
                            Qt.PenJoinStyle.RoundJoin))
        painter.drawLine(a, b)
        painter.end()

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
        layer.offset_x = self._move_base_offset[0] + dx
        layer.offset_y = self._move_base_offset[1] + dy

    # ── text ─────────────────────────────────────────────────────────────────

    def _ask_text(self):
        """クリック後に呼ばれる。_text_pos が確定している前提。"""
        text, ok = QInputDialog.getText(self, "テキスト入力", "テキスト:")
        if not (ok and text):
            self._text_pos = None
            return
        font, ok2 = QFontDialog.getFont(QFont("Arial", 40), self)
        if not ok2:
            self._text_pos = None
            return
        color = QColorDialog.getColor(self.pen_color, self)
        if color.isValid():
            self.draw_text(text, font, color)
        else:
            self._text_pos = None

    def draw_text(self, text: str, font: QFont, color: QColor):
        layer = self.layer_stack.active
        if not layer or layer.is_group or not self._text_pos:
            return
        self._save_history()
        painter = QPainter(layer.image)  # type: ignore
        painter.setFont(font)
        painter.setPen(QPen(color))
        painter.drawText(self._text_pos, text)
        painter.end()
        self._text_pos = None
        self.update()

    # ── selection ────────────────────────────────────────────────────────────

    def copy_selection(self):
        layer = self.layer_stack.active
        if not layer or layer.is_group or not self._selection_rect:
            return
        src: QImage = layer.image  # type: ignore
        region = src.copy(self._selection_rect)
        if self._lasso_mask:
            mask_crop = self._lasso_mask.copy(self._selection_rect)
            p = QPainter(region)
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            p.drawImage(0, 0, mask_crop)
            p.end()
        self._clipboard_image = region
        self._clipboard_offset = self._selection_rect.topLeft()

    def paste_selection(self):
        layer = self.layer_stack.active
        if not layer or layer.is_group or not self._clipboard_image:
            return
        self._save_history()
        img: QImage = layer.image  # type: ignore
        w, h = img.width(), img.height()
        overlay = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        overlay.fill(Qt.GlobalColor.transparent)
        op = QPainter(overlay)
        op.drawImage(self._clipboard_offset, self._clipboard_image)
        op.end()
        p = QPainter(img)
        p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        p.drawImage(0, 0, overlay)
        p.end()
        self._selection_rect = None
        self._lasso_mask = None
        self.update()

    def delete_selection(self):
        layer = self.layer_stack.active
        if not layer or layer.is_group or not self._selection_rect:
            return
        self._save_history()
        painter = QPainter(layer.image)  # type: ignore
        if self._lasso_mask:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationOut)
            painter.drawImage(0, 0, self._lasso_mask)
        else:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(self._selection_rect, Qt.GlobalColor.transparent)
        painter.end()
        self._selection_rect = None
        self._lasso_mask = None
        self.update()

    def select_all(self):
        self._selection_rect = QRect(0, 0, self.layer_stack.width, self.layer_stack.height)
        self._lasso_mask = None
        self._sync_ant_timer()
        self.update()

    def deselect(self):
        if self._transform_image:
            self.cancel_transform()
        self._selection_rect = None
        self._lasso_mask = None
        self._lasso_path_points = []
        self._lasso_points = []
        self._sync_ant_timer()
        self.update()

    # ── transform ────────────────────────────────────────────────────────────

    def _lift_selection(self, layer: Layer):
        """選択範囲をフローティング化する。ピクセル消去は確定時（_commit_transform）に行う。"""
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
        self.update()

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

        if self._mesh_grid and self._mesh_grid_start:
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

        if self._perspective_corners and self._perspective_corners_start:
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

    def _commit_transform(self):
        layer = self._transform_layer or self.layer_stack.active
        if not layer or layer.is_group or not self._transform_image or not self._transform_rect:
            return

        self._save_history()

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

        if self._mesh_grid:
            result = self._warp_mesh_image()
            if result:
                warped_img, wx, wy = result
                painter = QPainter(img)
                painter.drawImage(wx - ox, wy - oy, warped_img)
                painter.end()
        elif self._perspective_corners:
            result = self._warp_perspective_image()
            if result:
                warped_img, wx, wy = result
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

        self._transform_image = None
        self._transform_rect = None
        self._transform_orig_rect = None
        self._transform_angle = 0.0
        self._transform_layer = None
        self._transform_erase_rect = None
        self._transform_erase_mask = None
        self._perspective_corners = None
        self._perspective_corners_start = None
        self._perspective_drag_idx = -1
        self._mesh_grid = None
        self._mesh_grid_start = None
        self._mesh_drag_idx = (-1, -1)
        self.update()

    def set_transform_mode(self, mode: str):
        """"standard" / "perspective" / "mesh" を切り替える。"""
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

    def set_perspective_mode(self, enabled: bool):
        self.set_transform_mode("perspective" if enabled else "standard")

    def lift_whole_layer(self) -> bool:
        """アクティブレイヤー全体をフローティング化して変形モードに入る。選択範囲は使わない。"""
        layer = self.layer_stack.active
        if not layer or layer.is_group:
            return False
        self.save_structure_history()
        self._selection_rect = None
        self._lasso_mask = None
        self._lift_selection(layer)  # type: ignore
        return True

    def apply_transform_percentage(self, scale_x_pct: float, scale_y_pct: float, angle_deg: float):
        """フローティング変形中に拡縮率(%)と回転角を適用してリアルタイムプレビューを更新する。
        scale_x_pct / scale_y_pct は元サイズを100%として指定する。
        _transform_rect の中心を固定しながらサイズを変える。"""
        if not self._transform_image or not self._transform_orig_rect:
            return
        orig = self._transform_orig_rect
        new_w = orig.width()  * scale_x_pct / 100.0
        new_h = orig.height() * scale_y_pct / 100.0
        cx = orig.center().x()
        cy = orig.center().y()
        self._transform_rect = QRectF(cx - new_w / 2, cy - new_h / 2, new_w, new_h)
        self._transform_angle = angle_deg
        self.update()

    def cancel_transform(self):
        """変形をキャンセル。lift時にはピクセル消去しないので単純破棄でOK。"""
        if not self._transform_image:
            return
        self._transform_image = None
        self._transform_rect = None
        self._transform_orig_rect = None
        self._transform_angle = 0.0
        self._transform_layer = None
        self._transform_erase_rect = None
        self._transform_erase_mask = None
        self._perspective_corners = None
        self._perspective_corners_start = None
        self._perspective_drag_idx = -1
        self._mesh_grid = None
        self._mesh_grid_start = None
        self._mesh_drag_idx = (-1, -1)
        self.update()

    def reset_state(self):
        """新規/開くなどでキャンバスを差し替える前に一切の作業状態を破棄する。"""
        self._end_stroke_cache()
        self._move_base_image = None
        self._move_base_pos = None
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
        self._perspective_corners = None
        self._perspective_corners_start = None
        self._perspective_drag_idx = -1
        self._mesh_grid = None
        self._mesh_grid_start = None
        self._mesh_drag_idx = (-1, -1)
        self._selection_rect = None
        self._lasso_mask = None
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
            # Ctrl+スクロール → ズーム
            factor = 1.15 if delta > 0 else 1 / 1.15
            self.set_zoom(self.zoom * factor)
            event.accept()
        else:
            super().wheelEvent(event)

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
                step = 10 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1
                ddx, ddy = arrow_map[event.key()]
                if layer and layer.is_group:
                    children = self._collect_leaf_layers(layer)
                    if children:
                        for child in children:
                            ox = getattr(child, 'offset_x', 0)
                            oy = getattr(child, 'offset_y', 0)
                            self._history.append(("pixel", id(child), child.image.copy(), ox, oy))  # type: ignore
                        self._redo_stack.clear()
                        if len(self._history) > HISTORY_LIMIT:
                            self._history = self._history[-HISTORY_LIMIT:]
                        for child in children:
                            child.offset_x += ddx * step  # type: ignore
                            child.offset_y += ddy * step  # type: ignore
                elif layer and not layer.is_group:
                    self._save_history()
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
