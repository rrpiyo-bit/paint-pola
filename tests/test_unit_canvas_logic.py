"""ユニットテスト: canvas.py の内部ロジック関数（GUIウィジェット不要なもの）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor, QPainter, QPen, QMouseEvent
from PyQt6.QtCore import Qt, QPoint, QPointF, QRect

app = QApplication.instance() or QApplication(sys.argv)

import numpy as np

import canvas as canvas_mod
_flood_fill = canvas_mod._flood_fill
_flood_fill_expanded = canvas_mod._flood_fill_expanded
_fill_closed_regions_in_area = canvas_mod._fill_closed_regions_in_area
Canvas = canvas_mod.Canvas

from layer import Layer, LayerStack

W, H = 60, 60


def px(image: QImage, x: int, y: int) -> QColor:
    return QColor.fromRgba(image.pixel(x, y))


def make_white_image(w=W, h=H) -> QImage:
    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.white)
    return img


def make_transparent_image(w=W, h=H) -> QImage:
    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    return img


# ── _flood_fill ───────────────────────────────────────────────────────────────

class TestFloodFill:
    def test_fill_solid_white_canvas(self):
        img = make_white_image()
        red = QColor(255, 0, 0, 255)
        _flood_fill(img, 30, 30, red, None)
        c = px(img, 30, 30)
        assert c.red() == 255 and c.green() == 0

    def test_fill_entire_canvas(self):
        """境界なしの白キャンバス全体が塗られる。"""
        img = make_white_image()
        blue = QColor(0, 0, 255, 255)
        _flood_fill(img, 0, 0, blue, None)
        assert px(img, 0, 0).blue() > 200
        assert px(img, W-1, H-1).blue() > 200

    def test_fill_transparent_image(self):
        img = make_transparent_image()
        green = QColor(0, 255, 0, 255)
        _flood_fill(img, 10, 10, green, None)
        c = px(img, 10, 10)
        assert c.green() == 255

    def test_fill_out_of_bounds_noop(self):
        """範囲外座標でクラッシュしない。"""
        img = make_white_image(10, 10)
        _flood_fill(img, 50, 50, QColor(255, 0, 0, 255), None)

    def test_fill_same_color_noop(self):
        """同じ色への塗りつぶしは変化なし。"""
        img = make_white_image()
        before_pixel = img.pixel(30, 30)
        _flood_fill(img, 30, 30, QColor(255, 255, 255, 255), None)
        assert img.pixel(30, 30) == before_pixel

    def test_fill_stops_at_different_color_pixel(self):
        """flood fill は塗り対象色と異なるピクセルで止まる。"""
        img = make_white_image()
        # 左半分を黒で塗る
        p = QPainter(img)
        p.fillRect(0, 0, W // 2, H, QColor(0, 0, 0, 255))
        p.end()
        blue = QColor(0, 0, 255, 255)
        # 右半分（白）を塗る
        _flood_fill(img, W - 5, H // 2, blue, None)
        # 右半分は青になっている
        assert px(img, W - 5, H // 2).blue() > 200
        # 黒い左半分は変わらない（黒 ≠ 白 なので塗られない）
        c_black = px(img, 5, H // 2)
        assert c_black.blue() < 50 and c_black.red() < 50

    def test_fill_with_ref_image_transparent_target(self):
        """参照レイヤーで塗る領域を制御する。
        ref が透明なピクセル (alpha<=10) を「塗れる」場所として扱う。"""
        target = make_transparent_image()
        # ref も透明（alpha=0）なら target 上に塗れる
        ref = make_transparent_image()
        # ref の中央だけ不透明にして「塗れない」壁にする
        p = QPainter(ref)
        p.fillRect(25, 25, 10, 10, QColor(0, 0, 0, 255))
        p.end()
        red = QColor(255, 0, 0, 255)
        _flood_fill(target, 0, 0, red, ref)
        # 開始点は赤で塗られる（ref が透明なので塗れる）
        assert px(target, 0, 0).red() > 200
        # 壁の中（ref が不透明）は塗られない
        assert px(target, 30, 30).alpha() == 0


# ── _flood_fill_expanded ──────────────────────────────────────────────────────

class TestFloodFillExpanded:
    def test_expand_zero_matches_normal(self):
        img1 = make_white_image()
        img2 = make_white_image()
        # まず両方に閉じた枠を描く
        for img in (img1, img2):
            p = QPainter(img)
            p.setPen(QPen(QColor(0, 0, 0, 255)))
            p.drawRect(15, 15, 30, 30)
            p.end()
        red = QColor(255, 0, 0, 255)
        _flood_fill(img1, 30, 30, red, None)
        _flood_fill_expanded(img2, 30, 30, red, None, 0)
        # 同じ結果
        assert px(img1, 30, 30).red() > 200
        assert px(img2, 30, 30).red() > 200

    def test_expand_positive_extends_beyond_boundary(self):
        """拡張ありは境界を越えて塗る。"""
        img = make_white_image()
        p = QPainter(img)
        p.setPen(QPen(QColor(0, 0, 0, 255)))
        p.drawRect(15, 15, 30, 30)
        p.end()
        red = QColor(255, 0, 0, 255)
        _flood_fill_expanded(img, 30, 30, red, None, 5)
        # 境界近く（枠の直外側）も赤になっているはず
        assert px(img, 12, 30).red() > 200

    def test_expand_negative_result_differs_from_positive(self):
        """拡張(+)と縮小(-)で結果が異なることを確認する。"""
        # 小さい領域を塗った後、+ と - で結果が変わることを確認
        img_plus = make_white_image()
        img_minus = make_white_image()
        # 両方に閉じた小領域を黒ピクセルで作る（左半分=黒, 右半分=白）
        for img in (img_plus, img_minus):
            p = QPainter(img)
            p.fillRect(0, 0, W // 2, H, QColor(0, 0, 0, 255))
            p.end()
        red = QColor(255, 0, 0, 255)
        _flood_fill_expanded(img_plus, W - 5, H // 2, red, None, 3)   # 拡張
        _flood_fill_expanded(img_minus, W - 5, H // 2, red, None, -3)  # 縮小
        # 拡張版は白領域の境界近くまで塗られる（赤ピクセルが黒の近くにある）
        # 縮小版は白領域の中央のみ塗られる
        center_plus = px(img_plus, W - 5, H // 2).red()
        center_minus = px(img_minus, W - 5, H // 2).red()
        # どちらも中央は赤（塗られる）
        assert center_plus > 200
        assert center_minus > 200
        # 拡張版は黒境界に近いピクセルも赤になっている
        near_boundary_plus = px(img_plus, W // 2 + 2, H // 2).red()
        near_boundary_minus = px(img_minus, W // 2 + 2, H // 2).red()
        # 拡張 >= 縮小 であることを確認（縮小版は境界近くが白に戻る）
        assert near_boundary_plus >= near_boundary_minus


# ── _fill_closed_regions_in_area（投げなわ選択内の閉領域塗りつぶし）──────────────

class TestFillClosedRegionsInArea:
    """クリスタ風「投げなわで囲んだ範囲内の閉じた線画領域だけを塗る」機能。
    大キャンバス・多数の閉領域でも高速に処理できるよう、cv2.connectedComponents の
    ラベルをそのまま numpy で一括書き込みする実装になっている（QImage.setPixel の
    逐次呼び出しによるフリーズ/クラッシュを避けるため）。"""

    def test_closed_region_is_filled(self):
        img = make_transparent_image()
        p = QPainter(img)
        p.setPen(QPen(QColor(0, 0, 0, 255)))
        p.drawRect(10, 10, 30, 30)  # 完全に閉じた四角
        p.end()

        # area_mask は画像端に達しないようにする（端に達すると外周は常に閉扱いになる
        # 既存の境界判定仕様のため、テストの意図がぼやけるのを避ける）
        area_mask = np.zeros((H, W), dtype=np.uint8)
        area_mask[5:H - 5, 5:W - 5] = 1

        filled = _fill_closed_regions_in_area(img, area_mask, QColor(0, 255, 0, 255), None)
        assert filled == 1
        assert px(img, 25, 25).green() == 255
        # 境界線自体は塗り替えられない
        assert px(img, 10, 25).green() != 255

    def test_open_region_is_not_filled(self):
        img = make_transparent_image()
        p = QPainter(img)
        p.setPen(QPen(QColor(0, 0, 0, 255)))
        # 下辺のない、閉じていない四角
        p.drawLine(10, 10, 40, 10)
        p.drawLine(10, 10, 10, 40)
        p.drawLine(40, 10, 40, 40)
        p.end()

        area_mask = np.zeros((H, W), dtype=np.uint8)
        area_mask[5:H - 5, 5:W - 5] = 1

        filled = _fill_closed_regions_in_area(img, area_mask, QColor(0, 255, 0, 255), None)
        assert filled == 0

    def test_empty_area_mask_fills_nothing(self):
        img = make_transparent_image()
        area_mask = np.zeros((H, W), dtype=np.uint8)
        filled = _fill_closed_regions_in_area(img, area_mask, QColor(255, 0, 0, 255), None)
        assert filled == 0

    def test_multiple_closed_regions_all_filled(self):
        """投げなわ内に複数の閉領域があれば全て塗られる（多数領域での一括処理を確認）。"""
        img = make_transparent_image()
        p = QPainter(img)
        p.setPen(QPen(QColor(0, 0, 0, 255)))
        p.drawRect(5, 5, 10, 10)
        p.drawRect(20, 5, 10, 10)
        p.drawRect(5, 20, 10, 10)
        p.drawRect(20, 20, 10, 10)
        p.end()

        area_mask = np.zeros((H, W), dtype=np.uint8)
        area_mask[0:H, 0:W - 5] = 1  # 4つの四角を含み画像端に達しない範囲

        filled = _fill_closed_regions_in_area(img, area_mask, QColor(255, 0, 255, 255), None)
        assert filled == 4
        assert px(img, 10, 10).red() == 255
        assert px(img, 25, 10).red() == 255
        assert px(img, 10, 25).red() == 255
        assert px(img, 25, 25).red() == 255


# ── Canvas._apply_lasso_fill（投げなわツールのマウス操作フロー）───────────────

class TestApplyLassoFill:
    """貼り付け直後のレイヤー等、layer.image がキャンバスより大きく offset_x/offset_y
    を持つ場合にクラッシュしないこと（実際のユーザーファイルで再現した不具合の回帰確認）。
    投げなわの点はキャンバス座標系で来るため、layer.image のローカル座標系に変換してから
    処理する必要がある。変換を忘れると area_mask と layer.image の shape が食い違い、
    numpy の broadcast エラーで落ちる。"""

    def test_oversized_offset_layer_does_not_crash(self):
        """キャンバスより大きく offset を持つレイヤーでもクラッシュしない。"""
        stack = LayerStack(200, 200)
        lyr = Layer("big", 400, 400)
        lyr.image.fill(Qt.GlobalColor.transparent)
        lyr.offset_x = -50
        lyr.offset_y = -50
        stack.layers = [lyr]
        stack.active_path = [0]

        c = Canvas(stack)
        pts = [QPoint(40, 40), QPoint(110, 40), QPoint(110, 110), QPoint(40, 110)]
        c._apply_lasso_fill(lyr, pts)  # 例外が出ないこと

    def test_oversized_offset_layer_fills_correct_local_position(self):
        """キャンバス座標で指定した投げなわが、レイヤーのローカル座標系の正しい位置に反映される。"""
        stack = LayerStack(200, 200)
        lyr = Layer("big", 400, 400)
        lyr.image.fill(Qt.GlobalColor.transparent)
        lyr.offset_x = -50
        lyr.offset_y = -50
        # レイヤーローカル座標 (100,100)-(150,150) に閉じた四角 → キャンバス座標では (50,50)-(100,100)
        p = QPainter(lyr.image)
        p.setPen(QPen(QColor(0, 0, 0, 255)))
        p.drawRect(100, 100, 50, 50)
        p.end()
        stack.layers = [lyr]
        stack.active_path = [0]

        c = Canvas(stack)
        c.pen_color = QColor(255, 0, 0, 255)
        # キャンバス座標 (40,40)-(110,110) で四角を囲む
        pts = [QPoint(40, 40), QPoint(110, 40), QPoint(110, 110), QPoint(40, 110)]
        c._apply_lasso_fill(lyr, pts)

        assert px(lyr.image, 125, 125).red() == 255   # 四角の内側（ローカル座標）
        assert px(lyr.image, 10, 10).alpha() == 0      # 範囲外は変化なし


class TestSelectLayerAlpha:
    """レイヤーサムネイルCtrlクリックで、レイヤーの不透明部分の形の選択範囲を作る機能。"""

    def test_basic_shape_selection(self):
        """不透明な四角の部分だけが選択範囲になる。"""
        stack = LayerStack(200, 200)
        lyr = Layer("shape", 200, 200)
        lyr.image.fill(Qt.GlobalColor.transparent)
        p = QPainter(lyr.image)
        p.fillRect(50, 50, 40, 30, QColor(0, 0, 0, 255))
        p.end()
        stack.layers = [lyr]
        stack.active_path = [0]

        c = Canvas(stack)
        ok = c.select_layer_alpha(lyr)

        assert ok is True
        assert c._selection_rect is not None
        assert c._selection_rect.left() == 50 and c._selection_rect.top() == 50
        assert c._selection_rect.width() == 40 and c._selection_rect.height() == 30
        assert c._lasso_mask is not None
        assert px(c._lasso_mask, 70, 60).alpha() > 0    # 図形の内側
        assert px(c._lasso_mask, 10, 10).alpha() == 0   # 図形の外側

    def test_offset_layer_maps_to_canvas_coordinates(self):
        """offset_x/offset_y を持つレイヤーでもキャンバス座標系で選択範囲が作られる。"""
        stack = LayerStack(200, 200)
        lyr = Layer("shape", 400, 400)
        lyr.image.fill(Qt.GlobalColor.transparent)
        lyr.offset_x = -50
        lyr.offset_y = -50
        p = QPainter(lyr.image)
        # レイヤーローカル座標 (100,100)-(150,150) → キャンバス座標では (50,50)-(100,100)
        p.fillRect(100, 100, 50, 50, QColor(0, 0, 0, 255))
        p.end()
        stack.layers = [lyr]
        stack.active_path = [0]

        c = Canvas(stack)
        ok = c.select_layer_alpha(lyr)

        assert ok is True
        assert c._selection_rect.left() == 50 and c._selection_rect.top() == 50
        assert c._selection_rect.width() == 50 and c._selection_rect.height() == 50

    def test_empty_layer_returns_false(self):
        """完全に透明なレイヤーでは選択範囲を作らない。"""
        stack = LayerStack(100, 100)
        lyr = Layer("empty", 100, 100)
        lyr.image.fill(Qt.GlobalColor.transparent)
        stack.layers = [lyr]
        stack.active_path = [0]

        c = Canvas(stack)
        ok = c.select_layer_alpha(lyr)

        assert ok is False
        assert c._selection_rect is None

    def test_group_layer_returns_false(self):
        """グループレイヤーは対象外。"""
        from layer import GroupLayer
        stack = LayerStack(100, 100)
        grp = GroupLayer("group")
        stack.layers = [grp]
        stack.active_path = [0]

        c = Canvas(stack)
        ok = c.select_layer_alpha(grp)

        assert ok is False


class TestCommitTransformGrowsLayer:
    """拡大縮小・回転でキャンバス外に絵がはみ出しても、layer.image が自動で
    拡張されクリップされないこと（レイヤー全体を拡大するとキャンバス外の
    部分が切れてしまう不具合の回帰確認）。"""

    def test_scale_up_beyond_canvas_is_not_clipped(self):
        stack = LayerStack(100, 100)
        lyr = Layer("shape", 100, 100)
        lyr.image.fill(Qt.GlobalColor.transparent)
        p = QPainter(lyr.image)
        # lift_whole_layer は不透明部分の外接矩形を変形基準にするため、この矩形
        # (40,40,20,20) が拡縮の中心・基準サイズになる。500%に拡大するとその
        # 中心 (50,50) を軸に 100x100 まで広がり、ちょうどキャンバスと同サイズに
        # なってしまいはみ出さないため、余裕を持って 900% まで拡大する。
        p.fillRect(40, 40, 20, 20, QColor(255, 0, 0, 255))
        p.end()
        stack.layers = [lyr]
        stack.active_path = [0]

        c = Canvas(stack)
        assert c.lift_whole_layer() is True
        # 900%に拡大 → 中心固定なのでキャンバスの外まで大きくはみ出す
        c.apply_transform_percentage(900.0, 900.0, 0.0)
        c._commit_transform()

        # レイヤーの端（かつてのキャンバス境界付近）に赤色が残っている＝クリップされていない
        ox, oy = lyr.offset_x, lyr.offset_y
        # offset分だけ左上にずれているはずなので、画像は元のキャンバスより大きい
        assert lyr.image.width() > 100 or lyr.image.height() > 100
        assert ox < 0 or oy < 0

        # 拡大後の中心付近は赤で塗られているはず
        cx, cy = 50 - ox, 50 - oy
        assert px(lyr.image, cx, cy).red() == 255

    def test_rotated_scale_up_is_not_clipped(self):
        """回転を伴う変形でも、回転後のバウンディングボックスに収まるよう拡張される。"""
        stack = LayerStack(100, 100)
        lyr = Layer("shape", 100, 100)
        lyr.image.fill(Qt.GlobalColor.transparent)
        p = QPainter(lyr.image)
        # lift_whole_layer は不透明部分の外接矩形 (0,0,100,100 の全面) を基準に
        # するよう、キャンバス全域に近い矩形を塗って回転後のバウンディングボックス
        # が確実にキャンバスをはみ出すようにする。
        p.fillRect(5, 5, 90, 90, QColor(0, 255, 0, 255))
        p.end()
        stack.layers = [lyr]
        stack.active_path = [0]

        c = Canvas(stack)
        assert c.lift_whole_layer() is True
        c.apply_transform_percentage(100.0, 100.0, 45.0)
        c._commit_transform()

        # 45度回転すると軸並行バウンディングボックスは元の矩形より大きくなる
        assert lyr.image.width() > 100 or lyr.image.height() > 100


class TestTransformModeResetsAfterCommit:
    """変形モード（パース・メッシュ）は変形確定後に標準へ戻ること。
    ツールオプションパネルの「変形モード」コンボはツール切替のたびに
    表示が「標準」にリセットされるため、内部フラグ（_perspective_mode /
    _mesh_mode）を残したままにすると、2回目以降の変形で表示（標準）と
    実際の挙動（パース/メッシュのハンドル）がずれ、回転ハンドルが出ない・
    頂点がつかめないなどの不具合になる（回帰確認）。"""

    def _make_canvas(self):
        stack = LayerStack(200, 200)
        lyr = Layer("l", 200, 200)
        lyr.image.fill(Qt.GlobalColor.transparent)
        p = QPainter(lyr.image)
        p.fillRect(20, 20, 60, 60, QColor(255, 0, 0, 255))
        p.end()
        stack.layers = [lyr]
        stack.active_path = [0]
        c = Canvas(stack)
        c._selection_rect = None
        return c, lyr

    def test_perspective_mode_does_not_leak_into_next_transform(self):
        c, lyr = self._make_canvas()
        assert c.lift_whole_layer() is True
        c.set_transform_mode("perspective")
        assert c._perspective_corners is not None
        c._commit_transform()

        assert c.transform_mode == "standard"

        assert c.lift_whole_layer() is True
        assert c._perspective_corners is None
        assert c._mesh_grid is None
        assert c.transform_mode == "standard"

    def test_mesh_mode_does_not_leak_into_next_transform(self):
        c, lyr = self._make_canvas()
        assert c.lift_whole_layer() is True
        c.set_transform_mode("mesh")
        assert c._mesh_grid is not None
        c._commit_transform()

        assert c.transform_mode == "standard"

        assert c.lift_whole_layer() is True
        assert c._mesh_grid is None
        assert c._perspective_corners is None

    def test_cancel_transform_also_resets_mode(self):
        c, lyr = self._make_canvas()
        assert c.lift_whole_layer() is True
        c.set_transform_mode("perspective")
        c.cancel_transform()

        assert c.transform_mode == "standard"


class TestPerspectiveCornerDragViaSelectRect:
    """SELECT_RECT ツールの「選択範囲内クリックで変形」から自由変形（パース）に
    入った場合でも、四隅を個別にドラッグして斜めに変形できること。
    以前は Tool.TRANSFORM 専用の _handle_transform_press だけが
    _perspective_corners_start / _mesh_grid_start を初期化しており、
    SELECT_RECT/LASSO 経由のドラッグ開始処理ではその初期化が漏れていたため、
    _drag_transform の 'if self._perspective_corners_start' 判定が常に False になり
    頂点ドラッグが一切効かなかった（回帰確認）。"""

    def _press(self, c, x, y):
        c.mousePressEvent(QMouseEvent(
            QMouseEvent.Type.MouseButtonPress, QPointF(x, y),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier))

    def _move(self, c, x, y):
        c.mouseMoveEvent(QMouseEvent(
            QMouseEvent.Type.MouseMove, QPointF(x, y),
            Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier))

    def _release(self, c, x, y):
        c.mouseReleaseEvent(QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease, QPointF(x, y),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier))

    def test_dragging_one_corner_skews_independently(self):
        from tools import Tool

        stack = LayerStack(200, 200)
        lyr = Layer("l", 200, 200)
        stack.layers = [lyr]
        stack.active_path = [0]

        c = Canvas(stack)
        c.resize(200, 200)
        c.zoom = 1.0
        c.tool = Tool.SELECT_RECT
        c.select_mode = "select"

        self._press(c, 20, 20)
        self._release(c, 100, 100)

        c.select_mode = "transform"
        self._press(c, 50, 50)   # 選択範囲内クリック → lift
        self._release(c, 50, 50)
        c.set_transform_mode("perspective")
        before = list(c._perspective_corners)

        self._press(c, 20, 20)   # tl ハンドル
        self._move(c, 0, 60)     # tl だけを動かす（他の3隅は固定のまま）
        self._release(c, 0, 60)

        after = c._perspective_corners
        assert after[0] != before[0]          # tl は動いた
        assert after[1] == before[1]          # tr は固定
        assert after[2] == before[2]          # br は固定
        assert after[3] == before[3]          # bl は固定


# ── バケツ塗り: 隙間閉じ / 薄い線を拾う感度 ────────────────────────────────

def _boxed_ref(gap: int = 0, line_alpha: int = 255, w: int = 100, h: int = 100):
    """(20,20)-(79,79) の枠線1pxの矩形。上辺中央に gap px の切れ目を空ける。"""
    ref = QImage(w, h, QImage.Format.Format_ARGB32)
    ref.fill(QColor(0, 0, 0, 0))
    p = QPainter(ref)
    p.setPen(QPen(QColor(0, 0, 0, 255), 1))
    p.drawRect(20, 20, 59, 59)
    p.end()
    for x in range(45, 45 + gap):
        ref.setPixelColor(x, 20, QColor(0, 0, 0, 0))
    if line_alpha != 255:
        # 上辺の一部だけを「薄いが確かに描かれている」線に置き換える
        for x in range(45, 56):
            ref.setPixelColor(x, 20, QColor(0, 0, 0, line_alpha))
    return ref


def _fill_and_count(ref, close_gap=0, threshold=10, seed=(50, 50)):
    img = QImage(ref.width(), ref.height(), QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    _flood_fill(img, seed[0], seed[1], QColor(255, 0, 0, 255),
                ref, close_gap, threshold)
    return sum(1 for y in range(ref.height()) for x in range(ref.width())
               if img.pixelColor(x, y).alpha() > 0)


# 閉じた矩形の内側 58x58
INSIDE = 3364


class TestFillCloseGap:
    """線画が途切れていても塗ってくれる機能。"""

    def test_gap_leaks_without_close(self):
        """隙間があると、閉じない限り外へ漏れる（この機能が必要な理由）。"""
        assert _fill_and_count(_boxed_ref(gap=3), close_gap=0) > INSIDE * 2

    @pytest.mark.parametrize("gap,close", [(1, 1), (3, 2), (3, 3), (5, 3), (5, 5)])
    def test_gap_closed_stops_leak(self, gap, close):
        """隙間幅の半分以上を指定すれば漏れが止まる。"""
        n = _fill_and_count(_boxed_ref(gap=gap), close_gap=close)
        assert n < INSIDE * 1.2

    @pytest.mark.parametrize("close", [1, 2, 3, 5])
    def test_fill_does_not_shrink(self, close):
        """隙間閉じは判定用マスクだけを太らせるので、塗り範囲は痩せない。
        （膨張し直さない実装だと close px 分だけ内側に縮んでしまう）"""
        n = _fill_and_count(_boxed_ref(gap=1), close_gap=close)
        assert n > INSIDE * 0.95

    def test_close_gap_does_not_cross_line(self):
        """膨張し直すときに線を越えて外側へはみ出さない。"""
        assert _fill_and_count(_boxed_ref(gap=1), close_gap=5) <= INSIDE

    def test_huge_close_gap_falls_back(self):
        """閉じ幅が大きすぎてシードごと潰れる場合は、隙間閉じを諦めて通常塗り。
        （何も塗れずに無反応になるのを防ぐ）"""
        assert _fill_and_count(_boxed_ref(gap=0), close_gap=40) == INSIDE

    def test_no_gap_unaffected(self):
        """隙間がない絵では結果が変わらない。"""
        assert _fill_and_count(_boxed_ref(), close_gap=3) <= INSIDE


class TestFillLineSensitivity:
    """色が薄い線が「途切れ」と誤判定されるのを防ぐ機能。"""

    def test_sensitivity_to_threshold_range(self):
        s2t = canvas_mod._sensitivity_to_threshold
        assert s2t(0) == canvas_mod.LINE_ALPHA_THRESHOLD  # 既定の挙動
        assert s2t(100) == 0                               # alpha があれば線
        assert s2t(0) > s2t(50) > s2t(100)                 # 上げるほど拾う
        assert s2t(-50) == s2t(0) and s2t(500) == s2t(100)  # 範囲外はクランプ

    @pytest.mark.parametrize("alpha", [3, 6, 9])
    def test_faint_line_leaks_at_default(self, alpha):
        """既定では alpha 10 以下の線をすり抜けて漏れる（この機能が必要な理由）。"""
        ref = _boxed_ref(line_alpha=alpha)
        assert _fill_and_count(ref, threshold=10) > INSIDE * 2

    @pytest.mark.parametrize("alpha", [3, 6, 9])
    def test_max_sensitivity_stops_faint_line(self, alpha):
        """感度100%なら alpha が少しでもあれば線として堰き止める。"""
        ref = _boxed_ref(line_alpha=alpha)
        thr = canvas_mod._sensitivity_to_threshold(100)
        assert _fill_and_count(ref, threshold=thr) == INSIDE

    def test_normal_line_unaffected_by_sensitivity(self):
        """濃い線しかない絵では感度を変えても結果が変わらない。"""
        ref = _boxed_ref()
        base = _fill_and_count(ref, threshold=10)
        for s in (0, 50, 100):
            thr = canvas_mod._sensitivity_to_threshold(s)
            assert _fill_and_count(ref, threshold=thr) == base

    def test_seed_on_faint_line_is_blocked_at_high_sensitivity(self):
        """感度を上げると、薄い線の上をクリックしても塗られない（線扱いになる）。"""
        ref = _boxed_ref(line_alpha=6)
        thr = canvas_mod._sensitivity_to_threshold(100)
        assert _fill_and_count(ref, threshold=thr, seed=(50, 20)) == 0


class TestCanvasFillOptionDefaults:
    """Canvas 側の既定値が従来の挙動を変えないこと。"""

    def test_defaults(self):
        ls = LayerStack(50, 50)
        ls.layers = [Layer("L", 50, 50)]
        ls.active_path = [0]
        c = Canvas(ls)
        assert c.fill_close_gap == 0
        assert c.fill_line_sensitivity == 0
        # 感度0% は従来の alpha>10 判定と等価
        assert canvas_mod._sensitivity_to_threshold(c.fill_line_sensitivity) == 10


class TestFillOnOpaqueBackground:
    """白背景で描かれた線画レイヤーを参照にしても塗れること（不透明＝全部線 の誤判定対策）。"""

    def _white_bg_ref(self, w=100, h=100):
        ref = QImage(w, h, QImage.Format.Format_ARGB32)
        ref.fill(QColor(255, 255, 255, 255))
        p = QPainter(ref)
        p.setPen(QPen(QColor(0, 0, 0, 255), 3))
        p.drawRect(20, 20, 59, 59)
        p.end()
        return ref

    def _fill(self, ref, seed):
        img = QImage(ref.width(), ref.height(), QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        _flood_fill(img, seed[0], seed[1], QColor(255, 0, 0, 255), ref)
        return img

    def test_inside_fills_on_white_background(self):
        img = self._fill(self._white_bg_ref(), (50, 50))
        assert img.pixelColor(50, 50).alpha() > 0

    def test_line_still_blocks_leak_to_outside(self):
        img = self._fill(self._white_bg_ref(), (50, 50))
        assert img.pixelColor(2, 2).alpha() == 0

    def test_outside_fills_without_touching_inside(self):
        img = self._fill(self._white_bg_ref(), (2, 2))
        assert img.pixelColor(2, 2).alpha() > 0
        assert img.pixelColor(50, 50).alpha() == 0

    def test_transparent_background_unchanged(self):
        """透明背景の通常ケースは従来どおり（地色判定が誤発動しない）。"""
        assert _fill_and_count(_boxed_ref()) == INSIDE

    def test_dark_opaque_background_still_treated_as_line(self):
        """暗いベタ塗り面は地色ではないので、従来どおり境界のまま。"""
        ref = QImage(100, 100, QImage.Format.Format_ARGB32)
        ref.fill(QColor(20, 20, 20, 255))
        img = self._fill(ref, (50, 50))
        assert img.pixelColor(50, 50).alpha() == 0



class TestFillReferenceMode:
    """複数参照（クリスタ相当）: 編集中レイヤーに描いた囲み線で塗りが止まるか。

    参照レイヤーが線画だけのとき、塗る側に描いた〇は境界判定に入らないため
    素通りして外まで漏れていた。ref_self では自分のレイヤーの不透明部分も
    境界に含める。ref（参照のみ）は従来どおりで、アニメ塗りの塗り分け用。
    """

    CYAN = QColor(0, 174, 205, 255)
    RED = QColor(255, 0, 0, 255)
    SIZE = 200

    def _ref_lineart(self):
        """線画レイヤー。〇は含まない（〇は編集側に描かれている）。"""
        ref = QImage(self.SIZE, self.SIZE, QImage.Format.Format_ARGB32)
        ref.fill(Qt.GlobalColor.transparent)
        p = QPainter(ref)
        p.setPen(QPen(QColor(0, 0, 0, 255), 4))
        p.drawLine(0, 170, self.SIZE, 170)
        p.end()
        return ref

    def _target_with_circle(self):
        """編集中レイヤー。ここに水色の〇を描いてある。"""
        t = QImage(self.SIZE, self.SIZE, QImage.Format.Format_ARGB32)
        t.fill(Qt.GlobalColor.transparent)
        p = QPainter(t)
        p.setPen(QPen(self.CYAN, 12))
        p.drawEllipse(50, 40, 90, 90)
        p.end()
        return t

    def _mask(self, img, color):
        ptr = img.bits()
        ptr.setsize(img.sizeInBytes())
        a = np.frombuffer(ptr, dtype=np.uint8).reshape(self.SIZE, self.SIZE, 4)
        want = np.array([color.blue(), color.green(),
                         color.red(), color.alpha()], dtype=np.uint8)
        return np.all(a == want, axis=2)

    def test_self_line_blocks_fill(self):
        """ref_self: 〇の中をクリックしたら、〇の外へ漏れない。"""
        t = self._target_with_circle()
        _flood_fill(t, 95, 85, self.RED, self._ref_lineart(), 0, 10, True)
        red = self._mask(t, self.RED)
        assert red[85, 95], "〇の中が塗れていない"
        assert not red[20, 20], "〇の外まで漏れている"
        assert red.mean() < 0.30, "塗り面積が広すぎる（漏れている）"

    def test_reference_only_keeps_old_behavior(self):
        """ref: 従来どおり自分の線は無視する（アニメ塗りの塗り分け用）。"""
        t = self._target_with_circle()
        _flood_fill(t, 95, 85, self.RED, self._ref_lineart(), 0, 10, False)
        red = self._mask(t, self.RED)
        assert red[20, 20], "参照のみモードの挙動が変わってしまった"

    def test_default_is_reference_only(self):
        """引数を省略したときは従来の挙動（＝既存の呼び出しを壊さない）。"""
        a = self._target_with_circle()
        b = self._target_with_circle()
        _flood_fill(a, 95, 85, self.RED, self._ref_lineart())
        _flood_fill(b, 95, 85, self.RED, self._ref_lineart(), 0, 10, False)
        assert self._mask(a, self.RED).sum() == self._mask(b, self.RED).sum()

    def test_enclosing_line_is_preserved(self):
        """塗っても囲み線そのものは消えない。"""
        t = self._target_with_circle()
        before = self._mask(t, self.CYAN).sum()
        _flood_fill(t, 95, 85, self.RED, self._ref_lineart(), 0, 10, True)
        assert self._mask(t, self.CYAN).sum() == before

    def test_seed_on_own_line_does_nothing(self):
        """囲み線そのものをクリックしても、線は境界なので塗られない。"""
        t = self._target_with_circle()
        _flood_fill(t, 95, 45, self.RED, self._ref_lineart(), 0, 10, True)
        assert self._mask(t, self.RED).sum() == 0

    @pytest.mark.parametrize("gap,expand", [(0, 0), (3, 0), (0, 2), (3, 2)])
    def test_no_leak_with_gap_and_expand(self, gap, expand):
        """隙間閉じ・拡張と併用しても、自分の線を越えて漏れない。"""
        t = self._target_with_circle()
        _flood_fill_expanded(t, 95, 85, self.RED, self._ref_lineart(),
                             expand, gap, 10, True)
        assert not self._mask(t, self.RED)[20, 20], \
            f"隙間={gap} 拡張={expand} で漏れた"

    def test_canvas_default_mode(self):
        """Canvas の既定は「参照＋編集」。手描きを囲んで塗る使い方が事故らない。"""
        ls = LayerStack(50, 50)
        ls.layers = [Layer("L", 50, 50)]
        ls.active_path = [0]
        c = Canvas(ls)
        assert c.fill_reference_mode == "ref_self"


class TestInvertSelection:
    """選択範囲の反転（編集メニュー / 選択ツールのオプション）。"""

    def _canvas(self, w=60, h=60):
        stack = LayerStack(w, h)
        lyr = Layer("a", w, h)
        lyr.image.fill(Qt.GlobalColor.transparent)
        stack.layers = [lyr]
        stack.active_path = [0]
        return Canvas(stack), lyr

    def _selected(self, c) -> int:
        m = c._lasso_mask
        ptr = m.bits(); ptr.setsize(m.height() * m.width() * 4)
        a = np.frombuffer(ptr, dtype=np.uint8).reshape(m.height(), m.width(), 4)
        return int((a[:, :, 3] > 0).sum())

    def test_rect_selection_inverts(self):
        """矩形選択を反転すると、矩形の外側だけが選ばれる。"""
        c, _ = self._canvas()
        c._selection_rect = QRect(10, 10, 20, 20)
        c.invert_selection()
        assert self._selected(c) == 60 * 60 - 20 * 20
        assert px(c._lasso_mask, 15, 15).alpha() == 0    # 元の選択の中
        assert px(c._lasso_mask, 2, 2).alpha() > 0       # 外側

    def test_lasso_selection_inverts(self):
        """投げなわ選択でも反転でき、選択＋非選択で全面になる。"""
        c, _ = self._canvas()
        poly = [QPoint(5, 5), QPoint(40, 5), QPoint(40, 40), QPoint(5, 40)]
        c._lasso_mask = canvas_mod._mask_from_polygon(poly, 60, 60)
        c._selection_rect = QRect(5, 5, 35, 35)
        before = self._selected(c)
        c.invert_selection()
        after = self._selected(c)
        assert before + after == 60 * 60

    def test_twice_restores_original(self):
        """2回反転すれば元の選択に戻る。"""
        c, _ = self._canvas()
        c._selection_rect = QRect(10, 10, 20, 20)
        c.invert_selection()
        c.invert_selection()
        assert self._selected(c) == 20 * 20
        assert c._selection_rect == QRect(10, 10, 20, 20)

    def test_bbox_shrinks_to_selected_area(self):
        """反転後の外接矩形は、実際に選ばれている範囲に合う。"""
        c, _ = self._canvas()
        # 左上を選んで反転すると、右下側だけが残る
        c._selection_rect = QRect(0, 0, 60, 30)
        c.invert_selection()
        assert c._selection_rect.top() == 30
        assert c._selection_rect.bottom() == 59

    def test_outline_is_built(self):
        """反転後も輪郭（点線表示用）が作られる。

        これがないと点線がキャンバス外周にしか出ず、
        どこが抜けているのか見た目で分からない。
        """
        c, _ = self._canvas()
        c._selection_rect = QRect(10, 10, 20, 20)
        c.invert_selection()
        assert c._selection_outline_path is not None
        assert c._selection_outline_path.elementCount() > 0

    def test_full_canvas_inversion_deselects(self):
        """全面選択を反転すると何も残らないので選択解除になる。"""
        c, _ = self._canvas()
        c.select_all()
        c.invert_selection()
        assert c._selection_rect is None
        assert c._lasso_mask is None

    def test_without_selection_is_noop(self):
        """選択がないときは何も起きない（例外も出ない）。"""
        c, _ = self._canvas()
        c.invert_selection()
        assert c._selection_rect is None

    def test_has_selection(self):
        c, _ = self._canvas()
        assert c.has_selection() is False
        c._selection_rect = QRect(1, 1, 5, 5)
        assert c.has_selection() is True

    def test_inverted_selection_drives_delete(self):
        """反転した選択で削除すると、元の選択内だけが残る。"""
        c, lyr = self._canvas()
        p = QPainter(lyr.image)
        p.fillRect(0, 0, 60, 60, QColor(255, 0, 0, 255))
        p.end()
        c._selection_rect = QRect(10, 10, 20, 20)
        c.invert_selection()
        c.delete_selection()
        assert px(lyr.image, 20, 20).alpha() > 0    # 元の選択内は残る
        assert px(lyr.image, 2, 2).alpha() == 0     # 外側は消える
