"""ユニットテスト: canvas.py の内部ロジック関数（GUIウィジェット不要なもの）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor, QPainter, QPen, QMouseEvent
from PyQt6.QtCore import Qt, QPoint, QPointF

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
        p.fillRect(40, 40, 20, 20, QColor(255, 0, 0, 255))
        p.end()
        stack.layers = [lyr]
        stack.active_path = [0]

        c = Canvas(stack)
        assert c.lift_whole_layer() is True
        # 500%に拡大 → 中心固定なのでキャンバスの外まで大きくはみ出す
        c.apply_transform_percentage(500.0, 500.0, 0.0)
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
        p.fillRect(45, 10, 10, 10, QColor(0, 255, 0, 255))
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
