"""ユニットテスト: canvas.py の内部ロジック関数（GUIウィジェット不要なもの）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor, QPainter, QPen
from PyQt6.QtCore import Qt, QPoint

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
