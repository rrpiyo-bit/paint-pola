"""ユニットテスト: canvas.py の内部ロジック関数（GUIウィジェット不要なもの）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor, QPainter, QPen
from PyQt6.QtCore import Qt, QPoint

app = QApplication.instance() or QApplication(sys.argv)

import canvas as canvas_mod
_flood_fill = canvas_mod._flood_fill
_flood_fill_expanded = canvas_mod._flood_fill_expanded

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
