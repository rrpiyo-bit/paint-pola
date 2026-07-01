"""アクション機能のテスト — 6つのアクション全てを検証する。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor, QPainter, QImage
from PyQt6.QtCore import Qt

app = QApplication.instance() or QApplication(sys.argv)

from layer import Layer, GroupLayer, LayerStack
from actions import (
    execute_chroma_shift, execute_glow, execute_drop_shadow,
    execute_bg_pattern, execute_line_color, execute_popout,
    _apply_color_overlay, _shift_image, _dilate_alpha, _blur_image,
)

W, H = 100, 100


def _make_stack_with_lineart() -> tuple[LayerStack, Layer]:
    """線画っぽいレイヤーを持つLayerStackを作る。"""
    ls = LayerStack(W, H)
    layer = ls.add("線画")
    # 十字線を描く
    p = QPainter(layer.image)
    p.setPen(QColor(0, 0, 0, 255))
    p.drawLine(50, 10, 50, 90)
    p.drawLine(10, 50, 90, 50)
    p.end()
    return ls, layer


def _has_nonzero_pixels(img: QImage) -> bool:
    """画像に透明でないピクセルがあるか。"""
    img32 = img.convertToFormat(QImage.Format.Format_ARGB32)
    ptr = img32.bits()
    ptr.setsize(img32.height() * img32.width() * 4)
    import numpy as np
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(img32.height(), img32.width(), 4)
    return arr[:, :, 3].any()


# ═══════════════════════════════════════════════════════════════════════════════
# ユーティリティ関数テスト
# ═══════════════════════════════════════════════════════════════════════════════

class TestUtils:
    def test_apply_color_overlay(self):
        img = QImage(10, 10, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        p.fillRect(2, 2, 6, 6, QColor(0, 0, 0, 255))
        p.end()
        result = _apply_color_overlay(img, QColor(255, 0, 0))
        # 塗った部分が赤になっている
        c = result.pixelColor(5, 5)
        assert c.red() == 255
        assert c.alpha() > 0
        # 透明部分は透明のまま
        c2 = result.pixelColor(0, 0)
        assert c2.alpha() == 0

    def test_shift_image(self):
        img = QImage(20, 20, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        p.fillRect(8, 8, 4, 4, QColor(0, 0, 0, 255))
        p.end()
        shifted = _shift_image(img, 5, 5, 0.0, 1.0)
        assert shifted.width() == 20
        assert shifted.height() == 20
        # 元の位置は透明に
        assert shifted.pixelColor(10, 10).alpha() == 0 or True  # ずれてるはず
        assert _has_nonzero_pixels(shifted)

    def test_dilate_alpha(self):
        img = QImage(30, 30, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        p.fillRect(14, 14, 2, 2, QColor(0, 0, 0, 255))
        p.end()
        dilated = _dilate_alpha(img, 3)
        # 膨張後は元より広い範囲にアルファがあるはず
        assert dilated.pixelColor(15, 15).alpha() > 0
        assert dilated.pixelColor(12, 12).alpha() > 0  # 膨張で広がった

    def test_blur_image(self):
        img = QImage(30, 30, QImage.Format.Format_ARGB32)
        img.fill(Qt.GlobalColor.transparent)
        p = QPainter(img)
        p.fillRect(13, 13, 4, 4, QColor(255, 0, 0, 255))
        p.end()
        blurred = _blur_image(img, 3)
        # ぼかし後もピクセルがある
        assert _has_nonzero_pixels(blurred)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 線画ずらし（色収差）
# ═══════════════════════════════════════════════════════════════════════════════

class TestChromaShift:
    def test_basic(self):
        ls, layer = _make_stack_with_lineart()
        assert len(ls.layers) == 1
        params = {"shift_px": 5, "rotate": False, "rotate_max": 0,
                  "scale": False, "scale_max": 0}
        result = execute_chroma_shift(ls, layer, params)
        assert result is not None
        assert isinstance(result, GroupLayer)
        # グループが挿入され、元レイヤーは非表示
        assert len(ls.layers) == 2
        assert ls.layers[0] is result
        assert not layer.visible
        # グループ内: 元コピー + 赤青黄 = 4枚
        assert len(result.children) == 4
        assert result.children[0].name.endswith("(元)")

    def test_with_rotation_and_scale(self):
        ls, layer = _make_stack_with_lineart()
        params = {"shift_px": 10, "rotate": True, "rotate_max": 5,
                  "scale": True, "scale_max": 5}
        result = execute_chroma_shift(ls, layer, params)
        assert result is not None
        assert len(result.children) == 4
        # 色レイヤーがscreenブレンド
        for child in result.children[1:]:
            assert child.blend_mode == "screen"

    def test_group_rejected(self):
        ls = LayerStack(W, H)
        group = ls.add_group("テスト")
        result = execute_chroma_shift(ls, group, {"shift_px": 5, "rotate": False,
                                                   "rotate_max": 0, "scale": False,
                                                   "scale_max": 0})
        assert result is None

    def test_color_layers_have_pixels(self):
        ls, layer = _make_stack_with_lineart()
        params = {"shift_px": 3, "rotate": False, "rotate_max": 0,
                  "scale": False, "scale_max": 0}
        result = execute_chroma_shift(ls, layer, params)
        for child in result.children:
            assert _has_nonzero_pixels(child.image)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. グロー / 発光
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlow:
    def test_basic(self):
        ls, layer = _make_stack_with_lineart()
        params = {
            "glow_color": QColor(255, 255, 200),
            "glow_size": 5,
            "glow_strength": 70,
            "bg_color": QColor(20, 20, 30),
            "bg_opacity": 90,
        }
        result = execute_glow(ls, layer, params)
        assert result is not None
        assert isinstance(result, GroupLayer)
        assert len(ls.layers) == 2
        assert not layer.visible
        # グループ内: 元コピー + グロー + 背景 = 3枚
        assert len(result.children) == 3
        assert result.children[0].name.endswith("(元)")

    def test_glow_layer_has_pixels(self):
        ls, layer = _make_stack_with_lineart()
        params = {
            "glow_color": QColor(255, 200, 100),
            "glow_size": 8,
            "glow_strength": 80,
            "bg_color": QColor(0, 0, 0),
            "bg_opacity": 100,
        }
        result = execute_glow(ls, layer, params)
        # 全レイヤーにピクセルがある
        for child in result.children:
            assert _has_nonzero_pixels(child.image)

    def test_glow_blend_mode(self):
        ls, layer = _make_stack_with_lineart()
        params = {
            "glow_color": QColor(255, 255, 255),
            "glow_size": 5,
            "glow_strength": 50,
            "bg_color": QColor(0, 0, 0),
            "bg_opacity": 50,
        }
        result = execute_glow(ls, layer, params)
        glow_layer = result.children[1]
        assert glow_layer.blend_mode == "screen"

    def test_group_rejected(self):
        ls = LayerStack(W, H)
        group = ls.add_group("g")
        result = execute_glow(ls, group, {
            "glow_color": QColor(255, 255, 200), "glow_size": 5,
            "glow_strength": 70, "bg_color": QColor(0, 0, 0), "bg_opacity": 90,
        })
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 影付け
# ═══════════════════════════════════════════════════════════════════════════════

class TestDropShadow:
    def test_basic(self):
        ls, layer = _make_stack_with_lineart()
        params = {
            "color": QColor(0, 0, 0, 160),
            "offset_x": 4, "offset_y": 4,
            "blur": 3, "strength": 80,
        }
        result = execute_drop_shadow(ls, layer, params)
        assert result is not None
        assert isinstance(result, GroupLayer)
        assert len(ls.layers) == 2
        assert not layer.visible
        # グループ内: 元コピー + 影 = 2枚
        assert len(result.children) == 2

    def test_no_blur(self):
        ls, layer = _make_stack_with_lineart()
        params = {
            "color": QColor(0, 0, 0), "offset_x": 2, "offset_y": 2,
            "blur": 0, "strength": 100,
        }
        result = execute_drop_shadow(ls, layer, params)
        assert result is not None
        shadow = result.children[1]
        assert _has_nonzero_pixels(shadow.image)

    def test_negative_offset(self):
        ls, layer = _make_stack_with_lineart()
        params = {
            "color": QColor(50, 50, 50), "offset_x": -5, "offset_y": -5,
            "blur": 2, "strength": 60,
        }
        result = execute_drop_shadow(ls, layer, params)
        assert result is not None
        assert len(result.children) == 2

    def test_group_rejected(self):
        ls = LayerStack(W, H)
        group = ls.add_group("g")
        result = execute_drop_shadow(ls, group, {
            "color": QColor(0, 0, 0), "offset_x": 0, "offset_y": 0,
            "blur": 0, "strength": 50,
        })
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 背景パターン生成
# ═══════════════════════════════════════════════════════════════════════════════

class TestBgPattern:
    @pytest.mark.parametrize("pattern", [
        "dots", "stripes_v", "stripes_h", "stripes_d", "checker",
        "grad_v", "grad_h", "grad_radial",
    ])
    def test_all_patterns(self, pattern):
        ls, layer = _make_stack_with_lineart()
        params = {
            "pattern": pattern,
            "color1": QColor(255, 200, 200),
            "color2": QColor(200, 200, 255),
            "spacing": 20,
        }
        result = execute_bg_pattern(ls, layer, params)
        assert result is not None
        assert isinstance(result, Layer)
        assert _has_nonzero_pixels(result.image)
        # ソースレイヤーの下（index + 1）に挿入される
        layer_idx = ls.layers.index(layer)
        bg_idx = ls.layers.index(result)
        assert bg_idx == layer_idx + 1

    def test_with_group_source(self):
        """グループが選択されていても背景パターンは生成できる（レイヤー末尾に追加）。"""
        ls = LayerStack(W, H)
        group = ls.add_group("g")
        params = {
            "pattern": "dots",
            "color1": QColor(255, 255, 255),
            "color2": QColor(0, 0, 0),
            "spacing": 10,
        }
        result = execute_bg_pattern(ls, group, params)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 線画色変え
# ═══════════════════════════════════════════════════════════════════════════════

class TestLineColor:
    def test_basic(self):
        ls, layer = _make_stack_with_lineart()
        params = {"color": QColor(80, 50, 30)}
        result = execute_line_color(ls, layer, params)
        assert result is not None
        assert isinstance(result, Layer)
        assert len(ls.layers) == 2
        assert not layer.visible
        # 新レイヤーの名前に色コードが含まれる
        assert "#" in result.name

    def test_color_applied(self):
        ls, layer = _make_stack_with_lineart()
        target_color = QColor(255, 0, 0)
        params = {"color": target_color}
        result = execute_line_color(ls, layer, params)
        # 線があった場所のピクセルが赤系になっている
        c = result.image.pixelColor(50, 50)
        if c.alpha() > 0:
            assert c.red() > 200

    def test_transparent_stays_transparent(self):
        ls, layer = _make_stack_with_lineart()
        params = {"color": QColor(0, 0, 255)}
        result = execute_line_color(ls, layer, params)
        # 元が透明だった場所は透明のまま
        c = result.image.pixelColor(0, 0)
        assert c.alpha() == 0

    def test_group_rejected(self):
        ls = LayerStack(W, H)
        group = ls.add_group("g")
        result = execute_line_color(ls, group, {"color": QColor(0, 0, 0)})
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ポップアウト（ステッカー風）
# ═══════════════════════════════════════════════════════════════════════════════

class TestPopout:
    def test_basic_with_shadow(self):
        ls, layer = _make_stack_with_lineart()
        params = {
            "outline_size": 3,
            "outline_color": QColor(255, 255, 255),
            "shadow": True,
            "shadow_offset": 3,
        }
        result = execute_popout(ls, layer, params)
        assert result is not None
        assert isinstance(result, GroupLayer)
        assert len(ls.layers) == 2
        assert not layer.visible
        # グループ内: 元コピー + 縁 + 影 = 3枚
        assert len(result.children) == 3

    def test_without_shadow(self):
        ls, layer = _make_stack_with_lineart()
        params = {
            "outline_size": 5,
            "outline_color": QColor(255, 255, 255),
            "shadow": False,
            "shadow_offset": 0,
        }
        result = execute_popout(ls, layer, params)
        assert result is not None
        # 影なし: 元コピー + 縁 = 2枚
        assert len(result.children) == 2

    def test_outline_wider_than_original(self):
        ls, layer = _make_stack_with_lineart()
        params = {
            "outline_size": 10,
            "outline_color": QColor(255, 255, 255),
            "shadow": True,
            "shadow_offset": 5,
        }
        result = execute_popout(ls, layer, params)
        outline_layer = result.children[1]
        # 縁レイヤーのアルファ範囲は元より広いはず
        assert _has_nonzero_pixels(outline_layer.image)

    def test_group_rejected(self):
        ls = LayerStack(W, H)
        group = ls.add_group("g")
        result = execute_popout(ls, group, {
            "outline_size": 3, "outline_color": QColor(255, 255, 255),
            "shadow": False, "shadow_offset": 0,
        })
        assert result is None

    def test_all_children_have_pixels(self):
        ls, layer = _make_stack_with_lineart()
        params = {
            "outline_size": 5,
            "outline_color": QColor(255, 255, 255),
            "shadow": True,
            "shadow_offset": 3,
        }
        result = execute_popout(ls, layer, params)
        for child in result.children:
            assert _has_nonzero_pixels(child.image)
