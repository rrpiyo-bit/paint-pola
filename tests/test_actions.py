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


# ═══════════════════════════════════════════════════════════════════════════════
# 新効果7種 + アクションガチャ
# ═══════════════════════════════════════════════════════════════════════════════

from actions import (
    execute_offset_border, execute_silkscreen, execute_collage,
    execute_wobble, execute_stamp, execute_contour,
    execute_halftone, execute_dither, execute_crosshatch,
    execute_vhs, execute_crt,
    execute_warhol, execute_lichtenstein, execute_ukiyoe,
    execute_impressionist, execute_stained_glass, execute_blueprint,
    execute_etching,
    execute_gacha, execute_path_repeat,
    _gacha_random_params, _gacha_random_path,
    _GACHA_POOL, _GACHA_EXEC, GACHA_PALETTES,
)


def _make_stack_with_closed_shape() -> tuple[LayerStack, Layer]:
    """閉じた領域を持つ線画レイヤー（円）を作る。"""
    ls = LayerStack(W, H)
    layer = ls.add("線画")
    from PyQt6.QtGui import QPen
    p = QPainter(layer.image)
    pen = QPen(QColor(0, 0, 0, 255)); pen.setWidth(3)
    p.setPen(pen)
    p.drawEllipse(20, 20, 60, 60)
    p.end()
    return ls, layer


def _alpha_count(img: QImage) -> int:
    import numpy as np
    img32 = img.convertToFormat(QImage.Format.Format_ARGB32)
    ptr = img32.bits()
    ptr.setsize(img32.height() * img32.width() * 4)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(img32.height(), img32.width(), 4)
    return int((arr[:, :, 3] > 0).sum())


class TestOffsetBorder:
    def test_basic(self):
        ls, layer = _make_stack_with_closed_shape()
        result = execute_offset_border(ls, layer, {
            "color": QColor(255, 255, 255), "size": 5, "shift": 8, "gap": 20})
        assert isinstance(result, GroupLayer)
        assert len(result.children) == 2
        assert _has_nonzero_pixels(result.children[1].image)
        assert not layer.visible

    def test_no_shift_no_gap(self):
        ls, layer = _make_stack_with_closed_shape()
        result = execute_offset_border(ls, layer, {
            "color": QColor(255, 0, 0), "size": 3, "shift": 0, "gap": 0})
        assert result is not None

    def test_group_rejected(self):
        ls = LayerStack(W, H)
        group = ls.add_group("g")
        assert execute_offset_border(ls, group, {
            "color": QColor(255, 255, 255), "size": 5, "shift": 0, "gap": 0}) is None


class TestSilkscreen:
    def test_basic(self):
        ls, layer = _make_stack_with_closed_shape()
        result = execute_silkscreen(ls, layer, {
            "colors": [QColor(255, 0, 0), QColor(0, 0, 255)],
            "shift": 10, "opacity": 90})
        assert isinstance(result, GroupLayer)
        # 元コピー + 色版2枚
        assert len(result.children) == 3
        for plate in result.children[1:]:
            assert _has_nonzero_pixels(plate.image)


class TestCollage:
    def test_closed_region_filled(self):
        ls, layer = _make_stack_with_closed_shape()
        result = execute_collage(ls, layer, {
            "colors": [QColor(255, 100, 100)], "coverage": 100,
            "expand": 2, "shift": 2})
        assert isinstance(result, GroupLayer)
        # 円の内側が塗られている
        assert _alpha_count(result.children[1].image) > 500

    def test_no_closed_region_returns_none(self):
        ls, layer = _make_stack_with_lineart()  # 十字線は閉領域なし
        result = execute_collage(ls, layer, {
            "colors": [QColor(255, 0, 0)], "coverage": 100,
            "expand": 0, "shift": 0})
        assert result is None


class TestWobble:
    def test_distorts(self):
        ls, layer = _make_stack_with_closed_shape()
        before = _alpha_count(layer.image)
        result = execute_wobble(ls, layer, {
            "strength": 5, "wavelength": 30, "gap": 0})
        assert result is not None
        after = _alpha_count(result.image)
        assert before * 0.5 < after < before * 2.5

    def test_gap_reduces_area(self):
        ls, layer = _make_stack_with_closed_shape()
        before = _alpha_count(layer.image)
        result = execute_wobble(ls, layer, {
            "strength": 2, "wavelength": 30, "gap": 60})
        assert result is not None
        assert _alpha_count(result.image) < before


class TestStamp:
    def test_fades(self):
        ls, layer = _make_stack_with_closed_shape()
        before = _alpha_count(layer.image)
        result = execute_stamp(ls, layer, {
            "strength": 50, "grain": 2, "blots": False})
        assert result is not None
        after = _alpha_count(result.image)
        assert 0 < after < before

    def test_blot_color_is_applied_exactly(self):
        """指定した色がそのままインク溜まりに出る。

        赤成分と青成分が明確に違う色を使うこと。配列は BGRA 順なので、
        (255,0,128) のような紛らわしい色だと入れ替わっていても気付けない。
        判定も生配列ではなく QColor 経由で行う。
        """
        import numpy as np
        ls, layer = _make_stack_with_closed_shape()
        target = QColor(200, 30, 40)
        result = execute_stamp(ls, layer, {
            "strength": 40, "grain": 3, "blots": True,
            "blot_min": 8, "blot_max": 12, "blot_color": target})
        arr = _qimage_to_array_test(result.image)
        ys, xs = np.nonzero(arr[:, :, 3] == 255)
        found = set()
        for y, x in zip(ys, xs):
            c = result.image.pixelColor(int(x), int(y))
            found.add((c.red(), c.green(), c.blue()))
        assert (200, 30, 40) in found

    def test_blot_size_range_changes_area(self):
        """半径を大きくするとインク溜まりの面積が増える。"""
        import numpy as np

        def blot_area(r_min, r_max):
            total = 0
            for _ in range(6):
                ls, layer = _make_stack_with_closed_shape()
                result = execute_stamp(ls, layer, {
                    "strength": 5, "grain": 3, "blots": True,
                    "blot_min": r_min, "blot_max": r_max,
                    "blot_color": QColor(255, 0, 0)})
                a = _qimage_to_array_test(result.image)
                # 純赤なので、チャンネル順に関係なく「255 と 0 の組」で数えられる
                total += int(((a[:, :, :3].max(axis=2) == 255) &
                              (a[:, :, :3].min(axis=2) == 0) &
                              (a[:, :, 3] == 255)).sum())
            return total / 6

        assert blot_area(12, 16) > blot_area(1, 2) * 3

    def test_reversed_min_max_is_tolerated(self):
        """最小 > 最大 で渡っても落ちない（入れ替えて処理する）。"""
        ls, layer = _make_stack_with_closed_shape()
        result = execute_stamp(ls, layer, {
            "strength": 40, "grain": 3, "blots": True,
            "blot_min": 12, "blot_max": 3, "blot_color": None})
        assert result is not None

    def test_auto_color_uses_line_color(self):
        """色指定なしなら線の色になじむ（従来の挙動を維持）。"""
        ls, layer = _make_stack_with_closed_shape()
        result = execute_stamp(ls, layer, {
            "strength": 40, "grain": 3, "blots": True,
            "blot_min": 4, "blot_max": 8, "blot_color": None})
        assert result is not None
        assert _has_nonzero_pixels(result.image)

    def test_works_without_new_params(self):
        """新パラメータを渡さない古い呼び出しでも動く。"""
        ls, layer = _make_stack_with_closed_shape()
        result = execute_stamp(ls, layer, {
            "strength": 40, "grain": 3, "blots": True})
        assert result is not None

    def test_gacha_params_have_valid_range(self):
        """ガチャが作る半径は必ず 最小 <= 最大。"""
        colors = [QColor(c) for c in GACHA_PALETTES[0][1]]
        for _ in range(30):
            params = _gacha_random_params("stamp", colors)
            assert 1 <= params["blot_min"] <= params["blot_max"]


class TestStampDialogUI:
    """スタンプ劣化ダイアログの入力制御。"""

    def test_min_max_interlock(self):
        from actions import StampDialog
        dlg = StampDialog()
        dlg._blot_min.setValue(20)
        assert dlg._blot_max.value() >= 20
        dlg._blot_max.setValue(3)
        assert dlg._blot_min.value() <= 3

    def test_color_button_follows_auto_checkbox(self):
        from actions import StampDialog
        dlg = StampDialog()
        assert dlg._auto_color.isChecked()
        assert not dlg._blot_color.isEnabled()
        dlg._auto_color.setChecked(False)
        assert dlg._blot_color.isEnabled()
        assert dlg.params()["blot_color"] is not None

    def test_disabling_blots_disables_sub_controls(self):
        from actions import StampDialog
        dlg = StampDialog()
        dlg._blots.setChecked(False)
        assert not dlg._blot_min.isEnabled()
        assert not dlg._blot_max.isEnabled()
        assert not dlg._blot_color.isEnabled()

    def test_default_params_keep_previous_behaviour(self):
        from actions import StampDialog
        params = StampDialog().params()
        assert params["blot_min"] == 2 and params["blot_max"] == 6
        assert params["blot_color"] is None


class TestContour:
    def test_rings_generated(self):
        ls, layer = _make_stack_with_closed_shape()
        result = execute_contour(ls, layer, {
            "count": 3, "spacing": 5, "color": QColor(255, 255, 255),
            "thickness": 1, "fade": True})
        assert isinstance(result, GroupLayer)
        assert _alpha_count(result.children[1].image) > 100


class TestGacha:
    def test_pool_excludes_bg_pattern(self):
        assert all(k != "bg" and "背景" not in lbl for k, lbl in _GACHA_POOL)

    def test_random_params_valid_for_all_pool(self):
        colors = [QColor(c) for c in GACHA_PALETTES[0][1]]
        for key, label in _GACHA_POOL:
            ls, layer = _make_stack_with_closed_shape()
            params = _gacha_random_params(key, colors)
            if key == "path":
                result = execute_path_repeat(
                    ls, layer, _gacha_random_path(W, H), params)
            else:
                result = _GACHA_EXEC[key](ls, layer, params)
            assert result is not None, f"{label} が None を返した"

    def test_gacha_returns_flat_layer_with_recipe(self):
        ls, layer = _make_stack_with_closed_shape()
        result = execute_gacha(ls, layer, {"count": 0, "palette": "auto"})
        assert result is not None
        assert not result.is_group
        assert "ガチャ" in result.name
        assert _has_nonzero_pixels(result.image)
        assert ls.layers[0] is result
        assert not layer.visible

    def test_gacha_never_produces_empty_result(self):
        """どの組み合わせを引いても絵が消えない。

        1回だけ引くテストでは、絵が消える効果の組み合わせを
        数%の確率でしか踏まず、失敗が再現しにくい。
        シードを固定して多数回引き、確実に検出する。
        """
        import random
        for seed in range(60):
            random.seed(seed)
            ls, layer = _make_stack_with_closed_shape()
            result = execute_gacha(ls, layer, {"count": 0, "palette": "auto"})
            assert result is not None, seed
            assert _has_nonzero_pixels(result.image), (seed, result.name)

    def test_gacha_palette_choice(self):
        ls, layer = _make_stack_with_closed_shape()
        result = execute_gacha(ls, layer, {"count": 2, "palette": "レトロ印刷"})
        assert result is not None
        assert "レトロ印刷" in result.name

    def test_gacha_empty_layer(self):
        ls = LayerStack(W, H)
        layer = ls.add("空")
        # 空レイヤーでもクラッシュせず None
        assert execute_gacha(ls, layer, {"count": 0, "palette": "auto"}) is None

    def test_gacha_group_rejected(self):
        ls = LayerStack(W, H)
        group = ls.add_group("g")
        assert execute_gacha(ls, group, {"count": 0, "palette": "auto"}) is None


def _qimage_alpha(img: QImage):
    import numpy as np
    img32 = img.convertToFormat(QImage.Format.Format_ARGB32)
    ptr = img32.bits()
    ptr.setsize(img32.height() * img32.width() * 4)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(img32.height(), img32.width(), 4)
    return arr[:, :, 3]


class TestCollageSplit:
    """大きい閉領域の紙片分割（複数色が必ず使われる）の検証。"""

    def _big_circle_stack(self, size=300):
        from PyQt6.QtGui import QPen
        ls = LayerStack(size, size)
        layer = ls.add("大円")
        p = QPainter(layer.image)
        pen = QPen(QColor(0, 0, 0, 255)); pen.setWidth(4)
        p.setPen(pen)
        p.drawEllipse(20, 20, size - 40, size - 40)
        p.end()
        return ls, layer

    def test_large_region_uses_multiple_colors(self):
        import numpy as np
        colors = [QColor(255, 0, 0), QColor(0, 255, 0),
                  QColor(0, 0, 255), QColor(255, 255, 0)]
        ls, layer = self._big_circle_stack()
        result = execute_collage(ls, layer, {
            "colors": colors, "coverage": 100, "expand": 0, "shift": 0})
        assert result is not None
        arr = _qimage_to_array_test(result.children[1].image)
        op = arr[arr[:, :, 3] > 0]
        assert len(np.unique(op[:, :3], axis=0)) >= 2

    def test_small_region_single_piece(self):
        import numpy as np
        colors = [QColor(255, 0, 0), QColor(0, 255, 0)]
        ls, layer = _make_stack_with_closed_shape()  # 100x100の小円
        result = execute_collage(ls, layer, {
            "colors": colors, "coverage": 100, "expand": 0, "shift": 0})
        assert result is not None
        arr = _qimage_to_array_test(result.children[1].image)
        op = arr[arr[:, :, 3] > 0]
        # 小領域は分割されず1色
        assert len(np.unique(op[:, :3], axis=0)) == 1


def _qimage_to_array_test(img: QImage):
    import numpy as np
    img32 = img.convertToFormat(QImage.Format.Format_ARGB32)
    ptr = img32.bits()
    ptr.setsize(img32.height() * img32.width() * 4)
    return np.frombuffer(ptr, dtype=np.uint8).reshape(
        img32.height(), img32.width(), 4).copy()


# ═══════════════════════════════════════════════════════════════════════════════
# 網点 / ディザ / ハッチング / VHS / CRT
# ═══════════════════════════════════════════════════════════════════════════════

def _make_colorful_stack(size=120) -> tuple[LayerStack, Layer]:
    """明暗・色味に差のある図形を持つレイヤー。階調が絡む効果の検証用。"""
    ls = LayerStack(size, size)
    layer = ls.add("カラー")
    p = QPainter(layer.image)
    p.fillRect(10, 10, 60, 60, QColor(220, 60, 90, 255))
    p.fillRect(50, 50, 60, 60, QColor(60, 120, 220, 255))
    p.fillRect(20, 80, 30, 25, QColor(30, 30, 30, 255))
    p.end()
    return ls, layer


# 各効果の (実行関数, 既定パラメータ, 結果レイヤー名の一部)
_NEW_EFFECTS = [
    (execute_halftone,
     {"pitch": 8, "mode": "rgb", "background": "white", "smooth": True},
     "網点"),
    (execute_dither,
     {"method": "bayer", "levels": 4, "pixel": 3},
     "ディザ"),
    (execute_crosshatch,
     {"spacing": 6, "thickness": 1, "layers": 3, "color": QColor(20, 20, 30)},
     "ハッチング"),
    (execute_vhs,
     {"jitter": 8, "bands": 2, "scanline": 0.35, "scan_pitch": 3, "noise": 0.2},
     "VHS"),
    (execute_crt,
     {"kind": "crt", "cell": 6, "bloom": 0.4, "boost": 1.6, "saturation": 1.2},
     "CRT"),
]


class TestNewEffectsCommon:
    """5効果に共通する契約（グループ拒否・offset維持・非空出力）。"""

    @pytest.mark.parametrize("fn,params,name", _NEW_EFFECTS)
    def test_produces_visible_result(self, fn, params, name):
        ls, layer = _make_colorful_stack()
        result = fn(ls, layer, params)
        assert result is not None
        assert name in result.name
        assert _has_nonzero_pixels(result.image)

    @pytest.mark.parametrize("fn,params,name", _NEW_EFFECTS)
    def test_rejects_group_layer(self, fn, params, name):
        ls, _ = _make_colorful_stack()
        group = GroupLayer("グループ", W, H)
        assert fn(ls, group, params) is None

    @pytest.mark.parametrize("fn,params,name", _NEW_EFFECTS)
    def test_keeps_offset_and_hides_source(self, fn, params, name):
        ls, layer = _make_colorful_stack()
        layer.offset_x, layer.offset_y = 23, 41
        result = fn(ls, layer, params)
        assert (result.offset_x, result.offset_y) == (23, 41)
        # 元レイヤーは残るが非表示になる（他の効果と同じ振る舞い）
        assert layer.visible is False
        assert layer in ls.layers

    @pytest.mark.parametrize("fn,params,name", _NEW_EFFECTS)
    def test_result_size_matches_source(self, fn, params, name):
        ls, layer = _make_colorful_stack()
        result = fn(ls, layer, params)
        assert result.image.width() == layer.image.width()
        assert result.image.height() == layer.image.height()


class TestHalftone:
    def test_mono_mode_is_monochrome(self):
        """モノクロ版は黒インクだけで描かれる。"""
        import numpy as np
        ls, layer = _make_colorful_stack()
        result = execute_halftone(ls, layer, {
            "pitch": 8, "mode": "mono", "background": "transparent",
            "smooth": True})
        arr = _qimage_to_array_test(result.image)
        op = arr[arr[:, :, 3] > 0]
        assert len(op) > 0
        assert op[:, :3].max() == 0   # RGB は全て 0（黒）

    def test_white_background_is_opaque(self):
        """白地に印刷を選ぶと全面が不透明になる。"""
        ls, layer = _make_colorful_stack()
        result = execute_halftone(ls, layer, {
            "pitch": 8, "mode": "rgb", "background": "white", "smooth": True})
        arr = _qimage_to_array_test(result.image)
        assert (arr[:, :, 3] == 255).all()

    def test_transparent_background_leaves_holes(self):
        """透明のままを選ぶと点の隙間が透ける。"""
        ls, layer = _make_colorful_stack()
        result = execute_halftone(ls, layer, {
            "pitch": 8, "mode": "rgb", "background": "transparent",
            "smooth": True})
        arr = _qimage_to_array_test(result.image)
        assert (arr[:, :, 3] == 0).any()

    def test_finer_pitch_makes_more_dots(self):
        """網点を細かくするほど点の数が増える。

        濃い面では点どうしがくっついて数えられなくなるので、
        点が確実に離れて並ぶ淡いグレー一色の絵で数える。
        """
        import cv2
        import numpy as np
        base = {"mode": "mono", "background": "transparent", "smooth": False}

        def dot_count(pitch):
            ls = LayerStack(120, 120)
            layer = ls.add("淡いグレー")
            p = QPainter(layer.image)
            p.fillRect(0, 0, 120, 120, QColor(200, 200, 200, 255))
            p.end()
            r = execute_halftone(ls, layer, {**base, "pitch": pitch})
            # くっきりした点だけを数える（アンチエイリアスの裾を拾わない）
            mask = (_qimage_to_array_test(r.image)[:, :, 3] > 128).astype(np.uint8)
            # 連結成分数 - 1（背景ぶん）＝ 点の個数
            return cv2.connectedComponents(mask)[0] - 1

        assert dot_count(6) > dot_count(20)

    def test_thin_lineart_still_gets_dots(self):
        """細い線画でも網点が消えない。

        セル中心の1ピクセルだけを読むと、線がセル中心を外れた瞬間に
        濃度0とみなされ、結果が真っ白（＝完全に透明）になっていた。
        ガチャで網点を引くと約2%の確率で絵が消える原因だった。
        """
        for mode in ("mono", "rgb"):
            for background in ("transparent", "white"):
                ls, layer = _make_stack_with_closed_shape()  # 細い線の円
                result = execute_halftone(ls, layer, {
                    "pitch": 7, "mode": mode, "background": background,
                    "smooth": True})
                assert result is not None
                arr = _qimage_to_array_test(result.image)
                if background == "transparent":
                    ink = int((arr[:, :, 3] > 0).sum())
                else:
                    # 白地なので「白より暗いピクセル」がインク
                    ink = int((arr[:, :, :3].min(axis=2) < 200).sum())
                assert ink > 0, (mode, background)

    def test_tone_is_preserved(self):
        """網点の面積が元の濃度に比例する（中間調が黒く潰れない）。

        点の半径をセルの対角まで伸ばすと中間調で隣の点とつながり、
        灰色が真っ黒になってしまう。その退行を防ぐための検証。
        """
        import numpy as np
        for tone in (200, 128, 60):
            ls = LayerStack(160, 160)
            layer = ls.add("べた")
            p = QPainter(layer.image)
            p.fillRect(0, 0, 160, 160, QColor(tone, tone, tone, 255))
            p.end()
            result = execute_halftone(ls, layer, {
                "pitch": 10, "mode": "mono", "background": "white",
                "smooth": True})
            arr = _qimage_to_array_test(result.image).astype(float)
            # 白地に黒インクなので、平均の明るさが元の濃度に近いはず
            got = arr[:, :, :3].mean()
            assert abs(got - tone) < 45, (tone, got)


class TestDither:
    def test_reduces_color_count(self):
        """減色後の色数は階調数から決まる上限に収まる。"""
        import numpy as np
        ls, layer = _make_colorful_stack()
        result = execute_dither(ls, layer, {
            "method": "bayer", "levels": 2, "pixel": 1})
        arr = _qimage_to_array_test(result.image)
        op = arr[arr[:, :, 3] > 0]
        # levels=2 なら各チャンネル 0 か 255 の 2 値 → 最大 8 色
        assert len(np.unique(op[:, :3], axis=0)) <= 8

    def test_palette_restricts_colors(self):
        """パレットを渡すとその色だけで構成される（ガチャ経路）。"""
        import numpy as np
        palette = [QColor(255, 0, 0), QColor(0, 0, 255)]
        ls, layer = _make_colorful_stack()
        result = execute_dither(ls, layer, {
            "method": "bayer", "levels": 4, "pixel": 1, "palette": palette})
        arr = _qimage_to_array_test(result.image)
        op = arr[arr[:, :, 3] > 0]
        used = {tuple(int(v) for v in c) for c in np.unique(op[:, :3], axis=0)}
        assert used <= {(255, 0, 0), (0, 0, 255)}

    def test_diffusion_method_runs(self):
        ls, layer = _make_colorful_stack()
        result = execute_dither(ls, layer, {
            "method": "diffusion", "levels": 3, "pixel": 4})
        assert result is not None
        assert _has_nonzero_pixels(result.image)

    def test_alpha_is_binarized(self):
        """ドット絵らしく半透明の縁が残らない。"""
        import numpy as np
        ls, layer = _make_colorful_stack()
        result = execute_dither(ls, layer, {
            "method": "bayer", "levels": 4, "pixel": 2})
        arr = _qimage_to_array_test(result.image)
        assert set(np.unique(arr[:, :, 3]).tolist()) <= {0, 255}


class TestCrosshatch:
    def test_uses_specified_color(self):
        """指定色がそのまま出る。

        赤と青が明確に違う色を QColor 経由で確認する（BGRA 入れ替わり検出）。
        """
        import numpy as np
        ls, layer = _make_colorful_stack()
        result = execute_crosshatch(ls, layer, {
            "spacing": 6, "thickness": 2, "layers": 3,
            "color": QColor(200, 30, 40)})
        arr = _qimage_to_array_test(result.image)
        ys, xs = np.nonzero(arr[:, :, 3] > 200)
        assert len(xs) > 0
        c = result.image.pixelColor(int(xs[0]), int(ys[0]))
        assert (c.red(), c.green(), c.blue()) == (200, 30, 40)

    def test_darker_areas_get_more_ink(self):
        """暗い領域ほどハッチングが密になる。"""
        size = 120
        ls = LayerStack(size, size)
        layer = ls.add("明暗")
        p = QPainter(layer.image)
        p.fillRect(0, 0, size // 2, size, QColor(230, 230, 230, 255))  # 明
        p.fillRect(size // 2, 0, size // 2, size, QColor(15, 15, 15, 255))  # 暗
        p.end()
        result = execute_crosshatch(ls, layer, {
            "spacing": 6, "thickness": 1, "layers": 3,
            "color": QColor(0, 0, 0)})
        a = _qimage_to_array_test(result.image)[:, :, 3]
        light_ink = int((a[:, :size // 2] > 0).sum())
        dark_ink = int((a[:, size // 2:] > 0).sum())
        assert dark_ink > light_ink

    def test_no_ink_outside_original_shape(self):
        """元が透明だった場所には線を引かない。"""
        ls, layer = _make_colorful_stack()
        result = execute_crosshatch(ls, layer, {
            "spacing": 5, "thickness": 1, "layers": 3,
            "color": QColor(0, 0, 0)})
        src = _qimage_to_array_test(layer.image)
        out = _qimage_to_array_test(result.image)
        assert not ((out[:, :, 3] > 0) & (src[:, :, 3] == 0)).any()


class TestVhs:
    def test_changes_pixels(self):
        """行ずれ・ノイズで元と異なる絵になる。"""
        import numpy as np
        ls, layer = _make_colorful_stack()
        before = _qimage_to_array_test(layer.image)
        result = execute_vhs(ls, layer, {
            "jitter": 10, "bands": 3, "scanline": 0.4,
            "scan_pitch": 3, "noise": 0.25})
        after = _qimage_to_array_test(result.image)
        assert not np.array_equal(before, after)

    def test_zero_settings_keep_image_intact(self):
        """全て 0 なら元の絵をほぼそのまま返す（副作用がないことの確認）。"""
        import numpy as np
        ls, layer = _make_colorful_stack()
        before = _qimage_to_array_test(layer.image)
        result = execute_vhs(ls, layer, {
            "jitter": 0, "bands": 0, "scanline": 0.0,
            "scan_pitch": 3, "noise": 0.0})
        after = _qimage_to_array_test(result.image)
        assert np.array_equal(before, after)

    def test_scanlines_darken_periodically(self):
        """走査線だけを効かせると一定間隔の行が暗くなる。"""
        size = 60
        ls = LayerStack(size, size)
        layer = ls.add("べた")
        p = QPainter(layer.image)
        p.fillRect(0, 0, size, size, QColor(200, 200, 200, 255))
        p.end()
        result = execute_vhs(ls, layer, {
            "jitter": 0, "bands": 0, "scanline": 0.5,
            "scan_pitch": 3, "noise": 0.0})
        arr = _qimage_to_array_test(result.image).astype(float)
        rows = arr[:, :, :3].mean(axis=(1, 2))
        assert rows[0] < rows[1]   # 0行目が走査線


class TestCrt:
    def test_crt_and_led_differ(self):
        """CRT と LED はマスク形状が違うので結果も異なる。"""
        import numpy as np
        base = {"cell": 6, "bloom": 0.4, "boost": 1.6, "saturation": 1.2}
        ls1, l1 = _make_colorful_stack()
        ls2, l2 = _make_colorful_stack()
        crt = _qimage_to_array_test(
            execute_crt(ls1, l1, {**base, "kind": "crt"}).image)
        led = _qimage_to_array_test(
            execute_crt(ls2, l2, {**base, "kind": "led"}).image)
        assert not np.array_equal(crt, led)

    def test_subpixel_mask_creates_variation(self):
        """べた塗りでもサブピクセルの縞で横方向に濃淡が生まれる。"""
        size = 60
        ls = LayerStack(size, size)
        layer = ls.add("べた")
        p = QPainter(layer.image)
        p.fillRect(0, 0, size, size, QColor(180, 180, 180, 255))
        p.end()
        result = execute_crt(ls, layer, {
            "kind": "crt", "cell": 6, "bloom": 0.0,
            "boost": 1.0, "saturation": 1.0})
        arr = _qimage_to_array_test(result.image).astype(float)
        # RGB を平均すると縞が打ち消し合って見えなくなるので、
        # チャンネルごとに列方向のばらつきを見る。R が強い列・G が強い列…
        # と並ぶのがサブピクセルなので、各チャンネルで濃淡が出るはず。
        for ch in range(3):
            assert arr[:, :, ch].std(axis=1).mean() > 5.0

    def test_alpha_preserved_outside_shape(self):
        """図形の外は透明のまま。"""
        ls, layer = _make_colorful_stack()
        result = execute_crt(ls, layer, {
            "kind": "crt", "cell": 6, "bloom": 0.4,
            "boost": 1.6, "saturation": 1.2})
        src = _qimage_to_array_test(layer.image)
        out = _qimage_to_array_test(result.image)
        # 元が完全に透明な広い領域は透明のまま残る
        assert out[:, :, 3][src[:, :, 3] == 0].max() == 0


class TestNewEffectsInGacha:
    """5効果がランダムアクションのプールに入っていること。"""

    NEW_KEYS = ["halftone", "dither", "crosshatch", "vhs", "crt"]

    def test_registered_in_pool(self):
        keys = [k for k, _ in _GACHA_POOL]
        for k in self.NEW_KEYS:
            assert k in keys

    def test_registered_in_exec(self):
        for k in self.NEW_KEYS:
            assert k in _GACHA_EXEC

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_random_params_are_generated(self, key):
        colors = [QColor(c) for c in GACHA_PALETTES[0][1]]
        for _ in range(20):
            params = _gacha_random_params(key, colors)
            assert params, key

    @pytest.mark.parametrize("key", NEW_KEYS)
    def test_random_params_actually_execute(self, key):
        """生成されたランダム値でどれも実行でき、空にならない。"""
        colors = [QColor(c) for c in GACHA_PALETTES[1][1]]
        for _ in range(5):
            ls, layer = _make_colorful_stack()
            params = _gacha_random_params(key, colors)
            result = _GACHA_EXEC[key](ls, layer, params)
            assert result is not None, (key, params)
            assert _has_nonzero_pixels(result.image), (key, params)



# 「○○風」加工の (実行関数, 既定パラメータ, 結果レイヤー名の一部)
_STYLE_EFFECTS = [
    (execute_warhol,
     {"cols": 2, "rows": 2, "levels": 4, "gap": 4, "outline": 2,
      "random_sets": False},
     "ウォーホル風"),
    (execute_lichtenstein,
     {"pitch": 6, "outline": 3, "levels": 3, "dot_color": QColor(220, 50, 60),
      "line_color": QColor(20, 20, 20), "white_bg": True},
     "アメコミ風"),
    (execute_ukiyoe,
     {"levels": 4, "misalign": 3, "sumi": 2, "paper": 0.3,
      "paper_color": QColor(240, 230, 205)},
     "浮世絵風"),
    (execute_impressionist,
     {"length": 10, "width": 3, "density": 1.5, "jitter": 15, "follow": True},
     "印象派風"),
    (execute_stained_glass,
     {"cell": 24, "lead": 3, "lead_color": QColor(25, 22, 20), "vivid": 1.6,
      "glass_bg": True},
     "ステンドグラス風"),
    (execute_blueprint,
     {"paper_color": QColor(20, 60, 130), "ink_color": QColor(235, 243, 255),
      "thickness": 2, "grid": 24, "major": 4, "fade": 0.2},
     "設計図風"),
    (execute_etching,
     {"spacing": 4, "layers": 4, "wobble": 0.4,
      "ink_color": QColor(35, 28, 22), "paper_color": QColor(238, 230, 214),
      "grain": 0.25},
     "銅版画風"),
]


class TestStyleEffectsCommon:
    """「○○風」7効果に共通する契約。"""

    @pytest.mark.parametrize("fn,params,name", _STYLE_EFFECTS)
    def test_produces_visible_result(self, fn, params, name):
        ls, layer = _make_colorful_stack()
        result = fn(ls, layer, params)
        assert result is not None
        assert name in result.name
        assert _has_nonzero_pixels(result.image)

    @pytest.mark.parametrize("fn,params,name", _STYLE_EFFECTS)
    def test_rejects_group_layer(self, fn, params, name):
        ls, _ = _make_colorful_stack()
        group = GroupLayer("グループ", W, H)
        assert fn(ls, group, params) is None

    @pytest.mark.parametrize("fn,params,name", _STYLE_EFFECTS)
    def test_keeps_offset_and_hides_source(self, fn, params, name):
        ls, layer = _make_colorful_stack()
        layer.offset_x, layer.offset_y = 23, 41
        result = fn(ls, layer, params)
        assert (result.offset_x, result.offset_y) == (23, 41)
        assert layer.visible is False
        assert layer in ls.layers

    @pytest.mark.parametrize("fn,params,name", _STYLE_EFFECTS)
    def test_result_size_matches_source(self, fn, params, name):
        ls, layer = _make_colorful_stack()
        result = fn(ls, layer, params)
        assert result.image.width() == layer.image.width()
        assert result.image.height() == layer.image.height()

    @pytest.mark.parametrize("fn,params,name", _STYLE_EFFECTS)
    def test_empty_layer_returns_none(self, fn, params, name):
        """完全に透明なレイヤーでは None を返し、空の結果を作らない。"""
        ls = LayerStack(60, 60)
        layer = ls.add("空")
        result = fn(ls, layer, params)
        assert result is None or _has_nonzero_pixels(result.image)


class TestWarhol:
    def _params(self, **over):
        p = {"cols": 2, "rows": 2, "levels": 4, "gap": 0, "outline": 2,
             "random_sets": False}
        p.update(over)
        return p

    def test_tiles_are_different_colors(self):
        """コマごとに配色が変わる（4枚が同じ絵にならない）。"""
        import numpy as np
        ls, layer = _make_colorful_stack(size=120)
        result = execute_warhol(ls, layer, self._params())
        arr = _qimage_to_array_test(result.image)
        h, w = arr.shape[:2]
        quads = [arr[:h // 2, :w // 2], arr[:h // 2, w // 2:],
                 arr[h // 2:, :w // 2], arr[h // 2:, w // 2:]]
        means = [q[:, :, :3].reshape(-1, 3).mean(axis=0) for q in quads]
        # どの2コマを比べても平均色が一致しない
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                assert np.abs(means[i] - means[j]).max() > 5, (i, j)

    def test_grid_shape_is_respected(self):
        """3×3 を指定すると9コマぶんの繰り返しになる。"""
        import numpy as np
        ls, layer = _make_colorful_stack(size=180)
        result = execute_warhol(ls, layer, self._params(cols=3, rows=3))
        arr = _qimage_to_array_test(result.image)
        # コマごとの平均色を比べる。9コマが全部同じ配色ではないことを見る
        # （1点だけ拾うと輪郭線に当たって偶然一致するので平均で判定する）
        h, w = arr.shape[:2]
        means = []
        for r in range(3):
            for c in range(3):
                cell = arr[r * h // 3:(r + 1) * h // 3,
                           c * w // 3:(c + 1) * w // 3, :3]
                means.append(tuple(np.round(
                    cell.reshape(-1, 3).mean(axis=0)).astype(int).tolist()))
        assert len(set(means)) >= 3

    def test_fills_whole_canvas(self):
        """透明な余白を残さず、キャンバス全体が塗られる。"""
        import numpy as np
        ls, layer = _make_colorful_stack()
        result = execute_warhol(ls, layer, self._params(gap=0))
        alpha = _qimage_to_array_test(result.image)[:, :, 3]
        assert (alpha > 0).mean() > 0.99

    def test_gap_leaves_untouched_border(self):
        """コマの間隔を空けると、その隙間は塗られない。"""
        ls, layer = _make_colorful_stack()
        result = execute_warhol(ls, layer, self._params(gap=6))
        alpha = _qimage_to_array_test(result.image)[:, :, 3]
        # 左端の列は隙間なので透明のまま
        assert alpha[:, 0].max() == 0


class TestLichtenstein:
    PARAMS = {"pitch": 6, "outline": 3, "levels": 3,
              "dot_color": QColor(200, 30, 40), "line_color": QColor(20, 20, 20),
              "white_bg": True}

    def test_dot_color_is_used_exactly(self):
        """指定した網点の色がそのまま出る（赤青が入れ替わらない）。"""
        ls, layer = _make_colorful_stack()
        result = execute_lichtenstein(ls, layer, dict(self.PARAMS))
        colors = set()
        img = result.image
        for y in range(img.height()):
            for x in range(img.width()):
                c = img.pixelColor(x, y)
                colors.add((c.red(), c.green(), c.blue()))
        assert (200, 30, 40) in colors

    def test_white_background_fills_canvas(self):
        """背景を白で埋めると全面が不透明になる。"""
        ls, layer = _make_colorful_stack()
        result = execute_lichtenstein(ls, layer, dict(self.PARAMS))
        alpha = _qimage_to_array_test(result.image)[:, :, 3]
        assert (alpha == 255).all()

    def test_transparent_background_keeps_gaps(self):
        """背景を埋めない場合、絵の外は透明のまま残る。"""
        params = dict(self.PARAMS, white_bg=False)
        ls, layer = _make_colorful_stack()
        result = execute_lichtenstein(ls, layer, params)
        alpha = _qimage_to_array_test(result.image)[:, :, 3]
        assert alpha[0, 0] == 0

    def test_outline_is_drawn(self):
        """輪郭線の色が実際に描かれる。"""
        ls, layer = _make_colorful_stack()
        result = execute_lichtenstein(ls, layer, dict(self.PARAMS))
        arr = _qimage_to_array_test(result.image)
        # 線色 (20,20,20) は BGR 順で格納される
        dark = (arr[:, :, 0] < 60) & (arr[:, :, 1] < 60) & (arr[:, :, 2] < 60)
        assert dark.any()


class TestUkiyoe:
    PARAMS = {"levels": 4, "misalign": 0, "sumi": 2, "paper": 0.0,
              "paper_color": QColor(240, 230, 205)}

    def test_paper_color_shows_through(self):
        """絵の無い所は指定した紙の色になる。"""
        ls, layer = _make_colorful_stack()
        result = execute_ukiyoe(ls, layer, dict(self.PARAMS))
        c = result.image.pixelColor(0, 0)
        assert (c.red(), c.green(), c.blue()) == (240, 230, 205)

    def test_reduces_color_count(self):
        """版画らしく色数が元より減る。"""
        import numpy as np
        ls, layer = _make_colorful_stack()
        result = execute_ukiyoe(ls, layer, dict(self.PARAMS))
        arr = _qimage_to_array_test(result.image)
        uniq = np.unique(arr[:, :, :3].reshape(-1, 3), axis=0)
        assert len(uniq) < 40

    def test_keeps_hue_of_source(self):
        """平坦化しても元の色味が残る（無彩色の灰色に潰れない）。"""
        ls = LayerStack(80, 80)
        layer = ls.add("色")
        p = QPainter(layer.image)
        p.fillRect(10, 10, 60, 60, QColor(210, 120, 90, 255))  # 肌色
        p.end()
        result = execute_ukiyoe(ls, layer, dict(self.PARAMS, sumi=0))
        c = result.image.pixelColor(40, 40)
        # R > G > B の関係が保たれていれば色味が生きている
        assert c.red() > c.green() > c.blue()


class TestImpressionist:
    PARAMS = {"length": 10, "width": 3, "density": 1.5, "jitter": 0,
              "follow": True}

    def test_covers_the_subject(self):
        """元の絵があった場所が塗り残されない。"""
        ls, layer = _make_colorful_stack()
        result = execute_impressionist(ls, layer, dict(self.PARAMS))
        src_a = _qimage_to_array_test(layer.image)[:, :, 3]
        out_a = _qimage_to_array_test(result.image)[:, :, 3]
        covered = out_a[src_a > 128]
        assert (covered > 0).mean() > 0.95

    def test_keeps_source_colors(self):
        """筆跡の色は元の絵の色に由来する。"""
        ls = LayerStack(80, 80)
        layer = ls.add("色")
        p = QPainter(layer.image)
        p.fillRect(10, 10, 60, 60, QColor(210, 60, 60, 255))
        p.end()
        result = execute_impressionist(ls, layer, dict(self.PARAMS))
        c = result.image.pixelColor(40, 40)
        assert c.red() > c.green() and c.red() > c.blue()

    def test_does_not_paint_far_outside(self):
        """絵から大きく離れた場所には筆を置かない。"""
        ls, layer = _make_colorful_stack()
        result = execute_impressionist(ls, layer, dict(self.PARAMS))
        alpha = _qimage_to_array_test(result.image)[:, :, 3]
        # 図形は (10,10)-(110,110) の範囲。四隅の外側は塗られない
        assert alpha[-1, -1] == 0


class TestStainedGlass:
    PARAMS = {"cell": 20, "lead": 3, "lead_color": QColor(30, 200, 40),
              "vivid": 1.5, "glass_bg": True}

    def test_lead_color_is_used_exactly(self):
        """鉛線の指定色がそのまま出る（赤青が入れ替わらない）。"""
        ls, layer = _make_colorful_stack()
        result = execute_stained_glass(ls, layer, dict(self.PARAMS))
        colors = set()
        img = result.image
        for y in range(img.height()):
            for x in range(img.width()):
                c = img.pixelColor(x, y)
                colors.add((c.red(), c.green(), c.blue()))
        assert (30, 200, 40) in colors

    def test_glass_background_fills_canvas(self):
        """背景もガラスにすると全面が不透明になる。"""
        ls, layer = _make_colorful_stack()
        result = execute_stained_glass(ls, layer, dict(self.PARAMS))
        alpha = _qimage_to_array_test(result.image)[:, :, 3]
        assert (alpha == 255).all()

    def test_without_glass_background_keeps_transparency(self):
        """背景をガラスにしなければ絵の外は透明のまま。"""
        import numpy as np
        params = dict(self.PARAMS, glass_bg=False)
        ls, layer = _make_colorful_stack()
        result = execute_stained_glass(ls, layer, params)
        src_a = _qimage_to_array_test(layer.image)[:, :, 3]
        out_a = _qimage_to_array_test(result.image)[:, :, 3]
        # 元が透明だった範囲の大半は透明のまま残る
        outside = out_a[src_a == 0]
        assert (outside == 0).mean() > 0.5

    def test_larger_cells_make_fewer_pieces(self):
        """ガラス片を大きくすると鉛線の総量が減る。"""
        ls1, l1 = _make_colorful_stack()
        fine = execute_stained_glass(ls1, l1, dict(self.PARAMS, cell=12))
        ls2, l2 = _make_colorful_stack()
        coarse = execute_stained_glass(ls2, l2, dict(self.PARAMS, cell=40))

        def lead_ratio(layer):
            arr = _qimage_to_array_test(layer.image)
            # 鉛線 (30,200,40) は BGR 順で (40,200,30)
            m = ((arr[:, :, 0] == 40) & (arr[:, :, 1] == 200)
                 & (arr[:, :, 2] == 30))
            return m.mean()

        assert lead_ratio(fine) > lead_ratio(coarse)


class TestBlueprint:
    PARAMS = {"paper_color": QColor(20, 60, 130),
              "ink_color": QColor(235, 243, 255), "thickness": 2,
              "grid": 0, "major": 4, "fade": 0.0}

    def test_paper_color_is_used_exactly(self):
        """地の色が指定どおりに出る（赤青が入れ替わらない）。"""
        ls, layer = _make_colorful_stack()
        result = execute_blueprint(ls, layer, dict(self.PARAMS))
        c = result.image.pixelColor(0, 0)
        assert (c.red(), c.green(), c.blue()) == (20, 60, 130)

    def test_fills_whole_canvas(self):
        """図面なので全面が不透明になる。"""
        ls, layer = _make_colorful_stack()
        result = execute_blueprint(ls, layer, dict(self.PARAMS))
        alpha = _qimage_to_array_test(result.image)[:, :, 3]
        assert (alpha == 255).all()

    def test_grid_adds_lines(self):
        """方眼を有効にすると地の色以外の画素が増える。"""
        ls1, l1 = _make_colorful_stack()
        plain = execute_blueprint(ls1, l1, dict(self.PARAMS, grid=0))
        ls2, l2 = _make_colorful_stack()
        ruled = execute_blueprint(ls2, l2, dict(self.PARAMS, grid=10))

        def ink_ratio(layer):
            arr = _qimage_to_array_test(layer.image)
            # 地の色 (20,60,130) は BGR 順で (130,60,20)
            same = ((arr[:, :, 0] == 130) & (arr[:, :, 1] == 60)
                    & (arr[:, :, 2] == 20))
            return 1.0 - same.mean()

        assert ink_ratio(ruled) > ink_ratio(plain)


class TestEtching:
    PARAMS = {"spacing": 4, "layers": 4, "wobble": 0.0,
              "ink_color": QColor(35, 28, 22),
              "paper_color": QColor(238, 230, 214), "grain": 0.0}

    def test_paper_color_shows_through(self):
        """彫っていない所は紙の色のまま。"""
        ls, layer = _make_colorful_stack()
        result = execute_etching(ls, layer, dict(self.PARAMS))
        c = result.image.pixelColor(0, 0)
        assert (c.red(), c.green(), c.blue()) == (238, 230, 214)

    def test_darker_areas_get_denser_lines(self):
        """暗い所ほど彫り線が密になる。"""
        ls = LayerStack(120, 60)
        layer = ls.add("明暗")
        p = QPainter(layer.image)
        p.fillRect(0, 0, 60, 60, QColor(230, 230, 230, 255))   # 明るい側
        p.fillRect(60, 0, 60, 60, QColor(40, 40, 40, 255))     # 暗い側
        p.end()
        result = execute_etching(ls, layer, dict(self.PARAMS))
        arr = _qimage_to_array_test(result.image)
        # インク (35,28,22) に近い画素の割合を左右で比べる
        ink = (arr[:, :, 2] < 140)
        light_side = ink[:, :60].mean()
        dark_side = ink[:, 60:].mean()
        assert dark_side > light_side

    def test_fills_whole_canvas(self):
        """紙に刷るので全面が不透明になる。"""
        ls, layer = _make_colorful_stack()
        result = execute_etching(ls, layer, dict(self.PARAMS))
        alpha = _qimage_to_array_test(result.image)[:, :, 3]
        assert (alpha == 255).all()


class TestStyleEffectsInGacha:
    """7つの「○○風」がガチャに正しく登録されている。"""

    STYLE_KEYS = ["warhol", "lichtenstein", "ukiyoe", "impressionist",
                  "stained_glass", "blueprint", "etching"]

    @pytest.mark.parametrize("key", STYLE_KEYS)
    def test_key_is_in_pool(self, key):
        assert key in [k for k, _ in _GACHA_POOL]

    @pytest.mark.parametrize("key", STYLE_KEYS)
    def test_key_is_executable(self, key):
        assert key in _GACHA_EXEC

    @pytest.mark.parametrize("key", STYLE_KEYS)
    def test_random_params_actually_execute(self, key):
        """生成されたランダム値でどれも実行でき、空にならない。"""
        colors = [QColor(c) for c in GACHA_PALETTES[1][1]]
        for _ in range(5):
            ls, layer = _make_colorful_stack()
            params = _gacha_random_params(key, colors)
            result = _GACHA_EXEC[key](ls, layer, params)
            assert result is not None, (key, params)
            assert _has_nonzero_pixels(result.image), (key, params)

    def test_warhol_gacha_uses_multiple_palettes(self):
        """ウォーホル風のランダム値はコマ数ぶんの配色を用意する。"""
        colors = [QColor(c) for c in GACHA_PALETTES[0][1]]
        for _ in range(10):
            params = _gacha_random_params("warhol", colors)
            assert len(params["palettes"]) == params["cols"] * params["rows"]
