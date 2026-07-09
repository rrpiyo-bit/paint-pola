"""ユニットテスト: Layer.rasterize() と LayerStack.merge_down() の効果焼き込み (GUIなし)

背景: 縁取り等の効果を設定したレイヤーを統合すると、効果が消えてしまう/
統合後の新しい内容に対して効果が二重適用されたままになる不具合があった。
rasterize() は効果を画像に焼き込んで設定を無効化する新機能。
merge_down() は焼き込み漏れがないよう image_with_effects() を使うよう修正した。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor, QPainter
from PyQt6.QtCore import Qt

app = QApplication.instance() or QApplication(sys.argv)

from layer import Layer, LayerStack

W, H = 60, 60


def px(image: QImage, x: int, y: int) -> QColor:
    return QColor.fromRgba(image.pixel(x, y))


def make_border_layer(name="L", w=W, h=H, fill_rect=(15, 15, 20, 20)) -> Layer:
    lyr = Layer(name, w, h)
    lyr.image.fill(Qt.GlobalColor.transparent)
    p = QPainter(lyr.image)
    p.fillRect(*fill_rect, QColor(255, 0, 0, 255))
    p.end()
    lyr.border_enabled = True
    lyr.border_size = 3
    lyr.border_color = QColor(0, 0, 255, 255)
    return lyr


class TestLayerRasterize:
    def test_rasterize_disables_all_effect_flags(self):
        lyr = make_border_layer()
        lyr.shadow_enabled = True
        lyr.glow_enabled = True
        lyr.blur_enabled = True
        lyr.hsl_enabled = True
        lyr.rasterize()
        assert lyr.border_enabled is False
        assert lyr.shadow_enabled is False
        assert lyr.glow_enabled is False
        assert lyr.blur_enabled is False
        assert lyr.hsl_enabled is False

    def test_rasterize_preserves_visual_output(self):
        """ラスタライズ前後で image_with_effects() の見た目が変化しないこと。"""
        lyr = make_border_layer()
        before = lyr.image_with_effects().convertToFormat(QImage.Format.Format_ARGB32)
        lyr.rasterize()
        after = lyr.image_with_effects().convertToFormat(QImage.Format.Format_ARGB32)
        assert before.size() == after.size()

        def raw(img):
            b = img.bits(); b.setsize(img.sizeInBytes())
            return bytes(b)

        assert raw(before) == raw(after)

    def test_rasterize_bakes_border_into_pixels(self):
        """ラスタライズ後、border色がimage自体に焼き込まれていること。"""
        lyr = make_border_layer()
        lyr.rasterize()
        # 縁取り色(青)が image (無加工の生ピクセル) に含まれる
        b = lyr.image.bits(); b.setsize(lyr.image.sizeInBytes())
        import numpy as np
        arr = np.frombuffer(b, dtype=np.uint8).reshape(H, W, 4)
        blue_bgra = np.array([255, 0, 0, 255], dtype=np.uint8)  # BGRA
        assert np.any(np.all(arr == blue_bgra, axis=2))

    def test_rasterize_noop_pixels_when_no_effects_enabled(self):
        lyr = Layer("plain", W, H)
        lyr.image.fill(QColor(10, 20, 30, 255))
        before = lyr.image.copy()
        lyr.rasterize()

        def raw(img):
            b = img.bits(); b.setsize(img.sizeInBytes())
            return bytes(b)

        assert raw(before) == raw(lyr.image)


class TestMergeDownBakesEffects:
    def test_merge_down_bakes_lower_layer_border(self):
        """下側レイヤーに縁取りが設定されている場合、統合後に画像へ焼き込まれ、
        フラグはリセットされて二重適用されないこと。"""
        stack = LayerStack(W, H)
        upper = Layer("upper", W, H)
        upper.image.fill(Qt.GlobalColor.transparent)
        lower = make_border_layer("lower", W, H, fill_rect=(10, 10, 15, 15))
        stack.layers = [upper, lower]
        stack.active_path = [0]  # upper がアクティブ→下の lower に統合

        assert stack.merge_down() is True
        assert len(stack.layers) == 1
        merged = stack.layers[0]

        assert merged.border_enabled is False
        assert merged.shadow_enabled is False
        assert merged.glow_enabled is False
        assert merged.blur_enabled is False
        assert merged.hsl_enabled is False

        # 縁取り色が焼き込まれている
        import numpy as np
        img = merged.image_with_effects().convertToFormat(QImage.Format.Format_ARGB32)
        b = img.bits(); b.setsize(img.sizeInBytes())
        arr = np.frombuffer(b, dtype=np.uint8).reshape(img.height(), img.width(), 4)
        blue_bgra = np.array([255, 0, 0, 255], dtype=np.uint8)
        assert np.any(np.all(arr == blue_bgra, axis=2))

    def test_merge_down_bakes_upper_layer_effects_too(self):
        """上側レイヤーに効果が設定されている場合も、統合後に焼き込まれること。"""
        stack = LayerStack(W, H)
        upper = make_border_layer("upper", W, H, fill_rect=(5, 5, 10, 10))
        lower = Layer("lower", W, H)
        lower.image.fill(Qt.GlobalColor.transparent)
        stack.layers = [upper, lower]
        stack.active_path = [0]

        assert stack.merge_down() is True
        merged = stack.layers[0]
        assert merged.border_enabled is False

    def test_merge_down_no_effects_unaffected(self):
        """効果を使っていない通常の統合は従来通り成功すること（回帰確認）。"""
        stack = LayerStack(W, H)
        upper = Layer("upper", W, H)
        upper.image.fill(QColor(255, 0, 0, 255))
        lower = Layer("lower", W, H)
        lower.image.fill(QColor(0, 255, 0, 255))
        stack.layers = [upper, lower]
        stack.active_path = [0]

        assert stack.merge_down() is True
        assert len(stack.layers) == 1
        assert px(stack.layers[0].image, 0, 0).red() == 255
