"""ユニットテスト: Layer / GroupLayer / LayerStack (GUIなし)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor
from PyQt6.QtCore import Qt

app = QApplication.instance() or QApplication(sys.argv)

from layer import Layer, GroupLayer, LayerStack

W, H = 100, 100


def px(image: QImage, x: int, y: int) -> QColor:
    """QImage.pixel() の戻り値を正しく QColor に変換する。"""
    return QColor.fromRgba(image.pixel(x, y))


# ── Layer ────────────────────────────────────────────────────────────────────

class TestLayer:
    def test_init_transparent(self):
        layer = Layer("test", W, H)
        assert layer.image.width() == W
        assert layer.image.height() == H
        assert px(layer.image, 0, 0).alpha() == 0

    def test_clear(self):
        layer = Layer("test", W, H)
        from PyQt6.QtGui import QPainter
        p = QPainter(layer.image)
        p.fillRect(0, 0, W, H, QColor(255, 0, 0, 255))
        p.end()
        layer.clear()
        assert px(layer.image, 0, 0).alpha() == 0

    def test_is_group_false(self):
        assert Layer("x", W, H).is_group is False

    def test_image_with_border_passthrough_when_disabled(self):
        layer = Layer("test", W, H)
        layer.border_enabled = False
        assert layer.image_with_border() is layer.image

    def test_image_with_border_adds_outline(self):
        layer = Layer("test", W, H)
        layer.border_enabled = True
        layer.border_size = 3
        layer.border_color = QColor(255, 0, 0, 255)
        from PyQt6.QtGui import QPainter
        # 中央に白い四角を描く（背景は透明なので境界が明確）
        p = QPainter(layer.image)
        p.fillRect(40, 40, 20, 20, QColor(255, 255, 255, 255))
        p.end()
        result = layer.image_with_border()
        assert result is not layer.image  # 新しい画像が返る
        # 縁取り部分（描画エリアの外側付近）に赤ピクセルが存在するはず
        found_red = False
        for y in range(H):
            for x in range(W):
                c = px(result, x, y)
                if c.red() > 200 and c.green() < 50 and c.alpha() > 200:
                    found_red = True
                    break
            if found_red:
                break
        assert found_red, "縁取りピクセルが見つからない"

    def test_border_zero_size_passthrough(self):
        layer = Layer("test", W, H)
        layer.border_enabled = True
        layer.border_size = 0
        assert layer.image_with_border() is layer.image

    def test_image_with_border_no_gap_at_antialiased_edge(self):
        """線と同じ色の縁をつけても、アンチエイリアシングの縁ピクセルが
        中間色のまま残らない（線と縁の境目に薄い隙間が見えるバグの回帰確認）。"""
        from PyQt6.QtGui import QPainter, QPen
        layer = Layer("test", W, H)
        layer.image.fill(Qt.GlobalColor.transparent)
        p = QPainter(layer.image)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setPen(QPen(QColor(0, 0, 0, 255), 4))
        p.drawLine(10, 50, 90, 50)
        p.end()

        layer.border_enabled = True
        layer.border_size = 3
        layer.border_color = QColor(0, 0, 0, 255)  # 線と同じ黒縁

        result = layer.image_with_border()
        # 縁取り膨張範囲内は alpha=255 の黒で連続しているはず（隙間となる
        # 半透明・別色ピクセルが残っていないこと）
        for y in range(45, 56):
            for x in range(10, 90):
                c = px(result, x, y)
                if c.alpha() > 0:
                    assert c.alpha() == 255 and c.red() == 0 and c.green() == 0 and c.blue() == 0


# ── GroupLayer ───────────────────────────────────────────────────────────────

class TestGroupLayer:
    def test_is_group_true(self):
        assert GroupLayer("g", W, H).is_group is True

    def test_composite_empty_is_transparent(self):
        g = GroupLayer("g", W, H)
        result = g.composite()
        assert result.width() == W
        # グループが空の場合は透明
        assert px(result, 0, 0).alpha() == 0

    def test_composite_single_child(self):
        g = GroupLayer("g", W, H)
        child = Layer("c", W, H)
        from PyQt6.QtGui import QPainter
        p = QPainter(child.image)
        p.fillRect(0, 0, W, H, QColor(0, 0, 255, 255))
        p.end()
        g.children.append(child)
        result = g.composite()
        c = px(result, 50, 50)
        assert c.blue() > 200 and c.alpha() > 200

    def test_invisible_child_not_composited(self):
        g = GroupLayer("g", W, H)
        child = Layer("c", W, H)
        from PyQt6.QtGui import QPainter
        p = QPainter(child.image)
        p.fillRect(0, 0, W, H, QColor(255, 0, 0, 255))
        p.end()
        child.visible = False
        g.children.append(child)
        result = g.composite()
        # 非表示なので透明のまま
        assert px(result, 50, 50).alpha() == 0

    def test_composite_opacity(self):
        g = GroupLayer("g", W, H)
        child = Layer("c", W, H)
        child.opacity = 128
        from PyQt6.QtGui import QPainter
        p = QPainter(child.image)
        p.fillRect(0, 0, W, H, QColor(0, 0, 255, 255))
        p.end()
        g.children.append(child)
        result = g.composite()
        c = px(result, 50, 50)
        # 半透明の青 → alpha < 255
        assert 0 < c.alpha() < 255

    def test_resize_children(self):
        g = GroupLayer("g", W, H)
        child = Layer("c", W, H)
        g.children.append(child)
        g.resize(50, 50)
        assert child.image.width() == 50
        assert child.image.height() == 50


# ── LayerStack ───────────────────────────────────────────────────────────────

class TestLayerStack:
    def test_init_empty(self):
        ls = LayerStack(W, H)
        assert ls.layers == []
        assert ls.active is None

    def test_add_returns_layer(self):
        ls = LayerStack(W, H)
        layer = ls.add("レイヤー1")
        assert isinstance(layer, Layer)
        assert len(ls.layers) == 1
        assert ls.active is layer

    def test_add_inserts_at_active(self):
        ls = LayerStack(W, H)
        a = ls.add("A")
        b = ls.add("B")
        # B が active_index=0 に挿入されるため B が先頭
        assert ls.layers[0] is b
        assert ls.layers[1] is a

    def test_remove_keeps_at_least_one(self):
        ls = LayerStack(W, H)
        ls.add("A")
        ls.remove(0)
        assert len(ls.layers) == 1

    def test_remove_second_layer(self):
        ls = LayerStack(W, H)
        ls.add("A")
        ls.add("B")
        assert len(ls.layers) == 2
        ls.remove(1)
        assert len(ls.layers) == 1

    def test_active_clamps_after_remove(self):
        ls = LayerStack(W, H)
        ls.add("A")
        ls.add("B")
        ls.active_index = 1
        ls.remove(1)
        assert ls.active_index == 0

    def test_set_active_valid(self):
        ls = LayerStack(W, H)
        ls.add("A")
        ls.add("B")
        ls.set_active(1)
        assert ls.active_index == 1

    def test_set_active_out_of_range(self):
        ls = LayerStack(W, H)
        ls.add("A")
        ls.set_active(99)  # 範囲外 → 変化なし
        assert ls.active_index == 0

    def test_move_layer(self):
        ls = LayerStack(W, H)
        a = ls.add("A")
        b = ls.add("B")
        # B=index0, A=index1
        ls.move(0, 1)
        assert ls.layers[0] is a
        assert ls.layers[1] is b

    def test_active_in_group(self):
        ls = LayerStack(W, H)
        grp = ls.add_group("G")
        child = Layer("c", W, H)
        grp.children.append(child)
        ls.set_active(0, 0)
        assert ls.active is child

    def test_active_group_itself_when_child_index_minus1(self):
        ls = LayerStack(W, H)
        grp = ls.add_group("G")
        grp.children.append(Layer("c", W, H))
        ls.set_active(0, -1)
        assert ls.active is grp

    # ── merge_down ────────────────────────────────────────────────────────────

    def test_merge_down_two_layers(self):
        ls = LayerStack(W, H)
        a = ls.add("A")  # 先に追加 → index1
        b = ls.add("B")  # 後で挿入 → index0（上）
        ls.set_active(0)
        result = ls.merge_down()
        assert result is True
        assert len(ls.layers) == 1

    def test_merge_down_preserves_pixels(self):
        ls = LayerStack(W, H)
        a = ls.add("A")
        b = ls.add("B")
        ls.set_active(0)  # b が上
        from PyQt6.QtGui import QPainter
        p = QPainter(b.image)
        p.fillRect(0, 0, 10, 10, QColor(255, 0, 0, 255))
        p.end()
        p = QPainter(a.image)
        p.fillRect(50, 50, 10, 10, QColor(0, 0, 255, 255))
        p.end()
        ls.merge_down()
        merged = ls.layers[0]
        # b の赤が残っているはず
        assert px(merged.image, 5, 5).red() > 200
        # a の青も残っているはず
        assert px(merged.image, 55, 55).blue() > 200

    def test_merge_down_fails_at_bottom(self):
        ls = LayerStack(W, H)
        ls.add("A")
        ls.set_active(0)
        assert ls.merge_down() is False

    def test_merge_down_fails_on_group(self):
        ls = LayerStack(W, H)
        ls.add_group("G")
        ls.add("A")
        ls.set_active(0)  # A が上（index0）
        assert ls.merge_down() is False

    def test_merge_down_in_group(self):
        ls = LayerStack(W, H)
        grp = ls.add_group("G")
        c1 = Layer("c1", W, H)
        c2 = Layer("c2", W, H)
        grp.children.extend([c1, c2])
        ls.set_active(0, 0)
        result = ls.merge_down()
        assert result is True
        assert len(grp.children) == 1

    # ── merge_all_visible ─────────────────────────────────────────────────────

    def test_merge_all_visible_collapses_layers(self):
        ls = LayerStack(W, H)
        ls.add("A")
        ls.add("B")
        ls.add("C")
        assert ls.merge_all_visible() is True
        assert len(ls.layers) == 1
        assert ls.layers[0].name == "統合レイヤー"

    def test_merge_all_visible_keeps_hidden(self):
        ls = LayerStack(W, H)
        ls.add("A")
        b = ls.add("B")
        b.visible = False
        ls.merge_all_visible()
        names = [l.name for l in ls.layers]
        assert "統合レイヤー" in names
        assert "B" in names

    def test_merge_all_visible_empty(self):
        ls = LayerStack(W, H)
        assert ls.merge_all_visible() is False

    # ── フォルダ結合(_draw_layers_to)のクリッピング反映 ─────────────────────────
    # グループを結合すると clipping フラグが無視され、クリッピングされているはずの
    # レイヤーが全面に描画されてしまうバグの回帰確認（実際のユーザーファイルで
    # 「色が消えて黒線だけになる」症状として再現した）。

    def test_group_merge_respects_clipping(self):
        from PyQt6.QtGui import QPainter
        ls = LayerStack(W, H)
        grp = ls.add_group("G")
        # 下: 中央に小さい不透明な四角（線画相当）
        base = Layer("base", W, H)
        base.image.fill(Qt.GlobalColor.transparent)
        p = QPainter(base.image)
        p.fillRect(40, 40, 20, 20, QColor(0, 0, 0, 255))
        p.end()
        # 上: 全面を塗る色レイヤー、clipping=True（下のレイヤーの形状でマスクされるべき）
        color = Layer("color", W, H)
        color.image.fill(QColor(0, 255, 0, 255))
        color.clipping = True
        # children はトップが先頭（[0]=color が [1]=base の上にクリップされる）
        grp.children.extend([color, base])

        ls.active_path = [0]
        assert ls.merge_all_visible() is True
        merged = ls.layers[0].image
        # クリッピングされていれば、四角の外側は透明のまま（緑で塗り潰されない）
        assert px(merged, 5, 5).alpha() == 0
        # 四角の内側は緑色（色レイヤーがクリップされて反映されている）
        c = px(merged, 50, 50)
        assert c.green() > 200 and c.red() < 50

    # ── composite ─────────────────────────────────────────────────────────────

    def test_composite_transparent_background(self):
        ls = LayerStack(W, H)
        result = ls.composite()
        c = px(result, 0, 0)
        assert c.alpha() == 0

    def test_composite_layer_color(self):
        ls = LayerStack(W, H)
        layer = ls.add("A")
        from PyQt6.QtGui import QPainter
        p = QPainter(layer.image)
        p.fillRect(0, 0, W, H, QColor(0, 255, 0, 255))
        p.end()
        result = ls.composite()
        c = px(result, 50, 50)
        assert c.green() > 200

    def test_composite_opacity_preserves_alpha(self):
        ls = LayerStack(W, H)
        layer = ls.add("A")
        layer.opacity = 128
        from PyQt6.QtGui import QPainter
        p = QPainter(layer.image)
        p.fillRect(0, 0, W, H, QColor(0, 0, 255, 255))
        p.end()
        result = ls.composite()
        c = px(result, 50, 50)
        assert 120 <= c.alpha() <= 130
        assert c.blue() > 200

    def test_composite_invisible_layer_skipped(self):
        ls = LayerStack(W, H)
        layer = ls.add("A")
        layer.visible = False
        from PyQt6.QtGui import QPainter
        p = QPainter(layer.image)
        p.fillRect(0, 0, W, H, QColor(0, 0, 255, 255))
        p.end()
        result = ls.composite()
        c = px(result, 50, 50)
        assert c.alpha() == 0

    def test_add_group(self):
        ls = LayerStack(W, H)
        grp = ls.add_group("G")
        assert grp.is_group
        assert len(ls.layers) == 1
