"""ユニットテスト: Layer / GroupLayer / LayerStack (GUIなし)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor, QPainter
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

    # ── フォルダ内のグループが絡むクリッピング ──────────────────────────────
    # フォルダ内では「クリップ先がサブフォルダ」「クリッピングフラグ付きサブフォルダ」
    # のどちらも無視され、クリップ側が全面にそのまま描画されるバグの回帰確認。
    # （トップレベルの LayerStack.composite では従来から正しく動いていた）

    @staticmethod
    def _solid(x, y, w, h, color):
        from PyQt6.QtGui import QPainter
        lyr = Layer("l", W, H)
        p = QPainter(lyr.image)
        p.fillRect(x, y, w, h, color)
        p.end()
        return lyr

    def test_nested_clip_onto_subfolder_in_composite(self):
        # フォルダ内: 全面赤(clipping=True) の下にサブフォルダ（青四角）
        red = self._solid(0, 0, W, H, QColor(255, 0, 0, 255))
        red.clipping = True
        sub = GroupLayer("sub", W, H)
        sub.children = [self._solid(20, 20, 30, 30, QColor(0, 0, 255, 255))]
        outer = GroupLayer("outer", W, H)
        outer.children = [red, sub]
        ls = LayerStack(W, H)
        ls.layers = [outer]
        img = ls.composite()
        assert px(img, 30, 30).red() > 200        # 四角の内側は赤
        assert px(img, 80, 80).alpha() == 0        # 外側は透明のまま

    def test_nested_clipping_flagged_subfolder_in_composite(self):
        # フォルダ内: クリッピングフラグ付きサブフォルダ（全面緑）の下に青四角
        sub = GroupLayer("sub", W, H)
        sub.children = [self._solid(0, 0, W, H, QColor(0, 255, 0, 255))]
        sub.clipping = True
        base = self._solid(20, 20, 30, 30, QColor(0, 0, 255, 255))
        outer = GroupLayer("outer", W, H)
        outer.children = [sub, base]
        ls = LayerStack(W, H)
        ls.layers = [outer]
        img = ls.composite()
        assert px(img, 30, 30).green() > 200
        assert px(img, 80, 80).alpha() == 0

    def test_nested_clip_onto_subfolder_in_merge(self):
        # 統合（_draw_layers_to）でも同じクリッピングが反映されること
        red = self._solid(0, 0, W, H, QColor(255, 0, 0, 255))
        red.clipping = True
        sub = GroupLayer("sub", W, H)
        sub.children = [self._solid(20, 20, 30, 30, QColor(0, 0, 255, 255))]
        outer = GroupLayer("outer", W, H)
        outer.children = [red, sub]
        ls = LayerStack(W, H)
        ls.layers = [outer]
        ls.active_path = [0]
        assert ls.merge_all_visible() is True
        merged = ls.layers[0]
        ox, oy = merged.offset_x, merged.offset_y
        assert px(merged.image, 30 - ox, 30 - oy).red() > 200
        assert px(merged.image, 80 - ox, 80 - oy).alpha() == 0

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


class TestEffectCache:
    """効果適用済み画像のキャッシュ。

    再描画のたびに縁取りやぼかしを計算し直すと、
    レイヤーが増えるほど描画が重くなる。
    """

    def _layer(self):
        l = Layer("t", 120, 120)
        p = QPainter(l.image)
        p.fillRect(20, 20, 60, 60, QColor(220, 40, 40))
        p.end()
        l.border_enabled = True
        l.border_size = 3
        return l

    def _count_recompute(self, monkeypatch):
        calls = []
        orig = Layer._compute_effects
        def spy(self):
            calls.append(1)
            return orig(self)
        monkeypatch.setattr(Layer, "_compute_effects", spy)
        return calls

    def test_second_call_uses_cache(self, monkeypatch):
        l = self._layer()
        l.image_with_effects()
        calls = self._count_recompute(monkeypatch)
        l.image_with_effects()
        l.image_with_effects()
        assert not calls, "キャッシュが効いていない"

    def test_reading_image_does_not_break_cache(self, monkeypatch):
        """効果の適用自体がキャッシュを失効させないこと。

        QImage.bits() は書き込み用でデタッチが走り cacheKey が
        変わる。効果の中でこれを使うと、毎回キャッシュが
        外れて黙って遅くなる（テストでは落ちない）ので固める。
        """
        l = self._layer()
        l.shadow_enabled = True
        l.glow_enabled = True
        l.blur_enabled = True
        l.hsl_enabled = True
        l.hsl_hue = 20
        before = l.image.cacheKey()
        l.image_with_effects()
        assert l.image.cacheKey() == before,             "効果の適用で元画像の cacheKey が変わっている"
        calls = self._count_recompute(monkeypatch)
        l.image_with_effects()
        assert not calls, "効果ONのときにキャッシュが毎回外れている"

    # (先に有効化しておく効果, 変更する項目, 変更後の値)
    # 各パラメータは対応する *_enabled が True のときだけ効くので、
    # 先に有効化してから変えないと見た目は変わらない。
    @pytest.mark.parametrize("enable,attr,value", [
        (None, "border_size", 9),
        (None, "border_color", QColor(0, 255, 0)),
        (None, "shadow_enabled", True),
        (None, "glow_enabled", True),
        (None, "blur_enabled", True),
        # hsl_enabled 単体は色相/彩度/明度が全部0だと何も変えないので、
        # 値を入れた状態で有効化する形で確認する。
        ("hsl_hue_set", "hsl_enabled", True),
        ("hsl_enabled", "hsl_hue", 90),
        ("hsl_enabled", "hsl_saturation", 60),
        ("hsl_enabled", "hsl_lightness", 40),
        ("shadow_enabled", "shadow_offset_x", 15),
        ("shadow_enabled", "shadow_offset_y", 15),
        ("shadow_enabled", "shadow_blur", 12),
        ("shadow_enabled", "shadow_color", QColor(0, 0, 255)),
        ("shadow_enabled", "shadow_strength", 40),
        ("glow_enabled", "glow_size", 20),
        ("glow_enabled", "glow_color", QColor(255, 0, 0)),
        ("glow_enabled", "glow_strength", 30),
        ("blur_enabled", "blur_radius", 8),
        ("blur_enabled", "blur_strength", 50),
    ])
    def test_param_change_recomputes(self, enable, attr, value):
        """設定を変えたら必ず見た目が追従すること。

        キーに入れ忘れた項目は「変えても画面に反映されない」
        という不具合になるので、全項目を網羅する。
        """
        l = self._layer()
        if enable == "hsl_hue_set":
            l.hsl_hue = 120
        elif enable:
            setattr(l, enable, True)
        first = l.image_with_effects().copy()
        setattr(l, attr, value)
        assert l.image_with_effects().copy() != first, \
            f"{attr} を変えても結果が変わらない"

    def test_drawing_recomputes(self):
        l = self._layer()
        first = l.image_with_effects().copy()
        p = QPainter(l.image)
        p.fillRect(0, 0, 15, 15, QColor(0, 0, 255))
        p.end()
        assert l.image_with_effects().copy() != first,             "画像を描き換えても結果が変わらない"

    def test_no_effects_returns_live_image(self):
        """効果なしのときは別名参照をキャッシュしないこと。"""
        l = Layer("t", 60, 60)
        assert l.image_with_effects() is l.image
        assert l._effect_cache is None
        p = QPainter(l.image)
        p.fillRect(0, 0, 20, 20, QColor(255, 0, 0))
        p.end()
        assert l.image_with_effects() is l.image
