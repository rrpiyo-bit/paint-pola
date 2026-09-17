"""ユニットテスト: ベクターレイヤー (GUIなし)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor, QMouseEvent
from PyQt6.QtCore import Qt, QPointF, QEvent

app = QApplication.instance() or QApplication(sys.argv)

from vector import (VectorStroke, VectorLayer, catmull_rom_to_path,
                    rdp_simplify, drop_near_duplicates, simplify_input)

W, H = 100, 100


def px(image: QImage, x: int, y: int) -> QColor:
    return QColor.fromRgba(image.pixel(x, y))


# ── 曲線 ─────────────────────────────────────────────────────────────────────

class TestCurve:
    def test_passes_through_control_points(self):
        """Catmull-Rom は制御点を必ず通る。「点を動かす＝線がそこを通る」の土台。"""
        pts = [(10.0, 10.0), (30.0, 50.0), (60.0, 20.0), (90.0, 70.0)]
        path = catmull_rom_to_path(pts, smooth=True)
        flat = [p for poly in path.toSubpathPolygons() for p in poly]
        for x, y in pts:
            assert any(abs(p.x() - x) < 0.5 and abs(p.y() - y) < 0.5 for p in flat), \
                f"制御点 ({x},{y}) を通っていない"

    def test_single_point(self):
        path = catmull_rom_to_path([(5.0, 5.0)])
        assert not path.isEmpty()

    def test_empty(self):
        assert catmull_rom_to_path([]).isEmpty()

    def test_two_points_is_straight(self):
        # 2点はベジェ1本で表されるが、形は完全な直線でなければならない
        path = catmull_rom_to_path([(0.0, 0.0), (10.0, 0.0)], smooth=True)
        xs = []
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            pt = path.pointAtPercent(t)
            assert abs(pt.y()) < 1e-6, "2点の線が曲がっている"
            xs.append(pt.x())
        assert xs == sorted(xs)
        assert abs(xs[0]) < 1e-6 and abs(xs[-1] - 10.0) < 1e-6


# ── 間引き ───────────────────────────────────────────────────────────────────

class TestSimplify:
    def test_rdp_keeps_ends(self):
        pts = [(0.0, 0.0), (1.0, 0.1), (2.0, 0.0), (3.0, 0.1), (10.0, 0.0)]
        out = rdp_simplify(pts, 1.0)
        assert out[0] == pts[0] and out[-1] == pts[-1]

    def test_rdp_drops_collinear(self):
        pts = [(float(i), 0.0) for i in range(20)]
        assert len(rdp_simplify(pts, 1.0)) == 2

    def test_rdp_keeps_corner(self):
        pts = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 10.0)]
        assert len(rdp_simplify(pts, 1.0)) == 3

    def test_rdp_long_stroke_no_recursion_error(self):
        """再帰だとスタックが尽きる長さ。明示スタック実装であることの確認。"""
        import math
        pts = [(float(i), math.sin(i * 0.05) * 50) for i in range(20000)]
        out = rdp_simplify(pts, 0.01)
        assert len(out) >= 2

    def test_drop_near_duplicates(self):
        pts = [(0.0, 0.0), (0.5, 0.0), (5.0, 0.0)]
        assert drop_near_duplicates(pts, 2.0) == [(0.0, 0.0), (5.0, 0.0)]

    def test_simplify_input_keeps_shape(self):
        pts = [(0.0, 0.0), (50.0, 0.0), (50.0, 50.0)]
        out = simplify_input(pts, 5.0)
        assert out[0] == pts[0] and out[-1] == pts[-1]
        assert len(out) == 3


# ── VectorLayer ──────────────────────────────────────────────────────────────

class TestVectorLayer:
    def test_is_vector(self):
        assert VectorLayer("v", W, H).is_vector is True

    def test_raster_layer_is_not_vector(self):
        from layer import Layer, GroupLayer
        assert Layer("r", W, H).is_vector is False
        assert GroupLayer("g", W, H).is_vector is False

    def test_empty_image_is_canvas_size(self):
        lyr = VectorLayer("v", W, H)
        assert lyr.image.size().width() == W
        assert lyr.image.size().height() == H

    def test_draws_after_add(self):
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(10.0, 50.0), (90.0, 50.0)],
                                    width=6.0, color=(255, 0, 0, 255)))
        assert px(lyr.image, 50, 50).alpha() > 200

    def test_in_canvas_stroke_keeps_canvas_size(self):
        """キャンバスに収まる線では image を広げない。
        広げると offset が描くたびに動いて移動ツールが狂う。"""
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(20.0, 20.0), (80.0, 80.0)], width=4.0))
        img = lyr.image
        assert (img.width(), img.height()) == (W, H)
        assert (lyr.offset_x, lyr.offset_y) == (0, 0)

    def test_overflow_expands_image(self):
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(-50.0, 50.0), (50.0, 50.0)], width=4.0))
        img = lyr.image
        assert lyr.offset_x < 0
        assert img.width() > W

    def test_width_change_redraws(self):
        lyr = VectorLayer("v", W, H)
        s = VectorStroke(points=[(10.0, 50.0), (90.0, 50.0)], width=2.0,
                         color=(0, 0, 0, 255))
        lyr.add_stroke(s)
        assert px(lyr.image, 50, 42).alpha() < 50
        s.width = 30.0
        lyr.mark_dirty()
        assert px(lyr.image, 50, 42).alpha() > 200

    def test_translate_strokes(self):
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(10.0, 10.0), (20.0, 10.0)]))
        lyr.translate_strokes(5.0, 7.0)
        assert lyr.strokes[0].points[0] == (15.0, 17.0)

    def test_scale_strokes_scales_width(self):
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(10.0, 10.0), (20.0, 20.0)], width=10.0))
        lyr.scale_strokes(2.0, 2.0)
        assert lyr.strokes[0].points[1] == (40.0, 40.0)
        assert lyr.strokes[0].width == pytest.approx(20.0)

    def test_copy_strokes_is_deep(self):
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(1.0, 1.0), (2.0, 2.0)]))
        snap = lyr.copy_strokes()
        lyr.translate_strokes(100.0, 100.0)
        assert snap[0].points[0] == (1.0, 1.0)

    def test_image_setter_stops_rerender(self):
        """履歴から絵を直接入れられたら、それを上書きして描き直さない。"""
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(10.0, 50.0), (90.0, 50.0)], width=6.0))
        blank = QImage(W, H, QImage.Format.Format_ARGB32)
        blank.fill(0)
        lyr.image = blank
        assert px(lyr.image, 50, 50).alpha() == 0

    def test_set_canvas_size_redraws(self):
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(10.0, 10.0), (20.0, 20.0)]))
        lyr.set_canvas_size(200, 150)
        assert (lyr.image.width(), lyr.image.height()) == (200, 150)

    def test_to_raster_drops_vectorness(self):
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(10.0, 50.0), (90.0, 50.0)], width=6.0))
        raster = lyr.to_raster()
        assert raster.is_vector is False
        assert px(raster.image, 50, 50).alpha() > 200

    def test_offset_move_alone_does_not_move_strokes(self):
        """点はキャンバス座標なので offset を動かしても絵は動かない。
        移動ツール・矢印キーが translate_strokes を使わねばならない理由。"""
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(10.0, 50.0), (90.0, 50.0)], width=6.0))
        _ = lyr.image
        lyr.offset_x = 30
        lyr.mark_dirty()
        _ = lyr.image      # 読んだ時点で描き直される
        assert lyr.offset_x == 0, "再レンダーで offset が戻らないと絵がずれる"

    def test_crop_resize_keeps_stroke_position(self):
        """キャンバスを広げたとき、offset 分だけ点が動いて絵の位置が保たれる。"""
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(10.0, 10.0), (20.0, 20.0)], width=4.0))
        lyr.translate_strokes(50, 50)       # _resize_layer_image の crop 分岐と同じ
        lyr.set_canvas_size(200, 200)
        assert lyr.strokes[0].points[0] == (60.0, 60.0)
        assert (lyr.image.width(), lyr.image.height()) == (200, 200)

    def test_scale_resize_keeps_stroke_position(self):
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(10.0, 10.0), (20.0, 20.0)], width=4.0))
        lyr.scale_strokes(2.0, 2.0)         # _resize_layer_image の scale 分岐と同じ
        lyr.set_canvas_size(200, 200)
        assert lyr.strokes[0].points[1] == (40.0, 40.0)
        assert lyr.strokes[0].width == pytest.approx(8.0)

    def test_effect_cache_invalidated_by_redraw(self):
        """効果は image.cacheKey() でキャッシュされるので、
        再レンダーで自動的に無効化されること。"""
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(10.0, 50.0), (90.0, 50.0)], width=4.0))
        first = lyr.image_with_effects()
        lyr.strokes[0].width = 40.0
        lyr.mark_dirty()
        second = lyr.image_with_effects()
        assert first.cacheKey() != second.cacheKey()


# ── キャンバス上での振る舞い ─────────────────────────────────────────────────

class TestCanvasIntegration:
    @staticmethod
    def make():
        from layer import LayerStack
        from canvas import Canvas
        ls = LayerStack(W, H)
        ls.add("レイヤー1")
        lyr = VectorLayer("v", W, H)
        ls.layers.insert(0, lyr)
        ls.set_active_path([0])
        c = Canvas(ls)
        c.resize(W, H)
        return c, lyr

    @staticmethod
    def press(c, x, y):
        from PyQt6.QtCore import QPoint
        wp = c._c2w().map(QPointF(x, y))
        c.mousePressEvent(QMouseEvent(
            QEvent.Type.MouseButtonPress, wp, Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))

    def test_raster_tools_are_blocked(self):
        """画像に直接描く道具は止める。通してしまうと、描いた直後は見えるのに
        次の描き直しで消えるという、原因の分からないバグになる。"""
        from tools import Tool
        for tool in (Tool.FILL, Tool.BLUR, Tool.LINE, Tool.RECT, Tool.ELLIPSE,
                     Tool.TEXT, Tool.LASSO_FILL, Tool.ERASER):
            c, lyr = self.make()
            msgs = []
            c.status_message.connect(msgs.append)
            c.tool = tool
            self.press(c, 50, 50)
            assert msgs, f"{tool} がベクターレイヤーで止まっていない"
            assert c._drawing is False

    def test_pen_is_allowed(self):
        from tools import Tool
        c, lyr = self.make()
        c.tool = Tool.PEN
        self.press(c, 50, 50)
        assert c._vector_points == [(50.0, 50.0)]

    def test_raster_layer_tools_still_work(self):
        """ラスターレイヤーの挙動は変えない。"""
        from tools import Tool
        c, lyr = self.make()
        c.layer_stack.set_active_path([1])
        msgs = []
        c.status_message.connect(msgs.append)
        c.tool = Tool.RECT
        self.press(c, 50, 50)
        assert not msgs

    # ── 選択モード ──────────────────────────────────────────────────────────

    @staticmethod
    def press_mod(c, x, y, mods=Qt.KeyboardModifier.NoModifier):
        wp = c._c2w().map(QPointF(x, y))
        c.mousePressEvent(QMouseEvent(
            QEvent.Type.MouseButtonPress, wp, Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton, mods))

    @staticmethod
    def move_to(c, x, y, mods=Qt.KeyboardModifier.NoModifier):
        wp = c._c2w().map(QPointF(x, y))
        c.mouseMoveEvent(QMouseEvent(
            QEvent.Type.MouseMove, wp, Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton, mods))

    @staticmethod
    def release(c, x, y):
        wp = c._c2w().map(QPointF(x, y))
        c.mouseReleaseEvent(QMouseEvent(
            QEvent.Type.MouseButtonRelease, wp, Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))

    def select_setup(self):
        """線が1本あるベクターレイヤーを選択モードで用意する。"""
        from tools import Tool
        c, lyr = self.make()
        c.tool = Tool.PEN
        lyr.add_stroke(VectorStroke(
            points=[(20.0, 50.0), (50.0, 50.0), (80.0, 50.0)], width=6.0))
        c.vector_pen_mode = "select"
        return c, lyr

    def test_click_selects_line(self):
        c, lyr = self.select_setup()
        self.press_mod(c, 50, 50)
        assert c._vector_selected is lyr.strokes[0]

    def test_click_empty_deselects(self):
        c, lyr = self.select_setup()
        self.press_mod(c, 50, 50)
        self.press_mod(c, 10, 10)
        assert c._vector_selected is None

    def test_drag_point_moves_it(self):
        c, lyr = self.select_setup()
        s = lyr.strokes[0]
        self.press_mod(c, 50, 50)          # 選ぶ
        self.press_mod(c, 50, 50)          # 真ん中の点を掴む
        assert c._vector_drag == ("point", 1)
        self.move_to(c, 50, 70)
        self.release(c, 50, 70)
        assert s.points[1] == (50.0, 70.0)
        assert c._vector_drag is None

    def test_click_without_drag_leaves_no_history(self):
        """掴んだだけで動かさなかったら、元に戻すが空振りしないよう履歴を戻す。"""
        c, lyr = self.select_setup()
        self.press_mod(c, 50, 50)
        before = len(c._history)
        self.press_mod(c, 50, 50)
        self.release(c, 50, 50)
        assert len(c._history) == before

    def test_alt_shift_click_toggles_handles(self):
        c, lyr = self.select_setup()
        s = lyr.strokes[0]
        mods = (Qt.KeyboardModifier.AltModifier
                | Qt.KeyboardModifier.ShiftModifier)
        self.press_mod(c, 50, 50)
        self.press_mod(c, 50, 50, mods)
        assert s.has_handles(1)
        self.press_mod(c, 50, 50, mods)
        assert not s.has_handles(1)

    def test_handles_do_not_change_shape(self):
        """ハンドルを出しただけでは線の形が変わってはいけない。"""
        from vector import stroke_to_path
        c, lyr = self.select_setup()
        s = lyr.strokes[0]
        before = [stroke_to_path(s).pointAtPercent(t / 10.0)
                  for t in range(11)]
        s.set_handles(1, s.auto_handles(1))
        after = [stroke_to_path(s).pointAtPercent(t / 10.0)
                 for t in range(11)]
        for a, b in zip(before, after):
            assert abs(a.x() - b.x()) < 1e-6
            assert abs(a.y() - b.y()) < 1e-6

    def test_alt_click_deletes_point(self):
        c, lyr = self.select_setup()
        s = lyr.strokes[0]
        self.press_mod(c, 50, 50)
        self.press_mod(c, 50, 50, Qt.KeyboardModifier.AltModifier)
        assert len(s.points) == 2
        assert len(s.handles) == 2

    def test_alt_click_refuses_below_two_points(self):
        c, lyr = self.select_setup()
        s = lyr.strokes[0]
        del s.points[2]
        del s.handles[2]
        self.press_mod(c, 50, 50)
        self.press_mod(c, 50, 50, Qt.KeyboardModifier.AltModifier)
        assert len(s.points) == 2

    def test_shift_click_inserts_point(self):
        c, lyr = self.select_setup()
        s = lyr.strokes[0]
        self.press_mod(c, 50, 50)
        self.press_mod(c, 35, 50, Qt.KeyboardModifier.ShiftModifier)
        assert len(s.points) == 4
        assert len(s.handles) == 4

    def test_delete_key_removes_selected_line(self):
        from PyQt6.QtGui import QKeyEvent
        c, lyr = self.select_setup()
        self.press_mod(c, 50, 50)
        c.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete,
                                  Qt.KeyboardModifier.NoModifier))
        assert lyr.strokes == []
        assert c._vector_selected is None

    def test_escape_deselects(self):
        from PyQt6.QtGui import QKeyEvent
        c, lyr = self.select_setup()
        self.press_mod(c, 50, 50)
        c.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                  Qt.KeyboardModifier.NoModifier))
        assert c._vector_selected is None
        assert len(lyr.strokes) == 1

    def test_width_change_and_undo(self):
        c, lyr = self.select_setup()
        s = lyr.strokes[0]
        self.press_mod(c, 50, 50)
        c.set_selected_stroke_width(20)
        assert lyr.strokes[0].width == 20.0
        c.undo()
        assert lyr.strokes[0].width == 6.0

    def test_apply_color(self):
        from PyQt6.QtGui import QColor as QC
        c, lyr = self.select_setup()
        c.pen_color = QC(255, 0, 0, 255)
        self.press_mod(c, 50, 50)
        c.apply_color_to_selected_stroke()
        assert lyr.strokes[0].color == (255, 0, 0, 255)

    def test_switching_to_draw_mode_clears_selection(self):
        c, lyr = self.select_setup()
        self.press_mod(c, 50, 50)
        c.set_vector_pen_mode("draw")
        assert c._vector_selected is None


# ── ハンドル（Option C） ─────────────────────────────────────────────────────

class TestHandles:
    def test_new_stroke_has_none_handles(self):
        s = VectorStroke(points=[(0.0, 0.0), (1.0, 1.0)])
        assert s.handles == [None, None]

    def test_handles_stay_in_sync_with_points(self):
        s = VectorStroke(points=[(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])
        assert len(s.handles) == 3

    def test_copy_is_independent(self):
        s = VectorStroke(points=[(0.0, 0.0), (1.0, 1.0)])
        s.set_handles(0, (1.0, 2.0, 3.0, 4.0))
        t = s.copy()
        t.set_handles(0, None)
        assert s.handles[0] == (1.0, 2.0, 3.0, 4.0)

    def test_handle_points_are_absolute(self):
        s = VectorStroke(points=[(10.0, 10.0), (20.0, 10.0)])
        s.set_handles(0, (-2.0, 0.0, 2.0, 0.0))
        assert s.handle_points(0) == ((8.0, 10.0), (12.0, 10.0))
        assert s.handle_points(1) is None

    def test_bounds_include_handle_tips(self):
        s = VectorStroke(points=[(10.0, 10.0), (20.0, 10.0)], width=1.0)
        plain = s.bounds()
        s.set_handles(0, (0.0, -50.0, 0.0, 50.0))
        assert s.bounds().top() < plain.top()

    def test_scale_scales_handles(self):
        lyr = VectorLayer("v", W, H)
        s = VectorStroke(points=[(10.0, 10.0), (20.0, 10.0)])
        s.set_handles(0, (-2.0, -4.0, 2.0, 4.0))
        lyr.add_stroke(s)
        lyr.scale_strokes(2.0, 3.0)
        assert s.handles[0] == (-4.0, -12.0, 4.0, 12.0)

    def test_translate_leaves_handles_alone(self):
        lyr = VectorLayer("v", W, H)
        s = VectorStroke(points=[(10.0, 10.0), (20.0, 10.0)])
        s.set_handles(0, (-2.0, -4.0, 2.0, 4.0))
        lyr.add_stroke(s)
        lyr.translate_strokes(5.0, 5.0)
        assert s.handles[0] == (-2.0, -4.0, 2.0, 4.0)
        assert s.points[0] == (15.0, 15.0)

    def test_manual_handle_changes_curve(self):
        from vector import stroke_to_path
        s = VectorStroke(points=[(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)])
        # t=0.5 は制御点そのものなので動かない。その手前で測る。
        mid_before = stroke_to_path(s).pointAtPercent(0.25)
        s.set_handles(1, (-20.0, -30.0, 20.0, 30.0))
        mid_after = stroke_to_path(s).pointAtPercent(0.25)
        assert abs(mid_before.y() - mid_after.y()) > 1e-3


# ── 保存データのサニタイズ ───────────────────────────────────────────────────

class TestSafeStrokes:
    """壊れた `.pola` を開いてもクラッシュしないこと。"""

    @staticmethod
    def parse(raw):
        from main import MainWindow
        return MainWindow._safe_strokes(raw)

    def test_not_a_list(self):
        assert self.parse(None) == []
        assert self.parse("abc") == []
        assert self.parse({"points": []}) == []

    def test_skips_bad_items(self):
        out = self.parse([None, 5, "x", {"points": "nope"}, {}])
        assert out == []

    def test_skips_bad_points(self):
        out = self.parse([{"points": [[1, 2], "x", [3], [4, 5]]}])
        assert len(out) == 1
        assert out[0].points == [(1.0, 2.0), (4.0, 5.0)]

    def test_rejects_nan_and_inf(self):
        out = self.parse([{"points": [[float("nan"), 0], [float("inf"), 0], [1, 1]]}])
        assert out[0].points == [(1.0, 1.0)]

    def test_clamps_coordinates(self):
        out = self.parse([{"points": [[1e12, -1e12], [0, 0]]}])
        assert out[0].points[0] == (100000.0, -100000.0)

    def test_bad_width_falls_back(self):
        assert self.parse([{"points": [[0, 0], [1, 1]], "width": "fat"}])[0].width == 5.0
        assert self.parse([{"points": [[0, 0], [1, 1]], "width": -9}])[0].width == 0.1
        assert self.parse([{"points": [[0, 0], [1, 1]],
                            "width": float("nan")}])[0].width == 5.0

    def test_bad_color_falls_back(self):
        assert self.parse([{"points": [[0, 0]], "color": "red"}])[0].color == (0, 0, 0, 255)
        assert self.parse([{"points": [[0, 0]], "color": [1, 2]}])[0].color == (0, 0, 0, 255)

    def test_rgb_gets_opaque_alpha(self):
        assert self.parse([{"points": [[0, 0]], "color": [10, 20, 30]}])[0].color == \
            (10, 20, 30, 255)

    def test_clamps_color(self):
        assert self.parse([{"points": [[0, 0]], "color": [999, -5, 30, 40]}])[0].color == \
            (255, 0, 30, 40)

    def test_good_data_survives(self):
        out = self.parse([{"points": [[1, 2], [3, 4]], "width": 7.5,
                           "color": [1, 2, 3, 4], "smooth": False}])
        assert len(out) == 1
        assert out[0].points == [(1.0, 2.0), (3.0, 4.0)]
        assert out[0].width == 7.5
        assert out[0].color == (1, 2, 3, 4)
        assert out[0].smooth is False
