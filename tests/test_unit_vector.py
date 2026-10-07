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
                     Tool.TEXT, Tool.LASSO_FILL):
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

    def test_delpoint_mode_deletes_on_plain_click(self):
        """「制御点を削除」の役割なら、Alt を押さなくても点が消える。"""
        c, lyr = self.select_setup()
        s = lyr.strokes[0]
        self.press_mod(c, 50, 50)          # 線を選ぶ
        c.set_vector_pen_mode("delpoint")
        self.press_mod(c, 50, 50)          # 真ん中の点をただクリック
        assert len(s.points) == 2
        assert len(s.handles) == 2

    def test_delpoint_mode_refuses_below_two_points(self):
        c, lyr = self.select_setup()
        s = lyr.strokes[0]
        del s.points[2]
        del s.handles[2]
        self.press_mod(c, 50, 50)
        c.set_vector_pen_mode("delpoint")
        self.press_mod(c, 50, 50)
        assert len(s.points) == 2

    def test_delpoint_mode_does_not_drag_points(self):
        """消す役割のときに点を掴んでしまうと、消すつもりが動いてしまう。"""
        c, lyr = self.select_setup()
        s = lyr.strokes[0]
        self.press_mod(c, 50, 50)
        c.set_vector_pen_mode("delpoint")
        self.press_mod(c, 20, 50)          # 端の点（消すと2点になる）
        assert c._vector_drag is None

    def test_delpoint_mode_keeps_selection_visible(self):
        """役割を変えただけでは選択は外れない（続けて点を消せるように）。"""
        c, lyr = self.select_setup()
        s = lyr.strokes[0]
        self.press_mod(c, 50, 50)
        c.set_vector_pen_mode("delpoint")
        assert c._vector_selected is s

    def test_delpoint_mode_can_still_select_another_line(self):
        c, lyr = self.select_setup()
        other = VectorStroke(points=[(20.0, 20.0), (80.0, 20.0)], width=6.0)
        lyr.add_stroke(other)
        c.set_vector_pen_mode("delpoint")
        self.press_mod(c, 50, 20)
        assert c._vector_selected is other

    def test_unknown_mode_falls_back_to_draw(self):
        c, lyr = self.select_setup()
        c.set_vector_pen_mode("なにこれ")
        assert c.vector_pen_mode == "draw"

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


# ── 消去 ─────────────────────────────────────────────────────────────────────

class TestEraseGeometry:
    """交点まで消す計算。線の形はそのままに、区間だけ落ちること。"""

    @staticmethod
    def cross():
        """横線1本を縦線2本が x=30 と x=70 で横切る形。"""
        h = VectorStroke(points=[(0.0, 50.0), (100.0, 50.0)], width=4.0)
        v1 = VectorStroke(points=[(30.0, 0.0), (30.0, 100.0)], width=4.0)
        v2 = VectorStroke(points=[(70.0, 0.0), (70.0, 100.0)], width=4.0)
        return h, [h, v1, v2]

    def test_cut_positions_finds_both_crossings(self):
        from vector import cut_positions
        h, all_ = self.cross()
        cuts = cut_positions(h, all_)
        assert len(cuts) == 2
        assert cuts[0] == pytest.approx(30.0, abs=1.0)
        assert cuts[1] == pytest.approx(70.0, abs=1.0)

    def test_middle_click_splits_into_two(self):
        from vector import erase_between_cuts
        h, all_ = self.cross()
        out = erase_between_cuts(h, 50.0, 50.0, all_, 6.0)
        assert len(out) == 2
        assert out[0].points[0][0] == pytest.approx(0.0, abs=1.0)
        assert out[0].points[-1][0] == pytest.approx(30.0, abs=1.0)
        assert out[1].points[0][0] == pytest.approx(70.0, abs=1.0)
        assert out[1].points[-1][0] == pytest.approx(100.0, abs=1.0)

    def test_end_click_leaves_one_piece(self):
        from vector import erase_between_cuts
        h, all_ = self.cross()
        out = erase_between_cuts(h, 10.0, 50.0, all_, 6.0)
        assert len(out) == 1
        assert out[0].points[0][0] == pytest.approx(30.0, abs=1.0)
        assert out[0].points[-1][0] == pytest.approx(100.0, abs=1.0)

    def test_no_intersection_removes_whole_line(self):
        from vector import erase_between_cuts
        lone = VectorStroke(points=[(0.0, 10.0), (100.0, 10.0)], width=4.0)
        assert erase_between_cuts(lone, 50.0, 10.0, [lone], 6.0) == []

    def test_far_click_is_a_miss(self):
        from vector import erase_between_cuts
        h, all_ = self.cross()
        assert erase_between_cuts(h, 50.0, 90.0, all_, 6.0) is None

    def test_erase_keeps_the_far_end(self):
        """drop_near_duplicates が終点を落とすと、消した反対側が縮む。"""
        from vector import erase_between_cuts
        h, all_ = self.cross()
        out = erase_between_cuts(h, 50.0, 50.0, all_, 6.0)
        assert out[1].points[-1][0] == pytest.approx(100.0, abs=0.2)

    def test_drop_near_duplicates_keeps_endpoint(self):
        from vector import drop_near_duplicates
        pts = [(0.0, 0.0), (10.0, 0.0), (10.5, 0.0)]
        assert drop_near_duplicates(pts, 2.0)[-1] == (10.5, 0.0)


class TestEraseOnCanvas(TestCanvasIntegration):
    """消しゴムが実際にキャンバスで効くこと。"""

    def setup_strokes(self, lyr):
        lyr.strokes = [
            VectorStroke(points=[(0.0, 50.0), (100.0, 50.0)], width=4.0),
            VectorStroke(points=[(30.0, 0.0), (30.0, 100.0)], width=4.0),
            VectorStroke(points=[(70.0, 0.0), (70.0, 100.0)], width=4.0),
        ]

    def erase(self, c, x, y):
        from PyQt6.QtGui import QMouseEvent
        self.press(c, x, y)
        wp = c._c2w().map(QPointF(x, y))
        c.mouseReleaseEvent(QMouseEvent(
            QEvent.Type.MouseButtonRelease, wp, Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier))

    def test_cut_mode_splits_the_line(self):
        from tools import Tool
        c, lyr = self.make()
        self.setup_strokes(lyr)
        c.tool = Tool.ERASER
        self.erase(c, 50.0, 50.0)
        assert len(lyr.strokes) == 4      # 横線が2本に割れた

    def test_whole_mode_removes_the_line(self):
        from tools import Tool
        c, lyr = self.make()
        self.setup_strokes(lyr)
        c.tool = Tool.ERASER
        c.set_vector_erase_mode("whole")
        self.erase(c, 50.0, 50.0)
        assert len(lyr.strokes) == 2

    def test_miss_leaves_no_history(self):
        from tools import Tool
        c, lyr = self.make()
        self.setup_strokes(lyr)
        c.tool = Tool.ERASER
        c.zoom = 1.0
        before = len(c._history)
        self.erase(c, 50.0, 95.0)
        assert len(lyr.strokes) == 3
        assert len(c._history) == before

    def test_undo_restores_the_line(self):
        from tools import Tool
        c, lyr = self.make()
        self.setup_strokes(lyr)
        c.tool = Tool.ERASER
        self.erase(c, 50.0, 50.0)
        c.undo()
        assert len(lyr.strokes) == 3
        assert lyr.strokes[0].points[-1][0] == pytest.approx(100.0)

    def test_erase_mode_default_is_cut(self):
        c, lyr = self.make()
        assert c.vector_erase_mode == "cut"


class TestDuplicateKeepsVector:
    """複製してもベクターのままであること。

    ここが Layer を作ってしまうと、複製した瞬間に絵だけのラスターになり、
    二度と線を編集できなくなる（見た目は同じなので気づきにくい）。
    """

    @staticmethod
    def panel_with(layer):
        from layer import LayerStack
        from layer_panel import LayerPanel
        ls = LayerStack(W, H)
        ls.add("レイヤー1")
        ls.layers.insert(0, layer)
        ls.set_active_path([0])
        panel = LayerPanel(ls)
        return panel, ls

    @staticmethod
    def vector_layer():
        lyr = VectorLayer("v", W, H)
        lyr.add_stroke(VectorStroke(points=[(0.0, 50.0), (100.0, 50.0)],
                                    width=7.0, color=(200, 10, 10, 255)))
        return lyr

    def test_duplicate_stays_vector(self):
        src = self.vector_layer()
        panel, ls = self.panel_with(src)
        panel.refresh()
        panel._duplicate()
        copy = ls.layers[0]
        assert isinstance(copy, VectorLayer)
        assert copy.is_vector is True
        assert len(copy.strokes) == 1
        assert copy.strokes[0].points == [(0.0, 50.0), (100.0, 50.0)]
        assert copy.strokes[0].width == 7.0
        assert copy.strokes[0].color == (200, 10, 10, 255)

    def test_duplicate_strokes_are_independent(self):
        """片方を直してももう片方が変わらないこと。"""
        src = self.vector_layer()
        panel, ls = self.panel_with(src)
        panel.refresh()
        panel._duplicate()
        copy = ls.layers[0]
        copy.strokes[0].width = 30.0
        assert src.strokes[0].width == 7.0

    def test_duplicate_keeps_layer_props(self):
        src = self.vector_layer()
        src.opacity = 50
        src.blend_mode = "multiply"
        panel, ls = self.panel_with(src)
        panel.refresh()
        panel._duplicate()
        copy = ls.layers[0]
        assert copy.opacity == 50
        assert copy.blend_mode == "multiply"

    def test_duplicate_renders_from_its_own_strokes(self):
        """複製直後に線を変えたら、絵もついてくること。"""
        src = self.vector_layer()
        panel, ls = self.panel_with(src)
        panel.refresh()
        panel._duplicate()
        copy = ls.layers[0]
        before = copy.image.cacheKey()
        copy.strokes[0].width = 40.0
        copy.mark_dirty()
        assert copy.image.cacheKey() != before

    def test_duplicate_group_keeps_vector_child(self):
        """フォルダごと複製しても、中のベクターがラスターにならないこと。"""
        from layer import GroupLayer
        g = GroupLayer("g", W, H)
        g.children.append(self.vector_layer())
        panel, ls = self.panel_with(g)
        panel.refresh()
        panel._duplicate()
        copy = ls.layers[0]
        assert isinstance(copy.children[0], VectorLayer)
        assert len(copy.children[0].strokes) == 1


def _alpha_count(img) -> int:
    """不透明な画素の数。間引いて数える（全画素だと遅い）。"""
    return sum(1 for y in range(0, img.height(), 3)
               for x in range(0, img.width(), 3)
               if img.pixelColor(x, y).alpha() > 0)


class TestRasterizeAndMergeGuard:
    """ベクターをラスタライズする操作と、統合でベクターを失う前の確認。

    どちらも「黙って編集できなくなる」のを防ぐためのもの。見た目は
    同じまま線だけが消えるので、気づかないまま作業が進むのが一番困る。
    """

    @staticmethod
    def win_with_vector(strokes=1):
        from main import MainWindow
        w = MainWindow()
        lyr = VectorLayer("ベク", w.layer_stack.width, w.layer_stack.height)
        for i in range(strokes):
            lyr.add_stroke(VectorStroke(
                points=[(10.0, 10.0 + i * 20), (90.0, 60.0 + i * 20)],
                width=8.0, color=(255, 0, 0, 255)))
        w.layer_stack.layers.insert(0, lyr)
        w.layer_stack.set_active_path([0])
        return w, lyr

    @staticmethod
    def answer(monkeypatch, value, log=None):
        """確認ダイアログの答えを決め打ちする。log に呼び出しを記録する。"""
        from PyQt6.QtWidgets import QMessageBox
        import main as m

        def fake(parent, title, text, *a, **k):
            if log is not None:
                log.append(title)
            return value
        monkeypatch.setattr(m.QMessageBox, "question", staticmethod(fake))
        return QMessageBox

    def test_rasterize_replaces_vector_with_raster(self, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        from layer import Layer
        w, lyr = self.win_with_vector()
        self.answer(monkeypatch, QMessageBox.StandardButton.Yes)
        w._rasterize_vector()
        got = w.layer_stack.layers[0]
        assert type(got) is Layer          # VectorLayer ではなくなる
        assert got.is_vector is False
        assert got.name == "ベク"

    def test_rasterize_keeps_the_picture(self, monkeypatch):
        """線は捨てても、焼いた絵は残っていること。"""
        from PyQt6.QtWidgets import QMessageBox
        w, lyr = self.win_with_vector()
        before = _alpha_count(lyr.image)
        assert before > 0
        self.answer(monkeypatch, QMessageBox.StandardButton.Yes)
        w._rasterize_vector()
        assert _alpha_count(w.layer_stack.layers[0].image) == before

    def test_rasterize_keeps_layer_properties(self, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        w, lyr = self.win_with_vector()
        lyr.opacity = 128
        lyr.visible = False
        lyr.clipping = True
        self.answer(monkeypatch, QMessageBox.StandardButton.Yes)
        w._rasterize_vector()
        got = w.layer_stack.layers[0]
        assert (got.opacity, got.visible, got.clipping) == (128, False, True)

    def test_rasterize_can_be_cancelled(self, monkeypatch):
        """No を押したら何も起きないこと。"""
        from PyQt6.QtWidgets import QMessageBox
        w, lyr = self.win_with_vector()
        self.answer(monkeypatch, QMessageBox.StandardButton.No)
        w._rasterize_vector()
        assert w.layer_stack.layers[0] is lyr
        assert len(lyr.strokes) == 1

    def test_rasterize_asks_first(self, monkeypatch):
        """確認なしに焼いてしまわないこと。"""
        from PyQt6.QtWidgets import QMessageBox
        log = []
        w, lyr = self.win_with_vector()
        self.answer(monkeypatch, QMessageBox.StandardButton.No, log)
        w._rasterize_vector()
        assert log == ["ベクターをラスタライズ"]

    def test_rasterize_ignores_raster_layer(self, monkeypatch):
        """ふつうのレイヤーを選んでいるときは何もしない（落ちない）こと。"""
        from PyQt6.QtWidgets import QMessageBox
        log = []
        w, lyr = self.win_with_vector()
        w.layer_stack.set_active_path([1])          # ラスター側
        self.answer(monkeypatch, QMessageBox.StandardButton.Yes, log)
        w._rasterize_vector()
        assert log == []                            # 確認すら出さない
        assert isinstance(w.layer_stack.layers[0], VectorLayer)

    def test_rasterize_drops_orphan_history(self, monkeypatch):
        """差し替えで迷子になった履歴が残らないこと。

        履歴は id() でレイヤーを指しているので、捨てないと undo が
        別のレイヤーへ当たる。
        """
        from PyQt6.QtWidgets import QMessageBox
        w, lyr = self.win_with_vector()
        w.canvas._save_history()                    # 古いレイヤーを指す履歴
        self.answer(monkeypatch, QMessageBox.StandardButton.Yes)
        w._rasterize_vector()
        live = w.canvas._all_layer_ids()
        assert all(e[0] == "structure" or e[1] in live
                   for e in w.canvas._history)

    def test_rasterize_is_in_the_menu(self):
        """メニューに項目があること（追記漏れの検出）。"""
        from PyQt6.QtWidgets import QMenu
        from main import MainWindow
        w = MainWindow()
        texts = [a.text() for m in w.menuBar().findChildren(QMenu)
                 for a in m.actions()]
        assert "ベクターをラスタライズ" in texts
        # 効果だけを焼く既存の項目と別物であること
        assert "レイヤーをラスタライズ" in texts

    def test_merge_down_asks_before_losing_vector(self, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        log = []
        w, lyr = self.win_with_vector()
        before = len(w.layer_stack.layers)
        self.answer(monkeypatch, QMessageBox.StandardButton.No, log)
        w._merge_down()
        assert log == ["ベクターレイヤーの統合"]
        assert len(w.layer_stack.layers) == before   # 中止された
        assert w.layer_stack.layers[0] is lyr

    def test_merge_down_proceeds_on_yes(self, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        w, lyr = self.win_with_vector()
        before = len(w.layer_stack.layers)
        self.answer(monkeypatch, QMessageBox.StandardButton.Yes)
        w._merge_down()
        assert len(w.layer_stack.layers) == before - 1

    def test_merge_without_vector_does_not_ask(self, monkeypatch):
        """ベクターが絡まないときに確認を出すと、ただの邪魔になる。"""
        from PyQt6.QtWidgets import QMessageBox
        from main import MainWindow
        log = []
        w = MainWindow()
        w.layer_stack.add("レイヤー2")
        w.layer_stack.set_active_path([0])
        before = len(w.layer_stack.layers)
        self.answer(monkeypatch, QMessageBox.StandardButton.Yes, log)
        w._merge_down()
        assert log == []
        assert len(w.layer_stack.layers) == before - 1

    def test_merge_all_visible_asks(self, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        log = []
        w, lyr = self.win_with_vector()
        self.answer(monkeypatch, QMessageBox.StandardButton.No, log)
        w._merge_all_visible()
        assert log == ["ベクターレイヤーの統合"]
        assert w.layer_stack.layers[0] is lyr

    def test_merge_marked_asks(self, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        log = []
        w, lyr = self.win_with_vector()
        lyr.merge_marked = True
        w.layer_stack.layers[1].merge_marked = True
        self.answer(monkeypatch, QMessageBox.StandardButton.No, log)
        w.layer_panel.refresh()
        w._merge_marked_layers()
        assert log == ["ベクターレイヤーの統合"]
        assert w.layer_stack.layers[0] is lyr

    def test_hidden_vector_in_group_is_found(self, monkeypatch):
        """グループの中のベクターも見落とさないこと。

        入れ子をたどらないと、フォルダに入れた線が黙って焼かれる。
        """
        from PyQt6.QtWidgets import QMessageBox
        from layer import GroupLayer
        from main import MainWindow
        w = MainWindow()
        grp = GroupLayer("フォルダ", w.layer_stack.width, w.layer_stack.height)
        inner = VectorLayer("奥のベク", w.layer_stack.width,
                            w.layer_stack.height)
        grp.children.append(inner)
        assert w._vectors_in([grp]) == [inner]

    def test_confirm_lists_vector_names(self, monkeypatch):
        """どのレイヤーが対象かが文面に出ること。"""
        import main as m
        from PyQt6.QtWidgets import QMessageBox
        seen = []

        def fake(parent, title, text, *a, **k):
            seen.append(text)
            return QMessageBox.StandardButton.No
        monkeypatch.setattr(m.QMessageBox, "question", staticmethod(fake))
        w, lyr = self.win_with_vector()
        w._merge_down()
        assert "ベク" in seen[0]
