"""調査で見つかったバグの再発防止。

どれも「操作したのに黙って何も起きない／絵が消える」系で、
メニューのショートカットやロック、ベクターレイヤー特有の経路で起きていた。
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if not os.environ.get("PAINTPOLA_TEST_GUI"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt, QPointF, QEvent
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog, QFileDialog

app = QApplication.instance() or QApplication(sys.argv)

import main as M
from main import MainWindow, TransformPercentDialog
from tools import Tool
from vector import VectorLayer, VectorStroke


@pytest.fixture
def win():
    w = MainWindow()
    w.show()
    w.activateWindow()
    QApplication.processEvents()
    yield w
    w.close()


@pytest.fixture
def say_yes(monkeypatch):
    monkeypatch.setattr(QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)


def add_vector(w, y=100.0):
    ls = w.layer_stack
    v = VectorLayer("v", ls.width, ls.height)
    v.add_stroke(VectorStroke(points=[(100.0, y), (400.0, y)], width=20.0))
    ls.layers.append(v)
    ls.active_path = [len(ls.layers) - 1]
    return v


def click(canvas, x, y):
    wp = canvas._c2w().map(QPointF(x, y))
    for typ, btns in ((QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
                      (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton)):
        ev = QMouseEvent(typ, wp, Qt.MouseButton.LeftButton, btns,
                         Qt.KeyboardModifier.NoModifier)
        if typ == QEvent.Type.MouseButtonPress:
            canvas.mousePressEvent(ev)
        else:
            canvas.mouseReleaseEvent(ev)


class TestKeysReachCanvas:
    """メニューのショートカットがキャンバスのキー操作を奪わないこと。"""

    def test_delete_removes_selected_vector_stroke(self, win):
        c = win.canvas
        v = add_vector(win, y=50.0)
        c.tool = Tool.PEN
        c.vector_pen_mode = "select"
        c.setFocus()
        click(c, 250, 50)
        assert c._vector_selected is not None
        QTest.keyClick(c, Qt.Key.Key_Delete)
        assert v.strokes == []

    def test_escape_cancels_path_pick(self, win):
        c = win.canvas
        c.setFocus()
        c._path_pick_active = True
        QTest.keyClick(c, Qt.Key.Key_Escape)
        assert not c._path_pick_active

    def test_escape_cancels_transform(self, win):
        c = win.canvas
        c.setFocus()
        assert c.lift_whole_layer()
        QTest.keyClick(c, Qt.Key.Key_Escape)
        assert c._transform_image is None


class TestLockedLayer:
    def test_arrow_key_does_not_move(self, win):
        c = win.canvas
        layer = win.layer_stack.layers[0]
        layer.locked = True
        c.tool = Tool.MOVE
        c.setFocus()
        QTest.keyClick(c, Qt.Key.Key_Right)
        assert layer.offset_x == 0

    def test_delete_and_cut_keep_pixels(self, win):
        c = win.canvas
        layer = win.layer_stack.layers[0]
        layer.locked = True
        before = layer.image.copy()
        c.select_all()
        c.delete_selection()
        c.cut_selection()
        assert layer.image == before

    def test_cannot_lift(self, win):
        win.layer_stack.layers[0].locked = True
        assert not win.canvas.lift_whole_layer()
        assert win.canvas._transform_image is None


class TestVectorLayer:
    def test_effect_rasterize_keeps_lines(self, win, say_yes):
        """効果の焼き込みをベクターにかけても、次の編集で線が消えない。"""
        v = add_vector(win)
        v.border_enabled = True
        win._rasterize_layer()
        layer = win.layer_stack.active
        assert not layer.is_vector
        assert layer.image.pixelColor(250, 100).alpha() > 0

    def test_transform_scales_strokes(self, win):
        """変形の結果が線データに反映され、描き直しても元に戻らない。"""
        v = add_vector(win)
        c = win.canvas
        assert c.lift_whole_layer()
        c.apply_transform_percentage(50, 50, 0)
        c._commit_transform()
        v.mark_dirty()
        xs = [x for x in range(0, 600, 2) if v.image.pixelColor(x, 100).alpha() > 0]
        assert xs and 150 < min(xs) and max(xs) < 350

    def test_transform_undo_restores_strokes(self, win):
        v = add_vector(win)
        before = [list(s.points) for s in v.strokes]
        c = win.canvas
        c.lift_whole_layer()
        c.apply_transform_percentage(50, 50, 0)
        c._commit_transform()
        c.undo()
        assert [list(s.points) for s in win.layer_stack.active.strokes] == before

    def test_filter_on_vector_requires_conversion(self, win, monkeypatch):
        """フィルターはベクターに直接かけない（断ったら何も変わらない）。"""
        monkeypatch.setattr(QMessageBox, "question",
                            lambda *a, **k: QMessageBox.StandardButton.No)
        v = add_vector(win)
        win._filter_blur()
        assert win.layer_stack.active is v


class TestUndoAndDirty:
    def test_import_as_layer_is_undoable(self, win, monkeypatch, tmp_path):
        path = str(tmp_path / "a.png")
        win.layer_stack.layers[0].image.save(path)
        monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                            staticmethod(lambda *a, **k: ([path], "")))
        n = len(win.layer_stack.layers)
        win._import_as_layer()
        assert len(win.layer_stack.layers) == n + 1
        assert win._dirty
        win.canvas.undo()
        assert len(win.layer_stack.layers) == n

    def test_paste_makes_new_layer_and_is_undoable(self, win):
        c = win.canvas
        n = len(win.layer_stack.layers)
        c.select_all()
        c.copy_selection()
        win._paste()
        assert len(win.layer_stack.layers) == n + 1
        c.undo()
        assert len(win.layer_stack.layers) == n

    def test_dialog_escape_cancels_transform(self, win):
        c = win.canvas
        c.lift_whole_layer()
        d = TransformPercentDialog(c, win)
        d.show()
        d._sx_spin.setValue(50)
        QTest.keyClick(d, Qt.Key.Key_Escape)
        assert c._transform_image is None


class TestNewDocument:
    def test_status_and_anim_frames_reset(self, win, monkeypatch):
        class FakeDlg:
            def __init__(self, *a, **k):
                pass

            def exec(self):
                return QDialog.DialogCode.Accepted

            def values(self):
                return (800, 600)

        monkeypatch.setattr(M, "NewCanvasDialog", FakeDlg)
        win.anim_panel.frames.append(win.layer_stack.composite())
        win._new()
        assert win._status_canvas_size.text() == "800 x 600 px"
        assert len(win.anim_panel.frames) == 0


class TestTransformMenu:
    def test_ctrl_t_lifts_and_switches_tool(self, win):
        win._transform_with_tool()
        assert win.canvas._transform_image is not None
        assert win.canvas.tool == Tool.TRANSFORM

    def test_ctrl_t_with_selection_lifts_once(self, win):
        c = win.canvas
        c.select_all()
        win._transform_with_tool()
        n = len(c._history)
        win._transform_with_tool()
        assert len(c._history) == n

    def test_flip_without_floating_commits(self, win):
        from PyQt6.QtGui import QPainter, QColor
        layer = win.layer_stack.layers[0]
        p = QPainter(layer.image)
        p.fillRect(0, 0, 10, 10, QColor(255, 0, 0))
        p.end()
        w = win.layer_stack.width
        win._transform_flip(True)
        assert win.canvas._transform_image is None
        img = win.layer_stack.composite()
        assert img.pixelColor(w - 5, 5) == QColor(255, 0, 0)
        assert img.pixelColor(5, 5) != QColor(255, 0, 0)


class TestToolKeysWithoutCanvasFocus:
    def test_tool_key_after_clicking_toolbar(self, win):
        """ツールボタンにフォーカスがあってもツールキーが効く。"""
        from PyQt6.QtWidgets import QAbstractButton
        btn = win.toolbar.findChildren(QAbstractButton)[0]
        btn.setFocus()
        QApplication.processEvents()
        QTest.keyClick(btn, Qt.Key.Key_E)
        assert win.canvas.tool == Tool.ERASER

    def test_text_input_keeps_letters(self, win):
        """数値・文字入力欄で打った文字はツール切替にならない。"""
        from PyQt6.QtWidgets import QLineEdit
        edit = QLineEdit(win)
        edit.show()
        edit.setFocus()
        before = win.canvas.tool
        QTest.keyClick(edit, Qt.Key.Key_E)
        assert win.canvas.tool == before
        assert edit.text() == "e"


# ── 機能ごとの不具合調査で見つかったもの ─────────────────────────────────────

from PyQt6.QtCore import QPoint, QRect
from PyQt6.QtGui import QColor, QFont, QImage, QPainter
from layer import Layer, GroupLayer


def new_layer(w, name="L"):
    ls = w.layer_stack
    lyr = Layer(name, ls.width, ls.height)
    # layers[0] が一番上。アプリと同じく、選択中のレイヤーのすぐ上に足す。
    idx = ls.active_path[0] if ls.active_path else 0
    ls.layers.insert(idx, lyr)
    ls.active_path = [idx]
    return lyr


def fill(img, rect, color):
    p = QPainter(img)
    p.fillRect(rect, color)
    p.end()


def drag(canvas, pts):
    def ev(typ, x, y, btns):
        wp = canvas._c2w().map(QPointF(x, y))
        return QMouseEvent(typ, wp, Qt.MouseButton.LeftButton, btns,
                           Qt.KeyboardModifier.NoModifier)
    x0, y0 = pts[0]
    canvas.mousePressEvent(ev(QEvent.Type.MouseButtonPress, x0, y0, Qt.MouseButton.LeftButton))
    for x, y in pts[1:]:
        canvas.mouseMoveEvent(ev(QEvent.Type.MouseMove, x, y, Qt.MouseButton.LeftButton))
    x, y = pts[-1]
    canvas.mouseReleaseEvent(ev(QEvent.Type.MouseButtonRelease, x, y, Qt.MouseButton.NoButton))


class TestHistory:
    def test_undo_skips_entry_of_deleted_layer(self, win):
        """消したレイヤーの履歴が残っていても、他の取り消しが止まらない。"""
        c = win.canvas
        b = new_layer(win, "B")
        c._save_history()
        fill(b.image, QRect(0, 0, 10, 10), QColor(0, 0, 255))
        a = new_layer(win, "A")
        c._save_history()
        fill(a.image, QRect(0, 0, 10, 10), QColor(255, 0, 0))
        win.layer_stack.layers.remove(a)
        win.layer_stack.active_path = [win.layer_stack.layers.index(b)]
        c.undo()
        assert b.image.pixelColor(5, 5).alpha() == 0

    def test_opacity_drag_is_one_undo(self, win):
        lyr = win.layer_stack.active
        n = len(win.canvas._history)
        for v in (200, 150, 100):
            win.layer_panel._opacity.setValue(v)
        assert lyr.opacity == 100
        assert len(win.canvas._history) == n + 1
        win.canvas.undo()
        assert win.layer_stack.active.opacity == 255

    def test_visibility_and_rename_are_undoable(self, win, monkeypatch):
        from PyQt6.QtWidgets import QInputDialog
        lyr = win.layer_stack.active
        win.layer_panel._on_visibility(lyr, False)
        win.canvas.undo()
        assert win.layer_stack.active.visible
        lyr = win.layer_stack.active
        monkeypatch.setattr(QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("新しい名前", True)))
        old = lyr.name
        win.layer_panel._on_rename(lyr)
        assert lyr.name == "新しい名前"
        win.canvas.undo()
        assert win.layer_stack.active.name == old

    def test_move_click_without_drag_adds_no_history(self, win):
        c = win.canvas
        c.tool = Tool.MOVE
        n = len(c._history)
        click(c, 50, 50)
        assert len(c._history) == n
        drag(c, [(50, 50), (60, 50)])
        assert len(c._history) == n + 1


class TestCompositing:
    def test_stacked_clips_do_not_leak_outside_base(self, win):
        win.layer_stack.layers[0].visible = False
        base = new_layer(win, "base")
        fill(base.image, QRect(0, 0, 10, 10), QColor(0, 0, 255))
        clip = new_layer(win, "clip")
        clip.clipping = True
        clip.image.fill(QColor(255, 0, 0))
        clip2 = new_layer(win, "clip2")
        clip2.clipping = True
        clip2.image.fill(QColor(0, 255, 0))
        img = win.layer_stack.composite()
        assert img.pixelColor(5, 5) == QColor(0, 255, 0)
        assert img.pixelColor(50, 50).alpha() == 0

    def test_pass_through_toggle_is_undoable(self, win):
        g = GroupLayer("g", win.layer_stack.width, win.layer_stack.height)
        win.layer_stack.layers.append(g)
        win.layer_stack.active_path = [len(win.layer_stack.layers) - 1]
        win.layer_panel._sync_settings_tab()
        chk = win.layer_panel._pass_check
        assert chk.isEnabled()
        before = g.pass_through
        chk.setChecked(not before)
        assert g.pass_through == (not before)
        win.canvas.undo()
        assert win.layer_stack.active.pass_through == before

    def test_merge_down_into_vector_keeps_lines(self, win):
        add_vector(win)
        top = new_layer(win, "top")
        fill(top.image, QRect(0, 0, 10, 10), QColor(255, 0, 0))
        assert win.layer_stack.merge_down()
        merged = win.layer_stack.active
        assert not merged.is_vector
        def at(x, y):
            return merged.image.pixelColor(x - merged.offset_x, y - merged.offset_y)
        assert at(250, 100).alpha() > 0
        assert at(5, 5) == QColor(255, 0, 0)

    def test_new_layer_names_are_unique(self, win):
        p = win.layer_panel
        p._add()
        p._add()
        names = [l.name for l in win.layer_stack.layers]
        assert len(names) == len(set(names))


class TestTools:
    def test_fill_outside_layer_buffer(self, win):
        lyr = new_layer(win)
        lyr.image = QImage(10, 10, QImage.Format.Format_ARGB32)
        lyr.image.fill(Qt.GlobalColor.transparent)
        c = win.canvas
        c.tool = Tool.FILL
        c.fill_reference_mode = "ref_self"
        c.pen_color = QColor(255, 0, 0)
        click(c, 300, 300)
        assert win.layer_stack.composite().pixelColor(300, 300) == QColor(255, 0, 0)

    def test_text_respects_selection(self, win):
        lyr = new_layer(win)
        c = win.canvas
        c._selection_rect = QRect(0, 0, 5, 5)
        c._lasso_mask = None
        c._text_pos = QPoint(100, 100)
        f = QFont()
        f.setPixelSize(40)
        c.draw_text("■■■", f, QColor(0, 0, 0))
        img = lyr.image
        ox, oy = lyr.offset_x, lyr.offset_y
        assert not any(img.pixelColor(x, y).alpha()
                       for x in range(0, img.width(), 2) for y in range(0, img.height(), 2)
                       if not (x + ox < 5 and y + oy < 5))

    def test_blur_click_respects_selection(self, win):
        lyr = new_layer(win)
        fill(lyr.image, QRect(100, 100, 50, 50), QColor(0, 0, 0))
        before = lyr.image.copy()
        c = win.canvas
        c._selection_rect = QRect(0, 0, 5, 5)
        c._lasso_mask = None
        c.tool = Tool.BLUR
        click(c, 100, 100)
        assert lyr.image == before

    def test_eyedropper_ignores_transparent(self, win):
        win.layer_stack.layers[0].visible = False
        c = win.canvas
        picked = []
        c.color_picked.connect(picked.append)
        c.tool = Tool.EYEDROPPER
        click(c, 50, 50)
        assert picked == []

    def test_symmetry_mirrors_around_canvas_center_on_moved_layer(self, win):
        lyr = new_layer(win)
        lyr.offset_x = 40
        cx = win.layer_stack.width // 2
        # ローカル10 = キャンバス50 → 折り返して 2cx-50 → ローカルへ戻すと -40
        assert win.canvas._mirror_x(QPoint(10, 5)).x() == 2 * cx - 50 - 40


class TestFileAndCanvas:
    def test_anim_frames_survive_save(self, win, tmp_path):
        img = win.layer_stack.composite()
        win.anim_panel.set_frames([img, img], 12)
        path = str(tmp_path / "a.pola")
        win._write_pola(path)
        win.anim_panel.set_frames([], None)
        win._load_pola(path)
        assert len(win.anim_panel.frames) == 2
        assert win.anim_panel.fps == 12

    def test_failed_save_keeps_old_file(self, win, tmp_path, monkeypatch):
        path = str(tmp_path / "a.pola")
        win._write_pola(path)
        good = open(path, "rb").read()
        monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)

        def boom(*a):
            raise OSError("disk full")
        monkeypatch.setattr(win, "_write_anim_frames", boom)
        win._write_pola(path)
        assert open(path, "rb").read() == good
        assert not os.path.exists(path + ".saving")

    def test_resize_keeps_vector_position_with_offset(self, win, monkeypatch):
        v = add_vector(win)
        v.offset_x, v.offset_y = 30, 30
        monkeypatch.setattr(M.ResizeCanvasDialog, "exec",
                            lambda self: QDialog.DialogCode.Accepted)
        monkeypatch.setattr(M.ResizeCanvasDialog, "values",
                            lambda self: (win.layer_stack.width + 100,
                                          win.layer_stack.height, "crop", (0, 0)))
        win._resize_canvas()
        assert v.strokes[0].points[0] == (100.0, 100.0)

    def test_gaussian_blur_has_no_dark_halo(self, win, monkeypatch):
        from PyQt6.QtWidgets import QInputDialog
        lyr = new_layer(win)
        fill(lyr.image, QRect(100, 100, 50, 50), QColor(255, 255, 255))
        monkeypatch.setattr(QInputDialog, "getInt", staticmethod(lambda *a, **k: (5, True)))
        win._filter_blur()
        col = lyr.image.pixelColor(98, 120)
        assert col.alpha() > 0
        assert min(col.red(), col.green(), col.blue()) > 240


class TestVectorSelectionAfterUndo:
    def test_width_change_after_undo_hits_visible_stroke(self, win):
        """取り消し後も同じ線が選ばれたままで、太さ変更が見えている線に効く。"""
        v = add_vector(win, y=50.0)
        c = win.canvas
        c.tool = Tool.PEN
        c.set_vector_pen_mode("select")
        click(c, 250, 50)
        c.set_selected_stroke_width(30)
        c.undo()
        assert any(s is c._vector_selected for s in v.strokes)
        c.set_selected_stroke_width(40)
        assert v.strokes[0].width == 40


class TestMergeDownSelection:
    def test_merged_layer_becomes_active(self, win):
        lower = new_layer(win, "lower")
        new_layer(win, "upper")
        assert win.layer_stack.merge_down()
        assert win.layer_stack.active is lower


class TestLassoFillOptions:
    """囲み内塗りつぶしにもバケツ塗りと同じオプションが効く。"""

    @staticmethod
    def _ring(gap=False):
        import numpy as np
        img = QImage(60, 60, QImage.Format.Format_ARGB32_Premultiplied)
        img.fill(0)
        p = QPainter(img)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0))
        p.drawRect(10, 10, 40, 2)   # 上
        p.drawRect(10, 48, 40, 2)   # 下
        p.drawRect(10, 10, 2, 40)   # 左
        p.drawRect(48, 10, 2, 40)   # 右
        if gap:
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.drawRect(29, 48, 2, 2)  # 下辺に 2px の切れ目
        p.end()
        area = np.ones((60, 60), np.uint8)
        area[0, :] = area[-1, :] = area[:, 0] = area[:, -1] = 0
        return img, area

    def test_expand_paints_under_line(self):
        from canvas import _fill_closed_regions_in_area
        red = QColor(255, 0, 0)
        img, area = self._ring()
        assert _fill_closed_regions_in_area(img, area, red, None) == 1
        assert QColor.fromRgba(img.pixel(11, 30)) == QColor(0, 0, 0)  # 線の上は塗らない
        img, area = self._ring()
        _fill_closed_regions_in_area(img, area, red, None, expand=2)
        assert QColor.fromRgba(img.pixel(11, 30)) == red              # 線の下まで広がる
        assert QColor.fromRgba(img.pixel(5, 30)).alpha() == 0         # 外側には漏れない

    def test_close_gap_fills_broken_ring(self):
        from canvas import _fill_closed_regions_in_area
        red = QColor(255, 0, 0)
        img, area = self._ring(gap=True)
        assert _fill_closed_regions_in_area(img, area, red, None) == 0
        img, area = self._ring(gap=True)
        assert _fill_closed_regions_in_area(img, area, red, None, close_gap=2) == 1
        assert QColor.fromRgba(img.pixel(13, 13)) == red              # 角まで痩せずに塗る
        assert QColor.fromRgba(img.pixel(30, 55)).alpha() == 0        # 切れ目の外へは漏れない

    def test_canvas_passes_fill_options(self, win, monkeypatch):
        import canvas as C
        got = {}
        def fake(image, area, color, ref, *args):
            got["args"] = args
            return 1
        monkeypatch.setattr(C, "_fill_closed_regions_in_area", fake)
        c = win.canvas
        c.fill_expand, c.fill_close_gap = 3, 4
        c.fill_line_sensitivity, c.fill_reference_mode = 100, "ref_self"
        c.tool = Tool.LASSO_FILL
        drag(c, [(20, 20), (80, 20), (80, 80), (20, 80)])
        assert got["args"] == (3, 4, C._sensitivity_to_threshold(100), True)

    def test_options_panel_shows_fill_settings(self, win):
        from tool_options_panel import _SliderSpin
        win.tool_options.set_tool(Tool.LASSO_FILL)
        keys = {getattr(w, "_opt_key", "") for w in win.tool_options.findChildren(_SliderSpin)}
        assert {"fill_expand", "fill_close_gap", "fill_line_sensitivity"} <= keys


class TestStabilization:
    """ペン・消しゴムの手ブレ補正（指やマウスで描くときのガタつき対策）。"""

    @staticmethod
    def _zigzag():
        # 横にまっすぐ引いたつもりで、上下に 12px ずつブレている
        return [(100 + i * 4, 100 if i % 2 == 0 else 112) for i in range(50)]

    @staticmethod
    def _painted_rows(img, x0, x1):
        ys = [y for y in range(60, 160)
              if any(QColor.fromRgba(img.pixel(x, y)).alpha() > 0 for x in range(x0, x1))]
        return min(ys), max(ys)

    def _draw(self, win, level, pts, tool=Tool.PEN):
        c = win.canvas
        lyr = new_layer(win)
        c.set_stabilization(level)
        c.pen_size = 2
        c.tool = tool
        drag(c, pts)
        return c, lyr

    def test_zero_follows_hand_exactly(self, win):
        c, lyr = self._draw(win, 0, self._zigzag())
        lo, hi = self._painted_rows(lyr.image, 150, 250)
        assert hi - lo >= 10  # 表示倍率で数px丸められるので幅で見る

    def test_high_value_smooths_jitter(self, win):
        c, lyr = self._draw(win, 10, self._zigzag())
        lo, hi = self._painted_rows(lyr.image, 150, 250)
        assert hi - lo <= 6

    def test_line_reaches_release_point(self, win):
        """補正で遅れた分も、離したときに指の位置まで追いつく。"""
        pts = [(100 + i * 10, 100) for i in range(21)]  # 100 → 300
        c, lyr = self._draw(win, 30, pts)
        assert QColor.fromRgba(lyr.image.pixel(299 - lyr.offset_x, 100 - lyr.offset_y)).alpha() > 0

    def test_eraser_is_smoothed_too(self, win):
        c = win.canvas
        lyr = new_layer(win)
        fill(lyr.image, lyr.image.rect(), QColor(0, 0, 0))
        c.set_stabilization(10)
        c.eraser_size = 2
        c.tool = Tool.ERASER
        drag(c, self._zigzag())
        # ブレの山の位置は消えずに残る
        assert QColor.fromRgba(lyr.image.pixel(200, 113)).alpha() > 0

    def test_vector_pen_is_smoothed(self, win):
        v = add_vector(win)
        v.strokes.clear()
        c = win.canvas
        c.set_stabilization(10)
        c.tool = Tool.PEN
        c.set_vector_pen_mode("draw")
        drag(c, self._zigzag())
        pts = v.strokes[-1].points
        # 確定時に点が間引かれるので、両端を除いた点の上下の幅で見る
        mid = [y for x, y in pts[1:-1]]
        assert max(mid) - min(mid) <= 6
        assert pts[-1][0] == pytest.approx(296.0, abs=2.0)

    def test_option_shown_and_saved(self, win):
        from tool_options_panel import _SliderSpin
        saved = {}
        class FakeSettings:
            def setValue(self, k, v): saved[k] = v
        win._settings = FakeSettings()
        for tool in (Tool.PEN, Tool.ERASER):
            win.tool_options.set_tool(tool, stabilization=7)
            spins = [w for w in win.tool_options.findChildren(_SliderSpin)
                     if getattr(w, "_opt_key", "") == "stabilization"]
            assert spins
        win.tool_options.stabilization_changed.emit(12)
        assert win.canvas.stabilization == 12
        assert win.canvas._stabilizer.smooth == 13
        assert saved == {"stabilization": 12}

    def test_brush_change_keeps_strength(self, win):
        win.canvas.set_stabilization(20)
        win.canvas.set_brush("soft")
        assert win.canvas._stabilizer.smooth == 21


class TestTaper:
    """入り抜き: 描き終わると線の始め・終わりが指定の細さまで細くなる。"""

    @staticmethod
    def _thick(lyr, x, erased=False):
        """キャンバス座標 x の列で、描かれている（消されている）縦の幅。"""
        lx = x - lyr.offset_x
        n = 0
        for y in range(lyr.image.height()):
            a = QColor.fromRgba(lyr.image.pixel(lx, y)).alpha()
            n += (a == 0) if erased else (a > 128)
        return n

    def _setup(self, win, tool=Tool.PEN, **t):
        c = win.canvas
        lyr = new_layer(win)
        c.set_stabilization(0)
        c.pen_size = 20
        c.eraser_size = 20
        c.tool = tool
        key = "pen" if tool == Tool.PEN else "eraser"
        for f, v in t.items():
            c.set_taper(key, f, v)
        return c, lyr

    LINE = [(100 + i * 10, 200) for i in range(31)]  # 100 → 400

    def test_width_function(self):
        from canvas import _taper_width
        assert _taper_width(0, 300, 20, 50, 50, 0) == 1.0
        assert _taper_width(150, 300, 20, 50, 50, 0) == 20.0
        assert _taper_width(300, 300, 20, 50, 50, 0) == 1.0
        assert _taper_width(300, 300, 20, 0, 50, 50) == 10.0
        # 入り＋抜きより短い線でも、真ん中は太さが出る
        assert _taper_width(30, 60, 20, 100, 100, 0) == 20.0
        # 描いている途中（終わりが未定）は抜きを効かせない
        assert _taper_width(300, None, 20, 0, 50, 0) == 20.0

    def test_out_thins_end_after_release(self, win):
        c, lyr = self._setup(win, out=100)
        drag(c, self.LINE)
        mid, end = self._thick(lyr, 250), self._thick(lyr, 390)
        assert mid >= 18
        assert end <= mid // 3
        assert self._thick(lyr, 110) >= 18  # 入りは 0 なので始まりは太いまま

    def test_in_thins_start(self, win):
        c, lyr = self._setup(win, **{"in": 100})
        drag(c, self.LINE)
        assert self._thick(lyr, 110) <= self._thick(lyr, 250) // 3
        assert self._thick(lyr, 390) >= 18

    def test_tip_percent(self, win):
        c, lyr = self._setup(win, out=100, tip=50)
        drag(c, self.LINE)
        assert 8 <= self._thick(lyr, 398) <= 14

    def test_eraser_taper(self, win):
        c, lyr = self._setup(win, Tool.ERASER, out=100)
        fill(lyr.image, lyr.image.rect(), QColor(0, 0, 0))
        drag(c, self.LINE)
        mid, end = self._thick(lyr, 250, True), self._thick(lyr, 390, True)
        assert mid >= 18 and end <= mid // 3

    def test_undo_is_one_step(self, win):
        c, lyr = self._setup(win, out=100)
        drag(c, self.LINE)
        c.undo()
        assert self._thick(lyr, 250) == 0

    def test_keeps_existing_art_when_layer_grows(self, win):
        """描いている途中にレイヤーが広がっても、元の絵の位置がずれない。"""
        c, lyr = self._setup(win, out=50)
        fill(lyr.image, QRect(50, 50, 4, 4), QColor(255, 0, 0))
        w = win.layer_stack.width
        drag(c, [(w - 60 + i * 10, 300) for i in range(12)])  # キャンバスの外まで
        px = QColor.fromRgba(lyr.image.pixel(51 - lyr.offset_x, 51 - lyr.offset_y))
        assert px.red() == 255 and px.alpha() == 255

    def test_selection_still_clips(self, win):
        c, lyr = self._setup(win, out=100)
        c._selection_rect = QRect(0, 0, 250, 1000)
        drag(c, self.LINE)
        assert self._thick(lyr, 200) > 0
        assert self._thick(lyr, 300) == 0

    def test_no_taper_keeps_old_path(self, win):
        c, lyr = self._setup(win)
        drag(c, self.LINE)
        assert c._taper_stroke is None
        assert self._thick(lyr, 390) >= 18

    def test_options_per_tool_and_saved(self, win):
        from tool_options_panel import _SliderSpin
        saved = {}
        class FakeSettings:
            def setValue(self, k, v): saved[k] = v
        win._settings = FakeSettings()
        win._on_tool_change(Tool.PEN)
        keys = {getattr(w, "_opt_key", "") for w in win.tool_options.findChildren(_SliderSpin)
                if w.isVisibleTo(win.tool_options)}
        assert {"taper_in", "taper_out", "taper_tip"} <= keys
        win.tool_options.taper_changed.emit("out", 80)
        win._on_tool_change(Tool.ERASER)
        win.tool_options.taper_changed.emit("out", 30)
        assert win.canvas.taper["pen"]["out"] == 80
        assert win.canvas.taper["eraser"]["out"] == 30
        assert saved == {"taper/pen/out": 80, "taper/eraser/out": 30}
