"""コンポーネントテスト: Canvas ウィジェット（描画ロジック・状態管理）"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor, QMouseEvent, QPainter, QKeyEvent, QPen
from PyQt6.QtCore import Qt, QPoint, QPointF, QEvent

app = QApplication.instance() or QApplication(sys.argv)

from layer import LayerStack, Layer
from canvas import Canvas
from tools import Tool

W, H = 200, 200


def px(image: QImage, x: int, y: int) -> QColor:
    return QColor.fromRgba(image.pixel(x, y))


@pytest.fixture
def canvas():
    ls = LayerStack(W, H)
    ls.add("レイヤー1")
    c = Canvas(ls)
    c.resize(W, H)
    c.show()
    return c


def _press(c, x, y, btn=Qt.MouseButton.LeftButton):
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y), QPointF(x, y),
                     btn, btn, Qt.KeyboardModifier.NoModifier)
    c.mousePressEvent(ev)


def _move(c, x, y):
    ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(x, y), QPointF(x, y),
                     Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier)
    c.mouseMoveEvent(ev)


def _release(c, x, y, btn=Qt.MouseButton.LeftButton):
    ev = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(x, y), QPointF(x, y),
                     btn, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    c.mouseReleaseEvent(ev)


# ── 初期状態 ─────────────────────────────────────────────────────────────────

class TestCanvasInit:
    def test_default_tool_is_pen(self, canvas):
        assert canvas.tool == Tool.PEN

    def test_default_pen_color_black(self, canvas):
        assert canvas.pen_color == QColor(0, 0, 0, 255)

    def test_drawing_false_initially(self, canvas):
        assert canvas._drawing is False

    def test_no_selection_initially(self, canvas):
        assert canvas._selection_rect is None
        assert canvas._transform_image is None


# ── undo / redo ───────────────────────────────────────────────────────────────

class TestUndoRedo:
    def test_undo_restores_image(self, canvas):
        layer = canvas.layer_stack.active
        canvas._save_history()
        p = QPainter(layer.image)
        p.fillRect(0, 0, W, H, QColor(255, 0, 0, 255))
        p.end()
        red_pixel = layer.image.pixel(50, 50)
        canvas.undo()
        assert layer.image.pixel(50, 50) != red_pixel

    def test_redo_reapplies_change(self, canvas):
        layer = canvas.layer_stack.active
        canvas._save_history()
        p = QPainter(layer.image)
        p.fillRect(0, 0, W, H, QColor(0, 0, 255, 255))
        p.end()
        blue_pixel = layer.image.pixel(50, 50)
        canvas.undo()
        canvas.redo()
        assert layer.image.pixel(50, 50) == blue_pixel

    def test_undo_empty_does_not_crash(self, canvas):
        canvas._history.clear()
        canvas.undo()

    def test_redo_empty_does_not_crash(self, canvas):
        canvas._redo_stack.clear()
        canvas.redo()

    def test_new_action_clears_redo(self, canvas):
        layer = canvas.layer_stack.active
        canvas._save_history()
        canvas.undo()
        assert len(canvas._redo_stack) > 0
        canvas._save_history()
        assert len(canvas._redo_stack) == 0

    def test_history_limit(self, canvas):
        from canvas import HISTORY_LIMIT
        for _ in range(HISTORY_LIMIT + 10):
            canvas._save_history()
        assert len(canvas._history) <= HISTORY_LIMIT

    def test_purge_orphan_removes_deleted_layer(self, canvas):
        ls = canvas.layer_stack
        new_layer = ls.add("temp")
        lid = id(new_layer)
        canvas._history.append(("pixel", lid, new_layer.image.copy()))
        ls.remove(0)
        canvas.purge_orphan_history()
        ids_in_history = [e[1] for e in canvas._history if e[0] == "pixel"]
        assert lid not in ids_in_history

    def test_structure_undo_restores_added_layer(self, canvas):
        ls = canvas.layer_stack
        orig_count = len(ls.layers)
        canvas.save_structure_history()
        ls.add("extra")
        assert len(ls.layers) == orig_count + 1
        canvas.undo()
        assert len(ls.layers) == orig_count

    def test_structure_undo_restores_removed_layer(self, canvas):
        ls = canvas.layer_stack
        ls.add("second")
        orig_count = len(ls.layers)
        canvas.save_structure_history()
        ls.remove(0)
        assert len(ls.layers) == orig_count - 1
        canvas.undo()
        assert len(ls.layers) == orig_count

    def test_structure_redo(self, canvas):
        ls = canvas.layer_stack
        canvas.save_structure_history()
        ls.add("extra")
        assert len(ls.layers) == 2
        canvas.undo()
        assert len(ls.layers) == 1
        canvas.redo()
        assert len(ls.layers) == 2


# ── select_mode ───────────────────────────────────────────────────────────────

class TestSelectMode:
    def test_default_select_mode(self, canvas):
        assert canvas.select_mode == "select"

    def test_set_select_mode(self, canvas):
        canvas.select_mode = "transform"
        assert canvas.select_mode == "transform"


# ── reset_state ───────────────────────────────────────────────────────────────

class TestResetState:
    def test_reset_clears_drawing(self, canvas):
        canvas._drawing = True
        canvas.reset_state()
        assert canvas._drawing is False

    def test_reset_clears_selection(self, canvas):
        from PyQt6.QtCore import QRect
        canvas._selection_rect = QRect(10, 10, 50, 50)
        canvas.reset_state()
        assert canvas._selection_rect is None

    def test_reset_clears_lasso(self, canvas):
        canvas._lasso_points = [QPoint(1, 1), QPoint(2, 2)]
        canvas.reset_state()
        assert canvas._lasso_points == []


# ── ペン描画（マウスイベント経由） ──────────────────────────────────────────

class TestPenDraw:
    def test_pen_press_sets_drawing(self, canvas):
        canvas.tool = Tool.PEN
        _press(canvas, 100, 100)
        assert canvas._drawing is True

    def test_pen_release_clears_drawing(self, canvas):
        canvas.tool = Tool.PEN
        _press(canvas, 100, 100)
        _release(canvas, 100, 100)
        assert canvas._drawing is False

    def test_pen_stamp_draws_pixel(self, canvas):
        """press 時の stamp で点が描かれる (RoundBrush.stamp 修正後)。"""
        layer = canvas.layer_stack.active
        canvas.tool = Tool.PEN
        canvas.pen_color = QColor(255, 0, 0, 255)
        canvas.pen_size = 20
        cp = canvas._widget_to_canvas(QPoint(100, 100))
        _press(canvas, 100, 100)
        _release(canvas, 100, 100)
        found = any(
            px(layer.image, x, y).red() > 200 and px(layer.image, x, y).alpha() > 200
            for x in range(max(0, cp.x()-15), min(W, cp.x()+15))
            for y in range(max(0, cp.y()-15), min(H, cp.y()+15))
        )
        assert found, "ペン押下で点が描かれていない"

    def test_pen_stroke_draws_line(self, canvas):
        """drag でストロークが描かれる。"""
        layer = canvas.layer_stack.active
        canvas.tool = Tool.PEN
        canvas.pen_color = QColor(0, 0, 255, 255)
        canvas.pen_size = 10
        _press(canvas, 50, 100)
        _move(canvas, 80, 100)
        _move(canvas, 110, 100)
        _release(canvas, 110, 100)
        cp_start = canvas._widget_to_canvas(QPoint(50, 100))
        cp_end = canvas._widget_to_canvas(QPoint(110, 100))
        found = any(
            px(layer.image, x, y).blue() > 200 and px(layer.image, x, y).alpha() > 200
            for x in range(max(0, cp_start.x()), min(W, cp_end.x()+5))
            for y in range(max(0, cp_start.y()-8), min(H, cp_start.y()+8))
        )
        assert found, "ペンストロークが描かれていない"

    def test_pen_adds_history(self, canvas):
        canvas.tool = Tool.PEN
        before_len = len(canvas._history)
        _press(canvas, 100, 100)
        assert len(canvas._history) > before_len


# ── 消しゴム ─────────────────────────────────────────────────────────────────

class TestEraser:
    def test_eraser_removes_pixel(self, canvas):
        layer = canvas.layer_stack.active
        p = QPainter(layer.image)
        p.fillRect(85, 85, 30, 30, QColor(0, 0, 255, 255))
        p.end()
        canvas.tool = Tool.ERASER
        canvas.eraser_size = 40
        cp = canvas._widget_to_canvas(QPoint(100, 100))
        _press(canvas, 100, 100)
        _release(canvas, 100, 100)
        c = px(layer.image, cp.x(), cp.y())
        assert c.alpha() == 0, f"消しゴム後もピクセルが残っている (alpha={c.alpha()})"


# ── 矩形選択 ─────────────────────────────────────────────────────────────────

class TestSelectRect:
    def test_drag_creates_selection(self, canvas):
        canvas.tool = Tool.SELECT_RECT
        canvas.select_mode = "select"
        _press(canvas, 20, 20)
        _move(canvas, 80, 80)
        _release(canvas, 80, 80)
        assert canvas._selection_rect is not None

    def test_click_no_selection(self, canvas):
        """ドラッグなしのクリックは選択なし。"""
        canvas.tool = Tool.SELECT_RECT
        canvas.select_mode = "select"
        _press(canvas, 50, 50)
        _release(canvas, 50, 50)
        assert canvas._selection_rect is None

    def test_deselect_clears_selection(self, canvas):
        from PyQt6.QtCore import QRect
        canvas._selection_rect = QRect(10, 10, 50, 50)
        canvas.deselect()
        assert canvas._selection_rect is None

    def test_selection_rect_is_normalized(self, canvas):
        """右→左ドラッグでも正規化された矩形になる。"""
        canvas.tool = Tool.SELECT_RECT
        canvas.select_mode = "select"
        _press(canvas, 120, 120)
        _move(canvas, 40, 40)
        _release(canvas, 40, 40)
        sel = canvas._selection_rect
        if sel:
            assert sel.width() > 0 and sel.height() > 0


# ── グループ描画禁止メッセージ ────────────────────────────────────────────────

class TestGroupLayerDrawBlocked:
    def test_drawing_on_group_emits_status(self, canvas):
        ls = canvas.layer_stack
        grp = ls.add_group("G")
        ls.set_active(0, -1)
        messages = []
        canvas.status_message.connect(messages.append)
        canvas.tool = Tool.PEN
        _press(canvas, 100, 100)
        assert any("グループ" in m for m in messages)

    def test_drawing_on_group_does_not_modify_children(self, canvas):
        ls = canvas.layer_stack
        grp = ls.add_group("G")
        child = Layer("c", W, H)
        grp.children.append(child)
        ls.set_active(0, -1)
        canvas.tool = Tool.PEN
        canvas.pen_color = QColor(255, 0, 0, 255)
        _press(canvas, 100, 100)
        cp = canvas._widget_to_canvas(QPoint(100, 100))
        assert px(child.image, cp.x(), cp.y()).alpha() == 0


# ── alt eyedropper ───────────────────────────────────────────────────────────

class TestAltEyedropper:
    def test_alt_key_switches_to_eyedropper(self, canvas):
        canvas.tool = Tool.PEN
        ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Alt,
                       Qt.KeyboardModifier.AltModifier)
        canvas.keyPressEvent(ev)
        assert canvas._alt_eyedropper is True
        assert canvas.tool == Tool.EYEDROPPER

    def test_alt_release_restores_tool(self, canvas):
        canvas.tool = Tool.PEN
        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Alt,
                          Qt.KeyboardModifier.AltModifier)
        canvas.keyPressEvent(press)
        release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Alt,
                            Qt.KeyboardModifier.AltModifier)
        canvas.keyReleaseEvent(release)
        assert canvas._alt_eyedropper is False
        assert canvas.tool == Tool.PEN

    def test_alt_release_resets_drawing_flag(self, canvas):
        canvas.tool = Tool.PEN
        canvas._drawing = True
        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Alt,
                          Qt.KeyboardModifier.AltModifier)
        canvas.keyPressEvent(press)
        release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Alt,
                            Qt.KeyboardModifier.AltModifier)
        canvas.keyReleaseEvent(release)
        assert canvas._drawing is False

    def test_alt_eyedropper_picks_color(self, canvas):
        """Alt スポイトでクリックした点の色が pen_color に設定される。"""
        layer = canvas.layer_stack.active
        p = QPainter(layer.image)
        p.fillRect(0, 0, W, H, QColor(200, 100, 50, 255))
        p.end()
        canvas.tool = Tool.PEN
        picked = []
        canvas.color_picked.connect(picked.append)
        # Alt 押し
        ev_press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Alt,
                             Qt.KeyboardModifier.AltModifier)
        canvas.keyPressEvent(ev_press)
        # クリックで色取得
        _press(canvas, 100, 100)
        _release(canvas, 100, 100)
        assert len(picked) > 0, "color_picked が emit されていない"


# ── 構造 undo/redo ──────────────────────────────────────────────────────────

class TestStructureUndoRedo:
    def test_undo_add_layer(self, canvas):
        ls = canvas.layer_stack
        orig_count = len(ls.layers)
        canvas.save_structure_history()
        ls.add("new")
        canvas.undo()
        assert len(ls.layers) == orig_count

    def test_undo_remove_layer(self, canvas):
        ls = canvas.layer_stack
        ls.add("second")
        canvas.save_structure_history()
        ls.remove(0)
        canvas.undo()
        assert len(ls.layers) == 2

    def test_undo_preserves_layer_name(self, canvas):
        ls = canvas.layer_stack
        ls.add("named_layer")
        canvas.save_structure_history()
        ls.remove(0)
        canvas.undo()
        names = [l.name for l in ls.layers]
        assert "named_layer" in names

    def test_structure_redo_re_applies(self, canvas):
        ls = canvas.layer_stack
        canvas.save_structure_history()
        ls.add("extra")
        assert len(ls.layers) == 2
        canvas.undo()
        assert len(ls.layers) == 1
        canvas.redo()
        assert len(ls.layers) == 2

    def test_pixel_then_structure_undo_no_crash(self, canvas):
        ls = canvas.layer_stack
        canvas._save_history()
        layer = ls.active
        p = QPainter(layer.image)
        p.fillRect(0, 0, 10, 10, QColor(255, 0, 0))
        p.end()
        canvas.save_structure_history()
        ls.add("extra")
        canvas.undo()
        assert len(ls.layers) == 1
        canvas.undo()  # pixel undo on now-replaced layer — should not crash


# ── 投げなわ ────────────────────────────────────────────────────────────────

class TestLasso:
    def test_lasso_creates_selection(self, canvas):
        canvas.tool = Tool.LASSO
        canvas.select_mode = "select"
        _press(canvas, 50, 50)
        _move(canvas, 100, 50)
        _move(canvas, 100, 100)
        _move(canvas, 50, 100)
        _release(canvas, 50, 100)
        assert canvas._selection_rect is not None

    def test_lasso_stores_path_points(self, canvas):
        canvas.tool = Tool.LASSO
        canvas.select_mode = "select"
        _press(canvas, 50, 50)
        _move(canvas, 100, 50)
        _move(canvas, 100, 100)
        _release(canvas, 100, 100)
        assert len(canvas._lasso_path_points) > 0

    def test_deselect_clears_lasso_path(self, canvas):
        from PyQt6.QtCore import QRect
        canvas._selection_rect = QRect(10, 10, 50, 50)
        canvas._lasso_path_points = [QPoint(10, 10), QPoint(60, 10), QPoint(60, 60)]
        canvas.deselect()
        assert canvas._selection_rect is None
        assert len(canvas._lasso_path_points) == 0

    def test_deselect_clears_lasso_points(self, canvas):
        canvas._lasso_points = [QPoint(1, 1), QPoint(2, 2)]
        canvas.deselect()
        assert len(canvas._lasso_points) == 0


class TestHistoryMemoryLimit:
    """履歴はレイヤー画像の丸ごとコピーを持つため、件数だけで制限すると
    大きなキャンバスで GB 級になりメモリ不足で落ちる。総バイト数でも
    打ち切られることを確認する。"""

    def _big_canvas(self):
        import canvas as canvas_mod
        ls = LayerStack(2500, 2500)
        ls.add("レイヤー1")
        c = canvas_mod.Canvas(ls)
        c.resize(W, H)
        return c

    def test_history_capped_by_memory(self):
        import canvas as canvas_mod
        c = self._big_canvas()
        for _ in range(canvas_mod.HISTORY_LIMIT + 20):
            c._save_history()
        total = sum(c._entry_bytes(e) for e in c._history)
        assert total <= canvas_mod.HISTORY_MEMORY_LIMIT, \
            f"履歴が {total/1e6:.0f}MB まで膨らんでいる"
        assert len(c._history) < canvas_mod.HISTORY_LIMIT, \
            "大きなキャンバスでは件数が上限より減るはず"

    def test_history_keeps_minimum_entries(self):
        import canvas as canvas_mod
        c = self._big_canvas()
        for _ in range(canvas_mod.HISTORY_LIMIT + 20):
            c._save_history()
        assert len(c._history) >= canvas_mod.HISTORY_MIN_ENTRIES, \
            "直近の操作を取り消せる件数は必ず残す"

    def test_small_canvas_keeps_full_depth(self, canvas):
        """小さいキャンバスでは従来どおり件数上限まで残る。"""
        import canvas as canvas_mod
        for _ in range(canvas_mod.HISTORY_LIMIT + 10):
            canvas._save_history()
        assert len(canvas._history) == canvas_mod.HISTORY_LIMIT

    def test_big_canvas_undo_depth_is_practical(self):
        """大きなキャンバスでも実用的な回数を戻せること。

        実際に戻せる回数は件数上限ではなくメモリ上限で決まる。
        ここが小さいと「戻れる数が少ない」という形で表に出るので、
        2500x2500 で最低 30 回は戻せることを固定する。
        """
        import canvas as canvas_mod
        c = self._big_canvas()
        for _ in range(canvas_mod.HISTORY_LIMIT + 20):
            c._save_history()
        assert len(c._history) >= 30, \
            f"2500x2500 で {len(c._history)} 回しか戻せない"

    def test_undo_redo_still_works_after_trim(self):
        c = self._big_canvas()
        layer = c.layer_stack.active
        layer.image.fill(QColor(0, 0, 0, 0))
        for i in range(60):
            c._save_history()
            layer.image.setPixelColor(i, 0, QColor(255, 0, 0, 255))
        before = len(c._history)
        c.undo()
        assert len(c._history) == before - 1
        assert len(c._redo_stack) == 1
        c.redo()
        assert len(c._history) == before


class TestLassoFillShapes:
    """囲って塗る（LASSO_FILL）が様々な形・大きさの投げなわで落ちないこと。
    極小・自己交差・キャンバス外・負座標など異常な入力でも例外を投げない。"""

    def _ring(self, cx, cy, r, n=24):
        import math
        return [QPoint(int(cx + r * math.cos(2 * math.pi * i / n)),
                       int(cy + r * math.sin(2 * math.pi * i / n))) for i in range(n)]

    @pytest.mark.parametrize("name", [
        "tiny", "small", "large", "self_intersect", "degenerate",
        "duplicate", "beyond_canvas", "negative", "one_px",
    ])
    def test_shapes_do_not_crash(self, canvas, name):
        shapes = {
            "tiny": [QPoint(5, 5), QPoint(15, 5), QPoint(10, 15)],
            "small": self._ring(50, 50, 8),
            "large": self._ring(100, 100, 500),
            "self_intersect": [QPoint(10, 10), QPoint(150, 150), QPoint(150, 10), QPoint(10, 150)],
            "degenerate": [QPoint(20, 20), QPoint(80, 20), QPoint(50, 20)],
            "duplicate": [QPoint(30, 30)] * 5 + [QPoint(90, 30), QPoint(90, 90)],
            "beyond_canvas": self._ring(100, 100, 5000),
            "negative": [QPoint(-100, -100), QPoint(50, -100), QPoint(50, 50), QPoint(-100, 50)],
            "one_px": [QPoint(60, 60), QPoint(61, 60), QPoint(61, 61), QPoint(60, 61)],
        }
        layer = canvas.layer_stack.active
        layer.image.fill(QColor(0, 0, 0, 0))
        canvas.pen_color = QColor(255, 0, 0, 255)
        canvas._apply_lasso_fill(layer, shapes[name])

    def test_fills_closed_region(self, canvas):
        """閉じた線の内側が実際に塗られる。"""
        layer = canvas.layer_stack.active
        layer.image.fill(QColor(0, 0, 0, 0))
        p = QPainter(layer.image)
        p.setPen(Qt.GlobalColor.black)
        p.drawRect(40, 40, 60, 60)
        p.end()
        canvas.pen_color = QColor(255, 0, 0, 255)
        # 矩形を完全に含む半径で囲う（対角より大きく取る）
        canvas._apply_lasso_fill(layer, self._ring(70, 70, 60, n=40))
        assert px(layer.image, 70, 70).red() > 200

    def test_offset_layer_does_not_crash(self, canvas):
        """移動後（offset 付き・キャンバスと別サイズ）のレイヤーでも落ちない。"""
        layer = canvas.layer_stack.active
        layer.image = QImage(80, 80, QImage.Format.Format_ARGB32)
        layer.image.fill(QColor(0, 0, 0, 0))
        layer.offset_x, layer.offset_y = 60, 40
        canvas.pen_color = QColor(0, 128, 255, 255)
        canvas._apply_lasso_fill(layer, self._ring(100, 80, 50))

    def test_negative_offset_layer_does_not_crash(self, canvas):
        layer = canvas.layer_stack.active
        layer.image = QImage(120, 120, QImage.Format.Format_ARGB32)
        layer.image.fill(QColor(0, 0, 0, 0))
        layer.offset_x, layer.offset_y = -70, -50
        canvas.pen_color = QColor(0, 200, 0, 255)
        canvas._apply_lasso_fill(layer, self._ring(20, 20, 60))

    def test_group_layer_is_rejected_safely(self, canvas):
        """グループレイヤーが対象でも例外にならず、何もしない。"""
        from layer import GroupLayer
        g = GroupLayer("g", W, H)
        canvas._apply_lasso_fill(g, self._ring(50, 50, 30))


class TestFillReference:
    """バケツ塗りの参照レイヤー。

    参照画像は「塗る対象のレイヤーの座標系・サイズ」で
    返され、対象レイヤー自身は含まないことが条件。
    """

    def _stack(self):
        """グループG[線画, レイヤー2] + グループ外に1枚。

        線画には x=50 に縦線を引いてある。
        """
        ls = LayerStack(100, 100)
        g = ls.add_group("G")
        line = ls.add("線画")
        other = ls.add("レイヤー2")
        for l in (line, other):
            ls.layers.remove(l)
            g.children.append(l)
        outside = ls.add("外")
        p = QPainter(line.image)
        p.setPen(QPen(QColor(0, 0, 0), 3))
        p.drawLine(50, 0, 50, 100)
        p.end()
        return ls, g, line, other, outside

    def _alpha(self, img):
        w, h = img.width(), img.height()
        ptr = img.constBits()
        ptr.setsize(w * h * 4)
        return np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)[:, :, 3]

    def test_layer_in_group_is_referenced(self):
        """フォルダ内の1枚を参照にして、同じフォルダの別のレイヤーで塗る。"""
        ls, g, line, other, outside = self._stack()
        line.reference = True
        c = Canvas(ls)
        c.resize(100, 100)
        ref = c._build_fill_reference(other)
        assert ref is not None
        assert int((self._alpha(ref) > 0).sum()) == 300

    def test_group_reference_excludes_target_layer(self):
        """グループを参照にしたとき、中にある塗る対象を含めないこと。

        含めてしまうと自分の塗った色が境界になり、
        「参照レイヤーのみ」を選んでも塗り漏れが出る。
        """
        ls, g, line, other, outside = self._stack()
        p = QPainter(other.image)
        p.fillRect(0, 0, 40, 40, QColor(0, 0, 255))
        p.end()
        g.reference = True
        c = Canvas(ls)
        c.resize(100, 100)
        ref = c._build_fill_reference(other)
        assert ref is not None
        a = self._alpha(ref)
        assert a[10, 10] == 0, "塗る対象自身が参照に混入している"
        assert int((a > 0).sum()) == 300

    def test_group_reference_from_outside(self):
        """フォルダごと参照にして、フォルダ外のレイヤーで塗る。"""
        ls, g, line, other, outside = self._stack()
        g.reference = True
        c = Canvas(ls)
        c.resize(100, 100)
        ref = c._build_fill_reference(outside)
        assert ref is not None
        assert int((self._alpha(ref) > 0).sum()) == 300

    @pytest.mark.parametrize("ox,oy", [(0, 0), (20, 0), (-20, 0), (0, -30), (0, 25)])
    def test_reference_follows_layer_offset(self, ox, oy):
        """対象レイヤーがずれていても参照の線が正しい位置に残ること。

        以前は QImage.copy() で切り出していたため、offset がマイナスの
        ときに範囲外が透明で埋まり、線が消えて「参照が効かない」
        状態になっていた。
        """
        ls, g, line, other, outside = self._stack()
        line.reference = True
        other.offset_x, other.offset_y = ox, oy
        c = Canvas(ls)
        c.resize(100, 100)
        ref = c._build_fill_reference(other)
        assert ref is not None
        assert ref.width() == other.image.width()
        assert ref.height() == other.image.height()
        a = self._alpha(ref)
        cols = np.nonzero((a > 0).any(axis=0))[0]
        assert len(cols) > 0, "参照の線が丸ごと消えている"
        # 線はローカル座標で x = 50 - offset_x に来る
        assert abs(int(cols.min()) - (50 - ox)) <= 2

    def test_reference_none_when_no_reference_layer(self):
        ls, g, line, other, outside = self._stack()
        c = Canvas(ls)
        c.resize(100, 100)
        assert c._build_fill_reference(other) is None

    def test_target_itself_as_reference_is_ignored(self):
        """塗る対象自身だけが参照なら、参照なしと同じになること。"""
        ls, g, line, other, outside = self._stack()
        other.reference = True
        c = Canvas(ls)
        c.resize(100, 100)
        assert c._build_fill_reference(other) is None
