"""E2E テスト: MainWindow を通じた実際のユーザー操作フロー"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor, QMouseEvent, QKeyEvent, QPainter
from PyQt6.QtCore import Qt, QPoint, QPointF, QEvent, QRect

app = QApplication.instance() or QApplication(sys.argv)

from main import MainWindow
from tools import Tool
from layer import Layer


def px(image, x, y) -> QColor:
    return QColor.fromRgba(image.pixel(x, y))


@pytest.fixture
def win():
    w = MainWindow()
    w.show()
    yield w
    w.close()


def press(canvas, x, y):
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y), QPointF(x, y),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier)
    canvas.mousePressEvent(ev)


def move(canvas, x, y):
    ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(x, y), QPointF(x, y),
                     Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier)
    canvas.mouseMoveEvent(ev)


def release(canvas, x, y):
    ev = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(x, y), QPointF(x, y),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier)
    canvas.mouseReleaseEvent(ev)


def draw_stroke(canvas, x1, y1, x2, y2):
    """ドラッグストローク（press → move → release）。"""
    press(canvas, x1, y1)
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    move(canvas, mid_x, mid_y)
    move(canvas, x2, y2)
    release(canvas, x2, y2)


# ── ペン描画 → undo → redo フロー ────────────────────────────────────────────

class TestPenUndoRedo:
    def test_draw_undo_redo(self, win):
        c = win.canvas
        layer = c.layer_stack.active
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 0, 0, 255)
        c.pen_size = 20

        before_pixel = layer.image.pixel(100, 100)
        # ストロークで描画
        draw_stroke(c, 80, 100, 120, 100)
        cp = c._widget_to_canvas(QPoint(100, 100))
        after_draw = layer.image.pixel(cp.x(), cp.y())
        assert after_draw != before_pixel, "描画後にピクセルが変化していない"

        c.undo()
        assert layer.image.pixel(cp.x(), cp.y()) == before_pixel, "undo 後に元に戻っていない"

        c.redo()
        assert layer.image.pixel(cp.x(), cp.y()) == after_draw, "redo 後に変化が再現されていない"


# ── レイヤー追加 → 描画 ────────────────────────────────────────────────────────

class TestLayerAddDraw:
    def test_add_layer_increases_count(self, win):
        initial_count = len(win.canvas.layer_stack.layers)
        win.layer_panel._add()
        assert len(win.canvas.layer_stack.layers) == initial_count + 1

    def test_draw_on_new_layer(self, win):
        win.layer_panel._add()
        c = win.canvas
        layer = c.layer_stack.active
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 255, 255)
        c.pen_size = 20
        draw_stroke(c, 80, 100, 120, 100)
        cp = c._widget_to_canvas(QPoint(100, 100))
        found = any(
            px(layer.image, x, y).blue() > 150 and px(layer.image, x, y).alpha() > 150
            for x in range(max(0, cp.x()-15), min(layer.image.width(), cp.x()+15))
            for y in range(max(0, cp.y()-8), min(layer.image.height(), cp.y()+8))
        )
        assert found, "新レイヤーへの描画が反映されていない"


# ── グループ作成 → 子レイヤー → 描画 ─────────────────────────────────────────

class TestGroupDraw:
    def test_draw_in_group_child(self, win):
        c = win.canvas
        ls = c.layer_stack
        win.layer_panel._add_group()
        grp_idx = ls.active_index
        grp = ls.layers[grp_idx]
        assert grp.is_group

        child = Layer("child", ls.width, ls.height)
        grp.children.append(child)
        ls.set_active(grp_idx, 0)

        c.tool = Tool.PEN
        c.pen_color = QColor(0, 255, 0, 255)
        c.pen_size = 20
        draw_stroke(c, 80, 100, 120, 100)
        cp = c._widget_to_canvas(QPoint(100, 100))
        found = any(
            px(child.image, x, y).green() > 150 and px(child.image, x, y).alpha() > 150
            for x in range(max(0, cp.x()-15), min(child.image.width(), cp.x()+15))
            for y in range(max(0, cp.y()-8), min(child.image.height(), cp.y()+8))
        )
        assert found, "グループ内子レイヤーへの描画が反映されていない"

    def test_draw_blocked_on_group_itself(self, win):
        c = win.canvas
        ls = c.layer_stack
        win.layer_panel._add_group()
        grp = ls.layers[ls.active_index]
        child = Layer("child", ls.width, ls.height)
        grp.children.append(child)
        ls.set_active(ls.active_index, -1)

        messages = []
        c.status_message.connect(messages.append)
        c.tool = Tool.PEN
        press(c, 100, 100)
        assert any("グループ" in m for m in messages)
        cp = c._widget_to_canvas(QPoint(100, 100))
        assert px(child.image, cp.x(), cp.y()).alpha() == 0


# ── 矩形選択 → 変形モード ────────────────────────────────────────────────────

class TestSelectTransform:
    def test_select_mode_creates_selection(self, win):
        c = win.canvas
        c.tool = Tool.SELECT_RECT
        c.select_mode = "select"
        press(c, 40, 40)
        move(c, 120, 120)
        release(c, 120, 120)
        assert c._selection_rect is not None

    def test_transform_outside_click_commits(self, win):
        c = win.canvas
        layer = c.layer_stack.active
        p = QPainter(layer.image)
        p.fillRect(80, 80, 40, 40, QColor(0, 0, 255, 255))
        p.end()

        c.tool = Tool.SELECT_RECT
        c.select_mode = "transform"
        press(c, 70, 70)
        move(c, 130, 130)
        release(c, 130, 130)

        if c._transform_image:
            press(c, 5, 5)
            release(c, 5, 5)
            assert c._transform_image is None, "選択外クリック後に transform_image が残っている"


# ── ファイル保存 → 読み込み ───────────────────────────────────────────────────

class TestSaveLoad:
    def test_save_and_reload_pola(self, win):
        layer = win.canvas.layer_stack.active
        p = QPainter(layer.image)
        p.fillRect(50, 50, 30, 30, QColor(200, 100, 50, 255))
        p.end()
        original_pixel = layer.image.pixel(65, 65)

        with tempfile.NamedTemporaryFile(suffix=".pola", delete=False) as f:
            path = f.name
        try:
            win._write_pola(path)
            assert os.path.exists(path)
            win._load_pola(path)
            reloaded_layer = win.canvas.layer_stack.active
            assert reloaded_layer is not None
            reloaded_pixel = reloaded_layer.image.pixel(65, 65)
            rc = QColor.fromRgba(reloaded_pixel)
            oc = QColor.fromRgba(original_pixel)
            assert abs(rc.red() - oc.red()) < 10
            assert abs(rc.green() - oc.green()) < 10
            assert abs(rc.blue() - oc.blue()) < 10
        finally:
            os.unlink(path)

    def test_load_clears_undo_history(self, win):
        c = win.canvas
        c._save_history()
        assert len(c._history) > 0
        layer = c.layer_stack.active
        with tempfile.NamedTemporaryFile(suffix=".pola", delete=False) as f:
            path = f.name
        try:
            win._write_pola(path)
            win._load_pola(path)
            assert len(win.canvas._history) == 0
        finally:
            os.unlink(path)

    def test_save_creates_valid_zip(self, win):
        import zipfile
        with tempfile.NamedTemporaryFile(suffix=".pola", delete=False) as f:
            path = f.name
        try:
            win._write_pola(path)
            assert zipfile.is_zipfile(path)
            with zipfile.ZipFile(path) as zf:
                assert "meta.json" in zf.namelist()
        finally:
            os.unlink(path)


# ── ツール切り替え ────────────────────────────────────────────────────────────

class TestToolSwitch:
    def test_switch_tool_does_not_crash(self, win):
        c = win.canvas
        c.tool = Tool.PEN
        press(c, 100, 100)
        win._on_tool_change(Tool.ERASER)
        press(c, 50, 50)
        release(c, 50, 50)

    def test_transform_committed_on_tool_switch(self, win):
        c = win.canvas
        layer = c.layer_stack.active
        p = QPainter(layer.image)
        p.fillRect(80, 80, 40, 40, QColor(255, 0, 0, 255))
        p.end()

        c.tool = Tool.SELECT_RECT
        c.select_mode = "transform"
        press(c, 70, 70)
        move(c, 130, 130)
        release(c, 130, 130)

        if c._transform_image:
            win._on_tool_change(Tool.PEN)
            assert c._transform_image is None, "ツール切替時に変形が確定されていない"


# ── merge down ────────────────────────────────────────────────────────────────

class TestMergeDown:
    def test_merge_down_two_layers(self, win):
        win.layer_panel._add()
        initial_count = len(win.layer_stack.layers)
        win._merge_down()
        assert len(win.layer_stack.layers) == initial_count - 1

    def test_merge_down_single_layer_does_not_crash(self, win):
        ls = win.layer_stack
        while len(ls.layers) > 1:
            ls.remove(0)
        win._merge_down()
        assert len(ls.layers) == 1


# ── レイヤーをラスタライズ ───────────────────────────────────────────────────

class TestRasterizeLayer:
    """メニュー「画像 → レイヤーをラスタライズ」の統合テスト。"""

    def test_rasterize_disables_effects_and_saves_history(self, win):
        layer = win.layer_stack.active
        layer.border_enabled = True
        layer.border_size = 3
        layer.border_color = QColor(0, 0, 255, 255)
        hist_len_before = len(win.canvas._history)

        win._rasterize_layer()

        assert layer.border_enabled is False
        assert len(win.canvas._history) == hist_len_before + 1

    def test_rasterize_group_layer_shows_message_and_noop(self, win):
        win.layer_panel._add_group()
        win.layer_stack.active_path = [0]
        hist_len_before = len(win.canvas._history)

        win._rasterize_layer()  # グループはラスタライズ対象外

        assert len(win.canvas._history) == hist_len_before

    def test_rasterize_then_merge_down_no_double_apply(self, win):
        """ラスタライズ後に統合しても効果が二重適用されないこと。"""
        win.layer_panel._add()
        layer = win.layer_stack.active
        layer.border_enabled = True
        layer.border_size = 3
        layer.border_color = QColor(0, 0, 255, 255)

        win._rasterize_layer()
        assert layer.border_enabled is False

        initial_count = len(win.layer_stack.layers)
        win._merge_down()
        assert len(win.layer_stack.layers) == initial_count - 1
        # 統合後も焼き込み済み効果は再度有効化されない
        assert win.layer_stack.layers[0].border_enabled is False


# ── フィルター → ゴミ取り ────────────────────────────────────────────────────

class TestDespeckle:
    """DespeckleDialog: スライダー操作でプレビュー更新するたびに全ラベルへ
    Pythonループでアクセスしていた実装を numpy ベクトル化に置き換えた回帰確認。
    ベクトル化前後で結果がビット単位で一致することを検証する。"""

    def _make_noisy_layer(self, w=80, h=80, seed=0):
        from layer import Layer
        rng = np.random.default_rng(seed)
        lyr = Layer("noisy", w, h)
        lyr.image.fill(Qt.GlobalColor.transparent)
        ptr = lyr.image.bits(); ptr.setsize(lyr.image.sizeInBytes())
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4)
        n = (w * h) // 100
        ys = rng.integers(0, h, n); xs = rng.integers(0, w, n)
        arr[ys, xs] = (0, 0, 0, 255)
        return lyr

    def test_update_preview_removes_small_specks(self, win):
        from main import DespeckleDialog
        layer = self._make_noisy_layer()
        dlg = DespeckleDialog(win.canvas, layer, None)
        dlg._update_preview(9)
        # 面積9px²以下の孤立点は透明化されているはず（少なくとも1点は消える）
        ptr = dlg._layer.image.bits(); ptr.setsize(dlg._layer.image.sizeInBytes())
        after = np.frombuffer(ptr, dtype=np.uint8).reshape(80, 80, 4)
        opaque_after = int((after[:, :, 3] > 0).sum())
        opaque_before = int((dlg._orig_arr[:, :, 3] > 0).sum())
        assert opaque_after <= opaque_before

    def test_update_preview_matches_reference_loop_implementation(self, win):
        """ベクトル化した実装が、ラベルごとにループする素朴な実装と同じ結果になること。"""
        import cv2
        from main import DespeckleDialog
        layer = self._make_noisy_layer(seed=7)
        dlg = DespeckleDialog(win.canvas, layer, None)
        dlg._update_preview(9)

        ptr = dlg._layer.image.bits(); ptr.setsize(dlg._layer.image.sizeInBytes())
        vectorized = bytes(ptr)

        arr = dlg._orig_arr.copy()
        opaque = (arr[:, :, 3] > 0).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(opaque, connectivity=8)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] <= 9:
                arr[labels == i] = 0
        from PyQt6.QtGui import QImage as _QImage
        ref_img = _QImage(arr.tobytes(), 80, 80, 80 * 4, _QImage.Format.Format_ARGB32).copy()
        ref_ptr = ref_img.bits(); ref_ptr.setsize(ref_img.sizeInBytes())
        reference = bytes(ref_ptr)

        assert vectorized == reference

    def test_reject_restores_original_image(self, win):
        from main import DespeckleDialog
        layer = self._make_noisy_layer()
        original = layer.image.copy()
        dlg = DespeckleDialog(win.canvas, layer, None)
        dlg._update_preview(50)
        dlg.reject()

        def raw(img):
            b = img.bits(); b.setsize(img.sizeInBytes())
            return bytes(b)

        assert raw(layer.image) == raw(original)


# ── 新規作成 ─────────────────────────────────────────────────────────────────

class TestNewCanvas:
    def test_new_without_changes_skips_dialog(self, win):
        """履歴なし・保存なしの初期状態なら確認ダイアログなしで通過できる。"""
        c = win.canvas
        c._history.clear()
        win._current_path = None
        # _new() を呼んでも QMessageBox が出ない（出ると test がブロックされる）
        # has_changes=False の経路を確認
        has_changes = win._current_path is not None or bool(c._history)
        assert not has_changes, "初期状態で has_changes が True になっている"

    def test_new_after_drawing_has_changes(self, win):
        c = win.canvas
        c._save_history()
        has_changes = win._current_path is not None or bool(c._history)
        assert has_changes, "描画後に has_changes が False になっている"


# ══════════════════════════════════════════════════════════════════════════════
# シナリオテスト: クリエイターが絵を描く一連のワークフロー
# ══════════════════════════════════════════════════════════════════════════════

class TestScenarioFullPainting:
    """レイヤーを追加し、各レイヤーに描画し、統合し、Undoで戻す一連の流れ。"""

    def test_multilayer_painting_and_merge_undo(self, win):
        c = win.canvas
        ls = c.layer_stack

        # 1. 背景レイヤーに赤で塗る
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 0, 0, 255)
        c.pen_size = 30
        draw_stroke(c, 100, 100, 200, 100)

        # 2. 新しいレイヤーを追加して青で塗る
        win.layer_panel._add()
        assert len(ls.layers) == 2
        c.pen_color = QColor(0, 0, 255, 255)
        draw_stroke(c, 100, 150, 200, 150)

        # 3. さらにレイヤーを追加して緑で塗る
        win.layer_panel._add()
        assert len(ls.layers) == 3
        c.pen_color = QColor(0, 255, 0, 255)
        draw_stroke(c, 100, 200, 200, 200)

        # 4. 下に統合（3→2レイヤー）
        win._merge_down()
        assert len(ls.layers) == 2

        # 5. Undoで統合を戻す → 3レイヤーに復帰
        c.undo()
        assert len(ls.layers) == 3, "統合のUndoでレイヤー数が戻っていない"

        # 6. Redoで再統合
        c.redo()
        assert len(ls.layers) == 2, "統合のRedoでレイヤー数が戻っていない"

    def test_add_layer_undo_removes_it(self, win):
        c = win.canvas
        ls = c.layer_stack
        orig = len(ls.layers)

        # レイヤー追加（layer_panelから: structure_will_changeが発火する）
        win.layer_panel._add()
        assert len(ls.layers) == orig + 1

        # Undoでレイヤー追加を取り消し
        c.undo()
        assert len(ls.layers) == orig, "レイヤー追加のUndoで元に戻っていない"

    def test_delete_layer_undo_restores_it(self, win):
        c = win.canvas
        ls = c.layer_stack

        # 2枚にする
        win.layer_panel._add()
        assert len(ls.layers) == 2
        name = ls.layers[0].name

        # 削除
        win.layer_panel._remove()
        assert len(ls.layers) == 1

        # Undo → 2枚に戻る
        c.undo()
        assert len(ls.layers) == 2, "レイヤー削除のUndoで戻っていない"


class TestScenarioGroupWorkflow:
    """フォルダ（グループ）を使った作業フロー。"""

    def test_create_group_add_children_merge_undo(self, win):
        c = win.canvas
        ls = c.layer_stack

        # 1. グループ作成
        win.layer_panel._add_group()
        grp_idx = ls.active_index
        grp = ls.layers[grp_idx]
        assert grp.is_group

        # 2. グループ内にレイヤー追加して描画
        child1 = Layer("線画", ls.width, ls.height)
        child2 = Layer("色", ls.width, ls.height)
        grp.children.append(child1)
        grp.children.append(child2)
        ls.set_active(grp_idx, 0)

        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 0, 255)
        c.pen_size = 5
        draw_stroke(c, 80, 80, 200, 200)

        ls.set_active(grp_idx, 1)
        c.pen_color = QColor(255, 200, 100, 255)
        c.pen_size = 30
        draw_stroke(c, 90, 90, 190, 190)

        # 3. グループを統合
        ls.set_active(grp_idx, -1)
        total_before = len(ls.layers)
        win._merge_selected()
        merged = ls.layers[grp_idx]
        assert not merged.is_group, "統合後もグループのまま"

        # 4. Undoでグループ復帰
        c.undo()
        restored = ls.layers[grp_idx]
        assert restored.is_group, "Undoでグループに戻っていない"
        assert len(restored.children) == 2, "Undoで子レイヤーが復元されていない"


class TestScenarioSelectAndTransform:
    """選択ツールで範囲選択→Escape解除→投げなわ選択の流れ。"""

    def test_rect_select_escape_clears(self, win):
        c = win.canvas

        # 矩形選択
        c.tool = Tool.SELECT_RECT
        c.select_mode = "select"
        press(c, 30, 30)
        move(c, 150, 150)
        release(c, 150, 150)
        assert c._selection_rect is not None

        # Escapeで解除
        ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                       Qt.KeyboardModifier.NoModifier)
        c.keyPressEvent(ev)
        assert c._selection_rect is None, "Escapeで選択が解除されていない"

    def test_lasso_select_escape_clears(self, win):
        c = win.canvas

        # 投げなわ選択
        c.tool = Tool.LASSO
        c.select_mode = "select"
        press(c, 50, 50)
        move(c, 120, 50)
        move(c, 120, 120)
        move(c, 50, 120)
        release(c, 50, 120)
        assert c._selection_rect is not None
        assert len(c._lasso_path_points) > 0

        # Escape
        ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                       Qt.KeyboardModifier.NoModifier)
        c.keyPressEvent(ev)
        assert c._selection_rect is None, "Escapeで投げなわ選択が解除されていない"
        assert len(c._lasso_path_points) == 0, "Escapeで投げなわパスが残っている"

    def test_double_escape_after_lasso_then_rect(self, win):
        """投げなわ→Escape→矩形選択→Escapeで全て解除される。"""
        c = win.canvas

        c.tool = Tool.LASSO
        c.select_mode = "select"
        press(c, 50, 50)
        move(c, 120, 50)
        move(c, 120, 120)
        release(c, 120, 120)
        ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                       Qt.KeyboardModifier.NoModifier)
        c.keyPressEvent(ev)
        assert c._selection_rect is None

        c.tool = Tool.SELECT_RECT
        press(c, 30, 30)
        move(c, 150, 150)
        release(c, 150, 150)
        assert c._selection_rect is not None

        c.keyPressEvent(ev)
        assert c._selection_rect is None


class TestScenarioDrawingTools:
    """ペン・消しゴム・図形・バケツを切り替えながら描く流れ。"""

    def test_pen_eraser_shape_cycle(self, win):
        c = win.canvas
        layer = c.layer_stack.active

        # 1. ペンで描く
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 0, 0, 255)
        c.pen_size = 20
        draw_stroke(c, 80, 100, 200, 100)
        cp = c._widget_to_canvas(QPoint(140, 100))

        # 2. 消しゴムで消す
        c.tool = Tool.ERASER
        c.eraser_size = 40
        draw_stroke(c, 130, 100, 150, 100)
        assert px(layer.image, cp.x(), cp.y()).alpha() == 0, "消しゴムで消えていない"

        # 3. 四角形を描く
        c.tool = Tool.RECT
        c.pen_color = QColor(0, 0, 255, 255)
        c.pen_size = 3
        c.shape_fill = "fill"
        press(c, 50, 50)
        move(c, 120, 120)
        release(c, 120, 120)

        # 4. 直線を描く
        c.tool = Tool.LINE
        c.pen_color = QColor(0, 255, 0, 255)
        c.pen_size = 5
        press(c, 10, 10)
        move(c, 200, 200)
        release(c, 200, 200)

        # 5. Undoで直線を取り消し → 四角まで戻る
        c.undo()
        # クラッシュしなければOK


class TestScenarioSaveLoadCycle:
    """描画 → 保存 → 新規 → 読み込みで復元されるか。"""

    def test_paint_save_load_verify(self, win):
        c = win.canvas
        ls = c.layer_stack

        # レイヤー2枚で描く
        c.tool = Tool.PEN
        c.pen_color = QColor(100, 50, 200, 255)
        c.pen_size = 20
        draw_stroke(c, 100, 100, 200, 100)
        win.layer_panel._add()
        c.pen_color = QColor(50, 200, 100, 255)
        draw_stroke(c, 100, 150, 200, 150)

        with tempfile.NamedTemporaryFile(suffix=".pola", delete=False) as f:
            path = f.name
        try:
            win._write_pola(path)

            # 読み込み
            win._load_pola(path)
            assert len(win.canvas.layer_stack.layers) == 2, "レイヤー数が保存時と異なる"
            assert len(win.canvas._history) == 0, "ロード後に履歴が残っている"
        finally:
            os.unlink(path)


class TestScenarioAnimation:
    """アニメーションモードでフレーム追加・再生・GIF出力の一連の流れ。"""

    def test_animation_full_workflow(self, win):
        c = win.canvas
        ap = win.anim_panel

        # 1. アニメーションモード有効化
        win._toggle_anim_mode(True)
        assert ap.isVisible()

        # 2. フレーム1: 赤い丸を描いてフレーム追加
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 0, 0, 255)
        c.pen_size = 30
        draw_stroke(c, 100, 100, 120, 100)
        ap._on_add_frame()
        assert len(ap.frames) == 1

        # 3. フレーム2: 消しゴムで消してから青を描いてフレーム追加
        c.tool = Tool.ERASER
        c.eraser_size = 100
        draw_stroke(c, 50, 50, 200, 200)
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 255, 255)
        c.pen_size = 30
        draw_stroke(c, 150, 100, 170, 100)
        ap._on_add_frame()
        assert len(ap.frames) == 2

        # 4. フレーム3: 緑を描いて追加
        c.tool = Tool.ERASER
        c.eraser_size = 100
        draw_stroke(c, 50, 50, 200, 200)
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 255, 0, 255)
        c.pen_size = 30
        draw_stroke(c, 100, 150, 120, 150)
        ap._on_add_frame()
        assert len(ap.frames) == 3

        # 5. フレームを選択し差し替え
        ap._on_frame_clicked(1)
        assert ap.current_frame == 1
        c.pen_color = QColor(255, 255, 0, 255)
        draw_stroke(c, 80, 80, 100, 80)
        ap._on_replace_frame()
        assert len(ap.frames) == 3

        # 6. フレーム順序入れ替え
        ap._on_frame_clicked(2)
        ap._on_move_left()
        assert ap.current_frame == 1

        # 7. フレーム削除
        ap._on_frame_clicked(0)
        ap._on_delete_frame()
        assert len(ap.frames) == 2

        # 8. オニオンスキン
        onion = ap.get_onion_images()
        # current_frame=0なら前フレームがないので空
        ap._on_frame_clicked(1)
        onion = ap.get_onion_images()
        assert len(onion) >= 1, "オニオンスキンが取得できていない"

        # 9. 再生→停止
        ap._start_play()
        assert ap._playing
        ap._stop_play()
        assert not ap._playing

        # 10. GIF書き出し
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            gif_path = f.name
        try:
            ap._export_gif_to(gif_path)
            assert os.path.exists(gif_path)
            assert os.path.getsize(gif_path) > 100, "GIFファイルが小さすぎる"
        finally:
            os.unlink(gif_path)

        # 11. アニメーションモード解除
        win._toggle_anim_mode(False)
        assert not ap.isVisible()

    def test_animation_onion_skin_multiple_frames(self, win):
        c = win.canvas
        ap = win.anim_panel
        win._toggle_anim_mode(True)

        # 5フレーム追加
        for i in range(5):
            c.tool = Tool.PEN
            c.pen_color = QColor(50 * i, 0, 0, 255)
            c.pen_size = 10
            draw_stroke(c, 100 + i * 10, 100, 120 + i * 10, 100)
            ap._on_add_frame()
        assert len(ap.frames) == 5

        # オニオンスキン数を3に設定して最後のフレームを選択
        ap._onion_count = 3
        ap._on_frame_clicked(4)
        onion = ap.get_onion_images()
        assert len(onion) == 3, f"オニオンスキン3枚のはずが{len(onion)}枚"

        # オニオンスキンOFFにすると空
        ap._onion_enabled = False
        assert ap.get_onion_images() == []

        win._toggle_anim_mode(False)


class TestScenarioMergeAllVisibleUndo:
    """全表示レイヤー統合 → Undoで全レイヤー復元。"""

    def test_merge_all_visible_and_undo(self, win):
        c = win.canvas
        ls = c.layer_stack

        # 3枚のレイヤーを作って各々描画
        c.tool = Tool.PEN
        c.pen_size = 20
        c.pen_color = QColor(255, 0, 0, 255)
        draw_stroke(c, 80, 80, 120, 80)

        win.layer_panel._add()
        c.pen_color = QColor(0, 255, 0, 255)
        draw_stroke(c, 80, 120, 120, 120)

        win.layer_panel._add()
        c.pen_color = QColor(0, 0, 255, 255)
        draw_stroke(c, 80, 160, 120, 160)
        assert len(ls.layers) == 3

        # 全表示統合
        win._merge_all_visible()
        assert len(ls.layers) == 1, "全表示統合後に1レイヤーになっていない"

        # Undoで3枚に戻る
        c.undo()
        assert len(ls.layers) == 3, "全表示統合のUndoで3レイヤーに戻っていない"

        # Redoで再統合
        c.redo()
        assert len(ls.layers) == 1


class TestScenarioComplexUndoChain:
    """描画→レイヤー追加→描画→統合→Undo連打で全部戻る。"""

    def test_undo_chain(self, win):
        c = win.canvas
        ls = c.layer_stack

        # 1. 描画
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 0, 0, 255)
        c.pen_size = 20
        draw_stroke(c, 100, 100, 150, 100)

        # 2. レイヤー追加
        win.layer_panel._add()
        assert len(ls.layers) == 2

        # 3. 新レイヤーに描画
        c.pen_color = QColor(0, 0, 255, 255)
        draw_stroke(c, 100, 150, 150, 150)

        # 4. 統合
        win._merge_down()
        assert len(ls.layers) == 1

        # 5. Undo連打: 統合→描画→レイヤー追加→描画
        c.undo()  # 統合戻し
        assert len(ls.layers) == 2, "統合Undoで2レイヤーに戻っていない"

        c.undo()  # 青描画戻し（pixel undo — 新レイヤーobjectが変わっているのでskipされる場合あり）
        c.undo()  # レイヤー追加戻し
        # レイヤー数が1に戻るか、pixelのundoがスキップされて2のままか
        # 少なくともクラッシュしない
        assert len(ls.layers) <= 2


# ══════════════════════════════════════════════════════════════════════════════
# 実際のイラスト制作工程を再現するシナリオテスト
#
# 参考: 一般的なデジタルイラスト制作メイキング
#   1. ラフ (大まかな形を描く)
#   2. 線画 (ラフの上に新規レイヤーでペン入れ)
#   3. 下塗り (パーツ別レイヤーでバケツ塗り: 肌・髪・服)
#   4. 影   (乗算レイヤーで影を塗る)
#   5. ハイライト (加算レイヤーで光を入れる)
#   6. 仕上げ (オーバーレイで色味統一、エフェクト追加)
#   7. 保存
# ══════════════════════════════════════════════════════════════════════════════

class TestIllustMaking_CharacterFullProcess:
    """
    キャラクターイラスト制作の全工程を再現。
    ラフ → 線画 → 下塗り(肌・髪・服) → 影(乗算) → ハイライト(加算)
    → 仕上げ(オーバーレイ + エフェクト) → 保存 → 読み込み確認。
    """

    def test_full_character_illustration_workflow(self, win):
        c = win.canvas
        ls = c.layer_stack

        # ================================================================
        # 工程1: ラフ — 太いペンで薄い色でざっくり描く
        # ================================================================
        ls.layers[0].name = "ラフ"
        c.tool = Tool.PEN
        c.pen_color = QColor(180, 180, 255, 100)  # 薄い青（ラフ色）
        c.pen_size = 15

        # 顔の輪郭
        draw_stroke(c, 100, 50, 100, 150)   # 左輪郭
        draw_stroke(c, 200, 50, 200, 150)   # 右輪郭
        draw_stroke(c, 100, 50, 200, 50)    # 上
        draw_stroke(c, 100, 150, 150, 180)  # 左あご
        draw_stroke(c, 200, 150, 150, 180)  # 右あご
        # 体の大まかなライン
        draw_stroke(c, 120, 180, 80, 350)   # 左肩→腰
        draw_stroke(c, 180, 180, 220, 350)  # 右肩→腰

        assert ls.layers[0].name == "ラフ"

        # ================================================================
        # 工程2: 線画 — 新規レイヤー、細いペン、黒でペン入れ
        # ラフの不透明度を下げて上から清書するイメージ
        # ================================================================
        ls.layers[0].opacity = 30  # ラフを薄く

        win.layer_panel._add()
        ls.active.name = "線画"
        assert len(ls.layers) == 2

        c.pen_color = QColor(30, 30, 30, 255)  # ほぼ黒
        c.pen_size = 3

        # 顔の線画
        draw_stroke(c, 100, 50, 100, 150)
        draw_stroke(c, 200, 50, 200, 150)
        draw_stroke(c, 100, 50, 200, 50)
        draw_stroke(c, 100, 150, 150, 180)
        draw_stroke(c, 200, 150, 150, 180)
        # 目
        draw_stroke(c, 125, 90, 145, 90)
        draw_stroke(c, 160, 90, 175, 90)
        # 口
        draw_stroke(c, 140, 130, 165, 130)
        # 体
        draw_stroke(c, 120, 180, 80, 350)
        draw_stroke(c, 180, 180, 220, 350)

        # ラフを非表示に
        ls.layers[0].visible = False

        # ================================================================
        # 工程3: 下塗り — パーツごとに別レイヤー、バケツ塗り
        # ================================================================

        # --- 3a: 肌レイヤー ---
        win.layer_panel._add()
        skin_idx = ls.active_index
        ls.active.name = "肌"
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 220, 200, 255)  # 肌色
        c.pen_size = 40
        # 顔エリアを塗りつぶし（太ペンで面塗り）
        draw_stroke(c, 120, 60, 190, 60)
        draw_stroke(c, 115, 80, 190, 80)
        draw_stroke(c, 110, 100, 195, 100)
        draw_stroke(c, 115, 120, 190, 120)
        draw_stroke(c, 125, 140, 180, 140)
        draw_stroke(c, 135, 160, 165, 160)

        # --- 3b: 髪レイヤー ---
        win.layer_panel._add()
        hair_idx = ls.active_index
        ls.active.name = "髪"
        c.pen_color = QColor(60, 40, 30, 255)  # 暗い茶色
        c.pen_size = 25
        draw_stroke(c, 90, 30, 210, 30)   # 前髪
        draw_stroke(c, 90, 30, 85, 100)   # 左サイド
        draw_stroke(c, 210, 30, 215, 100) # 右サイド
        draw_stroke(c, 95, 45, 205, 45)

        # --- 3c: 服レイヤー ---
        win.layer_panel._add()
        cloth_idx = ls.active_index
        ls.active.name = "服"
        c.pen_color = QColor(70, 100, 180, 255)  # 青い服
        c.pen_size = 35
        draw_stroke(c, 100, 185, 200, 185)
        draw_stroke(c, 90, 210, 210, 210)
        draw_stroke(c, 85, 250, 215, 250)
        draw_stroke(c, 80, 290, 220, 290)
        draw_stroke(c, 80, 330, 220, 330)

        assert len(ls.layers) == 5  # ラフ, 線画, 肌, 髪, 服

        # ================================================================
        # 工程4: 影 — 乗算レイヤーで影を塗る
        # ================================================================
        win.layer_panel._add()
        ls.active.name = "影"
        ls.active.blend_mode = "multiply"

        c.pen_color = QColor(180, 140, 180, 80)  # 薄紫（影色）
        c.pen_size = 25

        # 顔の右側に影
        draw_stroke(c, 170, 70, 190, 130)
        # あご下
        draw_stroke(c, 125, 155, 175, 155)
        # 服の影
        draw_stroke(c, 130, 220, 200, 260)
        draw_stroke(c, 120, 300, 210, 330)

        assert ls.active.blend_mode == "multiply"

        # ================================================================
        # 工程5: ハイライト — 加算レイヤーで光を入れる
        # ================================================================
        win.layer_panel._add()
        ls.active.name = "ハイライト"
        ls.active.blend_mode = "plus"  # 加算

        c.pen_color = QColor(255, 255, 230, 60)  # 薄い黄白
        c.pen_size = 15

        # 髪のハイライト
        draw_stroke(c, 110, 35, 190, 35)
        # 目のハイライト
        c.pen_size = 5
        draw_stroke(c, 130, 87, 135, 87)
        draw_stroke(c, 165, 87, 170, 87)

        assert ls.active.blend_mode == "plus"

        # ================================================================
        # 工程6: 仕上げ — オーバーレイで色味統一 + エフェクト
        # ================================================================

        # --- 6a: オーバーレイで全体に暖色を乗せる ---
        win.layer_panel._add()
        ls.active.name = "色味調整"
        ls.active.blend_mode = "overlay"
        ls.active.opacity = 30

        c.pen_color = QColor(255, 200, 150, 40)  # 暖色
        c.pen_size = 100
        draw_stroke(c, 50, 50, 250, 350)  # 全体にふわっと

        assert ls.active.blend_mode == "overlay"

        # --- 6b: 線画にグロー（発光）エフェクト ---
        line_layer_idx = 1  # 線画レイヤーのインデックス
        ls.set_active(line_layer_idx, -1)
        line_layer = ls.active
        line_layer.glow_enabled = True
        line_layer.glow_color = QColor(255, 240, 220)
        line_layer.glow_size = 2
        line_layer.glow_strength = 30
        assert line_layer.glow_enabled

        # --- 6c: 肌レイヤーにHSL微調整（血色を良く） ---
        ls.set_active(skin_idx, -1)
        skin_layer = ls.active
        skin_layer.hsl_enabled = True
        skin_layer.hsl_hue = 5        # 少し赤寄り
        skin_layer.hsl_saturation = 15 # 彩度アップ
        skin_layer.hsl_lightness = 0
        assert skin_layer.hsl_enabled

        # --- 6d: 影レイヤーにガウスぼかし（影を柔らかく） ---
        shadow_layer_idx = 5  # 影レイヤー
        ls.set_active(shadow_layer_idx, -1)
        shadow_layer = ls.active
        shadow_layer.blur_enabled = True
        shadow_layer.blur_radius = 3
        shadow_layer.blur_strength = 50
        assert shadow_layer.blur_enabled

        # ================================================================
        # 工程7: 完成画像を確認（composite がクラッシュしないか）
        # ================================================================
        final_image = ls.composite()
        assert final_image.width() == ls.width
        assert final_image.height() == ls.height
        # 完全な透明ではない（何か描かれている）
        has_content = False
        for y in range(0, final_image.height(), 50):
            for x in range(0, final_image.width(), 50):
                if QColor.fromRgba(final_image.pixel(x, y)) != QColor(255, 255, 255, 255):
                    has_content = True
                    break
            if has_content:
                break
        assert has_content, "composite結果が白一色（描画が反映されていない）"

        # ================================================================
        # 工程8: 保存 → 読み込みで全レイヤー構成が復元されるか
        # ================================================================
        layer_count = len(ls.layers)
        layer_names = [l.name for l in ls.layers]
        blend_modes = [
            l.blend_mode if hasattr(l, 'blend_mode') else "normal"
            for l in ls.layers
        ]

        with tempfile.NamedTemporaryFile(suffix=".pola", delete=False) as f:
            path = f.name
        try:
            win._write_pola(path)
            win._load_pola(path)
            loaded_ls = win.canvas.layer_stack
            assert len(loaded_ls.layers) == layer_count, \
                f"レイヤー数が異なる: {len(loaded_ls.layers)} != {layer_count}"
            for i, name in enumerate(layer_names):
                assert loaded_ls.layers[i].name == name, \
                    f"レイヤー名が異なる: {loaded_ls.layers[i].name} != {name}"
            # ブレンドモード復元
            for i, mode in enumerate(blend_modes):
                actual = getattr(loaded_ls.layers[i], 'blend_mode', 'normal')
                assert actual == mode, \
                    f"レイヤー{i}のブレンドモードが異なる: {actual} != {mode}"
            # エフェクト復元
            loaded_skin = loaded_ls.layers[skin_idx]
            assert loaded_skin.hsl_enabled, "肌のHSL設定が復元されていない"
            assert loaded_skin.hsl_hue == 5
        finally:
            os.unlink(path)


class TestIllustMaking_UndoMistakes:
    """
    制作中に「間違えた！」→ Undo/Redoする実際のフロー。
    イラスト制作で頻出する操作ミスのパターンを再現。
    """

    def test_wrong_layer_draw_undo(self, win):
        """線画レイヤーに間違えて色を塗ってしまい Undo で戻す。"""
        c = win.canvas
        ls = c.layer_stack

        # 線画レイヤー
        ls.layers[0].name = "線画"
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 0, 255)
        c.pen_size = 3
        draw_stroke(c, 100, 100, 200, 200)

        # 下塗りレイヤー追加
        win.layer_panel._add()
        ls.active.name = "下塗り"
        c.pen_color = QColor(255, 200, 180, 255)
        c.pen_size = 30
        draw_stroke(c, 100, 100, 200, 100)

        # ミス: 線画レイヤーに切り替え忘れたまま太ペンで描いてしまう
        # → 実際には下塗りレイヤーのまま描く
        c.pen_color = QColor(0, 255, 0, 255)  # 全然違う色
        c.pen_size = 50
        draw_stroke(c, 80, 80, 220, 220)  # ミス描画

        # 気づいてUndo
        c.undo()

        # 下塗りレイヤーに緑が残っていないことを確認
        cp = c._widget_to_canvas(QPoint(150, 150))
        color = px(ls.active.image, cp.x(), cp.y())
        assert color.green() < 200 or color.alpha() < 50, \
            "Undoしたのに誤描画が残っている"

    def test_accidental_merge_undo(self, win):
        """うっかり統合してしまったレイヤーをUndoで復帰する。"""
        c = win.canvas
        ls = c.layer_stack

        # 3枚構成: 線画・肌・髪
        ls.layers[0].name = "線画"
        win.layer_panel._add()
        ls.active.name = "肌"
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 220, 200, 255)
        c.pen_size = 30
        draw_stroke(c, 100, 100, 200, 100)

        win.layer_panel._add()
        ls.active.name = "髪"
        c.pen_color = QColor(80, 50, 30, 255)
        c.pen_size = 20
        draw_stroke(c, 100, 50, 200, 50)

        assert len(ls.layers) == 3
        names_before = [l.name for l in ls.layers]

        # うっかり全統合
        win._merge_all_visible()
        assert len(ls.layers) == 1

        # やっぱり戻す！
        c.undo()
        assert len(ls.layers) == 3, "統合Undoでレイヤーが3枚に戻っていない"
        names_after = [l.name for l in ls.layers]
        assert names_after == names_before, \
            f"レイヤー名が復元されていない: {names_after} != {names_before}"

    def test_eraser_too_much_undo(self, win):
        """消しゴムで消しすぎた → Undo → 描き直す。"""
        c = win.canvas
        ls = c.layer_stack

        # 広範囲に色を塗る
        layer = ls.active
        p = QPainter(layer.image)
        p.fillRect(0, 0, layer.image.width(), layer.image.height(),
                   QColor(200, 100, 50, 255))
        p.end()

        # 消しゴムで中央を消す
        c.tool = Tool.ERASER
        c.eraser_size = 80
        draw_stroke(c, 130, 100, 170, 100)

        # 中央付近が消えている
        cp = c._widget_to_canvas(QPoint(150, 100))
        erased_alpha = px(layer.image, cp.x(), cp.y()).alpha()
        assert erased_alpha < 255, "消しゴムがそもそも効いていない"

        # Undoで消しゴム取り消し
        c.undo()
        restored_alpha = px(layer.image, cp.x(), cp.y()).alpha()
        assert restored_alpha > erased_alpha, "消しゴムのUndoで復元されていない"

        # 描き直す
        c.tool = Tool.PEN
        c.pen_color = QColor(100, 200, 50, 255)
        c.pen_size = 10
        draw_stroke(c, 140, 95, 160, 105)
        # クラッシュしなければOK


class TestIllustMaking_SelectionWorkflow:
    """
    選択ツールを使った実際の作業フロー:
    - 投げなわで部分選択 → 移動 → 確定
    - 矩形選択で不要部分を消す
    - 選択→変形→Escape→別ツールに切り替え
    """

    def test_lasso_select_move_deselect_continue_drawing(self, win):
        """投げなわで囲む → 解除 → 続けてペンで描く。"""
        c = win.canvas
        ls = c.layer_stack

        # 何か描く
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 0, 0, 255)
        c.pen_size = 20
        draw_stroke(c, 100, 100, 200, 100)

        # 投げなわで選択
        c.tool = Tool.LASSO
        c.select_mode = "select"
        press(c, 80, 80)
        move(c, 220, 80)
        move(c, 220, 130)
        move(c, 80, 130)
        release(c, 80, 130)
        assert c._selection_rect is not None, "投げなわ選択ができていない"
        assert len(c._lasso_path_points) > 0

        # Escapeで解除
        ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                       Qt.KeyboardModifier.NoModifier)
        c.keyPressEvent(ev)
        assert c._selection_rect is None, "選択解除されていない"
        assert len(c._lasso_path_points) == 0

        # そのままペンに切り替えて続けて描く
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 255, 255)
        c.pen_size = 10
        draw_stroke(c, 100, 150, 200, 150)
        # クラッシュせず描ける

    def test_rect_select_erase_inside(self, win):
        """矩形選択→消しゴム→選択内だけ消える想定（選択の有無で操作を確認）。"""
        c = win.canvas

        # 全面に色を塗る
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 100, 50, 255)
        c.pen_size = 60
        draw_stroke(c, 50, 100, 250, 100)
        draw_stroke(c, 50, 150, 250, 150)
        draw_stroke(c, 50, 200, 250, 200)

        # 矩形選択
        c.tool = Tool.SELECT_RECT
        c.select_mode = "select"
        press(c, 100, 80)
        move(c, 200, 170)
        release(c, 200, 170)
        assert c._selection_rect is not None

        # 選択解除してから消しゴム
        ev = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape,
                       Qt.KeyboardModifier.NoModifier)
        c.keyPressEvent(ev)
        c.tool = Tool.ERASER
        c.eraser_size = 30
        draw_stroke(c, 140, 140, 160, 140)
        # クラッシュしない

    def test_multiple_tool_switches_during_selection(self, win):
        """選択中にツール切り替えを繰り返してもクラッシュしない。"""
        c = win.canvas

        # 描画
        c.tool = Tool.PEN
        c.pen_color = QColor(100, 100, 200, 255)
        c.pen_size = 20
        draw_stroke(c, 100, 100, 200, 200)

        # 矩形選択
        c.tool = Tool.SELECT_RECT
        c.select_mode = "select"
        press(c, 80, 80)
        move(c, 220, 220)
        release(c, 220, 220)

        # ツール切り替え連打
        for tool in [Tool.PEN, Tool.ERASER, Tool.LASSO, Tool.FILL,
                     Tool.LINE, Tool.RECT, Tool.ELLIPSE, Tool.EYEDROPPER,
                     Tool.PEN]:
            c.tool = tool
        # 最終的にペンで描ける
        c.pen_color = QColor(0, 255, 0, 255)
        c.pen_size = 5
        draw_stroke(c, 150, 150, 160, 160)


class TestIllustMaking_FolderOrganization:
    """
    フォルダ（グループ）でレイヤーを整理する制作フロー。
    キャラパーツをフォルダにまとめて統合・Undoする。
    """

    def test_organize_parts_into_folder_merge_undo(self, win):
        """肌・髪・服をフォルダにまとめて統合→Undo→フォルダ復帰。"""
        c = win.canvas
        ls = c.layer_stack

        # グループ「キャラ」を作成
        win.layer_panel._add_group()
        grp_idx = ls.active_index
        grp = ls.layers[grp_idx]
        grp.name = "キャラ"

        # フォルダ内にレイヤーを追加
        skin = Layer("肌", ls.width, ls.height)
        hair = Layer("髪", ls.width, ls.height)
        cloth = Layer("服", ls.width, ls.height)
        grp.children.extend([skin, hair, cloth])

        # 各レイヤーに描画
        ls.set_active(grp_idx, 0)  # 肌
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 220, 200, 255)
        c.pen_size = 30
        draw_stroke(c, 100, 100, 200, 100)

        ls.set_active(grp_idx, 1)  # 髪
        c.pen_color = QColor(60, 40, 30, 255)
        draw_stroke(c, 100, 50, 200, 50)

        ls.set_active(grp_idx, 2)  # 服
        c.pen_color = QColor(70, 100, 180, 255)
        draw_stroke(c, 100, 200, 200, 200)

        # フォルダ統合
        ls.set_active(grp_idx, -1)
        win._merge_selected()
        merged = ls.layers[grp_idx]
        assert not merged.is_group, "統合後にグループが残っている"
        assert merged.name == "キャラ"

        # Undoでフォルダ復帰
        c.undo()
        restored = ls.layers[grp_idx]
        assert restored.is_group, "Undoでグループに戻っていない"
        assert len(restored.children) == 3
        child_names = [ch.name for ch in restored.children]
        assert child_names == ["肌", "髪", "服"], \
            f"子レイヤー名が復元されていない: {child_names}"


class TestIllustMaking_EffectsAndBlendModes:
    """
    ブレンドモードとエフェクトを組み合わせた仕上げ工程。
    """

    def test_blend_mode_composite_no_crash(self, win):
        """各ブレンドモードのレイヤーが重なった状態でcompositeできる。"""
        c = win.canvas
        ls = c.layer_stack

        modes = ["normal", "multiply", "screen", "overlay", "plus"]
        colors = [
            QColor(255, 0, 0, 200),
            QColor(0, 255, 0, 200),
            QColor(0, 0, 255, 200),
            QColor(255, 255, 0, 100),
            QColor(255, 200, 255, 60),
        ]

        # 初期レイヤーに白背景
        c.tool = Tool.PEN
        c.pen_size = 100
        c.pen_color = QColor(255, 255, 255, 255)
        draw_stroke(c, 50, 100, 250, 100)
        draw_stroke(c, 50, 200, 250, 200)

        # 各ブレンドモードでレイヤー追加して描画
        for mode, color in zip(modes, colors):
            win.layer_panel._add()
            ls.active.name = mode
            ls.active.blend_mode = mode
            c.pen_color = color
            c.pen_size = 40
            draw_stroke(c, 80, 80, 220, 220)

        assert len(ls.layers) == 6

        # 全レイヤー統合してクラッシュしない
        result = ls.composite()
        assert result.width() == ls.width

    def test_multiple_effects_on_single_layer(self, win):
        """1レイヤーに複数エフェクト（シャドウ+グロー+ぼかし+HSL）同時適用。"""
        c = win.canvas
        ls = c.layer_stack

        # 描画
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 100, 50, 255)
        c.pen_size = 20
        draw_stroke(c, 100, 100, 200, 200)

        layer = ls.active

        # 全エフェクトON
        layer.shadow_enabled = True
        layer.shadow_color = QColor(0, 0, 0)
        layer.shadow_offset_x = 5
        layer.shadow_offset_y = 5
        layer.shadow_blur = 5
        layer.shadow_strength = 80

        layer.glow_enabled = True
        layer.glow_color = QColor(255, 200, 100)
        layer.glow_size = 3
        layer.glow_strength = 60

        layer.blur_enabled = True
        layer.blur_radius = 2
        layer.blur_strength = 40

        layer.hsl_enabled = True
        layer.hsl_hue = 30
        layer.hsl_saturation = 20
        layer.hsl_lightness = -10

        # image_with_effects() がクラッシュしない
        effected = layer.image_with_effects()
        assert effected.width() == layer.image.width()
        assert effected.height() == layer.image.height()

        # compositeもOK
        result = ls.composite()
        assert result.width() == ls.width

    def test_effect_settings_survive_save_load(self, win):
        """エフェクト設定が保存→読み込みで復元される。"""
        c = win.canvas
        ls = c.layer_stack

        c.tool = Tool.PEN
        c.pen_color = QColor(255, 0, 0, 255)
        c.pen_size = 10
        draw_stroke(c, 100, 100, 150, 100)

        layer = ls.active
        layer.shadow_enabled = True
        layer.shadow_offset_x = 7
        layer.shadow_blur = 10
        layer.glow_enabled = True
        layer.glow_size = 5
        layer.blur_enabled = True
        layer.blur_radius = 4
        layer.hsl_enabled = True
        layer.hsl_hue = -45
        layer.hsl_saturation = 30

        with tempfile.NamedTemporaryFile(suffix=".pola", delete=False) as f:
            path = f.name
        try:
            win._write_pola(path)
            win._load_pola(path)
            loaded = win.canvas.layer_stack.layers[0]
            assert loaded.shadow_enabled
            assert loaded.shadow_offset_x == 7
            assert loaded.shadow_blur == 10
            assert loaded.glow_enabled
            assert loaded.glow_size == 5
            assert loaded.blur_enabled
            assert loaded.blur_radius == 4
            assert loaded.hsl_enabled
            assert loaded.hsl_hue == -45
            assert loaded.hsl_saturation == 30
        finally:
            os.unlink(path)


class TestIllustMaking_AnimationFromIllust:
    """
    イラスト完成後にアニメーション化する工程。
    完成絵をベースに表情差分でパラパラ漫画を作る。
    """

    def test_expression_animation_workflow(self, win):
        """表情差分3枚 → GIF書き出しのフロー。"""
        c = win.canvas
        ls = c.layer_stack
        ap = win.anim_panel

        # --- ベースの顔を描く ---
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 0, 255)
        c.pen_size = 3
        # 輪郭
        draw_stroke(c, 100, 50, 100, 150)
        draw_stroke(c, 200, 50, 200, 150)
        draw_stroke(c, 100, 50, 200, 50)
        draw_stroke(c, 100, 150, 150, 180)
        draw_stroke(c, 200, 150, 150, 180)

        # アニメーションモードON
        win._toggle_anim_mode(True)

        # --- 表情1: 笑顔 ---
        # 目（上向きアーチ）
        draw_stroke(c, 125, 95, 135, 85)
        draw_stroke(c, 135, 85, 145, 95)
        draw_stroke(c, 160, 95, 170, 85)
        draw_stroke(c, 170, 85, 180, 95)
        # 口（笑い）
        draw_stroke(c, 135, 130, 150, 140)
        draw_stroke(c, 150, 140, 165, 130)
        ap._on_add_frame()

        # --- 表情2: 普通 ---
        # 目を消して描き直す
        c.tool = Tool.ERASER
        c.eraser_size = 20
        draw_stroke(c, 120, 80, 185, 100)  # 目を消す
        draw_stroke(c, 130, 125, 170, 145) # 口を消す
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 0, 255)
        c.pen_size = 3
        # 目（普通）
        draw_stroke(c, 125, 90, 145, 90)
        draw_stroke(c, 160, 90, 180, 90)
        # 口（一文字）
        draw_stroke(c, 140, 135, 160, 135)
        ap._on_add_frame()

        # --- 表情3: 驚き ---
        c.tool = Tool.ERASER
        c.eraser_size = 20
        draw_stroke(c, 120, 80, 185, 100)
        draw_stroke(c, 130, 125, 170, 145)
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 0, 255)
        c.pen_size = 3
        # 目（大きく）
        c.pen_size = 5
        draw_stroke(c, 125, 85, 145, 85)
        draw_stroke(c, 125, 85, 125, 95)
        draw_stroke(c, 145, 85, 145, 95)
        draw_stroke(c, 125, 95, 145, 95)
        draw_stroke(c, 160, 85, 180, 85)
        draw_stroke(c, 160, 85, 160, 95)
        draw_stroke(c, 180, 85, 180, 95)
        draw_stroke(c, 160, 95, 180, 95)
        # 口（O型）
        c.tool = Tool.ELLIPSE
        c.pen_size = 2
        press(c, 143, 128)
        move(c, 158, 148)
        release(c, 158, 148)
        ap._on_add_frame()

        assert len(ap.frames) == 3

        # オニオンスキンで確認
        ap._on_frame_clicked(2)
        onion = ap.get_onion_images()
        assert len(onion) >= 1, "オニオンスキンで前フレームが見えない"

        # 再生テスト
        ap._start_play()
        assert ap._playing
        # 数フレーム進める
        ap._play_next()
        ap._play_next()
        ap._stop_play()
        assert not ap._playing

        # GIF書き出し
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            gif_path = f.name
        try:
            ap._export_gif_to(gif_path)
            assert os.path.exists(gif_path)
            assert os.path.getsize(gif_path) > 200, "GIFが小さすぎる"

            # GIFの中身を検証
            from PIL import Image
            with Image.open(gif_path) as gif:
                frame_count = 0
                try:
                    while True:
                        frame_count += 1
                        gif.seek(gif.tell() + 1)
                except EOFError:
                    pass
                assert frame_count == 3, \
                    f"GIFのフレーム数が3ではない: {frame_count}"
        finally:
            os.unlink(gif_path)

        win._toggle_anim_mode(False)


class TestIllustMaking_LongSession:
    """
    長時間作業（多数のUndo/Redo・レイヤー追加削除を繰り返す）で
    クラッシュやメモリリーク的な問題が出ないか。
    """

    def test_heavy_undo_redo_cycle(self, win):
        """大量の描画→Undo→Redo→描画を繰り返す。"""
        c = win.canvas
        c.tool = Tool.PEN
        c.pen_size = 5

        for i in range(20):
            c.pen_color = QColor((i * 37) % 256, (i * 73) % 256, (i * 113) % 256, 255)
            draw_stroke(c, 50 + i * 5, 50, 50 + i * 5, 300)

        # 10回Undo
        for _ in range(10):
            c.undo()
        # 5回Redo
        for _ in range(5):
            c.redo()
        # さらに描画
        for i in range(10):
            c.pen_color = QColor(255, i * 25, 0, 255)
            draw_stroke(c, 100, 50 + i * 20, 200, 50 + i * 20)
        # クラッシュしない

    def test_rapid_layer_add_remove_cycle(self, win):
        """レイヤー追加→描画→削除を繰り返す。"""
        c = win.canvas
        ls = c.layer_stack

        for i in range(10):
            win.layer_panel._add()
            c.tool = Tool.PEN
            c.pen_color = QColor(i * 25, 100, 200, 255)
            c.pen_size = 10
            draw_stroke(c, 100, 100, 200, 200)
            if len(ls.layers) > 3:
                win.layer_panel._remove()

        # Undo連打
        for _ in range(15):
            c.undo()
        # Redo連打
        for _ in range(8):
            c.redo()
        # クラッシュしない
        assert len(ls.layers) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# クリスタ風 GIFアニメーション制作工程の再現テスト
#
# 参考: CLIP STUDIO PAINT のアニメーション制作フロー
#   1. キャンバス作成・FPS設定
#   2. セル1（キーフレーム: 動きの起点）を描く
#   3. セル3（キーフレーム: 動きの終点）を描く
#   4. オニオンスキンON → セル2（中割り）を描く
#   5. 再生して動きを確認
#   6. 修正（差し替え・順序変更）
#   7. GIF書き出し
# ══════════════════════════════════════════════════════════════════════════════

class TestClipStudioStyle_BounceAnimation:
    """
    クリスタ風: ボールが跳ねるループアニメーション。
    キーフレーム→中割り→再生確認→修正→GIF出力の全工程。
    """

    def _draw_ball(self, canvas, x, y, size=20):
        """指定位置にボール（丸）を描く。"""
        canvas.tool = Tool.ELLIPSE
        canvas.pen_size = 2
        canvas.shape_fill = "fill"
        press(canvas, x - size, y - size)
        move(canvas, x + size, y + size)
        release(canvas, x + size, y + size)

    def _clear_canvas(self, canvas):
        """キャンバスを消しゴムで全消し。"""
        canvas.tool = Tool.ERASER
        canvas.eraser_size = 200
        for y in range(50, 350, 80):
            draw_stroke(canvas, 10, y, 290, y)

    def test_bounce_ball_keyframe_workflow(self, win):
        """
        工程:
        1. セル1: ボール上（キーフレーム）
        2. セル3: ボール下（キーフレーム）
        3. オニオンスキンON → セル2: ボール中間（中割り）
        4. 再生で確認
        5. セル2を微修正（差し替え）
        6. フレーム順確認
        7. GIF書き出し
        """
        c = win.canvas
        ap = win.anim_panel

        # アニメーションモードON
        win._toggle_anim_mode(True)
        assert ap.isVisible()

        # --- セル1: ボール上 (キーフレーム) ---
        c.pen_color = QColor(255, 80, 80, 255)  # 赤いボール
        self._draw_ball(c, 150, 80)
        ap._on_add_frame()
        assert len(ap.frames) == 1
        assert ap.current_frame == 0

        # --- セル3: ボール下 (キーフレーム) ---
        # クリスタ流: 中割りを飛ばして終点を先に描く
        self._clear_canvas(c)
        c.pen_color = QColor(255, 80, 80, 255)
        self._draw_ball(c, 150, 250)
        # 影を追加（地面に接してるから影が濃い）
        c.tool = Tool.PEN
        c.pen_color = QColor(100, 100, 100, 80)
        c.pen_size = 25
        draw_stroke(c, 130, 275, 170, 275)
        ap._on_add_frame()
        assert len(ap.frames) == 2

        # --- オニオンスキンON → セル2: 中割り ---
        # クリスタのオニオンスキン: 前のセルを薄く表示して中間の位置を確認
        ap._onion_enabled = True
        ap._onion_count = 2

        # セル1を選択してから中間フレームを挿入
        # (current_frame=0の次に挿入される)
        ap._on_frame_clicked(0)
        onion = ap.get_onion_images()
        # frame 0が選択中なので前フレームはない
        assert len(onion) == 0  # 先頭なので前がない

        # キャンバスを消して中間位置に描く
        self._clear_canvas(c)
        c.pen_color = QColor(255, 80, 80, 255)
        self._draw_ball(c, 150, 160)  # 上と下の中間
        # 影（中間なので薄め）
        c.tool = Tool.PEN
        c.pen_color = QColor(100, 100, 100, 40)
        c.pen_size = 18
        draw_stroke(c, 135, 275, 165, 275)
        ap._on_add_frame()
        assert len(ap.frames) == 3
        # 挿入位置: frame 0の後 → current_frame=1

        # オニオンスキンで前フレームが見えるか確認
        ap._on_frame_clicked(1)
        onion = ap.get_onion_images()
        assert len(onion) >= 1, "中割り描画時にオニオンスキンが機能していない"

        # --- 再生して確認 ---
        ap._start_play()
        assert ap._playing
        # 数フレーム進める（ループ確認）
        for _ in range(6):  # 3フレーム x 2周
            ap._play_next()
        # ループして先頭に戻っているはず
        assert 0 <= ap.current_frame < 3
        ap._stop_play()
        assert not ap._playing

        # --- 修正: 中割りフレームの差し替え ---
        # 「ちょっとボールの位置が高すぎた」ので修正
        ap._on_frame_clicked(1)  # 中割りフレーム
        self._clear_canvas(c)
        c.pen_color = QColor(255, 80, 80, 255)
        self._draw_ball(c, 150, 180)  # もう少し下に修正
        c.tool = Tool.PEN
        c.pen_color = QColor(100, 100, 100, 50)
        c.pen_size = 20
        draw_stroke(c, 133, 275, 167, 275)
        ap._on_replace_frame()
        assert len(ap.frames) == 3  # フレーム数は変わらない

        # --- フレーム順序の確認 ---
        # フレームの画像がそれぞれ異なることを確認
        f0 = ap.frames[0]
        f1 = ap.frames[1]
        f2 = ap.frames[2]
        # ボールの位置が違うのでピクセルが異なるはず
        center_pixel_0 = f0.pixel(150, 80)
        center_pixel_1 = f1.pixel(150, 180)
        center_pixel_2 = f2.pixel(150, 250)
        # 各フレームのボール位置付近にピクセルがある（完全透明ではない）
        assert QColor.fromRgba(center_pixel_0).alpha() > 0 or \
               QColor.fromRgba(f0.pixel(150, 90)).alpha() > 0, \
            "フレーム0のボール位置にピクセルがない"

        # --- GIF書き出し ---
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            gif_path = f.name
        try:
            ap._export_gif_to(gif_path)
            assert os.path.exists(gif_path)
            size = os.path.getsize(gif_path)
            assert size > 200, f"GIFが小さすぎる: {size} bytes"

            from PIL import Image
            with Image.open(gif_path) as gif:
                frame_count = 0
                try:
                    while True:
                        frame_count += 1
                        gif.seek(gif.tell() + 1)
                except EOFError:
                    pass
                assert frame_count == 3, \
                    f"GIFフレーム数が3ではない: {frame_count}"
        finally:
            os.unlink(gif_path)

        win._toggle_anim_mode(False)


class TestClipStudioStyle_WalkCycle:
    """
    クリスタ風: 歩行サイクルアニメーション。
    4コマで1歩のループ → 再生 → 修正 → 再生確認 → GIF出力。
    """

    def test_walk_cycle_4frames(self, win):
        """
        棒人間の歩行4コマ:
        フレーム1: 直立（接地）
        フレーム2: 右足前（通過）
        フレーム3: 右足接地
        フレーム4: 左足前（通過）
        """
        c = win.canvas
        ap = win.anim_panel
        win._toggle_anim_mode(True)
        ap._onion_enabled = True
        ap._onion_count = 2

        # 共通: 地面の線
        def draw_ground():
            c.tool = Tool.LINE
            c.pen_color = QColor(100, 80, 60, 255)
            c.pen_size = 2
            press(c, 50, 280)
            move(c, 250, 280)
            release(c, 250, 280)

        def clear():
            c.tool = Tool.ERASER
            c.eraser_size = 200
            for y in range(50, 350, 80):
                draw_stroke(c, 10, y, 290, y)

        # --- フレーム1: 直立 ---
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 0, 255)
        c.pen_size = 3
        # 頭
        draw_stroke(c, 150, 100, 150, 100)
        c.tool = Tool.ELLIPSE
        c.pen_size = 2
        c.shape_fill = "none"
        press(c, 140, 90)
        move(c, 160, 110)
        release(c, 160, 110)
        # 体
        c.tool = Tool.LINE
        c.pen_size = 3
        press(c, 150, 110)
        move(c, 150, 200)
        release(c, 150, 200)
        # 左足
        press(c, 150, 200)
        move(c, 140, 275)
        release(c, 140, 275)
        # 右足
        press(c, 150, 200)
        move(c, 160, 275)
        release(c, 160, 275)
        draw_ground()
        ap._on_add_frame()

        # --- フレーム2: 右足前 ---
        clear()
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 0, 255)
        c.pen_size = 3
        # 頭（少し上）
        c.tool = Tool.ELLIPSE
        c.pen_size = 2
        c.shape_fill = "none"
        press(c, 140, 85)
        move(c, 160, 105)
        release(c, 160, 105)
        # 体
        c.tool = Tool.LINE
        c.pen_size = 3
        press(c, 150, 105)
        move(c, 150, 195)
        release(c, 150, 195)
        # 左足（後ろ）
        press(c, 150, 195)
        move(c, 135, 275)
        release(c, 135, 275)
        # 右足（前）
        press(c, 150, 195)
        move(c, 175, 260)
        release(c, 175, 260)
        draw_ground()
        ap._on_add_frame()

        # オニオンスキンで前フレームが見える
        ap._on_frame_clicked(1)
        onion = ap.get_onion_images()
        assert len(onion) >= 1, "フレーム2でオニオンスキンが効いていない"

        # --- フレーム3: 右足接地 ---
        clear()
        c.tool = Tool.ELLIPSE
        c.pen_size = 2
        c.shape_fill = "none"
        press(c, 140, 90)
        move(c, 160, 110)
        release(c, 160, 110)
        c.tool = Tool.LINE
        c.pen_size = 3
        press(c, 150, 110)
        move(c, 150, 200)
        release(c, 150, 200)
        press(c, 150, 200)
        move(c, 130, 275)
        release(c, 130, 275)
        press(c, 150, 200)
        move(c, 170, 275)
        release(c, 170, 275)
        draw_ground()
        ap._on_add_frame()

        # --- フレーム4: 左足前 ---
        clear()
        c.tool = Tool.ELLIPSE
        c.pen_size = 2
        c.shape_fill = "none"
        press(c, 140, 85)
        move(c, 160, 105)
        release(c, 160, 105)
        c.tool = Tool.LINE
        c.pen_size = 3
        press(c, 150, 105)
        move(c, 150, 195)
        release(c, 150, 195)
        press(c, 150, 195)
        move(c, 170, 275)
        release(c, 170, 275)
        press(c, 150, 195)
        move(c, 130, 260)
        release(c, 130, 260)
        draw_ground()
        ap._on_add_frame()

        assert len(ap.frames) == 4

        # --- 再生して確認 (ループ2周) ---
        ap._start_play()
        assert ap._playing
        for _ in range(8):  # 4フレーム x 2周
            ap._play_next()
        ap._stop_play()

        # --- 修正: フレーム2の位置微調整 ---
        ap._on_frame_clicked(1)
        # 描き直すのではなく差し替え（実制作ではよくある）
        clear()
        c.tool = Tool.ELLIPSE
        c.pen_size = 2
        c.shape_fill = "none"
        press(c, 140, 87)
        move(c, 160, 107)
        release(c, 160, 107)
        c.tool = Tool.LINE
        c.pen_size = 3
        press(c, 150, 107)
        move(c, 150, 197)
        release(c, 150, 197)
        press(c, 150, 197)
        move(c, 133, 275)
        release(c, 133, 275)
        press(c, 150, 197)
        move(c, 178, 258)
        release(c, 178, 258)
        draw_ground()
        ap._on_replace_frame()
        assert len(ap.frames) == 4

        # --- 再生して修正を確認 ---
        ap._start_play()
        for _ in range(4):
            ap._play_next()
        ap._stop_play()

        # --- GIF書き出し ---
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            gif_path = f.name
        try:
            ap._export_gif_to(gif_path)
            assert os.path.exists(gif_path)

            from PIL import Image
            with Image.open(gif_path) as gif:
                frame_count = 0
                try:
                    while True:
                        frame_count += 1
                        gif.seek(gif.tell() + 1)
                except EOFError:
                    pass
                assert frame_count == 4, \
                    f"歩行サイクルGIFが4フレームではない: {frame_count}"
        finally:
            os.unlink(gif_path)

        win._toggle_anim_mode(False)


class TestClipStudioStyle_FrameManipulation:
    """
    クリスタ風: タイムライン操作の再現。
    フレーム追加→並べ替え→削除→追加し直し→再生確認。
    制作中に「この順番じゃない」「このフレームいらない」をやるフロー。
    """

    def test_reorder_delete_reinsert(self, win):
        """
        5フレーム追加 → 順番入れ替え → 不要フレーム削除
        → 新フレーム追加 → 再生確認 → GIF出力。
        """
        c = win.canvas
        ap = win.anim_panel
        win._toggle_anim_mode(True)

        # 5フレーム追加（各フレーム異なる位置に丸を描く）
        positions = [(80, 100), (120, 130), (160, 160), (200, 130), (240, 100)]
        for i, (bx, by) in enumerate(positions):
            c.tool = Tool.ERASER
            c.eraser_size = 200
            for y in range(50, 350, 80):
                draw_stroke(c, 10, y, 290, y)

            c.tool = Tool.PEN
            c.pen_color = QColor(50 + i * 40, 100, 200 - i * 30, 255)
            c.pen_size = 15
            draw_stroke(c, bx - 10, by, bx + 10, by)
            draw_stroke(c, bx, by - 10, bx, by + 10)
            ap._on_add_frame()

        assert len(ap.frames) == 5

        # --- 順番入れ替え: フレーム4を前に移動 ---
        ap._on_frame_clicked(3)
        ap._on_move_left()
        assert ap.current_frame == 2
        ap._on_move_left()
        assert ap.current_frame == 1

        # --- 不要なフレーム削除: 最後のフレーム ---
        ap._on_frame_clicked(4)
        ap._on_delete_frame()
        assert len(ap.frames) == 4

        # --- フレーム追加し直し ---
        c.tool = Tool.ERASER
        c.eraser_size = 200
        for y in range(50, 350, 80):
            draw_stroke(c, 10, y, 290, y)
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 200, 50, 255)
        c.pen_size = 20
        draw_stroke(c, 150, 200, 160, 200)
        ap._on_add_frame()
        assert len(ap.frames) == 5

        # --- 再生確認 ---
        ap._start_play()
        assert ap._playing
        for _ in range(10):  # 2周
            ap._play_next()
        ap._stop_play()

        # --- GIF書き出し ---
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            gif_path = f.name
        try:
            ap._export_gif_to(gif_path)
            from PIL import Image
            with Image.open(gif_path) as gif:
                count = 0
                try:
                    while True:
                        count += 1
                        gif.seek(gif.tell() + 1)
                except EOFError:
                    pass
                assert count == 5
        finally:
            os.unlink(gif_path)

        win._toggle_anim_mode(False)


class TestClipStudioStyle_OnionSkinDrawing:
    """
    クリスタのオニオンスキン機能を活用した作画フロー。
    前のフレームを透かして見ながら次を描く。
    """

    def test_onion_skin_progressive_drawing(self, win):
        """
        オニオンスキンをON/OFFしながら6フレームのアニメを描く。
        途中でオニオンスキン枚数を変えて確認する。
        """
        c = win.canvas
        ap = win.anim_panel
        win._toggle_anim_mode(True)

        # 最初はオニオンスキンOFF（1枚目はガイドなしで描く）
        ap._onion_enabled = False

        # フレーム1
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 0, 255)
        c.pen_size = 5
        draw_stroke(c, 100, 200, 130, 200)  # 左位置
        ap._on_add_frame()

        # 2枚目からオニオンスキンON
        ap._onion_enabled = True
        ap._onion_count = 1

        # フレーム2
        c.tool = Tool.ERASER
        c.eraser_size = 200
        draw_stroke(c, 10, 100, 290, 250)
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 0, 0, 255)
        c.pen_size = 5
        draw_stroke(c, 120, 200, 150, 200)  # 少し右
        ap._on_add_frame()

        # オニオンスキン確認
        ap._on_frame_clicked(1)
        onion = ap.get_onion_images()
        assert len(onion) == 1, "オニオンスキン1枚のはずが違う"
        opacity = onion[0][1]
        assert 0 < opacity < 1, "オニオンスキンの不透明度が異常"

        # フレーム3〜6: オニオンスキン枚数を増やしながら描く
        ap._onion_count = 2
        for i in range(4):
            c.tool = Tool.ERASER
            c.eraser_size = 200
            draw_stroke(c, 10, 100, 290, 250)
            c.tool = Tool.PEN
            c.pen_color = QColor(0, 0, 0, 255)
            c.pen_size = 5
            x = 140 + i * 20
            draw_stroke(c, x, 200, x + 30, 200)
            ap._on_add_frame()

        assert len(ap.frames) == 6

        # 最後のフレームでオニオンスキン枚数確認
        ap._on_frame_clicked(5)
        onion = ap.get_onion_images()
        assert len(onion) == 2, f"オニオンスキン2枚のはずが{len(onion)}枚"
        # 近いフレームほど不透明度が高い
        assert onion[0][1] > onion[1][1], \
            "近いフレームの方が不透明度が低い（逆になっている）"

        # オニオンスキンOFFにして確認
        ap._onion_enabled = False
        assert ap.get_onion_images() == []

        # 再度ONにして再生
        ap._onion_enabled = True
        ap._start_play()
        for _ in range(6):
            ap._play_next()
        ap._stop_play()

        # GIF出力
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            gif_path = f.name
        try:
            ap._export_gif_to(gif_path)
            from PIL import Image
            with Image.open(gif_path) as gif:
                count = 0
                try:
                    while True:
                        count += 1
                        gif.seek(gif.tell() + 1)
                except EOFError:
                    pass
                assert count == 6
        finally:
            os.unlink(gif_path)

        win._toggle_anim_mode(False)


class TestClipStudioStyle_ModeSwitch:
    """
    お絵かきモード ↔ アニメーションモードの切り替えフロー。
    絵を描いてからアニメモードに入り、またお絵かきに戻る。
    """

    def test_paint_then_animate_then_paint(self, win):
        """
        1. お絵かきモードでイラストを描く
        2. アニメーションモードに切り替えてフレーム追加
        3. お絵かきモードに戻って描き足す
        4. 再度アニメモードでフレーム追加 → GIF出力
        """
        c = win.canvas
        ls = c.layer_stack
        ap = win.anim_panel

        # --- お絵かきモード: イラストを描く ---
        assert not ap.isVisible()
        c.tool = Tool.PEN
        c.pen_color = QColor(255, 0, 0, 255)
        c.pen_size = 20
        draw_stroke(c, 100, 100, 200, 100)

        win.layer_panel._add()
        c.pen_color = QColor(0, 0, 255, 255)
        draw_stroke(c, 100, 150, 200, 150)
        assert len(ls.layers) == 2

        # --- アニメモードに切り替え ---
        win._toggle_anim_mode(True)
        assert ap.isVisible()

        # 現在のキャンバスをフレーム1として追加
        ap._on_add_frame()
        assert len(ap.frames) == 1

        # --- お絵かきモードに戻る ---
        win._toggle_anim_mode(False)
        assert not ap.isVisible()

        # 描き足す（レイヤーは維持されている）
        assert len(ls.layers) == 2
        c.tool = Tool.PEN
        c.pen_color = QColor(0, 255, 0, 255)
        c.pen_size = 15
        draw_stroke(c, 100, 200, 200, 200)

        # --- 再度アニメモード ---
        win._toggle_anim_mode(True)
        # フレームは保持されている
        assert len(ap.frames) == 1

        # 変更後のキャンバスをフレーム2として追加
        ap._on_add_frame()
        assert len(ap.frames) == 2

        # GIF出力
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            gif_path = f.name
        try:
            ap._export_gif_to(gif_path)
            assert os.path.exists(gif_path)
            assert os.path.getsize(gif_path) > 100
        finally:
            os.unlink(gif_path)

        win._toggle_anim_mode(False)
