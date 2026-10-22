"""アニメーション（アニメーションフォルダ・タイムライン）のテスト"""
import sys, os, json, zipfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor, QPainter
from PyQt6.QtCore import Qt, QPoint

app = QApplication.instance() or QApplication(sys.argv)

import anim
from anim import BLANK
from layer import LayerStack, Layer, GroupLayer
from animation_panel import AnimationPanel

W, H = 40, 30


def px(img: QImage, x: int = 5, y: int = 5) -> QColor:
    return QColor.fromRgba(img.pixel(x, y))


def fill(layer, color: QColor):
    p = QPainter(layer.image)
    p.fillRect(0, 0, W, H, color)
    p.end()


def make_stack() -> LayerStack:
    st = LayerStack(W, H)
    bg = Layer("背景", W, H)
    fill(bg, QColor(255, 255, 255))
    st.layers.append(bg)
    st.active_path = [0]
    return st


def three_cels(st):
    """赤・緑・青のセル 1〜3 をコマ 0〜2 に置いたフォルダ。"""
    folder = anim.make_anim_folder(st)
    c1 = folder.children[0]
    fill(c1, QColor(255, 0, 0))
    c2 = anim.new_cel(st, folder, 1)
    fill(c2, QColor(0, 255, 0))
    c3 = anim.new_cel(st, folder, 2)
    fill(c3, QColor(0, 0, 255))
    return folder, c1, c2, c3


def rgb(img):
    c = px(img)
    return (c.red(), c.green(), c.blue())


# ── anim.py ──────────────────────────────────────────────────────────────

class TestFolderAndCels:
    def test_make_folder_puts_cel1_on_current_frame(self):
        st = make_stack()
        folder = anim.make_anim_folder(st)
        assert anim.is_anim_folder(folder)
        assert folder.pass_through
        assert [c.name for c in folder.children] == ["1"]
        assert folder.timeline[0] == folder.children[0].uid
        assert len(folder.timeline) == st.anim_length
        assert st.active is folder.children[0]
        # 背景はフォルダの外（全コマに出る）
        assert st.layers[-1].name == "背景"

    def test_new_cel_numbers_and_selects(self):
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        assert [c.name for c in folder.children] == ["3", "2", "1"]
        assert st.active is c3
        assert st.anim_frame == 2

    def test_new_cel_beyond_length_extends(self):
        st = make_stack()
        folder = anim.make_anim_folder(st)
        anim.new_cel(st, folder, st.anim_length)
        assert st.anim_length == anim.DEFAULT_LENGTH + 1
        assert len(folder.timeline) == st.anim_length

    def test_new_cel_copies_source(self):
        st = make_stack()
        folder, c1, *_ = three_cels(st)
        dup = anim.new_cel(st, folder, 3, c1)
        assert rgb(dup.image) == (255, 0, 0)
        assert dup.image.cacheKey() != c1.image.cacheKey()

    def test_anim_folders_found_inside_normal_group(self):
        st = make_stack()
        g = GroupLayer("g", W, H)
        inner = GroupLayer("a", W, H)
        inner.timeline = [None] * 8
        g.children.append(inner)
        st.layers.insert(0, g)
        assert anim.anim_folders(st.layers) == [inner]


class TestFrames:
    def test_hold_shows_previous_cel(self):
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        # コマ 3 以降は指定なし → セル3を出し続ける
        assert anim.cel_at(folder, 5) is c3
        anim.apply_frame(st, 5)
        assert [c.visible for c in folder.children] == [True, False, False]

    def test_blank_shows_nothing(self):
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        anim.set_spec(folder, 1, BLANK)
        assert anim.cel_at(folder, 1) is None
        assert rgb(anim.render_frame(st, 1)) == (255, 255, 255)

    def test_render_frame_and_background(self):
        st = make_stack()
        folder, *_ = three_cels(st)
        assert [rgb(anim.render_frame(st, f)) for f in range(3)] == [
            (255, 0, 0), (0, 255, 0), (0, 0, 255)]
        # 今のコマ（2）は元に戻っている
        assert st.anim_frame == 2
        assert rgb(st.composite()) == (0, 0, 255)

    def test_apply_frame_clamps(self):
        st = make_stack()
        three_cels(st)
        anim.apply_frame(st, 999)
        assert st.anim_frame == st.anim_length - 1
        anim.apply_frame(st, -3)
        assert st.anim_frame == 0

    def test_swap_specs_keeps_neighbors(self):
        """入れ替えても前後のコマの見た目は変わらない。"""
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        before = [anim.cel_at(folder, f) for f in range(st.anim_length)]
        anim.swap_specs(folder, 0, 2)
        after = [anim.cel_at(folder, f) for f in range(st.anim_length)]
        assert after[0] is c3 and after[2] is c1
        assert after[1] is c2
        assert after[3:] == before[3:]

    def test_swap_with_hold_frame(self):
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        anim.swap_specs(folder, 0, 5)   # 5 は c3 の続き
        assert anim.cel_at(folder, 0) is c3
        assert anim.cel_at(folder, 5) is c1
        assert anim.cel_at(folder, 4) is c3
        assert anim.cel_at(folder, 6) is c3

    def test_insert_and_delete_frame(self):
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        anim.insert_frame(st, 1)
        assert st.anim_length == anim.DEFAULT_LENGTH + 1
        assert [anim.cel_at(folder, f) for f in range(4)] == [c1, c1, c2, c3]
        anim.delete_frame(st, 1)
        assert st.anim_length == anim.DEFAULT_LENGTH
        assert [anim.cel_at(folder, f) for f in range(3)] == [c1, c2, c3]

    def test_delete_frame_carries_spec_to_hold(self):
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        # セル3を出し始めるコマを抜いても、続きのコマはセル3のまま
        anim.delete_frame(st, 2)
        assert [anim.cel_at(folder, f) for f in range(4)] == [c1, c2, c3, c3]
        # 次のコマに指定があれば引き継がない
        st2 = make_stack()
        f2, d1, d2, d3 = three_cels(st2)
        anim.delete_frame(st2, 1)
        assert [anim.cel_at(f2, f) for f in range(3)] == [d1, d3, d3]

    def test_delete_last_remaining_frame_noop(self):
        st = make_stack()
        anim.make_anim_folder(st)
        anim.set_length(st, 1)
        anim.delete_frame(st, 0)
        assert st.anim_length == 1

    def test_retime(self):
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        anim.retime(st, folder, 3)
        assert st.anim_length == 9
        assert [anim.cel_at(folder, f) for f in range(9)] == [c1] * 3 + [c2] * 3 + [c3] * 3
        anim.retime(st, folder, 1)
        assert [anim.cel_at(folder, f) for f in range(3)] == [c1, c2, c3]

    def test_set_length_truncates(self):
        st = make_stack()
        folder, *_ = three_cels(st)
        anim.set_length(st, 2)
        assert len(folder.timeline) == 2
        assert st.anim_frame == 1
        anim.set_length(st, 5000)
        assert st.anim_length == anim.MAX_LENGTH

    def test_renumber_by_appearance(self):
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        anim.swap_specs(folder, 0, 2)  # 順番: c3, c2, c1
        anim.renumber(st, folder)
        assert (c3.name, c2.name, c1.name) == ("1", "2", "3")
        assert folder.children == [c1, c2, c3]   # 上ほど大きい番号
        assert st.active is c3   # 選択していたレイヤーのまま

    def test_timelines_state_roundtrip(self):
        st = make_stack()
        folder, *_ = three_cels(st)
        state = anim.timelines_state(st)
        anim.set_length(st, 3)
        anim.set_spec(folder, 0, BLANK)
        anim.restore_timelines(st, state)
        assert st.anim_length == anim.DEFAULT_LENGTH
        assert folder.timeline == state["timelines"][folder.uid]
        assert folder.timeline is not state["timelines"][folder.uid]


def names(folder, n):
    return [c.name if (c := anim.cel_at(folder, f)) else "×" for f in range(n)]


class TestMoveCopy:
    def test_span_covers_hold(self):
        st = make_stack()
        folder, *_ = three_cels(st)
        assert anim.span(folder, 0) == (0, 0)
        assert anim.span(folder, 4) == (2, st.anim_length - 1)   # セル3の続き

    def test_move_leaves_blank_and_keeps_rest(self):
        st = make_stack()
        folder, *_ = three_cels(st)
        anim.set_length(st, 4)
        anim.move_specs(st, folder, 0, 0, 3)
        assert names(folder, 4) == ["×", "2", "3", "1"]

    def test_move_into_hold_keeps_following_frames(self):
        st = make_stack()
        folder, *_ = three_cels(st)
        anim.set_length(st, 6)
        anim.move_specs(st, folder, 1, 1, 4)
        assert names(folder, 6) == ["1", "×", "3", "3", "2", "3"]

    def test_move_range_overlapping(self):
        st = make_stack()
        folder, *_ = three_cels(st)
        anim.set_length(st, 4)
        anim.move_specs(st, folder, 0, 1, 1)   # 1,2 を1コマ後ろへ
        assert names(folder, 4) == ["×", "1", "2", "3"]

    def test_copy_keeps_source_and_extends_length(self):
        st = make_stack()
        folder, *_ = three_cels(st)
        anim.set_length(st, 3)
        anim.move_specs(st, folder, 0, 1, 2, copy=True)
        assert st.anim_length == 4
        assert names(folder, 4) == ["1", "2", "1", "2"]

    def test_clear_specs(self):
        st = make_stack()
        folder, *_ = three_cels(st)
        anim.set_length(st, 5)
        anim.clear_specs(folder, 1, 2)
        assert names(folder, 5) == ["1", "×", "×", "3", "3"]

    def test_shown_specs_treats_deleted_cel_as_blank(self):
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        folder.children.remove(c2)
        assert anim.shown_specs(folder)[:3] == [c1.uid, BLANK, c3.uid]


class TestOnion:
    def test_prev_blue_next_green_skip_same(self):
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        items = anim.onion_items(st, 1, 2, 2)
        assert [(c, col) for c, col, _ in items] == [
            (c1, anim.ONION_PREV), (c3, anim.ONION_NEXT)]
        # コマ 3,4 は c3 の続きなので後ろに同じセルを重ねない
        assert len(anim.onion_items(st, 2, 0, 3)) == 0

    def test_loop_wraps(self):
        st = make_stack()
        folder, c1, c2, c3 = three_cels(st)
        anim.set_length(st, 3)
        items = anim.onion_items(st, 0, 1, 0, loop=True)
        assert [c for c, _, _ in items] == [c3]
        assert anim.onion_items(st, 0, 1, 0, loop=False) == []

    def test_tinted(self):
        st = make_stack()
        folder, c1, *_ = three_cels(st)
        img = anim.tinted(c1, W, H, anim.ONION_PREV)
        assert rgb(img) == (40, 90, 255)


class TestMigrate:
    def test_old_frames_become_cels(self):
        st = make_stack()
        frames = []
        for col in (QColor(255, 0, 0), QColor(0, 255, 0)):
            img = QImage(W, H, QImage.Format.Format_ARGB32)
            img.fill(col)
            frames.append(img)
        folder = anim.migrate_frames(st, frames)
        assert st.layers[0] is folder
        assert st.active.name == "背景"   # 選択はずれない
        assert [rgb(anim.render_frame(st, f)) for f in range(2)] == [
            (255, 0, 0), (0, 255, 0)]
        assert anim.migrate_frames(st, []) is None


# ── パネル ──────────────────────────────────────────────────────────────

@pytest.fixture
def panel():
    st = make_stack()
    p = AnimationPanel(st)
    return p


class TestPanel:
    def test_new_folder_and_cel_buttons(self, panel):
        panel._on_new_cel()   # フォルダが無ければフォルダから作る
        folder = anim.anim_folders(panel.stack.layers)[0]
        assert len(folder.children) == 1
        panel._on_new_cel()   # 今のコマは埋まっている → 次のコマ
        assert folder.timeline[1] == folder.children[0].uid
        assert panel.stack.anim_frame == 1

    def test_dup_cel(self, panel):
        st = panel.stack
        folder, c1, c2, c3 = three_cels(st)
        panel.go_to(0)
        panel._on_dup_cel()
        assert anim.cel_at(folder, 1).name == "4"
        assert rgb(anim.cel_at(folder, 1).image) == (255, 0, 0)

    def test_go_to_selects_shown_cel(self, panel):
        st = panel.stack
        folder, c1, c2, c3 = three_cels(st)
        changed = []
        panel.frame_changed.connect(lambda: changed.append(1))
        panel.go_to(1)
        assert st.active is c2 and changed
        panel.next_frame()
        assert st.active is c3
        panel.prev_frame()
        panel.prev_frame()
        assert st.active is c1
        panel.prev_frame()   # ループで最後へ
        assert st.anim_frame == st.anim_length - 1

    def test_edit_emits_undo_signal(self, panel):
        st = panel.stack
        folder, *_ = three_cels(st)
        calls = []
        panel.anim_will_change.connect(lambda: calls.append(1))
        panel._on_cell_clicked(folder, 0)
        panel._on_cell_dragged(folder, 5)
        assert calls == [1]
        assert st.anim_frame == 5

    def test_double_click_empty_cell_makes_cel(self, panel):
        st = panel.stack
        folder, *_ = three_cels(st)
        anim.set_spec(folder, 4, BLANK)
        panel._on_cell_double_clicked(folder, 5)   # 空きの続き → 新しいセル
        assert anim.cel_at(folder, 5).name == "4"
        panel._on_cell_double_clicked(folder, 1)   # 埋まっている → 何もしない
        assert len(folder.children) == 4

    def test_length_spin(self, panel):
        three_cels(panel.stack)
        panel._len_spin.setValue(12)
        assert panel.stack.anim_length == 12
        assert all(len(f.timeline) == 12 for f in anim.anim_folders(panel.stack.layers))

    def test_context_menu_items(self, panel):
        from PyQt6.QtWidgets import QMenu
        folder, *_ = three_cels(panel.stack)
        m = QMenu()
        panel._fill_edit_menu(m, folder, 0)
        texts = [a.text() for a in m.actions()]
        assert any("空きコマ" in t for t in texts)
        assert any("2コマ打ち" in t for t in texts)

    def test_timeline_paints(self, panel):
        three_cels(panel.stack)
        panel.sync()
        img = panel._view.grab()
        assert not img.isNull()

    def test_hit_test(self, panel):
        from PyQt6.QtCore import QPoint
        from animation_panel import _NAME_W, _CELL_W, _HEADER_H, _ROW_H
        folder, *_ = three_cels(panel.stack)
        panel.sync()
        kind, f_, fr = panel._view.hit(QPoint(_NAME_W + 2 * _CELL_W + 5, _HEADER_H + 5))
        assert (kind, f_, fr) == ("cell", folder, 2)
        assert panel._view.hit(QPoint(_NAME_W + 5, 3))[0] == "header"


class TestSelectMovePaste:
    def test_click_selects_span_and_shift_extends(self, panel):
        st = panel.stack
        folder, *_ = three_cels(st)
        panel._on_cell_clicked(folder, 4)
        assert panel.selection() == (folder, 2, st.anim_length - 1)
        panel._on_cell_clicked(folder, 0)
        assert panel.selection() == (folder, 0, 0)
        panel._on_cell_clicked(folder, 1, True)
        assert panel.selection() == (folder, 0, 1)

    def test_drag_moves_selection_and_ctrl_copies(self, panel):
        st = panel.stack
        folder, *_ = three_cels(st)
        anim.set_length(st, 6)
        panel.sync()
        panel._on_cell_clicked(folder, 0)
        panel._on_cell_clicked(folder, 1, True)
        panel._on_cell_dragged(folder, 4)
        assert names(folder, 6) == ["×", "×", "3", "3", "1", "2"]
        assert panel.selection() == (folder, 4, 5)
        assert st.anim_frame == 4
        panel._on_cell_dragged(folder, 0, True)
        assert names(folder, 6) == ["1", "2", "3", "3", "1", "2"]

    def test_mouse_drag_on_view(self, panel):
        from PyQt6.QtTest import QTest
        st = panel.stack
        folder, *_ = three_cels(st)
        anim.set_length(st, 4)
        panel.sync()
        view = panel._view
        y = 18 + 25
        x = lambda f: 120 + f * 56 + 28
        QTest.mousePress(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         QPoint(x(0), y))
        QTest.mouseMove(view, QPoint(x(3), y))
        QTest.mouseRelease(view, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                           QPoint(x(3), y))
        assert names(folder, 4) == ["×", "2", "3", "1"]

    def test_copy_paste_duplicates_cels(self, panel):
        st = panel.stack
        folder, c1, c2, c3 = three_cels(st)
        anim.set_length(st, 5)
        panel.sync()
        panel._on_cell_clicked(folder, 0)
        panel._on_cell_clicked(folder, 1, True)
        assert panel.copy_frames()
        panel.go_to(3, folder)
        panel.paste_frames()
        assert names(folder, 5) == ["1", "2", "3", "4", "5"]
        new4, new5 = anim.cel_at(folder, 3), anim.cel_at(folder, 4)
        assert new4 is not c1 and rgb(new4.image) == (255, 0, 0)
        assert rgb(new5.image) == (0, 255, 0)
        # 複製なので描き直しても元のセルは変わらない
        fill(new4, QColor(0, 0, 0))
        assert rgb(c1.image) == (255, 0, 0)
        assert panel.selection() == (folder, 3, 4)
        assert st.active is new4

    def test_paste_reuse_and_into_other_folder(self, panel):
        st = panel.stack
        folder, c1, c2, c3 = three_cels(st)
        panel._on_cell_clicked(folder, 0)
        panel.copy_frames()
        panel.go_to(2, folder)
        panel.paste_frames(reuse=True)
        assert anim.cel_at(folder, 2) is c1 and len(folder.children) == 3
        # 別のフォルダへは使い回せないので複製になる
        other = anim.make_anim_folder(st)
        panel.go_to(1, other)
        panel.paste_frames(reuse=True)
        pasted = anim.cel_at(other, 1)
        assert pasted is not c1 and rgb(pasted.image) == (255, 0, 0)

    def test_cut_and_delete(self, panel):
        st = panel.stack
        folder, *_ = three_cels(st)
        anim.set_length(st, 3)
        panel.sync()
        panel._on_cell_clicked(folder, 1)
        panel.cut_frames()
        assert names(folder, 3) == ["1", "×", "3"]
        panel.go_to(0, folder)
        panel._on_cell_clicked(folder, 0)
        panel.paste_frames(reuse=True)
        assert names(folder, 3) == ["2", "×", "3"]
        panel._on_cell_clicked(folder, 2)
        panel.clear_frames()
        assert names(folder, 3) == ["2", "×", "×"]

    def test_keys_beat_window_shortcuts(self, panel):
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import QEvent
        st = panel.stack
        folder, *_ = three_cels(st)
        anim.set_length(st, 3)
        panel.sync()
        view = panel._view
        ev = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_C,
                       Qt.KeyboardModifier.ControlModifier)
        QApplication.sendEvent(view, ev)
        assert ev.isAccepted()
        panel._on_cell_clicked(folder, 0)

        def key(k, mods=Qt.KeyboardModifier.NoModifier):
            QApplication.sendEvent(view, QKeyEvent(QEvent.Type.KeyPress, k, mods))

        key(Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
        key(Qt.Key.Key_Right)
        key(Qt.Key.Key_Right)
        assert st.anim_frame == 2
        key(Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.ShiftModifier)
        assert names(folder, 3) == ["1", "2", "1"]
        key(Qt.Key.Key_Delete)
        assert names(folder, 3) == ["1", "2", "×"]
        # 関係ないキーは素通し
        ev = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_B,
                       Qt.KeyboardModifier.NoModifier)
        ev.ignore()
        QApplication.sendEvent(view, ev)
        assert not ev.isAccepted()

    def test_menu_has_copy_paste(self, panel):
        from PyQt6.QtWidgets import QMenu
        folder, *_ = three_cels(panel.stack)
        m = QMenu()
        panel._fill_edit_menu(m, folder, 0)
        acts = {a.text().split("（")[0]: a for a in m.actions()}
        assert not acts["貼り付け"].isEnabled()
        panel._on_cell_clicked(folder, 0)
        panel.copy_frames()
        panel._fill_edit_menu(m, folder, 0)
        assert all(a.isEnabled() for a in m.actions() if a.text().startswith("貼り付け"))

    def test_selection_dropped_when_folder_gone(self, panel):
        st = panel.stack
        folder, *_ = three_cels(st)
        panel._on_cell_clicked(folder, 0)
        st.layers.remove(folder)
        panel.sync()
        assert panel.selection() is None


class TestOnionOverlay:
    def test_reused_until_something_changes(self, panel):
        folder, c1, c2, c3 = three_cels(panel.stack)
        panel.go_to(1)
        ov = panel.onion_overlay()
        assert ov is not None
        assert panel.onion_overlay() is ov
        panel._on_onion_before(0)
        ov2 = panel.onion_overlay()
        assert ov2 is not ov
        fill(c3, QColor(0, 0, 0))
        assert panel.onion_overlay() is not ov2

    def test_none_when_off_or_hidden_or_playing(self, panel):
        three_cels(panel.stack)
        panel.go_to(1)
        panel._onion_enabled = False
        assert panel.onion_overlay() is None
        panel._onion_enabled = True
        panel.setVisible(False)
        assert panel.onion_overlay() is None
        panel.setVisible(True)
        assert panel.onion_overlay() is not None
        panel._start_play()
        assert panel.onion_overlay() is None
        panel._stop_play()


class TestPlayback:
    def test_play_advances_and_wraps(self, panel):
        st = panel.stack
        three_cels(st)
        anim.set_length(st, 3)
        panel.go_to(1)
        panel._start_play()
        assert panel._playing
        panel._play_next()
        assert panel.playing_frame() == 2
        assert rgb(panel.playback_image()) == (0, 0, 255)
        panel._play_next()
        assert panel.playing_frame() == 0
        # 再生中でもレイヤーの表示は今のコマのまま
        assert st.anim_frame == 1
        panel._stop_play()
        assert not panel._playing and panel.playback_image() is None
        assert st.anim_frame == 0   # 止めたコマに移る

    def test_no_loop_stops_at_end(self, panel):
        st = panel.stack
        three_cels(st)
        anim.set_length(st, 3)
        panel._loop_btn.setChecked(False)
        panel.go_to(2)
        panel._start_play()
        panel._play_next()
        assert not panel._playing

    def test_play_without_folder_does_nothing(self, panel):
        panel._start_play()
        assert not panel._playing


class TestExport:
    def test_no_folder_shows_message(self, panel, monkeypatch):
        called = []
        monkeypatch.setattr("animation_panel.QMessageBox.information",
                            lambda *a: called.append(True))
        panel._on_export_gif()
        assert called

    def test_gif(self, panel, tmp_path, monkeypatch):
        pytest.importorskip("PIL")
        monkeypatch.setattr("animation_panel.QMessageBox.information", lambda *a: None)
        three_cels(panel.stack)
        anim.set_length(panel.stack, 3)
        path = str(tmp_path / "a.gif")
        assert panel._export_gif_to(path)
        from PIL import Image
        with Image.open(path) as im:
            assert im.n_frames == 3

    def test_png_sequence(self, panel, tmp_path, monkeypatch):
        monkeypatch.setattr("animation_panel.QMessageBox.information", lambda *a: None)
        three_cels(panel.stack)
        anim.set_length(panel.stack, 3)
        assert panel._export_png_sequence_to(str(tmp_path)) == 3
        names = sorted(os.listdir(tmp_path))
        assert names == ["frame_001.png", "frame_002.png", "frame_003.png"]
        assert rgb(QImage(str(tmp_path / "frame_002.png"))) == (0, 255, 0)


# ── アプリに組み込んだ状態 ─────────────────────────────────────────────

@pytest.fixture
def win(monkeypatch):
    from PyQt6.QtCore import QSettings
    monkeypatch.setattr(QSettings, "setValue", lambda *a, **k: None)
    import main as M
    w = M.MainWindow()
    w._anim_mode_action.setChecked(True)
    yield w
    w._dirty = False
    w.close()
    w.deleteLater()


class TestInApp:
    def test_undo_timeline_edit(self, win):
        st = win.layer_stack
        ap = win.anim_panel
        ap._on_new_folder()
        folder = anim.anim_folders(st.layers)[0]
        ap._on_new_cel()
        before = list(folder.timeline)
        ap._edit(lambda: anim.set_spec(folder, 0, BLANK))
        assert win.canvas._history[-1][0] == "anim"
        win.canvas.undo()
        assert folder.timeline == before
        win.canvas.redo()
        assert folder.timeline[0] == BLANK

    def test_undo_new_cel_restores_timeline(self, win):
        st = win.layer_stack
        ap = win.anim_panel
        ap._on_new_folder()
        ap._on_new_cel()
        win.canvas.undo()   # セル2を足す前へ
        folder = anim.anim_folders(st.layers)[0]
        assert len(folder.children) == 1
        assert folder.timeline[1] is None
        win.canvas.undo()   # フォルダを作る前へ
        assert anim.anim_folders(st.layers) == []

    def test_save_and_load(self, win, tmp_path):
        st = win.layer_stack
        ap = win.anim_panel
        ap._on_new_folder()
        folder = anim.anim_folders(st.layers)[0]
        fill(folder.children[0], QColor(255, 0, 0))
        ap._on_new_cel()
        fill(folder.children[0], QColor(0, 0, 255))
        anim.set_spec(folder, 3, BLANK)
        anim.set_length(st, 5)
        ap.set_fps(12)
        ap.go_to(1)
        path = str(tmp_path / "a.pola")
        win._write_pola(path)
        with zipfile.ZipFile(path) as zf:
            meta = json.loads(zf.read("meta.json"))
        assert meta["anim"] == {"fps": 12, "length": 5, "frame": 1}
        win._init_canvas(W, H)
        win._document_replaced(None)
        assert st.anim_length == anim.DEFAULT_LENGTH
        win._load_pola(path)
        folder = anim.anim_folders(st.layers)[0]
        assert st.anim_length == 5 and st.anim_frame == 1
        assert ap.fps == 12
        assert [anim.cel_at(folder, f).name if anim.cel_at(folder, f) else None
                for f in range(5)] == ["1", "2", "2", None, None]
        cw, ch = st.width, st.height
        assert rgb(anim.render_frame(st, 0)) == (255, 0, 0)
        assert rgb(anim.render_frame(st, 1)) == (0, 0, 255)

    def test_load_old_format_migrates(self, win, tmp_path):
        st = win.layer_stack
        path = str(tmp_path / "old.pola")
        win._write_pola(path)
        # 旧形式: meta["anim"]["frames"] に合成済みのコマ
        with zipfile.ZipFile(path) as zf:
            meta = json.loads(zf.read("meta.json"))
            files = {n: zf.read(n) for n in zf.namelist() if n != "meta.json"}
        from PyQt6.QtCore import QBuffer, QByteArray, QIODevice
        names = []
        for i, col in enumerate((QColor(255, 0, 0), QColor(0, 255, 0))):
            img = QImage(st.width, st.height, QImage.Format.Format_ARGB32)
            img.fill(col)
            buf = QByteArray()
            bio = QBuffer(buf)
            bio.open(QIODevice.OpenModeFlag.WriteOnly)
            img.save(bio, "PNG")
            bio.close()
            files[f"frame_{i}.png"] = bytes(buf)
            names.append(f"frame_{i}.png")
        meta["anim"] = {"frames": names, "fps": 6}
        with zipfile.ZipFile(path, "w") as zf:
            for n, b in files.items():
                zf.writestr(n, b)
            zf.writestr("meta.json", json.dumps(meta))
        win._load_pola(path)
        folders = anim.anim_folders(st.layers)
        assert len(folders) == 1 and len(folders[0].children) == 2
        assert win.anim_panel.fps == 6
        assert rgb(anim.render_frame(st, 1)) == (0, 255, 0)

    def test_new_document_resets(self, win):
        st = win.layer_stack
        win.anim_panel._on_new_folder()
        anim.set_length(st, 20)
        win._init_canvas(W, H)
        win._document_replaced(None)
        assert st.anim_length == anim.DEFAULT_LENGTH and st.anim_frame == 0
        assert anim.anim_folders(st.layers) == []

    def test_frame_shortcut_actions(self, win):
        st = win.layer_stack
        win.anim_panel._on_new_folder()
        acts = {a.text(): a for a in win.menuBar().actions()}
        menu = acts["アニメーション"].menu()
        nxt = next(a for a in menu.actions() if a.text() == "次のコマ")
        assert nxt.shortcut().toString() == "."
        nxt.trigger()
        assert st.anim_frame == 1

    def test_canvas_shows_playback_and_onion(self, win):
        st = win.layer_stack
        ap = win.anim_panel
        ap._on_new_folder()
        folder = anim.anim_folders(st.layers)[0]
        fill(folder.children[0], QColor(255, 0, 0))
        ap._on_new_cel()
        calls = []
        orig = ap.onion_overlay
        win.canvas._get_onion_overlay = lambda: calls.append(orig()) or calls[-1]
        win.canvas.grab()
        assert calls and calls[-1] is not None
        win._anim_mode_action.setChecked(False)
        calls.clear()
        win.canvas.grab()
        assert all(r is None for r in calls)

    def test_duplicate_folder_keeps_timeline(self, win):
        st = win.layer_stack
        ap = win.anim_panel
        ap._on_new_folder()
        ap._on_new_cel()
        folder = anim.anim_folders(st.layers)[0]
        copy = win.layer_panel._deep_copy_layer(folder, st)
        assert anim.cel_at(copy, 1).name == "2"
        assert anim.cel_at(copy, 1) is not anim.cel_at(folder, 1)

    def test_undo_paste_removes_copied_cels(self, win):
        st = win.layer_stack
        ap = win.anim_panel
        ap._on_new_folder()
        folder = anim.anim_folders(st.layers)[0]
        ap._on_cell_clicked(folder, 0)
        ap.copy_frames()
        ap.go_to(2, folder)
        ap.paste_frames()
        assert len(folder.children) == 2
        win.canvas.undo()
        folder = anim.anim_folders(st.layers)[0]
        assert len(folder.children) == 1
        assert anim.cel_at(folder, 2).name == "1"   # セル1の続きに戻る

    def test_timeline_copy_does_not_trigger_canvas_copy(self, win):
        """タイムラインにフォーカスがあるときの Ctrl+C はキャンバスへ行かない。"""
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import QEvent
        ev = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_V,
                       Qt.KeyboardModifier.ControlModifier)
        QApplication.sendEvent(win.anim_panel._view, ev)
        assert ev.isAccepted()
