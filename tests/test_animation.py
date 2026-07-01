"""アニメーションパネルのテスト"""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage, QColor, QPainter
from PyQt6.QtCore import Qt

app = QApplication.instance() or QApplication(sys.argv)

from animation_panel import AnimationPanel

W, H = 100, 100


def make_image(color: QColor) -> QImage:
    img = QImage(W, H, QImage.Format.Format_ARGB32)
    p = QPainter(img)
    p.fillRect(0, 0, W, H, color)
    p.end()
    return img


def px(img: QImage, x: int, y: int) -> QColor:
    return QColor.fromRgba(img.pixel(x, y))


@pytest.fixture
def panel():
    p = AnimationPanel()
    p.set_composite_fn(lambda: make_image(QColor(255, 0, 0, 255)))
    return p


class TestFrameManagement:
    def test_add_frame(self, panel):
        panel._on_add_frame()
        assert len(panel.frames) == 1
        assert panel.current_frame == 0

    def test_add_multiple_frames(self, panel):
        panel._on_add_frame()
        panel._on_add_frame()
        panel._on_add_frame()
        assert len(panel.frames) == 3
        assert panel.current_frame == 2

    def test_add_frame_inserts_after_current(self, panel):
        panel._on_add_frame()
        panel._on_add_frame()
        panel._on_add_frame()
        # select frame 0, then add
        panel.current_frame = 0
        panel.set_composite_fn(lambda: make_image(QColor(0, 0, 255, 255)))
        panel._on_add_frame()
        assert len(panel.frames) == 4
        assert panel.current_frame == 1
        # new frame should be blue
        c = px(panel.frames[1], 50, 50)
        assert c.blue() > 200

    def test_delete_frame(self, panel):
        panel._on_add_frame()
        panel._on_add_frame()
        panel.current_frame = 0
        panel._on_delete_frame()
        assert len(panel.frames) == 1
        assert panel.current_frame == 0

    def test_delete_last_frame(self, panel):
        panel._on_add_frame()
        panel._on_delete_frame()
        assert len(panel.frames) == 0
        assert panel.current_frame == -1

    def test_delete_no_frames_noop(self, panel):
        panel._on_delete_frame()
        assert len(panel.frames) == 0

    def test_replace_frame(self, panel):
        panel._on_add_frame()
        original_pixel = panel.frames[0].pixel(50, 50)
        panel.set_composite_fn(lambda: make_image(QColor(0, 255, 0, 255)))
        panel._on_replace_frame()
        c = px(panel.frames[0], 50, 50)
        assert c.green() > 200

    def test_replace_no_frame_noop(self, panel):
        panel._on_replace_frame()  # should not crash

    def test_move_left(self, panel):
        panel.set_composite_fn(lambda: make_image(QColor(255, 0, 0, 255)))
        panel._on_add_frame()
        panel.set_composite_fn(lambda: make_image(QColor(0, 255, 0, 255)))
        panel._on_add_frame()
        # frame 1 (green) is selected
        panel._on_move_left()
        assert panel.current_frame == 0
        c = px(panel.frames[0], 50, 50)
        assert c.green() > 200

    def test_move_left_at_start_noop(self, panel):
        panel._on_add_frame()
        panel.current_frame = 0
        panel._on_move_left()
        assert panel.current_frame == 0

    def test_move_right(self, panel):
        panel.set_composite_fn(lambda: make_image(QColor(255, 0, 0, 255)))
        panel._on_add_frame()
        panel.set_composite_fn(lambda: make_image(QColor(0, 255, 0, 255)))
        panel._on_add_frame()
        panel.current_frame = 0  # select red
        panel._on_move_right()
        assert panel.current_frame == 1
        c = px(panel.frames[1], 50, 50)
        assert c.red() > 200

    def test_move_right_at_end_noop(self, panel):
        panel._on_add_frame()
        panel._on_move_right()
        assert panel.current_frame == 0


class TestOnionSkin:
    def test_onion_disabled_returns_empty(self, panel):
        panel._on_add_frame()
        panel._on_add_frame()
        panel._onion_enabled = False
        assert panel.get_onion_images() == []

    def test_onion_no_frames_returns_empty(self, panel):
        assert panel.get_onion_images() == []

    def test_onion_first_frame_returns_empty(self, panel):
        panel._on_add_frame()
        panel.current_frame = 0
        assert panel.get_onion_images() == []

    def test_onion_returns_previous_frames(self, panel):
        panel.set_composite_fn(lambda: make_image(QColor(255, 0, 0, 255)))
        panel._on_add_frame()
        panel.set_composite_fn(lambda: make_image(QColor(0, 255, 0, 255)))
        panel._on_add_frame()
        panel.set_composite_fn(lambda: make_image(QColor(0, 0, 255, 255)))
        panel._on_add_frame()
        # current = frame 2 (blue), should get frame 1 (green) and frame 0 (red)
        onion = panel.get_onion_images()
        assert len(onion) == 2
        # first entry is the most recent previous frame
        c0 = px(onion[0][0], 50, 50)
        assert c0.green() > 200
        c1 = px(onion[1][0], 50, 50)
        assert c1.red() > 200
        # opacity decreases
        assert onion[0][1] > onion[1][1]

    def test_onion_count_limits_results(self, panel):
        for _ in range(5):
            panel._on_add_frame()
        panel._onion_count = 1
        onion = panel.get_onion_images()
        assert len(onion) == 1


class TestPlayback:
    def test_play_starts_and_stops(self, panel):
        panel._on_add_frame()
        panel._on_add_frame()
        panel._start_play()
        assert panel._playing
        panel._stop_play()
        assert not panel._playing

    def test_play_next_wraps(self, panel):
        panel._on_add_frame()
        panel._on_add_frame()
        panel._on_add_frame()
        panel.current_frame = 2
        panel._play_next()
        assert panel.current_frame == 0

    def test_play_next_advances(self, panel):
        panel._on_add_frame()
        panel._on_add_frame()
        panel.current_frame = 0
        panel._play_next()
        assert panel.current_frame == 1

    def test_play_empty_stops(self, panel):
        panel._playing = True
        panel._play_next()
        assert not panel._playing


class TestClear:
    def test_clear_removes_all(self, panel):
        panel._on_add_frame()
        panel._on_add_frame()
        panel.clear()
        assert len(panel.frames) == 0
        assert panel.current_frame == -1


class TestExportGif:
    def test_export_no_frames_shows_message(self, panel, monkeypatch):
        # monkeypatch QMessageBox to avoid blocking
        called = []
        monkeypatch.setattr(
            "animation_panel.QMessageBox.information",
            lambda *a: called.append(True))
        panel._on_export_gif()
        assert called

    def test_export_gif_creates_file(self, panel):
        try:
            import PIL  # noqa: F401
        except ImportError:
            pytest.skip("Pillow not installed")
        panel._on_add_frame()
        panel._on_add_frame()
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
            path = f.name
        try:
            panel._export_gif_to(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)
