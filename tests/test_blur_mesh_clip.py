"""ぼかしブラシの黒い濁り・メッシュ変形の分身・クリッピングの線のふち塗り残しの回帰テスト。"""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF, QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter
from PyQt6.QtWidgets import QApplication

from brush import BlurBrush
from canvas import Canvas
from layer import Layer, LayerStack


@pytest.fixture(scope="module", autouse=True)
def app():
    yield QApplication.instance() or QApplication([])


def to_np(img):
    img = img.convertToFormat(QImage.Format.Format_ARGB32)
    b = img.constBits()
    b.setsize(img.sizeInBytes())
    return np.frombuffer(b, dtype=np.uint8).reshape(
        img.height(), img.bytesPerLine() // 4, 4)[:, :img.width()].copy()


def blank(w, h):
    img = QImage(w, h, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    return img


class TestBlurBrush:
    def test_edge_next_to_transparency_keeps_color(self):
        img = blank(60, 60)
        p = QPainter(img)
        p.fillRect(0, 0, 30, 60, QColor(255, 200, 0))
        p.end()
        BlurBrush(1.0).stroke_to(img, QPoint(30, 10), QPoint(30, 50),
                                 QColor(0, 0, 0), 20)
        a = to_np(img)
        vis = a[:, :, 3] > 8
        assert vis[:, 30:40].any()  # ふちがぼけて外へ広がる
        # 色（BGRA）は黒くならず元の黄色のまま
        assert a[vis][:, 2].min() >= 250
        assert a[vis][:, 1].min() >= 195
        assert a[vis][:, 0].max() <= 5

    def test_blurs_between_two_colors(self):
        img = blank(60, 60)
        p = QPainter(img)
        p.fillRect(0, 0, 30, 60, QColor(255, 0, 0))
        p.fillRect(30, 0, 30, 60, QColor(0, 0, 255))
        p.end()
        BlurBrush(1.0).stroke_to(img, QPoint(30, 30), QPoint(30, 30),
                                 QColor(0, 0, 0), 20)
        a = to_np(img)
        assert a[30, 30, 3] == 255
        assert 60 < a[30, 30, 2] < 200 and 60 < a[30, 30, 0] < 200


def mesh_owner(img, grid, n):
    return SimpleNamespace(_mesh_grid=grid, _transform_image=img,
                           _transform_rect=QRect(0, 0, img.width(), img.height()),
                           _mesh_div=n)


def even_grid(w, h, n):
    return [[QPointF(w * c / n, h * r / n) for c in range(n + 1)] for r in range(n + 1)]


class TestMeshWarp:
    def _img(self):
        img = blank(90, 90)
        p = QPainter(img)
        p.fillRect(40, 40, 10, 10, QColor(255, 0, 0))
        p.end()
        return img

    def test_identity_grid_keeps_image(self):
        img = self._img()
        out, ox, oy = Canvas._warp_mesh_image(mesh_owner(img, even_grid(90, 90, 3), 3))
        a = to_np(out)
        assert (ox, oy) == (0, 0)
        assert (a[:, :, 3] > 128).sum() == 100
        assert a[45, 45, 3] == 255 and a[45, 45, 2] == 255

    def test_moved_point_makes_no_copies(self):
        img = self._img()
        grid = even_grid(90, 90, 3)
        grid[1][1] = QPointF(25, 25)  # 内側の格子点を少しずらす
        out, ox, oy = Canvas._warp_mesh_image(mesh_owner(img, grid, 3))
        a = to_np(out)
        ys, xs = np.nonzero(a[:, :, 3] > 30)
        # 赤い四角は中央のセルにしかない。分身が他のセルに出ない
        assert len(xs) > 0
        assert xs.min() + ox >= 30 and ys.min() + oy >= 30
        assert xs.max() + ox < 60 and ys.max() + oy < 60


class TestClipping:
    def test_clip_color_covers_line_edges(self):
        stack = LayerStack(40, 40)
        bg = Layer("紙", 40, 40)
        bg.image.fill(QColor(255, 255, 255))
        line = Layer("線画", 40, 40)
        line.image.fill(QColor(0, 0, 0, 128))  # 線のふち（半透明）
        color = Layer("色", 40, 40)
        color.image.fill(QColor(255, 0, 0))
        color.clipping = True
        stack.layers = [color, line, bg]
        a = to_np(stack.composite())
        # 白の上に半透明の赤 → ピンク。下地の黒が混ざって暗くならない
        b, g, r = a[20, 20, :3]
        assert r >= 250
        assert 120 <= g <= 135 and 120 <= b <= 135

    def test_multiply_clip_still_works(self):
        stack = LayerStack(20, 20)
        paper = Layer("紙", 20, 20)
        paper.image.fill(QColor(255, 255, 255))
        base = Layer("下地", 20, 20)
        base.image.fill(QColor(200, 200, 200))
        shade = Layer("影", 20, 20)
        shade.image.fill(QColor(128, 255, 255))
        shade.clipping = True
        shade.blend_mode = "multiply"
        stack.layers = [shade, base, paper]
        b, g, r = to_np(stack.composite())[10, 10, :3]
        assert 98 <= r <= 102 and g == 200 and b == 200

    def test_clip_outside_base_is_hidden(self):
        stack = LayerStack(20, 20)
        paper = Layer("紙", 20, 20)
        paper.image.fill(QColor(255, 255, 255))
        base = Layer("下地", 20, 20)
        base.image.fill(Qt.GlobalColor.transparent)
        p = QPainter(base.image)
        p.fillRect(0, 0, 10, 20, QColor(0, 0, 0))
        p.end()
        clip = Layer("色", 20, 20)
        clip.image.fill(QColor(0, 0, 255))
        clip.clipping = True
        stack.layers = [clip, base, paper]
        a = to_np(stack.composite())
        assert tuple(a[10, 5, :3]) == (255, 0, 0)       # BGR: 青
        assert tuple(a[10, 15, :3]) == (255, 255, 255)  # 下地の外は白のまま
