"""解像度（dpi）・保存先フォルダの記憶・用紙サイズ候補・選択範囲での切り抜きのテスト。"""
import os
import sys
import zipfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QImage, QColor
from PyQt6.QtWidgets import QApplication, QDialog, QFileDialog

app = QApplication.instance() or QApplication(sys.argv)

import main as M
from layer import DEFAULT_DPI


class FakeSettings:
    def __init__(self, **values):
        self.d = dict(values)

    def value(self, key, default=None, *a, **k):
        return self.d.get(key, default)

    def setValue(self, key, value):
        self.d[key] = value

    def sync(self):
        pass


@pytest.fixture
def win():
    w = M.MainWindow()
    w._settings = FakeSettings()
    yield w
    w._dirty = False
    w.close()
    w.deleteLater()


def _small_canvas(win, w=100, h=80):
    win._init_canvas(w, h)
    win._document_replaced(None)


def _dpi_of(img: QImage) -> int:
    return round(img.dotsPerMeterX() * 0.0254)


# ── 用紙サイズの計算 ───────────────────────────────────────────────────

class TestPaperSize:
    def test_a4_300dpi_matches_known_pixels(self):
        assert M.paper_size_px("A4", 300) == (2480, 3508)
        assert M.paper_size_px("A4", 300, landscape=True) == (3508, 2480)

    def test_b5_350dpi(self):
        assert M.paper_size_px("B5（JIS）", 350) == (2508, 3541)

    def test_never_exceeds_canvas_limit(self):
        w, h = M.paper_size_px("A3", 1200)
        assert w <= M.MAX_CANVAS_PX and h <= M.MAX_CANVAS_PX


# ── 新規作成ダイアログ ─────────────────────────────────────────────────

class TestNewCanvasDialog:
    def test_paper_follows_dpi_and_orientation(self):
        d = M.NewCanvasDialog()
        d._preset.setCurrentText("A4")
        d._dpi.setCurrentText("350")
        d._orient.setCurrentIndex(1)
        assert d.dpi() == 350
        assert d.values() == M.paper_size_px("A4", 350, landscape=True)
        assert not d._w.isEnabled()
        assert "297.0 x 210.0 mm" in d._print_label.text()

    def test_custom_size_is_editable(self):
        d = M.NewCanvasDialog()
        d._preset.setCurrentText("カスタム")
        assert d._w.isEnabled() and d._h.isEnabled()
        d._w.setValue(1234)
        d._h.setValue(567)
        assert d.values() == (1234, 567)

    def test_bad_dpi_text_falls_back(self):
        d = M.NewCanvasDialog()
        d._dpi.setEditText("")
        assert d.dpi() == DEFAULT_DPI

    def test_new_document_gets_chosen_dpi(self, win, monkeypatch):
        class FakeDlg:
            def __init__(self, *a, **k):
                pass

            def exec(self):
                return QDialog.DialogCode.Accepted

            def values(self):
                return (500, 400)

            def dpi(self):
                return 350

        monkeypatch.setattr(M, "NewCanvasDialog", FakeDlg)
        win._new()
        assert win.layer_stack.dpi == 350
        assert (win.layer_stack.width, win.layer_stack.height) == (500, 400)
        assert win._status_dpi.text() == "350 dpi"


# ── 保存・読み込みでの解像度とサイズ ────────────────────────────────

class TestDpiPersistence:
    def test_pola_round_trip_keeps_size_and_dpi(self, win, tmp_path):
        _small_canvas(win, 321, 123)
        win.layer_stack.dpi = 600
        path = str(tmp_path / "a.pola")
        win._write_pola(path)
        _small_canvas(win, 50, 50)
        win._load_pola(path)
        ls = win.layer_stack
        assert (ls.width, ls.height, ls.dpi) == (321, 123, 600)
        assert win._status_canvas_size.text() == "321 x 123 px"
        assert win._status_dpi.text() == "600 dpi"

    def test_old_pola_without_dpi_opens_with_default(self, win, tmp_path):
        _small_canvas(win, 64, 32)
        win.layer_stack.dpi = 72
        path = str(tmp_path / "old.pola")
        win._write_pola(path)
        # dpi を持たない古い形式に書き換える
        with zipfile.ZipFile(path) as zf:
            entries = {n: zf.read(n) for n in zf.namelist()}
        meta = json.loads(entries["meta.json"])
        del meta["dpi"]
        entries["meta.json"] = json.dumps(meta).encode()
        with zipfile.ZipFile(path, "w") as zf:
            for n, data in entries.items():
                zf.writestr(n, data)
        win._load_pola(path)
        assert win.layer_stack.dpi == DEFAULT_DPI
        assert (win.layer_stack.width, win.layer_stack.height) == (64, 32)

    def test_broken_dpi_value_does_not_crash(self, win, tmp_path):
        _small_canvas(win, 20, 20)
        path = str(tmp_path / "bad.pola")
        win._write_pola(path)
        with zipfile.ZipFile(path) as zf:
            entries = {n: zf.read(n) for n in zf.namelist()}
        meta = json.loads(entries["meta.json"])
        meta["dpi"] = "abc"
        entries["meta.json"] = json.dumps(meta).encode()
        with zipfile.ZipFile(path, "w") as zf:
            for n, data in entries.items():
                zf.writestr(n, data)
        win._load_pola(path)
        assert win.layer_stack.dpi == DEFAULT_DPI

    def test_open_image_uses_its_size_and_dpi(self, win, tmp_path):
        img = QImage(321, 123, QImage.Format.Format_ARGB32)
        img.fill(QColor(255, 0, 0))
        dpm = round(150 / 0.0254)
        img.setDotsPerMeterX(dpm)
        img.setDotsPerMeterY(dpm)
        path = str(tmp_path / "scan.png")
        assert img.save(path)
        win._load_image_as_canvas(path)
        ls = win.layer_stack
        assert (ls.width, ls.height, ls.dpi) == (321, 123, 150)

    def test_image_without_dpi_opens_with_default(self, win, tmp_path):
        """dpi の記録が無い画像（スキャン・Web の画像など）は画面の 96dpi ではなく既定値で開く。"""
        import cv2
        import numpy as np
        for ext in ("png", "jpg", "bmp"):
            path = str(tmp_path / f"plain.{ext}")
            cv2.imwrite(path, np.zeros((12, 34, 3), np.uint8))
            win._load_image_as_canvas(path)
            ls = win.layer_stack
            assert (ls.width, ls.height, ls.dpi) == (34, 12, DEFAULT_DPI), ext

    def test_export_embeds_dpi(self, win, tmp_path, monkeypatch):
        _small_canvas(win, 40, 30)
        win.layer_stack.dpi = 350
        out = str(tmp_path / "out.png")
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            lambda *a, **k: (out, ""))
        win._export_png()
        assert _dpi_of(QImage(out)) == 350


# ── 前回のフォルダと既定のファイル名 ───────────────────────────────

class TestSaveFolderMemory:
    def test_export_defaults_to_pola_name_in_last_folder(self, win, tmp_path, monkeypatch):
        last = tmp_path / "illust"
        last.mkdir()
        other = tmp_path / "work"
        other.mkdir()
        win._settings = FakeSettings(last_save_dir=str(last))
        win._current_path = str(other / "ねこ.pola")
        seen = {}

        def fake_save(parent, title, start, filt):
            seen["start"] = start
            return ("", "")

        monkeypatch.setattr(QFileDialog, "getSaveFileName", fake_save)
        win._export_png()
        assert seen["start"] == os.path.join(str(last), "ねこ.png")

    def test_no_pola_uses_untitled(self, win, tmp_path):
        win._settings = FakeSettings(last_save_dir=str(tmp_path))
        win._current_path = None
        assert win._default_save_path(".pola") == os.path.join(str(tmp_path), "untitled.pola")

    def test_missing_folder_falls_back_to_pola_folder(self, win, tmp_path):
        win._settings = FakeSettings(last_save_dir=str(tmp_path / "消えたフォルダ"))
        win._current_path = str(tmp_path / "a.pola")
        assert win._default_save_path(".png") == os.path.join(str(tmp_path), "a.png")

    def test_saving_remembers_folder(self, win, tmp_path, monkeypatch):
        _small_canvas(win, 10, 10)
        dest = tmp_path / "save_here"
        dest.mkdir()
        target = str(dest / "x.pola")
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            lambda *a, **k: (target, ""))
        win._save_as_pola()
        assert os.path.exists(target)
        assert win._settings.d["last_save_dir"] == str(dest)
        # 次の書き出しは同じフォルダ・同じ名前から始まる
        assert win._default_save_path(".png") == os.path.join(str(dest), "x.png")

    def test_open_starts_in_last_open_folder(self, win, tmp_path, monkeypatch):
        win._settings = FakeSettings(last_open_dir=str(tmp_path))
        seen = {}

        def fake_open(parent, title, start, filt):
            seen["start"] = start
            return ("", "")

        monkeypatch.setattr(QFileDialog, "getOpenFileName", fake_open)
        win._open()
        assert seen["start"] == str(tmp_path)


# ── キャンバスサイズの変更 ─────────────────────────────────────────

class TestResizeCanvas:
    def test_dialog_paper_candidate_sets_size(self):
        d = M.ResizeCanvasDialog(100, 80, dpi=350)
        d._paper.setCurrentText("B5（JIS）")
        assert (d._w.value(), d._h.value()) == M.paper_size_px("B5（JIS）", 350)
        d._paper_orient.setCurrentIndex(1)
        assert (d._w.value(), d._h.value()) == M.paper_size_px("B5（JIS）", 350, True)

    def test_selection_button_only_with_selection(self):
        assert not M.ResizeCanvasDialog(100, 80).\
            _sel_btn.isEnabled()
        d = M.ResizeCanvasDialog(100, 80, selection_rect=QRect(10, 10, 20, 20))
        assert d._sel_btn.isEnabled()
        d._sel_btn.click()
        assert d.use_selection()
        assert d.result() == QDialog.DialogCode.Accepted

    def test_crop_to_selection_keeps_content_position(self, win):
        _small_canvas(win, 100, 80)
        bg = win.layer_stack.layers[0]
        bg.image.setPixelColor(30, 20, QColor(255, 0, 0))
        win.canvas._selection_rect = QRect(20, 10, 40, 30)
        win._crop_canvas_to_selection()
        ls = win.layer_stack
        assert (ls.width, ls.height) == (40, 30)
        assert ls.layers[0].image.pixelColor(10, 10) == QColor(255, 0, 0)
        assert win.canvas._selection_rect is None
        win.canvas.undo()
        assert (ls.width, ls.height) == (100, 80)
        assert ls.layers[0].image.pixelColor(30, 20) == QColor(255, 0, 0)

    def test_resize_during_transform_keeps_transformed_pixels(self, win):
        # 変形中にサイズを変えると変形が黙って捨てられていた
        _small_canvas(win, 100, 80)
        bg = win.layer_stack.layers[0]
        bg.image.fill(QColor(0, 0, 0, 0))
        bg.image.setPixelColor(2, 5, QColor(255, 0, 0))
        win.canvas._selection_rect = QRect(0, 0, 20, 20)
        assert win.canvas._lift_selection(bg)
        win.canvas.flip_transform_horizontal()
        win._apply_canvas_resize(120, 80, "crop", 0, 0)
        assert win.canvas._transform_image is None
        img = win.layer_stack.layers[0].image
        assert img.pixelColor(17, 5) == QColor(255, 0, 0)
        assert img.pixelColor(2, 5).alpha() == 0

    def test_crop_selection_clamped_to_canvas(self, win):
        _small_canvas(win, 100, 80)
        win.canvas._selection_rect = QRect(80, 60, 50, 50)
        win._crop_canvas_to_selection()
        assert (win.layer_stack.width, win.layer_stack.height) == (20, 20)

    def test_crop_without_selection_does_nothing(self, win, monkeypatch):
        from PyQt6.QtWidgets import QMessageBox
        monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
        _small_canvas(win, 100, 80)
        win._crop_canvas_to_selection()
        assert (win.layer_stack.width, win.layer_stack.height) == (100, 80)

    def test_resize_dialog_selection_path(self, win, monkeypatch):
        _small_canvas(win, 100, 80)
        win.canvas._selection_rect = QRect(0, 0, 50, 40)
        monkeypatch.setattr(M.ResizeCanvasDialog, "exec",
                            lambda self: QDialog.DialogCode.Accepted)
        monkeypatch.setattr(M.ResizeCanvasDialog, "use_selection", lambda self: True)
        win._resize_canvas()
        assert (win.layer_stack.width, win.layer_stack.height) == (50, 40)


# ── 解像度の変更 ──────────────────────────────────────────────────

class TestChangeDpi:
    def _run(self, win, monkeypatch, dpi, resample):
        def fake_exec(dlg):
            dlg._dpi.setCurrentText(str(dpi))
            dlg._resample.setChecked(resample)
            return QDialog.DialogCode.Accepted
        monkeypatch.setattr(M.DpiDialog, "exec", fake_exec)
        win._change_dpi()

    def test_change_dpi_only(self, win, monkeypatch):
        _small_canvas(win, 100, 80)
        self._run(win, monkeypatch, 600, False)
        ls = win.layer_stack
        assert (ls.width, ls.height, ls.dpi) == (100, 80, 600)
        assert win._status_dpi.text() == "600 dpi"
        win.canvas.undo()
        assert ls.dpi == DEFAULT_DPI
        assert win._status_dpi.text() == f"{DEFAULT_DPI} dpi"

    def test_keep_print_size_resamples(self, win, monkeypatch):
        _small_canvas(win, 100, 80)
        self._run(win, monkeypatch, 600, True)
        ls = win.layer_stack
        assert (ls.width, ls.height, ls.dpi) == (200, 160, 600)
        assert ls.layers[0].image.width() == 200
        win.canvas.undo()
        assert (ls.width, ls.height, ls.dpi) == (100, 80, DEFAULT_DPI)
        win.canvas.redo()
        assert (ls.width, ls.height, ls.dpi) == (200, 160, 600)
        assert win._status_dpi.text() == "600 dpi"

    def test_redo_dpi_only(self, win, monkeypatch):
        _small_canvas(win, 100, 80)
        self._run(win, monkeypatch, 72, False)
        win.canvas.undo()
        win.canvas.redo()
        assert win.layer_stack.dpi == 72

    def test_menu_items_exist(self, win):
        labels = set()
        for m in win.menuBar().actions():
            if m.menu():
                labels |= {a.text() for a in m.menu().actions()}
        assert "解像度（dpi）を変更..." in labels
        assert "選択範囲でキャンバスを切り抜く" in labels
