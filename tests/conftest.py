"""pytest 共通設定。QApplication をセッション単位で1つだけ生成する。"""
import os
import sys

# テスト中はウィンドウを実画面に出さない（w.show() のたびに画面が
# チラつき、フォーカスも奪われるため）。QApplication 生成前、かつ
# PyQt のウィジェット系 import より前に設定する必要がある。
# 実画面で目視確認したいときは PAINTPOLA_TEST_GUI=1 を付けて実行する。
if not os.environ.get("PAINTPOLA_TEST_GUI"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile

import pytest
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QMessageBox

# テストがテーマ等を切り替えると、そのまま本物の設定（レジストリ）に
# 書き込まれ、次にアプリを起動したとき最後にテストが選んだテーマで
# 始まっていた。設定の保存先をテスト専用の一時フォルダに向ける。
_SETTINGS_DIR = tempfile.mkdtemp(prefix="paintpola_settings_")
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, _SETTINGS_DIR)


@pytest.fixture(autouse=True)
def _fresh_settings():
    """前のテストが保存した設定（手ブレ補正・入り抜き等）を持ち越さない。"""
    s = QSettings(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                  "PaintPola", "PaintPola")
    s.clear()
    s.sync()
    yield


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture(autouse=True)
def _no_blocking_dialogs(monkeypatch):
    """QMessageBox のモーダルダイアログがテスト実行中に表示され、
    手動でクリックするまで止まってしまうのを防ぐ。
    - question/exec 系: 「保存しない」相当（Discard/No）を自動選択
    - information/warning/critical: 何もせず自動で閉じる（OK相当）
    """
    monkeypatch.setattr(QMessageBox, "question",
                         lambda *a, **k: QMessageBox.StandardButton.Discard)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "exec", lambda self: QMessageBox.StandardButton.Discard)
