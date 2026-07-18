"""pytest 共通設定。QApplication をセッション単位で1つだけ生成する。"""
import sys
import pytest
from PyQt6.QtWidgets import QApplication, QMessageBox


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
