"""pytest 共通設定。QApplication をセッション単位で1つだけ生成する。"""
import sys
import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app
