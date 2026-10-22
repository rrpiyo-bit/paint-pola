"""アプリ設定（QSettings）の入口。

QSettings("PaintPola", "PaintPola") は常にネイティブ形式（Windows では
レジストリ）を使い、テスト側から保存先を差し替えられない。そのせいで
テストが本物の設定を上書きし、選んだテーマが起動のたびに別のものに
なっていた。既定の形式に従う形で開き、テストでは一時フォルダの INI に
向けられるようにする（通常起動ではこれまでどおりネイティブ形式）。
"""
from PyQt6.QtCore import QSettings


def open_settings() -> QSettings:
    return QSettings(QSettings.defaultFormat(), QSettings.Scope.UserScope,
                     "PaintPola", "PaintPola")
