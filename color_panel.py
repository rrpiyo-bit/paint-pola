"""color_panel.py — カラーパネル

・HSV スライダー（Hue / Saturation / Value）
・最近使った色 (最大16色)
・カラーパレット（スウォッチ登録）
"""
from __future__ import annotations

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QSlider, QPushButton, QGridLayout, QFrame,
                              QSizePolicy, QColorDialog, QScrollArea, QComboBox)
from PyQt6.QtGui import QColor, QPainter, QLinearGradient, QBrush
from PyQt6.QtCore import Qt, QRect, pyqtSignal, QSize


class CollapsibleSection(QWidget):
    """タイトルクリックで内容を折りたたむセクション。"""
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._collapsed = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._toggle_btn = QPushButton(f"▼ {title}")
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 2px 4px; "
            "font-weight: bold; background: #e8e8e8; border: none; }"
            "QPushButton:hover { background: #d0d0d0; }")
        self._toggle_btn.setFixedHeight(22)
        self._toggle_btn.clicked.connect(self._toggle)
        outer.addWidget(self._toggle_btn)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(4, 2, 4, 4)
        self._body_layout.setSpacing(3)
        outer.addWidget(self._body)

    def add_widget(self, w: QWidget):
        self._body_layout.addWidget(w)

    def add_layout(self, layout):
        self._body_layout.addLayout(layout)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._body.setVisible(not self._collapsed)
        title = self._toggle_btn.text()[2:]  # "▼ " or "▶ " を除いたタイトル
        self._toggle_btn.setText(("▶ " if self._collapsed else "▼ ") + title)

_SWATCH_SIZE = 16
_PALETTE_COLS = 12

_DEFAULT_PALETTE: list[str] = [
    "#000000", "#ffffff", "#808080", "#c0c0c0",
    "#ff0000", "#ff8000", "#ffff00", "#00ff00",
    "#00ffff", "#0000ff", "#8000ff", "#ff00ff",
    "#800000", "#804000", "#808000", "#008000",
]

_PALETTES: dict[str, list[tuple[str, str]]] = {
    "基本色": [
        ("黒", "#000000"), ("白", "#ffffff"), ("灰", "#808080"), ("銀", "#c0c0c0"),
        ("赤", "#ff0000"), ("橙", "#ff8000"), ("黄", "#ffff00"), ("緑", "#00ff00"),
        ("水", "#00ffff"), ("青", "#0000ff"), ("紫", "#8000ff"), ("桃", "#ff00ff"),
        ("暗赤", "#800000"), ("茶", "#804000"), ("暗黄", "#808000"), ("暗緑", "#008000"),
    ],
    "日本の伝統色": [
        ("桜色", "#fef4f4"), ("薄桜", "#fdeff2"), ("桃色", "#f09199"), ("撫子色", "#e4007f"),
        ("紅梅色", "#e5004f"), ("躑躅色", "#e95295"), ("牡丹色", "#c1328e"), ("赤紫", "#eb6ea5"),
        ("紅色", "#c3272b"), ("朱色", "#eb6101"), ("茜色", "#b7282e"), ("臙脂色", "#9b003f"),
        ("深緋", "#c9171e"), ("柿色", "#ed6d3d"), ("黄丹", "#ee7800"), ("山吹色", "#f8b500"),
        ("鬱金色", "#fabf14"), ("黄蘗色", "#d9a62e"), ("刈安色", "#e8d77c"), ("菜の花色", "#ffec47"),
        ("卵色", "#fcd575"), ("蒲公英色", "#ffd900"), ("向日葵色", "#ffc20e"), ("金色", "#c9a825"),
        ("若草色", "#c3d825"), ("萌黄", "#aacf53"), ("草色", "#7b8d42"), ("苔色", "#69821b"),
        ("松葉色", "#3d6117"), ("千歳緑", "#316745"), ("深緑", "#004d25"), ("常磐色", "#007b43"),
        ("緑青", "#47885e"), ("若竹色", "#68be8d"), ("浅葱色", "#00a3af"), ("水浅葱", "#80aba9"),
        ("空色", "#87ceeb"), ("水色", "#bce2e8"), ("瑠璃色", "#1e50a2"), ("群青色", "#4c6cb3"),
        ("紺色", "#223a70"), ("藍色", "#165e83"), ("藍鉄色", "#393f4c"), ("紺青", "#192f60"),
        ("紫", "#7a4171"), ("藤色", "#bbb6d0"), ("菫色", "#7058a3"), ("江戸紫", "#745399"),
        ("京紫", "#772f6d"), ("薄紫", "#c0a2c7"), ("鳩羽色", "#95859c"), ("滅紫", "#594255"),
        ("利休鼠", "#888e7e"), ("鶯色", "#928c36"), ("狐色", "#c38743"), ("栗色", "#762f07"),
        ("黄土色", "#c49a6a"), ("飴色", "#deb068"), ("煉瓦色", "#b55233"), ("小豆色", "#96514d"),
        ("栗皮色", "#6c3524"), ("胡桃色", "#a86f4c"), ("琥珀色", "#bf783a"), ("丁子色", "#efcd9a"),
        ("鉛色", "#7b7c7d"), ("墨色", "#343434"), ("漆黒", "#0d0015"), ("生成り", "#fbfaf5"),
    ],
    "フランスの伝統色": [
        ("ブラン", "#ffffff"), ("ノワール", "#000000"), ("グリ", "#888888"),
        ("ルージュ", "#e60033"), ("ヴェルミヨン", "#e83929"), ("カーマイン", "#be0032"),
        ("フランボワーズ", "#c73e67"), ("ローズ", "#f19ca7"), ("コラーユ", "#ef8468"),
        ("サーモン", "#f3a68c"), ("アブリコ", "#f7a535"), ("オランジュ", "#ee7800"),
        ("マンダリン", "#f09629"), ("アンブル", "#c2894b"), ("キャラメル", "#bc763c"),
        ("カフェオレ", "#946c45"), ("ショコラ", "#462f21"), ("ノワゼット", "#8f6d3f"),
        ("シャンパーニュ", "#e8d3a9"), ("クレーム", "#e8d3c7"), ("ヴァニーユ", "#e8c59c"),
        ("ジョーヌ", "#ffd900"), ("シトロン", "#e8d44d"), ("マスタード", "#c4a317"),
        ("オリーブ", "#6b6f36"), ("セラドン", "#8db255"), ("ヴェール", "#009944"),
        ("エメロード", "#00a968"), ("ティヤール", "#00a497"), ("ターコワーズ", "#00afcc"),
        ("シエル", "#89c3eb"), ("アジュール", "#0075c2"), ("ブルー", "#0068b7"),
        ("マリーヌ", "#1b2a6b"), ("ロワヤル", "#1b3982"), ("ウルトラマリン", "#3d50b6"),
        ("アンディゴ", "#264882"), ("ラヴァンド", "#a096c1"), ("モーヴ", "#915da3"),
        ("ヴィオレ", "#7b1fa2"), ("プリュンヌ", "#5a0058"), ("ボルドー", "#6c2735"),
        ("ブルゴーニュ", "#6b2242"), ("リラ", "#c9a8cb"), ("フューシャ", "#d94177"),
    ],
    "中国の伝統色": [
        ("大紅", "#dc3023"), ("朱砂", "#be3f35"), ("胭脂", "#c63c5c"), ("桃紅", "#f0849a"),
        ("石榴紅", "#c94043"), ("棗紅", "#8a2b2b"), ("玫瑰紅", "#d4395e"), ("薔薇紅", "#e8638a"),
        ("橘紅", "#f07c3a"), ("琥珀", "#c48a3f"), ("杏黄", "#f0a846"), ("金黄", "#e8b830"),
        ("明黄", "#f5d128"), ("鵝黄", "#f0d695"), ("秋香", "#b6a254"), ("草綠", "#7f9b3f"),
        ("松綠", "#40725e"), ("竹青", "#6ba08a"), ("翠綠", "#15835c"), ("碧", "#468966"),
        ("石青", "#2f6e99"), ("霽青", "#204d6d"), ("群青", "#2e4c8e"), ("藏青", "#1c305c"),
        ("靛青", "#177cb0"), ("天青", "#68b0d8"), ("品藍", "#3346a0"), ("紫色", "#6a2c70"),
        ("紫檀", "#4c1c20"), ("紫棠", "#5d3142"), ("丁香", "#cba3c8"), ("雪白", "#f0fcff"),
        ("月白", "#d6ecf0"), ("水墨", "#5b5d5e"), ("玄色", "#1c1c1c"), ("墨色", "#333333"),
    ],
    "パステル": [
        ("ベビーピンク", "#ffb6c1"), ("ミスティローズ", "#ffe4e1"), ("ピーチパフ", "#ffdab9"),
        ("ラベンダーブラッシュ", "#fff0f5"), ("シアンライト", "#e0ffff"), ("アリスブルー", "#f0f8ff"),
        ("ラベンダー", "#e6e6fa"), ("ティッスルー", "#d8bfd8"), ("プラム", "#dda0dd"),
        ("ライトピンク", "#ffb3de"), ("ライトサーモン", "#ffa07a"), ("ライトコーラル", "#f08080"),
        ("ライトゴールド", "#fafad2"), ("レモンシフォン", "#fffacd"), ("パパイヤウィップ", "#ffefd5"),
        ("ハニーデュー", "#f0fff0"), ("ミントクリーム", "#f5fffa"), ("ライトグリーン", "#90ee90"),
        ("パウダーブルー", "#b0e0e6"), ("ライトブルー", "#add8e6"), ("ライトスカイ", "#87cefa"),
        ("ペリウィンクル", "#ccccff"), ("ライトスレート", "#b4a7d6"), ("モーヴ", "#e0b0ff"),
    ],
    "肌色・人物": [
        ("ペールピンク", "#fdeef4"), ("明るい肌", "#ffe0bd"), ("肌色", "#f5c9a6"),
        ("ピーチ", "#f8c89f"), ("アプリコット", "#f4a460"), ("タン", "#d2b48c"),
        ("サンド", "#c2b280"), ("シエナ", "#a0522d"), ("バーント", "#8b4513"),
        ("チェスナット", "#954535"), ("セピア", "#704214"), ("ウォルナット", "#5b3a29"),
        ("ココア", "#462f21"), ("エボニー", "#2b1d0e"), ("唇ピンク", "#e8909a"),
        ("唇レッド", "#c63f47"), ("チーク", "#f4a7b9"), ("アイシャドウ", "#b4a7d6"),
    ],
    "自然・風景": [
        ("空色", "#87ceeb"), ("青空", "#4a90d9"), ("夕焼け", "#f07f5e"), ("朝焼け", "#f5b199"),
        ("茜空", "#c94058"), ("雲色", "#e6e2d3"), ("霞色", "#c4bcb0"), ("夜空", "#0f1a3c"),
        ("星空", "#192236"), ("月光", "#d6d2c4"), ("若葉", "#a9d159"), ("深森", "#1e4d2b"),
        ("木漏れ日", "#e8d77c"), ("枯葉", "#c48d3f"), ("紅葉", "#c83c2d"), ("雪原", "#f0f3f4"),
        ("砂浜", "#eddcb1"), ("海原", "#005f8c"), ("珊瑚礁", "#00a6b6"), ("夕凪", "#8e7cc3"),
        ("花畑", "#e88abf"), ("桜並木", "#f9c2c8"), ("稲穂", "#d4a017"), ("石畳", "#8e8e8e"),
    ],
    "レトロ・ポップ・ネオン": [
        # ── レトロ ──
        ("クリーム", "#f5e6ca"), ("マスタード", "#c4a317"), ("テラコッタ", "#c75b39"),
        ("バーガンディ", "#6c2735"), ("オリーブ", "#6b6f36"), ("レトロ青", "#4a6670"),
        ("セピア", "#704214"), ("ダスティローズ", "#c08081"), ("アンティーク", "#d4a76a"),
        ("ヴィンテージ緑", "#5e7e5e"), ("レトロ橙", "#d4783a"), ("モカ", "#8b6f4e"),
        ("---", None),
        # ── ポップ ──
        ("ポップ赤", "#ff1744"), ("ポップ青", "#2979ff"), ("ポップ黄", "#ffea00"),
        ("ポップ緑", "#00e676"), ("ポップ桃", "#ff4081"), ("ポップ橙", "#ff9100"),
        ("ポップ紫", "#d500f9"), ("ポップ水", "#00e5ff"), ("ライム", "#c6ff00"),
        ("ポップ珊瑚", "#ff6e40"), ("ホットピンク", "#ff1493"), ("エレクトリック", "#7c4dff"),
        ("---", None),
        # ── ネオン ──
        ("ネオンピンク", "#ff00ff"), ("ネオングリーン", "#39ff14"), ("ネオンブルー", "#00f0ff"),
        ("ネオンイエロー", "#dfff00"), ("ネオンオレンジ", "#ff5f1f"), ("ネオンパープル", "#bc13fe"),
        ("ネオンレッド", "#ff073a"), ("ネオンシアン", "#00ffef"), ("ネオンライム", "#ccff00"),
        ("ネオンマゼンタ", "#ff00af"), ("ネオンアクア", "#00ffbf"), ("ネオンバイオレット", "#8f00ff"),
        ("---", None),
        # ── パステル ──
        ("Pベビーピンク", "#ffc1cc"), ("Pピーチ", "#ffdfba"), ("Pレモン", "#ffffba"),
        ("Pミント", "#baffc9"), ("Pスカイ", "#bae1ff"), ("Pラベンダー", "#e8baff"),
        ("Pローズ", "#ffd1dc"), ("Pアプリコット", "#ffe5b4"), ("Pバター", "#fff9c4"),
        ("Pシーフォーム", "#a8e6cf"), ("Pパウダー", "#bcd4e6"), ("Pライラック", "#d4a5e5"),
    ],
    "100 Brilliant Color": [
        ("", "#F98866"), ("", "#FF420E"), ("", "#80BD9E"), ("", "#89DA59"), ("---", None),
        ("", "#90afc5"), ("", "#336b87"), ("", "#2a3132"), ("", "#763626"), ("---", None),
        ("", "#46211a"), ("", "#693d3d"), ("", "#ba5536"), ("", "#a43820"), ("---", None),
        ("", "#505160"), ("", "#68828e"), ("", "#aebd38"), ("", "#598234"), ("---", None),
        ("", "#003b46"), ("", "#07575b"), ("", "#66a5ad"), ("", "#c4dfe6"), ("---", None),
        ("", "#2e4600"), ("", "#486b00"), ("", "#a2c523"), ("", "#7d4427"), ("---", None),
        ("", "#021c1e"), ("", "#004445"), ("", "#2c7873"), ("", "#6fb98f"), ("---", None),
        ("", "#375e97"), ("", "#fb6542"), ("", "#ffbb00"), ("", "#3f681c"), ("---", None),
        ("", "#98dbc6"), ("", "#5bc8ac"), ("", "#e6d72a"), ("", "#f18d9e"), ("---", None),
        ("", "#324851"), ("", "#86ac41"), ("", "#34675c"), ("", "#7da3a1"), ("---", None),
        ("", "#4cb5f5"), ("", "#b7b8b6"), ("", "#34675c"), ("", "#b3c100"), ("---", None),
        ("", "#f4cc70"), ("", "#de7a22"), ("", "#20948b"), ("", "#61b187"), ("---", None),
        ("", "#8d230f"), ("", "#1e434c"), ("", "#9b4f0f"), ("", "#c99e10"), ("---", None),
        ("", "#f1f1f2"), ("", "#bcbabe"), ("", "#a1d6e2"), ("", "#1995ad"), ("---", None),
        ("", "#9a9eab"), ("", "#5d535e"), ("", "#ec96a4"), ("", "#dfe166"), ("---", None),
        ("", "#011a27"), ("", "#063852"), ("", "#f0810f"), ("", "#e6df44"), ("---", None),
        ("", "#75b1a9"), ("", "#d9b44a"), ("", "#4f6457"), ("", "#acd0c0"), ("---", None),
        ("", "#eb8a44"), ("", "#f9dc24"), ("", "#4b7447"), ("", "#8eba43"), ("---", None),
        ("", "#363237"), ("", "#2d4262"), ("", "#73605b"), ("", "#d09683"), ("---", None),
        ("", "#f52549"), ("", "#fa6775"), ("", "#ffd64d"), ("", "#98c01c"), ("---", None),
        ("", "#2e2300"), ("", "#6e6702"), ("", "#c05805"), ("", "#db9501"), ("---", None),
        ("", "#50312f"), ("", "#cb0000"), ("", "#e4ea8c"), ("", "#3f6c45"), ("---", None),
        ("", "#34888c"), ("", "#7caa2d"), ("", "#f5e356"), ("", "#cb6318"), ("---", None),
        ("", "#0f1b07"), ("", "#ffffff"), ("", "#5c821a"), ("", "#c6d166"), ("---", None),
        ("", "#00293c"), ("", "#1e656d"), ("", "#f1f3ce"), ("", "#f62a00"), ("---", None),
        ("", "#626d71"), ("", "#cdcdc0"), ("", "#ddbc95"), ("", "#b38867"), ("---", None),
        ("", "#258039"), ("", "#f5be41"), ("", "#31a9b8"), ("", "#cf3721"), ("---", None),
        ("", "#ee693f"), ("", "#f69454"), ("", "#fcfdfe"), ("", "#739f3d"), ("---", None),
        ("", "#b9d9c3"), ("", "#752a07"), ("", "#fbcb7b"), ("", "#eb5e30"), ("---", None),
        ("", "#1e1f26"), ("", "#283655"), ("", "#4d648d"), ("", "#d0e1f9"), ("---", None),
        ("", "#f70025"), ("", "#f7efe2"), ("", "#f25c00"), ("", "#f9a603"), ("---", None),
        ("", "#a1be95"), ("", "#e2dfa2"), ("", "#92aac7"), ("", "#ed5752"), ("---", None),
        ("", "#4897d8"), ("", "#ffdb5c"), ("", "#fa6e59"), ("", "#f8a055"), ("---", None),
        ("", "#af4425"), ("", "#662e1c"), ("", "#ebdcb2"), ("", "#c9a66b"), ("---", None),
        ("", "#c1e1dc"), ("", "#ffccac"), ("", "#ffeb94"), ("", "#fdd475"), ("---", None),
        ("", "#4c3f54"), ("", "#d13525"), ("", "#f2c057"), ("", "#486824"), ("---", None),
        ("", "#faaf08"), ("", "#fa812f"), ("", "#fa4032"), ("", "#fef3e2"), ("---", None),
        ("", "#f4ec6a"), ("", "#bbcf4a"), ("", "#e73f0b"), ("", "#a11f0c"), ("---", None),
        ("", "#fef2e4"), ("", "#fd974f"), ("", "#c60000"), ("", "#805a3b"), ("---", None),
        ("", "#f77604"), ("", "#b8d20b"), ("", "#f56c57"), ("", "#231b12"), ("---", None),
        ("", "#7f152e"), ("", "#d61800"), ("", "#edae01"), ("", "#e94f08"), ("---", None),
        ("", "#eae2d6"), ("", "#d5c3aa"), ("", "#867666"), ("", "#e1b80d"), ("---", None),
        ("", "#265c00"), ("", "#68a225"), ("", "#b3de81"), ("", "#fdffff"), ("---", None),
        ("", "#a10115"), ("", "#d72c16"), ("", "#f0efea"), ("", "#c0b2b5"), ("---", None),
        ("", "#c7db00"), ("", "#7aa802"), ("", "#e78b2d"), ("", "#e4b600"), ("---", None),
        ("", "#301b28"), ("", "#523634"), ("", "#b6452c"), ("", "#ddc5a2"), ("---", None),
        ("", "#ebdf00"), ("", "#7e7b15"), ("", "#563e20"), ("", "#b38540"), ("---", None),
        ("", "#662225"), ("", "#b51d0a"), ("", "#ead39c"), ("", "#7d5e3c"), ("---", None),
        ("", "#4b4345"), ("", "#556dac"), ("", "#f79b77"), ("", "#755248"), ("---", None),
        ("", "#d8412f"), ("", "#fe7a47"), ("", "#fcfdfe"), ("", "#f5ca99"), ("---", None),
        ("", "#2988bc"), ("", "#2f496e"), ("", "#f4eade"), ("", "#ed8c72"), ("---", None),
        ("", "#000b29"), ("", "#d70026"), ("", "#f8f5f2"), ("", "#edb83d"), ("---", None),
        ("", "#1e0000"), ("", "#500805"), ("", "#9d331f"), ("", "#bc6d4f"), ("---", None),
        ("", "#f9ba32"), ("", "#426e86"), ("", "#f8f1e5"), ("", "#2f3131"), ("---", None),
        ("", "#04202c"), ("", "#304040"), ("", "#5b7065"), ("", "#c9d1c8"), ("---", None),
        ("", "#d24136"), ("", "#eb8a3e"), ("", "#ebb582"), ("", "#785a46"), ("---", None),
        ("", "#217ca3"), ("", "#e29930"), ("", "#32384d"), ("", "#211f30"), ("---", None),
        ("", "#003d47"), ("", "#128277"), ("", "#52958b"), ("", "#b9c4c9"), ("---", None),
        ("", "#506d2f"), ("", "#2a2922"), ("", "#f3ebdd"), ("", "#7d5642"), ("---", None),
        ("", "#f47d4a"), ("", "#e1315b"), ("", "#ffec5c"), ("", "#008dcb"), ("---", None),
        ("", "#a4cabc"), ("", "#eab364"), ("", "#b2473e"), ("", "#acbd78"), ("---", None),
        ("", "#16253d"), ("", "#002c54"), ("", "#efb509"), ("", "#cd7213"), ("---", None),
        ("", "#8593ae"), ("", "#5a4e4d"), ("", "#7e675e"), ("", "#dda288"), ("---", None),
        ("", "#2b616d"), ("", "#b2dbd5"), ("", "#ffffff"), ("", "#fa8d62"), ("---", None),
        ("", "#00cffa"), ("", "#ff0038"), ("", "#ffce38"), ("", "#020509"), ("---", None),
        ("", "#a5c3cf"), ("", "#f3d3b8"), ("", "#e59d5c"), ("", "#a99f3c"), ("---", None),
        ("", "#257985"), ("", "#5ea8a7"), ("", "#ffffff"), ("", "#ff4447"), ("---", None),
        ("", "#fcc875"), ("", "#baa896"), ("", "#e6ccb5"), ("", "#e38b75"), ("---", None),
        ("", "#335252"), ("", "#d4dde1"), ("", "#aa4b41"), ("", "#2d3033"), ("---", None),
        ("", "#ffccbb"), ("", "#6eb5c0"), ("", "#006c84"), ("", "#e2e8e4"), ("---", None),
        ("", "#8c0004"), ("", "#c8000a"), ("", "#e8a835"), ("", "#e2c499"), ("---", None),
        ("", "#2c4a52"), ("", "#537072"), ("", "#8e9b97"), ("", "#f4ebdb"), ("---", None),
        ("", "#c5001a"), ("", "#e4e3db"), ("", "#113743"), ("", "#c5beba"), ("---", None),
        ("", "#d35c37"), ("", "#bf9a77"), ("", "#d6c6b9"), ("", "#97b8c2"), ("---", None),
        ("", "#919636"), ("", "#524a3a"), ("", "#fffae1"), ("", "#5a5f37"), ("---", None),
        ("", "#52908b"), ("", "#e5e2ca"), ("", "#ddbc95"), ("", "#e7472e"), ("---", None),
        ("", "#2f2e33"), ("", "#d5d6d2"), ("", "#ffffff"), ("", "#3a5199"), ("---", None),
        ("", "#756867"), ("", "#d5d6d2"), ("", "#353c3f"), ("", "#ff8d3f"), ("---", None),
        ("", "#31a2ac"), ("", "#af1c1c"), ("", "#f0eff0"), ("", "#2f2f28"), ("---", None),
        ("", "#6c5f5b"), ("", "#cdab81"), ("", "#dac3b3"), ("", "#4f4a45"), ("---", None),
        ("", "#444c5c"), ("", "#ce5a57"), ("", "#78a5a3"), ("", "#e1b16a"), ("---", None),
        ("", "#20232a"), ("", "#acbebe"), ("", "#f4f4ef"), ("", "#a01d26"), ("---", None),
        ("", "#d55448"), ("", "#ffa577"), ("", "#f9f9ff"), ("", "#896e69"), ("---", None),
        ("", "#344d90"), ("", "#5cc5ef"), ("", "#ffb745"), ("", "#e7552c"), ("---", None),
        ("", "#080706"), ("", "#efefef"), ("", "#d1b280"), ("", "#594d46"), ("---", None),
        ("", "#5f968e"), ("", "#bfdccf"), ("", "#e05858"), ("", "#d5c9b1"), ("---", None),
        ("", "#962715"), ("", "#ffffff"), ("", "#1e1e20"), ("", "#bbc3c6"), ("---", None),
        ("", "#688b8a"), ("", "#a0b084"), ("", "#faefd4"), ("", "#a57c65"), ("---", None),
        ("", "#262f34"), ("", "#f34a4a"), ("", "#f1d3bc"), ("", "#615049"), ("---", None),
        ("", "#882426"), ("", "#cdbea7"), ("", "#323030"), ("", "#c29545"), ("---", None),
        ("", "#42313a"), ("", "#6c2d2c"), ("", "#9f4636"), ("", "#f1dcc9"), ("---", None),
        ("", "#fbcd4b"), ("", "#a3a599"), ("", "#282623"), ("", "#88a550"), ("---", None),
        ("", "#ffbebd"), ("", "#fcfcfa"), ("", "#337bae"), ("", "#1a405f"), ("---", None),
        ("", "#0f1f38"), ("", "#8e7970"), ("", "#f55449"), ("", "#1b4b5a"), ("---", None),
        ("", "#81715e"), ("", "#faae3d"), ("", "#e38533"), ("", "#e4535e"), ("---", None),
        ("", "#061283"), ("", "#fd3c3c"), ("", "#ffb74c"), ("", "#138d90"), ("---", None),
        ("", "#dddede"), ("", "#232122"), ("", "#a5c05b"), ("", "#7ba4a8"), ("---", None),
        ("", "#b3dbc0"), ("", "#fe0000"), ("", "#fdf6f6"), ("", "#67baca"), ("---", None),
        ("", "#a49592"), ("", "#727077"), ("", "#eed8c9"), ("", "#e99787"), ("---", None),
        ("", "#488a99"), ("", "#dbae58"), ("", "#fbe9e7"), ("", "#b4b4b4"),
    ],
}



class MiniSwatch(QFrame):
    """クリック可能な小さなカラースウォッチ。"""
    clicked = pyqtSignal(QColor)
    right_clicked = pyqtSignal(QColor)

    def __init__(self, color: QColor = QColor("black"), size: int = _SWATCH_SIZE, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(size, size)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self._refresh()

    def set_color(self, color: QColor):
        self._color = color
        self._refresh()

    def color(self) -> QColor:
        return self._color

    def _refresh(self):
        self.setStyleSheet(
            f"background-color: rgba({self._color.red()},{self._color.green()},"
            f"{self._color.blue()},{self._color.alpha()});"
            f"border: none;")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._color)
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(self._color)


class HsvSlider(QWidget):
    """H / S / V それぞれのスライダー。色が変わると color_changed を emit。
    ドラッグ操作が確定した（指を離した）時には color_committed も emit する
    （履歴への追加はこちらだけを使い、ドラッグ中の中間値で履歴を汚さないようにする）。"""
    color_changed = pyqtSignal(QColor)
    color_committed = pyqtSignal(QColor)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._h = self._make_slider(0, 359, "H")
        self._s = self._make_slider(0, 255, "S")
        self._v = self._make_slider(0, 255, "V")
        self._a = self._make_slider(0, 255, "A")

        for label, sl in [("H", self._h), ("S", self._s),
                           ("V", self._v), ("A", self._a)]:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(sl)
            layout.addLayout(row)

        self._block = False
        self._h.valueChanged.connect(self._emit)
        self._s.valueChanged.connect(self._emit)
        self._v.valueChanged.connect(self._emit)
        self._a.valueChanged.connect(self._emit)
        self._h.sliderReleased.connect(self._emit_committed)
        self._s.sliderReleased.connect(self._emit_committed)
        self._v.sliderReleased.connect(self._emit_committed)
        self._a.sliderReleased.connect(self._emit_committed)

        # 初期値: 黒・不透明
        self._h.setValue(0)
        self._s.setValue(0)
        self._v.setValue(0)
        self._a.setValue(255)

    def _make_slider(self, lo: int, hi: int, name: str) -> QSlider:
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(lo, hi)
        sl.setFixedHeight(18)
        return sl

    def _current(self) -> QColor:
        return QColor.fromHsv(self._h.value(), self._s.value(),
                               self._v.value(), self._a.value())

    def _emit(self):
        if self._block:
            return
        self.color_changed.emit(self._current())

    def _emit_committed(self):
        if self._block:
            return
        self.color_committed.emit(self._current())

    def set_color(self, color: QColor):
        self._block = True
        h, s, v, a = color.hsvHue(), color.hsvSaturation(), color.value(), color.alpha()
        self._h.setValue(max(0, h))
        self._s.setValue(s)
        self._v.setValue(v)
        self._a.setValue(a)
        self._block = False


class ColorPanel(QWidget):
    """カラーパネル全体。タイトルバーで全体を折りたたみ可能。"""
    color_changed = pyqtSignal(QColor)
    color_committed = pyqtSignal(QColor)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260)
        self._current = QColor("black")
        self._collapsed = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ヘッダー（折りたたみボタン）
        self._header = QPushButton("▼ カラー")
        self._header.setFlat(True)
        self._header.setStyleSheet(
            "QPushButton { text-align: left; padding: 3px 6px; "
            "font-weight: bold; background: #d0d8e8; border: none; border-bottom: 1px solid #aaa; }"
            "QPushButton:hover { background: #b8c8e0; }")
        self._header.setFixedHeight(24)
        self._header.clicked.connect(self._toggle)
        outer.addWidget(self._header)

        # 本体（折りたたみ対象）
        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(4, 4, 4, 4)
        body_layout.setSpacing(4)

        # 現在色プレビュー（クリックでダイアログ）
        self._preview = MiniSwatch(self._current, size=36)
        self._preview.setFixedSize(248, 26)
        self._preview.setToolTip("クリックでカラーダイアログ")
        self._preview.clicked.connect(self._pick_from_dialog)
        body_layout.addWidget(self._preview)

        # HSV スライダー（折りたたみセクション）
        self._sec_hsv = CollapsibleSection("HSV スライダー")
        self._hsv = HsvSlider()
        self._hsv.color_changed.connect(self._on_hsv_change)
        self._hsv.color_committed.connect(self._on_hsv_committed)
        self._sec_hsv.add_widget(self._hsv)
        body_layout.addWidget(self._sec_hsv)

        # パレット（折りたたみセクション）
        self._sec_palette = CollapsibleSection("パレット")

        # パレット種類プルダウン
        self._palette_combo = QComboBox()
        for name in _PALETTES:
            self._palette_combo.addItem(name)
        self._palette_combo.setFixedHeight(24)
        self._palette_combo.setStyleSheet("font-size:11px;")
        self._palette_combo.currentTextChanged.connect(self._on_palette_change)
        self._sec_palette.add_widget(self._palette_combo)

        add_btn = QPushButton("現在色を登録")
        add_btn.setFixedHeight(22)
        add_btn.clicked.connect(self._register_to_palette)
        self._sec_palette.add_widget(add_btn)

        self._palette_scroll = QScrollArea()
        self._palette_scroll.setWidgetResizable(True)
        self._palette_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._palette_scroll.setFixedHeight(100)
        self._palette_scroll.setStyleSheet("QScrollArea { border: none; }")
        self._palette_container = QWidget()
        self._palette_grid = QGridLayout(self._palette_container)
        self._palette_grid.setSpacing(0)
        self._palette_grid.setContentsMargins(0, 0, 0, 0)
        self._palette_scroll.setWidget(self._palette_container)
        self._palette_swatches: list[MiniSwatch] = []
        self._sec_palette.add_widget(self._palette_scroll)
        self._load_palette("基本色")
        body_layout.addWidget(self._sec_palette)

        outer.addWidget(self._body)

    def _toggle(self):
        self._collapsed = not self._collapsed
        self._body.setVisible(not self._collapsed)
        self._header.setText("▶ カラー" if self._collapsed else "▼ カラー")

    # ── public ───────────────────────────────────────────────────────────────

    def set_color(self, color: QColor):
        """外部から現在色をセットする（スポイト等）。"""
        self._current = color
        self._block_hsv = True
        self._hsv.set_color(color)
        self._block_hsv = False
        self._preview.set_color(color)

    def current_color(self) -> QColor:
        return self._current

    # ── internal ─────────────────────────────────────────────────────────────

    def _on_hsv_change(self, color: QColor):
        self._current = color
        self._preview.set_color(color)
        self.color_changed.emit(color)

    def _on_hsv_committed(self, color: QColor):
        self.color_committed.emit(color)

    def _on_swatch_click(self, color: QColor):
        self.set_color(color)
        self.color_changed.emit(color)

    def _on_palette_right_click(self, color: QColor):
        """右クリックで現在色をそのスウォッチに上書き登録。"""
        sender = self.sender()
        if isinstance(sender, MiniSwatch):
            sender.set_color(self._current)

    def _pick_from_dialog(self, _=None):
        c = QColorDialog.getColor(
            self._current, self, "色を選択",
            QColorDialog.ColorDialogOption.ShowAlphaChannel)
        if c.isValid():
            self.set_color(c)
            self.color_changed.emit(c)

    def _on_palette_change(self, name: str):
        self._load_palette(name)

    def _load_palette(self, name: str):
        for sw in self._palette_swatches:
            self._palette_grid.removeWidget(sw)
            sw.deleteLater()
        self._palette_swatches.clear()
        # remove spacer items
        while self._palette_grid.count():
            item = self._palette_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        entries = _PALETTES.get(name, _PALETTES["基本色"])
        has_sep = any(h is None for _, h in entries)
        if name == "100 Brilliant Color":
            cols = 5
        elif has_sep:
            cols = _PALETTE_COLS
        else:
            cols = _PALETTE_COLS
        row, col = 0, 0
        for label, hex_color in entries:
            if hex_color is None:
                if cols == 5:
                    sw = MiniSwatch(QColor(0, 0, 0, 0), _SWATCH_SIZE)
                    sw.setStyleSheet("background: transparent; border: none;")
                    sw.setEnabled(False)
                    self._palette_grid.addWidget(sw, row, col)
                    self._palette_swatches.append(sw)
                    col += 1
                    if col >= cols:
                        col = 0
                        row += 1
                else:
                    if col > 0:
                        row += 1
                        col = 0
                continue
            sw = MiniSwatch(QColor(hex_color))
            tip = f"{label} ({hex_color})" if label else hex_color
            sw.setToolTip(tip)
            sw.clicked.connect(self._on_swatch_click)
            sw.right_clicked.connect(self._on_palette_right_click)
            self._palette_grid.addWidget(sw, row, col)
            self._palette_swatches.append(sw)
            col += 1
            if col >= cols:
                col = 0
                row += 1

    def _register_to_palette(self):
        """現在色を最初の空白スウォッチに登録（なければ末尾に追加）。"""
        for sw in self._palette_swatches:
            if sw.color().name() == "#ffffff":
                sw.set_color(self._current)
                return
        sw = MiniSwatch(self._current)
        sw.setToolTip(f"カスタム ({self._current.name()})")
        sw.clicked.connect(self._on_swatch_click)
        sw.right_clicked.connect(self._on_palette_right_click)
        idx = len(self._palette_swatches)
        self._palette_grid.addWidget(sw, idx // _PALETTE_COLS, idx % _PALETTE_COLS)
        self._palette_swatches.append(sw)
