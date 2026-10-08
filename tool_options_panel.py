from __future__ import annotations

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                              QSpinBox, QSlider, QComboBox, QFrame,
                              QSizePolicy, QScrollArea, QAbstractSpinBox,
                              QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal

from tools import Tool
from brush import BRUSH_LABELS

# ブラシサイズのプリセット（クリックでその太さにする）
PEN_SIZE_PRESETS = (3, 5, 10, 20, 50)
ERASER_SIZE_PRESETS = (10, 20, 40, 80, 150)


class _SliderSpin(QWidget):
    """スライダーと数値入力を横に並べた複合ウィジェット。
    ドラッグで大まかに、スピンボックスで正確に調整できる。"""

    def __init__(self, value: int, lo: int, hi: int, suffix: str = "", parent=None):
        super().__init__(parent)
        col = QVBoxLayout(self)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(lo, hi)
        self._slider.setValue(value)
        self._spin = QSpinBox()
        self._spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.PlusMinus)
        self._spin.setRange(lo, hi)
        self._spin.setValue(value)
        if suffix:
            self._spin.setSuffix(suffix)
        self._spin.setMinimumWidth(90)
        col.addWidget(self._slider)
        spin_row = QHBoxLayout()
        spin_row.setContentsMargins(0, 0, 0, 0)
        spin_row.addStretch()
        spin_row.addWidget(self._spin)
        col.addLayout(spin_row)
        self._callbacks = []
        self._slider.valueChanged.connect(self._on_slider)
        self._spin.valueChanged.connect(self._on_spin)

    def _on_slider(self, v: int):
        if self._spin.value() != v:
            self._spin.blockSignals(True)
            self._spin.setValue(v)
            self._spin.blockSignals(False)
        self._emit(v)

    def _on_spin(self, v: int):
        if self._slider.value() != v:
            self._slider.blockSignals(True)
            self._slider.setValue(v)
            self._slider.blockSignals(False)
        self._emit(v)

    def _emit(self, v: int):
        if not self.signalsBlocked():
            for cb in self._callbacks:
                cb(v)

    def connect_changed(self, callback):
        self._callbacks.append(callback)

    def setValue(self, v: int):
        """外部からの同期用。コールバックは発火しない。"""
        blocked = self.blockSignals(True)
        self._slider.blockSignals(True)
        self._spin.blockSignals(True)
        self._slider.setValue(v)
        self._spin.setValue(v)
        self._slider.blockSignals(False)
        self._spin.blockSignals(False)
        self.blockSignals(blocked)

    def value(self) -> int:
        return self._spin.value()


class ToolOptionsPanel(QWidget):
    """ツールごとの詳細設定パネル。ツール切替で内容が変わる。"""

    # 各設定の変更シグナル
    pen_size_changed      = pyqtSignal(int)
    eraser_size_changed   = pyqtSignal(int)
    pen_opacity_changed   = pyqtSignal(int)   # ペンの不透明度 (1〜100 %)
    pen_transparent_toggled = pyqtSignal(bool)  # 透明色で描く
    eraser_soft_toggled   = pyqtSignal(bool)  # ふちをぼかして消す
    fill_tolerance_changed = pyqtSignal(int)  # バケツ塗りの色の誤差 (%)
    text_font_changed     = pyqtSignal(str)   # テキストのフォント名
    text_size_changed     = pyqtSignal(int)   # テキストの文字サイズ (px)
    stabilization_changed = pyqtSignal(int)   # 手ブレ補正の強さ（0=なし）
    taper_changed = pyqtSignal(str, int)      # 入り抜き ("in"/"out"/"tip", 値)
    brush_changed         = pyqtSignal(str)
    symmetry_toggled      = pyqtSignal(bool)
    shape_fill_changed    = pyqtSignal(str)
    fill_expand_changed   = pyqtSignal(int)   # バケツ塗り拡張px（負=縮小）
    fill_close_gap_changed = pyqtSignal(int)  # 線画の途切れを塞ぐpx
    fill_line_sensitivity_changed = pyqtSignal(int)  # 薄い線を拾う感度(%)
    fill_reference_mode_changed = pyqtSignal(str)  # 複数参照 ref / ref_self
    select_mode_changed   = pyqtSignal(str)   # "select" | "transform"
    invert_selection_requested = pyqtSignal()
    pivot_changed         = pyqtSignal(int, int)  # (ax, ay) 変形基準点
    pivot_mode_changed    = pyqtSignal(str)        # "preset" | "custom"
    transform_mode_changed = pyqtSignal(str)       # "standard" | "perspective" | "mesh"
    mesh_div_changed = pyqtSignal(int)              # メッシュ分割数
    blur_size_changed = pyqtSignal(int)
    blur_strength_changed = pyqtSignal(int)         # 0〜100 (%)
    # ベクターレイヤー用（ペン選択時のみ出る）
    vector_pen_mode_changed = pyqtSignal(str)       # "draw" | "select"
    vector_width_changed = pyqtSignal(int)          # 選択中の線の太さ
    vector_apply_color_requested = pyqtSignal()     # 今の色を選択中の線に塗る
    vector_delete_requested = pyqtSignal()          # 選択中の線を消す
    vector_smooth_toggled = pyqtSignal(bool)        # なめらか / 直線つなぎ
    vector_erase_mode_changed = pyqtSignal(str)     # "cut" | "whole"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # タイトルバー
        title_bar = QFrame()
        title_bar.setStyleSheet("background:#2a2a2a; color:white;")
        title_bar.setFixedHeight(28)
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(8, 2, 8, 2)
        self._title = QLabel("ツールオプション")
        self._title.setStyleSheet("color:white; font-weight:bold; font-size:11px;")
        tb_layout.addWidget(self._title)
        outer.addWidget(title_bar)

        # スクロールエリア内にコンテンツ
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background: #f5f5f5; }")
        self._content = QWidget()
        self._content.setStyleSheet("""
            QWidget { background: #f5f5f5; }
            QLabel { font-size: 12px; color: #222; }
            QSpinBox, QComboBox {
                min-height: 24px; font-size: 12px;
                background-color: #ffffff; color: #222;
                border: 1px solid #bbb; border-radius: 3px;
                padding-right: 2px;
            }
            QCheckBox { font-size: 12px; color: #222; }
            QSlider::groove:horizontal {
                height: 4px; background: #c8c8c8; border-radius: 2px; }
            QSlider::handle:horizontal {
                width: 14px; height: 14px; margin: -6px 0;
                background: #4a90d9; border-radius: 7px; }
        """)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(10, 10, 10, 10)
        self._content_layout.setSpacing(10)
        self._content_layout.addStretch()
        scroll.setWidget(self._content)
        outer.addWidget(scroll, 1)

        self._current_tool: Tool | None = None
        self._widgets: list[QWidget] = []

    # ── 外部から呼ぶ ──────────────────────────────────────────────────────────

    def set_tool(self, tool: Tool,
                 pen_size: int = 5, eraser_size: int = 20,
                 brush_key: str = "round", symmetry: bool = False,
                 shape_fill: str = "none", fill_expand: int = 0,
                 fill_close_gap: int = 0, fill_line_sensitivity: int = 0,
                 fill_reference_mode: str = "ref_self",
                 select_mode: str = "select",
                 transform_mode: str = "standard",
                 blur_size: int = 30, blur_strength: int = 50,
                 mesh_div: int = 3,
                 is_vector: bool = False,
                 vector_pen_mode: str = "draw",
                 vector_selected=None,
                 vector_erase_mode: str = "cut",
                 stabilization: int = 5,
                 taper: dict | None = None,
                 pen_opacity: int = 100,
                 pen_transparent: bool = False,
                 eraser_soft: bool = False,
                 fill_tolerance: int = 0,
                 text_font: str = "Arial",
                 text_size: int = 40):
        self._current_tool = tool
        self._clear()

        label_map = {
            Tool.PEN: "ペン",
            Tool.ERASER: "消しゴム",
            Tool.FILL: "バケツ塗り",
            Tool.LINE: "直線",
            Tool.RECT: "四角形",
            Tool.ELLIPSE: "楕円",
            Tool.SELECT_RECT: "矩形選択",
            Tool.LASSO: "投げなわ",
            Tool.LASSO_FILL: "囲み内塗りつぶし",
            Tool.BLUR: "ぼかし",
            Tool.MOVE: "移動",
            Tool.TRANSFORM: "自由変形",
            Tool.EYEDROPPER: "スポイト",
            Tool.TEXT: "テキスト",
        }
        self._title.setText(label_map.get(tool, "ツールオプション"))

        if tool == Tool.PEN and is_vector:
            self._build_vector_pen(pen_size, vector_pen_mode, vector_selected,
                                   stabilization)

        elif tool == Tool.PEN:
            self._add_spinbox("ブラシサイズ", pen_size, 1, 200,
                              lambda v: self.pen_size_changed.emit(v))
            self._add_size_presets("pen_size", PEN_SIZE_PRESETS,
                                   self.pen_size_changed)
            self._add_spinbox("不透明度", pen_opacity, 1, 100,
                              lambda v: self.pen_opacity_changed.emit(v),
                              key="pen_opacity", suffix=" %",
                              tooltip="線の濃さ。1本の線の中で重なっても濃くなりません。\n"
                                      "重ね塗りしたいときは線を分けて描きます。")
            self._add_toggle("透明色で描く", pen_transparent,
                             lambda v: self.pen_transparent_toggled.emit(v),
                             tooltip="ペンで描いた所が消えます。\n"
                                     "ブラシの形・不透明度のまま消したいとき用。")
            self._add_stabilization(stabilization)
            self._add_taper(taper)
            self._add_brush_combo(brush_key)
            self._add_toggle("対称定規", symmetry,
                             lambda v: self.symmetry_toggled.emit(v))

        elif tool == Tool.ERASER and is_vector:
            self._build_vector_eraser(eraser_size, vector_erase_mode)

        elif tool == Tool.ERASER:
            self._add_spinbox("消しゴムサイズ", eraser_size, 1, 300,
                              lambda v: self.eraser_size_changed.emit(v))
            self._add_size_presets("eraser_size", ERASER_SIZE_PRESETS,
                                   self.eraser_size_changed)
            self._add_toggle("ソフト（ふちをぼかして消す）", eraser_soft,
                             lambda v: self.eraser_soft_toggled.emit(v),
                             tooltip="ふちほど弱く消えるので、\n"
                                     "影やグラデーションを薄くするのに向いています。")
            self._add_stabilization(stabilization)
            self._add_taper(taper)

        elif tool in (Tool.FILL, Tool.LASSO_FILL):
            if tool == Tool.LASSO_FILL:
                self._add_label("囲んだ範囲の中で、線で閉じている\n"
                                "ところだけを塗ります。\n"
                                "マウスを離すと自動で実行されます。\n"
                                "下の設定はバケツ塗りと共通です。")
            self._add_fill_reference_combo(fill_reference_mode)
            if tool == Tool.FILL:
                self._add_spinbox("色の誤差", fill_tolerance, 0, 100,
                                  lambda v: self.fill_tolerance_changed.emit(v),
                                  key="fill_tolerance", suffix=" %",
                                  tooltip="クリックした所と、どれくらい違う色まで\n"
                                          "同じ色とみなして塗るか。\n"
                                          "参照レイヤーがないとき・「すべてのレイヤー」\n"
                                          "のときに効きます。")
            self._add_spinbox("拡張/縮小 (px)", fill_expand, -30, 30,
                              lambda v: self.fill_expand_changed.emit(v),
                              tooltip="正: 塗り範囲を広げる  負: 塗り範囲を縮める")
            self._add_spinbox("隙間を閉じる (px)", fill_close_gap, 0, 20,
                              lambda v: self.fill_close_gap_changed.emit(v),
                              key="fill_close_gap",
                              tooltip="線画がこの px 以内で途切れていても、\n"
                                      "つながっているとみなして塗ります。\n"
                                      "0 で無効。塗り範囲は痩せません。")
            self._add_spinbox("薄い線を拾う", fill_line_sensitivity, 0, 100,
                              lambda v: self.fill_line_sensitivity_changed.emit(v),
                              key="fill_line_sensitivity", suffix=" %",
                              tooltip="上げるほど薄いピクセルまで線として扱います。\n"
                                      "色が薄い線やアンチエイリアス部分が\n"
                                      "「途切れている」と判定されるのを防げます。")

        elif tool == Tool.BLUR:
            self._add_spinbox("ブラシサイズ", blur_size, 1, 200,
                              lambda v: self.blur_size_changed.emit(v),
                              key="blur_size")
            self._add_spinbox("ぼかし強度", blur_strength, 1, 100,
                              lambda v: self.blur_strength_changed.emit(v),
                              suffix=" %")

        elif tool in (Tool.RECT, Tool.ELLIPSE, Tool.LINE):
            self._add_spinbox("線の太さ", pen_size, 1, 200,
                              lambda v: self.pen_size_changed.emit(v))
            if tool in (Tool.RECT, Tool.ELLIPSE):
                self._add_fill_combo(shape_fill)

        elif tool in (Tool.SELECT_RECT, Tool.LASSO):
            self._add_select_mode_combo(select_mode)
            self._add_invert_selection_button()
            # 以下は「変形モード」のときだけ効く設定なので、区切って下にまとめる。
            self._add_separator()
            self._add_transform_mode_combo(transform_mode, mesh_div)
            self._add_pivot_selector()

        elif tool == Tool.TRANSFORM:
            self._add_transform_mode_combo(transform_mode, mesh_div)
            self._add_pivot_selector()

        elif tool == Tool.MOVE:
            self._add_label("矢印キーで 1px 移動\nShift+矢印で 10px 移動")

        elif tool == Tool.EYEDROPPER:
            self._add_label("Alt キーでも\n一時スポイトになります")

        elif tool == Tool.TEXT:
            self._add_font_combo(text_font)
            self._add_spinbox("文字サイズ", text_size, 4, 500,
                              lambda v: self.text_size_changed.emit(v),
                              key="text_size", suffix=" px")
            self._add_label("キャンバスをクリックして文字を入力します。\n"
                            "色は今のペンの色を使います。")

    def sync_pen_size(self, v: int):
        self.sync_size("pen_size", v)

    def sync_size(self, key: str, v: int):
        """キーボードやプリセットで変えた値を、表示中の入力欄に反映する。"""
        for w in self._widgets:
            if getattr(w, '_opt_key', None) == key:
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)

    # ── ビルダー ──────────────────────────────────────────────────────────────

    def _clear(self):
        layout = self._content_layout
        while layout.count() > 1:
            item = layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._widgets.clear()

    def _add_row(self, label: str, widget: QWidget, tooltip: str = ""):
        row = QWidget()
        rl = QVBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(2)
        lbl = QLabel(label)
        lbl.setStyleSheet("font-size:11px; color:#333;")
        if tooltip:
            lbl.setToolTip(tooltip)
        rl.addWidget(lbl)
        rl.addWidget(widget)
        self._content_layout.insertWidget(self._content_layout.count() - 1, row)
        self._widgets.append(widget)

    def _add_spinbox(self, label: str, value: int, lo: int, hi: int,
                     callback, tooltip: str = "", key: str = "",
                     suffix: str = ""):
        w = _SliderSpin(value, lo, hi, suffix)
        if tooltip:
            w.setToolTip(tooltip)
        if not key:
            # label から自動判定
            if label in ("ブラシサイズ", "線の太さ"):
                key = "pen_size"
            elif "消しゴム" in label:
                key = "eraser_size"
            elif "拡張" in label:
                key = "fill_expand"
        w._opt_key = key  # type: ignore
        w.connect_changed(callback)
        self._add_row(label, w, tooltip)

    def _add_stabilization(self, value: int):
        """ペン・消しゴム共通の手ブレ補正。指やマウスで描くときの
        ガタつきを抑える（クリップスタジオの「手ブレ補正」と同じ考え方）。"""
        from canvas import STABILIZATION_MAX
        self._add_spinbox("手ブレ補正", value, 0, STABILIZATION_MAX,
                          lambda v: self.stabilization_changed.emit(v),
                          key="stabilization",
                          tooltip="上げるほど線のガタつきが減り、なめらかになります。\n"
                                  "そのぶん線がカーソルから少し遅れて付いてきます。\n"
                                  "0 で補正なし。ペンと消しゴムで共通です。")

    def _add_taper(self, taper: dict | None):
        """入り抜き。描き終わると線の始め・終わりが指定の細さまで細くなる
        （筆圧ではなく長さで決める。クリップスタジオの「入り抜き」と同じ考え方）。"""
        from canvas import TAPER_MAX
        t = taper or {"in": 0, "out": 0, "tip": 0}
        self._add_spinbox("入り (px)", t["in"], 0, TAPER_MAX,
                          lambda v: self.taper_changed.emit("in", v),
                          key="taper_in",
                          tooltip="線の描き始めを、この長さをかけて\n"
                                  "細い先端から太くしていきます。0 で入りなし。")
        self._add_spinbox("抜き (px)", t["out"], 0, TAPER_MAX,
                          lambda v: self.taper_changed.emit("out", v),
                          key="taper_out",
                          tooltip="描き終わって離したときに、線の終わりを\n"
                                  "この長さをかけて細くします。0 で抜きなし。")
        self._add_spinbox("先端の太さ", t["tip"], 0, 100,
                          lambda v: self.taper_changed.emit("tip", v),
                          key="taper_tip", suffix=" %",
                          tooltip="入り・抜きの一番先の太さを、ブラシサイズに\n"
                                  "対する割合で決めます。0 % で一番細く尖ります。")

    def _add_brush_combo(self, current_key: str):
        cb = QComboBox()
        keys = list(BRUSH_LABELS.keys())
        for k in keys:
            cb.addItem(BRUSH_LABELS[k], k)
        idx = keys.index(current_key) if current_key in keys else 0
        cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(
            lambda i: self.brush_changed.emit(keys[i]))
        self._add_row("ブラシ種類", cb)

    def _add_fill_combo(self, current: str):
        cb = QComboBox()
        cb.addItem("枠線のみ", "none")
        cb.addItem("塗りのみ", "fill")
        cb.addItem("枠線＋塗り", "both")
        idx = {"none": 0, "fill": 1, "both": 2}.get(current, 0)
        cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(
            lambda i: self.shape_fill_changed.emit(cb.itemData(i)))
        self._add_row("図形塗り", cb)

    def _add_fill_reference_combo(self, current: str):
        """クリスタの「複数参照」に相当。どのレイヤーを見て塗りを止めるか。"""
        cb = QComboBox()
        cb.addItem("参照レイヤー＋編集レイヤー", "ref_self")
        cb.addItem("参照レイヤーのみ", "ref")
        cb.addItem("すべてのレイヤー", "all")
        idx = {"ref_self": 0, "ref": 1, "all": 2}.get(current, 0)
        cb.setCurrentIndex(idx)
        cb.setToolTip(
            "塗りを止める境界をどのレイヤーから探すか。\n\n"
            "参照レイヤー＋編集レイヤー:\n"
            "  今のレイヤーに描いた囲み線でも塗りが止まります。\n"
            "  自分で丸く囲ってその中だけ塗るとき用。\n\n"
            "参照レイヤーのみ:\n"
            "  線画レイヤーだけを境界にします。すでに塗った色は\n"
            "  無視するので、隣り合う色を塗り分けるとき塗り漏れが出ません。\n\n"
            "すべてのレイヤー:\n"
            "  見えている絵全体の色で境界を決めます。\n"
            "  参照レイヤーを設定しなくても、線画の下のレイヤーに塗れます。")
        cb.currentIndexChanged.connect(
            lambda i: self.fill_reference_mode_changed.emit(cb.itemData(i)))
        self._add_row("複数参照", cb)

    def _add_size_presets(self, key: str, sizes, signal):
        """よく使う太さをワンクリックで選ぶボタン列。"""
        row = QWidget()
        hl = QHBoxLayout(row)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(2)
        for size in sizes:
            btn = QPushButton(str(size))
            btn.setFixedHeight(22)
            btn.setStyleSheet("QPushButton { font-size:11px; padding:0 2px; }")
            btn.setToolTip(f"太さを {size} にする")
            btn._preset_size = size  # type: ignore

            def _pick(_=False, v=size):
                signal.emit(v)
                self.sync_size(key, v)
            btn.clicked.connect(_pick)
            hl.addWidget(btn)
        row._preset_key = key  # type: ignore
        self._content_layout.insertWidget(self._content_layout.count() - 1, row)
        self._widgets.append(row)

    def _add_font_combo(self, family: str):
        from PyQt6.QtWidgets import QFontComboBox
        from PyQt6.QtGui import QFont
        cb = QFontComboBox()
        cb.setCurrentFont(QFont(family))
        cb.currentFontChanged.connect(
            lambda f: self.text_font_changed.emit(f.family()))
        self._add_row("フォント", cb)

    def _add_toggle(self, label: str, value: bool, callback, tooltip: str = ""):
        from PyQt6.QtWidgets import QCheckBox
        cb = QCheckBox(label)
        cb.setChecked(value)
        if tooltip:
            cb.setToolTip(tooltip)
        cb.toggled.connect(callback)
        self._content_layout.insertWidget(self._content_layout.count() - 1, cb)
        self._widgets.append(cb)

    def _add_invert_selection_button(self):
        """選択範囲を反転する。選択がないときは押しても何も起きない。"""
        btn = QPushButton("選択範囲を反転")
        btn.setToolTip("選択されている部分と、されていない部分を入れ替えます。\n"
                       "背景だけを塗りたいときなど、囲みたくない側が\n"
                       "単純な形のときに便利です。（Ctrl+Shift+I）")
        btn.clicked.connect(lambda: self.invert_selection_requested.emit())
        # ラベル行は付けない（ボタン自身が何をするか示しているため）
        self._content_layout.insertWidget(self._content_layout.count() - 1, btn)
        self._widgets.append(btn)

    def _add_select_mode_combo(self, current: str):
        cb = QComboBox()
        cb.addItem("選択のみ", "select")
        cb.addItem("選択範囲内クリックで変形", "transform")
        idx = 0 if current == "select" else 1
        cb.setCurrentIndex(idx)
        cb.currentIndexChanged.connect(
            lambda i: self.select_mode_changed.emit(cb.itemData(i)))
        self._add_row("クリック時の動作", cb)

    def _add_transform_mode_combo(self, current: str = "standard",
                                  mesh_div_value: int = 3):
        cb = QComboBox()
        cb.addItem("拡縮・回転", "standard")
        cb.addItem("自由変形（4隅を個別に動かす）", "perspective")
        cb.addItem("メッシュ変形", "mesh")
        idx = {"standard": 0, "perspective": 1, "mesh": 2}.get(current, 0)
        cb.setCurrentIndex(idx)

        mesh_div = QSpinBox()
        mesh_div.setRange(2, 8)
        mesh_div.setValue(mesh_div_value)
        mesh_div.setPrefix("分割: ")
        mesh_div.setSuffix(" ×")
        mesh_div.setVisible(current == "mesh")
        mesh_div.valueChanged.connect(lambda v: self.mesh_div_changed.emit(v))

        def on_mode(i):
            mode = cb.itemData(i)
            mesh_div.setVisible(mode == "mesh")
            self.transform_mode_changed.emit(mode)

        cb.currentIndexChanged.connect(on_mode)
        self._add_row("変形モード", cb)
        self._content_layout.insertWidget(self._content_layout.count() - 1, mesh_div)
        self._widgets.append(mesh_div)

    def _add_pivot_selector(self):
        from main import AnchorWidget
        lbl = QLabel("変形基準点")
        lbl.setStyleSheet("color:#555; font-size:11px;")
        self._content_layout.insertWidget(self._content_layout.count() - 1, lbl)
        self._widgets.append(lbl)

        mode_cb = QComboBox()
        mode_cb.addItem("プリセット（9点）", "preset")
        mode_cb.addItem("任意（ドラッグ）", "custom")
        self._content_layout.insertWidget(self._content_layout.count() - 1, mode_cb)
        self._widgets.append(mode_cb)

        aw = AnchorWidget()
        aw.anchor_changed.connect(lambda ax, ay: self.pivot_changed.emit(ax, ay))
        self._content_layout.insertWidget(self._content_layout.count() - 1, aw)
        self._widgets.append(aw)

        hint = QLabel("キャンバス上の青い十字を\nドラッグで移動できます")
        hint.setStyleSheet("color:#888; font-size:10px;")
        hint.setWordWrap(True)
        hint.setVisible(False)
        self._content_layout.insertWidget(self._content_layout.count() - 1, hint)
        self._widgets.append(hint)

        def on_mode_change(i):
            mode = mode_cb.itemData(i)
            aw.setVisible(mode == "preset")
            hint.setVisible(mode == "custom")
            self.pivot_mode_changed.emit(mode)

        mode_cb.currentIndexChanged.connect(on_mode_change)

    def _build_vector_eraser(self, eraser_size: int, mode: str):
        """ベクターレイヤー選択中の消しゴムの設定。

        ピクセルを削るのではなく線を削るので、サイズは太さではなく
        「どこまで近ければ当たったとみなすか」の広さとして効く。
        """
        modes = [("交点まで消す", "cut"),
                 ("線ごと消す", "whole")]
        cb = QComboBox()
        for label, key in modes:
            cb.addItem(label, key)
        keys = [k for _, k in modes]
        cb.setCurrentIndex(keys.index(mode) if mode in keys else 0)
        cb.currentIndexChanged.connect(
            lambda i: self.vector_erase_mode_changed.emit(cb.itemData(i)))
        self._add_row("消去方法", cb,
                      tooltip="「交点まで消す」はクリックした所から\n"
                              "他の線とぶつかる所までを消します。\n"
                              "「線ごと消す」は当たった線を丸ごと消します。")

        self._add_spinbox("消しゴムサイズ", eraser_size, 1, 300,
                          lambda v: self.eraser_size_changed.emit(v),
                          tooltip="線に当たったとみなす広さです。")
        self._add_label("ベクターレイヤーです。消しゴムは\n"
                        "線そのものを消します。なぞると続けて消せます。")

    def _build_vector_pen(self, pen_size: int, mode: str, selected,
                          stabilization: int = 5):
        """ベクターレイヤー選択中のペンの設定。

        描く・選ぶ・制御点を削除する の3つを切り替えて使う。選ぶモードでは、
        選んでいる線の太さ・色・形をここから直せる。
        """
        modes = [("描く", "draw"),
                 ("線を選んで直す", "select"),
                 ("制御点を削除", "delpoint")]
        cb = QComboBox()
        for label, key in modes:
            cb.addItem(label, key)
        keys = [k for _, k in modes]
        cb.setCurrentIndex(keys.index(mode) if mode in keys else 0)
        cb.currentIndexChanged.connect(
            lambda i: self.vector_pen_mode_changed.emit(
                cb.itemData(i)))
        self._add_row("ペンの役割", cb,
                      tooltip="「描く」で新しい線を引き、\n"
                              "「線を選んで直す」で引いた線を編集し、\n"
                              "「制御点を削除」で点をクリックして減らします。")

        if mode == "draw":
            self._add_spinbox("ブラシサイズ", pen_size, 1, 200,
                              lambda v: self.pen_size_changed.emit(v))
            self._add_stabilization(stabilization)
            self._add_label("ベクターレイヤーです。引いた線は\n"
                            "あとから形も太さも色も変えられます。")
            return

        self._add_separator()

        if mode == "delpoint":
            if selected is None:
                self._add_label("線をクリックして選んでから、\n"
                                "消したい制御点（□）をクリックします。")
            else:
                self._add_label(
                    "制御点（□）をクリックすると消えます。\n"
                    "点が2つになったら、それ以上は減りません。\n"
                    "別の線に移るときは、その線をクリックします。")
            return

        if selected is None:
            self._add_label("線をクリックすると選べます。\n"
                            "選ぶと制御点（□）が出ます。")
            return

        self._add_spinbox("選択中の線の太さ", max(1, int(round(selected.width))),
                          1, 200, lambda v: self.vector_width_changed.emit(v),
                          key="vector_width")
        self._add_toggle("なめらかにつなぐ", bool(selected.smooth),
                         lambda v: self.vector_smooth_toggled.emit(v))

        btn = QPushButton("今の色を線に塗る")
        btn.setToolTip("選んでいる線の色を、いま選んでいる描画色にします。")
        btn.clicked.connect(lambda: self.vector_apply_color_requested.emit())
        self._content_layout.insertWidget(self._content_layout.count() - 1, btn)
        self._widgets.append(btn)

        btn_del = QPushButton("この線を削除")
        btn_del.clicked.connect(lambda: self.vector_delete_requested.emit())
        self._content_layout.insertWidget(self._content_layout.count() - 1, btn_del)
        self._widgets.append(btn_del)

        self._add_separator()
        self._add_label(
            "制御点（□）をドラッグ＝形を変える\n"
            "Alt+Shift+クリック＝ハンドルの出し入れ\n"
            "Alt+クリック＝その点を削除\n"
            "　（「制御点を削除」に切り替えても消せます）\n"
            "Shift+線の上をクリック＝点を追加\n"
            "ハンドル（○）をドラッグ＝曲がり具合\n"
            "　Alt を押しながらで片側だけ動く（角）")

    def _add_separator(self):
        """設定のまとまりを視覚的に区切る横線。"""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("color:#ccc;")
        self._content_layout.insertWidget(self._content_layout.count() - 1, line)

    def _add_label(self, text: str):
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#555; font-size:11px;")
        lbl.setWordWrap(True)
        self._content_layout.insertWidget(self._content_layout.count() - 1, lbl)
