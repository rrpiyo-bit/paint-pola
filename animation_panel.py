"""アニメーションのタイムラインパネル — CLIP STUDIO の「アニメーションフォルダ」風。

セルはアニメーションフォルダの中のレイヤーとして持ち、どのコマにどのセルを
出すかをタイムラインで指定する。データの扱いは anim.py にまとめてある。
"""
from __future__ import annotations

import os

from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
                              QLabel, QSpinBox, QScrollArea, QFileDialog,
                              QFrame, QMessageBox, QMenu, QToolButton)
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PyQt6.QtCore import Qt, QTimer, QRect, QPoint, QEvent, pyqtSignal

import anim
from anim import BLANK
from layer import LayerStack

_PREVIEW_SIZE = 320
_NAME_W = 120     # 左端のトラック名の列
_CELL_W = 56
_ROW_H = 50
_HEADER_H = 18
_BG_ROW_H = 22    # 背景（フォルダの外のレイヤー）の行
_CURRENT = QColor(220, 60, 60)


class PlaybackPreview(QWidget):
    """再生プレビューウィンドウ（フローティング）。"""
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("再生プレビュー")
        self.setFixedSize(_PREVIEW_SIZE + 20, _PREVIEW_SIZE + 40)
        self._image: QImage | None = None
        self._frame_label = QLabel("", self)
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._canvas = QWidget()
        self._canvas.setFixedSize(_PREVIEW_SIZE, _PREVIEW_SIZE)
        layout.addWidget(self._canvas)
        layout.addWidget(self._frame_label)

    def set_frame(self, img: QImage, index: int, total: int):
        self._image = img
        self._frame_label.setText(f"{index + 1} / {total}")
        self._canvas.update()
        self.update()

    def paintEvent(self, event):
        if self._image is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # チェッカーボード背景
        ox, oy = 8, 8
        for cy in range(0, _PREVIEW_SIZE, 8):
            for cx in range(0, _PREVIEW_SIZE, 8):
                color = QColor(200, 200, 200) if (cx // 8 + cy // 8) % 2 == 0 else QColor(255, 255, 255)
                p.fillRect(ox + cx, oy + cy, 8, 8, color)
        scaled = self._image.scaled(_PREVIEW_SIZE, _PREVIEW_SIZE,
                                     Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)
        dx = ox + (_PREVIEW_SIZE - scaled.width()) // 2
        dy = oy + (_PREVIEW_SIZE - scaled.height()) // 2
        p.drawImage(dx, dy, scaled)
        p.end()


class TimelineView(QWidget):
    """コマ番号の見出し・トラック（アニメーションフォルダ）ごとのセル指定を描く。

    左端のトラック名は横スクロールしても見えるよう、表示中の左端に描く。
    """
    cell_clicked = pyqtSignal(object, int, bool)    # (フォルダ or None, コマ, Shift で範囲を広げる)
    cell_double_clicked = pyqtSignal(object, int)
    cell_dragged = pyqtSignal(object, int, bool)    # (フォルダ, 選択範囲の移動先の先頭, 写すか)
    context_requested = pyqtSignal(object, int, QPoint)
    key_command = pyqtSignal(str)                   # "copy" "cut" "paste" "paste_ref" "clear" "prev" "next"

    # タイムラインにフォーカスがあるときは、メインウィンドウのショートカット
    # （キャンバスのコピー・貼り付け等）より先にこちらで受ける
    _KEYS = {
        (Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier): "copy",
        (Qt.Key.Key_X, Qt.KeyboardModifier.ControlModifier): "cut",
        (Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier): "paste",
        (Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier
         | Qt.KeyboardModifier.ShiftModifier): "paste_ref",
        (Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier): "clear",
        (Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier): "clear",
        (Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier): "prev",
        (Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier): "next",
    }

    def __init__(self, panel: "AnimationPanel"):
        super().__init__()
        self._panel = panel
        self._hbar = None
        self._thumbs: dict[int, tuple[object, QPixmap]] = {}
        # ドラッグ中: (フォルダ, つかんだコマと選択範囲の先頭の差)
        self._drag: tuple[object, int] | None = None
        self._drag_to: int | None = None   # 選択範囲を置く先頭のコマ
        self._drag_copy = False
        self.setMouseTracking(False)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def _key_command(self, event) -> str | None:
        mods = event.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                    | Qt.KeyboardModifier.ShiftModifier
                                    | Qt.KeyboardModifier.AltModifier)
        return self._KEYS.get((Qt.Key(event.key()), mods))

    def event(self, e):
        if e.type() == QEvent.Type.ShortcutOverride and self._key_command(e):
            e.accept()
            return True
        return super().event(e)

    def keyPressEvent(self, event):
        cmd = self._key_command(event)
        if cmd:
            self.key_command.emit(cmd)
            return
        super().keyPressEvent(event)

    def set_hbar(self, bar) -> None:
        self._hbar = bar
        bar.valueChanged.connect(lambda _v: self.update())

    @property
    def stack(self) -> LayerStack:
        return self._panel.stack

    def folders(self) -> list:
        return anim.anim_folders(self.stack.layers)

    def relayout(self) -> None:
        n = self.stack.anim_length
        rows = max(1, len(self.folders()))
        self.setFixedSize(_NAME_W + n * _CELL_W + 1,
                          _HEADER_H + rows * _ROW_H + _BG_ROW_H)
        self.update()

    def _left(self) -> int:
        return self._hbar.value() if self._hbar is not None else 0

    def frame_x(self, f: int) -> int:
        return _NAME_W + f * _CELL_W

    def hit(self, pos: QPoint):
        """("header"|"cell"|"bg"|"name", フォルダ or None, コマ) を返す。"""
        x, y = pos.x(), pos.y()
        f = (x - _NAME_W) // _CELL_W
        in_names = x < self._left() + _NAME_W
        if not in_names and not 0 <= f < self.stack.anim_length:
            return None
        if y < _HEADER_H:
            return ("header", None, f)
        row = (y - _HEADER_H) // _ROW_H
        folders = self.folders()
        if row < len(folders):
            return ("name" if in_names else "cell", folders[row], f)
        return ("name" if in_names else "bg", None, f)

    def _thumb(self, cel) -> QPixmap:
        key = anim.cel_key(cel)
        hit = self._thumbs.get(cel.uid)
        if hit is not None and hit[0] == key:
            return hit[1]
        if cel.is_group:
            img = cel.composite()
        else:
            img = cel.image_with_effects()
        pm = QPixmap.fromImage(img.scaled(
            _CELL_W - 6, _ROW_H - 18, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        self._thumbs[cel.uid] = (key, pm)
        return pm

    def paintEvent(self, event):
        p = QPainter(self)
        pal = self.palette()
        base, alt = pal.base().color(), pal.alternateBase().color()
        text, mid = pal.text().color(), pal.mid().color()
        hl = pal.highlight().color()
        clip = event.rect()
        p.fillRect(clip, pal.window())
        stack = self.stack
        n = stack.anim_length
        cur = stack.anim_frame
        folders = self.folders()
        active_folder = self._panel.current_folder()
        f0 = max(0, (clip.left() - _NAME_W) // _CELL_W)
        f1 = min(n - 1, (clip.right() - _NAME_W) // _CELL_W + 1)
        small = QFont(self.font())
        small.setPointSizeF(max(6.5, small.pointSizeF() - 1.5))
        p.setFont(small)

        # 見出し（コマ番号）
        for f in range(f0, f1 + 1):
            r = QRect(self.frame_x(f), 0, _CELL_W, _HEADER_H)
            p.fillRect(r, _CURRENT if f == cur else alt)
            p.setPen(QColor(255, 255, 255) if f == cur else text)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, str(f + 1))
            p.setPen(mid)
            p.drawLine(r.right(), 0, r.right(), _HEADER_H)

        # トラック
        play_f = self._panel.playing_frame()
        sel = self._panel.selection()
        sel_fill = QColor(hl.red(), hl.green(), hl.blue(), 70)
        for row, folder in enumerate(folders):
            y = _HEADER_H + row * _ROW_H
            for f in range(f0, f1 + 1):
                r = QRect(self.frame_x(f), y, _CELL_W, _ROW_H)
                p.fillRect(r, base if f % 2 == 0 else alt)
                if f == cur:
                    p.fillRect(r, QColor(_CURRENT.red(), _CURRENT.green(),
                                         _CURRENT.blue(), 45))
                if sel is not None and sel[0] is folder and sel[1] <= f <= sel[2]:
                    p.fillRect(r, sel_fill)
                v = folder.timeline[f] if f < len(folder.timeline) else None
                shown = anim.cel_at(folder, f)
                if v is not None and v != BLANK and shown is not None:
                    pm = self._thumb(shown)
                    p.fillRect(r.x() + 3, r.y() + 3, _CELL_W - 6, _ROW_H - 18,
                               QColor(255, 255, 255))
                    p.drawPixmap(r.x() + 3 + (_CELL_W - 6 - pm.width()) // 2,
                                 r.y() + 3 + (_ROW_H - 18 - pm.height()) // 2, pm)
                    p.setPen(text)
                    p.drawText(QRect(r.x() + 3, r.bottom() - 14, _CELL_W - 6, 14),
                               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                               shown.name)
                elif v == BLANK:
                    p.setPen(QPen(mid, 2))
                    c = r.center()
                    p.drawLine(c.x() - 7, c.y() - 7, c.x() + 7, c.y() + 7)
                    p.drawLine(c.x() - 7, c.y() + 7, c.x() + 7, c.y() - 7)
                elif shown is not None:
                    # 前のセルを出し続けているコマ: 線でつなぐ
                    p.setPen(QPen(hl, 3))
                    cy = r.y() + (_ROW_H - 18) // 2 + 3
                    p.drawLine(r.x(), cy, r.right(), cy)
                p.setPen(mid)
                p.drawLine(r.right(), r.top(), r.right(), r.bottom())
                if folder is active_folder and f == cur:
                    p.setPen(QPen(hl, 2))
                    p.drawRect(r.adjusted(1, 1, -2, -2))
            p.setPen(mid)
            p.drawLine(0, y + _ROW_H - 1, self.width(), y + _ROW_H - 1)
            # 選択範囲はサムネイルで塗りが隠れるので枠でも示す
            if sel is not None and sel[0] is folder:
                p.setPen(QPen(hl, 2))
                p.drawRect(QRect(self.frame_x(sel[1]), y,
                                 (sel[2] - sel[1] + 1) * _CELL_W, _ROW_H).adjusted(1, 1, -1, -1))
            # ドラッグ中: 置く先を点線の枠で示す（Ctrl で写すときは「＋」付き）
            if self._drag is not None and self._drag[0] is folder \
                    and sel is not None and self._drag_to is not None:
                size = sel[2] - sel[1] + 1
                r = QRect(self.frame_x(self._drag_to), y, size * _CELL_W, _ROW_H)
                p.setPen(QPen(hl, 2, Qt.PenStyle.DashLine))
                p.drawRect(r.adjusted(2, 2, -3, -3))
                if self._drag_copy:
                    p.drawText(r.adjusted(6, 4, -4, 0),
                               Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, "＋")

        # 背景の行（フォルダの外のレイヤーは全コマ共通）
        y = _HEADER_H + max(1, len(folders)) * _ROW_H
        bar = QRect(self.frame_x(f0), y + 4, (f1 - f0 + 1) * _CELL_W, _BG_ROW_H - 8)
        p.fillRect(bar, mid)

        if play_f is not None:
            x = self.frame_x(play_f)
            p.setPen(QPen(_CURRENT, 2))
            p.drawLine(x, 0, x, self.height())

        # トラック名（スクロールしても左端に残す）
        left = self._left()
        p.setFont(self.font())
        name_rect = QRect(left, 0, _NAME_W, self.height())
        p.fillRect(name_rect, pal.window())
        p.setPen(mid)
        p.drawLine(left + _NAME_W - 1, 0, left + _NAME_W - 1, self.height())
        for row, folder in enumerate(folders):
            r = QRect(left + 4, _HEADER_H + row * _ROW_H, _NAME_W - 8, _ROW_H)
            if folder is active_folder:
                p.fillRect(r.adjusted(-4, 0, 3, 0),
                           QColor(hl.red(), hl.green(), hl.blue(), 50))
            p.setPen(text if folder.visible else mid)
            p.drawText(r, Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
                       "🎞 " + folder.name)
        p.setPen(text)
        if not folders:
            p.drawText(QRect(left + 4, _HEADER_H, _NAME_W - 8, _ROW_H),
                       Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
                       "（アニメーションフォルダなし）")
        p.drawText(QRect(left + 4, y, _NAME_W - 8, _BG_ROW_H),
                   Qt.AlignmentFlag.AlignVCenter, "背景（全コマ）")
        p.end()

    def mousePressEvent(self, event):
        h = self.hit(event.position().toPoint())
        if h is None:
            return
        kind, folder, f = h
        if event.button() == Qt.MouseButton.RightButton:
            if kind in ("cell", "header", "bg"):
                self.context_requested.emit(folder, f, event.globalPosition().toPoint())
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if kind == "name":
            self.cell_clicked.emit(folder, self.stack.anim_frame, False)
            return
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        self.cell_clicked.emit(folder, f, shift)
        sel = self._panel.selection()
        # 選んだ範囲をつかんで動かせる（Shift で広げているときは動かさない）
        if kind == "cell" and not shift and sel is not None and sel[0] is folder:
            self._drag = (folder, f - sel[1])
            self._drag_to = sel[1]
            self._drag_copy = False

    def _drag_target(self, x: int) -> int:
        f = (x - _NAME_W) // _CELL_W
        return max(0, min(f - self._drag[1], anim.MAX_LENGTH - 1))

    def mouseMoveEvent(self, event):
        if self._drag is None:
            return
        to = self._drag_target(event.position().toPoint().x())
        copy = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if to != self._drag_to or copy != self._drag_copy:
            self._drag_to, self._drag_copy = to, copy
            self.update()

    def mouseReleaseEvent(self, event):
        drag, to = self._drag, self._drag_to
        copy = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        self._drag = self._drag_to = None
        self._drag_copy = False
        sel = self._panel.selection()
        if drag is not None and to is not None and sel is not None and to != sel[1]:
            self.cell_dragged.emit(drag[0], to, copy)
        self.update()

    def mouseDoubleClickEvent(self, event):
        h = self.hit(event.position().toPoint())
        if h is not None and h[0] == "cell":
            self.cell_double_clicked.emit(h[1], h[2])


class AnimationPanel(QWidget):
    """アニメーションタイムラインパネル。"""
    onion_skin_changed = pyqtSignal()   # オニオンスキンの表示が変わった
    # 作品の中身（セル・タイムライン・fps）が変わった（「未保存」印とパネル更新に使う）
    frames_changed = pyqtSignal()
    frame_changed = pyqtSignal()        # 今のコマが変わった（レイヤーの表示が変わる）
    anim_will_change = pyqtSignal()     # タイムラインだけを変える直前（取り消し用）
    structure_will_change = pyqtSignal()  # レイヤーを足す直前（取り消し用）
    playback_tick = pyqtSignal()

    def __init__(self, layer_stack: LayerStack | None = None, parent=None):
        super().__init__(parent)
        self.stack = layer_stack if layer_stack is not None else LayerStack()
        self._playing = False
        self._play_frame: int | None = None
        self._play_cache: dict[int, QImage] = {}
        self._loop = True
        self._onion_enabled = True
        self._onion_before = 2
        self._onion_after = 1
        # onion_overlay() の結果 (キー, 画像)
        self._onion_cache: tuple[tuple, QImage] | None = None
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_next)
        self._preview_win: PlaybackPreview | None = None
        # タイムラインで選んでいるコマ: (フォルダ, 先頭, 末尾)
        self._sel: tuple[object, int, int] | None = None
        self._sel_anchor: tuple[int, int] | None = None   # Shift で広げるときの起点
        # コピーしたコマ: コマごとのセル（空きコマは None）
        self._clip: list | None = None

        self.setFixedHeight(28 + _HEADER_H + 2 * _ROW_H + _BG_ROW_H + 24)
        self._build_ui()

    # ── UI ──

    def _btn(self, row, text: str, tip: str, slot, width: int | None = None,
             checkable: bool = False) -> QPushButton:
        b = QPushButton(text)
        b.setToolTip(tip)
        if width:
            b.setFixedWidth(width)
        b.setCheckable(checkable)
        if checkable:
            b.toggled.connect(slot)
        else:
            b.clicked.connect(slot)
        row.addWidget(b)
        return b

    def _spin(self, row, label: str, lo: int, hi: int, val: int, tip: str, slot,
              width: int = 50) -> QSpinBox:
        if label:
            row.addWidget(QLabel(label))
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        s.setFixedWidth(width)
        s.setToolTip(tip)
        s.valueChanged.connect(slot)
        row.addWidget(s)
        return s

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
        root.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(3)

        self._folder_btn = self._btn(top, "🎞＋フォルダ",
                                     "新しいアニメーションフォルダ（セル「1」入り）を作る",
                                     self._on_new_folder)
        self._cel_btn = self._btn(top, "＋セル",
                                  "新しいセルを作って今のコマ（埋まっていれば次のコマ）に置く",
                                  self._on_new_cel)
        self._dup_btn = self._btn(top, "⧉複製",
                                  "今のセルを写した新しいセルを次のコマに置く（中割り用）",
                                  self._on_dup_cel)
        top.addWidget(self._sep_v())
        self._first_btn = self._btn(top, "⏮", "最初のコマへ", lambda: self.go_to(0), 28)
        self._prev_btn = self._btn(top, "◀", "前のコマへ（,）", self.prev_frame, 28)
        self._play_btn = self._btn(top, "▶ 再生", "再生 / 停止", self._on_play_toggle)
        self._next_btn = self._btn(top, "▶", "次のコマへ（.）", self.next_frame, 28)
        self._last_btn = self._btn(top, "⏭", "最後のコマへ",
                                   lambda: self.go_to(self.stack.anim_length - 1), 28)
        self._loop_btn = self._btn(top, "🔁", "ループ再生", self._on_loop_toggle, 30, True)
        self._loop_btn.setChecked(True)
        self._fps_spin = self._spin(top, "fps", 1, 60, 8, "1秒あたりのコマ数",
                                    self._on_fps_changed, 46)
        self._len_spin = self._spin(top, "コマ数", 1, anim.MAX_LENGTH, self.stack.anim_length,
                                    "再生時間（コマ数）", self._on_length_changed, 54)
        top.addWidget(self._sep_v())
        self._onion_check = self._btn(top, "🧅", "オニオンスキン（前のコマ=青・後のコマ=緑）",
                                      self._on_onion_toggle, 30, True)
        self._onion_check.setChecked(True)
        self._onion_before_spin = self._spin(top, "前", 0, 5, 2, "前のコマを何枚出すか",
                                             self._on_onion_before, 38)
        self._onion_after_spin = self._spin(top, "後", 0, 5, 1, "後のコマを何枚出すか",
                                            self._on_onion_after, 38)
        top.addWidget(self._sep_v())

        more = QToolButton()
        more.setText("編集 ▾")
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._edit_menu = QMenu(more)
        self._fill_edit_menu(self._edit_menu, None, None)
        more.setMenu(self._edit_menu)
        self._edit_menu.aboutToShow.connect(
            lambda: self._fill_edit_menu(self._edit_menu, None, None))
        top.addWidget(more)

        self._preview_btn = self._btn(top, "プレビュー窓", "再生プレビューウィンドウを表示",
                                      self._on_preview_toggle, None, True)
        out = QToolButton()
        out.setText("書き出し ▾")
        out.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        out_menu = QMenu(out)
        out_menu.addAction("アニメーションGIF...", self._on_export_gif)
        out_menu.addAction("連番PNG...", self._on_export_png_sequence)
        out.setMenu(out_menu)
        top.addWidget(out)
        top.addStretch()
        self._frame_label = QLabel()
        top.addWidget(self._frame_label)
        root.addLayout(top)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._view = TimelineView(self)
        self._view.set_hbar(self._scroll.horizontalScrollBar())
        self._view.cell_clicked.connect(self._on_cell_clicked)
        self._view.cell_double_clicked.connect(self._on_cell_double_clicked)
        self._view.cell_dragged.connect(self._on_cell_dragged)
        self._view.context_requested.connect(self._on_context)
        self._view.key_command.connect(self._on_key_command)
        self._scroll.setWidget(self._view)
        root.addWidget(self._scroll, 1)
        self.sync()

    @staticmethod
    def _sep_v() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.VLine)
        f.setFixedWidth(2)
        return f

    def _fill_edit_menu(self, menu: QMenu, folder, frame) -> None:
        """右クリックと「編集」ボタンで共通のメニュー。folder/frame が None なら今の位置。"""
        menu.clear()
        folder = folder or self.current_folder()
        frame = self.stack.anim_frame if frame is None else frame
        has = folder is not None
        if has:
            sel = self._sel_or_current(folder, frame)
            what = f"コマ {sel[1] + 1}" if sel[1] == sel[2] else f"コマ {sel[1] + 1}〜{sel[2] + 1}"
            menu.addAction(f"コピー（{what}）\tCtrl+C", self.copy_frames)
            menu.addAction(f"切り取り（{what}）\tCtrl+X", self.cut_frames)
            can_paste = self._clip is not None
            a = menu.addAction(f"貼り付け（セルを複製）\tCtrl+V", self.paste_frames)
            a.setEnabled(can_paste)
            a = menu.addAction("貼り付け（同じセルを使い回す）\tCtrl+Shift+V",
                               lambda: self.paste_frames(reuse=True))
            a.setEnabled(can_paste and self._clip_in(folder))
            menu.addAction(f"空きコマにする（{what}）\tDelete", self.clear_frames)
            menu.addSeparator()
            cels = sorted(folder.children,
                          key=lambda c: (not c.name.isdigit(),
                                         int(c.name) if c.name.isdigit() else 0, c.name))
            sub = menu.addMenu(f"このコマ（{frame + 1}）に出すセル")
            for c in cels:
                sub.addAction(f"セル {c.name}",
                              lambda c=c: self._edit(lambda: anim.set_spec(folder, frame, c.uid)))
            sub.setEnabled(bool(cels))
            menu.addAction("指定を消す（前のセルを続ける）",
                           lambda: self._edit(lambda: anim.set_spec(folder, frame, None)))
            menu.addSeparator()
            menu.addAction("ここに新しいセル", lambda: self._new_cel_at(folder, frame))
        menu.addAction("コマを挿入（ここに1コマ足す）",
                       lambda: self._edit(lambda: anim.insert_frame(self.stack, frame)))
        menu.addAction("コマを削除（詰める）",
                       lambda: self._edit(lambda: anim.delete_frame(self.stack, frame)))
        if has:
            menu.addSeparator()
            for step in (1, 2, 3):
                menu.addAction(f"{step}コマ打ちにする（このフォルダ）",
                               lambda s=step: self._edit(lambda: anim.retime(self.stack, folder, s)))
            menu.addAction("セル番号を出てくる順に振り直す", lambda: self._renumber(folder))

    # ── 外部から呼ぶAPI ──

    def bind(self, layer_stack: LayerStack) -> None:
        self.stack = layer_stack
        self.sync()

    def current_folder(self):
        """操作の対象にするアニメーションフォルダ（選択中のセルのフォルダ）。"""
        st = self.stack
        active = st.active
        if active is not None:
            if anim.is_anim_folder(active):
                return active
            f = anim.folder_of(st, active)
            if f is not None:
                return f
        folders = anim.anim_folders(st.layers)
        return folders[0] if folders else None

    def sync(self) -> None:
        """レイヤー構成やタイムラインが外で変わったあと、表示を合わせる。"""
        st = self.stack
        for f in anim.anim_folders(st.layers):
            if len(f.timeline) != st.anim_length:
                f.timeline = (f.timeline + [None] * st.anim_length)[:st.anim_length]
        anim.apply_frame(st, st.anim_frame)
        if self._sel is not None:
            folder, s, e = self._sel
            if not any(f is folder for f in anim.anim_folders(st.layers)) \
                    or s >= st.anim_length:
                self._sel = self._sel_anchor = None
            else:
                self._sel = (folder, s, min(e, st.anim_length - 1))
        for spin, val in ((self._len_spin, st.anim_length),):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._play_cache.clear()
        self._view.relayout()
        self._update_label()
        self._update_preview()

    def clear(self) -> None:
        """新規・開くのあと。再生を止めて新しい作品に合わせる。"""
        self._stop_play()
        self._view._thumbs.clear()
        self._onion_cache = None
        self._sel = self._sel_anchor = None
        self.sync()

    @property
    def fps(self) -> int:
        return self._fps_spin.value()

    def set_fps(self, fps: int) -> None:
        self._fps_spin.blockSignals(True)
        self._fps_spin.setValue(max(1, min(60, int(fps))))
        self._fps_spin.blockSignals(False)

    def playing_frame(self) -> int | None:
        return self._play_frame if self._playing else None

    def playback_image(self) -> QImage | None:
        """再生中にキャンバスへ出す絵。止まっているときは None。"""
        if not self._playing or self._play_frame is None:
            return None
        return self._frame_image(self._play_frame)

    def onion_overlay(self) -> QImage | None:
        """キャンバスに重ねるオニオンスキンを1枚にまとめた画像。

        アニメーションモードでない（パネルを隠している）ときと再生中は None。
        前のコマのセルは青、後のコマのセルは緑に塗って出す（CLIP STUDIO と同じ）。
        セル・今のコマ・表示枚数が変わったときだけ作り直す。
        """
        # setVisible(False) で隠したときだけ。一度も表示していない単独の
        # パネルも isHidden() は True になるため、明示的に隠したかで見る。
        if self.isHidden() and self.testAttribute(
                Qt.WidgetAttribute.WA_WState_ExplicitShowHide):
            return None
        if not self._onion_enabled or self._playing:
            return None
        st = self.stack
        items = anim.onion_items(st, st.anim_frame, self._onion_before,
                                 self._onion_after, self._loop)
        if not items:
            return None
        key = (st.width, st.height) + tuple(
            (anim.cel_key(c), col.rgba(), op) for c, col, op in items)
        if self._onion_cache is None or self._onion_cache[0] != key:
            out = QImage(st.width, st.height, QImage.Format.Format_ARGB32_Premultiplied)
            out.fill(Qt.GlobalColor.transparent)
            p = QPainter(out)
            for c, col, op in items:
                p.setOpacity(op)
                p.drawImage(0, 0, anim.tinted(c, st.width, st.height, col))
            p.end()
            self._onion_cache = (key, out)
        return self._onion_cache[1]

    def go_to(self, frame: int, folder=None) -> None:
        """frame を今のコマにする。選択中のセルのフォルダ（または folder）で
        そのコマに出ているセルを選び、すぐ描けるようにする。"""
        if self._playing:
            self._stop_play()
        st = self.stack
        folder = folder or self.current_folder()
        anim.apply_frame(st, frame)
        if folder is not None:
            shown = anim.cel_at(folder, st.anim_frame)
            target = shown if shown is not None else folder
            path = st.find_path(target)
            if path:
                st.active_path = path
        self._after_frame_change()

    def next_frame(self) -> None:
        f = self.stack.anim_frame + 1
        if f >= self.stack.anim_length:
            f = 0 if self._loop else f - 1
        self.go_to(f)

    def prev_frame(self) -> None:
        f = self.stack.anim_frame - 1
        if f < 0:
            f = self.stack.anim_length - 1 if self._loop else 0
        self.go_to(f)

    # ── 内部 ──

    def _after_frame_change(self) -> None:
        self._view.update()
        self._update_label()
        self._update_preview()
        self.frame_changed.emit()
        self.onion_skin_changed.emit()

    def _edit(self, fn) -> None:
        """タイムラインだけを変える操作。取り消せるよう前の状態を控えてから行う。"""
        self.anim_will_change.emit()
        fn()
        self._changed()

    def _changed(self) -> None:
        self.sync()
        self.frames_changed.emit()
        self.frame_changed.emit()
        self.onion_skin_changed.emit()

    def _on_new_folder(self):
        self.structure_will_change.emit()
        anim.make_anim_folder(self.stack)
        self._changed()

    def _target_frame_for_new(self, folder) -> int:
        f = self.stack.anim_frame
        if folder.timeline[f] is not None:
            f += 1
        return f

    def _new_cel_at(self, folder, frame: int, source=None) -> None:
        self.structure_will_change.emit()
        anim.new_cel(self.stack, folder, frame, source)
        self._changed()

    def _on_new_cel(self):
        folder = self.current_folder()
        if folder is None:
            self._on_new_folder()
            return
        self._new_cel_at(folder, self._target_frame_for_new(folder))

    def _on_dup_cel(self):
        folder = self.current_folder()
        if folder is None:
            return
        src = anim.cel_at(folder, self.stack.anim_frame)
        if src is None:
            return
        self._new_cel_at(folder, self.stack.anim_frame + 1, src)

    def _renumber(self, folder) -> None:
        self.structure_will_change.emit()
        anim.renumber(self.stack, folder)
        self._changed()

    # ── コマの選択・移動・コピー＆貼り付け ──

    def selection(self) -> tuple[object, int, int] | None:
        return self._sel

    def select(self, folder, start: int, end: int | None = None) -> None:
        end = start if end is None else end
        n = self.stack.anim_length
        start, end = max(0, min(start, end)), min(n - 1, max(start, end))
        self._sel = (folder, start, end)
        self._sel_anchor = (start, end)
        self._view.update()

    def _sel_or_current(self, folder=None, frame=None) -> tuple[object, int, int] | None:
        """選択範囲。無ければ今のコマのセルが続いている範囲。"""
        folder = folder or self.current_folder()
        if folder is None:
            return None
        frame = self.stack.anim_frame if frame is None else frame
        if self._sel is not None and self._sel[0] is folder \
                and self._sel[1] <= frame <= self._sel[2]:
            return self._sel
        s, e = anim.span(folder, frame)
        return (folder, s, e)

    def _clip_in(self, folder) -> bool:
        """コピーしたセルがすべて folder のセルか（同じセルを使い回せるか）。"""
        return self._clip is not None and all(
            c is None or any(c is k for k in folder.children) for c in self._clip)

    def copy_frames(self) -> bool:
        sel = self._sel_or_current()
        if sel is None:
            return False
        folder, s, e = sel
        self._clip = [anim.cel_at(folder, f) for f in range(s, e + 1)]
        return True

    def cut_frames(self) -> None:
        if self.copy_frames():
            self.clear_frames()

    def clear_frames(self) -> None:
        sel = self._sel_or_current()
        if sel is None:
            return
        folder, s, e = sel
        self._edit(lambda: anim.clear_specs(folder, s, e))

    def paste_frames(self, reuse: bool = False) -> None:
        """コピーしたコマを今のコマから貼る。

        ふつうはセルを複製して貼る（貼った先を描き直しても元のセルは変わらない）。
        reuse なら同じセルを指定し直すだけ（同じ絵を何度も出す）。別のフォルダへは
        使い回せないので複製になる。
        """
        folder = self.current_folder()
        if folder is None or not self._clip:
            return
        to = self.stack.anim_frame
        clip = self._clip
        if reuse and self._clip_in(folder):
            values = [c.uid if c is not None else BLANK for c in clip]
            self._edit(lambda: anim.place_specs(self.stack, folder, to, values))
        else:
            from layer_panel import LayerPanel
            self.structure_will_change.emit()
            copies: dict[int, object] = {}
            for c in clip:
                if c is not None and id(c) not in copies:
                    new = LayerPanel._deep_copy_layer(c, self.stack)
                    new.name = anim.next_cel_name(folder)
                    folder.children.insert(0, new)
                    copies[id(c)] = new
            values = [copies[id(c)].uid if c is not None else BLANK for c in clip]
            anim.place_specs(self.stack, folder, to, values)
            self._changed()
        self.select(folder, to, to + len(clip) - 1)
        self.go_to(to, folder)

    def _on_key_command(self, cmd: str) -> None:
        if cmd == "prev":
            self.prev_frame()
        elif cmd == "next":
            self.next_frame()
        elif cmd == "copy":
            self.copy_frames()
        elif cmd == "cut":
            self.cut_frames()
        elif cmd == "paste":
            self.paste_frames()
        elif cmd == "paste_ref":
            self.paste_frames(reuse=True)
        elif cmd == "clear":
            self.clear_frames()

    def _on_cell_clicked(self, folder, frame: int, extend: bool = False):
        if folder is not None:
            s, e = anim.span(folder, frame)
            if extend and self._sel is not None and self._sel[0] is folder \
                    and self._sel_anchor is not None:
                a0, a1 = self._sel_anchor
                self._sel = (folder, min(a0, s), max(a1, e))
            else:
                self.select(folder, s, e)
        else:
            self._sel = self._sel_anchor = None
        self.go_to(frame, folder)

    def _on_cell_double_clicked(self, folder, frame: int):
        if folder.timeline[frame] is None and anim.cel_at(folder, frame) is None:
            self._new_cel_at(folder, frame)

    def _on_cell_dragged(self, folder, to: int, copy: bool = False):
        """選択範囲を to から始まる位置へ動かす（copy なら写す）。"""
        sel = self._sel
        if sel is None or sel[0] is not folder:
            return
        _, s, e = sel
        self._edit(lambda: anim.move_specs(self.stack, folder, s, e, to, copy))
        self.select(folder, to, to + e - s)
        self.go_to(to, folder)

    def _on_context(self, folder, frame: int, pos: QPoint):
        if folder is not None and not (self._sel is not None and self._sel[0] is folder
                                       and self._sel[1] <= frame <= self._sel[2]):
            self.select(folder, *anim.span(folder, frame))
        self.go_to(frame, folder)
        menu = QMenu(self)
        self._fill_edit_menu(menu, folder, frame)
        menu.exec(pos)

    def _on_length_changed(self, val: int):
        self._edit(lambda: anim.set_length(self.stack, val))

    def _on_play_toggle(self):
        if self._playing:
            self._stop_play()
        else:
            self._start_play()

    def _frame_image(self, f: int) -> QImage:
        img = self._play_cache.get(f)
        if img is None:
            img = anim.render_frame(self.stack, f)
            self._play_cache[f] = img
        return img

    def _start_play(self):
        if self.stack.anim_length < 2 or not anim.anim_folders(self.stack.layers):
            return
        self._play_cache.clear()
        self._playing = True
        self._play_frame = self.stack.anim_frame
        self._play_btn.setText("⏹ 停止")
        self._play_timer.start(max(1, 1000 // self._fps_spin.value()))
        self.playback_tick.emit()

    def _stop_play(self):
        was = self._playing
        self._playing = False
        self._play_btn.setText("▶ 再生")
        self._play_timer.stop()
        if was and self._play_frame is not None:
            f, self._play_frame = self._play_frame, None
            self.go_to(f)
        self.playback_tick.emit()

    def _play_next(self):
        n = self.stack.anim_length
        f = (self._play_frame or 0) + 1
        if f >= n:
            if not self._loop:
                self._stop_play()
                return
            f = 0
        self._play_frame = f
        self._view.update()
        self._update_label()
        self._update_preview()
        self.playback_tick.emit()

    def _on_loop_toggle(self, val: bool):
        self._loop = val
        self.onion_skin_changed.emit()

    def _on_fps_changed(self, val: int):
        if self._playing and val > 0:
            self._play_timer.start(max(1, 1000 // val))
        self.frames_changed.emit()

    def _on_preview_toggle(self, val: bool):
        if val:
            if self._preview_win is None:
                self._preview_win = PlaybackPreview()
            self._preview_win.show()
            self._update_preview()
        elif self._preview_win:
            self._preview_win.hide()

    def _update_preview(self):
        if self._preview_win and self._preview_win.isVisible() \
                and anim.anim_folders(self.stack.layers):
            f = self._play_frame if self._playing else self.stack.anim_frame
            img = self._frame_image(f) if self._playing else anim.render_frame(self.stack, f)
            self._preview_win.set_frame(img, f, self.stack.anim_length)

    def _on_onion_toggle(self, val: bool):
        self._onion_enabled = val
        self.onion_skin_changed.emit()

    def _on_onion_before(self, val: int):
        self._onion_before = val
        self.onion_skin_changed.emit()

    def _on_onion_after(self, val: int):
        self._onion_after = val
        self.onion_skin_changed.emit()

    def _all_frames(self) -> list[QImage]:
        return [anim.render_frame(self.stack, f) for f in range(self.stack.anim_length)]

    def _check_exportable(self, title: str) -> bool:
        if not anim.anim_folders(self.stack.layers):
            QMessageBox.information(self, title, "アニメーションフォルダがありません。")
            return False
        return True

    def _on_export_gif(self):
        if not self._check_exportable("GIF書き出し"):
            return
        path, _ = QFileDialog.getSaveFileName(self, "GIF書き出し", "", "GIF (*.gif)")
        if not path:
            return
        if not path.lower().endswith(".gif"):
            path += ".gif"
        self._export_gif_to(path)

    def _export_gif_to(self, path: str) -> bool:
        try:
            from PIL import Image
        except ImportError:
            QMessageBox.warning(self, "エラー",
                                "GIF書き出しには Pillow が必要です。\npip install Pillow")
            return False
        pil_frames = []
        for qimg in self._all_frames():
            img = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
            w, h = img.width(), img.height()
            ptr = img.bits()
            ptr.setsize(h * w * 4)
            pil_frames.append(Image.frombytes("RGBA", (w, h), bytes(ptr)))
        fps = self._fps_spin.value()
        try:
            pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:],
                               duration=1000 // fps, loop=0, disposal=2)
        except (OSError, ValueError) as e:
            QMessageBox.warning(self, "エラー", f"GIFの書き出しに失敗しました:\n{e}")
            return False
        QMessageBox.information(self, "GIF書き出し",
                                f"保存しました: {path}\n"
                                f"{len(pil_frames)} コマ / {fps} fps")
        return True

    def _on_export_png_sequence(self):
        if not self._check_exportable("連番PNG書き出し"):
            return
        folder = QFileDialog.getExistingDirectory(self, "連番PNGの保存先フォルダ")
        if folder:
            self._export_png_sequence_to(folder)

    def _export_png_sequence_to(self, folder: str) -> int:
        frames = self._all_frames()
        digits = max(3, len(str(len(frames))))
        for i, img in enumerate(frames):
            path = os.path.join(folder, f"frame_{i + 1:0{digits}d}.png")
            if not img.save(path, "PNG"):
                QMessageBox.warning(self, "エラー", f"書き出しに失敗しました:\n{path}")
                return i
        QMessageBox.information(self, "連番PNG書き出し",
                                f"{len(frames)} 枚を保存しました: {folder}")
        return len(frames)

    def _update_label(self):
        st = self.stack
        f = self._play_frame if self._playing and self._play_frame is not None else st.anim_frame
        self._frame_label.setText(f"コマ: {f + 1} / {st.anim_length}")
