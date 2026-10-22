"""アニメーションフォルダとタイムラインのデータ操作（GUIなし）。

CLIP STUDIO のアニメーションフォルダと同じ考え方にしている。

- アニメーションフォルダ（timeline を持つ GroupLayer）の直下のレイヤーが
  1枚1枚のセル。フォルダをセルにしてもよい。
- folder.timeline[f] はコマ f のセル指定。セルの uid か、BLANK（何も出さない）か、
  None（指定なし＝前のコマのセルを出し続ける）。同じセルを何コマも出す
  「2コマ打ち」などは指定なしのコマで表す。
- アニメーションフォルダの外のレイヤーは全コマに出る（背景・用紙）。
- 今のコマに出すセル以外はフォルダの中で非表示にする。合成・キャッシュ・
  ストローク中の高速表示は、ふだんの表示/非表示の仕組みをそのまま使える。
"""
from __future__ import annotations

from PyQt6.QtGui import QImage, QPainter, QColor
from PyQt6.QtCore import Qt

from layer import GroupLayer, Layer, _item_buffer

BLANK = 0  # uid は 1 から振られるので 0 は空きコマの印に使える
DEFAULT_LENGTH = 8
MAX_LENGTH = 1000


def is_anim_folder(item) -> bool:
    return item.is_group and getattr(item, "timeline", None) is not None


def anim_folders(items) -> list[GroupLayer]:
    """アニメーションフォルダを上から順に集める（普通のフォルダの中もたどる）。"""
    found = []
    for it in items:
        if it.is_group:
            if is_anim_folder(it):
                found.append(it)
            found.extend(anim_folders(it.children))
    return found


def folder_of(stack, item) -> GroupLayer | None:
    """item を直下のセルとして持つアニメーションフォルダ。"""
    for f in anim_folders(stack.layers):
        if any(c is item for c in f.children):
            return f
    return None


def _child(folder, uid):
    return next((c for c in folder.children if c.uid == uid), None)


def spec_frame(folder, frame: int) -> int | None:
    """frame に出るセルを決めている指定があるコマ。どこにも無ければ None。"""
    tl = folder.timeline
    for f in range(min(frame, len(tl) - 1), -1, -1):
        if tl[f] is not None:
            return f
    return None


def cel_at(folder, frame: int):
    """frame に出すセル。空きコマ・指定前・消したセルなら None。"""
    f = spec_frame(folder, frame)
    if f is None or folder.timeline[f] == BLANK:
        return None
    return _child(folder, folder.timeline[f])


def apply_frame(stack, frame: int) -> None:
    """frame を今のコマにして、各アニメーションフォルダで出すセルだけを表示する。"""
    frame = max(0, min(frame, stack.anim_length - 1))
    stack.anim_frame = frame
    for folder in anim_folders(stack.layers):
        shown = cel_at(folder, frame)
        for c in folder.children:
            c.visible = c is shown


def set_length(stack, n: int) -> None:
    """再生時間（コマ数）を変える。縮めたぶんの指定は消える。"""
    n = max(1, min(int(n), MAX_LENGTH))
    stack.anim_length = n
    for folder in anim_folders(stack.layers):
        tl = folder.timeline
        folder.timeline = (tl + [None] * n)[:n]
    if stack.anim_frame >= n:
        apply_frame(stack, n - 1)


def next_cel_name(folder) -> str:
    nums = [int(c.name) for c in folder.children if c.name.isdigit()]
    return str(max(nums, default=0) + 1)


def make_anim_folder(stack, name: str = "アニメーションフォルダ") -> GroupLayer:
    """今のレイヤーの上にアニメーションフォルダを作り、セル「1」を今のコマに
    置いて選択する。"""
    folder = GroupLayer(name, stack.width, stack.height)
    folder.pass_through = True
    folder.timeline = [None] * stack.anim_length
    stack.insert_above_active(folder)
    cel = Layer("1", stack.width, stack.height)
    folder.children.append(cel)
    folder.timeline[stack.anim_frame] = cel.uid
    stack.active_path = stack.find_path(cel)
    apply_frame(stack, stack.anim_frame)
    return folder


def new_cel(stack, folder, frame: int, source=None) -> Layer:
    """新しいセルを作って frame に指定し、選択する。

    source を渡すとその絵を写したセルにする（前のセルを元に少しだけ動かす中割り用）。
    """
    cel = Layer(next_cel_name(folder), stack.width, stack.height)
    if source is not None and not source.is_group:
        cel.image = source.image.copy()
        cel.offset_x = getattr(source, "offset_x", 0)
        cel.offset_y = getattr(source, "offset_y", 0)
    folder.children.insert(0, cel)
    if frame >= stack.anim_length:
        set_length(stack, frame + 1)
    folder.timeline[frame] = cel.uid
    apply_frame(stack, frame)
    stack.active_path = stack.find_path(cel)
    return cel


def set_spec(folder, frame: int, value) -> None:
    if 0 <= frame < len(folder.timeline):
        folder.timeline[frame] = value


def swap_specs(folder, a: int, b: int) -> None:
    """2つのコマの指定を入れ替える（ドラッグでの並べ替え）。

    指定なしのコマはその前のセルを出しているので、入れ替える前に今
    見えているセルを指定として書き込んでから入れ替える。こうしないと
    入れ替えたあとに前後のコマの見た目まで変わってしまう。
    """
    tl = folder.timeline
    if not (0 <= a < len(tl) and 0 <= b < len(tl)) or a == b:
        return
    for f in (a, b, a + 1, b + 1):
        if 0 <= f < len(tl) and tl[f] is None:
            c = cel_at(folder, f)
            tl[f] = c.uid if c is not None else BLANK
    tl[a], tl[b] = tl[b], tl[a]
    _drop_redundant(folder)


def shown_specs(folder) -> list:
    """コマごとに実際に出ているセルの uid（何も出ないコマは BLANK）。"""
    out = []
    cur = BLANK
    for v in folder.timeline:
        if v is not None:
            cur = v if v == BLANK or _child(folder, v) is not None else BLANK
        out.append(cur)
    return out


def _set_shown(folder, shown: list) -> None:
    folder.timeline = list(shown)
    _drop_redundant(folder)


def span(folder, frame: int) -> tuple[int, int]:
    """frame を含む「同じ指定が続いている」コマの範囲 (start, end)。

    タイムラインでセルのサムネイルと継続の線がひとかたまりに見える範囲で、
    クリックしたときにまとめて選ぶのに使う。
    """
    tl = folder.timeline
    frame = max(0, min(frame, len(tl) - 1))
    s = frame
    while s > 0 and tl[s] is None:
        s -= 1
    e = frame
    while e + 1 < len(tl) and tl[e + 1] is None:
        e += 1
    return s, e


def place_specs(stack, folder, to: int, values: list) -> None:
    """to から順に values（uid か BLANK）のセルを出すよう上書きする。

    後ろのコマの見た目は変えない。足りなければコマ数を延ばす。
    """
    to = max(0, to)
    values = values[:MAX_LENGTH - to]
    if to + len(values) > stack.anim_length:
        set_length(stack, to + len(values))
    shown = shown_specs(folder)
    shown[to:to + len(values)] = values
    _set_shown(folder, shown)


def clear_specs(folder, start: int, end: int) -> None:
    """start〜end のコマを空きにする。後ろのコマの見た目は変えない。"""
    shown = shown_specs(folder)
    for f in range(max(0, start), min(end, len(shown) - 1) + 1):
        shown[f] = BLANK
    _set_shown(folder, shown)


def move_specs(stack, folder, start: int, end: int, to: int, copy: bool = False) -> None:
    """start〜end のコマを to から始まる位置へ動かす（copy なら写す）。

    動かした先にあったコマは上書きし、元の場所は空きコマになる。
    """
    block = shown_specs(folder)[start:end + 1]
    if not copy:
        clear_specs(folder, start, end)
    place_specs(stack, folder, to, block)


def _drop_redundant(folder) -> None:
    """直前と同じ指定を「指定なし」にまとめる（見た目は変わらない）。"""
    tl = folder.timeline
    prev = None
    for f, v in enumerate(tl):
        if v is None:
            continue
        if v == prev or (prev is None and v == BLANK):
            tl[f] = None
        else:
            prev = v


def insert_frame(stack, frame: int) -> None:
    """frame の位置に1コマ挟む。挟んだコマは前のセルを出し続ける。"""
    frame = max(0, min(frame, stack.anim_length))
    for folder in anim_folders(stack.layers):
        folder.timeline.insert(frame, None)
    stack.anim_length = min(stack.anim_length + 1, MAX_LENGTH)
    for folder in anim_folders(stack.layers):
        folder.timeline = folder.timeline[:stack.anim_length]


def delete_frame(stack, frame: int) -> None:
    """frame のコマを抜いて詰める。次のコマが指定なしなら、消えるセルを引き継ぐ。"""
    if stack.anim_length <= 1 or not 0 <= frame < stack.anim_length:
        return
    for folder in anim_folders(stack.layers):
        tl = folder.timeline
        if frame + 1 < len(tl) and tl[frame + 1] is None and tl[frame] is not None:
            tl[frame + 1] = tl[frame]
        tl.pop(frame)
        _drop_redundant(folder)
    stack.anim_length -= 1
    apply_frame(stack, min(stack.anim_frame, stack.anim_length - 1))


def retime(stack, folder, step: int) -> None:
    """セルの並びはそのままで、1枚を step コマずつ出すタイミングに打ち直す。"""
    step = max(1, int(step))
    specs = [v for v in folder.timeline if v is not None]
    need = len(specs) * step
    if need > stack.anim_length:
        set_length(stack, need)
    tl = [None] * stack.anim_length
    for i, v in enumerate(specs):
        tl[i * step] = v
    folder.timeline = tl
    _drop_redundant(folder)


def renumber(stack, folder) -> None:
    """タイムラインに出てくる順にセルを 1, 2, 3… と名前を付け直す。

    出てこないセルはその後ろに続けて番号を振る。レイヤーの並びも
    番号順（上ほど大きい番号）に揃える。
    """
    active = stack.active
    order = []
    for v in folder.timeline:
        c = _child(folder, v) if v not in (None, BLANK) else None
        if c is not None and c not in order:
            order.append(c)
    order += [c for c in reversed(folder.children) if c not in order]
    for i, c in enumerate(order):
        c.name = str(i + 1)
    folder.children[:] = list(reversed(order))
    if active is not None:
        path = stack.find_path(active)
        if path:
            stack.active_path = path


def timelines_state(stack) -> dict:
    """タイムライン編集の取り消し用に、指定と再生時間を控える。"""
    return {"length": stack.anim_length, "frame": stack.anim_frame,
            "timelines": {f.uid: list(f.timeline) for f in anim_folders(stack.layers)}}


def restore_timelines(stack, state: dict) -> None:
    stack.anim_length = state["length"]
    tls = state["timelines"]
    for f in anim_folders(stack.layers):
        if f.uid in tls:
            f.timeline = list(tls[f.uid])
    apply_frame(stack, state["frame"])


def render_frame(stack, frame: int) -> QImage:
    """frame の見た目を合成して返す。今のコマの表示状態は元に戻す。

    画面用の合成キャッシュは1枚だけなので、それを使わずに合成する
    （再生や書き出しで別のコマを作るたびに画面のキャッシュが捨てられないように）。
    """
    current = stack.anim_frame
    apply_frame(stack, frame)
    try:
        return stack._composite_uncached(None)
    finally:
        apply_frame(stack, current)


ONION_PREV = QColor(40, 90, 255)
ONION_NEXT = QColor(30, 170, 60)


def onion_items(stack, frame: int, before: int, after: int,
                loop: bool = False) -> list[tuple[object, QColor, float]]:
    """オニオンスキンに出すセルと色・不透明度。今のコマと同じセルは出さない。"""
    out = []
    n = stack.anim_length
    for folder in anim_folders(stack.layers):
        if not folder.visible:
            continue
        here = cel_at(folder, frame)
        seen = {id(here)}
        for sign, count, color in ((-1, before, ONION_PREV), (1, after, ONION_NEXT)):
            for k in range(1, count + 1):
                f = frame + sign * k
                if loop:
                    f %= n
                elif not 0 <= f < n:
                    break
                c = cel_at(folder, f)
                if c is None or id(c) in seen:
                    continue
                seen.add(id(c))
                out.append((c, color, 0.45 / k))
    return out


def tinted(cel, w: int, h: int, color: QColor) -> QImage:
    """セルの絵を color 1色に塗った画像（オニオンスキン用）。"""
    buf = _item_buffer(cel, w, h, 0, 0)
    p = QPainter(buf)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    p.fillRect(buf.rect(), color)
    p.end()
    return buf


def cel_key(cel):
    """セルの見た目が変わったかを見るためのキー。"""
    if cel.is_group:
        from layer import LayerStack
        return ("g", cel.uid, LayerStack._render_key(cel.children))
    return ("l", cel.uid, cel._effect_key(),
            getattr(cel, "offset_x", 0), getattr(cel, "offset_y", 0))


def migrate_frames(stack, frames: list[QImage]) -> GroupLayer | None:
    """旧形式（コマごとに合成済みの絵）をアニメーションフォルダに変換する。"""
    if not frames:
        return None
    folder = GroupLayer("アニメーション（旧形式）", stack.width, stack.height)
    folder.pass_through = True
    length = max(stack.anim_length, len(frames))
    folder.timeline = [None] * length
    for i, img in enumerate(frames):
        cel = Layer(str(i + 1), stack.width, stack.height)
        buf = QImage(stack.width, stack.height, QImage.Format.Format_ARGB32_Premultiplied)
        buf.fill(Qt.GlobalColor.transparent)
        p = QPainter(buf)
        p.drawImage(0, 0, img)
        p.end()
        cel.image = buf.convertToFormat(QImage.Format.Format_ARGB32)
        folder.children.insert(0, cel)
        folder.timeline[i] = cel.uid
    stack.layers.insert(0, folder)
    if stack.active_path:
        stack.active_path = [stack.active_path[0] + 1] + list(stack.active_path[1:])
    set_length(stack, length)
    return folder
