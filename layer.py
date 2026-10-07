from __future__ import annotations
import itertools
import numpy as np
import cv2
from PyQt6.QtGui import QImage, QPainter, QColor
from PyQt6.QtCore import Qt

CANVAS_W = 2500
CANVAS_H = 2500

# ブレンドモード定義 (key, 表示名, QPainter.CompositionMode)
BLEND_MODES: list[tuple[str, str, QPainter.CompositionMode | None]] = [
    ("normal",     "通常",         None),
    ("multiply",   "乗算",         QPainter.CompositionMode.CompositionMode_Multiply),
    ("screen",     "スクリーン",   QPainter.CompositionMode.CompositionMode_Screen),
    ("overlay",    "オーバーレイ", QPainter.CompositionMode.CompositionMode_Overlay),
    ("plus",       "加算",         QPainter.CompositionMode.CompositionMode_Plus),
]

BLEND_KEY_TO_MODE: dict[str, QPainter.CompositionMode | None] = {
    k: m for k, _, m in BLEND_MODES
}
BLEND_KEYS: list[str] = [k for k, _, _ in BLEND_MODES]
BLEND_LABELS: dict[str, str] = {k: label for k, label, _ in BLEND_MODES}

# 履歴がレイヤーを指すための通し番号。id() はレイヤーを作り直すと
# 変わる（構造の undo で全レイヤーが作り直される）うえ、捨てたレイヤーの
# 番号が別のレイヤーに再利用されることもあるので使えない。
_UIDS = itertools.count(1)


def new_uid() -> int:
    return next(_UIDS)


def _premul_buffer(w: int, h: int) -> QImage:
    img = QImage(max(1, w), max(1, h), QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.transparent)
    return img


def _item_buffer(item, w: int, h: int, off_x: int, off_y: int, skip=None) -> QImage:
    """item 1枚ぶんの絵を、target と同じ大きさ・位置合わせのバッファに描いて返す。

    グループは中身だけを合成する（グループ自身の不透明度は掛けない）。
    クリッピングのマスクと、クリッピングする側の絵の両方に使う。
    バッファは target 全体を覆うので、item の画像の外側は透明として扱われる。
    """
    buf = _premul_buffer(w, h)
    if item.is_group:
        render_items(buf, item.children, off_x, off_y, skip)
    else:
        p = QPainter(buf)
        p.drawImage(getattr(item, 'offset_x', 0) - off_x,
                    getattr(item, 'offset_y', 0) - off_y,
                    item.image_with_effects())
        p.end()
    return buf


def _lerp_into(target: QImage, before: QImage, t: float) -> None:
    """target = before + (target - before) * t。通過グループの不透明度用。"""
    w, h = target.width(), target.height()
    a = _as_premul_array(before, w, h)
    b = _as_premul_array(target, w, h)
    out = (a.astype(np.float32) * (1.0 - t) + b.astype(np.float32) * t)
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    img = QImage(out.tobytes(), w, h, w * 4,
                 QImage.Format.Format_ARGB32_Premultiplied).copy()
    p = QPainter(target)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
    p.drawImage(0, 0, img)
    p.end()


def _as_premul_array(img: QImage, w: int, h: int) -> np.ndarray:
    src = img.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    ptr = src.constBits()
    ptr.setsize(h * w * 4)
    return np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()


def _draw_item(target: QImage, item, off_x: int, off_y: int, skip=None) -> None:
    """クリッピングでない item を target へ合成する。"""
    if item.is_group:
        if getattr(item, 'pass_through', False):
            # 通過: 中のレイヤーの合成モードがフォルダの下にも効く（CLIP STUDIO の既定）。
            if item.opacity >= 255:
                render_items(target, item.children, off_x, off_y, skip)
            else:
                before = target.copy()
                render_items(target, item.children, off_x, off_y, skip)
                _lerp_into(target, before, item.opacity / 255)
            return
        buf = _item_buffer(item, target.width(), target.height(), off_x, off_y, skip)
        p = QPainter(target)
        p.setOpacity(item.opacity / 255)
        p.drawImage(0, 0, buf)
        p.end()
        return
    p = QPainter(target)
    blend = BLEND_KEY_TO_MODE.get(getattr(item, 'blend_mode', 'normal'))
    if blend:
        p.setCompositionMode(blend)
    p.setOpacity(item.opacity / 255)
    p.drawImage(getattr(item, 'offset_x', 0) - off_x,
                getattr(item, 'offset_y', 0) - off_y, item.image_with_effects())
    p.end()


def render_items(target: QImage, items, off_x: int = 0, off_y: int = 0,
                 skip=None) -> None:
    """レイヤー列（上が先頭）を target に合成する。合成処理の本体はここだけ。

    target の (0,0) はキャンバス座標 (off_x, off_y) にあたる。画面表示・
    グループ・統合のすべてがここを通るので、見た目と統合結果が食い違わない。

    クリッピングは CLIP STUDIO と同じく「すぐ下のクリッピングでないレイヤー」
    （下地）に対して効く。クリッピングを何枚重ねても全部が同じ下地で切り抜かれ、
    下地を非表示にするとクリッピングしたレイヤーも消える。クリッピングした
    レイヤーの合成モードも有効。

    skip に渡したレイヤーは描かない（下地としてのマスクには使う）。
    """
    w, h = target.width(), target.height()
    i = len(items) - 1
    while i >= 0:
        base = items[i]
        # base の上に乗っているクリッピングレイヤーを集める（下から順）
        j = i - 1
        clips = []
        while j >= 0 and items[j].clipping:
            clips.append(items[j])
            j -= 1
        if base.visible:
            if base is not skip:
                _draw_item(target, base, off_x, off_y, skip)
            shown = [c for c in clips if c.visible and c is not skip]
            if shown:
                mask = _item_buffer(base, w, h, off_x, off_y, skip)
                for clip in shown:
                    src = _premul_buffer(w, h)
                    sp = QPainter(src)
                    sp.drawImage(0, 0, _item_buffer(clip, w, h, off_x, off_y, skip))
                    sp.setCompositionMode(
                        QPainter.CompositionMode.CompositionMode_DestinationIn)
                    sp.drawImage(0, 0, mask)
                    sp.end()
                    p = QPainter(target)
                    blend = (None if clip.is_group else
                             BLEND_KEY_TO_MODE.get(getattr(clip, 'blend_mode', 'normal')))
                    if blend:
                        p.setCompositionMode(blend)
                    p.setOpacity(clip.opacity / 255)
                    p.drawImage(0, 0, src)
                    p.end()
        i = j


class Layer:
    def __init__(self, name: str, w: int = CANVAS_W, h: int = CANVAS_H):
        self.uid = new_uid()
        self.name = name
        self.visible = True
        self.opacity = 255
        self.clipping = False
        self.reference = False
        # ロック。つけると描画・消しゴム・塗りつぶし・移動・変形から守られる。
        self.locked = False
        # レイヤーパネルの「統合対象」チェック。ファイルには保存しない一時的な印。
        self.merge_marked = False
        self.offset_x: int = 0
        self.offset_y: int = 0
        self.image = QImage(w, h, QImage.Format.Format_ARGB32)
        self.image.fill(Qt.GlobalColor.transparent)
        # ブレンドモード
        self.blend_mode: str = "normal"
        # レイヤー効果: 縁取り
        self.border_enabled: bool = False
        self.border_size: int = 3
        self.border_color: QColor = QColor(0, 0, 0, 255)
        # レイヤー効果: ドロップシャドウ
        self.shadow_enabled: bool = False
        self.shadow_color: QColor = QColor(0, 0, 0, 180)
        self.shadow_offset_x: int = 4
        self.shadow_offset_y: int = 4
        self.shadow_blur: int = 5
        self.shadow_strength: int = 100  # 0-100%
        # レイヤー効果: 光彩（外側グロー）
        self.glow_enabled: bool = False
        self.glow_color: QColor = QColor(255, 255, 200, 255)
        self.glow_size: int = 8
        self.glow_strength: int = 80  # 0-100%
        # レイヤー効果: ガウシアンぼかし
        self.blur_enabled: bool = False
        self.blur_radius: int = 3
        self.blur_strength: int = 100  # 0-100%
        # レイヤー効果: 色調補正
        self.hsl_enabled: bool = False
        self.hsl_hue: int = 0        # -180 ~ +180
        self.hsl_saturation: int = 0  # -100 ~ +100
        self.hsl_lightness: int = 0   # -100 ~ +100
        # 効果適用済み画像のキャッシュ（_effect_key, QImage）
        self._effect_cache: tuple[tuple, QImage] | None = None

    def _effect_key(self) -> tuple:
        """効果の見た目を決める値をすべて集めたキー。

        画像自体の変更は QImage.cacheKey() で拾う。これは QPainter 経由でも
        numpy で bits() を直接書き換えても変わるので、描画方法によらず
        取りこぼさない（レイヤーの全変更箇所を追いかけるより確実）。
        キーに入れ忘れた項目は「変えても画面に反映されない」不具合になるため、
        効果パラメータを追加したときはここにも必ず足すこと。
        """
        return (
            self.image.cacheKey(),
            self.border_enabled, self.border_size, self.border_color.rgba(),
            self.shadow_enabled, self.shadow_color.rgba(), self.shadow_offset_x,
            self.shadow_offset_y, self.shadow_blur, self.shadow_strength,
            self.glow_enabled, self.glow_color.rgba(), self.glow_size,
            self.glow_strength,
            self.blur_enabled, self.blur_radius, self.blur_strength,
            self.hsl_enabled, self.hsl_hue, self.hsl_saturation,
            self.hsl_lightness,
        )

    def clear(self):
        self.image.fill(Qt.GlobalColor.transparent)

    def image_with_border(self) -> QImage:
        if not self.border_enabled or self.border_size <= 0:
            return self.image
        img = self.image
        w, h = img.width(), img.height()
        # 読むだけなので constBits を使う。bits() は書き込み用でデタッチが
        # 走り、QImage.cacheKey() が変わってしまう。cacheKey は効果キャッシュ
        # の判定に使っているので、ここで変わると毎回キャッシュが外れる。
        ptr = img.constBits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()
        alpha = arr[:, :, 3]
        ksize = self.border_size * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        # 線画の内部の線1本1本にも縁がにじまないよう、まず不透明部分全体を
        # 「塗りつぶしたシルエット」（穴埋め済みの外形のみ）にしてから、その
        # シルエットを dilate する。これでイラスト全体の最外周にだけ縁が付く。
        opaque = (alpha > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(opaque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        silhouette = np.zeros_like(opaque)
        if contours:
            cv2.drawContours(silhouette, contours, -1, 255, thickness=cv2.FILLED)
        dilated = cv2.dilate(silhouette, kernel)
        # シルエットの外側にはみ出た dilate 領域だけを縁色で塗る。シルエット内部
        # （線画のアンチエイリアシング縁を含む）は縁色を敷かず、元画像のみを使う。
        border_area = (dilated > 0) & (silhouette == 0)
        border = np.zeros_like(arr)
        bc = self.border_color
        border[border_area] = [bc.blue(), bc.green(), bc.red(), bc.alpha()]
        border_img = QImage(border.tobytes(), w, h, w * 4, QImage.Format.Format_ARGB32).copy()
        out = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        out.fill(Qt.GlobalColor.transparent)
        p = QPainter(out)
        p.drawImage(0, 0, border_img)
        p.drawImage(0, 0, img)
        p.end()
        return out.convertToFormat(QImage.Format.Format_ARGB32)

    def image_with_effects(self) -> QImage:
        """全レイヤー効果を適用した画像を返す。

        結果はキャッシュする。再描画のたびに縁取りやぼかしを計算し直すと、
        レイヤーが増えるほど描画が重くなる（30枚・縁取りONで1回4秒超）。
        """
        key = self._effect_key()
        if self._effect_cache is not None and self._effect_cache[0] == key:
            return self._effect_cache[1]
        img = self._compute_effects()
        # 効果が何も有効でないときは self.image がそのまま返る。これを
        # キャッシュに持つと、レイヤー画像への描き込みで中身が変わり続ける
        # 別名参照になるので持たない（元々この経路は計算コストも無い）。
        if img is not self.image:
            self._effect_cache = (key, img)
        return img

    def _compute_effects(self) -> QImage:
        """効果を実際に適用する（キャッシュ無しの本体）。"""
        img = self.image_with_border()
        w, h = img.width(), img.height()

        # ドロップシャドウ
        if self.shadow_enabled and self.shadow_strength > 0:
            img = self._apply_shadow(img, w, h)

        # 光彩（外側グロー）
        if self.glow_enabled and self.glow_strength > 0 and self.glow_size > 0:
            img = self._apply_glow(img, w, h)

        # ガウシアンぼかし
        if self.blur_enabled and self.blur_radius > 0 and self.blur_strength > 0:
            img = self._apply_blur(img, w, h)

        # 色調補正
        if self.hsl_enabled and (self.hsl_hue != 0 or self.hsl_saturation != 0 or self.hsl_lightness != 0):
            img = self._apply_hsl(img, w, h)

        return img

    def _qimage_to_array(self, img: QImage) -> np.ndarray:
        """QImage を numpy 配列にコピーして返す（読み取り専用）。

        すぐ copy() するので constBits で十分。bits() だとデタッチで
        cacheKey が変わり、効果キャッシュが毎回外れる。
        """
        w, h = img.width(), img.height()
        ptr = img.constBits()
        ptr.setsize(h * w * 4)
        return np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()

    def _array_to_qimage(self, arr: np.ndarray, w: int, h: int) -> QImage:
        return QImage(arr.tobytes(), w, h, w * 4, QImage.Format.Format_ARGB32).copy()

    def _apply_shadow(self, img: QImage, w: int, h: int) -> QImage:
        arr = self._qimage_to_array(img)
        alpha = arr[:, :, 3].astype(np.float32)
        ksize = max(self.shadow_blur * 2 + 1, 1)
        blurred = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
        sc = self.shadow_color
        strength = self.shadow_strength / 100.0
        shadow = np.zeros((h, w, 4), dtype=np.uint8)
        shadow[:, :, 0] = sc.blue()
        shadow[:, :, 1] = sc.green()
        shadow[:, :, 2] = sc.red()
        shadow[:, :, 3] = np.clip(blurred * strength * (sc.alpha() / 255.0), 0, 255).astype(np.uint8)
        # offset
        M = np.float32([[1, 0, self.shadow_offset_x], [0, 1, self.shadow_offset_y]])
        shadow = cv2.warpAffine(shadow, M, (w, h))
        shadow_img = self._array_to_qimage(shadow, w, h)
        out = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        out.fill(Qt.GlobalColor.transparent)
        p = QPainter(out)
        p.drawImage(0, 0, shadow_img)
        p.drawImage(0, 0, img)
        p.end()
        return out.convertToFormat(QImage.Format.Format_ARGB32)

    def _apply_glow(self, img: QImage, w: int, h: int) -> QImage:
        arr = self._qimage_to_array(img)
        alpha = arr[:, :, 3].astype(np.float32)
        ksize = max(self.glow_size * 2 + 1, 3)
        blurred = cv2.GaussianBlur(alpha, (ksize, ksize), 0)
        gc = self.glow_color
        strength = self.glow_strength / 100.0
        glow = np.zeros((h, w, 4), dtype=np.uint8)
        glow[:, :, 0] = gc.blue()
        glow[:, :, 1] = gc.green()
        glow[:, :, 2] = gc.red()
        glow[:, :, 3] = np.clip(blurred * strength * (gc.alpha() / 255.0), 0, 255).astype(np.uint8)
        glow_img = self._array_to_qimage(glow, w, h)
        out = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        out.fill(Qt.GlobalColor.transparent)
        p = QPainter(out)
        p.drawImage(0, 0, glow_img)
        p.drawImage(0, 0, img)
        p.end()
        return out.convertToFormat(QImage.Format.Format_ARGB32)

    def _apply_blur(self, img: QImage, w: int, h: int) -> QImage:
        arr = self._qimage_to_array(img)
        ksize = max(self.blur_radius * 2 + 1, 3)
        blurred = cv2.GaussianBlur(arr, (ksize, ksize), 0)
        strength = self.blur_strength / 100.0
        if strength < 1.0:
            blended = (arr.astype(np.float32) * (1 - strength) + blurred.astype(np.float32) * strength)
            blurred = np.clip(blended, 0, 255).astype(np.uint8)
        return self._array_to_qimage(blurred, w, h)

    def _apply_hsl(self, img: QImage, w: int, h: int) -> QImage:
        arr = self._qimage_to_array(img)
        alpha = arr[:, :, 3].copy()
        bgr = arr[:, :, :3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + self.hsl_hue / 2.0) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + self.hsl_saturation * 2.55, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] + self.hsl_lightness * 2.55, 0, 255)
        bgr_out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        result = np.dstack([bgr_out, alpha])
        return self._array_to_qimage(result, w, h)

    def rasterize(self) -> None:
        """レイヤー効果（縁取り・ドロップシャドウ等）を画像に焼き込み、効果設定を無効化する。
        以後、統合しても効果が消えたり二重適用されたりしない。"""
        self.image = self.image_with_effects().convertToFormat(QImage.Format.Format_ARGB32)
        self.border_enabled = False
        self.shadow_enabled = False
        self.glow_enabled = False
        self.blur_enabled = False
        self.hsl_enabled = False

    @property
    def is_group(self) -> bool:
        return False

    @property
    def is_vector(self) -> bool:
        """ベクターレイヤーかどうか。is_group と同じように分岐に使う。"""
        return False


class GroupLayer:
    def __init__(self, name: str, w: int = CANVAS_W, h: int = CANVAS_H):
        self.uid = new_uid()
        self.name = name
        # 通過: 中のレイヤーの合成モードをフォルダの下にも効かせる。
        # 既定は False（フォルダ内だけで合成）。自動生成のフォルダ（線画ずらし等）は
        # 中だけで見た目が完結するよう作ってあるため。パネルから作るフォルダは
        # CLIP STUDIO に合わせて True にする。
        self.pass_through = False
        self.visible = True
        self.opacity = 255
        self.clipping = False
        self.reference = False
        # グループのロックは中のレイヤー全部に効く。
        self.locked = False
        self.collapsed = False  # True のとき子レイヤーをパネルで非表示
        self.children: list[Layer | GroupLayer] = []
        self._w = w
        self._h = h

    @property
    def is_group(self) -> bool:
        return True

    @property
    def is_vector(self) -> bool:
        return False

    def resize(self, w: int, h: int):
        self._w = w
        self._h = h
        for child in self.children:
            if child.is_group:
                child.resize(w, h)  # type: ignore
            elif child.is_vector:
                # ベクターは絵を差し替えても線から描き直されて元に戻るので、
                # 新しいキャンバスの大きさを教えて描き直させる。
                child.set_canvas_size(w, h)  # type: ignore
            else:
                # 通常レイヤーの子も新サイズの画像に差し替える（crop モード）
                new_img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
                new_img.fill(Qt.GlobalColor.transparent)
                p = QPainter(new_img)
                p.drawImage(0, 0, child.image)  # type: ignore
                p.end()
                child.image = new_img.convertToFormat(QImage.Format.Format_ARGB32)  # type: ignore

    def composite(self, skip: object = None) -> QImage:
        """グループ内を合成する。

        skip に渡したレイヤーは除外する。バケツ塗りの参照画像を作るとき、
        塗る対象のレイヤーがこのグループの中にあると、自分の塗った色まで
        境界になってしまうため。
        """
        result = _premul_buffer(self._w, self._h)
        render_items(result, self.children, 0, 0, skip)
        return result.convertToFormat(QImage.Format.Format_ARGB32)


class LayerStack:
    def __init__(self, w: int = CANVAS_W, h: int = CANVAS_H):
        self.layers: list[Layer | GroupLayer] = []
        self.active_path: list[int] = [0]
        self.width = w
        self.height = h

    # ── 後方互換プロパティ ──
    @property
    def active_index(self) -> int:
        return self.active_path[0] if self.active_path else 0

    @active_index.setter
    def active_index(self, value: int):
        if self.active_path:
            self.active_path[0] = value
        else:
            self.active_path = [value]

    @property
    def active_child_index(self) -> int:
        return self.active_path[1] if len(self.active_path) > 1 else -1

    @active_child_index.setter
    def active_child_index(self, value: int):
        if value < 0:
            self.active_path = self.active_path[:1]
        else:
            if len(self.active_path) < 2:
                self.active_path.append(value)
            else:
                self.active_path[1] = value
                self.active_path = self.active_path[:2]

    def _resolve_path(self, path: list[int] | None = None) -> Layer | GroupLayer | None:
        """パスをたどってレイヤーを返す。"""
        if path is None:
            path = self.active_path
        node: list[Layer | GroupLayer] = self.layers
        result: Layer | GroupLayer | None = None
        for idx in path:
            if idx < 0 or idx >= len(node):
                return result
            result = node[idx]
            if result.is_group:
                node = result.children  # type: ignore
            else:
                break
        return result

    @property
    def active(self) -> Layer | GroupLayer | None:
        if not self.layers:
            return None
        return self._resolve_path()

    @property
    def active_top(self) -> Layer | GroupLayer | None:
        """トップレベルのアクティブレイヤー（グループの場合はグループ自身）。"""
        if self.layers and self.active_path:
            idx = self.active_path[0]
            if 0 <= idx < len(self.layers):
                return self.layers[idx]
        return None

    def set_active(self, top_idx: int, child_idx: int = -1):
        if 0 <= top_idx < len(self.layers):
            if child_idx >= 0:
                self.active_path = [top_idx, child_idx]
            else:
                self.active_path = [top_idx]

    def set_active_path(self, path: list[int]):
        """任意の深さのパスでアクティブレイヤーを設定する。"""
        if path and 0 <= path[0] < len(self.layers):
            self.active_path = list(path)

    def find_path(self, target: Layer | GroupLayer) -> list[int] | None:
        """レイヤーオブジェクトのパスを再帰的に探す。"""
        def _search(items: list[Layer | GroupLayer], prefix: list[int]) -> list[int] | None:
            for i, item in enumerate(items):
                p = prefix + [i]
                if item is target:
                    return p
                if item.is_group:
                    found = _search(item.children, p)  # type: ignore
                    if found is not None:
                        return found
            return None
        return _search(self.layers, [])

    def parent_of(self, path: list[int]) -> tuple[list[Layer | GroupLayer], list[int]]:
        """パスの親コンテナとその親パスを返す。"""
        container: list[Layer | GroupLayer] = self.layers
        for idx in path[:-1]:
            if 0 <= idx < len(container) and container[idx].is_group:
                container = container[idx].children  # type: ignore
            else:
                break
        return container, path[:-1]

    @property
    def reference(self) -> Layer | None:
        """後方互換用。複数ある場合は最初の1枚を返す。"""
        refs = self.references
        return refs[0] if refs else None

    @property
    def references(self) -> list:
        return self.references_excluding(None)

    def references_excluding(self, skip: object) -> list:
        """参照フラグが立っている表示中のレイヤー（通常・グループ）を全て返す。
        グループは composite() で合成した画像を持つ疑似オブジェクトとして扱う。
        戻り値は .image を持つオブジェクトのリスト。

        skip に渡したレイヤーは、グループの合成結果からも取り除く。
        バケツ塗りで「塗る対象のレイヤー」を渡すために使う。参照にした
        グループの中に塗る対象が入っていると、自分の塗った色まで境界に
        なってしまい、「参照レイヤーのみ」を選んでも塗り漏れが出るため。
        """
        class _RefProxy:
            def __init__(self, img, opacity, offset_x=0, offset_y=0):
                self.image = img
                self.opacity = opacity
                self.offset_x = offset_x
                self.offset_y = offset_y

        result = []
        def _collect(items: list[Layer | GroupLayer]):
            for layer in items:
                if not layer.visible or layer is skip:
                    continue
                if layer.reference:
                    if layer.is_group:
                        # composite() はキャンバス全体サイズでオフセット 0,0 の画像を返す
                        result.append(_RefProxy(layer.composite(skip), layer.opacity, 0, 0))  # type: ignore
                    else:
                        result.append(layer)
                elif layer.is_group:
                    _collect(layer.children)  # type: ignore
        _collect(self.layers)
        return result

    def add(self, name: str | None = None) -> Layer:
        name = name or f"レイヤー {len(self.layers) + 1}"
        layer = Layer(name, self.width, self.height)
        idx = self.active_index if self.layers else 0
        self.layers.insert(idx, layer)
        self.active_index = idx  # 挿入後に新レイヤーを選択状態にする
        return layer

    def insert_above_active(self, layer) -> None:
        """今のレイヤーと同じ階層の、すぐ上に差し込んで選択する。

        フォルダの中で作業しているときにトップへ飛ばされないようにする。
        """
        path = list(self.active_path) or [0]
        container, parent_path = self.parent_of(path)
        at = min(path[-1], len(container))
        container.insert(at, layer)
        self.active_path = parent_path + [at]

    def add_group(self, name: str | None = None) -> GroupLayer:
        name = name or f"グループ {len(self.layers) + 1}"
        group = GroupLayer(name, self.width, self.height)
        idx = self.active_index if self.layers else 0
        self.layers.insert(idx, group)
        self.active_index = idx  # 挿入後に新グループを選択状態にする
        return group

    def remove(self, index: int):
        if len(self.layers) <= 1:
            return
        self.layers.pop(index)
        self.active_index = max(0, min(self.active_index, len(self.layers) - 1))

    def move(self, from_idx: int, to_idx: int):
        if 0 <= from_idx < len(self.layers) and 0 <= to_idx < len(self.layers):
            layer = self.layers.pop(from_idx)
            self.layers.insert(to_idx, layer)
            self.active_index = to_idx

    def merge_down(self) -> bool:
        """アクティブレイヤーを1つ下のレイヤーに統合する。成功すれば True を返す。
        グループレイヤーは統合対象外。"""
        path = self.active_path
        if not path:
            return False
        container, parent_path = self.parent_of(path)
        idx = path[-1]
        if idx >= len(container) - 1:
            return False
        upper = container[idx]
        lower = container[idx + 1]
        if upper.is_group or lower.is_group:
            return False
        if lower.is_vector:
            # ベクターのまま絵を入れると、次の描き直しで線だけに戻り
            # 統合したラスター部分が消える。先に同じ見た目のラスターへ置き換える。
            lower = lower.to_raster()  # type: ignore
            container[idx + 1] = lower

        # 両レイヤーのオフセット+画像サイズから統合に必要な範囲を計算
        u_ox, u_oy = getattr(upper, 'offset_x', 0), getattr(upper, 'offset_y', 0)
        l_ox, l_oy = getattr(lower, 'offset_x', 0), getattr(lower, 'offset_y', 0)
        min_x = min(u_ox, l_ox)
        min_y = min(u_oy, l_oy)
        max_x = max(u_ox + upper.image.width(), l_ox + lower.image.width())
        max_y = max(u_oy + upper.image.height(), l_oy + lower.image.height())
        mw = max(max_x - min_x, 1)
        mh = max(max_y - min_y, 1)

        merged = QImage(mw, mh, QImage.Format.Format_ARGB32_Premultiplied)
        merged.fill(Qt.GlobalColor.transparent)
        p = QPainter(merged)
        # 下が非表示でも絵は残す。統合先の絵が黙って消えるのを防ぐ。
        p.setOpacity(lower.opacity / 255)
        p.drawImage(l_ox - min_x, l_oy - min_y, lower.image_with_effects())
        upper_img = upper.image_with_effects()
        u_dx, u_dy = u_ox - min_x, u_oy - min_y
        if upper.clipping:
            # 上がクリッピング（マスク）中なら、下レイヤーの不透明部分だけに
            # 焼き込む。そうしないと、下の一部だけ色を変えていたべた塗りが
            # 統合で全面に広がってしまう。クリスタ・Photoshop 等と同じ挙動。
            # ここでの「下」は統合先そのものなので、下の α でマスクする。
            # 合成モードと併用できるよう、描画時ではなく先に上の画像を
            # 切り抜いておく（QPainter は合成モードを1つしか持てないため）。
            masked = QImage(mw, mh, QImage.Format.Format_ARGB32_Premultiplied)
            masked.fill(Qt.GlobalColor.transparent)
            mp = QPainter(masked)
            mp.drawImage(u_dx, u_dy, upper_img)
            mp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
            mp.drawImage(l_ox - min_x, l_oy - min_y, lower.image_with_effects())
            mp.end()
            upper_img = masked
            u_dx = u_dy = 0
        # 非表示レイヤーは画面に出ていないので統合結果にも入れない。
        # 入れてしまうと、消したつもりの絵が統合で復活して見える。
        if upper.visible:
            p.setOpacity(upper.opacity / 255)
            blend = BLEND_KEY_TO_MODE.get(getattr(upper, 'blend_mode', 'normal'))
            if blend:
                p.setCompositionMode(blend)
            p.drawImage(u_dx, u_dy, upper_img)
        p.end()
        lower.image = merged.convertToFormat(QImage.Format.Format_ARGB32)
        lower.opacity = 255
        # 両方とも非表示だったときだけ非表示のまま。どちらかが見えていた
        # なら、その絵が結果に入っているので表示にする。
        lower.visible = lower.visible or upper.visible
        lower.offset_x = min_x
        lower.offset_y = min_y
        # 統合後は効果を焼き込み済みなので、旧設定が残って二重適用されないようリセットする
        lower.border_enabled = False
        lower.shadow_enabled = False
        lower.glow_enabled = False
        lower.blur_enabled = False
        lower.hsl_enabled = False

        container.pop(idx)
        # 上のレイヤーを抜いたので、統合先は idx の位置に繰り上がっている。
        # idx - 1 にすると統合結果ではなく、その上の別レイヤーが選ばれてしまう。
        self.active_path = parent_path + [idx]
        return True

    def merge_marked(self, targets: list) -> bool:
        """指定したレイヤーだけを1枚に統合する。成功すれば True。

        条件: 2枚以上あり、すべて同じ親の中で連続していて、グループを含まないこと。
        （離れたレイヤーを統合すると間に挟まれたレイヤーの重なり順が壊れるため、
        他のお絵かきソフトと同じく連続したものだけを対象にする。）

        統合は下から順に merge_down を繰り返すだけなので、クリッピングの
        マスク処理や効果の焼き込みはそのまま引き継がれる。参照レイヤーに
        クリッピングして一部だけ色を変えている場合も、その見た目が保たれる。
        """
        if len(targets) < 2:
            return False
        if any(t.is_group for t in targets):
            return False

        # 全員が同じ親の中にいて、連続しているか確認する
        paths = [self.path_of(t) for t in targets]
        if any(p is None for p in paths):
            return False
        parents = {tuple(p[:-1]) for p in paths}  # type: ignore
        if len(parents) != 1:
            return False
        idxs = sorted(p[-1] for p in paths)  # type: ignore
        if idxs != list(range(idxs[0], idxs[0] + len(idxs))):
            return False

        parent_path = list(parents.pop())
        # 上から順に「下に統合」を繰り返す。1回統合するたびに下側へ詰まるので、
        # 常に同じ添字（一番上の対象）を指定すればよい。
        top = idxs[0]
        for _ in range(len(idxs) - 1):
            self.active_path = parent_path + [top]
            if not self.merge_down():
                return False
        self.active_path = parent_path + [top]
        return True

    def path_of(self, target) -> list | None:
        """レイヤーの位置（インデックスの並び）を返す。見つからなければ None。"""
        def _walk(items: list, prefix: list) -> list | None:
            for i, lyr in enumerate(items):
                if lyr is target:
                    return prefix + [i]
                if lyr.is_group:
                    found = _walk(lyr.children, prefix + [i])  # type: ignore
                    if found is not None:
                        return found
            return None
        return _walk(self.layers, [])

    def merge_all_visible(self) -> bool:
        """表示中のレイヤーを1枚に統合する。非表示レイヤーは破棄せず下に残す。
        画面外にはみ出た部分も保持する。"""
        if not self.layers:
            return False

        # 実際に画面に出ているものだけを統合する。非表示の下地に乗った
        # クリッピングは見えていないので、下地と一緒に残す。
        shown: list = []
        items = self.layers
        i = len(items) - 1
        while i >= 0:
            base = items[i]
            j = i - 1
            while j >= 0 and items[j].clipping:
                if base.visible and items[j].visible:
                    shown.append(items[j])
                j -= 1
            if base.visible:
                shown.append(base)
            i = j
        if not shown:
            return False
        min_x, min_y, mw, mh = self._visible_bounds(shown)

        merged = QImage(mw, mh, QImage.Format.Format_ARGB32_Premultiplied)
        merged.fill(Qt.GlobalColor.transparent)
        render_items(merged, self.layers, min_x, min_y)

        new_layer = Layer("統合レイヤー", mw, mh)
        new_layer.image = merged.convertToFormat(QImage.Format.Format_ARGB32)
        new_layer.offset_x = min_x
        new_layer.offset_y = min_y
        # 残すレイヤーは元の並び順のまま。統合結果は統合した中で一番下の位置に置く。
        lowest = max(self.layers.index(l) for l in shown)
        merged_ids = {id(l) for l in shown}
        new_layers = []
        for k, lyr in enumerate(self.layers):
            if k == lowest:
                new_layers.append(new_layer)
            elif id(lyr) not in merged_ids:
                new_layers.append(lyr)
        self.layers = new_layers
        self.active_path = [self.layers.index(new_layer)]
        return True

    def _visible_bounds(self, layers) -> tuple[int, int, int, int]:
        """レイヤーリストの全体バウンディングボックスを返す (min_x, min_y, w, h)。"""
        min_x = min_y = 0
        max_x = self.width
        max_y = self.height
        for lyr in layers:
            if lyr.is_group:
                vals = [min_x, min_y, max_x, max_y]
                self._expand_bounds_group_accum(lyr, vals)
                min_x, min_y, max_x, max_y = vals
            else:
                ox = getattr(lyr, 'offset_x', 0)
                oy = getattr(lyr, 'offset_y', 0)
                min_x = min(min_x, ox)
                min_y = min(min_y, oy)
                max_x = max(max_x, ox + lyr.image.width())
                max_y = max(max_y, oy + lyr.image.height())
        return min_x, min_y, max(max_x - min_x, 1), max(max_y - min_y, 1)

    def _folder_bounds(self, layers) -> tuple[int, int, int, int]:
        """フォルダ結合用: 渡されたレイヤー群だけのバウンディングボックスを返す
        (min_x, min_y, w, h)。_visible_bounds と違いキャンバス全体を初期値に
        含めない（そうしないと結合結果が常にキャンバス全面サイズになってしまう）。"""
        vals = None
        for lyr in layers:
            if lyr.is_group:
                if vals is None:
                    ox = getattr(lyr.children[0], 'offset_x', 0) if lyr.children else 0
                    oy = getattr(lyr.children[0], 'offset_y', 0) if lyr.children else 0
                    vals = [ox, oy, ox, oy]
                self._expand_bounds_group_accum(lyr, vals)
            else:
                ox = getattr(lyr, 'offset_x', 0)
                oy = getattr(lyr, 'offset_y', 0)
                mx = ox + lyr.image.width()
                my = oy + lyr.image.height()
                if vals is None:
                    vals = [ox, oy, mx, my]
                else:
                    vals[0] = min(vals[0], ox)
                    vals[1] = min(vals[1], oy)
                    vals[2] = max(vals[2], mx)
                    vals[3] = max(vals[3], my)
        if vals is None:
            return 0, 0, 1, 1
        min_x, min_y, max_x, max_y = vals
        return min_x, min_y, max(max_x - min_x, 1), max(max_y - min_y, 1)

    def _expand_bounds_group_accum(self, group, vals):
        for child in group.children:
            if child.is_group:
                self._expand_bounds_group_accum(child, vals)
            else:
                ox = getattr(child, 'offset_x', 0)
                oy = getattr(child, 'offset_y', 0)
                vals[0] = min(vals[0], ox)
                vals[1] = min(vals[1], oy)
                vals[2] = max(vals[2], ox + child.image.width())
                vals[3] = max(vals[3], oy + child.image.height())

    def composite(self, skip: object = None) -> QImage:
        """全レイヤーを合成する。skip を指定すると、そのトップレベルレイヤー
        （通常レイヤーのみ・クリッピングと無関係なもの）の描画だけを省略する
        （ストローク中の背景キャッシュ用）。"""
        result = QImage(self.width, self.height, QImage.Format.Format_ARGB32_Premultiplied)
        result.fill(Qt.GlobalColor.transparent)
        render_items(result, self.layers, 0, 0, skip)
        return result.convertToFormat(QImage.Format.Format_ARGB32)

    def can_fast_preview(self, layer: object) -> bool:
        """layer が単独で（クリッピングの授受なしに）差し替え描画できるトップレベル
        の通常レイヤーかどうかを返す。True ならストローク中の背景キャッシュが使える。"""
        if layer is None or layer.is_group:  # type: ignore
            return False
        try:
            i = self.layers.index(layer)
        except ValueError:
            return False
        if layer.clipping:  # type: ignore
            return False
        if i > 0 and self.layers[i - 1].clipping:
            return False
        return True
