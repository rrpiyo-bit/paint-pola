from __future__ import annotations
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


class Layer:
    def __init__(self, name: str, w: int = CANVAS_W, h: int = CANVAS_H):
        self.name = name
        self.visible = True
        self.opacity = 255
        self.clipping = False
        self.reference = False
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

    def clear(self):
        self.image.fill(Qt.GlobalColor.transparent)

    def image_with_border(self) -> QImage:
        if not self.border_enabled or self.border_size <= 0:
            return self.image
        img = self.image
        w, h = img.width(), img.height()
        ptr = img.bits()
        ptr.setsize(h * w * 4)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, w, 4).copy()
        alpha = arr[:, :, 3]
        ksize = self.border_size * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        dilated = cv2.dilate(alpha, kernel)
        border_only = (dilated > 0) & (alpha == 0)
        result = arr.copy()
        bc = self.border_color
        result[border_only] = [bc.blue(), bc.green(), bc.red(), bc.alpha()]
        border_img = QImage(result.tobytes(), w, h, w * 4, QImage.Format.Format_ARGB32).copy()
        out = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
        out.fill(Qt.GlobalColor.transparent)
        p = QPainter(out)
        p.drawImage(0, 0, border_img)
        p.drawImage(0, 0, img)
        p.end()
        return out.convertToFormat(QImage.Format.Format_ARGB32)

    def image_with_effects(self) -> QImage:
        """全レイヤー効果を適用した画像を返す。"""
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
        w, h = img.width(), img.height()
        ptr = img.bits()
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

    @property
    def is_group(self) -> bool:
        return False


class GroupLayer:
    def __init__(self, name: str, w: int = CANVAS_W, h: int = CANVAS_H):
        self.name = name
        self.visible = True
        self.opacity = 255
        self.clipping = False
        self.reference = False
        self.collapsed = False  # True のとき子レイヤーをパネルで非表示
        self.children: list[Layer | GroupLayer] = []
        self._w = w
        self._h = h

    @property
    def is_group(self) -> bool:
        return True

    def resize(self, w: int, h: int):
        self._w = w
        self._h = h
        for child in self.children:
            if child.is_group:
                child.resize(w, h)  # type: ignore
            else:
                # 通常レイヤーの子も新サイズの画像に差し替える（crop モード）
                new_img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
                new_img.fill(Qt.GlobalColor.transparent)
                p = QPainter(new_img)
                p.drawImage(0, 0, child.image)  # type: ignore
                p.end()
                child.image = new_img.convertToFormat(QImage.Format.Format_ARGB32)  # type: ignore

    def composite(self) -> QImage:
        result = QImage(self._w, self._h, QImage.Format.Format_ARGB32)
        result.fill(Qt.GlobalColor.transparent)
        p = QPainter(result)
        children = self.children
        for i in range(len(children) - 1, -1, -1):
            child = children[i]
            if not child.visible:
                continue
            if child.is_group:
                p.setOpacity(child.opacity / 255)
                p.drawImage(0, 0, child.composite())
                continue
            ox = getattr(child, 'offset_x', 0)
            oy = getattr(child, 'offset_y', 0)
            if (child.clipping  # type: ignore
                    and i < len(children) - 1
                    and not children[i + 1].is_group):
                below = children[i + 1]
                box = getattr(below, 'offset_x', 0)
                boy = getattr(below, 'offset_y', 0)
                mask_img = QImage(self._w, self._h, QImage.Format.Format_ARGB32)
                mask_img.fill(Qt.GlobalColor.transparent)
                mp = QPainter(mask_img)
                mp.setOpacity(child.opacity / 255)
                mp.drawImage(ox, oy, child.image_with_effects())  # type: ignore
                mp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
                mp.drawImage(box, boy, below.image_with_effects())  # type: ignore
                mp.end()
                p.setOpacity(1.0)
                p.drawImage(0, 0, mask_img)
            else:
                blend = BLEND_KEY_TO_MODE.get(getattr(child, 'blend_mode', 'normal'))
                if blend:
                    p.setCompositionMode(blend)
                else:
                    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                p.setOpacity(child.opacity / 255)
                p.drawImage(ox, oy, child.image_with_effects())  # type: ignore
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        p.end()
        return result


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
        """参照フラグが立っている表示中のレイヤー（通常・グループ）を全て返す。
        グループは composite() で合成した画像を持つ疑似オブジェクトとして扱う。
        戻り値は .image を持つオブジェクトのリスト。"""
        class _RefProxy:
            def __init__(self, img, opacity):
                self.image = img
                self.opacity = opacity

        result = []
        def _collect(items: list[Layer | GroupLayer]):
            for layer in items:
                if not layer.visible:
                    continue
                if layer.reference:
                    if layer.is_group:
                        result.append(_RefProxy(layer.composite(), layer.opacity))  # type: ignore
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
        p.setOpacity(lower.opacity / 255)
        p.drawImage(l_ox - min_x, l_oy - min_y, lower.image)
        p.setOpacity(upper.opacity / 255)
        p.drawImage(u_ox - min_x, u_oy - min_y, upper.image)
        p.end()
        lower.image = merged.convertToFormat(QImage.Format.Format_ARGB32)
        lower.opacity = 255
        lower.offset_x = min_x
        lower.offset_y = min_y

        container.pop(idx)
        if not container and parent_path:
            self.active_path = parent_path
        else:
            new_idx = idx - 1 if idx > 0 else 0
            self.active_path = parent_path + [new_idx]
        return True

    def merge_all_visible(self) -> bool:
        """表示中のレイヤーを1枚に統合する。非表示レイヤーは破棄せず下に残す。
        画面外にはみ出た部分も保持する。"""
        if not self.layers:
            return False
        hidden = [lyr for lyr in self.layers if not lyr.visible]

        # 全表示レイヤーの範囲を計算
        visible = [lyr for lyr in self.layers if lyr.visible]
        if not visible:
            return False
        bounds = self._visible_bounds(visible)
        min_x, min_y, mw, mh = bounds

        merged = QImage(mw, mh, QImage.Format.Format_ARGB32_Premultiplied)
        merged.fill(Qt.GlobalColor.transparent)
        p = QPainter(merged)
        for lyr in reversed(visible):
            self._draw_layer_to(p, lyr, min_x, min_y)
        p.end()

        new_layer = Layer("統合レイヤー", mw, mh)
        new_layer.image = merged.convertToFormat(QImage.Format.Format_ARGB32)
        new_layer.offset_x = min_x
        new_layer.offset_y = min_y
        self.layers = [new_layer] + hidden
        self.active_path = [0]
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

    def _draw_layer_to(self, p: QPainter, lyr, off_x: int, off_y: int):
        """統合用: lyrをオフセット補正して描画する。"""
        if lyr.is_group:
            for child in reversed(lyr.children):
                if child.visible:
                    self._draw_layer_to(p, child, off_x, off_y)
        else:
            ox = getattr(lyr, 'offset_x', 0)
            oy = getattr(lyr, 'offset_y', 0)
            img = lyr.image_with_effects() if hasattr(lyr, 'image_with_effects') else lyr.image
            p.setOpacity(lyr.opacity / 255)
            p.drawImage(ox - off_x, oy - off_y, img)

    def composite(self) -> QImage:
        result = QImage(self.width, self.height, QImage.Format.Format_ARGB32)
        result.fill(Qt.GlobalColor.transparent)
        if not self.layers:
            return result
        p = QPainter(result)

        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            if not layer.visible:
                continue

            ox = getattr(layer, 'offset_x', 0)
            oy = getattr(layer, 'offset_y', 0)

            clipping = layer.clipping and i < len(self.layers) - 1
            if clipping:
                below = self.layers[i + 1]
                box = getattr(below, 'offset_x', 0)
                boy = getattr(below, 'offset_y', 0)
                below_img = below.composite() if below.is_group else below.image_with_effects()  # type: ignore
                src_img = layer.composite() if layer.is_group else layer.image_with_effects()  # type: ignore
                mask_img = QImage(self.width, self.height, QImage.Format.Format_ARGB32_Premultiplied)
                mask_img.fill(Qt.GlobalColor.transparent)
                mp = QPainter(mask_img)
                mp.setOpacity(layer.opacity / 255)
                mp.drawImage(ox, oy, src_img)
                mp.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
                mp.drawImage(box, boy, below_img)
                mp.end()
                p.setOpacity(1.0)
                p.drawImage(0, 0, mask_img)
            elif layer.is_group:
                p.setOpacity(layer.opacity / 255)
                p.drawImage(0, 0, layer.composite())  # type: ignore
            else:
                blend = BLEND_KEY_TO_MODE.get(getattr(layer, 'blend_mode', 'normal'))
                if blend:
                    p.setCompositionMode(blend)
                else:
                    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                p.setOpacity(layer.opacity / 255)
                p.drawImage(ox, oy, layer.image_with_effects())  # type: ignore
                p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        p.end()
        return result
