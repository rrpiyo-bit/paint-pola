"""テストシナリオで描かれる絵とアニメーションを画像ファイルとして出力する。"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QColor, QMouseEvent, QPainter
from PyQt6.QtCore import Qt, QPointF, QEvent, QPoint

app = QApplication.instance() or QApplication(sys.argv)

from main import MainWindow
from tools import Tool
from layer import Layer

OUT = os.path.join(os.path.dirname(__file__), "images", "samples")
os.makedirs(OUT, exist_ok=True)


def press(c, x, y):
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y), QPointF(x, y),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier)
    c.mousePressEvent(ev)

def move(c, x, y):
    ev = QMouseEvent(QEvent.Type.MouseMove, QPointF(x, y), QPointF(x, y),
                     Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier)
    c.mouseMoveEvent(ev)

def release(c, x, y):
    ev = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(x, y), QPointF(x, y),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier)
    c.mouseReleaseEvent(ev)

def stroke(c, x1, y1, x2, y2):
    press(c, x1, y1)
    move(c, (x1+x2)//2, (y1+y2)//2)
    move(c, x2, y2)
    release(c, x2, y2)

def clear(c):
    c.tool = Tool.ERASER
    c.eraser_size = 200
    for y in range(50, 350, 80):
        stroke(c, 10, y, 290, y)

def save_composite(win, name):
    img = win.layer_stack.composite()
    path = os.path.join(OUT, name)
    img.save(path)
    print(f"  saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# 1. キャラクターイラスト全工程
# ══════════════════════════════════════════════════════════════════════════
print("=== キャラクターイラスト ===")
win = MainWindow()
c = win.canvas
ls = c.layer_stack

# ラフ
ls.layers[0].name = "ラフ"
c.tool = Tool.PEN
c.pen_color = QColor(180, 180, 255, 100)
c.pen_size = 15
stroke(c, 100, 50, 100, 150)
stroke(c, 200, 50, 200, 150)
stroke(c, 100, 50, 200, 50)
stroke(c, 100, 150, 150, 180)
stroke(c, 200, 150, 150, 180)
stroke(c, 120, 180, 80, 350)
stroke(c, 180, 180, 220, 350)
save_composite(win, "01_rough.png")

# 線画
ls.layers[0].opacity = 30
win.layer_panel._add()
ls.active.name = "線画"
c.pen_color = QColor(30, 30, 30, 255)
c.pen_size = 3
stroke(c, 100, 50, 100, 150)
stroke(c, 200, 50, 200, 150)
stroke(c, 100, 50, 200, 50)
stroke(c, 100, 150, 150, 180)
stroke(c, 200, 150, 150, 180)
stroke(c, 125, 90, 145, 90)
stroke(c, 160, 90, 175, 90)
stroke(c, 140, 130, 165, 130)
stroke(c, 120, 180, 80, 350)
stroke(c, 180, 180, 220, 350)
ls.layers[0].visible = False
save_composite(win, "02_lineart.png")

# 下塗り: 肌
win.layer_panel._add()
ls.active.name = "肌"
c.pen_color = QColor(255, 220, 200, 255)
c.pen_size = 40
for y in range(60, 170, 20):
    stroke(c, 110, y, 195, y)
save_composite(win, "03_skin.png")

# 下塗り: 髪
win.layer_panel._add()
ls.active.name = "髪"
c.pen_color = QColor(60, 40, 30, 255)
c.pen_size = 25
stroke(c, 90, 30, 210, 30)
stroke(c, 90, 30, 85, 100)
stroke(c, 210, 30, 215, 100)
stroke(c, 95, 45, 205, 45)
save_composite(win, "04_hair.png")

# 下塗り: 服
win.layer_panel._add()
ls.active.name = "服"
c.pen_color = QColor(70, 100, 180, 255)
c.pen_size = 35
for y in range(185, 350, 40):
    stroke(c, 80, y, 220, y)
save_composite(win, "05_cloth.png")

# 影 (乗算)
win.layer_panel._add()
ls.active.name = "影"
ls.active.blend_mode = "multiply"
c.pen_color = QColor(180, 140, 180, 80)
c.pen_size = 25
stroke(c, 170, 70, 190, 130)
stroke(c, 125, 155, 175, 155)
stroke(c, 130, 220, 200, 260)
stroke(c, 120, 300, 210, 330)
save_composite(win, "06_shadow.png")

# ハイライト (加算)
win.layer_panel._add()
ls.active.name = "ハイライト"
ls.active.blend_mode = "plus"
c.pen_color = QColor(255, 255, 230, 60)
c.pen_size = 15
stroke(c, 110, 35, 190, 35)
c.pen_size = 5
stroke(c, 130, 87, 135, 87)
stroke(c, 165, 87, 170, 87)
save_composite(win, "07_highlight.png")

# 仕上げ (オーバーレイ + エフェクト)
win.layer_panel._add()
ls.active.name = "色味調整"
ls.active.blend_mode = "overlay"
ls.active.opacity = 30
c.pen_color = QColor(255, 200, 150, 40)
c.pen_size = 100
stroke(c, 50, 50, 250, 350)

# エフェクト
ls.set_active(1, -1)
ls.active.glow_enabled = True
ls.active.glow_color = QColor(255, 240, 220)
ls.active.glow_size = 2
ls.active.glow_strength = 30

ls.set_active(2, -1)
ls.active.hsl_enabled = True
ls.active.hsl_hue = 5
ls.active.hsl_saturation = 15

ls.set_active(5, -1)
ls.active.blur_enabled = True
ls.active.blur_radius = 3
ls.active.blur_strength = 50

save_composite(win, "08_final.png")
win.close()

# ══════════════════════════════════════════════════════════════════════════
# 2. ボール跳ねアニメーション
# ══════════════════════════════════════════════════════════════════════════
print("\n=== ボール跳ねアニメーション ===")
win = MainWindow()
c = win.canvas
ap = win.anim_panel
win._toggle_anim_mode(True)

def draw_ball(cx, cy, size=20):
    c.tool = Tool.ELLIPSE
    c.pen_size = 2
    c.shape_fill = "fill"
    press(c, cx - size, cy - size)
    move(c, cx + size, cy + size)
    release(c, cx + size, cy + size)

# フレーム1: 上
c.pen_color = QColor(255, 80, 80, 255)
draw_ball(150, 80)
# 地面
c.tool = Tool.LINE
c.pen_color = QColor(100, 80, 60, 255)
c.pen_size = 2
press(c, 50, 290); move(c, 250, 290); release(c, 250, 290)
save_composite(win, "anim_bounce_f1.png")
ap._on_add_frame()

# フレーム2: 中間
clear(c)
c.pen_color = QColor(255, 80, 80, 255)
draw_ball(150, 165)
c.tool = Tool.PEN
c.pen_color = QColor(100, 100, 100, 50)
c.pen_size = 20
stroke(c, 133, 285, 167, 285)
c.tool = Tool.LINE
c.pen_color = QColor(100, 80, 60, 255)
c.pen_size = 2
press(c, 50, 290); move(c, 250, 290); release(c, 250, 290)
save_composite(win, "anim_bounce_f2.png")
ap._on_add_frame()

# フレーム3: 下
clear(c)
c.pen_color = QColor(255, 80, 80, 255)
draw_ball(150, 250)
c.tool = Tool.PEN
c.pen_color = QColor(100, 100, 100, 80)
c.pen_size = 25
stroke(c, 130, 285, 170, 285)
c.tool = Tool.LINE
c.pen_color = QColor(100, 80, 60, 255)
c.pen_size = 2
press(c, 50, 290); move(c, 250, 290); release(c, 250, 290)
save_composite(win, "anim_bounce_f3.png")
ap._on_add_frame()

# GIF出力
gif_path = os.path.join(OUT, "bounce.gif")
ap._export_gif_to(gif_path)
print(f"  saved: {gif_path}")
win.close()

# ══════════════════════════════════════════════════════════════════════════
# 3. 歩行サイクル
# ══════════════════════════════════════════════════════════════════════════
print("\n=== 歩行サイクル ===")
win = MainWindow()
c = win.canvas
ap = win.anim_panel
win._toggle_anim_mode(True)

def draw_ground():
    c.tool = Tool.LINE
    c.pen_color = QColor(100, 80, 60, 255)
    c.pen_size = 2
    press(c, 50, 280); move(c, 250, 280); release(c, 250, 280)

def draw_stickman(head_y, body_top, body_bot, lfoot_x, rfoot_x, lfoot_y=275, rfoot_y=275):
    # 頭
    c.tool = Tool.ELLIPSE
    c.pen_size = 2
    c.pen_color = QColor(0, 0, 0, 255)
    c.shape_fill = "none"
    press(c, 140, head_y - 10)
    move(c, 160, head_y + 10)
    release(c, 160, head_y + 10)
    # 体
    c.tool = Tool.LINE
    c.pen_size = 3
    press(c, 150, body_top); move(c, 150, body_bot); release(c, 150, body_bot)
    # 左足
    press(c, 150, body_bot); move(c, lfoot_x, lfoot_y); release(c, lfoot_x, lfoot_y)
    # 右足
    press(c, 150, body_bot); move(c, rfoot_x, rfoot_y); release(c, rfoot_x, rfoot_y)
    # 左腕
    press(c, 150, body_top + 15); move(c, rfoot_x - 10, body_top + 50); release(c, rfoot_x - 10, body_top + 50)
    # 右腕
    press(c, 150, body_top + 15); move(c, lfoot_x + 10, body_top + 50); release(c, lfoot_x + 10, body_top + 50)

frames_data = [
    (100, 110, 200, 140, 160, 275, 275),   # 直立
    (95, 105, 195, 135, 175, 275, 260),     # 右足前
    (100, 110, 200, 130, 170, 275, 275),    # 右足接地
    (95, 105, 195, 170, 130, 275, 260),     # 左足前
]

for i, (hy, bt, bb, lx, rx, ly, ry) in enumerate(frames_data):
    clear(c)
    draw_stickman(hy, bt, bb, lx, rx, ly, ry)
    draw_ground()
    save_composite(win, f"anim_walk_f{i+1}.png")
    ap._on_add_frame()

gif_path = os.path.join(OUT, "walk_cycle.gif")
ap._export_gif_to(gif_path)
print(f"  saved: {gif_path}")
win.close()

# ══════════════════════════════════════════════════════════════════════════
# 4. 表情差分アニメーション
# ══════════════════════════════════════════════════════════════════════════
print("\n=== 表情差分 ===")
win = MainWindow()
c = win.canvas
ap = win.anim_panel
win._toggle_anim_mode(True)

def draw_face_base():
    c.tool = Tool.PEN
    c.pen_color = QColor(0, 0, 0, 255)
    c.pen_size = 3
    stroke(c, 100, 50, 100, 150)
    stroke(c, 200, 50, 200, 150)
    stroke(c, 100, 50, 200, 50)
    stroke(c, 100, 150, 150, 180)
    stroke(c, 200, 150, 150, 180)
    # 肌色
    c.pen_color = QColor(255, 220, 200, 200)
    c.pen_size = 35
    for y in range(65, 165, 20):
        stroke(c, 115, y, 190, y)
    c.pen_color = QColor(0, 0, 0, 255)
    c.pen_size = 3

# 笑顔
clear(c)
draw_face_base()
stroke(c, 125, 95, 135, 85); stroke(c, 135, 85, 145, 95)
stroke(c, 160, 95, 170, 85); stroke(c, 170, 85, 180, 95)
stroke(c, 135, 130, 150, 140); stroke(c, 150, 140, 165, 130)
# 頬の赤み
c.pen_color = QColor(255, 150, 150, 60)
c.pen_size = 15
stroke(c, 110, 115, 125, 115)
stroke(c, 180, 115, 195, 115)
save_composite(win, "anim_face_smile.png")
ap._on_add_frame()

# 普通
clear(c)
draw_face_base()
c.pen_color = QColor(0, 0, 0, 255)
c.pen_size = 3
stroke(c, 125, 90, 145, 90)
stroke(c, 160, 90, 180, 90)
stroke(c, 140, 135, 160, 135)
save_composite(win, "anim_face_normal.png")
ap._on_add_frame()

# 驚き
clear(c)
draw_face_base()
c.pen_color = QColor(0, 0, 0, 255)
c.pen_size = 3
# 大きい目
stroke(c, 125, 82, 145, 82); stroke(c, 125, 82, 125, 98)
stroke(c, 145, 82, 145, 98); stroke(c, 125, 98, 145, 98)
stroke(c, 160, 82, 180, 82); stroke(c, 160, 82, 160, 98)
stroke(c, 180, 82, 180, 98); stroke(c, 160, 98, 180, 98)
# O型の口
c.tool = Tool.ELLIPSE
c.pen_size = 2
c.shape_fill = "none"
press(c, 143, 125); move(c, 158, 148); release(c, 158, 148)
save_composite(win, "anim_face_surprise.png")
ap._on_add_frame()

gif_path = os.path.join(OUT, "expressions.gif")
ap._export_gif_to(gif_path)
print(f"  saved: {gif_path}")
win.close()

print(f"\n完了！ {OUT} に出力しました。")
