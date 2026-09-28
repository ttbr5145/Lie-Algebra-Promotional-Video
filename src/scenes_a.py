# -*- coding: utf-8 -*-
"""场景 S0-S3: 开场标题 / SO(3)旋转之美 / 指数映射 / 不可交换性"""
import math
import numpy as np
from PIL import Image, ImageDraw
from engine import (W, H, FPS, Canvas, Background, Camera, vignette,
                    CYAN, MAGENTA, GOLD, VIOLET, GREEN, WHITE, ORANGE, BLUE, PINK,
                    seg, ease_in_out, ease_out_cubic, ease_in_cubic, ease_out_back,
                    clamp01, lerp, normalize, axis_angle_matrix,
                    subtitle, big_title, FONT_SANS_BOLD, FONT_SERIF_BOLD, get_font)


# ============================================================
# S0 开场: 星尘汇聚成「李代数」  390帧 / 13s
# ============================================================
class Scene0:
    N = 390
    def __init__(self):
        self.bg = Background(
            top=(3, 5, 14), bottom=(8, 10, 30),
            halo=[(W/2, 500, 760, (36, 56, 130), 0.42), (W/2, 1080, 900, (60, 30, 90), 0.25)],
            seed=11, n_stars=190)
        # 文字轮廓采样
        f = get_font(370, FONT_SERIF_BOLD)
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        d.text((W/2, 460), "李代数", font=f, fill=255, anchor="mm")
        mask = np.asarray(img)
        ys, xs = np.nonzero(mask > 120)
        rng = np.random.default_rng(5)
        idx = rng.choice(len(xs), size=1500, replace=False)
        self.tx = xs[idx].astype(np.float32)
        self.ty = ys[idx].astype(np.float32)
        # 粒子起点: 画面外围环带
        ang = rng.uniform(0, 2*np.pi, 1500)
        rad = rng.uniform(950, 1600, 1500)
        self.sx = W/2 + np.cos(ang) * rad
        self.sy = H/2 + np.sin(ang) * rad * 0.72
        self.delay = rng.uniform(0, 3.4, 1500)          # 起飞延迟
        self.dur = rng.uniform(1.1, 1.9, 1500)          # 飞行时长
        # 颜色: 按目标x从青到金
        k = (self.tx - self.tx.min()) / (self.tx.max() - self.tx.min() + 1e-6)
        self.col = np.stack([60 + 195*k, 200 + 20*(1-k), 255 - 165*k], 1)
        self.phase = rng.uniform(0, 6.28, 1500)

    def frame(self, i):
        t = i / FPS
        c = Canvas(self.bg.img)
        c.twinkle_stars(self.bg.stars, t)
        fly_t = (t - self.delay) / self.dur
        moving = fly_t < 1.0
        ft = ease_in_out(np.clip(fly_t, 0, 1))
        px = self.sx * (1 - ft) + self.tx * ft
        py = self.sy * (1 - ft) + self.ty * ft
        # 到达后的呼吸漂浮
        arrived = ~moving
        px = np.where(moving, px, self.tx + np.sin(t*1.3 + self.phase) * 1.6)
        py = np.where(moving, py, self.ty + np.cos(t*1.1 + self.phase) * 1.6)
        r = np.where(moving, 1.2 + 1.0*np.sin(np.pi*np.clip(fly_t,0,1)), 1.9)
        alpha = np.where(moving, 0.3 + 0.7*np.clip(fly_t,0,1), 1.0)
        # 逐粒子绘制(sprite)
        col_i = self.col
        for j in range(0, 1500, 1):
            c.add_dot((px[j], py[j]), float(r[j]), tuple(int(v) for v in col_i[j]), float(alpha[j]))
        # 标题本体微光(汇聚完成后)
        if t > 6.0:
            k = seg(t, 6.0, 9.5)
            c.text((W/2, 460), "李代数", 370, (255, 205, 90), "mm", 0.15*k,
                   glow=0.8*k, font_path=FONT_SERIF_BOLD)
        # 副标题
        if t > 8.2:
            k = ease_out_cubic(seg(t, 8.2, 10.2))
            c.text((W/2, 742), "对 称 性 的 语 言", 54, (215, 228, 255), "mm", 0.95*k, glow=0.55*k, tracking=6)
            c.text((W/2, 830), "THE ALGEBRA OF SYMMETRY", 26, (120, 140, 195), "mm", 0.8*k,
                   glow=0.35*k, tracking=14)
        arr = c.finish()
        # 黑场淡入淡出
        k = ease_in_cubic(seg(t, 0, 0.9)) * (1 - ease_in_cubic(seg(t, 12.1, 13.0)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S1 SO(3) 旋转之美  810帧 / 27s
# ============================================================
def fibonacci_sphere(n, r=1.0):
    pts = np.zeros((n, 3))
    phi = math.pi * (3 - math.sqrt(5))
    for i in range(n):
        y = 1 - (i / (n - 1)) * 2
        rad = math.sqrt(1 - y*y)
        th = phi * i
        pts[i] = [math.cos(th)*rad, y, math.sin(th)*rad]
    return pts * r

ICOSAHEDRON = None
def icosahedron(scale=1.0):
    global ICOSAHEDRON
    if ICOSAHEDRON is None:
        p = (1 + math.sqrt(5)) / 2
        vs = []
        for a in (-1, 1):
            for b in (-p, p):
                vs += [(0, a, b), (a, b, 0), (b, 0, a)]
        ICOSAHEDRON = np.array(vs, float)
    return ICOSAHEDRON * scale

def ico_edges():
    V = icosahedron(1.0)
    edges = []
    d = np.linalg.norm(V[:, None] - V[None, :], axis=-1)
    L = d[d > 1e-6].min()
    for i in range(12):
        for j in range(i+1, 12):
            if abs(d[i, j] - L) < 1e-6:
                edges.append((i, j))
    return edges

class Scene1:
    N = 810
    def __init__(self):
        self.bg = Background(
            top=(4, 6, 18), bottom=(7, 9, 26),
            halo=[(W/2, 500, 800, (30, 50, 120), 0.35)],
            seed=23, n_stars=170)
        self.ball = fibonacci_sphere(300, 1.55)
        self.ico = icosahedron(0.92 * 1.55 / 1.902)
        self.e9 = ico_edges()
        self.markers = np.array([[0.3, 0.9, 0.32], [-0.85, 0.35, 0.4], [0.1, -0.6, 0.79]])
        self.markers = self.markers / np.linalg.norm(self.markers, axis=1, keepdims=True) * 1.55

    def frame(self, i):
        t = i / FPS
        c = Canvas(self.bg.img)
        c.twinkle_stars(self.bg.stars, t)
        cam = Camera(yaw=0.55 + 0.05*math.sin(t*0.16), pitch=0.32 + 0.08*math.sin(t*0.11),
                     dist=5.1, target=(0, 0, 0))
        # 转轴: 前10s固定, 10-19s漂移, 之后稳定在新轴
        if t < 10:
            axis = normalize(np.array([0.35, 1.0, 0.42]))
        elif t < 19:
            k = ease_in_out((t-10)/9)
            a0 = np.array([0.35, 1.0, 0.42]); a1 = normalize(np.array([-0.6, 0.7, 0.75]))
            axis = normalize(a0*(1-k) + a1*k + np.array([0, 0.25*math.sin((t-10)*1.2)*k, 0]))
        else:
            axis = normalize(np.array([-0.6, 0.7, 0.75]))
        ang = 0.85 * t
        R = axis_angle_matrix(axis, ang)
        # 旋转后的球面点(带残影拖尾, 强化旋转感)
        pts = self.ball @ R.T
        p2, zz = cam.project(pts)
        order = np.argsort(-zz)
        ghosts = []
        for dphi, ga in ((0.10, 0.30), (0.20, 0.16), (0.32, 0.08)):
            gp = self.ball @ axis_angle_matrix(axis, ang - dphi).T
            g2, gz = cam.project(gp)
            ghosts.append((g2, gz, ga))
        for (g2, gz, ga) in ghosts:
            for j in order:
                al = clamp01(1.7 - gz[j] / 5.2) * ga
                if al < 0.02: continue
                c.dot(g2[j], max(1.0, 5.5/gz[j]), (80, 170, 240), al, glow=0.25)
        for j in order:
            depth = zz[j]
            sz = max(1.4, 7.6 / depth)
            al = clamp01(1.7 - depth / 5.2) * 0.68
            c.dot(p2[j], sz, (105, 200, 255), al, glow=0.6)
        # 二十面体线框
        iv = self.ico @ R.T
        ip2, izz = cam.project(iv)
        for (a, b) in self.e9:
            z = (izz[a] + izz[b]) / 2
            fade = clamp01(1.6 - z / 5.0)
            c.line(ip2[a], ip2[b], GOLD, 2, 0.8*fade, glow=0.8)
        for j in range(12):
            c.dot(ip2[j], max(1.5, 24.0/izz[j]), GOLD, 0.9*clamp01(1.6-izz[j]/5), glow=1.1)
        # 转轴箭头(不随物体旋转)
        a3 = axis * 1.9
        tip, zt = cam.project_single(a3)
        base, zb = cam.project_single(-a3)
        gl = 0.8 + 0.4*math.sin(t*2.4)
        c.line(base, tip, PINK, 2, 0.5, glow=gl*0.45)
        # 箭头头部
        d2 = (tip[0]-base[0], tip[1]-base[1])
        L = math.hypot(*d2) + 1e-6
        ux, uy = d2[0]/L, d2[1]/L
        for s in (-1, 1):
            c.line(tip, (tip[0] - ux*22 + uy*s*9, tip[1] - uy*22 - ux*s*9), PINK, 2, 0.7, glow=gl*0.5)
        c.dot(tip, 3.5, PINK, 0.85, glow=1.0*gl)
        # 标记点轨道大圆
        for m in self.markers:
            mm = m @ R.T
            h = float(m @ axis); cen = axis * h
            u = m - axis * h; u = u / (np.linalg.norm(u)+1e-9)
            v = np.cross(axis, u)
            rho = math.sqrt(max(0.0, 1.55**2 - h*h))
            ring = np.array([cen + rho*(u*math.cos(th) + v*math.sin(th)) for th in np.linspace(0, 2*np.pi, 72)])
            r2, rz = cam.project(ring)
            for k in range(71):
                zf = (rz[k] + rz[k+1]) / 2
                c.line(r2[k], r2[k+1], CYAN, 2, 0.22 * clamp01(1.6 - zf/5.2), glow=0.3)
            c.dot(cam.project_single(mm)[0], 5.5, CYAN, 0.95, glow=1.5)
        # 字幕
        y = H - 150
        if 0.6 < t < 5.2:
            a = ease_in_out(seg(t, 0.6, 1.3)) * (1 - ease_in_out(seg(t, 4.6, 5.2)))
            subtitle(c, y, "旋转 —— 宇宙中最熟悉的对称", alpha=a)
        if 6.4 < t < 12.0:
            a = ease_in_out(seg(t, 6.4, 7.2)) * (1 - ease_in_out(seg(t, 11.4, 12.0)))
            subtitle(c, y, "而所有的旋转放在一起，构成一个光滑、弯曲的空间", alpha=a)
        if 13.6 < t < 20.5:
            a = ease_in_out(seg(t, 13.6, 14.4)) * (1 - ease_in_out(seg(t, 19.9, 20.5)))
            subtitle(c, y, "数学家称之为 —— 李群", alpha=a)
        if t > 20.8:
            a = ease_out_cubic(seg(t, 20.8, 22.4)) * (1 - ease_in_out(seg(t, 25.9, 26.9)))
            c.text((W/2, H-190), "SO(3)", 86, GOLD, "mm", a, glow=1.2*a, tracking=8)
            c.text((W/2, H-108), "三维旋转群 · 一个天然的弯曲宇宙", 36, (200, 214, 245), "mm", 0.9*a, glow=0.4*a)
        arr = vignette(c.finish(), 0.32)
        k = ease_in_cubic(seg(t, 0, 0.8)) * (1 - ease_in_cubic(seg(t, 26.2, 27.0)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S2 指数映射: 无穷小 -> 李代数  810帧 / 27s
# ============================================================
class Scene2:
    N = 810
    def __init__(self):
        self.bg = Background(
            top=(3, 6, 16), bottom=(6, 9, 26),
            halo=[(W/2, 520, 780, (30, 60, 130), 0.38)],
            seed=31, n_stars=160)
        self.ball = fibonacci_sphere(240, 1.7)
        self.P = np.array([0.25, 0.60, 0.76]); self.P = self.P / np.linalg.norm(self.P) * 1.7

    def frame(self, i):
        t = i / FPS
        c = Canvas(self.bg.img)
        c.twinkle_stars(self.bg.stars, t)
        cam = Camera(yaw=0.62 + 0.045*math.sin(t*0.14), pitch=0.30 + 0.06*math.sin(t*0.1),
                     dist=4.6)
        R0 = axis_angle_matrix((0, 1, 0), 0.22)   # 整体缓慢自转
        R0 = R0 @ axis_angle_matrix((0, 0, 1), t * 0.05)
        ball = self.ball @ R0.T
        P = self.P @ R0.T
        # 切向量方向 e (固定于球面坐标系, 随整体转)
        e = normalize(np.cross(np.array([0, 0, 1]), self.P)) @ R0.T
        e = normalize(e - P * (e @ P) / (1.7*1.7))
        # --- 向量长度演化: 0-6s 从0长到1.0, 保持 ---
        vlen = 1.05 * ease_in_out(seg(t, 1.0, 6.5)) + 0.04*math.sin(t*2.2)
        # 球点
        p2, zz = cam.project(ball)
        order = np.argsort(-zz)
        for j in order:
            al = clamp01(1.7 - zz[j] / 5.2) * 0.62
            c.dot(p2[j], max(1.3, 7.2/zz[j]), (100, 195, 255), al, glow=0.5)
        # 球轮廓圈
        c0, zc = cam.project_single(np.zeros(3))
        rr = 1.7 / zc * cam.f
        n_ring = 80
        for k in range(n_ring):
            th1 = 2*np.pi*k/n_ring; th2 = 2*np.pi*(k+1)/n_ring
            c.line((c0[0]+rr*math.cos(th1), c0[1]+rr*math.sin(th1)*0.995),
                   (c0[0]+rr*math.cos(th2), c0[1]+rr*math.sin(th2)*0.995),
                   (120, 160, 230), 1, 0.20, glow=0.12)
        # 切平面圆盘 @ P
        u = normalize(np.cross(P, e)); v = np.cross(P, u)
        disk = np.array([P + 0.75*(u*math.cos(th) + v*math.sin(th)) for th in np.linspace(0, 2*np.pi, 56)])
        d2, dz = cam.project(disk)
        for k in range(55):
            c.line(d2[k], d2[k+1], VIOLET, 2, 0.5 * clamp01(1.7-dz.mean()/5), glow=0.5)
        # 弧线: exp(θX), θ = vlen/1.7
        theta = vlen / 1.7 * 1.7
        rot_ax = normalize(np.cross(P, e))
        if theta > 0.004:
            arc = np.array([axis_angle_matrix(rot_ax, theta * s) @ P for s in np.linspace(0, 1, 52)])
            a2, az = cam.project(arc)
            for k in range(51):
                zf = (az[k] + az[k+1]) / 2
                c.line(a2[k], a2[k+1], CYAN, 3, 0.95 * clamp01(1.7-zf/5), glow=1.25)
            c.dot(a2[-1], 5.0, CYAN, 0.95, glow=1.6)
        # 切向量箭头
        if vlen > 0.02:
            q1, z1 = cam.project_single(P)
            q2, z2 = cam.project_single(P + e * vlen)
            gl = 1.0 + 0.4*math.sin(t*2.6)
            c.line(q1, q2, GOLD, 3, 0.95, glow=gl)
            c.add_line(q1, q2, GOLD, 2.5, 0.30)
            ddx, ddy = q2[0]-q1[0], q2[1]-q1[1]
            L = math.hypot(ddx, ddy) + 1e-6
            ux, uy = ddx/L, ddy/L
            for s in (-1, 1):
                c.line(q2, (q2[0]-ux*24+uy*s*10, q2[1]-uy*24-ux*s*10), GOLD, 3, 0.9, glow=gl*0.7)
            c.dot(P, 4.5, WHITE, 0.95, glow=1.2)
            c.dot(q2, 3.8, GOLD, 0.95, glow=1.3*gl)
        # 公式 (8s 后)
        if t > 8.0:
            k = ease_out_cubic(seg(t, 8.0, 9.6)) * (1 - ease_in_cubic(seg(t, 26.0, 27.0)))
            c.text((W/2, H-236), "exp( θ X )", 66, GOLD, "mm", k, glow=1.1*k, tracking=4)
        # 字幕
        y = H - 130
        msgs = [
            (0.8, 5.4, "把旋转推到极限 —— 无穷小"),
            (6.8, 12.4, "每一点上，都住着“即将转动”的倾向：切向量"),
            (13.8, 19.4, "沿着它出发，掠过球面 —— 又回到了旋转"),
            (20.8, 26.2, "这些无穷小旋转的总和，就是李代数 so(3)"),
        ]
        for (a0, a1, s) in msgs:
            if a0 < t < a1 + 0.9:
                a = ease_in_out(seg(t, a0, a0+0.7)) * (1 - ease_in_out(seg(t, a1, a1+0.9)))
                subtitle(c, y, s, alpha=a)
        arr = vignette(c.finish(), 0.32)
        k = ease_in_cubic(seg(t, 0, 0.8)) * (1 - ease_in_cubic(seg(t, 26.2, 27.0)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S3 顺序的秘密: 旋转不可交换  660帧 / 22s
# ============================================================
CUBE_V = np.array([[sx, sy, sz] for sx in (-0.78, 0.78) for sy in (-0.78, 0.78) for sz in (-0.78, 0.78)])
CUBE_E = []
for a in range(8):
    for b in range(a+1, 8):
        diff = np.array(CUBE_V[a]) - np.array(CUBE_V[b])
        if abs(abs(diff).sum() - 1.56) < 1e-6:
            CUBE_E.append((a, b))

def edge_color(v1, v2):
    d = np.abs(np.array(v1) - np.array(v2))
    if d[0] > 0.5: return CYAN
    if d[1] > 0.5: return MAGENTA
    return GOLD

class Scene3:
    N = 660
    def __init__(self):
        self.bg = Background(
            top=(4, 5, 15), bottom=(8, 8, 24),
            halo=[(W*0.27, 460, 520, (28, 52, 120), 0.30), (W*0.73, 460, 520, (60, 28, 105), 0.30)],
            seed=47, n_stars=140)

    def _draw_cube(self, c, cam, M, ax_labels=True):
        V = CUBE_V @ M.T
        p2, zz = cam.project(V, center=cam.center if hasattr(cam, "center") else (W/2, H/2))
        for (a, b) in CUBE_E:
            col = edge_color(CUBE_V[a], CUBE_V[b])
            z = (zz[a] + zz[b]) / 2
            fade = clamp01(1.6 - z / 5.0)
            c.line(p2[a], p2[b], col, 3, 0.9*fade, glow=1.0)
        for j in range(8):
            c.dot(p2[j], max(1.8, 11.0/zz[j]), WHITE, 0.9*clamp01(1.6-zz[j]/5), glow=1.1)

    def frame(self, i):
        t = i / FPS
        c = Canvas(self.bg.img)
        c.twinkle_stars(self.bg.stars, t)
        # 姿态角
        def step(a0, a1): return ease_in_out(seg(t, a0, a1))
        e1 = step(4.0, 6.2); e2 = step(8.0, 10.2)
        # 左: 先X后Y; 右: 先Y后X
        ML = axis_angle_matrix((1, 0, 0), e2 * math.pi/2) @ axis_angle_matrix((0, 1, 0), e1 * math.pi/2)
        MR = axis_angle_matrix((0, 1, 0), e2 * math.pi/2) @ axis_angle_matrix((1, 0, 0), e1 * math.pi/2)
        for (cx, M, tag, tagcol) in ((W*0.27, ML, "先绕 X，再绕 Y", CYAN), (W*0.73, MR, "先绕 Y，再绕 X", MAGENTA)):
            cam = Camera(yaw=0.52, pitch=0.30, dist=5.45, target=(0, 0, 0))
            cam.center = (cx, 460)
            self._draw_cube(c, cam, M)
            if t > 3.0:
                a = ease_in_out(seg(t, 3.0, 3.8))
                c.text((cx, 150), tag, 40, tagcol, "mm", a, glow=0.7*a)
        # 坐标轴微标(立方体初始姿态下)
        # ≠ 闪现
        if t > 10.6:
            k = ease_out_back(seg(t, 10.6, 11.3), 2.2)
            flash = math.exp(-max(0.0, t - 10.6) * 2.2)
            if flash > 0.02:
                c.add_dot((W/2, 470), 500, (180, 200, 255), 0.30*flash)
            c.text((W/2, 470), "≠", 170, WHITE, "mm", clamp01(k), glow=1.5)
        # 公式
        if t > 12.0:
            k = ease_out_cubic(seg(t, 12.0, 13.2)) * (1 - ease_in_cubic(seg(t, 21.0, 22.0)))
            c.text((W/2, 820), "Rx(90°) · Ry(90°)   ≠   Ry(90°) · Rx(90°)", 50, (225, 232, 255), "mm", k, glow=0.7*k)
        # 字幕
        y = H - 110
        if 0.6 < t < 3.6:
            a = ease_in_out(seg(t, 0.6, 1.3)) * (1 - ease_in_out(seg(t, 3.0, 3.6)))
            subtitle(c, y, "但旋转有一个秘密：顺序，很重要", alpha=a)
        if 14.5 < t < 21.2:
            a = ease_in_out(seg(t, 14.5, 15.3)) * (1 - ease_in_out(seg(t, 20.6, 21.2)))
            subtitle(c, y, "两种顺序，两种结果 —— 旋转，会记住自己走过的路", alpha=a)
        arr = vignette(c.finish(), 0.30)
        k = ease_in_cubic(seg(t, 0, 0.7)) * (1 - ease_in_cubic(seg(t, 21.2, 22.0)))
        return (arr * k).astype(np.uint8)
