# -*- coding: utf-8 -*-
"""v2 场景 S4-S7: 李括号 / 对称之花E8 / 万物皆对称 / 终章beat-cut
节奏: BPM 96, 拍 0.625s, 小节 2.5s; 本文件 750+750+675+750 = 2925帧 = 97.5s
全片 2700+2925 = 5625帧 = 187.5s = 75 小节"""
import math
import os
import numpy as np
from PIL import Image, ImageDraw
from engine import (W, H, FPS, Canvas, Background, Camera, vignette,
                    CYAN, MAGENTA, GOLD, VIOLET, GREEN, WHITE, ORANGE, BLUE, PINK,
                    seg, ease_in_out, ease_out_cubic, ease_in_cubic, ease_out_back,
                    clamp01, normalize, axis_angle_matrix,
                    subtitle, subtitle2, chapter_card,
                    FONT_SANS_BOLD, FONT_SERIF_BOLD, get_font)
from e8 import get_e8
from scenes_a import poly_fill

BEAT = 0.625
BAR = 2.5

# ============================================================
# S4 李括号  750帧 / 25s (10小节)   章节卡IV + 双立方体 + 回路
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


class Scene4:
    N = 750
    CARD = (4, "李 括 号", "THE LIE BRACKET")

    def __init__(self):
        self.bg = Background(top=(4, 5, 15), bottom=(8, 8, 24),
                             halo=[(W*0.27, 430, 520, (28, 52, 120), 0.30),
                                   (W*0.73, 430, 520, (60, 28, 105), 0.30)],
                             seed=47, n_stars=140)

    def _draw_cube(self, c, cam, M, k_in=1.0):
        V = CUBE_V @ M.T
        p2, zz = cam.project(V, center=cam.center)
        for (a, b) in CUBE_E:
            col = edge_color(CUBE_V[a], CUBE_V[b])
            z = (zz[a] + zz[b]) / 2
            fade = clamp01(1.6 - z / 5.0)
            c.line(p2[a], p2[b], col, 3, 0.9*fade*k_in, glow=1.0)
        for j in range(8):
            c.dot(p2[j], max(1.8, 11.0/zz[j]), WHITE, 0.9*clamp01(1.6-zz[j]/5)*k_in, glow=1.1)

    def _content(self, c, tt):
        c.twinkle_stars(self.bg.stars, tt)
        # 双立方体: 拧1 0.625-1.875, 拧2 1.875-3.125
        e1 = ease_in_out(seg(tt, 0.625, 1.875))
        e2 = ease_in_out(seg(tt, 1.875, 3.125))
        ML = axis_angle_matrix((1, 0, 0), e2*math.pi/2) @ axis_angle_matrix((0, 1, 0), e1*math.pi/2)
        MR = axis_angle_matrix((0, 1, 0), e2*math.pi/2) @ axis_angle_matrix((1, 0, 0), e1*math.pi/2)
        k_cube = ease_out_cubic(seg(tt, 0, 0.625)) * (1 - ease_in_cubic(seg(tt, 6.25, 7.5)))
        for (cx, M, tag, tagcol) in ((W*0.27, ML, "先绕 X，再绕 Y", CYAN), (W*0.73, MR, "先绕 Y，再绕 X", MAGENTA)):
            cam = Camera(yaw=0.52, pitch=0.30, dist=5.45)
            cam.center = (cx, 440)
            self._draw_cube(c, cam, M, k_cube)
            if tt > 0.625 and k_cube > 0.02:
                c.text((cx, 160), tag, 38, tagcol, "mm", ease_in_out(seg(tt, 0.625, 1.25))*k_cube, glow=0.7)
        # ≠ 爆闪 4.375
        if tt > 4.375:
            k = ease_out_back(seg(tt, 4.375, 5.0), 2.2)
            flash = math.exp(-max(0.0, tt - 4.375) * 2.4)
            if flash > 0.02:
                c.add_dot((W/2, 430), 520, (180, 200, 255), 0.30*flash)
            c.text((W/2, 430), "≠", 165, WHITE, "mm", clamp01(k)*(1-ease_in_cubic(seg(tt, 6.25, 7.0))), glow=1.5)
        # 公式 5.0-7.5
        if tt > 5.0:
            k = ease_out_cubic(seg(tt, 5.0, 6.25)) * (1 - ease_in_cubic(seg(tt, 7.5, 8.4)))
            c.text((W/2, 810), "Rx(90°) · Ry(90°)   ≠   Ry(90°) · Rx(90°)", 46, (225, 232, 255), "mm", k, glow=0.7*k)
        # [X,Y]=Z 逐字符 9.375-10.625
        chars = [("[", WHITE, 9.375), ("X", CYAN, 9.5), (",", WHITE, 9.65), ("Y", MAGENTA, 9.8),
                 ("]", WHITE, 9.95), ("=", WHITE, 10.15), ("Z", GOLD, 10.3)]
        for j, (ch, col, t0) in enumerate(chars):
            if tt > t0:
                k = ease_out_back(seg(tt, t0, t0 + 0.45), 1.8)
                c.text((W/2 + (j - 3) * 105, 300), ch, 140, col, "mm", clamp01(k), glow=1.4*k)
        if tt > 10.625:
            f = math.exp(-(tt - 10.625) * 1.9)
            c.add_dot((W/2, 300), 430, (255, 220, 130), 0.22*f)
        # commutator 回路 12.0-20.0
        bx, by, S = W/2, 640, 150
        A = (bx - S, by + S*0.55); B = (bx + S, by + S*0.55)
        C2 = (bx + S, by - S*0.55); D = (bx - S, by - S*0.55)
        gap = 0.62
        e1 = ease_in_out(seg(tt, 12.0, 14.0)); e2 = ease_in_out(seg(tt, 14.0, 16.0))
        e3 = ease_in_out(seg(tt, 16.0, 18.0)); e4 = ease_in_out(seg(tt, 18.0, 20.0))
        edge_defs = [(A, B, e1, CYAN, "exp(X)"), (B, C2, e2, MAGENTA, "exp(Y)"),
                     (C2, D, e3, CYAN, "exp(−X)"), (D, A, e4*gap, MAGENTA, "exp(−Y)")]
        for (p1, p2, k, col, lab) in edge_defs:
            if k <= 0: continue
            q = (p1[0] + (p2[0]-p1[0])*k, p1[1] + (p2[1]-p1[1])*k)
            c.line(p1, q, col, 4, 0.85, glow=1.0)
            c.dot(q, 4.5, col, 0.95, glow=1.3)
            if k > 0.55:
                mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
                c.text((mx, my + (-36 if p1[1] == p2[1] and p1[1] < by else 36)), lab, 29,
                       col, "mm", seg(k, 0.55, 0.95), glow=0.5)
        # 缺口脉冲 19.375
        if tt > 19.375:
            k = ease_in_out(seg(tt, 19.375, 20.0))
            G = (D[0] + (A[0]-D[0])*gap, D[1] + (A[1]-D[1])*gap)
            c.dot(G, 6.5 + 2.5*math.sin(tt*3), GOLD, 0.95*k, glow=1.6)
            ph = math.exp(-(tt - 19.375) * 1.1)
            c.add_dot(G, 60 + (tt-19.375)*260, GOLD, 0.5*k*ph)
            c.text((G[0] - 85, G[1] + 6), "[X, Y]", 48, GOLD, "rm", k, glow=1.2*k)
            if tt > 20.625:
                ka = ease_in_out(seg(tt, 20.625, 21.25)) * (1 - ease_in_cubic(seg(tt, 23.0, 23.6)))
                c.text((W/2, 790), "[X, Y] —— 李代数的心跳", 34, (255, 214, 110), "mm", ka, glow=0.9*ka)
        # 双语字幕
        y = H - 108
        msgs = [
            (0.3125, 2.8125, "旋转有一个秘密：顺序，很重要", "ROTATION KEEPS A SECRET: ORDER MATTERS", 40),
            (3.4375, 5.625, "两种顺序，两种结果", "TWO ORDERS, TWO OUTCOMES", 40),
            (6.875, 9.0625, "那么，这个「差」，能被精确测量吗？", "CAN THIS DIFFERENCE BE MEASURED PRECISELY?", 40),
            (19.875, 23.0, "这个差，定义了一种全新的乘法 —— 李括号", "THIS GAP DEFINES A NEW PRODUCT — THE LIE BRACKET", 40),
        ]
        for (a0, a1, zh, en, sz) in msgs:
            if a0 < tt < a1 + 0.8:
                a = ease_in_out(seg(tt, a0, a0+0.625)) * (1 - ease_in_out(seg(tt, a1, a1+0.8)))
                subtitle2(c, y, zh, en, sz, a)

    def frame(self, i):
        t = i / FPS
        tt = t - 2.5
        if tt < 0:
            c = Canvas(self.bg.img)
            chapter_card(c, *self.CARD, alpha=1 - ease_in_cubic(seg(t, 1.9, 2.5)), y=470, t=t)
            arr = c.finish()
        else:
            c = Canvas(self.bg.img)
            self._content(c, tt)
            arr = vignette(c.finish(), 0.30)
        k = ease_in_cubic(seg(t, 0, 0.5)) * (1 - ease_in_cubic(seg(t, 24.6, 25.0)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S5 对称之花: 家族 + E8  750帧 / 25s (10小节)   章节卡V
# ============================================================
FAMILY = [
    ("so(n)", 330, 360, 0.625, CYAN, 50),
    ("su(n)", 960, 290, 1.25, CYAN, 50),
    ("sp(n)", 1590, 360, 1.875, CYAN, 50),
    ("sl(n)", 960, 480, 2.5, CYAN, 44),
    ("G2", 560, 620, 3.125, GOLD, 46),
    ("F4", 1360, 620, 3.75, GOLD, 46),
    ("E6", 420, 720, 4.375, GOLD, 50),
    ("E7", 1500, 720, 5.0, GOLD, 50),
    ("E8", 960, 660, 5.625, GOLD, 72),
]
FAM_LINKS = [(0, 1), (1, 2), (0, 3), (3, 2), (4, 8), (5, 8), (6, 8), (7, 8), (3, 8)]


class Scene5:
    N = 750
    CARD = (5, "对称之花", "FLOWERS OF SYMMETRY")

    def __init__(self):
        self.bg = Background(top=(3, 5, 14), bottom=(7, 8, 24),
                             halo=[(W/2, 540, 860, (40, 45, 120), 0.34)], seed=61, n_stars=210)
        self.e8 = get_e8()
        rr = np.linalg.norm(self.e8, axis=1)
        self.born = 8.5 + (rr / rr.max()) * 5.0 + 0.12 * np.random.default_rng(3).uniform(0, 1, 240)
        k = rr / rr.max()
        self.cols = np.stack([255 - 165*k, 200 - 60*k + 20, 90 + 165*k], 1)

    def _content(self, c, tt):
        c.twinkle_stars(self.bg.stars, tt)
        if tt < 8.3:
            fadeA = 1 - ease_in_cubic(seg(tt, 7.5, 8.2))
            for (a, b) in FAM_LINKS:
                (x1, y1, t1, c1, s1) = FAMILY[a][1:]
                (x2, y2, t2, c2, s2) = FAMILY[b][1:]
                ta = max(t1, t2) + 0.4
                if tt > ta:
                    k = ease_in_out(seg(tt, ta, ta + 0.5)) * 0.16 * fadeA
                    c.line((x1, y1), (x2, y2), (120, 130, 220), 1, k, glow=0.1)
            for (s, x, y, t0, col, sz) in FAMILY:
                if tt > t0:
                    k = ease_out_back(seg(tt, t0, t0 + 0.6), 1.9)
                    c.text((x, y), s, int(sz * (0.4 + 0.6*k)), col, "mm", clamp01(k)*fadeA, glow=1.1*k*fadeA)
                    c.dot((x - sz*1.2, y - sz*0.55), 2.2, col, 0.7*k*fadeA, glow=0.8)
            if 5.0 < tt < 8.0:
                a = ease_in_out(seg(tt, 5.0, 5.625)) * (1 - ease_in_cubic(seg(tt, 7.4, 8.0)))
                subtitle2(c, H - 130, "一个庞大而有序的宇宙 —— 经典与例外", "A VAST, ORDERED UNIVERSE — CLASSICAL AND EXCEPTIONAL", 38, a)
        if tt > 8.0:
            rot = (tt - 8.0) * 0.17
            crt, srt = math.cos(rot), math.sin(rot)
            Rm = np.array([[crt, -srt], [srt, crt]])
            breathe = 1 + 0.035 * math.sin((tt - 8.0) * 0.7)
            scale = 465 * breathe * ease_out_cubic(seg(tt, 8.0, 10.0))
            pts = self.e8 @ Rm.T * scale + np.array([W/2, 540])
            pulse = 0.75 + 0.25 * math.sin(tt * 1.1)
            for j in range(240):
                tb = self.born[j]
                if tt < tb: continue
                k = ease_out_back(seg(tt, tb, tb + 0.5), 2.0)
                r = float(np.linalg.norm(self.e8[j]))
                sz = (2.6 + 3.0 * (1 - r)) * k * (0.9 + 0.25 * pulse)
                col = tuple(int(v) for v in self.cols[j])
                c.dot(pts[j], sz, col, clamp01(k)*0.95, glow=0.9 + 0.6*pulse)
                c.add_dot(pts[j], sz*1.25, col, clamp01(k)*0.42)
            cg = ease_out_cubic(seg(tt, 9.0, 12.0))
            c.add_dot((W/2, 540), 460, (110, 130, 215), 0.09*cg)
            c.add_dot((W/2, 540), 190, (255, 238, 190), 0.45*cg)
            c.add_dot((W/2, 540), 62, (255, 250, 235), 0.7*cg)
            if 15.625 < tt < 19.375:
                a = ease_in_out(seg(tt, 15.625, 16.25)) * (1 - ease_in_out(seg(tt, 18.75, 19.375)))
                subtitle2(c, H - 155, "E8 —— 248 维的对称性殿堂", "E8 — A 248-DIMENSIONAL PALACE OF SYMMETRY", 44, a)
            if 20.0 < tt:
                a = ease_in_out(seg(tt, 20.0, 20.625)) * (1 - ease_in_out(seg(tt, 23.1, 23.7)))
                subtitle2(c, H - 155, "240 个根，构成一朵完美的对称之花", "240 ROOTS — ONE PERFECT FLOWER OF SYMMETRY", 40, a)
        if 7.8125 < tt < 8.9:
            f = math.exp(-abs(tt - 8.125) * 8) * 0.55
            arr = c.finish()
            arr = np.clip(arr.astype(np.float32) + f * 255 * np.array([0.85, 0.9, 1.0], np.float32), 0, 255).astype(np.uint8)
            return arr
        return None

    def frame(self, i):
        t = i / FPS
        tt = t - 2.5
        if tt < 0:
            c = Canvas(self.bg.img)
            chapter_card(c, *self.CARD, alpha=1 - ease_in_cubic(seg(t, 1.9, 2.5)), y=470, t=t)
            arr = c.finish()
        else:
            c = Canvas(self.bg.img)
            r = self._content(c, tt)
            arr = r if r is not None else vignette(c.finish(), 0.30)
        k = ease_in_cubic(seg(t, 0, 0.5)) * (1 - ease_in_cubic(seg(t, 24.6, 25.0)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S6 万物皆对称: 物理三幕  675帧 / 22.5s (9小节)   章节卡VI
# tt: 幕a 0-6.25 洛伦兹 / 幕b 6.25-12.5 八重道 / 幕c 12.5-20 诺特+落幅
# ============================================================
class Scene6:
    N = 675
    CARD = (6, "万物皆对称", "SYMMETRY RULES THE WORLD")

    def __init__(self):
        self.bg = Background(top=(3, 5, 14), bottom=(6, 8, 22),
                             halo=[(W/2, 520, 800, (30, 48, 115), 0.30)], seed=71, n_stars=180)
        self.octet = [
            (0.5, 0, "p", 6.875), (-0.5, 0, "n", 7.5),
            (1.0, -1, "Σ+", 8.125), (-1.0, -1, "Σ-", 8.75),
            (0.5, -2, "Ξ0", 9.375), (-0.5, -2, "Ξ-", 10.0),
            (0.0, -1, "Λ", 10.625),
        ]
        self.nobel = [
            ("时间平移", "能量", CYAN),
            ("空间平移", "动量", MAGENTA),
            ("空间旋转", "角动量", GREEN),
            ("相位变换", "电荷", GOLD),
        ]

    # ---- 幕a: 洛伦兹群 ----
    def _lorentz(self, c, t, tl):
        cx, cy, sc = W/2, 560, 168
        c.line((cx - 660, cy), (cx + 660, cy), (130, 145, 185), 2, 0.55, glow=0.25)
        c.line((cx, cy + 620), (cx, cy - 620), (130, 145, 185), 2, 0.55, glow=0.25)
        c.text((cx + 690, cy), "x", 34, (150, 165, 205), "mm", 0.8, glow=0.3)
        c.text((cx, cy - 650), "ct", 34, (150, 165, 205), "mm", 0.8, glow=0.3)
        for s in (-1, 1):
            c.line((cx, cy), (cx + s*600, cy - 600), GOLD, 3, 0.85, glow=1.2)
            c.line((cx, cy), (cx + s*600, cy + 600), GOLD, 2, 0.30, glow=0.4)
        c.add_dot((cx, cy), 300, (255, 200, 90), 0.05 + 0.02*math.sin(tl*2))
        a = 1.45
        xs = np.linspace(-2.7, 2.7, 80)
        cts = np.sqrt(a*a + xs*xs)
        hyp = np.stack([cx + xs*sc, cy - cts*sc], 1)
        for k in range(79):
            c.line(hyp[k], hyp[k+1], (185, 215, 255), 2, 0.32, glow=0.3)
        eta = 1.25 * math.sin(tl * 1.0 - 0.4)
        ch, sh = math.cosh(eta), math.sinh(eta)
        tdir = (cx + sh*560*0.99, cy - ch*560*0.99)
        sdir = (cx + ch*560*0.99, cy - sh*560*0.99)
        k = ease_in_out(seg(tl, 0.2, 1.0))
        c.line((cx, cy), tdir, CYAN, 2, 0.5*k, glow=0.5*k)
        c.line((cx, cy), sdir, MAGENTA, 2, 0.35*k, glow=0.35*k)
        ev = (cx + a*sh*sc, cy - a*ch*sc)
        for h in range(1, 7):
            eh = eta - h*0.05
            p = (cx + a*math.sinh(eh)*sc, cy - a*math.cosh(eh)*sc)
            c.add_dot(p, 4 - h*0.4, CYAN, 0.30*(1 - h/7))
        c.dot(ev, 6.5, CYAN, 0.95, glow=1.5)
        c.dot(ev, 2.8, WHITE, 0.95, glow=0.6)
        c.add_dot((cx, cy), 8, WHITE, 0.8)
        if 0.625 < tl < 2.5:
            aa = ease_in_out(seg(tl, 0.625, 1.25)) * (1 - ease_in_out(seg(tl, 2.2, 2.6)))
            subtitle2(c, 195, "时空，也有它的李代数", "SPACETIME HAS ITS OWN LIE ALGEBRA", 42, aa)
        elif 3.125 < tl:
            aa = ease_in_out(seg(tl, 3.125, 3.75)) * (1 - ease_in_out(seg(tl, 5.9, 6.25)))
            subtitle2(c, 195, "狭义相对论 —— 洛伦兹群 SO(3,1)：光锥岿然不动", "SPECIAL RELATIVITY — LORENTZ GROUP SO(3,1)", 40, aa)

    # ---- 幕b: 八重道 ----
    def _eightfold(self, c, t, tl):
        cx, ux, uy = W/2, 265, 172
        def pos(I3, S): return (cx + I3*ux, 500 - S*uy)
        hex_pts = [pos(*o[:2]) for o in self.octet[:6]]
        for k in range(6):
            a, b = hex_pts[k], hex_pts[(k+1) % 6]
            kk = ease_in_out(seg(tl, 4.2 + k*0.12, 4.2 + k*0.12 + 0.5))
            if kk > 0:
                q = (a[0] + (b[0]-a[0])*kk, a[1] + (b[1]-a[1])*kk)
                c.line(a, q, (110, 125, 200), 1, 0.30, glow=0.15)
        for (I3, S, name, t0) in self.octet:
            if tl < t0 - 6.25: continue
            tq = tl + 6.25
            k = ease_out_back(seg(tq, t0, t0 + 0.5), 2.0)
            p = pos(I3, S)
            col = GOLD if name == "Λ" else CYAN
            sz = 8 if name == "Λ" else 6.5
            c.dot(p, sz*k, col, clamp01(k)*0.95, glow=1.4)
            c.add_dot(p, sz*1.3, col, 0.25*clamp01(k))
            c.text((p[0], p[1] - 46), name, 40, WHITE, "mm", clamp01(k), glow=0.6)
        if tl + 6.25 < 9.4:
            a = ease_in_out(seg(tl, 0.1, 0.6)) * (1 - ease_in_out(seg(tl, 2.7, 3.1)))
            subtitle2(c, 200, "粒子世界里，同样旋转着对称", "IN THE PARTICLE WORLD, SYMMETRY SPINS TOO", 40, a)
        else:
            a = ease_in_out(seg(tl, 3.2, 3.8)) * (1 - ease_in_out(seg(tl, 5.9, 6.25)))
            subtitle2(c, 200, "八重道 —— 质子与中子，同一种 SU(3) 对称的投影", "THE EIGHTFOLD WAY — PROJECTIONS OF SU(3)", 38, a)

    # ---- 幕c: 诺特定理 ----
    def _noether(self, c, t, tl):
        rows_y = [400, 520, 640, 760]
        lx, rx = W*0.30, W*0.70
        c.text((lx, 280), "连 续 对 称", 40, (160, 200, 255), "mm", ease_in_out(seg(tl, 0.2, 0.8)), glow=0.6)
        c.text((rx, 280), "守 恒 律", 40, (255, 210, 110), "mm", ease_in_out(seg(tl, 0.2, 0.8)), glow=0.6)
        for j, (sym, con, col) in enumerate(self.nobel):
            t0 = 0.4 + j * 1.25
            if tl < t0: continue
            k1 = ease_out_cubic(seg(tl, t0, t0 + 0.35))
            y = rows_y[j]
            w1, h1 = 300, 70
            self._box(c, (lx, y), w1, h1, col, k1)
            c.text((lx, y), sym, 36, WHITE, "mm", k1, glow=0.5)
            k2 = ease_in_out(seg(tl, t0 + 0.35, t0 + 0.8))
            if k2 > 0:
                p1 = (lx + w1/2, y); p2 = (lx + w1/2 + (rx - 160 - (lx + w1/2))*k2, y)
                c.line(p1, p2, col, 3, 0.8, glow=0.9)
                c.dot(p2, 4, col, 0.9, glow=1.2)
            k3 = ease_out_back(seg(tl, t0 + 0.8, t0 + 1.15), 2.0)
            if k3 > 0:
                self._box(c, (rx, y), 260, 70, GOLD, clamp01(k3))
                c.text((rx, y), con, 38, GOLD, "mm", clamp01(k3), glow=1.0*k3)
        if 0.625 < tl < 5.6:
            aa = ease_in_out(seg(tl, 0.625, 1.25)) * (1 - ease_in_out(seg(tl, 5.0, 5.6)))
            subtitle2(c, 165, "诺特定理 —— 每一种连续对称，都守护一条守恒律", "NOETHER'S THEOREM — EVERY SYMMETRY GUARDS A CONSERVATION LAW", 36, aa)

    def _box(self, c, ctr, w, h, col, k):
        x, y = ctr
        pts = [(x-w/2, y-h/2), (x+w/2, y-h/2), (x+w/2, y+h/2), (x-w/2, y+h/2)]
        for j in range(4):
            c.line(pts[j], pts[(j+1) % 4], col, 2, 0.75*k, glow=0.5*k)
        c.add_dot(ctr, w*0.5, col, 0.05*k)

    def _content(self, c, tt):
        c.twinkle_stars(self.bg.stars, tt)
        if tt < 6.25:
            self._lorentz(c, tt, tt)
        elif tt < 12.5:
            self._eightfold(c, tt, tt - 6.25)
        else:
            tl = tt - 12.5
            if tl <= 5.8:
                self._noether(c, tt, tl)
            else:
                self._noether(c, tt, 5.8)
                # 落幅
                k = ease_out_cubic(seg(tl, 5.8, 6.6))
                c.d.rectangle([0, 0, W, H], fill=(2, 3, 12, int(205 * k)))
                c._has_ov = True
                c.text((W/2, 540), "对称，是宇宙的语法", 74, WHITE, "mm", k, glow=1.3*k,
                       font_path=FONT_SERIF_BOLD)
                c.text((W/2, 655), "SYMMETRY IS THE GRAMMAR OF THE UNIVERSE", 26,
                       (140, 160, 210), "mm", k*0.9, glow=0.3*k, tracking=4)
        # 幕间白闪 6.25 / 12.5
        for tc in (6.25, 12.5):
            if abs(tt - tc) < 0.42:
                c.add_dot((W/2, H/2), 1200, (200, 215, 255), 0.5 * math.exp(-abs(tt - tc) * 12))

    def frame(self, i):
        t = i / FPS
        tt = t - 2.5
        if tt < 0:
            c = Canvas(self.bg.img)
            chapter_card(c, *self.CARD, alpha=1 - ease_in_cubic(seg(t, 1.9, 2.5)), y=470, t=t)
            arr = c.finish()
        else:
            c = Canvas(self.bg.img)
            self._content(c, tt)
            arr = vignette(c.finish(), 0.30)
        k = ease_in_cubic(seg(t, 0, 0.5)) * (1 - ease_in_cubic(seg(t, 22.1, 22.5)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S7 终章: beat-cut 混剪 + 收敛爆炸 + 落版  750帧 / 25s (10小节)
# tt: 0-10 16连切(每拍一切) / 10-12.5 汇聚 / 12.5-15 E8绽放
#     15-17.5 蓄力 / 17.5 爆炸 / 18-24.5 落版 / 24-25 淡出
# ============================================================
SLICE_SPEC = [
    ("s0", 150), ("s4", 590), ("s5", 405), ("s2", 130),
    ("s1", 62),  ("s6", 430), ("s0", 255), ("s3", 200),
    ("s1", 300), ("s6", 120), ("s5", 235), ("s2", 420),
    ("s6", 585), ("s4", 210), ("s3", 640), ("s1", 790),
]
SLICE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunks", "finale_slices.npy")


def build_slices():
    from scenes_a import Scene0, Scene1, Scene2, Scene3
    cls_map = {"s0": Scene0, "s1": Scene1, "s2": Scene2, "s3": Scene3,
               "s4": Scene4, "s5": Scene5, "s6": Scene6}
    out = []
    for (name, fi) in SLICE_SPEC:
        Cls = cls_map[name]
        sc = Cls()
        out.append(sc.frame(min(fi, Cls.N - 1)))
        del sc
    return np.stack(out)


class Scene7:
    N = 750

    def __init__(self):
        self.bg = Background(top=(3, 4, 12), bottom=(6, 7, 20),
                             halo=[(W/2, 470, 720, (45, 48, 125), 0.36)], seed=83, n_stars=200)
        self.e8 = get_e8()
        # 切片缓存
        if os.path.exists(SLICE_CACHE):
            self.slices = np.load(SLICE_CACHE)
        else:
            os.makedirs(os.path.dirname(SLICE_CACHE), exist_ok=True)
            self.slices = build_slices()
            np.save(SLICE_CACHE, self.slices)
        # 汇聚粒子
        rng = np.random.default_rng(9)
        n = 900
        ang = rng.uniform(0, 2*np.pi, n)
        rad = rng.uniform(700, 1500, n)
        self.p0 = np.stack([W/2 + np.cos(ang)*rad, 470 + np.sin(ang)*rad*0.7], 1)
        self.delay = rng.uniform(0, 1.3, n)
        self.dur = rng.uniform(0.7, 1.3, n)
        scale = 355
        order = np.argsort(np.linalg.norm(self.e8, axis=1))
        tgt = self.e8[order][np.arange(n) % 240] * scale + np.array([W/2, 470])
        self.tgt = tgt
        kk = np.linalg.norm(self.e8[order][np.arange(n) % 240], axis=1) / 0.98
        self.pcol = np.stack([255 - 165*kk, 200 - 50*kk, 90 + 165*kk], 1)
        # 爆炸粒子(240根外冲)
        rr = np.linalg.norm(self.e8, axis=1)
        self.boom_dir = self.e8 / np.maximum(rr[:, None], 1e-6)
        self.boom_v = 500 + 900 * (rr / rr.max())

    # ---- beat-cut 混剪 ----
    def _beatcut(self, t):
        si = min(15, int(t / BEAT))
        k = (t - si * BEAT) / BEAT
        base = self.slices[si]
        s = 1.0 + 0.16 * ease_in_out(k)
        im = Image.fromarray(base)
        w2, h2 = int(W / s), int(H / s)
        x0, y0 = (W - w2) // 2, (H - h2) // 2
        im = im.crop((x0, y0, x0 + w2, y0 + h2)).resize((W, H), Image.BILINEAR)
        arr = np.asarray(im, np.float32)
        sh = int(7 * (1 - k)) + 1
        out = arr.copy()
        out[..., 0] = np.roll(arr[..., 0], sh, axis=1)
        out[..., 2] = np.roll(arr[..., 2], -sh, axis=1)
        if k < 0.14:
            out += (1 - k / 0.14) * 0.35 * 255
        out = vignette(np.clip(out, 0, 255).astype(np.uint8), 0.30)
        return out

    def _e8_bloom(self, c, t, t0, growth_speed=1.0):
        rot = (t - t0) * 0.14
        crt, srt = math.cos(rot), math.sin(rot)
        Rm = np.array([[crt, -srt], [srt, crt]])
        scale = 355 * ease_out_cubic(seg(t, t0, t0 + 1.5))
        pts = self.e8 @ Rm.T * scale + np.array([W/2, 470])
        pulse = 0.75 + 0.25 * math.sin(t * 1.2)
        for j in range(240):
            tb = t0 + 0.2 + (np.linalg.norm(self.e8[j]) / 0.98) * 2.0 * growth_speed
            if t < tb: continue
            k = ease_out_back(seg(t, tb, tb + 0.45), 2.0)
            r = float(np.linalg.norm(self.e8[j]))
            sz = (2.3 + 2.7*(1-r)) * k * (0.9 + 0.2*pulse)
            col = tuple(int(v) for v in (np.array([255, 200, 90])*(1-r) + np.array([90, 190, 255])*r))
            c.dot(pts[j], sz, col, 0.92*clamp01(k), glow=0.9 + 0.4*pulse)
        c.add_dot((W/2, 470), 430, (110, 130, 215), 0.08)
        c.add_dot((W/2, 470), 165, (255, 238, 190), 0.42)
        return pts

    def frame(self, i):
        t = i / FPS
        # ---- 段1: beat-cut 0-10 ----
        if t < 10.0:
            return self._beatcut(t)
        c = Canvas(self.bg.img)
        c.twinkle_stars(self.bg.stars, t)
        # ---- 段2: 汇聚 10-12.5 ----
        if t < 12.5:
            ft = ease_in_out(np.clip((t - 10.0 - self.delay) / self.dur, 0, 1))
            px = self.p0[:, 0]*(1-ft) + self.tgt[:, 0]*ft
            py = self.p0[:, 1]*(1-ft) + self.tgt[:, 1]*ft
            al = np.where(ft >= 1, 0.0, 0.5 + 0.4*ft)
            for j in range(900):
                if al[j] <= 0.01: continue
                c.add_dot((px[j], py[j]), 1.5, tuple(int(v) for v in self.pcol[j]), float(al[j]))
            c.add_dot((W/2, 470), 90 + 200*ease_in_out(seg(t, 12.0, 12.5)), (200, 210, 255), 0.25)
        # ---- 段3: E8 绽放 12.5-15 ----
        if 12.5 <= t < 17.5:
            self._e8_bloom(c, t, 12.5)
            # 蓄力脉动 15.0 / 16.25 / 17.125
            if t > 15.0:
                ph = t - 15.0
                cyc = 1.25 if ph < 1.25 else (1.0 if ph < 2.25 else 0.625)
                off = 0.0 if ph < 1.25 else (1.25 if ph < 2.25 else 2.25)
                beat = math.exp(-(ph - off) * 5)
                c.add_dot((W/2, 470), 500, (150, 140, 230), 0.10 * beat)
        # ---- 段4: 爆炸 17.5-19 ----
        if t >= 17.5:
            te = t - 17.5
            flash = math.exp(-te * 5.0)
            if flash > 0.01:
                c.add_dot((W/2, 470), 1600, (235, 240, 255), min(0.95, flash))
            rot = te * 0.10
            crt, srt = math.cos(rot), math.sin(rot)
            Rm = np.array([[crt, -srt], [srt, crt]])
            pts = self.e8 @ Rm.T * 355 + np.array([W/2, 470])
            fade = math.exp(-te * 1.1)
            for j in range(240):
                d = self.boom_v[j] * te * ease_out_cubic(min(1.0, te / 0.6))
                p = (W/2 + self.boom_dir[j][0]*d, 470 + self.boom_dir[j][1]*d*0.85)
                al = 0.85 * fade
                if al < 0.02: continue
                r = float(np.linalg.norm(self.e8[j]))
                col = tuple(int(v) for v in (np.array([255, 210, 100])*(1-r) + np.array([120, 190, 255])*r))
                c.add_dot(p, 2.4 + 2.0*(1-r), col, al)
            c.add_dot((W/2, 470), 260 * ease_out_cubic(min(1.0, te / 0.8)), (255, 238, 190), 0.5*fade)
        # ---- 落版 18-25 ----
        if t > 18.0:
            k1 = ease_out_cubic(seg(t, 18.2, 19.4))
            kdark = ease_in_out(seg(t, 17.9, 19.2)) * (1 - ease_in_cubic(seg(t, 23.9, 25.0)))
            c.d.rectangle([0, 0, W, H], fill=(2, 3, 10, int(175*kdark)))
            c._has_ov = True
            c.text((W/2, 470), "李 代 数", 96, (255, 208, 92), "mm", k1, glow=1.35*k1,
                   font_path=FONT_SERIF_BOLD)
            if t > 19.4:
                k2 = ease_out_cubic(seg(t, 19.4, 20.4))
                c.text((W/2, 605), "用代数，捕捉对称", 44, (222, 232, 255), "mm", k2, glow=0.55*k2)
            if t > 20.4:
                k3 = ease_out_cubic(seg(t, 20.4, 21.2))
                c.text((W/2, 680), "THE ALGEBRA OF SYMMETRY", 24, (120, 140, 195), "mm",
                       k3*0.85, glow=0.35*k3, tracking=13)
            if t > 21.4:
                k4 = ease_out_cubic(seg(t, 21.4, 22.4))
                subtitle2(c, 800, "献给每一个看见对称的人", "FOR EVERYONE WHO HAS SEEN SYMMETRY", 30, k4*0.9)
        arr = vignette(c.finish(), 0.30)
        k = 1 - ease_in_cubic(seg(t, 24.3, 25.0))
        return (arr * k).astype(np.uint8)
