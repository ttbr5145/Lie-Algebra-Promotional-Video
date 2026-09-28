# -*- coding: utf-8 -*-
"""场景 S4-S7: 李括号 / 家族与E8 / 宇宙中的李代数 / 尾声"""
import math
import numpy as np
from engine import (W, H, FPS, Canvas, Background, Camera, vignette,
                    CYAN, MAGENTA, GOLD, VIOLET, GREEN, WHITE, ORANGE, BLUE, PINK,
                    seg, ease_in_out, ease_out_cubic, ease_in_cubic, ease_out_back,
                    clamp01, normalize, axis_angle_matrix,
                    subtitle, FONT_SANS_BOLD, FONT_SERIF_BOLD, get_font)
from e8 import get_e8


# ============================================================
# S4 李括号: [X, Y]  540帧 / 18s
# ============================================================
class Scene4:
    N = 540
    def __init__(self):
        self.bg = Background(
            top=(5, 4, 16), bottom=(10, 7, 24),
            halo=[(W/2, 400, 700, (70, 30, 110), 0.32), (W/2, 800, 600, (25, 45, 110), 0.25)],
            seed=53, n_stars=150)

    def frame(self, i):
        t = i / FPS
        c = Canvas(self.bg.img)
        c.twinkle_stars(self.bg.stars, t)
        # 公式逐字符点亮
        chars = [("[", WHITE, 0.6), ("X", CYAN, 1.1), (",", WHITE, 1.4), ("Y", MAGENTA, 1.9),
                 ("]", WHITE, 2.4), ("=", WHITE, 3.2), ("Z", GOLD, 3.8)]
        for (ch, col, t0) in chars:
            if t > t0:
                k = ease_out_back(seg(t, t0, t0 + 0.55), 1.8)
                c.text((W/2 + (chars.index((ch, col, t0)) - 3) * 110, 300), ch, 150, col,
                       "mm", clamp01(k), glow=1.4*k)
        # 公式闪光
        if t > 4.4:
            f = math.exp(-(t - 4.4) * 1.8)
            c.add_dot((W/2, 300), 420, (255, 220, 130), 0.22*f)
        # 方框回路示意 (commutator square)
        bx, by, S = W/2, 610, 150
        # 四个顶点: A=(0,0) B=(1,0) C=(1,-1) D=(0,-1)  (屏幕y向下)
        A = (bx - S, by + S*0.55); B = (bx + S, by + S*0.55)
        C2 = (bx + S, by - S*0.55); D = (bx - S, by - S*0.55)
        gap = 0.62  # 缺口比例
        # 回路生长: e1: A->B (1.0-3.0s), e2: B->C(3.2-5.2), e3: C->D(5.4-7.4), e4: D->A方向但留缺口(7.6-9.6)
        e1 = ease_in_out(seg(t, 1.0, 3.0)); e2 = ease_in_out(seg(t, 3.3, 5.3))
        e3 = ease_in_out(seg(t, 5.6, 7.6)); e4 = ease_in_out(seg(t, 7.9, 9.9))
        edge_defs = [(A, B, e1, CYAN, "exp(X)"), (B, C2, e2, MAGENTA, "exp(Y)"),
                     (C2, D, e3, CYAN, "exp(−X)"), (D, A, e4*gap, MAGENTA, "exp(−Y)")]
        for (p1, p2, k, col, lab) in edge_defs:
            if k <= 0: continue
            q = (p1[0] + (p2[0]-p1[0])*k, p1[1] + (p2[1]-p1[1])*k)
            c.line(p1, q, col, 4, 0.85, glow=1.0)
            c.dot(q, 4.5, col, 0.95, glow=1.3)
            if k > 0.55:
                mx, my = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2 + (30 if p1[1] == p2[1] else 0)
                c.text((mx, my + (-34 if p1[1] == p2[1] and p1[1] < by else 34)), lab, 30,
                       col, "mm", seg(k, 0.55, 0.95), glow=0.5)
        # 缺口脉冲 -> 李括号
        if t > 10.2:
            k = ease_in_out(seg(t, 10.2, 11.0))
            G = (D[0] + (A[0]-D[0])*gap, D[1] + (A[1]-D[1])*gap)
            c.dot(G, 6.5 + 2.5*math.sin(t*3), GOLD, 0.95*k, glow=1.6)
            ph = math.exp(-(t - 10.2) * 1.1)
            c.add_dot(G, 60 + (t-10.2)*260, GOLD, 0.5*k*ph)
            c.text((G[0] - 80, G[1] + 4), "[X, Y]", 50, GOLD, "rm", k, glow=1.2*k)
        # 字幕
        y = H - 105
        msgs = [
            (0.7, 5.8, "把两种顺序的“差”，精确地测出来"),
            (7.2, 12.6, "这个差异，定义了一种全新的乘法 —— 李括号"),
            (13.6, 17.4, "[X, Y] —— 李代数的心跳"),
        ]
        for (a0, a1, s) in msgs:
            if a0 < t < a1 + 0.8:
                a = ease_in_out(seg(t, a0, a0+0.7)) * (1 - ease_in_out(seg(t, a1, a1+0.8)))
                subtitle(c, y, s, alpha=a, size=46)
        arr = vignette(c.finish(), 0.30)
        k = ease_in_cubic(seg(t, 0, 0.7)) * (1 - ease_in_cubic(seg(t, 17.3, 18.0)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S5 家族与 E8  900帧 / 30s
# ============================================================
FAMILY = [
    # (文字, x, y, t0, color, size)
    ("so(n)", 330, 360, 0.4, CYAN, 50),
    ("su(n)", 960, 290, 0.9, CYAN, 50),
    ("sp(n)", 1590, 360, 1.4, CYAN, 50),
    ("sl(n)", 960, 480, 1.9, CYAN, 44),
    ("G2", 560, 620, 2.6, GOLD, 46),
    ("F4", 1360, 620, 3.1, GOLD, 46),
    ("E6", 420, 720, 3.6, GOLD, 50),
    ("E7", 1500, 720, 4.1, GOLD, 50),
    ("E8", 960, 660, 4.8, GOLD, 72),
]
FAM_LINKS = [(0, 1), (1, 2), (0, 3), (3, 2), (4, 8), (5, 8), (6, 8), (7, 8), (3, 8)]

class Scene5:
    N = 900
    def __init__(self):
        self.bg = Background(
            top=(3, 5, 14), bottom=(7, 8, 24),
            halo=[(W/2, 540, 860, (40, 45, 120), 0.34)],
            seed=61, n_stars=210)
        self.e8 = get_e8()   # (240,2) in [-0.98,0.98]
        # 波前: 按半径排序的点亮时间
        rr = np.linalg.norm(self.e8, axis=1)
        self.born = 8.5 + (rr / rr.max()) * 5.0 + 0.12 * np.random.default_rng(3).uniform(0, 1, 240)
        ang = np.arctan2(self.e8[:, 1], self.e8[:, 0])
        # 颜色: 内金 -> 外青紫
        k = rr / rr.max()
        self.cols = np.stack([255 - 165*k, 200 - 60*k + 20, 90 + 165*k], 1)

    def frame(self, i):
        t = i / FPS
        c = Canvas(self.bg.img)
        c.twinkle_stars(self.bg.stars, t)
        if t < 8.3:
            # ---- A段: 家族星座 ----
            fadeA = 1 - ease_in_cubic(seg(t, 7.6, 8.3))
            for (a, b) in FAM_LINKS:
                (x1, y1, t1, c1, s1) = FAMILY[a][1], FAMILY[a][2], FAMILY[a][3], FAMILY[a][4], FAMILY[a][5]
                (x2, y2, t2, c2, s2) = FAMILY[b][1], FAMILY[b][2], FAMILY[b][3], FAMILY[b][4], FAMILY[b][5]
                ta = max(t1, t2) + 0.5
                if t > ta:
                    k = ease_in_out(seg(t, ta, ta + 0.6)) * 0.16 * fadeA
                    c.line((x1, y1), (x2, y2), (120, 130, 220), 1, k, glow=0.1)
            for (s, x, y, t0, col, sz) in FAMILY:
                if t > t0:
                    k = ease_out_back(seg(t, t0, t0 + 0.7), 1.9)
                    pul = 1 + 0.05 * math.sin(t*2 + x)
                    c.text((x, y), s, int(sz * (0.4 + 0.6*k)), col, "mm", clamp01(k)*fadeA,
                           glow=1.1*k*fadeA)
                    c.dot((x - sz*1.2, y - sz*0.55), 2.2, col, 0.7*k*fadeA, glow=0.8)
            if 5.6 < t:
                a = ease_in_out(seg(t, 5.6, 6.4)) * (1 - ease_in_cubic(seg(t, 7.5, 8.2)))
                subtitle(c, H - 120, "一个庞大而有序的宇宙 —— 经典与例外", alpha=a, size=46)
        # ---- B段: E8 绽放 ----
        if t > 8.0:
            rot = (t - 8.0) * 0.17
            crt, srt = math.cos(rot), math.sin(rot)
            Rm = np.array([[crt, -srt], [srt, crt]])
            breathe = 1 + 0.035 * math.sin((t - 8.0) * 0.7)
            scale = 465 * breathe * ease_out_cubic(seg(t, 8.0, 10.0))
            pts = self.e8 @ Rm.T * scale + np.array([W/2, 540])
            pulse = 0.75 + 0.25 * math.sin(t * 1.1)
            for j in range(240):
                tb = self.born[j]
                if t < tb: continue
                k = ease_out_back(seg(t, tb, tb + 0.5), 2.0)
                r = float(np.linalg.norm(self.e8[j]))
                sz = (2.6 + 3.0 * (1 - r)) * k * (0.9 + 0.25 * pulse)
                col = tuple(int(v) for v in self.cols[j])
                al = clamp01(k) * 0.95
                c.dot(pts[j], sz, col, al, glow=0.9 + 0.6*pulse)
                c.add_dot(pts[j], sz*1.25, col, al*0.42)
            # 中央花蕊光核(三层辐射)
            cg = ease_out_cubic(seg(t, 9.0, 12.0))
            c.add_dot((W/2, 540), 460, (110, 130, 215), 0.09*cg)
            c.add_dot((W/2, 540), 190, (255, 238, 190), 0.45*cg)
            c.add_dot((W/2, 540), 62, (255, 250, 235), 0.7*cg)
            # 字幕
            if 15.5 < t < 22.5:
                a = ease_in_out(seg(t, 15.5, 16.3)) * (1 - ease_in_out(seg(t, 21.7, 22.5)))
                subtitle(c, H - 150, "E8 —— 248 维的对称性殿堂", alpha=a, size=48)
            if 23.5 < t:
                a = ease_in_out(seg(t, 23.5, 24.3)) * (1 - ease_in_out(seg(t, 28.9, 29.7)))
                subtitle(c, H - 150, "240 个根，构成一朵完美的对称之花", alpha=a, size=46)
        # 白闪转场
        if 7.9 < t < 8.6:
            f = math.exp(-abs(t - 8.18) * 9)
            arr_f = None
        arr = vignette(c.finish(), 0.30)
        # 转场白闪叠加
        if 7.9 < t < 8.9:
            f = math.exp(-abs(t - 8.2) * 8) * 0.55
            arr = np.clip(arr.astype(np.float32) + f * 255 * np.array([0.85, 0.9, 1.0], np.float32), 0, 255).astype(np.uint8)
        k = ease_in_cubic(seg(t, 0, 0.7)) * (1 - ease_in_cubic(seg(t, 29.2, 30.0)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S6 宇宙中的李代数  900帧 / 30s
# ============================================================
class Scene6:
    N = 900
    def __init__(self):
        self.bg = Background(
            top=(3, 5, 14), bottom=(6, 8, 22),
            halo=[(W/2, 520, 800, (30, 48, 115), 0.30)],
            seed=71, n_stars=180)
        # 八重道数据: (I3, S, 名称, 点亮时间)
        self.octet = [
            (0.5, 0, "p", 0.8), (-0.5, 0, "n", 1.6),
            (1.0, -1, "Σ+", 2.6), (-1.0, -1, "Σ-", 3.4),
            (0.5, -2, "Ξ0", 4.4), (-0.5, -2, "Ξ-", 5.2),
            (0.0, -1, "Λ", 6.3),
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
        # 光锥填充
        c.line((cx - 620, cy + 620), (cx + 620, cy - 620), (255, 190, 70), 2, 0.16, glow=0.12)
        # 轴
        c.line((cx - 660, cy), (cx + 660, cy), (130, 145, 185), 2, 0.55, glow=0.25)
        c.line((cx, cy + 620), (cx, cy - 620), (130, 145, 185), 2, 0.55, glow=0.25)
        c.text((cx + 690, cy), "x", 34, (150, 165, 205), "mm", 0.8, glow=0.3)
        c.text((cx, cy - 650), "ct", 34, (150, 165, 205), "mm", 0.8, glow=0.3)
        # 光锥
        for s in (-1, 1):
            c.line((cx, cy), (cx + s*600, cy - 600), GOLD, 3, 0.85, glow=1.2)
            c.line((cx, cy), (cx + s*600, cy + 600), GOLD, 2, 0.30, glow=0.4)
        c.add_dot((cx, cy), 300, (255, 200, 90), 0.05 + 0.02*math.sin(tl*2))
        # 双曲线 ct²-x² = a²
        a = 1.45
        xs = np.linspace(-2.7, 2.7, 80)
        cts = np.sqrt(a*a + xs*xs)
        hyp = np.stack([cx + xs*sc, cy - cts*sc], 1)
        for k in range(79):
            c.line(hyp[k], hyp[k+1], (185, 215, 255), 2, 0.32, glow=0.3)
        # 双曲旋转动画 (0.5s起)
        eta = 1.25 * math.sin(tl * 0.6 - 0.4)
        ch, sh = math.cosh(eta), math.sinh(eta)
        # 随动时间轴/空间轴(剪刀)
        tdir = (cx + sh*560*0.99, cy - ch*560*0.99)
        sdir = (cx + ch*560*0.99, cy - sh*560*0.99)
        k = ease_in_out(seg(tl, 0.3, 1.2))
        c.line((cx, cy), tdir, CYAN, 2, 0.5*k, glow=0.5*k)
        c.line((cx, cy), sdir, MAGENTA, 2, 0.35*k, glow=0.35*k)
        # 事件点: 双曲线与时间轴交点 = (a sinh, a cosh)
        ev = (cx + a*sh*sc, cy - a*ch*sc)
        # 拖尾
        for h in range(1, 7):
            eh = eta - h*0.055
            eh = eh if eh != 0 else 0.01
            p = (cx + a*math.sinh(eh)*sc, cy - a*math.cosh(eh)*sc)
            c.add_dot(p, 4 - h*0.4, CYAN, 0.30*(1 - h/7))
        c.dot(ev, 6.5, CYAN, 0.95, glow=1.5)
        c.dot(ev, 2.8, WHITE, 0.95, glow=0.6)
        c.add_dot((cx, cy), 8, WHITE, 0.8)
        c.text((cx + 330, cy - 500), "ct² − x² = 常数", 28, (170, 225, 255), "lm",
               ease_in_out(seg(tl, 2.2, 3.0)), glow=0.4)
        if 5.5 < tl < 9.6:
            aa = ease_in_out(seg(tl, 5.5, 6.3)) * (1 - ease_in_out(seg(tl, 9.0, 9.6)))
            c.text((cx, 200), "狭义相对论 —— 时空的对称：洛伦兹群 SO(3,1)", 44,
                   (215, 228, 255), "mm", aa, glow=0.55)
            c.text((cx, 262), "双曲旋转 · 光锥岿然不动 · 光速不变", 33, (150, 165, 210), "mm",
                   aa*0.9, glow=0.35)
        else:
            aa = ease_in_out(seg(tl, 0.2, 0.9)) * (1 - ease_in_out(seg(tl, 5.2, 5.9)))
            c.text((cx, 200), "时空，也有它的李代数", 44, (215, 228, 255), "mm", aa, glow=0.55)

    # ---- 幕b: 八重道 ----
    def _eightfold(self, c, t, tl):
        cx, ux, uy = W/2, 265, 172
        def pos(I3, S): return (cx + I3*ux, 470 - S*uy)
        # 连线(六边形 + 中心辐条)
        hex_pts = [pos(*o[:2]) for o in self.octet[:6]]
        t_on = 6.8
        for k in range(6):
            a, b = hex_pts[k], hex_pts[(k+1) % 6]
            kk = ease_in_out(seg(tl, t_on + k*0.12, t_on + k*0.12 + 0.5))
            if kk > 0:
                q = (a[0] + (b[0]-a[0])*kk, a[1] + (b[1]-a[1])*kk)
                c.line(a, q, (110, 125, 200), 1, 0.30, glow=0.15)
        # 中心
        lam = pos(0, -1)
        # 粒子
        for (I3, S, name, t0) in self.octet:
            if tl < t0: continue
            k = ease_out_back(seg(tl, t0, t0 + 0.5), 2.0)
            p = pos(I3, S)
            col = GOLD if name == "Λ" else CYAN
            sz = 8 if name == "Λ" else 6.5
            c.dot(p, sz*k, col, clamp01(k)*0.95, glow=1.4)
            c.add_dot(p, sz*1.3, col, 0.25*clamp01(k))
            c.text((p[0], p[1] - 46), name, 40, WHITE, "mm", clamp01(k), glow=0.6)
            c.text((p[0], p[1] + 40), f"I3={I3:g}   S={S:g}", 22, (130, 145, 190), "mm",
                   clamp01(k)*0.8, glow=0.2)
        if tl > 7.5:
            a = ease_in_out(seg(tl, 7.5, 8.3)) * (1 - ease_in_out(seg(tl, 9.2, 9.8)))
            c.text((cx, 205), "粒子物理 —— SU(3) 对称：重子的八重道", 44, (215, 228, 255),
                   "mm", a, glow=0.55)
            c.text((cx, 267), "质子与中子，竟是同一对称性的不同投影", 33, (150, 165, 210),
                   "mm", a*0.9, glow=0.35)
        else:
            a = ease_in_out(seg(tl, 0.1, 0.6))
            c.text((cx, 205), "在粒子世界里，同样旋转着", 44, (215, 228, 255), "mm", a, glow=0.55)

    # ---- 幕c: 诺特定理 ----
    def _noether(self, c, t, tl):
        rows_y = [350, 490, 630, 770]
        lx, rx = W*0.30, W*0.70
        c.text((lx, 235), "连 续 对 称", 42, (160, 200, 255), "mm",
               ease_in_out(seg(tl, 0.2, 0.9)), glow=0.6)
        c.text((rx, 235), "守 恒 律", 42, (255, 210, 110), "mm",
               ease_in_out(seg(tl, 0.2, 0.9)), glow=0.6)
        for j, (sym, con, col) in enumerate(self.nobel):
            t0 = 0.9 + j * 1.85
            if tl < t0: continue
            k1 = ease_out_cubic(seg(tl, t0, t0 + 0.4))
            y = rows_y[j]
            # 左框
            w1, h1 = 300, 74
            self._box(c, (lx, y), w1, h1, col, k1)
            c.text((lx, y), sym, 38, WHITE, "mm", k1, glow=0.5)
            # 线
            k2 = ease_in_out(seg(tl, t0 + 0.45, t0 + 1.05))
            if k2 > 0:
                p1 = (lx + w1/2, y); p2 = (lx + w1/2 + (rx - 160 - (lx + w1/2))*k2, y)
                c.line(p1, p2, col, 3, 0.8, glow=0.9)
                c.dot(p2, 4, col, 0.9, glow=1.2)
            # 右框
            k3 = ease_out_back(seg(tl, t0 + 1.05, t0 + 1.45), 2.0)
            if k3 > 0:
                self._box(c, (rx, y), 260, 74, GOLD, clamp01(k3))
                c.text((rx, y), con, 40, GOLD, "mm", clamp01(k3), glow=1.0*k3)
        if tl > 8.6:
            aa = ease_in_out(seg(tl, 8.6, 9.4)) * (1 - ease_in_out(seg(tl, 9.3, 9.9)))
            c.text((W/2, 150), "诺特定理 —— 每一种连续对称，都守护着一条守恒律", 42,
                   (215, 228, 255), "mm", aa, glow=0.55)

    def _box(self, c, ctr, w, h, col, k):
        x, y = ctr
        pts = [(x-w/2, y-h/2), (x+w/2, y-h/2), (x+w/2, y+h/2), (x-w/2, y+h/2)]
        for j in range(4):
            c.line(pts[j], pts[(j+1) % 4], col, 2, 0.75*k, glow=0.5*k)
        c.add_dot(ctr, w*0.5, col, 0.05*k)

    def frame(self, i):
        t = i / FPS
        c = Canvas(self.bg.img)
        c.twinkle_stars(self.bg.stars, t)
        if t < 10.0:
            self._lorentz(c, t, t)
        elif t < 20.0:
            tl = t - 10.0
            self._eightfold(c, t, tl)
        else:
            tl = t - 20.0
            if tl <= 9.8:
                self._noether(c, t, tl)
            else:
                self._noether(c, t, 9.6)
                # 落幅: 半透明黑罩 + 大字
                k = ease_out_cubic(seg(tl, 9.8, 10.8))
                c.d.rectangle([0, 0, W, H], fill=(2, 3, 12, int(200 * k)))
                c._has_ov = True
                c.text((W/2, 545), "对称，是宇宙的语法", 76, WHITE, "mm", k, glow=1.3*k)
                c.text((W/2, 655), "Symmetry is the grammar of the universe", 27,
                       (140, 160, 210), "mm", k*0.9, glow=0.3*k, tracking=3)
        # 幕间快速白闪
        for tc in (10.0, 20.0):
            if abs(t - tc) < 0.45:
                f = math.exp(-abs(t - tc) * 12) * 0.5
                c.add_dot((W/2, H/2), 1200, (200, 215, 255), f)
        arr = vignette(c.finish(), 0.30)
        k = ease_in_cubic(seg(t, 0, 0.7)) * (1 - ease_in_cubic(seg(t, 29.2, 30.0)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S7 尾声  390帧 / 13s
# ============================================================
class Scene7:
    N = 390
    def __init__(self):
        self.bg = Background(
            top=(3, 4, 12), bottom=(6, 7, 20),
            halo=[(W/2, 500, 720, (45, 48, 125), 0.36)],
            seed=83, n_stars=200)
        self.e8 = get_e8()
        rng = np.random.default_rng(9)
        n = 900
        ang = rng.uniform(0, 2*np.pi, n)
        rad = rng.uniform(700, 1500, n)
        self.p0 = np.stack([W/2 + np.cos(ang)*rad, 480 + np.sin(ang)*rad*0.7], 1)
        self.delay = rng.uniform(0, 4.5, n)
        self.dur = rng.uniform(1.6, 2.6, n)
        # 目标: E8 根位置(屏幕坐标, 中心 960,470, scale 355)
        scale = 355
        order = np.argsort(np.linalg.norm(self.e8, axis=1))
        tgt = self.e8[order][np.arange(n) % 240] * scale + np.array([W/2, 470])
        self.tgt = tgt
        k = np.linalg.norm(self.e8[order][np.arange(n) % 240], axis=1) / 0.98
        self.pcol = np.stack([255 - 165*k, 200 - 50*k, 90 + 165*k], 1)

    def frame(self, i):
        t = i / FPS
        c = Canvas(self.bg.img)
        c.twinkle_stars(self.bg.stars, t)
        # E8 中心旋转
        rot = t * 0.14
        crt, srt = math.cos(rot), math.sin(rot)
        Rm = np.array([[crt, -srt], [srt, crt]])
        scale = 355 * (1 + 0.03*math.sin(t*0.8)) * ease_out_cubic(seg(t, 0, 1.5))
        pts = self.e8 @ Rm.T * scale + np.array([W/2, 470])
        pulse = 0.75 + 0.25*math.sin(t*1.2)
        for j in range(240):
            r = float(np.linalg.norm(self.e8[j]))
            sz = (2.3 + 2.7*(1-r)) * (0.9 + 0.2*pulse)
            col = tuple(int(v) for v in (np.array([255, 200, 90])*(1-r) + np.array([90, 190, 255])*r))
            c.dot(pts[j], sz, col, 0.92, glow=0.9 + 0.4*pulse)
            c.add_dot(pts[j], sz*1.2, col, 0.28)
        c.add_dot((W/2, 470), 430, (110, 130, 215), 0.08)
        c.add_dot((W/2, 470), 165, (255, 238, 190), 0.42)
        # 汇聚粒子
        ft = ease_in_out(np.clip((t - self.delay) / self.dur, 0, 1))
        px = self.p0[:, 0]*(1-ft) + self.tgt[:, 0]*ft
        py = self.p0[:, 1]*(1-ft) + self.tgt[:, 1]*ft
        al = np.where(ft >= 1, 0.0, 0.5 + 0.4*ft)
        for j in range(900):
            if al[j] <= 0.01: continue
            c.add_dot((px[j], py[j]), 1.5, tuple(int(v) for v in self.pcol[j]), float(al[j]))
        # 文字
        if t > 3.2:
            k1 = ease_out_cubic(seg(t, 3.2, 4.6))
            c.text((W/2, 790), "李 代 数", 92, (255, 208, 92), "mm", k1, glow=1.35*k1,
                   font_path=FONT_SERIF_BOLD)
        if t > 4.6:
            k2 = ease_out_cubic(seg(t, 4.6, 6.0))
            c.text((W/2, 905), "用代数，捕捉对称", 44, (222, 232, 255), "mm", k2, glow=0.55*k2)
        if t > 6.0:
            k3 = ease_out_cubic(seg(t, 6.0, 7.4))
            c.text((W/2, 972), "THE ALGEBRA OF SYMMETRY", 24, (120, 140, 195), "mm",
                   k3*0.85, glow=0.35*k3, tracking=13)
        arr = vignette(c.finish(), 0.30)
        k = ease_in_cubic(seg(t, 0, 0.9)) * (1 - ease_in_cubic(seg(t, 11.6, 13.0)))
        return (arr * k).astype(np.uint8)
