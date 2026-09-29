# -*- coding: utf-8 -*-
"""v2 场景 S0-S3: 悬念开场 / 历史幕 / 旋转 / 无穷小
节奏: BPM 96 -> 拍 0.625s, 小节 2.5s; 所有切点/字幕点对齐拍或小节
总时长 187.5s = 75 小节; 本文件 375+825+825+675 = 2700帧 = 90s"""
import math
import numpy as np
from PIL import Image, ImageDraw
from engine import (W, H, FPS, Canvas, Background, Camera, vignette, manuscript_bg,
                    CYAN, MAGENTA, GOLD, VIOLET, GREEN, WHITE, ORANGE, BLUE, PINK,
                    seg, ease_in_out, ease_out_cubic, ease_in_cubic, ease_out_back,
                    clamp01, lerp, normalize, axis_angle_matrix,
                    subtitle, subtitle2, chapter_card, big_title,
                    FONT_SANS_BOLD, FONT_SERIF_BOLD, get_font)

BEAT = 0.625
BAR = 2.5


def poly_fill(c, pts, col, alpha):
    """半透明多边形(自动 alpha_composite)"""
    if alpha <= 0.01: return
    a = int(255 * clamp01(alpha))
    pts = [(float(p[0]), float(p[1])) for p in pts]
    if a >= 250:
        c.d.polygon(pts, fill=tuple(col) + (255,)); c._has_ov = True
    else:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x0, y0 = int(min(xs)) - 2, int(min(ys)) - 2
        x1, y1 = int(max(xs)) + 3, int(max(ys)) + 3
        tmp = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
        td = ImageDraw.Draw(tmp)
        td.polygon([(p[0] - x0, p[1] - y0) for p in pts], fill=tuple(col) + (a,))
        c.ov.alpha_composite(tmp, (x0, y0)); c._has_ov = True


# ============================================================
# S0 悬念开场  375帧 / 12.5s (5小节)
# 引言 -> 雪花快闪 -> 魔方4连拧 -> 光点爆开 -> 终问
# ============================================================
class Scene0:
    N = 375

    # 魔方: 面四角(顶点索引 idx=4x+2y+z)
    FACES = [((1, 0, 0), [4, 6, 7, 5]), ((-1, 0, 0), [0, 2, 3, 1]),
             ((0, 1, 0), [2, 6, 7, 3]), ((0, -1, 0), [0, 4, 5, 1]),
             ((0, 0, 1), [1, 5, 7, 3]), ((0, 0, -1), [0, 4, 6, 2])]
    E3 = [np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])]

    def __init__(self):
        self.bg = Background(top=(2, 3, 9), bottom=(4, 5, 15),
                             halo=[(W/2, 470, 700, (20, 26, 70), 0.30)], seed=13, n_stars=110)
        rng = np.random.default_rng(21)
        n = 110
        self.ex_a = rng.uniform(0, 2*np.pi, n)
        self.ex_v = rng.uniform(320, 950, n)
        self.ex_r = rng.uniform(1.2, 2.8, n)
        # 魔方状态: 26 块 [格点pos, 朝向R, 贴纸dict]
        self.face_col = {(1, 0, 0): ORANGE, (-1, 0, 0): BLUE, (0, 1, 0): GOLD,
                         (0, -1, 0): (225, 225, 235), (0, 0, 1): MAGENTA, (0, 0, -1): CYAN}
        self.cub = []
        for x in (-1, 0, 1):
            for y in (-1, 0, 1):
                for z in (-1, 0, 1):
                    if x == y == z == 0: continue
                    p3 = np.array([x, y, z], float)
                    st = {}
                    for fn in self.face_col:
                        out = (p3[0] == fn[0] and fn[0] != 0) or (p3[1] == fn[1] and fn[1] != 0) \
                              or (p3[2] == fn[2] and fn[2] != 0)
                        st[fn] = self.face_col[fn] if out else (28, 32, 46)
                    self.cub.append([p3, np.eye(3), st])
        self.twists = [(0, 1, 1), (1, 0, -1), (2, -1, 1), (0, 0, 1)]
        self.tw_done = 0

    # ---- 魔方拧动状态机(逐帧顺序执行, 分片安全) ----
    def _rubik_commit(self, tt):
        while self.tw_done < 4 and tt >= 6.25 + self.tw_done * BEAT + BEAT - 1e-6:
            ax, layer, dirn = self.twists[self.tw_done]
            R90 = axis_angle_matrix(self.E3[ax], dirn * math.pi / 2)
            for b in self.cub:
                if b[0][ax] == layer:
                    b[0] = np.round(R90 @ b[0])
                    b[1] = R90 @ b[1]
            self.tw_done += 1

    def _draw_rubik(self, c, t):
        tt = t
        half, sp = 0.30, 0.66
        rot_anim = None
        for k, (ax, layer, dirn) in enumerate(self.twists):
            t0 = 6.25 + k * BEAT
            if t0 <= tt < t0 + BEAT:
                rot_anim = (ax, layer, dirn, (tt - t0) / BEAT)
        a_app = seg(t, 6.05, 6.45) * (1 - ease_in_cubic(seg(t, 8.75, 9.05)))
        if a_app <= 0.01: return
        cam = Camera(yaw=0.55 + 0.05 * math.sin(tt * 0.8), pitch=-0.34, dist=5.6)
        cam.center = (W/2, 490)
        cy, sy = math.cos(cam.yaw), math.sin(cam.yaw)
        cp, spp = math.cos(cam.pitch), math.sin(cam.pitch)
        # 相机世界位置: q = R@(p-t)+(0,0,d) => 相机(投影原点)在世界系 c = t - R^T@(0,0,d)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, cp, -spp], [0, spp, cp]])
        campos = -(Rz.T @ Rx.T @ np.array([0, 0, cam.dist]))
        # 块姿态
        blocks = []
        for (p3, R0, st) in self.cub:
            c3 = p3 * sp; Rb = R0
            if rot_anim and p3[rot_anim[0]] == rot_anim[1]:
                Ra = axis_angle_matrix(self.E3[rot_anim[0]], rot_anim[2] * math.pi/2 * ease_in_out(rot_anim[3]))
                c3 = Ra @ c3; Rb = Ra @ R0
            blocks.append((c3, Rb, st))
        # 投影
        allv = []
        for (c3, Rb, st) in blocks:
            vv = np.array([Rb @ np.array([sx, sy, sz]) * half + c3
                           for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            allv.append(vv)
        flat = np.concatenate(allv, 0)
        p2, zz = cam.project(flat, center=cam.center)
        # 远→近绘制 (cam z 越大越远)
        order = sorted(range(len(blocks)), key=lambda b: cam.view(blocks[b][0][None, :])[0, 2], reverse=True)
        sc = 0.9 * a_app
        for bi in order:
            (c3, Rb, st) = blocks[bi]
            vs = allv[bi]; p2b = p2[bi*8:(bi+1)*8]; zzb = zz[bi*8:(bi+1)*8]
            zblk = cam.view(c3[None, :])[0, 2]
            fade = clamp01(1.9 - zblk / 5.6)
            for (nl, idx4) in self.FACES:
                nw = Rb @ np.array(nl, float)
                fcen = vs[idx4].mean(0)
                if (nw @ (campos - fcen)) <= 0: continue
                col = st[nl]
                pts = [tuple(p2b[j]) for j in idx4]
                poly_fill(c, pts, tuple(int(v*sc) for v in col), 0.95 * a_app * (0.45 + 0.55*fade))
                for j in range(4):
                    c.line(pts[j], pts[(j+1) % 4], (10, 12, 22), 2, 0.85 * a_app, glow=0.0)
        # 出现/消失白闪
        if abs(t - 6.25) < 0.22:
            c.add_dot((W/2, 490), 1100, (215, 228, 255), 0.35 * math.exp(-abs(t-6.25)*16))

    # ---- 雪花 ----
    def _snowflake(self, c, cx, cy, R, growth, rot, alpha):
        col = (175, 226, 255)
        for k in range(6):
            a = rot + k * math.pi / 3
            ca, sa = math.cos(a), math.sin(a)
            def P(r, o=0.0):
                return (cx + ca*r - sa*o, cy + sa*r + ca*o)
            L1 = R * growth
            c.line(P(0), P(L1), col, 3, alpha * 0.95, glow=1.3)
            c.dot(P(L1), 3.5, col, alpha, glow=1.4)
            for (r0, ln) in ((0.28, 0.20), (0.50, 0.24), (0.72, 0.19)):
                if growth <= r0: continue
                g2 = clamp01((growth - r0) / 0.22)
                bl = R * ln * g2
                for s in (-1, 1):
                    a2 = a + s * math.pi / 3
                    p1 = P(r0 * R)
                    p2 = (p1[0] + math.cos(a2)*bl, p1[1] + math.sin(a2)*bl)
                    c.line(p1, p2, col, 2, alpha * 0.9, glow=1.0)
                    c.dot(p2, 2.2, col, alpha * 0.85, glow=1.0)
                    if g2 > 0.55 and r0 > 0.3:
                        a3 = a2 + s * math.pi / 3.2
                        p3 = (p2[0] + math.cos(a3)*bl*0.38, p2[1] + math.sin(a3)*bl*0.38)
                        c.line(p2, p3, col, 1, alpha * 0.7, glow=0.7)
        hexr = R * 0.10 * growth
        pts = [(cx + hexr*math.cos(rot + j*math.pi/3), cy + hexr*math.sin(rot + j*math.pi/3))
               for j in range(6)]
        for j in range(6):
            c.line(pts[j], pts[(j+1) % 6], col, 2, alpha * 0.9, glow=1.1)

    def frame(self, i):
        t = i / FPS
        c = Canvas(self.bg.img)
        c.twinkle_stars(self.bg.stars, t)
        # 引子光点 0-0.8
        if t < 0.9:
            k = ease_out_cubic(seg(t, 0, 0.7))
            px = lerp(W*0.18, W/2, k); py = lerp(H*0.82, H*0.46, k)
            c.add_dot((px, py), 5, (200, 220, 255), 0.9)
            c.add_line((lerp(W*0.18, px, 0.5), lerp(H*0.82, py, 0.5)), (px, py), (150, 190, 255), 2.2, 0.22*k)
            c.dot((px, py), 2.5, WHITE, 0.9, glow=1.5)
        # 引言 0.625-3.5
        if t > 0.8:
            a = ease_in_out(seg(t, 0.85, 1.5)) * (1 - ease_in_cubic(seg(t, 3.1, 3.6)))
            c.text((W/2, 440), "有些美，不在形状里。", 62, (222, 230, 250), "mm", a, glow=0.7*a,
                   font_path=FONT_SERIF_BOLD)
            c.text((W/2, 545), "SOME BEAUTY DOES NOT LIVE IN SHAPES", 24, (120, 140, 195), "mm",
                   a*0.85, glow=0.3*a, tracking=8)
            c.dot((W/2, 470), 3.2, (200, 220, 255), a*0.8, glow=1.2)
        # 雪花 3.75-6.25
        if 3.4 < t < 6.5:
            k_in = seg(t, 3.75, 4.375)
            a_out = 1 - ease_in_cubic(seg(t, 6.0, 6.35))
            alpha = clamp01(k_in) * a_out
            if alpha > 0.01:
                self._snowflake(c, W/2, 470, 275, ease_out_cubic(k_in), -0.5 + (t-3.75)*0.55, alpha)
                sub_a = ease_in_out(seg(t, 4.375, 4.9)) * a_out
                subtitle2(c, 900, "雪花 · 六重对称的极美", "SNOWFLAKE — SIXFOLD SYMMETRY", 34, sub_a)
        for tf in (3.7, 6.3):
            if abs(t - tf) < 0.25:
                c.add_dot((W/2, H/2), 1300, (210, 225, 255), 0.30 * math.exp(-abs(t - tf)*14))
        # 魔方 6.25-8.75
        self._rubik_commit(t)
        if 6.05 < t < 9.1:
            self._draw_rubik(c, t)
            if t > 6.9:
                a = ease_in_out(seg(t, 6.875, 7.5)) * (1 - ease_in_cubic(seg(t, 8.55, 8.9)))
                subtitle2(c, 910, "魔方 · 4325 亿亿种姿态", "RUBIK'S CUBE — 43 QUINTILLION STATES", 34, a)
        # 爆开 8.75-10
        if t > 8.75:
            te = t - 8.75
            c.add_dot((W/2, 470), 1300, (225, 235, 255), 0.5 * math.exp(-te * 5.5))
            for j in range(110):
                d = self.ex_v[j] * te * (0.5 + 0.5 * ease_out_cubic(min(1.0, te / 0.5)))
                al = 0.8 * math.exp(-te * 1.9)
                if al < 0.02: continue
                c.add_dot((W/2 + math.cos(self.ex_a[j])*d, 470 + math.sin(self.ex_a[j])*d*0.8),
                          self.ex_r[j], (255, 220, 150), al)
        # 终问 10.0-12.5
        if t > 9.9:
            a = ease_in_out(seg(t, 10.0, 10.9)) * (1 - ease_in_cubic(seg(t, 12.0, 12.5)))
            beat = math.exp(-((t - 10.625) % 1.25) * 3.5) if t > 10.5 else 0.0
            c.text((W/2, 470), "它们有什么共同点？", int(58 + 6*beat), (235, 240, 255), "mm", a,
                   glow=0.9*a + 0.5*beat*a, font_path=FONT_SERIF_BOLD)
            c.text((W/2, 580), "WHAT DO THEY HAVE IN COMMON?", 26, (130, 150, 200), "mm",
                   a*0.85, glow=0.3*a, tracking=9)
        arr = vignette(c.finish(), 0.34)
        k = ease_in_cubic(seg(t, 0, 0.7)) * (1 - ease_in_cubic(seg(t, 12.1, 12.5)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S1 历史幕  825帧 / 27.5s (11小节)
# 1832 伽罗瓦 -> 枪响 -> 群 -> 1873 Sophus Lie -> 李群 -> 章节卡I
# ============================================================
class Scene1:
    N = 825

    def __init__(self):
        self.ms = manuscript_bg(seed=17)
        self.bg = Background(top=(3, 5, 14), bottom=(7, 9, 26),
                             halo=[(W/2, 520, 820, (26, 44, 110), 0.32)], seed=29, n_stars=170)
        rng = np.random.default_rng(41)
        self.lines = []
        y = 290
        while y < 960:
            x0 = rng.uniform(150, 330); x1 = x0 + rng.uniform(280, 880)
            self.lines.append((x0, y, x1, y))
            y += rng.uniform(36, 54)
        self.dust = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.6, 1.6),
                      rng.uniform(0, 6.3)) for _ in range(26)]
        self.redp = [(rng.uniform(0, W), rng.uniform(-80, 460), rng.uniform(12, 40),
                      rng.uniform(0.5, 1.6)) for _ in range(18)]

    def _manuscript(self, c, t):
        kfl = ease_in_cubic(seg(t, 0, 1.1)) * (1 - ease_in_out(seg(t, 12.0, 13.6)))
        for (x0, y0, x1, y1) in self.lines:
            c.line((x0, y0), (x1, y1), (72, 60, 42), 1, 0.20 * kfl, glow=0.0)
        for (x, y, r, ph) in self.dust:
            b = (0.10 + 0.10 * math.sin(t*0.8 + ph)) * kfl
            c.add_dot((x + 8*math.sin(t*0.3 + ph), y + 6*math.cos(t*0.23 + ph)), r, (190, 160, 110), b)
        # 1832
        a1 = ease_out_cubic(seg(t, 0.625, 1.875)) * (1 - ease_in_cubic(seg(t, 5.6, 6.15)))
        c.text((W/2, 350), "1832", 225, (174, 142, 88), "mm", a1*0.92, glow=0.75*a1,
               font_path=FONT_SERIF_BOLD, tracking=12)
        c.line((W/2-330, 505), (W/2+330, 505), (140, 115, 75), 1, 0.4*a1, glow=0.0)
        a0 = ease_in_out(seg(t, 0.9, 1.6)) * (1 - ease_in_cubic(seg(t, 5.6, 6.15)))
        subtitle2(c, 570, "巴黎 · 决斗前夜", "PARIS — THE NIGHT BEFORE A DUEL", 30, a0,
                  color=(206, 186, 146), en_color=(142, 122, 92))
        # 遗言
        a2 = ease_in_out(seg(t, 2.5, 3.125)) * (1 - ease_in_cubic(seg(t, 5.6, 6.15)))
        c.text((W/2, 672), "「我有太多新想法，", 46, (224, 202, 156), "mm", a2, glow=0.5*a2,
               font_path=FONT_SERIF_BOLD)
        a3 = ease_in_out(seg(t, 3.75, 4.375)) * (1 - ease_in_cubic(seg(t, 5.6, 6.15)))
        c.text((W/2, 748), "可是，时间不够了。」", 46, (224, 202, 156), "mm", a3, glow=0.5*a3,
               font_path=FONT_SERIF_BOLD)
        c.text((W/2, 838), "—— 伽罗瓦，20 岁，次日清晨殒命于决斗", 26, (152, 130, 96), "mm",
               a3*0.9, glow=0.22)
        # 渐暗 + 心跳
        dark = ease_in_cubic(seg(t, 5.5, 6.25)) * 0.5
        hb = 0.0
        if 5.0 < t < 6.25:
            ph = (t - 5.0) % 0.625
            hb = math.exp(-ph * 9) * 0.22 * seg(t, 5.0, 5.3)
        kk = clamp01(dark + hb)
        if kk > 0.01:
            c.d.rectangle([0, 0, W, H], fill=(5, 3, 6, int(255*kk))); c._has_ov = True
        # 枪响后红暗落幕 6.25-9.5
        if t > 6.25:
            kr = ease_in_out(seg(t, 6.25, 7.5)) * (1 - ease_in_out(seg(t, 8.75, 9.8)))
            c.d.rectangle([0, 0, W, H], fill=(46, 7, 10, int(150*kr))); c._has_ov = True
            for (x, y, r, v) in self.redp:
                yy = y + (t - 6.25) * v * 36
                al = 0.16 * kr * math.exp(-abs(x - W/2) / 900)
                if 0 < yy < H:
                    c.add_dot((x, yy), r, (120, 22, 26), al)
        # 群句 8.75-11.6
        a4 = ease_in_out(seg(t, 8.75, 9.6)) * (1 - ease_in_cubic(seg(t, 11.0, 11.6)))
        c.text((W/2, 590), "他把短暂一生燃尽的思想，凝成一种语言 —— 群", 48, (234, 216, 172),
               "mm", a4, glow=0.6*a4, font_path=FONT_SERIF_BOLD)
        c.text((W/2, 685), "GROUP · 描述对称的数学", 28, (152, 130, 96), "mm", a4*0.9,
               glow=0.3, tracking=4)

    def _aurora_era(self, c, t):
        c.twinkle_stars(self.bg.stars, t)
        # 极光 13.0-19.5
        if 13.0 <= t < 19.5:
            a = seg(t, 13.0, 14.2) * (1 - seg(t, 18.8, 19.5))
            if a > 0.01:
                for b_ in range(3):
                    x0 = W * (0.05 + 0.33 * b_); ph = b_ * 2.1
                    for j in range(92):
                        x = x0 + j * 15 + 20 * math.sin(t*0.5 + j*0.09 + ph)
                        yt = 195 + 55*math.sin(x*0.004 + t*1.0 + ph) + 26*math.sin(x*0.009 - t*0.6)
                        ln = 250 + 85*math.sin(x*0.006 + t*0.8 + ph*1.3)
                        al = 0.11 * a * (0.6 + 0.4*math.sin(j*0.35 + t*2))
                        if al < 0.02: continue
                        col = (int(40+20*b_), 225, int(160-20*b_))
                        c.gd.line([(x/4, yt/4), (x/4, (yt+ln)/4)], fill=col, width=2)
                        c._has_glow = True
        if 13.9 < t < 16.4:
            a = ease_in_out(seg(t, 13.9, 14.7)) * (1 - ease_in_out(seg(t, 15.8, 16.4)))
            subtitle2(c, 700, "41 年后 · 挪威的寒夜", "FORTY-ONE YEARS LATER — A COLD NIGHT IN NORWAY", 40, a)
        if 16.6 < t < 18.6:
            a = ease_in_out(seg(t, 16.6, 17.3)) * (1 - ease_in_out(seg(t, 18.1, 18.6)))
            subtitle2(c, 700, "数学家索菲斯·李仰望极光，问出一个问题", "SOPHUS LIE WATCHED THE AURORA AND ASKED:", 40, a)
        # 李群落版 18.75
        if t > 18.75:
            kl = ease_out_back(seg(t, 18.75, 19.6), 1.9)
            aL = 1 - ease_in_cubic(seg(t, 22.4, 23.1))
            fl = math.exp(-max(0.0, t - 18.75) * 4)
            c.add_dot((W/2, 420), 900, (255, 225, 150), 0.30 * fl)
            rt = t - 18.75
            if rt < 1.4:
                rr = 80 + ease_out_cubic(rt / 1.4) * 560
                alr = 0.45 * (1 - rt / 1.4)
                n = 64
                pts = [(W/2 + rr*math.cos(j*2*math.pi/n), 420 + rr*math.sin(j*2*math.pi/n)) for j in range(n+1)]
                for j in range(n):
                    c.line(pts[j], pts[j+1], GOLD, 2, alr, glow=0.8)
            c.text((W/2, 420), "李 群", 130, (255, 214, 110), "mm", clamp01(kl)*aL, glow=1.3,
                   font_path=FONT_SERIF_BOLD)
            c.text((W/2, 552), "LIE GROUP", 34, (160, 172, 215), "mm",
                   clamp01(seg(t, 19.2, 19.8))*aL, glow=0.4, tracking=16)
        if 20.4 < t < 22.4:
            a = ease_in_out(seg(t, 20.4, 21.1)) * (1 - ease_in_out(seg(t, 21.9, 22.4)))
            subtitle2(c, 700, "连续对称的心脏，就是李代数", "AT ITS HEART LIES THE LIE ALGEBRA", 40, a)
        # 章节卡 I 22.9-27.5
        if t > 22.6:
            kfade = 1 - ease_in_cubic(seg(t, 26.9, 27.5))
            kbg = ease_in_out(seg(t, 22.6, 23.5)) * kfade
            c.d.rectangle([0, 0, W, H], fill=(3, 4, 10, int(238*kbg))); c._has_ov = True
            chapter_card(c, 1, "李 代 数", "LIE ALGEBRAS", alpha=kfade, y=470, t=t - 22.9)

    def frame(self, i):
        t = i / FPS
        if t < 12.0:
            img = self.ms
        elif t < 14.0:
            k = ease_in_out(seg(t, 12.0, 14.0))
            img = (self.ms.astype(np.float32)*(1-k) + self.bg.img.astype(np.float32)*k).astype(np.uint8)
        else:
            img = self.bg.img
        c = Canvas(img)
        if t < 12.0:
            self._manuscript(c, t)
        else:
            self._aurora_era(c, t)
        # 枪响白闪 18.75 全期叠加
        if t >= 6.25 and t < 7.0:
            f = math.exp(-(t - 6.25) * 7)
            c.d.rectangle([0, 0, W, H], fill=(235, 238, 245, int(235*f)))
            c._has_ov = True
        arr = vignette(c.finish(), 0.30)
        k = ease_in_cubic(seg(t, 0, 1.1)) * (1 - ease_in_cubic(seg(t, 27.1, 27.5)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S2 旋转 SO(3)  825帧 / 27.5s (11小节)   章节卡II + v1球体
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


class Scene2:
    N = 825
    CARD = (2, "旋 转", "ROTATION")

    def __init__(self):
        self.bg = Background(top=(4, 6, 18), bottom=(7, 9, 26),
                             halo=[(W/2, 500, 800, (30, 50, 120), 0.35)], seed=23, n_stars=170)
        self.ball = fibonacci_sphere(300, 1.55)
        self.ico = icosahedron(0.92 * 1.55 / 1.902)
        self.e9 = ico_edges()
        self.markers = np.array([[0.3, 0.9, 0.32], [-0.85, 0.35, 0.4], [0.1, -0.6, 0.79]])
        self.markers = self.markers / np.linalg.norm(self.markers, axis=1, keepdims=True) * 1.55

    def _content(self, c, tt):
        c.twinkle_stars(self.bg.stars, tt)
        cam = Camera(yaw=0.55 + 0.05*math.sin(tt*0.16), pitch=0.32 + 0.08*math.sin(tt*0.11), dist=5.1)
        if tt < 10:
            axis = normalize(np.array([0.35, 1.0, 0.42]))
        elif tt < 19:
            k = ease_in_out((tt-10)/9)
            a0 = np.array([0.35, 1.0, 0.42]); a1 = normalize(np.array([-0.6, 0.7, 0.75]))
            axis = normalize(a0*(1-k) + a1*k + np.array([0, 0.25*math.sin((tt-10)*1.2)*k, 0]))
        else:
            axis = normalize(np.array([-0.6, 0.7, 0.75]))
        ang = 0.85 * tt
        R = axis_angle_matrix(axis, ang)
        pts = self.ball @ R.T
        p2, zz = cam.project(pts)
        order = np.argsort(-zz)
        for dphi, ga in ((0.10, 0.30), (0.20, 0.16), (0.32, 0.08)):
            gp = self.ball @ axis_angle_matrix(axis, ang - dphi).T
            g2, gz = cam.project(gp)
            for j in order:
                al = clamp01(1.7 - gz[j] / 5.2) * ga
                if al < 0.02: continue
                c.dot(g2[j], max(1.0, 5.5/gz[j]), (80, 170, 240), al, glow=0.25)
        for j in order:
            depth = zz[j]
            c.dot(p2[j], max(1.4, 7.6/depth), (105, 200, 255), clamp01(1.7-depth/5.2)*0.68, glow=0.6)
        iv = self.ico @ R.T
        ip2, izz = cam.project(iv)
        for (a, b) in self.e9:
            z = (izz[a] + izz[b]) / 2
            c.line(ip2[a], ip2[b], GOLD, 2, 0.8*clamp01(1.6-z/5.0), glow=0.8)
        for j in range(12):
            c.dot(ip2[j], max(1.5, 24.0/izz[j]), GOLD, 0.9*clamp01(1.6-izz[j]/5), glow=1.1)
        a3 = axis * 1.9
        tip, zt = cam.project_single(a3)
        base, zb = cam.project_single(-a3)
        gl = 0.8 + 0.4*math.sin(tt*2.4)
        c.line(base, tip, PINK, 2, 0.5, glow=gl*0.45)
        d2 = (tip[0]-base[0], tip[1]-base[1])
        L = math.hypot(*d2) + 1e-6
        ux, uy = d2[0]/L, d2[1]/L
        for s in (-1, 1):
            c.line(tip, (tip[0]-ux*22+uy*s*9, tip[1]-uy*22-ux*s*9), PINK, 2, 0.7, glow=gl*0.5)
        c.dot(tip, 3.5, PINK, 0.85, glow=1.0*gl)
        for m in self.markers:
            mm = m @ R.T
            h = float(m @ axis); cen = axis * h
            u = m - axis * h; u = u / (np.linalg.norm(u)+1e-9)
            v = np.cross(axis, u)
            rho = math.sqrt(max(0.0, 1.55**2 - h*h))
            ring = np.array([cen + rho*(u*math.cos(th) + v*math.sin(th)) for th in np.linspace(0, 2*math.pi, 72)])
            r2, rz = cam.project(ring)
            for k in range(71):
                zf = (rz[k] + rz[k+1]) / 2
                c.line(r2[k], r2[k+1], CYAN, 2, 0.22 * clamp01(1.6-zf/5.2), glow=0.3)
            c.dot(cam.project_single(mm)[0], 5.5, CYAN, 0.95, glow=1.5)
        # 双语字幕
        y = H - 165
        msgs = [
            (0.625, 5.0, "旋转 —— 宇宙中最熟悉的对称", "ROTATION — THE MOST FAMILIAR SYMMETRY OF ALL"),
            (6.25, 11.25, "所有的旋转放在一起，构成一个光滑弯曲的空间", "ALL ROTATIONS TOGETHER FORM A SMOOTH, CURVED SPACE"),
            (13.125, 18.125, "数学家称之为 —— 李群", "MATHEMATICIANS CALL IT — A LIE GROUP"),
        ]
        for (a0, a1, zh, en) in msgs:
            if a0 < tt < a1 + 0.9:
                a = ease_in_out(seg(tt, a0, a0+0.625)) * (1 - ease_in_out(seg(tt, a1, a1+0.9)))
                subtitle2(c, y, zh, en, 42, a)
        if tt > 20.8:
            a = ease_out_cubic(seg(tt, 20.8, 22.4)) * (1 - ease_in_cubic(seg(tt, 24.2, 24.9)))
            c.text((W/2, H-235), "SO(3)", 86, GOLD, "mm", a, glow=1.2*a, tracking=8)
            c.text((W/2, H-140), "三维旋转群 · 一个天然的弯曲宇宙", 36, (200, 214, 245), "mm", 0.9*a, glow=0.4*a)
            c.text((W/2, H-88), "THE ROTATION GROUP IN THREE DIMENSIONS", 21, (125, 142, 195), "mm",
                   0.8*a, glow=0.25, tracking=6)
        return axis

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
            arr = vignette(c.finish(), 0.32)
        k = ease_in_cubic(seg(t, 0, 0.5)) * (1 - ease_in_cubic(seg(t, 27.1, 27.5)))
        return (arr * k).astype(np.uint8)


# ============================================================
# S3 无穷小 exp  675帧 / 22.5s (9小节)   章节卡III + 指数映射
# ============================================================
class Scene3:
    N = 675
    CARD = (3, "无 穷 小", "THE INFINITESIMAL")

    def __init__(self):
        self.bg = Background(top=(3, 6, 16), bottom=(6, 9, 26),
                             halo=[(W/2, 520, 780, (30, 60, 130), 0.38)], seed=31, n_stars=160)
        self.ball = fibonacci_sphere(240, 1.7)
        self.P = np.array([0.25, 0.60, 0.76]); self.P = self.P / np.linalg.norm(self.P) * 1.7

    def _content(self, c, tt):
        c.twinkle_stars(self.bg.stars, tt)
        cam = Camera(yaw=0.62 + 0.045*math.sin(tt*0.14), pitch=0.30 + 0.06*math.sin(tt*0.1), dist=4.6)
        R0 = axis_angle_matrix((0, 1, 0), 0.22)
        R0 = R0 @ axis_angle_matrix((0, 0, 1), tt * 0.05)
        ball = self.ball @ R0.T
        P = self.P @ R0.T
        e = normalize(np.cross(np.array([0, 0, 1]), self.P)) @ R0.T
        e = normalize(e - P * (e @ P) / (1.7*1.7))
        vlen = 1.05 * ease_in_out(seg(tt, 1.0, 6.5)) + 0.04*math.sin(tt*2.2)
        p2, zz = cam.project(ball)
        order = np.argsort(-zz)
        for j in order:
            c.dot(p2[j], max(1.3, 7.2/zz[j]), (100, 195, 255), clamp01(1.7-zz[j]/5.2)*0.62, glow=0.5)
        c0, zc = cam.project_single(np.zeros(3))
        rr = 1.7 / zc * cam.f
        for k in range(80):
            th1 = 2*math.pi*k/80; th2 = 2*math.pi*(k+1)/80
            c.line((c0[0]+rr*math.cos(th1), c0[1]+rr*math.sin(th1)*0.995),
                   (c0[0]+rr*math.cos(th2), c0[1]+rr*math.sin(th2)*0.995),
                   (120, 160, 230), 1, 0.20, glow=0.12)
        u = normalize(np.cross(P, e)); v = np.cross(P, u)
        disk = np.array([P + 0.75*(u*math.cos(th) + v*math.sin(th)) for th in np.linspace(0, 2*math.pi, 56)])
        d2, dz = cam.project(disk)
        for k in range(55):
            c.line(d2[k], d2[k+1], VIOLET, 2, 0.5 * clamp01(1.7-dz.mean()/5), glow=0.5)
        theta = vlen / 1.7 * 1.7
        rot_ax = normalize(np.cross(P, e))
        if theta > 0.004:
            arc = np.array([axis_angle_matrix(rot_ax, theta * s) @ P for s in np.linspace(0, 1, 52)])
            a2, az = cam.project(arc)
            for k in range(51):
                zf = (az[k] + az[k+1]) / 2
                c.line(a2[k], a2[k+1], CYAN, 3, 0.95 * clamp01(1.7-zf/5), glow=1.25)
            c.dot(a2[-1], 5.0, CYAN, 0.95, glow=1.6)
        if vlen > 0.02:
            q1, z1 = cam.project_single(P)
            q2, z2 = cam.project_single(P + e * vlen)
            gl = 1.0 + 0.4*math.sin(tt*2.6)
            c.line(q1, q2, GOLD, 3, 0.95, glow=gl)
            c.add_line(q1, q2, GOLD, 2.5, 0.30)
            ddx, ddy = q2[0]-q1[0], q2[1]-q1[1]
            L = math.hypot(ddx, ddy) + 1e-6
            ux, uy = ddx/L, ddy/L
            for s in (-1, 1):
                c.line(q2, (q2[0]-ux*24+uy*s*10, q2[1]-uy*24-ux*s*10), GOLD, 3, 0.9, glow=gl*0.7)
            c.dot(P, 4.5, WHITE, 0.95, glow=1.2)
            c.dot(q2, 3.8, GOLD, 0.95, glow=1.3*gl)
        if tt > 8.0:
            k = ease_out_cubic(seg(tt, 8.0, 9.6)) * (1 - ease_in_cubic(seg(tt, 21.0, 21.9)))
            c.text((W/2, H-245), "exp( θ X )", 66, GOLD, "mm", k, glow=1.1*k, tracking=4)
            c.text((W/2, H-165), "无穷小的旋转，合成有限的旋转", 30, (190, 202, 240), "mm", 0.85*k, glow=0.3*k)
        y = H - 105
        msgs = [
            (0.625, 4.375, "把旋转推到极限 —— 无穷小", "PUSH ROTATION TO ITS LIMIT — THE INFINITESIMAL"),
            (6.25, 11.25, "每一点上，都住着「即将转动」的倾向", "AT EVERY POINT LIVES A TENDENCY: AN INFINITESIMAL ROTATION"),
            (12.5, 16.25, "沿着它出发，掠过球面 —— 又回到旋转", "FOLLOW IT, SWEEP THE SPHERE — AND RETURN TO ROTATIONS"),
            (17.5, 21.2, "这些无穷小旋转的总和，就是李代数 so(3)", "THESE INFINITESIMAL ROTATIONS FORM THE ALGEBRA so(3)"),
        ]
        for (a0, a1, zh, en) in msgs:
            if a0 < tt < a1 + 0.8:
                a = ease_in_out(seg(tt, a0, a0+0.625)) * (1 - ease_in_out(seg(tt, a1, a1+0.8)))
                subtitle2(c, y, zh, en, 40, a)

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
            arr = vignette(c.finish(), 0.32)
        k = ease_in_cubic(seg(t, 0, 0.5)) * (1 - ease_in_cubic(seg(t, 22.1, 22.5)))
        return (arr * k).astype(np.uint8)
