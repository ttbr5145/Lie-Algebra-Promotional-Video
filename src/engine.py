# -*- coding: utf-8 -*-
"""
李代数宣传片 - 渲染引擎
分层架构: 静态深空背景 + RGBA叠加层(实心几何/文字) + 加法发光层(粒子/光轨) + 1/4分辨率辉光层
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1920, 1080
FPS = 30
FONT_SANS_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_SERIF_BOLD = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"

# ---------------- 缓动 ----------------
def clamp01(x):
    if isinstance(x, np.ndarray):
        return np.clip(x, 0.0, 1.0)
    return 0.0 if x < 0 else (1.0 if x > 1 else x)

def ease_in_out(t):
    t = clamp01(t)
    return t * t * (3 - 2 * t)

def ease_out_cubic(t):
    t = clamp01(t); u = 1 - t
    return 1 - u * u * u

def ease_in_cubic(t):
    t = clamp01(t)
    return t * t * t if not isinstance(t, np.ndarray) else t ** 3

def ease_out_back(t, k=1.7):
    if isinstance(t, np.ndarray):
        t = np.clip(t, 0.0, 1.0)
    else:
        t = clamp01(t)
    u = t - 1
    return 1 + (k + 1) * u * u * u + k * u * u
def smoothstep(a, b, x):
    return ease_in_out((x - a) / (b - a)) if b > a else (1.0 if x >= b else 0.0)

def lerp(a, b, t): return a + (b - a) * t

def seg(t, a, b):
    """t 秒线性映射到 [0,1]，夹紧"""
    return clamp01((t - a) / (b - a)) if b > a else (1.0 if t >= b else 0.0)

# ---------------- 颜色 ----------------
def rgb(r, g, b): return (r, g, b)
def scale_color(c, k): return (int(c[0] * k), int(c[1] * k), int(c[2] * k))

CYAN    = (0, 229, 255)
MAGENTA = (255, 61, 220)
GOLD    = (255, 196, 60)
VIOLET  = (150, 96, 255)
GREEN   = (60, 255, 160)
WHITE   = (235, 240, 255)
ORANGE  = (255, 130, 60)
BLUE    = (70, 120, 255)
PINK    = (255, 105, 180)

# ---------------- 3D 数学 ----------------
def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def axis_angle_matrix(axis, angle):
    """Rodrigues 旋转矩阵, axis 无需归一"""
    axis = np.asarray(axis, float)
    axis = normalize(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

class Camera:
    """轨道相机: 绕 target 旋转 (yaw/pitch/dist) + 透视投影"""
    def __init__(self, yaw=0.6, pitch=0.35, dist=4.2, target=(0, 0, 0), fov_deg=42.0):
        self.yaw, self.pitch, self.dist = yaw, pitch, dist
        self.target = np.asarray(target, float)
        self.f = (H / 2) / math.tan(math.radians(fov_deg) / 2)

    def view(self, pts):
        pts = np.asarray(pts, float)
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        R = Rx @ Rz
        cam = R @ (pts - self.target).T
        cam = cam.T
        cam[:, 2] += self.dist          # 相机在 +z
        return cam

    def project(self, pts, center=(W / 2, H / 2)):
        cam = self.view(pts)
        z = cam[:, 2]
        safe = np.maximum(z, 0.05)
        sx = center[0] + cam[:, 0] / safe * self.f
        sy = center[1] - cam[:, 1] / safe * self.f
        return np.stack([sx, sy], 1), z

    def project_single(self, pt, center=(W / 2, H / 2)):
        p2, z = self.project(np.asarray(pt, float)[None, :], center)
        return p2[0], z[0]

def draw_line_3d(canvas, p1, p2, color, width=2, alpha=1.0, z=None):
    """3D 线段(已投影坐标)"""
    canvas.line(tuple(p1), tuple(p2), color, width, alpha, z=z)

# ---------------- 画布 ----------------
class Canvas:
    def __init__(self, bg):
        self.bg = bg                                    # np.uint8 (H,W,3)
        self.img = Image.fromarray(bg).convert("RGBA")  # 基底
        self.ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.ov)
        self.add = np.zeros((H, W, 3), np.float32)      # 加法发光层
        gw, gh = W // 4, H // 4
        self.glow = Image.new("RGB", (gw, gh), (0, 0, 0))
        self.gd = ImageDraw.Draw(self.glow)
        self._has_ov = False
        self._has_add = False
        self._has_glow = False

    # ---- 实心层: 多层描边近似AA ----
    def line(self, p1, p2, color, width=2, alpha=1.0, z=None, glow=1.0):
        if alpha <= 0.003: return
        if z is not None and z < 0.1: return
        p1 = (float(p1[0]), float(p1[1])); p2 = (float(p2[0]), float(p2[1]))
        a = int(255 * clamp01(alpha))
        w = max(1, int(round(width)))
        if w >= 3:
            self.d.line([p1, p2], fill=color + (int(a * 0.16),), width=w * 2)
            self.d.line([p1, p2], fill=color + (int(a * 0.45),), width=int(w * 1.6))
        self.d.line([p1, p2], fill=color + (a,), width=w)
        if w >= 2 and width >= 3:
            self.d.line([p1, p2], fill=(255, 255, 255, int(a * 0.5)), width=max(1, w - 1))
        self._has_ov = True
        if glow > 0:
            gc = scale_color(color, min(1.6, glow))
            self.gd.line([(p1[0] / 4, p1[1] / 4), (p2[0] / 4, p2[1] / 4)],
                         fill=gc, width=max(2, int(width * 0.8)))
            self._has_glow = True

    def dot(self, p, r, color, alpha=1.0, glow=1.0, core=True):
        if alpha <= 0.003 or r < 0.3: return
        x, y = float(p[0]), float(p[1])
        rr = float(r)
        bbox = [x - rr, y - rr, x + rr, y + rr]
        a = int(255 * clamp01(alpha))
        if a < 255:
            ov = Image.new("RGBA", (int(2 * rr + 4), int(2 * rr + 4)), (0, 0, 0, 0))
            dd = ImageDraw.Draw(ov)
            dd.ellipse([2, 2, 2 + 2 * rr, 2 + 2 * rr], fill=color + (a,))
            self.ov.alpha_composite(ov, (int(x - rr - 2), int(y - rr - 2)))
        else:
            self.d.ellipse(bbox, fill=color + (255,))
        self._has_ov = True
        if core and rr >= 1.5:
            self.add_dot(p, rr * 0.55, (255, 255, 255), alpha * 0.5)
        if glow > 0:
            gr = max(1.0, rr / 4 * 1.4)
            self.gd.ellipse([x / 4 - gr, y / 4 - gr, x / 4 + gr, y / 4 + gr],
                            fill=scale_color(color, min(1.5, glow)))
            self._has_glow = True

    # ---- 加法发光层 (numpy sprite) ----
    def _sprite(self, r, color):
        size = int(r * 4)
        if size < 3: size = 3
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        c = size / 2 - 0.5
        d2 = (xx - c) ** 2 + (yy - c) ** 2
        s = np.exp(-d2 / (r * r * 0.9))
        return s[..., None] * np.asarray(color, np.float32)

    def add_dot(self, p, r, color, alpha=1.0):
        if alpha <= 0.01 or r < 0.5: return
        spr = self._sprite(r, color) * alpha
        h, w = spr.shape[:2]
        x0, y0 = int(round(p[0] - w / 2)), int(round(p[1] - h / 2))
        x1, y1 = x0 + w, y0 + h
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(W, x1), min(H, y1)
        if cx1 <= cx0 or cy1 <= cy0: return
        self.add[cy0:cy1, cx0:cx1] += spr[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        self._has_add = True

    def add_line(self, p1, p2, color, width=2, alpha=1.0, steps=None):
        """发光线段: 沿线撒 sprite"""
        if alpha <= 0.01: return
        p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
        L = float(np.linalg.norm(p2 - p1))
        n = steps or max(2, int(L / max(1.5, width * 0.8)))
        for i in range(n + 1):
            t = i / n
            p = p1 * (1 - t) + p2 * t
            fade = alpha * (0.35 + 0.65 * math.sin(math.pi * t) if steps is None else alpha)
            self.add_dot(p, width, color, fade)

    # ---- 文字 ----
    def text(self, xy, s, size, color=WHITE, anchor="mm", alpha=1.0,
             glow=0.9, font_path=FONT_SANS_BOLD, tracking=0.0):
        if alpha <= 0.01: return
        f = get_font(size, font_path)
        x, y = float(xy[0]), float(xy[1])
        a = int(255 * clamp01(alpha))
        if tracking > 0:
            self._text_tracked(x, y, s, f, color, a, tracking, anchor)
        else:
            self.d.text((x, y), s, font=f, fill=color + (a,), anchor=anchor)
        self._has_ov = True
        if glow > 0:
            col = tuple(int(c * min(1.0, glow)) for c in color)
            fg = get_font(max(6, int(size * 0.26)), font_path)
            if tracking > 0:
                self._glow_text_tracked(x / 4, y / 4, s, fg, col, tracking / 4, anchor)
            else:
                self.gd.text((x / 4, y / 4), s, font=fg, fill=col, anchor=anchor)
            self._has_glow = True

    def _glow_text_tracked(self, x, y, s, f, col, tr, anchor):
        widths = [self.gd.textlength(ch, font=f) for ch in s]
        total = sum(widths) + tr * (len(s) - 1)
        if anchor[0] == "m": cx = x - total / 2
        elif anchor[0] == "r": cx = x - total
        else: cx = x
        for ch, wd in zip(s, widths):
            self.gd.text((cx, y), ch, font=f, fill=col, anchor="l" + anchor[1])
            cx += wd + tr

    def _text_tracked(self, x, y, s, f, color, a, tr, anchor):
        widths = [self.d.textlength(ch, font=f) for ch in s]
        total = sum(widths) + tr * (len(s) - 1)
        if anchor[0] == "m": cx = x - total / 2
        elif anchor[0] == "r": cx = x - total
        else: cx = x
        for ch, wd in zip(s, widths):
            self.d.text((cx, y), ch, font=f, fill=color + (a,), anchor="l" + anchor[1])
            cx += wd + tr

    # ---- 背景星点闪烁 ----
    def twinkle_stars(self, stars, t):
        """stars: list of (x,y,r,base,phase)"""
        for (x, y, r, base, ph) in stars:
            b = base * (0.55 + 0.45 * math.sin(t * 1.7 + ph))
            self.add_dot((x, y), r, WHITE, b)
        # 不强制

    # ---- 完成: 合成 ----
    def finish(self, glow_strength=1.0):
        img = self.img
        if self._has_ov:
            img = Image.alpha_composite(img, self.ov)
        img = img.convert("RGB")
        arr = np.asarray(img, np.float32)
        if self._has_add:
            arr = arr + self.add
        if self._has_glow and glow_strength > 0:
            g = self.glow.filter(ImageFilter.GaussianBlur(7))
            g = g.resize((W, H), Image.BILINEAR)
            arr += np.asarray(g, np.float32) * glow_strength
        np.clip(arr, 0, 255, out=arr)
        return arr.astype(np.uint8)

# ---------------- 字体缓存 ----------------
_FONTS = {}
def get_font(size, path=FONT_SANS_BOLD):
    key = (size, path)
    if key not in _FONTS:
        _FONTS[key] = ImageFont.truetype(path, size)
    return _FONTS[key]

# ---------------- 背景 ----------------
class Background:
    """静态深空渐变背景 + 星点(星点每帧闪烁由 canvas.twinkle_stars 完成)"""
    def __init__(self, top=(4, 6, 16), bottom=(10, 14, 34), halo=None, seed=7, n_stars=170):
        self.halo = halo          # list of (cx,cy,radius,color,strength)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
        base = np.zeros((H, W, 3), np.float32)
        for i in range(3):
            base[..., i] = top[i] * (1 - yy) + bottom[i] * yy
        if halo:
            for (cx, cy, rad, col, k) in halo:
                d2 = ((xx * W - cx) ** 2 + (yy * H - cy) ** 2) / (rad * rad)
                g = np.exp(-d2)[..., None] * k
                base += g * np.asarray(col, np.float32)
        np.clip(base, 0, 255, out=base)
        self.img = base.astype(np.uint8)
        rng = np.random.default_rng(seed)
        self.stars = []
        for _ in range(n_stars):
            x = rng.uniform(0, W); y = rng.uniform(0, H)
            r = rng.uniform(0.7, 2.0)
            b = rng.uniform(0.25, 0.85)
            ph = rng.uniform(0, 6.28)
            self.stars.append((x, y, r, b, ph))

def vignette(arr, k=0.35):
    """暗角"""
    h, w = arr.shape[:2]
    yy = (np.linspace(-1, 1, h, dtype=np.float32))[:, None]
    xx = (np.linspace(-1, 1, w, dtype=np.float32))[None, :]
    d = np.sqrt(xx * xx + yy * yy)
    m = 1 - k * np.clip(d - 0.55, 0, 1) ** 2
    return (arr * m[..., None]).astype(np.uint8)

# ---------------- 转场 ----------------
def fade_black(arr, tin=0.0, tout=0.0, fin=0.8, fout=0.8):
    if tin > 0:
        arr = (arr * ease_in_cubic(seg(tin, 0, fin))).astype(np.uint8)
    if tout > 0:
        arr = (arr * (1 - ease_in_cubic(seg(tout, 0, fout)))).astype(np.uint8)
    return arr

def fade_black_f(arr_f, tin, tout, fin, fout):
    """float 版"""
    k = 1.0
    if tin is not None and fin > 0: k *= ease_in_cubic(seg(tin, 0, fin))
    if tout is not None and fout > 0: k *= 1 - ease_in_cubic(seg(tout, 0, fout))
    return arr_f * k

# ---------------- 常用组合 ----------------
def subtitle(canvas, y, main, size=44, alpha=1.0, color=(200, 214, 245), sub=None, sub_color=(120, 140, 190)):
    """主字幕 + 可选英文小字"""
    if alpha <= 0.01: return
    canvas.text((W / 2, y), main, size, color, "mm", alpha, glow=0.55)
    if sub:
        canvas.text((W / 2, y + size * 0.82), sub, int(size * 0.5), sub_color, "mm", alpha * 0.85, glow=0.4)

def big_title(canvas, y, main, size=130, alpha=1.0, color=WHITE, sub=None, sub_size=40):
    canvas.text((W / 2, y), main, size, color, "mm", alpha, glow=1.15)
    if sub:
        canvas.text((W / 2, y + size * 0.68), sub, sub_size, (150, 165, 210), "mm", alpha * 0.9, glow=0.5, tracking=int(sub_size*0.45))


# ================= v2: 双语字幕 / 章节卡 / 手稿纹理 =================

def subtitle2(canvas, y, zh, en, size=42, alpha=1.0, color=(210, 222, 248),
              en_color=(128, 148, 198), gap=0.86):
    """中英双语字幕"""
    if alpha <= 0.01: return
    canvas.text((W / 2, y), zh, size, color, "mm", alpha, glow=0.55)
    canvas.text((W / 2, y + size * gap), en, int(size * 0.42), en_color, "mm",
                alpha * 0.8, glow=0.3, tracking=int(size * 0.12))

ROMAN = {1: "Ⅰ", 2: "Ⅱ", 3: "Ⅲ", 4: "Ⅳ", 5: "Ⅴ", 6: "Ⅵ", 7: "Ⅶ", 8: "Ⅷ"}

def chapter_card(canvas, num, zh, en, alpha=1.0, y=430, t=0.0):
    """罗马数字章节卡: Ⅴ 伽罗瓦 GALOIS THEORY 样式"""
    if alpha <= 0.01: return
    cxx = W / 2
    # 装饰横线（左右展开）
    k = ease_out_cubic(seg(t, 0, 0.8))
    for s in (-1, 1):
        x1 = cxx + s * (150 + 260 * k)
        x2 = cxx + s * 150
        canvas.line((x1, y), (x2, y), (150, 160, 210), 1, 0.5 * alpha * k, glow=0.25)
    # 罗马数字
    canvas.text((cxx, y - 120), ROMAN.get(num, str(num)), 64, GOLD, "mm", alpha, glow=1.1)
    # 中文大字
    canvas.text((cxx, y + 10), zh, 96, WHITE, "mm", alpha, glow=1.2, tracking=14)
    # 英文
    canvas.text((cxx, y + 130), en, 34, (150, 165, 215), "mm", alpha * 0.92,
                glow=0.4, tracking=int(34 * 0.5))

def manuscript_bg(seed=17, dust_k=1.0):
    """旧纸手稿背景: 米黄纹理 + 斑点 + 暗角"""
    import numpy as np
    from PIL import Image as _Img, ImageFilter as _F
    rng = np.random.default_rng(seed)
    base = np.zeros((H, W, 3), np.float32)
    base[..., 0] = np.linspace(30, 22, H)[:, None]
    base[..., 1] = np.linspace(24, 18, H)[:, None]
    base[..., 2] = np.linspace(20, 15, H)[:, None]
    # 纸张微光(中央亮)
    yy = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    base += np.exp(-(xx * xx * 0.8 + yy * yy * 1.4))[..., None] * np.array([26, 20, 12], np.float32)
    # 纤维噪声
    noise = rng.normal(0, 1, (H // 4, W // 4, 1)).astype(np.float32)
    n = np.asarray(_Img.fromarray(((noise[:, :, 0] * 0.5 + 0.5) * 255).astype(np.uint8)).resize((W, H), _Img.BILINEAR), np.float32)
    base += ((n - 128) / 128 * 4.5)[..., None] * np.array([1.0, 0.85, 0.6], np.float32)
    # 霉斑/年份污渍
    for _ in range(24):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        r = rng.uniform(30, 130)
        d2 = (xx * W - cx) ** 2 + (yy * H - cy) ** 2
        base += np.exp(-d2 / (r * r))[..., None] * rng.uniform(1.5, 4.5, 3).astype(np.float32) * np.array([1.2, 0.7, 0.3])
    np.clip(base, 0, 255, out=base)
    return base.astype(np.uint8)
