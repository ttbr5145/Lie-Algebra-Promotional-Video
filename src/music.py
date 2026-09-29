# -*- coding: utf-8 -*-
"""v2 配乐合成: 187.5s 电影感电子乐 + SFX 音效轨
BPM 96 -> 拍 0.625s, 小节 2.5s; 和弦 Am->F->C->G 每小节一换(2.5s)
场景边界(全片秒): 0 / 12.5 / 40 / 67.5 / 90 / 115 / 140 / 162.5 / 187.5
关键 cue: 6.25-8.125 魔方click / 18.75 枪响 / 31.25 李群 / 35 章节卡I
          40/67.5/90/115/140 章节卡 / 96.875 不等号爆闪 / 123.5-128.5 E8波前
          162.5-172.5 16连hit / 172.5 riser / 180 boom+落版
"""
import math
import numpy as np
import wave

SR = 44100
DUR = 187.5
N = int(SR * DUR)
BPM = 96.0
BEAT = 60.0 / BPM          # 0.625s
BAR = 4 * BEAT             # 2.5s

# ---- 和弦 (A小调): Am -> F -> C -> G, 每小节 2.5s ----
CHORDS = [
    dict(pad=[110.0, 164.81, 220.0, 261.63], bass=55.0,
         arp=[220.0, 261.63, 329.63, 440.0, 523.25, 659.25]),
    dict(pad=[87.31, 130.81, 174.61, 220.0], bass=43.65,
         arp=[174.61, 220.0, 261.63, 349.23, 440.0, 523.25]),
    dict(pad=[130.81, 196.0, 261.63, 329.63], bass=65.41,
         arp=[196.0, 261.63, 329.63, 392.0, 523.25, 659.25]),
    dict(pad=[98.0, 146.83, 196.0, 246.94], bass=49.0,
         arp=[196.0, 246.94, 293.66, 392.0, 493.88, 587.33]),
]

L = np.zeros(N, np.float64)
R = np.zeros(N, np.float64)

def add(sig, t0, pan=0.5, gain=1.0):
    if sig is None: return
    i0 = int(t0 * SR)
    if i0 >= N: return
    n = min(len(sig), N - i0)
    gl, gr = np.cos(pan * np.pi / 2), np.sin(pan * np.pi / 2)
    L[i0:i0 + n] += sig[:n] * gain * gl * 1.414
    R[i0:i0 + n] += sig[:n] * gain * gr * 1.414

def tone(freq, dur, detune=0.0, nharm=0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    ph = 2 * np.pi * freq * t
    if detune > 0:
        s = np.sin(ph * (1 + detune)) + np.sin(ph * (1 - detune))
        s *= 0.5
    else:
        s = np.sin(ph)
    for h in range(2, 2 + nharm):
        s += np.sin(ph * h) / (h ** 1.5)
    return s

# ================= 基础乐器 =================
def kick(t0, gain=0.55):
    n = int(0.30 * SR)
    t = np.arange(n) / SR
    f = 42 + 130 * np.exp(-t * 26)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 13)
    add(s, t0, 0.5, gain)

def hat(t0, gain=0.12, dur=0.045):
    n = int(dur * SR)
    rng = np.random.default_rng(int(t0 * 1000) % 9999)
    s = np.diff(rng.standard_normal(n + 1))
    s *= np.exp(-np.arange(n) / SR * 90)
    add(s, t0, 0.5, gain)

def snare(t0, gain=0.20):
    n = int(0.16 * SR)
    rng = np.random.default_rng(int(t0 * 777) % 9999)
    s = np.diff(rng.standard_normal(n + 1))
    e = np.exp(-np.arange(n) / SR * 28)
    s = s * e + np.sin(2 * np.pi * 190 * np.arange(n) / SR) * e * 0.4
    add(s, t0, 0.5, gain)

def bass_note(t0, freq, gain=0.34):
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * freq * 2 * t)
    s *= np.minimum(t / 0.006, 1) * np.exp(-t * 6.5)
    add(s, t0, 0.5, gain)

def pad_chord(t0, freqs, dur, gain=1.0, bright=1.0):
    n = int((dur + 1.6) * SR)
    t = np.arange(n) / SR
    s = np.zeros(n)
    for f in freqs:
        v = np.sin(2 * np.pi * f * (1 + 0.0016) * t) + np.sin(2 * np.pi * f * (1 - 0.0016) * t)
        v *= 0.5
        for h in range(2, 9):
            amp = (1.0 / h ** 1.25) * (0.35 + 0.65 * bright)
            v += np.sin(2 * np.pi * f * h * t) * amp / len(freqs)
        s += v / len(freqs)
    a = np.minimum(t / 1.1, 1.0) ** 1.5
    rel = np.clip((n / SR - t) / 1.4, 0, 1)
    s *= a * rel
    add(s, t0, 0.5, gain)

def arp_note(t0, freq, gain=0.15):
    n = int(0.20 * SR)
    t = np.arange(n) / SR
    raw = np.sin(2 * np.pi * freq * t)
    s = (np.abs(raw) * 2 - 1) * 0.55 + raw * 0.45
    s *= np.exp(-t * 22) * np.minimum(t / 0.004, 1)
    add(s, t0, 0.5, gain)

def ping(t0, freq, gain=0.12):
    n = int(2.8 * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2.01 * t)
    s *= np.exp(-t * 2.2)
    add(s, t0, 0.5, gain)

# ================= SFX 音效 =================
def heartbeat(t0, gain=0.5):
    for (dt, g) in ((0.0, 1.0), (0.15, 0.65)):
        n = int(0.24 * SR)
        t = np.arange(n) / SR
        f = 55 + 42 * np.exp(-t * 30)
        s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 17) * g
        add(s, t0 + dt, 0.5, gain)

def gunshot(t0, gain=0.95):
    n = int(1.4 * SR)
    rng = np.random.default_rng(666)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    s = noise * np.exp(-t * 40) * 1.1
    f = 52 + 95 * np.exp(-t * 24)
    s += np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * 0.9
    s += noise * np.exp(-t * 3.0) * 0.10          # 长回响尾
    add(s, t0, 0.5, gain)

def whoosh(t0, dur=1.25, gain=0.30, up=True):
    n = int(dur * SR)
    t = np.arange(n) / SR
    rng = np.random.default_rng(int(t0 * 997) % 9999)
    nz = np.diff(rng.standard_normal(n + 1))
    f0, f1 = (180, 2600) if up else (2600, 180)
    fc = f0 * (f1 / f0) ** (t / max(dur, 1e-3))
    ph = 2 * np.pi * np.cumsum(fc) / SR
    s = nz * 0.55 + np.sin(ph) * 0.35
    e = np.sin(np.pi * t / max(dur, 1e-3)) ** 1.6
    add(s * e, t0, 0.5, gain)

def sparkle(t0, freqs, gain=0.11, gap=0.09):
    for j, f in enumerate(freqs):
        ping(t0 + j * gap, f, gain * (0.85 + 0.15 * math.sin(j * 1.7)))

def boom(t0, gain=1.0):
    n = int(2.8 * SR)
    t = np.arange(n) / SR
    f = 62 * np.exp(-t * 1.9) + 27
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.1)
    rng = np.random.default_rng(42)
    s += rng.standard_normal(n) * np.exp(-t * 15) * 0.55
    add(s, t0, 0.5, gain)

def click(t0, gain=0.28):
    n = int(0.06 * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * 1750 * t) * np.exp(-t * 130) + np.sin(2 * np.pi * 690 * t) * np.exp(-t * 60) * 0.6
    add(s, t0, 0.5, gain)

def hit(t0, freq=520, gain=0.40, decay=10.0):
    n = int(0.55 * SR)
    t = np.arange(n) / SR
    s = np.sign(np.sin(2 * np.pi * freq * t)) * 0.30
    for h in (2.01, 3.02, 4.71):
        s += np.sign(np.sin(2 * np.pi * freq * h * t)) * 0.16
    s *= np.exp(-t * decay)
    rng = np.random.default_rng(int(freq) % 9999)
    s += rng.standard_normal(n) * np.exp(-t * 70) * 0.5
    add(s, t0, 0.5, gain)

# ================= 编曲 =================
print("compose...")
NBAR = int(DUR / BAR)                     # 75 小节

def chord_at(t):
    return CHORDS[int(t / BAR) % 4]

# --- pad: 每小节一个和弦 ---
PAD_GAIN = lambda bar: (
    0.09 if bar < 5 else
    0.05 if bar == 7 else                 # 枪响小节压低
    0.12 if bar < 11 else
    0.15 if bar < 16 else
    0.15 if bar < 27 else
    0.16 if bar < 36 else
    0.18 if bar < 46 else
    0.15 if bar < 56 else
    0.17 if bar < 65 else
    0.14 if bar < 70 else
    0.15 if bar < 72 else
    0.20)
PAD_BRIGHT = lambda bar: (
    0.15 if bar < 5 else
    0.22 if bar < 11 else
    0.45 if bar < 16 else
    0.55 if bar < 27 else
    0.70 if bar < 36 else
    0.85 if bar < 46 else
    1.0 if bar < 56 else
    0.55 if bar < 65 else
    0.85 if bar < 70 else
    0.5 if bar < 72 else
    0.6)
for bar in range(NBAR):
    t0 = bar * BAR
    ch = CHORDS[bar % 4]
    freqs = ch["pad"] + ([329.63, 440.0] if bar >= 72 else [])
    pad_chord(t0, freqs, BAR, gain=PAD_GAIN(bar), bright=PAD_BRIGHT(bar))

# --- 节拍层 ---
n_beats = int(DUR / BEAT)
for b in range(n_beats):
    t0 = b * BEAT
    bib = b % 4
    # bass
    if 40 <= t0 < 162.5:
        bass_note(t0, chord_at(t0)["bass"], 0.30 if t0 >= 115 else 0.26)
    elif 162.5 <= t0 < 180:
        bass_note(t0, chord_at(t0)["bass"], 0.32)
    # kick
    if 40 <= t0 < 67.5 and bib in (0, 2):
        kick(t0, 0.40)
    elif 67.5 <= t0 < 90 and bib in (0, 2):
        kick(t0, 0.42)
    elif 90 <= t0 < 115 and bib in (0, 2):
        kick(t0, 0.45)
    elif 115 <= t0 < 140:
        kick(t0, 0.52)
    elif 140 <= t0 < 162.5 and bib == 0:
        kick(t0, 0.40)
    elif 162.5 <= t0 < 172.5:
        kick(t0, 0.50)
    elif 175 <= t0 < 180 and bib in (0, 2):
        kick(t0, 0.42)
    # snare
    if 90 <= t0 < 115 and bib in (1, 3):
        snare(t0, 0.14)
    elif 115 <= t0 < 140 and bib in (1, 3):
        snare(t0, 0.18)
    elif 162.5 <= t0 < 172.5 and bib in (1, 3):
        snare(t0, 0.20)
    # hihat
    if 40 <= t0 < 67.5:
        hat(t0 + BEAT / 2, 0.08)
    elif 67.5 <= t0 < 90:
        hat(t0 + BEAT / 2, 0.10)
    elif 90 <= t0 < 115:
        hat(t0 + BEAT / 2, 0.10); hat(t0, 0.06)
    elif 115 <= t0 < 140:
        for q in range(4):
            hat(t0 + q * BEAT / 4, 0.075 if q % 2 == 0 else 0.045)
    elif 140 <= t0 < 162.5:
        hat(t0 + BEAT / 2, 0.07)
    elif 162.5 <= t0 < 172.5:
        for q in range(4):
            hat(t0 + q * BEAT / 4, 0.09 if q % 2 == 0 else 0.05)

# --- arp: 16 分 ---
step = BEAT / 4
seq = [0, 2, 4, 1, 3, 5, 4, 2]
t = 67.5
i = 0
while t < 162.5:
    g = 0.09 if t < 115 else 0.15
    if 140 <= t < 162.5: g = 0.08
    f = chord_at(t)["arp"][seq[i % len(seq)]]
    arp_note(t, f, g)
    i += 1
    t += step
# 终章切片期 arp 高八度急促
t = 162.5; i = 0
while t < 172.5:
    f = chord_at(t)["arp"][seq[i % len(seq)]] * 2
    arp_note(t, f, 0.11)
    i += 1; t += step

# ================= SFX cue 轨 =================
print("sfx cues...")
# S0 悬念开场
heartbeat(0.0, 0.42); heartbeat(0.625, 0.42)
ping(3.75, 1318.5, 0.10)                                   # 雪花出现
for tc in (6.25, 6.875, 7.5, 8.125):                       # 魔方 4 连拧
    click(tc, 0.30)
whoosh(8.75, 1.0, 0.26)                                    # 爆开
heartbeat(10.0, 0.5)
heartbeat(10.625, 0.5); heartbeat(11.5625, 0.55); heartbeat(12.1875, 0.6)
# S1 历史幕
heartbeat(13.75, 0.45); heartbeat(14.375, 0.45)
heartbeat(15.625, 0.5); heartbeat(16.25, 0.5); heartbeat(16.875, 0.55)
gunshot(18.75)
whoosh(25.0, 1.5, 0.22)                                    # 极光
ping(26.4, 880, 0.09)
hit(31.25, 660, 0.42)                                      # 李群落版
sparkle(31.35, [523.25, 659.25, 784.0, 1046.5, 1318.5], 0.10)
hit(35.0, 520, 0.36)                                       # 章节卡 I
# 章节卡 hits
hit(40.0, 520, 0.40); whoosh(40.05, 0.7, 0.16)             # 卡 II 旋转
hit(67.5, 520, 0.40); whoosh(67.55, 0.7, 0.16)             # 卡 III 无穷小
hit(90.0, 520, 0.40); whoosh(90.05, 0.7, 0.16)             # 卡 IV 李括号
hit(96.875, 780, 0.44)                                     # ≠ 爆闪
hit(99.375, 430, 0.22)                                     # [X,Y] 点亮
sparkle(109.375, [659.25, 784.0, 987.77, 1318.5], 0.09)    # 李括号心跳
hit(115.0, 520, 0.40); whoosh(115.05, 0.7, 0.16)           # 卡 V 对称之花
hit(123.5, 880, 0.34)                                      # E8 白闪
sparkle(123.6, [659.25, 784.0, 880.0, 1046.5, 1318.5, 1568.0], 0.10)
hit(140.0, 520, 0.40); whoosh(140.05, 0.7, 0.16)           # 卡 VI 万物皆对称
hit(146.25, 600, 0.22)                                     # 幕界 a->b
hit(152.5, 600, 0.22)                                      # 幕界 b->c
ping(158.6, 659.25, 0.10); ping(159.4, 880.0, 0.10); ping(160.2, 1046.5, 0.10)  # 落幅三音
# S7 终章: 16 连 hit
for j in range(16):
    hit(162.5 + j * BEAT, 480 + j * 22, 0.30 + j * 0.013, decay=12)
whoosh(172.5, 3.0, 0.30)                                   # riser
heartbeat(177.5, 0.55); heartbeat(178.6, 0.62); heartbeat(179.3, 0.70)
boom(180.0, 1.0)                                           # 爆炸+落版
ping(181.0, 659.25, 0.12); ping(182.5, 880.0, 0.12); ping(184.0, 1046.5, 0.12)

# ================= 混响 =================
print("reverb...")
def simple_reverb(x):
    d0, d1, d2 = int(0.093 * SR), int(0.137 * SR), int(0.211 * SR)
    y = x.copy()
    y[d0:] += 0.30 * x[:-d0]
    y[d1:] += 0.20 * x[:-d1]
    y[d2:] += 0.13 * x[:-d2]
    y[d0+d1:] += 0.10 * x[:-(d0+d1)]
    return y

wet_L = simple_reverb(L) - L
wet_R = simple_reverb(R) - R
L += wet_L * 0.33
R += wet_R * 0.33

# ================= 母带 =================
print("master...")
fade = np.ones(N)
nf = int(6.0 * SR)
fade[-nf:] = np.linspace(1, 0, nf) ** 1.5
L *= fade; R *= fade
L = np.tanh(L * 1.12)
R = np.tanh(R * 1.12)
peak = max(np.abs(L).max(), np.abs(R).max())
k = 0.92 / peak
L *= k; R *= k

out = np.empty(N * 2, np.float32)
out[0::2] = L.astype(np.float32)
out[1::2] = R.astype(np.float32)
pcm = (np.clip(out, -1, 1) * 32767).astype(np.int16)

with wave.open("music.wav", "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("music.wav written:", DUR, "s")
