# -*- coding: utf-8 -*-
"""配乐合成: 180s 电影感电子乐, 结构与视频场景对齐
段落切换点: 13 / 40 / 67 / 89 / 107 / 137 / 167 (秒)
A小调, BPM 96, 和弦循环 Am -> F -> C -> G (每和弦 5s, 循环 20s)
"""
import numpy as np
import wave

SR = 44100
DUR = 180.0
N = int(SR * DUR)
BPM = 96.0
BEAT = 60.0 / BPM          # 0.625s
BAR = 4 * BEAT             # 2.5s

# ---- 和弦 (A小调): Am -> F -> C -> G, 每和弦 5s ----
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
CHORD_LEN = 5.0

L = np.zeros(N, np.float64)
R = np.zeros(N, np.float64)

def add(sig, t0, pan=0.5, gain=1.0):
    """混入信号"""
    if sig is None: return
    i0 = int(t0 * SR)
    if i0 >= N: return
    n = min(len(sig), N - i0)
    gl, gr = np.cos(pan * np.pi / 2), np.sin(pan * np.pi / 2)
    L[i0:i0 + n] += sig[:n] * gain * gl * 1.414
    R[i0:i0 + n] += sig[:n] * gain * gr * 1.414

def env_ad(n, a, d, curve=5.0):
    """attack-decay 包络"""
    t = np.arange(n) / SR
    e = np.minimum(t / max(a, 1e-4), 1.0)
    e *= np.exp(-np.maximum(t - a, 0) * curve / max(d, 1e-3) * 5)
    return e

def tone(freq, dur, kind="sine", detune=0.0, nharm=0):
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
    if kind == "tri":
        s = s / (np.abs(s) + 1e-6) * 0.6 + s * 0.4
    return s

def kick(t0, gain=0.55):
    n = int(0.30 * SR)
    t = np.arange(n) / SR
    f = 42 + 130 * np.exp(-t * 26)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 13)
    add(s, t0, 0.5, gain)

def hat(t0, gain=0.12, dur=0.045):
    n = int(dur * SR)
    rng = np.random.default_rng(int(t0 * 1000) % 9999)
    s = rng.standard_normal(n)
    s = np.diff(s, prepend=0)          # 高通
    s *= np.exp(-np.arange(n) / SR * 90)
    add(s, t0, 0.5, gain)

def snare(t0, gain=0.20):
    n = int(0.16 * SR)
    rng = np.random.default_rng(int(t0 * 777) % 9999)
    s = rng.standard_normal(n)
    s = np.diff(s, prepend=0)
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
    a = np.minimum(t / 1.3, 1.0) ** 1.5
    rel = np.clip((n / SR - t) / 1.5, 0, 1)
    s *= a * rel
    add(s, t0, 0.5, gain)

def arp_note(t0, freq, gain=0.15):
    n = int(0.20 * SR)
    t = np.arange(n) / SR
    raw = np.sin(2 * np.pi * freq * t)
    s = (np.abs(raw) * 2 - 1) * 0.55 + raw * 0.45     # 三角混合
    s *= np.exp(-t * 22) * np.minimum(t / 0.004, 1)
    add(s, t0, 0.5, gain)

def ping(t0, freq, gain=0.12):
    n = int(2.8 * SR)
    t = np.arange(n) / SR
    s = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(2 * np.pi * freq * 2.01 * t)
    s *= np.exp(-t * 2.2)
    add(s, t0, 0.5, gain)

# ================= 编曲 =================
print("compose...")
T_SWITCH = [13, 40, 67, 89, 107, 137, 167]

def chord_at(t):
    if t >= 175:
        return CHORDS[0]
    return CHORDS[int(t / CHORD_LEN) % 4]

def is_chord_change(t):
    if t >= 175: return False
    k = t / CHORD_LEN
    return abs(k - round(k)) < 1e-6 and int(round(k)) % 4 == 0

# --- pad: 每 5s 一个和弦 ---
tt = 0.0
while tt < 175:
    ch = chord_at(tt + 0.01)
    bright = 1.0 if (89 <= tt < 167) else 0.35
    g = 0.16
    if tt >= 167: g = 0.20
    pad_chord(tt, ch["pad"], min(CHORD_LEN, 175 - tt), gain=g, bright=bright)
    tt += CHORD_LEN
# 终止长音 Am
pad_chord(175, [110.0, 164.81, 220.0, 261.63, 329.63, 440.0], 5.0, gain=0.22, bright=0.5)

# --- ping 点缀 (开场/尾声) ---
for (t0, f) in [(1.6, 880), (5.6, 659.25), (9.6, 880), (169.0, 880), (173.5, 659.25)]:
    ping(t0, f, 0.10)

# --- 节拍层 ---
n_beats = int(DUR / BEAT)
rng = np.random.default_rng(7)
for b in range(n_beats):
    t0 = b * BEAT
    if t0 >= 175: break
    # bass: 13s起, 每拍
    if 13 <= t0 < 167:
        bass_note(t0, chord_at(t0)["bass"], 0.30 if t0 >= 107 else 0.26)
    # kick
    beat_in_bar = b % 4
    if 107 <= t0 < 137:
        kick(t0, 0.55)                                  # four-on-floor
    elif 13 <= t0 < 167:
        if beat_in_bar in (0, 2):
            kick(t0, 0.42)
    # snare: 高潮段 2/4 拍
    if 107 <= t0 < 167 and beat_in_bar in (1, 3):
        snare(t0, 0.17)
    # hihat
    if 89 <= t0 < 137:
        hat(t0 + BEAT / 2, 0.10)                        # 后半拍
        if t0 >= 107:
            hat(t0, 0.07)
    elif 40 <= t0 < 89:
        hat(t0 + BEAT / 2, 0.085)
    elif 13 <= t0 < 40 and beat_in_bar == 2:
        hat(t0 + BEAT / 2, 0.06)

# --- arp: 16 分音符 ---
step = BEAT / 4
t = 67.0
i = 0
seq = [0, 2, 4, 1, 3, 5, 4, 2]
while t < 167:
    g = 0.16 if t >= 107 else 0.09
    if t >= 137: g = 0.13
    ch = chord_at(t)
    f = ch["arp"][seq[i % len(seq)]]
    pan = 0.3 + 0.4 * (i % 2)
    arp_note(t, f, g)
    i += 1
    t += step

# ================= 混响 (发送自 pad/arp/ping 已在主混中, 这里对整体做轻延迟叠加) =================
print("reverb...")
def simple_reverb(x):
    d0, d1, d2 = int(0.093 * SR), int(0.137 * SR), int(0.211 * SR)
    y = x.copy()
    y[d0:] += 0.30 * x[:-d0]
    y[d1:] += 0.20 * x[:-d1]
    y[d2:] += 0.13 * x[:-d2]
    y[d0+d1:] += 0.10 * x[:-(d0+d1)]
    return y

# 只对高频成分做混响太复杂 —— 对整轨做 12% 湿声
wet_L = simple_reverb(L) - L
wet_R = simple_reverb(R) - R
L += wet_L * 0.35
R += wet_R * 0.35

# ================= 母带 =================
print("master...")
# 尾部淡出
fade = np.ones(N)
nf = int(7 * SR)
fade[-nf:] = np.linspace(1, 0, nf) ** 1.5
L *= fade; R *= fade
# soft clip + 归一化
L = np.tanh(L * 1.1)
R = np.tanh(R * 1.1)
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
