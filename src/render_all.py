# -*- coding: utf-8 -*-
"""并行渲染调度: 每个分片 pipe 给独立 ffmpeg 编码为 mp4 片段, 最后 concat"""
import os
import sys
import subprocess
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

W, H, FPS = 1920, 1080, 30
FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunks")

def get_scenes():
    from scenes_a import Scene0, Scene1, Scene2, Scene3
    from scenes_b import Scene4, Scene5, Scene6, Scene7
    return {
        "s0": Scene0, "s1": Scene1, "s2": Scene2, "s3": Scene3,
        "s4": Scene4, "s5": Scene5, "s6": Scene6, "s7": Scene7,
    }

def render_chunk(args):
    scene_name, f0, f1, out = args
    import numpy as np
    scenes = get_scenes()
    scene = scenes[scene_name]()
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-",
           "-c:v", "libopenh264", "-b:v", "16M", "-maxrate", "24M",
           "-pix_fmt", "yuv420p", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(f0, f1):
        frame = scene.frame(i)
        p.stdin.write(np.ascontiguousarray(frame).tobytes())
    p.stdin.close()
    ret = p.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed for {out}")
    return out

def build_tasks():
    tasks = []
    os.makedirs(FRAMES_DIR, exist_ok=True)
    scenes = get_scenes()
    for name, Cls in scenes.items():
        n = Cls.N
        chunk = 135
        k = 0
        for f0 in range(0, n, chunk):
            f1 = min(f0 + chunk, n)
            out = os.path.join(FRAMES_DIR, f"{name}_{k:02d}.mp4")
            tasks.append((name, f0, f1, out))
            k += 1
    return tasks

def concat(out_path):
    scenes = get_scenes()
    files = []
    for name in scenes.keys():
        ks = sorted(f for f in os.listdir(FRAMES_DIR) if f.startswith(name + "_") and f.endswith(".mp4"))
        files += [os.path.join(FRAMES_DIR, f) for f in ks]
    lst = os.path.join(FRAMES_DIR, "list.txt")
    with open(lst, "w") as fp:
        for f in files:
            fp.write(f"file '{f}'\n")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    "-i", lst, "-c", "copy", out_path], check=True)
    print("concat ->", out_path)

if __name__ == "__main__":
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    tasks = build_tasks()
    print(f"total chunks: {len(tasks)}, workers: {nproc}")
    with mp.Pool(nproc) as pool:
        for i, _ in enumerate(pool.imap_unordered(render_chunk, tasks)):
            print(f"[{i+1}/{len(tasks)}] done", flush=True)
    final = os.path.join(FRAMES_DIR, "video_noaudio.mp4")
    concat(final)
    print("ALL DONE")
