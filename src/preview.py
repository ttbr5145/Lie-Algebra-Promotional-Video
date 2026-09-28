# -*- coding: utf-8 -*-
"""预览: 渲染指定场景指定帧 -> preview/*.png"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image

def main():
    scenes = {}
    from scenes_a import Scene0, Scene1, Scene2, Scene3
    from scenes_b import Scene4, Scene5, Scene6, Scene7
    all_s = {"s0": Scene0, "s1": Scene1, "s2": Scene2, "s3": Scene3,
             "s4": Scene4, "s5": Scene5, "s6": Scene6, "s7": Scene7}
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preview")
    os.makedirs(outdir, exist_ok=True)
    # 参数: scene:frame,scene:frame ...
    spec = sys.argv[1] if len(sys.argv) > 1 else "s0:200,s1:400,s2:400,s3:500,s4:270,s5:700,s6:450,s7:250"
    for item in spec.split(","):
        name, frames = item.split(":")
        if name not in scenes:
            scenes[name] = all_s[name]()
            t0 = time.time()
            scenes[name].frame(0)
            print(f"{name} init+first frame: {time.time()-t0:.2f}s")
        sc = scenes[name]
        for f in [int(x) for x in frames.split("+")]:
            t0 = time.time()
            arr = sc.frame(f)
            dt = time.time() - t0
            Image.fromarray(arr).save(os.path.join(outdir, f"{name}_f{f:04d}.png"))
            print(f"{name} frame {f}: {dt*1000:.0f} ms")

if __name__ == "__main__":
    main()
