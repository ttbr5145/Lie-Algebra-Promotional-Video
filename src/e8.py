# -*- coding: utf-8 -*-
"""E8 根系统: Cartan矩阵 -> 反射闭包生成240根 -> Coxeter平面投影(30重对称)"""
import numpy as np
import math

_E8_CARTAN = np.array([
    [ 2, 0,-1, 0, 0, 0, 0, 0],
    [ 0, 2,-1, 0, 0, 0, 0, 0],
    [-1,-1, 2,-1, 0, 0, 0, 0],
    [ 0, 0,-1, 2,-1, 0, 0, 0],
    [ 0, 0, 0,-1, 2,-1, 0, 0],
    [ 0, 0, 0, 0,-1, 2,-1, 0],
    [ 0, 0, 0, 0, 0,-1, 2,-1],
    [ 0, 0, 0, 0, 0, 0,-1, 2],
], dtype=np.float64)

def _select_simple_roots(roots):
    """泛函法选简单根: 正根 = w·r>0; 简单根 = 不能写成两个正根之和的正根"""
    rng = np.random.default_rng(42)
    while True:
        w = rng.uniform(0.1, 1.0, 8)
        vals = roots @ w
        pos = roots[vals > 1e-9]
        plist = [tuple(np.round(r, 8)) for r in pos]
        pset = set(plist)
        simple = []
        for r in pos:
            decomp = False
            for q in pos:
                s = tuple(np.round(r - q, 8))
                if s in pset:
                    decomp = True
                    break
            if not decomp:
                simple.append(r)
        if len(simple) == 8:
            return np.array(simple)

def e8_roots():
    """直接构造 E8 的 240 个根: 112 个二进制型 + 128 个半整数型(偶数个负号)"""
    roots = []
    import itertools
    for i, j in itertools.combinations(range(8), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(8); v[i] = float(si); v[j] = float(sj)
                roots.append(v)
    for signs in itertools.product((1, -1), repeat=8):
        if sum(1 for s in signs if s < 0) % 2 == 0:
            roots.append(np.array(signs, float) / 2)
    roots = np.unique(np.round(np.array(roots), 10), axis=0)
    assert len(roots) == 240, f"E8 roots = {len(roots)}"
    simple = _select_simple_roots(roots)
    return roots, simple

def coxeter_project(roots, simple=None):
    """Coxeter 元投影: 本征值 e^(±2πi/30) 的本征向量 -> 2D"""
    if simple is None:
        simple = _select_simple_roots(roots)
    s = []
    for a in simple:
        s.append(np.eye(8) - 2 * np.outer(a, a) / (a @ a))
    C = np.eye(8)
    for m in s:
        C = C @ m
    w, v = np.linalg.eig(C)
    # 目标本征值
    target = np.exp(2j * np.pi / 30)
    idx = np.argmin(np.abs(w - target))
    u = v[:, idx]
    z = roots @ u
    pts = np.stack([z.real, -z.imag], 1)
    return pts

_cache = None
def get_e8():
    global _cache
    if _cache is None:
        roots, simple = e8_roots()
        pts = coxeter_project(roots, simple)
        r = np.max(np.abs(pts))
        pts = pts / r * 0.98          # 归一化到 [-1,1]
        _cache = pts
    return _cache

if __name__ == "__main__":
    pts = get_e8()
    print("roots projected:", pts.shape)
    print("range:", pts.min(), pts.max())
    # 对称性检查: 旋转 2π/15 应近似不动(作为集合)
    k = 2 * np.pi / 15
    R = np.array([[np.cos(k), -np.sin(k)], [np.sin(k), np.cos(k)]])
    rotated = pts @ R.T
    from scipy.spatial import cKDTree
    t1 = cKDTree(pts)
    d, _ = t1.query(rotated)
    print("max dist under 24-fold rotation (should be ~0):", d.max())
