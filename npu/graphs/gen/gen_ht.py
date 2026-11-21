#!/usr/bin/env python3
"""HT: faster exact variants of NF nf4_exact (same integer math, same I/O bytes). Execs NF/gen_nf2.py (read-only) with nf4 flags
(SUM=grp CB=cat SLICE=1, packed) and hooks:
  T2  : pass-23 tail output DepthToSpace + (1,2)/(1,2) pack conv -> ONE ConvTranspose (2,1)/(2,1) identity 64->32 (packed H0 direct)
  S19 / D19: pass-19 tail SpaceToDepth -> identity 2x2 s2 conv / DepthToSpace -> identity ConvTranspose (2,2)/(2,2)
  P07S / S23 / P13S: stride-2 convs as 1x1 convs on a shared SpaceToDepth (+ identity ConvTranspose unpack for packed D0 / p23 skip)
  SL0: slice-free kind h/i (NF SLICE=0)
  P07U: pass-07 reads X2 unpacked by a ConvTranspose (1,2)/(1,2) identity 32->16 (2x2 s2 conv on 540x960x16, as nf4_exactu's p07)
Identity weights 0/1 at scale 1 -> exact by construction; verified on the x86 HTP sim (all 12 layers, 2 inputs) and on the device.
Usage (docker h-qnn): VARIANTS='name:flags:dbg ...' gen_ht.py OUTDIR      flags = '+'-joined of T2, P07U, or 'base'"""
import sys, os, numpy as np
OD = sys.argv[1]; os.makedirs(OD, exist_ok=True)
os.environ.update(SUM='grp', CB='cat', SLICE='1')   # SL0 flag below switches SLICE off per variant
src = open('/nf/gen_nf2.py').read().split('\ndef padmac')[0]
a = "T['D0'] = blk2(g, L['p07'], T['X2'], 'p07')"; assert a in src; src = src.replace(a, "T['D0'] = p07hook(g, T)")
a = "T['F0'] = blk2(g, L['p13'], T['E2'], 'p13')"; assert a in src; src = src.replace(a, "T['F0'] = p13hook(g, T)")
b = "    d = g.qdq(g.node('DepthToSpace', [u.name], tag + '_d2s', blocksize=2, mode='DCR'), tag + '_d2sq')"; assert b in src
c = "        nn = skip.n; s2 = g.qdq(g.node('SpaceToDepth', [skip.name], tag + '_s2d', blocksize=2), tag + '_s2dq')"; assert c in src
src = src.replace(c, "        nn = skip.n; s2 = s19hook(g, skip, C4, Cs, tag)")
e = "        sd = A(g.conv_p(skip.name, Wi, np.zeros(C4), (2, 1), (2, 1), 0, np.zeros(C4, np.int64), tag + '_s2d'), 1, C4, False)"; assert e in src
src = src.replace(e, "        sd = s23hook(g, skip, C4, Cs, tag) if 'S23' in FL else A(g.conv_p(skip.name, Wi, np.zeros(C4), (2, 1), (2, 1), 0, np.zeros(C4, np.int64), tag + '_s2d'), 1, C4, False)")
src = src.replace(b, "    if 'D19' in FL and tag.startswith('p19'): return A(convT(g, u.name, d19W(C4, Cs), (2, 2), (2, 2), tag + '_ct'), 1, Cs, False)\n" + b)
src = src.replace(b, "    if 'T2' in FL and tag.startswith('p23') and g.pack: return t2hook(g, u, C4, Cs, tag)\n" + b)
src += r'''
FL = set()
def convT(g, x, W, k, st, tag):   # identity remap, weights (Cin, Cout, kh, kw) in {0,1}, per-tensor scale 1, no bias -> exact
    g.c(tag + '_wq', W.astype(np.int8))
    g.nodes.append(hp.make_node('DequantizeLinear', [tag + '_wq', g.c(tag + '_ws', np.float32(1.0)), g.c(tag + '_wz', np.int8(0))], [tag + '_w']))
    g.nodes.append(hp.make_node('ConvTranspose', [x, tag + '_w'], [tag + '_y'], kernel_shape=list(k), strides=list(st)))
    return g.qdq(tag + '_y', tag + '_o')
def t2hook(g, u, C4, Cs, tag):
    assert u.n == C4 and not u.k
    W = np.zeros((C4, 2 * Cs, 2, 1), np.int64)
    for dy in range(2):
        for dx in range(2):
            for c in range(Cs): W[(2 * dy + dx) * Cs + c, dx * Cs + c, dy, 0] = 1
    return A(convT(g, u.name, W, (2, 1), (2, 1), tag + '_ct'), 2, Cs, False)
def s19hook(g, skip, C4, Cs, tag):   # S19: SpaceToDepth(2) of E2 -> identity 2x2 s2 conv (out ch (2dy+dx)*Cs+c)
    if 'S19' not in FL: return s2dE2(g, skip) if 'P13S' in FL else g.qdq(g.node('SpaceToDepth', [skip.name], tag + '_s2d', blocksize=2), tag + '_s2dq')
    assert skip.n == Cs and not skip.k
    W = np.zeros((C4, Cs, 2, 2), np.int64)
    for dy in range(2):
        for dx in range(2):
            for c in range(Cs): W[(2 * dy + dx) * Cs + c, c, dy, dx] = 1
    return g.conv_p(skip.name, W, np.zeros(C4, np.int64), (2, 2), (2, 2), 0, np.zeros(C4, np.int64), tag + '_s2c')
def d19W(C4, Cs):   # D19: DepthToSpace(2, DCR) -> identity ConvTranspose (2,2)/(2,2)
    W = np.zeros((C4, Cs, 2, 2), np.int64)
    for dy in range(2):
        for dx in range(2):
            for c in range(Cs): W[(2 * dy + dx) * Cs + c, c, dy, dx] = 1
    return W
def s2dX2(g, X2):   # ONE SpaceToDepth(2) of packed X2 (270x240x128, ch (2dy+dq)*32 + p*16 + c), shared by P07S and S23
    if not hasattr(g, '_s2x2'): g._s2x2 = g.qdq(g.node('SpaceToDepth', [X2.name], 'x2_s2d', blocksize=2), 'x2_s2dq')
    return g._s2x2
def s23hook(g, skip, C4, Cs, tag):   # S23: pass-23 skip S2D of X2 = shared S2D + identity ConvTranspose (1,2)/(1,2) 128->64
    assert skip.P == 2 and skip.n == 32 and C4 == 64 and Cs == 16
    W = np.zeros((128, 64, 1, 2), np.int64)
    for dy in range(2):
        for q in range(2):
            for p_ in range(2):
                for c in range(16): W[(2 * dy + q) * 32 + p_ * 16 + c, (2 * dy + p_) * 16 + c, 0, q] = 1
    return A(convT(g, s2dX2(g, skip), W, (1, 2), (1, 2), tag + '_s2u'), 1, C4, False)
def s2dE2(g, E2):   # P13S: ONE SpaceToDepth(2) of E2 shared by pass-13 and the pass-19 tail skip
    if not hasattr(g, '_s2e2'): g._s2e2 = (E2.name, g.qdq(g.node('SpaceToDepth', [E2.name], 'e2_s2d', blocksize=2), 'e2_s2dq'))
    assert g._s2e2[0] == E2.name; return g._s2e2[1]
def p13hook(g, T):   # P13S: pass-13 2x2 s2 conv = 1x1 conv on S2D(E2) (ch (2dy+dx)*C+c)
    E2 = T['E2']
    if 'P13S' not in FL: return blk2(g, L['p13'], E2, 'p13')
    v = L['p13']; w = v['w']; o, C = w.shape[:2]; assert v['kind'] == 'd' and E2.P == 1 and E2.n == C and not E2.k and w.shape[2:] == (2, 2)
    Wl = np.zeros((o, 4 * C, 1, 1), np.int64)
    for dy in range(2):
        for dx in range(2): Wl[:, (2 * dy + dx) * C:(2 * dy + dx + 1) * C, 0, 0] = w[:, :, dy, dx]
    HW = HWd; old = HW['p13']; HW['p13'] = (135, 240)
    try: return g.L2([A(s2dE2(g, E2), 1, 4 * C, False)], Wl, v['b'], '1x1', v['sh'], 'p13')
    finally: HW['p13'] = old
def p07hook(g, T):
    X2 = T['X2']
    if 'P07S' in FL:   # pass-07 2x2 s2 = 1x1 conv on S2D(X2) (2 output px per 270x240 cell, packed D0) -> ConvTranspose unpack
        v = L['p07']; w = v['w']; o = w.shape[0]; assert X2.P == 2 and X2.n == 32 and not X2.k and w.shape[1:] == (16, 2, 2)
        S = A(s2dX2(g, X2), 2, 64, False, idx=lambda q, c2: (2 * (c2 // 32) + q) * 32 + c2 % 32)
        Wl = np.zeros((o, 64, 1, 1), np.int64)
        for dy in range(2):
            for p_ in range(2): Wl[:, dy * 32 + p_ * 16:dy * 32 + p_ * 16 + 16, 0, 0] = w[:, :, dy, p_]
        HW = HWd; old = HW['p07']; HW['p07'] = (270, 480)
        try: Dp = g.L2([S], Wl, v['b'], '1x1', v['sh'], 'p07')
        finally: HW['p07'] = old
        assert Dp.P == 2 and Dp.n == 2 * o and not Dp.k
        Wu = np.zeros((2 * o, o, 1, 2), np.int64)
        for q in range(2):
            for c in range(o): Wu[Dp.idx(q, c), c, 0, q] = 1
        return A(convT(g, Dp.name, Wu, (1, 2), (1, 2), 'p07_u'), 1, o, False)
    if 'P07U' not in FL: return blk2(g, L['p07'], X2, 'p07')
    assert X2.P == 2 and X2.n == 32 and not X2.k
    W = np.zeros((32, 16, 1, 2), np.int64)
    for p in range(2):
        for c in range(16): W[X2.idx(p, c), c, 0, p] = 1
    return blk2(g, L['p07'], A(convT(g, X2.name, W, (1, 2), (1, 2), 'p07_u'), 1, 16, False), 'p07')
'''
sys.argv = ['gen_nf2', OD]; ns = {'ns_HW': lambda: ns['HW'], '__name__': 'ht', '__file__': '/nf/gen_nf2.py'}
exec(compile(src, 'gen_nf2_ht', 'exec'), ns)
for v in os.environ['VARIANTS'].split():
    name, fl, dbg = v.split(':'); ns['FL'].clear(); ns['FL'].update(f for f in fl.split('+') if f != 'base'); ns['SLICE'] = 'SL0' not in ns['FL']
    m_ = ns['build2'](name, 'exact', 1, int(dbg)); mac, byt, ops = ns['stats'](m_)
    print('wrote %-12s flags %-10s MAC %.2f G act %.1f MB ops %s' % (name, fl, mac / 1e9, byt / 1e6, ops))
