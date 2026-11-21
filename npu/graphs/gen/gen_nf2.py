#!/usr/bin/env python3
"""NF round 2: cheaper graphs for the full FSR 4.1.1 network, same I/O bytes as nf_* (x and out: 960x540x16 int8^0x80 NHWC;
the packed graphs declare them as 1x540x480x32 = the same bytes, pixel pairs, as G2 g2_xp2s).
Builds on gen_nf.py (prefix exec'd: L incl. real pass-13, forward_ref, G, fold, pieces, VALS, stats).
Changes vs nf_exact:
 (a) the two rounding views are ONE conv with doubled output channels [A | B | K] (input read once), then a 1x1 'sum' conv
     [I I] at scale 1 (exact integer, int8 clamp = shader clamp) instead of 2 convs + Add.
 (K) the bias constants (values 1, 64) travel WITH the activations: every exact tensor = [values, K]; each doubled conv copies K
     (weight 1, shift 0) and the sum conv passes it on, so no per-layer Concat(x, const) copy; only x, G0 get one Concat.
 (s) no Slice: kind h/i 3x3 runs on the whole tensor with zero weights on the upper channels (grouped 3x3 = block-diagonal),
     the 1x1 expansion reads Concat(hc, y) with zero weights on y[:lo].
 (b) pixel-pair packing (2 px x 16 ch = 32 ch) for the 960x540 layers: model_1, model_2, model_12 (3x3 = pack3x3 rule, 1x1 =
     block-diagonal), model_3 reads packed X2 with a (2,1)/(2,1) conv; pass-23 tail skip S2D = identity (2,1)/(2,1) conv on
     packed X2; pass-23 tail output DepthToSpace -> identity (1,2)/(1,2) pack conv (+K by bias, scale 1 = exact) -> packed H0.
 (c) checked by ties.py: every conv of the network has tie-possible channels and real ties on the sim inputs -> no layer may
     use the single-view path; nothing is dropped.
Graphs: nf2_plain (b + s, single conv, HTP rounding), nf2_exact (a + K + s + b), nf2_exactu (a + K + s, not packed);
_d = + 12 pass outputs (values only) for the simulator. Usage (docker h-qnn): gen_nf2.py OUTDIR"""
import sys, os, numpy as np
OD = sys.argv[1]; os.makedirs(OD, exist_ok=True)
src = open('/nf/gen_nf.py').read().split('\nrows = []')[0]
sys.argv = ['gen_nf', OD]; ns = {'__name__': 'nf', '__file__': '/nf/gen_nf.py'}
exec(compile(src, 'gen_nf_prefix', 'exec'), ns)
L, G, fold, VALS, stats, NAMES, SHAPES = (ns[k] for k in ('L', 'G', 'fold', 'VALS', 'stats', 'NAMES', 'SHAPES'))
import onnx
from onnx import TensorProto as TP, helper as hp
K = len(VALS)
GRP = os.environ.get('SUM') == 'grp'
CBCAT = os.environ.get('CB') == 'cat'   # nf4: no K carried; Concat(inputs, const) per conv (as nf_exact)
SLICE = os.environ.get('SLICE') == '1'   # nf4: kind h/i read y[:lo] / y[lo:] through Slices (as nf_exact), not zero weights
HWd = ns['HW']


class A:   # physical tensor: P pixel groups x C logical channels (phys p*C+c), then K const channels if k
    def __init__(s, name, P, C, k, idx=None):
        s.name, s.P, s.C, s.k = name, P, C, k; s._idx = idx
        s.n = (len(set(idx(p, c) for p in range(P) for c in range(C))) + (K if k else 0)) if idx else P * C + (K if k else 0)
        s.kb = (s.n - K) if k else None
    def idx(s, p, c): return s._idx(p, c) if s._idx else p * C_(s) + c
def C_(t): return t.C


class NG(G):
    def __init__(s, mode, pack):
        G.__init__(s, 'plain' if mode == 'plain' else 'exact'); s.ex, s.pack = mode != 'plain', pack
    def kconst(s, t, hw):   # Concat(t, const) once
        if not s.ex or t.k or CBCAT: return t
        return A(s.cat([t.name, s.const(hw)], t.name + '_k'), t.P, t.C, True)
    def L2(s, ins, W, b, kind, sh, tag, relu=False, Pout=None, keepk=True):
        """logical conv W (o, sum C_t, kh, kw) on inputs ins (same P); kind 1x1 / 3x3 (pad 1) / d2 (2x2 s2) / pk (pack 1x2)"""
        P = ins[0].P
        if s.ex and CBCAT and not any(t.k for t in ins):
            H_, W_ = HWd[tag[:3]]; ins = list(ins) + [A(s.const((H_, W_ // P)), P, 0, True)]; keepk = False
        o = W.shape[0]; b = np.asarray(b, np.int64); sh = np.broadcast_to(np.asarray(sh, np.int64), (o,)).copy()
        Po = (1 if kind == 'd2' else P) if Pout is None else Pout
        offs, n = [], 0
        for t in ins: offs.append(n); n += t.n
        if kind == '1x1': ks, st, pd = (1, 1), (1, 1), 0
        elif kind == '3x3': ks, st, pd = (3, 3), (1, 1), 1
        else: ks, st, pd = ((2, 2), (2, 2), 0) if P == 1 else ((2, 1), (2, 1), 0)
        ctap = {'1x1': (0, 0), '3x3': (1, 1), 'd2': (0, 0)}[kind]
        def build(bv):   # -> (Po*o, n, kh, kw) physical weights, bias coefficients on the first const block (exact) or None
            Wp = np.zeros((Po * o, n) + ks, np.int64); ci = 0
            for t, off in zip(ins, offs):
                Wt = W[:, ci:ci + t.C]; ci += t.C
                for po in range(Po):
                    for pi in range(P):
                        for c in range(t.C):
                            col = off + t.idx(pi, c); rows = slice(po * o, (po + 1) * o)
                            if kind == '1x1':
                                if pi == po: Wp[rows, col, 0, 0] = Wt[:, c, 0, 0]
                            elif kind == '3x3':
                                if P == 1: Wp[rows, col] = Wt[:, c]
                                else:
                                    for kx2 in range(3):
                                        kx = 2 * (kx2 - 1) + pi - po + 1
                                        if 0 <= kx <= 2: Wp[rows, col, :, kx2] = Wt[:, c, :, kx]
                            else:
                                if P == 1: Wp[rows, col] = Wt[:, c]
                                else: Wp[rows, col, :, 0] = Wt[:, c, :, pi]
            assert ci == W.shape[1], (tag, ci, W.shape)
            if bv is not None:
                kt = [(t, off) for t, off in zip(ins, offs) if t.k][0]; r = bv.copy(); co = np.zeros((o, K), np.int64)
                for j in range(2, K): co[:, j] = np.clip(np.fix(r / 127.0), -127, 127); r = r - 127 * co[:, j]
                co[:, 1] = np.round(r / 64.0); co[:, 0] = r - 64 * co[:, 1]
                assert np.abs(co).max() <= 127 and ((co * np.array(VALS)).sum(1) == bv).all(), tag
                for po in range(Po): Wp[po * o:(po + 1) * o, kt[1] + kt[0].kb:kt[1] + kt[0].kb + K, ctap[0], ctap[1]] = co
            return Wp
        x = s.cat([t.name for t in ins], tag + '_in')
        if not s.ex:
            y = s.conv_p(x, build(None), np.tile(b, Po), ks, st, pd, np.tile(sh, Po), tag, relu)
            return A(y, Po, o, False)
        hh = np.where(sh > 0, 1 << np.maximum(sh - 1, 0), 0)
        bA = np.where(sh > 0, b + hh - 1, b); bB = np.where(sh > 0, b - hh, b - 1)
        WA, WB = build(bA), build(bB); M = Po * o; sh1 = np.tile(sh + 1, Po)
        if GRP:   # interleaved [A0 B0 A1 B1 .. K0 0 K1 0], sum = grouped 1x1 conv (groups of 2 channels, weights 1 1 / 1 0)
            nk = K if keepk else 0; Wd = np.zeros((2 * (M + nk), n) + ks, np.int64); shd = np.zeros(2 * (M + nk), np.int64)
            Wd[0:2 * M:2], Wd[1:2 * M:2] = WA, WB; shd[0:2 * M:2] = sh1; shd[1:2 * M:2] = sh1
            if keepk:
                kt = [(t, off) for t, off in zip(ins, offs) if t.k][0]
                for j in range(K): Wd[2 * (M + j), kt[1] + kt[0].kb + j, ctap[0], ctap[1]] = 1
            D = s.conv_p(x, Wd, np.zeros(Wd.shape[0], np.int64), ks, st, pd, shd, tag + 'D', relu)
            Ws = np.ones((M + nk, 2, 1, 1), np.int64); Ws[M:, 1] = 0
            y = s.conv_p(D, Ws, np.zeros(M + nk, np.int64), (1, 1), (1, 1), 0, np.zeros(M + nk, np.int64), tag + 'S', group=M + nk)
            return A(y, Po, o, keepk)
        Wd = [WA, WB]; shd = [sh1] * 2
        if keepk:
            kt = [(t, off) for t, off in zip(ins, offs) if t.k][0]; Wk = np.zeros((K, n) + ks, np.int64)
            for j in range(K): Wk[j, kt[1] + kt[0].kb + j, ctap[0], ctap[1]] = 1
            Wd.append(Wk); shd.append(np.zeros(K, np.int64))
        Wd = np.concatenate(Wd); shd = np.concatenate(shd)
        D = s.conv_p(x, Wd, np.zeros(Wd.shape[0], np.int64), ks, st, pd, shd, tag + 'D', relu)
        Ws = np.zeros((M + (K if keepk else 0), Wd.shape[0], 1, 1), np.int64)
        for j in range(M): Ws[j, j, 0, 0] = 1; Ws[j, M + j, 0, 0] = 1
        if keepk:
            for j in range(K): Ws[M + j, 2 * M + j, 0, 0] = 1
        y = s.conv_p(D, Ws, np.zeros(Ws.shape[0], np.int64), (1, 1), (1, 1), 0, np.zeros(Ws.shape[0], np.int64), tag + 'S')
        return A(y, Po, o, keepk)
    def conv_p(s, x, w, b, k, st, pd, sh, tag, relu=False, group=1):   # G.conv with (sy, sx) strides
        o = w.shape[0]; ws = (2.0 ** -np.asarray(sh, np.float64)).astype(np.float32); assert w.min() >= -128 and w.max() <= 127, tag
        from onnx import numpy_helper as nh
        s.c(tag + '_wq', w.astype(np.int8)); s.c(tag + '_bq', np.asarray(b).astype(np.int32))
        s.nodes.append(hp.make_node('DequantizeLinear', [tag + '_wq', s.c(tag + '_ws', ws), s.c(tag + '_wz', np.zeros(o, np.int8))], [tag + '_w'], axis=0))
        s.nodes.append(hp.make_node('DequantizeLinear', [tag + '_bq', s.c(tag + '_bs', ws), s.c(tag + '_bz', np.zeros(o, np.int32))], [tag + '_b'], axis=0))
        s.nodes.append(hp.make_node('Conv', [x, tag + '_w', tag + '_b'], [tag + '_y'], kernel_shape=list(k), strides=list(st), pads=[pd] * 4, group=group))
        return s.qdq(tag + '_y', tag + '_o', u8=relu)


def blk2(g, v, y, tag):
    if v['kind'] == 'd': return g.L2([y], v['w'], v['b'], 'd2', v['sh'], tag)
    lo = v['lo']; C = y.C; W1 = v['w1']
    if v.get('group', 1) == 2: W1 = np.zeros((32, 32, 3, 3), np.int64); W1[:16, :16] = v['w1'][:16]; W1[16:, 16:] = v['w1'][16:]
    if SLICE and v['kind'] != 'g':
        ylo = A(g.slice(y.name, 0, lo, tag + '_lo'), 1, lo, False); yhi = A(g.slice(y.name, lo, 2 * lo, tag + '_hi'), 1, lo, False)
        hc = g.L2([ylo], W1, v['b1'], '3x3', v['s1'], tag + 'c3')
        m = g.L2([hc, yhi], v['w2'], v['b2'], '1x1', v['s2'], tag + 'e', relu=True)
        Wf, P = fold(v['w3'], v['R']); return g.L2([m] + [y] * P, Wf, v['b3'], '1x1', v['sf'], tag + 'f')
    W1f = np.zeros((lo, C, 3, 3), np.int64); W1f[:, :lo] = W1
    hc = g.L2([y], W1f, v['b1'], '3x3', v['s1'], tag + 'c3')
    if v['kind'] == 'g': m = g.L2([hc], v['w2'], v['b2'], '1x1', v['s2'], tag + 'e', relu=True)
    else:
        W2 = np.zeros((v['w2'].shape[0], lo + C, 1, 1), np.int64); W2[:, :lo] = v['w2'][:, :lo]; W2[:, 2 * lo:] = v['w2'][:, lo:]
        m = g.L2([hc, y], W2, v['b2'], '1x1', v['s2'], tag + 'e', relu=True)
    Wf, P = fold(v['w3'], v['R'])
    return g.L2([m] + [y] * P, Wf, v['b3'], '1x1', v['sf'], tag + 'f')


def tail2(g, v, y, skip, tag):
    C4 = v['w4'].shape[0]; Cs = C4 // 4
    if skip.P == 1:   # p19: SpaceToDepth of E2 (+K block interleaved: phys q*(Cs+K)+c)
        nn = skip.n; s2 = g.qdq(g.node('SpaceToDepth', [skip.name], tag + '_s2d', blocksize=2), tag + '_s2dq')
        sd = A(s2, 1, C4, skip.k, idx=lambda p, c, nn=nn, Cs=Cs: (c // Cs) * nn + c % Cs)
        if skip.k: sd.kb, sd.n = Cs, 4 * nn
    else:             # p23: packed X2 -> identity (2,1)/(2,1) conv -> 64 ch (2dy+dx)*16+c, no K
        Wi = np.zeros((C4, skip.n, 2, 1), np.int64)
        for dy in range(2):
            for dx in range(2):
                for c in range(Cs): Wi[(2 * dy + dx) * Cs + c, skip.idx(dx, c), dy, 0] = 1
        sd = A(g.conv_p(skip.name, Wi, np.zeros(C4), (2, 1), (2, 1), 0, np.zeros(C4, np.int64), tag + '_s2d'), 1, C4, False)
    Wt, P = fold(v['w4'], np.full(C4, v['t_R']))
    u = g.L2([y] + [sd] * P, Wt, v['b4'], '1x1', v['t_sf'], tag, keepk=False)
    d = g.qdq(g.node('DepthToSpace', [u.name], tag + '_d2s', blocksize=2, mode='DCR'), tag + '_d2sq')
    if tag.startswith('p19'): return g.kconst(A(d, 1, Cs, False), (270, 480))
    if not g.pack: return g.kconst(A(d, 1, Cs, False), (540, 960))
    EK = g.ex and not CBCAT; Wp = np.zeros((2 * Cs + (K if EK else 0), Cs, 1, 2), np.int64); bp = np.zeros(Wp.shape[0], np.int64)
    for p in range(2):
        for c in range(Cs): Wp[p * Cs + c, c, 0, p] = 1
    if EK: bp[2 * Cs:] = VALS
    return A(g.conv_p(d, Wp, bp, (1, 2), (1, 2), 0, np.zeros(Wp.shape[0], np.int64), tag + '_pk'), 2, Cs, EK)


def build2(name, mode, pack, dbg):
    g = NG(mode, pack); P = 2 if pack else 1; Hx, Wx = (540, 480) if pack else (540, 960)
    x = g.kconst(A(g.qdq('x', 'xq'), P, 16, False), (Hx, Wx)); T = {}
    T['X1'] = blk2(g, L['p03'], x, 'p03'); T['X2'] = blk2(g, L['p05'], T['X1'], 'p05'); T['D0'] = blk2(g, L['p07'], T['X2'], 'p07')
    T['E1'] = blk2(g, L['p09'], T['D0'], 'p09'); T['E2'] = blk2(g, L['p11'], T['E1'], 'p11'); T['F0'] = blk2(g, L['p13'], T['E2'], 'p13')
    T['F1'] = blk2(g, L['p15'], T['F0'], 'p15'); T['F2'] = blk2(g, L['p17'], T['F1'], 'p17')
    T['G0'] = tail2(g, L['p19'], blk2(g, L['p19'], T['F2'], 'p19'), T['E2'], 'p19t'); T['G1'] = blk2(g, L['p21'], T['G0'], 'p21')
    T['H0'] = tail2(g, L['p23'], blk2(g, L['p23'], T['G1'], 'p23'), T['X2'], 'p23t'); T['OUT'] = blk2(g, L['p25'], T['H0'], 'p25')
    outs = NAMES if dbg else ['OUT']; onames = [('o_' + k) if k != 'OUT' else 'out' for k in outs]; shp = {}
    for k, on in zip(outs, onames):
        t = T[k]; c, hh, ww = SHAPES[k]
        if t.P == 2: c, ww = 2 * c, ww // 2
        src_ = t.name
        if t.k: src_ = g.qdq(g.slice(t.name, 0, t.P * t.C, on + '_v'), on + '_vq')
        g.nodes.append(hp.make_node('Identity', [src_], [on])); shp[on] = (c, hh, ww)
    vi = lambda n, sh_: hp.make_tensor_value_info(n, TP.FLOAT, [1] + list(sh_))
    m_ = hp.make_model(hp.make_graph(g.nodes, name, [vi('x', (32, 540, 480) if pack else (16, 540, 960))], [vi(on, shp[on]) for on in onames], initializer=g.inits),
                       opset_imports=[hp.make_opsetid('', 13)], producer_name='fsr4r-NF2')
    m_.ir_version = 8; onnx.checker.check_model(m_)
    io = ''.join('- IOName: %s\n  Datatype: uint8\n  QuantParam:\n    Type: QNN_DEFINITION_DEFINED\n    Scale: 1.0\n    Offset: -128\n' % n for n in ['x'] + onames)
    onnx.save(m_, '%s/%s.onnx' % (OD, name)); open('%s/%s.io.yaml' % (OD, name), 'w').write(io); return m_


def padmac(m_):   # MAC as a 32-channel-tiled HMX computes it (in/out channels rounded up to 32; grouped convs counted per real MAC)
    from onnx import shape_inference, numpy_helper as nh
    mi = shape_inference.infer_shapes(m_); sh = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(mi.graph.value_info) + list(mi.graph.output)}
    ini = {i.name: nh.to_array(i).shape for i in m_.graph.initializer}; prod = {n.output[0]: n for n in m_.graph.node}; tot = 0
    r32 = lambda v: -(-v // 32) * 32
    for n in m_.graph.node:
        if n.op_type != 'Conv': continue
        o = sh[n.output[0]]; w = ini[prod[n.input[1]].input[0]]; grp = [a.i for a in n.attribute if a.name == 'group']
        px = o[2] * o[3]
        tot += px * o[1] * int(np.prod(w[1:])) if grp and grp[0] > 1 else px * r32(o[1]) * r32(w[1]) * w[2] * w[3]
    return tot
VAR = [v.split(':') for v in os.environ.get('VARIANTS', 'nf2_plain:plain:1:0 nf2_exact:exact:1:0 nf2_exactu:exact:0:0 nf2_plain_d:plain:1:1 nf2_exact_d:exact:1:1 nf2_exactu_d:exact:0:1').split()]
rows = []
for name, mode, pack, dbg in [(a, b, int(c), int(d)) for a, b, c, d in VAR]:
    m_ = build2(name, mode, pack, dbg); print('wrote', name)
    if not dbg: mac, byt, ops = stats(m_); rows.append('%-10s MAC %.2f G  HMX-padded MAC %.2f G  act bytes %.1f MB  ops %s' % (name, mac / 1e9, padmac(m_) / 1e9, byt / 1e6, ops))
for f in ('nf_plain', 'nf_exact', 'nf2_plain', 'nf2_exact', 'nf2_exactu', 'nf3_exact', 'nf3_exactu', 'g2_xp2s'):
    m_ = onnx.load('/g2/build/g2_xp2s.onnx' if f == 'g2_xp2s' else '%s/%s.onnx' % (OD, f)); mac, byt, ops = stats(m_)
    rows.append('%-10s MAC %.2f G  HMX-padded MAC %.2f G  act bytes %.1f MB' % (f, mac / 1e9, padmac(m_) / 1e9, byt / 1e6))
open(OD + '/cost_table_%s.txt' % os.environ.get('TAG', '2'), 'w').write('\n'.join(rows) + '\n'); print('\n'.join(rows))
if not os.environ.get('NO_ORT') and any(v[0] == 'nf2_plain_d' for v in VAR):   # ORT (QuantizeLinear = rhe) of the plain packed graph must equal the shader reference
    import onnxruntime as ort
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    xs = np.fromfile(OD + '/x0.raw', np.uint8).astype(np.int64) - 128; T = dict(np.load(OD + '/ref0.npz'))
    s = ort.InferenceSession(OD + '/nf2_plain_d.onnx', so, providers=['CPUExecutionProvider'])
    res = s.run(None, {'x': xs.reshape(540, 480, 32).transpose(2, 0, 1)[None].astype(np.float32)}); lines = []
    for k, v in zip(NAMES, res):
        a = np.rint(v[0]).transpose(1, 2, 0).astype(np.int64).ravel(); d = np.abs(a - T[k].astype(np.int64).ravel())
        lines.append('ORT nf2_plain_d %-3s vs numpy shader ref: bytes differ %d max %d' % (k, int((d > 0).sum()), int(d.max())))
    open(OD + '/ort_check2.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
