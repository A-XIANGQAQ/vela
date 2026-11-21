#!/usr/bin/env python3
"""N n1: FULL FSR 4.1.1 INT8 network (model_1..12 = pass-03..25, general/1080/performance, 960x540 -> network output 960x540x16)
as int8 QDQ ONNX for QNN/HTP v73, real weights (parse() of the fsr4vk generators -> params.pkl, x_params.py; biases and gen_i weights
from initializers.bin), shader integer math (round-half-even after >>sh, clamp int8, ReLU = uint8 zp0 tensor clamped by the next
int8 Concat), H gen_r6.py conventions (activations scale 1 / uint8 zp 128, weights per-channel 2^-sh, concat with own QDQ).
Dataflow (U-Net, from the scratch offsets of the passes):
  X1=p03(X0) X2=p05(X1) [960x540x16]  D0=p07(X2) 2x2s2 16->32  E1=p09(D0) E2=p11(E1) [480x270x32]  F0=p13(E2) 2x2s2 32->64
  F1=p15(F0) F2=p17(F1) [240x135x64]  G0=p19(F2)+skip(E2, RES@8342464 - E2 assumed, D0 possible)  G1=p21(G0)  H0=p23(G1)+skip X2
  OUT=p25(H0).  p13 has no generator: SYNTHETIC weights (shape exact, h07-like statistics, sh 7) in graph AND reference.
Residual out=rhe((a+R*x)/2^sf): variant fold = x as extra 1x1 input channels with weight 64 x (R/64) copies (exact);
variant add = Add(rhe(a/2^sf), c*x) per channel group (c=R/2^sf in {1,0.5}; double rounding on ties, saturation of u).
Both tails (p19 transposed 64->32, p23 transposed 32->16) = Conv1x1 + DepthToSpace(DCR) + Add (fold would cost +3 GMAC).
Usage: gen_n1.py OUTDIR [x_real.raw]  (writes n1_fold/n1_add .onnx + .io.yaml, ref_out.raw (numpy shader-math reference), table)"""
import sys, os, pickle, struct, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); od = sys.argv[1]; os.makedirs(od, exist_ok=True)
P = pickle.load(open(HERE + '/params.pkl', 'rb'))
INIT = open(os.environ.get('INIT_BIN', '/b/initializers.bin'), 'rb').read()
sb = lambda w: [((w >> (8 * i)) & 0xff) - (256 if (w >> (8 * i)) & 0x80 else 0) for i in range(4)]
s32 = lambda v: v - (1 << 32) if v & 0x80000000 else v
i32 = lambda uidx, n: np.array(struct.unpack_from('<%di' % n, INIT, 4 * uidx), np.int64)


def put(W, o, words):  # words: {(key...): word} with key -> (input base channel, dy, dx)
    for (ib, dy, dx), w in words:
        for i, b in enumerate(sb(w)): W[o, ib + i, dy, dx] = b


# ---------------- parameters of every pass in one form: list of stages
def gblk(p):  # gen.py passes 03/05/25: 3x3 16->16, 1x1 16->32 relu, 1x1 32->16 + R*x
    q = P[p]; c = q['cons']; w1 = np.zeros((16, 16, 3, 3), np.int64); w2 = np.zeros((32, 16, 1, 1), np.int64); w3 = np.zeros((16, 32, 1, 1), np.int64)
    for o, d in q['W1'].items(): put(w1, o, [((4 * j, dy, dx), w) for (dy, dx, j), w in d.items()])
    for o, d in q['W2'].items(): put(w2, o, [((4 * j, 0, 0), w) for j, w in d.items()])
    for o, d in q['W3'].items(): put(w3, o, [((4 * j, 0, 0), w) for j, w in d.items()])
    R = int(round(c['ar'] / c['ab'])); sh = c['D'].bit_length() - 1
    return dict(kind='g', s1=c['s1'], s2=c['s2'], w1=w1, b1=i32(q['b1'], 16), w2=w2, b2=i32(q['b2'], 32), w3=w3, b3=i32(q['b3'], 16),
                sf=np.full(16, sh), R=np.full(16, R), lo=16)


def hblk(p):  # h09.py passes 09/11/21: 3x3 on ch<16, 1x1 [h,ch>=16] 32->64 relu, 1x1 64->32 + R*y
    q = P[p]; W, B = q['W'], q['bias']; w1 = np.zeros((16, 16, 3, 3), np.int64); w2 = np.zeros((64, 32, 1, 1), np.int64); w3 = np.zeros((32, 64, 1, 1), np.int64)
    for o, d in W[1].items(): put(w1, o, [((4 * j, dy, dx), w) for (_, dy, dx, j), w in d.items()])
    for o, d in W[2].items(): put(w2, o, [((4 * j + (16 if t == 'b' else 0), 0, 0), w) for (t, j), w in d.items()])
    for o, d in W[3].items(): put(w3, o, [((4 * (j - 4), 0, 0), w) for (_, j), w in d.items()])
    bb = lambda k, n: np.array([s32(B[k][o]) for o in range(n)], np.int64)
    return dict(kind='h', s1=q['s1'], s2=q['s2'], w1=w1, b1=bb(1, 16), w2=w2, b2=bb(2, 64), w3=w3, b3=bb(3, 32),
                sf=np.array([q['sfR'][o // 4][0] for o in range(32)]), R=np.array([q['sfR'][o // 4][1] for o in range(32)]), lo=16)


def iblk(p):  # gen_i passes 15/17/19: grouped 3x3 2x(16->16) on ch<32, 1x1 64->128 relu, 1x1 128->64 + M*y
    q = P[p]; w1 = np.zeros((32, 16, 3, 3), np.int64)
    for t in range(9):
        for o in range(32): w1[o, :, t // 3, t % 3] = q['w1'][t][o]
    r = dict(kind='i', s1=q['s1'], s2=q['s2'], w1=w1, b1=np.array(q['b1'], np.int64), w2=np.array(q['w2'], np.int64)[:, :, None, None],
             b2=np.array(q['b2'], np.int64), w3=np.array(q['w3'], np.int64)[:, :, None, None], b3=np.array(q['b3'], np.int64),
             sf=np.array([q['sh'][o >> 5] for o in range(64)]), R=np.array([q['M'][o >> 5] for o in range(64)]), lo=32, group=2)
    if p == 'p19':   # tail: 2x2 transposed 64->32 + 256*RES >> 9 ; channel (2dy+dx)*32+oc
        r['w4'] = np.concatenate([np.array(q['w4'][sp], np.int64) for sp in range(4)])[:, :, None, None]; r['b4'] = np.tile(np.array(q['b4'], np.int64), 4)
        r['t_sf'], r['t_R'] = 9, 256
    return r


def h23blk():
    q = P['p23']; w1 = np.zeros((16, 16, 3, 3), np.int64); w2 = np.zeros((64, 32, 1, 1), np.int64); w3 = np.zeros((32, 64, 1, 1), np.int64); w4 = np.zeros((64, 32, 1, 1), np.int64)
    for o, (b, d) in enumerate(q['S1']): put(w1, o, [((4 * k, dy, dx), w) for (dy, dx, k), w in d.items()])
    for o, (b, d) in enumerate(q['S2']): put(w2, o, [((4 * k + (16 if t == 'b' else 0), 0, 0), w) for (t, k), w in d.items()])
    for o, (b, d) in enumerate(q['S3']): put(w3, o, [((4 * k, 0, 0), w) for k, w in d.items()])
    b4 = np.zeros(64, np.int64)
    for (dy, dx, oc), (b, d) in q['S4'].items():
        put(w4, (2 * dy + dx) * 16 + oc, [((4 * k, 0, 0), w) for k, w in d.items()]); b4[(2 * dy + dx) * 16 + oc] = b
    return dict(kind='h', s1=8, s2=8, w1=w1, b1=np.array([b for b, _ in q['S1']], np.int64), w2=w2, b2=np.array([b for b, _ in q['S2']], np.int64),
                w3=w3, b3=np.array([b for b, _ in q['S3']], np.int64), sf=np.full(32, 9), R=np.full(32, 256), lo=16, w4=w4, b4=b4, t_sf=7, t_R=128)


def h07():
    W, bias = P['p07']; w = np.zeros((32, 16, 2, 2), np.int64)
    for o, d in W.items(): put(w, o, [((4 * j, dy, dx), v) for (dy, dx, j), v in d.items()])
    return dict(kind='d', w=w, b=np.array([s32(bias[o]) for o in range(32)], np.int64), sh=7)


def p13_synth(ref):
    rng = np.random.default_rng(13); sd = float(ref['w'].std())
    w = np.clip(np.rint(rng.normal(0, sd * 0.7, (64, 32, 2, 2))), -127, 127).astype(np.int64)
    return dict(kind='d', w=w, b=np.zeros(64, np.int64), sh=7, synthetic=True)


L = {}
for p in ('p03', 'p05', 'p25'): L[p] = gblk(p)
L['p07'] = h07(); L['p13'] = p13_synth(L['p07'])
for p in ('p09', 'p11', 'p21'): L[p] = hblk(p)
for p in ('p15', 'p17', 'p19'): L[p] = iblk(p)
L['p23'] = h23blk()
for p, v in sorted(L.items()):
    bs = [k for k in v if k.startswith('b')]
    print('%s %s biases max|b| %s  weights max|w| %s' % (p, v['kind'], {k: int(np.abs(v[k]).max()) for k in bs},
                                                        {k: int(np.abs(v[k]).max()) for k in v if k.startswith('w')}))


# ---------------- numpy reference (HWC int64), the shader math
def rq(a, s, relu=False):
    s = np.asarray(s, np.int64)
    if relu: a = np.maximum(a, 0)
    return np.clip((a + (1 << (s - 1)) - 1 + ((a >> s) & 1)) >> s, -128, 127)


def conv(x, w, b, stride=1, pad=0, group=1):
    O, I, kh, kw = w.shape; H, Wd, C = x.shape
    if pad: x = np.pad(x, ((pad, pad), (pad, pad), (0, 0)))
    Ho, Wo = (x.shape[0] - kh) // stride + 1, (x.shape[1] - kw) // stride + 1
    acc = np.broadcast_to(b, (Ho, Wo, O)).astype(np.int64).copy(); og = O // group
    for g in range(group):
        for dy in range(kh):
            for dx in range(kw):
                xs = x[dy:dy + stride * Ho:stride, dx:dx + stride * Wo:stride, g * I:(g + 1) * I].reshape(-1, I)
                acc[..., g * og:(g + 1) * og] += (xs @ w[g * og:(g + 1) * og, :, dy, dx].T).reshape(Ho, Wo, og)
    return acc


def d2s(a):  # DCR: channel (2dy+dx)*C + c
    H, W, C4 = a.shape; C = C4 // 4
    return a.reshape(H, W, 2, 2, C).transpose(0, 2, 1, 3, 4).reshape(2 * H, 2 * W, C)


def ref_block(v, y):
    if v['kind'] == 'd': return rq(conv(y, v['w'], v['b'], 2), v['sh'])
    lo = v['lo']; xin = y if v['kind'] == 'g' else y[..., :lo]
    h = rq(conv(xin, v['w1'], v['b1'], 1, 1, v.get('group', 1)), v['s1'])
    cat = h if v['kind'] == 'g' else np.concatenate([h, y[..., lo:]], -1)
    m = rq(conv(cat, v['w2'], v['b2']), v['s2'], relu=True)
    return rq(conv(m, v['w3'], v['b3']) + v['R'] * y, v['sf'])


def ref_tail(v, y, res):
    return rq(d2s(conv(y, v['w4'], v['b4'])) + v['t_R'] * res, v['t_sf'])


def forward_ref(x0):
    T = {}
    T['X1'] = ref_block(L['p03'], x0); T['X2'] = ref_block(L['p05'], T['X1']); T['D0'] = ref_block(L['p07'], T['X2'])
    T['E1'] = ref_block(L['p09'], T['D0']); T['E2'] = ref_block(L['p11'], T['E1']); T['F0'] = ref_block(L['p13'], T['E2'])
    T['F1'] = ref_block(L['p15'], T['F0']); T['F2'] = ref_block(L['p17'], T['F1'])
    T['G0'] = ref_tail(L['p19'], ref_block(L['p19'], T['F2']), T['E2']); T['G1'] = ref_block(L['p21'], T['G0'])
    T['H0'] = ref_tail(L['p23'], ref_block(L['p23'], T['G1']), T['X2']); T['OUT'] = ref_block(L['p25'], T['H0'])
    return T


# ---------------- layer table (960x540 -> 1080p performance)
HW = {'p03': (540, 960), 'p05': (540, 960), 'p25': (540, 960), 'p07': (270, 480), 'p09': (270, 480), 'p11': (270, 480), 'p21': (270, 480),
      'p23': (270, 480), 'p13': (135, 240), 'p15': (135, 240), 'p17': (135, 240), 'p19': (135, 240)}
tab = ['pass model out(HxWxC) layers | MAC (net) | MAC fold-graph']
tot = [0, 0]
for i, p in enumerate(('p03', 'p05', 'p07', 'p09', 'p11', 'p13', 'p15', 'p17', 'p19', 'p21', 'p23', 'p25')):
    v = L[p]; H_, W_ = HW[p]; px = H_ * W_
    if v['kind'] == 'd':
        mac = px * v['w'].size; desc = 'conv2x2s2 %d->%d sh7%s' % (v['w'].shape[1], v['w'].shape[0], ' (SYNTHETIC weights)' if v.get('synthetic') else '')
        fold = mac; oc = v['w'].shape[0]
    else:
        g = v.get('group', 1); m1 = v['w1'].size; m2 = v['w2'].size; m3 = v['w3'].size; ncp = int(v['R'].max() // 64); C = v['w3'].shape[0]
        mac = px * (m1 + m2 + m3 + C); fold = px * (m1 + m2 + m3 + C * C * ncp)
        desc = '%s3x3 %d->%d s%d | 1x1 %d->%d relu s%d | 1x1 %d->%d +R%s>>sf%s' % ('grouped(2) ' if g > 1 else '', v['w1'].shape[1] * g, v['w1'].shape[0], v['s1'],
               v['w2'].shape[1], v['w2'].shape[0], v['s2'], v['w3'].shape[1], C, sorted(set(v['R'].tolist())), sorted(set(v['sf'].tolist())))
        oc = C
        if 'w4' in v:
            t = px * v['w4'].size; mac += t; fold += t; desc += ' | tail convT2x2 %d->%d +%d*skip >>%d' % (v['w4'].shape[1], v['w4'].shape[0] // 4, v['t_R'], v['t_sf'])
            H_, W_, oc = 2 * H_, 2 * W_, v['w4'].shape[0] // 4
    tot[0] += mac; tot[1] += fold
    tab.append('pass-%s model_%-2d %4dx%-4dx%-3d %s | %.3f G | %.3f G' % (p[1:], i + 1, H_, W_, oc, desc, mac / 1e9, fold / 1e9))
tab.append('TOTAL network MAC %.2f G (fold graph %.2f G)' % (tot[0] / 1e9, tot[1] / 1e9))
open(od + '/layer_table.txt', 'w').write('\n'.join(tab) + '\n'); print('\n'.join(tab))


# ---------------- ONNX (QDQ) graph
try:
    import onnx
    from onnx import TensorProto as TP, helper as hp, numpy_helper as nh
except ImportError:
    onnx = None


class G:
    def __init__(s): s.nodes, s.inits, s.n = [], [], 0

    def c(s, name, arr): s.inits.append(nh.from_array(np.asarray(arr), name)); return name

    def qdq(s, x, tag, u8=False, scale=1.0):
        sc, z = s.c(tag + '_s', np.float32(scale)), s.c(tag + '_z', np.uint8(0) if u8 else np.uint8(128))
        s.nodes.append(hp.make_node('QuantizeLinear', [x, sc, z], [tag + '_q'])); s.nodes.append(hp.make_node('DequantizeLinear', [tag + '_q', sc, z], [tag]))
        return tag

    def dq_as(s, xq_tag, scale, tag):  # second view of an existing int8 tensor with another scale (x * c without touching the bytes)
        s.nodes.append(hp.make_node('DequantizeLinear', [xq_tag + '_q', s.c(tag + '_s', np.float32(scale)), s.c(tag + '_z', np.uint8(128))], [tag])); return tag

    def node(s, op, ins, tag, **kw): s.nodes.append(hp.make_node(op, ins, [tag], **kw)); return tag

    def conv(s, x, w, b, k, stride, pad, sh, tag, relu=False, group=1, oscale=1.0):
        o = w.shape[0]; sh = np.full(o, sh) if np.isscalar(sh) else np.asarray(sh)
        ws = (2.0 ** -sh.astype(np.float64)).astype(np.float32)
        assert w.min() >= -128 and w.max() <= 127
        s.c(tag + '_wq', w.astype(np.int8)); s.c(tag + '_bq', b.astype(np.int32))
        s.nodes.append(hp.make_node('DequantizeLinear', [tag + '_wq', s.c(tag + '_ws', ws), s.c(tag + '_wz', np.zeros(o, np.int8))], [tag + '_w'], axis=0))
        s.nodes.append(hp.make_node('DequantizeLinear', [tag + '_bq', s.c(tag + '_bs', ws), s.c(tag + '_bz', np.zeros(o, np.int32))], [tag + '_b'], axis=0))
        s.nodes.append(hp.make_node('Conv', [x, tag + '_w', tag + '_b'], [tag + '_y'], kernel_shape=list(k), strides=[stride] * 2, pads=[pad] * 4, group=group))
        return s.qdq(tag + '_y', tag + '_o', u8=relu, scale=oscale)

    def slice(s, x, a, b, tag):
        return s.node('Slice', [x, s.c(tag + '_a', np.array([a], np.int64)), s.c(tag + '_b', np.array([b], np.int64)), s.c(tag + '_x', np.array([1], np.int64))], tag)

    def cat(s, xs, tag): return s.qdq(s.node('Concat', xs, tag + '_c', axis=1), tag)


def groups(sf, R):  # consecutive channel ranges with equal (sf, R)
    out, st = [], 0
    for i in range(1, len(sf) + 1):
        if i == len(sf) or (sf[i], R[i]) != (sf[st], R[st]): out.append((st, i, int(sf[st]), int(R[st]))); st = i
    return out


def residual_add(g, m, y, w, b, sf, R, tag, k=(1, 1), up=False):
    """out = rhe((conv(m) + R*y) >> sf), per group: c = R/2^sf; u = conv with weight scale 2^-sf (c=1) -> Add(u, y)
       or c=0.5: Add(u, y viewed at scale 0.5) -- the Add requantises once (ties / u saturation differ from the shader)"""
    parts = []
    for a, e, s_, r_ in groups(sf, R) if not up else [(0, w.shape[0], int(sf[0]), int(R[0]))]:
        cc = r_ / 2.0 ** s_; assert cc in (1.0, 0.5), (tag, s_, r_)
        u = g.conv(m, w[a:e], b[a:e], k, 1, 0, s_, '%s_g%d' % (tag, a))
        if up: u = g.qdq(g.node('DepthToSpace', [u], tag + '_d2s', blocksize=2, mode='DCR'), tag + '_d2sq')
        ys = y if (a, e) == (0, w.shape[0]) or up else g.qdq(g.slice(y, a, e, '%s_ys%d' % (tag, a)), '%s_ysq%d' % (tag, a))
        yv = ys if cc == 1.0 else g.dq_as(ys, 0.5, '%s_yh%d' % (tag, a))
        parts.append(g.qdq(g.node('Add', [u, yv], '%s_add%d' % (tag, a)), '%s_o%d' % (tag, a)))
    return parts[0] if len(parts) == 1 else g.cat(parts, tag + '_cat')


def onnx_block(g, v, y, tag, mode):
    if v['kind'] == 'd': return g.conv(y, v['w'], v['b'], (2, 2), 2, 0, v['sh'], tag)
    lo = v['lo']; C = v['w3'].shape[0]
    if v['kind'] == 'g':
        h = g.conv(y, v['w1'], v['b1'], (3, 3), 1, 1, v['s1'], tag + 'c3'); cat = h
    else:
        h = g.conv(g.qdq(g.slice(y, 0, lo, tag + '_lo'), tag + '_loq'), v['w1'], v['b1'], (3, 3), 1, 1, v['s1'], tag + 'c3', group=v.get('group', 1))
        cat = g.cat([h, g.qdq(g.slice(y, lo, 2 * lo if v['kind'] == 'h' else 2 * lo, tag + '_hi'), tag + '_hiq')], tag + '_hc')
    m = g.conv(cat, v['w2'], v['b2'], (1, 1), 1, 0, v['s2'], tag + 'e', relu=True)
    if mode == 'add':
        return residual_add(g, g.qdq(m, tag + '_mc'), y, v['w3'], v['b3'], v['sf'], v['R'], tag + 'f')
    ncp = int(v['R'].max() // 64); Wf = np.zeros((C, v['w3'].shape[1] + ncp * C, 1, 1), np.int64); Wf[:, :v['w3'].shape[1]] = v['w3']
    for k_ in range(ncp):
        for o in range(C):
            if k_ < v['R'][o] // 64: Wf[o, v['w3'].shape[1] + k_ * C + o] = 64
    return g.conv(g.cat([m] + [y] * ncp, tag + '_fc'), Wf, v['b3'], (1, 1), 1, 0, v['sf'], tag + 'f')


def onnx_tail(g, v, y, skip, tag):
    return residual_add(g, y, skip, v['w4'], v['b4'], np.full(v['w4'].shape[0], v['t_sf']), np.full(v['w4'].shape[0], v['t_R']), tag, up=True)


def build(name, mode):
    g = G(); x = g.qdq('x', 'xq'); T = {}
    T['X1'] = onnx_block(g, L['p03'], x, 'p03', mode); T['X2'] = onnx_block(g, L['p05'], T['X1'], 'p05', mode)
    T['D0'] = onnx_block(g, L['p07'], T['X2'], 'p07', mode)
    T['E1'] = onnx_block(g, L['p09'], T['D0'], 'p09', mode); T['E2'] = onnx_block(g, L['p11'], T['E1'], 'p11', mode)
    T['F0'] = onnx_block(g, L['p13'], T['E2'], 'p13', mode)
    T['F1'] = onnx_block(g, L['p15'], T['F0'], 'p15', mode); T['F2'] = onnx_block(g, L['p17'], T['F1'], 'p17', mode)
    T['G0'] = onnx_tail(g, L['p19'], onnx_block(g, L['p19'], T['F2'], 'p19', mode), T['E2'], 'p19t')
    T['G1'] = onnx_block(g, L['p21'], T['G0'], 'p21', mode)
    T['H0'] = onnx_tail(g, L['p23'], onnx_block(g, L['p23'], T['G1'], 'p23', mode), T['X2'], 'p23t')
    out = onnx_block(g, L['p25'], T['H0'], 'p25', mode)
    g.nodes.append(hp.make_node('Identity', [out], ['out']))
    vi = lambda n, c, hh, ww: hp.make_tensor_value_info(n, TP.FLOAT, [1, c, hh, ww])
    m_ = hp.make_model(hp.make_graph(g.nodes, name, [vi('x', 16, 540, 960)], [vi('out', 16, 540, 960)], initializer=g.inits),
                       opset_imports=[hp.make_opsetid('', 13)], producer_name='fsr4r-N1')
    m_.ir_version = 8; onnx.checker.check_model(m_); return m_


io_yaml = ''.join('- IOName: %s\n  Datatype: uint8\n  QuantParam:\n    Type: QNN_DEFINITION_DEFINED\n    Scale: 1.0\n    Offset: -128\n' % n for n in ('x', 'out'))
for n, mode in (('n1_fold', 'fold'), ('n1_add', 'add')) if onnx else ():
    m_ = build(n, mode); onnx.save(m_, '%s/%s.onnx' % (od, n)); open('%s/%s.io.yaml' % (od, n), 'w').write(io_yaml)
    ops = [nd.op_type for nd in m_.graph.node]; print('wrote', n, len(ops), {o: ops.count(o) for o in ('Conv', 'Concat', 'Add', 'Slice', 'DepthToSpace')})
if len(sys.argv) > 2 and not os.environ.get('NO_REF'):
    x0 = (np.fromfile(sys.argv[2], np.uint8).astype(np.int64) - 128).reshape(540, 960, 16)
    T = forward_ref(x0)
    for k, v in T.items(): print('ref %-3s %s min %d max %d mean|v| %.2f sat %.4f%%' % (k, v.shape, v.min(), v.max(), np.abs(v).mean(), 100 * (np.abs(v + 0.5) >= 127.5).mean()))
    ((T['OUT'] + 128).astype(np.uint8)).tofile(od + '/ref_out.raw')
    np.savez_compressed(od + '/ref_T.npz', **{k: v.astype(np.int8) for k, v in T.items()})
    try:
        assert onnx is not None, 'no onnx'
        import onnxruntime as ort
        for n in ('n1_fold', 'n1_add'):
            s = ort.InferenceSession('%s/%s.onnx' % (od, n), providers=['CPUExecutionProvider'])
            o = s.run(None, {'x': x0.transpose(2, 0, 1)[None].astype(np.float32)})[0][0].transpose(1, 2, 0)
            o = np.rint(o).astype(np.int64); d = np.abs(o - T['OUT'])
            print('ORT %s vs numpy ref: bytes differ %d of %d (%.3f%%) max %d mean %.3e' % (n, int((d > 0).sum()), d.size, 100 * (d > 0).mean(), int(d.max()), d.mean()))
            ((np.clip(o, -128, 127) + 128).astype(np.uint8)).tofile('%s/ort_%s.raw' % (od, n))
    except (ImportError, AssertionError) as e:
        print('no onnxruntime:', e)
