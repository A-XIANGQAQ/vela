#!/usr/bin/env python3
"""NF: FULL FSR 4.1.1 int8 network (model_1..12 = pass-03..25, general/1080/performance) as ONE int8 QNN/HTP v73 graph,
960x540x16 in (written by our prepass) -> 960x540x16 out (= model_12 output, read by our postpass). Same I/O contract style as
Lite g2_xp2s: uint8 NHWC, scale 1, offset -128 (= the int8 shader byte ^ 0x80).
Layers / weights / shader integer math: N n1 (gen_n1.py prefix exec'd read-only: params of the fsr4vk_vela3 generators'
parse(), biases from initializers.bin, numpy reference forward_ref = rhe(>>sh), clamp int8), with N's SYNTHETIC pass-13
replaced by the REAL pass-13 weights (parse_p13.py -> p13_w.npz: 2x2 s2 32->64, sh 8 not 7).
Graph rules (H gen_r6 / G2 / FID): activations scale 1, weights per-channel 2^-sh, every Concat with its own QDQ, ReLU = uint8 zp0.
Residual R*y folded into the 1x1 conv as identity input channels, R split into int8 pieces (256 = 127+127+2, 128 = 127+1,
64 = 64) -> exact int32 accumulator. Tails (pass-19 convT 64->32 +256*E2 >>9, pass-23 convT 32->16 +128*X2 >>7): skip
SpaceToDepth(2) -> low-res 1x1 conv on [y, s2d(skip) pieces] (out channel (2dy+dx)*C+o, the shader's own order) -> DepthToSpace
(DCR): exact accumulator, no Add after rounding (N's tail Add double-rounds).
Modes: plain = one conv per layer, HTP rounding (half up) and HTP bias; exact = FID two views per layer:
rhe(a,s) = rhu(a + h - 1, s+1) + rhu(a - h, s+1) (h = 2^(s-1)), each view a conv at weight scale 2^-(s+1), bias through
constant input channels (values 1, 64[, 127..]) on the centre tap, HTP bias 0 (FID fid_cb: HTP bias rounds some ties down),
then Add(A, B) at scale 1 (exact, int8 clamp = shader clamp). Never dlbc.
Graphs: nf_plain, nf_exact (production: x -> out), nf_plain_d, nf_exact_d (+ every pass output o_X1..o_OUT for per-layer
fidelity). Also: numpy references for the sim inputs, ORT check of nf_plain_d (ORT QuantizeLinear = rhe -> must equal the
reference byte for byte), MAC / traffic table incl. G2 g2_xp2s.
Usage (docker h-qnn): gen_nf.py OUTDIR"""
import sys, os, numpy as np
OD = sys.argv[1]; os.makedirs(OD, exist_ok=True)
src = open('/n/n1/gen_n1.py').read().split('\n# ---------------- layer table')[0].replace("HERE + '/params.pkl'", "'/nf/params_n1.pkl'")
# UP2: Ultra Performance p23 constants (= UP/params/ref_up.py)
_old = 'sf=np.full(32, 9), R=np.full(32, 256), lo=16, w4=w4, b4=b4, t_sf=7, t_R=128)'
assert src.count(_old) == 1
src = src.replace(_old, "sf=np.full(32, 8 if q.get('variant') == 'ultraperf' else 9), R=np.full(32, 256), lo=16, w4=w4, b4=b4, t_sf=8 if q.get('variant') == 'ultraperf' else 7, t_R=256 if q.get('variant') == 'ultraperf' else 128)")
sys.argv = ['gen_n1', OD + '/n1tmp']; ns = {'__name__': 'n1', '__file__': '/n/n1/gen_n1.py'}
exec(compile(src, 'gen_n1_prefix', 'exec'), ns)
L, forward_ref = ns['L'], ns['forward_ref']
q = np.load('/nf/p13_w.npz'); L['p13'] = dict(kind='d', w=q['w'].astype(np.int64), b=q['b'].astype(np.int64), sh=int(q['sh']))
L['p07']['sh'] = np.load('/nf/p07_sh.npy')   # UP3: pass-07 store shift per channel (fix07.py)
import onnx
from onnx import TensorProto as TP, helper as hp, numpy_helper as nh, shape_inference

HW = {'p03': (540, 960), 'p05': (540, 960), 'p07': (540, 960), 'p09': (270, 480), 'p11': (270, 480), 'p13': (270, 480),
      'p15': (135, 240), 'p17': (135, 240), 'p19': (135, 240), 'p21': (270, 480), 'p23': (270, 480), 'p25': (540, 960)}   # INPUT res
def pieces(R):
    p = []
    while R > 0: p.append(min(R, 127)); R -= p[-1]
    return p
bmax = max(int(np.abs(v[k]).max()) for v in L.values() for k in v if k.startswith('b')) + 512
N127 = 0 if bmax <= 127 * 64 + 63 else -(-(bmax - 127 * 64 - 63) // (127 * 127))
VALS = [1, 64] + [127] * N127
print('max |bias|+h %d -> constant bias channels %s' % (bmax, VALS))


class G:
    def __init__(s, mode): s.nodes, s.inits, s.mode, s.consts = [], [], mode, {}
    def c(s, name, arr): s.inits.append(nh.from_array(np.asarray(arr), name)); return name
    def qdq(s, x, tag, u8=False):
        sc, z = s.c(tag + '_s', np.float32(1.0)), s.c(tag + '_z', np.uint8(0) if u8 else np.uint8(128))
        s.nodes.append(hp.make_node('QuantizeLinear', [x, sc, z], [tag + '_q'])); s.nodes.append(hp.make_node('DequantizeLinear', [tag + '_q', sc, z], [tag]))
        return tag
    def node(s, op, ins, tag, **kw): s.nodes.append(hp.make_node(op, ins, [tag], **kw)); return tag
    def slice(s, x, a, b, tag):
        return s.qdq(s.node('Slice', [x, s.c(tag + '_a', np.array([a], np.int64)), s.c(tag + '_b', np.array([b], np.int64)), s.c(tag + '_x', np.array([1], np.int64))], tag + '_sl'), tag)
    def cat(s, xs, tag): return xs[0] if len(xs) == 1 else s.qdq(s.node('Concat', xs, tag + '_c', axis=1), tag)
    def const(s, hw):
        n = 'const_%dx%d' % hw
        if n not in s.consts:
            cv = np.full((1, len(VALS)) + hw, 128, np.uint8)
            for i, v in enumerate(VALS): cv[0, i] += v
            s.c(n + '_q', cv); s.nodes.append(hp.make_node('DequantizeLinear', [n + '_q', s.c(n + '_s', np.float32(1.0)), s.c(n + '_z', np.uint8(128))], [n]))
            s.consts[n] = 1
        return n
    def conv(s, x, w, b, k, stride, pad, sh, tag, relu=False):
        o = w.shape[0]; sh = np.full(o, sh) if np.isscalar(sh) else np.asarray(sh); ws = (2.0 ** -sh.astype(np.float64)).astype(np.float32)
        assert w.min() >= -128 and w.max() <= 127, tag
        s.c(tag + '_wq', w.astype(np.int8)); s.c(tag + '_bq', np.asarray(b).astype(np.int32))
        s.nodes.append(hp.make_node('DequantizeLinear', [tag + '_wq', s.c(tag + '_ws', ws), s.c(tag + '_wz', np.zeros(o, np.int8))], [tag + '_w'], axis=0))
        s.nodes.append(hp.make_node('DequantizeLinear', [tag + '_bq', s.c(tag + '_bs', ws), s.c(tag + '_bz', np.zeros(o, np.int32))], [tag + '_b'], axis=0))
        s.nodes.append(hp.make_node('Conv', [x, tag + '_w', tag + '_b'], [tag + '_y'], kernel_shape=list(k), strides=[stride] * 2, pads=[pad] * 4))
        return s.qdq(tag + '_y', tag + '_o', u8=relu)
    def EX(s, ins, W, b, k, stride, pad, sh, tag, hw, relu=False):
        o = W.shape[0]; sh = np.full(o, sh) if np.isscalar(sh) else np.asarray(sh); b = np.asarray(b, np.int64)
        if s.mode == 'plain': return s.conv(s.cat(ins, tag + '_in'), W, b, k, stride, pad, sh, tag, relu)
        assert (sh > 0).all(); hh = 1 << (sh - 1)
        x = s.cat(ins + [s.const(hw)], tag + '_in'); I = W.shape[1]; cy, cx = k[0] // 2, k[1] // 2
        def Wb(bv):
            W2 = np.zeros((o, I + len(VALS)) + tuple(k), np.int64); W2[:, :I] = W; r = bv.copy(); co = np.zeros((o, len(VALS)), np.int64)
            for j in range(2, len(VALS)): co[:, j] = np.clip(np.fix(r / 127.0), -127, 127).astype(np.int64); r = r - 127 * co[:, j]
            co[:, 1] = np.round(r / 64.0).astype(np.int64); co[:, 0] = r - 64 * co[:, 1]
            assert np.abs(co).max() <= 127 and ((co * np.array(VALS)).sum(1) == bv).all(), tag
            W2[:, I:, cy, cx] = co; return W2
        A = s.conv(x, Wb(b + hh - 1), np.zeros(o), k, stride, pad, sh + 1, tag + 'A', relu)
        B = s.conv(x, Wb(b - hh), np.zeros(o), k, stride, pad, sh + 1, tag + 'B', relu)
        return s.qdq(s.node('Add', [A, B], tag + '_add'), tag + '_o')


def fold(w3, R):
    C, I3 = w3.shape[:2]; P = max(len(pieces(int(r))) for r in R); Wf = np.zeros((C, I3 + P * C, 1, 1), np.int64); Wf[:, :I3] = w3
    for o in range(C):
        for j, pv in enumerate(pieces(int(R[o]))): Wf[o, I3 + j * C + o, 0, 0] = pv
    return Wf, P


def blk(g, v, y, tag):
    hw = HW[tag]
    if v['kind'] == 'd': return g.EX([y], v['w'], v['b'], (2, 2), 2, 0, v['sh'], tag, hw)
    lo = v['lo']
    if v['kind'] == 'g': xin, W1 = [y], v['w1']
    else:
        xin = [g.slice(y, 0, lo, tag + '_lo')]; W1 = v['w1']
        if v.get('group', 1) == 2:   # grouped 3x3 2x(16->16) -> dense block-diagonal 32->32 (same integers)
            W1 = np.zeros((32, 32, 3, 3), np.int64); W1[:16, :16] = v['w1'][:16]; W1[16:, 16:] = v['w1'][16:]
    hc = g.EX(xin, W1, v['b1'], (3, 3), 1, 1, v['s1'], tag + 'c3', hw)
    ein = [hc] if v['kind'] == 'g' else [hc, g.slice(y, lo, 2 * lo, tag + '_hi')]
    m = g.EX(ein, v['w2'], v['b2'], (1, 1), 1, 0, v['s2'], tag + 'e', hw, relu=True)
    Wf, P = fold(v['w3'], v['R'])
    return g.EX([m] + [y] * P, Wf, v['b3'], (1, 1), 1, 0, v['sf'], tag + 'f', hw)


def tail(g, v, y, skip, tag):
    C4 = v['w4'].shape[0]; s2d = g.qdq(g.node('SpaceToDepth', [skip], tag + '_s2d', blocksize=2), tag + '_s2dq')
    Wt, P = fold(v['w4'], np.full(C4, v['t_R']))
    u = g.EX([y] + [s2d] * P, Wt, v['b4'], (1, 1), 1, 0, v['t_sf'], tag, HW[tag[:3]])
    return g.qdq(g.node('DepthToSpace', [u], tag + '_d2s', blocksize=2, mode='DCR'), tag + '_d2sq')


NAMES = ['X1', 'X2', 'D0', 'E1', 'E2', 'F0', 'F1', 'F2', 'G0', 'G1', 'H0', 'OUT']
SHAPES = dict(X1=(16, 540, 960), X2=(16, 540, 960), D0=(32, 270, 480), E1=(32, 270, 480), E2=(32, 270, 480), F0=(64, 135, 240),
              F1=(64, 135, 240), F2=(64, 135, 240), G0=(32, 270, 480), G1=(32, 270, 480), H0=(16, 540, 960), OUT=(16, 540, 960))


def build(name, mode, dbg):
    g = G(mode); x = g.qdq('x', 'xq'); T = {}
    T['X1'] = blk(g, L['p03'], x, 'p03'); T['X2'] = blk(g, L['p05'], T['X1'], 'p05'); T['D0'] = blk(g, L['p07'], T['X2'], 'p07')
    T['E1'] = blk(g, L['p09'], T['D0'], 'p09'); T['E2'] = blk(g, L['p11'], T['E1'], 'p11'); T['F0'] = blk(g, L['p13'], T['E2'], 'p13')
    T['F1'] = blk(g, L['p15'], T['F0'], 'p15'); T['F2'] = blk(g, L['p17'], T['F1'], 'p17')
    T['G0'] = tail(g, L['p19'], blk(g, L['p19'], T['F2'], 'p19'), T['E2'], 'p19t'); T['G1'] = blk(g, L['p21'], T['G0'], 'p21')
    T['H0'] = tail(g, L['p23'], blk(g, L['p23'], T['G1'], 'p23'), T['X2'], 'p23t'); T['OUT'] = blk(g, L['p25'], T['H0'], 'p25')
    vi = lambda n, sh_: hp.make_tensor_value_info(n, TP.FLOAT, [1] + list(sh_))
    outs = NAMES if dbg else ['OUT']; onames = [('o_' + k) if k != 'OUT' else 'out' for k in outs]
    for k, on in zip(outs, onames): g.nodes.append(hp.make_node('Identity', [T[k]], [on]))
    m_ = hp.make_model(hp.make_graph(g.nodes, name, [vi('x', (16, 540, 960))], [vi(on, SHAPES[k]) for k, on in zip(outs, onames)], initializer=g.inits),
                       opset_imports=[hp.make_opsetid('', 13)], producer_name='fsr4r-NF')
    m_.ir_version = 8; onnx.checker.check_model(m_)
    io = ''.join('- IOName: %s\n  Datatype: uint8\n  QuantParam:\n    Type: QNN_DEFINITION_DEFINED\n    Scale: 1.0\n    Offset: -128\n' % n for n in ['x'] + onames)
    onnx.save(m_, '%s/%s.onnx' % (OD, name)); open('%s/%s.io.yaml' % (OD, name), 'w').write(io)
    return m_, onames


def stats(m_):   # MAC (dense, as the HTP computes it) and int8 activation bytes written by compute/copy ops
    mi = shape_inference.infer_shapes(m_); sh = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(mi.graph.value_info) + list(mi.graph.output) + list(mi.graph.input)}
    ini = {i.name: nh.to_array(i).shape for i in m_.graph.initializer}; mac = 0; byt = 0; ops = {}
    prod = {}
    for n in m_.graph.node: prod[n.output[0]] = n
    for n in m_.graph.node:
        if n.op_type in ('QuantizeLinear', 'DequantizeLinear', 'Identity'): continue
        ops[n.op_type] = ops.get(n.op_type, 0) + 1; o = sh.get(n.output[0])
        if not o: continue
        byt += int(np.prod(o))
        if n.op_type == 'Conv':
            wn = prod[n.input[1]].input[0]; w = ini[wn]; mac += int(np.prod(o)) * int(np.prod(w[1:]))
    return mac, byt, ops


rows = []
for name, mode, dbg in (('nf_plain', 'plain', 0), ('nf_exact', 'exact', 0), ('nf_plain_d', 'plain', 1), ('nf_exact_d', 'exact', 1)):
    m_, on = build(name, mode, dbg)
    if not dbg: mac, byt, ops = stats(m_); rows.append('%-9s MAC %.2f G  act bytes %.1f MB  ops %s' % (name, mac / 1e9, byt / 1e6, ops))
    print('wrote', name)
for f in ('/g2/build/g2_xp2s.onnx',):
    mac, byt, ops = stats(onnx.load(f)); rows.append('%-9s MAC %.2f G  act bytes %.1f MB  ops %s' % ('g2_xp2s', mac / 1e9, byt / 1e6, ops))
open(OD + '/cost_table.txt', 'w').write('\n'.join(rows) + '\n'); print('\n'.join(rows))
# sim inputs + numpy references
xr = np.fromfile('/h/r6_build/x_real.raw', np.uint8)
rng = np.random.default_rng(7); xs = (xr.astype(np.int64) - 128).reshape(540, 960, 16)
xn = np.clip(xs + rng.integers(-24, 25, xs.shape), -128, 127)   # proxy input 2: x_real + uniform noise (other tie patterns)
(xs + 128).astype(np.uint8).tofile(OD + '/x0.raw'); (xn + 128).astype(np.uint8).tofile(OD + '/x1.raw')
for i, x0 in enumerate((xs, xn)):
    T = forward_ref(x0); np.savez_compressed('%s/ref%d.npz' % (OD, i), **{k: v.astype(np.int8) for k, v in T.items()})
    print('ref%d' % i, ' '.join('%s sat %.3f%%' % (k, 100 * (np.abs(v + 0.5) >= 127.5).mean()) for k, v in T.items()))
if not os.environ.get('NO_ORT'):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(OD + '/nf_plain_d.onnx', so, providers=['CPUExecutionProvider']); T = dict(np.load(OD + '/ref0.npz'))
    res = s.run(None, {'x': xs.transpose(2, 0, 1)[None].astype(np.float32)}); lines = []
    for k, v in zip(NAMES, res):
        d = np.abs(np.rint(v[0]).transpose(1, 2, 0).astype(np.int64) - T[k]); lines.append('ORT nf_plain_d %-3s vs numpy shader ref: bytes differ %d max %d' % (k, int((d > 0).sum()), int(d.max())))
    open(OD + '/ort_check.txt', 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
