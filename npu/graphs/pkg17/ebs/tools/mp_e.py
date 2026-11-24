#!/usr/bin/env python3
"""r6 micro-probe "E-BS": can the 11 relu 'e' view convs take their bias through the HTP int32 bias (no constant channels, no Concat,
34 -> 32 / 66 -> 64 input channels) when the relu lower clamp is supplied by the OUTPUT ZERO POINT z instead of a u8 zp-0 tensor?
  shipped:  D = conv([x, const 1, const 64]) -> u8 zp 0   (views A, B clamp at 0 = relu),  S = A + B  -> int8 (clamp 127)
  E-BS(z):  D = conv(x) + HTP bias (b_view - z * 2^t)      -> u8 zp z  (view' = view - z, lower clamp -z  <=> view >= 0 = relu),
            S = A' + B' + 2z -> int8                        (same integers; the upper clamp 255 - z + z >= 127 is harmless)
One graph: for every e layer its real D weights (read from the shipped ONNX) x a list of z, all convs on one random input
(1 x 66 x H x W: 64 random channels + the two constant planes 1 and 64 for the control conv 'c' = the shipped form).
  mp_e.py gen  SHIP_ONNX OUTDIR   -> mp.onnx, mp.io.yaml, x.raw (NHWC u8), exp.npz (expected u8 outputs, exact integer model)
  mp_e.py cmp  OUTDIR SIMDIR      -> misses per (layer, z)
Expected value: clamp(floor((acc + b - z 2^t + 2^(t-1)) / 2^t) + z, 0, 255)  (one round-half-up, t = s + 1)."""
import sys, os, numpy as np
E = ['p03e', 'p05e', 'p09e', 'p11e', 'p15e', 'p17e', 'p19e', 'p21e', 'p23e', 'p25e', 'p27e']
ZS = [int(z) for z in os.environ.get('ZS', '0 1 8 32 64 100 128 255').split()]
LAY = os.environ.get('LAYERS'); E = LAY.split() if LAY else E
H = W = int(os.environ.get('HW', 96))


def gen(ship, od):
    import onnx
    from onnx import TensorProto as TP, helper as hp, numpy_helper as nh
    os.makedirs(od, exist_ok=True); m = onnx.load(ship); ini = {i.name: nh.to_array(i) for i in m.graph.initializer}
    rng = np.random.default_rng(606); x = rng.integers(0, 256, (1, 66, H, W), dtype=np.uint8); x[0, 64] = 129; x[0, 65] = 192
    x.transpose(0, 2, 3, 1).tofile(od + '/x.raw'); xi = x[0].astype(np.int64) - 128
    nodes, inits, outs, exp, io = [], [], [], {}, []
    c = lambda n, a: (inits.append(nh.from_array(np.asarray(a), n)), n)[1]
    nodes.append(hp.make_node('QuantizeLinear', ['x', c('x_s', np.float32(1)), c('x_z', np.uint8(128))], ['xq_q']))
    nodes.append(hp.make_node('DequantizeLinear', ['xq_q', 'x_s', 'x_z'], ['xq']))
    for tag in E:
        wq = ini[tag + 'D_wq'].astype(np.int64); ws = ini[tag + 'D_ws'].astype(np.float64); assert not ini[tag + 'D_bq'].any(), tag
        R, Cin = wq.shape[:2]; t = np.rint(-np.log2(ws)).astype(np.int64); assert (2.0 ** -t == ws).all() and wq.shape[2:] == (1, 1)
        b = wq[:, Cin - 2, 0, 0] + 64 * wq[:, Cin - 1, 0, 0]; Wr = wq[:, :Cin - 2, 0, 0]; assert Cin - 2 in (32, 64), (tag, Cin)
        acc = np.tensordot(Wr, xi[:Cin - 2], 1)                                # (R, H, W)
        print('%s rows %d real in %d t %s bias range %d..%d' % (tag, R, Cin - 2, sorted(set(t.tolist())), b.min(), b.max()))
        for z in ['c'] + ZS:
            n = '%s_z%s' % (tag, z); W66 = np.zeros((R, 66, 1, 1), np.int64); W66[:, :Cin - 2, 0, 0] = Wr
            if z == 'c': W66[:, 64, 0, 0] = wq[:, Cin - 2, 0, 0]; W66[:, 65, 0, 0] = wq[:, Cin - 1, 0, 0]; bq = np.zeros(R, np.int64); zz = 0
            else: zz = z; bq = b - zz * (1 << t)
            c(n + '_wq', W66.astype(np.int8)); c(n + '_bq', bq.astype(np.int32)); c(n + '_ws', ws.astype(np.float32))
            nodes.append(hp.make_node('DequantizeLinear', [n + '_wq', n + '_ws', c(n + '_wz', np.zeros(R, np.int8))], [n + '_w'], axis=0))
            nodes.append(hp.make_node('DequantizeLinear', [n + '_bq', n + '_ws', c(n + '_bz', np.zeros(R, np.int32))], [n + '_b'], axis=0))
            nodes.append(hp.make_node('Conv', ['xq', n + '_w', n + '_b'], [n + '_y'], kernel_shape=[1, 1], strides=[1, 1], pads=[0] * 4))
            nodes.append(hp.make_node('QuantizeLinear', [n + '_y', c(n + '_s', np.float32(1)), c(n + '_z', np.uint8(zz))], [n + '_q']))
            nodes.append(hp.make_node('DequantizeLinear', [n + '_q', n + '_s', n + '_z'], [n + '_o']))
            nodes.append(hp.make_node('Identity', [n + '_o'], ['o_' + n])); outs.append(hp.make_tensor_value_info('o_' + n, TP.FLOAT, [1, R, H, W]))
            io.append(('o_' + n, zz))
            tt = t[:, None, None]; e = np.clip(((acc + b[:, None, None] - zz * (1 << tt) + (1 << (tt - 1))) >> tt) + zz, 0, 255)
            exp['o_' + n] = e.transpose(1, 2, 0).astype(np.uint8)              # HWC
    g = hp.make_graph(nodes, 'mp', [hp.make_tensor_value_info('x', TP.FLOAT, [1, 66, H, W])], outs, initializer=inits)
    mm = hp.make_model(g, opset_imports=[hp.make_opsetid('', 13)], producer_name='r6-mp'); mm.ir_version = 8; onnx.checker.check_model(mm)
    onnx.save(mm, od + '/mp.onnx'); np.savez_compressed(od + '/exp.npz', **exp)
    y = '- IOName: x\n  Datatype: uint8\n  QuantParam:\n    Type: QNN_DEFINITION_DEFINED\n    Scale: 1.0\n    Offset: -128\n'
    y += ''.join('- IOName: %s\n  Datatype: uint8\n  QuantParam:\n    Type: QNN_DEFINITION_DEFINED\n    Scale: 1.0\n    Offset: %d\n' % (n, -z) for n, z in io)
    open(od + '/mp.io.yaml', 'w').write(y); print('wrote', len(io), 'convs')


def cmp(od, sd):
    exp = np.load(od + '/exp.npz'); files = {f[:-len('_native.raw')] if f.endswith('_native.raw') else f[:-4]: os.path.join(r, f) for r, _, fs in os.walk(sd) for f in fs if f.endswith('.raw')}
    rows = {}
    for n in exp.files:
        if n not in files: print('MISSING', n); continue
        e = exp[n]; s = np.fromfile(files[n], np.uint8).reshape(e.shape); d = s.astype(np.int64) - e
        tag, z = n[2:].split('_z'); bad = d != 0; chans = np.nonzero(bad.reshape(-1, e.shape[-1]).any(0))[0]
        rows.setdefault(tag, {})[z] = (int(bad.sum()), len(chans), int(np.abs(d).max()), chans[:6].tolist())
    for tag, r in rows.items():
        print('MP %-5s ' % tag + '  '.join('z=%s: %d miss / %d ch / max %d %s' % (z, v[0], v[1], v[2], v[3] if v[0] else '') for z, v in r.items()))
    ok = {tag: [z for z, v in r.items() if z != 'c' and v[0] == 0] for tag, r in rows.items()}
    print('MPOK ' + '  '.join('%s:%s' % (t, ','.join(v) if v else 'NONE') for t, v in ok.items()))


if sys.argv[1] == 'gen': gen(sys.argv[2], sys.argv[3])
else: cmp(sys.argv[2], sys.argv[3])
