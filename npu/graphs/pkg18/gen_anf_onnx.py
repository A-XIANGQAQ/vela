"""ANF SR network as an int8 QDQ ONNX graph for qnn-onnx-converter (runs inside the h-qnn image: python3 + onnx 1.16).

All activations are uint8, scale 1, zero point 0 (float value == u8 byte == ANF int8 + 128). Then:
  - ONNX/QNN zero padding = u8 0 = ANF's out-of-bounds tap -128 (all layers);
  - ANF f = M*(acc_i8 + b) with acc_i8 = acc_u8 - 128*sum(w)  ->  conv bias (scale M) = b - 128*sum(w) + 128/M,
    so the conv output is f + 128 and QuantizeLinear(scale 1, zp 0) clamps it to [0, 255] = ANF's clamp [-128, 127].
  - weights int8 symmetric per output channel, scale M[oc] (all M are powers of two).
usage: python3 gen_anf_onnx.py OUTDIR NAME LW LH PHASE CIN(20|32) D2S(0|1) MODEL_DIR
writes NAME.onnx + NAME.io.yaml; input x [1, LH, LW, CIN] (NHWC custom IO), output y [1, LH/2, LW/2, 64] or [1, 2LH, 2LW, 4].
"""
import json, os, sys
import numpy as np
import onnx
from onnx import helper as hp, numpy_helper as nh, TensorProto as TP

OD, NAME, LW, LH, PH, CIN, D2S, MODEL = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
LAYERS = [('cnn1', 20, 32, 2, True), ('cnn2-0', 32, 32, 1, False), ('cnn2-1', 32, 32, 1, False), ('cnn3', 32, 64, 1, True)]

inits, nodes = [], []


def c(name, arr):
    inits.append(nh.from_array(np.asarray(arr), name))
    return name


def qdq(x, tag):
    s, z = c(tag + '_s', np.float32(1.0)), c(tag + '_z', np.uint8(0))
    nodes.append(hp.make_node('QuantizeLinear', [x, s, z], [tag + '_q']))
    nodes.append(hp.make_node('DequantizeLinear', [tag + '_q', s, z], [tag]))
    return tag


def load(name, ic, oc, ph):
    d = os.path.join(MODEL, name, str(ph))
    w = np.fromfile(os.path.join(d, '%s.weight_shape-%d-%d-3-3_int8.raw' % (name, oc, ic)), np.int8).reshape(oc, ic, 3, 3)
    b = np.fromfile(os.path.join(d, '%s.bias_shape-%d_int32.raw' % (name, oc)), '<i4').astype(np.int64)
    m = np.array(json.load(open(os.path.join(d, '%s.scales.json' % name)))['weights_scales'], np.float64)
    return w, b, m


x = 'x'
cur = qdq('x', 'xq')
H, W = LH, LW
NL = int(os.environ.get('ANF_NL', '4'))
for li, (n, ic, oc, st, jit) in enumerate(LAYERS[:NL]):
    w, b, m = load(n, ic, oc, PH if jit else 0)
    k = -np.log2(m)
    assert np.all(k == np.round(k)), 'non-pow2 scale'
    b2 = b - 128 * w.astype(np.int64).reshape(oc, -1).sum(1) + (128 * 2 ** k).astype(np.int64)
    assert np.abs(b2).max() < 2 ** 31
    wf = w
    if li == 0 and CIN > ic:
        wf = np.zeros((oc, CIN, 3, 3), np.int8)
        wf[:, :ic] = w
    tag = 'l%d' % li
    c(tag + '_wq', wf.astype(np.int8))
    c(tag + '_bq', b2.astype(np.int32))
    ws = m.astype(np.float32)
    nodes.append(hp.make_node('DequantizeLinear', [tag + '_wq', c(tag + '_ws', ws), c(tag + '_wz', np.zeros(oc, np.int8))], [tag + '_w'], axis=0))
    nodes.append(hp.make_node('DequantizeLinear', [tag + '_bq', c(tag + '_bs', ws), c(tag + '_bz', np.zeros(oc, np.int32))], [tag + '_b'], axis=0))
    pads = [1, 1, 1, 1] if st == 1 else [1, 1, 0 if H % 2 == 0 else 1, 0 if W % 2 == 0 else 1]
    nodes.append(hp.make_node('Conv', [cur, tag + '_w', tag + '_b'], [tag + '_y'], kernel_shape=[3, 3], strides=[st, st], pads=pads))
    cur = qdq(tag + '_y', tag + '_o')
    H, W = (H - 1) // st + 1, (W - 1) // st + 1
if D2S:
    nodes.append(hp.make_node('DepthToSpace', [cur], ['d2s'], blocksize=4, mode='DCR'))
    cur = qdq('d2s', 'd2sq')
    oshape = [1, 4, 4 * H, 4 * W]
else:
    oshape = [1, LAYERS[NL - 1][2], H, W]
nodes.append(hp.make_node('Identity', [cur], ['y']))
g = hp.make_graph(nodes, NAME, [hp.make_tensor_value_info('x', TP.FLOAT, [1, CIN, LH, LW])],
                  [hp.make_tensor_value_info('y', TP.FLOAT, oshape)], initializer=inits)
mdl = hp.make_model(g, opset_imports=[hp.make_opsetid('', 13)], producer_name='fsr4r-ANF')
mdl.ir_version = 8
onnx.checker.check_model(mdl)
os.makedirs(OD, exist_ok=True)
onnx.save(mdl, os.path.join(OD, NAME + '.onnx'))
io = ''.join('- IOName: %s\n  Datatype: uint8\n  QuantParam:\n    Type: QNN_DEFINITION_DEFINED\n    Scale: 1.0\n    Offset: 0\n' % n for n in ['x', 'y'])
open(os.path.join(OD, NAME + '.io.yaml'), 'w').write(io)
print('ONNX', NAME, 'in', [1, LH, LW, CIN], 'out', oshape)
