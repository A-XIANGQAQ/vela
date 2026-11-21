#!/usr/bin/env python3
"""K1H: full FSR 4.1.1 exact HTP graph ht_t2s7p13 (HT, flags T2+P07S+P13S) + the postpass K1 (pass-27) appended on the HTP.
K1 = 'g' block on the network OUT (same shape as pass-25 = model_12): 3x3 16->16 >>8 | 1x1 16->32 relu >>7 | 1x1 32->16 + 64*OUT >>6,
weights/biases from K1H/k1ref.py (parsed from FO k1a.comp, checked == the literal shader on 4 inputs). Built with the SAME exact
machinery as every network layer (NG.L2 two FID views + grouped sum, CB=cat bias constants, packed P=2): blk2(g, L['p27'], OUT, 'p27').
Execs HT/gen_ht.py's prefix (read-only) like HT2. Flags: K1 (append K1). dbg: 0 = production outputs, 2 = out + k1 (sim check).
Production output with K1: ONLY 'k1' (uint8 1x540x480x32 = K1 int8 ^ 0x80, NHWC [y][x][16]); the network 'out' is no longer exported
(K1 was its only reader). Graph 'k1only' (variant flag K1ONLY): x = network OUT (uint8 1x540x480x32) -> k1 (fast sim stress check).
Usage (docker h-qnn): VARIANTS='name:flags:dbg ...' gen_k1h.py OUTDIR"""
import sys, os, numpy as np
OD = sys.argv[1]
ht = open('/ht/gen_ht.py').read()
pre = ht.split("sys.argv = ['gen_nf2', OD]; ns =")[0]
g0 = {'__name__': 'k1hpre'}; exec(compile(pre, 'gen_ht_pre', 'exec'), g0); src = g0['src']
a = "T['OUT'] = blk2(g, L['p25'], T['H0'], 'p25')"; assert src.count(a) == 1
src = src.replace(a, a + "\n    if 'K1' in FL: T['K1'] = blk2(g, L['p27'], T['OUT'], 'p27')")
a = "outs = NAMES if dbg else ['OUT']; onames = [('o_' + k) if k != 'OUT' else 'out' for k in outs]"; assert src.count(a) == 1
src = src.replace(a, "outs = (NAMES if dbg == 1 else (['OUT', 'K1'] if dbg == 2 else ['OUT']))\n"
                     "    if 'K1' in FL and dbg == 0: outs = ['K1']\n"
                     "    if 'K1' in FL and dbg == 1: outs = outs + ['K1']\n"
                     "    onames = [{'OUT': 'out', 'K1': 'k1'}.get(k, 'o_' + k) for k in outs]")
src += r'''
def build_k1only(name):   # x = network OUT (packed) -> K1 -> k1 ; same exact machinery as build2
    g = NG('exact', 1); x = A(g.qdq('x', 'xq'), 2, 16, False)
    t = blk2(g, L['p27'], x, 'p27'); src_ = t.name
    if t.k: src_ = g.qdq(g.slice(t.name, 0, t.P * t.C, 'k1_v'), 'k1_vq')
    g.nodes.append(hp.make_node('Identity', [src_], ['k1']))
    vi = lambda n, sh_: hp.make_tensor_value_info(n, TP.FLOAT, [1] + list(sh_))
    m_ = hp.make_model(hp.make_graph(g.nodes, name, [vi('x', (32, 540, 480))], [vi('k1', (32, 540, 480))], initializer=g.inits),
                       opset_imports=[hp.make_opsetid('', 13)], producer_name='fsr4r-K1H')
    m_.ir_version = 8; onnx.checker.check_model(m_)
    io = ''.join('- IOName: %s\n  Datatype: uint8\n  QuantParam:\n    Type: QNN_DEFINITION_DEFINED\n    Scale: 1.0\n    Offset: -128\n' % n for n in ['x', 'k1'])
    onnx.save(m_, '%s/%s.onnx' % (OD, name)); open('%s/%s.io.yaml' % (OD, name), 'w').write(io); return m_
'''
sys.argv = ['gen_nf2', OD]; ns = {'__name__': 'k1h', '__file__': '/nf/gen_nf2.py'}
exec(compile(src, 'gen_nf2_k1h', 'exec'), ns)
q = np.load('/k1h/build/k1_w.npz')
P27 = {k: (q[k].astype(np.int64) if q[k].ndim else (str(q[k]) if q[k].dtype.kind == 'U' else int(q[k]))) for k in q.files}
assert P27['kind'] == 'g' and P27['w1'].shape == (16, 16, 3, 3) and P27['w3'].shape == (16, 32, 1, 1)
# same form as N/n1 gblk (pass-25): compare the dict keys/shapes with L['p25']
p25 = ns['L']['p25']; assert set(p25) == set(P27), (sorted(p25), sorted(P27))
for k in p25:
    if isinstance(p25[k], np.ndarray): assert p25[k].shape == P27[k].shape, k
ns['L']['p27'] = P27; ns['ns']['HW']['p27'] = (540, 960); ns['SHAPES']['K1'] = (16, 540, 960)
# numpy reference of the appended block with the network's own ref_block (N/n1) must equal k1ref's literal-shader output
nr = ns
while 'ref_block' not in nr: nr = nr['ns']
for k in ('ref0', 'rnd'):
    z = np.load('/k1h/build/k1in_%s.npz' % k); r = nr['ref_block'](P27, z['OUT'].astype(np.int64))
    print('ref_block(p27) vs literal k1a on %s: bytes differ %d' % (k, int((r.astype(np.int8) != z['K1']).sum())))
for v in os.environ['VARIANTS'].split():
    name, fl, dbg = v.split(':'); FL = set(f for f in fl.split('+') if f != 'base')
    ns['FL'].clear(); ns['FL'].update(FL); ns['SLICE'] = 'SL0' not in FL
    m_ = ns['build_k1only'](name) if 'K1ONLY' in FL else ns['build2'](name, 'exact', 1, int(dbg)); mac, byt, ops = ns['stats'](m_)
    print('wrote %-12s flags %-16s MAC %.2f G act %.1f MB ops %s' % (name, fl, mac / 1e9, byt / 1e6, ops))
