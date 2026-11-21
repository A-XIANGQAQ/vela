#!/usr/bin/env python3
"""N7 (2026-10-31): the exact nf4_k1h graph (K1H: HT ht_t2s7p13 + postpass K1 appended, two FID views + grouped sum, CB=cat bias constants,
packed P=2) at the 640x360 network grid of a 1280x720 output. Size substitution as F36 did for NF (whole-number tokens only:
540->360, 960->640, 270->180, 480->320, 135->90, 240->160) over read-only copies of NF/gen_nf.py, NF/gen_nf2.py, HT/gen_ht.py and
K1H/gen_k1h.py, written to /n7/build/src and chained by path. One contract change: the input x is uint8 1x360x480x32 = rows of 960 px
(the fused pre's literal 960-px HTP-input stride), columns 640..959 unused; the graph slices columns 0..639 (ONNX Slice axis 3 on the
dequantized input, then the same xq QDQ = exact). Output k1 uint8 1x360x320x32 (dense 640, K2's stride). dbg 2 adds out (network OUT).
Usage (docker h-qnn): VARIANTS='name:flags:dbg ...' python3 /n7/gen_n7.py /n7/build"""
import os, re, sys
SUB = [('540', '360'), ('960', '640'), ('270', '180'), ('480', '320'), ('135', '90'), ('240', '160')]
def sz(s):
    for a, b in SUB: s = re.sub(r'(?<![0-9])%s(?![0-9])' % a, b, s)
    return s
OD = sys.argv[1]; S = OD + '/src'; os.makedirs(S, exist_ok=True)
def put(name, t): open('%s/%s' % (S, name), 'w').write(t); return '%s/%s' % (S, name)
def one(t, a, b):
    assert t.count(a) == 1, (a, t.count(a)); return t.replace(a, b)
g1 = sz(open('/nf/gen_nf.py').read())
old = "xr = np.fromfile('/h/r6_build/x_real.raw', np.uint8)"
if old in g1: g1 = g1.replace(old, old + ".reshape(540, 960, 16)[:360, :640].ravel()")   # as F36 (only used if this part runs)
p1 = put('gen_nf.py', g1)
p2 = put('gen_nf2.py', one(sz(open('/nf/gen_nf2.py').read()), "open('/nf/gen_nf.py')", "open('%s')" % p1))
p3 = put('gen_ht.py', one(sz(open('/ht/gen_ht.py').read()), "open('/nf/gen_nf2.py')", "open('%s')" % p2))
k = one(sz(open('/k1h/gen_k1h.py').read()), "open('/ht/gen_ht.py')", "open('%s')" % p3)
XS = r'''
a = "x = g.kconst(A(g.qdq('x', 'xq'), P, 16, False), (Hx, Wx)); T = {}"; print('N7 input anchor count', src.count(a)); assert src.count(a) >= 1
src = src.replace(a, "x = g.kconst(A(g.qdq(g.node('Slice', ['x', g.c('xs_a', np.array([0], np.int64)), g.c('xs_b', np.array([Wx], np.int64)), "
                     "g.c('xs_x', np.array([3], np.int64))], 'x_sl'), 'xq'), P, 16, False), (Hx, Wx)); T = {}")
a = "vi('x', (32, 360, 320) if pack else (16, 360, 640))"; print('N7 vi anchor count', src.count(a)); assert src.count(a) >= 1
src = src.replace(a, "vi('x', (32, 360, 480) if pack else (16, 360, 960))")
'''
k = one(k, "src += r'''", XS + "src += r'''")
p4 = put('gen_k1h.py', k)
print('N7 sources', p1, p2, p3, p4)
sys.argv = [p4, OD]; exec(compile(k, p4, 'exec'), {'__name__': '__main__', '__file__': p4})
