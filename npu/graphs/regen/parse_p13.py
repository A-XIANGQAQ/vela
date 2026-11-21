#!/usr/bin/env python3
"""Weights of FSR 4.1.1 pass-13 (model_6, 2x2 stride-2 conv 32->64) from its spirv-cross GLSL (`spirv-cross --vulkan-semantics
pass-13.spv`, Performance preset) + initializers.bin. Writes w, b, sh (p13_w.npz). Method: evaluate every scalar
`uint _N = expr;` at gid = (0,0,0) to get byte addresses; per dot block: activation uvec4 at scratch byte 7712*(2y+1+dy) +
32x+16+16dx + plane base (8342464 = ch 0..15, 10440128 = ch 16..31: 2 planes 272x482x16), weight uvec4s at initializer byte
24576 + 32*o + 2048*t + 16*p; spvSDot(act.c, w.c) pairs channel 4c..4c+3 of plane p. Requant: roundEven(acc * 2^-8) (sh 8),
clamp int8. Bias: words 8192 + o (_20 uvec4 2048 / _14 byte 32768+16k). Every mapping asserted.
Usage: parse_p13.py P13_COMP INITIALIZERS_BIN OUT_NPZ"""
import re, sys, struct, numpy as np
src, ib, outp = sys.argv[1:4]
L = open(src).read().split('\n')
WORDS = struct.unpack('<%dI' % (len(open(ib, 'rb').read()) // 4), open(ib, 'rb').read())
sb = lambda w: [((w >> (8 * i)) & 0xff) - (256 if (w >> (8 * i)) & 0x80 else 0) for i in range(4)]
s32 = lambda v: v - (1 << 32) if v & 0x80000000 else v
V = {}
def ev(e):
    e = re.sub(r'gl_GlobalInvocationID\.[xyz]', '0', e); e = re.sub(r'(\d+)u\b', r'\1', e)
    e = re.sub(r'\b(_\d+)\b', lambda m: str(V[m.group(1)]), e); return eval(e)
acts, wts, comp = {}, {}, {}       # vec name -> (plane, dy, dx) / (o, t, p) ; scalar name -> (vec, comp)
blocks = []; cur = None; full = '\n'.join(L)
assert '* 0.00390625' in full and full.count('roundEven') == 64
for l in L:
    m = re.match(r'^\s*(?:uint )?(_\d+) = (_\d+)\.([xyzw]);$', l)
    if m: comp[m.group(1)] = (m.group(2), 'xyzw'.index(m.group(3))); continue
    m = re.match(r'^\s*uint (_\d+) = ([^;\[]+);$', l)
    if m:
        try: V[m.group(1)] = ev(m.group(2))
        except (KeyError, NameError, SyntaxError): pass
        continue
    m = re.match(r'^\s*uvec4 (_\d+) = uvec4\((_25|_14)\[[^\]]+\]\._m0\[(_\d+)\]', l)
    if m:
        a = 4 * V[m.group(3)]
        if m.group(2) == '_25':
            base = 8342464 if a < 10440128 else 10440128; rel = a - base
            dy, r2 = divmod(rel - 7712, 7712); dx = (r2 - 16) // 16
            assert dy in (0, 1) and dx in (0, 1) and r2 in (16, 32), (l, rel)
            acts[m.group(1)] = (0 if base == 8342464 else 1, dy, dx); cur = dict(act=m.group(1), w=[]); blocks.append(cur)
        elif a >= 32768: continue   # bias words 8196.. (accumulator init)
        else:
            rel = a - 24576; o, r2 = divmod(rel % 2048, 32); t = rel // 2048; p = r2 // 16
            assert r2 in (0, 16) and 0 <= t < 4, (l, rel)
            wts[m.group(1)] = (o, t, p); cur['w'].append(m.group(1))
        continue
    m = re.match(r'^\s*uint (_\d+) = (_\d+)\.([xyzw]);$', l)
    if m: comp[m.group(1)] = (m.group(2), 'xyzw'.index(m.group(3))); continue
    m = re.match(r'^\s*(_\d+) = (_\d+)\.([xyzw]);$', l)
    if m: comp[m.group(1)] = (m.group(2), 'xyzw'.index(m.group(3))); continue
    if 'spvSDot_uint_uint_uint(' in l and '=' in l:
        terms = re.findall(r'spvSDot_uint_uint_uint\((_\d+), (_\d+), 0\)', l)
        cur['terms'] = [(comp[a], comp[w]) for a, w in terms]
W = np.zeros((64, 32, 2, 2), np.int64); seen = np.zeros_like(W); tmap = {}
for bk in blocks:
    if 'terms' not in bk: continue
    plane, dy, dx = acts[bk['act']]
    for (av, ac), (wv, wc) in bk['terms']:
        assert av == bk['act'] and ac == wc and wv == bk['w'][0], bk
    for wv in bk['w']:
        o, t, p = wts[wv]; assert p == plane; tmap.setdefault(t, (dy, dx)); assert tmap[t] == (dy, dx)
        for k in range(4):
            for i, b in enumerate(sb(WORDS[(24576 + 32 * o + 2048 * t + 16 * p) // 4 + k])):
                W[o, 16 * p + 4 * k + i, dy, dx] = b; seen[o, 16 * p + 4 * k + i, dy, dx] += 1
assert (seen == 1).all(), (seen.min(), seen.max())
b = np.array([s32(WORDS[8192 + o]) for o in range(64)], np.int64)
acc_max = int((np.abs(W).sum((1, 2, 3)) * 128 + np.abs(b)).max())
np.savez(outp, w=W, b=b, sh=np.int64(8))
print('p13 parsed: blocks %d, taps %s, |w|max %d, |b|max %d, acc bound %d (< 2^24 %s)' % (len(blocks), tmap, np.abs(W).max(), np.abs(b).max(), acc_max, acc_max < 1 << 24))
