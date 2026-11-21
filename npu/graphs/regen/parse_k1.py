#!/usr/bin/env python3
"""Weights of the K1 block of FSR 4.1.1 pass 27 (3x3 conv 16->16, 1x1 16->32 + relu, 1x1 32->16 + residual), read from the
GLSL of the transformed pass 27 (t/q27.comp, see decompile.sh) by running its code, not by pattern-matching weights.
1. LIT: the K1 part of main() (the bounds-check block up to the four packed K1 words before 'value[5u]') is translated
   statement by statement to numpy (uint32 wrap, int32 arithmetic shifts, fp32 roundEven, SUDot; an if/else runs both
   branches and merges with np.where; the scratch buffer is the 962x542 uvec4 output region).
2. PROBE: the weights come from impulse inputs to that code: stage A accumulators (3x3), then the stage B and C accumulator
   sub-expressions (1x1).
3. CLEAN: a plain conv model with the probed weights; on random inputs LIT and CLEAN must give the same bytes (0 differ).
Writes OUT_DIR/k1_w.npz: kind 'g', s1, s2 (requant shifts of stages A and B), sf, R (per-channel store shift and residual
factor), lo, w1 b1 w2 b2 w3 b3 (int64).
Usage: parse_k1.py Q27_COMP OUT_DIR [ROWS]   (ROWS = height of the random check images, default 60; the width is 960)"""
import re, sys, os, numpy as np
q27, od = sys.argv[1:3]; os.makedirs(od, exist_ok=True); ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 60
S = open(q27).read().split('\n')
i0 = [i for i, l in enumerate(S) if re.match(r'^    if \(!\(\(_\d+ >= _\d+\) \|\| \(_\d+ >= _\d+\)\)\)$', l)][0]
m = re.match(r'^    if \(!\(\((_\d+) >= (_\d+)\) \|\| \((_\d+) >= (_\d+)\)\)\)$', S[i0]); VX, VW, VY, VH = m.groups()
VG = re.search(r'uint (_\d+) = gl_WorkGroupID\.x << 4u;', '\n'.join(S[:i0])).group(1)
i1 = [i for i in range(i0, len(S)) if 'PhysicalPointerFloat4NonWriteCBVArray(registers._m0).value[5u]' in S[i]][0]
blk = S[i0 + 2:i1]
FN = (('spvSUDot_uint_uint_uint', 'SUD'), ('roundEven', 'RE'), ('pack32', 'PACK'), ('unpack8', 'UNPACK8'), ('clamp', 'CLAMP'), ('max', 'MAX'),
      ('u8vec4', 'U8V4'), ('i8vec4', 'I8V4'), ('uvec4', 'UV4'), ('ivec4', 'IV4'), ('uint', 'UI'), ('int', 'I'), ('float', 'F'))
def tr(e):
    assert '?' not in e, e
    e = re.sub(r'_14\[_\d+\]\._m0\[', 'SCR[', e); e = e.replace('registers._m1', '0')
    e = re.sub(r'\b(\d+)u\b', r'\1', e)
    for a, b in FN: e = re.sub(r'\b%s\(' % a, b + '(', e)
    e = re.sub(r'\b(_\w+)\.([xyzw])\b', lambda q: '%s[%d]' % (q.group(1), 'xyzw'.index(q.group(2))), e)
    e = e.replace('&&', '&').replace('||', '|'); e = re.sub(r'!(?!=)', '~', e)
    return e
def parse(lines, k=0):   # -> list of ('s', var, code, text) | ('if', cond, then, else)
    out = []
    while k < len(lines):
        t = lines[k].strip(); k += 1
        if not t or re.match(r'^(uint|uvec4|int|bool|vec4|float) _\w+;$', t): continue
        if t == '}': return out, k
        if t.startswith('if ('):
            assert lines[k].strip() == '{'; th, k = parse(lines, k + 1); el = []
            if k < len(lines) and lines[k].strip() == 'else':
                assert lines[k + 1].strip() == '{'; el, k = parse(lines, k + 2)
            out.append(('if', tr(t[3:]), th, el)); continue
        m = re.match(r'^(?:uint |uvec4 |int |bool |float )?(_\w+) = (.*);$', t); assert m, t
        out.append(('s', m.group(1), compile(tr(m.group(2)), m.group(1), 'eval'), t))
    return out, k
PROG, _ = parse(blk)
U32 = np.uint32
def UI(x):
    if isinstance(x, (int, np.integer)): return int(x) & 0xFFFFFFFF
    if x.dtype == np.bool_: return x.astype(U32)
    assert x.dtype in (np.uint32, np.int32), x.dtype; return x.view(U32)
def I(x):
    if isinstance(x, (int, np.integer)): x = int(x) & 0xFFFFFFFF; return x - (1 << 32) if x >> 31 else x
    if x.dtype == np.float32: return x.astype(np.int32)
    assert x.dtype in (np.uint32, np.int32), x.dtype; return x.view(np.int32)
def F(x): assert x.dtype == np.int32; return x.astype(np.float32)
def RE(x): assert x.dtype == np.float32, x.dtype; return np.rint(x)
def MAX(a, b): return np.maximum(a, np.int32(b))
_bc = {}
def planes(v):
    k = id(v); h = _bc.get(k)
    if h is None or h[0] is not v:
        h = (v, [((v >> U32(8 * j)) & U32(0xFF)).astype(np.int64) for j in range(4)]); _bc[k] = h
    return h[1]
def SUD(c, v, z):
    ws = [((c >> (8 * j)) & 0xFF) - (256 if (c >> (8 * j)) & 0x80 else 0) for j in range(4)]
    if isinstance(v, (int, np.integer)) and np.ndim(v) == 0: return U32((sum(w * ((int(v) >> (8 * j)) & 0xFF) for j, w in enumerate(ws)) + int(z)) & 0xFFFFFFFF)
    p = planes(v); acc = ws[0] * p[0] + ws[1] * p[1] + ws[2] * p[2] + ws[3] * p[3]
    return (acc & 0xFFFFFFFF).astype(U32)
def UV4(*a):
    if len(a) == 4:
        b = np.broadcast_arrays(*[np.asarray(UI(t), U32) for t in a]); return np.stack(b)
    x = a[0]
    if isinstance(x, (int, np.integer)): return U32(x)
    if x.dtype == np.int8: return x.astype(np.int32).view(U32)
    return UI(x)
def IV4(*a):
    assert len(a) == 1; return I(a[0])
def CLAMP(v, lo, hi): return np.clip(v, lo, hi).astype(v.dtype)
def U8V4(v): return v.astype(np.int64).astype(np.uint8)
def PACK(v): v = v.astype(U32); return v[0] | (v[1] << U32(8)) | (v[2] << U32(16)) | (v[3] << U32(24))
def UNPACK8(w): return np.stack([((w >> U32(8 * j)) & U32(0xFF)).astype(np.uint8) for j in range(4)])
def I8V4(b): return b.view(np.int8)
class Scr:
    def __init__(s, a): s.a = a
    def __getitem__(s, i): return s.a[np.clip(np.asarray(i).astype(np.int64), 0, s.a.size - 1)]
BASE = dict(UI=UI, I=I, F=F, RE=RE, MAX=MAX, SUD=SUD, UV4=UV4, IV4=IV4, CLAMP=CLAMP, U8V4=U8V4, PACK=PACK, UNPACK8=UNPACK8, I8V4=I8V4, np=np)
def run(prog, ns):
    for st in prog:
        if st[0] == 's':
            with np.errstate(over='ignore'): ns[st[1]] = eval(st[2], ns)
        else:
            c = np.asarray(eval(compile(st[1], 'cond', 'eval'), ns)); a = dict(ns); b = dict(ns); run(st[2], a); run(st[3], b)
            for k in set(a) | set(b):
                va, vb = a.get(k), b.get(k)
                if va is vb: ns[k] = va; continue
                if va is None or vb is None: ns[k] = va if vb is None else vb; continue
                if isinstance(va, (int, np.integer)) and isinstance(vb, (int, np.integer)) and np.ndim(va) == 0 and np.ndim(vb) == 0 and int(va) == int(vb): ns[k] = va; continue
                va = np.asarray(va); vb = np.asarray(vb)
                if va.dtype != vb.dtype:
                    if va.dtype.kind in 'iu' and va.dtype != U32 and va.dtype != np.int32: va = va.astype(U32)
                    vb = vb.astype(va.dtype)
                ns[k] = np.where(c, va, vb)
    return ns
flat = [st for st in PROG if st[0] == 's']
packs = [st for st in flat if st[3].startswith('uint ') and 'pack32(u8vec4(clamp(ivec4(uvec4(' in st[3]]
fin = packs[-4:]; A = [st for st in packs if re.search(r'\+ (\d+)u\) \+ \(\(_\d+ >> (\d+)u\) & 1u\)\) >> int\(\d+u\)', st[3])]
A_, B_ = A[:4], A[4:12]; assert len(A) == 12, len(A)
def accs(st): return re.findall(r'uint\(int\(\((_\d+) \+ \d+u\) \+ \(\(_\d+ >> (\d+)u\) & 1u\)\) >> int\(\d+u\)\)', st[3])
s1 = {int(s) for st in A_ for _, s in accs(st)}; s2 = {int(s) for st in B_ for _, s in accs(st)}; assert len(s1) == 1 and len(s2) == 1, (s1, s2)
s1, s2 = s1.pop(), s2.pop()
Aw = [st[1] for st in A_]; Bw = [st[1] for st in B_]; Ow = [st[1] for st in fin]
print('K1 words: A', Aw, 's1', s1, '| B', Bw, 's2', s2, '| out', Ow, flush=True)
defs = {st[1]: st[3] for st in flat}
def k1_lit(x, rows=60):
    H, W, _ = x.shape; assert W + 2 == 962, W   # the GLSL row stride 15392 B = 962 uvec4
    reg = np.zeros((H + 2, W + 2, 16), np.int8); reg[1:-1, 1:-1] = x
    scr = Scr(np.ascontiguousarray(reg).view(U32).ravel()); out = np.zeros((H, W, 4), U32)
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows); yy, xx = np.mgrid[y0:y1, 0:W]; xx = xx.ravel().astype(U32); yy = yy.ravel().astype(U32)
        ns = dict(BASE, SCR=scr); ns.update({VX: xx, VY: yy, VW: U32(W), VH: U32(H), VG: (xx & U32(0xFFFFFFF0))}); _bc.clear()
        run(PROG, ns); out[y0:y1] = np.stack([np.broadcast_to(ns[w], xx.shape) for w in Ow], -1).reshape(y1 - y0, W, 4)
    return out.view(np.int8).reshape(H, W, 16)
def sub_after(t, key):   # balanced-paren text after key
    i = t.index(key) + len(key); d = 1; j = i
    while d: d += {'(': 1, ')': -1}.get(t[j], 0); j += 1
    return t[i:j - 1]
s32 = lambda a: np.asarray(a).astype(U32).view(np.int32).astype(np.int64)
def acc_A(x):   # stage-A accumulators (16, H, W) signed, literal code on rows 0..H-1 of a 960-wide image
    H, W, _ = x.shape; reg = np.zeros((H + 2, W + 2, 16), np.int8); reg[1:-1, 1:-1] = x
    yy, xx = np.mgrid[0:H, 0:8]; xx = xx.ravel().astype(U32); yy = yy.ravel().astype(U32)
    ns = dict(BASE, SCR=Scr(reg.view(U32).ravel())); ns.update({VX: xx, VY: yy, VW: U32(W), VH: U32(H), VG: (xx & U32(0xFFFFFFF0))}); _bc.clear()
    stop = [k for k, st in enumerate(PROG) if st[0] == 's' and st[1] == Aw[0]][0]
    run(PROG[:stop], ns)
    return np.stack([s32(np.broadcast_to(ns[v], xx.shape)) for st in A_ for v, _ in accs(st)]).reshape(16, H, 8)
z = np.zeros((5, 960, 16), np.int8); a0 = acc_A(z); b1 = a0[:, 2, 2].copy()
w1 = np.zeros((16, 16, 3, 3), np.int64)
for c in range(16):
    x = z.copy(); x[2, 2, c] = 1; d = acc_A(x) - a0
    for dy in range(3):
        for dx in range(3): w1[:, c, dy, dx] = d[:, 2 - (dy - 1), 2 - (dx - 1)]
    x[2, 2, c] = 2; assert ((acc_A(x) - a0) == 2 * d).all()
def ev(expr, env): ns = dict(BASE); ns.update(env); _bc.clear(); return s32(eval(compile(tr(expr), 'p', 'eval'), ns))
def words_of(vals):
    v = np.asarray(vals, np.int64).astype(np.int8).view(np.uint8).astype(U32)
    return [np.array([v[4*i] | (v[4*i+1] << U32(8)) | (v[4*i+2] << U32(16)) | (v[4*i+3] << U32(24))], U32) for i in range(len(v) // 4)]
Bx = [sub_after(defs[v], 'uint(max(int(') for st in B_ for v, _ in accs(st)]
def evB(h): env = dict(zip(Aw, words_of(h))); return np.array([ev(e, env)[0] for e in Bx])
b2 = evB(np.zeros(16)); w2 = np.zeros((32, 16, 1, 1), np.int64)
for c in range(16):
    e = np.zeros(16); e[c] = 1; w2[:, c, 0, 0] = evB(e) - b2; e[c] = -3; assert (evB(e) - b2 == -3 * w2[:, c, 0, 0]).all()
fv = [re.findall(r'uvec4\((_\d+), (_\d+), (_\d+), (_\d+)\)', defs[w])[-1] for w in Ow]
RX = r'^uint _\d+ = uint\(int\(roundEven\(\(\(float\(int\((_\d+)\.([xyzw])\)\) \* ([0-9.e+-]+)\) \+ \(float\(int\((.*)\)\) \* ([0-9.e+-]+)\)\)(?: \* ([0-9.e+-]+))?\)\)\);$'
Cx, RR, SF = [], [], []
for j, vs in enumerate(fv):
    for k, v in enumerate(vs):
        m = re.match(RX, defs[v]); assert m, defs[v][:200]
        cv, comp, fr, ex, fa, fm = m.groups(); sw = re.match(r'uvec4 _\d+ = uvec4\(i8vec4\(unpack8\((_\d+)\.([xyzw])\)\)\);', defs[cv])
        ch = 4 * 'xyzw'.index(sw.group(2)) + 'xyzw'.index(comp); assert ch == 4 * j + k, (v, ch)
        fm = float(fm) if fm else 1.0; fr, fa = float(fr) * fm, float(fa) * fm; sf = int(round(-np.log2(fa))); assert fa == 2.0 ** -sf; R = fr / fa; assert R == int(R)
        Cx.append(ex); RR.append(int(R)); SF.append(sf)
def evC(m_): env = dict(zip(Bw, words_of(m_))); return np.array([ev(e, env)[0] for e in Cx])
b3 = evC(np.zeros(32)); w3 = np.zeros((16, 32, 1, 1), np.int64)
for c in range(32):
    e = np.zeros(32); e[c] = 1; w3[:, c, 0, 0] = evC(e) - b3; e[c] = 5; assert (evC(e) - b3 == 5 * w3[:, c, 0, 0]).all()
V = dict(kind='g', s1=s1, s2=s2, sf=np.array(SF), R=np.array(RR), lo=16, w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)
print('probed: s1 %d s2 %d sf %s R %s |w1| %d |w2| %d |w3| %d |b| %d %d %d' % (s1, s2, sorted(set(SF)), sorted(set(RR)), np.abs(w1).max(), np.abs(w2).max(),
      np.abs(w3).max(), np.abs(b1).max(), np.abs(b2).max(), np.abs(b3).max()), flush=True)
assert max(np.abs(w).max() for w in (w1, w2, w3)) <= 128
np.savez(od + '/k1_w.npz', **V)
def rq(a, s, relu=False):
    a = np.maximum(a, 0) if relu else a; s = np.asarray(s); h = np.int64(1) << (s - 1)
    return np.clip((a + h - 1 + ((a >> s) & 1)) >> s, -128, 127)
def k1_clean(x):
    x = x.astype(np.int64); H, W, C = x.shape; xp = np.zeros((H + 2, W + 2, C), np.int64); xp[1:-1, 1:-1] = x; acc = np.zeros((H, W, 16), np.int64) + b1
    for dy in range(3):
        for dx in range(3): acc += xp[dy:dy + H, dx:dx + W] @ w1[:, :, dy, dx].T
    h = rq(acc, s1); m_ = rq(h @ w2[:, :, 0, 0].T + b2, s2, True)
    return rq(m_ @ w3[:, :, 0, 0].T + b3 + np.array(RR) * x, np.array(SF)).astype(np.int8)
rng = np.random.default_rng(27)
ins = {'rnd': rng.integers(-128, 128, (ROWS, 960, 16)).astype(np.int8), 'rnd_small': rng.integers(-8, 8, (ROWS, 960, 16)).astype(np.int8)}
tot = 0
for k, x in ins.items():
    a = k1_lit(x); b = k1_clean(x); d = int((a != b).sum()); tot += d
    print('%-9s LIT vs CLEAN bytes differ %d of %d  range %d..%d' % (k, d, a.size, a.min(), a.max()), flush=True)
print('TOTAL bytes differ', tot); assert tot == 0; print('ok')
