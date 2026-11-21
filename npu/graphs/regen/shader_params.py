#!/usr/bin/env python3
"""Integer parameters of the FSR 4.1.1 INT8 network, read from the spirv-cross GLSL of the fsr4vk passes and from
initializers.bin. Writes params_n1.pkl (the input of gen/gen_n1.py) and, for Ultra Performance, p07_sh.npy.

Usage: shader_params.py PRESET GLSL_DIR INITIALIZERS_BIN OUT_DIR
  PRESET    performance | ultraperf
  GLSL_DIR  the folder decompile.sh writes for that preset: o/pNN.comp (decompile of the original pass) and
            t/qNN.comp (decompile of the transformed pass, for passes 15, 17, 19)

How the values are found. Each pass is decompiled with `spirv-cross --vulkan-semantics`. Every scalar `uint _N = expr;`
whose expression only uses constants and the invocation id is evaluated at invocation (0, 0, 0); that gives the byte
address of every load. A load from the weight buffer gives the weight word (read from initializers.bin), a load from
the scratch buffer gives the activation tap (row, column, word), and each spvSDot(activation, weight) pairs them.
Requantisation shifts, store scales and residual factors are read from the literal constants of the store lines.
Every mapping is checked by asserts, so a different decompiler text stops the script instead of giving other numbers.
The variable names in the patterns (_14, _15, _20, _21, _25, _26, ...) are the names this SPIRV-Cross commit gives
(Dockerfile in this folder).

pickle layout (one key per pass, in this order):
  p03 p05 p25  residual blocks (3x3 16->16, 1x1 16->32, 1x1 32->16 + residual): W1 W2 W3 {out ch: {tap: weight word}},
               b1 b2 b3 (word index of the bias), cons (shifts, scales, offsets), header (GLSL lines before main)
  p07          (W, bias): 2x2 stride-2 conv 16->32, W {out ch: {(dy, dx, word): weight word}}, bias {ch: word}
  p09 p11 p21  3 fused stages (3x3 16->16, 1x1 32->64 + relu, 1x1 64->32 + residual): W {stage: {ch: {key: word}}},
               bias, s1 s2 (shifts), sfR (store shift, residual factor), fin (residual source and store offsets), lim
  p15 p17      bottleneck (grouped 3x3 2x16->16, 1x1 64->128 + relu, 1x1 128->64 + residual): offsets, shifts, scales,
               w1 w2 w3 (int8 lists), b1 b2 b3 (int32 lists)
  p19          as p15, then a 2x2 transposed conv 64->32 per sub-pixel + residual: w4 b4 and K4 (folded bias)
  p23          4 fused stages: S1 S2 S3 S4 (bias, {key: weight word}); Ultra Performance also has 'variant': 'ultraperf'
"""
import re, sys, os, struct, itertools, pickle
from fractions import Fraction
import numpy as np


def sb(w):
    return [((w >> (8 * i)) & 0xff) - (256 if (w >> (8 * i)) & 0x80 else 0) for i in range(4)]


def s32(v):
    return v - (1 << 32) if v & 0x80000000 else v


# ---------------------------------------------------------------- p03 p05 p25 (residual blocks)
def parse_res(path):
    L = open(path).read().split('\n')
    defs = {}
    for l in L:
        m = re.match(r'^\s*(?:uint|uvec4|int|bool)\s+(_\d+)\s*=\s*(.*);$', l)
        if m:
            defs[m.group(1)] = m.group(2)
    memo = {}

    def evvar(v):
        if v in memo:
            return memo[v]
        r = ev(defs[v])
        memo[v] = r
        return r

    def ev(expr, X=0, Y=0):
        e = expr.replace('gl_GlobalInvocationID.x', 'X').replace('gl_GlobalInvocationID.y', 'Y')
        e = e.replace('gl_GlobalInvocationID.z', '0')
        e = re.sub(r'(\d+)u\b', r'\1', e)
        e = re.sub(r'_\d+', lambda m: '(%d)' % evvar(m.group(0)), e)
        return eval(e, {'X': X, 'Y': Y})

    def trace_arr(expr):
        m = re.search(r'(_\d+)\[(\d+)u\]', expr)
        if m:
            return m.group(1), int(m.group(2))
        for v in re.findall(r'_\d+', expr):
            if v in defs:
                r = trace_arr(defs[v])
                if r:
                    return r
        return None

    first = next(l for l in L if re.search(r'uvec4 _\d+ = uvec4\(_26\[', l))
    in_off = evvar(re.search(r'_m0\[(_\d+)\]', first).group(1)) * 4

    def resolve_a(v):
        d = defs[v]
        m = re.match(r'^(_\d+)\.([xyzw])$', d)
        if m and defs[m.group(1)].startswith('uvec4(_26['):
            ad = re.search(r'_m0\[(_\d+)\]', defs[m.group(1)]).group(1)
            byte = evvar(ad) * 4 - in_off
            dy, rem = divmod(byte, 15392)
            assert rem % 16 == 0
            return ('in', dy, rem // 16, 'xyzw'.index(m.group(2)))
        if d.startswith('pack32('):
            arr, idx = trace_arr(d)
            return ('pk', arr, idx // 4)
        raise Exception('unknown activation %s = %s' % (v, d))

    tabs, loops = {}, []
    for i, l in enumerate(L):
        m = re.match(r'^\s*(_\d+)\[(\d+)u\] = (\d+)u;$', l)
        if m:
            tabs.setdefault(m.group(1), {})[int(m.group(2))] = int(m.group(3))
            continue
        if 'for (;;)' not in l:
            continue
        j = i + 1
        while 'spvSDot' not in L[j]:
            j += 1
        m = re.match(r'^\s*(_\d+)\[([^\]]+)\] = (.*);$', L[j])
        acc, idx, rhs = m.groups()
        mm = re.match(r'^(_\d+) \+ (\d+)u$', idx)
        if not mm:
            mm = re.match(r'^(_\d+) \+ (\d+)u$', defs[idx])
        base = int(mm.group(2))
        terms = re.findall(r'spvSDot_uint_uint_uint\((_\d+), (_\d+), 0\)', rhs)
        wmap = {}
        k = j
        while 'continue;' not in L[k]:
            mw = re.match(r'^\s*(_\d+) = (_\d+)\[(\d+)u \+ \((_\d+) \* 4u\)\];', L[k])
            if mw:
                wmap[mw.group(1)] = (mw.group(2), int(mw.group(3)))
            if re.search(r'== (\d+)u', L[k]):
                assert int(re.search(r'== (\d+)u', L[k]).group(1)) == 4
            k += 1
        loops.append((acc, base, [(resolve_a(a), wmap[w]) for a, w in terms], {t: dict(tabs[t]) for t in {x[0] for x in wmap.values()}}))
    W = {}
    for acc, base, terms, tb in loops:
        for it in range(4):
            c = base + it
            for a, (t, lane) in terms:
                d = W.setdefault(acc, {}).setdefault(c, {})
                assert a not in d, (acc, c, a)
                d[a] = tb[t][4 * it + lane]
    arrs = sorted(W)
    c1 = [a for a in arrs if next(iter(W[a][0]))[0] == 'in'][0]
    rest = [a for a in arrs if a != c1]
    # conv 2 has 4 inputs per channel, conv 3 has 8
    c2 = [a for a in rest if len(W[a][0]) == 4][0]
    c3 = [a for a in rest if len(W[a][0]) == 8][0]
    W1 = {c: {k: v for k, v in d.items()} for c, d in W[c1].items()}
    assert len(W1) == 16 and all(len(d) == 36 for d in W1.values())
    W2 = {c: {k[2]: v for k, v in d.items()} for c, d in W[c2].items()}
    assert len(W2) == 32 and all(sorted(d) == [0, 1, 2, 3] for d in W2.values())
    W3 = {c: {k[2]: v for k, v in d.items()} for c, d in W[c3].items()}
    assert len(W3) == 16 and all(sorted(d) == list(range(8)) for d in W3.values())
    W1 = {c: {(k[1], k[2], k[3]): v for k, v in d.items()} for c, d in W1.items()}
    # bias word offsets (uint index), contiguous per conv
    bias = {}
    for arr in (c1, c2, c3):
        got = {}
        for i, l in enumerate(L):
            m = re.match(r'^\s*%s\[(\d+)u\] = _21\[_\d+\]\._m0\[(_\d+)\]\.([xyzw]);$' % arr, l)
            if m:
                got[int(m.group(1))] = 4 * evvar(m.group(2)) + 'xyzw'.index(m.group(3))
            m = re.match(r'^\s*%s\[(\d+)u\] = (_\d+)\.([xyzw]);$' % arr, l)
            if m and defs.get(m.group(2), '').startswith('uvec4(_15['):
                ad = re.search(r'_m0\[(_\d+)\]', defs[m.group(2)]).group(1)
                got[int(m.group(1))] = evvar(ad) + 'xyzw'.index(m.group(3))
        n = len(W[arr])
        b0 = got[0]
        assert sorted(got) == list(range(n)) and all(got[k] == b0 + k for k in range(n)), (arr, got)
        bias[arr] = b0
    txt = '\n'.join(L)
    pk = [l for l in L if 'pack32(' in l and 'roundEven' not in l and '>> int(' in l]
    s1 = int(re.search(r'>> int\((\d+)u\)', next(l for l in pk if c1 + '[' in l)).group(1))
    s2 = int(re.search(r'>> int\((\d+)u\)', next(l for l in pk if c1 + '[' not in l)).group(1))
    mo = re.search(r'\? \((_\d+) \+ _55\) : 134217728u', txt)
    Mm = int(re.search(r'roundEven\(.*?\* (\d+)\.0\)', txt).group(1))
    ab = float(re.search(r'float\(int\(_\d+\[0u\]\)\) \* ([0-9.]+)\)', txt).group(1))
    ar = float(re.search(r'float\(int\(_\d+\.x\)\) \* ([0-9.]+)\)', txt).group(1))
    assert abs(ar - 64 * ab) < 1e-12 and abs(1 / (Mm * ab) - round(1 / (Mm * ab))) < 1e-9, (ar, ab, Mm)
    cons = dict(
        m1=int(re.search(r'registers\._m1 \+ (\d+)u', txt).group(1)),
        m2=int(re.search(r'registers\._m2 \+ (\d+)u', txt).group(1)),
        stride=int(re.search(r'(\d+)u \* gl_GlobalInvocationID\.y', txt).group(1)),
        in_off=in_off, out_off=evvar(mo.group(1)) - 15408, M=Mm, ar=ar, ab=ab, D=round(1 / (Mm * ab)),
        lim=int(re.search(r'\.value\[(\d+)u\]', txt).group(1)), s1=s1, s2=s2)
    return dict(W1=W1, W2=W2, W3=W3, b1=bias[c1], b2=bias[c2], b3=bias[c3], cons=cons, header=L[:txt[:txt.index('void main()')].count('\n')])


# ---------------------------------------------------------------- evaluator for p07, p09/p11/p21, p23
class Shader:
    """Decompiled pass: definitions `_N = expr` and an evaluator at invocation 0. WN = names of the weight buffer
    (initializers.bin words), MN = names of the scratch buffers (an address there is kept as a word index)."""
    def __init__(self, path, words, WN, MN):
        self.L = open(path).read().split('\n')
        self.words, self.WN, self.MN = words, WN, MN
        self.defs = {}
        for l in self.L:
            m = re.match(r'^\s*(?:uint|uvec2|uvec4|int|bool)\s+(_\d+)\s*=\s*(.*);$', l)
            if m:
                self.defs[m.group(1)] = m.group(2)

    def ctx(self, **env):
        return Ctx(self, env)


class Ctx:
    def __init__(self, sh, env):
        self.sh, self.env, self.memo = sh, env, {}

    def N(self, name):
        if name in self.env: return self.env[name]
        if name in self.memo: return self.memo[name]
        r = self.ev(self.sh.defs[name])
        self.memo[name] = r
        return r

    def ev(self, expr):
        WN, MN, WORDS = self.sh.WN, self.sh.MN, self.sh.words
        e = expr.replace('gl_GlobalInvocationID.x', '0').replace('gl_GlobalInvocationID.y', '0')
        e = e.replace('gl_GlobalInvocationID.z', '0').replace('gl_WorkGroupID.z', '0')
        e = e.replace('registers._m1', '0').replace('registers._m2', '0')
        e = re.sub(r'\b(\d+)u\b', r'\1', e)
        e = re.sub(r'(?:%s)\[[^\[\]]+\]\._m0\[([^\[\]]+)\]' % '|'.join(WN), r'W(\1)', e)
        e = re.sub(r'(?:%s)\[[^\[\]]+\]\._m0\[([^\[\]]+)\]' % '|'.join(MN), r'M(\1)', e)
        e = e.replace('uvec2(', 'V(').replace('uvec4(', 'V(')

        def sub(m):
            if m.group(2):
                return "N('%s')['xyzw'.index('%s')]" % (m.group(1), m.group(2))
            return "N('%s')" % m.group(1)
        e = re.sub(r'(_\d+)(?:\.([xyzw]))?', sub, e)
        return eval(e, {'N': self.N, 'W': lambda i: WORDS[i], 'M': lambda i: i, 'V': lambda *a: tuple(a)})

    def vecload(self, tok):
        """'_154' (= _153.x, _153 = uvec2(_20[..]._m0[a], ...)) or '_153.x' -> ('m'|'w', word index)"""
        m = re.match(r'^(_\d+)(?:\.([xyzw]))?$', tok)
        name, comp = m.group(1), m.group(2)
        d = self.sh.defs[name]
        if comp is None:
            mm = re.match(r'^(_\d+)\.([xyzw])$', d)
            name, comp = mm.group(1), mm.group(2)
            d = self.sh.defs[name]
        loads = re.findall(r'(%s)\[[^\[\]]+\]\._m0\[([^\[\]]+)\]' % '|'.join(self.sh.WN + self.sh.MN), d)
        which, ad = loads['xyzw'.index(comp)]
        return ('w' if which in self.sh.WN else 'm'), self.ev(ad)

    def chain(self, text):
        """flatten  head + D(a,w) + ...  ->  (bias word value, [(activation token, weight token)])"""
        terms = re.findall(r'spvSDot_uint_uint_uint\((.+?), (_\d+\.[xyzw]), 0\)', text)
        head = re.search(r'\(*(_\d+(?:\.[xyzw])?) \+ spvSDot', text).group(1)
        if '.' not in head and 'spvSDot' in self.sh.defs[head]:
            b, t0 = self.chain(self.sh.defs[head])
            return b, t0 + terms
        kind, wi = self.vecload(head)
        assert kind == 'w'
        return self.sh.words[wi], terms

    def weight(self, tok):
        kind, wi = self.vecload(tok)
        assert kind == 'w'
        return self.sh.words[wi]


# ---------------------------------------------------------------- p07 (2x2 stride-2 conv 16->32)
def parse_p07(path, WORDS):
    sh = Shader(path, WORDS, ['_14', '_20'], ['_25'])
    L = sh.L
    full = '\n'.join(L)
    c0 = sh.ctx()
    tab = {}
    bias = {}
    W = {}

    def bias_word(rhs):
        m = re.match(r'^_20\[[^\[\]]+\]\._m0\[([^\[\]]+)\]\.([xyzw])$', rhs)
        if m:
            return WORDS[4 * c0.ev(m.group(1)) + 'xyzw'.index(m.group(2))]
        kind, wi = c0.vecload(rhs)
        assert kind == 'w'
        return WORDS[wi]
    i = 0
    seen = False
    while i < len(L):
        l = L[i]
        m = re.match(r'^\s*_46\[(\d+)u\] = (_\d+\.[xyzw]);$', l)
        if m:
            kind, wi = c0.vecload(m.group(2)); assert kind == 'w'
            tab[int(m.group(1))] = WORDS[wi]; i += 1; continue
        m = re.match(r'^\s*_42\[(\d+)u\] = (.+);$', l)
        if m and not seen:
            bias[int(m.group(1))] = bias_word(m.group(2)); i += 1; continue
        if l.strip() == 'for (;;)':
            j = i
            while 'continue;' not in L[j]:
                j += 1
            block = L[i:j + 1]; btxt = '\n'.join(block)
            line = next(x for x in block if 'spvSDot' in x)
            tm = re.match(r'^\s*_42\[([^\]]+)\] = ', line)
            idx = tm.group(1)
            km = re.search(r'\+ (\d+)u', idx)
            K = int(km.group(1)) if km else int(re.search(r'uint %s = _\d+ \+ (\d+)u;' % re.escape(idx), btxt).group(1))
            wmap = {m.group(1): int(m.group(2)) for m in re.finditer(r'^\s*(_\d+) = _46\[(\d+)u \+ \(_\d+ \* 4u\)\];', btxt, re.M)}
            terms = re.findall(r'spvSDot_uint_uint_uint\((_\d+), (_\d+), 0\)', line)
            assert len(terms) == 4
            seen = True
            for it in range(4):
                d = W.setdefault(K + it, {})
                for act, wv in terms:
                    kind, wi = c0.vecload(act); assert kind == 'm'
                    rel = wi * 4 - 15392 - 16
                    dy, rem = divmod(rel, 15392)
                    dx, w4 = divmod(rem, 16)
                    assert dy in (0, 1) and dx in (0, 1) and w4 % 4 == 0, (act, rel)
                    key = (dy, dx, w4 // 4)
                    assert key not in d
                    d[key] = tab[4 * it + wmap[wv]]
            i = j + 1; continue
        i += 1
    assert len(W) == 32 and all(len(d) == 16 for d in W.values()), (len(W))
    assert sorted(bias) == list(range(32)), sorted(bias)
    # store scale: Performance '* 0.0078125'; Ultra Performance 'float _N = (_47 < 16u) ? 0.0078125 : 0.00390625' with
    # _47 = z << 5 (z = 0 at dispatch). The per-channel shift is p07_sh() below.
    assert ('* 0.0078125' in full or (re.search(r'float _\d+ = \(_47 < 16u\) \? 0\.0078125 : ', full) and 'uint _47 = gl_GlobalInvocationID.z << 5u;' in full)) and 'registers._m0 + 11u' in full
    m = max(sum(abs(x) for w in d.values() for x in sb(w)) * 128 + abs(s32(bias[c])) for c, d in W.items())
    assert m // 128 < 32000
    return W, bias


def p07_sh(path):
    """Store shift per output channel of pass 07. Ultra Performance: channels 0..15 store * (_47 < 16u ? 2^-7 : 2^-8)
    with _47 = z << 5 = 0, so 2^-7; channels 16..31 store * 2^-8. Gives [7]*16 + [8]*16."""
    t = open(path).read(); sh = {}
    tern = dict(re.findall(r'float (_\d+) = \(_47 < 16u\) \? ([0-9.]+) : [0-9.]+;', t))
    for ch, mul in re.findall(r'roundEven\(float\(int\(_42\[(\d+)u\]\)\) \* (_\d+|[0-9.]+)\)', t):
        v = float(tern[mul]) if mul.startswith('_') else float(mul); s = int(round(-np.log2(v))); assert 2.0 ** -s == v
        assert sh.setdefault(int(ch), s) == s
    assert sorted(sh) == list(range(32)); return np.array([sh[c] for c in range(32)], np.int64)


# ---------------------------------------------------------------- p09 p11 p21 (3 fused stages)
def parse_h09(path, WORDS):
    sh = Shader(path, WORDS, ['_15'], ['_20', '_26'])
    L = sh.L
    full = '\n'.join(L)
    decls = [(m.group(1), int(m.group(2))) for m in re.finditer(r'^\s*uint (_\d+)\[(\d+)\];', full, re.M)]
    assert [s for _, s in decls] == [16, 16, 8, 64, 16, 32, 16], decls
    acc1, tab1, buf, acc2, tab2, acc3, tab3 = [d[0] for d in decls]
    accs = {acc1: 1, acc2: 2, acc3: 3}
    tabnames = {tab1, tab2, tab3}
    packs = re.findall(r'^\s*uint (_\d+) = pack32\(u8vec4\(clamp\(ivec4\(uvec4\(uint\(int\(', full, re.M)
    assert len(packs) == 20, len(packs)

    def src_k(tok):
        m = re.match(r'^(_\d+)\[(\d+)u\]$', tok)
        if m and m.group(1) in accs:
            return accs[m.group(1)], int(m.group(2))
        d = sh.defs.get(tok, '')
        m = re.match(r'^uint\(max\(int\((.+)\), int\(0u\)\)\)$', d)
        if m:
            return src_k(m.group(1))
        if re.match(r'^_\d+\[\d+u\]$', d) or re.match(r'^_\d+$', d):
            return src_k(d)
        return None
    pmap = {}
    for v in packs:
        line = next(x for x in L if re.match(r'^\s*uint %s = pack32' % v, x))
        ks = []
        for tok in re.findall(r'_\d+(?:\[\d+u\])?', line.split('=', 1)[1]):
            r = src_k(tok)
            if r and r not in ks:
                ks.append(r)
        assert len(ks) == 4 and len(set(k // 4 for _, k in ks)) == 1 and [k % 4 for _, k in ks] == [0, 1, 2, 3] and len(set(s for s, _ in ks)) == 1, (v, ks)
        st, k0 = ks[0]
        assert st in (1, 2)
        pmap[v] = k0 // 4 + (0 if st == 1 else 4)
    assert sorted(pmap.values()) == list(range(20)), pmap
    c0 = sh.ctx(_66=7728, _67=7728)
    Abase = int(re.search(r'\(7712u \* gl_GlobalInvocationID\.y\) \+ _\d+\) \+ (\d+)u;', full).group(1))
    bufall = [(int(k), c0.ev(x)) for k, x in re.findall(r'%s\[(\d+)u\] = _26\[[^\]]+\]\._m0\[([^\]]+)\];' % re.escape(buf), full)]
    bufw = {k: v for k, v in bufall if k >= 4}
    buflast = dict(bufall)
    Bbase = bufw[4] * 4 - 7728
    assert sorted(bufw) == [4, 5, 6, 7] and [bufw[k] * 4 - 7728 - Bbase for k in bufw] == [0, 4, 8, 12]

    def bias_word(rhs):
        rhs = rhs.strip()
        m = re.match(r'^_21\[[^\[\]]+\]\._m0\[([^\[\]]+)\]\.([xyzw])$', rhs)
        if m:
            return WORDS[4 * c0.ev(m.group(1)) + 'xyzw'.index(m.group(2))]
        m = re.match(r'^(_\d+)\.([xyzw])$', rhs)
        if m:
            d = sh.defs[m.group(1)]
            m2 = re.match(r'^_21\[[^\[\]]+\]\._m0\[([^\[\]]+)\]$', d)
            if m2:
                return WORDS[4 * c0.ev(m2.group(1)) + 'xyzw'.index(m.group(2))]
            kind, wi = c0.vecload(rhs)
            assert kind == 'w'
            return WORDS[wi]
        if re.match(r'^_\d+$', rhs):
            return bias_word(sh.defs[rhs])
        raise Exception(rhs)

    def act_key(act):
        if act in pmap:
            return ('p', pmap[act])
        m = re.match(r'^_26\[[^\]]+\]\._m0\[([^\]]+)\]$', act)
        if m:
            rel = c0.ev(m.group(1)) * 4 - Bbase - 7728
            assert rel in (0, 4, 8, 12), rel
            return ('b', rel // 4)
        kind, wi = c0.vecload(act)
        assert kind == 'm'
        rel = wi - Abase // 4
        dy, rem = divmod(rel, 1928)
        dx, w = divmod(rem, 4)
        assert 0 <= dy < 3 and 0 <= dx < 3, (act, wi, wi * 4)
        return ('a', dy, dx, w)

    tabs = {t: {} for t in tabnames}
    bias = {1: {}, 2: {}, 3: {}}
    W = {1: {}, 2: {}, 3: {}}
    seen = set()
    pre = {}
    i = 0
    while i < len(L):
        l = L[i]
        m = re.match(r'^\s*(_\d+)\[(\d+)u\] = (\d+)u;$', l)
        if m and m.group(1) in tabnames:
            tabs[m.group(1)][int(m.group(2))] = int(m.group(3)); i += 1; continue
        mp = re.match(r'^\s*(_\d+) = (\d+)u;$', l)
        if mp:
            pre[mp.group(1)] = int(mp.group(2)); i += 1; continue
        m = re.match(r'^\s*(_\d+)\[(\d+)u\] = (.+);$', l)
        if m and m.group(1) in accs and accs[m.group(1)] not in seen:
            bias[accs[m.group(1)]][int(m.group(2))] = bias_word(m.group(3)); i += 1; continue
        if l.strip() == 'for (;;)':
            j = i
            while 'continue;' not in L[j]:
                j += 1
            block = L[i:j + 1]
            btxt = '\n'.join(block)
            line = next(x for x in block if 'spvSDot' in x)
            tm = re.match(r'^\s*(_\d+)\[([^\]]+)\] = ', line)
            arr, idx = tm.group(1), tm.group(2)
            assert arr in accs, arr
            km = re.search(r'\+ (\d+)u', idx)
            K = int(km.group(1)) if km else int(re.search(r'uint %s = _\d+ \+ (\d+)u;' % re.escape(idx), btxt).group(1))
            wmap = {m.group(1): (m.group(2), int(m.group(3))) for m in
                    re.finditer(r'^\s*(_\d+) = (_\d+)\[(\d+)u \+ \(_\d+ \* 4u\)\];', btxt, re.M)}
            terms = re.findall(r'spvSDot_uint_uint_uint\(((?:_26\[[^\]]+\]\._m0\[[^\]]+\])|_\d+), (_\d+), 0\)', line)
            assert len(terms) == 4, line
            st = accs[arr]
            seen.add(st)
            for it in range(4):
                d = W[st].setdefault(K + it, {})
                for act, wv in terms:
                    tab, off = wmap[wv]
                    assert tab in tabnames
                    key = act_key(act)
                    assert key not in d
                    # the first iteration's weight word is set before the loop (pre); later ones come from the table
                    d[key] = pre[wv] if (it == 0 and wv in pre) else tabs[tab][4 * it + off]
            i = j + 1
            continue
        i += 1
    assert len(W[1]) == 16 and all(len(d) == 36 for d in W[1].values())
    assert len(W[2]) == 64 and all(len(d) == 8 for d in W[2].values())
    assert len(W[3]) == 32 and all(len(d) == 16 for d in W[3].values())
    for st, nn in ((1, 16), (2, 64), (3, 32)):
        assert sorted(bias[st]) == list(range(nn)), (st, sorted(bias[st]))
    # requant shifts
    pat = r'\+ (\d+)u\) \+ \(\(\S+ >> (\d+)u\) & 1u\)\) >> int\((\d+)u\)'

    def shift_of(var):
        line = next(x for x in L if re.match(r'^\s*uint %s = pack32' % var, x))
        h, a, b = [int(v) for v in re.search(pat, line).groups()]
        assert a == b and h == (1 << (a - 1)) - 1, (h, a, b)
        return a
    s1, s2 = shift_of(packs[0]), shift_of(packs[4])
    # final stores
    stores = [x for x in L if 'roundEven' in x and '= pack32(' in x]
    assert len(stores) == 8
    lim = int(re.search(r'value\[(\d+)u\]', stores[0] if 'value[' in stores[0] else full[full.index('value['):]).group(1))
    fin = []
    for i, line in enumerate(stores):
        ks = [int(k) for k in re.findall(r'float\(int\(%s\[(\d+)u\]\)\)' % re.escape(acc3), line)]
        assert ks == [4 * i + j for j in range(4)], ks
        rv = re.search(r'float\(int\((_\d+)\.x\)\)', line).group(1)
        ex = re.search(r'uvec4 %s = uvec4\(i8vec4\(unpack8\((.+?)\)\)\);' % rv, full).group(1)
        mm = re.match(r'^_26\[[^\]]+\]\._m0\[([^\]]+)\]$', ex)
        mb = re.match(r'^%s\[(\d+)u\]$' % re.escape(buf), ex)
        if not mb and re.match(r'^_\d+$', ex) and re.match(r'^%s\[(\d+)u\]$' % re.escape(buf), sh.defs.get(ex, '')):
            mb = re.match(r'^%s\[(\d+)u\]$' % re.escape(buf), sh.defs[ex])
        widx = buflast[int(mb.group(1))] if mb else (c0.ev(mm.group(1)) if mm else c0.N(ex))
        om = re.search(r'\? \((_\d+) \+ (\d+)u\) : 134217728u', line)
        off = c0.N(om.group(1)) + int(om.group(2))
        fin.append((widx * 4, off))
    for p in range(2):
        for q in range(4):
            assert fin[4 * p + q][1] == fin[4 * p][1] + 4 * q and fin[4 * p][1] % 16 == 0

    def fl(tok):
        if tok.startswith('_'):
            mm = re.search(r'float %s = \(\(gl_GlobalInvocationID.z << 5u\) < 16u\) \? ([\d.e-]+) : ([\d.e-]+);' % tok, full)
            if not mm:  # Ultra Performance: float tok = _B ? a : b;  bool _B = (z << 5u) < 16u  (z = 0 -> a)
                mm = re.search(r'float %s = (_\d+) \? ([\d.e-]+) : ([\d.e-]+);' % tok, full)
                assert mm and re.search(r'bool %s = \(gl_GlobalInvocationID.z << 5u\) < 16u;' % mm.group(1), full), tok
                return Fraction(float(mm.group(2)))
            return Fraction(float(mm.group(1)))
        return Fraction(float(tok))
    sfR = []
    for i, line in enumerate(stores):
        rv0 = re.search(r'float\(int\((_\d+)\.x\)\)', line).group(1)
        m = re.search(r'\(\(float\(int\(%s\.x\)\) \* ([^\s)]+)\) \+ \(float\(int\(%s\[%du\]\)\) \* ([^\s)]+)\)\) \* ([^\s)]+)\)' % (rv0, re.escape(acc3), 4 * i), line)
        cr, ca, mu = [fl(tk) for tk in m.groups()]
        sfrac = ca * mu
        sf = sfrac.denominator.bit_length() - 1
        assert sfrac == Fraction(1, 1 << sf)
        R = cr / ca
        assert R.denominator == 1 and int(R) in (64, 128, 256, 512), R   # 512: Ultra Performance pass 09
        sfR.append((sf, int(R)))
    for p in range(2):
        assert len(set(sfR[4 * p:4 * p + 4])) == 1
    return dict(Abase=Abase, Bbase=Bbase, W=W, bias=bias, s1=s1, s2=s2, sfR=sfR, fin=fin, lim=lim)


# ---------------------------------------------------------------- p23 (4 fused stages)
def comps(text):
    return [m.group(1) for m in re.finditer(r'float\(int\((.+?)\)\) \* 0\.000244140625', text)]


def parse_p23(path, WORDS):
    sh = Shader(path, WORDS, ['_15'], ['_20', '_26'])
    P = {}
    full = '\n'.join(sh.L)
    assert 'registers._m2 + 18u' in full and 'registers._m1 + 11u' in full
    # Ultra Performance: stage 3 is (r/16 + a/4096)*16 and stage 4 (r/32 + a/8192)*32; Performance: stage 3 (r/16 + a/4096)*8
    UPV = full.count('* 16.0') == 4 and full.count('* 0.0001220703125') == 4
    assert full.count('* 0.0625') == 4 and full.count('* %s' % ('16.0' if UPV else '8.0')) == 4
    assert full.count('* 0.03125') == 4 and full.count('* 32.0') == 4
    if UPV:
        P['variant'] = 'ultraperf'
    assert 'value[11u]' in full and '8342464u' in full and '10440128u' in full and '12537792u' in full
    # stage 1: A taps -> S1 (16)
    S1 = []
    for i in range(4):
        c = sh.ctx(_105=i, _66=7728, _67=7728)
        for n in ('_1514', '_1518', '_1522', '_1526'):
            b, terms = c.chain(sh.defs[n])
            d = {}
            for a, w in terms:
                kind, wi = c.vecload(a)
                assert kind == 'm'
                rel = wi - 8342464 // 4
                dy, rem = divmod(rel, 1928)
                dx, wd = divmod(rem, 4)
                key = (dy, dx, wd)
                assert key not in d and 0 <= dy < 3 and 0 <= dx < 3
                d[key] = c.weight(w)
            assert len(d) == 36
            S1.append((s32(b), d))
    assert len(S1) == 16
    # stage 2: [S1 (4 words), B (4 words)] -> S2 (64), relu
    S2 = []
    for i in range(16):
        c = sh.ctx(_1650=i, _66=7728, _67=7728)
        for n in ('_1940', '_1941', '_1942', '_1943'):
            assert 'max(int(' in sh.defs[n]
            b, terms = c.chain(sh.defs[n])
            d = {}
            for a, w in terms:
                m = re.match(r'^_43\[(\d+)u\]$', a)
                if m:
                    key = ('s', int(m.group(1)))
                else:
                    m = re.match(r'^_20\[[^\]]+\]\._m0\[(_\d+)\]$', a)
                    assert m, a
                    wi = c.N(m.group(1))
                    rel = wi * 4 - 10440128 - 7728
                    assert rel in (0, 4, 8, 12)
                    key = ('b', rel // 4)
                assert key not in d
                d[key] = c.weight(w)
            assert len(d) == 8
            S2.append((s32(b), d))
    assert len(S2) == 64
    # stage 3: S2 (16 words) -> S3 (32) + residual
    S3 = []
    line = next(l for l in sh.L if l.strip().startswith('ivec4 _2578'))
    cm = comps(line)
    assert len(cm) == 4
    for i in range(8):
        c = sh.ctx(_1977=i, _66=7728, _67=7728)
        for inner in cm:
            b, terms = c.chain(inner)
            d = {}
            for a, w in terms:
                m = re.match(r'^_48\[(\d+)u\]$', a)
                key = int(m.group(1))
                assert key not in d
                d[key] = c.weight(w)
            assert len(d) == 16
            S3.append((s32(b), d))
    assert len(S3) == 32
    # stage 4: S3 (8 words) -> 2x2 x 16 outputs (the line is _2949 in Performance, _2950 in Ultra Performance)
    S4 = {}
    line = next(l for l in sh.L if re.match(r'^\s*ivec4 _(2949|2950) = clamp\(ivec4\(uvec4\(_\d+ \? uint\(', l))
    cm = comps(line) if not UPV else [m.group(1) for m in re.finditer(r'float\(int\((.+?)\)\) \* 0\.0001220703125', line)]
    assert len(cm) == 4
    for dy in range(2):
        for dx in range(2):
            for k in range(4):
                c = sh.ctx(_2586=dy, _2588=dx, _2608=k, _66=7728, _67=7728, _1973=0, _1974=0)
                for j, inner in enumerate(cm):
                    b, terms = c.chain(inner)
                    d = {}
                    for a, w in terms:
                        m = re.match(r'^_44\[(\d+)u\]$', a)
                        key = int(m.group(1))
                        assert key not in d
                        d[key] = c.weight(w)
                    S4[(dy, dx, 4 * k + j)] = (s32(b), d)
    for v in S4.values():
        assert len(v[1]) == 8, len(v[1])
    P['S1'], P['S2'], P['S3'], P['S4'] = S1, S2, S3, S4
    # stage 4 shares one bias per output channel across the 2x2 sub-pixels
    for dy, dx in itertools.product(range(2), range(2)):
        for oc in range(16):
            assert S4[(dy, dx, oc)][0] == S4[(0, 0, oc)][0]
    return P


# ---------------------------------------------------------------- p15 p17 (bottleneck), p19 (bottleneck + 2x2 transposed conv)
def parse_bottleneck(ps, path, init):
    T = open(path).read()
    P = {'ps': ps}
    P['header'] = T[:T.index('void main()')]
    P['m0'] = int(re.search(r'registers\._m0 \+ (\d+)u', T).group(1))
    P['m1'] = int(re.search(r'registers\._m1 \+ (\d+)u', T).group(1))
    P['IN'] = int(re.search(r'3872u \* gl_GlobalInvocationID\.y\) \+ _188\)\) \+ (\d+)u\) >> 2u', T).group(1))
    P['OUT'] = int(re.search(r'\(_\d+ & 12u\) \| (\d+)u\)', T).group(1))
    assert P['IN'] % 16 == 0 and P['OUT'] % 16 == 0
    ws = re.findall(r'\(\(\((\d+)u \* _\d+\) \+ (\d+)u\) \+ (\d+)u\) >> 2u', T)
    offs = {}
    for st, o, b in ws:
        offs.setdefault((int(st), int(b)), set()).add(int(o))
    # stage 1: stride 16, base b with +512k; stage 2: stride 64 (+0,16,32,48); stage 3: stride 128 (+0..112)
    keys16 = sorted(k for k in offs if k[0] == 16)
    keys64 = sorted(k for k in offs if k[0] == 64)
    keys128 = sorted(k for k in offs if k[0] == 128)
    assert len(keys16) == 1 and len(keys64) == 1 and len(keys128) == 1, (keys16, keys64, keys128)
    assert offs[keys16[0]] == {512 * k for k in range(9)}, offs[keys16[0]]
    assert offs[keys64[0]] == {0, 16, 32, 48} and offs[keys128[0]] == {16 * k for k in range(8)}
    P['W1'], P['W2'], P['W3'] = keys16[0][1], keys64[0][1], keys128[0][1]
    bs = re.findall(r'uint _\d+ = _\d+ \+ (\d+)u;\s+uvec4 _\d+ = uvec4\(_11\[', T)
    assert len(bs) == 3, bs
    P['B1'], P['B2'], P['B3'] = [int(b) * 4 for b in bs]
    # requant shifts: "+ H u) + ((x >> S u) & 1u)) >> int(S u)", stage 1 from the first store, stage 2 from the max() line
    m1 = re.search(r'_127\[\(_154 & 1073741823u\) \+ 0u\] = pack32\(u8vec4\(clamp\(ivec4\(uvec4\(uint\(int\(\(_129\[0u\] \+ (\d+)u\) \+ \(\(_129\[0u\] >> (\d+)u\) & 1u\)\) >> int\((\d+)u\)', T)
    m2 = re.search(r'_128\[\(_1872 & 1073741823u\) \+ 0u\] = pack32\(u8vec4\(clamp\(ivec4\(uvec4\(uint\(int\(\(_2555 \+ (\d+)u\) \+ \(\(_2555 >> (\d+)u\) & 1u\)\) >> int\((\d+)u\)', T)
    assert m1 and m2
    P['s1'], P['s2'] = int(m1.group(3)), int(m2.group(3))
    assert int(m1.group(1)) == (1 << (P['s1'] - 1)) - 1 and int(m2.group(1)) == (1 << (P['s2'] - 1)) - 1
    assert int(m1.group(2)) == P['s1'] and int(m2.group(2)) == P['s2']
    assert 'uint _2555 = uint(max(int(_131[0u]), int(0u)));' in T
    assert '_127[(_154 & 1073741823u) + 0u] = pack32(u8vec4(clamp(ivec4(uvec4(uint(int((_129[0u]' in T
    # stage 3 scales
    sl = re.search(r'float (_\d+) = \(_\d+ < 32u\) \? ([\d.]+) : ([\d.]+);', T)
    var, lo, hi = sl.group(1), float(sl.group(2)), float(sl.group(3))
    st = re.search(r'\? pack32\(u8vec4\(.*', T).group(0)
    assert '0.000244140625' in st
    if re.search(r'\* ' + var + r'\)\)\)', st):  # var is the output scale
        fr = float(re.search(r'float\(int\(_\d+\.x\)\) \* ([\d.]+)\)', st).group(1))
        P['sc'] = (lo, hi); P['fr'] = (fr, fr)
    else:
        sc = float(re.search(r'\* ([\d.]+)\)\)\)\)', st).group(1))
        P['fr'] = (lo, hi); P['sc'] = (sc, sc)
    # per half (oc < 32, oc >= 32): out = rhe((M*r + a) / 2^sh)
    P['sh'], P['M'] = [], []
    for h in range(2):
        fr, sc = P['fr'][h], P['sc'][h]
        ab = sc / 4096.0
        sh = int(round(np.log2(1 / ab)))
        assert abs(ab * (1 << sh) - 1) < 1e-12
        Mf = fr * sc * (1 << sh)
        assert abs(Mf - round(Mf)) < 1e-9 and round(Mf) % 64 == 0
        P['sh'].append(sh); P['M'].append(int(round(Mf)))

    def i8(off, n):
        return list(struct.unpack('<%db' % n, init[off:off + n]))

    def i32(off, n):
        return list(struct.unpack('<%di' % n, init[off:off + 4 * n]))
    # w1[tap][oc][16]; oc 0..31 (group = oc >> 4)
    P['w1'] = [[i8(P['W1'] + 512 * t + 16 * oc, 16) for oc in range(32)] for t in range(9)]
    P['w2'] = [i8(P['W2'] + 64 * oc, 64) for oc in range(128)]
    P['w3'] = [i8(P['W3'] + 128 * oc, 128) for oc in range(64)]
    P['b1'], P['b2'], P['b3'] = i32(P['B1'], 32), i32(P['B2'], 128), i32(P['B3'], 64)
    return P


def parse_p19(path, init):
    T = open(path).read()
    P = {'ps': '19', 'final': True}
    P['header'] = T[:T.index('void main()')]
    P['m0'] = 0
    assert 'registers._m1 + 11u' in T and 'registers._m2 + 18u' in T
    P['tdesc'] = 'registers._m1 + 11u'
    P['IN'] = int(re.search(r'\(_220 \+ (\d+)u\) >> 2u', T).group(1))
    assert P['IN'] == 12537792
    ws = re.findall(r'\(\(\((\d+)u \* _\d+\) \+ (\d+)u\) \+ (\d+)u\) >> 2u', T)
    offs = {}
    for st, o, b in ws:
        offs.setdefault((int(st), int(b)), set()).add(int(o))
    assert sorted(offs) == [(16, 76800), (64, 81536), (128, 90240)], sorted(offs)
    assert offs[(16, 76800)] == {512 * k + h for k in range(9) for h in (0, 8)}
    assert offs[(64, 81536)] == {8 * k for k in range(8)} and offs[(128, 90240)] == {8 * k for k in range(16)}
    P['W1'], P['W2'], P['W3'] = 76800, 81536, 90240
    bs = re.findall(r'uint _\d+ = _\d+ \+ (\d+)u;\s+uvec4 _\d+ = uvec4\(_12\[', T)
    assert [int(b) for b in bs] == [20352, 22432, 24608, 26720], bs
    P['B1'], P['B2'], P['B3'], P['B4'] = [int(b) * 4 for b in bs]
    P['W4'] = 98688
    assert '(_4916 + 98688u)' in T and '(_4916 + 98696u)' in T and '(4096u * _4841) + (_4846 << 11u)' in T
    assert '(_4902 | 8342464u)' in T and '(_4902 | 14659648u)' in T and '2097664u' in T and '7712u' in T
    P['RES'], P['OUT2'] = 8342464, 14659648
    assert '_2143 + 127u) + ((_2143 >> 8u) & 1u)) >> int(8u)' in T
    assert '_3071 + 63u) + ((_3071 >> 7u) & 1u)) >> int(7u)' in T
    assert 'uint _3071 = uint(max(int(' in T
    assert '* 0.0625) + (float(int((' in T and '* 0.00048828125)) * 16.0)' in T
    assert '* 0.03125) + (_5690 * 0.0001220703125)) * 16.0' in T
    P['s1'], P['s2'] = 8, 7
    P['ab3'] = 2.0 ** -11
    P['fr'], P['sc'], P['sh'], P['M'] = (0.0625, 0.0625), (16.0, 16.0), [7, 7], [128, 128]
    i8 = lambda off, n: list(struct.unpack('<%db' % n, init[off:off + n]))
    i32 = lambda off, n: list(struct.unpack('<%di' % n, init[off:off + 4 * n]))
    P['w1'] = [[i8(P['W1'] + 512 * t + 16 * oc, 16) for oc in range(32)] for t in range(9)]
    P['w2'] = [i8(P['W2'] + 64 * oc, 64) for oc in range(128)]
    P['w3'] = [i8(P['W3'] + 128 * oc, 128) for oc in range(64)]
    P['w4'] = [[i8(P['W4'] + 2048 * sp + 64 * oc, 64) for oc in range(32)] for sp in range(4)]
    P['b1'], P['b2'], P['b3'], P['b4'] = i32(P['B1'], 32), i32(P['B2'], 128), i32(P['B3'], 64), i32(P['B4'], 32)
    P['K4'] = [[P['b4'][oc] - 128 * sum(P['w4'][sp][oc]) - 128 * 256 + (128 << 9) + (1 << 8) for oc in range(32)] for sp in range(4)]
    return P


# ---------------------------------------------------------------- main
def main():
    preset, gd, ib, od = sys.argv[1:5]
    assert preset in ('performance', 'ultraperf'), preset
    init = open(ib, 'rb').read()
    WORDS = struct.unpack('<%dI' % (len(init) // 4), init)
    o = lambda n: '%s/o/p%s.comp' % (gd, n)
    q = lambda n: '%s/t/q%s.comp' % (gd, n)
    out = {}
    out['p03'] = parse_res(o('03'))
    out['p05'] = parse_res(o('05'))
    out['p25'] = parse_res(o('25'))
    out['p07'] = parse_p07(o('07'), WORDS)
    for n in ('09', '11', '21'):
        out['p' + n] = parse_h09(o(n), WORDS)
    out['p15'] = parse_bottleneck('15', q('15'), init)
    out['p17'] = parse_bottleneck('17', q('17'), init)
    out['p19'] = parse_p19(q('19'), init)
    out['p23'] = parse_p23(o('23'), WORDS)
    os.makedirs(od, exist_ok=True)
    with open(od + '/params_n1.pkl', 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print('params_n1.pkl:', ' '.join(out))
    if preset == 'ultraperf':
        sh = p07_sh(o('07'))
        np.save(od + '/p07_sh.npy', sh)
        print('p07_sh.npy:', sh.tolist())


if __name__ == '__main__':
    main()
