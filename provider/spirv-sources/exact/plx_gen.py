#!/usr/bin/env python3
# Exact-prepass variants of the shipped FSR 4.1.1 prepass "pl" (vela_assets/fsr411/general/1080/*/pass-01.spv,
# FNV-1a64 0x4d243005c298aaf8), made as word splices of pl's SPIR-V. The provider applies the same splices at load time
# (overlay/ov/vela-baked-pre.hpp, vela_exact_pre) and takes the result only if its FNV-1a64 equals the value printed here.
#
#   x (plx):  X0 flag, the convention of the Fast prepass f1 (fp_gen.py "x0f"): the X0 store (scratch SSBO, the GPU network's input)
#             runs only when bit 0 of the HTP-address push word (push word 4) is 0; the guard and the pointer of the HTP-input
#             store use the address with bit 0 cleared. Bit 0 clear = pl, instruction for instruction.
#   w (plw):  gl_WorkGroupID.y is read as 2 * y (the four loads), so no workgroup returns at the parity test; the host
#             dispatches half the group rows (WG study, 2026-11-02). Same threads, same pixels, same math.
#   xw (plxw): both.
#
# usage: plx_gen.py <pl pass-01.spv> <out dir> [<vela-baked-pre.hpp to rewrite between the markers>]
# writes <out>/pl.spv plx.spv plw.spv plxw.spv, <out>/exact_tables.inc and prints the hashes.
import struct, sys, os, hashlib

OP = dict(TypeBool=20, TypeInt=21, TypeVector=23, Constant=43, Load=61, Store=62, AccessChain=65, Decorate=71,
          CompositeExtract=81, CompositeInsert=82, Bitcast=124, IAdd=128, ISub=130, IEqual=170, INotEqual=171,
          ShiftLeftLogical=196, BitwiseOr=197, BitwiseXor=198, BitwiseAnd=199, SelectionMerge=247, Label=248,
          Branch=249, BranchConditional=250)
PL_FNV = 0x4d243005c298aaf8


def fnv(b):
    h = 0xcbf29ce484222325
    for x in b:
        h = ((h ^ x) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def ins(op, *w):
    return [((len(w) + 1) << 16) | OP[op], *w]


def parse(words):
    out = []; p = 5
    while p < len(words):
        n = words[p] >> 16
        assert n > 0 and p + n <= len(words)
        out.append((p, words[p] & 0xFFFF, words[p:p + n])); p += n
    return out


def build(words, want_x, want_w):
    I = parse(words); bound = words[3]; sp = []   # sp: (at, del, [words]) on pl's word offsets
    def new():
        nonlocal bound
        bound += 1; return bound - 1
    def const_u(v):
        tu = next(w[1] for _, o, w in I if o == OP['TypeInt'] and w[2] == 32 and w[3] == 0)
        c = [w[2] for _, o, w in I if o == OP['Constant'] and w[1] == tu and len(w) == 4 and w[3] == v]
        assert c, v
        return c[0]
    if want_x:
        seq = [OP[k] for k in ('Store', 'ShiftLeftLogical', 'IAdd', 'ISub', 'BitwiseXor', 'AccessChain', 'Load', 'CompositeExtract',
                               'CompositeExtract', 'BitwiseOr', 'INotEqual', 'SelectionMerge', 'BranchConditional', 'Label',
                               'AccessChain', 'Load', 'Bitcast', 'AccessChain', 'Store', 'Branch', 'Label')]
        sites = [i for i in range(len(I) - len(seq)) if [I[i + k][1] for k in range(len(seq))] == seq]
        assert len(sites) == 2, sites
        u1 = const_u(1); u0c = const_u(0); mask = const_u(0xFFFFFFFE)
        for i in sites:
            ac, ld, g0, g1, bor, ne = (I[i + k][2] for k in (5, 6, 7, 8, 9, 10))
            ld2, bc = I[i + 15][2], I[i + 16][2]
            t_ptr, reg, idx3 = ac[1], ac[3], ac[4]; t_v2, t_u, t_b, u0 = ld[1], g0[1], ne[1], ne[4]
            assert u0 == u0c and g0[3] == ld[2] and g0[4] == 0 and g1[4] == 1 and bor[3] == g0[2] and bor[4] == g1[2]
            assert I[i + 14][2][3] == reg and I[i + 14][2][4] == idx3 and bc[3] == ld2[2] and ld2[1] == t_v2
            n = [new() for _ in range(11)]
            sp.append((I[i][0], 0, ins('AccessChain', t_ptr, n[0], reg, idx3) + ins('Load', t_v2, n[1], n[0])
                       + ins('CompositeExtract', t_u, n[2], n[1], 0) + ins('BitwiseAnd', t_u, n[3], n[2], u1)
                       + ins('IEqual', t_b, n[4], n[3], u0) + ins('SelectionMerge', n[6], 0)
                       + ins('BranchConditional', n[4], n[5], n[6]) + ins('Label', n[5])))
            sp.append((I[i + 1][0], 0, ins('Branch', n[6]) + ins('Label', n[6])))               # after the X0 store
            sp.append((I[i + 9][0], 0, ins('BitwiseAnd', t_u, n[7], g0[2], mask)))              # guard: (x & ~1) | y
            sp.append((I[i + 9][0] + 3, 1, [n[7]]))
            sp.append((I[i + 16][0], 0, ins('CompositeExtract', t_u, n[8], ld2[2], 0) + ins('BitwiseAnd', t_u, n[9], n[8], mask)
                       + ins('CompositeInsert', t_v2, n[10], n[9], ld2[2], 0)))                   # pointer from (x & ~1, y)
            sp.append((I[i + 16][0] + 3, 1, [n[10]]))
    if want_w:
        var = [w[1] for _, o, w in I if o == OP['Decorate'] and len(w) == 4 and w[2] == 11 and w[3] == 26]   # BuiltIn WorkgroupId
        assert len(var) == 1
        full = [w for _, o, w in I if o == OP['Load'] and w[3] == var[0]]
        assert not full, 'whole-vector load of gl_WorkGroupID'
        u1 = const_u(1); k = 0
        for j, (p, o, w) in enumerate(I):
            if o == OP['AccessChain'] and w[3] == var[0]:
                assert len(w) == 5
                if w[4] != u1:
                    continue
                p2, o2, w2 = I[j + 1]
                assert o2 == OP['Load'] and w2[3] == w[2]
                uses = [x for _, ox, x in I if ox == OP['Load'] and x[3] == w[2]]
                assert len(uses) == 1
                t = new()
                sp.append((p2 + 2, 1, [t])); sp.append((p2 + 4, 0, ins('IAdd', w2[1], w2[2], t, t))); k += 1
        assert k == 4, k
    if sp:
        sp.append((3, 1, [bound]))
    sp.sort()
    return sp


def apply(words, sp):
    out = []; pos = 0
    for at, d, w in sp:
        assert at >= pos
        out += words[pos:at]; out += w; pos = at + d
    return out + words[pos:]


def main():
    src, outdir = sys.argv[1], sys.argv[2]
    raw = open(src, 'rb').read(); words = list(struct.unpack('<%dI' % (len(raw) // 4), raw))
    assert fnv(raw) == PL_FNV, 'input is not pl: %016x' % fnv(raw)
    os.makedirs(outdir, exist_ok=True)
    open(outdir + '/pl.spv', 'wb').write(raw)
    lines = ['// >>> exact-pre tables: generated by spirv-sources/exact/plx_gen.py from pl (FNV-1a64 0x4d243005c298aaf8); flat splices {at, delete, n, n words}']
    for name, x, w in (('plx', 1, 0), ('plw', 0, 1), ('plxw', 1, 1)):
        sp = build(words, x, w); res = apply(words, sp)
        b = struct.pack('<%dI' % len(res), *res); open('%s/%s.spv' % (outdir, name), 'wb').write(b)
        flat = []
        for at, d, ws in sp:
            flat += [at, d, len(ws)] + ws
        print('%s words %d -> %d bound %d -> %d splices %d table words %d fnv %016x md5 %s' % (
            name, len(words), len(res), words[3], res[3], len(sp), len(flat), fnv(b), hashlib.md5(b).hexdigest()))
        cn = {'plx': 'kVelaPlx', 'plw': 'kVelaPlw', 'plxw': 'kVelaPlxw'}[name]
        lines.append('inline constexpr uint64_t %sFnv = 0x%016xull;' % (cn, fnv(b)))
        body = ','.join('%uu' % v if v < 65536 else '0x%xu' % v for v in flat)
        rows = []; cur = ''
        for tok in body.split(','):
            if len(cur) + len(tok) > 150:
                rows.append(cur); cur = ''
            cur += tok + ','
        rows.append(cur.rstrip(','))
        lines.append('inline constexpr uint32_t %s[] = {\n    %s};' % (cn, '\n    '.join(rows)))
    lines.append('// <<< exact-pre tables')
    inc = '\n'.join(lines) + '\n'
    open(outdir + '/exact_tables.inc', 'w', newline='\n').write(inc)
    if len(sys.argv) > 3:
        h = open(sys.argv[3], encoding='utf-8').read()
        a = h.index('// >>> exact-pre tables'); e = h.index('// <<< exact-pre tables'); e = h.index('\n', e) + 1
        open(sys.argv[3], 'w', encoding='utf-8', newline='\n').write(h[:a] + inc + h[e:])
        print('rewrote', sys.argv[3])


main()
