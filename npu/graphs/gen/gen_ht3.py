#!/usr/bin/env python3
"""HT3 idea 4 (graph level): fewer residual copies in the exact full FSR 4.1.1 graph (base = HT ht_t2s7p13 = T2+P07S+P13S).
Residual R*y (R = 64/128/256) is folded into the 1x1 'f' / tail convs as P copies of y with int8 weight pieces <= 127
(R=128 -> 2 copies, R=256 -> 3). int8 also has -128: if a tensor is stored COMPLEMENTED, y' = ~y = -y - 1 (a bijection of
[-128,127], so the int8 clamp commutes: ~clamp(A+B) = clamp(-(A+B) - 1)), then R*y = sum of (-128)*y' pieces - R:
R=128 -> 1 copy, R=256 -> 2 copies. Exact integer identity for every consumer:
  w*y = (-w)*y' - w  (per physical column and tap; the '-w' goes onto the bias-constant channels at the SAME tap, which are
  zero-padded exactly where y' is, so 3x3 borders stay exact); the producer's grouped sum conv uses weights -1 -1, bias -1
  (scale 1, ratio 1 = exact). Layout ops (S2D, identity ConvTranspose/convs, Slice) commute with ~.
Flags (with gen_ht flags): NEG  = complement X2 D0 E1 E2 F0 F1 F2 G0 G1 (every residual / skip input with R >= 128)
                           NEGT = complement only X2, E2 (the two tail skips: p19t 3 -> 2 copies, p23t 2 -> 1)
                           NEGB = X2 D0 E1 E2 F0 (NEG minus the tensors whose 3x3 consumer has a -128 weight)
Debug (_d) pass outputs of complemented tensors are complemented too (cmp3.py undoes it).
Usage (docker h-qnn): VARIANTS='name:flags:dbg ...' gen_ht3.py OUTDIR"""
import sys
NEGALL = {'p05f', 'p07', 'p09f', 'p11f', 'p13', 'p15f', 'p17f', 'p19t', 'p21f'}
NEGT = {'p05f', 'p11f'}
NEGB = {'p05f', 'p07', 'p09f', 'p11f', 'p13'}   # no -128 dup on any 3x3 c3 (only on 1x1 e: 34 -> 50 ch, same 64 tile)
NEGNAMES = {'NEG': ['X2', 'D0', 'E1', 'E2', 'F0', 'F1', 'F2', 'G0', 'G1'], 'NEGT': ['X2', 'E2']}


def rep(s, a, b, n=1):
    assert s.count(a) == n, (a[:70], s.count(a)); return s.replace(a, b)


def ht3patch(src):
    # int8 has no +128: a logical weight -128 on a complemented input would become +128 -> -127 on y' plus -1 on a duplicate input
    src = rep(src, "        o = W.shape[0]; b = np.asarray(b, np.int64); sh = np.broadcast_to(np.asarray(sh, np.int64), (o,)).copy()",
              """        ci_ = 0; dups = []; W = np.asarray(W, np.int64).copy()
        for t in [t for t in ins if t.C]:
            if t.neg and (W[:, ci_:ci_ + t.C] == -128).any():
                m_ = W[:, ci_:ci_ + t.C] == -128; Wd_ = np.zeros_like(W[:, ci_:ci_ + t.C]); Wd_[m_] = -1; W[:, ci_:ci_ + t.C][m_] = -127; dups.append((t, Wd_))
            ci_ += t.C
        if dups:
            kin = [t for t in ins if not t.C]; ins = [t for t in ins if t.C] + [d[0] for d in dups] + kin
            W = np.concatenate([W] + [d[1] for d in dups], 1); DUPLOG.append((tag, len(dups), int(sum(int(d[1].any(axis=tuple(range(2, d[1].ndim))).any(0).sum()) for d in dups))))
        o = W.shape[0]; b = np.asarray(b, np.int64); sh = np.broadcast_to(np.asarray(sh, np.int64), (o,)).copy()""")
    src = rep(src, "s.name, s.P, s.C, s.k = name, P, C, k; s._idx = idx", "s.name, s.P, s.C, s.k = name, P, C, k; s._idx = idx; s.neg = False")
    old_b = '''            if bv is not None:
                kt = [(t, off) for t, off in zip(ins, offs) if t.k][0]; r = bv.copy(); co = np.zeros((o, K), np.int64)
                for j in range(2, K): co[:, j] = np.clip(np.fix(r / 127.0), -127, 127); r = r - 127 * co[:, j]
                co[:, 1] = np.round(r / 64.0); co[:, 0] = r - 64 * co[:, 1]
                assert np.abs(co).max() <= 127 and ((co * np.array(VALS)).sum(1) == bv).all(), tag
                for po in range(Po): Wp[po * o:(po + 1) * o, kt[1] + kt[0].kb:kt[1] + kt[0].kb + K, ctap[0], ctap[1]] = co
            return Wp'''
    new_b = '''            negc = [j for t, off in zip(ins, offs) if getattr(t, 'neg', False) for j in range(off, off + t.n)]
            corr = np.zeros((Po * o,) + ks, np.int64)
            if negc: corr = -Wp[:, negc].sum(1); Wp[:, negc] *= -1
            assert bv is not None or not negc, tag
            if bv is not None:
                kt = [(t, off) for t, off in zip(ins, offs) if t.k][0]; kb = kt[1] + kt[0].kb
                for ky in range(ks[0]):
                    for kx in range(ks[1]):
                        r0 = corr[:, ky, kx] + (np.tile(bv, Po) if (ky, kx) == tuple(ctap) else 0)
                        if not np.any(r0): continue
                        r = r0.copy(); co = np.zeros((Po * o, K), np.int64)
                        for j in range(2, K): co[:, j] = np.clip(np.fix(r / 127.0), -127, 127); r = r - 127 * co[:, j]
                        co[:, 1] = np.round(r / 64.0); co[:, 0] = r - 64 * co[:, 1]
                        assert np.abs(co).max() <= 127 and ((co * np.array(VALS)).sum(1) == r0).all(), (tag, ky, kx)
                        Wp[:, kb:kb + K, ky, kx] = co
            return Wp'''
    src = rep(src, old_b, new_b)
    src = rep(src, '''            Ws = np.ones((M + nk, 2, 1, 1), np.int64); Ws[M:, 1] = 0
            y = s.conv_p(D, Ws, np.zeros(M + nk, np.int64), (1, 1), (1, 1), 0, np.zeros(M + nk, np.int64), tag + 'S', group=M + nk)
            return A(y, Po, o, keepk)''', '''            Ws = np.ones((M + nk, 2, 1, 1), np.int64); Ws[M:, 1] = 0; bs_ = np.zeros(M + nk, np.int64); ng_ = tag in NEGOUT
            if ng_: Ws[:M] = -1; bs_[:M] = -1
            y = s.conv_p(D, Ws, bs_, (1, 1), (1, 1), 0, np.zeros(M + nk, np.int64), tag + 'S', group=M + nk)
            r_ = A(y, Po, o, keepk); r_.neg = ng_; return r_''')
    src = rep(src, "yhi = A(g.slice(y.name, lo, 2 * lo, tag + '_hi'), 1, lo, False)", "yhi = A(g.slice(y.name, lo, 2 * lo, tag + '_hi'), 1, lo, False); ylo.neg = yhi.neg = y.neg")
    src = rep(src, "Wf, P = fold(v['w3'], v['R'])", "Wf, P = foldn(v['w3'], v['R'], y.neg)", 2)
    src = rep(src, "Wt, P = fold(v['w4'], np.full(C4, v['t_R']))", "Wt, P = foldn(v['w4'], np.full(C4, v['t_R']), sd.neg)")
    src = rep(src, "sd = A(s2, 1, C4, skip.k, idx=lambda p, c, nn=nn, Cs=Cs: (c // Cs) * nn + c % Cs)", "sd = A(s2, 1, C4, skip.k, idx=lambda p, c, nn=nn, Cs=Cs: (c // Cs) * nn + c % Cs); sd.neg = skip.neg")
    a = "(2, 1), (2, 1), 0, np.zeros(C4, np.int64), tag + '_s2d'), 1, C4, False)"; src = rep(src, a, a + "; sd.neg = skip.neg")
    src = rep(src, "    if tag.startswith('p19'): return g.kconst(A(d, 1, Cs, False), (270, 480))",
              "    if tag.startswith('p19'): _a = A(d, 1, Cs, False); _a.neg = u.neg; return g.kconst(_a, (270, 480))")
    src = rep(src, "    return A(convT(g, u.name, W, (2, 1), (2, 1), tag + '_ct'), 2, Cs, False)",
              "    assert not u.neg; return A(convT(g, u.name, W, (2, 1), (2, 1), tag + '_ct'), 2, Cs, False)")
    src = rep(src, "S = A(s2dX2(g, X2), 2, 64, False, idx=lambda q, c2: (2 * (c2 // 32) + q) * 32 + c2 % 32)",
              "S = A(s2dX2(g, X2), 2, 64, False, idx=lambda q, c2: (2 * (c2 // 32) + q) * 32 + c2 % 32); S.neg = X2.neg")
    src = rep(src, "        return A(convT(g, Dp.name, Wu, (1, 2), (1, 2), 'p07_u'), 1, o, False)",
              "        _a = A(convT(g, Dp.name, Wu, (1, 2), (1, 2), 'p07_u'), 1, o, False); _a.neg = Dp.neg; return _a")
    src = rep(src, "    try: return g.L2([A(s2dE2(g, E2), 1, 4 * C, False)], Wl, v['b'], '1x1', v['sh'], 'p13')",
              "    try: _s = A(s2dE2(g, E2), 1, 4 * C, False); _s.neg = E2.neg; return g.L2([_s], Wl, v['b'], '1x1', v['sh'], 'p13')")
    # TB (lead lead 1): HTP's own int32 bias (symmetric, per-channel scale = in x w scale, offset 0) on the TBSET doubled convs, no const channels
    src = rep(src, "        if s.ex and CBCAT and not any(t.k for t in ins):", "        keepk = keepk and tag not in TBSET\n        if s.ex and CBCAT and not any(t.k for t in ins) and tag not in TBSET:")
    src = rep(src, "        WA, WB = build(bA), build(bB); M = Po * o; sh1 = np.tile(sh + 1, Po)",
              "        TB_ = tag in TBSET; WA, WB = (build(None), build(None)) if TB_ else (build(bA), build(bB)); M = Po * o; sh1 = np.tile(sh + 1, Po)")
    src = rep(src, "            D = s.conv_p(x, Wd, np.zeros(Wd.shape[0], np.int64), ks, st, pd, shd, tag + 'D', relu)",
              "            bd_ = np.zeros(Wd.shape[0], np.int64)\n            if TB_: bd_[0:2 * M:2] = np.tile(bA, Po); bd_[1:2 * M:2] = np.tile(bB, Po)\n            D = s.conv_p(x, Wd, bd_, ks, st, pd, shd, tag + 'D', relu)")
    src += r'''
NEGOUT = set(); DUPLOG = []; TBSET = set()
def foldn(w3, R, neg):   # complemented y': pieces of 128 (stored as -128 after the column negation in L2)
    if not neg: return fold(w3, R)
    pc = lambda r: [128] * (int(r) // 128) + ([int(r) % 128] if int(r) % 128 else [])
    C, I3 = w3.shape[:2]; P = max(len(pc(r)) for r in R); Wf = np.zeros((C, I3 + P * C, 1, 1), np.int64); Wf[:, :I3] = w3
    for o in range(C):
        for j, pv in enumerate(pc(R[o])): Wf[o, I3 + j * C + o, 0, 0] = pv
    return Wf, P
'''
    return src


ht = open('/ht/gen_ht.py').read()
ht = rep(ht, "sys.argv = ['gen_nf2', OD]; ns = {", "src = ht3patch(src)\nsys.argv = ['gen_nf2', OD]; ns = {")
ht = rep(ht, "    m_ = ns['build2'](name, 'exact', 1, int(dbg))",
         "    ns['TBSET'].clear(); ns['TBSET'].update({'p03c3'} if 'TB' in ns['FL'] else ());\n    ns['NEGOUT'].clear(); ns['NEGOUT'].update(NEGALL if 'NEG' in ns['FL'] else NEGB if 'NEGB' in ns['FL'] else NEGT if 'NEGT' in ns['FL'] else ())\n    m_ = ns['build2'](name, 'exact', 1, int(dbg)); print('dup inputs (tag, n, channels with -128):', ns['DUPLOG']); ns['DUPLOG'].clear()")
ht = ht.replace("__file__': '/nf/gen_nf2.py'}", "__file__': '/nf/gen_nf2.py', 'ht3patch': ht3patch}")
exec(compile(ht, 'gen_ht_ht3', 'exec'), {'__name__': 'ht3', 'ht3patch': ht3patch, 'NEGALL': NEGALL, 'NEGT': NEGT, 'NEGB': NEGB, 'rep': rep})

