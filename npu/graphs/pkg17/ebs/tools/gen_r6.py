#!/usr/bin/env python3
"""r6 E-BS (2026-11-23): gen_bp.py + relu bias split through the output zero point (env ZPO_<tag>D_o=z, flag Z<tag> on an e layer). Without those env vars and flags it is gen_bp.py.
BP (2026-11-01): V11 gen_v11.py + BQ gen_bq.py move flag Z<tag>@<d>k<k>@.. = BS on <tag> with D channel d moved by k*2^t (q_d -= k, r_d += k 2^t).
V11 (2026-11-01, from PK2 gen_pk2.py, unchanged patches): bias split (BS) per layer (flags Z<tag>) for EVERY graph sysext v10 ships.
KIND (env) picks the shipped builder chain the graph was made with, read-only:
  k1h : /k1h/gen_k1h.py (K1H = nf4_k1h with /b performance, /nf NF, /k1h K1H; UP3 = nf4up_k1h with /b ultraperf, /nf UP3/nf, /k1h UP3/k1h)
  n7  : /gn/gen_n7.py   (N7 = nf4_k1h_360, UP4a/u7 = nf4up_k1h_360)  size chain 540->360 ... then the same gen_k1h text
  n9  : /gn/gen_n9.py   (N9 = nf4_k1h_450, UP4a/u9 = nf4up_k1h_450)  size chain 540->452 + pad rows
  ht  : /ht/gen_ht.py   (HT ht_t2s7p13 = nf4_exact, no K1)
In every chain the source text `src` (gen_nf2 + gen_ht) gets HT3 ht3patch + PK2 pkpatch exactly as PK2 did (both are no-ops without
NEGT / BS flags: the unchanged rebuild must equal the shipped md5, which build_v11.sh checks).
Usage (docker h-qnn): KIND=.. VARIANTS='name:flags:dbg ...' INIT_BIN=/b/initializers.bin gen_v11.py OUTDIR"""
import sys, os, re, textwrap, numpy as np
OD = sys.argv[1]; os.makedirs(OD, exist_ok=True)
h3 = open('/ht3/gen_ht3.py').read().split("\nht = open('/ht/gen_ht.py')")[0]
KIND = os.environ.get('KIND', 'k1h')
if KIND in ('n7', 'n9'):   # V11: gen_n7/gen_n9 size gen_nf/gen_nf2/gen_ht (540->360 ...); size HT3's patch text the same way (anchors hold 270, 480)
    SUB_ = eval(re.search(r'^SUB = (\[.*\])$', open('/gn/gen_%s.py' % KIND).read(), re.M).group(1))
    for a_, b_ in SUB_: h3 = re.sub(r'(?<![0-9])%s(?![0-9])' % a_, b_, h3)
g3 = {'__name__': 'pk3'}; exec(compile(h3, 'gen_ht3_defs', 'exec'), g3); ht3patch, rep = g3['ht3patch'], g3['rep']


def pkpatch(src):
    src = rep(src, "        keepk = keepk and tag not in TBSET\n", "        keepk = keepk and tag not in TBSET and tag not in BSSET\n")
    src = rep(src, "and not any(t.k for t in ins) and tag not in TBSET:", "and not any(t.k for t in ins) and tag not in TBSET and tag not in BSSET:")
    src = rep(src, "            assert bv is not None or not negc, tag\n",
              "            assert bv is not None or not negc or ks == (1, 1), tag\n            if bv is None and negc: s._corr = corr[:, 0, 0]\n")
    src = rep(src, "        TB_ = tag in TBSET; WA, WB = (build(None), build(None)) if TB_ else",
              "        TB_ = tag in TBSET; BS_ = tag in BSSET; s._corr = 0; WA, WB = (build(None), build(None)) if (TB_ or BS_) else")
    src = rep(src, "            D = s.conv_p(x, Wd, bd_, ks, st, pd, shd, tag + 'D', relu)\n", """            BQ_ = None
            if BS_:
                assert CBCAT and not keepk and not any(t.k for t in ins), tag
                Mr = (1 << sh1).astype(np.int64); bfA = np.tile(bA, Po) + s._corr; bfB = np.tile(bB, Po) + s._corr
                qA = np.floor(bfA / Mr + 0.5).astype(np.int64); qB = np.floor(bfB / Mr + 0.5).astype(np.int64)
                if relu: qA = qA * 0 + int(os.environ.get('ZPO_' + tag + 'D_o', 0)); qB = qB * 0 + int(os.environ.get('ZPO_' + tag + 'D_o', 0))   # r6 E-BS: relu lower clamp = output zero point z, q = z on both views, S bias 2z
                for d_, k_ in BQMOVE.get(tag, ()):   # BQ: move D channel d_ by k_*2^t (q -= k_, r += k_ 2^t), exact identity
                    (qA if d_ % 2 == 0 else qB)[d_ // 2] -= k_
                rA, rB = bfA - qA * Mr, bfB - qB * Mr; bd_[0:2 * M:2] = rA; bd_[1:2 * M:2] = rB; BQ_ = qA + qB
                assert relu or np.abs(BQ_).max() <= 60, (tag, BQ_)
                BSLOG.append('%-6s relu %d t %s max|r|/2^t %.3f max|q| %d' % (tag, relu, sorted(set(sh1.tolist())), float(np.max(np.abs(np.r_[rA, rB]) / np.r_[Mr, Mr])), int(np.abs(np.r_[qA, qB]).max())))
                if BSBAD: BQ_ = BQ_ * 0
                BSARR[tag] = dict(rA=rA, rB=rB, qA=qA, qB=qB, t=sh1, swA=WA.sum((1, 2, 3)), swB=WB.sum((1, 2, 3)), relu=int(relu), corr=np.asarray(s._corr) + 0 * rA)
            D = s.conv_p(x, Wd, bd_, ks, st, pd, shd, tag + 'D', relu)
""")
    src = rep(src, "            if ng_: Ws[:M] = -1; bs_[:M] = -1\n",
              "            if ng_: Ws[:M] = -1; bs_[:M] = 0 if NEGBAD else -1\n            if BQ_ is not None: bs_[:M] += -BQ_ if ng_ else BQ_\n")
    src += "\nBSSET = set(); BSLOG = []; BSARR = {}; NEGBAD = False; BSBAD = False\n"
    return src


class AllBut:   # set-like: every tag except the given ones (and relu-only filter applied by the caller via names)
    def __init__(s, ex=()): s.ex = set(ex)
    def __contains__(s, t): return t not in s.ex


LOOPA = "    m_ = ns['build_k1only'](name) if 'K1ONLY' in FL else ns['build2'](name, 'exact', 1, int(dbg)); mac, byt, ops = ns['stats'](m_)"
LOOPB = """    ns['BQMOVE'] = {f.split('@')[0][1:]: [tuple(map(int, x.split('k'))) for x in f.split('@')[1:]] for f in FL if f[0] == 'Z' and '@' in f}
    FL = set(f.split('@')[0] for f in FL)
    ns['NEGOUT'].clear(); ns['NEGOUT'].update(NEGT_ if 'NEGT' in FL else ()); ns['NEGBAD'] = 'NEGBAD' in FL; ns['BSBAD'] = 'BSBAD' in FL
    ns['BSSET'] = AllBut() if 'BS' in FL else ({'p03c3'} if 'BS1' in FL else (bsgroups(FL) if (FL & {'BSC3', 'BSF', 'BSD'} or any(f[0] == 'Z' for f in FL)) else (AllBut(RELU_E) if 'BSN' in FL else set()))); ns['BSLOG'].clear(); ns['DUPLOG'].clear()
    m_ = ns['build_k1only'](name) if 'K1ONLY' in FL else ns['build2'](name, 'exact', 1, int(dbg)); mac, byt, ops = ns['stats'](m_)
    print('PKLOG', name, 'dup', ns['DUPLOG']); [print('PKLOG', name, 'BS', l) for l in ns['BSLOG']]
    if os.environ.get('BSDUMP'): np.savez(OD + '/bsarr_%s.npz' % name, **{k + '__' + f: v for k, d in ns['BSARR'].items() for f, v in d.items()}); ns['BSARR'].clear()"""
LI = (3, 5, 9, 11, 15, 17, 19, 21, 23, 25, 27)
def bsgroups(FL):   # BS on groups of view convs: BSC3 = the 3x3 'c3' convs, BSF = the 1x1 'f' (residual) convs, BSD = p07, p13, p19t, p23t
    s = set()
    if 'BSC3' in FL: s |= {'p%02dc3' % i for i in LI}
    if 'BSF' in FL: s |= {'p%02df' % i for i in LI}
    if 'BSD' in FL: s |= {'p07', 'p13', 'p19t', 'p23t'}
    s |= {f[1:] for f in FL if f[0] == 'Z'}
    return s
RELU_E = ['p%02de' % i for i in (3, 5, 9, 11, 15, 17, 19, 21, 23, 25, 27)]
G = {'__name__': 'pk', 'ht3patch': ht3patch, 'pkpatch': pkpatch, 'rep': rep, 'AllBut': AllBut, 'NEGT_': g3['NEGT'], 'RELU_E': RELU_E, 'bsgroups': bsgroups}
if KIND == 'ht':
    ht = open('/ht/gen_ht.py').read(); pre = ht.split("sys.argv = ['gen_nf2', OD]; ns =")[0]
    sys.argv = ['gen_ht', OD]; g0 = {'__name__': 'v11htpre'}; exec(compile(pre, 'gen_ht_pre', 'exec'), g0)
    src = pkpatch(ht3patch(g0['src']))
    sys.argv = ['gen_nf2', OD]; ns = {'ns_HW': lambda: ns['HW'], '__name__': 'ht', '__file__': '/nf/gen_nf2.py'}
    exec(compile(src, 'gen_nf2_ht_v11', 'exec'), ns)
    for v in os.environ['VARIANTS'].split():
        name, fl, dbg = v.split(':'); FL = set(f for f in fl.split('+') if f != 'base')
        ns['FL'].clear(); ns['FL'].update(FL); ns['SLICE'] = 'SL0' not in FL
        L_ = {'ns': ns, 'name': name, 'FL': FL, 'dbg': dbg, 'OD': OD, 'np': np, 'os': os}; L_.update(G)
        exec(compile(textwrap.dedent(LOOPB), 'v11loop', 'exec'), L_); mac, byt, ops = L_['mac'], L_['byt'], L_['ops']
        print('wrote %-12s flags %-10s MAC %.2f G act %.1f MB ops %s' % (name, fl, mac / 1e9, byt / 1e6, ops))
    sys.exit(0)
if KIND == 'k1h':
    k = open('/k1h/gen_k1h.py').read()
else:
    t = open('/gn/gen_%s.py' % KIND).read(); KOUT = {}
    t = rep(t, "sys.argv = [p4, OD]; exec(compile(k, p4, 'exec'), {'__name__': '__main__', '__file__': p4})", "KOUT['k'] = k")
    sys.argv = ['gen_' + KIND, OD]; exec(compile(t, 'gen_' + KIND, 'exec'), {'__name__': 'v11' + KIND, 'KOUT': KOUT}); k = KOUT['k']
k = rep(k, "for k in ('ref0', 'rnd'):", "for k in ():   # PK: literal-shader K1 check inputs not kept in UP3/k1h/build (moved); graph unaffected")
k = rep(k, "src = g0['src']", "src = g0['src']; src = ht3patch(src); src = pkpatch(src)")
k = rep(k, LOOPA, LOOPB)
sys.argv = ['gen_k1h', OD]
exec(compile(k, 'gen_k1h_v11', 'exec'), G)
