#!/usr/bin/env python3
"""The exact k1h graph chain (N9's size substitution) at any output size. Env: NSZ_W grid columns (round_up(w, 8) / 2), NSZ_HN grid rows
(round_up(h, 8) / 2), NSZ_HR real rows ((h & ~1) / 2), NSZ_WR real columns ((w & ~1) / 2).
  x  uint8 1 x HR x 480 x 32: rows of 960 px (the fused pre's literal stride). The graph slices the grid columns, zeroes an odd last real column
     and pads the rows HR..HN-1 with zeros, the X0 the reference runner sees after its per-frame clear.
  K1 reads OUT only at px < WR and py < HR (k1x bound, else int8 0): the graph zeroes those rows and that column first.
  k1 dense at K2's stride value[4].x / 2: 1 x HR x WR/2 x 32, or 1 x HR x WR x 16 for an odd WR."""
import os, re, sys
W, HN, HR, WR = (int(os.environ[k]) for k in ('NSZ_W', 'NSZ_HN', 'NSZ_HR', 'NSZ_WR'))
SUB = [('540', os.environ['NSZ_HN']), ('960', os.environ['NSZ_W']), ('270', str(int(os.environ['NSZ_HN']) // 2)), ('480', str(int(os.environ['NSZ_W']) // 2)), ('135', str(int(os.environ['NSZ_HN']) // 4)), ('240', str(int(os.environ['NSZ_W']) // 4))]
SM = dict(SUB)
def sz(s): return re.sub(r'(?<![0-9])(540|960|270|480|135|240)(?![0-9])', lambda m: SM[m.group(1)], s)
OD = sys.argv[1]; S = OD + '/src'; os.makedirs(S, exist_ok=True)
def put(name, t): open('%s/%s' % (S, name), 'w').write(t); return '%s/%s' % (S, name)
def one(t, a, b):
    assert t.count(a) == 1, (a, t.count(a)); return t.replace(a, b)
g1 = sz(open('/nf/gen_nf.py').read())
old = "xr = np.fromfile('/h/r6_build/x_real.raw', np.uint8)"
if old in g1: g1 = g1.replace(old, old + ".reshape(540, 960, 16)[:%d, :%d].ravel()" % (HN, W))
p1 = put('gen_nf.py', g1)
p2 = put('gen_nf2.py', one(sz(open('/nf/gen_nf2.py').read()), "open('/nf/gen_nf.py')", "open('%s')" % p1))
p3 = put('gen_ht.py', one(sz(open('/ht/gen_ht.py').read()), "open('/nf/gen_nf2.py')", "open('%s')" % p2))
k = one(sz(open('/k1h/gen_k1h.py').read()), "open('/ht/gen_ht.py')", "open('%s')" % p3)
def c64(n, v): return "g.c('%s', np.array(%s, np.int64))" % (n, v)
PADB = '[0, 0, 0, 0, 0, 0, %d, 0]' % (HN - HR); PADC = '[0, 0, 0, 0, 0, 16, 0, 0]'; ODD = WR != W
def oddcol(t, p, ind):   # statements: tensor t gets the second pixel of its last cell set to zero (Concat of the other cells, the kept 16 channels, 16 zero channels)
    return ind + ("%s_a = g.node('Slice', [%s, %s, %s, %s], '%s_a')\n" % (p, t, c64(p + '_a0', '[0]'), c64(p + '_a1', '[%d]' % (W // 2 - 1)), c64(p + '_ax', '[3]'), p)
        + ind + "%s_b = g.node('Pad', [g.node('Slice', [%s, %s, %s, %s], '%s_bs'), %s], '%s_bp')\n" % (p, t, c64(p + '_b0', '[%d, 0]' % (W // 2 - 1)), c64(p + '_b1', '[%d, 16]' % (W // 2)), c64(p + '_bx', '[3, 1]'), p, c64(p + '_cp', PADC), p)
        + ind + "g.nodes.append(hp.make_node('Concat', [g.qdq(%s_a, '%s_qa'), g.qdq(%s_b, '%s_qb')], ['%s_cc'], axis=3)); %s = '%s_cc'\n" % (p, p, p, p, p, t, p))
xa = "    xc_ = g.node('Slice', ['x', %s, %s, %s], 'x_sl')\n" % (c64('xs_a', '[0]'), c64('xs_b', '[Wx]'), c64('xs_x', '[3]'))
if ODD: xa += oddcol('xc_', 'x', '    ')
if HN > HR: xa += "    xc_ = g.node('Pad', [xc_, %s], 'x_pd')\n" % c64('xs_p', PADB)
xa += "    x = g.kconst(A(g.qdq(xc_, 'xq'), P, 16, False), (Hx, Wx)); T = {}"
XS = "\n" + "a = %r; assert src.count(a) == 1\nsrc = src.replace(a, %r)\n" % ("    x = g.kconst(A(g.qdq('x', 'xq'), P, 16, False), (Hx, Wx)); T = {}", xa)
XS += "a = %r; assert src.count(a) == 1\nsrc = src.replace(a, %r)\n" % ("vi('x', (32, %d, %d) if pack else (16, %d, %d))" % (HN, W // 2, HN, W), "vi('x', (32, %d, 480) if pack else (16, %d, 960))" % (HR, HR))
if HN > HR or ODD:
    ka = "    if 'K1' in FL:\n        to = T['OUT']; assert not to.k, 'OUT has const channels'; print('NSZ OUT P C n', to.P, to.C, to.n)\n"
    if HN > HR:
        ka += "        om = g.qdq(g.node('Pad', [g.node('Slice', [to.name, %s, %s, %s], 'om_sl'), %s], 'om_pd'), 'omq')\n" % (c64('om_a', '[0]'), c64('om_b', '[%d]' % HR), c64('om_x', '[2]'), c64('om_p', PADB))
    else:
        ka += "        om = to.name\n" + oddcol('om', 'om', '        ') + "        om = g.qdq(om, 'omq')\n"
    ka += "        T['K1'] = blk2(g, L['p27'], A(om, to.P, to.C, False, idx=to._idx), 'p27')"
    XS += "a = %r; assert src.count(a) == 1\nsrc = src.replace(a, %r)\n" % ("    if 'K1' in FL: T['K1'] = blk2(g, L['p27'], T['OUT'], 'p27')", ka)
oa = "        g.nodes.append(hp.make_node('Identity', [src_], [on])); shp[on] = (c, hh, ww)"
oq = ''
if HN > HR: oq += "        if on == 'k1': src_ = g.qdq(g.node('Slice', [src_, %s, %s, %s], 'k1r_sl'), 'k1r'); hh = %d\n" % (c64('k1r_a', '[0]'), c64('k1r_b', '[%d]' % HR), c64('k1r_x', '[2]'), HR)
if ODD:     # k1 to pixels: NHWC cells -> NHWC pixels (16 channels), back to the graph's layout, then the real columns only
    oq += ("        if on == 'k1':\n"
           "            g.nodes.append(hp.make_node('Transpose', [src_], ['k1p_t1'], perm=[0, 2, 3, 1]))\n"
           "            s1_ = g.qdq('k1p_t1', 'k1p_q1')\n"
           "            s2_ = g.qdq(g.node('Reshape', [s1_, %s], 'k1p_rs'), 'k1p_q2')\n"
           "            src_ = g.qdq(g.node('Slice', [s2_, %s, %s, %s], 'k1p_sl'), 'k1p_q4'); c = %d; hh = %d; ww = 16\n") % (
           c64('k1p_sh', '[1, %d, %d, 16]' % (HR, W)), c64('k1p_a', '[0]'), c64('k1p_b', '[%d]' % WR), c64('k1p_x', '[2]'), HR, WR)
if oq: XS += "a = %r; assert src.count(a) == 1\nsrc = src.replace(a, %r)\n" % (oa, oq + oa)
k = one(k, "src += r'''", XS + "src += r'''")
p4 = put('gen_k1h.py', k)
print('NSZ sources', p1, p2, p3, p4)
sys.argv = [p4, OD]; exec(compile(k, p4, 'exec'), {'__name__': '__main__', '__file__': p4})
