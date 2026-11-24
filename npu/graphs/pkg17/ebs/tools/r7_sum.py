#!/usr/bin/env python3
"""r7: write EBS17.md from the worker logs (work/w*.log RESULT lines) and the converter net.json files. Usage: python tools/r7_sum.py (cwd = AUDIT/r7/ebs)"""
import glob, json, os, hashlib
ORDER = ['nf4_exact', 'nf4_k1h', 'nf4_k1h_452', 'nf4_k1h_450', 'nf4_k1h_405', 'nf4_k1h_400', 'nf4_k1h_360', 'nf4_k1h_324', 'nf4_k1h_270',
         'nf4up_k1h', 'nf4up_k1h_452', 'nf4up_k1h_450', 'nf4up_k1h_405', 'nf4up_k1h_400', 'nf4up_k1h_360', 'nf4up_k1h_324', 'nf4up_k1h_270']
PUB = 'C:/src/ext/fsr4r/PUB/stage/npu/graphs'
att = {}
for f in sorted(glob.glob('work/w*.log'), key=os.path.getmtime):
    for l in open(f, errors='replace'):
        if l.startswith('RESULT|'):
            p = l.rstrip('\n').split('|'); att.setdefault(p[1], []).append((os.path.basename(f), p))
pad = lambda c: (c + 31) // 32 * 32
def macs(path):
    if not os.path.exists(path): return None
    g = json.load(open(path))['graph']; ns, ts = g['nodes'], g['tensors']; D = lambda n: ts[n]['dims']; tot = 0; ops = {}
    for n in ns.values():
        ops[n['type']] = ops.get(n['type'], 0) + 1
        if n['type'] == 'Conv2d':
            i, w, o = n['input_names'][0], n['input_names'][1], n['output_names'][0]; kh, kw, cg, co = D(w); grp = list(n['scalar_params'].get('group', {'x': 1}).values())[0]
            tot += D(o)[1] * D(o)[2] * kh * kw * (pad(D(i)[3]) * pad(D(o)[3]) if grp == 1 else cg * D(o)[3])
        elif n['type'] == 'TransposeConv2d':
            i, w, o = n['input_names'][0], n['input_names'][1], n['output_names'][0]; tot += D(i)[1] * D(i)[2] * D(w)[0] * D(w)[1] * D(i)[3] * D(o)[3]
    return tot / 1e9, sum(ops.values()), ops.get('Concat', 0)
mb = lambda s, k: next(('%.2f' % (int(x.split('=')[1]) / 1048576) for x in s.split() if x.startswith(k + '=')), '?')
rows, fails, first = [], [], []
for G in ORDER:
    a = att.get(G, [])
    if not a: rows.append('| %s | no result (build not finished) | | | | | | | | | |' % G); fails.append('%s: no RESULT line in work/w*.log (not built or still running when this file was written)' % G); continue
    full = [p for _, p in a if len(p) >= 19]; ok = [p for p in full if p[17] == 'PASS']; p = ok[-1] if ok else (full[-1] if full else a[-1][1])
    for _, q in a:
        if q is not p and len(q) >= 19 and q[17] != 'PASS': first.append('%s: attempt with %s -> E-BS md5 %s, check %s (ref %s), other inputs:%s -> %s' % (G, q[18], q[9][4:12], q[13], q[14][4:], q[16][6:], q[17]))
    if len(p) < 19: rows.append('| %s | %s | | | | | | | | | |' % (G, ' / '.join(p[2:]))); fails.append('%s: %s' % (G, ' / '.join(p[3:]))); continue
    m = p[4]; c = macs('work/ctl/%s/%s_net.json' % (G, m)); e = macs('work/ebs/%s/%s_net.json' % (G, m))
    keep = 'ctx8/%s.bin' % G; kmd5 = hashlib.md5(open(keep, 'rb').read()).hexdigest() if os.path.exists(keep) else None
    lib = 'b/%s/%s/libs/x86_64-linux-clang/lib%s.so' % (p[2], G, m)
    note = 'kept' if (p[17] == 'PASS' and kmd5 == p[9][4:] and os.path.exists(lib)) else ('NOT kept' if p[17] != 'PASS' else 'PASS but result files missing or different: check')
    rows.append('| %s | %s, %s, rows %s | %s (%s) | %s | %s | %s / %s (released %s / %s) | %s vs %s: %s | %s | %s | %s | %s |' % (
        G, p[3], m, p[5], p[8][7:], p[6][4:12], p[9][4:], p[10][5:], mb(p[11], 'spill_bytes'), mb(p[11], 'fill_bytes'), mb(p[12], 'spill_bytes'), mb(p[12], 'fill_bytes'),
        p[13][5:], p[14][4:], 'EQUAL' if p[15].endswith('yes') else 'DIFFERENT', p[16][6:].strip(), ('%.2f -> %.2f G, ops %d -> %d, Concat %d -> %d' % (c[0], e[0], c[1], e[1], c[2], e[2])) if c and e else '?',
        p[18][3:], p[17] + ', ' + note))
    if p[17] != 'PASS': fails.append('%s: control %s, check hash %s, other inputs:%s (z %s)' % (G, p[8][7:], 'equal' if p[15].endswith('yes') else 'DIFFERENT', p[16][6:], p[18][3:]))
fl = {k: open('%s/flags/%s.txt' % (PUB, k)).read().strip() for k in ('performance', 'ultraperf', 'exact')}
E11 = '+'.join('Zp%se' % x for x in ('03', '05', '09', '11', '15', '17', '19', '21', '23', '25', '27'))
npass = sum(1 for r in rows if 'PASS, kept' in r)
out = ['# EBS17: the E-BS form of the 17 shipped v73 graphs (r7, PC only)', '',
       'E-BS = r6 study section 2.5 (`C:/src/ext/fsr4r/AUDIT/r6/study/GRAPH_STUDY.md`): the 11 relu `e` convs (10 in `nf4_exact`) take their bias through the HTP int32 bias,',
       'the relu clamp is the output zero point z of the view conv, the pair-sum conv gets bias 2z; no constant channels on those convs. Same I/O, same graph names.',
       'Everything here was built and checked on the PC (docker `h-qnn`, QAIRT 2.47.0.260601, soc_model 43 / v73, `vtcm_mb 8`, `O 3`, x86 HTP simulator). No device run.', '',
       '**Result: %d of 17 graphs PASS all three checks and are kept in `ctx8/` and `b/`.**' % npass, '',
       'Checks per graph: (1) control = the same chain with the patch off must give the released file (md5 of `NPUPIPE/ctx/p14/v73/<graph>.bin`); (2) simulator on the checker input',
       '(byte i = (i*7 + (i>>12)) % 59, rows x 480 x 32), check hash as in `VTCMLOW/graphs/build7.sh`, must equal `NPUPIPE/rp6test/ref.txt`; (3) simulator on `noise` and `photo`',
       '(first rows of `T2/work/inputs/*.raw`): E-BS output vs the released file\'s output, bytes different (cmp).', '',
       '| graph | kind, graph name, input rows | control ok (control md5) | E-BS ctx md5 | bytes | compiler spill / fill MB, E-BS (released) | check hash vs ref | other inputs, bytes different | HMX-padded MAC, ops, Concat: released -> E-BS | z per layer | verdict |',
       '|---|---|---|---|---|---|---|---|---|---|---|'] + rows + ['',
       '## Flags', '',
       '- Performance graphs (`nf4_k1h*`): `%s+%s`' % (fl['performance'], E11),
       '- Ultra Performance graphs (`nf4up_k1h*`): `%s+%s`' % (fl['ultraperf'], E11),
       '- `nf4_exact` (graph `ht_t2s7p13`, no K1): `%s+%s`' % (fl['exact'], E11.replace('+Zp27e', '')),
       '- z per layer: env `ZPO_<tag>D_o=<z>` for the generator (table column "z per layer"; defaults in `tools/r7_build.sh`, Ultra Performance overrides in `zset_<graph>.txt`).', '',
       '## Earlier attempts that did not pass (kept out of the result folders)', ''] + (['- ' + x for x in first] or ['- none']) + ['',
       'Ultra Performance with z = 160 on all 11 layers (what the r6 micro-probe suggested) is NOT exact in the built graphs: every UP graph built that way differs from the released one.',
       'The micro-probe only tests the view conv `D`; with z = 160 the pair-sum conv `S` gets bias 2z = 320, a value above 256 that no Performance graph uses (their largest is 256).',
       'INFERENCE, not bisected: the `S` bias above 256 is what breaks. The second z set keeps every z <= 128 (values the same probe found exact per layer: `C:/src/ext/fsr4r/AUDIT/r6/study/probe/s3_up/cmp.txt`).', '',
       '## Failed / not determined', ''] + (['- ' + x for x in fails] or ['- no graph failed']) + [
       '- NPU milliseconds and device hashes of these files: not measured here (PC only).',
       '- v79 files: not built.',
       '- Why z = 160 fails (which op): not bisected.',
       '- Inputs other than the checker pattern are synthetic (noise, photo); no real pre-pass dump exists.', '',
       '## Files', '',
       '- `ctx8/<graph>.bin`: the E-BS context binaries (v73, vtcm_mb 8, O 3) of the graphs that passed.',
       '- `b/performance/<graph>/libs/x86_64-linux-clang/lib<graphname>.so`, `b/ultraperf/<graph>/libs/x86_64-linux-clang/libk1h_a.so`: their x86 model libraries (graphname `k1h_a`; `ht_t2s7p13` for nf4_exact).',
       '- `work/ctl/<graph>/` (control builds), `work/ebs/<graph>/` (E-BS builds: ONNX, model .cpp, net.json, logs, `sim/` outputs), `work/w*.log` (one `RESULT|...` line per graph and attempt).',
       '- `tools/`: `gen_r6.py`, `gen_nf_r6_performance.py`, `gen_nf_r6_ultraperf.py` (generator patch, copies of r6 + the same one-line patch on the UP `gen_nf.py`), `r7_build.sh` (in-container), `r7_driver.sh` (host), `fnv64.c`, `mkfixed.py`, `r7_sum.py` (this table).', '',
       '## Repeat the build', '', '```sh',
       'cd C:/src/ext/fsr4r/AUDIT/r7/ebs        # Git Bash; image h-qnn; SDK C:/src/ext/fsr4r/H/sdk/x/qairt/2.47.0.260601; fsr4vk at 391f62bd in C:/src/ext/fsr4vk',
       'sh tools/r7_driver.sh w1 nf4_exact nf4_k1h nf4_k1h_452 nf4_k1h_450 nf4_k1h_405 nf4_k1h_400 nf4_k1h_360 nf4_k1h_324 nf4_k1h_270     # one container per preset list',
       'sh tools/r7_driver.sh w3 nf4up_k1h nf4up_k1h_452 nf4up_k1h_450 nf4up_k1h_405 nf4up_k1h_400 nf4up_k1h_360 nf4up_k1h_324 nf4up_k1h_270',
       'python tools/r7_sum.py                  # rewrites EBS17.md from work/w*.log', '```',
       'Per graph the container does: generator (`KIND`, size variables from `PUB/stage/npu/graphs/graphs.txt`) -> `qnn-onnx-converter --custom_io ... --float_fallback --float_bitwidth 16` ->',
       '`qnn-model-lib-generator -t x86_64-linux-clang` -> `qnn-context-binary-generator` (htp json: `vtcm_mb 8`, `O 3`, `soc_model 43`, `dsp_arch v73`), first with the patch off (control), then E-BS;',
       'then `qnn-net-run --retrieve_context` on the three inputs. A graph is copied to `ctx8/` and `b/` only when all three checks pass.', '']
open('EBS17.md', 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
print('PASS kept %d of 17' % npass)
for r in rows: c = [x.strip() for x in r.strip('|').split('|')]; print(' | '.join([c[0], c[2][:3], c[3], c[6][-9:], c[7], c[-1]])[:210])
print('FAILS:', fails); print('EARLIER:', len(first))
