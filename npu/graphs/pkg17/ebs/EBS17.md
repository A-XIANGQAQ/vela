# EBS17: the E-BS form of the 17 shipped v73 graphs (r7, PC only)

E-BS = r6 study section 2.5 (`C:/src/ext/fsr4r/AUDIT/r6/study/GRAPH_STUDY.md`): the 11 relu `e` convs (10 in `nf4_exact`) take their bias through the HTP int32 bias,
the relu clamp is the output zero point z of the view conv, the pair-sum conv gets bias 2z; no constant channels on those convs. Same I/O, same graph names.
Everything here was built and checked on the PC (docker `h-qnn`, QAIRT 2.47.0.260601, soc_model 43 / v73, `vtcm_mb 8`, `O 3`, x86 HTP simulator). No device run.

**Result: 17 of 17 graphs PASS all three checks and are kept in `ctx8/` and `b/`.**

Checks per graph: (1) control = the same chain with the patch off must give the released file (md5 of `NPUPIPE/ctx/p14/v73/<graph>.bin`); (2) simulator on the checker input
(byte i = (i*7 + (i>>12)) % 59, rows x 480 x 32), check hash as in `VTCMLOW/graphs/build7.sh`, must equal `NPUPIPE/rp6test/ref.txt`; (3) simulator on `noise` and `photo`
(first rows of `T2/work/inputs/*.raw`): E-BS output vs the released file's output, bytes different (cmp).

| graph | kind, graph name, input rows | control ok (control md5) | E-BS ctx md5 | bytes | compiler spill / fill MB, E-BS (released) | check hash vs ref | other inputs, bytes different | HMX-padded MAC, ops, Concat: released -> E-BS | z per layer | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| nf4_exact | ht, ht_t2s7p13, rows 540 | yes (2b8e1499) | be4514cb7314cfcceb687edbe9799837 | 1462272 | 27.75 / 27.75 (released 28.13 / 28.25) | a83aff6f6a4633e8 vs a83aff6f6a4633e8: EQUAL | noise=0 photo=0 | 57.13 -> 51.03 G, ops 110 -> 107, Concat 22 -> 19 | p03e=128 p05e=128 p09e=128 p11e=8 p15e=112 p17e=128 p19e=100 p21e=128 p23e=128 p25e=64 | PASS, kept |
| nf4_k1h | k1h, k1h_a, rows 540 | yes (a2c73bc9) | 14455133568f1164d9a9eddda9037d60 | 1564672 | 27.78 / 27.78 (released 28.13 / 28.25) | 8a735e657feee21f vs 8a735e657feee21f: EQUAL | noise=0 photo=0 | 65.69 -> 58.52 G, ops 118 -> 114, Concat 24 -> 20 | p03e=128 p05e=128 p09e=128 p11e=8 p15e=112 p17e=128 p19e=100 p21e=128 p23e=128 p25e=64 p27e=64 | PASS, kept |
| nf4_k1h_452 | n9, k1h_a, rows 452 | yes (07aed14c) | bfdbcb6b4fadc2bd3c4b0648c8eabc86 | 1355776 | 8.43 / 8.43 (released 14.26 / 14.36) | 2742c79e72e89a76 vs 2742c79e72e89a76: EQUAL | noise=0 photo=0 | 45.82 -> 40.82 G, ops 119 -> 115, Concat 24 -> 20 | p03e=128 p05e=128 p09e=128 p11e=8 p15e=112 p17e=128 p19e=100 p21e=128 p23e=128 p25e=64 p27e=64 | PASS, kept |
| nf4_k1h_450 | n9, k1h_a, rows 450 | yes (d25d7852) | c3943ea22af549045bda232954ec62a9 | 1331200 | 8.43 / 8.43 (released 14.26 / 14.36) | 2b9d68691c111dbe vs 2b9d68691c111dbe: EQUAL | noise=0 photo=0 | 45.82 -> 40.82 G, ops 123 -> 119, Concat 24 -> 20 | p03e=128 p05e=128 p09e=128 p11e=8 p15e=112 p17e=128 p19e=100 p21e=128 p23e=128 p25e=64 p27e=64 | PASS, kept |
| nf4_k1h_405 | n9, k1h_a, rows 405 | yes (11742856) | bf3aa41d192b6c174b4a89a493d01928 | 1302528 | 4.39 / 4.39 (released 6.94 / 6.94) | 80e8e720a693b5d5 vs 80e8e720a693b5d5: EQUAL | noise=0 photo=0 | 37.22 -> 33.16 G, ops 123 -> 119, Concat 24 -> 20 | p03e=128 p05e=128 p09e=128 p11e=8 p15e=112 p17e=128 p19e=100 p21e=128 p23e=128 p25e=64 p27e=64 | PASS, kept |
| nf4_k1h_400 | n9, k1h_a, rows 400 | yes (f5a38f36) | 1561a870c332313dc798b545380af970 | 1200128 | 0.00 / 0.00 (released 0.00 / 0.00) | b6a428542d1f38a3 vs b6a428542d1f38a3: EQUAL | noise=0 photo=0 | 32.44 -> 28.90 G, ops 119 -> 115, Concat 24 -> 20 | p03e=128 p05e=128 p09e=128 p11e=8 p15e=112 p17e=128 p19e=100 p21e=128 p23e=128 p25e=64 p27e=64 | PASS, kept |
| nf4_k1h_360 | n7, k1h_a, rows 360 | yes (be867d7d) | b82d8bb8d07958540efd76b19b095941 | 1155072 | 0.00 / 0.00 (released 0.00 / 0.00) | 2606832826ab4c91 vs 2606832826ab4c91: EQUAL | noise=0 photo=0 | 29.20 -> 26.01 G, ops 119 -> 115, Concat 24 -> 20 | p03e=128 p05e=128 p09e=128 p11e=8 p15e=112 p17e=128 p19e=100 p21e=128 p23e=128 p25e=64 p27e=64 | PASS, kept |
| nf4_k1h_324 | n9, k1h_a, rows 324 | yes (8685a9b6) | 2fd5a87d116bee0d1fcb6dab9130c315 | 1126400 | 0.00 / 0.00 (released 22.29 / 22.29) | 8b967666e675804b vs 8b967666e675804b: EQUAL | noise=0 photo=0 | 23.65 -> 21.07 G, ops 119 -> 115, Concat 24 -> 20 | p03e=128 p05e=128 p09e=128 p11e=8 p15e=112 p17e=128 p19e=100 p21e=128 p23e=128 p25e=64 p27e=64 | PASS, kept |
| nf4_k1h_270 | n9, k1h_a, rows 270 | yes (3b4c1703) | aa9a1c61d02c2fb4a3aa48abc2279250 | 995328 | 0.00 / 0.00 (released 3.05 / 3.05) | 501dbf44d6b0410d vs 501dbf44d6b0410d: EQUAL | noise=0 photo=0 | 16.54 -> 14.74 G, ops 123 -> 119, Concat 24 -> 20 | p03e=128 p05e=128 p09e=128 p11e=8 p15e=112 p17e=128 p19e=100 p21e=128 p23e=128 p25e=64 p27e=64 | PASS, kept |
| nf4up_k1h | k1h, k1h_a, rows 540 | yes (b01ea30a) | ff63ee281d0248bdbca760997612a263 | 1589248 | 19.81 / 19.81 (released 28.13 / 28.25) | 60a3d9ad466fca76 vs 60a3d9ad466fca76: EQUAL | noise=0 photo=0 | 67.28 -> 60.12 G, ops 118 -> 114, Concat 24 -> 20 | p03e=48 p05e=128 p09e=64 p11e=64 p15e=64 p17e=64 p19e=64 p21e=48 p23e=48 p25e=64 p27e=128 | PASS, kept |
| nf4up_k1h_452 | n9, k1h_a, rows 452 | yes (e0d3f26d) | 6624436c9525b0cfa511b8bc4d4b4f67 | 1306624 | 0.00 / 0.00 (released 14.26 / 14.36) | 040ded774ade640a vs 040ded774ade640a: EQUAL | noise=0 photo=0 | 46.93 -> 41.93 G, ops 119 -> 115, Concat 24 -> 20 | p03e=48 p05e=128 p09e=64 p11e=64 p15e=64 p17e=64 p19e=64 p21e=48 p23e=48 p25e=64 p27e=128 | PASS, kept |
| nf4up_k1h_450 | n9, k1h_a, rows 450 | yes (39f3861c) | 1d2e5c04b887b34a3a1aa3d1ec4418e4 | 1396736 | 8.40 / 8.40 (released 14.26 / 14.36) | 90a028a11df7fc3e vs 90a028a11df7fc3e: EQUAL | noise=0 photo=0 | 46.93 -> 41.93 G, ops 123 -> 119, Concat 24 -> 20 | p03e=48 p05e=128 p09e=64 p11e=64 p15e=64 p17e=64 p19e=64 p21e=48 p23e=48 p25e=64 p27e=128 | PASS, kept |
| nf4up_k1h_405 | n9, k1h_a, rows 405 | yes (7648bca4) | 6ea9f76554aedc9529c3573ec18fbddf | 1302528 | 4.39 / 4.39 (released 6.94 / 6.94) | 1eb47d3c70a78111 vs 1eb47d3c70a78111: EQUAL | noise=0 photo=0 | 38.13 -> 34.07 G, ops 123 -> 119, Concat 24 -> 20 | p03e=48 p05e=128 p09e=64 p11e=64 p15e=64 p17e=64 p19e=64 p21e=48 p23e=48 p25e=64 p27e=128 | PASS, kept |
| nf4up_k1h_400 | n9, k1h_a, rows 400 | yes (307727e8) | 776b0054c0af9889d662eb663fce0cc7 | 1187840 | 0.00 / 0.00 (released 0.00 / 0.00) | 98d9a4339a8d0d76 vs 98d9a4339a8d0d76: EQUAL | noise=0 photo=0 | 33.23 -> 29.69 G, ops 119 -> 115, Concat 24 -> 20 | p03e=48 p05e=128 p09e=64 p11e=64 p15e=64 p17e=64 p19e=64 p21e=48 p23e=48 p25e=64 p27e=128 | PASS, kept |
| nf4up_k1h_360 | n7, k1h_a, rows 360 | yes (800b917c) | 1fbca3d883f4120ca0847e68d68508a9 | 1134592 | 0.00 / 0.00 (released 0.00 / 0.00) | f27ed0e09ef0d7b8 vs f27ed0e09ef0d7b8: EQUAL | noise=0 photo=0 | 29.90 -> 26.72 G, ops 119 -> 115, Concat 24 -> 20 | p03e=48 p05e=128 p09e=64 p11e=64 p15e=64 p17e=64 p19e=64 p21e=48 p23e=48 p25e=64 p27e=128 | PASS, kept |
| nf4up_k1h_324 | n9, k1h_a, rows 324 | yes (27e64c24) | fde547fd06dfac09914f9983e784c903 | 1118208 | 0.00 / 0.00 (released 22.29 / 22.29) | 0cfda9f2b7cabc25 vs 0cfda9f2b7cabc25: EQUAL | noise=0 photo=0 | 24.22 -> 21.64 G, ops 119 -> 115, Concat 24 -> 20 | p03e=48 p05e=128 p09e=64 p11e=64 p15e=64 p17e=64 p19e=64 p21e=48 p23e=48 p25e=64 p27e=128 | PASS, kept |
| nf4up_k1h_270 | n9, k1h_a, rows 270 | yes (aba14a85) | eb0f59e205e79f27ef573846016315ba | 991232 | 0.00 / 0.00 (released 3.05 / 3.05) | a71e9b4f04112039 vs a71e9b4f04112039: EQUAL | noise=0 photo=0 | 16.95 -> 15.14 G, ops 123 -> 119, Concat 24 -> 20 | p03e=48 p05e=128 p09e=64 p11e=64 p15e=64 p17e=64 p19e=64 p21e=48 p23e=48 p25e=64 p27e=128 | PASS, kept |

## Flags

- Performance graphs (`nf4_k1h*`): `T2+P07S+P13S+K1+Zp03c3+Zp27c3@26k-1+Zp05c3@10k-1+Zp25c3@1k-1+Zp09c3+Zp11c3@24k1+Zp15c3+Zp17c3+Zp19c3+Zp21c3@14k1+Zp23c3+Zp03f@5k1+Zp05f@12k1+Zp25f+Zp27f+Zp09f@48k1+Zp11f+Zp15f@30k1+Zp17f+Zp19f+Zp21f+Zp23f@46k1+Zp07@36k-1+Zp13+Zp23t+Zp03e+Zp05e+Zp09e+Zp11e+Zp15e+Zp17e+Zp19e+Zp21e+Zp23e+Zp25e+Zp27e`
- Ultra Performance graphs (`nf4up_k1h*`): `T2+P07S+P13S+K1+Zp03c3+Zp27c3+Zp05c3@9k-1+Zp25c3@18k1+Zp17c3@60k1+Zp15c3+Zp19c3+Zp03f@26k1+Zp05f@5k-1+Zp25f@19k-1+Zp27f+Zp07+Zp13+Zp19t+Zp23t+Zp09c3+Zp11c3+Zp21c3+Zp23c3+Zp09f+Zp11f+Zp15f+Zp21f+Zp17f@108k1+Zp19f@25k-1+Zp23f@54k1+Zp03e+Zp05e+Zp09e+Zp11e+Zp15e+Zp17e+Zp19e+Zp21e+Zp23e+Zp25e+Zp27e`
- `nf4_exact` (graph `ht_t2s7p13`, no K1): `T2+P07S+P13S+Zp03c3+Zp05c3@10k-1+Zp25c3@1k-1+Zp09c3+Zp11c3@24k1+Zp15c3+Zp17c3+Zp19c3+Zp21c3@14k1+Zp23c3+Zp03f@5k1+Zp05f@12k1+Zp25f+Zp09f@48k1+Zp11f+Zp15f@30k1+Zp17f+Zp19f+Zp21f+Zp23f@46k1+Zp07@36k-1+Zp13+Zp23t+Zp03e+Zp05e+Zp09e+Zp11e+Zp15e+Zp17e+Zp19e+Zp21e+Zp23e+Zp25e`
- z per layer: env `ZPO_<tag>D_o=<z>` for the generator (table column "z per layer"; defaults in `tools/r7_build.sh`, Ultra Performance overrides in `zset_<graph>.txt`).

## Earlier attempts that did not pass (kept out of the result folders)

- nf4up_k1h: attempt with z: p03e=160 p05e=160 p09e=160 p11e=160 p15e=160 p17e=160 p19e=160 p21e=160 p23e=160 p25e=160 p27e=160 -> E-BS md5 370217a8, check hash=5d487172027a34eb (ref 60a3d9ad466fca76), other inputs: noise=DIFF:6064196 photo=DIFF:6659930 -> FAIL
- nf4up_k1h_452: attempt with z: p03e=160 p05e=160 p09e=160 p11e=160 p15e=160 p17e=160 p19e=160 p21e=160 p23e=160 p25e=160 p27e=160 -> E-BS md5 e39330f8, check hash=0b63fa7919d73cc6 (ref 040ded774ade640a), other inputs: noise=DIFF:4228930 photo=DIFF:4641678 -> FAIL
- nf4up_k1h_405: attempt with z: p03e=160 p05e=160 p09e=160 p11e=160 p15e=160 p17e=160 p19e=160 p21e=160 p23e=160 p25e=160 p27e=160 -> E-BS md5 a16f954e, check hash=d1027efb2292cddd (ref 1eb47d3c70a78111), other inputs: noise=DIFF:3413123 photo=DIFF:3795464 -> FAIL
- nf4up_k1h_400: attempt with z: p03e=160 p05e=160 p09e=160 p11e=160 p15e=160 p17e=160 p19e=160 p21e=160 p23e=160 p25e=160 p27e=160 -> E-BS md5 f8298fa4, check hash=703e41dce8c5c780 (ref 98d9a4339a8d0d76), other inputs: noise=DIFF:2982993 photo=DIFF:3362833 -> FAIL
- nf4up_k1h_360: attempt with z: p03e=160 p05e=160 p09e=160 p11e=160 p15e=160 p17e=160 p19e=160 p21e=160 p23e=160 p25e=160 p27e=160 -> E-BS md5 e0e9183a, check hash=e471ec1a12c66c39 (ref f27ed0e09ef0d7b8), other inputs: noise=DIFF:2685048 photo=DIFF:3017244 -> FAIL

Ultra Performance with z = 160 on all 11 layers (what the r6 micro-probe suggested) is NOT exact in the built graphs: every UP graph built that way differs from the released one.
The micro-probe only tests the view conv `D`; with z = 160 the pair-sum conv `S` gets bias 2z = 320, a value above 256 that no Performance graph uses (their largest is 256).
INFERENCE, not bisected: the `S` bias above 256 is what breaks. The second z set keeps every z <= 128 (values the same probe found exact per layer: `C:/src/ext/fsr4r/AUDIT/r6/study/probe/s3_up/cmp.txt`).

## Failed / not determined

- no graph failed
- NPU milliseconds and device hashes of these files: not measured here (PC only).
- v79 files: not built.
- Why z = 160 fails (which op): not bisected.
- Inputs other than the checker pattern are synthetic (noise, photo); no real pre-pass dump exists.

## Files

- `ctx8/<graph>.bin`: the E-BS context binaries (v73, vtcm_mb 8, O 3) of the graphs that passed.
- `b/performance/<graph>/libs/x86_64-linux-clang/lib<graphname>.so`, `b/ultraperf/<graph>/libs/x86_64-linux-clang/libk1h_a.so`: their x86 model libraries (graphname `k1h_a`; `ht_t2s7p13` for nf4_exact).
- `work/ctl/<graph>/` (control builds), `work/ebs/<graph>/` (E-BS builds: ONNX, model .cpp, net.json, logs, `sim/` outputs), `work/w*.log` (one `RESULT|...` line per graph and attempt).
- `tools/`: `gen_r6.py`, `gen_nf_r6_performance.py`, `gen_nf_r6_ultraperf.py` (generator patch, copies of r6 + the same one-line patch on the UP `gen_nf.py`), `r7_build.sh` (in-container), `r7_driver.sh` (host), `fnv64.c`, `mkfixed.py`, `r7_sum.py` (this table).

## Repeat the build

```sh
cd C:/src/ext/fsr4r/AUDIT/r7/ebs        # Git Bash; image h-qnn; SDK C:/src/ext/fsr4r/H/sdk/x/qairt/2.47.0.260601; fsr4vk at 391f62bd in C:/src/ext/fsr4vk
sh tools/r7_driver.sh w1 nf4_exact nf4_k1h nf4_k1h_452 nf4_k1h_450 nf4_k1h_405 nf4_k1h_400 nf4_k1h_360 nf4_k1h_324 nf4_k1h_270     # one container per preset list
sh tools/r7_driver.sh w3 nf4up_k1h nf4up_k1h_452 nf4up_k1h_450 nf4up_k1h_405 nf4up_k1h_400 nf4up_k1h_360 nf4up_k1h_324 nf4up_k1h_270
python tools/r7_sum.py                  # rewrites EBS17.md from work/w*.log
```
Per graph the container does: generator (`KIND`, size variables from `PUB/stage/npu/graphs/graphs.txt`) -> `qnn-onnx-converter --custom_io ... --float_fallback --float_bitwidth 16` ->
`qnn-model-lib-generator -t x86_64-linux-clang` -> `qnn-context-binary-generator` (htp json: `vtcm_mb 8`, `O 3`, `soc_model 43`, `dsp_arch v73`), first with the patch off (control), then E-BS;
then `qnn-net-run --retrieve_context` on the three inputs. A graph is copied to `ctx8/` and `b/` only when all three checks pass.
