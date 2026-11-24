# VTCMLOW: plan small, store 8 (REPORT3 B.2) and test 7 (B.3), 2026-11-23

PC only. Nothing published, nothing run on a device. Toolchain: h-qnn image (QAIRT 2.47.0.260601 host tools, x86 HTP
simulator), vela-npu-build image (aarch64 gcc 13.3.0), armada:stable image (arm64 emulation, mock run).

## 1. Where the context binary stores the VTCM size

Method: the copy graph w4 (test 6 library, 3.9 MiB one-block copy) built for vtcm_mb 4, 5, 6, 7, 8 (`graphs/build7.sh`,
`work/run7.sh`; logs `work/c7/log/ctx_w4_v*.log`). The builds are deterministic: w4_v4 md5 cce10247... and w4_v8 8cebb579...
equal test 6's. w4 at 4 vs 5/6/7/8 differs in **19-23 bytes and in no other byte**, all of them encodings of the size:

| id | where (w4 file offset) | form |
|---|---|---|
| A | outer QnnSystemContext metadata, 0x11c | u32 bytes; `qnn-context-binary-utility` prints it as graphBlobInfo.vtcmSize in MB |
| B | HTP blob header tables (blob starts at the header's offset, 0x1000), 0x124c and 0x1280 | u32 bytes, each right after a u32 73 (dspArch) |
| S | `Sz` record of each of the 2 serialized graphs, 0x20dc and 0x50dc | u32 in 256-byte units (4 MiB = 0x4000) |
| O | `Op` record (compiler option string) of each serialized graph, 0x3e60 and 0x6e68 | ASCII `vtcm_size=4194304;` |
| T | same `Op` records, 0x34a7 and 0x64af | ASCII `02size_for_tiling=4;` (MB) |
| F | runtime header `fa 00 00 fa` after the last serialized graph, 0x73f8 and 0x7400 | u32 bytes, twice |

The HTP blob layout seen in all 21 builds (w4 x5, nf4_k1h_360 x5, nf4_k1h x5, nf4up_k1h_270 x5, m800): file header
`00000002 00000003`, then u64 metadata size / blob offset / blob size at 0x10 (blob size = the utility's contextBlobSize);
inside the blob, two serialized graphs at 4 KB-aligned offsets, each `[u32 magic][u16 words][u16 flags]` then records
`[u16 words][2-char tag]`: Id, Sz, Vr, Op, Mm, (Tc, Mu), Zz; then the runtime part with the `fa0000fa` header. Every build
has exactly the 9 fields A, B x2, S x2, O x2, F x2 (`patch_vtcm.py --scan`, all 21 files).

Header reading: `QnnHtpSystemContext.h` documents only the info the utility shows (graphBlobInfo: spillFillBufferSize,
vtcmSize in MB, optimizationLevel, htpDlbc, numHvxThreads); no checksum or signature field anywhere in
`include/QNN/HTP/*.h`, `System/*.h` (the only crc in the headers is a LoRA adapter check); the HTP backend doc
(`H/r10_npu/doc_htp_backend.html.txt`) mentions no context-binary checksum.

## 2. Checksum

**None over these bytes.** Proof: builds that differ only in the size differ in nothing but the fields above, so any
checksum, hash or signature covering them would differ too. The first word of each serialized graph (`5cfa9037` /
`2bf70973`) is a fixed magic, identical in w4, nf4_k1h_360 and m800. Negative controls on the simulator: flipping the magic
(0x2000) is refused at load (`graph.cc:3547::ERROR:bad header`, err 5005); flipping a byte inside the `Id` record (0x2010)
loads and gives the correct output. So the loader checks the format, not a checksum.

Other fields depending on the size: in w4 none (w4_v4 patched to 8 = w4_v8 built for 8 except the 2 `size_for_tiling`
bytes). In the FSR 4 graphs the plan itself depends on it (nf4_k1h_360 v6 vs v7: 406109 bytes differ, spill 5.7 MB both),
which is the point: the plan is kept, only the stored window size changes.

Options in the same `Op` string worth knowing (not used): `vtcm_offset=-1`, `vtcm_off_mb=0`, `vtcm_off_whence=` (empty),
`vtcm_total_mb=0`, `clear_on_release=1`. They look like the compiler's own VTCM-offset knobs; test 6's PGE offset config
did not change a byte (TRIED row on PGE), these were not tried.

## 3. patch_vtcm.py

`VTCMLOW/patch_vtcm.py IN OUT --mb 8` (or `--scan`). Rewrites A, B, S, O, F (9 fields) to the new size and nothing else;
keeps `size_for_tiling` (the size the plan was made for). No checksum to recompute (section 2). Refuses (exit 2) unless
the header sizes fit the file, the record chains end at their stated sizes, each graph has one Sz and one Op with one
`vtcm_size=`, there is one `fa0000fa` header, one A, exactly 2 B, and all fields hold the same 1..8 MiB size; re-scans
its output. md5 0b0d5de81159f7fcff850be5bf7fefda.

Patched to 8 (`graphs/patched/`, 19 bytes changed for 4/7, 21 for 5/6): w4_v4p..v7p, nf4_k1h_v4p..v7p (test 6 builds),
nf4up_k1h_270_v4p..v7p (test 6 builds), nf4_k1h_360_v4p..v7p (built here: `graphs/ctx/`, v8 sha256 3fea93d9... = the
package-14 build).

## 4. Simulator results (x86 HTP simulator, checker's u8 input; `work/sim_*.log`, `work/c7/info/`)

- `qnn-context-binary-utility` parses all 16 patched files: vtcmSize=8 for every one, spillFill unchanged
  (e.g. nf4_k1h_v4p 16646144 = nf4_k1h_v4).
- Outputs equal the reference hashes, same as unpatched:

| graph | builds simulated | check hash (all equal) | reference |
|---|---|---|---|
| w4 copy | v4p v5p v6p v7p | 8c43ab1ead7ee492, output = input | test 6 w4 |
| nf4_k1h | v4p v5p v6p v7p | 8a735e657feee21f | package 14 |
| nf4up_k1h_270 | v4p v5p v6p v7p | a71e9b4f04112039 | package 14 |
| nf4_k1h_360 | v8 v7 v6 v5 v4, v4p v5p v6p v7p | 2606832826ab4c91 | package 14 (`NPUPIPE/rp6test/ref.txt`) |

**What the simulator cannot show**: where the runtime places the window in the real VTCM. The simulator gives the same
output whatever the window position, so a correct simulator output only says the patched file loads and its plan is
intact. Whether the runtime then anchors an 8 MB window at the bottom (and whether it reads A, B, S, O or F for the size, or
something else) needs a device: our console first (`fsr4npu-check --runs 20` on the patched graphs: hash and time), then
the affected units with test 7.

## 5. Test 7 (`test7/`, not published)

- `test7/npu-rp6-test.sh` (sha256 f79c5c18c8a00d5846a7a2cb6db23e097c869fb37f07695003f6629008ea8828, md5 9e0c3c3f...),
  TAG npu-rp6-test7, URL `https://github.com/zrprod9-cmyk/vela/releases/download/npu-rp6-test7/npu-rp6-test7.tar.gz`.
  dash/sh/bash -n OK, no CR.
- `test7/npu-rp6-test7.tar.gz` sha256 29e2e96db699dd91881c8d4d94a44f64f7075e311435f0993a098754e76df353, 6743684 bytes;
  reproducible (`work/pack7.sh`: tar --sort=name, owner 0, mode go-w, mtime 2026-11-23 12:00Z, gzip -n -9; two packs gave
  the same sha). Contents and md5s: `work/md5_pkg7.txt`. Daemon = test5 build (50a7f1c7...); checker = test7 build
  (e42d9a31..., `test7/src`, = test5 checker + `--pattern/--per/--words`; the u8 path is unchanged, so P hashes compare
  with test 5/6). 22 graphs.
- Order: info (soc id/revision, serial as sha256 prefix only, cdsp md5, remoteproc states, uptime, cDSP/GPU temps) -> C1
  cold (m800, 00/ff, 3 runs each) -> M (m800, 00 ff 55 aa addr naddr, 3 runs each, up to 16 WORD lines per pattern) -> W
  (w4_v8, w4_v4, w4_v4p..v7p) -> F8 (released nf4_k1h, nf4_k1h_360, nf4up_k1h_270) -> F7p..F4p (the three patched FSR 4
  graphs each) -> C2 (60 s nf4_k1h loop, then 00/ff again; cold vs warm bad-word set compared by count + first + last +
  set hash) -> summary. 20 runs per P graph.
- Reading it: any FNp correct where F8 is wrong = the workaround works, the largest correct N is the unit's setting;
  w4_v4p correct where w4_v4 is wrong = the stored size moves the window to the bottom (the direct test).
  M: BITS = which bit is wrong and in which direction; CLASS (first run of each pattern, whose previous run had another
  pattern) = previous run's input / previous run's read (write did not reach the cell) / poison / another offset's address
  (aliasing, WORD prints the offset) / none; PAT "DIFFERENT set/values" across runs = unstable reads.
- Mock (`test7/mock/`, arm64 emulation, stub backend, memfd, stub graph files, real binaries): whole script 103 s with the
  60 s warm loop (1400 nf4_k1h stub runs); W 6/6 correct, F WRONG as expected (stub copies its input), M/C 0 bad words;
  with FSR4NPU_CHECK_INJECT=1 the WORD/BITS/CLASS/BADSET lines and the cold/warm comparison print
  (`mock_inject.txt`); wrong TEST_SHA -> "sha256 mismatch". The alias branch of CLASS was not hit by the injection.
- Deviation from REPORT3 B.3: the serial number is printed only as a 12-character sha256 prefix (the output is posted
  publicly); it still tells units apart.

## Not verified
- Real NPU behaviour of patched files: whether the runtime accepts them, where it puts the window, NPU time (device only).
- Which of the 9 fields the runtime actually reads; all 9 are patched together so the file stays self-consistent.
- The test7 checker on the real NPU (built with the test5 flags, no warnings; only run against the stub).
