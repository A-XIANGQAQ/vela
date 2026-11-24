# NPU package 17: the graph sets

Package 17 (`vela-npu-sysext-17.raw`, Vela 0.8.25) ships two sets of the 17 HTP v73 graphs for the FSR 4 network. They are the
same network as in package 15 and 16 and give the same output, bit for bit; only the way the compiler plans the work differs.
The 17 HTP v79 graphs (`v79/`, Snapdragon 8 Elite) are unchanged from package 15.

This folder holds the recipes and the hashes. The graph files themselves are not in the repository: they are in the release
image (`vela-npu-sysext-17.raw`, sha256 `8795c509be690e643f6739112aa4033d4c1fff3a5e55a0c58ce4178963046bc6`, file list in
[`../../vela-npu-sysext-17.contents.txt`](../../vela-npu-sysext-17.contents.txt)). The general build of the graphs (ONNX from
the captured shaders, QAIRT 2.47.0.260601, `qnn-context-binary-generator`) is described in
[`docs/npu-graphs.md`](../../../docs/npu-graphs.md); this page covers what is new in package 17.

| set | where in the image | used when |
|---|---|---|
| main set (17 files) | `/usr/lib64/vela-npu/nf4*.bin` | the start self-check passes (almost every unit) |
| low4 set (17 files + `SELFCHECK`) | `/usr/lib64/vela-npu/low4/` | the self-check of the main set gives a wrong result on this unit |

## What the start self-check does (package 16 and 17)

When the NPU service `fsr4npud` starts (`--graph-set auto`, the default; environment variable `FSR4NPUD_GRAPH_SET` =
`auto`, `released` or `low`), it runs the graph `nf4_k1h_360` three times on a fixed input and compares the output hash with
the known one (`2606832826ab4c91`).

- Right: the main set is used.
- Wrong: the same check runs on `low4/nf4_k1h_360`. If that is right, every graph is loaded from `low4/`. If it is wrong too, the
  service stops and Vela runs FSR 4 on the GPU, instead of showing a broken picture.
- The check cannot run (file or load error): the main set, as before.
- `released` skips the check (package 15 behaviour); `low` uses `low4/` without a check (tests).

Why: on some Retroid Pocket 6 and Odin 2 Portal units (issue #2) the upper part of the NPU's 8 MB fast memory (VTCM) window reads
wrong bits, so graphs that use that part give wrong pictures. Graphs planned for a 4 MB window do not use it.

The power governor (`FSR4NPUD_POWER_GOV`) is on by default in package 16 and 17: it lowers the NPU clock when the game has
time to spare. `FSR4NPUD_POWER_GOV=0` turns it off (the fixed vote of package 15). The graph sets above do not depend on it.

## The main set: E-BS graphs

**What E-BS is.** Every FSR 4 convolution layer that is followed by a ReLU used to get its bias from two extra constant input
channels (a plane of 1 and a plane of 64) and clamp at zero with an unsigned zero-point-0 output. In E-BS ("bias through the
HTP bias") the bias goes into the HTP's own int32 bias input, the lower clamp of the ReLU is the output's zero point `z`, and the
pair-sum convolution that follows gets the bias `2z`. The integers that come out are the same, so the result is bit for bit
the same; the graph has fewer channels (34 to 32 and 66 to 64 input channels) and 3 to 4 operations fewer, so it needs less
memory traffic. The `z` per layer is in [`EBS17.md`](ebs/EBS17.md); the eight Ultra Performance graphs use `z` lists
that never exceed 128 (`ebs/zset_*.txt`), because `z = 160` was not exact (the bias `2z` of the pair-sum convolution went above
256; not narrowed down further).

**How it was built.** [`ebs/tools/r7_driver.sh`](ebs/tools/r7_driver.sh) starts one container per list of graphs and runs
[`ebs/tools/r7_build.sh`](ebs/tools/r7_build.sh) in it. For each graph: the generator
(`gen_r6.py`, `gen_nf_r6_performance.py`, `gen_nf_r6_ultraperf.py`; these are the generators of
[`../gen/`](../gen/) plus the E-BS patch, switched by the environment variables `ZPO_<layer>D_o=<z>`) writes the ONNX graph, then
`qnn-onnx-converter` (`--custom_io`, `--float_fallback --float_bitwidth 16`), `qnn-model-lib-generator`
(`x86_64-linux-clang`) and `qnn-context-binary-generator` (`vtcm_mb 8`, `O 3`, `soc_model 43`, `dsp_arch v73`) with QAIRT
2.47.0.260601. Each graph is first built with the patch off (control): that file must equal the released one byte for byte.

**Checks before a graph was accepted** (17 of 17 passed; table in [`EBS17.md`](ebs/EBS17.md)):
1. the control build equals the released file;
2. on the x86 HTP simulator, the output hash for the checker's fixed input equals the reference hash of the released graph
   ([`ebs/ref.txt`](ebs/ref.txt));
3. on two other inputs (noise and a photo-like one) the E-BS output differs from the released graph's output in 0 bytes.

Then on a Retroid Pocket 6 (SM8550) every main-set file gave the reference hash in every check (17 checks per file, 40 or 20 runs
each), and the NPU time of each graph was measured with six loaded instances per graph.

**Which graph gets which file.** An E-BS file is used only where it was faster at the standard 8 MB plan, measured on the
device. For the two 1440x810 graphs (`nf4_k1h_405`, `nf4up_k1h_405`) it was not faster, so package 17 keeps the released file
there (same md5 as in package 15). The other 15 are E-BS files. NPU time per frame on the Retroid Pocket 6 (median of six
instances, released to E-BS): 1920x1080 Performance 8.82 to 8.64 ms, Ultra Performance 8.98 to 8.09 ms, 1600x900 and 1600x904
4 to 8 % less, 1280x800 5 to 6 % less, 1280x720 6 to 7 % less, 1152x648 31 % less, 960x540 16 % less.

## The low4 set

These are the same 17 graphs in E-BS form, compiled for a 4 MB VTCM window (`vtcm_mb 4`), for units where the main set fails the
self-check. After the build, the window size stored in the file is changed from 4 to 8 MB with
[`low4/patch_vtcm.py`](low4/patch_vtcm.py) (`--mb 8`). The compiler's plan (4 MB) is kept; only the stored size changes, so that
the runtime places the window at the bottom of the 8 MB it is asked for, below the faulty part. Notes on the file format and
the patch: [`low4/VTCMLOW_NOTES.md`](low4/VTCMLOW_NOTES.md).

Recipe: [`low4/build_vt.sh`](low4/build_vt.sh) (with [`low4/build7.sh`](low4/build7.sh) in the container) builds each E-BS graph at the
chosen `vtcm_mb` and patches it; its `patched/<graph>_v4.bin` files are the low4 files. (Package 16 built its low4 set from the released
graphs with [`low4/run_all_pkg16.sh`](low4/run_all_pkg16.sh).) The device hash of every low4 file is the reference hash.
They were faster than the package 16 low4 files in 17 of 17 cases (3 to 14 %, single instances).

**What is proven and what is not.** The low4 files give the right hash on the Retroid Pocket 6 (SM8550, 17 of 17 in 11 checks per file)
and on the simulator (17 of 17). The 4 MB plan of the *released* graph form was proven on two affected units (package 16). The 4 MB plans of
the E-BS form have not been run on an affected unit yet: it is the same window with another plan inside it. If such a unit fails
on them, the self-check refuses them too and Vela runs FSR 4 on the GPU.

## Hashes of every shipped graph (v73)

md5 and sha256 of the file as it is in the image. "package 15 md5" is the file that package 15 and 16 shipped.

### Main set

| file | output | form | md5 | sha256 | package 15 md5 |
|---|---|---|---|---|---|
| `nf4_k1h.bin` | 1920x1080 Performance | E-BS, 8 MB plan | `14455133568f1164d9a9eddda9037d60` | `728fd6055939d2f04b91ef499889dc7631d435a57c1f49b3a39ebb1d2809ea65` | `a2c73bc9723bc99780ebd1ed4e052765` |
| `nf4up_k1h.bin` | 1920x1080 Ultra Performance | E-BS, 8 MB plan | `ff63ee281d0248bdbca760997612a263` | `02b6b401e1e6127ac155cbe50a515ba8547a429efd8be2289521786341cffabc` | `b01ea30ab8ba300b0d5bc9ea89ab7098` |
| `nf4_exact.bin` | 1920x1080, exact (no K1) | E-BS, 8 MB plan | `be4514cb7314cfcceb687edbe9799837` | `6c8c760afea3d6a541ffeeab57033759a3e8d9617d6d2303e1814b941e9bc287` | `2b8e1499edade62543532b03602c27d3` |
| `nf4_k1h_452.bin` | 1600x904 Performance | E-BS, 8 MB plan | `bfdbcb6b4fadc2bd3c4b0648c8eabc86` | `f13200a6311b0b1a33fec69a3e129cc5cb5e673f4cc30388f337a6fd0c7a4e15` | `07aed14c54a538ff67f65ba2ed7b32c0` |
| `nf4up_k1h_452.bin` | 1600x904 Ultra Performance | E-BS, 8 MB plan | `6624436c9525b0cfa511b8bc4d4b4f67` | `2ac6a26617c2ad8095fcbb872873816a083454032defd98adfeae29e63527efb` | `e0d3f26d1040a86bb1c02827c85d480b` |
| `nf4_k1h_450.bin` | 1600x900 Performance | E-BS, 8 MB plan | `c3943ea22af549045bda232954ec62a9` | `c1e877218123d7bf2a2e83177563b39527201a0b100ef0f7884584c3ab7a46d8` | `d25d78529d2dc9bb93d019098233ce60` |
| `nf4up_k1h_450.bin` | 1600x900 Ultra Performance | E-BS, 8 MB plan | `1d2e5c04b887b34a3a1aa3d1ec4418e4` | `21e702825814c81d88c7d478b58eec22b88ab2bf42894eeebae97877b3d25346` | `39f3861c4c6e50ffbc55bb430fb1064c` |
| `nf4_k1h_405.bin` | 1440x810 Performance | released file kept (E-BS was not faster) | `11742856ea4cf733bfd93349553af398` | `ce350513503962d882c377efe851c16cc1dc88c565990360b9a7fa8d2dad23d4` | `11742856ea4cf733bfd93349553af398` |
| `nf4up_k1h_405.bin` | 1440x810 Ultra Performance | released file kept (E-BS was not faster) | `7648bca41d820ca49a4a60af99d9c9cc` | `3923a77274705492a46a2393439ea0c0d907780b460b038e119f1aa306315a21` | `7648bca41d820ca49a4a60af99d9c9cc` |
| `nf4_k1h_400.bin` | 1280x800 Performance | E-BS, 8 MB plan | `1561a870c332313dc798b545380af970` | `b217a47e2fa5d2e5a991e01eaffefaaa0c34bf892ce7aa044765100a64007d7b` | `f5a38f36957354f48fcedb6c8adda3f8` |
| `nf4up_k1h_400.bin` | 1280x800 Ultra Performance | E-BS, 8 MB plan | `776b0054c0af9889d662eb663fce0cc7` | `5ab429798b38a22906010799c3e069672c1c77527bce800873299ecd89779697` | `307727e8aaa1f76856e3b70ecb1642f1` |
| `nf4_k1h_360.bin` | 1280x720 Performance | E-BS, 8 MB plan | `b82d8bb8d07958540efd76b19b095941` | `e671abc0f39041512e0e30c325bfa52f112392fcd61ac6894e24d173cd56deb7` | `be867d7d4ede06c4d5b83cd69b863102` |
| `nf4up_k1h_360.bin` | 1280x720 Ultra Performance | E-BS, 8 MB plan | `1fbca3d883f4120ca0847e68d68508a9` | `592f9871a2f3a8f2e08b272969c9a0601eedade660f74ad453a7c32a87bbc794` | `800b917c548e5349fb67fe8ad0b52cb8` |
| `nf4_k1h_324.bin` | 1152x648 Performance | E-BS, 8 MB plan | `2fd5a87d116bee0d1fcb6dab9130c315` | `5e0b697293b1a66f5bcff238db3e610dc4365d612b68a84d3e7b26c5169362af` | `8685a9b656398e8b3673653da83b3b9a` |
| `nf4up_k1h_324.bin` | 1152x648 Ultra Performance | E-BS, 8 MB plan | `fde547fd06dfac09914f9983e784c903` | `2010cadb9fe6f9f4886520f055117001f3a9dc1e1bc57a55557ddd9089332f28` | `27e64c24c1961c010e7dbc5191b221d6` |
| `nf4_k1h_270.bin` | 960x540 Performance | E-BS, 8 MB plan | `aa9a1c61d02c2fb4a3aa48abc2279250` | `d6e46a2613ab07baca6becc84736ce1825153b477d982ab3f0e2c32d2e27bcbe` | `3b4c1703d7f462f8f73b28490c80e4fa` |
| `nf4up_k1h_270.bin` | 960x540 Ultra Performance | E-BS, 8 MB plan | `eb0f59e205e79f27ef573846016315ba` | `e5008388a8ae0c0b152493c18a5b7ece18dd67ffb92f52ec3b9cc1c9b210f9bc` | `aba14a85257280b487f765323b37e6f1` |

### low4 set

| file | output | md5 | sha256 | package 16 md5 |
|---|---|---|---|---|
| `low4/nf4_k1h.bin` | 1920x1080 Performance | `ec218760d4e2e477630b01b8cee77382` | `f8df9bf3b1384aae4f44070df06b590128c67b4af3cf1bcfe696c768e9a89ae5` | `cd8abcde214d65eb28a914cca519ccee` |
| `low4/nf4up_k1h.bin` | 1920x1080 Ultra Performance | `a78f87a9f4833fd8f4b8ec740db591a2` | `921db50b7cb44bd511ef9de5e66a46e02ecef207e78104ee9c0526c468a1268e` | `8fb6fd7572ab84a77a27b2e562b5214d` |
| `low4/nf4_exact.bin` | 1920x1080, exact (no K1) | `1d238885e3d406e5ff99d1fb4d077479` | `460c8d230f2194db114d45a8286f25f426dcb7bb5aaaabc1cfffa1147a3068d3` | `304a1d6e185799812c4e1a84a1c6441b` |
| `low4/nf4_k1h_452.bin` | 1600x904 Performance | `5dce097d05ace3c87d07a1f16db0e59d` | `f8f76cc4e5258d1324ac0c739d0554e82388ede8d2fcf6e3db4d6738d3ce8cf1` | `a61d90f4c2a2dc26c6e2c45abb921c41` |
| `low4/nf4up_k1h_452.bin` | 1600x904 Ultra Performance | `7cb137de2b614b4cdfbf5b3415fb403c` | `f427bc0cef74bcfa3b5b00f962a81b617d2263eb2929cde09fb59ac984a9107e` | `e80c674208bbf74c845e46e0585888f6` |
| `low4/nf4_k1h_450.bin` | 1600x900 Performance | `94b6dc9694df8ec2704fce6f2667d00c` | `67a8885b3fde0b0efbffec160b95c840b65e940abdb98c602b9ee11cadbe3b73` | `2e2f5ab7fd6aa05d8f493e93bf3220ad` |
| `low4/nf4up_k1h_450.bin` | 1600x900 Ultra Performance | `8bc541ab97e5d47ad76609871f53e5fc` | `e1e22bcf98f0e9f67df4f3dd9c80fefa37d1b0a317077991eaa711ce99565813` | `e8045d38a69665ec79b0f15d162c59b8` |
| `low4/nf4_k1h_405.bin` | 1440x810 Performance | `92472221c9a04fb753a80af0092ecd52` | `ce330e4a5f3d5780149464dcaa175a6eded8029539c62b9247cb687912550a9e` | `0f6d24f8e8e8b7ac0ae6802475e5b417` |
| `low4/nf4up_k1h_405.bin` | 1440x810 Ultra Performance | `258981eaa6888ce3caf5478e4dce4b08` | `b595856c8c4542a2f94857172d3a0487857127e4387259575aa85e92a30ac3f9` | `10fe1dc2121fddb26b27213341c17f61` |
| `low4/nf4_k1h_400.bin` | 1280x800 Performance | `8009a3b80ff3cd4338566069ed0d8804` | `c362a1f734857287b28ad518234f16f3a06c2a74974abae60bd0c8c59bff7426` | `afe1ca6af10135035fe0341d875d4fc6` |
| `low4/nf4up_k1h_400.bin` | 1280x800 Ultra Performance | `77d9030aaf22c33114ee4cc1d866d73b` | `e514ca90e8cfae91361ed71372c7e9463e655eb46eccff6d8353e7268dd21a15` | `ab297aec265e7df2baf27f8ef99efa9f` |
| `low4/nf4_k1h_360.bin` | 1280x720 Performance | `0d98ff7d7dcf44b4c286c0d707fc604f` | `f32361cefa59716d2332bfd577312cc0b5599f3bdc149c1391d4efbaa2e0cee5` | `e1e2b8121e844dbca7c3fe574f10e196` |
| `low4/nf4up_k1h_360.bin` | 1280x720 Ultra Performance | `6b78e5ecc1405399873f3f268e921366` | `e847d5b93e718025f16fd544b47eec09946a679bd181b123233ac3f1197a12e5` | `330ba8efa6defc63261001b82afb7363` |
| `low4/nf4_k1h_324.bin` | 1152x648 Performance | `8f638ea93b27ba11bc01a041cda087a2` | `bf581bba6f964e0d3ebcbedd66b42204ca00b18f6c3c739cbd2ec50e51cfebd9` | `c853965345bbcfbe7f9a52736452e3d3` |
| `low4/nf4up_k1h_324.bin` | 1152x648 Ultra Performance | `4b6530065a77c34718b6888d53a15ddd` | `bf496c632b5745fc66f2742b40a77030ea11b05190574ab62b3cee447dcb85e3` | `0070662801378560dc2e15f1bec3086d` |
| `low4/nf4_k1h_270.bin` | 960x540 Performance | `45039ada0a4c7b07ad3f1981a87a905e` | `410c81509447185565fb771325b5104a19c8bdbb6cd5a2955d5424312d999037` | `364e4beb5b0dc9a6d70e6983f116d56f` |
| `low4/nf4up_k1h_270.bin` | 960x540 Ultra Performance | `710f11fb140b048ef63b41a33f2a5aaf` | `a8f7d197bb5c130714a3aa1004c8cf8007946184a51a6d8c649203481b7c3f10` | `b96365462937046448b1de4b19c0095b` |

The self-check file: `low4/SELFCHECK` (md5 `2f1ab04287a4e97bdddeb9d0fc1e8f16`, sha256 `2f564896d046be47ec749572338f00053dc9b18c839d64715c9fc758e9e0246f`): `nf4_k1h_360 2606832826ab4c91`

The md5 of every file was also checked against the image build's list (`payload_md5_17.txt` of the build): all 17 + 17 + 1 equal.

## Rebuild

You need Docker, the QAIRT SDK 2.47.0.260601, a checkout of fsr4vk at the commit named in `docs/npu-graphs.md`, and the container image
`h-qnn` of that document. The scripts keep the paths of the machine they ran on (`C:/src/ext/fsr4r/...`, `C:/src/ext/fsr4vk`);
change the mounts at the top of `r7_driver.sh` and `build_vt.sh`.

```sh
# main set (one container per list)
sh ebs/tools/r7_driver.sh w1 nf4_exact nf4_k1h nf4_k1h_452 nf4_k1h_450 nf4_k1h_405 nf4_k1h_400 nf4_k1h_360 nf4_k1h_324 nf4_k1h_270
sh ebs/tools/r7_driver.sh w3 nf4up_k1h nf4up_k1h_452 nf4up_k1h_450 nf4up_k1h_405 nf4up_k1h_400 nf4up_k1h_360 nf4up_k1h_324 nf4up_k1h_270
# low4 set: E-BS graphs at vtcm_mb 4, then the stored size patched to 8 MB
SZ=4 sh low4/build_vt.sh
```

The compiler is deterministic: a rebuild gives the same file. The rebuilt main files can be compared with the hashes above.
