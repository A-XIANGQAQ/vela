# Vela FSR 4 provider for Vulkan: complete corresponding source

`upscale/assets/sgsr3/amd_fidelityfx_upscaler_vk.dll` in the Vela 0.8.25 zip is built from this folder. The build is reproducible: it fixes the image
base of the DLL, pins the MinGW packages and checks the Vulkan-Headers commit. `build.sh` prints the md5 of the DLL; it must be `221a16478be0ebc6a0949383ecee5e35`.

## What changed in 0.8.25 (G2, lighter frames)

Against the 0.8.24 provider (md5 `d230a1208c5ad78b6ea573536d41184f`, the sources are in this repository's history at the 0.8.24 commits) five files changed
(`overlay/ov/fsr411-recorder.hpp`, `fv_npu.hpp`, `quality-context.hpp`, `vela-baked-pre.hpp`, new `spirv-sources/exact/plx_gen.py`).
In exact FSR 4 with the NPU, the pre-pass loads a variant (flag X0) and, on frames where the NPU answers, the GPU skips the X0 store, the
`border_0` pass and 26 GPU-network dispatches ("slim frames": only after 8 OK acknowledgements from the NPU, full frames for 16 frames after a
failed one). The picture is the same bit for bit: 240 of 240 frames equal to the 0.8.24 DLL on the PC stand, and 119 of 119 on the
Retroid Pocket 6 with the slim path engaged; the GPU part is about 0.1 ms shorter per frame. `VELA_FP=0` gives the 0.8.24 behaviour;
`VELA_WG=1` also halves the pre-pass group rows (not recommended: no gain). FSR 4 Fast is unchanged.

## License
- Provider code (fsr4vk's own code and the Vela changes in `overlay/`): **GPL-3.0** (`fsr4vk/LICENSES/GPL-3.0.txt`).
  fsr4vk is not MIT: its README says the repository has no single license; its own code is GPL-3.0.
- AMD FSR 4.1.1 INT8 and FSR 4.1.2 Lite shaders and weights (`vela_assets/fsr411/`, `overlay/ov/vela-fast-spv.hpp`,
  `overlay/ov/vela-baked-pre.hpp`, `overlay/ov2/fsr412lite-blobs.hpp`, `overlay/gen/fsr4fu-spv.hpp`):
  AMD's MIT license, notice and origin in `../plugin/assets/licenses/fsr4vk/AMD-FidelityFX-SDK-MIT.md`.
- FFX API headers (`fsr4vk/amd-fidelityfx-sdk/`): AMD MIT (`fsr4vk/LICENSES/AMD-FidelityFX-SDK-MIT.md`, SDK 2.0.0 text).
- SR4 skipped-frame kernels (`overlay/gen/fsr4sr4-spv.hpp`, source in `sr4-kernels/`): GPL-2.0-or-later (rules from fsr4-shader-performance-overrides).
- Our own GLSL (`spirv-sources/fv`, `spirv-sources/npu`): GPL-3.0.
- Zstandard decoder (`fsr4vk/third_party/zstd`): BSD.

**No FSR 4.0.2 material**: this release build neither contains nor accepts the FSR 4.0.2 model. fsr4vk's 4.0.2 bundles
(`assets/general`, `assets/portable`, `assets/colors`) and its own `assets/fsr411` (replaced by Vela's modified copy in
`vela_assets/fsr411`) and `native/validation-data` are left out of `fsr4vk/` here; the build does not use them.

## What is where
| folder | content |
|---|---|
| `fsr4vk/` | https://github.com/dvj5411/fsr4vk at commit `391f62b` ("Delete provider/NATIVE-INPUTS.md"), files unchanged, without the folders above |
| `overlay/ov`, `overlay/ov2` | Vela's changed provider files (copied over `fsr4vk/native` and `fsr4vk/provider/ffx_vk_provider.cpp` by the build): FSR 4 Fast, its 1.5x mode for 1080p Ultra Performance, the NPU path (`fv_npu.hpp`), FSR 4.1.2 Lite, and the release change `build/prv_overlay.py` describes (no 4.0.2) |
| `overlay/gen` | generated SPIR-V headers: Lite NPU pack/unpack, fused kernels, SR4 kernels |
| `sr4-kernels/` | GLSL source, generators and build of the SR4 skipped-frame kernels (`sh sr4-kernels/build.sh`) |
| `spirv-sources/` | GLSL, scripts and builds of all SPIR-V headers except the SR4 kernels: `build.sh` (our own kernels) and `build_amd.sh` (the AMD-derived kernels; fetches AMD's FSR 4.1.2 Lite DLL, SDK v2.4.0, SHA-256 checked) |
| `vela_assets/fsr411` | the FSR 4.1.1 shader/initializer bundles the DLL embeds (Vela's modified passes) |
| `build/` | `dll_build_prv.sh` (the DLL build, runs in `ubuntu:24.04`), `prv_tools.py` (embed only `fsr411/`), `build_bench_prv.sh` (aarch64 test bench), `prv_overlay.py` (the exact edit that removed 4.0.2, as applied to the previous sources) |

## Build
1. Install Docker. From this folder run `sh build.sh` (Linux, macOS or Git Bash on Windows; ~10 min, network needed).
2. The build log must end with `PASS: frozen inputs, AMD64 exports, asset identities and normal/debug modes`.
3. `md5sum out/amd_fidelityfx_upscaler_vk.dll` must print `221a16478be0ebc6a0949383ecee5e35`. What makes the result the same on every run:
   - base image `ubuntu:24.04` with the MinGW packages pinned in `build/dll_build_prv.sh` (`g++-mingw-w64-x86-64-posix 13.2.0-6ubuntu1+26.1`,
     `binutils-mingw-w64-x86-64 2.41.90.20240122-1ubuntu1+11.4`; the versions are printed at the top of `out/build.log`);
   - Vulkan-Headers `v1.4.367`, checked against commit `9fbc2a782ee3f8a4db64e7d991e23da75e893f44`;
   - the linker image base is fixed to `0x250460000` (GNU ld otherwise picks a random base, which changes the checksum and every absolute address);
   - `SOURCE_DATE_EPOCH=1793577600`.
   A newer MinGW package in the Ubuntu archive than the pinned one makes the build stop at install time instead of giving a different DLL.
4. Optional checks of the generated kernels: `sh sr4-kernels/build.sh`, `sh spirv-sources/build.sh` and `sh spirv-sources/build_amd.sh` (each rebuilds its headers and compares them with `overlay/gen`, `overlay/ov` and `overlay/ov2`).
