# Vela's OptiScaler fork (GPL-3.0)

## Vela 0.10.0: new base (`base-b/`)

Since Vela 0.10.0 the fork is built on current upstream [optiscaler/OptiScaler](https://github.com/optiscaler/OptiScaler) commit
`6de775c5abab6909f543dc4b621fe68e89597ab7` (file `base-b/BASE`) instead of Sloptiscaler `f80e4da3`. The series in `base-b/patches/`
is: `0000` (Sloptiscaler `f80e4da3`'s additions re-applied on that upstream commit: the ARM64EC build and the SGSR 2 backend),
`0001`-`0059` (our patches of the list below, rebased onto the new code; the file names repeat the old numbers), `0060`-`0065`
(build fixes of the rebase) and `0066`-`0069` (OptiScaler.ini back to upstream; Unreal resource-barrier conditions restored; a stack
clear in the nvngx probe hooks, which a Wine bug on Proton 11 ARM64 needs). Apply them in file-name order with
`git am --3way --ignore-whitespace` on the base commit; the result is the tree the shipped DLL was built from (checked: the tree after
all 70 patches equals our `base-b` branch head `0e9b5b0c`). Checksums: `base-b/patches.md5`. `base-b/check_apply.sh` is our daily check that
the series still applies on upstream master (its paths are our working directories; edit `R`/`F` to use it).

Vela 0.10.0 ships `OptiScaler.dll` md5 `d1010f862b5617bb3a9b406840044168`. The pinned build container, compiler and tool versions of
the Build section below are unchanged; its clone-and-`git apply` step is replaced by checking out the base commit and applying the
`base-b` series as above. We did not re-run the container build from this public copy. The `patches/` folder and the text below
describe the 0.9.0 and earlier builds (base Sloptiscaler `f80e4da3`).

Base: [justradical/Sloptiscaler](https://github.com/justradical/Sloptiscaler) commit `f80e4da339cd4624212fd0546922571fe01205c1`
= ARM64EC build of [OptiScaler](https://github.com/optiscaler/OptiScaler) 10.0.0.1 with an SGSR 2 backend.
Our changes are the patches in `patches/`, applied in number order. License: GPL-3.0 (`LICENSE`, OptiScaler's).

The `OptiScaler.dll` in Vela 0.8.24 (md5 `3a02ef78a479ab4bd9a4ec6a3b5415d1`, 18 924 544 bytes) was built from the base +
**0001–0008, 0010–0021, 0023, 0024, 0024b–0024d and 0025–0040** (patch 0009 and the experimental upstream-sync patch 0000 are not part of this build and are not
included). Checksums of the patch files: `patches.md5`.

**Vela 0.9.0** (`OptiScaler.dll` md5 `bbb10a0b536c621ccbfb356fd4815ccc`) is the same base with patches **0001–0008, 0010–0021, 0023, 0024, 0024b–0024d and 0025–0059**; the new ones, 0049–0059, are in the table below.

| patch | what |
|---|---|
| 0001 | SGSR 2 input dumper; motion-vector scale and sampling fixes for full-resolution MV |
| 0002–0005 | SGSR 3 upscaler (D3D12), pixel-shader path, A/B timing; Vulkan interop through vkd3d-proton (student network, fp16 v1f) |
| 0006 | SGSR-FG frame generation passes (uses OptiScaler's FFX swapchain) |
| 0007–0008 | plugin build: temporal box, history rounding fixes (large: embedded shader bytecode) |
| 0010 | Vela live settings (model switch while the game runs) |
| 0011–0016 | FSR 4 bridge to the Vulkan FSR 4 provider (`Fsr4VkBridge.h`), live sub-rect / SGSR 2, NPU keys, FSR 3 through FFX, any render size |
| 0017 | FSR 4: depth flip, heaps, SGSR 2 fallback |
| 0018 | model `fsr4fast` (FSR 4 Fast) |
| 0019 | MV pass: converts display-size / jittered motion vectors (DLSS input) to the render-size, non-jittered vectors FSR 4 needs |
| 0020 | GPU identification without a held lock: fixes the start hang on Snapdragon 8 Elite; `VELA_OPTI_GPUTHREAD=0` turns the early identification off, `VELA_OPTI_LOGLEVEL=<n>` sets the log level |
| 0021 | an output the upscalers cannot write (R9G9B9E5) goes through an RGBA16F image, then a pack pass and a copy into the game's output |
| 0023 | a DLSS context that comes without a depth buffer is handled per handle; the game's other contexts keep FSR 4 |
| 0024 | looks for the depth buffer under another NGX parameter name before giving up; `OPTI_VELA_NODEPTH=1` forces the no-depth path (tests) |
| 0024b–0024d | a context without depth gets FSR 1 (EASU) of the colour, without history, inside a reversible tonemap that compiles under vkd3d-shader (stopgap, see the 0.8.22 known issues) |
| 0025 | depth buffers in D24S8 or D16 are copied to a 32-bit float image so FSR 4 can read them (Kingdom Come: Deliverance II); `VELA_DEPTH_COPY=1` forces the copy |
| 0026 | motion vectors in R16G16B16A16_SNORM are accepted |
| 0027 | DirectX 11 games: `Dx11Upscaler=sgsr3_12` runs SGSR 3 (FSR 4 VK provider, FSR 4 Fast, the NPU path) over the DX11 -> DX12 bridge; the Vela live file switches DX11 handles like DX12 (fsr4 / fsr4fast -> `sgsr3_12`, fsr3 -> `ffx_12` FSR 3.1, sgsr2 -> `sgsr2_12`, same `Vela live: handle N model X -> ...` lines); a DX11 backend that fails goes to SGSR 2 on Adreno |
| 0028 | an FSR 2 context OptiScaler does not adopt is passed to the game's own FSR 2 (upstream 19e61cf, 299f835; fixes the Bright Memory: Infinite DX11 crash) |
| 0029 | DX11 FSR 2 contexts created on a game D3D11 device are adopted |
| 0030 | the D3D11 context is flushed before the DX12 side reads shared images; `VELA_DX11_NOFLUSH=1` skips it |
| 0031 | `VELA_DX11_PROBE` readback diagnostics of the DX11 -> DX12 shared images |
| 0032 | every DX11 -> DX12 shared image also gets the render-target bind flag: without it DXVK and vkd3d-proton chose different memory layouts and the other side read garbage (black or white picture); `VELA_DX11_SHARED_RT=0` = old |
| 0033 | Unreal's FSR 2 plugin in DX11 (RHI backend: no native textures at the FFX API) is found at the D3D11 compute dispatches of its passes and replaced by OptiScaler's DX11 path; `VELA_UEFSR2=0` = left to the game |
| 0034 | 0033 takes the inputs and output at the shader slots the FFX pipeline gave them |
| 0035 | once adopted, the game's own FSR 2 passes are skipped; `VELA_UEFSR2_SKIP=0` runs them as well |
| 0036 | live model `dittosr`: DittoSR (github.com/Vancete/DittoSR) as the FFX upscaler, loaded from `dittosr/` next to OptiScaler, FSR 3 when it cannot run |
| 0037 | the log says which input API (DLSS, XeSS, FFX, Streamline) created each upscaler context |
| 0038 | Unreal Engine XeSS without a depth input: takes the engine's scene depth (port of upstream OptiScaler 7e94156), on by default, `VELA_UE_XESS_DEPTH=0` turns it off |
| 0039–0048 | see the file names in `patches/` (flat no-depth frames, Vulkan live settings and fallback, SGSR 2 depth view, deferred Vulkan setup, Unity Vulkan outside the render pass, pre-exposure, Vulkan shaders) |
| 0049 | DirectX 11 games whose own FSR 2 hands the FFX backend (FSR 3, DittoSR) D3D11 resources as FSR.reactive / FSR.transparencyAndComposition: they are ignored, the game no longer crashes (Bright Memory: Infinite) |
| 0050 | the model `anf`: ANF as GPU passes (SPIR-V in `AnfSpv.h`, written by us) and the Vulkan bridge `AnfVkBridge.h` that feeds the NPU family graph per frame |
| 0051 | SGSR 2 is shown until the ANF family answers on the NPU (a family loads on its first use) |
| 0052 | resampling for render sizes that are not the 2x canvas (Quality, Balanced, custom ratios) |
| 0053 | GPU timing of the ANF passes for the log |
| 0054 | late NPU: a frame whose NPU answer is not there is finished without it |
| 0055 | faster ANF passes |
| 0056 | the feature channel travels in the alpha of the history |
| 0057 | the FSR 4 Vulkan bridge is split so that native Vulkan games can use the same core |
| 0058 | FSR 4 and FSR 4 Fast in native Vulkan games (NGX Vulkan entry points), chosen by the Vela live file |
| 0059 | FSR 4 Vulkan debug log lines |
| 0039 | `VELA_NODEPTH_FLAT=1` test switch (off by default): FSR 4 with a flat depth instead of FSR 1 for contexts without depth |
| 0040 | Vela live models in native Vulkan games: SGSR 2, FSR 3 (AMD's `amd_fidelityfx_vk.dll`) and DittoSR; FSR 4 picks fall back to SGSR 2 |

## Build

Needs Docker on an x86-64 host and network access (GitHub, Docker Hub, Ubuntu mirrors). From the repository root:

    sh fork/build/build.sh out          # -> out/OptiScaler.dll (IMAGE_FILE_MACHINE_ARM64EC), out/build.log

`build.sh` starts `fork/build/build_in_container.sh` in a throw-away Docker volume. The script checks `patches.md5`,
clones Sloptiscaler at the base commit with its submodules, applies the patches with `git apply --ignore-whitespace`
(upstream files use CRLF) and runs upstream's `arm64ec/build.sh`, which cross-compiles with llvm-mingw.

Pinned inputs:

| input | pin |
|---|---|
| container image | `ubuntu:22.04@sha256:0a9887931b0e38af2e3487ed39e29436b0535c4f3f52e57745baaf79d1898956` (host tools there: cmake 3.22.1, ninja 1.10.1, git 2.34.1, Python 3.10.12; they do not produce code) |
| compiler | llvm-mingw 20260908 (ucrt, ubuntu-22.04-x86_64), tarball sha256 `2258c745e3155870c80793f3e8c80b28fbde11b9ff73c4c78783635b3440b092` |
| sources | Sloptiscaler `f80e4da3` plus its submodules; DirectXMath `f5d82574b03417d905fdd2f85bfa9bf7a2f1deac`; Detours `1dbd4ea281d73d478aa8d09a8b83fbd265f715f7`; freetype tag `VER-2-13-3` |
| source path | the tree is built at `/work/src`; the compiler stores source paths in the objects, so another path changes the DLL |
| build date | OptiScaler stores it in its version resource: today's date, or `FORK_BUILD_DATE=YYYY-MM-DD` |

### Reproducing the release DLL

The `OptiScaler.dll` of Vela 0.8.23 (md5 `f2ff97d36751a40eb20fe8479bd207b4`) is this build with

    FORK_BUILD_DATE=20261122 sh fork/build/build.sh out

A rebuild from a clean clone is byte-identical to the release DLL except two timestamps: the linker's PE timestamp
(`TimeDateStamp` in the COFF header) and the timestamp of the debug directory entry. With those two fields set to the
release values the md5 is the same.

By hand: clone Sloptiscaler, `git checkout f80e4da3`, `git submodule update --init --recursive`, `git apply --ignore-whitespace`
each patch in order, run `./arm64ec/build.sh <deps-dir>` from a checkout located at `/work/src`.
