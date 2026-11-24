# NPU package 18: the ANF graph families

Package 18 (`vela-npu-sysext-18.raw`, Vela 0.9.0, sha256 `b4f2f260be4fc2f9339411fafe480ac7ee505caa34e3e9428550a0754e700807`, file list in
[`../../vela-npu-sysext-18.contents.txt`](../../vela-npu-sysext-18.contents.txt)) adds five HTP v73 context binaries to package 17. The FSR 4 graphs
(main set, `low4/`, `v79/`) are the ones of package 17, see [`../pkg17/README.md`](../pkg17/README.md).

## What ANF is here

ANF is the 2x super-resolution network of Qualcomm's Adreno Neural Fusion SDK
(<https://github.com/SnapdragonGameStudios/adreno-neural-fusion>): four int8 convolutions that run at half the render size
(`cnn1` 20 to 32 channels 3x3 stride 2 with 64 jitter-phase weight sets, `cnn2-0` and `cnn2-1` 32 to 32, `cnn3` 32 to 64 with 64 phase sets;
46 240 parameters, all requantisation multipliers are powers of two). The network outputs, per output pixel, an offset, a blend
weight and a feature; the GPU passes of the fork (patches 0050-0056, our own shaders) do the pre-processing (motion vector pick, history
warp, packing) and the post-processing (sample the render at anchor + offset, blend with the warped history, inverse tonemap).
The network is the same one the SDK uses; only the NPU form (below) and the passes around it are ours.

## What is in the image

| file (in `/usr/lib64/vela-npu/`) | canvas (largest render it serves) | bytes | md5 | sha256 |
|---|---|---|---|---|
| `anf360s32.bin` | 640x360 | 3 964 928 | `29c8c32e783b9b902dba48367ce81c36` | `9c67a6eb2b9617c37d5519e39f8471391412b30ebb63a72341a5d4b278a41f37` |
| `anf480s32.bin` | 854x480 | 4 620 288 | `3f19f1c6dfc2e53dddc113894bc67207` | `bcafaca762a9f1a8ff22194b0981681e3218364cccb0c619c72fbb46b5f18da0` |
| `anf540s32.bin` | 960x540 | 4 751 360 | `96204b7490f18d442bdae34332db58c4` | `08131a1c3ffe5dfb282d6b6a9824e151bb39357f596eb9070e1436bf84757bcf` |
| `anf648s32.bin` | 1152x648 | 5 144 576 | `677d2b026c00d88bcc98ed343fe91689` | `5d387cc934ff3b0f517f28f6209d6247c6e4284a2a47b8a504f979ddfc766696` |
| `anf720s32.bin` | 1280x720 | 5 406 720 | `c7ea5b2c7b63847f8a216a96f614d7e5` | `3709fe0481633de16ebee0b693563b033ba693c09fade6c885011d34588bf0e4` |

Each file is one QNN context binary with 32 graphs `anf<canvasH>_p0` to `_p31`, one per jitter phase (the phase picks the weight set;
the daemon picks the graph from the phase byte the Vulkan layer sends, kick bits 40 to 47). A render smaller than the canvas is written
into the smallest canvas that covers it, with zero padding. The graphs are HTP v73 only (Snapdragon 8 Gen 2); the plugin shows the
model ANF only on that chip, when the NPU is ready and the image is package 18 or newer.

## How they were built

[`gen_anf_onnx.py`](gen_anf_onnx.py) writes an int8 QDQ ONNX graph per phase: activations are uint8 with scale 1 and zero point 0
(value = ANF's int8 + 128), so the zero padding of QNN is ANF's out-of-bounds value -128, and the `-128*sum(w)` and `+128` terms are
folded into each bias exactly. [`build_family.sh`](build_family.sh) (started by [`run_family.sh`](run_family.sh) in the same QAIRT image as
the FSR 4 graphs) runs `qnn-onnx-converter` (`--custom_io`, `--float_fallback --float_bitwidth 16`), `qnn-model-lib-generator`
(`x86_64-linux-clang`) and `qnn-context-binary-generator` (`vtcm_mb 8`, `O 3`, `soc_model 43`, `dsp_arch v73`), QAIRT 2.47.0.260601.
All tensors are fixed point; there is no float operation in the graphs. The scripts keep our local paths; the network data they
read is the SDK's `aiff-sr-v2s-prod-7p1-invpow2` model.

## Checks

- On the x86 HTP simulator and on a Snapdragon 8 Gen 2 console the output hash of a fixed ramp input is the same for every phase of every
  family we ran. The hashes of phase 0 are in `install-vela-npu.sh` (`WANT_ANF`): the installer checks them on a Snapdragon 8 Gen 2
  and says "not checked" on other chips.
- In games (Retroid Pocket 6, Deep Rock Galactic, 1920x1080 from 960x540): ANF 41.5 fps against 38.8 for FSR 4 and 43.9 for SGSR 2;
  the NPU time of ANF is 3.4 ms a frame (FSR 4: about 9 to 11 ms).
- Not checked: Snapdragon 8 Gen 3 and 8 Elite (v79 graphs are not built), a package 18 install by someone other than us.
