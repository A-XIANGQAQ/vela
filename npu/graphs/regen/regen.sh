#!/bin/bash
# Regenerate the parameter files of npu/graphs/params/ from the fsr4vk shaders, then compare them with the committed ones.
#
#   FSR4VK_DIR=<fsr4vk checkout> OUT_DIR=<output> npu/graphs/regen/regen.sh [performance] [ultraperf]
#
# FSR4VK_DIR  a checkout of https://github.com/dvj5411/fsr4vk at 391f62bd198ac9a289aba8ee0d4396db2104f3a2, mounted read-only
# OUT_DIR     receives <preset>/dec/ (decompiled passes) and <preset>/{params_n1.pkl, p13_w.npz, k1_w.npz[, p07_sh.npy]}
# IMAGE       Docker image built from ./Dockerfile (default: vela-spv)
# CPUS, MEM   container limits (default: 4, 8g)
# Then: python3 npu/graphs/regen/check.py "$OUT_DIR"   (exit 1 if a file differs from npu/graphs/params/)
# On Windows, run it from Git Bash and give the folders as Windows paths with forward slashes.
set -eu -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd)
: "${FSR4VK_DIR:?set FSR4VK_DIR}" "${OUT_DIR:?set OUT_DIR}"
IMAGE=${IMAGE:-vela-spv}; CPUS=${CPUS:-4}; MEM=${MEM:-8g}
[ $# -gt 0 ] || set -- performance ultraperf
mkdir -p "$OUT_DIR"
for P in "$@"; do
  echo "== $P"
  # MSYS_NO_PATHCONV stops Git Bash on Windows from rewriting the container paths; elsewhere it does nothing.
  MSYS_NO_PATHCONV=1 docker run --rm --cpus "$CPUS" -m "$MEM" \
    -v "$HERE:/regen:ro" -v "$REPO/provider/spirv-sources/tools:/tools:ro" -v "$FSR4VK_DIR:/fsr4vk:ro" -v "$OUT_DIR:/out" \
    "$IMAGE" sh -c "set -e
      I=/fsr4vk/assets/fsr411/general/1080/$P/initializers.bin; D=/out/$P
      sh /regen/decompile.sh $P \$D/dec
      python3 /regen/shader_params.py $P \$D/dec \$I \$D
      python3 /regen/parse_p13.py \$D/dec/o/p13.comp \$I \$D/p13_w.npz
      python3 /regen/parse_k1.py \$D/dec/t/q27.comp \$D" 2>&1 | tee "$OUT_DIR/$P.log"
done
