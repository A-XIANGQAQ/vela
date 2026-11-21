#!/bin/sh
# Inside the vela-spv container: decompile the fsr4vk passes one preset needs, for the parsers.
#   decompile.sh PRESET OUT_DIR       (PRESET = performance | ultraperf)
# Mounts: /fsr4vk = fsr4vk checkout (read-only), /tools = provider/spirv-sources/tools of this repository (read-only).
# Writes OUT_DIR/o/pNN.comp      spirv-cross --vulkan-semantics of the original pass-NN.spv (passes 03-27, odd)
#        OUT_DIR/t/qNN.comp      the same of the transformed pass (passes 15, 17, 19, 27):
#          spirv-dis -> sudot.py (every OpSDot becomes OpSUDot plus a correction term, exact modulo 2^32)
#                    -> zconst.py (every read of gl_WorkGroupID.z / gl_GlobalInvocationID.z becomes the constant 0)
#                    -> every OpLoopMerge gets the Unroll hint -> spirv-as -> spirv-val -> spirv-cross
#        The transform does not change the math; it gives straight-line GLSL that the parsers can read.
set -eu
P=$1; O=$2
A=/fsr4vk/assets/fsr411/general/1080/$P
T=$(mktemp -d)
mkdir -p $O/o $O/t
for n in 03 05 07 09 11 13 15 17 19 21 23 25 27; do
  s=$A/pass-$n.spv
  spirv-cross --vulkan-semantics $s --output $O/o/p$n.comp
  case $n in 15|17|19|27)
    spirv-dis $s -o $T/m0
    python3 /tools/sudot.py $T/m0 $T/m1 >/dev/null
    python3 /tools/zconst.py $T/m1 $T/m2 >/dev/null
    sed -i -E 's/OpLoopMerge (%[0-9A-Za-z_]+) (%[0-9A-Za-z_]+) None/OpLoopMerge \1 \2 Unroll/' $T/m2
    spirv-as --target-env vulkan1.1 $T/m2 -o $T/q.spv
    spirv-val --target-env vulkan1.1 $T/q.spv
    spirv-cross --vulkan-semantics $T/q.spv --output $O/t/q$n.comp;;
  esac
  echo "$P pass-$n $(md5sum < $s | cut -c1-12)"
done
