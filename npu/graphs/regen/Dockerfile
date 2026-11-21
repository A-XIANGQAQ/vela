# SPIR-V tools for regenerating the parameter files (params_n1.pkl, k1_w.npz, p13_w.npz) from the fsr4vk shaders.
#   docker build -t vela-spv npu/graphs/regen
# The parsers match the exact text that this SPIRV-Cross commit writes, so keep every version below as it is.
# numpy 2 is needed: parse_k1.py relies on its integer type rules (numpy 1.x widens uint32 to uint64 there).
FROM ubuntu:24.04@sha256:fe81560990fae86dc4743202b2cb4d3df98d0bcf949173cecef15ce1127ef852
ENV DEBIAN_FRONTEND=noninteractive
ARG SPIRV_CROSS=57e4efcc84112abcd426dc5ec248b5339bee04f2
RUN apt-get update -qq >/dev/null \
 && apt-get install -y -qq --no-install-recommends ca-certificates git cmake make g++ \
      "spirv-tools=2025.1~rc1-1~ubuntu0.24.04.2" "python3=3.12.3-0ubuntu2.1" "python3-pip=24.0+dfsg-1ubuntu1.3" >/dev/null \
 && pip3 install -q --break-system-packages "numpy==2.5.3" \
 && git clone -q https://github.com/KhronosGroup/SPIRV-Cross /sc && git -C /sc checkout -q $SPIRV_CROSS \
 && cmake -S /sc -B /sc/b -DCMAKE_BUILD_TYPE=Release -DSPIRV_CROSS_ENABLE_TESTS=OFF >/dev/null \
 && make -C /sc/b -j4 spirv-cross >/dev/null && cp /sc/b/spirv-cross /usr/local/bin/spirv-cross \
 && spirv-cross --revision && spirv-as --version
