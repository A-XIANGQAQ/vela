# QAIRT host tools (x86_64 Linux): ONNX converter, model-library generator, context-binary generator.
# The SDK is not in the image: download QAIRT 2.47.0.260601 from Qualcomm; build.sh mounts it at /qairt.
#   docker build -t vela-qnn .
# Pinned: the Ubuntu 22.04.5 base image by digest and the pip packages; the Ubuntu packages (Python 3.10.12, clang 14.0.0,
# PyYAML 5.4.1) come from the 22.04 archive. Built with --no-cache on 2026-11-22, this image rebuilt nf4_exact and
# nf4up_k1h_452 for v73 and v79 with the sha256 of expected.sha256. Against the package-14 image only patch releases
# differ (glibc 2.35-0ubuntu3.18 vs 3.15, perl, ca-certificates, linux-libc-dev); the pip packages are the same.
FROM ubuntu:22.04@sha256:0a9887931b0e38af2e3487ed39e29436b0535c4f3f52e57745baaf79d1898956
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq >/dev/null && apt-get install -y -qq python3 python3-pip python3-venv python3-yaml unzip make clang-14 libc++-14-dev libc++abi-14-dev g++ file >/dev/null \
 && pip3 install -q "numpy==1.26.4" "onnx==1.16.2" "onnxruntime==1.18.1" "protobuf==7.36.2" "pandas==2.3.3" "packaging==26.3" "typing-extensions==4.16.0"
