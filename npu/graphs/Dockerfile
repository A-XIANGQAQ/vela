# QAIRT host tools (x86_64 Linux): ONNX converter, model-library generator, context-binary generator.
# The SDK is not in the image: download QAIRT 2.47.0.260601 from Qualcomm; build.sh mounts it at /qairt.
#   docker build -t vela-qnn .
# Pinned versions are those of the image the package-14 graphs were built with
# (Ubuntu 22.04.5, Python 3.10.12, clang 14.0.0, PyYAML 5.4.1 from Ubuntu).
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq >/dev/null && apt-get install -y -qq python3 python3-pip python3-venv python3-yaml unzip make clang-14 libc++-14-dev libc++abi-14-dev g++ file >/dev/null \
 && pip3 install -q "numpy==1.26.4" "onnx==1.16.2" "onnxruntime==1.18.1" "protobuf==7.36.2" "pandas==2.3.3" "packaging==26.3" "typing-extensions==4.16.0"
