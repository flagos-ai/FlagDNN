#!/usr/bin/env bash
# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0
# Reproduce on the measurement host (omit the trailing precision for core):
# CUDNN_ROOT=/home/wangbingjie/py312-flagdnn/lib/python3.12/site-packages/nvidia/cudnn \
# CUDNN_FRONTEND_INCLUDE_DIR=/home/wangbingjie/py312-flagdnn/lib/python3.12/site-packages/include \
# PATH=/home/wangbingjie/py312-flagdnn/bin:$PATH \
# bash tools/performance_baseline/build_benchmark_nvidia_cudnn.sh /tmp/benchmark_nvidia_precision precision
# GPU-free verification: /tmp/benchmark_nvidia_precision --self-test
# Exclusive GPU execution: /tmp/benchmark_nvidia_precision > precision.jsonl 2> precision.log
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CUDA_ROOT:=/usr/local/cuda}"
: "${CUDNN_ROOT:?Set CUDNN_ROOT to a directory containing include/ and lib/}"
: "${CUDNN_FRONTEND_INCLUDE_DIR:?Set CUDNN_FRONTEND_INCLUDE_DIR to the directory containing cudnn_frontend.h}"
: "${NVIDIA_BUILD_DIR:=build/nvidia}"
NVIDIA_BUILD_DIR="$(cd "$NVIDIA_BUILD_DIR" && pwd)"
cmake --build "$NVIDIA_BUILD_DIR" --target flagdnn_benchmark_nvidia_adapter \
  flagdnn_validation_nvidia_tensor_io flagdnn_validation_cpu_reference flagdnn_benchmark_cases
NV_ADAPTER="$NVIDIA_BUILD_DIR/backends/nvidia/validation"
BENCHMARK_SOURCE=tools/performance_baseline/benchmark_nvidia_cudnn.cpp
if [[ ${2:-core} == precision ]]; then
  BENCHMARK_SOURCE=tools/performance_baseline/benchmark_nvidia_precision.cpp
fi
c++ -O2 -std=c++20 -ffunction-sections -fdata-sections \
  -I. -Iinclude -Itests -Ibenchmark -Ibackends/nvidia -I"$NVIDIA_BUILD_DIR/generated/include" \
  -I"$CUDA_ROOT/include" -I"$CUDNN_ROOT/include" -I"$CUDNN_FRONTEND_INCLUDE_DIR" \
  "$BENCHMARK_SOURCE" \
  "$NVIDIA_BUILD_DIR/benchmark/libflagdnn_benchmark_cases.a" \
  "$NVIDIA_BUILD_DIR/reference/cpu/libflagdnn_validation_cpu_reference.a" \
  "$NV_ADAPTER/libflagdnn_benchmark_nvidia_adapter.a" \
  "$NV_ADAPTER/libflagdnn_validation_nvidia_extended.a" \
  "$NV_ADAPTER/libflagdnn_validation_nvidia_attention.a" \
  "$NV_ADAPTER/libflagdnn_validation_nvidia_tensor_io.a" \
  -L"$NVIDIA_BUILD_DIR/src" -L"$CUDA_ROOT/lib64" -L"$CUDNN_ROOT/lib" \
  -Wl,--gc-sections -Wl,-rpath,"$NVIDIA_BUILD_DIR/src" \
  -Wl,-rpath,"$CUDA_ROOT/lib64" -Wl,-rpath,"$CUDNN_ROOT/lib" \
  -lflagdnn -l:libcudnn.so.9 -l:libcudnn_ext.so.9 -lcuda -lcudart \
  -lnvrtc -ldl -o "${1:-/tmp/benchmark_nvidia_cudnn}"
