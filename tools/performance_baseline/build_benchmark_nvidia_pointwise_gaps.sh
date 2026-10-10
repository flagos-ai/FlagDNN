#!/usr/bin/env bash
# Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CUDA_ROOT:=/usr/local/cuda}"
: "${CUDNN_ROOT:?Set CUDNN_ROOT to a directory containing include/ and lib/}"
: "${CUDNN_FRONTEND_INCLUDE_DIR:?Set CUDNN_FRONTEND_INCLUDE_DIR}"
: "${NVIDIA_BUILD_DIR:=build/nvidia}"
NVIDIA_BUILD_DIR="$(cd "$NVIDIA_BUILD_DIR" && pwd)"
cmake --build "$NVIDIA_BUILD_DIR" --target flagdnn_test_nvidia_adapter \
  flagdnn_validation_nvidia_tensor_io flagdnn_validation_cpu_reference
NV_ADAPTER="$NVIDIA_BUILD_DIR/backends/nvidia/validation"
c++ -O2 -std=c++20 -ffunction-sections -fdata-sections \
  -I. -Iinclude -Itests -Ibenchmark -Ibackends/nvidia \
  -I"$NVIDIA_BUILD_DIR/generated/include" -I"$CUDA_ROOT/include" \
  -I"$CUDNN_ROOT/include" -I"$CUDNN_FRONTEND_INCLUDE_DIR" \
  tools/performance_baseline/benchmark_nvidia_pointwise_gaps.cpp \
  -Wl,--start-group \
  "$NVIDIA_BUILD_DIR/reference/cpu/libflagdnn_validation_cpu_reference.a" \
  "$NV_ADAPTER/libflagdnn_validation_nvidia_extended.a" \
  "$NV_ADAPTER/libflagdnn_test_nvidia_adapter.a" \
  "$NV_ADAPTER/libflagdnn_validation_nvidia_attention.a" \
  "$NV_ADAPTER/libflagdnn_validation_nvidia_tensor_io.a" \
  -Wl,--end-group \
  -L"$NVIDIA_BUILD_DIR/src" -L"$CUDA_ROOT/lib64" -L"$CUDNN_ROOT/lib" \
  -Wl,--gc-sections -Wl,-rpath,"$NVIDIA_BUILD_DIR/src" \
  -Wl,-rpath,"$CUDA_ROOT/lib64" -Wl,-rpath,"$CUDNN_ROOT/lib" \
  -lflagdnn -l:libcudnn.so.9 -l:libcudnn_ext.so.9 -lcuda -lcudart \
  -lnvrtc -ldl -o "${1:-/tmp/benchmark_nvidia_pointwise_gaps}"
