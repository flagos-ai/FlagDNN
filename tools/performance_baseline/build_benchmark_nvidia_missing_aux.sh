#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0
# Link existing validation archives only; deliberately do not mutate a shared
# CMake build. First build flagdnn_test_nvidia_adapter and
# flagdnn_validation_nvidia_tensor_io and flagdnn_validation_cpu_reference
# in NVIDIA_BUILD_DIR if they are absent.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${CUDA_ROOT:=/usr/local/cuda}"
: "${NVIDIA_BUILD_DIR:=build/nvidia}"
NVIDIA_BUILD_DIR="$(cd "$NVIDIA_BUILD_DIR" && pwd)"
aux_cache="$NVIDIA_BUILD_DIR/CMakeCache.txt"
cache_value() { sed -n "s|^$1:[^=]*=||p" "$aux_cache" | head -n 1; }
: "${CUDNN_INCLUDE_DIR:=$(cache_value CUDNN_INCLUDE_DIR)}"
: "${CUDNN_LIBRARY:=$(cache_value CUDNN_LIBRARY)}"
: "${CUDNN_FRONTEND_INCLUDE_DIR:=$(cache_value CUDNN_FRONTEND_INCLUDE_DIR)}"
if [[ -n "${CUDNN_ROOT:-}" ]]; then
  CUDNN_INCLUDE_DIR="$CUDNN_ROOT/include"
  CUDNN_LIBRARY="$CUDNN_ROOT/lib/libcudnn.so.9"
fi
aux_cudnn_lib="$(dirname "$CUDNN_LIBRARY")"
aux_adapter="$NVIDIA_BUILD_DIR/backends/nvidia/validation"
aux_archives=(
  "$NVIDIA_BUILD_DIR/reference/cpu/libflagdnn_validation_cpu_reference.a"
  "$aux_adapter/libflagdnn_validation_nvidia_extended.a"
  "$aux_adapter/libflagdnn_test_nvidia_adapter.a"
  "$aux_adapter/libflagdnn_validation_nvidia_attention.a"
  "$aux_adapter/libflagdnn_validation_nvidia_tensor_io.a"
)
for aux_archive in "${aux_archives[@]}"; do
  [[ -f "$aux_archive" ]] || { echo "Missing $aux_archive; build validation dependencies first" >&2; exit 2; }
done
aux_revision="$(git rev-parse HEAD)"
"${CXX:-c++}" -O2 -std=c++20 -Wall -Wextra -ffunction-sections -fdata-sections \
  -DFLAGDNN_AUX_SOURCE_REVISION="\"$aux_revision\"" \
  -I. -Iinclude -Itests -Ibenchmark -Ibackends/nvidia \
  -I"$NVIDIA_BUILD_DIR/generated/include" -I"$CUDA_ROOT/include" \
  -I"$CUDNN_INCLUDE_DIR" -I"$CUDNN_FRONTEND_INCLUDE_DIR" \
  tools/performance_baseline/benchmark_nvidia_missing_aux.cpp \
  -Wl,--start-group "${aux_archives[@]}" -Wl,--end-group \
  -L"$NVIDIA_BUILD_DIR/src" -L"$CUDA_ROOT/lib64" -L"$aux_cudnn_lib" \
  -Wl,--gc-sections -Wl,-rpath,"$NVIDIA_BUILD_DIR/src" \
  -Wl,-rpath,"$CUDA_ROOT/lib64" -Wl,-rpath,"$aux_cudnn_lib" \
  -lflagdnn -l:libcudnn.so.9 -l:libcudnn_ext.so.9 -lcuda -lcudart \
  -lnvrtc -ldl -o "${1:-/tmp/benchmark_nvidia_missing_aux}"
