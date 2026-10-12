#!/usr/bin/env bash
# Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0
# Build diagnostic executables only; this script never initializes a GPU.
# Usage: bash tools/hipdnn_capability/build_probes.sh OUTPUT_DIR [HYGON_BUILD_DIR]
# Set CXX, DTK_ROOT or HIPDNN_ROOT when the SDK is not installed under /opt/dtk.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 OUTPUT_DIR [HYGON_BUILD_DIR]" >&2
  exit 2
fi
probe_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
probe_build=$(cd "${2:-$probe_repo/build/hygon}" && pwd)
mkdir -p "$1"
probe_output=$(cd "$1" && pwd)
probe_dtk=${DTK_ROOT:-/opt/dtk}
probe_hipdnn=${HIPDNN_ROOT:-$probe_dtk/hipdnn}
probe_cxx=${CXX:-c++}

probe_cases="$probe_build/benchmark/libflagdnn_benchmark_cases.a"
probe_cpu="$probe_build/reference/cpu/libflagdnn_validation_cpu_reference.a"
probe_reference="$probe_build/backends/hygon/validation/libflagdnn_validation_hygon_reference.a"
probe_reduction_cases="$probe_build/tests/CMakeFiles/flagdnn_test_common_objects.dir/common/reduction.cpp.o"
for probe_dependency in "$probe_cases" "$probe_cpu" "$probe_reference" \
  "$probe_reduction_cases" "$probe_hipdnn/include/hipdnn.h"; do
  if [[ ! -f "$probe_dependency" ]]; then
    echo "missing dependency: $probe_dependency; build the Hygon validation targets first" >&2
    exit 2
  fi
done

probe_flags=(
  -std=c++20 -O2 -Wall -Wextra -Werror -ffunction-sections -fdata-sections
  -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -D__HIP_ROCclr__=1
  -I"$probe_repo" -I"$probe_repo/include" -I"$probe_repo/benchmark"
  -I"$probe_repo/tests" -I"$probe_build/generated/include"
  -I"$probe_repo/backends/hygon/validation"
  -I"$probe_hipdnn/include" -isystem "$probe_dtk/include"
  -isystem "$probe_dtk/hip/include"
)
probe_vendor_links=(
  -Wl,--gc-sections -L"$probe_build/src" -L"$probe_hipdnn/lib" -L"$probe_dtk/lib"
  -Wl,-rpath,"$probe_build/src" -Wl,-rpath,"$probe_hipdnn/lib"
  -Wl,-rpath,"$probe_dtk/lib" -lflagdnn -lhipdnn -lamdhip64
)
probe_source="$probe_repo/tools/hipdnn_capability"
probe_io="$probe_repo/backends/hygon/validation/tensor_io.cpp"

"$probe_cxx" "${probe_flags[@]}" "$probe_source/pointwise_probe.cpp" \
  "$probe_repo/tests/common/pointwise.cpp" "$probe_io" \
  "$probe_repo/backends/hygon/validation/activation_layout.cpp" \
  "$probe_cases" "$probe_cpu" "${probe_vendor_links[@]}" \
  -o "$probe_output/pointwise_probe"

"$probe_cxx" "${probe_flags[@]}" "$probe_source/batchnorm_probe.cpp" \
  "$probe_io" "$probe_cases" "$probe_cpu" "${probe_vendor_links[@]}" \
  -o "$probe_output/batchnorm_probe"

"$probe_cxx" "${probe_flags[@]}" "$probe_source/reduction_probe.cpp" \
  "$probe_reduction_cases" "$probe_cpu" "$probe_reference" \
  "${probe_vendor_links[@]}" -o "$probe_output/reduction_probe"

# The oracle/poison contract is genuinely CPU-only: no HIP/hipDNN link.
"$probe_cxx" "${probe_flags[@]}" "$probe_source/pointwise_probe_contract.cpp" \
  "$probe_repo/tests/common/pointwise.cpp" "$probe_io" "$probe_cases" \
  "$probe_cpu" -Wl,--gc-sections -o "$probe_output/pointwise_probe_contract"

printf 'Built correctness probes in %s\n' "$probe_output"
printf 'CPU check: %s/pointwise_probe_contract\n' "$probe_output"
printf 'Each GPU probe supports --list and --case NAME --device N. Run cases in separate processes.\n'
