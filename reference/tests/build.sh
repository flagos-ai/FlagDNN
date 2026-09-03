#!/usr/bin/env bash

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_directory="$(cd -- "${script_directory}/../.." && pwd)"
build_directory="${FLAGDNN_REFERENCE_BUILD_DIR:-${FLAGDNN_BUILD_DIR:-build/nvidia}}"
[[ "${build_directory}" == /* ]] || build_directory="${source_directory}/${build_directory}"
cache_file="${build_directory}/CMakeCache.txt"
target_name="flagdnn_reference_cpu_pointwise_cudnn"
build_configuration="${FLAGDNN_BUILD_TYPE:-Release}"

fail() {
  echo "error: $*" >&2
  exit 2
}

cache_value() {
  sed -n "s/^${1}:[^=]*=//p" "${cache_file}" | head -n 1
}

jobs="${FLAGDNN_BUILD_JOBS:-4}"
[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || fail "FLAGDNN_BUILD_JOBS must be positive"

python_command="${FLAGDNN_CODEGEN_PYTHON:-}"
if [[ -z "${python_command}" && -f "${cache_file}" ]]; then
  python_command="$(cache_value FLAGDNN_CODEGEN_PYTHON)"
fi
if [[ -z "${python_command}" ]]; then
  python_command="$(command -v python3 || true)"
elif [[ "${python_command}" != */* ]]; then
  python_command="$(command -v "${python_command}" || true)"
fi

cmake_command="${FLAGDNN_CMAKE_COMMAND:-}"
if [[ -z "${cmake_command}" && -f "${cache_file}" ]]; then
  cmake_command="$(cache_value CMAKE_COMMAND)"
fi
if [[ -z "${cmake_command}" ]]; then
  cmake_command="$(command -v cmake || true)"
fi
if [[ -z "${cmake_command}" && -x "${python_command}" ]]; then
  cmake_command="$("${python_command}" -c \
    'import sysconfig; print(sysconfig.get_path("purelib") + "/cmake/data/bin/cmake")' \
    2>/dev/null || true)"
fi
if [[ "${cmake_command}" != */* ]]; then
  cmake_command="$(command -v "${cmake_command}" || true)"
fi
[[ -n "${cmake_command}" && -x "${cmake_command}" ]] || \
  fail "cmake was not found; set FLAGDNN_CMAKE_COMMAND"

configure_arguments=(
  -S "${source_directory}"
  -B "${build_directory}"
  -DFLAGDNN_BUILD_TESTS=ON
  "-DCMAKE_BUILD_TYPE=${build_configuration}")
if [[ -f "${cache_file}" ]]; then
  cached_backends="$(cache_value FLAGDNN_BACKENDS)"
  [[ ";${cached_backends};" == *";nvidia;"* ]] || \
    fail "${build_directory} is not configured with the NVIDIA backend"
else
  [[ -n "${python_command}" && -x "${python_command}" ]] || \
    fail "python3 was not found; set FLAGDNN_CODEGEN_PYTHON"
  configure_arguments+=(
    -DFLAGDNN_BACKENDS=nvidia
    "-DFLAGDNN_CODEGEN_PYTHON=${python_command}")
fi

echo "Configuring reference tests in ${build_directory}"
"${cmake_command}" "${configure_arguments[@]}" "$@"

echo "Building ${target_name}"
build_arguments=(--build "${build_directory}" --target "${target_name}"
                 --parallel "${jobs}")
if [[ -n "$(cache_value CMAKE_CONFIGURATION_TYPES)" ]]; then
  build_arguments+=(--config "${build_configuration}")
fi
"${cmake_command}" "${build_arguments[@]}"
