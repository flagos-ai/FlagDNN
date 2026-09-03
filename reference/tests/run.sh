#!/usr/bin/env bash

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_directory="$(cd -- "${script_directory}/../.." && pwd)"
build_directory="${FLAGDNN_REFERENCE_BUILD_DIR:-${FLAGDNN_BUILD_DIR:-build/nvidia}}"
[[ "${build_directory}" == /* ]] || build_directory="${source_directory}/${build_directory}"
cache_file="${build_directory}/CMakeCache.txt"
build_configuration="${FLAGDNN_BUILD_TYPE:-Release}"

fail() {
  echo "error: $*" >&2
  exit 2
}

cache_value() {
  sed -n "s/^${1}:[^=]*=//p" "${cache_file}" | head -n 1
}

[[ -f "${cache_file}" ]] || \
  fail "${build_directory} is not configured; run reference/tests/build.sh"

ctest_command="${FLAGDNN_CTEST_COMMAND:-$(command -v ctest || true)}"
if [[ -z "${ctest_command}" ]]; then
  cmake_command="$(cache_value CMAKE_COMMAND)"
  ctest_command="$(dirname -- "${cmake_command}")/ctest"
elif [[ "${ctest_command}" != */* ]]; then
  ctest_command="$(command -v "${ctest_command}" || true)"
fi
[[ -n "${ctest_command}" && -x "${ctest_command}" ]] || \
  fail "ctest was not found; set FLAGDNN_CTEST_COMMAND"

test_regex='^reference[.]cpu[.]pointwise[.](div|pow|mod|cmp_eq)[.]cudnn$'
echo "Running the four CPU pointwise cuDNN tests from ${build_directory}"
ctest_arguments=(--test-dir "${build_directory}" --output-on-failure
                 --no-tests=error -R "${test_regex}")
if [[ -n "$(cache_value CMAKE_CONFIGURATION_TYPES)" ]]; then
  ctest_arguments+=(-C "${build_configuration}")
fi
ctest_arguments+=("$@")
exec "${ctest_command}" "${ctest_arguments[@]}"
