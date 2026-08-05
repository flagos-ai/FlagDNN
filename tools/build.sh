#!/usr/bin/env bash

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_directory="$(cd -- "${script_directory}/.." && pwd)"

usage() {
  cat <<'EOF'
Build FlagDNN with CMake.

Usage:
  tools/build.sh [options] [-- <extra CMake arguments>]

Options:
  --build-dir PATH          Build directory (default: build/<backend>)
  --build-type TYPE         CMake build type (default: Release)
  --backends LIST           Semicolon-separated backends (default: nvidia)
  --engine NAME             libtriton_jit or external_artifact
  --python PATH             Python used by the external platform compiler
  --generator NAME          CMake generator (default: Ninja)
  --jobs N                  Parallel build jobs (default: up to 8)
  --tests / --no-tests      Enable or disable functional tests
  --benchmarks / --no-benchmarks
                            Enable or disable performance benchmarks
  --warnings-as-errors / --no-warnings-as-errors
                            Enable or disable -Werror
  --configure-only          Configure without compiling
  -h, --help                Show this help

Environment defaults:
  FLAGDNN_BUILD_DIR, FLAGDNN_BUILD_TYPE, FLAGDNN_BACKENDS,
  FLAGDNN_EXECUTION_ENGINE, FLAGDNN_CODEGEN_PYTHON,
  FLAGDNN_BUILD_TESTS, FLAGDNN_BUILD_BENCHMARKS,
  FLAGDNN_WARNINGS_AS_ERRORS, FLAGDNN_BUILD_JOBS.

Examples:
  tools/build.sh
  tools/build.sh --no-benchmarks
  tools/build.sh --backends ascend
  tools/build.sh --build-dir /tmp/flagdnn-build --jobs 4
  tools/build.sh -- --trace-expand -DPLATFORM_OPTION=value
EOF
}

fail() {
  echo "error: $*" >&2
  exit 2
}

require_value() {
  if [[ $# -lt 2 || -z "${2}" ]]; then
    fail "${1} requires a value"
  fi
}

absolute_from_source() {
  if [[ "${1}" == /* ]]; then
    printf '%s\n' "${1}"
  else
    printf '%s/%s\n' "${source_directory}" "${1}"
  fi
}

normalize_bool() {
  case "${1,,}" in
    1|on|true|yes) printf 'ON\n' ;;
    0|off|false|no) printf 'OFF\n' ;;
    *) fail "expected a boolean value, got '${1}'" ;;
  esac
}

default_jobs=8
if command -v nproc >/dev/null 2>&1; then
  processor_count="$(nproc)"
  if [[ "${processor_count}" =~ ^[1-9][0-9]*$ ]] &&
     (( processor_count < default_jobs )); then
    default_jobs="${processor_count}"
  fi
fi

build_directory="${FLAGDNN_BUILD_DIR:-}"
build_type="${FLAGDNN_BUILD_TYPE:-Release}"
backends="${FLAGDNN_BACKENDS:-nvidia}"
execution_engine="${FLAGDNN_EXECUTION_ENGINE:-libtriton_jit}"
codegen_python="${FLAGDNN_CODEGEN_PYTHON:-}"
generator="${FLAGDNN_CMAKE_GENERATOR:-Ninja}"
jobs="${FLAGDNN_BUILD_JOBS:-${default_jobs}}"
build_tests="$(normalize_bool "${FLAGDNN_BUILD_TESTS:-ON}")"
build_benchmarks="$(normalize_bool "${FLAGDNN_BUILD_BENCHMARKS:-ON}")"
warnings_as_errors="$(normalize_bool "${FLAGDNN_WARNINGS_AS_ERRORS:-ON}")"
configure_only=0
extra_cmake_arguments=()

while [[ $# -gt 0 ]]; do
  case "${1}" in
    --build-dir)
      require_value "$@"
      build_directory="${2}"
      shift 2
      ;;
    --build-type)
      require_value "$@"
      build_type="${2}"
      shift 2
      ;;
    --backends)
      require_value "$@"
      backends="${2}"
      if [[ "${backends}" == "none" ]]; then
        backends=""
      fi
      shift 2
      ;;
    --engine)
      require_value "$@"
      execution_engine="${2}"
      shift 2
      ;;
    --python)
      require_value "$@"
      codegen_python="${2}"
      shift 2
      ;;
    --generator)
      require_value "$@"
      generator="${2}"
      shift 2
      ;;
    --jobs)
      require_value "$@"
      jobs="${2}"
      shift 2
      ;;
    --tests)
      build_tests=ON
      shift
      ;;
    --no-tests)
      build_tests=OFF
      shift
      ;;
    --benchmarks)
      build_benchmarks=ON
      shift
      ;;
    --no-benchmarks)
      build_benchmarks=OFF
      shift
      ;;
    --warnings-as-errors)
      warnings_as_errors=ON
      shift
      ;;
    --no-warnings-as-errors)
      warnings_as_errors=OFF
      shift
      ;;
    --configure-only)
      configure_only=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      extra_cmake_arguments+=("$@")
      break
      ;;
    *)
      fail "unknown option '${1}'; use --help for usage"
      ;;
  esac
done

[[ "${jobs}" =~ ^[1-9][0-9]*$ ]] || fail "--jobs must be a positive integer"
case "${execution_engine}" in
  libtriton_jit|external_artifact) ;;
  *) fail "--engine must be libtriton_jit or external_artifact" ;;
esac

command -v cmake >/dev/null 2>&1 ||
  fail "cmake was not found; install CMake 3.23 or newer"
if [[ "${generator}" == "Ninja" ]]; then
  command -v ninja >/dev/null 2>&1 ||
    fail "ninja was not found; install Ninja or select another generator"
fi

if [[ -z "${codegen_python}" ]]; then
  codegen_python="$(command -v python3 || true)"
fi
[[ -n "${codegen_python}" ]] ||
  fail "python3 was not found; pass --python PATH"
if [[ "${codegen_python}" != */* ]]; then
  codegen_python="$(command -v "${codegen_python}" || true)"
fi
[[ -x "${codegen_python}" ]] ||
  fail "codegen Python is not executable: ${codegen_python}"

if [[ "${backends}" == "none" ]]; then
  backends=""
fi
if [[ -z "${build_directory}" ]]; then
  if [[ -z "${backends}" ]]; then
    build_directory="build/core"
  elif [[ "${backends}" == *";"* ]]; then
    build_directory="build/multi"
  elif [[ "${backends}" =~ ^[a-z][a-z0-9_]*$ ]]; then
    build_directory="build/${backends}"
  else
    fail "--backends must contain CMake backend names"
  fi
fi

build_directory="$(absolute_from_source "${build_directory}")"

cmake_arguments=(
  -S "${source_directory}"
  -B "${build_directory}"
  -G "${generator}"
  "-DCMAKE_BUILD_TYPE=${build_type}"
  "-DFLAGDNN_BACKENDS=${backends}"
  "-DFLAGDNN_BUILD_TESTS=${build_tests}"
  "-DFLAGDNN_BUILD_BENCHMARKS=${build_benchmarks}"
  "-DFLAGDNN_EXECUTION_ENGINE=${execution_engine}"
  "-DFLAGDNN_CODEGEN_PYTHON=${codegen_python}"
  "-DFLAGDNN_WARNINGS_AS_ERRORS=${warnings_as_errors}"
)

cmake_arguments+=("${extra_cmake_arguments[@]}")

echo "FlagDNN build configuration"
echo "  source:      ${source_directory}"
echo "  build:       ${build_directory}"
echo "  type:        ${build_type}"
echo "  backends:    ${backends:-<none>}"
echo "  engine:      ${execution_engine}"
echo "  tests:       ${build_tests}"
echo "  benchmarks:  ${build_benchmarks}"
echo "  python:      ${codegen_python}"
echo "  jobs:        ${jobs}"

cmake "${cmake_arguments[@]}"

if (( configure_only )); then
  echo "Configured FlagDNN at ${build_directory}"
  exit 0
fi

cmake --build "${build_directory}" --parallel "${jobs}"

echo "FlagDNN build completed"
echo "  core library: ${build_directory}/src/libflagdnn.so"
if [[ -n "${backends}" ]]; then
  IFS=';' read -r -a backend_names <<< "${backends}"
  for backend_name in "${backend_names[@]}"; do
    echo "  ${backend_name} plugin: ${build_directory}/backends/${backend_name}/libflagdnn_backend_${backend_name}.so"
  done
fi
echo "Install with: tools/install.sh --build-dir ${build_directory}"
