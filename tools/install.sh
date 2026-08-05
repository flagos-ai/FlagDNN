#!/usr/bin/env bash

# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_directory="$(cd -- "${script_directory}/.." && pwd)"

usage() {
  cat <<'EOF'
Install a configured FlagDNN build as a complete SDK tree.

Usage:
  tools/install.sh [options]

Options:
  --build-dir PATH   Configured build directory (default: build/<backend>)
  --prefix PATH      Install prefix (default: <build-dir>/install)
  --config NAME      Multi-config build configuration (default: Release)
  --strip            Strip installed binaries
  -h, --help         Show this help

Environment defaults:
  FLAGDNN_BUILD_DIR, FLAGDNN_INSTALL_PREFIX, FLAGDNN_BUILD_TYPE,
  FLAGDNN_BACKENDS.

Examples:
  tools/install.sh
  FLAGDNN_BACKENDS=ascend tools/install.sh
  tools/install.sh --prefix /opt/flagdnn
  tools/install.sh --build-dir /tmp/flagdnn-build --prefix /tmp/flagdnn-sdk
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

build_directory="${FLAGDNN_BUILD_DIR:-}"
install_prefix="${FLAGDNN_INSTALL_PREFIX:-}"
build_config="${FLAGDNN_BUILD_TYPE:-Release}"
selected_backends="${FLAGDNN_BACKENDS:-nvidia}"
strip_install=0

while [[ $# -gt 0 ]]; do
  case "${1}" in
    --build-dir)
      require_value "$@"
      build_directory="${2}"
      shift 2
      ;;
    --prefix)
      require_value "$@"
      install_prefix="${2}"
      shift 2
      ;;
    --config)
      require_value "$@"
      build_config="${2}"
      shift 2
      ;;
    --strip)
      strip_install=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown option '${1}'; use --help for usage"
      ;;
  esac
done

command -v cmake >/dev/null 2>&1 ||
  fail "cmake was not found; install CMake 3.23 or newer"

if [[ "${selected_backends}" == "none" ]]; then
  selected_backends=""
fi
if [[ -z "${build_directory}" ]]; then
  if [[ -z "${selected_backends}" ]]; then
    build_directory="build/core"
  elif [[ "${selected_backends}" == *";"* ]]; then
    build_directory="build/multi"
  elif [[ "${selected_backends}" =~ ^[a-z][a-z0-9_]*$ ]]; then
    build_directory="build/${selected_backends}"
  else
    fail "FLAGDNN_BACKENDS must contain CMake backend names"
  fi
fi

build_directory="$(absolute_from_source "${build_directory}")"
if [[ -z "${install_prefix}" ]]; then
  install_prefix="${build_directory}/install"
else
  install_prefix="$(absolute_from_source "${install_prefix}")"
fi

[[ "${install_prefix}" != "/" ]] || fail "refusing to install into /"
[[ -f "${build_directory}/CMakeCache.txt" ]] ||
  fail "${build_directory} is not configured; run tools/build.sh first"
[[ -e "${build_directory}/src/libflagdnn.so" ]] ||
  fail "libflagdnn.so is missing; compile the configured build first"

install_arguments=(
  --install "${build_directory}"
  --prefix "${install_prefix}"
  --config "${build_config}"
)
if (( strip_install )); then
  install_arguments+=(--strip)
fi

echo "Installing FlagDNN"
echo "  build:  ${build_directory}"
echo "  prefix: ${install_prefix}"

cmake "${install_arguments[@]}"

echo "FlagDNN SDK installed"
echo "  headers:   ${install_prefix}/include"
echo "  libraries: ${install_prefix}/lib"
echo "  CMake:     ${install_prefix}/lib/cmake/FlagDNN"
echo "  resources: ${install_prefix}/share/flagdnn"
