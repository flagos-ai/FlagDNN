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
  --build-dir PATH   Configured build directory (default: the sole build/*
                     tree, or build/nvidia when none exists yet)
  --prefix PATH      Install prefix (default: <build-dir>/install)
  --config NAME      Build configuration (default: the configuration recorded
                     by tools/build.sh, CMAKE_BUILD_TYPE, or Release for a
                     multi-config build)
  --strip            Strip installed binaries
  -h, --help         Show this help

Environment defaults:
  FLAGDNN_BUILD_DIR, FLAGDNN_INSTALL_PREFIX, FLAGDNN_BUILD_TYPE,
  FLAGDNN_BACKENDS.

Examples:
  tools/build.sh --backends hygon
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
build_config="${FLAGDNN_BUILD_TYPE:-}"
selected_backends="${FLAGDNN_BACKENDS:-}"
backends_explicit=0
if [[ "${FLAGDNN_BACKENDS+x}" == "x" ]]; then
  backends_explicit=1
fi
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
command -v realpath >/dev/null 2>&1 ||
  fail "realpath was not found; it is required for safe prefix validation"

if [[ "${selected_backends}" == "none" ]]; then
  selected_backends=""
fi
if [[ -z "${build_directory}" ]]; then
  if (( backends_explicit )); then
    if [[ -z "${selected_backends}" ]]; then
      build_directory="build/core"
    elif [[ "${selected_backends}" == *";"* ]]; then
      build_directory="build/multi"
    elif [[ "${selected_backends}" =~ ^[a-z][a-z0-9_]*$ ]]; then
      build_directory="build/${selected_backends}"
    else
      fail "FLAGDNN_BACKENDS must contain CMake backend names"
    fi
  else
    shopt -s nullglob
    configured_caches=("${source_directory}"/build/*/CMakeCache.txt)
    shopt -u nullglob
    if (( ${#configured_caches[@]} == 1 )); then
      build_directory="${configured_caches[0]%/CMakeCache.txt}"
    elif (( ${#configured_caches[@]} == 0 )); then
      build_directory="build/nvidia"
    else
      fail "multiple configured build trees found; pass --build-dir or set FLAGDNN_BACKENDS"
    fi
  fi
fi

build_directory="$(realpath -m -- "$(absolute_from_source "${build_directory}")")"
if [[ -z "${install_prefix}" ]]; then
  install_prefix="${build_directory}/install"
else
  install_prefix="$(absolute_from_source "${install_prefix}")"
fi
install_prefix="$(realpath -m -- "${install_prefix}")"

cached_build_type=""
configuration_types=""
if [[ -f "${build_directory}/CMakeCache.txt" ]]; then
  while IFS= read -r cache_line; do
    case "${cache_line}" in
      CMAKE_BUILD_TYPE:*=*)
        cached_build_type="${cache_line#*=}"
        ;;
      CMAKE_CONFIGURATION_TYPES:*=*)
        configuration_types="${cache_line#*=}"
        ;;
    esac
  done < "${build_directory}/CMakeCache.txt"
fi
if [[ -z "${build_config}" && -n "${cached_build_type}" ]]; then
  build_config="${cached_build_type}"
fi
if [[ -z "${build_config}" ]]; then
  configuration_marker="${build_directory}/.flagdnn-build-config"
  if [[ -f "${configuration_marker}" ]]; then
    IFS= read -r build_config < "${configuration_marker}" || true
  fi
fi
if [[ -z "${build_config}" && -n "${configuration_types}" ]]; then
  if [[ ";${configuration_types};" == *";Release;"* ]]; then
    build_config="Release"
  else
    build_config="${configuration_types%%;*}"
  fi
fi
if [[ -n "${build_config}" ]]; then
  [[ "${build_config}" =~ ^[A-Za-z0-9_.+-]+$ ]] ||
    fail "invalid build configuration '${build_config}'"
fi

[[ "${install_prefix}" != "/" ]] || fail "refusing to install into /"
[[ -f "${build_directory}/CMakeCache.txt" ]] ||
  fail "${build_directory} is not configured; run tools/build.sh first"

install_arguments=(
  --install "${build_directory}"
  --prefix "${install_prefix}"
)
if [[ -n "${build_config}" ]]; then
  install_arguments+=(--config "${build_config}")
fi
if (( strip_install )); then
  install_arguments+=(--strip)
fi

echo "Installing FlagDNN"
echo "  build:  ${build_directory}"
echo "  prefix: ${install_prefix}"
echo "  config: ${build_config:-NOCONFIG}"

install_manifest="${build_directory}/install_manifest.txt"
# CMake rewrites this file for each non-component install. Remove any previous
# manifest first so a successful command that fails to generate its own
# manifest cannot make validation consume stale paths from an earlier prefix.
rm -f -- "${install_manifest}"
cmake "${install_arguments[@]}"

[[ -f "${install_manifest}" && ! -L "${install_manifest}" ]] ||
  fail "cmake --install did not generate a fresh install manifest"
[[ -d "${install_prefix}" ]] ||
  fail "cmake --install did not create the requested install prefix"
canonical_install_prefix="$(realpath -e -- "${install_prefix}")"

installed_headers=""
installed_libraries=""
installed_cmake=""
installed_core=""
installed_targets_configuration=""
installed_resources=""
manifest_line=0
while IFS= read -r installed_file || [[ -n "${installed_file}" ]]; do
  ((manifest_line += 1))
  [[ -n "${installed_file}" ]] || continue
  if ! canonical_installed_file="$(realpath -e -- "${installed_file}")"; then
    fail "install manifest entry ${manifest_line} does not exist: ${installed_file}"
  fi
  case "${canonical_installed_file}" in
    "${canonical_install_prefix}"/*)
      ;;
    *)
      fail "install manifest entry ${manifest_line} is outside the requested prefix: ${installed_file}"
      ;;
  esac
  [[ -f "${canonical_installed_file}" ]] || continue
  case "${canonical_installed_file}" in
    */libflagdnn.so|*/libflagdnn.so.*|*/libflagdnn.dylib|*/libflagdnn.*.dylib|*/flagdnn.dll)
      installed_core="${canonical_installed_file}"
      installed_libraries="${canonical_installed_file%/*}"
      ;;
    */flagdnn/flagdnn.h)
      installed_headers="${canonical_installed_file%/flagdnn/flagdnn.h}"
      ;;
    */cmake/FlagDNN/FlagDNNConfig.cmake)
      installed_cmake="${canonical_installed_file%/FlagDNNConfig.cmake}"
      ;;
    */cmake/FlagDNN/FlagDNNTargets-*.cmake)
      installed_targets_configuration="${canonical_installed_file}"
      ;;
    */share/flagdnn/kernels/registry.json)
      installed_resources="${canonical_installed_file%/kernels/registry.json}"
      ;;
  esac
done < "${install_manifest}"

[[ -n "${installed_headers}" ]] ||
  fail "installation did not produce the public FlagDNN headers"
[[ -n "${installed_core}" ]] ||
  fail "installation did not produce the FlagDNN core library"
[[ -n "${installed_cmake}" ]] ||
  fail "installation did not produce the FlagDNN CMake package"
[[ -n "${installed_targets_configuration}" ]] ||
  fail "installation did not produce a CMake target configuration file"
[[ -n "${installed_resources}" ]] ||
  fail "installation did not produce the FlagDNN kernel registry"

echo "FlagDNN SDK installed"
echo "  prefix:    ${install_prefix}"
if [[ -n "${installed_headers}" ]]; then
  echo "  headers:   ${installed_headers}"
fi
if [[ -n "${installed_libraries}" ]]; then
  echo "  libraries: ${installed_libraries}"
fi
if [[ -n "${installed_cmake}" ]]; then
  echo "  CMake:     ${installed_cmake}"
fi
if [[ -n "${installed_resources}" ]]; then
  echo "  resources: ${installed_resources}"
fi
