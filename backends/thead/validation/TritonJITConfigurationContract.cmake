# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.23)

foreach(_required IN ITEMS TRITON_JIT_ROOT TRITON_ROOT
                           PYTHON_EXECUTABLE TEST_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is required")
  endif()
endforeach()
if(NOT IS_ABSOLUTE "${TEST_ROOT}" OR
   TEST_ROOT STREQUAL "/" OR
   TEST_ROOT STREQUAL "/tmp")
  message(FATAL_ERROR "TEST_ROOT must be a dedicated absolute directory")
endif()

set(_resolver "${CMAKE_CURRENT_LIST_DIR}/../cmake/ResolveTritonJIT.cmake")
set(_identity_contract
  "${CMAKE_CURRENT_LIST_DIR}/TritonIdentityContract.py")
if(NOT EXISTS "${_resolver}")
  message(FATAL_ERROR "THead TritonJIT resolver is missing: ${_resolver}")
endif()
if(NOT EXISTS "${_identity_contract}")
  message(FATAL_ERROR "THead Triton identity contract is missing")
endif()

set(_provenance_headers
  triton_jit/backend_config.h
  triton_jit/backend_policy.h
  triton_jit/backends/cuda_backend.h
  triton_jit/jit_function_arg.h
  triton_jit/jit_utils.h
  triton_jit/kernel_metadata.h
  triton_jit/triton_jit_function.h
  triton_jit/triton_kernel.h)

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}" "${TEST_ROOT}/child")

function(_make_build_layout root backend definition marker)
  file(MAKE_DIRECTORY
    "${root}/build/src"
    "${root}/include/triton_jit/backends"
    "${root}/scripts")
  file(WRITE "${root}/build/TritonJITConfig.cmake"
    "set(TritonJIT_BACKEND \"${backend}\")\n")
  file(WRITE "${root}/build/src/libtriton_jit.so" "library-${marker}\n")
  file(WRITE "${root}/build/TritonJITTargets.cmake"
    "set_target_properties(jit PROPERTIES "
    "INTERFACE_COMPILE_DEFINITIONS \"${definition}\" "
    "IMPORTED_LOCATION_NOCONFIG "
    "\"${root}/build/src/libtriton_jit.so\" "
    "IMPORTED_SONAME_NOCONFIG \"libtriton_jit.so\")\n")
  foreach(_header IN LISTS _provenance_headers)
    get_filename_component(_header_directory
      "${root}/include/${_header}" DIRECTORY)
    file(MAKE_DIRECTORY "${_header_directory}")
    file(WRITE "${root}/include/${_header}" "${_header}-${marker}\n")
  endforeach()
  file(WRITE "${root}/scripts/standalone_compile.py"
    "# standalone-${marker}\n")
  file(WRITE "${root}/scripts/gen_ssig.py" "# gen-ssig-${marker}\n")
endfunction()

function(_make_install_layout root backend definition marker)
  file(MAKE_DIRECTORY
    "${root}/lib/cmake/TritonJIT"
    "${root}/lib"
    "${root}/include/triton_jit/backends"
    "${root}/share/triton_jit/scripts")
  file(WRITE "${root}/lib/cmake/TritonJIT/TritonJITConfig.cmake"
    "set(TritonJIT_BACKEND \"${backend}\")\n")
  file(WRITE
    "${root}/lib/cmake/TritonJIT/TritonJITTargets-release.cmake"
    "set_target_properties(jit PROPERTIES "
    "INTERFACE_COMPILE_DEFINITIONS \"${definition}\" "
    "IMPORTED_LOCATION_RELEASE "
    "\"\${_IMPORT_PREFIX}/lib/libtriton_jit.so\" "
    "IMPORTED_SONAME_RELEASE \"libtriton_jit.so\")\n")
  file(WRITE "${root}/lib/libtriton_jit.so" "library-${marker}\n")
  foreach(_header IN LISTS _provenance_headers)
    get_filename_component(_header_directory
      "${root}/include/${_header}" DIRECTORY)
    file(MAKE_DIRECTORY "${_header_directory}")
    file(WRITE "${root}/include/${_header}" "${_header}-${marker}\n")
  endforeach()
  file(WRITE "${root}/share/triton_jit/scripts/standalone_compile.py"
    "# standalone-${marker}\n")
  file(WRITE "${root}/share/triton_jit/scripts/gen_ssig.py"
    "# gen-ssig-${marker}\n")
endfunction()

_make_build_layout("${TEST_ROOT}/a" CUDA BACKEND_CUDA a)
_make_build_layout("${TEST_ROOT}/b" CUDA BACKEND_CUDA b)
_make_build_layout("${TEST_ROOT}/wrong-backend" HCU BACKEND_HCU wrong)
_make_build_layout("${TEST_ROOT}/wrong-definition" CUDA BACKEND_HCU wrong-def)
_make_build_layout("${TEST_ROOT}/missing-standalone" CUDA BACKEND_CUDA no-standalone)
file(REMOVE "${TEST_ROOT}/missing-standalone/scripts/standalone_compile.py")
_make_build_layout("${TEST_ROOT}/missing-gen" CUDA BACKEND_CUDA no-gen)
file(REMOVE "${TEST_ROOT}/missing-gen/scripts/gen_ssig.py")
_make_build_layout("${TEST_ROOT}/missing-header" CUDA BACKEND_CUDA no-header)
file(REMOVE
  "${TEST_ROOT}/missing-header/include/triton_jit/backends/cuda_backend.h")
_make_install_layout("${TEST_ROOT}/installed" CUDA BACKEND_CUDA installed)
_make_install_layout("${TEST_ROOT}/installed64" CUDA BACKEND_CUDA installed64)
file(RENAME "${TEST_ROOT}/installed64/lib" "${TEST_ROOT}/installed64/lib64")
set(_installed64_targets
  "${TEST_ROOT}/installed64/lib64/cmake/TritonJIT/TritonJITTargets-release.cmake")
file(READ "${_installed64_targets}" _installed64_metadata)
string(REPLACE "/lib/libtriton_jit.so" "/lib64/libtriton_jit.so"
  _installed64_metadata "${_installed64_metadata}")
file(WRITE "${_installed64_targets}" "${_installed64_metadata}")

function(_make_fake_triton root backend_name distribution_version
         include_ppu_markers binary_extension)
  file(MAKE_DIRECTORY
    "${root}/triton/backends/nvidia"
    "${root}/triton/compiler"
    "${root}/triton-${distribution_version}.dist-info")
  file(WRITE "${root}/triton/__init__.py" "__version__ = '0.0-test'\n")
  file(WRITE "${root}/triton/backends/compiler.py"
    "class GPUTarget:\n"
    "    def __init__(self, backend, arch, warp_size):\n"
    "        self.backend = backend\n"
    "        self.arch = arch\n"
    "        self.warp_size = warp_size\n")
  set(_ppu_markers "")
  if(include_ppu_markers)
    set(_ppu_markers
      "# PPU_SDK llvm-irformatter --ppu-backend-options\n")
  endif()
  file(WRITE "${root}/triton/backends/nvidia/compiler.py"
    "${_ppu_markers}"
    "class CUDABackend:\n"
    "    def __init__(self, target):\n"
    "        self.target = target\n"
    "        self.binary_ext = '${binary_extension}'\n")
  file(WRITE "${root}/triton/backends/nvidia/driver.py"
    "class CudaDriver:\n"
    "    @staticmethod\n"
    "    def is_active():\n"
    "        return False\n")
  if(backend_name STREQUAL "")
    set(_catalog "backends = {}\n")
  else()
    set(_catalog
      "backends = {'${backend_name}': Entry(CUDABackend, CudaDriver)}\n")
  endif()
  file(WRITE "${root}/triton/backends/__init__.py"
    "from .nvidia.compiler import CUDABackend\n"
    "from .nvidia.driver import CudaDriver\n"
    "class Entry:\n"
    "    def __init__(self, compiler, driver):\n"
    "        self.compiler = compiler\n"
    "        self.driver = driver\n"
    "${_catalog}")
  file(WRITE "${root}/triton/compiler/__init__.py" "")
  file(WRITE "${root}/triton/compiler/compiler.py" "# generic compiler\n")
  file(WRITE "${root}/triton-${distribution_version}.dist-info/METADATA"
    "Metadata-Version: 2.1\nName: triton\nVersion: ${distribution_version}\n")
  file(WRITE "${root}/triton-${distribution_version}.dist-info/RECORD"
    "triton-${distribution_version}.dist-info/METADATA,,\n"
    "triton-${distribution_version}.dist-info/RECORD,,\n")
endfunction()

file(MAKE_DIRECTORY "${TEST_ROOT}/no-triton")
_make_fake_triton("${TEST_ROOT}/no-cuda" amd "0.0.0+ppu.test" TRUE cubin)
_make_fake_triton("${TEST_ROOT}/generic-triton" nvidia "0.0.0" TRUE cubin)
_make_fake_triton("${TEST_ROOT}/missing-ppu-path" nvidia
  "0.0.0+ppu.test" FALSE cubin)
_make_fake_triton("${TEST_ROOT}/wrong-binary" nvidia
  "0.0.0+ppu.test" TRUE hgbin)

file(WRITE "${TEST_ROOT}/child/CMakeLists.txt" [=[
cmake_minimum_required(VERSION 3.23)
project(FlagDNNTHeadTritonJITContract LANGUAGES NONE)

foreach(_required IN ITEMS RESOLVER JIT_ROOT CASE EXPECT_LAYOUT)
  if(NOT DEFINED ${_required})
    message(FATAL_ERROR "missing child argument: ${_required}")
  endif()
endforeach()

set(TritonJIT_DIR "/sentinel/global-jit")
add_library(TritonJIT::triton_jit INTERFACE IMPORTED GLOBAL)
set_property(TARGET TritonJIT::triton_jit PROPERTY FLAGDNN_SENTINEL "preserve")
include("${RESOLVER}")

set(_arguments ROOT "${JIT_ROOT}")
if(CASE STREQUAL "mixed-library")
  list(APPEND _arguments LIBRARY "${OTHER_ROOT}/build/src/libtriton_jit.so")
elseif(CASE STREQUAL "mixed-scripts")
  list(APPEND _arguments SCRIPT_DIR "${OTHER_ROOT}/scripts")
elseif(CASE STREQUAL "root-mismatch")
  list(APPEND _arguments CONFIG_DIR "${OTHER_ROOT}/build")
endif()
flagdnn_thead_resolve_triton_jit(THEAD_JIT ${_arguments})

foreach(_output IN ITEMS
    TARGET ROOT CONFIG_DIR CONFIG_FILE LIBRARY INCLUDE_DIR SCRIPT_DIR BACKEND
    LAYOUT SONAME PROVENANCE_SHA256 PROVENANCE_INPUT_FILES)
  if(NOT DEFINED THEAD_JIT_${_output} OR
     "${THEAD_JIT_${_output}}" STREQUAL "")
    message(FATAL_ERROR "resolver output is missing: ${_output}")
  endif()
endforeach()
if(NOT THEAD_JIT_BACKEND STREQUAL "CUDA")
  message(FATAL_ERROR "resolved JIT backend is not CUDA")
endif()
if(NOT THEAD_JIT_LAYOUT IN_LIST EXPECT_LAYOUT)
  message(FATAL_ERROR
    "resolved layout ${THEAD_JIT_LAYOUT} is not one of ${EXPECT_LAYOUT}")
endif()
if(NOT THEAD_JIT_PROVENANCE_SHA256 MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "JIT provenance is not a digest")
endif()
string(LENGTH "${THEAD_JIT_PROVENANCE_SHA256}" _digest_length)
if(NOT _digest_length EQUAL 64)
  message(FATAL_ERROR "JIT provenance is not SHA-256")
endif()
if(NOT TARGET FlagDNN::THeadTritonJIT OR
   NOT THEAD_JIT_TARGET STREQUAL "FlagDNN::THeadTritonJIT")
  message(FATAL_ERROR "private THead TritonJIT target is missing")
endif()
get_target_property(_links FlagDNN::THeadTritonJIT
  INTERFACE_LINK_LIBRARIES)
get_target_property(_includes FlagDNN::THeadTritonJIT
  INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_definitions FlagDNN::THeadTritonJIT
  INTERFACE_COMPILE_DEFINITIONS)
if(NOT _links STREQUAL THEAD_JIT_LIBRARY OR
   NOT _includes STREQUAL THEAD_JIT_INCLUDE_DIR OR
   NOT _definitions STREQUAL "BACKEND_CUDA;FMT_HEADER_ONLY=1")
  message(FATAL_ERROR "private THead JIT target is incoherent")
endif()
if(NOT TritonJIT_DIR STREQUAL "/sentinel/global-jit")
  message(FATAL_ERROR "resolver mutated global TritonJIT_DIR")
endif()
get_target_property(_sentinel TritonJIT::triton_jit FLAGDNN_SENTINEL)
if(NOT _sentinel STREQUAL "preserve")
  message(FATAL_ERROR "resolver mutated global TritonJIT target")
endif()

if(CASE STREQUAL "mutation")
  set(_before "${THEAD_JIT_PROVENANCE_SHA256}")
  file(APPEND "${JIT_ROOT}/include/triton_jit/backend_policy.h"
    "mutation\n")
  flagdnn_thead_resolve_triton_jit(MUTATED_JIT ROOT "${JIT_ROOT}")
  if(MUTATED_JIT_PROVENANCE_SHA256 STREQUAL _before)
    message(FATAL_ERROR "JIT provenance ignored a public-header mutation")
  endif()
endif()
]=])

function(_run_case name root other_root layout expect_success expected_error)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      -S "${TEST_ROOT}/child"
      -B "${TEST_ROOT}/build-${name}"
      "-DRESOLVER=${_resolver}"
      "-DJIT_ROOT=${root}"
      "-DOTHER_ROOT=${other_root}"
      "-DCASE=${name}"
      "-DEXPECT_LAYOUT=${layout}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}\n${_stderr}")
  if(expect_success)
    if(NOT _result EQUAL 0)
      message(FATAL_ERROR
        "positive TritonJIT contract ${name} failed:\n${_output}")
    endif()
  else()
    if(_result EQUAL 0)
      message(FATAL_ERROR "negative TritonJIT contract ${name} passed")
    endif()
    if(NOT _output MATCHES "${expected_error}")
      message(FATAL_ERROR
        "negative TritonJIT contract ${name} produced an unexpected error:\n"
        "${_output}")
    endif()
  endif()
endfunction()

file(REAL_PATH "${TRITON_JIT_ROOT}" _real_jit_root EXPAND_TILDE)
_run_case(real-valid "${_real_jit_root}" "${TEST_ROOT}/b"
  "build;install_lib;install_lib64" TRUE "")
_run_case(synthetic-valid "${TEST_ROOT}/a" "${TEST_ROOT}/b" build TRUE "")
_run_case(installed "${TEST_ROOT}/installed" "${TEST_ROOT}/b"
  install_lib TRUE "")
_run_case(installed64 "${TEST_ROOT}/installed64" "${TEST_ROOT}/b"
  install_lib64 TRUE "")
_run_case(mutation "${TEST_ROOT}/a" "${TEST_ROOT}/b" build TRUE "")
_run_case(mixed-library "${TEST_ROOT}/a" "${TEST_ROOT}/b" unused FALSE
  "do not belong to one coherent")
_run_case(mixed-scripts "${TEST_ROOT}/a" "${TEST_ROOT}/b" unused FALSE
  "do not belong to one coherent")
_run_case(root-mismatch "${TEST_ROOT}/a" "${TEST_ROOT}/b" unused FALSE
  "outside the selected THead TritonJIT root")
_run_case(wrong-backend "${TEST_ROOT}/wrong-backend" "${TEST_ROOT}/b"
  unused FALSE "requires a CUDA TritonJITConfig")
_run_case(wrong-definition "${TEST_ROOT}/wrong-definition" "${TEST_ROOT}/b"
  unused FALSE "BACKEND_CUDA")
_run_case(missing-standalone "${TEST_ROOT}/missing-standalone"
  "${TEST_ROOT}/b" unused FALSE "standalone_compile.py")
_run_case(missing-gen "${TEST_ROOT}/missing-gen" "${TEST_ROOT}/b"
  unused FALSE "gen_ssig.py")
_run_case(missing-header "${TEST_ROOT}/missing-header" "${TEST_ROOT}/b"
  unused FALSE "cuda_backend.h")

function(_run_identity_failure name root expected_error)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
      "PYTHONDONTWRITEBYTECODE=1"
      "PYTHONPATH=${root}"
      "${PYTHON_EXECUTABLE}" "${_identity_contract}"
      --root "${root}"
      --jit-root "${TRITON_JIT_ROOT}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  if(_result EQUAL 0)
    message(FATAL_ERROR "negative Triton identity ${name} passed")
  endif()
  set(_output "${_stdout}\n${_stderr}")
  if(NOT _output MATCHES "${expected_error}")
    message(FATAL_ERROR
      "negative Triton identity ${name} produced an unexpected error:\n"
      "${_output}")
  endif()
endfunction()

_run_identity_failure(no-triton "${TEST_ROOT}/no-triton"
  "outside configured root|configured root has no Triton package")
_run_identity_failure(no-cuda "${TEST_ROOT}/no-cuda"
  "has no CUDA codegen backend")
_run_identity_failure(generic-triton "${TEST_ROOT}/generic-triton"
  "not the PPU-qualified build")
_run_identity_failure(missing-ppu-path "${TEST_ROOT}/missing-ppu-path"
  "lacks the PPU compatibility path")
_run_identity_failure(wrong-binary "${TEST_ROOT}/wrong-binary"
  "unexpected Triton PPU binary extension")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "PYTHONDONTWRITEBYTECODE=1"
    "${PYTHON_EXECUTABLE}" "${_identity_contract}"
    --root "${TRITON_ROOT}"
    --jit-root "${TRITON_JIT_ROOT}"
  RESULT_VARIABLE _identity_result
  OUTPUT_VARIABLE _identity_stdout
  ERROR_VARIABLE _identity_stderr)
if(NOT _identity_result EQUAL 0)
  message(FATAL_ERROR
    "real Triton identity contract failed:\n${_identity_stdout}\n"
    "${_identity_stderr}")
endif()
string(JSON _compiler_backend GET "${_identity_stdout}"
  triton codegen_backend)
string(JSON _binary_extension GET "${_identity_stdout}"
  triton binary_extension)
string(JSON _jit_backend GET "${_identity_stdout}" libtriton_jit backend)
if(_compiler_backend STREQUAL "nvidia")
  set(_expected_binary_extension "cubin")
elseif(_compiler_backend STREQUAL "ppu")
  set(_expected_binary_extension "hgbin")
else()
  message(FATAL_ERROR "unexpected PPU codegen backend: ${_compiler_backend}")
endif()
if(NOT _binary_extension STREQUAL _expected_binary_extension OR
   NOT _jit_backend STREQUAL "CUDA")
  message(FATAL_ERROR "Triton/JIT identities were conflated")
endif()

message(STATUS "PASS THead CUDA-JIT and PPU-aware Triton identity contracts")
