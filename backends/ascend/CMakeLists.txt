# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

if(NOT FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
  message(FATAL_ERROR
    "FlagDNN Ascend supports only FLAGDNN_EXECUTION_ENGINE=libtriton_jit")
endif()

# Keep the Ascend capability document as the only target-policy data source.
# Materialize its constexpr C++ view only when this backend is selected so
# platform-neutral and other-backend builds do not depend on Ascend resources.
set(_flagdnn_ascend_capabilities
  "${CMAKE_CURRENT_SOURCE_DIR}/capabilities.json")
file(READ "${_flagdnn_ascend_capabilities}"
  _flagdnn_ascend_capabilities_json)
string(JSON _flagdnn_ascend_arch_count LENGTH
  "${_flagdnn_ascend_capabilities_json}" codegen_arches)
if(_flagdnn_ascend_arch_count LESS 1)
  message(FATAL_ERROR "Ascend codegen_arches must not be empty")
endif()
math(EXPR _flagdnn_ascend_arch_last
  "${_flagdnn_ascend_arch_count} - 1")
set(_flagdnn_ascend_arch_initializers "")
set(_flagdnn_ascend_arches "")
foreach(index RANGE 0 ${_flagdnn_ascend_arch_last})
  string(JSON arch GET "${_flagdnn_ascend_capabilities_json}"
    codegen_arches ${index})
  if(NOT arch MATCHES "^Ascend[0-9A-Za-z_-]+$")
    message(FATAL_ERROR "Invalid Ascend codegen architecture: ${arch}")
  endif()
  if(arch IN_LIST _flagdnn_ascend_arches)
    message(FATAL_ERROR "Duplicate Ascend codegen architecture: ${arch}")
  endif()
  list(APPEND _flagdnn_ascend_arches "${arch}")
  string(APPEND _flagdnn_ascend_arch_initializers "    \"${arch}\",\n")
endforeach()

string(JSON _flagdnn_ascend_runtime_detected_count LENGTH
  "${_flagdnn_ascend_capabilities_json}"
  target_policy runtime_detected_codegen_arches)
if(_flagdnn_ascend_runtime_detected_count LESS 1)
  message(FATAL_ERROR
    "Ascend runtime_detected_codegen_arches must not be empty")
endif()
math(EXPR _flagdnn_ascend_runtime_detected_last
  "${_flagdnn_ascend_runtime_detected_count} - 1")
set(_flagdnn_ascend_runtime_detected_initializers "")
set(_flagdnn_ascend_runtime_detected_arches "")
foreach(index RANGE 0 ${_flagdnn_ascend_runtime_detected_last})
  string(JSON arch GET "${_flagdnn_ascend_capabilities_json}"
    target_policy runtime_detected_codegen_arches ${index})
  if(NOT arch MATCHES "^Ascend[0-9A-Za-z_-]+$")
    message(FATAL_ERROR
      "Invalid runtime-detected Ascend codegen architecture: ${arch}")
  endif()
  if(NOT arch IN_LIST _flagdnn_ascend_arches OR
     arch IN_LIST _flagdnn_ascend_runtime_detected_arches)
    message(FATAL_ERROR
      "Invalid or duplicate runtime-detected Ascend architecture: ${arch}")
  endif()
  list(APPEND _flagdnn_ascend_runtime_detected_arches "${arch}")
  string(APPEND _flagdnn_ascend_runtime_detected_initializers
    "    \"${arch}\",\n")
endforeach()

foreach(field IN ITEMS environment runtime_soc codegen_arch)
  string(JSON _flagdnn_ascend_compatibility_${field} GET
    "${_flagdnn_ascend_capabilities_json}"
    target_policy compatibility_alias ${field})
endforeach()
if(NOT _flagdnn_ascend_compatibility_environment MATCHES
     "^[A-Z][A-Z0-9_]+$" OR
   NOT _flagdnn_ascend_compatibility_runtime_soc MATCHES
     "^Ascend[0-9A-Za-z_-]+$" OR
   NOT _flagdnn_ascend_compatibility_codegen_arch MATCHES
     "^Ascend[0-9A-Za-z_-]+$" OR
   NOT _flagdnn_ascend_compatibility_runtime_soc IN_LIST
     _flagdnn_ascend_runtime_detected_arches OR
   NOT _flagdnn_ascend_compatibility_codegen_arch IN_LIST
     _flagdnn_ascend_arches OR
   _flagdnn_ascend_compatibility_runtime_soc STREQUAL
     _flagdnn_ascend_compatibility_codegen_arch)
  message(FATAL_ERROR "Ascend compatibility alias policy is invalid")
endif()
configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/ascend_target_policy_data.hpp.in"
  "${PROJECT_BINARY_DIR}/generated/include/flagdnn/ascend_target_policy_data.hpp"
  @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_flagdnn_ascend_capabilities}")

set(CANN_ROOT "" CACHE PATH "Root of the CANN 9 installation")
if(NOT CANN_ROOT)
  foreach(environment_name IN ITEMS ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME)
    if(DEFINED ENV{${environment_name}} AND
       NOT "$ENV{${environment_name}}" STREQUAL "")
      set(CANN_ROOT "$ENV{${environment_name}}" CACHE PATH
        "Root of the CANN 9 installation" FORCE)
      break()
    endif()
  endforeach()
endif()
if(NOT CANN_ROOT)
  message(FATAL_ERROR
    "Ascend backend requires CANN_ROOT, ASCEND_HOME_PATH, or "
    "ASCEND_TOOLKIT_HOME")
endif()

include(cmake/TritonJITCompat.cmake)
flagdnn_ascend_prepare_triton_jit("${CANN_ROOT}")

if(NOT FLAGDNN_CODEGEN_PYTHON)
  message(FATAL_ERROR
    "Ascend libtriton_jit requires FLAGDNN_CODEGEN_PYTHON")
endif()
set(Python_EXECUTABLE "${FLAGDNN_CODEGEN_PYTHON}")
find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -I -c
    "import importlib.util,sys; names=('torch','triton','yaml','torch_npu'); missing=[n for n in names if importlib.util.find_spec(n) is None]; print(','.join(missing)); sys.exit(bool(missing))"
  TIMEOUT 10
  RESULT_VARIABLE _flagdnn_ascend_module_result
  OUTPUT_VARIABLE _flagdnn_ascend_missing_modules
  ERROR_VARIABLE _flagdnn_ascend_module_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _flagdnn_ascend_module_result EQUAL 0)
  message(FATAL_ERROR
    "FLAGDNN_CODEGEN_PYTHON is missing required Ascend compiler modules "
    "(${_flagdnn_ascend_missing_modules}): ${_flagdnn_ascend_module_error}")
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -I -c
    "import pathlib,sysconfig; print(pathlib.Path(sysconfig.get_paths()['purelib']).resolve())"
  TIMEOUT 10
  RESULT_VARIABLE _flagdnn_ascend_python_module_root_result
  OUTPUT_VARIABLE _flagdnn_ascend_python_module_root
  ERROR_VARIABLE _flagdnn_ascend_python_module_root_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _flagdnn_ascend_python_module_root_result EQUAL 0 OR
   NOT IS_DIRECTORY "${_flagdnn_ascend_python_module_root}")
  message(FATAL_ERROR
    "Cannot resolve the selected Ascend Python module root: "
    "${_flagdnn_ascend_python_module_root_error}")
endif()
file(REAL_PATH "${_flagdnn_ascend_python_module_root}"
  _flagdnn_ascend_python_module_root)
file(REAL_PATH "${Python_EXECUTABLE}"
  _flagdnn_ascend_python_program_canonical)
if(NOT EXISTS "${_flagdnn_ascend_python_program_canonical}")
  message(FATAL_ERROR "Cannot resolve the selected Ascend Python program")
endif()
file(SHA256 "${_flagdnn_ascend_python_program_canonical}"
  _flagdnn_ascend_python_program_sha256)

# A venv may intentionally use system-site-packages: the NPU Triton package
# can live in the venv while torch/torch_npu/PyYAML live in its pinned base
# interpreter.  Record every selected module origin independently rather than
# weakening the runtime check to accept any importable module on sys.path.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -I -c
    "import importlib.util,pathlib; names=('torch','triton','torch_npu','yaml'); print('\\n'.join(str(pathlib.Path(importlib.util.find_spec(n).origin).resolve()) for n in names))"
  TIMEOUT 10
  RESULT_VARIABLE _flagdnn_ascend_module_origins_result
  OUTPUT_VARIABLE _flagdnn_ascend_module_origins_output
  ERROR_VARIABLE _flagdnn_ascend_module_origins_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" _flagdnn_ascend_module_origins
  "${_flagdnn_ascend_module_origins_output}")
list(LENGTH _flagdnn_ascend_module_origins
  _flagdnn_ascend_module_origin_count)
if(NOT _flagdnn_ascend_module_origins_result EQUAL 0 OR
   NOT _flagdnn_ascend_module_origin_count EQUAL 4)
  message(FATAL_ERROR
    "Cannot resolve exact Ascend Python module origins: "
    "${_flagdnn_ascend_module_origins_error}")
endif()
list(GET _flagdnn_ascend_module_origins 0 _flagdnn_ascend_torch_module)
list(GET _flagdnn_ascend_module_origins 1 _flagdnn_ascend_triton_module)
list(GET _flagdnn_ascend_module_origins 2 _flagdnn_ascend_torch_npu_module)
list(GET _flagdnn_ascend_module_origins 3 _flagdnn_ascend_yaml_module)
foreach(module_path IN ITEMS
    "${_flagdnn_ascend_torch_module}"
    "${_flagdnn_ascend_triton_module}"
    "${_flagdnn_ascend_torch_npu_module}"
    "${_flagdnn_ascend_yaml_module}")
  if(NOT IS_ABSOLUTE "${module_path}" OR NOT EXISTS "${module_path}")
    message(FATAL_ERROR
      "Selected Ascend Python module origin is not a regular absolute file: "
      "${module_path}")
  endif()
endforeach()
file(SHA256 "${_flagdnn_ascend_torch_module}"
  _flagdnn_ascend_torch_module_sha256)
file(SHA256 "${_flagdnn_ascend_triton_module}"
  _flagdnn_ascend_triton_module_sha256)
file(SHA256 "${_flagdnn_ascend_torch_npu_module}"
  _flagdnn_ascend_torch_npu_module_sha256)
file(SHA256 "${_flagdnn_ascend_yaml_module}"
  _flagdnn_ascend_yaml_module_sha256)

if(NOT DEFINED TritonJIT_DIR OR TritonJIT_DIR STREQUAL "")
  set(_flagdnn_ascend_triton_jit_candidates)
  if(DEFINED ENV{TritonJIT_DIR} AND NOT "$ENV{TritonJIT_DIR}" STREQUAL "")
    list(APPEND _flagdnn_ascend_triton_jit_candidates "$ENV{TritonJIT_DIR}")
  endif()
  if(DEFINED ENV{LIBTRITON_JIT_ROOT} AND
     NOT "$ENV{LIBTRITON_JIT_ROOT}" STREQUAL "")
    list(APPEND _flagdnn_ascend_triton_jit_candidates
      "$ENV{LIBTRITON_JIT_ROOT}/build"
      "$ENV{LIBTRITON_JIT_ROOT}/lib/cmake/TritonJIT")
  endif()
  list(APPEND _flagdnn_ascend_triton_jit_candidates
    "${PROJECT_SOURCE_DIR}/../libtriton_jit/build")
  foreach(candidate IN LISTS _flagdnn_ascend_triton_jit_candidates)
    if(EXISTS "${candidate}/TritonJITConfig.cmake")
      file(REAL_PATH "${candidate}" candidate)
      set(TritonJIT_DIR "${candidate}" CACHE PATH
        "Directory containing NPU TritonJITConfig.cmake")
      break()
    endif()
  endforeach()
endif()
if(NOT DEFINED TritonJIT_DIR OR TritonJIT_DIR STREQUAL "")
  message(FATAL_ERROR "Cannot find the installed NPU TritonJIT package")
endif()

foreach(module_candidate IN ITEMS
    "${TritonJIT_DIR}"
    "${TritonJIT_DIR}/cmake"
    "${TritonJIT_DIR}/../cmake")
  if(EXISTS "${module_candidate}/FindTorch.cmake")
    file(REAL_PATH "${module_candidate}" module_candidate)
    list(PREPEND CMAKE_MODULE_PATH "${module_candidate}")
    break()
  endif()
endforeach()

# TritonJIT's FindTorch module treats torch.utils.cmake_prefix_path as
# Torch_ROOT, although current Torch packages place TorchConfig.cmake in its
# Torch/ child. Locate without importing torch (which may initialize the NPU
# runtime), then reject a stale/user-supplied package from another interpreter.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -I -c
    "import importlib.util,pathlib; s=importlib.util.find_spec('torch'); print(pathlib.Path(s.origin).parent/'share'/'cmake'/'Torch' if s and s.origin else '')"
  TIMEOUT 10
  RESULT_VARIABLE _flagdnn_ascend_torch_result
  OUTPUT_VARIABLE _flagdnn_ascend_torch_output
  ERROR_VARIABLE _flagdnn_ascend_torch_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" _flagdnn_ascend_torch_lines
  "${_flagdnn_ascend_torch_output}")
set(_flagdnn_ascend_torch_directory "")
foreach(candidate IN LISTS _flagdnn_ascend_torch_lines)
  string(STRIP "${candidate}" candidate)
  if(EXISTS "${candidate}/TorchConfig.cmake")
    file(REAL_PATH "${candidate}" _flagdnn_ascend_torch_directory)
  endif()
endforeach()
if(NOT _flagdnn_ascend_torch_result EQUAL 0 OR
   NOT _flagdnn_ascend_torch_directory)
  message(FATAL_ERROR
    "FLAGDNN_CODEGEN_PYTHON does not provide a usable Torch CMake package; "
    "reported '${_flagdnn_ascend_torch_output}': "
    "${_flagdnn_ascend_torch_error}")
endif()
if(Torch_DIR AND EXISTS "${Torch_DIR}")
  file(REAL_PATH "${Torch_DIR}" _flagdnn_ascend_requested_torch_directory)
  if(NOT _flagdnn_ascend_requested_torch_directory STREQUAL
         _flagdnn_ascend_torch_directory)
    message(FATAL_ERROR
      "Torch_DIR belongs to a different Python environment: "
      "${_flagdnn_ascend_requested_torch_directory}")
  endif()
endif()
set(Torch_DIR "${_flagdnn_ascend_torch_directory}" CACHE PATH
  "Torch CMake package used by NPU libtriton_jit" FORCE)
get_filename_component(_flagdnn_ascend_torch_cmake_root "${Torch_DIR}"
  DIRECTORY)
set(Torch_ROOT "${_flagdnn_ascend_torch_cmake_root}" CACHE PATH
  "Torch CMake prefix used by NPU libtriton_jit" FORCE)

find_package(TritonJIT 0.1.0 CONFIG REQUIRED)
if(NOT TritonJIT_BACKEND STREQUAL "NPU")
  message(FATAL_ERROR
    "FlagDNN Ascend requires TritonJIT_BACKEND=NPU; found "
    "${TritonJIT_BACKEND}")
endif()
get_target_property(_flagdnn_ascend_torch_include_directories
  Torch::Torch INTERFACE_INCLUDE_DIRECTORIES)
if(NOT _flagdnn_ascend_torch_include_directories)
  message(FATAL_ERROR "Torch::Torch exports no public include directories")
endif()
get_target_property(_flagdnn_ascend_torch_compile_options
  Torch::Torch INTERFACE_COMPILE_OPTIONS)
get_target_property(_flagdnn_ascend_torch_compile_definitions
  Torch::Torch INTERFACE_COMPILE_DEFINITIONS)
if(TARGET TritonJIT::triton_jit)
  set(_flagdnn_ascend_triton_jit_target TritonJIT::triton_jit)
elseif(TARGET triton_jit)
  set(_flagdnn_ascend_triton_jit_target triton_jit)
else()
  message(FATAL_ERROR
    "TritonJIT was found but exported no usable triton_jit target")
endif()
get_target_property(_flagdnn_ascend_triton_jit_imported
  "${_flagdnn_ascend_triton_jit_target}" IMPORTED)
if(_flagdnn_ascend_triton_jit_imported)
  # Validation is assembled later from a sibling directory and must inspect
  # the exact production LTJ identity without re-running package discovery.
  set_property(TARGET "${_flagdnn_ascend_triton_jit_target}"
    PROPERTY IMPORTED_GLOBAL TRUE)
endif()

function(_flagdnn_ascend_imported_location target output)
  set(location "")
  foreach(property IN ITEMS
      IMPORTED_LOCATION IMPORTED_LOCATION_NOCONFIG IMPORTED_LOCATION_RELEASE
      IMPORTED_LOCATION_RELWITHDEBINFO IMPORTED_LOCATION_DEBUG)
    if(NOT location)
      get_target_property(candidate "${target}" "${property}")
      if(candidate AND EXISTS "${candidate}")
        file(REAL_PATH "${candidate}" location)
      endif()
    endif()
  endforeach()
  if(NOT location)
    message(FATAL_ERROR "Cannot determine imported location for ${target}")
  endif()
  set("${output}" "${location}" PARENT_SCOPE)
endfunction()

_flagdnn_ascend_imported_location(
  "${_flagdnn_ascend_triton_jit_target}" _flagdnn_ascend_triton_jit_library)
_flagdnn_ascend_imported_location(Python::Python _flagdnn_ascend_python_library)
get_target_property(_flagdnn_ascend_ltj_include_candidates
  "${_flagdnn_ascend_triton_jit_target}" INTERFACE_INCLUDE_DIRECTORIES)
set(_flagdnn_ascend_ltj_include "")
foreach(candidate IN LISTS _flagdnn_ascend_ltj_include_candidates)
  if(IS_DIRECTORY "${candidate}" AND
     EXISTS "${candidate}/triton_jit/kernel_metadata.h")
    file(REAL_PATH "${candidate}" candidate)
    if(_flagdnn_ascend_ltj_include AND
       NOT _flagdnn_ascend_ltj_include STREQUAL "${candidate}")
      message(FATAL_ERROR
        "TritonJIT exports multiple incompatible public include roots")
    endif()
    set(_flagdnn_ascend_ltj_include "${candidate}")
  endif()
endforeach()
if(NOT _flagdnn_ascend_ltj_include)
  message(FATAL_ERROR
    "Cannot locate the public headers of the selected NPU TritonJIT")
endif()
set(FLAGDNN_TRITON_JIT_STANDALONE_COMPILER "" CACHE FILEPATH
  "Exact standalone_compile.py paired with NPU TritonJIT")
if(NOT FLAGDNN_TRITON_JIT_STANDALONE_COMPILER)
  get_filename_component(_flagdnn_ascend_ltj_root
    "${_flagdnn_ascend_ltj_include}" DIRECTORY)
  set(_flagdnn_ascend_standalone_candidate
    "${_flagdnn_ascend_ltj_root}/scripts/standalone_compile.py")
  if(EXISTS "${_flagdnn_ascend_standalone_candidate}")
    set(FLAGDNN_TRITON_JIT_STANDALONE_COMPILER
      "${_flagdnn_ascend_standalone_candidate}" CACHE FILEPATH
      "Exact standalone_compile.py paired with NPU TritonJIT" FORCE)
  endif()
endif()
if(NOT EXISTS "${FLAGDNN_TRITON_JIT_STANDALONE_COMPILER}" OR
   NOT EXISTS "${TritonJIT_DIR}/TritonJITConfig.cmake")
  message(FATAL_ERROR
    "NPU TritonJIT standalone compiler/config identity is incomplete")
endif()
file(REAL_PATH "${FLAGDNN_TRITON_JIT_STANDALONE_COMPILER}"
  _flagdnn_ascend_ltj_standalone)
file(REAL_PATH "${TritonJIT_DIR}/TritonJITConfig.cmake"
  _flagdnn_ascend_ltj_config)
file(REAL_PATH "${_flagdnn_ascend_python_library}"
  _flagdnn_ascend_python_library)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -I -c
    "import os,sysconfig; print(os.path.realpath(os.path.join(sysconfig.get_config_var('LIBDIR'),sysconfig.get_config_var('LDLIBRARY'))))"
  TIMEOUT 10
  RESULT_VARIABLE _flagdnn_ascend_python_library_result
  OUTPUT_VARIABLE _flagdnn_ascend_interpreter_library
  ERROR_VARIABLE _flagdnn_ascend_python_library_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _flagdnn_ascend_python_library_result EQUAL 0 OR
   NOT EXISTS "${_flagdnn_ascend_interpreter_library}")
  message(FATAL_ERROR
    "Cannot resolve libpython for FLAGDNN_CODEGEN_PYTHON: "
    "${_flagdnn_ascend_python_library_error}")
endif()
file(REAL_PATH "${_flagdnn_ascend_interpreter_library}"
  _flagdnn_ascend_interpreter_library)
if(NOT _flagdnn_ascend_interpreter_library STREQUAL
       _flagdnn_ascend_python_library)
  message(FATAL_ERROR
    "Python::Python (${_flagdnn_ascend_python_library}) differs from the "
    "runtime library of FLAGDNN_CODEGEN_PYTHON "
    "(${_flagdnn_ascend_interpreter_library})")
endif()
get_filename_component(_flagdnn_ascend_python_soname
  "${_flagdnn_ascend_python_library}" NAME)
if(NOT CMAKE_READELF)
  find_program(CMAKE_READELF readelf REQUIRED)
endif()
execute_process(
  COMMAND "${CMAKE_READELF}" -d "${_flagdnn_ascend_triton_jit_library}"
  RESULT_VARIABLE _flagdnn_ascend_readelf_result
  OUTPUT_VARIABLE _flagdnn_ascend_readelf_output
  ERROR_VARIABLE _flagdnn_ascend_readelf_error)
if(NOT _flagdnn_ascend_readelf_result EQUAL 0)
  message(FATAL_ERROR
    "Cannot inspect libtriton_jit dependencies: "
    "${_flagdnn_ascend_readelf_error}")
endif()
string(FIND "${_flagdnn_ascend_readelf_output}"
  "[${_flagdnn_ascend_python_soname}]" _flagdnn_ascend_python_needed)
if(_flagdnn_ascend_python_needed EQUAL -1)
  message(FATAL_ERROR
    "Python::Python (${_flagdnn_ascend_python_soname}) does not match the "
    "libpython ABI required by NPU libtriton_jit")
endif()

file(SHA256 "${FLAGDNN_ASCEND_ASCENDCL_LIBRARY}"
  _flagdnn_ascend_ascendcl_sha256)
file(SHA256 "${FLAGDNN_ASCEND_RUNTIME_LIBRARY}"
  _flagdnn_ascend_runtime_sha256)
file(SHA256 "${_flagdnn_ascend_triton_jit_library}"
  _flagdnn_ascend_triton_jit_sha256)
file(SHA256 "${_flagdnn_ascend_python_library}"
  _flagdnn_ascend_python_library_sha256)
file(SHA256 "${_flagdnn_ascend_ltj_standalone}"
  _flagdnn_ascend_ltj_standalone_sha256)
set(_flagdnn_ascend_tuning_source
  "${CMAKE_CURRENT_SOURCE_DIR}/tuning/common.yaml")
set(_flagdnn_ascend_kernel_names
  binary unary ternary layout reduction matmul convolution normalization)
set(_flagdnn_ascend_identity_sources "${_flagdnn_ascend_tuning_source}")
foreach(_flagdnn_ascend_kernel IN LISTS _flagdnn_ascend_kernel_names)
  list(APPEND _flagdnn_ascend_identity_sources
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${_flagdnn_ascend_kernel}.py")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${_flagdnn_ascend_identity_sources})
file(SHA256 "${_flagdnn_ascend_tuning_source}"
  _flagdnn_ascend_tuning_source_sha256)
foreach(_flagdnn_ascend_kernel IN LISTS _flagdnn_ascend_kernel_names)
  file(SHA256
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/${_flagdnn_ascend_kernel}.py"
    "_flagdnn_ascend_${_flagdnn_ascend_kernel}_source_sha256")
endforeach()
set(_flagdnn_ascend_identity_definitions
  FLAGDNN_ASCEND_CANN_ROOT="${FLAGDNN_ASCEND_CANN_ROOT}"
  FLAGDNN_ASCEND_CANN_VERSION="${FLAGDNN_ASCEND_CANN_VERSION}"
  FLAGDNN_ASCEND_CANN_INNER_VERSION="${FLAGDNN_ASCEND_CANN_INNER_VERSION}"
  FLAGDNN_ASCEND_ASCENDCL_PATH="${FLAGDNN_ASCEND_ASCENDCL_LIBRARY}"
  FLAGDNN_ASCEND_ASCENDCL_SHA256="${_flagdnn_ascend_ascendcl_sha256}"
  FLAGDNN_ASCEND_RUNTIME_PATH="${FLAGDNN_ASCEND_RUNTIME_LIBRARY}"
  FLAGDNN_ASCEND_RUNTIME_SHA256="${_flagdnn_ascend_runtime_sha256}"
  FLAGDNN_ASCEND_LIBTRITON_JIT_PATH="${_flagdnn_ascend_triton_jit_library}"
  FLAGDNN_ASCEND_LIBTRITON_JIT_SHA256="${_flagdnn_ascend_triton_jit_sha256}"
  FLAGDNN_ASCEND_PYTHON_LIBRARY_PATH="${_flagdnn_ascend_python_library}"
  FLAGDNN_ASCEND_PYTHON_LIBRARY_SHA256="${_flagdnn_ascend_python_library_sha256}"
  FLAGDNN_ASCEND_STANDALONE_PATH="${_flagdnn_ascend_ltj_standalone}"
  FLAGDNN_ASCEND_STANDALONE_SHA256="${_flagdnn_ascend_ltj_standalone_sha256}"
  FLAGDNN_ASCEND_PYTHON_MODULE_ROOT="${_flagdnn_ascend_python_module_root}"
  FLAGDNN_ASCEND_PYTHON_PROGRAM_PATH="${Python_EXECUTABLE}"
  FLAGDNN_ASCEND_PYTHON_PROGRAM_CANONICAL_PATH="${_flagdnn_ascend_python_program_canonical}"
  FLAGDNN_ASCEND_PYTHON_PROGRAM_SHA256="${_flagdnn_ascend_python_program_sha256}"
  FLAGDNN_ASCEND_TORCH_MODULE_PATH="${_flagdnn_ascend_torch_module}"
  FLAGDNN_ASCEND_TORCH_MODULE_SHA256="${_flagdnn_ascend_torch_module_sha256}"
  FLAGDNN_ASCEND_TRITON_MODULE_PATH="${_flagdnn_ascend_triton_module}"
  FLAGDNN_ASCEND_TRITON_MODULE_SHA256="${_flagdnn_ascend_triton_module_sha256}"
  FLAGDNN_ASCEND_TORCH_NPU_MODULE_PATH="${_flagdnn_ascend_torch_npu_module}"
  FLAGDNN_ASCEND_TORCH_NPU_MODULE_SHA256="${_flagdnn_ascend_torch_npu_module_sha256}"
  FLAGDNN_ASCEND_YAML_MODULE_PATH="${_flagdnn_ascend_yaml_module}"
  FLAGDNN_ASCEND_YAML_MODULE_SHA256="${_flagdnn_ascend_yaml_module_sha256}"
  FLAGDNN_ASCEND_TUNING_SOURCE_SHA256="${_flagdnn_ascend_tuning_source_sha256}"
  FLAGDNN_ASCEND_BINARY_SOURCE_SHA256="${_flagdnn_ascend_binary_source_sha256}"
  FLAGDNN_ASCEND_UNARY_SOURCE_SHA256="${_flagdnn_ascend_unary_source_sha256}"
  FLAGDNN_ASCEND_TERNARY_SOURCE_SHA256="${_flagdnn_ascend_ternary_source_sha256}"
  FLAGDNN_ASCEND_LAYOUT_SOURCE_SHA256="${_flagdnn_ascend_layout_source_sha256}"
  FLAGDNN_ASCEND_REDUCTION_SOURCE_SHA256="${_flagdnn_ascend_reduction_source_sha256}"
  FLAGDNN_ASCEND_MATMUL_SOURCE_SHA256="${_flagdnn_ascend_matmul_source_sha256}"
  FLAGDNN_ASCEND_CONVOLUTION_SOURCE_SHA256="${_flagdnn_ascend_convolution_source_sha256}"
  FLAGDNN_ASCEND_NORMALIZATION_SOURCE_SHA256="${_flagdnn_ascend_normalization_source_sha256}")

# The core owns the external compiler process. Preserve the configure-time
# canonical toolkit root so a deployment configured with CANN_ROOT (rather
# than ambient shell variables) still gives the sanitized child the exact
# ASCEND_HOME_PATH used to build the plugin.
target_compile_definitions(flagdnn PRIVATE
  FLAGDNN_HAS_ASCEND_TARGET_POLICY=1
  FLAGDNN_ASCEND_DEFAULT_CANN_ROOT="${FLAGDNN_ASCEND_CANN_ROOT}"
  FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE="flagdnn/ascend"
  FLAGDNN_ASCEND_DEFAULT_LIBTRITON_JIT_LIBRARY="${_flagdnn_ascend_triton_jit_library}"
  FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_STANDALONE_COMPILER="${_flagdnn_ascend_ltj_standalone}"
  FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_CONFIG="${_flagdnn_ascend_ltj_config}"
  FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_INCLUDE_DIRECTORY="${_flagdnn_ascend_ltj_include}"
  FLAGDNN_ASCEND_DEFAULT_PYTHON_MODULE_ROOT="${_flagdnn_ascend_python_module_root}")

# The public NPU TritonJIT headers include CANN's architecture-specific
# runtime/runtime/rt.h.  Keep this exact configure-time CANN include root on
# the production plugin as well as on the standalone embedded preflight.
find_path(_flagdnn_ascend_runtime_header_root runtime/runtime/rt.h
  PATHS
    "${FLAGDNN_ASCEND_CANN_ROOT}/aarch64-linux/pkg_inc"
    "${FLAGDNN_ASCEND_CANN_ROOT}/x86_64-linux/pkg_inc"
  NO_DEFAULT_PATH REQUIRED)
get_filename_component(_flagdnn_ascend_arch_root
  "${_flagdnn_ascend_runtime_header_root}" DIRECTORY)

flagdnn_add_backend_plugin(ascend INSTALL ABI_VERSION 3
  SOURCES
    artifact.cpp
    backend.cpp
    context.cpp
    engines/engine.cpp
    engines/embedded_python.cpp
    engines/libtriton_jit.cpp
    engines/python_stdout_containment.cpp
    engines/runtime_layout.cpp
    error.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/json.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp
  LINK_LIBRARIES
    Ascend::ascendcl
    Ascend::runtime
    "${_flagdnn_ascend_triton_jit_library}"
    fmt::fmt-header-only
    Python::Python
    ${CMAKE_DL_LIBS})
target_compile_definitions(flagdnn_backend_ascend PRIVATE
  BACKEND_NPU=1
  ${_flagdnn_ascend_identity_definitions}
  FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE="flagdnn/ascend")
if(_flagdnn_ascend_torch_compile_definitions)
  target_compile_definitions(flagdnn_backend_ascend PRIVATE
    ${_flagdnn_ascend_torch_compile_definitions})
endif()
if(_flagdnn_ascend_torch_compile_options)
  target_compile_options(flagdnn_backend_ascend PRIVATE
    ${_flagdnn_ascend_torch_compile_options})
endif()
target_include_directories(flagdnn_backend_ascend SYSTEM PRIVATE
  "${_flagdnn_ascend_ltj_include}"
  ${_flagdnn_ascend_torch_include_directories}
  "${FLAGDNN_ASCEND_CANN_ROOT}/include"
  "${FLAGDNN_ASCEND_CANN_ROOT}/include/aclnn"
  "${FLAGDNN_ASCEND_CANN_ROOT}/include/experiment"
  "${FLAGDNN_ASCEND_CANN_ROOT}/include/experiment/runtime"
  "${_flagdnn_ascend_arch_root}/include"
  "${_flagdnn_ascend_arch_root}/include/experiment"
  "${_flagdnn_ascend_arch_root}/include/experiment/msprof"
  "${_flagdnn_ascend_arch_root}/include/experiment/msprof/toolchain"
  "${_flagdnn_ascend_runtime_header_root}"
  "${_flagdnn_ascend_runtime_header_root}/runtime/runtime")
set_target_properties(flagdnn_backend_ascend PROPERTIES
  BUILD_WITH_INSTALL_RPATH YES
  INSTALL_RPATH "$ORIGIN/flagdnn/ascend"
  INSTALL_RPATH_USE_LINK_PATH NO)
if(UNIX AND NOT APPLE)
  # Prefer the plugin-private, hash-pinned JIT over ambient libraries. CANN,
  # Python and Torch remain explicit platform prerequisites and are resolved
  # from the target environment, not captured build-machine paths.
  target_link_options(flagdnn_backend_ascend PRIVATE
    "LINKER:--disable-new-dtags")
endif()

set(_flagdnn_ascend_private_runtime_install_directory
  "${CMAKE_INSTALL_LIBDIR}/flagdnn/ascend")
add_custom_command(TARGET flagdnn_backend_ascend POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory
    "$<TARGET_FILE_DIR:flagdnn_backend_ascend>/flagdnn/ascend"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${_flagdnn_ascend_triton_jit_library}"
    "$<TARGET_FILE_DIR:flagdnn_backend_ascend>/flagdnn/ascend/libtriton_jit.so"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${_flagdnn_ascend_ltj_standalone}"
    "$<TARGET_FILE_DIR:flagdnn_backend_ascend>/flagdnn/ascend/standalone_compile.py"
  VERBATIM)

install(FILES
  "${_flagdnn_ascend_triton_jit_library}"
  "${_flagdnn_ascend_ltj_standalone}"
  "${_flagdnn_ascend_ltj_config}"
  DESTINATION "${_flagdnn_ascend_private_runtime_install_directory}")
install(DIRECTORY "${_flagdnn_ascend_ltj_include}/triton_jit"
  DESTINATION
    "${_flagdnn_ascend_private_runtime_install_directory}/include"
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp")

install(FILES kernels/registry.json
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/ascend/kernels")
foreach(resource IN ITEMS add_plan.py compiler.py compiler_identity.py
                          capabilities.json tuning_decoder.py)
  install(FILES "${resource}"
    DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/ascend")
endforeach()
install(DIRECTORY kernels/
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/ascend/kernels"
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY tuning/
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/ascend/tuning"
  FILES_MATCHING PATTERN "*.yaml")
