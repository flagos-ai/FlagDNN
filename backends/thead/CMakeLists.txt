# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

if(NOT FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
  message(FATAL_ERROR
    "FlagDNN THead backend supports only the libtriton_jit execution engine")
endif()
if(NOT FLAGDNN_CODEGEN_PYTHON)
  message(FATAL_ERROR
    "THead libtriton_jit execution requires FLAGDNN_CODEGEN_PYTHON")
endif()

set(_flagdnn_thead_compiler_resources
  compiler.py
  compiler_identity.py
  python_environment_identity.py
  triton_compat.py
  codegen/__init__.py
  codegen/abi.py
  codegen/artifacts.py
  codegen/attention.py
  codegen/batchnorm.py
  codegen/convolution.py
  codegen/emit.py
  codegen/extended.py
  codegen/io.py
  codegen/layout.py
  codegen/matmul.py
  codegen/normalization.py
  codegen/pointwise.py
  codegen/reduction.py
  codegen/resources.py
  dispatch/__init__.py
  dispatch/attention.py
  dispatch/batchnorm.py
  dispatch/common.py
  dispatch/convolution.py
  dispatch/extended.py
  dispatch/extended_common.py
  dispatch/extended_schema.py
  dispatch/graph.py
  dispatch/layout.py
  dispatch/matmul.py
  dispatch/normalization.py
  dispatch/normalization_extended.py
  dispatch/pointwise.py
  dispatch/reduction.py
  dispatch/resample.py
  dispatch/selection.py
  dispatch/statistics.py
  dispatch/tensor.py
  dispatch/tuning.py
)
set(_flagdnn_thead_kernel_resources
  kernels/registry.json
  kernels/attention.py
  kernels/fp8_attention.py
  kernels/add_square.py
  kernels/convolution.py
  kernels/pow.py
  kernels/layout.py
  kernels/normalization.py
  tuning/common.yaml)
foreach(_resource IN LISTS
    _flagdnn_thead_compiler_resources
    _flagdnn_thead_kernel_resources)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_resource}")
    message(FATAL_ERROR
      "Required THead compiler resource is missing: ${_resource}")
  endif()
endforeach()

set(FLAGDNN_THEAD_PPU_SDK_ROOT "/usr/local/PPU_SDK"
  CACHE PATH "Root of one THead PPU SDK installation")
set(FLAGDNN_THEAD_TRITON_JIT_ROOT "${PROJECT_SOURCE_DIR}/../libtriton_jit"
  CACHE PATH "Root of one CUDA-backend libtriton_jit distribution")
if(NOT DEFINED FLAGDNN_THEAD_TRITON_ROOT OR
   FLAGDNN_THEAD_TRITON_ROOT STREQUAL "")
  execute_process(
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -X faulthandler -c
      "import pathlib,triton;print(pathlib.Path(triton.__file__).resolve().parent.parent)"
    RESULT_VARIABLE _flagdnn_thead_triton_discovery_result
    OUTPUT_VARIABLE _flagdnn_thead_triton_default_root
    ERROR_VARIABLE _flagdnn_thead_triton_discovery_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT "${_flagdnn_thead_triton_discovery_result}" STREQUAL "0" OR
     _flagdnn_thead_triton_default_root STREQUAL "")
    if(NOT "${_flagdnn_thead_triton_discovery_result}" MATCHES "^[0-9]+$")
      message(FATAL_ERROR
        "PPU FlagTree import process crashed or was terminated: "
        "${_flagdnn_thead_triton_discovery_result}.\n"
        "Selected Python: ${FLAGDNN_CODEGEN_PYTHON}\n"
        "Repair or reinstall PPU FlagTree for this interpreter; check that its "
        "native extension was built with consistent dependencies. "
        "Alternatively, select a working interpreter via tools/build.sh --python PATH.\n"
        "${_flagdnn_thead_triton_discovery_error}")
    endif()
    message(FATAL_ERROR
      "Selected Python '${FLAGDNN_CODEGEN_PYTHON}' cannot import 'triton'.\n"
      "PPU FlagTree provides the 'triton' Python module. Install PPU FlagTree "
      "for this interpreter, or select an interpreter containing it via "
      "tools/build.sh --python PATH.\n"
      "${_flagdnn_thead_triton_discovery_error}")
  endif()
else()
  set(_flagdnn_thead_triton_default_root
    "${FLAGDNN_THEAD_TRITON_ROOT}")
endif()
set(FLAGDNN_THEAD_TRITON_ROOT "${_flagdnn_thead_triton_default_root}"
  CACHE PATH "Python package root of the installed PPU-aware Triton distribution")
unset(_flagdnn_thead_triton_default_root)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ResolvePPUSDK.cmake")
set(_flagdnn_thead_ppu_arguments)
if(FLAGDNN_BUILD_TESTS OR FLAGDNN_BUILD_BENCHMARKS)
  list(APPEND _flagdnn_thead_ppu_arguments REQUIRE_ACDNN)
endif()
flagdnn_thead_resolve_ppu_sdk(_flagdnn_thead_ppu
  ${_flagdnn_thead_ppu_arguments})
unset(_flagdnn_thead_ppu_arguments)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ResolveTritonJIT.cmake")
flagdnn_thead_resolve_triton_jit(_flagdnn_thead_jit
  ROOT "${FLAGDNN_THEAD_TRITON_JIT_ROOT}")
file(SHA256 "${_flagdnn_thead_jit_LIBRARY}"
  _flagdnn_thead_jit_library_sha256)
file(SHA256 "${_flagdnn_thead_jit_SCRIPT_DIR}/standalone_compile.py"
  _flagdnn_thead_jit_standalone_sha256)
file(SHA256 "${_flagdnn_thead_jit_SCRIPT_DIR}/gen_ssig.py"
  _flagdnn_thead_jit_gen_ssig_sha256)

set(Python_EXECUTABLE "${FLAGDNN_CODEGEN_PYTHON}")
find_package(Python REQUIRED COMPONENTS Interpreter Development)
get_target_property(_flagdnn_thead_python_library Python::Python IMPORTED_LOCATION)
get_filename_component(_flagdnn_thead_python_library_directory
  "${_flagdnn_thead_python_library}" DIRECTORY)
execute_process(
  COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -c
    "import pathlib,torch\nroot=pathlib.Path(torch.__file__).resolve().parent\nprint(root/'include')\nprint(root/'include/torch/csrc/api/include')\nprint('1' if torch._C._GLIBCXX_USE_CXX11_ABI else '0')"
  RESULT_VARIABLE _flagdnn_thead_torch_result
  OUTPUT_VARIABLE _flagdnn_thead_torch_configuration
  ERROR_VARIABLE _flagdnn_thead_torch_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _flagdnn_thead_torch_result EQUAL 0)
  message(FATAL_ERROR
    "Cannot identify Torch headers for THead libtriton_jit: "
    "${_flagdnn_thead_torch_error}")
endif()
string(REPLACE "\n" ";" _flagdnn_thead_torch_configuration
  "${_flagdnn_thead_torch_configuration}")
list(LENGTH _flagdnn_thead_torch_configuration
  _flagdnn_thead_torch_configuration_count)
if(NOT _flagdnn_thead_torch_configuration_count EQUAL 3)
  message(FATAL_ERROR "Unexpected THead Torch header discovery result")
endif()
list(GET _flagdnn_thead_torch_configuration 0 _flagdnn_thead_torch_include)
list(GET _flagdnn_thead_torch_configuration 1 _flagdnn_thead_torch_api_include)
list(GET _flagdnn_thead_torch_configuration 2 _flagdnn_thead_torch_cxx11_abi)

if(NOT IS_DIRECTORY "${FLAGDNN_THEAD_TRITON_ROOT}")
  message(FATAL_ERROR
    "FLAGDNN_THEAD_TRITON_ROOT does not exist: "
    "${FLAGDNN_THEAD_TRITON_ROOT}")
endif()
file(REAL_PATH "${FLAGDNN_THEAD_TRITON_ROOT}"
  _flagdnn_thead_triton_root EXPAND_TILDE)
execute_process(
  COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -X faulthandler
    "${CMAKE_CURRENT_SOURCE_DIR}/validation/TritonIdentityContract.py"
    --root "${_flagdnn_thead_triton_root}"
    --jit-root "${_flagdnn_thead_jit_ROOT}"
  RESULT_VARIABLE _flagdnn_thead_triton_identity_result
  OUTPUT_VARIABLE _flagdnn_thead_triton_identity
  ERROR_VARIABLE _flagdnn_thead_triton_identity_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT "${_flagdnn_thead_triton_identity_result}" STREQUAL "0")
  message(FATAL_ERROR
    "THead compiler identity validation failed "
    "(${_flagdnn_thead_triton_identity_result}), Python: ${FLAGDNN_CODEGEN_PYTHON}\n"
    "${_flagdnn_thead_triton_identity_error}")
endif()
# The PPU-qualified distribution owns its actual backend and specialized
# frontend paths; FlagTree routes triton.compiler through spec/ppu.
foreach(_field IN ITEMS package_file compiler_file driver_file
    compiler_frontend_file distribution_metadata codegen_backend binary_extension)
  string(JSON _flagdnn_thead_triton_${_field} GET
    "${_flagdnn_thead_triton_identity}" triton "${_field}")
endforeach()
if(_flagdnn_thead_triton_codegen_backend STREQUAL "ppu")
  set(_flagdnn_thead_ppu_llc "${_flagdnn_thead_ppu_ROOT}/bin/ppu-llc")
  if(NOT EXISTS "${_flagdnn_thead_ppu_llc}")
    message(FATAL_ERROR "PPU compiler is missing: ${_flagdnn_thead_ppu_llc}")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_flagdnn_thead_ppu_llc}")
endif()
set(_flagdnn_thead_triton_init "${_flagdnn_thead_triton_package_file}")
set(_flagdnn_thead_triton_compiler "${_flagdnn_thead_triton_compiler_file}")
set(_flagdnn_thead_triton_driver "${_flagdnn_thead_triton_driver_file}")
set(_flagdnn_thead_triton_frontend "${_flagdnn_thead_triton_compiler_frontend_file}")
set(_flagdnn_thead_triton_metadata "${_flagdnn_thead_triton_distribution_metadata}")
set(_flagdnn_thead_triton_extension
  "${_flagdnn_thead_triton_root}/triton/_C/libtriton.so")
file(RELATIVE_PATH _flagdnn_thead_triton_frontend_relative
  "${_flagdnn_thead_triton_root}" "${_flagdnn_thead_triton_frontend}")
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/triton_compat.py"
  _flagdnn_thead_jit_compat_sha256)
file(SHA256 "${_flagdnn_thead_triton_init}"
  _flagdnn_thead_triton_init_sha256)
file(SHA256 "${_flagdnn_thead_triton_compiler}"
  _flagdnn_thead_triton_compiler_sha256)
file(SHA256 "${_flagdnn_thead_triton_driver}"
  _flagdnn_thead_triton_driver_sha256)
file(SHA256 "${_flagdnn_thead_triton_frontend}"
  _flagdnn_thead_triton_frontend_sha256)
file(SHA256 "${_flagdnn_thead_triton_metadata}"
  _flagdnn_thead_triton_metadata_sha256)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_flagdnn_thead_ppu_RELEASE_FILE}"
  ${_flagdnn_thead_jit_PROVENANCE_INPUT_FILES}
  "${_flagdnn_thead_triton_init}"
  "${_flagdnn_thead_triton_compiler}"
  "${_flagdnn_thead_triton_driver}"
  "${_flagdnn_thead_triton_frontend}"
  "${_flagdnn_thead_triton_extension}"
  "${_flagdnn_thead_triton_metadata}"
  "${CMAKE_CURRENT_SOURCE_DIR}/validation/TritonIdentityContract.py"
  ${_flagdnn_thead_compiler_resources}
  ${_flagdnn_thead_kernel_resources})

flagdnn_add_backend_plugin(thead
  ABI_VERSION 2
  INSTALL
  SOURCES
    artifact.cpp
    backend.cpp
    context.cpp
    engines/engine.cpp
    engines/libtriton_jit.cpp
    error.cpp
    ${PROJECT_SOURCE_DIR}/backends/autotune_policy.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/json.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp
  LINK_LIBRARIES
    FlagDNN::THeadCudaDriver
    FlagDNN::THeadTritonJIT
    Python::Python
    ${CMAKE_DL_LIBS})

target_include_directories(flagdnn_backend_thead SYSTEM PRIVATE
  "${_flagdnn_thead_jit_INCLUDE_DIR}"
  "${_flagdnn_thead_torch_include}"
  "${_flagdnn_thead_torch_api_include}")

target_compile_definitions(flagdnn_backend_thead PRIVATE
  _GLIBCXX_USE_CXX11_ABI=${_flagdnn_thead_torch_cxx11_abi}
  FLAGDNN_THEAD_PPU_SDK_VERSION="${_flagdnn_thead_ppu_VERSION}"
  FLAGDNN_THEAD_TRITON_ROOT="${_flagdnn_thead_triton_root}"
  FLAGDNN_THEAD_TRITON_CODEGEN_BACKEND="${_flagdnn_thead_triton_codegen_backend}"
  FLAGDNN_THEAD_TRITON_BINARY_EXTENSION="${_flagdnn_thead_triton_binary_extension}"
  FLAGDNN_THEAD_TRITON_FRONTEND_RELATIVE="${_flagdnn_thead_triton_frontend_relative}"
  FLAGDNN_THEAD_JIT_COMPAT_SHA256="${_flagdnn_thead_jit_compat_sha256}"
  FLAGDNN_THEAD_TRITON_METADATA="${_flagdnn_thead_triton_metadata}"
  FLAGDNN_THEAD_PPU_SDK_ROOT="${_flagdnn_thead_ppu_ROOT}"
  FLAGDNN_THEAD_JIT_SHA256="${_flagdnn_thead_jit_library_sha256}"
  FLAGDNN_THEAD_JIT_STANDALONE_SHA256="${_flagdnn_thead_jit_standalone_sha256}"
  FLAGDNN_THEAD_JIT_GEN_SSIG_SHA256="${_flagdnn_thead_jit_gen_ssig_sha256}"
  FLAGDNN_THEAD_TRITON_INIT_SHA256="${_flagdnn_thead_triton_init_sha256}"
  FLAGDNN_THEAD_TRITON_COMPILER_SHA256="${_flagdnn_thead_triton_compiler_sha256}"
  FLAGDNN_THEAD_TRITON_DRIVER_SHA256="${_flagdnn_thead_triton_driver_sha256}"
  FLAGDNN_THEAD_TRITON_FRONTEND_SHA256="${_flagdnn_thead_triton_frontend_sha256}"
  FLAGDNN_THEAD_TRITON_METADATA_SHA256="${_flagdnn_thead_triton_metadata_sha256}")
target_compile_definitions(flagdnn_backend_thead PRIVATE
  FLAGDNN_THEAD_TRITON_JIT_PROVENANCE_SHA256="${_flagdnn_thead_jit_PROVENANCE_SHA256}")

set(_flagdnn_thead_jit_install_directory
  "${CMAKE_INSTALL_LIBDIR}/flagdnn/thead")
set(_flagdnn_thead_jit_script_install_directory
  "${CMAKE_INSTALL_LIBDIR}/flagdnn/share/triton_jit/scripts")
set(_flagdnn_thead_compiler_environment_file
  "${CMAKE_CURRENT_BINARY_DIR}/flagdnn_thead_compiler_environment.json")
set(_flagdnn_thead_compiler_environment
  "{\n  \"schema_version\": 1,\n  \"backend\": \"CUDA\",\n  \"libtriton_jit_soname\": \"${_flagdnn_thead_jit_SONAME}\",\n  \"libtriton_jit_install_relative_path\": \"${_flagdnn_thead_jit_install_directory}/${_flagdnn_thead_jit_SONAME}\",\n  \"jit_script_install_relative_path\": \"${_flagdnn_thead_jit_script_install_directory}\",\n  \"libtriton_jit_sha256\": \"${_flagdnn_thead_jit_library_sha256}\",\n  \"triton_jit_provenance_sha256\": \"${_flagdnn_thead_jit_PROVENANCE_SHA256}\",\n  \"standalone_compile_sha256\": \"${_flagdnn_thead_jit_standalone_sha256}\",\n  \"gen_ssig_sha256\": \"${_flagdnn_thead_jit_gen_ssig_sha256}\"\n}\n")
file(GENERATE
  OUTPUT "${_flagdnn_thead_compiler_environment_file}"
  CONTENT "${_flagdnn_thead_compiler_environment}")
set_target_properties(flagdnn_backend_thead PROPERTIES
  BUILD_WITH_INSTALL_RPATH YES
  INSTALL_RPATH "$ORIGIN/flagdnn/thead"
  INSTALL_RPATH_USE_LINK_PATH NO)
if(UNIX AND NOT APPLE)
  target_link_options(flagdnn_backend_thead PRIVATE
    "LINKER:--disable-new-dtags"
    "LINKER:-z,nodelete")
endif()
add_custom_command(TARGET flagdnn_backend_thead POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory
    "$<TARGET_FILE_DIR:flagdnn_backend_thead>/flagdnn/thead"
  COMMAND "${CMAKE_COMMAND}" -E make_directory
    "$<TARGET_FILE_DIR:flagdnn_backend_thead>/flagdnn/share/triton_jit/scripts"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${_flagdnn_thead_jit_LIBRARY}"
    "$<TARGET_FILE_DIR:flagdnn_backend_thead>/flagdnn/thead/${_flagdnn_thead_jit_SONAME}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${_flagdnn_thead_jit_SCRIPT_DIR}/standalone_compile.py"
    "$<TARGET_FILE_DIR:flagdnn_backend_thead>/flagdnn/share/triton_jit/scripts/standalone_compile.py"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${_flagdnn_thead_jit_SCRIPT_DIR}/gen_ssig.py"
    "$<TARGET_FILE_DIR:flagdnn_backend_thead>/flagdnn/share/triton_jit/scripts/gen_ssig.py"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/triton_compat.py"
    "$<TARGET_FILE_DIR:flagdnn_backend_thead>/flagdnn/share/triton_jit/scripts/flagdnn_thead_jit_compat.py"
  VERBATIM)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/triton_compat.py"
  DESTINATION "${_flagdnn_thead_jit_script_install_directory}"
  RENAME flagdnn_thead_jit_compat.py)
install(FILES "${_flagdnn_thead_jit_LIBRARY}"
  DESTINATION "${_flagdnn_thead_jit_install_directory}"
  RENAME "${_flagdnn_thead_jit_SONAME}")
install(FILES
    "${_flagdnn_thead_jit_SCRIPT_DIR}/standalone_compile.py"
    "${_flagdnn_thead_jit_SCRIPT_DIR}/gen_ssig.py"
  DESTINATION "${_flagdnn_thead_jit_script_install_directory}")

# Validation is added later from the repository root. Publish immutable
# provenance on the backend target without leaking variables into common CMake.
set_target_properties(flagdnn_backend_thead PROPERTIES
  FLAGDNN_THEAD_PPU_SDK_ROOT "${_flagdnn_thead_ppu_ROOT}"
  FLAGDNN_THEAD_PPU_SDK_VERSION "${_flagdnn_thead_ppu_VERSION}"
  FLAGDNN_THEAD_CUDA_DRIVER_LIBRARY
    "${_flagdnn_thead_ppu_CUDA_DRIVER_LIBRARY}"
  FLAGDNN_THEAD_TRITON_JIT_ROOT "${_flagdnn_thead_jit_ROOT}"
  FLAGDNN_THEAD_TRITON_JIT_LIBRARY "${_flagdnn_thead_jit_LIBRARY}"
  FLAGDNN_THEAD_TRITON_JIT_SONAME "${_flagdnn_thead_jit_SONAME}"
  FLAGDNN_THEAD_TRITON_JIT_SHA256
    "${_flagdnn_thead_jit_library_sha256}"
  FLAGDNN_THEAD_TRITON_JIT_INCLUDE_DIR "${_flagdnn_thead_jit_INCLUDE_DIR}"
  FLAGDNN_THEAD_TRITON_JIT_PROVENANCE_SHA256
    "${_flagdnn_thead_jit_PROVENANCE_SHA256}"
  FLAGDNN_THEAD_TRITON_JIT_STANDALONE_SHA256
    "${_flagdnn_thead_jit_standalone_sha256}"
  FLAGDNN_THEAD_TRITON_JIT_GEN_SSIG_SHA256
    "${_flagdnn_thead_jit_gen_ssig_sha256}"
  FLAGDNN_THEAD_TRITON_ROOT "${_flagdnn_thead_triton_root}"
  FLAGDNN_THEAD_PYTHON_LIBRARY_DIR "${_flagdnn_thead_python_library_directory}"
  FLAGDNN_THEAD_TRITON_METADATA "${_flagdnn_thead_triton_metadata}"
  FLAGDNN_THEAD_TORCH_INCLUDE "${_flagdnn_thead_torch_include}"
  FLAGDNN_THEAD_TORCH_API_INCLUDE "${_flagdnn_thead_torch_api_include}"
  FLAGDNN_THEAD_TORCH_CXX11_ABI "${_flagdnn_thead_torch_cxx11_abi}")

foreach(_resource IN LISTS _flagdnn_thead_compiler_resources)
  get_filename_component(_directory "${_resource}" DIRECTORY)
  install(FILES "${_resource}"
    DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/thead/${_directory}")
endforeach()
install(FILES "${_flagdnn_thead_compiler_environment_file}"
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/thead")
install(FILES kernels/registry.json
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/thead/kernels")
install(DIRECTORY kernels/
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/thead/kernels"
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE)
install(FILES tuning/common.yaml
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/thead/tuning")
unset(_flagdnn_thead_compiler_resources)
unset(_flagdnn_thead_kernel_resources)

message(STATUS
  "FlagDNN THead PPU SDK ${_flagdnn_thead_ppu_VERSION}: "
  "${_flagdnn_thead_ppu_ROOT}")
message(STATUS
  "FlagDNN THead TritonJIT ${_flagdnn_thead_jit_BACKEND} "
  "(${_flagdnn_thead_jit_LAYOUT}): ${_flagdnn_thead_jit_LIBRARY}")
message(STATUS
  "FlagDNN THead PPU compiler packages (triton API): ${_flagdnn_thead_triton_root}")
