# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

foreach(_required IN ITEMS
    SOURCE_ROOT
    BUILD_ROOT
    TEST_ROOT
    CODEGEN_PYTHON
    PPU_SDK_ROOT
    BUILD_PLUGIN
    PLUGIN_SONAME
    JIT_ROOT
    JIT_SONAME
    JIT_SHA256
    JIT_PROVENANCE_SHA256
    STANDALONE_SHA256
    GEN_SSIG_SHA256
    TRITON_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "${_required} is missing")
  endif()
endforeach()
unset(_required)

foreach(_required_directory IN ITEMS
    "${SOURCE_ROOT}"
    "${BUILD_ROOT}"
    "${PPU_SDK_ROOT}"
    "${JIT_ROOT}"
    "${TRITON_ROOT}")
  if(NOT IS_DIRECTORY "${_required_directory}")
    message(FATAL_ERROR "required directory is missing: ${_required_directory}")
  endif()
endforeach()
unset(_required_directory)
foreach(_required_file IN ITEMS "${CODEGEN_PYTHON}" "${BUILD_PLUGIN}")
  if(NOT EXISTS "${_required_file}")
    message(FATAL_ERROR "required file is missing: ${_required_file}")
  endif()
endforeach()
unset(_required_file)
if(NOT PLUGIN_SONAME MATCHES
   "^libflagdnn_backend_thead\\.so\\.2$")
  message(FATAL_ERROR "invalid THead plugin SONAME: ${PLUGIN_SONAME}")
endif()
if(NOT JIT_SONAME MATCHES "^libtriton_jit\\.so(\\.[0-9]+)*$")
  message(FATAL_ERROR "invalid libtriton_jit SONAME: ${JIT_SONAME}")
endif()
foreach(_digest IN ITEMS
    JIT_SHA256 JIT_PROVENANCE_SHA256 STANDALONE_SHA256 GEN_SSIG_SHA256)
  string(LENGTH "${${_digest}}" _digest_length)
  if(NOT _digest_length EQUAL 64 OR
     NOT "${${_digest}}" MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "${_digest} is not a SHA-256 digest")
  endif()
endforeach()
unset(_digest)
unset(_digest_length)

file(READ
  "${SOURCE_ROOT}/backends/thead/validation/installed_consumer/thead_add.cpp"
  _thead_consumer_source)
foreach(_pointwise_contract IN ITEMS
    "std::array<PointwiseOperation, 37>"
    "acdnnOpTensor(MUL)+acdnnOpTensor(ADD)"
    "fe::PointwiseMode_t::GELU_FWD"
    "ACDNN_ACTIVATION_GELU"
    "acdnnActivationForward(GELU,NOT_PROPAGATE_NAN)"
    "fe::PointwiseMode_t::SQRT"
    "ACDNN_OP_TENSOR_SQRT"
    "acdnnOpTensor(SQRT)"
    "fe::PointwiseMode_t::NEG"
    "acdnnTransformTensor(alpha=-1,beta=0)"
    "fe::PointwiseMode_t::ABS"
    "acdnnTransformTensor(alpha=-1)+acdnnOpTensor(MAX)"
    "fe::PointwiseMode_t::CEIL"
    "ACDNN_POINTWISE_CEIL"
    "acdnnBackendExecute(POINTWISE_CEIL)"
    "fe::PointwiseMode_t::FLOOR"
    "ACDNN_POINTWISE_FLOOR"
    "acdnnBackendExecute(POINTWISE_FLOOR)"
    "fe::PointwiseMode_t::EXP"
    "ACDNN_POINTWISE_EXP"
    "acdnnBackendExecute(POINTWISE_EXP)"
    "fe::PointwiseMode_t::LOG"
    "ACDNN_POINTWISE_LOG"
    "acdnnBackendExecute(POINTWISE_LOG)"
    "fe::PointwiseMode_t::COS"
    "ACDNN_POINTWISE_COS"
    "acdnnBackendExecute(POINTWISE_COS)"
    "fe::PointwiseMode_t::RSQRT"
    "ACDNN_POINTWISE_RSQRT"
    "acdnnBackendExecute(POINTWISE_RSQRT)"
    "fe::PointwiseMode_t::SIN"
    "ACDNN_POINTWISE_SIN"
    "acdnnBackendExecute(POINTWISE_SIN)"
    "fe::PointwiseMode_t::TAN"
    "ACDNN_POINTWISE_TAN"
    "acdnnBackendExecute(POINTWISE_TAN)"
    "fe::PointwiseMode_t::SOFTPLUS_FWD"
    "ACDNN_POINTWISE_SOFTPLUS_FWD"
    "acdnnBackendExecute(POINTWISE_SOFTPLUS_FWD,beta=1)"
    "fe::PointwiseMode_t::SWISH_FWD"
    "ACDNN_POINTWISE_SWISH_FWD"
    "acdnnBackendExecute(POINTWISE_SWISH_FWD,beta=1.25)"
    "fe::PointwiseMode_t::GELU_APPROX_TANH_FWD"
    "ACDNN_POINTWISE_GELU_APPROX_TANH_FWD"
    "acdnnBackendExecute(POINTWISE_GELU_APPROX_TANH_FWD)"
    "fe::PointwiseMode_t::DIV"
    "ACDNN_POINTWISE_DIV"
    "acdnnBackendExecute(POINTWISE_DIV)"
    "fe::PointwiseMode_t::POW"
    "ACDNN_POINTWISE_POW"
    "acdnnBackendExecute(POINTWISE_POW)"
    "fe::PointwiseMode_t::SIGMOID_BWD"
    "ACDNN_POINTWISE_SIGMOID_BWD"
    "acdnnBackendExecute(POINTWISE_SIGMOID_BWD)"
    "fe::PointwiseMode_t::RECIPROCAL"
    "acdnnBackendExecute(POINTWISE_DIV,numerator=1)"
    "cuMemsetD32(installed reciprocal numerator)"
    "fe::PointwiseMode_t::CMP_EQ"
    "acdnnBackendExecute(POINTWISE_CMP_EQ)"
    "fe::PointwiseMode_t::CMP_NEQ"
    "acdnnBackendExecute(POINTWISE_CMP_NEQ)"
    "fe::PointwiseMode_t::CMP_GT"
    "acdnnBackendExecute(POINTWISE_CMP_GT)"
    "fe::PointwiseMode_t::CMP_GE"
    "acdnnBackendExecute(POINTWISE_CMP_GE)"
    "fe::PointwiseMode_t::CMP_LT"
    "acdnnBackendExecute(POINTWISE_CMP_LT)"
    "fe::PointwiseMode_t::CMP_LE"
    "acdnnBackendExecute(POINTWISE_CMP_LE)")
  string(FIND "${_thead_consumer_source}" "${_pointwise_contract}"
    _pointwise_contract_offset)
  if(_pointwise_contract_offset EQUAL -1)
    message(FATAL_ERROR
      "installed THead consumer is missing pointwise contract: ${_pointwise_contract}")
  endif()
endforeach()
unset(_pointwise_contract)
unset(_pointwise_contract_offset)
unset(_thead_consumer_source)

file(REAL_PATH "${BUILD_ROOT}" _build_root)
cmake_path(ABSOLUTE_PATH TEST_ROOT
  BASE_DIRECTORY "${_build_root}" NORMALIZE OUTPUT_VARIABLE _test_root)
cmake_path(IS_PREFIX _build_root "${_test_root}" NORMALIZE _test_below_build)
if(NOT _test_below_build OR _test_root STREQUAL _build_root)
  message(FATAL_ERROR "TEST_ROOT must be a strict child of BUILD_ROOT")
endif()

set(_clean_environment
  "${CMAKE_COMMAND}" -E env
  --unset=FLAGDNN_BACKEND
  --unset=FLAGDNN_BACKEND_PATH
  --unset=FLAGDNN_BACKEND_ROOT
  --unset=FLAGDNN_KERNEL_SOURCE_ROOT
  --unset=FLAGDNN_TUNING_ROOT
  --unset=FLAGDNN_THEAD_RESOURCE_ROOT
  --unset=FLAGDNN_THEAD_TRITON_ROOT
  --unset=FLAGDNN_THEAD_TRITON_JIT_ROOT
  --unset=FLAGDNN_THEAD_TRITON_JIT_DIR
  --unset=FLAGDNN_THEAD_TRITON_JIT_LIBRARY
  --unset=FLAGDNN_THEAD_TRITON_JIT_INCLUDE_DIR
  --unset=FLAGDNN_THEAD_TRITON_JIT_SCRIPT_DIR
  --unset=FLAGDNN_TRITON_JIT_LIBRARY
  --unset=FLAGDNN_TRITON_JIT_INCLUDE_DIR
  --unset=FLAGDNN_TRITON_JIT_SCRIPT_DIR
  --unset=TritonJIT_DIR
  --unset=TritonJIT_ROOT
  --unset=TRITONJIT_ROOT
  --unset=FLAGDNN_CODEGEN_COMPILER
  --unset=FLAGDNN_CODEGEN_PYTHON
  --unset=FLAGDNN_COMPILER
  --unset=FLAGDNN_COMPILER_EXECUTABLE
  --unset=FLAGDNN_EXECUTION_ENGINE
  --unset=FLAGDNN_CACHE_DIRECTORY
  --unset=PYTHONPATH
  --unset=PYTHONHOME
  --unset=LD_LIBRARY_PATH
  --)

function(_flagdnn_thead_run_step _description)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "${_description} failed (${_result})\n"
      "stdout:\n${_output}\nstderr:\n${_error}")
  endif()
  if(NOT _output STREQUAL "")
    message(STATUS "${_description}:\n${_output}")
  endif()
endfunction()

function(_flagdnn_thead_run_clean_step _description)
  _flagdnn_thead_run_step(
    "${_description}" ${_clean_environment} ${ARGN})
endfunction()

function(_flagdnn_thead_require_sha256 _path _expected)
  if(NOT EXISTS "${_path}")
    message(FATAL_ERROR "installed SDK resource is missing: ${_path}")
  endif()
  file(SHA256 "${_path}" _actual)
  if(NOT _actual STREQUAL _expected)
    message(FATAL_ERROR
      "installed SDK resource hash mismatch: ${_path}\n"
      "expected=${_expected}\nactual=${_actual}")
  endif()
endfunction()

find_program(_readelf NAMES readelf llvm-readelf REQUIRED)
function(_flagdnn_thead_read_dynamic _path _output)
  execute_process(
    COMMAND "${_readelf}" -d "${_path}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _dynamic
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "cannot read ELF metadata for ${_path}: ${_error}")
  endif()
  set(${_output} "${_dynamic}" PARENT_SCOPE)
endfunction()

function(_flagdnn_thead_require_rpath _path _expected)
  _flagdnn_thead_read_dynamic("${_path}" _dynamic)
  string(REGEX MATCH
    "\\((RPATH|RUNPATH)\\)[^\n]*\\[([^]]*)\\]"
    _record "${_dynamic}")
  if(_record STREQUAL "")
    message(FATAL_ERROR "${_path}: missing ELF RPATH")
  endif()
  if(NOT CMAKE_MATCH_2 STREQUAL _expected)
    message(FATAL_ERROR
      "${_path}: RPATH '${CMAKE_MATCH_2}' != '${_expected}'")
  endif()
  if(_dynamic MATCHES "\\(RUNPATH\\)")
    message(FATAL_ERROR "${_path}: expected fail-closed DT_RPATH")
  endif()
endfunction()

_flagdnn_thead_require_rpath("${BUILD_PLUGIN}" "$ORIGIN/flagdnn/thead")

file(REMOVE_RECURSE "${_test_root}")
file(MAKE_DIRECTORY "${_test_root}")
set(_sdk "${_test_root}/sdk")
set(_consumer_build "${_test_root}/consumer-build")
set(_cache "${_test_root}/cache")
file(MAKE_DIRECTORY "${_cache}")

_flagdnn_thead_run_step(
  "install isolated FlagDNN THead SDK"
  "${CMAKE_COMMAND}" --install "${_build_root}"
  --prefix "${_sdk}" --config Release)

if(EXISTS "${_sdk}/lib/cmake/FlagDNN/FlagDNNConfig.cmake")
  set(_flagdnn_dir "${_sdk}/lib/cmake/FlagDNN")
elseif(EXISTS "${_sdk}/lib64/cmake/FlagDNN/FlagDNNConfig.cmake")
  set(_flagdnn_dir "${_sdk}/lib64/cmake/FlagDNN")
else()
  message(FATAL_ERROR "installed SDK has no FlagDNNConfig.cmake")
endif()
get_filename_component(_sdk_library_directory
  "${_flagdnn_dir}/../.." REALPATH)
file(REAL_PATH "${_sdk_library_directory}/${PLUGIN_SONAME}"
  _installed_plugin)
file(REAL_PATH
  "${_sdk_library_directory}/flagdnn/thead/${JIT_SONAME}"
  _installed_jit)
set(_private_scripts
  "${_sdk_library_directory}/flagdnn/share/triton_jit/scripts")
set(_installed_compiler
  "${_sdk}/share/flagdnn/compiler/flagdnn_codegen/main.py")
set(_installed_provider
  "${_sdk}/share/flagdnn/backends/thead/compiler.py")
set(_installed_environment
  "${_sdk}/share/flagdnn/backends/thead/flagdnn_thead_compiler_environment.json")

foreach(_resource IN ITEMS
    "${_installed_plugin}"
    "${_installed_jit}"
    "${_private_scripts}/standalone_compile.py"
    "${_private_scripts}/gen_ssig.py"
    "${_installed_compiler}"
    "${_installed_provider}"
    "${_sdk}/share/flagdnn/backends/thead/compiler_identity.py"
    "${_sdk}/share/flagdnn/backends/thead/python_environment_identity.py"
    "${_installed_environment}"
    "${_sdk}/share/flagdnn/backends/thead/kernels/registry.json"
    "${_sdk}/share/flagdnn/backends/thead/kernels/add_square.py"
    "${_sdk}/share/flagdnn/backends/thead/kernels/convolution.py"
    "${_sdk}/share/flagdnn/backends/thead/kernels/normalization.py"
    "${_sdk}/share/flagdnn/backends/thead/tuning/common.yaml"
    "${_sdk}/share/flagdnn/kernels/registry.json")
  if(NOT EXISTS "${_resource}")
    message(FATAL_ERROR "installed SDK resource is missing: ${_resource}")
  endif()
endforeach()
unset(_resource)

_flagdnn_thead_require_rpath(
  "${_installed_plugin}" "$ORIGIN/flagdnn/thead")
_flagdnn_thead_require_sha256("${_installed_jit}" "${JIT_SHA256}")
_flagdnn_thead_require_sha256(
  "${_private_scripts}/standalone_compile.py" "${STANDALONE_SHA256}")
_flagdnn_thead_require_sha256(
  "${_private_scripts}/gen_ssig.py" "${GEN_SSIG_SHA256}")

file(READ "${_installed_environment}" _environment_json)
string(JSON _environment_schema ERROR_VARIABLE _json_error
  GET "${_environment_json}" schema_version)
if(_json_error OR NOT _environment_schema EQUAL 1)
  message(FATAL_ERROR "installed THead compiler environment schema is invalid")
endif()
string(JSON _environment_backend ERROR_VARIABLE _json_error
  GET "${_environment_json}" backend)
if(_json_error OR NOT _environment_backend STREQUAL "CUDA")
  message(FATAL_ERROR "installed THead compiler environment backend is invalid")
endif()
function(_flagdnn_thead_require_json_field _field _expected)
  string(JSON _actual ERROR_VARIABLE _json_error
    GET "${_environment_json}" "${_field}")
  if(_json_error OR NOT _actual STREQUAL _expected)
    message(FATAL_ERROR
      "installed THead compiler environment ${_field} is invalid")
  endif()
endfunction()
_flagdnn_thead_require_json_field(libtriton_jit_soname "${JIT_SONAME}")
_flagdnn_thead_require_json_field(libtriton_jit_sha256 "${JIT_SHA256}")
_flagdnn_thead_require_json_field(
  triton_jit_provenance_sha256 "${JIT_PROVENANCE_SHA256}")
_flagdnn_thead_require_json_field(
  standalone_compile_sha256 "${STANDALONE_SHA256}")
_flagdnn_thead_require_json_field(gen_ssig_sha256 "${GEN_SSIG_SHA256}")
string(JSON _library_relative ERROR_VARIABLE _json_error
  GET "${_environment_json}" libtriton_jit_install_relative_path)
file(RELATIVE_PATH _expected_library_relative "${_sdk}" "${_installed_jit}")
if(_json_error OR NOT _library_relative STREQUAL _expected_library_relative)
  message(FATAL_ERROR "installed compiler environment has wrong JIT path")
endif()
string(JSON _scripts_relative ERROR_VARIABLE _json_error
  GET "${_environment_json}" jit_script_install_relative_path)
file(RELATIVE_PATH _expected_scripts_relative "${_sdk}" "${_private_scripts}")
if(_json_error OR NOT _scripts_relative STREQUAL _expected_scripts_relative)
  message(FATAL_ERROR "installed compiler environment has wrong script path")
endif()

_flagdnn_thead_read_dynamic("${_installed_plugin}" _plugin_dynamic)
foreach(_needed IN ITEMS "libcuda.so.1" "${JIT_SONAME}")
  if(NOT _plugin_dynamic MATCHES
     "Shared library: \\[${_needed}\\]")
    message(FATAL_ERROR "installed plugin does not need ${_needed}")
  endif()
endforeach()
if(_plugin_dynamic MATCHES
   "Shared library: \\[lib(acdnn|acblas|cublas|cudnn|openblas|blas|lapack)")
  message(FATAL_ERROR "production THead plugin depends on a forbidden DNN/BLAS")
endif()

file(GLOB_RECURSE _installed_files LIST_DIRECTORIES FALSE "${_sdk}/*")
foreach(_installed_file IN LISTS _installed_files)
  get_filename_component(_installed_name "${_installed_file}" NAME)
  string(TOLOWER "${_installed_name}" _installed_name_lower)
  file(RELATIVE_PATH _installed_relative "${_sdk}" "${_installed_file}")
  if(_installed_relative MATCHES "(^|/)(validation|benchmark)(/|$)" OR
     _installed_name_lower MATCHES "^flagdnn_(test|benchmark)([._-]|$)")
    message(FATAL_ERROR
      "validation/test payload leaked into SDK: ${_installed_file}")
  endif()
  if(_installed_name_lower MATCHES
     "^(lib)?(acdnn|acblas|cublas|cudnn|openblas|blas|lapack)([._-]|$)")
    message(FATAL_ERROR
      "validation/reference payload leaked into SDK: ${_installed_file}")
  endif()
endforeach()
unset(_installed_file)
unset(_installed_name)
unset(_installed_name_lower)
unset(_installed_relative)

# Build the installed C ABI, C++ API, and THead/acDNN consumer before the
# external Triton qualification so compile/install regressions remain visible.
_flagdnn_thead_run_clean_step(
  "configure installed THead consumers"
  "${CMAKE_COMMAND}"
  -S "${SOURCE_ROOT}/backends/thead/validation/installed_consumer"
  -B "${_consumer_build}"
  "-DFlagDNN_DIR=${_flagdnn_dir}"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DFLAGDNN_SOURCE_ROOT=${SOURCE_ROOT}"
  "-DFLAGDNN_INSTALLED_THEAD_PPU_SDK_ROOT=${PPU_SDK_ROOT}"
  "-DFLAGDNN_INSTALLED_THEAD_SDK_ROOT=${_sdk}"
  "-DFLAGDNN_INSTALLED_THEAD_PLUGIN=${_installed_plugin}"
  "-DFLAGDNN_INSTALLED_THEAD_JIT=${_installed_jit}"
  "-DFLAGDNN_INSTALLED_THEAD_TRITON_ROOT=${TRITON_ROOT}"
  "-DFLAGDNN_INSTALLED_THEAD_CACHE_ROOT=${_cache}")
_flagdnn_thead_run_clean_step(
  "build installed THead consumers"
  "${CMAKE_COMMAND}" --build "${_consumer_build}" --parallel 2)
_flagdnn_thead_run_clean_step(
  "run installed C/C++ API consumers"
  "${CMAKE_CTEST_COMMAND}" --test-dir "${_consumer_build}"
  --output-on-failure -R "^installed\\.(c|cpp)$")

file(REAL_PATH "${TRITON_ROOT}" _triton_root)
find_program(_git NAMES git REQUIRED)
function(_flagdnn_thead_checkout_root _path _output)
  execute_process(
    COMMAND "${_git}" -C "${_path}" rev-parse --show-toplevel
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _checkout
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_result EQUAL 0 AND IS_DIRECTORY "${_checkout}")
    file(REAL_PATH "${_checkout}" _checkout_real)
    set(${_output} "${_checkout_real}" PARENT_SCOPE)
  else()
    set(${_output} "" PARENT_SCOPE)
  endif()
endfunction()
_flagdnn_thead_checkout_root("${_triton_root}" _triton_checkout)
if(NOT _triton_checkout STREQUAL "")
  message(FATAL_ERROR
    "THead installed-consumer environment prerequisite failure: "
    "TRITON_ROOT is inside the development checkout ${_triton_checkout}; "
    "install the PPU-aware Triton distribution independently")
endif()
if(NOT EXISTS "${_triton_root}/triton/__init__.py" OR
   NOT EXISTS "${_triton_root}/triton/backends/nvidia/compiler.py")
  message(FATAL_ERROR
    "THead installed-consumer environment prerequisite failure: "
    "installed Triton root is incomplete: ${_triton_root}")
endif()
file(GLOB _triton_metadata
  "${_triton_root}/triton-*.dist-info/METADATA")
list(LENGTH _triton_metadata _triton_metadata_count)
if(NOT _triton_metadata_count EQUAL 1)
  message(FATAL_ERROR
    "THead installed-consumer environment prerequisite failure: expected "
    "one installed Triton distribution metadata entry, found "
    "${_triton_metadata_count} below ${_triton_root}")
endif()
list(GET _triton_metadata 0 _triton_metadata_file)
file(READ "${_triton_metadata_file}" _triton_metadata_text)
if(NOT _triton_metadata_text MATCHES "Version:[ \t]*[^\r\n]*[Pp][Pp][Uu]")
  message(FATAL_ERROR
    "THead installed-consumer environment prerequisite failure: "
    "the installed Triton distribution is not PPU-qualified")
endif()

set(_identity "${_test_root}/installed-thead.identity")
_flagdnn_thead_run_clean_step(
  "identify installed THead compiler resources"
  "${CMAKE_COMMAND}" -E env
  "PYTHONDONTWRITEBYTECODE=1"
  "FLAGDNN_THEAD_TRITON_ROOT=${_triton_root}"
  "FLAGDNN_THEAD_PPU_SDK_ROOT=${PPU_SDK_ROOT}"
  "PPU_SDK=${PPU_SDK_ROOT}"
  "PPU_HOME=${PPU_SDK_ROOT}"
  "CUDA_PATH=${PPU_SDK_ROOT}/CUDA_SDK"
  "TRITON_PTXAS_PATH=${PPU_SDK_ROOT}/CUDA_SDK/bin/ptxas"
  "TRITON_IR_FORMATTER_PATH=${PPU_SDK_ROOT}/bin/llvm-irformatter"
  "TRITON_JIT_BACKEND=CUDA"
  "LD_LIBRARY_PATH=${PPU_SDK_ROOT}/targets/x86_64-linux/lib:${PPU_SDK_ROOT}/lib:${PPU_SDK_ROOT}/CUDA_SDK/lib64"
  "${CODEGEN_PYTHON}" "${_installed_compiler}"
  --identify --backend thead --target ppu_contract_cc80
  --execution-engine libtriton_jit
  --identity-output "${_identity}" --quiet)
file(STRINGS "${_identity}" _identity_lines LIMIT_COUNT 2)
list(LENGTH _identity_lines _identity_line_count)
if(NOT _identity_line_count EQUAL 2)
  message(FATAL_ERROR "installed THead identity manifest is malformed")
endif()
list(GET _identity_lines 1 _identity_metadata)
string(JSON _dependency_count ERROR_VARIABLE _json_error
  LENGTH "${_identity_metadata}" files)
if(_json_error OR _dependency_count LESS 1)
  message(FATAL_ERROR "installed THead identity has no dependency closure")
endif()

set(_required_identity_paths
  "${_installed_compiler}"
  "${_installed_provider}"
  "${_installed_environment}"
  "${_installed_jit}"
  "${_private_scripts}/standalone_compile.py"
  "${_private_scripts}/gen_ssig.py"
  "${_sdk}/share/flagdnn/kernels/registry.json"
  "${_sdk}/share/flagdnn/backends/thead/kernels/registry.json"
  "${_sdk}/share/flagdnn/backends/thead/kernels/add_square.py"
  "${_sdk}/share/flagdnn/backends/thead/kernels/convolution.py"
  "${_sdk}/share/flagdnn/backends/thead/kernels/normalization.py"
  "${_sdk}/share/flagdnn/backends/thead/tuning/common.yaml")
set(_observed_identity_paths)
math(EXPR _dependency_last "${_dependency_count} - 1")
foreach(_index RANGE 0 ${_dependency_last})
  string(JSON _dependency GET "${_identity_metadata}" files ${_index})
  if(EXISTS "${_dependency}")
    file(REAL_PATH "${_dependency}" _dependency_real)
  else()
    message(FATAL_ERROR
      "installed THead identity dependency is missing: ${_dependency}")
  endif()
  list(APPEND _observed_identity_paths "${_dependency_real}")
  cmake_path(IS_PREFIX SOURCE_ROOT "${_dependency_real}" NORMALIZE _in_source)
  cmake_path(IS_PREFIX _build_root "${_dependency_real}" NORMALIZE _in_build)
  cmake_path(IS_PREFIX _sdk "${_dependency_real}" NORMALIZE _in_sdk)
  cmake_path(IS_PREFIX JIT_ROOT "${_dependency_real}" NORMALIZE _in_jit_build)
  if((_in_source OR _in_build OR _in_jit_build) AND NOT _in_sdk)
    message(FATAL_ERROR
      "installed THead compiler leaked a source/build dependency: "
      "${_dependency_real}")
  endif()
endforeach()
foreach(_required_identity IN LISTS _required_identity_paths)
  file(REAL_PATH "${_required_identity}" _required_identity_real)
  if(NOT _required_identity_real IN_LIST _observed_identity_paths)
    message(FATAL_ERROR
      "installed THead identity omitted ${_required_identity_real}")
  endif()
endforeach()

_flagdnn_thead_run_clean_step(
  "run installed THead pointwise and AddSquare acDNN consumer"
  "${CMAKE_CTEST_COMMAND}" --test-dir "${_consumer_build}"
  --output-on-failure -V -R "^installed\\.thead_add$")

message(STATUS "Installed THead SDK consumer contract verified")
