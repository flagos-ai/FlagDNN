if(NOT DEFINED SOURCE_ROOT OR NOT IS_DIRECTORY "${SOURCE_ROOT}")
  message(FATAL_ERROR "SOURCE_ROOT is missing")
endif()
if(NOT DEFINED BUILD_ROOT OR NOT EXISTS "${BUILD_ROOT}/CMakeCache.txt")
  message(FATAL_ERROR "BUILD_ROOT is not a configured FlagDNN build")
endif()
if(NOT DEFINED TEST_ROOT OR TEST_ROOT STREQUAL "")
  message(FATAL_ERROR "TEST_ROOT is missing")
endif()
if(NOT DEFINED CODEGEN_PYTHON OR NOT EXISTS "${CODEGEN_PYTHON}")
  message(FATAL_ERROR "CODEGEN_PYTHON is missing")
endif()
foreach(_required_variable IN ITEMS
    BUILD_PLUGIN
    PLUGIN_SONAME
    JIT_SONAME
    JIT_SHA256
    JIT_PROVENANCE_SHA256
    STANDALONE_SHA256
    GEN_SSIG_SHA256
    ENVIRONMENT_IDENTITY
    MUSA_ROOT
    MUDNN_INCLUDE_DIR
    MUDNN_LIBRARY
    VALIDATION_LIBRARY_PATH)
  if(NOT DEFINED ${_required_variable} OR
     "${${_required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${_required_variable} is missing")
  endif()
endforeach()
if(NOT EXISTS "${BUILD_PLUGIN}")
  message(FATAL_ERROR "BUILD_PLUGIN does not exist: ${BUILD_PLUGIN}")
endif()
if(NOT JIT_SONAME MATCHES "^libtriton_jit\\.so(\\.[0-9]+)*$")
  message(FATAL_ERROR "JIT_SONAME is invalid: ${JIT_SONAME}")
endif()
foreach(_expected_hash IN ITEMS
    JIT_SHA256
    JIT_PROVENANCE_SHA256
    STANDALONE_SHA256
    GEN_SSIG_SHA256
    ENVIRONMENT_IDENTITY)
  string(LENGTH "${${_expected_hash}}" _expected_hash_length)
  if(NOT _expected_hash_length EQUAL 64 OR
     NOT "${${_expected_hash}}" MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "${_expected_hash} is not a SHA-256 digest")
  endif()
endforeach()
if(NOT DEFINED INSTALL_CONFIG)
  set(INSTALL_CONFIG "")
endif()

file(REAL_PATH "${SOURCE_ROOT}" SOURCE_ROOT EXPAND_TILDE)
file(REAL_PATH "${BUILD_ROOT}" BUILD_ROOT EXPAND_TILDE)
cmake_path(ABSOLUTE_PATH TEST_ROOT NORMALIZE OUTPUT_VARIABLE TEST_ROOT)
file(RELATIVE_PATH _test_relative "${BUILD_ROOT}" "${TEST_ROOT}")
if(_test_relative STREQUAL "" OR _test_relative MATCHES "^\\.\\.($|/)")
  message(FATAL_ERROR "TEST_ROOT must be a strict child of BUILD_ROOT")
endif()

set(_flagdnn_clean_environment
  "${CMAKE_COMMAND}" -E env
  --unset=FLAGDNN_BACKEND
  --unset=FLAGDNN_BACKEND_PATH
  --unset=FLAGDNN_BACKEND_ROOT
  --unset=FLAGDNN_KERNEL_SOURCE_ROOT
  --unset=FLAGDNN_TUNING_ROOT
  --unset=FLAGDNN_MTHREADS_ENVIRONMENT_REPORT
  --unset=FLAGDNN_MTHREADS_TRITON_JIT_ROOT
  --unset=FLAGDNN_MTHREADS_TRITON_JIT_DIR
  --unset=FLAGDNN_MTHREADS_TRITON_JIT_LIBRARY
  --unset=FLAGDNN_MTHREADS_TRITON_JIT_INCLUDE_DIR
  --unset=FLAGDNN_MTHREADS_TRITON_JIT_SCRIPT_DIR
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
  --unset=FLAGDNN_COMPILER_TIMEOUT_SECONDS
  --unset=FLAGDNN_EXECUTION_ENGINE
  --unset=FLAGDNN_CACHE_DIRECTORY
  --unset=PYTHONPATH
  --unset=PYTHONHOME
  "TRITON_JIT_BACKEND=MTGPU"
  "TORCH_DEVICE_BACKEND_AUTOLOAD=0"
  "PYTHONDONTWRITEBYTECODE=1"
  "LD_LIBRARY_PATH=${VALIDATION_LIBRARY_PATH}"
  --)

function(_flagdnn_run_step _description)
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

function(_flagdnn_run_clean_step _description)
  _flagdnn_run_step(
    "${_description}" ${_flagdnn_clean_environment} ${ARGN})
endfunction()

function(_flagdnn_run_clean_identity_step _description)
  foreach(_attempt RANGE 1 3)
    execute_process(
      COMMAND ${_flagdnn_clean_environment} ${ARGN}
      RESULT_VARIABLE _result
      OUTPUT_VARIABLE _output
      ERROR_VARIABLE _error)
    if(_result EQUAL 0)
      if(NOT _output STREQUAL "")
        message(STATUS "${_description}:\n${_output}")
      endif()
      return()
    endif()
    if(NOT _result EQUAL 75)
      message(FATAL_ERROR
        "${_description} failed (${_result})\n"
        "stdout:\n${_output}\nstderr:\n${_error}")
    endif()
    message(STATUS
      "${_description}: dependency snapshot changed on attempt "
      "${_attempt}/3; retrying")
  endforeach()
  message(FATAL_ERROR
    "${_description} failed after three temporary identity failures\n"
    "stdout:\n${_output}\nstderr:\n${_error}")
endfunction()

function(_flagdnn_require_sha256 _path _expected)
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

function(_flagdnn_require_exact_rpath _path _expected)
  find_program(_flagdnn_readelf NAMES readelf llvm-readelf REQUIRED)
  execute_process(
    COMMAND "${_flagdnn_readelf}" -d "${_path}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _dynamic
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "cannot read ELF dynamic section for ${_path}: ${_error}")
  endif()
  string(REGEX MATCH
    "\\((RPATH|RUNPATH)\\)[^\n]*\\[([^]]*)\\]"
    _record "${_dynamic}")
  if(_record STREQUAL "" OR NOT CMAKE_MATCH_2 STREQUAL _expected)
    message(FATAL_ERROR
      "${_path}: runtime path '${CMAKE_MATCH_2}' is not '${_expected}'")
  endif()
  if(_dynamic MATCHES "\\(RUNPATH\\)")
    message(FATAL_ERROR
      "${_path}: expected fail-closed DT_RPATH, got DT_RUNPATH")
  endif()
endfunction()

function(_flagdnn_is_within _output _root _candidate)
  file(RELATIVE_PATH _relative "${_root}" "${_candidate}")
  if(NOT IS_ABSOLUTE "${_relative}" AND
     NOT _relative MATCHES "^\\.\\.($|/)")
    set(${_output} TRUE PARENT_SCOPE)
  else()
    set(${_output} FALSE PARENT_SCOPE)
  endif()
endfunction()

_flagdnn_require_exact_rpath(
  "${BUILD_PLUGIN}" "$ORIGIN/flagdnn/mthreads")

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")
set(_sdk "${TEST_ROOT}/sdk")
set(_consumer_build "${TEST_ROOT}/consumer-build")
set(_consumer_cache "${TEST_ROOT}/consumer-cache")
file(MAKE_DIRECTORY "${_consumer_cache}")

set(_install_command
    "${CMAKE_COMMAND}" --install "${BUILD_ROOT}" --prefix "${_sdk}")
if(NOT INSTALL_CONFIG STREQUAL "")
  list(APPEND _install_command --config "${INSTALL_CONFIG}")
endif()
_flagdnn_run_step(
  "install isolated MThreads FlagDNN SDK" ${_install_command})

if(EXISTS "${_sdk}/lib/cmake/FlagDNN/FlagDNNConfig.cmake")
  set(_flagdnn_dir "${_sdk}/lib/cmake/FlagDNN")
elseif(EXISTS "${_sdk}/lib64/cmake/FlagDNN/FlagDNNConfig.cmake")
  set(_flagdnn_dir "${_sdk}/lib64/cmake/FlagDNN")
else()
  message(FATAL_ERROR "installed SDK has no FlagDNNConfig.cmake")
endif()
get_filename_component(
  _sdk_library_directory "${_flagdnn_dir}/../.." REALPATH)
set(_installed_plugin "${_sdk_library_directory}/${PLUGIN_SONAME}")
set(_installed_jit
    "${_sdk_library_directory}/flagdnn/mthreads/${JIT_SONAME}")
set(_private_script_directory
    "${_sdk_library_directory}/flagdnn/share/triton_jit/scripts")
set(_installed_compiler
    "${_sdk}/share/flagdnn/compiler/flagdnn_codegen/main.py")
set(_installed_provider_root
    "${_sdk}/share/flagdnn/backends/mthreads")
set(_installed_provider "${_installed_provider_root}/compiler.py")
set(_installed_environment
    "${_installed_provider_root}/flagdnn_mthreads_environment.json")
set(_installed_metadata
    "${_installed_provider_root}/flagdnn_mthreads_install.json")

foreach(_required IN ITEMS
    "${_sdk}/include/flagdnn/flagdnn.h"
    "${_sdk}/include/flagdnn/flagdnn.hpp"
    "${_sdk}/include/flagdnn_frontend.h"
    "${_installed_compiler}"
    "${_sdk}/share/flagdnn/compiler/flagdnn_codegen/provider_loader.py"
    "${_sdk}/share/flagdnn/compiler/flagdnn_codegen/kernel_registry.py"
    "${_sdk}/share/flagdnn/kernels/registry.json"
    "${_sdk}/share/flagdnn/kernels/common/binary.py"
    "${_installed_provider}"
    "${_installed_provider_root}/dispatch/graph.py"
    "${_installed_provider_root}/codegen/identity.py"
    "${_installed_provider_root}/dispatch/tensor.py"
    "${_installed_provider_root}/environment_identity.py"
    "${_installed_provider_root}/dispatch/program.py"
    "${_installed_provider_root}/kernels/binary.py"
    "${_installed_provider_root}/kernels/registry.json"
    "${_installed_provider_root}/tuning/mthreads.yaml"
    "${_installed_environment}"
    "${_installed_metadata}"
    "${_installed_plugin}"
    "${_installed_jit}"
    "${_private_script_directory}/standalone_compile.py"
    "${_private_script_directory}/gen_ssig.py")
  if(NOT EXISTS "${_required}")
    message(FATAL_ERROR "installed SDK resource is missing: ${_required}")
  endif()
endforeach()

_flagdnn_require_exact_rpath(
  "${_installed_plugin}" "$ORIGIN/flagdnn/mthreads")
_flagdnn_require_sha256("${_installed_jit}" "${JIT_SHA256}")
_flagdnn_require_sha256(
  "${_private_script_directory}/standalone_compile.py"
  "${STANDALONE_SHA256}")
_flagdnn_require_sha256(
  "${_private_script_directory}/gen_ssig.py" "${GEN_SSIG_SHA256}")

file(READ "${_installed_metadata}" _metadata)
file(READ "${_installed_environment}" _environment)
foreach(_metadata_field IN ITEMS backend schema_version environment_identity_sha256)
  string(JSON _metadata_${_metadata_field} ERROR_VARIABLE _metadata_error
    GET "${_metadata}" ${_metadata_field})
  if(_metadata_error)
    message(FATAL_ERROR
      "installed MThreads metadata lacks ${_metadata_field}")
  endif()
endforeach()
if(NOT _metadata_backend STREQUAL "mthreads" OR
   NOT _metadata_schema_version EQUAL 1 OR
   NOT _metadata_environment_identity_sha256 STREQUAL ENVIRONMENT_IDENTITY)
  message(FATAL_ERROR "installed MThreads metadata identity is invalid")
endif()
string(JSON _metadata_jit_relative GET
  "${_metadata}" libtriton_jit relative_path)
string(JSON _metadata_jit_sha256 GET
  "${_metadata}" libtriton_jit sha256)
string(JSON _metadata_jit_provenance GET
  "${_metadata}" libtriton_jit provenance_sha256)
string(JSON _metadata_jit_soname GET
  "${_metadata}" libtriton_jit soname)
get_filename_component(
  _metadata_jit_path
  "${_installed_provider_root}/${_metadata_jit_relative}"
  REALPATH)
if(NOT _metadata_jit_path STREQUAL _installed_jit OR
   NOT _metadata_jit_sha256 STREQUAL JIT_SHA256 OR
   NOT _metadata_jit_provenance STREQUAL JIT_PROVENANCE_SHA256 OR
   NOT _metadata_jit_soname STREQUAL JIT_SONAME)
  message(FATAL_ERROR "installed MThreads private JIT metadata differs")
endif()
foreach(_script IN ITEMS gen_ssig.py standalone_compile.py)
  string(JSON _script_relative GET
    "${_metadata}" scripts "${_script}" relative_path)
  string(JSON _script_sha256 GET
    "${_metadata}" scripts "${_script}" sha256)
  string(JSON _script_provenance GET
    "${_metadata}" scripts "${_script}" provenance_sha256)
  get_filename_component(
    _script_path "${_installed_provider_root}/${_script_relative}" REALPATH)
  if(_script STREQUAL "gen_ssig.py")
    set(_expected_script_sha256 "${GEN_SSIG_SHA256}")
    set(_provenance_resource triton_jit_gen_ssig)
  else()
    set(_expected_script_sha256 "${STANDALONE_SHA256}")
    set(_provenance_resource triton_jit_standalone_compile)
  endif()
  string(JSON _expected_script_provenance GET
    "${_environment}" resources "${_provenance_resource}" sha256)
  if(NOT _script_path STREQUAL
         "${_private_script_directory}/${_script}" OR
     NOT _script_sha256 STREQUAL _expected_script_sha256 OR
     NOT _script_provenance STREQUAL _expected_script_provenance OR
     (_script STREQUAL "gen_ssig.py" AND
      NOT _script_sha256 STREQUAL _script_provenance))
    message(FATAL_ERROR
      "installed MThreads ${_script} metadata differs")
  endif()
endforeach()

set(_identity_manifest "${TEST_ROOT}/installed-mthreads.identity")
_flagdnn_run_clean_identity_step(
  "identify installed MThreads compiler resources"
  "${CODEGEN_PYTHON}" "${_installed_compiler}"
  --identify
  --backend mthreads
  --target musa-mtgpu-cc31-w32
  --execution-engine libtriton_jit
  --identity-output "${_identity_manifest}"
  --quiet)
file(STRINGS "${_identity_manifest}" _identity_lines LIMIT_COUNT 2)
list(LENGTH _identity_lines _identity_line_count)
if(NOT _identity_line_count EQUAL 2)
  message(FATAL_ERROR "installed MThreads identity manifest is malformed")
endif()
list(GET _identity_lines 1 _identity_metadata)
string(JSON _dependencies_complete ERROR_VARIABLE _identity_error
  GET "${_identity_metadata}" dependencies_complete)
string(JSON _dependency_count ERROR_VARIABLE _count_error
  LENGTH "${_identity_metadata}" files)
if(_identity_error OR _count_error OR NOT _dependencies_complete OR
   _dependency_count LESS 1)
  message(FATAL_ERROR
    "installed MThreads compiler identity dependencies are incomplete")
endif()

function(_flagdnn_identity_requires_path _expected)
  file(REAL_PATH "${_expected}" _expected_real)
  math(EXPR _dependency_last "${_dependency_count} - 1")
  foreach(_index RANGE 0 ${_dependency_last})
    string(JSON _dependency GET "${_identity_metadata}" files ${_index})
    if(EXISTS "${_dependency}")
      file(REAL_PATH "${_dependency}" _dependency_real)
      if(_dependency_real STREQUAL _expected_real)
        return()
      endif()
    endif()
  endforeach()
  message(FATAL_ERROR
    "installed MThreads identity omitted SDK resource: ${_expected}")
endfunction()

function(_flagdnn_identity_forbids_path _forbidden)
  if(NOT EXISTS "${_forbidden}")
    return()
  endif()
  file(REAL_PATH "${_forbidden}" _forbidden_real)
  math(EXPR _dependency_last "${_dependency_count} - 1")
  foreach(_index RANGE 0 ${_dependency_last})
    string(JSON _dependency GET "${_identity_metadata}" files ${_index})
    if(EXISTS "${_dependency}")
      file(REAL_PATH "${_dependency}" _dependency_real)
      if(_dependency_real STREQUAL _forbidden_real)
        message(FATAL_ERROR
          "installed MThreads identity reached external JIT resource: "
          "${_forbidden_real}")
      endif()
    endif()
  endforeach()
endfunction()

foreach(_identity_resource IN ITEMS
    "${_installed_compiler}"
    "${_sdk}/share/flagdnn/compiler/flagdnn_codegen/provider_loader.py"
    "${_sdk}/share/flagdnn/compiler/flagdnn_codegen/kernel_registry.py"
    "${_sdk}/share/flagdnn/kernels/registry.json"
    "${_installed_provider}"
    "${_installed_provider_root}/dispatch/graph.py"
    "${_installed_provider_root}/codegen/identity.py"
    "${_installed_provider_root}/dispatch/tensor.py"
    "${_installed_provider_root}/environment_identity.py"
    "${_installed_provider_root}/dispatch/program.py"
    "${_installed_provider_root}/kernels/binary.py"
    "${_installed_provider_root}/kernels/registry.json"
    "${_installed_provider_root}/tuning/mthreads.yaml"
    "${_installed_environment}"
    "${_installed_metadata}"
    "${_installed_jit}"
    "${_private_script_directory}/standalone_compile.py"
    "${_private_script_directory}/gen_ssig.py")
  _flagdnn_identity_requires_path("${_identity_resource}")
endforeach()

foreach(_original_resource IN ITEMS
    triton_jit
    triton_jit_config
    triton_jit_gen_ssig
    triton_jit_header
    triton_jit_standalone_compile)
  string(JSON _original_path GET
    "${_environment}" resources ${_original_resource} realpath)
  _flagdnn_identity_forbids_path("${_original_path}")
endforeach()

file(REAL_PATH "${_sdk}" _sdk_real)
math(EXPR _dependency_last "${_dependency_count} - 1")
foreach(_index RANGE 0 ${_dependency_last})
  string(JSON _dependency GET "${_identity_metadata}" files ${_index})
  if(NOT EXISTS "${_dependency}")
    message(FATAL_ERROR
      "installed MThreads identity dependency disappeared: ${_dependency}")
  endif()
  file(REAL_PATH "${_dependency}" _dependency_real)
  _flagdnn_is_within(_inside_sdk "${_sdk_real}" "${_dependency_real}")
  if(_inside_sdk)
    continue()
  endif()
  _flagdnn_is_within(
    _inside_source "${SOURCE_ROOT}" "${_dependency_real}")
  _flagdnn_is_within(
    _inside_build "${BUILD_ROOT}" "${_dependency_real}")
  if(_inside_source OR _inside_build)
    message(FATAL_ERROR
      "installed MThreads identity reached source/build tree: "
      "${_dependency_real}")
  endif()
endforeach()

_flagdnn_run_clean_step(
  "configure installed MThreads consumer"
  "${CMAKE_COMMAND}"
  -S "${SOURCE_ROOT}/backends/mthreads/validation/integration/installed_consumer"
  -B "${_consumer_build}"
  "-DFlagDNN_DIR=${_flagdnn_dir}"
  "-DFLAGDNN_INSTALLED_MTHREADS_SDK_ROOT=${_sdk}"
  "-DFLAGDNN_INSTALLED_MTHREADS_PLUGIN=${_installed_plugin}"
  "-DFLAGDNN_INSTALLED_MTHREADS_JIT=${_installed_jit}"
  "-DFLAGDNN_INSTALLED_MTHREADS_SCRIPT_DIR=${_private_script_directory}"
  "-DFLAGDNN_INSTALLED_MTHREADS_CACHE=${_consumer_cache}"
  "-DFLAGDNN_INSTALLED_MUSA_ROOT=${MUSA_ROOT}"
  "-DFLAGDNN_INSTALLED_MUDNN_INCLUDE_DIR=${MUDNN_INCLUDE_DIR}"
  "-DFLAGDNN_INSTALLED_MUDNN_LIBRARY=${MUDNN_LIBRARY}"
  "-DFLAGDNN_INSTALLED_LIBRARY_PATH=${VALIDATION_LIBRARY_PATH}")

set(_consumer_build_command
    "${CMAKE_COMMAND}" --build "${_consumer_build}" --parallel 2)
if(NOT INSTALL_CONFIG STREQUAL "")
  list(APPEND _consumer_build_command --config "${INSTALL_CONFIG}")
endif()
_flagdnn_run_clean_step(
  "build installed MThreads consumer" ${_consumer_build_command})

set(_consumer_test_command
    "${CMAKE_COMMAND}" -E env
    --unset=PYTHONDONTWRITEBYTECODE
    --
    "${CMAKE_CTEST_COMMAND}" --test-dir "${_consumer_build}"
    --output-on-failure -V)
if(NOT INSTALL_CONFIG STREQUAL "")
  list(APPEND _consumer_test_command -C "${INSTALL_CONFIG}")
endif()
_flagdnn_run_clean_step(
  "run installed MThreads consumer" ${_consumer_test_command})

set(_leak_tokens
    "${SOURCE_ROOT}/backends"
    "${SOURCE_ROOT}/compiler"
    "${SOURCE_ROOT}/kernels"
    "${BUILD_ROOT}/backends/mthreads")
file(GLOB_RECURSE _cache_files LIST_DIRECTORIES FALSE "${_consumer_cache}/*")
foreach(_cache_file IN LISTS _cache_files)
  if(_cache_file MATCHES "\\.(json|py|txt)$")
    file(SIZE "${_cache_file}" _cache_size)
    if(_cache_size LESS 2097152)
      file(READ "${_cache_file}" _cache_contents)
      foreach(_leak IN LISTS _leak_tokens)
        string(FIND "${_cache_contents}" "${_leak}" _leak_position)
        if(NOT _leak_position EQUAL -1)
          message(FATAL_ERROR
            "installed MThreads artifact reached source/build resource: "
            "${_leak}")
        endif()
      endforeach()
    endif()
  endif()
endforeach()

message(STATUS "Installed MThreads SDK consumer contract verified")
