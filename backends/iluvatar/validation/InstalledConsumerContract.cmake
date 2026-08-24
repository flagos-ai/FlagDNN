if(NOT DEFINED SOURCE_ROOT OR NOT IS_DIRECTORY "${SOURCE_ROOT}")
  message(FATAL_ERROR "SOURCE_ROOT is missing")
endif()
if(NOT DEFINED BUILD_ROOT OR
   NOT EXISTS "${BUILD_ROOT}/CMakeCache.txt")
  message(FATAL_ERROR "BUILD_ROOT is not a configured FlagDNN build")
endif()
if(NOT DEFINED TEST_ROOT OR TEST_ROOT STREQUAL "")
  message(FATAL_ERROR "TEST_ROOT is missing")
endif()
foreach(_required IN ITEMS
    CODEGEN_PYTHON
    COREX_ROOT
    COREX_DRIVER
    JIT_CONFIG_DIR
    JIT_LIBRARY
    JIT_INCLUDE_DIR
    JIT_SCRIPT_DIR)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "" OR
     NOT EXISTS "${${_required}}")
    message(FATAL_ERROR "${_required} must name an existing path")
  endif()
endforeach()
if(NOT DEFINED EXTERNAL_PYTHONPATH OR EXTERNAL_PYTHONPATH STREQUAL "")
  message(FATAL_ERROR "EXTERNAL_PYTHONPATH is missing")
endif()
string(REPLACE ":" ";" _external_python_paths "${EXTERNAL_PYTHONPATH}")
foreach(_python_path IN LISTS _external_python_paths)
  if(NOT IS_DIRECTORY "${_python_path}" OR
     _python_path MATCHES "^${SOURCE_ROOT}(/|$)" OR
     _python_path MATCHES "^${BUILD_ROOT}(/|$)")
    message(FATAL_ERROR
      "external Python path is missing or points into FlagDNN: ${_python_path}")
  endif()
endforeach()
if(NOT JIT_CONFIG_DIR STREQUAL "/usr/local/lib/cmake/TritonJIT")
  message(FATAL_ERROR
    "installed consumer must use /usr/local/lib/cmake/TritonJIT")
endif()

set(_clean_environment
  "${CMAKE_COMMAND}" -E env
  --unset=FLAGDNN_BACKEND
  --unset=FLAGDNN_BACKEND_PATH
  --unset=FLAGDNN_BACKEND_ROOT
  --unset=FLAGDNN_KERNEL_SOURCE_ROOT
  --unset=FLAGDNN_TUNING_ROOT
  --unset=FLAGDNN_CODEGEN_COMPILER
  --unset=FLAGDNN_CODEGEN_PYTHON
  --unset=FLAGDNN_COMPILER
  --unset=FLAGDNN_COMPILER_EXECUTABLE
  --unset=FLAGDNN_COMPILER_TIMEOUT_SECONDS
  --unset=FLAGDNN_EXECUTION_ENGINE
  --unset=FLAGDNN_CACHE_DIRECTORY
  --unset=FLAGDNN_ILUVATAR_COREX_ROOT
  --unset=FLAGDNN_ILUVATAR_CUDA_INCLUDE_DIR
  --unset=FLAGDNN_ILUVATAR_CUDA_DRIVER_LIBRARY
  --unset=FLAGDNN_ILUVATAR_TRITON_JIT_ROOT
  --unset=FLAGDNN_ILUVATAR_TRITON_JIT_DIR
  --unset=FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY
  --unset=FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR
  --unset=FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR
  --unset=FLAGDNN_ILUVATAR_PYTHONPATH
  --unset=LIBTRITON_JIT_ROOT
  --unset=TritonJIT_DIR
  --unset=TritonJIT_ROOT
  --unset=TRITONJIT_ROOT
  --unset=LD_LIBRARY_PATH
  --unset=PYTHONPATH
  --unset=PYTHONHOME)

function(_run_step description)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _output
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "${description} failed (${_result})\n"
      "stdout:\n${_output}\nstderr:\n${_error}")
  endif()
  if(NOT _output STREQUAL "")
    message(STATUS "${description}:\n${_output}")
  endif()
endfunction()

function(_run_clean_step description)
  _run_step("${description}" ${_clean_environment} ${ARGN})
endfunction()

function(_require_identity_path metadata expected)
  file(REAL_PATH "${expected}" _expected_real)
  string(JSON _count ERROR_VARIABLE _json_error
    LENGTH "${metadata}" files)
  if(_json_error OR _count LESS 1)
    message(FATAL_ERROR "installed compiler identity has no dependencies")
  endif()
  math(EXPR _last "${_count} - 1")
  foreach(_index RANGE 0 ${_last})
    string(JSON _dependency GET "${metadata}" files ${_index})
    if(EXISTS "${_dependency}")
      file(REAL_PATH "${_dependency}" _dependency_real)
      if(_dependency_real STREQUAL _expected_real)
        return()
      endif()
    endif()
  endforeach()
  message(FATAL_ERROR
    "installed compiler identity omitted resource: ${expected}")
endfunction()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")
set(_sdk "${TEST_ROOT}/sdk")
set(_consumer_source "${TEST_ROOT}/consumer-source")
set(_consumer_build "${TEST_ROOT}/consumer-build")
set(_cache "${TEST_ROOT}/cache")
file(MAKE_DIRECTORY "${_cache}")

_run_clean_step(
  "install isolated Iluvatar FlagDNN SDK"
  -- "${SOURCE_ROOT}/tools/install.sh"
  --build-dir "${BUILD_ROOT}"
  --prefix "${_sdk}")

if(EXISTS "${_sdk}/lib/cmake/FlagDNN/FlagDNNConfig.cmake")
  set(_flagdnn_dir "${_sdk}/lib/cmake/FlagDNN")
  set(_library_directory "${_sdk}/lib")
elseif(EXISTS "${_sdk}/lib64/cmake/FlagDNN/FlagDNNConfig.cmake")
  set(_flagdnn_dir "${_sdk}/lib64/cmake/FlagDNN")
  set(_library_directory "${_sdk}/lib64")
else()
  message(FATAL_ERROR "installed SDK has no FlagDNNConfig.cmake")
endif()

set(_installed_compiler
  "${_sdk}/share/flagdnn/compiler/flagdnn_codegen/main.py")
set(_installed_provider
  "${_sdk}/share/flagdnn/backends/iluvatar/compiler.py")
set(_installed_identity
  "${_sdk}/share/flagdnn/backends/iluvatar/compiler_identity.py")
set(_installed_environment
  "${_sdk}/share/flagdnn/backends/iluvatar/python_environment_identity.py")
set(_installed_platform_registry
  "${_sdk}/share/flagdnn/backends/iluvatar/kernels/registry.json")
set(_installed_common_registry
  "${_sdk}/share/flagdnn/kernels/registry.json")
set(_installed_common_kernel
  "${_sdk}/share/flagdnn/kernels/common/binary.py")
set(_installed_platform_binary_kernel
  "${_sdk}/share/flagdnn/backends/iluvatar/kernels/binary.py")
set(_installed_platform_unary_kernel
  "${_sdk}/share/flagdnn/backends/iluvatar/kernels/unary.py")
set(_installed_platform_layout_kernel
  "${_sdk}/share/flagdnn/backends/iluvatar/kernels/layout.py")
set(_installed_platform_tuning
  "${_sdk}/share/flagdnn/backends/iluvatar/tuning/common.yaml")
file(GLOB _installed_plugins
  "${_library_directory}/libflagdnn_backend_iluvatar.so*")
list(LENGTH _installed_plugins _plugin_count)
if(_plugin_count LESS 1)
  message(FATAL_ERROR "installed SDK has no Iluvatar backend plugin")
endif()
foreach(_required IN ITEMS
    "${_installed_compiler}"
    "${_installed_provider}"
    "${_installed_identity}"
    "${_installed_environment}"
    "${_installed_platform_registry}"
    "${_installed_common_registry}"
    "${_installed_common_kernel}"
    "${_installed_platform_binary_kernel}"
    "${_installed_platform_unary_kernel}"
    "${_installed_platform_layout_kernel}"
    "${_installed_platform_tuning}")
  if(NOT EXISTS "${_required}")
    message(FATAL_ERROR "installed SDK resource is missing: ${_required}")
  endif()
endforeach()

set(_install_manifest "${BUILD_ROOT}/install_manifest.txt")
if(NOT EXISTS "${_install_manifest}")
  message(FATAL_ERROR "install_manifest.txt was not generated")
endif()
file(STRINGS "${_install_manifest}" _installed_files)
foreach(_installed_file IN LISTS _installed_files)
  get_filename_component(_installed_name "${_installed_file}" NAME)
  if(_installed_name MATCHES
       "^(libtriton_jit\\.so(\\..*)?|standalone_compile\\.py|gen_ssig\\.py|libcudnn\\.so(\\..*)?|cuda\\.h)$")
    message(FATAL_ERROR
      "FlagDNN installed forbidden external resource: ${_installed_file}")
  endif()
endforeach()

find_program(_readelf NAMES readelf llvm-readelf REQUIRED)
list(GET _installed_plugins 0 _installed_plugin)
execute_process(
  COMMAND "${_readelf}" -d "${_installed_plugin}"
  RESULT_VARIABLE _readelf_result
  OUTPUT_VARIABLE _dynamic_section
  ERROR_VARIABLE _readelf_error)
if(NOT _readelf_result EQUAL 0)
  message(FATAL_ERROR "cannot inspect installed plugin: ${_readelf_error}")
endif()
if(NOT _dynamic_section MATCHES "\\$ORIGIN" OR
   NOT _dynamic_section MATCHES "/usr/local/lib")
  message(FATAL_ERROR
    "installed plugin RUNPATH does not expose external /usr/local IX JIT")
endif()
if(_dynamic_section MATCHES "${SOURCE_ROOT}" OR
   _dynamic_section MATCHES "${BUILD_ROOT}" OR
   _dynamic_section MATCHES "/home/wbj/libtriton_jit")
  message(FATAL_ERROR
    "installed plugin RUNPATH contains a source/build-tree dependency")
endif()

set(_identity_output "${TEST_ROOT}/installed-iluvatar.identity")
_run_clean_step(
  "identify installed Iluvatar compiler resources"
  "FLAGDNN_ILUVATAR_COREX_ROOT=${COREX_ROOT}"
  "FLAGDNN_ILUVATAR_TRITON_JIT_DIR=${JIT_CONFIG_DIR}"
  "FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY=${JIT_LIBRARY}"
  "FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR=${JIT_INCLUDE_DIR}"
  "FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR=${JIT_SCRIPT_DIR}"
  "PYTHONPATH=${EXTERNAL_PYTHONPATH}"
  -- "${CODEGEN_PYTHON}" "${_installed_compiler}"
  --identify
  --backend iluvatar
  --target corex_71
  --execution-engine libtriton_jit
  --identity-output "${_identity_output}"
  --quiet)
file(STRINGS "${_identity_output}" _identity_lines LIMIT_COUNT 2)
list(LENGTH _identity_lines _identity_line_count)
if(NOT _identity_line_count EQUAL 2)
  message(FATAL_ERROR "installed compiler identity manifest is malformed")
endif()
list(GET _identity_lines 1 _identity_metadata)
string(JSON _dependencies_complete ERROR_VARIABLE _identity_error
  GET "${_identity_metadata}" dependencies_complete)
if(_identity_error OR NOT _dependencies_complete)
  message(FATAL_ERROR
    "installed Iluvatar compiler did not report complete dependencies")
endif()
foreach(_identity_resource IN ITEMS
    "${_installed_compiler}"
    "${_installed_provider}"
    "${_installed_identity}"
    "${_installed_environment}"
    "${_installed_platform_registry}"
    "${_installed_common_registry}"
    "${_installed_platform_binary_kernel}"
    "${_installed_platform_unary_kernel}"
    "${_installed_platform_layout_kernel}"
    "${_installed_platform_tuning}"
    "${JIT_LIBRARY}"
    "${JIT_SCRIPT_DIR}/standalone_compile.py"
    "${JIT_SCRIPT_DIR}/gen_ssig.py")
  _require_identity_path("${_identity_metadata}" "${_identity_resource}")
endforeach()
string(JSON _dependency_count LENGTH "${_identity_metadata}" files)
math(EXPR _dependency_last "${_dependency_count} - 1")
foreach(_index RANGE 0 ${_dependency_last})
  string(JSON _dependency GET "${_identity_metadata}" files ${_index})
  if(_dependency MATCHES "^${_sdk}(/|$)")
    continue()
  endif()
  if(_dependency MATCHES "^${SOURCE_ROOT}(/|$)" OR
     _dependency MATCHES "^${BUILD_ROOT}(/|$)" OR
     _dependency MATCHES "^/home/wbj/libtriton_jit(/|$)")
    message(FATAL_ERROR
      "installed compiler identity leaked a source/build path: ${_dependency}")
  endif()
endforeach()

set(_compiler_wrapper "${TEST_ROOT}/compiler-wrapper.sh")
set(_compiler_log "${TEST_ROOT}/compiler.log")
file(WRITE "${_compiler_log}" "")
file(WRITE "${_compiler_wrapper}"
  "#!/bin/sh\n"
  "for arg in \"$@\"; do\n"
  "  if [ \"$arg\" = \"--request\" ]; then\n"
  "    printf 'compile\\n' >> \"$FLAGDNN_TEST_COMPILER_LOG\"\n"
  "  fi\n"
  "done\n"
  "PYTHONPATH=\"\$FLAGDNN_TEST_EXTERNAL_PYTHONPATH\" "
  "exec \"\$FLAGDNN_TEST_REAL_PYTHON\" \"\$@\"\n")
file(CHMOD "${_compiler_wrapper}"
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)

file(COPY
  "${SOURCE_ROOT}/backends/iluvatar/validation/installed_consumer/"
  DESTINATION "${_consumer_source}")
_run_clean_step(
  "configure installed Iluvatar consumer"
  -- "${CMAKE_COMMAND}"
  -S "${_consumer_source}"
  -B "${_consumer_build}"
  "-DFlagDNN_DIR=${_flagdnn_dir}"
  "-DTritonJIT_DIR=${JIT_CONFIG_DIR}"
  "-DFLAGDNN_INSTALLED_SDK=${_sdk}"
  "-DFLAGDNN_INSTALLED_LIBRARY_DIRECTORY=${_library_directory}"
  "-DFLAGDNN_INSTALLED_CODEGEN_PYTHON=${CODEGEN_PYTHON}"
  "-DFLAGDNN_INSTALLED_COMPILER_WRAPPER=${_compiler_wrapper}"
  "-DFLAGDNN_INSTALLED_COMPILER_LOG=${_compiler_log}"
  "-DFLAGDNN_INSTALLED_CACHE_DIRECTORY=${_cache}"
  "-DFLAGDNN_INSTALLED_COREX_ROOT=${COREX_ROOT}"
  "-DFLAGDNN_INSTALLED_COREX_DRIVER=${COREX_DRIVER}"
  "-DFLAGDNN_INSTALLED_TRITON_JIT_CONFIG_DIR=${JIT_CONFIG_DIR}"
  "-DFLAGDNN_INSTALLED_TRITON_JIT_LIBRARY=${JIT_LIBRARY}"
  "-DFLAGDNN_INSTALLED_TRITON_JIT_INCLUDE_DIR=${JIT_INCLUDE_DIR}"
  "-DFLAGDNN_INSTALLED_TRITON_JIT_SCRIPT_DIR=${JIT_SCRIPT_DIR}"
  "-DFLAGDNN_INSTALLED_EXTERNAL_PYTHONPATH=${EXTERNAL_PYTHONPATH}")
_run_clean_step(
  "build installed Iluvatar consumer"
  -- "${CMAKE_COMMAND}" --build "${_consumer_build}" --parallel 2)
_run_clean_step(
  "run installed Iluvatar consumer"
  -- "${CMAKE_CTEST_COMMAND}" --test-dir "${_consumer_build}" -V)

file(STRINGS "${_compiler_log}" _compiler_invocations)
list(LENGTH _compiler_invocations _compile_count)
if(NOT _compile_count EQUAL 1)
  message(FATAL_ERROR
    "two installed Graph builds must compile exactly once; got ${_compile_count}")
endif()
file(GLOB_RECURSE _materialized_sources
  "${_cache}/*/generated_stage_*.py")
list(LENGTH _materialized_sources _materialized_count)
if(NOT _materialized_count EQUAL 3)
  message(FATAL_ERROR
    "installed cache must contain exactly three materialized Graph kernels")
endif()
list(SORT _materialized_sources)
set(_expected_kernel_sources
  "${_installed_platform_binary_kernel}"
  "${_installed_platform_unary_kernel}"
  "${_installed_platform_layout_kernel}")
set(_expected_operations add relu transpose)
set(_expected_source_names binary.py unary.py layout.py)
file(GLOB_RECURSE _artifact_manifests "${_cache}/*/manifest.json")
list(LENGTH _artifact_manifests _manifest_count)
if(NOT _manifest_count EQUAL 1)
  message(FATAL_ERROR
    "installed cache must contain exactly one artifact manifest")
endif()
list(GET _artifact_manifests 0 _artifact_manifest)
file(READ "${_artifact_manifest}" _artifact_metadata)
foreach(_index RANGE 0 2)
  list(GET _materialized_sources ${_index} _materialized_source)
  list(GET _expected_kernel_sources ${_index} _installed_kernel_source)
  list(GET _expected_operations ${_index} _expected_operation)
  list(GET _expected_source_names ${_index} _expected_source_name)
  string(JSON _operation GET
    "${_artifact_metadata}" program stages ${_index} operation)
  string(JSON _ownership GET
    "${_artifact_metadata}" program stages ${_index} kernel ownership)
  string(JSON _provider GET
    "${_artifact_metadata}" program stages ${_index} kernel provider)
  string(JSON _source_name GET
    "${_artifact_metadata}" program stages ${_index} kernel source)
  string(JSON _materialized_name GET
    "${_artifact_metadata}" program stages ${_index}
    kernel materialized_source path)
  string(JSON _recorded_materialized_sha256 GET
    "${_artifact_metadata}" program stages ${_index}
    kernel materialized_source sha256)
  get_filename_component(_actual_materialized_name
    "${_materialized_source}" NAME)
  if(NOT _operation STREQUAL _expected_operation OR
     NOT _ownership STREQUAL "platform" OR
     NOT _provider STREQUAL "iluvatar_triton" OR
     NOT _source_name STREQUAL _expected_source_name OR
     NOT _materialized_name STREQUAL _actual_materialized_name)
    message(FATAL_ERROR
      "installed Graph stage ${_index} provenance metadata is invalid")
  endif()
  file(SHA256 "${_materialized_source}" _materialized_sha256)
  if(NOT _materialized_sha256 STREQUAL _recorded_materialized_sha256)
    message(FATAL_ERROR
      "installed Graph stage ${_index} materialized hash is invalid")
  endif()
  file(SHA256 "${_installed_kernel_source}" _installed_kernel_sha256)
  string(JSON _registry_source_sha256 ERROR_VARIABLE _registry_hash_error
    GET "${_artifact_metadata}" program stages ${_index}
    kernel registry_source_sha256)
  if(_registry_hash_error)
    set(_registry_source_sha256 "${_materialized_sha256}")
  endif()
  if(NOT _registry_source_sha256 STREQUAL _installed_kernel_sha256)
    message(FATAL_ERROR
      "Graph kernel ${_index} did not originate in the installed SDK")
  endif()
endforeach()

message(STATUS
  "PASS installed Iluvatar consumer; external_jit=${JIT_LIBRARY}")
