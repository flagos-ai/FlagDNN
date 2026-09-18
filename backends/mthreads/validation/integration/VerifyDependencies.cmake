cmake_minimum_required(VERSION 3.23)

foreach(_required_variable IN ITEMS
    FLAGDNN_SOURCE_DIR
    FLAGDNN_BINARY_ROOT
    FLAGDNN_CODEGEN_PYTHON
    FLAGDNN_MUSA_ROOT
    FLAGDNN_TRITON_JIT_DIR)
  if(NOT DEFINED ${_required_variable} OR
     "${${_required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${_required_variable} is required")
  endif()
endforeach()

file(REAL_PATH "${FLAGDNN_SOURCE_DIR}" _source_dir EXPAND_TILDE)
file(MAKE_DIRECTORY "${FLAGDNN_BINARY_ROOT}")
file(REAL_PATH "${FLAGDNN_BINARY_ROOT}" _binary_root EXPAND_TILDE)
cmake_path(IS_PREFIX _source_dir "${_binary_root}" NORMALIZE _inside_source)
if(NOT _inside_source)
  message(FATAL_ERROR
    "configuration contract binary root must remain below the source tree")
endif()

set(_base_arguments
    -DFLAGDNN_BACKENDS=mthreads
    -DFLAGDNN_DEFAULT_BACKEND=mthreads
    -DFLAGDNN_EXECUTION_ENGINE=libtriton_jit
    -DFLAGDNN_BUILD_TESTS=OFF
    -DFLAGDNN_BUILD_BENCHMARKS=OFF
    -DBUILD_TESTING=OFF
    "-DFLAGDNN_CODEGEN_PYTHON=${FLAGDNN_CODEGEN_PYTHON}"
    "-DFLAGDNN_MTHREADS_TRITON_JIT_DIR=${FLAGDNN_TRITON_JIT_DIR}")

function(_flagdnn_expect_configure_failure name expected_message)
  set(_build_dir "${_binary_root}/${name}")
  file(REMOVE_RECURSE "${_build_dir}")
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}"
      -S "${_source_dir}"
      -B "${_build_dir}"
      ${_base_arguments}
      ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  if(_result EQUAL 0)
    message(FATAL_ERROR
      "${name} unexpectedly configured successfully")
  endif()
  set(_diagnostic "${_stdout}\n${_stderr}")
  string(REGEX REPLACE "[ \t\r\n]+" " "
    _normalized_diagnostic "${_diagnostic}")
  string(FIND
    "${_normalized_diagnostic}"
    "${expected_message}"
    _message_offset)
  if(_message_offset EQUAL -1)
    message(FATAL_ERROR
      "${name} failed without the required diagnostic "
      "'${expected_message}':\n${_diagnostic}")
  endif()
endfunction()

function(_flagdnn_require_python_cache_isolation)
  set(_name preserved_python_cache)
  set(_build_dir "${_binary_root}/${_name}")
  set(_sentinel "${_build_dir}/caller-python-sentinel")
  file(REMOVE_RECURSE "${_build_dir}")
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "FLAGDNN_MTHREADS_MUSA_ROOT=${_build_dir}/caller-musa-sentinel"
      "${CMAKE_COMMAND}"
      -S "${_source_dir}"
      -B "${_build_dir}"
      ${_base_arguments}
      "-DFLAGDNN_MTHREADS_MUSA_ROOT=${FLAGDNN_MUSA_ROOT}"
      "-DPython_EXECUTABLE:FILEPATH=${_sentinel}"
      "-DTritonJIT_DIR:PATH=${_build_dir}/caller-jit-sentinel"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "${_name} failed to configure:\n${_stdout}\n${_stderr}")
  endif()
  file(STRINGS "${_build_dir}/CMakeCache.txt" _python_cache
    REGEX "^Python_EXECUTABLE:")
  if(NOT _python_cache STREQUAL
      "Python_EXECUTABLE:FILEPATH=${_sentinel}")
    message(FATAL_ERROR
      "mthreads rewrote the caller Python cache: ${_python_cache}")
  endif()
endfunction()

# An omitted backend-specific path must use the normal environment hints.
set(_auto_build_dir "${_binary_root}/automatic_dependency_paths")
file(REMOVE_RECURSE "${_auto_build_dir}")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env
    "MUSA_HOME=${FLAGDNN_MUSA_ROOT}"
    --unset=FLAGDNN_MTHREADS_MUSA_ROOT
    --unset=FLAGDNN_MTHREADS_TRITON_JIT_DIR
    --unset=TritonJIT_DIR
    --unset=LIBTRITON_JIT_ROOT
    "${CMAKE_COMMAND}" -S "${_source_dir}" -B "${_auto_build_dir}"
    ${_base_arguments}
    -DFLAGDNN_MTHREADS_MUSA_ROOT:PATH=
    -DFLAGDNN_MTHREADS_TRITON_JIT_DIR:PATH=
    "-DTritonJIT_DIR:PATH=${FLAGDNN_TRITON_JIT_DIR}"
  RESULT_VARIABLE _auto_result
  OUTPUT_VARIABLE _auto_stdout
  ERROR_VARIABLE _auto_stderr)
if(NOT _auto_result EQUAL 0)
  message(FATAL_ERROR
    "automatic dependency discovery failed:\n${_auto_stdout}\n${_auto_stderr}")
endif()
file(STRINGS "${_auto_build_dir}/CMakeCache.txt" _auto_paths
  REGEX "^FLAGDNN_MTHREADS_(MUSA_ROOT|TRITON_JIT_DIR):PATH=")
foreach(_expected IN ITEMS
    "FLAGDNN_MTHREADS_MUSA_ROOT:PATH=${FLAGDNN_MUSA_ROOT}"
    "FLAGDNN_MTHREADS_TRITON_JIT_DIR:PATH=${FLAGDNN_TRITON_JIT_DIR}")
  if(NOT _expected IN_LIST _auto_paths)
    message(FATAL_ERROR "automatic dependency discovery did not select ${_expected}")
  endif()
endforeach()

_flagdnn_expect_configure_failure(
  missing_triton_jit_directory
  "FLAGDNN_MTHREADS_TRITON_JIT_DIR is not a directory"
  "-DFLAGDNN_MTHREADS_MUSA_ROOT=${FLAGDNN_MUSA_ROOT}"
  "-DFLAGDNN_MTHREADS_TRITON_JIT_DIR=${_binary_root}/missing-jit")

_flagdnn_expect_configure_failure(
  invalid_musa_root
  "FLAGDNN_MTHREADS_MUSA_ROOT is not a directory"
  "-DFLAGDNN_MTHREADS_MUSA_ROOT=${_binary_root}/missing-musa")

_flagdnn_require_python_cache_isolation()

message(STATUS
  "mthreads configure-failure contracts passed: "
  "invalid root, missing TritonJIT directory; "
  "automatic discovery, explicit JIT selection and Python cache isolation passed")
