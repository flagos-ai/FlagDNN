include_guard(GLOBAL)

function(_flagdnn_mthreads_json_get out_value json)
  string(JSON _value ERROR_VARIABLE _error GET "${json}" ${ARGN})
  if(NOT _error STREQUAL "NOTFOUND")
    message(FATAL_ERROR "invalid mthreads environment JSON: ${_error}")
  endif()
  set("${out_value}" "${_value}" PARENT_SCOPE)
endfunction()

function(flagdnn_mthreads_resolve_dependencies out_prefix)
  set(options)
  set(one_value_arguments
      MUSA_ROOT
      CODEGEN_PYTHON
      TRITON_JIT_DIR
      ENVIRONMENT_REPORT)
  cmake_parse_arguments(
    MTHREADS
    "${options}"
    "${one_value_arguments}"
    ""
    ${ARGN})

  if(MTHREADS_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "flagdnn_mthreads_resolve_dependencies received unknown arguments: "
      "${MTHREADS_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT MTHREADS_MUSA_ROOT)
    message(FATAL_ERROR "FLAGDNN_MTHREADS_MUSA_ROOT is required")
  endif()
  if(NOT MTHREADS_CODEGEN_PYTHON)
    message(FATAL_ERROR "FLAGDNN_CODEGEN_PYTHON is required for mthreads")
  endif()
  if(NOT MTHREADS_TRITON_JIT_DIR)
    message(FATAL_ERROR "FLAGDNN_MTHREADS_TRITON_JIT_DIR is required")
  endif()
  if(NOT MTHREADS_ENVIRONMENT_REPORT)
    message(FATAL_ERROR "mthreads environment report output is required")
  endif()

  if(NOT IS_DIRECTORY "${MTHREADS_MUSA_ROOT}")
    message(FATAL_ERROR
      "FLAGDNN_MTHREADS_MUSA_ROOT is not a directory: "
      "${MTHREADS_MUSA_ROOT}")
  endif()
  if(NOT EXISTS "${MTHREADS_CODEGEN_PYTHON}")
    message(FATAL_ERROR
      "FLAGDNN_CODEGEN_PYTHON does not exist: "
      "${MTHREADS_CODEGEN_PYTHON}")
  endif()
  if(NOT IS_DIRECTORY "${MTHREADS_TRITON_JIT_DIR}")
    message(FATAL_ERROR
      "FLAGDNN_MTHREADS_TRITON_JIT_DIR is not a directory: "
      "${MTHREADS_TRITON_JIT_DIR}")
  endif()

  file(REAL_PATH "${MTHREADS_MUSA_ROOT}" _musa_root EXPAND_TILDE)
  get_filename_component(
    _codegen_python "${MTHREADS_CODEGEN_PYTHON}" ABSOLUTE)
  file(REAL_PATH "${MTHREADS_TRITON_JIT_DIR}" _triton_jit_dir EXPAND_TILDE)
  set(_triton_jit_config "${_triton_jit_dir}/TritonJITConfig.cmake")
  if(NOT EXISTS "${_triton_jit_config}")
    message(FATAL_ERROR
      "TritonJITConfig.cmake is missing below "
      "FLAGDNN_MTHREADS_TRITON_JIT_DIR")
  endif()
  file(REAL_PATH "${_triton_jit_config}" _triton_jit_config EXPAND_TILDE)
  if(EXISTS "${_triton_jit_dir}/CMakeCache.txt")
    # Build-tree exports keep headers and scripts in the source checkout.
    load_cache("${_triton_jit_dir}" READ_WITH_PREFIX _jit_build_
      CMAKE_HOME_DIRECTORY)
    set(_triton_jit_prefix "${_jit_build_CMAKE_HOME_DIRECTORY}")
    if(NOT IS_DIRECTORY "${_triton_jit_prefix}/include/triton_jit")
      message(FATAL_ERROR "TritonJIT build-tree source directory is missing")
    endif()
  else()
    get_filename_component(_triton_jit_cmake_root "${_triton_jit_dir}" DIRECTORY)
    get_filename_component(_triton_jit_library_root "${_triton_jit_cmake_root}" DIRECTORY)
    get_filename_component(_triton_jit_prefix "${_triton_jit_library_root}" DIRECTORY)
  endif()
  get_filename_component(
    _environment_directory "${MTHREADS_ENVIRONMENT_REPORT}" DIRECTORY)
  file(MAKE_DIRECTORY "${_environment_directory}")
  get_filename_component(
    _environment_report "${MTHREADS_ENVIRONMENT_REPORT}" ABSOLUTE)

  set(_environment_collector
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../environment_identity.py")
  if(NOT EXISTS "${_environment_collector}")
    message(FATAL_ERROR
      "required mthreads environment identity helper is missing: "
      "${_environment_collector}")
  endif()

  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "MUSA_HOME=${_musa_root}"
      "FLAGDNN_MTHREADS_MUSA_ROOT=${_musa_root}"
      "MTHREADS_TRITON_JIT_PREFIX=${_triton_jit_prefix}"
      "MTHREADS_TRITON_JIT_CONFIG=${_triton_jit_config}"
      "${_codegen_python}" "${_environment_collector}"
      --output "${_environment_report}"
    RESULT_VARIABLE _validation_result
    OUTPUT_VARIABLE _validation_output
    ERROR_VARIABLE _validation_error)
  if(NOT _validation_result EQUAL 0)
    string(STRIP "${_validation_error}" _validation_error)
    message(FATAL_ERROR
      "mthreads environment discovery failed: ${_validation_error}")
  endif()

  file(READ "${_environment_report}" _environment_json)
  _flagdnn_mthreads_json_get(
    _identity "${_environment_json}" identity_sha256)
  _flagdnn_mthreads_json_get(
    _reported_root "${_environment_json}"
    resources musa_root realpath)
  _flagdnn_mthreads_json_get(
    _driver_library "${_environment_json}"
    resources musa_driver realpath)
  _flagdnn_mthreads_json_get(
    _runtime_library "${_environment_json}"
    resources musa_runtime realpath)
  _flagdnn_mthreads_json_get(
    _mudnn_library "${_environment_json}"
    resources mudnn realpath)
  _flagdnn_mthreads_json_get(
    _driver_header "${_environment_json}"
    resources musa_driver_header realpath)
  _flagdnn_mthreads_json_get(
    _runtime_header "${_environment_json}"
    resources musa_runtime_header realpath)
  _flagdnn_mthreads_json_get(
    _mudnn_header "${_environment_json}"
    resources mudnn_header realpath)
  string(JSON _jit_dependency_count ERROR_VARIABLE _dependency_error
    LENGTH "${_environment_json}" resources triton_jit dependencies)
  if(NOT _dependency_error STREQUAL "NOTFOUND")
    message(FATAL_ERROR
      "invalid mthreads TritonJIT dependency identity: "
      "${_dependency_error}")
  endif()

  set(_runtime_library_directories)
  if(_jit_dependency_count GREATER 0)
    math(EXPR _jit_dependency_last "${_jit_dependency_count} - 1")
    foreach(_dependency_index RANGE 0 ${_jit_dependency_last})
      _flagdnn_mthreads_json_get(
        _dependency_name "${_environment_json}"
        resources triton_jit dependencies ${_dependency_index} name)
      _flagdnn_mthreads_json_get(
        _dependency_library "${_environment_json}"
        resources triton_jit dependencies ${_dependency_index} resolved)
      if(NOT IS_ABSOLUTE "${_dependency_library}")
        message(FATAL_ERROR
          "validated TritonJIT dependency path is not absolute: "
          "${_dependency_name}")
      endif()
      if(NOT EXISTS "${_dependency_library}" OR
         IS_DIRECTORY "${_dependency_library}")
        message(FATAL_ERROR
          "validated TritonJIT dependency disappeared: "
          "${_dependency_name}=${_dependency_library}")
      endif()
      file(REAL_PATH "${_dependency_library}"
        _dependency_realpath EXPAND_TILDE)
      if(NOT _dependency_realpath STREQUAL _dependency_library)
        message(FATAL_ERROR
          "validated TritonJIT dependency path is not canonical: "
          "${_dependency_name}=${_dependency_library}")
      endif()
      get_filename_component(
        _dependency_directory "${_dependency_realpath}" DIRECTORY)
      list(APPEND
        _runtime_library_directories "${_dependency_directory}")
    endforeach()
  endif()
  list(REMOVE_DUPLICATES _runtime_library_directories)

  if(NOT _musa_root STREQUAL _reported_root)
    message(FATAL_ERROR
      "canonical MUSA root differs from environment identity: "
      "${_musa_root} != ${_reported_root}")
  endif()
  foreach(_required_file IN ITEMS
      "${_driver_library}"
      "${_runtime_library}"
      "${_mudnn_library}"
      "${_driver_header}"
      "${_runtime_header}"
      "${_mudnn_header}")
    if(NOT EXISTS "${_required_file}")
      message(FATAL_ERROR
        "validated mthreads dependency disappeared: ${_required_file}")
    endif()
  endforeach()

  get_filename_component(_driver_include_dir "${_driver_header}" DIRECTORY)
  get_filename_component(_runtime_include_dir "${_runtime_header}" DIRECTORY)
  get_filename_component(_mudnn_include_dir "${_mudnn_header}" DIRECTORY)
  if(NOT _driver_include_dir STREQUAL _runtime_include_dir OR
     NOT _driver_include_dir STREQUAL _mudnn_include_dir)
    message(FATAL_ERROR
      "MUSA and muDNN headers do not share one canonical include directory")
  endif()
  set(_musa_include_dir "${_driver_include_dir}")

  if(NOT TARGET FlagDNNMthreads::musa_driver)
    add_library(FlagDNNMthreads::musa_driver SHARED IMPORTED GLOBAL)
    set_target_properties(
      FlagDNNMthreads::musa_driver
      PROPERTIES
        IMPORTED_LOCATION "${_driver_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${_musa_include_dir}"
        FLAGDNN_MTHREADS_REALPATH "${_driver_library}")
  endif()
  if(NOT TARGET FlagDNNMthreads::musa_runtime)
    add_library(FlagDNNMthreads::musa_runtime SHARED IMPORTED GLOBAL)
    set_target_properties(
      FlagDNNMthreads::musa_runtime
      PROPERTIES
        IMPORTED_LOCATION "${_runtime_library}"
        INTERFACE_INCLUDE_DIRECTORIES "${_musa_include_dir}"
        FLAGDNN_MTHREADS_REALPATH "${_runtime_library}")
  endif()
  if(NOT TARGET MUSA::musa_runtime)
    add_library(MUSA::musa_runtime INTERFACE IMPORTED GLOBAL)
    set_target_properties(
      MUSA::musa_runtime
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${_musa_include_dir}"
        INTERFACE_LINK_LIBRARIES "FlagDNNMthreads::musa_runtime")
  endif()

  foreach(_field IN ITEMS
      MUSA_INCLUDE_DIR
      MUSA_DRIVER_LIBRARY
      MUSA_RUNTIME_LIBRARY
      MUDNN_LIBRARY
      MUDNN_INCLUDE_DIR
      MUSA_ROOT_REALPATH
      ENVIRONMENT_IDENTITY_SHA256
      ENVIRONMENT_REPORT
      CODEGEN_PYTHON
      RUNTIME_LIBRARY_DIRECTORIES)
    unset(_output_value)
    if(_field STREQUAL "MUSA_INCLUDE_DIR")
      set(_output_value "${_musa_include_dir}")
    elseif(_field STREQUAL "MUSA_DRIVER_LIBRARY")
      set(_output_value "${_driver_library}")
    elseif(_field STREQUAL "MUSA_RUNTIME_LIBRARY")
      set(_output_value "${_runtime_library}")
    elseif(_field STREQUAL "MUDNN_LIBRARY")
      set(_output_value "${_mudnn_library}")
    elseif(_field STREQUAL "MUDNN_INCLUDE_DIR")
      set(_output_value "${_mudnn_include_dir}")
    elseif(_field STREQUAL "MUSA_ROOT_REALPATH")
      set(_output_value "${_musa_root}")
    elseif(_field STREQUAL "ENVIRONMENT_IDENTITY_SHA256")
      set(_output_value "${_identity}")
    elseif(_field STREQUAL "ENVIRONMENT_REPORT")
      set(_output_value "${_environment_report}")
    elseif(_field STREQUAL "CODEGEN_PYTHON")
      set(_output_value "${_codegen_python}")
    elseif(_field STREQUAL "RUNTIME_LIBRARY_DIRECTORIES")
      set(_output_value "${_runtime_library_directories}")
    endif()
    set("${out_prefix}_${_field}" "${_output_value}" PARENT_SCOPE)
  endforeach()
endfunction()
