if(NOT DEFINED BUILD_PLUGIN OR NOT EXISTS "${BUILD_PLUGIN}")
  message(FATAL_ERROR "BUILD_PLUGIN must name the Iluvatar plugin")
endif()
foreach(_required IN ITEMS COREX_ROOT COREX_DRIVER JIT_LIBRARY JIT_CONFIG_DIR)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "" OR
     NOT EXISTS "${${_required}}")
    message(FATAL_ERROR "${_required} must name an existing path")
  endif()
endforeach()

find_program(_readelf NAMES readelf llvm-readelf REQUIRED)
find_program(_ldd NAMES ldd REQUIRED)
find_program(_nm NAMES nm llvm-nm REQUIRED)
file(REAL_PATH "${COREX_ROOT}" _corex_root)
file(REAL_PATH "${COREX_DRIVER}" _corex_driver)
file(REAL_PATH "${JIT_LIBRARY}" _jit_library)

file(READ "${JIT_CONFIG_DIR}/TritonJITConfig.cmake" _jit_config)
if(NOT _jit_config MATCHES
     "set[ \t\r\n]*\\([ \t\r\n]*TritonJIT_BACKEND[ \t\r\n]+\"?IX")
  message(FATAL_ERROR "selected TritonJIT package does not report IX")
endif()

function(_inspect_plugin path label)
  file(REAL_PATH "${path}" _plugin)
  execute_process(
    COMMAND "${_readelf}" -d "${_plugin}"
    RESULT_VARIABLE _readelf_result
    OUTPUT_VARIABLE _dynamic
    ERROR_VARIABLE _readelf_error)
  if(NOT _readelf_result EQUAL 0)
    message(FATAL_ERROR
      "${label}: cannot read plugin dynamic section: ${_readelf_error}")
  endif()

  string(REGEX MATCHALL
    "\\(NEEDED\\)[^\n]*\\[[^]]+\\]" _needed_records "${_dynamic}")
  set(_needed)
  foreach(_record IN LISTS _needed_records)
    string(REGEX REPLACE ".*\\[([^]]+)\\].*" "\\1" _name "${_record}")
    list(APPEND _needed "${_name}")
  endforeach()
  list(SORT _needed)
  foreach(_required_pattern IN ITEMS
      "^libcuda\\.so"
      "^libtriton_jit\\.so"
      "^libpython[0-9.]*\\.so")
    set(_found FALSE)
    foreach(_name IN LISTS _needed)
      if(_name MATCHES "${_required_pattern}")
        set(_found TRUE)
      endif()
    endforeach()
    if(NOT _found)
      message(FATAL_ERROR
        "${label}: missing required direct dependency ${_required_pattern}")
    endif()
  endforeach()
  foreach(_name IN LISTS _needed)
    string(TOLOWER "${_name}" _lower_name)
    if(_lower_name MATCHES
       "^lib(cudnn|cublas|hipdnn|miopen|rocblas|amdhip64|galaxyhip)\\.so")
      message(FATAL_ERROR
        "${label}: forbidden direct DNN/BLAS dependency: ${_name}")
    endif()
  endforeach()

  execute_process(
    COMMAND "${_ldd}" "${_plugin}"
    RESULT_VARIABLE _ldd_result
    OUTPUT_VARIABLE _resolved
    ERROR_VARIABLE _ldd_error)
  if(NOT _ldd_result EQUAL 0 OR _resolved MATCHES "not found")
    message(FATAL_ERROR
      "${label}: unresolved production dependency:\n${_resolved}${_ldd_error}")
  endif()
  string(REGEX MATCH
    "libcuda\\.so[^ \t\r\n]*[ \t]+=>[ \t]+([^ \t\r\n]+)"
    _cuda_record "${_resolved}")
  set(_resolved_cuda "${CMAKE_MATCH_1}")
  string(REGEX MATCH
    "libtriton_jit\\.so[^ \t\r\n]*[ \t]+=>[ \t]+([^ \t\r\n]+)"
    _jit_record "${_resolved}")
  set(_resolved_jit "${CMAKE_MATCH_1}")
  if(_resolved_cuda STREQUAL "" OR _resolved_jit STREQUAL "")
    message(FATAL_ERROR
      "${label}: ldd did not resolve CoreX Driver and IX JIT")
  endif()
  file(REAL_PATH "${_resolved_cuda}" _resolved_cuda)
  file(REAL_PATH "${_resolved_jit}" _resolved_jit)
  if(NOT _resolved_cuda STREQUAL _corex_driver)
    message(FATAL_ERROR
      "${label}: libcuda resolved outside selected CoreX: ${_resolved_cuda}")
  endif()
  file(RELATIVE_PATH _cuda_relative "${_corex_root}" "${_resolved_cuda}")
  if(_cuda_relative STREQUAL ".." OR _cuda_relative MATCHES "^\\.\\./")
    message(FATAL_ERROR
      "${label}: selected libcuda is outside CoreX root")
  endif()
  if(NOT _resolved_jit STREQUAL _jit_library)
    message(FATAL_ERROR
      "${label}: libtriton_jit resolved to ${_resolved_jit}, "
      "expected ${_jit_library}")
  endif()

  execute_process(
    COMMAND "${_nm}" -D --defined-only "${_plugin}"
    RESULT_VARIABLE _nm_result
    OUTPUT_VARIABLE _symbols
    ERROR_VARIABLE _nm_error)
  if(NOT _nm_result EQUAL 0)
    message(FATAL_ERROR "${label}: cannot inspect exports: ${_nm_error}")
  endif()
  string(REGEX MATCHALL
    "[^\n]*flagdnn[^\n]*" _flagdnn_symbols "${_symbols}")
  list(LENGTH _flagdnn_symbols _flagdnn_symbol_count)
  if(NOT _flagdnn_symbol_count EQUAL 1 OR
     NOT _flagdnn_symbols MATCHES
         "flagdnnBackendGetApiV2@@FLAGDNN_BACKEND_2")
    message(FATAL_ERROR
      "${label}: plugin exported an unexpected FlagDNN ABI surface:\n"
      "${_symbols}")
  endif()

  file(SHA256 "${_plugin}" _sha256)
  set(${label}_PATH "${_plugin}" PARENT_SCOPE)
  set(${label}_SHA256 "${_sha256}" PARENT_SCOPE)
  set(${label}_NEEDED "${_needed}" PARENT_SCOPE)
  set(${label}_CUDA "${_resolved_cuda}" PARENT_SCOPE)
  set(${label}_JIT "${_resolved_jit}" PARENT_SCOPE)
endfunction()

function(_json_array input_variable output_variable)
  set(_result "[")
  set(_separator "")
  foreach(_value IN LISTS ${input_variable})
    string(APPEND _result "${_separator}\"${_value}\"")
    set(_separator ",")
  endforeach()
  string(APPEND _result "]")
  set(${output_variable} "${_result}" PARENT_SCOPE)
endfunction()

_inspect_plugin("${BUILD_PLUGIN}" BUILD)
set(_installed_json "null")
if(DEFINED INSTALLED_PLUGIN AND NOT INSTALLED_PLUGIN STREQUAL "")
  if(NOT EXISTS "${INSTALLED_PLUGIN}")
    message(FATAL_ERROR "INSTALLED_PLUGIN does not exist")
  endif()
  _inspect_plugin("${INSTALLED_PLUGIN}" INSTALLED)
  _json_array(INSTALLED_NEEDED _installed_needed_json)
  set(_installed_json
    "{\"path\":\"${INSTALLED_PATH}\",\"sha256\":\"${INSTALLED_SHA256}\",\"direct_needed\":${_installed_needed_json},\"resolved_cuda\":\"${INSTALLED_CUDA}\",\"resolved_triton_jit\":\"${INSTALLED_JIT}\"}")
endif()

if(DEFINED OUTPUT_JSON AND NOT OUTPUT_JSON STREQUAL "")
  get_filename_component(_output_directory "${OUTPUT_JSON}" DIRECTORY)
  file(MAKE_DIRECTORY "${_output_directory}")
  _json_array(BUILD_NEEDED _build_needed_json)
  file(WRITE "${OUTPUT_JSON}"
    "{\"schema_version\":1,\"phase\":1,\"backend\":\"iluvatar\","
    "\"target\":\"corex_71\",\"execution_engine\":\"libtriton_jit\","
    "\"triton_jit_backend\":\"IX\","
    "\"corex_root\":\"${_corex_root}\","
    "\"corex_driver\":\"${_corex_driver}\","
    "\"triton_jit\":\"${_jit_library}\","
    "\"build_plugin\":{\"path\":\"${BUILD_PATH}\","
    "\"sha256\":\"${BUILD_SHA256}\","
    "\"direct_needed\":${_build_needed_json},"
    "\"resolved_cuda\":\"${BUILD_CUDA}\","
    "\"resolved_triton_jit\":\"${BUILD_JIT}\"},"
    "\"installed_plugin\":${_installed_json}}\n")
endif()

message(STATUS
  "PASS Iluvatar dependency boundary; direct=${BUILD_NEEDED}; "
  "cuda=${BUILD_CUDA}; jit=${BUILD_JIT}; backend=IX")
