if(NOT DEFINED CORE_LIBRARY OR NOT EXISTS "${CORE_LIBRARY}")
  message(FATAL_ERROR "CORE_LIBRARY is missing")
endif()
if(NOT DEFINED BACKEND_LIBRARY OR NOT EXISTS "${BACKEND_LIBRARY}")
  message(FATAL_ERROR "BACKEND_LIBRARY is missing")
endif()

find_program(READELF_EXECUTABLE readelf REQUIRED)
find_program(NM_EXECUTABLE nm REQUIRED)

execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${CORE_LIBRARY}"
  RESULT_VARIABLE core_result
  OUTPUT_VARIABLE core_dynamic
  ERROR_VARIABLE core_error)
if(NOT core_result EQUAL 0)
  message(FATAL_ERROR "readelf failed for core library: ${core_error}")
endif()

foreach(forbidden IN ITEMS libcuda libpython libtorch libcudnn libcublas)
  if(core_dynamic MATCHES "Shared library:.*${forbidden}")
    message(FATAL_ERROR
      "libflagdnn core unexpectedly depends on ${forbidden}:\n${core_dynamic}")
  endif()
endforeach()

execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${BACKEND_LIBRARY}"
  RESULT_VARIABLE backend_result
  OUTPUT_VARIABLE backend_dynamic
  ERROR_VARIABLE backend_error)
if(NOT backend_result EQUAL 0)
  message(FATAL_ERROR "readelf failed for NVIDIA backend: ${backend_error}")
endif()

if(NOT backend_dynamic MATCHES "Shared library:.*libcuda")
  message(FATAL_ERROR
    "NVIDIA backend does not declare its libcuda dependency:\n${backend_dynamic}")
endif()
set(_flagdnn_backend_forbidden libcudnn)
if(NOT EXECUTION_ENGINE STREQUAL "libtriton_jit")
  list(APPEND _flagdnn_backend_forbidden libpython libtorch)
endif()
foreach(forbidden IN LISTS _flagdnn_backend_forbidden)
  if(backend_dynamic MATCHES "Shared library:.*${forbidden}")
    message(FATAL_ERROR
      "NVIDIA backend unexpectedly depends on ${forbidden}:\n${backend_dynamic}")
  endif()
endforeach()

function(read_defined_dynamic_symbols library output_variable)
  execute_process(
    COMMAND "${NM_EXECUTABLE}" -D --defined-only "${library}"
    RESULT_VARIABLE symbol_result
    OUTPUT_VARIABLE symbol_output
    ERROR_VARIABLE symbol_error)
  if(NOT symbol_result EQUAL 0)
    message(FATAL_ERROR "nm failed for ${library}: ${symbol_error}")
  endif()
  set("${output_variable}" "${symbol_output}" PARENT_SCOPE)
endfunction()

read_defined_dynamic_symbols("${CORE_LIBRARY}" core_symbols)
string(REPLACE "\n" ";" core_symbol_lines "${core_symbols}")
foreach(line IN LISTS core_symbol_lines)
  if(line STREQUAL "")
    continue()
  endif()
  string(REGEX MATCH "[^ \t]+$" symbol "${line}")
  if(NOT symbol MATCHES "^FLAGDNN_0\\.1$" AND
     NOT symbol MATCHES "^flagdnn[A-Za-z0-9_]*@@FLAGDNN_0\\.1$")
    message(FATAL_ERROR
      "libflagdnn exports a non-public symbol '${symbol}':\n${core_symbols}")
  endif()
endforeach()
if(NOT core_symbols MATCHES "flagdnnGetVersion@@FLAGDNN_0\\.1")
  message(FATAL_ERROR
    "libflagdnn does not export the expected versioned C API:\n${core_symbols}")
endif()

read_defined_dynamic_symbols("${BACKEND_LIBRARY}" backend_symbols)
string(REPLACE "\n" ";" backend_symbol_lines "${backend_symbols}")
foreach(line IN LISTS backend_symbol_lines)
  if(line STREQUAL "")
    continue()
  endif()
  string(REGEX MATCH "[^ \t]+$" symbol "${line}")
  if(NOT symbol MATCHES "^FLAGDNN_BACKEND_2$" AND
     NOT symbol MATCHES "^flagdnnBackendGetApiV2@@FLAGDNN_BACKEND_2$")
    message(FATAL_ERROR
      "NVIDIA backend exports an unexpected symbol '${symbol}':\n${backend_symbols}")
  endif()
endforeach()
if(NOT backend_symbols MATCHES
   "flagdnnBackendGetApiV2@@FLAGDNN_BACKEND_2")
  message(FATAL_ERROR
    "NVIDIA backend does not export its versioned ABI entry:\n${backend_symbols}")
endif()

message(STATUS
  "FlagDNN core/backend dependency and symbol boundaries verified")
