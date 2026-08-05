if(NOT DEFINED BACKEND_LIBRARY OR NOT EXISTS "${BACKEND_LIBRARY}")
  message(FATAL_ERROR "BACKEND_LIBRARY is missing")
endif()
if(NOT DEFINED BACKEND_NAME OR BACKEND_NAME STREQUAL "")
  message(FATAL_ERROR "BACKEND_NAME is missing")
endif()

find_program(READELF_EXECUTABLE readelf REQUIRED)
find_program(NM_EXECUTABLE nm REQUIRED)

execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${BACKEND_LIBRARY}"
  RESULT_VARIABLE dynamic_result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE dynamic_error)
if(NOT dynamic_result EQUAL 0)
  message(FATAL_ERROR
    "readelf failed for ${BACKEND_NAME} backend: ${dynamic_error}")
endif()

foreach(required IN LISTS REQUIRED_DEPENDENCIES)
  if(NOT dynamic_section MATCHES "Shared library:.*${required}")
    message(FATAL_ERROR
      "${BACKEND_NAME} backend is missing ${required}:\n${dynamic_section}")
  endif()
endforeach()
foreach(forbidden IN LISTS FORBIDDEN_DEPENDENCIES)
  if(dynamic_section MATCHES "Shared library:.*${forbidden}")
    message(FATAL_ERROR
      "${BACKEND_NAME} backend unexpectedly depends on ${forbidden}:\n${dynamic_section}")
  endif()
endforeach()

execute_process(
  COMMAND "${NM_EXECUTABLE}" -D --defined-only "${BACKEND_LIBRARY}"
  RESULT_VARIABLE symbol_result
  OUTPUT_VARIABLE symbols
  ERROR_VARIABLE symbol_error)
if(NOT symbol_result EQUAL 0)
  message(FATAL_ERROR
    "nm failed for ${BACKEND_NAME} backend: ${symbol_error}")
endif()

string(REPLACE "\n" ";" symbol_lines "${symbols}")
foreach(line IN LISTS symbol_lines)
  if(line STREQUAL "")
    continue()
  endif()
  string(REGEX MATCH "[^ \t]+$" symbol "${line}")
  if(NOT symbol MATCHES "^FLAGDNN_BACKEND_2$" AND
     NOT symbol MATCHES
       "^flagdnnBackendGetApiV2@@FLAGDNN_BACKEND_2$")
    message(FATAL_ERROR
      "${BACKEND_NAME} backend exports unexpected symbol '${symbol}':\n${symbols}")
  endif()
endforeach()
if(NOT symbols MATCHES
   "flagdnnBackendGetApiV2@@FLAGDNN_BACKEND_2")
  message(FATAL_ERROR
    "${BACKEND_NAME} backend does not export its versioned ABI entry:\n${symbols}")
endif()

message(STATUS
  "${BACKEND_NAME} backend dependency and symbol contract verified")
