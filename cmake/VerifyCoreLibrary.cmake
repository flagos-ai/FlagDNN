if(NOT DEFINED CORE_LIBRARY OR NOT EXISTS "${CORE_LIBRARY}")
  message(FATAL_ERROR "CORE_LIBRARY is missing")
endif()

find_program(READELF_EXECUTABLE readelf REQUIRED)
find_program(NM_EXECUTABLE nm REQUIRED)

execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${CORE_LIBRARY}"
  RESULT_VARIABLE dynamic_result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE dynamic_error)
if(NOT dynamic_result EQUAL 0)
  message(FATAL_ERROR "readelf failed for core library: ${dynamic_error}")
endif()
if(NOT dynamic_section MATCHES "Library soname:.*libflagdnn\\.so\\.1\\]")
  message(FATAL_ERROR
    "libflagdnn does not use the execution-contract-v2 SONAME:\n${dynamic_section}")
endif()

execute_process(
  COMMAND "${NM_EXECUTABLE}" -D --defined-only "${CORE_LIBRARY}"
  RESULT_VARIABLE symbol_result
  OUTPUT_VARIABLE symbols
  ERROR_VARIABLE symbol_error)
if(NOT symbol_result EQUAL 0)
  message(FATAL_ERROR "nm failed for core library: ${symbol_error}")
endif()

string(REPLACE "\n" ";" symbol_lines "${symbols}")
foreach(line IN LISTS symbol_lines)
  if(line STREQUAL "")
    continue()
  endif()
  string(REGEX MATCH "[^ \t]+$" symbol "${line}")
  if(symbol MATCHES "^FLAGDNN_0\\.[12]$" OR
     symbol STREQUAL
       "flagdnnGetExecutionContractVersion@@FLAGDNN_0.2" OR
     symbol STREQUAL "flagdnnSetPointwiseBinaryOperationDescriptorWithAttributes@@FLAGDNN_0.2" OR
     (symbol MATCHES "^flagdnn[A-Za-z0-9_]*@@FLAGDNN_0\\.1$" AND
      NOT symbol MATCHES "^flagdnnGetExecutionContractVersion"))
    continue()
  endif()
  message(FATAL_ERROR
    "libflagdnn exports an unexpected symbol '${symbol}':\n${symbols}")
endforeach()

if(NOT symbols MATCHES "flagdnnGetVersion@@FLAGDNN_0\\.1")
  message(FATAL_ERROR
    "libflagdnn does not retain its existing FLAGDNN_0.1 API:\n${symbols}")
endif()
if(NOT symbols MATCHES
   "flagdnnGetExecutionContractVersion@@FLAGDNN_0\\.2")
  message(FATAL_ERROR
    "libflagdnn does not export the contract query at FLAGDNN_0.2:\n${symbols}")
endif()

message(STATUS "FlagDNN core SONAME and symbol-version contract verified")
