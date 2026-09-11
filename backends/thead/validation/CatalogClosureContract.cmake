# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

foreach(_required IN ITEMS SOURCE_ROOT BUILD_ROOT CTEST_COMMAND)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "Catalog closure contract requires ${_required}")
  endif()
endforeach()

include("${SOURCE_ROOT}/cmake/Operators.cmake")

execute_process(
  COMMAND "${CTEST_COMMAND}" --test-dir "${BUILD_ROOT}"
    --show-only=json-v1
  RESULT_VARIABLE _catalog_result
  OUTPUT_VARIABLE _catalog_json
  ERROR_VARIABLE _catalog_error)
if(NOT _catalog_result EQUAL 0)
  message(FATAL_ERROR
    "Cannot list the configured THead CTest catalog:\n${_catalog_error}")
endif()

string(JSON _test_count ERROR_VARIABLE _json_error
  LENGTH "${_catalog_json}" tests)
if(_json_error)
  message(FATAL_ERROR "CTest catalog is not valid JSON: ${_json_error}")
endif()

set(_functional_actual)
set(_benchmark_actual)
if(_test_count GREATER 0)
  math(EXPR _last_test "${_test_count} - 1")
  foreach(_index RANGE 0 ${_last_test})
    string(JSON _name GET "${_catalog_json}" tests ${_index} name)
    if(_name MATCHES "^functional\\.thead\\.([a-z][a-z0-9_]*)$")
      list(APPEND _functional_actual "${CMAKE_MATCH_1}")
    elseif(_name MATCHES "^benchmark\\.thead\\.([a-z][a-z0-9_]*)$")
      list(APPEND _benchmark_actual "${CMAKE_MATCH_1}")
    endif()
  endforeach()
endif()

function(_flagdnn_thead_require_exact_catalog _suite _expected _actual)
  set(_expected_values ${${_expected}})
  set(_actual_values ${${_actual}})
  list(SORT _expected_values)
  list(SORT _actual_values)
  set(_unique_values ${_actual_values})
  list(REMOVE_DUPLICATES _unique_values)

  set(_missing ${_expected_values})
  list(REMOVE_ITEM _missing ${_actual_values})
  set(_extra ${_actual_values})
  list(REMOVE_ITEM _extra ${_expected_values})
  if(NOT _actual_values STREQUAL _unique_values OR _missing OR _extra)
    list(LENGTH _expected_values _expected_count)
    list(LENGTH _actual_values _actual_count)
    message(FATAL_ERROR
      "THead ${_suite} CTest catalog is not closed: "
      "actual=${_actual_count} expected=${_expected_count}; "
      "missing=${_missing}; extra=${_extra}")
  endif()
endfunction()

_flagdnn_thead_require_exact_catalog(
  "functional" FLAGDNN_FUNCTIONAL_OPERATORS _functional_actual)
_flagdnn_thead_require_exact_catalog(
  "benchmark" FLAGDNN_BENCHMARK_OPERATORS _benchmark_actual)

list(LENGTH _functional_actual _functional_count)
list(LENGTH _benchmark_actual _benchmark_count)
message(STATUS
  "PASS THead CTest catalog closure: functional=${_functional_count} "
  "benchmark=${_benchmark_count}")
