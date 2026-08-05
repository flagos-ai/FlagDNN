if(NOT DEFINED REFERENCE_EXECUTABLE OR
   NOT EXISTS "${REFERENCE_EXECUTABLE}")
  message(FATAL_ERROR "REFERENCE_EXECUTABLE is missing")
endif()

find_program(READELF_EXECUTABLE readelf REQUIRED)
execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${REFERENCE_EXECUTABLE}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE error_output)
if(NOT result EQUAL 0)
  message(FATAL_ERROR
    "readelf failed for native reference executable: ${error_output}")
endif()

foreach(required IN ITEMS libflagdnn libcudnn)
  if(NOT dynamic_section MATCHES "Shared library:.*${required}")
    message(FATAL_ERROR
      "native reference executable is missing ${required}:\n${dynamic_section}")
  endif()
endforeach()

foreach(forbidden IN ITEMS libpython libtorch)
  if(dynamic_section MATCHES "Shared library:.*${forbidden}")
    message(FATAL_ERROR
      "native reference executable unexpectedly depends on ${forbidden}:\n${dynamic_section}")
  endif()
endforeach()

message(STATUS "Native cuDNN reference dependency boundary verified")
