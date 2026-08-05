if(NOT DEFINED BINARY OR NOT EXISTS "${BINARY}")
  message(FATAL_ERROR "BINARY is missing")
endif()

find_program(READELF_EXECUTABLE readelf REQUIRED)
execute_process(
  COMMAND "${READELF_EXECUTABLE}" -d "${BINARY}"
  RESULT_VARIABLE result
  OUTPUT_VARIABLE dynamic_section
  ERROR_VARIABLE error_output)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "readelf failed for ${BINARY}: ${error_output}")
endif()

foreach(forbidden IN ITEMS
        libcuda libcudnn libcublas libascendcl libacl libpython libtorch)
  if(dynamic_section MATCHES "Shared library:.*${forbidden}")
    message(FATAL_ERROR
      "platform-neutral binary unexpectedly depends on ${forbidden}:\n${dynamic_section}")
  endif()
endforeach()

message(STATUS "Platform-neutral dependency boundary verified")
