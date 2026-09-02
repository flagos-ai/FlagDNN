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
    "readelf failed for Hygon reference executable: ${error_output}")
endif()

foreach(required IN ITEMS libflagdnn libgalaxyhip libhipdnn)
  if(NOT dynamic_section MATCHES "Shared library:.*${required}")
    message(FATAL_ERROR
      "Hygon reference executable is missing ${required}:\n${dynamic_section}")
  endif()
endforeach()

# hipDNN is the only vendor numerical reference library allowed in the Hygon
# suite. The functional-only CPU semantic oracle is linked statically and adds
# no vendor dependency. hipDNN's transitive DTK dependencies remain
# vendor-owned and are not linked by the validation target itself.
foreach(forbidden IN ITEMS
    libMIOpen libmiopen librocblas libhipblas
    libcuda libcudnn libcublas libpython libtorch)
  if(dynamic_section MATCHES "Shared library:.*${forbidden}")
    message(FATAL_ERROR
      "Hygon reference executable unexpectedly depends on ${forbidden}:\n"
      "${dynamic_section}")
  endif()
endforeach()

message(STATUS "Native Hygon reference dependency boundary verified")
