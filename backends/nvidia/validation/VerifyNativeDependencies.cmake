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

# Public symbol versions and plugin exports have one authoritative checker.
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/VerifyCoreLibrary.cmake")
set(BACKEND_NAME nvidia)
set(BACKEND_ABI_VERSION 2)
set(REQUIRED_DEPENDENCIES libcuda libtriton_jit)
set(FORBIDDEN_DEPENDENCIES libcudnn)
include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/VerifyBackendPlugin.cmake")
