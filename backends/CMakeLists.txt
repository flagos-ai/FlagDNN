# Backend plugins are selected in one place. Each platform subdirectory owns
# its SDK discovery, runtime target, compiler provider, kernels and install
# rules; the repository root remains platform-neutral.

set(_flagdnn_backends ${FLAGDNN_BACKENDS})
list(REMOVE_DUPLICATES _flagdnn_backends)

foreach(platform IN LISTS _flagdnn_backends)
  if(NOT platform MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "Invalid FLAGDNN_BACKENDS entry: ${platform}")
  endif()
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${platform}/CMakeLists.txt")
    message(FATAL_ERROR
      "Backend ${platform} has no implementation at backends/${platform}")
  endif()
  add_subdirectory("${platform}")
endforeach()
