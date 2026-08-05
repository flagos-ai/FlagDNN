include_guard(GLOBAL)

function(flagdnn_add_backend_plugin backend_name)
  set(options INSTALL)
  set(one_value_arguments)
  set(multi_value_arguments SOURCES INCLUDE_DIRECTORIES LINK_LIBRARIES)
  cmake_parse_arguments(
    FLAGDNN_BACKEND
    "${options}"
    "${one_value_arguments}"
    "${multi_value_arguments}"
    ${ARGN})

  string(LENGTH "${backend_name}" backend_name_length)
  if(NOT backend_name MATCHES "^[a-z][a-z0-9_]*$" OR
     backend_name_length GREATER 63)
    message(FATAL_ERROR
      "FlagDNN backend name '${backend_name}' is not loader-safe")
  endif()
  if(NOT FLAGDNN_BACKEND_SOURCES)
    message(FATAL_ERROR
      "flagdnn_add_backend_plugin(${backend_name}) requires SOURCES")
  endif()

  set(target "flagdnn_backend_${backend_name}")
  if(TARGET "${target}")
    message(FATAL_ERROR "FlagDNN backend target '${target}' already exists")
  endif()

  add_library("${target}" SHARED ${FLAGDNN_BACKEND_SOURCES})
  add_library("FlagDNN::backend_${backend_name}" ALIAS "${target}")
  target_compile_features("${target}" PRIVATE cxx_std_20)
  target_include_directories("${target}" PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/generated/include
    ${FLAGDNN_BACKEND_INCLUDE_DIRECTORIES})
  if(FLAGDNN_BACKEND_LINK_LIBRARIES)
    target_link_libraries("${target}" PRIVATE
      ${FLAGDNN_BACKEND_LINK_LIBRARIES})
  endif()
  flagdnn_enable_warnings("${target}")

  if(UNIX AND NOT APPLE)
    target_link_options("${target}" PRIVATE
      "LINKER:--version-script=${PROJECT_SOURCE_DIR}/cmake/flagdnn_backend.map")
    set_property(TARGET "${target}" APPEND PROPERTY LINK_DEPENDS
      "${PROJECT_SOURCE_DIR}/cmake/flagdnn_backend.map")
  endif()
  set_target_properties("${target}" PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION 2.0.0
    SOVERSION 2)

  if(FLAGDNN_BACKEND_INSTALL)
    install(TARGETS "${target}"
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endfunction()
