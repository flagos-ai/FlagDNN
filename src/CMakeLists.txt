add_library(flagdnn SHARED
  api.cpp
  backend_loader.cpp
  error.cpp
  graph/graph.cpp
  graph/ir.cpp
  graph/lowering/convolution.cpp
  graph/lowering/attention.cpp
  graph/lowering/dispatch.cpp
  graph/lowering/layout.cpp
  graph/lowering/matmul.cpp
  graph/lowering/normalization.cpp
  graph/lowering/pointwise.cpp
  graph/lowering/reduction.cpp
  graph/tensor.cpp
  graph/validation.cpp
  runtime/artifact.cpp
  runtime/cache.cpp
  runtime/compiler_client.cpp
  runtime/context.cpp
  runtime/executable.cpp
  runtime/sha256.cpp)
add_library(FlagDNN::flagdnn ALIAS flagdnn)

target_compile_features(flagdnn PUBLIC cxx_std_20)
target_compile_definitions(flagdnn PRIVATE
  FLAGDNN_BUILD_SHARED
  FLAGDNN_DEFAULT_COMPILER_EXECUTABLE="${FLAGDNN_CODEGEN_PYTHON}"
  FLAGDNN_DEFAULT_CODEGEN_COMPILER="${FLAGDNN_CODEGEN_COMPILER}"
  FLAGDNN_INSTALLED_RESOURCE_RELATIVE="${FLAGDNN_RESOURCE_PATH_FROM_LIBRARY}"
  FLAGDNN_DEFAULT_BACKEND="${FLAGDNN_DEFAULT_BACKEND}"
  FLAGDNN_DEFAULT_EXECUTION_ENGINE="${FLAGDNN_EXECUTION_ENGINE}")
target_include_directories(flagdnn
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(flagdnn PRIVATE ${CMAKE_DL_LIBS})

if(UNIX AND NOT APPLE)
  target_link_options(flagdnn PRIVATE
    "LINKER:--version-script=${PROJECT_SOURCE_DIR}/cmake/flagdnn.map")
  set_property(TARGET flagdnn APPEND PROPERTY LINK_DEPENDS
    "${PROJECT_SOURCE_DIR}/cmake/flagdnn.map")
endif()

flagdnn_enable_warnings(flagdnn)
set_target_properties(flagdnn PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR})

install(TARGETS flagdnn
  EXPORT FlagDNNTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES
  "${PROJECT_BINARY_DIR}/generated/include/flagdnn/version.h"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/flagdnn")
