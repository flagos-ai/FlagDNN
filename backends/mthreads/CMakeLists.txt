set(FLAGDNN_MTHREADS_MUSA_ROOT ""
    CACHE PATH "Canonical MUSA Toolkit root for the mthreads backend")
set(FLAGDNN_MTHREADS_TRITON_JIT_DIR ""
    CACHE PATH "Directory containing the MUSA TritonJITConfig.cmake")

if(NOT FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
  message(FATAL_ERROR
    "mthreads requires FLAGDNN_EXECUTION_ENGINE=libtriton_jit")
endif()
if(NOT FLAGDNN_CODEGEN_PYTHON)
  message(FATAL_ERROR "FLAGDNN_CODEGEN_PYTHON is required for mthreads")
endif()
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(MthreadsDependencies)
include(MthreadsTritonJIT)

# Keep automatic discovery backend-local and preserve explicit cache overrides.
if(NOT FLAGDNN_MTHREADS_MUSA_ROOT)
  if(DEFINED ENV{FLAGDNN_MTHREADS_MUSA_ROOT} AND
     NOT "$ENV{FLAGDNN_MTHREADS_MUSA_ROOT}" STREQUAL "")
    set(_flagdnn_musa_root "$ENV{FLAGDNN_MTHREADS_MUSA_ROOT}")
  elseif(MUSA_HOME)
    set(_flagdnn_musa_root "${MUSA_HOME}")
  elseif(DEFINED ENV{MUSA_HOME} AND NOT "$ENV{MUSA_HOME}" STREQUAL "")
    set(_flagdnn_musa_root "$ENV{MUSA_HOME}")
  else()
    set(_flagdnn_musa_root "/usr/local/musa")
  endif()
  set(FLAGDNN_MTHREADS_MUSA_ROOT "${_flagdnn_musa_root}"
      CACHE PATH "Canonical MUSA Toolkit root for the mthreads backend" FORCE)
endif()

if(NOT FLAGDNN_MTHREADS_TRITON_JIT_DIR)
  set(_flagdnn_mthreads_jit_candidates)
  if(DEFINED ENV{FLAGDNN_MTHREADS_TRITON_JIT_DIR} AND
     NOT "$ENV{FLAGDNN_MTHREADS_TRITON_JIT_DIR}" STREQUAL "")
    list(APPEND _flagdnn_mthreads_jit_candidates
      "$ENV{FLAGDNN_MTHREADS_TRITON_JIT_DIR}")
  endif()
  if(TritonJIT_DIR)
    list(APPEND _flagdnn_mthreads_jit_candidates "${TritonJIT_DIR}")
  endif()
  if(DEFINED ENV{TritonJIT_DIR} AND NOT "$ENV{TritonJIT_DIR}" STREQUAL "")
    list(APPEND _flagdnn_mthreads_jit_candidates "$ENV{TritonJIT_DIR}")
  endif()
  if(DEFINED ENV{LIBTRITON_JIT_ROOT} AND
     NOT "$ENV{LIBTRITON_JIT_ROOT}" STREQUAL "")
    list(APPEND _flagdnn_mthreads_jit_candidates
      "$ENV{LIBTRITON_JIT_ROOT}/build"
      "$ENV{LIBTRITON_JIT_ROOT}/lib/cmake/TritonJIT")
  endif()
  list(APPEND _flagdnn_mthreads_jit_candidates
    "${PROJECT_SOURCE_DIR}/../libtriton_jit/build"
    "${PROJECT_SOURCE_DIR}/../libtriton_jit/lib/cmake/TritonJIT"
    "/usr/local/lib/cmake/TritonJIT")
  foreach(_candidate IN LISTS _flagdnn_mthreads_jit_candidates)
    if(EXISTS "${_candidate}/TritonJITConfig.cmake")
      get_filename_component(_candidate "${_candidate}" ABSOLUTE)
      set(FLAGDNN_MTHREADS_TRITON_JIT_DIR "${_candidate}"
          CACHE PATH "Directory containing the MUSA TritonJITConfig.cmake" FORCE)
      break()
    endif()
  endforeach()
  if(NOT FLAGDNN_MTHREADS_TRITON_JIT_DIR)
    message(FATAL_ERROR
      "Cannot find a MUSA TritonJIT package. Build ../libtriton_jit/build "
      "with BACKEND=MUSA, or set FLAGDNN_MTHREADS_TRITON_JIT_DIR to the "
      "directory containing TritonJITConfig.cmake.")
  endif()
endif()
message(STATUS "mthreads TritonJIT config: ${FLAGDNN_MTHREADS_TRITON_JIT_DIR}")

flagdnn_mthreads_resolve_dependencies(
  FLAGDNN_MTHREADS_DEPENDENCIES
  MUSA_ROOT "${FLAGDNN_MTHREADS_MUSA_ROOT}"
  CODEGEN_PYTHON "${FLAGDNN_CODEGEN_PYTHON}"
  TRITON_JIT_DIR "${FLAGDNN_MTHREADS_TRITON_JIT_DIR}"
  ENVIRONMENT_REPORT
    "${CMAKE_CURRENT_BINARY_DIR}/flagdnn_mthreads_environment.json")

# Keep the interpreter override local to this backend.  Writing either
# Python cache entry with FORCE would leak MThreads configuration into other
# backend subdirectories in a multi-backend build.
set(Python_EXECUTABLE "${FLAGDNN_CODEGEN_PYTHON}")
find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(_flagdnn_mthreads_private_root
    "${CMAKE_CURRENT_BINARY_DIR}")
flagdnn_mthreads_resolve_triton_jit(
  FLAGDNN_MTHREADS_JIT
  CONFIG_DIR "${FLAGDNN_MTHREADS_TRITON_JIT_DIR}"
  EXPECTED_BACKEND MUSA
  ENVIRONMENT_REPORT
    "${FLAGDNN_MTHREADS_DEPENDENCIES_ENVIRONMENT_REPORT}"
  CODEGEN_PYTHON
    "${FLAGDNN_MTHREADS_DEPENDENCIES_CODEGEN_PYTHON}"
  MUSA_ROOT "${FLAGDNN_MTHREADS_DEPENDENCIES_MUSA_ROOT_REALPATH}"
  PRIVATE_ROOT "${_flagdnn_mthreads_private_root}")

file(READ
  "${FLAGDNN_MTHREADS_DEPENDENCIES_ENVIRONMENT_REPORT}"
  _flagdnn_mthreads_environment_json)
string(JSON _flagdnn_mthreads_torch_init GET
  "${_flagdnn_mthreads_environment_json}"
  python modules torch realpath)
get_filename_component(
  _flagdnn_mthreads_torch_package
  "${_flagdnn_mthreads_torch_init}"
  DIRECTORY)
get_filename_component(
  _flagdnn_mthreads_python_site_packages
  "${_flagdnn_mthreads_torch_package}"
  DIRECTORY)
if(NOT IS_DIRECTORY "${_flagdnn_mthreads_python_site_packages}")
  message(FATAL_ERROR
    "validated mthreads Python site-packages directory disappeared")
endif()

flagdnn_add_backend_plugin(mthreads INSTALL ABI_VERSION 2
  SOURCES
    artifact.cpp
    ${PROJECT_SOURCE_DIR}/backends/autotune_policy.cpp
    autotune.cpp
    backend.cpp
    context.cpp
    engines/engine.cpp
    engines/libtriton_jit.cpp
    error.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/json.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp
  LINK_LIBRARIES
    FlagDNNMthreads::musa_driver
    FlagDNNMthreads::musa_runtime
    FlagDNNMthreads::triton_jit_private
    Python::Python
    ${CMAKE_DL_LIBS})

file(SHA256
  "${FLAGDNN_MTHREADS_JIT_SCRIPT_DIR}/gen_ssig.py"
  _flagdnn_mthreads_gen_ssig_sha256)
file(SHA256
  "${FLAGDNN_MTHREADS_JIT_SCRIPT_DIR}/standalone_compile.py"
  _flagdnn_mthreads_standalone_compile_sha256)
string(JSON _flagdnn_mthreads_gen_ssig_provenance_sha256 GET
  "${_flagdnn_mthreads_environment_json}"
  resources triton_jit_gen_ssig sha256)
string(JSON _flagdnn_mthreads_standalone_provenance_sha256 GET
  "${_flagdnn_mthreads_environment_json}"
  resources triton_jit_standalone_compile sha256)

target_compile_definitions(
  flagdnn_backend_mthreads
  PRIVATE
    BACKEND_MUSA=1
    "FLAGDNN_MTHREADS_ENVIRONMENT_IDENTITY=\"${FLAGDNN_MTHREADS_DEPENDENCIES_ENVIRONMENT_IDENTITY_SHA256}\""
    "FLAGDNN_MTHREADS_JIT_ORIGINAL_SHA256=\"${FLAGDNN_MTHREADS_JIT_PROVENANCE_SHA256}\""
    "FLAGDNN_MTHREADS_JIT_PRIVATE_SHA256=\"${FLAGDNN_MTHREADS_JIT_PRIVATE_SHA256}\""
    "FLAGDNN_MTHREADS_JIT_GEN_SSIG_SHA256=\"${_flagdnn_mthreads_gen_ssig_sha256}\""
    "FLAGDNN_MTHREADS_JIT_STANDALONE_SHA256=\"${_flagdnn_mthreads_standalone_compile_sha256}\""
    "FLAGDNN_MTHREADS_PYTHONPATH=\"${_flagdnn_mthreads_python_site_packages}\"")
set_target_properties(
  flagdnn_backend_mthreads
  PROPERTIES
    BUILD_RPATH "$ORIGIN/flagdnn/mthreads"
    BUILD_RPATH_USE_ORIGIN FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/flagdnn/mthreads"
    INSTALL_RPATH_USE_LINK_PATH FALSE
    CXX_EXTENSIONS OFF)
if(UNIX AND NOT APPLE)
  target_link_options(
    flagdnn_backend_mthreads
    PRIVATE "LINKER:--disable-new-dtags")
endif()

# Values below are read by the platform validation subdirectory only. Keeping
# them as INTERNAL cache entries avoids public target propagation and prevents
# the production plugin from acquiring muDNN/reference dependencies.
set(FLAGDNN_MTHREADS_VALIDATION_MUDNN_LIBRARY
    "${FLAGDNN_MTHREADS_DEPENDENCIES_MUDNN_LIBRARY}"
    CACHE INTERNAL "")
set(FLAGDNN_MTHREADS_VALIDATION_MUDNN_INCLUDE_DIR
    "${FLAGDNN_MTHREADS_DEPENDENCIES_MUDNN_INCLUDE_DIR}"
    CACHE INTERNAL "")
set(FLAGDNN_MTHREADS_VALIDATION_ENVIRONMENT_REPORT
    "${FLAGDNN_MTHREADS_DEPENDENCIES_ENVIRONMENT_REPORT}"
    CACHE INTERNAL "")
set(FLAGDNN_MTHREADS_VALIDATION_PRIVATE_JIT
    "${FLAGDNN_MTHREADS_JIT_LIBRARY}"
    CACHE INTERNAL "" FORCE)
set(FLAGDNN_MTHREADS_VALIDATION_JIT_SONAME
    "${FLAGDNN_MTHREADS_JIT_SONAME}"
    CACHE INTERNAL "" FORCE)
set(FLAGDNN_MTHREADS_VALIDATION_JIT_PRIVATE_SHA256
    "${FLAGDNN_MTHREADS_JIT_PRIVATE_SHA256}"
    CACHE INTERNAL "" FORCE)
set(FLAGDNN_MTHREADS_VALIDATION_JIT_PROVENANCE_SHA256
    "${FLAGDNN_MTHREADS_JIT_PROVENANCE_SHA256}"
    CACHE INTERNAL "" FORCE)
set(FLAGDNN_MTHREADS_VALIDATION_JIT_GEN_SSIG_SHA256
    "${_flagdnn_mthreads_gen_ssig_sha256}"
    CACHE INTERNAL "" FORCE)
set(FLAGDNN_MTHREADS_VALIDATION_JIT_STANDALONE_SHA256
    "${_flagdnn_mthreads_standalone_compile_sha256}"
    CACHE INTERNAL "" FORCE)
set(FLAGDNN_MTHREADS_VALIDATION_ENVIRONMENT_IDENTITY
    "${FLAGDNN_MTHREADS_DEPENDENCIES_ENVIRONMENT_IDENTITY_SHA256}"
    CACHE INTERNAL "" FORCE)
set(FLAGDNN_MTHREADS_VALIDATION_MUSA_ROOT
    "${FLAGDNN_MTHREADS_DEPENDENCIES_MUSA_ROOT_REALPATH}"
    CACHE INTERNAL "" FORCE)
set(FLAGDNN_MTHREADS_VALIDATION_RUNTIME_LIBRARY_DIRECTORIES
    "${FLAGDNN_MTHREADS_DEPENDENCIES_RUNTIME_LIBRARY_DIRECTORIES}"
    CACHE INTERNAL "" FORCE)

install(
  FILES
    compiler.py
    environment_identity.py
  DESTINATION
    "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/mthreads")
install(
  FILES "${FLAGDNN_MTHREADS_DEPENDENCIES_ENVIRONMENT_REPORT}"
  RENAME flagdnn_mthreads_environment.json
  DESTINATION
    "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/mthreads")
set(FLAGDNN_MTHREADS_INSTALL_JIT_RELATIVE_PATH
    "../../../../${CMAKE_INSTALL_LIBDIR}/flagdnn/mthreads/${FLAGDNN_MTHREADS_JIT_SONAME}")
set(FLAGDNN_MTHREADS_INSTALL_SCRIPT_RELATIVE_ROOT
    "../../../../${CMAKE_INSTALL_LIBDIR}/flagdnn/share/triton_jit/scripts")
configure_file(
  cmake/flagdnn_mthreads_install.json.in
  "${CMAKE_CURRENT_BINARY_DIR}/flagdnn_mthreads_install.json"
  @ONLY)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/flagdnn_mthreads_install.json"
  DESTINATION
    "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/mthreads")
install(
  FILES kernels/registry.json
  DESTINATION
    "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/mthreads/kernels")
install(
  DIRECTORY kernels/
  DESTINATION
    "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/mthreads/kernels"
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE)
install(
  DIRECTORY tuning/
  DESTINATION
    "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/mthreads/tuning"
  FILES_MATCHING PATTERN "*.yaml")

message(STATUS
  "mthreads MUSA root: "
  "${FLAGDNN_MTHREADS_DEPENDENCIES_MUSA_ROOT_REALPATH}")
message(STATUS
  "mthreads environment identity: "
  "${FLAGDNN_MTHREADS_DEPENDENCIES_ENVIRONMENT_IDENTITY_SHA256}")
message(STATUS
  "mthreads private TritonJIT: ${FLAGDNN_MTHREADS_JIT_LIBRARY}")

install(DIRECTORY dispatch codegen
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/mthreads"
  FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
