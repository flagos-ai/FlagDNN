set(FLAGDNN_CUDA_INCLUDE_DIR ""
    CACHE PATH "CUDA Driver API include directory (directory containing cuda.h)")
set(FLAGDNN_CUDA_DRIVER_LIBRARY ""
    CACHE FILEPATH "Path to the CUDA Driver library")

if(FLAGDNN_CUDA_INCLUDE_DIR OR FLAGDNN_CUDA_DRIVER_LIBRARY)
  if(NOT FLAGDNN_CUDA_INCLUDE_DIR OR NOT FLAGDNN_CUDA_DRIVER_LIBRARY)
    message(FATAL_ERROR
      "FLAGDNN_CUDA_INCLUDE_DIR and FLAGDNN_CUDA_DRIVER_LIBRARY must be set together")
  endif()
  if(NOT EXISTS "${FLAGDNN_CUDA_INCLUDE_DIR}/cuda.h")
    message(FATAL_ERROR "cuda.h was not found under FLAGDNN_CUDA_INCLUDE_DIR")
  endif()
  if(NOT EXISTS "${FLAGDNN_CUDA_DRIVER_LIBRARY}")
    message(FATAL_ERROR "FLAGDNN_CUDA_DRIVER_LIBRARY does not exist")
  endif()
  set(_flagdnn_cuda_include "${FLAGDNN_CUDA_INCLUDE_DIR}")
  set(_flagdnn_cuda_driver "${FLAGDNN_CUDA_DRIVER_LIBRARY}")
else()
  find_package(CUDAToolkit REQUIRED)
  set(_flagdnn_cuda_include "${CUDAToolkit_INCLUDE_DIRS}")
  set(_flagdnn_cuda_driver CUDA::cuda_driver)
endif()

set(_flagdnn_cuda_engine_libraries)
if(FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
  if(NOT FLAGDNN_CODEGEN_PYTHON)
    message(FATAL_ERROR
      "libtriton_jit execution requires FLAGDNN_CODEGEN_PYTHON")
  endif()

  execute_process(
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -c
      "import torch, triton, yaml"
    RESULT_VARIABLE _flagdnn_nvidia_python_dependencies_result
    ERROR_VARIABLE _flagdnn_nvidia_python_dependencies_error)
  if(NOT _flagdnn_nvidia_python_dependencies_result EQUAL 0)
    string(STRIP "${_flagdnn_nvidia_python_dependencies_error}"
      _flagdnn_nvidia_python_dependencies_error)
    message(FATAL_ERROR
      "FLAGDNN_CODEGEN_PYTHON must provide Torch, Triton, and PyYAML for "
      "the NVIDIA libtriton_jit engine: "
      "${_flagdnn_nvidia_python_dependencies_error}")
  endif()

  if(NOT DEFINED TritonJIT_DIR OR TritonJIT_DIR STREQUAL "")
    set(_flagdnn_triton_jit_candidates)
    if(DEFINED ENV{TritonJIT_DIR} AND
       NOT "$ENV{TritonJIT_DIR}" STREQUAL "")
      list(APPEND _flagdnn_triton_jit_candidates "$ENV{TritonJIT_DIR}")
    endif()
    if(DEFINED ENV{LIBTRITON_JIT_ROOT} AND
       NOT "$ENV{LIBTRITON_JIT_ROOT}" STREQUAL "")
      list(APPEND _flagdnn_triton_jit_candidates
        "$ENV{LIBTRITON_JIT_ROOT}/build"
        "$ENV{LIBTRITON_JIT_ROOT}/lib/cmake/TritonJIT")
    endif()
    list(APPEND _flagdnn_triton_jit_candidates
      "${PROJECT_SOURCE_DIR}/../libtriton_jit/build")
    foreach(_flagdnn_triton_jit_candidate IN LISTS
        _flagdnn_triton_jit_candidates)
      if(EXISTS
          "${_flagdnn_triton_jit_candidate}/TritonJITConfig.cmake")
        get_filename_component(_flagdnn_triton_jit_candidate
          "${_flagdnn_triton_jit_candidate}" ABSOLUTE)
        set(TritonJIT_DIR "${_flagdnn_triton_jit_candidate}"
          CACHE PATH "Directory containing TritonJITConfig.cmake")
        break()
      endif()
    endforeach()
  endif()

  if(DEFINED TritonJIT_DIR AND NOT TritonJIT_DIR STREQUAL "")
    foreach(_flagdnn_triton_jit_module_candidate IN ITEMS
        "${TritonJIT_DIR}"
        "${TritonJIT_DIR}/cmake"
        "${TritonJIT_DIR}/../cmake")
      if(EXISTS
          "${_flagdnn_triton_jit_module_candidate}/FindTorch.cmake")
        get_filename_component(_flagdnn_triton_jit_module_candidate
          "${_flagdnn_triton_jit_module_candidate}" ABSOLUTE)
        list(PREPEND CMAKE_MODULE_PATH
          "${_flagdnn_triton_jit_module_candidate}")
        break()
      endif()
    endforeach()
  endif()

  if(NOT DEFINED Torch_DIR OR Torch_DIR STREQUAL "")
    execute_process(
      COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -c
        "import pathlib, torch; print(pathlib.Path(torch.utils.cmake_prefix_path) / 'Torch')"
      RESULT_VARIABLE _flagdnn_torch_directory_result
      OUTPUT_VARIABLE _flagdnn_torch_directory
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_flagdnn_torch_directory_result EQUAL 0 AND
       EXISTS "${_flagdnn_torch_directory}/TorchConfig.cmake")
      set(Torch_DIR "${_flagdnn_torch_directory}"
        CACHE PATH "Torch CMake package used by libtriton_jit")
    endif()
  endif()

  set(Python_EXECUTABLE "${FLAGDNN_CODEGEN_PYTHON}")
  find_package(TritonJIT 0.1.0 CONFIG REQUIRED)
  if(TARGET TritonJIT::triton_jit)
    set(_flagdnn_triton_jit_target TritonJIT::triton_jit)
  elseif(TARGET triton_jit)
    # Older build-tree exports did not preserve TritonJIT's namespace.
    set(_flagdnn_triton_jit_target triton_jit)
  else()
    message(FATAL_ERROR
      "TritonJIT was found but did not export a usable triton_jit target")
  endif()
  if(NOT TritonJIT_BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR
      "FlagDNN NVIDIA backend requires a CUDA build of libtriton_jit")
  endif()
  execute_process(
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -c
      "import site; print(site.getsitepackages()[0])"
    RESULT_VARIABLE _flagdnn_site_packages_result
    OUTPUT_VARIABLE _flagdnn_site_packages
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _flagdnn_site_packages_result EQUAL 0 OR
     NOT IS_DIRECTORY "${_flagdnn_site_packages}")
    message(FATAL_ERROR
      "Cannot locate site-packages for FLAGDNN_CODEGEN_PYTHON")
  endif()
  list(APPEND _flagdnn_cuda_engine_libraries
    ${_flagdnn_triton_jit_target} Python::Python)

  set(_flagdnn_triton_jit_library "")
  foreach(_flagdnn_location_property IN ITEMS
      IMPORTED_LOCATION IMPORTED_LOCATION_NOCONFIG
      IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_RELWITHDEBINFO
      IMPORTED_LOCATION_DEBUG)
    if(NOT _flagdnn_triton_jit_library)
      get_target_property(_flagdnn_candidate_location
        ${_flagdnn_triton_jit_target} ${_flagdnn_location_property})
      if(_flagdnn_candidate_location AND
         EXISTS "${_flagdnn_candidate_location}")
        set(_flagdnn_triton_jit_library "${_flagdnn_candidate_location}")
      endif()
    endif()
  endforeach()
  if(NOT _flagdnn_triton_jit_library)
    message(FATAL_ERROR
      "Cannot determine the libtriton_jit library used by the NVIDIA backend")
  endif()
  file(SHA256 "${_flagdnn_triton_jit_library}"
    _flagdnn_triton_jit_library_sha256)
  execute_process(
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -c
      "import platform, triton, torch; print('|'.join((platform.python_version(), triton.__version__, torch.__version__)))"
    RESULT_VARIABLE _flagdnn_jit_environment_result
    OUTPUT_VARIABLE _flagdnn_jit_environment
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _flagdnn_jit_environment_result EQUAL 0 OR
     _flagdnn_jit_environment STREQUAL "")
    message(FATAL_ERROR
      "Cannot identify the Python/Triton/Torch environment for libtriton_jit")
  endif()
  string(SHA256 _flagdnn_jit_environment_sha256
    "${_flagdnn_jit_environment}")
  string(SUBSTRING "${_flagdnn_triton_jit_library_sha256}" 0 16
    _flagdnn_triton_jit_library_short_sha256)
  string(SUBSTRING "${_flagdnn_jit_environment_sha256}" 0 16
    _flagdnn_jit_environment_short_sha256)
  set(_flagdnn_triton_jit_build_identity
    "${_flagdnn_triton_jit_library_short_sha256}-${_flagdnn_jit_environment_short_sha256}")
endif()

flagdnn_add_backend_plugin(nvidia INSTALL
  SOURCES
    artifact.cpp
    ${PROJECT_SOURCE_DIR}/backends/autotune_policy.cpp
    autotune.cpp
    backend.cpp
    context.cpp
    engines/engine.cpp
    engines/external_artifact.cpp
    engines/libtriton_jit.cpp
    error.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/json.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp
  INCLUDE_DIRECTORIES
    ${_flagdnn_cuda_include}
  LINK_LIBRARIES
    ${_flagdnn_cuda_driver}
    ${_flagdnn_cuda_engine_libraries})

if(FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
  target_compile_definitions(flagdnn_backend_nvidia PRIVATE
    FLAGDNN_HAS_LIBTRITON_JIT=1
    FLAGDNN_LIBTRITON_JIT_PYTHONPATH="${_flagdnn_site_packages}"
    FLAGDNN_LIBTRITON_JIT_BUILD_IDENTITY="${_flagdnn_triton_jit_build_identity}")
  # The plugin is loaded with dlopen and is not part of the public CMake
  # export. Preserve paths to its externally supplied JIT/Python/Torch shared
  # libraries in installed builds; $ORIGIN also supports colocated packaging.
  set_target_properties(flagdnn_backend_nvidia PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH YES)
endif()

install(FILES compiler.py compiler_identity.py
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/nvidia")
install(FILES kernels/registry.json
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/nvidia/kernels")
install(DIRECTORY kernels/
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/nvidia/kernels"
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY tuning/
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/nvidia/tuning"
  FILES_MATCHING PATTERN "*.yaml")
