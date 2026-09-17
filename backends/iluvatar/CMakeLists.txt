if(NOT FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
  message(FATAL_ERROR
    "FlagDNN Iluvatar backend supports only the libtriton_jit execution engine")
endif()
if(NOT FLAGDNN_CODEGEN_PYTHON)
  message(FATAL_ERROR
    "Iluvatar libtriton_jit execution requires FLAGDNN_CODEGEN_PYTHON")
endif()

set(FLAGDNN_ILUVATAR_COREX_ROOT ""
  CACHE PATH "Root of the selected CoreX SDK")
set(FLAGDNN_ILUVATAR_CUDA_INCLUDE_DIR ""
  CACHE PATH "CoreX directory containing cuda.h")
set(FLAGDNN_ILUVATAR_CUDA_DRIVER_LIBRARY ""
  CACHE FILEPATH "Selected CoreX libcuda shared object")
set(FLAGDNN_ILUVATAR_CUDNN_INCLUDE_DIR ""
  CACHE PATH "Validation-only CoreX directory containing cudnn.h")
set(FLAGDNN_ILUVATAR_CUDNN_LIBRARY ""
  CACHE FILEPATH "Validation-only CoreX libcudnn.so.7")
set(FLAGDNN_ILUVATAR_TRITON_JIT_ROOT ""
  CACHE PATH "Root of the external IX libtriton_jit distribution")
set(FLAGDNN_ILUVATAR_TRITON_JIT_DIR ""
  CACHE PATH "Directory containing IX TritonJITConfig.cmake")
set(FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY ""
  CACHE FILEPATH "Path to the external IX libtriton_jit shared object")
set(FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR ""
  CACHE PATH "Path to the external IX libtriton_jit public headers")
set(FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR ""
  CACHE PATH "Path to the external IX libtriton_jit scripts")
set(FLAGDNN_ILUVATAR_PYTHONPATH ""
  CACHE STRING "Python module search path for IX libtriton_jit")

set(_flagdnn_iluvatar_resources
  compiler.py
  codegen/__init__.py
  codegen/arguments.py
  codegen/emit.py
  codegen/environment.py
  codegen/identity.py
  codegen/io.py
  codegen/jit.py
  dispatch/__init__.py
  dispatch/attention.py
  dispatch/causal_convolution.py
  dispatch/common.py
  dispatch/convolution.py
  dispatch/extended.py
  dispatch/extended_common.py
  dispatch/graph.py
  dispatch/graph_tensor.py
  dispatch/index.py
  dispatch/nn.py
  dispatch/nn_common.py
  dispatch/normalization.py
  dispatch/normalization_extended.py
  dispatch/pointwise.py
  dispatch/position_embedding.py
  dispatch/random.py
  dispatch/resample.py
  dispatch/selection.py
  dispatch/statistics.py
  dispatch/tensor.py
  dispatch/tuning.py
  kernels/random.py
  kernels/statistics.py
  kernels/causal_convolution.py
  kernels/activation_backward.py
  kernels/attention.py
  kernels/binary.py
  kernels/convolution.py
  kernels/layout.py
  kernels/matmul.py
  kernels/normalization.py
  kernels/reduction.py
  kernels/registry.json
  kernels/ternary.py
  kernels/unary.py
  tuning/common.yaml
  tuning/convolution.yaml
  tuning/normalization.yaml
)
foreach(_resource IN LISTS _flagdnn_iluvatar_resources)
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${_resource}")
    message(FATAL_ERROR "Required Iluvatar resource is missing: ${_resource}")
  endif()
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/${_resource}")
endforeach()

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ResolveCoreX.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ResolveTritonJIT.cmake")
flagdnn_iluvatar_resolve_corex(_flagdnn_iluvatar_corex)
if(FLAGDNN_BUILD_TESTS OR FLAGDNN_BUILD_BENCHMARKS)
  flagdnn_iluvatar_resolve_corex_cudnn(
    _flagdnn_iluvatar_cudnn "${_flagdnn_iluvatar_corex_ROOT}")
  set(FLAGDNN_ILUVATAR_RESOLVED_CUDNN_INCLUDE_DIR
    "${_flagdnn_iluvatar_cudnn_INCLUDE_DIR}" CACHE INTERNAL "")
  set(FLAGDNN_ILUVATAR_RESOLVED_CUDNN_LIBRARY
    "${_flagdnn_iluvatar_cudnn_LIBRARY}" CACHE INTERNAL "")
  set(FLAGDNN_ILUVATAR_RESOLVED_CUDA_RUNTIME_LIBRARY
    "${_flagdnn_iluvatar_cudnn_CUDA_RUNTIME_LIBRARY}" CACHE INTERNAL "")
  set(FLAGDNN_ILUVATAR_RESOLVED_CUDNN_HEADER_VERSION
    "${_flagdnn_iluvatar_cudnn_HEADER_VERSION}" CACHE INTERNAL "")
  set(FLAGDNN_ILUVATAR_RESOLVED_CUDNN_SONAME
    "${_flagdnn_iluvatar_cudnn_SONAME}" CACHE INTERNAL "")
endif()

execute_process(
  COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -c "import torch, triton, yaml"
  RESULT_VARIABLE _flagdnn_iluvatar_python_result
  ERROR_VARIABLE _flagdnn_iluvatar_python_error)
if(NOT _flagdnn_iluvatar_python_result EQUAL 0)
  string(STRIP "${_flagdnn_iluvatar_python_error}"
    _flagdnn_iluvatar_python_error)
  message(FATAL_ERROR
    "FLAGDNN_CODEGEN_PYTHON must provide Torch, Triton, and PyYAML for "
    "the Iluvatar libtriton_jit engine: "
    "${_flagdnn_iluvatar_python_error}")
endif()

set(Python_EXECUTABLE "${FLAGDNN_CODEGEN_PYTHON}")
find_package(Python REQUIRED COMPONENTS Interpreter Development)
get_target_property(_flagdnn_iluvatar_python_library
  Python::Python IMPORTED_LOCATION)
if(NOT _flagdnn_iluvatar_python_library)
  get_target_property(_flagdnn_iluvatar_python_library
    Python::Python IMPORTED_LOCATION_RELEASE)
endif()
if(NOT _flagdnn_iluvatar_python_library)
  message(FATAL_ERROR "Cannot resolve Python::Python shared library")
endif()
get_filename_component(_flagdnn_iluvatar_python_library
  "${_flagdnn_iluvatar_python_library}" REALPATH)
get_filename_component(_flagdnn_iluvatar_python_soname
  "${_flagdnn_iluvatar_python_library}" NAME)
set(CUDAToolkit_ROOT "${_flagdnn_iluvatar_corex_ROOT}")
flagdnn_iluvatar_resolve_triton_jit(_flagdnn_iluvatar_jit)

if(NOT FLAGDNN_ILUVATAR_PYTHONPATH)
  execute_process(
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -c
      "import os,pathlib,triton,torch,yaml; roots=dict.fromkeys(str(pathlib.Path(module.__file__).resolve().parent.parent) for module in (torch,triton,yaml)); print(os.pathsep.join(roots))"
    RESULT_VARIABLE _flagdnn_iluvatar_pythonpath_result
    OUTPUT_VARIABLE _flagdnn_iluvatar_pythonpath
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _flagdnn_iluvatar_pythonpath_result EQUAL 0 OR
     _flagdnn_iluvatar_pythonpath STREQUAL "")
    message(FATAL_ERROR
      "Cannot locate Python package roots for FLAGDNN_CODEGEN_PYTHON")
  endif()
else()
  set(_flagdnn_iluvatar_pythonpath "${FLAGDNN_ILUVATAR_PYTHONPATH}")
endif()

set(FLAGDNN_ILUVATAR_RESOLVED_COREX_ROOT
  "${_flagdnn_iluvatar_corex_ROOT}" CACHE INTERNAL "")
set(FLAGDNN_ILUVATAR_RESOLVED_CUDA_INCLUDE_DIR
  "${_flagdnn_iluvatar_corex_CUDA_INCLUDE_DIR}" CACHE INTERNAL "")
set(FLAGDNN_ILUVATAR_RESOLVED_CUDA_DRIVER_LIBRARY
  "${_flagdnn_iluvatar_corex_CUDA_DRIVER_LIBRARY}" CACHE INTERNAL "")
set(FLAGDNN_ILUVATAR_RESOLVED_COREX_SDK_VERSION
  "${_flagdnn_iluvatar_corex_SDK_VERSION}" CACHE INTERNAL "")
set(FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_TARGET
  "${_flagdnn_iluvatar_jit_TARGET}" CACHE INTERNAL "")
set(FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_CONFIG_DIR
  "${_flagdnn_iluvatar_jit_CONFIG_DIR}" CACHE INTERNAL "")
set(FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_LIBRARY
  "${_flagdnn_iluvatar_jit_LIBRARY}" CACHE INTERNAL "")
set(FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_INCLUDE_DIR
  "${_flagdnn_iluvatar_jit_INCLUDE_DIR}" CACHE INTERNAL "")
set(FLAGDNN_ILUVATAR_RESOLVED_TRITON_JIT_SCRIPT_DIR
  "${_flagdnn_iluvatar_jit_SCRIPT_DIR}" CACHE INTERNAL "")
set(FLAGDNN_ILUVATAR_RESOLVED_PYTHONPATH
  "${_flagdnn_iluvatar_pythonpath}" CACHE INTERNAL "")

message(STATUS
  "FlagDNN Iluvatar CoreX: ${_flagdnn_iluvatar_corex_ROOT} "
  "(SDK ${_flagdnn_iluvatar_corex_SDK_VERSION})")
message(STATUS
  "FlagDNN Iluvatar CUDA Driver: "
  "${_flagdnn_iluvatar_corex_CUDA_DRIVER_LIBRARY}")
if(FLAGDNN_BUILD_TESTS OR FLAGDNN_BUILD_BENCHMARKS)
  message(STATUS
    "FlagDNN Iluvatar validation cuDNN: "
    "${_flagdnn_iluvatar_cudnn_LIBRARY} "
    "(header ${_flagdnn_iluvatar_cudnn_HEADER_VERSION})")
endif()
message(STATUS
  "FlagDNN Iluvatar TritonJIT: ${_flagdnn_iluvatar_jit_BACKEND} at "
  "${_flagdnn_iluvatar_jit_CONFIG_DIR}")


flagdnn_add_backend_plugin(iluvatar INSTALL
  SOURCES
    artifact.cpp
    backend.cpp
    context.cpp
    engines/libtriton_jit.cpp
    error.cpp
    ${PROJECT_SOURCE_DIR}/backends/autotune_policy.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/json.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp
  INCLUDE_DIRECTORIES
    "${_flagdnn_iluvatar_corex_CUDA_INCLUDE_DIR}"
    "${_flagdnn_iluvatar_jit_INCLUDE_DIR}"
  LINK_LIBRARIES
    "${_flagdnn_iluvatar_corex_CUDA_DRIVER_LIBRARY}"
    "${_flagdnn_iluvatar_jit_TARGET}"
    Python::Python
    ${CMAKE_DL_LIBS})
target_compile_definitions(flagdnn_backend_iluvatar PRIVATE
  BACKEND_IX=1
  FLAGDNN_ILUVATAR_PYTHONPATH="${_flagdnn_iluvatar_pythonpath}"
  FLAGDNN_ILUVATAR_PYTHON_LIBRARY_SONAME="${_flagdnn_iluvatar_python_soname}")
target_link_directories(flagdnn_backend_iluvatar PRIVATE
  "${_flagdnn_iluvatar_corex_ROOT}/lib64"
  "${_flagdnn_iluvatar_corex_ROOT}/lib")
set_target_properties(flagdnn_backend_iluvatar PROPERTIES
  INSTALL_RPATH "$ORIGIN;/usr/local/lib"
  INSTALL_RPATH_USE_LINK_PATH YES)

install(FILES
    compiler.py
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/iluvatar")
install(DIRECTORY codegen dispatch
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/iluvatar"
  FILES_MATCHING PATTERN "*.py" PATTERN "__pycache__" EXCLUDE)
install(FILES kernels/registry.json
  DESTINATION
    "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/iluvatar/kernels")
install(DIRECTORY kernels/
  DESTINATION
    "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/iluvatar/kernels"
  FILES_MATCHING
    PATTERN "*.py"
    PATTERN "__pycache__" EXCLUDE)
install(DIRECTORY tuning/
  DESTINATION
    "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/iluvatar/tuning"
  FILES_MATCHING
    PATTERN "*.yaml")
