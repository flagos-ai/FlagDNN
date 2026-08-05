cmake_policy(SET CMP0057 NEW)

if(NOT DEFINED SOURCE_ROOT)
  message(FATAL_ERROR "SOURCE_ROOT is required")
endif()

foreach(required_directory IN ITEMS
        tests/common
        tests/core
        benchmark/common
        backends
        src/graph/lowering)
  if(NOT IS_DIRECTORY "${SOURCE_ROOT}/${required_directory}")
    message(FATAL_ERROR
      "Required architecture directory is missing: ${required_directory}")
  endif()
endforeach()

foreach(required_file IN ITEMS
        backends/CMakeLists.txt
        backends/autotune_policy.cpp
        backends/autotune_policy.hpp
        src/graph/validation.cpp
        src/graph/validation.hpp
        src/graph/ir.cpp
        src/graph/ir.hpp
        src/graph/lowering/lowering.hpp
        src/graph/lowering/helpers.hpp
        src/graph/lowering/dispatch.cpp
        src/graph/lowering/pointwise.cpp
        src/graph/lowering/reduction.cpp
        src/graph/lowering/matmul.cpp
        src/graph/lowering/layout.cpp
        src/graph/lowering/convolution.cpp
        src/graph/lowering/normalization.cpp
        src/graph/lowering/attention.cpp)
  if(NOT EXISTS "${SOURCE_ROOT}/${required_file}")
    message(FATAL_ERROR
      "Required architecture file is missing: ${required_file}")
  endif()
endforeach()

file(GLOB backend_build_files
  "${SOURCE_ROOT}/backends/*/CMakeLists.txt")
if(NOT backend_build_files)
  message(FATAL_ERROR
    "At least one buildable backend is required under backends/<platform>")
endif()
set(active_backends)
foreach(build_file IN LISTS backend_build_files)
  get_filename_component(platform_directory "${build_file}" DIRECTORY)
  get_filename_component(platform "${platform_directory}" NAME)
  if(NOT platform MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "Invalid backend directory name: ${platform}")
  endif()
  list(APPEND active_backends "${platform}")
  foreach(required_validation_path IN ITEMS
          validation
          validation/CMakeLists.txt
          validation/functional
          validation/benchmark)
    if(NOT EXISTS "${platform_directory}/${required_validation_path}")
      message(FATAL_ERROR
        "Backend ${platform} is missing ${required_validation_path}")
    endif()
  endforeach()
endforeach()

foreach(forbidden_path IN ITEMS
        python
        devtools
        src/flag_dnn
        test_support
        tests/reference
        tests/functional
        tests/api
        tests/backend
        tests/native
        tests/integration
        tests/installed_consumer
        tests/platforms
        benchmark/reference
        benchmark/cases
        benchmark/platforms
        backends/common
        cmake/FindCUDNN.cmake
        cmake/NvidiaTests.cmake
        cmake/Benchmarks.cmake
        cmake/VerifyNativeDependencies.cmake
        cmake/VerifyReferenceDependencies.cmake
        src/graph/lowering/common.hpp
        src/graph/lowering/pointwise.hpp)
  if(EXISTS "${SOURCE_ROOT}/${forbidden_path}")
    message(FATAL_ERROR "Forbidden legacy path exists: ${forbidden_path}")
  endif()
endforeach()

foreach(public_script IN ITEMS tools/build.sh tools/install.sh)
  if(NOT EXISTS "${SOURCE_ROOT}/${public_script}")
    message(FATAL_ERROR "Required public script is missing: ${public_script}")
  endif()
  file(READ "${SOURCE_ROOT}/${public_script}" script_source)
  if(script_source MATCHES
     "(CUDNN_|CUDA_HOME|CANN_HOME|Torch_DIR|TritonJIT_DIR|LIBTRITON_JIT_ROOT)")
    message(FATAL_ERROR
      "Platform SDK discovery leaked into public script ${public_script}")
  endif()
endforeach()

file(GLOB test_first_level LIST_DIRECTORIES true "${SOURCE_ROOT}/tests/*")
set(_flagdnn_allowed_test_directories common core)
foreach(entry IN LISTS test_first_level)
  if(NOT IS_DIRECTORY "${entry}")
    continue()
  endif()
  get_filename_component(name "${entry}" NAME)
  if(NOT name IN_LIST _flagdnn_allowed_test_directories)
    message(FATAL_ERROR
      "Unexpected tests/ directory ${name}; only common and core are allowed")
  endif()
endforeach()

file(GLOB benchmark_first_level LIST_DIRECTORIES true
  "${SOURCE_ROOT}/benchmark/*")
foreach(entry IN LISTS benchmark_first_level)
  if(NOT IS_DIRECTORY "${entry}")
    continue()
  endif()
  get_filename_component(name "${entry}" NAME)
  if(NOT name STREQUAL "common")
    message(FATAL_ERROR
      "Unexpected benchmark/ directory ${name}; only common is allowed")
  endif()
endforeach()

file(GLOB_RECURSE nested_readmes "${SOURCE_ROOT}/*/README.md")
if(nested_readmes)
  message(FATAL_ERROR
    "Only the repository-root README.md is allowed: ${nested_readmes}")
endif()

include("${SOURCE_ROOT}/cmake/Operators.cmake")

file(GLOB cpp_entries "${SOURCE_ROOT}/tests/test_*.cpp")
set(actual_operators)
foreach(entry IN LISTS cpp_entries)
  get_filename_component(stem "${entry}" NAME_WE)
  string(REGEX REPLACE "^test_" "" operator "${stem}")
  file(READ "${entry}" source)

  if(source MATCHES "#[ \\t]*include[ \\t]*\"common/[^\"]+\\.hpp\"")
    list(APPEND actual_operators "${operator}")
  else()
    message(FATAL_ERROR
      "${entry} must consume a real tests/common contract")
  endif()
  if(source MATCHES
     "#[ \\t]*include[ \\t]*\"(platforms|validation|backends)/")
    message(FATAL_ERROR "${entry} must remain platform-neutral")
  endif()
endforeach()

set(expected_operators ${FLAGDNN_FUNCTIONAL_OPERATORS})
list(SORT actual_operators)
list(SORT expected_operators)
if(NOT "${actual_operators}" STREQUAL "${expected_operators}")
  message(FATAL_ERROR
    "C++ functional entries do not match the public operator manifest.\n"
    "Manifest: ${expected_operators}\n"
    "Entries:  ${actual_operators}")
endif()

file(GLOB_RECURSE common_test_sources
  "${SOURCE_ROOT}/tests/common/*.c"
  "${SOURCE_ROOT}/tests/common/*.cc"
  "${SOURCE_ROOT}/tests/common/*.cpp"
  "${SOURCE_ROOT}/tests/common/*.h"
  "${SOURCE_ROOT}/tests/common/*.hpp")
foreach(entry IN LISTS common_test_sources)
  file(READ "${entry}" source)
  if(source MATCHES
     "#[ \\t]*include[ \\t]*(<cuda|<cudnn|<acl|\"(platforms|validation|backends)/)")
    message(FATAL_ERROR
      "Platform SDK/include leaked into platform-neutral test source: ${entry}")
  endif()
endforeach()

file(GLOB_RECURSE python_test_sources
  "${SOURCE_ROOT}/tests/*.py"
  "${SOURCE_ROOT}/benchmark/*.py")
if(python_test_sources)
  message(FATAL_ERROR
    "Python tests and benchmarks are forbidden: ${python_test_sources}")
endif()

file(GLOB benchmark_cpp_entries "${SOURCE_ROOT}/benchmark/test_*.cpp")
set(actual_benchmark_operators)
foreach(entry IN LISTS benchmark_cpp_entries)
  get_filename_component(stem "${entry}" NAME_WE)
  string(REGEX REPLACE "^test_" "" operator "${stem}")
  file(READ "${entry}" source)

  if(NOT source MATCHES
     "#[ \\t]*include[ \\t]*\"common/cases\\.hpp\"")
    message(FATAL_ERROR
      "${entry} must consume the platform-neutral benchmark case catalog")
  endif()
  if(NOT source MATCHES
     "#[ \\t]*include[ \\t]*\"common/runner\\.hpp\"")
    message(FATAL_ERROR
      "${entry} must use the shared benchmark runner contract")
  endif()
  if(source MATCHES
     "#[ \\t]*include[ \\t]*\"(platforms|validation|backends)/")
    message(FATAL_ERROR "${entry} must remain platform-neutral")
  endif()
  list(APPEND actual_benchmark_operators "${operator}")
endforeach()

set(expected_benchmark_operators ${FLAGDNN_BENCHMARK_OPERATORS})
list(SORT actual_benchmark_operators)
list(SORT expected_benchmark_operators)
if(NOT "${actual_benchmark_operators}" STREQUAL
       "${expected_benchmark_operators}")
  message(FATAL_ERROR
    "C++ benchmark entries do not match the benchmark operator manifest.\n"
    "Manifest: ${expected_benchmark_operators}\n"
    "Entries:  ${actual_benchmark_operators}")
endif()

file(GLOB_RECURSE common_benchmark_sources
  "${SOURCE_ROOT}/benchmark/common/*.c"
  "${SOURCE_ROOT}/benchmark/common/*.cc"
  "${SOURCE_ROOT}/benchmark/common/*.cpp"
  "${SOURCE_ROOT}/benchmark/common/*.h"
  "${SOURCE_ROOT}/benchmark/common/*.hpp")
foreach(entry IN LISTS common_benchmark_sources)
  file(READ "${entry}" source)
  if(source MATCHES
     "#[ \\t]*include[ \\t]*(<cuda|<cudnn|<acl|\"(platforms|validation|backends)/)")
    message(FATAL_ERROR
      "Platform SDK/include leaked into platform-neutral benchmark source: ${entry}")
  endif()
endforeach()

file(GLOB_RECURSE lowering_sources
  "${SOURCE_ROOT}/src/graph/lowering/*.cpp"
  "${SOURCE_ROOT}/src/graph/lowering/*.hpp")
foreach(entry IN LISTS lowering_sources)
  file(READ "${entry}" source)
  if(source MATCHES "(nvidia|NVIDIA|cuda|CUDA|ascend|Ascend|iluvatar|Iluvatar)")
    message(FATAL_ERROR
      "Platform detail leaked into graph lowering: ${entry}")
  endif()
endforeach()

file(GLOB_RECURSE production_backend_sources
  "${SOURCE_ROOT}/backends/*.cpp"
  "${SOURCE_ROOT}/backends/*.hpp"
  "${SOURCE_ROOT}/backends/*/*.cpp"
  "${SOURCE_ROOT}/backends/*/*.hpp")
foreach(entry IN LISTS production_backend_sources)
  if(entry MATCHES "/validation/")
    continue()
  endif()
  file(READ "${entry}" source)
  if(source MATCHES "validation/")
    message(FATAL_ERROR
      "Production backend source depends on validation code: ${entry}")
  endif()
endforeach()

list(LENGTH active_backends backend_count)
list(LENGTH actual_operators operator_count)
list(LENGTH actual_benchmark_operators benchmark_operator_count)
message(STATUS
  "Verified ${backend_count} backend, ${operator_count} functional and ${benchmark_operator_count} benchmark entries")
