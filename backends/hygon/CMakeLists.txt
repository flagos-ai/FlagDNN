if(NOT FLAGDNN_EXECUTION_ENGINE STREQUAL "libtriton_jit")
  message(FATAL_ERROR
    "FlagDNN Hygon backend supports only the libtriton_jit execution engine")
endif()
list(FIND FLAGDNN_BACKENDS nvidia _flagdnn_hygon_nvidia_backend_index)
if(NOT _flagdnn_hygon_nvidia_backend_index EQUAL -1)
  message(WARNING
    "NVIDIA and Hygon libtriton_jit plugins can be configured and built "
    "together, but their same-SONAME CUDA/HCU JIT runtimes cannot be loaded "
    "safely in one process. Use separate processes; runtime identity checks "
    "reject a mismatched already-loaded image.")
endif()
unset(_flagdnn_hygon_nvidia_backend_index)
if(NOT FLAGDNN_CODEGEN_PYTHON)
  message(FATAL_ERROR
    "Hygon libtriton_jit execution requires FLAGDNN_CODEGEN_PYTHON")
endif()

# These files are the installed compiler provider, not optional development
# conveniences.  Fail at configure time instead of producing an SDK that can
# build successfully but cannot compile a Hygon graph after installation.
foreach(_flagdnn_hygon_required_resource IN ITEMS
    compiler.py
    codegen
    dispatch
    python_environment_identity.py
    kernels
    tuning)
  if(NOT EXISTS
      "${CMAKE_CURRENT_SOURCE_DIR}/${_flagdnn_hygon_required_resource}")
    message(FATAL_ERROR
      "Required Hygon compiler resource is missing: "
      "${_flagdnn_hygon_required_resource}")
  endif()
endforeach()
unset(_flagdnn_hygon_required_resource)

execute_process(
  COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -c "import torch, triton, yaml"
  RESULT_VARIABLE _flagdnn_hygon_python_dependencies_result
  ERROR_VARIABLE _flagdnn_hygon_python_dependencies_error)
if(NOT _flagdnn_hygon_python_dependencies_result EQUAL 0)
  string(STRIP "${_flagdnn_hygon_python_dependencies_error}"
    _flagdnn_hygon_python_dependencies_error)
  message(FATAL_ERROR
    "FLAGDNN_CODEGEN_PYTHON must provide Torch, Triton, and PyYAML for "
    "the Hygon libtriton_jit engine: "
    "${_flagdnn_hygon_python_dependencies_error}")
endif()

# DTK installs its HIP package below /opt/dtk.  Honor an explicit hip_DIR or
# normal CMAKE_PREFIX_PATH first and use the platform default only as a local
# discovery hint.
if(NOT DEFINED hip_DIR OR hip_DIR STREQUAL "")
  if(EXISTS "/opt/dtk/lib/cmake/hip/hip-config.cmake")
    set(hip_DIR "/opt/dtk/lib/cmake/hip"
      CACHE PATH "Directory containing hip-config.cmake")
  endif()
endif()
find_package(hip CONFIG REQUIRED)
if(NOT TARGET hip::host)
  message(FATAL_ERROR "DTK HIP package did not export hip::host")
endif()

# Do not consume TritonJIT's exported Torch interface: that interface carries
# training-framework DNN/BLAS packages and a trailing -std=c++17 into every
# downstream target.  FlagDNN needs only the JIT C++ ABI, its public headers,
# and the Python headers.  Importing that narrow ABI keeps the production
# plugin's direct dependency boundary at libtriton_jit + Python + HIP.
set(FLAGDNN_HYGON_TRITON_JIT_ROOT ""
  CACHE PATH
  "Root of one HCU libtriton_jit build or installation")
set(FLAGDNN_HYGON_TRITON_JIT_DIR ""
  CACHE PATH
  "Directory containing the HCU TritonJITConfig.cmake")
set(FLAGDNN_HYGON_TRITON_JIT_LIBRARY ""
  CACHE FILEPATH "Path to the HCU libtriton_jit shared library")
set(FLAGDNN_HYGON_TRITON_JIT_INCLUDE_DIR ""
  CACHE PATH "Path to the HCU libtriton_jit public headers")
set(FLAGDNN_HYGON_TRITON_JIT_SCRIPT_DIR ""
  CACHE PATH "Path to the libtriton_jit runtime Python scripts")

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/ResolveTritonJIT.cmake")
set(_flagdnn_hygon_triton_jit_arguments
  DEFAULT_ROOT "${PROJECT_SOURCE_DIR}/../libtriton_jit")
if(FLAGDNN_HYGON_TRITON_JIT_DIR)
  list(APPEND _flagdnn_hygon_triton_jit_arguments
    CONFIG_DIR "${FLAGDNN_HYGON_TRITON_JIT_DIR}")
endif()
if(FLAGDNN_HYGON_TRITON_JIT_ROOT)
  list(APPEND _flagdnn_hygon_triton_jit_arguments
    ROOT "${FLAGDNN_HYGON_TRITON_JIT_ROOT}")
endif()
if(FLAGDNN_HYGON_TRITON_JIT_LIBRARY)
  list(APPEND _flagdnn_hygon_triton_jit_arguments
    LIBRARY "${FLAGDNN_HYGON_TRITON_JIT_LIBRARY}")
endif()
if(FLAGDNN_HYGON_TRITON_JIT_INCLUDE_DIR)
  list(APPEND _flagdnn_hygon_triton_jit_arguments
    INCLUDE_DIR "${FLAGDNN_HYGON_TRITON_JIT_INCLUDE_DIR}")
endif()
if(FLAGDNN_HYGON_TRITON_JIT_SCRIPT_DIR)
  list(APPEND _flagdnn_hygon_triton_jit_arguments
    SCRIPT_DIR "${FLAGDNN_HYGON_TRITON_JIT_SCRIPT_DIR}")
endif()
flagdnn_hygon_resolve_triton_jit(_flagdnn_hygon_triton_jit
  ${_flagdnn_hygon_triton_jit_arguments})
unset(_flagdnn_hygon_triton_jit_arguments)
set(_flagdnn_triton_jit_library
  "${_flagdnn_hygon_triton_jit_LIBRARY}")
set(_flagdnn_triton_jit_include_dir
  "${_flagdnn_hygon_triton_jit_INCLUDE_DIR}")
set(_flagdnn_triton_jit_script_dir
  "${_flagdnn_hygon_triton_jit_SCRIPT_DIR}")
set(_flagdnn_triton_jit_provenance_sha256
  "${_flagdnn_hygon_triton_jit_PROVENANCE_SHA256}")
message(STATUS
  "FlagDNN Hygon TritonJIT: ${_flagdnn_hygon_triton_jit_LAYOUT} layout at "
  "${_flagdnn_hygon_triton_jit_PROVENANCE_ROOT}")

set(_flagdnn_hygon_prepared_standalone_compile
  "${CMAKE_CURRENT_BINARY_DIR}/flagdnn_hygon_standalone_compile.py")
execute_process(
  COMMAND "${FLAGDNN_CODEGEN_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/prepare_standalone_compile.py"
    --input "${_flagdnn_triton_jit_script_dir}/standalone_compile.py"
    --output "${_flagdnn_hygon_prepared_standalone_compile}"
  RESULT_VARIABLE _flagdnn_hygon_prepare_standalone_result
  ERROR_VARIABLE _flagdnn_hygon_prepare_standalone_error)
if(NOT _flagdnn_hygon_prepare_standalone_result EQUAL 0)
  string(STRIP "${_flagdnn_hygon_prepare_standalone_error}"
    _flagdnn_hygon_prepare_standalone_error)
  message(FATAL_ERROR
    "Cannot prepare the Hygon-private libtriton_jit compiler helper: "
    "${_flagdnn_hygon_prepare_standalone_error}")
endif()

file(SHA256 "${_flagdnn_hygon_prepared_standalone_compile}"
  _flagdnn_triton_jit_standalone_sha256)
file(SHA256 "${_flagdnn_triton_jit_script_dir}/gen_ssig.py"
  _flagdnn_triton_jit_gen_ssig_sha256)
file(SHA256 "${CMAKE_CURRENT_SOURCE_DIR}/python_environment_identity.py"
  _flagdnn_python_environment_identity_sha256)
string(SHA256 _flagdnn_triton_jit_scripts_sha256
  "${_flagdnn_triton_jit_standalone_sha256}:${_flagdnn_triton_jit_gen_ssig_sha256}:${_flagdnn_python_environment_identity_sha256}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${_flagdnn_hygon_triton_jit_PROVENANCE_INPUT_FILES}
  "${CMAKE_CURRENT_SOURCE_DIR}/prepare_standalone_compile.py")

set(Python_EXECUTABLE "${FLAGDNN_CODEGEN_PYTHON}")
find_package(Python REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -c
    "import pathlib, torch; root=pathlib.Path(torch.__file__).resolve().parent; print(';'.join((str(root/'include'), str(root/'include/torch/csrc/api/include'), '1' if torch._C._GLIBCXX_USE_CXX11_ABI else '0')))"
  RESULT_VARIABLE _flagdnn_torch_headers_result
  OUTPUT_VARIABLE _flagdnn_torch_headers
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _flagdnn_torch_headers_result EQUAL 0)
  message(FATAL_ERROR
    "Cannot identify the Torch headers used by HCU libtriton_jit")
endif()
list(LENGTH _flagdnn_torch_headers _flagdnn_torch_headers_count)
if(NOT _flagdnn_torch_headers_count EQUAL 3)
  message(FATAL_ERROR
    "Unexpected Torch header discovery result")
endif()
list(GET _flagdnn_torch_headers 0 _flagdnn_torch_include)
list(GET _flagdnn_torch_headers 1 _flagdnn_torch_api_include)
list(GET _flagdnn_torch_headers 2 _flagdnn_torch_cxx11_abi)

add_library(flagdnn_hygon_triton_jit SHARED IMPORTED)
set(_flagdnn_hygon_triton_jit_build_directory
  "${CMAKE_CURRENT_BINARY_DIR}/flagdnn/hygon")
set(_flagdnn_hygon_triton_jit_build_library
  "${_flagdnn_hygon_triton_jit_build_directory}/${_flagdnn_hygon_triton_jit_SONAME}")
# Mirror the installed private-library layout in the build tree and remove
# empty runtime-path entries from our copy of an upstream build-tree library.
# The dependency itself remains unchanged; identity uses the staged bytes.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/StageTritonJIT.cmake")
flagdnn_hygon_stage_triton_jit(
  "${_flagdnn_triton_jit_library}"
  "${_flagdnn_hygon_triton_jit_build_library}")
set_target_properties(flagdnn_hygon_triton_jit PROPERTIES
  IMPORTED_LOCATION "${_flagdnn_hygon_triton_jit_build_library}"
  IMPORTED_SONAME "${_flagdnn_hygon_triton_jit_SONAME}")
set(_flagdnn_triton_jit_target flagdnn_hygon_triton_jit)

# Install the same staged bytes used for linking and runtime identity. Rename
# to the exported SONAME so the plugin's DT_NEEDED entry is available.
set(_flagdnn_hygon_triton_jit_install_directory
  "${CMAKE_INSTALL_LIBDIR}/flagdnn/hygon")
set(_flagdnn_hygon_triton_jit_private_script_install_directory
  "${CMAKE_INSTALL_LIBDIR}/flagdnn/share/triton_jit/scripts")
install(FILES "${_flagdnn_hygon_triton_jit_build_library}"
  DESTINATION "${_flagdnn_hygon_triton_jit_install_directory}"
  RENAME "${_flagdnn_hygon_triton_jit_SONAME}")
# libtriton_jit computes its script root from its own mapped image. With the
# HCU image isolated below lib/flagdnn/hygon, that ABI resolves this private
# sibling directory. Do not also install these generic module names below
# share/triton_jit: that namespace belongs to a separately installed
# libtriton_jit package and would allow one consumer to overwrite another.
install(FILES "${_flagdnn_hygon_prepared_standalone_compile}"
  DESTINATION
    "${_flagdnn_hygon_triton_jit_private_script_install_directory}"
  RENAME standalone_compile.py)
install(FILES "${_flagdnn_triton_jit_script_dir}/gen_ssig.py"
  DESTINATION
    "${_flagdnn_hygon_triton_jit_private_script_install_directory}")

set(FLAGDNN_HYGON_PYTHONPATH ""
  CACHE STRING
  "Python module search path for embedded Hygon libtriton_jit; when empty, discover it from FLAGDNN_CODEGEN_PYTHON")
if(NOT FLAGDNN_HYGON_PYTHONPATH)
  execute_process(
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -c
      "import os,pathlib,triton,torch,yaml; roots=dict.fromkeys(str(pathlib.Path(module.__file__).resolve().parent.parent) for module in (torch,triton,yaml)); print(os.pathsep.join(roots))"
    RESULT_VARIABLE _flagdnn_python_path_result
    OUTPUT_VARIABLE _flagdnn_python_path
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _flagdnn_python_path_result EQUAL 0 OR
     _flagdnn_python_path STREQUAL "")
    message(FATAL_ERROR
      "Cannot locate Python package roots for FLAGDNN_CODEGEN_PYTHON")
  endif()
  set(_flagdnn_hygon_python_path "${_flagdnn_python_path}")
else()
  set(_flagdnn_hygon_python_path "${FLAGDNN_HYGON_PYTHONPATH}")
endif()

file(SHA256 "${_flagdnn_hygon_triton_jit_build_library}"
  _flagdnn_triton_jit_library_sha256)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${_flagdnn_triton_jit_library}"
  "${CMAKE_CURRENT_SOURCE_DIR}/python_environment_identity.py")
execute_process(
  COMMAND "${FLAGDNN_CODEGEN_PYTHON}"
    "${CMAKE_CURRENT_SOURCE_DIR}/python_environment_identity.py"
  RESULT_VARIABLE _flagdnn_jit_environment_result
  OUTPUT_VARIABLE _flagdnn_jit_environment
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _flagdnn_jit_environment_result EQUAL 0 OR
   _flagdnn_jit_environment STREQUAL "")
  message(FATAL_ERROR
    "Cannot identify the Python/Triton/Torch environment for libtriton_jit")
endif()
string(JSON _flagdnn_torch_module_sha256 ERROR_VARIABLE _flagdnn_json_error
  GET "${_flagdnn_jit_environment}" module_file_sha256 torch)
if(_flagdnn_json_error)
  message(FATAL_ERROR
    "Hygon Python environment has no torch module fingerprint: ${_flagdnn_json_error}")
endif()
string(JSON _flagdnn_torch_c_module_sha256 ERROR_VARIABLE _flagdnn_json_error
  GET "${_flagdnn_jit_environment}" module_file_sha256 "torch._C")
if(_flagdnn_json_error)
  message(FATAL_ERROR
    "Hygon Python environment has no torch._C fingerprint: ${_flagdnn_json_error}")
endif()
string(JSON _flagdnn_triton_module_sha256 ERROR_VARIABLE _flagdnn_json_error
  GET "${_flagdnn_jit_environment}" module_file_sha256 triton)
if(_flagdnn_json_error)
  message(FATAL_ERROR
    "Hygon Python environment has no Triton fingerprint: ${_flagdnn_json_error}")
endif()
string(JSON _flagdnn_yaml_module_sha256 ERROR_VARIABLE _flagdnn_json_error
  GET "${_flagdnn_jit_environment}" module_file_sha256 yaml)
if(_flagdnn_json_error)
  message(FATAL_ERROR
    "Hygon Python environment has no PyYAML fingerprint: ${_flagdnn_json_error}")
endif()
string(SHA256 _flagdnn_jit_environment_sha256
  "${_flagdnn_jit_environment}")
flagdnn_hygon_compose_triton_jit_build_identity(
  _flagdnn_triton_jit_build_identity
  PROVENANCE_SHA256 "${_flagdnn_triton_jit_provenance_sha256}"
  PYTHON_ENVIRONMENT_SHA256 "${_flagdnn_jit_environment_sha256}"
  JIT_SCRIPTS_SHA256 "${_flagdnn_triton_jit_scripts_sha256}")
set(_flagdnn_hygon_compiler_environment
  "{\n  \"schema_version\": 3,\n  \"libtriton_jit_soname\": \"${_flagdnn_hygon_triton_jit_SONAME}\",\n  \"libtriton_jit_install_relative_path\": \"${_flagdnn_hygon_triton_jit_install_directory}/${_flagdnn_hygon_triton_jit_SONAME}\",\n  \"jit_script_install_relative_path\": \"${_flagdnn_hygon_triton_jit_private_script_install_directory}\",\n  \"libtriton_jit_sha256\": \"${_flagdnn_triton_jit_library_sha256}\",\n  \"triton_jit_provenance_sha256\": \"${_flagdnn_triton_jit_provenance_sha256}\",\n  \"python_environment_sha256\": \"${_flagdnn_jit_environment_sha256}\",\n  \"python_environment_helper_sha256\": \"${_flagdnn_python_environment_identity_sha256}\",\n  \"standalone_compile_sha256\": \"${_flagdnn_triton_jit_standalone_sha256}\",\n  \"gen_ssig_sha256\": \"${_flagdnn_triton_jit_gen_ssig_sha256}\",\n  \"jit_scripts_sha256\": \"${_flagdnn_triton_jit_scripts_sha256}\",\n  \"build_identity\": \"${_flagdnn_triton_jit_build_identity}\"\n}\n")
set(_flagdnn_hygon_compiler_environment_file
  "${CMAKE_CURRENT_BINARY_DIR}/flagdnn_hygon_compiler_environment.json")
file(GENERATE
  OUTPUT "${_flagdnn_hygon_compiler_environment_file}"
  CONTENT "${_flagdnn_hygon_compiler_environment}")

flagdnn_add_backend_plugin(hygon INSTALL
  SOURCES
    artifact.cpp
    ${PROJECT_SOURCE_DIR}/backends/autotune_policy.cpp
    backend.cpp
    context.cpp
    engines/engine.cpp
    engines/libtriton_jit.cpp
    error.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/json.cpp
    ${PROJECT_SOURCE_DIR}/src/runtime/sha256.cpp
  LINK_LIBRARIES
    hip::host
    ${_flagdnn_triton_jit_target}
    Python::Python)

target_include_directories(flagdnn_backend_hygon SYSTEM PRIVATE
  ${_flagdnn_triton_jit_include_dir}
  ${_flagdnn_torch_include}
  ${_flagdnn_torch_api_include})

target_compile_definitions(flagdnn_backend_hygon PRIVATE
  BACKEND_HCU=1
  _GLIBCXX_USE_CXX11_ABI=${_flagdnn_torch_cxx11_abi}
  FLAGDNN_HAS_LIBTRITON_JIT=1
  FLAGDNN_LIBTRITON_JIT_PYTHONPATH="${_flagdnn_hygon_python_path}"
  FLAGDNN_LIBTRITON_JIT_BUILD_IDENTITY="${_flagdnn_triton_jit_build_identity}"
  FLAGDNN_LIBTRITON_JIT_PROVENANCE_SHA256="${_flagdnn_triton_jit_provenance_sha256}"
  FLAGDNN_LIBTRITON_JIT_SHA256="${_flagdnn_triton_jit_library_sha256}"
  FLAGDNN_LIBTRITON_JIT_PYTHON_ENVIRONMENT_SHA256="${_flagdnn_jit_environment_sha256}"
  FLAGDNN_LIBTRITON_JIT_PYTHON_ENVIRONMENT_HELPER_SHA256="${_flagdnn_python_environment_identity_sha256}"
  FLAGDNN_LIBTRITON_JIT_STANDALONE_SHA256="${_flagdnn_triton_jit_standalone_sha256}"
  FLAGDNN_LIBTRITON_JIT_GEN_SSIG_SHA256="${_flagdnn_triton_jit_gen_ssig_sha256}"
  FLAGDNN_LIBTRITON_JIT_TORCH_SHA256="${_flagdnn_torch_module_sha256}"
  FLAGDNN_LIBTRITON_JIT_TORCH_C_SHA256="${_flagdnn_torch_c_module_sha256}"
  FLAGDNN_LIBTRITON_JIT_TRITON_SHA256="${_flagdnn_triton_module_sha256}"
  FLAGDNN_LIBTRITON_JIT_YAML_SHA256="${_flagdnn_yaml_module_sha256}")

# The plugin is dlopen'ed and is not part of FlagDNN's public export. Install
# the exact selected HCU JIT runtime and scripts in a plugin-relative private
# tree so the SDK does not retain an absolute dependency on a libtriton_jit
# build tree. Torch and DTK remain environment prerequisites of that runtime.
set_target_properties(flagdnn_backend_hygon PROPERTIES
  BUILD_WITH_INSTALL_RPATH YES
  INSTALL_RPATH "$ORIGIN/flagdnn/hygon"
  INSTALL_RPATH_USE_LINK_PATH NO)
if(UNIX AND NOT APPLE)
  # DT_RPATH makes the private plugin-relative copy win over LD_LIBRARY_PATH. ELF can
  # still reuse an already-loaded object with the same SONAME, so the runtime
  # additionally hashes the mapped JIT image, imported scripts, and active
  # Python/Triton environment before using any generated kernel.
  target_link_options(flagdnn_backend_hygon PRIVATE
    "LINKER:--disable-new-dtags")
endif()
add_custom_command(TARGET flagdnn_backend_hygon POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E make_directory
    "$<TARGET_FILE_DIR:flagdnn_backend_hygon>/flagdnn/hygon"
  COMMAND "${CMAKE_COMMAND}" -E make_directory
    "$<TARGET_FILE_DIR:flagdnn_backend_hygon>/flagdnn/share/triton_jit/scripts"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${_flagdnn_hygon_triton_jit_build_library}"
    "$<TARGET_FILE_DIR:flagdnn_backend_hygon>/flagdnn/hygon/${_flagdnn_hygon_triton_jit_SONAME}"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${_flagdnn_hygon_prepared_standalone_compile}"
    "$<TARGET_FILE_DIR:flagdnn_backend_hygon>/flagdnn/share/triton_jit/scripts/standalone_compile.py"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${_flagdnn_triton_jit_script_dir}/gen_ssig.py"
    "$<TARGET_FILE_DIR:flagdnn_backend_hygon>/flagdnn/share/triton_jit/scripts/gen_ssig.py"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/python_environment_identity.py"
    "$<TARGET_FILE_DIR:flagdnn_backend_hygon>/flagdnn/share/triton_jit/scripts/flagdnn_python_environment_identity.py"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${_flagdnn_hygon_compiler_environment_file}"
    "$<TARGET_FILE_DIR:flagdnn_backend_hygon>/flagdnn_hygon_compiler_environment.json"
  VERBATIM)

install(FILES compiler.py python_environment_identity.py
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/hygon")
install(DIRECTORY dispatch codegen
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/hygon"
  FILES_MATCHING PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE)
install(FILES python_environment_identity.py
  DESTINATION
    "${_flagdnn_hygon_triton_jit_private_script_install_directory}"
  RENAME flagdnn_python_environment_identity.py)
install(FILES "${_flagdnn_hygon_compiler_environment_file}"
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/hygon")
install(DIRECTORY tuning/
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/hygon/tuning"
  FILES_MATCHING PATTERN "*.yaml")
install(DIRECTORY kernels/
  DESTINATION "${FLAGDNN_RESOURCE_INSTALL_DIR}/backends/hygon/kernels"
  FILES_MATCHING
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.py"
    PATTERN "*.json")

# Hygon validation is registered from a sibling directory, so publish only
# immutable resource-contract metadata as target properties instead of
# leaking directory-scope variables into common CMake code.
set_target_properties(flagdnn_backend_hygon PROPERTIES
  FLAGDNN_HYGON_TRITON_JIT_INCLUDE_DIR
    "${_flagdnn_triton_jit_include_dir}"
  FLAGDNN_HYGON_TRITON_JIT_SONAME
    "${_flagdnn_hygon_triton_jit_SONAME}"
  FLAGDNN_HYGON_TRITON_JIT_SHA256
    "${_flagdnn_triton_jit_library_sha256}"
  FLAGDNN_HYGON_TRITON_JIT_PROVENANCE_SHA256
    "${_flagdnn_triton_jit_provenance_sha256}"
  FLAGDNN_HYGON_TRITON_JIT_BUILD_IDENTITY
    "${_flagdnn_triton_jit_build_identity}"
  FLAGDNN_HYGON_TRITON_JIT_STANDALONE_SHA256
    "${_flagdnn_triton_jit_standalone_sha256}"
  FLAGDNN_HYGON_TRITON_JIT_GEN_SSIG_SHA256
    "${_flagdnn_triton_jit_gen_ssig_sha256}"
  FLAGDNN_HYGON_PYTHON_ENVIRONMENT_HELPER_SHA256
    "${_flagdnn_python_environment_identity_sha256}")
