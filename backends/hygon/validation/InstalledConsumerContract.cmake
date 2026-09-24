if(NOT DEFINED SOURCE_ROOT OR NOT IS_DIRECTORY "${SOURCE_ROOT}")
  message(FATAL_ERROR "SOURCE_ROOT is missing")
endif()
if(NOT DEFINED BUILD_ROOT OR NOT EXISTS "${BUILD_ROOT}/CMakeCache.txt")
  message(FATAL_ERROR "BUILD_ROOT is not a configured FlagDNN build")
endif()
if(NOT DEFINED TEST_ROOT OR TEST_ROOT STREQUAL "")
  message(FATAL_ERROR "TEST_ROOT is missing")
endif()
if(NOT DEFINED CODEGEN_PYTHON OR NOT EXISTS "${CODEGEN_PYTHON}")
  message(FATAL_ERROR "CODEGEN_PYTHON is missing")
endif()
foreach(required_variable IN ITEMS
    BUILD_PLUGIN
    PLUGIN_SONAME
    JIT_SONAME
    JIT_SHA256
    JIT_PROVENANCE_SHA256
    JIT_BUILD_IDENTITY
    STANDALONE_SHA256
    GEN_SSIG_SHA256
    ENVIRONMENT_HELPER_SHA256)
  if(NOT DEFINED ${required_variable} OR
     "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} is missing")
  endif()
endforeach()
if(NOT EXISTS "${BUILD_PLUGIN}")
  message(FATAL_ERROR "BUILD_PLUGIN does not exist: ${BUILD_PLUGIN}")
endif()
if(NOT JIT_SONAME MATCHES "^libtriton_jit\\.so(\\.[0-9]+)*$")
  message(FATAL_ERROR "JIT_SONAME is invalid: ${JIT_SONAME}")
endif()
foreach(expected_hash IN ITEMS
    JIT_SHA256 JIT_PROVENANCE_SHA256 STANDALONE_SHA256 GEN_SSIG_SHA256
    ENVIRONMENT_HELPER_SHA256)
  string(LENGTH "${${expected_hash}}" expected_hash_length)
  if(NOT expected_hash_length EQUAL 64 OR
     NOT "${${expected_hash}}" MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR "${expected_hash} is not a SHA-256 digest")
  endif()
endforeach()
unset(expected_hash_length)
string(LENGTH "${JIT_BUILD_IDENTITY}" jit_build_identity_length)
if(NOT jit_build_identity_length EQUAL 50 OR
   NOT JIT_BUILD_IDENTITY MATCHES
       "^[0-9a-f]+-[0-9a-f]+-[0-9a-f]+$")
  message(FATAL_ERROR
    "JIT_BUILD_IDENTITY is invalid: ${JIT_BUILD_IDENTITY}")
endif()
if(NOT DEFINED INSTALL_CONFIG OR INSTALL_CONFIG STREQUAL "")
  set(INSTALL_CONFIG Release)
endif()

# Every child that could discover FlagDNN runtime/compiler resources starts
# without inherited source-tree, build-tree, legacy TritonJIT, or compiler
# overrides. Individual test properties may then add only the installed SDK
# backend directory and the requested Python interpreter.
set(flagdnn_clean_environment
  "${CMAKE_COMMAND}" -E env
  --unset=FLAGDNN_BACKEND
  --unset=FLAGDNN_BACKEND_PATH
  --unset=FLAGDNN_BACKEND_ROOT
  --unset=FLAGDNN_KERNEL_SOURCE_ROOT
  --unset=FLAGDNN_TUNING_ROOT
  --unset=FLAGDNN_HYGON_TRITON_JIT_ROOT
  --unset=FLAGDNN_HYGON_TRITON_JIT_DIR
  --unset=FLAGDNN_HYGON_TRITON_JIT_LIBRARY
  --unset=FLAGDNN_HYGON_TRITON_JIT_INCLUDE_DIR
  --unset=FLAGDNN_HYGON_TRITON_JIT_SCRIPT_DIR
  --unset=FLAGDNN_HYGON_COMPILER_ENVIRONMENT
  --unset=FLAGDNN_HYGON_PYTHONPATH
  --unset=FLAGDNN_TRITON_JIT_LIBRARY
  --unset=FLAGDNN_TRITON_JIT_INCLUDE_DIR
  --unset=FLAGDNN_TRITON_JIT_SCRIPT_DIR
  --unset=TritonJIT_DIR
  --unset=TritonJIT_ROOT
  --unset=TRITONJIT_ROOT
  --unset=FLAGDNN_CODEGEN_COMPILER
  --unset=FLAGDNN_CODEGEN_PYTHON
  --unset=FLAGDNN_COMPILER
  --unset=FLAGDNN_COMPILER_EXECUTABLE
  --unset=FLAGDNN_COMPILER_TIMEOUT_SECONDS
  --unset=FLAGDNN_EXECUTION_ENGINE
  --unset=FLAGDNN_CACHE_DIRECTORY
  --unset=PYTHONPATH
  --unset=PYTHONHOME
  --)

function(flagdnn_run_step description)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "${description} failed (${result})\nstdout:\n${output}\nstderr:\n${error}")
  endif()
  if(NOT output STREQUAL "")
    message(STATUS "${description}:\n${output}")
  endif()
endfunction()

function(flagdnn_run_clean_step description)
  flagdnn_run_step("${description}" ${flagdnn_clean_environment} ${ARGN})
endfunction()

function(flagdnn_require_sha256 path expected)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "installed SDK resource is missing: ${path}")
  endif()
  file(SHA256 "${path}" actual)
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR
      "installed SDK resource hash mismatch: ${path}\n"
      "expected=${expected}\nactual=${actual}")
  endif()
endfunction()

function(flagdnn_require_elf_rpath path expected)
  find_program(flagdnn_readelf NAMES readelf llvm-readelf REQUIRED)
  execute_process(
    COMMAND "${flagdnn_readelf}" -d "${path}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE dynamic_section
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "cannot read ELF dynamic section for ${path}: ${error}")
  endif()
  string(REGEX MATCH
    "\\((RPATH|RUNPATH)\\)[^\n]*\\[([^]]*)\\]"
    rpath_record "${dynamic_section}")
  if(rpath_record STREQUAL "")
    message(FATAL_ERROR "${path}: ELF has no RPATH/RUNPATH")
  endif()
  set(actual "${CMAKE_MATCH_2}")
  if(NOT actual STREQUAL expected)
    message(FATAL_ERROR
      "${path}: unexpected ELF runtime path: '${actual}', "
      "expected '${expected}'")
  endif()
  if(dynamic_section MATCHES "\\(RUNPATH\\)")
    message(FATAL_ERROR "${path}: expected fail-closed DT_RPATH, got RUNPATH")
  endif()
endfunction()

function(flagdnn_require_elf_rpath_without_empty_component path)
  find_program(flagdnn_readelf NAMES readelf llvm-readelf REQUIRED)
  execute_process(
    COMMAND "${flagdnn_readelf}" -d "${path}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE dynamic_section
    ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR
      "cannot read ELF dynamic section for ${path}: ${error}")
  endif()
  string(REGEX MATCH
    "\\((RPATH|RUNPATH)\\)[^\n]*\\[([^]]*)\\]"
    rpath_record "${dynamic_section}")
  if(rpath_record STREQUAL "")
    # No search path has no empty component. The private staging step also
    # removes the tag when the upstream library contains only colon padding.
    return()
  endif()
  set(actual "${CMAKE_MATCH_2}")
  if(actual MATCHES "(^|:)(:|$)")
    message(FATAL_ERROR
      "${path}: ELF runtime path contains an empty component: '${actual}'")
  endif()
endfunction()

flagdnn_require_elf_rpath(
  "${BUILD_PLUGIN}" "$ORIGIN/flagdnn/hygon")

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")
set(sdk "${TEST_ROOT}/sdk")
set(consumer_build "${TEST_ROOT}/consumer-build")

flagdnn_run_step(
  "install isolated FlagDNN SDK"
  "${CMAKE_COMMAND}" --install "${BUILD_ROOT}"
  --prefix "${sdk}" --config "${INSTALL_CONFIG}")

if(EXISTS "${sdk}/lib/cmake/FlagDNN/FlagDNNConfig.cmake")
  set(flagdnn_dir "${sdk}/lib/cmake/FlagDNN")
elseif(EXISTS "${sdk}/lib64/cmake/FlagDNN/FlagDNNConfig.cmake")
  set(flagdnn_dir "${sdk}/lib64/cmake/FlagDNN")
else()
  message(FATAL_ERROR "installed SDK has no FlagDNNConfig.cmake")
endif()
get_filename_component(sdk_library_directory
  "${flagdnn_dir}/../.." REALPATH)
set(installed_plugin "${sdk_library_directory}/${PLUGIN_SONAME}")
set(installed_jit
  "${sdk_library_directory}/flagdnn/hygon/${JIT_SONAME}")
set(private_script_directory
  "${sdk_library_directory}/flagdnn/share/triton_jit/scripts")
set(installed_compiler
  "${sdk}/share/flagdnn/compiler/flagdnn_codegen/main.py")
set(installed_provider
  "${sdk}/share/flagdnn/backends/hygon/compiler.py")
set(installed_environment
  "${sdk}/share/flagdnn/backends/hygon/flagdnn_hygon_compiler_environment.json")

foreach(required IN ITEMS
    "${installed_compiler}"
    "${installed_provider}"
    "${sdk}/share/flagdnn/backends/hygon/kernels/binary_minmax.py"
    "${installed_environment}"
    "${installed_plugin}"
    "${installed_jit}"
    "${private_script_directory}/standalone_compile.py"
    "${private_script_directory}/gen_ssig.py"
    "${private_script_directory}/flagdnn_python_environment_identity.py")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR "installed SDK resource is missing: ${required}")
  endif()
endforeach()

flagdnn_require_elf_rpath(
  "${installed_plugin}" "$ORIGIN/flagdnn/hygon")
flagdnn_require_elf_rpath_without_empty_component("${installed_jit}")
flagdnn_require_sha256("${installed_jit}" "${JIT_SHA256}")
foreach(script_directory IN ITEMS "${private_script_directory}")
  flagdnn_require_sha256(
    "${script_directory}/standalone_compile.py" "${STANDALONE_SHA256}")
  flagdnn_require_sha256(
    "${script_directory}/gen_ssig.py" "${GEN_SSIG_SHA256}")
  flagdnn_require_sha256(
    "${script_directory}/flagdnn_python_environment_identity.py"
    "${ENVIRONMENT_HELPER_SHA256}")
endforeach()

file(READ "${installed_environment}" compiler_environment)
string(JSON environment_schema ERROR_VARIABLE json_error
  GET "${compiler_environment}" schema_version)
if(json_error OR NOT environment_schema EQUAL 3)
  message(FATAL_ERROR
    "installed compiler environment has the wrong schema")
endif()
string(JSON environment_soname ERROR_VARIABLE json_error
  GET "${compiler_environment}" libtriton_jit_soname)
if(json_error OR NOT environment_soname STREQUAL JIT_SONAME)
  message(FATAL_ERROR
    "installed compiler environment has the wrong private JIT SONAME")
endif()
file(RELATIVE_PATH expected_jit_relative "${sdk}" "${installed_jit}")
file(RELATIVE_PATH expected_scripts_relative
  "${sdk}" "${private_script_directory}")
string(JSON environment_jit_relative ERROR_VARIABLE json_error
  GET "${compiler_environment}" libtriton_jit_install_relative_path)
if(json_error OR NOT environment_jit_relative STREQUAL expected_jit_relative)
  message(FATAL_ERROR
    "installed compiler environment has the wrong private JIT path")
endif()
string(JSON environment_scripts_relative ERROR_VARIABLE json_error
  GET "${compiler_environment}" jit_script_install_relative_path)
if(json_error OR
   NOT environment_scripts_relative STREQUAL expected_scripts_relative)
  message(FATAL_ERROR
    "installed compiler environment has the wrong private script path")
endif()
string(JSON environment_jit_sha256 ERROR_VARIABLE json_error
  GET "${compiler_environment}" libtriton_jit_sha256)
if(json_error OR NOT environment_jit_sha256 STREQUAL JIT_SHA256)
  message(FATAL_ERROR
    "installed compiler environment has the wrong private JIT hash")
endif()
string(JSON environment_provenance ERROR_VARIABLE json_error
  GET "${compiler_environment}" triton_jit_provenance_sha256)
if(json_error OR
   NOT environment_provenance STREQUAL JIT_PROVENANCE_SHA256)
  message(FATAL_ERROR
    "installed compiler environment has the wrong JIT provenance")
endif()
string(JSON environment_build_identity ERROR_VARIABLE json_error
  GET "${compiler_environment}" build_identity)
if(json_error OR
   NOT environment_build_identity STREQUAL JIT_BUILD_IDENTITY)
  message(FATAL_ERROR
    "installed compiler environment has the wrong build identity")
endif()
string(JSON environment_python_sha256 ERROR_VARIABLE json_error
  GET "${compiler_environment}" python_environment_sha256)
string(JSON environment_scripts_sha256 ERROR_VARIABLE scripts_json_error
  GET "${compiler_environment}" jit_scripts_sha256)
foreach(environment_hash IN ITEMS
    environment_python_sha256 environment_scripts_sha256)
  string(LENGTH "${${environment_hash}}" environment_hash_length)
  if(json_error OR scripts_json_error OR
     NOT environment_hash_length EQUAL 64 OR
     NOT "${${environment_hash}}" MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR
      "installed compiler environment has an invalid identity hash")
  endif()
endforeach()
string(SUBSTRING "${environment_provenance}" 0 16 provenance_short)
string(SUBSTRING "${environment_python_sha256}" 0 16 python_short)
string(SUBSTRING "${environment_scripts_sha256}" 0 16 scripts_short)
set(expected_build_identity
  "${provenance_short}-${python_short}-${scripts_short}")
if(NOT environment_build_identity STREQUAL expected_build_identity)
  message(FATAL_ERROR
    "installed build identity does not consume JIT provenance")
endif()

# Exercise the installed compiler entry and provider loader without any
# override. The identity manifest names the actual compiler/provider and every
# JIT resource selected by the Hygon provider.
set(identity_manifest "${TEST_ROOT}/installed-hygon.identity")
flagdnn_run_clean_step(
  "identify installed Hygon compiler resources"
  "${CODEGEN_PYTHON}" "${installed_compiler}"
  --identify
  --backend hygon
  --target gfx936
  --execution-engine libtriton_jit
  --identity-output "${identity_manifest}"
  --quiet)
file(STRINGS "${identity_manifest}" identity_lines LIMIT_COUNT 2)
list(LENGTH identity_lines identity_line_count)
if(NOT identity_line_count EQUAL 2)
  message(FATAL_ERROR "installed compiler identity manifest is malformed")
endif()
list(GET identity_lines 1 identity_metadata)
string(JSON dependencies_complete ERROR_VARIABLE json_error
  GET "${identity_metadata}" dependencies_complete)
if(json_error OR NOT dependencies_complete)
  message(FATAL_ERROR
    "installed Hygon compiler did not report complete dependencies")
endif()

function(flagdnn_identity_requires_path expected)
  file(REAL_PATH "${expected}" expected_real)
  string(JSON dependency_count ERROR_VARIABLE dependency_error
    LENGTH "${identity_metadata}" files)
  if(dependency_error OR dependency_count LESS 1)
    message(FATAL_ERROR "installed compiler identity has no dependencies")
  endif()
  math(EXPR dependency_last "${dependency_count} - 1")
  foreach(index RANGE 0 ${dependency_last})
    string(JSON dependency GET "${identity_metadata}" files ${index})
    if(EXISTS "${dependency}")
      file(REAL_PATH "${dependency}" dependency_real)
      if(dependency_real STREQUAL expected_real)
        return()
      endif()
    endif()
  endforeach()
  message(FATAL_ERROR
    "installed compiler identity did not resolve SDK resource: ${expected}")
endfunction()

foreach(identity_resource IN ITEMS
    "${installed_compiler}"
    "${installed_provider}"
    "${installed_environment}"
    "${installed_jit}"
    "${private_script_directory}/standalone_compile.py"
    "${private_script_directory}/gen_ssig.py"
    "${private_script_directory}/flagdnn_python_environment_identity.py")
  flagdnn_identity_requires_path("${identity_resource}")
endforeach()

foreach(generic_helper IN ITEMS
    "${sdk}/share/triton_jit/scripts/standalone_compile.py"
    "${sdk}/share/triton_jit/scripts/gen_ssig.py"
    "${sdk}/share/triton_jit/scripts/flagdnn_python_environment_identity.py")
  if(EXISTS "${generic_helper}")
    message(FATAL_ERROR
      "FlagDNN must not install a generic TritonJIT helper: ${generic_helper}")
  endif()
endforeach()

flagdnn_run_clean_step(
  "configure installed Hygon consumer"
  "${CMAKE_COMMAND}"
  -S "${SOURCE_ROOT}/tests/core/installed_consumer"
  -B "${consumer_build}"
  "-DFlagDNN_DIR=${flagdnn_dir}"
  "-DCMAKE_BUILD_TYPE=Release"
  "-DFLAGDNN_INSTALLED_HYGON_ADD=ON"
  "-DFLAGDNN_INSTALLED_CODEGEN_PYTHON=${CODEGEN_PYTHON}"
  "-DFLAGDNN_INSTALLED_HYGON_SDK_ROOT=${sdk}"
  "-DFLAGDNN_INSTALLED_HYGON_JIT_LIBRARY=${installed_jit}"
  "-DFLAGDNN_INSTALLED_HYGON_JIT_SCRIPT_DIR=${private_script_directory}")

flagdnn_run_clean_step(
  "build installed Hygon consumer"
  "${CMAKE_COMMAND}" --build "${consumer_build}" --parallel 8)

flagdnn_run_clean_step(
  "run installed Hygon consumer"
  "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build}"
  --output-on-failure -V)

message(STATUS "Installed Hygon SDK consumer contract verified")
