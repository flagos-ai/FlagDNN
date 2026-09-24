# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED SOURCE_ROOT OR NOT DEFINED TEST_ROOT)
  message(FATAL_ERROR "SOURCE_ROOT and TEST_ROOT are required")
endif()

include("${SOURCE_ROOT}/backends/hygon/cmake/ResolveTritonJIT.cmake")

# Model the upstream HCU headers independently of the resolver's requirements.
# In particular, hcu_error.h is a local extension, not an upstream dependency.
set(_upstream_hcu_headers
  triton_jit/backend_config.h
  triton_jit/backend_policy.h
  triton_jit/backends/hcu_backend.h
  triton_jit/backends/npu_types.h
  triton_jit/jit_function_arg.h
  triton_jit/jit_utils.h
  triton_jit/kernel_metadata.h
  triton_jit/triton_jit_function.h
  triton_jit/triton_kernel.h)

function(_make_build_layout root backend marker)
  file(MAKE_DIRECTORY
    "${root}/build/src"
    "${root}/include/triton_jit/backends"
    "${root}/scripts")
  file(WRITE "${root}/build/TritonJITConfig.cmake"
    "set(TritonJIT_BACKEND \"${backend}\")\n")
  file(WRITE "${root}/build/src/libtriton_jit.so" "library-${marker}\n")
  file(WRITE "${root}/build/TritonJITTargets.cmake"
    "set_target_properties(jit PROPERTIES IMPORTED_LOCATION_NOCONFIG \"${root}/build/src/libtriton_jit.so\" IMPORTED_SONAME_NOCONFIG \"libtriton_jit.so\")\n")
  foreach(_header IN LISTS _upstream_hcu_headers)
    get_filename_component(_header_directory
      "${root}/include/${_header}" DIRECTORY)
    file(MAKE_DIRECTORY "${_header_directory}")
    file(WRITE "${root}/include/${_header}"
      "${_header}-${marker}\n")
  endforeach()
  file(WRITE "${root}/scripts/standalone_compile.py"
    "# standalone-${marker}\n")
  file(WRITE "${root}/scripts/gen_ssig.py" "# signature-${marker}\n")
endfunction()

function(_make_install_layout root backend marker)
  file(MAKE_DIRECTORY
    "${root}/lib/cmake/TritonJIT"
    "${root}/include/triton_jit/backends"
    "${root}/share/triton_jit/scripts")
  file(WRITE "${root}/lib/cmake/TritonJIT/TritonJITConfig.cmake"
    "set(TritonJIT_BACKEND \"${backend}\")\n")
  file(WRITE "${root}/lib/cmake/TritonJIT/TritonJITTargets-release.cmake"
    "set_target_properties(jit PROPERTIES IMPORTED_LOCATION_RELEASE \"\${_IMPORT_PREFIX}/lib/libtriton_jit.so\" IMPORTED_SONAME_RELEASE \"libtriton_jit.so\")\n")
  file(WRITE "${root}/lib/libtriton_jit.so" "library-${marker}\n")
  foreach(_header IN LISTS _upstream_hcu_headers)
    get_filename_component(_header_directory
      "${root}/include/${_header}" DIRECTORY)
    file(MAKE_DIRECTORY "${_header_directory}")
    file(WRITE "${root}/include/${_header}"
      "${_header}-${marker}\n")
  endforeach()
  file(WRITE "${root}/share/triton_jit/scripts/standalone_compile.py"
    "# standalone-${marker}\n")
  file(WRITE "${root}/share/triton_jit/scripts/gen_ssig.py"
    "# signature-${marker}\n")
endfunction()

function(_make_install_lib64_layout root backend marker)
  file(MAKE_DIRECTORY
    "${root}/lib64/cmake/TritonJIT"
    "${root}/include/triton_jit/backends"
    "${root}/share/triton_jit/scripts")
  file(WRITE "${root}/lib64/cmake/TritonJIT/TritonJITConfig.cmake"
    "set(TritonJIT_BACKEND \"${backend}\")\n")
  file(WRITE
    "${root}/lib64/cmake/TritonJIT/TritonJITTargets-release.cmake"
    "set_target_properties(jit PROPERTIES IMPORTED_LOCATION_RELEASE \"\${_IMPORT_PREFIX}/lib64/libtriton_jit.so\" IMPORTED_SONAME_RELEASE \"libtriton_jit.so\")\n")
  file(WRITE "${root}/lib64/libtriton_jit.so" "library-${marker}\n")
  foreach(_header IN LISTS _upstream_hcu_headers)
    get_filename_component(_header_directory
      "${root}/include/${_header}" DIRECTORY)
    file(MAKE_DIRECTORY "${_header_directory}")
    file(WRITE "${root}/include/${_header}"
      "${_header}-${marker}\n")
  endforeach()
  file(WRITE "${root}/share/triton_jit/scripts/standalone_compile.py"
    "# standalone-${marker}\n")
  file(WRITE "${root}/share/triton_jit/scripts/gen_ssig.py"
    "# signature-${marker}\n")
endfunction()

if(DEFINED CONTRACT_CHILD_MODE)
  if(CONTRACT_CHILD_MODE STREQUAL "mixed_library")
    flagdnn_hygon_resolve_triton_jit(_selected
      CONFIG_DIR "${TEST_ROOT}/a/build"
      LIBRARY "${TEST_ROOT}/b/build/src/libtriton_jit.so")
  elseif(CONTRACT_CHILD_MODE STREQUAL "mixed_scripts")
    flagdnn_hygon_resolve_triton_jit(_selected
      CONFIG_DIR "${TEST_ROOT}/a/build"
      SCRIPT_DIR "${TEST_ROOT}/b/scripts")
  elseif(CONTRACT_CHILD_MODE STREQUAL "wrong_backend")
    flagdnn_hygon_resolve_triton_jit(_selected
      CONFIG_DIR "${TEST_ROOT}/cuda/build")
  elseif(CONTRACT_CHILD_MODE STREQUAL "root_mismatch")
    flagdnn_hygon_resolve_triton_jit(_selected
      CONFIG_DIR "${TEST_ROOT}/a/build"
      ROOT "${TEST_ROOT}/b")
  elseif(CONTRACT_CHILD_MODE STREQUAL "missing_provenance_header")
    flagdnn_hygon_resolve_triton_jit(_selected
      ROOT "${TEST_ROOT}/incomplete")
  else()
    message(FATAL_ERROR "unknown CONTRACT_CHILD_MODE")
  endif()
  message(FATAL_ERROR "negative resolver contract unexpectedly succeeded")
endif()

file(REMOVE_RECURSE "${TEST_ROOT}")
_make_build_layout("${TEST_ROOT}/a" HCU a)
_make_build_layout("${TEST_ROOT}/b" HCU b)
_make_build_layout("${TEST_ROOT}/cuda" CUDA cuda)
_make_install_layout("${TEST_ROOT}/installed" HCU installed)
_make_install_lib64_layout("${TEST_ROOT}/installed64" HCU installed64)
_make_build_layout("${TEST_ROOT}/versioned" HCU versioned)
_make_build_layout("${TEST_ROOT}/incomplete" HCU incomplete)
file(RENAME
  "${TEST_ROOT}/versioned/build/src/libtriton_jit.so"
  "${TEST_ROOT}/versioned/build/src/libtriton_jit.so.7")
file(CREATE_LINK
  "${TEST_ROOT}/versioned/build/src/libtriton_jit.so.7"
  "${TEST_ROOT}/versioned/build/src/libtriton_jit.so" SYMBOLIC)
file(WRITE "${TEST_ROOT}/versioned/build/TritonJITTargets.cmake"
  "set_target_properties(jit PROPERTIES IMPORTED_LOCATION_NOCONFIG \"${TEST_ROOT}/versioned/build/src/libtriton_jit.so\" IMPORTED_SONAME_NOCONFIG \"libtriton_jit.so.7\")\n")

# A global CUDA package selection may coexist in the parent project. The Hygon
# resolver must neither read nor mutate its legacy/global CMake variable.
set(TritonJIT_DIR "/sentinel/cuda-triton-jit")
flagdnn_hygon_resolve_triton_jit(_selected ROOT "${TEST_ROOT}/a")
if(NOT _selected_CONFIG_DIR STREQUAL "${TEST_ROOT}/a/build" OR
   NOT _selected_LIBRARY STREQUAL
       "${TEST_ROOT}/a/build/src/libtriton_jit.so" OR
   NOT _selected_INCLUDE_DIR STREQUAL "${TEST_ROOT}/a/include" OR
   NOT _selected_SCRIPT_DIR STREQUAL "${TEST_ROOT}/a/scripts")
  message(FATAL_ERROR "coherent build-tree derivation failed")
endif()
if(NOT TritonJIT_DIR STREQUAL "/sentinel/cuda-triton-jit")
  message(FATAL_ERROR "Hygon resolver mutated the global TritonJIT_DIR")
endif()
if(NOT _selected_PROVENANCE_SHA256 MATCHES "^[0-9a-f]+$")
  message(FATAL_ERROR "resolver did not produce a provenance fingerprint")
endif()
foreach(_required_header IN LISTS _upstream_hcu_headers)
  list(FIND _selected_PROVENANCE_HEADER_FILES
    "${TEST_ROOT}/a/include/${_required_header}" _header_index)
  if(_header_index EQUAL -1)
    message(FATAL_ERROR
      "resolver provenance omits HCU header: ${_required_header}")
  endif()
endforeach()

string(SHA256 _contract_python_sha256 "contract-python-environment")
string(SHA256 _contract_scripts_sha256 "contract-prepared-scripts")
flagdnn_hygon_compose_triton_jit_build_identity(
  _baseline_build_identity
  PROVENANCE_SHA256 "${_selected_PROVENANCE_SHA256}"
  PYTHON_ENVIRONMENT_SHA256 "${_contract_python_sha256}"
  JIT_SCRIPTS_SHA256 "${_contract_scripts_sha256}")
string(SUBSTRING "${_selected_PROVENANCE_SHA256}" 0 16
  _baseline_provenance_short)
if(NOT _baseline_build_identity MATCHES
   "^${_baseline_provenance_short}-")
  message(FATAL_ERROR "build identity does not consume JIT provenance")
endif()

function(_expect_provenance_mutation input_file)
  file(READ "${input_file}" _original)
  file(APPEND "${input_file}" "\n# provenance-mutation\n")
  flagdnn_hygon_resolve_triton_jit(_mutated ROOT "${TEST_ROOT}/a")
  if(_mutated_PROVENANCE_SHA256 STREQUAL _selected_PROVENANCE_SHA256)
    message(FATAL_ERROR
      "TritonJIT provenance ignored input mutation: ${input_file}")
  endif()
  flagdnn_hygon_compose_triton_jit_build_identity(
    _mutated_build_identity
    PROVENANCE_SHA256 "${_mutated_PROVENANCE_SHA256}"
    PYTHON_ENVIRONMENT_SHA256 "${_contract_python_sha256}"
    JIT_SCRIPTS_SHA256 "${_contract_scripts_sha256}")
  if(_mutated_build_identity STREQUAL _baseline_build_identity)
    message(FATAL_ERROR
      "Hygon build identity ignored provenance mutation: ${input_file}")
  endif()
  file(WRITE "${input_file}" "${_original}")
endfunction()

foreach(_provenance_input IN LISTS _selected_PROVENANCE_INPUT_FILES)
  _expect_provenance_mutation("${_provenance_input}")
endforeach()

flagdnn_hygon_resolve_triton_jit(_installed ROOT "${TEST_ROOT}/installed")
if(NOT _installed_LAYOUT STREQUAL "install_lib" OR
   NOT _installed_CONFIG_DIR STREQUAL
       "${TEST_ROOT}/installed/lib/cmake/TritonJIT" OR
   NOT _installed_LIBRARY STREQUAL
       "${TEST_ROOT}/installed/lib/libtriton_jit.so" OR
   NOT _installed_SCRIPT_DIR STREQUAL
       "${TEST_ROOT}/installed/share/triton_jit/scripts")
  message(FATAL_ERROR "coherent installed-tree derivation failed")
endif()

flagdnn_hygon_resolve_triton_jit(_installed64
  ROOT "${TEST_ROOT}/installed64")
if(NOT _installed64_LAYOUT STREQUAL "install_lib64" OR
   NOT _installed64_CONFIG_DIR STREQUAL
       "${TEST_ROOT}/installed64/lib64/cmake/TritonJIT" OR
   NOT _installed64_LIBRARY STREQUAL
       "${TEST_ROOT}/installed64/lib64/libtriton_jit.so")
  message(FATAL_ERROR "coherent lib64 installed-tree derivation failed")
endif()

flagdnn_hygon_resolve_triton_jit(_versioned ROOT "${TEST_ROOT}/versioned")
if(NOT _versioned_LIBRARY STREQUAL
       "${TEST_ROOT}/versioned/build/src/libtriton_jit.so.7" OR
   NOT _versioned_SONAME STREQUAL "libtriton_jit.so.7")
  message(FATAL_ERROR "versioned/symlink library derivation failed")
endif()

function(_expect_failure mode expected)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DSOURCE_ROOT=${SOURCE_ROOT}"
      "-DTEST_ROOT=${TEST_ROOT}"
      "-DCONTRACT_CHILD_MODE=${mode}"
      -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  if(_result EQUAL 0)
    message(FATAL_ERROR "negative resolver contract ${mode} passed")
  endif()
  set(_output "${_stdout}\n${_stderr}")
  if(NOT _output MATCHES "${expected}")
    message(FATAL_ERROR
      "negative resolver contract ${mode} produced an unexpected error: "
      "${_output}")
  endif()
endfunction()

_expect_failure(mixed_library "overrides do not belong")
_expect_failure(mixed_scripts "overrides do not belong")
_expect_failure(wrong_backend "requires an HCU TritonJITConfig.cmake")
_expect_failure(root_mismatch "is outside the selected")
# Every upstream public header is required, even though local extensions are not.
foreach(_required_header IN LISTS _upstream_hcu_headers)
  set(_header_path "${TEST_ROOT}/incomplete/include/${_required_header}")
  file(READ "${_header_path}" _header_contents)
  file(REMOVE "${_header_path}")
  _expect_failure(missing_provenance_header "no complete, coherent")
  file(WRITE "${_header_path}" "${_header_contents}")
endforeach()

# An unmodified upstream build may have CMake's colon padding in RUNPATH.
# Exercise real ELF staging so source integrity and the loaded bytes are tested.
include("${SOURCE_ROOT}/backends/hygon/cmake/StageTritonJIT.cmake")
find_program(_contract_cxx NAMES c++ g++ clang++ REQUIRED)
find_program(_contract_readelf NAMES readelf llvm-readelf REQUIRED)
file(WRITE "${TEST_ROOT}/runtime.cpp" "extern \"C\" int contract() { return 0; }\n")
execute_process(
  COMMAND "${_contract_cxx}" -shared -fPIC
    "${TEST_ROOT}/runtime.cpp"
    "-Wl,-rpath,:$ORIGIN/deps::/existing/runtime/path::"
    -o "${TEST_ROOT}/upstream.so"
  RESULT_VARIABLE _compile_result
  ERROR_VARIABLE _compile_error)
if(NOT _compile_result EQUAL 0)
  message(FATAL_ERROR "cannot compile staging contract: ${_compile_error}")
endif()
file(SHA256 "${TEST_ROOT}/upstream.so" _upstream_before)
flagdnn_hygon_stage_triton_jit(
  "${TEST_ROOT}/upstream.so" "${TEST_ROOT}/private/runtime.so")
file(SHA256 "${TEST_ROOT}/upstream.so" _upstream_after)
file(SHA256 "${TEST_ROOT}/private/runtime.so" _staged_sha256)
if(NOT _upstream_before STREQUAL _upstream_after OR
   _upstream_before STREQUAL _staged_sha256)
  message(FATAL_ERROR "private staging did not isolate the upstream library")
endif()
execute_process(
  COMMAND "${_contract_readelf}" -d "${TEST_ROOT}/private/runtime.so"
  RESULT_VARIABLE _readelf_result
  OUTPUT_VARIABLE _staged_dynamic)
string(REGEX MATCH "\\((RPATH|RUNPATH)\\)[^\n]*\\[([^]]*)\\]"
  _staged_record "${_staged_dynamic}")
if(NOT _readelf_result EQUAL 0 OR _staged_record STREQUAL "" OR
   NOT CMAKE_MATCH_2 STREQUAL "$ORIGIN/deps:/existing/runtime/path")
  message(FATAL_ERROR "private staging did not preserve nonempty runtime paths")
endif()
# A library whose runtime path is already clean must retain identical bytes.
flagdnn_hygon_stage_triton_jit(
  "${TEST_ROOT}/private/runtime.so" "${TEST_ROOT}/restaged/runtime.so")
file(SHA256 "${TEST_ROOT}/restaged/runtime.so" _restaged_sha256)
if(NOT _staged_sha256 STREQUAL _restaged_sha256)
  message(FATAL_ERROR "private staging changed an already clean library")
endif()

# Removing an entirely empty search path is valid, as is an ELF without one.
foreach(_case IN ITEMS all_empty absent)
  set(_rpath_option)
  if(_case STREQUAL "all_empty")
    set(_rpath_option "-Wl,-rpath,::::")
  endif()
  set(_source "${TEST_ROOT}/${_case}.so")
  set(_staged "${TEST_ROOT}/private/${_case}.so")
  set(_restaged "${TEST_ROOT}/restaged/${_case}.so")
  execute_process(
    COMMAND "${_contract_cxx}" -shared -fPIC
      "${TEST_ROOT}/runtime.cpp" ${_rpath_option} -o "${_source}"
    RESULT_VARIABLE _compile_result
    ERROR_VARIABLE _compile_error)
  if(NOT _compile_result EQUAL 0)
    message(FATAL_ERROR "cannot compile ${_case} staging contract: ${_compile_error}")
  endif()
  file(SHA256 "${_source}" _source_before)
  flagdnn_hygon_stage_triton_jit("${_source}" "${_staged}")
  file(SHA256 "${_source}" _source_after)
  file(SHA256 "${_staged}" _staged_sha256)
  if(NOT _source_before STREQUAL _source_after)
    message(FATAL_ERROR "${_case} staging modified the upstream library")
  endif()
  if(_case STREQUAL "absent" AND
     NOT _source_before STREQUAL _staged_sha256)
    message(FATAL_ERROR "staging changed a library without a runtime path")
  endif()
  execute_process(
    COMMAND "${_contract_readelf}" -d "${_staged}"
    RESULT_VARIABLE _readelf_result
    OUTPUT_VARIABLE _staged_dynamic)
  if(NOT _readelf_result EQUAL 0 OR
     _staged_dynamic MATCHES "\\((RPATH|RUNPATH)\\)")
    message(FATAL_ERROR "${_case} staging left an empty runtime search path")
  endif()
  flagdnn_hygon_stage_triton_jit("${_staged}" "${_restaged}")
  file(SHA256 "${_restaged}" _restaged_sha256)
  if(NOT _staged_sha256 STREQUAL _restaged_sha256)
    message(FATAL_ERROR "${_case} restaging changed the normalized library")
  endif()
endforeach()

file(READ "${SOURCE_ROOT}/backends/hygon/CMakeLists.txt" _hygon_cmake)
if(NOT _hygon_cmake MATCHES "same-SONAME CUDA/HCU JIT runtimes" OR
   NOT _hygon_cmake MATCHES "Use separate processes")
  message(FATAL_ERROR
    "multi-backend same-process JIT limitation is not stated fail-closed")
endif()
foreach(_required_fragment IN ITEMS
    "CMAKE_INSTALL_LIBDIR}/flagdnn/hygon"
    "CMAKE_INSTALL_LIBDIR}/flagdnn/share/triton_jit/scripts"
    "BUILD_WITH_INSTALL_RPATH YES"
    "INSTALL_RPATH \"$ORIGIN/flagdnn/hygon\""
    "libtriton_jit_install_relative_path"
    "jit_script_install_relative_path"
    "_flagdnn_hygon_triton_jit_PROVENANCE_INPUT_FILES"
    "flagdnn_hygon_compose_triton_jit_build_identity("
    "PROVENANCE_SHA256"
    "triton_jit_provenance_sha256"
    "FLAGDNN_HYGON_TRITON_JIT_PROVENANCE_SHA256"
    "FLAGDNN_HYGON_TRITON_JIT_BUILD_IDENTITY")
  string(FIND "${_hygon_cmake}" "${_required_fragment}"
    _required_fragment_index)
  if(_required_fragment_index EQUAL -1)
    message(FATAL_ERROR
      "Hygon private JIT install/RPATH contract is missing: "
      "${_required_fragment}")
  endif()
endforeach()
foreach(_forbidden IN ITEMS
    "\\$\\{TritonJIT_DIR\\}"
    "\\$ENV\\{TritonJIT_DIR\\}"
    "set\\(TritonJIT_DIR"
    "FLAGDNN_TRITON_JIT_LIBRARY"
    "FLAGDNN_TRITON_JIT_INCLUDE_DIR"
    "FLAGDNN_TRITON_JIT_SCRIPT_DIR")
  if(_hygon_cmake MATCHES "${_forbidden}")
    message(FATAL_ERROR
      "Hygon CMake leaked a global/legacy TritonJIT selector: ${_forbidden}")
  endif()
endforeach()
