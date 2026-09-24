# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Keep the content fingerprint and configure dependencies on one explicit HCU
# upstream public-header closure. These headers are transitively read when FlagDNN
# compiles TritonJITFunction with BACKEND_HCU.
set(_flagdnn_hygon_triton_jit_provenance_headers
  triton_jit/backend_config.h
  triton_jit/backend_policy.h
  triton_jit/backends/hcu_backend.h
  triton_jit/backends/npu_types.h
  triton_jit/jit_function_arg.h
  triton_jit/jit_utils.h
  triton_jit/kernel_metadata.h
  triton_jit/triton_jit_function.h
  triton_jit/triton_kernel.h)

function(flagdnn_hygon_compose_triton_jit_build_identity output)
  cmake_parse_arguments(PARSE_ARGV 1 IDENTITY ""
    "PROVENANCE_SHA256;PYTHON_ENVIRONMENT_SHA256;JIT_SCRIPTS_SHA256" "")
  if(IDENTITY_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unexpected Hygon TritonJIT build-identity arguments: "
      "${IDENTITY_UNPARSED_ARGUMENTS}")
  endif()
  foreach(_name IN ITEMS
      PROVENANCE_SHA256 PYTHON_ENVIRONMENT_SHA256 JIT_SCRIPTS_SHA256)
    string(LENGTH "${IDENTITY_${_name}}" _digest_length)
    if(NOT _digest_length EQUAL 64 OR
       NOT "${IDENTITY_${_name}}" MATCHES "^[0-9a-f]+$")
      message(FATAL_ERROR
        "Hygon TritonJIT ${_name} must be a SHA-256 digest")
    endif()
    string(SUBSTRING "${IDENTITY_${_name}}" 0 16 _short_${_name})
  endforeach()
  set(${output}
    "${_short_PROVENANCE_SHA256}-${_short_PYTHON_ENVIRONMENT_SHA256}-${_short_JIT_SCRIPTS_SHA256}"
    PARENT_SCOPE)
endfunction()

function(_flagdnn_hygon_append_jit_root_candidates output root)
  if(root STREQUAL "")
    set(${output} "${${output}}" PARENT_SCOPE)
    return()
  endif()
  get_filename_component(_root "${root}" ABSOLUTE)
  set(_candidates ${${output}})
  list(APPEND _candidates
    "${_root}"
    "${_root}/build"
    "${_root}/lib/cmake/TritonJIT"
    "${_root}/lib64/cmake/TritonJIT")
  set(${output} "${_candidates}" PARENT_SCOPE)
endfunction()

# Select one HCU TritonJIT distribution as an indivisible tuple. A library,
# header tree, or script directory override is accepted only when all supplied
# paths match the same build-tree or installed-tree layout derived from the
# selected TritonJITConfig.cmake. This prevents independent cache entries from
# silently combining different ABI/provenance roots.
function(flagdnn_hygon_resolve_triton_jit output_prefix)
  cmake_parse_arguments(PARSE_ARGV 1 SELECT ""
    "CONFIG_DIR;ROOT;DEFAULT_ROOT;LIBRARY;INCLUDE_DIR;SCRIPT_DIR" "")
  if(SELECT_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unexpected Hygon TritonJIT resolver arguments: "
      "${SELECT_UNPARSED_ARGUMENTS}")
  endif()
  foreach(_argument IN ITEMS
      CONFIG_DIR ROOT DEFAULT_ROOT LIBRARY INCLUDE_DIR SCRIPT_DIR)
    if(NOT DEFINED SELECT_${_argument})
      set(SELECT_${_argument} "")
    endif()
  endforeach()

  set(_requested_config_dir "${SELECT_CONFIG_DIR}")
  set(_requested_root "${SELECT_ROOT}")
  if(_requested_config_dir STREQUAL "" AND _requested_root STREQUAL "")
    if(DEFINED ENV{FLAGDNN_HYGON_TRITON_JIT_DIR})
      set(_requested_config_dir "$ENV{FLAGDNN_HYGON_TRITON_JIT_DIR}")
    endif()
    if(DEFINED ENV{FLAGDNN_HYGON_TRITON_JIT_ROOT})
      set(_requested_root "$ENV{FLAGDNN_HYGON_TRITON_JIT_ROOT}")
    endif()
  endif()

  set(_config_candidates)
  if(NOT _requested_config_dir STREQUAL "")
    list(APPEND _config_candidates "${_requested_config_dir}")
  elseif(NOT _requested_root STREQUAL "")
    _flagdnn_hygon_append_jit_root_candidates(
      _config_candidates "${_requested_root}")
  else()
    _flagdnn_hygon_append_jit_root_candidates(
      _config_candidates "${SELECT_DEFAULT_ROOT}")
  endif()

  set(_config_dir "")
  foreach(_candidate IN LISTS _config_candidates)
    get_filename_component(_candidate "${_candidate}" ABSOLUTE)
    if(EXISTS "${_candidate}/TritonJITConfig.cmake")
      file(REAL_PATH "${_candidate}" _config_dir)
      break()
    endif()
  endforeach()
  if(_config_dir STREQUAL "")
    message(FATAL_ERROR
      "Cannot locate an HCU TritonJITConfig.cmake; set "
      "FLAGDNN_HYGON_TRITON_JIT_DIR or "
      "FLAGDNN_HYGON_TRITON_JIT_ROOT")
  endif()

  # When both a config directory and a root are supplied, the root remains a
  # provenance constraint rather than being silently ignored.
  if(NOT _requested_config_dir STREQUAL "" AND
     NOT _requested_root STREQUAL "")
    set(_root_candidates)
    _flagdnn_hygon_append_jit_root_candidates(
      _root_candidates "${_requested_root}")
    set(_config_matches_root FALSE)
    foreach(_candidate IN LISTS _root_candidates)
      if(EXISTS "${_candidate}/TritonJITConfig.cmake")
        file(REAL_PATH "${_candidate}" _candidate_real)
        if(_candidate_real STREQUAL _config_dir)
          set(_config_matches_root TRUE)
        endif()
      endif()
    endforeach()
    if(NOT _config_matches_root)
      message(FATAL_ERROR
        "FLAGDNN_HYGON_TRITON_JIT_DIR is outside the selected "
        "FLAGDNN_HYGON_TRITON_JIT_ROOT")
    endif()
  endif()

  set(_config_file "${_config_dir}/TritonJITConfig.cmake")
  file(STRINGS "${_config_file}" _backend_declarations
    REGEX "^[ \t]*set\\([ \t]*TritonJIT_BACKEND[ \t]+.*\\)[ \t]*$")
  list(LENGTH _backend_declarations _backend_declaration_count)
  if(NOT _backend_declaration_count EQUAL 1)
    message(FATAL_ERROR
      "FlagDNN Hygon backend requires an HCU TritonJITConfig.cmake: "
      "${_config_file}")
  endif()
  list(GET _backend_declarations 0 _backend_declaration)
  if(NOT _backend_declaration MATCHES
     "^[ \t]*set\\([ \t]*TritonJIT_BACKEND[ \t]+\"?HCU\"?[ \t]*\\)[ \t]*$")
    message(FATAL_ERROR
      "FlagDNN Hygon backend requires an HCU TritonJITConfig.cmake: "
      "${_config_file}")
  endif()

  get_filename_component(_build_root "${_config_dir}/.." ABSOLUTE)
  get_filename_component(_install_root "${_config_dir}/../../.." ABSOLUTE)
  set(_layout_build_library "${_config_dir}/src/libtriton_jit.so")
  set(_layout_build_include "${_build_root}/include")
  set(_layout_build_scripts "${_build_root}/scripts")
  set(_layout_build_root "${_build_root}")
  set(_layout_install_lib_library
    "${_install_root}/lib/libtriton_jit.so")
  set(_layout_install_lib_include "${_install_root}/include")
  set(_layout_install_lib_scripts
    "${_install_root}/share/triton_jit/scripts")
  set(_layout_install_lib_root "${_install_root}")
  set(_layout_install_lib64_library
    "${_install_root}/lib64/libtriton_jit.so")
  set(_layout_install_lib64_include "${_install_root}/include")
  set(_layout_install_lib64_scripts
    "${_install_root}/share/triton_jit/scripts")
  set(_layout_install_lib64_root "${_install_root}")

  set(_valid_layouts)
  foreach(_layout IN ITEMS build install_lib install_lib64)
    set(_library_variable "_layout_${_layout}_library")
    set(_include_variable "_layout_${_layout}_include")
    set(_scripts_variable "_layout_${_layout}_scripts")
    set(_layout_complete TRUE)
    if(NOT EXISTS "${${_library_variable}}" OR
       NOT EXISTS "${${_scripts_variable}}/standalone_compile.py" OR
       NOT EXISTS "${${_scripts_variable}}/gen_ssig.py")
      set(_layout_complete FALSE)
    endif()
    foreach(_header IN LISTS
        _flagdnn_hygon_triton_jit_provenance_headers)
      if(NOT EXISTS "${${_include_variable}}/${_header}")
        set(_layout_complete FALSE)
      endif()
    endforeach()
    if(_layout_complete)
      set(_layout_${_layout}_logical_library
        "${${_library_variable}}")
      file(REAL_PATH "${${_library_variable}}"
        _layout_${_layout}_library)
      file(REAL_PATH "${${_include_variable}}"
        _layout_${_layout}_include)
      file(REAL_PATH "${${_scripts_variable}}"
        _layout_${_layout}_scripts)
      file(REAL_PATH "${_layout_${_layout}_root}"
        _layout_${_layout}_root)
      list(APPEND _valid_layouts "${_layout}")
    endif()
  endforeach()
  if(NOT _valid_layouts)
    message(FATAL_ERROR
      "The selected HCU TritonJIT config has no complete, coherent "
      "library/include/scripts layout: ${_config_dir}")
  endif()

  foreach(_kind IN ITEMS LIBRARY INCLUDE_DIR SCRIPT_DIR)
    if(NOT "${SELECT_${_kind}}" STREQUAL "")
      if(NOT EXISTS "${SELECT_${_kind}}")
        message(FATAL_ERROR
          "FLAGDNN_HYGON_TRITON_JIT_${_kind} does not exist: "
          "${SELECT_${_kind}}")
      endif()
      file(REAL_PATH "${SELECT_${_kind}}" _explicit_${_kind})
    endif()
  endforeach()

  set(_selected_layout "")
  foreach(_layout IN LISTS _valid_layouts)
    set(_matches TRUE)
    foreach(_kind IN ITEMS LIBRARY INCLUDE_DIR SCRIPT_DIR)
      if(NOT "${SELECT_${_kind}}" STREQUAL "")
        if(_kind STREQUAL "LIBRARY")
          set(_layout_variable "_layout_${_layout}_library")
        elseif(_kind STREQUAL "INCLUDE_DIR")
          set(_layout_variable "_layout_${_layout}_include")
        else()
          set(_layout_variable "_layout_${_layout}_scripts")
        endif()
        if(NOT _explicit_${_kind} STREQUAL "${${_layout_variable}}")
          set(_matches FALSE)
        endif()
      endif()
    endforeach()
    if(_matches AND _selected_layout STREQUAL "")
      set(_selected_layout "${_layout}")
    endif()
  endforeach()
  if(_selected_layout STREQUAL "")
    message(FATAL_ERROR
      "Hygon TritonJIT library/include/scripts overrides do not belong to "
      "one coherent layout derived from ${_config_file}")
  endif()

  set(_library_variable "_layout_${_selected_layout}_library")
  set(_include_variable "_layout_${_selected_layout}_include")
  set(_scripts_variable "_layout_${_selected_layout}_scripts")
  set(_root_variable "_layout_${_selected_layout}_root")
  set(_logical_library_variable
    "_layout_${_selected_layout}_logical_library")
  set(_library "${${_library_variable}}")
  set(_logical_library "${${_logical_library_variable}}")
  set(_include "${${_include_variable}}")
  set(_scripts "${${_scripts_variable}}")
  set(_provenance_root "${${_root_variable}}")

  # Bind the selected image to the package's exported target metadata as well
  # as to its directory layout. This rejects a config copied next to a library
  # from another distribution, even when all conventional paths exist.
  if(CMAKE_SCRIPT_MODE_FILE)
    file(GLOB _target_files "${_config_dir}/TritonJITTargets*.cmake")
  else()
    file(GLOB _target_files CONFIGURE_DEPENDS
      "${_config_dir}/TritonJITTargets*.cmake")
  endif()
  list(SORT _target_files)
  if(NOT _target_files)
    message(FATAL_ERROR
      "The selected TritonJITConfig.cmake has no exported target metadata")
  endif()
  set(_target_metadata "")
  foreach(_target_file IN LISTS _target_files)
    file(READ "${_target_file}" _target_source)
    string(APPEND _target_metadata "${_target_source}\n")
  endforeach()
  if(_selected_layout STREQUAL "build")
    set(_expected_exported_location "${_logical_library}")
  elseif(_selected_layout STREQUAL "install_lib")
    set(_expected_exported_location
      "\${_IMPORT_PREFIX}/lib/libtriton_jit.so")
  else()
    set(_expected_exported_location
      "\${_IMPORT_PREFIX}/lib64/libtriton_jit.so")
  endif()
  string(FIND "${_target_metadata}" "${_expected_exported_location}"
    _exported_location_index)
  if(_exported_location_index EQUAL -1 AND
     _selected_layout STREQUAL "build")
    # Some exports canonicalize an unversioned symlink while others retain it.
    string(FIND "${_target_metadata}" "${_library}"
      _exported_location_index)
  endif()
  if(_exported_location_index EQUAL -1)
    message(FATAL_ERROR
      "TritonJIT target metadata does not bind the selected library: "
      "${_library}")
  endif()
  string(REGEX MATCH
    "IMPORTED_SONAME(_[A-Z0-9_]+)?[ \t\r\n]+\"(libtriton_jit\\.so(\\.[0-9]+)*)\""
    _soname_declaration "${_target_metadata}")
  if(_soname_declaration STREQUAL "")
    message(FATAL_ERROR
      "TritonJIT target metadata has no safe libtriton_jit SONAME")
  endif()
  set(_soname "${CMAKE_MATCH_2}")

  file(SHA256 "${_config_file}" _config_sha256)
  set(_provenance_header_files)
  set(_provenance_header_input "")
  foreach(_header IN LISTS _flagdnn_hygon_triton_jit_provenance_headers)
    set(_header_file "${_include}/${_header}")
    file(SHA256 "${_header_file}" _header_sha256)
    list(APPEND _provenance_header_files "${_header_file}")
    string(APPEND _provenance_header_input
      "${_header}=${_header_sha256}:")
  endforeach()
  file(SHA256 "${_library}" _library_sha256)
  file(SHA256 "${_scripts}/standalone_compile.py" _standalone_sha256)
  file(SHA256 "${_scripts}/gen_ssig.py" _gen_ssig_sha256)
  string(SHA256 _target_metadata_sha256 "${_target_metadata}")
  set(_provenance_input
    "config=${_config_sha256}:targets=${_target_metadata_sha256}:headers=${_provenance_header_input}library=${_library_sha256}:standalone=${_standalone_sha256}:gen_ssig=${_gen_ssig_sha256}")
  string(SHA256 _provenance_sha256 "${_provenance_input}")
  set(_provenance_input_files
    "${_config_file}"
    ${_target_files}
    ${_provenance_header_files}
    "${_library}"
    "${_scripts}/standalone_compile.py"
    "${_scripts}/gen_ssig.py")

  set(${output_prefix}_CONFIG_DIR "${_config_dir}" PARENT_SCOPE)
  set(${output_prefix}_CONFIG_FILE "${_config_file}" PARENT_SCOPE)
  set(${output_prefix}_TARGET_FILES "${_target_files}" PARENT_SCOPE)
  set(${output_prefix}_LIBRARY "${_library}" PARENT_SCOPE)
  set(${output_prefix}_SONAME "${_soname}" PARENT_SCOPE)
  set(${output_prefix}_INCLUDE_DIR "${_include}" PARENT_SCOPE)
  set(${output_prefix}_SCRIPT_DIR "${_scripts}" PARENT_SCOPE)
  set(${output_prefix}_PROVENANCE_ROOT "${_provenance_root}" PARENT_SCOPE)
  set(${output_prefix}_PROVENANCE_SHA256
    "${_provenance_sha256}" PARENT_SCOPE)
  set(${output_prefix}_PROVENANCE_HEADER_FILES
    "${_provenance_header_files}" PARENT_SCOPE)
  set(${output_prefix}_PROVENANCE_INPUT_FILES
    "${_provenance_input_files}" PARENT_SCOPE)
  set(${output_prefix}_LAYOUT "${_selected_layout}" PARENT_SCOPE)
endfunction()
