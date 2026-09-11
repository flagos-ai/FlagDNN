# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

set(_flagdnn_thead_triton_jit_headers
  triton_jit/backend_config.h
  triton_jit/backend_policy.h
  triton_jit/backends/cuda_backend.h
  triton_jit/jit_function_arg.h
  triton_jit/jit_utils.h
  triton_jit/kernel_metadata.h
  triton_jit/triton_jit_function.h
  triton_jit/triton_kernel.h)

function(_flagdnn_thead_jit_real input output description)
  if(input STREQUAL "" OR NOT EXISTS "${input}")
    message(FATAL_ERROR "${description} does not exist: ${input}")
  endif()
  file(REAL_PATH "${input}" _resolved EXPAND_TILDE)
  if(_resolved STREQUAL "" OR NOT EXISTS "${_resolved}")
    message(FATAL_ERROR "${description} has an unresolved symlink: ${input}")
  endif()
  set(${output} "${_resolved}" PARENT_SCOPE)
endfunction()

function(_flagdnn_thead_jit_require_below root path description)
  file(RELATIVE_PATH _relative "${root}" "${path}")
  if(IS_ABSOLUTE "${_relative}" OR
     _relative STREQUAL ".." OR
     _relative MATCHES "^\\.\\./")
    message(FATAL_ERROR
      "${description} is outside the selected THead TritonJIT root "
      "${root}: ${path}")
  endif()
endfunction()

function(_flagdnn_thead_append_jit_candidates output root)
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

function(_flagdnn_thead_define_jit_target library include_directory)
  set(_target FlagDNN::THeadTritonJIT)
  if(TARGET "${_target}")
    get_target_property(_existing_library
      "${_target}" INTERFACE_LINK_LIBRARIES)
    get_target_property(_existing_include
      "${_target}" INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(_existing_definitions
      "${_target}" INTERFACE_COMPILE_DEFINITIONS)
    if(NOT _existing_library STREQUAL library OR
       NOT _existing_include STREQUAL include_directory OR
       NOT _existing_definitions STREQUAL
         "BACKEND_CUDA;FMT_HEADER_ONLY=1")
      message(FATAL_ERROR
        "FlagDNN::THeadTritonJIT already has different provenance")
    endif()
    return()
  endif()
  add_library("${_target}" INTERFACE IMPORTED GLOBAL)
  set_target_properties("${_target}" PROPERTIES
    INTERFACE_LINK_LIBRARIES "${library}"
    INTERFACE_INCLUDE_DIRECTORIES "${include_directory}"
    INTERFACE_COMPILE_DEFINITIONS "BACKEND_CUDA;FMT_HEADER_ONLY=1")
endfunction()

function(flagdnn_thead_resolve_triton_jit output_prefix)
  cmake_parse_arguments(PARSE_ARGV 1 SELECT ""
    "CONFIG_DIR;ROOT;DEFAULT_ROOT;LIBRARY;INCLUDE_DIR;SCRIPT_DIR" "")
  if(SELECT_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "unexpected THead TritonJIT resolver arguments: "
      "${SELECT_UNPARSED_ARGUMENTS}")
  endif()
  foreach(_argument IN ITEMS
      CONFIG_DIR ROOT DEFAULT_ROOT LIBRARY INCLUDE_DIR SCRIPT_DIR)
    if(NOT DEFINED SELECT_${_argument})
      set(SELECT_${_argument} "")
    endif()
  endforeach()

  set(_requested_config "${SELECT_CONFIG_DIR}")
  set(_requested_root "${SELECT_ROOT}")
  if(_requested_config STREQUAL "" AND _requested_root STREQUAL "")
    if(NOT "${FLAGDNN_THEAD_TRITON_JIT_ROOT}" STREQUAL "")
      set(_requested_root "${FLAGDNN_THEAD_TRITON_JIT_ROOT}")
    elseif(NOT "${SELECT_DEFAULT_ROOT}" STREQUAL "")
      set(_requested_root "${SELECT_DEFAULT_ROOT}")
    endif()
  endif()

  set(_declared_root "")
  if(NOT _requested_root STREQUAL "")
    _flagdnn_thead_jit_real(
      "${_requested_root}" _declared_root "THead TritonJIT root")
  endif()

  set(_config_candidates)
  if(NOT _requested_config STREQUAL "")
    list(APPEND _config_candidates "${_requested_config}")
  elseif(NOT _declared_root STREQUAL "")
    _flagdnn_thead_append_jit_candidates(
      _config_candidates "${_declared_root}")
  endif()
  set(_config_dir "")
  foreach(_candidate IN LISTS _config_candidates)
    if(EXISTS "${_candidate}/TritonJITConfig.cmake")
      _flagdnn_thead_jit_real(
        "${_candidate}" _config_dir "TritonJIT config directory")
      break()
    endif()
  endforeach()
  if(_config_dir STREQUAL "")
    message(FATAL_ERROR
      "cannot locate a CUDA TritonJITConfig.cmake below the selected "
      "FLAGDNN_THEAD_TRITON_JIT_ROOT")
  endif()
  if(NOT _declared_root STREQUAL "")
    _flagdnn_thead_jit_require_below(
      "${_declared_root}" "${_config_dir}" "TritonJIT config directory")
  endif()

  set(_config_file "${_config_dir}/TritonJITConfig.cmake")
  file(STRINGS "${_config_file}" _backend_declarations
    REGEX "^[ \t]*set\\([ \t]*TritonJIT_BACKEND[ \t]+.*\\)[ \t]*$")
  list(LENGTH _backend_declarations _backend_count)
  if(NOT _backend_count EQUAL 1)
    message(FATAL_ERROR
      "THead requires a CUDA TritonJITConfig.cmake with one backend "
      "declaration: ${_config_file}")
  endif()
  list(GET _backend_declarations 0 _backend_declaration)
  if(NOT _backend_declaration MATCHES
     "^[ \t]*set\\([ \t]*TritonJIT_BACKEND[ \t]+['\"]?CUDA['\"]?[ \t]*\\)[ \t]*$")
    message(FATAL_ERROR
      "THead requires a CUDA TritonJITConfig.cmake: ${_config_file}")
  endif()

  get_filename_component(_build_root "${_config_dir}/.." ABSOLUTE)
  get_filename_component(_install_root "${_config_dir}/../../.." ABSOLUTE)
  set(_layout_build_root "${_build_root}")
  set(_layout_build_library "${_config_dir}/src/libtriton_jit.so")
  set(_layout_build_include "${_build_root}/include")
  set(_layout_build_scripts "${_build_root}/scripts")
  set(_layout_install_lib_root "${_install_root}")
  set(_layout_install_lib_library "${_install_root}/lib/libtriton_jit.so")
  set(_layout_install_lib_include "${_install_root}/include")
  set(_layout_install_lib_scripts
    "${_install_root}/share/triton_jit/scripts")
  set(_layout_install_lib64_root "${_install_root}")
  set(_layout_install_lib64_library
    "${_install_root}/lib64/libtriton_jit.so")
  set(_layout_install_lib64_include "${_install_root}/include")
  set(_layout_install_lib64_scripts
    "${_install_root}/share/triton_jit/scripts")

  set(_valid_layouts)
  foreach(_layout IN ITEMS build install_lib install_lib64)
    set(_library_variable "_layout_${_layout}_library")
    set(_include_variable "_layout_${_layout}_include")
    set(_scripts_variable "_layout_${_layout}_scripts")
    set(_complete TRUE)
    if(NOT EXISTS "${${_library_variable}}" OR
       NOT EXISTS "${${_scripts_variable}}/standalone_compile.py" OR
       NOT EXISTS "${${_scripts_variable}}/gen_ssig.py")
      set(_complete FALSE)
    endif()
    foreach(_header IN LISTS _flagdnn_thead_triton_jit_headers)
      if(NOT EXISTS "${${_include_variable}}/${_header}")
        set(_complete FALSE)
      endif()
    endforeach()
    if(_complete)
      set(_layout_${_layout}_logical_library
        "${${_library_variable}}")
      _flagdnn_thead_jit_real(
        "${${_library_variable}}" _layout_${_layout}_library
        "TritonJIT shared library")
      _flagdnn_thead_jit_real(
        "${${_include_variable}}" _layout_${_layout}_include
        "TritonJIT include directory")
      _flagdnn_thead_jit_real(
        "${${_scripts_variable}}" _layout_${_layout}_scripts
        "TritonJIT script directory")
      _flagdnn_thead_jit_real(
        "${_layout_${_layout}_root}" _layout_${_layout}_root
        "TritonJIT distribution root")
      list(APPEND _valid_layouts "${_layout}")
    endif()
  endforeach()
  if(NOT _valid_layouts)
    message(FATAL_ERROR
      "selected CUDA TritonJIT distribution has no complete coherent layout; "
      "required resources include libtriton_jit.so, cuda_backend.h, "
      "standalone_compile.py, and gen_ssig.py")
  endif()

  foreach(_kind IN ITEMS LIBRARY INCLUDE_DIR SCRIPT_DIR)
    if(NOT "${SELECT_${_kind}}" STREQUAL "")
      _flagdnn_thead_jit_real(
        "${SELECT_${_kind}}" _explicit_${_kind}
        "explicit THead TritonJIT ${_kind}")
    endif()
  endforeach()
  set(_selected_layout "")
  foreach(_layout IN LISTS _valid_layouts)
    set(_matches TRUE)
    foreach(_kind IN ITEMS LIBRARY INCLUDE_DIR SCRIPT_DIR)
      if(NOT "${SELECT_${_kind}}" STREQUAL "")
        if(_kind STREQUAL "LIBRARY")
          set(_layout_value "${_layout_${_layout}_library}")
        elseif(_kind STREQUAL "INCLUDE_DIR")
          set(_layout_value "${_layout_${_layout}_include}")
        else()
          set(_layout_value "${_layout_${_layout}_scripts}")
        endif()
        if(NOT _explicit_${_kind} STREQUAL _layout_value)
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
      "THead TritonJIT overrides do not belong to one coherent distribution")
  endif()

  set(_root "${_layout_${_selected_layout}_root}")
  set(_library "${_layout_${_selected_layout}_library}")
  set(_logical_library
    "${_layout_${_selected_layout}_logical_library}")
  set(_include "${_layout_${_selected_layout}_include}")
  set(_scripts "${_layout_${_selected_layout}_scripts}")
  if(NOT _declared_root STREQUAL "")
    foreach(_resource IN ITEMS
        "${_root}" "${_config_dir}" "${_library}" "${_include}" "${_scripts}")
      _flagdnn_thead_jit_require_below(
        "${_declared_root}" "${_resource}" "TritonJIT resource")
    endforeach()
  endif()

  file(GLOB _target_files "${_config_dir}/TritonJITTargets*.cmake")
  list(SORT _target_files)
  if(NOT _target_files)
    message(FATAL_ERROR
      "selected TritonJITConfig.cmake has no exported target metadata")
  endif()
  set(_target_metadata "")
  foreach(_target_file IN LISTS _target_files)
    file(READ "${_target_file}" _target_source)
    string(APPEND _target_metadata "${_target_source}\n")
  endforeach()
  string(FIND "${_target_metadata}" "BACKEND_CUDA" _definition_index)
  if(_definition_index EQUAL -1)
    message(FATAL_ERROR
      "selected TritonJIT target metadata does not export BACKEND_CUDA")
  endif()
  if(_selected_layout STREQUAL "build")
    set(_expected_location "${_logical_library}")
  elseif(_selected_layout STREQUAL "install_lib")
    set(_expected_location "\${_IMPORT_PREFIX}/lib/libtriton_jit.so")
  else()
    set(_expected_location "\${_IMPORT_PREFIX}/lib64/libtriton_jit.so")
  endif()
  string(FIND "${_target_metadata}" "${_expected_location}" _location_index)
  if(_location_index EQUAL -1 AND _selected_layout STREQUAL "build")
    string(FIND "${_target_metadata}" "${_library}" _location_index)
  endif()
  if(_location_index EQUAL -1)
    message(FATAL_ERROR
      "TritonJIT target metadata does not bind the selected shared library")
  endif()
  string(REGEX MATCH
    "IMPORTED_SONAME(_[A-Z0-9_]+)?[ \t\r\n]+\"(libtriton_jit\\.so(\\.[0-9]+)*)\""
    _soname_declaration "${_target_metadata}")
  if(_soname_declaration STREQUAL "")
    message(FATAL_ERROR
      "TritonJIT target metadata has no safe libtriton_jit SONAME")
  endif()
  set(_soname "${CMAKE_MATCH_2}")

  set(_provenance_input_files "${_config_file}" ${_target_files})
  file(SHA256 "${_config_file}" _config_sha256)
  string(SHA256 _target_sha256 "${_target_metadata}")
  set(_provenance_material
    "config=${_config_sha256}:targets=${_target_sha256}:")
  foreach(_header IN LISTS _flagdnn_thead_triton_jit_headers)
    set(_header_file "${_include}/${_header}")
    file(SHA256 "${_header_file}" _header_sha256)
    string(APPEND _provenance_material
      "header:${_header}=${_header_sha256}:")
    list(APPEND _provenance_input_files "${_header_file}")
  endforeach()
  foreach(_input IN ITEMS
      "${_library}"
      "${_scripts}/standalone_compile.py"
      "${_scripts}/gen_ssig.py")
    file(SHA256 "${_input}" _input_sha256)
    string(APPEND _provenance_material
      "file:${_input}=${_input_sha256}:")
    list(APPEND _provenance_input_files "${_input}")
  endforeach()
  string(SHA256 _provenance_sha256 "${_provenance_material}")

  _flagdnn_thead_define_jit_target("${_library}" "${_include}")

  set(${output_prefix}_TARGET "FlagDNN::THeadTritonJIT" PARENT_SCOPE)
  set(${output_prefix}_ROOT "${_root}" PARENT_SCOPE)
  set(${output_prefix}_CONFIG_DIR "${_config_dir}" PARENT_SCOPE)
  set(${output_prefix}_CONFIG_FILE "${_config_file}" PARENT_SCOPE)
  set(${output_prefix}_LIBRARY "${_library}" PARENT_SCOPE)
  set(${output_prefix}_INCLUDE_DIR "${_include}" PARENT_SCOPE)
  set(${output_prefix}_SCRIPT_DIR "${_scripts}" PARENT_SCOPE)
  set(${output_prefix}_BACKEND "CUDA" PARENT_SCOPE)
  set(${output_prefix}_LAYOUT "${_selected_layout}" PARENT_SCOPE)
  set(${output_prefix}_SONAME "${_soname}" PARENT_SCOPE)
  set(${output_prefix}_PROVENANCE_SHA256
    "${_provenance_sha256}" PARENT_SCOPE)
  set(${output_prefix}_PROVENANCE_INPUT_FILES
    "${_provenance_input_files}" PARENT_SCOPE)
endfunction()
