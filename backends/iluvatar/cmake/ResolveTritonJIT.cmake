# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(_flagdnn_iluvatar_jit_real input output description)
  if(input STREQUAL "" OR NOT EXISTS "${input}")
    message(FATAL_ERROR "${description} does not exist: ${input}")
  endif()
  file(REAL_PATH "${input}" _resolved EXPAND_TILDE)
  if(_resolved STREQUAL "" OR NOT EXISTS "${_resolved}")
    message(FATAL_ERROR "${description} has an unresolved symlink: ${input}")
  endif()
  set(${output} "${_resolved}" PARENT_SCOPE)
endfunction()

function(_flagdnn_iluvatar_jit_location target output)
  set(_location "")
  foreach(_property IN ITEMS
      IMPORTED_LOCATION IMPORTED_LOCATION_NOCONFIG
      IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_RELWITHDEBINFO
      IMPORTED_LOCATION_DEBUG)
    if(_location STREQUAL "")
      get_target_property(_candidate "${target}" "${_property}")
      if(_candidate AND NOT _candidate MATCHES "-NOTFOUND$" AND
         EXISTS "${_candidate}")
        set(_location "${_candidate}")
      endif()
    endif()
  endforeach()
  if(_location STREQUAL "")
    message(FATAL_ERROR
      "Cannot resolve the shared object behind ${target}")
  endif()
  _flagdnn_iluvatar_jit_real(
    "${_location}" _location "IX libtriton_jit shared object")
  set(${output} "${_location}" PARENT_SCOPE)
endfunction()

function(flagdnn_iluvatar_resolve_triton_jit output_prefix)
  set(_config_candidates)
  if(NOT "${FLAGDNN_ILUVATAR_TRITON_JIT_DIR}" STREQUAL "")
    list(APPEND _config_candidates
      "${FLAGDNN_ILUVATAR_TRITON_JIT_DIR}")
  elseif(NOT "${FLAGDNN_ILUVATAR_TRITON_JIT_ROOT}" STREQUAL "")
    list(APPEND _config_candidates
      "${FLAGDNN_ILUVATAR_TRITON_JIT_ROOT}"
      "${FLAGDNN_ILUVATAR_TRITON_JIT_ROOT}/build"
      "${FLAGDNN_ILUVATAR_TRITON_JIT_ROOT}/lib/cmake/TritonJIT"
      "${FLAGDNN_ILUVATAR_TRITON_JIT_ROOT}/lib64/cmake/TritonJIT")
  else()
    list(APPEND _config_candidates
      "/usr/local/lib/cmake/TritonJIT")
  endif()

  set(_config_dir "")
  foreach(_candidate IN LISTS _config_candidates)
    if(EXISTS "${_candidate}/TritonJITConfig.cmake")
      _flagdnn_iluvatar_jit_real(
        "${_candidate}" _config_dir "TritonJIT CMake package directory")
      break()
    endif()
  endforeach()
  if(_config_dir STREQUAL "")
    message(FATAL_ERROR
      "Cannot locate TritonJITConfig.cmake; set "
      "FLAGDNN_ILUVATAR_TRITON_JIT_DIR or "
      "FLAGDNN_ILUVATAR_TRITON_JIT_ROOT")
  endif()

  foreach(_module_candidate IN ITEMS
      "${_config_dir}"
      "${_config_dir}/../cmake")
    if(EXISTS "${_module_candidate}/FindTorch.cmake")
      file(REAL_PATH "${_module_candidate}" _module_candidate)
      list(PREPEND CMAKE_MODULE_PATH "${_module_candidate}")
      break()
    endif()
  endforeach()

  find_package(TritonJIT CONFIG REQUIRED
    PATHS "${_config_dir}" NO_DEFAULT_PATH)
  if(NOT TritonJIT_BACKEND STREQUAL "IX")
    message(FATAL_ERROR
      "FlagDNN Iluvatar backend requires an IX TritonJIT package; got "
      "${TritonJIT_BACKEND}")
  endif()
  if(TARGET TritonJIT::triton_jit)
    set(_target TritonJIT::triton_jit)
  elseif(TARGET triton_jit)
    set(_target triton_jit)
  else()
    message(FATAL_ERROR
      "TritonJIT package did not export TritonJIT::triton_jit")
  endif()

  get_target_property(_definitions "${_target}"
    INTERFACE_COMPILE_DEFINITIONS)
  if(NOT _definitions OR _definitions MATCHES "-NOTFOUND$")
    set(_definitions)
  endif()
  list(FIND _definitions BACKEND_IX _backend_ix_index)
  if(_backend_ix_index EQUAL -1)
    message(FATAL_ERROR
      "IX TritonJIT target must export the BACKEND_IX definition")
  endif()

  get_target_property(_target_includes "${_target}"
    INTERFACE_INCLUDE_DIRECTORIES)
  set(_target_include "")
  foreach(_candidate IN LISTS _target_includes)
    if(NOT _candidate MATCHES "^\\$<" AND
       EXISTS "${_candidate}/triton_jit/backends/ix_backend.h")
      _flagdnn_iluvatar_jit_real(
        "${_candidate}" _target_include "IX TritonJIT include directory")
      break()
    endif()
  endforeach()
  if(_target_include STREQUAL "")
    message(FATAL_ERROR
      "IX TritonJIT target has no usable public include directory")
  endif()
  foreach(_header IN ITEMS
      triton_jit/backends/ix_backend.h
      triton_jit/triton_jit_function.h
      triton_jit/jit_utils.h)
    if(NOT EXISTS "${_target_include}/${_header}")
      message(FATAL_ERROR "IX TritonJIT public header is missing: ${_header}")
    endif()
  endforeach()

  _flagdnn_iluvatar_jit_location("${_target}" _target_library)
  set(_include "${_target_include}")
  set(_library "${_target_library}")
  if(NOT "${FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR}" STREQUAL "")
    _flagdnn_iluvatar_jit_real(
      "${FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR}" _explicit_include
      "explicit IX TritonJIT include directory")
    if(NOT _explicit_include STREQUAL _target_include)
      message(FATAL_ERROR
        "FLAGDNN_ILUVATAR_TRITON_JIT_INCLUDE_DIR does not match the "
        "selected package target")
    endif()
    set(_include "${_explicit_include}")
  endif()
  if(NOT "${FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY}" STREQUAL "")
    _flagdnn_iluvatar_jit_real(
      "${FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY}" _explicit_library
      "explicit IX libtriton_jit shared object")
    if(NOT _explicit_library STREQUAL _target_library)
      message(FATAL_ERROR
        "FLAGDNN_ILUVATAR_TRITON_JIT_LIBRARY does not match the "
        "selected package target")
    endif()
    set(_library "${_explicit_library}")
  endif()

  set(_script_candidates)
  if(NOT "${FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR}" STREQUAL "")
    list(APPEND _script_candidates
      "${FLAGDNN_ILUVATAR_TRITON_JIT_SCRIPT_DIR}")
  endif()
  list(APPEND _script_candidates
    "${_config_dir}/../scripts"
    "${_config_dir}/../../../share/triton_jit/scripts")
  set(_scripts "")
  foreach(_candidate IN LISTS _script_candidates)
    if(EXISTS "${_candidate}/standalone_compile.py" AND
       EXISTS "${_candidate}/gen_ssig.py")
      _flagdnn_iluvatar_jit_real(
        "${_candidate}" _scripts "IX TritonJIT script directory")
      break()
    endif()
  endforeach()
  if(_scripts STREQUAL "")
    message(FATAL_ERROR
      "Cannot resolve standalone_compile.py and gen_ssig.py from the "
      "selected IX TritonJIT package")
  endif()

  if(NOT "${FLAGDNN_ILUVATAR_TRITON_JIT_ROOT}" STREQUAL "")
    _flagdnn_iluvatar_jit_real(
      "${FLAGDNN_ILUVATAR_TRITON_JIT_ROOT}" _declared_root
      "IX TritonJIT root")
    foreach(_selected IN ITEMS
        "${_config_dir}" "${_library}" "${_include}" "${_scripts}")
      file(RELATIVE_PATH _relative "${_declared_root}" "${_selected}")
      if(IS_ABSOLUTE "${_relative}" OR _relative STREQUAL ".." OR
         _relative MATCHES "^\\.\\./")
        message(FATAL_ERROR
          "selected IX TritonJIT resource is outside "
          "FLAGDNN_ILUVATAR_TRITON_JIT_ROOT: ${_selected}")
      endif()
    endforeach()
  endif()

  get_target_property(_aliased_target "${_target}" ALIASED_TARGET)
  if(_aliased_target)
    set(_shared_target "${_aliased_target}")
  else()
    set(_shared_target "${_target}")
  endif()
  get_target_property(_target_is_imported "${_shared_target}" IMPORTED)
  if(_target_is_imported)
    set_property(TARGET "${_shared_target}" PROPERTY IMPORTED_GLOBAL TRUE)
  endif()

  set(${output_prefix}_TARGET "${_shared_target}" PARENT_SCOPE)
  set(${output_prefix}_CONFIG_DIR "${_config_dir}" PARENT_SCOPE)
  set(${output_prefix}_LIBRARY "${_library}" PARENT_SCOPE)
  set(${output_prefix}_INCLUDE_DIR "${_include}" PARENT_SCOPE)
  set(${output_prefix}_SCRIPT_DIR "${_scripts}" PARENT_SCOPE)
  set(${output_prefix}_BACKEND "${TritonJIT_BACKEND}" PARENT_SCOPE)
endfunction()
