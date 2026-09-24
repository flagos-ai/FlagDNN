# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# CMake build-tree libraries can contain colon padding in their runtime path.
# Normalize only FlagDNN's private copy: empty entries search the process cwd.
# Keep every nonempty entry and leave the selected dependency untouched.
function(flagdnn_hygon_stage_triton_jit source destination)
  file(REAL_PATH "${source}" _source_real)
  get_filename_component(_destination_real "${destination}" ABSOLUTE)
  if(EXISTS "${destination}")
    file(REAL_PATH "${destination}" _destination_real)
  endif()
  if(_source_real STREQUAL _destination_real)
    message(FATAL_ERROR "Hygon TritonJIT staging must not overwrite its source")
  endif()
  # Replace a preexisting file/link so the staged image always owns its bytes.
  file(REMOVE "${destination}")
  configure_file("${source}" "${destination}" COPYONLY)
  find_program(_flagdnn_hygon_readelf NAMES readelf llvm-readelf REQUIRED)
  execute_process(
    COMMAND "${_flagdnn_hygon_readelf}" -d "${destination}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _dynamic_section
    ERROR_VARIABLE _error)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "Cannot inspect the Hygon-private TritonJIT runtime path: ${_error}")
  endif()
  string(REGEX MATCH
    "\\((RPATH|RUNPATH)\\)[^\n]*\\[([^]]*)\\]"
    _rpath_record "${_dynamic_section}")
  if(_rpath_record STREQUAL "")
    return()
  endif()
  set(_old_rpath "${CMAKE_MATCH_2}")
  string(REGEX REPLACE "^:+|:+$" "" _new_rpath "${_old_rpath}")
  string(REGEX REPLACE ":+" ":" _new_rpath "${_new_rpath}")
  if(_new_rpath STREQUAL "")
    file(RPATH_REMOVE FILE "${destination}")
  elseif(NOT _old_rpath STREQUAL _new_rpath)
    file(RPATH_CHANGE FILE "${destination}"
      OLD_RPATH "${_old_rpath}" NEW_RPATH "${_new_rpath}")
  endif()
endfunction()
