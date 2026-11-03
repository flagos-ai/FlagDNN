# Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

function(_flagdnn_ascend_import_or_verify target location includes)
  file(REAL_PATH "${location}" expected_location)
  if(TARGET "${target}")
    get_target_property(actual_location "${target}" IMPORTED_LOCATION)
    if(NOT actual_location)
      get_target_property(actual_location "${target}"
        IMPORTED_LOCATION_NOCONFIG)
    endif()
    if(NOT actual_location)
      message(FATAL_ERROR
        "Existing ${target} target has no imported library location")
    endif()
    file(REAL_PATH "${actual_location}" actual_location)
    if(NOT actual_location STREQUAL expected_location)
      message(FATAL_ERROR
        "Existing ${target} resolves to ${actual_location}, but selected CANN "
        "requires ${expected_location}")
    endif()
    return()
  endif()

  add_library("${target}" SHARED IMPORTED GLOBAL)
  set_target_properties("${target}" PROPERTIES
    IMPORTED_LOCATION "${expected_location}"
    INTERFACE_INCLUDE_DIRECTORIES "${includes}")
endfunction()

function(flagdnn_ascend_prepare_triton_jit CANN_ROOT_PATH)
  if(NOT IS_ABSOLUTE "${CANN_ROOT_PATH}" OR
     NOT IS_DIRECTORY "${CANN_ROOT_PATH}")
    message(FATAL_ERROR
      "CANN_ROOT must name an existing absolute CANN installation")
  endif()
  file(REAL_PATH "${CANN_ROOT_PATH}" canonical_cann_root)

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    set(ascend_arch_directory "aarch64-linux")
    set(expected_package_arch "aarch64")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    set(ascend_arch_directory "x86_64-linux")
    set(expected_package_arch "x86_64")
  else()
    message(FATAL_ERROR
      "Ascend backend does not support host processor ${CMAKE_SYSTEM_PROCESSOR}")
  endif()

  set(install_info_candidates)
  foreach(candidate IN ITEMS
      "${CANN_ROOT_PATH}/${ascend_arch_directory}/ascend_toolkit_install.info"
      "${canonical_cann_root}/${ascend_arch_directory}/ascend_toolkit_install.info"
      "${CANN_ROOT_PATH}/${ascend_arch_directory}/ascend_all_cann_install.info"
      "${canonical_cann_root}/${ascend_arch_directory}/ascend_all_cann_install.info")
    if(EXISTS "${candidate}")
      file(REAL_PATH "${candidate}" canonical_candidate)
      list(APPEND install_info_candidates "${canonical_candidate}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES install_info_candidates)
  list(LENGTH install_info_candidates install_info_count)
  if(install_info_count EQUAL 0)
    message(FATAL_ERROR
      "No ascend_toolkit_install.info or ascend_all_cann_install.info found "
      "below CANN_ROOT=${CANN_ROOT_PATH} (expected ${ascend_arch_directory})")
  endif()

  set(cann_version "")
  set(cann_inner_version "")
  set(cann_package_arch "")
  foreach(install_info IN LISTS install_info_candidates)
    file(STRINGS "${install_info}" install_info_lines)
    foreach(key IN ITEMS package_name version innerversion arch)
      set(matches ${install_info_lines})
      list(FILTER matches INCLUDE REGEX "^${key}=")
      list(LENGTH matches match_count)
      if(NOT match_count EQUAL 1)
        message(FATAL_ERROR
          "${install_info} must contain exactly one ${key}= entry")
      endif()
      list(GET matches 0 value)
      string(REGEX REPLACE "^[^=]+=" "" value "${value}")
      if(key STREQUAL "package_name")
        set(candidate_package_name "${value}")
      elseif(key STREQUAL "version")
        set(candidate_version "${value}")
      elseif(key STREQUAL "innerversion")
        set(candidate_inner_version "${value}")
      else()
        set(candidate_arch "${value}")
      endif()
    endforeach()
    # Toolkit and combined CANN installers use distinct package names.
    if(NOT candidate_package_name STREQUAL "Ascend-cann-toolkit" AND
       NOT candidate_package_name STREQUAL "Ascend-cann")
      message(FATAL_ERROR
        "${install_info} has unexpected package_name=${candidate_package_name}")
    endif()
    if(cann_version AND
       (NOT cann_version STREQUAL candidate_version OR
        NOT cann_inner_version STREQUAL candidate_inner_version OR
        NOT cann_package_arch STREQUAL candidate_arch))
      message(FATAL_ERROR
        "CANN install metadata candidates disagree under ${CANN_ROOT_PATH}")
    endif()
    set(cann_version "${candidate_version}")
    set(cann_inner_version "${candidate_inner_version}")
    set(cann_package_arch "${candidate_arch}")
  endforeach()

  if(cann_version VERSION_LESS "9.0" OR
     NOT cann_version VERSION_LESS "10.0")
    message(FATAL_ERROR
      "FlagDNN Ascend requires CANN >= 9.0 and < 10.0; found ${cann_version}")
  endif()
  if(NOT cann_package_arch STREQUAL expected_package_arch)
    message(FATAL_ERROR
      "CANN package arch ${cann_package_arch} does not match host "
      "${expected_package_arch}")
  endif()

  find_path(ascend_include_directory
    NAMES acl/acl.h acl/acl_rt.h
    PATHS
      "${canonical_cann_root}/${ascend_arch_directory}/include"
      "${canonical_cann_root}/${ascend_arch_directory}/pkg_inc"
    NO_DEFAULT_PATH
    NO_CACHE)
  find_library(ascendcl_library
    NAMES ascendcl
    PATHS
      "${canonical_cann_root}/${ascend_arch_directory}/lib64"
      "${canonical_cann_root}/lib64"
    NO_DEFAULT_PATH
    NO_CACHE)
  find_library(ascend_runtime_library
    NAMES runtime
    PATHS
      "${canonical_cann_root}/${ascend_arch_directory}/lib64"
      "${canonical_cann_root}/lib64"
    NO_DEFAULT_PATH
    NO_CACHE)
  if(NOT ascend_include_directory OR NOT ascendcl_library OR
     NOT ascend_runtime_library)
    message(FATAL_ERROR
      "CANN ${cann_version} is missing ACL headers, libascendcl.so, or "
      "libruntime.so")
  endif()
  file(REAL_PATH "${ascend_include_directory}" ascend_include_directory)
  file(REAL_PATH "${ascendcl_library}" ascendcl_library)
  file(REAL_PATH "${ascend_runtime_library}" ascend_runtime_library)

  # Installed NPU TritonJIT exports reference both names, but some releases
  # create only Ascend::ascendcl. Define and validate both before importing it.
  _flagdnn_ascend_import_or_verify(
    Ascend::ascendcl "${ascendcl_library}" "${ascend_include_directory}")
  _flagdnn_ascend_import_or_verify(
    Ascend::runtime "${ascend_runtime_library}" "${ascend_include_directory}")

  set(FLAGDNN_ASCEND_CANN_ROOT "${canonical_cann_root}" PARENT_SCOPE)
  set(FLAGDNN_ASCEND_CANN_VERSION "${cann_version}" PARENT_SCOPE)
  set(FLAGDNN_ASCEND_CANN_INNER_VERSION
      "${cann_inner_version}" PARENT_SCOPE)
  set(FLAGDNN_ASCEND_INCLUDE_DIR "${ascend_include_directory}" PARENT_SCOPE)
  set(FLAGDNN_ASCEND_ASCENDCL_LIBRARY "${ascendcl_library}" PARENT_SCOPE)
  set(FLAGDNN_ASCEND_RUNTIME_LIBRARY
      "${ascend_runtime_library}" PARENT_SCOPE)
endfunction()
