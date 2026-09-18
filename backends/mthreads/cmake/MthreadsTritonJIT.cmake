include_guard(GLOBAL)

function(_flagdnn_mthreads_imported_location out_value target)
  set(_location "")
  foreach(_property IN ITEMS
      IMPORTED_LOCATION
      IMPORTED_LOCATION_NOCONFIG
      IMPORTED_LOCATION_RELEASE
      IMPORTED_LOCATION_RELWITHDEBINFO
      IMPORTED_LOCATION_DEBUG)
    if(NOT _location)
      get_target_property(_candidate "${target}" "${_property}")
      if(_candidate AND EXISTS "${_candidate}")
        set(_location "${_candidate}")
      endif()
    endif()
  endforeach()
  if(NOT _location)
    message(FATAL_ERROR
      "cannot determine imported library location for ${target}")
  endif()
  file(REAL_PATH "${_location}" _location EXPAND_TILDE)
  set("${out_value}" "${_location}" PARENT_SCOPE)
endfunction()

function(flagdnn_mthreads_resolve_triton_jit out_prefix)
  set(options)
  set(one_value_arguments
      CONFIG_DIR
      EXPECTED_BACKEND
      ENVIRONMENT_REPORT
      CODEGEN_PYTHON
      MUSA_ROOT
      PRIVATE_ROOT)
  cmake_parse_arguments(
    MTHREADS_JIT
    "${options}"
    "${one_value_arguments}"
    ""
    ${ARGN})

  if(MTHREADS_JIT_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "flagdnn_mthreads_resolve_triton_jit received unknown arguments: "
      "${MTHREADS_JIT_UNPARSED_ARGUMENTS}")
  endif()
  foreach(_required_argument IN ITEMS
      CONFIG_DIR
      EXPECTED_BACKEND
      ENVIRONMENT_REPORT
      CODEGEN_PYTHON
      MUSA_ROOT
      PRIVATE_ROOT)
    if(NOT MTHREADS_JIT_${_required_argument})
      message(FATAL_ERROR
        "mthreads TritonJIT resolver requires ${_required_argument}")
    endif()
  endforeach()
  if(NOT MTHREADS_JIT_EXPECTED_BACKEND STREQUAL "MUSA")
    message(FATAL_ERROR
      "mthreads TritonJIT expected backend must be MUSA")
  endif()
  if(NOT IS_DIRECTORY "${MTHREADS_JIT_CONFIG_DIR}")
    message(FATAL_ERROR
      "FLAGDNN_MTHREADS_TRITON_JIT_DIR is not a directory: "
      "${MTHREADS_JIT_CONFIG_DIR}")
  endif()

  file(REAL_PATH "${MTHREADS_JIT_CONFIG_DIR}" _config_dir EXPAND_TILDE)
  set(_config_file "${_config_dir}/TritonJITConfig.cmake")
  if(NOT EXISTS "${_config_file}")
    message(FATAL_ERROR
      "TritonJITConfig.cmake is missing below "
      "FLAGDNN_MTHREADS_TRITON_JIT_DIR")
  endif()
  file(REAL_PATH "${_config_file}" _config_file EXPAND_TILDE)
  file(REAL_PATH "${MTHREADS_JIT_ENVIRONMENT_REPORT}"
       _environment_report EXPAND_TILDE)
  file(READ "${_environment_report}" _environment_json)
  _flagdnn_mthreads_json_get(
    _reported_config "${_environment_json}"
    resources triton_jit_config realpath)
  _flagdnn_mthreads_json_get(
    _reported_library "${_environment_json}"
    resources triton_jit realpath)
  _flagdnn_mthreads_json_get(
    _reported_library_sha256 "${_environment_json}"
    resources triton_jit sha256)
  _flagdnn_mthreads_json_get(
    _reported_soname "${_environment_json}"
    resources triton_jit soname)
  _flagdnn_mthreads_json_get(
    _reported_header "${_environment_json}"
    resources triton_jit_header realpath)
  _flagdnn_mthreads_json_get(
    _reported_gen_ssig "${_environment_json}"
    resources triton_jit_gen_ssig realpath)
  _flagdnn_mthreads_json_get(
    _reported_standalone_compile "${_environment_json}"
    resources triton_jit_standalone_compile realpath)
  _flagdnn_mthreads_json_get(
    _reported_torch_init "${_environment_json}"
    python modules torch realpath)
  _flagdnn_mthreads_json_get(
    _reported_prefix "${_environment_json}"
    resources triton_jit_prefix realpath)
  _flagdnn_mthreads_json_get(
    _patchelf "${_environment_json}"
    resources patchelf realpath)

  if(NOT _config_file STREQUAL _reported_config)
    message(FATAL_ERROR
      "TritonJIT config differs from environment identity: "
      "${_config_file} != ${_reported_config}")
  endif()
  foreach(_required_file IN ITEMS
      "${_reported_library}"
      "${_reported_header}"
      "${_reported_gen_ssig}"
      "${_reported_standalone_compile}"
      "${_reported_torch_init}"
      "${_patchelf}")
    if(NOT EXISTS "${_required_file}")
      message(FATAL_ERROR
        "validated TritonJIT resource disappeared: ${_required_file}")
    endif()
  endforeach()
  file(SHA256 "${_reported_library}" _actual_library_sha256)
  if(NOT _actual_library_sha256 STREQUAL _reported_library_sha256)
    message(FATAL_ERROR
      "libtriton_jit hash differs from environment identity")
  endif()
  if(NOT _reported_soname STREQUAL "libtriton_jit.so")
    message(FATAL_ERROR
      "mthreads requires libtriton_jit.so SONAME, got "
      "'${_reported_soname}'")
  endif()

  # The upstream TritonJIT FindTorch module otherwise performs another
  # FindPython call and may replace this backend's selected interpreter with a
  # caller cache entry. Resolve Torch from the environment discovered with the
  # selected codegen interpreter; keep both hints local to this resolver.
  get_filename_component(
    _reported_torch_package "${_reported_torch_init}" DIRECTORY)
  set(Torch_ROOT "${_reported_torch_package}/share/cmake")
  set(Torch_DIR "${Torch_ROOT}/Torch")
  if(NOT EXISTS "${Torch_DIR}/TorchConfig.cmake")
    message(FATAL_ERROR
      "validated Torch CMake package disappeared: ${Torch_DIR}")
  endif()
  set(MUSA_HOME "${MTHREADS_JIT_MUSA_ROOT}")
  # Upstream build-tree exports use source-side FindTorch and a build-local fmt.
  if(EXISTS "${_config_dir}/CMakeCache.txt")
    list(PREPEND CMAKE_MODULE_PATH "${_reported_prefix}/cmake")
    list(PREPEND CMAKE_PREFIX_PATH "${_config_dir}/_deps/fmt-build")
  endif()
  # A caller's generic TritonJIT_DIR must not override this backend's selection.
  set(TritonJIT_DIR "${_config_dir}")
  find_package(
    TritonJIT 0.1.0 CONFIG REQUIRED
    PATHS "${_config_dir}"
    NO_DEFAULT_PATH)
  if(NOT TritonJIT_BACKEND STREQUAL MTHREADS_JIT_EXPECTED_BACKEND)
    message(FATAL_ERROR
      "TritonJIT backend must be MUSA, got '${TritonJIT_BACKEND}'")
  endif()
  if(TARGET TritonJIT::triton_jit)
    set(_jit_target TritonJIT::triton_jit)
  elseif(TARGET triton_jit)
    set(_jit_target triton_jit)
  else()
    message(FATAL_ERROR "TritonJIT package does not export a triton_jit target")
  endif()
  _flagdnn_mthreads_imported_location(
    _exported_library "${_jit_target}")
  if(NOT _exported_library STREQUAL _reported_library)
    message(FATAL_ERROR
      "TritonJIT imported target differs from environment identity: "
      "${_exported_library} != ${_reported_library}")
  endif()

  get_target_property(
    _jit_include_directories
    "${_jit_target}"
    INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _jit_include_directories)
    message(FATAL_ERROR
      "TritonJIT imported target has no public include directory")
  endif()
  if(NOT TARGET Torch::Torch)
    message(FATAL_ERROR
      "TritonJIT configuration did not define Torch::Torch")
  endif()
  if(NOT TARGET fmt::fmt-header-only)
    message(FATAL_ERROR
      "TritonJIT configuration did not define fmt::fmt-header-only")
  endif()
  get_target_property(
    _torch_include_directories
    Torch::Torch
    INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(
    _torch_compile_options
    Torch::Torch
    INTERFACE_COMPILE_OPTIONS)
  get_target_property(
    _torch_compile_definitions
    Torch::Torch
    INTERFACE_COMPILE_DEFINITIONS)
  if(NOT _torch_include_directories)
    message(FATAL_ERROR "Torch::Torch has no include directories")
  endif()
  if(_torch_compile_options STREQUAL "_torch_compile_options-NOTFOUND")
    set(_torch_compile_options)
  endif()
  if(_torch_compile_definitions STREQUAL
     "_torch_compile_definitions-NOTFOUND")
    set(_torch_compile_definitions)
  endif()

  file(MAKE_DIRECTORY "${MTHREADS_JIT_PRIVATE_ROOT}")
  file(REAL_PATH "${MTHREADS_JIT_PRIVATE_ROOT}" _private_root EXPAND_TILDE)
  # Mirror the installed layout below the backend binary directory.  The
  # plugin can therefore use the same exact origin-relative RUNPATH in both
  # build and install trees, without an automatically injected absolute SDK
  # path or an empty search component.
  set(_private_library_directory
      "${_private_root}/flagdnn/mthreads")
  set(_private_script_directory
      "${_private_root}/flagdnn/share/triton_jit/scripts")
  file(MAKE_DIRECTORY
    "${_private_library_directory}"
    "${_private_script_directory}")
  set(_private_library
      "${_private_library_directory}/libtriton_jit.so")
  file(COPY_FILE
    "${_reported_library}"
    "${_private_library}"
    ONLY_IF_DIFFERENT)
  file(COPY_FILE
    "${_reported_gen_ssig}"
    "${_private_script_directory}/gen_ssig.py"
    ONLY_IF_DIFFERENT)
  set(_standalone_patcher
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/patch_standalone_compile.py")
  if(NOT EXISTS "${_standalone_patcher}")
    message(FATAL_ERROR
      "mthreads TritonJIT standalone patcher is missing")
  endif()
  execute_process(
    COMMAND
      "${MTHREADS_JIT_CODEGEN_PYTHON}"
      "${_standalone_patcher}"
      "${_reported_standalone_compile}"
      "${_private_script_directory}/standalone_compile.py"
    RESULT_VARIABLE _standalone_patch_result
    OUTPUT_VARIABLE _standalone_patch_output
    ERROR_VARIABLE _standalone_patch_error)
  if(NOT _standalone_patch_result EQUAL 0)
    string(STRIP "${_standalone_patch_error}" _standalone_patch_error)
    message(FATAL_ERROR
      "cannot patch the private mthreads TritonJIT helper: "
      "${_standalone_patch_error}")
  endif()

  execute_process(
    COMMAND "${_patchelf}" --set-rpath "$ORIGIN" "${_private_library}"
    RESULT_VARIABLE _patchelf_result
    OUTPUT_VARIABLE _patchelf_output
    ERROR_VARIABLE _patchelf_error)
  if(NOT _patchelf_result EQUAL 0)
    string(STRIP "${_patchelf_error}" _patchelf_error)
    message(FATAL_ERROR
      "cannot set private libtriton_jit RUNPATH: ${_patchelf_error}")
  endif()
  execute_process(
    COMMAND "${_patchelf}" --print-rpath "${_private_library}"
    RESULT_VARIABLE _runpath_result
    OUTPUT_VARIABLE _private_runpath
    ERROR_VARIABLE _runpath_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _runpath_result EQUAL 0 OR
     NOT _private_runpath STREQUAL "$ORIGIN")
    message(FATAL_ERROR
      "private libtriton_jit RUNPATH must be exactly $ORIGIN")
  endif()
  file(SHA256 "${_private_library}" _private_library_sha256)

  if(NOT TARGET FlagDNNMthreads::triton_jit_private)
    add_library(
      FlagDNNMthreads::triton_jit_private
      SHARED IMPORTED GLOBAL)
    set_target_properties(
      FlagDNNMthreads::triton_jit_private
      PROPERTIES
        IMPORTED_LOCATION "${_private_library}"
        IMPORTED_SONAME "${_reported_soname}"
        INTERFACE_INCLUDE_DIRECTORIES
          "${_jit_include_directories};${_torch_include_directories}"
        INTERFACE_COMPILE_DEFINITIONS
          "BACKEND_MUSA=1;${_torch_compile_definitions}"
        INTERFACE_COMPILE_OPTIONS "${_torch_compile_options}"
        INTERFACE_LINK_LIBRARIES "fmt::fmt-header-only"
        FLAGDNN_MTHREADS_ORIGINAL_REALPATH "${_reported_library}"
        FLAGDNN_MTHREADS_ORIGINAL_SHA256 "${_reported_library_sha256}"
        FLAGDNN_MTHREADS_PRIVATE_SHA256 "${_private_library_sha256}"
        FLAGDNN_MTHREADS_SCRIPT_DIR "${_private_script_directory}")
  endif()

  install(
    FILES "${_private_library}"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/flagdnn/mthreads")
  install(
    FILES
      "${_private_script_directory}/gen_ssig.py"
      "${_private_script_directory}/standalone_compile.py"
    DESTINATION
      "${CMAKE_INSTALL_LIBDIR}/flagdnn/share/triton_jit/scripts")

  set(_input_files
      "${_config_file}"
      "${_reported_library}"
      "${_reported_header}"
      "${_reported_gen_ssig}"
      "${_reported_standalone_compile}"
      "${_standalone_patcher}"
      "${_environment_report}")
  foreach(_field IN ITEMS
      LIBRARY
      SONAME
      INCLUDE_DIR
      SCRIPT_DIR
      PROVENANCE_ROOT
      PROVENANCE_SHA256
      PRIVATE_SHA256
      PRIVATE_ROOT
      INPUT_FILES)
    unset(_output_value)
    if(_field STREQUAL "LIBRARY")
      set(_output_value "${_private_library}")
    elseif(_field STREQUAL "SONAME")
      set(_output_value "${_reported_soname}")
    elseif(_field STREQUAL "INCLUDE_DIR")
      set(_output_value "${_jit_include_directories}")
    elseif(_field STREQUAL "SCRIPT_DIR")
      set(_output_value "${_private_script_directory}")
    elseif(_field STREQUAL "PROVENANCE_ROOT")
      set(_output_value "${_reported_prefix}")
    elseif(_field STREQUAL "PROVENANCE_SHA256")
      set(_output_value "${_reported_library_sha256}")
    elseif(_field STREQUAL "PRIVATE_SHA256")
      set(_output_value "${_private_library_sha256}")
    elseif(_field STREQUAL "PRIVATE_ROOT")
      set(_output_value "${_private_root}")
    elseif(_field STREQUAL "INPUT_FILES")
      set(_output_value "${_input_files}")
    endif()
    set("${out_prefix}_${_field}" "${_output_value}" PARENT_SCOPE)
  endforeach()
endfunction()
