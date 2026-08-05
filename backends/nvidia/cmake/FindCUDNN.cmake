include(FindPackageHandleStandardArgs)

set(CUDNN_ROOT "" CACHE PATH "Root directory of a cuDNN installation")
set(CUDNN_FRONTEND_INCLUDE_DIR "" CACHE PATH
    "Directory containing cudnn_frontend.h")

# Python wheels commonly install the cuDNN runtime and cuDNN Frontend headers
# under different roots. Keep that platform-specific layout knowledge here,
# while preserving explicit CMake cache values and CUDNN_ROOT as higher
# priority inputs.
if(FLAGDNN_CODEGEN_PYTHON AND NOT CUDNN_ROOT AND
   "$ENV{CUDNN_ROOT}" STREQUAL "" AND
   (NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY OR
    NOT CUDNN_FRONTEND_INCLUDE_DIR))
  execute_process(
    COMMAND "${FLAGDNN_CODEGEN_PYTHON}" -c [=[
from pathlib import Path
import site

roots = []
for value in [*site.getsitepackages(), site.getusersitepackages()]:
    path = Path(value)
    if path not in roots:
        roots.append(path)

cudnn_root = next(
    (root / "nvidia" / "cudnn" for root in roots
     if (root / "nvidia" / "cudnn" / "include" / "cudnn.h").is_file()),
    None,
)
frontend = next(
    (root / "include" for root in roots
     if (root / "include" / "cudnn_frontend.h").is_file()),
    None,
)
library = None
if cudnn_root is not None:
    for name in ("libcudnn.so.9", "libcudnn.so"):
        candidate = cudnn_root / "lib" / name
        if candidate.exists():
            library = candidate
            break

values = (
    cudnn_root / "include" if cudnn_root is not None else "",
    library or "",
    frontend or "",
)
print("|".join(str(value) for value in values))
]=]
    RESULT_VARIABLE _flagdnn_cudnn_wheel_result
    OUTPUT_VARIABLE _flagdnn_cudnn_wheel_output
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(_flagdnn_cudnn_wheel_result EQUAL 0)
    string(REPLACE "|" ";" _flagdnn_cudnn_wheel_paths
      "${_flagdnn_cudnn_wheel_output}")
    list(LENGTH _flagdnn_cudnn_wheel_paths
      _flagdnn_cudnn_wheel_path_count)
    if(_flagdnn_cudnn_wheel_path_count EQUAL 3)
      list(GET _flagdnn_cudnn_wheel_paths 0
        _flagdnn_cudnn_wheel_include)
      list(GET _flagdnn_cudnn_wheel_paths 1
        _flagdnn_cudnn_wheel_library)
      list(GET _flagdnn_cudnn_wheel_paths 2
        _flagdnn_cudnn_wheel_frontend)
      if(NOT CUDNN_INCLUDE_DIR AND
         EXISTS "${_flagdnn_cudnn_wheel_include}/cudnn.h")
        set(CUDNN_INCLUDE_DIR "${_flagdnn_cudnn_wheel_include}"
          CACHE PATH "Directory containing cudnn.h" FORCE)
      endif()
      if(NOT CUDNN_LIBRARY AND
         EXISTS "${_flagdnn_cudnn_wheel_library}")
        set(CUDNN_LIBRARY "${_flagdnn_cudnn_wheel_library}"
          CACHE FILEPATH "Path to the cuDNN shared library" FORCE)
      endif()
      if(NOT CUDNN_FRONTEND_INCLUDE_DIR AND
         EXISTS "${_flagdnn_cudnn_wheel_frontend}/cudnn_frontend.h")
        set(CUDNN_FRONTEND_INCLUDE_DIR
          "${_flagdnn_cudnn_wheel_frontend}"
          CACHE PATH "Directory containing cudnn_frontend.h" FORCE)
      endif()
    endif()
  endif()
endif()

find_path(CUDNN_INCLUDE_DIR
  NAMES cudnn.h
  HINTS
    "${CUDNN_ROOT}"
    "$ENV{CUDNN_ROOT}"
  PATH_SUFFIXES include)

find_library(CUDNN_LIBRARY
  NAMES cudnn
  HINTS
    "${CUDNN_ROOT}"
    "$ENV{CUDNN_ROOT}"
  PATH_SUFFIXES lib lib64)

if(NOT CUDNN_LIBRARY)
  find_file(CUDNN_LIBRARY
    NAMES libcudnn.so.9
    HINTS
      "${CUDNN_ROOT}"
      "$ENV{CUDNN_ROOT}"
    PATH_SUFFIXES lib lib64)
endif()

if(NOT CUDNN_FRONTEND_INCLUDE_DIR)
  find_path(CUDNN_FRONTEND_INCLUDE_DIR
    NAMES cudnn_frontend.h
    HINTS
      "${CUDNN_ROOT}"
      "$ENV{CUDNN_ROOT}"
    PATH_SUFFIXES include)
endif()

find_package_handle_standard_args(CUDNN
  REQUIRED_VARS
    CUDNN_INCLUDE_DIR
    CUDNN_LIBRARY
    CUDNN_FRONTEND_INCLUDE_DIR)

if(CUDNN_FOUND AND NOT TARGET CUDNN::cudnn)
  add_library(CUDNN::cudnn SHARED IMPORTED)
  set_target_properties(CUDNN::cudnn PROPERTIES
    IMPORTED_LOCATION "${CUDNN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}")
endif()

if(CUDNN_FOUND AND NOT TARGET CUDNN::frontend)
  add_library(CUDNN::frontend INTERFACE IMPORTED)
  set_target_properties(CUDNN::frontend PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_FRONTEND_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES CUDNN::cudnn)
endif()

mark_as_advanced(
  CUDNN_INCLUDE_DIR
  CUDNN_LIBRARY
  CUDNN_FRONTEND_INCLUDE_DIR)
