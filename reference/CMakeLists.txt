add_subdirectory(cpu)

# The CPU reference is shared by non-NVIDIA validation builds, so keep its
# cuDNN oracle test conditional on the NVIDIA SDK already selected by the
# build. This preserves Hygon-only and Ascend-only configurations.
if("nvidia" IN_LIST FLAGDNN_BACKENDS)
  add_subdirectory(tests)
endif()
