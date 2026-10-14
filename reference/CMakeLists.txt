add_subdirectory(cpu)

add_executable(flagdnn_reference_cpu_pointwise_contract
  tests/cpu_pointwise_contract.cpp)
target_link_libraries(flagdnn_reference_cpu_pointwise_contract PRIVATE
  flagdnn_validation_cpu_reference)
flagdnn_enable_warnings(flagdnn_reference_cpu_pointwise_contract)
add_test(NAME reference.cpu.pointwise.contract
  COMMAND flagdnn_reference_cpu_pointwise_contract)
set_tests_properties(reference.cpu.pointwise.contract PROPERTIES
  LABELS "reference;cpu;pointwise;host;contract")

foreach(_reference_contract IN ITEMS index storage normalization)
  add_executable(flagdnn_reference_cpu_${_reference_contract}_contract
    tests/cpu_${_reference_contract}_contract.cpp)
  target_link_libraries(flagdnn_reference_cpu_${_reference_contract}_contract
    PRIVATE flagdnn_validation_cpu_reference)
  flagdnn_enable_warnings(flagdnn_reference_cpu_${_reference_contract}_contract)
  add_test(NAME reference.cpu.${_reference_contract}.contract
    COMMAND flagdnn_reference_cpu_${_reference_contract}_contract)
  set_tests_properties(reference.cpu.${_reference_contract}.contract PROPERTIES
    LABELS "reference;cpu;${_reference_contract};host;contract")
endforeach()

# The CPU reference is shared by non-NVIDIA validation builds, so keep its
# cuDNN oracle test conditional on the NVIDIA SDK already selected by the
# build. This preserves Hygon-only and Ascend-only configurations.
if("nvidia" IN_LIST FLAGDNN_BACKENDS)
  add_subdirectory(tests)
endif()
