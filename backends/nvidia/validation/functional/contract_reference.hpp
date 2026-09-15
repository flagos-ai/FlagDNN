/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_CONTRACT_REFERENCE_HPP_
#define FLAGDNN_NVIDIA_CONTRACT_REFERENCE_HPP_
#include <algorithm>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "validation/cuda_driver.hpp"
#include "validation/functional/cudnn_graph.hpp"
namespace flagdnn::testing::cuda {
inline TestTensor contract_tensor(std::int64_t uid, flagdnnDataType_t type,
                                  std::span<const std::int64_t> dimensions,
                                  std::span<const std::int64_t> strides) {
  return {uid,
          type,
          {dimensions.begin(), dimensions.end()},
          {strides.begin(), strides.end()}};
}
// Run a cuDNN comparator on the contract's existing device inputs. The output
// allocation is private so the comparator cannot overwrite FlagDNN's result.
template <class Value>
std::vector<Value> run_contract_reference(
    TestExecutable& executable, std::span<const flagdnnBinding_t> bindings,
    std::int64_t output_uid, std::size_t count, CUstream stream,
    std::span<const Value> initial = {}) {
  static_assert(std::is_trivially_copyable_v<Value>);
  if (!initial.empty() && initial.size() != count)
    throw std::invalid_argument(
        "cuDNN contract output initialization size mismatch");
  std::vector<Value> result(count);
  if (!initial.empty())
    std::copy(initial.begin(), initial.end(), result.begin());
  DeviceBuffer output(result.size() * sizeof(Value));
  DeviceBuffer workspace(executable.workspace_size());
  output.copy_from_host(result.data(), result.size() * sizeof(Value), stream);
  std::vector<flagdnnBinding_t> pointers(bindings.begin(), bindings.end());
  bool found = false;
  for (auto& binding : pointers) {
    if (binding.uid == output_uid) {
      binding.device_pointer = output.opaque();
      found = true;
    }
  }
  if (!found) pointers.push_back({output_uid, output.opaque()});
  executable.execute(pointers, workspace.opaque(), executable.workspace_size(),
                     reinterpret_cast<flagdnnStream_t>(stream));
  output.copy_to_host(result.data(), result.size() * sizeof(Value), stream);
  check_cuda(cuStreamSynchronize(stream),
             "cuDNN contract reference synchronize");
  return result;
}
}  // namespace flagdnn::testing::cuda
#endif
