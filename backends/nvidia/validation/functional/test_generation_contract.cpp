/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <flagdnn/flagdnn.hpp>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "validation/cuda_driver.hpp"
#include "validation/functional/contract_reference.hpp"
#include "validation/functional/cudnn_extended.hpp"
namespace {
using namespace flagdnn::validation::nvidia;
struct Cache {
  std::filesystem::path path;
  ~Cache() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};
void check_generation(flagdnn::Handle& handle, Stream& stream,
                      flagdnnDataType_t type) {
  constexpr std::size_t extent = 17, count = 49, byte_offset = 16;
  const std::array<std::int64_t, 1> shape{extent}, strides{1};
  flagdnn::TensorDescriptor output(1, type, shape, strides);
  flagdnn::OperationDescriptor operation("gen_index");
  operation.set_output("output", output);
  operation.set_attribute("axis", std::int64_t{0});
  // The generic descriptor API makes compute dtype optional, including for
  // operations that have no input from which to infer it.
  operation.finalize();
  flagdnn::Graph graph;
  graph.add(operation);
  graph.finalize();
  flagdnn::Executable executable(handle, graph);
  std::vector<std::uint32_t> values(count, 0x5A5A5A5AU);
  DeviceBuffer device(byte_offset + values.size() * sizeof(values[0]));
  device.copy_from_host_at(values.data(), values.size() * sizeof(values[0]),
                           byte_offset, stream.get());
  DeviceBuffer workspace(executable.workspace_size());
  const std::array<flagdnnBinding_t, 1> bindings{
      {{1, device.opaque_at(byte_offset)}}};
  executable.execute(bindings, workspace.opaque(), executable.workspace_size(),
                     stream.opaque());
  device.copy_to_host_at(values.data(), values.size() * sizeof(values[0]),
                         byte_offset, stream.get());
  stream.synchronize();
  flagdnn::testing::IndexTestCase reference_case;
  reference_case.name = "generic_gen_index_cudnn";
  reference_case.operation = "gen_index";
  reference_case.output = {1, type, {extent}, {1}};
  auto reference = flagdnn::testing::build_cudnn_index(reference_case);
  const auto expected =
      flagdnn::testing::cuda::run_contract_reference<std::uint32_t>(
          *reference, bindings, 1, extent, stream.get());
  if (!std::equal(expected.begin(), expected.end(), values.begin()))
    throw std::runtime_error("generic gen_index differs from cuDNN");
  for (std::size_t i = extent; i < count; ++i)
    if (values[i] != 0x5A5A5A5AU)
      throw std::runtime_error("generic gen_index overwrote output guard");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("expected compiler executable and entry");
    char pattern[] = "/tmp/flagdnn-generation-contract-XXXXXX";
    const char* directory = mkdtemp(pattern);
    if (!directory) throw std::runtime_error("mkdtemp failed");
    Cache cache{directory};
    DriverContext context;
    Stream stream;
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path.string());
    for (auto type : {FLAGDNN_DATA_INT32, FLAGDNN_DATA_FLOAT32})
      check_generation(handle, stream, type);
    std::cout << "PASS generic input-free descriptor inference, output offset "
                 "and guard\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
