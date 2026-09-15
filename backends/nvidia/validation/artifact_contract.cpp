/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/nvidia/artifact.hpp"
#include "runtime/sha256.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

// Host-only artifact boundary probe. No CUDA context or kernel is created.
int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("expected fixture directory");
    }
    std::ifstream request_file(std::filesystem::path(argv[1]) / "request.json");
    const std::string request{std::istreambuf_iterator<char>(request_file), {}};
    const auto hash = flagdnn::native::sha256(request);
    const flagdnnBackendBuildInputV2 input{
        sizeof(flagdnnBackendBuildInputV2), request.data(), request.size(), argv[1], hash.c_str()};
    flagdnn::cuda::EngineBuildContext context;
    context.target_fingerprint = "sm_90";
    context.device_identity = "host-contract";
    const auto artifact = flagdnn::cuda::parse_cuda_artifact(context, input);
    std::cout << "PASS stages=" << artifact.stages.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
