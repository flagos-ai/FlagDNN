/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>

#include <array>
#include <barrier>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

class TemporaryCache {
 public:
  TemporaryCache() {
    const auto pattern = (std::filesystem::temp_directory_path() /
                          "flagdnn-build-environment-XXXXXX").string();
    std::vector<char> name(pattern.begin(), pattern.end());
    name.push_back('\0');
    const char* created = mkdtemp(name.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path = created;
  }
  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
  TemporaryCache(const TemporaryCache&) = delete;
  TemporaryCache& operator=(const TemporaryCache&) = delete;
  std::filesystem::path path;
};

void set_environment(const char* name, const char* value) {
  if (setenv(name, value, 1) != 0) {
    throw std::runtime_error("cannot configure test environment");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      throw std::runtime_error("expected compiler and preparation mode");
    }
    const std::string mode(argv[2]);
    set_environment("FLAGDNN_CONTRACT_PREPARATION", mode.c_str());
    TemporaryCache cache_a;
    TemporaryCache cache_b;
    flagdnn::Handle handle_a("contract", 0);
    flagdnn::Handle handle_b("contract", 0);
    handle_a.set_compiler(argv[1], "contract", cache_a.path.string());
    handle_b.set_compiler(argv[1], "contract", cache_b.path.string());
    constexpr std::array<std::int64_t, 1> dimensions{8}, strides{1};
    flagdnn::TensorDescriptor input(1, FLAGDNN_DATA_FLOAT32, dimensions, strides);
    flagdnn::TensorDescriptor output(2, FLAGDNN_DATA_FLOAT32, dimensions, strides);
    flagdnn::Graph graph;
    graph.relu(input, output);
    graph.finalize();
    if (mode != "parallel" && mode != "parallel_handles" &&
        mode != "parallel_cold_handles") {
      bool rejected = false;
      try {
        flagdnn::Executable executable(handle_a, graph);
      } catch (const flagdnn::Error& error) {
        rejected = error.status() == FLAGDNN_STATUS_INTERNAL_ERROR;
      }
      if (!rejected) {
        throw std::runtime_error("invalid preparation was not rejected");
      }
      // Failed initialization must be retryable, not permanently poison builds.
      set_environment("FLAGDNN_CONTRACT_PREPARATION", "success");
      flagdnn::Executable recovered(handle_a, graph);
    } else {
      if (mode == "parallel_handles") {
        flagdnn::Executable warm_environment(handle_a, graph);
      }
      set_environment("FLAGDNN_CONTRACT_PARALLEL_BUILDS", "1");
      std::barrier start(2);
      std::array<std::exception_ptr, 2> errors{};
      const auto build = [&](flagdnn::Handle& handle, std::size_t index) {
        start.arrive_and_wait();
        try {
          flagdnn::Executable executable(handle, graph);
        } catch (...) {
          errors[index] = std::current_exception();
        }
      };
      std::thread first([&] { build(handle_a, 0); });
      std::thread second([&] {
        build(mode == "parallel" ? handle_a : handle_b, 1);
      });
      first.join();
      second.join();
      for (const auto& error : errors) {
        if (error) {
          std::rethrow_exception(error);
        }
      }
    }
    std::cout << "PASS prepared environment " << mode << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
