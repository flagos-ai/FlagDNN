/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <unistd.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fe = ::flagdnn_frontend;

void check_frontend(fe::error_t status, const char* operation) {
  if (status.is_bad()) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + status.get_message());
  }
}

class TemporaryCache {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-add-jit-build-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
};

fe::graph::Graph::Tensor make_tensor(fe::graph::Graph& graph,
                                     const char* name,
                                     std::int64_t uid) {
  return graph.tensor(fe::graph::Tensor_attributes()
                          .set_name(name)
                          .set_uid(uid)
                          .set_data_type(fe::DataType_t::FLOAT)
                          .set_dim({1024})
                          .set_stride({1}));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: native_nvidia_add_jit_build_smoke COMPILER_EXECUTABLE COMPILER_ENTRY\n";
      return 2;
    }
    if (setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
      throw std::runtime_error("cannot select libtriton_jit engine");
    }

    TemporaryCache cache;
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());

    fe::graph::Graph graph;
    graph.set_name("add_jit_build_smoke")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(true);
    const auto left = make_tensor(graph, "left", 1);
    const auto right = make_tensor(graph, "right", 2);
    auto output = graph.pointwise(
        left,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT)
            .set_alpha(1.0));
    output->set_name("output")
        .set_uid(3)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({1024})
        .set_stride({1})
        .set_output(true);

    check_frontend(graph.build(handle, {fe::HeurMode_t::A}),
                   "FlagDNN Add JIT graph build");

    std::size_t manifest_count = 0;
    std::size_t selection_count = 0;
    std::size_t cubin_count = 0;
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(cache.path())) {
      if (!entry.is_regular_file()) {
        continue;
      }
      if (entry.path().filename() == "manifest.json") {
        ++manifest_count;
      }
      if (entry.path().filename().string().starts_with(
              ".flagdnn-autotune-v1-stage-")) {
        ++selection_count;
      }
      if (entry.path().extension() == ".cubin") {
        ++cubin_count;
      }
    }
    if (manifest_count != 1 || selection_count != 1 || cubin_count != 0) {
      throw std::runtime_error(
          "Add JIT build did not produce one manifest/selection and zero cubins");
    }

    std::cout << "PASS Add C++ frontend -> JIT -> autotune build chain; "
              << "workspace=" << graph.get_workspace_size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
