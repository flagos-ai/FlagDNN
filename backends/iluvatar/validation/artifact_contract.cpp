/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/iluvatar/artifact.hpp"
#include "backends/iluvatar/error.hpp"
#include "runtime/sha256.hpp"

#include <flagdnn/version.h>

#include <unistd.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::string_view kCompilerIdentity =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

void expect(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

void write_file(const std::filesystem::path &path, std::string_view bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create artifact fixture file");
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    throw std::runtime_error("cannot write artifact fixture file");
  }
}

std::string replace_once(std::string value, std::string_view before,
                         std::string_view after) {
  const std::size_t position = value.find(before);
  if (position == std::string::npos) {
    throw std::runtime_error("artifact mutation pattern was not found: " +
                             std::string(before));
  }
  value.replace(position, before.size(), after);
  return value;
}

std::string replace_last_once(std::string value, std::string_view before,
                              std::string_view after) {
  const std::size_t position = value.rfind(before);
  if (position == std::string::npos) {
    throw std::runtime_error("artifact mutation pattern was not found: " +
                             std::string(before));
  }
  value.replace(position, before.size(), after);
  return value;
}

std::string tensor_argument(std::int64_t uid, bool workspace) {
  std::ostringstream output;
  output << "{\"kind\":\"" << (workspace ? "workspace_tensor" : "tensor")
         << "\",\"uid\":" << uid << ",\"size\":16,\"alignment\":16";
  if (workspace) {
    output << ",\"workspace_offset\":" << (uid == 3 ? 0 : 16);
  }
  output << '}';
  return output.str();
}

std::string variant(std::string_view variant_id,
                    const std::vector<std::pair<std::int64_t, bool>> &uids) {
  std::ostringstream arguments;
  arguments << '[';
  for (std::size_t index = 0; index < uids.size(); ++index) {
    if (index != 0) {
      arguments << ',';
    }
    arguments << tensor_argument(uids[index].first, uids[index].second);
  }
  if (!uids.empty()) {
    arguments << ',';
  }
  arguments << "{\"kind\":\"scalar_i32\",\"value\":4}]";

  std::ostringstream output;
  output << "{\"variant_id\":\"" << variant_id
         << "\",\"full_signature\":\"add_kernel(*fp32,*fp32,*fp32,i32)\""
         << ",\"argument_count\":" << (uids.size() + 1)
         << ",\"arguments\":" << arguments.str()
         << ",\"compile_options\":{\"num_warps\":1,\"num_stages\":1}"
         << ",\"launch\":{\"grid\":[1,1,1],\"block\":[64,1,1],"
            "\"shared_memory\":0}}";
  return output.str();
}

class Fixture final {
public:
  Fixture()
      : directory_(std::filesystem::temp_directory_path() /
                   ("flagdnn-iluvatar-artifact-" +
                    std::to_string(static_cast<long long>(::getpid())))),
        outside_(directory_.parent_path() /
                 (directory_.filename().string() + "-outside.py")) {
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
    std::filesystem::remove(outside_, ignored);
    std::filesystem::create_directories(directory_);
    for (std::size_t index = 0; index < sources_.size(); ++index) {
      sources_[index] = "import triton\nimport triton.language as tl\n";
      const auto path =
          directory_ / ("generated_stage_" + std::to_string(index) + ".py");
      write_file(path, sources_[index]);
      hashes_[index] = flagdnn::native::sha256(sources_[index]);
    }
    write_file(outside_, sources_[0]);
    std::filesystem::create_symlink(outside_, directory_ / "linked.py");
  }

  ~Fixture() {
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
    std::filesystem::remove(outside_, ignored);
  }

  [[nodiscard]] std::string graph() const {
    std::ostringstream output;
    output << "{\"schema_version\":3,\"flagdnn_version\":\""
           << FLAGDNN_VERSION_STRING
           << "\",\"backend\":\"iluvatar\",\"target\":\"corex_71\""
           << ",\"compiler_identity\":\"" << kCompilerIdentity << "\""
           << ",\"build_options\":{\"heuristic_modes\":[\"A\"],"
              "\"autotune\":false},\"graph\":{\"name\":\"artifact contract\""
           << ",\"tensor_count\":5,\"tensors\":[" << graph_tensor(1, false)
           << ',' << graph_tensor(2, false) << ',' << graph_tensor(3, true)
           << ',' << graph_tensor(4, false) << ',' << graph_tensor(5, true)
           << "],\"node_count\":3,\"nodes\":[" << graph_node(0, 1, 2, 3) << ','
           << graph_node(1, 3, 2, 5) << ',' << graph_node(2, 3, 5, 4) << "]}}";
    return output.str();
  }

  [[nodiscard]] std::string manifest(std::string_view request_hash,
                                     bool incompatible_variant = false) const {
    std::ostringstream output;
    output << "{\"schema_version\":1,"
           << "\"artifact_kind\":\"flagdnn_execution_program\","
           << "\"flagdnn_version\":\"" << FLAGDNN_VERSION_STRING << "\","
           << "\"graph_ir_schema_version\":3,\"backend_abi_version\":2,"
           << "\"backend\":\"iluvatar\",\"target\":\"corex_71\","
           << "\"warp_size\":64,\"engine\":\"libtriton_jit\","
           << "\"request_sha256\":\"" << request_hash << "\","
           << "\"compiler\":{\"provider\":\"iluvatar_triton\","
           << "\"provider_version\":\"1\",\"identity_sha256\":\""
           << kCompilerIdentity << "\"},\"external_uids\":[1,2,4],"
           << "\"tensor_count\":5,\"tensors\":[" << manifest_tensor(1, false, 0)
           << ',' << manifest_tensor(2, false, 0) << ','
           << manifest_tensor(3, true, 0) << ',' << manifest_tensor(4, false, 0)
           << ',' << manifest_tensor(5, true, 16)
           << "],\"workspace\":{\"size\":256,\"alignment\":256},"
           << "\"program\":{\"schema_version\":1,\"stage_count\":3,"
           << "\"stages\":["
           << stage(0, "[]", {{1, false}, {2, false}, {3, true}},
                    incompatible_variant)
           << ',' << stage(1, "[0]", {{3, true}, {2, false}, {5, true}}, false)
           << ','
           << stage(2, "[0,1]", {{3, true}, {5, true}, {4, false}}, false)
           << "]}}";
    return output.str();
  }

  [[nodiscard]] flagdnn::iluvatar::ExecutionProgramArtifact
  load(const std::string &graph_ir, const std::string &manifest_bytes) const {
    write_file(directory_ / "manifest.json", manifest_bytes);
    const std::string request_hash = flagdnn::native::sha256(graph_ir);
    const std::string artifact_directory = directory_.string();
    flagdnnBackendBuildInputV2 input{};
    input.struct_size = sizeof(input);
    input.graph_ir = graph_ir.data();
    input.graph_ir_size = graph_ir.size();
    input.artifact_directory = artifact_directory.c_str();
    input.request_sha256 = request_hash.c_str();
    flagdnn::iluvatar::EngineBuildContext context{};
    context.target_fingerprint = "corex_71";
    context.device_identity =
        "corex_71-driver10020-00000000000000000000000000000000";
    return flagdnn::iluvatar::load_and_validate_artifact(context, input);
  }

  [[nodiscard]] const std::string &source_hash(std::size_t index) const {
    return hashes_.at(index);
  }

private:
  static std::string graph_tensor(std::int64_t uid, bool is_virtual) {
    std::ostringstream output;
    output << "{\"uid\":" << uid
           << ",\"data_type\":\"float32\",\"dimensions\":[4],"
              "\"strides\":[1],\"alignment\":16,\"virtual\":"
           << (is_virtual ? "true" : "false") << '}';
    return output.str();
  }

  static std::string manifest_tensor(std::int64_t uid, bool is_virtual,
                                     std::size_t offset) {
    std::ostringstream output;
    output << "{\"uid\":" << uid
           << ",\"data_type\":\"float32\",\"dimensions\":[4],"
              "\"strides\":[1],\"alignment\":16,\"virtual\":"
           << (is_virtual ? "true" : "false") << ",\"storage_size\":16";
    if (is_virtual) {
      output << ",\"workspace_offset\":" << offset;
    }
    output << '}';
    return output.str();
  }

  static std::string graph_node(std::size_t id, std::int64_t left,
                                std::int64_t right, std::int64_t output_uid) {
    std::ostringstream output;
    output << "{\"id\":" << id << ",\"type\":\"add\",\"name\":\"add_" << id
           << "\",\"compute_data_type\":\"float32\",\"inputs\":["
           << "{\"name\":\"a\",\"uid\":" << left << "},"
           << "{\"name\":\"b\",\"uid\":" << right
           << "}],\"outputs\":[{\"name\":\"output\",\"uid\":" << output_uid
           << "}],\"attributes\":{}}";
    return output.str();
  }

  [[nodiscard]] std::string
  stage(std::size_t id, std::string_view dependencies,
        const std::vector<std::pair<std::int64_t, bool>> &uids,
        bool incompatible_variant) const {
    const std::string base_variant = variant("default", uids);
    std::string variants = base_variant;
    if (incompatible_variant && id == 0) {
      variants += ',';
      variants += variant("bad_abi", {{1, false}, {2, false}, {5, true}});
    }
    std::ostringstream output;
    output << "{\"stage_id\":" << id << ",\"source_node_ids\":[" << id
           << "],\"dependencies\":" << dependencies
           << ",\"operation\":\"add\",\"kernel\":{"
           << "\"provider\":\"common_triton\",\"ownership\":\"common\","
           << "\"function\":\"add_kernel\",\"materialized_source\":{"
           << "\"path\":\"generated_stage_" << id
           << ".py\",\"size\":" << sources_.at(id).size() << ",\"sha256\":\""
           << hashes_.at(id) << "\"}},\"variants\":[" << variants << "]}";
    return output.str();
  }

  std::filesystem::path directory_;
  std::filesystem::path outside_;
  std::array<std::string, 3> sources_;
  std::array<std::string, 3> hashes_;
};

void expect_compilation_failure(const Fixture &fixture, std::string_view name,
                                const std::string &graph_ir,
                                const std::string &manifest) {
  try {
    (void)fixture.load(graph_ir, manifest);
  } catch (const flagdnn::iluvatar::IluvatarError &error) {
    expect(error.result() == FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
           std::string(name) + " returned the wrong backend result");
    return;
  }
  throw std::runtime_error(std::string(name) +
                           " artifact mutation unexpectedly passed");
}

} // namespace

int main() {
  try {
    Fixture fixture;
    const std::string graph_ir = fixture.graph();
    const std::string request_hash = flagdnn::native::sha256(graph_ir);
    const std::string valid = fixture.manifest(request_hash);
    const auto artifact = fixture.load(graph_ir, valid);
    expect(artifact.backend == "iluvatar", "backend was not retained");
    expect(artifact.target == "corex_71", "target was not retained");
    expect(artifact.engine == "libtriton_jit", "engine was not retained");
    expect(artifact.external_uids == std::vector<std::int64_t>({1, 2, 4}),
           "external UID set was not retained");
    expect(artifact.tensors.size() == 5 && artifact.stages.size() == 3,
           "artifact program content was not retained");
    expect(artifact.workspace_size == 256 &&
               artifact.workspace_alignment == 256,
           "workspace contract was not retained");

    std::string dead_virtual_graph =
        replace_once(graph_ir, R"json("name":"b","uid":5)json",
                     R"json("name":"b","uid":3)json");
    std::string dead_virtual_manifest = replace_last_once(
        valid, tensor_argument(5, true), tensor_argument(3, true));
    dead_virtual_manifest = replace_once(
        std::move(dead_virtual_manifest),
        R"json("stage_id":2,"source_node_ids":[2],"dependencies":[0,1])json",
        R"json("stage_id":2,"source_node_ids":[2],"dependencies":[0])json");
    dead_virtual_manifest =
        replace_once(std::move(dead_virtual_manifest), request_hash,
                     flagdnn::native::sha256(dead_virtual_graph));
    const auto dead_virtual_artifact =
        fixture.load(dead_virtual_graph, dead_virtual_manifest);
    expect(dead_virtual_artifact.stages.size() == 3,
           "dead virtual Graph output was not accepted");

    std::string provider_workspace_manifest =
        replace_once(valid, R"json("argument_count":4)json",
                     R"json("argument_count":5)json");
    provider_workspace_manifest = replace_once(
        std::move(provider_workspace_manifest),
        R"json({"kind":"scalar_i32","value":4})json",
        R"json({"kind":"workspace_tensor","uid":6,"size":16,"alignment":16,"workspace_offset":32},{"kind":"scalar_i32","value":4})json");
    const auto provider_workspace_artifact =
        fixture.load(graph_ir, provider_workspace_manifest);
    expect(provider_workspace_artifact.stages.front()
                   .variants.front()
                   .arguments.at(3)
                   .workspace_offset == 32,
           "provider-local workspace argument was not retained");

    const std::string scalar_graph = replace_once(
        graph_ir,
        R"json("uid":4,"data_type":"float32","dimensions":[4],"strides":[1])json",
        R"json("uid":4,"data_type":"float32","dimensions":[],"strides":[])json");
    const std::string scalar_hash = flagdnn::native::sha256(scalar_graph);
    std::string scalar_manifest = fixture.manifest(scalar_hash);
    scalar_manifest = replace_once(
        std::move(scalar_manifest),
        R"json("uid":4,"data_type":"float32","dimensions":[4],"strides":[1],"alignment":16,"virtual":false,"storage_size":16)json",
        R"json("uid":4,"data_type":"float32","dimensions":[],"strides":[],"alignment":16,"virtual":false,"storage_size":4)json");
    scalar_manifest =
        replace_once(std::move(scalar_manifest),
                     R"json("kind":"tensor","uid":4,"size":16)json",
                     R"json("kind":"tensor","uid":4,"size":4)json");
    const auto scalar_artifact = fixture.load(scalar_graph, scalar_manifest);
    const auto scalar_tensor = std::find_if(
        scalar_artifact.tensors.begin(), scalar_artifact.tensors.end(),
        [](const flagdnn::iluvatar::TensorArtifact &tensor) {
          return tensor.uid == 4;
        });
    expect(scalar_tensor != scalar_artifact.tensors.end() &&
               scalar_tensor->dimensions.empty() &&
               scalar_tensor->strides.empty() &&
               scalar_tensor->storage_size == 4,
           "rank-zero tensor contract was not retained");

    const auto fail = [&](std::string_view name, std::string mutated) {
      expect_compilation_failure(fixture, name, graph_ir, mutated);
    };
    const std::string partial_rank_graph = replace_once(
        graph_ir,
        R"json("uid":4,"data_type":"float32","dimensions":[4],"strides":[1])json",
        R"json("uid":4,"data_type":"float32","dimensions":[],"strides":[1])json");
    expect_compilation_failure(
        fixture, "partial rank-zero tensor", partial_rank_graph,
        fixture.manifest(flagdnn::native::sha256(partial_rank_graph)));

    fail("wrong backend", replace_once(valid, "\"backend\":\"iluvatar\"",
                                       "\"backend\":\"nvidia\""));
    fail("wrong target", replace_once(valid, "\"target\":\"corex_71\"",
                                      "\"target\":\"sm_71\""));
    fail("wrong engine", replace_once(valid, "\"engine\":\"libtriton_jit\"",
                                      "\"engine\":\"external_artifact\""));
    fail("wrong schema",
         replace_once(valid, "{\"schema_version\":1", "{\"schema_version\":2"));
    fail("wrong request hash",
         replace_once(valid, request_hash, std::string(64, '0')));
    fail("duplicate tensor UID", replace_once(valid, "{\"uid\":2,\"data_type\"",
                                              "{\"uid\":1,\"data_type\""));
    fail("missing tensor UID", replace_once(valid, "{\"uid\":2,\"data_type\"",
                                            "{\"uid\":99,\"data_type\""));
    fail("shape mismatch",
         replace_once(valid, "\"dimensions\":[4]", "\"dimensions\":[5]"));
    fail("stride mismatch",
         replace_once(valid, "\"strides\":[1]", "\"strides\":[2]"));
    fail("dtype mismatch", replace_once(valid, "\"data_type\":\"float32\"",
                                        "\"data_type\":\"float16\""));
    fail("external binding mismatch",
         replace_once(valid, "\"external_uids\":[1,2,4]",
                      "\"external_uids\":[1,4]"));
    fail("cyclic dependency",
         replace_once(valid,
                      "\"stage_id\":1,\"source_node_ids\":[1],"
                      "\"dependencies\":[0]",
                      "\"stage_id\":1,\"source_node_ids\":[1],"
                      "\"dependencies\":[1]"));
    fail("missing dependency",
         replace_once(valid,
                      "\"stage_id\":2,\"source_node_ids\":[2],"
                      "\"dependencies\":[0,1]",
                      "\"stage_id\":2,\"source_node_ids\":[2],"
                      "\"dependencies\":[1]"));
    fail("unsafe source path",
         replace_once(valid, "generated_stage_0.py", "../escape.py"));
    fail("source symlink escape",
         replace_once(valid, "generated_stage_0.py", "linked.py"));
    fail("wrong source size",
         replace_once(valid, "\"path\":\"generated_stage_0.py\",\"size\":43",
                      "\"path\":\"generated_stage_0.py\",\"size\":44"));
    fail("wrong source hash",
         replace_once(valid, fixture.source_hash(0), std::string(64, '0')));
    fail("argument count overflow", replace_once(valid, "\"argument_count\":4",
                                                 "\"argument_count\":4097"));
    fail("stage count overflow",
         replace_once(valid, "\"stage_count\":3", "\"stage_count\":65537"));
    fail("workspace overflow", replace_once(valid, "\"workspace_offset\":16",
                                            "\"workspace_offset\":248"));
    fail("workspace overlap", replace_once(valid, "\"workspace_offset\":16",
                                           "\"workspace_offset\":0"));
    fail("workspace alignment", replace_once(valid,
                                             "\"workspace\":{\"size\":256,"
                                             "\"alignment\":256}",
                                             "\"workspace\":{\"size\":256,"
                                             "\"alignment\":3}"));
    expect_compilation_failure(fixture, "provider workspace overlap", graph_ir,
                               replace_once(provider_workspace_manifest,
                                            R"json("workspace_offset":32)json",
                                            R"json("workspace_offset":0)json"));
    expect_compilation_failure(fixture, "variant ABI mismatch", graph_ir,
                               fixture.manifest(request_hash, true));

    std::cout << "PASS Iluvatar artifact schema contract\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
