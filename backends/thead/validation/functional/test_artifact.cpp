// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backends/thead/artifact.hpp"
#include "backends/thead/error.hpp"
#include "runtime/sha256.hpp"

#include <flagdnn/version.h>

#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::string_view kCompilerIdentity =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr std::string_view kCandidateIdentity =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

void require(bool condition, std::string message) {
  if (!condition) {
    throw std::runtime_error(std::move(message));
  }
}

void write_file(const std::filesystem::path& path, std::string_view bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create artifact fixture file");
  }
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output) {
    throw std::runtime_error("cannot write artifact fixture file");
  }
}

std::string replace_once(std::string value,
                         std::string_view before,
                         std::string_view after) {
  const std::size_t position = value.find(before);
  if (position == std::string::npos) {
    throw std::runtime_error("artifact mutation pattern was not found: " +
                             std::string(before));
  }
  value.replace(position, before.size(), after);
  return value;
}

std::string tensor_argument(std::int64_t uid, bool workspace) {
  std::ostringstream output;
  output << "{\"kind\":\""
         << (workspace ? "workspace_tensor" : "tensor") << "\",\"uid\":"
         << uid << ",\"size\":16,\"alignment\":16";
  if (workspace) {
    output << ",\"workspace_offset\":" << (uid == 3 ? 0 : 16);
  }
  output << '}';
  return output.str();
}

std::string variant(
    std::string_view variant_id,
    const std::vector<std::pair<std::int64_t, bool>>& tensor_uids,
    unsigned int block = 64) {
  std::ostringstream output;
  output << "{\"variant_id\":\"" << variant_id
         << "\",\"full_signature\":"
            "\"*fp32:16,*fp32:16,*fp32:16,i32,1,1.0,64\","
         << "\"argument_count\":" << tensor_uids.size() + 1
         << ",\"arguments\":[";
  for (std::size_t index = 0; index < tensor_uids.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << tensor_argument(tensor_uids[index].first,
                              tensor_uids[index].second);
  }
  output << ",{\"kind\":\"scalar_i32\",\"name\":\"n_elements\","
            "\"value\":4}],"
         << "\"compile_options\":{\"num_warps\":1,\"num_stages\":1,"
            "\"maxnreg\":null,\"ppu_compiler_options\":{}},"
         << "\"launch\":{\"grid\":[1,1,1],\"block\":[" << block
         << ",1,1],\"shared_memory\":0}}";
  return output.str();
}

class Fixture final {
 public:
  Fixture() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-thead-artifact-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = ::mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed for artifact fixture");
    }
    directory_ = created;
    outside_ = directory_.parent_path() /
               (directory_.filename().string() + "-outside.py");
    for (std::size_t index = 0; index < sources_.size(); ++index) {
      sources_[index] =
          "import triton\nimport triton.language as tl\n# stage " +
          std::to_string(index) + "\n";
      const auto source = directory_ /
                          ("generated_stage_" + std::to_string(index) + ".py");
      write_file(source, sources_[index]);
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

  Fixture(const Fixture&) = delete;
  Fixture& operator=(const Fixture&) = delete;

  [[nodiscard]] std::string graph() const {
    std::ostringstream output;
    output << "{\"schema_version\":3,\"flagdnn_version\":\""
           << FLAGDNN_VERSION_STRING
           << "\",\"backend\":\"thead\","
              "\"target\":\"ppu_contract_cc80\","
              "\"compiler_identity\":\""
           << kCompilerIdentity
           << "\",\"build_options\":{\"heuristic_modes\":[\"A\"],"
              "\"autotune\":true},\"graph\":{\"name\":\"artifact contract\","
              "\"tensor_count\":5,\"tensors\":["
           << graph_tensor(1, false) << ',' << graph_tensor(2, false) << ','
           << graph_tensor(3, true) << ',' << graph_tensor(4, false) << ','
           << graph_tensor(5, true)
           << "],\"node_count\":3,\"nodes\":[" << graph_node(0, 1, 2, 3)
           << ',' << graph_node(1, 3, 2, 5) << ','
           << graph_node(2, 3, 5, 4) << "]}}";
    return output.str();
  }

  [[nodiscard]] std::string manifest(std::string_view request_hash) const {
    std::ostringstream output;
    output << "{\"schema_version\":1,"
              "\"artifact_kind\":\"flagdnn_execution_program\","
              "\"flagdnn_version\":\""
           << FLAGDNN_VERSION_STRING
           << "\",\"graph_ir_schema_version\":3,\"backend_abi_version\":2,"
              "\"backend\":\"thead\",\"target\":\"ppu_contract_cc80\","
              "\"warp_size\":32,\"engine\":\"libtriton_jit\","
              "\"request_sha256\":\""
           << request_hash
           << "\",\"compiler\":{\"provider\":\"thead_triton\","
              "\"provider_version\":\"1\",\"identity_sha256\":\""
           << kCompilerIdentity
           << "\"},\"external_uids\":[1,2,4],\"tensor_count\":5,"
              "\"tensors\":["
           << manifest_tensor(1, false, 0) << ','
           << manifest_tensor(2, false, 0) << ','
           << manifest_tensor(3, true, 0) << ','
           << manifest_tensor(4, false, 0) << ','
           << manifest_tensor(5, true, 16)
           << "],\"workspace\":{\"size\":256,\"alignment\":256},"
              "\"program\":{\"schema_version\":1,\"stage_count\":3,"
              "\"stages\":["
           << stage(0, "[]", {{1, false}, {2, false}, {3, true}}, false)
           << ','
           << stage(1, "[0]", {{3, true}, {2, false}, {5, true}}, true)
           << ','
           << stage(2, "[0,1]", {{3, true}, {5, true}, {4, false}}, false)
           << "]}}";
    return output.str();
  }

  [[nodiscard]] std::string fused_manifest(
      std::string_view request_hash) const {
    std::ostringstream output;
    output << "{\"schema_version\":1,"
              "\"artifact_kind\":\"flagdnn_execution_program\","
              "\"flagdnn_version\":\""
           << FLAGDNN_VERSION_STRING
           << "\",\"graph_ir_schema_version\":3,\"backend_abi_version\":2,"
              "\"backend\":\"thead\",\"target\":\"ppu_contract_cc80\","
              "\"warp_size\":32,\"engine\":\"libtriton_jit\","
              "\"request_sha256\":\""
           << request_hash
           << "\",\"compiler\":{\"provider\":\"thead_triton\","
              "\"provider_version\":\"1\",\"identity_sha256\":\""
           << kCompilerIdentity
           << "\"},\"external_uids\":[1,2,4],\"tensor_count\":5,"
              "\"tensors\":["
           << manifest_tensor(1, false, std::nullopt) << ','
           << manifest_tensor(2, false, std::nullopt) << ','
           << manifest_tensor(3, true, std::nullopt) << ','
           << manifest_tensor(4, false, std::nullopt) << ','
           << manifest_tensor(5, true, std::nullopt)
           << "],\"workspace\":{\"size\":0,\"alignment\":256},"
              "\"program\":{\"schema_version\":1,\"stage_count\":1,"
              "\"stages\":["
           << fused_stage() << "]}}";
    return output.str();
  }

  [[nodiscard]] flagdnn::thead::ExecutionProgramArtifact load(
      const std::string& graph_ir, const std::string& manifest_bytes) const {
    write_file(directory_ / "manifest.json", manifest_bytes);
    const std::string artifact_directory = directory_.string();
    const std::string request_hash = flagdnn::native::sha256(graph_ir);
    flagdnnBackendBuildInputV2 input{};
    input.struct_size = sizeof(input);
    input.graph_ir = graph_ir.data();
    input.graph_ir_size = graph_ir.size();
    input.artifact_directory = artifact_directory.c_str();
    input.request_sha256 = request_hash.c_str();
    flagdnn::thead::EngineBuildContext context{};
    context.target_fingerprint = "ppu_contract_cc80";
    context.device_identity = "ppu_contract_cc80-sdk2-driver0-pci0-uuid0";
    return flagdnn::thead::load_and_validate_artifact(context, input);
  }

  [[nodiscard]] const std::string& source_hash(std::size_t index) const {
    return hashes_.at(index);
  }

  [[nodiscard]] std::size_t source_size(std::size_t index) const {
    return sources_.at(index).size();
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

  static std::string manifest_tensor(std::int64_t uid,
                                     bool is_virtual,
                                     std::optional<std::size_t> offset) {
    std::ostringstream output;
    output << "{\"uid\":" << uid
           << ",\"data_type\":\"float32\",\"dimensions\":[4],"
              "\"strides\":[1],\"alignment\":16,\"virtual\":"
           << (is_virtual ? "true" : "false")
           << ",\"storage_size\":16";
    if (is_virtual && offset.has_value()) {
      output << ",\"workspace_offset\":" << *offset;
    }
    output << '}';
    return output.str();
  }

  static std::string graph_node(std::size_t id,
                                std::int64_t left,
                                std::int64_t right,
                                std::int64_t output_uid) {
    std::ostringstream output;
    output << "{\"id\":" << id
           << ",\"type\":\"add\",\"name\":\"add_" << id
           << "\",\"compute_data_type\":\"float32\",\"inputs\":["
              "{\"name\":\"left\",\"uid\":"
           << left << "},{\"name\":\"right\",\"uid\":" << right
           << "}],\"outputs\":[{\"name\":\"output\",\"uid\":"
           << output_uid << "}],\"attributes\":{\"alpha\":1}}";
    return output.str();
  }

  [[nodiscard]] std::string stage(
      std::size_t id,
      std::string_view dependencies,
      const std::vector<std::pair<std::int64_t, bool>>& tensor_uids,
      bool autotune) const {
    std::ostringstream variants;
    variants << variant("default", tensor_uids);
    if (autotune) {
      variants << ',' << variant("block128", tensor_uids, 128);
    }
    std::ostringstream output;
    output << "{\"stage_id\":" << id << ",\"source_node_ids\":[" << id
           << "],\"dependencies\":" << dependencies
           << ",\"operation\":\"add\",\"kernel\":{"
              "\"provider\":\"common_triton\",\"ownership\":\"common\","
              "\"function\":\"binary_contiguous_kernel\","
              "\"registry_sha256\":\""
           << kCandidateIdentity
           << "\","
              "\"materialized_source\":{\"path\":\"generated_stage_"
           << id << ".py\",\"size\":" << sources_.at(id).size()
           << ",\"sha256\":\"" << hashes_.at(id)
           << "\"}},\"variants\":[" << variants.str() << "],\"tuning\":";
    if (autotune) {
      output << "{\"warmup\":2,\"repetitions\":3,"
                "\"candidate_identity\":\""
             << kCandidateIdentity
             << "\",\"selection_cache\":\"selection/stage_1.json\"}";
    } else {
      output << "null";
    }
    output << '}';
    return output.str();
  }

  [[nodiscard]] std::string fused_stage() const {
    const std::vector<std::pair<std::int64_t, bool>> tensor_uids = {
        {1, false}, {2, false}, {4, false}};
    std::ostringstream output;
    output << "{\"stage_id\":0,\"source_node_ids\":[0,1,2],"
              "\"dependencies\":[],\"operation\":\"add\",\"kernel\":{"
              "\"provider\":\"common_triton\",\"ownership\":\"common\","
              "\"function\":\"binary_contiguous_kernel\","
              "\"registry_sha256\":\""
           << kCandidateIdentity
           << "\",\"materialized_source\":{\"path\":"
              "\"generated_stage_0.py\",\"size\":"
           << sources_.at(0).size() << ",\"sha256\":\"" << hashes_.at(0)
           << "\"}},\"variants\":[" << variant("default", tensor_uids)
           << "],\"tuning\":null}";
    return output.str();
  }

  std::filesystem::path directory_;
  std::filesystem::path outside_;
  std::array<std::string, 3> sources_;
  std::array<std::string, 3> hashes_;
};

void require_compilation_failure(const Fixture& fixture,
                                 std::string_view name,
                                 const std::string& graph_ir,
                                 const std::string& manifest,
                                 std::string_view expected_detail) {
  try {
    (void)fixture.load(graph_ir, manifest);
  } catch (const flagdnn::thead::TheadError& error) {
    require(error.result() == FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
            std::string(name) + " returned the wrong error category");
    require(std::string_view(error.what()).find(expected_detail) !=
                std::string_view::npos,
            std::string(name) + " did not report " +
                std::string(expected_detail) + ": " + error.what());
    return;
  }
  throw std::runtime_error(std::string(name) +
                           " artifact mutation unexpectedly passed");
}

}  // namespace

int main() {
  try {
    Fixture fixture;
    const std::string graph_ir = fixture.graph();
    const std::string request_hash = flagdnn::native::sha256(graph_ir);
    const std::string valid = fixture.manifest(request_hash);
    const auto artifact = fixture.load(graph_ir, valid);
    require(artifact.backend == "thead", "artifact backend was not retained");
    require(artifact.target == "ppu_contract_cc80",
            "artifact target was not retained");
    require(artifact.engine == "libtriton_jit",
            "artifact engine was not retained");
    require(artifact.external_uids ==
                std::vector<std::int64_t>({1, 2, 4}),
            "artifact external UIDs were not retained");
    require(artifact.tensors.size() == 5 && artifact.stages.size() == 3,
            "artifact tensor/stage model was not retained");
    require(artifact.workspace_size == 256 &&
                artifact.workspace_alignment == 256,
            "artifact workspace contract was not retained");
    require(artifact.stages.at(1).variants.size() == 2 &&
                artifact.stages.at(1).autotune,
            "artifact autotune variants were not retained");
    require(!artifact.stages.at(1).variants.at(0).maxnreg.has_value() &&
                artifact.stages.at(1)
                    .variants.at(0)
                    .ppu_compiler_options.empty(),
            "artifact PPU compile-option identity was not retained");
    require(artifact.stages.at(0).registry_sha256 == kCandidateIdentity,
            "artifact registry identity was not retained");

    // FP8 software conversion uses an explicitly declared byte pointer ABI.
    // Graph types and allocation sizes remain tied to the original FP8 tensor.
    for (const std::string_view type : {"fp8_e4m3", "fp8_e5m2"}) {
      const std::string type_field = "\"data_type\":\"" + std::string(type) + "\"";
      const auto fp8_graph = replace_once(graph_ir, "\"data_type\":\"float32\"", type_field);
      auto fp8_manifest = fixture.manifest(flagdnn::native::sha256(fp8_graph));
      fp8_manifest = replace_once(fp8_manifest, "\"data_type\":\"float32\"", type_field);
      fp8_manifest = replace_once(fp8_manifest, "\"storage_size\":16", "\"storage_size\":4");
      fp8_manifest = replace_once(fp8_manifest, tensor_argument(1, false),
          "{\"kind\":\"tensor\",\"uid\":1,\"size\":4,\"alignment\":16,"
          "\"storage_view\":\"fp8_bytes\"}");
      fp8_manifest = replace_once(fp8_manifest, "*fp32:16", "*i8:16");
      const auto fp8_artifact = fixture.load(fp8_graph, fp8_manifest);
      require(fp8_artifact.stages[0].variants[0].arguments[0].fp8_storage_bytes,
              "FP8 byte view was not preserved");
      require_compilation_failure(fixture, "undeclared FP8 byte view", fp8_graph,
          replace_once(fp8_manifest, ",\"storage_view\":\"fp8_bytes\"", ""), "signature");
      require_compilation_failure(fixture, "FP8 byte view pointer mismatch", fp8_graph,
          replace_once(fp8_manifest, "*i8:16", "*fp32:16"), "signature");
      require_compilation_failure(fixture, "unknown FP8 storage view", fp8_graph,
          replace_once(fp8_manifest, "\"fp8_bytes\"", "\"unknown\""), "storage_view");
    }
    require_compilation_failure(fixture, "FP8 byte view on FP32 tensor", graph_ir,
        replace_once(valid, tensor_argument(1, false),
            "{\"kind\":\"tensor\",\"uid\":1,\"size\":16,\"alignment\":16,"
            "\"storage_view\":\"fp8_bytes\"}"), "storage_view");

    const auto fused_artifact =
        fixture.load(graph_ir, fixture.fused_manifest(request_hash));
    require(fused_artifact.workspace_size == 0 &&
                fused_artifact.stages.size() == 1,
            "fused internal virtual tensors unexpectedly require workspace");

    const auto fail = [&](std::string_view name,
                          std::string mutated,
                          std::string_view detail) {
      require_compilation_failure(fixture, name, graph_ir, mutated, detail);
    };
    fail("unproven scalar divisibility",
         replace_once(valid, ",i32,", ",i32:16,"), "signature");
    fail("unsupported scalar divisibility",
         replace_once(valid, ",i32,", ",i32:8,"), "signature");
    // A divisible scalar remains a runtime argument, unlike :1, which
    // libtriton_jit specializes away. No device execution is needed here.
    auto hinted = replace_once(valid, ",i32,", ",i32:16,");
    hinted = replace_once(hinted, "\"name\":\"n_elements\",\"value\":4",
                          "\"name\":\"stride\",\"value\":16");
    const auto hinted_artifact = fixture.load(graph_ir, hinted);
    require(hinted_artifact.stages[0].variants[0].runtime_signature.back() == "i32:16"
                && hinted_artifact.stages[0].variants[0].arguments.back().scalar_i32 == 16,
            "divisible scalar was not retained in the runtime ABI");

    fail("wrong schema",
         replace_once(valid, "{\"schema_version\":1",
                      "{\"schema_version\":2"),
         "schema_version");
    fail("wrong backend",
         replace_once(valid, "\"backend\":\"thead\"",
                      "\"backend\":\"nvidia\""),
         "backend");
    fail("wrong target",
         replace_once(valid, "\"target\":\"ppu_contract_cc80\"",
                      "\"target\":\"sm_80\""),
         "target");
    fail("wrong engine",
         replace_once(valid, "\"engine\":\"libtriton_jit\"",
                      "\"engine\":\"external_artifact\""),
         "engine");
    fail("duplicate JSON field",
         replace_once(valid, "{\"schema_version\":1,",
                      "{\"schema_version\":1,\"schema_version\":1,"),
         "duplicate object key");
    fail("unknown JSON field",
         replace_once(valid, "{\"schema_version\":1,",
                      "{\"schema_version\":1,\"unexpected\":0,"),
         "unknown field");
    fail("path traversal",
         replace_once(valid, "generated_stage_0.py", "../escape.py"),
         "unsafe source path");
    fail("symlink escape",
         replace_once(valid, "generated_stage_0.py", "linked.py"),
         "escaped artifact root");
    fail("source hash mismatch",
         replace_once(valid, fixture.source_hash(0), std::string(64, '0')),
         "source hash");
    fail("source size mismatch",
         replace_once(
             valid,
             "\"path\":\"generated_stage_0.py\",\"size\":" +
                 std::to_string(fixture.source_size(0)),
             "\"path\":\"generated_stage_0.py\",\"size\":999"),
         "source size");
    fail("invalid registry hash",
         replace_once(valid,
                      "\"registry_sha256\":\"" +
                          std::string(kCandidateIdentity) + "\"",
                      "\"registry_sha256\":\"invalid\""),
         "kernel identity");
    fail("duplicate tensor UID",
         replace_once(valid, "{\"uid\":2,\"data_type\"",
                      "{\"uid\":1,\"data_type\""),
         "tensor UID");
    fail("duplicate stage UID",
         replace_once(valid, "\"stage_id\":1", "\"stage_id\":0"),
         "stage ID");
    fail("cyclic stage dependency",
         replace_once(valid,
                      "\"stage_id\":1,\"source_node_ids\":[1],"
                      "\"dependencies\":[0]",
                      "\"stage_id\":1,\"source_node_ids\":[1],"
                      "\"dependencies\":[1]"),
         "dependency");
    fail("missing stage dependency",
         replace_once(valid,
                      "\"stage_id\":2,\"source_node_ids\":[2],"
                      "\"dependencies\":[0,1]",
                      "\"stage_id\":2,\"source_node_ids\":[2],"
                      "\"dependencies\":[1]"),
         "missing dependency");
    fail("invalid grid",
         replace_once(valid, "\"grid\":[1,1,1]", "\"grid\":[0,1,1]"),
         "grid");
    fail("invalid block",
         replace_once(valid, "\"block\":[64,1,1]",
                      "\"block\":[2048,1,1]"),
         "block");
    fail("invalid shared memory",
         replace_once(valid, "\"shared_memory\":0",
                      "\"shared_memory\":1073741825"),
         "shared memory");
    fail("missing maxnreg identity",
         replace_once(valid, "\"maxnreg\":null,", ""),
         "maxnreg");
    fail("invalid maxnreg",
         replace_once(valid, "\"maxnreg\":null", "\"maxnreg\":0"),
         "maxnreg");
    fail("unapproved PPU compiler option",
         replace_once(valid, "\"ppu_compiler_options\":{}",
                      "\"ppu_compiler_options\":{\"unknown\":\"1\"}"),
         "unapproved PPU compiler option");
    fail("invalid PPU compiler option value",
         replace_once(valid, "\"ppu_compiler_options\":{}",
                      "\"ppu_compiler_options\":{\"debug\":true}"),
         "PPU compiler option value");
    fail("argument count mismatch",
         replace_once(valid, "\"argument_count\":4",
                      "\"argument_count\":3"),
         "argument_count");
    fail("argument kind mismatch",
         replace_once(valid,
                      "{\"kind\":\"tensor\",\"uid\":1,\"size\":16,"
                      "\"alignment\":16}",
                      "{\"kind\":\"scalar_i32\",\"name\":\"left\","
                      "\"value\":4}"),
         "signature");
    fail("pointer element type mismatch",
         replace_once(valid, "*fp32:16", "*fp16:16"),
         "signature");
    fail("argument alignment mismatch",
         replace_once(valid,
                      "{\"kind\":\"tensor\",\"uid\":1,\"size\":16,"
                      "\"alignment\":16}",
                      "{\"kind\":\"tensor\",\"uid\":1,\"size\":16,"
                      "\"alignment\":8}"),
         "alignment");
    fail("workspace overflow",
         replace_once(valid, "\"workspace_offset\":16",
                      "\"workspace_offset\":256"),
         "workspace range");
    fail("workspace overlap",
         replace_once(valid, "\"workspace_offset\":16",
                      "\"workspace_offset\":0"),
         "workspace overlap");
    fail("workspace argument offset mismatch",
         replace_once(
             valid,
             "{\"kind\":\"workspace_tensor\",\"uid\":5,\"size\":16,"
             "\"alignment\":16,\"workspace_offset\":16}",
             "{\"kind\":\"workspace_tensor\",\"uid\":5,\"size\":16,"
             "\"alignment\":16,\"workspace_offset\":32}"),
         "workspace offset does not match tensor");
    fail("duplicate variant ID",
         replace_once(valid, "\"variant_id\":\"block128\"",
                      "\"variant_id\":\"default\""),
         "variant ID");
    std::string scalar_mismatch = valid;
    const std::size_t second_variant =
        scalar_mismatch.find("\"variant_id\":\"block128\"");
    const std::size_t second_variant_scalar =
        scalar_mismatch.find("\"value\":4", second_variant);
    require(second_variant != std::string::npos &&
                second_variant_scalar != std::string::npos,
            "cannot construct scalar-ABI mismatch mutation");
    scalar_mismatch.replace(second_variant_scalar,
                            std::string_view("\"value\":4").size(),
                            "\"value\":5");
    fail("variant scalar mismatch", std::move(scalar_mismatch),
         "incompatible argument ABI");
    fail("incomplete autotune identity",
         replace_once(valid,
                      ",\"candidate_identity\":\"" +
                          std::string(kCandidateIdentity) + "\"",
                      ""),
         "candidate_identity");
    fail("non-finite NaN",
         replace_once(valid, "\"shared_memory\":0",
                      "\"shared_memory\":NaN"),
         "invalid JSON");
    fail("non-finite infinity",
         replace_once(valid, "\"shared_memory\":0",
                      "\"shared_memory\":1e999"),
         "floating-point number is out of range");
    fail("external binding mismatch",
         replace_once(valid, "\"external_uids\":[1,2,4]",
                      "\"external_uids\":[1,4]"),
         "external_uids");
    fail("request hash mismatch",
         replace_once(valid, request_hash, std::string(64, 'c')),
         "request SHA-256");

    std::cout << "PASS THead artifact schema and mutation contract\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
