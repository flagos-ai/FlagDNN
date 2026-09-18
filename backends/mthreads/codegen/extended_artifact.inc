/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
// Included in artifact.cpp's private validation namespace. New operator plans
// use the same runtime ABI, with tensor bounds checked independently of
// Python.
const std::map<std::string, std::string, std::less<>>& extended_sources() {
  static const std::map<std::string, std::string, std::less<>> sources = {
      {"relu_backward", "activation_backward.py"},
      {"tanh_backward", "activation_backward.py"},
      {"elu_backward", "activation_backward.py"},
      {"gelu_backward", "activation_backward.py"},
      {"softplus_backward", "activation_backward.py"},
      {"swish_backward", "activation_backward.py"},
      {"gelu_approx_tanh_backward", "activation_backward.py"},
      {"concatenate", "index.py"},
      {"gen_index", "index.py"},
      {"genstats", "statistics.py"},
      {"bn_finalize", "statistics.py"},
      {"resample", "resample.py"},
      {"rope", "position_embedding.py"},
      {"rope_backward", "position_embedding.py"},
      {"causal_conv1d", "causal_convolution.py"},
      {"matmul", "fp8_matmul.py"},
      {"matmul_fp8", "fp8_matmul.py"},
      {"moe_grouped_matmul", "moe_matmul.py"},
      {"moe_grouped_matmul_bwd", "moe_matmul.py"},
      {"instancenorm", "normalization_extended.py"},
      {"adalayernorm", "normalization_extended.py"},
      {"instancenorm_backward", "normalization_extended.py"},
      {"adalayernorm_backward", "normalization_extended.py"},
      {"layernorm_backward", "normalization_extended.py"},
      {"rmsnorm_backward", "normalization_extended.py"},
      {"batchnorm_backward", "normalization_extended.py"}};
  return sources;
}

bool registered_extended_function(const std::string& source,
                                  const std::string& function) {
  static const std::map<std::string, std::set<std::string>, std::less<>>
      functions = {
          {"kernels/activation_backward.py",
           {"activation_backward_contiguous_kernel",
            "activation_backward_strided_kernel"}},
          {"kernels/index.py",
           {"gen_index_kernel", "concatenate_copy_kernel"}},
          {"kernels/statistics.py", {"genstats_kernel", "bn_finalize_kernel"}},
          {"kernels/resample.py", {"resample_kernel"}},
          {"kernels/position_embedding.py", {"rope_kernel"}},
          {"kernels/causal_convolution.py", {"causal_conv1d_kernel"}},
          {"kernels/fp8_matmul.py", {"fp8_matmul_kernel"}},
          {"kernels/moe_matmul.py", {"moe_matmul_kernel"}},
          {"kernels/normalization_extended.py",
           {"extended_normalization_forward",
            "extended_normalization_backward", "compact_normalization_forward",
            "compact_batchnorm_backward"}},
      };
  const auto it = functions.find(source);
  return it != functions.end() && it->second.contains(function);
}

const std::vector<std::string>& extended_arguments(
    const std::string& function) {
  static const std::map<std::string, std::vector<std::string>, std::less<>>
      names = {
          {"activation_backward_contiguous_kernel",
           {"x_ptr", "y_ptr", "out_ptr", "n_elements"}},
          {"activation_backward_strided_kernel",
           {"x_ptr", "y_ptr", "out_ptr", "n_elements"}},
          {"gen_index_kernel", {"output_ptr"}},
          {"concatenate_copy_kernel", {"input_ptr", "output_ptr"}},
          {"genstats_kernel", {"x_ptr", "sum_ptr", "sq_sum_ptr"}},
          {"bn_finalize_kernel",
           {"sum_ptr", "sq_sum_ptr", "scale_ptr", "bias_ptr", "prev_mean_ptr",
            "prev_var_ptr", "eq_scale_ptr", "eq_bias_ptr", "mean_ptr",
            "inv_ptr", "next_mean_ptr", "next_var_ptr"}},
          {"resample_kernel", {"x_ptr", "y_ptr", "index_ptr"}},
          {"rope_kernel", {"x_ptr", "freqs_ptr", "y_ptr"}},
          {"causal_conv1d_kernel",
           {"x_ptr", "weight_ptr", "bias_ptr", "y_ptr"}},
          {"fp8_matmul_kernel",
           {"a_ptr", "b_ptr", "sa_ptr", "sb_ptr", "c_ptr"}},
          {"moe_matmul_kernel",
           {"token_ptr", "matrix_ptr", "offsets_ptr", "index_ptr", "ks_ptr",
            "output_ptr"}},
          {"extended_normalization_forward",
           {"x_ptr", "scale_ptr", "bias_ptr", "y_ptr", "mean_ptr", "inv_ptr"}},
          {"compact_normalization_forward",
           {"x_ptr", "scale_ptr", "bias_ptr", "y_ptr", "mean_ptr", "inv_ptr"}},
          {"extended_normalization_backward",
           {"dy_ptr", "x_ptr", "scale_ptr", "mean_ptr", "inv_ptr", "dx_ptr",
            "dscale_ptr", "dbias_ptr"}},
          {"compact_batchnorm_backward",
           {"dy_ptr", "x_ptr", "scale_ptr", "mean_ptr", "inv_ptr", "dx_ptr",
            "dscale_ptr", "dbias_ptr"}},
      };
  return names.at(function);
}

bool is_extended_request(const JsonValue& request) {
  const auto& nodes = request.at("graph").at("nodes").as_array();
  if (nodes.size() != 1) return false;
  const auto& operation = nodes[0].at("type").as_string();
  if (operation == "matmul") {
    for (const auto& tensor : request.at("graph").at("tensors").as_array())
      if (tensor.at("data_type").as_string().starts_with("fp8_")) return true;
    return false;
  }
  return extended_sources().contains(operation);
}

MthreadsArtifact parse_extended_artifact(
    const EngineBuildContext& context, const flagdnnBackendBuildInputV2& input,
    const JsonValue& request, std::string_view graph_ir,
    const std::string& request_hash) {
  require_exact_keys(request,
                     {"schema_version", "flagdnn_version", "backend", "target",
                      "build_options", "graph", "compiler_identity"},
                     "extended request");
  if (request.at("schema_version").as_int() != 3 ||
      request.at("backend").as_string() != "mthreads" ||
      request.at("target").as_string() != context.target_fingerprint ||
      request.at("flagdnn_version").as_string() != FLAGDNN_VERSION_STRING ||
      !is_lower_sha256(request.at("compiler_identity").as_string()))
    artifact_error("extended request identity mismatch");
  const auto& graph = request.at("graph");
  const auto& node = graph.at("nodes").as_array().at(0);
  const auto& operation = node.at("type").as_string();
  const std::string source_name =
      "kernels/" + extended_sources().at(operation);
  const auto& tensors = graph.at("tensors").as_array();
  if (graph.at("node_count").as_int() != 1 ||
      checked_size(graph.at("tensor_count").as_int(), "tensor_count") !=
          tensors.size())
    artifact_error("extended request graph counts differ");
  std::map<std::int64_t, TensorSpec> registry;
  MthreadsArtifact result;
  for (const auto& value : tensors) {
    auto tensor = parse_tensor(value);
    if (tensor.is_virtual || registry.contains(tensor.uid))
      artifact_error("extended request has virtual or duplicate tensors");
    result.binding_uids.push_back(tensor.uid);
    registry.emplace(tensor.uid, std::move(tensor));
  }
  const auto inputs =
      parse_ports(node.at("inputs"), registry, "extended inputs");
  const auto outputs =
      parse_ports(node.at("outputs"), registry, "extended outputs");
  std::set<std::int64_t> referenced;
  for (const auto& [name, uid] : inputs) {
    (void)name;
    referenced.insert(uid);
  }
  for (const auto& [name, uid] : outputs) {
    (void)name;
    if (!referenced.insert(uid).second)
      artifact_error("extended output aliases an input");
  }
  if (outputs.empty() || referenced.size() != registry.size())
    artifact_error("extended graph has missing outputs or unused tensors");

  const std::filesystem::path directory(input.artifact_directory);
  if (std::filesystem::is_symlink(directory) ||
      !std::filesystem::is_directory(directory))
    artifact_error("extended artifact directory is unsafe");
  const auto root = std::filesystem::canonical(directory);
  const auto manifest = flagdnn::native::json::parse(
      read_file(root / "manifest.json", kMaximumMetadataSize, "manifest"));
  require_exact_keys(
      manifest,
      {"schema_version", "artifact_kind", "flagdnn_version", "backend",
       "target", "engine", "request_sha256", "compiler_identity",
       "source_sha256", "workspace_size", "workspace_alignment",
       "external_binding_uids", "program", "files"},
      "extended manifest");
  for (const auto key :
       {"flagdnn_version", "backend", "target", "compiler_identity"})
    if (manifest.at(key).as_string() != request.at(key).as_string())
      artifact_error("extended manifest identity differs");
  if (manifest.at("schema_version").as_int() !=
          kMthreadsArtifactSchemaVersion ||
      manifest.at("artifact_kind").as_string() !=
          "flagdnn_execution_program" ||
      manifest.at("engine").as_string() != "libtriton_jit" ||
      manifest.at("request_sha256").as_string() != request_hash)
    artifact_error("extended manifest schema or request differs");
  result.workspace_size =
      checked_size(manifest.at("workspace_size").as_int(), "workspace");
  result.workspace_alignment =
      checked_size(manifest.at("workspace_alignment").as_int(), "alignment");
  if (result.workspace_size != kPhaseOneWorkspaceSize ||
      result.workspace_alignment != 256)
    artifact_error("extended workspace differs from ABI");
  const auto& bindings = manifest.at("external_binding_uids").as_array();
  if (bindings.size() != result.binding_uids.size())
    artifact_error("extended binding count differs");
  for (std::size_t i = 0; i < bindings.size(); ++i)
    if (bindings[i].as_int() != result.binding_uids[i])
      artifact_error("extended binding order differs");
  const auto& files = manifest.at("files").as_array();
  if (files.size() != 1) artifact_error("extended source count differs");
  std::string source_hash;
  const auto source =
      validate_source(root, files[0], source_name, source_hash);
  if (manifest.at("source_sha256").as_string() != source_hash)
    artifact_error("extended source hash differs");
  const auto& program = manifest.at("program");
  require_exact_keys(program, {"schema_version", "stage_count", "stages"},
                     "extended program");
  const auto& stages = program.at("stages").as_array();
  const std::size_t expected_stages =
      operation == "concatenate" ? inputs.size() : 1;
  if (program.at("schema_version").as_int() !=
          kMthreadsExecutionProgramVersion ||
      checked_size(program.at("stage_count").as_int(), "stage count") !=
          stages.size() ||
      stages.size() != expected_stages || stages.size() > kMaximumStages)
    artifact_error("extended stage count differs");
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto& value = stages[i];
    require_exact_keys(value,
                       {"id", "node_id", "operation", "dependencies", "source",
                        "function", "variants", "autotune"},
                       "extended stage");
    StageArtifact stage;
    stage.id = checked_size(value.at("id").as_int(), "stage id");
    stage.node_id = value.at("node_id").as_int();
    stage.source = source;
    stage.function_name = value.at("function").as_string();
    if (stage.id != i || stage.node_id != node.at("id").as_int() ||
        value.at("operation").as_string() != operation ||
        value.at("source").as_string() != source_name ||
        !registered_extended_function(source_name, stage.function_name))
      artifact_error("extended stage metadata differs");
    const auto& dependencies = value.at("dependencies").as_array();
    if (dependencies.size() != (i == 0 ? 0U : 1U))
      artifact_error("extended dependencies differ");
    if (i != 0) {
      if (dependencies[0].as_int() != static_cast<std::int64_t>(i - 1))
        artifact_error("extended dependency is not prior stage");
      stage.dependencies.push_back(i - 1);
    }
    const auto& tuning = value.at("autotune");
    require_exact_keys(tuning,
                       {"enabled", "warmup", "repetitions", "selection_cache"},
                       "extended tuning");
    const std::string selection =
        "tuning/stage-" + std::to_string(i) + ".json";
    if (tuning.at("enabled").as_bool() || tuning.at("warmup").as_int() != 1 ||
        tuning.at("repetitions").as_int() != 1 ||
        tuning.at("selection_cache").as_string() != selection)
      artifact_error("extended fixed launch tuning differs");
    stage.autotune = false;
    stage.warmup = 1;
    stage.repetitions = 1;
    stage.selection_cache = root / selection;
    const auto& variants = value.at("variants").as_array();
    if (variants.size() != 1)
      artifact_error("extended fixed launch must have one variant");
    const auto& variant_value = variants[0];
    require_exact_keys(
        variant_value,
        {"variant_id", "source", "source_sha256", "function", "full_signature",
         "grid", "num_warps", "num_stages", "arguments"},
        "extended variant");
    KernelVariantArtifact variant;
    variant.variant_id = variant_value.at("variant_id").as_string();
    variant.source = source;
    variant.function_name = variant_value.at("function").as_string();
    variant.full_signature = variant_value.at("full_signature").as_string();
    if (variant.variant_id != "default" ||
        variant.function_name != stage.function_name ||
        variant_value.at("source").as_string() != source_name ||
        variant_value.at("source_sha256").as_string() != source_hash ||
        variant.full_signature.empty() ||
        variant.full_signature.size() > 32768)
      artifact_error("extended variant metadata differs");
    const auto& grid = variant_value.at("grid").as_array();
    if (grid.size() != 3) artifact_error("extended grid rank differs");
    for (std::size_t axis = 0; axis < 3; ++axis) {
      variant.grid[axis] =
          checked_positive_unsigned(grid[axis].as_int(), "grid");
      if (variant.grid[axis] > (axis == 0 ? 2147483647U : 65535U))
        artifact_error("extended grid exceeds device limits");
    }
    variant.num_warps = checked_positive_unsigned(
        variant_value.at("num_warps").as_int(), "num_warps");
    variant.num_stages = checked_positive_unsigned(
        variant_value.at("num_stages").as_int(), "num_stages");
    if (variant.num_warps != 4 || variant.num_stages != 1)
      artifact_error("extended launch policy differs");
    const auto& arguments = variant_value.at("arguments").as_array();
    const auto& argument_names = extended_arguments(stage.function_name);
    if (arguments.size() != argument_names.size() ||
        arguments.size() > kMaximumArguments)
      artifact_error("extended argument count differs from kernel ABI");
    std::set<std::string> semantic_names;
    std::size_t signature_offset = 0;
    for (const auto& argument : arguments) {
      const auto& semantic = argument.at("semantic_name").as_string();
      if (semantic != argument_names[variant.arguments.size()] ||
          !printable_ascii_identifier(semantic, 256) ||
          !semantic_names.insert(semantic).second)
        artifact_error("extended argument names are invalid");
      const auto& kind = argument.at("kind").as_string();
      ArgumentSpec parsed;
      if (kind == "tensor") {
        const auto uid = argument.at("uid").as_int();
        const auto it = registry.find(uid);
        if (it == registry.end())
          artifact_error("extended tensor is absent from graph");
        parsed = parse_expected_argument(argument, ArgumentKind::kTensor,
                                         semantic, uid, "");
        parsed.storage_size = it->second.storage_size;
        parsed.alignment = it->second.alignment;
        parsed.data_type = it->second.data_type;
      } else if (kind == "scalar_i32" && semantic == "n_elements") {
        const auto count = node.at("attributes").at("n_elements").as_int();
        if (count <= 0 || count > std::numeric_limits<std::int32_t>::max())
          artifact_error("extended scalar exceeds int32");
        parsed = parse_expected_argument(
            argument, ArgumentKind::kScalarI32, semantic, count,
            i32_bits(static_cast<std::int32_t>(count)));
      } else
        artifact_error("extended argument kind is unsupported");
      // Runtime arguments precede constexpr parameters in all registered
      // common kernels.
      const auto comma = variant.full_signature.find(',', signature_offset);
      const auto token = variant.full_signature.substr(
          signature_offset, comma - signature_offset);
      if (kind == "tensor") {
        const auto& tensor = registry.at(parsed.uid);
        const bool raw_copy = operation == "concatenate";
        const auto base = token.substr(0, token.find(':'));
        const auto bytes = element_size(tensor.data_type);
        const bool raw_match = raw_copy && ((bytes == 1 && base == "*u8") ||
                                            (bytes == 2 && base == "*u16") ||
                                            (bytes == 4 && base == "*i32"));
        if (!raw_match && token != pointer_token(tensor))
          artifact_error("extended pointer signature differs from tensor");
      } else if (token != "i32")
        artifact_error("extended scalar signature differs");
      if (comma == std::string::npos)
        artifact_error("extended signature lacks constexpr parameters");
      signature_offset = comma + 1;
      variant.arguments.push_back(std::move(parsed));
    }
    stage.variants.push_back(std::move(variant));
    result.stages.push_back(std::move(stage));
  }
  validate_artifact_tree(root, graph_ir, source_name, result.stages.size());
  return result;
}
