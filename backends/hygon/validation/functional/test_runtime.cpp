/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <flagdnn/flagdnn.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "hip_driver.hpp"
#include "pointwise_reference.hpp"
#include "tensor_io.hpp"

namespace {

namespace hv = flagdnn::validation::hygon;
namespace io = flagdnn::validation::hygon::tensor_io;

bool process_maps_contains(const std::string &needle) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void require_native_process_clean(const char *stage) {
  const bool has_python = process_maps_contains("libpython");
  const bool has_torch = process_maps_contains("libtorch");
  std::cout << stage << "_has_libpython=" << std::boolalpha << has_python << " "
            << stage << "_has_libtorch=" << has_torch << '\n';
#if !defined(FLAGDNN_EXPECT_LIBTRITON_JIT)
  if (has_python || has_torch) {
    throw std::runtime_error(
        "native runtime test unexpectedly loaded Python or Torch");
  }
#endif
}

class TemporaryCache {
public:
  TemporaryCache() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "flagdnn-hygon-native-runtime-XXXXXX")
                              .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char *created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

  [[nodiscard]] std::size_t manifest_count() const {
    std::size_t result = 0;
    for (const auto &entry :
         std::filesystem::recursive_directory_iterator(path_)) {
      if (entry.is_regular_file() &&
          entry.path().filename() == "manifest.json") {
        ++result;
      }
    }
    return result;
  }

private:
  std::filesystem::path path_;
};

void require_workspace_contract(const flagdnn::Executable &executable);

flagdnn::TensorDescriptor make_tensor(std::int64_t uid) {
  const std::array<std::int64_t, 3> dimensions = {1, 1, 1024};
  const std::array<std::int64_t, 3> strides = {1024, 1024, 1};
  return flagdnn::TensorDescriptor(uid, FLAGDNN_DATA_FLOAT32, dimensions,
                                   strides);
}

flagdnn::TensorDescriptor make_strided_tensor(std::int64_t uid) {
  const std::array<std::int64_t, 2> dimensions = {2, 3};
  const std::array<std::int64_t, 2> strides = {5, 1};
  return flagdnn::TensorDescriptor(uid, FLAGDNN_DATA_FLOAT32, dimensions,
                                   strides);
}

flagdnn::Executable build_add(const flagdnn::Handle &handle,
                              const flagdnn::TensorDescriptor &left,
                              const flagdnn::TensorDescriptor &right,
                              const flagdnn::TensorDescriptor &output) {
  flagdnn::Graph graph;
  graph.pointwise(left, right, FLAGDNN_POINTWISE_ADD, output);
  graph.finalize();
  return flagdnn::Executable(handle, graph);
}

flagdnn::Executable
build_add_autotune(const flagdnn::Handle &handle,
                   const flagdnn::TensorDescriptor &left,
                   const flagdnn::TensorDescriptor &right,
                   const flagdnn::TensorDescriptor &output) {
  flagdnn::Graph graph;
  graph.pointwise(left, right, FLAGDNN_POINTWISE_ADD, output);
  graph.finalize();
  flagdnnBuildOptions_t options = FLAGDNN_BUILD_OPTIONS_INITIALIZER;
  options.flags =
      FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_A | FLAGDNN_BUILD_OPTION_AUTOTUNE;
  return flagdnn::Executable(handle, graph, &options);
}

flagdnn::Executable build_add_square(const flagdnn::Handle &handle,
                                     const flagdnn::TensorDescriptor &left,
                                     const flagdnn::TensorDescriptor &right,
                                     const flagdnn::TensorDescriptor &square,
                                     const flagdnn::TensorDescriptor &output) {
  flagdnn::Graph graph;
  graph.pointwise(right, right, FLAGDNN_POINTWISE_MUL, square);
  graph.pointwise(left, square, FLAGDNN_POINTWISE_ADD, output);
  graph.finalize();
  return flagdnn::Executable(handle, graph);
}

template <typename Build>
void require_build_failure(Build &&build, flagdnnStatus_t expected_status,
                           const char *message) {
  try {
    auto unexpected = build();
    (void)unexpected;
  } catch (const flagdnn::Error &error) {
    if (error.status() == expected_status) {
      return;
    }
    throw;
  }
  throw std::runtime_error(message);
}

template <typename Function>
void require_flagdnn_failure(Function &&function,
                             flagdnnStatus_t expected_status,
                             const char *message) {
  try {
    function();
  } catch (const flagdnn::Error &error) {
    if (error.status() == expected_status) {
      return;
    }
    throw;
  }
  throw std::runtime_error(message);
}

template <typename Function>
void require_flagdnn_failure_contains(Function &&function,
                                      flagdnnStatus_t expected_status,
                                      std::string_view expected_detail,
                                      const char *message) {
  try {
    function();
  } catch (const flagdnn::Error &error) {
    if (error.status() == expected_status &&
        std::string_view(error.what()).find(expected_detail) !=
            std::string_view::npos) {
      return;
    }
    throw;
  }
  throw std::runtime_error(message);
}

template <typename Build>
void require_build_failure_contains(Build &&build,
                                    flagdnnStatus_t expected_status,
                                    std::string_view expected_detail,
                                    const char *message) {
  try {
    auto unexpected = build();
    (void)unexpected;
  } catch (const flagdnn::Error &error) {
    if (error.status() == expected_status &&
        std::string_view(error.what()).find(expected_detail) !=
            std::string_view::npos) {
      return;
    }
    throw;
  }
  throw std::runtime_error(message);
}

void write_manifest_mutator(const std::filesystem::path &path) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create manifest mutation compiler");
  }
  output << R"PY(import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

real_compiler = os.environ["FLAGDNN_VALIDATION_REAL_COMPILER"]
completed = subprocess.run([sys.executable, real_compiler, *sys.argv[1:]])
if completed.returncode != 0:
    raise SystemExit(completed.returncode)
if "--output-dir" not in sys.argv:
    raise SystemExit(0)

output_dir = Path(sys.argv[sys.argv.index("--output-dir") + 1])
manifest_path = output_dir / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
mutation = os.environ["FLAGDNN_VALIDATION_MANIFEST_MUTATION"]

stage = manifest["program"]["stages"][0]
entry = stage.get("variants", [stage])[0]

def signature_tokens():
    return entry["full_signature"].split(",")

def write_signature(tokens):
    entry["full_signature"] = ",".join(tokens)

def later_autotune_variant():
    variants = stage.get("variants")
    if not isinstance(variants, list) or len(variants) < 2:
        raise RuntimeError("autotune mutation needs at least two variants")
    return variants[1]

if mutation == "node_count":
    manifest["graph_node_count"] += 1
elif mutation == "source_hash":
    manifest["source_sha256"] = "0" * 64
elif mutation == "argument_abi":
    changed = False
    for argument in entry["argument_abi"]:
        if argument["kind"] == "tensor":
            argument["size"] += 4
            changed = True
            break
    if not changed:
        raise RuntimeError("manifest has no external tensor ABI")
elif mutation == "tensor_role":
    tensor = next(
        argument
        for argument in entry["argument_abi"]
        if argument["kind"] == "tensor"
    )
    tensor["role"] = "output" if tensor["role"] != "output" else "left"
elif mutation == "scalar_name":
    scalar = next(
        argument
        for argument in entry["argument_abi"]
        if argument["kind"] == "scalar_i32"
    )
    scalar["name"] = "wrong_elements"
elif mutation == "kernel_function":
    stage["kernel"]["function"] = "layout_copy_kernel"
elif mutation == "fixed_stage_in_autotune_graph":
    stages = manifest["program"]["stages"]
    if not stages or "variants" not in stages[0]:
        raise RuntimeError("fixed-stage mutation needs an autotuned stage")
    fixed = stages[0]
    selected = fixed["variants"][0]
    del fixed["variants"]
    del fixed["tuning"]
    fixed.update(selected)
elif mutation == "without_pointer_range":
    for current_stage in manifest["program"]["stages"]:
        for current_entry in current_stage.get("variants", [current_stage]):
            current_entry["full_signature"] = current_entry[
                "full_signature"
            ].replace(":16S", ":16").replace(":S", "")
elif mutation == "scratch_overlap":
    workspace_size = manifest["workspace_size"]
    for stage in manifest["program"]["stages"]:
        for entry in stage.get("variants", [stage]):
            entry["launch"]["global_scratch_size"] = workspace_size
elif mutation == "workspace_alignment":
    manifest["workspace_alignment"] *= 2
elif mutation == "materialized_source":
    descriptor = stage["kernel"]["materialized_source"]
    source_path = output_dir / descriptor["file"]
    source = source_path.read_bytes() + b"\n# artifact mutation\n"
    source_path.write_bytes(source)
    descriptor["size"] = len(source)
    descriptor["sha256"] = hashlib.sha256(source).hexdigest()
elif mutation == "op_kind":
    tokens = signature_tokens()
    tokens[-3] = str(int(tokens[-3]) + 1)
    write_signature(tokens)
elif mutation == "alpha":
    tokens = signature_tokens()
    tokens[-2] = "2.0"
    write_signature(tokens)
elif mutation == "n_elements":
    changed = False
    for argument in entry["argument_abi"]:
        if argument["kind"] == "scalar_i32":
            argument["value"] += 1
            changed = True
            break
    if not changed:
        raise RuntimeError("manifest has no scalar_i32 argument")
elif mutation == "stride":
    if stage["kernel"]["function"] != "binary_strided_kernel":
        raise RuntimeError("stride mutation needs a strided pointwise kernel")
    tokens = signature_tokens()
    tokens[19] = str(int(tokens[19]) + 1)
    write_signature(tokens)
elif mutation == "block_size":
    tokens = signature_tokens()
    tokens[-1] = "3"
    write_signature(tokens)
elif mutation == "grid":
    entry["launch"]["grid"][0] += 1
elif mutation == "autotune_later_variant":
    variant = later_autotune_variant()
    variant["variant_id"] = "mutated_config_1"
elif mutation == "autotune_config_block_size":
    variant = later_autotune_variant()
    tokens = variant["full_signature"].split(",")
    old_block = int(tokens[-1])
    new_block = old_block * 2 if old_block < 65536 else old_block // 2
    tokens[-1] = str(new_block)
    variant["full_signature"] = ",".join(tokens)
    variant["config"]["META"]["BLOCK_SIZE"] = new_block
    n_elements = next(
        argument["value"]
        for argument in variant["argument_abi"]
        if argument["kind"] == "scalar_i32"
    )
    variant["launch"]["grid"][0] = (
        n_elements + new_block - 1
    ) // new_block
elif mutation == "autotune_num_warps":
    variant = later_autotune_variant()
    old_warps = variant["compile_options"]["num_warps"]
    new_warps = 8 if old_warps != 8 else 4
    variant["compile_options"]["num_warps"] = new_warps
    variant["config"]["num_warps"] = new_warps
    variant["launch"]["block"][0] = new_warps * 64
elif mutation == "autotune_num_stages":
    variant = later_autotune_variant()
    old_stages = variant["compile_options"]["num_stages"]
    new_stages = 3 if old_stages != 3 else 2
    variant["compile_options"]["num_stages"] = new_stages
    variant["config"]["num_stages"] = new_stages
elif mutation == "autotune_variant_abi":
    variant = later_autotune_variant()
    tensor_argument = next(
        argument
        for argument in variant["argument_abi"]
        if argument["kind"] == "tensor"
    )
    tensor_argument["size"] += 4
elif mutation == "autotune_candidate_identity":
    later_autotune_variant()
    stage["tuning"]["candidate_identity"] = "0" * 64
else:
    raise RuntimeError("unknown manifest mutation")

manifest_path.write_text(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")),
    encoding="utf-8",
)
)PY";
  if (!output) {
    throw std::runtime_error("cannot write manifest mutation compiler");
  }
}

void test_manifest_rejection_contracts(const char *compiler_executable,
                                       const char *compiler_entry) {
#if defined(FLAGDNN_EXPECT_LIBTRITON_JIT)
  TemporaryCache wrapper_directory;
  const std::filesystem::path wrapper =
      wrapper_directory.path() / "mutate_manifest.py";
  write_manifest_mutator(wrapper);
  if (setenv("FLAGDNN_VALIDATION_REAL_COMPILER", compiler_entry, 1) != 0) {
    throw std::runtime_error(
        "cannot configure real compiler for mutation test");
  }

  const auto run_case = [&](std::string_view mutation,
                            std::string_view expected_detail,
                            bool needs_virtual_tensor,
                            bool needs_strided_tensors, bool enable_autotune) {
    TemporaryCache cache;
    if (setenv("FLAGDNN_VALIDATION_MANIFEST_MUTATION",
               std::string(mutation).c_str(), 1) != 0) {
      throw std::runtime_error("cannot configure manifest mutation");
    }
    flagdnn::Handle handle("hygon", 0);
    handle.set_compiler(compiler_executable, wrapper.string(),
                        cache.path().string());
    const flagdnn::TensorDescriptor left =
        needs_strided_tensors ? make_strided_tensor(1) : make_tensor(1);
    const flagdnn::TensorDescriptor right =
        needs_strided_tensors ? make_strided_tensor(2) : make_tensor(2);
    const flagdnn::TensorDescriptor output =
        needs_strided_tensors ? make_strided_tensor(3) : make_tensor(3);
    if (!needs_virtual_tensor) {
      require_build_failure_contains(
          [&] {
            return enable_autotune
                       ? build_add_autotune(handle, left, right, output)
                       : build_add(handle, left, right, output);
          },
          FLAGDNN_STATUS_COMPILATION_FAILED, expected_detail,
          "mutated manifest was accepted");
      return;
    }
    if (enable_autotune) {
      throw std::runtime_error(
          "autotune mutation unexpectedly requested a virtual graph");
    }
    flagdnn::TensorDescriptor square = make_tensor(4);
    square.set_virtual();
    require_build_failure_contains(
        [&] { return build_add_square(handle, left, right, square, output); },
        FLAGDNN_STATUS_COMPILATION_FAILED, expected_detail,
        "scratch-overlapping manifest was accepted");
  };

  const auto run = [&](std::string_view mutation,
                       std::string_view expected_detail,
                       bool needs_virtual_tensor, bool needs_strided_tensors) {
    run_case(mutation, expected_detail, needs_virtual_tensor,
             needs_strided_tensors, false);
  };
  const auto run_autotune = [&](std::string_view mutation,
                                std::string_view expected_detail) {
    run_case(mutation, expected_detail, false, false, true);
  };

  run("node_count", "graph node count does not match", false, false);
  run("source_hash", "root source identity is inconsistent", false, false);
  run("argument_abi", "external tensor ABI does not match", false, false);
  run("scratch_overlap", "global scratch overlaps graph workspace", true,
      false);
  run("workspace_alignment", "workspace alignment does not match workspace ABI",
      false, false);
  run("materialized_source",
      "materialized source identity does not match execution stage", false,
      false);
  run("op_kind", "OP_KIND does not match request Graph IR", false, false);
  run("alpha", "ALPHA does not match request Graph IR", false, false);
  run("n_elements", "int32 argument does not match request Graph attribute",
      false, false);
  run("tensor_role", "tensor role/UID does not match request Graph port",
      false, false);
  run("scalar_name", "pointwise argument ABI does not match request", false,
      false);
  run("kernel_function", "kernel function is not allowed", false, false);
  run("stride", "DIM/STRIDE constants do not match request Graph IR", false,
      true);
  run("block_size", "BLOCK_SIZE does not match compiler tuning contract", false,
      false);
  run("grid", "launch grid does not match request Graph IR", false, false);
  run_autotune("autotune_later_variant",
               "autotune candidate identity does not match variants");
  run_autotune("autotune_config_block_size",
               "autotune candidate identity does not match variants");
  run_autotune("autotune_num_warps",
               "autotune candidate identity does not match variants");
  run_autotune("autotune_num_stages",
               "autotune candidate identity does not match variants");
  run_autotune("autotune_variant_abi",
               "external tensor ABI does not match request Graph IR");
  run_autotune("autotune_candidate_identity",
               "autotune candidate identity does not match variants");

  const auto run_positive = [&](std::string_view mutation,
                                bool autotune_graph) {
    TemporaryCache cache;
    if (setenv("FLAGDNN_VALIDATION_MANIFEST_MUTATION",
               std::string(mutation).c_str(), 1) != 0) {
      throw std::runtime_error("cannot configure positive manifest mutation");
    }
    flagdnn::Handle handle("hygon", 0);
    handle.set_compiler(compiler_executable, wrapper.string(),
                        cache.path().string());
    const flagdnn::TensorDescriptor left = make_tensor(1);
    const flagdnn::TensorDescriptor right = make_tensor(2);
    const flagdnn::TensorDescriptor output = make_tensor(3);
    auto executable = autotune_graph
                          ? build_add_autotune(handle, left, right, output)
                          : build_add(handle, left, right, output);
    require_workspace_contract(executable);
  };
  run_positive("without_pointer_range", false);
  run_positive("fixed_stage_in_autotune_graph", true);
  (void)unsetenv("FLAGDNN_VALIDATION_MANIFEST_MUTATION");
  (void)unsetenv("FLAGDNN_VALIDATION_REAL_COMPILER");
  std::cout << "PASS manifest_rejection_and_compatibility_contracts\n";
#else
  (void)compiler_executable;
  (void)compiler_entry;
#endif
}

void test_add_validation_contract(const flagdnn::Handle &handle) {
  {
    const std::array<std::int64_t, 2> left_dimensions = {2, 3};
    const std::array<std::int64_t, 2> left_strides = {3, 1};
    const std::array<std::int64_t, 1> right_dimensions = {4};
    const std::array<std::int64_t, 1> right_strides = {1};
    flagdnn::TensorDescriptor left(101, FLAGDNN_DATA_FLOAT32, left_dimensions,
                                   left_strides);
    flagdnn::TensorDescriptor right(102, FLAGDNN_DATA_FLOAT32, right_dimensions,
                                    right_strides);
    flagdnn::TensorDescriptor output(103, FLAGDNN_DATA_FLOAT32, left_dimensions,
                                     left_strides);
    require_build_failure(
        [&] { return build_add(handle, left, right, output); },
        FLAGDNN_STATUS_INVALID_VALUE, "invalid Add broadcast was not rejected");
  }
  {
    const std::array<std::int64_t, 2> dimensions = {2, 3};
    const std::array<std::int64_t, 2> contiguous = {3, 1};
    const std::array<std::int64_t, 2> overlapping = {1, 1};
    flagdnn::TensorDescriptor left(104, FLAGDNN_DATA_FLOAT32, dimensions,
                                   overlapping);
    flagdnn::TensorDescriptor right(105, FLAGDNN_DATA_FLOAT32, dimensions,
                                    contiguous);
    flagdnn::TensorDescriptor output(106, FLAGDNN_DATA_FLOAT32, dimensions,
                                     contiguous);
    require_build_failure(
        [&] { return build_add(handle, left, right, output); },
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "overlapping Add input strides were not rejected");
  }
  std::cout << "PASS add_validation_contract\n";
}

void require_workspace_contract(const flagdnn::Executable &executable) {
#if defined(FLAGDNN_EXPECT_LIBTRITON_JIT)
  if (executable.workspace_size() == 0) {
    throw std::runtime_error(
        "libtriton_jit executable is missing its runtime scratch workspace");
  }
#else
  if (executable.workspace_size() != 0) {
    throw std::runtime_error(
        "external artifact executable unexpectedly needs workspace");
  }
#endif
}

struct AddBuffers {
  hv::DeviceBuffer left;
  hv::DeviceBuffer right;
  hv::DeviceBuffer output;
  std::array<flagdnnBinding_t, 3> bindings;

  explicit AddBuffers(std::size_t bytes)
      : left(bytes), right(bytes),
        output(bytes), bindings{flagdnnBinding_t{1, left.opaque()},
                                flagdnnBinding_t{2, right.opaque()},
                                flagdnnBinding_t{3, output.opaque()}} {}
};

void initialize_inputs(AddBuffers &buffers, std::span<const std::uint8_t> left,
                       std::span<const std::uint8_t> right,
                       hv::Stream &stream) {
  buffers.left.copy_from_host(left.data(), left.size(), stream.get());
  buffers.right.copy_from_host(right.data(), right.size(), stream.get());
}

std::vector<float> read_output(const hv::DeviceBuffer &buffer,
                               std::size_t element_count, hv::Stream &stream) {
  std::vector<std::uint8_t> bytes(element_count * sizeof(float));
  buffer.copy_to_host(bytes.data(), bytes.size(), stream.get());
  stream.synchronize();
  return io::decode(bytes, FLAGDNN_DATA_FLOAT32, element_count);
}

double compare(std::span<const float> actual,
               std::span<const float> reference) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error(
        "FlagDNN and hipDNN runtime outputs differ in size");
  }
  double maximum = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double difference = std::abs(static_cast<double>(actual[index]) -
                                       static_cast<double>(reference[index]));
    if (!std::isfinite(difference)) {
      throw std::runtime_error(
          "runtime comparison produced a non-finite error");
    }
    maximum = std::max(maximum, difference);
  }
  return maximum;
}

void test_execute_argument_contracts(
    const flagdnn::Executable &executable,
    std::span<const std::uint8_t> encoded_left,
    std::span<const std::uint8_t> encoded_right, hv::Stream &stream) {
  AddBuffers buffers(encoded_left.size());
  initialize_inputs(buffers, encoded_left, encoded_right, stream);
  hv::DeviceBuffer workspace_storage(executable.workspace_size() + 256);
  void *const aligned_workspace = workspace_storage.opaque_at(256);
  if (reinterpret_cast<std::uintptr_t>(workspace_storage.opaque()) % 256 != 0 ||
      reinterpret_cast<std::uintptr_t>(aligned_workspace) % 256 != 0) {
    throw std::runtime_error(
        "HIP allocation does not satisfy 256-byte alignment");
  }

  require_flagdnn_failure(
      [&] {
        executable.execute(buffers.bindings, workspace_storage.opaque_at(16),
                           executable.workspace_size() - 1, stream.opaque());
      },
      FLAGDNN_STATUS_INVALID_VALUE, "undersized workspace was not rejected");

  require_flagdnn_failure_contains(
      [&] {
        executable.execute(buffers.bindings, workspace_storage.opaque_at(16),
                           executable.workspace_size(), stream.opaque());
      },
      FLAGDNN_STATUS_INVALID_VALUE,
      "workspace base address is not 256-byte aligned",
      "misaligned workspace was not rejected");
  executable.execute(buffers.bindings, aligned_workspace,
                     executable.workspace_size(), stream.opaque());

  std::array<flagdnnBinding_t, 3> null_pointer_bindings = buffers.bindings;
  null_pointer_bindings[0].device_pointer = nullptr;
  require_flagdnn_failure(
      [&] {
        executable.execute(null_pointer_bindings, aligned_workspace,
                           executable.workspace_size(), stream.opaque());
      },
      FLAGDNN_STATUS_INVALID_VALUE,
      "null tensor binding pointer was not rejected");

  const flagdnnStatus_t null_array_status = flagdnnExecuteAsync(
      executable.get(), nullptr, buffers.bindings.size(), aligned_workspace,
      executable.workspace_size(), stream.opaque());
  if (null_array_status != FLAGDNN_STATUS_INVALID_VALUE) {
    throw std::runtime_error("null binding array was not rejected");
  }

  const std::size_t tensor_bytes = encoded_left.size();
  if (tensor_bytes < 32 || tensor_bytes % 16 != 0) {
    throw std::runtime_error(
        "binding alias test tensor size violates alignment assumptions");
  }
  hv::DeviceBuffer alias_storage(tensor_bytes * 3);
  const auto make_alias_bindings = [&](std::size_t right_offset) {
    return std::array<flagdnnBinding_t, 3>{
        flagdnnBinding_t{1, alias_storage.opaque()},
        flagdnnBinding_t{2, alias_storage.opaque_at(right_offset)},
        flagdnnBinding_t{3, alias_storage.opaque_at(tensor_bytes * 2)}};
  };

  alias_storage.copy_from_host_at(encoded_left.data(), tensor_bytes, 0,
                                  stream.get());
  const auto identical_bindings = make_alias_bindings(0);
  executable.execute(identical_bindings, aligned_workspace,
                     executable.workspace_size(), stream.opaque());
  std::vector<std::uint8_t> identical_output(tensor_bytes);
  alias_storage.copy_to_host_at(identical_output.data(), tensor_bytes,
                                tensor_bytes * 2, stream.get());
  stream.synchronize();
  const std::size_t element_count = tensor_bytes / sizeof(float);
  const std::vector<float> aliased_input =
      io::decode(encoded_left, FLAGDNN_DATA_FLOAT32, element_count);
  const std::vector<float> aliased_actual =
      io::decode(identical_output, FLAGDNN_DATA_FLOAT32, element_count);
  std::vector<float> aliased_expected(element_count);
  for (std::size_t index = 0; index < element_count; ++index) {
    aliased_expected[index] = aliased_input[index] + aliased_input[index];
  }
  if (compare(aliased_actual, aliased_expected) != 0.0) {
    throw std::runtime_error(
        "input-input alias changed Hygon Add execution semantics");
  }

  std::array<flagdnnBinding_t, 3> overflowing_bindings = buffers.bindings;
  overflowing_bindings[0].device_pointer = reinterpret_cast<void *>(
      std::numeric_limits<std::uintptr_t>::max() - tensor_bytes / 2);
  require_flagdnn_failure_contains(
      [&] {
        executable.execute(overflowing_bindings, aligned_workspace,
                           executable.workspace_size(), stream.opaque());
      },
      FLAGDNN_STATUS_INVALID_VALUE,
      "tensor binding device address range overflows",
      "overflowing tensor binding address range was not rejected");

  alias_storage.copy_from_host_at(encoded_right.data(), tensor_bytes,
                                  tensor_bytes, stream.get());
  const auto adjacent_bindings = make_alias_bindings(tensor_bytes);
  executable.execute(adjacent_bindings, aligned_workspace,
                     executable.workspace_size(), stream.opaque());
  stream.synchronize();
  std::cout << "PASS external_binding_alias_contracts\n";

  executable.execute(buffers.bindings, aligned_workspace,
                     executable.workspace_size(), stream.opaque());
  stream.synchronize();
  std::cout << "PASS execute_argument_contracts\n";
}

double execute_and_measure(const flagdnn::Executable &executable,
                           AddBuffers &buffers, hv::Stream &stream) {
  hv::DeviceBuffer workspace(executable.workspace_size());
  executable.execute(buffers.bindings, workspace.opaque(),
                     executable.workspace_size(), stream.opaque());
  stream.synchronize();

  for (int index = 0; index < 10; ++index) {
    executable.execute(buffers.bindings, workspace.opaque(),
                       executable.workspace_size(), stream.opaque());
  }
  stream.synchronize();
  hv::EventTimer timer;
  return timer.measure_microseconds(stream.get(), 200, [&] {
    executable.execute(buffers.bindings, workspace.opaque(),
                       executable.workspace_size(), stream.opaque());
  });
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: native_hygon_integration COMPILER_EXECUTABLE "
                   "COMPILER_ENTRY\n";
      return 2;
    }
#if defined(FLAGDNN_EXPECT_LIBTRITON_JIT)
    if (setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
      throw std::runtime_error("cannot select libtriton_jit engine");
    }
#endif
    require_native_process_clean("before_codegen");
    hv::DeviceGuard device;
    hv::Stream stream;
    TemporaryCache cache;

    std::unique_ptr<flagdnn::Executable> executable;
    std::unique_ptr<flagdnn::Executable> cached_executable;
    {
      flagdnn::Handle handle("hygon", 0);
      handle.set_compiler(argv[1], argv[2], cache.path().string());
      const flagdnn::TensorDescriptor left = make_tensor(1);
      const flagdnn::TensorDescriptor right = make_tensor(2);
      const flagdnn::TensorDescriptor output = make_tensor(3);

      executable = std::make_unique<flagdnn::Executable>(
          build_add(handle, left, right, output));
      require_workspace_contract(*executable);
      if (cache.manifest_count() != 1) {
        throw std::runtime_error("expected exactly one compiled Add artifact");
      }
      require_native_process_clean("after_codegen");

      handle.set_compiler("/definitely/missing/flagdnn-python", argv[2],
                          cache.path().string());
      cached_executable = std::make_unique<flagdnn::Executable>(
          build_add(handle, left, right, output));
      require_workspace_contract(*cached_executable);
      if (cache.manifest_count() != 1) {
        throw std::runtime_error("cache hit unexpectedly created an artifact");
      }
      std::cout << "PASS cache_hit_without_python\n";
      test_add_validation_contract(handle);
    }
    std::cout << "PASS executable_outlives_handle_graph_and_descriptors\n";
    test_manifest_rejection_contracts(argv[1], argv[2]);

    std::vector<float> host_left(1024);
    std::vector<float> host_right(1024);
    for (std::size_t index = 0; index < host_left.size(); ++index) {
      host_left[index] =
          static_cast<float>(static_cast<int>(index % 37) - 18) / 7.0F;
      host_right[index] =
          static_cast<float>(static_cast<int>(index % 11) - 5) / 13.0F;
    }
    const std::vector<std::uint8_t> encoded_left =
        io::encode(host_left, FLAGDNN_DATA_FLOAT32);
    const std::vector<std::uint8_t> encoded_right =
        io::encode(host_right, FLAGDNN_DATA_FLOAT32);

    std::vector<hv::ReferenceTensor> reference_tensors = {
        {1, FLAGDNN_DATA_FLOAT32, {1, 1, 1024}, {1024, 1024, 1}, 0},
        {2, FLAGDNN_DATA_FLOAT32, {1, 1, 1024}, {1024, 1024, 1}, 0},
        {3, FLAGDNN_DATA_FLOAT32, {1, 1, 1024}, {1024, 1024, 1}, 0},
    };
    hv::HipdnnPointwiseOperation reference_operation;
    reference_operation.kind = hv::HipdnnPointwiseKind::kAdd;
    reference_operation.unavailable_reason.clear();
    const hv::HipdnnCapability capability = hv::hipdnn_pointwise_capability(
        reference_operation, reference_tensors, true);
    hv::require_valid_hipdnn_adapter_contract(capability, "hygon runtime add");
    if (!capability.supported) {
      std::cout << "[SKIP][hipdnn] op=add case=hygon_runtime_add reason="
                << capability.reason << ' ' << hv::hipdnn_environment() << '\n';
      return 77;
    }

    AddBuffers reference_buffers(encoded_left.size());
    initialize_inputs(reference_buffers, encoded_left, encoded_right, stream);
    hv::HipdnnPointwisePlan reference(reference_operation, reference_tensors);
    hv::DeviceBuffer reference_workspace(reference.workspace_size());
    reference.execute(reference_buffers.bindings, reference_workspace.opaque(),
                      reference.workspace_size(), stream.opaque());
    stream.synchronize();
    const std::vector<float> expected =
        read_output(reference_buffers.output, host_left.size(), stream);

    test_execute_argument_contracts(*executable, encoded_left, encoded_right,
                                    stream);

    AddBuffers primary_buffers(encoded_left.size());
    initialize_inputs(primary_buffers, encoded_left, encoded_right, stream);
    const double primary_us =
        execute_and_measure(*executable, primary_buffers, stream);
    const std::vector<float> primary =
        read_output(primary_buffers.output, host_left.size(), stream);

    AddBuffers cached_buffers(encoded_left.size());
    initialize_inputs(cached_buffers, encoded_left, encoded_right, stream);
    const double cached_us =
        execute_and_measure(*cached_executable, cached_buffers, stream);
    const std::vector<float> cached =
        read_output(cached_buffers.output, host_left.size(), stream);

    const double primary_error = compare(primary, expected);
    const double cached_error = compare(cached, expected);
    if (primary_error != 0.0 || cached_error != 0.0) {
      throw std::runtime_error(
          "native runtime Add differs from hipDNN primitive");
    }
    require_native_process_clean("after_execution");
    std::cout << "PASS native_runtime_add hipdnn_max_abs_error="
              << std::max(primary_error, cached_error) << '\n';
    std::cout << std::fixed << std::setprecision(3)
              << "steady_state_us add=" << primary_us
              << " cached_add=" << cached_us << '\n';
    std::cout << "ALL_HYGON_NATIVE_RUNTIME_TESTS_PASSED\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "HYGON_NATIVE_RUNTIME_TEST_FAILED: " << error.what() << '\n';
    return 1;
  }
}
