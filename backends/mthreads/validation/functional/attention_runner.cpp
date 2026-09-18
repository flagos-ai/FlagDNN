/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <musa_runtime_api.h>
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
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "backends/mthreads/validation/functional/tensor_io_adapter.hpp"
#include "backends/mthreads/validation/musa_driver.hpp"
#include "backends/mthreads/validation/paired_timing.hpp"
#include "common/attention.hpp"

namespace flagdnn::testing {
namespace {

namespace mv = validation::mthreads;
namespace io = validation::mthreads::tensor_io;

class TemporaryCache final {
 public:
  explicit TemporaryCache(std::string_view operation) {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         ("flagdnn-mthreads-" + std::string(operation) +
          "-functional-XXXXXX"))
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed for MThreads Attention cache");
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

std::vector<float> make_values(std::size_t count,
                               std::size_t tensor_index,
                               float scale) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 37U + tensor_index * 19U) % 101U) - 50;
    result[index] = scale * static_cast<float>(centered) /
                    static_cast<float>(53U + tensor_index);
  }
  return result;
}

class TensorAllocation final {
 public:
  TensorAllocation(const TestTensor& specification,
                   std::span<const float> logical,
                   mv::Stream& stream)
      : specification_(specification),
        initial_(io::encode(io::scatter(logical, specification),
                            specification.data_type)),
        buffer_(specification.binding_byte_offset + initial_.size(), 256) {
    buffer_.copy_from_host_at(initial_.data(),
                              initial_.size(),
                              specification_.binding_byte_offset,
                              stream.get());
    logical_ = io::gather(
        io::decode(initial_, specification_.data_type), specification_);
  }

  static std::unique_ptr<TensorAllocation> input(
      const TestTensor& specification,
      std::size_t tensor_index,
      mv::Stream& stream,
      float scale) {
    return std::make_unique<TensorAllocation>(
        specification,
        make_values(io::element_count(specification), tensor_index, scale),
        stream);
  }

  static std::unique_ptr<TensorAllocation> scalar(
      const Fp8Scalar& scalar,
      mv::Stream& stream) {
    const float value[] = {scalar.value};
    return std::make_unique<TensorAllocation>(
        scalar.tensor, value, stream);
  }

  static std::unique_ptr<TensorAllocation> output(
      const TestTensor& specification,
      mv::Stream& stream) {
    return std::make_unique<TensorAllocation>(
        specification,
        std::vector<float>(io::element_count(specification),
                           io::kPaddingSentinel),
        stream);
  }

  [[nodiscard]] void* pointer() const {
    return buffer_.opaque_at(specification_.binding_byte_offset);
  }

  [[nodiscard]] const std::vector<float>& logical() const noexcept {
    return logical_;
  }

  [[nodiscard]] std::vector<std::uint8_t> read_bytes(
      mv::Stream& stream) const {
    std::vector<std::uint8_t> result(initial_.size());
    buffer_.copy_to_host_at(result.data(),
                            result.size(),
                            specification_.binding_byte_offset,
                            stream.get());
    stream.synchronize();
    return result;
  }

  [[nodiscard]] std::vector<float> read_logical(
      mv::Stream& stream,
      std::string_view provider,
      bool require_padding) const {
    const std::vector<std::uint8_t> encoded = read_bytes(stream);
    if (require_padding) {
      io::require_padding_unchanged(provider, encoded, specification_);
    }
    return io::gather(
        io::decode(encoded, specification_.data_type), specification_);
  }

  void require_unchanged(mv::Stream& stream,
                         std::string_view description) const {
    io::require_bytes_equal(description, read_bytes(stream), initial_);
  }

  void refresh_baseline(mv::Stream& stream) {
    initial_ = read_bytes(stream);
    logical_ = io::gather(
        io::decode(initial_, specification_.data_type), specification_);
  }

 private:
  TestTensor specification_;
  std::vector<std::uint8_t> initial_;
  mv::DeviceBuffer buffer_;
  std::vector<float> logical_;
};

struct AllocationPair {
  std::unique_ptr<TensorAllocation> production;
  std::unique_ptr<TensorAllocation> reference;
};

AllocationPair input_pair(const TestTensor& tensor,
                          std::size_t index,
                          mv::Stream& stream,
                          float scale) {
  const std::vector<float> values =
      make_values(io::element_count(tensor), index, scale);
  return {std::make_unique<TensorAllocation>(tensor, values, stream),
          std::make_unique<TensorAllocation>(tensor, values, stream)};
}

AllocationPair value_pair(const TestTensor& tensor,
                          std::span<const float> values,
                          mv::Stream& stream) {
  return {std::make_unique<TensorAllocation>(tensor, values, stream),
          std::make_unique<TensorAllocation>(tensor, values, stream)};
}

AllocationPair scalar_pair(const Fp8Scalar& scalar, mv::Stream& stream) {
  const float value[] = {scalar.value};
  return value_pair(scalar.tensor, value, stream);
}

AllocationPair output_pair(const TestTensor& tensor, mv::Stream& stream) {
  return {TensorAllocation::output(tensor, stream),
          TensorAllocation::output(tensor, stream)};
}

void append(std::vector<flagdnnBinding_t>& bindings,
            const TestTensor& tensor,
            const TensorAllocation& allocation) {
  bindings.push_back({tensor.uid, allocation.pointer()});
}

void append_pair(std::vector<flagdnnBinding_t>& production,
                 std::vector<flagdnnBinding_t>& reference,
                 const TestTensor& tensor,
                 const AllocationPair& allocation) {
  append(production, tensor, *allocation.production);
  append(reference, tensor, *allocation.reference);
}

void append_pair(std::vector<flagdnnBinding_t>& production,
                 std::vector<flagdnnBinding_t>& reference,
                 const Fp8Scalar& scalar,
                 const AllocationPair& allocation) {
  append_pair(production, reference, scalar.tensor, allocation);
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare_tensor(std::string_view case_name,
                        std::string_view tensor_name,
                        const TestTensor& specification,
                        const TensorAllocation& actual,
                        const TensorAllocation& reference,
                        double absolute_tolerance,
                        double relative_tolerance,
                        mv::Stream& stream,
                        std::string_view reference_provider) {
  static_cast<void>(specification);
  const std::vector<float> left =
      actual.read_logical(stream, "FlagDNN", true);
  const std::vector<float> right =
      reference.read_logical(stream, reference_provider, true);
  if (left.size() != right.size()) {
    throw std::runtime_error("Attention output sizes differ");
  }
  Accuracy result;
  for (std::size_t index = 0; index < left.size(); ++index) {
    const double absolute = std::abs(
        static_cast<double>(left[index]) - right[index]);
    const double relative = absolute /
        std::max({std::abs(static_cast<double>(left[index])),
                  std::abs(static_cast<double>(right[index])),
                  1.0e-30});
    result.maximum_absolute = std::max(result.maximum_absolute, absolute);
    result.maximum_relative = std::max(result.maximum_relative, relative);
    if (!std::isfinite(absolute) ||
        (absolute > absolute_tolerance && relative > relative_tolerance)) {
      std::ostringstream message;
      message << case_name << " differs at " << tensor_name << " element "
              << index << ": FlagDNN=" << left[index] << ", "
              << reference_provider << '=' << right[index]
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << absolute_tolerance
              << ", rtol=" << relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

void require_pair_unchanged(const AllocationPair& pair,
                            mv::Stream& stream,
                            std::string_view name) {
  pair.production->require_unchanged(
      stream, "FlagDNN " + std::string(name));
  pair.reference->require_unchanged(
      stream, "reference " + std::string(name));
}

void execute(AttentionExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             mv::DeviceBuffer& workspace,
             mv::Stream& stream) {
  executable.prepare(bindings, stream.opaque());
  executable.execute(bindings,
                     workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read MThreads Attention artifact");
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

using PathSet = std::set<std::filesystem::path>;
using SelectionContents = std::map<std::filesystem::path, std::string>;

PathSet manifest_paths(const std::filesystem::path& cache) {
  PathSet result;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(cache)) {
    if (entry.is_regular_file() &&
        entry.path().filename() == "manifest.json") {
      result.insert(std::filesystem::relative(entry.path(), cache));
    }
  }
  return result;
}

SelectionContents selection_contents(const std::filesystem::path& cache) {
  SelectionContents result;
  for (const auto& entry :
       std::filesystem::recursive_directory_iterator(cache)) {
    if (entry.is_regular_file() &&
        entry.path().parent_path().filename() == "tuning" &&
        entry.path().filename().string().starts_with("stage-") &&
        entry.path().extension() == ".json") {
      result.emplace(std::filesystem::relative(entry.path(), cache),
                     read_file(entry.path()));
    }
  }
  return result;
}

void verify_new_manifest(const std::filesystem::path& cache,
                         const PathSet& before,
                         std::initializer_list<std::string_view> functions) {
  const PathSet after = manifest_paths(cache);
  std::vector<std::filesystem::path> added;
  std::set_difference(after.begin(),
                      after.end(),
                      before.begin(),
                      before.end(),
                      std::back_inserter(added));
  if (added.size() != 1) {
    throw std::runtime_error(
        "Attention build did not publish exactly one new manifest");
  }
  const std::string manifest = read_file(cache / added.front());
  for (const std::string_view token :
       {std::string_view("\"backend\": \"mthreads\""),
        std::string_view("\"engine\": \"libtriton_jit\""),
        std::string_view("kernels/attention.py")}) {
    if (manifest.find(token) == std::string::npos) {
      throw std::runtime_error(
          "MThreads Attention manifest is missing " + std::string(token));
    }
  }
  for (const std::string_view function : functions) {
    if (manifest.find("\"function\": \"" + std::string(function) +
                      "\"") == std::string::npos) {
      throw std::runtime_error(
          "MThreads Attention manifest is missing kernel " +
          std::string(function));
    }
  }
}

template <typename Builder>
void verify_autotune_cache_hit(const std::filesystem::path& cache,
                               Builder&& rebuild) {
  const PathSet manifests = manifest_paths(cache);
  const SelectionContents selections = selection_contents(cache);
  if (selections.empty()) {
    throw std::runtime_error(
        "MThreads Attention autotune did not persist a selection");
  }
  auto cached = rebuild();
  static_cast<void>(cached);
  if (manifest_paths(cache) != manifests ||
      selection_contents(cache) != selections) {
    throw std::runtime_error(
        "MThreads Attention repeated build rewrote its autotune cache");
  }
}

struct HostForward {
  std::vector<float> output;
  std::vector<float> stats;
};

std::size_t offset(const TestTensor& tensor,
                   std::int64_t b,
                   std::int64_t h,
                   std::int64_t s,
                   std::int64_t d) {
  return static_cast<std::size_t>(
      (((b * tensor.dimensions[1] + h) * tensor.dimensions[2] + s) *
       tensor.dimensions[3]) + d);
}

HostForward host_forward(const SdpaBackwardTestCase& test_case,
                         std::span<const float> q,
                         std::span<const float> k,
                         std::span<const float> v,
                         std::span<const float> bias) {
  const std::int64_t batch = test_case.q.dimensions[0];
  const std::int64_t query_heads = test_case.q.dimensions[1];
  const std::int64_t key_heads = test_case.k.dimensions[1];
  const std::int64_t value_heads = test_case.v.dimensions[1];
  const std::int64_t sequence_q = test_case.q.dimensions[2];
  const std::int64_t sequence_kv = test_case.k.dimensions[2];
  const std::int64_t head_dimension = test_case.q.dimensions[3];
  const std::int64_t value_dimension = test_case.v.dimensions[3];
  const double scale = test_case.options.attention_scale.value_or(
      1.0F / std::sqrt(static_cast<float>(head_dimension)));
  const bool causal =
      test_case.options.diagonal_band_right_bound.has_value();
  HostForward result;
  result.output.resize(io::element_count(test_case.output));
  result.stats.resize(static_cast<std::size_t>(
      batch * query_heads * sequence_q));
  std::vector<double> scores(static_cast<std::size_t>(sequence_kv));
  std::vector<double> probabilities(static_cast<std::size_t>(sequence_kv));
  for (std::int64_t b = 0; b < batch; ++b) {
    for (std::int64_t h = 0; h < query_heads; ++h) {
      const std::int64_t kh = h / (query_heads / key_heads);
      const std::int64_t vh = h / (query_heads / value_heads);
      for (std::int64_t m = 0; m < sequence_q; ++m) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::int64_t n = 0; n < sequence_kv; ++n) {
          if (causal && n > m) {
            scores[static_cast<std::size_t>(n)] =
                -std::numeric_limits<double>::infinity();
            continue;
          }
          double score = 0.0;
          for (std::int64_t d = 0; d < head_dimension; ++d) {
            score += static_cast<double>(
                         q[offset(test_case.q, b, h, m, d)]) *
                     static_cast<double>(
                         k[offset(test_case.k, b, kh, n, d)]);
          }
          score *= scale;
          if (test_case.bias.has_value()) {
            const std::int64_t bb =
                test_case.bias->dimensions[0] == 1 ? 0 : b;
            const std::int64_t bh =
                test_case.bias->dimensions[1] == 1 ? 0 : h;
            score += bias[offset(*test_case.bias, bb, bh, m, n)];
          }
          scores[static_cast<std::size_t>(n)] = score;
          maximum = std::max(maximum, score);
        }
        double denominator = 0.0;
        for (std::int64_t n = 0; n < sequence_kv; ++n) {
          const double probability =
              std::isfinite(scores[static_cast<std::size_t>(n)])
                  ? std::exp(scores[static_cast<std::size_t>(n)] - maximum)
                  : 0.0;
          probabilities[static_cast<std::size_t>(n)] = probability;
          denominator += probability;
        }
        result.stats[static_cast<std::size_t>(
            (b * query_heads + h) * sequence_q + m)] =
            static_cast<float>(maximum + std::log(denominator));
        for (std::int64_t d = 0; d < value_dimension; ++d) {
          double value = 0.0;
          for (std::int64_t n = 0; n < sequence_kv; ++n) {
            value += probabilities[static_cast<std::size_t>(n)] /
                     denominator *
                     v[offset(test_case.v, b, vh, n, d)];
          }
          result.output[offset(test_case.output, b, h, m, d)] =
              static_cast<float>(value);
        }
      }
    }
  }
  return result;
}

void run_forward_case(const SdpaTestCase& test_case,
                      flagdnn::Handle& handle,
                      const std::filesystem::path& cache,
                      mv::Stream& stream) {
  const PathSet before = manifest_paths(cache);
  auto reference = build_sdpa_reference(test_case);
  auto production = build_flagdnn_sdpa(handle, test_case);
  verify_new_manifest(cache, before, {"_sdpa_fwd_kernel"});
  AllocationPair q = input_pair(test_case.q, 0, stream, 0.5F);
  AllocationPair k = input_pair(test_case.k, 1, stream, 0.5F);
  AllocationPair v = input_pair(test_case.v, 2, stream, 0.5F);
  std::optional<AllocationPair> bias;
  if (test_case.bias.has_value()) {
    bias.emplace(input_pair(*test_case.bias, 3, stream, 0.25F));
  }
  AllocationPair output = output_pair(test_case.output, stream);
  std::optional<AllocationPair> stats;
  if (test_case.stats.has_value()) {
    stats.emplace(output_pair(*test_case.stats, stream));
  }
  std::vector<flagdnnBinding_t> production_bindings;
  std::vector<flagdnnBinding_t> reference_bindings;
  append_pair(production_bindings, reference_bindings, test_case.q, q);
  append_pair(production_bindings, reference_bindings, test_case.k, k);
  append_pair(production_bindings, reference_bindings, test_case.v, v);
  if (bias.has_value()) {
    append_pair(production_bindings,
                reference_bindings,
                *test_case.bias,
                *bias);
  }
  append_pair(production_bindings,
              reference_bindings,
              test_case.output,
              output);
  if (stats.has_value()) {
    append_pair(production_bindings,
                reference_bindings,
                *test_case.stats,
                *stats);
  }
  mv::DeviceBuffer production_workspace(
      production->workspace_size(), 256);
  mv::DeviceBuffer reference_workspace(reference->workspace_size(), 256);
  stream.synchronize();
  execute(*production,
          production_bindings,
          production_workspace,
          stream);
  execute(*reference, reference_bindings, reference_workspace, stream);
  mv::timing::paired(
      test_case.name, stream,
      [&] {
        execute(*production, production_bindings, production_workspace,
                stream);
      },
      [&] {
        execute(*reference, reference_bindings, reference_workspace, stream);
      });
  stream.synchronize();
  require_pair_unchanged(q, stream, "Q");
  require_pair_unchanged(k, stream, "K");
  require_pair_unchanged(v, stream, "V");
  if (bias.has_value()) {
    require_pair_unchanged(*bias, stream, "bias");
  }
  const Accuracy output_accuracy = compare_tensor(
      test_case.name,
      "output",
      test_case.output,
      *output.production,
      *output.reference,
      test_case.output_absolute_tolerance,
      test_case.output_relative_tolerance,
      stream,
      "muDNN");
  Accuracy stats_accuracy;
  if (stats.has_value()) {
    stats_accuracy = compare_tensor(
        test_case.name,
        "stats",
        *test_case.stats,
        *stats->production,
        *stats->reference,
        test_case.stats_absolute_tolerance,
        test_case.stats_relative_tolerance,
        stream,
        "muDNN");
  }
  if (test_case.autotune) {
    verify_autotune_cache_hit(
        cache, [&] { return build_flagdnn_sdpa(handle, test_case); });
  }
  std::cout << test_case.name
            << ": FlagDNN Graph vs direct muDNN C++ PASS output_max_abs="
            << output_accuracy.maximum_absolute
            << " output_max_rel=" << output_accuracy.maximum_relative;
  if (stats.has_value()) {
    std::cout << " stats_max_abs=" << stats_accuracy.maximum_absolute
              << " stats_max_rel=" << stats_accuracy.maximum_relative;
  }
  std::cout << '\n';
}

void run_backward_case(const SdpaBackwardTestCase& test_case,
                       flagdnn::Handle& handle,
                       const std::filesystem::path& cache,
                       mv::Stream& stream) {
  const PathSet before = manifest_paths(cache);
  auto reference = build_sdpa_backward_reference(test_case);
  auto production = build_flagdnn_sdpa_backward(handle, test_case);
  verify_new_manifest(cache, before, {"_sdpa_bwd_dq_dbias_kernel"});
  AllocationPair q = input_pair(test_case.q, 10, stream, 0.5F);
  AllocationPair k = input_pair(test_case.k, 11, stream, 0.5F);
  AllocationPair v = input_pair(test_case.v, 12, stream, 0.5F);
  AllocationPair doutput =
      input_pair(test_case.doutput, 13, stream, 0.25F);
  std::optional<AllocationPair> bias;
  std::span<const float> bias_values;
  if (test_case.bias.has_value()) {
    bias.emplace(input_pair(*test_case.bias, 14, stream, 0.25F));
    bias_values = bias->production->logical();
  }
  const HostForward primal = host_forward(
      test_case,
      q.production->logical(),
      k.production->logical(),
      v.production->logical(),
      bias_values);
  AllocationPair output =
      value_pair(test_case.output, primal.output, stream);
  AllocationPair stats =
      value_pair(test_case.stats, primal.stats, stream);
  AllocationPair dq = output_pair(test_case.dq, stream);
  AllocationPair dk = output_pair(test_case.dk, stream);
  AllocationPair dv = output_pair(test_case.dv, stream);
  std::optional<AllocationPair> dbias;
  if (test_case.dbias.has_value()) {
    dbias.emplace(output_pair(*test_case.dbias, stream));
  }
  std::vector<flagdnnBinding_t> production_bindings;
  std::vector<flagdnnBinding_t> reference_bindings;
  append_pair(production_bindings, reference_bindings, test_case.q, q);
  append_pair(production_bindings, reference_bindings, test_case.k, k);
  append_pair(production_bindings, reference_bindings, test_case.v, v);
  if (bias.has_value()) {
    append_pair(production_bindings,
                reference_bindings,
                *test_case.bias,
                *bias);
  }
  append_pair(production_bindings,
              reference_bindings,
              test_case.output,
              output);
  append_pair(production_bindings,
              reference_bindings,
              test_case.doutput,
              doutput);
  append_pair(production_bindings,
              reference_bindings,
              test_case.stats,
              stats);
  append_pair(production_bindings, reference_bindings, test_case.dq, dq);
  append_pair(production_bindings, reference_bindings, test_case.dk, dk);
  append_pair(production_bindings, reference_bindings, test_case.dv, dv);
  if (dbias.has_value()) {
    append_pair(production_bindings,
                reference_bindings,
                *test_case.dbias,
                *dbias);
  }
  mv::DeviceBuffer production_workspace(
      production->workspace_size(), 256);
  mv::DeviceBuffer reference_workspace(reference->workspace_size(), 256);
  stream.synchronize();
  execute(*production,
          production_bindings,
          production_workspace,
          stream);
  execute(*reference, reference_bindings, reference_workspace, stream);
  mv::timing::paired(
      test_case.name, stream,
      [&] {
        execute(*production, production_bindings, production_workspace,
                stream);
      },
      [&] {
        execute(*reference, reference_bindings, reference_workspace, stream);
      });
  stream.synchronize();
  require_pair_unchanged(q, stream, "Q");
  require_pair_unchanged(k, stream, "K");
  require_pair_unchanged(v, stream, "V");
  require_pair_unchanged(output, stream, "O");
  require_pair_unchanged(doutput, stream, "dO");
  require_pair_unchanged(stats, stream, "stats");
  if (bias.has_value()) {
    require_pair_unchanged(*bias, stream, "bias");
  }
  Accuracy maximum;
  const auto compare_gradient = [&](std::string_view name,
                                    const TestTensor& tensor,
                                    const AllocationPair& values) {
    const Accuracy accuracy = compare_tensor(
        test_case.name,
        name,
        tensor,
        *values.production,
        *values.reference,
        test_case.absolute_tolerance,
        test_case.relative_tolerance,
        stream,
        "muDNN");
    maximum.maximum_absolute =
        std::max(maximum.maximum_absolute, accuracy.maximum_absolute);
    maximum.maximum_relative =
        std::max(maximum.maximum_relative, accuracy.maximum_relative);
  };
  compare_gradient("dq", test_case.dq, dq);
  compare_gradient("dk", test_case.dk, dk);
  compare_gradient("dv", test_case.dv, dv);
  if (dbias.has_value()) {
    compare_gradient("dbias", *test_case.dbias, *dbias);
  }
  if (test_case.autotune) {
    verify_autotune_cache_hit(cache, [&] {
      return build_flagdnn_sdpa_backward(handle, test_case);
    });
  }
  std::cout << test_case.name
            << ": FlagDNN Graph vs direct muDNN C++ PASS max_abs="
            << maximum.maximum_absolute
            << " max_rel=" << maximum.maximum_relative << '\n';
}

void run_fp8_forward_case(const SdpaFp8TestCase& test_case,
                          flagdnn::Handle& handle,
                          const std::filesystem::path& cache,
                          mv::Stream& stream) {
  const PathSet before = manifest_paths(cache);
  auto reference = build_sdpa_fp8_reference(test_case);
  auto production = build_flagdnn_sdpa_fp8(handle, test_case);
  verify_new_manifest(
      cache, before,
      {"_zero_sdpa_fp8_fwd_amax_kernel", "_sdpa_fp8_fwd_kernel"});
  AllocationPair q = input_pair(test_case.q, 20, stream, 1.0F);
  AllocationPair k = input_pair(test_case.k, 21, stream, 1.0F);
  AllocationPair v = input_pair(test_case.v, 22, stream, 1.0F);
  AllocationPair descale_q = scalar_pair(test_case.descale_q, stream);
  AllocationPair descale_k = scalar_pair(test_case.descale_k, stream);
  AllocationPair descale_v = scalar_pair(test_case.descale_v, stream);
  AllocationPair descale_s = scalar_pair(test_case.descale_s, stream);
  AllocationPair scale_s = scalar_pair(test_case.scale_s, stream);
  AllocationPair scale_o = scalar_pair(test_case.scale_o, stream);
  AllocationPair output = output_pair(test_case.output, stream);
  std::optional<AllocationPair> stats;
  if (test_case.stats.has_value()) {
    stats.emplace(output_pair(*test_case.stats, stream));
  }
  AllocationPair amax_s = output_pair(test_case.amax_s, stream);
  AllocationPair amax_o = output_pair(test_case.amax_o, stream);
  std::vector<flagdnnBinding_t> production_bindings;
  std::vector<flagdnnBinding_t> reference_bindings;
  append_pair(production_bindings, reference_bindings, test_case.q, q);
  append_pair(production_bindings, reference_bindings, test_case.k, k);
  append_pair(production_bindings, reference_bindings, test_case.v, v);
  append_pair(production_bindings,
              reference_bindings,
              test_case.descale_q,
              descale_q);
  append_pair(production_bindings,
              reference_bindings,
              test_case.descale_k,
              descale_k);
  append_pair(production_bindings,
              reference_bindings,
              test_case.descale_v,
              descale_v);
  append_pair(production_bindings,
              reference_bindings,
              test_case.descale_s,
              descale_s);
  append_pair(production_bindings,
              reference_bindings,
              test_case.scale_s,
              scale_s);
  append_pair(production_bindings,
              reference_bindings,
              test_case.scale_o,
              scale_o);
  append_pair(production_bindings,
              reference_bindings,
              test_case.output,
              output);
  if (stats.has_value()) {
    append_pair(production_bindings,
                reference_bindings,
                *test_case.stats,
                *stats);
  }
  append_pair(production_bindings,
              reference_bindings,
              test_case.amax_s,
              amax_s);
  append_pair(production_bindings,
              reference_bindings,
              test_case.amax_o,
              amax_o);
  mv::DeviceBuffer production_workspace(
      production->workspace_size(), 256);
  mv::DeviceBuffer reference_workspace(reference->workspace_size(), 256);
  stream.synchronize();
  execute(*production,
          production_bindings,
          production_workspace,
          stream);
  execute(*reference, reference_bindings, reference_workspace, stream);
  mv::timing::paired(
      test_case.name, stream,
      [&] {
        execute(*production, production_bindings, production_workspace,
                stream);
      },
      [&] {
        execute(*reference, reference_bindings, reference_workspace, stream);
      });
  stream.synchronize();
  for (const auto& [name, allocation] :
       std::array<std::pair<std::string_view, const AllocationPair*>, 9>{
           {{"Q", &q},
            {"K", &k},
            {"V", &v},
            {"descale_q", &descale_q},
            {"descale_k", &descale_k},
            {"descale_v", &descale_v},
            {"descale_s", &descale_s},
            {"scale_s", &scale_s},
            {"scale_o", &scale_o}}}) {
    require_pair_unchanged(*allocation, stream, name);
  }
  const Accuracy output_accuracy = compare_tensor(
      test_case.name,
      "output",
      test_case.output,
      *output.production,
      *output.reference,
      test_case.output_absolute_tolerance,
      test_case.output_relative_tolerance,
      stream,
      "FP8 oracle");
  Accuracy stats_accuracy;
  if (stats.has_value()) {
    stats_accuracy = compare_tensor(
        test_case.name,
        "stats",
        *test_case.stats,
        *stats->production,
        *stats->reference,
        test_case.stats_absolute_tolerance,
        test_case.stats_relative_tolerance,
        stream,
        "FP8 oracle");
  }
  const Accuracy amax_s_accuracy = compare_tensor(
      test_case.name,
      "amax_s",
      test_case.amax_s,
      *amax_s.production,
      *amax_s.reference,
      test_case.amax_absolute_tolerance,
      test_case.amax_relative_tolerance,
      stream,
      "FP8 oracle");
  const Accuracy amax_o_accuracy = compare_tensor(
      test_case.name,
      "amax_o",
      test_case.amax_o,
      *amax_o.production,
      *amax_o.reference,
      test_case.amax_absolute_tolerance,
      test_case.amax_relative_tolerance,
      stream,
      "FP8 oracle");
  if (test_case.autotune) {
    verify_autotune_cache_hit(
        cache, [&] { return build_flagdnn_sdpa_fp8(handle, test_case); });
  }
  std::cout << test_case.name
            << ": FlagDNN Graph vs independent FP8 oracle PASS"
            << " output_max_abs=" << output_accuracy.maximum_absolute
            << " amax_s_abs=" << amax_s_accuracy.maximum_absolute
            << " amax_o_abs=" << amax_o_accuracy.maximum_absolute;
  if (stats.has_value()) {
    std::cout << " stats_max_abs=" << stats_accuracy.maximum_absolute;
  }
  std::cout << '\n';
}

void run_fp8_backward_case(const SdpaFp8BackwardTestCase& test_case,
                           flagdnn::Handle& handle,
                           const std::filesystem::path& cache,
                           mv::Stream& stream) {
  const PathSet before = manifest_paths(cache);
  auto reference = build_sdpa_fp8_backward_reference(test_case);
  auto production = build_flagdnn_sdpa_fp8_backward(handle, test_case);
  verify_new_manifest(
      cache, before,
      {"_zero_sdpa_fp8_bwd_amax_kernel", "_sdpa_fp8_bwd_dq_kernel",
       "_sdpa_fp8_bwd_dkdv_kernel"});
  AllocationPair q = input_pair(test_case.q, 30, stream, 1.0F);
  AllocationPair k = input_pair(test_case.k, 31, stream, 1.0F);
  AllocationPair v = input_pair(test_case.v, 32, stream, 1.0F);
  AllocationPair doutput =
      input_pair(test_case.doutput, 33, stream, 0.5F);
  const std::array<const Fp8Scalar*, 12> scale_specs{{
      &test_case.descale_q,
      &test_case.descale_k,
      &test_case.descale_v,
      &test_case.descale_o,
      &test_case.descale_doutput,
      &test_case.descale_s,
      &test_case.descale_dp,
      &test_case.scale_s,
      &test_case.scale_dq,
      &test_case.scale_dk,
      &test_case.scale_dv,
      &test_case.scale_dp,
  }};
  std::vector<AllocationPair> scales;
  scales.reserve(scale_specs.size());
  for (const Fp8Scalar* scalar : scale_specs) {
    scales.push_back(scalar_pair(*scalar, stream));
  }

  const std::int64_t auxiliary_uid = test_case.amax_dp.uid + 1;
  SdpaFp8TestCase primal;
  primal.name = test_case.name + "::validation_primal";
  primal.q = test_case.q;
  primal.k = test_case.k;
  primal.v = test_case.v;
  primal.descale_q = test_case.descale_q;
  primal.descale_k = test_case.descale_k;
  primal.descale_v = test_case.descale_v;
  primal.descale_s = test_case.descale_s;
  primal.scale_s = test_case.scale_s;
  primal.scale_o = {{auxiliary_uid,
                     FLAGDNN_DATA_FLOAT32,
                     {1, 1, 1, 1},
                     {1, 1, 1, 1}},
                    1.0F / test_case.descale_o.value};
  primal.output = test_case.output;
  primal.stats = test_case.stats;
  primal.amax_s = {auxiliary_uid + 1,
                   FLAGDNN_DATA_FLOAT32,
                   {1, 1, 1, 1},
                   {1, 1, 1, 1}};
  primal.amax_o = {auxiliary_uid + 2,
                   FLAGDNN_DATA_FLOAT32,
                   {1, 1, 1, 1},
                   {1, 1, 1, 1}};
  primal.options = test_case.options;
  auto primal_reference = build_sdpa_fp8_reference(primal);
  AllocationPair primal_scale_o = scalar_pair(primal.scale_o, stream);
  AllocationPair output = output_pair(test_case.output, stream);
  AllocationPair stats = output_pair(test_case.stats, stream);
  AllocationPair primal_amax_s = output_pair(primal.amax_s, stream);
  AllocationPair primal_amax_o = output_pair(primal.amax_o, stream);
  std::vector<flagdnnBinding_t> primal_production_bindings;
  std::vector<flagdnnBinding_t> primal_reference_bindings;
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.q,
              q);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.k,
              k);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.v,
              v);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.descale_q,
              scales[0]);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.descale_k,
              scales[1]);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.descale_v,
              scales[2]);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.descale_s,
              scales[5]);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.scale_s,
              scales[7]);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.scale_o,
              primal_scale_o);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.output,
              output);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              *primal.stats,
              stats);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.amax_s,
              primal_amax_s);
  append_pair(primal_production_bindings,
              primal_reference_bindings,
              primal.amax_o,
              primal_amax_o);
  mv::DeviceBuffer primal_workspace(primal_reference->workspace_size(), 256);
  execute(*primal_reference,
          primal_production_bindings,
          primal_workspace,
          stream);
  execute(*primal_reference,
          primal_reference_bindings,
          primal_workspace,
          stream);
  stream.synchronize();
  output.production->refresh_baseline(stream);
  output.reference->refresh_baseline(stream);
  stats.production->refresh_baseline(stream);
  stats.reference->refresh_baseline(stream);

  AllocationPair dq = output_pair(test_case.dq, stream);
  AllocationPair dk = output_pair(test_case.dk, stream);
  AllocationPair dv = output_pair(test_case.dv, stream);
  const std::array<const TestTensor*, 4> amax_specs{{
      &test_case.amax_dq,
      &test_case.amax_dk,
      &test_case.amax_dv,
      &test_case.amax_dp,
  }};
  std::vector<AllocationPair> amaxes;
  amaxes.reserve(amax_specs.size());
  for (const TestTensor* amax : amax_specs) {
    amaxes.push_back(output_pair(*amax, stream));
  }
  std::vector<flagdnnBinding_t> production_bindings;
  std::vector<flagdnnBinding_t> reference_bindings;
  append_pair(production_bindings, reference_bindings, test_case.q, q);
  append_pair(production_bindings, reference_bindings, test_case.k, k);
  append_pair(production_bindings, reference_bindings, test_case.v, v);
  append_pair(production_bindings,
              reference_bindings,
              test_case.output,
              output);
  append_pair(production_bindings,
              reference_bindings,
              test_case.doutput,
              doutput);
  append_pair(production_bindings,
              reference_bindings,
              test_case.stats,
              stats);
  for (std::size_t index = 0; index < scale_specs.size(); ++index) {
    append_pair(production_bindings,
                reference_bindings,
                *scale_specs[index],
                scales[index]);
  }
  append_pair(production_bindings, reference_bindings, test_case.dq, dq);
  append_pair(production_bindings, reference_bindings, test_case.dk, dk);
  append_pair(production_bindings, reference_bindings, test_case.dv, dv);
  for (std::size_t index = 0; index < amax_specs.size(); ++index) {
    append_pair(production_bindings,
                reference_bindings,
                *amax_specs[index],
                amaxes[index]);
  }
  mv::DeviceBuffer production_workspace(
      production->workspace_size(), 256);
  mv::DeviceBuffer reference_workspace(reference->workspace_size(), 256);
  stream.synchronize();
  execute(*production,
          production_bindings,
          production_workspace,
          stream);
  execute(*reference, reference_bindings, reference_workspace, stream);
  mv::timing::paired(
      test_case.name, stream,
      [&] {
        execute(*production, production_bindings, production_workspace,
                stream);
      },
      [&] {
        execute(*reference, reference_bindings, reference_workspace, stream);
      });
  stream.synchronize();
  require_pair_unchanged(q, stream, "Q");
  require_pair_unchanged(k, stream, "K");
  require_pair_unchanged(v, stream, "V");
  require_pair_unchanged(output, stream, "O");
  require_pair_unchanged(doutput, stream, "dO");
  require_pair_unchanged(stats, stream, "stats");
  for (std::size_t index = 0; index < scales.size(); ++index) {
    require_pair_unchanged(
        scales[index], stream, "scale " + std::to_string(index));
  }
  Accuracy maximum_gradient;
  for (const auto& [name, tensor, allocation] :
       std::array<std::tuple<std::string_view,
                             const TestTensor*,
                             const AllocationPair*>,
                  3>{{{"dq", &test_case.dq, &dq},
                      {"dk", &test_case.dk, &dk},
                      {"dv", &test_case.dv, &dv}}}) {
    const Accuracy accuracy = compare_tensor(
        test_case.name,
        name,
        *tensor,
        *allocation->production,
        *allocation->reference,
        test_case.gradient_absolute_tolerance,
        test_case.gradient_relative_tolerance,
        stream,
        "FP8 oracle");
    maximum_gradient.maximum_absolute = std::max(
        maximum_gradient.maximum_absolute, accuracy.maximum_absolute);
    maximum_gradient.maximum_relative = std::max(
        maximum_gradient.maximum_relative, accuracy.maximum_relative);
  }
  Accuracy maximum_amax;
  constexpr std::array<std::string_view, 4> amax_names{
      "amax_dq", "amax_dk", "amax_dv", "amax_dp"};
  for (std::size_t index = 0; index < amax_specs.size(); ++index) {
    const Accuracy accuracy = compare_tensor(
        test_case.name,
        amax_names[index],
        *amax_specs[index],
        *amaxes[index].production,
        *amaxes[index].reference,
        test_case.amax_absolute_tolerance,
        test_case.amax_relative_tolerance,
        stream,
        "FP8 oracle");
    maximum_amax.maximum_absolute =
        std::max(maximum_amax.maximum_absolute, accuracy.maximum_absolute);
    maximum_amax.maximum_relative =
        std::max(maximum_amax.maximum_relative, accuracy.maximum_relative);
  }
  if (test_case.autotune) {
    verify_autotune_cache_hit(cache, [&] {
      return build_flagdnn_sdpa_fp8_backward(handle, test_case);
    });
  }
  std::cout << test_case.name
            << ": FlagDNN Graph vs independent FP8 oracle PASS"
            << " gradient_max_abs=" << maximum_gradient.maximum_absolute
            << " gradient_max_rel=" << maximum_gradient.maximum_relative
            << " amax_max_abs=" << maximum_amax.maximum_absolute
            << " amax_max_rel=" << maximum_amax.maximum_relative << '\n';
}

void configure_jit() {
  if (setenv("FLAGDNN_EXECUTION_ENGINE", "libtriton_jit", 1) != 0) {
    throw std::runtime_error("cannot select libtriton_jit engine");
  }
}

template <typename Case, typename Runner>
int run_suite(int argc,
              char** argv,
              std::span<const Case> cases,
              std::string_view filter_environment,
              std::string_view cache_name,
              std::string_view suite_name,
              Runner&& runner) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    configure_jit();
    mv::check_musa(musaSetDevice(0), "musaSetDevice(Attention)");
    mv::Stream stream;
    TemporaryCache cache(cache_name);
    flagdnn::Handle handle("mthreads", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    const std::string environment(filter_environment);
    const char* filter = std::getenv(environment.c_str());
    std::size_t executed = 0;
    std::size_t skipped = 0;
    std::vector<Case> selected(cases.begin(), cases.end());
    if (mv::benchmark_enabled()) {
      // The public benchmark builders also exercise longer sequences.
      const auto performance = [] {
        if constexpr (std::is_same_v<Case, SdpaTestCase>)
          return make_sdpa_benchmark_cases();
        else if constexpr (std::is_same_v<Case, SdpaBackwardTestCase>)
          return make_sdpa_backward_benchmark_cases();
        else if constexpr (std::is_same_v<Case, SdpaFp8TestCase>)
          return make_sdpa_fp8_benchmark_cases();
        else
          return make_sdpa_fp8_backward_benchmark_cases();
      }();
      selected.insert(selected.end(), performance.begin(), performance.end());
    }
    for (const Case& test_case : selected) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      try {
        runner(test_case, handle, cache.path(), stream);
        ++executed;
      } catch (const mv::ReferenceUnsupported& error) {
        ++skipped;
        mv::report_skip(test_case.name, error);
      }
    }
    if (executed + skipped == 0) {
      throw std::runtime_error(
          std::string(filter_environment) + " matched no Attention cases");
    }
    return mv::report_cases(std::string(suite_name), executed, skipped);
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace

int run_sdpa_functional_test(int argc,
                             char** argv,
                             std::span<const SdpaTestCase> cases) {
  return run_suite(argc,
                   argv,
                   cases,
                   "FLAGDNN_SDPA_CASE",
                   "sdpa",
                   "FLAGDNN_SDPA_FUNCTIONAL",
                   run_forward_case);
}

int run_sdpa_backward_functional_test(
    int argc,
    char** argv,
    std::span<const SdpaBackwardTestCase> cases) {
  return run_suite(argc,
                   argv,
                   cases,
                   "FLAGDNN_SDPA_BACKWARD_CASE",
                   "sdpa-backward",
                   "FLAGDNN_SDPA_BACKWARD_FUNCTIONAL",
                   run_backward_case);
}

int run_sdpa_fp8_functional_test(int argc, char** argv,
                                 std::span<const SdpaFp8TestCase> cases) {
  return run_suite(argc,
                   argv,
                   cases,
                   "FLAGDNN_SDPA_FP8_CASE",
                   "sdpa-fp8",
                   "FLAGDNN_SDPA_FP8_FUNCTIONAL",
                   run_fp8_forward_case);
}

int run_sdpa_fp8_backward_functional_test(
    int argc, char** argv, std::span<const SdpaFp8BackwardTestCase> cases) {
  return run_suite(argc,
                   argv,
                   cases,
                   "FLAGDNN_SDPA_FP8_BACKWARD_CASE",
                   "sdpa-fp8-backward",
                   "FLAGDNN_SDPA_FP8_BACKWARD_FUNCTIONAL",
                   run_fp8_backward_case);
}

}  // namespace flagdnn::testing
