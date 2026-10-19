/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "cpu_cudnn.hpp"

#include <algorithm>
#include <cudnn.h>
#include <cstdlib>
#include <limits>

namespace flagdnn::reference::tests {
Values quantize(std::span<const float> x, flagdnnDataType_t type) {
  return io::decode(io::encode(x, type, encoding), type, x.size(), encoding);
}
Arrays inputs_for(const std::vector<t::TestTensor>& tensors) {
  Arrays result;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    Values values(io::element_count(tensors[i]));
    for (std::size_t j = 0; j < values.size(); ++j)
      values[j] = float(int((j * 17 + i * 11) % 41) - 20) / float(13 + i);
    result.push_back(quantize(values, tensors[i].data_type));
  }
  return result;
}
void compare(std::span<const float> actual, std::span<const float> expected,
             double atol, double rtol) {
  if (actual.size() != expected.size() || actual.empty())
    throw std::runtime_error("empty output or size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double a = actual[i], b = expected[i], error = std::abs(a - b);
    if (!std::isfinite(a) || !std::isfinite(b) ||
        !(error <= atol || error <= rtol * std::max(std::abs(a), std::abs(b))))
      throw std::runtime_error("element " + std::to_string(i) +
          ": cuDNN=" + std::to_string(a) + " CPU=" + std::to_string(b) +
          " atol=" + std::to_string(atol) + " rtol=" + std::to_string(rtol));
  }
}
t::TestTensor tensor(std::int64_t uid, std::vector<std::int64_t> shape,
                     flagdnnDataType_t type) {
  auto strides = shape;
  std::int64_t stride = 1;
  for (std::size_t i = shape.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= shape[i - 1];
  }
  return {uid, type, std::move(shape), std::move(strides)};
}
std::vector<std::vector<std::int64_t>> shapes(
    const std::vector<t::TestTensor>& tensors) {
  std::vector<std::vector<std::int64_t>> result;
  for (const auto& spec : tensors) result.push_back(spec.dimensions);
  return result;
}
void Suite::run(const std::string& name, const std::function<void()>& action) {
  try {
    action();
    ++passed;
    std::cout << "PASS " << name << std::endl;
  } catch (const std::exception& error) {
    ++failed;
    std::cerr << "FAIL " << name << ": " << error.what() << std::endl;
  }
}
Result Suite::execute(const std::vector<t::TestTensor>& inputs,
                      const std::vector<t::TestTensor>& outputs,
                      std::unique_ptr<t::TestExecutable> op, Arrays values, RawArrays raw_inputs) {
  if (!op || outputs.empty()) throw std::invalid_argument("missing cuDNN output");
  if (values.empty()) values = inputs_for(inputs);
  if (values.size() != inputs.size()) throw std::invalid_argument("input count");
  if (!raw_inputs.empty() && raw_inputs.size() != inputs.size())
    throw std::invalid_argument("raw input count");
  auto tensors = inputs;
  tensors.insert(tensors.end(), outputs.begin(), outputs.end());
  std::vector<std::vector<std::uint8_t>> initial;
  std::vector<std::unique_ptr<gpu::DeviceBuffer>> devices;
  std::vector<flagdnnBinding_t> bindings;
  constexpr std::size_t guard = 256;
  std::set<std::int64_t> uids;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const auto& spec = tensors[i];
    if (!uids.insert(spec.uid).second) throw std::invalid_argument("duplicate UID");
    Values logical(io::element_count(spec), -97.0F);
    if (i < inputs.size()) {
      if (values[i].size() != logical.size()) throw std::invalid_argument("input size");
      logical = values[i] = quantize(values[i], spec.data_type);
    }
    auto bytes = io::encode(io::scatter(logical, spec), spec.data_type, encoding);
    if (i < inputs.size() && !raw_inputs.empty()) {
      const auto width = io::data_type_size(spec.data_type);
      if (spec.data_type == FLAGDNN_DATA_BOOLEAN ||
          raw_inputs[i].size() != logical.size() * width)
        throw std::invalid_argument("raw input size/type");
      for (std::size_t j = 0; j < logical.size(); ++j)
        std::copy_n(raw_inputs[i].begin() + j * width, width,
                    bytes.begin() + io::logical_offset(j, spec) * width);
      values[i] = io::decode(raw_inputs[i], spec.data_type, logical.size(), encoding);
    }
    const auto offset = guard + spec.binding_byte_offset;
    initial.emplace_back(offset + bytes.size() + guard, 0xa5);
    std::copy(bytes.begin(), bytes.end(), initial.back().begin() + offset);
    devices.push_back(std::make_unique<gpu::DeviceBuffer>(initial.back().size()));
    devices.back()->copy_from_host(initial.back().data(), initial.back().size(), stream.get());
    bindings.push_back({spec.uid, devices.back()->opaque_at(offset)});
  }
  stream.synchronize();
  op->prepare(bindings, stream.opaque());
  gpu::DeviceBuffer workspace(op->workspace_size());
  op->execute(bindings, workspace.opaque(), op->workspace_size(), stream.opaque());
  stream.synchronize();
  Result result{std::move(values), {}, {}, {}};
  for (std::size_t i = 0; i < inputs.size(); ++i)
    result.input_bytes.push_back(raw_inputs.empty() ?
        io::encode(result.inputs[i], inputs[i].data_type, io::BooleanEncoding::kByte) :
        raw_inputs[i]);
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    auto read = initial[i];
    devices[i]->copy_to_host(read.data(), read.size(), stream.get());
    stream.synchronize();
    if (i < inputs.size()) {
      if (read != initial[i]) throw std::runtime_error("cuDNN modified an input");
      continue;
    }
    const auto& spec = tensors[i];
    const auto offset = guard + spec.binding_byte_offset;
    const auto bytes = io::encoded_byte_count(spec, encoding);
    if (!std::equal(read.begin(), read.begin() + offset, initial[i].begin()) ||
        !std::equal(read.begin() + offset + bytes, read.end(),
                    initial[i].begin() + offset + bytes))
      throw std::runtime_error("output guard overwritten");
    const auto physical = io::decode(std::span(read).subspan(offset, bytes),
        spec.data_type, io::storage_element_count(spec), encoding);
    std::vector<bool> occupied(io::storage_element_count(spec), false);
    Bytes logical_bytes;
    const auto width = io::data_type_size(spec.data_type);
    for (std::size_t j = 0; j < io::element_count(spec); ++j) {
      const auto index = io::logical_offset(j, spec);
      occupied[index] = true;
      if (spec.data_type == FLAGDNN_DATA_BOOLEAN)
        logical_bytes.push_back((read[offset + index / 8] >> (index % 8)) & 1U);
      else logical_bytes.insert(logical_bytes.end(), read.begin() + offset + index * width,
                                read.begin() + offset + (index + 1) * width);
    }
    // Compare representation bytes, since the sentinel rounds in FP8/Boolean.
    for (std::size_t j = 0; j < occupied.size(); ++j) {
      if (occupied[j]) continue;
      if (spec.data_type == FLAGDNN_DATA_BOOLEAN) {
        if (((read[offset + j / 8] ^ initial[i][offset + j / 8]) >> (j % 8)) & 1U)
          throw std::runtime_error("cuDNN modified output padding bit");
      } else if (!std::equal(read.begin() + offset + j * width,
                            read.begin() + offset + (j + 1) * width,
                            initial[i].begin() + offset + j * width))
        throw std::runtime_error("cuDNN modified output padding bytes");
    }
    result.output_bytes.push_back(std::move(logical_bytes));
    result.outputs.push_back(io::gather(physical, spec));
  }
  return result;
}
Result Suite::check(const std::vector<t::TestTensor>& inputs,
                    const std::vector<t::TestTensor>& outputs,
                    std::unique_ptr<t::TestExecutable> executable,
                    const Oracle& oracle, double atol, double rtol,
                    Arrays values) {
  auto result = execute(inputs, outputs, std::move(executable), std::move(values));
  const auto expected = oracle(result.inputs);
  if (expected.size() != outputs.size()) throw std::runtime_error("CPU output count");
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto rounded = quantize(expected[i], outputs[i].data_type);
    try {
      compare(result.outputs[i], rounded, atol, rtol);
    } catch (const std::exception& error) {
      throw std::runtime_error("output " + std::to_string(i) + ": " + error.what());
    }
  }
  return result;
}
}  // namespace flagdnn::reference::tests

int main(int argc, char** argv) {
  namespace ref = flagdnn::reference::tests;
  if (argc != 2) { std::cerr << "usage: cpu_cudnn FAMILY\n"; return 2; }
  try {
    ref::gpu::DriverContext context;
    ref::Suite suite;
    std::cout << "cuDNN runtime version=" << cudnnGetVersion() << std::endl;
    // Make sure a broken comparator cannot silently bless the CPU implementation.
    for (const auto& bad : {ref::Values{9.0F}, ref::Values{NAN}, ref::Values{}}) {
      bool rejected = false;
      try { ref::compare(bad, ref::Values{1.0F}, 1e-5, 1e-5); }
      catch (const std::runtime_error&) { rejected = true; }
      if (!rejected) throw std::runtime_error("comparison negative control failed");
    }
    const std::string family = argv[1];
    if (family == "matrix") ref::matrix(suite);
    else if (family == "matrix_tf32") ref::matrix_tf32(suite);
    else if (family == "normalization") ref::normalization(suite);
    else if (family == "attention") ref::attention(suite);
    else if (family == "storage") ref::storage(suite);
    else if (family == "pointwise") ref::pointwise(suite);
    else if (family == "formulas") ref::formulas(suite);
    else ref::extended(suite, family);
    std::cout << "TOTAL passed=" << suite.passed << " failed=" << suite.failed << '\n';
    if (suite.failed) return 1;

    return suite.passed == 0 ? 1 : 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n'; return 1;
  }
}
