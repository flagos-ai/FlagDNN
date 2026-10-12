/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
// CPU-only oracle/coverage guard test; links no HIP runtime or hipDNN library.
#include "pointwise_probe_data.hpp"

#include <iostream>

namespace pp = flagdnn::tools::pointwise_probe;

int main() {
  try {
    const auto cases = pp::catalog();
    std::map<std::string, std::set<std::string>> covered;
    std::size_t checked = 0, odd = 0, noncontiguous = 0;
    for (const auto &value : cases) {
      const auto &spec = value.specification;
      const auto &output = pp::bm::output_tensor(spec);
      if (pp::io::element_count(output) > 32768)
        continue;
      const auto data = pp::prepare(value);
      if (!pp::compare(value, data, data.expected_storage).passed())
        throw std::runtime_error("complete output rejected: " + spec.name);
      const auto empty = pp::compare(value, data, data.poison_storage);
      if (empty.passed() || empty.mismatches != data.expected.size() ||
          empty.unchanged != data.expected.size() || empty.nonfinite != 0)
        throw std::runtime_error("finite empty write not fully rejected: " +
                                 spec.name);
      auto partial = data.expected_storage;
      const auto bytes = pp::io::data_type_size(output.data_type);
      const auto last =
          pp::io::logical_offset(data.expected.size() - 1, output) * bytes;
      std::copy_n(data.poison_storage.begin() +
                      static_cast<std::ptrdiff_t>(last),
                  bytes, partial.begin() + static_cast<std::ptrdiff_t>(last));
      const auto tail = pp::compare(value, data, partial);
      if (tail.passed() || tail.mismatches != 1 ||
          tail.first != data.expected.size() - 1)
        throw std::runtime_error("missing tail not rejected: " + spec.name);
      covered[value.operation].insert(pp::dtype(output.data_type));
      ++checked;
      odd += data.expected.size() % 2 != 0;
      noncontiguous +=
          output.strides != pp::bm::contiguous_strides(output.dimensions);
    }
    if (covered.size() != 18 || odd == 0 || noncontiguous == 0)
      throw std::runtime_error(
          "probe CPU coverage missing operations/tails/layouts");
    const std::set<std::string> floating = {"fp32", "fp16", "bfloat16"};
    for (const auto &[operation, types] : covered)
      for (const auto &type : floating)
        if (!types.contains(type))
          throw std::runtime_error("missing floating control: " + operation +
                                   "/" + type);
    for (const auto &type : {"int32", "bool", "fp8_e4m3", "fp8_e5m2"})
      if (!covered.at("identity").contains(type))
        throw std::runtime_error("missing identity dtype coverage");

    const auto integer =
        std::find_if(cases.begin(), cases.end(), [](const auto &value) {
          return value.specification.name == "identity_int32_2x3x4";
        });
    if (integer == cases.end())
      throw std::runtime_error("missing integer exactness fixture");
    const auto integer_data = pp::prepare(*integer);
    auto rounded = integer_data.expected_storage;
    const std::int32_t wrong_integer = 16777216;
    std::memcpy(rounded.data() + 3 * sizeof(std::int32_t), &wrong_integer,
                sizeof(wrong_integer));
    const auto integer_check = pp::compare(*integer, integer_data, rounded);
    if (integer_check.mismatches != 1 || integer_check.first != 3 ||
        integer_check.max_absolute != 0)
      throw std::runtime_error("integer low-bit loss escaped bit-exact oracle");

    // Exercise genuine holes independently of the current dense catalog.
    pp::Case padded{"identity", {}, "contract"};
    padded.specification.name = "identity_contract_strided";
    padded.specification.operation = pp::bm::Operation::kPointwise;
    padded.specification.pointwise_mode = FLAGDNN_POINTWISE_IDENTITY;
    padded.specification.tensors = {
        pp::bm::strided_tensor(1, {1, 3, 5}, {31, 9, 1}),
        pp::bm::strided_tensor(2, {1, 3, 5}, {37, 11, 1})};
    const auto padded_data = pp::prepare(padded);
    auto broken = padded_data.expected_storage;
    broken.at(5 * sizeof(float)) ^= 1;
    if (pp::compare(padded, padded_data, broken).padding_errors != 1)
      throw std::runtime_error("output padding overwrite not rejected");
    std::cout << "POINTWISE_PROBE_CONTRACT: PASS catalog=" << cases.size()
              << " checked=" << checked << " odd=" << odd
              << " noncontiguous=" << noncontiguous << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "POINTWISE_PROBE_CONTRACT: FAIL " << error.what() << '\n';
    return 1;
  }
}
