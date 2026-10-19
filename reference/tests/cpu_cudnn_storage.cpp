/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "cpu_cudnn_benchmark.hpp"
#include "common/layout.hpp"
#include "common/index.hpp"
#include "reference/cpu/layout.hpp"
#include "reference/cpu/pointwise.hpp"
#include <array>
#include "reference/cpu/index.hpp"
#include "validation/functional/cudnn_extended.hpp"

#include <cstring>
#include <tuple>

namespace flagdnn::reference::tests {
void storage(Suite& suite) {
  for (const auto type : {FLAGDNN_DATA_INT32, FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16,
                          FLAGDNN_DATA_BFLOAT16, FLAGDNN_DATA_FP8_E4M3, FLAGDNN_DATA_FP8_E5M2}) {
    suite.run("identity_exact_storage_dtype_" + std::to_string(type), [&] {
      t::PointwiseTestCase c;
      c.name = "identity_exact_storage"; c.mode = FLAGDNN_POINTWISE_IDENTITY;
      c.inputs = {tensor(1, {1, 64, 1, 1}, type)};
      c.output = tensor(2, {1, 64, 1, 1}, type);
      c.input_domains = {t::PointwiseInputDomain::kReal};
      Bytes raw;
      if (type == FLAGDNN_DATA_INT32) {
        const std::array<std::int32_t, 8> patterns{
            0, 1, -1, 16777217, -16777219, 2147483647, -2147483647, -2147483647 - 1};
        for (std::size_t i = 0; i < 64; ++i) {
          const auto value = patterns[i % patterns.size()];
          const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
          raw.insert(raw.end(), bytes, bytes + sizeof(value));
        }
      } else {
        Values values(64);
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] = i % 7 == 0 ? -0.0F : float(int(i % 11) - 5) / 4;
        raw = io::encode(values, type, encoding);
      }
      const auto r = suite.execute(c.inputs, {c.output}, t::build_pointwise_reference(c), {}, {raw});
      if (r.output_bytes[0] != cpu::evaluate_identity_bytes(raw))
        throw std::runtime_error("identity changed representation bits");
    });
  }

  for (const auto op : {t::LayoutOperation::kReshape, t::LayoutOperation::kTranspose,
                        t::LayoutOperation::kSlice}) {
    std::set<std::tuple<int, std::size_t, bool>> covered;
    for (const auto& c : t::make_layout_cases(op)) {
      // cuDNN has no standalone Boolean layout transform. Byte-preserving
      // CPU layout logic is covered with the supported 2/4-byte dtypes.
      if (c.input.data_type == FLAGDNN_DATA_BOOLEAN) continue;
      const auto key = std::tuple{int(c.input.data_type), c.input.dimensions.size(),
                                 c.input.binding_byte_offset != 0};
      if (io::element_count(c.input) > 100000 || !covered.insert(key).second) continue;
      suite.run(c.name, [&] {
        const auto name = op == t::LayoutOperation::kReshape ? "reshape" :
                          op == t::LayoutOperation::kTranspose ? "transpose" : "slice";
        cpu::LayoutParameters p{name, c.input.dimensions, c.output.dimensions,
                                c.permutation, c.slices, c.slice_strides};
        const auto r = suite.check({c.input}, {c.output}, t::build_layout_reference(c),
            [&](const Arrays& x) {
              const auto logical = cpu::evaluate_layout(p, x[0]);
              const auto bytes = io::encode(x[0], c.input.data_type, encoding);
              const auto copied = cpu::evaluate_layout_bytes(p, bytes, io::data_type_size(c.input.data_type));
              compare(io::decode(copied, c.output.data_type, logical.size(), encoding),
                      logical, 0, 0);
              for (std::size_t j = 0; j < logical.size(); ++j)
                if (logical[j] != x[0].at(cpu::layout_source_index(p, j)))
                  throw std::runtime_error("layout source mapping mismatch");
              return Arrays{logical};
            }, 0, 0);
        auto spec = formula_case(op == t::LayoutOperation::kReshape ? b::Operation::kReshape :
            op == t::LayoutOperation::kTranspose ? b::Operation::kTranspose : b::Operation::kSlice,
            {c.input}, {c.output});
        spec.reshape.dimensions = c.output.dimensions;
        spec.reshape.strides = c.output.strides;
        spec.transpose.permutation = c.permutation;
        spec.slice.slices = c.slices; spec.slice.strides = c.slice_strides;
        check_formula(spec, r, 0, 0);
      });
    }
  }
  for (const auto op : {"concatenate", "gen_index"}) {
    std::set<std::tuple<int, int, std::size_t>> covered;
    for (const auto& c : t::make_index_cases(op)) {
      const auto key = std::tuple{int(c.output.data_type), int(c.axis), c.inputs.size()};
      if (io::element_count(c.output) > 100000 || !covered.insert(key).second) continue;
      suite.run(c.name, [&] {
        cpu::IndexParameters p{c.operation, shapes(c.inputs), c.output.dimensions, c.axis};
        suite.check(c.inputs, {c.output}, t::build_cudnn_index(c),
            [&](const Arrays& x) {
              std::vector<std::vector<std::byte>> raw;
              std::vector<std::vector<std::uint8_t>> bytes;
              // CPU byte references use one byte per Boolean element.
              for (std::size_t i = 0; i < x.size(); ++i) {
                bytes.push_back(io::encode(x[i], c.inputs[i].data_type,
                                          io::BooleanEncoding::kByte));
                raw.emplace_back(bytes.back().size());
                std::memcpy(raw.back().data(), bytes.back().data(), bytes.back().size());
              }
              const auto out = cpu::evaluate_index_bytes(p, raw, c.output.data_type);
              std::vector<std::uint8_t> output(out.size());
              std::memcpy(output.data(), out.data(), out.size());
              if (c.operation == "concatenate" &&
                  output != cpu::evaluate_concatenate_bytes(
                      p, bytes, io::data_type_size(c.output.data_type)))
                throw std::runtime_error("concatenate byte entry points disagree");
              auto result = io::decode(output, c.output.data_type, io::element_count(c.output),
                                       io::BooleanEncoding::kByte);
              for (std::size_t j = 0; j < result.size(); ++j) {
                const auto [input, index] = cpu::index_source(p, j);
                const float expected = c.operation == "gen_index" ? float(index) : x.at(input).at(index);
                if (result[j] != expected) throw std::runtime_error("index mapping mismatch");
              }
              return Arrays{result};
            }, 0, 0);
      });
    }
  }
}
}  // namespace flagdnn::reference::tests
