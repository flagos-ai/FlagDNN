/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TOOLS_NVIDIA_SAMPLED_REFERENCE_HPP_
#define FLAGDNN_TOOLS_NVIDIA_SAMPLED_REFERENCE_HPP_

#include "../../benchmark/common/case.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

namespace standalone_cudnn {
namespace bench = flagdnn::benchmarking;
using Coordinates = std::vector<std::int64_t>;

inline Coordinates coordinates(std::size_t flat, const Coordinates& shape) {
  Coordinates result(shape.size());
  for (std::size_t i = shape.size(); i-- > 0;) {
    result[i] = flat % shape[i];
    flat /= shape[i];
  }
  return result;
}

inline std::size_t logical_offset(const Coordinates& coordinate,
                                  const Coordinates& shape) {
  std::size_t result = 0;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    result = result * shape[i] + coordinate[i];
  }
  return result;
}

inline std::size_t product(const Coordinates& dimensions,
                           std::size_t start = 0) {
  std::size_t n = 1;
  for (std::size_t i = start; i < dimensions.size(); ++i) n *= dimensions[i];
  return n;
}

// Stable xorshift samples, plus the first, middle and last logical output.
// The exact chosen indices are also serialized in each measurement record.
inline std::vector<std::size_t> sample_indices(std::size_t n,
                                              std::size_t limit) {
  std::set<std::size_t> selected{0, n - 1, n / 2};
  std::uint64_t state = 0x42f0e1eba9ea3693ULL;
  while (selected.size() < std::min(n, limit)) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    selected.insert(state % n);
  }
  return {selected.begin(), selected.end()};
}

// Inputs are logical row-major values decoded from their real device storage.
// Accumulate only the selected output elements on the CPU. The caller rounds
// these values to the output storage dtype before comparing against cuDNN.
template <typename Accumulator = float>
inline std::vector<float> sampled_reference(
    const bench::BenchmarkCase& spec,
    const std::vector<std::vector<float>>& inputs,
    const std::vector<std::size_t>& indices) {
  std::vector<float> result;
  const auto& out = spec.tensors.back().dimensions;
  if (spec.operation == bench::Operation::kMatmul) {
    const auto& a = spec.tensors[0].dimensions;
    const auto& b = spec.tensors[1].dimensions;
    const auto batch_base = [&](const Coordinates& shape,
                                const Coordinates& coord) {
      std::size_t offset = 0;
      const auto leading = out.size() - shape.size();
      for (std::size_t i = 0; i + 2 < shape.size(); ++i) {
        offset = offset * shape[i] +
                 (shape[i] == 1 ? 0 : coord[leading + i]);
      }
      return offset * shape[shape.size() - 2] * shape.back();
    };
    for (auto index : indices) {
      const auto coord = coordinates(index, out);
      const auto aa = batch_base(a, coord) + coord[out.size() - 2] * a.back();
      const auto bb = batch_base(b, coord) + coord.back();
      Accumulator value = 0;
      for (std::int64_t k = 0; k < a.back(); ++k) {
        value += static_cast<Accumulator>(inputs[0][aa + k]) * inputs[1][bb + k * b.back()];
      }
      result.push_back(value);
    }
    return result;
  }

  const bool fprop = spec.operation == bench::Operation::kConvolutionFprop;
  const bool dgrad = spec.operation == bench::Operation::kConvolutionDgrad;
  if (!fprop && !dgrad &&
      spec.operation != bench::Operation::kConvolutionWgrad) {
    throw std::invalid_argument(
        "sampled oracle supports matmul and convolution only");
  }
  const auto& x = spec.tensors[fprop ? 0 : dgrad ? 2 : 1].dimensions;
  const auto& w = spec.tensors[fprop || dgrad ? 1 : 2].dimensions;
  const auto& y = spec.tensors[fprop ? 2 : 0].dimensions;
  const auto& attr = spec.convolution;
  Coordinates ws(w.begin() + 2, w.end()), ys(y.begin() + 2, y.end());
  const auto nk = product(ws), ny = product(ys);
  const auto cg = x[1] / attr.groups, kg = y[1] / attr.groups;
  const auto kernel_coordinate = [&](Coordinates q) {
    if (attr.mode == bench::ConvolutionMode::kConvolution) {
      for (std::size_t axis = 0; axis < q.size(); ++axis) {
        q[axis] = ws[axis] - 1 - q[axis];
      }
    }
    return q;
  };

  for (auto index : indices) {
    const auto coord = coordinates(index, out);
    Accumulator value = 0;
    if (fprop) {
      const auto n = coord[0], k = coord[1], group = k / kg;
      for (std::int64_t c = 0; c < cg; ++c) {
        for (std::size_t rf = 0; rf < nk; ++rf) {
          auto r = coordinates(rf, ws), wr = kernel_coordinate(r);
          Coordinates xc(x.size());
          xc[0] = n;
          xc[1] = group * cg + c;
          bool valid = true;
          for (std::size_t d = 0; d < ws.size(); ++d) {
            xc[d + 2] = coord[d + 2] * attr.stride[d] - attr.pre_padding[d] +
                        r[d] * attr.dilation[d];
            valid &= xc[d + 2] >= 0 && xc[d + 2] < x[d + 2];
          }
          if (valid) {
            Coordinates wc{k, c};
            wc.insert(wc.end(), wr.begin(), wr.end());
            value += static_cast<Accumulator>(inputs[0][logical_offset(xc, x)]) *
                     inputs[1][logical_offset(wc, w)];
          }
        }
      }
    } else if (dgrad) {
      const auto n = coord[0], c = coord[1], group = c / cg;
      for (std::int64_t k = group * kg; k < (group + 1) * kg; ++k) {
        for (std::size_t rf = 0; rf < nk; ++rf) {
          auto r = coordinates(rf, ws), wr = kernel_coordinate(r);
          Coordinates yc(y.size());
          yc[0] = n;
          yc[1] = k;
          bool valid = true;
          for (std::size_t d = 0; d < ws.size(); ++d) {
            const auto t = coord[d + 2] + attr.pre_padding[d] -
                           r[d] * attr.dilation[d];
            valid &= t >= 0 && t % attr.stride[d] == 0;
            yc[d + 2] = t / attr.stride[d];
            valid &= yc[d + 2] < y[d + 2];
          }
          if (valid) {
            Coordinates wc{k, c % cg};
            wc.insert(wc.end(), wr.begin(), wr.end());
            value += static_cast<Accumulator>(inputs[0][logical_offset(yc, y)]) *
                     inputs[1][logical_offset(wc, w)];
          }
        }
      }
    } else {
      const auto k = coord[0], c = (k / kg) * cg + coord[1];
      const auto r =
          kernel_coordinate(Coordinates(coord.begin() + 2, coord.end()));
      for (std::int64_t n = 0; n < x[0]; ++n) {
        for (std::size_t yf = 0; yf < ny; ++yf) {
          const auto q = coordinates(yf, ys);
          Coordinates xc(x.size());
          xc[0] = n;
          xc[1] = c;
          bool valid = true;
          for (std::size_t d = 0; d < ws.size(); ++d) {
            xc[d + 2] = q[d] * attr.stride[d] - attr.pre_padding[d] +
                        r[d] * attr.dilation[d];
            valid &= xc[d + 2] >= 0 && xc[d + 2] < x[d + 2];
          }
          if (valid) {
            value += static_cast<Accumulator>(inputs[0][(n * y[1] + k) * ny + yf]) *
                     inputs[1][logical_offset(xc, x)];
          }
        }
      }
    }
    result.push_back(value);
  }
  return result;
}
}  // namespace standalone_cudnn
#endif  // FLAGDNN_TOOLS_NVIDIA_SAMPLED_REFERENCE_HPP_
