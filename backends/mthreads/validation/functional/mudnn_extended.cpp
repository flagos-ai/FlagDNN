/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/mthreads/validation/functional/mudnn_extended.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace flagdnn::testing::mthreads {
namespace io = mv::tensor_io;
namespace {
Shape contiguous(const Shape& shape) {
  Shape strides(shape.size());
  std::int64_t stride = 1;
  for (std::size_t i = shape.size(); i > 0; --i) {
    strides[i - 1] = stride;
    stride *= shape[i - 1];
  }
  return strides;
}
md::Tensor::Type dtype(flagdnnDataType_t type) {
  switch (type) {
    case FLAGDNN_DATA_FLOAT32:
      return md::Tensor::Type::FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return md::Tensor::Type::HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return md::Tensor::Type::BFLOAT16;
    case FLAGDNN_DATA_INT32:
      return md::Tensor::Type::INT32;
    case FLAGDNN_DATA_BOOLEAN:
      return md::Tensor::Type::BOOL;
    case FLAGDNN_DATA_FP8_E4M3:
      return md::Tensor::Type::FP8_E4M3;
    case FLAGDNN_DATA_FP8_E5M2:
      return md::Tensor::Type::FP8_E5M2;
    case FLAGDNN_DATA_FP8_E8M0:
      throw mv::ReferenceUnsupported(
          "muDNN 3.1.5 Tensor::Type has no FP8 E8M0 type");
  }
  throw std::invalid_argument("unknown tensor dtype");
}
}  // namespace
NativeProgram::NativeProgram(std::span<const TestTensor> inputs,
                             std::span<const TestTensor> outputs,
                             bool raw_storage)
    : raw_storage_(raw_storage),
      input_count_(inputs.size()),
      external_count_(inputs.size() + outputs.size()) {
  for (const auto& tensor : inputs) append(tensor);
  for (const auto& tensor : outputs) append(tensor);
}
std::size_t NativeProgram::append(TestTensor descriptor) {
  const auto index = tensors_.size();
  tensors_.emplace_back();
  auto& tensor = tensors_.back();
  mv::check_mudnn(
      tensor.SetType(raw_storage_ &&
                             (descriptor.data_type == FLAGDNN_DATA_FP8_E4M3 ||
                              descriptor.data_type == FLAGDNN_DATA_FP8_E5M2 ||
                              descriptor.data_type == FLAGDNN_DATA_FP8_E8M0)
                         ? md::Tensor::Type::UINT8
                         : dtype(descriptor.data_type)),
      "muDNN tensor type");
  mv::check_mudnn(tensor.SetNdInfo(
                      static_cast<int>(descriptor.dimensions.size()),
                      descriptor.dimensions.data(), descriptor.strides.data()),
                  "muDNN tensor dimensions");
  const bool channels_last =
      descriptor.strides.size() > 1 && descriptor.strides[1] == 1;
  if (descriptor.dimensions.size() == 3)
    mv::check_mudnn(tensor.SetFormat(channels_last ? md::Tensor::Format::NWC
                                                   : md::Tensor::Format::NCW),
                    "muDNN tensor format");
  if (descriptor.dimensions.size() == 4)
    mv::check_mudnn(tensor.SetFormat(channels_last ? md::Tensor::Format::NHWC
                                                   : md::Tensor::Format::NCHW),
                    "muDNN tensor format");
  if (descriptor.dimensions.size() == 5)
    mv::check_mudnn(
        tensor.SetFormat(channels_last ? md::Tensor::Format::NDHWC
                                       : md::Tensor::Format::NCDHW),
        "muDNN tensor format");
  descriptors_.push_back(std::move(descriptor));
  addresses_.push_back(nullptr);
  return index;
}
std::size_t NativeProgram::temporary(const Shape& shape,
                                     flagdnnDataType_t type) {
  const auto index = append({-1, type, shape, contiguous(shape)});
  const auto bytes =
      io::encode(std::vector<float>(io::element_count(descriptors_[index])),
                 type)
          .size();
  buffers_.push_back(std::make_unique<mv::DeviceBuffer>(bytes, 256));
  addresses_[index] = buffers_.back()->opaque();
  mv::check_mudnn(tensors_[index].SetAddr(addresses_[index]),
                  "muDNN temporary address");
  return index;
}
std::size_t NativeProgram::view(std::size_t parent, const Shape& shape,
                                const Shape& strides, std::size_t offset) {
  const auto index =
      append({-1, descriptors_[parent].data_type, shape, strides});
  views_.push_back({index, parent, offset});
  return index;
}
md::MemoryMaintainer NativeProgram::maintainer() {
  return [this](std::size_t size) {
    if (!size) return md::MemoryHandler(nullptr, [](void*) {});
    if (workspace_cursor_ == workspace_buffers_.size())
      workspace_buffers_.push_back(
          std::make_unique<mv::DeviceBuffer>(size, 256));
    auto& buffer = workspace_buffers_.at(workspace_cursor_++);
    if (size > buffer->size())
      throw std::runtime_error("muDNN workspace changed after warmup");
    return md::MemoryHandler(buffer->opaque(), [](void*) {});
  };
}
void NativeProgram::execute(std::span<const flagdnnBinding_t> bindings, void*,
                            std::size_t, flagdnnStream_t stream) {
  mv::check_mudnn(handle_.SetStream(reinterpret_cast<musaStream_t>(stream)),
                  "muDNN stream");
  workspace_cursor_ = 0;
  for (std::size_t i = 0; i < external_count_; ++i) {
    const auto found = std::find_if(
        bindings.begin(), bindings.end(),
        [&](const auto& b) { return b.uid == descriptors_[i].uid; });
    if (found == bindings.end())
      throw std::runtime_error("missing native tensor binding");
    addresses_[i] = found->device_pointer;
    mv::check_mudnn(tensors_[i].SetAddr(addresses_[i]),
                    "muDNN binding address");
  }
  for (const auto& view : views_) {
    addresses_[view.index] =
        static_cast<std::byte*>(addresses_[view.parent]) + view.offset;
    mv::check_mudnn(tensors_[view.index].SetAddr(addresses_[view.index]),
                    "muDNN view address");
  }
  for (const auto& op : operations_) op();
}
void NativeProgram::unary_to(std::size_t out, Unary mode, std::size_t in,
                             double alpha) {
  if (mode == Unary::IDENTITY &&
      descriptor(out).data_type == descriptor(in).data_type) {
    auto op = std::make_shared<md::Permute>();
    add([this, op, out, in] {
      mv::check_mudnn(op->Run(handle_, tensor(out), tensor(in)),
                      "muDNN Permute::Run");
    });
    return;
  }
  auto op = std::make_shared<md::Unary>();
  if (mode == Unary::IDENTITY) mode = Unary::CAST;
  mv::check_mudnn(op->SetMode(mode), "muDNN Unary mode");
  mv::check_mudnn(op->SetAlpha(alpha), "muDNN Unary alpha");
  add([this, op, out, in] {
    mv::check_mudnn(op->Run(handle_, tensor(out), tensor(in)),
                    "muDNN Unary::Run");
  });
}
std::size_t NativeProgram::unary(Unary mode, std::size_t in, double alpha) {
  const auto out = temporary(descriptor(in).dimensions);
  unary_to(out, mode, in, alpha);
  return out;
}
void NativeProgram::binary_to(std::size_t out, Binary mode, std::size_t a,
                              std::size_t b) {
  auto op = std::make_shared<md::Binary>();
  mv::check_mudnn(op->SetMode(mode), "muDNN Binary mode");
  add([this, op, out, a, b] {
    mv::check_mudnn(op->Run(handle_, tensor(out), tensor(a), tensor(b)),
                    "muDNN Binary::Run");
  });
}
std::size_t NativeProgram::binary(Binary mode, std::size_t a, std::size_t b) {
  auto shape = descriptor(a).dimensions;
  const auto other = descriptor(b).dimensions;
  if (shape.size() != other.size())
    throw std::runtime_error("reference broadcast rank mismatch");
  for (std::size_t i = 0; i < shape.size(); ++i)
    shape[i] = std::max(shape[i], other[i]);
  const auto out = temporary(shape);
  binary_to(out, mode, a, b);
  return out;
}
void NativeProgram::reduce_to(std::size_t out, Reduce mode, std::size_t in,
                              const std::vector<int>& axes) {
  if (axes.empty()) {
    unary_to(out, Unary::IDENTITY, in);
    return;
  }
  // Reduce expects dense storage on this SDK. Materialize with native
  // Permute so public input/output strides and their padding are preserved.
  if (descriptor(in).strides != contiguous(descriptor(in).dimensions)) {
    const auto dense =
        temporary(descriptor(in).dimensions, descriptor(in).data_type);
    unary_to(dense, Unary::IDENTITY, in);
    reduce_to(out, mode, dense, axes);
    return;
  }
  if (descriptor(out).strides != contiguous(descriptor(out).dimensions)) {
    const auto dense =
        temporary(descriptor(out).dimensions, descriptor(out).data_type);
    reduce_to(dense, mode, in, axes);
    unary_to(out, Unary::IDENTITY, dense);
    return;
  }
  auto op = std::make_shared<md::Reduce>();
  mv::check_mudnn(op->SetMode(mode), "muDNN Reduce mode");
  mv::check_mudnn(op->SetDim(static_cast<int>(axes.size()), axes.data()),
                  "muDNN Reduce axes");
  add([this, op, out, in] {
    mv::check_mudnn(op->Run(handle_, tensor(out), tensor(in), maintainer()),
                    "muDNN Reduce::Run");
  });
}
std::size_t NativeProgram::reduce(Reduce mode, std::size_t in,
                                  const std::vector<int>& axes) {
  auto shape = descriptor(in).dimensions;
  for (auto axis : axes) shape.at(static_cast<std::size_t>(axis)) = 1;
  const auto out = temporary(shape);
  reduce_to(out, mode, in, axes);
  return out;
}
std::size_t NativeProgram::floating(std::size_t in) {
  return descriptor(in).data_type == FLAGDNN_DATA_FLOAT32
             ? in
             : unary(Unary::IDENTITY, in);
}

std::unique_ptr<TestExecutable> reference(const IndexTestCase& c) {
  const std::array outputs{c.output};
  if (c.operation == "gen_index") {
    auto p = std::make_unique<NativeProgram>(c.inputs, outputs);
    const auto rank = static_cast<std::int64_t>(c.output.dimensions.size());
    const auto axis = c.axis < 0 ? c.axis + rank : c.axis;
    if (axis < 0 || axis >= rank)
      throw std::invalid_argument("muDNN gen_index axis is out of range");
    const auto extent = c.output.dimensions[axis];
    const auto ones = p->temporary({extent}, c.output.data_type);
    const bool direct = rank == 1 && c.output.strides[0] == 1;
    const auto indices =
        direct ? p->output(0) : p->temporary({extent}, c.output.data_type);
    auto fill = std::make_shared<md::Fill>();
    mv::check_mudnn(c.output.data_type == FLAGDNN_DATA_INT32
                        ? fill->SetValue(std::int64_t{1})
                        : fill->SetValue(1.0),
                    "muDNN Fill::SetValue(gen_index)");
    p->add([r = p.get(), fill, ones] {
      mv::check_mudnn(fill->Run(r->handle(), r->tensor(ones)),
                      "muDNN Fill::Run(gen_index ones)");
    });
    auto scan = std::make_shared<md::Scan>();
    mv::check_mudnn(scan->SetMode(md::Scan::Mode::EXCLUSIVE),
                    "muDNN Scan::SetMode(gen_index)");
    mv::check_mudnn(scan->SetOpType(md::Scan::ScanOpType::ADD),
                    "muDNN Scan::SetOpType(gen_index)");
    mv::check_mudnn(c.output.data_type == FLAGDNN_DATA_INT32
                        ? scan->SetInitVal(std::int64_t{0})
                        : scan->SetInitVal(0.0),
                    "muDNN Scan::SetInitVal(gen_index)");
    p->add([r = p.get(), scan, ones, indices] {
      mv::check_mudnn(scan->Run(r->handle(), r->tensor(indices),
                                r->tensor(ones), r->maintainer()),
                      "muDNN Scan::Run(gen_index)");
    });
    if (!direct) {
      Shape broadcast_shape(rank, 1);
      broadcast_shape[axis] = extent;
      const auto broadcast =
          p->view(indices, broadcast_shape, contiguous(broadcast_shape));
      const auto zeros = p->temporary(c.output.dimensions, c.output.data_type);
      auto zero = std::make_shared<md::Fill>();
      mv::check_mudnn(zero->SetValue(0.0),
                      "muDNN Fill::SetValue(gen_index zeros)");
      p->add([r = p.get(), zero, zeros] {
        mv::check_mudnn(zero->Run(r->handle(), r->tensor(zeros)),
                        "muDNN Fill::Run(gen_index zeros)");
      });
      p->binary_to(p->output(0), Binary::ADD, broadcast, zeros);
    }
    return p;
  }
  auto p = std::make_unique<NativeProgram>(c.inputs, outputs, true);
  auto op = std::make_shared<md::Concat>();
  mv::check_mudnn(op->SetAxis(static_cast<int>(c.axis)), "muDNN Concat axis");
  std::vector<std::size_t> contiguous_inputs;
  for (std::size_t i = 0; i < c.inputs.size(); ++i) {
    const auto dense =
        p->temporary(c.inputs[i].dimensions, c.inputs[i].data_type);
    p->unary_to(dense, Unary::IDENTITY, i);
    contiguous_inputs.push_back(dense);
  }
  const auto out = p->temporary(c.output.dimensions, c.output.data_type);
  p->add([r = p.get(), op, contiguous_inputs, out] {
    std::vector<md::Tensor> inputs;
    for (const auto i : contiguous_inputs) inputs.push_back(r->tensor(i));
    mv::check_mudnn(op->Run(r->handle(), r->tensor(out),
                            static_cast<int>(inputs.size()), inputs.data()),
                    "muDNN Concat::Run");
  });
  p->unary_to(p->output(0), Unary::IDENTITY, out);
  return p;
}
std::unique_ptr<TestExecutable> reference(
    const ExtendedNormalizationTestCase& c) {
  auto p = std::make_unique<NativeProgram>(c.inputs, c.outputs);
  std::vector<int> axes(c.axes.begin(), c.axes.end());
  const bool backward = c.operation.ends_with("_backward");
  const bool rms = c.operation == "rmsnorm_backward";
  if (!backward) {
    const auto x = p->floating(0);
    p->reduce_to(p->output(1), Reduce::MEAN, x, axes);
    const auto center = p->binary(Binary::SUB, x, p->output(1));
    const auto variance =
        p->reduce(Reduce::MEAN, p->unary(Unary::SQUARE, center), axes);
    p->unary_to(p->output(2), Unary::RSQRT,
                p->unary(Unary::ADD, variance, c.epsilon));
    const auto z = p->binary(Binary::MUL, center, p->output(2));
    const auto y = p->binary(Binary::ADD, p->binary(Binary::MUL, z, 1), 2);
    p->unary_to(p->output(0), Unary::IDENTITY, y);
  } else {
    const auto dy = p->floating(0), x = p->floating(1);
    const auto inv = rms ? 3U : 4U;
    const auto center = rms ? x : p->binary(Binary::SUB, x, 3);
    const auto z = p->binary(Binary::MUL, center, inv);
    std::vector<int> parameter_axes;
    for (std::size_t i = 0; i < c.inputs[1].dimensions.size(); ++i)
      if (c.inputs[2].dimensions[i] == 1 && c.inputs[1].dimensions[i] != 1)
        parameter_axes.push_back(static_cast<int>(i));
    p->reduce_to(p->output(1), Reduce::ADD, p->binary(Binary::MUL, dy, z),
                 parameter_axes);
    p->reduce_to(p->output(2), Reduce::ADD, dy, parameter_axes);
    const auto weighted = p->binary(Binary::MUL, dy, 2);
    const auto correction = p->binary(
        Binary::MUL, z,
        p->reduce(Reduce::MEAN, p->binary(Binary::MUL, weighted, z), axes));
    auto dx = p->binary(Binary::SUB, weighted, correction);
    if (!rms)
      dx = p->binary(Binary::SUB, dx, p->reduce(Reduce::MEAN, weighted, axes));
    p->unary_to(p->output(0), Unary::IDENTITY,
                p->binary(Binary::MUL, dx, inv));
  }
  return p;
}
std::unique_ptr<TestExecutable> reference(const StatisticsTestCase& c) {
  auto p = std::make_unique<NativeProgram>(c.inputs, c.outputs);
  if (c.operation == "genstats") {
    const auto x = p->floating(0);
    std::vector<int> axes;
    for (std::size_t i = 0; i < c.inputs[0].dimensions.size(); ++i)
      if (i != 1) axes.push_back(static_cast<int>(i));
    p->reduce_to(p->output(0), Reduce::ADD, x, axes);
    p->reduce_to(p->output(1), Reduce::ADD, p->unary(Unary::SQUARE, x), axes);
  } else {
    const auto mean = p->unary(Unary::DIV, 0, c.accum_count);
    const auto variance =
        p->unary(Unary::MAX,
                 p->binary(Binary::SUB, p->unary(Unary::DIV, 1, c.accum_count),
                           p->unary(Unary::SQUARE, mean)),
                 0.0);
    p->unary_to(p->output(2), Unary::IDENTITY, mean);
    p->unary_to(p->output(3), Unary::RSQRT,
                p->unary(Unary::ADD, variance, c.epsilon));
    p->binary_to(p->output(0), Binary::MUL, 2, p->output(3));
    p->binary_to(p->output(1), Binary::SUB, 3,
                 p->binary(Binary::MUL, mean, p->output(0)));
    if (c.inputs.size() == 6) {
      p->binary_to(p->output(4), Binary::ADD,
                   p->unary(Unary::MUL, 4, 1 - c.momentum),
                   p->unary(Unary::MUL, mean, c.momentum));
      p->binary_to(
          p->output(5), Binary::ADD, p->unary(Unary::MUL, 5, 1 - c.momentum),
          p->unary(Unary::MUL, variance,
                   c.momentum * (c.accum_count > 1
                                     ? c.accum_count / (c.accum_count - 1)
                                     : 0)));
    }
  }
  return p;
}
std::unique_ptr<TestExecutable> reference(const ResampleTestCase& c) {
  auto p = std::make_unique<NativeProgram>(c.inputs, c.outputs);
  if (c.mode == 3 || c.mode == 4) {
    const auto input =
        p->temporary(c.inputs[0].dimensions, c.inputs[0].data_type);
    const auto output =
        p->temporary(c.outputs[0].dimensions, c.outputs[0].data_type);
    p->unary_to(input, Unary::IDENTITY, 0);
    auto op = std::make_shared<md::Interpolate>();
    mv::check_mudnn(op->SetMode(c.mode == 3 ? md::Interpolate::Mode::LINEAR
                                            : md::Interpolate::Mode::NEAREST),
                    "muDNN Interpolate mode");
    std::vector<float> scale;
    for (std::size_t i = 2; i < c.inputs[0].dimensions.size(); ++i)
      scale.push_back(static_cast<float>(c.inputs[0].dimensions[i]) /
                      c.outputs[0].dimensions[i]);
    mv::check_mudnn(
        op->SetScaleInfo(static_cast<int>(scale.size()), scale.data()),
        "muDNN Interpolate scale");
    mv::check_mudnn(op->SetAlignCorners(c.align_corners),
                    "muDNN Interpolate align corners");
    p->add([r = p.get(), op, input, output] {
      mv::check_mudnn(
          op->Run(r->handle(), r->tensor(output), r->tensor(input)),
          "muDNN Interpolate::Run");
    });
    p->unary_to(p->output(0), Unary::IDENTITY, output);
  } else {
    std::size_t input = 0;
    std::vector<int> window(c.window.begin(), c.window.end()),
        stride(c.stride.begin(), c.stride.end()),
        pad(c.pre.begin(), c.pre.end()), dilation(window.size(), 1);
    if (c.pre != c.post) {
      // Explicit padding preserves the public asymmetric max-pooling shape.
      auto shape = c.inputs[0].dimensions;
      for (std::size_t i = 0; i < c.pre.size(); ++i)
        shape[i + 2] += c.pre[i] + c.post[i];
      input = p->temporary(shape, c.inputs[0].data_type);
      auto fill = std::make_shared<md::Fill>();
      mv::check_mudnn(fill->SetValue(-std::numeric_limits<double>::infinity()),
                      "muDNN pooling padding value");
      p->add([r = p.get(), fill, input] {
        mv::check_mudnn(fill->Run(r->handle(), r->tensor(input)),
                        "muDNN pooling padding");
      });
      const auto strides = contiguous(shape);
      std::size_t offset = 0;
      for (std::size_t i = 0; i < c.pre.size(); ++i)
        offset += c.pre[i] * strides[i + 2];
      offset *= io::data_type_size(c.inputs[0].data_type);
      const auto interior =
          p->view(input, c.inputs[0].dimensions, strides, offset);
      p->unary_to(interior, Unary::IDENTITY, 0);
      std::fill(pad.begin(), pad.end(), 0);
    }
    auto op = std::make_shared<md::Pooling>();
    const auto mode = c.mode == 5 ? md::Pooling::Mode::MAXPOOL
                      : c.mode == 1
                          ? md::Pooling::Mode::AVGPOOL_COUNT_WITHOUT_PAD
                          : md::Pooling::Mode::AVGPOOL_COUNT_PAD;
    mv::check_mudnn(op->SetMode(mode), "muDNN Pooling mode");
    mv::check_mudnn(
        op->SetNdInfo(static_cast<int>(window.size()), window.data(),
                      pad.data(), stride.data(), dilation.data()),
        "muDNN Pooling geometry");
    p->add([r = p.get(), op, input] {
      md::Tensor indices;
      mv::check_mudnn(op->Run(r->handle(), r->tensor(r->output(0)),
                              r->tensor(input), indices),
                      "muDNN Pooling::Run");
    });
    if (c.mode == 5) {
      // muDNN initializes MAXPOOL with the lowest finite value. The public
      // negative-infinity padding contract also includes windows containing
      // padding only; fill those geometric border regions with native Fill.
      auto fill = std::make_shared<md::Fill>();
      mv::check_mudnn(fill->SetValue(-std::numeric_limits<double>::infinity()),
                      "muDNN empty pooling window value");
      const auto fill_region = [&](std::size_t axis, std::int64_t begin,
                                   std::int64_t count) {
        if (count <= 0) return;
        auto shape = c.outputs[0].dimensions;
        shape[axis] = count;
        const auto offset = begin * c.outputs[0].strides[axis] *
                            io::data_type_size(c.outputs[0].data_type);
        const auto region =
            p->view(p->output(0), shape, c.outputs[0].strides, offset);
        p->add([r = p.get(), fill, region] {
          mv::check_mudnn(fill->Run(r->handle(), r->tensor(region)),
                          "muDNN empty pooling window Fill::Run");
        });
      };
      for (std::size_t axis = 0; axis < c.pre.size(); ++axis) {
        const auto extent = c.outputs[0].dimensions[axis + 2];
        const auto leading =
            c.pre[axis] < c.window[axis]
                ? 0
                : (c.pre[axis] - c.window[axis]) / c.stride[axis] + 1;
        const auto trailing =
            std::min(extent, (c.inputs[0].dimensions[axis + 2] + c.pre[axis] +
                              c.stride[axis] - 1) /
                                 c.stride[axis]);
        fill_region(axis + 2, 0, std::min(extent, leading));
        fill_region(axis + 2, trailing, extent - trailing);
      }
    }
  }
  return p;
}
std::unique_ptr<TestExecutable> reference(const RoPETestCase& c) {
  auto p = std::make_unique<NativeProgram>(c.inputs, c.outputs);
  // muDNN Rope uses BSHD metadata. Materialize a contiguous input and output
  // so all public strided layouts keep their external padding untouched.
  const auto& d = c.inputs[0].dimensions;
  Shape shape{d[0], d[2], d[1], d[3]};
  Shape strides{c.inputs[0].strides[0], c.inputs[0].strides[2],
                c.inputs[0].strides[1], c.inputs[0].strides[3]};
  const auto input_view = p->view(0, shape, strides);
  const auto input = p->temporary(shape, c.inputs[0].data_type);
  p->unary_to(input, Unary::IDENTITY, input_view);
  const auto dense_frequencies = p->unary(Unary::IDENTITY, 1);
  const auto width = c.inputs[1].dimensions.back();
  const auto frequencies =
      p->view(dense_frequencies, {d[2], width}, {width, 1});
  const auto output = p->temporary(shape, c.outputs[0].data_type);
  auto rotation_input = input;
  auto rotation_output = output;
  std::size_t rotated_output_view = output;
  if (width != d[3]) {
    // Public RoPE rotates the trailing channels. muDNN requires its input
    // width to equal the frequency width, so rotate a native dense slice.
    auto rotated_shape = shape;
    rotated_shape.back() = width;
    const auto offset = static_cast<std::size_t>(d[3] - width) *
                        io::data_type_size(c.inputs[0].data_type);
    const auto input_tail =
        p->view(input, rotated_shape, contiguous(shape), offset);
    rotation_input = p->temporary(rotated_shape, c.inputs[0].data_type);
    rotation_output = p->temporary(rotated_shape, c.outputs[0].data_type);
    p->unary_to(rotation_input, Unary::IDENTITY, input_tail);
    p->unary_to(output, Unary::IDENTITY, input);
    rotated_output_view =
        p->view(output, rotated_shape, contiguous(shape), offset);
  }
  auto op = std::make_shared<md::Rope>();
  mv::check_mudnn(op->SetRotaryInterleaved(false), "muDNN Rope split half");
  mv::check_mudnn(op->SetBatchFirst(true), "muDNN Rope batch first");
  p->add([r = p.get(), op, input = rotation_input, frequencies,
          output = rotation_output,
          backward = c.operation == "rope_backward"] {
    const auto status =
        backward ? op->RunBwd(r->handle(), r->tensor(output), r->tensor(input),
                              r->tensor(frequencies))
                 : op->Run(r->handle(), r->tensor(output), r->tensor(input),
                           r->tensor(frequencies));
    mv::check_mudnn(status, "muDNN Rope::Run/RunBwd");
  });
  if (width != d[3])
    p->unary_to(rotated_output_view, Unary::IDENTITY, rotation_output);
  const auto out = p->view(p->output(0), shape,
                           {c.outputs[0].strides[0], c.outputs[0].strides[2],
                            c.outputs[0].strides[1], c.outputs[0].strides[3]});
  p->unary_to(out, Unary::MUL, output, c.output_scale);
  return p;
}
}  // namespace flagdnn::testing::mthreads
