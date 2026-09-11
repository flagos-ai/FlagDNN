// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "tensor_io.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>

namespace tv = flagdnn::validation::thead;
using flagdnn::testing::TestTensor;

void probe(acdnnPointwiseMode_t mode, std::string_view name,
           bool expect_not_supported) {
  const bool is_mod = mode == ACDNN_POINTWISE_MOD;
  const bool unary = mode == ACDNN_POINTWISE_LOGICAL_NOT;
  const auto dtype = is_mod ? FLAGDNN_DATA_FLOAT32 : FLAGDNN_DATA_BOOLEAN;
  TestTensor x{1, dtype, {1, 1, 1, 16}, {16, 16, 16, 1}};
  TestTensor y = x;
  y.uid = 2;
  TestTensor z = x;
  z.uid = 3;
  tv::BackendPointwiseReferenceSpecification specification;
  specification.mode = mode;
  specification.inputs = {x};
  if (!unary) specification.inputs.push_back(y);
  specification.output = z;
  specification.primitive = std::string(name);
  try {
    auto reference = tv::make_acdnn_backend_pointwise_reference(specification);
    tv::DeviceStream stream;
    tv::DeviceBuffer a(64), b(64), output(64);
    tv::DeviceBuffer workspace(reference->workspace_size());
    std::array<std::uint8_t, 16> left{}, right{}, actual{}, expected{};
    std::array<float, 16> mod_left{}, mod_right{};
    for (std::size_t i = 0; i < left.size(); ++i) {
      left[i] = (i % 4) / 2;
      right[i] = i % 2;
      expected[i] = mode == ACDNN_POINTWISE_LOGICAL_AND
                        ? std::uint8_t(left[i] && right[i])
                        : mode == ACDNN_POINTWISE_LOGICAL_OR
                              ? std::uint8_t(left[i] || right[i])
                              : std::uint8_t(!left[i]);
      mod_left[i] = 5.5F;
      mod_right[i] = 2.0F;
    }
    if (is_mod) {
      tv::copy_to_device_async<float>(a, mod_left, 0, stream.get());
      tv::copy_to_device_async<float>(b, mod_right, 0, stream.get());
    } else {
      tv::copy_to_device_async<std::uint8_t>(a, left, 0, stream.get());
      tv::copy_to_device_async<std::uint8_t>(b, right, 0, stream.get());
    }
    const std::array<flagdnnBinding_t, 3> bindings{{
        {1, a.data()}, {2, b.data()}, {3, output.data()}}};
    reference->prepare(bindings, stream.opaque());
    reference->execute(bindings, workspace.data(), workspace.size(), stream.opaque());
    tv::copy_from_device_async<std::uint8_t>(actual, output, 0, stream.get());
    tv::check_driver(cuStreamSynchronize(stream.get()), "gap probe synchronize");
    if (expect_not_supported || actual == expected) {
      throw std::runtime_error(std::string(name) +
          " native primitive changed: requalify before replacing the acDNN composite reference");
    }
    std::cout << name << ": confirmed acdnn_semantic_mismatch\n";
  } catch (const tv::AcdnnStatusError& error) {
    const bool rejected_mode =
        error.status() == ACDNN_STATUS_NOT_SUPPORTED ||
        (error.status() == ACDNN_STATUS_BAD_PARAM &&
         std::string_view(error.what()).starts_with(
             "acdnnBackendFinalize(pointwise operation)"));
    if (!expect_not_supported || !rejected_mode) {
      throw;
    }
    std::cout << name << ": confirmed descriptor rejection ("
              << error.what() << ")\n";
  }
}

int main() {
  try {
    tv::check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
    probe(ACDNN_POINTWISE_MOD, "mod", true);
    probe(ACDNN_POINTWISE_LOGICAL_NOT, "logical_not", true);
    probe(ACDNN_POINTWISE_LOGICAL_AND, "logical_and", false);
    probe(ACDNN_POINTWISE_LOGICAL_OR, "logical_or", false);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "acDNN gap requalification: FAIL " << error.what() << '\n';
    return 1;
  }
}
