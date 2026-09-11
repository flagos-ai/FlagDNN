// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "acdnn_fp8_codec.hpp"
#include "acdnn_reference.hpp"
#include "backend_pointwise_reference.hpp"
#include "tensor_io.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tv = flagdnn::validation::thead;
using flagdnn::testing::TestTensor;

namespace {
void roundtrip(flagdnnDataType_t dtype, tv::DeviceStream &stream) {
  // Enumerate finite storage codes. The identity property exercises both GPU
  // conversions without introducing a CPU implementation of FP8 arithmetic.
  std::vector<std::uint8_t> codes;
  for (int code = 0; code < 256; ++code) {
    if ((dtype == FLAGDNN_DATA_FP8_E4M3 && (code & 127) == 127) ||
        (dtype == FLAGDNN_DATA_FP8_E5M2 && (code & 127) >= 124)) continue;
    codes.push_back(static_cast<std::uint8_t>(code));
  }
  const auto count = static_cast<std::int64_t>(codes.size());
  TestTensor input{1, dtype, {1, 1, count}, {count, count, 1}};
  TestTensor decoded = input, encoded = input;
  decoded.uid = 2;
  decoded.data_type = FLAGDNN_DATA_FLOAT32;
  encoded.uid = 3;
  auto decode = tv::make_acdnn_fp8_codec(input, decoded);
  auto encode = tv::make_acdnn_fp8_codec(decoded, encoded);
  tv::DeviceBuffer bytes(codes.size()), values(codes.size() * sizeof(float));
  tv::DeviceBuffer result(codes.size()), dw(decode->workspace_size());
  tv::DeviceBuffer ew(encode->workspace_size());
  tv::copy_to_device_async<std::uint8_t>(bytes, codes, 0, stream.get());
  const std::array<flagdnnBinding_t, 2> db{{{1, bytes.data()}, {2, values.data()}}};
  const std::array<flagdnnBinding_t, 2> eb{{{2, values.data()}, {3, result.data()}}};
  decode->execute(db, dw.data(), dw.size(), stream.opaque());
  encode->execute(eb, ew.data(), ew.size(), stream.opaque());
  std::vector<std::uint8_t> actual(codes.size());
  tv::copy_from_device_async<std::uint8_t>(actual, result, 0, stream.get());
  tv::check_driver(cuStreamSynchronize(stream.get()), "FP8 roundtrip synchronize");
  if (actual != codes) throw std::runtime_error("acDNN FP8 finite roundtrip failed");
  std::cout << "PASS acDNN FP8 finite roundtrip: dtype=" << dtype
            << " codes=" << codes.size() << '\n';
}

void rounding(flagdnnDataType_t dtype, tv::DeviceStream &stream) {
  // Fixed format boundary fixtures, including signed zero, ties to even,
  // subnormal rounding and saturation. Every conversion executes in acDNN.
  const bool e4 = dtype == FLAGDNN_DATA_FP8_E4M3;
  const std::array<float, 12> inputs = e4
      ? std::array<float, 12>{0.0F, -0.0F, 1.0625F, 1.1875F, -1.0625F,
                             -1.1875F, 0x1p-10F, 0x1.8p-9F, 448.0F,
                             1024.0F, -1024.0F, 0x1.cp-7F}
      : std::array<float, 12>{0.0F, -0.0F, 1.125F, 1.375F, -1.125F,
                             -1.375F, 0x1p-17F, 0x1.8p-16F, 57344.0F,
                             65536.0F, -65536.0F, 0x1.cp-15F};
  const std::array<std::uint8_t, 12> expected = e4
      ? std::array<std::uint8_t, 12>{0, 128, 56, 58, 184, 186, 0, 2, 126, 126, 254, 7}
      : std::array<std::uint8_t, 12>{0, 128, 60, 62, 188, 190, 0, 2, 123, 123, 251, 4};
  TestTensor input{1, FLAGDNN_DATA_FLOAT32, {1, 1, 12}, {12, 12, 1}};
  TestTensor output{2, dtype, {1, 1, 12}, {12, 12, 1}};
  auto codec = tv::make_acdnn_fp8_codec(input, output);
  tv::DeviceBuffer values(sizeof(inputs)), bytes(expected.size());
  tv::DeviceBuffer workspace(codec->workspace_size());
  tv::copy_to_device_async<float>(values, inputs, 0, stream.get());
  const std::array<flagdnnBinding_t, 2> bindings{{{1, values.data()}, {2, bytes.data()}}};
  codec->execute(bindings, workspace.data(), workspace.size(), stream.opaque());
  std::array<std::uint8_t, 12> actual{};
  tv::copy_from_device_async<std::uint8_t>(actual, bytes, 0, stream.get());
  tv::check_driver(cuStreamSynchronize(stream.get()), "FP8 rounding synchronize");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (actual[index] != expected[index]) {
      throw std::runtime_error("acDNN FP8 rounding fixture " + std::to_string(index) +
          " expected=" + std::to_string(expected[index]) +
          " actual=" + std::to_string(actual[index]));
    }
  }
}

void invalid_storage_views() {
  tv::BackendPointwiseReferenceSpecification valid;
  valid.inputs = {{1, FLAGDNN_DATA_FP8_E4M3, {1, 1, 4}, {4, 4, 1}}};
  valid.output = {2, FLAGDNN_DATA_FLOAT32, {1, 1, 4}, {4, 4, 1}};
  valid.primitive = "acDNN FP8 codec contract";
  valid.fp8_storage_bytes = true;
  for (int index = 0; index < 6; ++index) {
    auto invalid = valid;
    switch (index) {
      case 0: invalid.fp8_storage_bytes = false; break;
      case 1: invalid.mode = ACDNN_POINTWISE_ABS; break;
      case 2: invalid.alpha1 = 2.0F; break;
      case 3: invalid.inputs[0].data_type = FLAGDNN_DATA_FLOAT32; break;
      case 4: invalid.output.data_type = FLAGDNN_DATA_FP8_E5M2; break;
      case 5: invalid.constant_one_numerator = true; break;
    }
    try {
      (void)tv::make_acdnn_backend_pointwise_reference(invalid);
    } catch (const std::invalid_argument &) {
      continue;
    }
    throw std::runtime_error("invalid FP8 storage view was accepted");
  }
}
}  // namespace

int main() {
  try {
    tv::check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
    tv::DeviceStream stream;
    invalid_storage_views();
    for (auto dtype : {FLAGDNN_DATA_FP8_E4M3, FLAGDNN_DATA_FP8_E5M2}) {
      roundtrip(dtype, stream);
      rounding(dtype, stream);
    }
    std::cout << "PASS acDNN FP8 codec contract\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "acDNN FP8 codec contract: FAIL " << error.what() << '\n';
    return 1;
  }
}
