/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_plan.hpp"
#include "validation/functional/raw_io.hpp"
#include "reference/cpu/pointwise.hpp"
#include <array>
#include <cmath>
#include <limits>

namespace {
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

void check_fp8_storage() {
  using namespace flagdnn::testing;
  namespace io = ascend::io;
  for (const auto type : {FLAGDNN_DATA_FP8_E4M3, FLAGDNN_DATA_FP8_E5M2,
                          FLAGDNN_DATA_FP8_E8M0}) {
    const TestTensor spec{1, type, {16, 16}, {35, 2}};
    require(io::data_type_size(type) == 1, "FP8 storage must be one byte");
    auto bytes = ascend::storage_bytes(spec, [](std::size_t i) { return i; });
    ascend::check_raw_padding(bytes, spec);
    const auto logical = ascend::logical_bytes(bytes, spec);
    for (std::size_t i = 0; i < logical.size(); ++i)
      require(logical[i] == i, "raw FP8 payload changed");
    bytes[1] ^= 1;
    bool rejected = false;
    try {
      ascend::check_raw_padding(bytes, spec);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    require(rejected, "FP8 padding corruption was not rejected");

    for (unsigned value = 0; value < 256; ++value) {
      const std::array<std::uint8_t, 1> original{
          static_cast<std::uint8_t>(value)};
      const auto decoded = io::decode(original, type, 1);
      if (std::isfinite(decoded[0]))
        require(io::encode(decoded, type)[0] == original[0],
                "finite FP8 codec roundtrip changed");
    }

    ascend::Plan copy({spec}, [](ascend::Plan &) {}, true);
    require(copy.uses_byte_storage_view(), "FP8 copy view marker missing");
    auto *tensor = copy.tensor(spec, nullptr);
    aclDataType actual;
    require(aclGetDataType(tensor, &actual) == 0 && actual == ACL_UINT8,
            "FP8 copy descriptor must expose UINT8 storage");
    ascend::Plan arithmetic({spec}, [](ascend::Plan &) {});
    rejected = false;
    try {
      (void)arithmetic.tensor(spec, nullptr);
    } catch (const ascend::Unsupported &) {
      rejected = true;
    }
    require(rejected && !arithmetic.uses_byte_storage_view(),
            "byte-copy support enabled FP8 arithmetic");
  }
  const std::array<float, 6> powers{1, 2, 4, 8, 16, 32};
  require(io::encode(powers, FLAGDNN_DATA_FP8_E4M3) ==
              std::vector<std::uint8_t>{56, 64, 72, 80, 88, 96},
          "identity E4M3 reference inputs changed");
  require(io::encode(powers, FLAGDNN_DATA_FP8_E5M2) ==
              std::vector<std::uint8_t>{60, 64, 68, 72, 76, 80},
          "identity E5M2 reference inputs changed");
}

void check_integer_reference_gather() {
  using namespace flagdnn::testing;
  constexpr auto minimum = std::numeric_limits<std::int32_t>::min();
  constexpr auto maximum = std::numeric_limits<std::int32_t>::max();
  const std::vector<std::int32_t> source{minimum,16777217,-7,maximum,7,0};
  const TestTensor spec{2, FLAGDNN_DATA_INT32, {2,3}, {13,2}, 64};
  auto storage = ascend::integer_bytes(spec, [&](std::size_t i) { return source[i]; });
  ascend::check_raw_padding(storage, spec);
  const auto gathered = ascend::logical_values<std::int32_t>(storage, spec);
  require(gathered == source, "INT32 gapped gather lost full-range values");
  require(ascend::logical_bytes(storage, spec) ==
              ascend::value_bytes<std::int32_t>(source),
          "INT32 gapped gather changed encoded values");
  storage.pop_back();
  bool rejected = false;
  try {
    (void)ascend::logical_values<std::int32_t>(storage, spec);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "INT32 gather accepted truncated storage");
}
} // namespace

int main() {
  using namespace flagdnn::testing;
  check_fp8_storage();
  check_integer_reference_gather();
  // Descriptors do not require device allocation. Repeated ownership transfer
  // catches double destruction when one tensor participates in several lists.
  for (int repeat = 0; repeat < 100; ++repeat) {
    ascend::Plan plan({}, [](ascend::Plan &) {});
    const TestTensor spec{1, FLAGDNN_DATA_FLOAT32, {2, 3}, {3, 1}};
    auto *tensor = plan.tensor(spec, nullptr);
    plan.list({tensor});
    plan.list({tensor, tensor});
  }
}
