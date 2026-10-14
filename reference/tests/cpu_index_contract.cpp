/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/index.hpp"

#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace cpu = flagdnn::reference::cpu;
using Bytes = std::vector<std::byte>;
using Inputs = std::vector<Bytes>;

template <typename T> Bytes encode(const std::vector<T>& values) {
  Bytes bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}
template <typename T> T at(const Bytes& bytes, std::size_t index) {
  T value;
  std::memcpy(&value, bytes.data() + index * sizeof(T), sizeof(T));
  return value;
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <typename Exception = std::invalid_argument, typename F>
void rejected(const char* name, F run) {
  try { run(); } catch (const Exception&) { return; }
  throw std::runtime_error(std::string(name) + " was accepted");
}

void concatenate() {
  const cpu::IndexParameters p{"concatenate", {{2, 1}, {2, 2}}, {2, 3}, -1};
  const Inputs inputs{encode<std::int32_t>({16777217, -16777217}),
      encode<std::int32_t>({std::numeric_limits<std::int32_t>::min(),
          std::numeric_limits<std::int32_t>::max(), 3, -4})};
  require(cpu::evaluate_index_bytes(p, inputs, FLAGDNN_DATA_INT32) ==
      encode<std::int32_t>({16777217, std::numeric_limits<std::int32_t>::min(),
          std::numeric_limits<std::int32_t>::max(), -16777217, 3, -4}),
      "concatenate lost exact INT32 values or row order");
  const std::vector<std::pair<std::size_t, std::size_t>> expected{
      {0, 0}, {1, 0}, {1, 1}, {0, 1}, {1, 2}, {1, 3}};
  for (std::size_t i = 0; i < expected.size(); ++i)
    require(cpu::index_source(p, i) == expected[i], "legacy source map differs");

  for (const auto type : {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16,
         FLAGDNN_DATA_BFLOAT16, FLAGDNN_DATA_BOOLEAN, FLAGDNN_DATA_FP8_E4M3,
         FLAGDNN_DATA_FP8_E5M2, FLAGDNN_DATA_FP8_E8M0}) {
    const std::size_t width = type == FLAGDNN_DATA_FLOAT32 ? 4 :
        (type == FLAGDNN_DATA_FLOAT16 || type == FLAGDNN_DATA_BFLOAT16) ? 2 : 1;
    Inputs raw{Bytes(2 * width), Bytes(4 * width)};
    for (std::size_t i = 0; i < raw[0].size(); ++i)
      raw[0][i] = static_cast<std::byte>((i * 37 + 127) % 256);
    for (std::size_t i = 0; i < raw[1].size(); ++i)
      raw[1][i] = static_cast<std::byte>((i * 83 + 255) % 256);
    Bytes wanted;
    for (const auto& [input, index] : expected)
      wanted.insert(wanted.end(), raw[input].begin() + index * width,
                    raw[input].begin() + (index + 1) * width);
    require(cpu::evaluate_index_bytes(p, raw, type) == wanted,
            "concatenate changed representation bits");
  }
}

void gen_index() {
  require(cpu::evaluate_index_bytes({"gen_index", {}, {2, 3}, 0}, {},
      FLAGDNN_DATA_FLOAT32) == encode<float>({0, 0, 0, 1, 1, 1}),
      "FP32 gen_index axis0 differs");
  require(cpu::evaluate_index_bytes({"gen_index", {}, {2, 3}, -1}, {},
      FLAGDNN_DATA_INT32) == encode<std::int32_t>({0, 1, 2, 0, 1, 2}),
      "INT32 gen_index negative axis differs");
  // Above 2^24, routing an INT32 coordinate through float loses odd values.
  constexpr std::int64_t extent = 16777218;
  const auto wide = cpu::evaluate_index_bytes(
      {"gen_index", {}, {extent}, 0}, {}, FLAGDNN_DATA_INT32);
  require(at<std::int32_t>(wide, 16777216) == 16777216 &&
          at<std::int32_t>(wide, 16777217) == 16777217,
          "INT32 gen_index coordinate rounded through float");
}

void invalid() {
  for (const cpu::IndexParameters& p : {
      cpu::IndexParameters{"unknown", {{1}}, {1}, 0},
      {"gen_index", {}, {}, 0}, {"gen_index", {}, {2, 3}, -3},
      {"gen_index", {}, {2, 3}, 2}, {"gen_index", {}, {0}, 0},
      {"gen_index", {}, {-1}, 0}, {"gen_index", {{1}}, {1}, 0},
      {"concatenate", {}, {1}, 0},
      {"concatenate", {{1}}, {1, 1}, 0},
      {"concatenate", {{1, 3}}, {1, 2}, 0},
      {"concatenate", {{2}, {3}}, {4}, 0}})
    rejected("invalid geometry", [&] { (void)cpu::index_source(p, 0); });
  rejected<std::out_of_range>("output index outside shape", [] {
    (void)cpu::index_source({"gen_index", {}, {2, 3}, -1}, 6);
  });
  constexpr auto largest = std::numeric_limits<std::int64_t>::max();
  rejected<std::overflow_error>("element count overflow", [] {
    (void)cpu::index_source({"gen_index", {}, {largest, 3}, 0}, 0);
  });
  rejected<std::overflow_error>("axis sum overflow", [] {
    (void)cpu::index_source({"concatenate", {{largest}, {largest}, {largest}},
                            {largest}, 0}, 0);
  });
  rejected<std::overflow_error>("byte count overflow", [] {
    (void)cpu::evaluate_index_bytes({"gen_index", {}, {largest}, 0}, {},
                                    FLAGDNN_DATA_FLOAT32);
  });
  rejected<std::overflow_error>("INT32 coordinate overflow", [] {
    (void)cpu::evaluate_index_bytes({"gen_index", {}, {2147483649LL}, 0}, {},
                                    FLAGDNN_DATA_INT32);
  });
  rejected("gen_index unsupported dtype", [] {
    (void)cpu::evaluate_index_bytes({"gen_index", {}, {1}, 0}, {},
                                    FLAGDNN_DATA_FLOAT16);
  });
  rejected("missing input", [] {
    (void)cpu::evaluate_index_bytes({"concatenate", {{1}}, {1}, 0}, {},
                                    FLAGDNN_DATA_INT32);
  });
  rejected("input byte size mismatch", [] {
    (void)cpu::evaluate_index_bytes({"concatenate", {{1}}, {1}, 0},
                                    Inputs{Bytes(1)}, FLAGDNN_DATA_INT32);
  });
}

int main() {
  try {
    concatenate(); gen_index(); invalid();
    std::cout << "PASS CPU index contracts: bit-exact concatenate, typed gen_index and invalid geometry\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
