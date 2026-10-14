/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/pointwise.hpp"
#include "reference/cpu/layout.hpp"
#include "reference/cpu/index.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace cpu = flagdnn::reference::cpu;
using Bytes = std::vector<std::uint8_t>;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template <class F> void rejects(F f) {
  try { f(); } catch (const std::invalid_argument&) { return; }
  throw std::runtime_error("invalid storage reference accepted");
}
int main() {
  try {
    Bytes payloads(256);
    std::iota(payloads.begin(), payloads.end(), 0);
    auto copied = cpu::evaluate_identity_bytes(payloads);
    require(copied == payloads, "identity changed FP8 payloads");
    copied[0] = 255;
    require(payloads[0] == 0, "identity output aliases input");
    require(cpu::evaluate_identity_bytes({}).empty(), "empty identity differs");

    cpu::LayoutParameters p;
    p.operation = "reshape";
    p.input_dimensions = {16, 16};
    p.output_dimensions = {4, 64};
    require(cpu::evaluate_layout_bytes(p, payloads) == payloads,
            "reshape changed FP8 payloads");
    p.operation = "transpose";
    p.input_dimensions = {2, 128};
    p.output_dimensions = {128, 2};
    p.permutation = {1, 0};
    Bytes expected;
    for (int i = 0; i < 128; ++i) {
      expected.push_back(i);
      expected.push_back(i + 128);
    }
    require(cpu::evaluate_layout_bytes(p, payloads) == expected,
            "transpose changed source/payload");
    std::swap(p.input_dimensions, p.output_dimensions);
    require(cpu::evaluate_layout_bytes(p, expected) == payloads,
            "inverse transpose changed payloads");

    p = {"slice", {16, 16}, {8, 5}, {}, {{1, 16}, {0, 15}}, {2, 3}};
    expected.clear();
    for (int row = 1; row < 16; row += 2)
      for (int col = 0; col < 15; col += 3)
        expected.push_back(row * 16 + col);
    require(cpu::evaluate_layout_bytes(p, payloads) == expected,
            "slice strides selected incorrect payloads");
    p.slice_strides[0] = 0;
    rejects([&] { (void)cpu::evaluate_layout_bytes(p, payloads); });
    p = {"transpose", {2, 2}, {2, 2}, {0, 0}, {}, {}};
    rejects([&] { (void)cpu::evaluate_layout_bytes(p, Bytes(4)); });
    p = {"reshape", {2, 2}, {2, 3}, {}, {}, {}};
    rejects([&] { (void)cpu::evaluate_layout_bytes(p, Bytes(4)); });
    rejects([&] { (void)cpu::evaluate_layout_bytes(p, Bytes(3)); });
    rejects([&] { (void)cpu::evaluate_layout_bytes(p, Bytes(4), 0); });

    // Multi-byte elements retain INT32 bits beyond the FP32 exact range.
    const Bytes words{0x01,0,0,0x01, 0,0,0,0x80, 0xff,0xff,0xff,0x7f,
                       0xff,0xff,0xff,0xff};
    p = {"transpose", {2,2}, {2,2}, {1,0}, {}, {}};
    const Bytes transposed{0x01,0,0,0x01, 0xff,0xff,0xff,0x7f,
                            0,0,0,0x80, 0xff,0xff,0xff,0xff};
    require(cpu::evaluate_layout_bytes(p, words, 4) == transposed,
            "INT32 transpose changed bytes");

    cpu::IndexParameters concat{"concatenate", {{2,64}, {2,64}}, {2,128}, -1};
    const Bytes first(payloads.begin(), payloads.begin() + 128);
    const Bytes second(payloads.begin() + 128, payloads.end());
    expected.clear();
    for (int row = 0; row < 2; ++row) {
      expected.insert(expected.end(), first.begin() + row*64, first.begin() + (row+1)*64);
      expected.insert(expected.end(), second.begin() + row*64, second.begin() + (row+1)*64);
    }
    require(cpu::evaluate_concatenate_bytes(concat, {first, second}) == expected,
            "concatenate negative axis changed payloads");
    require(cpu::evaluate_concatenate_bytes(
        {"concatenate", {{2,1}, {2,1}}, {2,2}, 1}, {{0,1}, {1,0}}) == Bytes({0,1,1,0}),
        "Boolean concatenate differs");
    require(cpu::evaluate_concatenate_bytes(
        {"concatenate", {{4}}, {4}, 0}, {words}, 4) == words,
        "single-input INT32 concatenate differs");
    require(cpu::evaluate_concatenate_bytes(
        {"concatenate", {{2,1},{2,1}}, {2,2}, -1},
        {Bytes(words.begin(), words.begin()+8), Bytes(words.begin()+8, words.end())},
        4) == transposed, "multi-input INT32 concatenate changed element width");
    concat.axis = 2;
    rejects([&] { (void)cpu::evaluate_concatenate_bytes(concat, {first, second}); });
    concat.axis = 1;
    concat.output_shape[1] = 127;
    rejects([&] { (void)cpu::evaluate_concatenate_bytes(concat, {first, second}); });
    concat.output_shape = {3,128};
    rejects([&] { (void)cpu::evaluate_concatenate_bytes(concat, {first, second}); });
    std::cout << "PASS CPU raw storage contracts\n";
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
