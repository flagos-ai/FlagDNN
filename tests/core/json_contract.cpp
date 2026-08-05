/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "runtime/json.hpp"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string_view>

namespace {

bool rejects(std::string_view input) {
  try {
    (void)flagdnn::native::json::parse(input);
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  using flagdnn::native::json::parse;

  const auto document = parse(
      R"({"fraction":-0.75,"exponent":1.25e2,"integer":42,"zero":-0.0})");
  if (std::abs(document.at("fraction").as_double() + 0.75) > 1.0e-12 ||
      std::abs(document.at("exponent").as_double() - 125.0) > 1.0e-12 ||
      document.at("integer").as_int() != std::int64_t{42} ||
      std::signbit(document.at("zero").as_double()) == 0) {
    std::cerr << "valid JSON numbers were parsed incorrectly\n";
    return 1;
  }

  constexpr std::string_view invalid_numbers[] = {
      "01", "-01", "1.", "1e", "1e+", "1e9999"};
  for (const auto input : invalid_numbers) {
    if (!rejects(input)) {
      std::cerr << "invalid JSON number was accepted: " << input << std::endl;
      return 1;
    }
  }
  return 0;
}
