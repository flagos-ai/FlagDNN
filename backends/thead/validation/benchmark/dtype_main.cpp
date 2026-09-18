// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <exception>
#include <iostream>
#include <string_view>

#include "common/dtype_runner.hpp"
int main(int argc, char** argv) {
  try {
    if (argc == 4 && (std::string_view(argv[1]) == "--dump-cases" ||
                      std::string_view(argv[1]) == "--probe-reference"))
      return flagdnn::testing::run_native_dtype_benchmark(2, argv, argv[2],
                                                          argv[3]);
    if (argc != 5) return 2;
    return flagdnn::testing::run_native_dtype_benchmark(argc - 2, argv, argv[3],
                                                        argv[4]);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
