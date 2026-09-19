/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/ascend/error.hpp"
#include "backends/ascend/extended_artifact.hpp"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

// Artifact validation must remain independent of runtime initialization.
// Link error.cpp without loading the driver; a runtime call is a test failure.
extern "C" const char *aclGetRecentErrMsg() { std::abort(); }

int main(int argc, char **argv) {
  if (argc < 5 || argc % 2 != 1)
    return 2;
  try {
    auto read = [](const char *path) {
      std::ifstream input(path);
      std::stringstream text;
      text << input.rdbuf();
      return flagdnn::native::json::parse(text.str());
    };
    const auto request = read(argv[1]);
    for (int i = 3; i < argc; i += 2) {
      const std::string expected = argv[i + 1];
      bool rejected = false;
      try {
        const auto artifact = flagdnn::ascend::parse_extended_artifact(
            request, read(argv[i]), argv[2]);
        if (artifact.stages.size() != 1)
          return 2;
        for (const auto &candidate : artifact.stages.front().candidates)
          if (!candidate.standalone_compilation)
            throw std::runtime_error(
                "extended kernels must use owned compiler workspace");
      } catch (const std::exception &error) {
        if (expected.empty() ||
            std::string(error.what()).find(expected) == std::string::npos)
          throw;
        rejected = true;
      }
      if (rejected != !expected.empty())
        throw std::runtime_error("malformed artifact was accepted");
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
