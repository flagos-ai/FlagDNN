// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "capability.hpp"
#include "numeric_types.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>

namespace tv = flagdnn::validation::thead;

int main() {
  try {
    struct Case {
      std::string name;
    };
    const std::array<Case, 3> shared{{{"existing"}, {"new"}, {"unsupported"}}};
    const std::map<std::string, int, std::less<>> catalog{
        {"existing", 1}, {"new", 1}, {"unsupported", 0}};
    const auto selected =
        tv::select_catalog_cases(std::span<const Case>(shared), catalog);
    if (selected.size() != 3 || selected[0].name != "existing" ||
        selected[1].name != "new" || selected[2].name != "unsupported") {
      throw std::runtime_error("backend case selection changed its catalog");
    }
    const auto require_rejected = [](auto&& operation) {
      try {
        operation();
      } catch (const std::invalid_argument&) {
        return;
      }
      throw std::runtime_error("invalid case selection or dtype was accepted");
    };
    require_rejected([&] {
      (void)tv::select_catalog_cases(
          std::span<const Case>(std::array<Case, 1>{{{"missing"}}}), catalog);
    });
    require_rejected([&] {
      const std::array<Case, 2> duplicates{{{"existing"}, {"existing"}}};
      (void)tv::select_catalog_cases(std::span<const Case>(duplicates),
                                     catalog);
    });
    const std::array<float, 3> values{-1.0F, 0.0F, 1.0F};
    for (const auto type :
         {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
      const auto bytes = tv::encode_floating(type, values);
      const auto decoded = tv::decode_floating(type, bytes);
      if (!std::equal(values.begin(), values.end(), decoded.begin(),
                      decoded.end())) {
        throw std::runtime_error("existing floating I/O changed");
      }
    }
    for (const auto type : {FLAGDNN_DATA_INT32, FLAGDNN_DATA_FP8_E8M0}) {
      if (tv::element_size(type) != (type == FLAGDNN_DATA_INT32 ? 4U : 1U))
        throw std::runtime_error("raw copy dtype width is incorrect");
      require_rejected([&] { (void)tv::encode_floating(type, values); });
      require_rejected([&] { (void)tv::decode_floating(type, {}); });
    }
    std::cout << "THead catalog selection and numeric types: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
