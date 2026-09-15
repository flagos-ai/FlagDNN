/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "graph/lowering/lowering.hpp"
#include "graph/lowering/helpers.hpp"
#include <cmath>
#include <limits>
namespace flagdnn::native {
LoweredOperation lower_rng(const OperationSpec& operation) {
  require_port_count(operation, 0, 1);
  const auto& output = require_port(operation.outputs, "output", "output");
  require_non_overlapping_tensor(output, "RNG output");
  require_floating_data_type(output, "RNG output must use FP32, FP16 or BF16");
  if (output.dimensions.empty() || output.dimensions.size() > 8)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "RNG output rank must be 1..8");
  const auto distribution = integer_attribute(operation, "distribution");
  const auto seed = integer_attribute(operation, "seed"),
             offset = integer_attribute(operation, "offset");
  const auto probability = real_attribute(operation, "probability");
  if (distribution < 1 || distribution > 3 || !std::isfinite(probability) ||
      probability < 0 || probability > 1)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "RNG distribution or probability is invalid");
  if (offset < 0 || offset > std::numeric_limits<std::int64_t>::max() -
                                 output.element_count())
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "RNG counter range is invalid");
  return {{{"distribution", distribution},
           {"seed", seed},
           {"offset", offset},
           {"n_elements", output.element_count()}},
          {{"probability", probability}},
          {}};
}
}  // namespace flagdnn::native
