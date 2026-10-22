/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/case.hpp"
#include "validation_guards.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
namespace bench = flagdnn::benchmarking;
namespace io = flagdnn::validation::hygon::tensor_io;

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

template <class Action> void rejects(Action action, std::string_view message) {
  bool failed = false;
  try {
    action();
  } catch (const std::exception &) {
    failed = true;
  }
  require(failed, message);
}

void filter_contract() {
  for (const std::string operation : {"logical_and", "logical_or"}) {
    const std::string first = operation + "_bool_1x1x16";
    const std::string second = operation + "_bool_1x1x32";
    const std::vector<std::string_view> cases = {first, second};
    for (const std::string_view filter :
         {std::string_view{}, std::string_view{first},
          std::string_view{"missing"}}) {
      std::ostringstream output, error;
      const int status = bench::skip_unreferenced_benchmark_cases(
          3, operation, cases, filter, "no matching NVIDIA reference", output,
          error);
      if (filter == "missing") {
        require(status == 1 && output.str().empty() &&
                    error.str().find("_BENCHMARK_FAILED:") !=
                        std::string::npos &&
                    error.str().find("did not match") != std::string::npos,
                "unmatched native filter must fail normally");
      } else {
        const std::string counts = filter.empty()
                                       ? "SKIP cases=2 executed=0 skipped=2"
                                       : "SKIP cases=1 executed=0 skipped=1";
        require(status == 77 && error.str().empty() &&
                    output.str().find(counts) != std::string::npos,
                "native skip accounting differs");
        require(filter.empty() ||
                    output.str().find(second) == std::string::npos,
                "filtered native skip included an unselected case");
      }
    }
    std::ostringstream output, error;
    require(bench::skip_unreferenced_benchmark_cases(
                2, operation, cases, {}, "unused", output, error) == 2,
            "invalid argc must preserve usage failure");
  }
}

void execution_status_contract() {
  // Keep the raw 0.8-baseline evaluation, but do not turn it into an execution
  // failure when all measurement and correctness work completed successfully.
  for (const bool baseline_passed : {false, true}) {
    std::ostringstream output;
    output << "{\"equivalent_baseline\":{\"passed\":"
           << (baseline_passed ? "true" : "false") << "}}\n";
    require(bench::finish_benchmark_execution("FLAGDNN_DIV_BENCHMARK", 45, 45,
                                              0, output) == 0,
            "reported performance evaluation changed execution exit status");
    require(output.str().find("PASS cases=45 executed=45 skipped=0") !=
                    std::string::npos &&
                output.str().find("FAIL") == std::string::npos,
            "completed measurements must retain PASS accounting");
    require(output.str().find(baseline_passed
                                  ? "\"passed\":true"
                                  : "\"passed\":false") != std::string::npos,
            "raw performance evaluation was not preserved");
  }
  std::ostringstream mixed, skipped;
  require(bench::finish_benchmark_execution("MIXED", 7, 5, 2, mixed) == 0 &&
              mixed.str().find("PASS cases=7 executed=5 skipped=2") !=
                  std::string::npos,
          "mixed executed/skipped accounting differs");
  require(bench::finish_benchmark_execution("SKIPPED", 2, 0, 2, skipped) ==
                  77 &&
              skipped.str().find("SKIP cases=2 executed=0 skipped=2") !=
                  std::string::npos,
          "all-skipped suite lost its skip exit code");
  for (const auto counts : {std::array<std::size_t, 3>{0, 0, 0},
                            std::array<std::size_t, 3>{1, 2, 0},
                            std::array<std::size_t, 3>{2, 1, 0}}) {
    std::ostringstream output;
    bool failed = false;
    try {
      (void)bench::finish_benchmark_execution("INVALID", counts[0], counts[1],
                                              counts[2], output);
    } catch (const std::invalid_argument &) {
      failed = true;
    }
    require(failed && output.str().empty(),
            "invalid execution accounting must remain a hard error");
  }
}

void bit_exact_identity_contract() {
  const auto fixture = bench::fp8_identity_fixture(36);
  for (std::size_t index = 0; index < fixture.size(); ++index)
    require(fixture[index] == static_cast<float>(1U << (index % 6)),
            "FP8 identity fixture differs from NVIDIA's six-value input");
  for (const auto dtype : {FLAGDNN_DATA_FP8_E4M3, FLAGDNN_DATA_FP8_E5M2}) {
    require(io::decode(io::encode(fixture, dtype), dtype, fixture.size()) ==
                fixture,
            "NVIDIA identity fixture is not exact in the target codec");
    const bench::TensorSpec input = {1, dtype, {2, 3}, {8, 2}};
    const bench::TensorSpec output = {2, dtype, {2, 3}, {7, 2}};
    const auto logical = bench::fp8_identity_fixture(6);
    const auto original = io::encode(io::scatter(logical, input), dtype);
    const auto expected = io::encode(io::scatter(logical, output), dtype);
    const auto check = [&](const auto &actual_input,
                           const auto &actual_output) {
      bench::require_bit_exact_identity("mock FP8 identity", input, output,
                                        original, actual_input, actual_output);
    };
    check(original, expected);
    require(io::decode(expected, dtype, expected.size())[1] !=
                io::kPaddingSentinel,
            "FP8 padding contract no longer exercises encoded-only semantics");
    for (std::size_t index = 0; index < logical.size(); ++index) {
      auto changed = expected;
      changed[io::logical_offset(index, output)] ^= 1U;
      rejects([&] { check(original, changed); },
              "identity accepted a changed logical output byte");
    }
    auto changed_padding = expected;
    changed_padding[1] ^= 1U;
    rejects([&] { check(original, changed_padding); },
            "identity accepted changed output padding");
    for (const std::size_t offset : {std::size_t{0}, std::size_t{1}}) {
      auto changed_input = original;
      changed_input[offset] ^= 1U;
      rejects([&] { check(changed_input, expected); },
              "identity accepted changed input data or input padding");
    }
    auto incomplete = io::encode(
        io::scatter(bench::capture_output_poison(logical, dtype, 0, 0), output),
        dtype);
    rejects([&] { check(original, incomplete); },
            "bit-exact identity accepted an empty writer");
    for (std::size_t index = 0; index + 1 < logical.size(); ++index)
      incomplete[io::logical_offset(index, output)] =
          original[io::logical_offset(index, input)];
    rejects([&] { check(original, incomplete); },
            "bit-exact identity accepted a partial writer");

    // A floating comparison accepts a sign-bit-only zero change. The raw-byte
    // helper must reject it even when both decodes yield numerically equal 0.
    const bench::TensorSpec zero = {3, dtype, {1}, {1}};
    const auto positive_zero = io::encode(std::vector<float>{0.0F}, dtype);
    const auto negative_zero = io::encode(std::vector<float>{-0.0F}, dtype);
    require(positive_zero != negative_zero &&
                io::decode(positive_zero, dtype, 1) ==
                    io::decode(negative_zero, dtype, 1),
            "signed-zero byte contract does not distinguish raw encoding");
    bench::require_bit_exact_identity("zero", zero, zero, positive_zero,
                                      positive_zero, positive_zero);
    rejects(
        [&] {
          bench::require_bit_exact_identity("zero", zero, zero, positive_zero,
                                            positive_zero, negative_zero);
        },
        "bit-exact identity compared decoded numbers instead of bytes");
    rejects(
        [&] {
          bench::require_bit_exact_identity("zero", zero, zero, positive_zero,
                                            negative_zero, negative_zero);
        },
        "matching corrupted input/output replaced the original byte oracle");
    auto truncated = expected;
    truncated.pop_back();
    rejects([&] { check(original, truncated); },
            "bit-exact identity accepted truncated output bytes");
    auto wrong_type = output;
    wrong_type.data_type = FLAGDNN_DATA_FLOAT16;
    rejects(
        [&] {
          bench::require_bit_exact_identity("wrong type", input, wrong_type,
                                            original, original, expected);
        },
        "bit-exact identity accepted mismatched dtypes");
    auto wrong_count = output;
    wrong_count.dimensions = {1, 3};
    rejects(
        [&] {
          bench::require_bit_exact_identity("wrong count", input, wrong_count,
                                            original, original, expected);
        },
        "bit-exact identity accepted mismatched logical sizes");
  }
}

void supplemental_bit_exact_identity_contract() {
  for (const auto dtype : {FLAGDNN_DATA_BFLOAT16, FLAGDNN_DATA_INT32,
                           FLAGDNN_DATA_BOOLEAN}) {
    // These spans begin at the bound tensor entrance, as in runner.cpp. Input
    // and output have different noncompact physical layouts and byte offsets.
    const bench::TensorSpec input = {11, dtype, {2, 3}, {9, 2}, 16};
    const bench::TensorSpec output = {12, dtype, {2, 3}, {8, 2}, 32};
    const std::vector<float> logical =
        dtype == FLAGDNN_DATA_BOOLEAN ? std::vector<float>{1, 0, 1, 0, 0, 1}
                                     : std::vector<float>{0, -0.0F, 1, -1,
                                                          1.5F, -97};
    const auto width = io::data_type_size(dtype);
    auto original = io::encode(io::scatter(logical, input), dtype);
    if (dtype == FLAGDNN_DATA_INT32) {
      // Preserve integers that a float-based comparison cannot distinguish.
      // This tests the byte helper, without changing the NVIDIA input fixture.
      const std::array<std::int32_t, 6> values = {
          16777216, 16777217, std::numeric_limits<std::int32_t>::max(),
          std::numeric_limits<std::int32_t>::min(), -16777217, 0};
      for (std::size_t index = 0; index < values.size(); ++index)
        std::memcpy(original.data() + io::logical_offset(index, input) * width,
                    &values[index], width);
    }
    auto expected = io::encode(io::scatter(logical, output), dtype);
    const auto copy_element = [&](auto &bytes, std::size_t index) {
      std::copy_n(original.begin() + io::logical_offset(index, input) * width,
                  width,
                  bytes.begin() + io::logical_offset(index, output) * width);
    };
    for (std::size_t index = 0; index < logical.size(); ++index)
      copy_element(expected, index);
    const auto check = [&](const auto &actual_input, const auto &actual_output) {
      bench::require_bit_exact_identity("supplemental identity", input, output,
                                        original, actual_input, actual_output);
    };
    check(original, expected);
    for (std::size_t index = 0; index < logical.size(); ++index) {
      for (std::size_t byte = 0; byte < width; ++byte) {
        auto changed_output = expected;
        auto changed_input = original;
        changed_output[io::logical_offset(index, output) * width + byte] ^= 1U;
        changed_input[io::logical_offset(index, input) * width + byte] ^= 1U;
        rejects([&] { check(original, changed_output); },
                "supplemental identity accepted a changed output byte");
        rejects([&] { check(changed_input, expected); },
                "supplemental identity accepted a changed input byte");
        rejects([&] { check(changed_input, changed_output); },
                "matching corruption replaced the original byte oracle");
      }
    }
    for (const bool checking_input : {false, true}) {
      const auto &tensor = checking_input ? input : output;
      std::vector<bool> occupied(io::storage_element_count(tensor), false);
      for (std::size_t index = 0; index < logical.size(); ++index)
        occupied[io::logical_offset(index, tensor)] = true;
      for (std::size_t element = 0; element < occupied.size(); ++element) {
        if (occupied[element])
          continue;
        for (std::size_t byte = 0; byte < width; ++byte) {
          auto changed = checking_input ? original : expected;
          changed[element * width + byte] ^= 1U;
          rejects(
              [&] {
                check(checking_input ? changed : original,
                      checking_input ? expected : changed);
              },
              "supplemental identity accepted modified padding");
        }
      }
    }
    auto incomplete = expected;
    for (std::size_t index = 0; index < logical.size(); ++index)
      for (std::size_t byte = 0; byte < width; ++byte)
        incomplete[io::logical_offset(index, output) * width + byte] ^= 0xFFU;
    for (std::size_t written = 0; written < logical.size(); ++written) {
      rejects([&] { check(original, incomplete); },
              "supplemental identity accepted empty or partial writes");
      copy_element(incomplete, written);
    }
    check(original, incomplete);

    // All three codecs can decode distinct representations to equal floats:
    // BF16 +/-0, INT32 adjacent values above 2^24, and BOOL true bytes 1/2.
    auto numerically_equal = expected;
    const auto first = io::logical_offset(0, output) * width;
    if (dtype == FLAGDNN_DATA_BFLOAT16) {
      const auto negative_zero = io::encode(std::vector<float>{-0.0F}, dtype);
      std::copy(negative_zero.begin(), negative_zero.end(),
                numerically_equal.begin() + first);
    } else if (dtype == FLAGDNN_DATA_INT32) {
      const std::int32_t adjacent = 16777217;
      std::memcpy(numerically_equal.data() + first, &adjacent, width);
    } else {
      numerically_equal[first] = 2U;
    }
    require(numerically_equal != expected &&
                io::decode(numerically_equal, dtype,
                           io::storage_element_count(output)) ==
                    io::decode(expected, dtype,
                               io::storage_element_count(output)),
            "supplemental fixture does not distinguish bytes from numbers");
    rejects([&] { check(original, numerically_equal); },
            "supplemental identity compared decoded numbers, not bytes");
  }
}

enum class Writer { kEmpty, kPartial, kFull };

// Deliberately independent of the poison selector: this is runner.cpp's
// production comparison, including its denominator floor and strict bounds.
bool numerical_mismatch(float actual, float expected, double atol,
                        double rtol) {
  const double left = actual, right = expected;
  if (left == right)
    return false;
  const double absolute = std::abs(left - right);
  const double relative =
      absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
  return !std::isfinite(absolute) || (absolute > atol && relative > rtol);
}

void poison_contract(flagdnnDataType_t dtype, std::vector<float> expected,
                     double atol = 0.0, double rtol = 0.0) {
  const auto poison = bench::capture_output_poison(expected, dtype, atol, rtol);
  const auto decoded =
      io::decode(io::encode(poison, dtype), dtype, poison.size());
  require(decoded == poison, "poison changed after a second dtype roundtrip");
  for (std::size_t i = 0; i < decoded.size(); ++i) {
    require(std::isfinite(decoded[i]),
            "poison must remain finite after dtype conversion");
    require(numerical_mismatch(decoded[i], expected[i], atol, rtol),
            "encoded poison can pass the production accuracy rule");
    require(expected[i] + 0.0F * decoded[i] == expected[i],
            "finite poison contaminated a beta-zero writer");
  }
  if (dtype == FLAGDNN_DATA_BOOLEAN) {
    const auto bytes = io::encode(poison, dtype);
    require(std::all_of(bytes.begin(), bytes.end(),
                        [](auto value) { return value == 0xA5U; }),
            "boolean poison lost its reserved byte");
  }
}

void finite_poison_contract() {
  const auto finite_types = {FLAGDNN_DATA_FLOAT32,  FLAGDNN_DATA_FLOAT16,
                             FLAGDNN_DATA_BFLOAT16, FLAGDNN_DATA_FP8_E4M3,
                             FLAGDNN_DATA_FP8_E5M2, FLAGDNN_DATA_FP8_E8M0,
                             FLAGDNN_DATA_BOOLEAN,  FLAGDNN_DATA_INT32};
  for (auto dtype : finite_types) {
    const std::vector<float> expected =
        dtype == FLAGDNN_DATA_BOOLEAN ? std::vector<float>{0, 1}
        : dtype == FLAGDNN_DATA_FP8_E8M0
            ? std::vector<float>{0x1p-127F, 1, 2, 0x1p127F}
            : std::vector<float>{0.0F, -0.0F, 1.0F, -1.0F, -97.0F};
    poison_contract(dtype, expected, 0.01, 0.01);
    rejects(
        [&] {
          (void)bench::capture_output_poison(std::vector<float>{0}, dtype, 0,
                                             1);
        },
        "relative tolerance one accepted a finite poison against zero");
    rejects(
        [&] {
          (void)bench::capture_output_poison(std::vector<float>{1}, dtype, 0,
                                             2);
        },
        "relative tolerance two accepted a finite poison");
    rejects(
        [&] {
          (void)bench::capture_output_poison(std::vector<float>{1}, dtype,
                                             std::numeric_limits<double>::max(),
                                             0);
        },
        "impossible absolute tolerance did not fail closed");
  }
  for (auto dtype :
       {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
    poison_contract(dtype, {-97, 0, 97, 1, -1}, 0.001, 0.01);
    poison_contract(dtype, {100, -100}, 0, 1.99);
    // Neither +/-1, dtype endpoints nor -expected satisfy both inequalities.
    // A representable interior value such as -1.75 is nevertheless valid.
    poison_contract(dtype, {1, -1}, 2.5, 1.5);
  }
  poison_contract(
      FLAGDNN_DATA_FLOAT32,
      {std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
       std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(),
       std::numeric_limits<float>::denorm_min(),
       -std::numeric_limits<float>::denorm_min()});
  poison_contract(FLAGDNN_DATA_FLOAT32,
                  {std::numeric_limits<float>::denorm_min()}, 0, 1);
  poison_contract(FLAGDNN_DATA_FLOAT32, {std::numeric_limits<float>::max()},
                  std::numeric_limits<float>::max(), 1.5);
  poison_contract(FLAGDNN_DATA_FLOAT16, {65504, -65504, 0x1p-24F, -0x1p-24F});
  poison_contract(FLAGDNN_DATA_BFLOAT16,
                  {0x1.fep127F, -0x1.fep127F, 0x1p-133F, -0x1p-133F});
  poison_contract(FLAGDNN_DATA_INT32,
                  {-2147483648.0F, 2147483520.0F, -1073741824, 1073741824, 0});

  // Enumerate every finite FP8 encoding, including signed zero, subnormals,
  // extremes and E8M0's exclusively positive scale values.
  for (auto dtype :
       {FLAGDNN_DATA_FP8_E4M3, FLAGDNN_DATA_FP8_E5M2, FLAGDNN_DATA_FP8_E8M0}) {
    std::vector<std::uint8_t> bytes(256);
    for (std::size_t i = 0; i < bytes.size(); ++i)
      bytes[i] = static_cast<std::uint8_t>(i);
    auto expected = io::decode(bytes, dtype, bytes.size());
    expected.erase(
        std::remove_if(expected.begin(), expected.end(),
                       [](float value) { return !std::isfinite(value); }),
        expected.end());
    poison_contract(dtype, expected, 0.001, 0.01);
  }

  // A host candidate near -expected is insufficient when the dtype rounds it
  // into the acceptance interval. Do not return that pre-conversion value.
  for (const auto &[dtype, expected] :
       {std::pair{FLAGDNN_DATA_FLOAT16, 1.0004F},
        std::pair{FLAGDNN_DATA_BFLOAT16, 1.003F},
        std::pair{FLAGDNN_DATA_FP8_E4M3, 1.04F},
        std::pair{FLAGDNN_DATA_FP8_E5M2, 1.1F}}) {
    rejects(
        [&] {
          (void)bench::capture_output_poison(std::vector<float>{expected},
                                             dtype, 0, 1.9999);
        },
        "dtype rounding allowed an ineffective poison");
  }
  rejects(
      [&] {
        (void)bench::capture_output_poison(std::vector<float>{0},
                                           FLAGDNN_DATA_FLOAT16, 65504, 0);
      },
      "equality to absolute tolerance must not count as a mismatch");
  for (double invalid : {-1.0, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    rejects(
        [&] {
          (void)bench::capture_output_poison(std::vector<float>{1},
                                             FLAGDNN_DATA_FLOAT32, invalid, 0);
        },
        "invalid absolute tolerance was accepted");
    rejects(
        [&] {
          (void)bench::capture_output_poison(std::vector<float>{1},
                                             FLAGDNN_DATA_FLOAT32, 0, invalid);
        },
        "invalid relative tolerance was accepted");
  }
  for (float invalid : {std::numeric_limits<float>::quiet_NaN()}) {
    rejects(
        [&] {
          (void)bench::capture_output_poison(std::vector<float>{invalid},
                                             FLAGDNN_DATA_FLOAT32, 0, 0);
        },
        "NaN reference was accepted");
  }
  const float infinity = std::numeric_limits<float>::infinity();
  for (auto dtype :
       {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16, FLAGDNN_DATA_BFLOAT16}) {
    poison_contract(dtype, {infinity, -infinity}, 0.001, 0.01);
    require(!numerical_mismatch(infinity, infinity, 0, 0) &&
                !numerical_mismatch(-infinity, -infinity, 0, 0) &&
                numerical_mismatch(infinity, -infinity, 0, 0),
            "infinity comparison does not distinguish its sign");
  }
  for (auto dtype : {FLAGDNN_DATA_BOOLEAN, FLAGDNN_DATA_INT32,
                     FLAGDNN_DATA_FP8_E4M3, FLAGDNN_DATA_FP8_E5M2,
                     FLAGDNN_DATA_FP8_E8M0})
    rejects(
        [&] {
          (void)bench::capture_output_poison(
              std::vector<float>{infinity}, dtype, 0, 0);
        },
        "reference infinity not preserved by the codec was accepted");
  rejects(
      [&] {
        (void)bench::capture_output_poison(
            std::vector<float>{1}, static_cast<flagdnnDataType_t>(-1), 0, 0);
      },
      "unsupported dtype was accepted");
}

void replay_contract(flagdnnDataType_t dtype, std::vector<float> expected,
                     Writer writer) {
  // Keep the existing padded contracts unchanged. FP8 padding's -97 codec is
  // not exact; use contiguous FP8 tensors rather than change padding semantics.
  const bool fp8 = dtype == FLAGDNN_DATA_FP8_E4M3 ||
                   dtype == FLAGDNN_DATA_FP8_E5M2 ||
                   dtype == FLAGDNN_DATA_FP8_E8M0;
  const bench::TensorSpec output = {7,
                                    dtype,
                                    {2, 3},
                                    fp8 ? std::vector<std::int64_t>{3, 1}
                                        : std::vector<std::int64_t>{7, 2}};
  constexpr double atol = 0.001, rtol = 0.01;
  expected = io::decode(io::encode(expected, dtype), dtype, expected.size());
  // Reproduce the dangerous starting state: direct probe and warmup have
  // already left the expected values in every logical output position.
  auto bytes = io::encode(io::scatter(expected, output), dtype);
  std::vector<double> samples(20, 2.0);
  const auto original_samples = samples;
  const int timed_executions = static_cast<int>(samples.size()) * 50;
  bool timer_active = false;
  std::string trace;
  int graph_replays = 0;
  const auto action = [&] {
    return bench::validate_fresh_capture(
        [&] {
          require(!timer_active, "reset entered a timed sample");
          trace += 'R';
          bytes = io::encode(io::scatter(bench::capture_output_poison(
                                             expected, dtype, atol, rtol),
                                         output),
                             dtype);
        },
        [&] {
          require(!timer_active, "postcheck replay entered a timed sample");
          trace += 'L';
          ++graph_replays;
          auto physical =
              io::decode(bytes, dtype, io::storage_element_count(output));
          const std::size_t writes = writer == Writer::kFull ? expected.size()
                                     : writer == Writer::kPartial
                                         ? expected.size() - 1
                                         : 0;
          for (int invocation = 0; invocation < 50; ++invocation)
            for (std::size_t i = 0; i < writes; ++i)
              // Pooling kernels overwrite infinite outputs directly.
              // Retain the beta-zero adversarial writer for finite outputs.
              physical[io::logical_offset(i, output)] =
                  std::isinf(expected[i])
                      ? expected[i]
                      : expected[i] +
                            0.0F * physical[io::logical_offset(i, output)];
          bytes = io::encode(physical, dtype);
        },
        [&] {
          require(!timer_active, "postcheck sync entered a timed sample");
          trace += 'S';
        },
        [&] {
          trace += 'V';
          const auto physical =
              io::decode(bytes, dtype, io::storage_element_count(output));
          io::require_padding_unchanged("mock captured graph", physical,
                                        output);
          const auto actual = io::gather(physical, output);
          for (std::size_t i = 0; i < expected.size(); ++i)
            require(!numerical_mismatch(actual[i], expected[i], atol, rtol),
                    "captured graph left an unwritten output");
          return 17;
        });
  };
  if (writer == Writer::kFull)
    require(action() == 17, "fresh validation result was lost");
  else
    rejects(action, "empty or partially-writing graph was accepted");
  require(trace == "RLSV" && graph_replays == 1,
          "fresh capture validation order or replay count differs");
  require(samples == original_samples && timed_executions == 1000,
          "fresh replay changed the 20 x 50 timed samples");
}

void short_circuit_contract() {
  for (int fail = 0; fail < 4; ++fail) {
    int calls = 0;
    const auto step = [&] {
      if (calls++ == fail)
        throw std::runtime_error("injected failure");
    };
    rejects([&] { bench::validate_fresh_capture(step, step, step, step); },
            "fresh replay swallowed an error");
    require(calls == fail + 1, "fresh replay continued after an error");
  }
}
} // namespace

int main() {
  try {
    filter_contract();
    execution_status_contract();
    finite_poison_contract();
    bit_exact_identity_contract();
    supplemental_bit_exact_identity_contract();
    for (auto writer : {Writer::kEmpty, Writer::kPartial, Writer::kFull}) {
      for (auto dtype : {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16,
                         FLAGDNN_DATA_BFLOAT16}) {
        replay_contract(dtype, {-97, 0, 1, -2, 0.75F, -97}, writer);
        replay_contract(dtype, std::vector<float>(6, -97), writer);
        const float infinity = std::numeric_limits<float>::infinity();
        replay_contract(dtype, {infinity, -infinity, 0, 1, -1, -infinity},
                        writer);
      }
      replay_contract(FLAGDNN_DATA_INT32,
                      {-1073741824, 0, 1073741824, 65535, -97, 2}, writer);
      replay_contract(FLAGDNN_DATA_INT32,
                      {-2147483648.0F, 2147483520.0F, 0, -1, 1, 65535}, writer);
      replay_contract(FLAGDNN_DATA_BOOLEAN, std::vector<float>(6, 0), writer);
      replay_contract(FLAGDNN_DATA_BOOLEAN, std::vector<float>(6, 1), writer);
      replay_contract(FLAGDNN_DATA_BOOLEAN, {0, 1, 0, 1, 0, 1}, writer);
      replay_contract(FLAGDNN_DATA_FP8_E4M3,
                      {0, -0.0F, -97, 448, -448, 0x1p-9F}, writer);
      replay_contract(FLAGDNN_DATA_FP8_E5M2,
                      {0, -0.0F, -97, 57344, -57344, 0x1p-16F}, writer);
      replay_contract(FLAGDNN_DATA_FP8_E8M0,
                      {0x1p-127F, 1, 2, 128, 0x1p127F, 0x1p-126F}, writer);
    }
    short_circuit_contract();
    std::cout << "PASS: performance-independent execution status; unmatched "
                 "native filters; fresh captured replay rejects "
                 "empty/partial writes across FP32/FP16/BF16/FP8/INT32/BOOL; "
                 "finite dtype-aware poison for finite/infinite references, "
                 "beta-zero writers and tolerance "
                 "boundaries; FP8 identity original-byte checks and encoded "
                 "padding; "
                 "BF16/INT32/BOOL bit-exact noncompact identity and input "
                 "immutability; "
                 "padding, sample accounting and error propagation\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
