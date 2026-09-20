// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "corex_cudnn_status.hpp"
#include "functional/raw_reference.hpp"

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace f = flagdnn::iluvatar::validation::functional;
namespace v = flagdnn::iluvatar::validation;
using Values = std::vector<float>;
using Tensor = flagdnn::testing::TestTensor;

void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}

// This executable writes independently specified expected values to device
// buffers. It tests reference selection/comparison, not a production kernel.
class KnownOutput final : public flagdnn::testing::TestExecutable {
public:
  KnownOutput(Tensor output, const Values &expected)
      : output_(std::move(output)),
        bytes_(f::encode(f::scatter(expected, output_), output_.data_type)) {}
  std::size_t workspace_size() const noexcept override { return 0; }
  void execute(std::span<const flagdnnBinding_t> bindings, void *, std::size_t,
               flagdnnStream_t stream) override {
    for (const auto &binding : bindings) {
      if (binding.uid != output_.uid) continue;
      f::check_cuda(cuMemcpyHtoDAsync(
          reinterpret_cast<CUdeviceptr>(binding.device_pointer), bytes_.data(),
          bytes_.size(), reinterpret_cast<CUstream>(stream)), "contract write");
    }
  }
private:
  Tensor output_;
  std::vector<std::uint8_t> bytes_;
};

int main(int argc, char **argv) {
  try {
    struct Case {
      const char *name;
      flagdnnPointwiseMode_t mode;
      Values expected;
    };
    const std::array cases = {
        Case{"div", FLAGDNN_POINTWISE_DIV, {1.0F / 3, 0.5F}},
        Case{"pow", FLAGDNN_POINTWISE_POW, {1, 16}},
        Case{"mod", FLAGDNN_POINTWISE_MOD, {1, 2}},
        Case{"cmp_eq", FLAGDNN_POINTWISE_CMP_EQ, {1, 0}},
    };
    const f::BuildExecutable unsupported = []()
        -> std::unique_ptr<flagdnn::testing::TestExecutable> {
      throw v::CorexCudnnStatusError(CUDNN_STATUS_NOT_SUPPORTED, "contract");
    };
    for (const auto &test : cases) {
      for (auto dtype : {FLAGDNN_DATA_FLOAT32, FLAGDNN_DATA_FLOAT16,
                         FLAGDNN_DATA_BFLOAT16}) {
        const Tensor left{1, dtype, {2}, {2}, 0};
        const Tensor right{2, dtype, {2}, {2}, 0};
        const bool comparison = test.mode == FLAGDNN_POINTWISE_CMP_EQ;
        const Tensor output{3, comparison ? FLAGDNN_DATA_BOOLEAN : dtype,
                            {2}, {2}, 0};
        f::CasePlan plan{test.name, "cpu_fallback_contract",
                        {{left, f::InputDomain::kReal, {1, 2}},
                         {right, f::InputDomain::kReal,
                          comparison ? Values{1, 4} : Values{3, 4}}},
                        {{output, 0, 0, "output"}}};
        f::FunctionalSuite suite(argc, argv, test.name, "CONTRACT_FUNCTIONAL");
        suite.run(plan, [&] {
          return std::make_unique<KnownOutput>(output, test.expected);
        }, unsupported, {}, true,
            f::binary_cpu_reference(plan, test.mode));
        require(suite.finish() == 0, "CPU fallback did not count as executed");
      }
    }

    const Tensor tensor{1, FLAGDNN_DATA_FLOAT32, {2}, {1}, 0};
    Tensor right = tensor, output = tensor;
    right.uid = 2;
    output.uid = 3;
    f::CasePlan plan{"div", "div_fp32_1x1x16",
                    {{tensor, f::InputDomain::kReal, {1, 2}},
                     {right, f::InputDomain::kReal, {3, 4}}},
                    {{output, 0, 0, "output"}}};
    const f::BuildExecutable correct = [&] {
      return std::make_unique<KnownOutput>(output, Values{1.0F / 3, 0.5F});
    };
    for (auto mode : {FLAGDNN_POINTWISE_ADD, FLAGDNN_POINTWISE_SUB,
                      FLAGDNN_POINTWISE_MUL, FLAGDNN_POINTWISE_MAX,
                      FLAGDNN_POINTWISE_MIN, FLAGDNN_POINTWISE_RELU_FWD}) {
      require(!f::binary_cpu_reference(plan, mode),
              "unrelated operator unexpectedly acquired a CPU fallback");
    }
    // Catalog-declared gaps must use CPU without trying the vendor API.
    const f::BuildExecutable forbidden = []()
        -> std::unique_ptr<flagdnn::testing::TestExecutable> {
      throw std::runtime_error("unexpected cuDNN call");
    };
    {
      f::FunctionalSuite suite(argc, argv, "div", "CONTRACT_FUNCTIONAL");
      suite.run(plan, correct, forbidden, {}, false,
                f::binary_cpu_reference(plan, FLAGDNN_POINTWISE_DIV));
      require(suite.finish() == 0, "catalog gap did not use CPU");
    }
    // Wrong numerical results must fail, even though production ran normally.
    {
      f::FunctionalSuite suite(argc, argv, "div", "CONTRACT_FUNCTIONAL");
      bool rejected = false;
      try {
        suite.run(plan, [&] {
          return std::make_unique<KnownOutput>(output, Values{9, 9});
        }, forbidden, {}, false,
                  f::binary_cpu_reference(plan, FLAGDNN_POINTWISE_DIV));
      } catch (const std::runtime_error &error) {
        rejected = std::string(error.what()).find("differs in") != std::string::npos;
      }
      require(rejected, "CPU fallback accepted incorrect output");
    }
    const f::HostReference forbidden_cpu = [](const auto &)
        -> std::vector<Values> {
      throw std::runtime_error("unexpected CPU reference");
    };
    // Supported cuDNN remains preferred; benchmark never substitutes CPU.
    {
      f::FunctionalSuite suite(argc, argv, "div", "CONTRACT_FUNCTIONAL");
      suite.run(plan, correct, correct, {}, true, forbidden_cpu);
      require(suite.finish() == 0, "supported cuDNN reference was replaced");
    }
    {
      f::FunctionalSuite suite(argc, argv, "div", "CONTRACT_BENCHMARK");
      suite.run(plan, correct, unsupported, {}, true, forbidden_cpu);
      require(suite.finish() == 77, "benchmark did not remain skipped");
    }
    // Invalid cuDNN invocations remain errors, not fallback opportunities.
    {
      f::FunctionalSuite suite(argc, argv, "div", "CONTRACT_FUNCTIONAL");
      bool rejected = false;
      try {
        suite.run(plan, correct, []()
            -> std::unique_ptr<flagdnn::testing::TestExecutable> {
          throw v::CorexCudnnStatusError(CUDNN_STATUS_BAD_PARAM, "contract");
        }, {}, true, forbidden_cpu);
      } catch (const v::CorexCudnnStatusError &error) {
        rejected = error.status() == CUDNN_STATUS_BAD_PARAM;
      }
      require(rejected, "cuDNN error was hidden by CPU fallback");
    }
    std::cout << "PASS CPU fallback selection, accuracy and benchmark isolation\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
