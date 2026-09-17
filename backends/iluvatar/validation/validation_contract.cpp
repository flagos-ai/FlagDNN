// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "corex_cudnn_reference.hpp"
#include "corex_cudnn_status.hpp"

#include "common/add.hpp"
#include "common/attention.hpp"
#include "common/composite.hpp"
#include "common/convolution.hpp"
#include "common/layout.hpp"
#include "common/matmul.hpp"
#include "common/normalization.hpp"
#include "common/pointwise.hpp"
#include "common/reduction.hpp"

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::testing {

int run_pointwise_functional_test(int, char **,
                                  std::span<const PointwiseTestCase>,
                                  std::string_view) {
  throw std::logic_error("catalog contract runner stub must not execute");
}

} // namespace flagdnn::testing

namespace {

using CaseMap = std::map<std::string, std::set<std::string>>;
using flagdnn::iluvatar::validation::CorexCudnnCapabilityCatalog;
using flagdnn::iluvatar::validation::CorexCudnnCatalogClassification;
using flagdnn::testing::PointwiseCaseDefinition;
using flagdnn::testing::PointwiseInputDomain;

template <typename Cases>
void add_cases(CaseMap &output, std::string operation, const Cases &cases) {
  auto &names = output[std::move(operation)];
  for (const auto &test_case : cases) {
    if (!names.insert(test_case.name).second) {
      throw std::runtime_error("neutral case generator returned a duplicate");
    }
  }
  if (names.empty()) {
    throw std::runtime_error("neutral case generator returned no cases");
  }
}

enum class PointwiseArity { kUnary, kBinary, kTernary };

struct PointwiseSpecification {
  const char *operation;
  flagdnnPointwiseMode_t mode;
  PointwiseArity arity;
};

CaseMap neutral_cases() {
  using namespace flagdnn::testing;
  CaseMap output;
  const std::vector<PointwiseSpecification> pointwise = {
      {"relu_backward", FLAGDNN_POINTWISE_RELU_BWD, PointwiseArity::kBinary},
      {"leaky_relu_backward", FLAGDNN_POINTWISE_RELU_BWD,
       PointwiseArity::kBinary},
      {"tanh_backward", FLAGDNN_POINTWISE_TANH_BWD, PointwiseArity::kBinary},
      {"elu_backward", FLAGDNN_POINTWISE_ELU_BWD, PointwiseArity::kBinary},
      {"gelu_backward", FLAGDNN_POINTWISE_GELU_BWD, PointwiseArity::kBinary},
      {"gelu_approx_tanh_backward", FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD,
       PointwiseArity::kBinary},
      {"softplus_backward", FLAGDNN_POINTWISE_SOFTPLUS_BWD,
       PointwiseArity::kBinary},
      {"swish_backward", FLAGDNN_POINTWISE_SWISH_BWD, PointwiseArity::kBinary},
      {"abs", FLAGDNN_POINTWISE_ABS, PointwiseArity::kUnary},
      {"binary_select", FLAGDNN_POINTWISE_BINARY_SELECT,
       PointwiseArity::kTernary},
      {"ceil", FLAGDNN_POINTWISE_CEIL, PointwiseArity::kUnary},
      {"cmp_eq", FLAGDNN_POINTWISE_CMP_EQ, PointwiseArity::kBinary},
      {"cmp_ge", FLAGDNN_POINTWISE_CMP_GE, PointwiseArity::kBinary},
      {"cmp_gt", FLAGDNN_POINTWISE_CMP_GT, PointwiseArity::kBinary},
      {"cmp_le", FLAGDNN_POINTWISE_CMP_LE, PointwiseArity::kBinary},
      {"cmp_lt", FLAGDNN_POINTWISE_CMP_LT, PointwiseArity::kBinary},
      {"cmp_neq", FLAGDNN_POINTWISE_CMP_NEQ, PointwiseArity::kBinary},
      {"cos", FLAGDNN_POINTWISE_COS, PointwiseArity::kUnary},
      {"div", FLAGDNN_POINTWISE_DIV, PointwiseArity::kBinary},
      {"elu", FLAGDNN_POINTWISE_ELU_FWD, PointwiseArity::kUnary},
      {"erf", FLAGDNN_POINTWISE_ERF, PointwiseArity::kUnary},
      {"exp", FLAGDNN_POINTWISE_EXP, PointwiseArity::kUnary},
      {"floor", FLAGDNN_POINTWISE_FLOOR, PointwiseArity::kUnary},
      {"gelu", FLAGDNN_POINTWISE_GELU_FWD, PointwiseArity::kUnary},
      {"gelu_approx_tanh", FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD,
       PointwiseArity::kUnary},
      {"identity", FLAGDNN_POINTWISE_IDENTITY, PointwiseArity::kUnary},
      {"leaky_relu", FLAGDNN_POINTWISE_RELU_FWD, PointwiseArity::kUnary},
      {"log", FLAGDNN_POINTWISE_LOG, PointwiseArity::kUnary},
      {"logical_and", FLAGDNN_POINTWISE_LOGICAL_AND, PointwiseArity::kBinary},
      {"logical_not", FLAGDNN_POINTWISE_LOGICAL_NOT, PointwiseArity::kUnary},
      {"logical_or", FLAGDNN_POINTWISE_LOGICAL_OR, PointwiseArity::kBinary},
      {"max", FLAGDNN_POINTWISE_MAX, PointwiseArity::kBinary},
      {"min", FLAGDNN_POINTWISE_MIN, PointwiseArity::kBinary},
      {"mod", FLAGDNN_POINTWISE_MOD, PointwiseArity::kBinary},
      {"mul", FLAGDNN_POINTWISE_MUL, PointwiseArity::kBinary},
      {"neg", FLAGDNN_POINTWISE_NEG, PointwiseArity::kUnary},
      {"pow", FLAGDNN_POINTWISE_POW, PointwiseArity::kBinary},
      {"reciprocal", FLAGDNN_POINTWISE_RECIPROCAL, PointwiseArity::kUnary},
      {"relu", FLAGDNN_POINTWISE_RELU_FWD, PointwiseArity::kUnary},
      {"rsqrt", FLAGDNN_POINTWISE_RSQRT, PointwiseArity::kUnary},
      {"scale", FLAGDNN_POINTWISE_MUL, PointwiseArity::kBinary},
      {"sigmoid", FLAGDNN_POINTWISE_SIGMOID_FWD, PointwiseArity::kUnary},
      {"sigmoid_backward", FLAGDNN_POINTWISE_SIGMOID_BWD,
       PointwiseArity::kBinary},
      {"sin", FLAGDNN_POINTWISE_SIN, PointwiseArity::kUnary},
      {"softplus", FLAGDNN_POINTWISE_SOFTPLUS_FWD, PointwiseArity::kUnary},
      {"sqrt", FLAGDNN_POINTWISE_SQRT, PointwiseArity::kUnary},
      {"sub", FLAGDNN_POINTWISE_SUB, PointwiseArity::kBinary},
      {"swish", FLAGDNN_POINTWISE_SWISH_FWD, PointwiseArity::kUnary},
      {"tan", FLAGDNN_POINTWISE_TAN, PointwiseArity::kUnary},
      {"tanh", FLAGDNN_POINTWISE_TANH_FWD, PointwiseArity::kUnary},
  };
  for (const PointwiseSpecification &specification : pointwise) {
    const PointwiseCaseDefinition definition{
        .operation_name = specification.operation,
        .mode = specification.mode,
        .input_domain = PointwiseInputDomain::kReal,
    };
    switch (specification.arity) {
    case PointwiseArity::kUnary:
      add_cases(output, specification.operation,
                make_unary_pointwise_cases(definition));
      break;
    case PointwiseArity::kBinary:
      add_cases(output, specification.operation,
                make_binary_pointwise_cases(definition));
      break;
    case PointwiseArity::kTernary:
      add_cases(output, specification.operation,
                make_binary_select_cases(definition));
      break;
    }
  }

  add_cases(output, "add", make_add_cases());
  add_cases(output, "add_square", make_add_square_cases());
  add_cases(output, "batchnorm", make_batchnorm_cases());
  add_cases(output, "batchnorm_inference", make_batchnorm_inference_cases());
  add_cases(output, "conv_bias_relu", make_conv_bias_relu_cases());
  add_cases(output, "conv_dgrad",
            make_convolution_cases(ConvolutionDirection::kDgrad));
  add_cases(output, "conv_fprop",
            make_convolution_cases(ConvolutionDirection::kFprop));
  add_cases(output, "conv_wgrad",
            make_convolution_cases(ConvolutionDirection::kWgrad));
  add_cases(output, "layernorm", make_layernorm_cases());
  add_cases(output, "matmul", make_matmul_cases());
  add_cases(output, "reduction", make_reduction_cases());
  add_cases(output, "reshape", make_layout_cases(LayoutOperation::kReshape));
  add_cases(output, "rmsnorm", make_rmsnorm_cases());
  add_cases(output, "sdpa", make_sdpa_cases());
  add_cases(output, "sdpa_backward", make_sdpa_backward_cases());
  add_cases(output, "sdpa_fp8", make_sdpa_fp8_cases());
  add_cases(output, "sdpa_fp8_backward", make_sdpa_fp8_backward_cases());
  add_cases(output, "slice", make_layout_cases(LayoutOperation::kSlice));
  add_cases(output, "transpose",
            make_layout_cases(LayoutOperation::kTranspose));
  return output;
}

std::set<std::string> parse_operator_block(const std::string &source,
                                           std::string_view variable) {
  const std::string marker = "set(" + std::string(variable);
  const std::size_t start = source.find(marker);
  if (start == std::string::npos) {
    throw std::runtime_error("operator manifest is missing " +
                             std::string(variable));
  }
  const std::size_t end = source.find(')', start + marker.size());
  if (end == std::string::npos) {
    throw std::runtime_error("operator manifest block is unterminated");
  }
  std::string block =
      source.substr(start + marker.size(), end - start - marker.size());
  std::istringstream input(block);
  std::set<std::string> result;
  std::string token;
  while (input >> token) {
    if (token.starts_with('#')) {
      std::getline(input, token);
      continue;
    }
    if (!token.starts_with("${")) {
      result.insert(std::move(token));
    }
  }
  return result;
}

void require_operator_manifest(const CaseMap &cases, const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open neutral operator manifest");
  }
  const std::string source{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
  std::set<std::string> expected =
      parse_operator_block(source, "FLAGDNN_BENCHMARK_OPERATORS");
  const auto functional =
      parse_operator_block(source, "FLAGDNN_FUNCTIONAL_OPERATORS");
  expected.insert(functional.begin(), functional.end());
  const auto activation =
      parse_operator_block(source, "FLAGDNN_ACTIVATION_BACKWARD_EXTENSIONS");
  expected.insert(activation.begin(), activation.end());
  std::set<std::string> actual;
  for (const auto &[operation, operation_cases] : cases) {
    (void)operation_cases;
    actual.insert(operation);
  }
  if (actual != expected) {
    throw std::runtime_error(
        "generated capability operators do not match Operators.cmake");
  }
}

std::vector<std::string> primitive_sequence(std::string_view operation) {
  if (operation == "add")
    return {"cudnnOpTensor(ADD)"};
  if (operation == "mul" || operation == "scale")
    return {"cudnnOpTensor(MUL)"};
  if (operation == "min")
    return {"cudnnOpTensor(MIN)"};
  if (operation == "max")
    return {"cudnnOpTensor(MAX)"};
  if (operation == "sqrt")
    return {"cudnnOpTensor(SQRT)"};
  if (operation == "abs")
    return {"cudnnOpTensor(MAX,x,-x)"};
  if (operation == "logical_and")
    return {"cudnnOpTensor(MUL,bool)"};
  if (operation == "logical_or")
    return {"cudnnOpTensor(MAX,bool)"};
  if (operation == "logical_not")
    return {"cudnnOpTensor(ADD,1-x,bool)"};
  if (operation == "leaky_relu")
    return {"cudnnOpTensor(MAX,x,slope*x)"};
  if (operation == "leaky_relu_backward")
    return {"cudnnActivationBackward(RELU)", "cudnnOpTensor(ADD)"};
  if (operation == "sub")
    return {"cudnnOpTensor(ADD,alpha2=-alpha)"};
  if (operation == "neg")
    return {"cudnnTransformTensor(alpha=-1)"};
  if (operation == "identity")
    return {"cudnnTransformTensor(alpha=1)"};
  if (operation == "add_square")
    return {"cudnnOpTensor(ADD)", "cudnnOpTensor(MUL,self)"};
  if (operation == "relu")
    return {"cudnnActivationForward(RELU)"};
  if (operation == "sigmoid")
    return {"cudnnActivationForward(SIGMOID)"};
  if (operation == "tanh")
    return {"cudnnActivationForward(TANH)"};
  if (operation == "elu")
    return {"cudnnActivationForward(ELU)"};
  if (operation == "swish" || operation == "softplus_backward")
    return {"cudnnOpTensor(ADD,beta*x)", "cudnnActivationForward(SIGMOID)",
            "cudnnOpTensor(MUL)"};
  if (operation == "swish_backward")
    return {"cudnnOpTensor(ADD,beta*x)", "cudnnActivationForward(SIGMOID)",
            "cudnnOpTensor(MUL)", "cudnnActivationBackward(SIGMOID)",
            "cudnnOpTensor(ADD)"};
  if (operation == "gelu")
    return {"cudnnActivationForward(GELU)"};
  if (operation == "gelu_approx_tanh")
    return {"cudnnActivationForward(GELU_TAHN)"};
  if (operation == "sigmoid_backward")
    return {"cudnnActivationForward(SIGMOID)",
            "cudnnActivationBackward(SIGMOID)"};
  if (operation == "relu_backward")
    return {"cudnnActivationForward(RELU)", "cudnnActivationBackward(RELU)"};
  if (operation == "tanh_backward")
    return {"cudnnActivationForward(TANH)", "cudnnActivationBackward(TANH)"};
  if (operation == "elu_backward")
    return {"cudnnActivationForward(ELU)", "cudnnActivationBackward(ELU)"};
  if (operation == "rmsnorm")
    return {"cudnnRmsNormalizationForward", "cudnnOpTensor(ADD,bias)"};
  if (operation == "reduction")
    return {"cudnnReduceTensor"};
  if (operation == "conv_fprop")
    return {"cudnnConvolutionForward"};
  if (operation == "conv_dgrad")
    return {"cudnnConvolutionBackwardData"};
  if (operation == "conv_wgrad")
    return {"cudnnConvolutionBackwardFilter"};
  if (operation == "conv_bias_relu")
    return {"cudnnConvolutionForward", "cudnnOpTensor(ADD,bias)",
            "cudnnActivationForward(RELU)"};
  if (operation == "batchnorm")
    return {"cudnnBatchNormalizationForwardTraining"};
  if (operation == "batchnorm_inference")
    return {"cudnnBatchNormalizationForwardInference"};
  if (operation == "reshape" || operation == "transpose" ||
      operation == "slice")
    return {"cudnnTransformTensor"};
  return {};
}

std::string skip_reason(std::string_view operation) {
  if (operation == "sdpa_fp8" || operation == "sdpa_fp8_backward") {
    return "DTYPE_UNSUPPORTED";
  }
  if (operation == "leaky_relu")
    return "ATTRIBUTE_UNSUPPORTED";
  return "NO_EXACT_CUDNN_PRIMITIVE";
}

std::string skip_detail(std::string_view operation) {
  if (operation == "sdpa_fp8" || operation == "sdpa_fp8_backward") {
    return "CoreX cudnn.h 7605 has no FP8 cudnnDataType_t";
  }
  if (operation == "sdpa" || operation == "sdpa_backward") {
    return "CoreX libcudnn.so.7 exports no Flash Attention entry points";
  }
  if (operation == "leaky_relu") {
    return "classic cuDNN ReLU has no negative-slope attribute";
  }
  return "no exact classic CoreX cuDNN primitive or sequence";
}

void write_catalog(const CaseMap &cases, const std::string &path) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot create CoreX cuDNN catalog template");
  }
  output << "{\n"
         << "  \"schema_version\": 1,\n"
         << "  \"baseline\": {\n"
         << "    \"corex\": \"4.4.0\",\n"
         << "    \"target\": \"corex_71\",\n"
         << "    \"cudnn_header\": 7605,\n"
         << "    \"cudnn_runtime\": 7605\n"
         << "  },\n"
         << "  \"operators\": {\n";
  std::size_t operation_index = 0;
  for (const auto &[operation, operation_cases] : cases) {
    const std::vector<std::string> primitives = primitive_sequence(operation);
    const bool candidate = !primitives.empty();
    output << "    \"" << operation << "\": {\n"
           << "      \"cases\": {\n";
    std::size_t case_index = 0;
    for (const std::string &case_name : operation_cases) {
      output << "        \"" << case_name << "\": {\n"
             << "          \"classification\": \""
             << (candidate ? "candidate" : "semantic_unsupported") << "\",\n"
             << "          \"reason\": \""
             << (candidate ? "" : skip_reason(operation)) << "\",\n"
             << "          \"detail\": \""
             << (candidate ? "pending real-device CoreX cuDNN qualification"
                           : skip_detail(operation))
             << "\",\n"
             << "          \"primitive_sequence\": [";
      for (std::size_t index = 0; index < primitives.size(); ++index) {
        output << (index == 0 ? "" : ", ") << '"' << primitives[index] << '"';
      }
      output << "]\n        }"
             << (++case_index == operation_cases.size() ? "\n" : ",\n");
    }
    output << "      }\n    }"
           << (++operation_index == cases.size() ? "\n" : ",\n");
  }
  output << "  }\n}\n";
  if (!output) {
    throw std::runtime_error("failed to write CoreX cuDNN catalog template");
  }
}

std::string fixture(std::string classification, std::string reason,
                    std::string primitives, std::string baseline = "7605",
                    std::string operation = "op",
                    std::string case_name = "case") {
  return "{\"schema_version\":1,\"baseline\":{"
         "\"corex\":\"4.4.0\",\"target\":\"corex_71\","
         "\"cudnn_header\":" +
         baseline +
         ",\"cudnn_runtime\":7605},"
         "\"operators\":{\"" +
         operation + "\":{\"cases\":{\"" + case_name +
         "\":{\"classification\":\"" + classification + "\",\"reason\":\"" +
         reason + "\",\"primitive_sequence\":" + primitives + "}}}}}";
}

template <typename Function>
void require_rejected(std::string_view name, Function &&function) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error("negative catalog fixture passed: " +
                           std::string(name));
}

void run_negative_fixtures() {
  require_rejected("duplicate case", [] {
    (void)CorexCudnnCapabilityCatalog::parse(
        "{\"schema_version\":1,\"baseline\":{"
        "\"corex\":\"4.4.0\",\"target\":\"corex_71\","
        "\"cudnn_header\":7605,\"cudnn_runtime\":7605},"
        "\"operators\":{\"op\":{\"cases\":{\"case\":{},"
        "\"case\":{}}}}}");
  });
  require_rejected("unknown classification", [] {
    (void)CorexCudnnCapabilityCatalog::parse(
        fixture("maybe", "", "[\"cudnnOpTensor\"]"));
  });
  require_rejected("unknown reason", [] {
    (void)CorexCudnnCapabilityCatalog::parse(
        fixture("semantic_unsupported", "UNKNOWN", "[]"));
  });
  require_rejected("supported with skip reason", [] {
    (void)CorexCudnnCapabilityCatalog::parse(fixture(
        "qualified_supported", "SHAPE_UNSUPPORTED", "[\"cudnnOpTensor\"]"));
  });
  require_rejected("unsupported without reason", [] {
    (void)CorexCudnnCapabilityCatalog::parse(
        fixture("semantic_unsupported", "", "[]"));
  });
  require_rejected("missing primitive sequence", [] {
    std::string value = fixture("candidate", "", "[]");
    const std::string field = ",\"primitive_sequence\":[]";
    value.erase(value.find(field), field.size());
    (void)CorexCudnnCapabilityCatalog::parse(value);
  });
  require_rejected("wrong baseline", [] {
    (void)CorexCudnnCapabilityCatalog::parse(
        fixture("candidate", "", "[\"cudnnOpTensor\"]", "8900"));
  });
  require_rejected("missing operator", [] {
    const auto catalog = CorexCudnnCapabilityCatalog::parse(
        fixture("candidate", "", "[\"cudnnOpTensor\"]"));
    catalog.require_exact_cases({{"different", {"case"}}});
  });
  require_rejected("missing case", [] {
    const auto catalog = CorexCudnnCapabilityCatalog::parse(
        fixture("candidate", "", "[\"cudnnOpTensor\"]"));
    catalog.require_exact_cases({{"op", {"different"}}});
  });
  require_rejected("case owned by wrong operator", [] {
    const auto catalog = CorexCudnnCapabilityCatalog::parse(fixture(
        "candidate", "", "[\"cudnnOpTensor\"]", "7605", "wrong", "case"));
    catalog.require_exact_cases({{"right", {"case"}}});
  });
}

void run_status_contract() {
  using namespace flagdnn::iluvatar::validation;
  constexpr std::array statuses{
      CUDNN_STATUS_SUCCESS,
      CUDNN_STATUS_NOT_INITIALIZED,
      CUDNN_STATUS_ALLOC_FAILED,
      CUDNN_STATUS_BAD_PARAM,
      CUDNN_STATUS_INTERNAL_ERROR,
      CUDNN_STATUS_INVALID_VALUE,
      CUDNN_STATUS_ARCH_MISMATCH,
      CUDNN_STATUS_MAPPING_ERROR,
      CUDNN_STATUS_EXECUTION_FAILED,
      CUDNN_STATUS_NOT_SUPPORTED,
      CUDNN_STATUS_LICENSE_ERROR,
      CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING,
      CUDNN_STATUS_RUNTIME_IN_PROGRESS,
      CUDNN_STATUS_RUNTIME_FP_OVERFLOW,
  };
  for (const cudnnStatus_t status : statuses) {
    if (cudnn_status_is_runtime_capability(status) !=
        (status == CUDNN_STATUS_NOT_SUPPORTED)) {
      throw std::runtime_error("cuDNN runtime status classification drifted");
    }
  }
  check_cudnn(CUDNN_STATUS_SUCCESS, "status success fixture");
  try {
    check_cudnn(CUDNN_STATUS_BAD_PARAM, "status error fixture");
  } catch (const CorexCudnnStatusError &error) {
    if (error.status() == CUDNN_STATUS_BAD_PARAM) {
      return;
    }
  }
  throw std::runtime_error("CoreX cuDNN status error lost its status code");
}

} // namespace

int main(int argc, char **argv) {
  try {
    const CaseMap cases = neutral_cases();
    if (argc == 4 && std::string_view(argv[1]) == "--write-template") {
      require_operator_manifest(cases, argv[3]);
      write_catalog(cases, argv[2]);
      return EXIT_SUCCESS;
    }
    const bool require_closed =
        argc == 4 && std::string_view(argv[3]) == "--require-closed";
    if (argc != 3 && !require_closed) {
      std::cerr << "usage: validation_contract <catalog> <Operators.cmake>\n";
      return EXIT_FAILURE;
    }
    require_operator_manifest(cases, argv[2]);
    run_negative_fixtures();
    run_status_contract();
    const CorexCudnnCapabilityCatalog catalog =
        CorexCudnnCapabilityCatalog::load(argv[1]);
    catalog.require_exact_cases(cases);

    std::size_t total = 0;
    std::size_t qualified = 0;
    std::size_t candidates = 0;
    std::size_t static_skips = 0;
    for (const auto &[operation, operation_cases] : catalog.records()) {
      (void)operation;
      total += operation_cases.size();
      for (const auto &[case_name, record] : operation_cases) {
        (void)case_name;
        qualified += record.classification ==
                     CorexCudnnCatalogClassification::kQualifiedSupported;
        candidates += record.classification ==
                      CorexCudnnCatalogClassification::kCandidate;
        static_skips += record.classification ==
                        CorexCudnnCatalogClassification::kSemanticUnsupported;
      }
    }
    if (total == 0 || qualified + candidates == 0 || static_skips == 0) {
      throw std::runtime_error("catalog classifications are incomplete");
    }
    if (require_closed && candidates != 0) {
      throw std::runtime_error("catalog closure still contains candidates");
    }
    std::cout << "PASS CoreX cuDNN capability catalog: operators="
              << catalog.records().size() << " cases=" << total
              << " qualified=" << qualified << " candidates=" << candidates
              << " static_skips=" << static_skips << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << "FAIL CoreX cuDNN capability catalog: " << error.what()
              << '\n';
    return EXIT_FAILURE;
  }
}
