/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/backend_api.h"
#include "backends/mthreads/artifact.hpp"
#include "backends/mthreads/context.hpp"
#include "backends/mthreads/error.hpp"
#include "runtime/sha256.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot open contract input: " + path.string());
  }
  return {
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>()};
}

template <typename Function>
void expect_compilation_failure(
    Function&& function, std::string_view label) {
  try {
    std::forward<Function>(function)();
  } catch (const flagdnn::mthreads::MthreadsError& error) {
    if (error.result() != FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED) {
      throw std::runtime_error(
          "contract rejection had the wrong status: " +
          std::string(label));
    }
    return;
  }
  throw std::runtime_error(
      "invalid artifact/build input was accepted: " + std::string(label));
}

void validate_result(
    const flagdnn::mthreads::MthreadsArtifact& artifact) {
  if (artifact.workspace_size < 4096 ||
      artifact.workspace_alignment != 256 ||
      artifact.binding_uids.empty() || artifact.binding_uids.size() > 32 ||
      artifact.stages.empty()) {
    throw std::runtime_error(
        "parsed mthreads artifact does not expose the expected ABI");
  }
  for (std::size_t stage_index = 0;
       stage_index < artifact.stages.size(); ++stage_index) {
    const auto& stage = artifact.stages[stage_index];
    if (stage.id != stage_index || stage.variants.empty()) {
      throw std::runtime_error(
          "parsed mthreads stage is incomplete");
    }
    for (const auto& variant : stage.variants) {
      const bool matmul =
          variant.function_name == "matmul_strided_kernel" ||
          variant.function_name == "matmul_descriptor_kernel" ||
          variant.function_name == "matmul_tle_kernel";
      const bool convolution_backward =
          variant.function_name == "conv_dgrad_nd_kernel" ||
          variant.function_name == "conv_wgrad_nd_kernel";
      const bool fused_convolution =
          variant.function_name == "conv_bias_relu_2d_kernel";
      std::size_t expected_arguments = artifact.binding_uids.size() +
          ((matmul || convolution_backward || fused_convolution) ? 0U : 1U);
      if (variant.function_name == "_conv_fprop2d_im2col_kernel") {
        expected_arguments = 2;
      } else if (
          variant.function_name == "_conv_fprop2d_im2col_mm_kernel") {
        expected_arguments = 3;
      } else if (variant.function_name ==
          "_conv_dgrad2d_dense_pack_filter_kernel") {
        expected_arguments = 2;
      } else if (variant.function_name ==
                 "_conv_dgrad2d_dense_pack_loss_kernel") {
        expected_arguments = 2;
      } else if (variant.function_name ==
                 "_conv_dgrad2d_dense_mm_kernel") {
        expected_arguments = 3;
      } else if (variant.function_name ==
                 "_conv_wgrad2d_p5_pack_image_kernel") {
        expected_arguments = 2;
      } else if (variant.function_name == "_conv_wgrad2d_p5_mm_kernel") {
        expected_arguments = 3;
      } else if (variant.function_name ==
                 "_conv_wgrad2d_im2row_kernel") {
        expected_arguments = 2;
      } else if (variant.function_name ==
                 "_conv_wgrad_nd_im2row_kernel") {
        expected_arguments = 2;
      } else if (variant.function_name == "_conv_wgrad2d_rowmajor_kernel") {
        expected_arguments = 3;
      } else if (variant.function_name ==
                 "_conv_wgrad_nd_rowmajor_kernel") {
        expected_arguments = 3;
      } else if (variant.function_name ==
                 "_conv_wgrad_nd_reduce_kernel") {
        expected_arguments = 2;
      } else if (
          variant.function_name == "_conv_wgrad2d_direct_split_kernel" ||
          variant.function_name == "_conv_wgrad2d_1x1_split_kernel") {
        expected_arguments = 3;
      } else if (variant.function_name ==
                 "_conv_wgrad2d_stem_split_kernel") {
        expected_arguments = 3;
      } else if (variant.function_name ==
                 "_conv_wgrad2d_stem_reduce_kernel") {
        expected_arguments = 2;
      } else if (variant.function_name == "layer_norm_kernel") {
        expected_arguments = 7;
      } else if (variant.function_name == "rms_norm_kernel") {
        expected_arguments = 6;
      } else if (variant.function_name == "batch_norm_nchw_kernel") {
        expected_arguments = 10;
      } else if (variant.function_name == "batch_norm_kernel") {
        expected_arguments = 13;
      } else if (
          variant.function_name == "batch_norm_inference_nchw_kernel") {
        expected_arguments = 6;
      } else if (
          variant.function_name == "batch_norm_inference_kernel") {
        expected_arguments = 9;
      } else if (variant.function_name == "_sdpa_fwd_kernel") {
        expected_arguments = 37;
      } else if (variant.function_name == "_zero_contiguous_kernel") {
        expected_arguments = 2;
      } else if (
          variant.function_name == "_sdpa_bwd_dq_dbias_kernel") {
        expected_arguments = 15;
      } else if (variant.function_name == "_sdpa_bwd_dkdv_kernel") {
        expected_arguments = 14;
      } else if (variant.function_name == "_sdpa_bwd_dk_kernel") {
        expected_arguments = 13;
      } else if (variant.function_name == "_sdpa_bwd_dv_kernel") {
        expected_arguments = 11;
      } else if (
          variant.function_name == "_zero_sdpa_fp8_fwd_amax_kernel") {
        expected_arguments = 2;
      } else if (variant.function_name == "_sdpa_fp8_fwd_kernel") {
        expected_arguments = 45;
      } else if (
          variant.function_name == "_zero_sdpa_fp8_bwd_amax_kernel") {
        expected_arguments = 4;
      } else if (
          variant.function_name == "_sdpa_fp8_bwd_dq_kernel") {
        expected_arguments = 22;
      } else if (
          variant.function_name == "_sdpa_fp8_bwd_dkdv_kernel") {
        expected_arguments = 27;
      }
      if (variant.arguments.size() != expected_arguments ||
          variant.source.empty() || variant.function_name.empty() ||
          variant.full_signature.empty()) {
        throw std::runtime_error(
            "parsed mthreads kernel variant is incomplete");
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 5) {
      throw std::invalid_argument(
          "usage: artifact_contract "
          "<request> <artifact-directory> <target> <valid|invalid>");
    }
    const std::filesystem::path request_path(argv[1]);
    const std::filesystem::path artifact_directory(argv[2]);
    const std::string target(argv[3]);
    const std::string expectation(argv[4]);
    if (expectation != "valid" && expectation != "invalid") {
      throw std::invalid_argument("expectation must be valid or invalid");
    }

    const std::string request = read_file(request_path);
    const std::string request_hash = flagdnn::native::sha256(request);
    flagdnn::mthreads::EngineBuildContext context;
    context.target_fingerprint = target;
    flagdnnBackendBuildInputV2 input = {
        sizeof(flagdnnBackendBuildInputV2),
        request.data(),
        request.size(),
        artifact_directory.c_str(),
        request_hash.c_str()};

    if (expectation == "invalid") {
      expect_compilation_failure(
          [&] {
            static_cast<void>(
                flagdnn::mthreads::parse_mthreads_artifact(context, input));
          },
          artifact_directory.string());
      return 0;
    }

    validate_result(
        flagdnn::mthreads::parse_mthreads_artifact(context, input));

    flagdnnBackendBuildInputV2 malformed = input;
    malformed.struct_size =
        static_cast<std::uint32_t>(sizeof(input) - 1);
    expect_compilation_failure(
        [&] {
          static_cast<void>(
              flagdnn::mthreads::parse_mthreads_artifact(
                  context, malformed));
        },
        "short build-input structure");
    malformed = input;
    malformed.graph_ir = nullptr;
    expect_compilation_failure(
        [&] {
          static_cast<void>(
              flagdnn::mthreads::parse_mthreads_artifact(
                  context, malformed));
        },
        "null Graph IR");
    malformed = input;
    malformed.graph_ir_size = 0;
    expect_compilation_failure(
        [&] {
          static_cast<void>(
              flagdnn::mthreads::parse_mthreads_artifact(
                  context, malformed));
        },
        "empty Graph IR");
    malformed = input;
    const std::string wrong_hash(64, '0');
    malformed.request_sha256 = wrong_hash.c_str();
    expect_compilation_failure(
        [&] {
          static_cast<void>(
              flagdnn::mthreads::parse_mthreads_artifact(
                  context, malformed));
        },
        "wrong request hash");
    malformed = input;
    malformed.artifact_directory = nullptr;
    expect_compilation_failure(
        [&] {
          static_cast<void>(
              flagdnn::mthreads::parse_mthreads_artifact(
                  context, malformed));
        },
        "null artifact directory");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
