/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
// Functional capability probe only: no performance timing is collected.
// --list is CPU-only. Run each --case in a fresh process to isolate HIP errors.
// Exit codes: 0 pass, 77 explicit hipDNN NOT_SUPPORTED, 2 wrong result/guard
// corruption, 3 Graph API incompatibility after direct passed, 1 other errors.
#include "common/reduction.hpp"
#include "hipdnn_reference.hpp"
#include "reference/cpu/matrix.hpp"
#include "tensor_io.hpp"

#include <hip/hip_runtime_api.h>
#include <hipdnn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace hv = flagdnn::validation::hygon;
namespace io = hv::tensor_io;
namespace cpu = flagdnn::reference::cpu;
using Case = flagdnn::testing::ReductionTestCase;
using Tensor = flagdnn::testing::TestTensor;

std::string quote(std::string_view value) {
  std::string result = "\"";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      result += '\\';
      result += static_cast<char>(c);
    } else if (c == '\n') result += "\\n";
    else if (c == '\r') result += "\\r";
    else if (c == '\t') result += "\\t";
    else if (c < 32) result += '?';
    else result += static_cast<char>(c);
  }
  return result + '"';
}

std::string_view dtype(flagdnnDataType_t type) {
  if (type == FLAGDNN_DATA_FLOAT32) return "fp32";
  if (type == FLAGDNN_DATA_FLOAT16) return "fp16";
  if (type == FLAGDNN_DATA_BFLOAT16) return "bfloat16";
  throw std::invalid_argument("unexpected probe dtype");
}

hipdnnDataType_t vendor_dtype(flagdnnDataType_t type) {
  if (type == FLAGDNN_DATA_FLOAT32) return HIPDNN_DATA_FLOAT;
  if (type == FLAGDNN_DATA_FLOAT16) return HIPDNN_DATA_HALF;
  if (type == FLAGDNN_DATA_BFLOAT16) return HIPDNN_DATA_BFLOAT16;
  throw std::invalid_argument("unexpected probe dtype");
}

hipdnnReduceTensorOp_t vendor_mode(flagdnnReductionMode_t mode) {
  if (mode == FLAGDNN_REDUCTION_ADD) return HIPDNN_REDUCE_TENSOR_ADD;
  if (mode == FLAGDNN_REDUCTION_AVG) return HIPDNN_REDUCE_TENSOR_AVG;
  if (mode == FLAGDNN_REDUCTION_MUL) return HIPDNN_REDUCE_TENSOR_MUL;
  throw std::invalid_argument("unexpected reduction mode");
}

void hip_check(hipError_t status, std::string_view operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(status));
}

class WrongResult final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

std::vector<Case> catalog() {
  std::vector<Case> result;
  int fp32 = 0, fp16 = 0, bf16 = 0;
  for (const auto &value : flagdnn::testing::make_reduction_cases()) {
    if (value.name.find("_to_fp32_") == std::string::npos ||
        value.output.data_type != FLAGDNN_DATA_FLOAT32)
      continue;
    const auto type = value.input.data_type;
    if (type == FLAGDNN_DATA_BFLOAT16) {
      if (value.input.dimensions.size() != 2 ||
          value.mode != FLAGDNN_REDUCTION_ADD)
        continue;
      ++bf16;
    } else if (type == FLAGDNN_DATA_FLOAT16) ++fp16;
    else if (type == FLAGDNN_DATA_FLOAT32) ++fp32;
    else continue;
    flagdnn::testing::validate_reduction_case(value);
    if (value.input.binding_byte_offset != 16 ||
        value.output.binding_byte_offset != 32)
      throw std::runtime_error("shared reduction offsets changed");
    result.push_back(value);
  }
  if (fp32 != 36 || fp16 != 36 || bf16 != 14)
    throw std::runtime_error("expected 36 FP32 controls + 50 mixed-dtype cases");
  return result;
}

void array(std::span<const std::int64_t> values) {
  std::cout << '[';
  for (std::size_t i = 0; i < values.size(); ++i)
    std::cout << (i ? "," : "") << values[i];
  std::cout << ']';
}

struct Descriptors { Tensor input, output; };
Descriptors descriptors(const Case &value, bool rank2_boundary_view);

void metadata(const Case &value, bool rank2_boundary_view = false) {
  std::cout << "{\"case_name\":" << quote(value.name)
            << ",\"input_dtype\":" << quote(dtype(value.input.data_type))
            << ",\"output_dtype\":\"fp32\",\"compute_dtype\":\"fp32\""
            << ",\"control\":"
            << (value.input.data_type == FLAGDNN_DATA_FLOAT32 ? "true" : "false")
            << ",\"axis\":" << value.axis << ",\"keep_dimensions\":"
            << (value.keep_dimensions ? "true" : "false")
            << ",\"input_shape\":";
  array(value.input.dimensions);
  std::cout << ",\"input_strides\":";
  array(value.input.strides);
  std::cout << ",\"output_shape\":";
  array(value.output.dimensions);
  std::cout << ",\"output_strides\":";
  array(value.output.strides);
  std::cout << ",\"input_binding_byte_offset\":"
            << value.input.binding_byte_offset
            << ",\"output_binding_byte_offset\":"
            << value.output.binding_byte_offset;
  const auto actual = descriptors(value, rank2_boundary_view);
  std::cout << ",\"descriptor_view\":"
            << quote(rank2_boundary_view ? "rank2_boundary_no_copy" : "logical_rank")
            << ",\"actual_input_descriptor_shape\":";
  array(actual.input.dimensions);
  std::cout << ",\"actual_input_descriptor_strides\":";
  array(actual.input.strides);
  std::cout << ",\"actual_output_descriptor_shape\":";
  array(actual.output.dimensions);
  std::cout << ",\"actual_output_descriptor_strides\":";
  array(actual.output.strides);
}

struct Buffer {
  void *data = nullptr;
  std::size_t size;
  explicit Buffer(std::size_t bytes) : size(bytes) {
    hip_check(hipMalloc(&data, size), "hipMalloc");
  }
  ~Buffer() { if (data) (void)hipFree(data); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void *at(std::size_t offset) {
    return static_cast<std::uint8_t *>(data) + offset;
  }
  void upload(std::span<const std::uint8_t> bytes) {
    if (bytes.size() != size) throw std::logic_error("buffer upload size");
    hip_check(hipMemcpy(data, bytes.data(), size, hipMemcpyHostToDevice),
              "hipMemcpy H2D");
  }
  std::vector<std::uint8_t> download() const {
    std::vector<std::uint8_t> bytes(size);
    hip_check(hipMemcpy(bytes.data(), data, size, hipMemcpyDeviceToHost),
              "hipMemcpy D2H");
    return bytes;
  }
};

struct Context {
  hipStream_t stream = nullptr;
  hipdnnHandle_t handle = nullptr;
  hipdnnTensorDescriptor_t input = nullptr, output = nullptr;
  hipdnnReduceTensorDescriptor_t reduction = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
  bool capturing = false;
  ~Context() {
    if (capturing) {
      hipGraph_t abandoned = nullptr;
      (void)hipStreamEndCapture(stream, &abandoned);
      if (abandoned) (void)hipGraphDestroy(abandoned);
    }
    if (executable) (void)hipGraphExecDestroy(executable);
    if (graph) (void)hipGraphDestroy(graph);
    if (reduction) (void)hipdnnDestroyReduceTensorDescriptor(reduction);
    if (output) (void)hipdnnDestroyTensorDescriptor(output);
    if (input) (void)hipdnnDestroyTensorDescriptor(input);
    if (handle) (void)hipdnnDestroy(handle);
    if (stream) (void)hipStreamDestroy(stream);
  }
};

void set_descriptor(hipdnnTensorDescriptor_t descriptor, const Tensor &tensor) {
  const std::vector<int> dims(tensor.dimensions.begin(), tensor.dimensions.end());
  const std::vector<int> strides(tensor.strides.begin(), tensor.strides.end());
  hv::check_hipdnn_status(
      hipdnnSetTensorNdDescriptor(descriptor, vendor_dtype(tensor.data_type),
                                 static_cast<int>(dims.size()), dims.data(),
                                 strides.data()),
      "hipdnnSetTensorNdDescriptor");
}

Tensor reduced_descriptor(const Case &value) {
  Tensor result = value.output;
  if (!value.keep_dimensions) {
    const auto axis = static_cast<std::size_t>(value.axis);
    const std::int64_t stride = axis < result.dimensions.size()
        ? result.dimensions[axis] * result.strides[axis] : 1;
    result.dimensions.insert(result.dimensions.begin() + value.axis, 1);
    result.strides.insert(result.strides.begin() + value.axis, stride);
  }
  return result;
}

bool dense(const Tensor &tensor) {
  std::int64_t stride = 1;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    if (tensor.strides[axis - 1] != stride) return false;
    stride *= tensor.dimensions[axis - 1];
  }
  return true;
}

Descriptors descriptors(const Case &value, bool rank2_boundary_view) {
  Descriptors result{value.input, reduced_descriptor(value)};
  if (!rank2_boundary_view) return result;
  const auto rank = value.input.dimensions.size();
  if (rank < 3 || !dense(value.input) || !dense(value.output) ||
      (value.axis != 0 && value.axis != static_cast<std::int32_t>(rank - 1)))
    throw std::invalid_argument(
        "rank2 boundary view requires dense rank>=3 and first/last reduction axis");
  const auto reduced = value.input.dimensions[static_cast<std::size_t>(value.axis)];
  const auto remaining = static_cast<std::int64_t>(io::element_count(value.output));
  if (value.axis == 0) {
    result.input.dimensions = {reduced, remaining};
    result.input.strides = {remaining, 1};
    result.output.dimensions = {1, remaining};
    result.output.strides = {remaining, 1};
  } else {
    result.input.dimensions = {remaining, reduced};
    result.input.strides = {reduced, 1};
    result.output.dimensions = {remaining, 1};
    result.output.strides = {1, 1};
  }
  return result;
}

void descriptor_self_test(const std::vector<Case> &cases) {
  int accepted = 0, rejected_middle = 0, rejected_stride = 0;
  for (const auto &value : cases) {
    if (value.input.data_type != FLAGDNN_DATA_FLOAT16 ||
        value.input.dimensions.size() < 3)
      continue;
    if (value.axis != 0 &&
        value.axis != static_cast<std::int32_t>(value.input.dimensions.size() - 1)) {
      bool rejected = false;
      try { (void)descriptors(value, true); }
      catch (const std::invalid_argument &) { rejected = true; }
      if (!rejected) throw std::runtime_error("middle-axis view accepted");
      ++rejected_middle;
      continue;
    }
    const auto actual = descriptors(value, true);
    if (actual.input.uid != value.input.uid ||
        actual.output.uid != value.output.uid ||
        actual.input.data_type != value.input.data_type ||
        actual.output.data_type != value.output.data_type ||
        actual.input.binding_byte_offset != 16 ||
        actual.output.binding_byte_offset != 32 ||
        io::element_count(actual.input) != io::element_count(value.input) ||
        io::element_count(actual.output) != io::element_count(value.output))
      throw std::runtime_error("rank2 view changed storage contract");
    const auto raw = io::encode(
        flagdnn::testing::reduction_host_input(value), value.input.data_type);
    const auto input = io::decode(raw, value.input.data_type,
                                  io::element_count(value.input));
    const auto expected = cpu::evaluate_reduction(
        {value.input.dimensions, value.axis, value.mode}, input);
    const auto view_result = cpu::evaluate_reduction(
        {actual.input.dimensions, value.axis == 0 ? 0 : 1, value.mode}, input);
    if (expected != view_result)
      throw std::runtime_error("rank2 view changed logical reduction result");
    ++accepted;
    auto padded = value;
    padded.input.strides[0] += 1;
    bool rejected = false;
    try { (void)descriptors(padded, true); }
    catch (const std::invalid_argument &) { rejected = true; }
    if (!rejected) throw std::runtime_error("padded view accepted");
    ++rejected_stride;
  }
  if (accepted != 9 || rejected_middle != 6 || rejected_stride != 9)
    throw std::runtime_error("unexpected rank2 view contract case counts");
  std::cout << "descriptor_self_test PASS equivalent=9 middle_rejected=6 "
               "padded_rejected=9; no GPU initialized\n";
}

float poison_value(float expected, int round) {
  const float magnitude =
      std::max(1024.0F + float(round), 4.0F * std::abs(expected) + 32.0F);
  const float result = expected >= 0 ? -magnitude : magnitude;
  if (!std::isfinite(result)) throw std::runtime_error("poison overflow");
  return result;
}

void validate(const Case &value, const Buffer &input, const Buffer &output,
              std::span<const std::uint8_t> initial_input,
              std::span<const float> reference, std::string_view phase,
              int round) {
  if (input.download() != std::vector<std::uint8_t>(initial_input.begin(),
                                                  initial_input.end()))
    throw WrongResult(std::string(phase) + ": input or input guards changed");
  const auto bytes = output.download();
  const auto offset = value.output.binding_byte_offset;
  const auto count = io::storage_element_count(value.output);
  const auto size = count * io::data_type_size(value.output.data_type);
  for (std::size_t i = 0; i < bytes.size(); ++i)
    if ((i < offset || i >= offset + size) && bytes[i] != 0xA5)
      throw WrongResult(std::string(phase) + ": output guard changed at byte " +
                        std::to_string(i));
  const auto physical = io::decode(
      std::span<const std::uint8_t>(bytes).subspan(offset, size),
      value.output.data_type, count);
  io::require_padding_unchanged("hipdnn", physical, value.output);
  const auto actual = io::gather(physical, value.output);
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double error = std::abs(double(actual[i]) - reference[i]);
    const double tolerance = value.absolute_tolerance +
        value.relative_tolerance * std::abs(double(reference[i]));
    if (!std::isfinite(actual[i]) || !std::isfinite(reference[i]) ||
        error > tolerance) {
      // Diagnostic only: never accept HALF writes to a declared FP32 output.
      // This distinguishes incorrect arithmetic from a vendor output ABI bug.
      const auto half = io::decode(
          std::span<const std::uint8_t>(bytes).subspan(offset, reference.size() * 2),
          FLAGDNN_DATA_FLOAT16, reference.size());
      const auto half_expected = io::decode(
          io::encode(reference, FLAGDNN_DATA_FLOAT16),
          FLAGDNN_DATA_FLOAT16, reference.size());
      std::size_t half_matches = 0, untouched_tail = 0;
      for (std::size_t j = 0; j < reference.size(); ++j)
        half_matches += half[j] == half_expected[j];
      const auto tail_start = (reference.size() + 1) / 2;
      for (std::size_t j = tail_start; j < actual.size(); ++j)
        untouched_tail += actual[j] == poison_value(reference[j], round);
      throw WrongResult(std::string(phase) + ": element " + std::to_string(i) +
                        " expected=" + std::to_string(reference[i]) +
                        " actual=" + std::to_string(actual[i]) +
                        "; half_matches_quantized_reference=" +
                        std::to_string(half_matches) + "/" +
                        std::to_string(reference.size()) +
                        "; untouched_fp32_tail=" +
                        std::to_string(untouched_tail) + "/" +
                        std::to_string(actual.size() - tail_start));
    }
  }
}

void run(const Case &value, int device, int iterations, bool rank2_boundary_view,
         std::string &phase, bool &direct_passed) {
  const auto actual = descriptors(value, rank2_boundary_view);
  phase = "initialization";
  hip_check(hipSetDevice(device), "hipSetDevice");
  Context context;
  hip_check(hipStreamCreateWithFlags(&context.stream, hipStreamNonBlocking),
            "hipStreamCreateWithFlags");
  hv::check_hipdnn_status(hipdnnCreate(&context.handle), "hipdnnCreate");
  hv::check_hipdnn_status(hipdnnSetStream(context.handle, context.stream),
                         "hipdnnSetStream");
  hv::check_hipdnn_status(hipdnnCreateTensorDescriptor(&context.input),
                         "hipdnnCreateTensorDescriptor(input)");
  hv::check_hipdnn_status(hipdnnCreateTensorDescriptor(&context.output),
                         "hipdnnCreateTensorDescriptor(output)");
  set_descriptor(context.input, actual.input);
  set_descriptor(context.output, actual.output);
  hv::check_hipdnn_status(hipdnnCreateReduceTensorDescriptor(&context.reduction),
                         "hipdnnCreateReduceTensorDescriptor");
  hv::check_hipdnn_status(
      hipdnnSetReduceTensorDescriptor(
          context.reduction, vendor_mode(value.mode), HIPDNN_DATA_FLOAT,
          HIPDNN_PROPAGATE_NAN, HIPDNN_REDUCE_TENSOR_NO_INDICES,
          HIPDNN_32BIT_INDICES), "hipdnnSetReduceTensorDescriptor");
  std::size_t workspace_size = 0;
  phase = "workspace";
  hv::check_hipdnn_status(
      hipdnnGetReductionWorkspaceSize(context.handle, context.reduction,
                                     context.input, context.output,
                                     &workspace_size),
      "hipdnnGetReductionWorkspaceSize");
  const auto input_bytes = io::encode(
      io::scatter(flagdnn::testing::reduction_host_input(value), value.input),
      value.input.data_type);
  const auto quantized = io::gather(
      io::decode(input_bytes, value.input.data_type,
                 io::storage_element_count(value.input)), value.input);
  const auto reference = cpu::evaluate_reduction(
      {value.input.dimensions, value.axis, value.mode}, quantized);
  std::vector<std::uint8_t> input_initial(
      value.input.binding_byte_offset + input_bytes.size() + 64, 0xA5);
  std::copy(input_bytes.begin(), input_bytes.end(),
            input_initial.begin() + value.input.binding_byte_offset);
  Buffer input(input_initial.size());
  input.upload(input_initial);
  const auto output_size = io::storage_element_count(value.output) * 4;
  Buffer output(value.output.binding_byte_offset + output_size + 64);
  Buffer workspace(std::max<std::size_t>(workspace_size, 1));
  auto reset = [&](int round) {
    std::vector<float> poison(reference.size());
    for (std::size_t i = 0; i < poison.size(); ++i)
      poison[i] = poison_value(reference[i], round);
    const auto encoded = io::encode(io::scatter(poison, value.output),
                                     value.output.data_type);
    std::vector<std::uint8_t> bytes(output.size, 0xA5);
    std::copy(encoded.begin(), encoded.end(),
              bytes.begin() + value.output.binding_byte_offset);
    output.upload(bytes);
  };
  auto launch = [&] {
    constexpr float alpha = 1.0F, beta = 0.0F;
    hv::check_hipdnn_status(
        hipdnnReduceTensor(context.handle, context.reduction, nullptr, 0,
                           workspace.data, workspace_size, &alpha,
                           context.input, input.at(value.input.binding_byte_offset),
                           &beta, context.output,
                           output.at(value.output.binding_byte_offset)),
        "hipdnnReduceTensor");
  };
  for (int round = 0; round < 2; ++round) {
    phase = round == 0 ? "direct" : "direct_fresh";
    reset(round);
    for (int iteration = 0; iteration < iterations; ++iteration) launch();
    hip_check(hipStreamSynchronize(context.stream), "direct synchronize");
    validate(value, input, output, input_initial, reference, phase, round);
  }
  direct_passed = true;
  phase = "graph_capture";
  hip_check(hipStreamBeginCapture(context.stream, hipStreamCaptureModeGlobal),
            "hipStreamBeginCapture");
  context.capturing = true;
  for (int iteration = 0; iteration < iterations; ++iteration) launch();
  const auto capture_status = hipStreamEndCapture(context.stream, &context.graph);
  context.capturing = false;
  hip_check(capture_status, "hipStreamEndCapture");
  hip_check(hipGraphInstantiate(&context.executable, context.graph, nullptr,
                               nullptr, 0), "hipGraphInstantiate");
  for (int round = 0; round < 2; ++round) {
    phase = round == 0 ? "graph" : "graph_fresh";
    reset(2 + round);
    hip_check(hipGraphLaunch(context.executable, context.stream), "hipGraphLaunch");
    hip_check(hipStreamSynchronize(context.stream), "graph synchronize");
    validate(value, input, output, input_initial, reference, phase, 2 + round);
  }
}
} // namespace

int main(int argc, char **argv) {
  std::string name;
  int device = 0;
  int iterations = 50;
  bool list = false, rank2_boundary_view = false, self_test = false;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string_view arg = argv[i];
      if (arg == "--list") list = true;
      else if (arg == "--rank2-boundary-view") rank2_boundary_view = true;
      else if (arg == "--descriptor-self-test") self_test = true;
      else if (arg == "--case" && i + 1 < argc) name = argv[++i];
      else if (arg == "--device" && i + 1 < argc) device = std::stoi(argv[++i]);
      else if (arg == "--iterations" && i + 1 < argc)
        iterations = std::stoi(argv[++i]);
      else throw std::invalid_argument(
          "usage: reduction_probe --list | --case NAME [--device N] "
          "[--iterations N (default 50)] [--rank2-boundary-view] | "
          "--descriptor-self-test");
    }
    if (iterations <= 0) throw std::invalid_argument("iterations must be positive");
    const auto cases = catalog();
    if (self_test) { descriptor_self_test(cases); return 0; }
    if (list) {
      for (const auto &value : cases) { metadata(value); std::cout << "}\n"; }
      return 0;
    }
    const auto found = std::find_if(cases.begin(), cases.end(),
                                   [&](const Case &value) { return value.name == name; });
    if (found == cases.end()) throw std::invalid_argument("--case did not match");
    (void)descriptors(*found, rank2_boundary_view); // Reject invalid views on CPU.
    std::string phase, status = "passed", error;
    bool direct_passed = false;
    int code = 0;
    try { run(*found, device, iterations, rank2_boundary_view, phase, direct_passed); }
    catch (const WrongResult &failure) {
      status = "wrong_result"; error = failure.what(); code = 2;
    } catch (const hv::HipdnnStatusError &failure) {
      error = failure.what();
      if (direct_passed) { status = "graph_incompatible"; code = 3; }
      else if (failure.status() == HIPDNN_STATUS_NOT_SUPPORTED) {
        status = "unsupported"; code = 77;
      } else { status = "api_error"; code = 1; }
    } catch (const std::exception &failure) {
      error = failure.what();
      status = direct_passed ? "graph_incompatible" : "api_error";
      code = direct_passed ? 3 : 1;
    }
    metadata(*found, rank2_boundary_view);
    std::cout << ",\"device\":" << device << ",\"iterations\":" << iterations
              << ",\"status\":" << quote(status)
              << ",\"phase\":" << quote(phase)
              << ",\"direct_passed\":" << (direct_passed ? "true" : "false")
              << ",\"error\":" << quote(error) << "}\n";
    return code;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
