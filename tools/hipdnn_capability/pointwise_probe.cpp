/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
// Diagnostic only: raw hipDNN calls intentionally bypass capability allowlists.
// Run ONE exact case per process to isolate unsupported API/context failures.
#include "backends/hygon/validation/activation_layout.hpp"
#include "pointwise_probe_data.hpp"

#include <hip/hip_runtime.h>
#include <hipdnn.h>

#include <iomanip>
#include <iostream>
#include <sstream>

namespace pp = flagdnn::tools::pointwise_probe;
namespace hv = flagdnn::validation::hygon;

namespace {
std::string quote(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\')
      out << '\\' << c;
    else if (c < 0x20)
      out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
          << unsigned(c) << std::dec;
    else
      out << c;
  }
  out << '"';
  return out.str();
}
template <class T> std::string array(const std::vector<T> &values) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i)
      out << ',';
    out << values[i];
  }
  out << ']';
  return out.str();
}
template <class Tensor> std::string tensors(const std::vector<Tensor> &values) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i)
      out << ',';
    const auto &t = values[i];
    out << "{\"uid\":" << t.uid
        << ",\"dtype\":" << quote(pp::dtype(t.data_type))
        << ",\"dimensions\":" << array(t.dimensions)
        << ",\"strides\":" << array(t.strides)
        << ",\"binding_byte_offset\":" << t.binding_byte_offset << '}';
  }
  out << ']';
  return out.str();
}
std::string metadata(const pp::Case &value) {
  const auto &spec = value.specification;
  const auto &attrs = spec.pointwise_attributes;
  std::ostringstream out;
  out << "\"case\":" << quote(spec.name)
      << ",\"operation\":" << quote(value.operation)
      << ",\"catalog\":" << quote(value.catalog)
      << ",\"tensors\":" << tensors(spec.tensors)
      << ",\"absolute_tolerance\":" << spec.absolute_tolerance
      << ",\"relative_tolerance\":" << spec.relative_tolerance
      << ",\"alpha\":" << spec.add_alpha
      << ",\"pointwise_attributes\":{\"flags\":" << attrs.flags
      << ",\"elu_alpha\":" << attrs.elu_alpha
      << ",\"swish_beta\":" << attrs.swish_beta << '}';
  return out.str();
}

struct ApiError : std::runtime_error {
  std::string library, api;
  int code;
  bool unsupported;
  ApiError(std::string library_value, std::string api_value, int code_value,
           bool unsupported_value, const char *message)
      : std::runtime_error(message), library(std::move(library_value)),
        api(std::move(api_value)), code(code_value),
        unsupported(unsupported_value) {}
};
void hip(hipError_t status, const char *api) {
  if (status != hipSuccess)
    throw ApiError("HIP", api, status, false, hipGetErrorString(status));
}
void dnn(hipdnnStatus_t status, const char *api) {
  if (status != HIPDNN_STATUS_SUCCESS)
    throw ApiError("hipDNN", api, status, status == HIPDNN_STATUS_NOT_SUPPORTED,
                   hipdnnGetErrorString(status));
}
hipdnnDataType_t dtype(flagdnnDataType_t type) {
  switch (type) {
  case FLAGDNN_DATA_FLOAT32:
    return HIPDNN_DATA_FLOAT;
  case FLAGDNN_DATA_FLOAT16:
    return HIPDNN_DATA_HALF;
  case FLAGDNN_DATA_BFLOAT16:
    return HIPDNN_DATA_BFLOAT16;
  case FLAGDNN_DATA_INT32:
    return HIPDNN_DATA_INT32;
  case FLAGDNN_DATA_BOOLEAN:
    return HIPDNN_DATA_BOOLEAN;
  case FLAGDNN_DATA_FP8_E4M3:
    return HIPDNN_DATA_FP8_E4M3;
  case FLAGDNN_DATA_FP8_E5M2:
    return HIPDNN_DATA_FP8_E5M2;
  default:
    throw std::invalid_argument("no hipDNN enum for requested dtype");
  }
}

class Probe {
public:
  pp::Case value;
  pp::Data data;
  std::string stage = "host_prepare", path, identity_api;
  int device;
  explicit Probe(pp::Case selected, int device_id, std::string selected_path,
                 std::string selected_identity_api)
      : value(std::move(selected)), data(pp::prepare(value)),
        path(std::move(selected_path)),
        identity_api(std::move(selected_identity_api)), device(device_id) {}
  ~Probe() {
    if (executable)
      (void)hipGraphExecDestroy(executable);
    if (graph)
      (void)hipGraphDestroy(graph);
    if (activation)
      (void)hipdnnDestroyActivationDescriptor(activation);
    if (first_op)
      (void)hipdnnDestroyOpTensorDescriptor(first_op);
    if (second_op)
      (void)hipdnnDestroyOpTensorDescriptor(second_op);
    for (auto tensor : descriptors)
      if (tensor)
        (void)hipdnnDestroyTensorDescriptor(tensor);
    if (handle)
      (void)hipdnnDestroy(handle);
    for (auto buffer : buffers)
      if (buffer)
        (void)hipFree(buffer);
    if (workspace)
      (void)hipFree(workspace);
    if (start)
      (void)hipEventDestroy(start);
    if (stop)
      (void)hipEventDestroy(stop);
    if (stream)
      (void)hipStreamDestroy(stream);
  }
  void initialize() {
    stage = "device_initialize";
    hip(hipSetDevice(device), "hipSetDevice");
    hipDeviceProp_t props{};
    hip(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");
    int runtime = 0, driver = 0;
    hip(hipRuntimeGetVersion(&runtime), "hipRuntimeGetVersion");
    hip(hipDriverGetVersion(&driver), "hipDriverGetVersion");
    std::cout << "{\"record\":\"environment\",\"device\":" << device
              << ",\"name\":" << quote(props.name)
              << ",\"architecture\":" << quote(props.gcnArchName)
              << ",\"hip_runtime\":" << runtime << ",\"hip_driver\":" << driver
              << ",\"hipdnn_runtime\":" << hipdnnGetVersion()
              << ",\"hipdnn_header\":" << HIPDNN_VERSION << '}' << std::endl;
    hip(hipStreamCreate(&stream), "hipStreamCreate");
    dnn(hipdnnCreate(&handle), "hipdnnCreate");
    dnn(hipdnnSetStream(handle, reinterpret_cast<hipdnnStream_t>(stream)),
        "hipdnnSetStream");
    const auto &name = value.operation;
    binary = name == "add" || name == "sub" || name == "mul" ||
             name == "scale" || name == "max" || name == "min" ||
             name == "add_square";
    backward = name.ends_with("_backward");
    for (const auto &t : value.specification.tensors)
      descriptor_tensors.push_back(
          {t.uid, t.data_type, t.dimensions, t.strides, t.binding_byte_offset});
    if (!binary)
      descriptor_tensors =
          hv::hipdnn_activation_descriptor_tensors(descriptor_tensors);
    const auto rank = descriptor_tensors.back().dimensions.size();
    stage = "tensor_descriptor";
    descriptors.resize(descriptor_tensors.size(), nullptr);
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
      auto &t = descriptor_tensors[i];
      while (t.dimensions.size() < rank) {
        const auto leading = pp::io::storage_element_count(t);
        t.dimensions.insert(t.dimensions.begin(), 1);
        t.strides.insert(t.strides.begin(), static_cast<std::int64_t>(leading));
      }
      std::vector<int> dimensions(t.dimensions.begin(), t.dimensions.end());
      std::vector<int> strides(t.strides.begin(), t.strides.end());
      dnn(hipdnnCreateTensorDescriptor(&descriptors[i]),
          "hipdnnCreateTensorDescriptor");
      dnn(hipdnnSetTensorNdDescriptor(descriptors[i], dtype(t.data_type),
                                      static_cast<int>(rank), dimensions.data(),
                                      strides.data()),
          "hipdnnSetTensorNdDescriptor");
    }
    std::cout << "{\"record\":\"descriptors\"," << metadata(value)
              << ",\"descriptor_tensors\":" << tensors(descriptor_tensors)
              << '}' << std::endl;
    stage = "operation_descriptor";
    if (binary) {
      const auto mode =
          name == "max"   ? HIPDNN_OP_TENSOR_MAX
          : name == "min" ? HIPDNN_OP_TENSOR_MIN
          : name == "mul" || name == "scale" || name == "add_square"
              ? HIPDNN_OP_TENSOR_MUL
              : HIPDNN_OP_TENSOR_ADD;
      dnn(hipdnnCreateOpTensorDescriptor(&first_op),
          "hipdnnCreateOpTensorDescriptor");
      dnn(hipdnnSetOpTensorDescriptor(first_op, mode, HIPDNN_DATA_FLOAT,
                                      HIPDNN_PROPAGATE_NAN),
          "hipdnnSetOpTensorDescriptor");
      if (name == "add_square") {
        dnn(hipdnnCreateOpTensorDescriptor(&second_op),
            "hipdnnCreateOpTensorDescriptor(add)");
        dnn(hipdnnSetOpTensorDescriptor(second_op, HIPDNN_OP_TENSOR_ADD,
                                        HIPDNN_DATA_FLOAT,
                                        HIPDNN_PROPAGATE_NAN),
            "hipdnnSetOpTensorDescriptor(add)");
      }
    } else if (value.operation != "identity" || identity_api != "transform") {
      const auto base = backward ? name.substr(0, name.size() - 9) : name;
      const std::map<std::string, hipdnnActivationMode_t> modes = {
          {"identity", HIPDNN_ACTIVATION_IDENTITY},
          {"abs", HIPDNN_ACTIVATION_ABS},
          {"relu", HIPDNN_ACTIVATION_RELU},
          {"sigmoid", HIPDNN_ACTIVATION_SIGMOID},
          {"tanh", HIPDNN_ACTIVATION_TANH},
          {"elu", HIPDNN_ACTIVATION_ELU},
          {"swish", HIPDNN_ACTIVATION_SWISH}};
      dnn(hipdnnCreateActivationDescriptor(&activation),
          "hipdnnCreateActivationDescriptor");
      dnn(hipdnnSetActivationDescriptor(
              activation, modes.at(base), HIPDNN_PROPAGATE_NAN,
              base == "elu" ? value.specification.pointwise_attributes.elu_alpha
                            : 0),
          "hipdnnSetActivationDescriptor");
      if (base == "swish")
        dnn(hipdnnSetActivationDescriptorSwishBeta(
                activation,
                value.specification.pointwise_attributes.swish_beta),
            "hipdnnSetActivationDescriptorSwishBeta");
    }
    stage = "allocate_upload";
    buffers.resize(descriptors.size(), nullptr);
    for (std::size_t i = 0; i < buffers.size(); ++i) {
      const auto &t = value.specification.tensors[i];
      const auto bytes = pp::io::storage_element_count(t) *
                         pp::io::data_type_size(t.data_type);
      hip(hipMalloc(&buffers[i], t.binding_byte_offset + bytes + 64),
          "hipMalloc");
      hip(hipMemsetAsync(buffers[i], 0xA5, t.binding_byte_offset + bytes + 64,
                         stream),
          "hipMemsetAsync");
      if (i < data.inputs.size())
        hip(hipMemcpyAsync(pointer(i), data.inputs[i].data(), bytes,
                           hipMemcpyHostToDevice, stream),
            "hipMemcpyAsync(input)");
    }
    hip(hipMalloc(&workspace, data.poison_storage.size()),
        "hipMalloc(workspace)");
    hip(hipEventCreate(&start), "hipEventCreate(start)");
    hip(hipEventCreate(&stop), "hipEventCreate(stop)");
    seed();
  }

  bool run() {
    bool passed = true;
    // Direct checks are independent evidence, not fallback from a Graph error.
    if (path == "direct" || path == "both")
      passed = run_path(false) && passed;
    if (path == "graph" || path == "both")
      passed = run_path(true) && passed;
    return passed;
  }

private:
  hipStream_t stream = nullptr;
  hipdnnHandle_t handle = nullptr;
  hipdnnOpTensorDescriptor_t first_op = nullptr, second_op = nullptr;
  hipdnnActivationDescriptor_t activation = nullptr;
  std::vector<hipdnnTensorDescriptor_t> descriptors;
  std::vector<hv::ReferenceTensor> descriptor_tensors;
  std::vector<void *> buffers;
  void *workspace = nullptr;
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
  hipEvent_t start = nullptr, stop = nullptr;
  bool binary = false, backward = false;
  void *pointer(std::size_t i) {
    return static_cast<std::uint8_t *>(buffers.at(i)) +
           value.specification.tensors.at(i).binding_byte_offset;
  }
  void seed() {
    hip(hipMemcpyAsync(pointer(buffers.size() - 1), data.poison_storage.data(),
                       data.poison_storage.size(), hipMemcpyHostToDevice,
                       stream),
        "hipMemcpyAsync(finite_poison)");
    hip(hipMemsetAsync(workspace, 0, data.poison_storage.size(), stream),
        "hipMemsetAsync(workspace)");
    hip(hipStreamSynchronize(stream), "hipStreamSynchronize(seed)");
  }
  void execute() {
    constexpr float one = 1, zero = 0;
    if (value.operation == "identity" && identity_api == "transform") {
      dnn(hipdnnTransformTensor(handle, &one, descriptors[0], pointer(0), &zero,
                                descriptors[1], pointer(1)),
          "hipdnnTransformTensor(identity)");
    } else if (backward) {
      dnn(hipdnnActivationForward(handle, activation, &one, descriptors[1],
                                  pointer(1), &zero, descriptors[2], workspace),
          "hipdnnActivationForward(backward_y)");
      dnn(hipdnnActivationBackward(handle, activation, &one, descriptors[2],
                                   workspace, descriptors[0], pointer(0),
                                   descriptors[1], pointer(1), &zero,
                                   descriptors[2], pointer(2)),
          "hipdnnActivationBackward");
    } else if (!binary) {
      const float alpha = value.specification.add_alpha;
      dnn(hipdnnActivationForward(handle, activation, &alpha, descriptors[0],
                                  pointer(0), &zero, descriptors[1],
                                  pointer(1)),
          "hipdnnActivationForward");
    } else if (value.operation == "add_square") {
      dnn(hipdnnOpTensor(handle, first_op, &one, descriptors[1], pointer(1),
                         &one, descriptors[1], pointer(1), &zero,
                         descriptors[2], workspace),
          "hipdnnOpTensor(square)");
      dnn(hipdnnOpTensor(handle, second_op, &one, descriptors[0], pointer(0),
                         &one, descriptors[2], workspace, &zero, descriptors[2],
                         pointer(2)),
          "hipdnnOpTensor(add)");
    } else {
      const float alpha =
          value.operation == "add"   ? value.specification.add_alpha
          : value.operation == "sub" ? -value.specification.add_alpha
                                     : 1;
      dnn(hipdnnOpTensor(handle, first_op, &one, descriptors[0], pointer(0),
                         &alpha, descriptors[1], pointer(1), &zero,
                         descriptors[2], pointer(2)),
          "hipdnnOpTensor");
    }
  }
  bool validate(const std::string &mode) {
    hip(hipStreamSynchronize(stream), "hipStreamSynchronize(validate)");
    std::vector<std::uint8_t> actual(data.poison_storage.size());
    hip(hipMemcpy(actual.data(), pointer(buffers.size() - 1), actual.size(),
                  hipMemcpyDeviceToHost),
        "hipMemcpy(output)");
    auto result = pp::compare(value, data, actual);
    const auto &output = pp::bm::output_tensor(value.specification);
    std::vector<std::uint8_t> guards(output.binding_byte_offset +
                                     actual.size() + 64);
    hip(hipMemcpy(guards.data(), buffers.back(), guards.size(),
                  hipMemcpyDeviceToHost),
        "hipMemcpy(guards)");
    for (std::size_t i = 0; i < guards.size(); ++i)
      if ((i < output.binding_byte_offset ||
           i >= output.binding_byte_offset + actual.size()) &&
          guards[i] != 0xA5)
        ++result.padding_errors;
    std::cout << "{\"record\":\"validation\"," << metadata(value)
              << ",\"device\":" << device << ",\"path\":" << quote(mode)
              << ",\"stage\":" << quote(stage) << ",\"status\":"
              << quote(result.passed() ? "pass" : "wrong_result")
              << ",\"oracle\":"
              << quote(value.operation == "identity" ? "bit_exact" : "cpu_full")
              << ",\"logical_elements\":" << data.expected.size()
              << ",\"mismatches\":" << result.mismatches
              << ",\"unchanged\":" << result.unchanged
              << ",\"nonfinite\":" << result.nonfinite
              << ",\"padding_errors\":" << result.padding_errors
              << ",\"first_mismatch\":"
              << (result.mismatches ? std::to_string(result.first) : "null")
              << ",\"max_absolute\":" << result.max_absolute
              << ",\"max_relative\":" << result.max_relative << '}'
              << std::endl;
    return result.passed();
  }
  bool run_path(bool captured) {
    const std::string mode = captured ? "graph" : "direct";
    // The non-timed warmup populates vendor caches before stream capture.
    stage = mode + "_warmup";
    seed();
    execute();
    hip(hipStreamSynchronize(stream), "hipStreamSynchronize(warmup)");
    if (captured) {
      stage = "graph_capture";
      seed();
      hip(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed),
          "hipStreamBeginCapture");
      try {
        execute();
      } catch (...) {
        (void)hipStreamEndCapture(stream, &graph);
        throw;
      }
      hip(hipStreamEndCapture(stream, &graph), "hipStreamEndCapture");
      hip(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
          "hipGraphInstantiate");
    }
    const auto launch = [&] {
      if (captured)
        hip(hipGraphLaunch(executable, stream), "hipGraphLaunch");
      else
        execute();
    };
    stage = mode + "_fresh_before_timing";
    seed();
    launch();
    const bool before = validate(mode);
    stage = mode + "_timing_smoke";
    hip(hipEventRecord(start, stream), "hipEventRecord(start)");
    for (int i = 0; i < 3; ++i)
      launch();
    hip(hipEventRecord(stop, stream), "hipEventRecord(stop)");
    hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
    float milliseconds = 0;
    hip(hipEventElapsedTime(&milliseconds, start, stop), "hipEventElapsedTime");
    if (!std::isfinite(milliseconds) || milliseconds < 0)
      throw std::runtime_error("invalid timing smoke event duration");
    std::cout << "{\"record\":\"timing_smoke\",\"case\":"
              << quote(value.specification.name) << ",\"path\":" << quote(mode)
              << ",\"iterations\":3,\"event_ms\":" << milliseconds
              << ",\"performance_baseline_eligible\":false}" << std::endl;
    stage = mode + "_fresh_after_timing";
    seed();
    launch();
    return validate(mode) && before;
  }
};
} // namespace

int main(int argc, char **argv) {
  std::string selected, path = "both", identity_api = "activation";
  int device = 0;
  bool list = false;
  std::unique_ptr<Probe> probe;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--list")
        list = true;
      else if (arg == "--case" && i + 1 < argc)
        selected = argv[++i];
      else if (arg == "--device" && i + 1 < argc)
        device = std::stoi(argv[++i]);
      else if (arg == "--path" && i + 1 < argc)
        path = argv[++i];
      else if (arg == "--identity-api" && i + 1 < argc)
        identity_api = argv[++i];
      else
        throw std::invalid_argument(
            "usage: pointwise_probe --list | --case NAME [--device N] [--path "
            "direct|graph|both] [--identity-api activation|transform]");
    }
    if (path != "direct" && path != "graph" && path != "both")
      throw std::invalid_argument("invalid path");
    if (identity_api != "activation" && identity_api != "transform")
      throw std::invalid_argument("invalid identity API");
    const auto cases = pp::catalog();
    if (list) {
      for (const auto &value : cases)
        std::cout << "{\"record\":\"catalog\"," << metadata(value) << "}\n";
      return 0;
    }
    const auto found =
        std::find_if(cases.begin(), cases.end(), [&](const auto &value) {
          return value.specification.name == selected;
        });
    if (found == cases.end())
      throw std::invalid_argument("exact case name not found");
    if (identity_api == "transform" && found->operation != "identity")
      throw std::invalid_argument("TransformTensor probe is identity-only");
    std::cout << "{\"record\":\"selected\"," << metadata(*found) << '}'
              << std::endl;
    std::cout << "{\"record\":\"api_route\",\"identity_api\":"
              << quote(identity_api) << '}' << std::endl;
    probe = std::make_unique<Probe>(*found, device, path, identity_api);
    probe->initialize();
    const bool passed = probe->run();
    std::cout << "{\"record\":\"result\",\"case\":" << quote(selected)
              << ",\"path\":" << quote(path)
              << ",\"status\":" << quote(passed ? "pass" : "wrong_result")
              << '}' << std::endl;
    return passed ? 0 : 1;
  } catch (const ApiError &error) {
    std::cout << "{\"record\":\"result\",\"case\":" << quote(selected)
              << ",\"path\":" << quote(path)
              << ",\"stage\":" << quote(probe ? probe->stage : "host_prepare")
              << ",\"status\":"
              << quote(error.unsupported ? "unsupported" : "api_error")
              << ",\"library\":" << quote(error.library)
              << ",\"api\":" << quote(error.api) << ",\"code\":" << error.code
              << ",\"error\":" << quote(error.what()) << '}' << std::endl;
    return error.unsupported ? 3 : 1;
  } catch (const std::exception &error) {
    std::cout << "{\"record\":\"result\",\"case\":" << quote(selected)
              << ",\"stage\":" << quote(probe ? probe->stage : "host_prepare")
              << ",\"status\":\"probe_error\",\"error\":" << quote(error.what())
              << '}' << std::endl;
    return 1;
  }
}
