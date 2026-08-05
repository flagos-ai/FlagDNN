#include <flagdnn/flagdnn.hpp>

#include <cuda.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void check_cuda(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* message = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &message);
  std::ostringstream output;
  output << operation << " failed";
  if (name != nullptr) {
    output << " (" << name << ")";
  }
  if (message != nullptr) {
    output << ": " << message;
  }
  throw std::runtime_error(output.str());
}

bool process_maps_contains(const std::string& needle) {
  std::ifstream maps("/proc/self/maps");
  std::string line;
  while (std::getline(maps, line)) {
    if (line.find(needle) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void require_native_process_clean(const char* stage) {
  const bool has_python = process_maps_contains("libpython");
  const bool has_torch = process_maps_contains("libtorch");
  std::cout << stage << "_has_libpython=" << std::boolalpha << has_python
            << " " << stage << "_has_libtorch=" << has_torch << '\n';
#if !defined(FLAGDNN_EXPECT_LIBTRITON_JIT)
  if (has_python || has_torch) {
    throw std::runtime_error(
        "native test process unexpectedly loaded libpython or libtorch");
  }
#endif
}

class TemporaryCache {
 public:
  TemporaryCache() {
    std::string pattern =
        (std::filesystem::temp_directory_path() /
         "flagdnn-native-integration-XXXXXX")
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  TemporaryCache(const TemporaryCache&) = delete;
  TemporaryCache& operator=(const TemporaryCache&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

  [[nodiscard]] std::size_t manifest_count() const {
    std::size_t result = 0;
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(path_)) {
      if (entry.is_regular_file() && entry.path().filename() == "manifest.json") {
        ++result;
      }
    }
    return result;
  }

 private:
  std::filesystem::path path_;
};

class DriverContext {
 public:
  DriverContext() {
    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuDeviceGet(&device_, 0), "cuDeviceGet");
    check_cuda(cuDevicePrimaryCtxRetain(&context_, device_),
               "cuDevicePrimaryCtxRetain");
    check_cuda(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
  }

  ~DriverContext() {
    if (context_ != nullptr) {
      (void)cuDevicePrimaryCtxRelease(device_);
    }
  }

  DriverContext(const DriverContext&) = delete;
  DriverContext& operator=(const DriverContext&) = delete;

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
};

class Stream {
 public:
  Stream() {
    check_cuda(cuStreamCreate(&value_, CU_STREAM_NON_BLOCKING),
               "cuStreamCreate");
  }

  ~Stream() {
    if (value_ != nullptr) {
      (void)cuStreamDestroy(value_);
    }
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  [[nodiscard]] CUstream get() const noexcept { return value_; }
  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(value_);
  }

  void synchronize() const {
    check_cuda(cuStreamSynchronize(value_), "cuStreamSynchronize");
  }

 private:
  CUstream value_ = nullptr;
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0) {
      check_cuda(cuMemAlloc(&value_, bytes_), "cuMemAlloc");
    }
  }

  ~DeviceBuffer() {
    if (value_ != 0) {
      (void)cuMemFree(value_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* opaque() const noexcept {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(value_));
  }

  void copy_from_host(const void* source, CUstream stream) const {
    check_cuda(cuMemcpyHtoDAsync(value_, source, bytes_, stream),
               "cuMemcpyHtoDAsync");
  }

  void copy_to_host(void* destination, CUstream stream) const {
    check_cuda(cuMemcpyDtoHAsync(destination, value_, bytes_, stream),
               "cuMemcpyDtoHAsync");
  }

 private:
  CUdeviceptr value_ = 0;
  std::size_t bytes_ = 0;
};

template <std::size_t BindingCount>
void execute_once(
    const flagdnn::Executable& executable,
    const std::array<flagdnnBinding_t, BindingCount>& bindings,
    CUstream stream) {
  DeviceBuffer workspace(executable.workspace_size());
  executable.execute(bindings,
                     workspace.opaque(),
                     executable.workspace_size(),
                     reinterpret_cast<void*>(stream));
  check_cuda(cuStreamSynchronize(stream),
             "workspace-backed execute synchronize");
}

std::vector<std::int64_t> contiguous_strides(
    std::span<const std::int64_t> dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t index = dimensions.size(); index != 0; --index) {
    result[index - 1] = stride;
    stride *= dimensions[index - 1];
  }
  return result;
}

std::size_t storage_element_count(
    std::span<const std::int64_t> dimensions,
    std::span<const std::int64_t> strides) {
  std::size_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    maximum_offset += static_cast<std::size_t>(dimensions[axis] - 1) *
                      static_cast<std::size_t>(strides[axis]);
  }
  return maximum_offset + 1;
}

std::size_t logical_offset(
    std::size_t logical_index,
    std::span<const std::int64_t> dimensions,
    std::span<const std::int64_t> strides) {
  std::size_t result = 0;
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(dimensions[current]);
    const std::size_t coordinate = logical_index % dimension;
    logical_index /= dimension;
    result += coordinate * static_cast<std::size_t>(strides[current]);
  }
  return result;
}

flagdnn::TensorDescriptor make_tensor(
    std::int64_t uid,
    std::initializer_list<std::int64_t> dimensions,
    flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32) {
  const std::vector<std::int64_t> shape(dimensions);
  const std::vector<std::int64_t> strides = contiguous_strides(shape);
  return flagdnn::TensorDescriptor(uid, data_type, shape, strides);
}

flagdnn::Executable build_relu(flagdnn::Handle const& handle,
                               flagdnn::TensorDescriptor const& input,
                               flagdnn::TensorDescriptor const& output) {
  flagdnn::Graph graph;
  graph.relu(input, output);
  graph.finalize();
  return flagdnn::Executable(handle, graph);
}

flagdnn::Executable build_add(flagdnn::Handle const& handle,
                              flagdnn::TensorDescriptor const& left,
                              flagdnn::TensorDescriptor const& right,
                              flagdnn::TensorDescriptor const& output) {
  flagdnn::Graph graph;
  graph.pointwise(left, right, FLAGDNN_POINTWISE_ADD, output);
  graph.finalize();
  return flagdnn::Executable(handle, graph);
}

flagdnn::Executable build_add_with_alpha(
    flagdnn::Handle const& handle,
    flagdnn::TensorDescriptor const& left,
    flagdnn::TensorDescriptor const& right,
    flagdnn::TensorDescriptor const& output,
    double alpha) {
  flagdnn::Graph graph;
  graph.pointwise(left, right, FLAGDNN_POINTWISE_ADD, output, alpha);
  graph.finalize();
  return flagdnn::Executable(handle, graph);
}

flagdnn::Executable build_reduction_sum(
    flagdnn::Handle const& handle,
    flagdnn::TensorDescriptor const& input,
    std::int32_t axis,
    bool keep_dimensions,
    flagdnn::TensorDescriptor const& output) {
  flagdnn::Graph graph;
  graph.reduction_sum(input, axis, keep_dimensions, output);
  graph.finalize();
  return flagdnn::Executable(handle, graph);
}

flagdnn::Executable build_reduction(
    flagdnn::Handle const& handle,
    flagdnn::TensorDescriptor const& input,
    flagdnnReductionMode_t mode,
    std::int32_t axis,
    bool keep_dimensions,
    flagdnn::TensorDescriptor const& output) {
  flagdnn::Graph graph;
  graph.reduction(input, mode, axis, keep_dimensions, output);
  graph.finalize();
  return flagdnn::Executable(handle, graph);
}

flagdnn::Executable build_conv2d_fprop(
    flagdnn::Handle const& handle,
    flagdnn::TensorDescriptor const& input,
    flagdnn::TensorDescriptor const& filter,
    std::span<const std::int64_t, 2> padding,
    std::span<const std::int64_t, 2> stride,
    std::span<const std::int64_t, 2> dilation,
    std::int64_t groups,
    flagdnn::TensorDescriptor const& output) {
  flagdnn::Graph graph;
  graph.conv2d_fprop(
      input, filter, padding, stride, dilation, groups, output);
  graph.finalize();
  return flagdnn::Executable(handle, graph);
}

void require_close(const char* name,
                   std::span<const float> actual,
                   std::span<const float> expected,
                   float tolerance) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(std::string(name) + " output size mismatch");
  }
  float maximum_error = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    maximum_error =
        std::max(maximum_error, std::fabs(actual[index] - expected[index]));
  }
  if (!std::isfinite(maximum_error) || maximum_error > tolerance) {
    std::ostringstream message;
    message << name << " verification failed: max_abs_error="
            << maximum_error << ", tolerance=" << tolerance;
    throw std::runtime_error(message.str());
  }
  std::cout << "PASS " << name << " max_abs_error=" << maximum_error << '\n';
}

template <std::size_t BindingCount>
double benchmark_executable(const flagdnn::Executable& executable,
                      const std::array<flagdnnBinding_t, BindingCount>& bindings,
                      CUstream stream,
                      int iterations = 200) {
  DeviceBuffer workspace(executable.workspace_size());
  for (int index = 0; index < 10; ++index) {
    executable.execute(bindings,
                       workspace.opaque(),
                       executable.workspace_size(),
                       reinterpret_cast<void*>(stream));
  }
  CUevent start = nullptr;
  CUevent stop = nullptr;
  check_cuda(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate(start)");
  try {
    check_cuda(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate(stop)");
    check_cuda(cuEventRecord(start, stream), "cuEventRecord(start)");
    for (int index = 0; index < iterations; ++index) {
      executable.execute(bindings,
                         workspace.opaque(),
                         executable.workspace_size(),
                         reinterpret_cast<void*>(stream));
    }
    check_cuda(cuEventRecord(stop, stream), "cuEventRecord(stop)");
    check_cuda(cuEventSynchronize(stop), "cuEventSynchronize(stop)");
    float milliseconds = 0.0F;
    check_cuda(cuEventElapsedTime(&milliseconds, start, stop),
               "cuEventElapsedTime");
    (void)cuEventDestroy(stop);
    (void)cuEventDestroy(start);
    return static_cast<double>(milliseconds) * 1000.0 /
           static_cast<double>(iterations);
  } catch (...) {
    if (stop != nullptr) {
      (void)cuEventDestroy(stop);
    }
    (void)cuEventDestroy(start);
    throw;
  }
}

double test_relu(const flagdnn::Executable& executable, CUstream stream) {
  constexpr int elements = 1024;
  std::vector<float> input(elements);
  std::vector<float> expected(elements);
  std::vector<float> output(elements);
  for (int index = 0; index < elements; ++index) {
    input[index] = static_cast<float>((index % 37) - 18) / 7.0F;
    expected[index] = std::max(input[index], 0.0F);
  }
  DeviceBuffer x(input.size() * sizeof(float));
  DeviceBuffer y(output.size() * sizeof(float));
  x.copy_from_host(input.data(), stream);
  const std::array<flagdnnBinding_t, 2> bindings = {
      flagdnnBinding_t{1, x.opaque()}, flagdnnBinding_t{2, y.opaque()}};
  execute_once(executable, bindings, stream);
  y.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream), "ReLU synchronize");
  require_close("relu", output, expected, 0.0F);
  return benchmark_executable(executable, bindings, stream);
}

double test_relu_16bit(
    const flagdnn::Executable& executable,
    CUstream stream,
    flagdnnDataType_t data_type,
    std::int64_t input_uid,
    std::int64_t output_uid,
    const char* name) {
  if (data_type != FLAGDNN_DATA_FLOAT16 &&
      data_type != FLAGDNN_DATA_BFLOAT16) {
    throw std::invalid_argument("16-bit ReLU test received another dtype");
  }
  constexpr std::size_t elements = 1024;
  const std::array<std::uint16_t, 6> patterns =
      data_type == FLAGDNN_DATA_FLOAT16
          ? std::array<std::uint16_t, 6>{0xc000, 0xbc00, 0xb800, 0, 0x3800, 0x4000}
          : std::array<std::uint16_t, 6>{0xc000, 0xbf80, 0xbf00, 0, 0x3f00, 0x4000};
  std::vector<std::uint16_t> input(elements);
  std::vector<std::uint16_t> expected(elements);
  std::vector<std::uint16_t> output(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    input[index] = patterns[index % patterns.size()];
    expected[index] =
        (input[index] & 0x8000U) == 0U ? input[index] : 0U;
  }
  DeviceBuffer x(input.size() * sizeof(std::uint16_t));
  DeviceBuffer y(output.size() * sizeof(std::uint16_t));
  x.copy_from_host(input.data(), stream);
  const std::array<flagdnnBinding_t, 2> bindings = {
      flagdnnBinding_t{input_uid, x.opaque()},
      flagdnnBinding_t{output_uid, y.opaque()}};
  execute_once(executable, bindings, stream);
  y.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream), "16-bit ReLU synchronize");
  if (output != expected) {
    throw std::runtime_error(std::string(name) + " bitwise verification failed");
  }
  std::cout << "PASS " << name << " bit_exact=true\n";
  return benchmark_executable(executable, bindings, stream);
}

double test_add(const flagdnn::Executable& executable, CUstream stream) {
  constexpr int elements = 1024;
  std::vector<float> left(elements);
  std::vector<float> right(elements);
  std::vector<float> expected(elements);
  std::vector<float> output(elements);
  for (int index = 0; index < elements; ++index) {
    left[index] = std::sin(static_cast<float>(index) * 0.01F);
    right[index] = std::cos(static_cast<float>(index) * 0.02F);
    expected[index] = left[index] + right[index];
  }
  DeviceBuffer x(left.size() * sizeof(float));
  DeviceBuffer y(right.size() * sizeof(float));
  DeviceBuffer out(output.size() * sizeof(float));
  x.copy_from_host(left.data(), stream);
  y.copy_from_host(right.data(), stream);
  const std::array<flagdnnBinding_t, 3> bindings = {
      flagdnnBinding_t{5, out.opaque()},
      flagdnnBinding_t{3, x.opaque()},
      flagdnnBinding_t{4, y.opaque()}};
  execute_once(executable, bindings, stream);
  out.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream), "Add synchronize");
  require_close("add", output, expected, 0.0F);
  return benchmark_executable(executable, bindings, stream);
}

double test_add_16bit(const flagdnn::Executable& executable,
                      CUstream stream,
                      flagdnnDataType_t data_type,
                      std::int64_t left_uid,
                      std::int64_t right_uid,
                      std::int64_t output_uid,
                      const char* name) {
  if (data_type != FLAGDNN_DATA_FLOAT16 &&
      data_type != FLAGDNN_DATA_BFLOAT16) {
    throw std::invalid_argument("16-bit Add test received another dtype");
  }
  constexpr std::size_t elements = 1024;
  const std::array<std::uint16_t, 6> left_patterns =
      data_type == FLAGDNN_DATA_FLOAT16
          ? std::array<std::uint16_t, 6>{
                0x3c00, 0x4000, 0xbc00, 0x3800, 0, 0xc000}
          : std::array<std::uint16_t, 6>{
                0x3f80, 0x4000, 0xbf80, 0x3f00, 0, 0xc000};
  const std::array<std::uint16_t, 6> right_patterns =
      data_type == FLAGDNN_DATA_FLOAT16
          ? std::array<std::uint16_t, 6>{
                0x3c00, 0x3c00, 0x3800, 0x3800, 0x3c00, 0x3c00}
          : std::array<std::uint16_t, 6>{
                0x3f80, 0x3f80, 0x3f00, 0x3f00, 0x3f80, 0x3f80};
  const std::array<std::uint16_t, 6> expected_patterns =
      data_type == FLAGDNN_DATA_FLOAT16
          ? std::array<std::uint16_t, 6>{
                0x4000, 0x4200, 0xb800, 0x3c00, 0x3c00, 0xbc00}
          : std::array<std::uint16_t, 6>{
                0x4000, 0x4040, 0xbf00, 0x3f80, 0x3f80, 0xbf80};
  std::vector<std::uint16_t> left(elements);
  std::vector<std::uint16_t> right(elements);
  std::vector<std::uint16_t> expected(elements);
  std::vector<std::uint16_t> output(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    const std::size_t pattern = index % left_patterns.size();
    left[index] = left_patterns[pattern];
    right[index] = right_patterns[pattern];
    expected[index] = expected_patterns[pattern];
  }
  DeviceBuffer left_buffer(left.size() * sizeof(std::uint16_t));
  DeviceBuffer right_buffer(right.size() * sizeof(std::uint16_t));
  DeviceBuffer output_buffer(output.size() * sizeof(std::uint16_t));
  left_buffer.copy_from_host(left.data(), stream);
  right_buffer.copy_from_host(right.data(), stream);
  const std::array<flagdnnBinding_t, 3> bindings = {
      flagdnnBinding_t{left_uid, left_buffer.opaque()},
      flagdnnBinding_t{right_uid, right_buffer.opaque()},
      flagdnnBinding_t{output_uid, output_buffer.opaque()}};
  execute_once(executable, bindings, stream);
  output_buffer.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream), "16-bit Add synchronize");
  if (output != expected) {
    throw std::runtime_error(std::string(name) +
                             " bitwise verification failed");
  }
  std::cout << "PASS " << name << " bit_exact=true\n";
  return benchmark_executable(executable, bindings, stream);
}

double test_add_strided_broadcast_alpha(
    const flagdnn::Executable& executable,
    CUstream stream) {
  constexpr float alpha = -0.75F;
  const std::array<std::int64_t, 3> left_dimensions = {2, 3, 4};
  const std::array<std::int64_t, 3> left_strides = {31, 9, 2};
  const std::array<std::int64_t, 2> right_dimensions = {1, 4};
  const std::array<std::int64_t, 2> right_strides = {13, 3};
  const std::array<std::int64_t, 3> output_dimensions = {2, 3, 4};
  const std::array<std::int64_t, 3> output_strides = {37, 11, 2};
  constexpr std::size_t logical_elements = 24;
  constexpr float padding_sentinel = 12345.0F;

  std::vector<float> left(
      storage_element_count(left_dimensions, left_strides), -1000.0F);
  std::vector<float> right(
      storage_element_count(right_dimensions, right_strides), -2000.0F);
  std::vector<float> output(
      storage_element_count(output_dimensions, output_strides),
      padding_sentinel);
  std::vector<float> expected(logical_elements);
  std::vector<bool> output_positions(output.size(), false);

  for (std::size_t index = 0; index < logical_elements; ++index) {
    const std::size_t offset =
        logical_offset(index, left_dimensions, left_strides);
    left[offset] =
        static_cast<float>(static_cast<int>(index) - 12) / 7.0F;
  }
  for (std::size_t index = 0; index < 4; ++index) {
    const std::size_t offset =
        logical_offset(index, right_dimensions, right_strides);
    right[offset] =
        static_cast<float>(static_cast<int>(index) - 2) / 5.0F;
  }
  for (std::size_t index = 0; index < logical_elements; ++index) {
    const std::size_t left_offset =
        logical_offset(index, left_dimensions, left_strides);
    const std::size_t right_index = index % 4;
    const std::size_t right_offset =
        logical_offset(right_index, right_dimensions, right_strides);
    const std::size_t output_offset =
        logical_offset(index, output_dimensions, output_strides);
    expected[index] = left[left_offset] + alpha * right[right_offset];
    output_positions[output_offset] = true;
  }

  DeviceBuffer left_buffer(left.size() * sizeof(float));
  DeviceBuffer right_buffer(right.size() * sizeof(float));
  DeviceBuffer output_buffer(output.size() * sizeof(float));
  left_buffer.copy_from_host(left.data(), stream);
  right_buffer.copy_from_host(right.data(), stream);
  output_buffer.copy_from_host(output.data(), stream);
  const std::array<flagdnnBinding_t, 3> bindings = {
      flagdnnBinding_t{15, left_buffer.opaque()},
      flagdnnBinding_t{16, right_buffer.opaque()},
      flagdnnBinding_t{17, output_buffer.opaque()}};
  execute_once(executable, bindings, stream);
  output_buffer.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream), "strided Add synchronize");

  std::vector<float> actual(logical_elements);
  for (std::size_t index = 0; index < logical_elements; ++index) {
    actual[index] = output[logical_offset(
        index, output_dimensions, output_strides)];
  }
  require_close("add_strided_broadcast_alpha", actual, expected, 1.0e-6F);
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (!output_positions[index] && output[index] != padding_sentinel) {
      throw std::runtime_error("strided Add overwrote output padding");
    }
  }
  std::cout << "PASS add_strided_output_padding_untouched=true\n";
  return benchmark_executable(executable, bindings, stream);
}

double test_reduction(const flagdnn::Executable& executable, CUstream stream) {
  constexpr int rows = 7;
  constexpr int columns = 256;
  std::vector<float> input(rows * columns);
  std::vector<float> expected(rows, 0.0F);
  std::vector<float> output(rows);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < columns; ++column) {
      const float value =
          static_cast<float>(((row * columns + column) % 29) - 14) / 13.0F;
      input[row * columns + column] = value;
      expected[row] += value;
    }
  }
  DeviceBuffer x(input.size() * sizeof(float));
  DeviceBuffer out(output.size() * sizeof(float));
  x.copy_from_host(input.data(), stream);
  const std::array<flagdnnBinding_t, 2> bindings = {
      flagdnnBinding_t{6, x.opaque()}, flagdnnBinding_t{7, out.opaque()}};
  execute_once(executable, bindings, stream);
  out.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream), "Reduction synchronize");
  require_close("reduction_sum", output, expected, 2.0e-5F);
  return benchmark_executable(executable, bindings, stream);
}

void test_reduction_middle_axis(const flagdnn::Executable& executable,
                                CUstream stream) {
  constexpr int outer = 2;
  constexpr int reduction = 4;
  constexpr int inner = 3;
  std::vector<float> input(outer * reduction * inner);
  std::vector<float> expected(outer * inner, 0.0F);
  std::vector<float> output(outer * inner);
  for (int outer_index = 0; outer_index < outer; ++outer_index) {
    for (int reduction_index = 0; reduction_index < reduction;
         ++reduction_index) {
      for (int inner_index = 0; inner_index < inner; ++inner_index) {
        const int input_index =
            (outer_index * reduction + reduction_index) * inner +
            inner_index;
        const float value =
            static_cast<float>((input_index % 13) - 6) / 7.0F;
        input[input_index] = value;
        expected[outer_index * inner + inner_index] += value;
      }
    }
  }
  DeviceBuffer x(input.size() * sizeof(float));
  DeviceBuffer out(output.size() * sizeof(float));
  x.copy_from_host(input.data(), stream);
  const std::array<flagdnnBinding_t, 2> bindings = {
      flagdnnBinding_t{18, x.opaque()}, flagdnnBinding_t{19, out.opaque()}};
  execute_once(executable, bindings, stream);
  out.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream),
             "middle-axis Reduction synchronize");
  require_close("reduction_sum_middle_axis_keepdim",
                output,
                expected,
                2.0e-5F);
}

void test_reduction_mode(const flagdnn::Executable& executable,
                         CUstream stream,
                         flagdnnReductionMode_t mode,
                         const char* name) {
  constexpr int outer = 2;
  constexpr int reduction = 4;
  constexpr int inner = 3;
  std::vector<float> input(outer * reduction * inner);
  std::vector<float> expected(
      outer * inner, mode == FLAGDNN_REDUCTION_MUL ? 1.0F : 0.0F);
  std::vector<float> output(outer * inner);
  for (int outer_index = 0; outer_index < outer; ++outer_index) {
    for (int reduction_index = 0; reduction_index < reduction;
         ++reduction_index) {
      for (int inner_index = 0; inner_index < inner; ++inner_index) {
        const int input_index =
            (outer_index * reduction + reduction_index) * inner +
            inner_index;
        const float value =
            static_cast<float>((input_index % 7) + 1) / 8.0F;
        input[input_index] = value;
        float& accumulator = expected[outer_index * inner + inner_index];
        if (mode == FLAGDNN_REDUCTION_MUL) {
          accumulator *= value;
        } else {
          accumulator += value;
        }
      }
    }
  }
  if (mode == FLAGDNN_REDUCTION_AVG) {
    for (float& value : expected) {
      value /= static_cast<float>(reduction);
    }
  }
  DeviceBuffer x(input.size() * sizeof(float));
  DeviceBuffer out(output.size() * sizeof(float));
  x.copy_from_host(input.data(), stream);
  const std::array<flagdnnBinding_t, 2> bindings = {
      flagdnnBinding_t{18, x.opaque()}, flagdnnBinding_t{19, out.opaque()}};
  execute_once(executable, bindings, stream);
  out.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream), "Reduction mode synchronize");
  require_close(name, output, expected, 2.0e-5F);
}

double test_reduction_strided(const flagdnn::Executable& executable,
                              CUstream stream) {
  const std::array<std::int64_t, 3> input_dimensions = {2, 4, 3};
  const std::array<std::int64_t, 3> input_strides = {12, 1, 4};
  const std::array<std::int64_t, 3> output_dimensions = {2, 1, 3};
  const std::array<std::int64_t, 3> output_strides = {7, 5, 2};
  constexpr float padding_sentinel = 12345.0F;
  constexpr std::size_t logical_input_elements = 24;
  constexpr std::size_t logical_output_elements = 6;

  std::vector<float> input(
      storage_element_count(input_dimensions, input_strides), -1000.0F);
  std::vector<float> output(
      storage_element_count(output_dimensions, output_strides),
      padding_sentinel);
  std::vector<float> expected(logical_output_elements, 0.0F);
  std::vector<bool> output_positions(output.size(), false);
  for (std::size_t logical_index = 0;
       logical_index < logical_input_elements;
       ++logical_index) {
    const float value =
        static_cast<float>(static_cast<int>(logical_index) - 12) / 7.0F;
    input[logical_offset(logical_index, input_dimensions, input_strides)] =
        value;
    const std::size_t n = logical_index / 12;
    const std::size_t w = logical_index % 3;
    expected[n * 3 + w] += value;
  }
  for (std::size_t logical_index = 0;
       logical_index < logical_output_elements;
       ++logical_index) {
    output_positions[logical_offset(
        logical_index, output_dimensions, output_strides)] = true;
  }

  DeviceBuffer input_buffer(input.size() * sizeof(float));
  DeviceBuffer output_buffer(output.size() * sizeof(float));
  input_buffer.copy_from_host(input.data(), stream);
  output_buffer.copy_from_host(output.data(), stream);
  const std::array<flagdnnBinding_t, 2> bindings = {
      flagdnnBinding_t{26, input_buffer.opaque()},
      flagdnnBinding_t{27, output_buffer.opaque()}};
  execute_once(executable, bindings, stream);
  output_buffer.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream), "strided Reduction synchronize");

  std::vector<float> actual(logical_output_elements);
  for (std::size_t logical_index = 0;
       logical_index < logical_output_elements;
       ++logical_index) {
    actual[logical_index] = output[logical_offset(
        logical_index, output_dimensions, output_strides)];
  }
  require_close("reduction_sum_strided", actual, expected, 2.0e-5F);
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (!output_positions[index] && output[index] != padding_sentinel) {
      throw std::runtime_error("strided Reduction overwrote output padding");
    }
  }
  std::cout << "PASS reduction_strided_output_padding_untouched=true\n";
  return benchmark_executable(executable, bindings, stream);
}

void test_reduction_scalar(const flagdnn::Executable& executable,
                           CUstream stream) {
  std::vector<float> input = {
      -1.5F, 0.25F, 2.0F, -3.25F, 4.5F, 1.0F, -0.75F, 2.25F};
  const std::vector<float> expected = {
      std::accumulate(input.begin(), input.end(), 0.0F)};
  std::vector<float> output(1);
  DeviceBuffer x(input.size() * sizeof(float));
  DeviceBuffer out(sizeof(float));
  x.copy_from_host(input.data(), stream);
  const std::array<flagdnnBinding_t, 2> bindings = {
      flagdnnBinding_t{20, x.opaque()}, flagdnnBinding_t{21, out.opaque()}};
  execute_once(executable, bindings, stream);
  out.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream), "scalar Reduction synchronize");
  require_close("reduction_sum_scalar", output, expected, 2.0e-5F);
}

void test_reduction_16bit_middle_axis(
    const flagdnn::Executable& executable,
    CUstream stream,
    flagdnnDataType_t data_type,
    std::int64_t input_uid,
    std::int64_t output_uid,
    const char* name) {
  if (data_type != FLAGDNN_DATA_FLOAT16 &&
      data_type != FLAGDNN_DATA_BFLOAT16) {
    throw std::invalid_argument(
        "16-bit Reduction test received another dtype");
  }
  const std::uint16_t one =
      data_type == FLAGDNN_DATA_FLOAT16 ? 0x3c00U : 0x3f80U;
  const std::uint16_t negative_one =
      data_type == FLAGDNN_DATA_FLOAT16 ? 0xbc00U : 0xbf80U;
  const std::uint16_t half =
      data_type == FLAGDNN_DATA_FLOAT16 ? 0x3800U : 0x3f00U;
  std::vector<std::uint16_t> input(8);
  for (int inner = 0; inner < 2; ++inner) {
    input[inner] = one;
    input[2 + inner] = negative_one;
    input[4 + inner] = half;
    input[6 + inner] = half;
  }
  const std::vector<std::uint16_t> expected(2, one);
  std::vector<std::uint16_t> output(2);
  DeviceBuffer x(input.size() * sizeof(std::uint16_t));
  DeviceBuffer out(output.size() * sizeof(std::uint16_t));
  x.copy_from_host(input.data(), stream);
  const std::array<flagdnnBinding_t, 2> bindings = {
      flagdnnBinding_t{input_uid, x.opaque()},
      flagdnnBinding_t{output_uid, out.opaque()}};
  execute_once(executable, bindings, stream);
  out.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream), "16-bit Reduction synchronize");
  if (output != expected) {
    throw std::runtime_error(
        std::string(name) + " bitwise verification failed");
  }
  std::cout << "PASS " << name << " bit_exact=true\n";
}

double test_conv(const flagdnn::Executable& executable, CUstream stream) {
  constexpr int n = 1;
  constexpr int c = 2;
  constexpr int h = 5;
  constexpr int w = 5;
  constexpr int k = 2;
  constexpr int r = 3;
  constexpr int s = 3;
  constexpr int oh = 5;
  constexpr int ow = 5;
  constexpr int pad_h = 1;
  constexpr int pad_w = 1;
  constexpr int output_count = n * k * oh * ow;

  std::vector<float> input(n * c * h * w);
  std::vector<float> weights(k * c * r * s);
  std::vector<float> expected(output_count, 0.0F);
  std::vector<float> output(output_count);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] =
        static_cast<float>((static_cast<int>(index) % 17) - 8) / 9.0F;
  }
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] =
        static_cast<float>((static_cast<int>(index) % 11) - 5) / 12.0F;
  }
  for (int ni = 0; ni < n; ++ni) {
    for (int ko = 0; ko < k; ++ko) {
      for (int ho = 0; ho < oh; ++ho) {
        for (int wo = 0; wo < ow; ++wo) {
          float accumulator = 0.0F;
          for (int ci = 0; ci < c; ++ci) {
            for (int ri = 0; ri < r; ++ri) {
              for (int si = 0; si < s; ++si) {
                const int hi = ho + ri - pad_h;
                const int wi = wo + si - pad_w;
                if (hi < 0 || hi >= h || wi < 0 || wi >= w) {
                  continue;
                }
                const int input_index = ((ni * c + ci) * h + hi) * w + wi;
                const int weight_index = ((ko * c + ci) * r + ri) * s + si;
                accumulator += input[input_index] * weights[weight_index];
              }
            }
          }
          expected[((ni * k + ko) * oh + ho) * ow + wo] = accumulator;
        }
      }
    }
  }

  DeviceBuffer x(input.size() * sizeof(float));
  DeviceBuffer filter(weights.size() * sizeof(float));
  DeviceBuffer out(output.size() * sizeof(float));
  x.copy_from_host(input.data(), stream);
  filter.copy_from_host(weights.data(), stream);
  const std::array<flagdnnBinding_t, 3> bindings = {
      flagdnnBinding_t{8, x.opaque()},
      flagdnnBinding_t{9, filter.opaque()},
      flagdnnBinding_t{10, out.opaque()}};
  execute_once(executable, bindings, stream);
  out.copy_to_host(output.data(), stream);
  check_cuda(cuStreamSynchronize(stream), "Conv2D synchronize");
  require_close("conv2d_fprop", output, expected, 2.0e-5F);
  return benchmark_executable(executable, bindings, stream);
}

template <std::size_t ExecutableCount>
void require_workspace_contract(
    const std::array<const flagdnn::Executable*, ExecutableCount>& executables) {
  for (const flagdnn::Executable* executable : executables) {
#if defined(FLAGDNN_EXPECT_LIBTRITON_JIT)
    if (executable->workspace_size() == 0) {
      throw std::runtime_error(
          "libtriton_jit executable is missing its runtime scratch workspace");
    }
#else
    if (executable->workspace_size() != 0) {
      throw std::runtime_error(
          "external artifact executable unexpectedly needs workspace");
    }
#endif
  }
}

void test_relu_validation_contract(flagdnn::Handle const& handle) {
  const auto require_build_failure = [](
      auto&& build,
      flagdnnStatus_t expected_status,
      const char* message) {
    try {
      auto unexpected = build();
      (void)unexpected;
    } catch (const flagdnn::Error& error) {
      if (error.status() == expected_status) {
        return;
      }
      throw;
    }
    throw std::runtime_error(message);
  };

  {
    auto input = make_tensor(101, {16}, FLAGDNN_DATA_FLOAT16);
    auto output = make_tensor(102, {16}, FLAGDNN_DATA_BFLOAT16);
    require_build_failure(
        [&] { return build_relu(handle, input, output); },
        FLAGDNN_STATUS_INVALID_VALUE,
        "mixed ReLU data types were not rejected");
  }

  const std::array<std::int64_t, 2> dimensions = {2, 3};
  const std::array<std::int64_t, 2> contiguous = {3, 1};
  const std::array<std::int64_t, 2> overlapping = {1, 1};
  {
    flagdnn::TensorDescriptor input(
        103, FLAGDNN_DATA_FLOAT32, dimensions, overlapping);
    flagdnn::TensorDescriptor output(
        104, FLAGDNN_DATA_FLOAT32, dimensions, contiguous);
    require_build_failure(
        [&] { return build_relu(handle, input, output); },
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "overlapping ReLU input strides were not rejected");
  }
  {
    flagdnn::TensorDescriptor input(
        105, FLAGDNN_DATA_FLOAT32, dimensions, contiguous);
    flagdnn::TensorDescriptor output(
        106, FLAGDNN_DATA_FLOAT32, dimensions, overlapping);
    require_build_failure(
        [&] { return build_relu(handle, input, output); },
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "overlapping ReLU output strides were not rejected");
  }
}

void test_add_validation_contract(flagdnn::Handle const& handle) {
  const auto require_build_failure = [](
      auto&& build,
      flagdnnStatus_t expected_status,
      const char* message) {
    try {
      auto unexpected = build();
      (void)unexpected;
    } catch (const flagdnn::Error& error) {
      if (error.status() == expected_status) {
        return;
      }
      throw;
    }
    throw std::runtime_error(message);
  };

  {
    auto left = make_tensor(106, {2, 3});
    auto right = make_tensor(107, {4});
    auto output = make_tensor(108, {2, 3});
    require_build_failure(
        [&] { return build_add(handle, left, right, output); },
        FLAGDNN_STATUS_INVALID_VALUE,
        "invalid Add broadcast was not rejected");
  }
  {
    auto left = make_tensor(109, {2, 1});
    auto right = make_tensor(110, {1, 3});
    auto output = make_tensor(111, {2, 2});
    require_build_failure(
        [&] { return build_add(handle, left, right, output); },
        FLAGDNN_STATUS_INVALID_VALUE,
        "incorrect Add output shape was not rejected");
  }
  {
    auto left = make_tensor(123, {2, 3}, FLAGDNN_DATA_FLOAT16);
    auto right = make_tensor(124, {2, 3}, FLAGDNN_DATA_FLOAT16);
    auto output = make_tensor(125, {2, 3}, FLAGDNN_DATA_BFLOAT16);
    require_build_failure(
        [&] { return build_add(handle, left, right, output); },
        FLAGDNN_STATUS_INVALID_VALUE,
        "mixed Add data types were not rejected");
  }
  {
    const std::array<std::int64_t, 2> dimensions = {2, 3};
    const std::array<std::int64_t, 2> overlapping_strides = {1, 1};
    auto left = make_tensor(112, {2, 3});
    auto right = make_tensor(113, {2, 3});
    flagdnn::TensorDescriptor output(
        114,
        FLAGDNN_DATA_FLOAT32,
        dimensions,
        overlapping_strides);
    require_build_failure(
        [&] { return build_add(handle, left, right, output); },
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "overlapping Add output strides were not rejected");
  }
  std::int64_t uid = 126;
  for (const flagdnnDataType_t data_type : {
           FLAGDNN_DATA_FLOAT32,
           FLAGDNN_DATA_FLOAT16,
           FLAGDNN_DATA_BFLOAT16}) {
    const std::array<std::int64_t, 2> dimensions = {2, 3};
    const std::array<std::int64_t, 2> overlapping_strides = {1, 1};
    flagdnn::TensorDescriptor left(
        uid++, data_type, dimensions, overlapping_strides);
    auto right = make_tensor(uid++, {2, 3}, data_type);
    auto output = make_tensor(uid++, {2, 3}, data_type);
    require_build_failure(
        [&] { return build_add(handle, left, right, output); },
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "overlapping Add input strides were not rejected");
  }
}

void test_reduction_validation_contract(flagdnn::Handle const& handle) {
  const auto require_build_failure = [](
      auto&& build,
      flagdnnStatus_t expected_status,
      const char* message) {
    try {
      auto unexpected = build();
      (void)unexpected;
    } catch (const flagdnn::Error& error) {
      if (error.status() == expected_status) {
        return;
      }
      throw;
    }
    throw std::runtime_error(message);
  };

  {
    auto input = make_tensor(115, {2, 3, 4});
    auto output = make_tensor(116, {2, 3});
    require_build_failure(
        [&] { return build_reduction_sum(handle, input, 3, false, output); },
        FLAGDNN_STATUS_INVALID_VALUE,
        "out-of-range Reduction axis was not rejected");
  }
  {
    auto input = make_tensor(117, {2, 3, 4});
    auto output = make_tensor(118, {2, 4});
    require_build_failure(
        [&] { return build_reduction_sum(handle, input, 1, true, output); },
        FLAGDNN_STATUS_INVALID_VALUE,
        "incorrect Reduction output shape was not rejected");
  }
  {
    auto input = make_tensor(119, {2, 3}, FLAGDNN_DATA_FLOAT16);
    auto output = make_tensor(120, {2, 1}, FLAGDNN_DATA_BFLOAT16);
    require_build_failure(
        [&] { return build_reduction_sum(handle, input, 1, true, output); },
        FLAGDNN_STATUS_INVALID_VALUE,
        "mixed Reduction data types were not rejected");
  }
  {
    const std::array<std::int64_t, 3> dimensions = {2, 3, 4};
    const std::array<std::int64_t, 3> overlapping_strides = {1, 1, 1};
    flagdnn::TensorDescriptor input(121,
                                    FLAGDNN_DATA_FLOAT32,
                                    dimensions,
                                    overlapping_strides);
    auto output = make_tensor(122, {2, 1, 4});
    require_build_failure(
        [&] { return build_reduction_sum(handle, input, 1, true, output); },
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "overlapping Reduction input strides were not rejected");
  }
}

void test_conv_validation_contract(flagdnn::Handle const& handle) {
  const auto require_build_failure = [](
      auto&& build,
      flagdnnStatus_t expected_status,
      const char* message) {
    try {
      auto unexpected = build();
      (void)unexpected;
    } catch (const flagdnn::Error& error) {
      if (error.status() == expected_status) {
        return;
      }
      throw;
    }
    throw std::runtime_error(message);
  };
  const std::array<std::int64_t, 2> padding = {1, 1};
  const std::array<std::int64_t, 2> stride = {1, 1};
  const std::array<std::int64_t, 2> dilation = {1, 1};

  {
    auto input = make_tensor(140, {1, 2, 5, 5}, FLAGDNN_DATA_FLOAT16);
    auto filter = make_tensor(141, {2, 2, 3, 3}, FLAGDNN_DATA_FLOAT16);
    auto output = make_tensor(142, {1, 2, 5, 5}, FLAGDNN_DATA_BFLOAT16);
    require_build_failure(
        [&] {
          return build_conv2d_fprop(
              handle, input, filter, padding, stride, dilation, 1, output);
        },
        FLAGDNN_STATUS_INVALID_VALUE,
        "mixed Conv2D data types were not rejected");
  }
  {
    auto input = make_tensor(143, {1, 2, 5, 5});
    auto filter = make_tensor(144, {2, 2, 3, 3});
    auto output = make_tensor(145, {1, 2, 5, 5});
    require_build_failure(
        [&] {
          return build_conv2d_fprop(
              handle, input, filter, padding, stride, dilation, 0, output);
        },
        FLAGDNN_STATUS_INVALID_VALUE,
        "non-positive Conv2D groups were not rejected");
  }
  {
    auto input = make_tensor(146, {1, 4, 5, 5});
    auto filter = make_tensor(147, {6, 4, 3, 3});
    auto output = make_tensor(148, {1, 6, 5, 5});
    require_build_failure(
        [&] {
          return build_conv2d_fprop(
              handle, input, filter, padding, stride, dilation, 2, output);
        },
        FLAGDNN_STATUS_INVALID_VALUE,
        "invalid grouped Conv2D filter channels were not rejected");
  }
  {
    const std::array<std::int64_t, 4> dimensions = {1, 2, 5, 5};
    const std::array<std::int64_t, 4> overlapping_strides = {50, 1, 1, 1};
    flagdnn::TensorDescriptor input(
        149, FLAGDNN_DATA_FLOAT32, dimensions, overlapping_strides);
    auto filter = make_tensor(150, {2, 2, 3, 3});
    auto output = make_tensor(151, {1, 2, 5, 5});
    require_build_failure(
        [&] {
          return build_conv2d_fprop(
              handle, input, filter, padding, stride, dilation, 1, output);
        },
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "overlapping Conv2D input strides were not rejected");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      std::cerr << "usage: native_nvidia_integration COMPILER_EXECUTABLE COMPILER_ENTRY\n";
      return 2;
    }
    require_native_process_clean("before_codegen");
    DriverContext caller_context;
    Stream caller_stream;
    TemporaryCache cache;
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());

    auto relu_input = make_tensor(1, {1024});
    auto relu_output = make_tensor(2, {1024});
    auto relu_fp16_input = make_tensor(11, {1024}, FLAGDNN_DATA_FLOAT16);
    auto relu_fp16_output = make_tensor(12, {1024}, FLAGDNN_DATA_FLOAT16);
    auto relu_bf16_input = make_tensor(13, {1024}, FLAGDNN_DATA_BFLOAT16);
    auto relu_bf16_output = make_tensor(14, {1024}, FLAGDNN_DATA_BFLOAT16);
    auto add_left = make_tensor(3, {1024});
    auto add_right = make_tensor(4, {1024});
    auto add_output = make_tensor(5, {1024});
    auto add_fp16_left =
        make_tensor(28, {1024}, FLAGDNN_DATA_FLOAT16);
    auto add_fp16_right =
        make_tensor(29, {1024}, FLAGDNN_DATA_FLOAT16);
    auto add_fp16_output =
        make_tensor(30, {1024}, FLAGDNN_DATA_FLOAT16);
    auto add_bf16_left =
        make_tensor(31, {1024}, FLAGDNN_DATA_BFLOAT16);
    auto add_bf16_right =
        make_tensor(32, {1024}, FLAGDNN_DATA_BFLOAT16);
    auto add_bf16_output =
        make_tensor(33, {1024}, FLAGDNN_DATA_BFLOAT16);
    const std::array<std::int64_t, 3> strided_add_dimensions = {2, 3, 4};
    const std::array<std::int64_t, 3> strided_add_left_strides = {31, 9, 2};
    const std::array<std::int64_t, 2> strided_add_right_dimensions = {1, 4};
    const std::array<std::int64_t, 2> strided_add_right_strides = {13, 3};
    const std::array<std::int64_t, 3> strided_add_output_strides = {37, 11, 2};
    flagdnn::TensorDescriptor strided_add_left(
        15,
        FLAGDNN_DATA_FLOAT32,
        strided_add_dimensions,
        strided_add_left_strides);
    flagdnn::TensorDescriptor strided_add_right(
        16,
        FLAGDNN_DATA_FLOAT32,
        strided_add_right_dimensions,
        strided_add_right_strides);
    flagdnn::TensorDescriptor strided_add_output(
        17,
        FLAGDNN_DATA_FLOAT32,
        strided_add_dimensions,
        strided_add_output_strides);
    auto reduction_input = make_tensor(6, {7, 256});
    auto reduction_output = make_tensor(7, {7});
    auto reduction_middle_input = make_tensor(18, {2, 4, 3});
    auto reduction_middle_output = make_tensor(19, {2, 1, 3});
    const std::array<std::int64_t, 3> strided_reduction_input_dimensions = {
        2, 4, 3};
    const std::array<std::int64_t, 3> strided_reduction_input_strides = {
        12, 1, 4};
    const std::array<std::int64_t, 3> strided_reduction_output_dimensions = {
        2, 1, 3};
    const std::array<std::int64_t, 3> strided_reduction_output_strides = {
        7, 5, 2};
    flagdnn::TensorDescriptor strided_reduction_input(
        26,
        FLAGDNN_DATA_FLOAT32,
        strided_reduction_input_dimensions,
        strided_reduction_input_strides);
    flagdnn::TensorDescriptor strided_reduction_output(
        27,
        FLAGDNN_DATA_FLOAT32,
        strided_reduction_output_dimensions,
        strided_reduction_output_strides);
    auto reduction_fp16_input =
        make_tensor(22, {1, 4, 2}, FLAGDNN_DATA_FLOAT16);
    auto reduction_fp16_output =
        make_tensor(23, {1, 1, 2}, FLAGDNN_DATA_FLOAT16);
    auto reduction_bf16_input =
        make_tensor(24, {1, 4, 2}, FLAGDNN_DATA_BFLOAT16);
    auto reduction_bf16_output =
        make_tensor(25, {1, 1, 2}, FLAGDNN_DATA_BFLOAT16);
    auto reduction_scalar_input = make_tensor(20, {8});
    auto reduction_scalar_output = make_tensor(21, {});
    auto conv_input = make_tensor(8, {1, 2, 5, 5});
    auto conv_filter = make_tensor(9, {2, 2, 3, 3});
    auto conv_output = make_tensor(10, {1, 2, 5, 5});

    auto relu_executable = build_relu(handle, relu_input, relu_output);
    auto relu_fp16_executable =
        build_relu(handle, relu_fp16_input, relu_fp16_output);
    auto relu_bf16_executable =
        build_relu(handle, relu_bf16_input, relu_bf16_output);
    auto add_executable =
        build_add(handle, add_left, add_right, add_output);
    auto add_fp16_executable = build_add(
        handle, add_fp16_left, add_fp16_right, add_fp16_output);
    auto add_bf16_executable = build_add(
        handle, add_bf16_left, add_bf16_right, add_bf16_output);
    auto strided_add_executable = build_add_with_alpha(handle,
                                                        strided_add_left,
                                                        strided_add_right,
                                                        strided_add_output,
                                                        -0.75);
    auto reduction_executable = build_reduction_sum(
        handle, reduction_input, -1, false, reduction_output);
    auto reduction_middle_executable = build_reduction_sum(
        handle,
        reduction_middle_input,
        1,
        true,
        reduction_middle_output);
    auto reduction_avg_executable = build_reduction(
        handle,
        reduction_middle_input,
        FLAGDNN_REDUCTION_AVG,
        1,
        true,
        reduction_middle_output);
    auto reduction_mul_executable = build_reduction(
        handle,
        reduction_middle_input,
        FLAGDNN_REDUCTION_MUL,
        1,
        true,
        reduction_middle_output);
    auto strided_reduction_executable = build_reduction_sum(
        handle,
        strided_reduction_input,
        1,
        true,
        strided_reduction_output);
    auto reduction_fp16_executable = build_reduction_sum(
        handle,
        reduction_fp16_input,
        1,
        true,
        reduction_fp16_output);
    auto reduction_bf16_executable = build_reduction_sum(
        handle,
        reduction_bf16_input,
        1,
        true,
        reduction_bf16_output);
    auto reduction_scalar_executable = build_reduction_sum(
        handle,
        reduction_scalar_input,
        0,
        false,
        reduction_scalar_output);
    const std::array<std::int64_t, 2> padding = {1, 1};
    const std::array<std::int64_t, 2> stride = {1, 1};
    const std::array<std::int64_t, 2> dilation = {1, 1};
    auto conv_executable = build_conv2d_fprop(handle,
                                                 conv_input,
                                                 conv_filter,
                                                 padding,
                                                 stride,
                                                 dilation,
                                                 1,
                                                 conv_output);
    require_workspace_contract(
        std::array<const flagdnn::Executable*, 16>{
            &relu_executable,
            &relu_fp16_executable,
            &relu_bf16_executable,
            &add_executable,
            &add_fp16_executable,
            &add_bf16_executable,
            &strided_add_executable,
            &reduction_executable,
            &reduction_middle_executable,
            &reduction_avg_executable,
            &reduction_mul_executable,
            &strided_reduction_executable,
            &reduction_fp16_executable,
            &reduction_bf16_executable,
            &reduction_scalar_executable,
            &conv_executable});
    require_native_process_clean("after_codegen");
    if (cache.manifest_count() != 16) {
      throw std::runtime_error("expected exactly sixteen compiled artifacts");
    }

    handle.set_compiler("/definitely/missing/flagdnn-python",
                       argv[2],
                       cache.path().string());
    auto cached_relu = build_relu(handle, relu_input, relu_output);
    auto cached_relu_fp16 =
        build_relu(handle, relu_fp16_input, relu_fp16_output);
    auto cached_relu_bf16 =
        build_relu(handle, relu_bf16_input, relu_bf16_output);
    auto cached_add =
        build_add(handle, add_left, add_right, add_output);
    auto cached_add_fp16 = build_add(
        handle, add_fp16_left, add_fp16_right, add_fp16_output);
    auto cached_add_bf16 = build_add(
        handle, add_bf16_left, add_bf16_right, add_bf16_output);
    auto cached_strided_add = build_add_with_alpha(handle,
                                                   strided_add_left,
                                                   strided_add_right,
                                                   strided_add_output,
                                                   -0.75);
    auto cached_reduction = build_reduction_sum(
        handle, reduction_input, -1, false, reduction_output);
    auto cached_reduction_middle = build_reduction_sum(
        handle,
        reduction_middle_input,
        1,
        true,
        reduction_middle_output);
    auto cached_reduction_avg = build_reduction(
        handle,
        reduction_middle_input,
        FLAGDNN_REDUCTION_AVG,
        1,
        true,
        reduction_middle_output);
    auto cached_reduction_mul = build_reduction(
        handle,
        reduction_middle_input,
        FLAGDNN_REDUCTION_MUL,
        1,
        true,
        reduction_middle_output);
    auto cached_strided_reduction = build_reduction_sum(
        handle,
        strided_reduction_input,
        1,
        true,
        strided_reduction_output);
    auto cached_reduction_fp16 = build_reduction_sum(
        handle,
        reduction_fp16_input,
        1,
        true,
        reduction_fp16_output);
    auto cached_reduction_bf16 = build_reduction_sum(
        handle,
        reduction_bf16_input,
        1,
        true,
        reduction_bf16_output);
    auto cached_reduction_scalar = build_reduction_sum(
        handle,
        reduction_scalar_input,
        0,
        false,
        reduction_scalar_output);
    auto cached_conv = build_conv2d_fprop(handle,
                                                   conv_input,
                                                   conv_filter,
                                                   padding,
                                                   stride,
                                                   dilation,
                                                   1,
                                                   conv_output);
    require_workspace_contract(
        std::array<const flagdnn::Executable*, 16>{
            &cached_relu,
            &cached_relu_fp16,
            &cached_relu_bf16,
            &cached_add,
            &cached_add_fp16,
            &cached_add_bf16,
            &cached_strided_add,
            &cached_reduction,
            &cached_reduction_middle,
            &cached_reduction_avg,
            &cached_reduction_mul,
            &cached_strided_reduction,
            &cached_reduction_fp16,
            &cached_reduction_bf16,
            &cached_reduction_scalar,
            &cached_conv});
    std::cout << "PASS cache_hit_without_python\n";
    test_relu_validation_contract(handle);
    std::cout << "PASS relu_validation_contract\n";
    test_add_validation_contract(handle);
    std::cout << "PASS add_validation_contract\n";
    test_reduction_validation_contract(handle);
    std::cout << "PASS reduction_validation_contract\n";
    test_conv_validation_contract(handle);
    std::cout << "PASS conv2d_validation_contract\n";
    {
      flagdnn::Handle completed_handle = std::move(handle);
    }
    std::cout << "PASS executable_outlives_handle_graph_and_descriptors\n";

    const double relu_us = test_relu(relu_executable, caller_stream.get());
    const double relu_fp16_us = test_relu_16bit(
        relu_fp16_executable,
        caller_stream.get(),
        FLAGDNN_DATA_FLOAT16,
        11,
        12,
        "relu_fp16");
    const double relu_bf16_us = test_relu_16bit(
        relu_bf16_executable,
        caller_stream.get(),
        FLAGDNN_DATA_BFLOAT16,
        13,
        14,
        "relu_bfloat16");
    const double add_us = test_add(add_executable, caller_stream.get());
    const double add_fp16_us = test_add_16bit(add_fp16_executable,
                                              caller_stream.get(),
                                              FLAGDNN_DATA_FLOAT16,
                                              28,
                                              29,
                                              30,
                                              "add_fp16");
    const double add_bf16_us = test_add_16bit(add_bf16_executable,
                                              caller_stream.get(),
                                              FLAGDNN_DATA_BFLOAT16,
                                              31,
                                              32,
                                              33,
                                              "add_bfloat16");
    const double strided_add_us = test_add_strided_broadcast_alpha(
        strided_add_executable, caller_stream.get());
    const double reduction_us =
        test_reduction(reduction_executable, caller_stream.get());
    test_reduction_middle_axis(
        reduction_middle_executable, caller_stream.get());
    test_reduction_mode(reduction_avg_executable,
                        caller_stream.get(),
                        FLAGDNN_REDUCTION_AVG,
                        "reduction_avg_middle_axis_keepdim");
    test_reduction_mode(reduction_mul_executable,
                        caller_stream.get(),
                        FLAGDNN_REDUCTION_MUL,
                        "reduction_mul_middle_axis_keepdim");
    const double strided_reduction_us = test_reduction_strided(
        strided_reduction_executable, caller_stream.get());
    test_reduction_16bit_middle_axis(reduction_fp16_executable,
                                     caller_stream.get(),
                                     FLAGDNN_DATA_FLOAT16,
                                     22,
                                     23,
                                     "reduction_sum_fp16_middle_axis");
    test_reduction_16bit_middle_axis(reduction_bf16_executable,
                                     caller_stream.get(),
                                     FLAGDNN_DATA_BFLOAT16,
                                     24,
                                     25,
                                     "reduction_sum_bfloat16_middle_axis");
    test_reduction_scalar(reduction_scalar_executable, caller_stream.get());
    const double conv_us = test_conv(conv_executable, caller_stream.get());
    caller_stream.synchronize();
    require_native_process_clean("after_execution");
    std::cout << std::fixed << std::setprecision(3)
              << "steady_state_us relu_fp32=" << relu_us
              << " relu_fp16=" << relu_fp16_us
              << " relu_bfloat16=" << relu_bf16_us << " add=" << add_us
              << " add_fp16=" << add_fp16_us
              << " add_bfloat16=" << add_bf16_us
              << " add_strided_broadcast_alpha=" << strided_add_us
              << " reduction_sum=" << reduction_us
              << " reduction_sum_strided=" << strided_reduction_us
              << " conv2d_fprop=" << conv_us << '\n';
    std::cout << "ALL_NATIVE_C_API_TESTS_PASSED\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "NATIVE_C_API_TEST_FAILED: " << error.what() << '\n';
    return 1;
  }
}
