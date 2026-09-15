/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_PAIRED_MEASUREMENT_HPP_
#define FLAGDNN_NVIDIA_PAIRED_MEASUREMENT_HPP_
#include <array>

#include "common/common.hpp"
#include "validation/benchmark/timing.hpp"
namespace flagdnn::testing::cuda {
namespace timing = flagdnn::validation::nvidia::timing;
inline void measure_paired_execution(
    std::string_view name, TestExecutable& flagdnn, TestExecutable& cudnn,
    std::span<const flagdnnBinding_t> flagdnn_bindings,
    std::span<const flagdnnBinding_t> cudnn_bindings,
    DeviceBuffer& flagdnn_workspace, DeviceBuffer& cudnn_workspace,
    Stream& stream, timing::RuntimeMeasurements flagdnn_measurements,
    timing::RuntimeMeasurements cudnn_measurements) {
  std::array<TestExecutable*, 2> executables{&flagdnn, &cudnn};
  std::array bindings{flagdnn_bindings, cudnn_bindings};
  std::array workspaces{&flagdnn_workspace, &cudnn_workspace};
  std::array measurements{flagdnn_measurements, cudnn_measurements};
  const auto execute = [&](std::size_t provider) {
    executables[provider]->execute(
        bindings[provider], workspaces[provider]->opaque(),
        executables[provider]->workspace_size(), stream.opaque());
  };
  for (int warmup = 0; warmup < 10; ++warmup) {
    execute(0);
    execute(1);
  }
  stream.synchronize();
  timing::CapturedExecutionBatch flagdnn_batch(stream.get(), 50,
                                               [&] { execute(0); });
  timing::CapturedExecutionBatch cudnn_batch(stream.get(), 50,
                                             [&] { execute(1); });
  std::array batches{&flagdnn_batch, &cudnn_batch};
  flagdnn_batch.launch(stream.get());
  cudnn_batch.launch(stream.get());
  stream.synchronize();
  EventTimer timer;
  std::array<std::vector<double>, 2> samples;
  for (int sample = 0; sample < 20; ++sample) {
    for (int order = 0; order < 2; ++order) {
      const auto provider = (sample + order) % 2;
      const auto value = timer.measure_microseconds(stream.get(), 1, [&] {
        batches[provider]->launch(stream.get());
      }) / 50.0;
      if (!std::isfinite(value) || value <= 0)
        throw std::runtime_error("invalid GPU timing sample");
      samples[provider].push_back(value);
    }
  }
  for (int sample = 0; sample < 20; ++sample) {
    for (int order = 0; order < 2; ++order) {
      const auto provider = (sample + order) % 2;
      stream.synchronize();
      const auto begin = timing::HostClock::now();
      for (int iteration = 0; iteration < 50; ++iteration) execute(provider);
      measurements[provider].host_submit_us.push_back(
          timing::host_microseconds_since(begin) / 50.0);
      stream.synchronize();
    }
  }
  timing::emit_samples("flagdnn", name, samples[0], measurements[0]);
  timing::emit_samples("cudnn", name, samples[1], measurements[1]);
}
}  // namespace flagdnn::testing::cuda
#endif
