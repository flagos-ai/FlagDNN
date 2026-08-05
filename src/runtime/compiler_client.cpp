/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "runtime/compiler_client.hpp"

#include "error.hpp"
#include "runtime/context.hpp"

#include <spawn.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <charconv>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

extern char** environ;

namespace flagdnn::native {
namespace {

std::atomic<std::uint64_t> identity_temporary_counter{0};

std::chrono::seconds compiler_timeout() {
  constexpr std::uint64_t default_seconds = 1800;
  constexpr std::uint64_t maximum_seconds = 86400;
  const char* configured = std::getenv("FLAGDNN_COMPILER_TIMEOUT_SECONDS");
  if (configured == nullptr || configured[0] == '\0') {
    return std::chrono::seconds(default_seconds);
  }
  const std::string_view value(configured);
  std::uint64_t seconds = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), seconds);
  if (error != std::errc{} || end != value.data() + value.size() ||
      seconds == 0 || seconds > maximum_seconds) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "FLAGDNN_COMPILER_TIMEOUT_SECONDS must be an integer in [1, 86400]");
  }
  return std::chrono::seconds(seconds);
}

bool is_sha256(std::string_view value) {
  if (value.size() != 64) {
    return false;
  }
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

class TemporaryFile {
 public:
  explicit TemporaryFile(std::filesystem::path path)
      : path_(std::move(path)) {}

  ~TemporaryFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

 private:
  std::filesystem::path path_;
};

void run_compiler_process(const RuntimeContext& context,
                          const std::vector<std::string>& owned_arguments,
                          std::string_view action) {
  if (context.compiler_executable().empty() || context.compiler().empty()) {
    throw ApiError(
        FLAGDNN_STATUS_COMPILATION_FAILED,
        "external compiler is not configured; call flagdnnSetCompilerConfig");
  }
  if (owned_arguments.size() < 2 ||
      owned_arguments[0] != context.compiler_executable() ||
      owned_arguments[1] != context.compiler()) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "external compiler argument construction is invalid");
  }

  std::vector<char*> arguments;
  arguments.reserve(owned_arguments.size() + 1);
  for (const std::string& argument : owned_arguments) {
    arguments.push_back(const_cast<char*>(argument.c_str()));
  }
  arguments.push_back(nullptr);

  const std::chrono::seconds timeout = compiler_timeout();
  posix_spawnattr_t spawn_attributes;
  int attribute_result = posix_spawnattr_init(&spawn_attributes);
  if (attribute_result != 0) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cannot initialize external compiler process attributes: " +
                       std::string(std::strerror(attribute_result)));
  }
  attribute_result = posix_spawnattr_setflags(
      &spawn_attributes, POSIX_SPAWN_SETPGROUP);
  if (attribute_result == 0) {
    // A process-group value of zero creates a group whose ID is the child PID.
    // This lets timeout handling terminate Python plus ptxas/other descendants.
    attribute_result = posix_spawnattr_setpgroup(&spawn_attributes, 0);
  }
  if (attribute_result != 0) {
    (void)posix_spawnattr_destroy(&spawn_attributes);
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cannot configure external compiler process group: " +
                       std::string(std::strerror(attribute_result)));
  }

  pid_t child = -1;
  const int spawn_result = posix_spawnp(&child,
                                        arguments[0],
                                        nullptr,
                                        &spawn_attributes,
                                        arguments.data(),
                                        environ);
  (void)posix_spawnattr_destroy(&spawn_attributes);
  if (spawn_result != 0) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cannot start external compiler for " +
                       std::string(action) + ": " +
                       std::string(std::strerror(spawn_result)));
  }

  const auto deadline = std::chrono::steady_clock::now() + timeout;
  int status = 0;
  pid_t wait_result = -1;
  for (;;) {
    wait_result = waitpid(child, &status, WNOHANG);
    if (wait_result == child) {
      break;
    }
    if (wait_result < 0 && errno != EINTR) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      (void)kill(-child, SIGKILL);
      do {
        wait_result = waitpid(child, &status, 0);
      } while (wait_result < 0 && errno == EINTR);
      throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                     "external compiler " + std::string(action) +
                         " timed out after " +
                         std::to_string(timeout.count()) +
                         " seconds");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (wait_result != child) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "waitpid failed for external compiler: " +
                       std::string(std::strerror(errno)));
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::ostringstream message;
    message << "external compiler " << action << " failed";
    if (WIFEXITED(status)) {
      message << " with exit code " << WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      message << " with signal " << WTERMSIG(status);
    }
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED, message.str());
  }
}

std::string read_compiler_identity(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "external compiler did not produce its identity");
  }
  std::string value((std::istreambuf_iterator<char>(input)),
                    std::istreambuf_iterator<char>());
  if (input.bad() || value.size() > 128) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "compiler identity file is invalid");
  }
  while (!value.empty() &&
         (value.back() == '\n' || value.back() == '\r')) {
    value.pop_back();
  }
  if (!is_sha256(value)) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "external compiler returned an invalid identity");
  }
  return value;
}

}  // namespace

std::string query_compiler_identity(
    RuntimeContext& context,
    const std::filesystem::path& graph_cache_directory) {
  const std::uint64_t serial =
      identity_temporary_counter.fetch_add(1);
  const std::filesystem::path identity_output =
      graph_cache_directory /
      (".identity.tmp." + std::to_string(getpid()) + "." +
       std::to_string(serial));
  TemporaryFile cleanup(identity_output);

  run_compiler_process(context,
                       {context.compiler_executable(),
                        context.compiler(),
                        "--identify",
                        "--backend",
                        context.backend_name(),
                        "--target",
                        context.target_fingerprint(),
                        "--execution-engine",
                        context.execution_engine(),
                        "--identity-output",
                        identity_output.string(),
                        "--quiet"},
                       "identity query");

  return read_compiler_identity(identity_output);
}

void compile_external_artifact(
    const RuntimeContext& context,
    const std::filesystem::path& request,
    const std::filesystem::path& output_directory) {
  run_compiler_process(context,
                       {context.compiler_executable(),
                        context.compiler(),
                        "--request",
                        request.string(),
                        "--output-dir",
                        output_directory.string(),
                        "--execution-engine",
                        context.execution_engine(),
                        "--quiet"},
                       "compile");
}

}  // namespace flagdnn::native
