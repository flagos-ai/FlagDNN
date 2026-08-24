/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

/* Ascend development-only runtime integration; never linked into the plugin. */

#include "validation/acl_runtime.hpp"
#include "validation/development_environment.hpp"

#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <acl/acl_rt.h>

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <algorithm>
#include <barrier>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

namespace acl = flagdnn::validation::ascend;
namespace fe = flagdnn_frontend;

constexpr std::size_t kElementCount = 1024;
constexpr std::int64_t kLeftUid = 7101;
constexpr std::int64_t kRightUid = 7102;
constexpr std::int64_t kOutputUid = 7103;
constexpr std::int64_t kIntermediateUid = 7104;
constexpr float kOutputSentinel = -1234.5F;

struct Options {
  std::string mode;
  std::string compiler_executable;
  std::string compiler_entry;
};

void require(bool condition, std::string message) {
  if (!condition) {
    throw std::runtime_error(std::move(message));
  }
}

void check_frontend(const fe::error_t& status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             status.get_message());
  }
}

void marker(std::string_view value) {
  std::cerr << value << '\n';
  std::cerr.flush();
}

constexpr std::size_t kMaximumCacheFiles = 4096;
constexpr std::uint64_t kMaximumCacheBytes = 1ULL << 30U;
constexpr std::size_t kMaximumCacheDepth = 64;

class FileDescriptor {
 public:
  explicit FileDescriptor(int value) noexcept : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0) {
      (void)::close(value_);
    }
  }

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  [[nodiscard]] int get() const noexcept { return value_; }

 private:
  int value_ = -1;
};

class DirectoryStream {
 public:
  explicit DirectoryStream(int descriptor) {
    value_ = ::fdopendir(descriptor);
    if (value_ == nullptr) {
      const int error = errno;
      (void)::close(descriptor);
      throw std::runtime_error("fdopendir failed while snapshotting the "
                               "Ascend cache: " +
                               std::string(std::strerror(error)));
    }
  }

  ~DirectoryStream() {
    if (value_ != nullptr) {
      (void)::closedir(value_);
    }
  }

  DirectoryStream(const DirectoryStream&) = delete;
  DirectoryStream& operator=(const DirectoryStream&) = delete;

  [[nodiscard]] DIR* get() const noexcept { return value_; }

 private:
  DIR* value_ = nullptr;
};

class Sha256Digest {
 public:
  void update(const char* data, std::size_t size) {
    total_size_ += static_cast<std::uint64_t>(size);
    while (size != 0) {
      const std::size_t available = block_.size() - block_size_;
      const std::size_t count = std::min(size, available);
      for (std::size_t index = 0; index < count; ++index) {
        block_[block_size_ + index] =
            static_cast<std::uint8_t>(data[index]);
      }
      block_size_ += count;
      data += count;
      size -= count;
      if (block_size_ == block_.size()) {
        transform(block_.data());
        block_size_ = 0;
      }
    }
  }

  [[nodiscard]] std::string finish() {
    const std::uint64_t bit_size = total_size_ * 8U;
    block_[block_size_++] = 0x80U;
    if (block_size_ > 56) {
      while (block_size_ < block_.size()) {
        block_[block_size_++] = 0;
      }
      transform(block_.data());
      block_size_ = 0;
    }
    while (block_size_ < 56) {
      block_[block_size_++] = 0;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
      block_[block_size_++] =
          static_cast<std::uint8_t>((bit_size >> shift) & 0xffU);
    }
    transform(block_.data());

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (const std::uint32_t value : state_) {
      output << std::setw(8) << value;
    }
    return output.str();
  }

 private:
  [[nodiscard]] static constexpr std::uint32_t rotate_right(
      std::uint32_t value,
      unsigned int amount) noexcept {
    return (value >> amount) | (value << (32U - amount));
  }

  void transform(const std::uint8_t* input) noexcept {
    static constexpr std::array<std::uint32_t, 64> kRoundConstants = {
        0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
        0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
        0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
        0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
        0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
        0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
        0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
        0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
        0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
        0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
        0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
        0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
        0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
        0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
        0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
        0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const std::size_t offset = index * 4;
      words[index] = (static_cast<std::uint32_t>(input[offset]) << 24U) |
                     (static_cast<std::uint32_t>(input[offset + 1]) << 16U) |
                     (static_cast<std::uint32_t>(input[offset + 2]) << 8U) |
                     static_cast<std::uint32_t>(input[offset + 3]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const std::uint32_t small_zero =
          rotate_right(words[index - 15], 7) ^
          rotate_right(words[index - 15], 18) ^
          (words[index - 15] >> 3U);
      const std::uint32_t small_one =
          rotate_right(words[index - 2], 17) ^
          rotate_right(words[index - 2], 19) ^
          (words[index - 2] >> 10U);
      words[index] = words[index - 16] + small_zero + words[index - 7] +
                     small_one;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sum_one =
          rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      const std::uint32_t choice = (e & f) ^ ((~e) & g);
      const std::uint32_t temporary_one =
          h + sum_one + choice + kRoundConstants[index] + words[index];
      const std::uint32_t sum_zero =
          rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temporary_two = sum_zero + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary_one;
      d = c;
      c = b;
      b = a;
      a = temporary_one + temporary_two;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_ = {
      0x6a09e667U,
      0xbb67ae85U,
      0x3c6ef372U,
      0xa54ff53aU,
      0x510e527fU,
      0x9b05688cU,
      0x1f83d9abU,
      0x5be0cd19U};
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_ = 0;
  std::uint64_t total_size_ = 0;
};

[[nodiscard]] bool same_time(const timespec& left,
                             const timespec& right) noexcept {
  return left.tv_sec == right.tv_sec && left.tv_nsec == right.tv_nsec;
}

struct CacheFileIdentity {
  std::uint64_t size = 0;
  std::string sha256;

  [[nodiscard]] bool operator==(const CacheFileIdentity&) const = default;
};

struct CacheSnapshot {
  std::uint64_t root_device = 0;
  std::uint64_t root_inode = 0;
  std::uint64_t total_bytes = 0;
  std::map<std::string, CacheFileIdentity> files;

  [[nodiscard]] bool operator==(const CacheSnapshot&) const = default;
};

void update_u64(Sha256Digest& digest, std::uint64_t value) {
  std::array<char, 8> encoded{};
  for (std::size_t index = 0; index < encoded.size(); ++index) {
    const unsigned int shift =
        static_cast<unsigned int>((encoded.size() - index - 1U) * 8U);
    encoded[index] = static_cast<char>((value >> shift) & 0xffU);
  }
  digest.update(encoded.data(), encoded.size());
}

[[nodiscard]] std::string cache_manifest_sha256(
    const CacheSnapshot& snapshot) {
  Sha256Digest digest;
  update_u64(digest, snapshot.root_device);
  update_u64(digest, snapshot.root_inode);
  update_u64(digest, snapshot.total_bytes);
  update_u64(digest, static_cast<std::uint64_t>(snapshot.files.size()));
  for (const auto& [path, identity] : snapshot.files) {
    update_u64(digest, static_cast<std::uint64_t>(path.size()));
    digest.update(path.data(), path.size());
    update_u64(digest, identity.size);
    digest.update(identity.sha256.data(), identity.sha256.size());
  }
  return digest.finish();
}

[[nodiscard]] CacheFileIdentity snapshot_regular_file(
    int directory,
    const char* name,
    const std::string& relative,
    const struct stat& path_status,
    std::uint64_t remaining_bytes) {
  if (path_status.st_size < 0 || path_status.st_nlink != 1 ||
      static_cast<std::uint64_t>(path_status.st_size) > remaining_bytes) {
    throw std::runtime_error(
        "Ascend cache regular file exceeds its bounded no-follow snapshot: " +
        relative);
  }
  const int descriptor =
      ::openat(directory,
               name,
               O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (descriptor < 0) {
    throw std::runtime_error("cannot open Ascend cache file " + relative +
                             " without following links: " +
                             std::string(std::strerror(errno)));
  }
  FileDescriptor owner(descriptor);
  struct stat before {};
  if (::fstat(owner.get(), &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_nlink != 1 || before.st_dev != path_status.st_dev ||
      before.st_ino != path_status.st_ino ||
      before.st_size != path_status.st_size) {
    throw std::runtime_error(
        "Ascend cache file changed while it was opened: " + relative);
  }

  Sha256Digest digest;
  std::array<char, 64U * 1024U> buffer{};
  std::uint64_t bytes_read = 0;
  for (;;) {
    const ssize_t count = ::read(owner.get(), buffer.data(), buffer.size());
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count < 0) {
      throw std::runtime_error("cannot read Ascend cache file " + relative +
                               ": " + std::strerror(errno));
    }
    if (count == 0) {
      break;
    }
    const std::uint64_t unsigned_count = static_cast<std::uint64_t>(count);
    if (unsigned_count >
        static_cast<std::uint64_t>(before.st_size) - bytes_read) {
      throw std::runtime_error(
          "Ascend cache file grew while it was hashed: " + relative);
    }
    digest.update(buffer.data(), static_cast<std::size_t>(count));
    bytes_read += unsigned_count;
  }
  if (bytes_read != static_cast<std::uint64_t>(before.st_size)) {
    throw std::runtime_error(
        "Ascend cache file was truncated while it was hashed: " + relative);
  }

  struct stat after {};
  if (::fstat(owner.get(), &after) != 0 || before.st_dev != after.st_dev ||
      before.st_ino != after.st_ino || before.st_size != after.st_size ||
      before.st_nlink != after.st_nlink ||
      !same_time(before.st_mtim, after.st_mtim) ||
      !same_time(before.st_ctim, after.st_ctim)) {
    throw std::runtime_error(
        "Ascend cache file changed while it was hashed: " + relative);
  }
  return {bytes_read, digest.finish()};
}

void snapshot_cache_directory(int directory,
                              std::string_view prefix,
                              std::size_t depth,
                              CacheSnapshot* snapshot) {
  if (depth > kMaximumCacheDepth) {
    throw std::runtime_error("Ascend cache directory nesting exceeds 64");
  }
  const int duplicate = ::fcntl(directory, F_DUPFD_CLOEXEC, 0);
  if (duplicate < 0) {
    throw std::runtime_error(
        "cannot duplicate Ascend cache directory descriptor: " +
        std::string(std::strerror(errno)));
  }
  struct stat directory_before {};
  if (::fstat(directory, &directory_before) != 0 ||
      !S_ISDIR(directory_before.st_mode)) {
    throw std::runtime_error(
        "Ascend cache directory changed before it was enumerated");
  }
  DirectoryStream entries(duplicate);
  for (;;) {
    errno = 0;
    dirent* entry = ::readdir(entries.get());
    if (entry == nullptr) {
      if (errno != 0) {
        throw std::runtime_error("cannot enumerate Ascend cache directory: " +
                                 std::string(std::strerror(errno)));
      }
      break;
    }
    const std::string_view name(entry->d_name);
    if (name == "." || name == "..") {
      continue;
    }
    const std::string relative =
        prefix.empty() ? std::string(name)
                       : std::string(prefix) + "/" + std::string(name);
    struct stat status {};
    if (::fstatat(directory, entry->d_name, &status, AT_SYMLINK_NOFOLLOW) != 0) {
      throw std::runtime_error("cannot stat Ascend cache entry " + relative +
                               ": " + std::strerror(errno));
    }
    if (S_ISDIR(status.st_mode)) {
      if (depth == kMaximumCacheDepth) {
        throw std::runtime_error("Ascend cache directory nesting exceeds 64");
      }
      const int child = ::openat(directory,
                                 entry->d_name,
                                 O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                     O_NOFOLLOW);
      if (child < 0) {
        throw std::runtime_error(
            "cannot open Ascend cache directory " + relative +
            " without following links: " + std::strerror(errno));
      }
      FileDescriptor child_owner(child);
      struct stat opened {};
      if (::fstat(child_owner.get(), &opened) != 0 ||
          !S_ISDIR(opened.st_mode) || opened.st_dev != status.st_dev ||
          opened.st_ino != status.st_ino) {
        throw std::runtime_error(
            "Ascend cache directory changed while it was opened: " +
            relative);
      }
      snapshot_cache_directory(
          child_owner.get(), relative, depth + 1U, snapshot);
      continue;
    }
    if (!S_ISREG(status.st_mode)) {
      throw std::runtime_error(
          "Ascend cache contains a link or non-regular entry: " + relative);
    }
    if (snapshot->files.size() >= kMaximumCacheFiles) {
      throw std::runtime_error("Ascend cache exceeds 4096 regular files");
    }
    CacheFileIdentity identity = snapshot_regular_file(
        directory,
        entry->d_name,
        relative,
        status,
        kMaximumCacheBytes - snapshot->total_bytes);
    snapshot->total_bytes += identity.size;
    const bool inserted =
        snapshot->files.emplace(relative, std::move(identity)).second;
    if (!inserted) {
      throw std::runtime_error(
          "Ascend cache snapshot contains a duplicate relative path: " +
          relative);
    }
  }
  struct stat directory_after {};
  if (::fstat(directory, &directory_after) != 0 ||
      !S_ISDIR(directory_after.st_mode) ||
      directory_before.st_dev != directory_after.st_dev ||
      directory_before.st_ino != directory_after.st_ino ||
      !same_time(directory_before.st_mtim, directory_after.st_mtim) ||
      !same_time(directory_before.st_ctim, directory_after.st_ctim)) {
    throw std::runtime_error(
        "Ascend cache directory changed while it was enumerated");
  }
}

[[nodiscard]] CacheSnapshot snapshot_cache_no_follow(
    const std::filesystem::path& root) {
  if (root.empty() || !root.is_absolute()) {
    throw std::runtime_error("Ascend cache snapshot root must be absolute");
  }
  const int descriptor =
      ::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  if (descriptor < 0) {
    throw std::runtime_error(
        "cannot open Ascend cache snapshot root without following links: " +
        std::string(std::strerror(errno)));
  }
  FileDescriptor owner(descriptor);
  struct stat status {};
  if (::fstat(owner.get(), &status) != 0 || !S_ISDIR(status.st_mode) ||
      status.st_uid != ::geteuid() ||
      (status.st_mode & 07777) != S_IRWXU) {
    throw std::runtime_error(
        "Ascend cache snapshot root is not an effective-UID-owned 0700 "
        "directory");
  }

  CacheSnapshot snapshot;
  snapshot.root_device = static_cast<std::uint64_t>(status.st_dev);
  snapshot.root_inode = static_cast<std::uint64_t>(status.st_ino);
  snapshot_cache_directory(owner.get(), "", 0, &snapshot);
  return snapshot;
}

[[nodiscard]] std::filesystem::path configured_triton_cache() {
  const char* value = std::getenv("TRITON_CACHE_DIR");
  if (value == nullptr || value[0] == '\0') {
    throw std::runtime_error(
        "TRITON_CACHE_DIR is unavailable for the resource cache snapshot");
  }
  return std::filesystem::path(value);
}

struct ResourceCacheSnapshots {
  CacheSnapshot graph;
  CacheSnapshot production;
};

void emit_cache_snapshot(std::string_view mode,
                         std::string_view phase,
                         std::string_view scope,
                         const CacheSnapshot& snapshot) {
  std::cout << "ASCEND_DEVELOPMENT_CACHE_SNAPSHOT mode=" << mode
            << " phase=" << phase << " scope=" << scope
            << " files=" << snapshot.files.size()
            << " bytes=" << snapshot.total_bytes
            << " manifest_sha256=" << cache_manifest_sha256(snapshot)
            << " root_device=" << snapshot.root_device
            << " root_inode=" << snapshot.root_inode << '\n';
  std::cout.flush();
}

[[nodiscard]] std::string escaped_cache_path(std::string_view path) {
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const unsigned char character : path) {
    if (character >= 0x20U && character <= 0x7eU && character != '\\') {
      output << static_cast<char>(character);
    } else {
      output << "\\x" << std::setw(2) << static_cast<unsigned int>(character);
    }
  }
  return output.str();
}

[[nodiscard]] std::string first_cache_difference(
    const CacheSnapshot& before,
    const CacheSnapshot& after) {
  auto before_it = before.files.begin();
  auto after_it = after.files.begin();
  while (before_it != before.files.end() || after_it != after.files.end()) {
    if (after_it == after.files.end() ||
        (before_it != before.files.end() &&
         before_it->first < after_it->first)) {
      return "removed:" + escaped_cache_path(before_it->first);
    }
    if (before_it == before.files.end() || after_it->first < before_it->first) {
      return "added:" + escaped_cache_path(after_it->first);
    }
    if (!(before_it->second == after_it->second)) {
      return "changed:" + escaped_cache_path(before_it->first);
    }
    ++before_it;
    ++after_it;
  }
  return "root-identity";
}

void require_cache_unchanged(std::string_view mode,
                             std::string_view scope,
                             const CacheSnapshot& before,
                             const CacheSnapshot& after) {
  if (!(before == after)) {
    throw std::runtime_error(
        "Ascend resource execute windows changed the no-follow cache snapshot "
        "for mode " +
        std::string(mode) + " scope " + std::string(scope) +
        ": difference=" +
        first_cache_difference(before, after) +
        " before_files=" + std::to_string(before.files.size()) +
        " after_files=" + std::to_string(after.files.size()) +
        " before_bytes=" + std::to_string(before.total_bytes) +
        " after_bytes=" + std::to_string(after.total_bytes));
  }
  std::cout << "ASCEND_DEVELOPMENT_CACHE_UNCHANGED PASS mode=" << mode
            << " scope=" << scope
            << " files=" << before.files.size()
            << " bytes=" << before.total_bytes
            << " manifest_sha256=" << cache_manifest_sha256(before) << '\n';
}

[[nodiscard]] ResourceCacheSnapshots snapshot_resource_caches(
    const acl::DevelopmentEnvironment& development) {
  const std::filesystem::path production = configured_triton_cache();
  require(development.graph_cache() != production,
          "Graph and production Triton cache roots must be distinct");
  ResourceCacheSnapshots result = {
      snapshot_cache_no_follow(development.graph_cache()),
      snapshot_cache_no_follow(production)};
  require(result.graph.root_device != result.production.root_device ||
              result.graph.root_inode != result.production.root_inode,
          "Graph and production Triton cache roots resolve to one directory");
  return result;
}

void emit_resource_cache_snapshots(
    std::string_view mode,
    std::string_view phase,
    const ResourceCacheSnapshots& snapshots) {
  emit_cache_snapshot(mode, phase, "graph", snapshots.graph);
  emit_cache_snapshot(mode, phase, "production", snapshots.production);
}

void require_resource_caches_unchanged(
    std::string_view mode,
    const ResourceCacheSnapshots& before,
    const ResourceCacheSnapshots& after) {
  require_cache_unchanged(mode, "graph", before.graph, after.graph);
  require_cache_unchanged(
      mode, "production", before.production, after.production);
}

class HostTemporaryDirectory {
 public:
  HostTemporaryDirectory() {
    std::array<char, 42> pattern{};
    constexpr std::string_view kTemplate =
        "/tmp/flagdnn-ascend-cache-snapshot-XXXXXX";
    static_assert(kTemplate.size() + 1U == pattern.size());
    std::copy(kTemplate.begin(), kTemplate.end(), pattern.begin());
    char* created = ::mkdtemp(pattern.data());
    if (created == nullptr) {
      throw std::runtime_error(
          "cannot create host cache-snapshot temporary directory: " +
          std::string(std::strerror(errno)));
    }
    path_ = created;
  }

  ~HostTemporaryDirectory() {
    if (cleaned_) {
      return;
    }
    std::error_code error;
    (void)std::filesystem::remove_all(path_, error);
    if (error) {
      std::cerr << "Ascend host cache-snapshot cleanup failed: "
                << error.message() << '\n';
    }
  }

  HostTemporaryDirectory(const HostTemporaryDirectory&) = delete;
  HostTemporaryDirectory& operator=(const HostTemporaryDirectory&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

  void cleanup() {
    std::error_code error;
    const std::uintmax_t removed =
        std::filesystem::remove_all(path_, error);
    if (error || removed == 0U) {
      throw std::runtime_error(
          "cannot remove host cache-snapshot temporary directory: " +
          (error ? error.message() : std::string("root was missing")));
    }
    cleaned_ = true;
  }

 private:
  std::filesystem::path path_;
  bool cleaned_ = false;
};

void write_host_snapshot_file(const std::filesystem::path& path,
                              std::string_view contents,
                              bool replace) {
  const int flags = replace ? O_WRONLY | O_TRUNC | O_CLOEXEC | O_NOFOLLOW
                            : O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC |
                                  O_NOFOLLOW;
  const int descriptor = ::open(path.c_str(), flags, S_IRUSR | S_IWUSR);
  if (descriptor < 0) {
    throw std::runtime_error("cannot write host cache-snapshot file: " +
                             std::string(std::strerror(errno)));
  }
  FileDescriptor owner(descriptor);
  std::size_t offset = 0;
  while (offset < contents.size()) {
    const ssize_t count = ::write(
        owner.get(), contents.data() + offset, contents.size() - offset);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      throw std::runtime_error("cannot complete host cache-snapshot write: " +
                               std::string(std::strerror(errno)));
    }
    offset += static_cast<std::size_t>(count);
  }
}

void test_resource_cache_snapshot_host() {
  HostTemporaryDirectory temporary;
  const std::filesystem::path nested = temporary.path() / "nested";
  std::error_code filesystem_error;
  require(std::filesystem::create_directory(nested, filesystem_error) &&
              !filesystem_error,
          "cannot create host cache-snapshot nested directory: " +
              filesystem_error.message());
  write_host_snapshot_file(temporary.path() / "alpha.bin", "abc", false);
  write_host_snapshot_file(nested / "beta.bin", "cache-snapshot", false);

  const CacheSnapshot before = snapshot_cache_no_follow(temporary.path());
  const CacheSnapshot identical = snapshot_cache_no_follow(temporary.path());
  require(before == identical,
          "two unchanged host cache snapshots were not identical");
  require(before.files.size() == 2U && before.total_bytes == 17U,
          "host cache snapshot returned the wrong file count or byte total");
  const auto alpha = before.files.find("alpha.bin");
  require(alpha != before.files.end() && alpha->second.size == 3U &&
              alpha->second.sha256 ==
                  "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "host cache snapshot SHA-256 implementation is incorrect");

  write_host_snapshot_file(temporary.path() / "alpha.bin", "xyz", true);
  const CacheSnapshot mutated = snapshot_cache_no_follow(temporary.path());
  bool mutation_detected = false;
  try {
    require_cache_unchanged(
        "resource_cache_snapshot_host", "fixture", before, mutated);
  } catch (const std::runtime_error&) {
    mutation_detected = true;
  }
  require(mutation_detected,
          "same-size host cache content mutation was not detected");

  const std::filesystem::path link = temporary.path() / "escape-link";
  require(::symlink("/etc/passwd", link.c_str()) == 0,
          "cannot create host cache-snapshot symlink fixture: " +
              std::string(std::strerror(errno)));
  bool symlink_rejected = false;
  try {
    (void)snapshot_cache_no_follow(temporary.path());
  } catch (const std::runtime_error&) {
    symlink_rejected = true;
  }
  require(symlink_rejected,
          "host cache no-follow snapshot accepted a symbolic link");
  temporary.cleanup();
  std::cout << "ASCEND_DEVELOPMENT_CACHE_SNAPSHOT_HOST PASS files="
            << before.files.size() << " bytes=" << before.total_bytes
            << " mutation_detected=1 symlink_rejected=1\n";
}

[[nodiscard]] aclrtContext current_context() {
  aclrtContext context = nullptr;
  acl::check_acl(aclrtGetCurrentContext(&context),
                 "aclrtGetCurrentContext(caller)");
  require(context != nullptr,
          "caller aclrtSetDevice did not establish a default context");
  return context;
}

void set_thread_context(aclrtContext context) {
  require(context != nullptr, "saved caller default context is null");
  acl::check_acl(aclrtSetCurrentContext(context),
                 "aclrtSetCurrentContext(caller thread)");
  require(current_context() == context,
          "caller thread did not retain the saved default context");
}

void configure_compiler(
    flagdnn::Handle& handle,
    const Options& options,
    const acl::DevelopmentEnvironment& development) {
  require(!options.compiler_executable.empty() &&
              !options.compiler_entry.empty(),
          "compiler executable and entry must not be empty");
  handle.set_compiler(options.compiler_executable,
                      options.compiler_entry,
                      development.graph_cache().string());
}

class AddProgram {
 public:
  explicit AddProgram(flagdnn::Handle& handle)
      : handle_(handle), graph_(std::make_shared<fe::graph::Graph>()) {
    graph_->set_name("ascend_development_runtime_add")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(false);

    const auto left = graph_->tensor(
        fe::graph::Tensor_attributes()
            .set_name("left")
            .set_uid(kLeftUid)
            .set_data_type(fe::DataType_t::FLOAT)
            .set_dim({static_cast<std::int64_t>(kElementCount)})
            .set_stride({1}));
    const auto right = graph_->tensor(
        fe::graph::Tensor_attributes()
            .set_name("right")
            .set_uid(kRightUid)
            .set_data_type(fe::DataType_t::FLOAT)
            .set_dim({static_cast<std::int64_t>(kElementCount)})
            .set_stride({1}));
    auto output = graph_->pointwise(
        left,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    output->set_name("output")
        .set_uid(kOutputUid)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({static_cast<std::int64_t>(kElementCount)})
        .set_stride({1})
        .set_output(true);

    check_frontend(graph_->build(handle_, {fe::HeurMode_t::A}),
                   "development Add graph build");
    std::int64_t workspace_size = -1;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "development Add workspace query");
    require(workspace_size == 0,
            "development Add unexpectedly requires Graph workspace");
  }

  [[nodiscard]] fe::error_t execute(
      std::span<const flagdnnBinding_t> bindings,
      flagdnnStream_t stream) const {
    return graph_->execute(handle_, bindings, nullptr, 0, stream);
  }

 private:
  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
};

class ChainedAddProgram {
 public:
  explicit ChainedAddProgram(flagdnn::Handle& handle)
      : handle_(handle), graph_(std::make_shared<fe::graph::Graph>()) {
    graph_->set_name("ascend_development_runtime_chained_add")
        .set_io_data_type(fe::DataType_t::FLOAT)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT)
        .set_autotune(false);
    const auto make_input = [this](const char* name, std::int64_t uid) {
      return graph_->tensor(
          fe::graph::Tensor_attributes()
              .set_name(name)
              .set_uid(uid)
              .set_data_type(fe::DataType_t::FLOAT)
              .set_dim({static_cast<std::int64_t>(kElementCount)})
              .set_stride({1}));
    };
    const auto left = make_input("left", kLeftUid);
    const auto right = make_input("right", kRightUid);
    auto intermediate = graph_->pointwise(
        left,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("first_add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    intermediate->set_name("intermediate")
        .set_uid(kIntermediateUid)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({static_cast<std::int64_t>(kElementCount)})
        .set_stride({1})
        .set_is_virtual(true);
    auto output = graph_->pointwise(
        intermediate,
        right,
        fe::graph::Pointwise_attributes()
            .set_name("second_add")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    output->set_name("output")
        .set_uid(kOutputUid)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({static_cast<std::int64_t>(kElementCount)})
        .set_stride({1})
        .set_output(true);

    check_frontend(graph_->build(handle_, {fe::HeurMode_t::A}),
                   "development chained Add graph build");
    std::int64_t workspace_size = -1;
    check_frontend(graph_->get_workspace_size(workspace_size),
                   "development chained Add workspace query");
    require(workspace_size > 0,
            "chained Add did not allocate virtual Graph workspace");
    workspace_size_ = static_cast<std::size_t>(workspace_size);
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept {
    return workspace_size_;
  }

  [[nodiscard]] fe::error_t execute(
      std::span<const flagdnnBinding_t> bindings,
      void* workspace,
      flagdnnStream_t stream) const {
    return graph_->execute(
        handle_, bindings, workspace, workspace_size_, stream);
  }

 private:
  flagdnn::Handle& handle_;
  std::shared_ptr<fe::graph::Graph> graph_;
  std::size_t workspace_size_ = 0;
};

class AddBuffers {
 public:
  explicit AddBuffers(acl::Stream& stream)
      : left_(byte_count()), right_(byte_count()), output_(byte_count()) {
    left_host_.resize(kElementCount);
    right_host_.resize(kElementCount);
    for (std::size_t index = 0; index < kElementCount; ++index) {
      const int left_value = static_cast<int>(index % 29U) - 14;
      const int right_value = static_cast<int>((index * 7U) % 31U) - 15;
      left_host_[index] = static_cast<float>(left_value) * 0.25F;
      right_host_[index] = static_cast<float>(right_value) * 0.125F;
    }
    left_.copy_from_host_at(
        left_host_.data(), byte_count(), 0, stream.get());
    right_.copy_from_host_at(
        right_host_.data(), byte_count(), 0, stream.get());
    fill_output(stream, kOutputSentinel);
  }

  [[nodiscard]] std::array<flagdnnBinding_t, 3> bindings() const {
    return {{{kLeftUid, left_.opaque()},
             {kRightUid, right_.opaque()},
             {kOutputUid, output_.opaque()}}};
  }

  void fill_output(acl::Stream& stream, float value) {
    const std::vector<float> initial(kElementCount, value);
    output_.copy_from_host_at(initial.data(), byte_count(), 0, stream.get());
    /* The caller-owned host storage must outlive the asynchronous copy. */
    stream.synchronize();
  }

  void require_add_output(acl::Stream& stream) const {
    require_add_output(stream, 1.0F);
  }

  void require_add_output(acl::Stream& stream, float right_scale) const {
    const std::vector<float> actual = read_output(stream);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const float expected =
          left_host_[index] + right_scale * right_host_[index];
      if (!std::isfinite(actual[index]) ||
          std::abs(actual[index] - expected) > 1.0e-6F) {
        throw std::runtime_error(
            "development Add output differs at element " +
            std::to_string(index));
      }
    }
  }

  void require_output_value(acl::Stream& stream, float expected) const {
    const std::vector<float> actual = read_output(stream);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      if (actual[index] != expected) {
        throw std::runtime_error(
            "rejected development execute changed output element " +
            std::to_string(index));
      }
    }
  }

 private:
  [[nodiscard]] static constexpr std::size_t byte_count() noexcept {
    return kElementCount * sizeof(float);
  }

  [[nodiscard]] std::vector<float> read_output(acl::Stream& stream) const {
    std::vector<float> result(kElementCount);
    output_.copy_to_host_at(result.data(), byte_count(), 0, stream.get());
    stream.synchronize();
    return result;
  }

  acl::DeviceBuffer left_;
  acl::DeviceBuffer right_;
  acl::DeviceBuffer output_;
  std::vector<float> left_host_;
  std::vector<float> right_host_;
};

class Notify {
 public:
  Notify() {
    acl::check_acl(aclrtCreateNotify(&value_, ACL_NOTIFY_DEFAULT),
                   "aclrtCreateNotify");
  }

  ~Notify() {
    if (value_ != nullptr) {
      const aclError status = aclrtDestroyNotify(value_);
      if (status != ACL_SUCCESS) {
        std::cerr << "Ascend development runtime cleanup: "
                  << acl::acl_error_message(status, "aclrtDestroyNotify")
                  << '\n';
      }
    }
  }

  Notify(const Notify&) = delete;
  Notify& operator=(const Notify&) = delete;

  [[nodiscard]] aclrtNotify get() const noexcept { return value_; }

 private:
  aclrtNotify value_ = nullptr;
};

void execute_and_require(AddProgram& program,
                         AddBuffers& buffers,
                         acl::Stream& stream,
                         int iterations) {
  require(iterations > 0, "development execute count must be positive");
  const auto bindings = buffers.bindings();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    check_frontend(program.execute(bindings, stream.opaque()),
                   "development Add execute");
  }
  stream.synchronize();
  buffers.require_add_output(stream);
}

void test_caller_ownership(aclrtContext saved_context) {
  acl::Stream stream;
  acl::DeviceBuffer buffer(sizeof(float));
  {
    flagdnn::Handle handle("ascend", 0);
    require(handle.backend_name() == "ascend",
            "development handle selected the wrong backend");
    require(!handle.target_fingerprint().empty(),
            "development handle returned an empty target fingerprint");
    require(current_context() == saved_context,
            "handle creation replaced the caller default context");
  }

  require(current_context() == saved_context,
          "handle destruction replaced the caller default context");
  constexpr float input = 19.25F;
  float output = 0.0F;
  buffer.copy_from_host_at(&input, sizeof(input), 0, stream.get());
  buffer.copy_to_host_at(&output, sizeof(output), 0, stream.get());
  stream.synchronize();
  require(output == input,
          "caller stream stopped working after handle destruction");
  std::cout << "ASCEND_DEVELOPMENT_CALLER_OWNERSHIP PASS\n";
}

struct ThreadOutcome {
  std::unique_ptr<flagdnn::Handle> handle;
  std::string error;
};

void test_concurrent_initialization(
    aclrtContext saved_context,
    const Options& options,
    const acl::DevelopmentEnvironment& development) {
  std::array<ThreadOutcome, 2> outcomes;
  std::barrier start_gate(2);
  const auto create_handle = [&](std::size_t index) {
    try {
      set_thread_context(saved_context);
    } catch (const std::exception& error) {
      outcomes[index].error = error.what();
    }
    /* Always reach the gate so one failed thread cannot strand its peer. */
    start_gate.arrive_and_wait();
    if (!outcomes[index].error.empty()) {
      return;
    }
    try {
      outcomes[index].handle =
          std::make_unique<flagdnn::Handle>("ascend", 0);
    } catch (const std::exception& error) {
      outcomes[index].error = error.what();
    }
  };
  marker("BEGIN_CONCURRENT_CREATE");
  std::array<std::thread, 2> threads = {
      std::thread(create_handle, 0), std::thread(create_handle, 1)};
  for (std::thread& thread : threads) {
    thread.join();
  }
  marker("END_CONCURRENT_CREATE");

  for (std::size_t index = 0; index < outcomes.size(); ++index) {
    require(outcomes[index].error.empty(),
            "concurrent handle " + std::to_string(index) +
                " failed: " + outcomes[index].error);
    require(outcomes[index].handle != nullptr,
            "concurrent handle creation returned null");
  }
  require(outcomes[0].handle->target_fingerprint() ==
              outcomes[1].handle->target_fingerprint(),
          "concurrent handles observed different Ascend domains");

  std::string progress_error;
  marker("BEGIN_CROSS_THREAD_RAW_PROGRESS");
  std::thread progress([&] {
    try {
      set_thread_context(saved_context);
      configure_compiler(*outcomes[1].handle, options, development);
      acl::Stream stream;
      AddBuffers buffers(stream);
      AddProgram program(*outcomes[1].handle);
      execute_and_require(program, buffers, stream, 2);
    } catch (const std::exception& error) {
      progress_error = error.what();
    }
  });
  progress.join();
  marker("END_CROSS_THREAD_RAW_PROGRESS");
  require(progress_error.empty(),
          "cross-thread raw progress failed after embedded Python released "
          "its initial GIL: " +
              progress_error);

  outcomes[1].handle.reset();
  outcomes[0].handle.reset();
  require(current_context() == saved_context,
          "concurrent handle teardown replaced caller context");
  std::cout << "ASCEND_DEVELOPMENT_CONCURRENCY PASS\n";
}

void test_cache(const Options& options,
                const acl::DevelopmentEnvironment& development) {
  flagdnn::Handle handle("ascend", 0);
  configure_compiler(handle, options, development);
  acl::Stream stream;
  AddBuffers buffers(stream);

  marker("BEGIN_CREATE");
  auto program = std::make_unique<AddProgram>(handle);
  marker("END_CREATE");
  marker("BEGIN_EXECUTE");
  const auto bindings = buffers.bindings();
  for (int iteration = 0; iteration < 16; ++iteration) {
    check_frontend(program->execute(bindings, stream.opaque()),
                   "cache-mode Add execute");
  }
  marker("END_EXECUTE");
  stream.synchronize();
  buffers.require_add_output(stream);
  std::cout << "ASCEND_DEVELOPMENT_CACHE PASS executes=16\n";
}

void test_repeated_lifecycle(
    const Options& options,
    const acl::DevelopmentEnvironment& development) {
  acl::Stream stream;
  AddBuffers buffers(stream);
  for (int cycle = 0; cycle < 4; ++cycle) {
    std::cout << "ASCEND_DEVELOPMENT_REPEAT_BEGIN cycle=" << cycle << '\n';
    {
      flagdnn::Handle handle("ascend", 0);
      configure_compiler(handle, options, development);
      AddProgram program(handle);
      execute_and_require(program, buffers, stream, 4);
    }
    constexpr float round_trip_input = 32.5F;
    float round_trip_output = 0.0F;
    acl::DeviceBuffer round_trip(sizeof(float));
    round_trip.copy_from_host_at(
        &round_trip_input, sizeof(float), 0, stream.get());
    round_trip.copy_to_host_at(
        &round_trip_output, sizeof(float), 0, stream.get());
    stream.synchronize();
    require(round_trip_output == round_trip_input,
            "caller stream failed after repeated handle teardown");
    std::cout << "ASCEND_DEVELOPMENT_REPEAT_END cycle=" << cycle << '\n';
  }
  std::cout << "ASCEND_DEVELOPMENT_REPEAT PASS cycles=4 executes=16\n";
}

void test_graph_workspace(
    const Options& options,
    const acl::DevelopmentEnvironment& development) {
  flagdnn::Handle handle("ascend", 0);
  configure_compiler(handle, options, development);
  acl::Stream stream;
  AddBuffers buffers(stream);
  ChainedAddProgram program(handle);
  acl::DeviceBuffer workspace(program.workspace_size());
  require(workspace.opaque() != nullptr,
          "chained Add workspace allocation returned null");
  require(reinterpret_cast<std::uintptr_t>(workspace.opaque()) % 256U == 0U,
          "chained Add workspace is not 256-byte aligned");
  const auto bindings = buffers.bindings();
  check_frontend(program.execute(bindings, workspace.opaque(), stream.opaque()),
                 "development chained Add execute");
  stream.synchronize();
  buffers.require_add_output(stream, 2.0F);
  std::cout << "ASCEND_DEVELOPMENT_WORKSPACE PASS bytes="
            << program.workspace_size() << " stages=2\n";
}

struct MemorySample {
  std::uint64_t rss_bytes = 0;
  std::uint64_t hbm_free_bytes = 0;
  std::uint64_t hbm_total_bytes = 0;
  std::uint64_t hbm_used_bytes = 0;
};

[[nodiscard]] std::uint64_t resident_set_bytes() {
  std::ifstream status("/proc/self/status");
  require(static_cast<bool>(status), "cannot read /proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (!line.starts_with("VmRSS:")) {
      continue;
    }
    std::istringstream fields(line.substr(6));
    std::uint64_t kibibytes = 0;
    std::string unit;
    fields >> kibibytes >> unit;
    require(static_cast<bool>(fields) && unit == "kB" &&
                kibibytes <=
                    std::numeric_limits<std::uint64_t>::max() / 1024U,
            "invalid VmRSS entry in /proc/self/status");
    return kibibytes * 1024U;
  }
  throw std::runtime_error("/proc/self/status has no VmRSS entry");
}

[[nodiscard]] MemorySample memory_sample() {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  acl::check_acl(aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes),
                 "aclrtGetMemInfo(ACL_HBM_MEM)");
  require(free_bytes <= total_bytes,
          "aclrtGetMemInfo returned free HBM greater than total HBM");
  return MemorySample{resident_set_bytes(),
                      static_cast<std::uint64_t>(free_bytes),
                      static_cast<std::uint64_t>(total_bytes),
                      static_cast<std::uint64_t>(total_bytes - free_bytes)};
}

void require_growth_budget(
    std::string_view name,
    const std::array<std::uint64_t, 6>& values) {
  constexpr long double kFourMebibytes = 4.0L * 1024.0L * 1024.0L;
  constexpr long double kSlopeLimit = 256.0L * 1024.0L;
  const long double baseline =
      (static_cast<long double>(values[1]) +
       static_cast<long double>(values[2])) /
      2.0L;
  const long double ending =
      (static_cast<long double>(values[4]) +
       static_cast<long double>(values[5])) /
      2.0L;
  const long double growth = std::max(0.0L, ending - baseline);
  const long double growth_limit =
      std::max(kFourMebibytes, 0.005L * baseline);

  long double mean = 0.0L;
  for (std::size_t index = 1; index < values.size(); ++index) {
    mean += static_cast<long double>(values[index]);
  }
  mean /= 5.0L;
  long double numerator = 0.0L;
  long double denominator = 0.0L;
  for (std::size_t index = 1; index < values.size(); ++index) {
    const long double window = static_cast<long double>(index + 1U);
    const long double centered_window = window - 4.0L;
    numerator += centered_window *
                 (static_cast<long double>(values[index]) - mean);
    denominator += centered_window * centered_window;
  }
  require(denominator > 0.0L, "resource slope denominator is zero");
  const long double slope = std::max(0.0L, numerator / denominator);
  require(growth <= growth_limit,
          std::string(name) + " resource growth exceeds the budget");
  require(slope <= kSlopeLimit,
          std::string(name) + " resource slope exceeds 256 KiB/window");
  std::cout << "ASCEND_DEVELOPMENT_RESOURCE_METRIC name=" << name
            << " baseline=" << static_cast<std::uint64_t>(baseline)
            << " ending=" << static_cast<std::uint64_t>(ending)
            << " growth=" << static_cast<std::uint64_t>(growth)
            << " growth_limit="
            << static_cast<std::uint64_t>(growth_limit)
            << " slope_per_window="
            << static_cast<std::uint64_t>(slope) << '\n';
}

void require_graph_lifecycle_growth_budget(
    std::string_view name,
    const std::array<std::uint64_t, 10>& values) {
  constexpr long double kEightMebibytes = 8.0L * 1024.0L * 1024.0L;
  const long double baseline =
      (static_cast<long double>(values[0]) +
       static_cast<long double>(values[1])) /
      2.0L;
  const long double ending =
      (static_cast<long double>(values[8]) +
       static_cast<long double>(values[9])) /
      2.0L;
  const long double growth = std::max(0.0L, ending - baseline);
  const long double growth_limit =
      std::max(kEightMebibytes, 0.01L * baseline);
  require(growth <= growth_limit,
          std::string(name) +
              " Graph lifecycle resource growth exceeds the budget");
  std::cout << "ASCEND_DEVELOPMENT_GRAPH_RESOURCE_METRIC name=" << name
            << " baseline=" << static_cast<std::uint64_t>(baseline)
            << " ending=" << static_cast<std::uint64_t>(ending)
            << " growth=" << static_cast<std::uint64_t>(growth)
            << " growth_limit="
            << static_cast<std::uint64_t>(growth_limit) << '\n';
}

void test_resource_budget(
    const Options& options,
    const acl::DevelopmentEnvironment& development) {
  constexpr int kWarmupIterations = 1000;
  constexpr int kWindowCount = 6;
  constexpr int kIterationsPerWindow = 10000;
  flagdnn::Handle handle("ascend", 0);
  configure_compiler(handle, options, development);
  acl::Stream stream;
  AddBuffers buffers(stream);
  marker("BEGIN_CREATE");
  AddProgram program(handle);
  marker("END_CREATE");
  const auto bindings = buffers.bindings();
  marker("BEGIN_EXECUTE");
  for (int iteration = 0; iteration < kWarmupIterations; ++iteration) {
    check_frontend(program.execute(bindings, stream.opaque()),
                   "resource warmup Add execute");
  }
  stream.synchronize();
  const ResourceCacheSnapshots cache_before =
      snapshot_resource_caches(development);
  emit_resource_cache_snapshots(
      "resource_budget", "before", cache_before);

  std::array<MemorySample, kWindowCount> samples{};
  for (int window = 0; window < kWindowCount; ++window) {
    for (int iteration = 0; iteration < kIterationsPerWindow; ++iteration) {
      check_frontend(program.execute(bindings, stream.opaque()),
                     "resource window Add execute");
    }
    stream.synchronize();
    samples[static_cast<std::size_t>(window)] = memory_sample();
    const MemorySample& sample = samples[static_cast<std::size_t>(window)];
    std::cout << "ASCEND_DEVELOPMENT_RESOURCE_SAMPLE window=" << window + 1
              << " rss=" << sample.rss_bytes
              << " hbm_free=" << sample.hbm_free_bytes
              << " hbm_total=" << sample.hbm_total_bytes
              << " hbm_used=" << sample.hbm_used_bytes << '\n';
  }
  const ResourceCacheSnapshots cache_after =
      snapshot_resource_caches(development);
  emit_resource_cache_snapshots("resource_budget", "after", cache_after);
  require_resource_caches_unchanged(
      "resource_budget", cache_before, cache_after);
  buffers.require_add_output(stream);
  marker("END_EXECUTE");

  std::array<std::uint64_t, kWindowCount> rss{};
  std::array<std::uint64_t, kWindowCount> hbm{};
  for (std::size_t index = 0; index < samples.size(); ++index) {
    rss[index] = samples[index].rss_bytes;
    hbm[index] = samples[index].hbm_used_bytes;
  }
  require_growth_budget("rss", rss);
  require_growth_budget("hbm", hbm);
  std::cout << "ASCEND_DEVELOPMENT_RESOURCE_BUDGET PASS warmup="
            << kWarmupIterations << " windows=" << kWindowCount
            << " iterations_per_window=" << kIterationsPerWindow << '\n';
}

void test_resource_graph_lifecycle(
    const Options& options,
    const acl::DevelopmentEnvironment& development) {
  constexpr int kWarmupLifecycles = 20;
  constexpr int kMeasuredLifecycles = 100;
  constexpr int kExecutionsPerLifecycle = 4;
  constexpr int kSampleInterval = 10;
  constexpr std::size_t kSampleCount =
      static_cast<std::size_t>(kMeasuredLifecycles / kSampleInterval);
  static_assert(kMeasuredLifecycles % kSampleInterval == 0);

  flagdnn::Handle handle("ascend", 0);
  configure_compiler(handle, options, development);
  acl::Stream stream;
  AddBuffers buffers(stream);
  const auto bindings = buffers.bindings();

  const auto run_lifecycle = [&](bool first_lifecycle) {
    if (first_lifecycle) {
      marker("BEGIN_CREATE");
    }
    {
      AddProgram program(handle);
      if (first_lifecycle) {
        marker("END_CREATE");
        marker("BEGIN_EXECUTE");
      }
      for (int execution = 0; execution < kExecutionsPerLifecycle;
           ++execution) {
        check_frontend(program.execute(bindings, stream.opaque()),
                       "Graph lifecycle resource Add execute");
      }
      stream.synchronize();
    }
  };

  run_lifecycle(true);
  for (int lifecycle = 1; lifecycle < kWarmupLifecycles; ++lifecycle) {
    run_lifecycle(false);
  }
  const ResourceCacheSnapshots cache_before =
      snapshot_resource_caches(development);
  emit_resource_cache_snapshots(
      "resource_graph_lifecycle", "before", cache_before);

  std::array<MemorySample, kSampleCount> samples{};
  for (int lifecycle = 0; lifecycle < kMeasuredLifecycles; ++lifecycle) {
    run_lifecycle(false);
    if ((lifecycle + 1) % kSampleInterval != 0) {
      continue;
    }
    const std::size_t sample_index =
        static_cast<std::size_t>((lifecycle + 1) / kSampleInterval - 1);
    samples[sample_index] = memory_sample();
    const MemorySample& sample = samples[sample_index];
    std::cout << "ASCEND_DEVELOPMENT_GRAPH_RESOURCE_SAMPLE sample="
              << sample_index + 1U
              << " completed_lifecycles=" << lifecycle + 1
              << " rss=" << sample.rss_bytes
              << " hbm_free=" << sample.hbm_free_bytes
              << " hbm_total=" << sample.hbm_total_bytes
              << " hbm_used=" << sample.hbm_used_bytes << '\n';
  }
  const ResourceCacheSnapshots cache_after =
      snapshot_resource_caches(development);
  emit_resource_cache_snapshots(
      "resource_graph_lifecycle", "after", cache_after);
  require_resource_caches_unchanged(
      "resource_graph_lifecycle", cache_before, cache_after);
  marker("END_EXECUTE");
  buffers.require_add_output(stream);

  std::array<std::uint64_t, kSampleCount> rss{};
  std::array<std::uint64_t, kSampleCount> hbm{};
  for (std::size_t index = 0; index < samples.size(); ++index) {
    rss[index] = samples[index].rss_bytes;
    hbm[index] = samples[index].hbm_used_bytes;
  }
  require_graph_lifecycle_growth_budget("rss", rss);
  require_graph_lifecycle_growth_budget("hbm", hbm);
  std::cout
      << "ASCEND_DEVELOPMENT_RESOURCE_GRAPH_LIFECYCLE_BUDGET PASS "
      << "warmup_lifecycles=" << kWarmupLifecycles
      << " measured_lifecycles=" << kMeasuredLifecycles
      << " executes_per_lifecycle=" << kExecutionsPerLifecycle
      << " sample_interval=" << kSampleInterval << '\n';
}

void test_resource_dual_handle(
    const Options& options,
    const acl::DevelopmentEnvironment& development) {
  constexpr int kWarmupPerHandle = 1000;
  constexpr int kWindowCount = 6;
  constexpr int kExecutionsPerHandlePerWindow = 5000;

  flagdnn::Handle first_handle("ascend", 0);
  configure_compiler(first_handle, options, development);
  flagdnn::Handle second_handle("ascend", 0);
  configure_compiler(second_handle, options, development);
  acl::Stream stream;
  AddBuffers buffers(stream);
  const auto bindings = buffers.bindings();

  marker("BEGIN_CREATE");
  AddProgram first_program(first_handle);
  AddProgram second_program(second_handle);
  marker("END_CREATE");
  marker("BEGIN_EXECUTE");
  for (int iteration = 0; iteration < kWarmupPerHandle; ++iteration) {
    check_frontend(first_program.execute(bindings, stream.opaque()),
                   "first-handle resource warmup Add execute");
    check_frontend(second_program.execute(bindings, stream.opaque()),
                   "second-handle resource warmup Add execute");
  }
  stream.synchronize();
  const ResourceCacheSnapshots cache_before =
      snapshot_resource_caches(development);
  emit_resource_cache_snapshots(
      "resource_dual_handle", "before", cache_before);

  std::array<MemorySample, kWindowCount> samples{};
  for (int window = 0; window < kWindowCount; ++window) {
    for (int iteration = 0;
         iteration < kExecutionsPerHandlePerWindow;
         ++iteration) {
      check_frontend(first_program.execute(bindings, stream.opaque()),
                     "first-handle resource window Add execute");
      check_frontend(second_program.execute(bindings, stream.opaque()),
                     "second-handle resource window Add execute");
    }
    stream.synchronize();
    samples[static_cast<std::size_t>(window)] = memory_sample();
    const MemorySample& sample = samples[static_cast<std::size_t>(window)];
    std::cout << "ASCEND_DEVELOPMENT_DUAL_HANDLE_RESOURCE_SAMPLE window="
              << window + 1
              << " rss=" << sample.rss_bytes
              << " hbm_free=" << sample.hbm_free_bytes
              << " hbm_total=" << sample.hbm_total_bytes
              << " hbm_used=" << sample.hbm_used_bytes << '\n';
  }
  const ResourceCacheSnapshots cache_after =
      snapshot_resource_caches(development);
  emit_resource_cache_snapshots(
      "resource_dual_handle", "after", cache_after);
  require_resource_caches_unchanged(
      "resource_dual_handle", cache_before, cache_after);
  marker("END_EXECUTE");
  buffers.require_add_output(stream);

  std::array<std::uint64_t, kWindowCount> rss{};
  std::array<std::uint64_t, kWindowCount> hbm{};
  for (std::size_t index = 0; index < samples.size(); ++index) {
    rss[index] = samples[index].rss_bytes;
    hbm[index] = samples[index].hbm_used_bytes;
  }
  require_growth_budget("dual_handle_rss", rss);
  require_growth_budget("dual_handle_hbm", hbm);
  std::cout << "ASCEND_DEVELOPMENT_RESOURCE_DUAL_HANDLE_BUDGET PASS "
            << "handles=2 warmup_per_handle=" << kWarmupPerHandle
            << " windows=" << kWindowCount
            << " executes_per_handle_per_window="
            << kExecutionsPerHandlePerWindow << '\n';
}

void test_environment_drift(
    const Options& options,
    const acl::DevelopmentEnvironment& development) {
  flagdnn::Handle handle("ascend", 0);
  configure_compiler(handle, options, development);
  acl::Stream stream;
  AddBuffers buffers(stream);
  AddProgram program(handle);
  execute_and_require(program, buffers, stream, 1);

  buffers.fill_output(stream, kOutputSentinel);
  stream.synchronize();
  require(::setenv("TRITON_ALL_BLOCKS_PARALLEL", "true", 1) == 0,
          "cannot inject frozen Ascend environment drift");

  /*
   * A prepared raw launch is intentionally independent of the compiler
   * environment.  Scanning getenv/stat state on every execute would put
   * control-plane work on the steady-state hot path.  Enter a
   * configuration-sensitive operation instead: constructing another Ascend
   * handle must observe the frozen-domain drift and latch the process before
   * any further raw launch can start.
   */
  marker("BEGIN_DRIFT_REJECT");
  flagdnnHandle_t rejected = nullptr;
  const flagdnnStatus_t create_status =
      flagdnnCreateWithBackendName("ascend", 0, &rejected);
  const std::string create_error = flagdnnGetLastErrorString();
  if (rejected != nullptr) {
    (void)flagdnnDestroy(rejected);
  }
  marker("END_DRIFT_REJECT");
  require(create_status == FLAGDNN_STATUS_NOT_SUPPORTED &&
              rejected == nullptr,
          "environment drift allowed another handle: " + create_error);
  require(create_error.find("configuration changed") !=
              std::string::npos,
          "environment drift returned an unrelated diagnostic: " +
              create_error);

  const auto bindings = buffers.bindings();
  const fe::error_t first = program.execute(bindings, stream.opaque());
  require(first.get_status() == FLAGDNN_STATUS_NOT_SUPPORTED,
          "terminal environment drift did not reject execute: " +
              first.get_message());
  stream.synchronize();
  buffers.require_output_value(stream, kOutputSentinel);

  const fe::error_t latched = program.execute(bindings, stream.opaque());
  require(latched.get_status() == FLAGDNN_STATUS_NOT_SUPPORTED,
          "terminal environment drift was not latched for execute");
  buffers.require_output_value(stream, kOutputSentinel);
  std::cout << "ASCEND_DEVELOPMENT_ENVIRONMENT_DRIFT PASS "
               "raw_output_unchanged=1 terminal_latched=1\n";
}

void test_no_active_synchronization(
    aclrtContext saved_context,
    const Options& options,
    const acl::DevelopmentEnvironment& development,
    bool use_default_stream) {
  flagdnn::Handle handle("ascend", 0);
  configure_compiler(handle, options, development);
  acl::Stream caller_stream;
  acl::Stream release_stream;
  AddBuffers buffers(caller_stream);
  AddProgram program(handle);
  aclrtStream blocked_stream = caller_stream.get();
  if (use_default_stream) {
    acl::check_acl(aclrtCtxGetCurrentDefaultStream(&blocked_stream),
                   "aclrtCtxGetCurrentDefaultStream(no-active-sync)");
    require(blocked_stream != nullptr,
            "caller context returned a null default stream");
  }
  Notify release_notify;
  acl::check_acl(
      aclrtWaitAndResetNotify(
          release_notify.get(), blocked_stream, 10000),
      "aclrtWaitAndResetNotify(caller stream)");

  struct ExecuteResult {
    flagdnnStatus_t status = FLAGDNN_STATUS_INTERNAL_ERROR;
    std::string message;
  } result;
  std::promise<void> returned;
  std::future<void> returned_future = returned.get_future();
  const auto bindings = buffers.bindings();
  marker("BEGIN_ASYNC_EXECUTE");
  std::thread submitter([&] {
    try {
      set_thread_context(saved_context);
      const fe::error_t status = program.execute(
          bindings, use_default_stream ? nullptr : caller_stream.opaque());
      result.status = status.get_status();
      result.message = status.get_message();
    } catch (const std::exception& error) {
      result.status = FLAGDNN_STATUS_INTERNAL_ERROR;
      result.message = error.what();
    }
    returned.set_value();
  });

  const bool returned_without_release =
      returned_future.wait_for(std::chrono::seconds(2)) ==
      std::future_status::ready;
  const aclError release_status =
      aclrtRecordNotify(release_notify.get(), release_stream.get());
  submitter.join();
  marker("END_ASYNC_EXECUTE");

  acl::check_acl(release_status,
                 "aclrtRecordNotify(release caller stream)");
  acl::check_acl(aclrtSynchronizeStream(blocked_stream),
                 "aclrtSynchronizeStream(blocked caller stream)");
  require(result.status == FLAGDNN_STATUS_SUCCESS,
          "asynchronous execute failed: " + result.message);
  buffers.require_add_output(caller_stream);
  require(returned_without_release,
          "flagdnnExecuteAsync waited for work already queued on the caller "
          "stream");
  std::cout << "ASCEND_DEVELOPMENT_NO_ACTIVE_SYNC PASS timeout_ms=2000 stream="
            << (use_default_stream ? "default" : "explicit") << '\n';
}

[[nodiscard]] bool known_mode(std::string_view mode) {
  return mode == "ownership" || mode == "concurrency" || mode == "cache" ||
         mode == "repeat" || mode == "workspace" ||
         mode == "resource_budget" ||
         mode == "resource_graph_lifecycle" ||
         mode == "resource_dual_handle" ||
         mode == "resource_cache_snapshot_host" ||
         mode == "environment_drift" ||
         mode == "no_active_sync" || mode == "no_active_sync_default" ||
         mode == "async";
}

void dispatch(const Options& options,
              aclrtContext saved_context,
              const acl::DevelopmentEnvironment& development) {
  if (options.mode == "ownership") {
    test_caller_ownership(saved_context);
  } else if (options.mode == "concurrency") {
    test_concurrent_initialization(saved_context, options, development);
  } else if (options.mode == "cache") {
    test_cache(options, development);
  } else if (options.mode == "repeat") {
    test_repeated_lifecycle(options, development);
  } else if (options.mode == "workspace") {
    test_graph_workspace(options, development);
  } else if (options.mode == "resource_budget") {
    test_resource_budget(options, development);
  } else if (options.mode == "resource_graph_lifecycle") {
    test_resource_graph_lifecycle(options, development);
  } else if (options.mode == "resource_dual_handle") {
    test_resource_dual_handle(options, development);
  } else if (options.mode == "environment_drift") {
    test_environment_drift(options, development);
  } else if (options.mode == "no_active_sync" || options.mode == "async") {
    test_no_active_synchronization(
        saved_context, options, development, false);
  } else if (options.mode == "no_active_sync_default") {
    test_no_active_synchronization(
        saved_context, options, development, true);
  } else {
    throw std::invalid_argument("unknown development runtime mode");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: " << argv[0]
              << " MODE COMPILER_EXECUTABLE COMPILER_ENTRY\n"
              << "modes: ownership concurrency cache repeat workspace "
                 "resource_budget resource_graph_lifecycle "
                 "resource_dual_handle "
                 "resource_cache_snapshot_host "
                 "environment_drift no_active_sync "
                 "no_active_sync_default\n";
    return 2;
  }

  const Options options = {argv[1], argv[2], argv[3]};
  if (!known_mode(options.mode)) {
    std::cerr << "unknown development runtime mode: " << options.mode
              << '\n';
    return 2;
  }

  try {
    if (options.mode == "resource_cache_snapshot_host") {
      test_resource_cache_snapshot_host();
      return 0;
    }
    acl::DevelopmentEnvironment development("runtime-" + options.mode);
    acl::AclRuntime runtime;
    const std::string soc = acl::soc_name();
    development.prepare_target(soc);
    const aclrtContext saved_context = current_context();
    std::cout << "ASCEND_DEVELOPMENT_RUNTIME_BEGIN mode=" << options.mode
              << " soc=" << soc
              << " cann_runtime=" << acl::runtime_package_version()
              << " scope=development\n";
    dispatch(options, saved_context, development);
    require(current_context() == saved_context,
            "runtime mode replaced the caller default context");
    runtime.finalize();
    development.cleanup();
    std::cout << "ASCEND_DEVELOPMENT_RUNTIME PASS mode=" << options.mode
              << " caller_acl_reference_count=0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ASCEND_DEVELOPMENT_RUNTIME FAILED mode=" << options.mode
              << " reason=" << error.what() << '\n';
    return 1;
  }
}
