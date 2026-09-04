/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#pragma once

#include <mudnn.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::validation::mthreads {
namespace mudnn_workspace_detail {

constexpr std::size_t kAllocationAlignment = 256;

inline std::size_t checked_add(std::size_t left,
                               std::size_t right,
                               std::string_view operation) {
  if (left > std::numeric_limits<std::size_t>::max() - right) {
    throw std::overflow_error(
        std::string(operation) + " workspace arithmetic overflows");
  }
  return left + right;
}

inline std::size_t align_up(std::size_t value,
                            std::string_view operation) {
  const std::size_t remainder = value % kAllocationAlignment;
  return remainder == 0
             ? value
             : checked_add(value,
                           kAllocationAlignment - remainder,
                           operation);
}

class Arena final : public std::enable_shared_from_this<Arena> {
 public:
  Arena(void* pointer, std::size_t size, std::string operation)
      : pointer_(static_cast<std::byte*>(pointer)),
        size_(size),
        operation_(std::move(operation)) {
    if (size_ != 0 && pointer_ == nullptr) {
      throw std::invalid_argument(operation_ + " workspace arena is null");
    }
    if (size_ != 0) {
      free_blocks_.push_back({0, size_});
    }
  }

  musa::dnn::MemoryHandler allocate(std::size_t requested) {
    if (requested == 0) {
      return musa::dnn::MemoryHandler(nullptr, [](void*) {});
    }
    std::size_t largest_available = 0;
    for (std::size_t index = 0; index < free_blocks_.size(); ++index) {
      const Block block = free_blocks_[index];
      const std::size_t start = align_up(block.offset, operation_);
      const std::size_t block_end =
          checked_add(block.offset, block.size, operation_);
      if (start > block_end) {
        continue;
      }
      largest_available =
          std::max(largest_available, block_end - start);
      if (requested > block_end - start) {
        continue;
      }
      const std::size_t allocation_end =
          checked_add(start, requested, operation_);
      free_blocks_.erase(free_blocks_.begin() +
                         static_cast<std::ptrdiff_t>(index));
      if (block.offset < start) {
        free_blocks_.push_back({block.offset, start - block.offset});
      }
      if (allocation_end < block_end) {
        free_blocks_.push_back(
            {allocation_end, block_end - allocation_end});
      }
      normalize();
      const std::shared_ptr<Arena> self = shared_from_this();
      return musa::dnn::MemoryHandler(
          pointer_ + start,
          [self, start, requested](void*) {
            self->release(start, requested);
          });
    }
    throw std::runtime_error(
        operation_ + " workspace request " + std::to_string(requested) +
        " exceeds largest available block " +
        std::to_string(largest_available) + " in arena " +
        std::to_string(size_));
  }

 private:
  struct Block {
    std::size_t offset = 0;
    std::size_t size = 0;
  };

  void release(std::size_t offset, std::size_t size) {
    free_blocks_.push_back({offset, size});
    normalize();
  }

  void normalize() {
    std::sort(
        free_blocks_.begin(),
        free_blocks_.end(),
        [](const Block& left, const Block& right) {
          return left.offset < right.offset;
        });
    std::vector<Block> merged;
    merged.reserve(free_blocks_.size());
    for (const Block block : free_blocks_) {
      if (merged.empty()) {
        merged.push_back(block);
        continue;
      }
      Block& previous = merged.back();
      const std::size_t previous_end =
          checked_add(previous.offset, previous.size, operation_);
      if (block.offset > previous_end) {
        merged.push_back(block);
        continue;
      }
      const std::size_t block_end =
          checked_add(block.offset, block.size, operation_);
      if (block_end > previous_end) {
        previous.size = block_end - previous.offset;
      }
    }
    free_blocks_ = std::move(merged);
  }

  std::byte* pointer_ = nullptr;
  std::size_t size_ = 0;
  std::string operation_;
  std::vector<Block> free_blocks_;
};

}  // namespace mudnn_workspace_detail

inline musa::dnn::MemoryMaintainer make_mudnn_workspace_maintainer(
    void* pointer,
    std::size_t size,
    std::string_view operation) {
  auto arena = std::make_shared<mudnn_workspace_detail::Arena>(
      pointer, size, std::string(operation));
  return [arena](std::size_t requested) {
    return arena->allocate(requested);
  };
}

}  // namespace flagdnn::validation::mthreads
