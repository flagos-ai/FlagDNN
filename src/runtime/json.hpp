/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_RUNTIME_JSON_HPP_
#define FLAGDNN_RUNTIME_JSON_HPP_

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace flagdnn::native::json {

class Value {
 public:
  using Array = std::vector<Value>;
  using Object = std::map<std::string, Value, std::less<>>;

  Value(std::nullptr_t value = nullptr) : storage_(value) {}
  explicit Value(bool value) : storage_(value) {}
  explicit Value(std::int64_t value) : storage_(value) {}
  explicit Value(double value) : storage_(value) {}
  explicit Value(std::string value) : storage_(std::move(value)) {}
  explicit Value(Array value) : storage_(std::move(value)) {}
  explicit Value(Object value) : storage_(std::move(value)) {}

  [[nodiscard]] bool is_null() const noexcept;
  [[nodiscard]] bool as_bool() const;
  [[nodiscard]] std::int64_t as_int() const;
  [[nodiscard]] double as_double() const;
  [[nodiscard]] const std::string& as_string() const;
  [[nodiscard]] const Array& as_array() const;
  [[nodiscard]] const Object& as_object() const;
  [[nodiscard]] const Value& at(std::string_view key) const;

 private:
  std::variant<
      std::nullptr_t, bool, std::int64_t, double, std::string, Array, Object>
      storage_;
};

Value parse(std::string_view input);

}  // namespace flagdnn::native::json

#endif  // FLAGDNN_RUNTIME_JSON_HPP_
