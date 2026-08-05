/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "json.hpp"

#include <charconv>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace flagdnn::native::json {
namespace {

[[noreturn]] void type_error(const char* expected) {
  throw std::runtime_error(std::string("JSON value is not ") + expected);
}

class Parser {
 public:
  explicit Parser(std::string_view input) : input_(input) {}

  Value parse_document() {
    skip_space();
    Value result = parse_value();
    skip_space();
    if (position_ != input_.size()) {
      fail("unexpected trailing characters");
    }
    return result;
  }

 private:
  [[noreturn]] void fail(const char* message) const {
    throw std::runtime_error(std::string("invalid JSON at byte ") +
                             std::to_string(position_) + ": " + message);
  }

  void skip_space() {
    while (position_ < input_.size() &&
           std::isspace(static_cast<unsigned char>(input_[position_])) != 0) {
      ++position_;
    }
  }

  bool consume(char value) {
    if (position_ < input_.size() && input_[position_] == value) {
      ++position_;
      return true;
    }
    return false;
  }

  void expect(char value) {
    if (!consume(value)) {
      fail("unexpected character");
    }
  }

  Value parse_value() {
    if (position_ >= input_.size()) {
      fail("unexpected end of input");
    }
    switch (input_[position_]) {
      case '{':
        return parse_object();
      case '[':
        return parse_array();
      case '"':
        return Value(parse_string());
      case 't':
        parse_literal("true");
        return Value(true);
      case 'f':
        parse_literal("false");
        return Value(false);
      case 'n':
        parse_literal("null");
        return Value(nullptr);
      default:
        if (input_[position_] == '-' ||
            std::isdigit(
                static_cast<unsigned char>(input_[position_])) != 0) {
          return parse_number();
        }
        fail("unexpected value");
    }
  }

  Value parse_object() {
    expect('{');
    skip_space();
    Value::Object result;
    if (consume('}')) {
      return Value(std::move(result));
    }
    while (true) {
      if (position_ >= input_.size() || input_[position_] != '"') {
        fail("object key must be a string");
      }
      std::string key = parse_string();
      skip_space();
      expect(':');
      skip_space();
      auto [iterator, inserted] =
          result.emplace(std::move(key), parse_value());
      (void)iterator;
      if (!inserted) {
        fail("duplicate object key");
      }
      skip_space();
      if (consume('}')) {
        break;
      }
      expect(',');
      skip_space();
    }
    return Value(std::move(result));
  }

  Value parse_array() {
    expect('[');
    skip_space();
    Value::Array result;
    if (consume(']')) {
      return Value(std::move(result));
    }
    while (true) {
      result.emplace_back(parse_value());
      skip_space();
      if (consume(']')) {
        break;
      }
      expect(',');
      skip_space();
    }
    return Value(std::move(result));
  }

  static unsigned int hex_digit(char value) {
    if (value >= '0' && value <= '9') {
      return static_cast<unsigned int>(value - '0');
    }
    if (value >= 'a' && value <= 'f') {
      return static_cast<unsigned int>(value - 'a' + 10);
    }
    if (value >= 'A' && value <= 'F') {
      return static_cast<unsigned int>(value - 'A' + 10);
    }
    throw std::runtime_error("invalid JSON unicode escape");
  }

  void append_unicode_escape(std::string& output) {
    if (input_.size() - position_ < 4) {
      fail("truncated unicode escape");
    }
    unsigned int codepoint = 0;
    for (int index = 0; index < 4; ++index) {
      codepoint = (codepoint << 4U) | hex_digit(input_[position_++]);
    }
    if (codepoint <= 0x7fU) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ffU) {
      output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    } else {
      output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
      output.push_back(
          static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
    }
  }

  std::string parse_string() {
    expect('"');
    std::string result;
    while (position_ < input_.size()) {
      const char value = input_[position_++];
      if (value == '"') {
        return result;
      }
      if (static_cast<unsigned char>(value) < 0x20U) {
        fail("control character in string");
      }
      if (value != '\\') {
        result.push_back(value);
        continue;
      }
      if (position_ >= input_.size()) {
        fail("truncated escape");
      }
      switch (input_[position_++]) {
        case '"':
          result.push_back('"');
          break;
        case '\\':
          result.push_back('\\');
          break;
        case '/':
          result.push_back('/');
          break;
        case 'b':
          result.push_back('\b');
          break;
        case 'f':
          result.push_back('\f');
          break;
        case 'n':
          result.push_back('\n');
          break;
        case 'r':
          result.push_back('\r');
          break;
        case 't':
          result.push_back('\t');
          break;
        case 'u':
          append_unicode_escape(result);
          break;
        default:
          fail("unknown string escape");
      }
    }
    fail("unterminated string");
  }

  Value parse_number() {
    const std::size_t start = position_;
    if (consume('-') && position_ >= input_.size()) {
      fail("truncated number");
    }
    if (consume('0')) {
      if (position_ < input_.size() &&
          std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
        fail("leading zero in number");
      }
    } else {
      if (position_ >= input_.size() ||
          std::isdigit(static_cast<unsigned char>(input_[position_])) == 0) {
        fail("expected number");
      }
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
        ++position_;
      }
    }

    bool floating_point = false;
    if (consume('.')) {
      floating_point = true;
      if (position_ >= input_.size() ||
          std::isdigit(static_cast<unsigned char>(input_[position_])) == 0) {
        fail("fraction requires at least one digit");
      }
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
        ++position_;
      }
    }
    if (position_ < input_.size() &&
        (input_[position_] == 'e' || input_[position_] == 'E')) {
      floating_point = true;
      ++position_;
      if (position_ < input_.size() &&
          (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      if (position_ >= input_.size() ||
          std::isdigit(static_cast<unsigned char>(input_[position_])) == 0) {
        fail("exponent requires at least one digit");
      }
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
        ++position_;
      }
    }

    const char* first = input_.data() + start;
    const char* last = input_.data() + position_;
    if (!floating_point) {
      std::int64_t integer = 0;
      const auto conversion = std::from_chars(first, last, integer);
      if (conversion.ec != std::errc{} || conversion.ptr != last) {
        fail("integer is out of range");
      }
      return Value(integer);
    }

    double real = 0.0;
    const auto conversion =
        std::from_chars(first, last, real, std::chars_format::general);
    if (conversion.ec != std::errc{} || conversion.ptr != last ||
        !std::isfinite(real)) {
      fail("floating-point number is out of range");
    }
    return Value(real);
  }

  void parse_literal(std::string_view literal) {
    if (input_.substr(position_, literal.size()) != literal) {
      fail("invalid literal");
    }
    position_ += literal.size();
  }

  std::string_view input_;
  std::size_t position_ = 0;
};

}  // namespace

bool Value::is_null() const noexcept {
  return std::holds_alternative<std::nullptr_t>(storage_);
}

bool Value::as_bool() const {
  const auto* value = std::get_if<bool>(&storage_);
  if (value == nullptr) {
    type_error("a boolean");
  }
  return *value;
}

std::int64_t Value::as_int() const {
  const auto* value = std::get_if<std::int64_t>(&storage_);
  if (value == nullptr) {
    type_error("an integer");
  }
  return *value;
}

double Value::as_double() const {
  if (const auto* value = std::get_if<double>(&storage_)) {
    return *value;
  }
  if (const auto* value = std::get_if<std::int64_t>(&storage_)) {
    return static_cast<double>(*value);
  }
  type_error("a number");
}

const std::string& Value::as_string() const {
  const auto* value = std::get_if<std::string>(&storage_);
  if (value == nullptr) {
    type_error("a string");
  }
  return *value;
}

const Value::Array& Value::as_array() const {
  const auto* value = std::get_if<Array>(&storage_);
  if (value == nullptr) {
    type_error("an array");
  }
  return *value;
}

const Value::Object& Value::as_object() const {
  const auto* value = std::get_if<Object>(&storage_);
  if (value == nullptr) {
    type_error("an object");
  }
  return *value;
}

const Value& Value::at(std::string_view key) const {
  const Object& object = as_object();
  const auto iterator = object.find(key);
  if (iterator == object.end()) {
    throw std::runtime_error("missing JSON field: " + std::string(key));
  }
  return iterator->second;
}

Value parse(std::string_view input) { return Parser(input).parse_document(); }

}  // namespace flagdnn::native::json
