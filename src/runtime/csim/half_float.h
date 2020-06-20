/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef RUNTIME_CSIM_HALF_FLOAT_H_
#define RUNTIME_CSIM_HALF_FLOAT_H_

#include <stdint.h>

class half {
 public:
  half() {
    bits = 0;
  }

  half(const half &a) {
    bits = a.bits;
  }

  half(uint16_t frac, uint16_t exp, uint16_t sign) {
    IEEE.frac = frac;
    IEEE.exp = exp;
    IEEE.sign = sign;
  }

  // implicit float promotion to half
  explicit half(const float a) {
    IEEE_Float f;
    f.bits = a;
    IEEE.sign = f.IEEE.sign;

    int exp_ = f.IEEE.exp - 127;
    if (exp_ < -24) {
      IEEE.frac = 0;
      IEEE.exp = 0;
    } else if (exp_ > 15) {
      IEEE.frac = static_cast<bool>(f.IEEE.frac);
      IEEE.exp = 31;
    } else if (exp_ < -14) {
      IEEE.frac = (1 << (24 + exp_)) + (f.IEEE.frac >> (-1 - exp_));
      IEEE.exp = 0;
    } else {
      IEEE.frac = f.IEEE.frac >> 13;
      IEEE.exp = exp_ + 15;
    }
  }

  ~half() {}

  float ToFloat() const {
    IEEE_Float f;
    f.IEEE.sign = IEEE.sign;

    if (IEEE.exp == 0) {
      if (IEEE.frac == 0) {
        f.IEEE.frac = 0;
        f.IEEE.exp = 0;
      } else {
        float sign = IEEE.sign ? -1.0f : 1.0f;
        return sign * IEEE.frac / static_cast<float>(1 << 24);
      }
    } else if (IEEE.exp == 31) {
      f.IEEE.exp = 0xFF;
      f.IEEE.frac = static_cast<bool>(IEEE.frac);
    } else {
      f.IEEE.exp = IEEE.exp + 112;
      f.IEEE.frac = IEEE.frac << 13;
    }
    return f.bits;
  }

  operator float() const {
    return ToFloat();
  }

  operator double() const {
    return static_cast<double>(ToFloat());
  }

  operator int64_t() const {
    return static_cast<int64_t>(ToFloat());
  }

  operator uint64_t() const {
    return static_cast<uint64_t>(ToFloat());
  }

  operator int32_t() const {
    return static_cast<int32_t>(ToFloat());
  }

  operator uint32_t() const {
    return static_cast<uint32_t>(ToFloat());
  }

  operator uint16_t() const {
    return static_cast<uint16_t>(ToFloat());
  }

  operator int16_t() const {
    return static_cast<int16_t>(ToFloat());
  }

  operator int8_t() const {
    return static_cast<int8_t>(ToFloat());
  }

  operator uint8_t() const {
    return static_cast<uint8_t>(ToFloat());
  }

  operator bool() const {
    return !IsZero();
  }

  bool IsNaN() const {
    return IEEE.frac != 0 && IEEE.exp == 31;
  }

  bool IsZero() const {
    return IEEE.frac == 0 && IEEE.exp == 0;
  }

  half operator+(const half &b) const {
    return half(ToFloat() + static_cast<float>(b));
  }

  half operator-(const half &b) const {
    return half(ToFloat() - static_cast<float>(b));
  }

  half operator*(const half &b) const {
    return half(ToFloat() * static_cast<float>(b));
  }

  half operator/(const half &b) const {
    return half(ToFloat() * static_cast<float>(b));
  }

  half operator+=(const half &b) {
    half res = half(ToFloat() + static_cast<float>(b));
    this->bits = res.bits;
    return res;
  }

  half operator-=(const half &b) {
    half res = half(ToFloat() - static_cast<float>(b));
    this->bits = res.bits;
    return res;
  }

  half operator*=(const half &b) {
    half res = half(ToFloat() * static_cast<float>(b));
    this->bits = res.bits;
    return res;
  }

  half operator/=(const half &b) {
    half res = half(ToFloat() / static_cast<float>(b));
    this->bits = res.bits;
    return res;
  }

  bool operator==(const half &b) const {
    if (IsNaN() && b.IsNaN()) {
      return true;
    } else if (IsZero() && b.IsZero()) {
      return true;
    } else {
      return bits == b.bits;
    }
  }

  bool operator!=(const half &b) const {
    return !(*this == b);
  }

  bool operator>(const half &b) const {
    return ToFloat() > static_cast<float>(b);
  }

  bool operator>=(const half &b) const {
    return ToFloat() >= static_cast<float>(b);
  }

  bool operator<(const half &b) const {
    return ToFloat() < static_cast<float>(b);
  }

  bool operator<=(const half &b) const {
    return ToFloat() <= static_cast<float>(b);
  }

  union {
    uint16_t bits;
    struct {
      uint16_t frac : 10;
      uint16_t exp : 5;
      uint16_t sign : 1;
    } IEEE;
  };

  union IEEE_Float {
    float bits;
    struct {
      uint32_t frac : 23;
      uint16_t exp : 8;
      uint16_t sign : 1;
    } IEEE;
  };
};

static inline std::ostream &operator<<(std::ostream &os, const half &value) {
  os << static_cast<float>(value);
  return os;
}

#ifdef DEBUG
static bool CheckHalf() {
  bool correct = true;
  for (int i = 0; i <= 0xFFFF; i++) {
    half h;
    h.bits = i;
    float f = static_cast<float>(h);
    half h1 = f;
    if (h != h1) {
      printf("Error! %x %x\n", h.bits, h1.bits);
      correct = false;
    }
  }
  return correct;
}
#endif  // DEBUG

#endif  // RUNTIME_CSIM_HALF_FLOAT_H_
