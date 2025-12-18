/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef AKG_UTILS_ANALYSISFORNPU_H
#define AKG_UTILS_ANALYSISFORNPU_H

#include <cmath>
#include <string>
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace akg {

constexpr auto kSoc910B1 = "Ascend910B1";
constexpr auto kSoc910B2 = "Ascend910B2";
constexpr auto kSoc910B2C = "Ascend910B2C";
constexpr auto kSoc910B3 = "Ascend910B3";
constexpr auto kSoc910B4 = "Ascend910B4";
constexpr auto kSoc910B4_1 = "Ascend910B4-1";

class HardwareConfig {
 public:
  HardwareConfig() = default;
  ~HardwareConfig() = default;
  HardwareConfig(uint32_t coreNumAic, uint32_t coreNumAiv, uint32_t l2Size, uint32_t l1Size, uint32_t l0aSize,
                 uint32_t l0bSize, uint32_t l0cSize, uint32_t ubSize)
      : coreNumAic(coreNumAic),
        coreNumAiv(coreNumAiv),
        l2(l2Size),
        l1(l1Size),
        l0a(l0aSize),
        l0b(l0bSize),
        l0c(l0cSize),
        ub(ubSize) {}

  uint32_t coreNumAic{0};
  uint32_t coreNumAiv{0};
  uint32_t l2{0};
  uint32_t l1{0};
  uint32_t l0a{0};
  uint32_t l0b{0};
  uint32_t l0c{0};
  uint32_t ub{0};
};

static const std::unordered_map<std::string, HardwareConfig> kHardwareConfigs = {
  {kSoc910B1, HardwareConfig{/*coreNumAic = */ 24,
                             /*coreNumAiv = */ 48,
                             /*l2 = */ 201326592,
                             /*l1 = */ 524288,
                             /*l0a = */ 65536,
                             /*l0b = */ 65536,
                             /*l0c = */ 131072,
                             /*ub = */ 196608}},
  {kSoc910B2, HardwareConfig{/*coreNumAic = */ 24,
                             /*coreNumAiv = */ 48,
                             /*l2 = */ 201326592,
                             /*l1 = */ 524288,
                             /*l0a = */ 65536,
                             /*l0b = */ 65536,
                             /*l0c = */ 131072,
                             /*ub = */ 196608}},
  {kSoc910B2C, HardwareConfig{/*coreNumAic = */ 24,
                              /*coreNumAiv = */ 48,
                              /*l2 = */ 201326592,
                              /*l1 = */ 524288,
                              /*l0a = */ 65536,
                              /*l0b = */ 65536,
                              /*l0c = */ 131072,
                              /*ub = */ 196608}},
  {kSoc910B3, HardwareConfig{/*coreNumAic = */ 20,
                             /*coreNumAiv = */ 40,
                             /*l2 = */ 201326592,
                             /*l1 = */ 524288,
                             /*l0a = */ 65536,
                             /*l0b = */ 65536,
                             /*l0c = */ 131072,
                             /*ub = */ 196608}},
  {kSoc910B4, HardwareConfig{/*coreNumAic = */ 20,
                             /*coreNumAiv = */ 40,
                             /*l2 = */ 100663296,
                             /*l1 = */ 524288,
                             /*l0a = */ 65536,
                             /*l0b = */ 65536,
                             /*l0c = */ 131072,
                             /*ub = */ 196608}},
  {kSoc910B4_1, HardwareConfig{/*coreNumAic = */ 20,
                               /*coreNumAiv = */ 40,
                               /*l2 = */ 176160768,
                               /*l1 = */ 524288,
                               /*l0a = */ 65536,
                               /*l0b = */ 65536,
                               /*l0c = */ 131072,
                               /*ub = */ 196608}},
};

class NpuInfo {
 public:
  NpuInfo(const NpuInfo &) = delete;
  NpuInfo &operator=(const NpuInfo &) = delete;
  ~NpuInfo() {}

  static NpuInfo &getInstance(const std::string &socVersion) {
    static NpuInfo hardware_info(socVersion);
    return hardware_info;
  }

  inline uint32_t getCoreNumAic() const { return hwConfig.coreNumAic; }
  inline uint32_t getCoreNumAiv() const { return hwConfig.coreNumAiv; }
  inline uint32_t getL2Size() const { return hwConfig.l2; }
  inline uint32_t getL1Size() const { return hwConfig.l1; }
  inline uint32_t getL0aSize() const { return hwConfig.l0a; }
  inline uint32_t getL0bSize() const { return hwConfig.l0b; }
  inline uint32_t getL0cSize() const { return hwConfig.l0c; }
  inline uint32_t getUbSize() const { return hwConfig.ub; }

  const HardwareConfig &getConfigByVersion(const std::string &socVersion) const {
    static HardwareConfig invalidHardwareConfig;
    const auto &it = kHardwareConfigs.find(socVersion);
    if (it == kHardwareConfigs.end()) {
      llvm::errs() << "The config for socVersion: " << socVersion << " does not exist.";
      return invalidHardwareConfig;
    }

    return it->second;
  }

 private:
  explicit NpuInfo(const std::string &socVersion) { hwConfig = getConfigByVersion(socVersion); }

  HardwareConfig hwConfig;
};

}  // namespace akg
}  // namespace mlir
#endif  // AKG_UTILS_ANALYSISFORNPU_H
