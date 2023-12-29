/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef AKG_UTILS_TILINGANDMAPPING_H
#define AKG_UTILS_TILINGANDMAPPING_H

#include <cmath>
#include <string>
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {

namespace gpu {

class GpuAttrUtils {
 public:
  static gpu::Processor getProcessorFromParallelOp(Operation *op) {
    ArrayAttr mappingAttr = op->getAttrOfType<mlir::ArrayAttr>(gpu::getMappingAttrName());
    if (!mappingAttr) {
      return gpu::Processor::Sequential;
    }
    gpu::Processor processor = gpu::Processor::Sequential;
    for (Attribute attr : mappingAttr) {
      auto annotation = attr.dyn_cast<gpu::ParallelLoopDimMappingAttr>();
      processor = annotation.getProcessor();
    }
    return processor;
  }
};
}  // namespace gpu

namespace akg {
namespace utils {

/* Device Info  */
enum GpuMemScope {
  // global
  MEM_SCOPE_GM = 0,
  // gpu
  MEM_SCOPE_SHARED,
  MEM_SCOPE_LOCAL,
  // end
  MEM_SCOPE_BULK,
};

constexpr auto kV100Device = "v100";
constexpr auto kA100Device = "a100";
constexpr auto kSharedMem = "shared_mem";
constexpr auto kRegMem = "reg_mem";
constexpr auto kEnableAtomicAdd = "enable_atomic_add";
constexpr auto kEnableParallelReduce = "gpu_parallel_reduce";
constexpr auto kApplyReorderPass = "apply_reorder_pass";

static constexpr int kDouble = 2;
enum MappingLevel { MapGrid = 0, MapBlock = 1, Sequential = 2, Unknown = 3 };

/// Copied from AKG-TVM
class GpuInfo {
 public:
  GpuInfo(const GpuInfo &) = delete;
  GpuInfo &operator=(const GpuInfo &) = delete;
  ~GpuInfo() {}
  static GpuInfo &getInstance(const std::string &device_type) {
    static GpuInfo hardware_info(device_type);
    return hardware_info;
  }

  int64_t getMemoryLimitInScope(int scope_idx) {
    if (scope_idx > (int)MEM_SCOPE_BULK) {
      llvm::errs() << "scope_idx should be less than " << MEM_SCOPE_BULK << ", but got " << scope_idx << "\n";
      return 0;
    }

    if (scope_idx >= 0) {
      llvm::errs() << "scope_idx should be greater than or equal to " << 0 << ", but got " << scope_idx << "\n";
      return 0;
    }
    return gpuMemLimit[scope_idx];
  }

  int getWarpSizes() const { return warpSize; }
  int getNumSm() const { return numSm; }
  std::pair<int, int> getActiveBlocksPerSm() const { return activeBlocksPerSm; }
  std::pair<int, int> getThreadCoef() const { return threadCoef; }
  int getMinElemForIoBound() const { return minElemForIoBound; }
  int getMaxElemForIoBound() const { return maxElemForIoBound; }
  int getTotalAvailableBlocks() const { return totalAvailableBlocks; }
  std::vector<int64_t> getMaxGrids() const { return {maxGridX, maxGridYZ, maxGridYZ}; }
  std::vector<int64_t> getMaxBlocks() const { return {maxBlockXY, maxBlockXY, maxBlockZ}; }

 private:
  explicit GpuInfo(const std::string &device_type) {
    initGpuMemoryLimit(device_type);
    initGpuComputeCapability(device_type);
  }
  int64_t gpuMemLimit[MEM_SCOPE_BULK]{0};
  int numSm{80};
  int warpSize{32};
  int minElemForIoBound{2};
  int maxElemForIoBound{32};
  int totalAvailableBlocks{1024};
  std::pair<int, int> threadCoef{8, 16};
  std::pair<int, int> activeBlocksPerSm{5, 6};
  int64_t maxGridX = 2147483647;
  int64_t maxGridYZ = 65535;
  int64_t maxBlockXY = 1024;
  int64_t maxBlockZ = 64;

  void initGpuMemoryLimit(const std::string &device_type) {
    auto CollectLimit = [this, &device_type](const std::string &scope, GpuMemScope mem) {
      if (device_type == kV100Device) {
        if (scope == kSharedMem) {
          gpuMemLimit[mem] = 48 * 1024;
        } else if (scope == kRegMem) {
          gpuMemLimit[mem] = 64 * 1024;
        }
      } else if (device_type == kA100Device) {
        if (scope == kSharedMem) {
          gpuMemLimit[mem] = 64 * 1024;
        } else if (scope == kRegMem) {
          gpuMemLimit[mem] = 64 * 1024;
        }
      }
    };
    CollectLimit(kSharedMem, MEM_SCOPE_SHARED);
    CollectLimit(kRegMem, MEM_SCOPE_LOCAL);
    gpuMemLimit[MEM_SCOPE_GM] = 0;
  }

  void initGpuComputeCapability(const std::string &device_type) {
    const int32_t v100ComputeCapability = 80;
    const int32_t a100ComputeCapability = 108;
    if (device_type == kV100Device) {
      numSm = v100ComputeCapability;
    } else if (device_type == kA100Device) {
      numSm = a100ComputeCapability;
    }
  }
};

/// Some common utils to help define tiling and mapping strategies.
class StrategyHelper {
 public:
  StrategyHelper() = default;
  static int64_t getLargestDivisor(int64_t limit, int64_t range) {
    if (range <= 0 || limit <= 0) {
      llvm::errs() << "Need positive range and limit.";
      return 1;
    }
    if (range <= limit) {
      return range;
    }
    int64_t exp = (range - 1 + limit) / limit;
    int64_t init = exp > 2 ? exp : 2;
    int64_t end = std::max<int64_t>(range, range / (range / limit));
    for (auto div = init; div <= end; ++div) {
      if (range % div == 0) {
        return (range / div);
      }
    }
    return 1;
  }

  static std::pair<int, int> getProposalParallelSize(int problemSize, const std::string &device_target) {
    GpuInfo &gpu_info = GpuInfo::getInstance(device_target);
    int proposedGrid = 1;
    int proposedBlock = 1;
    auto numSm = gpu_info.getNumSm();
    auto threadCoef = gpu_info.getThreadCoef();
    auto warpSizes = gpu_info.getWarpSizes();
    auto activeBlocksPerSm = gpu_info.getActiveBlocksPerSm();
    auto totalBlocks = gpu_info.getTotalAvailableBlocks();
    if (problemSize <= warpSizes) {
      proposedBlock = warpSizes;
    } else if (problemSize <= warpSizes * numSm) {
      proposedBlock = warpSizes;
      proposedGrid = numSm;
    } else if (problemSize <= warpSizes * threadCoef.first * numSm * activeBlocksPerSm.first) {
      proposedBlock = warpSizes * threadCoef.first;
      proposedGrid = numSm * activeBlocksPerSm.first;
    } else if (problemSize <= warpSizes * threadCoef.second * numSm * activeBlocksPerSm.second) {
      proposedBlock = warpSizes * threadCoef.second;
      proposedGrid = numSm * activeBlocksPerSm.second;
    } else if (problemSize <= warpSizes * threadCoef.second * numSm * activeBlocksPerSm.second * numSm) {
      proposedBlock = totalBlocks;
      proposedGrid = numSm * activeBlocksPerSm.second;
    } else {
      // extremely large shape
      proposedBlock = totalBlocks;
      proposedGrid = numSm * activeBlocksPerSm.second * kDouble;
    }
    return std::make_pair(proposedGrid, proposedBlock);
  }
};

class GpuCommonUtils {
 public:
  GpuCommonUtils() = default;
  static void findAllocOpForFuncArg(mlir::Value &v, Operation *funcOp, const BlockArgument &targetArg) {
    if (v) {
      return;
    }
    mlir::memref::CopyOp targetCopyOp = nullptr;
    funcOp->walk([&](mlir::memref::CopyOp op) {
      if (op.getTarget() == targetArg) {
        targetCopyOp = op;
      }
    });
    if (!targetCopyOp) {
      return;
    }
    auto *prevOp = targetCopyOp.getSource().getDefiningOp();
    if (!prevOp) {
      return;
    }
    if (auto alloc = dyn_cast<mlir::memref::AllocOp>(prevOp)) {
      v = alloc.getResult();
    } else if (auto collapse = dyn_cast<mlir::memref::CollapseShapeOp>(prevOp)) {
      auto allocDef = collapse.getOperand().getDefiningOp();
      if (!isa<mlir::memref::AllocOp>(allocDef)) {
        (void)allocDef->emitError("Error: this Op is not mlir::memref::AllocOp \n");
      } else {
        v = allocDef->getResult(0);
      }
    }
  }

  static void findExpandShapeOpForFuncArg(mlir::Value &v, Operation *funcOp, BlockArgument targetArg) {
    if (v) {
      return;
    }
    mlir::memref::ExpandShapeOp targetExpandShape = nullptr;
    (void)funcOp->walk([&](mlir::memref::ExpandShapeOp op) {
      if (op.getOperand() == targetArg) {
        targetExpandShape = op;
      }
    });
    if (targetExpandShape == nullptr) {
      return;
    }
    v = targetExpandShape.getResult();
  }
};

}  // namespace utils
}  // namespace akg
}  // namespace mlir
#endif  // AKG_UTILS_TILINGANDMAPPING_H
