#include <cstdint>

#include "acl/acl.h"
#include "add_custom_tiling.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "ops.h"

void add_custom_kernel(uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
                       uint8_t* x, uint8_t* y, uint8_t* z, uint8_t* tiling);

namespace ascend_kernel {

at::Tensor add_custom_torch(const at::Tensor& x1, const at::Tensor& x2) {
  TORCH_CHECK(x1.scalar_type() == at::kFloat, "only FP32 supported");
  TORCH_CHECK(x1.scalar_type() == x2.scalar_type(),
              "x1 and x2 must have the same dtype");
  TORCH_CHECK(x1.is_privateuseone(), "x1 must be on NPU");
  TORCH_CHECK(x2.is_privateuseone(), "x2 must be on NPU");
  TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 shapes must match");

  at::Tensor y = at::empty_like(x1);
  int64_t total_elements = x1.numel();
  TORCH_CHECK(total_elements > 0, "input tensors must not be empty");

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(true);

  int32_t device_id = -1;
  aclrtGetDevice(&device_id);
  int64_t available_core_num = 0;
  auto ret = aclrtGetDeviceInfo(
      device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM, &available_core_num);
  TORCH_CHECK(ret == ACL_SUCCESS && available_core_num > 0,
              "failed to get NPU vector core count");

  AddTilingData tiling;
  tiling.totalLength = total_elements;
  uint64_t total_tiles = (total_elements + TILE_LENGTH - 1) / TILE_LENGTH;
  uint64_t tiles_per_core =
      (total_tiles + available_core_num - 1) / available_core_num;
  tiling.blockNum = (total_tiles + tiles_per_core - 1) / tiles_per_core;
  tiling.numPerCore = tiles_per_core * TILE_LENGTH;
  tiling.tailNumLastCore =
      total_elements - tiling.numPerCore * (tiling.blockNum - 1);

  at::Tensor tiling_tensor = at::empty(
      {static_cast<int64_t>(sizeof(AddTilingData))},
      x1.options().dtype(at::kByte));
  aclrtMemcpy(tiling_tensor.mutable_data_ptr(), sizeof(AddTilingData), &tiling,
              sizeof(AddTilingData), ACL_MEMCPY_HOST_TO_DEVICE);

  add_custom_kernel(static_cast<uint32_t>(tiling.blockNum), nullptr, acl_stream,
                    reinterpret_cast<uint8_t*>(x1.mutable_data_ptr()),
                    reinterpret_cast<uint8_t*>(x2.mutable_data_ptr()),
                    reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
                    reinterpret_cast<uint8_t*>(tiling_tensor.mutable_data_ptr()));
  return y;
}

}  // namespace ascend_kernel
