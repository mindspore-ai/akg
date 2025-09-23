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

#ifndef MS_CUSTOM_OPS_INTERNAL_OP_PYBOOST_RUNNER_H_
#define MS_CUSTOM_OPS_INTERNAL_OP_PYBOOST_RUNNER_H_

#include <functional>
#include <optional>
#include <set>
#include <unordered_map>

#include "internal_pyboost_utils.h"
#include "internal_spinlock.h"
#include "internal_tiling_cache.h"
#include "module.h"
#include "ccsrc/base/ms_kernels_internal/internal_helper.h"
#include "mindspore/include/custom_op_api.h"
#include "lib/plugin/ascend/ms_kernels_internal/internal_kernel/include/internal.h"

namespace ms_custom_ops {
using namespace mindspore;
using TensorList = std::vector<ms::Tensor>;

class InternalPyboostRunner : public ms::pynative::PyboostRunner {
 public:
  using ms::pynative::PyboostRunner::PyboostRunner;
  virtual ~InternalPyboostRunner() = default;

  // Generic setup method for configuring the runner with parameters and
  // calculating hash keys
  template <typename... Args>
  void Setup(const std::string &op_name, const Args &...args) {
    // Calculate hash keys
    this->op_key_ = CalcInternalOpApiHash(op_name, args...);
    this->tiling_key_ = CalcInternalOpTilingHash(op_name, args...);
  }

  void GetOrCreateKernel(const TensorList &inputs, const TensorList &outputs);

 protected:
  size_t CalcWorkspace() override;

  virtual uint64_t GetOrGenerateOpKey(const uint64_t &op_key) const { return op_key; }

  virtual uint64_t GetOrGenerateOpTilingKey(const uint64_t &tiling_key) const { return tiling_key; }

  virtual bool UpdateParam() { return true; }

 protected:
  void TransDataType(const TensorList &ms_inputs, const TensorList &ms_outputs);

  TilingCacheItemPtr GetOrGenerateTiling();
  virtual internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                               const internal::OutputsImmutableInfoList &outputs) = 0;
  void TransInternalShapes(internal::ShapeInfoList *shapelist, const TensorList &tensorlist, bool is_input = false);

  static void UpdateAddr(std::vector<internal::RawDeviceAddr> *addrlist, const TensorList &tensorlist) {
    addrlist->resize(tensorlist.size());
    for (size_t i = 0; i < tensorlist.size(); i++) {
      if (!tensorlist[i].is_defined()) {
        addrlist->at(i) = nullptr;
      } else {
        addrlist->at(i) = tensorlist[i].GetDataPtr();
      }
    }
  }

  void GetWorkspace(const internal::InternalOpPtr &internal_op, internal::WsAddrList *internal_wss_addr);

  void LaunchKernel() override;

  uint64_t op_key_{0};
  uint64_t tiling_key_{0};
  internal::InternalOpPtr internal_op_{nullptr};
  inline static std::unordered_map<uint64_t, internal::InternalOpPtr> hash_map_;
  internal::DtypeInfoList internal_inputs_dtype_;
  internal::DtypeInfoList internal_outputs_dtype_;
  internal::ShapeInfoList internal_inputs_shape_;
  internal::ShapeInfoList internal_outputs_shape_;
  internal::InputsImmutableInfoList inputs_ii_;
  internal::OutputsImmutableInfoList outputs_ii_;
  TilingCacheItemPtr tiling_cache_item_{nullptr};

 private:
  void UpdateArgImmutableInfo(internal::ArgImmutableInfo *arginfo, const ms::Tensor &tensor, internal::DataType dtype);
  void UpdateArgImmutableInfo(std::vector<internal::ArgImmutableInfo> *arginfos, const TensorList &tensorlist,
                              bool is_input = false);

  SimpleSpinLock lock_;
};
}  // namespace ms_custom_ops
#endif  // MS_CUSTOM_OPS_INTERNAL_OP_PYBOOST_RUNNER_H_
