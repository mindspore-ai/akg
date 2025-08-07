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

#include "internal_pyboost_runner.h"

namespace ms::pynative {

void InternalPyboostRunner::GetOrCreateKernel(const TensorList &inputs,
                                              const TensorList &outputs) {
  auto key = GetOrGenerateOpKey(op_key_);
  auto it = hash_map_.find(key);
  if (it != hash_map_.end()) {
    internal_op_ = it->second;
    MS_LOG(DEBUG) << "Internal Op [" << this->op_name() << "] hit cache";
  } else {
    MS_LOG(DEBUG) << "Internal Op [" << this->op_name() << "] miss cache";
    TransDataType(inputs, outputs);
    UpdateArgImmutableInfo(&inputs_ii_, inputs, true);
    UpdateArgImmutableInfo(&outputs_ii_, outputs);
    internal_op_ = CreateKernel(inputs_ii_, outputs_ii_);
    MS_EXCEPTION_IF_NULL(internal_op_);
    auto status = internal_op_->Init();
    if (status != mindspore::internal::kInternalOk) {
      internal_op_ = nullptr;
      MS_LOG(EXCEPTION) << "Init internal kernel failed, kernel_name: "
                        << this->op_name();
      return;
    }
    hash_map_[key] = internal_op_;
  }

  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs, true);
  TransInternalShapes(&internal_outputs_shape_, outputs, false);

  if (!UpdateParam()) {
    MS_LOG(EXCEPTION) << "UpdateParam failed, kernel_name: " << this->op_name();
  }
  auto internal_ret = internal_op_->UpdateShape(internal_inputs_shape_,
                                                internal_outputs_shape_);
  if (internal_ret != mindspore::internal::kInternalOk) {
    MS_LOG(EXCEPTION) << "InternalKernel UpdateShape failed, kernel_name: "
                      << this->op_name();
  }

  tiling_cache_item_ = GetOrGenerateTiling();
}

size_t InternalPyboostRunner::CalcWorkspace() {
  MS_EXCEPTION_IF_NULL(internal_op_);
  auto workspace_size_list = internal_op_->GetWorkspaceSize();
  return std::accumulate(workspace_size_list.begin(), workspace_size_list.end(),
                         0);
}

void InternalPyboostRunner::TransDataType(const TensorList &ms_inputs,
                                          const TensorList &ms_outputs) {
  internal_inputs_dtype_.resize(ms_inputs.size());
  internal_outputs_dtype_.resize(ms_outputs.size());

  for (size_t i = 0; i < ms_inputs.size(); ++i) {
    if (!ms_inputs[i].is_defined()) {
      internal_inputs_dtype_[i] = mindspore::internal::DataType::kTypeNone;
      continue;
    }

    internal_inputs_dtype_[i] = TransInternalDataType(ms_inputs[i].data_type());
  }

  for (size_t i = 0; i < ms_outputs.size(); ++i) {
    if (!ms_outputs[i].is_defined()) {
      internal_outputs_dtype_[i] = mindspore::internal::DataType::kTypeNone;
      continue;
    }
    internal_outputs_dtype_[i] =
        TransInternalDataType(ms_outputs[i].data_type());
  }
}

TilingCacheItemPtr InternalPyboostRunner::GetOrGenerateTiling() {
  std::lock_guard<SimpleSpinLock> lock(lock_);
  auto key = GetOrGenerateOpTilingKey(tiling_key_);
  auto tiling_info_ptr = InternalTilingCache::GetInstance().Bind(key);
  if (tiling_info_ptr == nullptr) {
    // TODO check if need to bind device to current thread
    // device_context->device_res_manager_->BindDeviceToCurrentThread(false);
    MS_LOG(INFO) << "start create tiling info for " << this->op_name();
    auto tiling_size = internal_op_->GetTilingSize();
    auto host_addr = TilingMemMgr::GetInstance().pool_host_.Malloc(tiling_size);
    mindspore::internal::HostRunInfoPtr host_run_info_ptr = nullptr;
    auto status = internal_op_->Tiling(host_addr, &host_run_info_ptr);
    if (status != mindspore::internal::kInternalOk ||
        host_run_info_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Tiling error for " << this->op_name()
                        << ", status: " << status
                        << ", host_run_info_ptr: " << host_run_info_ptr;
    }
    auto device_addr =
        TilingMemMgr::GetInstance().pool_device_.Malloc(tiling_size);
    TilingMemMgr::GetInstance().CopyAsync(host_addr, device_addr, tiling_size);
    auto tiling_info =
        std::make_shared<mindspore::internal::TilingInfo>(device_addr, nullptr);
    tiling_info->host_run_info_ = host_run_info_ptr;
    auto workspace_size_list = internal_op_->GetWorkspaceSize();
    tiling_info->host_run_info_->SetWorkSpaceSize(workspace_size_list);
    tiling_info_ptr =
        std::make_shared<TilingCacheItem>(tiling_info, host_addr, tiling_size);
    if (TilingMemMgr::GetInstance().pool_device_.IsOneOffMem(device_addr)) {
      // tiling mem pool is full, comb out some items which are not recently
      // used with high probability
      auto erased_items =
          InternalTilingCache::GetInstance().CombOutSuspectedUselessItems();
      if (!erased_items.empty()) {
        for (auto &item : erased_items) {
          TilingMemMgr::GetInstance().pool_device_.Free(
              item->tiling_info_->tiling_addr_, item->size_);
          TilingMemMgr::GetInstance().pool_host_.Free(item->host_addr_,
                                                      item->size_);
        }
        TilingMemMgr::GetInstance().pool_device_.Rearrange();
        TilingMemMgr::GetInstance().pool_host_.Rearrange();
      }
      MS_LOG(INFO)
          << "The tiling memory pool is full, comb out not used items: "
          << erased_items.size();
    }
    (void)InternalTilingCache::GetInstance().Insert(key, tiling_info_ptr);
    MS_LOG(INFO) << "end create tiling info for " << this->op_name();
  }
  return tiling_info_ptr;
}

void InternalPyboostRunner::TransInternalShapes(
    mindspore::internal::ShapeInfoList *shapelist, const TensorList &tensorlist,
    bool is_input) {
  for (size_t i = 0; i < tensorlist.size(); i++) {
    if (!tensorlist[i].is_defined()) {
      shapelist->at(i) = mindspore::internal::ShapeInfo{0};
      continue;
    }

    if (!tensorlist[i].is_contiguous()) {
      if (is_input) {
        MS_LOG(EXCEPTION) << "For internal op [" << this->op_name()
                          << "], the input tensorlist[" << i
                          << "] is not contiguous: "
                          << ", please convert it to contiguous tensor using "
                             "tensor.contiguous().";
      } else {
        MS_LOG(EXCEPTION) << "For internal op [" << this->op_name()
                          << "], the output tensorlist[" << i
                          << "] is not contiguous: "
                          << ", please convert it to contiguous tensor using "
                             "tensor.contiguous().";
      }
    }

    auto shape = tensorlist[i].data_type() != kMetaTypeNone
                     ? TransInternalShape(tensorlist[i].shape())
                     : mindspore::internal::ShapeInfo{0};
    shapelist->at(i) = std::move(shape);
  }
}

void InternalPyboostRunner::UpdateArgImmutableInfo(
    internal::ArgImmutableInfo *arginfo, const ms::Tensor &tensor,
    internal::DataType dtype) {
  arginfo->SetDtype(dtype);
  if (!tensor.is_defined()) {
    arginfo->SetFormat(internal::TensorFormat::kFormatND);
    return;
  }
  arginfo->SetFormat(
      TransInternalFormat(GetFormatFromStrToEnum(tensor.format())));
}

void InternalPyboostRunner::UpdateArgImmutableInfo(
    std::vector<internal::ArgImmutableInfo> *arginfos,
    const TensorList &tensorlist, bool is_input) {
  arginfos->resize(tensorlist.size());
  for (size_t i = 0; i < tensorlist.size(); ++i) {
    if (is_input) {
      UpdateArgImmutableInfo(&(arginfos->at(i)), tensorlist[i],
                             internal_inputs_dtype_[i]);
    } else {
      UpdateArgImmutableInfo(&(arginfos->at(i)), tensorlist[i],
                             internal_outputs_dtype_[i]);
    }
  }
}

void InternalPyboostRunner::GetWorkspace(
    const internal::InternalOpPtr &internal_op,
    internal::WsAddrList *internal_wss_addr) {
  auto workspace_ptr = this->workspace_ptr();
  if (workspace_ptr == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(internal_op);
  auto workspace_size_list = internal_op->GetWorkspaceSize();
  internal_wss_addr->resize(workspace_size_list.size());

  size_t offset = 0;
  for (size_t i = 0; i < workspace_size_list.size(); i++) {
    auto work_ptr = workspace_ptr + offset;
    internal_wss_addr->at(i) = work_ptr;
    offset += workspace_size_list[i];
  }
}

void InternalPyboostRunner::LaunchKernel() {
  MS_EXCEPTION_IF_NULL(tiling_cache_item_);
  MS_EXCEPTION_IF_NULL(internal_op_);
  internal::InputsAddrList inputs_addr;
  internal::OutputsAddrList outputs_addr;
  InternalPyboostRunner::UpdateAddr(&inputs_addr, this->inputs());
  InternalPyboostRunner::UpdateAddr(&outputs_addr, this->outputs());
  internal::WsAddrList _internal_wss_addr;
  InternalPyboostRunner::GetWorkspace(internal_op_, &_internal_wss_addr);

  auto op_name = this->op_name();
  MS_LOG(DEBUG) << "Launch InternalKernel " << op_name << " start";
  internal_op_->SetTilingInfo(tiling_cache_item_->tiling_info_);
  auto &internal_wss_addr =
      const_cast<internal::WsAddrList &>(_internal_wss_addr);
  internal::InternalStatus status = internal_op_->Launch(
      inputs_addr, outputs_addr, internal_wss_addr, this->stream(), op_name);
  InternalTilingCache::GetInstance().Unbind(tiling_cache_item_);
  if (status != internal::InternalStatus::kInternalOk) {
    MS_LOG(EXCEPTION) << "Launch InternalKernel failed, kernel_name: "
                      << op_name;
  }
  MS_LOG(DEBUG) << "Launch InternalKernel " << op_name << " end";
}
} // namespace ms::pynative