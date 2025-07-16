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
#ifndef MS_CUSTOM_OPS_OP_DEF_ASCENDC_GRAPHMODE_ASCENDC_KERNEL_MOD_H_
#define MS_CUSTOM_OPS_OP_DEF_ASCENDC_GRAPHMODE_ASCENDC_KERNEL_MOD_H_
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <tuple>
#include <unordered_set>
#include <unordered_map>
#include <list>
#include <utility>

#include "common/kernel.h"
#include "include/common/utils/utils.h"
#include "kernel/ascend/acl_ir/op_api_exec.h"
#include "utils/ms_utils.h"
#include "plugin/res_manager/ascend/mem_manager/ascend_memory_manager.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "runtime/hardware/device_context_manager.h"
#include "plugin/device/ascend/kernel/custom/custom_kernel_factory.h"

namespace ms_custom_ops {
using namespace mindspore;
using namespace mindspore::kernel;
using namespace mindspore::device::ascend;
using namespace mindspore::ops;

using aclOpExecutor = device::ascend::aclOpExecutor;
using CallBackFunc = std::function<void()>;
using ProcessCache = device::ascend::ProcessCache;
using CacheTuple = std::tuple<uint64_t, aclOpExecutor *, ProcessCache, size_t>;

#define DEFINE_GET_WORKSPACE_FOR_RESIZE()                                                                       \
  template <typename... Args>                                                                                   \
  void GetWorkspaceForResize(const Args &... args) {                                                            \
    hash_id_ = device::ascend::AclnnHash(op_type_, args...);                                                    \
    size_t cur_workspace = 0;                                                                                   \
    auto iter = hash_map_.find(hash_id_);                                                                       \
    if (iter != hash_map_.end()) {                                                                              \
      MS_LOG(INFO) << "op " << op_type_ << " hit cache with hash id: " << hash_id_;                             \
      hash_cache_.splice(hash_cache_.begin(), hash_cache_, iter->second);                                       \
      cur_workspace = std::get<3>(hash_cache_.front());                                                         \
    } else {                                                                                                    \
      MS_LOG(INFO) << "op " << op_type_ << " miss cache with hash id: " << hash_id_;                            \
      auto [workspace, executor, cache, fail_cache] = GEN_EXECUTOR_FOR_RESIZE(op_type_, args...);               \
      cur_workspace = workspace;                                                                                \
      if (!fail_cache) {                                                                                        \
        hash_cache_.emplace_front(hash_id_, executor, cache, workspace);                                        \
        hash_map_[hash_id_] = hash_cache_.begin();                                                              \
        if (hash_cache_.size() > capacity_) {                                                                   \
          hash_map_.erase(std::get<0>(hash_cache_.back()));                                                     \
          auto release_func = std::get<2>(hash_cache_.back());                                                  \
          release_func(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});                        \
          hash_cache_.pop_back();                                                                               \
        }                                                                                                       \
      } else {                                                                                                  \
        hash_id_ = 0;                                                                                           \
        cache(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});                                 \
      }                                                                                                         \
    }                                                                                                           \
                                                                                                                \
    if (cur_workspace != 0) {                                                                                   \
      std::vector<size_t> workspace_size_list = {cur_workspace};                                                \
      SetWorkspaceSizeList(workspace_size_list);                                                                \
    }                                                                                                           \
  }                                                                                                             \
                                                                                                                \
  template <typename... Args>                                                                                   \
  std::pair<aclOpExecutor *, std::function<void()>> GetExecutor(const Args &... args) {                         \
    auto iter = hash_map_.find(hash_id_);                                                                       \
    if (capacity_ == 0 || hash_id_ == 0 || iter == hash_map_.end()) {                                           \
      aclOpExecutor *executor;                                                                                  \
      std::function<void()> release_func;                                                                       \
      std::tie(std::ignore, executor, release_func, hash_id_, std::ignore) =                                    \
        GEN_EXECUTOR_BOOST(op_type_, hash_id_, args...);                                                        \
      return std::make_pair(executor, release_func);                                                            \
    }                                                                                                           \
    const auto &cur_run = *(iter->second);                                                                      \
    UPDATE_TENSOR_FOR_LAUNCH(std::get<2>(cur_run), args...);                                                    \
    const auto &executor = std::get<1>(cur_run);                                                                \
    return std::make_pair(executor, nullptr);                                                                   \
  }                                                                                                             \
                                                                                                                \
  template <typename... Args>                                                                                   \
  void RunOp(void *stream_ptr, const std::vector<KernelTensor *> &workspace, const Args &... args) {            \
    auto [executor, release_func] = GetExecutor(args...);                                                       \
    if (workspace_size_list_.empty()) {                                                                         \
      RUN_OP_API_ASYNC(op_type_, nullptr, 0, executor, stream_ptr, release_func);                               \
    } else {                                                                                                    \
      if (workspace.empty()) {                                                                                  \
        MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";                                            \
      }                                                                                                         \
      auto workspace_tensor = workspace[0];                                                                     \
      if (workspace_tensor->size() != workspace_size_list_[0]) {                                                \
        MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"    \
                          << workspace_size_list_[0] << ", but get " << workspace_tensor->size();               \
      }                                                                                                         \
      RUN_OP_API_ASYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor, stream_ptr, \
                       release_func);                                                                           \
    }                                                                                                           \
  }                                                                                                             \
                                                                                                                \
  template <typename... Args>                                                                                   \
  std::tuple<aclOpExecutor *, ProcessCache, std::function<void()>> GetSyncExecutor(const Args &... args) {      \
    auto iter = hash_map_.find(hash_id_);                                                                       \
    if (capacity_ == 0 || hash_id_ == 0 || iter == hash_map_.end()) {                                           \
      aclOpExecutor *executor;                                                                                  \
      ProcessCache cache_func_ptr;                                                                              \
      std::function<void()> release_func;                                                                       \
      std::tie(std::ignore, executor, cache_func_ptr, release_func) = GEN_EXECUTOR(op_type_, args...);          \
      return std::make_tuple(executor, cache_func_ptr, release_func);                                           \
    }                                                                                                           \
    const auto &cur_run = *(iter->second);                                                                      \
    const auto &cache_func_ptr = std::get<2>(cur_run);                                                          \
    UPDATE_TENSOR_FOR_LAUNCH(cache_func_ptr, args...);                                                          \
    const auto &executor = std::get<1>(cur_run);                                                                \
    return std::make_tuple(executor, cache_func_ptr, nullptr);                                                  \
  }                                                                                                             \
                                                                                                                \
  template <typename... Args>                                                                                   \
  std::vector<ShapeVector> RunOpSync(void *stream_ptr, const std::vector<KernelTensor *> &workspace,            \
                                     const Args &... args) {                                                    \
    REGISTER_SYNC_OP(op_type_);                                                                                 \
    auto [executor, cache_func_ptr, release_func] = GetSyncExecutor(args...);                                   \
    if (workspace_size_list_.empty()) {                                                                         \
      RUN_OP_API_SYNC(op_type_, nullptr, 0, executor, stream_ptr);                                              \
    } else {                                                                                                    \
      if (workspace.empty()) {                                                                                  \
        MS_LOG(EXCEPTION) << "Failed to allocate workspace tensor!";                                            \
      }                                                                                                         \
      auto workspace_tensor = workspace[0];                                                                     \
      if (workspace_tensor->size() != workspace_size_list_[0]) {                                                \
        MS_LOG(EXCEPTION) << "Please check 'GetWorkSpaceInfo' and 'Launch' func. Expected workspace size is"    \
                          << workspace_size_list_[0] << ", but get " << workspace_tensor->size();               \
      }                                                                                                         \
      RUN_OP_API_SYNC(op_type_, workspace_tensor->device_ptr(), workspace_size_list_[0], executor, stream_ptr); \
    }                                                                                                           \
    const auto &all_acl_tensor = cache_func_ptr(device::ascend::ProcessCacheType::kGetOutputShape, {});         \
    if (release_func) {                                                                                         \
      release_func();                                                                                           \
    }                                                                                                           \
    return all_acl_tensor;                                                                                      \
  }

class AscendCKernelMod : public KernelMod {
 public:
  explicit AscendCKernelMod(std::string &&op_type) : op_type_(std::move(op_type)) {
    auto capaticy_from_user = GetCacheCapaticy();
    if (capaticy_from_user >= 0) {
      capacity_ = LongToSize(capaticy_from_user);
      MS_LOG(INFO) << "Set ascendc cache queue length of kbyk to " << capacity_;
    }
  }
  ~AscendCKernelMod();

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  virtual void GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  }
  virtual bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                      const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  void set_fullname(const std::string &fullname) override { fullname_ = fullname; }

  void ResetDeivceAddress(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {}

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override;
  bool IsNeedUpdateOutputShapeAndSize() override { return false; }
  std::vector<KernelAttr> GetOpSupport() override { MS_LOG(EXCEPTION) << "This interface is not support in ascendc."; }

  template <typename... Args>
  void UpdateWorkspace(const std::tuple<Args...> &args) {
    auto real_workspace_size = static_cast<size_t>(std::get<0>(args));
    if (real_workspace_size != 0) {
      std::vector<size_t> workspace_size_list = {real_workspace_size};
      SetWorkspaceSizeList(workspace_size_list);
    }

    constexpr size_t kBoostGeneratorSize = 5;
    if constexpr (std::tuple_size_v<std::tuple<Args...>> == kBoostGeneratorSize) {
      hash_id_ = std::get<kHashIdIndex>(args);
    }
  }

  void SetDynamic(bool is_dynamic) { is_dynamic_ = is_dynamic; }

  void ClearOpsWorkSpaceList() {
    ops_workspace_size_idx_ = 0;
    ops_workspace_size_map_.clear();
    workspace_size_list_.clear();
  }

 protected:
  template <typename T>
  T GetRequiredAttr(const std::string &attr_name) {
    auto attr_value = primitive_->GetAttr(attr_name);
    return GetValue<T>(attr_value);
  }

  aclOpExecutor *executor_{nullptr};
  CallBackFunc release_func_{nullptr};
  std::string op_type_;
  uint64_t hash_id_{0};
  std::unordered_map<std::string, std::pair<size_t, size_t>> ops_workspace_size_map_;
  size_t ops_workspace_size_idx_{0};
  static bool is_dynamic_;
  std::unordered_map<uint64_t, std::list<CacheTuple>::iterator> hash_map_;
  std::list<CacheTuple> hash_cache_;
  size_t capacity_{64};

  static constexpr size_t kHashIdIndex = 3;

private:
  std::string fullname_;
};

#define MS_ASCENDC_KERNEL_FACTORY_REG(NAME, DERIVE_CLASS) MS_CUSTOM_KERNEL_FACTORY_REG(#NAME, DERIVE_CLASS)

}  // namespace ms_custom_ops

#endif  // MS_CUSTOM_OPS_OP_DEF_ASCENDC_GRAPHMODE_ASCENDC_KERNEL_MOD_H_
