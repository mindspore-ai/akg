# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# 全局变量存储配置信息
_collected_config_timings = {}

# ============================================================================
# AKG_restore_copy Triton kernel
# 参考 l2_cache_clear.py 的设计：使用带 AKG_ 前缀的专用 kernel，
# 便于在 profiler 的 op_statistic.csv 中按名字精确过滤。
# ============================================================================

AKG_RESTORE_COPY_KERNEL_NAME = "AKG_restore_copy"

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass

if _TRITON_AVAILABLE:
    @triton.jit
    def AKG_restore_copy(
        dst_ptr, src_ptr, n_elements,
        BLOCK_SIZE: tl.constexpr, CORE_NUM: tl.constexpr,
    ):
        """
        restore_value 专用 copy kernel。

        kernel 名称带 AKG_ 前缀，在 profiler 中显示为 AKG_restore_copy，
        可精确过滤，不会误删用户代码中的 TensorMove 等同名操作。
        """
        pid = tl.program_id(0)
        num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
        for block_idx in range(pid, num_blocks, CORE_NUM):
            block_start = block_idx * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            data = tl.load(src_ptr + offsets, mask=mask)
            tl.store(dst_ptr + offsets, data, mask=mask)


def _get_vec_core_num():
    try:
        import torch_npu
        return torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
    except Exception:
        return 40


def akg_restore_copy(dst, src):
    """用 AKG_restore_copy kernel 执行 tensor copy，替代 tensor.copy_()。"""
    import torch
    n = dst.numel()
    dst_flat = dst.view(-1)
    src_flat = src.view(-1)
    core_num = _get_vec_core_num()
    BLOCK_SIZE = 1024
    grid = (core_num,)
    AKG_restore_copy[grid](dst_flat, src_flat, n,
                           BLOCK_SIZE=BLOCK_SIZE, CORE_NUM=core_num)
    torch.npu.synchronize()


# ============================================================================
# _bench patch: 禁用原生 restore_value 的 copy_()，
# 让 kernel_call 只包含纯 kernel，restore 交给 benchmarker 用命名 kernel 做。
# ============================================================================

_restore_info = None


def _patch_autotuner_bench(autotuner_module):
    """Patch Autotuner._bench，在 restore_value 场景下接管 pre_hook。"""
    original_bench = getattr(autotuner_module.Autotuner, '_bench', None)
    if original_bench is None:
        return
    if getattr(original_bench, '_akg_bench_patched', False):
        return

    _noop = lambda *a, **kw: None

    def patched_bench(self, *args, config, **meta):
        global _restore_info

        if not (_TRITON_AVAILABLE and hasattr(self, 'restore_value') and self.restore_value):
            _restore_info = None
            return original_bench(self, *args, config=config, **meta)

        saved = {}
        for name in self.restore_value:
            idx = self.fn.arg_names.index(name)
            saved[idx] = args[idx].clone()
        _restore_info = {'saved': saved, 'args': list(args)}

        orig_rv = self.restore_value
        orig_ph = getattr(self, 'pre_hook', None)
        orig_posth = getattr(self, 'post_hook', None)
        self.restore_value = None
        self.pre_hook = _noop
        self.post_hook = _noop

        try:
            result = original_bench(self, *args, config=config, **meta)
        finally:
            self.restore_value = orig_rv
            self.pre_hook = orig_ph
            self.post_hook = orig_posth
            _restore_info = None

        return result

    patched_bench._akg_bench_patched = True
    autotuner_module.Autotuner._bench = patched_bench


# ============================================================================
# 需要过滤的底层实现参数
# ============================================================================

_FILTERED_CONFIG_PARAMS = {
    'num_warps',
    'num_ctas',
    'num_stages',
    'num_buffers_warp_spec',
    'num_consumer_groups',
    'reg_dec_producer',
    'reg_inc_consumer',
    'maxnreg'
}


def _filter_config_string(config_str: str) -> str:
    """过滤配置字符串，移除底层实现参数"""
    params = []
    for param in config_str.split(','):
        param = param.strip()
        if not param:
            continue
        if ':' in param:
            param_name = param.split(':', 1)[0].strip()
        elif '=' in param:
            param_name = param.split('=', 1)[0].strip()
        else:
            params.append(param)
            continue
        if param_name not in _FILTERED_CONFIG_PARAMS:
            params.append(param)
    return ', '.join(params)


def patch_triton_autotuner():
    """动态补丁 triton autotuner，添加配置信息收集 + _bench restore_value 接管。"""
    try:
        import triton.runtime.autotuner as autotuner_module
    except ImportError:
        return True

    try:
        import triton.runtime.autotiling_tuner as autotiling_module
    except ImportError:
        autotiling_module = None

    if not hasattr(autotuner_module, 'Autotuner'):
        return True

    original_autotuner_run = getattr(autotuner_module.Autotuner, 'run', None)
    if original_autotuner_run is None:
        return True
    if getattr(original_autotuner_run, '_akg_run_patched', False):
        return True

    original_autotiling_run = None
    if autotiling_module and hasattr(autotiling_module, 'AutoTilingTuner'):
        original_autotiling_run = getattr(autotiling_module.AutoTilingTuner, 'run', None)

    # Patch _bench 接管 restore_value
    _patch_autotuner_bench(autotuner_module)

    def _process_config_timings(self):
        if not (hasattr(self, 'best_config') and
                hasattr(self, 'configs_timings') and
                self.configs_timings and
                isinstance(self.configs_timings, dict)):
            return

        func_name = "unknown_function"
        try:
            if hasattr(self, 'base_fn') and hasattr(self.base_fn, '__name__'):
                func_name = self.base_fn.__name__
            elif hasattr(self, 'fn') and hasattr(self.fn, '__name__'):
                func_name = self.fn.__name__
        except (AttributeError, TypeError):
            pass

        try:
            sorted_timings = sorted(self.configs_timings.items(), key=lambda x: x[1])
            config_data = []
            for i, (config, timing) in enumerate(sorted_timings):
                try:
                    is_best = config == self.best_config
                    timing_value = timing[0] if isinstance(timing, list) else timing
                    timing_us = timing_value
                    config_str = _filter_config_string(str(config))
                    config_data.append({
                        "config": config_str,
                        "timing_us": float(timing_us),
                        "is_best": is_best,
                        "rank": i + 1
                    })
                except (TypeError, ValueError, AttributeError):
                    continue

            if config_data:
                global _collected_config_timings
                if func_name not in _collected_config_timings:
                    _collected_config_timings[func_name] = config_data

                    if os.getenv("TRITON_PRINT_AUTOTUNING", None) == "1":
                        print(f"All config timings for {func_name}:")
                        for i, (config, timing) in enumerate(sorted_timings):
                            try:
                                status = " (BEST)" if config == self.best_config else ""
                                timing_value = timing[0] if isinstance(timing, list) else timing
                                timing_us = timing_value
                                config_str = _filter_config_string(str(config))
                                print(f"  Config {i+1}: {config_str} -> {timing_us:.4f}us{status}")
                            except (TypeError, ValueError, AttributeError):
                                continue

        except (TypeError, ValueError, AttributeError):
            pass

    def patched_autotuner_run(self, *args, **kwargs):
        result = original_autotuner_run(self, *args, **kwargs)
        try:
            _process_config_timings(self)
        except Exception:
            pass
        return result

    def patched_autotiling_run(self, *args, **kwargs):
        result = original_autotiling_run(self, *args, **kwargs)
        try:
            _process_config_timings(self)
        except Exception:
            pass
        return result

    try:
        patched_autotuner_run._akg_run_patched = True
        autotuner_module.Autotuner.run = patched_autotuner_run
    except (AttributeError, TypeError):
        pass

    if original_autotiling_run is not None:
        try:
            patched_autotiling_run._akg_run_patched = True
            autotiling_module.AutoTilingTuner.run = patched_autotiling_run
        except (AttributeError, TypeError):
            pass

    return True


def get_collected_config_timings():
    global _collected_config_timings
    return _collected_config_timings.copy()


def clear_collected_config_timings():
    global _collected_config_timings
    _collected_config_timings = {}


def patch_driver_benchmarker():
    """补丁 driver.active.get_benchmarker()，让 autotune 使用 profiler_npu。

    当 _restore_info 不为空时（即 _bench 禁用了原生 restore_value），
    benchmarker 自动用 AKG_restore_copy kernel 包装 kernel_call，
    profiler 按 kernel 名字精确过滤，不会误删用户的 TensorMove 操作。
    """
    try:
        from triton.runtime import driver

        if hasattr(driver.active.get_benchmarker, '_akg_agents_patched'):
            return True

        original_get_benchmarker = driver.active.get_benchmarker

        def patched_get_benchmarker():
            def custom_benchmarker(kernel_call, quantiles=(0.5, 0.2, 0.8)):
                try:
                    from akg_agents.op.verifier.profiler import profiler_npu

                    if _restore_info is not None:
                        saved = _restore_info['saved']
                        args = _restore_info['args']

                        def wrapped_call():
                            for idx, saved_val in saved.items():
                                akg_restore_copy(args[idx], saved_val)
                            kernel_call()

                        fn_to_profile = wrapped_call
                    else:
                        fn_to_profile = kernel_call

                    time_us = profiler_npu(
                        fn_to_profile,
                        warmup=5,
                        active=30,
                        suppress_warnings=True,
                        clear_l2_cache=True,
                        dsl="triton_ascend",
                        filter_restore_copy=(_restore_info is not None),
                    )
                    return [time_us] * 3

                except ImportError:
                    original_benchmarker = original_get_benchmarker()
                    return original_benchmarker(kernel_call, quantiles)

            return custom_benchmarker

        driver.active.get_benchmarker = patched_get_benchmarker
        driver.active.get_benchmarker._akg_agents_patched = True
        return True

    except ImportError:
        return False
    except Exception as e:
        print(f"Warning: Failed to patch driver benchmarker: {e}")
        return False


def apply_triton_patches():
    """应用所有triton补丁"""
    success1 = patch_triton_autotuner()
    success2 = patch_driver_benchmarker()
    return success1 or success2


if __name__ != "__main__":
    apply_triton_patches()

if __name__ == "__main__":
    print("Testing Triton patches...")
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    success1 = patch_triton_autotuner()
    success2 = patch_driver_benchmarker()

    if success1:
        print("Autotuner patch applied successfully!")
    if success2:
        print("Driver benchmarker patch applied successfully!")

    if not any([success1, success2]):
        print("Failed to apply patches")
